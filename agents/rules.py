"""Детерминированные проверки поверх базы знаний.

Используются: (1) как самостоятельный офлайн-режим, (2) как подсказки
(«pre-check») для LLM-агентов, (3) Verifier-ом для проверки заявлений LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from knowledge.antipatterns import ANTIPATTERNS
from knowledge.fintech_controls import CONTROLS, Control
from knowledge.language import METRIC_PATTERNS, METRIC_QUALIFIERS, VAGUE_TERMS
from knowledge.tz_structure import SECTIONS, Section
from models import Category, ControlCheck, Finding, Severity
from parsing import Document

_DIGIT = re.compile(r"\d")
_HEADING = re.compile(r"^\s*(#{1,6}\s+|\d+(\.\d+)*\.?\s+[А-ЯA-ZЁ])")
_LEAD_ID = re.compile(r"^\s*(?:[-*•|]\s*)?(?:#+\s*)?(?:(?:[A-ZА-Я]{1,5}[-.]?)?\d+(?:\.\d+)*\.?\)?\s+)?", re.I)


def strip_ids(line: str) -> str:
    """Убирает нумерацию и ID требования в начале строки (FR-01., 3.2.1, - ), чтобы
    цифры из ID не принимались за метрику."""
    return _LEAD_ID.sub("", line, count=1)

DOWNGRADE = {"critical": "major", "major": "minor", "minor": "minor"}


def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE)


def _clip(s: str, n: int = 200) -> str:
    s = s.strip().lstrip("#-*• ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


# ── Разделы ТЗ ────────────────────────────────────────────────────────────────

@dataclass
class SectionCheck:
    section: Section
    lines: list[int]
    numeric_lines: list[int]

    @property
    def present(self) -> bool:
        return bool(self.numeric_lines if self.section.needs_number else self.lines)


def scan_sections(doc: Document) -> list[SectionCheck]:
    out = []
    for s in SECTIONS:
        lines: set[int] = set()
        for p in s.patterns:
            lines.update(doc.search(p))
        lines_sorted = sorted(lines)
        numeric = [n for n in lines_sorted if _DIGIT.search(strip_ids(doc.line(n)))]
        out.append(SectionCheck(s, lines_sorted, numeric))
    return out


def section_findings(doc: Document, checks: list[SectionCheck]) -> list[Finding]:
    out = []
    for c in checks:
        s = c.section
        if c.present:
            continue
        if s.needs_number and c.lines:
            ev = [_clip(doc.line(n)) for n in c.lines[:2]]
            out.append(Finding(
                agent="completeness", category=Category(s.category), severity=Severity(DOWNGRADE[s.severity]),
                title=f"{s.title}: упомянуто, но без измеримых значений",
                description=f"Тема затронута (L{', L'.join(map(str, c.lines[:3]))}), но не задано ни одного числового значения — требование нельзя проверить. {s.why}",
                evidence=ev, lines=c.lines[:3], recommendation=s.recommendation, question=s.question,
                control_id=s.id, confidence=0.75,
            ))
        else:
            out.append(Finding(
                agent="completeness", category=Category(s.category), severity=Severity(s.severity),
                title=f"Отсутствует раздел: {s.title}",
                description=s.why, recommendation=s.recommendation, question=s.question,
                control_id=s.id, confidence=0.7,
            ))
    return out


# ── Финтех/ИБ-контроли ────────────────────────────────────────────────────────

def check_control(doc: Document, c: Control) -> ControlCheck:
    hit_patterns, lines = 0, set()
    for p in c.patterns:
        found = doc.search(p)
        if found:
            hit_patterns += 1
            lines.update(found)
    if hit_patterns >= c.min_hits_covered:
        status = "covered"
    elif hit_patterns >= 1:
        status = "partial"
    else:
        status = "missing"
    ln = sorted(lines)[:6]
    return ControlCheck(
        control_id=c.id, title=c.title, area=c.area, regulation=c.regulation, status=status,
        evidence=[_clip(doc.line(n), 160) for n in ln[:3]], lines=ln,
        comment=f"признаков найдено: {hit_patterns} из {len(c.patterns)}",
    )


def scan_controls(doc: Document, domains: list[str]) -> list[ControlCheck]:
    ds = set(domains)
    out = []
    for c in CONTROLS:
        if "*" in c.applies_to or ds.intersection(c.applies_to):
            out.append(check_control(doc, c))
        else:
            out.append(ControlCheck(control_id=c.id, title=c.title, area=c.area, regulation=c.regulation,
                                    status="not_applicable", comment="домен не затронут в ТЗ"))
    return out


def control_finding(c: Control, check: ControlCheck, agent: str = "fintech_security") -> Finding | None:
    if check.status in ("covered", "not_applicable"):
        return None
    partial = check.status == "partial"
    sev = DOWNGRADE[c.severity] if partial else c.severity
    if partial:
        title = f"Не полностью описано: {c.title}"
        desc = f"Тема упомянута, но без ключевых деталей. {c.why}"
        if check.comment:
            desc += f" ({check.comment})"
    else:
        title = f"Не описано: {c.title}"
        desc = c.why
    return Finding(
        agent=agent, category=Category(c.category), severity=Severity(sev), title=title, description=desc,
        evidence=check.evidence[:2] if partial else [], lines=check.lines[:3] if partial else [],
        regulation=c.regulation, recommendation=c.recommendation, suggested_text=c.suggested_text,
        question=c.question, control_id=c.id, confidence=0.7 if partial else 0.75,
    )


# ── Опасные требования ────────────────────────────────────────────────────────

def scan_antipatterns(doc: Document, agent: str = "fintech_security") -> list[Finding]:
    out = []
    for ap in ANTIPATTERNS:
        rx, unless = _rx(ap.pattern), (_rx(ap.unless) if ap.unless else None)
        for i, line in enumerate(doc.lines, start=1):
            l = line.replace("ё", "е").replace("Ё", "Е")
            if rx.search(l) and not (unless and unless.search(l)):
                out.append(Finding(
                    agent=agent, category=Category(ap.category), severity=Severity(ap.severity),
                    title=f"Опасное требование: {ap.title}", description=ap.description,
                    evidence=[_clip(line, 240)], lines=[i], regulation=ap.regulation,
                    recommendation=ap.recommendation, suggested_text=ap.suggested_text,
                    question=ap.question, control_id=ap.id, confidence=0.85,
                ))
    return out


# ── Язык требований ───────────────────────────────────────────────────────────

_NFR_CONTEXT = _rx(r"производительн|скорост|быстр|мгновенн|отклик|надежн|доступн|безопасн|нагрузк|пользовател|время|защищ")


def scan_vague(doc: Document) -> list[Finding]:
    out = []
    compiled = [(_rx(v.pattern), v) for v in VAGUE_TERMS]
    for i, line in enumerate(doc.lines, start=1):
        if not line.strip() or _HEADING.match(line) and len(line) < 60:
            continue
        l = line.replace("ё", "е").replace("Ё", "Е")
        body = strip_ids(l)
        hits = []
        for rx, v in compiled:
            m = rx.search(l)
            if not m:
                continue
            if v.kind == "metric" and _DIGIT.search(body):
                continue  # рядом есть число — вероятно, метрика задана
            hits.append((m.group(0), v))
        if not hits:
            continue
        kinds = {v.kind for _, v in hits}
        if "tbd" in kinds:
            term, v = next((h for h in hits if h[1].kind == "tbd"))
            out.append(Finding(
                agent="consistency", category=Category.completeness, severity=Severity.major,
                title="Открытый вопрос в тексте ТЗ (TBD / «уточнить позже»)",
                description=f"Строка L{i} содержит незакрытый вопрос («{term}»). Разработка по такому требованию начнётся с допущений.",
                evidence=[_clip(line)], lines=[i], recommendation=v.hint,
                question=f"Требование в L{i} помечено как неопределённое («{term}»): какое решение принято?",
                confidence=0.9,
            ))
            hits = [h for h in hits if h[1].kind != "tbd"]
            if not hits:
                continue
        terms = ", ".join(f"«{t}»" for t, _ in hits)
        metric = any(v.kind == "metric" for _, v in hits)
        sev = Severity.major if metric and _NFR_CONTEXT.search(l) else Severity.minor
        cat = Category.testability if metric else Category.ambiguity
        hint = "; ".join(dict.fromkeys(v.hint for _, v in hits))
        out.append(Finding(
            agent="consistency", category=cat, severity=sev,
            title=f"Размытая формулировка: {terms}",
            description=f"Формулировка {terms} в L{i} допускает разное толкование и не может быть проверена при приёмке.",
            evidence=[_clip(line)], lines=[i], recommendation=f"Сделать требование измеримым: {hint}.",
            question=f"Что конкретно означает {terms} в требовании L{i}? Предлагаем: {hint}.",
            confidence=0.75 if metric else 0.65,
        ))
    return out


def _num(v: str) -> float:
    return float(re.sub(r"\s", "", v).replace(",", "."))


def _unit_scale(u: str | None) -> float:
    u = (u or "").lower()
    if u.startswith(("мс", "ms")):
        return 0.001
    if u.startswith("мин"):
        return 60.0
    return 1.0


def scan_metric_conflicts(doc: Document) -> list[Finding]:
    """Ищет разные значения одной и той же метрики в разных местах ТЗ."""
    out = []
    for metric, pattern in METRIC_PATTERNS.items():
        rx = _rx(pattern)
        values: dict[str, dict[float, list[int]]] = {}
        for i, line in enumerate(doc.lines, start=1):
            l = strip_ids(line).replace("ё", "е").lower()
            for m in rx.finditer(l):
                gd = m.groupdict()
                raw = gd.get("v") or gd.get("v2")
                if not raw:
                    continue
                try:
                    val = _num(raw) * _unit_scale(gd.get("u"))
                except ValueError:
                    continue
                q = ""
                if metric in METRIC_QUALIFIERS:
                    quals, default = METRIC_QUALIFIERS[metric]
                    # ищем уточнитель только в клаузе (между , и ;), где стоит значение
                    vs, ve = m.span("v" if gd.get("v") else "v2")
                    left = max(l.rfind(",", 0, vs), l.rfind(";", 0, vs)) + 1
                    rights = [p for p in (l.find(",", ve), l.find(";", ve)) if p != -1]
                    clause = l[left: min(rights) if rights else len(l)]
                    q = next((name for name, p in quals.items() if re.search(p, clause)), default)
                    if q == "skip":
                        continue
                values.setdefault(q, {}).setdefault(val, []).append(i)
        for q, vals in values.items():
            if len(vals) < 2:
                continue
            lines = sorted({n for ns in vals.values() for n in ns})
            label = metric + (f" ({q})" if q else "")
            shown = ", ".join(f"{('%g' % v)} (L{', L'.join(map(str, ns[:2]))})" for v, ns in vals.items())
            out.append(Finding(
                agent="consistency", category=Category.contradiction,
                severity=Severity.major if q != "без уточнения периода" else Severity.minor,
                title=f"Противоречивые значения: {label}",
                description=f"Для метрики «{label}» в документе указаны разные значения: {shown}.",
                evidence=[_clip(doc.line(n)) for n in lines[:3]], lines=lines[:6],
                recommendation="Оставить одно значение в одном месте ТЗ (single source of truth) и сослаться на него из остальных разделов.",
                question=f"Какое значение метрики «{label}» верное: {shown}?",
                confidence=0.7 if q != "без уточнения периода" else 0.5,
            ))
    return out


def scan_duplicates(doc: Document) -> list[Finding]:
    reqs = [(i, l.strip()) for i, l in enumerate(doc.lines, start=1) if len(l.strip()) > 50]
    out, seen = [], set()
    for a in range(len(reqs)):
        for b in range(a + 1, min(len(reqs), a + 400)):
            i, la = reqs[a]
            j, lb = reqs[b]
            if (i, j) in seen:
                continue
            r = SequenceMatcher(None, la.lower(), lb.lower()).quick_ratio()
            if r < 0.9:
                continue
            r = SequenceMatcher(None, la.lower(), lb.lower()).ratio()
            if r >= 0.9:
                seen.add((i, j))
                out.append(Finding(
                    agent="consistency", category=Category.contradiction, severity=Severity.minor,
                    title="Дублирующееся требование",
                    description=f"Строки L{i} и L{j} практически совпадают ({r:.0%}). Дубли расходятся при правках и порождают противоречия.",
                    evidence=[_clip(la), _clip(lb)], lines=[i, j],
                    recommendation="Оставить требование в одном месте и сослаться на него по ID.",
                    confidence=0.8,
                ))
    return out[:10]
