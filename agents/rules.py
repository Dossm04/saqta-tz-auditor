"""Детерминированные проверки поверх базы знаний.

Используются: (1) как самостоятельный офлайн-режим, (2) как подсказки
(«pre-check») для LLM-агентов, (3) Verifier-ом для проверки заявлений LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from knowledge.antipatterns import ANTIPATTERNS
from knowledge.domains import OUT_OF_SCOPE
from knowledge.fintech_controls import CONTROLS, Control
from knowledge.language import CONFLICT_PAIRS, METRIC_PATTERNS, METRIC_QUALIFIERS, SEGMENTS, VAGUE_TERMS
from knowledge.tz_structure import SECTIONS, Section
from models import Category, ControlCheck, Finding, Severity
from parsing import Document

_DIGIT = re.compile(r"\d")
_MD_HEADING = re.compile(r"^\s*#{1,6}\s+")
_NUM_TITLE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[А-ЯA-ZЁ]")
# глагол/модальность → это требование, а не заголовок раздела
_REQ_VERB = re.compile(r"\b(долж|необходим|обязан|требуется|может|могут|нужно)\w*|\w{3,}(ет|ит|ут|ют|ят|ат)(ся)?\b",
                       re.IGNORECASE)
# Нумерация/ID в начале строки. Голое число снимается только с точкой/скобкой («1.», «2)»)
# или многоуровневое перед заглавной («3.2.1 Текст»): «5 минут неактивности» — это значение, а не номер.
_LEAD_ID = re.compile(
    r"^\s*(?:[-*•|]\s*)?(?:#+\s*)?"
    r"(?:(?:[A-ZА-ЯЁ]{1,6}[-.]?\d+(?:\.\d+)*\.?\)?|\d+(?:\.\d+)*[.)]|\d+(?:\.\d+)+(?=\s+[А-ЯЁA-Z]))\s+)?"
)


def strip_ids(line: str) -> str:
    """Убирает нумерацию и ID требования в начале строки (FR-01., 3.2.1, 1), - ), чтобы
    цифры из ID не принимались за метрику. Ведущие количества («5 минут») сохраняются."""
    return _LEAD_ID.sub("", line, count=1)


def _is_heading(line: str) -> bool:
    """Заголовок раздела: markdown-заголовок или короткая нумерованная строка без глагола и точки в конце."""
    if _MD_HEADING.match(line):
        return True
    s = line.strip()
    if not _NUM_TITLE.match(s) or len(s) >= 60 or s.endswith((".", ";", "!", "?", ":")):
        return False
    body = strip_ids(s)
    return len(body.split()) <= 7 and not _REQ_VERB.search(body)


DOWNGRADE = {"critical": "major", "major": "minor", "minor": "minor"}


def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE)


def _norm(line: str) -> str:
    return line.replace("ё", "е").replace("Ё", "Е")


def _out_of_scope(line: str) -> bool:
    return bool(OUT_OF_SCOPE.search(_norm(line)))


def _clip(s: str, n: int = 200) -> str:
    s = s.strip().lstrip("#-*• ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


# ── Клаузы и отрицание ────────────────────────────────────────────────────────

_CLAUSE_STOP = re.compile(r"[;,():—]|\.\s")


def _clause_span(l: str, start: int, end: int) -> tuple[int, int]:
    """Границы клаузы (между , ; : ( ) — и концом предложения), в которой стоит фрагмент [start, end)."""
    left = 0
    for m in _CLAUSE_STOP.finditer(l, 0, start):
        left = m.end()
    m = _CLAUSE_STOP.search(l, end)
    return left, (m.start() if m else len(l))


_NEG_BEFORE = re.compile(
    r"(?:\b(?:без|нет|ни)\b|\bне\b(?!\s+(?:менее|более|позднее|позже|ранее|реже|чаще|выше|ниже|больше|меньше|дольше|только|превыша\w*))"
    r"|\bотключ\w*|\bотказ\w* от)(?:\s+[\w/-]+){0,3}\s*$",
    re.IGNORECASE,
)
_NEG_AFTER = re.compile(
    r"^[\w/-]*(?:\s+(?!не\b)[\w/-]+){0,3}?\s+(?:не\s+(?:требу|применя|использ|провод|проход|запрашива|предусм|поддержива|проверя|выполня"
    r"|осуществля|реализу|включа|нужн)\w*|отключа\w*|отсутству\w*)",
    re.IGNORECASE,
)


_INTERROGATIVE = re.compile(r"\b(?:не\s+[\w-]+|нет)\s+ли\b", re.IGNORECASE)


def is_negated(line: str, start: int, end: int) -> bool:
    """Фрагмент [start, end) стоит под отрицанием в своей клаузе: «без 3-D Secure»,
    «2FA не требуется», «3-D Secure отключается». Количественные «не менее/не более» — не отрицание,
    как и проверка условия «проверяется, не установил ли клиент самозапрет» / «нет ли просрочки»."""
    l = _norm(line)
    left, right = _clause_span(l, start, end)
    before = l[left:start]
    if _INTERROGATIVE.search(before):
        return False
    return bool(_NEG_BEFORE.search(before) or _NEG_AFTER.match(l[end:right]))


def _positive_hit(rx: re.Pattern, line: str) -> bool:
    """В строке есть совпадение, которое не стоит под отрицанием."""
    l = _norm(line)
    return any(not is_negated(l, m.start(), m.end()) for m in rx.finditer(l) if m.end() > m.start())


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
    """Покрытие контроля по признакам. Не засчитываются: строки вне скоупа («не входит в MVP»)
    и совпадения под отрицанием («без 3-D Secure», «2FA не требуется»)."""
    hit_patterns, lines = 0, set()
    scoped = [(i, line) for i, line in enumerate(doc.lines, start=1) if line.strip() and not _out_of_scope(line)]
    for p in c.patterns:
        rx = _rx(p)
        found = [i for i, line in scoped if _positive_hit(rx, line)]
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
    scoped = [(i, line, _norm(line)) for i, line in enumerate(doc.lines, start=1)
              if line.strip() and not _out_of_scope(line)]
    out = []
    for ap in ANTIPATTERNS:
        rx, unless = _rx(ap.pattern), (_rx(ap.unless) if ap.unless else None)
        absent = getattr(ap, "requires_absent", None)
        # условие на уровне документа: например, «короткий ПИН» опасен, только если нигде нет политики попыток
        if absent and any(_rx(absent).search(l) for _, _, l in scoped):
            continue
        for i, line, l in scoped:
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
        if not line.strip() or _is_heading(line):
            continue
        l = _norm(line)
        if OUT_OF_SCOPE.search(l):
            continue  # «Мгновенные переводы не входят в рамки проекта» — не требование
        body = strip_ids(l)
        hits = []
        for rx, v in compiled:
            m = rx.search(body)
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
    if u.startswith("час"):
        return 3600.0
    return 1.0


def _segment(text: str) -> str:
    """Сегмент клиентов/операций (уровень идентификации, межбанк, новое устройство…)."""
    return next((name for name, p in SEGMENTS.items() if re.search(p, text)), "")


def _segments(text: str) -> set[str]:
    return {name for name, p in SEGMENTS.items() if re.search(p, text)}


def _fmt(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") if v >= 10000 and v == int(v) else "%g" % v


def scan_metric_conflicts(doc: Document, semantic: bool = True) -> list[Finding]:
    """Ищет разные значения одной и той же метрики в разных местах ТЗ.

    Значения сравниваются только внутри одного уточнителя (период лимита, рабочие/календарные дни)
    и одного сегмента (упрощённая/полная идентификация и т.п.). При semantic=True к результату
    добавляются смысловые противоречия (scan_semantic_conflicts) — так их получает и вызывающий код,
    который знает только про scan_metric_conflicts."""
    out = []
    lines = [(i, strip_ids(line).replace("ё", "е").lower()) for i, line in enumerate(doc.lines, start=1)
             if line.strip() and not _out_of_scope(line)]
    for metric, pattern in METRIC_PATTERNS.items():
        rx = _rx(pattern)
        values: dict[tuple[str, str], dict[float, list[int]]] = {}
        for i, l in lines:
            for m in rx.finditer(l):
                gd = m.groupdict()
                vkey = "v" if gd.get("v") else "v2"
                raw = gd.get(vkey)
                if not raw:
                    continue
                try:
                    val = _num(raw) * _unit_scale(gd.get("u") or gd.get("u2"))
                except ValueError:
                    continue
                # уточнитель ищем только в клаузе (между , и ;), где стоит значение
                vs, ve = m.span(vkey)
                left = max(l.rfind(",", 0, vs), l.rfind(";", 0, vs)) + 1
                rights = [p for p in (l.find(",", ve), l.find(";", ve)) if p != -1]
                clause = l[left: min(rights) if rights else len(l)]
                q = ""
                if metric in METRIC_QUALIFIERS:
                    quals, default = METRIC_QUALIFIERS[metric]
                    q = next((name for name, p in quals.items() if re.search(p, clause)), default) or ""
                    if q == "skip":
                        continue
                seg = _segment(clause) or _segment(l)
                bucket = values.setdefault((q, seg), {}).setdefault(val, [])
                if i not in bucket:
                    bucket.append(i)
        for (q, seg), vals in values.items():
            if len(vals) < 2:
                continue
            lines_ = sorted({n for ns in vals.values() for n in ns})
            if len(lines_) < 2:
                continue  # разные значения в одной строке — это обычно диапазон, а не противоречие
            quals = "; ".join(x for x in (q, seg) if x)
            label = metric + (f" ({quals})" if quals else "")
            shown = ", ".join(f"{_fmt(v)} (L{', L'.join(map(str, ns[:2]))})" for v, ns in vals.items())
            weak = q == "без уточнения периода"
            out.append(Finding(
                agent="consistency", category=Category.contradiction,
                severity=Severity.minor if weak else Severity.major,
                title=f"Противоречивые значения: {label}",
                description=f"Для метрики «{label}» в документе указаны разные значения: {shown}.",
                evidence=[_clip(doc.line(n)) for n in lines_[:3]], lines=lines_[:6],
                recommendation="Оставить одно значение в одном месте ТЗ (single source of truth) и сослаться на него из остальных разделов.",
                question=f"Какое значение метрики «{label}» верное: {shown}?",
                confidence=0.5 if weak else 0.7,
            ))
    if semantic:
        out += scan_semantic_conflicts(doc)
    return out


def scan_semantic_conflicts(doc: Document) -> list[Finding]:
    """Смысловые противоречия без чисел: «без комиссии» vs «комиссия 1%» для одной темы,
    исключение «без подтверждения» vs правило «каждая операция подтверждается», нетто vs брутто,
    «только полностью идентифицированным» vs разрешение при упрощённой идентификации, ПДн в РК vs за рубежом."""
    scoped = [(i, line, _norm(line).lower()) for i, line in enumerate(doc.lines, start=1)
              if line.strip() and not _out_of_scope(line) and not _is_heading(line)]
    out: list[Finding] = []
    for cp in CONFLICT_PAIRS:
        ra, rb = _rx(cp.side_a), _rx(cp.side_b)
        ua = _rx(cp.unless_a) if cp.unless_a else None
        ub = _rx(cp.unless_b) if cp.unless_b else None
        side_a = [(i, line, l) for i, line, l in scoped if ra.search(l) and not (ua and ua.search(l))]
        side_b = [(i, line, l) for i, line, l in scoped if rb.search(l) and not (ub and ub.search(l))]
        if not side_a or not side_b:
            continue
        topics = cp.topics or (("", ""),)
        for topic, tp in topics:
            ta = [x for x in side_a if not tp or re.search(tp, x[2])]
            tb = [x for x in side_b if not tp or re.search(tp, x[2])]
            pair = None
            for a in ta:
                for b in tb:
                    if a[0] == b[0]:
                        continue
                    if cp.check_segments:
                        sa, sb = _segments(a[2]), _segments(b[2])
                        if sa and sb and not (sa & sb):
                            continue  # про разные сегменты клиентов — не противоречие
                    pair = (a, b)
                    break
                if pair:
                    break
            if not pair:
                continue
            (la, line_a, _), (lb, line_b, _) = pair
            fmt = {"a": la, "b": lb, "topic": topic or "операции"}
            title = f"Смысловое противоречие: {cp.title}" + (f" — {topic}" if topic else "")
            out.append(Finding(
                agent="consistency", category=Category.contradiction, severity=Severity(cp.severity),
                title=title,
                description=f"L{la}: {cp.label_a}; L{lb}: {cp.label_b}. {cp.why}",
                evidence=[_clip(line_a), _clip(line_b)], lines=sorted({la, lb}),
                recommendation=cp.recommendation,
                question=cp.question.format(**fmt),
                control_id=cp.id, confidence=0.7,
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
