"""Детерминированные проверки поверх базы знаний.

Используются: (1) как самостоятельный офлайн-режим, (2) как подсказки
(«pre-check») для LLM-агентов, (3) Verifier-ом для проверки заявлений LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from knowledge.antipatterns import ANTIPATTERNS
from knowledge.domains import OUT_OF_SCOPE, is_fintech
from knowledge.fintech_controls import CONTROLS, Control
from knowledge.language import (BOUND_MAX, BOUND_MAX_WORDS, BOUND_MIN, BOUND_MIN_WORDS, CHANNEL_QUALIFIERS,
                                CLIENT_QUALIFIERS, CONFLICT_PAIRS, METRIC_PATTERNS, METRIC_QUALIFIERS,
                                NON_REQ_SECTIONS, POLARITY_CONFLICT, POLARITY_FAMILIES, SEGMENTS, STOPWORDS,
                                TARGET_VERBS, VAGUE_TERMS, VALUE_CONFLICT, VALUE_PERIODS, WEAK_WORDS)
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


# ниже обычных 0.7, но выше порога Verifier (0.4): находка остаётся видна вместе с пояснением про язык
FOREIGN_SECTION_CONFIDENCE = 0.45
_FOREIGN_NOTE = {
    "en": "ТЗ на английском языке: офлайн-проверка ищет русские заголовки разделов, раздел может присутствовать "
          "под английским названием — проверьте вручную или используйте LLM-режим.",
    "kk": "ТЗ на казахском языке: офлайн-проверка ищет русские заголовки разделов, раздел может присутствовать "
          "под казахским названием — проверьте вручную или используйте LLM-режим.",
}


def section_findings(doc: Document, checks: list[SectionCheck]) -> list[Finding]:
    """«Отсутствует раздел» по офлайн-чеклисту. Для нерусского ТЗ такие находки не критичны и с низкой
    уверенностью: шаблоны заголовков русские, и «отсутствие» может быть просто непереведённым заголовком."""
    lang = getattr(doc, "language", "ru")
    foreign = lang in _FOREIGN_NOTE
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
        elif foreign:
            # заголовки разделов ищутся по-русски: в английском/казахском ТЗ раздел может быть, но не найтись
            out.append(Finding(
                agent="completeness", category=Category(s.category), severity=Severity.minor,
                title=f"Отсутствует раздел: {s.title}",
                description=f"{_FOREIGN_NOTE[lang]} {s.why}", recommendation=s.recommendation,
                question=s.question, control_id=s.id, confidence=FOREIGN_SECTION_CONFIDENCE,
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


# Контроли с applies_to="*" рассчитаны на «любой финтех-проект». Если ТЗ вообще не похоже на финтех
# (ни одного домена и ни одного финтех-признака), применяются только общие ИБ/ПДн-контроли из этого списка,
# а банковские и регуляторные (MFA клиента, certificate pinning, уведомление АРРФР, аутсорсинг/облако,
# язык клиентских уведомлений) помечаются «не применимо».
GENERIC_CONTROL_IDS = frozenset({
    "SEC-AUTH-03",    # тайм-аут сессии и блокировка по попыткам
    "SEC-AUTH-04",    # хранение и политика паролей
    "SEC-ACC-01",     # разграничение доступа, аудит администраторов
    "SEC-DATA-02",    # шифрование данных при хранении
    "PRIV-01",        # согласие на сбор и обработку ПДн (закон о ПДн РК — для любого оператора)
    "PRIV-02",        # локализация ПДн в РК (там же)
    "PRIV-03",        # сроки хранения и удаление ПДн
    "OPS-AUDIT-01",   # журналирование
    "APPSEC-01",      # безопасная разработка
})
NON_FINTECH_COMMENT = "ТЗ не похоже на финтех-проект — банковский/регуляторный контроль не применяется"


def is_generic_control(c: Control) -> bool:
    """Общий ИБ/ПДн-контроль, применимый и к нефинтех-системам."""
    return "*" in c.applies_to and c.id in GENERIC_CONTROL_IDS


def control_applicability(c: Control, domains: list[str] | set[str], fintech: bool = True) -> str | None:
    """None — контроль применим; иначе причина статуса «не применимо»."""
    if set(domains).intersection(c.applies_to):
        return None
    if "*" in c.applies_to:
        return None if fintech or is_generic_control(c) else NON_FINTECH_COMMENT
    return "домен не затронут в ТЗ"


def scan_controls(doc: Document, domains: list[str]) -> list[ControlCheck]:
    # пустой список доменов ещё не «не финтех»: короткое банковское ТЗ может не набрать двух совпадений
    fintech = bool(domains) or is_fintech(doc.text or "\n".join(doc.lines))
    out = []
    for c in CONTROLS:
        reason = control_applicability(c, domains, fintech)
        if reason is None:
            out.append(check_control(doc, c))
        else:
            out.append(ControlCheck(control_id=c.id, title=c.title, area=c.area, regulation=c.regulation,
                                    status="not_applicable", comment=reason))
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
        affirmed = _rx(ap.unless_affirmed) if getattr(ap, "unless_affirmed", None) else None
        by_sentence = getattr(ap, "scope", "line") == "sentence"
        for i, line, l in scoped:
            units = _sentences(l) if by_sentence else [l]
            if any(rx.search(u) and not (unless and unless.search(u)) and not (affirmed and _positive_hit(affirmed, u))
                   for u in units):
                out.append(Finding(
                    agent=agent, category=Category(ap.category), severity=Severity(ap.severity),
                    title=f"Опасное требование: {ap.title}", description=ap.description,
                    evidence=[_clip(line, 240)], lines=[i], regulation=ap.regulation,
                    recommendation=ap.recommendation, suggested_text=ap.suggested_text,
                    question=ap.question, control_id=ap.id, confidence=0.85,
                ))
    return out


# ── Язык требований ───────────────────────────────────────────────────────────

_UI_QUOTE = re.compile(r"«[^»\n]*»|“[^”\n]*”|\"[^\"\n]*\"")
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
        # текст в кавычках — сообщение интерфейса или название («Операция не может быть выполнена»), не требование
        body = _UI_QUOTE.sub(lambda m: " " * len(m.group(0)), strip_ids(l))
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


def scan_metric_conflicts(doc: Document, semantic: bool = True, generic: bool = True) -> list[Finding]:
    """Ищет разные значения одной и той же метрики в разных местах ТЗ.

    Значения сравниваются только внутри одного уточнителя (период лимита, рабочие/календарные дни)
    и одного сегмента (упрощённая/полная идентификация и т.п.). При generic=True добавляются обобщённые
    численные противоречия (scan_value_conflicts) по любой теме, при semantic=True — смысловые
    (scan_semantic_conflicts): так их получает и вызывающий код, который знает только про scan_metric_conflicts."""
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
            if len(lines_) == 2 and _parallel_lines(doc, *lines_):
                continue  # «время ответа API скоринга» / «время ответа API отчётов» — разные параметры
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
    if generic:
        # пары строк, уже покрытые специфичной метрикой, обобщённый детектор не дублирует
        covered = {(x, y) for f in out for x in f.lines for y in f.lines if x < y}
        out += scan_value_conflicts(doc, exclude=covered)
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
    # обобщённая полярность (CP-POLARITY) не дублирует пары строк, уже найденные специфичными парами
    covered = {(min(f.lines), max(f.lines)) for f in out if len(f.lines) >= 2}
    out += [f for f in scan_polarity_conflicts(doc) if (min(f.lines), max(f.lines)) not in covered]
    return out


# ── Обобщённые противоречия: тема утверждения ────────────────────────────────
# Тема сравнивается по основам содержательных слов (лёгкий стеммер: снятие окончания + 5 символов),
# поэтому «срок хранения резерва» и «резерв хранится» совпадают, а слова конкретной метрики не нужны.

_WORD_RX = re.compile(r"[^\W\d_]+")
_ENDINGS = tuple(sorted({
    "ировать", "ирования", "ированн", "ирует", "ируют", "ировал", "ирован",
    "ениями", "ениях", "ением", "ения", "ение", "ению", "ении", "аниями", "ании", "ание", "ания", "анием", "анию",
    "остями", "остью", "ость", "ости", "ациями", "ация", "ации", "ацию", "ацией",
    "иями", "иях", "иям", "ией", "ия", "ию",
    "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими", "ой", "ей", "ом", "ем", "ах", "ях", "ам", "ям",
    "ов", "ев", "ую", "юю", "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ых", "их", "ым", "им",
    "ает", "яет", "ует", "ют", "ят", "ут", "ет", "ит", "ать", "ять", "ить", "еть", "ал", "ил", "ла", "ли",
    "а", "я", "о", "е", "у", "ю", "ы", "и", "ь", "й",
}, key=len, reverse=True))
_ABBR = ((re.compile(r"\bмин\.(?=\s*[^\W\d])", re.IGNORECASE), "минимальная"),
         (re.compile(r"\bмакс\.(?=\s*[^\W\d])", re.IGNORECASE), "максимальная"))


# двухбуквенные глагольные окончания совпадают с концом коротких существительных (кредит, лимит, отчёт, канал):
# у них они снимаются только у возвратных глаголов («хранится») или если остаётся длинная основа
_VERB2 = frozenset({"ит", "ет", "ат", "ут", "ют", "ят", "ал", "ил", "ла", "ли"})


def stem(word: str) -> str:
    """Основа слова для сравнения тем: без возвратной частицы и окончания, не длиннее 5 символов."""
    w = word.lower().replace("ё", "е")
    reflexive = len(w) > 5 and w.endswith(("ся", "сь"))
    if reflexive:
        w = w[:-2]
    for e in _ENDINGS:
        if not w.endswith(e):
            continue
        rest = len(w) - len(e)
        if rest < (3 if len(e) == 1 else 4) or (e in _VERB2 and not reflexive and rest < 5):
            continue
        w = w[: -len(e)]
        break
    return w[:5]


_STOP_STEMS = frozenset(stem(w) for w in STOPWORDS)
_WEAK_STEMS = frozenset(stem(w) for w in WEAK_WORDS) - _STOP_STEMS


def content_stems(text: str) -> list[str]:
    """Основы содержательных слов по порядку: слова от 4 букв (аббревиатуры QR/KYB/АФМ — от 2), без служебных."""
    for rx, full in _ABBR:
        text = rx.sub(full, text)
    out = []
    for m in _WORD_RX.finditer(text):
        w = m.group(0)
        if len(w) < 4 and not (len(w) >= 2 and w.isupper()):
            continue
        s = stem(w)
        if s not in _STOP_STEMS:
            out.append(s)
    return out


def _strong(stems) -> set[str]:
    return set(stems) - _WEAK_STEMS


_QUOTED = re.compile(r"«([^»]{2,40})»")
_SENT_END = re.compile(r";|\.(?=\s+[«\"(]?[А-ЯЁA-Z0-9])|\.\s*$")


def _sentence_span(l: str, pos: int) -> tuple[int, int]:
    """Границы предложения (до «;» или точки перед заглавной), в котором стоит позиция pos."""
    left = 0
    for m in _SENT_END.finditer(l, 0, pos):
        left = m.end()
    m = _SENT_END.search(l, pos)
    return left, (m.start() if m else len(l))


def _sentences(l: str) -> list[str]:
    out, pos = [], 0
    while pos < len(l):
        s0, s1 = _sentence_span(l, pos)
        if l[s0:s1].strip():
            out.append(l[s0:s1])
        pos = max(s1 + 1, pos + 1)
    return out


def _qualifiers(text: str) -> frozenset[str]:
    t = text.lower()
    return frozenset(n for d in (SEGMENTS, CLIENT_QUALIFIERS) for n, p in d.items() if re.search(p, t))


def _channels(text: str) -> frozenset[str]:
    t = text.lower()
    return frozenset(n for n, p in CHANNEL_QUALIFIERS.items() if re.search(p, t))


def _products(text: str) -> frozenset[str]:
    return frozenset(q.strip().lower() for q in _QUOTED.findall(text))


def _req_lines(doc: Document) -> list[tuple[int, str]]:
    """Строки-требования: без заголовков, строк вне скоупа и разделов целей/этапов/рисков
    (там числа — бизнес-цели и план работ, а не требования к системе)."""
    out, skip = [], False
    for i, line in enumerate(doc.lines, start=1):
        if not line.strip():
            continue
        if _is_heading(line):
            title = _norm(strip_ids(_MD_HEADING.sub("", line))).strip().lower()
            skip = bool(re.search(NON_REQ_SECTIONS, title))
            continue
        if skip or _out_of_scope(line):
            continue
        out.append((i, line))
    return out


def _tables(doc: Document) -> dict[int, tuple[list[str], list[str]]]:
    """Строки markdown-таблиц: номер строки → (ячейки заголовка, ячейки строки). Заголовок — первая строка блока."""
    out: dict[int, tuple[list[str], list[str]]] = {}
    header: list[str] | None = None
    for i, line in enumerate(doc.lines, start=1):
        s = line.strip()
        if not s.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?|", c) for c in cells):
            continue
        if header is None:
            header = cells
            continue
        out[i] = (header, cells)
    return out


def _shared_ok(a: set[str], b: set[str], n: int = 2) -> bool:
    return len(_strong(a & b)) >= n


# ── Обобщённые численные противоречия ────────────────────────────────────────

_VALUE_RX = re.compile(
    r"(?<![\w.,+/:№-])(?P<num>\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
    r"(?:\s*(?P<mult>тыс\.?|млн\.?|млрд\.?)(?!\w))?"
    r"(?:\s*(?P<kind>календарн\w*|рабоч\w*|банковск\w*))?"
    r"\s*(?P<unit>₸|тенге\b|тг\b|kzt\b|usd\b|\$|долл\w*|eur\b|€|евро\b|руб\w*|rub\b|мрп\b|%|процент\w*|"
    r"мс\b|ms\b|миллисек\w*|сек\w*|минут\w*|мин(?:\.|\b)|час\w*|дн(?:ей|я|ю|ями|ях)\b|день\b|сут\w*|недел\w*|"
    r"месяц\w*|мес(?:\.|\b)|лет\b|год(?:а|ов)?\b)",
    re.IGNORECASE,
)
_T_PLUS = re.compile(r"(?<!\w)[TТ]\s*\+\s*(?P<num>\d{1,2})(?!\d)")
_DUR = (("мс", 0.001), ("ms", 0.001), ("миллисек", 0.001), ("сек", 1), ("мин", 60), ("час", 3600), ("дн", 86400),
        ("ден", 86400), ("сут", 86400), ("недел", 604800), ("мес", 2592000), ("лет", 31536000), ("год", 31536000))
_CUR = (("₸", "KZT"), ("тенге", "KZT"), ("тг", "KZT"), ("kzt", "KZT"), ("usd", "USD"), ("$", "USD"), ("долл", "USD"),
        ("eur", "EUR"), ("€", "EUR"), ("евро", "EUR"), ("руб", "RUB"), ("rub", "RUB"), ("мрп", "MRP"))


@dataclass
class _Value:
    line: int
    raw: str
    cls: str                 # money:KZT / pct / dur
    val: float
    kind: str                # рабочие (bd) / календарные (cd) дни
    direction: str           # min / max / exact
    core: frozenset[str]     # тема клаузы значения
    sent: frozenset[str]     # тема предложения
    head: str                # первое содержательное слово предложения (обычно подлежащее)
    words: tuple[tuple[str, str], ...]  # (слово, основа) клаузы — для заголовка находки
    period: frozenset[str]
    quals: frozenset[str]
    chans: frozenset[str]
    prods: frozenset[str]
    seq: tuple[str, ...] = ()  # содержательные основы темы по порядку — для распознавания параллельных требований
    verbs: frozenset[str] = frozenset()  # основы глаголов темы («хранится», «возвращаются»)


_VERB_FORM = re.compile(r"(?:ся|сь|ает|яет|ует|ать|ять|ить|еть)$")
_VERB_FORM2 = re.compile(r"(?:ит|ет|ют|ят|ут|ат)$")   # как у существительных «кредит», «бюджет» — только у длинных слов


def _verb_stems(text: str) -> frozenset[str]:
    return frozenset(stem(w) for w, _ in _words(text)
                     if (len(w) >= 6 and _VERB_FORM.search(w.lower())) or (len(w) >= 8 and _VERB_FORM2.search(w.lower())))


def _parallel(a: _Value, b: _Value) -> bool:
    """Однотипные требования с разными уточнениями: «время ответа API скоринга» / «время ответа торгового API» /
    «время ответа API отчётов». Главное слово (первое) общее, а каждая сторона отличается 1–2 своими
    определениями-существительными — это разные параметры. Если различие в сказуемом («резерв хранится 14 дней» /
    «резерв возвращается через 30 дней») — это одно и то же, сказанное по-разному, и значения сравниваются.
    Общий только конец («доступность шлюза» / «SLA шлюза») — тот же объект."""
    if not a.seq or not b.seq or a.seq[0] != b.seq[0]:
        return False
    ra, rb = set(a.seq) - set(b.seq), set(b.seq) - set(a.seq)
    return 0 < len(ra) <= 2 and 0 < len(rb) <= 2 and not ((ra | rb) & (a.verbs | b.verbs))


def _parallel_lines(doc: Document, la: int, lb: int) -> bool:
    """Значения одного класса в двух строках относятся к однотипным, но разным параметрам (см. _parallel)."""
    tables = _tables(doc)
    va = _line_values(la, doc.line(la), tables.get(la))
    vb = _line_values(lb, doc.line(lb), tables.get(lb))
    pairs = [(x, y) for x in va for y in vb if x.cls == y.cls]
    return bool(pairs) and all(_parallel(x, y) for x, y in pairs)


def _parse_value(m: re.Match) -> tuple[str, float, str] | None:
    try:
        num = float(re.sub(r"[   ]", "", m.group("num")).replace(",", "."))
    except ValueError:
        return None
    unit = m.group("unit").lower()
    mult = (m.group("mult") or "").lower()
    kind = (m.group("kind") or "").lower()
    kind = "bd" if kind.startswith(("рабоч", "банков")) else "cd" if kind.startswith("календ") else ""
    if unit.startswith(("%", "процент")):
        return "pct", num, ""
    for p, code in _CUR:
        if unit.startswith(p):
            num *= {"тыс": 1e3, "млн": 1e6, "млрд": 1e9}.get(mult.rstrip("."), 1.0)
            return f"money:{code}", num, ""
    if mult:
        return None
    if unit.startswith(("год", "лет")) and 1900 <= num <= 2100:
        return None              # «к концу 2027 года» — дата, а не срок
    for p, k in _DUR:
        if unit.startswith(p):
            return "dur", num * k, kind
    return None


def _direction(prefix: str, context: str) -> str:
    p = prefix.lower().rstrip()
    if re.search(r"(?:^|[\s(—–-])" + BOUND_MAX + r"$", p[-30:]):
        return "max"
    if re.search(r"(?:^|[\s(—–-])" + BOUND_MIN + r"$", p[-30:]):
        return "min"
    c = context.lower()
    mx, mn = bool(re.search(BOUND_MAX_WORDS, c)), bool(re.search(BOUND_MIN_WORDS, c))
    return "max" if mx and not mn else "min" if mn and not mx else "exact"


def _comma_span(l: str, pos: int, s0: int, s1: int) -> tuple[int, int]:
    """Границы части предложения [s0, s1) между запятыми, в которой стоит позиция pos."""
    left = max(l.rfind(",", s0, pos), s0 - 1) + 1
    right = min([p for p in (l.find(",", pos, s1),) if p != -1] or [s1])
    return left, right


def _blank_values(l: str) -> str:
    l = _VALUE_RX.sub(lambda m: " " * len(m.group(0)), l)
    return _T_PLUS.sub(lambda m: " " * len(m.group(0)), l)


def _words(text: str) -> tuple[tuple[str, str], ...]:
    for rx, full in _ABBR:
        text = rx.sub(full, text)
    return tuple((m.group(0), stem(m.group(0))) for m in _WORD_RX.finditer(text)
                 if len(m.group(0)) >= 4 or (len(m.group(0)) >= 2 and m.group(0).isupper()))


def _line_values(i: int, line: str, table: tuple[list[str], list[str]] | None) -> list[_Value]:
    out: list[_Value] = []
    if table is not None:
        header, cells = table
        if not cells or not cells[0]:
            return out
        for j, cell in enumerate(cells[1:], start=1):
            label = f"{cells[0]} {header[j] if j < len(header) else ''}"
            out += _scan_values(i, _norm(cell), label=_norm(label))
        return out
    return _scan_values(i, _norm(strip_ids(line)))


def _scan_values(i: int, l: str, label: str = "") -> list[_Value]:
    """Значения с единицей в тексте l. label — подпись ячейки таблицы (первая ячейка строки + заголовок столбца):
    тогда тема значения — подпись, а не текст ячейки."""
    out: list[_Value] = []
    blank = _blank_values(l)
    matches = [(m, _parse_value(m)) for m in _VALUE_RX.finditer(l)]
    matches += [(m, ("dur", float(m.group("num")) * 86400, "bd")) for m in _T_PLUS.finditer(l)]
    for m, parsed in matches:
        if not parsed:
            continue
        cls, val, kind = parsed
        s0, s1 = _sentence_span(l, m.start())
        sentence = l[s0:s1]
        if re.search(TARGET_VERBS, sentence.lower()):
            continue             # «сократить срок с 5 до 1 дня» — цель, а не требование
        c0, c1 = _clause_span(l, m.start(), m.end())
        c0, c1 = max(c0, s0), min(c1, s1)
        # часть предложения между запятыми: «Максимальная сумма … — 5 000 000 ₸» — слово границы и период стоят до тире
        k0, k1 = _comma_span(l, m.start(), s0, s1)
        comma = l[k0:k1]
        if label:
            core_text, sent_text = label, f"{label} {blank}"
            prefix = ctx = f"{label} {l[:m.start()]}"
        else:
            core_text, sent_text = blank[c0:c1], blank[s0:s1]
            prefix, ctx = l[c0:m.start()], l[k0:m.start()]   # граница рядом с числом / слово границы в части предложения
        core = content_stems(core_text)
        if len(_strong(core)) < 2:   # короткая клауза («(20 000 ₸ по вкладам)») — тема берётся из предложения
            core_text = sent_text
            core = content_stems(core_text)
        sent = content_stems(sent_text)
        head = next((s for s in sent if s not in _WEAK_STEMS), "")
        # продукт значения: в его части предложения, иначе ближайший слева («по карте «А» — 1 000 ₸, по карте «Б» — 5 000 ₸»)
        prods = _products(label + " " + comma) or frozenset(
            q.strip().lower() for q in _QUOTED.findall(l[s0:m.start()])[-1:]) or _products(sentence)
        out.append(_Value(
            line=i, raw=m.group(0).strip(), cls=cls, val=val, kind=kind, direction=_direction(prefix, ctx),
            core=frozenset(core), sent=frozenset(sent), head=head, words=_words(core_text),
            # период (сутки/месяц/операция) уточняет суммы и проценты; у сроков «ежедневно» — частота, а не период.
            # Ищется в части предложения между запятыми: «Лимит в сутки — 500 000 ₸» — период слева от тире.
            period=frozenset() if cls == "dur" else
            frozenset(n for n, p in VALUE_PERIODS.items() if re.search(p, (label + " " + comma).lower())),
            quals=_qualifiers(label + " " + sentence), chans=_channels(label + " " + sentence), prods=prods,
            seq=tuple(s for s in core if s not in _WEAK_STEMS), verbs=_verb_stems(core_text),
        ))
    return out


_VALUE_SUBJECT_MIN = 0.6      # доля темы одной клаузы, найденная в предложении другой
_VALUE_SUBJECT_HEAD = 0.5     # порог, если совпадает и главное слово (подлежащее) обоих предложений


def _value_subject_score(a: _Value, b: _Value) -> float:
    """Похожесть темы двух значений: ≥2 общих содержательных основы и доля темы клаузы одного значения,
    найденная в предложении другого. 0 — темы разные."""
    if not _shared_ok(set(a.sent), set(b.sent)) or _parallel(a, b):
        return 0.0
    ca = len(a.core & b.sent) / len(a.core) if a.core else 0.0
    cb = len(b.core & a.sent) / len(b.core) if b.core else 0.0
    score = max(ca, cb)
    if score >= _VALUE_SUBJECT_MIN or (score >= _VALUE_SUBJECT_HEAD and a.head and a.head == b.head):
        return score
    return 0.0


def _values_comparable(a: _Value, b: _Value) -> bool:
    if a.line == b.line or a.cls != b.cls:
        return False
    if a.kind and b.kind and a.kind != b.kind:
        return False             # рабочие vs календарные дни явно различаются
    if a.period != b.period or a.quals != b.quals:
        return False             # другой период (сутки/операция) или сегмент клиентов
    if a.chans and b.chans and not (a.chans & b.chans):
        return False
    if a.prods and b.prods and not (a.prods & b.prods):
        return False             # разные продукты/категории
    return True


def _values_conflict(a: _Value, b: _Value) -> bool:
    if abs(a.val - b.val) <= 1e-9 * max(abs(a.val), abs(b.val), 1.0):
        return False
    if {a.direction, b.direction} == {"min", "max"}:
        lo, hi = (a, b) if a.direction == "min" else (b, a)
        return lo.val > hi.val   # «не менее X» и «не более Y» совместимы, пока X ≤ Y
    return True


def _subject_phrase(a: _Value, b: _Value) -> str:
    shared = _strong(a.core & b.sent) or _strong(a.sent & b.sent)
    seen, words = set(), []
    for w, s in a.words:
        if s in shared and s not in seen:
            seen.add(s)
            words.append(w.lower())
    return " ".join(words[:6])


def scan_value_conflicts(doc: Document, exclude: set[tuple[int, int]] | None = None) -> list[Finding]:
    """Обобщённые численные противоречия: одно и то же значение (сумма, %, срок) для одной темы задано
    в разных строках по-разному. Тема — основы содержательных слов клаузы/предложения (для таблиц —
    первая ячейка строки и заголовок столбца). Не сравниваются: разные единицы и валюты, разные периоды
    (сутки/месяц/операция), сегменты и классы клиентов, разные продукты в кавычках; «не менее X» и
    «не более Y» при X ≤ Y; строки целей и плана работ. exclude — пары строк, уже найденные специфичными метриками."""
    tables = _tables(doc)
    values = [v for i, line in _req_lines(doc) for v in _line_values(i, line, tables.get(i))]
    best: dict[tuple[int, int], tuple[float, _Value, _Value]] = {}
    for x in range(len(values)):
        for y in range(x + 1, len(values)):
            a, b = values[x], values[y]
            if not _values_comparable(a, b) or not _values_conflict(a, b):
                continue
            score = _value_subject_score(a, b)
            if not score:
                continue
            key = (min(a.line, b.line), max(a.line, b.line))
            if exclude and key in exclude:
                continue
            if key not in best or score > best[key][0]:
                best[key] = (score, a, b) if a.line < b.line else (score, b, a)
    out = []
    for (la, lb), (score, a, b) in sorted(best.items()):
        subj = _subject_phrase(a, b) or "параметр"
        out.append(Finding(
            agent="consistency", category=Category.contradiction, severity=Severity.major,
            title=f"Противоречивые значения: {subj}",
            description=f"Для «{subj}» указаны разные значения: {a.raw} (L{la}) и {b.raw} (L{lb}). {VALUE_CONFLICT.why}",
            evidence=[_clip(doc.line(la)), _clip(doc.line(lb))], lines=[la, lb],
            recommendation=VALUE_CONFLICT.recommendation,
            question=f"Какое значение верное для «{subj}»: {a.raw} (L{la}) или {b.raw} (L{lb})?",
            control_id=VALUE_CONFLICT.id, confidence=round(0.55 + 0.15 * min(1.0, score), 2),
        ))
    return out


# ── Обобщённые смысловые противоречия: противоположная модальность ────────────

@dataclass
class _Polar:
    line: int
    family: int
    side: str                # neg / pos
    text: str
    topic: frozenset[str]    # тема предложения без слов-маркеров
    focus: frozenset[str]    # объект маркера: слова внутри маркера и по 3 содержательных слова вокруг
    line_topic: frozenset[str]
    quals: frozenset[str]
    chans: frozenset[str]
    prods: frozenset[str]
    anchors: frozenset[tuple[str, str]] = frozenset()   # («до»/«после», основа события)


_EXCEPTION = re.compile(r"\bкроме\b|\bза исключением\b|\bисключени")
_BEFORE = re.compile(r"\bдо\s+(?:момента\s+)?([^\W\d_]{4,})", re.IGNORECASE)
_AFTER = re.compile(r"\b(?:после|с момента|по истечении|по окончании|по завершении)\s+([^\W\d_]{4,})", re.IGNORECASE)


def _anchors(text: str) -> frozenset[tuple[str, str]]:
    return frozenset([("до", stem(m.group(1))) for m in _BEFORE.finditer(text)]
                     + [("после", stem(m.group(1))) for m in _AFTER.finditer(text)])


def _time_split(a: frozenset[tuple[str, str]], b: frozenset[tuple[str, str]]) -> bool:
    """«До списания можно отменить» и «после списания безотзывно» делят время по одному событию — не противоречие."""
    return any((("после" if k == "до" else "до"), s) in b for k, s in a)


def _focus(sentence: str, start: int, end: int, keep_inside: bool) -> set[str]:
    before = content_stems(sentence[:start])[-3:]
    after = content_stems(sentence[end:])[:3]
    inside = content_stems(sentence[start:end]) if keep_inside else []
    return set(before + after + inside)


def _polar_units(i: int, line: str, table: tuple[list[str], list[str]] | None) -> list[tuple[str, str]]:
    """(текст предложения, контекст для квалификаторов): для таблицы — ячейка с подписью строки."""
    if table is not None:
        header, cells = table
        if not cells:
            return []
        return [(_norm(f"{cells[0]}: {c}"), _norm(f"{cells[0]} {c}")) for c in cells[1:] if c]
    return [(s, s) for s in _sentences(_norm(strip_ids(line)))]


def _polar_records(doc: Document) -> list[_Polar]:
    tables = _tables(doc)
    fams = [(_rx(f.neg), _rx(f.pos)) for f in POLARITY_FAMILIES]
    out: list[_Polar] = []
    for i, line in _req_lines(doc):
        line_topic = frozenset(content_stems(_norm(strip_ids(line))))
        for sent, ctx in _polar_units(i, line, tables.get(i)):
            low = sent.lower()
            for k, (rneg, rpos) in enumerate(fams):
                gate = POLARITY_FAMILIES[k].marker_is_topic
                negs = [m for m in rneg.finditer(low) if m.end() > m.start()]
                poss = [m for m in rpos.finditer(low) if m.end() > m.start() and not is_negated(low, m.start(), m.end())
                        and not any(n.start() <= m.start() < n.end() for n in negs)]
                if bool(negs) == bool(poss):
                    continue     # ни одного маркера или исключение внутри одного предложения
                side, ms = ("neg", negs) if negs else ("pos", poss)
                marker = set()
                if not gate:
                    for m in ms:
                        marker |= set(content_stems(sent[m.start():m.end()]))
                focus: set[str] = set()
                for m in ms:
                    focus |= _focus(sent, m.start(), m.end(), keep_inside=gate)
                out.append(_Polar(
                    line=i, family=k, side=side, text=sent,
                    topic=frozenset(set(content_stems(sent)) - marker), focus=frozenset(focus - marker),
                    line_topic=line_topic - marker, quals=_qualifiers(ctx), chans=_channels(ctx), prods=_products(ctx),
                    anchors=_anchors(sent),
                ))
    return out


def scan_polarity_conflicts(doc: Document) -> list[Finding]:
    """Одно действие с противоположной модальностью в разных строках: запрет vs разрешение, фиксировано vs
    пересматривается, только с согласия vs автоматически, безотзывно vs можно отозвать, до завершения проверки
    нельзя vs можно на время проверки. Тема совпадает, если общий объект маркера (focus) и ≥2 общих
    содержательных основы. Не сравниваются строки про разные сегменты/продукты и строки с явным исключением."""
    recs = _polar_records(doc)
    out: list[Finding] = []
    seen: set[tuple[int, int]] = set()
    for a in recs:
        if a.side != "neg":
            continue
        for b in recs:
            if b.side != "pos" or b.family != a.family or b.line == a.line:
                continue
            key = (min(a.line, b.line), max(a.line, b.line))
            if key in seen:
                continue
            # общее правило (без сегмента) против частного случая — противоречие; правила для двух разных
            # сегментов/продуктов — нет
            if (a.quals and b.quals and a.quals != b.quals) or (a.prods and b.prods and not (a.prods & b.prods)):
                continue
            if a.chans and b.chans and not (a.chans & b.chans):
                continue
            if _EXCEPTION.search(a.text.lower()) or _EXCEPTION.search(b.text.lower()):
                continue
            if _time_split(a.anchors, b.anchors):
                continue
            fam = POLARITY_FAMILIES[a.family]
            if not _strong(a.focus & b.focus):
                continue         # нет общего объекта у маркеров
            if not (_shared_ok(set(a.topic), set(b.topic))
                    or (not fam.sentence_topic and _shared_ok(set(a.line_topic), set(b.line_topic)))):
                continue
            seen.add(key)
            shared =[w for w, s in _words(a.text) if s in _strong(a.focus & b.focus)]
            topic = " ".join(dict.fromkeys(w.lower() for w in shared)) or fam.name
            la, lb = key
            cp = POLARITY_CONFLICT
            out.append(Finding(
                agent="consistency", category=Category.contradiction, severity=Severity(cp.severity),
                title=f"Смысловое противоречие: {fam.name} — {topic}",
                description=f"L{a.line}: {fam.neg_label}; L{b.line}: {fam.pos_label}. {cp.why}",
                evidence=[_clip(doc.line(la)), _clip(doc.line(lb))], lines=[la, lb],
                recommendation=cp.recommendation, question=cp.question.format(a=a.line, b=b.line, topic=topic),
                control_id=cp.id, confidence=0.6,
            ))
    return out


DUP_RATIO = 0.9      # порог похожести строк-дублей (SequenceMatcher.ratio)
DUP_WINDOW = 400     # сравниваем строку с 399 следующими длинными строками
DUP_MAX = 10


def scan_duplicates(doc: Document) -> list[Finding]:
    """Почти совпадающие длинные строки (ratio ≥ 0.9) в окне из 400 следующих.

    Результат тот же, что при прямом сравнении каждой пары SequenceMatcher-ом, но дорогой ratio()
    считается только для пар, прошедших точные верхние оценки ratio (ни одна не отсекает настоящий дубль):
    1) по длинам — real_quick_ratio: 2·min(|a|,|b|)/(|a|+|b|);
    2) по мультимножествам символов — quick_ratio (счётчики символов считаются один раз на строку);
    3) по 4-граммам: при ratio = 2M/T совпавших блоков не больше T − 2M + 1, поэтому позиций строки a,
       чья 4-грамма целиком лежит в совпавшем блоке (и значит есть в b), не меньше M − 3·(T − 2M + 1).
    Поиск останавливается на DUP_MAX-й находке: прежде всё, что после неё, всё равно отбрасывалось."""
    reqs = [(i, l.strip()) for i, l in enumerate(doc.lines, start=1) if len(l.strip()) > 50]
    low = [l.lower() for _, l in reqs]
    lens = [len(s) for s in low]
    grams: list[set[str] | None] = [None] * len(low)     # множества 4-грамм — лениво, только для пар после 1–2
    counts: list[dict[str, int]] = []
    for s in low:
        c: dict[str, int] = {}
        for ch in s:
            c[ch] = c.get(ch, 0) + 1
        counts.append(c)
    n = len(reqs)
    out: list[Finding] = []
    for a in range(n):
        na, ca = lens[a], counts[a]
        for b in range(a + 1, min(n, a + DUP_WINDOW)):
            nb = lens[b]
            total = na + nb
            if 2.0 * (na if na < nb else nb) / total < DUP_RATIO:          # real_quick_ratio
                continue
            cb = counts[b]
            small, big = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
            inter = 0
            for ch, v in small.items():
                w = big.get(ch)
                if w:
                    inter += v if v < w else w
            if 2.0 * inter / total < DUP_RATIO:                             # quick_ratio
                continue
            gb = grams[b]
            if gb is None:
                gb = grams[b] = {low[b][k:k + 4] for k in range(nb - 3)}
            sa = low[a]
            shared = sum(1 for k in range(na - 3) if sa[k:k + 4] in gb)
            m_min = DUP_RATIO * total / 2                                    # M при ratio ровно 0.9
            if shared < m_min - 3 * (total - 2 * m_min + 1) - 1e-6:         # 4-граммы
                continue
            r = SequenceMatcher(None, low[a], low[b]).ratio()
            if r < DUP_RATIO:
                continue
            i, la = reqs[a]
            j, lb = reqs[b]
            out.append(Finding(
                agent="consistency", category=Category.contradiction, severity=Severity.minor,
                title="Дублирующееся требование",
                description=f"Строки L{i} и L{j} практически совпадают ({r:.0%}). Дубли расходятся при правках и порождают противоречия.",
                evidence=[_clip(la), _clip(lb)], lines=[i, j],
                recommendation="Оставить требование в одном месте и сослаться на него по ID.",
                confidence=0.8,
            ))
            if len(out) >= DUP_MAX:
                return out
    return out
