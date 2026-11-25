"""Проверка требований по ISO/IEC/IEEE 29148:2018 (п. 5.2.5, 5.2.6) и IEEE 830-1998 (п. 4.3).

Конвейер (агенты с узкими ролями):
  1. Segmenter          — нарезка документа на требования (agents/segmenter.py);
  2. Rule Checker       — детерминированные сигналы по каждой характеристике (размытость, TBD, пустые поля,
                          несколько правил в одном пункте, опасные требования, числовые/смысловые противоречия,
                          согласованность статусов API, семантика HTTP-методов);
  3. Requirement Critic — LLM оценивает каждое требование пакетами по 10 характеристикам + риски ИБ;
  4. Set Critic         — LLM оценивает набор: противоречия, пропуски, чек-лист документа, вопросы автору;
  5. Grounding Verifier — каждая цитата ищется в документе; неподтверждённые претензии отбрасываются;
  6. Verdict            — сводка, критичные дефекты (непроверяемые/противоречивые/неполные) и итог.
Без LLM работают шаги 1, 2, 5, 6 — отчёт строится всегда.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agents import rules
from agents.base import LLMClient, load_prompt
from agents.segmenter import ReqUnit, Segmentation, segment
from config import LLMConfig
from knowledge.iso29148 import BY_CODE, CRITICAL_CLASSES, INDIVIDUAL_CODES, SET_CODES
from knowledge.language import VAGUE_TERMS
from parsing import Document

SEV_RANK = {"critical": 0, "major": 1, "minor": 2}
BATCH = 9


# ── модели результата ─────────────────────────────────────────────────────────

class Issue(BaseModel):
    characteristic: str
    severity: str = "major"
    problem: str
    evidence: list[str] = Field(default_factory=list)
    lines: list[int] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)      # ID требований (R-xx)
    fix: str = ""
    source: str = "rules"                                  # rules | llm | rules+llm


class UnitReview(BaseModel):
    rid: str
    doc_ref: str
    section: str
    kind: str
    text: str
    quote: str
    lines: list[int]
    pages: list[int] = Field(default_factory=list)
    statuses: dict[str, str] = Field(default_factory=dict)  # code → ok | issue | warn | na
    issues: list[Issue] = Field(default_factory=list)
    rewrite: str = ""
    mode: str = "rules"


class SetReview(BaseModel):
    statuses: dict[str, str] = Field(default_factory=dict)
    comments: dict[str, str] = Field(default_factory=dict)
    issues: list[Issue] = Field(default_factory=list)
    missing: list[dict] = Field(default_factory=list)
    checklist: list[dict] = Field(default_factory=list)
    questions: list[dict] = Field(default_factory=list)
    summary: str = ""
    top_fixes: list[str] = Field(default_factory=list)
    mode: str = "rules"


class IsoReport(BaseModel):
    document_name: str
    document_title: str
    run_at: str
    model: str
    agent_version: str
    mode: str
    duration_s: float
    units: list[UnitReview]
    set_review: SetReview
    verdict: str = ""
    verdict_level: str = ""          # ready | conditional | not_ready
    priorities: list[str] = Field(default_factory=list)
    critical: dict[str, list[str]] = Field(default_factory=dict)   # класс → ["R-06: …"]
    stats: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict] = Field(default_factory=list)


# ── вспомогательное ───────────────────────────────────────────────────────────

_SENT = re.compile(r"(?<=[.!?])\s+(?=[«\"(]?[А-ЯЁA-Z])")
_IMPL = re.compile(r"\b(get|post|put|patch|delete|rest|json|controller|endpoint|эндпоинт|таблиц\w* бд|поле бд|sql)\b", re.I)
_PLACEHOLDER_VAL = re.compile(r"(?::\s*|Prod:\s*)[—–-]\s*(?:$|;|\.)")
_EMPTY_TEMPLATE = re.compile(r"^(заполня\w*|будет заполнен\w*|будут заполнен\w*|tbd|tba|—|-|н/д)\b", re.I)
_STATE_CHANGE = re.compile(r"сохран|измен|удал|создан|обновл|save|update|delete|create|set", re.I)
_VAGUE = [(re.compile(v.pattern, re.I), v) for v in VAGUE_TERMS]


def units_key(t: str) -> str:
    m = re.match(r"^\s*(?:Поле|Parameter|Параметр):\s*([^;]+)", t)
    return m.group(1).strip() if m else ""


def _same_table(doc: Document, a: int, b: int) -> bool:
    """Строки a и b — строки одной таблицы (разные строки таблицы — это разные случаи, а не противоречие)."""
    lo, hi = sorted((a, b))
    return all(doc.line(n).strip().startswith("|") for n in range(lo, hi + 1))


def _first_words(s: str, n: int = 14) -> str:
    w = s.split()
    return " ".join(w[:n])


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if len(s.strip()) > 3]


def _issue(code: str, sev: str, problem: str, unit: ReqUnit | None, evidence: str | list[str] = "",
           fix: str = "", related: list[str] | None = None, lines: list[int] | None = None) -> Issue:
    ev = [evidence] if isinstance(evidence, str) and evidence else list(evidence or [])
    return Issue(characteristic=code, severity=sev, problem=problem, evidence=ev,
                 lines=lines if lines is not None else (list(unit.lines[:1]) if unit else []),
                 related=related or [], fix=fix, source="rules")


# ── правила по отдельному требованию ──────────────────────────────────────────

REQ_KINDS = {"functional", "nonfunctional", "data", "integration", "business_rule", "ui", "api_logic", "business"}


def rule_review(u: ReqUnit) -> list[Issue]:
    out: list[Issue] = []
    t = u.text
    low = t.lower().replace("ё", "е")
    # полнота: TBD, незаполненные поля и разделы шаблона
    for rx, v in _VAGUE:
        if v.kind != "tbd":
            continue
        m = rx.search(low)
        if m:
            out.append(_issue("complete", "major", f"Незакрытый вопрос в тексте («{m.group(0)}»): требование не определено.",
                              u, t[m.start():m.start() + 60] or t, "Закрыть вопрос до начала разработки и зафиксировать решение."))
            break
    if _EMPTY_TEMPLATE.match(low) and len(t) < 80:
        out.append(_issue("complete", "critical", f"Раздел «{u.section.split(' › ')[-1]}» не заполнен: вместо требований — «{t}».",
                          u, t, "Заполнить раздел конкретными требованиями (что и как тестируется, критерии и данные)."))
        out.append(_issue("conforming", "major", "Обязательный раздел шаблона оставлен незаполненным.", u, t,
                          "Заполнить раздел по шаблону или явно указать «не требуется» с обоснованием."))
    # «—» вместо значения: только в смысловых колонках (значение/описание/логика) и для Prod-окружения;
    # «DB field: —» в ответе означает «поля в БД нет» и пропуском не является
    for part in t.split(";"):
        hdr, _, val = part.partition(":")
        h = hdr.strip().lower()
        if (h in ("значение", "value", "описание", "description", "logic", "логика") and val.strip() in ("—", "–", "-", "")) \
                or re.search(r"Prod:\s*[—–-]\s*$", part.strip()):
            evidence = part.strip()
            if h in ("значение", "value") and units_key(t):
                evidence = f"{units_key(t)}; {part.strip()}"
            out.append(_issue("complete", "major", "Значение не заполнено («—»): требование нельзя реализовать и проверить.",
                              u, re.sub(r"^Поле:\s*", "", evidence)[:80], "Указать значение или явно отметить «не применимо» с причиной."))
            break
    # однозначность и проверяемость: размытые формулировки
    body_has_digit = bool(re.search(r"\d", rules.strip_ids(t))) or bool(re.search(
        r"(в течение|за|не более|не позднее|не реже|раз в)\s+(\w+\s+)?(секунд|минут|час|сутк|сут|дн|недел|месяц|год)", low))
    for rx, v in _VAGUE:
        if v.kind == "tbd":
            continue
        m = rx.search(low)
        if not m:
            continue
        if v.kind == "metric" and body_has_digit:
            continue
        code = "verifiable" if v.kind == "metric" else "unambiguous"
        sev = "major" if u.kind in ("nonfunctional", "functional") and v.kind == "metric" else "minor"
        out.append(_issue(code, sev, f"Размытая формулировка «{m.group(0)}» допускает разное толкование.", u,
                          t[max(0, m.start() - 20): m.end() + 20].strip(), f"Сделать требование измеримым: {v.hint}."))
    if u.kind == "nonfunctional" and not body_has_digit:
        out.append(_issue("verifiable", "major", "Нефункциональное требование без измеримого значения — его нельзя проверить тестом.",
                          u, _first_words(t), "Задать числовой критерий и условия измерения (нагрузка, перцентиль, окружение)."))
    # единичность: несколько правил в одном пункте
    sents = _sentences(t)
    if u.kind in REQ_KINDS and len(sents) >= 2 and u.kind != "business":
        sev = "major" if len(sents) >= 3 else "minor"
        out.append(_issue("singular", sev, f"В одном пункте {len(sents)} самостоятельных утверждения — их нельзя "
                          f"независимо реализовать, протестировать и проследить.", u, _first_words(sents[1], 12),
                          "Разбить на отдельные требования с собственными ID (по одному правилу на пункт)."))
    # прослеживаемость: нет уникального идентификатора
    if u.kind in {"functional", "nonfunctional", "data", "integration", "ui"} and not u.explicit_id:
        out.append(_issue("traceable", "minor", "У требования нет уникального идентификатора — на него нельзя сослаться в тестах и задачах.",
                          u, _first_words(t, 10), "Присвоить ID (например, НФТ-1, ДАН-1, ИНТ-1) по аналогии с Ф-1…Ф-4."))
    # уместность: детали реализации в бизнес-требованиях
    if u.kind in {"functional", "business"} and _IMPL.search(t):
        m = _IMPL.search(t)
        out.append(_issue("appropriate", "minor", f"Деталь реализации («{m.group(0)}») в требовании уровня «что должно быть».",
                          u, t[max(0, m.start() - 20): m.end() + 20].strip(), "Перенести техническое решение в спецификацию интерфейса."))
    return out


def api_set_rules(seg: Segmentation, doc: Document) -> list[Issue]:
    """Правила согласованности спецификации API (метод, статусы, тело запроса)."""
    out: list[Issue] = []
    units = seg.units
    iface = [u for u in units if u.kind in ("interface", "api_logic")]
    if not iface:
        return out
    method_u = next((u for u in iface if re.search(r"rest type|http method|метод запроса", u.text, re.I)), None)
    desc_u = next((u for u in iface if re.search(r"^поле:\s*description|назначение метода", u.text, re.I)), None)
    url_u = next((u for u in iface if re.search(r"request url|url", u.text, re.I) and "/" in u.text), None)
    if method_u and re.search(r"\bGET\b", method_u.text):
        changers = [u for u in (desc_u, url_u) if u and _STATE_CHANGE.search(u.text)]
        logic_change = [u for u in units if u.kind in ("api_logic", "business_rule") and re.search(r"замен|сохран|измен", u.text, re.I)]
        if changers or logic_change:
            rel = [u.rid for u in changers + logic_change[:2]]
            ev = [re.sub(r"^Поле:\s*", "", method_u.text)] + [_first_words(u.text, 10) for u in (changers or logic_change)[:1]]
            out.append(_issue("set_consistent", "critical",
                              "Метод изменяет состояние (сохраняет/заменяет настройку), но объявлен как GET: GET должен быть безопасным "
                              "и идемпотентным, параметры попадают в URL и логи прокси, запросы кэшируются и повторяются.",
                              method_u, [method_u.quote.strip("| ")[:80]] + [], "Использовать PUT/POST (или PATCH) с телом запроса; "
                              "для чтения порога описать отдельный GET-метод.", related=[method_u.rid] + rel,
                              lines=method_u.lines + [l for u in (changers or logic_change)[:1] for l in u.lines]))
    # коды, которые упоминаются, но не описаны в перечне статусов
    described = set()
    for u in units:
        m = re.search(r"код (\d{3})(?!, пример)", u.doc_ref)
        if m:
            described.add(m.group(1))
    mentioned: dict[str, ReqUnit] = {}
    for u in iface:
        for code in re.findall(r"\b([1-5]\d\d)\b", u.text):
            if code not in described and code not in mentioned and not re.search(r"\b32 bit|\d{3} \d{3}", u.text):
                mentioned[code] = u
    for code, u in mentioned.items():
        if code in ("100",):
            continue
        out.append(_issue("set_consistent", "major", f"Код ответа {code} упоминается, но не описан в перечне статусов метода.", u,
                          [m.group(0) for m in [re.search(rf".{{0,30}}\b{code}\b.{{0,30}}", u.text)] if m],
                          f"Описать код {code} (условие, тело ответа) или убрать упоминание.", related=[u.rid], lines=u.lines[:1]))
    # Content-Type при отсутствии тела
    ct = next((u for u in iface if re.search(r"content-type", u.text, re.I)), None)
    nobody = next((u for u in iface if re.search(r"тело не используется|body is not used|без тела", u.text, re.I)), None)
    if ct and nobody:
        ev_nobody = re.search(r"[^.;]*(?:тело не используется|body is not used|без тела)[^.;]*", nobody.text, re.I)
        out.append(_issue("set_consistent", "minor", "Указан Content-Type тела запроса, но тело не используется.", ct,
                          [re.search(r"Content-Type[^.;]*", ct.text, re.I).group(0).strip(),
                           ev_nobody.group(0).strip() if ev_nobody else _first_words(nobody.text, 4)],
                          "Либо передавать параметры в теле (application/json), либо убрать Content-Type.",
                          related=list(dict.fromkeys([ct.rid, nobody.rid])), lines=sorted(set(ct.lines + nobody.lines))))
    return out


def set_rules(seg: Segmentation, doc: Document) -> tuple[list[Issue], dict[str, str]]:
    out: list[Issue] = []
    by_line = {l: u for u in seg.units for l in u.lines}
    # числовые и смысловые противоречия (детекторы Consistency-агента)
    for f in rules.scan_metric_conflicts(doc):
        ls = sorted(set(f.lines))
        if len(ls) >= 2 and all(_same_table(doc, ls[0], x) for x in ls[1:]):
            continue
        rel = list(dict.fromkeys(by_line[l].rid for l in f.lines if l in by_line))
        out.append(Issue(characteristic="set_consistent", severity="critical" if f.severity.value != "minor" else "major",
                         problem=f.description or f.title, evidence=f.evidence[:2], lines=f.lines[:4], related=rel,
                         fix=f.recommendation, source="rules"))
    out += api_set_rules(seg, doc)
    text = doc.text.lower().replace("ё", "е")
    # версия документа
    if re.search(r"верси\w*[^.\n]{0,40}не заполн|верси\w*/дат\w*[^.\n]{0,40}не заполн", text) or not re.search(r"верси\w*\s*[:№]?\s*\d", text):
        ln = next((i for i, l in enumerate(doc.lines, 1) if re.search(r"верси", l, re.I)), None)
        out.append(Issue(characteristic="set_modifiable", severity="major",
                         problem="Версия и дата документа не зафиксированы — нельзя однозначно сослаться на согласованную редакцию.",
                         evidence=[doc.lines[ln - 1].strip()[:90]] if ln else [], lines=[ln] if ln else [],
                         fix="Добавить номер версии, дату и историю изменений.", source="rules"))
    # приоритеты
    if not re.search(r"приоритет|must|should|moscow|обязательн\w* к реализ|критичност", text):
        out.append(Issue(characteristic="set_ranked", severity="minor",
                         problem="У требований не указаны приоритет и стабильность — нельзя планировать MVP и регресс.",
                         fix="Добавить приоритет (MoSCoW или P1–P3) каждому требованию.", source="rules"))
    # чек-лист ревью не заполнен
    if seg.checklist and all(any("/" in v for v in r["values"][:2]) for r in seg.checklist if r["values"]):
        ln = seg.checklist[0]["line"]
        out.append(Issue(characteristic="set_verifiable", severity="major",
                         problem=f"Чек-лист ревью требований ({len(seg.checklist)} пунктов) не заполнен: отметки «Да / Нет» не проставлены.",
                         evidence=[doc.lines[ln - 1].strip()[:90]], lines=[ln], fix="Провести ревью и проставить отметки с примечаниями.",
                         source="rules"))
    # блок согласующих пуст
    for i, l in enumerate(doc.lines, 1):
        if re.search(r"согласующ", l, re.I) and l.lstrip().startswith("#"):
            rows = [x for x in doc.lines[i:i + 8] if x.strip().startswith("|")
                    and not ("-" in x and set(x.replace("|", "").strip()) <= set("-: "))]
            if len(rows) >= 2 and all(not c.strip() for r in rows[1:] for c in r.strip().strip("|").split("|")):
                out.append(Issue(characteristic="set_traceable", severity="minor",
                                 problem="Блок согласующих пуст: нет ответственных и подписей — требования не согласованы с владельцами систем.",
                                 evidence=[l.strip("# ").strip()], lines=[i], fix="Указать согласующих (бизнес, ИБ, архитектура, процессинг).",
                                 source="rules"))
            break
    statuses: dict[str, str] = {}
    for i in out:
        statuses[i.characteristic] = "issue"
    return out, statuses


# ── LLM: проверка требований и набора ─────────────────────────────────────────

class _RawReviews(BaseModel):
    reviews: list[dict[str, Any]]


class _RawContra(BaseModel):
    contradictions: list[dict[str, Any]]


async def _llm_contradictions(client, seg: Segmentation, stats: dict) -> list[dict]:
    focus = [u for u in seg.units if u.kind in ("functional", "business_rule", "acceptance", "nonfunctional", "business",
                                                "integration", "api_logic", "interface", "data", "process", "ui")]
    table = "\n".join(f"{u.rid} | {u.doc_ref} | {u.kind} | {u.text}" + (f" | пример: {' '.join(u.extra)[:200]}" if u.extra else "")
                      for u in focus)
    raw, tokens = await client.chat_json(load_prompt("iso_contradictions"), f"## Требования\n{table}", _RawContra)
    stats["tokens"] += tokens
    return raw.contradictions


class _RawSet(BaseModel):
    statuses: dict[str, Any] = Field(default_factory=dict)
    comments: dict[str, Any] = Field(default_factory=dict)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    missing: list[dict[str, Any]] = Field(default_factory=list)
    checklist: list[dict[str, Any]] = Field(default_factory=list)
    questions: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    top_fixes: list[Any] = Field(default_factory=list)


def _norm_status(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in ("ok", "pass", "yes", "да", "выполнено", "✅"):
        return "ok"
    if s in ("issue", "fail", "no", "нет", "нарушено", "❌", "violation"):
        return "issue"
    if s in ("warn", "partial", "частично", "⚠️"):
        return "warn"
    return "na"


def _norm_sev(v: Any) -> str:
    s = str(v or "").strip().lower()
    return {"high": "critical", "критично": "critical", "critical": "critical", "blocker": "critical",
            "medium": "major", "major": "major", "важно": "major", "low": "minor", "minor": "minor", "мелочь": "minor"}.get(s, "major")


def _ground(doc: Document, quotes: list[str]) -> tuple[list[str], list[int]]:
    good, lines = [], []
    for q in quotes:
        q = str(q or "").strip().strip("«»\"")
        if len(q) < 3:
            continue
        ln, score = doc.locate_quote(q)
        if ln and score >= 0.72:
            good.append(q)
            lines.append(ln)
    return good, lines


def _units_table(units: list[ReqUnit], full: bool = False) -> str:
    rows = []
    for u in units:
        t = u.text if full else u.short
        rows.append(f"{u.rid} | {u.doc_ref} | {u.kind} | L{u.lines[0]} | {t}")
    return "\n".join(rows)


async def _llm_batch(client, doc: Document, seg: Segmentation, batch: list[ReqUnit], hints: dict[str, list[Issue]],
                     stats: dict) -> dict[str, dict]:
    hint_lines = []
    for u in batch:
        hs = hints.get(u.rid, [])
        if hs:
            hint_lines.append(f"{u.rid}: " + "; ".join(f"[{h.characteristic}] {h.problem}" for h in hs[:6]))
    payload = "\n\n".join(
        f"### {u.rid} (в документе: {u.doc_ref}; раздел: {u.section}; вид: {u.kind}; строки: "
        f"{', '.join('L' + str(l) for l in u.lines[:4])})\n{u.quote}" + (f"\nПримеры/код: {' | '.join(u.extra)[:400]}" if u.extra else "")
        for u in batch)
    user = (f"## Документ\n<tz>\n{doc.numbered(80_000)}\n</tz>\n\n## Все требования документа (для перекрёстных ссылок)\n"
            f"{_units_table(seg.units)}\n\n## Сигналы правил\n{chr(10).join(hint_lines) or '—'}\n\n"
            f"## Пакет для проверки ({len(batch)} требований: {', '.join(u.rid for u in batch)})\n{payload}")
    raw, tokens = await client.chat_json(load_prompt("iso_requirement"), user, _RawReviews)
    stats["tokens"] += tokens
    out = {}
    ids = {u.rid for u in batch}
    for r in raw.reviews:
        rid = str(r.get("rid", "")).strip()
        if rid in ids:
            out[rid] = r
    return out


async def _llm_set(client, doc: Document, seg: Segmentation, rule_issues: list[Issue], stats: dict) -> _RawSet:
    hints = "\n".join(f"- [{i.characteristic}] {i.problem} ({', '.join(i.related)})" for i in rule_issues[:25]) or "—"
    checklist = "\n".join(f"- {r['item']}" for r in seg.checklist) or "— (чек-листа в документе нет)"
    user = (f"## Документ\n<tz>\n{doc.numbered(80_000)}\n</tz>\n\n## Требования (ID в отчёте | ID в документе | вид | строка | текст)\n"
            f"{_units_table(seg.units, full=True)}\n\n## Сигналы правил\n{hints}\n\n## Чек-лист ревью из документа\n{checklist}")
    raw, tokens = await client.chat_json(load_prompt("iso_set"), user, _RawSet)
    stats["tokens"] += tokens
    return raw


def _merge_issue_lists(rule: list[Issue], llm: list[Issue]) -> list[Issue]:
    """Дубли «правило + LLM» про одну характеристику и одну строку склеиваются; текст берётся у LLM."""
    out = list(llm)
    for r in rule:
        dup = next((x for x in out if x.characteristic == r.characteristic and (set(x.lines) & set(r.lines) or not r.lines)), None)
        if dup:
            if dup.source == "llm":
                dup.source = "rules+llm"
            continue
        out.append(r)
    return sorted(out, key=lambda i: (SEV_RANK.get(i.severity, 1), i.characteristic))


def _statuses_from(issues: list[Issue], codes: list[str], llm_statuses: dict[str, str] | None, na: set[str]) -> dict[str, str]:
    st = {}
    for c in codes:
        st[c] = "na" if c in na else "ok"
        if llm_statuses and c in llm_statuses:
            st[c] = llm_statuses[c]
    for i in issues:
        if i.characteristic in st:
            st[i.characteristic] = "issue" if (i.source != "rules" or i.severity != "minor") else (
                "issue" if st[i.characteristic] == "issue" else "warn")
    return st


def _na_for(u: ReqUnit) -> set[str]:
    if u.kind == "acceptance":
        return {"singular", "traceable", "appropriate"}
    if u.kind in ("interface", "architecture", "process"):
        return {"traceable"}
    return set()


async def _run(doc: Document, cfg: LLMConfig, use_llm: bool, on_event=None) -> tuple[Segmentation, list[UnitReview], SetReview, dict]:
    stats = {"tokens": 0, "rejected": 0, "llm_calls": 0, "llm_failed": 0, "notes": []}
    emit = on_event or (lambda *a: None)
    t0 = time.perf_counter()
    emit("segmenter", "start")
    seg = segment(doc)
    emit("segmenter", "done", f"{len(seg.units)} требований")
    by_line = {l: u for u in seg.units for l in u.lines}

    # правила по каждому требованию + опасные требования + размытые формулировки из Consistency-агента
    emit("rules", "start")
    rule_issues: dict[str, list[Issue]] = {u.rid: rule_review(u) for u in seg.units}
    for f in rules.scan_antipatterns(doc):
        u = next((by_line[l] for l in f.lines if l in by_line), None)
        if u:
            rule_issues[u.rid].append(Issue(characteristic="security", severity=f.severity.value, problem=f"{f.title}. {f.description}",
                                            evidence=f.evidence[:1], lines=f.lines[:2], fix=f.suggested_text or f.recommendation,
                                            source="rules"))
    set_issues_rules, _ = set_rules(seg, doc)
    emit("rules", "done", f"сигналов: {sum(len(v) for v in rule_issues.values()) + len(set_issues_rules)}")

    llm_reviews: dict[str, dict] = {}
    raw_set: _RawSet | None = None
    contra_raw: list[dict] = []
    model_used = "офлайн-правила"
    if use_llm and cfg.enabled:
        client = LLMClient(cfg)
        if getattr(cfg, "privacy_enabled", False):
            try:
                from privacy import PrivacyLLMClient
                client = PrivacyLLMClient(client)
                client.prime(doc.text)
            except Exception:
                pass
        batches = [seg.units[i:i + BATCH] for i in range(0, len(seg.units), BATCH)]
        emit("llm", "start", f"{len(batches)} пакетов + проверка набора")

        async def one(b):
            stats["llm_calls"] += 1
            try:
                return await _llm_batch(client, doc, seg, b, rule_issues, stats)
            except Exception as e:
                stats["llm_failed"] += 1
                stats["notes"].append(f"пакет {b[0].rid}–{b[-1].rid}: {str(e)[:120]}")
                return {}

        async def set_task():
            stats["llm_calls"] += 1
            try:
                return await _llm_set(client, doc, seg, set_issues_rules, stats)
            except Exception as e:
                stats["llm_failed"] += 1
                stats["notes"].append(f"проверка набора: {str(e)[:120]}")
                return None

        async def contra_task():
            stats["llm_calls"] += 1
            try:
                return await _llm_contradictions(client, seg, stats)
            except Exception as e:
                stats["llm_failed"] += 1
                stats["notes"].append(f"поиск противоречий: {str(e)[:120]}")
                return []

        results = await asyncio.gather(*(one(b) for b in batches), set_task(), contra_task())
        for r in results[:-2]:
            llm_reviews.update(r)
        raw_set = results[-2]
        contra_raw = results[-1] or []
        model_used = getattr(client, "active_model", cfg.model)
        emit("llm", "done", f"вызовов {stats['llm_calls']}, неуспешных {stats['llm_failed']}")

    # сборка по требованиям с заземлением цитат
    emit("verifier", "start")
    rids = {u.rid for u in seg.units}
    unit_reviews: list[UnitReview] = []
    pages = getattr(doc, "line_pages", None)
    for u in seg.units:
        llm = llm_reviews.get(u.rid)
        llm_issues: list[Issue] = []
        llm_statuses = None
        rewrite = ""
        if llm:
            llm_statuses = {c: _norm_status(v) for c, v in (llm.get("statuses") or {}).items() if c in INDIVIDUAL_CODES}
            for it in llm.get("issues") or []:
                if not isinstance(it, dict):
                    continue
                code = str(it.get("characteristic", "")).strip()
                if code not in INDIVIDUAL_CODES:
                    continue
                ev = it.get("evidence") or []
                ev = [ev] if isinstance(ev, str) else [str(x) for x in ev]
                good, lines = _ground(doc, ev)
                if not good:
                    stats["rejected"] += 1        # претензия без подтверждённой цитаты — отбрасывается
                    continue
                rel = [str(x).strip() for x in (it.get("related") or []) if str(x).strip() in rids and str(x).strip() != u.rid]
                llm_issues.append(Issue(characteristic=code, severity=_norm_sev(it.get("severity")),
                                        problem=str(it.get("problem", "")).strip()[:600], evidence=good, lines=lines,
                                        related=rel, fix=str(it.get("fix", "")).strip()[:600], source="llm"))
            rewrite = str(llm.get("rewrite") or "").strip()[:800]
        rule_ = [] if llm is None else []
        merged = _merge_issue_lists(rule_issues[u.rid], llm_issues)
        # rules в LLM-режиме: оставляем только сигналы, которые LLM не опроверг явным «ok» (кроме ИБ и TBD)
        if llm_statuses:
            merged = [i for i in merged if i.source != "rules" or llm_statuses.get(i.characteristic) != "ok"
                      or i.characteristic in ("security", "complete")]
        st = _statuses_from(merged, INDIVIDUAL_CODES, llm_statuses, _na_for(u))
        explained = {i.characteristic for i in merged}
        for c, v in st.items():
            if v == "issue" and c not in explained:
                st[c] = "warn"        # «нарушено» без конкретики и цитаты в отчёт не идёт — только как замечание
        unit_reviews.append(UnitReview(
            rid=u.rid, doc_ref=u.doc_ref, section=u.section, kind=u.kind, text=u.text, quote=u.quote, lines=u.lines,
            pages=sorted({pages[l - 1] for l in u.lines if pages and l - 1 < len(pages)}) if pages else [],
            statuses=st, issues=merged, rewrite=rewrite, mode="llm" if llm else "rules"))

    # набор
    set_llm_issues: list[Issue] = []
    sr = SetReview(mode="llm" if raw_set else "rules")
    if raw_set:
        for it in raw_set.issues:
            code = str(it.get("characteristic", "")).strip()
            if code not in SET_CODES:
                continue
            ev = it.get("evidence") or []
            ev = [ev] if isinstance(ev, str) else [str(x) for x in ev]
            good, lines = _ground(doc, ev)
            if not good:
                stats["rejected"] += 1
                continue
            rel = [str(x).strip() for x in (it.get("requirements") or it.get("related") or []) if str(x).strip() in rids]
            set_llm_issues.append(Issue(characteristic=code, severity=_norm_sev(it.get("severity")),
                                        problem=str(it.get("problem", "")).strip()[:700], evidence=good, lines=lines,
                                        related=rel, fix=str(it.get("fix", "")).strip()[:600], source="llm"))
        sr.missing = [m for m in raw_set.missing if isinstance(m, dict) and m.get("topic")][:15]
        for m in sr.missing:
            m["related"] = [r for r in (m.get("related") or []) if r in rids]
        sr.checklist = [c for c in raw_set.checklist if isinstance(c, dict) and c.get("item")]
        sr.questions = [q for q in raw_set.questions if isinstance(q, dict) and q.get("question")][:14]
        for q in sr.questions:
            q["related"] = [r for r in (q.get("related") or []) if r in rids]
        sr.summary = str(raw_set.summary or "").strip()[:1200]
        sr.top_fixes = [str(x).strip()[:300] for x in raw_set.top_fixes if str(x).strip()][:10]
        sr.comments = {k: str(v)[:600] for k, v in raw_set.comments.items() if k in SET_CODES}
    # противоречия от Contradiction Hunter (с заземлением), дубли с уже найденными склеиваются
    for c in contra_raw:
        if not isinstance(c, dict):
            continue
        ev = c.get("evidence") or []
        ev = [ev] if isinstance(ev, str) else [str(x) for x in ev]
        good, lines = _ground(doc, ev)
        rel = [str(x).strip() for x in (c.get("requirements") or []) if str(x).strip() in rids]
        if len(good) < 1 or len(rel) < 2:
            stats["rejected"] += 1
            continue
        dup = next((x for x in set_llm_issues if x.characteristic == "set_consistent" and len(set(x.related) & set(rel)) >= 2), None)
        # противоречие уже отмечено у отдельного требования (характеристика «Корректность»)?
        if dup:
            continue
        set_llm_issues.append(Issue(characteristic="set_consistent", severity=_norm_sev(c.get("severity")),
                                    problem=str(c.get("problem", "")).strip()[:700], evidence=good, lines=lines, related=rel,
                                    fix=str(c.get("fix", "")).strip()[:600], source="llm"))
    # правило-противоречия, которые LLM уже описал, склеиваются по пересечению требований
    merged_set = list(set_llm_issues)
    for r in set_issues_rules:
        dup = next((x for x in merged_set if x.characteristic == r.characteristic and set(x.related) & set(r.related)
                    and (set(x.lines) & set(r.lines) or not r.lines)), None)
        if dup:
            dup.source = "rules+llm"
        else:
            merged_set.append(r)
    sr.issues = sorted(merged_set, key=lambda i: (SEV_RANK.get(i.severity, 1), i.characteristic))
    llm_set_st = {c: _norm_status(v) for c, v in (raw_set.statuses.items() if raw_set else []) if c in SET_CODES}
    sr.statuses = _statuses_from(sr.issues, SET_CODES, llm_set_st or None, set())
    for c, v in sr.statuses.items():
        if v == "issue" and not any(i.characteristic == c for i in sr.issues) and not sr.comments.get(c):
            sr.statuses[c] = "warn"
    sr.checklist = _complete_checklist(seg, sr)
    if len(sr.questions) < 6:
        sr.questions += _fallback_questions(unit_reviews, sr, 8 - len(sr.questions))
    emit("verifier", "done", f"отклонено неподтверждённых претензий: {stats['rejected']}")
    stats["duration"] = round(time.perf_counter() - t0, 2)
    stats["model"] = model_used
    return seg, unit_reviews, sr, stats


_CHECKLIST_RULES = [   # (ключ пункта чек-листа, характеристика набора, по которой отвечает агент)
    ("протестирован", "set_verifiable"), ("противореч", "set_consistent"), ("версион", "set_modifiable"),
    ("покрывает бизнес", "set_complete"), ("структурирован", "set_comprehensible"), ("согласован", "set_traceable"),
    ("интеграцион", "set_complete"), ("модель данных", "set_complete"), ("бизнес-процесс", "set_comprehensible"),
]


def _complete_checklist(seg: Segmentation, sr: SetReview) -> list[dict]:
    """Каждый пункт чек-листа документа получает ответ: от LLM или — если его нет — по оценкам набора."""
    from difflib import SequenceMatcher

    if not seg.checklist:
        return sr.checklist
    out = []
    llm = list(sr.checklist)
    for row in seg.checklist:
        item = row["item"]
        best = max(llm, key=lambda c: SequenceMatcher(None, item.lower(), str(c.get("item", "")).lower()).ratio(), default=None)
        if best and SequenceMatcher(None, item.lower(), str(best.get("item", "")).lower()).ratio() >= 0.55:
            out.append({"item": item, "answer": best.get("answer", ""), "comment": best.get("comment", ""), "source": "LLM"})
            llm.remove(best)
            continue
        code = next((c for k, c in _CHECKLIST_RULES if k in item.lower()), None)
        st = sr.statuses.get(code) if code else None
        ans = {"ok": "Да", "issue": "Нет", "warn": "Частично"}.get(st or "", "Требует экспертной оценки")
        rel = [i for i in sr.issues if i.characteristic == code][:1]
        comment = (rel[0].problem[:200] if rel else BY_CODE[code].name + ": " + {"ok": "нарушений не выявлено", "issue": "есть нарушения",
                   "warn": "есть замечания"}.get(st or "", "")) if code else "Пункт требует решения ответственного лица"
        out.append({"item": item, "answer": ans, "comment": comment, "source": "правила"})
    return out


def _fallback_questions(units: list[UnitReview], sr: SetReview, n: int) -> list[dict]:
    pool = [(i, ", ".join(i.related) or "набор") for i in sr.issues] + [(i, u.rid) for u in units for i in u.issues]
    pool.sort(key=lambda x: SEV_RANK.get(x[0].severity, 1))
    out, seen = [], set()
    for i, ref in pool:
        if i.severity == "minor" or i.problem[:50] in seen:
            continue
        seen.add(i.problem[:50])
        opts = [i.fix] if i.fix else []
        opts.append("Оставить как есть и зафиксировать обоснование в документе")
        out.append({"question": f"{ref}: {i.problem.rstrip('.')} — какое решение принимаем?", "related": i.related or [ref],
                    "options": opts, "addressee": "Аналитик / владелец продукта"})
        if len(out) >= n:
            break
    return out


# ── итог ──────────────────────────────────────────────────────────────────────

def _verdict(units: list[UnitReview], sr: SetReview) -> tuple[str, str, dict[str, list[str]], list[str], dict]:
    all_issues = [(u.rid, u.doc_ref, i) for u in units for i in u.issues] + [("набор", ", ".join(i.related), i) for i in sr.issues]
    sev = {s: sum(1 for *_, i in all_issues if i.severity == s) for s in ("critical", "major", "minor")}
    critical: dict[str, list[str]] = {"непроверяемые": [], "противоречивые": [], "неполные": []}
    for rid, ref, i in all_issues:
        cls = CRITICAL_CLASSES.get(i.characteristic)
        if cls == "противоречивые" and i.characteristic == "correct" and not i.related:
            continue
        if cls and i.severity in ("critical", "major"):
            label = f"{rid} ({ref})" if rid != "набор" else (f"набор: {ref}" if ref else "набор")
            critical[cls].append(f"{label}: {i.problem}")
    n_crit = sev["critical"]
    with_issues = sum(1 for u in units if any(i.severity in ("critical", "major") for i in u.issues))
    if n_crit or critical["противоречивые"]:
        level, verdict = "not_ready", (
            f"Нет. Начинать разработку и тестирование по документу нельзя: критичных нарушений — {n_crit}; "
            f"противоречий — {len(critical['противоречивые'])}, случаев неполноты — {len(critical['неполные'])}, "
            f"непроверяемых формулировок — {len(critical['непроверяемые'])}. "
            f"Требований с существенными замечаниями — {with_issues} из {len(units)}.")
    elif sev["major"]:
        level, verdict = "conditional", (
            f"Условно. Разработку можно начинать после исправления {sev['major']} важных замечаний; критичных дефектов нет.")
    else:
        level, verdict = "ready", "Да. Документ пригоден для начала разработки и тестирования; замечания носят редакционный характер."
    # приоритеты: противоречия → неполнота → непроверяемость → ИБ → прочее
    order = {"set_consistent": 0, "correct": 1, "set_complete": 2, "complete": 3, "verifiable": 4, "set_verifiable": 5, "security": 6}
    ranked = sorted(all_issues, key=lambda x: (SEV_RANK.get(x[2].severity, 1), order.get(x[2].characteristic, 9)))
    priorities, seen = [], set()
    for rid, ref, i in ranked:
        key = (i.problem[:60])
        if key in seen:
            continue
        seen.add(key)
        where = f"{rid} ({ref})" if rid != "набор" else (", ".join(i.related) or "набор")
        priorities.append(f"{where} — {i.problem} → {i.fix}" if i.fix else f"{where} — {i.problem}")
        if len(priorities) >= 8:
            break
    by_char: dict[str, int] = {}
    for *_, i in all_issues:
        by_char[i.characteristic] = by_char.get(i.characteristic, 0) + 1
    stats = {"requirements": len(units), "with_issues": with_issues, "issues_total": len(all_issues), **sev,
             "by_characteristic": by_char, "unit_issues": sum(len(u.issues) for u in units), "set_issues": len(sr.issues)}
    return verdict, level, critical, priorities, stats


def agent_version() -> str:
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True,
                               timeout=5).stdout.strip()
        tag = subprocess.run(["git", "describe", "--tags", "--abbrev=0"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip()
        if commit:
            return f"{tag + ' · ' if tag else ''}коммит {commit}{' (+ локальные изменения)' if dirty else ''}"
    except Exception:
        pass
    return "версия не определена (нет git)"


def review_document(doc: Document, cfg: LLMConfig | None = None, use_llm: bool = True, on_event=None) -> IsoReport:
    from datetime import datetime, timedelta, timezone

    cfg = cfg or LLMConfig.from_env()
    seg, units, sr, stats = asyncio.run(_run(doc, cfg, use_llm, on_event))
    verdict, level, critical, priorities, st = _verdict(units, sr)
    if sr.top_fixes:
        priorities = sr.top_fixes[:8]
    mode = "LLM + правила" if (use_llm and cfg.enabled and stats["llm_calls"] > stats["llm_failed"]) else "правила (офлайн)"
    if use_llm and cfg.enabled and 0 < stats["llm_failed"] < stats["llm_calls"]:
        mode += f" (часть пакетов без LLM: {stats['llm_failed']} из {stats['llm_calls']})"
    almaty = timezone(timedelta(hours=5))
    return IsoReport(
        document_name=doc.name, document_title=seg.title, run_at=datetime.now(almaty).strftime("%Y-%m-%d %H:%M (UTC+5)"),
        model=stats["model"], agent_version=agent_version(), mode=mode, duration_s=stats["duration"], units=units,
        set_review=sr, verdict=verdict, verdict_level=level, priorities=priorities, critical=critical,
        stats={**st, "tokens": stats["tokens"], "rejected": stats["rejected"], "llm_calls": stats["llm_calls"],
               "llm_failed": stats["llm_failed"], "notes": stats["notes"], "checklist_rows": len(seg.checklist)},
        trace=[{"agent": "Segmenter", "result": f"{len(seg.units)} требований"},
               {"agent": "Rule Checker", "result": f"{sum(1 for u in units for i in u.issues if 'rules' in i.source)} сигналов по требованиям"},
               {"agent": "Requirement Critic (LLM)", "result": f"{sum(1 for u in units if u.mode == 'llm')} из {len(units)} требований"},
               {"agent": "Set Critic (LLM)", "result": "выполнено" if sr.mode == "llm" else "не выполнено — использованы правила"},
               {"agent": "Contradiction Hunter (LLM)", "result": f"противоречий в наборе: "
                                                                 f"{sum(1 for i in sr.issues if i.characteristic == 'set_consistent')}"},
               {"agent": "Grounding Verifier", "result": f"отклонено неподтверждённых претензий: {stats['rejected']}"}],
    )
