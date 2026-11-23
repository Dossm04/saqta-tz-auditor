"""Question Generator — превращает проверенные находки в повестку встречи с автором ТЗ.

Гарантии (в обоих режимах):
  * каждая критичная находка, каждое противоречие, каждое опасное требование (AP-*) и каждый
    открытый вопрос (TBD) покрыты вопросом — под них резервируются места сверх лимита встречи;
  * вопрос по противоречию — закрытый: варианты ответа = конфликтующие значения с номерами строк;
  * вопросы без привязки к находке отбрасываются, дубли склеиваются;
  * каждый вопрос проходит детерминированный линтер (чеклист качества), в LLM-режиме
    провалившие чеклист вопросы один раз отправляются модели на исправление;
  * варианты ответа, допущение по умолчанию и последствие берутся из базы знаний, поэтому и
    офлайн-вопросы закрытые.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, ValidationError

from agents.base import Agent, Context, load_prompt
from knowledge.antipatterns import ANTIPATTERNS
from knowledge.fintech_controls import CONTROLS_BY_ID
from knowledge.tz_structure import SECTIONS
from models import Category, Finding, Priority, Question, QuestionQuality, Severity
from parsing import Document

_SECTIONS = {s.id: s for s in SECTIONS}
_APS = {a.id: a for a in ANTIPATTERNS}
_PRIO = {Severity.critical: Priority.P0, Severity.major: Priority.P1, Severity.minor: Priority.P2}
_ORDER = {Priority.P0: 0, Priority.P1: 1, Priority.P2: 2}
_CAT_ADDRESSEE = {
    Category.compliance: "Комплаенс", Category.security: "ИБ", Category.fintech: "Архитектор",
    Category.contradiction: "Бизнес-заказчик", Category.ambiguity: "Бизнес-заказчик",
    Category.testability: "Бизнес-заказчик / QA", Category.completeness: "Бизнес-заказчик",
}
_IMPACT = {
    Category.compliance: "Риск предписания или штрафа регулятора и задержки запуска",
    Category.security: "Риск утечки данных или кражи денег клиентов, инцидент ИБ",
    Category.fintech: "Риск финансовых потерь: двойные списания, расхождения сверки, мошеннические операции",
    Category.contradiction: "Разработка и QA реализуют разные значения — дефект в проде и спор при приёмке",
    Category.completeness: "Разработка начнётся с допущений — переделки и сдвиг сроков",
    Category.testability: "Требование нельзя проверить при приёмке — спор с заказчиком",
    Category.ambiguity: "Разработка и QA трактуют требование по-разному — переделки и спор при приёмке",
}
LIMITS = {Priority.P0: 10, Priority.P1: 8, Priority.P2: 4}   # вопросы на одну встречу; сверх — «дополнительные»
MINUTES = {Priority.P0: 5, Priority.P1: 3, Priority.P2: 1}   # бюджет времени встречи на вопрос

# Варианты для размытых формулировок: семейства терминов → типовые измеримые уровни (как предложение, не норма)
_VAGUE_OPTIONS: list[tuple[re.Pattern, list[str], str]] = [
    (re.compile(r"быстр|мгновенн|моментальн|задерж|скорост|производительн", re.I),
     ["p95 ≤ 1 с при пиковой нагрузке (например, 200 RPS)", "p95 ≤ 3 с при пиковой нагрузке",
      "Асинхронно: подтверждение приёма ≤ 1 с, итоговый статус ≤ 30 с с push-уведомлением"],
     "p95 ≤ 3 с при пиковой нагрузке; значение подтверждается нагрузочным тестом"),
    (re.compile(r"надежн|стабильн|бесперебойн|доступност", re.I),
     ["SLA 99,9% в месяц (≈ 43 мин простоя), RTO 1 ч, RPO 5 мин", "SLA 99,5% в месяц (≈ 3,6 ч простоя), RTO 4 ч, RPO 1 ч",
      "24/7 с регламентным окном (например, ночью 02:00–04:00 по Астане)"],
     "SLA 99,9% в месяц, RTO 1 ч, RPO 5 мин"),
    (re.compile(r"удобн|интуитивн|понятн|user-?friendly|прост", re.I),
     ["Ключевой сценарий — не более 3 шагов, SUS ≥ 70 на юзабилити-тесте (5 клиентов)",
      "Соответствие дизайн-системе банка + юзабилити-тест до релиза",
      "Метрика воронки: ≥ 80% клиентов завершают сценарий без обращения в поддержку"],
     "Ключевой сценарий — не более 3 шагов, SUS ≥ 70"),
    (re.compile(r"больш|много|множеств|массов|нагрузк", re.I),
     ["До 100 000 активных клиентов и 200 RPS в пике", "До 1 000 000 активных клиентов и 1 000 RPS в пике",
      "Горизонтальное масштабирование: x3 к пиковой нагрузке прошлого года"],
     "Объём на 12 месяцев вперёд с запасом x2; значение подтверждается заказчиком"),
    (re.compile(r"своевременн|кратчайш|оперативн|asap", re.I),
     ["Не позднее 15 минут", "В течение 1 рабочего часа", "До конца операционного дня (по времени Астаны, UTC+5)"],
     "Не позднее 15 минут; для ручных процессов — в течение 1 рабочего часа"),
    (re.compile(r"безопасн|защищ", re.I),
     ["TLS 1.2+ при передаче и AES-256 с ключами в HSM/KMS при хранении", "Токенизация чувствительных данных + маскирование в UI и логах",
      "Требования ИБ банка (внутренний стандарт) — со ссылкой на пункты"],
     "TLS 1.2+ при передаче, AES-256 при хранении, ключи в HSM/KMS"),
    (re.compile(r"возможност|необходимост|желательн|рекомендуется|может быть", re.I),
     ["Обязательное требование MVP (Must)", "Желательное (Should) — реализуется, если укладывается в спринт",
      "Убрать из ТЗ текущего релиза (Won't)"],
     "Желательное (Should); не блокирует приёмку MVP"),
]

# противоречия: число + единица; ID требований (FR-01, 3.2.1) не считаются значениями
_VALUE = re.compile(
    r"(?<![\w.,-])(\d{1,3}(?:[  ]\d{3})+|\d+(?:[.,]\d+)?)\s*"
    r"(%|₸|тенге|тг\b|kzt|мрп|руб\w*|usd|\$|мс\b|ms\b|миллисек\w*|сек\w*|с\b|мин\w*|час\w*|ч\b|дн\w*|суток|rps|tps)?", re.I)
_LEAD_ID = re.compile(r"^\s*(?:[-*•|]\s*)?(?:#+\s*)?(?:(?:[A-ZА-ЯЁ]{1,5}[-.]?\d+(?:\.\d+)*|\d+(?:\.\d+)*)[.):]?\s+)?")
_MONEY = re.compile(r"₸|тенге|\bтг\b|kzt|мрп|лимит|сумм|комисси|остат", re.I)

_BAD_OPT = re.compile(r"^\s*(да|нет|не знаю|другое|иное|прочее|затрудняюсь\w*( ответить)?|на усмотрение\b.*|"
                      r"по согласованию|tbd|n/?a|без изменений|оставить как есть)\s*[.!]?\s*$", re.I)
_GENERIC = re.compile(r"^\s*(уточните|опишите|предоставьте|расскажите|поясните|дайте)\s+(требования|детали|"
                      r"информацию|подробнее|подробности|больше)\b|^\s*как должно быть описано требование", re.I)


class RawQuestions(BaseModel):
    questions: list[dict[str, Any]] = []


# ── база знаний ───────────────────────────────────────────────────────────────

@dataclass
class KBMeta:
    why: str = ""
    addressee: str = ""
    options: list[str] = field(default_factory=list)
    default_assumption: str = ""
    impact: str = ""
    recommended: int | None = None


def kb_meta(f: Finding) -> KBMeta:
    """Метаданные вопроса из базы знаний по control_id (контроль, раздел или опасное требование).
    Поля читаются через getattr: у старых записей базы знаний их может не быть."""
    cid = f.control_id or ""
    obj = CONTROLS_BY_ID.get(cid) or _SECTIONS.get(cid) or _APS.get(cid)
    if obj is None:
        return KBMeta()
    rec = getattr(obj, "recommended_option", None)
    return KBMeta(
        why=str(getattr(obj, "why", "") or getattr(obj, "description", "") or ""),
        addressee=str(getattr(obj, "addressee", "") or ("ИБ / Комплаенс" if cid in _APS else "")),
        options=[str(o) for o in (getattr(obj, "answer_options", None) or ()) if str(o).strip()],
        default_assumption=str(getattr(obj, "default_assumption", "") or ""),
        impact=str(getattr(obj, "impact", "") or ""),
        recommended=rec if isinstance(rec, int) and not isinstance(rec, bool) else None,
    )


# ── классификация находок ─────────────────────────────────────────────────────

def is_tbd(f: Finding) -> bool:
    t = f.title.lower()
    return t.startswith("открытый вопрос") or "tbd" in t or "уточнить позже" in t


def is_duplicate(f: Finding) -> bool:
    return f.category == Category.contradiction and f.title.lower().startswith("дублир")


def is_dangerous(f: Finding) -> bool:
    return (f.control_id or "").upper().startswith("AP-")


def must_cover(f: Finding) -> bool:
    """Находки, по которым вопрос обязателен независимо от лимитов встречи."""
    return (f.severity == Severity.critical or f.category == Category.contradiction
            or is_dangerous(f) or is_tbd(f))


def _is_minor_vague(f: Finding) -> bool:
    return (f.category in (Category.ambiguity, Category.testability) and f.severity == Severity.minor
            and not f.control_id and not is_tbd(f))


def _clip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s.strip().lstrip("#-*•| ").strip())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _similar(a: str, b: str, threshold: float = 0.88) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


# ── линтер вопросов ───────────────────────────────────────────────────────────

def _same_option(a: str, b: str) -> bool:
    """Варианты совпадают по смыслу: одинаковы после нормализации или почти одинаковы и с теми же числами
    (варианты «Верно 500 000 ₸ (L42)» и «Верно 1 000 000 ₸ (L73)» — разные)."""
    na, nb = (re.sub(r"[^\w]+", " ", x.lower()).strip() for x in (a, b))
    if na == nb:
        return True
    return re.findall(r"\d+", na) == re.findall(r"\d+", nb) and _similar(na, nb, 0.9)


def substantive_options(q: Question) -> list[str]:
    out: list[str] = []
    for o in q.answer_options:
        o = o.strip()
        if len(o) < 3 or _BAD_OPT.match(o):
            continue
        if any(_same_option(o, x) for x in out):
            continue
        out.append(o)
    return out


def lint_question(q: Question) -> list[str]:
    """Детерминированный чеклист качества вопроса. Пустой список — вопрос прошёл."""
    errs = []
    text = q.question.strip()
    if not text.endswith("?"):
        errs.append("не заканчивается «?»")
    if len(text) > 400:
        errs.append("длиннее 400 символов")
    if _GENERIC.search(text):
        errs.append("общий вопрос без привязки к ТЗ")
    if re.sub(r"«[^»]*»|\"[^\"]*\"", "", text).count("?") > 1:
        errs.append("несколько вопросов в одном")
    opts = substantive_options(q)
    if q.priority in (Priority.P0, Priority.P1) and len(opts) < 2:
        errs.append("меньше 2 содержательных вариантов ответа")
    if len(opts) < len([o for o in q.answer_options if o.strip() and not _BAD_OPT.match(o.strip())]):
        errs.append("варианты почти совпадают")
    if not q.context.strip() and not q.related_findings:
        errs.append("нет цитаты ТЗ и связанной находки")
    return errs


# ── построение вопросов правилами ─────────────────────────────────────────────

def _values(line: str) -> list[tuple[str, float, bool]]:
    """[(как в тексте, число, есть ли единица)] — значения в строке без ID требования в начале."""
    body = _LEAD_ID.sub("", line, count=1)
    out = []
    for m in _VALUE.finditer(body):
        raw, unit = m.group(1), (m.group(2) or "")
        try:
            num = float(re.sub(r"[  ]", "", raw).replace(",", "."))
        except ValueError:
            continue
        shown = re.sub(r" ", " ", raw) + (("" if unit == "%" else " ") + unit if unit else "")
        out.append((shown.strip(), num, bool(unit)))
    return out


def _topic(f: Finding) -> str:
    t = re.sub(r"^(противоречивые значения|противоречие)\s*:\s*", "", f.title.strip(), flags=re.I)
    return _clip(t, 80)


def contradiction_question(f: Finding, doc: Document | None, kb: KBMeta | None = None) -> Question:
    """Закрытый вопрос по противоречию: варианты ответа — конфликтующие значения (или формулировки) с L<n>."""
    lines = [n for n in dict.fromkeys(f.lines) if doc is None or 1 <= n <= len(doc.lines)][:4]
    texts = [doc.line(n) if doc is not None else "" for n in lines]
    if len(lines) < 2 and len(f.evidence) >= 2:          # LLM дал цитаты без номеров строк
        texts, lines = f.evidence[:4], [0] * len(f.evidence[:4])
    per_line = [_values(t) for t in texts]
    picks: list[tuple[str, int]] = []
    for idx, vals in enumerate(per_line):
        others = [{v[1] for v in pv} for j, pv in enumerate(per_line) if j != idx]
        distinct = [v for v in vals if others and not all(v[1] in o for o in others)]
        cand = [v for v in distinct if v[2]] or distinct
        if cand:
            picks.append((cand[0][0], lines[idx]))
    seen_vals = {p[0] for p in picks}
    loc = lambda n: f" (L{n})" if n else ""
    topic = _topic(f)
    if len(picks) >= 2 and len(seen_vals) >= 2:
        shown = [f"{v}{loc(n)}" for v, n in picks]
        question = f"Какое значение верное для «{topic}»: {' или '.join(shown)}?"
        options = [f"Верно {v}{loc(n)} — остальные места ТЗ привести к этому значению" for v, n in picks]
    else:
        shown = [f"«{_clip(t, 70)}»{loc(n)}" for t, n in zip(texts, lines)]
        question = f"Какое из требований действует: {' или '.join(shown[:3])}?"
        options = [f"Действует{loc(n) or ''} «{_clip(t, 70)}» — второе требование исправить" for t, n in zip(texts, lines)][:3]
    options.append("Оба верны для разных условий — явно разграничить их в ТЗ (по каналу, сумме, роли или уровню идентификации)")
    money = bool(_MONEY.search(" ".join(texts) + " " + f.title))
    prio = Priority.P0 if f.severity == Severity.critical or money else _PRIO[f.severity]
    return Question(
        question=question,
        why=f"В ТЗ одно и то же задано по-разному ({', '.join(shown[:3])}).",
        impact=(kb.impact if kb and kb.impact else _IMPACT[Category.contradiction]),
        priority=prio, addressee="Бизнес-заказчик", related_findings=[f.id], answer_options=options,
        blocking=prio == Priority.P0,
        default_assumption="До решения заказчика в разработку берётся более строгое (консервативное) значение; "
                           "оба места ТЗ помечаются как открытые.",
    )


def _vague_question(f: Finding) -> Question:
    term = f.title.split(":", 1)[-1].strip() or "размытая формулировка"
    line = f"L{f.lines[0]}" if f.lines else "ТЗ"
    options, default = [], ""
    for rx, opts, dflt in _VAGUE_OPTIONS:
        if rx.search(term.replace("ё", "е")):
            options, default = list(opts), dflt
            break
    if not options:
        options = ["Заменить формулировку числовым критерием приёмки (значение утверждает заказчик)",
                   "Удалить формулировку — она не проверяется при приёмке"]
        default = "Формулировка не считается требованием, пока не задан числовой критерий приёмки"
    prio = _PRIO[f.severity]
    return Question(
        question=f"Какое измеримое значение закладываем вместо {term} в {line}?",
        why=(f.description or "Размытую формулировку разработка и QA трактуют по-разному.")[:260],
        impact=_IMPACT.get(f.category, _IMPACT[Category.ambiguity]), priority=prio,
        addressee="Бизнес-заказчик", related_findings=[f.id], answer_options=options,
        recommended_option=None, default_assumption=default, blocking=prio == Priority.P0,
    )


def _tbd_question(f: Finding) -> Question:
    line = f"L{f.lines[0]}" if f.lines else "ТЗ"
    quote = f"«{_clip(f.evidence[0], 90)}»" if f.evidence else ""
    return Question(
        question=f"Какое решение принято по открытому вопросу в {line} {quote}?".replace("  ", " ").replace(" ?", "?"),
        why="Требование помечено как неопределённое — разработка по нему начнётся с допущений.",
        impact=_IMPACT[Category.completeness], priority=Priority.P1 if f.severity != Severity.critical else Priority.P0,
        addressee="Бизнес-заказчик", related_findings=[f.id],
        answer_options=["Зафиксировать решение в ТЗ до старта спринта (ответственный — автор ТЗ)",
                        "Вынести функцию из MVP в следующий релиз",
                        "Реализовать по допущению команды и подтвердить на демо"],
        recommended_option=0,
        default_assumption="Функция не берётся в спринт, пока решение не внесено в ТЗ; в плане — как риск сроков.",
    )


def _recommended(options: list[str], kb: KBMeta, default: str) -> int | None:
    if kb.recommended is not None and 0 <= kb.recommended < len(options):
        return kb.recommended
    for i, o in enumerate(options):
        if "рекоменд" in o.lower() or o.startswith(("Принять предложенное:", "Убрать требование и сделать безопасно")):
            return i
    if default and options:
        scores = [SequenceMatcher(None, default.lower(), o.lower()).ratio() for o in options]
        best = max(range(len(options)), key=lambda i: scores[i])
        if scores[best] >= 0.45:
            return best
    return None


def _decision_options(f: Finding) -> list[str]:
    """Закрытые варианты-решения, когда в базе знаний нет своих: принять предложенную формулировку /
    закрыть существующим решением банка / осознанно отложить. Без формулировки — вариантов нет."""
    proposal = f.suggested_text or f.recommendation
    if not proposal:
        return []
    if is_dangerous(f):
        return [f"Убрать требование и сделать безопасно: «{_clip(proposal, 150)}»",
                "Оставить только с компенсирующими мерами и письменным согласованием ИБ и комплаенса",
                "Исключить функцию из MVP до решения ИБ"]
    defer = ("Отложить до промышленного запуска (только пилот на сотрудниках) — по согласованию с комплаенсом"
             if f.category == Category.compliance else
             "Не делать в этом релизе — риск принимает владелец продукта, решение фиксируется в ТЗ")
    return [f"Принять предложенное: «{_clip(proposal, 150)}»",
            "Закрыть существующим решением или стандартом банка — указать систему и документ", defer]


def build_question(f: Finding, doc: Document | None = None) -> Question:
    """Вопрос по одной находке из правил и базы знаний (офлайн и гарантия покрытия в LLM-режиме)."""
    kb = kb_meta(f)
    if f.category == Category.contradiction and not is_duplicate(f):
        return contradiction_question(f, doc, kb)
    if is_tbd(f):
        return _tbd_question(f)
    if (f.category in (Category.ambiguity, Category.testability) and not f.control_id
            and f.title.lower().startswith("размыт")):
        return _vague_question(f)
    options = list(kb.options) or _decision_options(f)
    text = (f.question or "").strip()
    if not text or _GENERIC.search(text):
        text = (f"Какое решение принимаем по теме «{_clip(f.title, 90)}»?" if options else
                f"Что именно должно быть зафиксировано в ТЗ по теме «{_clip(f.title, 90)}»: значение, ответственный и срок?")
    if f.evidence and f.lines and not re.search(r"\bL\d+", text):
        text = f"В L{f.lines[0]} указано «{_clip(f.evidence[0], 90)}». {text}"
    default = kb.default_assumption or (_clip(f.suggested_text, 220) if f.suggested_text else "")
    impact = kb.impact or _IMPACT.get(f.category, "")
    if f.regulation and f.category == Category.compliance and not kb.impact:
        impact += f" ({_clip(f.regulation, 90)})"
    prio = _PRIO[f.severity]
    if is_dangerous(f) and prio == Priority.P2:
        prio = Priority.P1
    return Question(
        question=text, why=_clip(kb.why or f.description or f.title, 260), impact=impact, priority=prio,
        addressee=kb.addressee or _CAT_ADDRESSEE.get(f.category, "Бизнес-заказчик"),
        related_findings=[f.id], answer_options=options,
        recommended_option=_recommended(options, kb, default),
        default_assumption=default, blocking=f.severity == Severity.critical,
    )


def _batch_vague(findings: list[Finding]) -> Question | None:
    if not findings:
        return None
    terms = "; ".join(f"{f.title.split(':', 1)[-1].strip()} (L{f.lines[0]})" for f in findings[:8] if f.lines)
    return Question(
        question=f"Утверждаете ли замену размытых формулировок измеримыми значениями: {terms}?",
        why="Размытые формулировки разработчики и тестировщики трактуют по-разному — споры при приёмке.",
        impact=_IMPACT[Category.ambiguity], priority=Priority.P2, addressee="Бизнес-заказчик",
        related_findings=[f.id for f in findings],
        answer_options=["Да, заменить каждую формулировку числовым критерием (значения предложит аналитик)",
                        "Удалить формулировки, которые не проверяются при приёмке"],
        recommended_option=0,
        default_assumption="Аналитик подставляет измеримые значения и выносит их на подтверждение в протоколе.",
    )


def _batch_duplicates(findings: list[Finding]) -> Question | None:
    if not findings:
        return None
    pairs = "; ".join("/".join(f"L{n}" for n in f.lines[:2]) for f in findings[:8])
    return Question(
        question=f"Какую из дублирующихся формулировок оставить единственным источником ({pairs})?",
        why="Дубли расходятся при правках и порождают противоречия.",
        impact=_IMPACT[Category.contradiction], priority=Priority.P2, addressee="Бизнес-заказчик",
        related_findings=[f.id for f in findings],
        answer_options=["Оставить первое упоминание, во втором — ссылка по ID требования",
                        "Объединить дубли в одно требование с новым ID и обновить ссылки"],
        recommended_option=0,
        default_assumption="Остаётся первое упоминание; повторы заменяются ссылкой на ID.",
    )


# ── сборка, гарантии, метрики ─────────────────────────────────────────────────

def backfill(q: Question, by_id: dict[str, Finding], doc: Document | None = None) -> Question:
    """Дополняет вопрос LLM данными из базы знаний и находок: варианты, допущение, последствие."""
    related = [by_id[r] for r in q.related_findings if r in by_id]
    if len(substantive_options(q)) < 2:
        contra = next((f for f in related if f.category == Category.contradiction and not is_duplicate(f)), None)
        kb_opts = next((kb_meta(f).options for f in related if len(kb_meta(f).options) >= 2), [])
        if contra is not None:
            cq = contradiction_question(contra, doc)
            q.answer_options = cq.answer_options
            q.default_assumption = q.default_assumption or cq.default_assumption
        elif kb_opts:
            q.answer_options = list(kb_opts)
        else:
            fallback = next((o for o in (_decision_options(f) for f in related) if o), [])
            if fallback:
                q.answer_options = fallback
                q.recommended_option = 0
    for f in related:
        kb = kb_meta(f)
        if not q.default_assumption:
            q.default_assumption = kb.default_assumption or (
                _clip(f.suggested_text, 220) if f.suggested_text else "")
        if not q.impact:
            q.impact = kb.impact or _IMPACT.get(f.category, "")
        if not q.why:
            q.why = _clip(kb.why or f.description, 260)
    if q.recommended_option is not None and not 0 <= q.recommended_option < len(q.answer_options):
        q.recommended_option = None
    q.addressee = q.addressee.strip() or "Бизнес-заказчик"
    return q


def ensure_coverage(qs: list[Question], findings: list[Finding], doc: Document | None) -> list[Question]:
    """Каждая обязательная находка (critical, противоречие, AP-*, TBD) должна попасть в related_findings."""
    covered = {r for q in qs for r in q.related_findings}
    dups = [f for f in findings if is_duplicate(f) and f.id not in covered]
    for f in findings:
        if not must_cover(f) or f.id in covered or is_duplicate(f):
            continue
        q = build_question(f, doc)
        existing = next((x for x in qs if _similar(x.question, q.question)), None)
        if existing:
            existing.related_findings.append(f.id)
        else:
            qs.append(q)
        covered.add(f.id)
    batch = _batch_duplicates(dups)
    if batch:
        qs.append(batch)
    return qs


def finalize(qs: list[Question], findings: list[Finding]) -> tuple[list[Question], list[Question]]:
    """Контекст, линтер, лимиты встречи с резервом под обязательные находки, порядок повестки.
    Возвращает (вопросы встречи, дополнительные вопросы сверх лимита)."""
    by_id = {f.id: f for f in findings}
    must = {f.id for f in findings if must_cover(f)}
    for q in qs:
        q.related_findings = list(dict.fromkeys(q.related_findings))
        if not q.context:
            for fid in q.related_findings:
                f = by_id.get(fid)
                if f and f.evidence and f.lines:
                    q.context = f"L{f.lines[0]}: «{_clip(f.evidence[0], 200)}»"
                    break
        if q.priority == Priority.P0:
            q.blocking = True
        if q.recommended_option is not None and not 0 <= q.recommended_option < len(q.answer_options):
            q.recommended_option = None
        q.lint = lint_question(q)

    def first_line(q: Question) -> int:
        ls = [n for r in q.related_findings if r in by_id for n in by_id[r].lines]
        return min(ls) if ls else 10 ** 6

    # отбор: сначала по приоритету; вопрос, закрывающий ещё не покрытую обязательную находку, берётся сверх лимита
    pool = sorted(qs, key=lambda q: (_ORDER[q.priority], not q.blocking, -len(must & set(q.related_findings)),
                                     bool(q.lint), first_line(q)))
    counts = {p: 0 for p in Priority}
    covered_must: set[str] = set()
    main, extra = [], []
    for q in pool:
        new_must = (must & set(q.related_findings)) - covered_must
        if counts[q.priority] >= LIMITS[q.priority] and not new_must:
            extra.append(q)
            continue
        counts[q.priority] += 1
        covered_must |= must & set(q.related_findings)
        main.append(q)
    # повестка встречи: приоритет → адресат (решения одного стейкхолдера подряд) → порядок по тексту ТЗ
    main.sort(key=lambda q: (_ORDER[q.priority], q.addressee, first_line(q)))
    extra.sort(key=lambda q: (_ORDER[q.priority], q.addressee, first_line(q)))
    for i, q in enumerate(main + extra, start=1):
        q.id = f"Q-{i:02d}"
    return main, extra


def question_quality(main: list[Question], extra: list[Question], findings: list[Finding], repaired: int = 0) -> QuestionQuality:
    n = len(main)
    by_id = {f.id: f for f in findings}
    share = lambda k: round(k / n, 2) if n else 0.0
    covered = {r for q in main for r in q.related_findings}
    crit = [f.id for f in findings if f.severity == Severity.critical]
    must = [f.id for f in findings if must_cover(f)]
    p01 = [q for q in main if q.priority in (Priority.P0, Priority.P1)]
    grounded = sum(1 for q in main if q.context or any(by_id.get(r) and by_id[r].lines for r in q.related_findings))
    return QuestionQuality(
        total=n, lint_passed=sum(1 for q in main if not q.lint),
        with_options=share(sum(1 for q in main if len(substantive_options(q)) >= 2)),
        closed_p0p1=round(sum(1 for q in p01 if len(substantive_options(q)) >= 2) / len(p01), 2) if p01 else 0.0,
        grounded=share(grounded), with_default=share(sum(1 for q in main if q.default_assumption.strip())),
        critical_covered=f"{sum(1 for c in crit if c in covered)}/{len(crit)}",
        must_cover_covered=f"{sum(1 for m in must if m in covered)}/{len(must)}",
        findings_covered=f"{sum(1 for f in findings if f.id in covered)}/{len(findings)}",
        overflow=len(extra), repaired=repaired, agenda_minutes=sum(MINUTES[q.priority] for q in main),
    )


def finding_cards(findings: list[Finding], doc: Document | None = None, limit: int = 70) -> str:
    """Полные карточки находок для LLM: цитата, черновик вопроса, варианты/адресат/допущение из базы знаний,
    норма, пометка Verifier. Мелкие размытые формулировки сведены в одну карточку."""
    rows = []
    minor = [f for f in findings if _is_minor_vague(f)]
    if minor:
        terms = "; ".join(f"{_clip(f.title.split(':', 1)[-1], 40)} (L{f.lines[0]})" for f in minor[:10] if f.lines)
        rows.append(f"[{','.join(f.id for f in minor)}] minor/ambiguity | Размытые формулировки: {terms} | "
                    f"задай ОДИН пакетный P2-вопрос «утвердите значения»")
    for f in [f for f in findings if not _is_minor_vague(f)][:limit]:
        kb = kb_meta(f)
        p = [f"[{f.id}] {f.severity.value}/{f.category.value} | {f.title}"]
        if f.evidence:
            p.append(f"цитата L{f.lines[0] if f.lines else '?'}: «{_clip(f.evidence[0], 160)}»")
        elif f.lines:
            p.append("строки: " + ", ".join(f"L{x}" for x in f.lines[:4]))
        else:
            p.append("в ТЗ отсутствует")
        if f.category == Category.contradiction and not is_duplicate(f):
            cq = contradiction_question(f, doc)
            p.append("конфликт: " + " / ".join(o.split(" — ")[0] for o in cq.answer_options[:-1]))
        if f.description:
            p.append(f"суть: {_clip(f.description, 160)}")
        if f.verification_note:
            p.append(f"пометка Verifier: {_clip(f.verification_note, 140)}")
        if f.question:
            p.append(f"черновик: {_clip(f.question, 200)}")
        if kb.options:
            p.append("варианты БЗ: " + " / ".join(_clip(o, 100) for o in kb.options[:4]))
        if kb.default_assumption:
            p.append(f"допущение БЗ: {_clip(kb.default_assumption, 140)}")
        if kb.why:
            p.append(f"почему важно: {_clip(kb.why, 140)}")
        if kb.addressee:
            p.append(f"адресат БЗ: {kb.addressee}")
        if f.regulation:
            p.append(f"норма: {_clip(f.regulation, 100)}")
        if f.suggested_text:
            p.append(f"предлагаемая формулировка: {_clip(f.suggested_text, 140)}")
        if must_cover(f):
            p.append("ОБЯЗАТЕЛЬНО покрыть вопросом")
        rows.append(" | ".join(p))
    return "\n".join(rows)


def _parse(items: list[Any], known: set[str]) -> list[tuple[Any, Question]]:
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        clean = {k: v for k, v in item.items() if k not in ("id", "lint", "context")}
        try:
            q = Question.model_validate(clean)
        except ValidationError:
            continue
        q.related_findings = [r for r in q.related_findings if r in known]
        if q.question.strip() and q.related_findings:      # вопрос без проверенной находки — выдумка
            out.append((item.get("ref"), q))
    return out


def dedupe(qs: list[Question]) -> list[Question]:
    merged: list[Question] = []
    for q in qs:
        dup = next((m for m in merged if _similar(m.question, q.question)
                    or (set(m.related_findings) == set(q.related_findings) and m.addressee == q.addressee)), None)
        if dup is None:
            merged.append(q)
            continue
        dup.related_findings = list(dict.fromkeys(dup.related_findings + q.related_findings))
        if len(substantive_options(dup)) < len(substantive_options(q)):
            dup.answer_options, dup.recommended_option = q.answer_options, q.recommended_option
        for attr in ("why", "impact", "default_assumption", "context"):
            if not getattr(dup, attr) and getattr(q, attr):
                setattr(dup, attr, getattr(q, attr))
        if _ORDER[q.priority] < _ORDER[dup.priority]:
            dup.priority = q.priority
        dup.blocking = dup.blocking or q.blocking
    return merged


class QuestionAgent(Agent):
    name = "questions"
    title = "Question Generator"
    role = "Формирует повестку встречи: закрытые вопросы с вариантами, допущением по умолчанию и адресатом"
    stage = 4

    # ── офлайн-режим ──
    def run_rules(self, ctx: Context) -> tuple[int, str]:
        findings = ctx.findings
        minor_vague = [f for f in findings if _is_minor_vague(f)]
        dups = [f for f in findings if is_duplicate(f)]
        skip = {id(f) for f in minor_vague + dups}
        qs: list[Question] = []
        for f in findings:
            if id(f) in skip:
                continue
            q = build_question(f, ctx.doc)
            existing = next((x for x in qs if _similar(x.question, q.question)), None)
            if existing:
                existing.related_findings.append(f.id)
                continue
            qs.append(q)
        for batch in (_batch_vague(minor_vague), _batch_duplicates(dups)):
            if batch:
                qs.append(batch)
        qs = [backfill(q, {f.id: f for f in findings}, ctx.doc) for q in qs]
        return self._store(ctx, qs, repaired=0, note=f"из {len(findings)} находок")

    # ── LLM-режим ──
    async def run_llm(self, ctx: Context) -> tuple[int, str]:
        if not ctx.findings:
            return self._store(ctx, [], 0, "нет находок")
        by_id = {f.id: f for f in ctx.findings}
        dm = ctx.doc_model
        cards = finding_cards(ctx.findings, ctx.doc)
        user = (f"## Контекст ТЗ\n{dm.title}: {dm.summary}\n"
                f"Участники: {', '.join(dm.actors) or '—'}\nИнтеграции: {', '.join(dm.integrations) or '—'}\n"
                f"Данные: {', '.join(dm.data_categories) or '—'}\n\n## Проверенные находки\n{cards}")
        system = load_prompt("questions")
        out = await self.ask(ctx, system, user, RawQuestions)
        qs = dedupe([q for _, q in _parse(out.questions, set(by_id))])
        qs = [backfill(q, by_id, ctx.doc) for q in qs]
        n_llm = len(qs)
        qs = ensure_coverage(qs, ctx.findings, ctx.doc)

        # линтер + один вызов на исправление провалившихся вопросов
        failing = [(i, q, lint_question(q)) for i, q in enumerate(qs) if lint_question(q)]
        repaired = 0
        if failing:
            rows = "\n".join(f"- ref {k}: {q.question} | варианты: {' / '.join(q.answer_options) or '—'} | "
                             f"находки: {', '.join(q.related_findings)} → нарушения: {', '.join(errs)}"
                             for k, (_, q, errs) in enumerate(failing, start=1))
            rel = {r for _, q, _ in failing for r in q.related_findings}
            fix_user = (f"Исправь ТОЛЬКО эти вопросы по чеклисту качества. Для каждого верни объект с полем `ref` "
                        f"(номер из списка) и всеми полями схемы; сохрани `related_findings`.\n\n## Вопросы с нарушениями\n"
                        f"{rows}\n\n## Карточки связанных находок\n"
                        f"{finding_cards([f for f in ctx.findings if f.id in rel], ctx.doc)}")
            try:
                fixed = await self.ask(ctx, system, fix_user, RawQuestions)
                for ref, q in _parse(fixed.questions, set(by_id)):
                    try:
                        k = int(str(ref).strip())
                    except (TypeError, ValueError):
                        continue
                    if not 1 <= k <= len(failing):
                        continue
                    i, old, errs = failing[k - 1]
                    q = backfill(q, by_id, ctx.doc)
                    if len(lint_question(q)) < len(errs):
                        q.related_findings = list(dict.fromkeys(old.related_findings + q.related_findings))
                        qs[i] = q
                        repaired += 1
            except Exception:   # исправление — бонус; без него остаются вопросы с пометками линтера
                pass
        return self._store(ctx, qs, repaired, f"от LLM {n_llm}, исправлено по чеклисту {repaired}")

    def _store(self, ctx: Context, qs: list[Question], repaired: int, note: str) -> tuple[int, str]:
        qs = ensure_coverage(qs, ctx.findings, ctx.doc)
        main, extra = finalize(qs, ctx.findings)
        ctx.questions = main
        quality = question_quality(main, extra, ctx.findings, repaired)
        ctx.report_extra["question_quality"] = quality
        ctx.report_extra["extra_questions"] = extra
        return len(main), (f"{note}; чеклист {quality.lint_passed}/{quality.total}, "
                           f"обязательные находки покрыты {quality.must_cover_covered}"
                           + (f", доп. вопросов {len(extra)}" if extra else ""))
