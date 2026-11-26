"""Domain Critic — проверки предметной области платёжных и банковских требований.

Стандарт ISO 29148 говорит, ЧТО проверять (однозначность, полнота, проверяемость), но не знает, где в финтехе
обычно прячутся дефекты. Этот агент добавляет доменный чек-лист:

  time        граница периода задана городом, а не часовым поясом; смешаны моменты операции (совершение/проведение);
  money       суммы в разных единицах (тиыны и тенге) без правила пересчёта;
  idempotency повторная доставка или повтор запроса без ключа идемпотентности;
  reversal    возврат/отмена меняет накопленную величину, но события, которые от неё зависят, не описаны;
  nfr         НФТ со сроком, но без условий измерения (перцентиль, нагрузка, точки отсчёта);
  retention   хранение данных без срока хранения и требований к защите;
  ui          ошибки ввода без текстов и правил валидации;
  precondition условие применимости без поведения вне условия (неактивная карта, другая валюта);
  acceptance  правила, не покрытые критериями приёмки.

Детерминированные правила работают всегда (офлайн), LLM-агент дополняет их тем, что правилами не поймать.
Каждая находка LLM проходит тот же Grounding Verifier, что и остальные.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from agents.base import load_prompt
from agents.segmenter import ReqUnit, Segmentation
from parsing import Document

AGENT = "domain"

TOPIC_KW = {
    "time": ("час", "пояс", "utc", "алмат", "провед", "совершен", "момент операц"),
    "money": ("тиын", "тенге", "единиц", "копе", "округл"),
    "idempotency": ("идемпот", "повторн", "дубл", "ключ"),
    "reversal": ("возврат", "отмен", "сторн"),
    "nfr": ("перцентил", "нагрузк", "измер", "отсчет", "отсчёт"),
    "retention": ("хранени", "удален", "срок"),
    "ui": ("текст", "сообщени", "валидац"),
    "precondition": ("неактивн", "заблок", "перевыпуск", "валют"),
    "acceptance": ("критери", "приемк", "приёмк"),
}

_TZ = re.compile(r"utc|gmt|asia/|часов\w* пояс|\+0?[3-6]:?00", re.I)
_PERIOD = re.compile(r"\bмесяц|календарн|\bсутк|\bдата\b|\bдате\b|\bдаты\b|\bза день|полноч|\b\d{1,2}:\d{2}\b", re.I)
# граница учётного периода (сброс счётчика/лимита), а не длительность («90 календарных дней»)
_BOUNDARY = re.compile(r"календарн\w*\s+месяц|месяц\s+определя|\bв\s+сутки|\bза\s+сутки|\bв\s+месяц\b|\bза\s+месяц\b|"
                       r"конц\w*\s+(?:текущего\s+)?(?:календарного\s+)?месяц|полноч", re.I)
_CITY = re.compile(r"\b(?:по|в)\s+(?:времени\s+)?(?:алмат|астан|нур-султан|москв|местн)\w*|по местному времени|по времени клиента", re.I)
_MOMENT_A = re.compile(r"совершени|совершен\w* покупк|авторизац", re.I)
_MOMENT_B = re.compile(r"проведен|проведённ|клиринг|списани|posted", re.I)
_MINOR_UNIT = re.compile(r"тиын|копе[ейк]|\bцент|minor unit", re.I)
_MAJOR_UNIT = re.compile(r"тенге|₸|\bkzt\b|рубл|доллар|\busd\b", re.I)
_RETRY = re.compile(r"повторн\w*\s+(?:доставк|отправк|запрос|вызов|получени|обработк)|дубл\w*|at-least-once|ретра\w*|retry|redeliver",
                    re.I)
_IDEM_KEY = re.compile(r"идемпотент|idempoten|ключ\w*\s+(?:дедупл|идемпот)|дедупликац\w*\s+по|по\s+идентификатору\s+(?:операци|событи)|"
                       r"(?:с\s+)?(?:тем\s+же|тот\s+же|одним\s+и\s+тем\s+же)\s+(?:идентификатор|ключ|номер)", re.I)
_MINOR_DEFINED = re.compile(r"целым\s+числом|минимальн\w*\s+единиц|1\s*₸\s*=\s*100|100\s+тиын|пересч[её]т\w*|при\s+отображении", re.I)
_PROTECTED = re.compile(r"маскир|шифр|токен", re.I)
_REVERSAL = re.compile(r"(?<!без)возврат|отмен\w*\s+(?:покупк|операц|платеж|платёж)|сторн|reversal|refund|chargeback", re.I)
_TRIGGER = re.compile(r"порог|лимит|достижени|достигн|максимум\s+один|не более одного|однократн", re.I)
_NOTIFY = re.compile(r"push|уведомлен|оповещ|сообщени|sms|смс", re.I)
_TIME_BOUND = re.compile(r"(?:в течение|за|не более|не позднее|в пределах)\s+(?:\d+\s*)?(?:\w+\s+)?(?:мс|миллисекунд|секунд|минут|час)", re.I)
_MEASURE = re.compile(r"перцентил|\bp\d{2}\b|\d+\s*%|процент|нагрузк|\brps\b|\btps\b|запрос\w*\s+в\s+секунд|одновременн|пиков", re.I)
_INSTANT = re.compile(r"в момент|мгновенн|сразу|в реальном времени|онлайн", re.I)
_STORE = re.compile(r"\bхранит|\bхранени|\bсохраня\w* в (?:бд|базе)|store", re.I)
_RETENTION = re.compile(r"срок\w*\s+хранени|хранить\s+(?:не более|не менее|в течение)|\d+\s*(?:лет|год|дн|месяц)\w*\s+хран|удаля|удалени|архив|retention",
                        re.I)
_PII = re.compile(r"идентификатор\w* карт|номер\w* карт|\bиин\b|телефон|паспорт|фио|pan\b", re.I)
_UI_ERR = re.compile(r"ошибк\w*[^.]{0,40}сообщени|сообщени\w*[^.]{0,40}ошибк", re.I)
_UI_TEXT = re.compile(r"текст\w*\s+(?:ошибк|сообщени)|сообщени\w*\s+«|валидац\w*\s*:", re.I)
_PRECOND = re.compile(r"активн\w*\s+карт|держател\w*\s+активн|только\s+для|только\s+(?:клиент|держател)|карт\w*\s+(?:kzt|в тенге)|в валюте",
                      re.I)
_OUTSIDE = re.compile(r"неактивн|заблокир|блокировк|закрыт\w*\s+карт|перевыпуск|в противном случае|иначе\b", re.I)

# правило → (что ищем в требованиях, что должно быть в критериях приёмки)
_ACCEPT_CONCEPTS = [
    ("возврат покупки", _REVERSAL, _REVERSAL),
    ("изменение порога/настройки", re.compile(r"при изменении\s+(?:порог|лимит|настро)|изменени\w*\s+(?:порог|лимит|настро)", re.I),
     re.compile(r"измен\w*\s+(?:порог|лимит|настро)|нов\w*\s+(?:порог|лимит)", re.I)),
    ("ограничение частоты уведомлений", re.compile(r"максимум\s+один|не более одного|однократн", re.I),
     re.compile(r"втор\w*\s+(?:push|уведомл)|повторн\w*\s+(?:достижен|push|уведомл)|ещ[её]\s+одн|не\s+(?:приходит|отправля)", re.I)),
    ("исключённые операции", re.compile(r"исключа|не учитыва|не входят", re.I), re.compile(r"перев[её]|перевод|сняти|исключ", re.I)),
]
_RANGE = re.compile(r"от\s+(\d[\d\s ]*\d|\d)\s+до\s+(\d[\d\s ]*\d|\d)", re.I)


def _norm(t: str) -> str:
    return t.lower().replace("ё", "е")


def _snip(text: str, m: re.Match | None, before: int = 25, after: int = 45) -> str:
    if not m:
        return " ".join(text.split()[:12])
    s = max(0, m.start() - before)
    e = min(len(text), m.end() + after)
    # обрезаем по словам, чтобы цитата находилась в документе
    while s > 0 and text[s - 1] not in " .,;:«(":
        s -= 1
    while e < len(text) and text[e] not in " .,;:»)":
        e += 1
    return text[s:e].strip(" .,;")


def _mk(code: str, sev: str, topic: str, problem: str, u: ReqUnit | None, evidence: list[str], fix: str,
        related: list[str] | None = None, lines: list[int] | None = None):
    from agents.iso_review import Issue

    return Issue(characteristic=code, severity=sev, problem=problem, evidence=[e for e in evidence if e], fix=fix,
                 lines=lines if lines is not None else (list(u.lines[:1]) if u else []), related=related or [],
                 source="rules", agent=f"{AGENT}:{topic}")


def unit_rules(seg: Segmentation, doc: Document) -> dict[str, list]:
    """Доменные сигналы, привязанные к конкретному требованию."""
    out: dict[str, list] = {u.rid: [] for u in seg.units}
    doc_low = _norm(doc.text)
    req_units = [u for u in seg.units if u.kind in ("functional", "business_rule", "data", "integration", "nonfunctional")]
    has_idem_key = bool(_IDEM_KEY.search(doc_low))
    has_outside = bool(_OUTSIDE.search(doc_low))
    business = [u for u in seg.units if u.kind == "business" and _INSTANT.search(u.text)]

    for u in seg.units:
        t, low = u.text, _norm(u.text)
        issues = out[u.rid]
        if u.kind in ("functional", "business_rule", "data"):
            # время: граница периода
            if _PERIOD.search(low) and not _TZ.search(low):
                city = _CITY.search(t)
                if city:
                    issues.append(_mk(
                        "unambiguous", "major", "time",
                        f"Граница периода задана городом («{city.group(0).strip()}»), а не часовым поясом. С 1 марта 2024 "
                        f"Казахстан перешёл на единый UTC+5, и системы с разными версиями базы часовых поясов посчитают "
                        f"полночь по-разному; операции около полуночи попадут не в тот месяц.", u, [_snip(t, city)],
                        "Указать часовой пояс явно (UTC+5 или зона Asia/Almaty с версией tzdata) и хранить время операции в UTC."))
                elif u.kind != "data" and _BOUNDARY.search(t):
                    m = _BOUNDARY.search(t)
                    issues.append(_mk(
                        "unambiguous", "minor", "time",
                        "Граница календарного периода не привязана к часовому поясу: процессинг, сервис расходов и приложение "
                        "могут считать начало месяца по-разному.", u, [_snip(t, m)],
                        "Указать часовой пояс, по которому определяется период."))
            # время: какой момент операции
            ma, mb = _MOMENT_A.search(low), _MOMENT_B.search(low)
            if ma and mb:
                issues.append(_mk(
                    "unambiguous", "major", "time",
                    "В одном требовании смешаны два момента операции: «совершение» (авторизация) и «проведение» (клиринг). "
                    "Между ними могут пройти дни, поэтому неясно, по какой дате относить покупку к месяцу и какие покупки "
                    "уже входят в расходы.", u, [_snip(t, _MOMENT_A.search(t), 15, 30), _snip(t, _MOMENT_B.search(t), 15, 30)],
                    "Определить термины в глоссарии и указать одну отметку времени (авторизация или проведение) для учёта."))
        # деньги: разные единицы
        mi, ma_ = _MINOR_UNIT.search(t), _MAJOR_UNIT.search(t)
        if mi and ma_ and u.kind not in ("acceptance",) and not _MINOR_DEFINED.search(low):
            issues.append(_mk(
                "unambiguous", "major", "money",
                f"Суммы в одном требовании заданы в разных единицах («{mi.group(0)}» и «{ma_.group(0)}») без правила пересчёта "
                "и округления: при сравнении с порогом легко ошибиться в 100 раз.", u, [_snip(t, mi, 20, 10), _snip(t, ma_, 20, 10)],
                "Хранить и сравнивать все суммы в одной единице (тиыны), явно описать перевод порога из тенге."))
        # данные: срок хранения и защита
        if (u.kind == "data" or "данн" in u.section.lower()) and _STORE.search(low) and not _RETENTION.search(low):
            pii = _PII.search(low) and not _PROTECTED.search(low)
            issues.append(_mk(
                "complete", "major", "retention",
                "Не указаны срок хранения и правила удаления данных" + (
                    ", а также защита идентификаторов клиента и карты (маскирование, шифрование, доступ)." if pii else "."),
                u, [_snip(t, _STORE.search(t), 0, 40)],
                "Задать срок хранения для каждого вида данных, правило удаления и требования к защите в соответствии с политикой ИБ банка."))
        # идемпотентность
        mr = _RETRY.search(low)
        if mr and not has_idem_key and u.kind in ("functional", "business_rule", "integration", "nonfunctional", "api_logic"):
            issues.append(_mk(
                "complete", "major", "idempotency",
                "Требуется обрабатывать повторы, но не определён механизм идемпотентности: по какому ключу распознаётся "
                "повтор, сколько ключ хранится и что возвращается при повторе. Без этого возможны двойной учёт суммы и "
                "повторные уведомления.", u, [_snip(t, _RETRY.search(t), 0, 50)],
                "Указать ключ идемпотентности (например, идентификатор операции процессинга), срок его хранения и ответ на повтор."))
        # возврат против событий по порогу
        if _REVERSAL.search(low) and u.kind in ("functional", "business_rule") and not _NOTIFY.search(low) \
                and re.search(r"(?<!без)возврат\w*\s+(?:уменьша|увеличива|корректир|пересчит|учитыва|входит|сниж)", low):
            triggers = [x for x in req_units if x.rid != u.rid and _TRIGGER.search(_norm(x.text)) and _NOTIFY.search(_norm(x.text))]
            covered = any(_REVERSAL.search(_norm(x.text)) and _NOTIFY.search(_norm(x.text)) for x in seg.units)
            if triggers and not covered:
                refs = ", ".join(f"{x.doc_ref}" for x in triggers[:2])
                issues.append(_mk(
                    "complete", "major", "reversal",
                    f"Возврат уменьшает накопленную сумму, но не описано, как это влияет на уведомления по порогу ({refs}): "
                    "отправляется ли push повторно после возврата и нового достижения, что делать с возвратом покупки прошлого месяца "
                    "и с возвратом после уже отправленного уведомления.", u, [_snip(t, _REVERSAL.search(t), 0, 50)],
                    "Описать правила для возврата: пересчёт счётчика, повторное достижение порога, возврат за прошлый месяц.",
                    related=[x.rid for x in triggers[:2]]))
        # НФТ без условий измерения
        if u.kind == "nonfunctional":
            mt = _TIME_BOUND.search(low)
            if mt and not _MEASURE.search(low):
                extra = ""
                rel = []
                if business:
                    extra = (f" Кроме того, цель «{_snip(business[0].text, _INSTANT.search(business[0].text), 0, 10)}» и срок "
                             f"«{mt.group(0)}» расходятся: неясно, что считается «моментом» для клиента.")
                    rel = [business[0].rid]
                issues.append(_mk(
                    "verifiable", "major", "nfr",
                    f"Срок «{mt.group(0)}» задан без условий измерения: не указаны доля событий (перцентиль), нагрузка, "
                    f"точки начала и конца отсчёта и поведение при недоступности смежных систем." + extra,
                    u, [_snip(t, _TIME_BOUND.search(t), 10, 30)],
                    "Например: «95% событий передаются в сервис push не позднее 60 с от получения события процессинга при нагрузке "
                    "до N событий/с; 99,9% — не позднее 5 мин».", related=rel))
        # UI: ошибки без текстов
        if u.kind == "ui" and _UI_ERR.search(low) and not _UI_TEXT.search(low):
            issues.append(_mk(
                "complete", "minor", "ui",
                "Сказано, что при ошибке ввода появляется сообщение, но не заданы правила валидации поля и тексты ошибок "
                "(пустое значение, не число, вне диапазона).", u, [_snip(t, _UI_ERR.search(t), 10, 20)],
                "Перечислить проверки поля и текст сообщения для каждой ошибки."))
        # условие применимости без поведения вне условия
        if u.kind in ("functional", "business_rule") and _PRECOND.search(low) and not has_outside:
            issues.append(_mk(
                "complete", "minor", "precondition",
                "Требование ограничено условием, но не описано поведение вне его: неактивная или заблокированная карта, "
                "карта в другой валюте, перевыпуск карты (переносится ли настройка и накопленная сумма).",
                u, [_snip(t, _PRECOND.search(t), 10, 20)],
                "Описать, что видит и что получает клиент для карт вне условия и при смене статуса карты."))
    return out


def set_rules(seg: Segmentation, doc: Document) -> list:
    """Доменные сигналы по набору: покрытие правил критериями приёмки."""
    issues = []
    acc = [u for u in seg.units if u.kind == "acceptance"]
    reqs = [u for u in seg.units if u.kind in ("functional", "business_rule")]
    if acc and reqs:
        acc_text = _norm(" ".join(u.text for u in acc))
        gaps, rel = [], []
        for name, req_rx, acc_rx in _ACCEPT_CONCEPTS:
            src = [u for u in reqs if req_rx.search(_norm(u.text))]
            if src and not acc_rx.search(acc_text):
                gaps.append(f"{name} ({src[0].doc_ref})")
                rel.append(src[0].rid)
        for u in reqs:
            m = _RANGE.search(u.text)
            if m:
                lo, hi = (re.sub(r"\D", "", g) for g in m.groups())
                numbers = set(re.findall(r"\d+", re.sub(r"(?<=\d)[\s ](?=\d{3}\b)", "", acc_text)))
                if lo not in numbers and hi not in numbers:
                    gaps.append(f"границы диапазона {m.group(0)} ({u.doc_ref})")
                    rel.append(u.rid)
        if gaps:
            first = acc[0]
            issues.append(_mk(
                "set_verifiable", "major", "acceptance",
                "Критерии приёмки не покрывают часть правил: " + "; ".join(gaps) + ". По ним нельзя принять реализацию.",
                None, [first.quote.strip("| ")[:80]],
                "Добавить сценарии приёмки на каждое правило, включая граничные значения и отрицательные случаи.",
                related=list(dict.fromkeys(rel + [u.rid for u in acc])), lines=sorted({l for u in acc for l in u.lines})))
    return issues


# ── LLM-агент ─────────────────────────────────────────────────────────────────

class _RawDomain(BaseModel):
    findings: list[dict[str, Any]] = Field(default_factory=list)


async def llm_findings(client, doc: Document, seg: Segmentation, rule_hints: list[str], stats: dict) -> list[dict]:
    # таблица требований содержит дословный текст каждого пункта — полный документ не дублируем (быстрее и дешевле)
    table = "\n".join(f"{u.rid} | {u.section} › {u.doc_ref} | {u.kind} | L{u.lines[0]} | {u.text}" for u in seg.units)
    user = (f"## Документ «{seg.title}»: требования (ID | раздел › место в документе | вид | строка | дословный текст)\n{table}\n\n"
            f"## Что уже нашли доменные правила (не повторяй, дополняй)\n" + ("\n".join(rule_hints) or "—"))
    raw, tokens = await client.chat_json(load_prompt("iso_domain"), user, _RawDomain)
    stats["tokens"] += tokens
    return raw.findings


def topic_of(text: str) -> str:
    low = _norm(text)
    best, score = "", 0
    for topic, kws in TOPIC_KW.items():
        s = sum(1 for k in kws if k in low)
        if s > score:
            best, score = topic, s
    return best
