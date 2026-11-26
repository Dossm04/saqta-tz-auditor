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
    "security": ("принадлеж", "лог", "query", "url", "маскир", "персональн", "pan", "push", "sms", "смс"),
    "placeholder": ("не заполн", "позже", "открыт", "перечн", "тариф"),
    "terms": ("не определ", "порог", "термин", "крупн", "значительн"),
    "api": ("curl", "метод", "статус", "код", "пример"),
}

# ── безопасность API и уведомлений ─────────────────────────────────────────────
_OWNERSHIP = re.compile(r"принадлеж|владел\w*|owner|соответств\w*\s+(?:клиент|сесси|токен)|из\s+(?:сесси|токен)", re.I)
_ID_PARAM = re.compile(r"Parameter:\s*`?(\w*(?:Id|_id|ID))`?;", re.I)
_PII_PARAM = re.compile(r"телефон|phone|\bиин\b|\biin\b|номер\w*\s+карт|\bpan\b|card\s*number|паспорт|фио|e-?mail", re.I)
_PAN_PATH = re.compile(r"/\{?(?:pan|cardnumber|card_number|cardNumber)\}?|номер\w*\s+карт\w*[^.]{0,40}(?:в\s+(?:url|пут))", re.I)
_LOG_SECRET = re.compile(r"(?:лог\w*|журнал\w*)[^.]{0,60}(?:полн\w*\s+url|url\s+(?:каждого\s+)?запрос|authorization|заголов\w*\s+авториз|токен)", re.I)
_NOTIFY_SENSITIVE = re.compile(r"(?:push|sms|смс|уведомлени\w*)\w*\s+(?:[^.,\s]+\s+){0,4}?(?:с|со|содерж\w*|включа\w*|переда[её]т\w*|"
                               r"указыва\w*|:)\s+[^.]{0,100}?(?:полн\w*\s+номер\w*\s+карт|номер\w*\s+карт\w*(?!\s*\(?(?:послед|маск))|остат\w*\s+(?:на\s+сч|задолж|баланс)|остат\w*|баланс\w*|\bиин\b|номер\w*\s+договор)", re.I)
_MASKED = re.compile(r"маск|последн\w*\s+(?:4|четыр)", re.I)
# ── незаполненные места и открытые перечни ─────────────────────────────────────
_INLINE_HOLE = re.compile(r"(?:кроме|затем|включая|а также|далее|после этого)\s*[—–]\s*(?=[.,;)]|$)|заполня\w*\s+позже|будет\s+определ\w*\s+позже|"
                          r"определ\w*\s+позднее", re.I)
def _tbd_regex() -> re.Pattern:
    from knowledge.language import VAGUE_TERMS

    pats = [v.pattern for v in VAGUE_TERMS if v.kind == "tbd"]
    return re.compile("|".join(f"(?:{p})" for p in pats) or r"(?!x)x", re.I)


_TBD_ANY = _tbd_regex()
_OPEN_REF = re.compile(r"по\s+тарифам\s+банка|по\s+решению\s+банка|прочи\w*\s+операци\w*|и\s+т\.\s?д\.|и\s+другие|и\s+прочее", re.I)
_UNDEF_TERM = re.compile(r"\b(крупн\w+|значительн\w+|существенн\w+|небольш\w+|большо\w+)\s+(перевод\w*|сумм\w*|операци\w*|платеж\w*|платёж\w*|"
                         r"погашени\w*|покупк\w*)", re.I)
# ── объединение нескольких действий в одном пункте ─────────────────────────────
_ACTIONS = re.compile(r"\b(провер\w+|показыва\w+|отправля\w+|записыва\w+|списыва\w+|зачисля\w+|формиру\w+|пересчитыва\w+|начисля\w+|"
                      r"отража\w+|созда[её]т|блокиру\w+|уведомля\w+|сохраня\w+|рассчитыва\w+|предлага\w+|меняет|изменяет)", re.I)
_MONEY_OP = re.compile(r"перевод|платеж|платёж|погашени|списани|transfer|payment|repay|payout|выплат", re.I)

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
        # незаполненное место внутри формулировки и открытые перечни
        mh = _INLINE_HOLE.search(t)
        if mh and not _TBD_ANY.search(low):      # TBD уже отмечает Rule Checker
            issues.append(_mk("complete", "major", "placeholder",
                              "В формулировке оставлено незаполненное место — требование нельзя реализовать и проверить.",
                              u, [_snip(t, mh, 30, 10)], "Дописать недостающее значение или перечень."))
        mo = _OPEN_REF.search(t)
        if mo and u.kind not in ("acceptance",):
            issues.append(_mk("complete", "major", "placeholder",
                              f"Значение задано открытой ссылкой («{mo.group(0)}»): нет конкретного размера или закрытого перечня, "
                              "поэтому реализацию нельзя проверить.", u, [_snip(t, mo, 20, 15)],
                              "Указать конкретные значения (или ссылку на документ с версией) и закрытый перечень."))
        # термин-порог без определения
        mu = _UNDEF_TERM.search(t)
        if mu and not re.search(r"\d", re.sub(r"[ФF]-\d+", "", t)):
            issues.append(_mk("unambiguous", "major", "terms",
                              f"Термин «{mu.group(0)}» не определён: не задан порог, поэтому неясно, к каким операциям применяется правило.",
                              u, [_snip(t, mu, 10, 20)], "Определить порог в тенге и описать процесс и сроки для таких операций."))
        # несколько действий в одном предложении
        if u.kind in ("functional", "business_rule") and len(re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z])", t.strip())) < 2:
            acts = {m.group(1).lower()[:6] for m in _ACTIONS.finditer(t)}
            if len(acts) >= 3:
                issues.append(_mk("singular", "major", "terms",
                                  f"В одном пункте объединены {len(acts)} самостоятельных действия — их нельзя независимо реализовать, "
                                  "протестировать и проследить.", u, [" ".join(t.split()[:12])],
                                  "Разбить на отдельные требования с собственными ID."))
        # чувствительные данные в уведомлениях и логах, номер карты в URL
        mn = _NOTIFY_SENSITIVE.search(t)
        if mn and not _MASKED.search(mn.group(0)):
            issues.append(_mk("security", "major", "security",
                              "В уведомлении (push/SMS) передаются чувствительные данные: их видно на заблокированном экране и у оператора связи.",
                              u, [_snip(t, mn, 0, 10)], "Передавать только маскированные данные (последние 4 цифры), без остатка и ИИН."))
        ml = _LOG_SECRET.search(t)
        if ml:
            issues.append(_mk("security", "major", "security",
                              "В логи пишутся полный URL или данные авторизации: персональные данные и токены попадут в журналы и системы мониторинга.",
                              u, [_snip(t, ml, 10, 10)], "Логировать без query-строки и заголовков авторизации, маскировать персональные данные."))
        mp = _PAN_PATH.search(t)
        if mp:
            issues.append(_mk("security", "critical", "security",
                              "Номер карты (PAN) передаётся в URL: он попадёт в логи шлюза, прокси и браузера — нарушение PCI DSS.",
                              u, [_snip(t, mp, 15, 15)], "Использовать внутренний идентификатор или токен карты, PAN — только в теле запроса по защищённому каналу."))
        # условие применимости без поведения вне условия
        if u.kind in ("functional", "business_rule") and _PRECOND.search(low) and not has_outside:
            issues.append(_mk(
                "complete", "minor", "precondition",
                "Требование ограничено условием, но не описано поведение вне его: неактивная или заблокированная карта, "
                "карта в другой валюте, перевыпуск карты (переносится ли настройка и накопленная сумма).",
                u, [_snip(t, _PRECOND.search(t), 10, 20)],
                "Описать, что видит и что получает клиент для карт вне условия и при смене статуса карты."))
    _api_unit_rules(seg, doc, out, has_idem_key)
    return out


def _api_unit_rules(seg: Segmentation, doc: Document, out: dict[str, list], has_idem_key: bool) -> None:
    """Безопасность и надёжность метода API: проверка владельца, ПДн в query, идемпотентность денежных операций."""
    logic = [u for u in seg.units if u.kind == "api_logic"]
    params = [u for u in seg.units if u.kind == "interface" and u.text.startswith("Parameter:")]
    req_params = [u for u in params if re.search(r"query|path|запрос", u.section, re.I) and not re.search(r"ответ|response", u.section, re.I)]
    id_params = [u for u in req_params if _ID_PARAM.search(u.text) or re.search(r"path", u.section, re.I)]
    if logic and id_params and not any(_OWNERSHIP.search(u.text) for u in logic):
        names = ", ".join(m.group(1) for u in id_params[:3] for m in [re.match(r"Parameter:\s*`?([\w.-]+)", u.text)] if m)
        out[logic[0].rid].append(_mk(
            "security", "critical", "security",
            f"Логика метода не проверяет, что объект из запроса ({names}) принадлежит клиенту из сессии: подставив чужой "
            "идентификатор, можно действовать с чужой картой или счётом (IDOR).", logic[0], [logic[0].quote.strip("| ")[:70]],
            "Добавить шаг «проверить, что объект принадлежит клиенту из токена»; для чужого объекта возвращать 404.",
            related=[u.rid for u in id_params[:3]]))
    pii = [u for u in req_params if re.search(r"query", u.section, re.I) and _PII_PARAM.search(u.text)]
    if pii:
        names = ", ".join(re.match(r"Parameter:\s*`?([\w.-]+)", u.text).group(1) for u in pii[:4] if re.match(r"Parameter:\s*`?([\w.-]+)", u.text))
        out[pii[0].rid].append(_mk(
            "security", "major", "security",
            f"Персональные данные ({names}) передаются в query-строке URL: они сохраняются в логах шлюза, прокси и истории браузера.",
            pii[0], [pii[0].quote.strip("| ")[:70]], "Передавать персональные данные в теле запроса (POST) или брать из сессии.",
            related=[u.rid for u in pii[1:4]]))
    method_u = next((u for u in seg.units if u.kind == "interface" and re.search(r"rest type|http method|метод запроса", u.text, re.I)), None)
    desc = " ".join(u.text for u in seg.units if u.kind == "interface" and re.search(r"description|request url|назначение", u.text, re.I))
    if method_u and not has_idem_key and _MONEY_OP.search(desc):
        curl = [u for u in seg.units if u.kind == "interface" and re.search(r"\bcurl\b", u.text, re.I)]
        money_steps = [u for u in logic if re.search(r"списа|зачисл|погаси|перев[её]|начисл", u.text, re.I)]
        out[method_u.rid].append(_mk(
            "complete", "major", "idempotency",
            "Метод проводит денежную операцию, но идемпотентность не определена: повтор запроса при таймауте приведёт к двойному списанию.",
            method_u, [method_u.quote.strip("| ")[:60]],
            "Ввести ключ идемпотентности (заголовок Idempotency-Key или идентификатор операции клиента) и описать ответ на повтор.",
            related=[u.rid for u in curl[:1] + money_steps[:2]]))


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
        # числовые лимиты и пороги из требований, ни одно значение которых не встречается в приёмке
        acc_numbers = set(re.findall(r"\d+", re.sub(r"(?<=\d)[\s ](?=\d{3}\b)", "", acc_text)))
        # только для критериев-сценариев (таблица «условия — действие — результат»), а не общих формулировок
        scenario_acc = any(re.search(r"ожидаемый результат|исходные условия|действие:", x.text, re.I) for x in acc)
        for u in (reqs if scenario_acc else []):
            if u.rid in rel or not re.search(r"лимит|минимальн|максимальн|порог|комисси|не более|не менее|не позднее|срок", u.text, re.I):
                continue
            body = re.sub(r"[ФF]-\d+", "", u.text)
            nums = set(re.findall(r"\d+", re.sub(r"(?<=\d)[\s ](?=\d{3}\b)", "", body)))
            if nums and not nums & acc_numbers:
                gaps.append(f"{u.doc_ref}: «{' '.join(u.text.split()[:8])}…»")
                rel.append(u.rid)
            if len(gaps) >= 6:
                break
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
    issues += _api_set_rules(seg, doc)
    return issues


def _api_set_rules(seg: Segmentation, doc: Document) -> list:
    issues = []
    iface = [u for u in seg.units if u.kind == "interface"]
    # метод в CURL не совпадает с объявленным
    method_u = next((u for u in iface if re.search(r"rest type|http method|метод запроса", u.text, re.I)), None)
    curl_u = next((u for u in iface if re.search(r"\bcurl\b", u.text, re.I)), None)
    if method_u and curl_u:
        declared = re.search(r"\b(GET|POST|PUT|PATCH|DELETE)\b", method_u.text)
        mx = re.search(r"-X\s+'?\"?(GET|POST|PUT|PATCH|DELETE)", curl_u.text)
        used = mx.group(1) if mx else ("GET" if re.search(r"\scurl\s+-G\b|\s-G\s", " " + curl_u.text) else
                                       "POST" if re.search(r"\s(?:-d|--data)\b", curl_u.text) else None)
        if declared and used and declared.group(1) != used:
            issues.append(_mk("set_consistent", "major", "api",
                              f"Метод объявлен как {declared.group(1)}, а в примере CURL используется {used}: разработчик и тестировщик "
                              "реализуют разные вызовы.", method_u, [method_u.quote.strip("| ")[:60], curl_u.quote.strip("| ")[:70]],
                              "Привести CURL и объявленный метод к одному значению.", related=[method_u.rid, curl_u.rid],
                              lines=sorted(set(method_u.lines + curl_u.lines))))
    # перечень статусов не совпадает с примерами ответов
    listed = {m.group(1): u for u in seg.units for m in [re.search(r"код (\d{3})$", u.doc_ref)] if m}
    examples = {m.group(1): u for u in seg.units for m in [re.search(r"код (\d{3}), пример", u.doc_ref)] if m}
    if listed and examples:
        only_list = sorted(set(listed) - set(examples))
        only_ex = sorted(set(examples) - set(listed))
        if only_list or only_ex:
            parts = []
            if only_list:
                parts.append(f"в перечне есть {', '.join(only_list)}, но нет примера ответа")
            if only_ex:
                parts.append(f"есть пример ответа {', '.join(only_ex)}, но код не описан в перечне")
            us = [listed[c] for c in only_list] + [examples[c] for c in only_ex]
            issues.append(_mk("set_consistent", "major", "api",
                              "Перечень кодов ответа не совпадает с примерами: " + "; ".join(parts) + ".", us[0],
                              [u.quote.strip("| ")[:60] for u in us[:2]], "Согласовать перечень кодов с примерами: у каждого кода — условие и пример.",
                              related=[u.rid for u in us[:4]], lines=sorted({l for u in us[:4] for l in u.lines[:1]})))
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
