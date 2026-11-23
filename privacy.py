"""Режим конфиденциальности для банка: маскирование ПДн перед отправкой текста во внешнюю LLM.

Что маскируется (обратимо, плейсхолдеры стабильны в рамках одной проверки):
  * ИИН / БИН РК — 12 цифр; контрольная сумма по алгоритму РК или «ИИН/БИН» рядом + правдоподобная дата
  * номер карты (PAN) — 13–19 цифр, в т.ч. группами через пробел/дефис, с проверкой Луна
  * телефоны РК — +7 / 8, с пробелами, дефисами, скобками
  * e-mail
  * IBAN KZ — KZ + 18 букв/цифр (можно группами по 4)
  * (опционально) названия организаций в кавычках после АО/ТОО/ИП/ООО… и «ИП Фамилия И.О.»

Маскирование никогда не пересекает перевод строки — число строк сохраняется,
поэтому номера L<n> в промптах и ответах моделей остаются валидными.

Использование:
    masked, mapping = mask_text(text)          # mapping: {"[ИИН_1]": "900101300125", ...}
    restored = unmask_obj(llm_result, mapping)  # str / list / dict / pydantic-модель

    llm = PrivacyLLMClient(LLMClient(cfg))      # тот же интерфейс chat_json(system, user, schema)
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

# ── Плейсхолдеры ──────────────────────────────────────────────────────────────
TAGS = {
    "iin": "ИИН",
    "bin": "БИН",
    "card": "КАРТА",
    "phone": "ТЕЛ",
    "email": "EMAIL",
    "iban": "IBAN",
    "org": "ОРГ",
}
_TAG_ALT = "|".join(TAGS.values())
# допускаем, что модель «потеряла» квадратные скобки: ИИН_1 → тоже восстанавливаем
_PLACEHOLDER_RE = re.compile(rf"\[?\b({_TAG_ALT})_(\d+)\b\]?")


def _placeholder(tag: str, n: int) -> str:
    return f"[{tag}_{n}]"


# ── Валидаторы ────────────────────────────────────────────────────────────────
def luhn_ok(digits: str) -> bool:
    """Проверка номера карты по алгоритму Луна."""
    if not digits.isdigit() or len(digits) < 2:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iin_checksum_ok(iin: str) -> bool:
    """Контрольный разряд ИИН/БИН РК: веса 1..11, при остатке 10 — веса 3..11,1,2; остаток 10 → невалиден."""
    if len(iin) != 12 or not iin.isdigit():
        return False
    d = [int(c) for c in iin]
    for weights in (range(1, 12), [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]):
        k = sum(w * x for w, x in zip(weights, d[:11])) % 11
        if k != 10:
            return k == d[11]
    return False


def _iin_date_ok(iin: str) -> bool:
    """ГГММДД + 7-й разряд (век и пол: 1–6; 0 — иностранцы)."""
    mm, dd, century = int(iin[2:4]), int(iin[4:6]), int(iin[6])
    return 1 <= mm <= 12 and 1 <= dd <= 31 and century <= 6


def _bin_head_ok(bin_: str) -> bool:
    """БИН: ГГММ + тип (4 — резидент ЮЛ, 5 — нерезидент, 6 — ИП(С)) + признак (0–3)."""
    mm, kind, attr = int(bin_[2:4]), int(bin_[4]), int(bin_[5])
    return 1 <= mm <= 12 and kind in (4, 5, 6) and attr <= 3


def is_valid_iin(iin: str, strict: bool = True) -> bool:
    """ИИН РК: правдоподобная дата рождения, разряд века и (strict) контрольная сумма."""
    if len(iin) != 12 or not iin.isdigit() or not _iin_date_ok(iin):
        return False
    return iin_checksum_ok(iin) if strict else True


def is_valid_bin(bin_: str, strict: bool = True) -> bool:
    if len(bin_) != 12 or not bin_.isdigit() or not _bin_head_ok(bin_):
        return False
    return iin_checksum_ok(bin_) if strict else True


# ── Шаблоны (ни один не пересекает \n: только [ \t]) ──────────────────────────
_SP = r"[ \t]"
_SEP = r"[ \t\-]"
_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}(?![\w-])")
_IBAN_RE = re.compile(r"(?<![A-Za-z0-9])KZ\d{2}(?:[ \t]?[0-9A-Z]{4}){4}(?![A-Za-z0-9])", re.IGNORECASE)
# карта: сплошные 13–19 цифр или типичные группировки 4-4-4-4(-3), 4-6-5, 4-6-4
_CARD_RE = re.compile(
    r"(?<![\d\w])(?:"
    rf"\d{{4}}(?:{_SEP}\d{{4}}){{3}}(?:{_SEP}\d{{1,3}})?"
    rf"|\d{{4}}{_SEP}\d{{6}}{_SEP}\d{{4,5}}"
    r"|\d{13,19}"
    r")(?![\d\w])"
)
# телефон РК: +7 / 8, код в скобках или без, разделители пробел/дефис
_PHONE_RE = re.compile(
    r"(?<![\w+])(?:"
    rf"\+{_SP}?7{_SEP}?\(?\d{{3}}\)?{_SEP}?\d{{3}}{_SEP}?\d{{2}}{_SEP}?\d{{2}}"
    rf"|8{_SEP}?\(?[3-9]\d{{2}}\)?{_SEP}?\d{{3}}{_SEP}?\d{{2}}{_SEP}?\d{{2}}"
    r")(?!\d)"
)
_IIN_RE = re.compile(r"(?<![\d\w])\d{12}(?![\d\w])")
# «ИИН», «БИН клиента:», «ИИН/БИН №» — ключевое слово на той же строке, без цифр между ним и номером
_ID_KEYWORD_RE = re.compile(r"(?<![\w])(ИИН|БИН|IIN|BIN|идентификационн\w*\s+номер)(?![\w])[^\n\d]{0,30}$",
                            re.IGNORECASE)
# госорганы (РГУ/РГП/ГУ) не маскируем: регулятор важен для анализа и не является тайной
_ORG_FORMS = r"АО|ТОО|ИП|ООО|ПАО|ЗАО|ОАО|НАО|ЧУ|ОФ|ОЮЛ|LLP|JSC|LLC|Ltd"
# публичная инфраструктура рынка — оставляем как есть, модели нужен контекст
_PUBLIC_ORG_RE = re.compile(
    r"национальн\w*\s+банк|межбанковск\w*\s+расч|кредитн\w*\s+бюро|фондов\w*\s+бирж|KASE|"
    r"электронн\w*\s+правительств|государственн\w*\s+корпораци|правительство\s+для\s+граждан",
    re.IGNORECASE,
)
_ORG_QUOTED_RE = re.compile(
    rf"(?<![\w])((?:{_ORG_FORMS})\.?{_SP}*)([«\"“„'])([^«»\"“”„'\n]{{1,80}})([»\"”“'])"
)
# ИП Фамилия И.О. / ИП Фамилия Имя
_IP_PERSON_RE = re.compile(
    rf"(?<![\w])(ИП{_SP}+)([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
    rf"(?:{_SP}+[А-ЯЁ]\.{_SP}?[А-ЯЁ]\.?|{_SP}+[А-ЯЁ][а-яё]+(?:{_SP}+[А-ЯЁ][а-яё]+)?)?)"
)


# ── Маскирование ──────────────────────────────────────────────────────────────
class _Masker:
    """Хранит обратимую таблицу; одинаковое значение → одинаковый плейсхолдер."""

    def __init__(self, mapping: dict[str, str] | None, text: str) -> None:
        self.mapping: dict[str, str] = mapping if mapping is not None else {}
        self.reverse: dict[tuple[str, str], str] = {}
        self.counters: Counter[str] = Counter()
        for ph, value in self.mapping.items():
            m = _PLACEHOLDER_RE.fullmatch(ph)
            if m:
                self.reverse[(m.group(1), value)] = ph
                self.counters[m.group(1)] = max(self.counters[m.group(1)], int(m.group(2)))
        self.text = text
        self.stats: Counter[str] = Counter()

    def ph(self, tag: str, value: str) -> str:
        existing = self.reverse.get((tag, value))
        if existing:
            self.stats[tag] += 1
            return existing
        n = self.counters[tag] + 1
        # не занимаем плейсхолдер, который уже встречается в исходном тексте буквально
        while _placeholder(tag, n) in self.text:
            n += 1
        self.counters[tag] = n
        ph = _placeholder(tag, n)
        self.mapping[ph] = value
        self.reverse[(tag, value)] = ph
        self.stats[tag] += 1
        return ph


def mask_text(text: str, mapping: dict[str, str] | None = None, *, orgs: bool = True,
              stats: Counter[str] | None = None) -> tuple[str, dict[str, str]]:
    """Заменяет ПДн на плейсхолдеры [ИИН_1], [КАРТА_1], [ТЕЛ_1], [EMAIL_1], [IBAN_1], [ОРГ_1].

    mapping — существующая таблица (плейсхолдер → оригинал); дополняется на месте, чтобы
    несколько вызовов в одной проверке давали согласованные плейсхолдеры.
    Возвращает (замаскированный текст, mapping). Число строк не меняется.
    """
    if not text:
        return text, mapping if mapping is not None else {}
    mk = _Masker(mapping, text)
    out = text

    # порядок важен: e-mail и IBAN до цифровых шаблонов, карта (Луна) до телефона и ИИН
    out = _EMAIL_RE.sub(lambda m: mk.ph(TAGS["email"], m.group(0)), out)
    out = _IBAN_RE.sub(lambda m: mk.ph(TAGS["iban"], m.group(0)), out)

    def card(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits) and len(set(digits)) > 1:
            return mk.ph(TAGS["card"], m.group(0))
        return m.group(0)

    out = _CARD_RE.sub(card, out)
    out = _PHONE_RE.sub(lambda m: mk.ph(TAGS["phone"], m.group(0)), out)

    def iin(m: re.Match[str]) -> str:
        s = m.group(0)
        before = m.string[max(0, m.start() - 40):m.start()]
        kw = _ID_KEYWORD_RE.search(before)
        if kw and kw.group(1).upper() in ("БИН", "BIN"):
            if is_valid_bin(s, strict=False) or iin_checksum_ok(s):
                return mk.ph(TAGS["bin"], s)
        if is_valid_iin(s, strict=not kw):
            return mk.ph(TAGS["iin"], s)
        if is_valid_bin(s, strict=True):
            return mk.ph(TAGS["bin"], s)
        if kw and iin_checksum_ok(s):
            return mk.ph(TAGS["iin"], s)
        return s

    out = _IIN_RE.sub(iin, out)

    if orgs:
        def org(m: re.Match[str]) -> str:
            name = m.group(3)
            if _PLACEHOLDER_RE.fullmatch(name.strip()) or _PUBLIC_ORG_RE.search(name) or not name.strip():
                return m.group(0)
            return f"{m.group(1)}{m.group(2)}{mk.ph(TAGS['org'], name)}{m.group(4)}"

        out = _ORG_QUOTED_RE.sub(org, out)
        out = _IP_PERSON_RE.sub(lambda m: f"{m.group(1)}{mk.ph(TAGS['org'], m.group(2))}", out)

    if stats is not None:
        stats.update(mk.stats)
    return out, mk.mapping


def unmask_text(text: str, mapping: dict[str, str]) -> str:
    """Возвращает оригинальные значения на место плейсхолдеров (за один проход)."""
    if not text or not mapping:
        return text

    def repl(m: re.Match[str]) -> str:
        return mapping.get(_placeholder(m.group(1), int(m.group(2))), m.group(0))

    return _PLACEHOLDER_RE.sub(repl, text)


def unmask_obj(obj: Any, mapping: dict[str, str]) -> Any:
    """Рекурсивно восстанавливает плейсхолдеры в str / list / tuple / dict / pydantic-моделях."""
    if not mapping:
        return obj
    if isinstance(obj, str):
        return unmask_text(obj, mapping)
    if isinstance(obj, list):
        return [unmask_obj(x, mapping) for x in obj]
    if isinstance(obj, tuple):
        return tuple(unmask_obj(x, mapping) for x in obj)
    if isinstance(obj, dict):
        return {unmask_obj(k, mapping): unmask_obj(v, mapping) for k, v in obj.items()}
    dump = getattr(obj, "model_dump", None)
    validate = getattr(type(obj), "model_validate", None)
    if callable(dump) and callable(validate):
        try:
            return validate(unmask_obj(dump(), mapping))
        except Exception:  # схема не приняла восстановленные значения — лучше плейсхолдер, чем падение
            return obj
    return obj


def describe_stats(stats: Counter[str] | dict[str, int]) -> str:
    """'ИИН: 2, КАРТА: 1' — для трассировки/бейджа в UI."""
    return ", ".join(f"{tag}: {n}" for tag, n in sorted(stats.items(), key=lambda kv: -kv[1]) if n)


# ── Обёртка над LLM-клиентом ──────────────────────────────────────────────────
class PrivacyLLMClient:
    """Прозрачная обёртка над любым клиентом с методом `async chat_json(system, user, schema)`.

    Маскирует system и user перед отправкой, восстанавливает значения в провалидированном
    ответе. Таблица замен общая на всю проверку, поэтому все агенты видят одинаковые
    плейсхолдеры. Остальные атрибуты (active_model, switches, transient_errors, cache_hits…)
    делегируются внутреннему клиенту.
    """

    def __init__(self, inner: Any, *, orgs: bool = True) -> None:
        self.inner = inner
        self.orgs = orgs
        self.mapping: dict[str, str] = {}
        self.masked_stats: Counter[str] = Counter()   # сколько замен сделано (по типам, с повторами)
        self.calls = 0

    def mask(self, text: str) -> str:
        masked, _ = mask_text(text, self.mapping, orgs=self.orgs, stats=self.masked_stats)
        return masked

    def prime(self, text: str) -> None:
        """Заранее пронумеровать ПДн документа в порядке появления (стабильные плейсхолдеры и кэш)."""
        mask_text(text, self.mapping, orgs=self.orgs)

    async def chat_json(self, system: str, user: str, schema: Any) -> tuple[Any, int]:
        self.calls += 1
        result, tokens = await self.inner.chat_json(self.mask(system), self.mask(user), schema)
        return unmask_obj(result, self.mapping), tokens

    @property
    def privacy_note(self) -> str:
        """Строка для трассировки: что и сколько замаскировано."""
        uniq = Counter(m.group(1) for ph in self.mapping if (m := _PLACEHOLDER_RE.fullmatch(ph)))
        if not uniq:
            return "режим конфиденциальности: ПДн в тексте не найдены"
        return "режим конфиденциальности: замаскировано перед отправкой — " + describe_stats(uniq)

    def __getattr__(self, name: str) -> Any:
        # вызывается только для отсутствующих атрибутов; защита от рекурсии до инициализации
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)
