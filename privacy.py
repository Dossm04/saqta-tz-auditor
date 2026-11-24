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

PrivacyLLMClient дополнительно маскирует всё, что уже есть в таблице замен (mask_known):
значение без окружающего контекста, переформатированный номер и обрезанные начало/конец номера.
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


# ── Поиск уже известных значений (защита от «перефразированных» утечек) ──────────
# После unmask_obj результаты LLM попадают обратно в промпты следующих агентов уже без
# контекста, нужного детекторам («ТОО «…»», «ИИН:»), иногда переформатированными или
# обрезанными. Поэтому всё, что уже есть в таблице замен, ищется ещё и буквально.
_DIGIT_TAGS = {TAGS["card"], TAGS["phone"], TAGS["iin"], TAGS["bin"]}
_CI_TAGS = {TAGS["email"], TAGS["iban"]}
# разделители, которыми модель может «переформатировать» номер: 4111-1111…, (701) 123-45-67
_DSEP = r"[ \t\-(). ]{0,3}"
# защитная альтернатива: плейсхолдеры (в т.ч. без скобок) не трогаем
_PH_PROTECT = rf"\[?\b(?:{_TAG_ALT})_\d+\b\]?"
_MIN_FRAGMENT = 4          # обрезанный хвост/начало номера: минимум цифр, чтобы маскировать
# (?<!\w): номера строк «L1234:» и идентификаторы «FR12» не считаются фрагментами
_FRAGMENT_RE = re.compile(rf"({_PH_PROTECT})|(?<!\w)(\d+(?:[ \t\-]\d+)*)")


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _phone_core(digits: str) -> str:
    """Номер РК без кода страны: +7 701 … / 8 701 … / 7 701 … → 701…"""
    return digits[1:] if len(digits) == 11 and digits[0] in "78" else digits


def _digit_pattern(digits: str) -> str:
    return _DSEP.join(re.escape(c) for c in digits)


def _value_pattern(tag: str, value: str) -> str | None:
    """Регэксп для поиска известного значения в произвольном (возможно, переформатированном) тексте."""
    if tag in _DIGIT_TAGS:
        digits = _digits(value)
        if len(digits) < 7:
            return None
        if tag == TAGS["phone"]:
            core = _phone_core(digits)
            return rf"(?<!\d)(?:(?:\+{_DSEP}7|8|7){_DSEP})?\(?{_digit_pattern(core)}(?!\d)"
        return rf"(?<!\d){_digit_pattern(digits)}(?!\d)"
    v = value.strip()
    if len(v) < 2:
        return None
    if tag == TAGS["iban"]:
        body = r"[ \t]?".join(re.escape(c) for c in re.sub(r"\s", "", v))
        return rf"(?<![A-Za-z0-9])(?i:{body})(?![A-Za-z0-9])"
    body = r"[ \t ]+".join(re.escape(w) for w in v.split())
    if tag in _CI_TAGS:
        body = f"(?i:{body})"
    return rf"(?<!\w){body}(?!\w)"


def _numeric_keys(tag: str, value: str) -> list[str]:
    """Цифровые формы значения, по которым ищутся обрезанные фрагменты (префиксы/суффиксы)."""
    if tag not in _DIGIT_TAGS:
        return []
    digits = _digits(value)
    if len(digits) < 7:
        return []
    keys = [digits]
    if tag == TAGS["phone"] and _phone_core(digits) != digits:
        keys.append(_phone_core(digits))
    return keys


class _KnownValues:
    """Однопроходная замена известных значений на их плейсхолдеры (длинные — первыми).

    Регэксп — одна альтернация: сначала «защита» существующих плейсхолдеров, затем значения
    по убыванию длины. Так замены не каскадируют и короткое значение не режет длинное.
    Кэшируется и перестраивается только при росте таблицы замен.
    """

    def __init__(self) -> None:
        self._size = -1
        self._rx: re.Pattern[str] | None = None
        self._groups: list[str] = []                 # индекс группы → плейсхолдер
        self._numeric: list[tuple[str, str, str]] = []   # (цифры значения, плейсхолдер, тег)

    def _build(self, mapping: dict[str, str]) -> None:
        self._size = len(mapping)
        items: list[tuple[int, str, str]] = []
        self._numeric = []
        for ph, value in mapping.items():
            m = _PLACEHOLDER_RE.fullmatch(ph)
            if not m or not isinstance(value, str):
                continue
            tag = m.group(1)
            pat = _value_pattern(tag, value)
            if pat:
                key_len = len(_digits(value)) if tag in _DIGIT_TAGS else len(value.strip())
                items.append((key_len, pat, ph))
            for key in _numeric_keys(tag, value):
                self._numeric.append((key, ph, tag))
        items.sort(key=lambda it: -it[0])
        self._groups = [""] + [ph for _, _, ph in items]   # группа 1 — защита плейсхолдеров
        if not items:
            self._rx = None
            return
        alts = [f"({_PH_PROTECT})"] + [f"({pat})" for _, pat, _ in items]
        self._rx = re.compile("|".join(alts))

    def sub(self, text: str, mapping: dict[str, str], stats: Counter[str] | None) -> str:
        if not text or not mapping:
            return text
        if len(mapping) != self._size:
            self._build(mapping)
        if self._rx is not None:
            def repl(m: re.Match[str]) -> str:
                idx = m.lastindex or 1
                if idx == 1:
                    return m.group(0)
                ph = self._groups[idx - 1]
                if stats is not None:
                    tm = _PLACEHOLDER_RE.fullmatch(ph)
                    if tm:
                        stats[tm.group(1)] += 1
                return ph

            text = self._rx.sub(repl, text)
        if self._numeric:
            text = self._sub_fragments(text, stats)
        return text

    def _fragment_owner(self, digits: str, *, prefix: bool) -> tuple[str, str] | None:
        """Плейсхолдер значения, для которого digits — собственный префикс (prefix=True) или суффикс."""
        if len(digits) < _MIN_FRAGMENT:
            return None
        for key, ph, tag in self._numeric:
            if len(digits) < len(key) and (key.startswith(digits) if prefix else key.endswith(digits)):
                return ph, tag
        return None

    def _sub_fragments(self, text: str, stats: Counter[str] | None) -> str:
        """Обрезанные номера: «… ИИН (к): 8807255» (префикс) или «…5500313 в профиле» (суффикс).

        В серии чисел через пробел/дефис ищется хвост серии, совпадающий с началом известного
        значения (обрезка справа), или начало серии, совпадающее с его концом (обрезка слева).
        """
        def repl(m: re.Match[str]) -> str:
            if m.group(1):
                return m.group(0)
            run = m.group(2)
            parts = re.split(r"([ \t\-])", run)      # [число, разделитель, число, …]
            nums = parts[0::2]
            seps = parts[1::2]

            def join(a: int, b: int) -> str:
                out = nums[a]
                for k in range(a + 1, b):
                    out += seps[k - 1] + nums[k]
                return out

            # 1) хвост серии — префикс известного значения (чем длиннее хвост, тем лучше)
            for i in range(len(nums)):
                owner = self._fragment_owner("".join(nums[i:]), prefix=True)
                if owner:
                    if stats is not None:
                        stats[owner[1]] += 1
                    head = join(0, i) + seps[i - 1] if i else ""
                    return head + owner[0]
            # 2) начало серии — суффикс известного значения
            for j in range(len(nums), 0, -1):
                owner = self._fragment_owner("".join(nums[:j]), prefix=False)
                if owner:
                    if stats is not None:
                        stats[owner[1]] += 1
                    tail = seps[j - 1] + join(j, len(nums)) if j < len(nums) else ""
                    return owner[0] + tail
            return run

        return _FRAGMENT_RE.sub(repl, text)


def mask_known(text: str, mapping: dict[str, str], stats: Counter[str] | None = None) -> str:
    """Маскирует значения, уже известные таблице замен, без опоры на контекст.

    Находит: значение дословно (с границами слова; e-mail и IBAN — без учёта регистра),
    номера карт/телефонов/ИИН/БИН в другом форматировании (по цифрам) и обрезанные
    начала/концы таких номеров длиной от 4 цифр.
    """
    return _KnownValues().sub(text, mapping, stats)


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
        self._known = _KnownValues()

    def mask(self, text: str) -> str:
        """Детекторы ПДн + поиск уже известных значений.

        Второй шаг нужен потому, что ответы LLM восстанавливаются (unmask) и снова попадают
        в промпты следующих агентов — уже без контекста («ТОО «…»», «ИИН:»), в другом
        форматировании или обрезанными до фиксированной длины.
        """
        masked, _ = mask_text(text, self.mapping, orgs=self.orgs, stats=self.masked_stats)
        return self._known.sub(masked, self.mapping, self.masked_stats)

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
