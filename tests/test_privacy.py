"""Режим конфиденциальности: маскирование ПДн, обратимость, обёртка LLM-клиента, конфиг."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROVIDER_PRESETS, LLMConfig, is_local_url, preset_overrides  # noqa: E402
from privacy import (  # noqa: E402
    PrivacyLLMClient,
    iin_checksum_ok,
    is_valid_bin,
    is_valid_iin,
    luhn_ok,
    mask_text,
    unmask_obj,
    unmask_text,
)


def _with_check_digit(prefix11: str) -> str:
    """Достраивает 11 цифр до валидного ИИН/БИН (подбор контрольного разряда)."""
    for c in "0123456789":
        if iin_checksum_ok(prefix11 + c):
            return prefix11 + c
    raise AssertionError(f"для {prefix11} нет валидного контрольного разряда")


VALID_IIN = _with_check_digit("88072550031")      # 25.07.1988, мужчина XX века
VALID_IIN_2 = _with_check_digit("01120360042")    # 03.12.2001, XXI век
VALID_BIN = _with_check_digit("05084000123")      # авг. 2005, ЮЛ-резидент


def _bad_check(iin: str) -> str:
    return iin[:11] + str((int(iin[11]) + 1) % 10)


# ── Валидаторы ────────────────────────────────────────────────────────────────
class TestValidators:
    def test_iin_valid(self):
        assert is_valid_iin(VALID_IIN)
        assert is_valid_iin(VALID_IIN_2)

    def test_iin_bad_checksum(self):
        assert not is_valid_iin(_bad_check(VALID_IIN))
        assert is_valid_iin(_bad_check(VALID_IIN), strict=False)   # дата правдоподобна

    @pytest.mark.parametrize("value", [
        "881325500310",   # месяц 13
        "880700500310",   # день 00
        "880725900310",   # 7-й разряд 9
        "12345678901",    # 11 цифр
        "88072550031a",
    ])
    def test_iin_invalid_structure(self, value):
        assert not is_valid_iin(value, strict=False)

    def test_bin(self):
        assert is_valid_bin(VALID_BIN)
        assert not is_valid_iin(VALID_BIN, strict=False)          # 5–6 разряды БИН ≠ день рождения
        assert not is_valid_bin(VALID_IIN)

    @pytest.mark.parametrize("pan", ["4111111111111111", "5500005555555559", "378282246310005",
                                     "6011111111111117", "4000056655665556", "2200000000000004"])
    def test_luhn_ok(self, pan):
        assert luhn_ok(pan)

    @pytest.mark.parametrize("pan", ["4111111111111112", "1234567812345678", "abc"])
    def test_luhn_bad(self, pan):
        assert not luhn_ok(pan)


# ── Маскирование по типам ─────────────────────────────────────────────────────
class TestMasking:
    def test_iin_valid_without_keyword(self):
        masked, mp = mask_text(f"Получатель {VALID_IIN} подтверждён")
        assert masked == "Получатель [ИИН_1] подтверждён"
        assert mp == {"[ИИН_1]": VALID_IIN}

    def test_iin_with_keyword_but_bad_checksum_still_masked(self):
        # тестовые ИИН в ТЗ часто «выдуманы» — при ключевом слове достаточно правдоподобной даты
        fake = _bad_check(VALID_IIN_2)
        masked, _ = mask_text(f"ИИН заёмщика: {fake}")
        assert fake not in masked and "[ИИН_1]" in masked

    def test_random_12_digits_not_masked(self):
        for num in ("123456789012", "999999999999", _bad_check(VALID_IIN)):
            masked, mp = mask_text(f"Номер заявки {num} в очереди")
            assert num in masked and not mp

    def test_bin_with_keyword(self):
        masked, mp = mask_text(f"БИН контрагента — {VALID_BIN}")
        assert masked == "БИН контрагента — [БИН_1]" and mp["[БИН_1]"] == VALID_BIN

    @pytest.mark.parametrize("pan", [
        "4111111111111111", "4111 1111 1111 1111", "4111-1111-1111-1111",
        "3782 822463 10005", "5500 0055 5555 5559", "4000 0566 5566 5556",
    ])
    def test_card_formats(self, pan):
        masked, mp = mask_text(f"Оплата картой {pan}, сумма 1000 ₸")
        assert masked == "Оплата картой [КАРТА_1], сумма 1000 ₸"
        assert mp["[КАРТА_1]"] == pan

    @pytest.mark.parametrize("text", [
        "Номер договора 1234567812345678",          # 16 цифр, но не по Луна
        "сумма 1 000 000 000 000 тенге",            # группы по 3 — не карта
        "маска 4111 11** **** 1111",                 # уже замаскирована
        "идентификатор 0000000000000000",            # все нули проходят Луна, но не карта
    ])
    def test_card_false_positives(self, text):
        masked, mp = mask_text(text)
        assert masked == text and not mp

    @pytest.mark.parametrize("phone", [
        "+7 701 123 45 67", "+7 (701) 123-45-67", "+77011234567", "87011234567",
        "8 (727) 250-11-22", "8-707-555-44-33", "+7 717 255 66 77",
        "8 800 080 12 34",
    ])
    def test_phone_formats(self, phone):
        masked, mp = mask_text(f"Звонить по номеру {phone} в рабочее время")
        assert masked == "Звонить по номеру [ТЕЛ_1] в рабочее время", masked
        assert mp["[ТЕЛ_1]"] == phone

    @pytest.mark.parametrize("text", [
        "версия 8.1.2 от 2024-07-15", "лимит 800 000 тенге", "SLA 99.9%, RTO 4 ч",
        "приложение 7 к договору 123", "код ответа 8000",
    ])
    def test_phone_false_positives(self, text):
        masked, mp = mask_text(text)
        assert masked == text and not mp

    def test_email(self):
        masked, mp = mask_text("Пишите на a.seitkali@my-bank.kz или support+tz@example.com.")
        assert masked == "Пишите на [EMAIL_1] или [EMAIL_2]."
        assert set(mp.values()) == {"a.seitkali@my-bank.kz", "support+tz@example.com"}

    @pytest.mark.parametrize("iban", ["KZ86125KZT5004100100", "KZ86 125K ZT50 0410 0100", "KZ12 3456 7890 1234 5678"])
    def test_iban(self, iban):
        masked, mp = mask_text(f"Счёт получателя: {iban}; банк-бенефициар")
        assert masked == "Счёт получателя: [IBAN_1]; банк-бенефициар"
        assert mp["[IBAN_1]"] == iban

    def test_orgs_quoted_and_ip(self):
        text = 'Заказчик АО «Банк Восток», исполнитель ТОО "Дата Софт", субподряд ИП Сейткали А.Б.'
        masked, mp = mask_text(text)
        assert masked == 'Заказчик АО «[ОРГ_1]», исполнитель ТОО "[ОРГ_2]", субподряд ИП [ОРГ_3]'
        assert mp["[ОРГ_1]"] == "Банк Восток" and mp["[ОРГ_3]"] == "Сейткали А.Б."

    def test_orgs_optional_and_public_infra_kept(self):
        text = "Обмен с АО «Кредитный партнёр» и АО «Первое кредитное бюро»"
        assert mask_text(text, orgs=False)[0] == text
        masked, _ = mask_text(text)
        assert "Первое кредитное бюро" in masked and "Кредитный партнёр" not in masked

    def test_no_pii_text_unchanged(self):
        text = "Система должна обрабатывать 1000 TPS.\nВремя ответа — не более 2 с (p95).\nL12: FR-07"
        assert mask_text(text) == (text, {})

    def test_same_value_same_placeholder_and_shared_mapping(self):
        mp: dict[str, str] = {}
        a, _ = mask_text("клиент a@b.kz, копия a@b.kz", mp)
        b, _ = mask_text("ответ на a@b.kz и c@d.kz", mp)
        assert a == "клиент [EMAIL_1], копия [EMAIL_1]"
        assert b == "ответ на [EMAIL_1] и [EMAIL_2]"

    def test_literal_placeholder_in_source_not_reused(self):
        masked, mp = mask_text("В шаблоне пишут [EMAIL_1], реальный адрес x@y.kz")
        assert masked == "В шаблоне пишут [EMAIL_1], реальный адрес [EMAIL_2]"
        assert unmask_text(masked, mp) == "В шаблоне пишут [EMAIL_1], реальный адрес x@y.kz"

    def test_line_count_and_line_numbers_preserved(self):
        lines = [
            f"L1: ИИН {VALID_IIN}, карта 4111 1111 1111 1111",
            "L2: телефон +7 (701) 123-45-67",
            "",
            "L4: e-mail ops@bank.kz; IBAN KZ86 125K ZT50 0410 0100",
            "L5: АО «Банк Восток»\r",
            "L6: обычная строка",
        ]
        text = "\n".join(lines) + "\n"
        masked, mp = mask_text(text)
        assert masked.count("\n") == text.count("\n")
        assert len(masked.splitlines()) == len(text.splitlines())
        for orig, new in zip(text.splitlines(), masked.splitlines()):
            assert new.split(":")[0] == orig.split(":")[0]
        assert unmask_text(masked, mp) == text
        for value in (VALID_IIN, "4111 1111 1111 1111", "ops@bank.kz", "Банк Восток"):
            assert value not in masked


# ── Восстановление ────────────────────────────────────────────────────────────
class Finding(BaseModel):
    title: str
    quote: str
    lines: list[int] = []


class FindingList(BaseModel):
    findings: list[Finding]
    meta: dict[str, str] = {}


class TestUnmask:
    def test_nested_json_roundtrip(self):
        text = f"ИИН {VALID_IIN} передаётся на ops@bank.kz по телефону 87011234567"
        masked, mp = mask_text(text)
        llm_json = {
            "findings": [
                {"title": "ИИН в открытом виде", "quote": masked.split(" передаётся")[0], "lines": [3]},
                {"title": "Канал", "quote": "[EMAIL_1]", "tags": ("[ТЕЛ_1]", 5)},
            ],
            "by_value": {"[ИИН_1]": ["[EMAIL_1]", None, 1.5]},
        }
        out = unmask_obj(llm_json, mp)
        assert out["findings"][0]["quote"] == f"ИИН {VALID_IIN}"
        assert out["findings"][1]["quote"] == "ops@bank.kz"
        assert out["findings"][1]["tags"] == ("87011234567", 5)
        assert out["by_value"] == {VALID_IIN: ["ops@bank.kz", None, 1.5]}
        assert out["findings"][0]["lines"] == [3]
        assert unmask_obj(masked, mp) == text

    def test_pydantic_model(self):
        mp = {"[КАРТА_1]": "4111111111111111", "[ОРГ_1]": "Банк Восток"}
        model = FindingList(findings=[Finding(title="PAN [КАРТА_1] в логах", quote="АО «[ОРГ_1]»", lines=[7])],
                            meta={"src": "[ОРГ_1]"})
        out = unmask_obj(model, mp)
        assert isinstance(out, FindingList)
        assert out.findings[0].title == "PAN 4111111111111111 в логах"
        assert out.findings[0].quote == "АО «Банк Восток»"
        assert out.meta == {"src": "Банк Восток"}

    def test_model_lost_brackets_and_unknown_placeholders(self):
        mp = {"[ТЕЛ_1]": "+7 701 000 00 00"}
        assert unmask_text("звонок на ТЕЛ_1 и [ТЕЛ_9]", mp) == "звонок на +7 701 000 00 00 и [ТЕЛ_9]"

    def test_no_chained_replacement(self):
        # значение, похожее на плейсхолдер, не раскрывается повторно
        mp = {"[ОРГ_1]": "[EMAIL_1]", "[EMAIL_1]": "x@y.kz"}
        assert unmask_text("[ОРГ_1]", mp) == "[EMAIL_1]"

    def test_non_str_passthrough(self):
        mp = {"[ИИН_1]": VALID_IIN}
        assert unmask_obj(42, mp) == 42 and unmask_obj(None, mp) is None
        assert unmask_obj("[ИИН_1]", {}) == "[ИИН_1]"


# ── Обёртка LLM-клиента ───────────────────────────────────────────────────────
class FakeInner:
    """Имитирует LLMClient: запоминает, что ушло «наружу», и цитирует строку документа."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.active_model = "fake-model"
        self.switches = ["a → b (429)"]
        self.transient_errors = 2
        self.cache_hits = 1

    async def chat_json(self, system, user, schema):
        self.sent.append((system, user))
        first = user.splitlines()[0]
        return schema.model_validate({"findings": [{"title": f"Найдено в {first}", "quote": first, "lines": [1]}]}), 123


class TestPrivacyClient:
    DOC = f"L1: Клиент ИИН {VALID_IIN}, тел. +7 701 123 45 67\nL2: Карта 4111111111111111 пишется в лог"

    def test_masks_outgoing_and_unmasks_result(self):
        inner = FakeInner()
        client = PrivacyLLMClient(inner)
        result, tokens = asyncio.run(client.chat_json("Контакт аудитора: audit@bank.kz", self.DOC, FindingList))
        system_sent, user_sent = inner.sent[0]
        for secret in (VALID_IIN, "+7 701 123 45 67", "4111111111111111", "audit@bank.kz"):
            assert secret not in user_sent and secret not in system_sent
        assert "[ИИН_1]" in user_sent and "[EMAIL_1]" in system_sent
        assert len(user_sent.splitlines()) == len(self.DOC.splitlines())
        assert tokens == 123
        assert isinstance(result, FindingList)
        assert result.findings[0].quote == self.DOC.splitlines()[0]

    def test_delegates_attributes(self):
        inner = FakeInner()
        client = PrivacyLLMClient(inner)
        assert client.active_model == "fake-model"
        assert client.switches == ["a → b (429)"]
        assert client.transient_errors == 2 and client.cache_hits == 1
        inner.cache_hits = 5
        assert client.cache_hits == 5
        with pytest.raises(AttributeError):
            _ = client.no_such_attribute

    def test_consistent_placeholders_across_calls(self):
        inner = FakeInner()
        client = PrivacyLLMClient(inner)
        client.prime(self.DOC)
        asyncio.run(client.chat_json("", "L2: Карта 4111111111111111 пишется в лог", FindingList))
        asyncio.run(client.chat_json("", self.DOC, FindingList))
        assert "[КАРТА_1]" in inner.sent[0][1] and "[КАРТА_1]" in inner.sent[1][1]
        assert "[ИИН_1]" in inner.sent[1][1]
        assert "ИИН: 1" in client.privacy_note and "КАРТА: 1" in client.privacy_note

    def test_privacy_note_without_pii(self):
        client = PrivacyLLMClient(FakeInner())
        asyncio.run(client.chat_json("sys", "L1: без персональных данных", FindingList))
        assert "не найдены" in client.privacy_note


# ── Конфигурация ──────────────────────────────────────────────────────────────
class TestConfig:
    @pytest.mark.parametrize("url", [
        "http://localhost:11434/v1", "http://127.0.0.1:8000/v1", "http://[::1]:11434/v1",
        "http://llm.local/v1", "http://host.docker.internal:11434/v1", "http://10.20.30.40:8000/v1",
        "http://192.168.1.5/v1", "localhost:11434",
    ])
    def test_local_urls(self, url):
        assert is_local_url(url)
        cfg = LLMConfig(base_url=url, api_key="k")
        assert cfg.is_local and not cfg.privacy_enabled

    @pytest.mark.parametrize("url", [
        None, "", "https://api.openai.com/v1", "https://generativelanguage.googleapis.com/v1beta/openai/",
        "http://8.8.8.8/v1", "https://localhost.evil.com/v1",
    ])
    def test_remote_urls(self, url):
        assert not is_local_url(url)
        assert LLMConfig(base_url=url, api_key="k").privacy_enabled

    def test_explicit_modes(self):
        assert LLMConfig(base_url="http://localhost:11434/v1", privacy="on").privacy_enabled
        assert not LLMConfig(base_url="https://api.openai.com/v1", privacy="off").privacy_enabled
        assert LLMConfig(base_url="https://api.openai.com/v1", privacy="garbage").privacy_enabled

    def test_from_env(self, monkeypatch):
        for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_FALLBACK_MODELS", "LLM_MAX_CONCURRENCY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("LLM_PRIVACY", "OFF")
        assert LLMConfig.from_env().privacy == "off"
        monkeypatch.setenv("LLM_PRIVACY", "true")
        assert LLMConfig.from_env().privacy == "on"
        monkeypatch.delenv("LLM_PRIVACY")
        cfg = LLMConfig.from_env()
        assert cfg.privacy == "auto"
        # прежние поля и значения по умолчанию не изменились
        assert cfg.model == "gpt-4o-mini" and cfg.max_concurrency == 3 and cfg.fallback_models == ()
        assert cfg.with_overrides(privacy="on", model=None).privacy == "on"

    def test_presets(self):
        assert {"Офлайн-правила", "Локальная модель (Ollama, контур банка)", "Google Gemini", "OpenAI",
                "shai.pro (уточнить на кикоффе)"} <= set(PROVIDER_PRESETS)
        base = LLMConfig(api_key=None, model="m0", fallback_models=("x",))
        ollama = base.with_preset("Локальная модель (Ollama, контур банка)")
        assert ollama.base_url == "http://localhost:11434/v1" and ollama.model == "qwen2.5:14b-instruct"
        assert ollama.max_concurrency == 1 and ollama.fallback_models == () and ollama.enabled
        assert ollama.is_local and not ollama.privacy_enabled
        gemini = base.with_preset("Google Gemini")
        assert gemini.model == "gemini-3.5-flash-lite"
        assert gemini.models == ["gemini-3.5-flash-lite", "gemini-flash-lite-latest",
                                 "gemini-3.1-flash-lite", "gemini-3-flash-preview"]
        assert gemini.privacy_enabled
        shai = base.with_preset("shai.pro (уточнить на кикоффе)")
        assert shai.model == "m0"                       # пустые поля пресета не затирают настройки
        assert base.with_preset("Офлайн-правила") == base
        assert preset_overrides("нет такого") == {}
        assert "offline" not in preset_overrides("Офлайн-правила")
