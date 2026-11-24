"""Утечки ПДн в режиме конфиденциальности и валидация требований LLM-Reader.

* известные значения (организации, ИИН, карты, телефоны) маскируются и без контекста детекторов —
  после unmask ответы LLM снова попадают в промпты следующих агентов;
* переформатированные и обрезанные номера (префикс/суффикс от 4 цифр) тоже маскируются;
* Reader обрезает строки по границе слова, а не посреди числа;
* требования LLM-Reader со ссылкой на несуществующую/пустую строку отбрасываются;
* сквозной тест: фейковая LLM через оркестратор, privacy="on" — ни один промпт не содержит
  оригинальное название организации, ИИН, PAN или их обрезки.
"""
from __future__ import annotations

import asyncio
import re
import sys
from collections import Counter
from pathlib import Path

import pytest
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.reader import ReaderAgent, clip_words, merge_models, valid_requirements  # noqa: E402
from config import LLMConfig  # noqa: E402
from models import DocumentModel, Requirement  # noqa: E402
from parsing import load_text  # noqa: E402
from privacy import PrivacyLLMClient, iin_checksum_ok, mask_known, mask_text, unmask_text  # noqa: E402


def _with_check_digit(prefix11: str) -> str:
    for c in "0123456789":
        if iin_checksum_ok(prefix11 + c):
            return prefix11 + c
    raise AssertionError(prefix11)


IIN = _with_check_digit("88072550031")
PAN = "4111 1111 1111 1111"
PHONE = "+7 701 123 45 67"
ORG = "Ромашка Финанс"


def _primed(text: str) -> dict[str, str]:
    _, mp = mask_text(text)
    return mp


SOURCE = f"Заказчик ТОО «{ORG}». ИИН: {IIN}, карта {PAN}, тел. {PHONE}, почта ops@romashka.kz"


# ── #1: известные значения без контекста ──────────────────────────────────────
class TestKnownValues:
    def test_org_without_legal_form_is_masked(self):
        mp = _primed(SOURCE)
        out = mask_known(f"Платформа переводов {ORG} для клиентов", mp)
        assert ORG not in out and "[ОРГ_1]" in out

    def test_org_with_other_whitespace(self):
        mp = _primed(SOURCE)
        out = mask_known("Сервис Ромашка Финанс и Ромашка  Финанс", mp)
        assert "Ромашка" not in out and out.count("[ОРГ_1]") == 2

    def test_boundary_safe_short_value(self):
        mp = {"[ОРГ_1]": "Банк"}
        out = mask_known("Банкомат и Банковский счёт; Банк выдаёт карту", mp)
        assert out == "Банкомат и Банковский счёт; [ОРГ_1] выдаёт карту"

    def test_longest_value_first(self):
        mp = {"[ОРГ_1]": "Банк", "[ОРГ_2]": "Банк Восток"}
        assert mask_known("клиент Банк Восток и Банк", mp) == "клиент [ОРГ_2] и [ОРГ_1]"

    def test_single_pass_no_cascade_into_placeholders(self):
        # значение совпадает с частью плейсхолдера — сами плейсхолдеры не портятся
        mp = {"[ОРГ_1]": "ОРГ", "[ОРГ_2]": "Восток"}
        out = mask_known("[ОРГ_2] и ОРГ_2, компания ОРГ, Восток", mp)
        assert out == "[ОРГ_2] и ОРГ_2, компания [ОРГ_1], [ОРГ_2]"
        assert unmask_text(out, mp) == "Восток и Восток, компания ОРГ, Восток"

    def test_email_case_insensitive(self):
        mp = _primed(SOURCE)
        assert mask_known("пишите OPS@Romashka.KZ", mp) == "пишите [EMAIL_1]"

    @pytest.mark.parametrize("variant", ["4111-1111-1111-1111", "4111111111111111", "4111 1111  1111 1111"])
    def test_reformatted_card(self, variant):
        mp = _primed(SOURCE)
        assert mask_known(f"карта {variant} в логе", mp) == "карта [КАРТА_1] в логе"

    @pytest.mark.parametrize("variant", ["87011234567", "+77011234567", "(701) 123-45-67", "7011234567",
                                         "8 (701) 123 45 67"])
    def test_reformatted_phone(self, variant):
        mp = _primed(SOURCE)
        assert mask_known(f"звонок на {variant} утром", mp) == "звонок на [ТЕЛ_1] утром"

    def test_reformatted_iin(self):
        mp = _primed(SOURCE)
        spaced = f"{IIN[:4]} {IIN[4:8]} {IIN[8:]}"
        assert mask_known(f"клиент {spaced}", mp) == "клиент [ИИН_1]"

    def test_unrelated_numbers_untouched(self):
        mp = _primed(SOURCE)
        text = "L12: лимит 100 000 тенге, SLA 99.9%, код 8000, версия 2.4.1, FR-1234, 30 дней"
        assert mask_known(text, mp) == text

    def test_stats_counted(self):
        inner = _Recorder()
        client = PrivacyLLMClient(inner)
        client.prime(SOURCE)
        asyncio.run(client.chat_json("", f"Продукт {ORG}; карта 4111-1111-1111-1111", _Out))
        assert client.masked_stats["ОРГ"] == 1 and client.masked_stats["КАРТА"] == 1
        assert ORG not in inner.sent[0][1] and "4111" not in inner.sent[0][1]


# ── #8: обрезанные номера ─────────────────────────────────────────────────────
class TestFragments:
    def test_iin_prefix_after_truncation(self):
        mp = _primed(SOURCE)
        out = mask_known(f'"с полями ФИО и ИИН (к): {IIN[:7]}", "FR-04', mp)
        assert IIN[:7] not in out and "[ИИН_1]" in out

    def test_iin_suffix_after_truncation(self):
        mp = _primed(SOURCE)
        out = mask_known(f"…{IIN[-6:]} в разделе профиля", mp)
        assert IIN[-6:] not in out and out.startswith("…[ИИН_1]")

    def test_pan_prefix_with_groups(self):
        mp = _primed(SOURCE)
        out = mask_known("сохраняет 100 4111 1111 11", mp)
        assert out == "сохраняет 100 [КАРТА_1]"

    def test_short_runs_not_masked(self):
        mp = _primed(SOURCE)
        assert mask_known(f"код {IIN[:3]} и шаг 1", mp) == f"код {IIN[:3]} и шаг 1"

    def test_line_numbers_not_masked(self):
        mp = {"[ИИН_1]": "123456789012"}
        assert mask_known("L1234: строка", mp) == "L1234: строка"


class TestReaderTruncation:
    LINE = f"FR-03. Система должна показывать клиенту карточку заёмщика с полями ФИО и ИИН (к): {IIN} в разделе профиля."

    def test_clip_words_does_not_split_numbers(self):
        cut = clip_words(self.LINE, 90)
        assert len(cut) <= 90 and IIN[:7] not in cut and cut.endswith("(к):")
        assert clip_words("короткая", 90) == "короткая"
        assert clip_words("abc def", 5) == "abc"
        assert clip_words("1234567890", 5) == "12345"      # нечего сохранить — жёсткий срез
        assert clip_words("ИИН:880725500313", 9) == "ИИН:"

    def test_rules_model_features_and_requirements(self):
        doc = load_text(f"# ТЗ\n{self.LINE}\n{'FR-04. Система должна хранить ' + PAN + ' ' + 'x' * 200}\n")
        from agents.base import Context
        ctx = Context(doc=doc, cfg=LLMConfig())
        dm = ReaderAgent()._rules_model(ctx)
        blob = " ".join(dm.features + [r.text for r in dm.requirements])
        digits_runs = [re.sub(r"\D", "", m) for m in re.findall(r"\d[\d \-]*\d", blob)]
        for run in digits_runs:
            assert run in (IIN, re.sub(r"\D", "", PAN)) or len(run) < 4 or not (
                IIN.startswith(run) or re.sub(r"\D", "", PAN).startswith(run)), blob


# ── #24: валидация строк требований LLM-Reader ───────────────────────────────
DOC_LINES = ["# ТЗ", "", "FR-01. Система должна регистрировать клиента.", "FR-02. Система должна логировать вход."]


class TestRequirementLines:
    def test_valid_requirements_filter(self):
        reqs = [Requirement(id="a", text="ok", line=3), Requirement(id="b", text="empty", line=2),
                Requirement(id="c", text="out", line=700), Requirement(id="d", text="zero", line=0),
                Requirement(id="e", text="neg", line=-1), Requirement(id="f", text="no line", line=None)]
        assert [r.id for r in valid_requirements(reqs, DOC_LINES)] == ["a", "f"]

    def test_merge_falls_back_to_rules_when_all_invalid(self):
        rules = DocumentModel(title="t", requirements=[Requirement(id="FR-01", text="r", line=3)])
        llm = DocumentModel(title="t", requirements=[Requirement(id=f"X{i}", text="x", line=700 + i) for i in range(12)])
        dm = merge_models(rules, llm, DOC_LINES)
        assert [r.id for r in dm.requirements] == ["FR-01"]

    def test_merge_keeps_valid_llm_requirements(self):
        rules = DocumentModel(title="t", requirements=[Requirement(id="FR-01", text="r", line=3)])
        llm = DocumentModel(title="t", requirements=[Requirement(id="L1", text="x", line=4),
                                                     Requirement(id="L2", text="x", line=99)])
        assert [r.id for r in merge_models(rules, llm, DOC_LINES).requirements] == ["L1"]
        # без строк документа — прежнее поведение (обратная совместимость)
        assert len(merge_models(rules, llm).requirements) == 2

    def test_run_llm_filters_requirement_lines(self, monkeypatch):
        from agents.base import Context
        doc = load_text("\n".join(DOC_LINES) + "\n")
        ctx = Context(doc=doc, cfg=LLMConfig())
        agent = ReaderAgent()
        agent.run_rules(ctx)

        async def fake_ask(self, ctx, system, user, schema):
            return schema.model_validate({"title": "Demo", "requirements": [
                {"id": f"R{i}", "text": "x", "line": 700 + i} for i in range(5)]})

        monkeypatch.setattr(ReaderAgent, "ask", fake_ask)
        asyncio.run(agent.run_llm(ctx))
        lines = [r.line for r in ctx.doc_model.requirements]
        assert lines and all(1 <= ln <= len(doc.lines) and doc.line(ln).strip() for ln in lines)


# ── Сквозной тест: фейковая LLM через оркестратор ────────────────────────────
class _Out(BaseModel):
    ok: bool = True


class _Recorder:
    active_model = "fake"
    cache_hits = 0
    transient_errors = 0

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.switches: list[str] = []

    async def chat_json(self, system, user, schema):
        self.sent.append((system, user))
        return schema.model_validate({}), 1


E2E_TEXT = f"""# ТЗ: Переводы для ТОО «{ORG}»

## 1. Цели
Система должна обеспечить переводы между клиентами быстро и удобно.

## 2. Требования
FR-01: Тестовый клиент с ИИН {IIN} должен видеть баланс быстро.
FR-02: Уведомление на номер {PHONE} отправляется по возможности.
FR-03. Система должна показывать клиенту карточку заёмщика с полями ФИО и ИИН (к): {IIN} в разделе профиля.
FR-04: Карта {PAN} используется для тестов, лимит определяется позже (TBD).
FR-05. Система должна сохранять номер карты в логах сервиса для разбора инцидентов по карте {PAN}.
"""


def _fragment_leaks(prompt: str) -> list[str]:
    """Цифровые серии от 4 цифр, являющиеся подстрокой ИИН или PAN (обрезки номеров)."""
    secrets = (IIN, re.sub(r"\D", "", PAN), "7011234567")
    leaks = []
    for m in re.finditer(r"(?<![\w\[])\d+(?:[ \t\-]\d+)*", prompt):
        run = re.sub(r"\D", "", m.group(0))
        for i in range(len(run)):
            for j in range(i + 4, len(run) + 1):
                if any(run[i:j] in s for s in secrets):
                    leaks.append(m.group(0))
                    break
            else:
                continue
            break
    return leaks


def test_end_to_end_no_leaks_through_orchestrator(monkeypatch):
    import agents.orchestrator as orch

    sent: list[str] = []

    class FakeLLM:
        active_model = "fake-model"
        cache_hits = 0
        transient_errors = 0

        def __init__(self, cfg):
            self.cfg = cfg
            self.switches: list[str] = []

        async def chat_json(self, system, user, schema):
            sent.append(system + "\n" + user)
            name = schema.__name__
            org = (re.search(r"\[ОРГ_\d+\]", user) or re.search(r"\[ОРГ_\d+\]", system))
            org_ph = org.group(0) if org else "[ОРГ_1]"
            card = re.search(r"\[КАРТА_\d+\]", user)
            card_ph = card.group(0) if card else "[КАРТА_1]"
            iin = re.search(r"\[ИИН_\d+\]", user)
            iin_ph = iin.group(0) if iin else "[ИИН_1]"
            lines = [l for l in user.splitlines() if re.match(r"L\d+: ", l)]
            if name in ("RawFindings", "RawFintech"):
                fs = []
                for l in lines:
                    m = re.match(r"L(\d+): (.*)", l)
                    if "[" in m.group(2) and "_" in m.group(2):
                        fs.append({"category": "ambiguity", "severity": "major",
                                   # LLM перефразирует без «ТОО «…»» и без «ИИН:»
                                   "title": f"Размытость у {org_ph}: номер {iin_ph}",
                                   "description": f"Карта {card_ph} клиента {org_ph} в строке {m.group(1)}",
                                   "evidence": [m.group(2)], "lines": [int(m.group(1))], "confidence": 0.9,
                                   "suggested_text": f"Для {org_ph} указать точную формулировку"})
                data: dict = {"findings": fs}
                if name == "RawFintech":
                    data["controls"] = []
                return schema.model_validate(data), 10
            if name == "DocumentModel":
                return schema.model_validate({
                    "title": f"Переводы {org_ph}",
                    "summary": f"Сервис {org_ph} для клиента {iin_ph} с картой {card_ph}.",
                    "domains": ["transfers"],
                    "features": [f"Карта {card_ph} клиента {org_ph}"],
                    "requirements": [{"id": "X-1", "text": "фантом", "line": 999}],
                }), 10
            if name == "RawQuestions":
                ids = re.findall(r"\[(F-\d+)\]", user)
                return schema.model_validate({"questions": [{
                    "question": f"Какой SLA у {org_ph}?", "related_findings": ids[:1],
                    "answer_options": ["1 c", "2 c"], "priority": "P1"}]}), 10
            if name == "SummaryOut":
                return schema.model_validate({"summary": f"Резюме по {org_ph}", "strengths": []}), 10
            if name == "RawRevision":
                return schema.model_validate({"findings": [], "withdrawn": []}), 10
            return schema.model_validate({}), 10

    monkeypatch.setattr(orch, "LLMClient", FakeLLM)
    cfg = LLMConfig(api_key="x", base_url="https://api.example.com/v1", privacy="on", cache=False)
    doc = load_text(E2E_TEXT, "t.md")
    r = orch.analyze(doc, cfg, use_llm=True)

    assert sent, "фейковая LLM не вызывалась"
    assert len(sent) >= 5
    pan_digits = re.sub(r"\D", "", PAN)
    for i, prompt in enumerate(sent):
        for secret in (ORG, "Ромашка", IIN, PAN, pan_digits, PHONE, "ops@romashka.kz"):
            assert secret not in prompt, f"утечка {secret!r} в промпте #{i}"
        assert not _fragment_leaks(prompt), f"обрезок номера в промпте #{i}: {_fragment_leaks(prompt)}"
    # результат для пользователя восстановлен
    assert ORG in r.document.title
    # фантомное требование LLM-Reader (L999) отброшено
    assert all(q.line is None or 1 <= q.line <= len(doc.lines) for q in r.document.requirements)
    assert r.document.requirements


def test_privacy_note_counts_known_value_hits():
    inner = _Recorder()
    client = PrivacyLLMClient(inner)
    client.prime(SOURCE)
    before = Counter(client.masked_stats)
    asyncio.run(client.chat_json(f"О {ORG}", f"ИИН {IIN[:7]}", _Out))
    gained = Counter(client.masked_stats) - before
    assert gained["ОРГ"] == 1 and gained["ИИН"] == 1
    assert "ИИН: 1" in client.privacy_note and "ОРГ: 1" in client.privacy_note
