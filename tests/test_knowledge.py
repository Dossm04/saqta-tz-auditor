"""Тесты базы знаний: целостность контролей и разделов, актуальность ссылок на нормы,
и признаки новых контролей на «свежих» формулировках (не из демо-ТЗ)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.rules import check_control  # noqa: E402
from knowledge import antipatterns  # noqa: E402
from knowledge.domains import DOMAIN_LABELS  # noqa: E402
from knowledge.fintech_controls import (  # noqa: E402
    CONTROLS, CONTROLS_BY_ID, REGULATION_WHITELIST, applicable_controls,
)
from knowledge.tz_structure import SECTIONS  # noqa: E402
from parsing import load_text  # noqa: E402

PROMPTS = ROOT / "prompts"


# ── Целостность ───────────────────────────────────────────────────────────────

def test_ids_unique():
    assert len({c.id for c in CONTROLS}) == len(CONTROLS)
    assert len({s.id for s in SECTIONS}) == len(SECTIONS)


@pytest.mark.parametrize("c", CONTROLS, ids=lambda c: c.id)
def test_control_has_options_and_default(c):
    assert len(c.answer_options) >= 2, c.id
    assert len(set(c.answer_options)) == len(c.answer_options), c.id
    assert all(o.strip() for o in c.answer_options)
    assert c.default_assumption.strip(), c.id
    assert c.severity in ("critical", "major", "minor")
    assert c.category in ("security", "compliance", "fintech")
    assert set(c.applies_to) <= set(DOMAIN_LABELS) | {"*"}, c.id
    assert 1 <= c.min_hits_covered <= len(c.patterns)


@pytest.mark.parametrize("s", SECTIONS, ids=lambda s: s.id)
def test_section_has_options_and_default(s):
    assert len(s.answer_options) >= 2, s.id
    assert len(set(s.answer_options)) == len(s.answer_options), s.id
    assert s.default_assumption.strip(), s.id


def test_all_regexes_compile():
    for c in CONTROLS:
        for p in c.patterns:
            re.compile(p, re.IGNORECASE)
    for s in SECTIONS:
        for p in s.patterns:
            re.compile(p, re.IGNORECASE)
    for ap in getattr(antipatterns, "ANTIPATTERNS", []):
        re.compile(ap.pattern, re.IGNORECASE)
        if getattr(ap, "unless", None):
            re.compile(ap.unless, re.IGNORECASE)


# ── Актуальность норм ─────────────────────────────────────────────────────────

def _kb_texts():
    for c in CONTROLS:
        yield c.id, " ".join([c.regulation, c.why, c.recommendation, c.suggested_text, c.question,
                              c.default_assumption, *c.answer_options])


def test_no_repealed_nbrk_48():
    for cid, text in _kb_texts():
        assert "НБРК №48" not in text and "№48 от 27.03.2018" not in text, cid
    for p in PROMPTS.glob("*.md"):
        assert "№48" not in p.read_text(encoding="utf-8"), p.name


def test_current_standard_versions_pinned():
    for cid, text in _kb_texts():
        if "PCI DSS" in text:
            assert re.search(r"PCI DSS v4\.0(?!\.1)", text) is None, cid
        if "ASVS" in text:
            assert re.search(r"ASVS 4\.0(?!\.3)", text) is None, cid
    assert "АРРФР №53" in CONTROLS_BY_ID["SEC-AUTH-01"].regulation


def test_uncertain_acts_are_hedged():
    """Новые и поэтапно вводимые акты должны сопровождаться оговоркой о проверке редакции."""
    for cid, text in _kb_texts():
        for act in ("258-VIII", "205-VIII", "255-VIII", "Правила проведения биометрической"):
            if act in text:
                assert "проверьте" in text, (cid, act)


def test_whitelist_is_in_common_prompt():
    common = (PROMPTS / "_common.md").read_text(encoding="utf-8")
    for name in REGULATION_WHITELIST:
        assert name in common, name
    assert "<tz>" in common


# ── Применимость (нет шума на несвязанных ТЗ) ─────────────────────────────────

def test_lending_controls_only_for_lending():
    ids = {c.id for c in applicable_controls(["transfers", "cards", "mobile"])}
    assert not {"LEND-01", "LEND-COOL-01", "LEND-BAN-01", "LEND-AUTO-01", "REG-ESIGN-01"} & ids
    assert {"TX-AFC-01", "APPSEC-DBO-01", "PRIV-04"} <= ids
    lend = {c.id for c in applicable_controls(["lending"])}
    assert {"LEND-COOL-01", "LEND-BAN-01", "LEND-AUTO-01", "REG-ESIGN-01"} <= lend
    assert "TX-AFC-01" not in lend and "APPSEC-DBO-01" not in lend
    api_only = {c.id for c in applicable_controls(["api"])}
    assert "APPSEC-DBO-01" not in api_only and "PRIV-04" not in api_only


# ── Признаки новых контролей на свежих формулировках ─────────────────────────

def _status(cid: str, text: str) -> str:
    return check_control(load_text(text), CONTROLS_BY_ID[cid]).status


@pytest.mark.parametrize("cid,text,expected", [
    ("TX-AFC-01", "Банк обменивается с антифрод-центром Национального Банка данными о транзакциях, имеющих признаки мошенничества.", "covered"),
    ("TX-AFC-01", "Каждый перевод проверяется внутренней антифрод-системой банка.", "missing"),
    ("LEND-COOL-01", "Сумма беззалогового кредита перечисляется не раньше чем через восемь часов после подписания; "
                     "до перечисления заёмщик повторно подтверждает своё согласие на получение кредита.", "covered"),
    ("LEND-COOL-01", "Действует период охлаждения: если сумма превышает 255 МРП, деньги поступают на карту на следующий день.", "covered"),
    ("LEND-COOL-01", "Займ зачисляется на карту мгновенно после подписания договора.", "missing"),
    ("LEND-BAN-01", "До одобрения заявки проверяется, не установил ли клиент самозапрет на получение займов "
                    "и нет ли у него просрочки свыше 90 календарных дней.", "covered"),
    ("LEND-BAN-01", "Заявка одобряется по результатам скоринга.", "missing"),
    ("LEND-AUTO-01", "Если в выдаче отказано автоматически, клиенту сообщаются основные причины отказа,\n"
                     "и он может запросить пересмотр заявки сотрудником банка.", "covered"),
    ("LEND-AUTO-01", "Решение по заявке принимает скоринговая модель.", "partial"),
    ("PRIV-04", "Данные о счетах и остатках клиентов относятся к банковской тайне и не передаются в нейросетевые сервисы;\n"
                "в тестовых контурах применяются обезличенные копии.", "covered"),
    ("PRIV-04", "Выписки клиентов выгружаются в BI-систему подрядчика.", "missing"),
    ("REG-ESIGN-01", "Заявление подписывается ЭЦП через NCALayer; такая подпись равнозначна собственноручной.", "covered"),
    ("REG-ESIGN-01", "Договор подписывается OTP-кодом.", "missing"),
    ("APPSEC-DBO-01", "Установка приложения возможна только из Google Play и App Store;\n"
                      "при входе с нового телефона клиенту приходит SMS-оповещение.", "covered"),
    ("APPSEC-DBO-01", "Приложение поддерживает тёмную тему.", "missing"),
    ("SEC-3RD-01", "Сотрудники аутсорсинговой компании получают доступ к серверам только на время выполнения работ,\n"
                   "подписав обязательство о неразглашении.", "covered"),
    ("BIO-01", "Лицо клиента сверяется с фотографией из госбазы через ЦОИД с проверкой живости.", "covered"),
    ("REG-SEC-01", "Об инцидентах ИБ подразделение безопасности сообщает в Агентство по регулированию финансового рынка.", "covered"),
])
def test_new_control_patterns_on_fresh_phrasings(cid, text, expected):
    assert _status(cid, text) == expected


def test_good_sample_covers_all_applicable_controls():
    from agents.orchestrator import analyze
    from config import LLMConfig
    from parsing import load_path

    r = analyze(load_path(ROOT / "samples" / "good_tz.md"), LLMConfig(), use_llm=False)
    assert r.overall_score >= 90
    assert r.verdict == "Готово к разработке"
    missing = [c.control_id for c in r.controls if c.status == "missing"]
    assert not missing, missing
