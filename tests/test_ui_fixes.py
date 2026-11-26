"""Регрессионные тесты исправлений UI (app.py), ТЗ v2 (improve.py) и документации.

Все тесты офлайн: LLM не вызывается (провайдер «Офлайн-правила» или анализ не запускается)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA = "Локальная модель (Ollama, контур банка)"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_FALLBACK_MODELS", "LLM_TIMEOUT", "LLM_CALL_BUDGET", "LLM_PRIVACY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.chdir(ROOT)


def _app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app_fintech.py"), default_timeout=90)
    at.session_state["app_mode"] = "Финтех-аудит ТЗ (повестка, ТЗ v2)"
    return at


def _button(at, text):
    return next(b for b in at.button if text in b.label)


def _select_preset(at, text):
    sb = at.sidebar.selectbox[0]
    return sb.set_value(next(o for o in sb.options if text in o)).run()


def _offline_app(source: str | None = None, text: str | None = None):
    at = _app().run()
    assert not at.exception
    _select_preset(at, "Офлайн")
    if source:
        src = next(r for r in at.radio if source in r.options)
        src.set_value(source).run()
    if text is not None:
        at.text_area[0].set_value(text).run()
    assert not at.exception, at.exception
    return at


def _analyze(at):
    _button(at, "Анализировать").click().run()
    assert not at.exception, at.exception
    return at


def _metric(at, label):
    return next(m for m in at.metric if label in m.label).value


def _warnings(at) -> str:
    return " ".join(str(w.value) for w in at.warning)


def _gemini_env(monkeypatch, key="secret-gemini"):
    monkeypatch.setenv("LLM_BASE_URL", GEMINI_URL)
    monkeypatch.setenv("LLM_API_KEY", key)
    monkeypatch.setenv("LLM_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "gemini-flash-lite-latest,gemini-3.1-flash-lite")
    monkeypatch.setenv("LLM_TIMEOUT", "120")


# ── #2 / #11: пресеты провайдеров ─────────────────────────────────────────────

def test_ollama_preset_keeps_timeouts_and_does_not_get_cloud_key(monkeypatch):
    _gemini_env(monkeypatch)
    at = _app().run()
    _select_preset(at, "Ollama")
    assert not at.exception, at.exception
    cfg = at.session_state["llm_cfg"]
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.timeout == 600.0 and cfg.call_budget_s == 900.0
    assert cfg.fallback_models == ()            # не наследует Gemini-модели из .env
    assert cfg.api_key == "ollama"              # ключ пресета, а не облачный ключ из .env
    assert cfg.max_concurrency == 1


def test_openai_preset_never_reuses_env_key_of_other_provider(monkeypatch):
    _gemini_env(monkeypatch)
    at = _app().run()
    _select_preset(at, "OpenAI")
    cfg = at.session_state["llm_cfg"]
    assert cfg.base_url == "https://api.openai.com/v1"
    assert cfg.api_key is None, "ключ Gemini из .env не должен уходить в OpenAI"
    assert cfg.fallback_models == ()
    key_input = next(t for t in at.sidebar.text_input if t.label == "API key")
    assert "вставьте ключ" in key_input.proto.placeholder
    # без ключа LLM-режим недоступен — анализ пойдёт на правилах, тумблера LLM нет
    assert not [t for t in at.sidebar.toggle if "LLM" in t.label]


def test_typed_key_is_stripped(monkeypatch):
    _gemini_env(monkeypatch)
    at = _app().run()
    _select_preset(at, "OpenAI")
    key_input = next(t for t in at.sidebar.text_input if t.label == "API key")
    key_input.set_value("  sk-typed \n").run()
    assert at.session_state["llm_cfg"].api_key == "sk-typed"


def test_env_preset_uses_from_env_and_strips_key(monkeypatch):
    _gemini_env(monkeypatch, key="  secret-gemini  ")
    at = _app().run()
    _select_preset(at, "Из .env")
    cfg = at.session_state["llm_cfg"]
    assert cfg.api_key == "secret-gemini"
    assert cfg.base_url == GEMINI_URL and cfg.timeout == 120.0
    assert cfg.fallback_models == ("gemini-flash-lite-latest", "gemini-3.1-flash-lite")
    # тот же эндпоинт в пресете Gemini — ключ из .env допустим
    _select_preset(at, "Gemini")
    from config import PROVIDER_PRESETS

    cfg = at.session_state["llm_cfg"]
    assert cfg.api_key == "secret-gemini"
    assert cfg.fallback_models == tuple(PROVIDER_PRESETS["Google Gemini"]["fallback_models"])


def test_preset_without_endpoint_blocks_analysis(monkeypatch):
    _gemini_env(monkeypatch)
    at = _app().run()
    _select_preset(at, "shai.pro")
    assert not at.exception, at.exception
    assert "Base URL" in _warnings(at)
    assert _button(at, "Анализировать").disabled
    assert at.session_state["llm_cfg"].api_key is None


# ── #4: повреждённый шаблон ───────────────────────────────────────────────────

def _choose_template(at, option):
    r = next(r for r in at.radio if option in r.options)
    return r.set_value(option).run()


def test_broken_template_does_not_crash(monkeypatch):
    import parsing

    real = parsing.load_path

    def broken(path, *a, **kw):
        if Path(path).name == "template_example.md":
            raise ValueError("файл повреждён")
        return real(path, *a, **kw)

    monkeypatch.setattr(parsing, "load_path", broken)
    at = _offline_app()
    _choose_template(at, "Пример шаблона банка")
    assert not at.exception, at.exception
    assert any("Не удалось прочитать шаблон" in str(e.value) for e in at.error)
    assert not _button(at, "Анализировать").disabled


def test_template_without_sections_warns(monkeypatch):
    import knowledge.template as tpl

    monkeypatch.setattr(tpl, "sections_from_template", lambda doc: [])
    at = _offline_app()
    _choose_template(at, "Пример шаблона банка")
    assert not at.exception, at.exception
    assert "не найдено ни одного раздела" in _warnings(at)


# ── #16, #27, #14, #28: повестка, счётчики, доп. вопросы, экспорт ─────────────

def test_agenda_counters_extra_questions_and_memoized_exports(monkeypatch):
    import report_docx

    calls = {"n": 0}
    real = report_docx.to_docx

    def counting(r, *a, **kw):
        calls["n"] += 1
        return real(r, *a, **kw)

    monkeypatch.setattr(report_docx, "to_docx", counting)
    at = _analyze(_offline_app())
    assert calls["n"] == 1
    report = at.session_state["report"]
    n_main, n_extra = len(report.questions), len(report.extra_questions)
    assert n_extra > 0

    # #27: качество вопросов — «прошли чеклист N/total», а не голое число
    caps = " ".join(str(c.value) for c in at.caption)
    assert re.search(rf"\d+/{n_main}\b", caps.split("Качество вопросов", 1)[1])

    # #16: счётчик «Отвечено» обновляется в тот же клик
    _button(at, "Принять рекомендуемые").click().run()
    assert not at.exception, at.exception
    accepted = sum(1 for q in report.questions if q.answer_options and q.priority.value in ("P0", "P1"))
    assert _metric(at, "Отвечено") == f"{accepted}/{n_main}"
    text = " ".join(str(m.value) for m in at.markdown)
    assert f"внесено ответов — {accepted}" in text

    # #14: дополнительные вопросы — отдельный сворачиваемый блок с теми же виджетами
    exp = next(e for e in at.expander if "Дополнительные вопросы" in e.label)
    assert f"({n_extra})" in exp.label
    q_extra = next(q for q in report.extra_questions if q.answer_options)
    radio = next(r for r in at.radio if r.key and r.key.endswith(f"_{q_extra.id}"))
    radio.set_value(radio.options[1]).run()
    assert not at.exception, at.exception
    assert _metric(at, "Отвечено") == f"{accepted}/{n_main} +1 доп."
    text = " ".join(str(m.value) for m in at.markdown)
    assert f"внесено ответов — {accepted + 1}" in text
    assert q_extra.id in at.session_state["answers"]

    # повестка (.md) включает дополнительные вопросы
    agenda = at.session_state["exports"]["agenda"]
    assert "Дополнительные вопросы" in agenda and f"**{q_extra.id} [" in agenda

    # #28: Word-отчёт не пересобирается на каждый клик по ответам
    assert calls["n"] == 1
    _button(at, "Сформировать ТЗ v2").click().run()
    assert not at.exception, at.exception
    assert calls["n"] == 2   # новый анализ → новый экспорт


# ── #31: перепроверка ТЗ v2 с текущим выбором шаблона ─────────────────────────

def test_v2_recheck_uses_current_template_selection():
    at = _offline_app()
    _choose_template(at, "Пример шаблона банка")
    _analyze(at)
    assert "Соответствие шаблону ТЗ" in " ".join(str(m.value) for m in at.markdown)
    _choose_template(at, "Без шаблона")
    caps = " ".join(str(c.value) for c in at.caption)
    assert "текущим выбором шаблона: без шаблона" in caps
    _button(at, "Сформировать ТЗ v2").click().run()
    assert not at.exception, at.exception
    assert "Соответствие шаблону ТЗ" not in " ".join(str(m.value) for m in at.markdown)
    assert at.session_state["report"].template_check in (None, [])


# ── #17 / #18 / #19: проверки входа и баннеры ─────────────────────────────────

def test_too_short_document_blocks_analysis():
    at = _offline_app("Вставить текст", "# ТЗ\nПереводы между счетами.\nЛимит 100 000 тенге.")
    assert "недостаточно для оценки" in _warnings(at)
    assert _button(at, "Анализировать").disabled


ENGLISH_TZ = """# Technical specification: mobile wallet
## 1. Purpose
The wallet lets customers send P2P transfers by phone number.
## 2. Requirements
- The system stores the card number and CVV for repeat payments.
- Transfers up to 50 000 KZT are executed without confirmation.
- The app must work fast.
"""

KAZAKH_TZ = """# Техникалық тапсырма: мобильді әмиян
## 1. Мақсаты
Әмиян клиенттерге телефон нөмірі бойынша аударым жасауға мүмкіндік береді.
## 2. Талаптар
- Жүйе қайталама төлемдер үшін карта нөмірі мен CVV сақтайды.
- 50 000 теңгеге дейінгі аударымдар растаусыз орындалады.
- Қосымша жылдам жұмыс істеуі керек.
"""


@pytest.mark.parametrize("text,lang", [(ENGLISH_TZ, "английском"), (KAZAKH_TZ, "казахском")])
def test_non_russian_tz_warns_to_use_llm(text, lang):
    at = _offline_app("Вставить текст", text)
    w = _warnings(at)
    assert lang in w and "LLM" in w
    assert not _button(at, "Анализировать").disabled   # предупреждение, а не блокировка


def test_russian_sample_has_no_language_or_fintech_warning():
    at = _analyze(_offline_app())
    w = _warnings(at)
    assert "языке" not in w and "не похоже на финтех" not in w


def test_non_fintech_tz_gets_banner():
    library = ("# ТЗ: сайт школьной библиотеки\n## 1. Назначение\nСайт показывает каталог книг школьной библиотеки.\n"
               "## 2. Требования\n- Ученик ищет книгу по названию.\n- Библиотекарь добавляет книги через админку.\n"
               "- Сайт должен быстро работать.\n")
    at = _analyze(_offline_app("Вставить текст", library))
    assert "не похоже на финтех" in _warnings(at)


def test_language_profile_of_samples_is_russian():
    """Ни один русскоязычный образец не должен получать предупреждение о языке."""
    import ast

    src = (ROOT / "app_fintech.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_letters_profile")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "app_fintech.py", "exec"), ns)
    for p in sorted((ROOT / "samples").glob("*_tz.md")):
        latin, kaz = ns["_letters_profile"](p.read_text(encoding="utf-8"))
        assert latin <= 0.5 and kaz <= 0.02, p.name
    latin, kaz = ns["_letters_profile"](ENGLISH_TZ)
    assert latin > 0.5
    latin, kaz = ns["_letters_profile"](KAZAKH_TZ)
    assert kaz > 0.02


# ── #25 / #14: improve.py — приложение с ответами ─────────────────────────────

def _offline_report(path: Path):
    from agents.orchestrator import analyze
    from config import LLMConfig
    from parsing import load_path

    doc = load_path(path)
    return doc, analyze(doc, LLMConfig(), use_llm=False)


def _recommended(q) -> str:
    ro = getattr(q, "recommended_option", None)
    return q.answer_options[ro if isinstance(ro, int) and 0 <= ro < len(q.answer_options) else 0]


def test_answers_appendix_does_not_reintroduce_dangerous_quotes():
    from agents.orchestrator import analyze
    from config import LLMConfig
    from improve import build_revised_tz
    from models import Severity
    from parsing import load_text

    doc, r1 = _offline_report(ROOT / "samples" / "sample_tz.md")
    answers = {q.id: _recommended(q) for q in r1.questions if q.answer_options and q.priority.value in ("P0", "P1")}
    text, stats = build_revised_tz(doc, r1, answers)
    head, appendix = text.split("## Уточнения автора ТЗ (ответы на вопросы)", 1)
    assert stats["answers"] == len(answers)
    from improve import _norm

    doc_norm = _norm(doc.text)
    for quoted in re.findall(r"«([^»]*)»", appendix):   # цитаты опасных строк ТЗ не переносятся
        assert _norm(quoted.rstrip("…. ")) not in doc_norm, quoted
    for q in r1.questions:
        if q.id in answers:
            assert q.question not in appendix
            assert f"**{q.id}**" in appendix
    first_appendix_line = head.count("\n") + 1
    r2 = analyze(load_text(text, "sample_tz_v2.md"), LLMConfig(), use_llm=False)
    crit_in_appendix = [f for f in r2.findings if f.severity == Severity.critical
                        and any(n >= first_appendix_line for n in f.lines)]
    assert not crit_in_appendix, [(f.title, f.lines) for f in crit_in_appendix]
    assert r2.overall_score > r1.overall_score


def test_strip_doc_quotes_keeps_authors_own_quotes():
    from improve import _norm, _strip_doc_quotes

    doc_norm = _norm("- FR-17. Переводы до 50 000 тенге выполняются без подтверждения для удобства клиента.")
    ans = "Действует (L43) «FR-17. Переводы до 50 000 тенге выполняются без подтверждения для удо…» — второе исправить"
    assert _strip_doc_quotes(ans, doc_norm) == "Действует (L43) — второе исправить"
    own = "Интеграция с «Kaspi Pay» и «Halyk ePay» обязательна"
    assert _strip_doc_quotes(own, doc_norm) == own


def test_answers_to_extra_questions_reach_tz_v2():
    from improve import build_revised_tz

    doc, r = _offline_report(ROOT / "samples" / "sample_tz.md")
    assert r.extra_questions
    q = r.extra_questions[0]
    text, stats = build_revised_tz(doc, r, {q.id: "Ответ на дополнительный вопрос"})
    assert stats["answers"] == 1
    assert f"**{q.id}**" in text and "Ответ на дополнительный вопрос" in text
    # неизвестный ID вопроса не попадает в ТЗ и не считается внесённым
    text2, stats2 = build_revised_tz(doc, r, {"Q-999": "x"})
    assert stats2["answers"] == 0 and "Уточнения автора" not in text2


# ── документация: цифры совпадают с базой знаний и кодом ──────────────────────

def test_docs_numbers_match_knowledge_base():
    from knowledge.antipatterns import ANTIPATTERNS
    from knowledge.fintech_controls import CONTROLS

    n_ctrl, n_ap = len(CONTROLS), len(ANTIPATTERNS)
    for name in ("docs/FINTECH_MODE.md", "docs/PITCH_FINTECH.md"):
        t = (ROOT / name).read_text(encoding="utf-8")
        ctrl = {int(x) for x in re.findall(r"(\d+)\s+(?:\*\*)?(?:финтех/ИБ-)?контрол", t)}
        aps = {int(x) for x in re.findall(r"(\d+)\s+(?:\*\*)?тип\w* опасных требований", t)}
        assert ctrl == {n_ctrl}, (name, ctrl, n_ctrl)
        assert aps == {n_ap}, (name, aps, n_ap)


def test_pitch_states_measured_v2_numbers_and_names_holdout_v2():
    t = (ROOT / "docs" / "PITCH_FINTECH.md").read_text(encoding="utf-8")
    readme = (ROOT / "docs" / "FINTECH_MODE.md").read_text(encoding="utf-8")
    assert "балл растёт, критичные → 0" not in t        # вместо обещания — измеренные числа
    m = re.search(r"балл (\d+) → (\d+),\s*критичных (\d+) → (\d+)", t)
    assert m, "PITCH должен называть измеренные числа ТЗ v1 → v2"
    v1, v2, c1, c2 = m.groups()
    assert f"**{v1}/100" in readme and f"**{v2}/100" in readme and f"критичных {c1} → {c2}" in readme
    assert "кредитные карты" in t and "Open Banking" in t and "зарплатный проект" in t
    assert "116/116" in t and "24/48" in t
    assert "вкладке «Шаблон ТЗ»" not in t


def test_readme_formula_and_env_example():
    readme = (ROOT / "docs" / "FINTECH_MODE.md").read_text(encoding="utf-8")
    assert "max(0.5, уверенность)" in readme
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "# LLM_TIMEOUT=600" in env and "# LLM_CALL_BUDGET=900" in env
    assert "LLM_MAX_DOC_CHARS" in env and "организаций" in env


def test_scorer_confidence_factor_matches_readme():
    src = (ROOT / "agents" / "scorer.py").read_text(encoding="utf-8")
    assert "max(0.5" in src
