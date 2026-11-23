"""Smoke-тест Streamlit UI без браузера: анализ, повестка, ТЗ v2 → экран «до/после»."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.chdir(ROOT)


def _app():
    from streamlit.testing.v1 import AppTest

    os.environ["LLM_API_KEY"] = ""
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)


def _button(at, text):
    return next(b for b in at.button if text in b.label)


def test_app_full_flow():
    at = _app().run()
    assert not at.exception
    # провайдер «Офлайн-правила», чтобы тест не зависел от .env
    sb = at.sidebar.selectbox[0]
    sb.set_value(next(o for o in sb.options if "Офлайн" in o)).run()
    _button(at, "Анализировать").click().run()
    assert not at.exception, at.exception
    assert any("Повестка встречи" in str(t.label) for t in at.tabs)
    _button(at, "Принять рекомендуемые").click().run()
    assert not at.exception, at.exception
    _button(at, "Сформировать ТЗ v2").click().run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "До / После" in text
