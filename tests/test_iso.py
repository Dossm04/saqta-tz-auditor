"""Режим кейса хакатона: проверка требований по ISO/IEC/IEEE 29148 и IEEE 830."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import iso_review  # noqa: E402
from agents.iso_review import review_document  # noqa: E402
from agents.segmenter import segment  # noqa: E402
from config import LLMConfig  # noqa: E402
from parsing import load_path, load_text  # noqa: E402
from report_iso import to_markdown  # noqa: E402

MD = ROOT / "examples" / "case_requirements.md"
PDF = ROOT / "examples" / "case_requirements.pdf"


def test_segmentation_keeps_document_ids_and_skips_service_sections():
    seg = segment(load_path(MD))
    refs = [u.doc_ref for u in seg.units]
    assert {"Ф-1", "Ф-2", "Ф-3", "Ф-4"} <= set(refs)
    assert any(r.startswith("Критерии приемки, строка") for r in refs)
    assert any("параметр «amount»" in r for r in refs)
    assert any(r.startswith("Логика работы метода, п.") for r in refs)
    assert any("код 401" in r for r in refs)
    # служебные разделы не являются требованиями
    assert not any(u.section.lower().startswith("часть a. новый функционал › общие сведения") for u in seg.units)
    assert not any("Подразделение" in u.text for u in seg.units)
    assert len(seg.checklist) >= 10            # чек-лист ревью сохранён отдельно
    assert all(u.rid == f"R-{k:02d}" for k, u in enumerate(seg.units, start=1))


@pytest.mark.skipif(not PDF.exists(), reason="нет PDF-примера")
def test_pdf_and_markdown_segment_the_same_way():
    a = [u.doc_ref for u in segment(load_path(MD)).units]
    b = [u.doc_ref for u in segment(load_path(PDF)).units]
    common = set(a) & set(b)
    assert len(common) >= 0.95 * len(a)
    doc = load_path(PDF)
    assert getattr(doc, "line_pages", None) and len(doc.line_pages) == len(doc.lines)
    assert any(l.startswith("| Расходы 48 500 ₸") for l in doc.lines)     # таблица восстановлена построчно


def test_offline_report_has_all_required_sections_and_references():
    r = review_document(load_path(MD), LLMConfig(), use_llm=False)
    md = to_markdown(r)
    for head in ("## 1. Паспорт прогона", "## 2. Как агент нарезал требования", "## 3. Проверка каждого требования",
                 "## 4. Проверка всего набора требований", "## 5. Сводка", "## 6. Итог"):
        assert head in md
    for field in ("Проверяемый документ", "Дата и время прогона", "Модель", "Агент"):
        assert field in md
    assert r.verdict_level == "not_ready"
    # каждое нарушение привязано к месту: строки или связанные требования
    for u in r.units:
        for i in u.issues:
            assert i.lines or i.related, (u.rid, i.problem)
    titles = " ".join(i.problem for i in r.set_review.issues)
    assert "GET" in titles and "201" in titles and "404" in titles


def test_no_false_contradiction_between_rows_of_one_acceptance_table():
    r = review_document(load_path(MD), LLMConfig(), use_llm=False)
    for i in r.set_review.issues:
        if i.characteristic == "set_consistent":
            assert not ("48 500" in " ".join(i.evidence) and "19 900" in " ".join(i.evidence))


def test_rules_flag_vague_tbd_and_unfilled_fields_on_fresh_text():
    text = ("# ТЗ\n## Функциональные требования\n* **Ф-1.** Система должна быстро обрабатывать заявки и т.д.\n"
            "## Нефункциональные требования\nСервис работает надёжно.\n## Тестирование\nбудет заполнено позже\n")
    r = review_document(load_text(text), LLMConfig(), use_llm=False)
    chars = {i.characteristic for u in r.units for i in u.issues}
    assert {"unambiguous", "verifiable", "complete"} <= chars


class _FakeLLM:
    active_model = "fake"
    cache_hits = 0
    transient_errors = 0
    switches: list = []

    def __init__(self, cfg):
        pass

    async def chat_json(self, system, user, schema):
        if "Contradiction Hunter" in system:
            return schema.model_validate({"contradictions": [
                {"requirements": ["R-07", "R-08"], "evidence": ["максимум один такой push за календарный месяц",
                                                                "независимо от истории предыдущей настройки"],
                 "problem": "Лимит одного push в месяц противоречит push «независимо от истории».", "fix": "Уточнить.",
                 "severity": "critical"},
                {"requirements": ["R-01", "R-02"], "evidence": ["выдуманная цитата, которой нет"], "problem": "x", "severity": "major"}]}), 10
        if "набор требований целиком" in system:
            return schema.model_validate({"statuses": {"set_consistent": "issue"}, "issues": [], "missing": [],
                                          "checklist": [], "questions": [], "summary": "Резюме.", "top_fixes": []}), 10
        rids = [l.split()[1] for l in user.splitlines() if l.startswith("### R-")]
        return schema.model_validate({"reviews": [
            {"rid": rid, "statuses": {"unambiguous": "issue"},
             "issues": [{"characteristic": "unambiguous", "severity": "major", "problem": "Фантом", "evidence": "текста нет в документе"}]}
            for rid in rids]}), 10


def test_llm_path_grounds_claims_and_adds_semantic_contradictions(monkeypatch):
    monkeypatch.setattr(iso_review, "LLMClient", _FakeLLM)
    r = review_document(load_path(MD), LLMConfig(api_key="x", privacy="off", cache=False), use_llm=True)
    assert r.stats["rejected"] >= len(r.units)          # фантомные цитаты отброшены
    assert not any(i.problem == "Фантом" for u in r.units for i in u.issues)
    # статус «нарушено» без конкретики понижен до замечания
    assert all(u.statuses.get("unambiguous") != "issue" or any(i.characteristic == "unambiguous" for i in u.issues) for u in r.units)
    contra = [i for i in r.set_review.issues if i.characteristic == "set_consistent"]
    assert any(set(i.related) == {"R-07", "R-08"} for i in contra)
    assert len(r.set_review.checklist) >= 10            # чек-лист дополнен до полного
    assert len(r.set_review.questions) >= 6


def test_cli_one_command(tmp_path):
    out = tmp_path / "report.md"
    p = subprocess.run([sys.executable, str(ROOT / "validate.py"), str(MD), str(out), "--offline"],
                       capture_output=True, text=True, encoding="utf-8", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert p.returncode == 0, p.stderr
    assert out.exists() and "## 6. Итог" in out.read_text(encoding="utf-8")
