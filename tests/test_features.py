"""Тесты функций вокруг конвейера: шаблон ТЗ, Word-отчёт, оценка времени, eval-набор."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.orchestrator import analyze  # noqa: E402
from config import LLMConfig  # noqa: E402
from knowledge.template import match_template, sections_from_template  # noqa: E402
from parsing import load_path, load_text  # noqa: E402

TEMPLATE = ROOT / "samples" / "template_example.md"


def test_template_sections_and_matching():
    tpl = load_path(TEMPLATE)
    secs = sections_from_template(tpl)
    titles = [s.title for s in secs]
    assert "Функциональные требования" in titles and "Критерии приёмки" in titles
    assert not any("Шаблон технического задания" in t for t in titles)   # название документа — не раздел
    doc = load_text("# ТЗ\n## 5. Функциональные требования\nFR-01\n## Критерии приемки\nтест\n")
    m = {x.section.title: x.present for x in match_template(doc, secs)}
    assert m["Функциональные требования"] and m["Критерии приёмки"]
    assert not m["Интеграции"]


def test_template_findings_in_report_and_placeholders():
    doc = load_text("# ТЗ\n## Функциональные требования\nFR-01. Перевод по номеру <указать лимит>.\n")
    r = analyze(doc, LLMConfig(), use_llm=False, template=load_path(TEMPLATE))
    assert r.template_check and any(not t["present"] for t in r.template_check)
    titles = [f.title for f in r.findings]
    assert any(t.startswith("Нет раздела из шаблона") for t in titles)
    assert any(t == "Незаполненное поле шаблона" for t in titles)


def test_docx_export_and_time_estimate():
    import docx

    from report_docx import to_docx

    r = analyze(load_path(ROOT / "samples" / "sample_tz.md"), LLMConfig(), use_llm=False)
    assert r.manual_review_hours >= 1.0
    d = docx.Document(io.BytesIO(to_docx(r)))
    text = "\n".join(p.text for p in d.paragraphs)
    assert "Уточняющие вопросы" in text and "CVV" in text
    assert len(d.tables) >= 3


def test_eval_labels_are_valid_and_point_to_real_lines():
    for f in (ROOT / "eval").glob("*.expected.json"):
        exp = json.loads(f.read_text(encoding="utf-8"))
        stem = f.name.removesuffix(".expected.json")
        sample = exp.get("sample") or next(
            (f"samples/{c}" for c in (f"{stem}.md", f"{stem}_tz.md") if (ROOT / "samples" / c).exists()), None)
        assert sample, f"нет ТЗ для {f.name}"
        n_lines = len((ROOT / sample).read_text(encoding="utf-8").splitlines())
        assert exp["planted"], f.name
        for p in exp["planted"]:
            assert p["kind"]
            for ln in p.get("lines", []):
                assert 1 <= ln <= n_lines, f"{f.name}: {p['id']} L{ln} вне документа"
