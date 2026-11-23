"""Тесты конвейера: офлайн-режим и LLM-режим на фейковой модели (без сети)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import base  # noqa: E402
from agents.orchestrator import analyze  # noqa: E402
from config import LLMConfig  # noqa: E402
from parsing import load_path, load_text  # noqa: E402

SAMPLE = ROOT / "samples" / "sample_tz.md"


def test_offline_pipeline_finds_key_issues():
    r = analyze(load_path(SAMPLE), LLMConfig(), use_llm=False)
    titles = " | ".join(f.title for f in r.findings)
    assert "CVV" in titles                                   # опасное требование
    assert "номер карты" in titles.lower() or "логах" in titles   # PAN в логах
    assert "Лимит" in titles and "Доступность" in titles     # численные противоречия
    assert r.verdict_level == "not_ready"
    assert all(f.id.startswith("F-") for f in r.findings)
    crit = {f.id for f in r.findings if f.severity.value == "critical"}
    covered = {fid for q in r.questions for fid in q.related_findings}
    assert crit <= covered, "каждая критичная находка должна быть покрыта вопросом"


def test_good_spec_scores_higher():
    bad = analyze(load_path(SAMPLE), LLMConfig(), use_llm=False)
    good = analyze(load_path(ROOT / "samples" / "good_tz.md"), LLMConfig(), use_llm=False)
    assert good.overall_score > bad.overall_score + 15


def test_quote_location_is_fuzzy():
    doc = load_text("L1\nFR-12. Система сохраняет номер карты, срок действия и CVV привязанной карты.\n")
    ln, score = doc.locate_quote("сохраняет номер карты, срок действия и CVV")
    assert ln == 2 and score == 1.0
    ln, score = doc.locate_quote("этого текста в документе нет совсем")
    assert score < 0.75


def test_docx_and_revised_tz_roundtrip(tmp_path):
    import docx

    from improve import build_revised_tz
    from parsing import load_bytes

    d = docx.Document()
    d.add_heading("ТЗ: платёжный сервис", level=1)
    d.add_paragraph("FR-01. Система сохраняет CVV карты для повторных платежей.")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Лимит", "суточный лимит 100 000 тенге"
    path = tmp_path / "tz.docx"
    d.save(path)
    doc = load_bytes(path.read_bytes(), "tz.docx")
    assert doc.lines[0].startswith("# ")
    assert any("| Лимит |" in l for l in doc.lines)

    r = analyze(doc, LLMConfig(), use_llm=False)
    assert any("CVV" in f.title for f in r.findings)
    revised, stats = build_revised_tz(doc, r, {})
    assert stats["replaced"] >= 1
    r2 = analyze(load_text(revised, "tz_v2.md"), LLMConfig(), use_llm=False)
    assert not any("CVV" in f.title for f in r2.findings)
    assert r2.overall_score > r.overall_score


class FakeLLM:
    """Имитирует ответы LLM, включая одну галлюцинацию, которую должен отсечь Verifier."""

    active_model = "fake-model"
    cache_hits = 0
    transient_errors = 0

    def __init__(self, cfg):
        self.cfg = cfg
        self.switches: list[str] = []

    async def chat_json(self, system, user, schema):
        if "агент **Reader**" in system:
            data = {"title": "DemoBank Mobile", "summary": "Мобильный банк", "product_type": "мобильный банк",
                    "domains": ["transfers", "cards", "kyc", "mobile"], "requirements": [{"id": "FR-01", "text": "Регистрация", "line": 17}]}
        elif "агент **Completeness**" in system:
            data = {"findings": [{"category": "completeness", "severity": "major",
                                  "title": "Не описан перевод незарегистрированному получателю",
                                  "description": "…", "evidence": [], "confidence": 0.8}]}
        elif "агент **Consistency**" in system:
            data = {"findings": [
                {"category": "contradiction", "severity": "critical", "title": "Переводы без подтверждения vs подтверждение каждой операции",
                 "evidence": ["Переводы до 50 000 тенге выполняются без подтверждения", "Каждая операция подтверждается SMS-кодом"],
                 "lines": [43, 44], "confidence": 0.9},
                {"category": "ambiguity", "severity": "major", "title": "Выдуманная проблема",
                 "evidence": ["система должна летать на Марс в режиме турбо"], "lines": [999], "confidence": 0.9},
            ]}
        elif "агент **FinTech/Security**" in system:
            data = {"controls": [{"control_id": "SEC-AUTH-01", "status": "covered", "evidence": ["вход по отпечатку сетчатки глаза"]},
                                 {"control_id": "PRIV-02", "status": "missing"}],
                    "findings": [{"category": "compliance", "severity": "critical", "title": "Хранение CVV",
                                  "evidence": ["сохраняет номер карты, срок действия и CVV"], "lines": [35],
                                  "regulation": "PCI DSS v4.0", "control_id": "AP-CVV-STORE", "confidence": 0.95}]}
        elif "агент **Question Generator**" in system:
            data = {"questions": [{"question": "Можно ли заменить хранение CVV токенизацией?", "priority": "P0",
                                   "related_findings": ["F-01", "F-99"], "answer_options": ["Да", "Нет"], "blocking": True}]}
        else:
            data = {"summary": "Резюме от LLM.", "strengths": ["Есть интеграции"]}
        return schema.model_validate(data), 100


def test_llm_pipeline_with_fake_model(monkeypatch):
    monkeypatch.setattr(base, "LLMClient", FakeLLM)
    from agents import orchestrator
    monkeypatch.setattr(orchestrator, "LLMClient", FakeLLM)
    r = analyze(load_path(SAMPLE), LLMConfig(api_key="fake"), use_llm=True)

    assert r.mode == "llm"
    assert r.verifier.rejected_hallucinations >= 1                     # «Марс» отсечён
    assert not any("Марс" in f.title or "Выдуманная" in f.title for f in r.findings)
    ctrl = {c.control_id: c for c in r.controls}
    assert ctrl["SEC-AUTH-01"].status != "covered"                     # цитата LLM не подтверждена
    cvv = [f for f in r.findings if "CVV" in f.title]
    assert len(cvv) == 1 and len(cvv[0].sources) >= 1                 # дубль LLM+правил склеен
    assert all("F-99" not in q.related_findings for q in r.questions)   # битая ссылка удалена
    crit = {f.id for f in r.findings if f.severity.value == "critical"}
    assert crit <= {fid for q in r.questions for fid in q.related_findings}
    assert r.summary == "Резюме от LLM."
