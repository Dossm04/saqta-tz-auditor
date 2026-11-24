"""Регрессии устойчивости LLM-слоя: быстрый отказ при недоступном провайдере, гонка удаления модели (404),
разбор JSON с хвостом, скаляры вместо списков, обязательные ключи схем, critic loop без «отмывания»
галлюцинаций, применимость контролей, окна ТЗ при ревизии, отмена фрагментов и учёт токенов при ошибке."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import base, orchestrator  # noqa: E402
from agents.analyzers import (CompletenessAgent, ConsistencyAgent, FinTechSecurityAgent, RawFindings,  # noqa: E402
                              RawFintech, RawRevision, ask_doc, parse_findings, revise_findings,
                              revision_context)
from agents.base import Agent, Context, LLMClient, LLMError, extract_json, is_fatal_connection_error  # noqa: E402
from agents.orchestrator import analyze  # noqa: E402
from config import LLMConfig  # noqa: E402
from models import ControlCheck, Finding, Question  # noqa: E402
from parsing import load_text  # noqa: E402

REQ = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions")


def _resp(content: str, tokens: int = 7):
    return SimpleNamespace(usage=SimpleNamespace(total_tokens=tokens),
                           choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _client(create, **cfg) -> LLMClient:
    cfg.setdefault("cache", False)
    c = LLMClient(LLMConfig(api_key="k", base_url="http://127.0.0.1:9/v1", **cfg))
    c.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return c


def _conn_error(cause: BaseException) -> openai.APIConnectionError:
    err = openai.APIConnectionError(request=REQ)
    err.__cause__ = cause
    return err


# ── #33 extract_json ──────────────────────────────────────────────────────────

def test_extract_json_ignores_prose_with_braces_after_json():
    assert extract_json('{"findings": []}\nПримечание: см. {раздел 3} и [L12].') == {"findings": []}
    assert extract_json('Вот ответ: {"a": {"b": [1, 2]}} — готово }') == {"a": {"b": [1, 2]}}
    assert extract_json('```json\n{"x": 1}\n```') == {"x": 1}
    # объект предпочтительнее массива, даже если массив встретился раньше
    assert extract_json('Строки [3, 4], ответ: {"findings": [{"title": "t"}]} и ещё }') == \
        {"findings": [{"title": "t"}]}
    assert extract_json('только массив [1, 2] и текст ]') == [1, 2]
    with pytest.raises(ValueError):
        extract_json("никакого JSON {нет} здесь")


# ── #3 быстрый отказ при недоступном провайдере ───────────────────────────────

@pytest.mark.parametrize("cause", [httpx.ConnectError("[Errno 111] Connection refused"),
                                   httpx.LocalProtocolError("Illegal header value b'Bearer key '"),
                                   httpx.ConnectError("[Errno 11001] getaddrinfo failed")])
def test_unreachable_provider_fails_fast(monkeypatch, cause):
    monkeypatch.setattr(LLMClient, "CONNECT_RETRY_PAUSE_S", 0.0)
    calls = []

    async def create(**kw):
        calls.append(kw["model"])
        raise _conn_error(cause)

    c = _client(create, call_budget_s=150.0, fallback_models=("m2", "m3"))
    t0 = time.monotonic()
    with pytest.raises(LLMError, match="нет соединения"):
        asyncio.run(c.chat_json("sys", "user", RawFindings))
    assert len(calls) == 2                       # не более одного повтора
    assert time.monotonic() - t0 < 5
    assert is_fatal_connection_error(_conn_error(cause))


def test_transient_connection_errors_still_retry(monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_a, **_k: real_sleep(0))     # без экспоненциальных пауз
    calls = []

    async def create(**kw):
        calls.append(kw["model"])
        if len(calls) <= 3:
            raise _conn_error(httpx.ReadError("server disconnected"))     # обрыв, а не отказ в соединении
        return _resp('{"findings": []}')

    c = _client(create, call_budget_s=60.0)
    out, _ = asyncio.run(c.chat_json("sys", "user", RawFindings))
    assert out.findings == [] and len(calls) == 4
    assert not is_fatal_connection_error(openai.APITimeoutError(request=REQ))


# ── #21 гонка удаления модели после 404 ───────────────────────────────────────

def test_404_removed_concurrently_falls_through_to_remaining_model():
    calls = []
    holder: dict = {}

    async def create(**kw):
        calls.append(kw["model"])
        if kw["model"] == "m1":
            # параллельный агент уже получил 404 по m1 и убрал её из очереди
            if "m1" in holder["c"]._models:
                holder["c"]._models.remove("m1")
            raise openai.NotFoundError("model not found", response=httpx.Response(404, request=REQ), body=None)
        return _resp('{"findings": []}')

    c = holder["c"] = _client(create, model="m1", fallback_models=("m2",))
    out, _ = asyncio.run(c.chat_json("sys", "user", RawFindings))
    assert out.findings == [] and calls == ["m1", "m2"]


def test_404_on_last_model_raises():
    async def create(**kw):
        raise openai.NotFoundError("model not found", response=httpx.Response(404, request=REQ), body=None)

    with pytest.raises(LLMError, match="не найдена"):
        asyncio.run(_client(create, model="only").chat_json("sys", "user", RawFindings))


# ── #23 обязательные ключи + #32 токены при ошибке ─────────────────────────────

def test_miskeyed_json_is_retried_and_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "CACHE_DIR", tmp_path)
    answers = iter(['{"issues": [{"title": "x"}]}', '{}', '{"findings": [{"title": "ok"}]}'])
    sent = []

    async def create(**kw):
        sent.append(kw["messages"])
        return _resp(next(answers), tokens=5)

    c = _client(create, cache=True)
    out, tokens = asyncio.run(c.chat_json("sys", "user", RawFindings))
    assert out.findings == [{"title": "ok"}] and tokens == 15 and len(sent) == 3
    assert "не прошёл проверку схемы" in sent[1][-1]["content"]
    assert [p.read_text(encoding="utf-8") for p in tmp_path.glob("*.json")] == ['{"findings": [{"title": "ok"}]}']

    # мис-ключённый ответ до исчерпания попыток: ошибка, в кэш ничего не пишется
    for p in tmp_path.glob("*.json"):
        p.unlink()

    async def bad(**kw):
        return _resp('{"issues": []}', tokens=4)

    c2 = _client(bad, cache=True, max_retries=2)
    with pytest.raises(LLMError) as ei:
        asyncio.run(c2.chat_json("sys", "user2", RawFindings))
    assert ei.value.tokens == 12 and not list(tmp_path.glob("*.json"))
    with pytest.raises(ValueError):
        RawFintech.model_validate({"findings": []})
    with pytest.raises(ValueError):
        RawRevision.model_validate({})
    assert RawRevision.model_validate({"withdrawn": [1]}).findings == []


def test_failed_call_tokens_are_counted_by_agent():
    class _LLM:
        async def chat_json(self, system, user, schema):
            raise LLMError("модель не вернула валидный JSON", tokens=33)

    class _A(Agent):
        name = "completeness"

    ctx = Context(doc=load_text("# ТЗ\nТекст.\n"), cfg=LLMConfig(api_key="k"), llm=_LLM())
    a = _A()
    with pytest.raises(LLMError):
        asyncio.run(a.ask(ctx, "s", "u", RawFindings))
    assert a.tokens == 33


# ── #5 скаляры вместо списков ─────────────────────────────────────────────────

def test_scalar_lists_from_llm_are_normalised():
    f = Finding(title="t", lines=12, evidence=7)
    assert f.lines == [12] and f.evidence == ["7"]
    assert Finding(title="t", lines="123").lines == [123]                 # не [1, 2, 3]
    assert Finding(title="t", lines="L12, L14").lines == [12, 14]
    assert Finding(title="t", lines=["L3", 5.0, "x", True]).lines == [3, 5]
    assert Question(question="q?", related_findings="F-01").related_findings == ["F-01"]
    assert Question(question="q?", related_findings=3, answer_options=None).related_findings == ["3"]
    cc = ControlCheck(control_id="X", evidence="цитата", lines=4)
    assert cc.evidence == ["цитата"] and cc.lines == [4]
    got = parse_findings([{"title": "A", "lines": 5, "evidence": "цитата"},
                          {"title": None}, {"title": "B", "related_findings": 1, "lines": {"a": "L7"}}], "x")
    assert [(g.title, g.lines) for g in got] == [("A", [5]), ("B", [7])]


# ── #6 ревизия не меняет суть находки и не «отмывает» её без цитаты ─────────────

LONG_TAIL = "Перевод между кошельками выполняется в кратчайшие сроки."


def test_revision_keeps_category_severity_and_control_from_original():
    ctx = Context(doc=load_text(f"# ТЗ\n## Требования\n{LONG_TAIL}\n"), cfg=LLMConfig(api_key="k"))
    orig = Finding(agent="consistency", category="ambiguity", severity="minor", title="Срок не измерим",
                   evidence=["моментально"], lines=[30], control_id=None)

    class _A(Agent):
        name = "consistency"

        async def ask(self, c, system, user, schema):
            return schema.model_validate({"findings": [{
                "ref": 1, "category": "security", "severity": "critical", "control_id": "SEC-PCI-01",
                "evidence": [LONG_TAIL], "lines": [3], "title": "Срок перевода не измерим"}], "withdrawn": []})

    rec = SimpleNamespace(best_line=3, best_score=0.5, reason="цитата не найдена")
    out = asyncio.run(revise_findings(_A(), ctx, [(orig, rec)]))
    f = out[0]
    assert (f.category.value, f.severity.value, f.control_id) == ("ambiguity", "minor", None)
    assert f.evidence == [LONG_TAIL] and f.lines == [3] and f.title == "Срок перевода не измерим" and f.revision == 1


CRITIC_TZ = f"""# ТЗ кошелька
## Требования
FR-01. {LONG_TAIL}
FR-02. Лимит пополнения — 150 000 тенге в сутки.
"""


class _NoQuoteRevisionLLM:
    """Consistency цитирует выдуманную строку; при ревизии «исправляет» только номер строки, без цитаты."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.active_model, self.cache_hits, self.transient_errors, self.switches = "fake", 0, 0, []

    async def chat_json(self, system, user, schema):
        if "вернул тебе твои находки" in system:
            data = {"findings": [{"ref": 1, "category": "security", "severity": "critical", "lines": [3],
                                  "evidence": ["перевод выполняется мгновенно при любой нагрузке"]}], "withdrawn": []}
        elif "агент **Consistency**" in system:
            data = {"findings": [{"category": "ambiguity", "severity": "major", "title": "Срок перевода не измерим",
                                  "evidence": ["переводы проходят моментально всегда"], "lines": [30],
                                  "confidence": 0.8}]}
        elif "агент **Reader**" in system:
            data = {"title": "Кошелёк", "domains": ["transfers", "mobile"]}
        elif "агент **Question Generator**" in system:
            data = {"questions": []}
        elif "агент **Scorer**" in system:
            data = {"summary": "Кратко.", "strengths": []}
        else:
            data = {k: [] for k in schema.model_fields}
        return schema.model_validate(data), 5


def test_critic_loop_does_not_restore_revision_without_grounded_quote(monkeypatch):
    monkeypatch.setattr(orchestrator, "LLMClient", _NoQuoteRevisionLLM)
    r = analyze(load_text(CRITIC_TZ), LLMConfig(api_key="fake"), use_llm=True)
    v = r.verifier
    assert v.revision_returned == 1 and v.revision_restored == 0 and not v.restored
    assert not any(f.title == "Срок перевода не измерим" for f in r.findings)
    rec = next(x for x in v.rejected if x.title == "Срок перевода не измерим")
    assert "цитата не найдена" in rec.outcome
    assert v.rejected_hallucinations == len(v.rejected)
    assert not any(f.category.value == "security" and f.agent == "consistency" for f in r.findings)


# ── #7 LLM не меняет применимость контроля ─────────────────────────────────────

def test_llm_cannot_make_not_applicable_control_missing():
    doc = load_text("# ТЗ кредитного калькулятора\n## Требования\nКлиент видит расчёт ставки по займу.\n")
    ctx = Context(doc=doc, cfg=LLMConfig(api_key="k"))
    ctx.domains = ["lending"]                                              # карт нет → PCI не применим

    class _LLM:
        async def chat_json(self, system, user, schema):
            return schema.model_validate({
                "controls": [{"control_id": "SEC-PCI-01", "status": "missing", "comment": "нет PCI"},
                             {"control_id": "SEC-PCI-02", "status": "covered", "evidence": ["Клиент видит расчёт"]}],
                "findings": []}), 1

    ctx.llm = _LLM()
    asyncio.run(FinTechSecurityAgent().run_llm(ctx))
    by_id = {c.control_id: c for c in ctx.controls}
    assert by_id["SEC-PCI-01"].status == "not_applicable" and by_id["SEC-PCI-02"].status == "not_applicable"
    assert not any(f.control_id in ("SEC-PCI-01", "SEC-PCI-02") for f in ctx.findings_by_agent["fintech_security"])


# ── #15/#22 ревизия длинного ТЗ видит нужные строки, а не обрезанное начало ────

def test_revision_context_uses_windows_around_late_lines():
    lines = ["# ТЗ"] + [f"## Раздел {s}\n" + "\n".join(f"Пункт {s}.{i}: система обрабатывает платёж типа {s}-{i}."
                                                    for i in range(30)) for s in range(10)]
    doc = load_text("\n".join(lines))
    ctx = Context(doc=doc, cfg=LLMConfig(api_key="k", max_doc_chars=2000))
    late = len(doc.lines) - 5
    assert f"L{late}:" not in ctx.numbered_doc                            # в обрезанном начале её нет
    f = Finding(title="Находка в конце", evidence=["несуществующая цитата"], lines=[late])
    rec = SimpleNamespace(best_line=late - 2, best_score=0.4, reason="цитата не найдена")
    text = revision_context(ctx, [(f, rec)])
    assert f"L{late}: {doc.lines[late - 1]}" in text and "<tz>" in text and "Оглавление" in text
    assert "L3: " not in text.split("<tz>", 1)[1]                         # начало документа не отправляется
    assert len(text.split("<tz>", 1)[1]) <= 2000 + 200
    short = Context(doc=load_text(CRITIC_TZ), cfg=LLMConfig(api_key="k"))
    assert "L4: FR-02" in revision_context(short, [(f, rec)])              # короткое ТЗ — целиком


# ── #23 Completeness/Consistency не теряют находки правил при пустом ответе LLM ──

class _EmptyLLM:
    async def chat_json(self, system, user, schema):
        return schema.model_validate({k: [] for k in schema.model_fields}), 1


def test_empty_llm_answer_keeps_rule_findings():
    doc = load_text("# ТЗ\n## Требования\nСистема должна работать быстро и удобно.\n"
                    "Лимит перевода — 100 000 тенге.\nЛимит перевода — 300 000 тенге.\nTBD: уточнить позже.\n")
    ctx = Context(doc=doc, cfg=LLMConfig(api_key="k"), llm=_EmptyLLM())
    comp, cons = CompletenessAgent(), ConsistencyAgent()
    offline = Context(doc=doc, cfg=LLMConfig())
    comp.run_rules(offline)
    rules_comp = offline.findings_by_agent["completeness"]
    rules_cons = cons._rule_findings(ctx)
    assert rules_comp and rules_cons
    n, note = asyncio.run(comp.run_llm(ctx))
    assert n == len(rules_comp) and "правил" in note
    n, note = asyncio.run(cons.run_llm(ctx))
    assert n == len(rules_cons) and "правил" in note


# ── #32 отмена фрагментов при ошибке, #35 нормализация статусов при слиянии ─────

def _long_ctx(**kw) -> Context:
    body = ["# ТЗ"] + [f"## Раздел {s}\n" + "\n".join(f"Пункт {s}.{i} описывает обработку платежа." for i in range(20))
                       for s in range(6)]
    return Context(doc=load_text("\n".join(body)), cfg=LLMConfig(api_key="x", max_doc_chars=900), **kw)


def test_chunk_failure_cancels_pending_chunks_and_reports_tokens():
    state = {"cancelled": 0, "started": 0}

    class _LLM:
        async def chat_json(self, system, user, schema):
            state["started"] += 1
            if "Фрагмент 1/" in user:
                await asyncio.sleep(0.01)
                raise LLMError("модель не вернула валидный JSON", tokens=11)
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                state["cancelled"] += 1
                raise
            return schema.model_validate({"findings": []}), 100

    class _A(Agent):
        name = "completeness"

    ctx = _long_ctx(llm=_LLM())
    a = _A()
    t0 = time.monotonic()
    with pytest.raises(LLMError):
        asyncio.run(ask_doc(a, ctx, "sys", "## Подсказки", RawFindings))
    assert time.monotonic() - t0 < 5
    assert state["started"] > 1 and state["cancelled"] == state["started"] - 1
    assert a.tokens == 11


@pytest.mark.parametrize("winner", ["Покрыто", "fully covered", "COVERED"])
def test_chunk_merge_normalises_status_before_ranking(winner):
    ctx = _long_ctx()
    seen = []

    class _A:
        name = "fintech_security"

        async def ask(self, c, system, user, schema):
            seen.append(user)
            k = len(seen)
            st = winner if k == 1 else "partial"
            return schema.model_validate({"controls": [{"control_id": "SEC-X", "status": st, "evidence": [f"q{k}"]}],
                                          "findings": []})

    out = asyncio.run(ask_doc(_A(), ctx, "sys", "## Подсказки", RawFintech))
    assert len(seen) > 2
    (ctrl,) = out.controls
    assert ctrl["status"] == "covered" and "q1" in ctrl["evidence"]
