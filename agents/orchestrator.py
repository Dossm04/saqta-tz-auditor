"""Orchestrator — управляет графом агентов и собирает итоговый отчёт.

Граф (стадии выполняются последовательно, агенты внутри стадии — параллельно):

  1. Reader                                   → DocumentModel (домены, требования)
  2. Completeness ║ Consistency ║ FinTech/Sec  → находки + матрица контролей
  3. Verifier (Critic)                        → заземление, дедупликация, консенсус
  4. Question Generator                       → приоритизированные вопросы
  5. Scorer                                   → оценки, вердикт, резюме
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

from agents.analyzers import CompletenessAgent, ConsistencyAgent, FinTechSecurityAgent
from agents.base import Agent, Context, LLMClient
from agents.questions import QuestionAgent
from agents.reader import ReaderAgent
from agents.scorer import ScorerAgent
from agents.verifier import VerifierAgent
from config import LLMConfig
from models import AgentRun, Report
from parsing import Document

EventCallback = Callable[[str, Agent, AgentRun | None], None]


def build_pipeline() -> list[list[Agent]]:
    return [
        [ReaderAgent()],
        [CompletenessAgent(), ConsistencyAgent(), FinTechSecurityAgent()],
        [VerifierAgent()],
        [QuestionAgent()],
        [ScorerAgent()],
    ]


PIPELINE_INFO = [[(a.name, a.title, a.role, a.uses_llm) for a in stage] for stage in build_pipeline()]


class Orchestrator:
    def __init__(self, cfg: LLMConfig, use_llm: bool = True, on_event: EventCallback | None = None):
        self.cfg = cfg
        self.use_llm = use_llm and cfg.enabled
        self.on_event = on_event or (lambda *a: None)

    async def _run_agent(self, agent: Agent, ctx: Context) -> AgentRun:
        self.on_event("start", agent, None)
        run = await agent.run(ctx)
        self.on_event("done", agent, run)
        return run

    async def run(self, doc: Document, template: Document | None = None) -> Report:
        started = time.perf_counter()
        ctx = Context(doc=doc, cfg=self.cfg, llm=LLMClient(self.cfg) if self.use_llm else None, template=template)
        trace: list[AgentRun] = []
        for stage in build_pipeline():
            runs = await asyncio.gather(*(self._run_agent(a, ctx) for a in stage))
            trace.extend(runs)

        ex = ctx.report_extra
        modes = {r.mode for r in trace if r.agent != "verifier"}
        mode = "llm" if modes == {"llm"} else "rules" if modes == {"rules"} else "hybrid"
        model = "rules-engine"
        if ctx.llm is not None:
            model = ctx.llm.active_model
            notes = []
            if ctx.llm.cache_hits:
                notes.append(f"ответов из кэша: {ctx.llm.cache_hits}")
            if ctx.llm.switches:
                notes.append(f"сбоев провайдера: {ctx.llm.transient_errors}; переключения моделей: "
                             + "; ".join(ctx.llm.switches[:6]))
            if notes:
                trace.append(AgentRun(agent="llm", title="LLM-клиент", stage=0, mode="llm", note=" | ".join(notes)))
        return Report(
            document_name=doc.name, document=ctx.doc_model,
            overall_score=ex.get("overall", 0), verdict=ex.get("verdict", "—"), verdict_level=ex.get("verdict_level", "not_ready"),
            summary=ex.get("summary", ""), strengths=ex.get("strengths", []), dimensions=ex.get("dimensions", []),
            findings=ctx.findings, questions=ctx.questions, controls=ctx.controls, trace=trace, verifier=ctx.verifier,
            mode=mode, model=model,
            total_duration_s=round(time.perf_counter() - started, 2), total_tokens=sum(r.tokens for r in trace),
            line_count=len(doc.lines), template_check=ex.get("template", []),
            manual_review_hours=manual_review_hours(doc, ctx),
        )


def manual_review_hours(doc: Document, ctx: Context) -> float:
    """Консервативная оценка ручного ревью тем же объёмом: чтение + разбор требований +
    проверка ИБ/комплаенса + подготовка вопросов. Формула открыта, чтобы её можно было оспорить."""
    minutes = (30                                    # первичное чтение и контекст
               + 2.0 * len(ctx.doc_model.requirements)  # разбор каждого требования
               + 0.4 * doc.non_empty_count             # сверка текста
               + (45 if ctx.domains else 0)            # проверка ИБ и регуляторики специалистом
               + 3.0 * len(ctx.questions))             # формулировка уточняющих вопросов
    return round(minutes / 60 * 2) / 2               # с точностью до получаса


def analyze(doc: Document, cfg: LLMConfig | None = None, use_llm: bool = True,
            on_event: EventCallback | None = None, template: Document | None = None) -> Report:
    """Синхронная обёртка (для CLI и Streamlit)."""
    cfg = cfg or LLMConfig.from_env()
    return asyncio.run(Orchestrator(cfg, use_llm=use_llm, on_event=on_event).run(doc, template=template))
