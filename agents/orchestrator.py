"""Orchestrator — управляет графом агентов и собирает итоговый отчёт.

Граф (стадии выполняются последовательно, агенты внутри стадии — параллельно):

  1. Reader (правила, мгновенно)              → DocumentModel (домены, требования)
  2. Completeness ║ Consistency ║ FinTech/Sec  → находки + матрица контролей
     ║ Reader (LLM-обогащение модели документа — параллельно, не блокирует аналитиков)
  3. Verifier (Critic)                        → заземление, дедупликация, консенсус
     ↺ critic loop (только LLM, 1 раунд): отклонённые находки возвращаются агенту-автору
       с причиной и ближайшей строкой; агент исправляет или снимает; Verifier перепроверяет
  4. Question Generator                       → повестка встречи: закрытые вопросы
  5. Scorer                                   → оценки, вердикт, резюме

Каждая передача результата фиксируется как сообщение агент → агент (Report.messages).
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

from agents import rules
from agents.analyzers import CompletenessAgent, ConsistencyAgent, FinTechSecurityAgent, revise_findings
from agents.base import Agent, Context, LLMClient
from agents.questions import QuestionAgent
from agents.reader import ReaderAgent
from agents.scorer import ScorerAgent
from agents.verifier import VerifierAgent
from config import LLMConfig
from knowledge.domains import DOMAIN_LABELS
from knowledge.fintech_controls import CONTROLS_BY_ID
from models import AgentRun, Finding, Report, Severity
from parsing import Document

EventCallback = Callable[[str, Agent, AgentRun | None], None]
MAX_REVISION_ROUNDS = 1
_AUX_AGENTS = {"verifier", "llm", "critic"}


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

    async def _run_agent(self, agent: Agent, ctx: Context, force_rules: bool = False) -> AgentRun:
        self.on_event("start", agent, None)
        run = await agent.run(ctx, force_rules=force_rules)
        self.on_event("done", agent, run)
        return run

    async def run(self, doc: Document, template: Document | None = None) -> Report:
        started = time.perf_counter()
        ctx = Context(doc=doc, cfg=self.cfg, llm=LLMClient(self.cfg) if self.use_llm else None, template=template)
        ctx.started = started
        llm_mode = ctx.llm is not None
        reader_s, analysts, verifier_s, questions_s, scorer_s = build_pipeline()
        reader = reader_s[0]
        trace: list[AgentRun] = []

        # 1) Reader правилами — мгновенно; аналитики стартуют сразу
        ctx.send("orchestrator", reader.name, "document", doc.non_empty_count, f"ТЗ «{doc.name}», строк: {len(doc.lines)}")
        r1 = await self._run_agent(reader, ctx, force_rules=True)
        if llm_mode:
            r1.note = "мгновенный разбор правилами (LLM-обогащение — параллельно со стадией 2); " + r1.note
        trace.append(r1)
        rules_domains = list(ctx.domains)
        for a in analysts:
            ctx.send(reader.name, a.name, "doc_model", len(ctx.domains),
                     f"домены: {', '.join(ctx.domains) or '—'}; требований: {len(ctx.doc_model.requirements)}")

        # 2) аналитики ║ LLM-обогащение Reader
        jobs = [self._run_agent(a, ctx) for a in analysts]
        if llm_mode:
            jobs.append(self._run_reader_enrich(reader, ctx))
        runs = await asyncio.gather(*jobs)
        trace.extend(runs)
        if llm_mode:
            self._topup_domains(ctx, rules_domains, trace)
        verifier = verifier_s[0]
        for a in analysts:
            fs = ctx.findings_by_agent.get(a.name, [])
            ctx.send(a.name, verifier.name, "findings", len(fs),
                     f"критичных {sum(f.severity == Severity.critical for f in fs)}, с цитатами {sum(bool(f.evidence) for f in fs)}")

        # 3) Verifier + critic loop
        trace.append(await self._run_agent(verifier, ctx))
        if llm_mode:
            for _ in range(MAX_REVISION_ROUNDS):
                critic = await self._critic_round(ctx, analysts, verifier, trace)
                if critic is None:
                    break
                trace.append(critic)

        # 4) вопросы, 5) оценка
        qa, sc = questions_s[0], scorer_s[0]
        ctx.send(verifier.name, qa.name, "verified_findings", len(ctx.findings),
                 f"после проверки: {len(ctx.findings)} (дублей {ctx.verifier.merged_duplicates}, "
                 f"отсечено {ctx.verifier.rejected_hallucinations})")
        if llm_mode and trace and any(r.agent == reader.name and r.mode == "llm" for r in trace):
            ctx.send(reader.name, qa.name, "doc_model", len(ctx.doc_model.actors) + len(ctx.doc_model.integrations),
                     "уточнённая модель документа: участники, интеграции, данные")
        trace.append(await self._run_agent(qa, ctx))
        ctx.send(qa.name, sc.name, "questions", len(ctx.questions),
                 f"P0: {sum(q.priority.value == 'P0' for q in ctx.questions)}")
        ctx.send(verifier.name, sc.name, "verified_findings", len(ctx.findings), "находки для формулы оценки")
        trace.append(await self._run_agent(sc, ctx))
        ex = ctx.report_extra
        ctx.send(sc.name, "orchestrator", "score", ex.get("overall", 0), str(ex.get("verdict", "")))

        modes = {r.mode for r in trace
                 if r.agent not in _AUX_AGENTS and not (llm_mode and r.agent == reader.name and r.stage == 1)}
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
            messages=ctx.messages, extra_questions=ex.get("extra_questions", []),
            **({"question_quality": ex["question_quality"]} if ex.get("question_quality") is not None else {}),
        )

    async def _run_reader_enrich(self, reader: Agent, ctx: Context) -> AgentRun:
        self.on_event("start", reader, None)
        run = await reader.run(ctx)
        run.stage = 2
        run.title = f"{reader.title} (LLM)"
        self.on_event("done", reader, run)
        return run

    def _topup_domains(self, ctx: Context, rules_domains: list[str], trace: list[AgentRun]) -> None:
        """LLM-Reader нашёл домены, которых не было у правил: проверяем ставшие применимыми контроли правилами
        (FinTech-агент уже отработал по доменам правил, повторный LLM-вызов не нужен)."""
        added = [d for d in ctx.domains if d not in rules_domains]
        if not added:
            return
        by_id = {c.control_id: c for c in ctx.controls}
        new_findings = []
        for cid, c in CONTROLS_BY_ID.items():
            cur = by_id.get(cid)
            if cur is None or cur.status != "not_applicable" or not set(added) & set(c.applies_to):
                continue
            chk = rules.check_control(ctx.doc, c)
            chk.comment = f"применим по домену от LLM-Reader ({', '.join(sorted(set(added) & set(c.applies_to)))}); {chk.comment}"
            by_id[cid] = chk
            f = rules.control_finding(c, chk)
            if f:
                new_findings.append(f)
        ctx.controls = [by_id[c.control_id] for c in ctx.controls]
        ctx.findings_by_agent.setdefault("fintech_security", []).extend(new_findings)
        ctx.send("reader", "fintech_security", "domains_added", len(added),
                 f"новые домены: {', '.join(DOMAIN_LABELS.get(d, d) for d in added)}; "
                 f"дополнительных находок по контролям: {len(new_findings)}")

    async def _critic_round(self, ctx: Context, analysts: list[Agent], verifier: VerifierAgent,
                            trace: list[AgentRun]) -> AgentRun | None:
        """Обратная связь Verifier → аналитик: находки, отклонённые из-за цитат, возвращаются автору
        с причиной и ближайшей строкой; автор исправляет или снимает; Verifier перепроверяет."""
        llm_agents = {r.agent for r in trace if r.mode == "llm"}
        by_name = {a.name: a for a in analysts}
        groups: dict[str, list] = {}
        for orig, rec in ctx.rejected:
            if (orig.agent in by_name and orig.agent in llm_agents and not orig.revision
                    and not rec.reason.startswith("низкая уверенность")):
                groups.setdefault(orig.agent, []).append((orig, rec))
        if not groups:
            return None
        started = time.perf_counter()
        prev = ctx.verifier
        tokens_before = sum(by_name[n].tokens for n in groups)
        results = await asyncio.gather(*(revise_findings(by_name[n], ctx, items) for n, items in groups.items()),
                                       return_exceptions=True)
        returned = fixed = withdrawn = 0
        revised_of: dict[int, Finding] = {}          # id(исходной находки) → исправленная
        withdrawn_recs: list = []
        notes = []
        for (name, items), res in zip(groups.items(), results):
            returned += len(items)
            ctx.send(verifier.name, name, "rejected", len(items),
                     "; ".join(f"«{o.title[:50]}»: {r.reason}" for o, r in items[:3]))
            if isinstance(res, BaseException):
                for _, r in items:
                    r.outcome = "агент не ответил — находка остаётся отклонённой"
                notes.append(f"{name}: ошибка ревизии ({str(res)[:60]})")
                continue
            for k, (o, r) in enumerate(items):
                if k in res:
                    revised_of[id(o)] = res[k]
                else:
                    r.outcome = "снято агентом после замечания Verifier"
                    withdrawn_recs.append(r)
            n_fixed, n_withdrawn = len(res), len(items) - len(res)
            fixed += n_fixed
            withdrawn += n_withdrawn
            returned_ids = {id(o) for o, _ in items}
            keep = [f for f in ctx.findings_by_agent.get(name, []) if id(f) not in returned_ids]
            ctx.findings_by_agent[name] = keep + [res[k] for k in sorted(res)]
            ctx.send(name, verifier.name, "revision", n_fixed, f"исправлено {n_fixed}, снято {n_withdrawn}")
            notes.append(f"{name}: исправлено {n_fixed}, снято {n_withdrawn}")

        restored_recs = []
        if fixed:
            # перепроверка всех находок (Verifier работает с копиями — исходные находки агентов не мутируют)
            verifier.run_rules(ctx)
            st = ctx.verifier
            failed_again = {id(o) for o, _ in ctx.rejected if o.revision}
            for o, r in ctx.rejected:
                if o.revision:
                    r.outcome = "исправлено агентом, но цитата снова не подтверждена"
            for items in groups.values():
                for o, r in items:
                    rv = revised_of.get(id(o))
                    if rv is not None and id(rv) not in failed_again:
                        r.outcome = "исправлено агентом и подтверждено повторно"
                        restored_recs.append(r)
            st.rejected = st.rejected + withdrawn_recs       # снятые агентом в повторную проверку не попали
            st.input_findings = prev.input_findings
        else:
            st = prev                                         # всё снято: перепроверять нечего
        st.restored = restored_recs
        st.rejected_hallucinations = len(st.rejected)
        st.revision_returned, st.revision_restored, st.revision_withdrawn = returned, len(restored_recs), withdrawn
        ctx.verifier = st
        return AgentRun(agent="critic", title="Critic loop (Verifier ↔ аналитики)", stage=3, mode="llm",
                        produced=len(restored_recs), duration_s=round(time.perf_counter() - started, 2),
                        tokens=sum(by_name[n].tokens for n in groups) - tokens_before,
                        note=f"возвращено {returned}, исправлено {fixed}, снято {withdrawn}, "
                             f"восстановлено {len(restored_recs)}" + (f" | {'; '.join(notes)}" if notes else ""))


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
