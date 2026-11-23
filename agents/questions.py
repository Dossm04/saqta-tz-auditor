"""Question Generator — превращает проверенные находки в приоритизированные уточняющие вопросы."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, ValidationError

from agents.base import Agent, Context, compact_findings, load_prompt
from knowledge.antipatterns import ANTIPATTERNS
from knowledge.fintech_controls import CONTROLS_BY_ID
from knowledge.tz_structure import SECTIONS
from models import Category, Finding, Priority, Question, Severity

_SECTIONS = {s.id: s for s in SECTIONS}
_APS = {a.id: a for a in ANTIPATTERNS}
_PRIO = {Severity.critical: Priority.P0, Severity.major: Priority.P1, Severity.minor: Priority.P2}
_CAT_ADDRESSEE = {
    Category.compliance: "Комплаенс", Category.security: "ИБ", Category.fintech: "Архитектор / Бэкенд",
    Category.contradiction: "Бизнес-заказчик", Category.ambiguity: "Бизнес-заказчик",
    Category.testability: "Бизнес-заказчик / QA", Category.completeness: "Бизнес-заказчик",
}
LIMITS = {Priority.P0: 10, Priority.P1: 8, Priority.P2: 4}   # вопросы на одну встречу, остальное — в находках


class RawQuestions(BaseModel):
    questions: list[dict[str, Any]] = []


def _kb_meta(f: Finding) -> tuple[str | None, str | None, list[str]]:
    """(why, addressee, answer_options) из базы знаний по control_id."""
    cid = f.control_id or ""
    if cid in CONTROLS_BY_ID:
        c = CONTROLS_BY_ID[cid]
        return c.why, c.addressee, list(c.answer_options)
    if cid in _SECTIONS:
        s = _SECTIONS[cid]
        return s.why, s.addressee, []
    if cid in _APS:
        return _APS[cid].description, "ИБ / Комплаенс", []
    return None, None, []


def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.8


class QuestionAgent(Agent):
    name = "questions"
    title = "Question Generator"
    role = "Формирует приоритизированные уточняющие вопросы с вариантами ответа и адресатом"
    stage = 4

    # ── офлайн-режим ──
    def run_rules(self, ctx: Context) -> tuple[int, str]:
        findings = ctx.findings
        minor_vague = [f for f in findings if f.category in (Category.ambiguity, Category.testability)
                       and f.severity == Severity.minor and not f.control_id]
        qs: list[Question] = []
        for f in findings:
            if f in minor_vague:
                continue
            why, addressee, options = _kb_meta(f)
            text = f.question or f"Как должно быть описано требование «{f.title}»?"
            existing = next((q for q in qs if _similar(q.question, text)), None)
            if existing:
                existing.related_findings.append(f.id)
                continue
            qs.append(Question(
                question=text, why=(why or f.description)[:260], priority=_PRIO[f.severity],
                addressee=addressee or _CAT_ADDRESSEE.get(f.category, "Бизнес-заказчик"),
                related_findings=[f.id], answer_options=options, blocking=f.severity == Severity.critical,
            ))
        if minor_vague:
            terms = "; ".join(f"{f.title.split(':', 1)[-1].strip()} (L{f.lines[0]})" for f in minor_vague[:8] if f.lines)
            qs.append(Question(
                question=f"Уточните измеримые значения для размытых формулировок: {terms}.",
                why="Размытые формулировки разработчики и тестировщики трактуют по-разному — споры при приёмке.",
                priority=Priority.P2, addressee="Бизнес-заказчик", related_findings=[f.id for f in minor_vague],
            ))
        ctx.questions = self._finalize(qs, ctx.findings)
        return len(ctx.questions), f"из {len(findings)} находок"

    # ── LLM-режим ──
    async def run_llm(self, ctx: Context) -> tuple[int, str]:
        if not ctx.findings:
            ctx.questions = []
            return 0, "нет находок"
        user = (f"## ТЗ: {ctx.doc_model.title}\n{ctx.doc_model.summary}\n\n"
                f"## Проверенные находки\n{compact_findings(ctx.findings, 70)}")
        out = await self.ask(ctx, load_prompt("questions"), user, RawQuestions)
        known = {f.id for f in ctx.findings}
        qs: list[Question] = []
        for item in out.questions:
            try:
                q = Question.model_validate(item)
            except ValidationError:
                continue
            q.related_findings = [r for r in q.related_findings if r in known]
            if q.question.strip():
                qs.append(q)
        # гарантия: каждая критичная находка покрыта вопросом
        covered = {r for q in qs for r in q.related_findings}
        for f in ctx.findings:
            if f.severity == Severity.critical and f.id not in covered:
                why, addressee, options = _kb_meta(f)
                qs.append(Question(question=f.question or f"Как будет закрыт риск «{f.title}»?",
                                   why=(why or f.description)[:260], priority=Priority.P0,
                                   addressee=addressee or _CAT_ADDRESSEE.get(f.category, "Бизнес-заказчик"),
                                   related_findings=[f.id], answer_options=options, blocking=True))
        ctx.questions = self._finalize(qs, ctx.findings)
        return len(ctx.questions), f"вопросов по {len(covered)} находкам"

    @staticmethod
    def _finalize(qs: list[Question], findings: list[Finding]) -> list[Question]:
        by_id = {f.id: f for f in findings}
        for q in qs:
            if q.context:
                continue
            for fid in q.related_findings:
                f = by_id.get(fid)
                if f and f.evidence and f.lines:
                    q.context = f"L{f.lines[0]}: «{f.evidence[0][:200]}»"
                    break
        order = {Priority.P0: 0, Priority.P1: 1, Priority.P2: 2}
        qs.sort(key=lambda q: (order[q.priority], not q.blocking))
        counts = {p: 0 for p in Priority}
        out = []
        for q in qs:
            if counts[q.priority] >= LIMITS[q.priority]:
                continue
            counts[q.priority] += 1
            out.append(q)
        for i, q in enumerate(out, start=1):
            q.id = f"Q-{i:02d}"
        return out
