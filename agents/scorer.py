"""Scorer — прозрачная детерминированная оценка качества ТЗ + резюме для руководителя.

Баллы считаются формулой (воспроизводимо и объяснимо жюри/заказчику),
LLM только пишет текст резюме.
"""
from __future__ import annotations

import math

from pydantic import BaseModel

from agents.base import Agent, Context, compact_findings, load_prompt
from knowledge.domains import DOMAIN_LABELS
from models import Category, DimensionScore, Severity

# ключ, название, вес, категории находок
DIMENSIONS = [
    ("completeness", "Полнота", 0.20, {Category.completeness}),
    ("clarity", "Однозначность", 0.15, {Category.ambiguity}),
    ("consistency", "Непротиворечивость", 0.10, {Category.contradiction}),
    ("testability", "Тестируемость", 0.15, {Category.testability}),
    ("security", "Безопасность и комплаенс", 0.40, {Category.security, Category.compliance, Category.fintech}),
]
PENALTY = {Severity.critical: 18.0, Severity.major: 8.0, Severity.minor: 3.0}
DECAY = 70.0   # score = 100 * exp(-penalty / DECAY): одна критичная ≈ -23 балла, плавное насыщение


class SummaryOut(BaseModel):
    summary: str
    strengths: list[str] = []


def compute_scores(ctx: Context) -> tuple[list[DimensionScore], int]:
    dims = []
    for key, name, weight, cats in DIMENSIONS:
        fs = [f for f in ctx.findings if f.category in cats]
        penalty = sum(PENALTY[f.severity] * max(0.5, f.confidence) for f in fs)
        score = 100 * math.exp(-penalty / DECAY)
        comment = ""
        if key == "security":
            app = [c for c in ctx.controls if c.status != "not_applicable"]
            if app:
                coverage = 100 * sum(1.0 if c.status == "covered" else 0.5 if c.status == "partial" else 0 for c in app) / len(app)
                score = 0.5 * score + 0.5 * coverage
                comment = f"покрытие контролей {coverage:.0f}%"
        issues = {s.value: sum(1 for f in fs if f.severity == s) for s in Severity}
        dims.append(DimensionScore(key=key, name=name, score=round(score), weight=weight, issues=issues, comment=comment))
    overall = round(sum(d.score * d.weight for d in dims))
    return dims, overall


def verdict_for(overall: int, n_critical: int) -> tuple[str, str]:
    if overall >= 80 and n_critical == 0:
        return "Готово к разработке", "ready"
    if overall >= 55 and n_critical <= 2:
        return "Требует доработки", "needs_work"
    return "Не готово к разработке", "not_ready"


class ScorerAgent(Agent):
    name = "scorer"
    title = "Scorer"
    role = "Детерминированная оценка по 5 измерениям, вердикт готовности, резюме для руководителя"
    stage = 5

    def _score(self, ctx: Context) -> None:
        dims, overall = compute_scores(ctx)
        n_crit = sum(1 for f in ctx.findings if f.severity == Severity.critical)
        verdict, level = verdict_for(overall, n_crit)
        ctx.report_extra.update(dimensions=dims, overall=overall, verdict=verdict, verdict_level=level)

    def run_rules(self, ctx: Context) -> tuple[int, str]:
        self._score(ctx)
        ex = ctx.report_extra
        crit = [f for f in ctx.findings if f.severity == Severity.critical]
        major = [f for f in ctx.findings if f.severity == Severity.major]
        app = [c for c in ctx.controls if c.status != "not_applicable"]
        cov = sum(c.status == "covered" for c in app)
        top = "; ".join(f.title for f in (crit + major)[:3])
        domains = ", ".join(DOMAIN_LABELS.get(d, d) for d in ctx.domains) or "не определены"
        ex["summary"] = (
            f"ТЗ «{ctx.doc_model.title}» (области: {domains}) оценено в {ex['overall']}/100 — {ex['verdict'].lower()}. "
            f"Найдено {len(crit)} критичных и {len(major)} важных замечаний; покрыто {cov} из {len(app)} применимых "
            f"финтех/ИБ-контролей. Главные риски: {top or 'не выявлены'}. "
            f"В первую очередь закройте вопросы приоритета P0 ({sum(q.priority.value == 'P0' for q in ctx.questions)} шт.)."
        )
        ex["strengths"] = self._strengths(ctx)
        return len(ex["dimensions"]), f"итог {ex['overall']}/100"

    async def run_llm(self, ctx: Context) -> tuple[int, str]:
        self._score(ctx)
        ex = ctx.report_extra
        dims = "\n".join(f"- {d.name}: {d.score}/100 {d.comment}" for d in ex["dimensions"])
        covered = [c.title for c in ctx.controls if c.status == "covered"]
        user = (f"## ТЗ\n{ctx.doc_model.title}: {ctx.doc_model.summary}\n\n## Оценки\nИтог: {ex['overall']}/100, вердикт: {ex['verdict']}\n{dims}\n\n"
                f"## Покрытые контроли\n{'; '.join(covered) or '—'}\n\n## Находки\n{compact_findings(ctx.findings, 25)}")
        out = await self.ask(ctx, load_prompt("scorer"), user, SummaryOut)
        ex["summary"] = out.summary
        ex["strengths"] = out.strengths[:4] or self._strengths(ctx)
        return len(ex["dimensions"]), f"итог {ex['overall']}/100"

    @staticmethod
    def _strengths(ctx: Context) -> list[str]:
        out = []
        covered = [c for c in ctx.controls if c.status == "covered"]
        if covered:
            out.append("Описаны контроли: " + ", ".join(c.title for c in covered[:4]))
        if len(ctx.doc_model.requirements) >= 10:
            out.append(f"Выделено {len(ctx.doc_model.requirements)} требований — хорошая основа для трассировки")
        if ctx.doc_model.integrations:
            out.append("Перечислены интеграции: " + ", ".join(ctx.doc_model.integrations[:5]))
        return out or ["Документ структурирован по разделам"]
