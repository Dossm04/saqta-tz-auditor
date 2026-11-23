"""Три агента-аналитика, работающих параллельно: Completeness, Consistency, FinTech/Security."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from agents import rules
from agents.base import Agent, Context, doc_model_brief, load_prompt
from knowledge.fintech_controls import CONTROLS_BY_ID, applicable_controls
from knowledge.template import template_findings
from models import ControlCheck, Finding


class RawFindings(BaseModel):
    findings: list[dict[str, Any]] = []


class RawFintech(BaseModel):
    controls: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []


def parse_findings(raw: list[dict[str, Any]], agent: str) -> list[Finding]:
    """Валидирует находки по одной — одна битая находка не роняет весь ответ."""
    out = []
    for item in raw:
        try:
            f = Finding.model_validate({**item, "agent": agent})
        except ValidationError:
            continue
        if f.confidence >= 0.4 and f.title.strip():
            out.append(f)
    return out


def _user_message(ctx: Context, extra: str) -> str:
    return (
        f"## Модель документа (от агента Reader)\n{doc_model_brief(ctx.doc_model)}\n\n"
        f"{extra}\n\n## Текст ТЗ\n{ctx.numbered_doc}"
    )


# ── Completeness ──────────────────────────────────────────────────────────────

class CompletenessAgent(Agent):
    name = "completeness"
    title = "Completeness"
    role = "Пропущенные разделы и требования: цели, роли, NFR, сценарии ошибок, критерии приёмки"
    stage = 2

    def _template(self, ctx: Context) -> tuple[list[Finding], str]:
        """Проверка по шаблону: детерминированная, добавляется в обоих режимах."""
        if ctx.template is None:
            return [], ""
        findings, matches = template_findings(ctx.doc, ctx.template)
        ctx.report_extra["template"] = [
            {"id": m.section.id, "title": m.section.title, "present": m.present, "matched": m.matched,
             "line": m.line, "score": m.score} for m in matches]
        ok = sum(m.present for m in matches)
        return findings, f"; шаблон: {ok}/{len(matches)} разделов"

    def run_rules(self, ctx: Context) -> tuple[int, str]:
        checks = rules.scan_sections(ctx.doc)
        tpl, tpl_note = self._template(ctx)
        findings = rules.section_findings(ctx.doc, checks) + tpl
        ctx.findings_by_agent[self.name] = findings
        present = sum(1 for c in checks if c.present)
        return len(findings), f"разделов найдено {present}/{len(checks)}{tpl_note}"

    async def run_llm(self, ctx: Context) -> tuple[int, str]:
        checks = rules.scan_sections(ctx.doc)
        tpl, tpl_note = self._template(ctx)
        checklist = "\n".join(
            f"- {c.section.id} «{c.section.title}» — пред-проверка: "
            f"{'present' if c.present else 'missing'}"
            f"{' (строки: ' + ', '.join('L' + str(n) for n in c.lines[:5]) + ')' if c.lines else ''}"
            f"{' [нужны числовые значения]' if c.section.needs_number else ''}"
            for c in checks
        )
        extra = f"## Чеклист разделов и пред-проверка\n{checklist}"
        if ctx.template is not None:
            tpl_rows = "\n".join(f"- {t['id']} «{t['title']}»: {'есть' if t['present'] else 'НЕТ'}"
                                 for t in ctx.report_extra.get("template", []))
            extra += (f"\n\n## Обязательный шаблон ТЗ (разделы и наличие в документе)\n{tpl_rows}\n"
                      "Отсутствующие разделы шаблона уже учтены отдельно — не дублируй их; оцени, достаточно ли "
                      "содержательно заполнены присутствующие разделы шаблона.")
        system = load_prompt("completeness", COMMON=load_prompt("_common"))
        out = await self.ask(ctx, system, _user_message(ctx, extra), RawFindings)
        findings = parse_findings(out.findings, self.name) + tpl
        ctx.findings_by_agent[self.name] = findings
        return len(findings), f"пред-проверка: разделов {sum(c.present for c in checks)}/{len(checks)}{tpl_note}"


# ── Consistency ───────────────────────────────────────────────────────────────

class ConsistencyAgent(Agent):
    name = "consistency"
    title = "Consistency"
    role = "Противоречия, дубли, размытые и непроверяемые формулировки (ISO 29148)"
    stage = 2

    def _rule_findings(self, ctx: Context) -> list[Finding]:
        return rules.scan_vague(ctx.doc) + rules.scan_metric_conflicts(ctx.doc) + rules.scan_duplicates(ctx.doc)

    def run_rules(self, ctx: Context) -> tuple[int, str]:
        findings = self._rule_findings(ctx)
        ctx.findings_by_agent[self.name] = findings
        n_contra = sum(1 for f in findings if f.category.value == "contradiction")
        return len(findings), f"противоречий-кандидатов: {n_contra}"

    async def run_llm(self, ctx: Context) -> tuple[int, str]:
        hints = self._rule_findings(ctx)
        hint_text = "\n".join(f"- {h.title} | L{', L'.join(map(str, h.lines))} | {h.evidence[0] if h.evidence else ''}"
                              for h in hints[:40]) or "— нет —"
        system = load_prompt("consistency", COMMON=load_prompt("_common"))
        out = await self.ask(ctx, system, _user_message(ctx, f"## Подсказки пред-проверки\n{hint_text}"), RawFindings)
        findings = parse_findings(out.findings, self.name)
        # «TBD / уточнить позже» — буквальный факт, оставляем всегда
        findings += [h for h in hints if h.title.startswith("Открытый вопрос")]
        ctx.findings_by_agent[self.name] = findings
        return len(findings), f"подсказок правил: {len(hints)}"


# ── FinTech / Security ────────────────────────────────────────────────────────

class FinTechSecurityAgent(Agent):
    name = "fintech_security"
    title = "FinTech/Security"
    role = "ИБ, AML/KYC, PCI DSS, ПДн (локализация в РК), антифрод, идемпотентность, опасные требования"
    stage = 2

    def run_rules(self, ctx: Context) -> tuple[int, str]:
        checks = rules.scan_controls(ctx.doc, ctx.domains)
        findings = [f for c in checks if c.status in ("missing", "partial")
                    if (f := rules.control_finding(CONTROLS_BY_ID[c.control_id], c))]
        findings += rules.scan_antipatterns(ctx.doc)
        ctx.controls = checks
        ctx.findings_by_agent[self.name] = findings
        return len(findings), self._coverage_note(checks)

    async def run_llm(self, ctx: Context) -> tuple[int, str]:
        rule_checks = {c.control_id: c for c in rules.scan_controls(ctx.doc, ctx.domains)}
        applicable = applicable_controls(ctx.domains)
        antipatterns = rules.scan_antipatterns(ctx.doc)
        ctrl_text = "\n".join(
            f"- {c.id} [{c.area}] {c.title} | норма: {c.regulation} | пред-проверка: {rule_checks[c.id].status}"
            f"{' (L' + ', L'.join(map(str, rule_checks[c.id].lines[:4])) + ')' if rule_checks[c.id].lines else ''}"
            for c in applicable
        )
        ap_text = "\n".join(f"- {a.control_id}: {a.title} | L{a.lines[0]}: {a.evidence[0]}" for a in antipatterns) or "— не найдено —"
        system = load_prompt("fintech_security", COMMON=load_prompt("_common"))
        user = _user_message(ctx, f"## Применимые контроли (домены: {', '.join(ctx.domains)})\n{ctrl_text}\n\n"
                                  f"## Кандидаты в опасные требования (пред-проверка)\n{ap_text}")
        out = await self.ask(ctx, system, user, RawFintech)

        # 1) матрица контролей: оценка LLM поверх пред-проверки, но covered/partial — только с подтверждённой цитатой
        merged = dict(rule_checks)
        for item in out.controls:
            try:
                cc = ControlCheck.model_validate(item)
            except ValidationError:
                continue
            base = rule_checks.get(cc.control_id)
            if base is None:
                continue
            cc.title, cc.area, cc.regulation = base.title, base.area, base.regulation
            if cc.status in ("covered", "partial"):
                located = [ctx.doc.locate_quote(q) for q in cc.evidence]
                good = [ln for ln, score in located if ln and score >= 0.75]
                if not good:
                    cc.status = base.status if base.status != "covered" else "partial"
                    cc.comment = (cc.comment + " | цитата LLM не подтверждена — статус по пред-проверке").strip(" |")
                    cc.evidence, cc.lines = base.evidence, base.lines
                else:
                    cc.lines = sorted(set(good))
            merged[cc.control_id] = cc
        ctx.controls = list(merged.values())

        # 2) находки LLM + опасные требования из правил (дубли уберёт Verifier)
        findings = parse_findings(out.findings, self.name) + antipatterns
        # 3) гарантия полноты: каждый missing/partial контроль должен иметь находку
        covered_ids = {f.control_id for f in findings if f.control_id}
        for c in ctx.controls:
            if c.status in ("missing", "partial") and c.control_id not in covered_ids:
                f = rules.control_finding(CONTROLS_BY_ID[c.control_id], c)
                if f:
                    findings.append(f)
        ctx.findings_by_agent[self.name] = findings
        return len(findings), self._coverage_note(ctx.controls)

    @staticmethod
    def _coverage_note(checks: list[ControlCheck]) -> str:
        app = [c for c in checks if c.status != "not_applicable"]
        cov = sum(c.status == "covered" for c in app)
        part = sum(c.status == "partial" for c in app)
        return f"контроли: покрыто {cov}, частично {part}, из {len(app)} применимых"
