"""Три агента-аналитика, работающих параллельно: Completeness, Consistency, FinTech/Security."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from agents import rules
from agents.base import TZ_NOTE, Agent, Context, doc_model_brief, load_prompt, tz_wrap
from knowledge.fintech_controls import CONTROLS_BY_ID, applicable_controls
from knowledge.template import template_findings
from models import CONTROL_STATUSES, ControlCheck, Finding

# поля, которыми владеет конвейер (ID, авторство, результат проверки): из ответа LLM они не принимаются
SERVER_OWNED = frozenset({"id", "agent", "sources", "verified", "verification_note", "revision"})


class RawFindings(BaseModel):
    findings: list[dict[str, Any]] = []


class RawFintech(BaseModel):
    controls: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []


def parse_findings(raw: list[dict[str, Any]], agent: str) -> list[Finding]:
    """Валидирует находки по одной — одна битая находка не роняет весь ответ."""
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        clean = {k: v for k, v in item.items() if k not in SERVER_OWNED}
        try:
            f = Finding.model_validate({**clean, "agent": agent})
        except ValidationError:
            continue
        if f.confidence >= 0.4 and f.title.strip():
            out.append(f)
    return out


def _user_message(ctx: Context, extra: str) -> str:
    return (
        f"## Модель документа (от агента Reader)\n{doc_model_brief(ctx.doc_model)}\n\n"
        f"{extra}\n\n## Текст ТЗ\n{TZ_NOTE}\n{ctx.tz_block}"
    )


_STATUS_RANK = {"covered": 3, "partial": 2, "missing": 1, "not_applicable": 0}


def chunk_size(ctx: Context) -> int:
    return int(getattr(ctx.cfg, "chunk_chars", 0) or ctx.cfg.max_doc_chars)


def is_long(ctx: Context) -> bool:
    """ТЗ не помещается в один запрос: вместо молчаливой обрезки — фрагменты по разделам."""
    return len(ctx.doc.numbered()) > ctx.cfg.max_doc_chars


def _chunk_message(ctx: Context, extra: str, k: int, n: int, a: int, b: int, text: str) -> str:
    return (
        f"## Модель документа (от агента Reader)\n{doc_model_brief(ctx.doc_model)}\n\n{extra}\n\n"
        f"## Оглавление всего ТЗ\n{ctx.doc.outline()}\n\n"
        f"## Фрагмент {k}/{n}: L{a}–L{b}\nТЗ длинное и передано частями. Анализируй только этот фрагмент; не утверждай, "
        f"что темы нет в ТЗ, если оглавление показывает подходящий раздел вне фрагмента (об отсутствии суди по "
        f"оглавлению и пред-проверке — она видит весь документ).\n{TZ_NOTE}\n{tz_wrap(text)}"
    )


async def ask_doc(agent: Agent, ctx: Context, system: str, extra: str, schema: type[BaseModel]) -> BaseModel:
    """Один запрос с полным ТЗ или, для длинного ТЗ, по запросу на каждый фрагмент (параллельно, семафор
    клиента ограничивает нагрузку) с объединением ответов: находки складываются (дубли уберёт Verifier),
    статус контроля — лучший из фрагментов с объединением цитат."""
    if not is_long(ctx):
        return await agent.ask(ctx, system, _user_message(ctx, extra), schema)
    import asyncio

    parts = ctx.doc.chunks(chunk_size(ctx))
    outs = await asyncio.gather(*(agent.ask(ctx, system, _chunk_message(ctx, extra, k, len(parts), a, b, text), schema)
                                  for k, (a, b, text) in enumerate(parts, start=1)))
    ctx.send("orchestrator", agent.name, "chunks", len(parts), f"ТЗ разбито на {len(parts)} фрагментов по разделам")
    data: dict[str, list] = {}
    for o in outs:
        for key, val in o.model_dump().items():
            if isinstance(val, list):
                data.setdefault(key, []).extend(val)
    if "controls" in data:
        best: dict[str, dict] = {}
        for item in data["controls"]:
            if not isinstance(item, dict) or not item.get("control_id"):
                continue
            cid = item["control_id"]
            st = str(item.get("status", "")).lower()
            cur = best.get(cid)
            if cur is None or _STATUS_RANK.get(st, 1) > _STATUS_RANK.get(str(cur.get("status", "")).lower(), 1):
                ev = (cur or {}).get("evidence", []) or []
                best[cid] = {**item, "evidence": list(dict.fromkeys([*(item.get("evidence") or []), *ev]))}
            else:
                cur["evidence"] = list(dict.fromkeys([*(cur.get("evidence") or []), *(item.get("evidence") or [])]))
        data["controls"] = list(best.values())
    return schema.model_validate(data)


def merge_control_status(cc: ControlCheck, base: ControlCheck, locate) -> ControlCheck:
    """Сводит оценку контроля от LLM с пред-проверкой правил.

    * нераспознанный статус → статус пред-проверки;
    * «не применимо» для применимого контроля принимается только при пустой пред-проверке и с обоснованием;
    * covered/partial — только с подтверждённой цитатой."""
    cc.title, cc.area, cc.regulation = base.title, base.area, base.regulation
    if cc.status not in CONTROL_STATUSES:
        cc.status, cc.evidence, cc.lines = base.status, base.evidence, base.lines
        cc.comment = (cc.comment + " | статус LLM не распознан — по пред-проверке").strip(" |")
    if cc.status == "not_applicable" and base.status != "not_applicable":
        if base.status != "missing" or not cc.comment.strip():
            cc.status, cc.evidence, cc.lines = base.status, base.evidence, base.lines
            cc.comment = (cc.comment + " | LLM пометил «не применимо» без оснований — статус по пред-проверке").strip(" |")
        else:
            cc.comment += " | LLM: не применимо (проверьте обоснование)"
    if cc.status in ("covered", "partial"):
        located = [locate(q) for q in cc.evidence]
        good = [ln for ln, score in located if ln and score >= 0.75]
        if not good:
            cc.status = base.status if base.status != "covered" else "partial"
            cc.comment = (cc.comment + " | цитата LLM не подтверждена — статус по пред-проверке").strip(" |")
            cc.evidence, cc.lines = base.evidence, base.lines
        else:
            cc.lines = sorted(set(good))
    return cc


# ── Ревизия по замечаниям Verifier (critic loop, только LLM-режим) ─────────────

REVISE_PROMPT = """Ты — агент-аналитик в мульти-агентной системе проверки ТЗ финтех-проектов. Агент **Verifier (Critic)**
вернул тебе твои находки: их цитаты не найдены в тексте ТЗ (или ссылаются на несуществующие строки).
Для каждой возвращённой находки реши одно из двух:
1. **Исправить**: если дефект в ТЗ действительно есть — найди и процитируй ДОСЛОВНО строку ТЗ (поле `evidence`),
   укажи её номер (`lines`). Используй подсказку «ближайшая строка», если она подходит. Можно уточнить заголовок и описание.
2. **Снять**: если подтверждающей строки в ТЗ нет — не включай находку в ответ и перечисли её номер в `withdrawn`.
Не добавляй новых находок. Поле `ref` — номер возвращённой находки.
""" + TZ_NOTE + """
Ответ — ТОЛЬКО JSON: {"findings": [{"ref": 1, "category": "...", "severity": "...", "title": "...", "description": "...",
"evidence": ["дословная цитата"], "lines": [12], "recommendation": "...", "suggested_text": "...", "control_id": null,
"confidence": 0.8}], "withdrawn": [2]}"""


class RawRevision(BaseModel):
    findings: list[dict[str, Any]] = []
    withdrawn: list[Any] = []


async def revise_findings(agent: Agent, ctx: Context, items: list[tuple[Finding, Any]]) -> dict[int, Finding]:
    """Один раунд ревизии: агент-автор исправляет цитаты или снимает находки.
    Возвращает {индекс возвращённой находки (с 0): исправленная находка с revision=1}; остальные сняты."""
    rows = []
    for k, (f, rec) in enumerate(items, start=1):
        near = ""
        if getattr(rec, "best_line", None):
            near = f"\n   ближайшая строка L{rec.best_line} ({rec.best_score:.0%}): «{ctx.doc.line(rec.best_line).strip()[:200]}»"
        quotes = "; ".join(f"«{q[:160]}»" for q in f.evidence[:3]) or "—"
        rows.append(f"{k}. [{f.severity.value}/{f.category.value}] {f.title}\n   описание: {f.description[:240]}\n"
                    f"   твоя цитата: {quotes}; строки: {', '.join('L' + str(n) for n in f.lines[:4]) or '—'}\n"
                    f"   замечание Verifier: {getattr(rec, 'reason', '')}{near}")
    user = f"## Возвращённые находки\n" + "\n".join(rows) + f"\n\n## Текст ТЗ\n{tz_wrap(ctx.numbered_doc)}"
    out = await agent.ask(ctx, REVISE_PROMPT, user, RawRevision)
    withdrawn = set()
    for w in out.withdrawn:
        try:
            withdrawn.add(int(str(w).strip()))
        except ValueError:
            continue
    revised: dict[int, Finding] = {}
    for item in out.findings:
        if not isinstance(item, dict):
            continue
        try:
            ref = int(str(item.get("ref", "")).strip())
        except ValueError:
            continue
        if not 1 <= ref <= len(items) or ref - 1 in revised or ref in withdrawn:
            continue
        orig = items[ref - 1][0]
        base = {"control_id": orig.control_id, "category": orig.category.value, "severity": orig.severity.value,
                "title": orig.title, "description": orig.description, "recommendation": orig.recommendation,
                "suggested_text": orig.suggested_text, "regulation": orig.regulation, "question": orig.question,
                "confidence": orig.confidence}
        parsed = parse_findings([{**base, **{k: v for k, v in item.items() if k != "ref" and v not in (None, "")}}],
                                agent.name)
        if parsed:
            parsed[0].revision = 1
            revised[ref - 1] = parsed[0]
    return revised


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
        ctx.send("rules", self.name, "hints", len(checks),
                 f"пред-проверка разделов: есть {sum(c.present for c in checks)} из {len(checks)}")
        out = await ask_doc(self, ctx, system, extra, RawFindings)
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
        ctx.send("rules", self.name, "hints", len(hints),
                 f"подсказки: размытых {sum(h.category.value in ('ambiguity', 'testability') for h in hints)}, "
                 f"числовых конфликтов {sum(h.category.value == 'contradiction' for h in hints)}")
        out = await ask_doc(self, ctx, system, f"## Подсказки пред-проверки\n{hint_text}", RawFindings)
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
        extra = (f"## Применимые контроли (домены: {', '.join(ctx.domains)})\n{ctrl_text}\n\n"
                 f"## Кандидаты в опасные требования (пред-проверка)\n{ap_text}")
        ctx.send("rules", self.name, "hints", len(applicable) + len(antipatterns),
                 f"применимых контролей {len(applicable)}, кандидатов в опасные требования {len(antipatterns)}")
        out = await ask_doc(self, ctx, system, extra, RawFintech)

        # 1) матрица контролей: оценка LLM поверх пред-проверки, но covered/partial — только с подтверждённой цитатой,
        #    «не применимо» — только с обоснованием, нераспознанный статус — по пред-проверке
        merged = dict(rule_checks)
        for item in out.controls:
            if not isinstance(item, dict):
                continue
            try:
                cc = ControlCheck.model_validate(item)
            except ValidationError:
                continue
            base = rule_checks.get(cc.control_id)
            if base is None:
                continue
            merged[cc.control_id] = merge_control_status(cc, base, ctx.doc.locate_quote)
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
