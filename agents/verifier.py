"""Verifier (Critic) — детерминированный контроль качества находок.

1. Заземление: каждая цитата ищется в ТЗ; находки «о тексте» без подтверждённой
   цитаты отбрасываются как галлюцинации.
2. Перепроверка «отсутствия»: если агент утверждает, что темы нет, а правила
   находят её упоминания — серьёзность понижается, добавляется пометка.
3. Дедупликация между агентами + «консенсус»: если находку подтвердили
   несколько агентов, уверенность растёт.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from agents import rules
from agents.base import Agent, Context
from knowledge.fintech_controls import CONTROLS_BY_ID
from knowledge.tz_structure import SECTIONS
from models import Category, Finding, Severity, VerifierStats

QUOTE_THRESHOLD = 0.75
MIN_CONFIDENCE = 0.4
_SECTION_IDS = {s.id for s in SECTIONS}
_SEC_FAMILY = {Category.security, Category.compliance, Category.fintech}
_TEXT_FAMILY = {Category.ambiguity, Category.testability}


def _norm_title(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"^(опасное требование|не описано|не полностью описано|отсутствует раздел|размытая формулировка)\s*:?", "", s)
    return re.sub(r"[^\w\s]", " ", s).strip()


def _sim(a: Finding, b: Finding) -> float:
    return SequenceMatcher(None, _norm_title(a.title), _norm_title(b.title)).ratio()


def _close(a: Category, b: Category) -> bool:
    return a == b or ({a, b} <= _SEC_FAMILY) or ({a, b} <= _TEXT_FAMILY)


def _claims_about_text(f: Finding) -> bool:
    return (f.category in (Category.ambiguity, Category.contradiction)
            or (f.control_id or "").startswith("AP-") or f.title.lower().startswith("опасное требование"))


def _merge(primary: Finding, other: Finding) -> None:
    if other.severity.rank < primary.severity.rank:
        primary.severity = other.severity
    for q in other.evidence:
        if q not in primary.evidence:
            primary.evidence.append(q)
    primary.lines = sorted(set(primary.lines) | set(other.lines))
    for src in other.sources or [other.agent]:
        if src not in primary.sources:
            primary.sources.append(src)
    if not primary.description:
        primary.description = other.description
    for attr in ("regulation", "suggested_text", "question", "control_id"):
        if not getattr(primary, attr) and getattr(other, attr):
            setattr(primary, attr, getattr(other, attr))
    if not primary.recommendation:
        primary.recommendation = other.recommendation
    primary.confidence = max(primary.confidence, other.confidence)
    if primary.verified is None:
        primary.verified = other.verified


class VerifierAgent(Agent):
    name = "verifier"
    title = "Verifier (Critic)"
    role = "Проверяет цитаты по тексту ТЗ, отсекает галлюцинации, перепроверяет «отсутствие», убирает дубли"
    stage = 3
    uses_llm = False

    def run_rules(self, ctx: Context) -> tuple[int, str]:
        doc = ctx.doc
        # фиксированный порядок агентов: результат не зависит от того, кто из параллельных агентов закончил первым
        all_f = [f for name in sorted(ctx.findings_by_agent) for f in ctx.findings_by_agent[name]]
        st = VerifierStats(input_findings=len(all_f))
        section_present = {c.section.id: c for c in rules.scan_sections(doc)}
        kept: list[Finding] = []

        for f in all_f:
            f.sources = f.sources or [f.agent]
            n_lines = len(doc.lines)
            # 1) заземление цитат
            if f.evidence:
                good_q, good_l = [], []
                for q in f.evidence:
                    ln, score = doc.locate_quote(q)
                    if ln and score >= QUOTE_THRESHOLD:
                        good_q.append(q)
                        good_l.append(ln)
                if good_q:
                    f.evidence, f.verified = good_q, True
                    f.lines = sorted(set(good_l) | {l for l in f.lines if 1 <= l <= n_lines})
                elif _claims_about_text(f):
                    st.rejected_hallucinations += 1
                    continue
                else:
                    f.evidence, f.verified = [], None
                    f.lines = [l for l in f.lines if 1 <= l <= n_lines]
                    f.verification_note = "цитата агента не найдена в ТЗ и удалена"
            else:
                f.lines = [l for l in f.lines if 1 <= l <= n_lines]
                if _claims_about_text(f) and not f.lines:
                    st.rejected_hallucinations += 1
                    continue
                if f.lines and _claims_about_text(f):
                    f.verified = True

            # 2) перепроверка утверждений об отсутствии
            if not f.evidence and f.control_id and f.title.lower().startswith(("отсутствует", "не описан", "нет ")):
                hit_lines: list[int] = []
                if f.control_id in _SECTION_IDS and section_present[f.control_id].present:
                    hit_lines = section_present[f.control_id].lines
                elif f.control_id in CONTROLS_BY_ID:
                    chk = rules.check_control(doc, CONTROLS_BY_ID[f.control_id])
                    if chk.status == "covered":
                        hit_lines = chk.lines
                if hit_lines:
                    f.severity = Severity(rules.DOWNGRADE[f.severity.value])
                    f.confidence = min(f.confidence, 0.55)
                    f.verification_note = (f"в ТЗ есть упоминания темы (L{', L'.join(map(str, hit_lines[:4]))}) — "
                                           f"серьёзность понижена; проверьте, достаточно ли они конкретны")
                    st.downgraded += 1

            if f.confidence < MIN_CONFIDENCE:
                st.rejected_hallucinations += 1
                continue
            kept.append(f)

        # 3) дедупликация + консенсус
        kept.sort(key=lambda x: (x.severity.rank, -x.confidence, x.agent, x.title))
        groups: list[Finding] = []
        for f in kept:
            target = None
            for g in groups:
                sim = _sim(f, g)
                same_lines = bool(f.lines) and set(f.lines) == set(g.lines)
                close = _close(f.category, g.category)
                # одни и те же строки ТЗ: один дефект, описанный разными агентами разными словами
                if same_lines and (close or len(f.lines) > 1 or sim >= 0.35):
                    target = g
                    break
                if not close:
                    continue
                same_ctrl = bool(f.control_id) and f.control_id == g.control_id
                overlap = bool(set(f.lines) & set(g.lines))
                if same_ctrl or (overlap and (f.category in _TEXT_FAMILY or sim >= 0.3)) or sim >= 0.78:
                    target = g
                    break
            if target is None:
                groups.append(f)
            else:
                _merge(target, f)
                st.merged_duplicates += 1
        for g in groups:
            if len(g.sources) > 1:
                g.confidence = min(1.0, g.confidence + 0.1)

        groups.sort(key=lambda x: (x.severity.rank, _CAT_ORDER.get(x.category, 9), x.lines[0] if x.lines else 10 ** 6, x.title))
        for i, g in enumerate(groups, start=1):
            g.id = f"F-{i:02d}"
        ctx.findings = groups
        st.output_findings = len(groups)
        ctx.verifier = st
        return len(groups), (f"вход {st.input_findings} → выход {st.output_findings}: дублей {st.merged_duplicates}, "
                             f"отсечено {st.rejected_hallucinations}, понижено {st.downgraded}")


_CAT_ORDER = {Category.compliance: 0, Category.security: 1, Category.fintech: 2, Category.contradiction: 3,
              Category.completeness: 4, Category.testability: 5, Category.ambiguity: 6}
