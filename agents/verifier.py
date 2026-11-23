"""Verifier (Critic) — детерминированный контроль качества находок.

1. Заземление: каждая цитата ищется в ТЗ; находки «о тексте» без подтверждённой
   цитаты отбрасываются как галлюцинации.
2. Перепроверка «отсутствия»: если агент утверждает, что темы нет, а правила
   находят её упоминания — серьёзность понижается, добавляется пометка.
3. Дедупликация между агентами + «консенсус»: если находку подтвердили
   несколько агентов, уверенность растёт.

Каждое решение (отсечение, склейка, понижение) записывается с человекочитаемой
причиной в VerifierStats — это видно в отчёте. Отклонённые находки складываются
в ctx.rejected: в LLM-режиме оркестратор возвращает их агенту-автору (critic loop).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from agents import rules
from agents.base import Agent, Context
from knowledge.fintech_controls import CONTROLS_BY_ID
from knowledge.tz_structure import SECTIONS
from models import Category, DowngradeRecord, Finding, MergeRecord, RejectedRecord, Severity, VerifierStats

QUOTE_THRESHOLD = 0.75
MIN_CONFIDENCE = 0.4
_SECTION_IDS = {s.id for s in SECTIONS}
_SEC_FAMILY = {Category.security, Category.compliance, Category.fintech}
_TEXT_FAMILY = {Category.ambiguity, Category.testability}
# служебные префиксы заголовков, которые генерируют правила: без них сходство считается по сути темы
_TITLE_PREFIX = re.compile(
    r"^(нет раздела из шаблона|опасное требование|не полностью описано|не описано|не описан|отсутствует раздел|"
    r"размытая формулировка|противоречивые значения|открытый вопрос в тексте тз)\s*:?\s*")


def _norm_title(s: str) -> str:
    s = s.lower().replace("ё", "е").strip()
    s = _TITLE_PREFIX.sub("", s)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def _sim(a: Finding, b: Finding) -> float:
    return SequenceMatcher(None, _norm_title(a.title), _norm_title(b.title)).ratio()


def _close(a: Category, b: Category) -> bool:
    return a == b or ({a, b} <= _SEC_FAMILY) or ({a, b} <= _TEXT_FAMILY)


def _claims_about_text(f: Finding) -> bool:
    return (f.category in (Category.ambiguity, Category.contradiction)
            or (f.control_id or "").startswith("AP-") or f.title.lower().startswith("опасное требование"))


def merge_reason(f: Finding, g: Finding) -> str | None:
    """Причина склеить находку f с группой g или None. Разные известные контроли не склеиваются,
    если заголовки явно не об одном (это разные пробелы, даже если цитируют одну строку)."""
    sim = _sim(f, g)
    diff_ctrl = bool(f.control_id and g.control_id) and f.control_id != g.control_id
    if diff_ctrl and sim < 0.6:
        return None
    same_lines = bool(f.lines) and set(f.lines) == set(g.lines)
    close = _close(f.category, g.category)
    # одни и те же строки ТЗ: один дефект, описанный разными агентами разными словами
    # (типичный случай — LLM-находка без control_id и находка правил с ним)
    if same_lines and (close or len(f.lines) > 1 or sim >= 0.35):
        return f"те же строки ТЗ (L{', L'.join(map(str, sorted(f.lines)[:4]))})"
    if not close:
        return None
    if f.control_id and f.control_id == g.control_id:
        return f"тот же контроль {f.control_id}"
    overlap = set(f.lines) & set(g.lines)
    if overlap and (f.category in _TEXT_FAMILY or sim >= 0.3):
        return f"пересекаются строки L{', L'.join(map(str, sorted(overlap)[:3]))}, сходство заголовков {sim:.2f}"
    if sim >= 0.78 and not diff_ctrl:
        return f"сходство заголовков {sim:.2f}"
    return None


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
    if other.verification_note and not primary.verification_note:
        primary.verification_note = other.verification_note
    primary.revision = max(primary.revision, other.revision)


class VerifierAgent(Agent):
    name = "verifier"
    title = "Verifier (Critic)"
    role = "Проверяет цитаты по тексту ТЗ, отсекает галлюцинации, перепроверяет «отсутствие», убирает дубли"
    stage = 3
    uses_llm = False

    def run_rules(self, ctx: Context) -> tuple[int, str]:
        doc = ctx.doc
        n_lines = len(doc.lines)
        # фиксированный порядок агентов: результат не зависит от того, кто из параллельных агентов закончил первым.
        # Verifier работает с копиями: исходные находки агентов не мутируют (их можно перепроверить повторно).
        originals = [f for name in sorted(ctx.findings_by_agent) for f in ctx.findings_by_agent[name]]
        st = VerifierStats(input_findings=len(originals))
        section_present = {c.section.id: c for c in rules.scan_sections(doc)}
        kept: list[Finding] = []
        rejected: list[tuple[Finding, RejectedRecord]] = []
        downgrade_of: dict[int, DowngradeRecord] = {}

        def reject(orig: Finding, f: Finding, reason: str, best_line: int | None = None, best: float = 0.0) -> None:
            rec = RejectedRecord(title=f.title, agent=f.agent, reason=reason, best_line=best_line,
                                 best_score=round(best, 2), revision=f.revision)
            st.rejected.append(rec)
            rejected.append((orig, rec))

        for orig in originals:
            f = orig.model_copy(deep=True)
            # поля, которыми владеет только Verifier: агент (или LLM) не может сам «подтвердить» находку
            f.sources = [f.agent] if f.agent else []
            f.verified = None
            f.verification_note = None
            # 1) заземление цитат
            if f.evidence:
                good_q, good_l = [], []
                best_line, best_score = None, 0.0
                for q in f.evidence:
                    ln, score = doc.locate_quote(q)
                    if score > best_score:
                        best_line, best_score = ln, score
                    if ln and score >= QUOTE_THRESHOLD:
                        good_q.append(q)
                        good_l.append(ln)
                if good_q:
                    f.evidence, f.verified = good_q, True
                    f.lines = sorted(set(good_l) | {l for l in f.lines if 1 <= l <= n_lines})
                elif _claims_about_text(f):
                    near = f" (ближайшая строка L{best_line}, совпадение {best_score:.0%})" if best_line else ""
                    reject(orig, f, f"цитата не найдена в ТЗ{near}", best_line, best_score)
                    continue
                else:
                    f.evidence, f.verified = [], None
                    f.lines = [l for l in f.lines if 1 <= l <= n_lines]
                    f.verification_note = "цитата агента не найдена в ТЗ и удалена"
            else:
                bad = [l for l in f.lines if not 1 <= l <= n_lines]
                f.lines = [l for l in f.lines if 1 <= l <= n_lines]
                if _claims_about_text(f) and not f.lines:
                    reason = (f"строки вне документа (L{', L'.join(map(str, bad[:3]))}, всего строк {n_lines})"
                              if bad else "утверждение о тексте ТЗ без цитаты и номера строки")
                    reject(orig, f, reason)
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
                    old = f.severity
                    f.severity = Severity(rules.DOWNGRADE[f.severity.value])
                    f.confidence = min(f.confidence, 0.55)
                    f.verification_note = (f"в ТЗ есть упоминания темы (L{', L'.join(map(str, hit_lines[:4]))}) — "
                                           f"серьёзность понижена; проверьте, достаточно ли они конкретны")
                    st.downgraded += 1
                    st.downgrades.append(DowngradeRecord(
                        title=f.title, agent=f.agent, from_severity=old.value, to_severity=f.severity.value,
                        reason="агент утверждает, что темы нет, но правила нашли её упоминания",
                        hit_lines=hit_lines[:6]))
                    downgrade_of[id(f)] = st.downgrades[-1]

            if f.confidence < MIN_CONFIDENCE:
                reject(orig, f, f"низкая уверенность агента ({f.confidence:.2f} < {MIN_CONFIDENCE})")
                continue
            if f.revision and not f.verification_note:
                f.verification_note = "находка исправлена агентом по замечанию Verifier и подтверждена повторно"
            kept.append(f)

        # 3) дедупликация + консенсус
        kept.sort(key=lambda x: (x.severity.rank, -x.confidence, x.agent, x.title))
        groups: list[Finding] = []
        merge_log: list[tuple[Finding, MergeRecord]] = []
        owner: dict[int, Finding] = {}
        for f in kept:
            target, reason = None, None
            for g in groups:
                reason = merge_reason(f, g)
                if reason:
                    target = g
                    break
            if target is None:
                groups.append(f)
            else:
                _merge(target, f)
                owner[id(f)] = target
                st.merged_duplicates += 1
                rec = MergeRecord(kept_title=target.title, merged_title=f.title, from_agent=f.agent, reason=reason or "")
                st.merges.append(rec)
                merge_log.append((target, rec))
        for g in groups:
            if len(g.sources) > 1:
                g.confidence = min(1.0, g.confidence + 0.1)

        groups.sort(key=lambda x: (x.severity.rank, _CAT_ORDER.get(x.category, 9), x.lines[0] if x.lines else 10 ** 6, x.title))
        for i, g in enumerate(groups, start=1):
            g.id = f"F-{i:02d}"
        for g, rec in merge_log:
            rec.kept_id = g.id
        # понижение у находки, поглощённой другой группой, показываем с ID итоговой группы
        for f in kept:
            if id(f) in downgrade_of:
                downgrade_of[id(f)].id = owner.get(id(f), f).id
        ctx.findings = groups
        ctx.rejected = rejected
        st.rejected_hallucinations = len(st.rejected)
        st.output_findings = len(groups)
        ctx.verifier = st
        return len(groups), (f"вход {st.input_findings} → выход {st.output_findings}: дублей {st.merged_duplicates}, "
                             f"отсечено {st.rejected_hallucinations}, понижено {st.downgraded}")


_CAT_ORDER = {Category.compliance: 0, Category.security: 1, Category.fintech: 2, Category.contradiction: 3,
              Category.completeness: 4, Category.testability: 5, Category.ambiguity: 6}
