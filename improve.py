"""Черновик «ТЗ v2»: исправляет опасные требования, дописывает недостающие и
вносит ответы на уточняющие вопросы. Замыкает цикл «проверка → правка → перепроверка».
"""
from __future__ import annotations

import re

from models import Category, Report
from parsing import Document

_LEAD = re.compile(r"^(\s*(?:[-*•]\s*)?(?:[A-ZА-Я]{1,5}[-.]?\d+(?:\.\d+)*\.?\s+)?)", re.I)

_AREA_TITLES = {
    Category.compliance: "Регуляторные требования и защита данных",
    Category.security: "Информационная безопасность",
    Category.fintech: "Обработка операций и денег",
    Category.completeness: "Недостающие разделы",
    Category.testability: "Измеримость и критерии приёмки",
    Category.ambiguity: "Уточнённые формулировки",
    Category.contradiction: "Устранённые противоречия",
}


def build_revised_tz(doc: Document, report: Report, answers: dict[str, str] | None = None) -> tuple[str, dict[str, int]]:
    answers = {k: v.strip() for k, v in (answers or {}).items() if v and v.strip()}
    stats = {"replaced": 0, "added": 0, "answers": len(answers)}

    # 1) строки с опасными/размытыми требованиями заменяем предложенной формулировкой
    replace: dict[int, tuple[str, str]] = {}
    for f in report.findings:
        if not (f.suggested_text and len(f.lines) == 1 and f.evidence):
            continue
        if (f.control_id or "").startswith("AP-") or f.category in (Category.ambiguity, Category.testability):
            replace.setdefault(f.lines[0], (f.suggested_text, f.id))

    out: list[str] = []
    for i, line in enumerate(doc.lines, start=1):
        if i in replace:
            new_text, fid = replace[i]
            prefix = _LEAD.match(line).group(1) if _LEAD.match(line) else ""
            out.append(f"{prefix}{new_text} _(исправлено по {fid})_")
            stats["replaced"] += 1
        else:
            out.append(line)

    # 2) недостающие требования — отдельным разделом, сгруппированным по областям
    groups: dict[Category, list[str]] = {}
    for f in report.findings:
        if f.suggested_text and not f.evidence and f.lines == [] and f.id not in {v[1] for v in replace.values()}:
            groups.setdefault(f.category, []).append(f"- {f.suggested_text} _({f.id})_")
    if groups:
        out += ["", "## Дополнительные требования (по результатам проверки)", ""]
        for cat, items in groups.items():
            out += [f"### {_AREA_TITLES.get(cat, cat.label)}", *items, ""]
            stats["added"] += len(items)

    # 3) ответы на уточняющие вопросы
    if answers:
        qmap = {q.id: q for q in report.questions}
        out += ["", "## Уточнения автора ТЗ (ответы на вопросы)", ""]
        for qid, ans in answers.items():
            q = qmap.get(qid)
            if q:
                out += [f"- **{qid}. {q.question}**", f"  Ответ: {ans}"]
    return "\n".join(out), stats
