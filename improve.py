"""Черновик «ТЗ v2»: исправляет опасные требования, дописывает недостающие и
вносит ответы на уточняющие вопросы. Замыкает цикл «проверка → правка → перепроверка».
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from models import Category, Finding, Report
from parsing import Document

# сохраняемый префикс строки: отступ, маркер списка, '#', нумерация (1. / 3.2.1 / 2)) и ID требования (FR-01:)
_LEAD = re.compile(
    r"^(\s*(?:[-*•]\s+)?(?:#+\s*)?"
    r"(?:(?:[A-ZА-ЯЁ]{1,5}[-.]?\d+(?:\.\d+)*|\d+(?:\.\d+)*)[.):]?\s+)?"
    r"(?:[A-ZА-ЯЁ]{1,5}[-.]?\d+(?:\.\d+)*[.):]?\s+)?)")
_ID_CELL = re.compile(r"^\s*(?:[A-ZА-ЯЁ]{1,5}[-.]?\d+(?:\.\d+)*|\d+(?:\.\d+)*)[.):]?\s*$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[«»\"'`*_]", "", s.lower().replace("ё", "е"))).strip()


def _target_cell(cells: list[str], f: Finding) -> int | None:
    """Индекс ячейки строки таблицы, к которой относится находка: содержит цитату, совпадает с шаблоном
    опасного требования или (по умолчанию) самая длинная текстовая ячейка, не ID."""
    text_idx = [i for i, c in enumerate(cells) if c.strip() and not _ID_CELL.match(c)]
    if not text_idx:
        return None
    for q in f.evidence:
        nq = _norm(q)
        for i in text_idx:
            nc = _norm(cells[i])
            if nc and (nc in nq or nq in nc):
                return i
    if (f.control_id or "").startswith("AP-"):
        try:
            from knowledge.antipatterns import ANTIPATTERNS
            ap = next((a for a in ANTIPATTERNS if a.id == f.control_id), None)
        except Exception:   # база знаний недоступна — выбираем ячейку эвристикой
            ap = None
        if ap is not None:
            rx = re.compile(ap.pattern, re.I)
            for i in text_idx:
                if rx.search(cells[i].replace("ё", "е")):
                    return i
    if f.evidence:
        best = max(text_idx, key=lambda i: SequenceMatcher(None, _norm(cells[i]), _norm(f.evidence[0])).ratio())
        return best
    return max(text_idx, key=lambda i: len(cells[i]))


def replace_line(line: str, new_text: str, fid: str, f: Finding | None = None) -> str:
    """Заменяет текст требования, сохраняя оформление: нумерацию, ID, маркер списка;
    в строке таблицы меняется только ячейка с дефектом (ID, приоритет и другие столбцы остаются)."""
    note = f" _(исправлено по {fid})_"
    if line.lstrip().startswith("|") and line.count("|") >= 2:
        parts = line.split("|")
        inner = parts[1:-1] if line.rstrip().endswith("|") else parts[1:]
        idx = _target_cell(inner, f) if f is not None else None
        if idx is None:
            idx = max(range(len(inner)), key=lambda i: len(inner[i])) if inner else None
        if idx is not None:
            inner[idx] = " " + (new_text + note).replace("|", "\\|") + " "
            tail = [parts[-1]] if line.rstrip().endswith("|") else []
            return "|".join([parts[0], *inner, *tail])
    m = _LEAD.match(line)
    prefix = m.group(1) if m else ""
    return f"{prefix}{new_text}{note}"

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
    replace: dict[int, Finding] = {}
    for f in report.findings:
        if not (f.suggested_text and len(f.lines) == 1 and f.evidence):
            continue
        if (f.control_id or "").startswith("AP-") or f.category in (Category.ambiguity, Category.testability):
            replace.setdefault(f.lines[0], f)

    out: list[str] = []
    for i, line in enumerate(doc.lines, start=1):
        if i in replace:
            f = replace[i]
            out.append(replace_line(line, f.suggested_text or "", f.id, f))
            stats["replaced"] += 1
        else:
            out.append(line)

    # 2) недостающие требования — отдельным разделом, сгруппированным по областям
    groups: dict[Category, list[str]] = {}
    for f in report.findings:
        if f.suggested_text and not f.evidence and f.lines == [] and f.id not in {v.id for v in replace.values()}:
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
