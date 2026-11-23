"""Экспорт отчёта в Word (.docx): в банках ТЗ и замечания к нему живут в Word."""
from __future__ import annotations

import io

from docx import Document as DocxDocument
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

from knowledge import DISCLAIMER
from knowledge.domains import DOMAIN_LABELS
from models import Report

_SEV_COLOR = {"critical": "DC2626", "major": "D97706", "minor": "CA8A04"}
_STATUS = {"covered": "Покрыто", "partial": "Частично", "missing": "Нет", "not_applicable": "Н/п"}
_STATUS_COLOR = {"covered": "DCFCE7", "partial": "FEF3C7", "missing": "FEE2E2"}


def _shade(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def _table(doc, header: list[str], rows: list[list[str]], widths: list[float] | None = None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = h
        c.paragraphs[0].runs[0].bold = True
        _shade(c, "E0E7FF")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    return t


def to_docx(r: Report) -> bytes:
    doc = DocxDocument()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)

    doc.add_heading(f"Отчёт о качестве ТЗ: {r.document.title}", level=0)
    sc = r.severity_counts()
    cov, part, total = r.control_coverage()
    p = doc.add_paragraph()
    run = p.add_run(f"{r.overall_score}/100 — {r.verdict}")
    run.bold = True
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor.from_string("16A34A" if r.overall_score >= 80 else "D97706" if r.overall_score >= 55 else "DC2626")
    doc.add_paragraph(
        f"Файл: {r.document_name} · режим: {r.mode} ({r.model}) · анализ: {r.total_duration_s} с "
        f"(ручное ревью ≈ {r.manual_review_hours} ч) · критично: {sc['critical']}, важно: {sc['major']}, "
        f"мелочь: {sc['minor']} · контроли ИБ/регулятора: {cov} покрыто, {part} частично из {total}")
    doc.add_paragraph("Области ТЗ: " + (", ".join(DOMAIN_LABELS.get(d, d) for d in r.document.domains) or "—"))

    doc.add_heading("Резюме", level=1)
    doc.add_paragraph(r.summary)
    if r.strengths:
        doc.add_paragraph("Сильные стороны:").runs[0].bold = True
        for s in r.strengths:
            doc.add_paragraph(s, style="List Bullet")
    _table(doc, ["Измерение", "Балл", "Вес", "Комментарий"],
           [[d.name, d.score, f"{int(d.weight * 100)}%", d.comment] for d in r.dimensions])

    doc.add_heading("Уточняющие вопросы к автору ТЗ", level=1)
    for q in r.questions:
        h = doc.add_paragraph()
        h.add_run(f"{q.id} · {q.priority.label} · {q.addressee}{' · БЛОКЕР' if q.blocking else ''}").bold = True
        doc.add_paragraph(q.question)
        if q.context:
            cp = doc.add_paragraph(q.context)
            cp.runs[0].italic = True
        if q.why:
            doc.add_paragraph(f"Зачем: {q.why}")
        for o in q.answer_options:
            doc.add_paragraph(f"☐ {o}", style="List Bullet")
        doc.add_paragraph("Ответ: ________________________________________________")

    doc.add_heading("Находки", level=1)
    for f in r.findings:
        h = doc.add_paragraph()
        run = h.add_run(f"{f.id} [{f.severity.label}] {f.title}")
        run.bold = True
        run.font.color.rgb = RGBColor.from_string(_SEV_COLOR[f.severity.value])
        meta = f"{f.category.label}" + (f" · строки L{', L'.join(map(str, f.lines[:5]))}" if f.lines else "") \
               + f" · агенты: {', '.join(f.sources)} · уверенность {f.confidence:.0%}"
        mp = doc.add_paragraph(meta)
        mp.runs[0].font.size = Pt(8.5)
        if f.description:
            doc.add_paragraph(f.description)
        for e in f.evidence[:2]:
            ep = doc.add_paragraph(f"«{e}»")
            ep.runs[0].italic = True
        if f.regulation:
            doc.add_paragraph(f"Норма / стандарт: {f.regulation}")
        if f.recommendation:
            doc.add_paragraph(f"Рекомендация: {f.recommendation}")
        if f.suggested_text:
            sp = doc.add_paragraph()
            sp.add_run("Предлагаемая формулировка: ").bold = True
            sp.add_run(f.suggested_text)

    doc.add_heading("Матрица финтех/ИБ-контролей", level=1)
    rows = [[c.control_id, c.area, c.title, _STATUS.get(c.status, c.status), ", ".join(f"L{n}" for n in c.lines[:4])]
            for c in r.controls if c.status != "not_applicable"]
    t = _table(doc, ["ID", "Область", "Контроль", "Статус", "Строки"], rows)
    for i, c in enumerate([c for c in r.controls if c.status != "not_applicable"], start=1):
        if c.status in _STATUS_COLOR:
            _shade(t.rows[i].cells[3], _STATUS_COLOR[c.status])

    if r.template_check:
        doc.add_heading("Соответствие шаблону ТЗ", level=1)
        _table(doc, ["Раздел шаблона", "Есть в ТЗ", "Найденный заголовок"],
               [[t["title"], "да" if t["present"] else "НЕТ", t.get("matched") or ""] for t in r.template_check])

    doc.add_heading("Как работала система", level=1)
    _table(doc, ["Стадия", "Агент", "Режим", "Время, с", "Результат", "Примечание"],
           [[t.stage, t.title, t.mode, t.duration_s, t.produced, t.note[:120]] for t in r.trace])
    v = r.verifier
    doc.add_paragraph(f"Verifier: на входе {v.input_findings} находок → {v.output_findings} после проверки "
                      f"(дублей объединено {v.merged_duplicates}, неподтверждённых отсечено {v.rejected_hallucinations}, "
                      f"понижено {v.downgraded}).")
    dp = doc.add_paragraph(DISCLAIMER)
    dp.runs[0].italic = True
    dp.runs[0].font.size = Pt(8)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
