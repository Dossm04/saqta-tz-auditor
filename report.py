"""Экспорт отчёта: Markdown (для людей) и JSON (для интеграций)."""
from __future__ import annotations

from knowledge import DISCLAIMER
from knowledge.domains import DOMAIN_LABELS
from models import Report

_STATUS = {"covered": "✅ покрыто", "partial": "🟨 частично", "missing": "❌ нет", "not_applicable": "— н/п"}


def to_markdown(r: Report) -> str:
    sc = r.severity_counts()
    cov, part, total = r.control_coverage()
    out = [
        f"# Отчёт о качестве ТЗ: {r.document.title}",
        "",
        f"**Файл:** {r.document_name} · **Режим:** {r.mode} ({r.model}) · **Время:** {r.total_duration_s} с "
        f"(ручное ревью ≈ {r.manual_review_hours} ч)",
        "",
        f"## Итог: {r.overall_score}/100 — {r.verdict}",
        "",
        r.summary,
        "",
        f"🔴 Критично: **{sc['critical']}** · 🟠 Важно: **{sc['major']}** · 🟡 Мелочь: **{sc['minor']}** · "
        f"Финтех/ИБ-контроли: **{cov}** покрыто, **{part}** частично из **{total}** применимых",
        "",
        "| Измерение | Балл | Вес | Комментарий |",
        "|---|---|---|---|",
    ]
    out += [f"| {d.name} | {d.score} | {int(d.weight * 100)}% | {d.comment} |" for d in r.dimensions]
    if r.strengths:
        out += ["", "**Сильные стороны:**", *[f"- {s}" for s in r.strengths]]
    out += ["", f"**Области ТЗ:** {', '.join(DOMAIN_LABELS.get(d, d) for d in r.document.domains) or '—'}", ""]

    out += ["## Уточняющие вопросы к автору ТЗ", ""]
    for q in r.questions:
        out.append(f"### {q.id} · {q.priority.label} · {q.addressee}{' · ⛔ блокер' if q.blocking else ''}")
        out.append(f"**{q.question}**")
        if q.context:
            out.append(f"\n> {q.context}")
        if q.why:
            out.append(f"\n_Зачем:_ {q.why}")
        if q.answer_options:
            out.append("\n_Варианты ответа:_")
            out += [f"- [ ] {o}" for o in q.answer_options]
        if q.related_findings:
            out.append(f"\n_Связанные находки:_ {', '.join(q.related_findings)}")
        out.append("")

    out += ["## Находки", ""]
    for f in r.findings:
        loc = f" · L{', L'.join(map(str, f.lines[:5]))}" if f.lines else ""
        out.append(f"### {f.id} {f.severity.icon} {f.title}")
        out.append(f"_{f.severity.label} · {f.category.label}{loc} · агенты: {', '.join(f.sources)} · уверенность {f.confidence:.0%}_")
        out.append("")
        if f.description:
            out.append(f.description)
        for e in f.evidence[:3]:
            out.append(f"> {e}")
        if f.regulation:
            out.append(f"\n**Норма / стандарт:** {f.regulation}")
        if f.recommendation:
            out.append(f"\n**Рекомендация:** {f.recommendation}")
        if f.suggested_text:
            out.append(f"\n**Предлагаемая формулировка:** «{f.suggested_text}»")
        if f.verification_note:
            out.append(f"\n_Verifier: {f.verification_note}_")
        out.append("")

    out += ["## Матрица финтех/ИБ-контролей", "", "| ID | Область | Контроль | Статус | Строки |", "|---|---|---|---|---|"]
    for c in r.controls:
        if c.status == "not_applicable":
            continue
        out.append(f"| {c.control_id} | {c.area} | {c.title} | {_STATUS.get(c.status, c.status)} | "
                   f"{', '.join('L' + str(n) for n in c.lines[:4])} |")

    if r.template_check:
        ok = sum(t["present"] for t in r.template_check)
        out += ["", f"## Соответствие шаблону ТЗ ({ok}/{len(r.template_check)})", "",
                "| Раздел шаблона | В ТЗ | Найденный заголовок |", "|---|---|---|"]
        out += [f"| {t['title']} | {'✅' if t['present'] else '❌'} | {t.get('matched') or ''} |" for t in r.template_check]

    v = r.verifier
    out += ["", "## Трассировка агентов", "", "| Стадия | Агент | Режим | Время, с | Результат | Примечание |", "|---|---|---|---|---|---|"]
    out += [f"| {t.stage} | {t.title} | {t.mode} | {t.duration_s} | {t.produced} | {t.note} |" for t in r.trace]
    out += ["", f"Verifier: на входе {v.input_findings} находок → {v.output_findings} после проверки "
                f"(объединено дублей: {v.merged_duplicates}, отсечено неподтверждённых: {v.rejected_hallucinations}, "
                f"понижено: {v.downgraded}).", "", f"_{DISCLAIMER}_", ""]
    return "\n".join(out)


def to_json(r: Report) -> str:
    return r.model_dump_json(indent=2)
