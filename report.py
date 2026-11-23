"""Экспорт отчёта: Markdown (для людей) и JSON (для интеграций)."""
from __future__ import annotations

from knowledge import DISCLAIMER
from knowledge.domains import DOMAIN_LABELS
from models import Question, Report

_STATUS = {"covered": "✅ покрыто", "partial": "🟨 частично", "missing": "❌ нет", "not_applicable": "— н/п"}


def _cell(s: object) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def question_quality_line(r: Report) -> str:
    qq = r.question_quality
    if not qq.total:
        return ""
    return (f"Качество вопросов: чеклист прошли **{qq.lint_passed}/{qq.total}** · с вариантами ответа "
            f"**{qq.with_options:.0%}** (закрытых среди P0/P1 — {qq.closed_p0p1:.0%}) · с привязкой к ТЗ "
            f"**{qq.grounded:.0%}** · с допущением по умолчанию {qq.with_default:.0%} · критичные находки покрыты "
            f"**{qq.critical_covered}**, обязательные (критичные, противоречия, опасные, TBD) — **{qq.must_cover_covered}**"
            + (f" · исправлено LLM по чеклисту: {qq.repaired}" if qq.repaired else "")
            + f" · встреча ≈ {qq.agenda_minutes} мин")


def _question_md(q: Question) -> list[str]:
    out = [f"### {q.id} · {q.priority.label} · {q.addressee}{' · ⛔ блокер' if q.blocking else ''}",
           f"**{q.question}**"]
    if q.context:
        out.append(f"\n> {q.context}")
    if q.why:
        out.append(f"\n_Зачем:_ {q.why}")
    if q.impact:
        out.append(f"\n_Если не ответить:_ {q.impact}")
    if q.answer_options:
        out.append("\n_Варианты ответа:_")
        out += [f"- [ ] {o}{' ★ рекомендуем' if i == q.recommended_option else ''}" for i, o in enumerate(q.answer_options)]
    if q.default_assumption:
        out.append(f"\n_Если ответа нет до старта спринта — закладываем:_ {q.default_assumption}")
    if q.related_findings:
        out.append(f"\n_Связанные находки:_ {', '.join(q.related_findings)}")
    if q.lint:
        out.append(f"\n_Линтер вопроса:_ {'; '.join(q.lint)}")
    out.append("")
    return out


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

    out += ["## Уточняющие вопросы к автору ТЗ (повестка встречи)", ""]
    ql = question_quality_line(r)
    if ql:
        out += [ql, ""]
    for q in r.questions:
        out += _question_md(q)
    if r.extra_questions:
        out += ["### Дополнительные вопросы (сверх лимита одной встречи)", ""]
        out += [f"- **{q.id}** [{q.priority.value}] ({q.addressee}) {q.question}"
                + (f" — варианты: {' / '.join(q.answer_options)}" if q.answer_options else "")
                + (f" · находки: {', '.join(q.related_findings)}" if q.related_findings else "")
                for q in r.extra_questions]
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
    out += [f"| {t.stage} | {t.title} | {t.mode} | {t.duration_s} | {t.produced} | {_cell(t.note)} |" for t in r.trace]
    out += ["", f"Verifier: на входе {v.input_findings} находок → {v.output_findings} после проверки "
                f"(объединено дублей: {v.merged_duplicates}, отсечено неподтверждённых: {v.rejected_hallucinations}, "
                f"понижено: {v.downgraded})."]
    if v.revision_returned:
        out.append(f"Обратная связь Critic → аналитики: возвращено {v.revision_returned}, восстановлено после "
                   f"исправления {v.revision_restored}, снято агентами {v.revision_withdrawn}.")
    out += verifier_details_md(r)
    if r.messages:
        out += ["", "### Сообщения между агентами", "", "| t, с | От | Кому | Тип | Кол-во | Содержание |", "|---|---|---|---|---|---|"]
        out += [f"| {m.ts} | {m.sender} | {m.receiver} | {m.kind} | {m.count} | {_cell(m.summary)} |" for m in r.messages]
    out += ["", f"_{DISCLAIMER}_", ""]
    return "\n".join(out)


def verifier_details_md(r: Report) -> list[str]:
    """Прозрачность Verifier: что отсечено, что склеено, что понижено — и почему."""
    v, out = r.verifier, []
    if v.rejected:
        out += ["", "### Verifier: отсечённые находки", "", "| Находка | Агент | Причина | Итог |", "|---|---|---|---|"]
        out += [f"| {_cell(x.title)} | {x.agent} | {_cell(x.reason)} | {_cell(x.outcome)} |" for x in v.rejected]
    if v.restored:
        out += ["", "### Verifier: восстановлены после исправления агентом", "", "| Находка | Агент | Исходная причина |", "|---|---|---|"]
        out += [f"| {_cell(x.title)} | {x.agent} | {_cell(x.reason)} |" for x in v.restored]
    if v.merges:
        out += ["", "### Verifier: склеенные дубли", "", "| Итоговая находка | Поглощена | От агента | Причина |", "|---|---|---|---|"]
        out += [f"| {x.kept_id} {_cell(x.kept_title)} | {_cell(x.merged_title)} | {x.from_agent} | {_cell(x.reason)} |" for x in v.merges]
    if v.downgrades:
        out += ["", "### Verifier: пониженная серьёзность", "", "| Находка | Агент | Было → стало | Причина | Строки |", "|---|---|---|---|---|"]
        out += [f"| {x.id} {_cell(x.title)} | {x.agent} | {x.from_severity} → {x.to_severity} | {_cell(x.reason)} | "
                f"{', '.join('L' + str(n) for n in x.hit_lines[:4])} |" for x in v.downgrades]
    return out


def to_json(r: Report) -> str:
    return r.model_dump_json(indent=2)
