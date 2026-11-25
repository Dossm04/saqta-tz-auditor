"""Отчёт о проверке требований (Markdown) — состав по п. 6 ТЗ хакатона:
паспорт прогона → нарезка требований → проверка каждого требования → проверка набора → сводка → итог.
"""
from __future__ import annotations

from agents.iso_review import IsoReport, UnitReview
from agents.segmenter import KIND_LABEL
from knowledge.iso29148 import BY_CODE, INDIVIDUAL, SECURITY, SET, SEVERITY_ICON, SEVERITY_LABEL, STATUS_ICON

_STATUS_TEXT = {"ok": "соответствует", "issue": "нарушено", "warn": "замечание", "na": "не применимо"}


def _esc(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def _where(u: UnitReview) -> str:
    lines = sorted(set(u.lines))
    loc = f"стр. {', '.join(map(str, u.pages))}" if u.pages else ""
    ln = f"L{lines[0]}" + (f"–L{lines[-1]}" if len(lines) > 1 else "") if lines else ""
    return ", ".join(x for x in (loc, ln) if x)


def _issue_md(i, idx: int) -> list[str]:
    c = BY_CODE.get(i.characteristic)
    name = c.name if c else i.characteristic
    out = [f"{idx}. {SEVERITY_ICON.get(i.severity, '')} **{SEVERITY_LABEL.get(i.severity, i.severity)} · {name}** — {i.problem}"]
    for e in i.evidence[:3]:
        out.append(f"   > «{_esc(e)}»" + (f" (L{i.lines[i.evidence.index(e)]})" if i.lines and i.evidence.index(e) < len(i.lines) else ""))
    if i.related:
        out.append(f"   Связано: {', '.join(i.related)}")
    if i.fix:
        out.append(f"   **Исправить:** {i.fix}")
    out.append(f"   _Источник: {'LLM-критик' if i.source == 'llm' else 'правила' if i.source == 'rules' else 'правила и LLM-критик'}_")
    return out


def to_markdown(r: IsoReport) -> str:
    s = r.stats
    units = r.units
    out: list[str] = [f"# Отчёт о проверке требований: {r.document_title}", ""]

    # 1. Паспорт
    out += ["## 1. Паспорт прогона", "", "| Поле | Значение |", "|---|---|",
            f"| Проверяемый документ | {_esc(r.document_title)} (файл `{_esc(r.document_name)}`) |",
            f"| Дата и время прогона | {r.run_at} |",
            f"| Модель | {_esc(r.model)} |",
            f"| Режим | {_esc(r.mode)} |",
            f"| Агент | Saqta TZ Auditor, {_esc(r.agent_version)} |",
            "| Критерии | ISO/IEC/IEEE 29148:2018, п. 5.2.5 (отдельное требование) и 5.2.6 (набор); IEEE Std 830-1998, п. 4.3 |",
            f"| Длительность | {r.duration_s} с |", ""]

    # 2. Нарезка
    out += ["## 2. Как агент нарезал требования", "",
            "Единица проверки — пункт с идентификатором, абзац раздела требований, строка таблицы требований "
            "(критерии приёмки, параметры API) или описание кода ответа. Если в документе номера не было, агент "
            "присвоил производный идентификатор (раздел + номер абзаца/пункта/строки) и сохранил цитату.", "",
            "| ID в отчёте | ID в документе | Раздел | Вид | Где | Краткая формулировка (цитата) |", "|---|---|---|---|---|---|"]
    for u in units:
        short = u.text if len(u.text) <= 140 else u.text[:137].rsplit(" ", 1)[0] + "…"
        out.append(f"| {u.rid} | {_esc(u.doc_ref)} | {_esc(u.section)} | {KIND_LABEL.get(u.kind, u.kind)} | {_where(u)} | {_esc(short)} |")
    out.append("")

    # 3. Проверка каждого требования
    codes = [c.code for c in INDIVIDUAL] + [SECURITY.code]
    out += ["## 3. Проверка каждого требования", "",
            "Матрица: ✅ соответствует · ❌ нарушено · ⚠️ замечание · — не применимо. Колонки — характеристики ISO 29148 п. 5.2.5 "
            "и IEEE 830 п. 4.3; последняя колонка — риски ИБ и регуляторики (дополнительная проверка).", "",
            "| ID | " + " | ".join(BY_CODE[c].name for c in codes) + " |", "|---|" + "---|" * len(codes)]
    for u in units:
        out.append(f"| {u.rid} | " + " | ".join(STATUS_ICON.get(u.statuses.get(c, "na"), "—") for c in codes) + " |")
    out.append("")
    for u in units:
        n = len(u.issues)
        head = f"### {u.rid} · {u.doc_ref}"
        out += [head, "", f"> {_esc(u.quote)[:600]}", "", f"_{u.section} · {KIND_LABEL.get(u.kind, u.kind)} · {_where(u)}_", ""]
        bad = [c for c in codes if u.statuses.get(c) in ("issue", "warn")]
        if not n:
            out += ["Нарушений не выявлено.", ""]
            continue
        out.append(f"**Нарушения ({n}):** " + ", ".join(BY_CODE[c].name for c in bad))
        out.append("")
        for k, i in enumerate(u.issues, start=1):
            out += _issue_md(i, k)
        if u.rewrite:
            out += ["", f"**Предлагаемая формулировка:** {u.rewrite}"]
        out.append("")

    # 4. Проверка набора
    sr = r.set_review
    out += ["## 4. Проверка всего набора требований", "", "| Характеристика | Пункт стандарта | Оценка | Комментарий |", "|---|---|---|---|"]
    for c in SET:
        st = sr.statuses.get(c.code, "na")
        comment = sr.comments.get(c.code, "")
        if not comment:
            rel = [i for i in sr.issues if i.characteristic == c.code]
            comment = "; ".join(f"{i.problem[:140]} ({', '.join(i.related)})" if i.related else i.problem[:160] for i in rel[:2])
        out.append(f"| {c.name} | {c.source} | {STATUS_ICON.get(st, '—')} {_STATUS_TEXT.get(st, st)} | {_esc(comment)} |")
    out.append("")
    contra = [i for i in sr.issues if i.characteristic == "set_consistent"]
    other = [i for i in sr.issues if i.characteristic != "set_consistent"]
    if contra:
        out += [f"### 4.1. Противоречия ({len(contra)})", ""]
        for k, i in enumerate(contra, start=1):
            out += _issue_md(i, k)
        out.append("")
    if other:
        out += [f"### 4.2. Прочие нарушения уровня набора ({len(other)})", ""]
        for k, i in enumerate(other, start=1):
            out += _issue_md(i, k)
        out.append("")
    if sr.missing:
        out += [f"### 4.3. Пропущенные требования ({len(sr.missing)})", "", "| Чего не хватает | Почему важно | Предлагаемое требование | Связано |", "|---|---|---|---|"]
        for m in sr.missing:
            out.append(f"| {_esc(m.get('topic', ''))} | {_esc(m.get('why', ''))} | {_esc(m.get('suggested_requirement', ''))} | {', '.join(m.get('related') or [])} |")
        out.append("")
    if sr.checklist:
        out += ["### 4.4. Чек-лист ревью из документа, заполненный агентом", "", "| Пункт чек-листа | Ответ | Обоснование |", "|---|---|---|"]
        for c in sr.checklist:
            out.append(f"| {_esc(c.get('item', ''))} | {_esc(str(c.get('answer', '')))} | {_esc(c.get('comment', ''))} |")
        out.append("")

    # 5. Сводка
    out += ["## 5. Сводка", "",
            f"- Требований проверено: **{s['requirements']}**; с существенными нарушениями (критичными или важными): **{s['with_issues']}**.",
            f"- Нарушений всего: **{s['issues_total']}** — 🔴 критичных {s['critical']}, 🟠 важных {s['major']}, 🟡 мелких {s['minor']} "
            f"(по отдельным требованиям {s['unit_issues']}, по набору {s['set_issues']}).",
            f"- Отклонено неподтверждённых претензий LLM (цитата не найдена в документе): {s.get('rejected', 0)}.", "",
            "| Характеристика | Нарушений |", "|---|---|"]
    for code, cnt in sorted(s.get("by_characteristic", {}).items(), key=lambda kv: -kv[1]):
        out.append(f"| {BY_CODE[code].name if code in BY_CODE else code} | {cnt} |")
    out += ["", "**Критичные дефекты:**", ""]
    for cls in ("непроверяемые", "противоречивые", "неполные"):
        items = r.critical.get(cls, [])
        out.append(f"- **{cls.capitalize()} ({len(items)}):**")
        for x in items[:12]:
            out.append(f"  - {_esc(x)[:300]}")
        if len(items) > 12:
            out.append(f"  - … и ещё {len(items) - 12}")
    out.append("")

    # 6. Итог
    out += ["## 6. Итог", "", f"**Можно ли начинать разработку и тестирование?** {r.verdict}", ""]
    if sr.summary:
        out += [sr.summary, ""]
    out += ["**Что исправить в первую очередь:**", ""]
    out += [f"{k}. {_esc(p)}" for k, p in enumerate(r.priorities, start=1)]
    out.append("")

    # Приложения
    if sr.questions:
        out += ["## Приложение A. Уточняющие вопросы к автору документа", ""]
        for k, q in enumerate(sr.questions, start=1):
            rel = ", ".join(q.get("related") or [])
            out.append(f"{k}. **{_esc(q.get('question', ''))}**" + (f" ({rel})" if rel else "") +
                       (f" — адресат: {_esc(q.get('addressee', ''))}" if q.get("addressee") else ""))
            for o in q.get("options") or []:
                out.append(f"   - [ ] {_esc(str(o))}")
        out.append("")
    out += ["## Приложение B. Как работал агент", "", "| Агент | Результат |", "|---|---|"]
    out += [f"| {t['agent']} | {_esc(t['result'])} |" for t in r.trace]
    if s.get("notes"):
        out += ["", "Замечания прогона: " + "; ".join(_esc(n) for n in s["notes"])]
    out += ["", "_Каждое нарушение привязано к месту в документе (ID требования, строка `L…`, страница PDF) и подтверждено "
            "дословной цитатой; претензии LLM без цитаты, найденной в документе, отбрасываются автоматически._", ""]
    return "\n".join(out)
