"""Страница Streamlit: проверка требований по ISO/IEC/IEEE 29148 и IEEE 830 (задание хакатона)."""
from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from agents.iso_review import IsoReport, review_document
from knowledge.iso29148 import BY_CODE, INDIVIDUAL, SECURITY, SET, SEVERITY_ICON, SEVERITY_LABEL, STATUS_ICON
from parsing import Document, load_bytes, load_text
from report_iso import to_markdown

AGENTS = [
    ("segmenter", "Segmenter", "Нарезка документа на требования с ID из документа или производными ID и цитатами"),
    ("rules", "Rule Checker", "Правила по каждой характеристике: размытость, TBD, пустые поля, единичность, API, противоречия"),
    ("domain", "Domain Critic", "Чек-лист области: время, суммы и округление, идемпотентность, возвраты, роли, коды ошибок API, приёмка; Scenario Critic ищет пропущенные сценарии"),
    ("llm", "LLM-критики", "Requirement Critic ‖ Set Critic ‖ Contradiction Hunter ‖ Domain Critic ‖ Scenario Critic — параллельно"),
    ("verifier", "Grounding Verifier", "Каждая цитата ищется в документе; претензии без подтверждения отбрасываются"),
]


def _color(level: str) -> str:
    return {"ready": "#16a34a", "conditional": "#d97706"}.get(level, "#dc2626")


def render(cfg, use_llm: bool, root: Path) -> None:
    ss = st.session_state
    st.title("Агент проверки требований")
    st.caption("Загрузите документ требований: агент проверит его по ISO/IEC/IEEE 29148 и IEEE 830 и составит отчёт.")
    src = st.radio("Документ", ["Загрузить файл", "Вставить текст"], horizontal=True, key="iso_src", label_visibility="collapsed")
    doc: Document | None = None
    try:
        if src == "Загрузить файл":
            up = st.file_uploader("PDF, DOCX, MD или TXT", type=["pdf", "docx", "md", "txt"], key="iso_up")
            if up is not None:
                doc = load_bytes(up.getvalue(), up.name)
        else:
            t = st.text_area("Текст требований", height=200, key="iso_txt")
            if t.strip():
                doc = load_text(t)
    except Exception as e:
        st.error(f"Не удалось прочитать файл: {e}")
    if doc is not None:
        with st.expander(f"Распознанный текст ({doc.non_empty_count} строк)"):
            st.code(doc.text, language="markdown")

    if st.button("🔍 Проверить требования", type="primary", disabled=doc is None):
        cols = st.columns(len(AGENTS))
        holders = {}
        for col, (key, title, role) in zip(cols, AGENTS):
            holders[key] = col.empty()
            holders[key].markdown(f'<div class="card"><div class="t">⏸️ {title}</div><div class="r">{role}</div></div>',
                                  unsafe_allow_html=True)

        def on_event(agent: str, phase: str, note: str = "") -> None:
            h = holders.get(agent)
            title, role = next(((t, r) for k, t, r in AGENTS if k == agent), (agent, ""))
            if not h:
                return
            if phase == "start":
                h.markdown(f'<div class="card run"><div class="t">⏳ {title}</div><div class="r">{role}</div></div>', unsafe_allow_html=True)
            else:
                h.markdown(f'<div class="card ok"><div class="t">✅ {title}</div><div class="r">{role}</div>'
                           f'<div class="s">{html.escape(note)}</div></div>', unsafe_allow_html=True)

        ss.iso_report = review_document(doc, cfg, use_llm=use_llm, on_event=on_event)
        ss.iso_md = to_markdown(ss.iso_report)
        ss.iso_name = Path(doc.name).stem

    r: IsoReport | None = ss.get("iso_report")
    if r is None:
        st.info("Загрузите документ и нажмите «Проверить требования».")
        return
    s = r.stats
    st.divider()
    st.markdown(f'<div style="font-size:1.25rem;font-weight:800;color:{_color(r.verdict_level)}">{html.escape(r.verdict)}</div>',
                unsafe_allow_html=True)
    st.caption(f"{r.document_title} · {r.run_at} · модель: {r.model} · {r.mode} · {r.agent_version} · {r.duration_s} с")
    m = st.columns(6)
    m[0].metric("Требований", s["requirements"])
    m[1].metric("С нарушениями", s["with_issues"])
    m[2].metric("🔴 Критичных", s["critical"])
    m[3].metric("🟠 Важных", s["major"])
    m[4].metric("🟡 Мелких", s["minor"])
    m[5].metric("Отсечено LLM-претензий", s.get("rejected", 0), help="Цитата не найдена в документе")

    tabs = st.tabs(["Итог и сводка", "Нарезка требований", "Проверка требований", "Проверка набора", "Вопросы автору", "Отчёт"])
    with tabs[0]:
        if r.set_review.summary:
            st.markdown(r.set_review.summary)
        st.markdown("**Что исправить в первую очередь:**")
        for k, p in enumerate(r.priorities, 1):
            st.markdown(f"{k}. {p}")
        st.markdown("**Критичные дефекты:**")
        for cls in ("противоречивые", "неполные", "непроверяемые"):
            items = r.critical.get(cls, [])
            with st.expander(f"{cls.capitalize()}: {len(items)}", expanded=cls == "противоречивые"):
                for x in items:
                    st.markdown(f"- {x}")
    with tabs[1]:
        st.dataframe([{"ID в отчёте": u.rid, "ID в документе": u.doc_ref, "Раздел": u.section, "Вид": u.kind,
                       "Стр.": ", ".join(map(str, u.pages)), "Строки": f"L{min(u.lines)}" if u.lines else "",
                       "Формулировка": u.text} for u in r.units], hide_index=True, width="stretch")
    with tabs[2]:
        codes = [c.code for c in INDIVIDUAL] + [SECURITY.code]
        st.dataframe([{"ID": u.rid, **{BY_CODE[c].name: STATUS_ICON.get(u.statuses.get(c, "na"), "—") for c in codes}}
                      for u in r.units], hide_index=True, width="stretch")
        only_bad = st.toggle("Только требования с нарушениями", value=True)
        for u in r.units:
            if only_bad and not u.issues:
                continue
            worst = min((i.severity for i in u.issues), key=lambda x: {"critical": 0, "major": 1, "minor": 2}[x], default="")
            with st.expander(f"{SEVERITY_ICON.get(worst, '✅')} {u.rid} · {u.doc_ref} — {u.text[:90]}"):
                st.markdown(f"> {u.quote}")
                for i in u.issues:
                    st.markdown(f"{SEVERITY_ICON.get(i.severity, '')} **{SEVERITY_LABEL.get(i.severity, '')} · "
                                f"{BY_CODE.get(i.characteristic).name if i.characteristic in BY_CODE else i.characteristic}** — {i.problem}")
                    for e in i.evidence[:2]:
                        st.caption(f"«{e}»")
                    if i.fix:
                        st.caption(f"Исправить: {i.fix}")
                if u.rewrite:
                    st.success(f"Предлагаемая формулировка: {u.rewrite}")
    with tabs[3]:
        sr = r.set_review
        st.dataframe([{"Характеристика": c.name, "Стандарт": c.source,
                       "Оценка": STATUS_ICON.get(sr.statuses.get(c.code, "na"), "—"), "Комментарий": sr.comments.get(c.code, "")}
                      for c in SET], hide_index=True, width="stretch")
        st.markdown(f"**Противоречия ({sum(1 for i in sr.issues if i.characteristic == 'set_consistent')})**")
        for i in [i for i in sr.issues if i.characteristic == "set_consistent"]:
            st.markdown(f"{SEVERITY_ICON.get(i.severity, '')} **{', '.join(i.related)}** — {i.problem}")
            for e in i.evidence[:3]:
                st.caption(f"«{e}»")
        if sr.missing:
            st.markdown("**Пропущенные требования**")
            st.dataframe(sr.missing, hide_index=True, width="stretch")
        if sr.checklist:
            st.markdown("**Чек-лист ревью из документа, заполненный агентом**")
            st.dataframe(sr.checklist, hide_index=True, width="stretch")
    with tabs[4]:
        for k, q in enumerate(r.set_review.questions, 1):
            st.markdown(f"**{k}. {q.get('question', '')}** {('(' + ', '.join(q.get('related') or []) + ')') if q.get('related') else ''}")
            for o in q.get("options") or []:
                st.markdown(f"- [ ] {o}")
    with tabs[5]:
        st.download_button("⬇️ Скачать отчёт (.md)", ss.iso_md, file_name=f"{ss.get('iso_name', 'report')}_report.md", type="primary")
        st.download_button("⬇️ Скачать отчёт (.json)", r.model_dump_json(indent=2), file_name=f"{ss.get('iso_name', 'report')}_report.json")
        with st.expander("Предпросмотр отчёта", expanded=True):
            st.markdown(ss.iso_md)
