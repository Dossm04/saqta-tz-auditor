"""Streamlit UI: Saqta TZ Auditor — мульти-агентная проверка качества финтех-ТЗ.

Запуск: streamlit run app.py
"""
from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from agents.orchestrator import PIPELINE_INFO, analyze
from config import LLMConfig
from improve import build_revised_tz
from knowledge import DISCLAIMER
from knowledge.domains import DOMAIN_LABELS
from models import Category, Report, Severity
from parsing import Document, load_bytes, load_path, load_text
from report import to_json, to_markdown

ROOT = Path(__file__).resolve().parent
SAMPLES = {
    "Демо-ТЗ: мобильный банк (с дефектами)": ROOT / "samples" / "sample_tz.md",
    "Демо-ТЗ: мобильный банк v2 (исправленное)": ROOT / "samples" / "good_tz.md",
}

st.set_page_config(page_title="Saqta TZ Auditor", page_icon="🛡️", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 10px 12px; margin-bottom: 8px;
       background: rgba(128,128,128,.06);}
.card.run {border-color: #3b82f6; background: rgba(59,130,246,.08);}
.card.ok {border-color: #16a34a; background: rgba(22,163,74,.08);}
.card.warn {border-color: #d97706; background: rgba(217,119,6,.10);}
.card.err {border-color: #dc2626; background: rgba(220,38,38,.10);}
.card .t {font-weight: 700;} .card .r {font-size: .78rem; opacity: .75;} .card .s {font-size: .8rem; margin-top: 4px;}
.score {font-size: 3.2rem; font-weight: 800; line-height: 1;}
.pill {display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: .78rem; font-weight: 600;
       margin-right: 6px; border: 1px solid rgba(128,128,128,.35);}
.p0 {background: rgba(220,38,38,.15); border-color: #dc2626;} .p1 {background: rgba(217,119,6,.15); border-color: #d97706;}
.p2 {background: rgba(202,138,4,.12); border-color: #ca8a04;}
.bar {height: 9px; border-radius: 6px; background: rgba(128,128,128,.2); overflow: hidden;}
.bar > div {height: 100%; border-radius: 6px;}
.doc {font-family: ui-monospace, Consolas, monospace; font-size: .82rem; line-height: 1.5;}
.doc .ln {display:inline-block; width: 3.2em; opacity: .5; text-align: right; margin-right: .8em;}
.doc .row {padding: 1px 6px; border-radius: 4px; white-space: pre-wrap;}
.doc .critical {background: rgba(220,38,38,.18);} .doc .major {background: rgba(217,119,6,.18);}
.doc .minor {background: rgba(202,138,4,.14);} .doc .tag {font-size: .72rem; opacity: .85; margin-left: .6em;}
blockquote {margin: .3em 0;}
</style>
""",
    unsafe_allow_html=True,
)


def _color(score: int) -> str:
    return "#16a34a" if score >= 80 else "#d97706" if score >= 55 else "#dc2626"


# ── Sidebar: настройки LLM ────────────────────────────────────────────────────
env_cfg = LLMConfig.from_env()
with st.sidebar:
    st.markdown("### 🛡️ Saqta TZ Auditor")
    st.caption("Мульти-агентная проверка качества ТЗ финтех-проектов · BCC-HUB × MNU FinTech Hackathon")
    st.divider()
    st.markdown("**LLM-провайдер** (OpenAI-совместимый API)")
    base_url = st.text_input("Base URL", value=env_cfg.base_url or "", placeholder="https://api.openai.com/v1")
    api_key = st.text_input("API key", value=env_cfg.api_key or "", type="password")
    model = st.text_input("Модель", value=env_cfg.model)
    cfg = env_cfg.with_overrides(base_url=base_url or None, api_key=api_key or None, model=model)
    use_llm = st.toggle("Использовать LLM-агентов", value=cfg.enabled, disabled=not cfg.enabled,
                        help="Без ключа агенты работают на детерминированных правилах базы знаний")
    if cfg.enabled and use_llm:
        st.success(f"Режим: LLM-агенты + правила ({cfg.model})", icon="🧠")
    else:
        st.info("Режим: офлайн-правила (без LLM). Добавьте ключ, чтобы включить LLM-агентов.", icon="⚙️")
    st.divider()
    st.caption(DISCLAIMER)


# ── Заголовок и ввод ──────────────────────────────────────────────────────────
st.title("Проверка качества технического задания")
st.caption("Чтение ТЗ → анализ требований → проверка (полнота, однозначность, ИБ и регулятор РК) → вопросы и оценка")

src = st.radio("Источник ТЗ", [*SAMPLES.keys(), "Загрузить файл", "Вставить текст"], horizontal=True)
doc: Document | None = None
if src in SAMPLES:
    doc = load_path(SAMPLES[src])
    with st.expander("Посмотреть текст демо-ТЗ"):
        st.code(doc.text, language="markdown")
elif src == "Загрузить файл":
    up = st.file_uploader("ТЗ в формате .md, .txt, .docx или .pdf", type=["md", "txt", "docx", "pdf"])
    if up is not None:
        doc = load_bytes(up.getvalue(), up.name)
        st.caption(f"Загружено: {doc.name}, {doc.non_empty_count} непустых строк")
else:
    txt = st.text_area("Текст ТЗ", height=220, placeholder="Вставьте текст технического задания…")
    if txt.strip():
        doc = load_text(txt)

with st.expander("📐 Шаблон ТЗ (необязательно): проверить, все ли его разделы на месте"):
    tpl_src = st.radio("Шаблон", ["Без шаблона", "Пример шаблона банка", "Загрузить шаблон организаторов"],
                       horizontal=True, label_visibility="collapsed")
    template: Document | None = None
    if tpl_src == "Пример шаблона банка":
        template = load_path(ROOT / "samples" / "template_example.md")
    elif tpl_src == "Загрузить шаблон организаторов":
        tup = st.file_uploader("Шаблон в .md, .txt, .docx или .pdf", type=["md", "txt", "docx", "pdf"], key="tpl")
        if tup is not None:
            template = load_bytes(tup.getvalue(), tup.name)
    if template is not None:
        from knowledge.template import sections_from_template

        secs = sections_from_template(template)
        st.caption(f"Разделов в шаблоне: {len(secs)} — " + "; ".join(s.title for s in secs[:12]))

ss = st.session_state
ss.setdefault("history", [])


def render_pipeline(container):
    """Рисует граф агентов и возвращает плейсхолдеры карточек по имени агента."""
    holders = {}
    with container:
        cols = st.columns(len(PIPELINE_INFO))
        for col, stage in zip(cols, PIPELINE_INFO):
            with col:
                for name, title, role, _ in stage:
                    holders[name] = st.empty()
                    holders[name].markdown(f'<div class="card"><div class="t">⏸️ {title}</div><div class="r">{role}</div></div>',
                                           unsafe_allow_html=True)
    return holders


def run(doc: Document, label: str) -> None:
    st.markdown("#### Агенты за работой")
    holders = render_pipeline(st.container())
    roles = {n: r for stage in PIPELINE_INFO for n, _, r, _ in stage}

    def on_event(kind, agent, r):
        if kind == "start":
            holders[agent.name].markdown(
                f'<div class="card run"><div class="t">⏳ {agent.title}</div><div class="r">{roles[agent.name]}</div></div>',
                unsafe_allow_html=True)
        else:
            cls = "err" if r.status == "error" else "warn" if r.mode == "fallback" else "ok"
            icon = "❌" if r.status == "error" else "⚠️" if r.mode == "fallback" else "✅"
            holders[agent.name].markdown(
                f'<div class="card {cls}"><div class="t">{icon} {agent.title}</div><div class="r">{roles[agent.name]}</div>'
                f'<div class="s"><b>{r.produced}</b> · {r.mode} · {r.duration_s} с<br>{html.escape(r.note[:140])}</div></div>',
                unsafe_allow_html=True)

    report = analyze(doc, cfg, use_llm=use_llm, on_event=on_event, template=ss.get("template"))
    ss.report, ss.doc = report, doc
    ss.history.append((label, report.overall_score, report.severity_counts()["critical"]))
    ss.answers = {}
    ss.run_id = ss.get("run_id", 0) + 1


c1, c2 = st.columns([1.3, 4])
if c1.button("🚀 Анализировать", type="primary", disabled=doc is None, width="stretch"):
    ss.template = template
    run(doc, doc.name)

report: Report | None = ss.get("report")
if report is None:
    st.info("Выберите или загрузите ТЗ и нажмите «Анализировать».")
    st.stop()

doc = ss.doc
st.divider()

# ── Сводка ────────────────────────────────────────────────────────────────────
sc = report.severity_counts()
cov, part, total = report.control_coverage()
left, mid, right = st.columns([1.1, 2.2, 1.6])
with left:
    delta = ""
    if len(ss.history) >= 2:
        prev = ss.history[-2][1]
        d = report.overall_score - prev
        delta = f'<div style="font-size:.9rem;color:{_color(80 if d >= 0 else 0)}">{"▲" if d >= 0 else "▼"} {abs(d)} к прошлой проверке ({prev})</div>'
    st.markdown(f'<div class="score" style="color:{_color(report.overall_score)}">{report.overall_score}<span style="font-size:1.2rem;opacity:.6">/100</span></div>'
                f'<div style="font-weight:700;color:{_color(report.overall_score)}">{report.verdict}</div>{delta}',
                unsafe_allow_html=True)
    st.markdown(f'<div style="margin-top:.4rem;font-size:.92rem">⏱ <b>{report.total_duration_s} с</b> вместо '
                f'<b>≈{report.manual_review_hours} ч</b> ручного ревью</div>', unsafe_allow_html=True,
                help="Оценка ручного ревью: 30 мин чтение + 2 мин на требование + 0,4 мин на строку + "
                     "45 мин проверка ИБ/регуляторики + 3 мин на вопрос")
    st.caption(f"{report.document.title} · {report.mode} · {report.model}"
               + (f" · {report.total_tokens} ток." if report.total_tokens else ""))
with mid:
    st.markdown(report.summary)
    if report.strengths:
        st.markdown("**Сильные стороны:** " + " · ".join(report.strengths))
with right:
    for d in report.dimensions:
        st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:.86rem"><span>{d.name} <span style="opacity:.55">×{d.weight:.2f}</span></span><b>{d.score}</b></div>'
                    f'<div class="bar"><div style="width:{d.score}%;background:{_color(d.score)}"></div></div>',
                    unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🔴 Критично", sc["critical"])
m2.metric("🟠 Важно", sc["major"])
m3.metric("🟡 Мелочь", sc["minor"])
m4.metric("❓ Вопросов", len(report.questions), help=f"P0: {sum(q.priority.value == 'P0' for q in report.questions)}")
m5.metric("🛡️ Контроли ИБ/регулятора", f"{cov}+{part}½ / {total}")
st.caption("Области ТЗ: " + (", ".join(DOMAIN_LABELS.get(d, d) for d in report.document.domains) or "—"))

tabs = st.tabs([f"❓ Уточняющие вопросы ({len(report.questions)})", f"🔎 Находки ({len(report.findings)})",
                "🛡️ Матрица контролей", "📄 Документ", "🧠 Агенты и архитектура", "⬇️ Экспорт"])

# ── Вопросы + цикл доработки ──────────────────────────────────────────────────
with tabs[0]:
    st.caption("Вопросы к автору ТЗ, отсортированы по приоритету. Выберите вариант или впишите свой — "
               "затем сформируйте ТЗ v2 и перепроверьте его.")
    findings_by_id = {f.id: f for f in report.findings}
    ss.setdefault("answers", {})
    for q in report.questions:
        pcls = q.priority.value.lower()
        with st.container(border=True):
            st.markdown(f'<span class="pill {pcls}">{q.priority.label}</span><span class="pill">{html.escape(q.addressee)}</span>'
                        f'{"<span class=pill>⛔ блокер</span>" if q.blocking else ""}<span class="pill">{q.id}</span>',
                        unsafe_allow_html=True)
            st.markdown(f"**{q.question}**")
            if q.context:
                st.markdown(f"> {q.context}")
            if q.why:
                st.caption(f"Зачем: {q.why}")
            rel = [findings_by_id[r] for r in q.related_findings if r in findings_by_id]
            if rel:
                st.caption("Связано: " + ", ".join(f"{f.id} {f.severity.icon} {f.title}" for f in rel[:4]))
            opts = list(q.answer_options)
            if opts:
                choice = st.radio("Ответ", ["—", *opts, "Свой ответ"], key=f"r_{ss.run_id}_{q.id}", horizontal=False,
                                  label_visibility="collapsed")
                if choice == "Свой ответ":
                    ss.answers[q.id] = st.text_input("Ваш ответ", key=f"t_{ss.run_id}_{q.id}")
                elif choice != "—":
                    ss.answers[q.id] = choice
                else:
                    ss.answers.pop(q.id, None)
            else:
                ans = st.text_input("Ответ", key=f"t_{ss.run_id}_{q.id}", placeholder="Ответ автора ТЗ…", label_visibility="collapsed")
                if ans.strip():
                    ss.answers[q.id] = ans
                else:
                    ss.answers.pop(q.id, None)

    st.divider()
    revised, stats = build_revised_tz(doc, report, ss.answers)
    st.markdown(f"**ТЗ v2 (черновик):** заменено опасных/размытых строк — {stats['replaced']}, "
                f"добавлено требований — {stats['added']}, внесено ответов — {stats['answers']}.")
    b1, b2 = st.columns(2)
    if b1.button("🔁 Сформировать ТЗ v2 и перепроверить", type="primary", width="stretch"):
        name = Path(doc.name).stem + "_v2.md"
        run(load_text(revised, name), name)
        st.rerun()
    b2.download_button("⬇️ Скачать ТЗ v2 (.md)", revised, file_name=Path(doc.name).stem + "_v2.md", width="stretch")
    if len(ss.history) > 1:
        st.markdown("**История проверок:** " + " → ".join(f"{n}: **{s}**" for n, s, _ in ss.history))

# ── Находки ───────────────────────────────────────────────────────────────────
with tabs[1]:
    f1, f2, f3 = st.columns(3)
    sev_sel = f1.multiselect("Серьёзность", [s.label for s in Severity], default=[s.label for s in Severity])
    cats = sorted({f.category for f in report.findings}, key=lambda c: c.value)
    cat_sel = f2.multiselect("Категория", [c.label for c in cats], default=[c.label for c in cats])
    agents = sorted({a for f in report.findings for a in f.sources})
    ag_sel = f3.multiselect("Агент", agents, default=agents)
    shown = [f for f in report.findings if f.severity.label in sev_sel and f.category.label in cat_sel
             and set(f.sources) & set(ag_sel)]
    st.caption(f"Показано {len(shown)} из {len(report.findings)}")
    for f in shown:
        loc = f" · L{', L'.join(map(str, f.lines[:4]))}" if f.lines else ""
        with st.expander(f"{f.severity.icon} {f.id} · {f.title}{loc}", expanded=f.severity == Severity.critical):
            st.markdown(f"<span class='pill'>{f.severity.label}</span><span class='pill'>{f.category.label}</span>"
                        f"<span class='pill'>агенты: {', '.join(f.sources)}</span><span class='pill'>уверенность {f.confidence:.0%}</span>"
                        f"{'<span class=pill>✔ цитата проверена</span>' if f.verified else ''}", unsafe_allow_html=True)
            if f.description:
                st.markdown(f.description)
            for e in f.evidence[:3]:
                st.markdown(f"> {e}")
            if f.regulation:
                st.markdown(f"**Норма / стандарт:** {f.regulation}")
            if f.recommendation:
                st.markdown(f"**Рекомендация:** {f.recommendation}")
            if f.suggested_text:
                st.success(f"**Предлагаемая формулировка:** {f.suggested_text}")
            if f.verification_note:
                st.caption(f"Verifier: {f.verification_note}")

# ── Матрица контролей ─────────────────────────────────────────────────────────
with tabs[2]:
    st.caption("Чеклист финтех/ИБ-контролей для РК: НБРК/АРРФР, ПОД/ФТ, ПДн, платежи, PCI DSS, OWASP. "
               "Применимость определяется доменами ТЗ.")
    icon = {"covered": "✅ покрыто", "partial": "🟨 частично", "missing": "❌ нет", "not_applicable": "— н/п"}
    show_na = st.toggle("Показать неприменимые", value=False)
    rows = [{"ID": c.control_id, "Область": c.area, "Контроль": c.title, "Статус": icon.get(c.status, c.status),
             "Строки ТЗ": ", ".join(f"L{n}" for n in c.lines[:4]), "Норма": c.regulation, "Комментарий": c.comment}
            for c in report.controls if show_na or c.status != "not_applicable"]
    st.dataframe(rows, width="stretch", hide_index=True,
                 column_config={"Норма": st.column_config.TextColumn(width="large")})
    if report.template_check:
        ok = sum(t["present"] for t in report.template_check)
        st.markdown(f"**Соответствие шаблону ТЗ: {ok}/{len(report.template_check)} разделов**")
        st.dataframe([{"Раздел шаблона": t["title"], "В ТЗ": "✅" if t["present"] else "❌",
                       "Найденный заголовок": t.get("matched") or "", "Сходство": f"{t['score']:.0%}"}
                      for t in report.template_check], width="stretch", hide_index=True)

# ── Документ с разметкой ─────────────────────────────────────────────────────
with tabs[3]:
    by_line: dict[int, list] = {}
    for f in report.findings:
        for n in f.lines:
            by_line.setdefault(n, []).append(f)
    rows = []
    for i, line in enumerate(doc.lines, start=1):
        fs = by_line.get(i, [])
        cls = min((f.severity for f in fs), key=lambda s: s.rank).value if fs else ""
        tag = f'<span class="tag">◀ {", ".join(f.id for f in fs)}</span>' if fs else ""
        title = html.escape("; ".join(f"{f.id}: {f.title}" for f in fs))
        rows.append(f'<div class="row {cls}" title="{title}"><span class="ln">{i}</span>{html.escape(line) or "&nbsp;"}{tag}</div>')
    st.caption("Строки с находками подсвечены (наведите курсор, чтобы увидеть находки).")
    st.markdown(f'<div class="doc">{"".join(rows)}</div>', unsafe_allow_html=True)

# ── Агенты и архитектура ─────────────────────────────────────────────────────
with tabs[4]:
    st.graphviz_chart("""
digraph G {
  rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.35;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, fillcolor="#eef2ff", color="#6366f1"];
  {rank=same; comp; cons; sec; kb;}
  edge [color="#94a3b8"];
  doc [label="ТЗ\\n(.md/.docx/.pdf)", shape=note, fillcolor="#f1f5f9", color="#64748b"];
  kb [label="База знаний\\nконтроли РК · антипаттерны\\nГОСТ 34.602 · ISO 29148", shape=cylinder, fillcolor="#fef3c7", color="#d97706"];
  orch [label="Orchestrator", fillcolor="#e0e7ff"];
  reader [label="1 · Reader\\nструктура, домены"];
  comp [label="2 · Completeness\\nполнота"]; cons [label="2 · Consistency\\nпротиворечия, размытость"];
  sec [label="2 · FinTech/Security\\nИБ · AML/KYC · PCI · ПДн", fillcolor="#fee2e2", color="#dc2626"];
  ver [label="3 · Verifier (Critic)\\nцитаты · дубли · консенсус", fillcolor="#dcfce7", color="#16a34a"];
  q [label="4 · Question Generator\\nвопросы P0–P2", fillcolor="#fee2e2", color="#dc2626"];
  sc [label="5 · Scorer\\nоценка · вердикт"];
  out [label="Отчёт + ТЗ v2", shape=note, fillcolor="#f1f5f9", color="#64748b"];
  doc -> orch -> reader; reader -> comp; reader -> cons; reader -> sec;
  kb -> comp [style=dashed]; kb -> cons [style=dashed]; kb -> sec [style=dashed]; kb -> ver [style=dashed];
  comp -> ver; cons -> ver; sec -> ver; ver -> q -> sc -> out;
  out -> doc [label="ответы → перепроверка", style=dotted, fontsize=9];
}
""")
    st.markdown("**Трассировка запуска**")
    st.dataframe([{"Стадия": t.stage, "Агент": t.title, "Режим": t.mode, "Статус": t.status, "Время, с": t.duration_s,
                   "Результат": t.produced, "Токены": t.tokens, "Примечание": t.note} for t in report.trace],
                 width="stretch", hide_index=True)
    v = report.verifier
    st.markdown(f"**Verifier:** на входе {v.input_findings} находок → объединено дублей {v.merged_duplicates}, "
                f"отсечено неподтверждённых {v.rejected_hallucinations}, понижено {v.downgraded} → **{v.output_findings}** в отчёте.")
    st.caption("Каждый агент работает в двух режимах: LLM (узкий промпт + JSON-схема) и детерминированные правила. "
               "При сбое LLM агент автоматически откатывается на правила (режим fallback).")

# ── Экспорт ───────────────────────────────────────────────────────────────────
with tabs[5]:
    stem = Path(report.document_name).stem
    from report_docx import to_docx

    e0, e1, e2, e3 = st.columns(4)
    e0.download_button("📄 Отчёт (Word)", to_docx(report), file_name=f"{stem}_review.docx", width="stretch",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary")
    e1.download_button("📝 Отчёт (Markdown)", to_markdown(report), file_name=f"{stem}_review.md", width="stretch")
    e2.download_button("🧾 Отчёт (JSON)", to_json(report), file_name=f"{stem}_review.json", width="stretch")
    q_md = "\n\n".join(f"{q.id} [{q.priority.value}] ({q.addressee}) {q.question}" +
                       ("".join(f"\n  - [ ] {o}" for o in q.answer_options)) for q in report.questions)
    e3.download_button("❓ Только вопросы", q_md, file_name=f"{stem}_questions.md", width="stretch")
    with st.expander("Предпросмотр отчёта"):
        st.markdown(to_markdown(report))
