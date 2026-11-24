"""Streamlit UI: Saqta TZ Auditor — мульти-агентная проверка качества финтех-ТЗ.

Запуск: streamlit run app.py
"""
from __future__ import annotations

import html
from dataclasses import replace
from pathlib import Path
from typing import Any

import streamlit as st

import config as config_mod
from agents.orchestrator import PIPELINE_INFO, analyze
from config import LLMConfig
from improve import build_revised_tz
from knowledge import DISCLAIMER
from knowledge.domains import DOMAIN_LABELS
from models import Report, Severity
from parsing import Document, load_bytes, load_path, load_text
from report import question_quality_line, to_json, to_markdown

ROOT = Path(__file__).resolve().parent
SAMPLES = {
    "Мобильный банк (с дефектами)": ROOT / "samples" / "sample_tz.md",
    "Мобильный банк v2 (эталон)": ROOT / "samples" / "good_tz.md",
    "Онлайн-кредитование / BNPL": ROOT / "samples" / "lending_tz.md",
    "Интернет-эквайринг / API": ROOT / "samples" / "acquiring_tz.md",
    "Электронный кошелёк": ROOT / "samples" / "wallet_tz.md",
}
SAMPLES = {k: v for k, v in SAMPLES.items() if v.exists()}
MEETING_MIN = {"P0": 5, "P1": 3, "P2": 1}   # минут на обсуждение вопроса

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
.big {font-size: 2.4rem; font-weight: 800; line-height: 1.1;}
.pill {display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: .78rem; font-weight: 600;
       margin: 0 6px 4px 0; border: 1px solid rgba(128,128,128,.35);}
.p0 {background: rgba(220,38,38,.15); border-color: #dc2626;} .p1 {background: rgba(217,119,6,.15); border-color: #d97706;}
.p2 {background: rgba(202,138,4,.12); border-color: #ca8a04;}
.bar {height: 9px; border-radius: 6px; background: rgba(128,128,128,.2); overflow: hidden;}
.bar > div {height: 100%; border-radius: 6px;}
.doc {font-family: ui-monospace, Consolas, monospace; font-size: .82rem; line-height: 1.5;}
.doc .ln {display:inline-block; width: 3.2em; opacity: .5; text-align: right; margin-right: .8em;}
.doc .row {padding: 1px 6px; border-radius: 4px; white-space: pre-wrap;}
.doc .critical {background: rgba(220,38,38,.18);} .doc .major {background: rgba(217,119,6,.18);}
.doc .minor {background: rgba(202,138,4,.14);} .doc .tag {font-size: .72rem; opacity: .85; margin-left: .6em;}
.assume {font-size: .82rem; padding: 4px 8px; border-left: 3px solid #6366f1; background: rgba(99,102,241,.08); margin: 4px 0;}
blockquote {margin: .3em 0;}
</style>
""",
    unsafe_allow_html=True,
)


def _color(score: int) -> str:
    return "#16a34a" if score >= 80 else "#d97706" if score >= 55 else "#dc2626"


def _g(obj: Any, name: str, default: Any = None) -> Any:
    """Чтение поля у pydantic-объекта или dict (новые поля могут отсутствовать)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _same_endpoint(a: str | None, b: str | None) -> bool:
    return (a or "").strip().rstrip("/").lower() == (b or "").strip().rstrip("/").lower()


def _letters_profile(text: str) -> tuple[float, float]:
    """(доля латиницы среди букв, доля казахских букв әғқңөұүһі среди кириллицы)."""
    latin = sum(1 for ch in text if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))
    cyr = sum(1 for ch in text if "Ѐ" <= ch <= "ӿ")
    kaz = sum(1 for ch in text.lower() if ch in "әғқңөұүһі")
    letters = latin + cyr
    return (latin / letters if letters else 0.0), (kaz / cyr if cyr else 0.0)


MIN_DOC_LINES = 5                         # меньше — оценка бессмысленна, анализ блокируется
FINTECH_DOMAINS = {"transfers", "cards", "kyc", "lending", "deposits", "crypto"}


# ── Sidebar: провайдер LLM ────────────────────────────────────────────────────
env_cfg = LLMConfig.from_env()
ENV_PRESET = "Из .env"
presets = {ENV_PRESET: {}, **config_mod.PROVIDER_PRESETS}

with st.sidebar:
    st.markdown("### 🛡️ Saqta TZ Auditor")
    st.caption("Мульти-агентная проверка качества ТЗ финтех-проектов · BCC-HUB × MNU FinTech Hackathon")
    st.divider()
    preset_name = st.selectbox("Провайдер LLM", list(presets.keys()))
    preset = presets[preset_name] or {}
    if _g(preset, "note"):
        st.caption(_g(preset, "note"))
    offline = bool(_g(preset, "offline", False))
    # Конфиг строится из пресета целиком (тайм-ауты, бюджет ожидания, резервные модели, приватность, ключ),
    # поверх него — только то, что пользователь поменял в полях ниже.
    if preset_name == ENV_PRESET:
        base_cfg, pov = env_cfg, {}
        default_url, default_model = env_cfg.base_url or "", env_cfg.model
    else:
        base_cfg, pov = env_cfg.with_preset(preset_name), config_mod.preset_overrides(preset_name)
        default_url, default_model = pov.get("base_url") or "", pov.get("model") or ""
    base_url = st.text_input("Base URL", value=default_url, disabled=offline, key=f"llm_url::{preset_name}",
                             placeholder="https://…/v1")
    url = base_url.strip() or None
    # ключ из .env отправляем только на тот же эндпоинт, для которого он задан в .env
    env_key_ok = bool((env_cfg.api_key or "").strip()) and _same_endpoint(url, env_cfg.base_url)
    preset_key = (pov.get("api_key") or "").strip()
    api_key = st.text_input("API key", value="", type="password", disabled=offline, key=f"llm_key::{preset_name}",
                            placeholder="ключ из пресета ✓" if preset_key else
                            "ключ задан в .env ✓" if env_key_ok else "вставьте ключ")
    model = st.text_input("Модель", value=default_model, disabled=offline, key=f"llm_model::{preset_name}")
    key = api_key.strip() or preset_key or ((env_cfg.api_key or "").strip() if env_key_ok else "") or None
    if "fallback_models" in pov:
        fallbacks = tuple(pov["fallback_models"] or ())
    else:      # резервные модели из .env имеют смысл только для того же провайдера
        fallbacks = env_cfg.fallback_models if _same_endpoint(url, env_cfg.base_url) else ()
    cfg_problem = ""
    if not offline and preset_name != ENV_PRESET and not url:
        cfg_problem = "Укажите Base URL провайдера — без него анализ с LLM недоступен."
    elif not offline and not model.strip():
        cfg_problem = "Укажите модель — без неё анализ с LLM недоступен."
    cfg = replace(base_cfg, base_url=url, api_key=key, model=model.strip() or base_cfg.model,
                  fallback_models=fallbacks)
    local = bool(getattr(cfg, "is_local", False))
    if local and not cfg.api_key:
        cfg = cfg.with_overrides(api_key="local")
    st.session_state["llm_cfg"] = cfg   # для отладки и UI-тестов
    if cfg_problem:
        st.warning(cfg_problem, icon="⚠️")
    use_llm = (not offline) and not cfg_problem and cfg.enabled and st.toggle("Использовать LLM-агентов", value=True)
    if not offline and not cfg_problem and not cfg.enabled:
        st.caption("Для этого провайдера не задан ключ — анализ пойдёт на офлайн-правилах.")
    if use_llm:
        st.success(f"LLM-агенты + правила · {cfg.model}", icon="🧠")
        if local:
            st.caption("🔒 Модель в контуре банка — данные не покидают периметр")
        elif getattr(cfg, "privacy_enabled", False):
            st.caption("🔒 Режим приватности: ИИН, карты, телефоны, e-mail маскируются перед отправкой в облако")
        if (ROOT / ".cache" / "llm").exists():
            st.caption("⚡ Повторный анализ того же ТЗ берётся из кэша ответов — мгновенно")
    else:
        st.info("Офлайн-режим: детерминированные агенты на базе знаний (без LLM).", icon="⚙️")
    st.divider()
    st.caption(DISCLAIMER)


# ── Ввод ──────────────────────────────────────────────────────────────────────
st.title("Проверка качества технического задания")
st.caption("Чтение ТЗ → анализ требований → проверка (полнота, однозначность, ИБ и регулятор РК) → уточняющие вопросы и оценка")

src = st.radio("Источник ТЗ", [*SAMPLES.keys(), "Загрузить файл", "Вставить текст"], horizontal=True)
doc: Document | None = None
try:
    if src in SAMPLES:
        doc = load_path(SAMPLES[src])
        with st.expander("Посмотреть текст ТЗ"):
            st.code(doc.text, language="markdown")
    elif src == "Загрузить файл":
        up = st.file_uploader("ТЗ в формате .md, .txt, .docx или .pdf", type=["md", "txt", "docx", "pdf"])
        if up is not None:
            doc = load_bytes(up.getvalue(), up.name)
            if doc.non_empty_count == 0:
                st.warning("В файле не найден текст (возможно, это скан PDF без текстового слоя).")
                doc = None
            else:
                st.caption(f"Загружено: {doc.name}, {doc.non_empty_count} непустых строк")
    else:
        txt = st.text_area("Текст ТЗ", height=220, placeholder="Вставьте текст технического задания…")
        if txt.strip():
            doc = load_text(txt)
except Exception as e:  # повреждённый docx/pdf не должен ронять демо
    st.error(f"Не удалось прочитать файл: {e}")
    doc = None

with st.expander("📐 Шаблон ТЗ (необязательно): проверить, все ли его разделы на месте"):
    tpl_src = st.radio("Шаблон", ["Без шаблона", "Пример шаблона банка", "Загрузить шаблон организаторов"],
                       horizontal=True, label_visibility="collapsed")
    template: Document | None = None
    try:   # повреждённый или переименованный файл шаблона не должен ронять страницу
        if tpl_src == "Пример шаблона банка":
            template = load_path(ROOT / "samples" / "template_example.md")
        elif tpl_src == "Загрузить шаблон организаторов":
            tup = st.file_uploader("Шаблон в .md, .txt, .docx или .pdf", type=["md", "txt", "docx", "pdf"], key="tpl")
            if tup is not None:
                template = load_bytes(tup.getvalue(), tup.name)
    except Exception as e:
        st.error(f"Не удалось прочитать шаблон: {e}")
        template = None
    if template is not None:
        from knowledge.template import sections_from_template

        try:
            secs = sections_from_template(template)
        except Exception as e:
            st.error(f"Не удалось разобрать шаблон: {e}")
            secs, template = [], None
        if template is not None and not secs:
            st.warning("В шаблоне не найдено ни одного раздела (скан PDF без текста или .docx без стилей заголовков) — "
                       "проверка по шаблону выполняться не будет.")
            template = None
        elif secs:
            st.caption(f"Разделов в шаблоне: {len(secs)} — " + "; ".join(s.title for s in secs[:12]))
tpl_label = ("без шаблона" if template is None else
             "пример шаблона банка" if tpl_src == "Пример шаблона банка" else f"«{template.name}»")

# Проверки входа до анализа: слишком короткий текст и язык, на который не рассчитаны офлайн-правила
block_reason = ""
if doc is not None:
    if doc.non_empty_count < MIN_DOC_LINES:
        block_reason = (f"В ТЗ всего {doc.non_empty_count} непустых строк — этого недостаточно для оценки "
                        f"(нужно хотя бы {MIN_DOC_LINES}). Добавьте текст требований.")
        st.warning(block_reason, icon="⚠️")
    latin_share, kaz_share = _letters_profile(doc.text)
    if latin_share > 0.5 or kaz_share > 0.02:
        lang = "английском" if latin_share > 0.5 else "казахском"
        st.warning(f"Похоже, ТЗ на {lang} языке: офлайн-правила рассчитаны на русский язык, часть разделов и "
                   "опасных требований может быть не распознана. Используйте LLM-режим.", icon="🌐")

ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("answers", {})
ss.setdefault("run_id", 0)


def render_pipeline(container):
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


def run(doc: Document, label: str, keep_prev: bool = False) -> None:
    st.markdown("#### Агенты за работой")
    holders = render_pipeline(st.container())
    roles = {n: r for stage in PIPELINE_INFO for n, _, r, _ in stage}

    def on_event(kind, agent, r):
        if agent.name not in holders:
            return
        if kind == "start":
            holders[agent.name].markdown(
                f'<div class="card run"><div class="t">⏳ {agent.title}</div><div class="r">{roles.get(agent.name, "")}</div></div>',
                unsafe_allow_html=True)
        else:
            cls = "err" if r.status == "error" else "warn" if r.mode == "fallback" else "ok"
            icon = "❌" if r.status == "error" else "⚠️" if r.mode == "fallback" else "✅"
            holders[agent.name].markdown(
                f'<div class="card {cls}"><div class="t">{icon} {agent.title}</div><div class="r">{roles.get(agent.name, "")}</div>'
                f'<div class="s"><b>{r.produced}</b> · {r.mode} · {r.duration_s} с<br>{html.escape(r.note[:140])}</div></div>',
                unsafe_allow_html=True)

    report = analyze(doc, cfg, use_llm=use_llm, on_event=on_event, template=ss.get("template"))
    ss.prev_report = ss.get("report") if keep_prev else None
    ss.report, ss.doc = report, doc
    ss.history.append((label, report.overall_score, report.severity_counts()["critical"]))
    ss.answers = {}
    ss.run_id += 1


c1, c2, _ = st.columns([1.3, 1.1, 3])
if c1.button("🚀 Анализировать", type="primary", disabled=doc is None or bool(block_reason or cfg_problem),
             width="stretch"):
    ss.template, ss.template_label = template, tpl_label
    run(doc, doc.name)
if c2.button("↺ Начать заново", width="stretch", disabled="report" not in ss):
    for k in ("report", "doc", "prev_report", "history", "answers", "template", "template_label", "exports", "exports_for"):
        ss.pop(k, None)
    st.rerun()

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
    st.markdown(f'<div class="score" style="color:{_color(report.overall_score)}">{report.overall_score}'
                f'<span style="font-size:1.2rem;opacity:.6">/100</span></div>'
                f'<div style="font-weight:700;color:{_color(report.overall_score)}">{report.verdict}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="margin-top:.4rem;font-size:.92rem">⏱ <b>{report.total_duration_s} с</b> вместо '
                f'<b>≈{_g(report, "manual_review_hours", 0)} ч</b> ручного ревью</div>', unsafe_allow_html=True,
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
        st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:.86rem"><span>{d.name} '
                    f'<span style="opacity:.55">×{d.weight:.2f}</span></span><b>{d.score}</b></div>'
                    f'<div class="bar"><div style="width:{d.score}%;background:{_color(d.score)}"></div></div>',
                    unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🔴 Критично", sc["critical"])
m2.metric("🟠 Важно", sc["major"])
m3.metric("🟡 Мелочь", sc["minor"])
m4.metric("❓ Вопросов", len(report.questions), help=f"P0: {sum(q.priority.value == 'P0' for q in report.questions)}")
m5.metric("🛡️ Контроли ИБ/регулятора", f"{cov}+{part}½ / {total}")
st.caption("Области ТЗ: " + (", ".join(DOMAIN_LABELS.get(d, d) for d in report.document.domains) or "—"))
if not FINTECH_DOMAINS & set(report.document.domains):
    st.warning("ТЗ не похоже на финтех-проект: не найдено ни переводов, ни карт, ни KYC, ни кредитов, ни депозитов. "
               "Финтех- и регуляторные проверки рассчитаны на банковские продукты — замечания про АРРФР, MFA "
               "клиентов, certificate pinning и т. п. могут быть неприменимы к этому ТЗ.", icon="🏦")

# ── До / После ────────────────────────────────────────────────────────────────
prev: Report | None = ss.get("prev_report")
if prev is not None:
    with st.container(border=True):
        st.markdown("#### 🔁 До / После: ТЗ v1 → ТЗ v2")
        a, b, c = st.columns(3)
        d_score = report.overall_score - prev.overall_score
        a.markdown(f'<div class="big"><span style="color:{_color(prev.overall_score)}">{prev.overall_score}</span> → '
                   f'<span style="color:{_color(report.overall_score)}">{report.overall_score}</span></div>'
                   f'<div style="color:{"#16a34a" if d_score >= 0 else "#dc2626"};font-weight:700">'
                   f'{"▲" if d_score >= 0 else "▼"} {abs(d_score)} баллов</div>', unsafe_allow_html=True)
        pc, rc = prev.severity_counts(), sc
        b.metric("Критичных замечаний", rc["critical"], delta=rc["critical"] - pc["critical"], delta_color="inverse")
        pcov, rcov = prev.control_coverage(), report.control_coverage()
        c.metric("Контролей покрыто", f"{rcov[0]} / {rcov[2]}", delta=rcov[0] - pcov[0])
        prev_dims = {d.key: d.score for d in prev.dimensions}
        st.dataframe([{"Измерение": d.name, "Было": prev_dims.get(d.key, 0), "Стало": d.score,
                       "Δ": d.score - prev_dims.get(d.key, 0)} for d in report.dimensions], hide_index=True, width="stretch")

        def key(f):
            return f.control_id or f.title
        now_keys = {key(f) for f in report.findings}
        closed = [f for f in prev.findings if key(f) not in now_keys]
        if closed:
            st.markdown(f"**Закрыто замечаний: {len(closed)}** — " + "; ".join(
                f"{f.severity.icon} {f.title}" for f in sorted(closed, key=lambda x: x.severity.rank)[:12]))

tabs = st.tabs([f"🗓 Повестка встречи ({len(report.questions)})", f"🔎 Находки ({len(report.findings)})",
                "🛡️ Матрица контролей", "📄 Документ", "🧠 Агенты и архитектура", "⬇️ Экспорт"])

# ── Повестка встречи: вопросы по адресатам + цикл доработки ─────────────────
with tabs[0]:
    findings_by_id = {f.id: f for f in report.findings}
    qs = report.questions
    extra_qs = list(_g(report, "extra_questions", []) or [])
    rid = ss.run_id

    def _rec_index(q) -> int:
        ro = _g(q, "recommended_option")
        return ro if isinstance(ro, int) and 0 <= ro < len(q.answer_options) else 0

    def _opt_labels(q) -> list[str]:
        ri = _rec_index(q)
        return [f"★ {o}" if i == ri else o for i, o in enumerate(q.answer_options)]

    def _accept_recommended():
        for q in qs:
            if q.answer_options and q.priority.value in ("P0", "P1"):
                ss[f"r_{rid}_{q.id}"] = _opt_labels(q)[_rec_index(q)]

    def _answer_from_state(q) -> str:
        """Ответ на вопрос по состоянию виджетов (до их отрисовки), чтобы счётчики не отставали на клик."""
        own = ss.get(f"t_{rid}_{q.id}")
        own = own.strip() if isinstance(own, str) else ""
        if not q.answer_options:
            return own
        choice = ss.get(f"r_{rid}_{q.id}")
        if not isinstance(choice, str) or choice == "—":
            return ""
        choice = choice.removeprefix("★ ")
        return own if choice == "Свой ответ" else choice

    ss.answers = {q.id: a for q in [*qs, *extra_qs] if (a := _answer_from_state(q))}
    main_ids = {q.id for q in qs}
    answered = sum(1 for qid in ss.answers if qid in main_ids)
    answered_extra = len(ss.answers) - answered

    total_min = sum(MEETING_MIN.get(q.priority.value, 2) for q in qs)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Вопросов", len(qs))
    k2.metric("Блокеров P0", sum(q.priority.value == "P0" for q in qs))
    k3.metric("Время встречи", f"≈{total_min} мин", help="P0 — 5 мин, P1 — 3 мин, P2 — 1 мин")
    k4.metric("Отвечено", f"{answered}/{len(qs)}" + (f" +{answered_extra} доп." if answered_extra else ""))
    qq_line = question_quality_line(report) if _g(report, "question_quality") is not None else ""
    if qq_line:
        st.caption(qq_line)
    g1, g2 = st.columns([2, 2])
    group = g1.radio("Группировка", ["По адресату", "По приоритету"], horizontal=True, label_visibility="collapsed")
    g2.button("✅ Принять рекомендуемые ответы для P0/P1", on_click=_accept_recommended, width="stretch")

    def _render_question(q) -> None:
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
            if _g(q, "impact"):
                st.caption(f"Если не ответить: {_g(q, 'impact')}")
            if _g(q, "default_assumption"):
                st.markdown(f'<div class="assume">Если ответа нет до старта — закладываем: {html.escape(_g(q, "default_assumption"))}</div>',
                            unsafe_allow_html=True)
            rel = [findings_by_id[r] for r in q.related_findings if r in findings_by_id]
            if rel:
                st.caption("Связано: " + ", ".join(f"{f.id} {f.severity.icon} {f.title}" for f in rel[:4]))
            if q.answer_options:
                choice = st.radio("Ответ", ["—", *_opt_labels(q), "Свой ответ"], key=f"r_{rid}_{q.id}",
                                  label_visibility="collapsed")
                if choice == "Свой ответ":
                    st.text_input("Ваш ответ", key=f"t_{rid}_{q.id}")
            else:
                st.text_input("Ответ", key=f"t_{rid}_{q.id}", placeholder="Ответ автора ТЗ…",
                              label_visibility="collapsed")

    if group == "По адресату":
        groups: dict[str, list] = {}
        for q in qs:
            groups.setdefault(q.addressee or "Бизнес-заказчик", []).append(q)
        ordered = sorted(groups.items(), key=lambda kv: min(MEETING_MIN.get(q.priority.value, 9) * -1 for q in kv[1]))
    else:
        ordered = [(p, [q for q in qs if q.priority.value == p]) for p in ("P0", "P1", "P2")]
        ordered = [(f"{p} — " + {"P0": "блокеры", "P1": "до старта", "P2": "по ходу"}[p], g) for p, g in ordered if g]

    for gname, gqs in ordered:
        mins = sum(MEETING_MIN.get(q.priority.value, 2) for q in gqs)
        st.markdown(f"##### {gname} · {len(gqs)} вопр. · ≈{mins} мин")
        for q in gqs:
            _render_question(q)

    if extra_qs:
        with st.expander(f"➕ Дополнительные вопросы ({len(extra_qs)}) — не вошли в лимит встречи, "
                         "можно ответить письменно"):
            st.caption("Ответы на эти вопросы тоже попадают в ТЗ v2 и в повестку (.md).")
            for q in sorted(extra_qs, key=lambda x: x.priority.value):
                _render_question(q)

    st.divider()
    revised, stats = build_revised_tz(doc, report, ss.answers)
    st.markdown(f"**ТЗ v2 (черновик):** заменено опасных/размытых строк — {stats.get('replaced', 0)}, "
                f"добавлено требований — {stats.get('added', 0)}, внесено ответов — {stats.get('answers', 0)}.")
    b1, b2 = st.columns(2)
    if b1.button("🔁 Сформировать ТЗ v2 и перепроверить", type="primary", width="stretch"):
        name = Path(doc.name).stem + "_v2.md"
        ss.template, ss.template_label = template, tpl_label   # текущий выбор шаблона, а не зафиксированный при анализе
        run(load_text(revised, name), name, keep_prev=True)
        st.rerun()
    b1.caption(f"Перепроверка выполняется с текущим выбором шаблона: {tpl_label}.")
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
    for i, f in enumerate(shown):
        loc = f" · L{', L'.join(map(str, f.lines[:4]))}" if f.lines else ""
        with st.expander(f"{f.severity.icon} {f.id} · {f.title}{loc}", expanded=i < 3 and f.severity == Severity.critical):
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
    st.caption("Чеклист финтех/ИБ-контролей для РК: АРРФР/НБРК, ПОД/ФТ, ПДн, платежи, PCI DSS, OWASP. "
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
    rows_html = []
    for i, line in enumerate(doc.lines, start=1):
        fs = by_line.get(i, [])
        cls = min((f.severity for f in fs), key=lambda s: s.rank).value if fs else ""
        tag = f'<span class="tag">◀ {", ".join(f.id for f in fs)}</span>' if fs else ""
        title = html.escape("; ".join(f"{f.id}: {f.title}" for f in fs))
        rows_html.append(f'<div class="row {cls}" title="{title}"><span class="ln">{i}</span>{html.escape(line) or "&nbsp;"}{tag}</div>')
    st.caption("Строки с находками подсвечены (наведите курсор, чтобы увидеть находки).")
    st.markdown(f'<div class="doc">{"".join(rows_html)}</div>', unsafe_allow_html=True)

# ── Агенты и архитектура ─────────────────────────────────────────────────────
with tabs[4]:
    trace = {t.agent: t for t in report.trace}
    messages = list(_g(report, "messages", []) or [])

    def node(name: str, label: str, color: str = "#6366f1", fill: str = "#eef2ff") -> str:
        t = trace.get(name)
        extra = f"\\n{t.mode} · {t.duration_s} с · {t.produced}" if t else ""
        if t and t.mode == "fallback":
            color, fill = "#d97706", "#fef3c7"
        return f'{name} [label="{label}{extra}", color="{color}", fillcolor="{fill}"];'

    edges = []
    if messages:
        seen = set()
        for m in messages:
            s, r = _g(m, "sender", ""), _g(m, "receiver", "")
            if not s or not r:
                continue
            lab = str(_g(m, "summary", "") or _g(m, "kind", ""))[:38].replace('"', "'")
            cnt = _g(m, "count")
            if cnt not in (None, ""):
                lab = f"{cnt} · {lab}"
            k = (s, r, lab)
            if k in seen:
                continue
            seen.add(k)
            style = ', style=dashed, color="#dc2626"' if "revis" in str(_g(m, "kind", "")) or "ревиз" in lab else ""
            edges.append(f'"{s}" -> "{r}" [label="{lab}", fontsize=9{style}];')
    if not edges:
        edges = ['doc -> reader;', 'reader -> completeness;', 'reader -> consistency;', 'reader -> fintech_security;',
                 'completeness -> verifier;', 'consistency -> verifier;', 'fintech_security -> verifier;',
                 'verifier -> questions;', 'questions -> scorer;', 'scorer -> out;']
    dot = "\n".join([
        'digraph G {', 'rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.4;',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12];',
        'edge [color="#94a3b8", fontname="Helvetica"];',
        'doc [label="ТЗ", shape=note, fillcolor="#f1f5f9", color="#64748b"];',
        'kb [label="База знаний\\nконтроли РК · антипаттерны\\nГОСТ 34.602 · ISO 29148", shape=cylinder, fillcolor="#fef3c7", color="#d97706"];',
        node("reader", "Reader"), node("completeness", "Completeness"), node("consistency", "Consistency"),
        node("fintech_security", "FinTech/Security", "#dc2626", "#fee2e2"),
        node("verifier", "Verifier (Critic)", "#16a34a", "#dcfce7"),
        node("questions", "Question Generator", "#dc2626", "#fee2e2"), node("scorer", "Scorer"),
        'out [label="Отчёт + ТЗ v2", shape=note, fillcolor="#f1f5f9", color="#64748b"];',
        '{rank=same; completeness; consistency; fintech_security; kb;}',
        'kb -> completeness [style=dotted]; kb -> consistency [style=dotted]; kb -> fintech_security [style=dotted]; kb -> verifier [style=dotted];',
        *edges, '}'])
    st.markdown("**Граф реального запуска** (подписи рёбер — сообщения между агентами)" if messages else "**Архитектура агентов**")
    st.graphviz_chart(dot)

    if messages:
        with st.expander(f"Журнал сообщений между агентами ({len(messages)})"):
            st.dataframe([{"От": _g(m, "sender"), "Кому": _g(m, "receiver"), "Тип": _g(m, "kind"),
                           "Кол-во": _g(m, "count"), "Содержание": _g(m, "summary")} for m in messages],
                         hide_index=True, width="stretch")

    st.markdown("**Трассировка запуска**")
    st.dataframe([{"Стадия": t.stage, "Агент": t.title, "Режим": t.mode, "Статус": t.status, "Время, с": t.duration_s,
                   "Результат": t.produced, "Токены": t.tokens, "Примечание": t.note} for t in report.trace],
                 width="stretch", hide_index=True)

    v = report.verifier
    st.markdown(f"**Verifier (Critic):** на входе {v.input_findings} находок → объединено дублей {v.merged_duplicates}, "
                f"отсечено неподтверждённых {v.rejected_hallucinations}, понижено {v.downgraded} → **{v.output_findings}** в отчёте.")
    for attr, title in (("rejected", "Отсечено (галлюцинации / неподтверждённые цитаты)"),
                        ("downgraded_items", "Понижено (тема упоминается в ТЗ)"), ("downgrades", "Понижено (тема упоминается в ТЗ)"),
                        ("merges", "Объединено (одна проблема от разных агентов)"),
                        ("restored", "Восстановлено после ревизии (critic loop)")):
        items = _g(v, attr)
        if items:
            with st.expander(f"{title}: {len(items)}"):
                st.dataframe([i if isinstance(i, dict) else (i.model_dump() if hasattr(i, "model_dump") else {"": str(i)})
                              for i in items], hide_index=True, width="stretch")
    st.caption("Каждый агент работает в двух режимах: LLM (узкий промпт + JSON-схема) и детерминированные правила. "
               "При сбое провайдера агент перебирает резервные модели, затем откатывается на правила (fallback).")

# ── Экспорт ───────────────────────────────────────────────────────────────────
with tabs[5]:
    import report_docx

    def _agenda_md(r: Report) -> str:
        def block(items) -> list[str]:
            out: list[str] = []
            for q in items:
                out.append(f"**{q.id} [{q.priority.value}]** {q.question}")
                if q.context:
                    out.append(f"> {q.context}")
                out += [f"- [ ] {o}" for o in q.answer_options]
                if _g(q, "default_assumption"):
                    out.append(f"_По умолчанию:_ {_g(q, 'default_assumption')}")
                out.append("Ответ: ______________________\n")
            return out

        agenda: list[str] = []
        by_addr: dict[str, list] = {}
        for q in r.questions:
            by_addr.setdefault(q.addressee, []).append(q)
        for addr, items in by_addr.items():
            agenda.append(f"## {addr} — {len(items)} вопр.\n")
            agenda += block(items)
        extra = list(_g(r, "extra_questions", []) or [])
        if extra:
            agenda.append(f"## Дополнительные вопросы (вне лимита встречи) — {len(extra)} вопр.\n")
            agenda += block(sorted(extra, key=lambda x: x.priority.value))
        return f"# Повестка: уточнение ТЗ «{r.document.title}»\n\n" + "\n".join(agenda)

    # Экспорт строится один раз на анализ (run_id), а не на каждый клик по ответам в повестке
    if ss.get("exports_for") != ss.run_id or "exports" not in ss:
        exports: dict[str, Any] = {"md": to_markdown(report), "json": to_json(report), "agenda": _agenda_md(report),
                                   "docx": None, "docx_error": ""}
        try:
            exports["docx"] = report_docx.to_docx(report)
        except Exception as e:  # Word-экспорт не должен ронять страницу
            exports["docx_error"] = str(e)
        ss.exports, ss.exports_for = exports, ss.run_id
    exports = ss.exports

    stem = Path(report.document_name).stem
    e0, e1, e2, e3 = st.columns(4)
    if exports["docx"] is not None:
        e0.download_button("📄 Отчёт (Word)", exports["docx"], file_name=f"{stem}_review.docx", width="stretch",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary")
    else:
        e0.warning(f"Word-отчёт не сформирован: {exports['docx_error']}")
    e1.download_button("📝 Отчёт (Markdown)", exports["md"], file_name=f"{stem}_review.md", width="stretch")
    e2.download_button("🧾 Отчёт (JSON)", exports["json"], file_name=f"{stem}_review.json", width="stretch")
    e3.download_button("🗓 Повестка встречи (.md)", exports["agenda"], file_name=f"{stem}_agenda.md", width="stretch")
    with st.expander("Предпросмотр отчёта"):
        st.markdown(exports["md"])
