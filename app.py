"""Saqta TZ Auditor — веб-интерфейс агента проверки требований (ISO/IEC/IEEE 29148, IEEE 830).

Запуск: streamlit run app.py   (то же самое из командной строки: python validate.py <вход> <отчёт.md>)
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import app_iso
from config import GEMINI_BASE_URL, LLMConfig

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Saqta TZ Auditor", page_icon="🛡️", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 10px 12px; margin-bottom: 8px;
       background: rgba(128,128,128,.06);}
.card.run {border-color: #3b82f6; background: rgba(59,130,246,.08);}
.card.ok {border-color: #16a34a; background: rgba(22,163,74,.08);}
.card .t {font-weight: 700;} .card .r {font-size: .78rem; opacity: .75;} .card .s {font-size: .8rem; margin-top: 4px;}
blockquote {margin: .3em 0;}
</style>
""",
    unsafe_allow_html=True,
)

env_cfg = LLMConfig.from_env()
with st.sidebar:
    st.markdown("### 🛡️ Saqta TZ Auditor")
    mode = st.radio("Режим проверки", ["ИИ + правила", "Только правила (без ключа)"], key="iso_mode")
    cfg = env_cfg
    if mode == "ИИ + правила":
        if not env_cfg.enabled:
            key = st.text_input("Ключ Gemini API", type="password", key="iso_key",
                                help="Бесплатный ключ: aistudio.google.com/apikey. Ключ используется только в этом сеансе.")
            if key.strip():
                cfg = env_cfg.with_overrides(api_key=key.strip(), base_url=env_cfg.base_url or GEMINI_BASE_URL)
        with st.expander("Модель"):
            model = st.text_input("Модель", value=cfg.model, key="iso_model")
            if model.strip() and model.strip() != cfg.model:
                cfg = cfg.with_overrides(model=model.strip())
    use_llm = mode == "ИИ + правила" and cfg.enabled
    if mode == "ИИ + правила" and not cfg.enabled:
        st.warning("Ключ не задан: проверка пойдёт только на правилах.", icon="⚠️")
    elif use_llm:
        st.success(f"Модель: {cfg.model}", icon="🧠")
    st.session_state["llm_cfg"] = cfg

app_iso.render(cfg, use_llm, ROOT)
