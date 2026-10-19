"""Noon Fraud Intelligence Console — Streamlit dashboard.

Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Noon · Fraud Intelligence Console", page_icon="🟡", layout="wide",
                   initial_sidebar_state="expanded")

from src import config as C                                    # noqa: E402  (after set_page_config)
from src.state import init_state, threshold_slider, set_threshold  # noqa: E402
from src.styling import hero, inject_css                       # noqa: E402
from views import eda, explainer, gan_lab, overview, roi, rules, threshold  # noqa: E402

PAGES = {
    "🏠 Overview": overview,
    "🎚️ Threshold lab": threshold,
    "💰 ROI simulator": roi,
    "🔍 Explainer": explainer,
    "⛏️ Rule mining": rules,
    "📈 GAN vs real EDA": eda,
    "🧬 GAN lab": gan_lab,
}

# Keep widget values of pages that are not currently rendered (Streamlit drops them otherwise).
# (button keys end in _btn / _run / _clear and must NOT be re-assigned: they are read-only.)
for _k in list(st.session_state.keys()):
    if not str(_k).endswith(("_btn", "_run", "_clear")):
        st.session_state[_k] = st.session_state[_k]

inject_css()
init_state()


def sidebar() -> dict:
    with st.sidebar:
        st.markdown('<div class="sb-brand">noon</div><div class="sb-sub">Fraud Intelligence Console</div>', unsafe_allow_html=True)
        model_key = st.selectbox("Model", list(C.MODELS), format_func=lambda k: C.MODELS[k]["label"], key="model_key")
        eval_key = st.selectbox("Evaluation set", list(C.EVAL_SETS), format_func=lambda k: C.EVAL_SETS[k]["label"], key="eval_key")
        st.divider()
        t = threshold_slider("thr_sidebar", "Fraud threshold")
        st.button("Reset to 0.50", on_click=set_threshold, args=(C.DEFAULT_THRESHOLD,), width="stretch")
        st.caption("Orders with P(fraud) above the threshold are flagged.")
        st.divider()
        st.caption("Real referee set = test only. Never used for training.")
    return dict(model_key=model_key, eval_key=eval_key, threshold=t)


ctx = sidebar()
hero("Fraud Intelligence Console", "GAN-augmented fraud detection · UAE food delivery")

st.session_state.setdefault("nav", "🏠 Overview")
page = st.segmented_control("Page", list(PAGES), key="nav", label_visibility="collapsed")
PAGES[page or "🏠 Overview"].render(ctx)
