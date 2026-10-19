"""Session-state helpers. One source of truth (`threshold`) drives two synchronised sliders."""
import streamlit as st

from src import config as C


def init_state() -> None:
    st.session_state.setdefault("threshold", C.DEFAULT_THRESHOLD)


def set_threshold(value: float) -> None:
    """Callback-safe setter (call from on_click / on_change)."""
    v = float(min(max(value, 0.0), 1.0))
    st.session_state["threshold"] = v
    st.session_state["thr_sidebar"] = v
    st.session_state["thr_page"] = v


def _from_widget(key: str) -> None:
    set_threshold(st.session_state[key])


def threshold_slider(key: str, label: str = "Classification threshold") -> float:
    """Render a slider bound to the shared threshold. `key` is 'thr_sidebar' or 'thr_page'."""
    st.session_state[key] = st.session_state["threshold"]
    st.slider(label, 0.0, 1.0, step=0.01, key=key, on_change=_from_widget, args=(key,),
              help="Fraud is predicted when P(fraud) is above this value.")
    return st.session_state["threshold"]
