"""Real-time explainer: SHAP / LIME breakdown for a single transaction."""
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.data import load_eval_set
from src.explain import (HAS_LIME, HAS_SHAP, fmt_value, gauge_fig, global_fig, lime_fig, lime_single,
                         shap_global, shap_single, waterfall_fig)
from src.models import predict_one, score_eval_set
from src.styling import plot, section

_KEYS = list(C.PRESETS["Ordinary lunch order"].keys())


_BOUNDS = {"order_amount": (10.0, 2000.0), "account_age_days": (0, 2000), "hour_of_day": (0, 23),
           "num_orders_last_7d": (0, 30), "delivery_distance_km": (0.0, 40.0), "prev_refunds": (0, 10)}


def _apply(tx: dict) -> None:
    """Write a transaction into the widget state (values clamped to the slider ranges)."""
    for k, v in tx.items():
        if k in _BOUNDS:
            lo, hi = _BOUNDS[k]
            v = min(max(v, lo), hi)
            v = float(v) if isinstance(lo, float) else int(v)
        elif k in C.BINARY:
            v = bool(v)
        st.session_state[f"ex_{k}"] = v


def _preset_cb() -> None:
    _apply(C.PRESETS[st.session_state["ex_preset"]])


def _pick_cb(rows: pd.DataFrame) -> None:
    if len(rows):
        i = np.random.randint(len(rows))
        _apply({k: (rows.iloc[i][k].item() if hasattr(rows.iloc[i][k], "item") else rows.iloc[i][k]) for k in _KEYS})


def _input_panel() -> dict:
    ss = st.session_state
    if "ex_order_amount" not in ss:
        _apply(C.PRESETS["Ordinary lunch order"])
    a, b = st.columns(2)
    with a:
        order_amount = st.slider("Order amount (AED)", 10.0, 2000.0, step=5.0, key="ex_order_amount")
        account_age = st.slider("Account age (days)", 0, 2000, key="ex_account_age_days")
        hour = st.slider("Hour of day", 0, 23, key="ex_hour_of_day")
        n7 = st.slider("Orders in last 7 days", 0, 30, key="ex_num_orders_last_7d")
        dist = st.slider("Delivery distance (km)", 0.0, 40.0, step=0.1, key="ex_delivery_distance_km")
    with b:
        pay = st.selectbox("Payment method", C.PAYMENTS, key="ex_payment_method")
        refunds = st.slider("Previous refunds", 0, 10, key="ex_prev_refunds")
        disc = int(st.toggle("Discount code used", key="ex_discount_applied"))
        dev = int(st.toggle("New / changed device", key="ex_device_change"))
        loc = int(st.toggle("Billing ≠ delivery location", key="ex_location_mismatch"))
    return dict(order_amount=float(order_amount), account_age_days=int(account_age), hour_of_day=int(hour),
                num_orders_last_7d=int(n7), delivery_distance_km=float(dist), payment_method=pay,
                discount_applied=disc, prev_refunds=int(refunds), device_change=dev, location_mismatch=loc)


def _reasons(contribs: dict, tx: dict, k: int = 4) -> None:
    for f, v in sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]:
        cls, word = ("f", "raises") if v > 0 else ("s", "lowers")
        st.markdown(f'<div class="reason"><b class="{cls}">{v*100:+.1f} pp</b> · '
                    f'{C.FEATURE_LABELS[f]} = <b>{fmt_value(f, tx[f])}</b> {word} fraud risk</div>',
                    unsafe_allow_html=True)


def render(ctx: dict) -> None:
    t, mk, ek = ctx["threshold"], ctx["model_key"], ctx["eval_key"]
    section("Real-time explainer", "Why did the model call this transaction FRAUD or LEGIT? "
            "Edit any field — the verdict and explanation recompute instantly.")
    if not (HAS_SHAP or HAS_LIME):
        st.error("Install `shap` and/or `lime` (see requirements.txt) to enable explanations.")
        return

    src = st.radio("Transaction source", ["✍️ Build my own", "📂 Pick from the evaluation set"], horizontal=True, key="ex_src")
    left, right = st.columns([5, 7], gap="large")
    with left:
        with st.expander("Transaction details", expanded=True):
            if src.startswith("✍️"):
                st.selectbox("Load a preset", list(C.PRESETS), key="ex_preset", on_change=_preset_cb, index=None,
                             placeholder="Choose a scenario…")
            else:
                df = load_eval_set(ek); p_all = score_eval_set(mk, ek)
                y = df[C.TARGET].to_numpy(); pred = p_all > t
                groups = {"🚨 Caught fraud (TP)": (pred & (y == 1)), "😬 Missed fraud (FN)": (~pred & (y == 1)),
                          "⚠️ False alarm (FP)": (pred & (y == 0)), "✅ Correct legit (TN)": (~pred & (y == 0))}
                g = st.selectbox("Show me a…", list(groups), key="ex_group")
                rows = df[groups[g]]
                st.caption(f"{len(rows):,} transactions in this group at threshold {t:.2f}")
                st.button("🎲 Load a random one", on_click=_pick_cb, args=(rows,), type="primary", disabled=len(rows) == 0)
            tx = _input_panel()

    prob = predict_one(mk, tx)
    flagged = prob > t
    with right:
        a, b = st.columns([1, 1])
        with a:
            st.markdown(f'<div class="verdict {"fraud" if flagged else "legit"}">'
                        f'{"🚨 FRAUD" if flagged else "✅ LEGIT"}<small>risk score {prob*100:.1f}% · threshold {t*100:.0f}%</small></div>',
                        unsafe_allow_html=True)
        with b:
            st.plotly_chart(gauge_fig(prob, t), width="stretch", config={"displayModeBar": False})

        methods = [m for m, ok in [("SHAP", HAS_SHAP), ("LIME", HAS_LIME)] if ok]
        method = st.radio("Explanation method", methods + (["Both"] if len(methods) == 2 else []), horizontal=True, key="ex_method")
        items = tuple(sorted(tx.items()))
        if method in ("SHAP", "Both") and HAS_SHAP:
            sh = shap_single(mk, items)
            plot(waterfall_fig(sh["base"], sh["contribs"], tx, sh["prob"], t))
            _reasons(sh["contribs"], tx)
        if method in ("LIME", "Both") and HAS_LIME:
            li = lime_single(mk, ek, items)
            plot(lime_fig(li))
            st.caption("LIME fits a simple linear model to perturbed copies of this order; it shows local direction "
                       "and relative strength, not an exact probability decomposition (SHAP is exact for trees).")

    with st.expander("🌍 Global view — which features drive the model overall? (SHAP)"):
        if not HAS_SHAP:
            st.info("Requires `shap`.")
        else:
            c1, c2, c3 = st.columns(3)
            nf = c1.slider("Fraud rows in sample", 10, 65, 40, 5, key="gx_nf")
            nl = c2.slider("Legit rows in sample", 20, 300, 100, 20, key="gx_nl")
            c3.caption("Exact SHAP on large forests is CPU-heavy (~10–60 s). Results are cached.")
            if st.button("Compute global explanation", key="gx_btn"):
                st.session_state["gx_run"] = (mk, ek, nf, nl)
            if st.session_state.get("gx_run") == (mk, ek, nf, nl):
                plot(global_fig(shap_global(mk, ek, nf, nl)))
            else:
                st.caption("Press the button to (re)compute for the current model / dataset / sample size.")
