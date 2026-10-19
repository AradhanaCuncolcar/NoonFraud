"""GAN Lab: sliders that perturb the CTGAN output to simulate generation-time noise / variance."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.data import fraud_source
from src.eda import combined, color_map, hist_fig
from src.gan_sim import (DEFAULTS, fidelity_table, lie_detector, retrain_baselines, retrain_test, rules_ok, simulate)
from src.models import predict_proba_df
from src.styling import plot, section, show_kpi, style_fig


def _reset() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[f"gan_{k}"] = v


def render(ctx: dict) -> None:
    ss = st.session_state
    for k, v in DEFAULTS.items():
        ss.setdefault(f"gan_{k}", v)
    section("GAN lab", "Turn the generator's knobs and watch the synthetic fraud data drift away from reality. "
            "The simulator perturbs the rows my CTGAN actually generated (see 'How it works').")

    left, right = st.columns([1, 3], gap="large")
    with left:
        with st.expander("🎛️ Generator controls", expanded=True):
            noise = st.slider("Latent noise σ", 0.0, 2.0, step=0.05, key="gan_noise",
                              help="Independent Gaussian jitter on every continuous column (in std units).")
            var = st.slider("Variance scale (temperature)", 0.2, 3.0, step=0.1, key="gan_var_scale",
                            help="<1 squeezes toward the mean (mode collapse), >1 stretches the distribution.")
            flip = st.slider("Categorical noise", 0.0, 0.5, step=0.01, key="gan_flip",
                             help="Probability of flipping each binary flag / re-drawing payment method.")
            n = st.slider("Synthetic rows", 200, 15000, step=200, key="gan_n_rows")
            enforce = st.toggle("Reject rows that break business rules", key="gan_enforce",
                                help="R1: refunds ≤ account age ÷ 3 · R2: young accounts can't have huge order counts.")
            seed = st.number_input("Random seed", 0, 9999, key="gan_seed")
            st.button("↺ Reset to defaults", on_click=_reset)
        with st.expander("ℹ️ How it works"):
            st.markdown(
                "- Base = **15,000 rows my CTGAN generated**.\n"
                "- *Noise* adds independent jitter → breaks correlations, fattens tails.\n"
                "- *Variance* stretches each column around its mean → keeps correlations.\n"
                "- *Categorical noise* randomises flags/payment.\n"
                "- Rule rejection mimics the notebook's filter.\n"
                "- *Realism score* = 100 × (1 − mean distance to the reference over all 10 features; "
                "KS for numbers, flag-rate gap for flags, total-variation for payment).\n\n"
                "This is a **post-generation simulator**: re-sampling CTGAN's latent noise itself needs PyTorch + CTGAN, "
                "which this lightweight dashboard avoids.")
    params = dict(n_rows=int(n), noise=float(noise), var_scale=float(var), flip=float(flip), enforce=bool(enforce), seed=int(seed))
    ss["gan_params"] = params               # shared with the EDA and Rule-mining pages
    sim, info = simulate(**params)
    base, _ = simulate(**DEFAULTS)

    with right:
        ref_key = st.radio("Compare against", ["real", "noon_gan"], horizontal=True, key="gan_ref",
                           format_func=lambda k: {"real": "Real fraud (65 orders)", "noon_gan": "Noon GAN fraud (proxy for real)"}[k])
        fid = fidelity_table(sim, ref_key); fid0 = fidelity_table(base, ref_key)
        score, score0 = 1 - fid["distance"].mean(), 1 - fid0["distance"].mean()
        det = predict_proba_df(ctx["model_key"], sim) > ctx["threshold"]
        k = st.columns(4)
        d = score - score0
        show_kpi(k[0], "Realism score", f"{score*100:.1f}", f"{'▲' if d > 0 else '▼'} {abs(d)*100:.1f} vs defaults · 100 = identical" if abs(d) > 1e-4 else "defaults", "green" if d >= -0.005 else "red")
        show_kpi(k[1], "Business-rule violations", f"{info['violation_rate']*100:.1f}%",
                 "rejected before output" if enforce else f"{(~rules_ok(sim)).mean()*100:.1f}% remain in output", "red" if info["violation_rate"] > .05 else "")
        show_kpi(k[2], "Caught by deployed model", f"{det.mean()*100:.1f}%", f"{ctx['model_key']} at threshold {ctx['threshold']:.2f}", "cyan")
        show_kpi(k[3], "Lie-detector accuracy", f"{lie_detector(sim)*100:.0f}%", "vs Noon GAN · 50% = indistinguishable")

        st.write("")
        a, b = st.columns(2)
        with a:
            ref = fraud_source(ref_key)
            feat = st.selectbox("Feature", C.NUMERIC, key="gan_feat", format_func=lambda f: C.FEATURE_LABELS[f])
            cm_ = {"Simulation": C.FRAUD_RED, "Defaults (unperturbed GAN)": C.PINK,
                   C.FRAUD_SOURCES[ref_key]["label"]: C.FRAUD_SOURCES[ref_key]["color"]}
            parts = []
            for nm, df_ in [("Simulation", sim), ("Defaults (unperturbed GAN)", base), (C.FRAUD_SOURCES[ref_key]["label"], ref)]:
                x = df_.sample(min(len(df_), 2000), random_state=0).copy(); x["source"] = nm; parts.append(x)
            plot(hist_fig(pd.concat(parts), feat, cm_, 40, feat in ("order_amount", "account_age_days", "delivery_distance_km"), True, False))
        with b:
            f2 = fid.merge(fid0[["feature", "distance"]], on="feature", suffixes=("", "_default"))
            fig = go.Figure()
            fig.add_bar(y=f2["label"], x=f2["distance_default"], name="Defaults", orientation="h", marker_color=C.PINK)
            fig.add_bar(y=f2["label"], x=f2["distance"], name="Current settings", orientation="h", marker_color=C.FRAUD_RED)
            fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"))
            fig.update_xaxes(title="Distance to reference (lower = closer)")
            plot(style_fig(fig, 400, "Per-feature fidelity"))

    with st.expander("🧊 3-D view of the simulation vs real fraud", expanded=False):
        from src.eda import scatter3d_fig
        sub = combined(("real", "my_gan", "sim"), tuple(params.values()), 600, 0)
        plot(scatter3d_fig(sub, color_map(("real", "my_gan", "sim")), "order_amount", "account_age_days", "hour_of_day",
                           3, .7, ("order_amount", "account_age_days")))

    section("Downstream effect", "Does noisier synthetic fraud still teach the classifier? Retrain a random forest on "
            "my 159,200 legit seed orders + 800 seed fraud + this simulation, then score it on Noon's real referee set.")
    with st.expander("🧪 Retrain & test (takes ~15–30 s)", expanded=True):
        c1, c2 = st.columns([1, 3])
        run = c1.button("Run experiment", type="primary", key="gan_run")
        c1.caption("Referee set = test only. 100 trees, threshold 0.5.")
        hist = ss.setdefault("gan_hist", [])
        if run:
            res = retrain_test(**params)
            label = f"σ={noise:.2f} · ×{var:.1f} · flip {flip:.0%} · {n:,} rows" + ("" if enforce else " · no rules")
            hist[:] = [h for h in hist if h["Run"] != label] + [dict(Run=label, **res)]
        base_tbl = retrain_baselines().reset_index().rename(columns={"index": "Run"})
        tbl = pd.concat([base_tbl, pd.DataFrame(hist)], ignore_index=True)
        if len(tbl):
            tbl["Run"] = tbl["Run"].astype(str)
            long = tbl.melt(id_vars="Run", value_vars=["recall", "precision", "f1"], var_name="Metric", value_name="Score")
            fig = px.bar(long, x="Run", y="Score", color="Metric", barmode="group",
                         color_discrete_sequence=[C.NOON_YELLOW, C.CYAN, C.PINK])
            fig.update_layout(xaxis_tickangle=-20, yaxis_range=[0, 1])
            with c2:
                plot(style_fig(fig, 340, "Real-data performance by generator setting"))
            st.dataframe(tbl.rename(columns={"fp": "False alarms", "tp": "Frauds caught"}).round(3), hide_index=True, width="stretch")
            if hist and st.button("Clear experiment log", key="gan_clear"):
                ss["gan_hist"] = []; st.rerun()
