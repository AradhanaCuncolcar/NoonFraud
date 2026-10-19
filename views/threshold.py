"""Confidence-threshold lab: precision / recall / F1 / ROC-AUC respond to the slider instantly."""
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.data import load_eval_set
from src.metrics import confusion, derive, global_scores, threshold_sweep
from src.models import score_eval_set
from src.state import threshold_slider
from src.styling import plot, section, show_kpi, style_fig


def _confusion_fig(d: dict) -> go.Figure:
    txt = [[f"<b>{d['tn']:,}</b><br>true negatives", f"<b>{d['fp']:,}</b><br>false positives"],
           [f"<b>{d['fn']:,}</b><br>false negatives", f"<b>{d['tp']:,}</b><br>true positives"]]
    fig = go.Figure(go.Heatmap(z=[[0, 1], [1, 0]], x=["Predicted LEGIT", "Predicted FRAUD"],
                               y=["Actually LEGIT", "Actually FRAUD"], text=txt, texttemplate="%{text}",
                               colorscale=[[0, "#1F1F24"], [1, "#5A5200"]], showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, 330, "Confusion matrix")


def render(ctx: dict) -> None:
    t, mk, ek = ctx["threshold"], ctx["model_key"], ctx["eval_key"]
    y = load_eval_set(ek)[C.TARGET].to_numpy(); p = score_eval_set(mk, ek)
    d = derive(confusion(y, p, t)); d0 = derive(confusion(y, p, C.DEFAULT_THRESHOLD))
    g = global_scores(mk, ek)

    section("Confidence-threshold lab", f"A transaction is flagged as FRAUD when its fraud probability is above "
            f"**{t:.2f}**. Move the slider in the sidebar (or the one below) — metrics refresh instantly.")
    threshold_slider("thr_page", "Classification threshold (synced with the sidebar)")

    def delta(new, old):
        diff = (new - old) * 100
        cls = "up" if diff > 0.05 else "down" if diff < -0.05 else ""
        return f'<span class="{cls}">{diff:+.1f} pts vs 0.50</span>' if abs(diff) >= 0.05 else "same as 0.50"

    c = st.columns(4)
    show_kpi(c[0], "Precision", f"{d['precision']*100:.1f}%", delta(d["precision"], d0["precision"]) +
             f" · {d['tp']}/{d['flagged']} flags correct", "cyan")
    show_kpi(c[1], "Recall", f"{d['recall']*100:.1f}%", delta(d["recall"], d0["recall"]) +
             f" · {d['tp']}/{d['tp']+d['fn']} frauds caught", "green")
    show_kpi(c[2], "F1-score", f"{d['f1']:.3f}", delta(d["f1"], d0["f1"]))
    show_kpi(c[3], "ROC-AUC", f"{g['roc_auc']:.3f}", "threshold-independent · dot on the curve moves")

    st.write("")
    a, b = st.columns(2, gap="large")
    with a:
        sweep = threshold_sweep(mk, ek)
        fig = go.Figure()
        for col, name, color in [("precision", "Precision", C.CYAN), ("recall", "Recall", C.SAFE_GREEN),
                                 ("f1", "F1", C.NOON_YELLOW)]:
            fig.add_scatter(x=sweep["threshold"], y=sweep[col], name=name, line=dict(color=color, width=3))
        fig.add_vline(x=t, line_dash="dash", line_color=C.FRAUD_RED, annotation_text=f"t = {t:.2f}")
        fig.update_xaxes(title="Threshold"); fig.update_yaxes(title="Score", range=[0, 1.02])
        plot(style_fig(fig, 360, "Precision–recall trade-off across thresholds"))
    with b:
        fig = go.Figure()
        fig.add_scatter(x=g["fpr"], y=g["tpr"], name=f"ROC (AUC {g['roc_auc']:.3f})", line=dict(color=C.NOON_YELLOW, width=3),
                        fill="tozeroy", fillcolor="rgba(254,238,0,.08)")
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color=C.MUTED, dash="dot"))
        fig.add_scatter(x=[d["fpr"]], y=[d["recall"]], mode="markers", name="Current threshold",
                        marker=dict(size=15, color=C.FRAUD_RED, line=dict(color="white", width=2)))
        fig.update_xaxes(title="False-positive rate"); fig.update_yaxes(title="True-positive rate (recall)")
        plot(style_fig(fig, 360, "ROC curve · operating point"))

    a, b = st.columns([2, 3], gap="large")
    with a:
        plot(_confusion_fig(d))
    with b:
        with st.expander("How to read this", expanded=True):
            st.markdown(
                f"- At **t = {t:.2f}** the model flags **{d['flagged']:,}** of {len(y):,} orders "
                f"({d['flagged']/len(y)*100:.2f} %).\n"
                f"- Lower the threshold → *more* fraud caught (recall ↑) but *more* innocent customers "
                f"blocked (precision ↓).\n"
                f"- **ROC-AUC** measures ranking quality over *all* thresholds, so it is constant by definition; "
                f"the red dot shows where the current threshold sits on the curve.\n"
                f"- With only {int(y.sum())} fraud rows, treat small differences as noise.")
        with st.expander("Best thresholds by metric"):
            sweep = threshold_sweep(mk, ek)
            rows = []
            for col, nm in [("f1", "Max F1"), ("recall", "Max recall with precision ≥ 30%"), ("precision", "Max precision with recall ≥ 30%")]:
                if nm.startswith("Max recall"):
                    s = sweep[sweep["precision"] >= .3]; r = s.loc[s["recall"].idxmax()] if len(s) else None
                elif nm.startswith("Max precision"):
                    s = sweep[sweep["recall"] >= .3]; r = s.loc[s["precision"].idxmax()] if len(s) else None
                else:
                    r = sweep.loc[sweep["f1"].idxmax()]
                if r is not None:
                    rows.append({"Objective": nm, "Threshold": round(r["threshold"], 2),
                                 "Precision": round(r["precision"], 3), "Recall": round(r["recall"], 3),
                                 "F1": round(r["f1"], 3), "False alarms": int(r["fp"])})
            st.dataframe(rows, hide_index=True, width="stretch")
