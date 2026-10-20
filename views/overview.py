"""Overview page: headline numbers + the notebook's model scorecard."""
import plotly.express as px
import streamlit as st

from src import config as C
from src.data import load_benchmark, load_eval_set
from src.metrics import confusion, derive, global_scores
from src.models import score_eval_set
from src.styling import plot, section, show_kpi, style_fig

NAMES = {"S0": "S0 · seed only", "S2": "S2 · seed + my GAN", "N1": "N1 · my legit + Noon GAN (final)",
         "N2": "N2 · + duplicates", "N3": "N3 · my fraud + Noon GAN", "N3b": "N3b · balanced weights",
         "Noon B": "Noon B · real + Noon GAN"}


def render(ctx: dict) -> None:
    df = load_eval_set(ctx["eval_key"]); y = df[C.TARGET].to_numpy()
    p = score_eval_set(ctx["model_key"], ctx["eval_key"])
    d = derive(confusion(y, p, ctx["threshold"]))
    g = global_scores(ctx["model_key"], ctx["eval_key"])

    section("Fraud-detection command centre", "Live view of the selected model on the selected evaluation set. "
            "Change model, dataset or threshold in the sidebar — every page updates.")
    c = st.columns(5)
    show_kpi(c[0], "Orders evaluated", f"{len(df):,}", f"{int(y.sum())} fraud · {y.mean()*100:.2f}% base rate")
    show_kpi(c[1], "Fraud caught (recall)", f"{d['recall']*100:.1f}%", f"{d['tp']} of {int(y.sum())} frauds", "green")
    show_kpi(c[2], "Precision", f"{d['precision']*100:.1f}%", f"{d['flagged']} orders flagged", "cyan")
    show_kpi(c[3], "False alarms", f"{d['fp']:,}", f"{d['fpr']*100:.3f}% of legit orders", "red")
    show_kpi(c[4], "ROC-AUC", f"{g['roc_auc']:.3f}", f"PR-AUC {g['pr_auc']:.3f}")

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        section("Model scorecard on Noon's real referee set", "Numbers from the lab notebook at the default 0.5 cut-off.")
        b = load_benchmark()
        b["Model"] = b["model"].map(NAMES).fillna(b["model"])
        long = b.melt(id_vars="Model", value_vars=["recall", "precision", "f1"], var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=[C.NOON_YELLOW, C.CYAN, C.PINK])
        fig.update_layout(xaxis_tickangle=-25, yaxis_range=[0, 1])
        plot(style_fig(fig, 400))
    with right:
        section("The story in three lines")
        st.markdown(
            "- 🕵️ **Fraud is rare** (~0.3 % of real orders), so accuracy lies — recall and precision matter.\n"
            "- 🧬 **A GAN** generates extra fraud examples; the classifier learns *what fraud looks like*.\n"
            "- 🏁 Noon's GAN learned from **real** fraud, mine from **assumptions** — that is the gap "
            "(recall 0.82 vs 0.40).")
        with st.expander("What can I do in this dashboard?", expanded=True):
            st.markdown(
                "1. **GAN pipeline** – the 7 assignment steps: my GAN vs Noon's GAN, feature by feature, through to the classifiers.\n"
                "2. **Threshold lab** – slide the cut-off, watch precision / recall / F1 / ROC move.\n"
                "3. **ROI simulator** – put dollar costs on false alarms and missed fraud.\n"
                "4. **Explainer** – SHAP / LIME breakdown of any single transaction.\n"
                "5. **Rule mining** – Apriori / FP-Growth patterns hiding in fraud.\n"
                "6. **GAN vs real EDA** – interactive histograms and 3-D scatter.\n"
                "7. **GAN lab** – turn noise / variance knobs and see the synthetic data change.")
        with st.expander("Data & integrity notes"):
            st.markdown(
                "- The referee set is **test-only**: no model in this dashboard is trained or tuned on it.\n"
                "- It has only **65 fraud rows**, so recall carries roughly ±10–12 points (95 %).\n"
                "- Rule-mining and EDA *describe* the referee fraud; they never fit a model to it.")
