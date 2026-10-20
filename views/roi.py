"""Business ROI simulator: price false positives and false negatives."""
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.data import load_eval_set
from src.metrics import cost_curve, fn_cost_vector, total_cost
from src.models import score_eval_set
from src.state import set_threshold
from src.styling import compact, plot, section, show_kpi, style_fig

FN_MODES = ["Fixed $ per missed fraud", "Order value × multiplier + fee"]




def render(ctx: dict) -> None:
    t, mk, ek = ctx["threshold"], ctx["model_key"], ctx["eval_key"]
    df = load_eval_set(ek); y = df[C.TARGET].to_numpy(); p = score_eval_set(mk, ek)

    section("Business ROI simulator", "Put a dollar value on each kind of mistake. Total impact is computed "
            "directly from the confusion matrix at the current threshold.")
    with st.expander("💲 Cost assumptions", expanded=True):
        a, b, c = st.columns(3)
        c_fp = a.number_input("Cost of a FALSE POSITIVE ($)", 0.0, 1000.0, 8.0, 0.5, key="roi_fp",
                              help="Good customer wrongly blocked: lost margin, support call, churn risk.")
        mode = b.radio("Cost of a FALSE NEGATIVE", FN_MODES, key="roi_mode")
        c_rev = c.number_input("Manual review cost per flagged order ($)", 0.0, 100.0, 0.0, 0.25, key="roi_rev",
                               help="Optional analyst time for every flagged order (TP + FP).")
        fixed = mult = fee = 0.0; fx = 0.2723
        if mode == FN_MODES[0]:
            fixed = b.number_input("$ per missed fraud", 0.0, 5000.0, 75.0, 5.0, key="roi_fixed")
        else:
            d1, d2, d3 = st.columns(3)
            mult = d1.number_input("Loss multiplier (× order value)", 0.0, 5.0, 1.0, 0.1, key="roi_mult",
                                   help="1.0 = order value lost; 1.5 adds goods/delivery cost etc.")
            fee = d2.number_input("Chargeback / penalty fee ($)", 0.0, 500.0, 15.0, 1.0, key="roi_fee")
            fx = d3.number_input("AED → USD rate", 0.05, 1.0, 0.2723, 0.001, format="%.4f", key="roi_fx")
        e1, e2 = st.columns(2)
        daily = e1.number_input("Assumed orders per day", 1000, 5_000_000, 100_000, 5000, key="roi_daily",
                                help="Scales the evaluation-set result to a production volume.")
        horizon = e2.selectbox("Report impact per", ["day", "month (30d)", "year (365d)"], index=2, key="roi_h")

    fnc = fn_cost_vector(df.loc[df[C.TARGET] == 1, "order_amount"].to_numpy(), mode, fixed, mult, fee, fx)
    r = total_cost(y, p, t, c_fp, fnc, c_rev)
    days = {"day": 1, "month (30d)": 30, "year (365d)": 365}[horizon]
    scale = daily * days / len(y)
    money = lambda v: compact(v * scale, "$")          # $5.89M / $554.8K / $8,420

    curve = cost_curve(mk, ek, c_fp, mode, fixed, mult, fee, fx, c_rev)
    best = curve.loc[curve["total"].idxmin()]

    st.write("")
    k = st.columns(4)
    show_kpi(k[0], f"Total cost / {horizon.split()[0]}", money(r["total"]),
             f"vs {money(r['baseline_no_model'])} with no model", "red")
    show_kpi(k[1], "Net benefit of the model", money(r["net_benefit"]),
             f"{r['net_benefit']/r['baseline_no_model']*100:.0f}% of fraud loss avoided" if r["baseline_no_model"] else "", "green")
    show_kpi(k[2], "Missed-fraud loss (FN)", money(r["missed_loss"]), f"{r['fn']} frauds slipped through")
    show_kpi(k[3], "Friction cost (FP)", money(r["fp_cost"] + r["review_cost"]),
             f"{r['fp']} good orders blocked" + (" + review" if c_rev else ""), "cyan")

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        comp = [("No model (miss all fraud)", r["baseline_no_model"], C.MUTED),
                ("Fraud loss avoided", -r["prevented"], C.SAFE_GREEN),
                ("False-positive cost", r["fp_cost"], C.CYAN)]
        if c_rev:
            comp.append(("Review cost", r["review_cost"], C.PURPLE))
        fig = go.Figure(go.Waterfall(
            x=[c[0] for c in comp] + ["Total with model"], measure=["absolute"] + ["relative"] * (len(comp) - 1) + ["total"],
            y=[c[1] * scale for c in comp] + [0],
            text=[money(c[1]) if i != 1 else "-" + money(r["prevented"]) for i, c in enumerate(comp)] + [money(r["total"])],
            increasing=dict(marker=dict(color=C.CYAN)), decreasing=dict(marker=dict(color=C.SAFE_GREEN)),
            totals=dict(marker=dict(color=C.NOON_YELLOW)), connector=dict(line=dict(color=C.GRID))))
        fig.update_yaxes(title=f"USD per {horizon.split()[0]}", tickprefix="$", tickformat="~s")
        plot(style_fig(fig, 400, "Where the money goes"))
    with right:
        fig = go.Figure()
        fig.add_scatter(x=curve["threshold"], y=curve["total"] * scale, name="Total cost", line=dict(color=C.NOON_YELLOW, width=3))
        fig.add_scatter(x=curve["threshold"], y=curve["missed_loss"] * scale, name="Missed fraud", line=dict(color=C.FRAUD_RED, dash="dot"))
        fig.add_scatter(x=curve["threshold"], y=(curve["fp_cost"] + curve["review_cost"]) * scale, name="Friction (FP + review)", line=dict(color=C.CYAN, dash="dot"))
        fig.add_hline(y=r["baseline_no_model"] * scale, line_dash="dash", line_color=C.MUTED, annotation_text="no model")
        fig.add_vline(x=t, line_dash="dash", line_color="white", annotation_text=f"current {t:.2f}")
        fig.add_scatter(x=[best["threshold"]], y=[best["total"] * scale], mode="markers", name="Cost-optimal",
                        marker=dict(size=14, color=C.SAFE_GREEN, symbol="star", line=dict(color="white", width=1)))
        fig.update_xaxes(title="Threshold"); fig.update_yaxes(title=f"USD per {horizon.split()[0]}", tickprefix="$", tickformat="~s")
        plot(style_fig(fig, 400, "Cost vs threshold"))

    a, b = st.columns([2, 3], gap="large")
    with a:
        st.metric("Cost-optimal threshold", f"{best['threshold']:.2f}",
                  delta=f"saves {money(r['total'] - best['total'])} vs current" if r["total"] > best["total"] + 1e-9 else "current is optimal",
                  delta_color="normal")
        st.button("Apply optimal threshold", type="primary", on_click=set_threshold, args=(best["threshold"],),
                  disabled=abs(best["threshold"] - t) < 0.005)
    with b:
        with st.expander("Confusion-matrix cost table", expanded=True):
            fn_avg = fnc.mean()
            st.dataframe([
                {"Outcome": "True positive (fraud blocked)", "Count": r["tp"], "Unit value": f"saves ${fn_avg:,.0f} avg", "Impact": money(r["prevented"])},
                {"Outcome": "False positive (good order blocked)", "Count": r["fp"], "Unit value": f"-${c_fp:,.2f}", "Impact": "-" + money(r["fp_cost"])},
                {"Outcome": "False negative (fraud missed)", "Count": r["fn"], "Unit value": f"-${fn_avg:,.0f} avg", "Impact": "-" + money(r["missed_loss"])},
                {"Outcome": "True negative (good order passed)", "Count": r["tn"], "Unit value": "$0", "Impact": "$0"},
            ], hide_index=True, width="stretch")
        with st.expander("Method"):
            st.markdown(
                "`total = FN × cost_FN + FP × cost_FP + (TP + FP) × review_cost`; baseline = every fraud missed. "
                "Results are scaled by `orders/day × days ÷ evaluation rows`, keeping the evaluation set's fraud rate "
                f"({y.mean()*100:.2f} %). Impact is *very* sensitive to the cost ratio — vary it.")
