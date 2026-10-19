"""Fraud rule-mining overlay (Apriori / FP-Growth)."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.gan_sim import DEFAULTS, simulate
from src.rules import mine_patterns, mine_rules, pattern_support_across_sources
from src.styling import plot, section, show_kpi, style_fig

SRC = {"real": "Real fraud (Noon referee · 65 orders)", "noon_gan": "Noon GAN fraud (1,033 distinct)",
       "my_gan": "My GAN fraud (5,000 rows)"}


def render(ctx: dict) -> None:
    section("Fraud rule-mining overlay", "Which combinations of behaviours keep showing up in fraudulent orders? "
            "Patterns are mined *inside the fraud rows* and then checked against real legit orders.")
    c = st.columns([2, 1, 1, 1])
    source = c[0].selectbox("Mine patterns from", list(SRC) + ["sim"], key="ru_src",
                            format_func=lambda k: SRC.get(k, "GAN Lab simulation (current settings)"))
    algo = c[1].radio("Algorithm", ["FP-Growth", "Apriori"], key="ru_algo", horizontal=True)
    min_sup = c[2].slider("Min support (in fraud)", 0.05, 0.60, 0.15, 0.01, key="ru_sup")
    max_len = c[3].slider("Max pattern size", 1, 5, 3, key="ru_len")

    sim_params = tuple(st.session_state.get("gan_params", DEFAULTS).values()) if source == "sim" else None
    sim_df = simulate(*sim_params)[0] if source == "sim" else None
    pats = mine_patterns(source, algo, min_sup, max_len, sim_params, sim_df)
    if pats.empty:
        st.warning("No pattern reaches this support — lower the minimum support.")
        return

    k = st.columns(4)
    best = pats[pats["size"] >= 2].head(1)
    show_kpi(k[0], "Patterns found", f"{len(pats):,}", f"{algo} · support ≥ {min_sup:.0%}")
    if len(best):
        b = best.iloc[0]
        show_kpi(k[1], "Strongest 2+ pattern", f"{b['enrichment']:.0f}×",
                 b["pattern"], "red")
        show_kpi(k[2], "…its precision on real data", f"{b['real_precision']*100:.1f}%",
                 f"{b['real_fraud_hits']} fraud vs {b['real_legit_hits']} legit orders match", "cyan")
        show_kpi(k[3], "…its recall on real data", f"{b['real_recall']*100:.1f}%", "share of real fraud it covers", "green")

    st.write("")
    a, b = st.columns([3, 2], gap="large")
    with a:
        min_size = st.slider("Only show patterns with at least N items", 1, max_len, min(2, max_len), key="ru_min")
        top = st.slider("Top-N patterns", 5, 30, 12, key="ru_top")
        view = pats[pats["size"] >= min_size].head(top).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=view["enrichment"],
            y=view["pattern"], orientation="h", marker=dict(color=view["support"], colorscale=[[0, C.CYAN], [1, C.NOON_YELLOW]],
                                                            colorbar=dict(title="support")),
            text=[f"{e:.0f}×" for e in view["enrichment"]], textposition="outside",
            customdata=view[["support", "legit_support", "real_precision"]],
            hovertemplate="%{y}<br>support in fraud %{customdata[0]:.1%}<br>support in real legit %{customdata[1]:.2%}"
                          "<br>real precision %{customdata[2]:.1%}<extra></extra>"))
        fig.update_xaxes(title="Enrichment = P(pattern | fraud) ÷ P(pattern | real legit), smoothed", type="log")
        plot(style_fig(fig, 120 + 34 * len(view), "Top fraud patterns (log scale)"))
    with b:
        with st.expander("What am I looking at?", expanded=True):
            st.markdown(
                "- **Support** – share of fraud orders containing the pattern.\n"
                "- **Enrichment** – how many times more common it is in fraud than in real legit orders (smoothed by +1 legit order so rare patterns are not inflated).\n"
                "- **Real precision / recall** – what the pattern would score *as a stand-alone rule* "
                "on the real referee set.\n"
                + ("- ⚠️ Source = real fraud, so these are **in-sample descriptions**, not a validated rule.\n"
                   if source == "real" else
                   "- Source is synthetic: real precision/recall tells you whether the **GAN's patterns hold in real life**.\n"))
        pat_list = tuple(tuple(i) for i in pats[pats["size"] >= min_size].head(6)["items"])
        if pat_list and source != "sim":
            cmp = pattern_support_across_sources(pat_list)
            fig = px.bar(cmp, x="support", y="pattern", color="source", barmode="group", orientation="h",
                         color_discrete_sequence=[C.NOON_YELLOW, C.CYAN, C.PINK], labels={"support": "support in fraud", "pattern": ""})
            fig.update_layout(yaxis=dict(autorange="reversed"))
            plot(style_fig(fig, 420, "Do the GANs reproduce these patterns?"))

    t1, t2 = st.tabs(["📋 Pattern table", "🔗 Association rules (A → B)"])
    with t1:
        st.dataframe(
            pats[["pattern", "size", "support", "legit_support", "enrichment", "real_fraud_hits", "real_legit_hits",
                  "real_precision", "real_recall"]], hide_index=True, width="stretch", height=380,
            column_config={
                "support": st.column_config.ProgressColumn("Support (fraud)", format="%.1f%%", min_value=0, max_value=1),
                "legit_support": st.column_config.NumberColumn("Support (real legit)", format="%.2f%%"),
                "enrichment": st.column_config.NumberColumn("Enrichment ×", format="%.1f"),
                "real_precision": st.column_config.NumberColumn("Real precision", format="%.1f%%"),
                "real_recall": st.column_config.NumberColumn("Real recall", format="%.1f%%")})
    with t2:
        d1, d2 = st.columns(2)
        min_conf = d1.slider("Min confidence", 0.3, 1.0, 0.6, 0.05, key="ru_conf")
        rules = mine_rules(source, algo, min_sup, min_conf, max_len, sim_params, sim_df)
        if rules.empty:
            st.info("No rules at these thresholds — lower min support or min confidence.")
        else:
            d2.caption(f"{len(rules)} rules · confidence = P(B | A) *within fraud*")
            fig = px.scatter(rules.head(200), x="support", y="confidence", color="lift", size="lift", hover_name="rule",
                             color_continuous_scale=[[0, C.CYAN], [1, C.NOON_YELLOW]])
            plot(style_fig(fig, 340, "Rule map: support × confidence (colour = lift)"))
            st.dataframe(rules.head(60), hide_index=True, width="stretch", height=300)
