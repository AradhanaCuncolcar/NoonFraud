"""Dynamic Plotly EDA: synthetic GAN fraud vs real data."""
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import ks_2samp

from src import config as C
from src.eda import (LOG_DEFAULT, color_map, combined, hist_fig, ks_fig, ks_vs_real, scatter3d_fig,
                     signature_fig, signature_rates)
from src.gan_sim import DEFAULTS
from src.data import fraud_source
from src.styling import plot, section, show_kpi

ALL = ["real", "noon_gan", "my_gan", "sim", "real_legit"]


def render(ctx: dict) -> None:
    section("Synthetic GAN data vs real data", "Interactive comparison — hover, zoom, rotate, toggle series in the legend.")
    sim_params = tuple(st.session_state.get("gan_params", DEFAULTS).values())

    with st.expander("⚙️ Data selection", expanded=True):
        a, b, c = st.columns([3, 1, 1])
        keys = a.multiselect("Sources to compare", ALL, default=["real", "noon_gan", "my_gan"], key="eda_src",
                             format_func=lambda k: C.FRAUD_SOURCES[k]["label"],
                             help="'GAN Lab simulation' uses the settings from the GAN Lab page.")
        per = b.slider("Max rows per source", 100, 5000, 1500, 100, key="eda_per")
        seed = c.number_input("Sample seed", 0, 999, 0, key="eda_seed")
        sim_on = "sim" in keys
        if sim_on:
            st.caption(f"GAN Lab simulation → noise {sim_params[1]}, variance ×{sim_params[2]}, flag-flip {sim_params[3]:.0%}, {sim_params[0]:,} rows.")
    if not keys:
        st.info("Pick at least one source.")
        return
    keys_t = tuple(keys)
    df = combined(keys_t, sim_params, per, int(seed))
    cmap = color_map(keys)

    k = st.columns(len(keys))
    for col, key in zip(k, keys):
        n = int((df["source"] == C.FRAUD_SOURCES[key]["label"]).sum())
        show_kpi(col, C.FRAUD_SOURCES[key]["label"], f"{n:,}", "rows shown", "cyan" if key != "real" else "")
    st.write("")

    t1, t2, t3, t4, t5 = st.tabs(["📊 Histograms", "🧊 3-D scatter", "📏 Distance to real", "🧬 Fraud signatures", "🆚 GAN-to-GAN"])
    with t1:
        a, b, c, d = st.columns([2, 1, 1, 1])
        feat = a.selectbox("Feature", C.NUMERIC + C.BINARY, key="eda_feat", format_func=lambda f: C.FEATURE_LABELS[f])
        bins = b.slider("Bins", 5, 100, 40, key="eda_bins")
        log_x = c.toggle("Log scale", value=feat in LOG_DEFAULT, key=f"eda_log_{feat}")
        norm = d.toggle("Normalise (density)", value=True, key="eda_norm")
        cum = d.toggle("Cumulative", value=False, key="eda_cum")
        plot(hist_fig(df, feat, cmap, bins, log_x and feat in C.NUMERIC, norm, cum))
    with t2:
        with st.expander("3-D controls", expanded=True):
            a, b, c, d, e = st.columns(5)
            opts = C.NUMERIC
            x = a.selectbox("X axis", opts, 0, key="eda_x", format_func=lambda f: C.FEATURE_LABELS[f])
            y = b.selectbox("Y axis", opts, 1, key="eda_y", format_func=lambda f: C.FEATURE_LABELS[f])
            z = c.selectbox("Z axis", opts, 2, key="eda_z", format_func=lambda f: C.FEATURE_LABELS[f])
            size = d.slider("Point size", 1, 8, 3, key="eda_size")
            opac = e.slider("Opacity", 0.2, 1.0, 0.7, 0.05, key="eda_opac")
            logs = st.multiselect("Log-scale axes", [x, y, z], default=[v for v in (x, y, z) if v in LOG_DEFAULT],
                                  key="eda_logs", format_func=lambda f: C.FEATURE_LABELS[f])
        sub = combined(keys_t, sim_params, min(per, 800), int(seed))
        plot(scatter3d_fig(sub, cmap, x, y, z, size, opac, tuple(logs)))
        st.caption("Tip: drag to rotate, scroll to zoom, click a legend entry to hide a source. Counts are capped at 800 per source for smooth rotation.")
    with t3:
        syn = tuple(k for k in keys if k not in ("real", "real_legit"))
        if not syn:
            st.info("Add a synthetic source (Noon GAN, My GAN or GAN Lab) to measure its distance to real fraud.")
        else:
            ks = ks_vs_real(syn, sim_params)
            plot(ks_fig(ks, cmap))
            with st.expander("What is the KS distance?"):
                st.markdown("The Kolmogorov–Smirnov statistic is the largest gap between two cumulative distributions "
                            "(0 = identical, 1 = completely disjoint). With only 65 real fraud rows, differences below ~0.1 are within noise.")
            st.dataframe(ks.pivot(index="feature", columns="source", values="KS").round(3), width="stretch")
    with t4:
        sig = signature_rates(tuple(k for k in keys if k != "real_legit") or ("real",), sim_params)
        plot(signature_fig(sig, cmap))
        if "real_legit" in keys:
            st.caption("Real legit orders are shown only in the histogram / 3-D tabs; signatures compare fraud sources.")

    with t5:
        section("Direct comparison: My GAN vs Noon GAN",
                "Compare the two synthetic fraud datasets directly, independently of the real-data comparison.")
        my_gan = fraud_source("my_gan").copy()
        noon_gan = fraud_source("noon_gan").copy()
        if my_gan.empty or noon_gan.empty:
            st.warning("One of the GAN datasets is empty; check the files in the data folder.")
            return
        st.caption(f"Rows available: My GAN = {len(my_gan):,}; Noon GAN = {len(noon_gan):,}.")
        rows = []
        for feature in C.NUMERIC:
            rows.append({"Feature": C.FEATURE_LABELS[feature], "Metric": "KS distance",
                         "Distance": float(ks_2samp(my_gan[feature].dropna(), noon_gan[feature].dropna()).statistic)})
        for feature in C.BINARY:
            rows.append({"Feature": C.FEATURE_LABELS[feature], "Metric": "Positive-rate gap",
                         "Distance": float(abs(my_gan[feature].mean() - noon_gan[feature].mean()))})
        my_pay = my_gan["payment_method"].value_counts(normalize=True)
        noon_pay = noon_gan["payment_method"].value_counts(normalize=True)
        payment_tv = 0.5 * sum(abs(my_pay.get(v, 0.0) - noon_pay.get(v, 0.0)) for v in C.PAYMENTS)
        rows.append({"Feature": C.FEATURE_LABELS["payment_method"], "Metric": "Total-variation distance",
                     "Distance": float(payment_tv)})
        compare = pd.DataFrame(rows)
        a, b, c = st.columns(3)
        show_kpi(a, "Features compared", str(len(compare)), "configured fraud features")
        show_kpi(b, "Mean distance", f"{compare['Distance'].mean():.3f}", "lower = more similar")
        largest = compare.loc[compare["Distance"].idxmax(), "Feature"]
        show_kpi(c, "Largest difference", largest, "feature with greatest distance")
        fig = px.bar(compare.sort_values("Distance", ascending=False), x="Feature", y="Distance", color="Metric",
                     labels={"Distance": "Distance (lower = more similar)", "Feature": ""},
                     title="Feature-level distance: My GAN vs Noon GAN")
        fig.update_layout(xaxis_tickangle=-30)
        plot(fig)
        st.dataframe(compare.sort_values("Distance", ascending=False).round({"Distance": 3}),
                     width="stretch", hide_index=True)
        st.caption("Numeric features use KS distance; binary features use absolute positive-rate gaps; payment method uses total-variation distance. These are descriptive comparisons, not proof that the datasets are statistically identical.")
