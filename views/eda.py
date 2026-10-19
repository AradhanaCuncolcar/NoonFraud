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
                "This comparison isolates the two synthetic fraud datasets, independent of the source-selection controls above.")
        my_gan = fraud_source("my_gan").copy()
        noon_gan = fraud_source("noon_gan").copy()
        st.caption(f"Rows available: My GAN = {len(my_gan):,}; Noon GAN = {len(noon_gan):,}. Noon GAN is deduplicated by the data loader where applicable.")

        # Numeric features use the two-sample KS statistic; binary features use the absolute
        # difference in positive rates; payment method uses total-variation distance.
        rows = []
        for feature in C.NUMERIC:
            rows.append({
                "Feature": C.FEATURE_LABELS[feature],
                "Feature key": feature,
                "Comparison metric": "KS distance",
                "Distance": float(ks_2samp(my_gan[feature], noon_gan[feature]).statistic),
            })
        for feature in C.BINARY:
            rows.append({
                "Feature": C.FEATURE_LABELS[feature],
                "Feature key": feature,
                "Comparison metric": "Absolute positive-rate gap",
                "Distance": float(abs(my_gan[feature].mean() - noon_gan[feature].mean())),
            })
        my_pay = my_gan["payment_method"].value_counts(normalize=True)
        noon_pay = noon_gan["payment_method"].value_counts(normalize=True)
        payment_tv = 0.5 * sum(abs(my_pay.get(v, 0.0) - noon_pay.get(v, 0.0)) for v in C.PAYMENTS)
        rows.append({
            "Feature": C.FEATURE_LABELS["payment_method"],
            "Feature key": "payment_method",
            "Comparison metric": "Total-variation distance",
            "Distance": float(payment_tv),
        })
        compare = pd.DataFrame(rows)

        avg_distance = float(compare["Distance"].mean())
        c1, c2, c3 = st.columns(3)
        show_kpi(c1, "Features compared", f"{len(compare)}", "all configured fraud features")
        show_kpi(c2, "Mean feature distance", f"{avg_distance:.3f}", "lower = more similar")
        closest = compare.loc[compare["Distance"].idxmin(), "Feature"]
        farthest = compare.loc[compare["Distance"].idxmax(), "Feature"]
        show_kpi(c3, "Largest difference", farthest, f"smallest difference: {closest}")

        fig = px.bar(compare.sort_values("Distance", ascending=False), x="Feature", y="Distance",
                     color="Comparison metric",
                     color_discrete_sequence=[C.CYAN, C.PINK, C.ORANGE],
                     hover_data={"Feature key": True, "Distance": ":.3f"},
                     labels={"Distance": "Distribution distance (lower = more similar)", "Feature": ""})
        fig.update_layout(xaxis_tickangle=-30, yaxis_range=[0, max(0.05, float(compare["Distance"].max()) * 1.15)])
        plot(fig)
        st.dataframe(compare[["Feature", "Comparison metric", "Distance"]].sort_values("Distance", ascending=False).round({"Distance": 3}),
                     width="stretch", hide_index=True)
        st.markdown("**How to interpret this:** KS distance is used for numeric features, the gap in positive rates for binary flags, and total-variation distance for payment method. A distance of 0 means the measured distributions match; lower values indicate closer similarity. The mean is a simple summary across metrics, not a formal overall statistical test.")
