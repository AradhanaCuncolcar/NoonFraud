"""GAN pipeline: the 7 steps of the assignment, run side by side for MY GAN and NOON'S GAN.

  1  Noon describes the data → I create a synthetic dataset with a GAN
  2  Build a classifier from that GAN dataset (fraud / not fraud)            → model S2
  3  Take Noon's own GAN dataset
  4  Compare my GAN dataset with Noon's GAN dataset, feature by feature  ★
  5  Run classification on Noon's GAN dataset                                → model N1
  6  Run the classifiers on Noon's actual (real) data
  7  Compare with Noon's own classifier                                      → model Noon B
"""
import pandas as pd
import streamlit as st

from src import compare as K
from src import compare_plots as P
from src import config as C
from src.data import fraud_source, load_eval_set, load_noon_gan, load_seed_train
from src.metrics import confusion, derive, global_scores
from src.models import load_bundle, predict_proba_df, score_eval_set
from src.styling import plot, section, show_kpi

NAMES = {"S2": "S2 · my GAN", "N1": "N1 · Noon GAN", "NoonB": "Noon B · real + Noon GAN"}
STEPS = [  # (short label, title, model / data used)
    ("① My GAN", "Create dataset with my GAN", "CTGAN on my assumptions"),
    ("② My classifier", "Classifier on my GAN dataset", "S2 · seed + my GAN fraud"),
    ("③ Noon GAN", "Take Noon's GAN dataset", "Noon's synthetic fraud"),
    ("④ GAN vs GAN ★", "Compare the two GAN datasets", "feature by feature"),
    ("⑤ Noon-GAN classifier", "Classify using Noon's GAN data", "N1 · my legit + Noon GAN fraud"),
    ("⑥ Real data", "Run on Noon's actual data", "referee set · 65 real frauds"),
    ("⑦ vs Noon's classifier", "Compare with Noon's classifier", "Noon B · real + Noon GAN"),
]
SUMMARY = "★ Summary"
VERDICT_EMOJI = {"Match": "🟢 Match", "Close": "🟠 Close", "Differs": "🔴 Differs"}


# ------------------------------------------------------------------------------------ cached helpers
@st.cache_data(show_spinner=False)
def _frames(dedupe: bool, with_real: bool) -> dict:
    f = {"My GAN": fraud_source("my_gan"), "Noon GAN": fraud_source("noon_gan", dedupe_noon=dedupe)}
    if with_real:
        f["Real fraud"] = fraud_source("real")
    return f


@st.cache_data(show_spinner=False)
def _comparison(dedupe: bool) -> dict:
    f = _frames(dedupe, True)
    return dict(table=K.feature_table(f["My GAN"], f["Noon GAN"], f["Real fraud"]),
                rel_gap=K.relationship_gap(f["My GAN"], f["Noon GAN"]))


@st.cache_data(show_spinner="Running the lie-detector…")
def _lie(dedupe: bool) -> dict:
    f = _frames(dedupe, False)
    return K.lie_detector(f["My GAN"], f["Noon GAN"])


@st.cache_data(show_spinner="Scoring every model on every fraud set…")
def _fraud_probs(dedupe: bool) -> dict:
    f = _frames(dedupe, True)
    return {NAMES[m]: {s: predict_proba_df(m, df) for s, df in f.items()} for m in NAMES}


@st.cache_data(show_spinner=False)
def _seed_counts() -> tuple[int, int]:
    """(rows, fraud rows) of my seed set. The 30 MB seed file is only needed for these two numbers here, so if it is
    missing from a deployment (e.g. left out of the repo) fall back to the documented 159,200 legit + 800 fraud."""
    try:
        seed = load_seed_train()
        return len(seed), int(seed[C.TARGET].sum())
    except FileNotFoundError:
        return 160_000, 800


def _metrics(mk: str, ek: str, t: float) -> dict:
    y = load_eval_set(ek)[C.TARGET].to_numpy(); p = score_eval_set(mk, ek)
    g = global_scores(mk, ek)
    return dict(model=NAMES[mk], **derive(confusion(y, p, t)), roc_auc=g["roc_auc"], pr_auc=g["pr_auc"])


def _kpi_row(d: dict, tone_fp: str = "red") -> None:
    c = st.columns(4)
    show_kpi(c[0], "Fraud caught (recall)", f"{d['recall'] * 100:.1f}%", f"{d['tp']} of {d['tp'] + d['fn']} frauds", "green")
    show_kpi(c[1], "Precision", f"{d['precision'] * 100:.1f}%", f"{d['tp']} of {d['flagged']} flags correct", "cyan")
    show_kpi(c[2], "F1-score", f"{d['f1']:.3f}", "balance of the two")
    show_kpi(c[3], "False alarms", f"{d['fp']:,}", f"{d['fpr'] * 100:.3f}% of legit orders", tone_fp)


# ------------------------------------------------------------------------------------ page
def _strip(active: int | None) -> None:
    cells = "".join(
        f'<div class="step{" star" if i == 3 else ""}{" active" if active == i else ""}">'
        f'<div class="n">STEP {i + 1}</div><div class="t">{t}</div><div class="m">{m}</div></div>'
        for i, (_, t, m) in enumerate(STEPS))
    st.markdown(f'<div class="steps">{cells}</div>', unsafe_allow_html=True)


def render(ctx: dict) -> None:
    t = ctx["threshold"]
    section("GAN pipeline · my GAN vs Noon's GAN",
            "The assignment's 7 steps, run end to end. The big question: can my GAN generate fraud that looks like Noon's — "
            "and does a classifier built on it work on real orders?")
    labels = [SUMMARY] + [s[0] for s in STEPS]
    st.session_state.setdefault("pipe_step", SUMMARY)
    choice = st.segmented_control("Pipeline step", labels, key="pipe_step", label_visibility="collapsed") or SUMMARY
    idx = None if choice == SUMMARY else labels.index(choice) - 1
    _strip(idx)
    st.caption(f"Models: S2 / N1 / Noon B · sidebar threshold **{t:.2f}** applies to every classifier on this page "
               f"(0.50 reproduces the notebook).")
    [_summary, _s1, _s2, _s3, _s4, _s5, _s6, _s7][0 if idx is None else idx + 1](ctx)


# ------------------------------------------------------------------------------------ summary
def _summary(ctx: dict) -> None:
    t = ctx["threshold"]
    comp = _comparison(True); tab = comp["table"]; lie = _lie(True)
    seed_n, seed_fraud = _seed_counts(); ref = load_eval_set("referee")
    s2_seed, n1_seed = _metrics("S2", "seed_test", t), _metrics("N1", "seed_test", t)
    s2, n1, nb = (_metrics(m, "referee", t) for m in ("S2", "N1", "NoonB"))
    probs = _fraud_probs(True)
    cnt = tab["verdict"].value_counts()
    rows = [
        ("1", "Create dataset with my GAN", "CTGAN",
         f"5,000 synthetic fraud rows · my seed is {seed_fraud / seed_n * 100:.2f}% fraud (real: {ref[C.TARGET].mean() * 100:.2f}%)"),
        ("2", "Classifier on my GAN dataset", "S2",
         f"On my own synthetic test: recall {s2_seed['recall']:.2f}, precision {s2_seed['precision']:.2f}"),
        ("3", "Take Noon's GAN dataset", "—", f"{len(load_noon_gan()):,} rows, only {len(load_noon_gan(True)):,} distinct"),
        ("4", "Compare the two datasets ★", "10 features",
         f"similarity {K.similarity_score(tab):.0f}/100 · {cnt.get('Match', 0)} match · {cnt.get('Close', 0)} close · "
         f"{cnt.get('Differs', 0)} differ · lie-detector {lie['accuracy'] * 100:.0f}%"),
        ("5", "Classify with Noon's GAN data", "N1",
         f"N1 flags {(probs[NAMES['N1']]['Noon GAN'] > t).mean() * 100:.0f}% of Noon GAN fraud; my S2 flags "
         f"{(probs[NAMES['S2']]['Noon GAN'] > t).mean() * 100:.0f}%"),
        ("6", "Run on Noon's actual data", "S2, N1",
         f"S2: recall {s2['recall']:.2f} · {s2['fp']} false alarms  |  N1: recall {n1['recall']:.2f} · {n1['fp']} false alarms"),
        ("7", "Compare with Noon's classifier", "Noon B",
         f"Noon B: recall {nb['recall']:.2f} · precision {nb['precision']:.2f} · {nb['fp']} false alarms"),
    ]
    c = st.columns(4)
    show_kpi(c[0], "Dataset similarity (step 4)", f"{K.similarity_score(tab):.0f}/100", f"{cnt.get('Differs', 0)} feature(s) clearly differ", "cyan")
    show_kpi(c[1], "Lie-detector (step 4)", f"{lie['accuracy'] * 100:.0f}%", "50% = indistinguishable", "red" if lie["accuracy"] > .8 else "")
    show_kpi(c[2], "My best real-data recall", f"{max(s2['recall'], n1['recall']) * 100:.0f}%", "S2 or N1 on 65 real frauds", "green")
    show_kpi(c[3], "Noon's classifier recall", f"{nb['recall'] * 100:.0f}%", f"{nb['fp']} false alarms", "")
    st.write("")
    st.dataframe(pd.DataFrame(rows, columns=["Step", "What happens", "Model / data", "Result (live)"]),
                 hide_index=True, width="stretch")
    a, b = st.columns([3, 2], gap="large")
    with a:
        plot(P.metrics_fig(pd.DataFrame([s2, n1, nb]), "Step 7 in one picture · real referee orders"))
    with b:
        section("Where my GAN falls short")
        f = _frames(True, False)
        for line in K.gap_story(tab, f["My GAN"], f["Noon GAN"]):
            st.markdown(f"- {line}")
        st.caption("Open **④ GAN vs GAN** for the full feature-by-feature breakdown.")


# ------------------------------------------------------------------------------------ step 1
def _s1(ctx: dict) -> None:
    seed_n, seed_fraud = _seed_counts(); ref = load_eval_set("referee"); mine = fraud_source("my_gan")
    seed_rate = seed_fraud / seed_n
    section("Step 1 · Create a dataset with my GAN",
            "Noon only gave the column names. I reasoned out a seed dataset, then used a CTGAN to generate extra synthetic fraud.")
    c = st.columns(4)
    show_kpi(c[0], "Seed dataset", f"{seed_n:,}", f"{seed_fraud} fraud · {seed_rate * 100:.2f}%", "cyan")
    show_kpi(c[1], "Real fraud rate (referee)", f"{ref[C.TARGET].mean() * 100:.2f}%", f"{int(ref[C.TARGET].sum())} of {len(ref):,}", "")
    show_kpi(c[2], "GAN fraud rows (delivered)", f"{len(mine):,}", "synthetic · is_fraud = 1", "green")
    show_kpi(c[3], "Seed vs real rarity", f"{seed_rate / ref[C.TARGET].mean():.1f}×", "my seed is this many times fraudier", "red")
    st.write("")
    a, b = st.columns([3, 2], gap="large")
    with a:
        st.markdown("**Sample of my GAN's fraud**")
        st.dataframe(mine.head(12), hide_index=True, width="stretch")
    with b:
        st.markdown("**Summary of my GAN's numeric features**")
        st.dataframe(K.numeric_summary({"My GAN": mine}).drop(columns="Source").round(1), hide_index=True, width="stretch")
    st.info("Next → **② My classifier**: does a model trained on this data actually find fraud?")


# ------------------------------------------------------------------------------------ step 2
def _s2(ctx: dict) -> None:
    t = ctx["threshold"]; d = _metrics("S2", "seed_test", t)
    section("Step 2 · Classifier built from my GAN dataset",
            "S2 = random forest (200 trees) trained on my seed data + my GAN's fraud. First test: my own synthetic test set.")
    _kpi_row(d)
    st.write("")
    a, b = st.columns([2, 3], gap="large")
    with a:
        plot(P.confusion_fig(d, f"S2 on my synthetic test (40,000 orders · 200 fraud) · t={t:.2f}"))
    with b:
        st.markdown(
            f"- The test set comes from the **same imagined world** as the training data, so this is the *friendliest* exam S2 will get.\n"
            f"- Even here it catches **{d['recall'] * 100:.0f}%** of fraud — fraud is rare, so recall is the number to watch, not accuracy "
            f"({d['accuracy'] * 100:.2f}% accurate!).\n"
            f"- Steps ⑥–⑦ show what happens when S2 meets Noon's real orders.")
    st.info("Next → **③ Noon GAN**: Noon's own synthetic fraud.")


# ------------------------------------------------------------------------------------ step 3
def _s3(ctx: dict) -> None:
    raw, uniq = load_noon_gan(False), load_noon_gan(True)
    section("Step 3 · Take Noon's GAN dataset", "Noon's GAN was trained on REAL fraud, so this is the reference for 'what GAN fraud should look like'.")
    c = st.columns(3)
    show_kpi(c[0], "Rows in Noon's file", f"{len(raw):,}", "all labelled is_fraud = 1", "cyan")
    show_kpi(c[1], "Distinct rows", f"{len(uniq):,}", f"{len(uniq) / len(raw) * 100:.0f}% of the file", "red")
    show_kpi(c[2], "My GAN delivered", "5,000", "synthetic fraud rows", "")
    st.write("")
    a, b = st.columns([3, 2], gap="large")
    with a:
        st.markdown("**Sample of Noon's GAN fraud**")
        st.dataframe(uniq.head(12), hide_index=True, width="stretch")
    with b:
        st.markdown("**Summary of Noon's numeric features**")
        st.dataframe(K.numeric_summary({"Noon GAN": uniq}).drop(columns="Source").round(1), hide_index=True, width="stretch")
        st.caption("The file repeats rows; the comparison (step ④) de-duplicates it by default so repeats do not count twice.")
    st.info("Next → **④ GAN vs GAN ★**: compare the two datasets feature by feature.")


# ------------------------------------------------------------------------------------ step 4
def _s4(ctx: dict) -> None:
    section("Step 4 · My GAN vs Noon's GAN — feature by feature",
            "Goal: can I generate a dataset similar to Noon's? Every column is compared on its own, then the relationships between columns.")
    with st.expander("⚙️ Options", expanded=False):
        a, b = st.columns(2)
        dedupe = a.toggle("De-duplicate Noon's file (1,033 distinct rows)", value=True, key="pipe_dedupe")
        with_real = b.toggle("Also show REAL fraud (65 rows) as a reference", value=True, key="pipe_real",
                             help="Real fraud is only used to describe / compare, never to train.")
        st.caption("Distance: KS statistic (numbers) · gap in rate (flags) · total-variation (payment). 0 = identical, 1 = completely different. "
                   f"Match < {K.MATCH_BELOW} · Close < {K.CLOSE_BELOW} · else Differs.")
    frames = _frames(dedupe, with_real)
    comp = _comparison(dedupe); tab = comp["table"]; lie = _lie(dedupe)
    cnt = tab["verdict"].value_counts()

    c = st.columns(5)
    show_kpi(c[0], "Similarity score", f"{K.similarity_score(tab):.0f}/100", "100 = identical on every feature", "cyan")
    show_kpi(c[1], "Features that match", f"{cnt.get('Match', 0)} / 10", "distance below 0.10", "green")
    show_kpi(c[2], "Close / differ", f"{cnt.get('Close', 0)} / {cnt.get('Differs', 0)}", "orange / red verdicts", "red" if cnt.get("Differs", 0) else "")
    show_kpi(c[3], "Relationship gap", f"{comp['rel_gap']:.2f}", "mean |Δ correlation| between columns", "")
    show_kpi(c[4], "Lie-detector accuracy", f"{lie['accuracy'] * 100:.0f}%", "50% = can't tell them apart", "red" if lie["accuracy"] > .8 else "")
    st.write("")

    t1, t2, t3, t4, t5 = st.tabs(["📋 Scorecard", "🔎 One feature", "🧩 All features", "🔗 Relationships", "🕵️ Lie-detector"])
    with t1:
        show = tab.copy(); show["verdict"] = show["verdict"].map(VERDICT_EMOJI)
        cols = ["label", "kind", "my_gan", "noon_gan"] + (["real_fraud"] if with_real else []) + ["distance", "verdict"] + \
               (["closer_to_real"] if with_real else [])
        ren = {"label": "Feature", "kind": "Type", "my_gan": "My GAN", "noon_gan": "Noon GAN", "real_fraud": "Real fraud",
               "distance": "Distance", "verdict": "Verdict", "closer_to_real": "Closer to real"}
        st.dataframe(show[cols].rename(columns=ren), hide_index=True, width="stretch",
                     column_config={"Distance": st.column_config.ProgressColumn("Distance", min_value=0.0, max_value=1.0, format="%.3f")})
        a, b = st.columns(2, gap="large")
        with a:
            plot(P.distance_fig(tab))
        with b:
            if with_real:
                plot(P.real_distance_fig(tab))
            else:
                st.info("Turn on the real-fraud reference in ⚙️ Options to see which GAN is closer to reality.")
        section("Where the two GANs disagree most")
        for line in K.gap_story(tab, frames["My GAN"], frames["Noon GAN"]):
            st.markdown(f"- {line}")
        if with_real:
            n_noon = int((tab["closer_to_real"] == "Noon GAN").sum()); n_my = int((tab["closer_to_real"] == "My GAN").sum())
            st.markdown(f"- Against **real** fraud, Noon's GAN is closer on **{n_noon}** features, mine on **{n_my}** "
                        f"({10 - n_noon - n_my} ties). With only 65 real rows, treat gaps < 0.1 as noise.")
    with t2:
        a, b, c3, d = st.columns([2, 1, 1, 1])
        feat = a.selectbox("Feature", C.FEATURES, key="pipe_feat", format_func=lambda f: C.FEATURE_LABELS[f])
        numeric = K.kind_of(feat) == "numeric"
        bins = b.slider("Bins", 10, 80, 40, key="pipe_bins", disabled=not numeric)
        log_x = c3.toggle("Log scale", value=feat in P.LOG_FEATURES, key=f"pipe_log_{feat}", disabled=not numeric)
        cum = d.toggle("Cumulative", value=False, key="pipe_cum", disabled=not numeric)
        row = tab[tab["feature"] == feat].iloc[0]
        k = st.columns(3)
        show_kpi(k[0], "Distance (My vs Noon)", f"{row['distance']:.3f}", VERDICT_EMOJI[row["verdict"]], {"Match": "green", "Close": "", "Differs": "red"}[row["verdict"]])
        show_kpi(k[1], "My GAN", row["my_gan"], "", "")
        show_kpi(k[2], "Noon GAN", row["noon_gan"], "", "cyan")
        st.write("")
        plot(P.feature_fig(frames, feat, bins, log_x and numeric, cum and numeric))
        if numeric:
            ns = K.numeric_summary(frames)
            st.dataframe(ns[ns["Feature"] == C.FEATURE_LABELS[feat]].drop(columns="Feature").round(2),
                         hide_index=True, width="stretch")
    with t3:
        plot(P.grid_fig(frames))
        st.caption("Densities of the numeric columns (account age, amount and distance on a log10 axis) · bars show the share of fraud rows with the flag = 1 "
                   "and the payment-method mix.")
    with t4:
        plot(P.corr_fig(frames["My GAN"], frames["Noon GAN"]))
        st.markdown(f"A GAN can match every column's histogram and still wire the columns together differently. The **relationship gap** "
                    f"(mean absolute difference between the two correlation matrices) is **{comp['rel_gap']:.2f}**. "
                    "Bright cells in the right-hand panel are pairs of features that move together in one GAN but not the other.")
    with t5:
        imp = lie["importance"].rename("value").reset_index().rename(columns={"index": "feature"}); imp["series"] = "Lie-detector"
        a, b = st.columns([2, 3], gap="large")
        with a:
            verdict = ("easily tells them apart" if lie["accuracy"] > .8 else "can partly tell them apart" if lie["accuracy"] > .6
                       else "can barely tell them apart")
            st.markdown(f"A random forest trained to guess *which GAN made this row* reaches **{lie['accuracy'] * 100:.0f}%** "
                        f"accuracy (5-fold CV, {lie['n_per_class']:,} distinct rows per GAN). It {verdict}.")
            top = lie["importance"].sort_values(ascending=False).head(3)
            st.markdown("**Biggest giveaways:** " + ", ".join(f"{C.FEATURE_LABELS[f]} ({v * 100:.0f}%)" for f, v in top.items()))
        with b:
            plot(P.importance_fig(imp, "Which features give the generator away?", "Share of the lie-detector's decision"))
    st.info("Next → **⑤ Noon-GAN classifier**: what happens when a classifier is built from Noon's GAN data?")


# ------------------------------------------------------------------------------------ step 5
def _s5(ctx: dict) -> None:
    t = ctx["threshold"]
    probs = _fraud_probs(True)
    flag = lambda m, s: float((probs[NAMES[m]][s] > t).mean() * 100)
    section("Step 5 · Run classification on Noon's GAN dataset",
            "N1 = random forest trained on my legit orders + Noon's GAN fraud. Noon's file is fraud-only, so the measurable thing is how much of it each model recognises.")
    c = st.columns(4)
    show_kpi(c[0], "N1 flags Noon GAN fraud", f"{flag('N1', 'Noon GAN'):.0f}%", "in-sample: N1 trained on it", "green")
    show_kpi(c[1], "S2 flags Noon GAN fraud", f"{flag('S2', 'Noon GAN'):.0f}%", "my classifier, unseen data", "red")
    show_kpi(c[2], "N1 flags my GAN fraud", f"{flag('N1', 'My GAN'):.0f}%", "does Noon-style learning cover mine?", "")
    show_kpi(c[3], "S2 flags my GAN fraud", f"{flag('S2', 'My GAN'):.0f}%", "in-sample for S2", "cyan")
    st.write("")
    det = K.detection_matrix(probs, t)
    det["fraud_set"] = det["fraud_set"].map({"My GAN": "My GAN fraud", "Noon GAN": "Noon GAN fraud", "Real fraud": "REAL fraud (65)"})
    a, b = st.columns([3, 2], gap="large")
    with a:
        plot(P.detection_fig(det))
    with b:
        st.markdown(
            "- **Diagonal = home turf.** Each classifier recognises the fraud it was trained on; that is *not* evidence it works.\n"
            f"- **Off-diagonal = the cross-check.** S2 flags only {flag('S2', 'Noon GAN'):.0f}% of Noon's GAN fraud; "
            f"N1 flags only {flag('N1', 'My GAN'):.0f}% of mine: the two GANs teach different definitions of fraud.\n"
            "- **The right-hand group is the one that matters** — real fraud. Whoever recognises more of it learned the better definition of fraud.\n"
            "- N1 saw Noon's file during training, so its score on that file is optimistic; the real-data run in step ⑥ is the fair test.")
        st.dataframe(det.pivot(index="model", columns="fraud_set", values="flagged").round(1).astype(str) + "%", width="stretch")
    st.info("Next → **⑥ Real data**: both classifiers meet Noon's actual orders.")


# ------------------------------------------------------------------------------------ step 6
def _s6(ctx: dict) -> None:
    t = ctx["threshold"]
    section("Step 6 · Run my classifiers on Noon's actual data",
            "The referee set: 20,000 real orders, 65 of them fraud. Test only; no model here was trained on it.")
    a, b = st.columns(2, gap="large")
    for col, mk, blurb in [(a, "S2", "Trained on my seed + **my GAN's** fraud"), (b, "N1", "Trained on my legit + **Noon GAN's** fraud")]:
        d = _metrics(mk, "referee", t)
        with col:
            st.markdown(f"#### {NAMES[mk]}"); st.caption(blurb)
            _kpi_row(d)
            plot(P.confusion_fig(d, f"{NAMES[mk]} on real orders · t={t:.2f}"))
    s2, n1 = _metrics("S2", "referee", t), _metrics("N1", "referee", t)
    st.markdown(
        f"**Reading it:** swapping my GAN's fraud for Noon's GAN fraud moves real-fraud recall from **{s2['recall'] * 100:.0f}%** to "
        f"**{n1['recall'] * 100:.0f}%** ({s2['tp']} → {n1['tp']} of 65 frauds) at the cost of {n1['fp'] - s2['fp']:+d} false alarms. "
        "Both models use my legit orders and the same random-forest settings; what differs is the fraud they learned from "
        "(my seed + my GAN for S2, Noon's GAN for N1). That gap is what GAN quality buys.")
    st.info("Next → **⑦ vs Noon's classifier**: the final comparison.")


# ------------------------------------------------------------------------------------ step 7
def _s7(ctx: dict) -> None:
    t = ctx["threshold"]
    section("Step 7 · Compare with Noon's actual classifier",
            "Noon B = trained on Noon's real data + Noon's GAN fraud. Same real orders, same threshold.")
    rows = [_metrics(m, "referee", t) for m in ("S2", "N1", "NoonB")]
    df = pd.DataFrame(rows)
    a, b = st.columns(2, gap="large")
    with a:
        plot(P.metrics_fig(df, "Recall · precision · F1 on real orders"))
    with b:
        zoom = st.select_slider("ROC zoom (max false-positive rate)", [0.01, 0.02, 0.05, 0.1, 1.0], value=0.05, key="pipe_zoom")
        curves = {NAMES[m]: global_scores(m, "referee") for m in NAMES}
        plot(P.roc_fig(curves, {r["model"]: r for r in rows}, zoom))
    tbl = df.rename(columns={"model": "Model", "tp": "Frauds caught", "fn": "Frauds missed", "fp": "False alarms",
                             "tn": "Legit passed", "recall": "Recall", "precision": "Precision", "f1": "F1",
                             "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})
    st.dataframe(tbl[["Model", "Recall", "Precision", "F1", "Frauds caught", "Frauds missed", "False alarms", "Legit passed",
                      "ROC-AUC", "PR-AUC"]].round(3), hide_index=True, width="stretch")

    s2, n1, nb = rows
    c = st.columns(3)
    show_kpi(c[0], "Noon B vs N1 (my best)", f"{(nb['recall'] - n1['recall']) * 100:+.0f} pts recall", f"{nb['tp'] - n1['tp']:+d} frauds caught", "green")
    show_kpi(c[1], "…at the price of", f"{nb['fp'] - n1['fp']:+d} false alarms", f"{nb['fp']} vs {n1['fp']}", "red")
    show_kpi(c[2], "Noon B vs my first model (S2)", f"{(nb['recall'] - s2['recall']) * 100:+.0f} pts recall", f"{nb['tp'] - s2['tp']:+d} frauds caught", "cyan")
    st.write("")

    section("What each classifier looks at", "Random-forest feature importance: which columns drive each model's decision.")
    imp = pd.concat([K.importances_by_feature(load_bundle(m)["model"], load_bundle(m)["columns"]).rename("value")
                     .reset_index().rename(columns={"index": "feature"}).assign(series=NAMES[m]) for m in NAMES])
    imp_fig = P.importance_fig(imp, "Feature importance by model")
    for tr in imp_fig.data:
        tr.marker.color = P.MODEL_COLORS.get(tr.name, tr.marker.color)
    plot(imp_fig)
    tops = {m: K.importances_by_feature(load_bundle(m)["model"], load_bundle(m)["columns"]).sort_values(ascending=False).head(2).index.tolist()
            for m in NAMES}
    st.markdown(" · ".join(f"**{NAMES[m]}** leans on {C.FEATURE_LABELS[f[0]]} and {C.FEATURE_LABELS[f[1]]}" for m, f in tops.items()))

    with st.expander("Why Noon's classifier wins", expanded=True):
        st.markdown(
            "- **Whose GAN learned from real fraud?** Noon's GAN was trained on real fraud, so its synthetic rows carry the true patterns "
            "(night-time orders, device changes, billing ≠ delivery mismatch — see step ④). Mine learned from my *assumptions*.\n"
            "- **Noon B also sees real legit orders and real fraud**; N1 and S2 only ever saw my imagined legit orders.\n"
            "- **Honest caveats:** only 65 real frauds, so recall carries ± ~10–12 points (95%). Noon B is Noon's own model, built with access to "
            "real data that I never had, so it is a ceiling to aim for, not a like-for-like contest.")
