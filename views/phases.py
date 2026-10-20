"""Phases & report: where every deliverable of the 4-phase assignment lives, plus the ≈1-page report with live numbers."""
import pandas as pd
import streamlit as st

from src import compare as K
from src import config as C
from src.data import fraud_source, load_benchmark, load_eval_set
from src.styling import section, show_kpi
from views import pipeline as PL

COVERAGE = [  # (phase, brief step, deliverable, where on the dashboard)
    ("1 · Build blind", "1.1", "Reason out what fraud looks like (assumptions, and how right they were)", "🔬 GAN pipeline → ① → *How right were my assumptions?*"),
    ("1 · Build blind", "1.2", "Seed dataset and the rarity decision", "🔬 GAN pipeline → ① → *Rarity decision*"),
    ("1 · Build blind", "1.3", "Classifier on the seed; feel the problem (confusion matrix)", "🔬 GAN pipeline → ② → *Before vs after* (S0)"),
    ("1 · Build blind", "1.4", "Real GAN + evidence it trained (loss curve)", "🔬 GAN pipeline → ① → *Evidence the GAN trained*"),
    ("1 · Build blind", "1.5", "Retrain with GAN fraud; recall / precision / F1 / false positives", "🔬 GAN pipeline → ② → *Before vs after* (S0 → S2)"),
    ("2 · Reveal", "2.1", "Comparison EDA: my GAN vs Noon's GAN", "🔬 GAN pipeline → ④ (scorecard, one feature, all features) · 📈 GAN vs real EDA"),
    ("2 · Reveal", "2.2", "Fraud proportion: is mine realistic?", "🔬 GAN pipeline → ④ → ⚖️ Fraud proportion"),
    ("2 · Reveal", "2.3", "Lie-detector test", "🔬 GAN pipeline → ④ → 🕵️ Lie-detector · 🧬 GAN lab"),
    ("2 · Reveal", "2.4", "Reflect: what Noon captured, what I'd change", "🔬 GAN pipeline → ④ → 💭 Reflect"),
    ("3 · Referee", "3.1", "My classifier on the real data (recall, precision, F1, confusion matrix)", "🔬 GAN pipeline → ⑥ · 🎚️ Threshold lab"),
    ("3 · Referee", "3.2", "Noon's classifier on the same real data", "🔬 GAN pipeline → ⑦"),
    ("3 · Referee", "3.3", "Final bar chart: who caught more, who flagged fewer innocents", "🔬 GAN pipeline → ⑦ · ★ Summary"),
    ("3 · Referee", "3.4", "Explain the gap", "🔬 GAN pipeline → ⑦ → *Why Noon's classifier wins* · below"),
    ("4 · Ship it", "4", "Fraud-check tool: type a transaction → FRAUD / LEGIT", "🔍 Explainer"),
    ("Submit", "8.4", "≈1-page written report", "this page → *Report*"),
    ("Submit", "8.2–8.3", "Synthetic fraud CSV and metrics", "this page → *Downloads*"),
    ("Submit", "8.1", "Run notebook with outputs", "outside the dashboard (your .ipynb)"),
]


def _report(ctx: dict) -> str:
    t = ctx["threshold"]
    chk = K.assumption_check(load_eval_set("seed_test"), load_eval_set("referee"))
    seed_n, seed_fraud = PL._seed_counts(); ref = load_eval_set("referee")
    s0, s2b = PL._bench("S0"), PL._bench("S2")
    s2, n1, nb = (PL._metrics(m, "referee", t) for m in ("S2", "N1", "NoonB"))
    f = PL._frames(True, False); comp = PL._comparison(True); tab = comp["table"]; lie = PL._lie(True)
    cnt = tab["verdict"].value_counts()
    right = chk[chk["verdict"].str.startswith("✅")]["label"].tolist()
    off = chk[~chk["verdict"].str.startswith(("✅", "➖"))]["label"].tolist()
    gaps = " ".join(K.gap_story(tab, f["My GAN"], f["Noon GAN"], top=2)).replace("**", "")
    return f"""# Fraud detection at Noon: GAN-augmented classifier

*Draft written from the dashboard numbers (referee set: 20,000 real orders, 65 fraud; threshold {t:.2f}). Edit before submitting.*

## 1. What did I assume fraud looked like, and how right was I?
I treated fraud as a *rare* event ({seed_fraud / seed_n * 100:.2f}% of my seed vs {ref[C.TARGET].mean() * 100:.2f}% of real orders) and built my seed world around the fraud signals in the data dictionary.
Checked against real data, **{len(right)} of 10 assumptions were right** ({', '.join(right) or 'none'}).
Where I was off: {', '.join(off) or 'nowhere'}.

## 2. Did the GAN improve my fraud detection?
{"Yes, on the number that matters." if s2b['recall'] > s0['recall'] else "Not clearly."} Adding my GAN's fraud to the seed lifted **recall from {s0['recall'] * 100:.1f}% to {s2b['recall'] * 100:.1f}%**
({s0['tp']} → {s2b['tp']} of 65 frauds), precision went {s0['precision'] * 100:.0f}% → {s2b['precision'] * 100:.0f}%, F1 {s0['f1']:.2f} → {s2b['f1']:.2f},
and false alarms rose {s0['fp']} → {s2b['fp']}. Accuracy would have hidden this: the seed-only model is {(s0['tp'] + s0['tn']) / (s0['tp'] + s0['tn'] + s0['fp'] + s0['fn']) * 100:.2f}% accurate.

## 3. How did my GAN and classifier compare to Noon's on real data, and why?
| Model | Recall | Precision | F1 | Frauds caught | False alarms |
|---|---|---|---|---|---|
| S2 · seed + my GAN | {s2['recall']:.2f} | {s2['precision']:.2f} | {s2['f1']:.2f} | {s2['tp']} / 65 | {s2['fp']} |
| N1 · my legit + Noon GAN | {n1['recall']:.2f} | {n1['precision']:.2f} | {n1['f1']:.2f} | {n1['tp']} / 65 | {n1['fp']} |
| Noon B · real + Noon GAN | {nb['recall']:.2f} | {nb['precision']:.2f} | {nb['f1']:.2f} | {nb['tp']} / 65 | {nb['fp']} |

Noon's classifier caught the most real fraud ({nb['tp']} of 65); the fewest false alarms came from {min((s2, n1, nb), key=lambda r: r['fp'])['model']} ({min(r['fp'] for r in (s2, n1, nb))}).
**Why the gap:** Noon's GAN learned from *real* fraud, mine learned from my *assumptions*. Feature by feature my fraud and Noon's are {cnt.get('Match', 0)} match / {cnt.get('Close', 0)} close / {cnt.get('Differs', 0)} differ,
and a lie-detector tells them apart {lie['accuracy'] * 100:.0f}% of the time (50% = indistinguishable). Largest gaps: {gaps}
Noon B also trained on real legitimate orders, which I never had.

## 4. Why not just ask an LLM to "make more fraud rows"?
A GAN **learns a joint distribution from examples**: the generator is trained against a discriminator until its rows are statistically hard to tell from the real fraud it saw,
so correlations between columns (e.g. night-time + new device + location mismatch) are reproduced, and any number of rows can be drawn with the same statistics.
An LLM has no training signal tying its output to *my* data: it prints plausible-looking rows from general priors, cannot be checked against a target distribution, and cannot show training evidence.

*Caveat: with only 65 real frauds, recall carries roughly ±10–12 points (95%).*
"""


def render(ctx: dict) -> None:
    section("Phases & report", "Where every deliverable of the 4-phase assignment lives on this dashboard, and the ≈1-page report.")
    st.dataframe(pd.DataFrame(COVERAGE, columns=["Phase", "Brief step", "Deliverable", "Where"]), hide_index=True, width="stretch")
    st.write("")
    section("Report", "Every figure below is computed live from the data and models.")
    md = _report(ctx)
    st.markdown(md)
    st.write("")
    section("Downloads")
    c = st.columns(3)
    c[0].download_button("⬇️ Report (Markdown)", md, "noon_fraud_report.md", "text/markdown", width="stretch")
    c[1].download_button("⬇️ My GAN fraud (CSV, 5,000 rows)", fraud_source("my_gan").to_csv(index=False), "my_synthetic_fraud.csv", "text/csv", width="stretch")
    c[2].download_button("⬇️ Model metrics (CSV)", load_benchmark().to_csv(index=False), "model_metrics.csv", "text/csv", width="stretch")
