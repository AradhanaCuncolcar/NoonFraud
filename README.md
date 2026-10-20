# 🟡 Noon · Fraud Intelligence Console

Interactive Streamlit dashboard built on top of the **Noon Fraud Lab** submission
(GAN-augmented fraud detection for UAE food delivery). Noon-yellow-on-black theme, nine pages, fully self-contained
(data + trained models are in this folder).

## Run it
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
streamlit run app.py                                   # http://localhost:8501
```
`scikit-learn` is pinned to **1.8.0** because the bundled `.joblib` models were pickled with it.
Tested with Streamlit 1.66, Plotly 7.1, SHAP 0.54, mlxtend 0.25, LIME 0.2.

## Pages → requested features
| Page | What it does |
|---|---|
| 🏠 **Overview** | KPIs for the selected model/dataset + the notebook's model scorecard (S0 … Noon B). |
| 🔬 **GAN pipeline** | The assignment's 7 steps (Phase 1: steps 1–2 · Phase 2: 3–4 · Phase 3: 5–7) side by side, with: assumption check, rarity decision, GAN-training loss curve (upload `gan.loss_values` CSV or save it as `data/ctgan_loss.csv`), S0 → S2 before/after, fraud-proportion and reflection tabs, and Noon's own GAN lift. **my GAN vs Noon's GAN**: ① my GAN dataset → ② classifier on it (S2) → ③ Noon's GAN dataset → ④ **feature-by-feature comparison** (scorecard with a distance + Match/Close/Differs verdict for all 10 features, per-feature drill-down, all-features grid, correlation structure, lie-detector, optional real-fraud reference) → ⑤ classification on Noon's GAN data (N1, cross-detection chart) → ⑥ both classifiers on Noon's real orders → ⑦ comparison with Noon's own classifier (Noon B): metrics, ROC, confusion matrices, feature importance. |
| 📋 **Phases & report** | Coverage map of the 4 assignment phases (brief step → where it lives on the dashboard), a ≈1-page report whose numbers are computed live (assumptions, GAN before/after, vs Noon and why, why-not-an-LLM), and downloads (report, my 5,000 GAN fraud rows, model metrics). |
| 🎚️ **Threshold lab** (#3) | Threshold slider (synced with the sidebar) → Precision, Recall, F1, ROC-AUC, confusion matrix, trade-off curve, ROC curve with the live operating point. |
| 💰 **ROI simulator** (#4) | Custom $ cost per False Positive / False Negative (fixed *or* order-value-based), optional review cost, scaling to daily/monthly/yearly volume, cost-vs-threshold curve, one-click "apply cost-optimal threshold". |
| 🔍 **Explainer** (#2) | Build or pick a transaction → verdict + risk gauge + **SHAP waterfall** and/or **LIME** breakdown; global SHAP summary (bar + beeswarm). |
| ⛏️ **Rule mining** (#5) | **FP-Growth / Apriori** on fraud rows (real, Noon GAN, my GAN, or the GAN-Lab simulation): top patterns with support, enrichment vs real legit, stand-alone precision/recall on real data, association rules (A → B) and a rule map. |
| 📈 **GAN vs real EDA** (#6) | Plotly histograms (log, density, cumulative), **3-D scatter** with selectable axes, KS-distance bars, fraud-signature bars. |
| 🧬 **GAN lab** (#7) | Sliders for latent noise σ, variance scale, categorical noise, row count, rule rejection, seed → realism score, rule violations, lie-detector, detectability by the deployed model, live histograms and an optional retrain-and-test experiment. Settings flow into the EDA and Rule-mining pages. |

Sidebar: model (N1 final · S2 phase-1 · Noon B), evaluation set (real referee · synthetic seed test), shared threshold.

## Project layout
```
app.py            entry point: page config, sidebar, navigation
src/              logic (no page layout)
  config.py       paths, palette, feature lists, presets
  styling.py      custom CSS, Plotly "noon" template, KPI cards
  data.py         @st.cache_data loaders
  models.py       model loading (cache_resource) + scoring
  metrics.py      threshold metrics, ROC, cost model
  explain.py      SHAP / LIME + figures
  rules.py        Apriori / FP-Growth mining
  gan_sim.py      GAN-lab perturbation simulator + retrain test
  eda.py          Plotly EDA figures
  compare.py      GAN-vs-GAN feature distances, verdicts, lie-detector, cross-detection (pure pandas/sklearn)
  compare_plots.py  Plotly figures for the GAN pipeline page
  state.py        shared-threshold session-state helpers
views/            one module per page (st.columns / st.expander layouts)
data/ models/     CSVs and trained random forests
.streamlit/config.toml   dark + Noon-yellow theme
```
Heavy computations are wrapped in `@st.cache_data`; model objects use `@st.cache_resource`.
Only the selected page renders, so interactions stay fast.

## Things worth knowing (honest notes)
* **ROC-AUC does not depend on the threshold** (it ranks across all thresholds), so it stays constant as you slide;
  the red dot on the ROC curve is what moves. Precision, recall and F1 update instantly.
* Prediction rule is `P(fraud) > threshold`, which reproduces the notebook's confusion matrices exactly at 0.50
  (N1: 26 TP / 38 FP; S2: 13 / 31; Noon B: 53 / 71).
* **GAN pipeline, step ④ distances** are on a 0–1 scale: KS statistic (numeric), gap in rate (flags), total-variation (payment).
  Match < 0.10 · Close < 0.25 · Differs ≥ 0.25. Noon's file is de-duplicated by default (toggle in the page's options); the
  lie-detector always de-duplicates so repeated rows cannot leak across CV folds. Real fraud (65 rows) is only a descriptive
  reference. Step ⑤ is fraud-only data, so it measures *how much* of each fraud set a classifier recognises (N1 has seen Noon's file in training).
* **GAN lab is a post-generation simulator.** It perturbs the 15,000 rows your CTGAN actually generated
  (noise / variance / flag flips / rule rejection). It does *not* re-sample CTGAN's latent vector — that would need
  PyTorch + CTGAN. Treat it as "what if the generator were noisier / more dispersed", not as a retrained GAN.
* The retrain experiment uses 100 trees on 159,200 seed legit + 800 seed fraud + the simulated rows, scored on the
  referee set at 0.5 (≈15–30 s). Reference rows come from `data/gan_lab_baselines.csv` (regenerate with `python -m src.gan_sim`).
* The referee set has only **65 fraud rows** → recall is ± ~10–12 points (95 %). Rule mining and EDA *describe* it;
  no model is trained or tuned on it. Patterns mined from real fraud are in-sample descriptions.
* Rule enrichment is smoothed (+1 legit order) so patterns with 0–2 legit matches aren't inflated.
* Noon's GAN file has 3,000 rows but only 1,033 distinct ones; the dashboard de-duplicates it.
