"""Local + global explanations for individual predictions (SHAP TreeExplainer and LIME).

SHAP values are computed in *probability* space for the fraud class, so
    base_value + sum(contributions) == predicted fraud probability
and the one-hot payment columns are folded back into a single `payment_method` feature.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.data import load_eval_set
from src.models import featurize, load_bundle, predict_proba_df
from src.styling import style_fig

try:
    import shap
    HAS_SHAP = True
except Exception:  # pragma: no cover
    HAS_SHAP = False
try:
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except Exception:  # pragma: no cover
    HAS_LIME = False


# --------------------------------------------------------------------------------------- helpers
def fmt_value(feature: str, v) -> str:
    if feature in C.BINARY:
        return "Yes" if int(v) == 1 else "No"
    if feature == "payment_method":
        return str(v)
    if feature == "order_amount":
        return f"AED {float(v):,.0f}"
    if feature == "hour_of_day":
        return f"{int(v):02d}:00"
    if feature == "delivery_distance_km":
        return f"{float(v):.1f} km"
    return f"{int(v):,}"


def _fold(sv_row: np.ndarray, columns: list[str]) -> dict:
    """Collapse the 13 model columns (with 4 payment dummies) into the 10 business features."""
    out = {}
    for col, v in zip(columns, sv_row):
        name = "payment_method" if col.startswith("payment_method_") else col
        out[name] = out.get(name, 0.0) + float(v)
    return out


@st.cache_resource(show_spinner=False)
def _tree_explainer(model_key: str):
    return shap.TreeExplainer(load_bundle(model_key)["model"])


def _class1(sv) -> np.ndarray:
    """Handle both shap output layouts: (n, f, 2) array or list of two (n, f) arrays."""
    return sv[1] if isinstance(sv, list) else sv[..., 1]


# --------------------------------------------------------------------------------------- SHAP
@st.cache_data(show_spinner=False)
def shap_single(model_key: str, tx_items: tuple) -> dict:
    tx = dict(tx_items)
    b = load_bundle(model_key)
    X = featurize(pd.DataFrame([tx]), b["columns"])
    ex = _tree_explainer(model_key)
    sv = _class1(ex.shap_values(X))[0]
    base = float(np.ravel(ex.expected_value)[1])
    contribs = _fold(sv, b["columns"])
    return dict(base=base, contribs=contribs, prob=float(b["model"].predict_proba(X)[0, 1]))


@st.cache_data(show_spinner="Computing SHAP values on a stratified sample…")
def shap_global(model_key: str, eval_key: str, n_fraud: int, n_legit: int, seed: int = 0) -> pd.DataFrame:
    """Long table (row, feature, shap, value, value_norm) for a stratified sample of the eval set."""
    df = load_eval_set(eval_key)
    fr = df[df[C.TARGET] == 1]; lg = df[df[C.TARGET] == 0]
    s = pd.concat([fr.sample(min(n_fraud, len(fr)), random_state=seed),
                   lg.sample(min(n_legit, len(lg)), random_state=seed)])
    b = load_bundle(model_key)
    X = featurize(s, b["columns"])
    sv = _class1(_tree_explainer(model_key).shap_values(X))
    rng_lo = df[C.FEATURES].select_dtypes("number").min(); rng_hi = df[C.FEATURES].select_dtypes("number").max()
    rows = []
    for i, (_, r) in enumerate(s.iterrows()):
        folded = _fold(sv[i], b["columns"])
        for f, v in folded.items():
            if f == "payment_method":
                norm = C.PAYMENTS.index(r[f]) / (len(C.PAYMENTS) - 1)
            else:
                lo, hi = rng_lo[f], rng_hi[f]
                norm = (r[f] - lo) / (hi - lo) if hi > lo else 0.0
            rows.append((i, int(r[C.TARGET]), f, v, r[f], float(np.clip(norm, 0, 1))))
    return pd.DataFrame(rows, columns=["row", "is_fraud", "feature", "shap", "value", "value_norm"])


# --------------------------------------------------------------------------------------- LIME
@st.cache_resource(show_spinner=False)
def _lime_explainer(eval_key: str):
    df = load_eval_set(eval_key)
    s = pd.concat([df[df[C.TARGET] == 1], df[df[C.TARGET] == 0].sample(4000, random_state=0)])[C.FEATURES].copy()
    s["payment_method"] = s["payment_method"].map({p: i for i, p in enumerate(C.PAYMENTS)})
    cat_idx = [C.FEATURES.index(c) for c in C.BINARY + C.CATEGORICAL]
    return LimeTabularExplainer(
        s.to_numpy(dtype=float), feature_names=C.FEATURES, class_names=["legit", "fraud"],
        categorical_features=cat_idx, categorical_names={C.FEATURES.index("payment_method"): C.PAYMENTS},
        discretize_continuous=True, random_state=0)


@st.cache_data(show_spinner="Fitting LIME local surrogate…")
def lime_single(model_key: str, eval_key: str, tx_items: tuple, num_samples: int = 3000) -> dict:
    tx = dict(tx_items)

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        d = pd.DataFrame(arr, columns=C.FEATURES)
        d["payment_method"] = d["payment_method"].round().clip(0, 3).astype(int).map(dict(enumerate(C.PAYMENTS)))
        p = predict_proba_df(model_key, d)
        return np.column_stack([1 - p, p])

    x = np.array([tx[f] if f != "payment_method" else C.PAYMENTS.index(tx[f]) for f in C.FEATURES], dtype=float)
    exp = _lime_explainer(eval_key).explain_instance(x, predict_fn, num_features=len(C.FEATURES),
                                                     num_samples=num_samples, labels=(1,))
    labels = [t[0] for t in exp.as_list(label=1)]
    idx = [i for i, _ in exp.as_map()[1]]
    return dict(rows=[(C.FEATURES[i], lab, w) for i, lab, (_, w) in zip(idx, labels, exp.as_map()[1])],
                local_pred=float(exp.local_pred[0]) if hasattr(exp, "local_pred") else None,
                intercept=float(exp.intercept[1]))


# --------------------------------------------------------------------------------------- figures
def waterfall_fig(base: float, contribs: dict, tx: dict, prob: float, threshold: float) -> go.Figure:
    items = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    labels = [f"{C.FEATURE_LABELS[k]} = {fmt_value(k, tx[k])}" for k, _ in items]
    vals = [v * 100 for _, v in items]
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(vals) + ["total"],
        y=["Model base rate"] + labels + ["Predicted fraud risk"],
        x=[base * 100] + vals + [prob * 100],
        text=[f"{base*100:.1f}%"] + [f"{v:+.1f}" for v in vals] + [f"{prob*100:.1f}%"],
        textposition="outside", connector=dict(line=dict(color=C.GRID)),
        increasing=dict(marker=dict(color=C.FRAUD_RED)), decreasing=dict(marker=dict(color=C.SAFE_GREEN)),
        totals=dict(marker=dict(color=C.NOON_YELLOW))))
    fig.add_vline(x=threshold * 100, line_dash="dash", line_color=C.NOON_YELLOW,
                  annotation_text=f"threshold {threshold:.0%}", annotation_position="top")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Fraud probability (%) — red pushes toward FRAUD, green toward LEGIT")
    return style_fig(fig, 470, "SHAP breakdown of this prediction")


def lime_fig(lime_out: dict) -> go.Figure:
    rows = sorted(lime_out["rows"], key=lambda r: abs(r[2]))
    fig = go.Figure(go.Bar(
        x=[r[2] * 100 for r in rows], y=[r[1] for r in rows], orientation="h",
        marker_color=[C.FRAUD_RED if r[2] > 0 else C.SAFE_GREEN for r in rows],
        text=[f"{r[2]*100:+.1f}" for r in rows], textposition="outside"))
    fig.update_xaxes(title="Local surrogate weight (pp of fraud probability)")
    return style_fig(fig, 470, "LIME local surrogate for this prediction")


def gauge_fig(prob: float, threshold: float) -> go.Figure:
    flagged = prob > threshold
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number=dict(suffix="%", font=dict(size=40)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=C.MUTED),
                   bar=dict(color=C.FRAUD_RED if flagged else C.SAFE_GREEN, thickness=0.3),
                   bgcolor=C.PANEL, borderwidth=0,
                   threshold=dict(line=dict(color=C.NOON_YELLOW, width=4), thickness=0.85, value=threshold * 100))))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0))
    return fig


def global_fig(long: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    imp = long.groupby("feature")["shap"].apply(lambda s: s.abs().mean()).sort_values()
    order = list(imp.index)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.38, 0.62], horizontal_spacing=0.03,
                        subplot_titles=("Mean |SHAP| (pp)", "Impact per transaction (colour = feature value)"))
    fig.add_trace(go.Bar(x=imp.values * 100, y=[C.FEATURE_LABELS[f] for f in order], orientation="h",
                         marker_color=C.NOON_YELLOW, showlegend=False), 1, 1)
    rng = np.random.default_rng(0)
    pos = {f: i for i, f in enumerate(order)}
    y = long["feature"].map(pos) + rng.uniform(-0.3, 0.3, len(long))
    fig.add_trace(go.Scatter(
        x=long["shap"] * 100, y=y, mode="markers", showlegend=False,
        marker=dict(size=5, opacity=.75, color=long["value_norm"], colorscale=[[0, C.CYAN], [1, C.FRAUD_RED]],
                    colorbar=dict(title="value", tickvals=[0, 1], ticktext=["low", "high"], len=.6)),
        customdata=np.column_stack([long["feature"].map(C.FEATURE_LABELS), long["value"]]),
        hovertemplate="%{customdata[0]} = %{customdata[1]}<br>SHAP %{x:.2f} pp<extra></extra>"), 1, 2)
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))),
                     ticktext=[C.FEATURE_LABELS[f] for f in order], row=1, col=2)
    fig.add_vline(x=0, line_color=C.GRID, row=1, col=2)
    return style_fig(fig, 480)
