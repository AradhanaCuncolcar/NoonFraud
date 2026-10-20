"""Plotly figures for the GAN-pipeline page (My GAN vs Noon GAN vs real fraud)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import compare as K
from src import config as C
from src.styling import style_fig

VERDICT_COLORS = {"Match": C.SAFE_GREEN, "Close": C.ORANGE, "Differs": C.FRAUD_RED}
LOG_FEATURES = {"order_amount", "account_age_days", "delivery_distance_km"}


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _x(s: pd.Series, feature: str) -> np.ndarray:
    v = s.to_numpy(float)
    return np.log10(v + (1 if feature == "account_age_days" else 0.001)) if feature in LOG_FEATURES else v


def source_colors(names: list[str]) -> dict:
    palette = {"My GAN": C.PINK, "Noon GAN": C.CYAN, "Real fraud": C.NOON_YELLOW}
    return {n: palette.get(n, C.PURPLE) for n in names}


# ------------------------------------------------------------------------------------ step 4 figures
def distance_fig(table: pd.DataFrame) -> go.Figure:
    """Horizontal bars: how far apart the two GANs are on each feature, coloured by verdict."""
    t = table.sort_values("distance")
    fig = px.bar(t, x="distance", y="label", color="verdict", color_discrete_map=VERDICT_COLORS, orientation="h",
                 category_orders={"verdict": ["Match", "Close", "Differs"]},
                 labels={"distance": "Distance between My GAN and Noon GAN (0 = identical)", "label": ""})
    for x, txt in [(K.MATCH_BELOW, "match"), (K.CLOSE_BELOW, "close")]:
        fig.add_vline(x=x, line_dash="dot", line_color=C.MUTED, annotation_text=txt, annotation_position="top")
    fig.update_xaxes(range=[0, max(0.5, float(t["distance"].max()) * 1.1)])
    return style_fig(fig, 420, "Feature-by-feature distance")


def real_distance_fig(table: pd.DataFrame) -> go.Figure:
    """Grouped bars: each GAN's distance to the real fraud, per feature."""
    t = table.melt(id_vars="label", value_vars=["my_to_real", "noon_to_real"], var_name="gan", value_name="dist")
    t["gan"] = t["gan"].map({"my_to_real": "My GAN", "noon_to_real": "Noon GAN"})
    order = table.sort_values("noon_to_real")["label"].tolist()
    fig = px.bar(t, x="label", y="dist", color="gan", barmode="group",
                 color_discrete_map=source_colors(["My GAN", "Noon GAN"]), category_orders={"label": order},
                 labels={"dist": "Distance to real fraud (lower = closer)", "label": ""})
    fig.update_layout(xaxis_tickangle=-30)
    return style_fig(fig, 380, "Which GAN is closer to REAL fraud? (65 real rows → read gaps < 0.1 as noise)")


def feature_fig(frames: dict[str, pd.DataFrame], feature: str, bins: int = 40, log_x: bool = False,
                cumulative: bool = False) -> go.Figure:
    """Drill-down for ONE feature. Numeric → overlaid density (or ECDF); flag / payment → grouped bars."""
    cmap = source_colors(list(frames))
    k = K.kind_of(feature)
    if k == "numeric":
        parts = []
        for name, df in frames.items():
            parts.append(pd.DataFrame({"value": _x(df[feature], feature) if log_x else df[feature].to_numpy(float),
                                       "source": name}))
        d = pd.concat(parts, ignore_index=True)
        fig = px.histogram(d, x="value", color="source", color_discrete_map=cmap, nbins=bins, barmode="overlay",
                           histnorm="probability density", opacity=0.6, cumulative=cumulative, marginal="box",
                           labels={"value": C.FEATURE_LABELS[feature] + (" (log10)" if log_x else "")})
        fig.update_layout(bargap=0.03)
    elif k == "binary":
        rows = [dict(source=n, rate=float(df[feature].mean() * 100)) for n, df in frames.items()]
        fig = px.bar(pd.DataFrame(rows), x="source", y="rate", color="source", color_discrete_map=cmap, text="rate",
                     labels={"rate": f"% of fraud with {C.FEATURE_LABELS[feature].lower()} = 1", "source": ""})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_yaxes(range=[0, 100]); fig.update_layout(showlegend=False)
    else:
        sh = K.categorical_shares(frames, feature)
        fig = px.bar(sh, x="category", y="share", color="source", barmode="group", color_discrete_map=cmap,
                     labels={"share": "% of fraud", "category": ""})
    return style_fig(fig, 460, f"{C.FEATURE_LABELS[feature]}: {' vs '.join(frames)}")


def grid_fig(frames: dict[str, pd.DataFrame]) -> go.Figure:
    """All ten features at a glance (2 x 5 small multiples)."""
    cmap = source_colors(list(frames))
    feats = C.FEATURES
    titles = [C.FEATURE_LABELS[f] + (" (log10)" if f in LOG_FEATURES else "") for f in feats]
    fig = make_subplots(rows=2, cols=5, subplot_titles=titles, vertical_spacing=0.18, horizontal_spacing=0.05)
    for i, f in enumerate(feats):
        r, c = i // 5 + 1, i % 5 + 1
        k = K.kind_of(f)
        if k == "numeric":
            pooled = np.concatenate([_x(df[f], f) for df in frames.values()])
            lo, hi = np.percentile(pooled, [0.5, 99.5])
            edges = np.arange(-0.5, 24, 1.0) if f == "hour_of_day" else np.linspace(lo, hi, 25)
            centers = (edges[:-1] + edges[1:]) / 2
        for n, df in frames.items():
            show = i == 0
            if k == "numeric":
                dens, _ = np.histogram(np.clip(_x(df[f], f), edges[0], edges[-1]), bins=edges, density=True)
                fig.add_trace(go.Scatter(x=centers, y=dens, mode="lines", name=n, legendgroup=n, showlegend=show,
                                         line=dict(color=cmap[n], width=2, shape="spline"), fill="tozeroy",
                                         fillcolor=_rgba(cmap[n], 0.18)), row=r, col=c)
            elif k == "binary":
                fig.add_trace(go.Bar(x=[n], y=[df[f].mean() * 100], name=n, legendgroup=n, showlegend=False,
                                     marker_color=cmap[n], text=[f"{df[f].mean() * 100:.0f}%"], textposition="outside"),
                              row=r, col=c)
            else:
                sh = df[f].value_counts(normalize=True)
                fig.add_trace(go.Bar(x=C.PAYMENTS, y=[sh.get(p, 0) * 100 for p in C.PAYMENTS], name=n,
                                     legendgroup=n, showlegend=False, marker_color=cmap[n]), row=r, col=c)
        if k == "binary":
            fig.update_yaxes(range=[0, 105], row=r, col=c)
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_annotations(font_size=12)
    style_fig(fig, 540)
    fig.update_layout(margin=dict(t=100), legend=dict(x=0, y=1.17))   # legend above the subplot titles
    return fig


def corr_fig(mine: pd.DataFrame, noon: pd.DataFrame) -> go.Figure:
    """Do the two GANs wire the columns together the same way? (Spearman correlations + their difference)"""
    cm, cn = K.corr_matrix(mine), K.corr_matrix(noon)
    labels = [K.SHORT[c] for c in K.CORR_COLS]
    scale = [[0, C.CYAN], [0.5, "#17171A"], [1, C.NOON_YELLOW]]
    fig = make_subplots(rows=1, cols=3, subplot_titles=["My GAN", "Noon GAN", "Difference (My − Noon)"],
                        horizontal_spacing=0.06)
    for j, (m, zr) in enumerate([(cm, 1.0), (cn, 1.0), (cm - cn, 0.5)], start=1):
        fig.add_trace(go.Heatmap(z=m.to_numpy(), x=labels, y=labels, zmin=-zr, zmax=zr, colorscale=scale,
                                 texttemplate="%{z:.2f}", textfont=dict(size=9), showscale=False,
                                 hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"), row=1, col=j)
        fig.update_yaxes(autorange="reversed", row=1, col=j)
        fig.update_xaxes(tickangle=-45, row=1, col=j)
    return style_fig(fig, 470)


def importance_fig(imp: pd.DataFrame, title: str, ytitle: str = "Importance") -> go.Figure:
    """imp: columns feature, series, value."""
    imp = imp.copy(); imp["label"] = imp["feature"].map(C.FEATURE_LABELS)
    order = imp.groupby("label")["value"].mean().sort_values(ascending=False).index.tolist()
    fig = px.bar(imp, x="label", y="value", color="series", barmode="group", category_orders={"label": order},
                 color_discrete_map=source_colors(imp["series"].unique().tolist()),
                 labels={"value": ytitle, "label": ""})
    fig.update_layout(xaxis_tickangle=-30)
    return style_fig(fig, 400, title)


# ------------------------------------------------------------------------------------ steps 5-7 figures
MODEL_COLORS = {"S2 · my GAN": C.PINK, "N1 · Noon GAN": C.CYAN, "Noon B · real + Noon GAN": C.NOON_YELLOW}


def detection_fig(det: pd.DataFrame) -> go.Figure:
    fig = px.bar(det, x="fraud_set", y="flagged", color="model", barmode="group", text="flagged",
                 color_discrete_map=MODEL_COLORS, labels={"flagged": "% of these fraud rows flagged", "fraud_set": ""})
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig.update_yaxes(range=[0, 112])
    return style_fig(fig, 400, "Who recognises whose fraud?")


def metrics_fig(rows: pd.DataFrame, title: str) -> go.Figure:
    """rows: model, recall, precision, f1."""
    long = rows.melt(id_vars="model", value_vars=["recall", "precision", "f1"], var_name="Metric", value_name="Score")
    fig = px.bar(long, x="model", y="Score", color="Metric", barmode="group", text="Score",
                 color_discrete_sequence=[C.NOON_YELLOW, C.CYAN, C.PINK], labels={"model": ""})
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_yaxes(range=[0, 1.1])
    return style_fig(fig, 400, title)


def confusion_fig(d: dict, title: str) -> go.Figure:
    txt = [[f"<b>{d['tn']:,}</b><br>true negatives", f"<b>{d['fp']:,}</b><br>false positives"],
           [f"<b>{d['fn']:,}</b><br>false negatives", f"<b>{d['tp']:,}</b><br>true positives"]]
    fig = go.Figure(go.Heatmap(z=[[0, 1], [1, 0]], x=["Pred. LEGIT", "Pred. FRAUD"], y=["Actually LEGIT", "Actually FRAUD"],
                               text=txt, texttemplate="%{text}", colorscale=[[0, "#1F1F24"], [1, "#5A5200"]],
                               showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, 300, title)


def roc_fig(curves: dict[str, dict], points: dict[str, dict], zoom: float = 0.05) -> go.Figure:
    fig = go.Figure()
    for name, g in curves.items():
        fig.add_scatter(x=g["fpr"], y=g["tpr"], name=f"{name} (AUC {g['roc_auc']:.3f})",
                        line=dict(color=MODEL_COLORS.get(name, C.PURPLE), width=3))
        p = points[name]
        fig.add_scatter(x=[p["fpr"]], y=[p["recall"]], mode="markers", showlegend=False,
                        marker=dict(size=13, color=MODEL_COLORS.get(name, C.PURPLE), line=dict(color="white", width=2)))
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color=C.MUTED, dash="dot"))
    fig.update_xaxes(title="False-positive rate", range=[0, zoom]); fig.update_yaxes(title="Recall", range=[0, 1.02])
    return style_fig(fig, 400, f"ROC on real orders (FPR ≤ {zoom:.0%}) · dots = current threshold")
