"""Dynamic Plotly EDA: synthetic GAN data vs real data."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import ks_2samp

from src import config as C
from src.data import fraud_source
from src.gan_sim import simulate
from src.styling import style_fig

LOG_DEFAULT = {"order_amount", "account_age_days", "delivery_distance_km"}


@st.cache_data(show_spinner=False)
def combined(source_keys: tuple, sim_params: tuple, per_source: int, seed: int = 0) -> pd.DataFrame:
    """Stack (a sample of) several sources with a `source` label column."""
    parts = []
    for k in source_keys:
        if k == "sim":
            df = simulate(*sim_params)[0]
        else:
            df = fraud_source(k)
        df = df.sample(min(per_source, len(df)), random_state=seed).copy() if per_source else df.copy()
        df["source"] = C.FRAUD_SOURCES[k]["label"]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def color_map(source_keys) -> dict:
    return {C.FRAUD_SOURCES[k]["label"]: C.FRAUD_SOURCES[k]["color"] for k in source_keys}


def hist_fig(df: pd.DataFrame, feature: str, cmap: dict, bins: int, log_x: bool, normalize: bool,
             cumulative: bool) -> go.Figure:
    d = df.copy()
    x = feature
    if log_x:
        d[feature + " (log10)"] = np.log10(d[feature].clip(lower=1e-3) + (1 if feature == "account_age_days" else 0))
        x = feature + " (log10)"
    fig = px.histogram(d, x=x, color="source", color_discrete_map=cmap, nbins=bins, barmode="overlay",
                       histnorm="probability density" if normalize else None, opacity=0.62,
                       cumulative=cumulative, marginal="box")
    fig.update_layout(bargap=0.03)
    return style_fig(fig, 460, f"Distribution of {C.FEATURE_LABELS[feature].lower()}")


def scatter3d_fig(df: pd.DataFrame, cmap: dict, x: str, y: str, z: str, size: int, opacity: float,
                  log_axes: tuple) -> go.Figure:
    d = df.copy()
    for ax in (x, y, z):
        if ax in log_axes:
            d[ax] = np.log10(d[ax] + (1 if ax in ("account_age_days", "num_orders_last_7d", "prev_refunds") else 0.001))
    # tiny jitter on discrete integer axes so points do not stack
    rng = np.random.default_rng(0)
    for ax in (x, y, z):
        if ax in ("hour_of_day", "num_orders_last_7d", "prev_refunds"):
            d[ax] = d[ax] + rng.uniform(-0.25, 0.25, len(d))
    lab = lambda a: C.FEATURE_LABELS[a] + (" (log10)" if a in log_axes else "")
    fig = px.scatter_3d(d, x=x, y=y, z=z, color="source", color_discrete_map=cmap, opacity=opacity,
                        labels={x: lab(x), y: lab(y), z: lab(z)})
    fig.update_traces(marker=dict(size=size, line=dict(width=0)))
    fig.update_layout(scene=dict(
        xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=C.GRID),
        yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=C.GRID),
        zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=C.GRID)), margin=dict(l=0, r=0, t=40, b=0))
    return style_fig(fig, 620, "3-D view: where does each source place its transactions?")


@st.cache_data(show_spinner=False)
def ks_vs_real(source_keys: tuple, sim_params: tuple) -> pd.DataFrame:
    """KS statistic per numeric feature (0 = identical distribution) of each synthetic source vs real fraud."""
    real = fraud_source("real"); rows = []
    for k in source_keys:
        if k in ("real", "real_legit"):
            continue
        s = simulate(*sim_params)[0] if k == "sim" else fraud_source(k)
        for c in C.NUMERIC:
            rows.append(dict(source=C.FRAUD_SOURCES[k]["label"], feature=C.FEATURE_LABELS[c],
                             KS=float(ks_2samp(s[c], real[c]).statistic)))
    return pd.DataFrame(rows)


def ks_fig(ks: pd.DataFrame, cmap: dict) -> go.Figure:
    fig = px.bar(ks, x="feature", y="KS", color="source", barmode="group", color_discrete_map=cmap,
                 labels={"KS": "KS distance vs real fraud (lower = closer)", "feature": ""})
    return style_fig(fig, 360, "Distribution distance to real fraud")


@st.cache_data(show_spinner=False)
def signature_rates(source_keys: tuple, sim_params: tuple) -> pd.DataFrame:
    """Share of each source that is 'night', 'new account', 'device change', ... (fraud signatures)."""
    defs = {"Night (0–4h)": lambda d: d.hour_of_day <= 4, "New account (<30d)": lambda d: d.account_age_days < 30,
            "Basket > AED 200": lambda d: d.order_amount > 200, "Order burst (5+/7d)": lambda d: d.num_orders_last_7d >= 5,
            "Discount": lambda d: d.discount_applied == 1, "Device change": lambda d: d.device_change == 1,
            "Location mismatch": lambda d: d.location_mismatch == 1,
            "Prior refund": lambda d: d.prev_refunds >= 1}
    rows = []
    for k in source_keys:
        s = simulate(*sim_params)[0] if k == "sim" else fraud_source(k)
        for name, fn in defs.items():
            rows.append(dict(source=C.FRAUD_SOURCES[k]["label"], signature=name, share=float(fn(s).mean() * 100)))
    return pd.DataFrame(rows)


def signature_fig(sig: pd.DataFrame, cmap: dict) -> go.Figure:
    fig = px.bar(sig, x="signature", y="share", color="source", barmode="group", color_discrete_map=cmap,
                 labels={"share": "% of source", "signature": ""})
    return style_fig(fig, 360, "Fraud signatures: share of each source")
