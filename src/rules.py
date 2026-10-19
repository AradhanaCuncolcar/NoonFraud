"""Fraud rule-mining overlay: Apriori / FP-Growth patterns found in fraudulent transactions.

Workflow
1. Every transaction is turned into a "basket" of boolean items (night order, new account, ...).
2. Frequent itemsets are mined *inside the fraud rows* of the chosen source (real / Noon GAN / my GAN).
3. Each pattern is then scored against REAL orders: how common is it among real legit orders
   (enrichment) and, if used as a rule, what precision / recall would it have on the real referee set?
"""
import numpy as np
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

from src import config as C
from src.data import fraud_source

ITEM_DEFS = {
    "Night order (0–4h)":        lambda d: d["hour_of_day"] <= 4,
    "Evening order (18–23h)":    lambda d: d["hour_of_day"] >= 18,
    "New account (<30d)":        lambda d: d["account_age_days"] < 30,
    "Established account (≥1y)": lambda d: d["account_age_days"] >= 365,
    "Big basket (>AED 200)":     lambda d: d["order_amount"] > 200,
    "Small basket (<AED 40)":    lambda d: d["order_amount"] < 40,
    "Order burst (5+ in 7d)":    lambda d: d["num_orders_last_7d"] >= 5,
    "Long distance (>10 km)":    lambda d: d["delivery_distance_km"] > 10,
    "Discount used":             lambda d: d["discount_applied"] == 1,
    "Prior refund(s)":           lambda d: d["prev_refunds"] >= 1,
    "Device change":             lambda d: d["device_change"] == 1,
    "Location mismatch":         lambda d: d["location_mismatch"] == 1,
    "Pay: card":                 lambda d: d["payment_method"] == "card",
    "Pay: wallet":               lambda d: d["payment_method"] == "wallet",
    "Pay: cod":                  lambda d: d["payment_method"] == "cod",
    "Pay: applepay":             lambda d: d["payment_method"] == "applepay",
}


def to_items(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: fn(df).astype(bool) for name, fn in ITEM_DEFS.items()}, index=df.index)


def _support(items: pd.DataFrame, itemset) -> np.ndarray:
    return items[list(itemset)].all(axis=1).to_numpy()


@st.cache_data(show_spinner="Mining frequent patterns…")
def mine_patterns(source: str, algo: str, min_support: float, max_len: int, sim_signature: tuple | None = None,
                  sim_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Frequent itemsets among fraud rows of `source`, enriched with real-data statistics."""
    fraud = sim_df if source == "sim" else fraud_source(source)
    fi = (fpgrowth if algo == "FP-Growth" else apriori)(to_items(fraud), min_support=min_support,
                                                         use_colnames=True, max_len=max_len)
    if fi.empty:
        return fi
    real_f = to_items(fraud_source("real")); real_l = to_items(fraud_source("real_legit"))
    n_f, n_l = len(real_f), len(real_l)
    rows = []
    for _, r in fi.iterrows():
        items = sorted(r["itemsets"])
        mf = _support(real_f, items).sum(); ml = _support(real_l, items).sum()
        legit_sup = ml / n_l
        smoothed = (ml + 1) / n_l                 # +1 legit order: avoids dividing by ~0 for rare patterns
        rows.append(dict(
            pattern=" ∧ ".join(items), items=tuple(items), size=len(items), support=r["support"],
            legit_support=legit_sup,
            enrichment=r["support"] / smoothed,
            real_fraud_hits=int(mf), real_legit_hits=int(ml),
            real_precision=(mf / (mf + ml)) if mf + ml else np.nan, real_recall=mf / n_f))
    out = pd.DataFrame(rows)
    out["enrichment_plot"] = out["enrichment"]
    return out.sort_values(["enrichment", "support"], ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner="Deriving association rules…")
def mine_rules(source: str, algo: str, min_support: float, min_conf: float, max_len: int,
               sim_signature: tuple | None = None, sim_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Association rules A → B *within fraudulent transactions* (co-occurrence structure of fraud)."""
    fraud = sim_df if source == "sim" else fraud_source(source)
    fi = (fpgrowth if algo == "FP-Growth" else apriori)(to_items(fraud), min_support=min_support,
                                                         use_colnames=True, max_len=max(max_len, 2))
    if len(fi) < 2:
        return pd.DataFrame()
    try:
        ru = association_rules(fi, num_itemsets=len(fi), metric="confidence", min_threshold=min_conf)
    except TypeError:                                   # older mlxtend
        ru = association_rules(fi, metric="confidence", min_threshold=min_conf)
    if ru.empty:
        return ru
    ru["rule"] = ru["antecedents"].map(lambda s: " ∧ ".join(sorted(s))) + "  →  " + \
        ru["consequents"].map(lambda s: " ∧ ".join(sorted(s)))
    return ru[["rule", "support", "confidence", "lift", "leverage", "conviction"]] \
        .replace(np.inf, np.nan).sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def pattern_support_across_sources(patterns: tuple) -> pd.DataFrame:
    """Support of each pattern inside Real / Noon-GAN / My-GAN fraud (does the GAN reproduce the pattern?)."""
    rows = []
    for key in ("real", "noon_gan", "my_gan"):
        items = to_items(fraud_source(key))
        for p in patterns:
            rows.append(dict(pattern=" ∧ ".join(p), source=C.FRAUD_SOURCES[key]["label"],
                             support=float(_support(items, p).mean())))
    return pd.DataFrame(rows)
