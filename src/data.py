"""Cached data loaders. Every function returns a fresh copy-safe DataFrame via @st.cache_data."""
import pandas as pd
import streamlit as st

from src import config as C

_COLS = C.FEATURES + [C.TARGET]


@st.cache_data(show_spinner=False)
def load_eval_set(key: str) -> pd.DataFrame:
    df = pd.read_csv(C.DATA_DIR / C.EVAL_SETS[key]["file"])
    return df[_COLS].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_noon_gan(dedupe: bool = False) -> pd.DataFrame:
    """Noon's GAN fraud file. It contains 3,000 rows but only ~1,033 distinct ones."""
    df = pd.read_csv(C.DATA_DIR / "noon_gan_fraud.csv")[_COLS]
    return df.drop_duplicates().reset_index(drop=True) if dedupe else df


@st.cache_data(show_spinner=False)
def load_my_gan_pool() -> pd.DataFrame:
    """15,000 rows sampled from my CTGAN (the first 5,000 are the delivered submission file)."""
    return pd.read_csv(C.DATA_DIR / "my_gan_pool.csv")[_COLS]


@st.cache_data(show_spinner=False)
def load_seed_train() -> pd.DataFrame:
    """My full seed training set (159,200 legit + 800 fraud) used by the GAN-Lab retrain test."""
    return pd.read_csv(C.DATA_DIR / "seed_train.csv.gz")[_COLS]


@st.cache_data(show_spinner=False)
def load_benchmark() -> pd.DataFrame:
    """Headline metrics from the notebook (Step 7) at the default 0.5 threshold."""
    df = pd.read_csv(C.DATA_DIR / "benchmark_metrics.csv", index_col=0)
    return df.reset_index().rename(columns={"index": "model"})


@st.cache_data(show_spinner=False)
def fraud_source(key: str, dedupe_noon: bool = True) -> pd.DataFrame:
    """Return the fraud-only rows for a source key from config.FRAUD_SOURCES."""
    if key == "real":
        d = load_eval_set("referee")
        return d[d[C.TARGET] == 1].reset_index(drop=True)
    if key == "real_legit":
        d = load_eval_set("referee")
        return d[d[C.TARGET] == 0].reset_index(drop=True)
    if key == "noon_gan":
        return load_noon_gan(dedupe=dedupe_noon)
    if key == "my_gan":
        return load_my_gan_pool().iloc[:5000].reset_index(drop=True)   # the delivered 5,000 rows
    raise KeyError(key)
