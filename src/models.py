"""Model loading and scoring."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.data import load_eval_set


@st.cache_resource(show_spinner="Loading model…")
def load_bundle(model_key: str) -> dict:
    """{'model': RandomForest, 'columns': [...]}; cached as a resource (not pickled per call)."""
    return joblib.load(C.MODEL_DIR / C.MODELS[model_key]["file"])


def featurize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot encode payment_method and align to the model's training columns."""
    X = pd.get_dummies(df[C.FEATURES], columns=["payment_method"]).reindex(columns=columns, fill_value=0)
    return X.astype(float)


def predict_proba_df(model_key: str, df: pd.DataFrame) -> np.ndarray:
    b = load_bundle(model_key)
    return b["model"].predict_proba(featurize(df, b["columns"]))[:, 1]


@st.cache_data(show_spinner="Scoring evaluation set…")
def score_eval_set(model_key: str, eval_key: str) -> np.ndarray:
    """Fraud probabilities for the selected model on the selected evaluation set."""
    return predict_proba_df(model_key, load_eval_set(eval_key))


def predict_one(model_key: str, tx: dict) -> float:
    return float(predict_proba_df(model_key, pd.DataFrame([tx]))[0])
