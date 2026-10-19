"""Threshold metrics, curves and the business-cost model (all vectorised + cached)."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src import config as C
from src.data import load_eval_set
from src.models import score_eval_set

# NOTE: predicted fraud  <=>  p > t.  This reproduces sklearn's predict() at t = 0.5 exactly
# (a tie at exactly 0.5 goes to the legit class), so the dashboard matches the notebook numbers.


def confusion(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = p > t
    tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn)


def derive(cm: dict) -> dict:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    n = tp + fp + fn + tn
    return dict(precision=prec, recall=rec, f1=f1, accuracy=(tp + tn) / n,
                fpr=fp / (fp + tn) if fp + tn else 0.0, flagged=tp + fp, **cm)


@st.cache_data(show_spinner=False)
def global_scores(model_key: str, eval_key: str) -> dict:
    """Threshold-independent quantities: ROC curve, ROC-AUC, PR-AUC."""
    y = load_eval_set(eval_key)[C.TARGET].to_numpy()
    p = score_eval_set(model_key, eval_key)
    fpr, tpr, _ = roc_curve(y, p)
    return dict(roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
                fpr=fpr, tpr=tpr)


@st.cache_data(show_spinner=False)
def threshold_sweep(model_key: str, eval_key: str, n: int = 101) -> pd.DataFrame:
    """Precision / recall / F1 / counts across a threshold grid (for the trade-off chart)."""
    y = load_eval_set(eval_key)[C.TARGET].to_numpy()
    p = score_eval_set(model_key, eval_key)
    rows = []
    for t in np.linspace(0, 1, n):
        rows.append(dict(threshold=t, **derive(confusion(y, p, t))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ business ROI
def fn_cost_vector(amounts: np.ndarray, mode: str, fixed: float, mult: float, fee: float, fx: float) -> np.ndarray:
    """Cost of missing each fraudulent order (USD)."""
    if mode == "Fixed $ per missed fraud":
        return np.full(len(amounts), fixed, dtype=float)
    return amounts * fx * mult + fee


def total_cost(y, p, t, c_fp, fn_costs_fraud, c_review) -> dict:
    """Cost breakdown at threshold t. `fn_costs_fraud` is the cost vector for the fraud rows (aligned to y==1)."""
    pred = p > t
    fraud_idx = np.flatnonzero(y == 1)
    caught = pred[fraud_idx]
    missed = ~caught
    fp = int((pred & (y == 0)).sum())
    tp = int(caught.sum()); fn = int(missed.sum())
    missed_loss = float(fn_costs_fraud[missed].sum())
    prevented = float(fn_costs_fraud[caught].sum())
    fp_cost = fp * c_fp
    review = (tp + fp) * c_review
    total = missed_loss + fp_cost + review
    baseline = float(fn_costs_fraud.sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=int(((~pred) & (y == 0)).sum()), missed_loss=missed_loss,
                fp_cost=fp_cost, review_cost=review, total=total, prevented=prevented,
                baseline_no_model=baseline, net_benefit=baseline - total)


@st.cache_data(show_spinner=False)
def cost_curve(model_key: str, eval_key: str, c_fp: float, mode: str, fixed: float, mult: float,
               fee: float, fx: float, c_review: float, n: int = 101) -> pd.DataFrame:
    df = load_eval_set(eval_key)
    y = df[C.TARGET].to_numpy(); p = score_eval_set(model_key, eval_key)
    fnc = fn_cost_vector(df.loc[df[C.TARGET] == 1, "order_amount"].to_numpy(), mode, fixed, mult, fee, fx)
    return pd.DataFrame([dict(threshold=t, **total_cost(y, p, t, c_fp, fnc, c_review))
                         for t in np.linspace(0, 1, n)])
