"""GAN Lab: perturbation simulator layered on top of the real CTGAN output.

Re-running CTGAN with a different latent noise requires PyTorch + the SDV/CTGAN stack, which makes a
lightweight dashboard heavy. Instead the simulator takes the rows my trained CTGAN actually generated
(a 15,000-row pool) and *degrades / stretches* them in controlled ways that mimic what happens when a
GAN is sampled with too much noise, too much variance, or with a poorly-fit categorical head:

  noise        independent Gaussian jitter on every continuous column (std units, in a log/hour space)
               -> breaks correlations, fattens tails, creates impossible rows
  var_scale    stretches (>1) or squeezes (<1) each column around its mean, keeping the correlations
               -> the "temperature" of the generator (mode collapse <1, over-dispersion >1)
  flip         probability to flip each binary flag and to re-draw payment_method uniformly
               -> a noisy categorical/discriminator signal
  enforce      reject rows violating the business rules R1/R2 (refund needs history, young-account velocity)
               exactly like the rejection step used in the notebook
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src import config as C
from src.data import fraud_source, load_my_gan_pool, load_seed_train, load_eval_set
from src.models import featurize
from src.metrics import confusion, derive

DEFAULTS = dict(n_rows=2000, noise=0.0, var_scale=1.0, flip=0.0, enforce=True, seed=42)
SIM_COLS = ["order_amount", "account_age_days", "hour_of_day", "num_orders_last_7d",
            "delivery_distance_km", "prev_refunds"]

# column -> (forward transform, inverse transform, lower clip, upper clip, rounding)
_SPECS = {
    "order_amount":         (np.log,   np.exp,   15.0, 900.0, 2),
    "account_age_days":     (np.log1p, np.expm1, 1.0, 1500.0, 0),
    "delivery_distance_km": (np.log,   np.exp,   0.3, 30.0, 1),
    "num_orders_last_7d":   (np.log1p, np.expm1, 0.0, 40.0, 0),
    "prev_refunds":         (np.log1p, np.expm1, 0.0, 10.0, 0),
}


def rules_ok(df: pd.DataFrame) -> pd.Series:
    """R1: a refund needs prior orders; R2: accounts <7 days old cannot exceed ~4 orders/day of life."""
    r1 = df["prev_refunds"] <= df["account_age_days"] // 3
    r2 = ~((df["account_age_days"] < 7) & (df["num_orders_last_7d"] > 4 * df["account_age_days"]))
    return r1 & r2


@st.cache_data(show_spinner=False)
def _pool_stats() -> dict:
    pool = load_my_gan_pool()
    return {c: (fwd(pool[c].astype(float)).mean(), fwd(pool[c].astype(float)).std())
            for c, (fwd, *_rest) in _SPECS.items()}


def _perturb(b: pd.DataFrame, noise: float, var_scale: float, flip: float, rng: np.random.Generator) -> pd.DataFrame:
    b = b.copy()
    stats = _pool_stats()
    for c, (fwd, inv, lo, hi, nd) in _SPECS.items():
        mu, sd = stats[c]
        z = fwd(b[c].astype(float).to_numpy())
        z = mu + var_scale * (z - mu) + noise * sd * rng.standard_normal(len(b))
        x = np.clip(inv(z), lo, hi)
        b[c] = np.round(x, nd) if nd else np.round(x).astype(int)
    h = b["hour_of_day"].to_numpy(float) + noise * 6.0 * rng.standard_normal(len(b))
    b["hour_of_day"] = (np.round(h) % 24).astype(int)
    for c in C.BINARY:
        f = rng.random(len(b)) < flip
        b[c] = np.where(f, 1 - b[c], b[c]).astype(int)
    re = rng.random(len(b)) < flip
    b["payment_method"] = np.where(re, rng.choice(C.PAYMENTS, len(b)), b["payment_method"])
    return b


@st.cache_data(show_spinner=False)
def simulate(n_rows: int, noise: float, var_scale: float, flip: float, enforce: bool, seed: int) -> tuple:
    """Return (synthetic fraud DataFrame, info dict with raw rule-violation rate and rejection stats)."""
    pool = load_my_gan_pool()
    rng = np.random.default_rng(seed)
    kept, raw_total, raw_bad, rounds = [], 0, 0, 0
    need = n_rows
    while need > 0 and rounds < 10:
        rounds += 1
        base = pool.iloc[rng.choice(len(pool), size=max(need, 50), replace=need > len(pool))]
        cand = _perturb(base.reset_index(drop=True), noise, var_scale, flip, rng)
        ok = rules_ok(cand)
        raw_total += len(cand); raw_bad += int((~ok).sum())
        cand = cand[ok] if enforce else cand
        kept.append(cand.iloc[:need]); need -= len(kept[-1])
    out = pd.concat(kept, ignore_index=True)[C.FEATURES + [C.TARGET]]
    out[C.TARGET] = 1
    return out, dict(violation_rate=raw_bad / max(raw_total, 1), rounds=rounds, requested=n_rows, produced=len(out))


# ------------------------------------------------------------------------------------ fidelity
@st.cache_data(show_spinner=False)
def fidelity_table(sim: pd.DataFrame, reference_key: str) -> pd.DataFrame:
    """KS distance for numeric columns, total-variation distance for flags/payment, vs a reference fraud set."""
    ref = fraud_source(reference_key)
    rows = []
    for c in SIM_COLS:
        rows.append(dict(feature=c, kind="numeric", distance=float(ks_2samp(sim[c], ref[c]).statistic)))
    for c in C.BINARY:
        rows.append(dict(feature=c, kind="flag", distance=float(abs(sim[c].mean() - ref[c].mean()))))
    ps, pr = sim["payment_method"].value_counts(normalize=True), ref["payment_method"].value_counts(normalize=True)
    rows.append(dict(feature="payment_method", kind="categorical",
                     distance=float(0.5 * sum(abs(ps.get(k, 0) - pr.get(k, 0)) for k in C.PAYMENTS))))
    out = pd.DataFrame(rows)
    out["label"] = out["feature"].map(C.FEATURE_LABELS)
    return out


@st.cache_data(show_spinner="Running lie-detector…")
def lie_detector(sim: pd.DataFrame, seed: int = 0) -> float:
    """Cross-validated accuracy of a classifier separating the simulation from Noon's GAN fraud (0.5 = indistinguishable)."""
    noon = fraud_source("noon_gan")
    n = min(len(noon), len(sim), 1000)
    a = sim.sample(n, random_state=seed); b = noon.sample(n, random_state=seed)
    both = pd.concat([a, b], ignore_index=True)
    X = pd.get_dummies(both[C.FEATURES], columns=["payment_method"]).astype(float)
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = RandomForestClassifier(n_estimators=60, max_depth=10, random_state=seed, n_jobs=-1)
    return float(cross_val_score(clf, X, y, cv=3).mean())


# ------------------------------------------------------------------------------------ retrain test
def _train_eval(fraud_rows: pd.DataFrame, seed: int = 0) -> dict:
    seed_df = load_seed_train()
    legit = seed_df[seed_df[C.TARGET] == 0]
    train = pd.concat([legit, fraud_rows[C.FEATURES + [C.TARGET]]], ignore_index=True)
    X = pd.get_dummies(train[C.FEATURES], columns=["payment_method"]).astype(float)
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(X, train[C.TARGET])
    ref = load_eval_set("referee")
    p = clf.predict_proba(featurize(ref, list(X.columns)))[:, 1]
    d = derive(confusion(ref[C.TARGET].to_numpy(), p, 0.5))
    return dict(recall=d["recall"], precision=d["precision"], f1=d["f1"], fp=d["fp"], tp=d["tp"])


@st.cache_data(show_spinner="Retraining a random forest on the simulated fraud…")
def retrain_test(n_rows: int, noise: float, var_scale: float, flip: float, enforce: bool, seed: int) -> dict:
    sim, _ = simulate(n_rows, noise, var_scale, flip, enforce, seed)
    seed_df = load_seed_train()
    return _train_eval(pd.concat([seed_df[seed_df[C.TARGET] == 1], sim]))


@st.cache_data(show_spinner=False)
def retrain_baselines() -> pd.DataFrame:
    """Reference rows for the retrain table (precomputed once with the same harness -> data/gan_lab_baselines.csv)."""
    return pd.read_csv(C.DATA_DIR / "gan_lab_baselines.csv", index_col=0)


def compute_baselines() -> pd.DataFrame:
    """Regenerates data/gan_lab_baselines.csv (run `python -m src.gan_sim` from the project root)."""
    seed_df = load_seed_train()
    seed_fraud = seed_df[seed_df[C.TARGET] == 1]
    clean, _ = simulate(**DEFAULTS)
    rows = {"Seed fraud only (800 rows, no GAN)": _train_eval(seed_fraud),
            "Seed + 800 + unperturbed GAN (2,000 rows)": _train_eval(pd.concat([seed_fraud, clean]))}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    compute_baselines().to_csv(C.DATA_DIR / "gan_lab_baselines.csv")
