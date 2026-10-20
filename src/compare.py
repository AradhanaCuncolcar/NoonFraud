"""Feature-by-feature comparison of two synthetic-fraud sets (my GAN vs Noon's GAN).

Pure pandas / numpy / scipy / scikit-learn: no Streamlit, no Plotly. The page (`views/pipeline.py`)
wraps these functions in caches and `src/compare_plots.py` draws them.

One *distance* per feature, always on a 0-1 scale (0 = identical, 1 = completely different):
  numeric      Kolmogorov-Smirnov statistic (largest gap between the two cumulative distributions)
  binary flag  absolute gap between the two "1" rates
  payment      total-variation distance between the two category shares
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src import config as C

MATCH_BELOW = 0.10      # distance < 0.10  -> "Match"   (with ~1,000-5,000 rows KS noise is ~0.05)
CLOSE_BELOW = 0.25      # distance < 0.25  -> "Close", otherwise "Differs"
TIE_MARGIN = 0.03       # "closer to real fraud" is a tie inside this margin (real set has only 65 rows)
CORR_COLS = C.NUMERIC + C.BINARY
SHORT = {"order_amount": "amount", "account_age_days": "acct age", "hour_of_day": "hour",
         "num_orders_last_7d": "orders 7d", "delivery_distance_km": "distance", "prev_refunds": "refunds",
         "discount_applied": "discount", "device_change": "device", "location_mismatch": "mismatch",
         "payment_method": "payment"}


def kind_of(feature: str) -> str:
    return "numeric" if feature in C.NUMERIC else "binary" if feature in C.BINARY else "categorical"


def verdict(distance: float) -> str:
    return "Match" if distance < MATCH_BELOW else "Close" if distance < CLOSE_BELOW else "Differs"


def feature_distance(a: pd.DataFrame, b: pd.DataFrame, feature: str) -> float:
    k = kind_of(feature)
    if k == "numeric":
        return float(ks_2samp(a[feature], b[feature]).statistic)
    if k == "binary":
        return float(abs(a[feature].mean() - b[feature].mean()))
    pa = a[feature].value_counts(normalize=True); pb = b[feature].value_counts(normalize=True)
    return float(0.5 * sum(abs(pa.get(p, 0.0) - pb.get(p, 0.0)) for p in C.PAYMENTS))


def _summary(df: pd.DataFrame, feature: str) -> str:
    k = kind_of(feature)
    if k == "numeric":
        return f"mean {df[feature].mean():.1f} · median {df[feature].median():.0f}"
    if k == "binary":
        return f"{df[feature].mean() * 100:.1f}% = 1"
    top = df[feature].value_counts(normalize=True).head(2)
    return " · ".join(f"{i} {v * 100:.0f}%" for i, v in top.items())


def feature_table(mine: pd.DataFrame, noon: pd.DataFrame, real: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per feature: how each source summarises it, the My-vs-Noon distance and a verdict.

    If `real` (the 65 real fraud rows) is given, also says which GAN sits closer to real fraud.
    """
    rows = []
    for f in C.FEATURES:
        d = feature_distance(mine, noon, f)
        r = dict(feature=f, label=C.FEATURE_LABELS[f], kind=kind_of(f),
                 my_gan=_summary(mine, f), noon_gan=_summary(noon, f), distance=d, verdict=verdict(d))
        if real is not None:
            dm, dn = feature_distance(mine, real, f), feature_distance(noon, real, f)
            r.update(real_fraud=_summary(real, f), my_to_real=dm, noon_to_real=dn,
                     closer_to_real="Tie" if abs(dm - dn) < TIE_MARGIN else "My GAN" if dm < dn else "Noon GAN")
        rows.append(r)
    return pd.DataFrame(rows)


def similarity_score(table: pd.DataFrame) -> float:
    """0-100: 100 x (1 - mean per-feature distance)."""
    return float(100 * (1 - table["distance"].mean()))


def numeric_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean / std / min / p5 / median / p95 / max of every numeric feature for each source."""
    rows = []
    for f in C.NUMERIC:
        for name, df in frames.items():
            s = df[f]
            rows.append({"Feature": C.FEATURE_LABELS[f], "Source": name, "Mean": s.mean(), "Std": s.std(),
                         "Min": s.min(), "P5": s.quantile(.05), "Median": s.median(),
                         "P95": s.quantile(.95), "Max": s.max()})
    return pd.DataFrame(rows)


def categorical_shares(frames: dict[str, pd.DataFrame], feature: str = "payment_method") -> pd.DataFrame:
    rows = []
    for name, df in frames.items():
        vc = df[feature].value_counts(normalize=True)
        rows += [dict(source=name, category=p, share=float(vc.get(p, 0.0) * 100)) for p in C.PAYMENTS]
    return pd.DataFrame(rows)


def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of the 9 numeric + flag columns (robust to the heavy-tailed amounts)."""
    return df[CORR_COLS].corr(method="spearman").fillna(0.0)


def relationship_gap(mine: pd.DataFrame, noon: pd.DataFrame) -> float:
    """Mean absolute difference between the two correlation matrices (off-diagonal only)."""
    diff = (corr_matrix(mine) - corr_matrix(noon)).abs().to_numpy()
    n = diff.shape[0]
    return float(diff[~np.eye(n, dtype=bool)].mean())


def importances_by_feature(model, columns: list[str]) -> pd.Series:
    """Random-forest importance per original feature (the four payment dummies are summed)."""
    imp = pd.Series(model.feature_importances_, index=columns)
    out = {f: float(imp[[c for c in columns if c == f or c.startswith(f + "_")]].sum()) for f in C.FEATURES}
    return pd.Series(out)


def lie_detector(mine: pd.DataFrame, noon: pd.DataFrame, seed: int = 0, n_max: int = 1500) -> dict:
    """Can a classifier tell the two GANs' fraud apart?  0.50 = indistinguishable, 1.00 = trivially different.

    Both sets are de-duplicated first: Noon's file repeats rows, and a repeated row landing in both the train
    and test fold would inflate the accuracy. Also returns which features give the generator away.
    """
    a = mine.drop_duplicates().reset_index(drop=True); b = noon.drop_duplicates().reset_index(drop=True)
    n = min(len(a), len(b), n_max)
    a = a.sample(n, random_state=seed); b = b.sample(n, random_state=seed)
    both = pd.concat([a, b], ignore_index=True)
    X = pd.get_dummies(both[C.FEATURES], columns=["payment_method"]).astype(float)
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    acc = float(cross_val_score(clf, X, y, cv=5).mean())
    clf.fit(X, y)
    return dict(accuracy=acc, n_per_class=n, importance=importances_by_feature(clf, list(X.columns)))


def detection_matrix(probs: dict[str, dict[str, np.ndarray]], threshold: float) -> pd.DataFrame:
    """% of each fraud set flagged by each model.  probs[model][fraud_set] = array of P(fraud)."""
    rows = [dict(model=m, fraud_set=s, flagged=float((p > threshold).mean() * 100))
            for m, sets in probs.items() for s, p in sets.items()]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ plain-English story
def gap_story(table: pd.DataFrame, mine: pd.DataFrame, noon: pd.DataFrame, top: int = 3) -> list[str]:
    """The `top` features where the two GANs differ most, in one sentence each (Markdown)."""
    out = []
    for _, r in table.sort_values("distance", ascending=False).head(top).iterrows():
        f, lab = r["feature"], r["label"]
        if r["kind"] == "numeric":
            m, n = mine[f].median(), noon[f].median()
            extra = ""
            if f == "hour_of_day":
                extra = (f" Night-time (0–4h) orders are {(mine[f] <= 4).mean() * 100:.0f}% of my fraud vs "
                         f"{(noon[f] <= 4).mean() * 100:.0f}% of Noon's.")
            out.append(f"**{lab}** (distance {r['distance']:.2f}) — median {m:.0f} in my GAN vs {n:.0f} in Noon's: "
                       f"Noon's fraud is {'lower' if n < m else 'higher'} on this feature.{extra}")
        elif r["kind"] == "binary":
            m, n = mine[f].mean(), noon[f].mean()
            out.append(f"**{lab}** (gap {r['distance'] * 100:.0f} pts) — set in {m * 100:.0f}% of my fraud vs "
                       f"{n * 100:.0f}% of Noon's: my GAN {'under' if m < n else 'over'}-uses this signal.")
        else:
            pm = mine[f].value_counts(normalize=True); pn = noon[f].value_counts(normalize=True)
            p = max(C.PAYMENTS, key=lambda k: abs(pm.get(k, 0) - pn.get(k, 0)))
            out.append(f"**{lab}** (distance {r['distance']:.2f}) — biggest gap on `{p}`: "
                       f"{pm.get(p, 0) * 100:.0f}% of my fraud vs {pn.get(p, 0) * 100:.0f}% of Noon's.")
    return out
