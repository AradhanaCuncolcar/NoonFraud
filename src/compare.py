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


# ------------------------------------------------------------------------------------ Phase 1: how right were my assumptions?
EFFECT_MATERIAL = 0.10     # |effect| below this = "not a signal"


def _rank_effect(fraud: pd.Series, legit: pd.Series) -> float:
    """2 x AUC - 1: +1 = fraud always higher than legit, -1 = always lower, 0 = no difference."""
    from scipy.stats import rankdata
    n1, n0 = len(fraud), len(legit)
    r = rankdata(np.r_[fraud.to_numpy(float), legit.to_numpy(float)])
    return float(2 * ((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) - 1)


def _fmt(x: float) -> str:
    return f"{x:.0f}" if abs(x) >= 10 else f"{x:.1f}"


def _contrast(df: pd.DataFrame, f: str) -> tuple[float, str, str | None]:
    """(effect of being fraud on feature f, plain-English text, top category for payment)."""
    fr, lg = df[df[C.TARGET] == 1], df[df[C.TARGET] == 0]
    k = kind_of(f)
    if k == "numeric":
        return (_rank_effect(fr[f], lg[f]), f"median {_fmt(fr[f].median())} (fraud) vs {_fmt(lg[f].median())} (legit)", None)
    if k == "binary":
        a, b = fr[f].mean(), lg[f].mean()
        return float(a - b), f"{a:.0%} of fraud vs {b:.0%} of legit", None
    pf, pl = fr[f].value_counts(normalize=True), lg[f].value_counts(normalize=True)
    cat = max(C.PAYMENTS, key=lambda p: pf.get(p, 0) - pl.get(p, 0))
    return float(pf.get(cat, 0) - pl.get(cat, 0)), f"`{cat}`: {pf.get(cat, 0):.0%} of fraud vs {pl.get(cat, 0):.0%} of legit", cat


def assumption_check(mine: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """Compare what makes a row fraud in MY seed world with what makes it fraud in the REAL data.

    `mine` / `real` are labelled sets (is_fraud 0/1). Effects are on a -1..+1 scale (see _rank_effect / rate gap).
    """
    rows = []
    for f in C.FEATURES:
        a, ta, ca = _contrast(mine, f); b, tb, cb = _contrast(real, f)
        big_a, big_b = abs(a) >= EFFECT_MATERIAL, abs(b) >= EFFECT_MATERIAL
        if not big_a and not big_b:
            v = "➖ Not a signal in either"
        elif big_a and not big_b:
            v = ("⚠️ Overstated a weak signal" if np.sign(a) == np.sign(b) and abs(b) >= EFFECT_MATERIAL / 2
                 else "❌ Assumed a signal that is not there")
        elif big_b and not big_a:
            v = "⚠️ Missed a real signal"
        elif kind_of(f) == "categorical" and ca != cb:
            v = "⚠️ Different method flagged"
        elif np.sign(a) != np.sign(b):
            v = "❌ Opposite direction"
        else:
            v = "✅ Right" if 0.5 <= abs(b) / abs(a) <= 2 else "⚠️ Right direction, wrong strength"
        rows.append(dict(feature=f, label=C.FEATURE_LABELS[f], my_assumption=ta, in_reality=tb,
                         my_effect=a, real_effect=b, verdict=v))
    return pd.DataFrame(rows)


def read_loss_log(src) -> pd.DataFrame | None:
    """Read a CTGAN loss log (the DataFrame `gan.loss_values` saved as CSV). Returns Epoch / Generator / Discriminator loss, or None."""
    if src is None:
        return None
    try:
        df = pd.read_csv(src)
    except Exception:
        return None
    low = {c.lower(): c for c in df.columns}
    pick = lambda key: next((low[c] for c in low if key in c), None)
    g, d, e = pick("generator"), pick("discriminator") or pick("distriminator"), pick("epoch")
    if g is None or d is None:
        return None
    return pd.DataFrame({"Epoch": df[e] if e else np.arange(len(df)), "Generator loss": df[g], "Discriminator loss": df[d]})


# ------------------------------------------------------------------------------------ Phase 2.4: reflect
def reflection(table: pd.DataFrame, mine: pd.DataFrame, noon: pd.DataFrame, top: int = 3) -> dict:
    """What Noon's GAN captured that mine missed, where mine was better, and what to change (all from the numbers)."""
    t = table.copy(); t["margin"] = t["my_to_real"] - t["noon_to_real"]
    captured = [f"**{r.label}** — Noon's GAN is closer to real fraud (distance {r.noon_to_real:.2f} vs my {r.my_to_real:.2f})."
                for r in t[t["margin"] > TIE_MARGIN].sort_values("margin", ascending=False).head(top).itertuples()]
    better = [f"**{r.label}** — mine is closer to real fraud (distance {r.my_to_real:.2f} vs Noon's {r.noon_to_real:.2f})."
              for r in t[t["margin"] < -TIE_MARGIN].sort_values("margin").head(2).itertuples()]
    changes = []
    for r in table.sort_values("distance", ascending=False).head(top).itertuples():
        f = r.feature
        if r.kind == "numeric":
            changes.append(f"**{r.label}**: shift my fraud toward Noon's pattern (median {mine[f].median():.0f} → {noon[f].median():.0f}).")
        elif r.kind == "binary":
            changes.append(f"**{r.label}**: change how often it is set in my fraud ({mine[f].mean():.0%} → {noon[f].mean():.0%}).")
        else:
            pm, pn = mine[f].value_counts(normalize=True), noon[f].value_counts(normalize=True)
            p = max(C.PAYMENTS, key=lambda k: abs(pm.get(k, 0) - pn.get(k, 0)))
            changes.append(f"**{r.label}**: rebalance `{p}` in my fraud ({pm.get(p, 0):.0%} → {pn.get(p, 0):.0%}).")
    changes.append("Run the lie-detector on every GAN version *before* training a classifier on it, and stop when it can no longer tell the sets apart.")
    return dict(captured=captured, better=better, changes=changes)
