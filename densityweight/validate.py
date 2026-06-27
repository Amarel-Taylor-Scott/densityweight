"""Held-out + null validation for density weighting.

The discipline is borrowed wholesale from the honest-weighting playbook: a
weighting only *counts* if it beats both uniform weighting and a magnitude-matched
null on a split it never touched.

1. Split into an early ``low`` (train) block and a late ``high`` (held-out) block.
2. Choose the weighting — driver (feature / temporal density), curve, and power —
   using ONLY ``low``, by a 2-fold transfer scored on a *de-duplicated* eval
   (eval rows weighted by 1/density), so the selection optimizes the effective,
   non-redundant population rather than the crowded raw one.
3. Refit on all of ``low`` with the chosen weighting and score ``high`` (drawn
   from the base distribution, scored unweighted — it is the real population).
4. NULL: permute the chosen weights across the ``low`` rows (same magnitudes,
   redundancy structure destroyed), refit, score ``high`` — ``n_null`` times.

A win requires ``learned > null p95`` AND ``learned > uniform``. Otherwise the
honest verdict is "no gain" — randomly downweighting the same number of rows did
just as well, so the *placement* on the redundant rows bought nothing.
"""

from __future__ import annotations

import numpy as np

from .core import pearson, weighted_ridge
from .curves import CURVES, normalize
from .density import effective_n, feature_density, temporal_redundancy

DRIVERS = {"feature": feature_density, "temporal": temporal_redundancy}
POWERS = (0.5, 1.0, 2.0)


def _weighted_pearson(pred, y, w) -> float:
    pred = np.asarray(pred, float).ravel()
    y = np.asarray(y, float).ravel()
    w = np.asarray(w, float).ravel()
    sw = w.sum()
    if sw <= 0:
        return pearson(pred, y)
    pm = (w * pred).sum() / sw
    ym = (w * y).sum() / sw
    a = pred - pm
    b = y - ym
    cov = (w * a * b).sum()
    d = np.sqrt((w * a * a).sum() * (w * b * b).sum())
    return float(cov / d) if d > 0 else 0.0


def _two_fold_score(X, y, driver, curve, power, k, lam) -> float:
    """Density-weighted 2-fold transfer within the train block. The eval fold is
    scored de-duplicated (1/density weighting) so the criterion rewards fitting
    the effective population, not the crowded raw one. Uses only ``X``/``y``."""
    n = len(y)
    half = n // 2
    folds = ((np.arange(half), np.arange(half, n)),
             (np.arange(half, n), np.arange(half)))
    scores = []
    for tr, ev in folds:
        w_tr = curve(driver(X[tr], k=k), power=power)
        pred = weighted_ridge(X[tr], y[tr], X[ev], sw=w_tr, lam=lam)
        w_ev = normalize(1.0 / np.maximum(driver(X[ev], k=k), 1e-12))
        scores.append(_weighted_pearson(pred, y[ev], w_ev))
    return float(np.mean(scores))


def held_out(X, y, *, low_frac=0.6, k=15, n_null=200, lam=1.0, seed=0) -> dict:
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n = len(y)
    cut = max(2, int(n * low_frac))
    low, high = np.arange(cut), np.arange(cut, n)
    Xlo, ylo, Xhi, yhi = X[low], y[low], X[high], y[high]

    uniform = pearson(weighted_ridge(Xlo, ylo, Xhi, lam=lam), yhi)

    # --- select (driver, curve, power) on the train block only ---------------
    best = None
    for dname, driver in DRIVERS.items():
        for cname, curve in CURVES.items():
            for power in POWERS:
                s = _two_fold_score(Xlo, ylo, driver, curve, power, k, lam)
                if best is None or s > best[0]:
                    best = (s, dname, cname, power)
    _, b_driver, b_curve, b_power = best

    # --- refit on all of low with the chosen weighting, score held-out -------
    dens = DRIVERS[b_driver](Xlo, k=k)
    w = CURVES[b_curve](dens, power=b_power)
    learned = pearson(weighted_ridge(Xlo, ylo, Xhi, sw=w, lam=lam), yhi)

    # --- magnitude-matched null: permute the weights across low rows ---------
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_null)
    for i in range(n_null):
        wn = w[rng.permutation(len(w))]
        nulls[i] = pearson(weighted_ridge(Xlo, ylo, Xhi, sw=wn, lam=lam), yhi)
    p95 = float(np.quantile(nulls, 0.95))

    return {
        "learned": learned, "uniform": uniform, "lift": learned - uniform,
        "null_p95": p95, "null_mean": float(nulls.mean()),
        "null_p": float((nulls >= learned).mean()),
        "beats_null": bool(learned > p95 and (learned - uniform) > 0.005),
        "best_curve": b_curve, "best_driver": b_driver, "best_power": b_power,
        "effective_n_ratio": effective_n(w) / len(w),
        "weights": w,
    }
