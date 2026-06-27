"""Local-density / redundancy estimators and the weights derived from them.

The premise: in a fine time series or clustered table, near-duplicate rows
*over-count*. A row sitting in a crowded neighborhood carries little new
information beyond its neighbors, yet a uniform fit counts it in full — so dense
regions dominate the estimate. These functions score how crowded each row is so
the redundant ones can be downweighted (weight ∝ 1/density).

Two drivers, both numpy-only and brute-force (fine up to a couple thousand rows):

* :func:`feature_density` — crowding in *feature* space (clusters / duplicates):
  1 / mean distance to the k nearest neighbors, in standardized coordinates.
* :func:`temporal_redundancy` — crowding in *time*: similarity to the rows in a
  local window i-k..i+k, i.e. autocorrelation / consecutive duplication.

:func:`density_weights` turns a density vector into mean-1 sample weights, and
:func:`effective_n` reports the resulting effective sample size — how many truly
independent rows the weighting implies, (Σw)² / Σw², which collapses well below
``n`` exactly when redundancy is present.
"""

from __future__ import annotations

import numpy as np

from .curves import EPS, inverse, normalize


def _standardize(X) -> np.ndarray:
    X = np.asarray(X, float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def feature_density(X, k=15) -> np.ndarray:
    """Per-row local density in standardized feature space: ``1 / mean distance
    to the k nearest neighbors``. High = crowded / redundant (near-duplicates),
    low = isolated. Brute-force pairwise distances (O(n²); fine for n ≲ 2000)."""
    Z = _standardize(X)
    n = Z.shape[0]
    if n < 2:
        return np.ones(n)
    sq = (Z * Z).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.maximum(d2, 0.0, out=d2)
    dist = np.sqrt(d2)
    np.fill_diagonal(dist, np.inf)               # exclude self
    kk = int(min(k, n - 1))
    idx = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
    knn = np.take_along_axis(dist, idx, axis=1)
    mean_knn = knn.mean(axis=1)
    return 1.0 / np.maximum(mean_knn, EPS)


def temporal_redundancy(X, k=5) -> np.ndarray:
    """Per-row temporal density: ``1 / mean distance to the rows in the window
    i-k..i+k`` (standardized). High = a row that closely repeats its temporal
    neighbors — autocorrelated / near-duplicated in time."""
    Z = _standardize(X)
    n = Z.shape[0]
    if n < 2:
        return np.ones(n)
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        neigh = np.concatenate([np.arange(lo, i), np.arange(i + 1, hi)])
        if neigh.size == 0:
            out[i] = 1.0
            continue
        d = np.sqrt(((Z[neigh] - Z[i]) ** 2).sum(axis=1)).mean()
        out[i] = 1.0 / max(d, EPS)
    return out


def density_weights(density, floor=0.1, power=1.0) -> np.ndarray:
    """Mean-1 weights ∝ (1/density)**power. ``floor`` is the minimum relative
    weight (keeps even the densest rows contributing a little), after which the
    vector is renormalized back to mean 1."""
    w = inverse(density, power=power)            # mean-1, ∝ (1/density)**power
    w = np.maximum(w, float(floor))
    return normalize(w)


def effective_n(weights) -> float:
    """Kish effective sample size (Σw)² / Σw². Equals n for uniform weights and
    drops toward the count of *distinct* groups as redundancy is downweighted."""
    w = np.asarray(weights, float)
    s = float(w.sum())
    s2 = float((w * w).sum())
    return (s * s) / s2 if s2 > 0 else 0.0
