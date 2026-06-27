"""Numeric core: Pearson correlation and a per-sample weighted ridge.

The ridge uses the weighted Gram solve (``A = Xcᵀ W Xc``, ``b = Xcᵀ W (y-ȳ)``)
so every refit is cheap — the held-out selection sweep and the hundreds of null
permutations all reuse it. Sample weights are always renormalized to mean 1, so
weighting only changes the *relative* influence of rows, never the effective
regularization strength: a uniform-weight fit and an unweighted fit coincide.
"""

from __future__ import annotations

import numpy as np


def pearson(a, b) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def weighted_ridge(X_tr, y_tr, X_ev, sw=None, lam=1.0):
    """Per-sample weighted ridge via the weighted Gram trick. ``sw`` is
    renormalized to mean 1 (so ``sw=None`` == uniform == plain ridge)."""
    X_tr = np.asarray(X_tr, float)
    y_tr = np.asarray(y_tr, float).ravel()
    X_ev = np.asarray(X_ev, float)
    n, p = X_tr.shape
    w = np.ones(n) if sw is None else np.asarray(sw, float)
    w = w / w.mean() if w.mean() > 0 else np.ones(n)
    sw_sum = w.sum()
    mu = (X_tr * w[:, None]).sum(axis=0) / sw_sum
    ybar = float((y_tr * w).sum() / sw_sum)
    Xc = X_tr - mu
    A = Xc.T @ (Xc * w[:, None])
    b = Xc.T @ (w * (y_tr - ybar))
    beta = np.linalg.solve(A + lam * np.eye(p), b)
    return (X_ev - mu) @ beta + ybar
