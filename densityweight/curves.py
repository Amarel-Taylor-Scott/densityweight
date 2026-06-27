"""Curves that map a local-density score → a per-row weight.

All of them are *monotone decreasing in density* — a row in a crowded
neighborhood (high density = redundant) gets a small weight, an isolated row a
large one — they differ only in how hard they bend:

* ``inverse``      — weight ∝ (1/density)**power; the default, proportional to
  the inverse neighborhood density, so a 10× denser region is downweighted ~10×.
* ``inverse_sqrt`` — weight ∝ (1/density)**(power/2); a gentler taper for when
  pure 1/density is too aggressive (a handful of near-duplicates would otherwise
  dominate the downweighting).
* ``rank_low``     — weight from the *rank* of density, not its magnitude; robust
  to a few extreme densities (exact duplicates → density ≈ ∞), since only the
  ordering matters.

Every curve returns weights normalized to mean 1 via :func:`normalize`, so they
are interchangeable in :func:`densityweight.core.weighted_ridge`.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalize(w) -> np.ndarray:
    """Clip to non-negative and rescale to mean 1 (falls back to uniform)."""
    w = np.asarray(w, float)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)
    m = w.mean()
    return w / m if m > 0 else np.ones_like(w)


def inverse(density, power=1.0) -> np.ndarray:
    inv = 1.0 / np.maximum(np.asarray(density, float), EPS)
    return normalize(inv ** power)


def inverse_sqrt(density, power=1.0) -> np.ndarray:
    inv = 1.0 / np.maximum(np.asarray(density, float), EPS)
    return normalize(np.sqrt(inv) ** power)


def rank_low(density, power=1.0) -> np.ndarray:
    """Rank-based: the densest row gets the smallest weight, irrespective of how
    extreme its density is. ``order`` is 0 (sparsest) .. n-1 (densest)."""
    d = np.asarray(density, float).ravel()
    n = len(d)
    if n == 0:
        return np.ones(0)
    order = np.argsort(np.argsort(d))            # 0..n-1, high = denser
    sparse_rank = (n - order).astype(float)      # high for low-density rows
    return normalize(sparse_rank ** power)


CURVES = {
    "inverse": inverse,
    "inverse_sqrt": inverse_sqrt,
    "rank_low": rank_low,
}
