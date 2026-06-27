"""Synthetic datasets that isolate exactly what densityweight is meant to fix.

* :func:`make_redundant_cluster` — a clean *base* population plus a DENSE cluster
  of near-duplicate rows (a few seed points replicated with tiny jitter), heavily
  over-represented in the training region. When ``biased`` the cluster follows a
  slightly different coefficient vector, so a uniform fit — dominated by the
  redundant copies — is pulled off the base law. The held-out tail is drawn from
  the base distribution, so downweighting the dense rows by 1/density recovers the
  base fit and should beat a magnitude-matched null.
* :func:`make_uniform_density` — a roughly uniform-density population with no
  redundancy. There is nothing to gain by density weighting, so it must NOT beat
  the null (the honest negative control).

Rows are returned in time order with the dense cluster confined to the first
``low_frac`` of rows, matching :func:`densityweight.validate.held_out`'s split.
"""

from __future__ import annotations

import numpy as np


def make_redundant_cluster(n=1600, p=8, dup_frac=0.4, noise=0.3, seed=0,
                           biased=True, low_frac=0.6, m_seeds=16, jitter=0.02,
                           bias_strength=2.5):
    """Base population + an over-represented dense cluster of near-duplicates.

    ``dup_frac`` of the rows are dense (all inside the first ``low_frac``, i.e.
    the train region); they are ``m_seeds`` distinct points each replicated many
    times with Gaussian ``jitter``. With ``biased`` the cluster's targets follow
    ``beta + bias_strength * unit_delta`` while the base follows ``beta``.

    Returns ``(X, y, {"dense_idx": idx})`` with ``idx`` the dense-row indices.
    """
    rng = np.random.default_rng(seed)
    cut = int(n * low_frac)
    n_dense = min(int(n * dup_frac), cut)

    beta = rng.standard_normal(p)
    delta = rng.standard_normal(p)
    delta = delta / (np.linalg.norm(delta) + 1e-12) * bias_strength
    beta_dense = beta + delta if biased else beta

    X = np.empty((n, p))
    y = np.empty(n)
    is_dense = np.zeros(n, bool)

    # dense rows: random positions within the train region, interleaved with base
    dense_pos = rng.choice(cut, size=n_dense, replace=False)
    is_dense[dense_pos] = True
    seeds = rng.standard_normal((m_seeds, p))
    which = rng.integers(0, m_seeds, size=n_dense)
    Xd = seeds[which] + jitter * rng.standard_normal((n_dense, p))
    X[dense_pos] = Xd
    y[dense_pos] = Xd @ beta_dense + noise * rng.standard_normal(n_dense)

    # base rows: everywhere else (fills the rest of train + the whole held-out tail)
    base_pos = np.where(~is_dense)[0]
    Xb = rng.standard_normal((base_pos.size, p))
    X[base_pos] = Xb
    y[base_pos] = Xb @ beta + noise * rng.standard_normal(base_pos.size)

    return X, y, {"dense_idx": np.where(is_dense)[0]}


def make_uniform_density(n=1500, p=8, noise=1.0, seed=0):
    """Roughly uniform density, no redundancy — the honest negative control.
    Density weighting here only reshuffles homoscedastic Gaussian rows, so it must
    not beat the null. Returns ``(X, y, {})``."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y = X @ beta + noise * rng.standard_normal(n)
    return X, y, {}
