"""Tests for densityweight — the mechanism (dense rows scored denser, redundancy
collapses the effective sample size, downweighting them beats a held-out null)
and the honesty control (uniform density must NOT beat the null). Requires numpy;
no network."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from densityweight import (  # noqa: E402
    pearson, weighted_ridge, feature_density, density_weights, effective_n,
    held_out, CURVES,
)
from densityweight.synth import (  # noqa: E402
    make_redundant_cluster, make_uniform_density,
)


def test_weighted_ridge_recovers_linear():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 4))
    beta = np.array([1.0, -2.0, 0.5, 3.0])
    y = X @ beta
    pred = weighted_ridge(X[:90], y[:90], X[90:], lam=1e-6)
    assert pearson(pred, y[90:]) > 0.999


def test_weighted_ridge_weights_are_mean_one_invariant():
    # scaling all weights by a constant must not change the fit (mean-1 normalize)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + 0.1 * rng.standard_normal(80)
    w = rng.random(60) + 0.2
    a = weighted_ridge(X[:60], y[:60], X[60:], sw=w, lam=1.0)
    b = weighted_ridge(X[:60], y[:60], X[60:], sw=w * 7.0, lam=1.0)
    assert np.allclose(a, b)


def test_feature_density_higher_on_planted_cluster():
    X, y, info = make_redundant_cluster(seed=0)
    cut = 960  # held_out's default low block (0.6 * 1600)
    dens = feature_density(X[:cut], k=15)
    dense = info["dense_idx"]
    dense = dense[dense < cut]
    base = np.setdiff1d(np.arange(cut), dense)
    assert dens[dense].mean() > 5.0 * dens[base].mean(), (
        dens[dense].mean(), dens[base].mean())


def test_effective_n_below_n_for_nonuniform():
    X, _, _ = make_redundant_cluster(seed=0)
    w = density_weights(feature_density(X[:960], k=15))
    assert effective_n(w) < len(w)
    # uniform weights → effective n == n exactly
    assert abs(effective_n(np.ones(500)) - 500) < 1e-9


def test_curves_normalized_and_positive():
    rng = np.random.default_rng(3)
    density = np.abs(rng.standard_normal(200)) + 0.05
    for name, fn in CURVES.items():
        for power in (0.5, 1.0, 2.0):
            w = fn(density, power=power)
            assert (w >= 0).all(), name
            assert abs(w.mean() - 1.0) < 1e-9, (name, power, w.mean())
            # monotone: the densest row gets no more weight than the sparsest
            assert w[np.argmax(density)] <= w[np.argmin(density)] + 1e-9, name


def test_held_out_redundant_cluster_beats_null():
    # downweighting the redundant cluster (the right rows) beats chance on held-out
    X, y, _ = make_redundant_cluster(seed=0)
    r = held_out(X, y, n_null=200, seed=0)
    assert r["lift"] > 0.01, r["lift"]
    assert r["beats_null"], r
    assert r["effective_n_ratio"] < 0.85, r["effective_n_ratio"]


def test_held_out_uniform_density_does_not_beat_null():
    # no redundancy → no honest gain; the null must not be beaten
    X, y, _ = make_uniform_density(seed=0)
    r = held_out(X, y, n_null=200, seed=0)
    assert r["lift"] < 0.03, r["lift"]
    assert not r["beats_null"], r


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("\n%d passed" % len(fns))
