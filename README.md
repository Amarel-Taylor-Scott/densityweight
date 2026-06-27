# densityweight

> Non-naive sample weighting that corrects for **redundancy / density**. In a
> fine time series or clustered table, near-duplicate rows *over-count* — the
> effective sample size is far below `n`, and a uniform fit is dominated by the
> dense regions, biasing it. densityweight scores each row's local density and
> **downweights crowded rows (weight ∝ 1/local_density)** so the model is driven
> by the *distinct* information, not the redundant copies. The gain is validated
> on a held-out split against a **magnitude-matched null**. numpy-only.

```python
from densityweight import held_out
from densityweight.synth import make_redundant_cluster
X, y, info = make_redundant_cluster()          # rows in time order
r = held_out(X, y)
print(r["uniform"], r["learned"], r["beats_null"], r["effective_n_ratio"])
```

```
$ densityweight demo
## 1. Redundant cluster (a dense near-duplicate blob over-represented in train)
  effective sample size: 346 / 960 rows  (ratio 0.361 — redundancy collapses n)
  weighting chosen     : feature / inverse  (power 1.0)
  held-out transfer    : 0.9836   (uniform 0.6382, lift +0.3455)
  null p95 / p         : 0.6712 / 0.000
  VERDICT              : REAL — beats held-out null
  -> mean weight on dense/redundant rows 0.08 vs base rows 2.84 (dense suppressed)

## 2. Uniform density control (no redundancy — must NOT beat null)
  effective sample size: 800 / 900 rows  (ratio 0.889 — redundancy ≈ none)
  held-out transfer    : 0.9452   (uniform 0.9451, lift +0.0001)
  VERDICT              : no gain over null (the honest answer)
```

## The idea

Row counts lie when rows are redundant. Thirty near-identical samples from one
busy minute of a sensor, or one over-sampled cluster, count thirty times in a
plain fit — so the model is pulled toward whatever is *frequent*, not whatever is
*representative*. The honest quantity is the **effective sample size**,
(Σw)² / Σw², which collapses far below `n` exactly when rows duplicate each other.

densityweight measures local crowding and assigns each row a weight inversely
proportional to it, so a region with 30 near-duplicates contributes like ~1, not
30. Two transparent drivers:

| Driver | Crowding in… | Score |
|---|---|---|
| `feature_density` | feature space (clusters / duplicate rows) | 1 / mean distance to the k nearest neighbors (standardized) |
| `temporal_redundancy` | time (autocorrelation / repeats) | 1 / mean distance to the rows in a local window `i-k..i+k` |

A small family of **curves** shapes density → weight — `inverse` (∝ 1/density),
`inverse_sqrt` (gentler), and `rank_low` (rank-based, robust to exact
duplicates) — each normalized to mean 1, so weighting only changes the *relative*
influence of rows, never the regularization strength.

## Honesty is the whole point

Downweighting is only worth anything if it downweights the **right** rows. So
nothing counts unless it beats a held-out null. `held_out`:

1. splits into an early **train** block and a late **held-out** block;
2. picks the weighting (driver, curve, power) using **only train**, by a 2-fold
   transfer scored on a *de-duplicated* eval — never touching held-out;
3. refits on train with that weighting and scores the held-out block;
4. compares against the same weights **permuted across the train rows** (identical
   magnitudes, redundancy structure destroyed) — `n_null` times.

A win requires `learned > null p95` **and** `learned > uniform`. If a random
re-shuffle of the same weights does just as well, the *placement* on the
redundant rows bought nothing — and the verdict says **"no gain over null"**
rather than dressing up noise as a discovery. The uniform-density control above
returns exactly that, which is how you know the win on the redundant cluster is
real.

## API

```python
from densityweight import (
    feature_density, temporal_redundancy, density_weights, effective_n,
    held_out, weighted_ridge,
)

dens = feature_density(X, k=15)        # high = crowded / redundant
w    = density_weights(dens, power=1)  # mean-1 weights ∝ 1/density
effective_n(w)                         # (Σw)² / Σw²  — distinct-row count

r = held_out(X, y, low_frac=0.6, k=15, n_null=200)
r["learned"], r["uniform"], r["lift"], r["beats_null"]
r["best_driver"], r["best_curve"], r["best_power"], r["effective_n_ratio"]
r["weights"]                           # the learned per-row train weights
```

## CLI

```bash
densityweight demo                       # redundant cluster + a uniform-density control
densityweight fit data.csv --target y    # effective-n + held-out null verdict on your CSV
```

## Pairs with

[`stablefit`](https://github.com/Amarel-Taylor-Scott/stablefit) (weights for
*stability* — which features and which inconsistent rows to trust). stablefit
asks *which rows are wrong*; densityweight asks *which rows are redundant* —
orthogonal levers, same held-out + null discipline throughout.

## Install

```bash
pip install numpy && git clone https://github.com/Amarel-Taylor-Scott/densityweight.git
```

MIT. Depends only on numpy.
