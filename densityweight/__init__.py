"""densityweight — sample weighting that corrects for redundancy / density.

In a fine time series or clustered table, near-duplicate rows over-count: the
effective sample size is far below ``n`` and a uniform fit is dominated by the
dense regions, biasing it. densityweight scores each row's local density and
downweights crowded rows (weight ∝ 1/density), so the model is driven by the
*distinct* information instead of the redundant copies — and the gain is
validated on a held-out split against a magnitude-matched null. numpy-only.

    from densityweight import held_out
    from densityweight.synth import make_redundant_cluster
    X, y, info = make_redundant_cluster()        # rows in time order
    r = held_out(X, y)
    print(r["uniform"], r["learned"], r["beats_null"], r["effective_n_ratio"])
"""

from __future__ import annotations

from . import synth
from .core import pearson, weighted_ridge
from .curves import CURVES, normalize, inverse, inverse_sqrt, rank_low
from .density import (density_weights, effective_n, feature_density,
                      temporal_redundancy)
from .validate import held_out

__all__ = [
    "synth", "pearson", "weighted_ridge",
    "CURVES", "normalize", "inverse", "inverse_sqrt", "rank_low",
    "density_weights", "effective_n", "feature_density", "temporal_redundancy",
    "held_out",
]
__version__ = "0.1.0"
