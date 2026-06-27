"""``densityweight`` CLI — a transparent demo and fit-your-own-CSV.

    densityweight demo                       # redundant cluster + a uniform control
    densityweight fit data.csv --target y    # effective-n, held-out + null verdict
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .synth import make_redundant_cluster, make_uniform_density
from .validate import held_out


def _verdict(res) -> str:
    return ("REAL — beats held-out null" if res["beats_null"]
            else "no gain over null (the honest answer)")


def _show(res, n_low):
    print("  effective sample size: %.0f / %d rows  (ratio %.3f — redundancy %s)"
          % (res["effective_n_ratio"] * n_low, n_low, res["effective_n_ratio"],
             "collapses n" if res["effective_n_ratio"] < 0.85 else "≈ none"))
    print("  weighting chosen     : %s / %s  (power %.1f)"
          % (res["best_driver"], res["best_curve"], res["best_power"]))
    print("  held-out transfer    : %.4f   (uniform %.4f, lift %+.4f)"
          % (res["learned"], res["uniform"], res["lift"]))
    print("  null p95 / p         : %.4f / %.3f" % (res["null_p95"], res["null_p"]))
    print("  VERDICT              : %s" % _verdict(res))


def cmd_demo(a) -> int:
    print("densityweight downweights rows in crowded neighborhoods (weight ∝")
    print("1/local_density) so redundant near-duplicates stop dominating the fit.")
    print("Every result is validated on a held-out split vs a magnitude-matched")
    print("null (the same weights permuted across rows).\n")

    print("## 1. Redundant cluster (a dense near-duplicate blob over-represented in train)")
    X, y, info = make_redundant_cluster(seed=a.seed)
    r = held_out(X, y, n_null=a.n_null, seed=0)
    n_low = len(r["weights"])
    _show(r, n_low)
    w = r["weights"]
    dense = info["dense_idx"]
    dense = dense[dense < n_low]
    base_mask = np.ones(n_low, bool)
    base_mask[dense] = False
    print("  -> mean weight on dense/redundant rows %.2f vs base rows %.2f (dense suppressed)\n"
          % (float(w[dense].mean()), float(w[base_mask].mean())))

    print("## 2. Uniform density control (no redundancy — must NOT beat null)")
    Xc, yc, _ = make_uniform_density(seed=a.seed)
    rc = held_out(Xc, yc, n_null=a.n_null, seed=0)
    _show(rc, len(rc["weights"]))
    print()
    return 0


def _load_csv(path, target):
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = [row for row in rdr if row]
    if target not in header:
        raise SystemExit("densityweight: target %r not found" % target)
    ti = header.index(target)
    fi = [i for i in range(len(header)) if i != ti]
    X = np.array([[float(row[i]) for i in fi] for row in rows], float)
    y = np.array([float(row[ti]) for row in rows], float)
    return X, y


def cmd_fit(a) -> int:
    X, y = _load_csv(a.file, a.target)
    print("# %s: %d rows x %d feats (rows assumed time-ordered)\n" % (a.file, *X.shape))
    r = held_out(X, y, low_frac=a.low_frac, k=a.k, n_null=a.n_null, lam=a.lam, seed=0)
    _show(r, len(r["weights"]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="densityweight", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="redundant cluster + uniform-density control")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--n-null", type=int, default=200, dest="n_null")
    d.set_defaults(fn=cmd_demo)
    p = sub.add_parser("fit", help="density-weight + held-out null verdict on your CSV")
    p.add_argument("file")
    p.add_argument("--target", required=True)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--low-frac", type=float, default=0.6, dest="low_frac")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--n-null", type=int, default=200, dest="n_null")
    p.set_defaults(fn=cmd_fit)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
