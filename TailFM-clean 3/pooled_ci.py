"""Pooled out-of-sample c-scan over the folds written by make_folds.py.

    python pooled_ci.py                                              # each fold's own u0.90 model
    python pooled_ci.py --level 0.80                                 # wrong-regime check
    python pooled_ci.py --same runs/u090/generated_windows_u0.90.npy # same days, in-sample model
    python pooled_ci.py --suffix _fx                                 # each fold's _fx sample
    python pooled_ci.py --folds folds_skew.json                      # another fold set

Every band day is scored by the model of its fold, which was not trained on it, with the
scenarios as generated.  For each scale c the model law of fold k is c * P_k, and
    PF95 / PF99    Kupiec on the exceptions summed over the folds,
    Z2 97.5 / 99   Z2 = 1 + sum_t X_t I_t / (N alpha ES_k(t)), null simulated fold by fold.
Exceptions and Z2 are scale-invariant, so c * P_k on x equals P_k on x / c.
"""
import argparse
import json
import os

import numpy as np

from backtest import PredictivePnL, pf_test
from csvio import load_returns
from regime import load_dates, period_mask

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/returns_clean.csv")
ap.add_argument("--folds", default="folds.json")
ap.add_argument("--level", type=float, default=0.9)
ap.add_argument("--same", default=None, help="score every fold with this one sample instead")
ap.add_argument("--suffix", default="", help="score generated_windows_u<level><suffix>.npy")
ap.add_argument("--n-sim", type=int, default=5000)
a = ap.parse_args()

cfg = json.load(open(a.folds))
lo, hi = cfg["band"]
r = load_returns(a.data, False)
dates = load_dates(a.data, len(r))
w = np.full(r.shape[1], 1 / r.shape[1])

xs, laws = [], []
for f in cfg["folds"]:
    gen = a.same or os.path.join(f["run"], f"generated_windows_u{a.level:.2f}{a.suffix}.npy")
    u = np.load(os.path.join(os.path.dirname(gen), "cond_u.npy"))     # the model's own u
    m = (u >= lo) & (u <= hi) & period_mask(dates, f["test_periods"])
    y = np.load(gen, mmap_mode="r")[:, 0, :] @ w
    xs.append(r[m] @ w)
    laws.append(PredictivePnL(y))
    print(f"{f['run']}: {m.sum()} band days in {len(f['test_periods'])} test ranges")
N = sum(x.size for x in xs)


def exc(c, al):
    return sum(int((x / c + P.var(al) < 0).sum()) for x, P in zip(xs, laws))


def tail_sum(X, P, al):
    return (X * (X + P.var(al) < 0)).sum(-1) / P.es(al)


rng = np.random.default_rng(0)
null = {al: 1 + sum(tail_sum(P.rvs((a.n_sim, x.size), rng), P, al) for x, P in zip(xs, laws))
        / (N * al) for al in (0.025, 0.01)}
print(f"\npooled N = {N} out-of-sample band days | exceptions at c = 1: "
      f"95%: {exc(1, 0.05)} (expected {0.05 * N:.1f}), 99%: {exc(1, 0.01)} (expected {0.01 * N:.1f})")
print("   c   PF95   PF99  Z2_97.5  Z2_99")
for c in np.arange(0.70, 1.41, 0.05):
    p = [pf_test(exc(c, al), N, al)["p_value"] for al in (0.05, 0.01)]
    for al in (0.025, 0.01):
        z = 1 + sum(tail_sum(x / c, P, al) for x, P in zip(xs, laws)) / (N * al)
        p.append(np.mean(null[al] < z))
    print(f"{c:4.2f}  " + "  ".join(f"{q:.3f}" for q in p))
