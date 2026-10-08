"""Is the backtest valid on THIS data?  Placebo test with historical simulation (HS).

For many random splits (same design as train_tailfm.py: stratified random blocks of
2*horizon rows, 20% held out), take HS = the training windows themselves as the
predictive law and backtest it on that split's held-out blocks with the repo's own
Kupiec / Acerbi-Szekely code.  No model is trained.

If the tests were valid for an unconditional law on this sample, HS would be rejected
in roughly 5% of the splits (a bit more, since HS is estimated from finite data).  A
rejection rate far above that means the rejections are produced by the data and the
test design (regime clustering, which crises land in the test blocks), not by any
generator.

    python placebo_hs.py --data data/returns_clean.csv --splits 50
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from csvio import load_returns
from splits import block_split
from backtest import (predictive_from_windows, realized_pnl, pf_test,
                      test_z1, test_z2, test_z3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/returns_clean.csv")
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--n", type=int, default=24, help="window length (as in training)")
    ap.add_argument("--horizon", type=int, default=10, help="split horizon (as in training)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--splits", type=int, default=50, help="number of random splits")
    ap.add_argument("--n-sim", type=int, default=2000)
    a = ap.parse_args()

    r = load_returns(a.data, a.prices)
    T, f = r.shape
    w = np.full(f, 1.0 / f)
    var_lv, es_lv = (0.95, 0.99), (0.975, 0.99)
    tests = [f"PF {lv:g}" for lv in var_lv] + \
            [f"{z} {lv:g}" for lv in es_lv for z in ("Z1", "Z2", "Z3")]
    p = {h: {k: [] for k in tests} for h in (1, a.horizon)}
    z = {h: {k: [] for k in tests[2:]} for h in (1, a.horizon)}

    t0 = time.time()
    for seed in range(a.splits):
        sp = block_split(T, a.n, a.horizon, test_frac=a.test_frac, seed=seed,
                         data=os.path.abspath(a.data))
        win = sp.train_windows(r, a.n)
        for h in (1, a.horizon):
            x = realized_pnl(sp.test_windows(r, h), w, h)
            hs = predictive_from_windows(win, w, h)          # exactly what make_hs.py scores
            for lv in var_lv:
                n_exc = int(((x + hs.var(1 - lv)) < 0).sum())
                p[h][f"PF {lv:g}"].append(pf_test(n_exc, x.size, 1 - lv)["p_value"])
            for lv in es_lv:
                for name, fn in (("Z1", test_z1), ("Z2", test_z2), ("Z3", test_z3)):
                    res = fn(x, hs, 1 - lv, n_sim=a.n_sim, seed=seed)
                    p[h][f"{name} {lv:g}"].append(res.p_value if res.defined else np.nan)
                    z[h][f"{name} {lv:g}"].append(res.statistic)
        if seed == 0:
            print(f"split seed 0 is the split train_tailfm.py uses by default; its id "
                  f"should match your run's split.json:\n  {sp.summary()}")
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{a.splits} splits ({time.time() - t0:.0f}s)")

    for h in (1, a.horizon):
        print(f"\n=== h = {h}: HS backtested on its own held-out blocks, "
              f"{a.splits} random splits ===")
        print(f"{'test':>10s} {'reject@5%':>10s} {'median Z':>10s} {'5%-q Z':>8s} {'seed 0 Z':>9s}")
        for k in tests:
            pv = np.array(p[h][k], dtype=float)
            rate = np.nanmean(pv < 0.05)
            if k in z[h]:
                zz = np.array(z[h][k], dtype=float)
                print(f"{k:>10s} {rate:10.0%} {np.nanmedian(zz):+10.3f} "
                      f"{np.nanquantile(zz, 0.05):+8.3f} {zz[0]:+9.3f}")
            else:
                print(f"{k:>10s} {rate:10.0%} {'':>10s} {'':>8s} {'':>9s}")
    print("\nValid test: reject@5% close to 5% for every row.  Far above 5%: the backtest "
          "rejects\nthe data's own distribution, so it cannot judge an unconditional "
          "generator on this sample.")


if __name__ == "__main__":
    main()
