"""Does the generator treat the positions of a window alike?

    python position_check.py --data data/returns_clean.csv --gen runs/clean_data/generated_windows.npy
    python position_check.py ... --pairs 4155867:4157572,4155894:4157300

Real windows are position-invariant by construction (every day sits at every position), so
any profile over the positions k = 0..n-1 of a generated sample is an artifact of the model,
and k = 0 is the day every 1-day score (generate.py table, evaluate.py --horizon 1,
run_backtest.py --horizon 1) uses.  Per position, against all real days:

    VaR95, VaR99, IQR   median over assets of gen / real (assets with real VaR95 < 2bp left out)
    rho                 mean pairwise rank correlation of the cross-section
    lamL                mean pairwise lambda_L(5%) = E[B (B - 1)] / (f (f - 1) q), B = # assets
                        below their 5% rank on that day (as stylized_facts.py)
    --pairs             lambda_L(5%) of chosen pairs
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def cross_section(X, q, pairs):
    """(rho, lamL, [lamL of each pair]) of the rows of X (N, f)."""
    U = rankdata(X, axis=0) / (X.shape[0] + 1)
    keep = U.std(axis=0) > 0
    V = U[:, keep]
    p = V.shape[1] * (V.shape[1] - 1)
    Z = (V - V.mean(axis=0)) / V.std(axis=0)
    rho = ((Z.sum(axis=1) ** 2 - (Z ** 2).sum(axis=1)).mean()) / p
    b = (V <= q).sum(axis=1)
    lam = (b * (b - 1.0)).mean() / (p * q)
    return rho, lam, [np.mean((U[:, i] <= q) & (U[:, j] <= q)) / q for i, j in pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="returns CSV the run was trained on")
    ap.add_argument("--gen", required=True, help="generated_windows*.npy, (M, n, f)")
    ap.add_argument("--pairs", default="", help="VALOR pairs A:B,C:D for lambda_L by position")
    ap.add_argument("--q", type=float, default=0.05)
    a = ap.parse_args()

    r = pd.read_csv(a.data, index_col=0)
    names = [str(c) for c in r.columns]
    R = r.to_numpy(dtype=float)
    g = np.load(a.gen, mmap_mode="r")
    M, n, f = g.shape
    if f != R.shape[1]:
        raise SystemExit(f"{a.gen} has {f} assets, {a.data} has {R.shape[1]}")
    pairs = [tuple(names.index(v) for v in s.split(":")) for s in a.pairs.split(",") if s]

    stats = lambda X: (-np.quantile(X, 0.05, axis=0), -np.quantile(X, 0.01, axis=0),
                       np.subtract(*np.quantile(X, [0.75, 0.25], axis=0)))
    v95r, v99r, iqr_r = stats(R)
    ok = v95r >= 2e-4
    rho_r, lam_r, pr = cross_section(R, a.q, pairs)

    head = "".join(f"  {s.replace(':', '/'):>17s}" for s in a.pairs.split(",") if s)
    print(f"{f} assets ({ok.sum()} with real VaR95 >= 2bp), {M} generated windows of {n} days\n")
    print(f"{'pos':>4s} {'VaR95':>7s} {'VaR99':>7s} {'IQR':>7s} {'rho':>7s} {'lamL':>7s}{head}")
    print(f"{'real':>4s} {1:7.3f} {1:7.3f} {1:7.3f} {rho_r:7.3f} {lam_r:7.3f}"
          + "".join(f"  {x:17.3f}" for x in pr))
    for k in range(n):
        X = np.asarray(g[:, k, :], dtype=float)
        v95, v99, iqr = stats(X)
        rho, lam, pg = cross_section(X, a.q, pairs)
        print(f"{k:4d} {np.median(v95[ok] / v95r[ok]):7.3f} {np.median(v99[ok] / v99r[ok]):7.3f} "
              f"{np.median(iqr[ok] / iqr_r[ok]):7.3f} {rho:7.3f} {lam:7.3f}"
              + "".join(f"  {x:17.3f}" for x in pg))
    print("\nreal windows are identical at every position: a profile in any column is the model's")


if __name__ == "__main__":
    main()
