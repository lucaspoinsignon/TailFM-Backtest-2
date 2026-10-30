"""What-if without retraining: give an existing sample the market-factor asymmetry of its
training days, and change nothing else.

    python factor_whatif.py --data data/returns_clean.csv --gen runs/all090/generated_windows_u0.90.npy
    python factor_whatif.py --data data/returns_clean.csv --gen runs/uncond/generated_windows.npy

Works in the run's latent space (its marginals.pkl: every marginal t_nu, so the market
score below measures the copula only).  b = market direction of the run's training rows
(first principal direction), s = z.b the market score of a day, and
A = (q50 - q05) / (q95 - q50) its downside / upside ratio.
    target   A of the run's training windows (all their days), restricted to those
             starting in the sample's u-band for a conditional sample (sidecar band)
    repeat --iters times
      1. two-piece map of the generated s about its median, so that A equals the target
         while the median and q95 - q05 stay the same; z <- z + (s' - s) b
      2. every feature back on its own generated values by ranks, so each asset's
         distribution is exactly the model's
Only the dependence between assets moves, toward the target asymmetry.  The target uses
training days only, so applied to the fold runs of make_folds.py the backtest stays
out-of-sample (pooled_ci.py --suffix _fx).
Writes <gen>_fx.npy and its sidecar next to --gen.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from csvio import load_returns
from regime import read_sidecar, sidecar_path
from splits import Split
from tailfm.base import asymmetry, market_direction


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="the CSV the run was trained on")
    ap.add_argument("--gen", required=True, help="generated windows (M, n, f) of a run")
    ap.add_argument("--p", type=float, default=0.05, help="quantile level of A")
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()

    run = os.path.dirname(os.path.abspath(a.gen))
    marg = pickle.load(open(os.path.join(run, "marginals.pkl"), "rb"))
    split = Split.load(os.path.join(run, "split.json"))
    G = np.load(a.gen, mmap_mode="r")
    M, n, f = G.shape
    meta = read_sidecar(a.gen) or {}
    band = meta.get("band")

    # ------------------------------------------------ target from the training days
    z_rows = marg.transform(load_returns(a.data, False))
    b = market_direction(z_rows[split.train_row_mask()])
    starts = split.train_window_starts(n, 1)
    if band:
        u = np.load(os.path.join(run, "cond_u.npy"))
        starts = starts[(u[starts] >= band[0]) & (u[starts] <= band[1])]
    s_real = (z_rows @ b)[starts[:, None] + np.arange(n)].ravel()
    target = asymmetry(s_real, a.p)
    print(f"{a.gen}: {M} windows; target from {starts.size} training windows"
          + (f" starting with u in [{band[0]:.2f}, {band[1]:.2f}]" if band else "")
          + f": market-score A({a.p:g}) = {target:.3f}, A(0.01) = {asymmetry(s_real, 0.01):.3f}")

    # ------------------------------------------------ generated sample in latent space
    z = np.empty((M * n, f), dtype=np.float32, order="F")
    for j0 in range(0, f, 32):
        blk = np.asarray(G[:, :, j0:j0 + 32], dtype=float).reshape(M * n, -1)
        for j in range(blk.shape[1]):
            z[:, j0 + j] = marg.marginals_[j0 + j].transform(blk[:, j])
    zs = np.sort(z, axis=0)                           # each feature's generated values
    b32 = b.astype(np.float32)
    s = z @ b
    print(f"generated: A({a.p:g}) = {asymmetry(s, a.p):.3f}, A(0.01) = {asymmetry(s, 0.01):.3f}")

    for it in range(a.iters):
        lo, med, hi = np.quantile(s, [a.p, 0.5, 1 - a.p])
        lo_t, hi_t = med - (hi - lo) * target / (1 + target), med + (hi - lo) / (1 + target)
        d = (np.where(s < med, (s - med) * ((med - lo_t) / (med - lo) - 1),
                      (s - med) * ((hi_t - med) / (hi - med) - 1))).astype(np.float32)
        for j in range(f):                            # 1. market score, 2. marginals back
            col = z[:, j] + d * b32[j]
            z[np.argsort(col, kind="stable"), j] = zs[:, j]
        s = z @ b
        print(f"  iteration {it + 1}: A({a.p:g}) = {asymmetry(s, a.p):.3f}, "
              f"A(0.01) = {asymmetry(s, 0.01):.3f}   (target {target:.3f})")
    del zs

    # ------------------------------------------------ back to returns: the model's own values
    out = np.empty(G.shape, dtype=G.dtype)
    for j0 in range(0, f, 32):
        blk = np.asarray(G[:, :, j0:j0 + 32]).reshape(M * n, -1)
        for j in range(blk.shape[1]):
            col = np.empty(M * n, dtype=G.dtype)
            col[np.argsort(z[:, j0 + j], kind="stable")] = np.sort(blk[:, j])
            out[:, :, j0 + j] = col.reshape(M, n)
    path = os.path.splitext(a.gen)[0] + "_fx.npy"
    np.save(path, out)
    with open(sidecar_path(path), "w") as fh:
        json.dump(dict(meta, factor_whatif=dict(source=os.path.basename(a.gen), p=a.p,
                                                target=target, iters=a.iters)), fh, indent=2)
    print(f"saved {path} (+ sidecar)")


if __name__ == "__main__":
    main()
