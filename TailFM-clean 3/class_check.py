"""Are the awkward VALORs generated correctly?  Real vs generated, one VALOR per row.

    python class_check.py --data data/returnscomp.csv --gen runs/all/generated_windows.npy \\
        --screen fig/screen/screen.csv --pairs fig/screen/duplicates.txt

VALORs: --valors A,B,... and / or every BAD / WARN row of --screen (screen_valors.py).
Day 1 of each generated window against every real day (an unconditional train-all run):

    P<0, P=0     loss frequency and atom at zero (stale days, accruals that never fall)
    q0.1 .. q99.9, min, max    quantiles; gen min / max far outside the real ones =
                 extrapolation the series does not support
Whole 24-day windows, against every real window:
    level        median of the window mean (an accrual's rate level), and its 10-90% range
    wsd          median within-window sd (an accrual: tiny next to the level)
    acf1         lag-1 autocorrelation pooled within windows (smoothed / stale pricing)
--pairs (duplicates.txt of screen_valors.py, "B  # A (corr ...)"): day-1 corr(A, B) and the
tracking error sd(B - beta A), beta the real OLS slope, real vs generated.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd


def acf1(W):
    W = W - W.mean(axis=1, keepdims=True)
    den = (W ** 2).sum()
    return float((W[:, 1:] * W[:, :-1]).sum() / den) if den > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--valors", default="")
    ap.add_argument("--screen", default=None, help="screen.csv: its BAD and WARN VALORs")
    ap.add_argument("--pairs", default=None, help="duplicates.txt of screen_valors.py")
    a = ap.parse_args()

    df = pd.read_csv(a.data, index_col=0)
    names = [str(c) for c in df.columns]
    R = df.to_numpy(dtype=float)
    g = np.load(a.gen, mmap_mode="r")
    M, n, f = g.shape
    if f != R.shape[1]:
        raise SystemExit(f"{a.gen} has {f} VALORs, {a.data} has {R.shape[1]}")
    vals = [s.strip() for s in a.valors.split(",") if s.strip()]
    if a.screen:
        s = pd.read_csv(a.screen, dtype={"valor": str})
        vals += [v for v in s.valor[s.status != "ok"] if v not in vals]
    Wr = np.lib.stride_tricks.sliding_window_view(R, n, axis=0)        # (T-n+1, f, n)

    q = [0.001, 0.01, 0.5, 0.99, 0.999]
    print(f"{M} generated windows vs {R.shape[0]} real days / {Wr.shape[0]} real windows\n")
    print(f"{'valor':>10s} {'':4s} {'P<0':>6s} {'P=0':>6s} " + " ".join(f"{'q' + str(100 * p):>9s}" for p in q)
          + f" {'min':>9s} {'max':>9s} | {'level':>9s} {'[10%':>9s} {'90%]':>9s} {'wsd':>8s} {'acf1':>6s}")
    for v in vals:
        j = names.index(v)
        x1, gj = R[:, j], np.asarray(g[:, :, j], dtype=float)
        for tag, x, W in (("real", x1, Wr[:, j, :]), ("gen", gj[:, 0], gj)):
            lv = W.mean(axis=1)
            print(f"{v if tag == 'real' else '':>10s} {tag:4s} {np.mean(x < 0):6.3f} {np.mean(x == 0):6.3f} "
                  + " ".join(f"{e:+9.2e}" for e in np.quantile(x, q))
                  + f" {x.min():+9.2e} {x.max():+9.2e} | {np.median(lv):+9.2e} "
                  + " ".join(f"{e:+9.2e}" for e in np.quantile(lv, [0.1, 0.9]))
                  + f" {np.median(W.std(axis=1)):8.2e} {acf1(W):+6.2f}")
    if a.pairs:
        print(f"\n{'pair':>22s} {'corr real':>10s} {'gen':>7s} {'TE real':>9s} {'gen':>9s} {'ratio':>6s}")
        for ln in open(a.pairs):
            m = re.match(r"\s*(\S+)\s*#\s*(\S+)", ln)
            if not m or m[1] not in names or m[2] not in names:
                continue
            b, c = names.index(m[1]), names.index(m[2])          # B = later column, A = partner
            xa, xb = R[:, c], R[:, b]
            beta = np.cov(xa, xb)[0, 1] / xa.var()
            ga, gb = np.asarray(g[:, 0, c], dtype=float), np.asarray(g[:, 0, b], dtype=float)
            te_r, te_g = np.std(xb - beta * xa), np.std(gb - beta * ga)
            print(f"{m[2] + '/' + m[1]:>22s} {np.corrcoef(xa, xb)[0, 1]:10.4f} {np.corrcoef(ga, gb)[0, 1]:7.4f} "
                  f"{te_r:9.2e} {te_g:9.2e} {te_g / te_r:6.2f}")


if __name__ == "__main__":
    main()
