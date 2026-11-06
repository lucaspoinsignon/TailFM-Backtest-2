"""Step 2.  Price panel -> log returns, with every drop listed by VALOR.

    python 02_returns.py --data data/prices.csv --out data/returns.csv

Four filters, each removing series that break something downstream: non-positive
prices (log undefined), poor calendar coverage, a high zero-return fraction (stale or
coarsely quantised quotes), and too few exceedances for the GPD fits in tailfm.evt.

By default a date is kept only if every VALOR has a price on it (after ffill), so one
late or gappy series removes that date for all.  --fill keeps the dates of the complete
VALORs (coverage >= --core-coverage) and fills the missing returns of the others from
their most correlated complete VALORs (proxy_fill); <out>_filled.csv marks those entries.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def proxy_fill(lp, core, k=5, seed=0):
    """Log returns of every column of lp (log prices, NaN = missing; the `core` columns are
    complete).  A missing return of column j is a_j + b_j' r_P + e*: ridge regression of its
    observed returns on its k most correlated core columns P, e* resampled from its residuals.
    Inside a gap with prices on both sides the total return over the gap is observed, so
    the gap's returns are shifted equally to add up to it.  Returns (ret, filled, report)."""
    rng = np.random.default_rng(seed)
    X = np.diff(lp[core].to_numpy(), axis=0)                     # (T-1, c), complete
    ret = lp.diff().iloc[1:].copy()
    filled = pd.DataFrame(False, index=ret.index, columns=ret.columns)
    rep = []
    for c in lp.columns.difference(core, sort=False):
        p = lp[c].to_numpy()
        r = np.diff(p)
        obs = np.isfinite(r)
        Xo, yo = X[obs] - X[obs].mean(axis=0), r[obs] - r[obs].mean()
        corr = np.abs(Xo.T @ yo) / (np.sqrt((Xo ** 2).sum(axis=0) * (yo ** 2).sum()) + 1e-300)
        top = np.argsort(corr)[::-1][:k]
        G = Xo[:, top].T @ Xo[:, top]
        b = np.linalg.solve(G + 1e-3 * np.trace(G) / top.size * np.eye(top.size), Xo[:, top].T @ yo)
        res = yo - Xo[:, top] @ b                                       # mean zero
        fit = r[obs].mean() + (X[:, top] - X[obs][:, top].mean(axis=0)) @ b
        out = np.where(obs, r, fit + rng.choice(res, r.size))
        d = np.diff(np.concatenate([[0], np.isnan(p).astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):   # prices p[s:e] missing
            if s > 0 and e < p.size:                                          # p[s-1], p[e] observed
                out[s - 1:e] += (p[e] - p[s - 1] - out[s - 1:e].sum()) / (e - s + 1)
        ret[c] = out
        filled[c] = ~obs
        rep.append((c, (~obs).mean(), 1 - res.var() / r[obs].var(), [core[i] for i in top[:3]]))
    return ret, filled, rep


def report(cols, stat, why, unit=""):
    if len(cols) == 0:
        return
    print(f"\ndropped {len(cols)} VALOR -- {why}:")
    for c in cols:
        print(f"  {c:>12}  {stat[c]:.4g}{unit}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="price panel from 01_panel.py")
    p.add_argument("--out", required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--max-nonpos-frac", type=float, default=0.01,
                   help="above this fraction of non-positive cells, drop the column")
    p.add_argument("--min-coverage", type=float, default=0.98)
    p.add_argument("--ffill-limit", type=int, default=2)
    p.add_argument("--fill", action="store_true",
                   help="fill the missing returns of incomplete VALORs instead of dropping dates")
    p.add_argument("--core-coverage", type=float, default=0.98,
                   help="--fill: VALORs with this coverage fix the dates and serve as proxies")
    p.add_argument("--proxies", type=int, default=5, help="--fill: proxies per filled VALOR")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-zero-frac", type=float, default=0.05)
    p.add_argument("--q-tail", type=float, default=0.05,
                   help="must match train_tailfm.py --q-tail")
    p.add_argument("--min-exceedances", type=int, default=30)
    p.add_argument("--test-frac", type=float, default=0.2)
    a = p.parse_args()

    px = pd.read_csv(a.data, index_col=0, parse_dates=True).sort_index()
    px.columns = [str(c) for c in px.columns]
    if a.start:
        px = px[px.index >= pd.Timestamp(a.start)]
    if a.end:
        px = px[px.index <= pd.Timestamp(a.end)]
    print(f"{a.data}: {px.shape[0]} dates x {px.shape[1]} VALOR  "
          f"{px.index[0].date()} -> {px.index[-1].date()}")

    # ---- non-positive prices
    npf = (px <= 0).sum() / px.notna().sum()
    bad = list(px.columns[npf > a.max_nonpos_frac])
    report(bad, npf, f"non-positive price fraction > {a.max_nonpos_frac}")
    px = px.drop(columns=bad)
    iso = int((px <= 0).sum().sum())
    if iso:
        print(f"\nmasked {iso} isolated non-positive cells to NaN across "
              f"{int(((px <= 0).sum() > 0).sum())} VALOR")
        px = px.mask(px <= 0)

    # ---- coverage
    cov = px.notna().mean()
    bad = list(cov.index[cov < a.min_coverage])
    report(bad, cov, f"coverage < {a.min_coverage}")
    px = px.drop(columns=bad)
    if px.shape[1] == 0:
        raise SystemExit("--min-coverage dropped every column")

    before = len(px)
    if a.ffill_limit > 0:
        px = px.ffill(limit=a.ffill_limit)
    filled = None
    if a.fill:
        core = [c for c in px.columns if cov[c] >= a.core_coverage]
        px = px[px[core].notna().all(axis=1)]
        print(f"\nrows {before} -> {len(px)} after ffill(limit={a.ffill_limit}) + dates of the "
              f"{len(core)} VALOR with coverage >= {a.core_coverage}")
        ret, filled, rep = proxy_fill(np.log(px), core, a.proxies, a.seed)
        print(f"\nfilled {len(rep)} VALOR (missing returns from proxies; R2 = proxy fit on its own days):")
        print(f"  {'VALOR':>12}  {'filled':>6}  {'R2':>5}  top proxies")
        for c, fr, r2, top in sorted(rep, key=lambda x: -x[1]):
            print(f"  {c:>12}  {fr:6.1%}  {r2:5.2f}  {', '.join(top)}")
    else:
        px = px.dropna(how="any")
        print(f"\nrows {before} -> {len(px)} after ffill(limit={a.ffill_limit}) + dropna")
        ret = np.log(px / px.shift(1)).dropna(how="any")
    print(f"returns: {ret.shape[0]} x {ret.shape[1]}")

    # ---- stale quotes
    zf = (ret == 0.0).mean()
    bad = list(ret.columns[zf > a.max_zero_frac])
    report(bad, zf, f"zero-return fraction > {a.max_zero_frac}")
    ret = ret.drop(columns=bad)

    # ---- EVT feasibility, on the training rows the marginals are fitted to
    tr = ret.iloc[:int((1 - a.test_frac) * len(ret))]
    n_lo = (tr < tr.quantile(a.q_tail)).sum()
    n_hi = (tr > tr.quantile(1 - a.q_tail)).sum()
    worst = pd.concat([n_lo, n_hi], axis=1).min(axis=1)
    bad = list(ret.columns[worst < a.min_exceedances])
    report(bad, worst, f"fewer than {a.min_exceedances} exceedances in a tail",
           " obs")
    ret = ret.drop(columns=bad)
    if ret.shape[1] == 0:
        raise SystemExit("everything filtered out; loosen the thresholds")

    ret.index.name = "Date"
    ret.to_csv(a.out, float_format="%.8e")
    if filled is not None:
        path = os.path.splitext(a.out)[0] + "_filled.csv"
        filled[ret.columns].astype(np.int8).to_csv(path)
        print(f"\nfilled entries: {int(filled[ret.columns].to_numpy().sum())} of {ret.size} "
              f"({filled[ret.columns].to_numpy().mean():.1%}), marked 1 in {path}")
    T, f = ret.shape
    ntr = int((1 - a.test_frac) * T)
    print(f"\nwrote {a.out}: T={T}  f={f}  "
          f"{ret.index[0].date()} -> {ret.index[-1].date()}")
    print(f"train rows {ntr}  exceedances/tail {int(a.q_tail * ntr)}  "
          f"n*f at n=24 is {24 * f} vs {ntr // 24} non-overlapping windows")


if __name__ == "__main__":
    main()
