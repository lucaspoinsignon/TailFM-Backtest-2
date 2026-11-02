"""Are the generated money-market VALORs good?  Real vs generated, VALOR by VALOR.

    python mm_check.py --data data/returns_mm.csv --gen runs/mm090/generated_windows_u0.90.npy
    python mm_check.py --data data/returns_mm.csv --gen runs/mm_uncond/generated_windows.npy

--data is the CSV of 02_returns.py --mm-data, whose money-market columns hold the
innovation eps = r - carry (money_market.py); --gen a sample of a run trained on it.  The
real windows are chosen as in stylized_facts.py (training windows; for a conditional
sample those starting in its u-band; the selection flags override).

Table, per modelled VALOR (real, generated, '*' when the 90% interval of gen - real
excludes 0; real intervals from a block bootstrap in calendar time, generated iid):
    carry %/yr       median carry of the real windows, annualised (context, not tested)
    q0.1 .. q99.9    quantiles of eps on day 1, in bp (the outer two show whether the
                     generated tails stay plausible)
    quiet %          share of days with |eps| < 0.01 bp: nothing but the carry
    ACF lag 1        rank autocorrelation of eps within windows
    corr eq          rank correlation of eps with the equal-weight equity portfolio, day 1
    co-crash %       share of the VALOR's worst-5% days that are also the equities'
                     worst-5% days (5 if unrelated, 100 if they always fall together)
    24d q05, q95     quantiles of eps summed over the window, in bp
About 10% of cells are flagged by chance.  Statistics driven by long regimes (rate
levels, crises) vary more between histories than the bootstrap shows, so read isolated
flags with caution; a VALOR flagged on most columns is a real misfit.

Figures, next to --gen:
    mm_check[_<selection>]_qq.png     QQ of day-1 eps per VALOR, generated vs real, with
                                      the pointwise 90% band of the real quantile
    mm_check[_<selection>]_paths.png  real price path over the window (cumulative log
                                      return, bp) against the fan of carry + generated eps
                                      (median, 50% and 90%), at three window starts spread
                                      over the real windows, for --assets (default: the six
                                      VALORs with the widest real eps); dashed = carry only,
                                      held at its value at the start
Carry-only VALORs (r = carry, nothing to generate) are listed.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csvio import feature_names_from_csv, load_returns
from money_market import carry_path, read_meta
from regime import (add_selection_args, describe_selection, load_dates, read_sidecar,
                    real_window_starts, resolve_selection, selection_tag)
from run_logging import tee_output
from splits import add_split_args, add_split_source_args, require_split
from stylized_facts import PGRID, Ranker, block_indices, bootstrap, ci, diff_flag, lagcorr

COLS = [("q001", "q0.1 bp"), ("q01", "q1 bp"), ("q05", "q5 bp"), ("q95", "q95 bp"),
        ("q99", "q99 bp"), ("q999", "q99.9 bp"), ("quiet", "quiet %"), ("acf1", "ACF lag 1"), ("rho_eq", "corr eq"),
        ("cocrash", "co-crash %"), ("s05", "24d q05 bp"), ("s95", "24d q95 bp")]


class MMStats:
    """Statistics of one VALOR's innovation windows E (N, n), with the equity portfolio's
    day-1 return P1 (N,), on the sample and on row-resamples (ranks recomputed)."""

    def __init__(self, E, P1):
        self.x1, self.s = E[:, 0], E.sum(axis=1)
        self.acf = Ranker(E, True)
        self.pair = Ranker(np.column_stack([E[:, 0], P1]), False)

    def stats(self, idx):
        x1 = self.x1[idx]
        q = np.quantile(x1, [0.001, 0.01, 0.05, 0.95, 0.99, 0.999]) * 1e4
        s05, s95 = np.quantile(self.s[idx], [0.05, 0.95]) * 1e4
        R, U = self.acf(idx), self.pair(idx)
        return dict(q001=q[0], q01=q[1], q05=q[2], q95=q[3], q99=q[4], q999=q[5],
                    quiet=100 * np.mean(np.abs(x1) < 1e-6), acf1=lagcorr(R, R, 1),
                    rho_eq=float(np.corrcoef(U.T)[0, 1]),
                    cocrash=100 * np.mean((U[:, 0] <= 0.05) & (U[:, 1] <= 0.05)) / 0.05,
                    s05=s05, s95=s95)


def run(args):
    rng = np.random.default_rng(args.seed)
    gen_path = args.gen.rpartition("=")[2]
    G = np.load(gen_path, mmap_mode="r")
    r = load_returns(args.data, False)
    T, f = r.shape
    names = feature_names_from_csv(args.data, f)
    meta = read_meta(args.data)
    if meta is None:
        raise SystemExit(f"no money-market metadata next to {args.data}: "
                         "run 02_returns.py --mm-data first")
    if G.shape[2] != f:
        raise SystemExit(f"{gen_path} has {G.shape[2]} features, {args.data} has {f}")
    carry = pd.read_csv(carry_path(args.data), index_col=0)
    if len(carry) != T:
        raise SystemExit(f"{carry_path(args.data)} has {len(carry)} rows, the data {T}")
    mm = [c for c in meta["mm_model"] if c in names]
    eq = [j for j, c in enumerate(names) if c not in set(meta["mm_model"])]

    split = require_split([args.gen], args, T, data=os.path.abspath(args.data),
                          allow_missing=args.allow_missing_split)
    n = G.shape[1]
    periods, excluded, band, holdout = resolve_selection(args, [read_sidecar(gen_path) or {}])
    u = np.load(os.path.join(os.path.dirname(os.path.abspath(gen_path)), "cond_u.npy")) if band else None
    dates = load_dates(args.data, T)
    starts = real_window_starts(split, n, 1, dates, periods, excluded, band, u, holdout)
    if starts.size < 50:
        raise SystemExit(f"only {starts.size} real windows match the selection")
    rows = starts[:, None] + np.arange(n)
    dd = np.r_[1.0, np.diff(dates).astype("timedelta64[D]").astype(float)]
    print(f"generated {G.shape} vs {starts.size} real {n}-day windows, "
          f"{describe_selection(periods, excluded, band, holdout)}; "
          f"{len(mm)} modelled money-market VALOR, {len(meta['mm_carry_only'])} carry only")

    idx_r = [block_indices(starts, args.block, rng) for _ in range(args.n_boot)]
    idx_g = [rng.integers(0, len(G), len(G)) for _ in range(args.n_boot)]
    G1 = np.asarray(G[:, 0, :], dtype=float)
    P1r, P1g = r[starts][:, eq].mean(axis=1), G1[:, eq].mean(axis=1)
    res, Eg = {}, {}
    for c in mm:
        j = names.index(c)
        Eg[c] = np.asarray(G[:, :, j], dtype=float)
        pr, rr = bootstrap(MMStats(r[rows, j], P1r).stats, starts.size, idx_r)
        pg, rg = bootstrap(MMStats(Eg[c], P1g).stats, len(G), idx_g)
        fl = {k: diff_flag(rr[k], rg[k])[2] for k, _ in COLS}
        cy = np.median(carry[c].to_numpy()[starts] / dd[starts]) * 365 * 100
        res[c] = (pr, rr, pg, fl, cy)

    # ---------------------------------------------------------------- table
    w = 17
    for title, keys in (("innovation eps, day 1", COLS[:7]),
                        ("dynamics, link to equities, 24-day sums", COLS[7:])):
        print(f"\n{title}  (real  gen, * = 90% interval of gen - real excludes 0)")
        print(f"  {'VALOR':>10} {'carry %/yr':>10} " + "".join(f"{lab:>{w}}" for _, lab in keys)
              + (f"{'flags':>7}" if keys[0][0] == "acf1" else ""))
        for c in mm:
            pr, _, pg, fl, cy = res[c]
            cells = "".join(f"{pr[k]:8.3f}{pg[k]:8.3f}{'*' if fl[k] else ' '}" for k, _ in keys)
            print(f"  {c:>10} {cy:10.2f} {cells}"
                  + (f"{sum(fl.values()):4d}/{len(COLS)}" if keys[0][0] == "acf1" else ""))
    tot = sum(sum(v[3].values()) for v in res.values())
    print(f"\n{tot} of {len(mm) * len(COLS)} cells flagged (about 10% by chance); a VALOR "
          "flagged on most columns is a real misfit")
    if meta["mm_carry_only"]:
        print("carry only (r = carry in every scenario, nothing to generate): "
              + ", ".join(meta["mm_carry_only"]))

    # ---------------------------------------------------------------- figures
    tag = selection_tag(band, periods, excluded, holdout)
    stem = os.path.join(args.outdir or os.path.dirname(os.path.abspath(gen_path)),
                        "mm_check" + (f"_{tag}" if tag else ""))
    nc = 4
    nr = -(-len(mm) // nc)
    fig, ax = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.4 * nr), squeeze=False)
    for a, c in zip(ax.ravel(), mm):
        xr, xg = r[starts, names.index(c)] * 1e4, Eg[c][:, 0] * 1e4
        qr, qg = np.quantile(xr, PGRID), np.quantile(xg, PGRID)
        lo, hi = np.quantile([np.quantile(xr[i], PGRID) for i in idx_r], [0.05, 0.95], axis=0)
        a.fill_between(qr, lo, hi, color="0.85")
        a.plot(qr, qg, ".", ms=2.5, color="C0")
        lim = [min(qr.min(), qg.min()), max(qr.max(), qg.max())]
        a.plot(lim, lim, color="0.4", lw=1)
        a.set_title(f"{c}  ({sum(res[c][3].values())}/{len(COLS)} flagged)", fontsize=9)
    for a in ax.ravel()[len(mm):]:
        a.axis("off")
    fig.suptitle("money-market innovations, day 1 (bp): generated vs real quantiles "
                 "(grey: 90% band of the real quantile)")
    fig.supxlabel("real quantile (bp)")
    fig.supylabel("generated quantile (bp)")
    fig.tight_layout()
    fig.savefig(stem + "_qq.png", dpi=110)
    plt.close(fig)

    show = args.assets or sorted(mm, key=lambda c: res[c][0]["q01"] - res[c][0]["q99"])[:6]
    picks = starts[[int(0.1 * starts.size), starts.size // 2, int(0.9 * starts.size)]]
    fig, ax = plt.subplots(len(show), 3, figsize=(14, 2.8 * len(show)), squeeze=False)
    days = np.arange(1, n + 1)
    for i, c in enumerate(show):
        j = names.index(c)
        cj = carry[c].to_numpy()
        for a, s in zip(ax[i], picks):
            real = np.cumsum(r[s:s + n, j] + cj[s:s + n]) * 1e4
            proj = cj[s] / dd[s] * dd[s:s + n]                      # accrual held at the start
            fan = np.percentile(np.cumsum(proj + Eg[c], axis=1) * 1e4, [5, 25, 50, 75, 95], axis=0)
            a.fill_between(days, fan[0], fan[4], color="C0", alpha=0.15, label="generated 90%")
            a.fill_between(days, fan[1], fan[3], color="C0", alpha=0.3, label="generated 50%")
            a.plot(days, fan[2], color="C0", lw=1, label="generated median")
            a.plot(days, np.cumsum(proj) * 1e4, "--", color="0.5", lw=1, label="carry only")
            a.plot(days, real, color="k", lw=1.4, label="real")
            a.set_title(f"{c}, start {str(dates[s])[:10]}", fontsize=9)
    ax[0, 0].legend(fontsize=7, loc="upper left")
    fig.supxlabel("day in window")
    fig.supylabel("cumulative log return (bp)")
    fig.suptitle("money market: real path vs carry + generated innovations")
    fig.tight_layout()
    fig.savefig(stem + "_paths.png", dpi=110)
    plt.close(fig)
    print(f"\nSaved: {stem}_qq.png, {stem}_paths.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="CSV of 02_returns.py --mm-data")
    ap.add_argument("--gen", required=True, help="generated windows .npy (M, n, f)")
    ap.add_argument("--assets", nargs="+", default=None,
                    help="money-market VALORs in the path figure (default: six widest)")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--block", type=int, default=250,
                    help="block length in rows of the real-data bootstrap")
    ap.add_argument("--seed", type=int, default=0)
    add_split_args(ap)
    add_split_source_args(ap)
    add_selection_args(ap, "real windows")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    gen_path = args.gen.rpartition("=")[2]
    periods, excluded, band, holdout = resolve_selection(args, [read_sidecar(gen_path) or {}])
    tag = selection_tag(band, periods, excluded, holdout)
    log = args.log or os.path.join(args.outdir or os.path.dirname(os.path.abspath(gen_path)),
                                   "mm_check" + (f"_{tag}" if tag else "") + ".log")
    with tee_output(log, header="mm_check.py"):
        run(args)
    print(f"Terminal report saved to {log}")


if __name__ == "__main__":
    main()
