"""Stylized facts of generated vs real returns on the model's 24-day windows, with
uncertainty intervals for both sides and for their difference.

    python stylized_facts.py --data data/returns_clean.csv --gen runs/uncond/generated_windows.npy
    python stylized_facts.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy
    python stylized_facts.py ... --assets 4155001          # also single columns, e.g. MSCI World

Reference = the real windows chosen as in evaluate.py (default: training windows; for a
conditional sample those whose first day has u in the sample's band; --no-band /
--cond-level / --cond-band / --holdout / --test-period / --exclude-period override).
Day-1 facts use the first day of each window (each real day once); dynamic facts use
whole windows.  Series: the equal-weight portfolio (--weights), each --assets column, and
a summary over all assets.

Why these statistics.  The Hill indices of these returns are about 2-3, so E|r|^p is
infinite for p >= alpha: skewness (p = 3) and kurtosis (p = 4) do not exist, and their
sample versions do not converge; they grow with the sample size, so 20,000 generated
days cannot be compared with 5,000 real ones.  Every statistic below is a functional that
exists for any law, estimated consistently at the usual rate:

    scale             IQR = q75 - q25
    tail weight       T99 = (q99 - q01) / IQR                     (Gaussian: 3.45)
    asymmetry         Hinkley S_p = (q_{1-p} + q_p - 2 q50) / (q_{1-p} - q_p), p = 5%, 1%
                      (0 if symmetric, < 0 if the loss tail is longer)
    tail index        Hill alpha on the top 5% of each tail.  k is 5% of the sample on
                      both sides, so both estimate the same functional
                      1 / E[log(Y / y95) | Y > y95] whatever the sample size.
    aggregation       T99 of h-day sums, and IQR(S_h) / (sqrt(h) IQR(S_1)) (1 for iid
                      Gaussian days), h = 5, 10, 24
    dependence in time  Spearman (rank) correlations pooled over windows and positions:
                      of r_t, r_t+k (no linear dependence), of |r_t|, |r_t+k| (volatility
                      clustering), of r_t, |r_t+k| (leverage, < 0 for equities).  For
                      alpha < 4 the sample ACF of r or |r| converges at the rate
                      n^(1 - 2/alpha) (n^0.13 for alpha = 2.3) to a non-Gaussian limit
                      (Davis & Mikosch 1998, Ann. Statist.); rank correlations converge at
                      sqrt(n) for any tail.
                      The pooled ACF of |r| mixes clustering inside a window with the spread
                      of volatility levels between windows (a model drawing one sigma per
                      window and iid days inside reproduces much of it).  Two rows separate
                      them: the within-window ACF of |r| (ranks of |r| inside each window;
                      -1/(n-1) = -0.04 for exchangeable days whatever their level), and the
                      dispersion (q75 - q25) / q50 of the windows' mean |r|.
    cross-section (portfolio section, day 1)
                      mean pairwise Spearman correlation, and mean pairwise tail
                      dependence lambda_L(5%) = P(U_i <= q, U_j <= q) / q, lambda_U(5%)
                      the same for the upper tail, on pseudo-observations (ranks).
                      lambda_L > lambda_U: assets fall together more than they rise.

Uncertainty (B replicates, 90% percentile intervals).  The statistic is recomputed on
each resample, ranks included.
    real        moving-block bootstrap in calendar time: blocks [c, c + --block) of trading
                days with c uniform, so every window is drawn with the same probability,
                volatility clustering up to the block length is kept, and windows of
                different regime episodes are resampled independently.
    generated   iid bootstrap over windows (generated windows are independent).
    difference  percentile interval of gen* - real*; '*' when it excludes 0.
With m rows, a model equal to the data would still show about 0.1 m flags (rows are
correlated, so they come in groups).  On the training windows the model was fit to the
very data it is compared with, which makes agreement easier; --holdout compares with days
it was not trained on.

How far to trust a flag.  Checked by simulation: "real" = one 5,387-day path, "generated"
= independent windows, both from the same factor GJR-GARCH-t (persistence 0.99, Hill
alpha about 2.3).  At nominal 10% the share of rows flagged was
    u-band comparison (conditional sample)           14%: about calibrated
    all days (unconditional sample)                   22% overall:
        rows marked + (depend on the mix of volatility levels: scale, tail weight, Hill,
        pooled |r| ACF, vol dispersion, rank corr, lambda_L, lambda_U)       35-45%
        other rows (skew, aggregation scale, within-window ACF, leverage,
        rank ACF of r, lambda_L - lambda_U)                                    0-20%
A 20-year path holds about one extreme volatility episode, which the stationary law
includes on average: a calm path shows lighter tails, weaker clustering and lower tail
dependence than the law it comes from, and a bootstrap of that path cannot produce the
episodes it lacks (longer blocks do not help: 60, 250 and 500 days gave the same rates).
So on the unconditional comparison a '+' row flagged is weak evidence; compare those
statistics within u-bands, where the volatility level is held fixed.  On iid data the
rate is 11% with --block 250 but 13% with 24: blocks must be much longer than a window,
since overlapping windows share days.
No flag is not agreement either: 20 years give limited power.  In the same simulation,
generated windows without the leverage effect (GJR gamma 0.12 -> 0) were flagged on 62% of
the leverage rows (all days) and 30% (u-band); a lighter factor tail (t5 -> t12) on 23% of
the tail rows (u-band).

Writes stylized_facts[_<selection>].log and .png next to the --gen file.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from csvio import feature_names_from_csv, load_returns
from regime import (add_selection_args, describe_selection, load_dates, read_sidecar,
                    real_window_starts, resolve_selection, selection_tag)
from run_logging import tee_output
from splits import add_split_args, add_split_source_args, require_split
from tailfm.evt import hill_estimator

LAGS = np.arange(1, 11)                 # figure
TABLE_LAGS = (1, 2, 5, 10)              # tables
HS = (5, 10, 24)
T99_GAUSS = (stats.norm.ppf(0.99) - stats.norm.ppf(0.01)) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
PGRID = np.linspace(0.002, 0.998, 300)


# ------------------------------------------------------------------ resampling
def block_indices(s, L, rng):
    """Moving-block bootstrap in calendar time over windows with sorted starts s: a block is
    the windows starting in [c, c + L) with c uniform on (s_0 - L, s_last], so every window
    lies in a block with the same probability L / (span + L) and gaps (other regime
    episodes) are never bridged; span / L + 1 blocks, so the resample has s.size windows on
    average (its size is random, like the number of selected days in a history)."""
    lo, hi = s[0] - L + 1, s[-1]
    while True:
        c = rng.integers(lo, hi + 1, -(-(hi - lo + 1) // L))
        a, b = np.searchsorted(s, c), np.searchsorted(s, c + L)
        idx = np.concatenate([np.arange(i, j) for i, j in zip(a, b)])
        if idx.size >= 10:
            return idx


class Ranker:
    """Pseudo-observations (average rank / (count + 1)) of the row-resample A[idx] of A (N, c),
    ranked over all entries (pooled) or within each column.  An entry's rank in the resample
    depends only on its tie group in A and on how often each row was drawn, so the ranks are
    exact (ties averaged) and cost O(N c) per replicate instead of a sort."""

    def __init__(self, A, pooled):
        N, c = A.shape
        if pooled:
            g = np.unique(A.ravel(), return_inverse=True)[1].reshape(N, c)
            self.col = np.zeros(g.max() + 1, dtype=int)
        else:
            d = stats.rankdata(A, method="dense", axis=0).astype(np.int64) - 1
            sizes = d.max(axis=0) + 1
            g = d + np.concatenate([[0], np.cumsum(sizes)[:-1]])
            self.col = np.repeat(np.arange(c), sizes)
        self.g, self.pooled = g, pooled

    def __call__(self, idx):
        N, c = self.g.shape
        m = np.bincount(idx, minlength=N).astype(float)
        w = np.bincount(self.g.ravel(), np.repeat(m, c), self.col.size)
        n = idx.size * (c if self.pooled else 1)            # entries per ranking
        rank = np.cumsum(w) - w - self.col * n + (w + 1) / 2
        return rank[self.g[idx]] / (n + 1)


def bootstrap(fn, N, idx_sets):
    """fn on the sample (rows 0..N-1) and on each resample."""
    point = fn(np.arange(N))
    reps = [fn(i) for i in idx_sets]
    return point, {k: np.array([r[k] for r in reps]) for k in point}


# ------------------------------------------------------------------ statistics
def quantile_stats(x):
    q01, q05, q25, q50, q75, q95, q99 = np.quantile(x, [.01, .05, .25, .5, .75, .95, .99])
    iqr = q75 - q25
    return dict(iqr=iqr, t99=(q99 - q01) / iqr if iqr > 0 else np.nan,
                hink05=(q95 + q05 - 2 * q50) / (q95 - q05) if q95 > q05 else np.nan,
                hink01=(q99 + q01 - 2 * q50) / (q99 - q01) if q99 > q01 else np.nan)


def lagcorr(Y, Z, k):
    """corr(Y[:, t], Z[:, t+k]) pooled over windows and positions t."""
    a, b = Y[:, :-k], Z[:, k:]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).mean() / np.sqrt((a * a).mean() * (b * b).mean()))


class Windows:
    """Statistics of one series' windows X (N, n) and of its row-resamples."""

    def __init__(self, X):
        self.X, self.rr, self.ra = X, Ranker(X, True), Ranker(np.abs(X), True)
        self.rw = stats.rankdata(np.abs(X), axis=1) / (X.shape[1] + 1)   # ranks within window
        self.vol = np.abs(X).mean(axis=1)                                 # window volatility

    def stats(self, idx, lags=LAGS, hs=HS):
        X = self.X[idx]
        x = X[:, 0]
        s = quantile_stats(x)
        s["hill_lo"], s["hill_hi"] = hill_estimator(x, 0.05, "lower"), hill_estimator(x, 0.05, "upper")
        Rr, Ra, Rw = self.rr(idx), self.ra(idx), self.rw[idx]
        for k in (k for k in lags if k < X.shape[1]):
            s[f"acf_r{k}"], s[f"acf_a{k}"], s[f"acfw{k}"], s[f"lev{k}"] = (
                lagcorr(Rr, Rr, k), lagcorr(Ra, Ra, k), lagcorr(Rw, Rw, k), lagcorr(Rr, Ra, k))
        q25, q50, q75 = np.quantile(self.vol[idx], [0.25, 0.5, 0.75])
        s["vol_disp"] = (q75 - q25) / q50 if q50 > 0 else np.nan
        for h in (h for h in hs if h <= X.shape[1]):
            qh = quantile_stats(X[:, :h].sum(axis=1))
            s[f"t99_{h}"], s[f"scale_{h}"] = qh["t99"], qh["iqr"] / (np.sqrt(h) * s["iqr"])
        return s


class CrossSection:
    """Mean pairwise Spearman correlation and tail dependence of day-1 vectors X1 (N, f)."""

    def __init__(self, X1, q=0.05):
        self.rank, self.q = Ranker(X1, False), q

    def stats(self, idx):
        U, q = self.rank(idx), self.q
        p = U.shape[1] * (U.shape[1] - 1)                   # ordered pairs i != j
        Z = (U - U.mean(axis=0)) / U.std(axis=0)
        a, bl, bu = Z.sum(axis=1), (U <= q).sum(axis=1), (U > 1 - q).sum(axis=1)
        lam_l, lam_u = (bl * (bl - 1)).mean() / (p * q), (bu * (bu - 1)).mean() / (p * q)
        return dict(rho=(a * a - (Z * Z).sum(axis=1)).mean() / p, lam_l=lam_l, lam_u=lam_u,
                    lam_asym=lam_l - lam_u)


# ------------------------------------------------------------------ report
LABELS = {
    "iqr": "IQR, day 1", "t99": f"tail weight T99, day 1 (Gauss {T99_GAUSS:.2f})",
    "hink05": "Hinkley skew 5% (<0: loss tail)", "hink01": "Hinkley skew 1%",
    "hill_lo": "Hill alpha, loss tail", "hill_hi": "Hill alpha, gain tail",
    "rho": "mean pairwise rank corr", "lam_l": "tail dependence lambda_L(5%)",
    "lam_u": "tail dependence lambda_U(5%)", "lam_asym": "lambda_L - lambda_U",
}
for h in HS:
    LABELS[f"t99_{h}"] = f"T99 of {h}-day sums"
    LABELS[f"scale_{h}"] = f"IQR({h}-day sum) / (sqrt({h}) IQR)"
for k in LAGS:
    LABELS[f"acf_r{k}"] = f"rank ACF of r, lag {k}"
    LABELS[f"acf_a{k}"] = f"rank ACF of |r|, lag {k}"
    LABELS[f"acfw{k}"] = f"within-window ACF of |r|, lag {k}"
    LABELS[f"lev{k}"] = f"rank corr(r_t, |r_t+{k}|)"
LABELS["vol_disp"] = "dispersion of window mean |r|"
SECTIONS = [
    ("marginal, day 1", ["iqr", "t99", "hink05", "hink01", "hill_lo", "hill_hi"]),
    ("aggregation", [f"{s}_{h}" for h in HS for s in ("t99", "scale")]),
    ("dependence in time", [f"{s}{k}" for s in ("acf_r", "acf_a", "acfw", "lev") for k in TABLE_LAGS]
     + ["vol_disp"]),
    ("cross-section, day 1", ["rho", "lam_l", "lam_u", "lam_asym"]),
]
# rows driven by the mix of volatility levels in the sample (see "How far to trust a flag")
LEVEL = ({"iqr", "t99", "hill_lo", "hill_hi", "vol_disp", "rho", "lam_l", "lam_u"}
         | {f"t99_{h}" for h in HS} | {f"acf_a{k}" for k in LAGS})


def row_label(k, mixed):
    return LABELS[k] + (" +" if mixed and k in LEVEL else "")


def ci(a):
    return np.nanquantile(a, 0.05), np.nanquantile(a, 0.95)


def diff_flag(rr, rg):
    lo, hi = ci(rg - rr)
    return lo, hi, bool(lo > 0 or hi < 0)


def report(title, R, G, idx_r, idx_g, mixed, cs=None):
    """Table for one series: windows R (real) and G (generated), each (N, n); `mixed` = the
    real windows are not restricted to a u-band; `cs` = the two CrossSection objects
    (real, generated) for the portfolio section."""
    pr, rr = bootstrap(Windows(R).stats, len(R), idx_r)
    pg, rg = bootstrap(Windows(G).stats, len(G), idx_g)
    if cs is not None:
        for p, r, c, N, idx in ((pr, rr, cs[0], len(R), idx_r), (pg, rg, cs[1], len(G), idx_g)):
            cp, cr = bootstrap(c.stats, N, idx)
            p.update(cp)
            r.update(cr)
    print(f"\n=== {title}: {len(R)} real windows, {len(G)} generated, B = {len(idx_r)} ===")
    print(f"  {'':<36s} {'real [90%]':>26s} {'gen [90%]':>26s} {'gen - real [90%]':>27s}")
    fmt = lambda v, lo, hi: f"{v:8.4f} [{lo:7.4f},{hi:7.4f}]"
    flagged = {}
    for head, keys in SECTIONS:
        keys = [k for k in keys if k in pr]
        if keys:
            print(f"  {head}")
        for k in keys:
            lo, hi, flagged[k] = diff_flag(rr[k], rg[k])
            print(f"    {row_label(k, mixed):<36s} {fmt(pr[k], *ci(rr[k]))} {fmt(pg[k], *ci(rg[k]))} "
                  f"{fmt(pg[k] - pr[k], lo, hi)} {'*' if flagged[k] else ''}")
    print(f"  {sum(flagged.values())} of {len(flagged)} rows flagged '*' (90% interval of "
          f"gen - real excludes 0); about {0.1 * len(flagged):.0f} expected by chance if the "
          f"model were exact")
    if mixed:
        rest = [flagged[k] for k in flagged if k not in LEVEL]
        print(f"  {sum(rest)} of {len(rest)} rows without '+'.  '+' rows depend on the mix of "
              f"volatility levels: on an all-days comparison they are flagged too often "
              f"(simulation: 35-45% at nominal 10%); compare them within u-bands")
    return pr, rr, pg, rg


def asset_summary(R, G, names, idx_r, idx_g, mixed):
    """Per asset: day-1 and lag-1 statistics; medians across assets and the share of assets
    whose 90% interval of gen - real excludes 0 (about 10% by chance)."""
    keys = ["iqr", "t99", "hink05", "hill_lo", "hill_hi", "acf_r1", "acf_a1", "acfw1", "lev1",
            "vol_disp"]
    res = {k: [] for k in keys}                         # (real, gen, flagged) per asset
    for a in range(0, R.shape[2], 32):
        Gb = np.asarray(G[:, :, a:a + 32], dtype=float)
        for j in range(a, min(a + 32, R.shape[2])):
            if R[:, 0, j].std() == 0 or Gb[:, 0, j - a].std() == 0:
                continue
            fr = lambda idx, W=Windows(R[:, :, j]): W.stats(idx, lags=(1,), hs=())
            fg = lambda idx, W=Windows(Gb[:, :, j - a]): W.stats(idx, lags=(1,), hs=())
            pr, rr = bootstrap(fr, len(R), idx_r)
            pg, rg = bootstrap(fg, len(Gb), idx_g)
            for k in keys:
                res[k].append((pr[k], pg[k], diff_flag(rr[k], rg[k])[2]))
    n = len(res[keys[0]])
    print(f"\n=== all assets ({n} of {len(names)} non-constant), B = {len(idx_r)} per asset ===")
    print(f"  {'':<36s} {'median real':>12s} {'median gen':>12s} {'median gen-real':>16s} "
          f"{'assets flagged':>15s}")
    for k in keys:
        r, g, fl = np.array(res[k]).T
        print(f"  {row_label(k, mixed):<36s} {np.nanmedian(r):12.4f} {np.nanmedian(g):12.4f} "
              f"{np.nanmedian(g - r):16.4f} {fl.mean():14.0%}")
    print("  assets flagged = share whose 90% interval of gen - real excludes 0 (about 10% by chance"
          + ("; more for '+' rows, see above)" if mixed else ")"))


# ------------------------------------------------------------------ figure
def errorbars(ax, xs, keys, p, r, color, label, dx, marker="o"):
    """Point estimates and their 90% intervals (which need not contain them)."""
    x = np.asarray(xs) + dx
    lo, hi = np.array([ci(r[k]) for k in keys]).T
    ax.plot(x, [p[k] for k in keys], marker + "-", color=color, ms=4, label=label)
    ax.vlines(x, lo, hi, color=color, lw=1)


def figure(R, G, pr, rr, pg, rg, idx_r, label, path):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    for a, (xr, xg, t) in zip(ax[0, :2], ((R[:, 0], G[:, 0], "day 1"),
                                         (R.sum(1), G.sum(1), f"{R.shape[1]}-day sum"))):
        qr, qg = np.quantile(xr, PGRID), np.quantile(xg, PGRID)
        lo, hi = np.quantile([np.quantile(xr[i], PGRID) for i in idx_r], [0.05, 0.95], axis=0)
        a.fill_between(qr, lo, hi, color="0.85", label="pointwise 90% band of real quantile")
        a.plot(qr, qg, ".", ms=3, color="C0", label="generated")
        lim = [min(qr.min(), qg.min()), max(qr.max(), qg.max())]
        a.plot(lim, lim, color="0.4", lw=1)
        a.set(title=f"QQ, {t}: generated vs real", xlabel="real quantile", ylabel="generated quantile")
        a.legend(loc="upper left")
    a = ax[0, 2]
    for x, c, lab in ((-R[:, 0], "k", "real"), (-G[:, 0], "C0", "generated")):
        s = np.sort(x[x > 0])[::-1]
        a.loglog(s, np.arange(1, s.size + 1) / x.size, color=c, lw=1.2, label=lab)
    a.set(title="loss tail, day 1: P(L > x)", xlabel="loss x")
    a.legend()
    L = [k for k in LAGS if f"acf_a{k}" in pr]
    a = ax[1, 0]
    for key, mk, t in (("acf_a", "o", "pooled"), ("acfw", "s", "within window")):
        errorbars(a, L, [f"{key}{k}" for k in L], pr, rr, "k", f"real, {t}", -0.08, mk)
        errorbars(a, L, [f"{key}{k}" for k in L], pg, rg, "C0", f"generated, {t}", 0.08, mk)
    a.axhline(-1 / (R.shape[1] - 1), color="0.6", lw=0.8, ls="--", label="within window, iid days")
    a.axhline(0, color="0.6", lw=0.8)
    a.set(title="rank ACF of |r| (volatility clustering)", xlabel="lag k (days)")
    a.legend(fontsize=8)
    a = ax[1, 1]
    errorbars(a, L, [f"lev{k}" for k in L], pr, rr, "k", "real", -0.08)
    errorbars(a, L, [f"lev{k}" for k in L], pg, rg, "C0", "generated", 0.08)
    for p, c in ((pr, "k"), (pg, "C0")):
        a.plot(L, [p[f"acf_r{k}"] for k in L], ":", color=c, lw=1)
    a.axhline(0, color="0.6", lw=0.8)
    a.set(title="rank corr(r_t, |r_t+k|) (leverage; dotted: rank ACF of r)", xlabel="lag k (days)")
    a.legend()
    a = ax[1, 2]
    hs = [1] + [h for h in HS if f"t99_{h}" in pr]
    keys = ["t99"] + [f"t99_{h}" for h in hs[1:]]
    errorbars(a, hs, keys, pr, rr, "k", "real", -0.3)
    errorbars(a, hs, keys, pg, rg, "C0", "generated", 0.3)
    a.axhline(T99_GAUSS, color="0.6", lw=0.8, ls="--", label="Gaussian")
    a.set(title="tail weight T99 of h-day sums", xlabel="h (days)")
    a.legend()
    fig.suptitle(f"Stylized facts: {label} (bars and band: 90% bootstrap intervals)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ main
def run(args):
    rng = np.random.default_rng(args.seed)
    gen_path = args.gen.rpartition("=")[2]
    G = np.load(gen_path, mmap_mode="r")
    r = load_returns(args.data, args.prices)
    T, f = r.shape
    names = feature_names_from_csv(args.data, f)
    split = require_split([args.gen], args, T, data=os.path.abspath(args.data),
                          allow_missing=args.allow_missing_split)
    n = G.shape[1]
    periods, excluded, band, holdout = resolve_selection(args, [read_sidecar(gen_path) or {}])
    u = np.load(os.path.join(os.path.dirname(os.path.abspath(gen_path)), "cond_u.npy")) if band else None
    dates = load_dates(args.data, T) if periods or excluded else None
    starts = real_window_starts(split, n, 1, dates, periods, excluded, band, u, holdout)
    if starts.size < 50:
        raise SystemExit(f"only {starts.size} real windows match the selection")
    R = np.stack([r[s:s + n] for s in starts])
    print(f"generated {G.shape} vs {R.shape[0]} real {n}-day windows, "
          f"{describe_selection(periods, excluded, band, holdout)}")
    periods_hit = np.unique(starts // args.block).size
    print(f"real: block bootstrap, blocks of {args.block} rows; the real windows fall in "
          f"{periods_hit} disjoint {args.block}-row periods"
          + ("  WARNING: under 10, so the real intervals rest on too few independent periods"
             if periods_hit < 10 else ""))

    idx_r = [block_indices(starts, args.block, rng) for _ in range(args.n_boot)]
    idx_g = [rng.integers(0, len(G), len(G)) for _ in range(args.n_boot)]
    w = (np.array([float(v) for v in args.weights.split(",")]) if args.weights
         else np.full(f, 1.0 / f))
    Pr, Pg = R @ w, np.asarray(G @ w, dtype=float)
    G1 = np.asarray(G[:, 0, :], dtype=float)
    keep = (R[:, 0].std(axis=0) > 0) & (G1.std(axis=0) > 0)
    cs = (CrossSection(R[:, 0, keep]), CrossSection(G1[:, keep]))
    title = "equal-weight portfolio" if not args.weights else "portfolio (--weights)"
    mixed = band is None
    res = report(title, Pr, Pg, idx_r, idx_g, mixed, cs)
    tag = selection_tag(band, periods, excluded, holdout)
    stem = os.path.join(args.outdir or os.path.dirname(os.path.abspath(gen_path)),
                        "stylized_facts" + (f"_{tag}" if tag else ""))
    paths = [figure(Pr, Pg, *res, idx_r, title, stem + ".png")]

    for a in args.assets or []:
        j = names.index(a)
        Gj = np.asarray(G[:, :, j], dtype=float)
        res = report(f"asset {a}", R[:, :, j], Gj, idx_r, idx_g, mixed)
        paths.append(figure(R[:, :, j], Gj, *res, idx_r, f"asset {a}", f"{stem}_{a}.png"))
    if not args.no_assets_summary:
        nb = args.n_boot_assets
        asset_summary(R, G, names, idx_r[:nb], idx_g[:nb], mixed)
    print("\nSaved: " + ", ".join(paths))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--gen", required=True, help="generated windows .npy (M, n, f)")
    ap.add_argument("--prices", action="store_true", help="input is prices, not returns")
    ap.add_argument("--weights", default=None, help="comma-separated weights (default: equal)")
    ap.add_argument("--assets", nargs="+", default=None, help="also these columns (VALORs)")
    ap.add_argument("--n-boot", type=int, default=200, help="bootstrap replicates")
    ap.add_argument("--n-boot-assets", type=int, default=100,
                    help="replicates per asset in the all-assets summary (<= --n-boot)")
    ap.add_argument("--block", type=int, default=250,
                    help="block length in rows (trading days) of the real-data bootstrap")
    ap.add_argument("--no-assets-summary", action="store_true", help="skip the all-assets summary")
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
                                   "stylized_facts" + (f"_{tag}" if tag else "") + ".log")
    with tee_output(log, header="stylized_facts.py"):
        run(args)
    print(f"Terminal report saved to {log}")


if __name__ == "__main__":
    main()
