"""Stylized facts of generated vs real returns, on the model's whole 24-day windows.

    python stylized_facts.py --data data/returns_clean.csv --gen runs/uncond/generated_windows.npy
    python stylized_facts.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy
    python stylized_facts.py ... --assets 4155001          # also single columns, e.g. MSCI World

Reference = the real windows chosen as in evaluate.py: by default the training windows,
for a conditional sample only those whose first day has u in the sample's band (its
.json sidecar); --no-band / --cond-level / --cond-band / --holdout / --test-period /
--exclude-period override.  Static facts use day 1 of each window (each real day once),
dynamic facts use the whole windows.

Series: the equal-weight portfolio (--weights to change it), each --assets column, and a
summary over all assets (median and quartiles across the f columns).

    1 heavy tails                 excess kurtosis, Hill tail index of each tail (top 5%)
    2 gain / loss asymmetry       skewness, A95 = (VaR95 of losses + mu) / (VaR95 of gains - mu)
    3 aggregational Gaussianity   excess kurtosis of h-day sums, variance ratio, h = 1, 5, 10, 24
    4 no linear autocorrelation   ACF of r, lags 1..10
    5 volatility clustering       ACF of |r|, lags 1..10
    6 leverage effect             corr(r_t, |r_t+k|), k = 1..10 (negative for equities)
    7 asymmetric correlation      mean pairwise correlation of the assets on all / the 10%
                                  worst / the 10% best equal-weight days (portfolio only)

ACFs are pooled over windows with the global mean and variance, so real and generated are
estimated the same way.  With a few hundred real windows (one regime band) the real
statistics are noisy, kurtosis and the tail index above all.

Writes stylized_facts[_<selection>].log and .png next to the first --gen.
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

LAGS = np.arange(1, 11)
HS = (1, 5, 10, 24)


# ------------------------------------------------------------------ statistics
def acf(y, z, k):
    """corr(y_t, z_t+k) pooled over windows (rows), global means and sds."""
    a, b = y[:, :-k] - y.mean(), z[:, k:] - z.mean()
    return float((a * b).mean() / (y.std() * z.std()))


def mean_corr(x):
    """Mean off-diagonal correlation of the columns of x (m, f); constant columns dropped."""
    sd = x.std(axis=0)
    z = (x[:, sd > 0] - x[:, sd > 0].mean(axis=0)) / sd[sd > 0]
    m, f = z.shape
    return float(((z.sum(axis=1) ** 2).sum() / m - f) / (f * (f - 1)))


def facts(P):
    """Stylized facts of one series given as windows P (M, n)."""
    x, n = P[:, 0], P.shape[1]
    mu = x.mean()
    out = dict(
        kurt=stats.kurtosis(x), skew=stats.skew(x),
        hill_lo=hill_estimator(x, 0.05, "lower"), hill_hi=hill_estimator(x, 0.05, "upper"),
        A95=(-np.quantile(x, 0.05) + mu) / (np.quantile(x, 0.95) - mu),
        acf_r=[acf(P, P, k) for k in LAGS if k < n],
        acf_abs=[acf(np.abs(P), np.abs(P), k) for k in LAGS if k < n],
        lev=[acf(P, np.abs(P), k) for k in LAGS if k < n],
    )
    hs = [h for h in HS if h <= n]
    sums = {h: P[:, :h].sum(axis=1) for h in hs}
    out["kurt_h"] = {h: stats.kurtosis(s) for h, s in sums.items()}
    out["vr_h"] = {h: s.var() / (h * x.var()) for h, s in sums.items()}
    return out


# ------------------------------------------------------------------ report
def row(name, r, g, fmt="{:8.3f}"):
    print(f"  {name:<34s} {fmt.format(r)}  {fmt.format(g)}")


def lag_rows(name, r, g):
    print(f"  {name}")
    print("    lag     " + " ".join(f"{k:>6d}" for k in LAGS[:len(r)]))
    print("    real    " + " ".join(f"{v:6.3f}" for v in r))
    print("    gen     " + " ".join(f"{v:6.3f}" for v in g))


def report(title, fr, fg, extra=None):
    print(f"\n=== {title} ===\n  {'':<34s} {'real':>8s}  {'gen':>8s}")
    row("excess kurtosis (day 1)", fr["kurt"], fg["kurt"])
    row("skewness (day 1)", fr["skew"], fg["skew"])
    row("Hill tail index, loss tail", fr["hill_lo"], fg["hill_lo"])
    row("Hill tail index, gain tail", fr["hill_hi"], fg["hill_hi"])
    row("A95 loss / gain (1 = symmetric)", fr["A95"], fg["A95"])
    for h in fr["kurt_h"]:
        row(f"excess kurtosis of {h}-day sums", fr["kurt_h"][h], fg["kurt_h"][h])
    for h in fr["vr_h"]:
        if h > 1:
            row(f"variance ratio, h = {h}", fr["vr_h"][h], fg["vr_h"][h])
    lag_rows("ACF of r (should be ~0)", fr["acf_r"], fg["acf_r"])
    lag_rows("ACF of |r| (volatility clustering, > 0)", fr["acf_abs"], fg["acf_abs"])
    lag_rows("corr(r_t, |r_t+k|) (leverage, < 0)", fr["lev"], fg["lev"])
    for name, (r, g) in (extra or {}).items():
        row(name, r, g)


def marginal_summary(R, G, names):
    """Median [q25, q75] across assets of the day-1 and lag-1 facts, real vs gen."""
    keys = ("kurt", "skew", "hill_lo", "hill_hi", "A95")
    vals = {k: ([], []) for k in keys + ("acf_abs1", "lev1")}
    for a in range(0, R.shape[2], 32):
        Gb = np.asarray(G[:, :, a:a + 32], dtype=float)          # one read per block
        for j in range(a, min(a + 32, R.shape[2])):
            _one_asset(R[:, :, j], Gb[:, :, j - a], vals)
    _print_summary(vals, len(names))


def _one_asset(Xr, Xg, vals):
    for side, X in enumerate((Xr, Xg)):
        x, mu = X[:, 0], X[:, 0].mean()
        v = dict(kurt=stats.kurtosis(x), skew=stats.skew(x),
                 hill_lo=hill_estimator(x, 0.05, "lower"),
                 hill_hi=hill_estimator(x, 0.05, "upper"),
                 A95=(-np.quantile(x, 0.05) + mu) / max(np.quantile(x, 0.95) - mu, 1e-12),
                 acf_abs1=acf(np.abs(X), np.abs(X), 1) if X.std() > 0 else np.nan,
                 lev1=acf(X, np.abs(X), 1) if X.std() > 0 else np.nan)
        for k in vals:
            vals[k][side].append(v[k])


def _print_summary(vals, f):
    print(f"\n=== all {f} assets: median [q25, q75] across assets ===")
    print(f"  {'':<26s} {'real':>24s}   {'gen':>24s}   median gen / real")
    labels = dict(kurt="excess kurtosis (day 1)", skew="skewness (day 1)",
                  hill_lo="Hill index, loss tail", hill_hi="Hill index, gain tail",
                  A95="A95 loss / gain", acf_abs1="ACF |r|, lag 1", lev1="leverage, lag 1")
    for k, (r, g) in vals.items():
        r, g = np.array(r), np.array(g)
        ok = np.isfinite(r) & np.isfinite(g)
        q = lambda a: f"{np.median(a):7.3f} [{np.quantile(a, .25):6.2f},{np.quantile(a, .75):6.2f}]"
        ratio = np.nanmedian(g[ok] / np.where(r[ok] == 0, np.nan, r[ok])) if k != "skew" else np.nan
        print(f"  {labels[k]:<26s} {q(r[ok]):>24s}   {q(g[ok]):>24s}   "
              + (f"{ratio:6.2f}" if np.isfinite(ratio) else "   --"))


def figure(Pr, Pg, fr, fg, label, path):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    p = np.linspace(0.001, 0.999, 400)
    for a, (xr, xg, t) in zip(ax[0, :2], ((Pr[:, 0], Pg[:, 0], "day 1"),
                                         (Pr.sum(1), Pg.sum(1), f"{Pr.shape[1]}-day sum"))):
        qr, qg = np.quantile(xr, p), np.quantile(xg, p)
        a.plot(qr, qg, ".", ms=3, color="C0")
        lim = [min(qr.min(), qg.min()), max(qr.max(), qg.max())]
        a.plot(lim, lim, color="0.4", lw=1)
        a.set(title=f"QQ, {t}: generated vs real", xlabel="real quantile", ylabel="generated quantile")
    a = ax[0, 2]
    for x, c, lab in ((-Pr[:, 0], "k", "real"), (-Pg[:, 0], "C0", "generated")):
        s = np.sort(x[x > 0])[::-1]
        a.loglog(s, np.arange(1, s.size + 1) / x.size, color=c, lw=1.2, label=lab)
    a.set(title="loss tail, day 1: P(L > x)", xlabel="loss x")
    a.legend()
    for a, key, t in ((ax[1, 0], "acf_abs", "ACF of |r| (volatility clustering)"),
                      (ax[1, 1], "lev", "corr(r_t, |r_t+k|) (leverage)")):
        L = LAGS[:len(fr[key])]
        a.plot(L, fr[key], "o-", color="k", label="real")
        a.plot(L, fg[key], "o-", color="C0", label="generated")
        if key == "acf_abs":
            a.plot(L, fr["acf_r"], ":", color="k", lw=1, label="real ACF of r")
            a.plot(L, fg["acf_r"], ":", color="C0", lw=1, label="gen ACF of r")
        a.axhline(0, color="0.6", lw=0.8)
        a.set(title=t, xlabel="lag k (days)")
        a.legend()
    a = ax[1, 2]
    hs = list(fr["kurt_h"])
    a.plot(hs, [fr["kurt_h"][h] for h in hs], "o-", color="k", label="real")
    a.plot(hs, [fg["kurt_h"][h] for h in hs], "o-", color="C0", label="generated")
    a.axhline(0, color="0.6", lw=0.8)
    a.set(title="excess kurtosis of h-day sums", xlabel="h (days)")
    a.legend()
    fig.suptitle(f"Stylized facts: {label}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ main
def run(args):
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
    if starts.size == 0:
        raise SystemExit("no real window matches the selection")
    R = np.stack([r[s:s + n] for s in starts])
    print(f"generated {G.shape} vs {R.shape[0]} real {n}-day windows, "
          f"{describe_selection(periods, excluded, band, holdout)}")

    w = (np.array([float(v) for v in args.weights.split(",")]) if args.weights
         else np.full(f, 1.0 / f))
    Pr, Pg = R @ w, np.asarray(G @ w, dtype=float)
    fr, fg = facts(Pr), facts(Pg)
    lo_r, hi_r = Pr[:, 0] <= np.quantile(Pr[:, 0], 0.1), Pr[:, 0] >= np.quantile(Pr[:, 0], 0.9)
    lo_g, hi_g = Pg[:, 0] <= np.quantile(Pg[:, 0], 0.1), Pg[:, 0] >= np.quantile(Pg[:, 0], 0.9)
    G1 = np.asarray(G[:, 0, :], dtype=float)
    extra = {"mean pairwise corr, all days": (mean_corr(R[:, 0]), mean_corr(G1)),
             "mean pairwise corr, worst 10%": (mean_corr(R[lo_r, 0]), mean_corr(G1[lo_g])),
             "mean pairwise corr, best 10%": (mean_corr(R[hi_r, 0]), mean_corr(G1[hi_g]))}
    report("equal-weight portfolio" if not args.weights else "portfolio (--weights)", fr, fg, extra)
    tag = selection_tag(band, periods, excluded, holdout)
    stem = os.path.join(args.outdir or os.path.dirname(os.path.abspath(gen_path)),
                        "stylized_facts" + (f"_{tag}" if tag else ""))
    paths = [figure(Pr, Pg, fr, fg, "equal-weight portfolio", stem + ".png")]

    for a in args.assets or []:
        j = names.index(a)
        fa_r, fa_g = facts(R[:, :, j]), facts(np.asarray(G[:, :, j], dtype=float))
        report(f"asset {a}", fa_r, fa_g)
        paths.append(figure(R[:, :, j], np.asarray(G[:, :, j], dtype=float), fa_r, fa_g,
                            f"asset {a}", f"{stem}_{a}.png"))
    marginal_summary(R, G, names)
    print("\nSaved: " + ", ".join(paths))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--gen", required=True, help="generated windows .npy (M, n, f)")
    ap.add_argument("--prices", action="store_true", help="input is prices, not returns")
    ap.add_argument("--weights", default=None, help="comma-separated weights (default: equal)")
    ap.add_argument("--assets", nargs="+", default=None, help="also these columns (VALORs)")
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
