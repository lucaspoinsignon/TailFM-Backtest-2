"""Does the EWMA volatility capture the regimes, and is the dependence of the
standardised returns stable over time once it is removed?

    python vol_corr_check.py --data data/returns_clean.csv --outdir fig/vol_corr

Writes three figures and prints one table:

  ewma_vol.png      EWMA volatility sigma_t of 30 random valors over the whole period:
                    (top) in % per day, log scale; (bottom) divided by each valor's own
                    median, so all valors are on one scale, with the median over ALL
                    valors in black.  Spikes that line up across valors are market-wide
                    regimes; spikes of single valors are idiosyncratic.
  pair_corr.png     rolling correlation of 30 random pairs: raw log returns (grey) vs
                    standardised returns eps = r / sigma (blue).  The shaded band is the
                    approximate 95% range a rolling correlation would wander in if the
                    true correlation were constant at its full-sample value -- rolling
                    estimates on a short window are noisy by construction.
  avg_corr.png      rolling average correlation over ALL pairs of valors, raw vs
                    standardised, above the market volatility level.

  Table: average pairwise correlation on calm / middle / stressed days (terciles of the
  market volatility level), raw vs standardised.  If standardising removes the regime
  effect on dependence, the standardised row is flat across the three columns.

EWMA: sigma^2_t = lam sigma^2_{t-1} + (1-lam) r^2_{t-1}, per valor, returns before t
only, lam = 0.94.  This is a description of the whole sample (no train/test split).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVENTS = {"Lehman": "2008-09-15", "US downgrade": "2011-08-05",
          "SNB floor": "2015-01-15", "COVID": "2020-03-16"}


def ewma_sigma(r, lam, init_var, floor_var):
    s2 = np.empty_like(r)
    s2[0] = init_var
    for t in range(1, r.shape[0]):
        s2[t] = np.maximum(lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2, floor_var)
    return np.sqrt(s2)


def avg_offdiag_corr(X):
    """Mean pairwise correlation of the columns of X, ignoring constant columns."""
    sd = X.std(axis=0)
    X = X[:, sd > 0]
    if X.shape[1] < 2:
        return np.nan
    C = np.corrcoef(X.T)
    iu = np.triu_indices_from(C, 1)
    return float(np.nanmean(C[iu]))


def mark_events(ax, dates):
    if not isinstance(dates, pd.DatetimeIndex):
        return
    for name, d in EVENTS.items():
        d = pd.Timestamp(d)
        if dates[0] <= d <= dates[-1]:
            ax.axvline(d, color="tab:red", lw=0.6, alpha=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/returns_clean.csv")
    ap.add_argument("--outdir", default="fig/vol_corr")
    ap.add_argument("--lam", type=float, default=0.94)
    ap.add_argument("--n-valors", type=int, default=30)
    ap.add_argument("--n-pairs", type=int, default=30)
    ap.add_argument("--window", type=int, default=63, help="rolling window, days")
    ap.add_argument("--stride", type=int, default=5, help="step of the all-pairs rolling average")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    df = pd.read_csv(a.data, index_col=0)
    try:
        df.index = pd.to_datetime(df.index)
    except (ValueError, TypeError):
        df.index = pd.RangeIndex(len(df))
    dates, names = df.index, [str(c) for c in df.columns]
    r = df.to_numpy(dtype=float)
    T, f = r.shape
    if not np.isfinite(r).all():
        raise SystemExit("non-finite returns: use the cleaned file (02_returns.py)")

    full_var = r.var(axis=0)
    full_var = np.where(full_var > 0, full_var, 1e-12)
    sig = ewma_sigma(r, a.lam, full_var, (0.05 ** 2) * full_var)
    eps = r / sig
    rel = sig / np.median(sig, axis=0)                # each valor's vol / its own median
    market = np.median(rel, axis=1)                   # market volatility level, (T,)

    # ---------------------------------------------------------------- figure 1
    pick = rng.choice(f, size=min(a.n_valors, f), replace=False)
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for j in pick:
        ax[0].plot(dates, 100 * sig[:, j], lw=0.6, alpha=0.7)
        ax[1].plot(dates, rel[:, j], lw=0.6, alpha=0.5)
    ax[1].plot(dates, market, color="k", lw=1.6, label=f"median over all {f} valors")
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    ax[0].set_ylabel("EWMA vol, % per day")
    ax[1].set_ylabel("vol / own median")
    ax[0].set_title(f"EWMA volatility (lambda={a.lam}) of {len(pick)} random valors")
    ax[1].legend(loc="upper left")
    for x in ax:
        mark_events(x, dates)
    fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "ewma_vol.png"), dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 2
    iu = np.array(np.triu_indices(f, 1)).T
    pairs = iu[rng.choice(len(iu), size=min(a.n_pairs, len(iu)), replace=False)]
    L = a.window
    ncol = 5
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.4 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for k, (i, j) in enumerate(pairs):
        ax = axes[k // ncol][k % ncol]
        cr = pd.Series(r[:, i], index=dates).rolling(L).corr(pd.Series(r[:, j], index=dates))
        ce = pd.Series(eps[:, i], index=dates).rolling(L).corr(pd.Series(eps[:, j], index=dates))
        rho = np.corrcoef(eps[:, i], eps[:, j])[0, 1]
        half = 1.96 * (1 - rho ** 2) / np.sqrt(L)
        ax.axhspan(rho - half, rho + half, color="tab:blue", alpha=0.12, lw=0)
        ax.plot(dates, cr, color="0.55", lw=0.6, label="raw")
        ax.plot(dates, ce, color="tab:blue", lw=0.7, label="standardised")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"{names[i]} / {names[j]}", fontsize=8)
        mark_events(ax, dates)
    for k in range(len(pairs), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    axes[0][0].legend(fontsize=7, loc="lower left")
    axes[0][0].set_ylim(-1, 1)
    fig.suptitle(f"rolling {L}-day correlation, raw vs standardised; band = 95% range "
                 f"under a constant correlation", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "pair_corr.png"), dpi=110)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 3
    ends = np.arange(L, T + 1, a.stride)
    avg_raw = np.array([avg_offdiag_corr(r[e - L:e]) for e in ends])
    avg_eps = np.array([avg_offdiag_corr(eps[e - L:e]) for e in ends])
    d_end = dates[ends - 1]
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
    ax[0].plot(d_end, avg_raw, color="0.45", lw=1.0, label="raw log returns")
    ax[0].plot(d_end, avg_eps, color="tab:blue", lw=1.2, label="standardised (r / EWMA)")
    ax[0].set_ylabel(f"avg pairwise corr ({L}-day)")
    ax[0].set_title(f"rolling average correlation over all {f * (f - 1) // 2} pairs")
    ax[0].legend(loc="upper left")
    ax[1].plot(dates, market, color="k", lw=0.8)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("market vol level")
    for x in ax:
        mark_events(x, dates)
    fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "avg_corr.png"), dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- table
    cuts = np.quantile(market, [1 / 3, 2 / 3])
    reg = np.digitize(market, cuts)
    lab = ("calm", "middle", "stressed")
    tab = {k: [avg_offdiag_corr(x[reg == g]) for g in range(3)]
           for k, x in (("raw returns", r), ("standardised", eps))}
    print(f"\naverage pairwise correlation by market-volatility tercile "
          f"({f} valors, {[int((reg == g).sum()) for g in range(3)]} days)")
    print(f"{'':>14s}" + "".join(f"{s:>10s}" for s in lab) + f"{'stress - calm':>15s}")
    for k, v in tab.items():
        print(f"{k:>14s}" + "".join(f"{x:10.3f}" for x in v) + f"{v[2] - v[0]:+15.3f}")
    print(f"\nrolling avg correlation over time: sd raw {np.nanstd(avg_raw):.3f}, "
          f"standardised {np.nanstd(avg_eps):.3f}; corr with market vol level: raw "
          f"{np.corrcoef(avg_raw, np.log(market[ends - 1]))[0, 1]:+.2f}, standardised "
          f"{np.corrcoef(avg_eps, np.log(market[ends - 1]))[0, 1]:+.2f}")
    print(f"figures: {a.outdir}/ewma_vol.png, pair_corr.png, avg_corr.png")


if __name__ == "__main__":
    main()
