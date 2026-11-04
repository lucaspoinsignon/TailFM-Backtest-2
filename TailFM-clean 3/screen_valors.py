"""Screen every VALOR of a returns CSV for series that break the EVT marginals.

    python screen_valors.py --data data/returnscomp.csv --out fig/screen
    python screen_valors.py --data data/returnscomp.csv --out fig/screen --reference data/returns_clean.csv

The lower GPD of tailfm.evt is fitted to the days below the q-quantile, which assumes that
quantile is a LOSS in the tail of a continuous law.  Four kinds of series break that:

    accrual     q-quantile >= 0: at least 1 - q of the days are gains (money market, cash,
                T-bill indices).  The "loss tail" is the low end of the accrual rate plus a few
                real down days, exceedances spanning orders of magnitude, so xi_hat >> 1.
    xi          Zhang-Stephens xi_hat >= 1 (no finite mean: ES undefined, the inverse PIT
                returns absurd losses) or <= -0.5 (point mass at the threshold).
    persistent  lag-1 autocorrelation of returns > 0.5: the return is a slowly moving level
                (an interest rate), not an innovation.
    jumps       max |r - median| / IQR > 50: flat series with rare jumps (stale NAV, resets).

BAD = accrual or xi (the fit is invalid); WARN = persistent, jumps, zero fraction > 2%, or
daily sd < 1bp.  Writes screen.csv (one row per VALOR), screen.png (price, returns and the
lower-tail exceedances with their GPD fit, for every flagged VALOR), weights_clean.txt
(equal weight on the non-BAD VALORs, 0 on the BAD ones) and, with --reference,
weights_reference.txt (equal weight on the VALORs of that CSV), for --weights of
evaluate.py / run_backtest.py.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fit_gpd_zs(y):
    """Zhang & Stephens (2009) (xi, beta), as evt_shrink.fit_gpd_zs but with the
    profile-likelihood weights computed as a softmax, which cannot overflow."""
    y = np.sort(np.asarray(y, dtype=float)[np.isfinite(y) & (np.asarray(y) > 0)])
    n = y.size
    if n < 5:
        return np.nan, np.nan
    m = 30 + int(np.sqrt(n))
    b = (1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))) / (3.0 * y[int(n / 4 + 0.5) - 1]) + 1.0 / y[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.log1p(-b[:, None] * y).mean(axis=1)
        L = np.nan_to_num(n * (np.log(-b / k) - k - 1.0), nan=-np.inf)
    w = np.exp(L - L.max())
    bh = float((b * w).sum() / w.sum())
    xi = float(np.log1p(-bh * y).mean())
    return xi, -xi / bh


def screen(r: pd.DataFrame, q: float) -> pd.DataFrame:
    rows = []
    for c in r.columns:
        x = r[c].to_numpy(dtype=float)
        u_lo, u_hi = np.quantile(x, q), np.quantile(x, 1 - q)
        iqr = np.subtract(*np.quantile(x, [0.75, 0.25]))
        xi_lo = fit_gpd_zs(u_lo - x[x < u_lo])[0]
        xi_hi = fit_gpd_zs(x[x > u_hi] - u_hi)[0]
        acf1 = np.corrcoef(x[1:], x[:-1])[0, 1] if x.std() > 0 else np.nan
        jump = np.abs(x - np.median(x)).max() / iqr if iqr > 0 else np.inf
        why = []
        if u_lo >= 0:
            why.append(f"accrual: {q:.0%} quantile {u_lo:+.1e} >= 0")
        if u_hi <= 0:
            why.append(f"no gains: {1 - q:.0%} quantile {u_hi:+.1e} <= 0")
        for side, xi in (("lo", xi_lo), ("hi", xi_hi)):
            if np.isfinite(xi) and not -0.5 < xi < 1.0:
                why.append(f"xi_{side} {xi:+.2f}")
        bad = bool(why)
        if acf1 > 0.5:
            why.append(f"persistent: acf1 {acf1:.2f}")
        if jump > 50:
            why.append(f"jumps: max|r-med|/IQR {jump:.0f}")
        if (x == 0).mean() > 0.02:
            why.append(f"zeros {(x == 0).mean():.1%}")
        if x.std() < 1e-4:
            why.append(f"sd {x.std() * 1e4:.2f}bp")
        rows.append(dict(valor=c, status="BAD" if bad else ("WARN" if why else "ok"),
                         sd_bp=x.std() * 1e4, neg_frac=(x < 0).mean(), zero_frac=(x == 0).mean(),
                         q_lo=u_lo, xi_lo=xi_lo, xi_hi=xi_hi, acf1=acf1, jump_iqr=jump,
                         why="; ".join(why)))
    order = {"BAD": 0, "WARN": 1, "ok": 2}
    return (pd.DataFrame(rows).assign(_o=lambda d: d.status.map(order))
            .sort_values(["_o", "sd_bp"]).drop(columns="_o").reset_index(drop=True))


def figure(r: pd.DataFrame, tab: pd.DataFrame, q: float, path: str, top: int) -> None:
    flagged = tab[tab.status != "ok"].head(top)
    if flagged.empty:
        return
    fig, axes = plt.subplots(len(flagged), 3, figsize=(15, 2.4 * len(flagged)), squeeze=False)
    for ax, (_, row) in zip(axes, flagged.iterrows()):
        x = r[row.valor]
        ax[0].plot(x.index, x.cumsum(), lw=0.8)
        ax[0].set_title(f"{row.valor} [{row.status}] cumulative log return", fontsize=8)
        ax[1].plot(x.index, x, lw=0.5)
        ax[1].set_title(row.why[:90], fontsize=7)
        v = x.to_numpy()
        u = np.quantile(v, q)
        y = np.sort(u - v[v < u])
        y = y[y > 0]
        if y.size >= 5:
            s = 1.0 - np.arange(y.size) / y.size
            ax[2].loglog(y, s, ".", ms=2)
            xi, beta = fit_gpd_zs(y)
            if np.isfinite(xi) and beta > 0:
                g = np.geomspace(y[0], y[-1], 200)
                sf = (np.exp(-g / beta) if abs(xi) < 1e-8 else
                      np.clip(1 + xi * g / beta, 0, None) ** (-1 / xi))
                ax[2].loglog(g, sf, "k-", lw=1, label=f"GPD xi={xi:+.2f}")
                ax[2].legend(fontsize=7)
        ax[2].set_title(f"lower-tail exceedances below the {q:.0%} quantile {u:+.1e}", fontsize=8)
        for a in ax:
            a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_weights(cols, keep, path):
    w = np.where(keep, 1.0 / keep.sum(), 0.0)
    with open(path, "w") as fh:
        fh.write(",".join(f"{v:.12g}" for v in w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="returns CSV, Date column first")
    ap.add_argument("--out", default="fig/screen")
    ap.add_argument("--q-tail", type=float, default=0.05, help="as train_tailfm.py --q-tail")
    ap.add_argument("--top", type=int, default=30, help="flagged VALORs plotted")
    ap.add_argument("--reference", default=None,
                    help="CSV whose VALORs get equal weight in weights_reference.txt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    r = pd.read_csv(a.data, index_col=0, parse_dates=True)
    r.columns = [str(c) for c in r.columns]
    tab = screen(r, a.q_tail)
    tab.to_csv(os.path.join(a.out, "screen.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80,
                           "display.float_format", "{:.3g}".format):
        print(tab[tab.status != "ok"].to_string(index=False))
    n = tab.status.value_counts()
    print(f"\n{len(tab)} VALORs: {n.get('BAD', 0)} BAD, {n.get('WARN', 0)} WARN, {n.get('ok', 0)} ok")
    print("BAD: " + ",".join(tab.valor[tab.status == "BAD"]))

    figure(r, tab, a.q_tail, os.path.join(a.out, "screen.png"), a.top)
    cols = list(r.columns)
    bad = set(tab.valor[tab.status == "BAD"])
    write_weights(cols, np.array([c not in bad for c in cols]), os.path.join(a.out, "weights_clean.txt"))
    if a.reference:
        ref = {str(c) for c in pd.read_csv(a.reference, index_col=0, nrows=0).columns}
        keep = np.array([c in ref for c in cols])
        write_weights(cols, keep, os.path.join(a.out, "weights_reference.txt"))
        print(f"weights_reference.txt: {keep.sum()} VALORs of {a.reference}")
    print(f"wrote {a.out}/screen.csv, screen.png, weights_clean.txt"
          + (", weights_reference.txt" if a.reference else ""))


if __name__ == "__main__":
    main()
