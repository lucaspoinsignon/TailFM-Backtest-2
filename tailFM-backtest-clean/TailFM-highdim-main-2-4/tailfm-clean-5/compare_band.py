"""Scorecard: several generated samples against the same real days, one table.

    python compare_band.py --data data/returns_clean.csv --run runs/cond_all \\
        --cond-band 0.85 0.95 \\
        --gen point=runs/cond_all/generated_windows_u0.90.npy \\
        --gen band=runs/cond_all/generated_windows_u0.85-0.95.npy

Real = the windows starting on the chosen days: --cond-band (u from --run's
cond_u.npy) and/or --test-period (windows lying inside the periods; any dates).
Columns:
  CVaR99, CVaR99.5   per-asset 1-day loss CVaR, median over assets of gen / real
  lamL(5%)           lower tail dependence, mean over ALL pairs, 1-day
  rho_S              Spearman correlation, mean over all pairs, 1-day
  1d / 10d           equal-weight portfolio loss VaR / CVaR in %, 10d = first 10 days
The real 10-day 99% figures rest on a handful of episodes: read them with care.
"""
import argparse
import numpy as np
from scipy.stats import rankdata
from csvio import load_returns
from regime import load_dates, period_mask

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--run", default=None, help="run whose cond_u.npy defines --cond-band")
ap.add_argument("--cond-band", type=float, nargs=2, default=None, metavar=("LO", "HI"))
ap.add_argument("--test-period", nargs=2, action="append", metavar=("START", "END"))
ap.add_argument("--gen", action="append", required=True, metavar="LABEL=PATH")
ap.add_argument("--q", type=float, default=0.05, help="tail-dependence quantile")
a = ap.parse_args()

r = load_returns(a.data, False); T, f = r.shape
gens = {s.split("=", 1)[0]: np.load(s.split("=", 1)[1]) for s in a.gen}
n = next(iter(gens.values())).shape[1]
inside = (period_mask(load_dates(a.data, T), a.test_period) if a.test_period
          else np.ones(T, dtype=bool))
c = np.concatenate([[0], np.cumsum(inside)])
starts = np.arange(T - n + 1)
starts = starts[c[starts + n] - c[starts] == n]          # whole window inside
if a.cond_band:
    if not a.run:
        raise SystemExit("--cond-band needs --run (for cond_u.npy)")
    u = np.load(f"{a.run}/cond_u.npy")
    starts = starts[(u[starts] >= a.cond_band[0]) & (u[starts] <= a.cond_band[1])]
real = np.stack([r[t:t + n] for t in starts])


def cvar(loss, p):                                       # loss (N, ...) -> per column
    k = max(int(round((1 - p) * loss.shape[0])), 1)
    return np.sort(loss, axis=0)[-k:].mean(axis=0)


def offdiag_mean(m):
    return (m.sum() - np.trace(m)) / (m.shape[0] * (m.shape[0] - 1))


def metrics(w):
    x1 = w[:, 0, :]
    U = (rankdata(x1, axis=0) - 0.5) / x1.shape[0]
    B = (U <= a.q).astype(float)
    lam = offdiag_mean(B.T @ B / x1.shape[0] / a.q)
    R = U - U.mean(axis=0)
    rho = offdiag_mean(np.corrcoef(R.T))
    p1 = -x1.mean(axis=1)
    p10 = -w[:, :10, :].mean(axis=2).sum(axis=1)
    return dict(cvar99=cvar(-x1, .99), cvar995=cvar(-x1, .995), lam=lam, rho=rho,
                v1=np.quantile(p1, .99) * 100, c1=cvar(p1, .99) * 100,
                v10_95=np.quantile(p10, .95) * 100, v10=np.quantile(p10, .99) * 100,
                c10_95=cvar(p10, .95) * 100, c10=cvar(p10, .99) * 100)


m0 = metrics(real)
print(f"{starts.size} real windows"
      + (f" with u in [{a.cond_band[0]}, {a.cond_band[1]}]" if a.cond_band else "")
      + (" inside " + ", ".join(f"{p} -> {q}" for p, q in a.test_period)
         if a.test_period else "") + "\n")
cols = ("CVaR99 CVaR99.5 lamL(5%)  rho_S | 1d VaR99 CVaR99 | 10d VaR95 VaR99 "
        "CVaR95 CVaR99")
print(f"{'':>12s} {cols}")


def row(label, m, ratio):
    c99 = "    1.00" if not ratio else f"{np.median(m['cvar99'] / m0['cvar99']):8.2f}"
    c995 = "     1.00" if not ratio else f"{np.median(m['cvar995'] / m0['cvar995']):9.2f}"
    print(f"{label:>12s} {c99}{c995} {m['lam']:8.3f} {m['rho']:6.3f} |"
          f" {m['v1']:8.2f} {m['c1']:6.2f} | {m['v10_95']:9.2f} {m['v10']:5.2f}"
          f" {m['c10_95']:6.2f} {m['c10']:6.2f}")


row("real", m0, False)
for label, g in gens.items():
    row(label, metrics(g), True)
