"""Which real windows make a regime band's 10-day tail: onsets (vol rising at the
start) or aftermaths (vol falling)?  No model involved.

    python diag_band.py --data data/returns_clean.csv --run runs/cond_multi2 \\
        --valor 4155001 --band 0.85 0.95

momentum = sigma_t / sigma_{t-L} of the conditioning series (> 1: vol rising).  If the
worst windows are mostly onsets while the band is mostly aftermaths, the vol level
alone cannot tell the model which windows can turn into a crash.
"""
import argparse
import numpy as np
import pandas as pd
from csvio import load_returns, feature_names_from_csv
from regime import ewma_sigma

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--run", required=True, help="run whose cond_u.npy defines the band")
ap.add_argument("--valor", required=True, help="conditioning VALOR of that run, or EW")
ap.add_argument("--band", type=float, nargs=2, required=True, metavar=("LO", "HI"))
ap.add_argument("--lam", type=float, default=0.94)
ap.add_argument("--lag", type=int, default=10)
ap.add_argument("--h", type=int, default=10, help="loss horizon in days")
ap.add_argument("--top", type=int, default=15)
a = ap.parse_args()

r = load_returns(a.data, False); T, f = r.shape
dates = pd.to_datetime(pd.read_csv(a.data, usecols=[0]).iloc[-T:, 0]).dt.date.to_numpy()
rj = r.mean(axis=1) if a.valor == "EW" else r[:, feature_names_from_csv(a.data, f).index(a.valor)]
sig = ewma_sigma(rj, a.lam, rj.var(), 1e-12)
mom = sig / np.concatenate([np.full(a.lag, sig[0]), sig[:-a.lag]])
u = np.load(f"{a.run}/cond_u.npy")

days = np.flatnonzero((u >= a.band[0]) & (u <= a.band[1]) & (np.arange(T) <= T - a.h))
loss = np.array([-r[t:t + a.h].mean(axis=1).sum() for t in days]) * 100
rising = mom[days] > 1
order = np.argsort(-loss)[:a.top]
print(f"band [{a.band[0]}, {a.band[1]}]: {days.size} window starts, "
      f"{rising.mean():.0%} with vol rising (momentum > 1, lag {a.lag})\n")
print(f"worst {a.top} {a.h}-day equal-weight losses:")
print("  start        loss%    u   sigma%  momentum")
for i in order:
    t = days[i]
    print(f"  {dates[t]}  {loss[i]:6.2f}  {u[t]:.3f}  {sig[t]*100:5.2f}   {mom[t]:5.2f}"
          f"  {'onset' if rising[i] else 'aftermath'}")
print(f"\nshare rising among the worst {a.top}: {rising[order].mean():.0%} "
      f"(band overall {rising.mean():.0%})")
for name, m in (("rising", rising), ("falling", ~rising)):
    l = np.sort(loss[m])
    k = max(int(round(0.05 * l.size)), 1)
    print(f"  {name:8s} N={m.sum():4d}  mean {l.mean():6.2f}%  VaR95 {np.quantile(l, .95):6.2f}%"
          f"  CVaR95 {l[-k:].mean():6.2f}%")
