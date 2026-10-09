"""Which volatility model?  FHS with four volatility models, nothing else changes.

    python vol_model_compare.py --data data/returns_clean.csv --splits 20

FHS = real training standardised returns eps = r / sigma, rescaled by sigma on the
forecast date (and rolled forward over h days).  TailFM on eps tracks FHS closely, so
the volatility model can be chosen here, before any retraining.

  EWMA 0.94   sigma^2_t = 0.94 sigma^2_{t-1} + 0.06 r^2_{t-1}          (current)
  EWMA 0.90   faster reaction
  EWMA 0.97   slower reaction (control)
  GJR-GARCH   sigma^2_t = w + (a + g 1{r_{t-1}<0}) r^2_{t-1} + b sigma^2_{t-1}, per valor,
              w = v (1 - a - g/2 - b) with v the training variance (mean reversion to v),
              (a, g, b) by Gaussian quasi-likelihood on the training rows over a grid

Every model uses returns before t only.  Two views:

  1. split seed 0 (your runs' split): same table as eval_eps.py.  The "EWMA 0.94"
     row reproduces eval_eps's FHS row (p-values differ slightly: Monte Carlo).
  2. --splits random splits: rejection rates (valid test ~5%), exception rates pooled
     over splits, by regime, and P(exception | exception the day before) -- the
     direct measure of bursts (5% if exceptions do not cluster).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from csvio import load_returns
from splits import block_split
from backtest.kupiec import pf_test
from eval_eps import CondLaw, z_tests

FLOOR = 0.05


# ------------------------------------------------------------- vol models
class EWMA:
    def __init__(self, lam):
        self.lam, self.name = lam, f"EWMA {lam:.2f}"

    def fit(self, r, train_mask):
        v = r[train_mask].var(axis=0)
        self.v = np.where(v > 0, v, 1e-12)
        self.lo = FLOOR ** 2 * self.v
        return self

    def step(self, s2, r_prev):
        return np.maximum(self.lam * s2 + (1 - self.lam) * r_prev ** 2, self.lo)


class GJR:
    name = "GJR-GARCH"
    A = (0.0, 0.02, 0.04, 0.06, 0.09, 0.12)
    G = (0.0, 0.04, 0.08, 0.12, 0.18)
    B = (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.975)

    def fit(self, r, train_mask):
        v = r[train_mask].var(axis=0)
        self.v = np.where(v > 0, v, 1e-12)
        self.lo = FLOOR ** 2 * self.v
        grid = np.array([(a, g, b) for a in self.A for g in self.G for b in self.B
                         if a + g / 2 + b <= 0.995 and a + g > 0])
        a, g, b = (grid[:, i:i + 1] for i in range(3))            # (K, 1)
        w = self.v[None, :] * (1 - a - g / 2 - b)                  # (K, f)
        s2 = np.broadcast_to(self.v, w.shape).copy()
        ll = np.zeros_like(w)
        for t in range(1, r.shape[0]):
            rp = r[t - 1]
            s2 = np.maximum(w + (a + g * (rp < 0)) * rp ** 2 + b * s2, self.lo)
            if train_mask[t]:
                ll -= 0.5 * (np.log(s2) + r[t] ** 2 / s2)
        best = ll.argmax(axis=0)                                   # (f,)
        self.a, self.g, self.b = grid[best, 0], grid[best, 1], grid[best, 2]
        self.w = self.v * (1 - self.a - self.g / 2 - self.b)
        return self

    def step(self, s2, r_prev):
        return np.maximum(self.w + (self.a + self.g * (r_prev < 0)) * r_prev ** 2
                          + self.b * s2, self.lo)


def sigma_path(model, r):
    s2 = np.empty_like(r)
    s2[0] = model.v
    for t in range(1, r.shape[0]):
        s2[t] = model.step(s2[t - 1], r[t - 1])
    return np.sqrt(s2)


def fhs_scenarios(model, eps, sig, starts, w, h):
    """(M, >=h, f) training eps -> (M, N) h-day P&L from sig[starts[i]]."""
    e = eps[:, :h, :]
    if h == 1:
        return (e[:, 0, :] * w) @ sig[starts].T
    S = np.empty((e.shape[0], len(starts)))
    for i, tau in enumerate(starts):
        s2 = np.broadcast_to(sig[tau] ** 2, (e.shape[0], e.shape[2])).copy()
        tot = np.zeros(e.shape[0])
        for k in range(h):
            rk = np.sqrt(s2) * e[:, k]
            tot += rk @ w
            s2 = model.step(s2, rk)
        S[:, i] = tot
    return S


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/returns_clean.csv")
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--n-sim", type=int, default=1000)
    a = ap.parse_args()

    r = load_returns(a.data, a.prices)
    T, f = r.shape
    w = np.full(f, 1.0 / f)
    rng = np.random.default_rng(0)
    xm = r @ w
    c1, c2 = np.concatenate([[0], np.cumsum(xm)]), np.concatenate([[0], np.cumsum(xm ** 2)])
    L, state = 63, np.full(T, np.nan)
    t_ = np.arange(L, T)
    state[t_] = np.sqrt((c2[t_] - c2[t_ - L]) / L - ((c1[t_] - c1[t_ - L]) / L) ** 2)

    makers = [lambda: EWMA(0.94), lambda: EWMA(0.90), lambda: EWMA(0.97), lambda: GJR()]
    names = [m().name for m in makers]
    H = (1, a.horizon)
    tests = ["PF95", "PF99", "Z1 97.5", "Z2 97.5", "Z3 97.5", "Z2 99", "Z3 99"]
    agg = {(m, h): dict(rej={k: [] for k in tests}, n=0, e95=0, e99=0,
                        reg_n=np.zeros(3), reg_e=np.zeros(3), prev=0, prev_e=0,
                        loss=0.0, es=0.0) for m in names for h in H}
    t0 = time.time()
    for seed in range(a.splits):
        sp = block_split(T, a.n, a.horizon, test_frac=a.test_frac, seed=seed,
                         data=os.path.abspath(a.data))
        trm = sp.train_row_mask()
        cuts = np.quantile(state[trm & np.isfinite(state)], [1 / 3, 2 / 3])
        if seed == 0:
            print(f"split seed 0 = your runs' split: {sp.summary()}")
            gjr_info, rows0 = None, {h: [] for h in H}
        for mk in makers:
            model = mk().fit(r, trm)
            if seed == 0 and isinstance(model, GJR):
                gjr_info = (np.median(model.a), np.median(model.g), np.median(model.b))
            sig = sigma_path(model, r)
            eps = sp.train_windows(r / sig, a.n)
            for h in H:
                starts = np.array([s + i * h for s in sorted(sp.test_starts)
                                   for i in range(sp.block // h)])
                x = np.array([r[s:s + h].sum(axis=0) @ w for s in starts])
                P = CondLaw(fhs_scenarios(model, eps, sig, starts, w, h))
                v95, v99, v975, es975 = P.var(0.05), P.var(0.01), P.var(0.025), P.es(0.025)
                e95, e99, hit = x < -v95, x < -v99, x < -v975
                z = z_tests(x, P, 0.025, a.n_sim, rng) + z_tests(x, P, 0.01, a.n_sim, rng)[1:]
                d = agg[(model.name, h)]
                d["rej"]["PF95"].append(pf_test(int(e95.sum()), x.size, 0.05)["p_value"] < 0.05)
                d["rej"]["PF99"].append(pf_test(int(e99.sum()), x.size, 0.01)["p_value"] < 0.05)
                for k, (_, p) in zip(tests[2:], z):
                    d["rej"][k].append(p < 0.05 if np.isfinite(p) else np.nan)
                d["n"] += x.size; d["e95"] += e95.sum(); d["e99"] += e99.sum()
                d["loss"] += (-x[hit]).sum(); d["es"] += es975[hit].sum()
                if h == 1:
                    reg = np.digitize(np.nan_to_num(state[starts], nan=cuts[0]), cuts)
                    d["reg_n"] += np.bincount(reg, minlength=3)
                    d["reg_e"] += np.bincount(reg, weights=e95, minlength=3)
                    same = np.diff(starts) == 1                     # consecutive days
                    d["prev"] += int((e95[:-1] & same).sum())
                    d["prev_e"] += int((e95[1:] & e95[:-1] & same).sum())
                if seed == 0:
                    ratio = (-x[hit]).sum() / es975[hit].sum() if hit.any() else np.nan
                    rows0[h].append(
                        f"  {model.name:>10s} {int(e95.sum()):6d} {int(e99.sum()):6d} "
                        f"{ratio:8.2f} | " + " ".join(
                            f"{'n/a' if not np.isfinite(s) else f'{s:+.2f} (p={p:.2f})':>13s}"
                            for s, p in z))
        if seed == 0:
            print(f"\n=== 1. split seed 0: same held-out days as eval_eps.py ===")
            for h in H:
                n_h = len(sorted(sp.test_starts)) * (sp.block // h)
                print(f"\n  h = {h} (expected exceptions {0.05 * n_h:.1f} / {0.01 * n_h:.1f})")
                print(f"  {'vol model':>10s} {'exc95':>6s} {'exc99':>6s} {'ES ratio':>8s} | "
                      + " ".join(f"{k:>13s}" for k in tests[2:]))
                print("\n".join(rows0[h]))
            if gjr_info:
                print(f"\n  GJR median parameters on this split: a={gjr_info[0]:.3f} "
                      f"g={gjr_info[1]:.3f} b={gjr_info[2]:.3f}")
            print()
        print(f"  [{seed + 1}/{a.splits} splits, {time.time() - t0:.0f}s]")

    for h in H:
        print(f"\n=== 2. h = {h}: {a.splits} random splits ===")
        hdr = f"{'vol model':>10s} {'exc95':>6s} {'exc99':>6s} {'ES ratio':>8s}"
        if h == 1:
            hdr += f" | {'95% calm':>8s} {'mid':>5s} {'stress':>6s} | {'P(e|e-1)':>8s}"
        print(hdr + " | reject@5%: " + " ".join(f"{k:>7s}" for k in tests))
        print(f"{'target':>10s} {'5.0%':>6s} {'1.0%':>6s} {'1.00':>8s}" +
              (f" | {'5%':>8s} {'5%':>5s} {'5%':>6s} | {'5%':>8s}" if h == 1 else "")
              + " |            " + " ".join(f"{'5%':>7s}" for _ in tests))
        for m in names:
            d = agg[(m, h)]
            line = (f"{m:>10s} {d['e95'] / d['n']:6.1%} {d['e99'] / d['n']:6.1%} "
                    f"{d['loss'] / d['es']:8.2f}")
            if h == 1:
                rr = d["reg_e"] / np.maximum(d["reg_n"], 1)
                pe = d["prev_e"] / max(d["prev"], 1)
                line += f" | {rr[0]:8.1%} {rr[1]:5.1%} {rr[2]:6.1%} | {pe:8.1%}"
            line += " |            " + " ".join(f"{np.nanmean(d['rej'][k]):7.0%}" for k in tests)
            print(line)
    print(f"\n(done in {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
