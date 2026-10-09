"""Same placebo as placebo_hs.py, with ONE change: condition on volatility.

For each random split, two predictive laws built from the SAME training windows and
scored on the SAME held-out blocks with the same tests:

  HS   unconditional: the training windows' h-step P&L, one law P for every date
  FHS  conditional (filtered historical simulation): the training windows are first
       divided by each asset's EWMA volatility forecast, eps = r / sigma, and rescaled
       by the volatility at the forecast date tau,

           sigma^2_{t,j} = lam sigma^2_{t-1,j} + (1-lam) r^2_{t-1,j}      (only past data)
           r_{tau+k} = sigma_{tau+k} * eps_k,   sigma rolled forward on the scenario path

       so P_tau changes with the market state at tau.

If HS is rejected far more often than 5% while FHS is close to 5%, the tests are valid
once the volatility regime is conditioned on, and the earlier rejections were the
regime clustering.  Self-contained: only reads the repo's csvio / splits / backtest.

v2 adds one diagnostic table and changes nothing else (the rejection tables are
identical to v1): at h = 1, the exception rate of HS and FHS by market regime,
pooled over the splits.  The regime of a held-out day is the tercile of the portfolio's
trailing 63-day realised volatility (returns strictly before that day; terciles cut on
the training rows).  A calibrated law has ~5% / ~1% in EVERY regime; the pattern shows
where a law is too cautious (rate below target) or too aggressive (above).

    python placebo_fhs.py --data data/returns_clean.csv --splits 50
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy import special

from csvio import load_returns
from splits import block_split
from backtest import (predictive_from_windows, realized_pnl, pf_test,
                      test_z1, test_z2, test_z3)

_EPS = 1e-12


# ------------------------------------------------------------ volatility filter
def ewma_sigma(r, lam, init_var, floor=0.05):
    """sigma_t from returns up to t-1 only; sigma^2_0 = training-row variance."""
    s2 = np.empty_like(r)
    s2[0] = init_var
    lo = floor ** 2 * init_var
    for t in range(1, r.shape[0]):
        s2[t] = np.maximum(lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2, lo)
    return np.sqrt(s2)


def fhs_scenarios(eps, sig, starts, w, h, lam, init_var, floor=0.05):
    """(M, n, f) innovation windows -> (M, N) h-step P&L, column i given the state at
    starts[i]: the EWMA recursion is run forward along each scenario path."""
    e = eps[:, :h, :]
    if h == 1:
        return (e[:, 0, :] * w) @ sig[starts].T
    lo = floor ** 2 * init_var
    S = np.empty((e.shape[0], len(starts)))
    for i, tau in enumerate(starts):
        s2 = np.broadcast_to(sig[tau] ** 2, (e.shape[0], e.shape[2])).copy()
        tot = np.zeros(e.shape[0])
        for k in range(h):
            rk = np.sqrt(s2) * e[:, k]
            tot += rk @ w
            s2 = np.maximum(lam * s2 + (1 - lam) * rk ** 2, lo)
        S[:, i] = tot
    return S


# ------------------------------------------------- per-date predictive law P_t
class CondLaw:
    """Column t of S = scenarios of the P&L on held-out date t."""

    def __init__(self, S):
        self.S = np.sort(S, axis=0)
        self.M, self.N = S.shape
        self.cols = np.arange(self.N)

    def ppf(self, p):
        p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
        p = np.broadcast_to(p, np.broadcast_shapes(p.shape, (self.N,)))
        k = np.clip(np.ceil(p * self.M).astype(np.int64) - 1, 0, self.M - 1)
        return self.S[k, self.cols]

    def cdf(self, x):
        u = (self.S <= x[None, :]).sum(axis=0) / self.M
        return np.clip(u, 0.5 / self.M, 1 - 0.5 / self.M)

    def var(self, a):
        return -self.ppf(a)

    def es(self, a):
        k = int(np.floor(a * self.M))
        integral = self.S[:k].sum(axis=0) / self.M
        if k < self.M:
            integral = integral + (a - k / self.M) * self.S[k]
        return -integral / a

    def es_estimator_mean(self, n, a):
        """Acerbi-Szekely eq. (11) per date, integrated exactly over the M cells."""
        m = int(n * a)
        A, B = n - m, m
        s = 1.0 - np.arange(self.M + 1) / self.M
        W = (n / m) * (B / (A + B) - s * special.betainc(A, B, s)
                       + A / (A + B) * special.betainc(A + 1, B, s))
        return -(np.diff(W) @ self.S)


# ----------------------------------------- Acerbi-Szekely with a time-varying P_t
def cond_z1_z2(x, P, a, n_sim, rng):
    var, es = P.var(a), P.es(a)
    ind = (x + var) < 0
    n, nt = x.size, int(ind.sum())
    z1 = (x[ind] / es[ind]).sum() / nt + 1 if nt else np.nan
    z2 = (x * ind / es).sum() / (n * a) + 1
    sims = P.ppf(rng.random((n_sim, n)))                     # X_t ~ P_t, independent
    si = (sims + var) < 0
    nts = si.sum(axis=1)
    s1 = ((sims * si / es).sum(axis=1) / np.maximum(nts, 1) + 1)[nts > 0]
    s2 = (sims * si / es).sum(axis=1) / (n * a) + 1
    p1 = float(np.mean(s1 < z1)) if nt else np.nan
    return (z1, p1), (z2, float(np.mean(s2 < z2)))


def cond_z3(x, P, a, n_sim, rng, chunk=200):
    n = x.size
    m = int(n * a)
    den = P.es_estimator_mean(n, a)
    u_m = np.sort(P.cdf(x))[:m]          # P_t^{-1} is monotone: only the m smallest ranks
    z = float(-np.mean(-P.ppf(u_m[:, None]).mean(axis=0) / den) + 1)
    zs = np.empty(n_sim)
    for i in range(0, n_sim, chunk):
        b = min(i + chunk, n_sim)
        v = np.partition(rng.random((b - i, n)), m - 1, axis=1)[:, :m]
        zs[i:b] = -(-P.ppf(v[:, :, None]).mean(axis=1) / den).mean(axis=1) + 1
    return z, float(np.mean(zs < z))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/returns_clean.csv")
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--n-sim", type=int, default=2000)
    ap.add_argument("--lam", type=float, default=0.94, help="EWMA decay (RiskMetrics)")
    a = ap.parse_args()

    r = load_returns(a.data, a.prices)
    T, f = r.shape
    w = np.full(f, 1.0 / f)
    var_lv, es_lv = (0.95, 0.99), (0.975, 0.99)
    tests = [f"PF {lv:g}" for lv in var_lv] + \
            [f"{z} {lv:g}" for lv in es_lv for z in ("Z1", "Z2", "Z3")]
    H = (1, a.horizon)
    pv = {m: {h: {k: [] for k in tests} for h in H} for m in ("HS", "FHS")}
    zv = {m: {h: {k: [] for k in tests[2:]} for h in H} for m in ("HS", "FHS")}
    rng = np.random.default_rng(0)
    # regime state: trailing 63-day realised vol of the portfolio, past data only
    xp = r @ w
    c1, c2 = np.concatenate([[0], np.cumsum(xp)]), np.concatenate([[0], np.cumsum(xp ** 2)])
    L = 63
    state_vol = np.full(T, np.nan)
    t_ = np.arange(L, T)
    state_vol[t_] = np.sqrt((c2[t_] - c2[t_ - L]) / L - ((c1[t_] - c1[t_ - L]) / L) ** 2)
    terc = {"days": np.zeros(3)}
    terc.update({(m, lv): np.zeros(3) for m in ("HS", "FHS") for lv in var_lv})

    t0 = time.time()
    for seed in range(a.splits):
        sp = block_split(T, a.n, a.horizon, test_frac=a.test_frac, seed=seed,
                         data=os.path.abspath(a.data))
        iv = sp.train_rows(r).var(axis=0)
        iv = np.where(iv > 0, iv, 1e-12)
        sig = ewma_sigma(r, a.lam, iv)
        win = sp.train_windows(r, a.n)               # identical windows for both laws
        eps = sp.train_windows(r / sig, a.n)
        for h in H:
            x = realized_pnl(sp.test_windows(r, h), w, h)
            starts = np.array([s + i * h for s in sorted(sp.test_starts)
                               for i in range(sp.block // h)])
            laws = {"HS": predictive_from_windows(win, w, h),
                    "FHS": CondLaw(fhs_scenarios(eps, sig, starts, w, h, a.lam, iv))}
            if h == 1:
                trm = sp.train_row_mask() & np.isfinite(state_vol)
                cuts = np.quantile(state_vol[trm], [1 / 3, 2 / 3])
                ok = np.isfinite(state_vol[starts])
                regime = np.digitize(state_vol[starts], cuts)        # 0 calm .. 2 stressed
                terc["days"] += np.bincount(regime[ok], minlength=3)
            for m, P in laws.items():
                for lv in var_lv:
                    if h == 1:
                        exc = ((x + P.var(1 - lv)) < 0)[ok]
                        terc[(m, lv)] += np.bincount(regime[ok], weights=exc, minlength=3)
                    n_exc = int(((x + P.var(1 - lv)) < 0).sum())
                    pv[m][h][f"PF {lv:g}"].append(pf_test(n_exc, x.size, 1 - lv)["p_value"])
                for lv in es_lv:
                    if m == "HS":
                        res = [fn(x, P, 1 - lv, n_sim=a.n_sim, seed=seed)
                               for fn in (test_z1, test_z2, test_z3)]
                        out = [(q.statistic, q.p_value if q.defined else np.nan) for q in res]
                    else:
                        (z1, z2) = cond_z1_z2(x, P, 1 - lv, a.n_sim, rng)
                        out = [z1, z2, cond_z3(x, P, 1 - lv, a.n_sim, rng)]
                    for name, (zz, pp) in zip(("Z1", "Z2", "Z3"), out):
                        pv[m][h][f"{name} {lv:g}"].append(pp)
                        zv[m][h][f"{name} {lv:g}"].append(zz)
        if seed == 0:
            print(f"split seed 0 (your run's split):\n  {sp.summary()}")
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{a.splits} splits ({time.time() - t0:.0f}s)")

    for h in H:
        print(f"\n=== h = {h}: rejection rate at 5% over {a.splits} random splits "
              f"(valid test: ~5%) ===")
        print(f"{'test':>10s} | {'HS reject':>9s} {'median Z':>9s} {'seed0 Z':>8s} | "
              f"{'FHS reject':>10s} {'median Z':>9s} {'seed0 Z':>8s}")
        for k in tests:
            row = f"{k:>10s} |"
            for m in ("HS", "FHS"):
                rate = np.nanmean(np.array(pv[m][h][k], float) < 0.05)
                if k in zv[m][h]:
                    zz = np.array(zv[m][h][k], float)
                    row += (f" {rate:{9 if m == 'HS' else 10}.0%} {np.nanmedian(zz):+9.3f}"
                            f" {zz[0]:+8.3f} |")
                else:
                    row += f" {rate:{9 if m == 'HS' else 10}.0%} {'':>9s} {'':>8s} |"
            print(row.rstrip(" |"))
        if h == 1:
            d = terc["days"]
            print(f"\n=== h = 1: exception rate by market regime (trailing 63-day vol "
                  f"tercile), pooled over {a.splits} splits ===")
            print(f"{'':>10s} {'calm':>9s} {'middle':>9s} {'stressed':>9s}   target")
            print(f"{'days':>10s} " + " ".join(f"{v:9.0f}" for v in d))
            for lv in var_lv:
                for m in ("HS", "FHS"):
                    rates = terc[(m, lv)] / np.maximum(d, 1)
                    print(f"{m + ' ' + format(lv, 'g'):>10s} "
                          + " ".join(f"{v:9.2%}" for v in rates) + f"   {1 - lv:.0%}")


if __name__ == "__main__":
    main()
