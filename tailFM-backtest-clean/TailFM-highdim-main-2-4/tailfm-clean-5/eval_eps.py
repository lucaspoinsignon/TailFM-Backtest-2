"""Evaluate TailFM trained on standardised returns (train_tailfm_eps.py), on ONE split.

    python eval_eps.py --data data/returns_clean.csv \\
        --gen eps_window=runs/eps_window --gen eps_none=runs/eps_none \\
        --orig runs/final3

Everything is scored on the held-out blocks of the split.json shared by the runs.
Equal-weight portfolio.

A. The standardised returns themselves (no volatility involved).  Real eps on the
   held-out rows are close to i.i.d., so comparing them with generated eps is a valid
   out-of-sample check of the generator alone:
     sd / q01 / q05 / kurtosis of the equal-weight eps-portfolio, mean pairwise
     correlation of eps, and the spread of window-level RMS (real eps windows have an
     almost constant scale; a generator that inherits a random scale shows a large one).

B. Risk backtest at h = 1 and h = 10, same held-out days for every model:
     HS            training windows, one law for every day      (reference, unconditional)
     TailFM orig   your original run, one law for every day     (if --orig is given)
     FHS           real training eps x sigma_tau                (benchmark for the eps runs)
     <label>       generated eps x sigma_tau                    (each --gen run)
   sigma_tau is each valor's EWMA volatility from returns before the forecast date tau,
   with the lam / start variances saved by train_tailfm_eps.py; over h days it is rolled
   forward on the scenario path.  FHS and the eps runs share sigma_tau exactly, so any
   difference between them is the generator.
     exc95 / exc99   exceptions (expected 5% / 1% of days)
     95% by regime   exception rate on calm / middle / stressed days (h = 1; tercile of
                     the portfolio's trailing 63-day vol, cuts from training rows)
     ES ratio        realised loss / predicted ES on the 97.5% exception days (target 1)
     Z1 Z2 Z3        Acerbi-Szekely at 97.5% (and Z2, Z3 at 99%) with the day-specific
                     law P_tau, Monte Carlo p-values (reject if p < 0.05)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy import special

from csvio import load_returns
from splits import Split

_EPS = 1e-12


# ----------------------------------------------------------------- volatility
def ewma_sigma(r, lam, init_var, floor=0.05):
    s2 = np.empty_like(r)
    s2[0] = init_var
    lo = floor ** 2 * init_var
    for t in range(1, r.shape[0]):
        s2[t] = np.maximum(lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2, lo)
    return np.sqrt(s2)


def scaled_scenarios(eps, sig, starts, w, h, lam, init_var, floor=0.05):
    """(M, >=h, f) eps windows -> (M, N) h-day P&L, column i started from sig[starts[i]],
    sigma rolled forward along each scenario path."""
    e = np.asarray(eps[:, :h, :], dtype=np.float32)
    w32 = w.astype(np.float32)
    if h == 1:
        return ((e[:, 0, :] * w32) @ sig[starts].T.astype(np.float32)).astype(float)
    lo = (floor ** 2 * init_var).astype(np.float32)
    lam = np.float32(lam)
    S = np.empty((e.shape[0], len(starts)))
    for i, tau in enumerate(starts):
        s2 = np.broadcast_to((sig[tau] ** 2).astype(np.float32), (e.shape[0], e.shape[2])).copy()
        tot = np.zeros(e.shape[0], dtype=np.float32)
        for k in range(h):
            rk = np.sqrt(s2) * e[:, k]
            tot += rk @ w32
            s2 = np.maximum(lam * s2 + (1 - lam) * rk * rk, lo)
        S[:, i] = tot
    return S


# ----------------------------------------------------- day-specific law P_tau
class CondLaw:
    """Column t of S = scenarios of the P&L on held-out date t (a constant column
    for an unconditional model)."""

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
        m = int(n * a)
        A, B = n - m, m
        s = 1.0 - np.arange(self.M + 1) / self.M
        W = (n / m) * (B / (A + B) - s * special.betainc(A, B, s)
                       + A / (A + B) * special.betainc(A + 1, B, s))
        return -(np.diff(W) @ self.S)


def z_tests(x, P, a, n_sim, rng, chunk=200):
    """Acerbi-Szekely Z1, Z2, Z3 for a day-specific law; (stat, p) each."""
    n = x.size
    var, es = P.var(a), P.es(a)
    ind = (x + var) < 0
    nt = int(ind.sum())
    z1 = (x[ind] / es[ind]).sum() / nt + 1 if nt else np.nan
    z2 = (x * ind / es).sum() / (n * a) + 1
    sims = P.ppf(rng.random((n_sim, n)))
    si = (sims + var) < 0
    nts = si.sum(axis=1)
    s1 = ((sims * si / es).sum(axis=1) / np.maximum(nts, 1) + 1)[nts > 0]
    s2 = (sims * si / es).sum(axis=1) / (n * a) + 1
    out = [(z1, float(np.mean(s1 < z1)) if nt else np.nan),
           (z2, float(np.mean(s2 < z2)))]
    m = int(n * a)
    if m < 1:
        return out + [(np.nan, np.nan)]
    den = P.es_estimator_mean(n, a)
    u_m = np.sort(P.cdf(x))[:m]
    z3 = float(-np.mean(-P.ppf(u_m[:, None]).mean(axis=0) / den) + 1)
    zs = np.empty(n_sim)
    for i in range(0, n_sim, chunk):
        b = min(i + chunk, n_sim)
        v = np.partition(rng.random((b - i, n)), m - 1, axis=1)[:, :m]
        zs[i:b] = -(-P.ppf(v[:, :, None]).mean(axis=1) / den).mean(axis=1) + 1
    return out + [(z3, float(np.mean(zs < z3)))]


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/returns_clean.csv")
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--gen", action="append", required=True, metavar="LABEL=DIR",
                    help="run directory of train_tailfm_eps.py; repeat to compare")
    ap.add_argument("--orig", default=None, help="run directory of the original model")
    ap.add_argument("--horizons", default="1,10")
    ap.add_argument("--max-scen", type=int, default=20000)
    ap.add_argument("--n-sim", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    runs = {}
    for spec in a.gen:
        lab, _, d = spec.rpartition("=")
        runs[lab or os.path.basename(d.rstrip("/"))] = d
    first = next(iter(runs.values()))
    r = load_returns(a.data, a.prices)
    T, f = r.shape
    w = np.full(f, 1.0 / f)
    sp = Split.load(os.path.join(first, "split.json"))
    sp.check_data(os.path.abspath(a.data), T)
    vf = json.load(open(os.path.join(first, "vol_filter.json")))
    for lab, d in runs.items():
        if Split.load(os.path.join(d, "split.json")).id() != sp.id():
            raise SystemExit(f"{lab}: different split from {first}")
        if json.load(open(os.path.join(d, "vol_filter.json")))["init_var"] != vf["init_var"]:
            raise SystemExit(f"{lab}: different volatility filter from {first}")
    if a.orig and Split.load(os.path.join(a.orig, "split.json")).id() != sp.id():
        print(f"WARNING: {a.orig} has a different split -- original model skipped")
        a.orig = None
    lam, floor = vf["lam"], vf["floor"]
    iv = np.asarray(vf["init_var"])
    sig = ewma_sigma(r, lam, iv, floor)
    eps_all = r / sig
    n = sp.n
    print(sp.summary())
    print(f"EWMA lam={lam}, floor={floor} | runs: {', '.join(runs)}"
          + (f" | original: {a.orig}" if a.orig else ""))

    gens = {lab: np.load(os.path.join(d, "generated_windows.npy"), mmap_mode="r")
            for lab, d in runs.items()}

    # ------------------------------------------------------- A. eps diagnostics
    def stats(rows):
        p = rows @ w
        c = np.corrcoef(rows.T)
        k = ((p - p.mean()) ** 4).mean() / p.var() ** 2
        return (p.std(), np.quantile(p, 0.01), np.quantile(p, 0.05), k,
                np.nanmean(c[np.triu_indices_from(c, 1)]))

    tr_rows, te_rows = eps_all[sp.train_row_mask()], eps_all[sp.test_row_mask()]
    tr_win = sp.train_windows(eps_all, n)
    rms = lambda W: np.sqrt((np.asarray(W) ** 2).mean(axis=(1, 2)))
    print("\n=== A. standardised returns eps (no volatility involved) ===")
    print(f"{'':>16s} {'sd':>7s} {'q01':>7s} {'q05':>7s} {'kurt':>6s} {'mean corr':>9s} "
          f"{'window-RMS sd':>13s}")
    row = lambda lab, st, wr: print(f"{lab:>16s} {st[0]:7.3f} {st[1]:+7.3f} {st[2]:+7.3f} "
                                    f"{st[3]:6.1f} {st[4]:9.3f} {wr:>13s}")
    row("real, train", stats(tr_rows), f"{rms(tr_win).std():.3f}")
    row("real, held-out", stats(te_rows), "-")
    for lab, g in gens.items():
        G = np.asarray(g[: min(len(g), 4000)])
        row(lab, stats(G.reshape(-1, f)), f"{rms(G).std():.3f}")
    print("(real held-out eps is the out-of-sample target; generated should match it)")

    # ---------------------------------------------------------- B. backtest
    xm = r @ w
    c1, c2 = np.concatenate([[0], np.cumsum(xm)]), np.concatenate([[0], np.cumsum(xm ** 2)])
    L, state = 63, np.full(T, np.nan)
    t_ = np.arange(L, T)
    state[t_] = np.sqrt((c2[t_] - c2[t_ - L]) / L - ((c1[t_] - c1[t_ - L]) / L) ** 2)
    trm = sp.train_row_mask() & np.isfinite(state)
    cuts = np.quantile(state[trm], [1 / 3, 2 / 3])

    raw_win = sp.train_windows(r, n)
    for h in [int(v) for v in a.horizons.split(",")]:
        starts = np.array([s + i * h for s in sorted(sp.test_starts)
                           for i in range(sp.block // h)])
        x = np.array([r[s:s + h].sum(axis=0) @ w for s in starts])
        N = x.size
        reg = np.digitize(np.nan_to_num(state[starts], nan=cuts[0]), cuts)

        def uncond(pnl):
            return np.repeat(pnl[:, None], N, axis=1)

        laws = {"HS": lambda: uncond(raw_win[:, :h, :].sum(axis=1) @ w)}
        if a.orig:
            go = np.load(os.path.join(a.orig, "generated_windows.npy"), mmap_mode="r")
            laws["TailFM orig"] = lambda go=go: uncond(
                np.asarray(go[: a.max_scen, :h, :]).sum(axis=1) @ w)
        laws["FHS"] = lambda: scaled_scenarios(tr_win, sig, starts, w, h, lam, iv, floor)
        for lab, g in gens.items():
            laws[lab] = lambda g=g: scaled_scenarios(g[: a.max_scen], sig, starts, w, h,
                                                     lam, iv, floor)

        print(f"\n=== B. h = {h}: {N} held-out {h}-day returns "
              f"(expected exceptions {0.05 * N:.1f} / {0.01 * N:.1f}) ===")
        hdr = f"{'model':>13s} {'exc95':>6s} {'exc99':>6s}"
        if h == 1:
            hdr += f" | {'95% calm':>8s} {'mid':>6s} {'stress':>6s}"
        hdr += (f" | {'ES ratio':>8s} | {'Z1 97.5':>13s} {'Z2 97.5':>13s} {'Z3 97.5':>13s}"
                f" | {'Z2 99':>13s} {'Z3 99':>13s}")
        print(hdr)
        for lab, build in laws.items():
            P = CondLaw(build())
            v95, v99, v975, es975 = P.var(0.05), P.var(0.01), P.var(0.025), P.es(0.025)
            e95, e99, hit = x < -v95, x < -v99, x < -v975
            line = f"{lab:>13s} {int(e95.sum()):6d} {int(e99.sum()):6d}"
            if h == 1:
                rate = [e95[reg == g].mean() if (reg == g).any() else np.nan for g in range(3)]
                line += f" | {rate[0]:8.1%} {rate[1]:6.1%} {rate[2]:6.1%}"
            ratio = (-x[hit]).sum() / es975[hit].sum() if hit.any() else np.nan
            line += f" | {ratio:8.2f} |"
            cells = z_tests(x, P, 0.025, a.n_sim, rng) + z_tests(x, P, 0.01, a.n_sim, rng)[1:]
            for i, (z, p) in enumerate(cells):
                txt = "n/a" if not np.isfinite(z) else f"{z:+.2f} (p={p:.2f})"
                line += f" {txt:>13s}" + (" |" if i == 2 else "")
            print(line)
            del P
    print(f"\n(done in {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
