"""Is a conditional run calibrated in every regime?  Each real day is scored under the model's
1-day law at its own regime u.

    python regime_calibration.py --data data/returnscomp.csv --run runs/all_u --bins 10

u in [0, 1] is cut into --bins equal bins.  Per bin, --num windows are generated with u drawn
from the real days of the bin (as generate.py --match) and their day 1 is the model's 1-day
law, of each asset and of the equal-weight portfolio.  Per bin:

    N                       real days in the bin
    VaR95 VaR99 ES97.5      portfolio, model / real, same estimators on both sides
    exc95 exc99             share of the bin's days beyond the model's VaR (target 5%, 1%)
    aVaR95 aVaR99           median over assets of model / real VaR: the marginals
    rho real / gen          mean pairwise rank correlation of the day-1 cross-section
Pooled over all days, each scored under its own bin: exceptions at 95 / 99% (Kupiec) and the
PIT histogram (10 deciles, ~10% each if calibrated).

A portfolio ratio that falls as u rises while the asset ratios stay at 1 = the joint tail of
stressed regimes too thin; both falling = the regime effect on the marginals too weak.  Days
of a bin are clustered in time (few independent episodes in the top bins), so read a single
bin's exceptions as indicative and the trend across bins as the evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from backtest import PredictivePnL, pf_test
from csvio import load_returns
from run_logging import tee_output


def rank_corr(X: np.ndarray) -> float:
    """Mean pairwise Spearman correlation of the rows of X (N, f), constant columns dropped."""
    U = np.argsort(np.argsort(X, axis=0), axis=0).astype(float)
    U = U[:, X.std(axis=0) > 0]
    Z = (U - U.mean(axis=0)) / U.std(axis=0)
    f = Z.shape[1]
    return float(((Z.sum(axis=1) ** 2 - (Z ** 2).sum(axis=1)).mean()) / (f * (f - 1)))


def calibrate(r, u, draw_day1, bins=10, num=10_000, w=None, seed=0):
    """r (T, f) real returns, u (T,) their regime, draw_day1(cond) -> (len(cond), f) generated
    day-1 returns at those regimes."""
    T, f = r.shape
    w = np.full(f, 1.0 / f) if w is None else w
    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 1.0, bins + 1)
    keep = -np.quantile(r, 0.05, axis=0) >= 2e-4          # assets with a measurable VaR
    pit, exc95, exc99 = [], 0, 0
    print(f"{'u bin':>11s} {'N':>5s} | {'VaR95':>6s} {'VaR99':>6s} {'ES97.5':>6s} | "
          f"{'exc95':>6s} {'exc99':>6s} | {'aVaR95':>6s} {'aVaR99':>6s} | {'rho real':>8s} {'gen':>6s}"
          "   (portfolio and asset columns: model / real)")
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        days = np.flatnonzero((u >= lo) & ((u < hi) if b < bins - 1 else (u <= hi)))
        if days.size < 100:
            print(f"[{lo:.2f},{hi:.2f}] {days.size:5d} | too few days, skipped")
            continue
        g = draw_day1(u[rng.choice(days, num)])
        R = r[days]
        x = R @ w
        Pm, Pr = PredictivePnL(g @ w), PredictivePnL(x)
        e95, e99 = int((x + Pm.var(0.05) < 0).sum()), int((x + Pm.var(0.01) < 0).sum())
        exc95, exc99 = exc95 + e95, exc99 + e99
        pit.append(Pm.cdf(x))
        av = lambda p: np.median(np.quantile(g[:, keep], p, axis=0) / np.quantile(R[:, keep], p, axis=0))
        print(f"[{lo:.2f},{hi:.2f}] {days.size:5d} | {Pm.var(0.05) / Pr.var(0.05):6.3f} "
              f"{Pm.var(0.01) / Pr.var(0.01):6.3f} {Pm.es(0.025) / Pr.es(0.025):6.3f} | "
              f"{e95 / days.size:6.1%} {e99 / days.size:6.1%} | {av(0.05):6.3f} {av(0.01):6.3f} | "
              f"{rank_corr(R):8.3f} {rank_corr(g):6.3f}")
    pit = np.concatenate(pit)
    N = pit.size
    p95, p99 = pf_test(exc95, N, 0.05), pf_test(exc99, N, 0.01)
    print(f"\npooled over {N} days, each under its own bin's law:")
    print(f"  95%: {exc95} exceptions (expected {0.05 * N:.0f}), Kupiec p = {p95['p_value']:.4f}")
    print(f"  99%: {exc99} exceptions (expected {0.01 * N:.0f}), Kupiec p = {p99['p_value']:.4f}")
    print("  PIT deciles (share of days, 0.100 each if calibrated): "
          + " ".join(f"{c:.3f}" for c in np.histogram(pit, np.linspace(0, 1, 11))[0] / N))
    print("  (exceptions cluster in time, so these p-values are optimistic)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="conditional run of train_tailfm.py")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--num", type=int, default=10_000, help="generated windows per bin")
    ap.add_argument("--ode-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-recalibrate", action="store_true",
                    help="score the raw flow (does the u-mixture recalibration distort a regime?)")
    a = ap.parse_args()

    import torch                                       # only needed to generate
    from generate import conditional_recalibrator
    from tailfm import VelocityField, sample

    cfg = json.load(open(os.path.join(a.run, "model_config.json")))
    if not cfg.get("cond"):
        raise SystemExit(f"{a.run} is unconditional: no regime to calibrate")
    marg = pickle.load(open(os.path.join(a.run, "marginals.pkl"), "rb"))
    f, n = len(marg.marginals_), cfg["n"]
    model = VelocityField(f=f, n_max=n, d_model=cfg["d_model"], depth=cfg["depth"],
                          pos_std=cfg["pos_std"], cond=True)
    model.load_state_dict(torch.load(os.path.join(a.run, "model_ema.pt"), map_location="cpu"))
    seed = cfg.get("seed", 0) if a.seed is None else a.seed
    kw = dict(n_steps=a.ode_steps or cfg.get("ode_steps", 100), mix_dim=cfg["mix_dim"],
              device=a.device or ("cuda" if torch.cuda.is_available() else "cpu"), seed=seed)
    r = load_returns(a.data, False)
    u = np.load(os.path.join(a.run, "cond_u.npy"))

    log = "regime_calibration" + ("_raw" if a.no_recalibrate else "") + ".log"
    with tee_output(os.path.join(a.run, log), header="regime_calibration.py"):
        recal = (lambda z: z) if a.no_recalibrate else conditional_recalibrator(model, marg, a.num, n, f, **kw)

        def draw_day1(cond):
            z = sample(model, len(cond), n, f, nu=marg.nu_, cond=cond, **kw).numpy()
            return marg.inverse_transform(recal(z)[:, 0, :])

        calibrate(r, u, draw_day1, a.bins, a.num, seed=seed)


if __name__ == "__main__":
    main()
