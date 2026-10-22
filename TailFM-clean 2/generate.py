"""Generate scenario windows from a trained run (train_tailfm.py), at any regime level.

    python generate.py --data data/returns_clean.csv --run runs/u090 --levels 0.8 0.9
    python generate.py ... --levels 0.9 --mode band --band-width 0.06
    python generate.py ... --match --cond-band 0.87 0.93
    python generate.py ... --match --test-period 2020-01-01 2020-12-31 --cond-band 0.85 1.0
    python generate.py --data data/returns_clean.csv --run runs/uncond        # unconditional
    python generate.py ... --levels 0.70 0.72 0.74 0.76 --no-save             # table only

Conditional runs:
    --levels U ...   --mode point: every window at u = U   -> generated_windows_u0.90.npy
                     --mode band:  u uniform in [U - w/2, U + w/2], w = --band-width
                                                           -> generated_windows_u0.85-0.95.npy
    --match          u resampled from the real days chosen by --test-period /
                     --exclude-period / --cond-band, i.e. the exact counterpart of a
                     backtest on those days    -> generated_windows_match_<selection>.npy
Unconditional runs: one sample                             -> generated_windows.npy

Each conditional sample gets a .json sidecar recording which real days validate it (the
band around U, or the matched selection); evaluate.py and run_backtest.py use it by
default.  Files go into --run, next to split.json and cond_u.npy, where those scripts
look for them.  After generation a table compares the equal-weight 1-day P&L of each
sample with the real days of its band.

Defaults reproduce what train_tailfm.py used to write: the run's seed, --num 20000 and
the run's ODE steps give bit-identical samples.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import torch
from scipy import stats

from csvio import load_returns
from regime import (band_around, load_dates, period_mask, selection_tag, write_sidecar)
from tailfm import VelocityField, sample


def conditional_recalibrator(model, marg, num, n, f, **kw):
    """Continuous monotone per-feature map z -> phi_j(z) on the flow output, so that
    the u ~ U(0, 1) mixture of conditional samples has exactly t_nu marginals in
    z-space, i.e. the EVT marginals F_hat_j after the inverse PIT.  u is uniform on
    the training windows, so that mixture must have them: the map removes the flow's
    marginal distortion, like the unconditional rank-recalibration, while keeping
    the copula and the ordering across regimes.

    phi_j is piecewise linear through (mixture order statistic k, t_nu quantile
    k/(K+1)) and linear beyond the mixture's range, with the slope of its outer 1%:
    both tails are t-like, so the map is asymptotically a rescaling there.  The GPD
    tails of F_hat_j then extrapolate smoothly, with no ties and no cap."""
    u_mix = (np.arange(num) + 0.5) / num
    zs = np.sort(sample(model, num, n, f, nu=marg.nu_, cond=u_mix, **kw).numpy()
                 .reshape(-1, f), axis=0)
    K = zs.shape[0]
    tq = stats.t.ppf(np.arange(1, K + 1) / (K + 1.0), df=marg.nu_)
    m = max(K // 100, 1)
    k1, k2 = int(0.005 * K), int(0.995 * K)
    print(f"flow marginals before recalibration (u-mixture, z-space): median ratio of "
          f"the 0.5% / 99.5% quantile to t_nu = {np.median(zs[k1] / tq[k1]):.3f} / "
          f"{np.median(zs[k2] / tq[k2]):.3f}  (1 = exact)")

    def recal(z):
        flat = z.reshape(-1, f).astype(float)
        for j in range(f):
            zj, x = zs[:, j], flat[:, j]
            y = np.interp(x, zj, tq)
            lo, hi = x < zj[0], x > zj[-1]
            y[lo] = tq[0] + (x[lo] - zj[0]) * (tq[m] - tq[0]) / (zj[m] - zj[0])
            y[hi] = tq[-1] + (x[hi] - zj[-1]) * (tq[-1] - tq[-1 - m]) / (zj[-1] - zj[-1 - m])
            flat[:, j] = y
        return flat.reshape(z.shape)
    return recal


def generation_plan(level: float, mode: str, width: float, num: int, seed: int):
    """(cond for sample(), file suffix, sidecar) of one --levels value."""
    lo, hi = band_around(level, width)
    meta = dict(mode=mode, level=level, band=[lo, hi], test_period=None)
    if mode == "point":
        return level, f"_u{level:.2f}", meta
    u = lo + (hi - lo) * (np.arange(num) + 0.5) / num          # uniform in the band
    return np.random.default_rng(seed).permutation(u), f"_u{lo:.2f}-{hi:.2f}", meta


def rank_recalibrate(gen, marg):
    """x -> F_hat_j^{-1}(rank/(K+1)): each generated marginal becomes exactly F_hat_j and
    the learned copula is left invariant (Sklar).  Unconditional runs only."""
    num, n, f = gen.shape
    flat = gen.reshape(-1, f)
    K = flat.shape[0]
    rank = np.arange(1, K + 1, dtype=float) / (K + 1.0)
    for a in range(0, f, 32):                      # column blocks
        b = min(a + 32, f)
        blk = np.ascontiguousarray(flat[:, a:b].T)
        idx = np.argsort(blk, axis=1)
        u = np.empty(blk.shape)
        np.put_along_axis(u, idx, np.broadcast_to(rank, blk.shape), axis=1)
        for j in range(a, b):
            flat[:, j] = marg.marginals_[j].ppf(u[j - a])
    return flat.reshape(num, n, f)


def pnl_stats(x):
    """mean (bp), sd, VaR95, VaR99, ES97.5 (all in %) of an equal-weight 1-day P&L."""
    if x.size == 0:
        return "   (no real day in this band)"
    q = lambda p: -np.quantile(x, p) * 100
    es = -np.sort(x)[:max(int(round(0.025 * x.size)), 1)].mean() * 100
    return f"{x.mean() * 1e4:8.2f} {x.std() * 100:6.3f} {q(.05):7.3f} {q(.01):7.3f} {es:8.3f}"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the CSV the run was trained on")
    ap.add_argument("--run", required=True, help="output folder of train_tailfm.py")
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                    help="regime level(s) U in [0, 1] (conditional runs)")
    ap.add_argument("--mode", default="point", choices=["point", "band"])
    ap.add_argument("--band-width", type=float, default=None,
                    help="w: band of --mode band and of the real days that validate a "
                         "level (default: the run's, else 0.1)")
    ap.add_argument("--match", action="store_true",
                    help="u of the real days chosen by --test-period / --exclude-period "
                         "/ --cond-band instead of --levels")
    ap.add_argument("--test-period", nargs=2, action="append", metavar=("START", "END"))
    ap.add_argument("--exclude-period", nargs=2, action="append", metavar=("START", "END"))
    ap.add_argument("--cond-band", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--num", type=int, default=20_000, help="# generated windows per sample")
    ap.add_argument("--ode-steps", type=int, default=None, help="default: the run's, else 100")
    ap.add_argument("--seed", type=int, default=None, help="default: the run's training seed")
    ap.add_argument("--no-recalibrate", action="store_true",
                    help="skip the marginal recalibration (u-mixture / rank)")
    ap.add_argument("--no-save", action="store_true", help="print the table only")
    ap.add_argument("--device", type=str, default=None)
    # runs trained before model_config.json existed (unconditional only)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    return ap.parse_args()


def main():
    a = parse_args()
    cfg_path = os.path.join(a.run, "model_config.json")
    cfg = (json.load(open(cfg_path)) if os.path.exists(cfg_path)
           else dict(n=a.n, d_model=a.d_model, depth=a.depth, pos_std=0.1, mix_dim="window"))
    is_cond = cfg.get("cond", cfg.get("cond_valor") is not None)
    seed = cfg.get("seed", 0) if a.seed is None else a.seed
    width = a.band_width or cfg.get("band_width", 0.1)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    kw = dict(n_steps=a.ode_steps or cfg.get("ode_steps", 100), mix_dim=cfg["mix_dim"],
              device=device, seed=seed)

    marg = pickle.load(open(os.path.join(a.run, "marginals.pkl"), "rb"))
    f, n = len(marg.marginals_), cfg["n"]
    model = VelocityField(f=f, n_max=n, d_model=cfg["d_model"], depth=cfg["depth"],
                          pos_std=cfg["pos_std"], cond=is_cond)
    model.load_state_dict(torch.load(os.path.join(a.run, "model_ema.pt"), map_location="cpu"))
    r = load_returns(a.data, False)
    T = r.shape[0]
    if r.shape[1] != f:
        raise SystemExit(f"{a.data} has {r.shape[1]} columns, the run was trained on {f}")
    x_real = r.mean(axis=1)                                   # equal-weight 1-day P&L
    print(f"run {a.run}: f={f}, n={n}, {'conditional on ' + str(cfg.get('cond_valor')) if is_cond else 'unconditional'}"
          f" | num={a.num}, seed={seed}, ode_steps={kw['n_steps']}, device={device}")

    # ------------------------------------------------- what to generate
    plans = []                                                # (cond, suffix, sidecar, real days)
    if not is_cond:
        if a.levels or a.match:
            raise SystemExit(f"{a.run} is unconditional: no --levels / --match")
        plans.append((None, "", None, np.arange(T)))
    else:
        u = np.load(os.path.join(a.run, "cond_u.npy"))
        fits = np.arange(T) <= T - n                          # a whole window fits
        if a.match:
            if not (a.test_period or a.exclude_period or a.cond_band):
                raise SystemExit("--match needs --test-period, --exclude-period and/or --cond-band")
            sel = fits.copy()
            if a.test_period or a.exclude_period:
                dates = load_dates(a.data, T)
                if a.test_period:
                    sel &= period_mask(dates, a.test_period)
                if a.exclude_period:
                    sel &= ~period_mask(dates, a.exclude_period)
            if a.cond_band:
                sel &= (u >= a.cond_band[0]) & (u <= a.cond_band[1])
            days = np.flatnonzero(sel)
            if days.size == 0:
                raise SystemExit("no day matches the selection")
            periods = [list(p) for p in a.test_period] if a.test_period else None
            excluded = [list(p) for p in a.exclude_period] if a.exclude_period else None
            band = list(a.cond_band) if a.cond_band else None
            plans.append((u[np.random.default_rng(seed).choice(days, a.num)],
                          "_match_" + selection_tag(band, periods, excluded),
                          dict(mode="match", band=band, test_period=periods,
                               exclude_period=excluded), days))
        else:
            if not a.levels:
                raise SystemExit("conditional run: pass --levels U ... or --match")
            for lev in a.levels:
                c, suffix, meta = generation_plan(lev, a.mode, width, a.num, seed)
                lo, hi = meta["band"]
                plans.append((c, suffix, meta, np.flatnonzero((u >= lo) & (u <= hi) & fits)))

    # ------------------------------------------------------- generate
    recal = (conditional_recalibrator(model, marg, a.num, n, f, **kw)
             if is_cond and not a.no_recalibrate else None)
    print(f"\n{'sample':>34s} {'N':>6s} {'mean_bp':>8s} {'sd%':>6s} {'VaR95%':>7s} "
          f"{'VaR99%':>7s} {'ES97.5%':>8s}   (equal-weight 1-day P&L)")
    for c, suffix, meta, days in plans:
        z = sample(model, a.num, n, f, nu=marg.nu_, cond=c, **kw).numpy()
        gen = marg.inverse_transform(z if recal is None else recal(z))
        del z
        if not is_cond and not a.no_recalibrate:
            gen = rank_recalibrate(gen, marg)
        name = f"generated_windows{suffix}.npy"
        print(f"{'real days ' + (describe_band(meta) if meta else '(all)'):>34s} "
              f"{days.size:6d} {pnl_stats(x_real[days])}")
        print(f"{name:>34s} {a.num:6d} {pnl_stats(gen[:, 0, :].mean(axis=1))}")
        if not a.no_save:
            path = os.path.join(a.run, name)
            np.save(path, gen)
            if meta:
                write_sidecar(path, dict(meta, cond_valor=cfg.get("cond_valor"), num=a.num,
                                         seed=seed, ode_steps=kw["n_steps"],
                                         recalibrated=recal is not None))
    if not a.no_save:
        print(f"\nsaved into {a.run}/ (with a .json sidecar per conditional sample)")


def describe_band(meta):
    band = meta.get("band")
    return f"u in [{band[0]:.2f}, {band[1]:.2f}]" if band else "matched"


if __name__ == "__main__":
    main()
