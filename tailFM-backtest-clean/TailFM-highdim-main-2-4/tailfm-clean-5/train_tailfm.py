"""Stage 1 of 3: fit tail-aware flow matching and generate scenario windows.

    python train_tailfm.py --data data/returns_clean.csv --outdir runs/final

Input is a CSV (one column per feature, optional header, rows = time steps) or a .npy
array of shape (T, f), holding log returns unless --prices is passed.

Pipeline: purged random-block split (splits.py) -> EVT marginals on the training rows
with pooled xi -> PIT to t_nu -> CFM training -> sampling -> inverse PIT -> rank
recalibration.  Writes split.json, generated_windows.npy, model_ema.pt, marginals.pkl
and report.log into --outdir.

Nothing is scored here: diagnostics live in evaluate.py and coverage tests in
run_backtest.py, so the held-out rows are never touched at this stage.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy import stats

from csvio import load_returns, feature_names_from_csv
from evt_shrink import shrink_ensemble
from run_logging import tee_output
from splits import add_split_args, build_split
from tailfm import MarginalEnsemble, VelocityField, train_cfm, sample


def ewma_sigma(r, lam, init_var, floor_var):
    """RiskMetrics EWMA vol; sigma_t uses returns up to t-1, so it is known at t."""
    s2 = np.empty_like(r)
    s2[0] = init_var
    for t in range(1, r.shape[0]):
        s2[t] = np.maximum(lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2, floor_var)
    return np.sqrt(s2)


def conditional_recalibrator(model, marg, num, n, f, pool=None, **kw):
    """Continuous monotone per-feature map z -> phi_j(z) on the flow output, so that
    the mixture of conditional samples over the training conditions has exactly t_nu
    marginals in z-space, i.e. the EVT marginals F_hat_j after the inverse PIT.  That
    mixture must have them: the map removes the flow's marginal distortion, like the
    unconditional rank-recalibration, while keeping the copula and the ordering
    across regimes.  One condition: u ~ U(0, 1), its training distribution.  Several
    (`pool`, the (N, k) training condition vectors): resampled from `pool`.

    phi_j is piecewise linear through (mixture order statistic k, t_nu quantile
    k/(K+1)) and linear beyond the mixture's range, with the slope of its outer 1%:
    both tails are t-like, so the map is asymptotically a rescaling there.  The GPD
    tails of F_hat_j then extrapolate smoothly, with no ties and no cap."""
    u_mix = ((np.arange(num) + 0.5) / num if pool is None
             else pool[np.random.default_rng(0).integers(0, len(pool), num)])
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


def print_regime_periods(dates, u, ref, lev, half=0.05, gap=5, top=8):
    """The training days whose regime u is within `half` of `lev`, grouped into
    episodes (band days at most `gap` rows apart are merged)."""
    lo, hi = max(lev - half, 0.0), min(lev + half, 1.0)
    idx = np.flatnonzero((u >= lo) & (u <= hi))
    print(f"regime u = {lev:.2f}: band [{lo:.2f}, {hi:.2f}] = EWMA vol "
          f"{np.quantile(ref, lo) * 100:.2f}%-{np.quantile(ref, hi) * 100:.2f}%, "
          f"{idx.size} training days", end="")
    if idx.size == 0:
        print()
        return
    eps = np.split(idx, np.flatnonzero(np.diff(idx) > gap) + 1)
    print(f" in {len(eps)} episodes; the {min(top, len(eps))} longest:")
    for e in sorted(sorted(eps, key=len, reverse=True)[:top], key=lambda e: e[0]):
        print(f"    {str(dates[e[0]])[:10]} -> {str(dates[e[-1]])[:10]}  "
              f"({e.size} days in band)")
    yrs, cnt = np.unique(dates[idx].astype("datetime64[Y]").astype(int) + 1970,
                         return_counts=True)
    print("    days per year: " + ", ".join(f"{y}: {c}" for y, c in zip(yrs, cnt)))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV or .npy of shape (T, f)")
    ap.add_argument("--prices", action="store_true", help="input is prices, not returns")
    ap.add_argument("--n", type=int, default=24, help="window length")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=10,
                    help="risk horizon the held-out blocks are cut for; recorded in "
                         "split.json and reused by evaluate.py / run_backtest.py")
    add_split_args(ap)
    ap.add_argument("--train-period", nargs=2, action="append", metavar=("START", "END"),
                    help="restrict training to [START, END] (YYYY-MM-DD); repeat for a union")
    ap.add_argument("--test-period", nargs=2, action="append", metavar=("START", "END"),
                    help="test on [START, END] (YYYY-MM-DD); repeat for a union")
    ap.add_argument("--train-all", action="store_true",
                    help="no held-out blocks: train on every day (the backtest is then "
                         "in-sample, see run_backtest.py --cond-band)")
    ap.add_argument("--q-tail", type=float, default=0.05,
                    help="EVT threshold quantile, per tail")
    ap.add_argument("--nu", type=float, default=5.0,
                    help="degrees of freedom of the t_nu latent space.  The PIT makes "
                         "every marginal exactly t_nu for ANY nu, so this is a design "
                         "parameter rather than an estimate: nu <= 4 gives the CFM "
                         "target x1-x0 an infinite fourth moment and hence "
                         "infinite-variance gradients.  5 is the smallest value with "
                         "E z^4 < inf, and the generated copula is insensitive to it.")
    ap.add_argument("--pos-std", type=float, default=0.1,
                    help="init scale of the positional embedding; controls how much "
                         "within-window volatility clustering the model produces")
    ap.add_argument("--shrink-c", type=float, default=1.0,
                    help="Efron-Morris cap on the xi pooling, in standard errors")
    ap.add_argument("--no-shrink", action="store_true",
                    help="disable pooling of xi across features")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--mix-dim", default="window", choices=["window", "time"],
                    help="coordinates sharing the base mixing variable W")
    ap.add_argument("--gen", type=int, default=20_000, help="# generated windows")
    ap.add_argument("--ode-steps", type=int, default=100)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no-recalibrate", action="store_true",
                    help="disable rank-recalibration of generated marginals")
    ap.add_argument("--gen-cond", type=float, nargs="+", default=None,
                    help="regime level(s) u in [0, 1] = train percentile of the EWMA "
                         "vol of --cond-valor; turns conditioning on and writes one "
                         "generated_windows_u<u>.npy per level.  Omit: unconditional")
    ap.add_argument("--cond-valor", type=str, default=None,
                    help="VALOR (CSV column name) whose EWMA vol is the regime, or EW "
                         "for the equal-weight portfolio of all features")
    ap.add_argument("--ewma-lam", type=float, default=0.94)
    ap.add_argument("--cond-momentum", type=int, default=0, metavar="L",
                    help="add a second condition: the train percentile of the vol "
                         "momentum log(sigma_t / sigma_{t-L}).  --gen-cond levels then "
                         "refer to the first condition, the second is resampled from "
                         "the training days within 0.05 of each level")
    ap.add_argument("--oversample", type=float, nargs=2, default=None,
                    metavar=("Q", "W"),
                    help="draw the training windows with u >= Q W times as often")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="runs/final")
    ap.add_argument("--log", type=str, default=None,
                    help="text file receiving a copy of everything printed "
                         "(default: {outdir}/report.log)")
    return ap.parse_args()


def run(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------- data
    r = load_returns(args.data, args.prices)
    T, f = r.shape
    names = feature_names_from_csv(args.data, f)
    split = build_split(args, T, data=os.path.abspath(args.data))
    if args.train_period or args.test_period:
        dates = pd.to_datetime(pd.read_csv(args.data, usecols=[0]).iloc[-T:, 0])
        rows = lambda a, b: [int(dates.searchsorted(pd.Timestamp(a))),
                             int(dates.searchsorted(pd.Timestamp(b), side="right"))]
        if args.train_period:
            split.train_ranges = [rows(a, b) for a, b in args.train_period]
        if args.test_period:
            split.test_starts = [s for a, b in args.test_period for c, d in [rows(a, b)]
                                 for s in range(c, d - split.block + 1, split.block)]
            split.sampling = "dates"
    if args.train_all:
        split.test_starts = []
    split.save(f"{args.outdir}/split.json")
    real = split.train_windows(r, args.n, args.stride)
    train_r = split.train_rows(r)
    print(f"data: T={T}, f={f} | device={device}")
    print(split.summary())
    print(f"train windows {real.shape} | marginal rows {train_r.shape[0]}")

    # ---------------------------------------- optional regime condition u in [0, 1]
    # u = train percentile of the EWMA vol of one VALOR at each window's first day
    cond = None
    if args.gen_cond:
        if not args.cond_valor:
            raise SystemExit("--gen-cond needs --cond-valor")
        rj = (r.mean(axis=1) if args.cond_valor == "EW"
              else r[:, names.index(args.cond_valor)])
        tr = split.train_row_mask()
        sig = ewma_sigma(rj, args.ewma_lam, init_var=rj[tr].var(), floor_var=1e-12)
        ref = np.sort(sig[tr])
        np.save(f"{args.outdir}/cond_sigma_ref.npy", ref)     # u <-> sigma lookup
        u = np.searchsorted(ref, sig, side="right") / ref.size
        np.save(f"{args.outdir}/cond_u.npy", u)              # u of every day, for the backtest
        cx = u[:, None]
        if args.cond_momentum:                               # vol rising or falling
            L = args.cond_momentum
            mom = np.log(sig / np.concatenate([np.full(L, sig[0]), sig[:-L]]))
            mref = np.sort(mom[tr])
            cx = np.column_stack([u, np.searchsorted(mref, mom, side="right") / mref.size])
        np.save(f"{args.outdir}/cond_x.npy", cx)             # all conditions, every day
        starts = split.train_window_starts(args.n, args.stride)
        cond = torch.tensor(cx[starts] if cx.shape[1] > 1 else u[starts],
                            dtype=torch.float32)
        weights = None
        if args.oversample:
            q, w = args.oversample
            weights = torch.tensor(np.where(u[starts] >= q, w, 1.0), dtype=torch.float32)
            print(f"oversampling: the {int((u[starts] >= q).sum())} training windows "
                  f"with u >= {q} are drawn {w:g}x as often")
        print(f"condition: EWMA vol of {args.cond_valor} (lambda={args.ewma_lam}); "
              f"sigma at u = 0.1 / 0.5 / 0.9: "
              + " / ".join(f"{np.quantile(ref, q):.4f}" for q in (0.1, 0.5, 0.9)))
        if not args.data.endswith(".npy"):                   # dates are in the CSV
            dates = pd.to_datetime(pd.read_csv(args.data, usecols=[0])
                                   .iloc[-T:, 0]).to_numpy()
            for lev in args.gen_cond:
                print_regime_periods(dates[tr], u[tr], ref, lev)

    # -------------------------------------------------- EVT marginals + PIT
    # fitted on the training ROWS, not on `real`: windows overlap at stride 1, so
    # every interior date would be counted ~n times in the exceedance counts
    marg = MarginalEnsemble(q_tail=args.q_tail, nu=args.nu).fit(train_r)
    if not args.no_shrink:
        print("EVT: pooling xi across features (empirical Bayes)")
        shrink_ensemble(marg, train_r, c=args.shrink_c)
    su = marg.summary()
    print(f"EVT: nu={marg.nu_:.2f}, q_tail={args.q_tail}, "
          f"xi_lower median {np.median(su['xi_lo']):+.3f} "
          f"[{su['xi_lo'].min():+.3f}, {su['xi_lo'].max():+.3f}], "
          f"xi_upper median {np.median(su['xi_hi']):+.3f} "
          f"[{su['xi_hi'].min():+.3f}, {su['xi_hi'].max():+.3f}]")
    z = torch.tensor(marg.transform(real), dtype=torch.float32)
    print(f"PIT: max|z| = {np.abs(z.numpy()).max():.1f} (correct fits stay under ~100)")

    # --------------------------------------------------------------- training
    k = 0 if cond is None else (1 if cond.ndim == 1 else cond.shape[1])
    if k:                       # conditional runs: reproducible initialisation
        torch.manual_seed(args.seed)
    model = VelocityField(f=f, n_max=args.n, d_model=args.d_model, depth=args.depth,
                          pos_std=args.pos_std, cond=k)
    ema, _ = train_cfm(model, z, nu=marg.nu_, steps=args.steps,
                       batch_size=args.batch, mix_dim=args.mix_dim,
                       device=device, seed=args.seed, cond=cond,
                       weights=weights if k else None)
    torch.save(ema.shadow.state_dict(), f"{args.outdir}/model_ema.pt")
    if k:                       # what cond_check.py needs to rebuild the model
        json.dump(dict(n=args.n, d_model=args.d_model, depth=args.depth,
                       pos_std=args.pos_std, cond=k, mix_dim=args.mix_dim,
                       ode_steps=args.ode_steps, cond_valor=args.cond_valor,
                       ewma_lam=args.ewma_lam, cond_momentum=args.cond_momentum,
                       oversample=args.oversample, seed=args.seed),
                  open(f"{args.outdir}/model_config.json", "w"), indent=2)
    pickle.dump(marg, open(f"{args.outdir}/marginals.pkl", "wb"))

    # --------------------------------------------------------------- sampling
    # conditional runs recalibrate through the u-mixture (conditional_recalibrator):
    # the plain rank-recalibration would force every level to the unconditional F_hat_j
    pool = cx[starts] if k > 1 else None
    recal = (conditional_recalibrator(ema.shadow, marg, args.gen, args.n, f, pool=pool,
                                      n_steps=args.ode_steps, mix_dim=args.mix_dim,
                                      device=device, seed=args.seed)
             if cond is not None and not args.no_recalibrate else None)
    for lev in (args.gen_cond or [None]):
        c_lev = lev                 # several conditions: resample them near the level
        if pool is not None:
            near = np.flatnonzero(np.abs(pool[:, 0] - lev) <= 0.05)
            c_lev = pool[np.random.default_rng(args.seed).choice(near, args.gen)]
        z_gen = sample(ema.shadow, args.gen, args.n, f, nu=marg.nu_,
                       n_steps=args.ode_steps, mix_dim=args.mix_dim,
                       device=device, seed=args.seed, cond=c_lev).numpy()
        gen = marg.inverse_transform(z_gen if recal is None else recal(z_gen))
        del z_gen
        if not args.no_recalibrate and lev is None:
            # rank-recalibration: x -> F_hat_j^{-1}(rank/(K+1)) makes each generated
            # marginal exactly F_hat_j and leaves the learned copula invariant (Sklar)
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
            gen = flat.reshape(args.gen, args.n, f)
        np.save(f"{args.outdir}/generated_windows"
                f"{'' if lev is None else f'_u{lev:.2f}'}.npy", gen)

    print(f"\nSaved: {args.outdir}/{{split.json, generated_windows.npy, "
          f"model_ema.pt, marginals.pkl}}")
    print(f"Next:  python evaluate.py    --data {args.data} --gen {args.outdir}"
          f"/generated_windows.npy")
    print(f"       python run_backtest.py --data {args.data} --gen {args.outdir}"
          f"/generated_windows.npy")


def main():
    args = parse_args()
    log_path = args.log or f"{args.outdir}/report.log"
    with tee_output(log_path, header="train_tailfm.py"):
        run(args)
    print(f"Terminal report saved to {log_path}")


if __name__ == "__main__":
    main()
