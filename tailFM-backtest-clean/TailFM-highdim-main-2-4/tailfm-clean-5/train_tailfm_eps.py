"""train_tailfm.py on STANDARDISED log returns: one change, everything else identical.

    python train_tailfm_eps.py --data data/returns_clean.csv --outdir runs/eps_none \
        --mix-dim none   [same --steps / --d-model / ... as your reference run]

Before anything else, every valor is divided by its own EWMA volatility,

    sigma^2_{t,j} = lam sigma^2_{t-1,j} + (1 - lam) r^2_{t-1,j}     (returns before t only)
    eps_{t,j}     = r_{t,j} / sigma_{t,j}

with lam = 0.94 (fixed, not fitted), sigma^2_0 = variance of the valor on the training
rows, and a floor at 5% of that standard deviation.  The rest of the pipeline (split,
EVT marginals, PIT, flow matching, sampling, rank recalibration) runs unchanged on eps,
so generated_windows.npy holds standardised returns.  vol_filter.json records lam and
the start variances so eval_eps.py can put the volatility back.

--mix-dim adds 'none' (one chi^2 per coordinate: independent t_nu base coordinates).
The velocity field cannot undo a random scale shared by a whole window, and real eps
windows have an almost constant scale, so 'window' is expected to over-disperse eps.
Run once with --mix-dim window (strictly one change) and once with --mix-dim none.
tailfm/ is not modified: 'none' is provided by a local sample_base passed to cfm.

Original docstring follows.
----------------------------------------------------------------------------------
Stage 1 of 3: fit tail-aware flow matching and generate scenario windows.

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
import torch

from csvio import load_returns, feature_names_from_csv
from evt_shrink import shrink_ensemble
from run_logging import tee_output
from splits import add_split_args, build_split
from tailfm import MarginalEnsemble, VelocityField, train_cfm, sample
import tailfm.cfm as _cfm


def sample_base_any(batch, n, f, nu, mix_dim="window", device="cpu", generator=None):
    """tailfm.base.sample_base plus mix_dim='none' (W drawn per coordinate)."""
    z = torch.randn(batch, n, f, device=device, generator=generator)
    w_shape = {"window": (batch, 1, 1), "time": (batch, n, 1), "none": (batch, n, f)}[mix_dim]
    w = torch.distributions.Gamma(nu / 2.0, 0.5).sample(w_shape).to(device)
    return z * torch.sqrt(torch.as_tensor(nu, device=device) / w.clamp_min(1e-8))


_cfm.sample_base = sample_base_any        # used by train_cfm and sample


def ewma_sigma(r, lam, init_var, floor=0.05):
    """sigma_t from returns up to t-1 only; (T, f) -> (T, f)."""
    s2 = np.empty_like(r)
    s2[0] = init_var
    lo = floor ** 2 * init_var
    for t in range(1, r.shape[0]):
        s2[t] = np.maximum(lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2, lo)
    return np.sqrt(s2)


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
    ap.add_argument("--mix-dim", default="window", choices=["window", "time", "none"],
                    help="coordinates sharing the base mixing variable W; 'none' = "
                         "one W per coordinate")
    ap.add_argument("--lam", type=float, default=0.94, help="EWMA decay")
    ap.add_argument("--vol-floor", type=float, default=0.05,
                    help="sigma floor as a fraction of the training-row sd")
    ap.add_argument("--gen", type=int, default=20_000, help="# generated windows")
    ap.add_argument("--ode-steps", type=int, default=100)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no-recalibrate", action="store_true",
                    help="disable rank-recalibration of generated marginals")
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
    split.save(f"{args.outdir}/split.json")

    # ------------------------------------------------- standardise by EWMA vol
    init_var = split.train_rows(r).var(axis=0)
    init_var = np.where(init_var > 0, init_var, 1e-12)
    sigma = ewma_sigma(r, args.lam, init_var, args.vol_floor)
    raw_train = split.train_rows(r)
    r = r / sigma
    with open(f"{args.outdir}/vol_filter.json", "w") as fh:
        json.dump(dict(lam=args.lam, floor=args.vol_floor,
                       init_var=init_var.tolist()), fh)
    eps_train = split.train_rows(r)
    kurt = lambda x: np.median(((x - x.mean(0)) ** 4).mean(0) / x.var(0) ** 2)
    acf2 = lambda x: np.median([np.corrcoef(c[1:] ** 2, c[:-1] ** 2)[0, 1] for c in x.T])
    print(f"standardised by EWMA(lam={args.lam}): median kurtosis {kurt(raw_train):.1f} -> "
          f"{kurt(eps_train):.1f}, median lag-1 ACF of squares {acf2(raw_train):+.3f} -> "
          f"{acf2(eps_train):+.3f} | base mix_dim={args.mix_dim}")

    real = split.train_windows(r, args.n, args.stride)
    train_r = split.train_rows(r)
    print(f"data: T={T}, f={f} | device={device}")
    print(split.summary())
    print(f"train windows {real.shape} | marginal rows {train_r.shape[0]}")

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
    model = VelocityField(f=f, n_max=args.n, d_model=args.d_model, depth=args.depth,
                          pos_std=args.pos_std)
    ema, _ = train_cfm(model, z, nu=marg.nu_, steps=args.steps,
                       batch_size=args.batch, mix_dim=args.mix_dim,
                       device=device, seed=args.seed)
    torch.save(ema.shadow.state_dict(), f"{args.outdir}/model_ema.pt")
    pickle.dump(marg, open(f"{args.outdir}/marginals.pkl", "wb"))

    # --------------------------------------------------------------- sampling
    z_gen = sample(ema.shadow, args.gen, args.n, f, nu=marg.nu_,
                   n_steps=args.ode_steps, mix_dim=args.mix_dim,
                   device=device, seed=args.seed)
    gen = marg.inverse_transform(z_gen.numpy())
    del z_gen
    if not args.no_recalibrate:
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
    np.save(f"{args.outdir}/generated_windows.npy", gen)

    print(f"\nSaved: {args.outdir}/{{split.json, generated_windows.npy, "
          f"model_ema.pt, marginals.pkl}}")
    print(f"Next:  python eval_eps.py --data {args.data} --eps-run {args.outdir}")


def main():
    args = parse_args()
    log_path = args.log or f"{args.outdir}/report.log"
    with tee_output(log_path, header="train_tailfm.py"):
        run(args)
    print(f"Terminal report saved to {log_path}")


if __name__ == "__main__":
    main()
