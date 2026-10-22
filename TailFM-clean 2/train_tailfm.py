"""Train TailFM.  Training only: generate.py draws scenario windows from the saved run.

    python train_tailfm.py --data data/returns_clean.csv --outdir runs/uncond
    python train_tailfm.py --data data/returns_clean.csv --cond-valor 4155001 \\
        --train-all --outdir runs/u090

Input is a CSV (one column per feature, optional header, rows = time steps) or a .npy
array of shape (T, f), holding log returns unless --prices is passed.

Pipeline: split (splits.py) -> EVT marginals on the training rows with pooled xi -> PIT
to t_nu -> conditional flow matching on the training windows.

Training days (default: the purged random held-out blocks of splits.py):
    --train-all                every day
    --train-period A B ...     only these periods
    --exclude-period A B ...   every day except these periods (combinable with
                               --train-period).  Any of the three: no held-out blocks;
                               choose the test days later in evaluate.py / run_backtest.py.

Regime conditioning (--cond-valor V): the flow is conditioned on u = train percentile of
the EWMA vol of VALOR V on each window's first day.  Without --cond-valor the model is
unconditional.

Writes into --outdir:
    split.json          training / held-out rows
    model_ema.pt        EMA weights of the velocity field
    model_config.json   architecture + conditioning, read by generate.py
    marginals.pkl       EVT marginals (returns <-> t_nu latent space)
    cond_u.npy          regime u of every day          (conditional runs)
    cond_sigma_ref.npy  sorted training EWMA vols, u <-> sigma lookup  (conditional runs)
    report.log          everything printed
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
from regime import describe_ranges, load_dates, mask_to_ranges, period_mask, regime_u
from run_logging import tee_output
from splits import add_split_args, build_split
from tailfm import MarginalEnsemble, VelocityField, train_cfm


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
    ap.add_argument("--train-all", action="store_true",
                    help="no held-out blocks: train on every day")
    ap.add_argument("--train-period", nargs=2, action="append", metavar=("START", "END"),
                    help="train only on [START, END] (YYYY-MM-DD, inclusive); repeat "
                         "for a union.  No held-out blocks")
    ap.add_argument("--exclude-period", nargs=2, action="append", metavar=("START", "END"),
                    help="leave [START, END] out of training; repeat for several.  "
                         "No held-out blocks")
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
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--cond-valor", type=str, default=None,
                    help="VALOR (CSV column name) whose EWMA vol is the regime; turns "
                         "conditioning on.  Omit: unconditional")
    ap.add_argument("--ewma-lam", type=float, default=0.94)
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
    if args.train_all or args.train_period or args.exclude_period:
        split.test_starts = []                             # no held-out blocks
    if args.train_period or args.exclude_period:
        dates = load_dates(args.data, T)
        keep = (period_mask(dates, args.train_period) if args.train_period
                else np.ones(T, dtype=bool))
        if args.exclude_period:
            keep &= ~period_mask(dates, args.exclude_period)
        split.train_ranges = mask_to_ranges(keep)
        print(f"training days: {int(keep.sum())} of {T} "
              f"({describe_ranges(dates, split.train_ranges)})")
    split.save(f"{args.outdir}/split.json")
    real = split.train_windows(r, args.n, args.stride)
    train_r = split.train_rows(r)
    print(f"data: T={T}, f={f} | device={device}")
    print(split.summary())
    print(f"train windows {real.shape} | marginal rows {train_r.shape[0]}")

    # ---------------------------------------- optional regime condition u in [0, 1]
    # u = train percentile of the EWMA vol of one VALOR at each window's first day
    cond = None
    if args.cond_valor:
        if args.cond_valor not in names:
            raise SystemExit(f"--cond-valor {args.cond_valor} is not a column of {args.data}")
        u, ref = regime_u(r[:, names.index(args.cond_valor)], split.train_row_mask(),
                          args.ewma_lam)
        np.save(f"{args.outdir}/cond_sigma_ref.npy", ref)     # u <-> sigma lookup
        np.save(f"{args.outdir}/cond_u.npy", u)              # u of every day
        cond = torch.tensor(u[split.train_window_starts(args.n, args.stride)],
                            dtype=torch.float32)
        print(f"condition: EWMA vol of {args.cond_valor} (lambda={args.ewma_lam}); "
              f"sigma at u = 0.1 / 0.5 / 0.9: "
              + " / ".join(f"{np.quantile(ref, q):.4f}" for q in (0.1, 0.5, 0.9)))

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
    torch.manual_seed(args.seed)    # reproducible initialisation (as conditional runs had)
    model = VelocityField(f=f, n_max=args.n, d_model=args.d_model, depth=args.depth,
                          pos_std=args.pos_std, cond=cond is not None)
    ema, _ = train_cfm(model, z, nu=marg.nu_, steps=args.steps,
                       batch_size=args.batch, mix_dim=args.mix_dim,
                       device=device, seed=args.seed, cond=cond)
    torch.save(ema.shadow.state_dict(), f"{args.outdir}/model_ema.pt")
    json.dump(dict(n=args.n, d_model=args.d_model, depth=args.depth,
                   pos_std=args.pos_std, mix_dim=args.mix_dim, cond=cond is not None,
                   cond_valor=args.cond_valor, ewma_lam=args.ewma_lam, seed=args.seed),
              open(f"{args.outdir}/model_config.json", "w"), indent=2)
    pickle.dump(marg, open(f"{args.outdir}/marginals.pkl", "wb"))

    print(f"\nSaved: {args.outdir}/{{split.json, model_ema.pt, model_config.json, "
          f"marginals.pkl{', cond_u.npy, cond_sigma_ref.npy' if cond is not None else ''}}}")
    print(f"Next:  python generate.py --data {args.data} --run {args.outdir}"
          + (" --levels 0.9" if cond is not None else ""))


def main():
    args = parse_args()
    log_path = args.log or f"{args.outdir}/report.log"
    with tee_output(log_path, header="train_tailfm.py"):
        run(args)
    print(f"Terminal report saved to {log_path}")


if __name__ == "__main__":
    main()
