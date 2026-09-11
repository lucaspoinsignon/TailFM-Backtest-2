"""Train the baselines (TimeVAE, TimeGAN, Tail-GAN) and score them with the same code
as tailfm, so the numbers are directly comparable.

    python run_baselines.py --data returns.csv --n 24 --gen 50000
    python run_baselines.py --data returns.csv --quick            # CPU smoke test
    python run_baselines.py --data returns.csv --n 24 \\
        --split runs/final/split.json --tailfm-gen runs/final/generated_windows.npy

Per baseline: the purged split of splits.py (pass --split whenever --tailfm-gen is
passed, or the models are fitted to different windows) -> each reference
implementation's own scaling, applied inside the baseline -> training -> sampling ->
print_report, estimate_risk and a Kupiec backtest -> comparison table and figures.
Everything printed also goes to {outdir}/report.log.

Unlike train_tailfm.py no rank-recalibration is applied: the baselines are evaluated
exactly as generated, tailfm's EVT marginals being part of the model.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csvio import load_returns, feature_names_from_csv
from figures import save_all_figures
from run_logging import tee_output
from splits import add_split_args, resolve_split, sidecar_path, Split
from backtest.risk import estimate_risk, portfolio_losses
from backtest.kupiec import pf_test
from diagnostics import print_report
from baselines import BASELINES


def parse_args():
    ap = argparse.ArgumentParser()
    # ---- data: keep identical to train_tailfm.py -----------------------------
    ap.add_argument("--data", required=True, help="CSV or .npy of shape (T, f)")
    ap.add_argument("--prices", action="store_true",
                    help="input is prices, not returns")
    ap.add_argument("--n", type=int, default=24, help="window length")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--gen", type=int, default=50_000, help="# generated windows")
    ap.add_argument("--horizon", type=int, default=10)
    add_split_args(ap)
    ap.add_argument("--split", type=str, default=None,
                    help="split.json written by train_tailfm.py.  Pass it whenever "
                         "--tailfm-gen is passed: the baselines must be trained on "
                         "the same windows tailfm was, or the comparison is between "
                         "models fitted to different data")
    ap.add_argument("--weights", type=str, default=None,
                    help="comma-separated portfolio weights (default: equal)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="baseline_out")
    ap.add_argument("--log", type=str, default=None,
                    help="text file receiving a copy of everything printed "
                         "(default: {outdir}/report.log)")
    ap.add_argument("--pair-select", default="spread",
                    choices=["worst", "random", "spread"])
    # ---- model selection ----------------------------------------------------
    ap.add_argument("--models", type=str, default="timevae,timegan,tailgan",
                    help="comma-separated subset of: timevae,timegan,tailgan")
    ap.add_argument("--tailfm-gen", type=str, default=None,
                    help="path to generated_windows.npy from a train_tailfm.py "
                         "run with the same data settings, to include tailfm "
                         "in the comparison table")
    ap.add_argument("--quick", action="store_true",
                    help="tiny training budgets for a CPU smoke test")
    ap.add_argument("--reuse", action="store_true",
                    help="for each requested model, load {outdir}/gen_<model>.npy "
                         "if it exists instead of retraining (recover from a "
                         "crashed evaluation, or re-run the evaluation and "
                         "figures without paying for training again)")
    # ---- per-model budgets (reference defaults; see baselines/*.py) ----------
    ap.add_argument("--timevae-epochs", type=int, default=1000)
    ap.add_argument("--timevae-batch", type=int, default=16)
    ap.add_argument("--timevae-recon-wt", type=float, default=3.0,
                    help="reference default 3.0; raise (e.g. 100) to counteract "
                         "posterior collapse on near-i.i.d. return windows")
    ap.add_argument("--timevae-latent", type=int, default=8)
    ap.add_argument("--timegan-iters", type=int, default=10_000,
                    help="iterations PER PHASE (reference default 50000)")
    ap.add_argument("--timegan-batch", type=int, default=128)
    ap.add_argument("--tailgan-epochs", type=int, default=3000)
    ap.add_argument("--tailgan-batch", type=int, default=1000)
    ap.add_argument("--tailgan-lr-g", type=float, default=1e-6)
    ap.add_argument("--tailgan-lr-d", type=float, default=1e-7)
    ap.add_argument("--tailgan-alphas", type=str, default="0.05",
                    help="comma-separated PnL tail levels the score targets "
                         "(reference default 0.05). To align training with the "
                         "evaluation levels use 0.05,0.01,0.005; note the "
                         "alpha=0.005 tail has only ~batch_size*0.005 order "
                         "statistics per batch, so keep the batch large.")
    args = ap.parse_args()

    if args.quick:
        args.timevae_epochs = 30
        args.timegan_iters = 100
        args.tailgan_epochs = 30
        args.tailgan_batch = 128
        args.gen = min(args.gen, 2048)

    # validate all paths before spending compute on training
    if args.tailfm_gen and not os.path.exists(args.tailfm_gen):
        ap.error(f"--tailfm-gen file not found: {args.tailfm_gen}\n"
                 "(checked up front so no training time is wasted; run "
                 "train_tailfm.py first or fix the path)")
    if not os.path.exists(args.data):
        ap.error(f"--data file not found: {args.data}")
    return args


def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    alphas = (0.95, 0.99, 0.995)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in BASELINES]
    assert not unknown, f"unknown model(s) {unknown}; choose from {list(BASELINES)}"

    # --------------------------------------------------------------- data
    r = load_returns(args.data, args.prices)
    T, f = r.shape
    names = feature_names_from_csv(args.data, f)
    split = resolve_split(args.split, args, T, data=os.path.abspath(args.data))

    # write split.json beside gen_<model>.npy as train_tailfm.py does; evaluate.py
    # and run_backtest.py read it and refuse to mix samples from different splits
    if args.tailfm_gen:
        tf = sidecar_path(args.tailfm_gen)
        if not os.path.exists(tf):
            raise SystemExit(
                f"--tailfm-gen has no split.json beside it ({tf}).  The baselines "
                f"must be trained on\nthe same windows tailfm was, or the "
                f"comparison table is between models fitted\nto different data.  "
                f"Rerun train_tailfm.py, which writes split.json.")
        ref = Split.load(tf)
        if ref.id() != split.id():
            raise SystemExit(
                f"split mismatch: --tailfm-gen was produced under split "
                f"{ref.id()}, this run\nuses {split.id()}.  Pass --split {tf} so "
                f"the baselines see the same training\nwindows tailfm saw.")
        print(f"split matches --tailfm-gen (id {split.id()})")
    split.save(f"{args.outdir}/split.json")
    real = split.train_windows(r, args.n, args.stride)
    print(f"data: T={T}, f={f} ({', '.join(names)})")
    print(split.summary())
    print(f"train windows {real.shape}")

    w = (np.array([float(v) for v in args.weights.split(",")])
         if args.weights else np.full(f, 1.0 / f))
    assert w.size == f, "--weights length must equal the number of features"
    L_test = portfolio_losses(split.test_windows(r, args.horizon),
                              weights=w, horizon=args.horizon)

    hparams = {
        "timevae": dict(max_epochs=args.timevae_epochs,
                        batch_size=args.timevae_batch,
                        reconstruction_wt=args.timevae_recon_wt,
                        latent_dim=args.timevae_latent),
        "timegan": dict(iterations=args.timegan_iters,
                        batch_size=args.timegan_batch),
        "tailgan": dict(n_epochs=args.tailgan_epochs,
                        batch_size=args.tailgan_batch,
                        lr_G=args.tailgan_lr_g, lr_D=args.tailgan_lr_d,
                        alphas=tuple(float(a) for a in
                                     args.tailgan_alphas.split(","))),
    }

    # ------------------------------------------------ train, generate, score
    gens: dict[str, np.ndarray] = {}
    for name in models:
        cache = f"{args.outdir}/gen_{name}.npy"
        if args.reuse and os.path.exists(cache):
            gen = np.load(cache)
            assert gen.shape[1:] == (args.n, f), \
                f"cached {cache} has window shape {gen.shape[1:]}, expected " \
                f"({args.n}, {f}); delete it or drop --reuse to retrain"
            print(f"\n################ {name} (reusing {cache}, "
                  f"{gen.shape[0]} windows) ################")
        else:
            print(f"\n################ {name} ################")
            gen = BASELINES[name](real, args.gen, seed=args.seed,
                                  device=args.device, **hparams[name])
            np.save(cache, gen)
        gens[name] = gen

    if args.tailfm_gen:
        gens["tailfm"] = np.load(args.tailfm_gen)
        assert gens["tailfm"].shape[1:] == (args.n, f), \
            "--tailfm-gen windows have incompatible shape; rerun train_tailfm.py " \
            "with the same --n and --data"

    summary: dict[str, dict] = {}
    for name, gen in gens.items():
        print(f"\n=== Diagnostics: {name} (vs real train windows) ===")
        print_report(real, gen, feature_names=names)
        report = estimate_risk(gen, alphas=alphas, weights=w,
                               horizon=args.horizon, n_boot=200, seed=args.seed)
        summary[name] = {}
        print(f"\n=== {name}: portfolio risk (h={args.horizon}) and Kupiec "
              f"backtest (held-out N={L_test.size}) ===")
        for a in alphas:
            rp = report[a]
            # run_backtest.py adds TUFF, the ES tests and the power column
            x = int((L_test > rp["var_gpd"]).sum())
            k = pf_test(x, L_test.size, 1.0 - a)
            summary[name][a] = (rp["var_gpd"], rp["cvar_gpd"],
                                k["exceedances"], k["expected"], k["p_value"])
            print(f"a={a:5.3f}: VaR {rp['var_gpd']:.5f} "
                  f"[{rp['var_ci'][0]:.5f},{rp['var_ci'][1]:.5f}]  "
                  f"CVaR {rp['cvar_gpd']:.5f} "
                  f"[{rp['cvar_ci'][0]:.5f},{rp['cvar_ci'][1]:.5f}]  | "
                  f"exceed {k['exceedances']}/{k['expected']:.1f}  "
                  f"p={k['p_value']:.3f}")

    # ------------------------------------------------------ comparison table
    print("\n" + "=" * 78)
    print("MODEL COMPARISON -- portfolio VaR/CVaR (GPD-refined) and Kupiec "
          "p-value on held-out data")
    print("=" * 78)
    header = f"{'model':>10s}" + "".join(
        f" | a={a:.3f}: VaR    CVaR    exc    p " for a in alphas)
    print(header)
    for name, per_a in summary.items():
        row = f"{name:>10s}"
        for a in alphas:
            v, c, exc, expd, p = per_a[a]
            row += f" | {v:7.4f} {c:7.4f} {exc:3d}/{expd:4.1f} {p:5.3f}"
        print(row)
    print("(Kupiec: p > 0.05 means the VaR level is not rejected; exc/exp = "
          "observed vs expected exceedances)")

    # ----------------------------------------------------------------- figures
    paths = save_all_figures(real, gens, names, args.outdir, weights=w,
                             horizon=args.horizon, pair_select=args.pair_select,
                             seed=args.seed)
    print("\nSaved: " + f"{args.outdir}/{{split.json, gen_<model>.npy}}, "
          + ", ".join(os.path.basename(p) for p in paths))


def main():
    args = parse_args()
    log_path = args.log or f"{args.outdir}/report.log"
    with tee_output(log_path, header="run_baselines.py"):
        run(args)
    print(f"Terminal report saved to {log_path}")


if __name__ == "__main__":
    main()
