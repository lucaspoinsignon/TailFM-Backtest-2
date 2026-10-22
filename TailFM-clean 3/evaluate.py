"""Score generated windows against the real ones.

    python evaluate.py --data data/returns_clean.csv --gen runs/final/generated_windows.npy
    python evaluate.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy
    python evaluate.py ... --gen runs/u090/generated_windows_u0.90.npy --no-band
    python evaluate.py ... --gen runs/u090/generated_windows_u0.90.npy --cond-band 0.87 0.93

Horizon: everything compares the first --horizon days of each window (default 1, the
day the regime u describes and the day the backtest scores): the report, the figures,
the VaR/CVaR table and the portfolio loss survival use h-day returns / losses.
--horizon 24 compares whole windows.  Only the novelty check keeps whole windows.

Reference = the real h-day periods the sample is compared with:
    default            the training days; for a conditional sample only those with u in
                       the sample's band (its .json sidecar)
    --no-band          every training day, whatever the regime (entire dataset for a
                       --train-all run)
    --cond-level U     days with u in [U - 0.03, U + 0.03]
    --cond-band LO HI  days with u in [LO, HI]
    --holdout          days the run was not trained on (band hold-out,
                       --exclude-period, held-out blocks)
    --test-period A B  days inside these dates, trained on or not (repeatable)
    --exclude-period A B   drop the days in these dates (repeatable)
The flags combine; for h > 1 a period is the h days starting on the selected day.

Six sections, each with a --no-* flag:

    print_report      Hill tail index, lambda_L(0.02) on the worst pairs, per-feature
                      VaR/CVaR, squared-return ACF
    estimate_risk     h-step portfolio VaR/CVaR with bootstrap percentile CIs
    marginal PIT      per-feature KS / Anderson-Darling on the held-out rows
    novelty           nearest-neighbour distance to the training windows
    density           pooled marginal KDE, real vs generated
    figures           QQ, tail dependence, per-feature densities, portfolio survival

By default the reference is the TRAINING days, the target distribution of a scenario
generator; --holdout / --test-period score days the run did not train on.  Coverage
tests are run_backtest.py.  The marginal PIT section always uses the held-out rows of
the split, and it tests F_hat_j alone, never the generated windows.

The training windows are the purged set in `split.json`, by default the one beside
--gen.  Compare several generators by repeating --gen, optionally with a label:

    python evaluate.py --data returns.csv \\
        --gen tailfm=runs/final/generated_windows.npy \\
        --gen timevae=runs/baselines/gen_timevae.npy --outdir runs/compare
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from csvio import load_returns, feature_names_from_csv
from diagnostics import marginal_pit_report, novelty_report, print_report
from figures import save_all_figures
from regime import (add_selection_args, describe_selection, load_dates, read_sidecar,
                    real_window_starts, resolve_selection, selection_tag)
from run_logging import tee_output
from splits import add_split_args, add_split_source_args, require_split, Split
from backtest.risk import estimate_risk, portfolio_losses, var_cvar_empirical


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV or .npy of shape (T, f)")
    ap.add_argument("--gen", action="append", required=True, metavar="[LABEL=]PATH",
                    help="generated_windows.npy; repeat to overlay several models")
    ap.add_argument("--prices", action="store_true", help="input is prices, not returns")
    ap.add_argument("--n", type=int, default=24, help="window length")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=1,
                    help="compare the first h days of each window: report, figures, "
                         "VaR/CVaR table, portfolio loss (default 1)")
    add_split_args(ap)
    add_split_source_args(ap)
    ap.add_argument("--alphas", type=str, default="0.95,0.99,0.995",
                    help="confidence levels for the VaR/CVaR table")
    ap.add_argument("--weights", type=str, default=None,
                    help="comma-separated portfolio weights (default: equal)")
    ap.add_argument("--n-boot", type=int, default=200,
                    help="bootstrap replicates for the VaR/CVaR CIs")
    ap.add_argument("--max-pairs", type=int, default=30,
                    help="feature pairs shown in the tail-dependence table/figure")
    ap.add_argument("--pair-select", default="worst",
                    choices=["worst", "random", "spread"],
                    help="'worst' ranks pairs by |lambda_gen - lambda_real| maxed "
                         "over models, so one degenerate model picks the panels for "
                         "all of them; use 'spread' when comparing several")
    ap.add_argument("--marginals", type=str, default=None,
                    help="marginals.pkl from training, for the PIT goodness-of-fit "
                         "section (default: next to the first --gen).  Scoring the "
                         "ensemble the generator was actually fitted with; a refit "
                         "would score a model nothing downstream used")
    ap.add_argument("--n-novelty", type=int, default=3000,
                    help="generated windows scored for novelty; the distance matrix "
                         "is O(n_novelty * n_train * n * f)")
    ap.add_argument("--density-scale", default="none",
                    choices=["none", "std", "minmax"],
                    help="per-feature rescaling for the pooled density figure, "
                         "fitted on the real training rows")
    ap.add_argument("--top", type=int, default=20,
                    help="features listed in the marginal PIT table")
    ap.add_argument("--no-figures", action="store_true", help="skip the PNGs")
    ap.add_argument("--no-report", action="store_true", help="skip print_report")
    ap.add_argument("--no-risk", action="store_true", help="skip the VaR/CVaR table")
    ap.add_argument("--no-marginals", action="store_true",
                    help="skip the PIT goodness-of-fit section")
    ap.add_argument("--no-novelty", action="store_true",
                    help="skip the nearest-neighbour novelty table")
    ap.add_argument("--no-density", action="store_true",
                    help="skip the pooled marginal density figure")
    # reference days; default: the samples' sidecar .json (how they were generated),
    # unconditional samples: all training days
    add_selection_args(ap, "real days")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default=None,
                    help="where the PNGs and the log go (default: dir of the "
                         "first --gen)")
    ap.add_argument("--log", type=str, default=None)
    return ap.parse_args()


def parse_gens(specs: list[str]) -> dict:
    """['a=x.npy', 'y.npy'] -> {'a': array, 'y': array}, insertion-ordered."""
    gens = {}
    for spec in specs:
        label, _, path = spec.rpartition("=")
        if not path:
            raise SystemExit(f"--gen {spec!r}: empty path")
        if not os.path.exists(path):
            raise SystemExit(f"--gen file not found: {path}")
        if not label:
            # runs/final/generated_windows.npy -> 'final'; gen_timevae.npy -> 'timevae'
            stem = os.path.splitext(os.path.basename(path))[0]
            label = (os.path.basename(os.path.dirname(os.path.abspath(path)))
                     if stem == "generated_windows" else stem.removeprefix("gen_"))
        arr = np.load(path)
        if arr.ndim != 3:
            raise SystemExit(f"{path}: expected (M, n, f), got shape {arr.shape}")
        gens[label] = arr
    return gens


def run(args):
    alphas = tuple(float(a) for a in args.alphas.split(","))
    gens = parse_gens(args.gen)

    # ------------------------------------------------------------------- data
    r = load_returns(args.data, args.prices)
    T, f = r.shape
    names = feature_names_from_csv(args.data, f)

    if args.split:
        split = Split.load(args.split)
        split.check_data(os.path.abspath(args.data), T)
        print(f"split: {args.split} (id {split.id()}, forced by --split)")
    else:
        split = require_split(args.gen, args, T, data=os.path.abspath(args.data),
                              allow_missing=args.allow_missing_split)
    print(split.summary())

    n = split.n or args.n
    real = split.train_windows(r, n, args.stride)
    print(f"data: T={T}, f={f} | real (train) windows {real.shape}")
    for label, g in gens.items():
        if g.shape[1:] != real.shape[1:]:
            raise SystemExit(
                f"--gen {label}: windows are {g.shape[1:]}, real are "
                f"{real.shape[1:]}.  Rerun with matching --n, or pass the "
                f"split.json the generator was trained with.")
        print(f"  gen[{label}]: {g.shape}")

    # the first h days of the generated windows against the matching real h-day periods;
    # novelty keeps every whole training window, since a generated window could copy any
    h = args.horizon
    if not 1 <= h <= n:
        raise SystemExit(f"--horizon must be in [1, {n}]")
    real_all, (periods, excluded, band, holdout) = real, args.selection
    u = (np.load(os.path.join(os.path.dirname(os.path.abspath(
        args.gen[0].rpartition("=")[2])), "cond_u.npy")) if band else None)
    dates = load_dates(args.data, T) if periods or excluded else None
    starts = real_window_starts(split, h, args.stride, dates, periods, excluded,
                                band, u, holdout)
    if starts.size == 0:
        raise SystemExit("no real day matches the selection")
    real = np.stack([r[s:s + h] for s in starts])
    gens_full, gens = gens, {k: g[:, :h] for k, g in gens.items()}
    selected = bool(band or periods or excluded or holdout)
    share = split.train_row_mask()[starts].mean()
    print(f"horizon h={h}: first {h} day(s) of each generated window vs {real.shape[0]} "
          f"real {h}-day periods, {describe_selection(periods, excluded, band, holdout)} "
          f"({share:.0%} on training days; novelty: all {real_all.shape[0]} whole "
          f"training windows)")

    w = (np.array([float(v) for v in args.weights.split(",")])
         if args.weights else np.full(f, 1.0 / f))
    assert w.size == f, "--weights length must equal the number of features"

    # ------------------------------------------------------------ diagnostics
    if not args.no_report:
        for label, g in gens.items():
            print(f"\n{'#' * 20} diagnostics: {label} vs real, h={h} {'#' * 20}")
            print_report(real, g, feature_names=names, max_pairs=args.max_pairs)

    # ------------------------------------------------------------ risk report
    # a summary of the generated law, not a validation of it: the CIs cover sampling
    # error in the M generated windows only.  Coverage testing is run_backtest.py.
    if not args.no_risk:
        L_real = portfolio_losses(real, weights=w, horizon=h)
        print(f"\n=== Portfolio VaR/CVaR of the h={h} loss (generated law vs the "
              f"{L_real.size} real {h}-day losses of the reference) ===")
        for label, g in gens.items():
            rep = estimate_risk(g, alphas=alphas, weights=w, horizon=h,
                                n_boot=args.n_boot, seed=args.seed)
            for a in alphas:
                rp = rep[a]
                vr, cr = var_cvar_empirical(L_real, a)
                print(f"  {label:>10s} a={a:5.3f}:  VaR {rp['var_gpd']:.5f} "
                      f"[{rp['var_ci'][0]:.5f},{rp['var_ci'][1]:.5f}]  "
                      f"CVaR {rp['cvar_gpd']:.5f} "
                      f"[{rp['cvar_ci'][0]:.5f},{rp['cvar_ci'][1]:.5f}]"
                      f"  |  real VaR {vr:.5f} CVaR {cr:.5f}")

    outdir = args.outdir or os.path.dirname(
        os.path.abspath(args.gen[0].rpartition("=")[2]))
    train_rows, test_rows = split.train_rows(r), r[split.test_row_mask()]

    # ----------------------------------------------------------- marginal PIT
    # the only section reading held-out rows, and a statement about the MARGINALS
    # rather than the generator: it asks whether F_hat_j was right
    if not args.no_marginals and test_rows.shape[0] == 0:
        print("\nno held-out rows in this split (--train-all / --train-period / "
              "--exclude-period run): marginal PIT section skipped")
    elif not args.no_marginals:
        mpath = args.marginals or os.path.join(
            os.path.dirname(os.path.abspath(args.gen[0].rpartition("=")[2])),
            "marginals.pkl")
        if os.path.exists(mpath):
            import pickle
            marg = pickle.load(open(mpath, "rb"))
            print(f"\nmarginals: {mpath} (nu={marg.nu_:.3f}, q_tail={marg.q_tail})")
            marginal_pit_report(marg, train_rows, test_rows, names, outdir,
                                top=args.top, figure=not args.no_figures)
        else:
            print(f"\nmarginals.pkl not found at {mpath} -- skipping the PIT "
                  f"goodness-of-fit section (pass --marginals, or --no-marginals)")

    # --------------------------------------------------------------- novelty
    if not args.no_novelty:
        novelty_report(real_all, gens_full, train_rows, n_gen=args.n_novelty,
                       seed=args.seed)

    # ---------------------------------------------------------------- figures
    paths = ([] if args.no_figures else
             save_all_figures(real, gens, names, outdir, weights=w, horizon=h,
                              prefix=(f"h{h}_" + (selection_tag(band, periods, excluded,
                                                                  holdout) + "_"
                                                    if selected else "")),
                              max_pairs=args.max_pairs,
                              pair_select=args.pair_select, seed=args.seed,
                              real_rows=None,
                              density=not args.no_density,
                              density_scale=args.density_scale))
    if paths:
        print(f"\nSaved: {outdir}/" + ", ".join(os.path.basename(p) for p in paths))
    return outdir


def main():
    args = parse_args()
    outdir = args.outdir or os.path.dirname(
        os.path.abspath(args.gen[0].rpartition("=")[2]))
    args.selection = resolve_selection(
        args, [read_sidecar(s.rpartition("=")[2]) or {} for s in args.gen])
    periods, excluded, band, holdout = args.selection
    tag = selection_tag(band, periods, excluded, holdout)
    log_path = args.log or os.path.join(
        outdir, f"evaluate_h{args.horizon}" + (f"_{tag}" if tag else "") + ".log")
    with tee_output(log_path, header="evaluate.py"):
        run(args)
    print(f"Terminal report saved to {log_path}")


if __name__ == "__main__":
    main()
