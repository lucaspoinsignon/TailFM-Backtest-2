"""Stage 3 of 3: out-of-sample coverage tests on the purged held-out blocks.

    python run_backtest.py --data data/returns_clean.csv \\
        --gen runs/final/generated_windows.npy
    python run_backtest.py --self-test      # reproduce the papers' published tables

Kupiec (1995) TUFF and PF, and Acerbi-Szekely (2014) Z1, Z2, Z3, comparing the
realised h-step portfolio P&L on the held-out blocks of `split.json` with the
predictive law implied by the generated windows.  The only stage that touches the
held-out rows.

Held-out h-step returns number about test_frac * T / h, so h = 10 leaves ~24
observations and h = 1 ~240.  At 24 observations Z1 is undefined unless an exception
occurs, Z3 needs T >= 1/alpha and is undefined outright, and PF has almost no power,
so both horizons run by default and every test prints its power.

Independence is assumed, not tested, by both papers and by this script: within a test
block the h-step returns are adjacent in calendar time.  Train with --test-block equal
to --horizon for one observation per block if that matters more than sample size.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from csvio import load_returns
from run_logging import tee_output
from splits import add_split_args, add_split_source_args, require_split, Split
from backtest import (predictive_from_windows, realized_pnl, pf_test, tuff_test,
                      first_failure, pf_nonrejection_region, pf_power,
                      test_z1, test_z2, test_z3, reproduce_exhibits,
                      Z2_TRAFFIC_LIGHT)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV or .npy of shape (T, f)")
    ap.add_argument("--gen", action="append", metavar="[LABEL=]PATH",
                    help="generated_windows.npy; repeat to backtest several models")
    ap.add_argument("--prices", action="store_true", help="input is prices, not returns")
    ap.add_argument("--n", type=int, default=24, help="window length")
    ap.add_argument("--horizon", type=str, default=None,
                    help="comma-separated risk horizons in steps "
                         "(default: '1,<split horizon>')")
    add_split_args(ap)
    add_split_source_args(ap)
    ap.add_argument("--var-levels", type=str, default="0.95,0.99",
                    help="VaR CONFIDENCE levels for the Kupiec tests; the tail "
                         "probability tested is p* = 1 - level")
    ap.add_argument("--es-levels", type=str, default="0.975,0.99",
                    help="ES CONFIDENCE levels for the Acerbi-Szekely tests; the "
                         "tail probability is alpha = 1 - level.  Basel uses "
                         "ES at 97.5%% against VaR at 99%%")
    ap.add_argument("--weights", type=str, default=None,
                    help="comma-separated portfolio weights (default: equal)")
    ap.add_argument("--n-sim", type=int, default=10_000,
                    help="Monte Carlo scenarios for the Acerbi-Szekely p-values")
    ap.add_argument("--gpd-tail", action="store_true",
                    help="refine the predictive loss tail with a GPD fit above its "
                         "90%% quantile instead of using the raw generated sample")
    ap.add_argument("--power-alt", type=str, default="2,3",
                    help="power of the PF test is reported against true rates that "
                         "are these multiples of p*")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default=None,
                    help="where the log goes (default: dir of the first --gen)")
    ap.add_argument("--log", type=str, default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="reproduce Kupiec's Exhibits 1/4/5/6 and the "
                         "Acerbi-Szekely size/power table, then exit")
    args = ap.parse_args()
    if not args.self_test:
        if not args.data:
            ap.error("--data is required (or use --self-test)")
        if not args.gen:
            ap.error("--gen is required (or use --self-test)")
    return args


def parse_gens(specs: list[str]) -> dict:
    gens = {}
    for spec in specs:
        label, _, path = spec.rpartition("=")
        if not os.path.exists(path):
            raise SystemExit(f"--gen file not found: {path}")
        if not label:
            stem = os.path.splitext(os.path.basename(path))[0]
            label = (os.path.basename(os.path.dirname(os.path.abspath(path)))
                     if stem == "generated_windows" else stem.removeprefix("gen_"))
        gens[label] = np.load(path)
    return gens


# --------------------------------------------------------------------- report
def design_block(n_obs: int, var_levels, es_levels, power_mults) -> None:
    """What this sample can and cannot detect, before any p-value is shown."""
    print(f"\n--- test design (N = {n_obs} held-out observations) ---")
    print(f"  {'level':>7s} {'p*':>7s} {'E[exc]':>7s} {'accept x in':>13s}  "
          + "  ".join(f"power@{m:g}p*" for m in power_mults))
    for lv in var_levels:
        p = 1.0 - lv
        lo, hi = pf_nonrejection_region(n_obs, p)
        pw = "  ".join(f"{pf_power(n_obs, p, min(m * p, 0.999)):9.2f}"
                       for m in power_mults)
        print(f"  {lv:7.3f} {p:7.4f} {n_obs * p:7.2f} {f'[{lo}, {hi}]':>13s}  {pw}")
    for lv in es_levels:
        a = 1.0 - lv
        m = int(n_obs * a)
        notes = []
        if n_obs * a < 1:
            notes.append("Z1 needs >=1 exception")
        if m < 1:
            notes.append(f"Z3 undefined ([Na]=0, needs N>={int(np.ceil(1 / a))})")
        print(f"  ES level {lv:.3f} (alpha={a:.4f}): E[exc] {n_obs * a:.2f}, "
              f"[N*alpha] = {m}" + ("  <-- " + "; ".join(notes) if notes else ""))


def backtest_one(label, x, pred, var_levels, es_levels, args) -> dict:
    n = x.size
    print(f"\n{'=' * 74}\n{label}: {pred}\n{'=' * 74}")

    print("\n--- Kupiec (1995) VaR verification ---")
    row = {}
    for lv in var_levels:
        p_star = 1.0 - lv
        var = pred.var(p_star)
        exceed = (x + var) < 0.0
        n_exc = int(exceed.sum())
        pf = pf_test(n_exc, n, p_star)
        v = first_failure(exceed)
        tu = tuff_test(v, p_star, n=n)
        lo, hi = pf_nonrejection_region(n, p_star)
        print(f"  level {lv:.3f} (p*={p_star:.4f})  VaR = {var:.5f}")
        print(f"    PF   : x = {n_exc} of {n} (expected {pf['expected']:.2f}, "
              f"rate {pf['observed_rate']:.4f}) | accept [{lo}, {hi}] | "
              f"LR = {pf['LR']:.3f}  p = {pf['p_value']:.4f} "
              f"(one-sided upper p = {pf['p_value_upper']:.4f})")
        if tu["censored"]:
            print(f"    TUFF : no failure in {n} observations -- censored, "
                  f"P(no failure | H0) = {tu['p_no_failure']:.4f}")
        else:
            print(f"    TUFF : V = {tu['V']} | LR = {tu['LR']:.3f}  "
                  f"p = {tu['p_value']:.4f}")
        row[("VaR", lv)] = (var, n_exc, pf["expected"], pf["p_value"])

    print("\n--- Acerbi-Szekely (2014) Expected Shortfall backtests ---")
    for lv in es_levels:
        a = 1.0 - lv
        print(f"  level {lv:.3f} (alpha={a:.4f})  VaR = {pred.var(a):.5f}  "
              f"ES = {pred.es(a):.5f}")
        r1 = test_z1(x, pred, a, n_sim=args.n_sim, seed=args.seed)
        r2 = test_z2(x, pred, a, n_sim=args.n_sim, seed=args.seed)
        r3 = test_z3(x, pred, a, n_sim=args.n_sim, seed=args.seed)
        for r in (r1, r2, r3):
            print("  " + r.line())
        if r2.defined:
            print(f"    Z2 traffic light: {r2.extra['traffic_light']} "
                  f"(yellow < {Z2_TRAFFIC_LIGHT['yellow']}, "
                  f"red < {Z2_TRAFFIC_LIGHT['red']}); simulated 5% level "
                  f"{r2.extra['sim_q05']:+.3f}")
        row[("ES", lv)] = (r1, r2, r3)
    return row


def run(args):
    gens = parse_gens(args.gen)
    r = load_returns(args.data, args.prices)
    T, f = r.shape

    # --horizon is a list here, but building a split (only reachable under
    # --allow-missing-split) takes one; the first requested horizon sets the block
    requested = [int(h) for h in args.horizon.split(",")] if args.horizon else None
    args.horizon = requested[0] if requested else 10
    if args.split:
        split = Split.load(args.split)
        split.check_data(os.path.abspath(args.data), T)
        print(f"split: {args.split} (id {split.id()}, forced by --split)")
    else:
        split = require_split(args.gen, args, T, data=os.path.abspath(args.data),
                              allow_missing=args.allow_missing_split)
    print(split.summary())

    horizons = requested if requested else sorted({1, split.horizon})
    var_levels = [float(v) for v in args.var_levels.split(",")]
    es_levels = [float(v) for v in args.es_levels.split(",")]
    power_mults = [float(m) for m in args.power_alt.split(",")]

    w = (np.array([float(v) for v in args.weights.split(",")])
         if args.weights else np.full(f, 1.0 / f))
    assert w.size == f, "--weights length must equal the number of features"

    for h in horizons:
        if h > split.block:
            print(f"\n### horizon h={h} skipped: longer than the test block "
                  f"({split.block})")
            continue
        tw = split.test_windows(r, h)
        x = realized_pnl(tw, w, h)
        print(f"\n\n{'#' * 74}\n### HORIZON h = {h}: {x.size} held-out "
              f"non-overlapping {h}-step returns\n{'#' * 74}")
        design_block(x.size, var_levels, es_levels, power_mults)
        for label, g in gens.items():
            if g.shape[1] < h:
                print(f"\n{label}: generated windows are {g.shape[1]} steps long, "
                      f"cannot form a {h}-step return -- skipped")
                continue
            pred = predictive_from_windows(g, w, h, gpd_tail=args.gpd_tail)
            backtest_one(label, x, pred, var_levels, es_levels, args)


def self_test() -> None:
    from scipy import stats
    from backtest.predictive import PredictivePnL

    print(reproduce_exhibits())
    print("\n\nACERBI-SZEKELY -- size and power against the paper's Table 2")
    print("H0: Student-t(100), T = 250, alpha = 2.5%, 400 replications\n")
    alpha, T, reps, n_sim = 0.025, 250, 400, 2000
    P = PredictivePnL(stats.t.rvs(100, size=200_000, random_state=1))
    print(f"  predictive VaR_1% {P.var(0.01):.2f} ES_2.5% {P.es(0.025):.2f} "
          f"(paper Table 2: 2.36 / 2.37)")
    paper = {(100, 0.041): None, (10, 0.041): (40.9, 54.8), (10, 0.104): (57.7, 67.7),
             (3, 0.041): (99.3, 99.8), (3, 0.104): (99.6, 99.9)}
    print(f"\n  {'H1':>5s} {'level':>7s} {'Z2':>7s} {'Z3':>7s}   paper (Z2, Z3)")
    for nu in (100, 10, 3):
        rng = np.random.default_rng(7)
        p2, p3 = [], []
        for i in range(reps):
            x = stats.t.rvs(nu, size=T, random_state=int(rng.integers(1 << 31)))
            p2.append(test_z2(x, P, alpha, n_sim=n_sim, seed=i).p_value)
            p3.append(test_z3(x, P, alpha, n_sim=n_sim, seed=i).p_value)
        p2, p3 = np.array(p2), np.array(p3)
        for lv in (0.041, 0.104):
            ref = paper.get((nu, lv))
            tag = f"({ref[0]}, {ref[1]})" if ref else "(size, should ~ level)"
            print(f"  {nu:5d} {lv:7.3f} {100 * np.mean(p2 < lv):7.1f} "
                  f"{100 * np.mean(p3 < lv):7.1f}   {tag}")
    print("\n(Monte Carlo standard error at 400 replications is ~2.5 points.)")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    outdir = args.outdir or os.path.dirname(
        os.path.abspath(args.gen[0].rpartition("=")[2]))
    log_path = args.log or os.path.join(outdir, "backtest.log")
    with tee_output(log_path, header="run_backtest.py"):
        run(args)
    print(f"Terminal report saved to {log_path}")


if __name__ == "__main__":
    main()
