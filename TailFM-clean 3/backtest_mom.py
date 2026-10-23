"""Backtest several runs on the same real days, split by vol direction.

    python momentum/backtest_mom.py --data data/returns_clean.csv \\
        --run level=runs/ho090 --run momentum=runs/ho090_mom \\
        --holdout --cond-level 0.9 --direction all rising falling

For each direction, each run's sample from generate_mom.py for the same selection
(generated_windows_mom_<selection>_<direction>.npy) is scored on those days: equal-weight
1-day P&L, Kupiec PF / TUFF at 95 / 99% and Acerbi-Szekely Z1 / Z2 / Z3 at 97.5 / 99%,
exactly as run_backtest.py.  A summary table closes the report.  The days (u, split,
direction) come from the first --run, so give runs trained with the same hold-out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                    # noqa: E402

from backtest import predictive_from_windows                         # noqa: E402
from csvio import load_returns, feature_names_from_csv               # noqa: E402
from mom_common import add_day_args, mom_tag, momentum, select_days, selection  # noqa: E402
from regime import load_dates                                         # noqa: E402
from run_backtest import backtest_one, design_block                   # noqa: E402
from run_logging import tee_output                                    # noqa: E402
from splits import Split                                              # noqa: E402


def run(a):
    runs = [s.split("=", 1) if "=" in s else [os.path.basename(s.rstrip("/")), s]
            for s in a.run]
    first = runs[0][1]
    cfg = json.load(open(os.path.join(first, "model_config.json")))
    r = load_returns(a.data, False)
    T, f = r.shape
    names = feature_names_from_csv(a.data, f)
    split = Split.load(os.path.join(first, "split.json"))
    tr = split.train_row_mask()
    u = np.load(os.path.join(first, "cond_u.npy"))
    lag = a.lag or cfg.get("mom_lag", 10)
    m, _ = momentum(r[:, names.index(cfg["cond_valor"])], tr, cfg["ewma_lam"], lag)
    band, periods, excluded = selection(a)
    dates = load_dates(a.data, T) if periods or excluded else None
    w = np.full(f, 1.0 / f)
    var_levels = [float(x) for x in a.var_levels.split(",")]
    es_levels = [float(x) for x in a.es_levels.split(",")]
    print(f"days from {first} (u, split); momentum lag {lag}; runs: "
          + ", ".join(f"{k}={d}" for k, d in runs))

    summary = []
    for d in a.direction:
        days = select_days(u, m, tr, dates, band, a.holdout, periods, excluded, d)
        tag = mom_tag(band, a.holdout, periods, excluded, d)
        print(f"\n\n{'#' * 74}\n### {tag}: {days.size} days, "
              f"{tr[days].mean() if days.size else 0:.0%} of them training days\n{'#' * 74}")
        if days.size == 0:
            continue
        x = r[days] @ w
        design_block(x.size, var_levels, es_levels, [2.0, 3.0])
        for label, folder in runs:
            path = os.path.join(folder, f"generated_windows_mom_{tag}.npy")
            if not os.path.exists(path):
                print(f"\n{label}: {path} missing -- run momentum/generate_mom.py on "
                      f"{folder} with the same selection")
                continue
            pred = predictive_from_windows(np.load(path), w, 1)
            row = backtest_one(f"{label} [{d}]", x, pred, var_levels, es_levels, a)
            summary.append((d, label, x.size, row))

    print(f"\n\n{'=' * 96}\nSUMMARY (exceptions: observed / expected, p-value)\n{'=' * 96}")
    print(f"{'direction':>10s} {'run':>12s} {'N':>5s} | " + " | ".join(
        f"VaR{100 * lv:g}: exc  p   " for lv in var_levels) + " | " + " | ".join(
        f"Z2 {100 * lv:g}: p " for lv in es_levels))
    for d, label, n_obs, row in summary:
        cells = [f"{row[('VaR', lv)][1]:3d}/{row[('VaR', lv)][2]:5.2f} {row[('VaR', lv)][3]:.3f}"
                 for lv in var_levels]
        zs = [f"{row[('ES', lv)][1].p_value:9.3f}" for lv in es_levels]
        print(f"{d:>10s} {label:>12s} {n_obs:5d} | " + " | ".join(cells) + " | "
              + " | ".join(zs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                    help="run folder with generate_mom.py samples; repeat to compare")
    add_day_args(ap)
    ap.add_argument("--var-levels", default="0.95,0.99")
    ap.add_argument("--es-levels", default="0.975,0.99")
    ap.add_argument("--n-sim", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    first = a.run[0].split("=", 1)[-1]
    log = a.log or os.path.join(first, "backtest_mom.log")
    with tee_output(log, header="momentum/backtest_mom.py"):
        run(a)
    print(f"Terminal report saved to {log}")


if __name__ == "__main__":
    main()
