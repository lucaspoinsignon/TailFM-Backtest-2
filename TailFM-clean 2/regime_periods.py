"""Which real days fall in a regime band?  Episodes and days per year; no model needed.

    python regime_periods.py --data data/returns_clean.csv --run runs/u090 --levels 0.8 0.9
    python regime_periods.py --data data/returns_clean.csv --run runs/u090 --cond-band 0.87 0.93
    python regime_periods.py ... --run runs/fold3 --levels 0.9 --train-only

Reads cond_u.npy and cond_sigma_ref.npy of a conditional run (train_tailfm.py), so u is
the run's own percentile scale.  Use it to choose the periods for --exclude-period /
--test-period, or to check how many days validate a level.
"""
import argparse
import json
import os

import numpy as np

from regime import band_around, load_dates, print_regime_periods
from splits import Split

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--run", required=True, help="conditional run: cond_u.npy, cond_sigma_ref.npy")
ap.add_argument("--levels", type=float, nargs="+", default=None)
ap.add_argument("--band-width", type=float, default=0.1, help="band around each level")
ap.add_argument("--cond-band", type=float, nargs=2, default=None, metavar=("LO", "HI"))
ap.add_argument("--train-only", action="store_true", help="only the run's training days")
ap.add_argument("--gap", type=int, default=5, help="rows between band days that split episodes")
ap.add_argument("--top", type=int, default=8, help="longest episodes listed")
a = ap.parse_args()

u = np.load(os.path.join(a.run, "cond_u.npy"))
ref = np.load(os.path.join(a.run, "cond_sigma_ref.npy"))
dates = load_dates(a.data, u.size)
if a.train_only:
    keep = Split.load(os.path.join(a.run, "split.json")).train_row_mask()
    u, dates = u[keep], dates[keep]
bands = ([tuple(a.cond_band)] if a.cond_band else []) + \
        [band_around(lev, a.band_width) for lev in a.levels or []]
if not bands:
    raise SystemExit("pass --levels and/or --cond-band")
cfg = os.path.join(a.run, "model_config.json")
valor = json.load(open(cfg)).get("cond_valor") if os.path.exists(cfg) else "?"
print(f"{a.run}: regime = EWMA vol of {valor}, {u.size} "
      f"{'training ' if a.train_only else ''}days\n")
for lo, hi in bands:
    print_regime_periods(dates, u, ref, lo, hi, gap=a.gap, top=a.top)
    print()
