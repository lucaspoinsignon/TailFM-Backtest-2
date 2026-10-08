"""Historical-simulation (HS) reference for run_backtest.py.

Saves the REAL training windows of an existing run as if they were a generator's
output, with the same split.json, so run_backtest.py scores them on exactly the same
held-out blocks as the TailFM run.  HS is what a perfect unconditional generator would
produce, so:

    HS rejected like TailFM  -> the rejections come from the data / test design
    HS passes, TailFM fails  -> the generator is worse than its own training data

    python make_hs.py --data data/returns_clean.csv --run runs/final3 --out runs/hs
    python run_backtest.py --data data/returns_clean.csv \\
        --gen tailfm=runs/final3/generated_windows.npy --gen hs=runs/hs/generated_windows.npy
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np

from csvio import load_returns
from splits import Split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/returns_clean.csv")
    ap.add_argument("--run", default="runs/final3",
                    help="TailFM run directory holding split.json")
    ap.add_argument("--out", default="runs/hs")
    ap.add_argument("--prices", action="store_true")
    a = ap.parse_args()

    r = load_returns(a.data, a.prices)
    sp = Split.load(os.path.join(a.run, "split.json"))
    sp.check_data(os.path.abspath(a.data), r.shape[0])   # same file the run was trained on
    win = sp.train_windows(r, sp.n)                      # (N, n, f), purged training windows

    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "generated_windows.npy"), win)
    shutil.copy(os.path.join(a.run, "split.json"), os.path.join(a.out, "split.json"))
    print(f"saved {win.shape} real training windows to {a.out}/generated_windows.npy "
          f"(split {sp.id()})")


if __name__ == "__main__":
    main()
