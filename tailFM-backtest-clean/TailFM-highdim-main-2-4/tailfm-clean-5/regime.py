"""Regime conditioning and test-day selection, shared by train_tailfm.py, cond_check.py,
run_backtest.py, evaluate.py and compare_band.py.

    u_t      train percentile of the EWMA vol of one VALOR on day t.  sigma_t uses
             returns up to t-1, so u_t is known at t.  Saved as cond_u.npy by
             train_tailfm.py for every day, trained on or not.
    sidecar  <generated>.json beside a conditional sample: how it was generated and
             therefore which real days validate it by default (`band`, `test_period`).
"""

from __future__ import annotations

import json
import os
import re

import numpy as np


def ewma_sigma(r, lam, init_var, floor_var):
    """RiskMetrics EWMA vol; sigma_t uses returns up to t-1, so it is known at t."""
    s2 = np.empty_like(r)
    s2[0] = init_var
    for t in range(1, r.shape[0]):
        s2[t] = np.maximum(lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2, floor_var)
    return np.sqrt(s2)


def load_dates(path: str, T: int) -> np.ndarray:
    """Dates of the T return rows (first CSV column), as datetime64."""
    if path.endswith(".npy"):
        raise SystemExit(f"{path}: dates are needed for periods; pass the CSV")
    import pandas as pd                     # only scripts using periods need pandas
    return pd.to_datetime(pd.read_csv(path, usecols=[0]).iloc[-T:, 0]).to_numpy()


def period_mask(dates: np.ndarray, periods) -> np.ndarray:
    """Rows inside any [START, END] of `periods` (YYYY-MM-DD, both inclusive)."""
    m = np.zeros(dates.size, dtype=bool)
    for a, b in periods:
        m |= (dates >= np.datetime64(a)) & (dates <= np.datetime64(b))
    return m


def mask_to_ranges(m: np.ndarray) -> list:
    """[[start, end), ...] of the runs of True in `m`."""
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def describe_ranges(dates, ranges, top: int = 6) -> str:
    s = ", ".join(f"{str(dates[a])[:10]} -> {str(dates[b - 1])[:10]}" for a, b in ranges[:top])
    return s + (f", ... ({len(ranges)} periods)" if len(ranges) > top else "")


def block_starts(m: np.ndarray, h: int) -> np.ndarray:
    """Starts of non-overlapping h-day blocks tiling each run of True in `m`."""
    return np.array([s for a, b in mask_to_ranges(m) for s in range(a, b - h + 1, h)],
                    dtype=int)


def band_around(level: float, width: float) -> tuple[float, float]:
    return round(max(level - width / 2, 0.0), 6), round(min(level + width / 2, 1.0), 6)


def sidecar_path(npy: str) -> str:
    return os.path.splitext(npy)[0] + ".json"


def write_sidecar(npy: str, meta: dict) -> None:
    with open(sidecar_path(npy), "w") as fh:
        json.dump(meta, fh, indent=2)


def read_sidecar(npy: str) -> dict | None:
    """The sample's sidecar; for files written before sidecars existed, the band is
    read from the name (_u0.90 -> [0.85, 0.95], _u0.85-0.95 -> itself)."""
    p = sidecar_path(npy)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    m = re.search(r"_u(\d*\.\d+)(?:-(\d*\.\d+))?", os.path.basename(npy))
    if not m:
        return None
    lo, hi = ((float(m.group(1)), float(m.group(2))) if m.group(2)
              else band_around(float(m.group(1)), 0.1))
    return dict(mode="legacy", band=[lo, hi], test_period=None)


def selection_tag(band, periods) -> str:
    """Short name of a test-day selection, for file names."""
    parts = []
    if band:
        parts.append(f"u{band[0]:.2f}-{band[1]:.2f}")
    for a, b in periods or []:
        parts.append(f"{a}_{b}")
    return "_".join(parts)
