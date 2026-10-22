"""Regime conditioning and real-day selection, shared by train_tailfm.py, generate.py,
evaluate.py, run_backtest.py, 06_compare.py and the analysis scripts.

    u_t      train percentile of the EWMA vol of one VALOR on day t.  sigma_t uses
             returns up to t-1, so u_t is known at t.  Saved as cond_u.npy by
             train_tailfm.py for every day, trained on or not.
    sidecar  <generated>.json beside a conditional sample: how it was generated and
             therefore which real days validate it by default (`band`, `test_period`,
             `exclude_period`).

Day selection (add_selection_args / resolve_selection): explicit flags win, else the
sample's sidecar, else nothing (all days, or the split's held-out blocks).
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


def regime_u(rj, train_mask, lam):
    """(u, ref): u_t = percentile of sigma_t among the training days' sigma (ref, sorted)."""
    sig = ewma_sigma(rj, lam, init_var=rj[train_mask].var(), floor_var=1e-12)
    ref = np.sort(sig[train_mask])
    return np.searchsorted(ref, sig, side="right") / ref.size, ref


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


def print_regime_periods(dates, u, ref, lo, hi, gap=5, top=8):
    """The days whose regime u is in [lo, hi], grouped into episodes (band days at most
    `gap` rows apart are merged), and their count per year."""
    idx = np.flatnonzero((u >= lo) & (u <= hi))
    print(f"regime u in [{lo:.2f}, {hi:.2f}] = EWMA vol "
          f"{np.quantile(ref, lo) * 100:.2f}%-{np.quantile(ref, hi) * 100:.2f}%, "
          f"{idx.size} days", end="")
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


# --------------------------------------------------------------------- sidecars
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


def selection_tag(band, periods, excluded=None) -> str:
    """Short name of a real-day selection, for file names."""
    parts = []
    if band:
        parts.append(f"u{band[0]:.2f}-{band[1]:.2f}")
    for a, b in periods or []:
        parts.append(f"{a}_{b}")
    for a, b in excluded or []:
        parts.append(f"ex{a}_{b}")
    return "_".join(parts)


# ---------------------------------------------------------------- day selection
def add_selection_args(ap, what: str = "days") -> None:
    """Flags choosing the real days a sample is scored against."""
    ap.add_argument("--test-period", nargs=2, action="append", metavar=("START", "END"),
                    help=f"only {what} inside [START, END] (YYYY-MM-DD, inclusive; "
                         f"repeatable), trained on or not")
    ap.add_argument("--exclude-period", nargs=2, action="append", metavar=("START", "END"),
                    help=f"leave out the {what} touching [START, END] (repeatable)")
    ap.add_argument("--cond-band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help=f"only {what} whose first day has regime u in [LO, HI] "
                         f"(cond_u.npy beside the sample)")
    ap.add_argument("--cond-level", type=float, default=None, metavar="U",
                    help="same with the band [U - w/2, U + w/2], w = --band-width")
    ap.add_argument("--band-width", type=float, default=0.1,
                    help="width w of the band around --cond-level (default 0.1)")
    ap.add_argument("--no-band", action="store_true",
                    help=f"{what} at every regime u: ignore the band in the sample's sidecar")


def resolve_selection(args, metas: list) -> tuple:
    """(periods, excluded, band): the flags of add_selection_args, else the sidecars of
    the samples when they all agree (`metas`: one dict per sample, {} if none)."""
    def from_sidecars(key):
        vals = {json.dumps(m.get(key)) for m in metas} or {"null"}
        if len(vals) > 1:
            print(f"WARNING: the samples were generated for different {key}s; "
                  f"pass the flag to choose one")
            return None
        return json.loads(vals.pop())

    if args.no_band:
        band = None
    elif args.cond_band:
        band = list(args.cond_band)
    elif args.cond_level is not None:
        band = list(band_around(args.cond_level, args.band_width))
    else:
        band = from_sidecars("band")
    periods = ([list(p) for p in args.test_period] if args.test_period
               else from_sidecars("test_period"))
    excluded = ([list(p) for p in args.exclude_period] if args.exclude_period
                else from_sidecars("exclude_period"))
    return periods or None, excluded or None, band or None


def describe_selection(periods, excluded, band) -> str:
    out = []
    if periods:
        out.append("inside " + ", ".join(f"{a} -> {b}" for a, b in periods))
    if excluded:
        out.append("without " + ", ".join(f"{a} -> {b}" for a, b in excluded))
    if band:
        out.append(f"regime u in [{band[0]:.2f}, {band[1]:.2f}]")
    return "; ".join(out) or "no restriction"


def real_window_starts(split, n, stride, dates, periods, excluded, band, u):
    """Starts of the real n-day windows a sample is compared with: the training windows,
    or every window lying inside `periods` (trained on or not); minus the windows touching
    `excluded`; then only those whose first day has u in `band`."""
    if periods:
        c = np.concatenate([[0], np.cumsum(period_mask(dates, periods))])
        starts = np.arange(split.T - n + 1)
        starts = starts[c[starts + n] - c[starts] == n]
    else:
        starts = split.train_window_starts(n, stride)
    if excluded:
        c = np.concatenate([[0], np.cumsum(period_mask(dates, excluded))])
        starts = starts[c[starts + n] - c[starts] == 0]
    if band:
        starts = starts[(u[starts] >= band[0]) & (u[starts] <= band[1])]
    return starts
