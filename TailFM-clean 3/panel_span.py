"""Which VALORs cost the panel its history?  02_returns.py keeps a date only if every VALOR
has a price on it (after ffill), so one late or gappy series removes that date for all.

    python panel_span.py --prices data/prices.csv --returns data/returns05.csv  [same flags as 02_returns.py]

Replays 02_returns.py up to its dropna (non-positive and coverage filters, ffill) and reports
    lost dates      before the first complete date (late starts), inside, after the last
    multi-day rows  returns of the output that span k > 1 dates of the price panel: every
                    interior lost date turns the next return of ALL series into a k-day return
    culprits        greedy: VALOR whose removal recovers the most dates, then the next, ...
                    first = first price date; lead / gap = its NaN dates before its first
                    price / after it (after ffill); T = dates kept once it and all above are out
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True, help="price panel from 01_panel.py")
    p.add_argument("--returns", required=True, help="output of 02_returns.py")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--max-nonpos-frac", type=float, default=0.01)
    p.add_argument("--min-coverage", type=float, default=0.98)
    p.add_argument("--ffill-limit", type=int, default=2)
    p.add_argument("--top", type=int, default=20)
    a = p.parse_args()

    px = pd.read_csv(a.prices, index_col=0, parse_dates=True).sort_index()
    px.columns = [str(c) for c in px.columns]
    if a.start:
        px = px[px.index >= pd.Timestamp(a.start)]
    if a.end:
        px = px[px.index <= pd.Timestamp(a.end)]
    px = px.loc[:, (px <= 0).sum() / px.notna().sum() <= a.max_nonpos_frac].mask(px <= 0)
    px = px.loc[:, px.notna().mean() >= a.min_coverage]          # as 02_returns.py
    if a.ffill_limit > 0:
        px = px.ffill(limit=a.ffill_limit)
    out = set(pd.read_csv(a.returns, index_col=0, nrows=0).columns.astype(str))
    na = px.isna().to_numpy()
    names = np.array(px.columns)
    kept = ~na.any(axis=1)
    pos = np.flatnonzero(kept)
    print(f"{a.prices}: {len(px)} dates {px.index[0].date()} -> {px.index[-1].date()}, "
          f"{px.shape[1]} VALOR pass the coverage filter ({len(out)} of them are in {a.returns})")
    if pos.size == 0:
        raise SystemExit("no complete date at all")
    lead, tail = pos[0], len(px) - 1 - pos[-1]
    inner = pos[-1] - pos[0] + 1 - pos.size
    print(f"complete dates {pos.size}: {px.index[pos[0]].date()} -> {px.index[pos[-1]].date()}  |  "
          f"lost {len(px) - pos.size}: {lead} before, {inner} inside, {tail} after")

    k = np.diff(pos)                                             # dates spanned by each return
    if (k > 1).any():
        v, c = np.unique(k[k > 1], return_counts=True)
        print(f"multi-day returns: {int((k > 1).sum())} of {k.size} returns span k > 1 dates "
              f"(k: " + ", ".join(f"{x}x{y}" for x, y in zip(v, c)) + ")")
    else:
        print("multi-day returns: none")

    first = px.apply(pd.Series.first_valid_index)
    started = np.cumsum(~na, axis=0) > 0                         # (T, f): on or after its first price
    print(f"\n{'VALOR':>12s} {'first':>10s} {'lead':>6s} {'gap':>6s} {'T':>6s}  in output")
    alive = np.ones(names.size, dtype=bool)
    for _ in range(min(a.top, names.size)):
        cnt = na[:, alive].sum(axis=1)
        lost = cnt > 0
        if not lost.any():
            break
        share = (na[lost][:, alive] / cnt[lost, None]).sum(axis=0)   # fractional responsibility
        j = np.flatnonzero(alive)[np.argmax(share)]
        alive[j] = False
        T = int((~na[:, alive].any(axis=1)).sum())
        print(f"{names[j]:>12s} {str(first[names[j]].date()):>10s} {int((na[:, j] & ~started[:, j]).sum()):6d} "
              f"{int((na[:, j] & started[:, j]).sum()):6d} {T:6d}  {'yes' if names[j] in out else 'no (dropped later)'}")


if __name__ == "__main__":
    main()
