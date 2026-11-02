"""Step 2.  Price panel -> log returns, with every drop listed by VALOR.

    python 02_returns.py --data data/prices.csv --out data/returns.csv

Four filters, each removing series that break something downstream: non-positive
prices (log undefined), poor calendar coverage, a high zero-return fraction (stale or
coarsely quantised quotes), and too few exceedances for the GPD fits in tailfm.evt.

Money market: the VALORs listed in MM_VALORS below (columns of --data), or instead those
given by --mm (VALORs, or a .txt file listing them), and/or every column of --mm-data (a
second price panel, same format):

    python 02_returns.py --data data/prices.csv --out data/returns_mm.csv     # MM_VALORS
    python 02_returns.py --data data/prices.csv --mm-data data/prices_panel_2000.csv \\
        --out data/returns_mm.csv
    python 02_returns.py --data data/prices.csv --mm data/mm_valors.txt --out data/returns_mm.csv
    python 02_returns.py --data data/prices.csv --mm 4157124 4157176 --out data/returns_mm.csv

Their prices are carried forward onto the dates of --data (a fund's own holidays are not
missing data), so they never remove rows; a VALOR must have a positive price on the
first date.  Their log returns are split into carry + innovation (money_market.py): the
zero-return and exceedance filters are applied to the innovations instead, and a VALOR
failing them is kept as "carry only" rather than dropped.  Writes <out>.json and
<out>_carry.csv next to <out>; without --mm / --mm-data nothing changes.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from money_market import carry_only, carry_path, meta_path, split_carry

# Money-market VALORs (column names of --data, as strings).  Their returns are split into
# carry + innovation; listed VALORs missing from --data are ignored with a warning.
# Empty: no money market.  --mm on the command line replaces this list.
MM_VALORS = [
    # "4157124",
    # "4157176",
]


def report(cols, stat, why, unit=""):
    if len(cols) == 0:
        return
    print(f"\ndropped {len(cols)} VALOR -- {why}:")
    for c in cols:
        print(f"  {c:>12}  {stat[c]:.4g}{unit}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="price panel from 01_panel.py")
    p.add_argument("--out", required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--max-nonpos-frac", type=float, default=0.01,
                   help="above this fraction of non-positive cells, drop the column")
    p.add_argument("--min-coverage", type=float, default=0.98)
    p.add_argument("--ffill-limit", type=int, default=2)
    p.add_argument("--max-zero-frac", type=float, default=0.05)
    p.add_argument("--q-tail", type=float, default=0.05,
                   help="must match train_tailfm.py --q-tail")
    p.add_argument("--min-exceedances", type=int, default=30)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--mm", nargs="+", default=None, metavar="VALOR",
                   help="money-market VALORs among the columns of --data, or one .txt "
                        "file listing them (see the docstring)")
    p.add_argument("--mm-data", default=None,
                   help="price panel of money-market VALORs (see the docstring)")
    p.add_argument("--carry-window", type=int, default=5,
                   help="rows in the median accrual of the money-market carry")
    a = p.parse_args()

    px = pd.read_csv(a.data, index_col=0, parse_dates=True).sort_index()
    px.columns = [str(c) for c in px.columns]
    if a.start:
        px = px[px.index >= pd.Timestamp(a.start)]
    if a.end:
        px = px[px.index <= pd.Timestamp(a.end)]
    print(f"{a.data}: {px.shape[0]} dates x {px.shape[1]} VALOR  "
          f"{px.index[0].date()} -> {px.index[-1].date()}")

    # ---- money-market columns of --data: set aside, handled below with --mm-data
    mm_parts = []
    ids = a.mm if a.mm is not None else MM_VALORS
    if ids:
        if len(ids) == 1 and ids[0].endswith(".txt"):
            ids = open(ids[0]).read().replace(",", " ").split()
        ids = list(dict.fromkeys(str(v).strip() for v in ids))
        missing = [v for v in ids if v not in px.columns]
        if missing:
            print(f"WARNING: money-market VALOR not in {a.data}, ignored: {missing}")
        ids = [v for v in ids if v in px.columns]
    if ids:
        mm_parts.append(px[ids])
        n0 = len(px)
        px = px.drop(columns=ids).dropna(how="all")      # dates priced by funds only
        print(f"--mm: {len(ids)} money-market VALOR taken out of {a.data} "
              f"({n0 - len(px)} dates with no other price dropped)")
    if a.mm_data:
        mm_parts.append(pd.read_csv(a.mm_data, index_col=0, parse_dates=True).sort_index())

    # ---- non-positive prices
    npf = (px <= 0).sum() / px.notna().sum()
    bad = list(px.columns[npf > a.max_nonpos_frac])
    report(bad, npf, f"non-positive price fraction > {a.max_nonpos_frac}")
    px = px.drop(columns=bad)
    iso = int((px <= 0).sum().sum())
    if iso:
        print(f"\nmasked {iso} isolated non-positive cells to NaN across "
              f"{int(((px <= 0).sum() > 0).sum())} VALOR")
        px = px.mask(px <= 0)

    # ---- coverage
    cov = px.notna().mean()
    bad = list(cov.index[cov < a.min_coverage])
    report(bad, cov, f"coverage < {a.min_coverage}")
    px = px.drop(columns=bad)
    if px.shape[1] == 0:
        raise SystemExit("--min-coverage dropped every column")

    before = len(px)
    if a.ffill_limit > 0:
        px = px.ffill(limit=a.ffill_limit)
    px = px.dropna(how="any")
    print(f"\nrows {before} -> {len(px)} after ffill(limit={a.ffill_limit}) + dropna")

    # ---- money market: carried forward onto these dates
    mm_cols = []
    if mm_parts:
        mm = pd.concat(mm_parts, axis=1).sort_index()
        mm.columns = [str(c) for c in mm.columns]
        clash = [c for c in mm.columns if c in px.columns]
        if clash or mm.columns.duplicated().any():
            raise SystemExit(f"money-market VALOR also among the equities or given twice: "
                             f"{clash or list(mm.columns[mm.columns.duplicated()])}")
        mm = mm.mask(mm <= 0)
        # date of each fund's latest own price, as of every row: carry accrues between them
        own = pd.DataFrame({c: pd.Series(mm.index, index=mm.index).where(mm[c].notna())
                            for c in mm.columns})
        own = own.reindex(own.index.union(px.index)).ffill().reindex(px.index)
        mm = mm.reindex(mm.index.union(px.index)).ffill().reindex(px.index)
        late = mm.iloc[0].isna()
        report(list(mm.columns[late]), pd.Series(mm.notna().mean(), index=mm.columns),
               f"money market without a positive price on {px.index[0].date()} "
               "(coverage shown)")
        mm, own = mm.loc[:, ~late], own.loc[:, ~late]
        mm_cols = list(mm.columns)
        px = px.join(mm)
        print(f"money market: {len(mm_cols)} VALOR, carried forward onto these {len(px)} dates")

    ret = np.log(px / px.shift(1)).dropna(how="any")
    print(f"returns: {ret.shape[0]} x {ret.shape[1]}")
    eq = [c for c in ret.columns if c not in mm_cols]

    # ---- stale quotes
    zf = (ret[eq] == 0.0).mean()
    bad = list(zf.index[zf > a.max_zero_frac])
    report(bad, zf, f"zero-return fraction > {a.max_zero_frac}")
    ret = ret.drop(columns=bad)

    # ---- EVT feasibility, on the training rows the marginals are fitted to
    eq = [c for c in ret.columns if c not in mm_cols]
    n_train = int((1 - a.test_frac) * len(ret))
    tr = ret[eq].iloc[:n_train]
    n_lo = (tr < tr.quantile(a.q_tail)).sum()
    n_hi = (tr > tr.quantile(1 - a.q_tail)).sum()
    worst = pd.concat([n_lo, n_hi], axis=1).min(axis=1)
    bad = list(worst.index[worst < a.min_exceedances])
    report(bad, worst, f"fewer than {a.min_exceedances} exceedances in a tail",
           " obs")
    ret = ret.drop(columns=bad)
    if ret.shape[1] == 0:
        raise SystemExit("everything filtered out; loosen the thresholds")

    # ---- money market: carry + innovation, the innovations replace the returns
    if mm_cols:
        dd = own.apply(lambda s: s.diff().dt.days).astype(float).reindex(ret.index)
        eps, carry, acc_last = split_carry(ret[mm_cols], dd, a.carry_window)
        only = carry_only(eps, n_train, a.q_tail, a.min_exceedances)
        model = [c for c in mm_cols if c not in only]
        acc = ret[mm_cols] / dd
        print(f"\nmoney market, carry = {a.carry_window}-row median accrual x calendar days "
              "since the fund's previous price:")
        print(f"  {'VALOR':>12} {'accrual %/yr':>13} {'[min, max]':>16} "
              f"{'eps q01 bp':>10} {'eps q99 bp':>10}  status")
        for c in mm_cols:
            ann = acc[c].rolling(20, min_periods=1).median() * 365 * 100
            q01, q99 = eps[c].quantile([0.01, 0.99]) * 1e4
            print(f"  {c:>12} {ann.median():13.2f} [{ann.min():6.2f},{ann.max():6.2f}] "
                  f"{q01:10.3f} {q99:10.3f}  "
                  + ("carry only (no innovation to model)" if c in only
                     else "modelled (innovation)"))
        ret[model] = eps[model]
        ret = ret.drop(columns=only)
        carry.index.name = "Date"
        carry.to_csv(carry_path(a.out), float_format="%.8e")
        with open(meta_path(a.out), "w") as fh:
            json.dump(dict(mm_model=model, mm_carry_only=only, carry_window=a.carry_window,
                           mm_data=a.mm_data, mm=a.mm,
                           accrual_last={c: float(v) for c, v in acc_last.items()}),
                      fh, indent=2)
        print(f"wrote {meta_path(a.out)} and {carry_path(a.out)}: {len(model)} modelled, "
              f"{len(only)} carry only")

    ret.index.name = "Date"
    ret.to_csv(a.out, float_format="%.8e")
    T, f = ret.shape
    ntr = int((1 - a.test_frac) * T)
    print(f"\nwrote {a.out}: T={T}  f={f}  "
          f"{ret.index[0].date()} -> {ret.index[-1].date()}")
    print(f"train rows {ntr}  exceedances/tail {int(a.q_tail * ntr)}  "
          f"n*f at n=24 is {24 * f} vs {ntr // 24} non-overlapping windows")


if __name__ == "__main__":
    main()
