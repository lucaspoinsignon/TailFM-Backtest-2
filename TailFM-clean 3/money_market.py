"""Money-market VALORs: carry + innovation.

A money-market price accrues the short rate, so its daily log return is almost known
one day ahead:

    r_t = c_t + eps_t,    c_t = a_{t-1} * dd_t,

dd_t = calendar days between the fund's own latest prices as of t-1 and t (3 on a
Monday, 0 on a row where the fund published no new price, so a fund holiday gives
r = c = eps = 0 and the next row carries all the days), and a_{t-1} = median of
r_s / dd_s over the priced rows among the `window` rows before t: the accrual per
calendar day, known at the close of t-1 (the median ignores a single jump; a longer
window absorbs less of a persistent mark-to-market drift but follows a rate change
later).  c_t is fixed by the past, so it is not risk; eps_t is: mark-to-market moves,
spread shocks, revaluations after rate surprises, NAV breaks.  The model is trained on
eps_t, and a scenario for day t is c_t plus a generated eps.  Over a multi-day window the
accrual is held at its value at the start; a rate change inside the window moves the path
by (change x days / 365), about 1 bp over 10 days for 50 bp.  Innovations below 1e-10
(the precision of a price file) are set to 0.

A VALOR whose innovations have fewer than --min-exceedances values beyond their 5% /
95% quantiles has nothing to model (a pure accrual series): it is "carry only",
r_t = c_t, left out of the model.

02_returns.py --mm-data writes, next to the returns CSV <stem>.csv:
    <stem>.csv          equity log returns, and eps_t for the modelled money-market VALORs
    <stem>.json         {"mm_model": [...], "mm_carry_only": [...], "carry_window": k,
                         "mm_data": ..., "accrual_last": {VALOR: a_T}}
    <stem>_carry.csv    c_t of every money-market VALOR on every row (carry-only ones too)
train_tailfm.py reads <stem>.json: those columns get empirical marginals without EVT
(tailfm.evt.EmpiricalMarginal: generated innovations are observed values with their
observed frequencies, never beyond the observed range).  mm_check.py compares generated
and real money market.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def meta_path(data: str) -> str:
    return os.path.splitext(data)[0] + ".json"


def carry_path(data: str) -> str:
    return os.path.splitext(data)[0] + "_carry.csv"


def read_meta(data: str) -> dict | None:
    """The money-market metadata written next to `data`, or None."""
    p = meta_path(data)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        meta = json.load(fh)
    return meta if isinstance(meta, dict) and "mm_model" in meta else None


def split_carry(ret: pd.DataFrame, dd: pd.DataFrame, window: int):
    """(eps, carry, accrual after the last row) of money-market log returns `ret` (rows in
    time order), `dd` the calendar days each row covers per VALOR (0 = no new price)."""
    acc = (ret / dd).where(dd > 0)                               # accrual per calendar day
    med = acc.rolling(window, min_periods=1).median().ffill()   # known at each close
    carry = med.shift(1) * dd
    carry.iloc[0] = ret.iloc[0]                                  # no past on the first row
    eps = ret - carry
    eps = eps.mask(eps.abs() < 1e-10, 0.0)                       # below the price file's precision
    return eps, carry, med.iloc[-1]


def carry_only(eps: pd.DataFrame, n_train: int, q_tail: float, min_exc: int) -> list:
    """VALORs whose innovations have fewer than `min_exc` values strictly beyond the
    q_tail / 1 - q_tail quantiles on the first `n_train` rows (as for the equities)."""
    tr = eps.iloc[:n_train]
    n_lo = (tr < tr.quantile(q_tail)).sum()
    n_hi = (tr > tr.quantile(1 - q_tail)).sum()
    return list(eps.columns[np.minimum(n_lo, n_hi) < min_exc])
