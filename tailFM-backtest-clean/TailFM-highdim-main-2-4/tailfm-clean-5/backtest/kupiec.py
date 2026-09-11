"""Kupiec (1995) VaR verification tests.

Kupiec, P. H., "Techniques for Verifying the Accuracy of Risk Measurement Models",
Journal of Derivatives 3(2), 73-84.

Both tests treat the exception sequence I_t = 1{X_t + VaR_t < 0} as independent
Bernoulli(p) draws and test H0: p = p*, p* = 1 - confidence.

TUFF, time until first failure (paper eq. 2).  T~ is geometric, Prob(T~ = V) =
p (1-p)^{V-1}, and the LR statistic against the unconstrained MLE p_hat = 1/V is

    LR(V, p*) = -2 log[ p* (1-p*)^{V-1} ] + 2 log[ (1/V) (1 - 1/V)^{V-1} ]  ~ chi2_1.

PF, proportion of failures (paper eq. 4).  x failures in n trials, p_hat = x/n,

    LR(x, n, p*) = -2 log[ (1-p*)^{n-x} (p*)^x ] + 2 log[ (1-x/n)^{n-x} (x/n)^x ].

At V = 1 the factor (1 - 1/V)^{V-1} is 0^0; it is 1 here, which is the limit of
(1-1/V)^{V-1} as V -> 1 from above and the convention that reproduces Exhibit 1.
The same convention (0 log 0 = 0) is used for x = 0 and x = n in the PF statistic.

Both test UNCONDITIONAL coverage only: a model putting all its exceptions in one week
passes as long as it has the right number of them.  Christoffersen's independence test
is not implemented -- with a block-sampled held-out set the exception sequence is not
contiguous in calendar time, so a run-based statistic would not mean what it usually
means.

Power is poor at realistic sample sizes: at n = 255 and p* = 0.01 the PF test rejects
a true rate of 0.02 only 25% of the time, which `pf_power` reproduces from Exhibit 6.
`run_backtest.py` therefore prints the power of every test it runs.

Every function is pure and takes plain integers, so the paper's exhibits reproduce
without any data; see `reproduce_exhibits`.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["pf_lr", "pf_test", "tuff_lr", "tuff_test", "first_failure",
           "pf_nonrejection_region", "tuff_nonrejection_region", "pf_power",
           "pf_max_sample_size", "reproduce_exhibits"]


def _xlogy(x: float, y: float) -> float:
    """x log y with the convention 0 log 0 = 0."""
    return 0.0 if x == 0 else x * np.log(max(y, 1e-300))


# --------------------------------------------------- proportion of failures
def pf_lr(x: int, n: int, p_star: float) -> float:
    """LR statistic of the PF test, paper eq. (4)."""
    if n <= 0:
        return float("nan")
    ll0 = _xlogy(n - x, 1.0 - p_star) + _xlogy(x, p_star)
    ll1 = _xlogy(n - x, 1.0 - x / n) + _xlogy(x, x / n)
    return float(-2.0 * (ll0 - ll1))


def pf_test(x: int, n: int, p_star: float) -> dict:
    """Two-sided chi2_1 p-value, plus the one-sided binomial p-value.

    The LR test is two-sided: at n = 255, p* = 0.01, x = 0 rejects, zero exceptions
    being as unlikely under p = 0.01 as seven.  Only under-estimation matters to a
    regulator, so `p_value_upper` = P(X >= x) under Bin(n, p*) is reported alongside.
    """
    lr = pf_lr(x, n, p_star)
    return dict(test="PF", n=int(n), exceedances=int(x), expected=n * p_star,
                observed_rate=x / n if n else float("nan"),
                LR=lr, p_value=float(stats.chi2.sf(lr, df=1)),
                p_value_upper=float(stats.binom.sf(x - 1, n, p_star)),
                p_value_lower=float(stats.binom.cdf(x, n, p_star)))


# --------------------------------------------------- time until first failure
def tuff_lr(v: int, p_star: float) -> float:
    """LR statistic of the TUFF test, paper eq. (2).  `v` is 1-based."""
    if v < 1:
        return float("nan")
    ll0 = np.log(p_star) + (v - 1) * np.log(1.0 - p_star)
    ll1 = -np.log(v) + (0.0 if v == 1 else (v - 1) * np.log(1.0 - 1.0 / v))
    return float(-2.0 * (ll0 - ll1))


def first_failure(exceed: np.ndarray) -> int | None:
    """1-based index of the first exception, or None if there is none."""
    idx = np.flatnonzero(np.asarray(exceed, dtype=bool))
    return int(idx[0]) + 1 if idx.size else None


def tuff_test(v: int | None, p_star: float, n: int | None = None) -> dict:
    """TUFF at the observed V.  v=None (no failure in n trials) is censored: T~ is only
    bounded below by n, so the case is reported as such with P(T~ > n) = (1-p*)^n
    rather than treating n as V.
    """
    if v is None:
        p_none = (1.0 - p_star) ** (n or 0)
        return dict(test="TUFF", V=None, censored=True, n=n, LR=float("nan"),
                    p_value=float("nan"), p_no_failure=float(p_none))
    lr = tuff_lr(v, p_star)
    return dict(test="TUFF", V=int(v), censored=False, n=n, LR=lr,
                p_value=float(stats.chi2.sf(lr, df=1)),
                p_no_failure=float("nan"))


# ---------------------------------------------------------- regions and power
def pf_nonrejection_region(n: int, p_star: float,
                           level: float = 0.05) -> tuple[int, int]:
    """Inclusive [x_lo, x_hi] of failure counts NOT rejected -- paper Exhibit 5."""
    crit = stats.chi2.ppf(1.0 - level, df=1)
    keep = [x for x in range(n + 1) if pf_lr(x, n, p_star) <= crit]
    return (keep[0], keep[-1]) if keep else (1, 0)      # empty region: lo > hi


def tuff_nonrejection_region(p_star: float, level: float = 0.05,
                             v_max: int = 100_000) -> tuple[int, int]:
    """Inclusive [V_lo, V_hi] not rejected -- paper Exhibit 1.  LR(V) is U-shaped, so
    the region is the interval between its two roots; scanning is exact for integers.
    """
    crit = stats.chi2.ppf(1.0 - level, df=1)
    keep = [v for v in range(1, v_max + 1) if tuff_lr(v, p_star) <= crit]
    return (keep[0], keep[-1]) if keep else (1, 0)


def pf_power(n: int, p_star: float, p_true: float, level: float = 0.05) -> float:
    """P(reject H0: p = p* | true rate p_true) for the PF test.  Exhibit 6 tabulates
    the type II error rate, which is 1 - this.
    """
    lo, hi = pf_nonrejection_region(n, p_star, level)
    x = np.arange(n + 1)
    accept = (x >= lo) & (x <= hi)
    return float(1.0 - stats.binom.pmf(x[accept], n, p_true).sum())


def pf_max_sample_size(x: int, p_star: float, level: float = 0.05,
                       n_max: int = 20_000) -> int | None:
    """Largest n at which `x` failures still reject p = p* -- paper Exhibit 4.  For
    fixed x the rate x/n falls with n, so the answer is the last n that rejects.
    """
    crit = stats.chi2.ppf(1.0 - level, df=1)
    best = None
    for n in range(max(x, 1), n_max + 1):
        if pf_lr(x, n, p_star) > crit and x / n > p_star:
            best = n
        elif best is not None and x / n <= p_star:
            break
    return best


# ------------------------------------------------------------- paper exhibits
def reproduce_exhibits() -> str:
    """Recompute Exhibits 1, 4, 5 and 6 of Kupiec (1995) as a correctness check.
    Run with `python run_backtest.py --self-test`.

    Every cell of Exhibits 4 and 5 reproduces and Exhibit 6 to the printed precision.
    Known differences: three upper bounds and one lower bound of Exhibit 1, where
    LR(V) is scanned over integers here and Kupiec appears to have solved
    LR(V) = 3.841 for continuous V; Exhibit 1 at p* = 0.050 and Exhibit 5 at
    (n = 255, p* = 0.010), where the paper reports a one-sided region and the
    two-sided LR test rejects x = 0; and the cells Exhibit 4 leaves blank at very
    small n (x = 1, p* >= 0.03), where the LR does reject.
    """
    out = ["EXHIBIT 1 -- TUFF non-rejection regions for V",
           f"{'p*':>6s} {'5% level':>18s} {'10% level':>18s}"]
    for p in (0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050):
        a = tuff_nonrejection_region(p, 0.05)
        b = tuff_nonrejection_region(p, 0.10)
        out.append(f"{p:6.3f} {f'{a[0] - 1} < V < {a[1] + 1}':>18s} "
                   f"{f'{b[0] - 1} < V < {b[1] + 1}':>18s}")

    out += ["", "EXHIBIT 4 -- max sample size n at which x failures reject p = p*",
            f"{'x':>3s}" + "".join(f"{f'p*={p}':>9s}"
                                   for p in (0.01, 0.02, 0.03, 0.04, 0.05))]
    for x in range(1, 11):
        row = f"{x:>3d}"
        for p in (0.01, 0.02, 0.03, 0.04, 0.05):
            n = pf_max_sample_size(x, p)
            row += f"{(n if n else '--'):>9}"
        out.append(row)

    out += ["", "EXHIBIT 5 -- PF(0.05) non-rejection regions for x",
            f"{'p*':>6s}" + "".join(f"{f'n={n}':>18s}" for n in (255, 510, 1000))]
    for p in (0.010, 0.025, 0.050, 0.075, 0.100):
        row = f"{p:6.3f}"
        for n in (255, 510, 1000):
            lo, hi = pf_nonrejection_region(n, p)
            row += f"{f'{lo - 1} < x < {hi + 1}':>18s}"
        out.append(row)

    out += ["", "EXHIBIT 6 -- PF(0.05) type II error rates",
            f"{'p*':>6s} {'p':>6s} {'n=255':>8s} {'n=510':>8s} {'n=1000':>8s}"]
    for p_star, alts in ((0.010, (0.011, 0.020, 0.030, 0.040)),
                         (0.025, (0.028, 0.030, 0.040, 0.050)),
                         (0.050, (0.055, 0.060, 0.075, 0.100)),
                         (0.075, (0.083, 0.100))):
        for p in alts:
            row = f"{p_star:6.3f} {p:6.3f}"
            for n in (255, 510, 1000):
                row += f"{1.0 - pf_power(n, p_star, p):8.3f}"
            out.append(row)
    return "\n".join(out)
