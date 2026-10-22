"""The model's predictive distribution P_t of the h-step portfolio P&L.

Sign convention, following Acerbi-Szekely: P&L space X = -L = sum_{t<=h} w' x_t, so
X < 0 is a loss, while `backtest.risk` works in loss space.  Both risk measures are
positive numbers,

    VaR_alpha = -F^{-1}(alpha),    ES_alpha = -(1/alpha) int_0^alpha F^{-1}(q) dq,

with `alpha` the TAIL probability, not the confidence level; a VaR exception is
X + VaR_alpha < 0.  The CLI converts confidence levels once, at the boundary.

The generator is unconditional, so P_t = P for every t and this class carries no time
index; the tests still take per-t arrays, so a conditional model (McNeil-Frey, a GARCH
filter) drops in unchanged.

Tail refinement is off by default: at M = 20000 the empirical quantile resolves
1/M = 5e-5, far finer than the alpha being tested, so the GPD extrapolation would only
put a parametric assumption inside a non-parametric test.  `gpd_tail=True` restores it
for small M.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .risk import portfolio_losses

_EPS = 1e-12


class PredictivePnL:
    """Empirical (optionally GPD-tailed) law of the h-step portfolio P&L."""

    def __init__(self, x: np.ndarray, gpd_tail: bool = False,
                 q_thresh: float = 0.90):
        x = np.asarray(x, dtype=float).ravel()
        x = x[np.isfinite(x)]
        if x.size < 100:
            raise ValueError(f"predictive sample has {x.size} points; the tail "
                             "quantiles are not resolved")
        self.x_ = np.sort(x)
        self.M = self.x_.size
        self.gpd_tail = bool(gpd_tail)
        self.q_thresh = float(q_thresh)
        self._fit_tail()

    # ------------------------------------------------------------------- tail
    def _fit_tail(self) -> None:
        """POT fit to the LOSS tail, i.e. the lower tail of X."""
        self.p_u_ = 0.0
        if not self.gpd_tail:
            return
        L = -self.x_
        u = float(np.quantile(L, self.q_thresh))
        exc = L[L > u] - u
        if exc.size < 10:
            self.gpd_tail = False
            return
        try:
            xi, _, beta = stats.genpareto.fit(exc, floc=0.0)
        except (ValueError, RuntimeError):
            self.gpd_tail = False
            return
        self.u_, self.xi_ = u, float(min(xi, 0.95))
        self.beta_, self.p_u_ = float(beta), 1.0 - self.q_thresh

    # -------------------------------------------------------------- functions
    def ppf(self, p: np.ndarray | float) -> np.ndarray:
        """Generalised inverse F^{-1}(p) = inf{x : F(x) >= p}, in P&L space."""
        p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
        out = self.x_[np.clip(np.ceil(p * self.M).astype(int) - 1, 0, self.M - 1)]
        if self.gpd_tail:
            lo = p < self.p_u_
            if np.any(lo):
                ratio = p[lo] / self.p_u_ if p.ndim else np.atleast_1d(p / self.p_u_)
                if abs(self.xi_) > 1e-8:
                    q = self.u_ + (self.beta_ / self.xi_) * (ratio ** (-self.xi_) - 1.0)
                else:
                    q = self.u_ - self.beta_ * np.log(ratio)
                out = np.asarray(out, dtype=float)
                out[lo] = -q
        return out

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        """Empirical F(x), clipped off {0, 1} so the ranks stay invertible."""
        x = np.asarray(x, dtype=float)
        u = np.searchsorted(self.x_, x, side="right") / self.M
        return np.clip(u, 0.5 / self.M, 1.0 - 0.5 / self.M)

    def rvs(self, size, rng: np.random.Generator) -> np.ndarray:
        """i.i.d. draws, by inversion, so they follow the same ppf the tests use."""
        return self.ppf(rng.random(size))

    def var(self, alpha: float) -> float:
        """VaR at TAIL probability alpha, positive."""
        return float(-self.ppf(alpha))

    def es(self, alpha: float) -> float:
        """ES at TAIL probability alpha, positive.  Exact for the empirical law: with
        k = floor(alpha M),

            int_0^alpha F^{-1} = (1/M) sum_{i<=k} x_(i) + (alpha - k/M) x_(k+1),

        so no order statistic is dropped when alpha M is not an integer.
        """
        if self.gpd_tail:
            v = self.var(alpha)
            return float((v + self.beta_ - self.xi_ * self.u_) / (1.0 - self.xi_)) \
                if alpha < self.p_u_ else self._es_empirical(alpha)
        return self._es_empirical(alpha)

    def _es_empirical(self, alpha: float) -> float:
        k = int(np.floor(alpha * self.M))
        integral = self.x_[:k].sum() / self.M
        if k < self.M:
            integral += (alpha - k / self.M) * self.x_[k]
        return float(-integral / alpha)

    def __repr__(self) -> str:
        return (f"PredictivePnL(M={self.M}, mean={self.x_.mean():+.5f}, "
                f"min={self.x_[0]:+.5f}, gpd_tail={self.gpd_tail})")


def predictive_from_windows(gen_windows: np.ndarray, weights: np.ndarray | None,
                            horizon: int, **kw) -> PredictivePnL:
    """(M, n, f) generated windows -> predictive law of the h-step P&L."""
    return PredictivePnL(-portfolio_losses(gen_windows, weights=weights,
                                           horizon=horizon), **kw)


def realized_pnl(test_windows: np.ndarray, weights: np.ndarray | None,
                 horizon: int) -> np.ndarray:
    """(N, h, f) held-out blocks -> (N,) realised h-step P&L, chronological."""
    return -portfolio_losses(test_windows, weights=weights, horizon=horizon)
