"""Acerbi-Szekely (2014) Expected Shortfall backtests Z1, Z2, Z3.

Acerbi, C. and Szekely, B., "Backtesting Expected Shortfall", MSCI, October 2014.

Three model-free, non-parametric statistics, each with EH0[Z] = 0 and EH1[Z] < 0
under risk under-estimation, so a NEGATIVE realisation is the signal and the test
is one-sided in the left tail.  Writing I_t = 1{X_t + VaR_{a,t} < 0}, N_T = sum I_t,
and with alpha the TAIL probability (2.5% in the Basel setting):

  Z1, ES after VaR (eq. 4).  From E[X/ES + 1 | X + VaR < 0] = 0,

        Z1 = (1/N_T) sum_t X_t I_t / ES_{a,t} + 1.

    An average taken over the exceptions themselves, so it is insensitive to how
    many there are: it tests the MAGNITUDE of the exceptions given that VaR has
    already been tested separately.  Undefined when N_T = 0.

  Z2, ES directly (eq. 6).  From the unconditional ES_a = -E[X I / alpha],

        Z2 = sum_t X_t I_t / (T alpha ES_{a,t}) + 1.

    Tests frequency and magnitude jointly; Z2 = 1 - (1 - Z1) N_T / (T alpha).
    Its null distribution is remarkably stable across tail shapes, which is what
    lets the paper propose the fixed traffic-light levels -0.70 and -1.8 (Table 4).

  Z3, ES from realised ranks (eq. 10).  With U_t = P_t(X_t) and the ES estimator
  ES^(N)_a(Y) = -(1/[Na]) sum_{i<=[Na]} Y_{i:N} on N draws,

        Z3 = -(1/T) sum_t ES^(T)_a(P_t^{-1}(U)) / E_V[ES^(T)_a(P_t^{-1}(V))] + 1,

    the denominator being the finite-sample mean of the same estimator under the
    model, computed analytically from eq. (11),

        E_V[ES^(N)_a] = -(N/[Na]) int_0^1 I_{1-p}(N - [Na], [Na]) P^{-1}(p) dp,

    which cancels the bias of the [Na]-order-statistic estimator.  Z3 tests the
    whole predicted distribution against first-order stochastic dominance, and is
    the most powerful of the three against tail-index misspecification.

Z1 needs at least one exception and Z3 needs [T alpha] >= 1, i.e. T >= 1/alpha = 40
at alpha = 2.5%; at the ten-day horizon a 20% held-out sample gives T ~ 24, so only Z2
is computable.  There is no asymptotic null, so p = P_{H0}(Z < Z_obs) is simulated per
eq. (12) -- conditioned on N_T > 0 for Z1, and drawn directly on the ranks
U ~ U(0,1)^T for Z3, which is exactly the null.

Under an unconditional generator P_t = P, so `pred` is a single `PredictivePnL`; the
statistics still take per-t arrays of VaR and ES so a conditional model drops in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import special

from .predictive import PredictivePnL

__all__ = ["ASResult", "z1_statistic", "z2_statistic", "z3_statistic",
           "test_z1", "test_z2", "test_z3", "es_estimator_mean",
           "Z2_TRAFFIC_LIGHT", "traffic_light"]

# Table 4: Z2 critical levels, stable enough across tail shapes to be fixed
Z2_TRAFFIC_LIGHT = {"yellow": -0.70, "red": -1.80}


@dataclass
class ASResult:
    name: str
    statistic: float
    p_value: float
    n_obs: int
    n_exceptions: int
    expected_exceptions: float
    alpha: float
    defined: bool = True
    reason: str = ""
    extra: dict = field(default_factory=dict)

    def line(self) -> str:
        if not self.defined:
            return f"  {self.name}: not defined -- {self.reason}"
        verdict = "reject" if self.p_value < 0.05 else "not rejected"
        return (f"  {self.name}: Z = {self.statistic:+8.4f}   p = "
                f"{self.p_value:.4f}  ({verdict} at 5%)")


def _undefined(name, reason, n, exc, expd, alpha) -> ASResult:
    return ASResult(name=name, statistic=float("nan"), p_value=float("nan"),
                    n_obs=n, n_exceptions=exc, expected_exceptions=expd,
                    alpha=alpha, defined=False, reason=reason)


def _as_array(v, n) -> np.ndarray:
    """Broadcast a scalar prediction to the T per-day values the formulas use."""
    return np.full(n, float(v)) if np.isscalar(v) else np.asarray(v, dtype=float)


# ------------------------------------------------------------------ statistics
def z1_statistic(x: np.ndarray, var_t, es_t) -> float:
    """Eq. (4).  Returns nan when there is no exception."""
    x = np.asarray(x, dtype=float)
    var_t, es_t = _as_array(var_t, x.size), _as_array(es_t, x.size)
    ind = (x + var_t) < 0.0
    n_exc = int(ind.sum())
    if n_exc == 0:
        return float("nan")
    return float((x[ind] / es_t[ind]).sum() / n_exc + 1.0)


def z2_statistic(x: np.ndarray, var_t, es_t, alpha: float) -> float:
    """Eq. (6)."""
    x = np.asarray(x, dtype=float)
    var_t, es_t = _as_array(var_t, x.size), _as_array(es_t, x.size)
    ind = (x + var_t) < 0.0
    return float((x * ind / es_t).sum() / (x.size * alpha) + 1.0)


def _es_hat(y: np.ndarray, m: int) -> np.ndarray:
    """Estimator (9): -(1/m) * mean of the m smallest values, row-wise."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    part = np.partition(y, m - 1, axis=1)[:, :m]
    return -part.sum(axis=1) / m


def es_estimator_mean(pred: PredictivePnL, n: int, alpha: float,
                      n_grid: int = 200_001) -> float:
    """E_V[ES^(n)_alpha(P^{-1}(V))] by the quadrature of eq. (11).

    The weight w(p) = (n/m) I_{1-p}(n-m, m) integrates to 1, so this is a weighted
    average of predictive quantiles -- the finite-sample expectation of the
    [n alpha]-order-statistic ES estimator, computed deterministically because it
    sits in the denominator of Z3.
    """
    m = int(n * alpha)
    if m < 1:
        return float("nan")
    p = (np.arange(n_grid) + 0.5) / n_grid                 # midpoint rule
    w = (n / m) * special.betainc(n - m, m, 1.0 - p)
    return float(-np.mean(w * pred.ppf(p)))


def z3_statistic(x: np.ndarray, pred: PredictivePnL, alpha: float,
                 denom: float | None = None) -> float:
    """Eq. (10) for a time-invariant predictive distribution: with P_t = P the inner
    term does not depend on t, so the average over t is a single evaluation.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    m = int(n * alpha)
    if m < 1:
        return float("nan")
    y = pred.ppf(pred.cdf(x))                    # P^{-1}(U_t), U_t = P(X_t)
    if denom is None:
        denom = es_estimator_mean(pred, n, alpha)
    return float(-_es_hat(y[None, :], m)[0] / denom + 1.0)


# ----------------------------------------------------------------- MC p-values
def test_z1(x: np.ndarray, pred: PredictivePnL, alpha: float,
            n_sim: int = 10_000, seed: int = 0) -> ASResult:
    x = np.asarray(x, dtype=float)
    n = x.size
    var_a, es_a = pred.var(alpha), pred.es(alpha)
    n_exc = int(((x + var_a) < 0.0).sum())
    if n_exc == 0:
        return _undefined("Z1", f"no VaR_{alpha:.1%} exception in {n} observations "
                                f"(expected {n * alpha:.1f}); Z1 averages over "
                                f"exceptions and needs N_T > 0",
                          n, 0, n * alpha, alpha)
    z_obs = z1_statistic(x, var_a, es_a)
    rng = np.random.default_rng(seed)
    sims = pred.rvs((n_sim, n), rng)
    ind = (sims + var_a) < 0.0
    nt = ind.sum(axis=1)
    ok = nt > 0
    z_sim = np.where(ok, (sims * ind).sum(axis=1) / es_a / np.maximum(nt, 1) + 1.0,
                     np.nan)[ok]
    return ASResult("Z1", z_obs, float(np.mean(z_sim < z_obs)), n, n_exc,
                    n * alpha, alpha,
                    extra=dict(var=var_a, es=es_a, n_sim_used=int(ok.sum())))


def test_z2(x: np.ndarray, pred: PredictivePnL, alpha: float,
            n_sim: int = 10_000, seed: int = 0) -> ASResult:
    x = np.asarray(x, dtype=float)
    n = x.size
    var_a, es_a = pred.var(alpha), pred.es(alpha)
    n_exc = int(((x + var_a) < 0.0).sum())
    z_obs = z2_statistic(x, var_a, es_a, alpha)
    rng = np.random.default_rng(seed)
    sims = pred.rvs((n_sim, n), rng)
    ind = (sims + var_a) < 0.0
    z_sim = (sims * ind).sum(axis=1) / (n * alpha * es_a) + 1.0
    return ASResult("Z2", z_obs, float(np.mean(z_sim < z_obs)), n, n_exc,
                    n * alpha, alpha,
                    extra=dict(var=var_a, es=es_a,
                               traffic_light=traffic_light(z_obs),
                               sim_q05=float(np.quantile(z_sim, 0.05))))


def test_z3(x: np.ndarray, pred: PredictivePnL, alpha: float,
            n_sim: int = 10_000, seed: int = 0) -> ASResult:
    x = np.asarray(x, dtype=float)
    n = x.size
    m = int(n * alpha)
    var_a = pred.var(alpha)
    n_exc = int(((x + var_a) < 0.0).sum())
    if m < 1:
        return _undefined("Z3", f"[T*alpha] = [{n} * {alpha:g}] = 0; Z3 averages "
                                f"the {int(np.ceil(1 / alpha))} deepest order "
                                f"statistics and needs T >= 1/alpha = "
                                f"{int(np.ceil(1 / alpha))} observations",
                          n, n_exc, n * alpha, alpha)
    denom = es_estimator_mean(pred, n, alpha)
    z_obs = z3_statistic(x, pred, alpha, denom=denom)
    rng = np.random.default_rng(seed)
    # under H0 the ranks are exactly i.i.d. U(0, 1), so simulate them directly
    y_sim = pred.ppf(rng.random((n_sim, n)))
    z_sim = -_es_hat(y_sim, m) / denom + 1.0
    return ASResult("Z3", z_obs, float(np.mean(z_sim < z_obs)), n, n_exc,
                    n * alpha, alpha,
                    extra=dict(m=m, denom=denom,
                               sim_q05=float(np.quantile(z_sim, 0.05))))


def traffic_light(z2: float) -> str:
    """Table 4 fixed-threshold zones for Z2, no simulation required."""
    if not np.isfinite(z2):
        return "n/a"
    if z2 < Z2_TRAFFIC_LIGHT["red"]:
        return "red"
    return "yellow" if z2 < Z2_TRAFFIC_LIGHT["yellow"] else "green"
