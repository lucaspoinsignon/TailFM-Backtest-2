"""VaR / CVaR estimation from generated scenarios.

Turns any (M, n, f) array of windows into portfolio losses and risk numbers, so the
baselines and tailfm are scored by the same code.  Loss convention
L = -(portfolio log-return over the horizon) = -sum_{t<=h} sum_j w_j x_{t,j}, and

    VaR_a(L)  = inf { l : P(L <= l) >= a }
    CVaR_a(L) = E[ L | L >= VaR_a(L) ].

Two estimators: the empirical order statistic and tail mean, and a GPD-refined version
fitted to exceedances over the 90% quantile of the generated losses, which lets a
exceed the empirical resolution 1 - 1/M.  Uncertainty by bootstrap over windows.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def portfolio_losses(windows: np.ndarray, weights: np.ndarray | None = None,
                     horizon: int | None = None) -> np.ndarray:
    """(M, n, f) windows -> (M,) losses L = -sum_{t<=h} w^T x_t."""
    windows = np.asarray(windows, dtype=float)
    M, n, f = windows.shape
    w = np.full(f, 1.0 / f) if weights is None else np.asarray(weights, dtype=float)
    h = n if horizon is None else min(horizon, n)
    return -(windows[:, :h, :] @ w).sum(axis=1)


def var_cvar_empirical(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    losses = np.asarray(losses, dtype=float)
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(var), float(tail.mean())


def var_cvar_gpd(losses: np.ndarray, alpha: float,
                 q_thresh: float = 0.90) -> tuple[float, float]:
    """GPD-refined VaR/ES on the loss sample (POT above its q_thresh quantile).

    Falls back to the empirical estimator when the POT fit is not identified: fewer
    than 10 strict exceedances (degenerate generators tie their losses) or a failing
    GPD MLE.
    """
    losses = np.asarray(losses, dtype=float)
    u = np.quantile(losses, q_thresh)
    exc = losses[losses > u] - u
    if exc.size < 10:
        return var_cvar_empirical(losses, alpha)
    try:
        xi, _, beta = stats.genpareto.fit(exc, floc=0.0)
    except (ValueError, RuntimeError):
        return var_cvar_empirical(losses, alpha)
    xi = min(xi, 0.95)  # guard: keep ES finite
    p_u = 1.0 - q_thresh
    ratio = (1.0 - alpha) / p_u
    if abs(xi) > 1e-8:
        var = u + (beta / xi) * (ratio ** (-xi) - 1.0)
    else:
        var = u - beta * np.log(ratio)
    es = var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)
    return float(var), float(es)


def estimate_risk(gen_windows: np.ndarray, alphas=(0.95, 0.99, 0.995),
                  weights: np.ndarray | None = None, horizon: int | None = None,
                  n_boot: int = 200, seed: int = 0) -> dict:
    """Full risk report from generated windows, with bootstrap percentile CIs."""
    rng = np.random.default_rng(seed)
    L = portfolio_losses(gen_windows, weights, horizon)
    M = L.size
    report = {}
    for a in alphas:
        v_e, c_e = var_cvar_empirical(L, a)
        v_g, c_g = var_cvar_gpd(L, a)
        boot = np.empty((n_boot, 2))
        for b in range(n_boot):
            Lb = L[rng.integers(0, M, M)]
            boot[b] = var_cvar_gpd(Lb, a)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        report[a] = dict(var_emp=v_e, cvar_emp=c_e, var_gpd=v_g, cvar_gpd=c_g,
                         var_ci=(lo[0], hi[0]), cvar_ci=(lo[1], hi[1]))
    return report
