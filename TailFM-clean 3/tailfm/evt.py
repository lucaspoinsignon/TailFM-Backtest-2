"""Semi-parametric marginals via peaks-over-threshold.  Per feature and tail,

    F(x) = q_lo * GPD_sf(u_lo - x; xi_lo, beta_lo)          for x <  u_lo
    F(x) = empirical (interpolated)                         for u_lo <= x <= u_hi
    F(x) = 1 - q_hi * GPD_sf(x - u_hi; xi_hi, beta_hi)      for x >  u_hi

with GPD_sf(y; xi, beta) = (1 + xi y / beta)^(-1/xi).  The PIT z = T_nu^{-1}(F(x))
makes every marginal exactly t_nu, matching the flow-matching base, so the flow only
transports the copula.  The two tails are fitted separately and the body is pinned to
F(u_lo) = q_lo, F(u_hi) = 1 - q_hi so the piecewise CDF is continuous.  The GPD
parameters are refined afterwards by evt_shrink.shrink_ensemble.

Two kinds of series break that model, and both are handled rather than dropped:

  one-signed series   an accrual (money market, T-bill index) has its q-quantile ABOVE 0, so
                      the "loss tail" is the low end of the accrual rate plus a few real
                      down days, exceedances spanning orders of magnitude, and the GPD fits
                      xi >> 1 (xi = 5.5 extrapolated a 1e10 loss).  Such a feature gets
                      tails="empirical": the empirical CDF over the whole observed range, no
                      GPD and no extrapolation past the sample min / max, which reproduces
                      its loss frequency (often 0) and its jumps exactly.  fit() switches to
                      it whenever u_lo >= 0 or u_hi <= 0 or a GPD fit fails;
                      evt_shrink.shrink_ensemble also when xi_hat is outside (-0.5, 1).
  ties                stale days (zero returns) and quantised accruals are atoms of F.  ppf
                      returns the tied value v on the whole jump [F(v-), F(v)], so the
                      generator reproduces the atoms, and transform(x, v) spreads a tied x
                      over its jump (distributional transform, Rueschendorf 2009), so the
                      latent z is exactly t_nu with no atom the flow cannot represent.

'lower'/'upper' refer to tails of the raw variable; losses L = -r are the lower tail.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

_EPS = 1e-12


def hill_estimator(x: np.ndarray, k_frac: float = 0.05, tail: str = "lower") -> float:
    """Hill estimate of the tail index alpha (heavier tail <=> smaller alpha),

        alpha_hat^{-1} = (1/k) sum_{i=1..k} log( X_(n-i+1) / X_(n-k) ),

    on the positive exceedances of the requested tail.  k is k_frac of the full
    sample, so k_frac matches the threshold quantile used elsewhere.
    """
    z = np.asarray(x, dtype=float)
    y = -z if tail == "lower" else z
    k = max(10, int(k_frac * y.size))
    y = np.sort(y[y > 0.0])
    if y.size < k + 1:
        return np.inf
    inv_alpha = np.mean(np.log(y[-k:] / y[-k - 1]))
    return 1.0 / max(inv_alpha, _EPS)


def fit_gpd(exc: np.ndarray) -> tuple[float, float]:
    """MLE of (xi, beta) for exceedances over a threshold, location fixed at 0."""
    y = np.asarray(exc, dtype=float)
    y = y[np.isfinite(y) & (y > 0.0)]
    n = y.size
    if n < 5:
        raise ValueError(
            f"GPD fit needs >=5 positive exceedances, got {n}; the threshold "
            "coincides with the sample extremum (a point mass there, e.g. an "
            "accrual series that never falls)")
    xi, _, beta = stats.genpareto.fit(y, floc=0.0)
    if not (np.isfinite(xi) and np.isfinite(beta) and beta > 0.0):
        raise ValueError(f"GPD MLE did not converge on {n} exceedances")
    return float(xi), float(max(beta, _EPS))


class SemiParametricMarginal:
    """Marginal model for one feature: empirical body + GPD tails (tails="gpd") or the
    empirical CDF over the whole observed range (tails="empirical"), + t_nu PIT."""

    def __init__(self, q_tail: float = 0.05, nu: float = 5.0, tails: str = "gpd"):
        assert 0.0 < float(q_tail) < 0.5 and tails in ("gpd", "empirical")
        self.q_tail, self.nu, self.tails = float(q_tail), float(nu), tails

    def fit(self, x: np.ndarray) -> "SemiParametricMarginal":
        x = np.sort(np.asarray(x, dtype=float).ravel())
        n = self.n_ = x.size
        q = self.q_tail
        u_lo, u_hi = np.quantile(x, q), np.quantile(x, 1.0 - q)
        # a GPD tail needs a loss threshold below 0 and a gain threshold above 0
        self.tails_ = self.tails if u_lo < 0.0 < u_hi else "empirical"
        if self.tails_ == "gpd":
            e_lo, e_hi = u_lo - x[x < u_lo], x[x > u_hi] - u_hi
            try:
                self.xi_lo_, self.beta_lo_ = fit_gpd(e_lo)
                self.xi_hi_, self.beta_hi_ = fit_gpd(e_hi)
            except ValueError:
                self.tails_ = "empirical"
        if self.tails_ == "gpd":
            self.u_lo_, self.u_hi_ = float(u_lo), float(u_hi)
            self.q_lo_, self.q_hi_ = e_lo.size / n, e_hi.size / n     # mass beyond u
            self.n_exc_lo_, self.n_exc_hi_ = int(e_lo.size), int(e_hi.size)
        else:
            self.u_lo_, self.u_hi_ = float(x[0]), float(x[-1])
            self.q_lo_ = self.q_hi_ = 0.0
            self.xi_lo_ = self.beta_lo_ = self.xi_hi_ = self.beta_hi_ = np.nan
            self.n_exc_lo_ = self.n_exc_hi_ = 0

        # body: each distinct value v carries its ECDF jump [pl, pr] on plotting positions
        # (i - 0.5)/n (pl = pr if v occurs once); between values F is linear
        ux, first = np.unique(x, return_index=True)
        if ux.size < 2:
            raise ValueError("constant series")
        pl = (first + 0.5) / n
        pr = (np.r_[first[1:], n] - 1 + 0.5) / n
        keep = (ux >= self.u_lo_) & (ux <= self.u_hi_)
        ux, pl, pr = ux[keep], pl[keep], pr[keep]
        if self.tails_ == "gpd":                       # pin the body to the tails
            if ux[0] > self.u_lo_:
                ux, pl, pr = np.r_[self.u_lo_, ux], np.r_[self.q_lo_, pl], np.r_[self.q_lo_, pr]
            else:
                pl[0] = self.q_lo_
            if ux[-1] < self.u_hi_:
                ux, pl, pr = (np.r_[ux, self.u_hi_], np.r_[pl, 1.0 - self.q_hi_],
                              np.r_[pr, 1.0 - self.q_hi_])
            else:
                pr[-1] = 1.0 - self.q_hi_
        self._ux, self._pl, self._pr = ux, pl, pr
        # ppf knots: (pl, v), (pr, v) for each v; a flat run on [pl, pr] makes v an atom
        P, X = np.column_stack([pl, pr]).ravel(), np.repeat(ux, 2)
        k = np.r_[True, np.diff(P) > 0]
        self._pp, self._px = P[k], X[k]
        return self

    def _compat(self) -> None:
        """Marginals pickled before tails_ / ties existed: same behaviour as then."""
        if not hasattr(self, "_ux"):
            self.tails_ = "gpd"
            self._ux, self._pl, self._pr = self._body_x, self._body_p, self._body_p
            self._pp, self._px = self._body_p, self._body_x

    def _body_cdf(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        ux, pl, pr = self._ux, self._pl, self._pr
        i = np.searchsorted(ux, x)                          # first knot >= x
        k = np.clip(i - 1, 0, ux.size - 2)                  # knot interval [k, k + 1]
        w = np.clip((x - ux[k]) / (ux[k + 1] - ux[k]), 0.0, 1.0)
        out = pr[k] + w * (pl[k + 1] - pr[k])
        ic = np.minimum(i, ux.size - 1)
        at = ux[ic] == x                                    # x is a knot: inside its jump
        out[at] = pl[ic[at]] + v[at] * (pr[ic[at]] - pl[ic[at]])
        out[x < ux[0]] = pl[0]
        out[x > ux[-1]] = pr[-1]
        return out

    def cdf(self, x: np.ndarray, v: np.ndarray | None = None) -> np.ndarray:
        """F(x); a tied x is placed at pl + v (pr - pl) inside its jump (v = 0.5 default)."""
        self._compat()
        x = np.asarray(x, dtype=float)
        v = np.full(x.shape, 0.5) if v is None else np.asarray(v, dtype=float)
        out = self._body_cdf(x, v)
        if self.tails_ == "gpd":
            lo, hi = x < self.u_lo_, x > self.u_hi_
            if lo.any():
                out[lo] = self.q_lo_ * stats.genpareto.sf(
                    self.u_lo_ - x[lo], c=self.xi_lo_, scale=self.beta_lo_)
            if hi.any():
                out[hi] = 1.0 - self.q_hi_ * stats.genpareto.sf(
                    x[hi] - self.u_hi_, c=self.xi_hi_, scale=self.beta_hi_)
        return np.clip(out, _EPS, 1.0 - _EPS)

    def ppf(self, p: np.ndarray) -> np.ndarray:
        """Generalised inverse; the "empirical" tails stop at the sample min / max."""
        self._compat()
        p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
        out = np.interp(p, self._pp, self._px)
        if self.tails_ == "gpd":
            lo, hi = p < self.q_lo_, p > 1.0 - self.q_hi_
            if lo.any():
                out[lo] = self.u_lo_ - stats.genpareto.isf(
                    p[lo] / self.q_lo_, c=self.xi_lo_, scale=self.beta_lo_)
            if hi.any():
                out[hi] = self.u_hi_ + stats.genpareto.isf(
                    (1.0 - p[hi]) / self.q_hi_, c=self.xi_hi_, scale=self.beta_hi_)
        return out

    def transform(self, x: np.ndarray, v: np.ndarray | None = None) -> np.ndarray:
        """x -> z with z ~ t_nu marginally; v as in cdf (uniforms: exactly t_nu with ties)."""
        return stats.t.ppf(self.cdf(x, v), df=self.nu)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return self.ppf(stats.t.cdf(np.asarray(z, dtype=float), df=self.nu))


class MarginalEnsemble:
    """Per-feature SemiParametricMarginal for arrays shaped (..., f).  `empirical`: feature
    indices forced to tails="empirical" (the others get it only when the rule applies)."""

    def __init__(self, q_tail: float = 0.05, nu: float = 5.0, empirical=()):
        self.q_tail, self.nu, self.empirical = float(q_tail), float(nu), tuple(empirical)

    def fit(self, x: np.ndarray) -> "MarginalEnsemble":
        x = np.asarray(x, dtype=float)
        f = x.shape[-1]
        cols = x.reshape(-1, f)
        self.nu_ = self.nu
        self.marginals_ = []
        for j in range(f):
            tails = "empirical" if j in self.empirical else "gpd"
            try:
                self.marginals_.append(
                    SemiParametricMarginal(self.q_tail, self.nu_, tails).fit(cols[:, j]))
            except ValueError as e:      # name the offending column
                raise ValueError(f"feature index {j}: {e}") from None
        return self

    def summary(self) -> dict:
        m = self.marginals_
        return dict(
            q_lo=np.array([x.q_lo_ for x in m]),
            q_hi=np.array([x.q_hi_ for x in m]),
            xi_lo=np.array([x.xi_lo_ for x in m]),
            xi_hi=np.array([x.xi_hi_ for x in m]),
            n_exc_lo=np.array([x.n_exc_lo_ for x in m]),
            n_exc_hi=np.array([x.n_exc_hi_ for x in m]),
            empirical=np.array([getattr(x, "tails_", "gpd") == "empirical" for x in m]),
        )

    def _apply(self, x: np.ndarray, method: str) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        shape, f = x.shape, x.shape[-1]
        flat = x.reshape(-1, f)
        out = np.stack([getattr(self.marginals_[j], method)(flat[:, j])
                        for j in range(f)], axis=-1)
        return out.reshape(shape)

    def transform(self, x, seed: int | None = 0):
        """x -> z.  Tied values are spread over their jump with uniforms drawn from `seed`
        (None: jump mid-points), so z is exactly t_nu; transform each ROW once and window
        afterwards, or one day would get a different z in each of its windows."""
        if seed is None:
            return self._apply(x, "transform")
        x = np.asarray(x, dtype=float)
        flat = x.reshape(-1, x.shape[-1])
        rng = np.random.default_rng(seed)
        out = np.stack([m.transform(flat[:, j], rng.random(flat.shape[0]))
                        for j, m in enumerate(self.marginals_)], axis=-1)
        return out.reshape(x.shape)

    def inverse_transform(self, z):  return self._apply(z, "inverse_transform")
