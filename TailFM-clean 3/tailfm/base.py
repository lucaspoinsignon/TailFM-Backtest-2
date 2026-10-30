"""Heavy-tailed source for flow matching: Student-t by normal variance mixing,

    x_0 = z * sqrt(nu / W),   z ~ N(0, I),   W ~ chi^2_nu.

Sharing W across a group of coordinates makes the group jointly elliptically t, hence
tail-dependent, where the Gaussian source is not.  mix_dim="window" shares W across
features and time, "time" across features only; neither creates within-window
volatility clustering, which comes from the velocity field (see model.pos_std).

An elliptical source is symmetric (x_0 and -x_0 have the same law), so its copula has
lambda_L = lambda_U.  `skew = (b, kappa)` makes it asymmetric along one direction b of
feature space (unit vector, the market factor): on every day the Gaussian coordinate
zeta = z.b is replaced by the two-piece normal

    psi = kappa * zeta (zeta < 0),  zeta / kappa (zeta >= 0),  centred and scaled to unit
    variance,

the orthogonal part of z untouched.  Days stay independent given W, both tails keep
the index nu, and the quantile asymmetry along b is (q50 - q_p) / (q_{1-p} - q50) =
kappa^2 for every p, with or without the mixing.  kappa = 1 is the elliptical source,
with the same random numbers.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def two_piece(zeta: torch.Tensor, kappa: float) -> torch.Tensor:
    """Two-piece normal from standard normal zeta: mean 0, variance 1, A(p) = kappa^2."""
    psi = torch.where(zeta < 0, kappa * zeta, zeta / kappa)
    m = (1 / kappa - kappa) / math.sqrt(2 * math.pi)
    return (psi - m) / math.sqrt((kappa ** 2 + kappa ** -2) / 2 - m ** 2)


def sample_base(batch: int, n: int, f: int, nu: float,
                mix_dim: str = "window",
                device: torch.device | str = "cpu",
                generator: torch.Generator | None = None,
                skew: tuple | None = None) -> torch.Tensor:
    z = torch.randn(batch, n, f, device=device, generator=generator)
    if skew is not None:
        b, kappa = skew
        b = torch.as_tensor(b, dtype=z.dtype, device=device)
        zeta = z @ b                                             # (batch, n)
        z = z + (two_piece(zeta, kappa) - zeta).unsqueeze(-1) * b
    w_shape = (batch, 1, 1) if mix_dim == "window" else (batch, n, 1)
    # chi^2_nu = Gamma(shape=nu/2, rate=1/2)
    w = torch.distributions.Gamma(nu / 2.0, 0.5).sample(w_shape).to(device)
    return z * torch.sqrt(torch.as_tensor(nu, device=device) / w.clamp_min(1e-8))


def asymmetry(s: np.ndarray, p: float = 0.05) -> float:
    """A(p) = (q50 - q_p) / (q_{1-p} - q50): > 1 when the lower tail is longer."""
    lo, med, hi = np.quantile(s, [p, 0.5, 1 - p])
    return float((med - lo) / (hi - med))


def market_direction(z_rows: np.ndarray) -> np.ndarray:
    """Unit first principal direction of the latent rows, oriented so that sum(b) > 0."""
    b = np.linalg.eigh(np.cov(z_rows, rowvar=False))[1][:, -1]
    return b if b.sum() > 0 else -b


def fit_skew(z_rows: np.ndarray, p: float = 0.05) -> tuple[np.ndarray, float]:
    """(b, kappa) from the latent training rows: b = market direction, kappa^2 = A(p) of
    the scores z.b.  The latent marginals are symmetric t_nu, so A measures the copula's
    asymmetry only."""
    b = market_direction(z_rows)
    return b, math.sqrt(asymmetry(z_rows @ b, p))
