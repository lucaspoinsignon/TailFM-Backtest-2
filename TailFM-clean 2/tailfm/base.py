"""Heavy-tailed source for flow matching: Student-t by normal variance mixing,

    x_0 = z * sqrt(nu / W),   z ~ N(0, I),   W ~ chi^2_nu.

Sharing W across a group of coordinates makes the group jointly elliptically t, hence
tail-dependent, where the Gaussian source is not.  mix_dim="window" shares W across
features and time, "time" across features only; neither creates within-window
volatility clustering, which comes from the velocity field (see model.pos_std).
"""

from __future__ import annotations

import torch


def sample_base(batch: int, n: int, f: int, nu: float,
                mix_dim: str = "window",
                device: torch.device | str = "cpu",
                generator: torch.Generator | None = None) -> torch.Tensor:
    z = torch.randn(batch, n, f, device=device, generator=generator)
    w_shape = (batch, 1, 1) if mix_dim == "window" else (batch, n, 1)
    # chi^2_nu = Gamma(shape=nu/2, rate=1/2)
    w = torch.distributions.Gamma(nu / 2.0, 0.5).sample(w_shape).to(device)
    return z * torch.sqrt(torch.as_tensor(nu, device=device) / w.clamp_min(1e-8))
