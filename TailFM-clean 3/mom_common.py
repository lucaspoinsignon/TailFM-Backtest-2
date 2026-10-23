"""Vol-direction (momentum) conditioning, shared by train_mom.py, generate_mom.py and
backtest_mom.py.  Nothing outside this folder is modified.

    m_t = sigma_t / sigma_{t-L}     EWMA vol of the conditioning VALOR, L = 10 days;
                                    both known at t, m_t > 1 <=> vol rising
    v_t = training percentile of m_t

A momentum run is conditioned on (u_t, v_t): u = vol level as in train_tailfm.py, v = vol
direction.  Flow, EVT marginals and training loop are the repo's; only the velocity field
gets a second condition embedding (VelocityField2).
"""
from __future__ import annotations

import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402
from scipy import stats                                               # noqa: E402

from regime import ewma_sigma, period_mask, selection_tag             # noqa: E402
from tailfm.base import sample_base                                   # noqa: E402
from tailfm.model import VelocityField, modulate, timestep_embedding  # noqa: E402

DIRECTIONS = ("all", "rising", "falling")


def momentum(rj, train_mask, lam, lag):
    """(m, v): m_t = sigma_t / sigma_{t-lag}, v_t = its percentile among the training days."""
    sig = ewma_sigma(rj, lam, init_var=rj[train_mask].var(), floor_var=1e-12)
    m = sig / np.concatenate([np.full(lag, sig[0]), sig[:-lag]])
    ref = np.sort(m[train_mask])
    return m, np.searchsorted(ref, m, side="right") / ref.size


class VelocityField2(VelocityField):
    """VelocityField conditioned on (u, v): each gets its own sinusoidal embedding + MLP,
    both added to the flow-time embedding, exactly as the 1-D condition is."""

    def __init__(self, **kw):
        super().__init__(cond=True, **kw)
        d = self.d_model
        self.c2_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, x, t, cond):
        B, n, _ = x.shape
        c = (self.t_mlp(timestep_embedding(t.reshape(B), self.d_model))
             + self.c_mlp(timestep_embedding(cond[:, 0], self.d_model))
             + self.c2_mlp(timestep_embedding(cond[:, 1], self.d_model)))
        h = self.in_proj(x) + self.pos[:, :n]
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.ada_out(c).chunk(2, dim=-1)
        return self.out_proj(modulate(self.norm_out(h), shift, scale))


@torch.no_grad()
def sample2(model, num, n, f, nu, cond, n_steps=100, mix_dim="window", device="cpu",
            batch_size=512, seed=0):
    """tailfm.cfm.sample with one condition per window: cond (num,) for a level-only
    model, (num, 2) for a momentum model.  Same noise and Heun steps as tailfm."""
    torch.manual_seed(seed)
    model = model.to(device).eval()
    c_all = torch.as_tensor(np.asarray(cond), dtype=torch.float32, device=device)
    out = []
    for start in range(0, num, batch_size):
        b = min(batch_size, num - start)
        x = sample_base(b, n, f, nu, mix_dim=mix_dim, device=device)
        c = c_all[start:start + b]
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t0 = torch.full((b, 1, 1), k * dt, device=device)
            v0 = model(x, t0, c)
            if k == n_steps - 1:            # terminal Euler step, avoids t = 1
                x = x + dt * v0
            else:
                v1 = model(x + dt * v0, t0 + dt, c)
                x = x + 0.5 * dt * (v0 + v1)
        out.append(x.cpu())
    return torch.cat(out, dim=0)


def recalibrator(model, marg, cond_mix, n, f, **kw):
    """generate.conditional_recalibrator for any condition: the mixture over `cond_mix`
    (the training conditions) is mapped to exact t_nu marginals by a monotone per-feature
    map, linear beyond the mixture's range."""
    num = len(cond_mix)
    zs = np.sort(sample2(model, num, n, f, marg.nu_, cond_mix, **kw).numpy()
                 .reshape(-1, f), axis=0)
    K = zs.shape[0]
    tq = stats.t.ppf(np.arange(1, K + 1) / (K + 1.0), df=marg.nu_)
    m = max(K // 100, 1)

    def recal(z):
        flat = z.reshape(-1, f).astype(float)
        for j in range(f):
            zj, x = zs[:, j], flat[:, j]
            y = np.interp(x, zj, tq)
            lo, hi = x < zj[0], x > zj[-1]
            y[lo] = tq[0] + (x[lo] - zj[0]) * (tq[m] - tq[0]) / (zj[m] - zj[0])
            y[hi] = tq[-1] + (x[hi] - zj[-1]) * (tq[-1] - tq[-1 - m]) / (zj[-1] - zj[-1 - m])
            flat[:, j] = y
        return flat.reshape(z.shape)
    return recal


def load_run(run):
    """(cfg, marginals, model, cond_dim) of a momentum or a level-only conditional run."""
    cfg = json.load(open(os.path.join(run, "model_config.json")))
    marg = pickle.load(open(os.path.join(run, "marginals.pkl"), "rb"))
    dim = cfg.get("cond_dim", 1)
    kw = dict(f=len(marg.marginals_), n_max=cfg["n"], d_model=cfg["d_model"],
              depth=cfg["depth"], pos_std=cfg["pos_std"])
    model = VelocityField2(**kw) if dim == 2 else VelocityField(cond=True, **kw)
    model.load_state_dict(torch.load(os.path.join(run, "model_ema.pt"), map_location="cpu"))
    return cfg, marg, model, dim


def add_day_args(ap):
    ap.add_argument("--cond-level", type=float, default=None, metavar="U",
                    help="days with u in [U - 0.03, U + 0.03]")
    ap.add_argument("--cond-band", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--holdout", action="store_true", help="days the run was not trained on")
    ap.add_argument("--test-period", nargs=2, action="append", metavar=("START", "END"))
    ap.add_argument("--exclude-period", nargs=2, action="append", metavar=("START", "END"))
    ap.add_argument("--direction", nargs="+", default=["all"], choices=DIRECTIONS,
                    help="vol rising (m > 1), falling (m <= 1) or all; several allowed")
    ap.add_argument("--lag", type=int, default=None,
                    help="momentum lag L in days (default: the run's, else 10)")


def selection(a):
    """(band, periods, excluded) of the day flags."""
    from regime import level_band
    band = (list(a.cond_band) if a.cond_band else
            list(level_band(a.cond_level)) if a.cond_level is not None else None)
    periods = [list(p) for p in a.test_period] if a.test_period else None
    excluded = [list(p) for p in a.exclude_period] if a.exclude_period else None
    return band, periods, excluded


def select_days(u, m, train_mask, dates, band, holdout, periods, excluded, direction):
    """Indices of the days matching the selection, chronological."""
    sel = np.ones(u.size, dtype=bool)
    if band:
        sel &= (u >= band[0]) & (u <= band[1])
    if holdout:
        sel &= ~train_mask
    if periods:
        sel &= period_mask(dates, periods)
    if excluded:
        sel &= ~period_mask(dates, excluded)
    if direction == "rising":
        sel &= m > 1.0
    elif direction == "falling":
        sel &= m <= 1.0
    return np.flatnonzero(sel)


def mom_tag(band, holdout, periods, excluded, direction):
    return selection_tag(band, periods, excluded, holdout) + f"_{direction}"
