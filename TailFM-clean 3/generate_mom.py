"""Generate for a set of real days, split by vol direction, from a momentum run
(train_mom.py) or a level-only run (train_tailfm.py).

    python momentum/generate_mom.py --data data/returns_clean.csv --run runs/ho090_mom \\
        --holdout --cond-level 0.9 --direction all rising falling
    python momentum/generate_mom.py --data data/returns_clean.csv --run runs/ho090 \\
        --holdout --cond-level 0.9 --direction all rising falling

One sample per --direction: each window is conditioned on the condition of a real day of
the selection, resampled: (u, v) for a momentum run, u for a level-only run (which cannot
tell rising from falling).  Days: --cond-level / --cond-band, --holdout, --test-period,
--exclude-period, then vol rising (m > 1) / falling / all.

Writes generated_windows_mom_<selection>_<direction>.npy + .json into --run; a table
compares each sample's equal-weight 1-day P&L with the real days.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

from csvio import load_returns, feature_names_from_csv               # noqa: E402
from generate import pnl_stats                                        # noqa: E402
from mom_common import (add_day_args, load_run, mom_tag, momentum,    # noqa: E402
                        recalibrator, sample2, select_days, selection)
from regime import load_dates, write_sidecar                          # noqa: E402
from splits import Split                                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="train_mom.py or train_tailfm.py run")
    add_day_args(ap)
    ap.add_argument("--num", type=int, default=20_000)
    ap.add_argument("--ode-steps", type=int, default=None, help="default: the run's, else 100")
    ap.add_argument("--seed", type=int, default=None, help="default: the run's seed")
    ap.add_argument("--no-recalibrate", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="print the table only")
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args()

    cfg, marg, model, dim = load_run(a.run)
    seed = cfg.get("seed", 0) if a.seed is None else a.seed
    lag = a.lag or cfg.get("mom_lag", 10)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    kw = dict(n_steps=a.ode_steps or cfg.get("ode_steps", 100), mix_dim=cfg["mix_dim"],
              device=device, seed=seed)
    n, f = cfg["n"], len(marg.marginals_)

    r = load_returns(a.data, False)
    T = r.shape[0]
    names = feature_names_from_csv(a.data, r.shape[1])
    split = Split.load(os.path.join(a.run, "split.json"))
    tr = split.train_row_mask()
    u = np.load(os.path.join(a.run, "cond_u.npy"))
    m, v = momentum(r[:, names.index(cfg["cond_valor"])], tr, cfg["ewma_lam"], lag)
    if dim == 2:
        v = np.load(os.path.join(a.run, "cond_v.npy"))       # the training values
    band, periods, excluded = selection(a)
    dates = load_dates(a.data, T) if periods or excluded else None
    print(f"run {a.run}: {'momentum (u, v)' if dim == 2 else 'level-only (u)'}, f={f}, "
          f"lag={lag} | num={a.num}, seed={seed}, ode_steps={kw['n_steps']}, device={device}")

    # recalibration over the training conditions (level-only: u uniform, as generate.py)
    recal = None
    if not a.no_recalibrate:
        if dim == 1:
            mix = (np.arange(a.num) + 0.5) / a.num
        else:
            starts = split.train_window_starts(n)
            mix = np.stack([u, v], axis=1)[np.random.default_rng(seed).choice(starts, a.num)]
        recal = recalibrator(model, marg, mix, n, f, **kw)

    x_real = r.mean(axis=1)
    print(f"\n{'sample':>58s} {'N':>6s} {'mean_bp':>8s} {'sd%':>6s} {'VaR95%':>7s} "
          f"{'VaR99%':>7s} {'ES97.5%':>8s}")
    for d in a.direction:
        days = select_days(u, m, tr, dates, band, a.holdout, periods, excluded, d)
        tag = mom_tag(band, a.holdout, periods, excluded, d)
        if days.size == 0:
            print(f"{tag:>58s}  no real day matches -- skipped")
            continue
        idx = np.random.default_rng(seed).choice(days, a.num)
        c = u[idx] if dim == 1 else np.stack([u[idx], v[idx]], axis=1)
        z = sample2(model, a.num, n, f, marg.nu_, c, **kw).numpy()
        gen = marg.inverse_transform(z if recal is None else recal(z))
        name = f"generated_windows_mom_{tag}.npy"
        print(f"{'real days ' + d:>58s} {days.size:6d} {pnl_stats(x_real[days])}")
        print(f"{name:>58s} {a.num:6d} {pnl_stats(gen[:, 0, :].mean(axis=1))}")
        if not a.no_save:
            path = os.path.join(a.run, name)
            np.save(path, gen)
            write_sidecar(path, dict(mode="mom_match", band=band, holdout=a.holdout,
                                     test_period=periods, exclude_period=excluded,
                                     direction=d, lag=lag, cond_dim=dim, num=a.num,
                                     seed=seed, cond_valor=cfg["cond_valor"]))


if __name__ == "__main__":
    main()
