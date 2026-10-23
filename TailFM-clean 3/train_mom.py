"""train_tailfm.py with a second condition: the vol direction.

    python momentum/train_mom.py --data data/returns_clean.csv --cond-valor 4155001 \\
        --steps 40000 --d-model 768 --holdout-level 0.9 --outdir runs/ho090_mom

Every flag of train_tailfm.py works the same (same split, same hold-out, same EVT
marginals, same training loop and seed), plus --mom-lag L (default 10).  The flow is
conditioned on (u_t, v_t): u = percentile of the EWMA vol (level), v = percentile of
m_t = sigma_t / sigma_{t-L} (direction).  With the same flags as a train_tailfm.py run, the
held-out days are identical, so the two models are compared on the same days.

Writes the train_tailfm.py files plus cond_v.npy and cond_m.npy; model_config.json has
cond_dim = 2.  Generate with momentum/generate_mom.py (generate.py does not load it).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

import train_tailfm                                                   # noqa: E402
from csvio import load_returns, feature_names_from_csv               # noqa: E402
from evt_shrink import shrink_ensemble                                # noqa: E402
from mom_common import VelocityField2, momentum                       # noqa: E402
from regime import (describe_ranges, holdout_ranges, level_band,      # noqa: E402
                    load_dates, mask_to_ranges, period_mask, regime_u)
from run_logging import tee_output                                    # noqa: E402
from splits import build_split                                        # noqa: E402
from tailfm import MarginalEnsemble, train_cfm                        # noqa: E402


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--mom-lag", type=int, default=10)
    extra, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = train_tailfm.parse_args()          # every train_tailfm.py flag, same defaults
    args.mom_lag = extra.mom_lag
    if not args.cond_valor:
        raise SystemExit("--cond-valor is required")
    return args


def run(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # ---------------------------------------------- data, split, hold-out (as train_tailfm)
    r = load_returns(args.data, args.prices)
    T, f = r.shape
    names = feature_names_from_csv(args.data, f)
    if args.cond_valor not in names:
        raise SystemExit(f"--cond-valor {args.cond_valor} is not a column of {args.data}")
    split = build_split(args, T, data=os.path.abspath(args.data))
    holdout = args.holdout_band or args.holdout_level is not None
    if args.train_all or args.train_period or args.exclude_period or holdout:
        split.test_starts = []
    if args.train_period or args.exclude_period or holdout:
        dates = load_dates(args.data, T)
        keep = (period_mask(dates, args.train_period) if args.train_period
                else np.ones(T, dtype=bool))
        if args.exclude_period:
            keep &= ~period_mask(dates, args.exclude_period)
        if holdout:
            band = (list(args.holdout_band) if args.holdout_band
                    else list(level_band(args.holdout_level)))
            u0, _ = regime_u(r[:, names.index(args.cond_valor)], keep, args.ewma_lam)
            ranges, held, n_band, n_eps = holdout_ranges(u0, keep, band, args.holdout_frac,
                                                         args.holdout_pad)
            per = [[str(dates[a])[:10], str(dates[b])[:10]] for a, b in ranges]
            for a, b in ranges:
                keep[a:b + 1] = False
            print(f"hold-out: {held} of {n_band} days with u in [{band[0]:.2f}, {band[1]:.2f}] "
                  f"({held / n_band:.0%}) in {len(ranges)} of {n_eps} episodes, +-"
                  f"{args.holdout_pad} days: " + ", ".join(f"{p} -> {q}" for p, q in per))
            json.dump(dict(band=band, frac=args.holdout_frac, pad=args.holdout_pad,
                           periods=per, band_days=n_band, held_band_days=held),
                      open(f"{args.outdir}/holdout.json", "w"), indent=2)
        split.train_ranges = mask_to_ranges(keep)
        print(f"training days: {int(keep.sum())} of {T} "
              f"({describe_ranges(dates, split.train_ranges)})")
    split.save(f"{args.outdir}/split.json")
    real = split.train_windows(r, args.n, args.stride)
    train_r = split.train_rows(r)
    print(f"data: T={T}, f={f} | device={device}")
    print(split.summary())

    # ------------------------------------------------ condition (u, v) of every day
    tr = split.train_row_mask()
    rj = r[:, names.index(args.cond_valor)]
    u, ref = regime_u(rj, tr, args.ewma_lam)
    m, v = momentum(rj, tr, args.ewma_lam, args.mom_lag)
    for name, arr in (("cond_u", u), ("cond_sigma_ref", ref), ("cond_m", m), ("cond_v", v)):
        np.save(f"{args.outdir}/{name}.npy", arr)
    starts = split.train_window_starts(args.n, args.stride)
    cond = torch.tensor(np.stack([u[starts], v[starts]], axis=1), dtype=torch.float32)
    print(f"condition: (u, v) of {args.cond_valor}, lambda={args.ewma_lam}, momentum lag "
          f"{args.mom_lag}; training windows starting with vol rising: "
          f"{(m[starts] > 1).mean():.0%}")

    # ------------------------------------------------ EVT marginals + PIT (as train_tailfm)
    marg = MarginalEnsemble(q_tail=args.q_tail, nu=args.nu).fit(train_r)
    if not args.no_shrink:
        shrink_ensemble(marg, train_r, c=args.shrink_c)
    z = torch.tensor(marg.transform(real), dtype=torch.float32)
    print(f"PIT: max|z| = {np.abs(z.numpy()).max():.1f}")

    # --------------------------------------------------------------- training
    torch.manual_seed(args.seed)
    model = VelocityField2(f=f, n_max=args.n, d_model=args.d_model, depth=args.depth,
                           pos_std=args.pos_std)
    ema, _ = train_cfm(model, z, nu=marg.nu_, steps=args.steps, batch_size=args.batch,
                       mix_dim=args.mix_dim, device=device, seed=args.seed, cond=cond)
    torch.save(ema.shadow.state_dict(), f"{args.outdir}/model_ema.pt")
    json.dump(dict(n=args.n, d_model=args.d_model, depth=args.depth, pos_std=args.pos_std,
                   mix_dim=args.mix_dim, cond=True, cond_dim=2, cond_valor=args.cond_valor,
                   ewma_lam=args.ewma_lam, mom_lag=args.mom_lag, seed=args.seed),
              open(f"{args.outdir}/model_config.json", "w"), indent=2)
    pickle.dump(marg, open(f"{args.outdir}/marginals.pkl", "wb"))
    print(f"\nSaved into {args.outdir}/.  Next: python momentum/generate_mom.py "
          f"--data {args.data} --run {args.outdir} --holdout --cond-level 0.9 "
          f"--direction all rising falling")


def main():
    args = parse_args()
    log_path = args.log or f"{args.outdir}/report.log"
    with tee_output(log_path, header="momentum/train_mom.py"):
        run(args)
    print(f"Terminal report saved to {log_path}")


if __name__ == "__main__":
    main()
