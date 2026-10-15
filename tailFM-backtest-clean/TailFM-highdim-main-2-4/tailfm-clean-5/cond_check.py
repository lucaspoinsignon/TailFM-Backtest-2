"""Regenerate from a saved --gen-cond run (no retraining) and compare real vs generated
equal-weight P&L on the matching real days.

Levels, as train_tailfm.py (--mode point: every window at u = U; band: u uniform in
[U - w/2, U + w/2]); the real days are those with u within w/2 of U:

    python cond_check.py --data data/returns_clean.csv --run runs/cond_all
    python cond_check.py ... --levels 0.3 0.9 --mode band --num 20000 --save

Matched (--match): the windows get the actual u of the chosen test days, resampled,
so the sample is the exact counterpart of a backtest on those days.  Choose the days
with --test-period (any dates, trained on or not: out-of-sample when not) and/or
--cond-band:

    python cond_check.py ... --match --test-period 2020-01-01 2020-12-31 \\
        --cond-band 0.85 0.95 --num 20000 --save

--save writes generated_windows_<...>.npy plus its .json sidecar into --run, so
evaluate.py and run_backtest.py validate it on the same days by default.  The
architecture comes from model_config.json; older runs use --n / --d-model / --depth.
"""
import argparse, json, os, pickle
import numpy as np, torch
from csvio import load_returns
from regime import (band_around, load_dates, period_mask, selection_tag,
                    write_sidecar)
from tailfm import VelocityField, sample
from train_tailfm import conditional_recalibrator, generation_plan

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--n", type=int, default=24)               # for runs without config
ap.add_argument("--d-model", type=int, default=512)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--levels", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9])
ap.add_argument("--mode", default="point", choices=["point", "band"])
ap.add_argument("--band-width", type=float, default=None,
                help="default: the run's (model_config.json), else 0.1")
ap.add_argument("--match", action="store_true",
                help="generate with the u of the days chosen by --test-period / "
                     "--cond-band instead of --levels")
ap.add_argument("--test-period", nargs=2, action="append", metavar=("START", "END"))
ap.add_argument("--cond-band", type=float, nargs=2, default=None, metavar=("LO", "HI"))
ap.add_argument("--num", type=int, default=5000)
ap.add_argument("--save", action="store_true",
                help="write the generated windows and their sidecar into --run")
ap.add_argument("--raw", action="store_true",
                help="skip the u-mixture recalibration, i.e. show the flow as it comes out")
a = ap.parse_args()

cfg_path = os.path.join(a.run, "model_config.json")
cfg = (json.load(open(cfg_path)) if os.path.exists(cfg_path)
       else dict(n=a.n, d_model=a.d_model, depth=a.depth, pos_std=0.1,
                 mix_dim="window", ode_steps=100))
width = a.band_width or cfg.get("band_width", 0.1)
r = load_returns(a.data, False); T, f = r.shape; n = cfg["n"]
u = np.load(f"{a.run}/cond_u.npy")
marg = pickle.load(open(f"{a.run}/marginals.pkl", "rb"))
model = VelocityField(f=f, n_max=n, d_model=cfg["d_model"], depth=cfg["depth"],
                      pos_std=cfg["pos_std"], cond=True)
model.load_state_dict(torch.load(f"{a.run}/model_ema.pt", map_location="cpu"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
kw = dict(n_steps=cfg["ode_steps"], mix_dim=cfg["mix_dim"], device=dev)
recal = None if a.raw else conditional_recalibrator(model, marg, a.num, n, f, **kw)
fits = np.arange(T) <= T - n                              # a whole window fits


def generate(c):
    z = sample(model, a.num, n, f, nu=marg.nu_, cond=c, **kw).numpy()
    return marg.inverse_transform(z if recal is None else recal(z))


def save(g, suffix, meta):
    if a.save:
        path = f"{a.run}/generated_windows{suffix}.npy"
        np.save(path, g)
        write_sidecar(path, dict(meta, cond_valor=cfg.get("cond_valor")))
        print(f"saved {path}")


def stats(w):                                            # (N, >=10, f) windows, EW P&L
    x1, x10 = w[:, 0, :].mean(axis=1), w[:, :10, :].mean(axis=2).sum(axis=1)
    q = lambda y, p: -np.quantile(y, p) * 100
    return (f"{x1.mean()*1e4:7.2f} {x1.std()*100:6.3f} {q(x1, .05):6.3f} {q(x1, .01):6.3f}"
            f" | 10d VaR95 {q(x10, .05):6.3f} VaR99 {q(x10, .01):6.3f}")


head = "mean(bp) sd(%) VaR95% VaR99% | 10-day loss %"
if a.match:
    if not (a.test_period or a.cond_band):
        raise SystemExit("--match needs --test-period and/or --cond-band")
    sel = fits.copy()
    if a.test_period:
        sel &= period_mask(load_dates(a.data, T), a.test_period)
    if a.cond_band:
        sel &= (u >= a.cond_band[0]) & (u <= a.cond_band[1])
    days = np.flatnonzero(sel)
    if days.size == 0:
        raise SystemExit("no day matches --test-period / --cond-band")
    g = generate(u[np.random.default_rng(0).choice(days, a.num)])
    save(g, "_match_" + selection_tag(a.cond_band, a.test_period),
         dict(mode="match", band=a.cond_band, test_period=a.test_period))
    real = np.stack([r[t:t + n] for t in days])
    print(f"{days.size} real windows starting on the chosen days")
    print(f"           {head}")
    print(f"  real     {stats(real)}")
    print(f"  gen      {stats(g)}")
else:
    print(f"mode {a.mode}; real days: |u - level| <= {width / 2:g}")
    print(f"  u     {head}")
    for lev in a.levels:
        c, suffix, meta = generation_plan(lev, a.mode, width, a.num, 0)
        g = generate(c)
        save(g, suffix, meta)
        lo, hi = band_around(lev, width)
        days = np.flatnonzero((u >= lo) & (u <= hi) & fits)
        real = np.stack([r[t:t + n] for t in days])
        print(f"  {lev:.2f} real {stats(real)}  N={days.size}")
        print(f"       gen  {stats(g)}")
