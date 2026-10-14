"""Real vs generated equal-weight P&L for a saved --gen-cond run, and regeneration
from the saved model (no retraining).

Levels (one-condition models): generated at exactly u vs real days with |u - level|
<= 0.05, for several levels.  --save writes generated_windows_u<U>.npy into --run.

    python cond_check.py --data data/returns_clean.csv --run runs/cond_all
    python cond_check.py ... --levels 0.3 0.6 0.9 --num 20000 --save

Matched band (any model; the experiment mode): the evaluation days are those whose
REFERENCE u (--ref-u, e.g. the baseline run's cond_u.npy) lies in --band.  The model
is fed its own condition vectors of those days (cond_x.npy of --run), resampled, so
models conditioned on different variables are scored on the same real days.  --save
writes generated_windows_match_<LO>-<HI>.npy into --run.

    python cond_check.py ... --run runs/exp_mom --ref-u runs/exp_base/cond_u.npy \\
        --band 0.85 0.95 --num 20000 --save

The architecture is read from model_config.json in --run when present (runs trained
after it existed); older runs use --n / --d-model / --depth.
"""
import argparse, json, os, pickle
import numpy as np, torch
from csvio import load_returns
from tailfm import VelocityField, sample
from train_tailfm import conditional_recalibrator

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--n", type=int, default=24)               # as in training
ap.add_argument("--d-model", type=int, default=512)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--levels", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9])
ap.add_argument("--ref-u", type=str, default=None,
                help="u file selecting the evaluation days for --band "
                     "(default: the run's own first condition)")
ap.add_argument("--band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                help="matched-band mode instead of --levels")
ap.add_argument("--num", type=int, default=5000)
ap.add_argument("--save", action="store_true",
                help="write the generated windows into --run")
ap.add_argument("--raw", action="store_true",
                help="skip the u-mixture recalibration, i.e. show the flow as it comes out")
a = ap.parse_args()

cfg_path = os.path.join(a.run, "model_config.json")
cfg = (json.load(open(cfg_path)) if os.path.exists(cfg_path)
       else dict(n=a.n, d_model=a.d_model, depth=a.depth, pos_std=0.1, cond=1,
                 mix_dim="window", ode_steps=100))
r = load_returns(a.data, False); T, f = r.shape; n = cfg["n"]
cx_path = os.path.join(a.run, "cond_x.npy")
cx = np.load(cx_path) if os.path.exists(cx_path) else np.load(f"{a.run}/cond_u.npy")[:, None]
k = cx.shape[1]
marg = pickle.load(open(f"{a.run}/marginals.pkl", "rb"))
model = VelocityField(f=f, n_max=n, d_model=cfg["d_model"], depth=cfg["depth"],
                      pos_std=cfg["pos_std"], cond=k)
model.load_state_dict(torch.load(f"{a.run}/model_ema.pt", map_location="cpu"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
kw = dict(n_steps=cfg["ode_steps"], mix_dim=cfg["mix_dim"], device=dev)
pool = cx[:T - n + 1]                                    # conditions of every window start
recal = None if a.raw else conditional_recalibrator(
    model, marg, a.num, n, f, pool=None if k == 1 else pool, **kw)


def generate(c):
    z = sample(model, a.num, n, f, nu=marg.nu_, cond=c, **kw).numpy()
    return marg.inverse_transform(z if recal is None else recal(z))


def stats(w):                                            # (N, >=10, f) windows, EW P&L
    x1, x10 = w[:, 0, :].mean(axis=1), w[:, :10, :].mean(axis=2).sum(axis=1)
    q = lambda y, p: -np.quantile(y, p) * 100
    return (f"{x1.mean()*1e4:7.2f} {x1.std()*100:6.3f} {q(x1, .05):6.3f} {q(x1, .01):6.3f}"
            f" | 10d VaR95 {q(x10, .05):6.3f} VaR99 {q(x10, .01):6.3f}")


head = "mean(bp) sd(%) VaR95% VaR99% | 10-day loss %"
if a.band:
    lo, hi = a.band
    ref = np.load(a.ref_u) if a.ref_u else cx[:, 0]
    days = np.flatnonzero((ref >= lo) & (ref <= hi) & (np.arange(T) <= T - n))
    c = cx[np.random.default_rng(0).choice(days, a.num)]
    g = generate(c[:, 0] if k == 1 else c)
    if a.save:
        path = f"{a.run}/generated_windows_match_{lo:.2f}-{hi:.2f}.npy"
        np.save(path, g)
        print(f"saved {path}")
    real = np.stack([r[t:t + n] for t in days])
    print(f"band [{lo}, {hi}] of {a.ref_u or 'own u'}: {days.size} real windows; "
          f"model conditions k={k}")
    print(f"           {head}")
    print(f"  real     {stats(real)}")
    print(f"  gen      {stats(g)}")
else:
    if k != 1:
        raise SystemExit("--levels needs a one-condition model; use --band")
    print("        real days |u - level| <= 0.05  |  generated at u")
    print(f"  u     {head}")
    for lev in a.levels:
        days = np.flatnonzero((np.abs(cx[:, 0] - lev) <= 0.05) & (np.arange(T) <= T - n))
        g = generate(lev)
        if a.save:
            np.save(f"{a.run}/generated_windows_u{lev:.2f}.npy", g)
        real = np.stack([r[t:t + n] for t in days])
        print(f"  {lev:.2f} real {stats(real)}  N={days.size}")
        print(f"       gen  {stats(g)}")
