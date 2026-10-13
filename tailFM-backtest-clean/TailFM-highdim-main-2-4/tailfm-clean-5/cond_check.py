"""Real vs generated equal-weight 1-day P&L across regimes u, from a saved --gen-cond run.

    python cond_check.py --data data/returns_clean.csv --run runs/cond_all
"""
import argparse, pickle
import numpy as np, torch
from csvio import load_returns
from tailfm import VelocityField, sample

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--n", type=int, default=24)               # as in training
ap.add_argument("--d-model", type=int, default=512)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--levels", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9])
ap.add_argument("--num", type=int, default=5000)
a = ap.parse_args()

r = load_returns(a.data, False); f = r.shape[1]
u = np.load(f"{a.run}/cond_u.npy")
marg = pickle.load(open(f"{a.run}/marginals.pkl", "rb"))
model = VelocityField(f=f, n_max=a.n, d_model=a.d_model, depth=a.depth, cond=True)
model.load_state_dict(torch.load(f"{a.run}/model_ema.pt", map_location="cpu"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

s = lambda y: (f"{y.mean()*1e4:7.2f} {y.std()*100:6.3f} "
               f"{-np.quantile(y, .05)*100:6.3f} {-np.quantile(y, .01)*100:6.3f}")
print("        real days |u - level| <= 0.05          generated day 1")
print("  u     mean(bp) sd(%) VaR95% VaR99%   N  |  mean(bp) sd(%) VaR95% VaR99%")
for lev in a.levels:
    m = np.abs(u - lev) <= 0.05
    x = r[m].mean(axis=1)                                          # real EW P&L
    z = sample(model, a.num, a.n, f, nu=marg.nu_, device=dev, cond=lev)
    g = marg.inverse_transform(z.numpy())[:, 0, :].mean(axis=1)    # generated EW P&L
    print(f"  {lev:.2f}  {s(x)} {m.sum():4d}  |  {s(g)}")
