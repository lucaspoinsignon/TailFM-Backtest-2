"""Step 6.  Generated vs real return windows, side by side.

    python 06_compare.py --data data/returns_clean.csv --run runs/final --dim 3
    python 06_compare.py --data data/returns_clean.csv --run runs/final \
        --valors V4156860,V4156861 --rank 1.0 --cumulative

Generated windows on the left, real training windows on the right, with the selected
VALORs overlaid in each panel, so what is compared is the joint path rather than the
marginals the EVT PIT matches by construction.

Without --valors the features are chosen greedily as the most mutually tail-dependent
group at q0.  Windows are picked at matched loss quantiles rather than at random;
--rank 1.0 shows the worst window in each sample.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from splits import add_split_args, resolve_split
from csvio import load_returns, feature_names_from_csv


def windows(x: np.ndarray, n: int, stride: int = 1) -> np.ndarray:
    idx = np.arange(0, len(x) - n + 1, stride)
    return np.stack([x[i:i + n] for i in idx])


def tail_dep_matrix(x: np.ndarray, q0: float) -> np.ndarray:
    """f x f matrix of lambda_hat(q0) = P(U_i < q0, U_j < q0) / q0."""
    n, f = x.shape
    u = np.empty((n, f), dtype=np.float32)
    for j in range(f):
        u[:, j] = (np.argsort(np.argsort(x[:, j])) + 1.0) / (n + 1.0)
    b = (u < q0).astype(np.float32)
    return (b.T @ b) / (q0 * n)


def most_dependent(lam: np.ndarray, dim: int) -> list[int]:
    """Greedy: start from the strongest pair, then add the feature with the highest
    mean lambda against the current set.
    """
    f = lam.shape[0]
    m = lam.copy(); np.fill_diagonal(m, -np.inf)
    i, j = np.unravel_index(np.argmax(m), m.shape)
    sel = [int(i), int(j)][:max(dim, 1)]
    while len(sel) < dim:
        score = m[sel].mean(axis=0)
        score[sel] = -np.inf
        sel.append(int(np.argmax(score)))
    return sorted(sel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True, help="outdir holding generated_windows.npy")
    p.add_argument("--out", default="fig/windows.png")
    p.add_argument("--prices", action="store_true")
    p.add_argument("--dim", type=int, default=3,
                   help="how many features to overlay in each panel")
    p.add_argument("--valors", default=None,
                   help="comma-separated names; overrides --dim")
    p.add_argument("--q0", type=float, default=0.05,
                   help="tail level used to pick the features when --valors is absent")
    p.add_argument("--n-windows", type=int, default=3, help="rows of the figure")
    p.add_argument("--n", type=int, default=24, help="window length, as fitted")
    p.add_argument("--horizon", type=int, default=10)
    add_split_args(p)
    p.add_argument("--split", type=str, default=None,
                    help="split.json from train_tailfm.py.  Without it the training set is rebuilt from --n/--test-frac/--horizon/--sampling and may not be the one the model saw.")
    p.add_argument("--rank", type=float, default=None,
                   help="loss quantile to select, e.g. 1.0 = worst window")
    p.add_argument("--cumulative", action="store_true",
                   help="plot cumulative return within the window")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    r = load_returns(a.data, a.prices)
    T, f = r.shape
    names = feature_names_from_csv(a.data, f)
    sp = resolve_split(a.split, a, T, data=os.path.abspath(a.data))
    print(sp.summary())
    train = sp.train_rows(r)

    if a.valors:
        sel_names = [s.strip() for s in a.valors.split(",")]
        miss = [s for s in sel_names if s not in names]
        if miss:
            raise SystemExit(f"not in {a.data}: {miss}\navailable: {names[:10]} ...")
        cols = [names.index(s) for s in sel_names]
    else:
        if not 1 <= a.dim <= f:
            raise SystemExit(f"--dim must be in [1, {f}]")
        lam = tail_dep_matrix(train, a.q0)
        cols = most_dependent(lam, a.dim)
        sel_names = [names[j] for j in cols]

    lam_sel = tail_dep_matrix(train[:, cols], a.q0)
    iu = np.triu_indices(len(cols), 1)
    lam_txt = (f", mean pairwise lambda({a.q0}) = {lam_sel[iu].mean():.2f}"
               if len(cols) > 1 else "")

    real = sp.train_windows(r, a.n)
    gen = np.load(f"{a.run}/generated_windows.npy")
    print(f"generated {gen.shape}   real {real.shape}   showing {sel_names}{lam_txt}")

    def pick(w):
        loss = -(w[:, :, cols].mean(axis=2)).sum(axis=1)   # equal-weight loss
        o = np.argsort(loss)
        if a.rank is not None:
            i = min(int(a.rank * (len(o) - 1)), len(o) - 1)
            return o[max(i - a.n_windows + 1, 0):i + 1][::-1]
        qs = np.linspace(0.5, 0.995, a.n_windows)
        return [o[int(q * (len(o) - 1))] for q in qs]

    ig, ir = pick(gen), pick(real)
    step = np.arange(a.n)
    nw = a.n_windows
    fig, axes = plt.subplots(nw, 2, figsize=(12, 3.0 * nw), squeeze=False, sharey=True)
    for row in range(nw):
        for col, (w, idx, tag) in enumerate(
                ((gen, ig, "generated"), (real, ir, "real"))):
            ax = axes[row][col]
            y = w[idx[row]][:, cols]
            if a.cumulative:
                y = np.cumsum(y, axis=0)
            for k, s in enumerate(sel_names):
                ax.plot(step, y[:, k], marker="o", ms=3, lw=1.2, label=s)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{tag}  window {idx[row]}", fontsize=10)
            ax.set_xlabel("step within window")
            if col == 0:
                ax.set_ylabel("cumulative return" if a.cumulative else "log return")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"Joint {a.n}-step windows, {len(cols)} VALOR "
                 f"({'worst' if a.rank else 'matched loss quantiles'}){lam_txt}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out, dpi=140)
    plt.close(fig)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
