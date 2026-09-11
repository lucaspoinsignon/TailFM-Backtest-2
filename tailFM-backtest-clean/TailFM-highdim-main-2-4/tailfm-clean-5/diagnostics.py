"""Model-agnostic comparison of generated windows with real ones.

Everything here takes plain (M, n, f) arrays, so tailfm and every baseline are scored
by the same code; that is why this sits outside `tailfm/`.

  print_report          Hill tail index per feature and tail, lambda_L(q) on the worst
                        pairs, per-feature VaR/CVaR, squared-return ACF
  marginal_pit_report   is each fitted marginal right?  Scored on held-out rows
  novelty_report        are the generated windows new, or replayed training history?
  acf, pseudo_obs       building blocks reused by summarize_runs.py

Cost drives two choices in the tail-dependence code: pseudo-observations are computed
once per sample and reused, and pairs are screened with a single f x f indicator
product so only the worst `max_pairs` get full lambda(q) curves.  `max_rows`
subsamples the pooled rows; 200k leaves ~4000 exceedances at q = 0.02.

The density comparison lives in `figures.density_figure`; these are tables.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tailfm.evt import hill_estimator
from backtest.risk import var_cvar_empirical


MAX_ROWS = 200_000
MAX_PAIRS = 200


def _pool(windows: np.ndarray, max_rows: int | None = None,
          seed: int = 0) -> np.ndarray:
    """(M, n, f) -> (M*n, f), optionally subsampled to `max_rows` rows."""
    w = np.asarray(windows, dtype=float)
    w = w.reshape(-1, w.shape[-1])
    if max_rows is not None and w.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        w = w[np.sort(rng.choice(w.shape[0], max_rows, replace=False))]
    return w


def pseudo_obs(x: np.ndarray, chunk: int = 64) -> np.ndarray:
    """(N, f) -> rank/(N+1) pseudo-observations, in column blocks to cap memory."""
    x = np.asarray(x, dtype=float)
    n, f = x.shape
    u = np.empty((n, f), dtype=np.float32)
    rank = np.arange(1, n + 1, dtype=np.float32) / (n + 1.0)
    for a in range(0, f, chunk):
        b = min(a + chunk, f)
        # sort along the contiguous axis, then scatter instead of a second argsort:
        # 2.5x faster than argsort(axis=0) on a C-ordered block, and bit-identical
        blk = np.ascontiguousarray(x[:, a:b].T)          # (cols, n)
        idx = np.argsort(blk, axis=1)
        out = np.empty(blk.shape, dtype=np.float32)
        np.put_along_axis(out, idx, np.broadcast_to(rank, blk.shape), axis=1)
        u[:, a:b] = out.T
    return u


def hill_table(real: np.ndarray, gen: np.ndarray, k_frac: float = 0.02,
               max_rows: int | None = MAX_ROWS) -> dict:
    R, G = _pool(real, max_rows), _pool(gen, max_rows)
    out = {}
    for j in range(R.shape[1]):
        out[j] = {t: (hill_estimator(R[:, j], k_frac, t),
                      hill_estimator(G[:, j], k_frac, t))
                  for t in ("lower", "upper")}
    return out


def tail_dependence_curve(u: np.ndarray, i: int, j: int,
                          q_grid: np.ndarray, tail: str = "lower") -> np.ndarray:
    """Empirical lambda(q) for one pair, from pseudo-observations u: (N, f).  Ranking
    is the expensive step and does not depend on the pair, so callers rank once.
    """
    ui, uj = (u[:, i], u[:, j]) if tail == "lower" else (1.0 - u[:, i], 1.0 - u[:, j])
    return np.array([np.mean((ui < q) & (uj < q)) / q for q in q_grid])


def tail_dependence_matrix(u: np.ndarray, q: float,
                           tail: str = "lower") -> np.ndarray:
    """All f(f-1)/2 values of lambda_hat(q) as one f x f matrix, in a single BLAS call:

        Lambda(q) = B^T B / (q N),      B_tj = 1{U_tj < q}.
    """
    v = u if tail == "lower" else 1.0 - u
    b = (v < q).astype(np.float32)
    return (b.T @ b) / (q * b.shape[0])


def tail_dependence_report(real: np.ndarray, gen: np.ndarray,
                           q_grid: np.ndarray | None = None,
                           tail: str = "lower",
                           max_pairs: int | None = MAX_PAIRS,
                           max_rows: int | None = MAX_ROWS,
                           seed: int = 0) -> dict:
    """{(i, j): (lambda_real(q_grid), lambda_gen(q_grid))} for the worst pairs.

    Pairs are ranked by |lambda_gen(q0) - lambda_real(q0)| at the largest q on the
    grid and only the top `max_pairs` get full curves; max_pairs=None restores every
    pair, which is O(f^2) curves.
    """
    if q_grid is None:
        q_grid = np.linspace(0.01, 0.10, 10)
    q_grid = np.asarray(q_grid, dtype=float)
    uR = pseudo_obs(_pool(real, max_rows, seed))
    uG = pseudo_obs(_pool(gen, max_rows, seed))
    f = uR.shape[1]

    q0 = float(q_grid.max())
    D = np.abs(tail_dependence_matrix(uG, q0, tail)
               - tail_dependence_matrix(uR, q0, tail))
    iu = np.triu_indices(f, 1)
    order = np.argsort(-D[iu])
    if max_pairs is not None:
        order = order[:max_pairs]

    out = {"q_grid": q_grid}
    for k in order:
        i, j = int(iu[0][k]), int(iu[1][k])
        out[(i, j)] = (tail_dependence_curve(uR, i, j, q_grid, tail),
                       tail_dependence_curve(uG, i, j, q_grid, tail))
    return out


def marginal_risk_table(real: np.ndarray, gen: np.ndarray,
                        alphas=(0.95, 0.99, 0.995),
                        max_rows: int | None = MAX_ROWS) -> dict:
    """Per-feature 1-step VaR/CVaR of the loss -x, real vs. generated."""
    R, G = _pool(real, max_rows), _pool(gen, max_rows)
    out = {}
    for j in range(R.shape[1]):
        out[j] = {a: (var_cvar_empirical(-R[:, j], a), var_cvar_empirical(-G[:, j], a))
                  for a in alphas}
    return out


def acf(x: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Mean-over-windows autocorrelation of a (M, n) array, lags 1..max_lag."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)
    denom = (x ** 2).sum(axis=1)
    return np.array([((x[:, :-k] * x[:, k:]).sum(axis=1) / (denom + 1e-12)).mean()
                     for k in range(1, max_lag + 1)])


def print_report(real: np.ndarray, gen: np.ndarray, feature_names=None,
                 max_pairs: int = 30, max_rows: int | None = MAX_ROWS) -> None:
    f = real.shape[-1]
    names = feature_names or [f"feat{j}" for j in range(f)]

    # pool once and pass through with max_rows=None (_pool is a no-op on 2-D input);
    # `real`/`gen` are still needed in 3-D for the ACF
    R2, G2 = _pool(real, max_rows), _pool(gen, max_rows)

    print("\n=== Hill tail index (smaller = heavier; gen should match real) ===")
    for j, d in hill_table(R2, G2, max_rows=None).items():
        for t in ("lower", "upper"):
            r, g = d[t]
            print(f"  {names[j]:>8s} {t:>5s}:  real {r:6.2f}   gen {g:6.2f}")

    n_pairs = f * (f - 1) // 2
    print(f"\n=== Lower tail dependence lambda_L(q=0.02): worst "
          f"{min(max_pairs, n_pairs)} of {n_pairs} pairs by |gen - real| ===")
    td = tail_dependence_report(R2, G2, q_grid=np.array([0.02]),
                                max_pairs=max_pairs, max_rows=None)
    rows = [(k, v) for k, v in td.items() if k != "q_grid"]
    for (i, j), val in rows:
        print(f"  ({names[i]},{names[j]}):  real {val[0][0]:.3f}   "
              f"gen {val[1][0]:.3f}   diff {val[1][0] - val[0][0]:+.3f}")

    print("\n=== Marginal 1-step VaR / CVaR of loss (-x) ===")
    for j, d in marginal_risk_table(R2, G2, max_rows=None).items():
        for a, ((vr, cr), (vg, cg)) in d.items():
            print(f"  {names[j]:>8s} a={a:5.3f}:  VaR real {vr:8.4f} gen {vg:8.4f}"
                  f"  |  CVaR real {cr:8.4f} gen {cg:8.4f}")

    print("\n=== ACF (lag 1..5), squared series: volatility clustering ===")
    for j in range(f):
        ar = acf(real[:, :, j] ** 2, 5); ag = acf(gen[:, :, j] ** 2, 5)
        print(f"  {names[j]:>8s} sq-ACF real {np.round(ar, 2)}  gen {np.round(ag, 2)}")


# ------------------------------------------------------- marginal PIT goodness-of-fit
def ad_stat(u: np.ndarray) -> float:
    """Anderson-Darling A^2 for uniformity; weights the tails heavily."""
    u = np.sort(np.clip(np.asarray(u, dtype=float), 1e-12, 1 - 1e-12))
    n = u.size
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(u) + np.log1p(-u[::-1]))))


def marginal_pit_report(marg, train_rows: np.ndarray, test_rows: np.ndarray,
                        names, outdir: str, top: int = 20, prefix: str = "",
                        figure: bool = True) -> dict:
    """PIT goodness-of-fit per feature, on train and on the held-out rows.

    If F_j is the true CDF then F_j(X_j) ~ U(0,1) and z = T_nu^{-1}(F_j(X_j)) ~ t_nu.

        KS       sup_u |F_hat_n(u) - u| on the PIT values, with its p-value
        AD       Anderson-Darling, which unlike KS weights the tails
        max|z|   largest transformed value; under t_nu at n ~ 1200 the expected
                 maximum is ~8-15, so past ~50 means F_j is contradicted by the data
        n_beyond test observations outside the training range, where the GPD
                 extrapolates

    The train columns are not a valid test -- the body is the empirical CDF of those
    rows, so A^2 is near 0 by construction -- and the test p-value assumes iid PIT
    values, which return series are not.  Read the rejection rate against a placebo
    split inside the training window, not against 5%.

    `marg` should be the MarginalEnsemble the generator was fitted with, not a refit.
    """
    rows = []
    for j, m in enumerate(marg.marginals_):
        u_tr, u_te = m.cdf(train_rows[:, j]), m.cdf(test_rows[:, j])
        z_tr, z_te = m.transform(train_rows[:, j]), m.transform(test_rows[:, j])
        ks_tr = stats.kstest(u_tr, "uniform")
        ks_te = stats.kstest(u_te, "uniform")
        rows.append(dict(
            feature=names[j], xi_lo=m.xi_lo_, xi_hi=m.xi_hi_,
            KS_train=ks_tr.statistic, p_train=ks_tr.pvalue,
            KS_test=ks_te.statistic, p_test=ks_te.pvalue,
            AD_train=ad_stat(u_tr), AD_test=ad_stat(u_te),
            n_exc_lo=m.n_exc_lo_, n_exc_hi=m.n_exc_hi_,
            max_abs_z=max(float(np.abs(z_tr).max()), float(np.abs(z_te).max())),
            n_beyond=int((test_rows[:, j] < train_rows[:, j].min()).sum()
                         + (test_rows[:, j] > train_rows[:, j].max()).sum()),
        ))
    order = np.argsort([-r["AD_test"] for r in rows])
    rows_sorted = [rows[i] for i in order]
    f = len(rows)

    hdr = ("feature", "xi_lo", "xi_hi", "KS_test", "p_test", "AD_train", "AD_test",
           "max|z|", "n_beyond")
    key = ("feature", "xi_lo", "xi_hi", "KS_test", "p_test", "AD_train", "AD_test",
           "max_abs_z", "n_beyond")
    print(f"\n=== Marginal PIT goodness-of-fit: worst {min(top, f)} of {f} "
          f"features by held-out Anderson-Darling ===")
    print("".join(f"{h:>12s}" for h in hdr))
    for r in rows_sorted[:top]:
        print("".join(f"{r[k]:>12s}" if k == "feature" else
                      (f"{r[k]:>12d}" if isinstance(r[k], int) else f"{r[k]:>12.4g}")
                      for k in key))
    n_rej = sum(r["p_test"] < 0.05 for r in rows)
    print(f"  rejected at 5% (held-out KS): {n_rej} of {f}"
          f"   |   xi_lo > 0.5: {sum(r['xi_lo'] > 0.5 for r in rows)}"
          f"   max|z| > 50: {sum(r['max_abs_z'] > 50 for r in rows)}"
          f"   max|z| > 500: {sum(r['max_abs_z'] > 500 for r in rows)}")
    print("  (the held-out PIT values are not iid, so compare the rejection rate "
          "against a\n   placebo split inside the training window, not against 5%)")

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{prefix}marginal_pit.csv")
    with open(csv_path, "w") as fh:
        fh.write(",".join(rows_sorted[0].keys()) + "\n")
        for r in rows_sorted:
            fh.write(",".join(str(v) for v in r.values()) + "\n")

    png_path = None
    if figure:
        png_path = os.path.join(outdir, f"{prefix}marginal_pit.png")
        _pit_figure(rows_sorted, marg, test_rows, names, png_path)
    return dict(rows=rows_sorted, csv=csv_path, png=png_path, n_rejected=n_rej)


def _pit_figure(rows, marg, test_rows, names, path) -> str:
    p_test = np.array([r["p_test"] for r in rows])
    xi_lo = np.array([r["xi_lo"] for r in rows])
    mz = np.array([r["max_abs_z"] for r in rows])
    ad_tr = np.array([r["AD_train"] for r in rows])
    ad_te = np.array([r["AD_test"] for r in rows])
    f = len(rows)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0][0].hist(np.clip(p_test, 0, 1), bins=25, color="C0")
    ax[0][0].axhline(f / 25, ls="--", color="k", lw=1, label="uniform if all fit")
    ax[0][0].set_title("held-out KS p-values")
    ax[0][0].set_xlabel("p")
    ax[0][0].legend(fontsize=8)

    ax[0][1].scatter(xi_lo, mz, s=8, alpha=.6)
    ax[0][1].set_yscale("log")
    ax[0][1].axhline(50, ls="--", color="k", lw=1)
    ax[0][1].axvline(0.5, ls="--", color="r", lw=1)
    ax[0][1].set_xlabel(r"$\hat\xi_{lower}$")
    ax[0][1].set_ylabel("max |z|")
    ax[0][1].set_title(r"extreme $z$ comes from runaway $\hat\xi$")

    ax[1][0].scatter(ad_tr, ad_te, s=8, alpha=.6)
    lim = [0, float(np.nanpercentile(ad_te, 99)) or 1.0]
    ax[1][0].plot(lim, lim, "k--", lw=1)
    ax[1][0].set_xlim(lim); ax[1][0].set_ylim(lim)
    ax[1][0].set_xlabel("AD train (optimistic)")
    ax[1][0].set_ylabel("AD held-out")
    ax[1][0].set_title("in-sample vs out-of-sample fit")

    for r, tag in ((rows[0], "worst"), (rows[-1], "best")):
        j = names.index(r["feature"])
        u = np.sort(marg.marginals_[j].cdf(test_rows[:, j]))
        ax[1][1].plot(np.linspace(0, 1, u.size), u, lw=1.5,
                      label=f"{r['feature']} ({tag})")
    ax[1][1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1][1].set_xlabel("uniform quantile")
    ax[1][1].set_ylabel("PIT quantile")
    ax[1][1].set_title("PP plot of the held-out PIT values")
    ax[1][1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- novelty
def nn_dist(A: np.ndarray, B: np.ndarray, chunk: int = 512,
            exclude_self: bool = False) -> np.ndarray:
    """For each row of A, the distance to its nearest row of B.  A, B are (N, D)."""
    b2 = (B ** 2).sum(1)
    out = np.empty(A.shape[0])
    for s in range(0, A.shape[0], chunk):
        a = A[s:s + chunk]
        d2 = (a ** 2).sum(1)[:, None] + b2[None, :] - 2.0 * (a @ B.T)
        if exclude_self:                      # drop the zero on the diagonal
            for i in range(a.shape[0]):
                d2[i, s + i] = np.inf
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def novelty_report(real: np.ndarray, gens: dict, train_rows: np.ndarray,
                   n_gen: int = 3000, seed: int = 0) -> dict:
    """Are the generated windows new, or reshuffled training history?

    Every other diagnostic scores a marginal or a pairwise summary, all of which a
    model replaying the training windows would pass perfectly.  Nearest-neighbour
    distance in standardised window space (each coordinate over its training sd):

        d_gen   for each generated window, distance to its closest training window
        d_real  for each training window, distance to its closest other one
        d_boot  the same for a bootstrap resample, 0 by construction, reported as a
                check that the metric detects copying

    Read median(d_gen) / median(d_real): ~1 is what a sample from the same law looks
    like, << 1 is interpolation inside the observed set, ~0 is copying.  The minimum
    matters as much as the median.
    """
    sd = train_rows.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    flat = lambda W: (W / sd).reshape(W.shape[0], -1)
    R = flat(real)
    rng = np.random.default_rng(seed)

    d_real = nn_dist(R, R, exclude_self=True)
    boot = R[rng.integers(0, R.shape[0], min(n_gen, R.shape[0]))]
    d_boot = nn_dist(boot, R)
    med = float(np.median(d_real))

    print(f"\n=== Novelty: nearest-neighbour distance to the {real.shape[0]} "
          f"training windows ===")
    print(f"{'sample':>16}{'median d':>11}{'ratio':>8}{'min d':>10}"
          f"{'q05':>10}{'frac < 0.5x':>13}")
    print("-" * 68)
    row = lambda nm, d: print(
        f"{nm:>16}{np.median(d):11.2f}{np.median(d) / med:8.2f}{d.min():10.2f}"
        f"{np.quantile(d, .05):10.2f}{(d < .5 * med).mean():13.3f}")
    row("real (LOO)", d_real)
    row("bootstrap copy", d_boot)
    print("-" * 68)

    out = {}
    for label, gen in gens.items():
        idx = rng.choice(gen.shape[0], min(n_gen, gen.shape[0]), replace=False)
        d = nn_dist(flat(gen[np.sort(idx)]), R)
        out[label] = dict(median=float(np.median(d)),
                          ratio=float(np.median(d) / med), min=float(d.min()),
                          q05=float(np.quantile(d, .05)),
                          frac_close=float((d < .5 * med).mean()))
        row(label, d)
    print("\nratio ~1 = generated windows sit as far from the training set as "
          "training\nwindows sit from each other;  <<1 = interpolating inside it;  "
          "~0 = copying.\n'frac < 0.5x' is the share of generated windows closer to "
          "a training window\nthan half the typical real-to-real distance.")
    return out
