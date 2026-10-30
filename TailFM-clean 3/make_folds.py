"""Plan K leave-episodes-out folds for a regime band and write the exact commands.

    python make_folds.py --run runs/u090 --band 0.87 0.93 --k 5

Band days (u in the band, u from --run's cond_u.npy) are grouped into episodes (a gap of more
than --gap rows starts a new one), and consecutive episodes into K folds with about equal
band-day counts.  Fold k is trained without each of its episodes widened by --pad rows on both
sides (an embargo), and tested on exactly those ranges; gap >= 2 * pad keeps the ranges of
different folds disjoint.  Calm periods between episodes stay in every fold's training set.
The training flags are copied from the "# command" line of --run's report.log (old runs
that generated during training too: --gen / --ode-steps / --no-recalibrate move to
generate.py, --gen-cond / --gen-mode / --band-width are dropped).  Writes folds.sh
(train, generate, backtest, one fold after the other) and folds.json (read by
pooled_ci.py).
"""
import argparse
import json
import shlex

import numpy as np

from regime import load_dates

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/returns_clean.csv")
ap.add_argument("--run", default="runs/u090", help="train-all run: cond_u.npy and report.log")
ap.add_argument("--band", type=float, nargs=2, default=[0.87, 0.93])
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--gap", type=int, default=50, help="rows between band days that start a new episode")
ap.add_argument("--pad", type=int, default=25, help="rows excluded from training on each side")
ap.add_argument("--levels", default="0.8 0.9", help="--levels generated in each fold run")
ap.add_argument("--level", type=float, default=0.9, help="level backtested on the band")
a = ap.parse_args()
assert a.gap >= 2 * a.pad, "--gap must be >= 2 * --pad, or folds would share test days"


def base_flags(log):
    """(training flags, generation flags) of the train-all run's command line, minus the
    ones each fold sets itself."""
    try:
        line = next(l for l in open(log) if l.startswith("# command"))
    except (OSError, StopIteration):
        print(f"WARNING: no '# command' line in {log}; edit <YOUR TRAINING FLAGS> in folds.sh")
        return "<YOUR TRAINING FLAGS>", ""
    tok, train, gen, i = shlex.split(line[len("# command"):])[1:], [], [], 0
    drop = {"--train-all": 0, "--outdir": 1, "--log": 1, "--exclude-period": 2,
            "--train-period": 2, "--gen-mode": 1, "--band-width": 1}
    to_gen = {"--gen": ("--num", 1), "--ode-steps": ("--ode-steps", 1),
              "--no-recalibrate": ("--no-recalibrate", 0)}
    while i < len(tok):
        key, has_eq = tok[i].split("=", 1)[0], "=" in tok[i]
        if key == "--gen-cond":
            i += 1
            if not has_eq:
                while i < len(tok) and not tok[i].startswith("--"):
                    i += 1
        elif key in drop:
            i += 1 + (0 if has_eq else drop[key])
        elif key in to_gen:
            name, k = to_gen[key]
            gen += [name] + ([tok[i].split("=", 1)[1]] if has_eq else tok[i + 1:i + 1 + k])
            i += 1 + (0 if has_eq else k)
        else:
            train.append(tok[i])
            i += 1
    return " ".join(shlex.quote(t) for t in train), " ".join(shlex.quote(t) for t in gen)


u = np.load(f"{a.run}/cond_u.npy")
T = u.size
dates = load_dates(a.data, T)
d = lambda i: str(dates[i])[:10]
idx = np.flatnonzero((u >= a.band[0]) & (u <= a.band[1]))
eps = np.split(idx, np.flatnonzero(np.diff(idx) > a.gap) + 1)
cum = np.cumsum([e.size for e in eps])
cuts = sorted({int(np.argmin(np.abs(cum - k * idx.size / a.k))) + 1 for k in range(1, a.k)})
bounds = [0] + cuts + [len(eps)]
folds = [eps[i:j] for i, j in zip(bounds[:-1], bounds[1:]) if j > i]

(flags, gen_flags), lev = base_flags(f"{a.run}/report.log"), f"{a.level:.2f}"
sh, meta = ["set -e"], []
print(f"{idx.size} band days in {len(eps)} episodes -> {len(folds)} folds")
for k, fe in enumerate(folds, 1):
    ranges = [(max(e[0] - a.pad, 0), min(e[-1] + a.pad, T - 1)) for e in fe]
    per = [(d(lo), d(hi)) for lo, hi in ranges]
    out, n_band = f"runs/fold{k}", sum(e.size for e in fe)
    print(f"  fold {k}: {n_band:4d} band days in {len(fe)} episodes, {per[0][0]} -> {per[-1][1]}, "
          f"{sum(hi - lo + 1 for lo, hi in ranges) / T:.1%} of rows left out of training")
    sh += [f"# fold {k}: {n_band} band days",
           f"python train_tailfm.py {flags} "
           + " ".join(f"--exclude-period {p} {q}" for p, q in per)
           + f" --outdir {out}",
           f"python generate.py --data {a.data} --run {out} --levels {a.levels}"
           + (f" {gen_flags}" if gen_flags else ""),
           f"python run_backtest.py --data {a.data} --gen {out}/generated_windows_u{lev}.npy "
           + " ".join(f"--test-period {p} {q}" for p, q in per)
           + f" --cond-band {a.band[0]} {a.band[1]} --horizon 1 --log {out}/bt_u{lev}.log", ""]
    meta.append(dict(run=out, test_periods=[list(p) for p in per]))
with open("folds.sh", "w") as fh:
    fh.write("\n".join(sh))
with open("folds.json", "w") as fh:
    json.dump(dict(band=a.band, folds=meta), fh, indent=2)
print("\nwrote folds.sh and folds.json; flags copied from the train-all run:\n  train:    "
      + flags + "\n  generate: " + (gen_flags or "(defaults)"))
