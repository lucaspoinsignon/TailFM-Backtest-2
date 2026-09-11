"""Train/test split over a return panel, shared by every script in the repo.

The default is a purged random-block split: test blocks are drawn from the whole
sample rather than taken as one contiguous stretch, so the held-out period is a
mixture of regimes rather than a statement about one.  Given a block length b:

1.  Tile [0, T) with the floor(T/b) candidate blocks [ib, (i+1)b).
2.  Draw K = round(test_frac * floor(T/b)) of them.  `sampling="stratified"` cuts the
    candidates into K contiguous strata and draws one block from each, so the test set
    spans the sample; `sampling="uniform"` draws K at random.
3.  Keep a training window [s, s+n) only if it contains no test row (the purge).

Step 3 is what makes the split leak-free: windows overlap at stride 1, so a test date
sits inside n distinct windows.  It is not free -- an isolated test block removes
b + n - 1 window starts rather than b.

A chronological split is the special case of one test block at the end, so
`chronological_split` returns the same object and every consumer keeps one code path.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, fields, asdict

import numpy as np


def file_fingerprint(path: str) -> str:
    """First 12 hex digits of the file's SHA-1, recorded in split.json so that
    `Split.check_data` can refuse a run whose --data was regenerated since training.
    """
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


@dataclass
class Split:
    """Row-level train/test partition of a (T, f) panel."""

    T: int
    n: int                       # model window length, needed by the purge
    block: int                   # test block length
    horizon: int                 # risk horizon the test blocks are cut for
    test_frac: float
    seed: int
    sampling: str
    test_starts: list = field(default_factory=list)
    data: str = ""
    data_sha1: str = ""

    # ------------------------------------------------------------------ masks
    def test_row_mask(self) -> np.ndarray:
        m = np.zeros(self.T, dtype=bool)
        for s in self.test_starts:
            m[s:s + self.block] = True
        return m

    def train_row_mask(self) -> np.ndarray:
        """Every row not inside a test block."""
        return ~self.test_row_mask()

    # ---------------------------------------------------------------- windows
    def train_window_starts(self, n: int | None = None,
                            stride: int = 1) -> np.ndarray:
        """Starts s with [s, s+n) free of test rows (the purge)."""
        n = self.n if n is None else n
        ok = self.train_row_mask().astype(np.int64)
        c = np.concatenate([[0], np.cumsum(ok)])           # c[i] = #ok in [0, i)
        s = np.arange(0, self.T - n + 1, stride)
        return s[(c[s + n] - c[s]) == n]

    def train_windows(self, r: np.ndarray, n: int | None = None,
                      stride: int = 1) -> np.ndarray:
        """(N, n, f) training windows: overlapping, purged, chronological order."""
        n = self.n if n is None else n
        r = np.asarray(r, dtype=float)
        starts = self.train_window_starts(n, stride)
        if starts.size == 0:
            raise ValueError(
                f"no training window of length {n} survives the purge "
                f"(block={self.block}, test_frac={self.test_frac}).  Use longer "
                f"test blocks (--test-block) or a smaller --test-frac.")
        return np.stack([r[s:s + n] for s in starts], axis=0)

    def train_rows(self, r: np.ndarray) -> np.ndarray:
        """(T_train, f) rows for the marginals: every non-blocked row.

        Rows are gathered, so consecutive rows are not consecutive in time: correct
        for the EVT marginals, wrong for anything reading a time axis.
        """
        return np.asarray(r, dtype=float)[self.train_row_mask()]

    def test_windows(self, r: np.ndarray, horizon: int | None = None) -> np.ndarray:
        """(N, h, f) non-overlapping h-step blocks cut out of the test blocks.

        floor(block / h) per test block, chronological, so the sequence can be read
        as Bernoulli trials by the Kupiec tests.
        """
        h = self.horizon if horizon is None else horizon
        r = np.asarray(r, dtype=float)
        out = []
        for s in sorted(self.test_starts):
            for i in range(self.block // h):
                out.append(r[s + i * h:s + (i + 1) * h])
        if not out:
            raise ValueError(f"horizon {h} exceeds the test block length "
                             f"{self.block}; nothing to backtest")
        return np.stack(out, axis=0)

    # ------------------------------------------------------------------- misc
    def id(self) -> str:
        """Short digest of everything that defines the partition; two splits are
        interchangeable iff their ids match.
        """
        payload = json.dumps(dict(
            T=self.T, n=self.n, block=self.block, horizon=self.horizon,
            sampling=self.sampling, test_starts=list(self.test_starts),
            data_sha1=self.data_sha1),
            sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

    def summary(self) -> str:
        te_rows = int(self.test_row_mask().sum())
        n_win = int(self.train_window_starts().size)
        n_raw = max(self.T - self.n + 1, 0)
        kind = ("chronological" if len(self.test_starts) == 1
                and self.test_starts[0] + self.block == self.T
                else f"{self.sampling} random-block")
        return (f"split[{self.id()}]: {kind}, {len(self.test_starts)} test block(s) of "
                f"{self.block} | rows train/test {self.T - te_rows}/{te_rows} | "
                f"train windows {n_win} of {n_raw} raw "
                f"({n_win / max(n_raw, 1):.0%} kept after purge) | "
                f"test h={self.horizon} blocks "
                f"{len(self.test_starts) * (self.block // self.horizon)}")

    def check_data(self, path: str, T: int) -> None:
        if T != self.T:
            raise SystemExit(
                f"split was built on T={self.T} rows, --data has {T}.  The row "
                f"indices do not refer to the same series; rebuild the split or "
                f"pass the data file used for training.")
        if self.data_sha1 and os.path.exists(path):
            got = file_fingerprint(path)
            if got != self.data_sha1:
                raise SystemExit(
                    f"--data has changed since the split was built "
                    f"(sha1 {got} != {self.data_sha1}).  Held-out rows may now be "
                    f"training rows; rerun train_tailfm.py or pass the original file.")

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "Split":
        with open(path) as fh:
            d = json.load(fh)
        # ignore retired fields so a split.json from an older run still loads
        known = {f.name for f in fields(Split)}
        dropped = sorted(set(d) - known)
        if dropped:
            print(f"note: {path} carries retired field(s) {', '.join(dropped)}; "
                  f"ignored")
        d = {k: v for k, v in d.items() if k in known}
        d["test_starts"] = [int(s) for s in d["test_starts"]]
        return Split(**d)


# --------------------------------------------------------------- constructors
def block_split(T: int, n: int, horizon: int, block: int | None = None,
                test_frac: float = 0.2, seed: int = 0,
                sampling: str = "stratified", data: str = "") -> Split:
    """Purged random-block split.  `block` defaults to 2 * horizon: each test block
    yields floor(block/h) held-out h-step losses but costs n-1 extra training window
    starts, so a longer block buys back training data with test-set diversity.
    """
    assert 0.0 < test_frac < 1.0, "--test-frac must be in (0, 1)"
    assert horizon >= 1 and n >= 1
    block = int(2 * horizon if block is None else block)
    if block < horizon:
        raise SystemExit(f"--test-block {block} < --horizon {horizon}: a test "
                         f"block must hold at least one h-step return")
    n_cand = T // block
    K = int(round(test_frac * n_cand))
    if K < 1:
        raise SystemExit(f"test_frac={test_frac} over {n_cand} candidate blocks of "
                         f"{block} rows selects {K} blocks; lower --test-block")
    rng = np.random.default_rng(seed)
    if sampling == "stratified":
        # one block per stratum, so the test set covers the whole sample
        edges = np.linspace(0, n_cand, K + 1).astype(int)
        idx = np.array([rng.integers(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])
                        if hi > lo])
    elif sampling == "uniform":
        idx = rng.choice(n_cand, size=K, replace=False)
    else:
        raise SystemExit(f"--sampling must be stratified or uniform, got {sampling!r}")
    starts = sorted(int(i) * block for i in np.unique(idx))
    return Split(T=int(T), n=int(n), block=block, horizon=int(horizon),
                 test_frac=float(test_frac), seed=int(seed),
                 sampling=sampling, test_starts=starts, data=data,
                 data_sha1=file_fingerprint(data) if data and os.path.exists(data) else "")


def chronological_split(T: int, n: int, horizon: int, test_frac: float = 0.2,
                        data: str = "") -> Split:
    """One test block covering the last `test_frac` of the sample.  Not the default;
    kept so `--sampling chronological` reproduces the previous numbers exactly.
    """
    cut = int((1.0 - test_frac) * T)
    return Split(T=int(T), n=int(n), block=int(T - cut), horizon=int(horizon),
                 test_frac=float(test_frac), seed=0,
                 sampling="chronological", test_starts=[cut], data=data,
                 data_sha1=file_fingerprint(data) if data and os.path.exists(data) else "")


def add_split_args(ap, *, test_frac: float = 0.2) -> None:
    """The split flags, identical in every script that builds one."""
    ap.add_argument("--test-frac", type=float, default=test_frac)
    ap.add_argument("--test-block", type=int, default=None,
                    help="test block length (default: 2 * --horizon).  Longer "
                         "blocks keep more training windows, shorter ones spread "
                         "the held-out sample over more regimes")
    ap.add_argument("--sampling", default="stratified",
                    choices=["stratified", "uniform", "chronological"],
                    help="how test blocks are placed; 'chronological' restores "
                         "the old last-20%% split")


def build_split(args, T: int, data: str = "") -> Split:
    """A Split from parsed `add_split_args` flags (+ --n, --horizon, --seed)."""
    if args.sampling == "chronological":
        return chronological_split(T, args.n, args.horizon, args.test_frac, data)
    return block_split(T, args.n, args.horizon, block=args.test_block,
                       test_frac=args.test_frac, seed=args.seed,
                       sampling=args.sampling, data=data)


def resolve_split(path: str | None, args, T: int, data: str = "") -> Split:
    """Load `path` if given, else rebuild from the CLI flags.  Rebuilding reproduces
    the split only when --n/--horizon/--test-frac/--test-block/--sampling/--seed match.
    """
    if path:
        sp = Split.load(path)
        sp.check_data(data, T)
        return sp
    return build_split(args, T, data)


def default_split_path(*candidates: str) -> str | None:
    """First existing split.json among `candidates` (dirs or files)."""
    for c in candidates:
        if not c:
            continue
        p = c if c.endswith(".json") else os.path.join(os.path.dirname(c) or ".",
                                                       "split.json")
        if os.path.exists(p):
            return p
    return None


def sidecar_path(gen_spec: str) -> str:
    """split.json expected to sit beside a `[LABEL=]PATH` generated-windows file."""
    path = gen_spec.rpartition("=")[2]
    return os.path.join(os.path.dirname(os.path.abspath(path)) or ".", "split.json")


def require_split(gen_specs, args, T: int, data: str = "",
                  allow_missing: bool = False) -> Split:
    """The one split every --gen in this run was produced under.  Errors otherwise.

    The split is deliberately not rebuilt here: every --gen must carry a split.json
    beside it and all of them must share an id, or the held-out blocks the backtest
    reads may be rows the model was fitted on.  --allow-missing-split rebuilds from
    the CLI flags and warns; it exists for samples generated before this check.
    """
    found, missing = {}, []
    for spec in gen_specs:
        sp_path = sidecar_path(spec)
        label = spec.rpartition("=")[2]
        if os.path.exists(sp_path):
            sp = Split.load(sp_path)
            found[sp.id()] = (sp, sp_path)
        else:
            missing.append((label, sp_path))

    if len(found) > 1:
        lines = "\n".join(f"    {i}  {p}" for i, (_, p) in found.items())
        raise SystemExit(
            "the --gen samples were produced under DIFFERENT train/test splits:\n"
            f"{lines}\n"
            "Comparing them would score each model against the other's training\n"
            "windows.  Retrain the baselines with --split pointing at the split.json\n"
            "of the run you want to compare against.")

    if missing and not found:
        if not allow_missing:
            raise SystemExit(
                "no split.json beside " + ", ".join(p for _, p in missing) + ".\n"
                "The split defines which rows the model was trained on, so it cannot\n"
                "be guessed here: rebuilding it from the CLI flags risks evaluating\n"
                "the model on its own training data.  Rerun train_tailfm.py (it now\n"
                "writes split.json), or pass --allow-missing-split to rebuild it from\n"
                "--n/--horizon/--test-frac/--test-block/--sampling/--seed.")
        sp = build_split(args, T, data)
        print(f"WARNING: no split.json found; rebuilt split[{sp.id()}] from the CLI "
              f"flags.\n         If these do not match the training run, the numbers "
              f"below are computed\n         on data the model was trained on.")
        return sp

    if missing:
        raise SystemExit(
            "some --gen samples carry a split.json and some do not:\n"
            + "\n".join(f"    missing: {p}" for _, p in missing) +
            "\nAll models in one comparison must be scored on the same split.")

    sp, sp_path = next(iter(found.values()))
    sp.check_data(data, T)
    print(f"split: {sp_path}  (id {sp.id()}, shared by all {len(gen_specs)} "
          f"--gen sample{'s' if len(gen_specs) > 1 else ''})")
    return sp


def add_split_source_args(ap) -> None:
    """Flags for scripts that consume a split rather than create one."""
    ap.add_argument("--split", type=str, default=None,
                    help="split.json to use (default: the one beside each --gen; "
                         "all --gen samples must agree)")
    ap.add_argument("--allow-missing-split", action="store_true",
                    help="rebuild the split from the CLI flags when no split.json "
                         "is found, instead of refusing.  Risks scoring the model "
                         "on its own training data -- see splits.require_split")
