#!/usr/bin/env python3
"""Per-step feature cache for stateless packed training (TrainConfig.pack_steps > 0, think != bottleneck).

Feature extraction costs ~9 ms per step in Python (0.2-0.45 s per 24-step window), which capped the trainer at ~30 windows/s
on 16 CPUs while the GPU could take ~60+. The cache runs the trainer's own per-seat pipeline (data._episode_task with 1-step
windows, then data.collate_windows) once per seat, stores every step's arrays (integers narrowed, floats exact fp32, zlib) and
serves ready [n, 1] chunks from DataLoader workers, so an update only moves pinned tensors to the GPU.

  build:  python step_cache.py build ROWS.tsv OUT_DIR [--procs 20] [--shard i/n] [--v6] [--min-final-quadrants 0]
  check:  python step_cache.py check ROWS.tsv OUT_DIR [--n 3]     (cached steps == the trainer's packed steps, exactly)
  ROWS.tsv: priority rows (archive, member, day, eid, seat, weight, ...; header line), @PACK@ already substituted.
"""
import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

VERSION = 2  # source-bound metadata required; rebuild v1 caches
TOKEN_KEYS = ("rows", "knownness", "token_mask", "extra", "v6_token", "x8_token")   # dim 1 = tokens (padded per chunk)
ONE_STEP = {"window": 1, "burn_in": 0, "stride": 1}


def _rows(path):
    with open(path) as f:
        rows = [l.rstrip("\n").split("\t") for l in f if l.strip()]
    if any(len(r) < 5 for r in rows[1:]):
        raise ValueError(f"incomplete cache rows in {path}; publish rows.tsv atomically")
    if any(int(r[5] or 1) < 0 for r in rows[1:] if len(r) > 5):
        raise ValueError("cache row weights must be nonnegative")
    return [(r[0], r[1], r[2], int(r[3]), int(r[4]), int(r[5]) if len(r) > 5 and r[5] else 1) for r in rows[1:]]


def _narrow(a):
    if a.dtype == np.bool_:
        return a
    if a.dtype.kind == "f":      # most float inputs are built as float16: store them as float16 when that is lossless
        with np.errstate(over="ignore", invalid="ignore"):
            h = a.astype(np.float16)
        return h if np.array_equal(h.astype(a.dtype), a, equal_nan=True) else a
    lo, hi = (int(a.min()), int(a.max())) if a.size else (0, 0)
    for t in (np.int8, np.int16, np.int32):
        if np.iinfo(t).min <= lo and hi <= np.iinfo(t).max:
            return a.astype(t)
    return a


def seat_steps(row, *, v6=False, min_final_quadrants=0, feature_schema="legacy", leaderboard=None, v8=False):
    """{key: np.ndarray [S, ...]} for one (episode, seat) row, exactly the trainer's per-step tensors (widest dtypes)."""
    import data
    archive, member, day, eid, seat = row[:5]
    kwargs = {"rank_max": None, "seats": "both", "leaderboard": data.load_leaderboard() if leaderboard is None else leaderboard,
              "windows": ONE_STEP, "min_final_quadrants": min_final_quadrants, "v6": v6, "feature_schema": feature_schema,
              "v8": v8}
    wins = []
    for chunk in data._episode_task((archive, member, day, int(eid), (int(seat),)), **kwargs):
        wins.extend(chunk)
    if not wins:
        return None
    b = data.collate_windows(wins)
    return {k: v[:, 0].numpy() for k, v in b.items() if torch.is_tensor(v) and v.dim() >= 2}


def encode(arrs):
    return pickle.dumps({"v": VERSION, "n": int(next(iter(arrs.values())).shape[0]),
                         "keys": {k: (str(a.dtype), str(n.dtype), n.shape, zlib.compress(np.ascontiguousarray(n).tobytes(), 1))
                                  for k, a in arrs.items() for n in (_narrow(a),)}}, protocol=5)


def decode(blob, widen=False):
    d = pickle.loads(blob)  # trusted local cache files only: pickle is not a safe interchange format
    if d.get("v") != VERSION:
        raise ValueError(f"unsupported cache version: {d.get('v')!r}; expected {VERSION}")
    out = {}
    for k, (orig, stored, shape, z) in d["keys"].items():
        a = np.frombuffer(zlib.decompress(z), dtype=stored).reshape(shape)
        out[k] = a.astype(orig) if widen else a
    return out, {k: v[0] for k, v in d["keys"].items()}


def _name(row):
    return f"{row[3]}-{row[4]}.stp"



def cache_identity(opts):
    """Source-bound packing contract. Called at cache open/build, never per step.

    Missing sibling sources are recorded as missing, not claimed to be verified.
    V8's loaded provider paths are fingerprinted separately by v8.schema().
    """
    directory = Path(__file__).resolve().parent
    files = ("data.py", "step_cache.py", "v8.py", "codec.py", "prefix.py", "job_eta.py", "job_eta_v7.py",
             "flip.py", "flip_v7.py", "calc_features.py", "calc_features_v7.py",
             "theorem_features.py", "theorem_features_v7.py", "noop_mask.py")
    sources = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
               if (directory / name).is_file() else None for name in files}
    import noop_mask
    info = {"version": VERSION, **opts, "packing_sources": sources, "noop_mask_schema": noop_mask.SCHEMA,
            "numpy_version": np.__version__, "torch_version": str(torch.__version__)}
    if opts.get("v8"):
        import v8
        info["v8_schema"] = v8.schema()
        info["v8_widths"] = list(v8.widths())
        info["v8_families"] = [m.__name__ for m in v8.available()]
    return info


def validate_metadata(path, opts=None):
    """Reject stale/partial caches at both build and reader entry points."""
    path = Path(path)
    try:
        meta = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable cache metadata {path}; rebuild in a new directory") from exc
    if meta.get("version") != VERSION:
        raise ValueError(f"cache version mismatch at {path}; rebuild in a new directory")
    keys = ("v6", "v8", "feature_schema", "min_final_quadrants")
    options = opts if opts is not None else {k: meta.get(k) for k in keys}
    expected = cache_identity(options)
    changed = [k for k, v in expected.items() if meta.get(k) != v]
    if changed:
        raise ValueError(f"cache metadata mismatch {changed} at {path}; rebuild in a new directory")
    return meta


def _build_one(args):
    row, out, opts = args
    dest = Path(out) / _name(row)
    if dest.exists():
        return row, "exists", 0.0
    t = time.time()
    import data
    data.OPP_LABEL_MASKED.clear()
    try:
        arrs = seat_steps(row, **opts)
    except Exception as e:   # a bad replay is reported, never silently dropped
        return row, f"error: {e!r}"[:300], time.time() - t
    if arrs is None:
        return row, "empty", time.time() - t
    tmp = dest.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(encode(arrs))
    tmp.replace(dest)
    masked = dict(data.OPP_LABEL_MASKED)
    return row, ("ok" if not masked else "ok opp_masked=" + json.dumps(masked, sort_keys=True)), time.time() - t


def build(a):
    import multiprocessing as mp
    rows = _rows(a.rows)
    i, n = (int(x) for x in a.shard.split("/"))
    todo = [r for j, r in enumerate(rows) if j % n == i]
    if a.reverse:            # a helper on another node works the same shard from the other end
        todo.reverse()
    if a.start:              # or starts part-way through (several helpers, little overlap)
        k = int(len(todo) * a.start)
        todo = todo[k:] + todo[:k]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    opts = {"v6": a.v6, "min_final_quadrants": a.min_final_quadrants, "feature_schema": a.feature_schema, "v8": a.v8}
    identity = cache_identity(opts)
    meta = out / "CACHE.json"
    if not meta.exists():
        if any(out.glob("*.stp")):
            raise ValueError(f"orphan cache seats in {out}; rebuild into a new directory")
        # Exclusive create prevents parallel builders from replacing another schema.
        try:
            with meta.open("x") as f:
                f.write(json.dumps({"rows": str(a.rows), **identity}, indent=1) + "\n")
        except FileExistsError:
            validate_metadata(meta, opts)
    else:
        validate_metadata(meta, opts)
    log = open(out / f"BUILD-{i}of{n}{'-rev' if a.reverse else ''}{f'-at{a.start:g}' if a.start else ''}.jsonl", "a")
    done = 0
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for row, status, dt in pool.imap_unordered(_build_one, [(r, str(out), opts) for r in todo], chunksize=1):
            done += 1
            log.write(json.dumps({"eid": row[3], "seat": row[4], "status": status, "s": round(dt, 2)}) + "\n")
            log.flush()
            if done % 50 == 0 or done == len(todo):
                print(f"{done}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


def check(a):
    """The cached steps of n rows equal the trainer's own packed steps (window 24 / burn-in 24 -> pack_batch), key by key."""
    import data
    import train as T
    rows = [r for r in _rows(a.rows) if (Path(a.out) / _name(r)).exists()][:a.n]
    if not rows:
        print("CHECK FAIL: zero cached seats checked")
        return 1
    validate_metadata(Path(a.out) / "CACHE.json", {"v6": a.v6, "v8": a.v8, "feature_schema": getattr(a, "feature_schema", "legacy"),
                                                    "min_final_quadrants": a.min_final_quadrants})
    lb = data.load_leaderboard()
    bad = 0
    for row in rows:
        cached, _ = decode((Path(a.out) / _name(row)).read_bytes(), widen=True)
        wins = []
        for chunk in data._episode_task((row[0], row[1], row[2], row[3], (row[4],)), rank_max=None, seats="both", leaderboard=lb,
                                        windows={"window": 24, "burn_in": 24, "stride": None}, v6=a.v6,
                                        min_final_quadrants=a.min_final_quadrants, v8=a.v8,
                                        feature_schema=getattr(a, "feature_schema", "legacy")):
            wins.extend(chunk)
        packed = T.pack_batch(data.collate_windows(wins), 10 ** 9)[0]
        steps = packed["step"][:, 0].numpy()
        order = np.argsort(steps)
        if not np.array_equal(steps[order], cached["step"]):
            print(json.dumps({"eid": row[3], "seat": row[4], "error": "step sets differ"})); bad += 1; continue
        for k, c in cached.items():
            p = packed[k][:, 0].numpy()[order]
            if k in TOKEN_KEYS:
                n = min(p.shape[1], c.shape[1])
                extra_p, extra_c = p[:, n:], c[:, n:]
                ok = np.array_equal(p[:, :n], c[:, :n]) and not extra_p.any() and not extra_c.any()
            else:
                ok = np.array_equal(p, c)
            if not ok:
                print(json.dumps({"eid": row[3], "seat": row[4], "key": k, "error": "mismatch"})); bad += 1
        print(json.dumps({"eid": row[3], "seat": row[4], "steps": int(len(steps)), "keys": len(cached)}), flush=True)
    print("CHECK", "PASS" if bad == 0 else f"FAIL {bad}")
    return 1 if bad else 0


def validate_for_model(cache_dir, model_cfg, train_cfg):
    """Validate declared contracts, not byte-level feature equivalence (run check separately)."""
    meta = json.loads((Path(cache_dir) / "CACHE.json").read_text())
    if meta.get("version") != VERSION:
        raise ValueError("cache metadata version does not match this reader")
    if meta.get("feature_schema", "legacy") != model_cfg.feature_schema:
        raise ValueError("cache feature_schema does not match the model")
    needs_v6 = bool(train_cfg.v6_data or model_cfg.v6_prev or model_cfg.job_eta_v6)
    if bool(meta.get("v6")) != needs_v6:
        raise ValueError("cache v6 inputs do not match the model/training configuration")
    widths = [model_cfg.x8_token_width, model_cfg.x8_unit_width, model_cfg.lattice_channels]
    if any(widths):
        if not meta.get("v8"):
            raise ValueError("v8-enabled model cannot train on a non-v8 cache")
        if meta.get("v8_widths") != widths:
            raise ValueError(f"cache v8 widths {meta.get('v8_widths')} do not match {widths}")
    return meta


class StepChunks(torch.utils.data.IterableDataset):
    """Endless [n, 1] chunks of `chunk` steps drawn at random from a rolling pool of `pool_seats` cached seats per worker;
    each worker walks its own share of the (weighted) seat list in a fresh seeded order every pass."""

    def __init__(self, cache_dir, chunk, seed, pool_seats=4, rows=None):
        self.dir, self.chunk, self.seed, self.pool_seats = Path(cache_dir), int(chunk), int(seed), int(pool_seats)
        if self.chunk <= 0 or self.pool_seats <= 0:
            raise ValueError("chunk and pool_seats must be positive")
        meta = validate_metadata(self.dir / "CACHE.json")
        self.noop_schema = meta.get("noop_mask_schema")
        self.rows_path = Path(rows or meta["rows"])
        rows = _rows(self.rows_path)
        self.all_files = [self.dir / _name(r) for r in rows for _ in range(r[5])]
        self.files = [f for f in self.all_files if f.exists()]
        if not self.files:
            raise ValueError(f"no cached seats in {cache_dir}")

    def _seats(self, wid, nw):
        rnd = 0
        while True:
            if rnd:
                # A snapshot of paths cannot discover rows appended after construction.
                rows = _rows(self.rows_path)
                self.all_files = [self.dir / _name(r) for r in rows for _ in range(r[5])]
                self.files = [f for f in self.all_files if f.exists()]
            files = list(self.files)
            random.Random(self.seed * 1_000_003 + rnd * 7919).shuffle(files)
            shard = files[wid::nw]
            if not shard:
                raise ValueError(f"cache worker {wid}/{nw} has no seats; reduce workers or finish cache build")
            yield from shard
            rnd += 1

    def __iter__(self):
        """Exactly pool_seats decoded seats live per worker (memory bound); each chunk slot draws a random live seat and
        pops one of its not-yet-drawn steps; an exhausted seat is replaced by the next one, so every step of every seat
        is drawn once per pass."""
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = random.Random(self.seed * 9_999_991 + wid)
        seats = self._seats(wid, nw)

        def load():
            arrs, dt = decode(next(seats).read_bytes())
            idx = list(range(arrs["step"].shape[0]))
            if not idx:
                raise ValueError("empty cached seat has zero steps")
            rng.shuffle(idx)
            return [arrs, idx], dt

        live, dtypes = [], None
        while len(live) < self.pool_seats:
            e, dtypes = load()
            live.append(e)
        while True:
            take = []
            for _ in range(self.chunk):
                j = rng.randrange(len(live))
                arrs, idx = live[j]
                take.append((arrs, idx.pop()))
                if not idx:
                    live[j], dtypes = load()
            yield collate_steps(take, dtypes, self.noop_schema)


def collate_steps(take, dtypes, noop_schema=None):
    if not take or any(set(s) != set(take[0][0]) for s, _ in take):
        raise ValueError("cache seat keys differ or batch is empty; rebuild a homogeneous cache")
    n = len(take)
    ntok = max(s["token_mask"].shape[1] for s, _ in take)
    out = {}
    for k in take[0][0]:
        first = take[0][0][k]
        shape = (n, 1, ntok) + first.shape[2:] if k in TOKEN_KEYS else (n, 1) + first.shape[1:]
        buf = np.zeros(shape, dtype=np.result_type(*{s[k].dtype for s, _ in take}))   # seats may narrow a key differently
        for j, (s, i) in enumerate(take):
            v = s[k][i]
            if k in TOKEN_KEYS:
                buf[j, 0, :v.shape[0]] = v
            else:
                buf[j, 0] = v
        out[k] = torch.from_numpy(buf)
    out["step_valid"] = torch.ones(n, 1, dtype=torch.bool)
    out["loss_mask"] = torch.ones(n, 1, dtype=torch.bool)
    out["first_valid"] = torch.zeros(n, dtype=torch.long)
    out["_dtypes"] = dtypes
    if noop_schema and "unit_loss_keep" in out:    # the trainer only uses a mask whose schema it recognises
        out["unit_noop_mask_schema"] = noop_schema
    return out


def loader(cache_dir, *, chunk, seed, workers, pool_seats=4):
    ds = StepChunks(cache_dir, chunk, seed, pool_seats=pool_seats)
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=True,
                                       persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
                                       multiprocessing_context="fork" if workers > 0 else None)   # workers never touch CUDA


WIDEN = {"int64": torch.long, "float32": torch.float32, "bool": torch.bool}


def to_device(chunk, device):
    """Pinned narrow chunk -> device tensors in the trainer's dtypes; step_valid / loss_mask stay on the host."""
    dt = chunk["_dtypes"]
    out = {}
    for k, v in chunk.items():
        if k == "_dtypes":
            continue
        if k in ("step_valid", "loss_mask", "first_valid") or not torch.is_tensor(v):
            out[k] = v
            continue
        out[k] = v.to(device, non_blocking=True).to(WIDEN.get(dt.get(k, "bool"), v.dtype))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "check"):
        p = sub.add_parser(name)
        p.add_argument("rows")
        p.add_argument("out")
        p.add_argument("--v6", action="store_true")
        p.add_argument("--v8", action="store_true")
        p.add_argument("--min-final-quadrants", type=int, default=0)
        p.add_argument("--feature-schema", default="legacy")
        if name == "build":
            p.add_argument("--procs", type=int, default=8)
            p.add_argument("--shard", default="0/1")
            p.add_argument("--reverse", action="store_true")
            p.add_argument("--start", type=float, default=0.0)
        else:
            p.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    return build(a) or 0 if a.cmd == "build" else check(a)


if __name__ == "__main__":
    sys.exit(main())
