"""Streaming per-step primitive-BC examples from raw public Kaggle replays.

Pairing (bc.from_replay_step): the actor view at boundary t of seat s
(bc.public_observation -> actor-visible whitelist) is labelled with the action
that seat submitted after observing it, replay["steps"][t+1][s]["action"],
for t = 0..718. Both seats of an episode are separate perspectives.

Split: whole episodes, deterministic sha256 bucket of the episode id
(train 90% / val 5% / test 5%). The 16 sealed confirmation episodes
(work/datasets/public-episodes/daily/SELECTION.json, split == "confirmation")
are dropped by member name before any byte of them is read.

Rank filter: a perspective is kept when its team's rank on the public
leaderboard CSV is <= rank_max (rank_max=None keeps every perspective,
including teams missing from the leaderboard).

Inputs: observation.pack(actor_obs, None, cfg) rows [N,80] stored as float16,
token_mask [N] bool, knownness [N,80] bool; unit_pos [MAX_UNITS,2] (x, y),
-1 for absent units, farmer first then hands in observation order.
Labels: codec.encode_step (unit_tok, unit_qty, order_tok, order_qty, exact).
features() is the single observation->input function; agent.py must use it so
training and inference see identical float16-rounded inputs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import io
import json
import multiprocessing as mp
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import codec  # noqa: E402
import flip  # noqa: E402
import calc_features  # noqa: E402
from farm_neat.policy import observation as features_mod  # noqa: E402
from farm_neat.training import bc  # noqa: E402

ARCHIVE_GLOB = "/mnt/steam/farm-data/kaggle/kaggriculture-episodes-2026-09-*/kaggriculture-episodes-2026-09-*.zip"
LEADERBOARD = Path("/mnt/ai/scratch/farmNEAT/cohort-scale-20260923/leaderboard/kaggriculture.zip")
SELECTION = ROOT / "work/datasets/public-episodes/daily/SELECTION.json"
SPLIT_SALT = b"primitive-bc-split-v1:"
SPLITS = {"train": (0, 900), "val": (900, 950), "test": (950, 1000),
          "heldout": (900, 1000), "all": (0, 1000)}
LAST_STEP = 718          # last boundary with a submitted successor action
STEPS = LAST_STEP + 1    # examples per perspective in a complete episode
N_ORDER_SLOTS = codec.MAX_ORDERS + 1


# ---------------------------------------------------------------- sources
def default_archives():
    return sorted(glob.glob(ARCHIVE_GLOB))


def sealed_ids(path=SELECTION):
    """IDs only; the sealed replays themselves are never opened."""
    entries = json.loads(Path(path).read_text())["episodes"]
    sealed = {int(e["episode_id"]) for e in entries if e["split"] == "confirmation"}
    if len(sealed) != 16:
        raise ValueError("expected the 16 sealed confirmation episode ids")
    return frozenset(sealed)


def split_bucket(episode_id):
    digest = hashlib.sha256(SPLIT_SALT + str(int(episode_id)).encode()).hexdigest()
    return int(digest, 16) % 1000


def in_split(episode_id, split):
    lo, hi = SPLITS[split]
    return lo <= split_bucket(episode_id) < hi


def load_leaderboard(path=LEADERBOARD):
    """TeamName -> best (smallest) public-leaderboard rank. Accepts a CSV or the Kaggle zip."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            text = z.read(name).decode("utf-8-sig")
    else:
        text = path.read_text(encoding="utf-8-sig")
    ranks = {}
    for row in csv.DictReader(io.StringIO(text)):
        name, rank = row["TeamName"], int(row["Rank"])
        ranks[name] = min(rank, ranks.get(name, rank))
    return ranks


def archive_day(path):
    name = Path(path).name
    return name.split("episodes-")[-1][:10]


def list_members(archive_paths=None, *, split="all", sealed=None):
    """[(archive, member, day, episode_id)] in archive then member order; sealed removed by name."""
    sealed = sealed_ids() if sealed is None else sealed
    out = []
    for path in archive_paths or default_archives():
        with zipfile.ZipFile(path) as z:
            names = sorted(n for n in z.namelist() if n.endswith(".json") and n[:-5].isdigit())
        for name in names:
            eid = int(name[:-5])
            if eid in sealed or not in_split(eid, split):
                continue
            out.append((str(path), name, archive_day(path), eid))
    return out


def read_header(archive, member, limit=65536):
    """Replay fields before "steps" (info, rewards, statuses, configuration) without the steps."""
    with zipfile.ZipFile(archive) as z, z.open(member) as f:
        head = f.read(limit)
    cut = head.find(b'"steps"')
    if cut < 0:
        raise ValueError("replay header not found in the first bytes")
    return json.loads(head[:cut].rstrip().rstrip(b",") + b"}")


def seat_ranks(header, leaderboard):
    teams = header.get("info", {}).get("TeamNames") or [None, None]
    return teams, [leaderboard.get(t) if t is not None else None for t in teams]


def keep_seat(rank, rank_max):
    return rank_max is None or (rank is not None and rank <= rank_max)


def seat_list(seats):
    if seats == "both":
        return (0, 1)
    if seats in (0, 1):
        return (seats,)
    raise ValueError("seats must be 'both', 0 or 1")


# ---------------------------------------------------------------- features
def pack_config(configuration):
    cfg = {k: configuration[k] for k in features_mod.CONFIG if k in (configuration or {})}
    if configuration and "marketParams" in configuration:
        cfg["marketParams"] = {item: dict(p) for item, p in configuration["marketParams"].items()}
    return cfg


def canonical_inventories(actor_obs):
    """Carrier inventories in alphabetical item order. Saved replays store them sorted, so training always saw that
    order; applying it at play keeps the inputs identical whatever order the live observation arrives in."""
    private = actor_obs.get("private")
    if isinstance(private, dict) and isinstance(private.get("inventories"), list):
        private["inventories"] = [dict(sorted(inv.items())) if isinstance(inv, dict) else inv
                                  for inv in private["inventories"]]
    return actor_obs


CALC_WIDTH = calc_features.N_PROD + calc_features.N_GLOB     # v4: appended after flip.WIDTH (v3 models never read them)
EXTRA_WIDTH = flip.WIDTH + CALC_WIDTH


def calc_columns(actor_obs, configuration, token_ids):
    """[N, CALC_WIDTH] float16: the calc_features product row on its MARKET token (first N_PROD columns), the global row on
    the GLOBAL token (last N_GLOB columns); zeros elsewhere."""
    c = calc_features.compute(actor_obs, configuration)
    out = np.zeros((len(token_ids), CALC_WIDTH), np.float16)
    row_of = {p: i for i, p in enumerate(calc_features.PRODUCTS)}
    for n, tid in enumerate(token_ids):
        if tid == "global":
            out[n, calc_features.N_PROD:] = c["calc_glob"]
        elif tid.startswith("market/") and tid[7:] in row_of:
            out[n, :calc_features.N_PROD] = c["calc_prod"][row_of[tid[7:]]]
    return out


def features(actor_obs, configuration):
    """Model inputs from an actor-visible observation (bc.actor_observation output with step)."""
    actor_obs = canonical_inventories(actor_obs)
    packed = features_mod.pack(actor_obs, None, pack_config(configuration))
    pos = np.full((codec.MAX_UNITS, 2), -1, np.int64)
    units = codec.unit_positions(actor_obs)[:codec.MAX_UNITS]
    if units:
        pos[:len(units)] = np.asarray(units, np.int64)
    extra, unit_extra, reach_eta = flip.token_features(actor_obs, packed["token_ids"], codec.MAX_UNITS, configuration)
    extra = np.concatenate([extra.astype(np.float16), calc_columns(actor_obs, configuration, packed["token_ids"])], axis=1)
    return {"rows": packed["rows"].astype(np.float16), "token_mask": packed["token_mask"],
            "knownness": packed["knownness"], "unit_pos": pos, "extra": extra, "unit_extra": unit_extra,
            "reach_eta": reach_eta}


def check_replay(replay):
    if (replay.get("name") != "kaggriculture" or not isinstance(replay.get("steps"), list)
            or len(replay["steps"]) != 720 or replay.get("statuses") != ["DONE", "DONE"]):
        raise ValueError("complete two-seat Kaggriculture replay required")
    cfg = replay.get("configuration", {})
    if any(cfg.get(k) != v for k, v in features_mod.CONFIG.items()):
        raise ValueError("replay mechanics configuration differs from the fixed schema")


def episode_examples(replay, seat, *, meta_base, steps=None):
    """Yield one example per boundary t (default 0..718) for one perspective."""
    cfg = replay["configuration"]
    for t in (range(STEPS) if steps is None else steps):
        obs = bc.public_observation(replay, t, seat)
        action = replay["steps"][t + 1][seat].get("action")
        ex = features(obs, cfg)
        ex.update(codec.encode_step(action, obs))
        ex["meta"] = {**meta_base, "step": t}
        yield ex


def _episode_task(task, *, rank_max, seats, leaderboard, windows=None):
    """Worker body: yields lists of examples (or windows) for one episode."""
    archive, member, day, eid = task
    header = read_header(archive, member)
    teams, ranks = seat_ranks(header, leaderboard)
    chosen = [s for s in seat_list(seats) if keep_seat(ranks[s], rank_max)]
    if not chosen:
        return
    with zipfile.ZipFile(archive) as z:
        replay = json.loads(z.read(member))
    check_replay(replay)
    if replay["info"]["EpisodeId"] != eid:
        raise ValueError("member name / EpisodeId mismatch")
    for seat in chosen:
        meta = {"episode_id": eid, "seat": seat, "team": teams[seat], "rank": ranks[seat],
                "opponent_rank": ranks[1 - seat], "day": day}
        if windows is None:
            chunk = []
            for ex in episode_examples(replay, seat, meta_base=meta):
                chunk.append(ex)
                if len(chunk) == 64:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
        else:
            examples = list(episode_examples(replay, seat, meta_base=meta))
            yield list(make_windows(examples, **windows))


# ---------------------------------------------------------------- streaming
_END = "__end__"


def _worker(task_q, out_q, kwargs):
    while True:
        task = task_q.get()
        if task is None:
            out_q.put((_END, None))
            return
        try:
            for chunk in _episode_task(task, **kwargs):
                out_q.put(("data", chunk))
        except Exception as error:  # a bad replay is skipped and reported, never silently
            out_q.put(("error", (task, repr(error))))


def _stream(tasks, workers, kwargs, stats, start_method="spawn"):
    stats.setdefault("episode_errors", [])
    if workers <= 0:
        for task in tasks:
            try:
                for chunk in _episode_task(task, **kwargs):
                    yield chunk
            except Exception as error:
                stats["episode_errors"].append((task, repr(error)))
        return
    ctx = mp.get_context(start_method)
    task_q, out_q = ctx.Queue(), ctx.Queue(maxsize=max(4, 4 * workers))
    for task in tasks:
        task_q.put(task)
    for _ in range(workers):
        task_q.put(None)
    procs = [ctx.Process(target=_worker, args=(task_q, out_q, kwargs), daemon=True) for _ in range(workers)]
    for p in procs:
        p.start()
    finished = 0
    try:
        while finished < workers:
            kind, value = out_q.get()
            if kind == _END:
                finished += 1
            elif kind == "error":
                stats["episode_errors"].append(value)
                print(f"data.py: skipped episode {value[0][3]}: {value[1]}", file=sys.stderr)
            else:
                yield value
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        # An abandoned stream leaves undelivered tasks in the queue buffers; without
        # this the interpreter blocks at exit flushing them into a pipe nobody reads.
        for q in (task_q, out_q):
            q.cancel_join_thread()
            q.close()


def _tasks(archive_paths, split, shuffle_seed, max_episodes):
    tasks = list_members(archive_paths, split=split)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(tasks)
    return tasks if max_episodes is None else tasks[:max_episodes]


def iter_examples(archive_paths=None, *, split="train", rank_max=None, seats="both", workers=0,
                  shuffle_seed=None, max_episodes=None, leaderboard=None, stats=None,
                  start_method="spawn"):
    """Yield per-step examples. workers=0 runs in-process (deterministic order);
    workers>0 uses that many processes with a bounded queue (episode order then
    depends on timing; the example set does not). start_method "spawn" (default)
    is safe under a multi-threaded (torch) parent; "fork" starts faster."""
    stats = {} if stats is None else stats
    lb = load_leaderboard() if leaderboard is None else leaderboard
    tasks = _tasks(archive_paths, split, shuffle_seed, max_episodes)
    kwargs = {"rank_max": rank_max, "seats": seats, "leaderboard": lb}
    for chunk in _stream(tasks, workers, kwargs, stats, start_method):
        yield from chunk


def make_windows(examples, *, window=24, burn_in=24, stride=None):
    """Contiguous per-perspective windows over one perspective's examples (step order).

    Scored steps [s, s+window) for s = 0, stride, ...; each is preceded by up to
    burn_in no-loss steps [max(0, s-burn_in), s). A window whose burn-in starts at
    step 0 begins from the 'none' field exactly as at inference."""
    stride = window if stride is None else stride
    if window < 1 or burn_in < 0 or stride < 1:
        raise ValueError("positive window/stride and non-negative burn_in required")
    steps = [ex["meta"]["step"] for ex in examples]
    if steps != list(range(steps[0], steps[0] + len(steps))):
        raise ValueError("examples must be one perspective's contiguous steps")
    for start in range(0, len(examples), stride):
        lo = max(0, start - burn_in)
        part = examples[lo:start + window]
        yield {"steps": part, "n_burn": start - lo, "n_scored": len(part) - (start - lo),
               "start_step": examples[start]["meta"]["step"],
               "from_episode_start": examples[lo]["meta"]["step"] == 0,
               "meta": {k: v for k, v in examples[0]["meta"].items() if k != "step"}}


def iter_windows(archive_paths=None, *, split="train", rank_max=None, seats="both", workers=0,
                 window=24, burn_in=24, stride=None, shuffle_seed=None, max_episodes=None,
                 leaderboard=None, stats=None, start_method="spawn"):
    """Yield contiguous windows (see make_windows) of one perspective each."""
    stats = {} if stats is None else stats
    lb = load_leaderboard() if leaderboard is None else leaderboard
    tasks = _tasks(archive_paths, split, shuffle_seed, max_episodes)
    kwargs = {"rank_max": rank_max, "seats": seats, "leaderboard": lb,
              "windows": {"window": window, "burn_in": burn_in, "stride": stride}}
    for chunk in _stream(tasks, workers, kwargs, stats, start_method):
        yield from chunk


# ---------------------------------------------------------------- collate
LABEL_KEYS = ("unit_tok", "unit_qty", "order_tok", "order_qty")


def _pad_orders(values):
    out = np.full(N_ORDER_SLOTS, -1, np.int64)
    out[:len(values)] = values
    return out


def collate(batch):
    """Pad entities to the batch's longest roster; torch tensors, rows float32."""
    import torch
    n = max(len(ex["rows"]) for ex in batch)
    b = len(batch)
    rows = np.zeros((b, n, features_mod.WIDTH), np.float32)
    known = np.zeros((b, n, features_mod.WIDTH), np.bool_)
    mask = np.zeros((b, n), np.bool_)
    for i, ex in enumerate(batch):
        k = len(ex["rows"])
        rows[i, :k], known[i, :k], mask[i, :k] = ex["rows"], ex["knownness"], ex["token_mask"]
    return {"rows": torch.from_numpy(rows), "knownness": torch.from_numpy(known),
            "token_mask": torch.from_numpy(mask),
            "unit_pos": torch.from_numpy(np.stack([ex["unit_pos"] for ex in batch])),
            "unit_tok": torch.tensor([ex["unit_tok"] for ex in batch], dtype=torch.long),
            "unit_qty": torch.tensor([ex["unit_qty"] for ex in batch], dtype=torch.long),
            "order_tok": torch.from_numpy(np.stack([_pad_orders(ex["order_tok"]) for ex in batch])),
            "order_qty": torch.from_numpy(np.stack([_pad_orders(ex["order_qty"]) for ex in batch])),
            "exact": torch.tensor([bool(ex["exact"]) for ex in batch]),
            "meta": [ex["meta"] for ex in batch]}


def collate_windows(windows):
    """[B, T] step layout with T = burn_in + window: scored steps always start at
    column burn_in; shorter burn-ins are left-padded, short tails right-padded.
    step_valid marks real steps, loss_mask scored steps, first_valid the column
    where the carried field starts from 'none'."""
    import torch
    burn = max(w["n_burn"] for w in windows)
    width = max(w["n_scored"] for w in windows)
    t_len = burn + width
    flat, where = [], []
    for i, w in enumerate(windows):
        for j, ex in enumerate(w["steps"]):
            flat.append(ex)
            where.append((i, burn - w["n_burn"] + j))
    steps = collate(flat)
    b = len(windows)
    n = steps["rows"].shape[1]
    out = {"rows": torch.zeros(b, t_len, n, features_mod.WIDTH),
           "knownness": torch.zeros(b, t_len, n, features_mod.WIDTH, dtype=torch.bool),
           "token_mask": torch.zeros(b, t_len, n, dtype=torch.bool),
           "unit_pos": torch.full((b, t_len, codec.MAX_UNITS, 2), -1, dtype=torch.long),
           "unit_tok": torch.full((b, t_len, codec.MAX_UNITS), -1, dtype=torch.long),
           "unit_qty": torch.full((b, t_len, codec.MAX_UNITS), -1, dtype=torch.long),
           "order_tok": torch.full((b, t_len, N_ORDER_SLOTS), -1, dtype=torch.long),
           "order_qty": torch.full((b, t_len, N_ORDER_SLOTS), -1, dtype=torch.long),
           "exact": torch.zeros(b, t_len, dtype=torch.bool),
           "step_valid": torch.zeros(b, t_len, dtype=torch.bool),
           "loss_mask": torch.zeros(b, t_len, dtype=torch.bool),
           "first_valid": torch.tensor([burn - w["n_burn"] for w in windows]),
           "from_episode_start": torch.tensor([w["from_episode_start"] for w in windows]),
           "step": torch.full((b, t_len), -1, dtype=torch.long),
           "meta": [w["meta"] for w in windows]}
    ii = torch.tensor([i for i, _ in where])
    cc = torch.tensor([col for _, col in where])
    for key in ("rows", "knownness", "token_mask", "unit_pos", "unit_tok", "unit_qty",
                "order_tok", "order_qty", "exact"):
        out[key][ii, cc] = steps[key]
    out["step_valid"][ii, cc] = True
    out["loss_mask"][ii, cc] = cc >= burn
    out["step"][ii, cc] = torch.tensor([ex["meta"]["step"] for ex in flat])
    return out


# ---------------------------------------------------------------- reports
def _header_task(task):
    archive, member, day, eid = task
    try:
        h = read_header(archive, member)
        return {"episode_id": eid, "day": day, "teams": h.get("info", {}).get("TeamNames"),
                "statuses": h.get("statuses"), "complete": h.get("statuses") == ["DONE", "DONE"]}
    except Exception as error:
        return {"episode_id": eid, "day": day, "error": repr(error)}


def count_examples(archive_paths=None, *, thresholds=(8, 50, 200, None), workers=8, leaderboard=None):
    """Header-only count of available perspectives/examples per rank threshold and split.
    Each complete (DONE, DONE) perspective contributes 719 labelled steps."""
    lb = load_leaderboard() if leaderboard is None else leaderboard
    tasks = list_members(archive_paths, split="all")
    with mp.get_context("fork").Pool(workers) as pool:
        headers = pool.map(_header_task, tasks, chunksize=16)
    out = {"episodes_listed": len(tasks), "header_errors": sum("error" in h for h in headers),
           "incomplete_episodes": sum(1 for h in headers if not h.get("complete")),
           "sealed_excluded": sorted(sealed_ids()), "thresholds": {}}
    teams_seen, unmatched = set(), set()
    for h in headers:
        for t in h.get("teams") or []:
            teams_seen.add(t)
            if t not in lb:
                unmatched.add(t)
    out["distinct_teams"], out["teams_not_on_leaderboard"] = len(teams_seen), len(unmatched)
    for thr in thresholds:
        key = "all" if thr is None else f"top{thr}"
        per_split = {s: {"perspectives": 0, "examples": 0, "episodes": 0} for s in ("train", "val", "test")}
        per_day = {}
        for h in headers:
            if not h.get("complete"):
                continue
            ranks = [lb.get(t) for t in h["teams"]]
            k = sum(keep_seat(r, thr) for r in ranks)
            if not k:
                continue
            split = next(s for s in ("train", "val", "test") if in_split(h["episode_id"], s))
            per_split[split]["perspectives"] += k
            per_split[split]["examples"] += k * STEPS
            per_split[split]["episodes"] += 1
            per_day[h["day"]] = per_day.get(h["day"], 0) + k * STEPS
        total = {f: sum(v[f] for v in per_split.values()) for f in ("perspectives", "examples", "episodes")}
        out["thresholds"][key] = {"total": total, "by_split": per_split, "examples_by_day": per_day}
    return out


def benchmark(archive_paths=None, *, workers=(1, 8), examples=4000, split="train", rank_max=None,
              start_method="spawn"):
    """Examples/s from a cold start (includes worker start and replay parse)."""
    result = {}
    for w in workers:
        start, n = time.perf_counter(), 0
        for _ in iter_examples(archive_paths, split=split, rank_max=rank_max, workers=w,
                               shuffle_seed=0, start_method=start_method):
            n += 1
            if n >= examples:
                break
        seconds = time.perf_counter() - start
        result[str(w)] = {"examples": n, "seconds": round(seconds, 2),
                          "examples_per_s": round(n / seconds, 1)}
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--out")
    b = sub.add_parser("bench")
    b.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    b.add_argument("--examples", type=int, default=4000)
    b.add_argument("--rank-max", type=int)
    b.add_argument("--start-method", default="spawn", choices=("spawn", "fork", "forkserver"))
    b.add_argument("--out")
    args = ap.parse_args(argv)
    if args.workers and max(args.workers if isinstance(args.workers, list) else [args.workers]) > 8:
        print("note: more than 8 processes requested", file=sys.stderr)
    if args.cmd == "count":
        report = count_examples(workers=args.workers)
    else:
        report = benchmark(workers=tuple(args.workers), examples=args.examples, rank_max=args.rank_max,
                           start_method=args.start_method)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
