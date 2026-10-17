"""Header-only population counts for the continuation phases -> PHASE-COUNTS.json.

Each population is data.select_seats (the training stream's own seat rule) applied to
every episode header of the given archives; recent_days uses data.recent_day_set.
An episode counts only when its header statuses are ["DONE", "DONE"] (data.check_replay
rejects the rest while streaming). Windows = perspectives x ceil(719 / window) with
burn_in 0 and stride = window, as fit-675 (window 24 -> 30 windows per perspective).
Replay bodies are not read, so a complete-looking episode that check_replay would still
reject (step count, configuration) is counted.

    python -B phase_counts.py --archive-glob '/mnt/.../kaggriculture-episodes-2026-09-*.zip' \
        --workers 4 --out ../../../../PHASE-COUNTS.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import data

# name -> (select_seats kwargs, recent_days); rank_max 200 is fit-675's setting
POPULATIONS = {
    "all_seats_rank200": (dict(rank_max=200, wins_only=False, min_decile=0), 0),
    "all_seats_any_rank": (dict(rank_max=None, wins_only=False, min_decile=0), 0),
    "wins_rank200": (dict(rank_max=200, wins_only=True, min_decile=0), 0),
    "recent5_wins_rank200": (dict(rank_max=200, wins_only=True, min_decile=0), 5),
    "recent5_top_wins_decile8_rank200": (dict(rank_max=200, wins_only=True, min_decile=8), 5),
    "recent5_top_wins_decile8_any_rank": (dict(rank_max=None, wins_only=True, min_decile=8), 5),
    "recent5_top_wins_rank50": (dict(rank_max=50, wins_only=True, min_decile=0), 5),
}


def _header(task):
    archive, member, day, eid = task
    try:
        return task, data.read_header(archive, member), None
    except Exception as error:  # reported, never silently dropped
        return task, None, repr(error)


def count(archives, *, workers=4, window=24, batch_sizes=(16, 64)):
    lb = data.load_leaderboard()
    st = data.load_strength()
    tasks = data.list_members(archives, split="all")
    with mp.get_context("fork").Pool(workers) as pool:
        rows = pool.map(_header, tasks, chunksize=32)
    errors = [(t[3], e) for t, _, e in rows if e]
    complete = [(t, h) for t, h, e in rows if h is not None and h.get("statuses") == ["DONE", "DONE"]]
    per_persp = math.ceil(data.STEPS / window)
    out = {"archives": [str(a) for a in archives], "episodes_listed": len(tasks), "header_errors": errors,
           "complete_episodes": len(complete), "windows_per_perspective": per_persp,
           "strength_table": str(data.STRENGTH_PATH), "leaderboard": str(data.LEADERBOARD),
           "populations": {}}
    for name, (kw, recent) in POPULATIONS.items():
        days = data.recent_day_set([t for t, _ in complete], recent)
        res = {"filters": {**kw, "recent_days": recent, "days": sorted(days)}, "by_split": {}}
        for split in ("train", "val", "test"):
            n_p, eps = 0, 0
            for t, h in complete:
                if t[2] not in days or not data.in_split(t[3], split):
                    continue
                _, _, chosen = data.select_seats(h, t[2], seats="both", leaderboard=lb, strength=st, **kw)
                n_p += len(chosen)
                eps += bool(chosen)
            res["by_split"][split] = {"episodes": eps, "perspectives": n_p, "windows": n_p * per_persp,
                                      "examples": n_p * data.STEPS}
        tr = res["by_split"]["train"]["windows"]
        res["train_updates_per_epoch"] = {str(b): math.ceil(tr / b) for b in batch_sizes}
        out["populations"][name] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive-glob", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.workers > 4:
        raise SystemExit("at most 4 worker processes")
    archives = sorted(glob.glob(a.archive_glob))
    rep = count(archives, workers=a.workers)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    summary = {k: {**v["by_split"]["train"], "updates_per_epoch": v["train_updates_per_epoch"]}
               for k, v in rep["populations"].items()}
    print(json.dumps({"complete_episodes": rep["complete_episodes"], "header_errors": len(rep["header_errors"]),
                      "train": summary}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
