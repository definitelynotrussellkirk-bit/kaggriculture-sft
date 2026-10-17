#!/usr/bin/env python3
"""CPU checks of the drafted patch (no training): archive_day, lenient encode on a marwar22 episode, worth tasks."""
import importlib.util, json, sys, zipfile, collections
from pathlib import Path
R = Path("/mnt/ai/scratch/farmNEAT/primitive-bc-v1-scale-20260926")
sys.path[:0] = [str(R / "work/runs/maintainer/primitive-bc"), str(R / "src")]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("data_patched", HERE / "data.py")
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
_sel = d.sealed_ids(R / "work/datasets/public-episodes/daily/SELECTION.json"); d.sealed_ids = lambda path=None: _sel
d.WORTH_FILE = HERE.parent / "WORTH.tsv.gz"
out = {}
out["archive_day"] = {n: d.archive_day("/x/" + n) for n in (
    "kaggriculture-episodes-2026-09-24.zip", "kaggriculture-topteams-2026-09-21-20260925T060723.zip",
    "kaggriculture-top30local-2026-09-22-20260925T230418.zip", "kaggriculture-topteams-unknown-day-20260925T041248.zip",
    "kaggriculture-episodes-2026-09-23-topteam-20260925T041043.zip")}
arch = "/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/top30-losses/archives/kaggriculture-topteams-2026-09-24-20260926T011951.zip"
rep = json.loads(zipfile.ZipFile(arch).read("113054243.json"))
seat = 0 if rep["rewards"][0] > rep["rewards"][1] else 1
ex = list(d.episode_examples(rep, seat, meta_base={"episode_id": 113054243, "seat": seat}))
out["lenient_113054243"] = {"winner": rep["info"]["TeamNames"][seat], "examples": len(ex),
                            "inexact_steps": sum(1 for e in ex if not e["exact"])}
for theta in (0.05, 0.5, 0.9):
    rows = d.load_worth(theta)
    mass = collections.Counter()
    for a, m, day, e, s, w in rows:
        pass
    tot = sum(r[5] for r in rows)
    top = max(r[5] for r in rows)
    tasks = d._worth_tasks(rows, "train", 12345, None)
    val = d._worth_tasks(rows, "val", 12345, None)
    out[f"theta_{theta}"] = {"rows_zip": len(rows), "sum_w": round(tot, 1), "expected_train_tasks_per_epoch":
                             round(sum(r[5] / top for r in rows if d.in_split(r[3], "train")), 1),
                             "train_tasks_epoch_seed12345": len(tasks), "val_tasks": len(val)}
rows = d.load_worth(0.5)
task = next(t for t in d._worth_tasks(rows, "train", 1, None))
wins = list(d._episode_task(task, rank_max=None, seats="both", leaderboard={}, windows={"window": 24, "burn_in": 0, "stride": None}))
out["episode_task_windows"] = {"task": [str(x) for x in task], "window_lists": len(wins), "windows": sum(len(w) for w in wins)}
print(json.dumps(out, indent=1, ensure_ascii=False))
(HERE / "TEST-WORTH-PATCH.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
