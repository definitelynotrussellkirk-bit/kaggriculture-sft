"""Held-out loss of the paired ablation arms on ONE fixed set of windows (the pilots' 256 val windows of the v5s rows), built
with the v6 arrays so v6 models read their previous turns; the cash and job-ETA arrays are always present. Eval mode: the aux
heads add nothing to the loss, so every model is scored on the same BC terms.

  run from the v5s replica root (its train_config names PRIORITY-V5S.tsv relative to it):
  python eval_arms.py OUT.json --data-from RESULTS/v5w-pA CKPT [CKPT ...] [--windows 256] [--device cuda]
"""
import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

import model as M  # noqa: E402
import train as T  # noqa: E402


BALANCED_ROWS = HERE / "PRIORITY-V6-NO4Q.tsv"


def balanced_batches(n_episodes, seed=20260927, per_batch=16):
    """PRO-RT-DATA-0927 finding 1: held_out_batches scores only game days 0/6/12/18/24. Here: distinct val-split episodes of the
    v6 rows, one 24-step window each (24-step burn-in when available), start day cycling 0..29, so every game day is scored."""
    import data
    rows = [l.rstrip("\n").split("\t") for l in open(BALANCED_ROWS)][1:]
    sealed = data.sealed_ids()
    seen, pool = set(), []
    for r in rows:
        eid = int(r[3])
        if eid in seen or eid in sealed or not data.in_split(eid, "val"):
            continue
        seen.add(eid)
        pool.append(r)
    random.Random(seed).shuffle(pool)
    windows, used, rejected = [], [], []
    if len(pool) < n_episodes:
        print(f"balanced: only {len(pool)} eligible val episodes for {n_episodes} requested", flush=True)
    # PRO-EXPERIMENTS-R2-0927 #5: fill every day's quota; a rejected candidate hands its day to the next candidate (with no
    # rejection this is exactly the old assignment day = i % 30 over the first n_episodes candidates)
    need = [n_episodes // 30 + (1 if d < n_episodes % 30 else 0) for d in range(30)]
    d_next = 0
    for r in pool:
        if not any(need):
            break
        while need[d_next % 30] == 0:
            d_next += 1
        day = d_next % 30
        start = 24 * day
        replay = json.loads(zipfile.ZipFile(r[0]).read(r[1]))
        try:   # PRO-RT-DATA-R2-0927 #1: the same identity/integrity checks as the training loader; a failure is recorded
            data.check_replay(replay)
            if replay["info"]["EpisodeId"] != int(r[3]):
                raise ValueError("member name / EpisodeId mismatch")
        except (KeyError, ValueError) as e:
            rejected.append((int(r[3]), day, str(e)[:80]))
            continue
        seat = int(r[4])
        meta = {"episode_id": int(r[3]), "seat": seat, "team": None, "rank": None, "opponent_rank": None, "day": r[2]}
        lo = max(0, start - 24)
        steps = range(lo, min(start + 24, len(replay["steps"]) - 1))
        ex = list(data.episode_examples(replay, seat, meta_base=meta, steps=steps))
        data.add_v6(ex, data.opponent_unit_tokens(replay, seat, [e["meta"]["step"] for e in ex]))
        burn = start - lo
        ws = [w for w in data.make_windows(ex, window=24, burn_in=burn) if w["steps"][w["n_burn"]]["meta"]["step"] == start]
        if ws:
            windows.append(ws[0])
            used.append((int(r[3]), seat, day))
            need[day] -= 1
            d_next += 1
        else:
            rejected.append((int(r[3]), day, "no window at the requested start"))
    if rejected:
        print(f"balanced: {len(rejected)} candidate(s) not scored: {rejected[:10]}", flush=True)
    if any(need):
        print(f"balanced: pool exhausted, days still short: {[d for d in range(30) if need[d]]}", flush=True)
    batches = [data.collate_windows(windows[i:i + per_batch]) for i in range(0, len(windows), per_batch)]
    balanced_batches.rejected = rejected
    return batches, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--data-from", required=True)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--balanced", type=int, default=300, help="day-balanced distinct val episodes (0 = off)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    dev = torch.device(a.device)
    blob, _ = T.load_checkpoint(a.data_from)
    cfg = T.TrainConfig.from_dict(blob["train_config"])
    cfg.workers, cfg.bf16, cfg.v6_data = 0, False, True
    net0 = T.build(M.ModelConfig.from_dict(blob["config"]), "cpu")
    batches = list(T.held_out_batches(net0, cfg, windows=a.windows))
    bal, used = balanced_batches(a.balanced) if a.balanced else ([], [])
    out = {"data_from": a.data_from, "windows": a.windows, "split": cfg.eval_split, "v6_arrays": True,
           "balanced": {"episodes": len(used), "requested": a.balanced, "days": sorted({d for _, _, d in used}),
                        "per_day": {d: sum(1 for _, _, x in used if x == d) for d in range(30)},
                        "episode_ids": [e for e, _, _ in used], "not_scored": getattr(balanced_batches, "rejected", []),
                        "rows": str(BALANCED_ROWS)}, "results": [], "status": "running",
           "expected_checkpoints": [str(c) for c in a.ckpts]}
    for c in a.ckpts:
        b, path = T.load_checkpoint(c)
        net = T.build(M.ModelConfig.from_dict(b["config"]), str(dev))
        net.load_state_dict(b["model"])
        rep = T.evaluate(net, batches, cfg, dev, free_run=False)
        rb = T.evaluate(net, bal, cfg, dev, free_run=False) if bal else {}
        mc = b["config"]
        row = {"checkpoint": str(path), "update": b.get("update"), "loss_per_step": rep.get("loss_per_step", rep.get("loss")),
               "v6": bool(mc.get("v6_prev")), "cash": bool(mc.get("prefix_cash_width")), "job_eta": bool(mc.get("job_eta_v6")),
               **{k: v for k, v in rep.items() if k.startswith("acc_") and "aux" not in k},
               "balanced_loss_per_step": rb.get("loss_per_step", rb.get("loss")) if rb else None,
               "balanced": {k: v for k, v in rb.items() if k.startswith("acc_") and "aux" not in k}}
        out["results"].append(row)
        print(json.dumps(row), flush=True)
        if len(out["results"]) == len(a.ckpts):
            out["status"] = "complete"   # PRO-EXPERIMENTS-R2-0927 #6: a partial table is never mistaken for the full one
        tmp = Path(a.out + ".tmp")
        tmp.write_text(json.dumps(out, indent=1) + "\n")
        tmp.replace(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
