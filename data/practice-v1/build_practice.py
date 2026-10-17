#!/usr/bin/env python3
"""Frozen "practice data" (owner 2026-09-28: "we need to setup our practice data", "that we use to establish all evals against",
"like -- THESE REPLAYS, etc", "and THIS VALIDATION SET", "even if it's not the best we'll eventually have, but helps us compare").

One immutable directory that every quick experiment trains on and is scored against:
  data/replays-NNN.bin + data/rows.tsv   TRAIN: winning seats rated >= 2880 (catalog view WINS-2880), train split, the recipe's
                                         recent window (default 2026-09-23..27), --train-rows sampled with a per-team cap
  val/val-NNN.bin + val/VAL.tsv          VAL: distinct val-split winning seats rated >= 2880 (default 2026-09-20..27), --val
                                         episodes; eval_practice.py scores one 24-step window each, start day = i % 30
  nfs/..., seeds/<name>/, run_arm.sh     same layout as build_pack.py, so run_arm.sh / arm-*-local.sbatch train from it
  MANIFEST.json                          sources, selection rules, per-member CRC32 + size of every copied replay

  python3 build_practice.py --out /mnt/ai/scratch/farmNEAT/practice-v1
"""
import argparse
import collections
import csv
import datetime as dt
import json
import random
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import build_pack as BP

VIEW = Path("/mnt/ai/scratch/farmNEAT/primitive-bc-data/catalog/views/WINS-2880.tsv")
SEEDS = {"pilotB-48456": Path("/mnt/ai/scratch/farmNEAT/pbc-colab-v6/seeds/control/ckpt-0048456.pt"),
         "v4b-g2-48236": Path("/mnt/ai/scratch/farmNEAT/primitive-bc-v1-v4-20260926/work/runs/maintainer/primitive-bc/results/v4b-g2/ckpt-0048236.pt")}


def days(a, b):
    d0, d1 = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    return {(d0 + dt.timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)}


def team(r):
    return r[6].split(":")[-1]


def pick(rows, n, cap_frac, seed):
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: (r[3], r[4]))
    rng.shuffle(rows)
    cap = max(1, int(n * cap_frac))
    per, out, seen = collections.Counter(), [], set()
    for r in rows:
        if len(out) >= n:
            break
        if per[team(r)] >= cap or int(r[3]) in seen:
            continue
        out.append(r)
        per[team(r)] += 1
        seen.add(int(r[3]))
    return out


def copy_group(rows, dest_dir, prefix):
    by = defaultdict(set)
    for r in rows:
        by[r[0]].add(r[1])
    names, fp = {}, {}
    for i, (arch, members) in enumerate(sorted(by.items())):
        tmp = dest_dir / f"tmp-{prefix}-{i:03d}.zip"
        subprocess.run(["nice", "-n", "19", "zip", "-q", arch, "--copy", *sorted(members), "--out", str(tmp)], check=True)
        with zipfile.ZipFile(tmp) as z:
            infos = {zi.filename: zi for zi in z.infolist()}
        if set(infos) != members:
            tmp.unlink()
            raise SystemExit(f"copy mismatch for {arch}: missing {sorted(members - set(infos))[:5]}")
        name = f"{prefix}-{i:03d}.bin"
        tmp.rename(dest_dir / name)
        names[arch] = name
        for m, zi in infos.items():
            fp[(arch, m)] = {"crc32": zi.CRC, "bytes": zi.file_size}
    return names, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-days", default="2026-09-23..2026-09-27")
    ap.add_argument("--val-days", default="2026-09-20..2026-09-27")
    ap.add_argument("--train-rows", type=int, default=3000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--team-cap", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=20260928)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists (practice data is immutable; build a new version)")
    sys.path[:0] = [str(BP.V6 / "src"), str(BP.PBC)]
    import data
    rows = [r for r in csv.reader(open(VIEW), delimiter="\t")][1:]
    sealed = data.sealed_ids()
    rows = [r for r in rows if int(r[3]) not in sealed]
    tdays, vdays = days(*a.train_days.split("..")), days(*a.val_days.split(".."))
    train_pool = [r for r in rows if r[2] in tdays and data.in_split(int(r[3]), "train")]
    val_pool = [r for r in rows if r[2] in vdays and data.in_split(int(r[3]), "val")]
    train = pick(train_pool, a.train_rows, a.team_cap, a.seed)
    val = pick(val_pool, a.val, 1.0, a.seed + 1)
    if len(train) < a.train_rows or len(val) < a.val:
        raise SystemExit(f"pool too small: train {len(train)}/{a.train_rows}, val {len(val)}/{a.val}")
    (out / "data").mkdir(parents=True)
    (out / "val").mkdir()
    BP.copy_code(out)
    tnames, tfp = copy_group(train, out / "data", "replays")
    vnames, vfp = copy_group(val, out / "val", "val")
    hdr = "archive\tmember\tday\teid\tseat\tweight\treason\n"
    with open(out / "data" / "rows.tsv", "w") as f:
        f.write(hdr)
        for r in train:
            f.write("\t".join(["@PACK@/data/" + tnames[r[0]], r[1], r[2], r[3], r[4], "1", r[6]]) + "\n")
    with open(out / "val" / "VAL.tsv", "w") as f:
        f.write(hdr)
        for r in val:
            f.write("\t".join(["@PACK@/val/" + vnames[r[0]], r[1], r[2], r[3], r[4], "1", r[6]]) + "\n")
    for name, src in SEEDS.items():
        (out / "seeds" / name).mkdir(parents=True)
        shutil.copy2(src, out / "seeds" / name / src.name)
        (out / "seeds" / name / "LATEST").write_text(src.name + "\n")
    shutil.copy2(BP.HERE / "run_arm.sh", out / "run_arm.sh")
    manifest = {"built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "view": str(VIEW),
                "view_sha256": BP.sha(VIEW), "train_days": a.train_days, "val_days": a.val_days, "seed": a.seed,
                "team_cap": a.team_cap, "train_pool": len(train_pool), "val_pool": len(val_pool),
                "train_rows": len(train), "val_episodes": len(val),
                "train_by_day": dict(collections.Counter(r[2] for r in train)),
                "train_top_teams": collections.Counter(team(r) for r in train).most_common(10),
                "val_by_day": dict(collections.Counter(r[2] for r in val)),
                "seeds": {k: {"path": str(v), "sha256": BP.sha(v)} for k, v in SEEDS.items()},
                "rows_tsv_sha256": BP.sha(out / "data" / "rows.tsv"), "val_tsv_sha256": BP.sha(out / "val" / "VAL.tsv"),
                "members": {f"{tnames.get(k[0]) or vnames.get(k[0])}:{k[1]}": v for k, v in {**tfp, **vfp}.items()}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "members"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
