#!/usr/bin/env python3
"""Paired-ablation pack (owner 2026-09-27: rent Colab A100/G4/L4, "u can attach kaggle datasets to colab jobs", "to make it train
over same data"). One directory, uploaded as a private Kaggle dataset and also used in place on NFS by local arms, so every arm
anywhere trains from the same parent on the same replay subset in the same order:

  nfs/primitive-bc-v6-20260927/...   v6 replica code (src/, primitive-bc/*.py, SELECTION.json), no results or row files
  nfs/cohort-scale-20260923/leaderboard/kaggriculture.zip   read by data.load_leaderboard
  data/replays-NNN.bin               the subset's replays, raw-copied per source archive (zip files; .bin so Kaggle keeps them)
  data/rows.tsv                      priority rows over the subset, archive paths as @PACK@/data/...
  seeds/<arm>/ckpt-*.pt + LATEST     training checkpoints the arms resume from
  run_arm.sh                         ARM UPDATES OUT_DIR [extra train.py args]

  python3 build_pack.py --out /mnt/ai/scratch/farmNEAT/pbc-colab-v6 --rows 1500
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

NFS = Path("/mnt/ai/scratch/farmNEAT")
V6 = NFS / "primitive-bc-v6-20260927"
PBC = V6 / "work/runs/maintainer/primitive-bc"
ROWS = PBC / "PRIORITY-V6-NO4Q.tsv"
LB = NFS / "cohort-scale-20260923/leaderboard/kaggriculture.zip"
HERE = Path(__file__).resolve().parent
SKIP_SUFFIX = (".tsv", ".tsv.gz", ".zip", ".pt", ".out", ".bak-with-own-rows")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_code(out):
    dst = out / "nfs" / V6.name
    shutil.copytree(V6 / "src", dst / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (dst / PBC.relative_to(V6)).mkdir(parents=True)
    for f in PBC.iterdir():
        if f.is_file() and not f.name.endswith(SKIP_SUFFIX) and not f.name.startswith("slurm-"):
            shutil.copy2(f, dst / PBC.relative_to(V6) / f.name)
    sel = V6 / "work/datasets/public-episodes/daily/SELECTION.json"
    (dst / sel.relative_to(V6)).parent.mkdir(parents=True)
    shutil.copy2(sel, dst / sel.relative_to(V6))
    lb = out / "nfs" / LB.relative_to(NFS)
    lb.parent.mkdir(parents=True)
    shutil.copy2(LB, lb)


def pick_rows(n, seed):
    sys.path[:0] = [str(V6 / "src"), str(PBC)]
    import data
    rows = [r for r in csv.reader(open(ROWS), delimiter="\t")][1:]
    sealed = data.sealed_ids()
    ok = [r for r in rows if data.in_split(int(r[3]), "train") and int(r[3]) not in sealed]
    random.Random(seed).shuffle(ok)
    return ok[:n], len(rows), len(ok)


def copy_replays(out, rows):
    by = defaultdict(set)
    for r in rows:
        by[r[0]].add(r[1])
    names = {}
    for i, (arch, members) in enumerate(sorted(by.items())):
        tmp = out / "data" / f"tmp-{i:03d}.zip"
        subprocess.run(["nice", "-n", "19", "zip", "-q", arch, "--copy", *sorted(members), "--out", str(tmp)], check=True)
        # Info-ZIP can exit 0 while omitting missing names or expanding wildcard names.
        # Priority-row members are exact names; verify the actual output before publishing.
        with zipfile.ZipFile(tmp) as copied:
            actual = set(copied.namelist())
        if actual != members:
            tmp.unlink()
            raise ValueError(f"replay copy mismatch for {arch}: missing={sorted(members - actual)}, "
                             f"unexpected={sorted(actual - members)}")
        name = f"replays-{i:03d}.bin"
        tmp.rename(out / "data" / name)
        names[arch] = name
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260927)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    (out / "data").mkdir(parents=True)
    copy_code(out)
    rows, total, eligible = pick_rows(a.rows, a.seed)
    names = copy_replays(out, rows)
    with open(out / "data" / "rows.tsv", "w") as f:
        f.write("archive\tmember\tday\teid\tseat\tweight\treason\n")
        for r in rows:
            f.write("\t".join(["@PACK@/data/" + names[r[0]]] + r[1:]) + "\n")
    shutil.copy2(HERE / "run_arm.sh", out / "run_arm.sh")
    manifest = {"built": "2026-09-27", "rows_source": str(ROWS), "rows_source_sha256": sha(ROWS), "rows_total": total,
                "rows_train_split": eligible, "rows_picked": len(rows), "pick_seed": a.seed,
                "weighted_rows": sum(int(r[5]) for r in rows), "replay_files": len(names),
                "replay_bytes": sum((out / "data" / n).stat().st_size for n in names.values()),
                "rows_tsv_sha256": sha(out / "data" / "rows.tsv")}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
