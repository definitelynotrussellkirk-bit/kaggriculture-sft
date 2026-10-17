#!/bin/bash
# External league game farm (owner 2026-09-29: laptop/Kaggle take queued games, return results; duplicates are fine).
# One array task = one game with the league's exact runner + CRN engine; output under OUT/<task>/ (game.json, replay).
#   sbatch -p cpu -c 3 --mem 3500 --array=0-N%K farm_game.sbatch PAIRS.tsv OUT
#SBATCH --job-name=league-farm
#SBATCH --time=02:00:00
set -euo pipefail
PAIRS=$1; OUT=$2
REL=/mnt/ai/scratch/farmNEAT/league/releases/20260928T150017Z
ENG=/mnt/ai/scratch/farmNEAT/league/engine/engine-20260927-d91cbd52
PY=/mnt/ai/scratch/farmNEAT/runtime-hosted-py311/venv/bin/python
if [ "$(uname -m)" = aarch64 ]; then     # DGX Spark (owner 09-29 "use it completely"): farmNEAT aarch64 site, torch 2.10 (CPU use)
  PY=/usr/bin/python3; export PYTHONPATH=/home/russ/farmneat/runtime-aarch64/site CUDA_VISIBLE_DEVICES=""
fi
line=$(awk -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n' "$PAIRS")
IFS=$'\t' read -r a_name a_arch b_name b_arch seed eid <<< "$line"
d="$OUT/$SLURM_ARRAY_TASK_ID"; mkdir -p "$d"; tmp=$(mktemp -d)
cd "$tmp"
# run-time receipt (PRO/ASTRA LOCKED-SPECS R3 P0 #1): the bytes actually played, hashed before and after the game
pre0=$(sha256sum "$a_arch" | cut -c1-64); pre1=$(sha256sum "$b_arch" | cut -c1-64); t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
nice -n 10 "$PY" -B "$REL/league_game.py" --engine "$ENG" --code "$ENG/code/primitive-bc" --seat0 "$a_arch" --seat1 "$b_arch" \
  --labels "$a_name" "$b_name" --world-seed "$seed" --sampler-seed 7100 --steps 719 --episode-id "$eid" \
  --template "$REL/replay-template.json" --out "$d" --crn-rng > "$d/game.log" 2>&1 || echo "runner exit $?" >> "$d/game.log"
echo -e "$a_name\t$b_name\t$seed\t$(hostname)" > "$d/PAIR.tsv"
post0=$(sha256sum "$a_arch" | cut -c1-64); post1=$(sha256sum "$b_arch" | cut -c1-64); t1=$(date -u +%Y-%m-%dT%H:%M:%SZ)
"$PY" - "$d" "$a_arch" "$b_arch" "$pre0" "$pre1" "$post0" "$post1" "$t0" "$t1" "$REL/league_game.py" "$ENG" <<'PYEOF' || echo "receipt failed" >> "$d/game.log"
import hashlib, json, os, sys
d, a, b, pre0, pre1, post0, post1, t0, t1, runner, eng = sys.argv[1:12]
fh = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else None
tree = hashlib.sha256()
for root, _, files in sorted(os.walk(eng)):
    for f in sorted(files):
        q = os.path.join(root, f); tree.update(os.path.relpath(q, eng).encode()); tree.update(fh(q).encode())
rec = {"schema": "farm-run-receipt/v1", "archives": [a, b], "archive_sha256_pre": [pre0, pre1], "archive_sha256_post": [post0, post1],
       "archive_bytes": [os.path.getsize(a), os.path.getsize(b)], "started_utc": t0, "finished_utc": t1,
       "runner_sha256": fh(runner), "engine_tree_sha256": tree.hexdigest(), "engine": os.path.basename(eng),
       "game_json_sha256": fh(os.path.join(d, "game.json")), "replay_sha256": fh(os.path.join(d, "replay.json.gz")),
       "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID"), "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID")}
open(os.path.join(d, "RUN-RECEIPT.json"), "w").write(json.dumps(rec, indent=1))
PYEOF
rm -rf "$tmp"
