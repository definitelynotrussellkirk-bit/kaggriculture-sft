#!/bin/bash
# Fail-closed plain-v9 builder: reviewed source pins + full cold pair + SHA-bound
# step-0 + consumer-time archive/receipt recheck. It does not upload anything.
#SBATCH -J hub-deploy
#SBATCH -p cpu,gpu
#SBATCH --exclude=gpu-skynet,gpu-spark,laptop-4060,gpu-3090,cpu-hub
#SBATCH -c 2
#SBATCH --mem 6G
#SBATCH -t 01:30:00
#SBATCH --chdir=/tmp
#SBATCH -o /mnt/ai/scratch/farmNEAT/tools-0929/logs/hub_deploy-%j.out
set -euo pipefail
CKPT=$1; NAME=$2; DEC=$3
T=/mnt/ai/scratch/farmNEAT/tools-0929
RS=$T/repo-snap
DEP=$RS/work/runs/maintainer/primitive-bc-v1-deploy-v9-decode
PYT=/mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12
SITE=/mnt/ai/scratch/farmNEAT/runtime/site
HOSTED=/mnt/ai/scratch/farmNEAT/runtime-hosted-py311/venv/bin/python
LIVE=/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927
D=/mnt/ai/scratch/farmNEAT/pbc-deploy/$NAME
GATE=$T/release_gate.py
STEP0=$T/structured_step0.py
STEP0_REPLAY=$T/step0-replay-structured.json.gz   # the 1.1 MB production step0-replay.json.gz fails structured_step0 (seat 1 first obs has no step)
SOURCE_PINS=$T/SOURCE-PINS.json

mkdir -p "$(dirname "$D")/.locks"; exec 8>"$(dirname "$D")/.locks/$NAME.lock"; flock -n 8 || { echo "another build holds the deploy-dir lock for $NAME" >&2; exit 2; }   # SHIP-PATH #11: one build per deploy dir
[ -e "$D" ] && { echo "exists $D" >&2; exit 2; }
for f in "$GATE" "$STEP0" "$STEP0_REPLAY" "$SOURCE_PINS"; do
  [ -f "$f" ] || { echo "missing gate input $f" >&2; exit 2; }
done
mkdir -p "$D"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "FAILED rc=$rc $NAME" >&2; fi' EXIT
echo "start $(date -Is) $NAME $CKPT $(hostname)"

PYTHONPATH=$LIVE/work/runs/maintainer/primitive-bc:$LIVE/src:$SITE \
  nice -n 10 "$PYT" -B "$LIVE/work/runs/maintainer/primitive-bc/ema_export.py" "$CKPT" "$D/ema.pt"
PYTHONPATH=$RS/src:$DEP:$SITE \
  nice -n 10 "$PYT" -B "$DEP/build_archive.py" build \
  --model "$D/ema.pt" --out "$D/$NAME.tar.gz" --storage fp16 \
  --temperature 0 --no-verify --decode "$DEC"
# The archive, not a mutable source tree, is the object verified below.
chmod 0444 "$D/$NAME.tar.gz"
PYTHONPATH=$RS/src:$DEP \
  nice -n 10 "$HOSTED" -B "$DEP/build_archive.py" verify \
  --archive "$D/$NAME.tar.gz" --opponents PASS --seats 0,1 --threads 1 \
  --timeout 60 --out "$D/VERIFY.json"
"$HOSTED" -B "$STEP0" "$D/$NAME.tar.gz" \
  --replay "$STEP0_REPLAY" --out "$D/STEP0.json"

# Write only a provisional marker. HUB-DONE appears only after the independent
# consumer-style gate reopens the archive and rechecks every receipt and source pin.
"$HOSTED" - "$D" "$NAME" "$CKPT" <<'PY'
import hashlib, json, os, sys
path, name, checkpoint = sys.argv[1:]
archive = f"{path}/{name}.tar.gz"
verify = json.load(open(f"{path}/VERIFY.json"))
sha = hashlib.sha256(open(archive, "rb").read()).hexdigest()
row = {
    "name": name,
    "checkpoint": checkpoint,
    "archive_sha256": sha,
    "complete": verify.get("complete"),
    "fallbacks": verify.get("fallbacks"),
    "sha_match": verify.get("archive_sha256") == sha,
    "verify_rerun": True,
}
tmp = f"{path}/HUB-CANDIDATE.json.tmp"
with open(tmp, "x") as handle:
    json.dump(row, handle, sort_keys=True)
    handle.write("\n")
os.replace(tmp, f"{path}/HUB-CANDIDATE.json")
PY

"$HOSTED" -B "$GATE" check \
  --archive "$D/$NAME.tar.gz" \
  --verify "$D/VERIFY.json" \
  --step0 "$D/STEP0.json" \
  --step0-probe "$STEP0" \
  --step0-replay "$STEP0_REPLAY" \
  --hub "$D/HUB-CANDIDATE.json" \
  --builder "$0" \
  --source-pins "$SOURCE_PINS" \
  --expected-decode "$DEC" \
  --out "$D/RELEASE-GATE.json"
mv "$D/HUB-CANDIDATE.json" "$D/HUB-DONE.json"
chmod 0444 "$D/VERIFY.json" "$D/STEP0.json" "$D/RELEASE-GATE.json" "$D/HUB-DONE.json"
echo "done $(date -Is) archive_sha=$("$HOSTED" -c 'import json,sys;print(json.load(open(sys.argv[1]))["archive_sha256"])' "$D/HUB-DONE.json")"
