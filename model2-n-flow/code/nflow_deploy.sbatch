#!/bin/bash
# N flow checkpoint -> EMA archive + cold verify (both seats vs PASS), node-side from the NFS repo snapshot.
#   sbatch nflow_deploy.sbatch CKPT NAME     -> /mnt/ai/scratch/farmNEAT/pbc-deploy/NAME/{ema.pt,NAME.tar.gz,VERIFY.json,HUB-DONE.json}
# (generic_watch.sh build_cmd: "sbatch --parsable /mnt/ai/scratch/farmNEAT/n-flow/code/nflow_deploy.sbatch {ckpt} {name}")
#SBATCH -J nflow-deploy
#SBATCH -p cpu,gpu
#SBATCH --exclude=gpu-skynet,gpu-spark,laptop-4060,gpu-3090
#SBATCH -c 2
#SBATCH --mem 4G
#SBATCH -t 01:00:00
#SBATCH --chdir=/tmp
#SBATCH -o /mnt/ai/scratch/farmNEAT/n-flow/logs/deploy-%j.out
set -euo pipefail
CKPT=$1; NAME=$2
RS=${NFLOW_RS:-/mnt/ai/scratch/farmNEAT/n-flow/repo-snap}; DEP=$RS/work/runs/maintainer/n-flow-deploy; CODE=/mnt/ai/scratch/farmNEAT/n-flow/code
PYT=/mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12; SITE=/mnt/ai/scratch/farmNEAT/runtime/site
HOSTED=/mnt/ai/scratch/farmNEAT/runtime-hosted-py311/venv/bin/python
D=${NFLOW_DEPLOY_ROOT:-/mnt/ai/scratch/farmNEAT/pbc-deploy}/$NAME
mkdir -p "$(dirname "$D")/.locks"; exec 8>"$(dirname "$D")/.locks/$NAME.lock"; flock -n 8 || { echo "another build holds the deploy-dir lock for $NAME" >&2; exit 2; }
[ -e "$D" ] && { echo "exists $D"; exit 2; }
mkdir -p "$D"; echo "start $(date -Is) $NAME $CKPT $(hostname)"
export CUDA_VISIBLE_DEVICES=""
PYTHONPATH=$SITE nice -n 10 $PYT -B $CODE/ema_export.py "$CKPT" "$D/ema.pt"
PYTHONPATH=$RS/src:$DEP:$SITE nice -n 10 $PYT -B $DEP/build_archive.py build --model "$D/ema.pt" --out "$D/$NAME.tar.gz" --storage fp16 --temperature 0 --no-verify
# cold verify; a harness protocol timeout from host I/O stalls (seen 04:0x-04:2x at load 34-63) is retried up to 3 times;
# every attempt is logged in VERIFY-ATTEMPTS.txt and only a complete attempt writes VERIFY.json
for i in 1 2 3; do
  rm -f "$D/VERIFY.json"
  if PYTHONPATH=$RS/src:$DEP nice -n ${NFLOW_VERIFY_NICE:-10} $HOSTED -B $DEP/build_archive.py verify --archive "$D/$NAME.tar.gz" \
       --opponents PASS --seats 0,1 --out "$D/VERIFY.json" > "$D/verify-attempt-$i.log" 2>&1; then
    echo "attempt $i ok $(date -Is) host $(hostname) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$D/VERIFY-ATTEMPTS.txt"
    cat "$D/verify-attempt-$i.log" | tail -2; break
  fi
  echo "attempt $i FAILED $(date -Is) host $(hostname) load $(cut -d' ' -f1-3 /proc/loadavg): $(tail -1 "$D/verify-attempt-$i.log")" >> "$D/VERIFY-ATTEMPTS.txt"
  [ $i = 3 ] && { echo "cold verify failed 3 times"; exit 4; }
  sleep 20
done
$HOSTED - "$D" "$NAME" "$CKPT" <<'PY'
import hashlib, json, sys
d, n, ck = sys.argv[1:]
v = json.load(open(f"{d}/VERIFY.json"))
sha = hashlib.sha256(open(f"{d}/{n}.tar.gz", "rb").read()).hexdigest()
checkpoint_sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
try:        # lineage from the checkpoint itself (review PRO NFLOW-LIVE #5): update, seed/base checkpoint, fresh init
    import torch
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    t4 = (blob.get("extra") or {}).get("base") or (blob.get("extra") or {}).get("t4_seed") or {}
    lineage = {"checkpoint_update": blob.get("update"), "fresh_init_run": blob.get("update") == 0,
               "base_checkpoint": t4.get("source"), "base_checkpoint_sha256": t4.get("source_sha256"),
               "seed_raw_from": t4.get("raw_from")}
except Exception as exc:  # noqa: BLE001
    lineage = {"lineage_error": f"{type(exc).__name__}: {exc}"[:200]}
json.dump({"name": n, "checkpoint": ck, "checkpoint_sha256": checkpoint_sha,
           "archive_sha256": sha, "complete": v.get("complete"), "fallbacks": v.get("fallbacks"),
           "sha_match": v.get("archive_sha256") == sha, "lineage": "checkpoint-bound", **lineage},
          open(f"{d}/HUB-DONE.json", "w"))
PY
echo "done $(date -Is)"
