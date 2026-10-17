#!/bin/bash
# One 4-hour chunk of a base-v1 model (PLAN.md): resume practice-v1/seeds/$ARM into $OUT (NFS), train for MAX_SECONDS on the
# full step cache, then export the EMA weights and score trunk + EMA on the practice VAL.
#   ARM=base1-a OUT=/mnt/ai/scratch/farmNEAT/base-v1/runs/a CHUNK=1 [END_UPDATE=N DECAY_UPDATES=2000] sbatch -w NODE chunk.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=04:30:00
#SBATCH --job-name=base-v1
#SBATCH --no-requeue
set -euo pipefail
: "${ARM:?}" "${OUT:?}" "${CHUNK:?}"
[[ "$CHUNK" =~ ^([1-9]|1[0-2])$ ]] || { echo "CHUNK must be 1..12" >&2; exit 2; }   # 9: checkpoint-boundary low-LR continuation; LR via EXTRA --lr
if [[ "$CHUNK" == 6 && ( -z "${END_UPDATE:-}" || -z "${DECAY_UPDATES:-}" ) ]]; then
  echo "final chunk requires END_UPDATE and DECAY_UPDATES (absolute completed-update horizon)" >&2; exit 2
fi
if [[ "$CHUNK" == 6 && "${END_UPDATE:-}" == auto ]]; then   # horizon from this run's mean rate (last 50 log intervals) minus 600 s startup
  END_UPDATE=$(python3 -c 'import json,sys,statistics as st
r=[json.loads(l) for l in open(sys.argv[1]) if l.startswith("{\"")]; r=[x for x in r if "update" in x and "seconds" in x][-50:]
spu=st.mean(x["seconds"]/max(1,x.get("updates_in_log",10)) for x in r)
print(int(r[-1]["update"] + (float(sys.argv[2]) - 600)/spu))' "$OUT/metrics.jsonl" "${MAX_SECONDS:-13500}")
  echo "chunk 6 horizon: END_UPDATE=$END_UPDATE" >&2
fi
END_UPDATE=${END_UPDATE:-1000000}; DECAY_UPDATES=${DECAY_UPDATES:-0}
[[ "$END_UPDATE" =~ ^[1-9][0-9]*$ && "$DECAY_UPDATES" =~ ^[0-9]+$ ]] || { echo "invalid horizon" >&2; exit 2; }
if (( CHUNK == 6 && DECAY_UPDATES == 0 )); then echo "final chunk needs positive decay" >&2; exit 2; fi
# EXTRA accepts whitespace-delimited flags, not embedded shell quoting; no eval or pathname expansion.
read -r -a extra_args <<< "${EXTRA:-}"
for arg in "${extra_args[@]}"; do
  case "$arg" in --updates|--updates=*|--decay-updates|--decay-updates=*)
    echo "set END_UPDATE/DECAY_UPDATES, not horizon flags in EXTRA" >&2; exit 2;;
  esac
done
if (( CHUNK > 1 )) && [[ ! -s "$OUT/LATEST" ]]; then
  echo "later chunk requires an existing LATEST; refusing silent restart from seed" >&2; exit 2
fi
start_latest=""; [[ ! -s "$OUT/LATEST" ]] || start_latest=$(tr -d '[:space:]' < "$OUT/LATEST")
NFS=/mnt/ai/scratch/farmNEAT; LOCAL=$HOME/farmneat-local; PACK=${PACK:-base-v1-pack}; VAL_PACK=practice-v1
rsync -a "$NFS/runtime/" "$LOCAL/fs/runtime/"; rsync -a --delete "$NFS/$PACK/" "$LOCAL/fs/$PACK/"
mkdir -p "$OUT"
# chunk-end hooks of the previous chunk may still hold memory: wait (<= 45 min) instead of letting the guard cancel at start
for i in $(seq 270); do [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge $(( ${MIN_AVAIL_GB:-8} + 6 )) ] && break; sleep 10; done
( while sleep 5; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$a" -lt "${MIN_AVAIL_GB:-8}" ]; then echo "memory guard: MemAvailable ${a} GB, cancelling job" >&2; scancel "$SLURM_JOB_ID"; fi
done ) &
guard_pid=$!
trap 'kill "$guard_pid" 2>/dev/null || true' EXIT
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True CPATH=/usr/include/python3.12
PY=$LOCAL/fs/runtime/python/bin/python3.12
# shellcheck disable=SC2086
PY=$PY EXTRA_PYTHONPATH=$LOCAL/fs/runtime/site WORK=$LOCAL/fs/pbc-work-base-v1-$ARM WORKERS=${WORKERS:-4} CKPT_EVERY=${CKPT_EVERY:-100} \
  bash "$LOCAL/fs/$PACK/run_arm.sh" "$ARM" "$END_UPDATE" "$OUT" \
  --grad-checkpoint false --compile-${COMPILE:-model} true --pack-steps ${PACK_STEPS:?set PACK_STEPS: 256 for B (21.0 GB measured), 128 for C} \
  --step-cache "${STEP_CACHE:-$NFS/base-v1-stepcache-v11}" --cache-pool-seats 6 --eval-every 0 --keep-ckpts 6 \
  --batch-windows 128 --warmup 300 --grow-lr-mult 1 --decay-updates "$DECAY_UPDATES" \
  --lr-schedule wsd --ema-decay ${EMA:-0.999} --max-seconds ${MAX_SECONDS:-13500} --mask-unit-noops true \
  "${extra_args[@]}" >> "$OUT/train.log" 2>&1
# chunk-end deliverables: keep the checkpoint, export EMA, score both on the practice VAL
last=$(tr -d '[:space:]' < "$OUT/LATEST")
[[ "$last" =~ ^ckpt-[0-9]+\.pt$ && -s "$OUT/$last" ]] || { echo "LATEST is invalid" >&2; exit 2; }
[[ "$last" != "$start_latest" ]] || { echo "training did not advance LATEST" >&2; exit 2; }
mkdir -p "$OUT/chunks"; cp "$OUT/$last" "$OUT/chunks/chunk$CHUNK-$last"
R=$LOCAL/fs/pbc-work-base-v1-$ARM/primitive-bc-v6-20260927
export PYTHONPATH=$R/work/runs/maintainer/primitive-bc:$R/src:$LOCAL/fs/runtime/site
cd $R/work/runs/maintainer/primitive-bc
$PY -B ema_export.py "$OUT/$last" "$OUT/chunks/chunk$CHUNK-ema-$last"
$PY -B $NFS/base-v1/tools/eval_practice.py "$NFS/$VAL_PACK" "$OUT/chunks/chunk$CHUNK-eval.json" \
  "$OUT/chunks/chunk$CHUNK-$last" "$OUT/chunks/chunk$CHUNK-ema-$last" >> "$OUT/train.log" 2>&1

# A successful VAL call is NOT a deployment/promotion gate. Required next: actual storage
# export -> roundtrip-policy wire parity -> cold 719-decision run -> paired native panel.
# Install source fixes in the staged practice runtime BEFORE running this job; this script
# does not patch the omitted run_arm.sh or inject files into its bundle automatically.
