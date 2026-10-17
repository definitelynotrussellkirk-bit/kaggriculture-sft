#!/bin/bash
# One continuation phase (or segment) of a primitive-BC fit on gpu-4090b. launch-train.sbatch plus
# seeding from a previous OUT and an absolute stop update. Not submitted by its author.
#
#   RUN_ROOT      repository replica on shared scratch (as launch-train.sbatch)
#   ARCHIVES      space-separated explicit archive list (preferred: pins the day set, so recent_days
#                 cannot move when a new daily zip lands); else
#   ARCHIVE_GLOB  daily replay archives visible from gpu-4090b
#   OUT           this phase's output directory; an existing OUT/LATEST resumes it
#   SEED_FROM     used only when OUT/LATEST is absent: the checkpoint named by SEED_FROM/LATEST is
#                 copied into OUT and becomes OUT/LATEST (the previous phase's OUT, or the seed dir)
#   PHASE_END     absolute update at which this job stops with a checkpoint; a rerun of the same job
#                 after an interruption resumes OUT/LATEST and still stops at PHASE_END
#   WORKERS       data.py worker processes (keep <= cpus-per-task - 2)
#   TRAIN_ARGS    extra train.py fit arguments (data flags, --updates = schedule total)
#   DEVICE        cuda (default); cpu only for a local dry run of this script
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-4090b
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=farmneat-pbc-v1-cont
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CUDA_VISIBLE_DEVICES:?}" "${RUN_ROOT:?}" "${OUT:?}" "${PHASE_END:?}"

run_root=$RUN_ROOT
archive_glob=${ARCHIVE_GLOB:-/mnt/ai/scratch/farmNEAT/primitive-bc-data/kaggriculture-episodes-2026-09-*.zip}
here="$run_root/work/runs/maintainer/primitive-bc"
out=$OUT
workers=${WORKERS:-12}
device=${DEVICE:-cuda}

cd "$run_root"
test -x work/venv/python
test -f RUN.py
test -f "$here/train.py" && test -f "$here/model.py" && test -f "$here/data.py" && test -f "$here/codec.py"
test -f "$here/strength.py" && test -f "$here/STRENGTH.json"
test -f work/datasets/public-episodes/daily/SELECTION.json
# ARCHIVES_FILE: read the archive list at job start (primitive-bc-data/TRAIN-ARCHIVES.txt grows as daily uploads are ingested),
# so every phase start and resume trains on the corpus as of that moment.
if [ -n "${ARCHIVES_FILE:-}" ]; then
  test -f "$ARCHIVES_FILE"
  ARCHIVES=$(grep -v '^[[:space:]]*$' "$ARCHIVES_FILE" | tr '\n' ' ')
  echo "{\"event\": \"archives_file\", \"file\": \"$ARCHIVES_FILE\", \"archives\": $(grep -c . "$ARCHIVES_FILE")}"
fi
if [ -n "${ARCHIVES:-}" ]; then
  # shellcheck disable=SC2206
  archive_list=($ARCHIVES)
  for a in "${archive_list[@]}"; do test -f "$a"; done
  data_args=(--archives "${archive_list[@]}")
else
  compgen -G "$archive_glob" > /dev/null
  data_args=(--archive-glob "$archive_glob")
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$run_root/src:$here:/mnt/ai/scratch/farmNEAT/runtime/site"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$out"
if [ ! -f "$out/LATEST" ]; then
  : "${SEED_FROM:?OUT has no LATEST and SEED_FROM is unset}"
  seed=$(tr -d '[:space:]' < "$SEED_FROM/LATEST")
  test -f "$SEED_FROM/$seed"
  cp "$SEED_FROM/$seed" "$out/$seed.tmp"
  mv "$out/$seed.tmp" "$out/$seed"
  printf '%s\n' "$seed" > "$out/LATEST.tmp"
  mv "$out/LATEST.tmp" "$out/LATEST"
  echo "{\"event\": \"seeded\", \"from\": \"$SEED_FROM/$seed\", \"out\": \"$out\"}"
fi
latest=$(tr -d '[:space:]' < "$out/LATEST")
current=${latest#ckpt-}
current=$((10#${current%.pt}))
remaining=$((PHASE_END - current))
echo "{\"event\": \"phase\", \"out\": \"$out\", \"current_update\": $current, \"phase_end\": $PHASE_END, \"stop_after\": $remaining}"
if [ "$remaining" -le 0 ]; then
  exit 0
fi
# shellcheck disable=SC2086
exec work/venv/python -B RUN.py storage run \
  --reserve-bytes 20000000000 --timeout 85800 --interval 60 --scan-timeout 60 \
  --pause-for-storage-check -- \
  work/venv/python -B "$here/train.py" fit --out "$out" --device "$device" --resume --stop-after "$remaining" \
    "${data_args[@]}" --workers "$workers" ${TRAIN_ARGS:-}
