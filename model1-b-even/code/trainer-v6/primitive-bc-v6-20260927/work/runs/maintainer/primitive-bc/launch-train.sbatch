#!/bin/bash
# Primitive-BC fit on gpu-4090b (pinned runtime, RUN.py storage gate). Not submitted by its author.
#
#   RUN_ROOT      repository replica on shared scratch (src/, RUN.py, work/venv/python -> pinned
#                 runtime, work/datasets/public-episodes/daily/SELECTION.json, this directory)
#   ARCHIVE_GLOB  daily replay archives visible from gpu-4090b
#   OUT           output directory; an existing OUT/LATEST resumes the run
#   WORKERS       data.py worker processes (keep <= cpus-per-task - 2)
#   TRAIN_ARGS    extra train.py fit arguments, e.g. "--rank-max 50 --updates 40000 --size 4m"
#
#   sbatch --export=ALL,RUN_ROOT=...,ARCHIVE_GLOB=...,TRAIN_ARGS="--rank-max 50" launch-train.sbatch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-4090b
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=farmneat-primitive-bc
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CUDA_VISIBLE_DEVICES:?}"

run_root=${RUN_ROOT:-/mnt/ai/scratch/farmNEAT/primitive-bc-source-20260923}
archive_glob=${ARCHIVE_GLOB:-/mnt/ai/scratch/farmNEAT/primitive-bc-data/kaggriculture-episodes-2026-09-*.zip}
here="$run_root/work/runs/maintainer/primitive-bc"
out=${OUT:-$here/results/fit-$SLURM_JOB_ID}
workers=${WORKERS:-12}

cd "$run_root"
test -x work/venv/python
test -f RUN.py
test -f "$here/train.py" && test -f "$here/model.py" && test -f "$here/data.py" && test -f "$here/codec.py"
test -f work/datasets/public-episodes/daily/SELECTION.json
compgen -G "$archive_glob" > /dev/null
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$run_root/src:$here:/mnt/ai/scratch/farmNEAT/runtime/site"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$out"
resume=()
if [ -f "$out/LATEST" ]; then resume=(--resume); fi
# shellcheck disable=SC2086
exec work/venv/python -B RUN.py storage run \
  --reserve-bytes 20000000000 --timeout 85800 --interval 60 --scan-timeout 60 \
  --pause-for-storage-check -- \
  work/venv/python -B "$here/train.py" fit --out "$out" --device cuda "${resume[@]}" \
    --archive-glob "$archive_glob" --workers "$workers" ${TRAIN_ARGS:-}
