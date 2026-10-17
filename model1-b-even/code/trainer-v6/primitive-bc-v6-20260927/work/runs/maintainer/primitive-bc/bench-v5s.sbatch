#!/bin/bash
# v5s speed benchmark (2026-09-27): two fresh fits of 60 updates each on gpu-4090b, same rows; A = v1 shape + SwiGLU,
# B = A + every growth-era input. Output: results/bench-A, results/bench-B (metrics.jsonl: seconds per 20 updates).
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-4090b
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=farmneat-v5s-bench
#SBATCH --no-requeue
set -euo pipefail
run_root=/mnt/ai/scratch/farmNEAT/primitive-bc-v5s-20260927
here=$run_root/work/runs/maintainer/primitive-bc
cd "$run_root"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$run_root/src:$here:/mnt/ai/scratch/farmNEAT/runtime/site"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mapfile -t ARCH < "$here/PRIORITY-V5S.files"
COMMON=(--size 12m --width 288 --layers 10 --heads 8 --dec-layers 2 --think off --ffn swiglu --updates 7000 --batch-windows 64
        --lr 3e-4 --min-lr-frac 0.1 --warmup 500 --priority-file work/runs/maintainer/primitive-bc/PRIORITY-V5S.tsv
        --priority-only true --workers 14 --log-every 20 --ckpt-every 1000 --eval-every 0 --archives "${ARCH[@]}")
for arm in A B; do
  feats=""
  [ "$arm" = B ] && feats="extra,unit_extra,calc,prefix,reach_bias,carrier_read,reach_job"
  echo "{\"event\": \"bench\", \"arm\": \"$arm\", \"features\": \"$feats\", \"start\": \"$(date -u +%FT%TZ)\"}"
  work/venv/python -B "$here/train.py" fit --out "$here/results/bench-$arm" --device cuda --stop-after 60 \
    --features "$feats" "${COMMON[@]}" 2>&1 | tail -5
done
