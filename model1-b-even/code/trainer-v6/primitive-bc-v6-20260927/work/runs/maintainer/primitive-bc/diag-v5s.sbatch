#!/bin/bash
# v5s pipeline diagnostic (2026-09-27): one fresh fit, 40 updates, full-input SwiGLU model (v4 shape), output to a file;
# SIGABRT after 600 s dumps every thread's stack (PYTHONFAULTHANDLER) if it stalls.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-4090b
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=farmneat-v5s-diag
#SBATCH --no-requeue
set -uo pipefail
run_root=/mnt/ai/scratch/farmNEAT/primitive-bc-v5s-20260927
here=$run_root/work/runs/maintainer/primitive-bc
cd "$run_root"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 PYTHONPATH="$run_root/src:$here:/mnt/ai/scratch/farmNEAT/runtime/site"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mapfile -t ARCH < "$here/PRIORITY-V5S.files"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
timeout --signal=ABRT 600 work/venv/python -u -B "$here/train.py" fit --out "$here/results/diag" --device cuda --stop-after 40 \
  --size 12m --width 288 --layers 12 --heads 8 --dec-layers 4 --think off --ffn swiglu \
  --features extra,unit_extra,calc,prefix,reach_bias,carrier_read,reach_job --updates 3000 --batch-windows 64 \
  --lr 3e-4 --min-lr-frac 0.1 --warmup 300 --priority-file work/runs/maintainer/primitive-bc/PRIORITY-V5S.tsv \
  --priority-only true --workers 14 --log-every 10 --ckpt-every 1000 --eval-every 0 --bf16 true --grad-checkpoint true --archives "${ARCH[@]}" > "$here/results/diag.log" 2>&1
echo "exit $?"
