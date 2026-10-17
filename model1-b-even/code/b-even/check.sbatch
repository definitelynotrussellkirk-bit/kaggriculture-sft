#!/bin/bash
#SBATCH -J b-even-check
#SBATCH -p gpu
#SBATCH -w gpu-4090b
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 00:40:00
#SBATCH --chdir=/tmp
#SBATCH -o /mnt/ai/scratch/farmNEAT/b-even/logs/check-%j.out
set -euo pipefail
TR=/mnt/ai/scratch/farmNEAT/base-v1-pack/nfs/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc
export OA3_TRAINER=$TR PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 \
  PYTHONPATH=$TR:/mnt/ai/scratch/farmNEAT/base-v1-pack/nfs/primitive-bc-v6-20260927/src:/mnt/ai/scratch/farmNEAT/runtime/site \
  PBC_BUCKET_GROUP=3 PBC_DEC_TRIM=1 PBC_EMA_EXTRA=0.998,0.9995 PBC_GRAD_CLIP=10 PBC_HIST_DROPOUT=0.5 PBC_JOBBIAS=fold PBC_SEAT_FRACTION=0.5 \
  PBC_TURN_NORM=1 PBC_TURN_W_MOVE=0.5 PBC_TURN_W_ORDER=2
/mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12 -B /mnt/ai/scratch/farmNEAT/b-even/beven_check.py /mnt/ai/scratch/farmNEAT/b-even/BEVEN-CHECK.json
