#!/bin/bash
# Backfill shard FIRST+array_index of N over ALL.tsv (same build command and memory guard as pbc-colab/cache-build-guard.sbatch).
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=cache-backfill
#SBATCH --no-requeue
set -euo pipefail
( while sleep 5; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); if [ "$a" -lt "${MIN_AVAIL_GB:-6}" ]; then echo "memory guard: MemAvailable ${a} GB, cancelling" >&2; scancel "${SLURM_JOB_ID}"; fi; done ) &
ROWS=${ROWS:-/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11/backfill/ALL2.tsv}; N=${N:-16}; FIRST=${FIRST:-8}
SHARD=$((FIRST + SLURM_ARRAY_TASK_ID))/$N
R=/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927
export PYTHONPATH=$R/work/runs/maintainer/primitive-bc:$R/src:/mnt/ai/scratch/farmNEAT/runtime/site
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $R/work/runs/maintainer/primitive-bc
nice -n 10 /mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12 -B step_cache.py build "$ROWS" /mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11 --procs 1 --shard "$SHARD" --v6 --v8
