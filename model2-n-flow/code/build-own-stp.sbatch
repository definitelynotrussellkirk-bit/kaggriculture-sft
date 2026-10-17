#!/bin/bash
#SBATCH -p cpu,gpu -c 4 --mem=6G -t 01:30:00 -J nflow-ownstp --exclude=gpu-4090b,gpu-3090,gpu-skynet,gpu-spark,laptop-4060
#SBATCH -o /mnt/ai/scratch/farmNEAT/n-flow/logs/ownstp-%j.out
# v11-identical step cache (v6 + v8, legacy schema) for our 09-29/09-30 losing seats that had no .stp (owner 09-30).
cd /mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc
export PYTHONPATH=/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927/src:/mnt/ai/scratch/farmNEAT/runtime/site
exec nice -n 10 /mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12 -B step_cache.py build \
  /mnt/ai/scratch/farmNEAT/n-flow/stp-own-loss/rows-own-loss.tsv /mnt/ai/scratch/farmNEAT/n-flow/stp-own-loss --v6 --v8 --procs 4
