#!/bin/bash
#SBATCH -p cpu,gpu -c 1 --mem=1500M --exclude=gpu-4090b,gpu-3090,gpu-skynet,gpu-spark,laptop-4060 -t 03:00:00 -J nflow-shards -o /mnt/ai/scratch/farmNEAT/n-flow/logs/shards-%A_%a.out
# N flow shard build, one seat list slice per array task: sbatch --array=0-N%K -w NODE build-shards.sbatch N+1
cd /mnt/ai/scratch/farmNEAT/n-flow/code
export PYTHONPATH=/mnt/ai/scratch/farmNEAT/runtime/site
exec nice -n 10 /mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12 -B nf_format.py build \
  /mnt/ai/scratch/farmNEAT/n-flow/roster/rows-nflow-v2.1ad4d8296e84.tsv /mnt/ai/scratch/farmNEAT/n-flow/shards-v1 \
  --shard $SLURM_ARRAY_TASK_ID/$1
