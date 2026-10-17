#!/bin/bash
# Winners top-up (coordinator 07:3x, d30/t4 recipe) for B or N B-even on gpu-skynet (owner desktop: <=4 workers, nice 19, 8 GiB guard, 16 GiB preflight): B's trainer (base-v1-pack, RUN-10 pbc/speed env), eager,
# from RUN_DIR/LATEST (the newest hourly keep, copied by the lane), roster ROWS, lr LR, 300 updates (cap MAX_S s), EMA continues.
#   sbatch --export=ALL,ROWS=..,LR=..,RUN=.. t4w-bn.sbatch
#SBATCH -J t4w-sky
#SBATCH -p gpu
#SBATCH -w gpu-skynet
#SBATCH --gres=gpu:4090:1
#SBATCH -c 8
#SBATCH --mem 20G
#SBATCH -t 01:00:00
#SBATCH --chdir=/tmp
#SBATCH -o /mnt/ai/scratch/farmNEAT/t4w/logs/t4w-%j.out
set -euo pipefail
NFS=/mnt/ai/scratch/farmNEAT; LOCAL=$HOME/farmneat-local; : "${ROWS:?}" "${LR:?}" "${RUN:?}"
a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); (( a >= 16 )) || { echo "memory preflight: $a GiB available < 16" >&2; exit 75; }
if awk -F'\t' 'NR>1 && $2=="tape"{f=1; exit} END{exit !f}' "$ROWS"; then echo "tape rows in $ROWS" >&2; exit 2; fi
PYTHONPATH=$NFS/runtime/site $NFS/runtime/python/bin/python3.12 -B $NFS/broad/ckpt_select.py $RUN > $RUN/RESUME-SELECT-$SLURM_JOB_ID.json
RJ=$NFS/base-v1/runs/b/RUN-10.json
eval "$(python3 -c "import json,shlex;e=json.load(open('$RJ'))['pbc_env'];e.update(json.load(open('$RJ'))['speed_env']);print('\n'.join(f'export {k}={shlex.quote(v)}' for k,v in e.items()))")"
for v in $(compgen -e | grep -E '^(OA3_|N_)'); do unset "$v"; done; unset PBC_WORKER_W_TSV
export PBC_CACHE_ROWS=$ROWS CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True CPATH=/usr/include/python3.12
rsync -a $NFS/runtime/ $LOCAL/fs/runtime/; rsync -a --delete $NFS/base-v1-pack/ $LOCAL/fs/base-v1-pack/
( while sleep 2; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); (( a < ${MIN_AVAIL_GB:-8} )) && { echo "memory guard: $a GiB < ${MIN_AVAIL_GB:-8}" >&2; scancel "$SLURM_JOB_ID"; }; done ) & g=$!; trap 'kill $g 2>/dev/null || true' EXIT   # host RAM guard (SLURM --mem is not enforced; the 07:36 node death)
PY=$LOCAL/fs/runtime/python/bin/python3.12
# run_arm counts UPDATES from the pack seed (end = s0 + UPDATES); an extra stop arg would change the ARM-ID, so the 300 go here
S0=$(tr -d '[:space:]' < $LOCAL/fs/base-v1-pack/seeds/base1-b/LATEST); S0=${S0#ckpt-}; S0=$((10#${S0%.pt}))
CUR=$(tr -d '[:space:]' < $RUN/LATEST); CUR=${CUR#ckpt-}; CUR=$((10#${CUR%.pt})); U_ARG=$(( CUR + ${UPDATES:-300} - S0 ))
PY=$PY EXTRA_PYTHONPATH=$LOCAL/fs/runtime/site WORK=$LOCAL/fs/pbc-work-t4w WORKERS=4 CKPT_EVERY=100000 \
  nice -n 19 bash $LOCAL/fs/base-v1-pack/run_arm.sh base1-b $U_ARG $RUN \
  --grad-checkpoint false --compile-model true --pack-steps 256 --step-cache $NFS/base-v1-stepcache-v11 --cache-pool-seats 4 --eval-every 0 --keep-ckpts 3 \
  --batch-windows 128 --warmup 300 --grow-lr-mult 1 --decay-updates 0 --lr-schedule wsd --ema-decay 0.999 --max-seconds ${MAX_S:-2700} --mask-unit-noops true \
  --lr $LR --aux-scale 2.8 >> $RUN/train.log 2>&1
echo "done $(cat $RUN/LATEST)"
