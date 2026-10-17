#!/bin/bash
#SBATCH -p gpu -w gpu-skynet --gres=gpu:4090:1 -c 6 --mem=20G -t 01:10:00 -J nflow-arm
#SBATCH --chdir=/tmp
#SBATCH -o /mnt/ai/scratch/farmNEAT/n-flow/logs/arm-%j.out
# Matched N flow t4 arm on gpu-skynet (ASTRA NFLOW-IMPROVE E8, coordinator 09:5x; skynet is the owner's desktop:
# nice 19, <= 4 data workers, 16 GiB MemAvailable preflight, scancel below 8 GiB polled every 2 s — gpu-ops' guard lines).
#   sbatch -J nflow-arm-NAME arm_skynet.sbatch NAME BASE_CKPT ROSTER STOP_AT EMA [LR] [MAX_SECONDS]
# A bucketed roster (receipt bucket_probs_arg, e.g. t1) is drawn with --bucket-probs from its receipt.
# Seed = BASE with raw := EMA (make_t4_seed.py), lr 1e-4 wsd, no warmup, clip 10, --new-seed 41, 4 workers x 5 streams x
# accum 3 (60 streams/update), stop at update STOP_AT. Training only; the desktop submits release_run.sbatch afterwards.
set -euo pipefail
NAME=$1; BASE=$2; ROSTER=$3; STOP_AT=$4; EMA=$5; LR=${6:-1e-4}; MAXS=${7:-3300}
BP=$(python3 -c "import json;print(json.load(open('$ROSTER.receipt.json')).get('bucket_probs_arg') or '')" 2>/dev/null)
EXTRA=(); [ -n "$BP" ] && EXTRA=(--bucket-probs $BP)
NR=/mnt/ai/scratch/farmNEAT/n-flow; C=$NR/code; W=$NR/t4f/$NAME; O=$NR/runs/$NAME
a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); (( a >= 16 )) || { echo "memory preflight: $a GiB available < 16" >&2; exit 75; }
( while sleep 2; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); (( a < ${MIN_AVAIL_GB:-8} )) && { echo "memory guard: $a GiB < ${MIN_AVAIL_GB:-8}" >&2; scancel "$SLURM_JOB_ID"; }; done ) & g=$!; trap 'kill $g 2>/dev/null || true' EXIT
PY=/mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12
export PYTHONPATH=/mnt/ai/scratch/farmNEAT/runtime/site NF_PREFETCH=4
mkdir -p $W/seed $O; cd $C
cp $ROSTER $W/rows.tsv; cp $ROSTER.receipt.json $W/rows.tsv.receipt.json
[ -s $W/seed/$(basename $BASE) ] || $PY $C/make_t4_seed.py $BASE $W/seed/$(basename $BASE)
[ -f $O/LATEST ] || { cp $W/seed/$(basename $BASE) $O/; basename $BASE > $O/LATEST; }
ALLOW=lr,warmup,schedule,decay_updates,decay_start_utc,stop_utc,accum,streams_total,bucket_probs,roster_sha256,wdl_frac,ema,grad_clip,chunk,min_lr_frac,weight_decay,beta2,optimizer,muon_momentum,muon_ns_steps,adamw_lr_mult,w_wdl,w_next,w_opp
ARGS=(--roster $W/rows.tsv --shards $NR/shards-v1 --stp-dir /mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11 --out $O
      --max-seconds $MAXS --updates 200000 --lr $LR --warmup 0 --schedule wsd --decay-updates 0 --workers 4 --streams 5
      --chunk 24 --accum 3 --grad-clip 10 --ema $EMA --ckpt-every 250 --ckpt-minutes 5 --keep-ckpts 3 --log-every 10
      --new-seed 41 --stop-at $STOP_AT "${EXTRA[@]}" --allow-recipe-change $ALLOW)
echo "$PY -B train.py ${ARGS[*]}  # base $BASE (raw := EMA); code $(sha256sum train.py nf_data.py nf_format.py nf_model.py | cut -c1-12 | tr '\n' ' ')" > $W/CMD
nice -n 19 $PY -B train.py "${ARGS[@]}" > $O/train.log 2>&1
echo "done $(cat $O/LATEST)"
