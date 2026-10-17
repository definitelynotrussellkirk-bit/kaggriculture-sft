#!/bin/bash
# Step 5: Model 2 (N flow) from random init, a few updates on CPU.
#   bash 05_train_model2.sh [UPDATES]          (default 60)
# 5a: nf_format.py build projects each .stp onto the 320-node graph layout (one .nfz per seat; same encoder the loader would
#     apply to a .stp on the fly, so the shard only changes speed). 5b: train.py from random init with a tiny shape.
# The shipped model used the defaults (--width 256 --blocks 4 --heads 8 --rel-heads 4 --fast 128 --slow 256 --worker 64
# --dec-layers 2) on a GPU with --workers 4 --streams 4..8 --accum 8 (model2-n-flow/STAGES.md).
source "$(dirname "$0")/env.sh"
UPDATES=${1:-60}
OUT=$RUN/model2
rm -rf "$OUT" "$RUN/nfz"
export NFLOW_PBC_DIR=$NFLOW_DEPLOY
[ "${DEVICE:-cpu}" = cpu ] && export CUDA_VISIBLE_DEVICES=
cd "$NFLOW"
nice -n 10 "$PY" -B nf_format.py build "$RUN/roster-model2.tsv" "$RUN/nfz" --stp-dir "$RUN/stepcache"
ls -la "$RUN/nfz"
nice -n 10 "$PY" -B train.py --roster "$RUN/roster-model2.tsv" --out "$OUT" --shards "$RUN/nfz" --stp-dir "$RUN/stepcache" \
  --updates "$UPDATES" --workers 0 --streams 2 --chunk 24 --accum 1 --warmup 2 --ckpt-every "$UPDATES" --log-every 1 \
  --seed 17 --width 64 --blocks 1 --heads 4 --rel-heads 2 --fast 32 --slow 32 --worker 16 --dec-layers 1
ls -la "$OUT"
