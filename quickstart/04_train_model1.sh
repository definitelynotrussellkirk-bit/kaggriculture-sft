#!/bin/bash
# Step 4: Model 1 (B-even architecture family) from random init on the step cache, a few updates on CPU.
#   bash 04_train_model1.sh [UPDATES]          (default 30)
# Same two commands the real run used (seed writer, then train.py fit --resume, as base-v1-pack/run_arm.sh ran it), with a
# tiny shape and batch. The shipped model: --width 432 --layers 10 --heads 36 --dec-layers 8 --dec-heads 12
# --batch-windows 128 --warmup 300, trained on CUDA with --compile-model true (model1-b-even/STAGES.md).
source "$(dirname "$0")/env.sh"
UPDATES=${1:-30}
OUT=$RUN/model1
rm -rf "$OUT"; mkdir -p "$OUT/seed"
cd "$PBC"
export PYTHONPATH="$TRAINER/primitive-bc-v6-20260927/src:$PBC"
# The seed writer copies a CUDA generator state from a checkpoint that is not shipped; on CPU any generator state works.
"$PY" -c "import torch; torch.save({'generator_state': torch.Generator().get_state()}, '$OUT/rng-cpu.pt')"
nice -n 10 "$PY" -B make_practice_scratch.py "$OUT/seed" --updates "$UPDATES" --warmup 2 --lr 3e-4 --seed 101 \
  --batch-windows 2 --width ${WIDTH:-96} --layers ${LAYERS:-2} --heads ${HEADS:-4} --ffn swiglu --dec-layers 2 --dec-heads 4 \
  --dropout 0.05 --feature-dropout 0.1 --v6 --v8 --aux-turn --optimizer muon --rng-from "$OUT/rng-cpu.pt"
seed=$(tr -d '[:space:]' < "$OUT/seed/LATEST")
cp "$OUT/seed/$seed" "$OUT/$seed"; echo "$seed" > "$OUT/LATEST"
PBC_CACHE_ROWS=$RUN/roster-model1.tsv nice -n 10 "$PY" -B train.py fit --out "$OUT" --device cpu --resume \
  --stop-after "$UPDATES" --updates "$UPDATES" --priority-file "$RUN/roster-model1.tsv" --priority-only true \
  --archives "$RUN"/episodes/*.zip --workers 0 --ckpt-every "$UPDATES" --log-every 1 --grad-checkpoint false \
  --compile-model false --pack-steps 48 --step-cache "$RUN/stepcache" --cache-pool-seats 1 --eval-every 0 --keep-ckpts 2 \
  --batch-windows 2 --warmup 2 --lr-schedule wsd --ema-decay 0.999 --mask-unit-noops true
ls -la "$OUT"
