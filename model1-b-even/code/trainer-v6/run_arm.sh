#!/bin/bash
# One ablation arm from this pack (build_pack.py): resumes seeds/ARM for UPDATES updates into OUT (re-running continues where
# it stopped), on the pack's replay subset and rows, so every arm anywhere sees the same data in the same order.
#   bash run_arm.sh ARM UPDATES OUT [extra train.py fit args]
# env: PY (default python3), WORK (writable scratch for the code copy, default /tmp/pbc-work), WORKERS (data processes,
#      ROWS_FILE (rows TSV inside the pack, default data/rows.tsv; e.g. data/rows-500.tsv for the practice subset),
#      default nproc-1), DEVICE (default cuda), CKPT_EVERY (default 50), EXTRA_PYTHONPATH
set -euo pipefail
PACK=$(cd "$(dirname "$0")" && pwd)
ARM=$1; UPDATES=$2; OUT=$(realpath -m "$3"); shift 3   # absolute: the trainer runs from the code directory
PY=${PY:-python3}
WORK=${WORK:-/tmp/pbc-work}
SEED_DIR=$PACK/seeds/$ARM
[ -f "$SEED_DIR/LATEST" ] || { echo "no seed for arm $ARM in $SEED_DIR" >&2; exit 2; }
code=$WORK/primitive-bc-v6-20260927
here=$code/work/runs/maintainer/primitive-bc
# the code copy is keyed by the pack path AND the content hash of its code, so an updated pack never runs stale code
KEY="$PACK $(cd "$PACK/nfs" && find . -type f \( -name '*.py' -o -name '*.json' \) -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16)"
# rows.tsv is also cached in WORK; code-only identity leaves changed selections stale.
ROWS=${ROWS_FILE:-data/rows.tsv}
KEY="$KEY rows:$ROWS:$(sha256sum "$PACK/$ROWS" | cut -d' ' -f1)"
if [ ! -f "$WORK/.ready" ] || [ "$(cat "$WORK/.ready")" != "$KEY" ]; then
  rm -rf "$WORK"; mkdir -p "$WORK"
  cp -r "$PACK/nfs/primitive-bc-v6-20260927" "$WORK/"
  # nodes that mount the NAS keep the code byte-identical: several files are hashed into the step-cache identity
  if [ ! -d /mnt/ai/scratch/farmNEAT ]; then
    grep -rl "/mnt/ai/scratch/farmNEAT/" "$code" --include=*.py | xargs -r sed -i "s#/mnt/ai/scratch/farmNEAT/#$PACK/nfs/#g"
  fi
  sed "s#@PACK@#$PACK#g" "$PACK/$ROWS" > "$WORK/rows.tsv"
  echo "$KEY" > "$WORK/.ready"
fi
mkdir -p "$OUT"
seed=$(tr -d '[:space:]' < "$SEED_DIR/LATEST")
# the output directory belongs to one arm and one seed checkpoint: refuse to resume anything else
# chunk-varying flags (LR decay horizon, time box, EMA, cache location/pool, checkpoint retention) do not change the arm
id_args=(); skip=0
for a in "$@"; do
  if (( skip )); then skip=0; continue; fi
  case "$a" in
    --decay-updates|--max-seconds|--ema-decay|--step-cache|--cache-pool-seats|--keep-ckpts|--workers|--aux-scale|--lr|--pack-steps|--recur-train-min) skip=1; continue;;
    --decay-updates=*|--max-seconds=*|--ema-decay=*|--step-cache=*|--cache-pool-seats=*|--keep-ckpts=*|--workers=*|--aux-scale=*|--lr=*|--pack-steps=*|--recur-train-min=*) continue;;
  esac
  id_args+=("$a")
done
ID="$ARM $(sha256sum "$SEED_DIR/$seed" | cut -c1-16) rows:$ROWS ${id_args[*]}"
if [ -f "$OUT/ARM-ID" ] && [ "$(cat "$OUT/ARM-ID")" != "$ID" ]; then
  echo "refusing: $OUT holds [$(cat "$OUT/ARM-ID")], this run is [$ID]" >&2; exit 3
fi
if [ ! -f "$OUT/ARM-ID" ] && [ -f "$OUT/LATEST" ]; then
  echo "refusing: $OUT has checkpoints but no ARM-ID" >&2; exit 3
fi
echo "$ID" > "$OUT/ARM-ID"
if [ ! -f "$OUT/LATEST" ]; then
  cp "$SEED_DIR/$seed" "$OUT/$seed.tmp" && mv "$OUT/$seed.tmp" "$OUT/$seed"
  echo "$seed" > "$OUT/LATEST"
fi
s0=${seed#ckpt-}; s0=$((10#${s0%.pt}))
latest=$(tr -d '[:space:]' < "$OUT/LATEST"); cur=${latest#ckpt-}; cur=$((10#${cur%.pt}))
end=$((s0 + UPDATES)); remaining=$((end - cur))
echo "{\"arm\": \"$ARM\", \"seed_update\": $s0, \"current\": $cur, \"end\": $end, \"remaining\": $remaining}"
[ "$remaining" -le 0 ] && exit 0
cd "$code"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$code/src:$here${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}"
# shellcheck disable=SC2046
exec "$PY" -B "$here/train.py" fit --out "$OUT" --device "${DEVICE:-cuda}" --resume --stop-after "$remaining" --updates "$end" \
  --priority-file "$WORK/rows.tsv" --priority-only true --archives $(ls "$PACK"/data/replays-*.bin) \
  --workers "${WORKERS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}" --ckpt-every "${CKPT_EVERY:-50}" --log-every 10 "$@"
