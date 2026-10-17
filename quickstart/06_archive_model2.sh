#!/bin/bash
# Step 6 (optional): Model 2 checkpoint -> Kaggle submission archive -> one full game in a fresh process.
#   bash 06_archive_model2.sh
# The same three commands the shipped N flow archive went through (model2-n-flow/STAGES.md, "Export and archive"):
# ema_export.py, build_archive.py build (fp16, greedy), build_archive.py verify (cold process, 719 steps vs PASS).
source "$(dirname "$0")/env.sh"
OUT=$RUN/model2-archive
rm -rf "$OUT"; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=
ckpt=$RUN/model2/$(tr -d '[:space:]' < "$RUN/model2/LATEST")
cd "$NFLOW"
"$PY" -B ema_export.py "$ckpt" "$OUT/ema.pt"
cd "$NFLOW_DEPLOY"
export PYTHONPATH="$CODE/nflow-deploy/src:$NFLOW_DEPLOY"
nice -n 10 "$PY" -B build_archive.py build --model "$OUT/ema.pt" --out "$OUT/quickstart-nflow.tar.gz" --storage fp16 \
  --temperature 0 --no-verify
# play.py imports the opponent league, which is not in this repo; quickstart/shim/league.py supplies a PASS rival.
PYTHONPATH="$PYTHONPATH:$QS/shim" nice -n 10 "$PY" -B build_archive.py verify --archive "$OUT/quickstart-nflow.tar.gz" \
  --opponents PASS --seats 0 --out "$OUT/VERIFY.json"
"$PY" -I -c "import json,sys; v=json.load(open(sys.argv[1])); print(json.dumps({k: v[k] for k in v if k != 'games'})[:600])" "$OUT/VERIFY.json"
ls -la "$OUT"
