#!/bin/bash
# Step 2: replay -> per-step features for both seats (the v11 step cache both models trained from).
#   bash 02_features.sh EPISODE_ID DAY
# Runs step_cache.py build --v6 --v8, the flags the shipped cache used. One .stp per seat: all 719 decision steps with legacy
# entity tokens, v6 history/aux arrays, v8 feature families, and the action labels. Uses the trainer-v6 tree: its step_cache.py
# and packing sources are byte-identical to data/step-cache/code, and it also has v8_lattice.py, which the v8 schema imports.
source "$(dirname "$0")/env.sh"
EID=$1; DAY=$2
zipf=$RUN/episodes/kaggriculture-episodes-$DAY-$EID.zip
rows=$RUN/rows-cache.tsv
printf 'archive\tmember\tday\teid\tseat\tweight\treason\n' > "$rows"
for seat in 0 1; do printf '%s\t%s.json\t%s\t%s\t%s\t1\tquickstart\n' "$zipf" "$EID" "$DAY" "$EID" "$seat" >> "$rows"; done
cd "$PBC"
PYTHONPATH="$TRAINER/primitive-bc-v6-20260927/src:$PBC" nice -n 10 "$PY" -B step_cache.py build "$rows" "$RUN/stepcache" --procs 1 --v6 --v8
ls -la "$RUN/stepcache"
