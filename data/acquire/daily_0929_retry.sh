#!/bin/bash
# Hourly retry for the last official daily dump kaggriculture-episodes-2026-09-29 (owner 09-29 20:48: "grab that" /
# "that's our last official source"). Counts +2 per CLI call under flock (API calls are the sacred resource).
OUT=/mnt/ai/scratch/farmNEAT/primitive-bc-data
L=/mnt/ai/scratch/farmNEAT/watchers/daily_0929_retry.log
D=/home/russ/Desktop/farmNEAT/work/runs/maintainer/pbc-autosubmit
Z=$OUT/kaggriculture-episodes-2026-09-29.zip
for i in $(seq 1 30); do
  F=$D/KAGGLE-REQUESTS-$(date -u +%F).txt
  ( flock 9; n=$(cat "$F" 2>/dev/null || echo 0); echo $((n+2)) > "$F" ) 9>>"$F.lock"
  msg=$(cd "$OUT" && timeout 1800 kaggle datasets download -d kaggle/kaggriculture-episodes-2026-09-29 -p "$OUT" 2>&1 | tail -2 | tr '\n' ' ')
  if [ -s "$Z" ] && unzip -tq "$Z" >/dev/null 2>&1; then
    echo "$(date '+%F %T') attempt $i OK: $(stat -c %s "$Z") bytes, $(unzip -l "$Z" | tail -1)" >> "$L"; exit 0
  fi
  echo "$(date '+%F %T') attempt $i not ready: $msg" >> "$L"
  sleep 3600
done
echo "$(date '+%F %T') EXHAUSTED after 30 attempts: no valid zip" >> "$L"; exit 1
