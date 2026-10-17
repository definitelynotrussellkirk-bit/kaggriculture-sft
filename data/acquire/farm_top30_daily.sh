#!/bin/bash
# Owner 2026-09-25 ~23:00-23:25 UTC: "whenever someone in the top 30 loses ... find it and train that for sure", "study all
# defeats these players have -- even if it was on their way up", "aim for maybe using all 3k API pulls each day (towards the day
# roll out)". Twice per UTC day: 00:05 (backlog, cap 2000) and 22:00 (new losses, cap 2900) on the shared per-day counter
# work/runs/maintainer/pbc-autosubmit/KAGGLE-REQUESTS-<day>.txt (the hourly farm loop adds its own calls to the same file).
cd /mnt/ai/scratch/farmNEAT/primitive-bc-data/extra || exit 1
CNT_DIR=/home/russ/Desktop/farmNEAT/work/runs/maintainer/pbc-autosubmit
EXTRA=top30-losses/XSF-ALLSUB-LOSSES-20260925.json
run() {  # $1 = cap
  day=$(date -u +%F)
  echo "$(date -u +%FT%TZ) run cap $1 counter $(cat $CNT_DIR/KAGGLE-REQUESTS-$day.txt 2>/dev/null || echo 0)"
  nice -n 10 python3 farm_top30_losses.py --out top30-losses --counter $CNT_DIR/KAGGLE-REQUESTS-$day.txt --cap "$1" --top 30 \
    --extra-episodes $EXTRA 2>&1 | grep -v '^ok ' | tail -n 8
}
until_utc() {  # sleep until the next HH:MM UTC
  now=$(date -u +%s); t=$(date -u -d "$(date -u +%F) $1" +%s); [ "$t" -le "$now" ] && t=$((t + 86400)); sleep $((t - now))
}
while true; do
  until_utc 00:05; run 2000
  until_utc 22:00; run 2900
done
