#!/bin/bash
# Owner 2026-09-25: fetch top-team games through the Kaggle episode API, 2000 replay calls per UTC day (owner 04:20: use more, limit ~4k).
# Each UTC day: refresh the xishengfeng episode index (a dataset file, no episode call), then fetch_topteam_replays.py with the
# budget (>= 2 s apart, 6 h cooldown after any 429; 1 leaderboard call per run). Output: topteam/archives/*.zip (official member
# format) + topteam/MANIFEST.jsonl. Nothing here touches TRAIN-ARCHIVES.txt; the maintainer admits archives after verification.
cd /mnt/ai/scratch/farmNEAT/primitive-bc-data/extra || exit 1
# 2026-09-27: our team was renamed "Russell Kirk" -> "有辣条有权" (both excluded); owner "do it SLOWLY": >= 10 s between replay
# calls; the loop sleeps until the next 00:15 UTC before each run (a restart does not start a run immediately).
while true; do
  now=$(date -u +%s); next=$(date -u -d "$(date -u -d tomorrow +%F) 00:15" +%s); [ "$(date -u +%H%M)" -lt 0015 ] && next=$(date -u -d "$(date -u +%F) 00:15" +%s); sleep $((next - now))
  rm -rf index/tmp && mkdir -p index/tmp
  if timeout 900 kaggle datasets download xishengfeng/kaggriculture-replay-db -f replay_db/index/episodes_index.parquet -p index/tmp > index/tmp/dl.log 2>&1; then
    f=$(ls index/tmp/*.parquet 2>/dev/null | head -1)
    if [ -z "$f" ] && ls index/tmp/*.zip > /dev/null 2>&1; then (cd index/tmp && unzip -q -o ./*.zip); f=$(ls index/tmp/*.parquet 2>/dev/null | head -1); fi
    [ -n "$f" ] && mv "$f" "index/xsf-episodes_index-$(date -u +%Y%m%dT%H%M).parquet"
  fi
  idx=$(ls -t index/xsf-episodes_index-*.parquet | head -1)
  echo "$(date -u +%FT%T) run with $idx"
  nice -n 10 python3 fetch_topteam_replays.py --out topteam --xsf-index "$idx" --top-n 20 --list-top 20 --max-list-calls-per-day 60 --since 2026-09-03 --exclude-team "Russell Kirk" "有辣条有权" \
    --cap-sub-day 20 --max-replays-per-day 2000 --min-interval 10 --cooldown-hours 6 --pack-every 100 2>&1 | tail -n 5
  echo "$(date -u +%FT%T) run ended; calls today: $(python3 -c "import json; print(json.load(open('topteam/state.json'))['calls'])" 2>/dev/null)"
done
