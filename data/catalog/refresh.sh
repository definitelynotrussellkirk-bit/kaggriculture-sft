#!/bin/bash
# Hourly replay-catalog pipeline (catalog-refresh.timer -> catalog-refresh.service; also safe to run by hand).
# 1. catalog.py refresh     copy new hosted replays of ours, index every new file (daily zips, fetcher archives,
#                           league exports, tapes) -> EPISODES/COPIES/TEAMS
# 2. seats.py build         SEATS.parquet, SEAT_DAYS.parquet, views/*.tsv (new games appear with their cheap facts)
# 3. seats.py facts         exact re-play of games without facts yet, newest day first, until minute 40
# 4. seats.py rebuild-tapes xsf/capsule seed+action tapes -> zip-json, only once the facts backlog is small, until minute 47
# 5. catalog.py update + seats.py build again, so rebuilt zips and new facts are in the tables and selections
# 6. tape_tags.py appends only new exact-action receipts, then atomically refreshes the tape-seat view
# One run at a time (flock); every stage writes atomically, so a killed run loses at most its unfinished batch.
# Env: FACTS_MINUTES (40), REBUILD_MINUTES (47), REBUILD_AFTER_FACTS (2000), WORKERS (5; catalog.py uses 5 itself).
set -u
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p state
LOG=state/pipeline.log
if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 5000000 ]; then mv -f "$LOG" "$LOG.1"; fi
exec >>"$LOG" 2>&1
exec 9>state/.pipeline.lock
if ! flock -n 9; then echo "$(date -u +%FT%TZ) another pipeline run holds state/.pipeline.lock; skipping"; exit 0; fi
PY=/usr/bin/python3
START=$(date +%s)
FACTS_END=$((START + ${FACTS_MINUTES:-40} * 60))
REBUILD_END=$((START + ${REBUILD_MINUTES:-47} * 60))
log() { echo "$(date -u +%FT%TZ) [$$] $*"; }
mem_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
# run NEED_GB script args...: skipped while MemAvailable < 4 GB + NEED_GB (owner 2026-09-28 lifted the 12 GB reserve: "we dont have a limit");
# facts / rebuild-tapes also size their worker pools and stop launching work below 12 GB themselves.
# run NEED_GB script args...: optional stage (facts, rebuild-tapes, tape_tags): a failure is logged as FAIL and the pipeline goes on.
# req NEED_GB script args...: required stage (refresh, build, update): a failure or a memory skip is logged as FAIL and the
# pipeline exits non-zero (ASTRA-PRO-CHECK-DATA-PLAN-0930 F5: a failed required stage used to report success).
FAILED=0
run() {
  local need=$1; shift
  local free; free=$(mem_gb)
  if [ "$free" -lt $((4 + need)) ]; then log "skip (MemAvailable ${free} GB < 4 + ${need} GB): $1 $2"; return 0; fi
  log "start: $*"; nice -n 19 "$PY" -B "$@"; local rc=$?
  if [ $rc -ne 0 ]; then log "FAIL (optional stage, rc=$rc): $1 $2"; else log "end (rc=0): $1 $2"; fi
  return 0
}
req() {
  local need=$1; shift
  local free; free=$(mem_gb)
  if [ "$free" -lt $((4 + need)) ]; then log "FAIL required stage skipped (MemAvailable ${free} GB < 4 + ${need} GB): $1 $2"; exit 3; fi
  log "start: $*"; nice -n 19 "$PY" -B "$@"; local rc=$?
  if [ $rc -ne 0 ]; then log "FAIL required stage (rc=$rc): $1 $2; pipeline stops, catalog generation NOT refreshed"; exit 2; fi
  log "end (rc=0): $1 $2"
}
log "pipeline start (MemAvailable $(awk '/MemAvailable/ {printf "%.1f GB", $2/1048576}' /proc/meminfo))"
req 3 catalog.py refresh
req 2 seats.py build
run 1 seats.py facts --deadline-epoch "$FACTS_END" --workers "${WORKERS:-5}"
run 1 seats.py rebuild-tapes --deadline-epoch "$REBUILD_END" --after-facts "${REBUILD_AFTER_FACTS:-2000}" --workers "${WORKERS:-5}"
req 3 catalog.py update
req 2 seats.py build
run 2 tape_tags.py --day-from 2026-09-27 --workers 2
# corpus-final manifest (owner 09-29 20:48): dedup + provenance + recency tiers + views, rebuilt on cpu-hub via SLURM
job=$(sbatch --parsable -p cpu -w cpu-hub -c 2 --mem 12G /mnt/ai/scratch/farmNEAT/corpus-final/build_manifest.sbatch 2>&1); rc=$?
if [ "$rc" -ne 0 ] || ! [[ "$job" =~ ^[0-9]+(\;[[:alnum:]_.-]+)?$ ]]; then log "FAIL: corpus-final manifest submission (rc=$rc): $job"; exit 4; fi
log "corpus-final manifest queued: job $job (completion not verified here)"
log "pipeline done in $(( $(date +%s) - START )) s"
