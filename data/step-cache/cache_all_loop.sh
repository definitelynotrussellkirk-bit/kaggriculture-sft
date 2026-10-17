#!/bin/bash
# Continuous superset-cache builder (owner 09-29 16:40): every hour at :36 (after catalog refresh :20 and appender :32)
# list new unbuilt 09-27+ seats (new_rows.py) and build them as a 4-task SLURM array (09-30 00:28: on gpu-4090b spare CPUs; cpu-hub was starved by old backfill) (never on the desktop =
# gpu-skynet: its memory guard kills training), then refresh CACHE-COVERAGE.tsv. Stops after the 09-30 16:30 EDT data cut + 1 h.
B=/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11/backfill
end=$(date -d "2026-09-30 17:30 EDT" +%s)
while [ "$(date +%s)" -lt $end ] && [ ! -e $B/STOP ]; do
  now=$(date +%s); next=$(date -d "$(date +%Y-%m-%d\ %H):36" +%s); [ $next -le $now ] && next=$((next + 3600)); sleep $((next - now))
  stamp=$(date +%Y%m%d-%H%M); f=$B/NEW-$stamp.tsv
  n=$(nice -n 19 /usr/bin/python3 $B/new_rows.py $f 2>>$B/cache-all.log); prc=$?
  if [ $prc -ne 0 ] || ! [[ "$n" =~ ^[0-9]+$ ]]; then
    echo "$(date +%T) FAIL: new_rows.py rc=$prc out='$n' (NOT treated as 0 rows)" >> $B/cache-all.log; continue
  fi
  echo "$(date +%T) new unbuilt rows: $n ($f)" >> $B/cache-all.log
  if [ "$n" -gt 0 ]; then
    j=$(sbatch --parsable -J cache-new -p cpu -w cpu-hub -c 1 --mem 3000 --array=0-3%2 --export=ALL,ROWS=$f,N=4,FIRST=0,MIN_AVAIL_GB=10 -o $B/slurm-new-%A_%a.out $B/slurm_shard.sbatch 2>&1); src=$?
    if [ $src -ne 0 ] || ! [[ "$j" =~ ^[0-9]+ ]]; then echo "$(date +%T) FAIL: cache submission rc=$src: $j" >> $B/cache-all.log
    else echo "$(date +%T) cache job $j for $n rows" >> $B/cache-all.log; fi
  fi
  nice -n 19 /usr/bin/python3 $B/coverage.py >> $B/cache-all.log 2>&1
done
