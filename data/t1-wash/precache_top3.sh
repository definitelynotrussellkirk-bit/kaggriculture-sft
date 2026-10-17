#!/bin/bash
[ "$(date +%H%M)" -le 1900 ] || exit 0
BR=/mnt/ai/scratch/farmNEAT/broad; B=/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11/backfill
st=$(cd $BR && nice -n 19 /usr/bin/python3 -B precache_top3.py 2>&1 | tail -1)
n=$(( $(wc -l < $BR/t1/wash/PRECACHE-TOP3.tsv) - 1 )); j=""
if [ "$n" -gt 0 ] && ! squeue -h -n cache-top3 | grep -q .; then
  k=$(( n < 4 ? n : 4 ))
  j=$(sbatch --parsable -J cache-top3 -p cpu -w cpu-hub -c 1 --mem 3000 --array=0-$((k-1)) --export=ALL,ROWS=$BR/t1/wash/PRECACHE-TOP3.tsv,N=$k,FIRST=0 -o $B/slurm-top3-%A_%a.out $B/slurm_shard.sbatch 2>&1)
fi
echo "- R-PRECACHE-TOP3 ($(date +%H:%M)) $st; queued $n seats ${j:+job $j} (not in any pool share)" >> /home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md
python3 - "$st" >> /home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md <<'PY'
import json, sys, time, glob, os
st = json.loads(sys.argv[1]); C = '/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11'
cde = {k: v for k, v in st.items() if k.startswith('16736435:')}
print(f"- R-PRECACHE-CDE ({time.strftime('%H:%M')}) " + '; '.join(f"{k.split(':')[1]} (score {v['score']}): {v['listed_0930']} listed 09-30, {v['local']} local, {v['wins_local']} wins, {v['seats_to_cache']} seats still to cache" for k, v in cde.items()) + ' (0 pool share; promote via PROMOTE_T25)')
PY
