#!/bin/bash
# T1-ONLY wash rosters for the hourly washes (16:10, 17:10, 18:10): rebuild at :58 (MMPQ winning seats, mirror wins vs best 4x,
# held-out excluded), repoint broad/t1/wash/LATEST-NFLOW and LATEST-B, and queue caching of seats not yet in v11.
BR=/mnt/ai/scratch/farmNEAT/broad; L=$BR/t1/wash/WASH-HOURLY.log
cd $BR && nice -n 19 /usr/bin/python3 -B build_t1wash.py >> $L 2>&1 || { echo "$(date +%T) FAIL build_t1wash" >> $L; exit 2; }
echo "$(date +%T) LATEST-NFLOW -> $(readlink $BR/t1/wash/LATEST-NFLOW); LATEST-B -> $(readlink $BR/t1/wash/LATEST-B)" >> $L
n=$(( $(wc -l < $BR/t1/wash/UNCACHED.tsv) - 1 ))
if [ "$n" -gt 0 ] && ! squeue -h -n cache-wash | grep -q .; then
  B=/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11/backfill
  sbatch --parsable -J cache-wash -p cpu -w cpu-hub -c 1 --mem 3000 --array=0-5 --export=ALL,ROWS=$BR/t1/wash/UNCACHED.tsv,N=6,FIRST=0 -o $B/slurm-wash-%A_%a.out $B/slurm_shard.sbatch >> $L 2>&1
fi
python3 - "$(readlink -f $BR/t1/wash/LATEST-NFLOW)" "$(readlink -f $BR/t1/wash/LATEST-B)" >> /home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md <<'PY'
import json, os, sys, time
n = json.load(open(sys.argv[1] + '.receipt.json')); b = json.load(open(sys.argv[2] + '.receipt.json')); t = time.strftime('%H:%M')
subs = '; '.join(f"{k} {v['submission']} ({v['score']})" for k, v in n.get('best_submissions', {}).items())
print(f"- R-WASHMIX ({t}) {os.path.basename(sys.argv[1])} / {os.path.basename(sys.argv[2])}: seats {n.get('team_seats')} [{subs}]; "
      f"N flow probs {n['bucket_probs_arg']}; B realised {b.get('realised_share')}")
if n.get('mix_flag') or n.get('best_changed'):
    print(f"- R-WASHMIX-FLAG ({t}) {n.get('mix_flag')} {n.get('best_changed')} (mix kept 80/10/10)")
PY
