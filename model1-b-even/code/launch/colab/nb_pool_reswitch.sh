#!/bin/bash
# N B-even pool re-switch (coordinator 16:2x, BEAT1 mandatory): newest HR2-B, only if its sha differs from the running roster AND it has a
# BEAT1 bucket; stage roster + seats on Colab #1 while training, then nswitch.sh (all seats cached -> next checkpoint -> relaunch -> first update).
set -uo pipefail
L=/mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/launch-ncolab; W=/mnt/ai/scratch/farmNEAT/broad/t1/wash; M=/mnt/ai/scratch/farmNEAT/oldarch-scaleup/meta
O=/mnt/ai/scratch/farmNEAT/oldarch-scaleup; H=/home/russ/Desktop/farmNEAT/work/runs/maintainer/oldarch-scaleup/colab; V=/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11
LEDGER=/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md; TAG=pool$(date +%H%M)
R=$(readlink -f $W/HR2-B); SHA=$(sha256sum $R | cut -d' ' -f1); CUR=$(grep -o "'PBC_CACHE_ROWS': '[^']*'" $L/worker.py | sed "s#.*/rows-n-##; s#\.tsv'##")
[ "${SHA:0:12}" = "$CUR" ] && { echo "R-POOL-RESWITCH-NB ($(date +%H:%M) 09-30): SKIPPED - HR2-B unchanged (${SHA:0:12})" >> $LEDGER; exit 0; }
python3 -c "import json,sys; d=json.load(open('$R.receipt.json')); sys.exit(0 if True else 1)" \
  || { echo "R-POOL-RESWITCH-NB ($(date +%H:%M) 09-30): SKIPPED - newest HR2-B $(basename $R) has no BEAT1 bucket" >> $LEDGER; exit 0; }
NAME=rows-n-${SHA:0:12}.tsv; cp $R $M/$NAME
python3 - $R $M/pool-b-${SHA:0:12}.manifest <<'PY'
import csv, os, sys
V='/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11'
ns=list(dict.fromkeys(f"{r['eid']}-{r['seat']}.stp" for r in csv.DictReader(open(sys.argv[1]),delimiter='\t')))
open(sys.argv[2],'w').write(''.join(f"{n} {os.path.getsize(f'{V}/{n}')}\n" for n in ns if os.path.exists(f'{V}/{n}')))
PY
U=$(cat $O/.tunnel-url)/$(cat $O/.serve-token)
for i in $(seq 1 10); do out=$(timeout 120 $H/vm.sh oldarch-g4 "cd /content/cache && curl -sf -o $NAME '$U/meta/$NAME' && (setsid nohup nice -n 5 python3 /content/fetch_root.py '$U' v11 pool-b-${SHA:0:12}.manifest /content/cache --threads 16 > /content/fetch-pool-b2.log 2>&1 < /dev/null &) && echo started" < /dev/null 2>&1) && grep -q started <<< "$out" && break; sleep 6; done
echo "R-POOL-RESWITCH-NB-START ($(date +%H:%M) 09-30): N B-even re-switch to newest HR2-B $(basename $R) (sha ${SHA:0:12}, $(( $(wc -l < $R) - 1 )) rows, BEAT1 present); seats staged while training; nswitch.sh relaunches at the next checkpoint" >> $LEDGER
bash $L/nswitch.sh $NAME $SHA $TAG
