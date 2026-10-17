#!/bin/bash
# The ONE remaining N B-even re-switch (coordinator 16:5x): at ~17:50, only if the newest HR2-B has >= 720 act seats (+10% over a9ab27e1ba44's 654).
# Stage roster + every seat on nflow-x1 while training; nswitch-x1.sh stops train.py only after the newest checkpoint is pulled to the NAS.
set -uo pipefail
L=/mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/launch-ncolab; W=/mnt/ai/scratch/farmNEAT/broad/t1/wash; M=/mnt/ai/scratch/farmNEAT/oldarch-scaleup/meta
O=/mnt/ai/scratch/farmNEAT/oldarch-scaleup; H=/home/russ/Desktop/farmNEAT/work/runs/maintainer/oldarch-scaleup/colab
LEDGER=/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md
R=$(readlink -f $W/HR2-B); SHA=$(sha256sum $R | cut -d' ' -f1); N=$(( $(wc -l < $R) - 1 ))
[ "${SHA:0:12}" = a9ab27e1ba44 ] && { echo "R-NB-RESWITCH ($(date +%H:%M) 09-30): SKIPPED - HR2-B unchanged (a9ab27e1ba44)" >> $LEDGER; exit 0; }
[ $N -ge 720 ] || { echo "R-NB-RESWITCH ($(date +%H:%M) 09-30): SKIPPED - newest HR2-B $(basename $R) has $N act seats < 720 (rule: +10% over 654)" >> $LEDGER; exit 0; }
NAME=rows-n-${SHA:0:12}.tsv; cp $R $M/$NAME
python3 - $R $M/x1-pool2.manifest <<'PY'
import csv, os, sys
V='/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11'
ns=list(dict.fromkeys(f"{r['eid']}-{r['seat']}.stp" for r in csv.DictReader(open(sys.argv[1]),delimiter='\t')))
open(sys.argv[2],'w').write(''.join(f"{n} {os.path.getsize(f'{V}/{n}')}\n" for n in ns if os.path.exists(f'{V}/{n}')))
PY
U=$(cat $O/.tunnel-url)/$(cat $O/.serve-token)
for i in $(seq 1 20); do out=$(timeout 120 $H/vm.sh nflow-x1 "cd /content/bcache && curl -sf -o $NAME '$U/meta/$NAME' && (setsid nohup nice -n 5 python3 /content/fetch_root.py '$U' v11 x1-pool2.manifest /content/bcache --threads 16 > /content/fetch-x1pool2.log 2>&1 < /dev/null &) && echo started" < /dev/null 2>&1) && grep -q started <<< "$out" && break; sleep 8; done
echo "R-NB-RESWITCH-START ($(date +%H:%M) 09-30): N B-even re-switch to $(basename $R) (sha ${SHA:0:12}, $N act seats >= 720); seats staged on nflow-x1 while training; stop only after the newest checkpoint is on the NAS" >> $LEDGER
bash $L/nswitch-x1.sh $NAME $SHA x1r$(date +%H%M)
