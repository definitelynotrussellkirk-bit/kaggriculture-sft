#!/bin/bash
# The single remaining N B-even re-switch (owner "when there's only 1 hour left, we go to 80-10-10"; coordinator 17:0x): from 18:05 wait (to 18:14)
# for data-ops' 80/10/10 HR2-B (receipt target_share: first-place buckets T1W*+BEAT1 >= 0.75), stage roster + every seat on nflow-x1 while
# training, then nswitch-x1.sh: next checkpoint (or, at 18:15, the current one) pulled to the NAS first -> stop -> relaunch -> first update.
# The 18:25 guard (nb_guard_1825.sh) relaunches on the old pool file if B is not up by then.
set -uo pipefail
L=/mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/launch-ncolab; W=/mnt/ai/scratch/farmNEAT/broad/t1/wash; M=/mnt/ai/scratch/farmNEAT/oldarch-scaleup/meta
O=/mnt/ai/scratch/farmNEAT/oldarch-scaleup; H=/home/russ/Desktop/farmNEAT/work/runs/maintainer/oldarch-scaleup/colab
LEDGER=/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md
first_share() { python3 -c "
import json,sys; d=json.load(open(sys.argv[1]+'.receipt.json')); t=d.get('target_share') or {}
print(round(sum(v for k,v in t.items() if k.startswith('T1W') or k.startswith('BEAT1')),4))" "$1" 2>/dev/null || echo 0; }
ok8014() { python3 -c "
import json,sys; t=json.load(open(sys.argv[1]+'.receipt.json')).get('target_share') or {}
f=sum(v for k,v in t.items() if k.startswith('T1W') or k.startswith('BEAT1'))
sys.exit(0 if f>=0.75 and abs(t.get('T2W',0)-0.14)<0.005 and abs(t.get('T3W',0)-0.06)<0.005 else 1)" "$1" 2>/dev/null; }   # owner 18:0x: 80/14/6 (DECEM T2W .14, Victor T3W .06)
until R=$(readlink -f $W/HR2-B) && ok8014 $R; do
  [ "$(date +%H%M)" -ge 1820 ] && { echo "R-NB-RESWITCH ($(date +%H:%M) 09-30): SKIPPED - no 80/14/6 HR2-B by 18:20 (newest $(basename $R), first-place share $(first_share $R)); N B-even stays on a9ab27e1ba44" >> $LEDGER; exit 0; }
  sleep 20
done
SHA=$(sha256sum $R | cut -d' ' -f1); N=$(( $(wc -l < $R) - 1 )); NAME=rows-n-${SHA:0:12}.tsv; cp $R $M/$NAME
python3 - $R $M/x1-pool8.manifest <<'PY'
import csv, os, sys
V='/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11'
ns=list(dict.fromkeys(f"{r['eid']}-{r['seat']}.stp" for r in csv.DictReader(open(sys.argv[1]),delimiter='\t')))
open(sys.argv[2],'w').write(''.join(f"{n} {os.path.getsize(f'{V}/{n}')}\n" for n in ns if os.path.exists(f'{V}/{n}')))
PY
U=$(cat $O/.tunnel-url)/$(cat $O/.serve-token)
for i in $(seq 1 20); do out=$(timeout 120 $H/vm.sh nflow-x1 "cd /content/bcache && curl -sf -o $NAME '$U/meta/$NAME' && (setsid nohup nice -n 5 python3 /content/fetch_root.py '$U' v11 x1-pool8.manifest /content/bcache --threads 16 > /content/fetch-x1pool8.log 2>&1 < /dev/null &) && echo started" < /dev/null 2>&1) && grep -q started <<< "$out" && break; sleep 8; done
echo "R-NB-RESWITCH-START ($(date +%H:%M) 09-30): N B-even re-switch to the 80/14/6 pool $(basename $R) (sha ${SHA:0:12}, $N rows, first-place share $(first_share $R)); seats staged on nflow-x1 while training; stop only after the newest checkpoint is on the NAS (no new one by 18:15 -> current one)" >> $LEDGER
echo "$NAME $SHA" > $L/.pool8-target
CKPT_DEADLINE=1818 bash $L/nswitch-x1.sh $NAME $SHA x1p8
