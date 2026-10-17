#!/bin/bash
H=/home/russ/Desktop/farmNEAT/work/runs/maintainer/oldarch-scaleup/colab; L=/mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/launch-ncolab; O=/content/results/n1-beven-colab
LEDGER=/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md; LOG=$L/RELAUNCH-X1.log; exec >> $LOG 2>&1
SHA=a9ab27e1ba447b857a90a3b3654b41de6aa67008ab8c028f50ea1f9a7129d5bb; RN=rows-n-a9ab27e1ba44.tsv
chk="cd /content/bcache && python3 -c \"
import csv,hashlib,os
raw=open('$RN','rb').read(); assert hashlib.sha256(raw).hexdigest()=='$SHA'
r=list(csv.DictReader(raw.decode().splitlines(),delimiter='\t'))
m=[x for x in r if not (os.path.isfile(x['eid']+'-'+x['seat']+'.stp') and os.path.getsize(x['eid']+'-'+x['seat']+'.stp')>0)]
print('missing',len(m),'of',len(r))\"; cat $O/LATEST; ls /content/parts/n1b/SEED_NAME"
until r=$(timeout 60 $H/vm.sh nflow-x1 "$chk" < /dev/null 2>&1) && grep -q "^missing 0 of" <<< "$r" && grep -q "ckpt-0045920.pt" <<< "$r"; do echo "$(date +%T) $(tr '\n' ' ' <<< "$r" | cut -c1-150)"; sleep 10; done
echo "$(date +%T) staged: $(head -1 <<< "$r")"
cd /tmp && SKIP_PRE=1 ROWS_NAME=$RN EXPECTED_ROWS_SHA=$SHA bash $L/n-launch-x1.sh 2026-09-30T20:10:00-04:00 16 || { echo "n-launch-x1 failed"; exit 1; }
until r=$(timeout 60 $H/vm.sh nflow-x1 "grep -o '\"update\": [0-9]*' $O/metrics.jsonl 2>/dev/null | tail -1; true" < /dev/null 2>&1) && u=$(grep -o '"update": [0-9]*' <<< "$r" | grep -o '[0-9]*$') && [ -n "$u" ] && [ "$u" -gt 45920 ]; do sleep 15; done
echo "R-TRAIN-UP-NB ($(date +%H:%M) 09-30): N B-even training again, now on Colab #3 (nflow-x1, G4), after oldarch-g4 was recycled at 16:36 during the pool re-switch (down since 16:36:27). Resumed from ckpt-0045920 (sha cc45bc83, NAS) on the pool roster rows-t1wash-b.a9ab27e1ba44 (654 seats, 3 BEAT1). R-POOL-SWITCH-NB: sha a9ab27e1ba44, 654 seats, first update u$u. 8x, EMA .992, stop 20:05 EDT. Isolated dirs /content/bcode and /content/bcache on nflow-x1. Pull unit nx1-pull -> pbc-colab-v6-results/colab-n1-beven-colab (same NAS dir); keepalive nx1-keepalive" >> $LEDGER
echo "$(date +%T) first update $u"
