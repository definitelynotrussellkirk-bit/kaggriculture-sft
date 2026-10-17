#!/bin/bash
H=/home/russ/Desktop/farmNEAT/work/runs/maintainer/oldarch-scaleup/colab; L=/mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/launch-ncolab; O=/content/results/n1-beven-colab
LEDGER=/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md; LOG=$L/RELAUNCH-1530.log
exec >> $LOG 2>&1
until r=$(timeout 60 $H/vm.sh oldarch-g4 'tail -c 300 /content/fetch-rebuild.log' < /dev/null 2>&1) && grep -q '"done": true' <<< "$r"; do sleep 20; done
echo "$(date +%T) fetch done: $(tail -1 <<< "$r")"
cd /tmp && ROWS_NAME=rows-n-60327cd812e7.tsv EXPECTED_ROWS_SHA=60327cd812e7852b557ce9405b875436640108c723362254fc9beeba610c26b4 bash $L/n-launch.sh 2026-09-30T20:10:00-04:00 16 || { echo "n-launch failed"; exit 1; }
until r=$(timeout 60 $H/vm.sh oldarch-g4 "grep -o '\"update\": [0-9]*' $O/metrics.jsonl 2>/dev/null | tail -1; grep -cE 'Traceback' $O/train.log; true" < /dev/null 2>&1) && u=$(head -1 <<< "$r" | grep -o '[0-9]*$') && [ -n "$u" ] && [ "$u" -gt 45680 ]; do sleep 15; done
echo "R-TRAIN-UP ($(date +%H:%M) 09-30): N B-even training again on the fresh Colab #1 runtime (gpu-g4-s-kkb-usw4a1). Resumed from ckpt-0045680 (sha 983c9315) on roster rows-n-60327cd812e7 (generation b1c73e484343, LINEAGE=B, 3,862 seats all local). Config: 8x (batch-windows 1024), EMA .992 with extras .984/.996, seat fraction .25, cache-pool-seats 24, ckpt every 40, max-seconds to 20:05 EDT. First update u$u. Down 15:03 to $(date +%H:%M); 40 updates lost (45680 -> 45720)" >> $LEDGER
echo "$(date +%T) first update $u"
