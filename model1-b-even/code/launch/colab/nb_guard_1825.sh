#!/bin/bash
# 18:25 guard (coordinator 17:0x): N B-even must be training by 18:25. If train.py on nflow-x1 is not running (or has made no update after the
# re-switch's stop), relaunch at once on the OLD pool file rows-n-a9ab27e1ba44.tsv (all seats already on the VM) from the run's LATEST.
set -uo pipefail
L=/mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/launch-ncolab; H=/home/russ/Desktop/farmNEAT/work/runs/maintainer/oldarch-scaleup/colab; O=/content/results/n1-beven-colab
LEDGER=/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md; LOG=$L/GUARD-1825.log; exec >> $LOG 2>&1
OLD=rows-n-a9ab27e1ba44.tsv; OLDSHA=a9ab27e1ba447b857a90a3b3654b41de6aa67008ab8c028f50ea1f9a7129d5bb
r=$(timeout 90 $H/vm.sh nflow-x1 "pgrep -fc '[t]rain.py fit --out $O' || true; grep -o '\"update\": [0-9]*' $O/metrics.jsonl | tail -1; cat $O/LATEST; stat -c %Y $O/metrics.jsonl; date +%s" < /dev/null 2>&1)
echo "$(date +%T) $(tr '\n' ' ' <<< "$r")"
if grep -q "HTTP 404\|not found\|No such file" <<< "$r"; then
  echo "R-TRAIN-DOWN-NB ($(date +%H:%M) 09-30): 18:25 guard: nflow-x1 runtime unreachable/recycled ($(tail -1 <<< "$r" | cut -c1-120)); relaunch from the newest NAS checkpoint on a fresh G4 needed NOW (see CODEX-NEXT-TASK 16:57 block)" >> $LEDGER; exit 1
fi
st() { timeout 90 $H/vm.sh nflow-x1 "pgrep -fc '[t]rain.py fit --out $O' || true; grep -o '\"update\": [0-9]*' $O/metrics.jsonl | tail -1; cat $O/LATEST; stat -c %Y $O/metrics.jsonl; date +%s" < /dev/null 2>&1; }
np=$(sed -n 1p <<< "$r"); age=$(( $(sed -n 5p <<< "$r") - $(sed -n 4p <<< "$r") ))
if [ "${np:-0}" -ge 1 ] && [ "$age" -lt 240 ]; then echo "$(date +%T) B is training (metrics written ${age}s ago)"; exit 0; fi
# a relaunch may be in progress: give a running train.py up to 6 min to log, and a pending launch 2 min to start; never start a second trainer
for i in $(seq 1 24); do
  sleep 15; r=$(st); np=$(sed -n 1p <<< "$r"); age=$(( $(sed -n 5p <<< "$r") - $(sed -n 4p <<< "$r") ))
  [ "${np:-0}" -ge 1 ] && [ "$age" -lt 240 ] && { echo "$(date +%T) B is training (metrics written ${age}s ago)"; exit 0; }
  [ "${np:-0}" -eq 0 ] && [ $i -ge 8 ] && break
done
if [ "${np:-0}" -ge 1 ]; then echo "R-NB-GUARD-1825 ($(date +%H:%M) 09-30): ALERT - train.py runs on nflow-x1 but metrics.jsonl is ${age}s old; not starting a second trainer; check now" >> $LEDGER; exit 1; fi
for p in $(pgrep -f "[n]switch-x1.sh") $(pgrep -f "[n]b_reswitch_801010.sh"); do kill $p 2>/dev/null; done
echo "R-NB-GUARD-1825 ($(date +%H:%M) 09-30): N B-even not training at 18:25 (train.py count ${np:-?}, metrics age ${age}s); re-switch aborted; relaunching on the old pool $OLD from $(sed -n 3p <<< "$r")" >> $LEDGER
cd /tmp && SKIP_PRE=1 ROWS_NAME=$OLD EXPECTED_ROWS_SHA=$OLDSHA bash $L/n-launch-x1.sh 2026-09-30T20:10:00-04:00 16
until r=$(timeout 60 $H/vm.sh nflow-x1 "stat -c %Y $O/metrics.jsonl; date +%s; grep -o '\"update\": [0-9]*' $O/metrics.jsonl | tail -1" < /dev/null 2>&1) && [ $(( $(sed -n 2p <<< "$r") - $(sed -n 1p <<< "$r") )) -lt 60 ]; do sleep 15; done
echo "R-TRAIN-UP-NB ($(date +%H:%M) 09-30): 18:25 guard relaunch on $OLD; training again ($(sed -n 3p <<< "$r"))" >> $LEDGER
