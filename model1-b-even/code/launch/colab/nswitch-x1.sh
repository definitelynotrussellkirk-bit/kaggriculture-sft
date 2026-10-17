#!/bin/bash
# Roster switch for N B-even on Colab #1: wait until every seat of the staged roster is on the VM, then until the trainer's next checkpoint
# lands, stop train.py there, relaunch the same run from that checkpoint on the staged roster (n-launch.sh), and confirm the first update.
#   nswitch.sh ROWS_NAME ROWS_SHA256 TAG        e.g. nswitch.sh rows-n-d4623759c8fe.tsv d4623759... 1255
set -uo pipefail
ROWS_NAME=$1; SHA=$2; TAG=$3
S=nflow-x1; N=n1-beven-colab; O=/content/results/$N
H=/home/russ/Desktop/farmNEAT/work/runs/maintainer/oldarch-scaleup/colab; L=/mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/launch-ncolab
D=/mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/colab-$N; LOG=$L/SWITCH-$TAG.log
LEDGER=/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
vm() { local i out; for i in $(seq 1 20); do out=$(timeout 120 "$H/vm.sh" "$S" "$1" < /dev/null 2>&1) && { echo "$out"; return 0; }; sleep 6; done; echo "$out"; return 1; }
cachecheck="cd /content/bcache && python3 -c \"
import csv,hashlib,os
raw=open('$ROWS_NAME','rb').read(); assert hashlib.sha256(raw).hexdigest()=='$SHA', 'sha'
r=list(csv.DictReader(raw.decode().splitlines(),delimiter='\t'))
m=[x for x in r if not (os.path.isfile(x['eid']+'-'+x['seat']+'.stp') and os.path.getsize(x['eid']+'-'+x['seat']+'.stp')>0)]
print('missing',len(m),'of',len(r))\""
until r=$(vm "$cachecheck") && grep -q "^missing 0 of" <<< "$r"; do log "waiting for seats: $(tail -1 <<< "$r")"; sleep 30; done
log "roster $ROWS_NAME ($SHA) fully cached: $(tail -1 <<< "$r")"
l0=$(vm "cat $O/LATEST" | tail -1); log "current LATEST $l0; waiting for the next checkpoint"
until l1=$(vm "cat $O/LATEST" | tail -1) && [[ $l1 =~ ^ckpt-[0-9]+\.pt$ ]] && { [ "$l1" != "$l0" ] || { [ -n "${CKPT_DEADLINE:-}" ] && [ "$(date +%H%M)" -ge "$CKPT_DEADLINE" ]; }; }; do sleep 20; done
[ "$l1" = "$l0" ] && log "no new checkpoint by $CKPT_DEADLINE; stopping at $l1 (already on the NAS)"
# coordinator 16:5x: a runtime loss after the stop must cost nothing -> stop only once $l1 is pulled and sha-listed on the NAS
until grep -q " $l1\$" "$D/SHA256SUMS" 2>/dev/null && [ -s "$D/$l1" ]; do sleep 10; done
log "$l1 is on the NAS ($(grep " $l1\$" "$D/SHA256SUMS" | tail -1 | cut -c1-16)); stopping train.py"
T_STOP=$(date +%s)
r=$(vm "p=\$(python3 -c 'import json;print(json.load(open(\"$O/HEARTBEAT.json\"))[\"pid\"])'); ps -o pid,args -p \$p | tail -1 | cut -c1-80; c=\$(pgrep -P \$p | tr '\\n' ' '); kill \$p; for i in \$(seq 60); do kill -0 \$p 2>/dev/null || break; sleep 2; done; kill -0 \$p 2>/dev/null && echo alive || echo stopped; sleep 5; [ -n \"\$c\" ] && kill \$c 2>/dev/null; sleep 3; [ -n \"\$c\" ] && kill -9 \$c 2>/dev/null; cat $O/LATEST")
log "stop: $(tr '\n' ' ' <<< "$r")"
grep -q stopped <<< "$r" || { log "train.py did not stop"; exit 2; }
lat=$(tail -1 <<< "$r")
for i in $(seq 1 30); do pkill -0 -F "$L/exec.pid" 2>/dev/null || break; sleep 5; done
pkill -0 -F "$L/exec.pid" 2>/dev/null && kill "$(cat "$L/exec.pid")"
systemctl --user stop nx1-pull 2>/dev/null; [ -f "$D/pull.pid" ] && kill "$(cat "$D/pull.pid")" 2>/dev/null
vm "rm -f $O/DONE.json; echo cleared" > /dev/null; rm -f "$D/DONE.json"
u0=$((10#$(sed 's/ckpt-//; s/\.pt//' <<< "$lat")))
last=$(vm "grep -o '\"update\": [0-9]*' $O/metrics.jsonl | tail -1" | tail -1 | grep -o '[0-9]*$')
log "relaunch from $lat (u$u0); old trainer's last logged update $last"
ROWS_NAME=$ROWS_NAME EXPECTED_ROWS_SHA=$SHA bash "$L/n-launch-x1.sh" 2026-09-30T20:10:00-04:00 16 >> "$LOG" 2>&1 || { log "n-launch failed"; exit 3; }
base=${last:-$u0}
until r=$(vm "grep -o '\"update\": [0-9]*' $O/metrics.jsonl | tail -1; grep -cE 'Traceback|Error' $O/train.log; true") && u=$(head -1 <<< "$r" | grep -o '[0-9]*$') && [ -n "$u" ] && [ "$u" -gt "$base" ]; do
  grep -q '"returncode": [0-9]' <<< "$(vm "cat $O/HEARTBEAT.json")" && { log "trainer exited before its first update"; exit 4; }
  sleep 10; done
DT=$(( $(date +%s) - T_STOP ))
dr=$(vm "cat $O/draws/*.log 2>/dev/null | awk -v t=\$(stat -c %Y $O/COMMAND.json) '\$1>t{print \$2}' | sort -u > /tmp/drawn.txt; python3 -c \"
import csv
new=set(x['eid']+'-'+x['seat']+'.stp' for x in csv.DictReader(open('/content/bcache/$ROWS_NAME'),delimiter='\t'))
d=[l.strip() for l in open('/tmp/drawn.txt')]
print('drawn',len(d),'in-roster',sum(x in new for x in d))\"")
log "FIRST-UPDATE u$u (> $base) on $ROWS_NAME; downtime ${DT} s; draws since relaunch: $(tail -1 <<< "$dr")"
echo "R-POOL-SWITCH-NB R-NBEVEN-SWITCH-$TAG ($(date +%H:%M) 09-30): N B-even on Colab #1 switched to roster $ROWS_NAME (sha ${SHA:0:12}) from $lat; old trainer's last logged update $last; first update on the new roster u$u; downtime ${DT} s (stop to first update seen, 30 s polling); draws since relaunch: $(tail -1 <<< "$dr")" >> "$LEDGER"
