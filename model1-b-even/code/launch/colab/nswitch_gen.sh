#!/bin/bash
# N B-even (Colab #1) follows N flow's roster switches (coordinator 12:4x: both trainers switch together at 14:55/16:55): one minute after
# N flow's switch, take broad/LATEST-BUCKETS, build the LINEAGE=B roster (roster_for.sh: same generation and neutral shares, W_B instead of
# W_N), fetch the seats Colab #1 lacks (nice 19, NAS tunnel), stage the roster beside the live one, then nswitch.sh (next checkpoint ->
# relaunch -> first update). Skips when the generation is unchanged or after 17:15 (N flow's no-switch margin).
set -uo pipefail
L=/mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/launch-ncolab; B=/mnt/ai/scratch/farmNEAT/broad; M=/mnt/ai/scratch/farmNEAT/oldarch-scaleup/meta
H=/home/russ/Desktop/farmNEAT/work/runs/maintainer/oldarch-scaleup/colab; O=/mnt/ai/scratch/farmNEAT/oldarch-scaleup; V=/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11
LEDGER=/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md; TAG=$(date +%H%M); LOG=$L/SWITCHGEN-$TAG.log; S=oldarch-g4
exec >> $LOG 2>&1
vm() { local i out; for i in $(seq 1 20); do out=$(timeout 120 "$H/vm.sh" "$S" "$1" < /dev/null 2>&1) && { echo "$out"; return 0; }; sleep 6; done; echo "$out"; return 1; }
G=$(basename "$(readlink -f $B/LATEST-BUCKETS)"); CUR=$(cat $L/CURRENT-GEN 2>/dev/null)
echo "$(date -Is) latest $G current $CUR"
[ "$G" = "$CUR" ] && { echo "R-NBEVEN-SWITCH-$TAG ($(date +%H:%M) 09-30): SKIPPED - generation $G unchanged" >> $LEDGER; exit 0; }
t=$(date +%H%M); [ "$t" -ge 1715 ] && [ "$t" -lt 1831 ] && { echo "R-NBEVEN-SWITCH-$TAG ($(date +%H:%M) 09-30): SKIPPED - 17:15-18:30 no-switch window" >> $LEDGER; exit 0; }
R=$(cd /tmp && LINEAGE=B nice -n 19 bash $B/roster_for.sh) && [ -s "$R" ] || { echo "R-NBEVEN-SWITCH-$TAG ($(date +%H:%M) 09-30): SKIPPED - roster_for refused for $G" >> $LEDGER; exit 0; }
SHA=$(sha256sum $R | cut -d' ' -f1); NAME=rows-n-${SHA:0:12}.tsv
for i in 1 2 3 4 5; do timeout 300 nice -n 19 $HOME/.local/bin/colab upload -s $S $R /content/cache/$NAME && break; sleep 10; done
vm "sha256sum /content/cache/$NAME" | grep -q "^$SHA " || { echo "R-NBEVEN-SWITCH-$TAG ($(date +%H:%M) 09-30): SKIPPED - roster upload failed" >> $LEDGER; exit 0; }
# prune before fetching (coordinator 13:3x): drop cached seats used by neither the running roster nor the new one
CURROWS=$(grep -o "'PBC_CACHE_ROWS': '/content/cache/[^']*'" $L/worker.py | sed "s#.*/content/cache/##; s#'##")
[ -n "$CURROWS" ] && vm "cd /content/cache && python3 -c \"
import csv,os
keep=set()
for f in ('$CURROWS','$NAME'):
    if os.path.exists(f): keep|={x['eid']+'-'+x['seat']+'.stp' for x in csv.DictReader(open(f),delimiter='\\t')}
have=[n for n in os.listdir('.') if n.endswith('.stp')]
drop=[n for n in have if n not in keep]
assert len(keep)>10000, len(keep)
[os.remove(n) for n in drop]
print('prune: keep',len(keep),'had',len(have),'removed',len(drop))\"; df -BG /content | tail -1"
vm "ls /content/cache" > $L/.cache-$TAG.txt
python3 - $R $L/.cache-$TAG.txt $V $M/gen-${SHA:0:12}-c1-missing.manifest <<'PY' || exit 1
import csv,os,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); have=set(l.strip() for l in open(sys.argv[2]))
miss=sorted({f"{r['eid']}-{r['seat']}.stp" for r in rows} - have)
open(sys.argv[4],'w').write(''.join(f"{n} {os.path.getsize(os.path.join(sys.argv[3],n))}\n" for n in miss))
print('rows',len(rows),'missing',len(miss))
PY
U=$(cat $O/.tunnel-url)/$(cat $O/.serve-token)
[ -s $M/gen-${SHA:0:12}-c1-missing.manifest ] && vm "cd /content && (nohup nice -n 19 python3 /content/fetch_root.py '$U' v11 gen-${SHA:0:12}-c1-missing.manifest /content/cache --threads 8 > /content/fetch-${SHA:0:12}.log 2>&1 < /dev/null &); echo fetch started"
# ADV-CHECK-1600 #2: N B-even switches on its own gate (its B roster of the new generation), independent of N flow's outcome
bash $L/nswitch.sh $NAME $SHA $TAG && echo $G > $L/CURRENT-GEN
