#!/bin/bash
# Rolling "mini surprise" (owner 15:4x D-1544ho): every new local MMPQ game (newer than VAL-MMPQ-NEXT, not yet scored) stays out
# of every build until its MMPQ-seat perplexity is scored under the newest READY N flow wash, the newest N flow keep, and the B
# equivalents (newest B-family wash run, newest N B-even keep). Then it is added to surprise/DONE.json (released to builds) and
# logged as R-SURPRISE <eid>. Timer every 15 min until 19:00; one SLURM job per batch on cpu-hub.
[ "$(date +%H%M)" -le 1900 ] || exit 0
BR=/mnt/ai/scratch/farmNEAT/broad; S=$BR/surprise; cd $BR || exit 1
squeue -h -n surprise | grep -q . && exit 0                       # one batch at a time
T=$(date +%H%M%S)
python3 - "$S/BATCH-$T.json" "$S/BATCH-$T.rows.tsv" <<'PY' || exit 0
import json, sys, glob, os
sys.path.insert(0, '/mnt/ai/scratch/farmNEAT/broad'); import exclusion_union as XU
pend = sorted(XU.surprise_pending())
if not pend: sys.exit(1)
E = '/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/copy-mmpq'
man = {int(json.loads(l)['episode_id']): json.loads(l) for l in open(f'{E}/MANIFEST.jsonl') if l.strip()}
L = {}
for f in sorted(glob.glob(f'{E}/LISTINGS-*.json'), key=os.path.getmtime)[-3:]:
    for e in json.load(open(f))['episodes']: L[int(e['episode_id'])] = e
games = []
for e in pend:
    m, l = man[e], L[e]
    if 16681125 not in l['team_ids'] or l.get('state') != 'COMPLETED': continue
    z = l['team_ids'].index(16681125) if l['team_ids'].count(16681125) == 1 else (0 if l['rewards'][0] >= l['rewards'][1] else 1)
    games.append({'episode_id': e, 'mmpq_seat': z, 'seat': z, 'rewards': l['rewards'], 'team_names': l['team_names'], 'create_time': l['create_time'],
                  'archive': f"{E}/{m['archive']}", 'member': m['member']})
if not games: sys.exit(1)
json.dump({'games': games}, open(sys.argv[1], 'w'), indent=1)
open(sys.argv[2], 'w').write('archive\tmember\tday\teid\tseat\tweight\treason\n' + ''.join(
    f"{g['archive']}\t{g['member']}\t2026-09-30\t{g['episode_id']}\t{g['mmpq_seat']}\t1\tsurprise\n" for g in games))
print(len(games))
PY
RN=$(python3 -c "import json;print(json.load(open('/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/READY-NOW.json'))['flow']['name'])")
NW=/mnt/ai/scratch/farmNEAT/n-flow/runs/$RN/$(cat /mnt/ai/scratch/farmNEAT/n-flow/runs/$RN/LATEST 2>/dev/null)
[ -f "$NW" ] || NW=$(ls -t /mnt/ai/scratch/farmNEAT/n-flow/release-ckpts/$RN/*.pt 2>/dev/null | head -1)
NK=$(ls -t /mnt/ai/scratch/farmNEAT/n-flow/keeps/ckpt-*.pt | head -1)
BW=$(ls -t /mnt/ai/scratch/farmNEAT/t4w/runs/*-t1w/ckpt-*.pt 2>/dev/null | head -1)
BK=$(ls -t /mnt/ai/scratch/farmNEAT/hourly/dense/n1/n1-u*.pt | head -1)
RT="/mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12 -B"; PP=/mnt/ai/scratch/farmNEAT/corpus-final/site:/mnt/ai/scratch/farmNEAT/runtime/site
R=/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927
sbatch --parsable -J surprise -p cpu -w cpu-hub -c 2 --mem 9G -t 90 -o $S/job-$T-%j.out --export=ALL,VAL_SET=$S/BATCH-$T.json,WINDOW_EVERY=2 --wrap "
set -u
cd $R/work/runs/maintainer/primitive-bc && PYTHONPATH=$R/work/runs/maintainer/primitive-bc:$R/src:/mnt/ai/scratch/farmNEAT/runtime/site $RT step_cache.py build $S/BATCH-$T.rows.tsv /mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11 --procs 1 --v6 --v8
cd $BR && PYTHONPATH=$PP $RT val_mmpq_n.py $S/N-$T.json $NW $NK && PYTHONPATH=$PP $RT val_mmpq_b.py $S/B-$T.json ${BW:-$BK} $BK && python3 $BR/surprise_done.py $T '$NW' '$NK' '${BW:-none}' '$BK'"
