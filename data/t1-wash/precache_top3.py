#!/usr/bin/env python3
"""R-PRECACHE-TOP3 (owner 17:3x "anticipating future demand"): both seats of every local 09-30 game of the top-3 teams' OTHER live
submissions (live pair minus the pinned one) -> v11 cache queue (broad/t1/wash/PRECACHE-TOP3.tsv). Not in any pool share."""
import glob, json, os, sys
sys.path.insert(0, '/mnt/ai/scratch/farmNEAT/broad'); import build_t1 as B1
E = '/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra'; C = '/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11'
PINS = {16681125: 56701470, 16621497: 56704120, 16623559: 56705174}
ok = lambda e, z: os.path.exists(f'{C}/{e}-{z}.stp') and os.path.getsize(f'{C}/{e}-{z}.stp') > 0
rows, stat = [], {}
EXTRA = {16714162: {56712801}, 16628664: None, 16736838: None, 16736435: None}   # owner 17:5x-18:0x: MarvinTMB best only; Anton, Anaconda, CDE all live subs
for tid, pin in list(PINS.items()) + list(EXTRA.items()):
    d = f"{E}/{B1.SLUG.get(tid, f'copy-team-{tid}')}"
    _, _, _, live = B1.listing_subs(tid)
    other = sorted(set(live) - {pin}) if tid in PINS else sorted(pin if pin else set(live))
    L = {}
    for f in sorted(glob.glob(f'{d}/LISTINGS-*.json'), key=os.path.getmtime):
        for e in json.load(open(f)).get('episodes', []): L[int(e['episode_id'])] = e
    man = {}
    for mf in glob.glob(f'{E}/*/MANIFEST.jsonl'):
        for l in open(mf):
            if l.strip():
                r = json.loads(l)
                if int(r['episode_id']) in L: man[int(r['episode_id'])] = (os.path.join(os.path.dirname(mf), r['archive']), r['member'])
    for sub in other:
        eps = [e for e in L.values() if sub in (e.get('sub_ids') or []) and e.get('create_time', '')[:10] == '2026-09-30' and e.get('state') == 'COMPLETED']
        local = [e for e in eps if int(e['episode_id']) in man]
        wins = sum(1 for e in local if None not in e['rewards'] and e['rewards'][e['sub_ids'].index(sub)] > e['rewards'][1 - e['sub_ids'].index(sub)])
        unc = 0
        for e in local:
            a, m = man[int(e['episode_id'])]
            for z in (0, 1):
                if not ok(int(e['episode_id']), z): rows.append(f"{a}\t{m}\t2026-09-30\t{int(e['episode_id'])}\t{z}\t1\tprecache-top3:{sub}"); unc += 1
        stat[f'{tid}:{sub}'] = {'listed_0930': len(eps), 'local': len(local), 'wins_local': wins, 'seats_to_cache': unc, 'score': live.get(sub)}
out = '/mnt/ai/scratch/farmNEAT/broad/t1/wash/PRECACHE-TOP3.tsv'
open(out, 'w').write('archive\tmember\tday\teid\tseat\tweight\treason\n' + '\n'.join(rows) + ('\n' if rows else ''))
print(json.dumps(stat))
