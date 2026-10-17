#!/usr/bin/env python3
"""Release a scored surprise batch: per game R-SURPRISE ledger line with each checkpoint's perplexity; eids -> surprise/DONE.json."""
import json, math, os, sys
T = sys.argv[1]; S = '/mnt/ai/scratch/farmNEAT/broad/surprise'
names = {'N wash': sys.argv[2], 'N keep': sys.argv[3], 'B wash': sys.argv[4], 'B keep': sys.argv[5]}
per = {}
for f, labs in ((f'{S}/N-{T}.json', ['N wash', 'N keep']), (f'{S}/B-{T}.json', ['B wash', 'B keep'])):
    res = json.load(open(f))['results']
    for lab, r in zip(labs, res):
        for p in r['per_game']:
            per.setdefault(int(p['episode_id']), {})[lab] = (os.path.basename(r['checkpoint']), round(p['ppl'], 3), round(p['acc_step_exact'], 3))
done_p = f'{S}/DONE.json'; done = set(json.load(open(done_p))) if os.path.exists(done_p) else set()
games = {int(g['episode_id']): g for g in json.load(open(f'{S}/BATCH-{T}.json'))['games']}
with open('/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md', 'a') as L:
    for e, v in sorted(per.items()):
        g = games[e]
        L.write(f"- R-SURPRISE {e} ({g['create_time'][11:16]}Z, {' v '.join(g['team_names'])}, MMPQ seat {g['mmpq_seat']}) ppl/step-exact: "
                + '; '.join(f"{k} {v[k][0]} {v[k][1]}/{v[k][2]}" for k in ('N wash', 'N keep', 'B wash', 'B keep') if k in v) + ' (B every 2nd window)\n')
done |= set(per)
json.dump(sorted(done), open(done_p + '.tmp', 'w')); os.replace(done_p + '.tmp', done_p)
print(len(per))
