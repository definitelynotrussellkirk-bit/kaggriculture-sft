#!/usr/bin/env python3
"""Coverage index of the v11 superset step cache: every catalog seat with day >= 2026-09-27 plus the manual/topteam batches.
Writes base-v1-stepcache-v11/CACHE-COVERAGE.tsv (eid, seat, day, source, team, team_id, submission_id, result, kaggle,
tsv_ready, split, built) and prints counts.   /usr/bin/python3 coverage.py   (run from anywhere)"""
import csv, os, sys
import pandas as pd
sys.path[:0] = ['/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc',
                '/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927/src']
import data as T
C = '/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11'
CAT = '/mnt/ai/scratch/farmNEAT/primitive-bc-data/catalog/SEATS.parquet'
built = set(f[:-4] for f in os.listdir(C) if f.endswith('.stp'))
s = pd.read_parquet(CAT, columns=['episode_id', 'seat', 'day', 'source', 'team', 'team_id', 'submission_id', 'result', 'kaggle', 'tsv_ready'])
d = s[s.day.astype(str) >= '2026-09-27'].copy()
extra = []
for r in csv.DictReader(open('/home/russ/Desktop/farmNEAT/work/runs/maintainer/top8-manual-0930/MANUAL-SEATS-ALL.tsv'), delimiter='\t'):
    extra.append({'episode_id': int(r['episode_id']), 'seat': int(r['seat']), 'day': r['verified_game_day'] or 'unknown',
                  'source': 'manual-mmpq-111', 'team': r['team_name'], 'team_id': r['team_id'] or None,
                  'submission_id': r['submission_id'] or None, 'result': {'win': 'W', 'loss': 'L'}.get(r['outcome'], r['outcome']),
                  'kaggle': True, 'tsv_ready': True})
d = pd.concat([d, pd.DataFrame(extra)], ignore_index=True)
d['key'] = d.episode_id.astype('int64').astype(str) + '-' + d.seat.astype(int).astype(str)
d = d.drop_duplicates('key')
d['built'] = d.key.isin(built).astype(int)
d['split'] = [('train' if T.in_split(int(e), 'train') else 'val' if T.in_split(int(e), 'val') else 'test') for e in d.episode_id]
cols = ['episode_id', 'seat', 'day', 'source', 'team', 'team_id', 'submission_id', 'result', 'kaggle', 'tsv_ready', 'split', 'built']
tmp = C + '/CACHE-COVERAGE.tsv.tmp'
d[cols].to_csv(tmp, sep='\t', index=False)
os.replace(tmp, C + '/CACHE-COVERAGE.tsv')
k = d.kaggle == True
print(f"seats {len(d)} built {int(d.built.sum())} | kaggle {int(k.sum())} built {int(d[k].built.sum())} | "
      f"kaggle W {int((k & (d.result == 'W')).sum())} built {int(d[k & (d.result == 'W')].built.sum())} | "
      f"non-kaggle {int((~k).sum())} built {int(d[~k].built.sum())} | not tsv_ready {int((d.tsv_ready != True).sum())}")
