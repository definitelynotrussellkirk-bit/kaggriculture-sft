#!/usr/bin/env python3
"""Rows (trainer TSV) for every catalog seat with day >= 2026-09-27, tsv_ready, not yet built and not already listed in
backfill/ALL.tsv (the running backfill). Priority: Kaggle W newest first, Kaggle L, then non-Kaggle. Prints the row count."""
import os, sys
import pandas as pd
C = '/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11'; B = C + '/backfill'
out = sys.argv[1]
built = set(f[:-4] for f in os.listdir(C) if f.endswith('.stp'))
listed = set()
for line in open(B + '/ALL2.tsv').read().splitlines()[1:]:
    p = line.split('\t'); listed.add(f'{p[3]}-{p[4]}')
s = pd.read_parquet('/mnt/ai/scratch/farmNEAT/primitive-bc-data/catalog/SEATS.parquet',
                    columns=['episode_id', 'seat', 'day', 'kaggle', 'tsv_ready', 'archive', 'member', 'result', 'source'])
d = s[(s.day.astype(str) >= '2026-09-27') & (s.tsv_ready == True) & s.archive.notna()].copy()
d['key'] = d.episode_id.astype('int64').astype(str) + '-' + d.seat.astype(int).astype(str)
d = d[~d.key.isin(built) & ~d.key.isin(listed) & (d.episode_id != 115279125)]
d['pri'] = [0 if (k is True and r == 'W') else 1 if k is True else 2 for k, r in zip(d.kaggle, d.result)]
d = d.sort_values(['pri', 'day'], ascending=[True, False]).drop_duplicates('key')
with open(out, 'w') as f:
    f.write('archive\tmember\tday\teid\tseat\tweight\treason\n')
    for r in d.itertuples(index=False):
        f.write(f"{r.archive}\t{r.member}\t{r.day}\t{int(r.episode_id)}\t{int(r.seat)}\t1\tnew:p{r.pri}:{r.source}\n")
print(len(d))
