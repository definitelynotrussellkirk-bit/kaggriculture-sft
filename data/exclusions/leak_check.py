"""Sealed / practice-VAL / holdout-seed / split / blocked-sub / leak / MMPQ-VAL check of a roster, using extend_rows_c7's own sources
(trainer data.sealed_ids, data.in_split, practice-v1 VAL ids, decode-h2h holdout seeds via the catalog, BLOCKED_SUBS).  python leak_check.py ROSTER"""
import importlib.util, json, sys
sys.argv, ros = sys.argv[:1], sys.argv[1]
spec = importlib.util.spec_from_file_location('ext', '/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/extend_rows_c7.py'); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
import data as trainer_data
import pyarrow.parquet as pq
rows = [l.split('\t') for l in open(ros).read().splitlines()[1:] if l.strip()]
sealed = set(trainer_data.sealed_ids()); val = E.val_episode_ids(); blocked, nseeds = E.holdout_blocked_eids()
leak = {int(x) for x in open('/mnt/ai/scratch/farmNEAT/touchup-top8-0929/LEAK-HOLDOUT-IDS.txt').read().split()} | {115279125}
mval = {int(l.split('\t')[0]) for l in open('/mnt/ai/scratch/farmNEAT/mmpq-val/VAL-GAMES.tsv').read().splitlines()[1:] if l.strip()}
t = pq.read_table(E.CAT / 'SEATS.parquet', columns=['episode_id', 'seat', 'submission_id'])
K = {(int(r[3]), int(r[4])) for r in rows}
sub = {(int(e), int(s)): u for e, s, u in zip(*(t[c].to_pylist() for c in ['episode_id', 'seat', 'submission_id'])) if (int(e), int(s)) in K}
c = {'rows': len(rows), 'sealed': 0, 'practice_val': 0, 'holdout_seed_or_source': 0, 'not_train_split': 0, 'blocked_sub': 0, 'leak': 0, 'mmpq_val': 0}
for r in rows:
    e, s = int(r[3]), int(r[4])
    c['sealed'] += e in sealed; c['practice_val'] += e in val; c['holdout_seed_or_source'] += (e in blocked or E.blocked_source(r[0]))
    c['not_train_split'] += not trainer_data.in_split(e, 'train'); c['blocked_sub'] += (sub.get((e, s)) in E.BLOCKED_SUBS); c['leak'] += e in leak; c['mmpq_val'] += e in mval
c['list_sizes'] = {'sealed': len(sealed), 'practice_val': len(val), 'holdout_catalog_episodes': len(blocked), 'holdout_seeds': nseeds, 'leak': len(leak), 'mmpq_val': len(mval)}
print(json.dumps(c))
