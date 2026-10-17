#!/usr/bin/env python3
"""Versioned union of every never-train list, and roster intersection against it.
  exclusion_union.py build            -> broad/exclusions/EXCLUSIONS-<sha12>.tsv (+ .json, EXCLUSIONS-LATEST); prints the json
  exclusion_union.py check ROSTER...  -> overlap per roster (episode-level families by eid; quarantine by (eid, seat))
Families: practice_val, mmpq_val, top4_val, sealed, holdout_seed, leak, fixed (115279125), hash_val_split (catalog episodes
outside the train hash split), quarantine_sub (seats of BLOCKED_SUBS submissions; seat-level)."""
import glob, hashlib, json, os, sys
import pandas as pd
FS = '/mnt/ai/scratch/farmNEAT'
CAT = f'{FS}/primitive-bc-data/catalog'
OUT = f'{FS}/broad/exclusions'
TR = f'{FS}/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc'
sys.path[:0] = [TR, f'{FS}/primitive-bc-v6-20260927/src']
BLOCKED_SUBS = {56680736}


def ids(path, allow_empty=False):
    if not os.path.exists(path): raise SystemExit(f'REFUSED: missing {path}')
    lines = [l for l in open(path).read().splitlines() if l.strip() and not l.startswith('#')]
    h = lines[0].split('\t') if lines else []
    key = 'eid' if 'eid' in h else ('episode_id' if 'episode_id' in h else None)
    out = set()
    for l in (lines[1:] if key else lines):
        p = l.split('\t') if key else l.split()
        x = (p[h.index(key)] if key and len(p) > h.index(key) else (p[0] if p else '')).strip()
        if not x.isdigit(): raise SystemExit(f'REFUSED: malformed row in {path}: {l[:80]!r}')
        out.add(int(x))
    if not out and not allow_empty: raise SystemExit(f'REFUSED: {path} parsed to 0 ids')
    return out


def sealed_nonempty(T):
    s = {int(x) for x in T.sealed_ids()}
    if not s: raise SystemExit('REFUSED: sealed_ids() returned 0 ids')
    return s

def held_out_ids():
    """Every held-out MMPQ/FRESH episode (VAL-MMPQ, VAL-MMPQ10, VAL-FRESH): never in training until the final wash."""
    import json as _j, os as _o
    out = {}
    if _o.path.exists('/mnt/ai/scratch/farmNEAT/broad/NO-EVALS'):   # owner 16:1x D-1611pool: no evals -> nothing held out
        return out
    for f in ('VAL-MMPQ-0930.json', 'VAL-MMPQ10-0930.json', 'VAL-FRESH-0930.json', 'VAL-MMPQ-NEXT-0930.json'):
        p = f'/mnt/ai/scratch/farmNEAT/broad/{f}'
        if _o.path.exists(p): out[f] = sorted(int(g['episode_id']) for g in _j.load(open(p))['games'])
    out['SURPRISE-PENDING'] = sorted(surprise_pending())
    return out


def surprise_pending():
    """Owner 15:4x rolling mini surprise: MMPQ games newer than VAL-MMPQ-NEXT's cutoff that are local but not yet scored
    (broad/surprise/DONE.json) stay out of every build until scored."""
    import json as _j, os as _o, glob as _g
    B = '/mnt/ai/scratch/farmNEAT/broad'; E = '/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/copy-mmpq'
    if not _o.path.exists(f'{B}/VAL-MMPQ-NEXT-0930.json'): return set()
    nx = _j.load(open(f'{B}/VAL-MMPQ-NEXT-0930.json')); frozen = {int(g['episode_id']) for g in nx['games']}
    cut = max([g['create_time'] for g in nx['games']] + [nx['after_create_time']])
    done = set(_j.load(open(f'{B}/surprise/DONE.json'))) if _o.path.exists(f'{B}/surprise/DONE.json') else set()
    have = {int(_j.loads(l)['episode_id']) for l in open(f'{E}/MANIFEST.jsonl') if l.strip()}
    L = {}
    for f in sorted(_g.glob(f'{E}/LISTINGS-*.json'), key=_o.path.getmtime)[-3:]:
        for e in _j.load(open(f))['episodes']: L[int(e['episode_id'])] = e['create_time']
    return {e for e in have if L.get(e, '') > cut and e not in frozen and e not in done}


def build():
    import data as T
    fam = {'practice_val': ids(f'{FS}/practice-v1/val/VAL.tsv'), 'mmpq_val': ids(f'{FS}/mmpq-val/VAL-GAMES.tsv'),
           'top4_val': ids(f'{FS}/corpus-final/special/TOP4-BEATERS-VAL.eids', allow_empty=True),
           'val_mmpq': ({int(g['episode_id']) for g in json.load(open(f'{FS}/broad/VAL-MMPQ-0930.json'))['games']}
                        if os.path.exists(f'{FS}/broad/VAL-MMPQ-0930.json') else set()),
           'val_mmpq10': ({int(g['episode_id']) for g in json.load(open(f'{FS}/broad/VAL-MMPQ10-0930.json'))['games']}
                          if os.path.exists(f'{FS}/broad/VAL-MMPQ10-0930.json') else set()),
           'val_fresh': ({int(g['episode_id']) for g in json.load(open(f'{FS}/broad/VAL-FRESH-0930.json'))['games']}
                         if os.path.exists(f'{FS}/broad/VAL-FRESH-0930.json') else set()),
           'sealed': sealed_nonempty(T), 'leak': ids(f'{FS}/touchup-top8-0929/LEAK-HOLDOUT-IDS.txt'), 'fixed': {115279125}}
    if os.path.exists(f'{FS}/broad/NO-EVALS'):   # D-1611pool: no evals -> no eval holdout families
        for k in ('val_mmpq', 'val_mmpq10', 'val_fresh'): fam[k] = set()
    seeds = {int(l.split('\t')[0]) for l in open(f'{FS}/decode-h2h/HOLDOUT-EPISODES.txt').read().splitlines() if l.strip() and not l.startswith('#')}
    if not seeds: raise SystemExit('REFUSED: 0 holdout seeds')
    ep = pd.read_parquet(f'{CAT}/EPISODES.parquet', columns=['episode_id', 'seed'])
    fam['holdout_seed'] = set(ep[ep.seed.isin(seeds)].episode_id.astype(int))
    fam['hash_val_split'] = {int(e) for e in ep.episode_id if not T.in_split(int(e), 'train')}
    s = pd.read_parquet(f'{CAT}/SEATS.parquet', columns=['episode_id', 'seat', 'submission_id'])
    q = s[s.submission_id.isin(BLOCKED_SUBS)]
    rows = sorted((e, '*', f) for f, v in fam.items() for e in v) + sorted((int(e), str(int(x)), 'quarantine_sub') for e, x in zip(q.episode_id, q.seat))
    body = 'eid\tseat\tfamily\n' + ''.join(f'{e}\t{x}\t{f}\n' for e, x, f in rows)
    sha = hashlib.sha256(body.encode()).hexdigest()
    os.makedirs(OUT, exist_ok=True)
    p = f'{OUT}/EXCLUSIONS-{sha[:12]}.tsv'
    if not os.path.exists(p):
        open(p + '.tmp', 'w').write(body); os.replace(p + '.tmp', p)
    meta = {'sha256': sha, 'path': p, 'families': {f: len(v) for f, v in fam.items()} | {'quarantine_sub_seats': len(q)},
            'blocked_subs': sorted(BLOCKED_SUBS)}
    json.dump(meta, open(p[:-4] + '.json', 'w'), indent=1)
    lt = f'{OUT}/EXCLUSIONS-LATEST.tmp'
    if os.path.lexists(lt): os.remove(lt)
    os.symlink(p, lt); os.replace(lt, f'{OUT}/EXCLUSIONS-LATEST')
    return meta


def check(rosters, meta):
    x = pd.read_csv(meta['path'], sep='\t', dtype={'seat': str})
    ep = x[x.seat == '*']; qs = set(zip(x[x.seat != '*'].eid.astype(int), x[x.seat != '*'].seat.astype(int)))
    res = {}
    for r in rosters:
        d = pd.read_csv(r, sep='\t', usecols=['eid', 'seat'])
        keys = set(zip(d.eid.astype(int), d.seat.astype(int))); eids = {e for e, _ in keys}
        hit = ep[ep.eid.isin(eids)]
        per = hit.groupby('family').eid.apply(lambda v: sorted(set(int(e) for e in v))).to_dict()
        qh = sorted(keys & qs)
        if qh: per['quarantine_sub'] = [f'{e}-{s}' for e, s in qh]
        res[r] = {'rows': len(d), 'episodes': len(eids), 'overlap_episodes': len(set(int(e) for e in hit.eid) | {e for e, _ in qh}),
                  'by_family': {f: {'n': len(v), 'ids': v[:40]} for f, v in per.items()}}
    return res


if __name__ == '__main__':
    meta = build()
    if sys.argv[1:2] == ['check']:
        res = check(sys.argv[2:], meta)
        print(json.dumps({'exclusions': meta, 'rosters': res}, indent=1))
        if any(v['overlap_episodes'] for v in res.values()): sys.exit(1)      # a real overlap is a failure (ASTRA-5 #6)
    else:
        print(json.dumps(meta))
