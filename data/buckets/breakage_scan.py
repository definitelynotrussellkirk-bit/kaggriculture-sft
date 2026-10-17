#!/usr/bin/env python3
"""Objective-breakage scan of the LOSING seat (coordinator 11:1x): for every episode in the current bucket generation + t1 generation
not yet scanned, read the replay once and record for both seats: final status, steps, idle steps (farmer PASS, no hand actions,
no market orders, or an empty/None action). Incremental: results append to broad/breakage/SCAN-<task>.tsv; already-scanned
episodes are skipped.
  breakage_scan.py plan              -> broad/breakage/TODO.tsv (episodes to scan)
  breakage_scan.py scan i n          -> scans rows j % n == i of TODO.tsv
  breakage_scan.py broken            -> prints the broken-episode summary (used by the builders via load_broken())"""
import csv, glob, json, os, sys, zipfile
FS = '/mnt/ai/scratch/farmNEAT'; BR = f'{FS}/broad'; D = f'{BR}/breakage'
IDLE_MAX = 50; BAD_STATUS = {'ERROR', 'TIMEOUT', 'INVALID'}


def scanned():
    out = {}
    for f in glob.glob(f'{D}/SCAN-*.tsv'):
        for r in csv.DictReader(open(f), delimiter='\t'):
            out[int(r['eid'])] = r
    return out


def plan():
    os.makedirs(D, exist_ok=True)
    have = scanned(); todo = {}
    files = glob.glob(f'{os.path.realpath(BR + "/LATEST-BUCKETS")}/*.tsv') + glob.glob(f'{os.path.realpath(BR + "/t1/LATEST")}/*.tsv')
    for f in files:
        if 'UNCACHED' in f: continue
        for r in csv.DictReader(open(f), delimiter='\t'):
            e = int(r['eid'])
            if e not in have and e not in todo: todo[e] = (r['archive'], r['member'])
    with open(f'{D}/TODO.tsv', 'w') as w:
        w.write('eid\tarchive\tmember\n')
        for e, (a, m) in sorted(todo.items()): w.write(f'{e}\t{a}\t{m}\n')
    print(json.dumps({'already_scanned': len(have), 'todo': len(todo)}))


def idle(a):
    if not isinstance(a, dict): return True
    f = a.get('farmer'); h = a.get('hands') or []; m = a.get('market') or []
    return (f in (None, [], ['PASS'])) and all(x in (None, [], ['PASS']) for x in h) and not m


def scan(i, n):
    rows = [r for j, r in enumerate(csv.DictReader(open(f'{D}/TODO.tsv'), delimiter='\t')) if j % n == i]
    out = open(f'{D}/SCAN-{os.environ.get("SLURM_JOB_ID", "local")}-{i}.tsv', 'a')
    if out.tell() == 0: out.write('eid\tsteps\tstatus0\tstatus1\tidle0\tidle1\terror\n')
    zc = {}
    for r in rows:
        try:
            z = zc.get(r['archive']) or zc.setdefault(r['archive'], zipfile.ZipFile(r['archive']))
            rep = json.loads(z.read(r['member']))
            st = rep.get('steps') or []
            last = st[-1] if st else [{}, {}]
            s0, s1 = (last[0].get('status') if len(last) > 0 else None), (last[1].get('status') if len(last) > 1 else None)
            i0 = sum(1 for s in st[1:] if len(s) > 0 and idle(s[0].get('action')))
            i1 = sum(1 for s in st[1:] if len(s) > 1 and idle(s[1].get('action')))
            out.write(f"{r['eid']}\t{len(st)}\t{s0}\t{s1}\t{i0}\t{i1}\t\n")
        except Exception as ex:   # noqa: BLE001 - recorded, never silently dropped
            out.write(f"{r['eid']}\t\t\t\t\t\t{type(ex).__name__}\n")
        if len(zc) > 64: zc.clear()
    out.close()


def load_broken():
    """{eid: {seat: reason}} for seats with a bad final status or > IDLE_MAX idle steps."""
    b = {}
    for e, r in scanned().items():
        if r.get('error'): continue
        for k in (0, 1):
            why = []
            if (r[f'status{k}'] or '').upper() in BAD_STATUS: why.append(f"status {r[f'status{k}']}")
            if r[f'idle{k}'] and int(r[f'idle{k}']) > IDLE_MAX: why.append(f"idle {r[f'idle{k}']}")
            if why: b.setdefault(e, {})[k] = ','.join(why)
    return b


if __name__ == '__main__':
    if sys.argv[1] == 'plan': plan()
    elif sys.argv[1] == 'scan': scan(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == 'broken':
        b = load_broken(); s = scanned()
        print(json.dumps({'scanned': len(s), 'errors': sum(1 for r in s.values() if r.get('error')), 'broken_episodes': len(b),
                          'sample': dict(list(b.items())[:20])}))


def weak_subs(s, now=None, min_age_h=3.0, min_games=40, below=2400.0):
    """Age-gated weak submissions (owner 11:2x): at least min_age_h old (first catalogued game) or >= min_games catalogued games,
    and latest known rating (sub_score, else game_score; by create_time) still < below. Unknown rating -> not weak.
    s: SEATS frame with episode_id, create_time, submission_id, sub_score, game_score. Returns {sub: {...}}."""
    import pandas as pd, datetime as dt
    now = now or dt.datetime.now(dt.timezone.utc)
    t = s[s.submission_id.notna()].copy()
    t['ct'] = pd.to_datetime(t.create_time.astype(str).str.slice(0, 19), errors='coerce', utc=True)
    ss = t.sub_score.where(t.sub_score > 0); gs = t.game_score.where(t.game_score > 0)   # <= 0 (e.g. -1) = unknown, never weak
    t['r'] = ss.where(ss.notna(), gs)
    g = t.groupby('submission_id')
    first = g.ct.min(); n = g.episode_id.nunique()
    rated = t[t.r.notna()].sort_values('ct').groupby('submission_id').r.last()
    out = {}
    for sub, r in rated.items():
        age_h = (now - first[sub]).total_seconds() / 3600 if pd.notna(first[sub]) else 0
        if r < below and (age_h >= min_age_h or n[sub] >= min_games):
            out[int(sub)] = {'rating': round(float(r), 1), 'age_h': round(age_h, 1), 'games': int(n[sub])}
    return out
