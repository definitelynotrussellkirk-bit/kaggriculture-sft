#!/usr/bin/env python3
"""t1 generation (owner 09-30 09:2x-10:0x): every local Kaggle game of leaderboard #1 (MMPQ now) in five buckets drawn by bucket
probability (nf_data: bucket by --bucket-probs, then uniform within; no per-row weights):
  T1W-new  #1's wins, newest submission, vs other teams          x4
  T1W-old  #1's wins, older submissions, vs other teams          x1
  T1B-new  winners over #1's newest submission                   x4
  T1B-old  winners over #1's older submissions                   x2
  T1S      #1-vs-#1 games involving the newest submission, winner seat  x4
p(b) = n_b * mult_b / sum(n * mult) over non-empty buckets; an empty bucket gets p = 0 and a note (it is not silently dropped).
Newest submission = highest-ref submission in #1's newest save_team listing (T1_NEWEST_SUB overrides). Per-seat submission ids come
from #1's listings (sub_ids per seat), else SEATS.submission_id/opp_submission_id. Teams within T1_GAP (50) points of #1 join with the
same rule. Exclusion union (incl. MMPQ-VAL), static tapes, our seats, not-tsv-ready removed; only cached seats are written.
Writes broad/t1/gen-<sha12>/{<bucket>.tsv, EFFECTIVE-SHARES.json, RECEIPT.json}, broad/t1/UNCACHED-t1.tsv (T1S first), broad/t1/LATEST."""
import csv, glob, hashlib, json, os, sys, time, datetime as dt
import pandas as pd
FS = '/mnt/ai/scratch/farmNEAT'
BR = f'{FS}/broad'; OUT = f'{BR}/t1'
CAT = f'{FS}/primitive-bc-data/catalog'; EX = f'{FS}/primitive-bc-data/extra'; LB = f'{EX}/top30-losses'
C = f'{FS}/base-v1-stepcache-v11'
US = 16624026; GAP = float(os.environ.get('T1_GAP', 50))
MULT = {'T1W-new': 4.0, 'T1W-old': 1.0, 'T1B-new': 4.0, 'T1B-old': 2.0, 'T1S': 4.0}
SLUG = {16681125: 'copy-mmpq', 16623559: 'copy-decem', 16732748: 'copy-dsm', 16915014: 'copy-boey', 16770421: 'copy-vadim',
        16718819: 'copy-majkel', 16730612: 'copy-umg', 16712258: 'copy-yizhou'}
sys.path.insert(0, BR)
import exclusion_union as XU


def listing_subs(tid):
    d = f"{EX}/{SLUG.get(tid, f'copy-team-{tid}')}"
    seat_sub, subs, newest_file, score = {}, {}, None, {}
    for f in sorted(glob.glob(f'{d}/LISTINGS-*.json'), key=os.path.getmtime):
        L = json.load(open(f)); newest_file = f
        score = {}                                       # scores from the newest listing only (hourly re-check)
        for s in L.get('submissions') or []:
            subs[int(s.get('ref') or s.get('id'))] = str(s.get('dateSubmitted') or s.get('date') or '')
            try: score[int(s.get('ref') or s.get('id'))] = float(s.get('publicScore') or s.get('public_score'))
            except (TypeError, ValueError): pass
        ce = L.get('carried_episodes'); ce = ce if isinstance(ce, list) else []
        for e in (L.get('episodes') or []) + ce:
            for i, x in enumerate(e.get('sub_ids') or []):
                if x: seat_sub[(int(e['episode_id']), i)] = int(x)
    return seat_sub, subs, newest_file, score


def main():
    t0 = time.time()
    snap = sorted(glob.glob(f'{LB}/LEADERBOARD-*.csv'))[-1]
    lb = sorted((r for r in csv.DictReader(open(snap, encoding='utf-8-sig')) if int(r['TeamId']) != US), key=lambda r: int(r['Rank']))
    top = lb[0]; top_score = float(top['Score'])
    teams = [(int(r['TeamId']), r['TeamName'], float(r['Score'])) for r in lb if top_score - float(r['Score']) <= GAP]
    tids = {t for t, _, _ in teams}
    seat_sub, subs, newest = {}, {}, {}
    for t in tids:
        ss, sb, nf, sc = listing_subs(t); seat_sub.update(ss)
        # "his newest, strong policy" (owner 11:3x): the 4x submission is the highest-rated submission in the newest listing
        # (ties -> newest id); a brand-new submission takes over only once its rating passes the current one's
        strong = max(sc, key=lambda k: (sc[k], k)) if sc else (max(sb) if sb else None)
        newest[t] = int(os.environ['T1_NEWEST_SUB']) if (os.environ.get('T1_NEWEST_SUB') and t == int(top['TeamId'])) else strong
        subs[t] = {'submissions': {str(k): v for k, v in sorted(sb.items())}, 'scores_newest_listing': {str(k): v for k, v in sc.items()},
                   'strong_4x': newest[t], 'newest_id': max(sb) if sb else None, 'listing': nf}
    newest_set = {v for v in newest.values() if v}
    xmeta = XU.build()
    xu = pd.read_csv(xmeta['path'], sep='\t', dtype={'seat': str})
    # t1 only (coordinator 10:0x, owner "all MMPQ games we have"; no decisions by validation loss): the hash-VAL split and MMPQ-VAL
    # are NOT excluded here; leak, quarantine, sealed, practice VAL, TOP4 VAL, fixed and holdout seeds still are
    T1_DROP_FAMILIES = {'hash_val_split', 'mmpq_val'}
    xu = xu[~xu.family.isin(T1_DROP_FAMILIES)]
    xe = set(xu[xu.seat == '*'].eid.astype(int)) | {e for v in XU.held_out_ids().values() for e in v}; xq = set(zip(xu[xu.seat != '*'].eid.astype(int), xu[xu.seat != '*'].seat.astype(int)))
    s = pd.read_parquet(f'{CAT}/SEATS.parquet', columns=['episode_id', 'seat', 'day', 'kaggle', 'result', 'is_us', 'team_id', 'opp_team_id',
                        'team', 'opp_team', 'tsv_ready', 'archive', 'member', 'static', 'submission_id', 'opp_submission_id', 'create_time', 'sub_score', 'game_score'])
    s['day'] = s.day.astype(str)
    g = s[(s.kaggle == True) & (s.result == 'W') & (s.team_id.isin(tids) | s.opp_team_id.isin(tids))].copy().drop_duplicates(['episode_id', 'seat'])
    g['sub'] = [seat_sub.get((int(e), int(z)), x) for e, z, x in zip(g.episode_id, g.seat, g.submission_id)]
    g['opp_sub'] = [seat_sub.get((int(e), 1 - int(z)), x) for e, z, x in zip(g.episode_id, g.seat, g.opp_submission_id)]
    ex = {'candidates': int(len(g))}
    import breakage_scan as BS   # objective breakage of the TRAINED seat only (winners over broken/weak opponents stay)
    brk = BS.load_broken(); weak = BS.weak_subs(s)
    drop = {'broken_seat': pd.Series([int(z) in brk.get(int(e), {}) for e, z in zip(g.episode_id, g.seat)], index=g.index, dtype=bool),
            'weak_sub_seat': g.submission_id.isin(list(weak)) | pd.Series([x in weak for x in g['sub']], index=g.index, dtype=bool),
            'our_seat': g.is_us == True, 'static': g.static == True, 'not_tsv_ready': g.tsv_ready != True,
            'exclusion_union': g.episode_id.isin(xe) | pd.Series([(int(e), int(z)) in xq for e, z in zip(g.episode_id, g.seat)], index=g.index, dtype=bool)}
    for k, m in drop.items():
        ex[f'excluded_{k}'] = int(m.sum()); g = g[~m.reindex(g.index, fill_value=False)]
    def bucket(r):
        me, op = r.team_id in tids, r.opp_team_id in tids
        if me and op and r.team_id == r.opp_team_id:
            return 'T1S' if (r.sub in newest_set or r.opp_sub in newest_set) else None
        if me: return 'T1W-new' if r.sub in newest_set else 'T1W-old'
        return 'T1B-new' if r.opp_sub in newest_set else 'T1B-old'
    g['bucket'] = [bucket(r) for r in g.itertuples(index=False)]
    ex['t1s_not_involving_newest'] = int(g.bucket.isna().sum()); g = g[g.bucket.notna()]
    g['cached'] = pd.Series([os.path.exists(p) and os.path.getsize(p) > 0 for p in (f'{C}/{int(e)}-{int(z)}.stp' for e, z in zip(g.episode_id, g.seat))],
                            index=g.index, dtype=bool)
    g = g.sort_values(['day', 'episode_id', 'seat'])
    hdr = 'archive\tmember\tday\teid\tseat\tweight\treason\n'
    row = lambda r: f"{r.archive}\t{r.member}\t{r.day}\t{int(r.episode_id)}\t{int(r.seat)}\t1\t{r.bucket}:{r.team}:sub{r.sub}\n"
    names = list(MULT)
    bodies = {b: hdr + ''.join(row(r) for r in g[g.cached & (g.bucket == b)].itertuples(index=False)) for b in names}
    n = {b: int((g.cached & (g.bucket == b)).sum()) for b in names}
    mass = {b: n[b] * MULT[b] for b in names}; tot = sum(mass.values())
    probs = {b: (round(mass[b] / tot, 6) if tot else 0.0) for b in names}
    ident = json.dumps({'multipliers': MULT, 'teams': sorted(tids), 'newest': {str(k): v for k, v in newest.items()}}, sort_keys=True)
    sha = hashlib.sha256((ident + '\n' + ''.join(bodies[b] for b in names)).encode()).hexdigest()
    gd = f'{OUT}/gen-{sha[:12]}'
    if not os.path.exists(gd):
        tmp = f'{OUT}/.tmp-{sha[:12]}-{os.getpid()}'; os.makedirs(tmp)
        for b in names: open(f'{tmp}/{b}.tsv', 'w').write(bodies[b])
        eff = {'schema': 't1-effective-shares/v1', 'generation': sha[:12], 'rule': 'p(b) = n_b*mult_b / sum(n*mult); uniform within a bucket; no row weights',
               'multipliers': MULT, 'cached_seats': n, 'effective_shares': probs, 'files': [f'{b}.tsv' for b in names],
               'empty_buckets': {b: 'no cached seats this build; p = 0 (not renormalised away)' for b in names if n[b] == 0},
               'bucket_probs_arg': ','.join(f'{b}={probs[b]:.6f}' for b in names)}
        json.dump(eff, open(f'{tmp}/EFFECTIVE-SHARES.json', 'w'), indent=1)
        rec = {'schema': 't1/v2', 'sha256': sha, 'identity_json': ident, 'built_utc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               'leaderboard': os.path.basename(snap), 'top1': {'team_id': int(top['TeamId']), 'name': top['TeamName'], 'score': top_score},
               'second': ({'team_id': int(lb[1]['TeamId']), 'name': lb[1]['TeamName'], 'score': float(lb[1]['Score']), 'gap': round(top_score - float(lb[1]['Score']), 1)} if len(lb) > 1 else None),
               'gap_rule_points': GAP, 'teams': [{'team_id': t, 'name': nm, 'score': sc} for t, nm, sc in teams], 'submissions': {str(k): v for k, v in subs.items()},
               'cached_seats': n, 'uncached_seats': {b: int((~g.cached & (g.bucket == b)).sum()) for b in names}, 'effective_shares': probs,
               'wins_by_sub': {str(k): int(v) for k, v in g[g.cached & g.bucket.str.startswith('T1W')].groupby('sub').size().items()},
               'losses_by_sub': {str(k): int(v) for k, v in g[g.cached & g.bucket.str.startswith('T1B')].groupby('opp_sub').size().items()},
               't1s_episodes': sorted(int(e) for e in g[g.bucket == 'T1S'].episode_id), **ex, 'exclusion_union_sha256': xmeta['sha256'], 'exclusion_families_applied': sorted(set(xu.family)),
               'exclusion_families_not_applied': sorted(T1_DROP_FAMILIES),
               'seconds': round(time.time() - t0, 1)}
        json.dump(rec, open(f'{tmp}/RECEIPT.json', 'w'), indent=1, default=str)
        os.rename(tmp, gd)
    unc = g[~g.cached].copy(); unc['o'] = unc.bucket.map({'T1W-new': 0, 'T1S': 1, 'T1B-new': 2, 'T1B-old': 3, 'T1W-old': 4})
    open(f'{OUT}/UNCACHED-t1.tsv.tmp', 'w').write(hdr + ''.join(row(r) for r in unc.sort_values('o').itertuples(index=False)))
    os.replace(f'{OUT}/UNCACHED-t1.tsv.tmp', f'{OUT}/UNCACHED-t1.tsv')
    import subprocess; subprocess.run([sys.executable, '-B', f'{BR}/t1_to_b.py', gd], check=False, stdout=subprocess.DEVNULL)   # B roster twin
    lt = f'{OUT}/LATEST.tmp'
    if os.path.lexists(lt): os.remove(lt)
    os.symlink(gd, lt); os.replace(lt, f'{OUT}/LATEST')
    held = {e for v in XU.held_out_ids().values() for e in v}
    bad = sorted({int(e) for e in g[g.cached].episode_id} & held)
    if bad: raise SystemExit(f'REFUSED: held-out episodes in t1: {bad[:10]}')
    print('T1 ' + json.dumps({'gen': gd, 'newest': newest, 'cached_seats': n, 'effective_shares': probs,
                              'uncached': int((~g.cached).sum()), 'second_gap': round(top_score - float(lb[1]['Score']), 1) if len(lb) > 1 else None}))


if __name__ == '__main__':
    main()
