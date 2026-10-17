#!/usr/bin/env python3
"""TOP4-BEATERS (owner 09-30 01:5x: "THE LAST THING WE TRAIN ON (and our validation set) will be games that defeat anyone in
the top 4 — so their losses"; "whoever maintains top 4, we want to train the most on their losses (e.g. the winner)").
Set = the WINNER seat of every Kaggle game (not ours, not a mirror) whose loser was a top-4 team on the game's day: top 4 of
ANY leaderboard snapshot (extra/top30-losses/LEADERBOARD-*.csv) taken that UTC day, union today's frozen top 4 (snapshot
LEADERBOARD-20260930T0530). Days before the first snapshot (09-25) use the earliest snapshot (flagged top4_source=proxy).
Exclusions: non-train hash split, practice VAL, MMPQ-VAL, sealed, decode holdout seeds, episode 115279125.
Split: VAL = newest-first episodes never present in any training roster used so far, stratified by loser team, until ~20%
per loser team; TRAIN = the rest. Existing VAL is kept on refresh (append-only); new episodes go to VAL first until 20%.
Writes special/TOP4-BEATERS-<sha12>/ {ALL.tsv, TRAIN.tsv (weight 4), VAL.tsv, MANIFEST.json}, special/TOP4-BEATERS-LATEST
(symlink), special/TOP4-BEATERS-VAL.eids (read by every roster builder as an exclusion)."""
import csv, glob, hashlib, json, os, re, sys, collections, datetime as dt
import pandas as pd
S = '/mnt/ai/scratch/farmNEAT/corpus-final/special'
CAT = '/mnt/ai/scratch/farmNEAT/primitive-bc-data/catalog'
LB = '/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/top30-losses'
C = '/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11'
TR = '/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc'
sys.path[:0] = [TR, '/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927/src']
import data as T
US = 16624026
FROZEN = f'{LB}/LEADERBOARD-20260930T0530.csv'
FS = '/mnt/ai/scratch/farmNEAT'
ROSTERS = sorted(set(glob.glob(f'{C}/rows*.tsv') + glob.glob(f'{C}/rows-c7-versions/*.tsv') +
                     glob.glob(f'{FS}/broad/rows-*.tsv') + [g for g in glob.glob(f'{FS}/broad/buckets/*/*.tsv') if re.fullmatch(r'[0-9a-f]{12}', os.path.basename(os.path.dirname(g)))] + glob.glob(f'{S}/TOP4-BEATERS-*/TRAIN.tsv') +
                     glob.glob(f'{FS}/d30/rows-*.tsv') + glob.glob(f'{FS}/n-flow/roster/rows-nflow*.tsv') +
                     glob.glob(f'{FS}/base-v1-oa3plan/rows-*.tsv') + glob.glob(f'{FS}/oldarch-scaleup/rows-*.tsv') +
                     glob.glob(f'{FS}/oldarch-scaleup/meta/rows-*.tsv') + glob.glob(f'{FS}/broad/rows-broad*.tsv') +
                     glob.glob('/home/russ/Desktop/farmNEAT/work/runs/maintainer/oldarch-late-replay-0929/filtered-0929-c7/*.tsv')))
BLOCKED_SUBS = {56680736}
LEAK = f'{FS}/touchup-top8-0929/LEAK-HOLDOUT-IDS.txt'
ROSTERS = [r for r in ROSTERS if 'unbuilt' not in r and 'removed' not in r and 'UNCACHED' not in r and 'missing' not in r]


def ids_from(path, required=True, allow_empty=False):
    """Header-aware: 'eid'/'episode_id' column if the header names one, else one id per line; any malformed row refuses."""
    if not os.path.exists(path):
        if required: raise SystemExit(f'REFUSED: required exclusion input missing: {path}')
        return set()
    lines = [l for l in open(path).read().splitlines() if l.strip() and not l.startswith('#')]
    out = set()
    if lines:
        h = lines[0].split('\t')
        key = 'eid' if 'eid' in h else ('episode_id' if 'episode_id' in h else None)
        for l in (lines[1:] if key else lines):
            p = l.split('\t'); i = h.index(key) if key else 0
            x = p[i].strip() if len(p) > i else ''
            if not x.isdigit(): raise SystemExit(f'REFUSED: malformed exclusion row in {path}: {l[:80]!r}')
            out.add(int(x))
    if required and not allow_empty and not out: raise SystemExit(f'REFUSED: required exclusion input parsed to 0 ids: {path}')
    return out

def top4(path):
    rows = [r for r in csv.DictReader(open(path, encoding='utf-8-sig')) if int(r['TeamId']) != US]
    rows.sort(key=lambda r: int(r['Rank']))
    return {int(r['TeamId']): r['TeamName'] for r in rows[:4]}


def main():
    snaps = collections.defaultdict(dict)
    for f in sorted(glob.glob(f'{LB}/LEADERBOARD-*.csv')):
        m = re.search(r'LEADERBOARD-(\d{4})(\d\d)(\d\d)T', f)
        snaps[f'{m.group(1)}-{m.group(2)}-{m.group(3)}'].update(top4(f))
    frozen = top4(FROZEN)
    first_day = min(snaps)
    s = pd.read_parquet(f'{CAT}/SEATS.parquet', columns=['episode_id', 'seat', 'day', 'kaggle', 'result', 'team', 'team_id', 'opp_team',
                        'opp_team_id', 'is_us', 'opp_is_us', 'tsv_ready', 'archive', 'member', 'submission_id', 'opp_submission_id', 'static'])
    w = s[(s.kaggle == True) & (s.result == 'W') & (s.is_us != True) & (s.opp_is_us != True) & s.opp_team_id.notna()
          & (s.team_id != s.opp_team_id) & (s.day.astype(str) >= '2026-09-03')
          & (s.day.astype(str) <= dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')) & (s.static != True) & ~s.submission_id.isin(BLOCKED_SUBS)].copy()
    def is_top4(row):
        d = str(row.day); tid = int(row.opp_team_id)
        if tid in frozen: return 'frozen'
        if d in snaps and tid in snaps[d]: return 'day'
        if d < first_day and tid in snaps[first_day]: return 'proxy'
        return None
    w['top4_source'] = [is_top4(r) for r in w.itertuples(index=False)]
    w = w[w.top4_source.notna()]
    # exclusions
    hold = {115279125}
    if not os.path.exists(LEAK): raise SystemExit(f'leak list missing: {LEAK}')
    lk = {int(x) for x in open(LEAK).read().split() if x.isdigit()}
    if not lk: raise SystemExit(f'REFUSED: leak list parsed to 0 ids: {LEAK}')
    hold |= lk
    for f in ('/mnt/ai/scratch/farmNEAT/practice-v1/val/VAL.tsv', '/mnt/ai/scratch/farmNEAT/mmpq-val/VAL-GAMES.tsv'):
        if not os.path.exists(f): raise SystemExit(f'REFUSED: required exclusion input missing: {f}')
        hold |= ids_from(f)
    sealed = {int(x) for x in T.sealed_ids()}          # fatal on failure (ASTRA GRABBER-CHAIN P0-2)
    if not sealed: raise SystemExit('REFUSED: sealed_ids() returned 0 ids')
    hold |= sealed
    seeds = {int(l.split('\t')[0]) for l in open('/mnt/ai/scratch/farmNEAT/decode-h2h/HOLDOUT-EPISODES.txt').read().splitlines()
             if l.strip() and not l.startswith('#')}
    if not seeds: raise SystemExit('REFUSED: holdout seed list parsed to 0 seeds')
    ep = pd.read_parquet(f'{CAT}/EPISODES.parquet', columns=['episode_id', 'seed'])
    hold |= set(ep[ep.seed.isin(seeds)].episode_id)
    excluded = collections.Counter()
    keep = []
    for r in w.itertuples(index=False):
        e = int(r.episode_id)
        if e in hold: excluded['holdout/val/sealed'] += 1; continue
        if not T.in_split(e, 'train'): excluded['non-train split'] += 1; continue
        keep.append(r)
    w = pd.DataFrame(keep, columns=w.columns)
    # ever trained?
    trained = set()
    for f in ROSTERS:
        lines = open(f).read().splitlines()          # unreadable roster = hard failure (ASTRA CHECK-BUCKETS P0-3)
        h = lines[0].split('\t') if lines else []
        ei = h.index('eid') if 'eid' in h else (h.index('episode_id') if 'episode_id' in h else 3)
        for line in lines[1:]:
            p = line.split('\t')
            if len(p) > ei and p[ei].isdigit(): trained.add(int(p[ei]))
    w['ever_trained'] = w.episode_id.astype(int).isin(trained)
    w['loser'] = w.opp_team.astype(str)
    w['loser_id'] = w.opp_team_id.astype('int64')
    # split (append-only VAL)
    prev = os.path.join(S, 'TOP4-BEATERS-VAL.eids')
    if not os.path.exists(prev): raise SystemExit(f'REFUSED: TOP4 VAL registry missing: {prev} (create it empty only for a first build)')
    val = set(int(x) for x in open(prev).read().split())
    # reservations are monotonic: a VAL episode found in a training roster stays reserved (never re-released to training);
    # it is reported and does not count toward the 20% target (ASTRA/PRO VERIFY-FIX-BUCKETS)
    contaminated = sorted(val & trained)
    if contaminated: print(f'ALERT: {len(contaminated)} reserved TOP4 VAL episodes appear in a training roster', file=sys.stderr)
    clean_val = val - set(contaminated)
    w = w.sort_values(['day', 'episode_id'], ascending=[False, False])
    for loser, g in w.groupby('loser_id'):
        target = round(0.2 * len(g))
        have = int(g.episode_id.isin(clean_val).sum())
        for e in g[~g.ever_trained].episode_id:
            if have >= target: break
            if int(e) not in val: val.add(int(e)); clean_val.add(int(e)); have += 1
    w['split'] = ['VAL' if int(e) in clean_val else ('RESERVED' if int(e) in val else 'TRAIN') for e in w.episode_id]
    w['cached'] = [os.path.exists(f'{C}/{int(e)}-{int(s_)}.stp') and os.path.getsize(f'{C}/{int(e)}-{int(s_)}.stp') > 0 for e, s_ in zip(w.episode_id, w.seat)]
    body = lambda df, wt: 'archive\tmember\tday\teid\tseat\tweight\treason\n' + ''.join(
        f"{r.archive}\t{r.member}\t{r.day}\t{int(r.episode_id)}\t{int(r.seat)}\t{wt}\ttop4beater:{r.split}:beat-{r.loser}:{r.top4_source}\n"
        for r in df.itertuples(index=False) if r.tsv_ready == True)
    allb = body(w.sort_values(['day', 'episode_id']), 4)
    sha = hashlib.sha256(allb.encode()).hexdigest()
    d = os.path.join(S, f'TOP4-BEATERS-{sha[:12]}')
    os.makedirs(d, exist_ok=True)
    open(f'{d}/ALL.tsv', 'w').write(allb)
    open(f'{d}/TRAIN.tsv', 'w').write(body(w[w.split == 'TRAIN'].sort_values(['day', 'episode_id']), 4))
    open(f'{d}/VAL.tsv', 'w').write(body(w[w.split == 'VAL'].sort_values(['day', 'episode_id']), 1))
    teams_any = {}
    for day, t4 in sorted(snaps.items()):
        for tid, name in t4.items(): teams_any.setdefault(name, []).append(day)
    per = w.groupby(['loser', 'split']).size().unstack(fill_value=0)
    man = {'sha256': sha, 'dir': d, 'built_utc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           'frozen_top4': frozen, 'top4_by_day': {k: list(v.values()) for k, v in sorted(snaps.items())},
           'teams_top4_any_day': {k: sorted(set(v)) for k, v in teams_any.items()},
           'winner_seats': int(len(w)), 'episodes': int(w.episode_id.nunique()), 'days': [str(w.day.min()), str(w.day.max())],
           'excluded': dict(excluded), 'per_loser': {k: {kk: int(vv) for kk, vv in row.items()} for k, row in per.iterrows()},
           'val_never_trained': int(((w.split == 'VAL') & ~w.ever_trained).sum()), 'val_ever_trained': int(((w.split == 'VAL') & w.ever_trained).sum()),
           'train_seats': int((w.split == 'TRAIN').sum()), 'val_seats': int((w.split == 'VAL').sum()),
           'cached_train': int(((w.split == 'TRAIN') & w.cached).sum()), 'not_tsv_ready': int((w.tsv_ready != True).sum()),
           'top4_source': w.top4_source.value_counts().to_dict(), 'rosters_checked': len(ROSTERS),
           'unknown_training_sets': ["O's raw-replay recent_days set (not reconstructed)"], 'val_contaminated_kept_reserved': contaminated[:50], 'val_contaminated_count': len(contaminated),
           'val_episodes_reserved': len(val), 'val_episodes_clean': len(clean_val),
           'val_target_pct': 20, 'val_pct_of_episodes': round(100 * int((w.split == 'VAL').sum()) / max(1, len(w)), 2),
           'excluded_static_or_blocked': 'static winners and submissions in BLOCKED_SUBS removed before selection'}
    json.dump(man, open(f'{d}/MANIFEST.json', 'w'), indent=1, default=str)
    tmp = prev + '.tmp'; open(tmp, 'w').write('\n'.join(str(e) for e in sorted(val)) + '\n'); os.replace(tmp, prev)
    link = os.path.join(S, 'TOP4-BEATERS-LATEST'); lt = link + '.tmp'
    if os.path.lexists(lt): os.remove(lt)
    os.symlink(d, lt); os.replace(lt, link)
    print(json.dumps(man, default=str))


if __name__ == '__main__':
    main()
