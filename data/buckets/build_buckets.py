#!/usr/bin/env python3
"""Sampling buckets A..E (DATA-PLAN-0930 §7, owner 02:38-02:41; coordinator spec 02:50).
Qualifying = Kaggle winner seats, not ours, day >= 09-03, not a mirror (winner team != loser team), tsv_ready, train split, not
in leak / MMPQ-VAL / sealed / practice-VAL / decode-holdout-seed / TOP4-BEATERS-VAL. Each seat lands once, in its highest bucket:
  A (.35) TOP4-BEATERS TRAIN (special/TOP4-BEATERS-LATEST/TRAIN.tsv) + 09-30 wins by top-12 teams (latest leaderboard snapshot)
  B (.20) other 09-30 wins          C (.20) 09-29 wins
  D (.15) 09-24..09-28, top 10% per day      E (.05) 09-03..09-23, top 10% per day
Top-10% ranking per day: mean of game_score and opp_game_score (at-time); where missing, mean of rating and opp_rating
(leaderboard), within the same per-day pool; ties broken by episode id.
Writes broad/buckets/<sha12>/{A..E}.tsv (archive member day eid seat weight reason; weight 1: sampling is by bucket then
uniform) + RECEIPT.json + UNCACHED-{A..E}.tsv, and points broad/LATEST-BUCKETS at the directory. Recipe: broad/BUCKET-RECIPE.json.
  python3 build_buckets.py [--no-publish]"""
import argparse, csv, glob, hashlib, json, os, sys, time, datetime as dt
import pandas as pd
BR = '/mnt/ai/scratch/farmNEAT/broad'
CAT = '/mnt/ai/scratch/farmNEAT/primitive-bc-data/catalog'
C = '/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11'
SP = '/mnt/ai/scratch/farmNEAT/corpus-final/special'
LB = '/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/top30-losses'
TR = '/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc'
sys.path[:0] = [TR, '/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927/src']
US = 16624026
BLOCKED_SUBS = {56680736}   # extend_rows_c7.BLOCKED_SUBS (PRO-TEAM-PROFILE-VICTOR-TUFA-0930)
LEAK = '/mnt/ai/scratch/farmNEAT/touchup-top8-0929/LEAK-HOLDOUT-IDS.txt'
RECIPE = {'schema': 'bucket-recipe/v1', 'source': 'base-v1/DATA-PLAN-0930.md section 7 (owner 02:38-02:41)',
          'shares': {'A': .35, 'B': .20, 'C': .20, 'D': .15, 'E': .05},
          'buckets': {'A': 'TOP4-BEATERS TRAIN (dynamic top 4, any day) + 09-30 wins by current top-12 teams',
                      'B': 'other 09-30 wins', 'C': '09-29 wins', 'D': '09-24..09-28 wins, top 10% per day',
                      'E': '09-03..09-23 wins, top 10% per day'},
          'qualifying': ['Kaggle', 'winner seat', 'not ours', 'not a mirror', 'day >= 2026-09-03', 'tsv_ready', 'trainer train split',
                         'not leak / MMPQ-VAL / sealed / practice-VAL / decode holdout seed / TOP4-BEATERS VAL'],
          'top10_rank': 'mean(game_score, opp_game_score) at game time; else mean(rating, opp_rating) leaderboard; per day',
          'sampling': 'pick bucket by share (empty buckets renormalised away), then a seat uniformly within it; no per-seat weights',
          'publish_refusal': 'any bucket that is non-empty upstream but loses > 10% of its seats to missing cache'}


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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--no-publish', action='store_true'); a = ap.parse_args()
    t0 = time.time()
    import data as T
    RP = os.environ.get('BUCKET_RECIPE', f'{BR}/BUCKET-RECIPE.json')
    R = json.load(open(RP))           # recipe file is the only source of shares (PRO CHECK-BUCKETS CRITICAL)
    BUCKETS = list(R['shares'].keys())                  # recipe order, e.g. A, BC, D, E (v4)
    shares = {b: float(R['shares'][b]) for b in BUCKETS}
    merge = R.get('merge') or {}                         # v4: {'B': 'BC', 'C': 'BC'}
    if 'beat_us_days' not in R: raise SystemExit('REFUSED: recipe lacks beat_us_days (use [] to admit no beat-us winners)')
    beat_us_days = set(R['beat_us_days'])            # [] = no beat-us winners admitted
    if not BUCKETS or any(not (v >= 0) for v in shares.values()) or sum(shares.values()) <= 0:
        raise SystemExit(f'REFUSED: recipe shares invalid: {shares}')
    if any(t not in BUCKETS or k in BUCKETS for k, t in merge.items()):
        raise SystemExit(f'REFUSED: recipe merge invalid for buckets {BUCKETS}: {merge}')
    if set(BUCKETS) - {'A', 'B', 'C', 'D', 'E', 'BC', 'L', 'M', 'X', 'W_B', 'W_N', 'W_O', 'S_RAND', 'S_BEST', 'N_D', 'N_E'}: raise SystemExit(f'REFUSED: recipe names unknown buckets: {BUCKETS}')
    seats_stat = os.stat(f'{CAT}/SEATS.parquet')          # pinned input snapshot (read once below)
    inputs = {'SEATS.parquet': {'mtime_utc': dt.datetime.fromtimestamp(seats_stat.st_mtime, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                                'bytes': seats_stat.st_size},
              'build_start_utc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
              'catalog_lag_min': round((time.time() - seats_stat.st_mtime) / 60, 1)}
    s = pd.read_parquet(f'{CAT}/SEATS.parquet', columns=['episode_id', 'seat', 'day', 'kaggle', 'result', 'is_us', 'opp_is_us', 'team_id',
                        'opp_team_id', 'team', 'tsv_ready', 'archive', 'member', 'game_score', 'opp_game_score', 'rating', 'opp_rating',
                        'static', 'submission_id', 'create_time', 'sub_score', 'opp_submission_id'])
    s['day'] = s.day.astype(str)
    nd = int(s.duplicated(['episode_id', 'seat'], keep=False).sum())   # PRO B&R P1: the duplicate gate must run where duplicates can exist
    if nd: raise SystemExit(f'REFUSED: catalog has {nd} duplicate (episode, seat) rows; refusing instead of collapsing them')
    w = s[(s.kaggle == True) & (s.result == 'W') & (s.is_us != True) & (s.day >= '2026-09-03') & (s.team_id != s.opp_team_id)].copy()
    if R.get('days_window'):   # owner 14:1x: "only the 29th and 30th now" (D/E gone; TOP4 rows outside the window leave too)
        w = w[w.day.isin(R['days_window'])]
    # exclusions
    hold = {115279125}
    hold |= ids_from('/mnt/ai/scratch/farmNEAT/practice-v1/val/VAL.tsv')
    hold |= ids_from('/mnt/ai/scratch/farmNEAT/mmpq-val/VAL-GAMES.tsv')
    if not os.path.exists(LEAK): raise SystemExit(f'leak list missing: {LEAK}')
    leak = {int(x) for x in open(LEAK).read().split() if x.isdigit()}
    if not leak: raise SystemExit(f'REFUSED: leak list parsed to 0 ids: {LEAK}')
    hold |= leak
    hold |= ids_from(f'{SP}/TOP4-BEATERS-VAL.eids', allow_empty=True)
    vm = '/mnt/ai/scratch/farmNEAT/broad/VAL-MMPQ-0930.json'   # owner 12:5x: MMPQ's 3 newest held-out games (VAL-MMPQ), frozen
    if os.path.exists(vm): hold |= {int(g['episode_id']) for g in json.load(open(vm))['games']}
    sys.path.insert(0, BR); import exclusion_union as _XU0
    hold |= {e for v in _XU0.held_out_ids().values() for e in v}   # VAL-MMPQ(10/NEXT), VAL-FRESH, unscored new MMPQ games
    vf = '/mnt/ai/scratch/farmNEAT/broad/VAL-FRESH-0930.json'   # coordinator 13:4x overfit test: 30 fresh held-out winner seats, frozen
    if os.path.exists(vf): hold |= {int(g['episode_id']) for g in json.load(open(vf))['games']}
    sealed = {int(x) for x in T.sealed_ids()}          # a failure here is fatal (ASTRA GRABBER-CHAIN P0-2)
    if not sealed: raise SystemExit('REFUSED: sealed_ids() returned 0 ids')
    hold |= sealed
    seeds = {int(l.split('\t')[0]) for l in open('/mnt/ai/scratch/farmNEAT/decode-h2h/HOLDOUT-EPISODES.txt').read().splitlines() if l.strip() and not l.startswith('#')}
    if not seeds: raise SystemExit('REFUSED: holdout seed list parsed to 0 seeds')
    ep = pd.read_parquet(f'{CAT}/EPISODES.parquet', columns=['episode_id', 'seed'])
    hold |= set(ep[ep.seed.isin(seeds)].episode_id)
    n0 = len(w)
    # objective breakage (coordinator 11:1x-11:3x): never train on a BROKEN seat (final ERROR/TIMEOUT/INVALID or > 50 idle steps, from
    # the replay scan) or on a WEAK submission's own seats (>= 3 h old or >= 40 games and latest rating < 2400); winners over them stay
    sys.path.insert(0, BR); import breakage_scan as BS
    brk = BS.load_broken(); weak = BS.weak_subs(s)
    bw = pd.Series([int(z) in brk.get(int(e), {}) for e, z in zip(w.episode_id, w.seat)], index=w.index, dtype=bool)
    wk = w.submission_id.isin(list(weak))
    breakage = {'broken_winner_seats': int(bw.sum()), 'weak_sub_winner_seats': int((wk & ~bw).sum()), 'weak_subs': len(weak),
                'weak_subs_removed': {str(k): weak[k] for k in sorted({int(x) for x in w[wk].submission_id})}, 'scanned_episodes': len(BS.scanned())}
    w = w[~bw & ~wk]
    pre = {'excluded_unknown_opponent': int(w.opp_team_id.isna().sum())}   # unknown loser: cannot prove not-us / not-mirror
    w = w[w.opp_team_id.notna()]
    pre |= {'excluded_static_winner': int((w.static == True).sum()),
           'excluded_blocked_submission': int(w.submission_id.isin(BLOCKED_SUBS).sum())}
    w = w[(w.static != True) & ~w.submission_id.isin(BLOCKED_SUBS)]
    bu_drop = (w.opp_is_us == True) & ~w.day.isin(beat_us_days)
    pre['excluded_beat_us_outside_days'] = int(bu_drop.sum())
    w = w[~bu_drop]
    ex = {**pre, 'excluded_holdout': int(w.episode_id.isin(hold).sum())}
    w = w[~w.episode_id.isin(hold)]
    tr = pd.Series([T.in_split(int(e), 'train') for e in w.episode_id], index=w.index, dtype=bool)
    ex['excluded_non_train_split'] = int(len(w) - sum(tr)); w = w[tr]
    ex['not_tsv_ready'] = int((w.tsv_ready != True).sum()); w = w[w.tsv_ready == True]
    # buckets
    top4 = set()
    t4p = f'{SP}/TOP4-BEATERS-LATEST/TRAIN.tsv'
    if not os.path.exists(t4p): raise SystemExit(f'REFUSED: TOP4 training input missing: {t4p}')
    if True:
        t4 = pd.read_csv(t4p, sep='\t', usecols=['eid', 'seat'])
        top4 = set(zip(t4.eid.astype(int), t4.seat.astype(int)))
    snap = sorted(glob.glob(f'{LB}/LEADERBOARD-*.csv'))[-1]
    inputs['leaderboard'] = {'file': os.path.basename(snap), 'mtime_utc': dt.datetime.fromtimestamp(os.path.getmtime(snap), dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
    inputs['top4_latest'] = os.path.realpath(f'{SP}/TOP4-BEATERS-LATEST')
    lbrows = sorted(csv.DictReader(open(snap, encoding='utf-8-sig')), key=lambda r: (int(r['Rank']), int(r['TeamId'])))
    a_top = R.get('a_top') or {'teams': 12, 'days': ['2026-09-30']}   # recipe: A takes wins of the live top-N teams on these days
    top12 = []
    for r in lbrows:
        if int(r['TeamId']) != US and int(r['TeamId']) not in top12 and len(top12) < int(a_top['teams']): top12.append(int(r['TeamId']))
    inputs['a_top_team_ids'] = top12
    key = list(zip(w.episode_id.astype(int), w.seat.astype(int)))
    in_top4 = pd.Series([k in top4 for k in key], index=w.index)
    d30 = w.day == '2026-09-30'
    w['bucket'] = None
    w.loc[in_top4 | (w.day.isin(a_top['days']) & w.team_id.isin(top12)), 'bucket'] = 'A'
    if a_top.get('beaters'):   # TOP10-BEATERS (owner 11:3x): winners over a current top-10 team on a_top days -> A (L still overrides for us)
        w.loc[w.day.isin(a_top['days']) & w.opp_team_id.isin(top12), 'bucket'] = 'A'
    if R.get('beat_us_bucket'):                          # our losses (recipe beat_us_bucket, owner 07:1x): winners over us on beat_us_days
        if R['beat_us_bucket'] not in BUCKETS: raise SystemExit(f"REFUSED: beat_us_bucket {R['beat_us_bucket']} not a recipe bucket")
        w.loc[(w.opp_is_us == True) & w.day.isin(beat_us_days), 'bucket'] = R['beat_us_bucket']
    xb = R.get('x_bucket') or {}
    if xb:   # owner 14:4x: winners over the NEWEST submission of a team ranked lo..hi (us excluded), today; X wins over any other bucket but L
        rk = {int(r['TeamId']): int(r['Rank']) for r in lbrows if int(r['TeamId']) != US}
        lo, hi = xb['ranks']; teams_x = {t for t, k in rk.items() if lo <= k <= hi}
        sub_all = pd.concat([s[['team_id', 'submission_id']].rename(columns={'team_id': 't', 'submission_id': 'u'}),
                             s[['opp_team_id', 'opp_submission_id']].rename(columns={'opp_team_id': 't', 'opp_submission_id': 'u'})]).dropna()
        newest = sub_all[sub_all.t.isin(teams_x)].groupby('t').u.max().to_dict()
        xm = w.day.isin(xb['days']) & w.opp_team_id.isin(teams_x) & (w.opp_is_us != True) & \
             pd.Series([newest.get(t) is not None and o == newest.get(t) for t, o in zip(w.opp_team_id, w.opp_submission_id)], index=w.index, dtype=bool)
        w.loc[xm, 'bucket'] = 'X'
        inputs['x_newest_subs'] = {str(int(k)): int(v) for k, v in newest.items()}
    mb = R.get('m_bucket') or {}
    if mb:   # owner 14:2x "dominating middle people": winner seats of today's games where a seat's team is ranked lo..hi (us excluded)
        rk = {int(r['TeamId']): int(r['Rank']) for r in lbrows if int(r['TeamId']) != US}
        lo, hi = mb['ranks']; mids = {t for t, k in rk.items() if lo <= k <= hi}
        w.loc[w.bucket.isna() & w.day.isin(mb['days']) & (w.team_id.isin(mids) | w.opp_team_id.isin(mids)), 'bucket'] = 'M'
        inputs['m_team_ids'] = sorted(mids)
    w.loc[w.bucket.isna() & d30, 'bucket'] = merge.get('B', 'B')
    w.loc[w.bucket.isna() & (w.day == '2026-09-29'), 'bucket'] = merge.get('C', 'C')
    inf = float('inf')
    at = (w.game_score + w.opp_game_score) / 2; at = at.where(at.abs() != inf)
    lbq = (w.rating + w.opp_rating) / 2; lbq = lbq.where(lbq.abs() != inf)
    w['quality'] = at.where(at.notna(), lbq)
    w['quality_src'] = ['at-time' if x == x else ('leaderboard' if y == y else 'none') for x, y in zip(at, lbq)]
    kept = {}
    for bucket, lo, hi in (('D', '2026-09-24', '2026-09-28'), ('E', '2026-09-03', '2026-09-23')):
        pool = w[w.bucket.isna() & (w.day >= lo) & (w.day <= hi)]
        ex[f'excluded_{bucket}_no_finite_quality'] = int(pool.quality.isna().sum())
        pool = pool[pool.quality.notna()]                # unrated rows are never 'top 10%'
        for day, g in pool.groupby('day'):
            g = g.sort_values(['quality', 'episode_id'], ascending=[False, True], na_position='last')
            k = max(1, int(round(0.10 * len(g))))
            w.loc[g.index[:k], 'bucket'] = bucket
            npool = R.get('n_pool') or {}
            if npool:   # N flow-only extra seats (coordinator 13:4x): ranks top-10%..top-25% of the day -> N_D / N_E (lineage N only)
                k2 = max(k, int(round(float(npool.get('top_frac', 0.25)) * len(g))))
                w.loc[g.index[k:k2], 'bucket'] = 'N_' + bucket
            kept[day] = {'pool': int(len(g)), 'kept': k, 'at_time_share': round(float((g.head(k).quality_src == 'at-time').mean()), 3)}
    # W_<lineage> (owner 09-30 08:5x "it reinforces our current policy"): OUR winning seats of 09-29/30 hosted games, per lineage;
    # each W bucket is consumed only by its own lineage's roster (lineage_views); our losing seats are never targets
    owc = R.get('own_win_buckets') or {}
    if owc:
        smap = {int(x): b for b, v in owc['buckets'].items() for x in v}
        # our seat's submission: SEATS.submission_id is empty for most hosted seats, so take it from the hosted-ours manifest
        # (our_sub, else the deployed version name -> submission learned from rows that carry both)
        hm = [json.loads(l) for l in open(f'{CAT}/../extra/hosted-ours/MANIFEST.jsonl') if l.strip()]
        v2s = {r['version']: int(r['our_sub']) for r in hm if r.get('our_sub') and r.get('version')}
        vpre = {k: int(v) for k, v in (owc.get('version_subs') or {}).items()}
        ep_sub = {}
        for r in hm:
            sub = int(r['our_sub']) if r.get('our_sub') else v2s.get(r.get('version'))
            if sub is None: sub = next((v for k, v in vpre.items() if str(r.get('version') or '').startswith(k)), None)
            if sub is not None: ep_sub[int(r['episode_id'])] = sub   # our seat itself comes from SEATS.is_us
        ow = s[(s.kaggle == True) & (s.result == 'W') & (s.is_us == True) & (s.opp_is_us != True) & (s.team_id != s.opp_team_id)
               & s.day.isin(owc['days'])].copy()
        ow['submission_id'] = [ep_sub.get(int(e), x) for e, x in zip(ow.episode_id, ow.submission_id)]
        ow = ow[ow.submission_id.isin(list(smap))]
        ow = ow[~ow.episode_id.isin(hold) & (ow.static != True) & (ow.tsv_ready == True)]
        ow = ow[[int(z) not in brk.get(int(e), {}) for e, z in zip(ow.episode_id, ow.seat)]]
        ow = ow[pd.Series([T.in_split(int(e), 'train') for e in ow.episode_id], index=ow.index, dtype=bool)]
        ow['bucket'] = [smap[int(x)] for x in ow.submission_id]; ow['quality'] = float('nan'); ow['quality_src'] = 'own'
        ex['own_win_selected'] = {k: int(v) for k, v in ow.bucket.value_counts().items()}
        w = pd.concat([w, ow[w.columns]])
    # self-play buckets (owner 13:3x): winner seats of our own zsp-* league streams; not catalog rows, so they join here with
    # neutral catalog fields; the gate checks bank worlds in sp_buckets.select, the rest below
    spc = {k: v for k, v in (R.get('selfplay_buckets') or {}).items() if isinstance(v, dict) and 'tags' in v}
    sprep = {}
    if spc:
        sys.path.insert(0, BR); import sp_buckets as SPB
        sprows, sprep = SPB.select(spc)
        extra = []
        for b, rs in sprows.items():
            for r in rs:
                extra.append({**{c: None for c in w.columns}, 'archive': r['archive'], 'member': r['member'], 'day': r['day'],
                              'episode_id': r['episode_id'], 'seat': r['seat'], 'team': r['team'], 'team_id': -1, 'opp_team_id': -2,
                              'kaggle': False, 'result': 'W', 'is_us': False, 'opp_is_us': False, 'tsv_ready': True, 'static': False,
                              'bucket': b, 'quality_src': 'selfplay'})
        if extra: w = pd.concat([w, pd.DataFrame(extra, columns=w.columns)], ignore_index=True)
    w = w[w.bucket.notna()].drop_duplicates(['episode_id', 'seat'])
    w['cached'] = pd.Series([os.path.exists(f'{C}/{int(e)}-{int(x)}.stp') and os.path.getsize(f'{C}/{int(e)}-{int(x)}.stp') > 0
                             for e, x in zip(w.episode_id, w.seat)], index=w.index, dtype=bool)
    # a self-play bucket below min_seats cached is not published this build (its seats wait in UNCACHED-<b>.tsv): no seats
    # without a share, no 0.0 share (N flow roster.py refuses shares <= 0)
    _spmin = int((R.get('selfplay_buckets') or {}).get('min_seats', 100))
    for _b in [k for k, v in (R.get('selfplay_buckets') or {}).items() if isinstance(v, dict) and 'tags' in v] + (['M'] if R.get('m_bucket') else []) + (['X'] if R.get('x_bucket') else []):
        _m = (w.bucket == _b)
        if int((_m & w.cached).sum()) < _spmin: w.loc[_m, 'cached'] = False
    # a neutral bucket with no publishable seats this build is OMITTED ENTIRELY: not in the files, the hash or identity_json
    # (N flow roster.py requires identity buckets == effective-share keys and refuses shares <= 0; ADV-CHECK-1600)
    _dyn = R.get('dynamic_shares') or {}
    omitted = [b for b in BUCKETS if _dyn.get(b) != 'lineage_only' and int(((w.bucket == b) & w.cached).sum()) == 0]
    omitted_uncached = {b: w[(w.bucket == b) & ~w.cached] for b in omitted}
    BUCKETS = [b for b in BUCKETS if b not in omitted]
    R = {**R, 'shares': {k: v for k, v in R['shares'].items() if k not in omitted},
         'dynamic_shares': {k: v for k, v in _dyn.items() if k not in omitted}}
    body = lambda df: 'archive\tmember\tday\teid\tseat\tweight\treason\n' + ''.join(
        f"{r.archive}\t{r.member}\t{r.day}\t{int(r.episode_id)}\t{int(r.seat)}\t1\tbucket{r.bucket}:{r.team}\n" for r in df.itertuples(index=False))
    # admission gate on the emitted seats (ASTRA-PRO-CHECK-BUCKETS-0930 P0-1): recompute every exclusion family
    em = w[w.cached]
    top4_val = ids_from(f'{SP}/TOP4-BEATERS-VAL.eids', allow_empty=True)
    fams = {'leak': em.episode_id.isin(leak), 'holdout_val_sealed_seed': em.episode_id.isin(hold), 'top4_val': em.episode_id.isin(top4_val),
            'static': em.static == True, 'blocked_sub': em.submission_id.isin(BLOCKED_SUBS), 'ours': (em.is_us == True) & ~em.bucket.astype(str).str.startswith('W_'),
            'mirror': em.team_id == em.opp_team_id, 'not_train_split': pd.Series([(not str(b).startswith('S_')) and not T.in_split(int(e), 'train') for e, b in zip(em.episode_id, em.bucket)], index=em.index),
            'duplicate_seat': em.duplicated(['episode_id', 'seat'], keep=False),
            'beat_us_outside_days': (em.opp_is_us == True) & ~em.day.isin(beat_us_days)}
    # one versioned union of every never-train list (coordinator 07:1x): its sha goes into the receipt, and it is a gate family
    import importlib.util
    _sp = importlib.util.spec_from_file_location('exclusion_union', f'{BR}/exclusion_union.py'); XU = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(XU)
    xmeta = XU.build()
    xu = pd.read_csv(xmeta['path'], sep='\t', dtype={'seat': str})
    xe = set(xu[xu.seat == '*'].eid.astype(int)); xq = set(zip(xu[xu.seat != '*'].eid.astype(int), xu[xu.seat != '*'].seat.astype(int)))
    fams['exclusion_union'] = em.episode_id.isin(xe) | pd.Series([(int(e), int(x)) in xq for e, x in zip(em.episode_id, em.seat)], index=em.index, dtype=bool)
    gate = {k: int(v.sum()) for k, v in fams.items()}
    gate_fail = {k: v for k, v in gate.items() if v}
    # generation identity binds the sampling semantics (recipe schema/version, shares, merge, cap rule, beat-us days, reserve)
    # to the published bytes: one ID can never stand for two sampling contracts (PRO VERIFY-FIX-BUCKETS)
    ident = json.dumps({k: R.get(k) for k in ('schema', 'version', 'shares', 'merge', 'small_bucket_cap', 'beat_us_days', 'beat_us_bucket', 'dynamic_shares', 'reserved', 'sampling', 'a_top', 'own_win_buckets')},
                       sort_keys=True, separators=(',', ':'))
    full = ''.join(body(em[em.bucket == b].sort_values(['day', 'episode_id', 'seat'])) for b in BUCKETS)
    sha = hashlib.sha256((ident + '\n' + full).encode()).hexdigest()
    final = f'{BR}/buckets/{sha[:12]}'
    out = f'{BR}/buckets/.tmp-{sha[:12]}-{os.getpid()}'
    os.makedirs(out)
    rec = {'sha256': sha, 'dir': final, 'built_utc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           'leaderboard_snapshot': os.path.basename(snap), 'top4_train': t4p if top4 else None, 'qualifying_winner_seats': n0, **ex, 'buckets': {}}
    refuse = []
    for b, g in omitted_uncached.items():   # omitted buckets' seats still get queued for caching
        open(f'{out}/UNCACHED-{b}.tsv', 'w').write(body(g.sort_values(['day', 'episode_id', 'seat'])))
    for b in BUCKETS:
        g = w[w.bucket == b].sort_values(['day', 'episode_id', 'seat'])
        open(f'{out}/{b}.tsv', 'w').write(body(g[g.cached]))
        open(f'{out}/UNCACHED-{b}.tsv', 'w').write(body(g[~g.cached]))
        n, c = len(g), int(g.cached.sum())
        rec['buckets'][b] = {'share': shares[b], 'seats': n, 'cached': c, 'uncached': n - c,
                             'days': [str(g.day.min()), str(g.day.max())] if n else None,
                             'quality_src': g.quality_src.value_counts().to_dict() if b in 'DE' else None}
        if n and (n - c) > 0.10 * n and (R.get('dynamic_shares') or {}).get(b) not in ('lineage_only', 'selfplay_fixed'): refuse.append(f'{b}: {n - c}/{n} uncached (> 10%)')
    live = {b: v['share'] for b, v in rec['buckets'].items() if v['cached'] > 0}
    rec['recipe'] = {'path': RP, 'sha256': hashlib.sha256(open(RP, 'rb').read()).hexdigest(), 'shares': R['shares'], 'reserved': R.get('reserved', {})}
    tot = sum(v for b, v in live.items() if b not in (R.get('dynamic_shares') or {}))
    base = {b: v / tot for b, v in live.items()} if tot else {}
    # dynamic buckets (coordinator 07:2x, bucket L = winners over us): share set so the bucket's per-seat draw rate is exactly
    # mult x A's per-seat rate AFTER the proportional rescale of the others: x = k/(1+k), k = mult * base_A * n_L / n_A
    dyn = {}
    lin_only = [b for b, r in (R.get('dynamic_shares') or {}).items() if r in ('lineage_only', 'selfplay_fixed')]
    for b in lin_only: base.pop(b, None); live.pop(b, None)
    for b, rule in (R.get('dynamic_shares') or {}).items():
        if rule in ('lineage_only', 'selfplay_fixed'): continue
        if rule != 'cap_max_vs_A': raise SystemExit(f'REFUSED: unknown dynamic share rule {b}: {rule}')
        if b in live and 'A' in live and rec['buckets']['A']['cached'] > 0:
            mult = float((R.get('small_bucket_cap') or {}).get('multiplier', 4))
            others = {k: v for k, v in base.items() if k != b}; ot = sum(others.values())
            kk = mult * (others['A'] / ot) * rec['buckets'][b]['cached'] / rec['buckets']['A']['cached']
            x = kk / (1 + kk)
            base = {k: v / ot * (1 - x) for k, v in others.items()}; base[b] = x
            dyn[b] = {'rule': rule, 'multiplier': mult, 'n': rec['buckets'][b]['cached'], 'n_A': rec['buckets']['A']['cached'], 'share': round(x, 6)}
    # small-bucket cap (owner 04:39, recipe v3): r_A = base_A / n_A; for k != A: share_k <= mult * r_A * n_k; all excess to A;
    # one pass from the pre-cap r_A (no fixed point). S is outside this (own rule, test branches only).
    cap = R.get('small_bucket_cap') or {}
    eff = dict(base)
    capped = {}
    if cap and 'A' in base and rec['buckets']['A']['cached'] > 0:
        mult = float(cap.get('multiplier', 4)); rA = base['A'] / rec['buckets']['A']['cached']
        excess = 0.0
        for b in live:
            if b == 'A' or b in dyn: continue
            lim = mult * rA * rec['buckets'][b]['cached']
            if base[b] > lim:
                capped[b] = {'base': round(base[b], 6), 'cap': round(lim, 6)}
                excess += base[b] - lim; eff[b] = lim
        eff['A'] = base['A'] + excess
    eff = {b: v for b, v in eff.items() if v > 0}; base = {b: v for b, v in base.items() if v > 0}
    rec['base_shares'] = {b: round(v, 6) for b, v in base.items()}
    rec['effective_shares'] = {b: round(v, 6) for b, v in eff.items()}
    rec['capped'] = capped
    rec['dynamic_shares'] = dyn
    # lineage views: neutral effective shares + that lineage's W bucket at min(W cap, x) with x = k/(1+k), k = mult*e_A*n_W/n_A,
    # everything else scaled by (1 - x): W's per-seat rate is exactly mult x A's unless the .05 cap binds
    # self-play shares (fixed fraction of the total, other buckets scaled by 1 - sum): a bucket below min_seats cached gets 0;
    # per-seat rate capped at cap_mult x A's (after scaling); one pass
    sp_eff = {}
    _dy = R.get('dynamic_shares') or {}
    fixed = {**{b: c for b, c in spc.items()}, **({'M': {'share': float(R['m_bucket']['share'])}} if R.get('m_bucket') and _dy.get('M') == 'selfplay_fixed' else {}),
             **({'X': {'share': float(R['x_bucket']['share'])}} if R.get('x_bucket') and _dy.get('X') == 'selfplay_fixed' else {})}
    if fixed and 'A' in eff and rec['buckets']['A']['cached'] > 0:
        mn = int((R.get('selfplay_buckets') or {}).get('min_seats', 100)); cm = float((R.get('selfplay_buckets') or {}).get('cap_mult', 4))
        want = {b: (float(c['share']) if rec['buckets'].get(b, {}).get('cached', 0) >= mn else 0.0) for b, c in fixed.items()}
        st = sum(want.values())
        for b, v in want.items():
            cap_b = cm * (eff['A'] * (1 - st) / rec['buckets']['A']['cached']) * rec['buckets'].get(b, {}).get('cached', 0)
            sp_eff[b] = min(v, cap_b)
        sp_eff = {b: v for b, v in sp_eff.items() if v > 0}   # never emit a 0.0 share (N flow roster.py refuses shares <= 0)
        tot_sp = sum(sp_eff.values())
        eff = {k: v * (1 - tot_sp) for k, v in eff.items()}; eff.update(sp_eff)
        rec['effective_shares'] = {b: round(v, 6) for b, v in eff.items()}
    rec['omitted_zero_share_buckets'] = sorted(set(omitted) | {b for b in BUCKETS if b not in eff and not str(b).startswith(('W_', 'N_'))})
    rec['selfplay'] = {'config': spc, 'selection': sprep, 'shares': {b: round(v, 6) for b, v in sp_eff.items()},
                       'min_seats': (R.get('selfplay_buckets') or {}).get('min_seats')}
    views = {}
    npool = R.get('n_pool') or {}
    for lin, b in (owc.get('lineages') or {}).items() if not npool else list((owc.get('lineages') or {}).items()):
        nW = rec['buckets'].get(b, {}).get('cached', 0)
        if not nW or 'A' not in eff: continue
        mult = float((R.get('small_bucket_cap') or {}).get('multiplier', 4))
        kk = mult * eff['A'] * nW / rec['buckets']['A']['cached']; x = min(float(owc.get('max_share', .05)), kk / (1 + kk))
        v = {k: round(val * (1 - x), 6) for k, val in eff.items()}; v[b] = round(x, 6)
        if npool and lin == npool.get('lineage', 'N'):   # D/E share spread uniformly per seat over D+N_D and E+N_E
            for base_b in ('D', 'E'):
                nb, nx = rec['buckets'].get(base_b, {}).get('cached', 0), rec['buckets'].get('N_' + base_b, {}).get('cached', 0)
                if base_b in v and nx:
                    tot_b = v[base_b]; v[base_b] = round(tot_b * nb / (nb + nx), 6); v['N_' + base_b] = round(tot_b * nx / (nb + nx), 6)
        views[lin] = {'bucket': b, 'n': nW, 'share': round(x, 6), 'rate_vs_A': round((x / nW) / (v['A'] / rec['buckets']['A']['cached']), 4), 'effective_shares': v}
    rec['lineage_views'] = views
    sres = float(((R.get('reserved') or {}).get('S') or {}).get('share', 0))
    rec['shares_incl_S_reserve'] = {**{b: round(v * (1 - sres), 6) for b, v in eff.items()}, 'S': sres}
    per_seat = {b: (eff[b] / rec['buckets'][b]['cached']) if rec['buckets'][b]['cached'] else None for b in eff}
    eff_doc = {'schema': 'effective-shares/v1', 'generation': None, 'recipe_sha256': None, 'rule': cap.get('rule'), 'dynamic_shares': dyn,
               'shares_incl_S_reserve': rec['shares_incl_S_reserve'], 'lineage_views': views,
               'base_shares_normalised_A_to_E': rec['base_shares'], 'effective_shares': rec['effective_shares'], 'capped': capped,
               'per_seat_draw_rate_relative_to_A': {b: (round(v / per_seat['A'], 4) if v is not None and per_seat.get('A') else None) for b, v in per_seat.items()},
               'cached_seats': {b: rec['buckets'][b]['cached'] for b in BUCKETS}, 'files': [f'{b}.tsv' for b in BUCKETS],
               'note': 'trainers sample buckets by effective_shares (not the raw recipe), then a seat uniformly within the bucket'}
    rec['empty_buckets'] = [b for b in BUCKETS if rec['buckets'][b]['cached'] == 0]
    rec['top10_per_day'] = kept
    rec['admission_gate'] = {'families': gate, 'status': 'FAIL' if gate_fail else 'PASS'}
    rec['opp_is_us_winners_kept'] = {b: int(((w.bucket == b) & w.cached & (w.opp_is_us == True)).sum()) for b in BUCKETS}
    rec['opp_is_us_winners_selected'] = {b: int(((w.bucket == b) & (w.opp_is_us == True)).sum()) for b in BUCKETS}
    rec['hosted_ours_source_rows'] = {b: int(((w.bucket == b) & w.cached & w.archive.astype(str).str.contains('hosted-ours')).sum()) for b in BUCKETS}
    sup = w.groupby(['bucket', 'day']).agg(selected=('cached', 'size'), cached=('cached', 'sum'))
    rec['support_per_day'] = {f'{b}:{d}': {'selected': int(r.selected), 'cached': int(r.cached)} for (b, d), r in sup.iterrows()}
    rec['zero_support_days'] = [f'{b}:{d}' for (b, d), r in sup.iterrows() if r.cached == 0]
    rec['inputs'] = inputs
    rec['hash_of'] = 'sha256(identity_json + newline + exact bytes of the published bucket files (cached seats) concatenated in recipe order); UNCACHED-*.tsv, RECEIPT.json and EFFECTIVE-SHARES.json are sidecars outside the hash'
    rec['identity_json'] = ident
    rec['policy_sha256'] = hashlib.sha256(ident.encode()).hexdigest()   # readers refuse a generation whose policy differs from the current recipe
    rec['breakage'] = breakage
    if 'X' in set(w.bucket.dropna()):
        _x = w[(w.bucket == 'X') & w.cached]
        rec['x_bucket'] = {'cached_seats': int(len(_x)), 'top_winner_teams': _x.team.value_counts().head(10).to_dict(), 'newest_subs_of_losers': len(inputs.get('x_newest_subs', {}))}
    sys.path.insert(0, BR); import exclusion_union as _XU
    _held = _XU.held_out_ids(); _hs = {e for v in _held.values() for e in v}
    _bad = sorted(set(int(e) for e in em.episode_id) & _hs)
    if _bad: raise SystemExit(f'REFUSED: held-out episodes in the generation: {_bad[:10]}')
    rec['held_out_check'] = {'sets': {k: len(v) for k, v in _held.items()}, 'present_in_generation': 0}
    rec['exclusion_union'] = {'sha256': xmeta['sha256'], 'path': xmeta['path'], 'families': xmeta['families']}
    t4m = json.load(open(f'{SP}/TOP4-BEATERS-LATEST/MANIFEST.json'))
    rec['top4_val'] = {'episodes_reserved': len(top4_val), 'val_seats': t4m.get('val_seats'), 'train_seats': t4m.get('train_seats'),
                       'val_pct_of_episodes': t4m.get('val_pct_of_episodes'), 'target_pct': 20, 'contaminated': t4m.get('val_contaminated_count')}
    if gate_fail: refuse.append(f'admission gate: {gate_fail}')
    if len(em) == 0: refuse.append('empty selection: 0 cached seats')
    if os.path.exists(final) and ''.join(open(f'{final}/{b}.tsv').read() if os.path.exists(f'{final}/{b}.tsv') else '' for b in BUCKETS) != full:
        refuse.append(f'existing {final} bucket bytes differ from its generation hash')
    rec['publish'] = 'REFUSED: ' + '; '.join(refuse) if refuse else ('skipped (--no-publish)' if a.no_publish else 'PUBLISHED')
    rec['seconds'] = round(time.time() - t0, 1)
    eff_doc['generation'] = sha[:12]; eff_doc['recipe_sha256'] = rec['recipe']['sha256']
    json.dump(eff_doc, open(f'{out}/EFFECTIVE-SHARES.json', 'w'), indent=1)
    json.dump(rec, open(f'{out}/RECEIPT.json', 'w'), indent=1, default=str)
    import shutil
    if refuse or a.no_publish:
        dest = f'{BR}/buckets/REFUSED-{sha[:12]}-{time.strftime("%Y%m%dT%H%M%S")}' if refuse else f'{BR}/buckets/DRYRUN-{sha[:12]}-{time.strftime("%Y%m%dT%H%M%S")}'
        os.rename(out, dest); out = dest                # never overwrites a published generation
    else:
        if os.path.exists(final):                       # same hash = identical published bytes: keep the existing directory untouched
            shutil.rmtree(out); out = final
        else:
            os.rename(out, final); out = final
        tmp = f'{BR}/LATEST-BUCKETS.tmp'
        if os.path.lexists(tmp): os.remove(tmp)
        os.symlink(out, tmp); os.replace(tmp, f'{BR}/LATEST-BUCKETS')
    rec['dir_written'] = out
    print(json.dumps({k: rec[k] for k in ('dir', 'dir_written', 'buckets', 'effective_shares', 'empty_buckets', 'publish', 'top4_val', 'seconds')}, default=str))
    if rec['publish'].startswith('REFUSED'): sys.exit(2)


if __name__ == '__main__':
    main()
