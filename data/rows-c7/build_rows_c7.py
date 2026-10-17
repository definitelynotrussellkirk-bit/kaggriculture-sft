#!/usr/bin/env python3
"""Chunk 7-8 SFT roster (PRO-SFT-FORWARD-0929 recipe) -> base-v1-stepcache-v11/rows-c7.tsv + rows-c7.report.json.

Run (desktop, shared):
  cd /mnt/ai/scratch/farmNEAT/base-v1-pack/nfs/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc
  systemd-run --user --scope -q -p MemoryMax=10G nice -n 19 env OMP_NUM_THREADS=3 PYTHONPATH=.:../../../../src \
    /usr/bin/python3 -B /home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/build_rows_c7.py [--dry-run]
The same command is the ~10:00 refresh for chunk 8 (reads the hourly-rebuilt SEATS.parquet; same rules and caps).

Components (weighted mass = sum(weight): the step-cache loader repeats a seat `weight` times per pass), overlap
precedence hard > league > core > lbonly > anchor, one row per (eid, seat):
  hard    newest 5 game days; winner over a current top-30 team or over us; policy score >= 2880; margin >= 2500;
          weight 2, or 3 if margin >= 10000 or (beat us and margin >= 5000). Newest day first.
  league  local-league winners, margin >= 2500, weight 1; base1-vs-base1 and weight twins dropped; tiers: external
          anchor (v4b/pB/fwd1) over base1 > base1 over an external anchor > external anchor over others > other lineages
          > base1 over others; tiers taken in order, each tier interleaved by day in proportion to its eligible rows.
  core    newest 5 game days; policy score >= 2880; margin >= 2500; weight 1, 2 if margin >= 10000. Newest day first.
  lbonly  newest 5 game days; no policy score, team lb_score >= 2880, margin >= 2500; weight 1; capped at 5%.
  anchor  2026-09-07 .. day before the recent window; game_score >= 2880 (sub_score when no game score); margin >= 2500;
          weight 1; per-day quotas proportional to eligible rows; beat-top30 rows then built rows first within a day.
Policy score = sub_score (the seat's submission's public score), else game_score (rating after the game); never the
team's current leaderboard score. No quadrant selection.
Caps on total weighted mass T: team <= 10%, submission <= 5%, day <= 25%, league <= 5%, lbonly <= 5%; solved as a
fixed point on T. Exclusions: trainer data.in_split(eid, 'train'), data.sealed_ids(), the practice-v1 VAL episodes,
episodes on a decode-h2h holdout world seed (HOLDOUT-EPISODES.txt via catalog EPISODES.seed), decode-h2h/decode-exp-0929 sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

CAT = Path('/mnt/ai/scratch/farmNEAT/primitive-bc-data/catalog')
CACHE = Path('/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11')
TRAINER = Path('/mnt/ai/scratch/farmNEAT/base-v1-pack/nfs/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc')
PRACTICE = Path('/mnt/ai/scratch/farmNEAT/practice-v1')
FIELDS = ('archive', 'member', 'day', 'eid', 'seat', 'weight', 'reason')
STEPS_PER_SEAT = 719
BASE1 = 'base1'
EXTERNAL = ('v4b', 'fwd1')          # plus any version containing the pB lineage tag (v5w-pB-*, pilotB-*)


def digest(seed, *parts):
    return hashlib.sha256('\x1f'.join(map(str, (seed,) + parts)).encode()).hexdigest()


HOLDOUT = Path('/mnt/ai/scratch/farmNEAT/decode-h2h/HOLDOUT-EPISODES.txt')
BLOCKED_SOURCES = ('decode-h2h', 'decode-exp-0929')   # final-selection bracket / decode experiments: never training data


def holdout_blocked_eids():
    """Episode ids played on a decode-h2h holdout world seed (catalog EPISODES.seed). Fails closed if the list is missing."""
    import pyarrow.parquet as pq
    seeds = {int(l.split('\t')[0]) for l in HOLDOUT.read_text().splitlines() if l.strip() and not l.startswith('#')}
    if not seeds:
        raise SystemExit(f'{HOLDOUT}: no seeds')
    ep = pq.read_table(CAT / 'EPISODES.parquet', columns=['episode_id', 'seed']).to_pandas()
    return {int(e) for e, s in zip(ep['episode_id'], ep['seed']) if s == s and s is not None and int(s) in seeds}, len(seeds)


def blocked_source(archive, sources=None):
    text = str(archive) + ' ' + ' '.join(map(str, sources if sources is not None else ()))
    return any(b in text for b in BLOCKED_SOURCES)


def val_episode_ids():
    man = json.loads((PRACTICE / 'MANIFEST.json').read_text())
    tsv = PRACTICE / 'val' / 'VAL.tsv'
    sha = hashlib.sha256(tsv.read_bytes()).hexdigest()
    if sha != man['val_tsv_sha256']:
        raise SystemExit(f'practice VAL.tsv sha256 {sha} != MANIFEST {man["val_tsv_sha256"]}')
    ids = {int(l.split('\t')[3]) for l in tsv.read_text().splitlines()[1:] if l.strip()}
    if len(ids) != man['val_episodes']:
        raise SystemExit(f'practice VAL: {len(ids)} ids, MANIFEST says {man["val_episodes"]}')
    return ids


def version_core(team):
    """'SELF:v4b-g2-48236-land-a1fx' -> 'v4b-g2-48236' (weights identity: rules/decoder suffixes dropped)."""
    v = str(team or '')
    v = v[5:] if v.startswith('SELF:') else v
    m = re.match(r'^(.*?-u?\d{4,6})(?:-|$)', v)
    return m.group(1) if m else v


def family(team):
    c = version_core(team)
    if c.startswith(BASE1):
        return 'base1'
    if c.startswith(EXTERNAL) or re.search(r'(^|-)p(ilot)?B(-|$)', c):
        return 'external'
    return 'other'


def load(args):
    import pyarrow as pa
    pa.set_cpu_count(3)
    pa.set_io_thread_count(3)
    import pyarrow.parquet as pq
    cols = ['episode_id', 'seat', 'day', 'kaggle', 'config_ok', 'tsv_ready', 'self_play', 'result', 'is_us', 'opp_is_us',
            'team', 'team_id', 'submission_id', 'opp_team', 'opp_team_id', 'margin', 'sub_score', 'game_score',
            'lb_score', 'lb_rank', 'opp_lb_rank', 'rating', 'rating_from', 'final_quadrants', 'archive', 'member',
            'sources']
    return pq.read_table(CAT / 'SEATS.parquet', columns=cols).to_pandas()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(CACHE / 'rows-c7.tsv'))
    ap.add_argument('--report', default=None, help='default: <out stem>.report.json')
    ap.add_argument('--unbuilt-out', default=None, help='default: <out stem>.unbuilt.tsv (rows without a .stp)')
    ap.add_argument('--total', type=int, default=17000, help='nominal weighted mass the component targets refer to')
    ap.add_argument('--hard', type=float, default=0.20)
    ap.add_argument('--league', type=float, default=0.05)
    ap.add_argument('--core', type=float, default=0.50)
    ap.add_argument('--anchor', type=float, default=0.25)
    ap.add_argument('--lbonly-cap', type=float, default=0.05)
    ap.add_argument('--team-cap', type=float, default=0.10)
    ap.add_argument('--sub-cap', type=float, default=0.05)
    ap.add_argument('--day-cap', type=float, default=0.25)
    ap.add_argument('--recent-days', type=int, default=5)
    ap.add_argument('--anchor-from', default='2026-09-07')
    ap.add_argument('--min-score', type=float, default=2880.0)
    ap.add_argument('--min-margin', type=float, default=2500.0)
    ap.add_argument('--steps-per-chunk', type=float, default=12.5e6)
    ap.add_argument('--seed', default='rows-c7-20260929')
    ap.add_argument('--dry-run', action='store_true', help='report only; write nothing')
    ap.add_argument('--tag', default='c7', help='reason prefix')
    ap.add_argument('--exclude-anchor-from', help='roster TSV: its anchor rows (reason <tag>:anchor) are not anchor '
                    'candidates (a fresh anchor sample for the next chunk); default off')
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    report_path = Path(a.report) if a.report else out.with_name(out.stem + '.report.json')
    unbuilt_path = Path(a.unbuilt_out) if a.unbuilt_out else out.with_name(out.stem + '.unbuilt.tsv')

    sys.path.insert(0, str(TRAINER))
    sys.path.insert(0, str(TRAINER.parents[3] / 'src'))
    import data as trainer_data                        # the trainer's own split / sealed rules
    import numpy as np
    import pandas as pd

    s = load(a)
    seats_mtime = os.path.getmtime(CAT / 'SEATS.parquet')
    sealed = set(trainer_data.sealed_ids())
    val_ids = val_episode_ids()
    eids = s['episode_id'].astype('int64')
    uniq = pd.unique(eids)
    train = {int(e) for e in uniq if trainer_data.in_split(int(e), 'train')}
    s['train'] = eids.isin(train)
    s['excl_sealed'] = eids.isin(sealed)
    s['excl_val'] = eids.isin(val_ids)
    blocked, n_hold_seeds = holdout_blocked_eids()
    s['excl_holdout'] = eids.isin(blocked) | pd.Series([blocked_source(a_, x_) for a_, x_ in zip(s['archive'], s['sources'])], index=s.index)
    s['team_key'] = np.where(s['team_id'].notna(), 'tid:' + s['team_id'].fillna(0).astype('int64').astype(str),
                             'name:' + s['team'].fillna('?').astype(str))
    s['sub_key'] = np.where(s['submission_id'].notna(),
                            'sub:' + s['submission_id'].fillna(0).astype('int64').astype(str),
                            'nosub:' + s['team_key'])
    s['is_league'] = s['sources'].map(lambda xs: 'league' in set(xs) if xs is not None else False)

    ok = s['tsv_ready'].eq(True) & s['train'] & ~s['excl_sealed'] & ~s['excl_val'] & ~s['excl_holdout'] & (s['result'] == 'W') \
        & s['day'].notna() & s['margin'].notna()
    pub = s[ok & s['kaggle'].eq(True) & s['config_ok'].eq(True) & ~s['self_play'].eq(True) & ~s['is_us'].eq(True)
            & ~s['is_league']].copy()
    lg = s[ok & s['is_league'] & s['team'].astype(str).str.startswith('SELF:')].copy()

    all_pub_days = sorted(d for d in s.loc[s['kaggle'].eq(True) & s['config_ok'].eq(True) & (s['result'] == 'W')
                                           & ~s['is_us'].eq(True), 'day'].dropna().unique()
                          if re.fullmatch(r'\d{4}-\d\d-\d\d', str(d)))
    recent = all_pub_days[-a.recent_days:]
    anchor_to = recent[0]

    # policy scores: recent = sub_score else game_score; anchor = game_score else sub_score; never lb_score
    pol_recent = pub['sub_score'].where(pub['sub_score'].notna(), pub['game_score'])
    pol_anchor = pub['game_score'].where(pub['game_score'].notna(), pub['sub_score'])
    pub['pol_src'] = np.where(pub['sub_score'].notna(), 'sub_score',
                              np.where(pub['game_score'].notna(), 'game_score',
                                       np.where(pub['lb_score'].notna(), 'lb_only', 'none')))
    in_recent = pub['day'].isin(recent)
    m_ok = pub['margin'] >= a.min_margin
    beat30 = pub['opp_lb_rank'].fillna(1e9) <= 30
    beatus = pub['opp_is_us'].eq(True)
    pub['beat30'] = beat30
    pub['beatus'] = beatus

    def rows_of(df, weight, comp, extra=None):
        recs = []
        for r, w in zip(df.itertuples(index=False), weight):
            recs.append({'archive': r.archive, 'member': r.member, 'day': r.day, 'eid': int(r.episode_id),
                         'seat': int(r.seat), 'weight': int(w), 'comp': comp, 'team_key': r.team_key,
                         'team': r.team, 'sub_key': r.sub_key, 'margin': float(r.margin),
                         'rating_from': getattr(r, 'pol_src', 'league'), 'q': r.final_quadrants,
                         'seats_rating_from': str(r.rating_from),
                         'beat30': bool(getattr(r, 'beat30', False)), 'beatus': bool(getattr(r, 'beatus', False)),
                         'opp_team': r.opp_team})
        if extra:
            for rec, e in zip(recs, extra):
                rec.update(e)
        return recs

    hard_df = pub[in_recent & (beat30 | beatus) & (pol_recent >= a.min_score) & m_ok]
    hard_w = np.where((hard_df['margin'] >= 10000) | (hard_df['opp_is_us'].eq(True) & (hard_df['margin'] >= 5000)), 3, 2)
    core_df = pub[in_recent & (pol_recent >= a.min_score) & m_ok]
    core_w = np.where(core_df['margin'] >= 10000, 2, 1)
    lbo_df = pub[in_recent & pub['sub_score'].isna() & pub['game_score'].isna() & (pub['lb_score'] >= a.min_score) & m_ok]
    anc_df = pub[(pub['day'] >= a.anchor_from) & (pub['day'] < anchor_to) & (pol_anchor >= a.min_score) & m_ok]
    prev_anchor = set()
    if a.exclude_anchor_from:
        for l in Path(a.exclude_anchor_from).read_text().splitlines()[1:]:
            f = l.split('\t')
            if len(f) > 6 and ':anchor:' in f[6]:
                prev_anchor.add((int(f[3]), int(f[4])))
        anc_df = anc_df[[(int(e), int(s)) not in prev_anchor for e, s in zip(anc_df['episode_id'], anc_df['seat'])]]

    # league: winners with margin >= 2500; drop base1-vs-base1 and weight twins; tier
    lg = lg[lg['margin'] >= a.min_margin].copy()
    lg['wfam'] = lg['team'].map(family)
    lg['lfam'] = lg['opp_team'].map(family)
    lg['twin'] = lg['team'].map(version_core) == lg['opp_team'].map(version_core)
    lg_drop_sib = (lg['wfam'] == 'base1') & (lg['lfam'] == 'base1')
    lg_drop_twin = lg['twin'] & ~lg_drop_sib
    lg_keep = lg[~lg_drop_sib & ~lg['twin']].copy()
    tier = np.select([(lg_keep['wfam'] == 'external') & (lg_keep['lfam'] == 'base1'),
                      (lg_keep['wfam'] == 'base1') & (lg_keep['lfam'] == 'external'),
                      (lg_keep['wfam'] == 'external'),
                      (lg_keep['wfam'] == 'other') & (lg_keep['lfam'] != 'base1'),
                      (lg_keep['wfam'] == 'other') & (lg_keep['lfam'] == 'base1'),
                      (lg_keep['wfam'] == 'base1')], [1, 2, 3, 4, 4, 5], 9)
    lg_keep['tier'] = tier
    lg_keep['pol_src'] = 'league'
    lg_keep['beat30'] = False
    lg_keep['beatus'] = False

    built = lambda eid, seat: (CACHE / f'{eid}-{seat}.stp').exists()

    cand = {
        'hard': rows_of(hard_df, hard_w, 'hard'),
        'league': rows_of(lg_keep, np.ones(len(lg_keep), int), 'league',
                          [{'tier': int(t), 'wver': version_core(w), 'lver': version_core(l)}
                           for t, w, l in zip(lg_keep['tier'], lg_keep['team'], lg_keep['opp_team'])]),
        'core': rows_of(core_df, core_w, 'core'),
        'lbonly': rows_of(lbo_df, np.ones(len(lbo_df), int), 'lbonly'),
        'anchor': rows_of(anc_df, np.ones(len(anc_df), int), 'anchor'),
    }
    for comp, recs in cand.items():
        for r in recs:
            r['built'] = built(r['eid'], r['seat'])
            r['h'] = digest(a.seed, comp, r['eid'], r['seat'])

    def newest_first(recs):
        return sorted(recs, key=lambda r: (-int(r['day'].replace('-', '')), r['h']))

    def proportional_by_day(recs, target_mass, key):
        """Day round-robin in proportion to each day's eligible rows (largest remainder), each day's queue by key."""
        by = defaultdict(list)
        for r in recs:
            by[r['day']].append(r)
        for d in by:
            by[d].sort(key=key)
        tot = sum(len(v) for v in by.values())
        if not tot:
            return []
        # interleave: emit rows so that after k emitted, day d has ~k*n_d/tot (stable, deterministic)
        order, taken = [], Counter()
        days = sorted(by)
        n = {d: len(by[d]) for d in days}
        for k in range(1, tot + 1):
            d = max(days, key=lambda d: (k * n[d] / tot - taken[d] if taken[d] < n[d] else -1e9, d))
            order.append(by[d][taken[d]])
            taken[d] += 1
        return order

    orders = {
        'hard': newest_first(cand['hard']),
        'league': [r for tier in sorted({r['tier'] for r in cand['league']})
                   for r in proportional_by_day([r for r in cand['league'] if r['tier'] == tier], None,
                                                key=lambda r: r['h'])],
        'core': newest_first(cand['core']),
        'lbonly': newest_first(cand['lbonly']),
        'anchor': proportional_by_day(cand['anchor'], None,
                                      key=lambda r: (not r['beat30'], not r['built'], r['h'])),
    }

    def select(T):
        cap_team, cap_sub, cap_day = a.team_cap * T, a.sub_cap * T, a.day_cap * T
        targets = {'hard': round(a.hard * a.total), 'league': min(round(a.league * a.total), math.floor(a.league * T)),
                   'core': round(a.core * a.total), 'lbonly': math.floor(a.lbonly_cap * T),
                   'anchor': round(a.anchor * a.total)}
        seen, chosen = set(), []
        tm, sm, dm = Counter(), Counter(), Counter()
        stats = {}
        for comp in ('hard', 'league', 'core', 'lbonly', 'anchor'):
            mass, rows, skip = 0, 0, Counter()
            for r in orders[comp]:
                k = (r['eid'], r['seat'])
                if k in seen:
                    skip['overlap'] += 1
                    continue
                w = r['weight']
                if mass + w > targets[comp]:
                    skip['target'] += 1
                    continue
                if comp != 'league' and tm[r['team_key']] + w > cap_team:
                    skip['team_cap'] += 1
                    continue
                if comp != 'league' and sm[r['sub_key']] + w > cap_sub:
                    skip['sub_cap'] += 1
                    continue
                if dm[r['day']] + w > cap_day:
                    skip['day_cap'] += 1
                    continue
                seen.add(k)
                chosen.append(r)
                mass += w
                rows += 1
                tm['league' if comp == 'league' else r['team_key']] += w
                if comp != 'league':
                    sm[r['sub_key']] += w
                dm[r['day']] += w
                if mass == targets[comp]:
                    break
            stats[comp] = {'target_mass': targets[comp], 'mass': mass, 'rows': rows,
                           'eligible_rows': len(cand[comp]), 'eligible_mass': sum(r['weight'] for r in cand[comp]),
                           'skipped': dict(skip), 'shortfall_mass': targets[comp] - mass}
        return chosen, stats

    T = float(a.total)
    for it in range(40):
        chosen, stats = select(T)
        tot = sum(r['weight'] for r in chosen)
        if abs(tot - T) < 1:
            break
        T = tot
    else:
        raise SystemExit('cap fixed point did not converge')
    tot = sum(r['weight'] for r in chosen)
    # verify caps against the final total
    tm, sm, dm, cm = Counter(), Counter(), Counter(), Counter()
    for r in chosen:
        if r['comp'] != 'league':
            tm[r['team_key']] += r['weight']
            sm[r['sub_key']] += r['weight']
        dm[r['day']] += r['weight']
        cm[r['comp']] += r['weight']
    keys = [(r['eid'], r['seat']) for r in chosen]
    checks = {
        'no_duplicate_seat': len(keys) == len(set(keys)),
        'team_le_10pct': max(tm.values()) <= a.team_cap * tot + 1e-9,
        'league_le_5pct': cm['league'] <= a.league * tot + 1e-9,
        'submission_le_5pct': max(sm.values()) <= a.sub_cap * tot + 1e-9,
        'day_le_25pct': max(dm.values()) <= a.day_cap * tot + 1e-9,
        'lbonly_le_5pct': cm['lbonly'] <= a.lbonly_cap * tot + 1e-9,
        'all_train_split': all(trainer_data.in_split(e, 'train') for e, _ in keys),
        'no_sealed': not any(e in sealed for e, _ in keys),
        'no_practice_val': not any(e in val_ids for e, _ in keys),
        'no_holdout_seed_or_decode_source': not any(e in blocked for e, _ in keys) and not any(blocked_source(r['archive']) for r in chosen),
        'policy_score_coverage_public': (sum(r['weight'] for r in chosen if r['rating_from'] in ('sub_score', 'game_score'))
                                         / max(1, sum(r['weight'] for r in chosen if r['comp'] != 'league'))),
    }
    failed = [k for k, v in checks.items() if v is False]
    if failed:
        raise SystemExit(f'roster checks failed: {failed}')

    # output order: component precedence, newest day first, digest (the cache builder walks rows in file order)
    comp_rank = {'hard': 0, 'league': 1, 'core': 2, 'lbonly': 3, 'anchor': 4}
    chosen.sort(key=lambda r: (comp_rank[r['comp']], -int(r['day'].replace('-', '')), r['h']))

    def reason(r):
        base = f"{a.tag}:{r['comp']}:{r['rating_from']}:m{int(r['margin'])}"
        if r['comp'] == 'hard':
            base += ':beatus' if r['beatus'] else ':beat30'
        if r['comp'] == 'league':
            base += f":t{r['tier']}:{r['wver']}>{r['lver']}"
        return f"{base}:{str(r['team']).replace(chr(9), ' ')}"

    def tsv(rows):
        lines = ['\t'.join(FIELDS) + '\n']
        for r in rows:
            lines.append(f"{r['archive']}\t{r['member']}\t{r['day']}\t{r['eid']}\t{r['seat']}\t{r['weight']}\t{reason(r)}\n")
        return ''.join(lines)

    def mass_by(key, rows=chosen, top=None):
        c = Counter()
        for r in rows:
            c[key(r)] += r['weight']
        items = c.most_common(top) if top else sorted(c.items())
        return {str(k): {'mass': v, 'share': round(v / tot, 4)} for k, v in items}

    team_names = {}
    for r in chosen:
        team_names.setdefault(r['team_key'], r['team'])
    mbin = lambda m: ('<2500' if m < 2500 else '2500-4999' if m < 5000 else '5000-9999' if m < 10000
                      else '10000-19999' if m < 20000 else '>=20000')
    built_rows = [r for r in chosen if r['built']]
    unbuilt = [r for r in chosen if not r['built']]
    exposures = a.steps_per_chunk / STEPS_PER_SEAT
    report = {
        'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'seats_parquet_mtime_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(seats_mtime)),
        'out': str(out), 'seed': a.seed, 'tag': a.tag,
        'excluded_previous_anchor_rows': len(prev_anchor), 'recent_days': recent, 'anchor_days': [a.anchor_from, f'< {anchor_to}'],
        'rows': len(chosen), 'weighted_mass': tot, 'nominal_total': a.total,
        'cap_fixed_point_T': T, 'components': stats,
        'component_share': {k: round(v / tot, 4) for k, v in cm.items()},
        'checks': checks,
        'mass_by_day': mass_by(lambda r: r['day']),
        'mass_by_day_component': {c: mass_by(lambda r: r['day'], [r for r in chosen if r['comp'] == c])
                                  for c in comp_rank},
        'mass_by_team_top25': {f"{team_names.get(k, k)} ({k})": v
                               for k, v in mass_by(lambda r: 'league(all SELF)' if r['comp'] == 'league'
                                                   else r['team_key'], top=25).items()},
        'distinct_teams': len({r['team_key'] for r in chosen if r['comp'] != 'league'}),
        'mass_by_submission_top15': mass_by(lambda r: r['sub_key'], [r for r in chosen if r['comp'] != 'league'], top=15),
        'distinct_submissions': len({r['sub_key'] for r in chosen if r['comp'] != 'league'}),
        'mass_by_rating_from': mass_by(lambda r: r['rating_from']),   # policy-score source used for admission
        'mass_by_seats_rating_from': mass_by(lambda r: r['seats_rating_from']),   # SEATS.rating_from column
        'mass_by_margin_bin': mass_by(lambda r: mbin(r['margin'])),
        'mass_by_weight': mass_by(lambda r: r['weight']),
        'mass_by_final_quadrants': mass_by(lambda r: 'nan' if r['q'] != r['q'] or r['q'] is None else int(r['q'])),
        'league_by_tier': mass_by(lambda r: r['tier'], [r for r in chosen if r['comp'] == 'league']),
        'league_eligible_by_tier': dict(sorted(Counter(r['tier'] for r in cand['league']).items())),
        'league_by_day': mass_by(lambda r: r['day'], [r for r in chosen if r['comp'] == 'league']),
        'league_by_winner_version_top15': mass_by(lambda r: r['wver'], [r for r in chosen if r['comp'] == 'league'], top=15),
        'league_dropped': {'base1_vs_base1': int(lg_drop_sib.sum()), 'weight_twins': int(lg_drop_twin.sum()),
                           'eligible_after_drop': len(lg_keep)},
        'hard_beatus_rows': sum(1 for r in chosen if r['comp'] == 'hard' and r['beatus']),
        'built': {'rows': len(built_rows), 'mass': sum(r['weight'] for r in built_rows),
                  'unbuilt_rows': len(unbuilt), 'unbuilt_mass': sum(r['weight'] for r in unbuilt),
                  'unbuilt_by_component': dict(Counter(r['comp'] for r in unbuilt)),
                  'unbuilt_by_day': dict(sorted(Counter(r['day'] for r in unbuilt).items()))},
        'passes': {'steps_per_seat_exposure': STEPS_PER_SEAT, 'scored_steps_per_chunk': a.steps_per_chunk,
                   'seat_exposures_per_chunk': round(exposures, 1),
                   'passes_per_chunk_if_all_built': round(exposures / tot, 3),
                   'passes_per_chunk_built_only_now': round(exposures / max(1, sum(r['weight'] for r in built_rows)), 3)},
        'exclusions': {'split': 'data.in_split(eid, "train")', 'sealed': len(sealed), 'practice_val_episodes': len(val_ids),
                       'holdout_seeds': n_hold_seeds, 'holdout_seed_episodes': len(blocked), 'blocked_sources': list(BLOCKED_SOURCES),
                       'trainer_data_py': str(TRAINER / 'data.py')},
        'seconds': round(time.time() - t0, 1),
    }
    text = json.dumps(report, indent=1, ensure_ascii=False, default=str) + '\n'
    if a.dry_run:
        print(text)
        return
    for path, body in ((out, tsv(chosen)), (unbuilt_path, tsv(unbuilt)), (report_path, text)):
        tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
        tmp.write_text(body, encoding='utf-8')
        os.replace(tmp, path)
    print(json.dumps({k: report[k] for k in ('rows', 'weighted_mass', 'component_share', 'built', 'passes')}, indent=1))


if __name__ == '__main__':
    main()
