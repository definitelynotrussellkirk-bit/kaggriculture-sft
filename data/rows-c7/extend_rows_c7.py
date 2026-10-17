#!/usr/bin/env python3
"""Owner data direction 2026-09-29 03:20-03:40 EDT: extend rows-c7.tsv in place (max-weight merge, atomic replace).

Added classes (winner seats only, public Kaggle games, not ours, reference configuration, zip-json copy stored):
  beattop   winner of a game a current top-30 team lost (BEAT-TOP30 view), day >= 2026-09-25: weight 3
            (the top-12-loser subset is reported separately)
  mmpq      every winner seat of every stored game of team 16681125 (M & M & P & Q): their wins and their beaters
  beatus    winner of every public game we lost, day >= 2026-09-26
  top12     wins of a current top-12 team (latest leaderboard), day >= 2026-09-28
  majkel    every win of Majkel1337 (team 16718819; owner 05:05-05:15 EDT); weight 3 for wins by his newest submission(s)
            (majkel_newest_subs), recency weights otherwise
Weights: beattop 3; the other classes by recency: 3 for day >= 2026-09-28, 2 for 09-25..09-27, 1 older. A seat already in
the roster keeps max(old, new) as its base weight (stored in the reason as ':b=N').
Opponent strength (owner 03:45 EDT, "let the WINS do most of the gating (WINS AGAINST THOSE THAT WIN OFTEN)"): every roster
row's final weight = max(base, tier) where tier = 3 if the beaten opponent's win share >= 0.70, 2 if >= 0.55, else 1.
Win share = wins / games of the loser's submission over all stored Kaggle games (SEATS.parquet); fewer than 20 stored games
-> the loser team's share (flag 't'); league rows -> the losing bot version's share over stored league games (flag 'l');
no record -> tier 1 (flag 'n'). Recomputed every run from the base, so tiers follow the hourly catalog. Reason tail:
':b=<base>:ot=<share><flag>'. Each written roster version is also kept as rows-c7-versions/rows-c7.<sha256[:12]>.tsv. Hard filter: trainer data.in_split(eid, 'train'), not data.sealed_ids(), not a
practice-v1 VAL episode, not a decode-h2h holdout world seed (HOLDOUT-EPISODES.txt via catalog EPISODES.seed) and
not from decode-h2h / decode-exp-0929 (counts of dropped seats reported; matching roster rows are removed). Idempotent: re-running adds only new seats / raises weights.

  cd /mnt/ai/scratch/farmNEAT/base-v1-pack/nfs/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc
  systemd-run --user --scope -q -p MemoryMax=10G nice -n 19 env OMP_NUM_THREADS=3 PYTHONPATH=.:../../../../src \
    /usr/bin/python3 -B /home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/extend_rows_c7.py [--dry-run]
Writes ROSTER (atomic), <roster stem>.extend.json (report, one entry appended per run in .extend-log.jsonl) and
<roster stem>.extend-unbuilt.tsv (rows of this run's added/raised seats without a built .stp, beattop/newest first).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

CAT = Path('/mnt/ai/scratch/farmNEAT/primitive-bc-data/catalog')
CACHE = Path('/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11')
TRAINER = Path('/mnt/ai/scratch/farmNEAT/base-v1-pack/nfs/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc')
PRACTICE = Path('/mnt/ai/scratch/farmNEAT/practice-v1')
FIELDS = ('archive', 'member', 'day', 'eid', 'seat', 'weight', 'reason')
MMPQ = 16681125
CLASSES = ('beattop', 'mmpq', 'majkel', 'beatus', 'top12')
MAJKEL = 16718819
MAJKEL_DIR = Path('/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/copy-majkel')
MIN_GAMES = 20
TAIL = re.compile(r':b=(\d+):ot=[^:\t]*$')


HOLDOUT = Path('/mnt/ai/scratch/farmNEAT/decode-h2h/HOLDOUT-EPISODES.txt')
MMPQ_VAL_DIR = Path('/mnt/ai/scratch/farmNEAT/mmpq-val')
BLOCKED_SUBS = {56680736}   # PRO-TEAM-PROFILE-VICTOR-TUFA-0930: different 4-quadrant policy, 2 games; quarantined from training
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


def majkel_newest_subs(window_hours=2.0):
    """Majkel1337's newest submission(s): those whose first listed episode (copy-majkel/LISTINGS-*.json) is within
    window_hours of the newest first-episode time (2026-09-29 05:08-05:09 UTC: 56663502 and 56663513, uploaded together)."""
    import datetime as dt
    import glob
    first = {}
    for f in glob.glob(str(MAJKEL_DIR / 'LISTINGS-*.json')):
        try:
            L = json.load(open(f))
        except (OSError, ValueError):
            continue
        for e in L.get('episodes', []):
            if e.get('policy_team') not in (None, MAJKEL):
                continue
            k = int(e['policy_sub'])
            first[k] = min(first.get(k, '9999'), str(e['create_time']))
    if not first:
        return set(), {}
    parse = lambda s: dt.datetime.fromisoformat(s[:19])
    newest = max(parse(v) for v in first.values())
    subs = {k for k, v in first.items() if (newest - parse(v)).total_seconds() <= window_hours * 3600}
    return subs, {str(k): first[k] for k in sorted(subs)}


def team_top_rated():
    """Owner 2026-09-29 19:07: a team's gold/upweight source is its top-rated active submission, not its newest. Per copy-* dir,
    the newest LISTINGS-2026*.json (no Kaggle calls): {submission_id: team_id} of every listed submission, {team_id: top-rated id}."""
    import glob
    listed, top = {}, {}
    for d in glob.glob(str(CAT.parent / 'extra' / 'copy-*') + '/'):
        fs = sorted(glob.glob(d + 'LISTINGS-2026*.json'))
        if not fs:
            continue
        try:
            L = json.load(open(fs[-1]))
        except (OSError, ValueError):
            continue
        subs = [x for x in L.get('submissions', []) if x.get('publicScore') not in (None, '')]
        if not subs or L.get('team_id') is None:
            continue
        tid = int(L['team_id'])
        for x in subs:
            listed[int(x['id'])] = tid
        top[tid] = int(max(subs, key=lambda x: float(x['publicScore']))['id'])
    return listed, top


MATURE_SEATS = 30   # PRO-TEAM-PROFILE-VADIM-0930: no recency/newest bonus for a listed submission with < 30 complete seats


def val_episode_ids():
    man = json.loads((PRACTICE / 'MANIFEST.json').read_text())
    tsv = PRACTICE / 'val' / 'VAL.tsv'
    if hashlib.sha256(tsv.read_bytes()).hexdigest() != man['val_tsv_sha256']:
        raise SystemExit('practice VAL.tsv does not match MANIFEST val_tsv_sha256')
    return {int(l.split('\t')[3]) for l in tsv.read_text().splitlines()[1:] if l.strip()}


def recency_weight(day):
    return 3 if day >= '2026-09-28' else 2 if day >= '2026-09-25' else 1


def component(reason):
    p = reason.split(':')
    return p[1] if len(p) > 1 and p[0] in ('c7', 'c7x') else 'other'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roster', default=str(CACHE / 'rows-c7.tsv'))
    ap.add_argument('--beattop-from', default='2026-09-25')
    ap.add_argument('--beatus-from', default='2026-09-26')
    ap.add_argument('--top12-from', default='2026-09-28')
    ap.add_argument('--top-n', type=int, default=12)
    ap.add_argument('--beat-top-n', type=int, default=30)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    roster = Path(a.roster)
    sys.path.insert(0, str(TRAINER))
    sys.path.insert(0, str(TRAINER.parents[3] / 'src'))
    import data as trainer_data
    import pyarrow as pa
    pa.set_cpu_count(3)
    pa.set_io_thread_count(3)
    import pyarrow.parquet as pq

    cols = ['episode_id', 'seat', 'day', 'kaggle', 'config_ok', 'tsv_ready', 'self_play', 'result', 'is_us', 'opp_is_us',
            'team', 'team_id', 'opp_team_id', 'lb_rank', 'opp_lb_rank', 'margin', 'archive', 'member',
            'submission_id', 'opp_submission_id', 'opp_team', 'sources']
    s = pq.read_table(CAT / 'SEATS.parquet', columns=cols).to_pandas()
    seats_mtime = os.path.getmtime(CAT / 'SEATS.parquet')
    s['is_w'] = (s['result'] == 'W').astype('int64')
    kg = s[s['kaggle'].eq(True)]
    sub_rec = kg[kg['submission_id'].notna()].groupby('submission_id')['is_w'].agg(['sum', 'size'])
    team_rec = kg[kg['team_id'].notna()].groupby('team_id')['is_w'].agg(['sum', 'size'])
    is_lg = s['sources'].map(lambda xs: 'league' in set(xs) if xs is not None else False)
    ver_rec = s[is_lg].groupby('team')['is_w'].agg(['sum', 'size'])
    sub_rec, team_rec, ver_rec = ({k: (int(a_), int(b_)) for k, a_, b_ in zip(d.index, d['sum'], d['size'])}
                                  for d in (sub_rec, team_rec, ver_rec))
    winners = s[s['result'] == 'W']
    opp_of = {(int(e), int(t)): (o_sub, o_tid, o_team, lgf) for e, t, o_sub, o_tid, o_team, lgf in
              zip(winners['episode_id'], winners['seat'], winners['opp_submission_id'], winners['opp_team_id'],
                  winners['opp_team'], is_lg[winners.index])}

    def opp_share(k):
        rec = opp_of.get(k)
        if rec is None:
            return None, 'n'
        o_sub, o_tid, o_team, lgf = rec
        if lgf:
            wr = ver_rec.get(o_team)
            return (wr[0] / wr[1], 'l') if wr else (None, 'n')
        if o_sub == o_sub and o_sub is not None and sub_rec.get(o_sub, (0, 0))[1] >= MIN_GAMES:
            wr = sub_rec[o_sub]
            return wr[0] / wr[1], ''
        if o_tid == o_tid and o_tid is not None and o_tid in team_rec:
            wr = team_rec[o_tid]
            return wr[0] / wr[1], 't'
        return None, 'n'
    w = s[s['kaggle'].eq(True) & s['config_ok'].eq(True) & ~s['self_play'].eq(True) & (s['result'] == 'W')
          & ~s['is_us'].eq(True) & s['day'].notna()]
    not_ready = int((~w['tsv_ready'].eq(True)).sum())
    w = w[w['tsv_ready'].eq(True)]
    day = w['day'].astype(str)
    masks = {
        'beattop': (w['opp_lb_rank'].fillna(1e9) <= a.beat_top_n) & (day >= a.beattop_from),
        'mmpq': w['team_id'].eq(MMPQ) | w['opp_team_id'].eq(MMPQ),
        'beatus': w['opp_is_us'].eq(True) & (day >= a.beatus_from),
        'top12': (w['lb_rank'].fillna(1e9) <= a.top_n) & (day >= a.top12_from),
        'majkel': w['team_id'].eq(MAJKEL),
    }
    majkel_new, majkel_new_first = majkel_newest_subs()
    listed_subs, top_rated = team_top_rated()
    gold_subs = set(top_rated.values())
    majkel_new = {top_rated[MAJKEL]} if MAJKEL in top_rated else majkel_new   # top-rated replaces newest (owner 19:07)
    # MMPQ-VAL (owner 2026-09-29 08:50 EDT: "our validation probably based on them", "their most recent policy"): every game of
    # MMPQ's newest submission (max submission id) with an even episode id, plus MMPQ games in the trainer's heldout split, is
    # validation; new rows from it are not added (rows already in the roster stay). List for the scorer: MMPQ_VAL_DIR/VAL-GAMES.tsv.
    mm = kg[kg['team_id'].eq(MMPQ) & kg['submission_id'].notna()]
    mmpq_newest = int(mm['submission_id'].max()) if len(mm) else None
    mmpq_val = {int(e) for e, sb in zip(mm['episode_id'], mm['submission_id'])
                if int(sb) == mmpq_newest and int(e) % 2 == 0}
    # CHECK-FRESHEN: a changing newest teacher must not release published VAL IDs.
    # Keep current scoring membership distinguishable from historical protection.
    mmpq_current_even = set(mmpq_val)
    import csv
    previous_val = MMPQ_VAL_DIR / 'VAL-GAMES.tsv'
    with previous_val.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        required = {'eid', 'seat', 'submission_id', 'day', 'result', 'reason', 'archive', 'member'}
        if not required <= set(reader.fieldnames or ()):
            raise SystemExit(f'{previous_val}: invalid frozen validation header')
        mmpq_prior_vrows = list(reader)
    if not mmpq_prior_vrows:
        raise SystemExit(f'{previous_val}: empty historical validation list; explicit bootstrap required')
    mmpq_val |= {int(r['eid']) for r in mmpq_prior_vrows}
    top12_loser = (w['opp_lb_rank'].fillna(1e9) <= a.top_n) & (day >= a.beattop_from)

    # roster as text rows (keep every existing field and order)
    lines = roster.read_text(encoding='utf-8').splitlines()
    if not lines or lines[0].split('\t')[:7] != list(FIELDS):
        raise SystemExit(f'{roster}: unexpected header')
    rows = [l.split('\t') for l in lines[1:] if l.strip()]
    index = {(int(r[3]), int(r[4])): i for i, r in enumerate(rows)}
    before_rows, before_mass = len(rows), sum(int(r[5]) for r in rows)
    weight_at_load = {(int(r[3]), int(r[4])): int(r[5]) for r in rows}
    base = {}
    for r in rows:
        m = TAIL.search(r[6])
        base[(int(r[3]), int(r[4]))] = int(m.group(1)) if m else int(r[5])
    before_comp = Counter()
    for r in rows:
        before_comp[component(r[6])] += int(r[5])

    sealed = set(trainer_data.sealed_ids())
    val_ids = val_episode_ids()
    blocked, n_hold_seeds = holdout_blocked_eids()
    # holdout protection (decode-h2h confirmation worlds): drop any roster row on a holdout seed or from a decode source
    # LOCKED-SPECS #8 (09-29 22:3x): the COMPLETE episode-level exclusion union applies to every final row, including rows admitted
    # before an exclusion existed (MMPQ-VAL, sealed, practice VAL, heldout split, blocked subs, TOP8 leak holdouts, holdout worlds).
    leak = {int(x) for x in Path('/mnt/ai/scratch/farmNEAT/touchup-top8-0929/LEAK-HOLDOUT-IDS.txt').read_text().split()} | {115279125}
    # TOP4-BEATERS (owner 09-30 01:5x): VAL episodes are never trained (exclusion); TRAIN seats get the top per-seat weight x4
    SP = Path(os.environ.get('TOP4_SPECIAL', '/mnt/ai/scratch/farmNEAT/corpus-final/special'))
    top4_val = {int(x) for x in (SP / 'TOP4-BEATERS-VAL.eids').read_text().split()} if (SP / 'TOP4-BEATERS-VAL.eids').exists() else set()
    leak |= top4_val
    top4_train = set()
    if (SP / 'TOP4-BEATERS-LATEST' / 'TRAIN.tsv').exists():
        _tl = (SP / 'TOP4-BEATERS-LATEST' / 'TRAIN.tsv').read_text().splitlines(); _h = _tl[0].split('\t')
        _ie, _is = (_h.index('eid'), _h.index('seat')) if 'eid' in _h else (_h.index('episode_id'), _h.index('seat'))
        top4_train = {(int(l.split('\t')[_ie]), int(l.split('\t')[_is])) for l in _tl[1:] if l.strip()}
    # TOP4 pending (data-ops 09-30 03:45): a NEW winner over a (dynamic) top-4 team is held back until the TOP4-BEATERS build has
    # assessed it (in ALL.tsv), so never-trained games can go to VAL first; the next hourly run adds its TRAIN seats.
    top4_all, top4_teams = set(), set()
    if (SP / 'TOP4-BEATERS-LATEST' / 'ALL.tsv').exists():
        top4_all = {int(l.split('\t')[3]) for l in (SP / 'TOP4-BEATERS-LATEST' / 'ALL.tsv').read_text().splitlines()[1:] if l.strip()}
        try:
            _m = json.loads((SP / 'TOP4-BEATERS-LATEST' / 'MANIFEST.json').read_text())
            top4_teams = {int(k) for k in _m.get('frozen_top4', {})}
            _names = {n for v in _m.get('top4_by_day', {}).values() for n in v}
        except (OSError, ValueError):
            _names = set()
        top4_teams |= {int(t) for t, n in zip(kg['team_id'], kg['team']) if n in _names and t == t}
    top4_mark = max(top4_all) if top4_all else 0   # episodes newer than the TOP4 build's newest are unassessed; older ones were judged
    sub_of = {(int(e), int(t)): int(u) for e, t, u in zip(kg['episode_id'], kg['seat'], kg['submission_id']) if u == u and u is not None}
    def excluded(r):
        e, t = int(r[3]), int(r[4])
        if e in blocked or blocked_source(r[0]): return 'holdout'
        if e in sealed: return 'sealed'
        if e in val_ids: return 'practice_val'
        if e in mmpq_val: return 'mmpq_val'
        if e in leak: return 'leak_holdout'
        if not trainer_data.in_split(e, 'train'): return 'not_train_split'
        if sub_of.get((e, t)) in BLOCKED_SUBS: return 'blocked_sub'
        return None
    removed_why = Counter(x for x in (excluded(r) for r in rows) if x)
    keep_rows = [r for r in rows if excluded(r) is None]
    holdout_removed = len(rows) - len(keep_rows)
    if holdout_removed:
        rows = keep_rows
        index = {(int(r[3]), int(r[4])): i for i, r in enumerate(rows)}
    stats = {c: Counter() for c in CLASSES}
    immature_keys = set()
    cand = {}          # (eid, seat) -> [weight, classes, record]
    for c in CLASSES:
        sub = w[masks[c]]
        for r in sub.itertuples(index=False):
            eid, seat = int(r.episode_id), int(r.seat)
            stats[c]['candidates'] += 1
            if not trainer_data.in_split(eid, 'train'):
                stats[c]['drop_not_train'] += 1
                continue
            if eid in sealed:
                stats[c]['drop_sealed'] += 1
                continue
            if eid in val_ids:
                stats[c]['drop_practice_val'] += 1
                continue
            if eid in mmpq_val:
                stats[c]['drop_mmpq_val'] += 1
                continue
            if r.submission_id == r.submission_id and int(r.submission_id) in BLOCKED_SUBS:
                stats[c]['drop_blocked_sub'] += 1
                continue
            if eid in blocked or blocked_source(r.archive, r.sources):
                stats[c]['drop_holdout'] += 1
                continue
            if top4_teams and r.opp_team_id == r.opp_team_id and int(r.opp_team_id) in top4_teams and eid not in top4_all and eid > top4_mark:
                stats[c]['hold_top4_pending'] += 1
                continue
            if eid in leak:   # LOCKED-SPECS-R3 (ASTRA #2): the same exclusion union as for existing rows
                stats[c]['drop_leak_holdout'] += 1
                continue
            own = int(r.submission_id) if r.submission_id == r.submission_id and r.submission_id is not None else None
            # LOCKED-SPECS #14 precedence (coordinator 09-29): beattop (opponent-strength exception) 3; else maturity guard FIRST
            # (an immature listed submission gets 1 and is not raised by top-rated or the opponent tier); else top-rated 3.
            if c == 'beattop':
                wt = 3
            elif own in listed_subs and sub_rec.get(own, (0, 0))[1] < MATURE_SEATS:
                wt = 1
                immature_keys.add((eid, seat))
                stats[c]['immature_no_bonus'] += 1
            elif own in gold_subs:
                wt = 3
            elif own in listed_subs:
                wt = min(2, recency_weight(str(r.day)))
            else:
                wt = recency_weight(str(r.day))
            k = (eid, seat)
            if k in cand:
                cand[k][0] = 1 if k in immature_keys else max(cand[k][0], wt)
                cand[k][1].append(c)
            else:
                cand[k] = [wt, [c], r]
    top12_keys = {(int(e), int(t)) for e, t in zip(w.loc[top12_loser, 'episode_id'], w.loc[top12_loser, 'seat'])}

    added, raised, unchanged = [], [], 0
    added_by, raised_by = Counter(), Counter()
    for k, (wt, classes, r) in cand.items():
        tag = '+'.join(classes)
        first = classes[0]
        if k in index:
            row = rows[index[k]]
            old = base[k]
            # LOCKED-SPECS #14: rows admitted or raised by this extension are recomputed from current facts (demotion allowed);
            # recipe rows keep max(recipe base, extension weight)
            ours = row[6].startswith('c7x:') or ':x=' in row[6]
            if (ours and wt != old) or wt > old:
                base[k] = wt
                if ':x=' not in row[6]:
                    tail = TAIL.search(row[6])
                    row[6] = TAIL.sub('', row[6]) + f':x={tag}' + (tail.group(0) if tail else '')
                raised.append((k, old, wt, first))
                raised_by[first] += 1
            else:
                unchanged += 1
            for c in classes:
                stats[c]['already_in_roster'] += 1
        else:
            reason = f'c7x:{first}:w{wt}:{tag}:m{int(r.margin) if r.margin == r.margin else 0}:{str(r.team).replace(chr(9), " ")}'
            rows.append([str(r.archive), str(r.member), str(r.day), str(k[0]), str(k[1]), str(wt), reason])
            index[k] = len(rows) - 1
            base[k] = wt
            added.append(k)
            added_by[first] += 1
            for c in classes:
                stats[c]['added'] += 1

    # LOCKED-SPECS-R3 (ASTRA #11) precedence for EVERY final row, recipe rows included: beattop winners keep the opponent-strength
    # path; otherwise a row whose own submission is listed and immature (< MATURE_SEATS catalog seats) is capped at 1 (demotion
    # allowed, never raised by top-rated, recency or tier); otherwise max(base, tier).
    beattop_all = {(int(e), int(t)) for e, t in zip(w.loc[masks['beattop'], 'episode_id'], w.loc[masks['beattop'], 'seat'])}
    for r in rows:
        k = (int(r[3]), int(r[4])); own = sub_of.get(k)
        if k in beattop_all:
            # CHECK-FRESHEN: another class may have overwritten cand[k] with 1.
            # Removing the immature flag alone does not restore the beattop floor.
            immature_keys.discard(k)
            base[k] = max(base[k], 3)
        elif own in listed_subs and sub_rec.get(own, (0, 0))[1] < MATURE_SEATS:
            immature_keys.add(k); base[k] = 1
    # TOP4-BEATERS TRAIN seats missing from the roster are added (cached winner seats only; exclusions apply)
    if top4_train and (SP / 'TOP4-BEATERS-LATEST' / 'TRAIN.tsv').exists():
        _tl = (SP / 'TOP4-BEATERS-LATEST' / 'TRAIN.tsv').read_text().splitlines(); _h = _tl[0].split('\t')
        if {'archive', 'member', 'day'} <= set(_h):
            for l in _tl[1:]:
                f = dict(zip(_h, l.split('\t'))); e = int(f.get('eid') or f.get('episode_id')); t_ = int(f['seat']); k = (e, t_)
                if k in index or not (CACHE / f'{e}-{t_}.stp').exists():
                    continue
                r_ = [f['archive'], f['member'], f['day'], str(e), str(t_), '4', 'c7x:top4:w4']
                if excluded(r_) is not None:
                    continue
                rows.append(r_); index[k] = len(rows) - 1; base[k] = 4; added.append(k); added_by['top4'] += 1
    # opponent-strength tier: final weight = max(base, tier), recomputed from the base every run
    tier_mass, src_rows, changed, raised_by_tier = Counter(), Counter(), 0, 0
    for r in rows:
        k = (int(r[3]), int(r[4]))
        share, flag = opp_share(k)
        tier = 1 if share is None else 3 if share >= 0.70 else 2 if share >= 0.55 else 1
        new = base[k] if k in immature_keys else max(base[k], tier)   # the maturity cap is applied last (LOCKED-SPECS #14)
        if k in top4_train:   # TOP4-BEATERS TRAIN: the highest per-seat weight, above beattop x3 and the maturity cap
            new = 4
        if tier > base[k]:
            raised_by_tier += 1
        if new != weight_at_load.get(k, -1):
            changed += 1
        r[5] = str(new)
        r[6] = TAIL.sub('', r[6]) + f':b={base[k]}:ot=' + ('na' if share is None else f'{share:.2f}') + flag
        tier_mass['>=0.70' if share is not None and share >= 0.70 else '0.55-0.70' if share is not None and share >= 0.55
                  else '<0.55' if share is not None else 'no record'] += new
        src_rows[{'': 'submission', 't': 'team_fallback', 'l': 'league_version', 'n': 'none'}[flag]] += 1
    after_rows, after_mass = len(rows), sum(int(r[5]) for r in rows)
    after_comp = Counter()
    for r in rows:
        after_comp[component(r[6])] += int(r[5])
    keys = [(int(r[3]), int(r[4])) for r in rows]
    assert len(keys) == len(set(keys)), 'duplicate (eid, seat) after merge'
    bad = [(r[3], r[4], excluded(r)) for r in rows if excluded(r) is not None]
    assert not bad, f'excluded rows in the final roster: {bad[:5]}'   # LOCKED-SPECS-R3: asserted on what will be written
    assert all(int(r[5]) == 1 for r in rows if (int(r[3]), int(r[4])) in immature_keys and (int(r[3]), int(r[4])) not in top4_train), 'maturity cap not applied'
    assert all(int(r[5]) == 4 for r in rows if (int(r[3]), int(r[4])) in top4_train), 'TOP4 TRAIN seat without x4'
    assert not any(int(r[3]) in top4_val for r in rows), 'TOP4 VAL episode in the roster'
    built = lambda k: (CACHE / f'{k[0]}-{k[1]}.stp').exists()
    touched = list(added)
    unbuilt = [k for k in touched if not built(k)]
    order = {'beattop': 0, 'mmpq': 1, 'majkel': 2, 'beatus': 3, 'top12': 4, 'top4': 5}
    unbuilt.sort(key=lambda k: (order[cand[k][1][0]], -int(rows[index[k]][2].replace('-', '') or 0), k))
    all_unbuilt = [k for k in keys if not built(k)]
    # weighted draw share of the priority class: every roster seat that is a winner over a current top-30 (or top-12)
    # team from --beattop-from on, whatever component admitted it
    beattop_keys = {(int(e), int(t)) for e, t in zip(w.loc[masks['beattop'], 'episode_id'], w.loc[masks['beattop'], 'seat'])}
    in_beattop = [r for r in rows if (int(r[3]), int(r[4])) in beattop_keys]
    in_top12 = [r for r in rows if (int(r[3]), int(r[4])) in top12_keys]
    mmpq_keys = {(int(e), int(t)) for e, t in zip(w.loc[masks['mmpq'], 'episode_id'], w.loc[masks['mmpq'], 'seat'])}
    in_mmpq = [r for r in rows if (int(r[3]), int(r[4])) in mmpq_keys]
    mmpq_team = {(int(e), int(t)) for e, t in zip(w.loc[w['team_id'].eq(MMPQ), 'episode_id'], w.loc[w['team_id'].eq(MMPQ), 'seat'])}
    mk = w['team_id'].eq(MAJKEL)
    majkel_keys = {(int(e), int(t)) for e, t in zip(w.loc[mk, 'episode_id'], w.loc[mk, 'seat'])}
    mkn = mk & w['submission_id'].isin(list(majkel_new))
    majkel_new_keys = {(int(e), int(t)) for e, t in zip(w.loc[mkn, 'episode_id'], w.loc[mkn, 'seat'])}
    by_day = Counter()
    for r in rows:
        by_day[r[2]] += int(r[5])
    report = {
        'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'seats_parquet_mtime_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(seats_mtime)),
        'roster': str(roster),
        'before': {'rows': before_rows, 'mass': before_mass,
                   'component_share': {k: round(v / before_mass, 4) for k, v in sorted(before_comp.items())}},
        'after': {'rows': after_rows, 'mass': after_mass,
                  'component_share': {k: round(v / after_mass, 4) for k, v in sorted(after_comp.items())}},
        'priority_class_beat_top30_from_' + a.beattop_from: {
            'roster_rows': len(in_beattop), 'roster_mass': sum(int(r[5]) for r in in_beattop),
            'share_of_weighted_draws': round(sum(int(r[5]) for r in in_beattop) / after_mass, 4),
            'top12_loser_subset_rows': len(in_top12), 'top12_loser_subset_mass': sum(int(r[5]) for r in in_top12),
            'top12_loser_subset_share': round(sum(int(r[5]) for r in in_top12) / after_mass, 4)},
        'mmpq_games_in_roster': {'rows': len(in_mmpq), 'mass': sum(int(r[5]) for r in in_mmpq),
                                 'mmpq_wins': sum(1 for r in in_mmpq if (int(r[3]), int(r[4])) in mmpq_team),
                                 'mmpq_beater_wins': sum(1 for r in in_mmpq if (int(r[3]), int(r[4])) not in mmpq_team),
                                 'share_of_weighted_draws': round(sum(int(r[5]) for r in in_mmpq) / after_mass, 4)},
        'classes': {c: dict(stats[c]) for c in CLASSES},
        'majkel': {'newest_subs_first_episode_utc': majkel_new_first,
                   'roster_rows': sum(1 for r in rows if (int(r[3]), int(r[4])) in majkel_keys),
                   'roster_mass': sum(int(r[5]) for r in rows if (int(r[3]), int(r[4])) in majkel_keys),
                   'share_of_weighted_draws': round(sum(int(r[5]) for r in rows if (int(r[3]), int(r[4])) in majkel_keys) / after_mass, 4),
                   'newest_sub_rows': sum(1 for r in rows if (int(r[3]), int(r[4])) in majkel_new_keys),
                   'newest_sub_mass': sum(int(r[5]) for r in rows if (int(r[3]), int(r[4])) in majkel_new_keys)},
        'this_run': {'added_rows': len(added), 'added_by_first_class': dict(added_by),
                     'added_mass': sum(int(rows[index[k]][5]) for k in added),
                     'raised_rows': len(raised), 'raised_by_first_class': dict(raised_by),
                     'raised_mass_delta': sum(n - o for _, o, n, _ in raised), 'unchanged_present': unchanged,
                     'unbuilt_touched_rows': len(unbuilt)},
        'opponent_strength': {'min_games_submission': MIN_GAMES,
                              'rows_weight_changed_vs_file_at_load': changed,
                              'rows_where_tier_exceeds_base': raised_by_tier,
                              'mass_by_beaten_opponent_win_share': {k: {'mass': v, 'share': round(v / after_mass, 4)}
                                                                    for k, v in sorted(tier_mass.items())},
                              'rows_by_share_source': dict(src_rows),
                              'mass_base_only': sum(base.values())},
        'holdout': {'seeds': n_hold_seeds, 'catalog_episodes_on_holdout_seeds': len(blocked),
                    'roster_rows_removed_this_run': holdout_removed, 'removed_by_reason': dict(removed_why),
                    'blocked_sources': list(BLOCKED_SOURCES)},
        'maturity_capped_rows': len(immature_keys),
        'top4': {'train_seats_listed': len(top4_train), 'train_seats_in_roster_x4': sum(1 for r in rows if (int(r[3]), int(r[4])) in top4_train),
                 'val_episodes_excluded': len(top4_val)},
        'roster_unbuilt_rows': len(all_unbuilt),
        'mass_by_day': {k: {'mass': v, 'share': round(v / after_mass, 4)} for k, v in sorted(by_day.items())},
        'not_tsv_ready_winner_seats_skipped': not_ready,
        'passes_per_chunk_12p5M': round(12.5e6 / 719 / after_mass, 3),
        'seconds': round(time.time() - t0, 1),
    }
    # MMPQ-VAL list: MMPQ's seat in every MMPQ-VAL game (newest-sub even eids) and every heldout-split game of its 3 newest subs
    subs3 = sorted({int(x) for x in mm['submission_id']})[-3:] if len(mm) else []
    in_roster = {(int(r[3]), int(r[4])) for r in rows}
    vrows = []
    for r in s[s['team_id'].eq(MMPQ) & s['kaggle'].eq(True) & s['submission_id'].isin(subs3)].itertuples(index=False):
        eid, seat = int(r.episode_id), int(r.seat)
        why = ('newest-even' if eid in mmpq_current_even else
               'heldout-split' if not trainer_data.in_split(eid, 'train') else
               'historical-val' if eid in mmpq_val else None)
        if why:
            vrows.append((eid, seat, int(r.submission_id), str(r.day), str(r.result), why, int((eid, seat) in in_roster),
                          str(r.archive), str(r.member)))
    # Persist old records even when their submission is no longer in subs3.
    val_by_key = {(v[0], v[1]): v for v in vrows}
    for old in mmpq_prior_vrows:
        eid, seat = int(old['eid']), int(old['seat'])
        val_by_key.setdefault((eid, seat),
            (eid, seat, int(old['submission_id']), old['day'], old['result'],
             'historical-val', int((eid, seat) in in_roster), old['archive'], old['member']))
    vrows = list(val_by_key.values())
    report['mmpq_val'] = {'newest_submission': mmpq_newest, 'games': len(vrows),
                          'by_reason': dict(Counter(v[5] for v in vrows)), 'mmpq_seat_in_roster': sum(v[6] for v in vrows)}
    text = json.dumps(report, indent=1, ensure_ascii=False) + '\n'
    if a.dry_run:   # LOCKED-SPECS-R3 #18: a dry run performs no writes (not even the shared MMPQ-VAL list)
        print(text)
        return
    MMPQ_VAL_DIR.mkdir(exist_ok=True)
    vt = MMPQ_VAL_DIR / f'VAL-GAMES.tsv.tmp{os.getpid()}'
    vt.write_text('eid\tseat\tsubmission_id\tday\tresult\treason\tin_roster\tarchive\tmember\n'
                  + ''.join('\t'.join(map(str, v)) + '\n' for v in sorted(vrows)), encoding='utf-8')
    os.replace(vt, MMPQ_VAL_DIR / 'VAL-GAMES.tsv')
    stem = roster.with_name(roster.stem)
    body = '\t'.join(FIELDS) + '\n' + ''.join('\t'.join(r) + '\n' for r in rows)
    ub = '\t'.join(FIELDS) + '\n' + ''.join('\t'.join(rows[index[k]]) + '\n' for k in unbuilt)
    sha = hashlib.sha256(body.encode('utf-8')).hexdigest()
    vdir = roster.parent / f'{roster.stem}-versions'
    vdir.mkdir(exist_ok=True)
    snap = vdir / f'{roster.stem}.{sha[:12]}.tsv'
    if not snap.exists():
        tmp = snap.with_name(snap.name + f'.tmp{os.getpid()}')
        tmp.write_text(body, encoding='utf-8')
        os.replace(tmp, snap)
    report['roster_sha256'] = sha
    report['snapshot'] = str(snap)
    text = json.dumps(report, indent=1, ensure_ascii=False) + '\n'
    for path, content in ((roster, body), (Path(f'{stem}.extend-unbuilt.tsv'), ub), (Path(f'{stem}.extend.json'), text)):
        tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
        tmp.write_text(content, encoding='utf-8')
        os.replace(tmp, path)
    with open(f'{stem}.extend-log.jsonl', 'a') as f:
        f.write(json.dumps({k: report[k] for k in ('built_utc', 'before', 'after', 'this_run', 'opponent_strength',
                                                  'roster_unbuilt_rows', 'roster_sha256')}) + '\n')
    print(json.dumps({k: report[k] for k in ('before', 'after', 'this_run', 'opponent_strength', 'roster_unbuilt_rows',
                                             'passes_per_chunk_12p5M', 'snapshot')}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
