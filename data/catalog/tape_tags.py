#!/usr/bin/python3
"""Recent replay action-bundle fingerprints; no effect on running trainers."""

from __future__ import annotations

import collections
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

CAT = Path(__file__).resolve().parent
CACHE = CAT / 'state/TAPE-ACTIONS-v1.jsonl'
VIEW = CAT / 'views/tape-seats.tsv'
SUBS = CAT / 'TAPE-SUBMISSIONS.tsv'
DEFAULT_REFERENCE = CAT / 'state/marketshock-reference-115161684.json'
REFERENCE_SHA256 = '69589a5ed533441d4fac8305054d6f353823d130e182a640401e260f76087323'
HORIZON = 151

MOVES = {'NORTH', 'SOUTH', 'EAST', 'WEST'}


def action_op(command):
    op = command[0] if isinstance(command, (list, tuple)) and command else 'PASS'
    good = command[1] if isinstance(command, (list, tuple)) and len(command) > 1 \
        and isinstance(command[1], str) else ''
    return ('MOVE' if op in MOVES else str(op), good)


def action_bundle(action):
    """Quantity/item-insensitive bundle from the Ice & Fire static-tape probe."""
    hands = action.get('hands') or []
    market = action.get('market') or []
    return (action_op(action.get('farmer')),
            len(hands),
            tuple(sorted(action_op(command) for command in hands)),
            tuple(sorted(set(action_op(command) for command in market))))


def divergence_p25(series):
    """First index where 1 − modal bundle share is strictly above 0.25."""
    if not series:
        return None
    horizon = min(map(len, series))
    for step in range(horizon):
        counts = collections.Counter(row[step] for row in series)
        if 1 - max(counts.values()) / len(series) > 0.25:
            return step
    return None


def marketshock_open_fraction(actions, reference):
    """Exact-action match share over replay action frames 4..150 inclusive."""
    if len(actions) < 151 or len(reference) < 151:
        raise ValueError('need action frames through index 150')
    return sum(actions[i] == reference[i] for i in range(4, 151)) / 147


def bundle_digest(action):
    raw = json.dumps(action_bundle(action), separators=(',', ':')).encode()
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def replay_seat_actions(replay, seat):
    """Action frames 1..151 correspond to physical actions 0..150."""
    steps = replay['steps']
    if len(steps) < HORIZON + 1:
        raise ValueError(f'need at least {HORIZON + 1} replay frames, got {len(steps)}')
    return [(steps[t][seat].get('action') or {}) for t in range(1, HORIZON + 1)]


def load_cache():
    rows = {}
    if CACHE.exists():
        with CACHE.open() as f:
            for n, line in enumerate(f, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'{CACHE}:{n}: incomplete receipt') from e
                eid = int(row['episode_id'])
                for seat, rec in enumerate(row['seats']):
                    key = (eid, seat)
                    if key in rows:
                        raise ValueError(f'duplicate tape receipt {key}')
                    rows[key] = rec
    return rows


def selected_seats(day_from, sources=None, day_to=None):
    import pyarrow.parquet as pq
    cols = ['episode_id', 'seat', 'day', 'team', 'team_id', 'submission_id',
            'source', 'format', 'archive', 'member', 'win', 'result', 'is_us', 'opp_is_us']
    seats = pq.read_table(CAT / 'SEATS.parquet', columns=cols).to_pandas()
    seats = seats[seats.day.ge(day_from) & seats.format.eq('zip-json') &
                  ~seats.source.isin(['league', 'selfplay-league', 'skirmishes', 'skirmishes-loose'])]
    if sources:
        seats = seats[seats.source.isin(sources)]
    if day_to:
        seats = seats[seats.day.le(day_to)]
    return seats


def score_submission(rows):
    """Keep seat-specific divergence separate; pooled p25 is diagnostic only."""
    by_seat = {seat: [r for r in rows if r['seat'] == seat] for seat in (0, 1)}
    pooled = divergence_p25([r['bundle'] for r in rows]) if len(rows) >= 4 else None
    side = {seat: divergence_p25([r['bundle'] for r in rr]) if len(rr) >= 4 else None
            for seat, rr in by_seat.items()}
    eligible = [side[seat] if side[seat] is not None else HORIZON
                for seat, rr in by_seat.items() if len(rr) >= 4]
    tape_score = min(eligible) if eligible else None
    return {'n_seats': len(rows), 'n0': len(by_seat[0]), 'n1': len(by_seat[1]),
            'p25_pooled': pooled, 'p25_seat0': side[0], 'p25_seat1': side[1],
            'tape_score': tape_score, 'tape_score_censored': tape_score == HORIZON,
            'side_coverage': sum(len(rr) >= 4 for rr in by_seat.values()),
            'mean_marketshock_open': sum(r['marketshock_open'] for r in rows) / len(rows)}


def scan_chunk(task):
    """Read a small independent ZIP-member batch, keeping cache writes in parent."""
    import orjson
    archive, items, ref_actions = task
    receipts, errors = [], []
    try:
        z = zipfile.ZipFile(archive)
    except Exception as e:
        return [], [json.dumps({'episode_id': eid, 'archive': archive, 'error': repr(e)})
                    for eid, _ in items]
    with z:
        for eid, member in items:
            try:
                replay = orjson.loads(z.read(member))
                pair = []
                for seat in (0, 1):
                    actions = replay_seat_actions(replay, seat)
                    pair.append({'bundle': [bundle_digest(a) for a in actions],
                                 'marketshock_open': marketshock_open_fraction(actions, ref_actions)})
                receipts.append(json.dumps({'episode_id': int(eid), 'seats': pair}, separators=(',', ':')))
            except Exception as e:
                errors.append(json.dumps({'episode_id': int(eid), 'archive': archive,
                                          'member': member, 'error': repr(e)}))
    return receipts, errors


def scan_batches(tasks, workers):
    if workers == 1:
        yield from map(scan_chunk, tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(scan_chunk, tasks)


def collect(seats, reference, limit=None, workers=1):
    import orjson
    ref_bytes = reference.read_bytes()
    if hashlib.sha256(ref_bytes).hexdigest() != REFERENCE_SHA256:
        raise ValueError(f'MarketShock reference hash changed: {reference}')
    ref = orjson.loads(ref_bytes)
    ref_actions = replay_seat_actions(ref, 1)
    existing = load_cache()
    game_rows = seats.sort_values(['archive', 'episode_id']).drop_duplicates('episode_id')
    missing = game_rows[~game_rows.episode_id.isin({x[0] for x in existing})]
    if limit is not None:
        missing = missing.head(limit)
    tasks = []
    for archive, group in missing.groupby('archive', sort=False):
        items = [(int(row.episode_id), row.member) for row in group.itertuples(index=False)]
        tasks.extend((archive, items[i:i + 12], ref_actions) for i in range(0, len(items), 12))
    t0 = time.monotonic()
    ok = errors = 0
    error_path = CAT / 'state/TAPE-ACTIONS-v1-ERRORS.jsonl'
    with CACHE.open('a') as out, error_path.open('a') as err:
        for recs, errs in scan_batches(tasks, workers):
            for line in recs:
                out.write(line + '\n')
            for line in errs:
                err.write(line + '\n')
            ok += len(recs)
            errors += len(errs)
            out.flush()
            err.flush()
            if (ok + errors) // 200 != (ok + errors - len(recs) - len(errs)) // 200:
                print(f'tape scan {ok} new games, {errors} errors, {time.monotonic()-t0:.0f}s', flush=True)
    print(f'tape scan done: {ok} new games, {errors} errors', flush=True)


def render_views(seats):
    import math
    cache = load_cache()
    needed = {(int(r.episode_id), int(r.seat)) for r in seats.itertuples(index=False)}
    missing = needed - set(cache)
    if missing:
        raise RuntimeError(f'refusing partial tape view: {len(missing)} of {len(needed)} seats missing; '
                           f'first={sorted(missing)[:3]}')
    groups = collections.defaultdict(list)
    rows = []
    for row in seats.itertuples(index=False):
        rec = cache.get((int(row.episode_id), int(row.seat)))
        if rec is None:
            continue
        sub = None if row.submission_id is None or math.isnan(row.submission_id) else int(row.submission_id)
        result_vs_us = row.result if bool(row.opp_is_us) and not bool(row.is_us) else ''
        out = {'episode_id': int(row.episode_id), 'seat': int(row.seat), 'day': row.day,
               'team': row.team, 'submission_id': sub, 'source': row.source,
               'bundle': rec['bundle'], 'marketshock_open': rec['marketshock_open'],
               'result_vs_us': result_vs_us}
        rows.append(out)
        if sub is not None:
            groups[sub].append(out)
    scores = {}
    for sub, rr in groups.items():
        scores[sub] = {'submission_id': sub, 'team': rr[0]['team'], **score_submission(rr)}
    tmp = SUBS.with_suffix('.tsv.tmp')
    with tmp.open('w') as f:
        f.write('submission_id\tteam\tn_seats\tn0\tn1\tp25_pooled\tp25_seat0\tp25_seat1\t'
                'tape_score\ttape_score_censored\tside_coverage\tmean_marketshock_open\n')
        for sub, r in sorted(scores.items(), key=lambda kv: (-kv[1]['n_seats'], kv[0])):
            def num(k):
                return r[k] if r[k] is not None else ''
            f.write(f'{sub}\t{r["team"]}\t{r["n_seats"]}\t{r["n0"]}\t{r["n1"]}\t'
                    f'{num("p25_pooled")}\t{num("p25_seat0")}\t{num("p25_seat1")}\t'
                    f'{num("tape_score")}\t{int(r["tape_score_censored"])}\t{r["side_coverage"]}\t'
                    f'{r["mean_marketshock_open"]:.6f}\n')
    os.replace(tmp, SUBS)
    tmp = VIEW.with_suffix('.tsv.tmp')
    with tmp.open('w') as f:
        f.write('episode_id\tseat\tday\tteam\tsubmission_id\tsubmission_n\tbundle_p25\t'
                'bundle_at_p25\ttape_score\ttape_score_censored\tmarketshock_open\tresult_vs_us\n')
        for r in sorted(rows, key=lambda x: (x['episode_id'], x['seat'])):
            score = scores.get(r['submission_id'])
            p25 = score[f'p25_seat{r["seat"]}'] if score else None
            bundle = r['bundle'][p25] if p25 is not None else ''
            f.write(f'{r["episode_id"]}\t{r["seat"]}\t{r["day"]}\t{r["team"]}\t'
                    f'{r["submission_id"] if r["submission_id"] is not None else ""}\t'
                    f'{score["n_seats"] if score else ""}\t{p25 if p25 is not None else ""}\t'
                    f'{bundle}\t{score["tape_score"] if score and score["tape_score"] is not None else ""}\t'
                    f'{int(score["tape_score_censored"]) if score else ""}\t'
                    f'{r["marketshock_open"]:.6f}\t{r["result_vs_us"]}\n')
    os.replace(tmp, VIEW)
    print(f'wrote {len(rows)} tape-seat tags, {len(scores)} submission groups; '
          f'{sum(v["tape_score"] is not None for v in scores.values())} with seat-stratified score', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--day-from', default='2026-09-27')
    p.add_argument('--day-to', help='inclusive pilot end day')
    p.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    p.add_argument('--limit', type=int)
    p.add_argument('--source', action='append', help='restrict pilot to one catalog source (repeatable)')
    p.add_argument('--collect-only', action='store_true')
    p.add_argument('--view-only', action='store_true')
    p.add_argument('--workers', type=int, default=1, choices=(1, 2, 3))
    a = p.parse_args()
    try:
        os.nice(19)
    except OSError:
        pass
    seats = selected_seats(a.day_from, a.source, a.day_to)
    print(f'{len(seats)} candidate seats, {seats.episode_id.nunique()} games', flush=True)
    if not a.view_only:
        collect(seats, a.reference, a.limit, a.workers)
    if not a.collect_only:
        if a.limit is not None:
            print('pilot limit set: receipt retained, no public view written', flush=True)
        else:
            render_views(seats)


if __name__ == '__main__':
    main()
