#!/usr/bin/python3
"""Per-seat facts of every stored game, ready-made training selections, and the rebuild of seed+action tapes.
Runs under /usr/bin/python3 (pyarrow, pandas, orjson, zstandard); lowers itself to nice 19; at most 5 workers + main.

  seats.py build                      # EPISODES + facts + leaderboard -> SEATS.parquet, SEAT_DAYS.parquet, views/ (~1 min)
  seats.py facts [--max-minutes M]    # exact re-play of episodes without facts yet, newest day first (resumable)
  seats.py pick --wins --min-rating 2880 --quads 4 --day-from 2026-09-20 --out x.tsv   # trainer priority TSV
  seats.py pick --preset v5s --out v5s.tsv          # = work/runs/maintainer/pbc-selfplay/build_v5s_rows.py
  seats.py show 113600120             # every fact of one game, both seats
  seats.py rebuild-tapes [--max-minutes M]          # xsf/capsule seed+action tapes -> extra/reconstructed-xsf zips
  seats.py status                     # coverage of facts / rebuild, pending counts

SEATS.parquet has one row per (stored episode, seat): who played, result, cash, margin, ratings (game score, submission
score, leaderboard), final quadrants, day of the 2nd/3rd/4th quadrant, hires and hire spend, units and $ sold and bought,
dusk discards, escapes, plant deaths, rot, end-of-game farm, and the copy to train from. SEAT_DAYS.parquet has the same
counters per game day. views/*.tsv are selections rebuilt every hour (views/INDEX.md). README.md "How to grab X" has examples.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import catalog as C  # noqa: E402

CAT = C.CAT
STATE = CAT / 'state'
FACTS_DIR = STATE / 'facts'
VIEWS = CAT / 'views'
FACTS_VERSION = 1
N_DAYS = 30
DAY_STEPS = 24
WORKERS = 5                        # + the main process = 6
MIN_FREE_GB = 12.0                 # owner's desktop: keep >= 12 GB available
GB_PER_FACTS_WORKER = 0.6
GB_PER_REBUILD_WORKER = 1.2
REBUILT = C.EXTRA / 'reconstructed-xsf'
PASS = {'farmer': ['PASS'], 'hands': [], 'market': []}
REF_CFG = {k: v for k, v in C.REFERENCE.items() if k not in ('seed', 'actTimeout', 'runTimeout')}
LB_DIR = C.EXTRA / 'top30-losses'

# per-day flows (summed over the day's steps) and dusk state (after the day's last step)
FLOW = ('sold_units', 'sales', 'bought', 'seed_spend', 'animal_spend', 'product_spend', 'animals_bought', 'hires',
        'hire_spend', 'land_spend', 'discards', 'escapes', 'plant_deaths', 'rot')
DUSK = ('cash', 'quadrants', 'animals', 'plants', 'weeds', 'shed_units', 'shed_value')
FORMAT_PREF = {'xsf-tape': 0, 'capsule': 1, 'tape-compact': 2, 'tape-selfplay': 2, 'recorded-game': 2, 'zip-json': 3,
               'json-gz': 4, 'json': 5}
LOOSE_FORMATS = ('json', 'json-gz', 'capsule', 'tape-compact', 'tape-selfplay', 'recorded-game')

ACT_RE = re.compile(rb'\{"action":\s*')
INFO_RE = re.compile(rb'\s*,\s*"info"\s*:')
QUAD_RE = re.compile(rb'"unlocked_quadrants":\s*(\[[^\]]*\])')

log = C.log


def utc_stamp():
    return C.utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')


def mem_available_gb():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0


def workers_for(want, gb_each):
    """Worker count that keeps MemAvailable >= MIN_FREE_GB after launch (0 = do not start)."""
    free = mem_available_gb()
    n = int((free - MIN_FREE_GB) // gb_each)
    return max(0, min(want, n))


# ------------------------------------------------------------------ exact re-play with counters (worker side)
_ENG = None


def engine():
    global _ENG
    if _ENG is None:
        src = str(C.REPO / 'src')
        if src not in sys.path:
            sys.path.insert(0, src)
        from farm_neat.game import exact_engine as X
        _ENG = (X, X.load_engine())
    return _ENG


HOOKED = ('_commit_unit', '_do_hire', '_do_buy_land', '_drop_inventories_to_shed', '_daily_refresh_animals',
          '_daily_refresh_plants', '_decay_plants')


def _count(farm, test):
    return sum(1 for row in farm['tiles'] for t in row if isinstance(t, dict) and test(t))


def _is_animal(t):
    return 'animal' in t


def _is_plant(t):
    return t.get('kind') == 'PLANT'


def _is_weed(t):
    return t.get('kind') == 'WEED'


def _plant_units(farm):
    return sum(int(t.get('yield_units') or 0) for row in farm['tiles'] for t in row
               if isinstance(t, dict) and t.get('kind') == 'PLANT')


def replay(seed, cfg, actions):
    """Exact re-play of recorded joint actions (actions[t] = [seat0, seat1], t = 0..718) with engine-level counters.
    Returns {'final_cash', 'flow', 'dusk', 'land_steps', 'sold_items', 'sold_usd', 'animals_by', 'final_quadrants'}."""
    X, E = engine()
    if (cfg or {}).get('crnRng'):   # CRN league/self-play game: its dusk rule (farm_neat/rollout/crn_rng.py) and engine
        from farm_neat.rollout import crn_rng
        env = crn_rng.environment(seed=int(seed), configuration=cfg)
        E = env.engine              # the counters below hook the engine this env steps with
    else:
        env = X.ExactKaggricultureEnvironment(seed=int(seed), configuration=cfg)
    flow = {k: [[0] * N_DAYS, [0] * N_DAYS] for k in FLOW}
    dusk = {k: [[None] * N_DAYS, [None] * N_DAYS] for k in DUSK}
    sold_items = [collections.Counter(), collections.Counter()]
    sold_usd = [collections.Counter(), collections.Counter()]
    animals_by = [collections.Counter(), collections.Counter()]
    bought_items = [collections.Counter(), collections.Counter()]
    bought_usd = [collections.Counter(), collections.Counter()]
    land_steps = [[], []]
    cur = [0]
    orig = {n: getattr(E, n) for n in HOOKED}

    def seat_of(farm):
        return 0 if env.state[0].observation.farms[0] is farm else 1

    def day():
        return min(N_DAYS - 1, cur[0] // DAY_STEPS)

    def commit(op, item, price, farm, private, market, cap=100):
        ok = orig['_commit_unit'](op, item, price, farm, private, market, cap)
        if ok:
            p, d = seat_of(farm), day()
            if op == 'SELL':
                flow['sold_units'][p][d] += 1
                flow['sales'][p][d] += price
                sold_items[p][item] += 1
                sold_usd[p][item] += price
            else:
                flow['bought'][p][d] += price
                key = f'{op[4:]}:{item}'           # SEED:WHEAT, PRODUCT:WHEAT, ANIMAL:COW
                bought_items[p][key] += 1
                bought_usd[p][key] += price
                if op == 'BUY_SEED':
                    flow['seed_spend'][p][d] += price
                elif op == 'BUY_ANIMAL':
                    flow['animal_spend'][p][d] += price
                    flow['animals_bought'][p][d] += 1
                    animals_by[p][item] += 1
                else:
                    flow['product_spend'][p][d] += price
        return ok

    def hire(farm, private, board, mult=1):
        m, h = farm['money'], farm['hires_today']
        orig['_do_hire'](farm, private, board, mult)
        if farm['hires_today'] != h:
            p, d = seat_of(farm), day()
            flow['hires'][p][d] += 1
            flow['hire_spend'][p][d] += m - farm['money']

    def land(farm, board):
        m, q = farm['money'], len(farm['unlocked_quadrants'])
        orig['_do_buy_land'](farm, board)
        if len(farm['unlocked_quadrants']) != q:
            p = seat_of(farm)
            flow['land_spend'][p][day()] += m - farm['money']
            land_steps[p].append(cur[0])

    def drop(private, cap):
        before = sum(private['shed'].values()) + sum(max(0, v) for inv in private['inventories'] for v in inv.values())
        orig['_drop_inventories_to_shed'](private, cap)
        p = 0 if env.state[0].observation.private is private else 1
        flow['discards'][p][day()] += before - sum(private['shed'].values())

    def animals(farm, d_):
        a = _count(farm, _is_animal)
        orig['_daily_refresh_animals'](farm, d_)
        flow['escapes'][seat_of(farm)][day()] += a - _count(farm, _is_animal)

    def plants(farm, d_, tpd):
        a = _count(farm, _is_plant)
        orig['_daily_refresh_plants'](farm, d_, tpd)
        flow['plant_deaths'][seat_of(farm)][day()] += a - _count(farm, _is_plant)

    def decay(farm, step):
        before = _plant_units(farm)
        orig['_decay_plants'](farm, step)
        lost = before - _plant_units(farm)
        if lost:
            flow['rot'][seat_of(farm)][day()] += lost

    for name, fn in zip(HOOKED, (commit, hire, land, drop, animals, plants, decay)):
        setattr(E, name, fn)
    try:
        for t, pair in enumerate(actions):
            if env.done:
                break
            cur[0] = t
            env.step(pair)
            if t % DAY_STEPS == DAY_STEPS - 1 or env.done:
                obs = env.state[0].observation
                prices = obs.market['prices']
                d = min(N_DAYS - 1, t // DAY_STEPS)
                for p in (0, 1):
                    farm, priv = obs.farms[p], env.state[p].observation.private
                    dusk['cash'][p][d] = float(farm['money'])
                    dusk['quadrants'][p][d] = len(farm['unlocked_quadrants'])
                    dusk['animals'][p][d] = _count(farm, _is_animal)
                    dusk['plants'][p][d] = _count(farm, _is_plant)
                    dusk['weeds'][p][d] = _count(farm, _is_weed)
                    dusk['shed_units'][p][d] = int(sum(priv['shed'].values()))
                    dusk['shed_value'][p][d] = float(sum(n * prices.get(k, 0) for k, n in priv['shed'].items()
                                                         if k in prices))
    finally:
        for name, fn in orig.items():
            setattr(E, name, fn)
    obs = env.state[0].observation
    return {'final_cash': [float(obs.farms[p]['money']) for p in (0, 1)], 'done': bool(env.done),
            'final_quadrants': [len(obs.farms[p]['unlocked_quadrants']) for p in (0, 1)],
            'flow': flow, 'dusk': dusk, 'land_steps': land_steps, 'sold_items': sold_items, 'sold_usd': sold_usd,
            'animals_by': animals_by, 'bought_items': bought_items, 'bought_usd': bought_usd}


def seat_records(eid, r, copy_key, fmt, status, error=None, quads=None, qsteps=None):
    """Two per-seat fact records from a replay() result r (or None when only the regex facts exist)."""
    out = []
    for p in (0, 1):
        rec = {'episode_id': int(eid), 'seat': p, 'facts_version': FACTS_VERSION, 'computed_utc': utc_stamp(),
               'status': status, 'error': (error or '')[:300] or None, 'copy': copy_key, 'copy_format': fmt}
        if r is not None:
            f, d = r['flow'], r['dusk']
            steps = r['land_steps'][p]
            hires = f['hires'][p]
            last = max((i for i in range(N_DAYS) if d['cash'][p][i] is not None), default=None)
            rec.update({
                'final_cash_replay': r['final_cash'][p], 'final_quadrants': r['final_quadrants'][p],
                'q2_day': steps[0] // DAY_STEPS if len(steps) > 0 else None,
                'q3_day': steps[1] // DAY_STEPS if len(steps) > 1 else None,
                'q4_day': steps[2] // DAY_STEPS if len(steps) > 2 else None,
                'land_spend': float(sum(f['land_spend'][p])), 'hires': int(sum(hires)),
                'hire_spend': float(sum(f['hire_spend'][p])), 'hire_days': sum(1 for x in hires if x),
                'max_hires_day': max(hires), 'first_hire_day': next((i for i, x in enumerate(hires) if x), None),
                'sold_units': int(sum(f['sold_units'][p])), 'sales_usd': float(sum(f['sales'][p])),
                'bought_usd': float(sum(f['bought'][p])), 'seed_spend': float(sum(f['seed_spend'][p])),
                'animal_spend': float(sum(f['animal_spend'][p])), 'product_spend': float(sum(f['product_spend'][p])),
                'animals_bought': int(sum(f['animals_bought'][p])), 'discards': int(sum(f['discards'][p])),
                'escapes': int(sum(f['escapes'][p])), 'plant_deaths': int(sum(f['plant_deaths'][p])),
                'rot_units': int(sum(f['rot'][p])),
                'end_animals': d['animals'][p][last] if last is not None else None,
                'end_plants': d['plants'][p][last] if last is not None else None,
                'end_weeds': d['weeds'][p][last] if last is not None else None,
                'end_shed_units': d['shed_units'][p][last] if last is not None else None,
                'end_shed_value': d['shed_value'][p][last] if last is not None else None,
                'peak_animals': max((x for x in d['animals'][p] if x is not None), default=None),
                'peak_plants': max((x for x in d['plants'][p] if x is not None), default=None),
                'sold_by_item': json.dumps({k: [r['sold_items'][p][k], round(r['sold_usd'][p][k], 2)]
                                            for k in sorted(r['sold_items'][p])}, separators=(',', ':')),
                'bought_by_item': json.dumps({k: [r['bought_items'][p][k], round(r['bought_usd'][p][k], 2)]
                                              for k in sorted(r['bought_items'][p])}, separators=(',', ':')),
                'animals_by_type': json.dumps(dict(sorted(r['animals_by'][p].items())), separators=(',', ':'))})
            for k in FLOW:
                rec['d_' + k] = [float(x) for x in f[k][p]]
            for k in DUSK:
                rec['d_' + k] = [None if x is None else float(x) for x in d[k][p]]
        else:
            if quads is not None:
                rec['final_quadrants'] = quads[p]
            if qsteps is not None:
                for k, name in ((2, 'q2_day'), (3, 'q3_day'), (4, 'q4_day')):
                    s = qsteps[p].get(k)
                    rec[name] = (s - 1) // DAY_STEPS if s else None
        out.append(rec)
    return out


def regex_quads(raw):
    """(final quadrants per seat, {k: first step whose observation shows k quadrants} per seat) from replay bytes."""
    found = QUAD_RE.findall(raw)
    if len(found) < 2:
        return None, None
    final = [len(json.loads(x)) for x in found[-2:]]
    qsteps = None
    if len(found) % 4 == 0:                   # steps[t][0].farms[0..1], steps[t][1].farms[0..1]
        qsteps = [{}, {}]
        for t in range(len(found) // 4):
            for p in (0, 1):
                n = found[4 * t + p].count(b',') + 1 if found[4 * t + p].strip(b'[] ') else 0
                for k in (2, 3, 4):
                    if n >= k and k not in qsteps[p]:
                        qsteps[p][k] = t
    return final, qsteps


def actions_from_raw(raw):
    """Joint actions from replay bytes without a full parse: each step entry starts {"action": ..., "info": ...}."""
    at = raw.find(b'"steps"')           # the header's "specification" also has an "action" object; skip it
    if at < 0:
        raise ValueError('no "steps" key')
    starts = [m.end() for m in ACT_RE.finditer(raw, at)]
    if len(starts) < 4 or len(starts) % 2:
        raise ValueError(f'{len(starts)} action keys')
    import orjson
    acts = []
    for s in starts:
        m = INFO_RE.search(raw, s, s + 400000)
        if not m:
            raise ValueError('action without a following "info" key')
        acts.append(orjson.loads(raw[s:m.start()]))
    return [[acts[2 * t] or PASS, acts[2 * t + 1] or PASS] for t in range(1, len(acts) // 2)]


def actions_full_parse(raw):
    import orjson
    g = orjson.loads(raw)
    steps = g['steps']
    return [[(steps[t + 1][p].get('action') or PASS) for p in (0, 1)] for t in range(len(steps) - 1)], g


def facts_from_replay_bytes(eid, raw, copy_key, fmt):
    head = C.parse_header(raw[:C.HEAD]) if b'"steps"' in raw[:C.HEAD] else C.parse_header(raw[:1 << 20])
    info = head.get('info') or {}
    rewards = [C.as_float(x) for x in C.pair(head.get('rewards'))]
    cfg = {k: v for k, v in (head.get('configuration') or {}).items() if k not in ('seed', 'actTimeout', 'runTimeout')}
    seed = info.get('seed')
    if seed is None or None in rewards:          # notebook replays write rewards / info after "steps"
        tail = C.tail_values(raw[-(1 << 16):])
        seed = seed if seed is not None else tail.get('seed')
        if None in rewards and tail.get('rewards'):
            rewards = [C.as_float(x) for x in C.pair(tail['rewards'])]
    quads, qsteps = regex_quads(raw)
    if seed is None:
        return seat_records(eid, None, copy_key, fmt, 'no-seed', 'replay header has no seed', quads, qsteps)
    err = None
    for attempt in ('regex', 'full'):     # the full parse only when the fast reader could not read the actions
        try:
            acts = actions_from_raw(raw) if attempt == 'regex' else actions_full_parse(raw)[0]
        except Exception as e:  # noqa: BLE001 - recorded per episode
            err = f'{attempt}: {e!r}'
            continue
        try:
            r = replay(seed, cfg, acts)
        except Exception as e:  # noqa: BLE001
            err = f'{attempt} re-play: {e!r}'
            break
        if None not in rewards and r['final_cash'] == rewards:
            return seat_records(eid, r, copy_key, fmt, 'ok')
        err = f'{attempt}: re-played cash {r["final_cash"]} != recorded {rewards}'
        break
    return seat_records(eid, None, copy_key, fmt, 'mismatch', err, quads, qsteps)


def facts_from_tape(eid, seed, cfg, actions, rewards, copy_key, fmt):
    rewards = [C.as_float(x) for x in C.pair(rewards)]
    try:
        r = replay(seed, cfg, actions)
    except Exception as e:  # noqa: BLE001
        return seat_records(eid, None, copy_key, fmt, 'error', repr(e))
    if None not in rewards and r['final_cash'] == rewards:
        return seat_records(eid, r, copy_key, fmt, 'ok')
    return seat_records(eid, None, copy_key, fmt, 'mismatch',
                        f're-played cash {r["final_cash"]} != recorded {rewards}', r['final_quadrants'], None)


XSF_READ = ['episode_id', 'seed', 'config', 'actions_zstd', 'reward_0', 'reward_1', 'first_bad_step', 'n_steps',
            'final_status', 'team_names', 'team_id_0', 'team_id_1', 'submission_id_0', 'submission_id_1',
            'initial_score_0', 'initial_score_1', 'updated_score_0', 'updated_score_1', 'create_time']


def xsf_rows(archive, shard, columns=XSF_READ):
    import pyarrow.parquet as pq
    with zipfile.ZipFile(archive) as z:
        return pq.read_table(io.BytesIO(z.read(shard)), columns=columns).to_pylist()


def xsf_actions(row):
    import orjson
    import zstandard
    return orjson.loads(zstandard.ZstdDecompressor().decompress(row['actions_zstd']))


def load_tape_file(path):
    import orjson
    with gzip.open(path, 'rb') if str(path).endswith('.gz') else open(path, 'rb') as f:
        return orjson.loads(f.read())


def facts_task(task):
    """One unit of worker work: [(eid, format, archive, member)] sharing a container. Returns (records, n_ok, seconds)."""
    C.be_nice()
    t0 = time.time()
    kind, items = task['kind'], task['items']
    recs = []
    try:
        if kind == 'zip-json':
            with zipfile.ZipFile(task['archive']) as z:
                for eid, fmt, archive, member in items:
                    key = f'{archive}:{member}'
                    try:
                        raw = z.read(member)
                    except Exception as e:  # noqa: BLE001
                        recs += seat_records(eid, None, key, fmt, 'error', f'read: {e!r}')
                        continue
                    recs += facts_from_replay_bytes(eid, raw, key, fmt)
                    del raw
        elif kind == 'xsf-tape':
            rows = xsf_rows(task['archive'], task['shard'])
            for eid, fmt, archive, member in items:
                key = f'{archive}:{member}'
                i = int(member.rsplit('#', 1)[1])
                row = rows[i] if i < len(rows) else None
                if row is None or int(row['episode_id']) != eid:
                    recs += seat_records(eid, None, key, fmt, 'error', 'xsf row not found')
                    continue
                if row['actions_zstd'] is None:
                    recs += seat_records(eid, None, key, fmt, 'no-actions', 'xsf row has no actions')
                    continue
                cfg = {k: v for k, v in json.loads(row['config'] or '{}').items()
                       if k not in ('seed', 'actTimeout', 'runTimeout')}
                recs += facts_from_tape(eid, row['seed'], cfg, xsf_actions(row), [row['reward_0'], row['reward_1']],
                                        key, fmt)
        else:                                     # loose files: json / json-gz replays, capsules, compact tapes
            for eid, fmt, archive, member in items:
                key = archive
                try:
                    if fmt in ('json', 'json-gz'):
                        with gzip.open(archive, 'rb') if archive.endswith('.gz') else open(archive, 'rb') as f:
                            raw = f.read()
                        recs += facts_from_replay_bytes(eid, raw, key, fmt)
                        del raw
                    elif fmt == 'capsule':
                        t = load_tape_file(archive)
                        cfg = {k: v for k, v in (t.get('configuration') or {}).items()
                               if k not in ('seed', 'actTimeout', 'runTimeout')}
                        recs += facts_from_tape(eid, t['seed'], cfg, t['joint_actions'], t.get('rewards'), key, fmt)
                    elif fmt == 'tape-compact':
                        t = load_tape_file(archive)
                        recs += facts_from_tape(eid, t['seed'], dict(REF_CFG), t['actions'], t.get('rewards'), key, fmt)
                    elif fmt == 'tape-selfplay':
                        t = load_tape_file(archive)
                        cfg = {k: v for k, v in (t.get('configuration') or REF_CFG).items()
                               if k not in ('seed', 'actTimeout', 'runTimeout')}
                        recs += facts_from_tape(eid, t['world_seed'], cfg, t['actions'], t.get('rewards'), key, fmt)
                    elif fmt == 'recorded-game':
                        t = load_tape_file(archive)
                        recs += facts_from_tape(eid, t['world_seed'], dict(REF_CFG), [x['actions'] for x in t['steps']],
                                                t.get('final_cash'), key, fmt)
                    else:
                        recs += seat_records(eid, None, key, fmt, 'unsupported', f'format {fmt}')
                except Exception as e:  # noqa: BLE001
                    recs += seat_records(eid, None, key, fmt, 'error', repr(e))
    except Exception as e:  # noqa: BLE001 - container failure: every item of the task is recorded as an error
        done = {r['episode_id'] for r in recs}
        for eid, fmt, archive, member in items:
            if eid not in done:
                recs += seat_records(eid, None, f'{archive}:{member}', fmt, 'error', f'container: {e!r}')
    n_ok = sum(1 for r in recs if r['status'] == 'ok') // 2
    return recs, n_ok, time.time() - t0


# ------------------------------------------------------------------ facts store (append-only shards)
def facts_schema():
    import pyarrow as pa
    L = pa.list_(pa.float32())             # engine money and prices are integers: exact in float32 below 16.7M
    cols = [('episode_id', pa.int64()), ('seat', pa.int8()), ('facts_version', pa.int16()), ('computed_utc', pa.string()),
            ('status', pa.string()), ('error', pa.string()), ('copy', pa.string()), ('copy_format', pa.string()),
            ('final_cash_replay', pa.float64()), ('final_quadrants', pa.int16()), ('q2_day', pa.int16()),
            ('q3_day', pa.int16()), ('q4_day', pa.int16()), ('land_spend', pa.float64()), ('hires', pa.int32()),
            ('hire_spend', pa.float64()), ('hire_days', pa.int16()), ('max_hires_day', pa.int32()),
            ('first_hire_day', pa.int16()), ('sold_units', pa.int32()), ('sales_usd', pa.float64()),
            ('bought_usd', pa.float64()), ('seed_spend', pa.float64()), ('animal_spend', pa.float64()),
            ('product_spend', pa.float64()), ('animals_bought', pa.int32()), ('discards', pa.int32()),
            ('escapes', pa.int32()), ('plant_deaths', pa.int32()), ('rot_units', pa.int32()), ('end_animals', pa.int16()),
            ('end_plants', pa.int16()), ('end_weeds', pa.int16()), ('end_shed_units', pa.int32()),
            ('end_shed_value', pa.float64()), ('peak_animals', pa.int16()), ('peak_plants', pa.int16()),
            ('sold_by_item', pa.string()), ('bought_by_item', pa.string()), ('animals_by_type', pa.string())]
    cols += [('d_' + k, L) for k in FLOW + DUSK]
    return pa.schema(cols)


SCALAR_FACTS = [n for n in facts_schema().names if not n.startswith('d_')]


def write_facts_shard(recs):
    import pyarrow as pa
    import pyarrow.parquet as pq
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f'F-{C.utc_now().strftime("%Y%m%dT%H%M%S%f")}-{os.getpid()}.parquet'
    tmp = FACTS_DIR / ('.' + name + '.tmp')
    pq.write_table(pa.Table.from_pylist(recs, schema=facts_schema()), tmp, compression='zstd')
    os.replace(tmp, FACTS_DIR / name)
    return FACTS_DIR / name


def facts_files():
    return sorted(p for p in FACTS_DIR.glob('*.parquet') if not p.name.startswith('.'))


def read_facts(columns=None):
    """All fact rows (consolidated + shards), latest record per (episode_id, seat): ok beats non-ok, then newest."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    files = facts_files()
    if not files:
        return facts_schema().empty_table() if columns is None else \
            pa.schema([facts_schema().field(c) for c in columns]).empty_table()
    need = None if columns is None else sorted(set(columns) | {'episode_id', 'seat', 'status', 'computed_utc',
                                                                  'facts_version'})
    tables = []
    for p in files:                      # older files may lack newer columns: read what each has, fill the rest
        names = pq.read_schema(p).names
        tables.append(pq.read_table(p, columns=None if need is None else [c for c in need if c in names]))
    t = pa.concat_tables(tables, promote_options='default')
    del tables
    fs = facts_schema()
    for c in (fs.names if need is None else need):
        if c not in t.column_names:
            t = t.append_column(fs.field(c), pa.nulls(t.num_rows, fs.field(c).type))
    t = t.filter(pc.equal(t['facts_version'], FACTS_VERSION))
    ok = pc.cast(pc.equal(t['status'], 'ok'), pa.int8())
    t = t.append_column('_ok', ok).append_column('_i', pa.array(range(t.num_rows), pa.int64()))
    t = t.sort_by([('episode_id', 'ascending'), ('seat', 'ascending'), ('_ok', 'descending'),
                   ('computed_utc', 'descending'), ('_i', 'descending')])
    eid, seat = t['episode_id'].to_numpy(), t['seat'].to_numpy()
    import numpy as np
    keep = np.ones(len(eid), dtype=bool)
    keep[1:] = (eid[1:] != eid[:-1]) | (seat[1:] != seat[:-1])
    t = t.filter(pa.array(keep)).drop_columns(['_ok', '_i'])
    if columns is not None:
        t = t.select([c for c in columns])
    return t


def consolidate_facts():
    """Fold shards into one FACTS-<stamp>.parquet (latest record per seat) once there are many; removes only the folded
    shards (our own generated files)."""
    files = facts_files()
    if len(files) < 12:
        return None
    import pyarrow.parquet as pq
    t = read_facts()
    name = f'FACTS-{C.utc_now().strftime("%Y%m%dT%H%M%S")}.parquet'
    tmp = FACTS_DIR / ('.' + name + '.tmp')
    pq.write_table(t, tmp, compression='zstd')
    os.replace(tmp, FACTS_DIR / name)
    for p in files:
        p.unlink()
    log(f'facts: consolidated {len(files)} files into {name} ({t.num_rows} seat rows)')
    return FACTS_DIR / name


# ------------------------------------------------------------------ pending episodes -> tasks (main side)
def load_stored():
    import pyarrow.parquet as pq
    cols = ['episode_id', 'stored', 'day', 'kaggle', 'config_ok', 'rewards', 'tsv_ready', 'formats']
    return [e for e in pq.read_table(CAT / 'EPISODES.parquet', columns=cols).to_pylist() if e['stored']]


def facts_copy_choice(copies):
    """Best copy to read facts from: cheapest exact source first (tapes), then zip members, then loose replays."""
    cands = [c for c in copies if c['format'] in FORMAT_PREF and not c.get('scan_error')]
    cands.sort(key=lambda c: (FORMAT_PREF[c['format']], c.get('config_ok') is False, c['rank'], c['archive'],
                              c['member']))
    return cands


def pending_tasks(day_from=None, limit=None, retry_errors=False):
    import pyarrow.parquet as pq
    eps = {e['episode_id']: e for e in load_stored()}
    t = pq.read_table(CAT / 'COPIES.parquet', columns=['episode_id', 'format', 'archive', 'member', 'rank', 'config_ok',
                                                        'scan_error', 'source'])
    copies = collections.defaultdict(list)
    for r in t.to_pylist():
        if r['episode_id'] in eps:
            copies[r['episode_id']].append(r)
    have = read_facts(['episode_id', 'seat', 'status', 'copy'])
    done = collections.defaultdict(dict)       # eid -> {copy: status}
    for eid, st, cp in zip(have['episode_id'].to_pylist(), have['status'].to_pylist(), have['copy'].to_pylist()):
        done[eid][cp] = st
    todo, skipped = [], collections.Counter()
    for eid, e in eps.items():
        if day_from and (e['day'] or '') < day_from:
            continue
        tried = done.get(eid, {})
        if any(s == 'ok' for s in tried.values()):
            continue
        cands = facts_copy_choice(copies.get(eid, []))
        if not cands:
            skipped['no supported copy'] += 1
            continue
        pick = None
        for c in cands:
            key = c['archive'] if c['format'] in LOOSE_FORMATS else \
                f"{c['archive']}:{c['member']}"
            st = tried.get(key)
            if st is None or (retry_errors and st == 'error'):
                pick = c
                break
        if pick is None:
            skipped['every copy already tried'] += 1
            continue
        todo.append((e['day'] or '', eid, pick))
    todo.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if limit:
        todo = todo[:limit]
    groups = collections.OrderedDict()
    for day, eid, c in todo:
        fmt = c['format']
        if fmt == 'zip-json':
            key = ('zip-json', c['archive'], None)
        elif fmt == 'xsf-tape':
            key = ('xsf-tape', c['archive'], c['member'].rsplit('#', 1)[0])
        else:
            key = ('loose', None, None)
        groups.setdefault(key, []).append((day, (eid, fmt, c['archive'], c['member'])))
    tasks = []
    for (kind, archive, shard), items in groups.items():
        size = {'zip-json': 12, 'xsf-tape': 1000, 'loose': 40}[kind]
        for i in range(0, len(items), size):
            chunk = items[i:i + size]
            tasks.append({'kind': kind, 'archive': archive, 'shard': shard, 'day': max(d for d, _ in chunk),
                          'items': [x for _, x in chunk]})
    tasks.sort(key=lambda t: t['day'], reverse=True)
    return tasks, len(todo), dict(skipped)


def run_pool(tasks, fn, workers, deadline, on_result, gb_each):
    """apply_async with a bounded window (results handled as they finish, whatever their order); stops submitting at
    the deadline or when MemAvailable < MIN_FREE_GB. A task that raises is logged and skipped (retried next run)."""
    import gc
    import multiprocessing as mp
    import queue
    if not tasks:
        return 0, 'nothing to do'
    gc.collect()
    gc.freeze()                      # forked workers do not copy the parent's objects when the collector runs
    state = {'stop': None, 'submitted': 0, 'inflight': 0, 'failed': 0}
    results = queue.Queue()
    it = iter(tasks)
    with mp.get_context('fork').Pool(workers, initializer=C.be_nice, maxtasksperchild=40) as pool:

        def submit():
            if state['stop']:
                return False
            if time.time() >= deadline:
                state['stop'] = 'deadline'
                return False
            if mem_available_gb() < MIN_FREE_GB:
                state['stop'] = f'MemAvailable {mem_available_gb():.1f} GB < {MIN_FREE_GB} GB'
                return False
            t = next(it, None)
            if t is None:
                state['stop'] = 'all tasks submitted'
                return False
            pool.apply_async(fn, (t,), callback=lambda r: results.put(('ok', r)),
                             error_callback=lambda e: results.put(('error', e)))
            state['submitted'] += 1
            state['inflight'] += 1
            return True

        for _ in range(workers + 2):          # two queued tasks: workers stay busy, little overrun past a deadline
            if not submit():
                break
        while state['inflight']:
            kind, r = results.get()
            state['inflight'] -= 1
            if kind == 'ok':
                on_result(r)
            else:
                state['failed'] += 1
                log(f'task failed (will be retried next run): {r!r}'[:400])
            submit()
    stop = state['stop'] + (f", {state['failed']} tasks failed" if state['failed'] else '')
    return state['submitted'], stop


def cmd_facts(a):
    C.be_nice()
    STATE.mkdir(parents=True, exist_ok=True)
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        lock = C.Lock(FACTS_DIR / '.facts.lock').__enter__()
    except SystemExit as e:
        print(json.dumps({'facts': 'skipped', 'why': str(e)}))
        return 0
    try:
        t0 = time.time()
        deadline = t0 + a.max_minutes * 60 if a.max_minutes else float('inf')
        if a.deadline_epoch:
            deadline = min(deadline, a.deadline_epoch)
        tasks, n_todo, skipped = pending_tasks(a.day_from, a.limit, a.retry_errors)
        workers = workers_for(a.workers, GB_PER_FACTS_WORKER)
        log(f'facts: {n_todo} episodes pending in {len(tasks)} tasks (skipped {skipped}); {workers} workers, '
            f'MemAvailable {mem_available_gb():.1f} GB')
        if not tasks or workers == 0:
            print(json.dumps({'facts': 'nothing to do' if not tasks else 'not enough free memory', 'pending': n_todo}))
            return 0
        buf, stats = [], collections.Counter()
        last_flush = [time.time()]
        last_build = [time.time()]

        def flush():
            if buf:
                write_facts_shard(buf)
                buf.clear()
            last_flush[0] = time.time()

        def on_result(res):
            recs, n_ok, secs = res
            buf.extend(recs)
            for r in recs:
                if r['seat'] == 0:
                    stats[r['status']] += 1
            stats['seconds'] += secs
            n = sum(v for k, v in stats.items() if k != 'seconds')
            if len(buf) >= 4000 or time.time() - last_flush[0] > 300:
                flush()
                log(f'facts: {n}/{n_todo} episodes ({dict((k, v) for k, v in stats.items() if k != "seconds")}), '
                    f'{time.time() - t0:.0f}s, MemAvailable {mem_available_gb():.1f} GB')
            if a.build_every and time.time() - last_build[0] > a.build_every * 60:
                flush()
                try:
                    build(quiet=True)
                except SystemExit as e:
                    log(f'facts: interim build skipped ({e})')
                last_build[0] = time.time()

        submitted, why = run_pool(tasks, facts_task, workers, deadline, on_result, GB_PER_FACTS_WORKER)
        flush()
        out = {'facts': 'done', 'pending_before': n_todo, 'tasks_run': submitted, 'tasks_total': len(tasks),
               'stopped': why, 'episodes': {k: v for k, v in stats.items() if k != 'seconds'},
               'cpu_seconds': round(stats['seconds'], 1), 'wall_seconds': round(time.time() - t0, 1)}
        print(json.dumps(out))
        return 0
    finally:
        lock.__exit__()


# ------------------------------------------------------------------ SEATS / SEAT_DAYS / views (main side)
def newest_leaderboard():
    files = sorted(LB_DIR.glob('LEADERBOARD-*.csv'), key=lambda p: C._lb_stamp(p.name) or '')
    if not files:
        return None, {}
    import csv
    lb = {}
    for r in csv.DictReader(open(files[-1], encoding='utf-8-sig')):
        tid = C.as_int(r.get('TeamId'))
        if tid is not None:
            lb[tid] = (C.as_int(r.get('Rank')), C.as_float(r.get('Score')), r.get('TeamName'))
    return files[-1], lb


def submission_scores():
    """submission id -> newest known public score (SUBMISSIONS-*.json, LISTINGS team policies, copy-* manifests)."""
    snaps = []
    for p in C.EXTRA.glob('**/SUBMISSIONS-*.json'):
        try:
            for s in json.loads(p.read_text()):
                if C.as_int(s.get('id')) and C.as_float(s.get('publicScore')) is not None:
                    snaps.append((p.stat().st_mtime, int(s['id']), float(s['publicScore'])))
        except Exception:  # noqa: BLE001
            continue
    for p in C.EXTRA.glob('**/LISTINGS-*.json'):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        for tid, t in (d.get('teams') or {}).items():
            for sid, sc in t.get('policies') or []:
                if C.as_int(sid) and C.as_float(sc) is not None:
                    snaps.append((p.stat().st_mtime, int(sid), float(sc)))
    for mf in C.EXTRA.glob('*/MANIFEST.jsonl'):
        mt = mf.stat().st_mtime
        try:
            for line in open(mf, encoding='utf-8'):
                if '"policy_sub_score"' not in line:
                    continue
                r = json.loads(line)
                if C.as_int(r.get('policy_sub')) and C.as_float(r.get('policy_sub_score')) is not None:
                    snaps.append((mt, int(r['policy_sub']), float(r['policy_sub_score'])))
        except Exception:  # noqa: BLE001
            continue
    out = {}
    for _, sid, sc in sorted(snaps):
        out[sid] = sc
    return out


def xsf_before_scores():
    """episode id -> [rating before the game per seat] from the xsf crawl rows (COPIES reason_detail)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(CAT / 'COPIES.parquet', columns=['episode_id', 'format', 'reason_detail'])
    t = t.filter(pc.equal(t['format'], 'xsf-tape'))
    out = {}
    for eid, rd in zip(t['episode_id'].to_pylist(), t['reason_detail'].to_pylist()):
        if not rd:
            continue
        try:
            s = (json.loads(rd).get('scan') or {}).get('initial_scores')
        except ValueError:
            continue
        if s:
            out[eid] = [C.as_float(x) for x in C.pair(s)]
    return out


SEAT_COLS = None   # filled by seats_schema()


def seats_schema():
    import pyarrow as pa
    S, I8, I16, I32, I64, F, B = pa.string(), pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.float64(), pa.bool_()
    cols = [('episode_id', I64), ('seat', I8), ('ep_order', I64), ('day', S), ('create_time', S), ('day_from', S),
            ('episode_type', S), ('kaggle', B), ('config_ok', B), ('tsv_ready', B), ('source', S), ('format', S),
            ('archive', S), ('member', S), ('sources', pa.list_(S)), ('n_copies', I32),
            ('team', S), ('team_id', I64), ('submission_id', I64), ('opp_team', S), ('opp_team_id', I64),
            ('opp_submission_id', I64), ('is_us', B), ('opp_is_us', B), ('our_version', S), ('self_play', B),
            ('static', B), ('final_cash', F), ('opp_cash', F), ('margin', F), ('result', S), ('win', B),
            ('game_score', F), ('game_score_before', F), ('opp_game_score', F), ('official_min_score', F),
            ('official_max_score', F), ('sub_score', F), ('opp_sub_score', F), ('lb_rank', I32), ('lb_score', F),
            ('opp_lb_rank', I32), ('opp_lb_score', F), ('rating', F), ('rating_from', S), ('opp_rating', F),
            ('age_days', I32), ('usability', F), ('opp_final_quadrants', I16), ('facts_status', S), ('facts_copy', S),
            ('facts_format', S), ('facts_error', S)]
    fs = facts_schema()
    skip = {'episode_id', 'seat', 'facts_version', 'computed_utc', 'status', 'error', 'copy', 'copy_format'}
    cols += [(n, fs.field(n).type) for n in SCALAR_FACTS if n not in skip]
    return pa.schema(cols)


def usability(age, rating, quads, tau=3.0, sigma=41.0, r0=2880.0, quad4_bonus=1.5):
    if age is None or rating is None:
        return None
    u = math.exp(-age / tau) * math.exp((rating - r0) / sigma)
    return u * math.exp(quad4_bonus) if quads == 4 else u


def build(quiet=False):
    """Join EPISODES (stored) + facts + leaderboard/submission scores -> SEATS.parquet, SEAT_DAYS.parquet, views/."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t0 = time.time()
    STATE.mkdir(parents=True, exist_ok=True)
    with C.Lock(STATE / '.seats-build.lock'):
        consolidate_facts()
        ep_cols = ['episode_id', 'kaggle', 'stored', 'n_copies', 'sources', 'tsv_ready', 'preferred_source',
                   'preferred_format', 'preferred_archive', 'preferred_member', 'day', 'day_from', 'create_time',
                   'episode_type', 'team_names', 'team_ids', 'submission_ids', 'rewards', 'winner_seat', 'config_ok',
                   'involves_us', 'our_seat', 'self_play', 'our_version', 'game_scores', 'official_avg_score',
                   'official_min_score', 'static_seats']
        eps = pq.read_table(CAT / 'EPISODES.parquet', columns=ep_cols).to_pylist()
        lb_file, lb = newest_leaderboard()
        subs = submission_scores()
        before = xsf_before_scores()
        facts = read_facts(SCALAR_FACTS)
        fidx = {(e, s): i for i, (e, s) in enumerate(zip(facts['episode_id'].to_pylist(), facts['seat'].to_pylist()))}
        fcols = {n: facts[n].to_pylist() for n in SCALAR_FACTS}
        today = C.utc_now().date()
        rows = []
        for order, e in enumerate(eps):
            if not e['stored']:
                continue
            names, tids, sids = C.pair(e['team_names']), C.pair(e['team_ids']), C.pair(e['submission_ids'])
            rw = [C.as_float(x) for x in C.pair(e['rewards'])]
            gs = [C.as_float(x) for x in C.pair(e['game_scores'])]
            gb = before.get(e['episode_id'], [None, None])
            static = C.pair(e['static_seats'])
            avg, mn = e['official_avg_score'], e['official_min_score']
            mx = 2 * avg - mn if avg is not None and mn is not None else None
            try:
                age = (today - dt.date.fromisoformat(e['day'])).days if e['day'] else None
            except ValueError:
                age = None
            ours = [bool(e['involves_us'] and (e['self_play'] or e['our_seat'] == s)) for s in (0, 1)]
            lbs = [lb.get(tids[s]) if tids[s] is not None else None for s in (0, 1)]
            fq = [None, None]
            for s in (0, 1):
                i = fidx.get((e['episode_id'], s))
                fq[s] = fcols['final_quadrants'][i] if i is not None else None
            for s in (0, 1):
                o = 1 - s
                res = None
                if None not in rw:
                    res = 'W' if rw[s] > rw[o] else 'L' if rw[s] < rw[o] else 'T'
                rating, rfrom = (gs[s], 'game_score') if gs[s] is not None else \
                    ((lbs[s][1], 'leaderboard') if lbs[s] and lbs[s][1] is not None else (None, None))
                opp_rating = gs[o] if gs[o] is not None else (lbs[o][1] if lbs[o] else None)
                row = {'episode_id': e['episode_id'], 'seat': s, 'ep_order': order, 'day': e['day'],
                       'create_time': e['create_time'], 'day_from': e['day_from'], 'episode_type': e['episode_type'],
                       'kaggle': e['kaggle'], 'config_ok': e['config_ok'], 'tsv_ready': e['tsv_ready'],
                       'source': e['preferred_source'], 'format': e['preferred_format'],
                       'archive': e['preferred_archive'], 'member': e['preferred_member'], 'sources': e['sources'],
                       'n_copies': e['n_copies'], 'team': names[s], 'team_id': tids[s], 'submission_id': sids[s],
                       'opp_team': names[o], 'opp_team_id': tids[o], 'opp_submission_id': sids[o], 'is_us': ours[s],
                       'opp_is_us': ours[o], 'our_version': e['our_version'] if ours[s] else None,
                       'self_play': e['self_play'], 'static': static[s], 'final_cash': rw[s], 'opp_cash': rw[o],
                       'margin': rw[s] - rw[o] if None not in rw else None, 'result': res, 'win': res == 'W',
                       'game_score': gs[s], 'game_score_before': gb[s], 'opp_game_score': gs[o],
                       'official_min_score': mn, 'official_max_score': mx,
                       'sub_score': subs.get(sids[s]) if sids[s] is not None else None,
                       'opp_sub_score': subs.get(sids[o]) if sids[o] is not None else None,
                       'lb_rank': lbs[s][0] if lbs[s] else None, 'lb_score': lbs[s][1] if lbs[s] else None,
                       'opp_lb_rank': lbs[o][0] if lbs[o] else None, 'opp_lb_score': lbs[o][1] if lbs[o] else None,
                       'rating': rating, 'rating_from': rfrom, 'opp_rating': opp_rating, 'age_days': age,
                       'usability': usability(age, rating, fq[s]) if res == 'W' else None,
                       'opp_final_quadrants': fq[o]}
                i = fidx.get((e['episode_id'], s))
                if i is None:
                    row['facts_status'] = 'pending'
                else:
                    row['facts_status'], row['facts_copy'], row['facts_format'], row['facts_error'] = (
                        fcols['status'][i], fcols['copy'][i], fcols['copy_format'][i], fcols['error'][i])
                    for n in SCALAR_FACTS:
                        if n not in ('episode_id', 'seat', 'facts_version', 'computed_utc', 'status', 'error', 'copy',
                                     'copy_format'):
                            row[n] = fcols[n][i]
                rows.append(row)
        del eps, fcols
        schema = seats_schema()
        table = pa.Table.from_pylist(rows, schema=schema)
        del rows
        tmp = CAT / 'SEATS.parquet.tmp'
        pq.write_table(table, tmp, compression='zstd')
        os.replace(tmp, CAT / 'SEATS.parquet')
        n_days = write_seat_days()
        views = write_views(lb_file)
        st = collections.Counter(table['facts_status'].to_pylist())
        summary = {'built_utc': utc_stamp(), 'seconds': round(time.time() - t0, 1), 'seat_rows': table.num_rows,
                   'episodes': table.num_rows // 2, 'leaderboard': str(lb_file) if lb_file else None,
                   'submission_scores': len(subs), 'facts_status_seats': dict(st), 'seat_day_rows': n_days,
                   'facts_files': [p.name for p in facts_files()], 'views': views}
        by_day = collections.defaultdict(collections.Counter)
        for d, s, seat in zip(table['day'].to_pylist(), table['facts_status'].to_pylist(), table['seat'].to_pylist()):
            if seat == 0:
                by_day[d or 'unknown'][s] += 1
        summary['facts_by_day'] = {d: dict(c) for d, c in sorted(by_day.items(), reverse=True)}
        tmp = CAT / 'SEATS-SUMMARY.json.tmp'
        tmp.write_text(json.dumps(summary, indent=1))
        os.replace(tmp, CAT / 'SEATS-SUMMARY.json')
    if not quiet:
        print(json.dumps({k: summary[k] for k in ('seat_rows', 'facts_status_seats', 'seat_day_rows', 'seconds')}))
    return summary


COUNT_FIELDS = {'sold_units', 'animals_bought', 'hires', 'discards', 'escapes', 'plant_deaths', 'rot', 'quadrants',
                'animals', 'plants', 'weeds', 'shed_units'}


def write_seat_days():
    """SEAT_DAYS.parquet: one row per (episode, seat, game day) for seats whose facts are exact re-plays. Rewritten only
    when the facts store changed since the last write."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    sig = json.dumps([(p.name, p.stat().st_size) for p in facts_files()])
    sig_file = STATE / 'SEAT_DAYS.sig'
    if (CAT / 'SEAT_DAYS.parquet').exists() and sig_file.exists() and sig_file.read_text() == sig:
        return pq.ParquetFile(CAT / 'SEAT_DAYS.parquet').metadata.num_rows
    cols = ['episode_id', 'seat', 'status'] + ['d_' + k for k in FLOW + DUSK]
    t = read_facts(cols)
    t = t.filter(pc.equal(t['status'], 'ok'))
    n = t.num_rows
    lens = pc.list_value_length(t['d_cash']).to_numpy(zero_copy_only=False) if n else np.zeros(0, dtype=np.int64)
    parent = np.repeat(np.arange(n), lens)
    starts = np.repeat(np.cumsum(lens) - lens, lens)
    game_day = (np.arange(len(parent)) - starts).astype(np.int16)
    out = {'episode_id': pc.take(t['episode_id'], pa.array(parent)), 'seat': pc.take(t['seat'], pa.array(parent)),
           'game_day': pa.array(game_day)}
    for k in FLOW + DUSK:
        col = pc.list_flatten(t['d_' + k])
        out[k] = pc.cast(col, pa.int32()) if k in COUNT_FIELDS else col
    table = pa.table(out)
    del t, out
    tmp = CAT / 'SEAT_DAYS.parquet.tmp'
    pq.write_table(table, tmp, compression='zstd')
    os.replace(tmp, CAT / 'SEAT_DAYS.parquet')
    sig_file.write_text(sig)
    return table.num_rows


# ------------------------------------------------------------------ selection (pick + views)
MANIFEST = Path('/mnt/ai/scratch/farmNEAT/corpus-final/MANIFEST.parquet')
MANIFEST_COLS = ['recency_tier', 'provenance', 'canonical_class', 'first_arrival_utc', 'lb_rank_at_fetch', 'grabber_fetch_log']


def load_seats(columns=None):
    """SEATS rows, joined (when present) with the corpus-final manifest's per-episode provenance and recency tier
    (owner 09-29: tier 1 = dailies 09-24..09-29, 2 = 09-28/09-29, 3 = top-12 grabber/manual downloads; 0 = older)."""
    import pandas as pd
    want = None if columns is None else [c for c in columns if c not in MANIFEST_COLS]
    if want is not None and 'episode_id' not in want:
        want = want + ['episode_id']
    df = pd.read_parquet(CAT / 'SEATS.parquet', columns=want)
    if (columns is None or any(c in MANIFEST_COLS for c in columns)) and MANIFEST.exists():
        m = pd.read_parquet(MANIFEST, columns=['episode_id'] + MANIFEST_COLS)
        df = df.merge(m, on='episode_id', how='left')
        df['recency_tier'] = df['recency_tier'].fillna(0).astype(int)
    return df


WHERE_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|==|!=|!~|=|>|<|~)\s*(.*?)\s*$')


def where_mask(df, expr):
    m = WHERE_RE.match(expr)
    if not m:
        raise SystemExit(f'--where {expr!r}: expected COLUMN OP VALUE, e.g. "hires>=100" or "team==DSM"')
    col, op, val = m.groups()
    if col not in df.columns:
        raise SystemExit(f'--where: no column {col!r}; columns: {", ".join(df.columns)}')
    if op in ('~', '!~'):                      # case-insensitive substring
        want = C.fold(val)
        hit = df[col].map(lambda x: isinstance(x, str) and want in C.fold(x)).astype(bool)
        return hit if op == '~' else ~hit
    v = val
    if val.lower() in ('true', 'false'):
        v = val.lower() == 'true'
    elif val.lower() in ('none', 'null'):
        v = None
    else:
        try:
            v = float(val) if re.search(r'[.eE]', val) else int(val)
        except ValueError:
            v = val
    s = df[col]
    if v is None:
        return s.isna() if op in ('=', '==') else s.notna()
    import operator
    fn = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt, '=': operator.eq,
          '==': operator.eq, '!=': operator.ne}[op]
    if s.dtype.kind in 'biuf':
        if isinstance(v, str):
            raise SystemExit(f'--where: column {col} is numeric, got {val!r}')
        return fn(s, v).fillna(False).astype(bool) if s.dtype.kind == 'f' else fn(s, v)

    def safe(x):
        try:
            return x is not None and x == x and bool(fn(x, v))
        except TypeError:
            return False
    return s.map(safe).astype(bool)


PRESETS = {
    # work/runs/maintainer/pbc-selfplay/build_v5s_rows.py (2026-09-27 12:38): winners (ties out), not ours, zip copy,
    # config ok, 09-19..09-27, rating = game score else team leaderboard score, u = exp(-age/3) exp((r-2880)/41)
    # x e^1.5 if the winner ended with 4 quadrants, top 9000 by u, weights 3/2/1 by thirds.
    # 14:55 version: plus the winner seat of every game we lost 09-26..09-27 (weight 2, "beat-us" rows).
    'v5s': dict(wins=True, exclude_us=True, day_from='2026-09-19', day_to='2026-09-27', ref_day='2026-09-27',
                usability=True, quad4_bonus=1.5, budget=9000, weights='3,2,1', reason='v5s', order='u',
                beat_us_since='2026-09-26', beat_us_weight=2),
    # the same rule on a moving window: the 9 newest game days, ages counted from today (UTC), our losses of the 2 newest
    'usability': dict(wins=True, exclude_us=True, recent_days=9, usability=True, quad4_bonus=1.5, budget=9000,
                      weights='3,2,1', reason='u9000', order='u', beat_us_days=2, beat_us_weight=2),
    'beat-us': dict(wins=True, exclude_us=True, where=['opp_is_us==True'], reason='beatus'),
    'wins2880': dict(wins=True, exclude_us=True, min_rating=2880.0, day_from='2026-09-03', reason='w2880'),
    'wins4q': dict(wins=True, exclude_us=True, quads=4, day_from='2026-09-03', reason='w4q'),
    'wins4q2880': dict(wins=True, exclude_us=True, quads=4, min_rating=2880.0, day_from='2026-09-03', reason='w4q2880'),
    'beat-top30': dict(wins=True, exclude_us=True, beat_top_n=30, day_from='2026-09-03', reason='beat30'),
    'recent-wins': dict(wins=True, exclude_us=True, recent_days=3, reason='recent'),
    'recent-2880': dict(wins=True, exclude_us=True, recent_days=3, min_rating=2880.0, reason='recent2880'),
    'ours-wins': dict(wins=True, us=True, reason='ours'),
    'league-wins': dict(wins=True, include_local=True, include_selfplay=True, source=['league'], reason='league',
                        any_config=True),
}


def pick_args(**over):
    """Namespace with every pick option at its default (views and tests call select() through this)."""
    ns = argparse.Namespace(wins=False, result=None, us=False, exclude_us=False, min_rating=None, min_game_score=None,
                            min_sub_score=None, min_official_score=None, day_from=None, day_to=None, recent_days=None, quads=None, min_quads=None,
                            q4_by=None, team=None, top_n=None, opp_top_n=None, beat_top_n=None, min_margin=None,
                            source=None, include_local=False, include_selfplay=False, any_config=False,
                            include_static=True, where=None, usability=False, tau=3.0, sigma=41.0, r0=2880.0,
                            quad4_bonus=1.5, ref_day=None, budget=None, weight=1, weights=None, reason='',
                            order='newest', facts_ok=False, beat_us_since=None, beat_us_days=None, beat_us_weight=2)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def select(df, a):
    """Filter SEATS rows (pandas) with pick options; returns the ordered selection with a weight and reason column."""
    import numpy as np
    m = df['tsv_ready'].eq(True)
    if getattr(a, 'tier_min', None) is not None:
        if 'recency_tier' not in df.columns:
            raise SystemExit('--tier-min needs /mnt/ai/scratch/farmNEAT/corpus-final/MANIFEST.parquet (not built yet)')
        m &= df['recency_tier'].ge(a.tier_min)
    if not a.include_local:
        m &= df['kaggle'].eq(True)
    if not a.any_config:
        m &= df['config_ok'].eq(True)
    if not a.include_selfplay:
        m &= ~df['self_play'].eq(True)
    if not a.include_static:
        m &= ~df['static'].eq(True)
    if a.wins:
        m &= df['result'] == 'W'
    if a.result:
        m &= df['result'] == {'win': 'W', 'loss': 'L', 'tie': 'T'}.get(a.result, a.result)
    if a.us:
        m &= df['is_us'].eq(True)
    if a.exclude_us:
        m &= ~df['is_us'].eq(True)
    today = C.utc_now().date()
    if a.recent_days:
        days = sorted(d for d in df.loc[m, 'day'].dropna().unique() if C.DAY_RE.fullmatch(str(d)))
        keep = set(days[-a.recent_days:])
        m &= df['day'].isin(keep)
    if a.day_from:
        m &= df['day'].fillna('') >= a.day_from
    if a.day_to:
        m &= df['day'].fillna('9999') <= a.day_to
    if a.min_rating is not None:
        m &= df['rating'].fillna(-1e9) >= a.min_rating
    if a.min_game_score is not None:
        m &= df['game_score'].fillna(-1e9) >= a.min_game_score
    if a.min_sub_score is not None:
        m &= df['sub_score'].fillna(-1e9) >= a.min_sub_score
    if a.min_official_score is not None:
        m &= df['official_min_score'].fillna(-1e9) >= a.min_official_score
    if a.quads is not None:
        m &= df['final_quadrants'] == a.quads
    if a.min_quads is not None:
        m &= df['final_quadrants'].fillna(-1) >= a.min_quads
    if a.q4_by is not None:
        m &= df['q4_day'].fillna(99) <= a.q4_by
    if a.min_margin is not None:
        m &= df['margin'].fillna(-1e18) >= a.min_margin
    if a.top_n is not None:
        m &= df['lb_rank'].fillna(10 ** 9) <= a.top_n
    if a.opp_top_n is not None:
        m &= df['opp_lb_rank'].fillna(10 ** 9) <= a.opp_top_n
    if a.beat_top_n is not None:
        m &= (df['result'] == 'W') & (df['opp_lb_rank'].fillna(10 ** 9) <= a.beat_top_n)
    if a.facts_ok:
        m &= df['facts_status'] == 'ok'
    if a.source:
        want = set(a.source)
        m &= df['sources'].map(lambda xs: bool(want & set(xs)) if xs is not None else False)
    if a.team:
        aliases, teams = C.load_aliases(), C.load_teams()
        tm = np.zeros(len(df), dtype=bool)
        folded = df['team'].map(lambda n: C.fold(n) if isinstance(n, str) else '')
        for text in a.team:
            ids, names, _ = C.resolve_team_arg(text, teams, aliases)
            tm |= (df['team_id'].isin(ids) | folded.isin(names)).to_numpy()
        m &= tm
    for expr in a.where or []:
        m &= where_mask(df, expr)
    sel = df[m].copy()
    if a.usability:
        ref = dt.date.fromisoformat(a.ref_day) if a.ref_day else today
        sel = sel[sel['rating'].notna() & sel['day'].notna()]
        age = [(ref - dt.date.fromisoformat(d)).days for d in sel['day']]
        bonus = math.exp(a.quad4_bonus) if a.quad4_bonus else 1.0
        sel['u'] = [math.exp(-g / a.tau) * math.exp((r - a.r0) / a.sigma) * (bonus if q == 4 else 1.0)
                    for g, r, q in zip(age, sel['rating'], sel['final_quadrants'])]
        sel['age'] = age
    if a.order == 'u' and 'u' in sel:
        sel = sel.sort_values(['u', 'ep_order'], ascending=[False, True], kind='mergesort')
    elif a.order == 'rating':
        sel = sel.sort_values(['rating', 'day', 'episode_id'], ascending=[False, False, False], kind='mergesort')
    else:
        sel = sel.sort_values(['day', 'episode_id', 'seat'], ascending=[False, False, True], kind='mergesort')
    if a.budget:
        sel = sel.head(a.budget)
    n = len(sel)
    if a.weights:
        ws = [int(x) for x in str(a.weights).split(',')]
        part = max(1, n // len(ws))
        sel['weight'] = [ws[min(i // part, len(ws) - 1)] for i in range(n)]
    else:
        sel['weight'] = a.weight
    tag = a.reason or 'pick'
    if 'u' in sel:
        sel['reason'] = [f'{tag}:u={u:.3g}:r={r:.0f}:age={g}:q4={int(q == 4)}'
                         for u, r, g, q in zip(sel['u'], sel['rating'], sel['age'], sel['final_quadrants'])]
    else:
        sel['reason'] = [f'{tag}:{"win" if res == "W" else "loss" if res == "L" else "tie"}:{t}'
                         for res, t in zip(sel['result'], sel['team'])]
    if a.beat_us_since or a.beat_us_days:
        # owner 2026-09-27 "make sure we're training over all our losses too": the winner seat of every game we lost
        # (any rating / configuration), appended after the ranked rows unless already selected
        import pandas as pd
        b = df[(df['result'] == 'W') & df['opp_is_us'].eq(True) & ~df['is_us'].eq(True) & df['tsv_ready'].eq(True)]
        days = sorted(d for d in b['day'].dropna().unique() if C.DAY_RE.fullmatch(str(d)))
        since = a.beat_us_since or (days[-a.beat_us_days] if len(days) >= a.beat_us_days else (days[0] if days else ''))
        until = a.day_to or a.ref_day or '9999'
        b = b[(b['day'].fillna('') >= since) & (b['day'].fillna('9999') <= until)]
        have = set(zip(sel['episode_id'], sel['seat']))
        b = b[[k not in have for k in zip(b['episode_id'], b['seat'])]].sort_values('ep_order', kind='mergesort').copy()
        b['weight'] = a.beat_us_weight
        b['reason'] = [f'{tag}:beat-us:{t}' for t in b['team']]
        sel = pd.concat([sel, b])
    return sel


def write_selection(sel, out, meta=None):
    out = Path(out)
    lines = ['archive\tmember\tday\teid\tseat\tweight\treason\n']
    for r in sel[['archive', 'member', 'day', 'episode_id', 'seat', 'weight', 'reason']].itertuples(index=False):
        lines.append(f'{r.archive}\t{r.member}\t{r.day or "unknown"}\t{r.episode_id}\t{r.seat}\t{r.weight}\t'
                     f'{str(r.reason).replace(chr(9), " ")}\n')
    tmp = out.with_name(out.name + '.tmp')
    tmp.write_text(''.join(lines), encoding='utf-8')
    os.replace(tmp, out)
    files = out.with_suffix('.files')
    tmp = files.with_name(files.name + '.tmp')
    tmp.write_text(''.join(sorted({f'{a}\n' for a in sel['archive']})), encoding='utf-8')
    os.replace(tmp, files)
    days = sel['day'].dropna()
    summ = {'rows': int(len(sel)), 'episodes': int(sel['episode_id'].nunique()),
            'weighted_rows': int(sel['weight'].sum()) if len(sel) else 0,
            'archives': int(sel['archive'].nunique()), 'days': [min(days), max(days)] if len(days) else None,
            'by_day': {str(k): int(v) for k, v in sel['day'].value_counts().sort_index(ascending=False).items()},
            'top_teams': {str(k): int(v) for k, v in sel['team'].value_counts().head(12).items()},
            'final_quadrants': {str(k): int(v) for k, v in sel['final_quadrants'].value_counts(dropna=False).items()},
            'facts_status': {str(k): int(v) for k, v in sel['facts_status'].value_counts().items()}}
    if 'u' in sel and len(sel):
        summ['u_cut'] = float(sel['u'].iloc[-1])
        summ['min_rating'] = float(sel['rating'].min())
    if meta:
        summ.update(meta)
    tmp = out.with_suffix('.summary.json.tmp')
    tmp.write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    os.replace(tmp, out.with_suffix('.summary.json'))
    return summ


VIEW_DEFS = [
    ('WINS-2880', 'wins2880', 'winning seats rated >= 2880 (game score, else team leaderboard score), 2026-09-03 on, '
                              'not ours'),
    ('WINS-4Q', 'wins4q', 'winning seats that ended with 4 quadrants, 2026-09-03 on, not ours'),
    ('WINS-4Q-2880', 'wins4q2880', 'winning seats with 4 quadrants and rating >= 2880, 2026-09-03 on, not ours'),
    ('USABILITY-9000', 'usability', 'top 9000 winners by u = exp(-age/3) exp((rating-2880)/41) x e^1.5 if 4 quadrants, '
                                    'last 9 game days, weights 3/2/1 by thirds, not ours; plus the winner of every game we '
                                    'lost on the 2 newest days (weight 2)'),
    ('BEAT-TOP30', 'beat-top30', 'winner of every stored game a current top-30 team lost, 2026-09-03 on, not ours'),
    ('RECENT-WINS', 'recent-wins', 'every winning seat of the 3 newest game days, any rating, not ours'),
    ('RECENT-2880', 'recent-2880', 'winning seats of the 3 newest game days rated >= 2880, not ours'),
    ('BEAT-US', 'beat-us', 'winner seat of every stored Kaggle game our team lost (any rating), newest first'),
    ('OURS-WINS', 'ours-wins', 'our own winning seats (Kaggle games)'),
    ('LEAGUE-WINS', 'league-wins', 'winning seats of our local league (extra/league)'),
]


def write_views(lb_file):
    VIEWS.mkdir(exist_ok=True)
    df = load_seats()
    out = {}
    lines = ['# Ready-made selections (rebuilt hourly by seats.py build)', '',
             f'Built {utc_stamp()} from SEATS.parquet; leaderboard {Path(lb_file).name if lb_file else "-"}. Each TSV is '
             'the trainer priority format (`archive member day eid seat weight reason`); `<name>.files` lists its '
             'archives, `<name>.summary.json` its counts. Rows point at zip-json copies only (tapes appear once '
             'rebuilt). Recreate or vary any of them with `seats.py pick --preset NAME [more filters] --out X.tsv`.', '',
             '| file | preset | rows | episodes | days | definition |', '|---|---|---|---|---|---|']
    for name, preset, desc in VIEW_DEFS:
        a = pick_args(**PRESETS[preset])
        sel = select(df, a)
        s = write_selection(sel, VIEWS / f'{name}.tsv', {'preset': preset, 'definition': desc})
        out[name] = {'rows': s['rows'], 'episodes': s['episodes'], 'days': s['days']}
        days = f"{s['days'][0]}..{s['days'][1]}" if s['days'] else '-'
        lines.append(f"| `{name}.tsv` | `{preset}` | {s['rows']} | {s['episodes']} | {days} | {desc} |")
    lines += ['', 'Re-play counters exist for seats with `facts_status == ok`; `final_quadrants` also for `mismatch` seats '
              '(read from the replay), so quadrant filters skip only seats still `pending`. Coverage by day: '
              '`SEATS-SUMMARY.json`.']
    tmp = VIEWS / 'INDEX.md.tmp'
    tmp.write_text('\n'.join(lines) + '\n')
    os.replace(tmp, VIEWS / 'INDEX.md')
    return out


def cmd_pick(a):
    base = dict(PRESETS[a.preset]) if a.preset else {}
    ns = pick_args(**base)
    for k, v in vars(a).items():
        if k in ('preset', 'out', 'cmd', 'columns'):
            continue
        if v is not None and v is not False:            # options given on the command line override the preset
            setattr(ns, k, v)
    df = load_seats()
    sel = select(df, ns)
    if a.out:
        s = write_selection(sel, a.out, {'preset': a.preset, 'args': {k: v for k, v in vars(ns).items()
                                                                          if v not in (None, False, '')}})
        print(json.dumps({k: s[k] for k in ('rows', 'episodes', 'weighted_rows', 'days', 'archives')} |
                         {'out': a.out}), file=sys.stderr)
    elif a.columns:
        cols = [c.strip() for c in a.columns.split(',')]
        sys.stdout.write(sel[cols].to_csv(sep='\t', index=False))
    else:
        write = sys.stdout.write
        write('archive\tmember\tday\teid\tseat\tweight\treason\n')
        for r in sel[['archive', 'member', 'day', 'episode_id', 'seat', 'weight', 'reason']].itertuples(index=False):
            write(f'{r.archive}\t{r.member}\t{r.day}\t{r.episode_id}\t{r.seat}\t{r.weight}\t{r.reason}\n')


# ------------------------------------------------------------------ show / status
def cmd_show(a):
    import pyarrow.parquet as pq
    eid = int(a.episode)
    t = pq.read_table(CAT / 'SEATS.parquet', filters=[('episode_id', '=', eid)]).to_pylist()
    if not t:
        print(f'{eid}: not in SEATS.parquet (not stored, or run seats.py build)')
        return
    t.sort(key=lambda r: r['seat'])
    width = max(len(k) for k in t[0])
    long_ = ('sold_by_item', 'bought_by_item', 'animals_by_type', 'archive', 'facts_copy', 'facts_error', 'sources')
    for k in t[0]:
        if k in long_:
            continue
        vals = ['' if r[k] is None else f'{r[k]:.1f}' if isinstance(r[k], float) else str(r[k]) for r in t]
        print(f'{k:{width}}  ' + ' | '.join(f'{v:>24}' for v in vals))
    for k in long_:
        for r in t:
            if r[k] not in (None, '', []):
                print(f'{k} (seat {r["seat"]}): {r[k]}')
    if a.days:
        rows = pq.read_table(CAT / 'SEAT_DAYS.parquet', filters=[('episode_id', '=', eid)]).to_pylist()
        if rows:
            cols = [c for c in rows[0] if c != 'episode_id']
            print('\t'.join(cols))
            for r in sorted(rows, key=lambda r: (r['seat'], r['game_day'])):
                print('\t'.join('' if r[c] is None else f'{r[c]:g}' if isinstance(r[c], float) else str(r[c])
                                for c in cols))


def cmd_status(a):
    s = json.loads((CAT / 'SEATS-SUMMARY.json').read_text()) if (CAT / 'SEATS-SUMMARY.json').exists() else {}
    tasks, n_todo, skipped = pending_tasks()
    rb = rebuild_candidates()
    counts = {}
    for name in ('MANIFEST.jsonl', 'REJECTS.jsonl'):
        p = REBUILT / name
        counts[name] = sum(1 for line in open(p, encoding='utf-8') if line.strip()) if p.exists() else 0
    print(json.dumps({'seats_built': s.get('built_utc'), 'seat_rows': s.get('seat_rows'),
                      'facts_status_seats': s.get('facts_status_seats'), 'facts_pending_episodes': n_todo,
                      'facts_pending_tasks': len(tasks), 'facts_skipped': skipped,
                      'rebuilt_episodes': counts['MANIFEST.jsonl'], 'rebuild_rejected': counts['REJECTS.jsonl'],
                      'rebuild_pending_episodes': len(rb), 'rebuild_pending_days': dict(sorted(collections.Counter(
                          d for d, *_ in rb).items(), reverse=True)), 'views': s.get('views'),
                      'timer': 'systemctl --user list-timers catalog-refresh.timer; log state/pipeline.log'}, indent=1))


# ------------------------------------------------------------------ rebuild seed+action tapes into zip-json
def replay_template():
    """Every non-step field of a reference-configuration official replay (cached in state/REPLAY-TEMPLATE.json)."""
    p = STATE / 'REPLAY-TEMPLATE.json'
    if p.exists():
        return json.loads(p.read_text())
    import orjson
    for z in sorted(C.DATA.glob('kaggriculture-episodes-*.zip'), reverse=True):
        with zipfile.ZipFile(z) as zz:
            member = next((n for n in zz.namelist() if re.fullmatch(r'\d+\.json', n)), None)
            if not member:
                continue
            g = orjson.loads(zz.read(member))
        cfg = dict(g['configuration'])
        cfg.pop('seed', None)
        ref = dict(C.REFERENCE)
        ref.pop('seed', None)
        if cfg == ref:
            tpl = {k: v for k, v in g.items() if k not in ('steps',)}
            tpl['template_from'] = f'{z}:{member}'
            tmp = p.with_suffix('.tmp')
            tmp.write_text(json.dumps(tpl, indent=1))
            os.replace(tmp, p)
            return tpl
    raise SystemExit('no official replay with the reference configuration')


def rebuilt_done():
    done = set()
    for name in ('MANIFEST.jsonl', 'REJECTS.jsonl'):
        p = REBUILT / name
        if p.exists():
            for line in open(p, encoding='utf-8'):
                if line.strip():
                    try:
                        r = json.loads(line)
                        if not r.get('transient'):      # read errors are retried; engine/tape failures are final
                            done.add(int(r['episode_id']))
                    except (ValueError, KeyError):
                        pass
    return done


def rebuild_candidates(day_from=None):
    """Stored episodes with no replay copy (zip or loose json) but an xsf or capsule tape: [(day, eid, format, archive,
    member)], newest day first, xsf before capsule."""
    import pyarrow.parquet as pq
    eps = {e['episode_id']: e for e in load_stored() if e['kaggle']}
    t = pq.read_table(CAT / 'COPIES.parquet', columns=['episode_id', 'format', 'archive', 'member', 'scan_error'])
    by = collections.defaultdict(list)
    for r in t.to_pylist():
        if r['episode_id'] in eps:
            by[r['episode_id']].append(r)
    done = rebuilt_done()
    out = []
    for eid, cs in by.items():
        if eid in done or any(c['format'] in C.REPLAY_FORMATS for c in cs):
            continue
        tape = sorted((c for c in cs if c['format'] in ('xsf-tape', 'capsule') and not c['scan_error']),
                      key=lambda c: (c['format'] != 'xsf-tape', c['archive'], c['member']))
        if not tape:
            continue
        day = eps[eid]['day'] or ''
        if day_from and day < day_from:
            continue
        c = tape[0]
        out.append((day, eid, c['format'], c['archive'], c['member']))
    out.sort(key=lambda x: (x[2] == 'xsf-tape', x[0], x[1]), reverse=True)     # xsf first, newest day first
    return out


_TEMPLATE = None


def _row_from_capsule(path):
    import zstandard
    t = load_tape_file(path)
    cfg = {k: v for k, v in (t.get('configuration') or {}).items() if k != 'seed'}
    rw = C.pair(t.get('rewards'))
    return {'episode_id': int(t['episode_id']), 'seed': int(t['seed']), 'config': json.dumps(cfg),
            'actions_zstd': zstandard.ZstdCompressor().compress(json.dumps(t['joint_actions']).encode()),
            'first_bad_step': -1, 'n_steps': 720, 'final_status': '["DONE", "DONE"]',
            'team_names': json.dumps(t.get('teams')), 'reward_0': rw[0], 'reward_1': rw[1],
            'team_id_0': None, 'team_id_1': None, 'submission_id_0': None, 'submission_id_1': None,
            'initial_score_0': None, 'initial_score_1': None, 'updated_score_0': None, 'updated_score_1': None,
            'create_time': None}


def rebuild_task(task):
    """Rebuild one batch of tapes into one zip (written as a hidden .tmp next to the final name, read back and hashed).
    Returns (tmp path, final name, manifest rows, reject rows, seconds)."""
    C.be_nice()
    t0 = time.time()
    src = str(C.REPO / 'src')
    if src not in sys.path:
        sys.path.insert(0, src)
    sar_dir = str(C.REPO / 'work/runs/maintainer/pbc-seed-action-replays')
    if sar_dir not in sys.path:
        sys.path.insert(0, sar_dir)
    import seed_action_replays as SAR
    tmp = REBUILT / 'archives' / ('.' + task['name'] + '.tmp')
    rows, rejects = [], []
    shards = {}
    try:
        with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
            for day, eid, fmt, archive, member in task['items']:
                try:
                    if fmt == 'xsf-tape':
                        shard, i = member.rsplit('#', 1)
                        if (archive, shard) not in shards:
                            shards.clear()
                            shards[(archive, shard)] = xsf_rows(archive, shard)
                        row = shards[(archive, shard)][int(i)]
                        if int(row['episode_id']) != eid:
                            raise ValueError('xsf row mismatch')
                    else:
                        row = _row_from_capsule(archive)
                    if row['n_steps'] != 720 or row['final_status'] != '["DONE", "DONE"]':
                        raise ValueError(f"incomplete source game: n_steps {row['n_steps']}, {row['final_status']}")
                    ep = SAR.reconstruct_episode(row, task['template'])
                    raw = json.dumps(ep, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
                    del ep
                    info = zipfile.ZipInfo(f'{eid}.json', date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    z.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
                    names = json.loads(row['team_names']) if row.get('team_names') else None
                    rw = [float(row['reward_0']), float(row['reward_1'])]
                    rows.append({'episode_id': eid, 'member': f'{eid}.json', 'sha256': hashlib.sha256(raw).hexdigest(),
                                 'bytes': len(raw), 'day': day or None, 'create_time': row.get('create_time'),
                                 'team_names': names, 'team_ids': [row.get('team_id_0'), row.get('team_id_1')],
                                 'submission_ids': [row.get('submission_id_0'), row.get('submission_id_1')],
                                 'rewards': rw, 'winner_seat': C.winner(rw), 'seed': int(row['seed']),
                                 'scores': [row.get('updated_score_0'), row.get('updated_score_1')],
                                 'scores_before': [row.get('initial_score_0'), row.get('initial_score_1')],
                                 'reason': 'xsf-rebuild' if fmt == 'xsf-tape' else 'capsule-rebuild',
                                 'source': 'xsf-raw' if fmt == 'xsf-tape' else 'capsules', 'source_copy':
                                     f'{archive}:{member}' if member else archive,
                                 'checks': {'engine_final_cash_equals_tape_rewards': True, 'joint_actions': 719,
                                            'statuses': ['DONE', 'DONE'], 'remainingOverageTime': 'omitted'}})
                    del raw
                except Exception as e:  # noqa: BLE001 - one bad tape never stops the batch
                    rejects.append({'episode_id': eid, 'day': day or None, 'source_copy': f'{archive}:{member}',
                                    'error': repr(e)[:300], 'rejected_utc': utc_stamp(),
                                    'transient': isinstance(e, (OSError, MemoryError, zipfile.BadZipFile))})
        with zipfile.ZipFile(tmp) as z:                     # read back every member and compare its hash
            for r in rows:
                h = hashlib.sha256()
                with z.open(r['member']) as f:
                    for chunk in iter(lambda: f.read(1 << 22), b''):
                        h.update(chunk)
                if h.hexdigest() != r['sha256']:
                    raise RuntimeError(f'zip readback hash mismatch {r["episode_id"]}')
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return str(tmp), task['name'], rows, rejects, time.time() - t0


def cmd_rebuild(a):
    C.be_nice()
    (REBUILT / 'archives').mkdir(parents=True, exist_ok=True)
    try:
        lock = C.Lock(REBUILT / '.rebuild.lock').__enter__()
    except SystemExit as e:
        print(json.dumps({'rebuild': 'skipped', 'why': str(e)}))
        return 0
    try:
        for old in (REBUILT / 'archives').glob('.*.tmp'):       # our own leftovers of an interrupted run
            old.unlink()
        t0 = time.time()
        deadline = t0 + a.max_minutes * 60 if a.max_minutes else float('inf')
        if a.deadline_epoch:
            deadline = min(deadline, a.deadline_epoch)
        if a.after_facts:
            _, n_facts, _ = pending_tasks()
            if n_facts > a.after_facts:
                print(json.dumps({'rebuild': 'waiting for facts backlog', 'facts_pending': n_facts}))
                return 0
        cands = rebuild_candidates(a.day_from)
        if a.limit:
            cands = cands[:a.limit]
        workers = workers_for(a.workers, GB_PER_REBUILD_WORKER)
        log(f'rebuild: {len(cands)} tapes pending; {workers} workers; MemAvailable {mem_available_gb():.1f} GB')
        if not cands or workers == 0:
            print(json.dumps({'rebuild': 'nothing to do' if not cands else 'not enough free memory',
                              'pending': len(cands)}))
            return 0
        template = replay_template()
        template = {k: v for k, v in template.items() if k != 'template_from'}
        stamp = C.utc_now().strftime('%Y%m%dT%H%M%S')
        tasks = []
        by_day = collections.OrderedDict()
        for c in cands:
            by_day.setdefault((c[0], c[2]), []).append(c)
        for (day, fmt), items in by_day.items():
            items.sort(key=lambda c: (c[3], c[4]))
            for i in range(0, len(items), a.batch):
                chunk = items[i:i + a.batch]
                tag = 'xsf' if fmt == 'xsf-tape' else 'capsule'
                tasks.append({'name': f'rebuilt-{tag}-{day or "unknown"}-{stamp}-{len(tasks):04d}.zip',
                              'items': chunk, 'template': template})
        stats = collections.Counter()

        def on_result(res):
            tmp, name, rows, rejects, secs = res
            if rows:
                os.replace(tmp, REBUILT / 'archives' / name)
                now = utc_stamp()
                with open(REBUILT / 'MANIFEST.jsonl', 'a', encoding='utf-8') as f:
                    f.write(''.join(json.dumps(dict(r, archive=f'archives/{name}', rebuilt_utc=now), sort_keys=True,
                                               ensure_ascii=False) + '\n' for r in rows))
                    f.flush()
                    os.fsync(f.fileno())
            else:
                Path(tmp).unlink(missing_ok=True)
            if rejects:
                with open(REBUILT / 'REJECTS.jsonl', 'a', encoding='utf-8') as f:
                    f.write(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rejects))
            stats['rebuilt'] += len(rows)
            stats['rejected'] += len(rejects)
            stats['zips'] += bool(rows)
            stats['cpu_seconds'] += secs
            log(f'rebuild: {name} {len(rows)} ok, {len(rejects)} rejected ({secs:.0f}s); total {stats["rebuilt"]} '
                f'of {len(cands)}, MemAvailable {mem_available_gb():.1f} GB')

        submitted, why = run_pool(tasks, rebuild_task, workers, deadline, on_result, GB_PER_REBUILD_WORKER)
        print(json.dumps({'rebuild': 'done', 'pending_before': len(cands), 'batches_run': submitted,
                          'batches_total': len(tasks), 'stopped': why, **{k: round(v, 1) for k, v in stats.items()},
                          'wall_seconds': round(time.time() - t0, 1)}))
        return 0
    finally:
        lock.__exit__()


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('build', help='SEATS.parquet + SEAT_DAYS.parquet + views/ from the catalog and the facts store')
    p = sp.add_parser('facts', help='exact re-play of stored games without facts (newest day first)')
    p.add_argument('--max-minutes', type=float, default=0, help='stop submitting work after this many minutes')
    p.add_argument('--deadline-epoch', type=float, default=0)
    p.add_argument('--workers', type=int, default=WORKERS)
    p.add_argument('--day-from')
    p.add_argument('--limit', type=int)
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--build-every', type=float, default=20, help='rebuild SEATS every N minutes during a long run')
    p = sp.add_parser('pick', help='trainer priority TSV from SEATS.parquet')
    p.add_argument('--preset', choices=sorted(PRESETS))
    p.add_argument('--wins', action='store_true', help='winning seats only')
    p.add_argument('--result', choices=['win', 'loss', 'tie'])
    p.add_argument('--us', action='store_true', help='only our seats (team id 16624026 under any name, SELF:)')
    p.add_argument('--exclude-us', action='store_true', help='never our seats')
    p.add_argument('--min-rating', type=float, help='rating = game score after the game, else team leaderboard score')
    p.add_argument('--min-game-score', type=float, help='Kaggle game score only (unknown = excluded)')
    p.add_argument('--min-sub-score', type=float, help="the seat's submission's public score")
    p.add_argument('--min-official-score', type=float,
                   help='official daily manifest min_score >= X (the lower-rated player of the game; official zips only)')
    p.add_argument('--day-from')
    p.add_argument('--day-to')
    p.add_argument('--recent-days', type=int, help='the N newest game days present in the selection')
    p.add_argument('--quads', type=int, help='final unlocked quadrants == N (needs facts)')
    p.add_argument('--min-quads', type=int)
    p.add_argument('--q4-by', type=int, help='4th quadrant bought by game day N (0-29)')
    p.add_argument('--team', nargs='+', help='team names / aliases (ALIASES.tsv)')
    p.add_argument('--top-n', type=int, help="seat's team current leaderboard rank <= N")
    p.add_argument('--opp-top-n', type=int, help="opponent's current leaderboard rank <= N")
    p.add_argument('--beat-top-n', type=int, help='winning seats whose opponent is currently ranked <= N')
    p.add_argument('--min-margin', type=float)
    p.add_argument('--source', nargs='+', help='episode has a copy in one of these catalog sources')
    p.add_argument('--include-local', action='store_true', help='keep local (non-Kaggle) games')
    p.add_argument('--include-selfplay', action='store_true', help='keep games where both seats are ours')
    p.add_argument('--any-config', action='store_true', help='keep games whose configuration differs from reference')
    p.add_argument('--facts-ok', action='store_true', help='only seats with exact re-play facts')
    p.add_argument('--tier-min', type=int, help='corpus-final recency_tier >= N (1 = dailies 09-24..29, 2 = 09-28/29, 3 = top-12 grabbed)')
    p.add_argument('--where', action='append',
                   help='any SEATS column: "hires>=100", "q4_day<=12", "team==DSM", "our_version~fwd1" (~ substring)')
    p.add_argument('--usability', action='store_true', help='compute u and order by it')
    p.add_argument('--tau', type=float)
    p.add_argument('--sigma', type=float)
    p.add_argument('--r0', type=float)
    p.add_argument('--quad4-bonus', type=float, help='log-multiplier for 4-quadrant seats (1.5 = x e^1.5; 0 = off)')
    p.add_argument('--ref-day', help='age reference day (default today, UTC)')
    p.add_argument('--budget', type=int, help='keep the first N rows in the chosen order')
    p.add_argument('--weight', type=int)
    p.add_argument('--weights', help='e.g. 3,2,1: weights by equal rank slices')
    p.add_argument('--order', choices=['newest', 'u', 'rating'])
    p.add_argument('--beat-us-since', help='also append the winner seat of every game we lost from this day on')
    p.add_argument('--beat-us-days', type=int, help='same, for the N newest days with such games')
    p.add_argument('--beat-us-weight', type=int, help='weight of those rows (default 2)')
    p.add_argument('--reason')
    p.add_argument('--columns', help='print these SEATS columns (TSV) instead of the trainer format')
    p.add_argument('--out')
    p = sp.add_parser('show', help='every fact of one stored episode')
    p.add_argument('episode')
    p.add_argument('--days', action='store_true', help='also the per-game-day table')
    sp.add_parser('status', help='coverage and pending counts')
    p = sp.add_parser('rebuild-tapes', help='xsf / capsule seed+action tapes -> zip-json in extra/reconstructed-xsf')
    p.add_argument('--max-minutes', type=float, default=0)
    p.add_argument('--deadline-epoch', type=float, default=0)
    p.add_argument('--workers', type=int, default=WORKERS)
    p.add_argument('--batch', type=int, default=60, help='episodes per zip')
    p.add_argument('--day-from')
    p.add_argument('--limit', type=int)
    p.add_argument('--after-facts', type=int, default=0,
                   help='only run when at most N episodes still wait for facts (0 = always)')
    a = ap.parse_args(argv)
    if a.cmd == 'build':
        C.be_nice()
        build()
    elif a.cmd == 'facts':
        return cmd_facts(a)
    elif a.cmd == 'pick':
        cmd_pick(a)
    elif a.cmd == 'show':
        cmd_show(a)
    elif a.cmd == 'status':
        cmd_status(a)
    elif a.cmd == 'rebuild-tapes':
        return cmd_rebuild(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
