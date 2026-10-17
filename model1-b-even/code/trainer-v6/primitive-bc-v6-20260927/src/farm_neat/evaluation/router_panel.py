"""Paired Router-family regression/evaluation, using only ScenarioEnvironment.

A run FIRST replays all 80 retained K0006-PANEL rows (including action hashes).
Only exact agreement opens the 72-world candidate/parent panel. Both roles play
fresh games in both seats against router, k0006, tape0, PASS, and a custom parent
when applicable. Paired margin = candidate's cash margin - parent's cash margin
against the SAME rival/world/seat. Identical identities never short-circuit play.

Registry v1: 8 catalog worlds + all 64 ordered pairs of SHOPS. Pair (i,j) has seed
1700000000+8*i+j; its remaining six shops come from a private Random(seed).
No random weeds. This is a registered, exposed fixed-world panel, NOT the ladder.
World bootstrap resamples the 72 two-seat world means, not 144 independent seats.

Candidate/parent refs may be builtins or ``module:factory``. A factory takes NO
world/seat/seed argument and returns a fresh callable (or .act adapter) with a
nonempty JSON .identity. Pin learned weights/configuration and all transitive
assets in that identity's repo-relative (or absolute) ``assets`` path:SHA map;
freeze decoding/RNG behavior. Factory and callable source files are also bound.
This is a trusted in-process integration interface, not a sandbox or a proof
that arbitrary user code declares every dependency. Only normal observations
and a copied, allowlisted public configuration reach actor calls.

New outputs are exclusive. --resume requires the same manifest/actors/sources,
a valid journal hash chain and exact job prefix. It never overwrites a completed
summary. Starts and results are separately flushed/fsynced and hash-bound. A torn
line fails closed (retain it, no silent repair). A kill IN a game leaves a counted
start with UNKNOWN completion and BLOCKS resume: no automatic replay, retry, or
invented cash. A kill after the result commit resumes at the next job. v1 journals
cannot resume under this v2 protocol. Deadlines remain cooperative, not a kill.

Runner implementation and evidence kind are frozen in the manifest. Replacing the
native runner (e.g. a fixture oracle) marks the ENTIRE run SYNTHETIC_DIAGNOSTIC:
zero physical-game counts, no native regression certificate or playing claim.
Runner substitution on resume is refused before a new attempt. Diagnostic 80-row
matches may exercise the downstream reducer, but cannot open a native panel.

CPU maintainer invocation (verify current cluster/runtime access first):
  sbatch --partition=cpu --no-requeue --cpus-per-task=2 --mem=8G \\
    --time=01:00:00 --job-name=farmneat-router-panel --chdir="$PWD" \\
    --output='work/runs/router-panel-%j.log' \\
    --wrap='PYTHONPATH=src python3 -B -m farm_neat.evaluation.router_panel \\
      --output work/runs/router-panel-v2 --candidate k0006 --parent k0006 \\
      --max-games 1232'
Replace python3 with the declared interpreter when a candidate touches weights.
A fresh custom-parent panel can require 1520 games. Continue an interrupted run
with identical arguments and --resume. Reviewers: total native games <= 4;
--max-games defaults to FOUR PER INVOCATION, not a cross-process authorization.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
import fcntl
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import weakref
from typing import Any, Callable

from farm_neat.evaluation import k0006, scenarios, shop_router
from farm_neat.evaluation.common import bound, canonical, digest, equal, file_sha, read, write_new
from farm_neat.game import exact_engine

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'farm-router-paired-panel-v2'
SCOPE = 'REGISTERED_EXPOSED_FIXED_WORLDS; NOT_LADDER_STRENGTH; NO_PROMOTION_AUTHORITY'
SHOPS = ('BAKERY', 'BRUNCH_SPOT', 'FARMERS_MARKET', 'ICE_CREAM_SHOP',
         'PET_CAFE', 'PIZZA_SHOP', 'SMOOTHIE_SHOP', 'YARN_STORE')
LEGACY_PATH = ROOT / 'work/runs/reviewer-play/router-fallback/K0006-PANEL.json'
LEGACY_SHA256 = '57ef9ea5c7aeb1dc212a15e113747d8b7ae00d4db1a560098fb02d9fe65cb577'
LEGACY_FIELDS = ('agent', 'opponent', 'world', 'seat', 'status', 'cash', 'margin',
                 'wdl', 'actions_sha256', 'error')
LEGACY_PAIRS = (('k0006', 'router'), ('k0006', 'tape0'), ('k0006', 'pass'),
                ('router', 'router'), ('router', 'tape0'))


@dataclass(frozen=True)
class World:
    id: str
    purpose: str
    seed: int
    shops: tuple[str, ...]
    random_weeds: bool
    cohort: str

    def scenario(self) -> dict:
        # Opponent is a schema placeholder, NOT an opponent loader: play_game
        # explicitly supplies and verifies both actual actors below.
        return {'id': self.id, 'purpose': self.purpose, 'seed': self.seed,
                'shops': list(self.shops), 'random_weeds': self.random_weeds, 'opponent': 'pass'}

    @property
    def sha256(self) -> str:
        return digest(asdict(self))


class WorldRegistry:
    def __init__(self, worlds):
        self.worlds = tuple(worlds)
        if not self.worlds:
            raise ValueError('nonempty world registry required')
        ids, seeds, physical = set(), set(), set()
        for world in self.worlds:
            scenarios.validate(world.scenario())
            if world.cohort not in ('catalog', 'pair64') or not isinstance(world.shops, tuple):
                raise ValueError('immutable registered world/cohort required')
            key = (world.seed, world.shops, world.random_weeds)
            if world.id in ids or world.seed in seeds or key in physical:
                raise ValueError('duplicate world ID, seed or physical world')
            ids.add(world.id); seeds.add(world.seed); physical.add(key)
        self._by_id = {w.id: w for w in self.worlds}

    def __getitem__(self, key: str) -> World:
        return self._by_id[key]

    def document(self) -> dict:
        return bound({'schema': SCHEMA + '-worlds', 'worlds': [asdict(w) for w in self.worlds]})

    @property
    def sha256(self) -> str:
        return self.document()['content_sha256']


def world_registry() -> WorldRegistry:
    if set(SHOPS) != set(exact_engine.load_engine().SHOPS):
        raise ValueError('registered shop vocabulary changed')
    worlds = [World(s['id'], s['purpose'], s['seed'], tuple(s['shops']), s['random_weeds'], 'catalog')
              for _, s in sorted(scenarios.catalog().items())]
    if len(worlds) != 8:
        raise ValueError('exactly eight retained catalog worlds required')
    for i, first in enumerate(SHOPS):
        for j, second in enumerate(SHOPS):
            seed = 1700000000 + 8 * i + j
            rng = random.Random(seed)
            shops = (first, second, *(rng.choice(SHOPS) for _ in range(6)))
            worlds.append(World(f'pair-{i:02d}-{j:02d}', 'Registered ordered first-two-shop pair',
                                seed, shops, False, 'pair64'))
    return WorldRegistry(worlds)


def _path_name(path: Path) -> str:
    path = path.absolute()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _fingerprint(target) -> dict:
    target = getattr(target, '__func__', target)
    if not inspect.isfunction(target) and not inspect.isclass(target):
        target = type(target).__call__
    path = inspect.getsourcefile(target)
    if not path:
        raise ValueError('actor identity needs inspectable Python factory/callable source')
    return {'path': _path_name(Path(path)), 'sha256': file_sha(path),
            'module': target.__module__, 'qualname': target.__qualname__,
            'line': getattr(getattr(target, '__code__', None), 'co_firstlineno', None)}


def _callable(actor):
    call = getattr(actor, 'act', actor)
    if not callable(call):
        raise ValueError('actor must be callable or expose .act')
    return call


def _actor_binding(actor, factory) -> dict:
    declared = copy.deepcopy(getattr(actor, 'identity', None))
    if not isinstance(declared, dict) or not declared:
        raise ValueError('actor must declare a nonempty JSON identity')
    if 'content_sha256' in declared:
        equal(declared, bound(declared), 'declared actor identity binding')
    assets = declared.get('assets', {})
    if not isinstance(assets, dict):
        raise ValueError('actor identity assets must be a path:SHA mapping')
    for name, sha in assets.items():
        equal(file_sha(ROOT / name), sha, 'actor identity asset ' + name)
    return bound({'declared': declared, 'factory': _fingerprint(factory),
                  'callable': _fingerprint(_callable(actor))})


@dataclass
class ActorSpec:
    name: str
    factory: Callable[[], Any]
    identity: dict
    _last: Any = field(default=None, repr=False)
    _seen: Any = field(default_factory=weakref.WeakValueDictionary, repr=False)
    _nonweak_seen: dict = field(default_factory=dict, repr=False)

    def _remember(self, actor):
        # Weak values detect any still-live pooled controller, not merely the
        # immediately preceding one. Dead per-game controllers are not retained.
        key = id(actor)
        if self._seen.get(key) is actor or self._nonweak_seen.get(key) is actor:
            raise ValueError('fresh actor instance required, not a reused controller pool')
        try:
            self._seen[key] = actor
        except TypeError:
            # Slot-only callables may lack weak-reference support. Retain those
            # conservatively rather than silently admitting reused game state.
            self._nonweak_seen[key] = actor

    @property
    def sha256(self) -> str:
        return self.identity['content_sha256']

    def spawn(self):
        actor = self.factory()
        if actor is self._last:
            raise ValueError('fresh actor instance required, not a reused controller')
        observed = _actor_binding(actor, self.factory)
        if canonical(observed) != canonical(self.identity):
            raise ValueError(f'actor identity at spawn: {self.name}; expected={self.sha256}; '
                             f'observed={observed["content_sha256"]}')
        self._remember(actor)
        self._last = actor
        return actor


def register_actor(name: str, factory: Callable[[], Any]) -> ActorSpec:
    if not isinstance(name, str) or not name or not callable(factory):
        raise ValueError('named actor factory required')
    first = factory()
    spec = ActorSpec(name, factory, _actor_binding(first, factory), first)
    spec._remember(first)
    spec.spawn()  # Reject singleton factories before opening an output or playing.
    return spec


class PassActor:
    identity = {'kind': 'PASS', 'action': exact_engine.PASS_ACTION}

    def __call__(self, observation, configuration=None):
        return copy.deepcopy(exact_engine.PASS_ACTION)


def load_actor(ref: str) -> ActorSpec:
    factories = {'router': shop_router.ShopRouterAdapter, 'k0006': k0006.K0006Adapter,
                 'tape0': scenarios.FixedTape, 'pass': PassActor}
    if ref in factories:
        return register_actor(ref, factories[ref])
    if ref.count(':') != 1:
        raise ValueError('actor must name router/k0006/tape0/pass or module:factory')
    module, name = ref.split(':')
    return register_actor(ref, getattr(importlib.import_module(module), name))


def legacy_projection(row: dict) -> dict:
    return {key: row[key] for key in LEGACY_FIELDS}


def legacy_oracle(path: Path = LEGACY_PATH) -> dict:
    equal(file_sha(path), LEGACY_SHA256, 'immutable retained panel SHA256')
    value = read(path)
    expected = [(a, b, w, s) for a, b in LEGACY_PAIRS for w in sorted(scenarios.catalog()) for s in (0, 1)]
    actual = [(r['agent'], r['opponent'], r['world'], r['seat']) for r in value['rows']]
    equal(actual, expected, 'retained panel row order and coverage')
    if any(r['status'] != 'COMPLETE' or set(r) != set(LEGACY_FIELDS) for r in value['rows']):
        raise ValueError('complete retained panel rows required')
    for name, module in (('router', shop_router), ('k0006', k0006)):
        equal(module.identity()['content_sha256'], value['identities'][name], 'retained actor identity ' + name)
    equal(file_sha(scenarios.__file__), value['sources']['scenarios'], 'retained scenario source')
    return value


def legacy_jobs(oracle: dict) -> list[dict]:
    return [{'phase': 'legacy', 'role': 'legacy', **{k: r[k] for k in ('world', 'seat', 'agent', 'opponent')}}
            for r in oracle['rows']]


def jobs_for(plan: dict) -> list[dict]:
    jobs = copy.deepcopy(plan['legacy_jobs'])
    for world in plan['registry']['worlds']:
        for rival in plan['rivals']:
            for seat in (0, 1):
                for role in ('candidate', 'parent'):
                    jobs.append({'phase': 'panel', 'role': role, 'world': world['id'], 'seat': seat,
                                 'agent': plan[role], 'opponent': rival})
    return jobs


def make_plan(candidate='k0006', parent='k0006', *, bootstrap_samples=10000,
              bootstrap_seed=20260917, timeout=300.) -> tuple[dict, dict, WorldRegistry]:
    if type(bootstrap_samples) is not int or not 100 <= bootstrap_samples <= 100000:
        raise ValueError('bootstrap samples must be an integer in [100,100000]')
    if type(bootstrap_seed) is not int or not 0 <= bootstrap_seed < 2**63:
        raise ValueError('explicit nonnegative bootstrap seed required')
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError('game timeout must be in (0,300] seconds')
    oracle = legacy_oracle()
    registry = world_registry()
    rivals = list(dict.fromkeys(['router', 'k0006', 'tape0', 'pass', parent]))
    actors = {name: load_actor(name) for name in dict.fromkeys([*rivals, candidate])}
    paths = [Path(__file__), Path(scenarios.__file__), Path(shop_router.__file__), Path(k0006.__file__),
             Path(exact_engine.__file__), exact_engine.ENGINE_PATH, exact_engine.ENGINE_CONFIG_PATH,
             scenarios.CATALOG, ROOT / 'src/farm_neat/evaluation/common.py',
             ROOT / 'src/farm_neat/evaluation/opponents.py']
    plan = {'schema': SCHEMA, 'scope': SCOPE, 'candidate': candidate, 'parent': parent, 'rivals': rivals,
            'actors': {name: spec.identity for name, spec in actors.items()}, 'registry': registry.document(),
            'legacy_sha256': LEGACY_SHA256, 'legacy_jobs': legacy_jobs(oracle),
            'execution': execution_binding(play_game),
            'source_pins': {_path_name(p): file_sha(p) for p in paths}, 'python': sys.version,
            'bootstrap': {'seed': bootstrap_seed, 'samples': bootstrap_samples,
                          'method': 'percentile95_linear; resample whole two-seat world means'},
            'timeout_seconds': float(timeout), 'engine': exact_engine.assert_bundled_engine_identity()}
    plan['jobs_sha256'] = digest(jobs_for(plan))
    plan['games_planned'] = len(jobs_for(plan))
    return bound(plan), actors, registry


def public_guard(actor, *, seat: int, deadline: float):
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError('seat must be 0 or 1')
    def call(observation, configuration):
        if type(observation.get('player')) is not int or observation['player'] != seat:
            raise ValueError('actor observation seat mismatch')
        if time.monotonic() > deadline:
            raise TimeoutError('panel cooperative game deadline')
        public_config = {k: copy.deepcopy(v) for k, v in configuration.items()
                         if k in exact_engine.DEFAULT_CONFIGURATION}
        result = actor(observation, public_config)
        if time.monotonic() > deadline:
            raise TimeoutError('panel cooperative game deadline')
        return result
    return call


def play_game(world: World, agent: ActorSpec, opponent: ActorSpec, seat: int, timeout: float) -> dict:
    """One native complete-game attempt; no oracle, targets or rival state passed to actors."""
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError('seat must be 0 or 1')
    identities = [opponent.sha256, opponent.sha256]; identities[seat] = agent.sha256
    row = {'world': world.id, 'world_sha256': world.sha256, 'seed': world.seed, 'seat': seat,
           'agent': agent.name, 'opponent': opponent.name,
           'actor_identity_hashes': {'agent': agent.sha256, 'opponent': opponent.sha256},
           'actor_identities_by_seat': identities, 'status': 'FAILED', 'cash': None, 'margin': None,
           'wdl': None, 'actions_sha256': None, 'error': None, 'steps': 0,
           'min_rival_cash': None, 'rival_cash_trough': None, 'physical_games_completed': 0,
           'evidence_kind': 'NATIVE_GAME',
           'execution_sha256': execution_binding(_NATIVE_PLAY_GAME)['content_sha256']}
    started = time.monotonic(); env = None
    try:
        actor, rival = agent.spawn(), opponent.spawn()
        if actor is rival:
            raise ValueError('fresh separate seat actors required')
        env = scenarios.ScenarioEnvironment(world.scenario())
        equal(env.scenario, world.scenario(), 'physical world/seed binding')
        row['scenario_identity_sha256'] = env.scenario_identity['content_sha256']
        calls = [_callable(rival), _callable(rival)]; calls[seat] = _callable(actor)
        result = env.run([public_guard(call, seat=s, deadline=started + timeout)
                          for s, call in enumerate(calls)], record_actions=True)
        row['physical_games_completed'] = int(env.done)
        if result.steps != 719 or result.statuses != ('DONE', 'DONE'):
            raise ValueError('complete native 719-step game required')
        cash = list(result.rewards); margin = cash[seat] - cash[1-seat]
        if any(not math.isfinite(c) for c in cash):
            raise ValueError('finite terminal cash required')
        row.update(status='COMPLETE', cash=cash, margin=margin, wdl=_wdl(margin), steps=result.steps,
                   actions_sha256=digest(result.actions), min_rival_cash=cash[1-seat],
                   rival_cash_trough=min(result.cash_history[1-seat]),
                   actions_by_seat_sha256=[digest(tape) for tape in result.actions],
                   nonpass_turns_by_seat=[sum(a != exact_engine.PASS_ACTION for a in tape)
                                         for tape in result.actions])
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
        if env is not None:
            row['steps'] = env.step_index
            row['physical_games_completed'] = int(env.done)
            row['failure_step'] = env.step_index
    row['seconds'] = time.monotonic() - started
    return row


# Preserve the actual native implementation separately from the injectable name.
# A substituted runner is diagnostic, even if it copies every historical row.
_NATIVE_PLAY_GAME = play_game


def execution_binding(runner) -> dict:
    return bound({'kind': 'NATIVE_GAME' if runner is _NATIVE_PLAY_GAME else 'SYNTHETIC_DIAGNOSTIC',
                  'runner': _fingerprint(runner),
                  'trust': 'Source-bound trusted in-process runner; not a sandbox.'})


def _wdl(margin):
    return 'WIN' if margin > 0 else 'LOSS' if margin < 0 else 'DRAW'


def _counts(margins):
    margins = list(margins)
    return {'wins': sum(m > 0 for m in margins), 'draws': sum(m == 0 for m in margins),
            'losses': sum(m < 0 for m in margins)}


def validate_game(row: dict, job: dict, plan: dict, world: World, expected=None) -> dict:
    """Recheck identities and arithmetic even on resumed rows; preserve first divergence."""
    try:
        equal(row['evidence_kind'], plan['execution']['kind'], 'game evidence kind')
        equal(row['execution_sha256'], plan['execution']['content_sha256'], 'game execution binding')
        for key in ('world', 'seat', 'agent', 'opponent'):
            equal(row[key], job[key], 'game ' + key)
        equal(row['world_sha256'], world.sha256, 'game world_sha256')
        equal(row['seed'], world.seed, 'game seed')
        ids = {role: plan['actors'][job[role]]['content_sha256'] for role in ('agent', 'opponent')}
        by_seat = [ids['opponent'], ids['opponent']]; by_seat[job['seat']] = ids['agent']
        equal(row['actor_identity_hashes'], ids, 'game actor identities')
        equal(row['actor_identities_by_seat'], by_seat, 'game actor identities by seat')
        if row['status'] != 'COMPLETE':
            raise ValueError('incomplete game: ' + str(row.get('error')))
        if row['steps'] != 719 or row['error'] is not None:
            raise ValueError('complete native 719-step game with no error required')
        cash = row['cash']; seat = job['seat']
        if len(cash) != 2 or any(type(c) not in (int, float) or not math.isfinite(c) for c in cash):
            raise ValueError('finite terminal cash pair required')
        equal(row['margin'], float(cash[seat] - cash[1-seat]), 'game cash margin')
        equal(row['wdl'], _wdl(row['margin']), 'game WDL')
        equal(row['min_rival_cash'], cash[1-seat], 'game terminal rival cash')
        if not math.isfinite(row['rival_cash_trough']) or row['rival_cash_trough'] > cash[1-seat]:
            raise ValueError('game rival cash trough inconsistent')
        sha = row['actions_sha256']
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('game action SHA256 required')
        if expected is not None:
            actual = legacy_projection(row)
            differences = {k: {'expected': expected[k], 'actual': actual[k]}
                           for k in LEGACY_FIELDS if canonical(actual[k]) != canonical(expected[k])}
            if differences:
                return {'passed': False, 'error': 'retained panel first divergence', 'differences': differences}
        return {'passed': True, 'error': None}
    except (ValueError, TypeError, KeyError, IndexError) as error:
        return {'passed': False, 'error': f'{type(error).__name__}: {error}'}


def world_bootstrap(values, *, seed: int, samples: int) -> list[float]:
    values = list(values)
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('nonempty finite world means required')
    if type(seed) is not int or type(samples) is not int or samples < 100:
        raise ValueError('integer bootstrap seed and at least 100 samples required')
    rng = random.Random(seed); n = len(values)
    means = sorted(math.fsum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    def quantile(p):
        at = (samples - 1) * p; lo = int(at); fraction = at - lo
        return means[lo] + fraction * (means[min(lo + 1, samples - 1)] - means[lo])
    return [quantile(.025), quantile(.975)]


def summarize_pairs(plan: dict, records: list[dict]) -> dict:
    wanted = [j for j in jobs_for(plan) if j['phase'] == 'panel']
    registry = WorldRegistry(World(**spec) for spec in plan['registry']['worlds'])
    equal(registry.document(), plan['registry'], 'paired registry')
    by_job = {}
    for record in records:
        job = record['job']
        if job['phase'] != 'panel':
            continue
        key = digest(job)
        if key in by_job:
            raise ValueError('duplicate paired game')
        if record['row']['status'] != 'COMPLETE':
            raise ValueError('incomplete paired game')
        validation = validate_game(record['row'], job, plan, registry[job['world']])
        if not validation['passed']:
            raise ValueError('invalid paired game: ' + validation['error'])
        by_job[key] = record['row']
    if set(by_job) != {digest(j) for j in wanted}:
        raise ValueError('complete matching world/seat/role/rival pairs required')
    def row(world, rival, seat, role):
        return by_job[digest({'phase': 'panel', 'role': role, 'world': world, 'seat': seat,
                              'agent': plan[role], 'opponent': rival})]
    result = {}
    for rival in plan['rivals']:
        worlds = []; all_rows = []; candidate_margins = []; parent_margins = []; differences = []
        for spec in plan['registry']['worlds']:
            cand = [row(spec['id'], rival, s, 'candidate') for s in (0, 1)]
            parent = [row(spec['id'], rival, s, 'parent') for s in (0, 1)]
            cm = [r['margin'] for r in cand]; pm = [r['margin'] for r in parent]
            delta = [a-b for a, b in zip(cm, pm)]
            worlds.append({'world': spec['id'], 'cohort': spec['cohort'],
                           'candidate_raw_margins_by_seat': cm, 'parent_raw_margins_by_seat': pm,
                           'paired_margins_by_seat': delta, 'paired_margin': statistics.mean(delta)})
            candidate_margins.extend(cm); parent_margins.extend(pm); differences.extend(delta)
            all_rows.extend([*cand, *parent])
        means = [w['paired_margin'] for w in worlds]
        ci = world_bootstrap(means, seed=plan['bootstrap']['seed'], samples=plan['bootstrap']['samples'])
        result[rival] = {'worlds': worlds, 'paired_mean': statistics.mean(means),
                         'world_bootstrap_ci95': ci, 'bootstrap': plan['bootstrap'],
                         'worlds_count': len(worlds), 'games': len(all_rows),
                         'candidate_wdl': _counts(candidate_margins), 'parent_wdl': _counts(parent_margins),
                         'paired_wdl': _counts(differences), 'world_paired_wdl': _counts(means),
                         'worst_world': min(worlds, key=lambda w: (w['paired_margin'], w['world'])),
                         'min_rival_cash': min(r['min_rival_cash'] for r in all_rows),
                         'rival_cash_trough': min(r['rival_cash_trough'] for r in all_rows),
                         'identity_hashes': {name: plan['actors'][name]['content_sha256']
                                             for name in dict.fromkeys([plan['candidate'], plan['parent'], rival])},
                         'execution': plan['execution'],
                         'interpretation': ('SYNTHETIC_DIAGNOSTIC_NOT_PLAYING_EVIDENCE'
                             if plan['execution']['kind'] != 'NATIVE_GAME' else
                             # Same candidate and parent must give zero paired margins (duplicate return 88dfeb73).
                             'INVALID_IDENTICAL_PARENT_CONTROL'
                             if plan['candidate'] == plan['parent'] and any(differences) else
                             'POSITIVE_FIXED_WORLD_PAIRED_INTERVAL' if ci[0] > 0 and len(worlds) > 1
                             else 'NO_POSITIVE_FIXED_WORLD_PAIRED_INTERVAL'), 'scope': SCOPE}
    return result


def _safe_path(path: Path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlinked output or journal refused')


def _read_chain(path: Path, plan: dict, *, results: bool) -> list[dict]:
    _safe_path(path); file_sha(path)
    jobs = jobs_for(plan); records = []; previous = plan['content_sha256']
    with path.open() as stream:
        for index, line in enumerate(stream):
            try:
                record = json.loads(line)
                if canonical(record) + '\n' != line:
                    raise ValueError('noncanonical or truncated journal record')
                equal(record, bound(record), 'journal content hash')
                equal(record['previous'], previous, 'journal previous hash')
                equal(record['index'], index, 'journal index')
                if index >= len(jobs):
                    raise ValueError('journal beyond registered jobs')
                equal(record['job'], jobs[index], 'journal job prefix')
                if results and records and not records[-1]['validation']['passed']:
                    raise ValueError('journal continued after failure')
                if not results:
                    equal(sorted(record), sorted(['index', 'previous', 'job', 'event', 'content_sha256']),
                          'attempt fields')
                    equal(record['event'], 'STARTED', 'attempt event')
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f'journal record {index}: {error}') from error
            records.append(record); previous = record['content_sha256']
    return records


def read_attempts(path: Path, plan: dict) -> list[dict]:
    """Read all durable starts, including a possibly uncommitted final attempt."""
    return _read_chain(path, plan, results=False)


def read_journal(path: Path, plan: dict) -> list[dict]:
    records = _read_chain(path, plan, results=True)
    attempts = read_attempts(path.with_name('attempts.jsonl'), plan)
    if not len(records) <= len(attempts) <= len(records) + 1:
        raise ValueError('attempt/result prefix length mismatch')
    for record, attempt in zip(records, attempts):
        equal(record.get('attempt_sha256'), attempt['content_sha256'], 'result/attempt binding')
    return records


def _write_record(path: Path, record: dict):
    with path.open('a') as stream:
        stream.write(canonical(record) + '\n'); stream.flush(); os.fsync(stream.fileno())


def _start_attempt(path: Path, plan: dict, attempts: list[dict], job: dict) -> dict:
    record = bound({'index': len(attempts), 'previous': attempts[-1]['content_sha256'] if attempts
                    else plan['content_sha256'], 'job': job, 'event': 'STARTED'})
    # This fsync MUST finish before the runner (or any actor) may be called.
    _write_record(path, record)
    attempts.append(record)
    return record


def _append(path: Path, plan: dict, records: list[dict], job: dict, row: dict,
            validation: dict, attempt: dict):
    record = bound({'index': len(records), 'previous': records[-1]['content_sha256'] if records
                    else plan['content_sha256'], 'job': job, 'row': row, 'validation': validation,
                    'attempt_sha256': attempt['content_sha256']})
    _write_record(path, record)
    records.append(record)


def _report(plan, records, attempts):
    if len(records) != len(attempts):
        raise ValueError('cannot summarize uncommitted game attempts as completed evidence')
    native = plan['execution']['kind'] == 'NATIVE_GAME'
    failure = next((r for r in records if not r['validation']['passed']), None)
    legacy = [r for r in records if r['job']['phase'] == 'legacy']
    matched = sum(r['validation']['passed'] for r in legacy)
    complete = len(records) == plan['games_planned'] and failure is None
    rows = [r['row'] for r in records if r['row']['status'] == 'COMPLETE']
    status = 'FAILED' if failure else 'COMPLETE' if complete else 'PARTIAL_LEGACY' if matched < 80 else 'PARTIAL_PANEL'
    return bound({'schema': SCHEMA + '-summary', 'status': status, 'scope': SCOPE,
                  'manifest_sha256': plan['content_sha256'], 'journal_records': len(records),
                  'execution': plan['execution'],
                  'physical_games_attempted': len(attempts) if native else 0,
                  'synthetic_attempts': 0 if native else len(attempts),
                  'attempt_records': len(attempts),
                  'accounting_scope': 'Write-ahead attempts matched to committed results; unknown starts block resume.',
                  'physical_games_completed': (sum(r['row'].get('physical_games_completed',
                      int(r['row']['status'] == 'COMPLETE')) for r in records) if native else 0),
                  'legacy_gate': {'matched': matched, 'required': 80,
                                  'status': ('EXACT_80_ROW_REPLAY' if native else
                                      'SYNTHETIC_80_ROW_MATCH_NOT_NATIVE') if matched == 80 else 'NOT_PASSED'},
                  'identity_hashes': {n: a['content_sha256'] for n, a in plan['actors'].items()},
                  'min_rival_cash': min((r['min_rival_cash'] for r in rows), default=None),
                  'rival_cash_trough': min((r['rival_cash_trough'] for r in rows), default=None),
                  'paired': summarize_pairs(plan, records) if complete else None,
                  'failure': failure, 'games_planned': plan['games_planned'],
                  'limits': ['Fixed registered worlds, not prospective ladder sampling.',
                             'Paired cash margin is diagnostic; raw game WDL is reported separately.',
                             'Incomplete panels do not produce a confidence interval or omit missing seats.',
                             'SHA bindings detect drift, not malicious undeclared code dependencies.']})


def run_panel(output: Path, *, candidate='k0006', parent='k0006', max_games=4, resume=False,
              bootstrap_samples=10000, bootstrap_seed=20260917, timeout=300.) -> dict:
    if type(max_games) is not int or not 1 <= max_games <= 100000:
        raise ValueError('positive bounded max_games required')
    output = Path(output); _safe_path(output)
    plan, actors, registry = make_plan(candidate, parent, bootstrap_samples=bootstrap_samples,
                                      bootstrap_seed=bootstrap_seed, timeout=timeout)
    if resume:
        if not output.is_dir():
            raise ValueError('resume requires an existing run directory')
    else:
        output.mkdir(parents=True, exist_ok=False)
    for name in ('run.lock', 'manifest.json', 'attempts.jsonl', 'journal.jsonl', 'summary.json'):
        _safe_path(output / name)
    with (output / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('run is already locked by another writer') from error
        if (output / 'summary.json').exists():
            raise FileExistsError('existing final summary is never overwritten')
        journal = output / 'journal.jsonl'
        attempt_path = output / 'attempts.jsonl'
        if resume:
            equal(read(output / 'manifest.json'), plan, 'resume manifest (sources/actors/worlds/settings)')
            records = read_journal(journal, plan)
            attempts = read_attempts(attempt_path, plan)
            if len(attempts) != len(records):
                raise ValueError(f'uncommitted game attempt: {len(attempts)} started, '
                                 f'{len(records)} committed; completion UNKNOWN; no automatic replay')
        else:
            write_new(output / 'manifest.json', plan)
            for path in (journal, attempt_path):
                with path.open('x') as stream:
                    stream.flush(); os.fsync(stream.fileno())
            with (output / 'manifest.json').open('rb') as stream:
                os.fsync(stream.fileno())
            # Persist creation of the write-ahead log before the first start.
            directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            records = []; attempts = []
        oracle = legacy_oracle(); jobs = jobs_for(plan)
        for i, record in enumerate(records):
            expected = oracle['rows'][i] if i < 80 else None
            validation = validate_game(record['row'], record['job'], plan, registry[record['job']['world']], expected)
            equal(record['validation'], validation, 'resumed row validation')
        if records and not records[-1]['validation']['passed']:
            report = _report(plan, records, attempts)
        else:
            for job in jobs[len(records):len(records) + max_games]:
                # Prior rows are validated, source-bound exact oracle rows. There
                # is no --skip-regression, imported-result admission or zero-ID shortcut.
                if job['phase'] == 'panel' and sum(r['job']['phase'] == 'legacy' and r['validation']['passed']
                                                  for r in records) != 80:
                    raise ValueError('all 80 exact legacy rows required before new worlds')
                runner = play_game
                equal(execution_binding(runner), plan['execution'], 'runner identity before attempt')
                attempt = _start_attempt(attempt_path, plan, attempts, job)
                row = runner(registry[job['world']], actors[job['agent']], actors[job['opponent']],
                             job['seat'], timeout)
                row['evidence_kind'] = plan['execution']['kind']
                row['execution_sha256'] = plan['execution']['content_sha256']
                if row['evidence_kind'] != 'NATIVE_GAME':
                    row['physical_games_completed'] = 0
                expected = oracle['rows'][len(records)] if job['phase'] == 'legacy' else None
                validation = validate_game(row, job, plan, registry[job['world']], expected)
                _append(journal, plan, records, job, row, validation, attempt)
                if not validation['passed']:
                    break
            report = _report(plan, records, attempts)
        path = output / ('summary.json' if report['status'] in ('COMPLETE', 'FAILED')
                         else f'checkpoint-{len(records):06d}.json')
        write_new(path, report)
        return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', default='k0006')
    parser.add_argument('--parent', default='k0006')
    parser.add_argument('--max-games', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260917)
    parser.add_argument('--timeout', type=float, default=300.)
    args = parser.parse_args(argv)
    try:
        report = run_panel(**vars(args))
    except (ValueError, FileExistsError, OSError, ImportError, AttributeError) as error:
        print(f'REFUSED: {type(error).__name__}: {error}', file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    # 0 only for a COMPLETE panel; a partial run is not success (duplicate return d5a31c94).
    return 1 if report['status'] == 'FAILED' else 0 if report['status'] == 'COMPLETE' else 3


if __name__ == '__main__':
    raise SystemExit(main())
