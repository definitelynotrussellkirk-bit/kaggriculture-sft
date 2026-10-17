"""Exact, evaluator-only V0 dawn forks and an explicitly invoked bounded executor.

Public data path: make_continuation -> capture_dawn -> propose_dawn (optional)
-> fork_dawn -> restore_dawn. ``None`` is explicit KEEP at THIS dawn, without
sampling/advancing actor RNG; later dawns call the restored actor policy. A fork
only stages a proposal. RestoredDawn.actions publishes via PolicyAgent and the
real FullSeasonPlan, reruns the responsive opponent, and returns joint actions.
The codec never advances the engine. The separately invoked execute_recovery_branch
helper owns a bounded native continuation; the maintainer must authorize its use.

Snapshots are closed, finite, insertion-order-preserving JSON. V0Continuation
owns compiler/controller memory. This envelope adds the exact ScenarioEnvironment,
active/draft separation, both policy states, explicit RNG and frozen continuation.
A snapshot is accepted ONLY before observe/publication at a non-opening dawn.
It is not proof of learner reachability: provenance and a real reached-state
pilot remain maintainer gates. No opening-v1 record or training schema is widened.

Supported policies: existing KEEP/trajectory/target/ScorerPolicy actors; PASS and
frozen ShopRouter opponents; explicitly supplied external codecs. External codecs
must implement identity, snapshot_state()/restore_state() with EXACT fields
{memory: dict, rng: {kind, state}}. RNG kind is NONE or PYTHON_RANDOM_V3. The codec
must represent ALL behavior-affecting memory and use only its own RNG (no ambient
random state). Factories are trusted code, not imported from snapshot paths; they
must return fresh instances. Frozen model weights may be shared READ-ONLY, never
mutable sampler/controller memory. This module verifies declared codecs, not
arbitrary hidden state in third-party Python objects.

CLI verifies DATA only (no actor/model instantiation or native steps):
  python -m farm_neat.rollout.v0_regret_dawn verify --branch B.json \
      --continuation C.json --output NEW.json
"""
from __future__ import annotations

import argparse
import copy
from collections import deque
from dataclasses import dataclass
import inspect
import json
import math
from pathlib import Path
import platform
import random
import sys
import weakref

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.epochs import DAWNS, V0Agent
from farm_neat.evaluation import v0_policy as P
from farm_neat.evaluation.v0_continuation import V0Continuation
from farm_neat.evaluation.scenarios import ScenarioEnvironment
from farm_neat.evaluation.shop_router import ShopRouterAdapter
from farm_neat.evaluation.router_panel import PassActor
from farm_neat.game.exact_engine import AttrDict

SCHEMA = 'farm-v0-regret-dawn-v1'
PHASE = 'BEFORE_OBSERVE_AND_PUBLICATION'
ROOT = Path(__file__).resolve().parents[3]
NONE_RNG = {'kind': 'NONE', 'state': None}
_LIVE = weakref.WeakValueDictionary()
SNAPSHOT_FIELDS = {'schema', 'phase', 'step', 'seat', 'world', 'agent_type', 'active',
                   'actor_policy', 'opponent_policy', 'engine_rng', 'draft',
                   'continuation', 'source_pins', 'content_sha256'}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _safe(value, depth=0):
    _require(depth <= 80, 'bounded JSON depth required')
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _require(math.isfinite(value), 'finite JSON required')
        return
    if type(value) is list:
        for item in value:
            _safe(item, depth + 1)
        return
    if type(value) is dict:
        _require(all(type(k) is str for k in value), 'string JSON keys required')
        for item in value.values():
            _safe(item, depth + 1)
        return
    raise ValueError('data-only JSON required')


def _plain(value):
    """Normalize only live codec outputs; wire input is validated without repair."""
    result = json.loads(P.wire(value))
    _safe(result)
    return result


def _closed(value, fields, name):
    _require(type(value) is dict and set(value) == set(fields), 'closed ' + name + ' fields required')


def _same(actual, expected, name):
    _require(P.wire(actual) == P.wire(expected), name + ' mismatch')


def _bound(value, name):
    _safe(value)
    _require(type(value) is dict and 'content_sha256' in value, name + ' hash required')
    _same(value, P.bind(value), name + ' hash')


def _runtime():
    return {'implementation': sys.implementation.name, 'python': platform.python_version(),
            'random_state_version': random.Random.VERSION}


def source_pins() -> dict:
    paths = ('src/farm_neat/rollout/v0_regret_dawn.py',
             'src/farm_neat/evaluation/router_panel.py', 'src/farm_neat/evaluation/shop_router.py',
             'src/farm_neat/evaluation/opponents.py',
             'src/farm_neat/evaluation/shop_router_0909/main.py',
             'src/farm_neat/evaluation/shop_router_0909/actions.json',
             'project/configs/v0-regret-teacher.json')
    pins = {**P.source_pins(), **{p: P.file_sha(ROOT / p) for p in paths}}
    return dict(sorted(pins.items()))


def _type_identity(policy, declared):
    cls = type(policy)
    source = Path(inspect.getfile(cls)).resolve()
    _require(source.is_relative_to(ROOT), 'policy codec source must be in this source tree')
    return {'type': cls.__module__ + ':' + cls.__qualname__,
            'source': {'path': str(source.relative_to(ROOT)), 'sha256': P.file_sha(source)},
            'declared': _plain(declared)}


def _check_identity(value):
    _closed(value, ('type', 'source', 'declared'), 'policy identity')
    _require(type(value['type']) is str and value['type'], 'explicit policy type required')
    _closed(value['source'], ('path', 'sha256'), 'policy source')
    path = value['source']['path']
    _require(type(path) is str, 'policy source path required')
    resolved = (ROOT / path).resolve()
    _require(resolved.is_relative_to(ROOT) and resolved.is_file(), 'policy source unavailable')
    _same(value['source']['sha256'], P.file_sha(resolved), 'policy source')
    _require(type(value['declared']) is dict and value['declared'], 'explicit policy identity required')


def _tuple_tree(value):
    return tuple(_tuple_tree(v) for v in value) if type(value) is list else value


def _check_rng(value):
    _closed(value, ('kind', 'state'), 'RNG')
    kind, state = value['kind'], value['state']
    if kind == 'NONE':
        _require(state is None, 'deterministic policy RNG must be explicit None')
    elif kind == 'PYTHON_RANDOM_V3':
        _require(type(state) is list and len(state) == 3 and type(state[0]) is int and state[0] == 3,
                 'complete Python RNG state required')
        words = state[1]
        _require(type(words) is list and len(words) == 625 and
                 all(type(v) is int and 0 <= v < 2**32 for v in words[:-1]) and
                 type(words[-1]) is int and 0 <= words[-1] <= 624 and
                 (state[2] is None or type(state[2]) in (int, float)), 'invalid Python RNG state')
        rng = random.Random(0)
        rng.setstate(_tuple_tree(state))
        _same(_plain(rng.getstate()), state, 'Python RNG round trip')
    elif kind == 'TORCH_GENERATOR':
        _closed(state, ('device', 'bytes'), 'torch RNG state')
        _require(type(state['device']) is str and state['device'] and
                 type(state['bytes']) is list and len(state['bytes']) > 0 and
                 all(type(v) is int and 0 <= v <= 255 for v in state['bytes']),
                 'complete torch generator state required')
    else:
        raise ValueError('unsupported or missing RNG codec')


def _policy_payload(policy, *, opponent=False):
    if opponent and type(policy) is ShopRouterAdapter:
        kind, state, declared, rng = 'ROUTER', policy.snapshot(), policy.identity, NONE_RNG
    elif opponent and type(policy) is PassActor:
        kind, state, declared, rng = 'PASS', {}, policy.identity, NONE_RNG
    else:
        raw = _plain(P._policy_snapshot(policy))
        kind, state = raw['kind'], raw
        if kind == 'EXTERNAL':
            state, declared = raw['state'], raw['identity']
            _closed(state, ('memory', 'rng'), 'external policy state')
            _require(type(state['memory']) is dict, 'explicit policy memory required')
            rng = state['rng']
        elif kind == 'SCORER' and not opponent:
            declared = {k: v for k, v in raw['state'].items() if k != 'generator_state'}
            rng = {'kind': 'TORCH_GENERATOR', 'state': {
                'device': raw['state']['device'], 'bytes': raw['state']['generator_state']}}
        else:
            _require(not opponent and kind in ('KEEP', 'TRAJECTORY', 'TARGET'),
                     'responsive opponent codec required; no future action tape')
            declared, rng = raw, NONE_RNG
    result = _plain({'kind': kind, 'identity': _type_identity(policy, declared),
                     'state': state, 'rng': rng})
    _check_policy(result, opponent=opponent)
    return result


def _check_policy(value, *, opponent=False):
    _closed(value, ('kind', 'identity', 'state', 'rng'), 'policy envelope')
    _check_identity(value['identity']); _check_rng(value['rng'])
    kind, state = value['kind'], value['state']
    _require(kind in (('ROUTER', 'PASS', 'EXTERNAL') if opponent else
                     ('KEEP', 'TRAJECTORY', 'TARGET', 'SCORER', 'EXTERNAL')), 'unsupported policy kind')
    if kind == 'EXTERNAL':
        _closed(state, ('memory', 'rng'), 'external policy state')
        _require(type(state['memory']) is dict, 'policy memory required')
        _check_rng(state['rng'])
        _require(state['rng']['kind'] != 'TORCH_GENERATOR', 'use ScorerPolicy for torch RNG')
        _same(state['rng'], value['rng'], 'policy RNG binding')
    elif kind == 'SCORER':
        _closed(state, ('kind', 'state'), 'scorer envelope')
        _closed(state['state'], ('identity', 'mode', 'seed', 'temperature', 'device', 'generator_state'),
                'scorer state')
        _same(state['kind'], kind, 'scorer kind')
        _same(value['identity']['declared'],
              {k: v for k, v in state['state'].items() if k != 'generator_state'}, 'scorer identity')
        _same(value['rng'], {'kind': 'TORCH_GENERATOR', 'state': {
            'device': state['state']['device'], 'bytes': state['state']['generator_state']}}, 'scorer RNG')
    elif kind == 'ROUTER':
        _closed(state, ('controller', 'identity', 'players'), 'router memory')
        _same(state['identity'], value['identity']['declared']['content_sha256'], 'router identity')
        _check_router_players(state['players'])
        _same(value['rng'], NONE_RNG, 'router RNG')
    elif kind == 'PASS':
        _same(state, {}, 'PASS memory'); _same(value['rng'], NONE_RNG, 'PASS RNG')
    else:
        _same(state['kind'], kind, 'actor kind')
        _same(state, value['identity']['declared'], 'actor identity')
        restored = P._restore_policy(state)
        _same(_plain(P._policy_snapshot(restored)), state, 'actor state round trip')
        _same(value['rng'], NONE_RNG, 'deterministic actor RNG')


def _check_router_players(players):
    _require(type(players) is dict and set(players) <= {'0', '1'}, 'router player rows required')
    fields = {'plan', 'last_step', 'day', 'queues', 'sale_due_step', 'advanced_sales'}
    for row in players.values():
        _closed(row, fields, 'router player state')
        _require(type(row['plan']) is int and 0 <= row['plan'] < 13 and
                 type(row['last_step']) is int and -1 <= row['last_step'] <= V.LAST_STEP and
                 type(row['day']) is int and -1 <= row['day'] <= 29 and
                 type(row['sale_due_step']) is int and -1 <= row['sale_due_step'] <= V.LAST_STEP,
                 'invalid router clock/plan memory')
        _require(type(row['queues']) is dict and all(
            type(k) is str and k.isdecimal() and str(int(k)) == k and
            type(v) is list and all(type(a) is list for a in v) for k, v in row['queues'].items()),
            'router worker queues required')
        _require(type(row['advanced_sales']) is dict and all(
            type(k) is str and type(v) is int and v >= 0 for k, v in row['advanced_sales'].items()),
            'router advanced-sales memory required')


def _remember(obj, *, fresh=False):
    if fresh:
        _require(_LIVE.get(id(obj)) is not obj, 'fresh policy instance required; live instance reuse refused')
    try:
        _LIVE[id(obj)] = obj
    except TypeError as exc:
        raise ValueError('policy codec instances must support weak references') from exc


def _restore_policy(value, *, opponent=False, factory=None):
    _check_policy(value, opponent=opponent)
    kind = value['kind']
    if factory is not None:
        _require(callable(factory), 'policy factory must be callable')
        policy = factory()
        _remember(policy, fresh=True)  # before restore can mutate a supplied object
        _same(_policy_payload(policy, opponent=opponent)['identity'], value['identity'],
              'supplied policy identity')
        if kind == 'EXTERNAL':
            # A fresh wrapper may still close over a shared mutable RNG. Detach
            # its instance graph BEFORE restore_state could mutate that RNG.
            # Ambient/module state and dishonest __deepcopy__ codecs remain
            # unsupported; trusted factories must own all behavior-affecting state.
            supplied = policy
            policy = copy.deepcopy(supplied)
            _require(policy is not supplied and type(policy) is type(supplied),
                     'external codec must support detached deep copy')
            _remember(policy, fresh=True)
        elif kind == 'SCORER':
            # Frozen weights may be shared; mutable sampler state may not.
            _require(not any(obj is not policy and isinstance(obj, P.ScorerPolicy) and
                             obj.generator is policy.generator for obj in list(_LIVE.values())),
                     'shared live torch generator refused; fresh sampler required')
    elif kind == 'ROUTER':
        policy = ShopRouterAdapter()
    elif kind == 'PASS':
        policy = PassActor()
    elif kind in ('KEEP', 'TARGET', 'TRAJECTORY'):
        policy = P._restore_policy(value['state'])
    else:
        raise ValueError('matching fresh external/scorer policy factory required')
    _remember(policy)
    if kind == 'ROUTER':
        # Rehydrate the actual frozen policy's DayState, NOT a copied future tape.
        state_type = type(policy.actor).act.__globals__['DayState']
        policy.actor.players = {}
        for seat, row in value['state']['players'].items():
            state = state_type()
            for name, item in row.items():
                setattr(state, name, {int(k): deque(copy.deepcopy(v)) for k, v in item.items()}
                        if name == 'queues' else copy.deepcopy(item))
            policy.actor.players[int(seat)] = state
    elif kind == 'EXTERNAL':
        policy.restore_state(copy.deepcopy(value['state']))
    elif kind == 'SCORER':
        policy.restore_state(copy.deepcopy(value['state']['state']))
    elif factory is not None and kind != 'PASS':
        # Built-in value policies are reconstructed by their existing owning codec.
        policy = P._restore_policy(value['state'])
        _remember(policy)
    _same(_policy_payload(policy, opponent=opponent), value, 'exact policy restore')
    return policy


def make_continuation(agent: V0Agent, opponent, *, seat: int) -> dict:
    """Freeze future policy identities, not seeds or policy-Q assumptions."""
    _require(type(seat) is int and seat in (0, 1), 'explicit seat required')
    actor, rival = _policy_payload(agent.policy), _policy_payload(opponent, opponent=True)
    _remember(agent.policy); _remember(opponent)
    return P.bind({'schema': SCHEMA + '-continuation', 'seat': seat,
        'actor_identity': actor['identity'], 'opponent_identity': rival['identity'],
        'future_actor': 'RESTORED_POLICY_AT_DAWNS',
        'future_opponent': 'RESPONSIVE_POLICY_EACH_OBSERVATION',
        'publication_steps': list(DAWNS), 'runtime': _runtime()})


def _check_continuation(value):
    _bound(value, 'continuation')
    _closed(value, ('schema', 'seat', 'actor_identity', 'opponent_identity', 'future_actor',
                    'future_opponent', 'publication_steps', 'runtime', 'content_sha256'), 'continuation')
    _require(value['schema'] == SCHEMA + '-continuation' and type(value['seat']) is int and
             value['seat'] in (0, 1), 'continuation schema/seat required')
    _same(value['future_actor'], 'RESTORED_POLICY_AT_DAWNS', 'future actor law')
    _same(value['future_opponent'], 'RESPONSIVE_POLICY_EACH_OBSERVATION', 'responsive opponent law')
    _same(value['publication_steps'], list(DAWNS), 'publication law')
    _same(value['runtime'], _runtime(), 'runtime/RNG implementation')
    _check_identity(value['actor_identity']); _check_identity(value['opponent_identity'])


def _world_data(env):
    _require(type(env) is ScenarioEnvironment and set(vars(env)) == {
        'scenario', 'engine', 'configuration', 'info', 'done', 'step_index', 'state', 'scenario_identity'},
        'exact ScenarioEnvironment fields required; custom world state is unsupported')
    return _plain({'schema': SCHEMA + '-world', 'scenario': env.scenario,
        'scenario_identity': env.scenario_identity, 'configuration': env.configuration,
        'info': env.info, 'step_index': env.step_index, 'done': env.done, 'state': env.state})


def _check_native_values(value, template, name):
    """Fixed dictionary fields/types from the pinned engine's own constructors."""
    _closed(value, template, name)
    for key, control in template.items():
        expected = (int, float) if type(control) is float else (type(control),)
        _require(type(value[key]) in expected, name + ' value type mismatch: ' + key)


def _check_tile(tile, engine):
    if tile is None or tile == 'LOCKED':
        return
    _require(type(tile) is dict and type(tile.get('kind')) is str, 'complete native tile required')
    if tile['kind'] == 'PLANT':
        _require(tile.get('crop') in engine.CROPS, 'native crop identity required')
        _check_native_values(tile, engine._new_plant(tile['crop'], 0, 24), 'plant state')
    elif 'animal' in tile:
        _require(tile['animal'] in engine.ANIMALS, 'native animal identity required')
        template = engine._new_animal(tile['animal'], 0)
        _check_native_values(tile, template, 'animal state')
        _same(tile['kind'], template['kind'], 'animal structure')
    else:
        structures = {v['structure'] for v in engine.ANIMALS.values()}
        _require(tile['kind'] in structures | {'WEED'}, 'native empty structure or weed required')
        _closed(tile, ('kind',), 'empty native tile')


def _check_world_contents(obs, template, env):
    """Reject partial nested state, without claiming engine reachability."""
    engine = env.engine
    private = obs['private']
    for name in ('seeds', 'shed'):
        _check_native_values(private[name], template['private'][name], 'private ' + name)
        _require(all(v >= 0 for v in private[name].values()), 'negative private inventory')
    inventory = private['inventories']
    own = obs['farms'][obs['player']]
    _require(type(inventory) is list and len(inventory) == len(own['hands']) + 1,
             'complete per-worker inventories required')
    for bag in inventory:
        _require(type(bag) is dict and set(bag) <= set(template['private']['shed']) and
                 all(type(v) is int and v >= 0 for v in bag.values()), 'invalid worker inventory')
    for name in ('prices', 'inventory'):
        _check_native_values(obs['market'][name], template['market'][name], 'market ' + name)
    if 'params' in template['market']:
        _same(obs['market']['params'], template['market']['params'], 'market parameter identity')
    shops = obs['town']['unlocked_shops']
    _require(type(shops) is list and all(type(s) is str and s in engine.SHOPS for s in shops),
             'complete legal town shops required')
    _same(shops, env.scenario['shops'][:min(8, obs['day'] // 3)], 'fixed scenario shop prefix')
    def position(pos):
        return type(pos) is list and len(pos) == 2 and all(type(n) is int and 0 <= n < 10 for n in pos)
    for farm in obs['farms']:
        _require(position(farm['farmer']) and type(farm['hands']) is list and
                 all(position(hand) for hand in farm['hands']), 'complete worker positions required')
        _require(type(farm['money']) in (int, float) and type(farm['hires_today']) is int and
                 farm['hires_today'] >= 0, 'native cash/hire state required')
        quadrants = farm['unlocked_quadrants']
        _require(type(quadrants) is list and all(type(q) is str for q in quadrants) and
                 set(quadrants) <= {'NW', 'NE', 'SW', 'SE'} and len(set(quadrants)) == len(quadrants),
                 'native quadrant state required')
        for row in farm['tiles']:
            for tile in row:
                _check_tile(tile, engine)


def _restore_world(world):
    _closed(world, ('schema', 'scenario', 'scenario_identity', 'configuration', 'info',
                    'step_index', 'done', 'state'), 'world')
    _require(world['schema'] == SCHEMA + '-world' and type(world['step_index']) is int and
             world['step_index'] in DAWNS[1:] and world['done'] is False, 'legal nonterminal dawn required')
    # Initialization constructs a schema/control but executes ZERO native turns.
    env = ScenarioEnvironment(copy.deepcopy(world['scenario']))
    _same(_plain(env.scenario), world['scenario'], 'scenario')
    _same(_plain(env.scenario_identity), world['scenario_identity'], 'scenario source identity')
    _same(_plain(env.configuration), world['configuration'], 'world configuration')
    _same(_plain(env.info), world['info'], 'world seed/RNG identity')
    rows = world['state']
    _require(type(rows) is list and len(rows) == 2, 'both full world seats required')
    for seat, row in enumerate(rows):
        _closed(row, ('observation', 'action', 'reward', 'status', 'info'), 'world seat')
        obs, template = row['observation'], env.state[seat].observation
        _closed(obs, template, 'world observation')
        _require(type(obs['player']) is int and obs['player'] == seat and row['status'] == 'ACTIVE',
                 'world seat/status mismatch')
        _require(type(obs['day']) is int and obs['day'] == world['step_index'] // 24 and
                 type(obs['hour']) is int and obs['hour'] == 0, 'world dawn clock mismatch')
        if seat == 0:
            _require(type(obs['step']) is int and obs['step'] == world['step_index'], 'world step mismatch')
        _closed(obs['private'], template['private'], 'private world observation')
        _closed(obs['private']['seeds'], template['private']['seeds'], 'private seed inventory')
        _require(type(obs['private']['inventories']) is list and type(obs['private']['shed']) is dict,
                 'complete private inventories required')
        _closed(obs['market'], template['market'], 'market state')
        _closed(obs['town'], template['town'], 'town state')
        _require(type(obs['farms']) is list and len(obs['farms']) == 2, 'both public farms required')
        for farm, base in zip(obs['farms'], template['farms']):
            _closed(farm, base, 'farm state')
            tiles = farm['tiles']
            _require(type(tiles) is list and len(tiles) == 10 and
                     all(type(r) is list and len(r) == 10 for r in tiles), 'complete tile grid required')
        _check_world_contents(obs, template, env)
    for key in ('farms', 'market', 'town'):
        _same(rows[1]['observation'][key], rows[0]['observation'][key], 'shared public world state')
    env.state = [AttrDict({**copy.deepcopy(row), 'observation': AttrDict(copy.deepcopy(row['observation']))})
                 for row in rows]
    # Preserve the engine's shared public objects as well as their ordered bytes.
    for key in ('farms', 'market', 'town'):
        env.state[1].observation[key] = env.state[0].observation[key]
    env.step_index, env.done = world['step_index'], world['done']
    env.configuration, env.info = AttrDict(copy.deepcopy(world['configuration'])), copy.deepcopy(world['info'])
    _same(_world_data(env), world, 'exact world restore')
    return env


def _engine_rng(world):
    # Source-pinned _end_of_day constructs a new local RNG at each day key. There
    # is NO live engine generator between steps; explicit None is not lost state.
    return {'kind': 'DAY_KEYED_PYTHON_RANDOM', 'seed': world['info']['seed'],
            'key_recipe': '(seed * 1000003) ^ zero_based_day',
            'next_end_of_day': world['step_index'] // 24, 'persistent_state': None,
            'draw_alignment': 'ACTION_DEPENDENT_EMPTY_TILES; NOT_CLAIMED', 'runtime': _runtime()}


def _agent_memory(agent, observation):
    _require(type(agent) in (V0Agent, P.PolicyAgent) and set(vars(agent)) == P.AGENT_FIELDS,
             'closed V0Agent fields required')
    memory = _plain(V0Continuation.capture(agent).to_dict())
    restored = V0Continuation.from_dict(memory).restore(None, agent_cls=type(agent), observation=observation)
    P._options(restored.projection_kw())
    _require(restored.hands_mode in ('daily', 'constant'), 'explicit hands mode required')
    holds = {p for p in range(len(V.PRODUCTS)) if V.PRIMITIVES[restored.plan.overlay(p)[0]] == 'HOLD'}
    _require(set(restored.hold_releases) == holds, 'every retained HOLD requires its original binding')
    _same(_plain(V0Continuation.capture(restored).to_dict()), memory, 'exact V0 memory restore')
    return memory


def _decision(agent, obs, draft):
    _closed(draft, ('raw_actions', 'canonical_plan'), 'draft')
    _require(type(draft['raw_actions']) is list, 'explicit raw action list required')
    checked = P.validate_edits(agent.actor_state(obs, initialize=False), agent.plan, draft['raw_actions'],
                               projection_kwargs=agent.projection_kw())
    _same(draft['canonical_plan'], checked.plan.canonical(), 'canonical successor Plan')
    return checked


def capture_dawn(env: ScenarioEnvironment, agent: V0Agent, opponent, *, continuation: dict,
                 draft: dict | None = None) -> dict:
    """Read-only capture before either seat acts at a legal non-opening dawn.

    ``draft`` is None or {raw_actions, canonical_plan}; it is NOT active, and is
    never implicitly selected by KEEP. Diagnostics in agent.records are excluded.
    """
    _check_continuation(continuation)
    seat = continuation['seat']
    step = P._boundary(agent, env.agent_observation(seat))
    result = P.bind({'schema': SCHEMA + '-snapshot', 'phase': PHASE, 'step': step, 'seat': seat,
        'world': _world_data(env), 'agent_type': type(agent).__name__,
        'active': _agent_memory(agent, env.agent_observation(seat)),
        'actor_policy': _policy_payload(agent.policy), 'opponent_policy': _policy_payload(opponent, opponent=True),
        'engine_rng': _engine_rng(_world_data(env)), 'draft': copy.deepcopy(draft),
        'continuation': copy.deepcopy(continuation), 'source_pins': source_pins()})
    validate_snapshot(result, continuation=continuation)
    _remember(agent.policy); _remember(opponent)
    return copy.deepcopy(result)


def validate_snapshot(snapshot: dict, *, continuation: dict) -> tuple[ScenarioEnvironment, V0Agent]:
    """Validate closed data/source/law joins; never instantiate an external policy."""
    _bound(snapshot, 'snapshot'); _closed(snapshot, SNAPSHOT_FIELDS, 'snapshot')
    _require(snapshot['schema'] == SCHEMA + '-snapshot' and snapshot['phase'] == PHASE,
             'snapshot schema/pre-publication phase required')
    _same(snapshot['source_pins'], source_pins(), 'snapshot source pins')
    _check_continuation(continuation)
    _same(snapshot['continuation'], continuation, 'frozen continuation')
    _require(type(snapshot['seat']) is int and snapshot['seat'] == continuation['seat'], 'seat mismatch')
    env = _restore_world(snapshot['world'])
    _require(type(snapshot['step']) is int and snapshot['step'] == env.step_index, 'snapshot step mismatch')
    _same(snapshot['engine_rng'], _engine_rng(snapshot['world']), 'engine RNG state')
    types = {'V0Agent': V0Agent, 'PolicyAgent': P.PolicyAgent}
    _require(snapshot['agent_type'] in types, 'unsupported agent type')
    observation = env.agent_observation(snapshot['seat'])
    agent = V0Continuation.from_dict(snapshot['active']).restore(
        None, agent_cls=types[snapshot['agent_type']], observation=observation)
    _same(_agent_memory(agent, observation), snapshot['active'], 'active memory')
    P._boundary(agent, env.agent_observation(snapshot['seat']))
    for key, opposite in (('actor_policy', False), ('opponent_policy', True)):
        _check_policy(snapshot[key], opponent=opposite)
        _same(snapshot[key]['identity'], continuation['opponent_identity' if opposite else 'actor_identity'],
              key + ' continuation identity')
    if snapshot['draft'] is not None:
        _decision(agent, env.agent_observation(snapshot['seat']), snapshot['draft'])
    return env, agent


def propose_dawn(snapshot: dict, raw_actions: list[int], *, continuation: dict) -> dict:
    """Verify the supplied own-prefix IDs. No sampling, search or target repair."""
    env, agent = validate_snapshot(snapshot, continuation=continuation)
    _require(type(raw_actions) is list, 'explicit raw action list required')
    result = P.validate_edits(agent.actor_state(env.agent_observation(snapshot['seat']), initialize=False), agent.plan, raw_actions,
                              projection_kwargs=agent.projection_kw())
    return P.bind({'schema': SCHEMA + '-proposal', 'snapshot_sha256': snapshot['content_sha256'],
                   'raw_actions': list(result.actions), 'canonical_plan': result.plan.canonical()})


def fork_dawn(snapshot: dict, proposal: dict | None, *, continuation: dict) -> dict:
    """Detached, fully bound branch. None means KEEP, not publish stored draft.

    Does not observe, compile, publish, sample a policy or advance a native step.
    All active memory and both policy RNGs remain identical to the captured cut.
    """
    validate_snapshot(snapshot, continuation=continuation)
    if proposal is None:
        selected = propose_dawn(snapshot, [V.STOP_ID], continuation=continuation)
    else:
        _bound(proposal, 'proposal')
        _closed(proposal, ('schema', 'snapshot_sha256', 'raw_actions', 'canonical_plan', 'content_sha256'),
                'proposal')
        selected = propose_dawn(snapshot, proposal['raw_actions'], continuation=continuation)
        _same(proposal, selected, 'source-bound proposal')
    return P.bind({'schema': SCHEMA + '-fork', 'snapshot': copy.deepcopy(snapshot),
                   'proposal': copy.deepcopy(selected), 'continuation': copy.deepcopy(continuation)})


def verify_fork(branch: dict, *, continuation: dict) -> dict:
    _bound(branch, 'fork')
    _closed(branch, ('schema', 'snapshot', 'proposal', 'continuation', 'content_sha256'), 'fork')
    _same(branch['continuation'], continuation, 'fork continuation')
    expected = fork_dawn(branch['snapshot'], branch['proposal'], continuation=continuation)
    _same(branch, expected, 'fork')
    return copy.deepcopy(branch)


def continuation_hashes(env, agent, opponent, *, continuation: dict, draft=None) -> dict:
    """Exact ordered runtime state hashes, not Plan/behavior/value equivalence."""
    world = _world_data(env)
    values = {'world': world, 'active': _plain(V0Continuation.capture(agent).to_dict()),
              'actor_policy': _policy_payload(agent.policy),
              'opponent_policy': _policy_payload(opponent, opponent=True),
              'engine_rng': _engine_rng(world), 'draft': draft, 'continuation': continuation}
    hashes = {key + '_sha256': P.wire_hash(value) for key, value in values.items()}
    return {**hashes, 'state_sha256': P.wire_hash(hashes)}


@dataclass
class RestoredDawn:
    """Live branch; actions() is a one-call-per-step controller operation, not a game.

    The root proposal is consumed exactly once. A failure poisons this instance;
    restore a fresh branch instead of retrying partially advanced policy memory.
    """
    env: ScenarioEnvironment
    agent: V0Agent
    other: object
    continuation: dict
    draft: dict | None
    proposal: dict
    _next_step: int
    _root: bool = True
    _failed: bool = False

    def hashes(self) -> dict:
        return continuation_hashes(self.env, self.agent, self.other,
                                   continuation=self.continuation, draft=self.draft)

    def actions(self) -> list[dict]:
        _require(not self._failed and not self.env.done and self.env.step_index == self._next_step,
                 'actions require next unconsumed step exactly once; restore after failure')
        seat = self.continuation['seat']
        policy = self.agent.policy
        try:
            if self._root:
                # Same real validator/compiler/publication path as every PolicyAgent.
                self.agent.policy = lambda *_: list(self.proposal['raw_actions'])
            own = self.agent(self.env.agent_observation(seat), copy.deepcopy(self.env.configuration))
            if self._root:
                _same(self.agent.plan.canonical(), self.proposal['canonical_plan'], 'published Plan')
            call = self.other if callable(self.other) else self.other.act
            rival = call(self.env.agent_observation(1 - seat), copy.deepcopy(self.env.configuration))
        except Exception:
            self._failed = True
            raise
        finally:
            self.agent.policy = policy
        self._root = False
        self._next_step += 1
        joint = [None, None]
        joint[seat], joint[1 - seat] = copy.deepcopy(own), copy.deepcopy(rival)
        return joint


def restore_dawn(branch: dict, *, continuation: dict, actor_factory=None,
                 opponent_factory=None) -> RestoredDawn:
    """Instantiate only trusted supplied factories or existing registered built-ins."""
    branch = verify_fork(branch, continuation=continuation)
    snapshot = branch['snapshot']
    env, _ = validate_snapshot(snapshot, continuation=continuation)
    policy = _restore_policy(snapshot['actor_policy'], factory=actor_factory)
    other = _restore_policy(snapshot['opponent_policy'], opponent=True, factory=opponent_factory)
    cls = {'V0Agent': V0Agent, 'PolicyAgent': P.PolicyAgent}[snapshot['agent_type']]
    agent = V0Continuation.from_dict(snapshot['active']).restore(
        policy, agent_cls=cls, observation=env.agent_observation(snapshot['seat']))
    runtime = RestoredDawn(env, agent, other, copy.deepcopy(continuation),
                           copy.deepcopy(snapshot['draft']), copy.deepcopy(branch['proposal']), env.step_index)
    recaptured = capture_dawn(env, agent, other, continuation=continuation, draft=runtime.draft)
    _same(recaptured, snapshot, 'full dawn restore')
    return runtime


def recovery_actor_context(snapshot: dict) -> dict:
    """Own/public projection facts plus execution-relevant OWN memory only.

    This is deliberately not an opening-v1 actor record. Started and bound HOLD
    rows remain ordered; the full controller, policy RNG, engine RNG, future shop
    schedule, rival private state, seed and provenance stay evaluator-only.
    Projection facts have the existing native scorer/record allowlist. They are
    not a sufficiency claim: exact execution always restores the whole snapshot.
    """
    from farm_neat.training import v0_regret_records as records
    from farm_neat.policy.v0_scorer import Prefix
    env, agent = validate_snapshot(snapshot, continuation=snapshot['continuation'])
    state = agent.actor_state(env.agent_observation(snapshot['seat']), initialize=False)
    facts = {key: _plain(getattr(state, key)) for key in sorted(records.OBS_FIELDS)}
    memory = {key: copy.deepcopy(snapshot['active'][key]) for key in (
        'schema', 'land_commitment', 'plan', 'allocation', 'started', 'hold_releases', 'labor_factor', 'hands_mode', 'belief')}
    return dict(schema='farm-v0-regret-recovery-actor-v2', observation=facts,
                prefix=Prefix(agent.plan).payload(), compiler_memory=memory)


def execute_recovery_branch(request: dict, *, actor_factory=None, opponent_factory=None,
                            max_steps: int | None = None) -> dict:
    """Explicit single-start native evaluator for ``search_recovery``.

    No default search calls this function. A maintainer must supply it as the
    trusted evaluator and register NATIVE_GAME scope. Fresh instances and exact
    restore are mandatory on EVERY call, including replays. ``max_steps`` is an
    optional stricter test/maintenance cap, never a fake terminal value. Partial
    runs return CENSORED with null WDL/cash. Caller owns process isolation; the
    searcher supplies a main-thread deadline, not a sandbox for external code.
    """
    import time
    _bound(request, 'recovery request')
    _closed(request, ('schema', 'branch', 'phase', 'role', 'attempt_id', 'candidate_id',
        'source_group_id', 'source_family_id', 'protocol_sha256', 'selection_sha256',
        'evaluator_sha256', 'evidence_scope', 'timeout_seconds', 'content_sha256'), 'recovery request')
    _require(request['schema'] == 'farm-v0-regret-recovery-branch-request-v1', 'recovery request schema')
    _require(request['evidence_scope'] == 'NATIVE_GAME', 'native scope required')
    _require(type(request['timeout_seconds']) in (int, float) and
             math.isfinite(request['timeout_seconds']) and request['timeout_seconds'] > 0, 'positive timeout required')
    _require(max_steps is None or type(max_steps) is int and max_steps >= 0, 'nonnegative step cap required')
    branch = request['branch']
    live = restore_dawn(branch, continuation=branch['continuation'],
                        actor_factory=actor_factory, opponent_factory=opponent_factory)
    started, cpu = time.monotonic(), time.process_time()
    tape, status, error = [], 'CENSORED', None
    try:
        while not live.env.done:
            if time.monotonic() - started >= request['timeout_seconds'] or (
                    max_steps is not None and len(tape) >= max_steps):
                error = 'BRANCH_BUDGET'
                break
            actions = live.actions()
            tape.append(copy.deepcopy(actions))
            live.env.step(actions)
        if live.env.done:
            _require(live.env.step_index == V.LAST_STEP + 1, 'unexpected terminal clock')
            _require(all(row.status == 'DONE' for row in live.env.state), 'incomplete terminal tuple')
            status = 'COMPLETE'
    except Exception as exc:
        # A partially advanced instance is never used again. No failed run has WDL.
        status, error = 'FAILED', type(exc).__name__ + ': ' + str(exc)
    cash = [float(row.reward) for row in live.env.state] if status == 'COMPLETE' else None
    seat = branch['snapshot']['seat']
    wdl = (int(cash[seat] > cash[1-seat]) - int(cash[seat] < cash[1-seat])) if cash is not None else None
    return P.bind(dict(schema='farm-v0-regret-recovery-branch-result-v1',
        request_sha256=request['content_sha256'], attempt_id=request['attempt_id'],
        initial_snapshot_sha256=branch['snapshot']['content_sha256'],
        proposal_sha256=branch['proposal']['content_sha256'], status=status,
        terminal_wdl=wdl, cash=cash, actions_sha256=P.wire_hash(tape),
        terminal_runtime_sha256=live.hashes()['state_sha256'], simulator_starts=1,
        simulator_steps=len(tape), physical_starts=1, evidence_scope='NATIVE_GAME', error=error,
        cpu_seconds=time.process_time()-cpu, elapsed_seconds=time.monotonic()-started))


def _read(path):
    path = Path(path)
    _require(path.stat().st_size <= 16_000_000, 'bounded input JSON required')
    def pairs(rows):
        result = {}
        for key, value in rows:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    with path.open() as stream:
        value = json.load(stream, object_pairs_hook=pairs)
    _safe(value)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('verify', help='source/restore-structure checks only; zero native steps')
    verify.add_argument('--branch', required=True)
    verify.add_argument('--continuation', required=True)
    verify.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        branch = verify_fork(_read(args.branch), continuation=_read(args.continuation))
        receipt = P.write_new(args.output, {'schema': SCHEMA + '-verification', 'status': 'VERIFIED_DATA_ONLY',
            'branch_sha256': branch['content_sha256'], 'snapshot_sha256': branch['snapshot']['content_sha256'],
            'continuation_sha256': branch['continuation']['content_sha256'], 'source_pins': source_pins(),
            'native_steps': 0, 'training_admission': False,
            'scope': 'DATA_AND_OWNED_MEMORY_CHECKS_ONLY; NO_EXTERNAL_CODEC_EXECUTION_OR_REACHABILITY_PROOF'})
        print(P.wire(receipt))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
