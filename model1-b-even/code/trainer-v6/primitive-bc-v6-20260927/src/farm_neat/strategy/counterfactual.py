"""Source-bound, evaluator-only counterfactual suffixes; no search or policy scoring.

capture(environment, [actor0, actor1], seat=0) captures BETWEEN joint turns,
BEFORE either next actor call (and before V0 observe/publication). restore returns
independent game and policy objects; evaluate(snapshot, proposal, continuation)
returns the shared SuffixResult. None means the unchanged captured continuation.

Supported game adapters: exact engine and ScenarioEnvironment. Supported actors:
unmodified K0006Adapter, Program, ShopRouterAdapter, and base V0Agent with Keep,
Trajectory, Random, or static TargetPolicy. Program retains the complete live
FullSeasonPlan graph; ShopRouter reloads verified immutable tapes and restores
selected plan, ordered weed deques and pending advanced-sale deductions. Unsupported code/state fails closed. V0 memory uses v0_oracle's
existing codec, adding policy/RNG and diagnostic records. K0006's source is never
edited: its nested namespaces, closure cells, function attributes and instance
state are rebuilt against newly executed frozen source. Object aliases, tuple/int
keys and dictionary insertion order survive JSON transport. No pickle, dynamic
imports, serialized code, or source text from a snapshot is executed.

A continuation object exposes adopt(actor, context, proposal) -> Adoption. It can
change ONLY the restored own actor, and receives actor-visible context, never the
snapshot, engine, seed, rival memory, or future shops. This is a trusted Python
extension boundary, not an adversarial sandbox. It handles ANY Proposal payload;
the evaluator does not enumerate economic families. The default adapter supports
one current raw-action override and a strict one-dawn V0 edit sequence. Custom
adopters own guard checks and optionally verify the requested effect after the
first transition. Unsupported/declined/failed adoption yields PROPOSAL_FAILED
with no cash, not a zero or a baseline score. Unfinished suffixes yield ERROR.

Raw-action override calls the original actor once before replacing its action,
so its memory advances normally and subsequent calls see actual observations.
Emission is not a fill/feasibility certificate. Realization reports requested and
emitted actions, first-transition cash changes and V0 compiler refusals; a custom
verify callback can fail unmet effects. Terminal cash/WDL are evaluator labels,
not actor estimates. compare returns MEASURED deltas, never general exact values.

Example (environment and actors are live, before their next joint action)::

    cut = capture(environment, actors, seat=0)
    control = evaluate(cut, None, max_steps=719, seconds=120)
    child = evaluate(cut, proposal, my_continuation, max_steps=719, seconds=120)
    deltas = compare(control, child)

Launch capture and fresh replay with PYTHONHASHSEED=0 (or the same explicit
seed). Python version, implementation and a string-hash fingerprint are bound:
a different hash seed is refused rather than silently changing set iteration.

The time limit is cooperative; use an external process timeout for untrusted or
blocking policies. No globals/RNG outside the declared codecs are supported.
The optional trace list is evaluator-only and retains every joint action and
both post-turn cash totals in order. This module never admits labels or selects
winners. Fixed-world suffix comparisons are conditional, not ladder strength.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
import random
import sys
import time
import types as pytypes
from typing import Any, Callable, Mapping, Sequence

from farm_neat.economy.orders import FullSeasonPlan, ScheduledEconomicOrders
from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.epochs import KeepPolicy, RandomPolicy, TargetPolicy, TrajectoryPolicy, V0Agent
from farm_neat.economy.v0.plan import Plan
from farm_neat.evaluation.k0006 import K0006Adapter
from farm_neat.evaluation.scenarios import Program, ScenarioEnvironment
from farm_neat.evaluation.shop_router import ShopRouterAdapter
from farm_neat.game.exact_engine import (AttrDict, DEFAULT_CONFIGURATION,
    ExactKaggricultureEnvironment, assert_bundled_engine_identity, load_engine)
from farm_neat.strategy.types import Estimate, Proposal, StrategyContext, SuffixResult, UNKNOWN
from farm_neat.training.teachers import v0_oracle

SCHEMA = 'farm-strategy-counterfactual-v1'
PHASE = 'BEFORE_BOTH_ACTORS_AND_OBSERVE'
ROOT = Path(__file__).resolve().parents[3]
_V0_FIELDS = frozenset(('policy', 'labor_factor', 'hands_mode', 'belief', 'plan', 'allocation',
                        'started', 'hold_releases', 'controller', 'game_id', 'records'))


def _json(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def source_pins() -> dict[str, str]:
    """Current implementation closure, not arbitrary paths chosen by a snapshot."""
    files = list((ROOT / 'src/farm_neat/economy').rglob('*.py'))
    files += [ROOT / p for p in (
        'src/farm_neat/economy/v0/modules.json', 'src/farm_neat/game/constants.py',
        'src/farm_neat/game/exact_engine.py', 'src/farm_neat/game/kaggriculture_engine_reference.py',
        'src/farm_neat/game/kaggriculture.json', 'src/farm_neat/evaluation/scenarios.py',
        'src/farm_neat/evaluation/common.py', 'src/farm_neat/evaluation/k0006.py',
        'src/farm_neat/evaluation/shop_router.py', 'src/farm_neat/evaluation/opponents.py',
        'src/farm_neat/evaluation/shop_router_0909/main.py',
        'src/farm_neat/evaluation/shop_router_0909/actions.json',
        'src/farm_neat/evaluation/shop_router_0909/LICENSE.txt',
        'src/farm_neat/evaluation/shop_router_0909/PROVENANCE.json',
        'src/farm_neat/evaluation/k0006_v43/main.py',
        'src/farm_neat/evaluation/k0006_v43/PROVENANCE.json',
        'src/farm_neat/strategy/types.py', 'src/farm_neat/strategy/counterfactual.py',
        'src/farm_neat/training/teachers/v0_oracle.py',
        'src/farm_neat/training/teachers/v0_plan_search.py',
        'src/farm_neat/training/v0_targets.py')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files))}


def _runtime_profile():
    return {'version': list(sys.version_info[:3]), 'implementation': sys.implementation.name,
            'string_hash': hash('farm-strategy-counterfactual-v1:continuation-order')}


def _k_registry(actor):
    """Source-owned symbols only, resolved from a fresh frozen namespace, not imports."""
    namespaces = {}
    def visit(ns, path):
        if any(ns is old for old in namespaces.values()):
            return
        namespaces[path] = ns
        for name, value in sorted(ns.items()):
            if name != '__builtins__' and isinstance(value, dict) and '__builtins__' in value:
                visit(value, path + '/' + name)
    visit(actor.actor.__globals__, 'root')
    symbols = {'builtin:AttrDict': AttrDict, **{'builtin:' + cls.__name__: cls
               for cls in (type(None), bool, bytes, float, int, str)}}
    local_names = {ns.get('__name__') for ns in namespaces.values()}
    functions, classes = [], []
    seen = set()
    def symbol(value, path):
        if id(value) in seen:
            return
        seen.add(id(value)); symbols[path] = value
        if isinstance(value, pytypes.FunctionType) and value.__globals__.get('__name__') in local_names:
            functions.append(value)
        if isinstance(value, type) and value.__module__ in local_names:
            classes.append(value)
            for name, member in sorted(vars(value).items()):
                if isinstance(member, (pytypes.FunctionType, staticmethod, classmethod)):
                    fn = member.__func__ if isinstance(member, (staticmethod, classmethod)) else member
                    symbol(fn, path + '/' + name)
    for path, ns in namespaces.items():
        symbols['namespace:' + path] = ns
        for name, value in sorted(ns.items()):
            if name == '__builtins__':
                continue
            if (isinstance(value, (pytypes.FunctionType, pytypes.BuiltinFunctionType, pytypes.ModuleType, type))
                    or type(value) is object or type(value).__module__ == '__future__'):
                symbol(value, path + ':' + name)
    return symbols, functions, classes


class _Graph:
    """Small data graph for these codecs. Executable objects must be source anchors."""
    def __init__(self, symbols=None):
        self.symbols = symbols or {'builtin:AttrDict': AttrDict}
        self.names = {id(v): k for k, v in reversed(list(self.symbols.items()))}
        self.nodes, self.memo = [], {}

    def pack(self, value):
        root = self._put(value)
        return {'root': root, 'nodes': self.nodes}

    def _put(self, value):
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError('nonfinite continuation data')
            return value
        symbol = self.names.get(id(value))
        if symbol and not isinstance(value, (dict, pytypes.FunctionType)):
            return {'symbol': symbol}
        key = id(value)
        if key in self.memo:
            return {'ref': self.memo[key]}
        index = len(self.nodes); self.memo[key] = index; self.nodes.append(None)
        node = {}
        if isinstance(value, dict):
            node['kind'] = 'namespace' if symbol and symbol.startswith('namespace:') else 'dict'
            if node['kind'] == 'namespace':
                node['symbol'] = symbol
            elif type(value) is not dict:
                cls = self.names.get(id(type(value)))
                if cls is None:
                    raise TypeError('unsupported dict subclass: ' + type(value).__name__)
                node['class'] = cls
                node['attrs'] = self._put(vars(value)) if hasattr(value, '__dict__') else None
            node['items'] = [[self._put(k), self._put(v)] for k, v in value.items()
                             if not (node['kind'] == 'namespace' and k == '__builtins__')]
        elif type(value) in (list, tuple, set, frozenset):
            node['kind'] = type(value).__name__
            # Current native actors use scalar/tuple sets. Stable order, no hash-seed dependence.
            values = sorted(value, key=lambda v: (type(v).__name__, repr(v))) if isinstance(value, (set, frozenset)) else value
            node['items'] = [self._put(v) for v in values]
        elif type(value) is bytes:
            node = {'kind': 'bytes', 'hex': value.hex()}
        elif isinstance(value, pytypes.FunctionType) and symbol:
            node = {'kind': 'function', 'symbol': symbol, 'attrs': self._put(vars(value)),
                    'defaults': self._put(value.__defaults__), 'kwdefaults': self._put(value.__kwdefaults__),
                    'closure': [self._put(c.cell_contents) for c in value.__closure__ or ()]}
        elif type(value) is random.Random:
            node = {'kind': 'random', 'state': self._put(value.getstate())}
        elif id(type(value)) in self.names and hasattr(value, '__dict__'):
            if getattr(type(value), '__slots__', None):
                raise TypeError('unsupported slotted continuation instance')
            node = {'kind': 'instance', 'class': self.names[id(type(value))], 'attrs': self._put(vars(value))}
        else:
            raise TypeError('unsupported continuation object: ' + type(value).__name__)
        self.nodes[index] = node
        return {'ref': index}

    def unpack(self, data):
        if not isinstance(data, dict) or set(data) != {'root', 'nodes'} or not isinstance(data['nodes'], list):
            raise ValueError('closed data graph required')
        nodes = data['nodes']; memo = {}; filling = set(); finished = set()
        def symbol(name):
            if name not in self.symbols:
                raise ValueError('unknown source symbol: ' + str(name))
            return self.symbols[name]
        def get(item):
            if item is None or type(item) in (bool, int, float, str):
                if type(item) is float and not math.isfinite(item):
                    raise ValueError('nonfinite graph value')
                return item
            if not isinstance(item, dict):
                raise ValueError('invalid graph reference')
            if set(item) == {'symbol'}:
                return symbol(item['symbol'])
            if set(item) != {'ref'} or type(item['ref']) is not int or not 0 <= item['ref'] < len(nodes):
                raise ValueError('invalid graph index')
            i = item['ref']
            if i in finished or i in filling and i in memo:
                return memo[i]
            if i in filling:
                raise ValueError('unsupported cycle through immutable tuple')
            node = nodes[i]; kind = node['kind']; filling.add(i)
            if kind == 'namespace':
                obj = symbol(node['symbol'])
                builtins = obj.get('__builtins__'); obj.clear(); obj['__builtins__'] = builtins
                memo[i] = obj
            elif kind == 'dict':
                cls = symbol(node['class']) if 'class' in node else dict
                if not isinstance(cls, type) or not issubclass(cls, dict):
                    raise ValueError('source dict class required')
                obj = dict.__new__(cls); memo[i] = obj
            elif kind == 'list':
                obj = []; memo[i] = obj
            elif kind == 'set':
                obj = set(); memo[i] = obj
            elif kind == 'instance':
                cls = symbol(node['class'])
                obj = object.__new__(cls); memo[i] = obj
            elif kind == 'function':
                obj = symbol(node['symbol'])
                if not isinstance(obj, pytypes.FunctionType):
                    raise ValueError('source function required')
                memo[i] = obj
            elif kind == 'random':
                obj = random.Random(); memo[i] = obj
            elif kind == 'bytes':
                obj = bytes.fromhex(node['hex']); memo[i] = obj
            elif kind in ('tuple', 'frozenset'):
                obj = (tuple if kind == 'tuple' else frozenset)(get(v) for v in node['items']); memo[i] = obj
            else:
                raise ValueError('unsupported graph kind: ' + str(kind))
            if kind in ('dict', 'namespace'):
                for k, v in node['items']:
                    key = get(k)
                    if key in obj:
                        raise ValueError('duplicate restored graph key')
                    obj[key] = get(v)
                if node.get('attrs') is not None:
                    object.__setattr__(obj, '__dict__', get(node['attrs']))
            elif kind == 'list':
                obj.extend(get(v) for v in node['items'])
            elif kind == 'set':
                obj.update(get(v) for v in node['items'])
            elif kind == 'instance':
                object.__setattr__(obj, '__dict__', get(node['attrs']))
            elif kind == 'function':
                object.__setattr__(obj, '__dict__', get(node['attrs'])); obj.__defaults__ = get(node['defaults'])
                obj.__kwdefaults__ = get(node['kwdefaults'])
                closure = obj.__closure__ or ()
                if len(closure) != len(node['closure']):
                    raise ValueError('source closure shape changed')
                for cell, value in zip(closure, node['closure']):
                    cell.cell_contents = get(value)
            elif kind == 'random':
                obj.setstate(get(node['state']))
            finished.add(i); filling.remove(i)
            return obj
        result = get(data['root'])
        if len(finished) != len(nodes):
            raise ValueError('unreachable graph nodes')
        return result


def _pack(value):
    return _Graph().pack(value)


def _unpack(value):
    return _Graph().unpack(value)


def snapshot_environment(env) -> dict:
    if type(env) not in (ExactKaggricultureEnvironment, ScenarioEnvironment):
        raise TypeError('unsupported environment adapter')
    assert_bundled_engine_identity()
    if env.engine is not load_engine():
        raise ValueError('substituted engine refused')
    return {'kind': type(env).__name__, 'data': _pack({k: v for k, v in vars(env).items() if k != 'engine'})}


def restore_environment(memory: dict):
    classes = {c.__name__: c for c in (ExactKaggricultureEnvironment, ScenarioEnvironment)}
    if set(memory) != {'kind', 'data'} or memory['kind'] not in classes:
        raise ValueError('unsupported environment snapshot')
    assert_bundled_engine_identity()
    env = object.__new__(classes[memory['kind']])
    env.__dict__.update(_unpack(memory['data'])); env.engine = load_engine()
    expected = {'configuration', 'info', 'done', 'step_index', 'state', 'engine'}
    if type(env) is ScenarioEnvironment:
        expected |= {'scenario', 'scenario_identity'}
    if set(vars(env)) != expected or type(env.step_index) is not int or not 0 <= env.step_index <= 719:
        raise ValueError('unsupported environment fields or phase')
    for key in ('farms', 'market', 'town'):
        if env.state[0].observation[key] is not env.state[1].observation[key]:
            raise ValueError('lost shared native observation state: ' + key)
    return env


def _snapshot_policy(policy):
    if type(policy) is KeepPolicy and not vars(policy):
        return {'kind': 'KEEP'}
    if type(policy) is TrajectoryPolicy and set(vars(policy)) == {'schedule'}:
        return {'kind': 'TRAJECTORY', 'data': _pack(policy.schedule)}
    if type(policy) is RandomPolicy and set(vars(policy)) == {'rng', 'stop', 'max_edits'}:
        return {'kind': 'RANDOM', 'data': _pack(vars(policy))}
    if type(policy) is TargetPolicy and type(policy.target) is Plan and set(vars(policy)) == {'target', 'residuals'}:
        return {'kind': 'TARGET', 'plan': policy.target.canonical(), 'residuals': _pack(policy.residuals)}
    raise TypeError('unsupported V0 policy; supply an explicit source-bound continuation codec')


def _restore_policy(memory):
    kind = memory.get('kind')
    if kind == 'KEEP' and set(memory) == {'kind'}:
        return KeepPolicy()
    if kind == 'TRAJECTORY' and set(memory) == {'kind', 'data'}:
        return TrajectoryPolicy(_unpack(memory['data']))
    if kind == 'RANDOM' and set(memory) == {'kind', 'data'}:
        policy = RandomPolicy(); policy.__dict__ = _unpack(memory['data']); return policy
    if kind == 'TARGET' and set(memory) == {'kind', 'plan', 'residuals'}:
        policy = TargetPolicy(Plan.from_canonical(memory['plan']))
        policy.residuals = _unpack(memory['residuals']); return policy
    raise ValueError('unsupported V0 policy snapshot')


_PROGRAM_FIELDS = frozenset(('program', 'controller', 'opening_refusals', 'revisions',
                             'capture', 'records', 'publications', 'identity'))
_PROGRAM_SYMBOLS = {'builtin:AttrDict': AttrDict, 'controller:FullSeasonPlan': FullSeasonPlan,
                    'controller:ScheduledEconomicOrders': ScheduledEconomicOrders}
_ROUTER_STATE_FIELDS = frozenset(('plan', 'last_step', 'day', 'queues', 'sale_due_step', 'advanced_sales'))


def _validate_program_actor(actor):
    """Validate against the owning controller codec, but retain the original graph.

    FullSeasonPlan.snapshot is the semantic validator, not a replacement for its
    live graph: active stage aliases, pending receipts and diagnostic aliases must
    survive even a JSON writer that sorts object keys.
    """
    if type(actor) is not Program or set(vars(actor)) != _PROGRAM_FIELDS:
        raise ValueError('unsupported Program fields')
    if not isinstance(actor.identity, dict) or 'command_budget' not in actor.identity:
        raise ValueError('invalid Program identity')
    original = Program(actor.program, actor.identity['command_budget'],
                       revisions=actor.revisions, capture=actor.capture)
    if actor.identity != original.identity or actor.program != original.program:
        raise ValueError('Program identity/program mismatch')
    if any(type(getattr(actor, key)) is not list for key in ('opening_refusals', 'records', 'publications')):
        raise ValueError('unsupported Program diagnostic fields')
    controller = actor.controller
    if controller is not None:
        if (type(controller) is not FullSeasonPlan
                or set(vars(controller)) != set(vars(FullSeasonPlan('codec-validation')))
                or type(controller.executor) is not ScheduledEconomicOrders
                or set(vars(controller.executor)) != set(vars(ScheduledEconomicOrders()))):
            raise ValueError('unsupported Program controller fields')
        validated = FullSeasonPlan.restore(controller.snapshot())
        if controller.orders != validated.orders or controller.rules != validated.rules:
            raise ValueError('Program active controller stages disagree')


def _router_types(actor):
    # Loader selects only the pinned source path; no snapshot controls imports.
    from farm_neat.evaluation import opponents
    from farm_neat.evaluation.shop_router import FOLDER, identity
    namespace = opponents._tape_module_agent(FOLDER / 'main.py').__globals__
    if (type(actor) is not ShopRouterAdapter or set(vars(actor)) != {'identity', 'actor'}
            or type(actor.actor) is not namespace['Policy']
            or set(vars(actor.actor)) != {'tapes', 'players'}):
        raise ValueError('unsupported ShopRouter fields')
    if actor.identity != identity():
        raise ValueError('frozen ShopRouter identity changed')
    # Tapes are source data, not continuation state. Refuse edits rather than
    # silently replacing a modified tape with the source on restore.
    if _hash(actor.actor.tapes) != _hash(json.loads((FOLDER / 'actions.json').read_text())):
        raise ValueError('frozen ShopRouter tape changed')
    return namespace['DayState']


def _router_memory(actor):
    state_class = _router_types(actor)
    if type(actor.actor.players) is not dict:
        raise ValueError('unsupported ShopRouter players')
    rows = []
    seen = set()
    for seat, state in actor.actor.players.items():
        if type(seat) is not int or seat not in (0, 1) or type(state) is not state_class:
            raise ValueError('unsupported ShopRouter player state')
        if set(vars(state)) != _ROUTER_STATE_FIELDS:
            raise ValueError('unsupported ShopRouter state fields')
        if (type(state.plan) is not int or not 0 <= state.plan < 13
                or type(state.last_step) is not int or not -1 <= state.last_step <= 718
                or type(state.day) is not int or not -1 <= state.day <= 29
                or type(state.sale_due_step) is not int or not -1 <= state.sale_due_step <= 718
                or type(state.queues) is not dict or type(state.advanced_sales) is not dict):
            raise ValueError('unsupported ShopRouter state values')
        if any(type(k) is not str or type(v) is not int or v < 0
               for k, v in state.advanced_sales.items()):
            raise ValueError('unsupported ShopRouter advanced sales')
        queues = []
        for worker, queue in state.queues.items():
            if type(worker) is not int or worker < 0 or type(queue) is not deque or queue.maxlen is not None:
                raise ValueError('unsupported ShopRouter queue')
            if any(type(action) is not list for action in queue):
                raise ValueError('unsupported ShopRouter queued action')
            # The frozen source never shares a deque or a DayState between seats.
            if id(queue) in seen:
                raise ValueError('unsupported shared ShopRouter queue')
            seen.add(id(queue))
            queues.append([worker, list(queue)])
        if id(state) in seen:
            raise ValueError('unsupported shared ShopRouter state')
        seen.add(id(state))
        rows.append([seat, {**vars(state), 'queues': queues}])
    return {'identity': actor.identity, 'players': rows}


def _new_actor_snapshot(kind, data):
    # Standalone new actor snapshots have the same source/runtime and corruption
    # guards as the full evaluator envelope. K0006/V0 wire formats are untouched.
    body = {'kind': kind, 'source_pins': source_pins(), 'python': _runtime_profile(), 'data': data}
    return {**body, 'sha256': hashlib.sha256(json.dumps(body, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()}


def _validate_new_actor_snapshot(memory):
    if set(memory) != {'kind', 'source_pins', 'python', 'data', 'sha256'}:
        raise ValueError('closed actor snapshot fields required')
    body = {key: memory[key] for key in ('kind', 'source_pins', 'python', 'data')}
    if body['source_pins'] != source_pins() or body['python'] != _runtime_profile():
        raise ValueError('actor snapshot source pins or Python/hash-seed runtime changed')
    expected = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if memory['sha256'] != expected:
        raise ValueError('actor snapshot hash binding mismatch')


def _restore_new_actor(memory):
    _validate_new_actor_snapshot(memory)
    if memory['kind'] == 'PROGRAM':
        actor = object.__new__(Program)
        actor.__dict__ = _Graph(_PROGRAM_SYMBOLS).unpack(memory['data'])
        _validate_program_actor(actor)
        return actor
    actor = ShopRouterAdapter()
    state_class = _router_types(actor)
    data = _unpack(memory['data'])
    if type(data) is not dict or set(data) != {'identity', 'players'} or data['identity'] != actor.identity:
        raise ValueError('invalid ShopRouter snapshot identity/fields')
    if type(data['players']) is not list:
        raise ValueError('invalid ShopRouter snapshot players')
    for seat, row in data['players']:
        if (type(seat) is not int or seat not in (0, 1) or seat in actor.actor.players
                or type(row) is not dict or set(row) != _ROUTER_STATE_FIELDS):
            raise ValueError('invalid ShopRouter snapshot state fields')
        queues = {}
        for worker, values in row['queues']:
            if type(worker) is not int or worker < 0 or worker in queues or type(values) is not list:
                raise ValueError('invalid ShopRouter snapshot queue')
            queues[worker] = deque(values)
        state = state_class()
        state.__dict__ = {**row, 'queues': queues}
        actor.actor.players[seat] = state
    _router_memory(actor)  # Closed live-state validation, including pending sales.
    return actor


def snapshot_actor(actor) -> dict:
    if type(actor) is K0006Adapter and set(vars(actor)) == {'identity', 'actor'}:
        if actor.actor is not actor.actor.__globals__.get('agent'):
            raise ValueError('frozen K0006 entrypoint replaced')
        symbols, functions, classes = _k_registry(actor)
        # Class methods are source anchored too: retain any mutable defaults/attributes.
        class_data = [(cls, {k: v for k, v in vars(cls).items() if not k.startswith('__')
                            and not isinstance(v, (pytypes.FunctionType, staticmethod, classmethod, property))})
                      for cls in classes]
        graph = _Graph(symbols).pack({'actor': vars(actor), 'namespace': actor.actor.__globals__,
                                     'functions': functions, 'class_data': class_data})
        return {'kind': 'K0006', 'identity': copy.deepcopy(actor.identity), 'data': graph}
    if type(actor) is V0Agent:
        if set(vars(actor)) != _V0_FIELDS:
            raise ValueError('unsupported V0Agent fields; no lossy continuation')
        return {'kind': 'V0', 'memory': _pack(v0_oracle.snapshot_agent(actor)),
                'policy': _snapshot_policy(actor.policy), 'records': _pack(actor.records)}
    if type(actor) is Program:
        _validate_program_actor(actor)
        return _new_actor_snapshot('PROGRAM', _Graph(_PROGRAM_SYMBOLS).pack(vars(actor)))
    if type(actor) is ShopRouterAdapter:
        return _new_actor_snapshot('SHOP_ROUTER', _pack(_router_memory(actor)))
    raise TypeError('unsupported actor; expected frozen K0006Adapter, base V0Agent, Program or ShopRouterAdapter')


def restore_actor(memory: dict):
    if not isinstance(memory, dict):
        raise ValueError('unsupported actor snapshot')
    if memory.get('kind') in ('PROGRAM', 'SHOP_ROUTER'):
        return _restore_new_actor(memory)
    if memory.get('kind') == 'K0006' and set(memory) == {'kind', 'identity', 'data'}:
        actor = K0006Adapter()
        if actor.identity != memory['identity']:
            raise ValueError('frozen K0006 identity changed')
        symbols, _, _ = _k_registry(actor)
        value = _Graph(symbols).unpack(memory['data'])
        actor.__dict__ = value['actor']
        if actor.actor.__globals__ is not value['namespace']:
            raise ValueError('K0006 global namespace alias lost')
        for cls, data in value['class_data']:
            for key, val in data.items():
                setattr(cls, key, val)
        return actor
    if memory.get('kind') == 'V0' and set(memory) == {'kind', 'memory', 'policy', 'records'}:
        actor = v0_oracle.restore_agent(_unpack(memory['memory']), _restore_policy(memory['policy']))
        actor.records = _unpack(memory['records'])
        return actor
    raise ValueError('unsupported actor snapshot')


@dataclass(frozen=True)
class Snapshot:
    """Immutable JSON bytes; evaluator-only, never accepted by StrategyContext."""
    payload: str

    @classmethod
    def from_dict(cls, value: Mapping):
        fields = {'schema', 'phase', 'seat', 'python', 'source_pins', 'environment', 'actors', 'sha256'}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError('closed snapshot fields required')
        body = {k: v for k, v in value.items() if k != 'sha256'}
        # Graph dict ordering is carried by pair arrays; envelope key order is immaterial.
        expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if value['sha256'] != expected:
            raise ValueError('snapshot hash binding mismatch')
        if value['schema'] != SCHEMA or value['phase'] != PHASE:
            raise ValueError('snapshot schema/phase mismatch')
        if type(value['seat']) is not int or value['seat'] not in (0, 1) or len(value['actors']) != 2:
            raise ValueError('two actors and one seat required')
        return cls(_json(value))

    def to_dict(self):
        return json.loads(self.payload)

    @property
    def sha256(self):
        return self.to_dict()['sha256']


@dataclass
class Runtime:
    environment: Any
    actors: list
    seat: int


def capture(environment, actors: Sequence, *, seat: int = 0) -> Snapshot:
    if len(actors) != 2 or actors[0] is actors[1]:
        raise ValueError('two separate seat actors required')
    if environment.done:
        raise ValueError('capture a nonterminal pre-action boundary')
    body = {'schema': SCHEMA, 'phase': PHASE, 'seat': seat,
            'python': _runtime_profile(), 'source_pins': source_pins(),
            'environment': snapshot_environment(environment), 'actors': [snapshot_actor(a) for a in actors]}
    sha = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return Snapshot.from_dict({**body, 'sha256': sha})


def restore(snapshot: Snapshot) -> Runtime:
    if not isinstance(snapshot, Snapshot):
        raise TypeError('source-bound Snapshot required')
    data = Snapshot.from_dict(snapshot.to_dict()).to_dict()
    if data['source_pins'] != source_pins() or data['python'] != _runtime_profile():
        raise ValueError('source pins or Python/hash-seed runtime changed; exact replay refused')
    env = restore_environment(data['environment'])
    if env.done:
        raise ValueError('nonterminal snapshot required')
    return Runtime(env, [restore_actor(a) for a in data['actors']], data['seat'])


def actor_context(observation, actor, *, derived=None) -> StrategyContext:
    """No evaluator envelope is passed; repeated published shops keep their order and multiplicity."""
    obs = copy.deepcopy(dict(observation))
    memory = {}
    if type(actor) is V0Agent:
        own = v0_oracle.snapshot_agent(actor)
        memory = {k: copy.deepcopy(own[k]) for k in ('plan', 'allocation', 'started', 'hold_releases',
                                                   'labor_factor', 'hands_mode', 'belief')}
    return StrategyContext(obs['step'], obs['player'], obs, memory,
                           tuple(obs.get('town', {}).get('unlocked_shops', ())), derived or {})


class ProposalFailed(ValueError):
    """A proposed adoption or required realization failed, not a measured payoff."""


@dataclass
class Adoption:
    """Own-actor intervention, with optional first-transition effect validation.

    first_action receives ONLY (own actor, own observation, public configuration).
    verify receives (pre-action StrategyContext, copied post-action own observation),
    returning None on success or a failure reason. It also works after step 718.
    """
    first_action: Callable | None = None
    realization: dict = field(default_factory=dict)
    verify: Callable | None = None


class DefaultContinuation:
    def adopt(self, actor, context: StrategyContext, proposal: Proposal) -> Adoption:
        candidate = proposal.candidate
        if not isinstance(candidate, dict):
            raise ProposalFailed('default continuation requires a typed candidate object')
        if proposal.guards:
            raise ProposalFailed('text guards require a consumer-supplied adopter; never evaluated as code')
        req = proposal.requirements
        if req.inventory or any(getattr(req, k).known for k in
                ('cash', 'reserve', 'market_slots', 'worker_hours', 'land_quadrants')):
            raise ProposalFailed('resource requirements need a consumer-supplied adopter to certify them')
        window = req.window
        if window is not None and not window[0] <= context.step <= window[1]:
            raise ProposalFailed('proposal outside its declared execution window')
        if candidate.get('kind') == 'action' and set(candidate) == {'kind', 'action'}:
            action = copy.deepcopy(candidate['action'])
            if (not isinstance(action, dict) or set(action) != {'farmer', 'hands', 'market'}
                    or not all(isinstance(action[k], list) for k in action)
                    or not action['farmer']):
                raise ProposalFailed('raw action requires farmer/hands/market arrays')
            def first(own, obs, config):
                _act(own, obs, config)
                return copy.deepcopy(action)
            return Adoption(first_action=first, realization={'kind': 'action', 'requested': action,
                            'certificate': 'EMITTED_ONLY; NOT_A_FILL_OR_FEASIBILITY_CERTIFICATE'})
        if candidate.get('kind') == 'v0_edits' and set(candidate) == {'kind', 'actions'}:
            if type(actor) is not V0Agent or context.step % 24:
                raise ProposalFailed('v0_edits require a base V0Agent at dawn')
            actions = copy.deepcopy(candidate['actions']); report = {'kind': 'v0_edits', 'requested': actions}
            def first(own, obs, config):
                prior = own.policy; witness = {}
                def decision(state, plan, masks):
                    # V0Agent has now observed the current state. Prefix checks are exact.
                    trace = v0_oracle.trace_edits(v0_oracle.capture_dawn(own, obs), actions)
                    witness.update(trace)
                    return list(trace['actions'])
                own.policy = decision
                try:
                    action = _act(own, obs, config)
                finally:
                    own.policy = prior
                if not witness or own.plan.canonical() != witness['plan']:
                    raise ProposalFailed('executed Plan differs from prefix-validated proposal')
                if v0_oracle.snapshot_agent(own)['hold_releases'] != witness['hold_releases']:
                    raise ProposalFailed('executed HOLD bindings differ from validated continuation')
                report.update(trace_sha256=witness['content_sha256'], plan=own.plan.canonical(),
                              compiler_rejected=own.records[-1]['rejected'], unplaced=own.records[-1]['unplaced'])
                if report['compiler_rejected']:
                    raise ProposalFailed('compiler/executor refused proposed commands')
                return action
            return Adoption(first_action=first, realization=report)
        raise ProposalFailed('unsupported proposal kind; supply its module adopter')


def _act(actor, observation, configuration):
    public = {k: copy.deepcopy(v) for k, v in configuration.items() if k in DEFAULT_CONFIGURATION}
    if type(observation.get('player')) is not int or observation['player'] not in (0, 1):
        raise ValueError('actor observation seat mismatch')
    call = actor.act if callable(getattr(actor, 'act', None)) else actor
    return call(copy.deepcopy(observation), public)


def evaluate(snapshot: Snapshot, proposal: Proposal | None, continuation_policy=None, *,
             max_steps: int = 719, seconds: float = 120., trace: list | None = None) -> SuffixResult:
    """Evaluate one independent branch. All exceptions/censoring carry no terminal cash."""
    key = proposal.key if isinstance(proposal, Proposal) else None
    report = {'schema': SCHEMA + '-realization', 'steps': 0}
    phase = 'RESTORE'
    started = time.monotonic()
    try:
        if proposal is not None and not isinstance(proposal, Proposal):
            raise ValueError('Proposal or None required')
        if type(max_steps) is not int or max_steps < 1 or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('positive explicit step/time budget required')
        runtime = restore(snapshot)
        report['snapshot_sha256'] = getattr(snapshot, 'sha256', None)
        env, actors, seat = runtime.environment, runtime.actors, runtime.seat
        start = env.step_index
        report.update(start_step=start, seat=seat)
        adoption = Adoption(); before = None
        if proposal is not None:
            # Context failure is an evaluator/interface ERROR, not a false proposal payoff.
            before = actor_context(env.agent_observation(seat), actors[seat])
            phase = 'ADOPT'
            adoption = (continuation_policy or DefaultContinuation()).adopt(actors[seat], before, Proposal.from_dict(proposal.to_dict()))
            if not isinstance(adoption, Adoption):
                raise ProposalFailed('adopter must explicitly return Adoption')
            _json(adoption.realization)
        tape_hash = hashlib.sha256(); cash0 = [f['money'] for f in env.agent_observation(seat)['farms']]
        while not env.done:
            phase = 'SUFFIX'
            if report['steps'] >= max_steps or time.monotonic() - started >= seconds:
                raise TimeoutError('suffix step/time budget exhausted; terminal value UNKNOWN')
            first = env.step_index == start
            obs = [env.agent_observation(s) for s in (0, 1)]
            joint = []
            for s in (0, 1):
                if first and proposal is not None and s == seat:
                    phase = 'ADOPT'
                    config = {k: copy.deepcopy(v) for k, v in env.configuration.items() if k in DEFAULT_CONFIGURATION}
                    action = (adoption.first_action(actors[s], copy.deepcopy(obs[s]), config)
                              if adoption.first_action else _act(actors[s], obs[s], config))
                else:
                    phase = 'SUFFIX'
                    action = _act(actors[s], obs[s], env.configuration)
                joint.append(action)
            phase = 'SUFFIX'
            if time.monotonic() - started >= seconds:
                raise TimeoutError('suffix time budget exhausted before transition')
            row = {'step': env.step_index, 'actions': copy.deepcopy(joint)}
            env.step(joint)
            after = env.agent_observation(seat)
            row['cash'] = [f['money'] for f in after['farms']]
            tape_hash.update((_json(row) + '\n').encode())
            if trace is not None:
                trace.append(copy.deepcopy(row))
            report['steps'] += 1
            if first and proposal is not None:
                report['adoption'] = copy.deepcopy(adoption.realization)
                report['emitted_action'] = copy.deepcopy(joint[seat])
                report['first_transition'] = {'own_cash_delta': row['cash'][seat] - cash0[seat],
                                             'rival_cash_delta': row['cash'][1-seat] - cash0[1-seat]}
                phase = 'ADOPT'
                if adoption.verify:
                    failure = adoption.verify(before, copy.deepcopy(after))
                    if failure is not None:
                        raise ProposalFailed(str(failure))
        phase = 'SUFFIX'
        if time.monotonic() - started >= seconds:
            raise TimeoutError('suffix time budget exhausted after final transition/verification')
        if env.step_index != 719 or any(r.status != 'DONE' for r in env.state):
            raise ValueError('complete native 719-step terminal boundary required')
        cash = [float(r.reward) for r in env.state]
        if cash != [float(f['money']) for f in env.agent_observation(seat)['farms']]:
            raise ValueError('terminal rewards disagree with bank cash')
        report.update(end_step=env.step_index, trace_sha256=tape_hash.hexdigest())
        return SuffixResult(key, 'COMPLETE', cash[seat], cash[1-seat], realization=report)
    except Exception as error:
        status = 'PROPOSAL_FAILED' if proposal is not None and (phase == 'ADOPT' or isinstance(error, ProposalFailed)) else 'ERROR'
        return SuffixResult(key, status, failure=f'{type(error).__name__}: {error}', realization=report)


def compare(control: SuffixResult, child: SuffixResult) -> dict[str, Estimate]:
    """Measured deltas only for completed, same-cut/seat, source-bound evaluator receipts."""
    left, right = control.realization or {}, child.realization or {}
    bound = (isinstance(left, dict) and isinstance(right, dict)
             and isinstance(left.get('snapshot_sha256'), str) and len(left['snapshot_sha256']) == 64
             and all(k in left and k in right and left[k] == right[k]
                     for k in ('snapshot_sha256', 'seat', 'start_step')))
    if control.status != 'COMPLETE' or child.status != 'COMPLETE' or not bound:
        return {k: UNKNOWN for k in ('own_cash_delta', 'rival_cash_delta', 'margin_delta')}
    return {'own_cash_delta': Estimate.measured(child.own_cash - control.own_cash),
            'rival_cash_delta': Estimate.measured(child.rival_cash - control.rival_cash),
            'margin_delta': Estimate.measured(child.margin - control.margin)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--proposal', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--max-steps', type=int, default=719)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; no overwrite or automatic retry')
    import os
    wall_started, cpu_started = time.monotonic(), time.process_time()
    snapshot = Snapshot.from_dict(json.loads(args.snapshot.read_text()))
    proposal = Proposal.from_dict(json.loads(args.proposal.read_text())) if args.proposal else None
    trace = []
    result = evaluate(snapshot, proposal, seconds=args.seconds, max_steps=args.max_steps, trace=trace)
    from dataclasses import asdict
    value = {'result': asdict(result), 'margin': result.margin,
             'wdl': None if result.outcome is None else result.outcome.value, 'trace': trace,
             'snapshot_sha256': snapshot.sha256, 'source_pins': source_pins(),
             'pid': os.getpid(), 'runtime': _runtime_profile(),
             'wall_seconds': time.monotonic() - wall_started, 'cpu_seconds': time.process_time() - cpu_started,
             'native_steps': len(trace), 'native_games_attempted': int(bool(trace))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        out.write(_json(value) + '\n')
    return 0 if result.status == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
