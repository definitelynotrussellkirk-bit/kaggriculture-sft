"""Closed, source-bound JSON records for V0 regret teaching (no games or fitting).

Public interfaces are ``validate_pair(document) -> dict``,
``actor_context_key(context) -> str``, and
``validate_pair_v1_entry_prefix(context, continuation) -> (state, prefix, options)``.
The pair validator returns a detached data-only copy, immutable by convention, or
raises RecordError with a stable ``code``. The entry guard joins the complete native
entry prefix to restored continuation memory; generic actor/search contexts may
still carry partial prefixes and are not opening pair-v1 admission.
The JSONL CLI is atomic with respect to admission: it prints no success until
EVERY line, including cross-record split/identity joins, passes. JSON nesting
is bounded at 64 levels and integers at 4096 bits (no silent truncation).

Wire format is the design's identity/learner/challenger/evaluation/comparison
sections plus actor_context and the existing V0Continuation.to_dict() payload.
``content_sha256`` hashes the whole pair except itself; ``path_sha256`` hashes
one whole proposal except itself. All other digests hash their entire named
payload. No hash excludes nested provenance. Source pins are exact local file
bytes, never caller-directed file reads. Hashes establish identity, not honesty.

Actor observations use precisely the public/own State facts named by the native
scorer's state signature, not a raw observation. Compiler memory uses the existing
continuation codec without game_id; game_id stays outside the actor. This opening
version refuses non-null intraday controllers rather than accepting opaque state.
Ordered allocation/started/HOLD rows are retained, never sorted as a quotient.
This does not add features to the scorer or claim that its current encoder uses
all these facts. Producers must not hide labels inside legitimate numeric facts.

Geometry rows reuse native Prefix.payload(), native successor class IDs and phi
hashes; they deliberately have no likelihood fields. A direct draw separately
retains actual sampled raw-step logps and RNG hashes. Search selection has no
behavior likelihood; teacher class-PATH likelihood is recomputed by downstream
losses, not confused with a final-Plan marginal or a direct-policy draw.

Validation checks source joins, exact own-prefix replay, numerical summaries and
admission prerequisites. It does NOT reproduce model likelihoods, execute/replay
worlds, authenticate external checkpoint/RNG hashes, or prove declared independent
worlds/registration chronology. Evaluator and admission consumers own those gates.
A record represents ONE seat/context/path pair. Two-seat experiments retain two
records/keys and cluster by world family; two seats never become two worlds.

    PYTHONPATH=src python3 -B -m farm_neat.training.v0_regret_records \
        validate --input PAIRS.jsonl
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import State
from farm_neat.evaluation.v0_continuation import V0Continuation, FIELDS as MEMORY_FIELDS
from farm_neat.game.constants import SHOPS
from farm_neat.policy import v0_scorer as A
from farm_neat.policy import v0_menu_fast
from farm_neat.training import v0_targets as T

SCHEMA = 'farm-v0-regret-pair-v1'
ACTOR_SCHEMA = 'farm-v0-regret-actor-context-v2'
PREFIX_SCHEMA = 'farm-v0-regret-prefix-class-v1'
SELECTION_SCHEMA = 'farm-v0-regret-selection-v1'
REGISTRATION_SCHEMA = 'farm-v0-regret-confirmation-registration-v1'
ROOT = Path(__file__).resolve().parents[3]
DESIGN_PATH = 'project/configs/v0-regret-teacher.json'
ENGINE_PATH = 'src/farm_neat/game/exact_engine.py'
SPLITS = frozenset(('discovery', 'confirmation', 'training', 'monitor', 'final'))
VERDICTS = frozenset(('CHALLENGER_BETTER', 'LEARNER_BETTER', 'TIE', 'INDETERMINATE',
                      'INVALID', 'CENSORED', 'UNSUPPORTED'))
OBS_FIELDS = frozenset(('player', 'step', 'cash', 'quadrants', 'quadrant_names', 'hands_today', 'hires_today',
                        'stock', 'seeds', 'inventory', 'params', 'shops', 'occupied'))


class RecordError(ValueError):
    """Stable machine code plus a JSON-style location, without dumping private data."""
    def __init__(self, code: str, path: str):
        self.code, self.path = code, path
        super().__init__(f'{code}: {path}')


def _require(ok: bool, code: str, path: str) -> None:
    if not ok:
        raise RecordError(code, path)


def _safe(value: Any, path: str = '$', depth: int = 0) -> None:
    _require(depth <= 64, 'JSON_LIMIT', path)
    if type(value) is int:
        _require(value.bit_length() <= 4096, 'JSON_LIMIT', path)
        return
    if value is None or type(value) in (str, bool):
        return
    if type(value) is float:
        _require(math.isfinite(value), 'NONFINITE_JSON', path)
    elif type(value) is list:
        for i, item in enumerate(value):
            _safe(item, f'{path}[{i}]', depth + 1)
    elif type(value) is dict:
        _require(all(type(k) is str for k in value), 'UNSAFE_JSON', path)
        for key, item in value.items():
            _safe(item, path + '.' + key, depth + 1)
    else:
        raise RecordError('UNSAFE_JSON', path)


def canonical_sha256(value: Any) -> str:
    """SHA256 of sorted compact finite JSON, matching the current V0 codecs."""
    return T._digest(value)


def _equal(left: Any, right: Any) -> bool:
    # True/1 and a coerced noncanonical Plan must not compare as the same JSON.
    return T._json(left) == T._json(right)


def _fields(value, required, path, optional=(), code='UNKNOWN_FIELD'):
    _require(type(value) is dict, 'INVALID_OBJECT', path)
    _require(not set(value) - set(required) - set(optional), code, path)
    _require(set(required) <= set(value), 'MISSING_FIELD', path)


def _schema(value, schema, path):
    _require(value == schema, 'UNSUPPORTED_SCHEMA', path)


def _int(value, path, low=0, high=None):
    _require(type(value) is int, 'INVALID_INTEGER', path)
    _require(value >= low and (high is None or value <= high), 'INTEGER_RANGE', path)
    return value


def _bool(value, path):
    _require(type(value) is bool, 'INVALID_BOOLEAN', path)
    return value


def _number(value, path, low=None, high=None):
    _require(type(value) in (int, float), 'INVALID_NUMBER', path)
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite, 'NONFINITE_JSON', path)
    _require((low is None or value >= low) and (high is None or value <= high), 'NUMBER_RANGE', path)
    return value


def _text(value, path, code='INVALID_STRING'):
    _require(type(value) is str and bool(value.strip()), code, path)
    return value


def _hash(value, path):
    _require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
             'INVALID_HASH', path)
    return value


def _hash_join(value, expected, code, path):
    _hash(value, path)
    _require(value == expected, code, path)


def _list(value, path):
    _require(type(value) is list, 'INVALID_LIST', path)
    return value


def _texts(value, path):
    rows = _list(value, path)
    for row in rows:
        _text(row, path)
    _require(len(rows) == len(set(rows)), 'DUPLICATE_ID', path)
    return rows


def source_pins() -> dict[str, str]:
    """Exact local closure. No mutable cache and no record-supplied pathname lookup."""
    paths = (DESIGN_PATH, ENGINE_PATH, 'docs/contracts/training.md',
             'src/farm_neat/game/kaggriculture.json',
             'src/farm_neat/economy/v0/epochs.py', 'src/farm_neat/economy/orders.py',
             'src/farm_neat/evaluation/v0_continuation.py',
             'src/farm_neat/policy/v0_menu_fast.py', 'src/farm_neat/training/v0_targets.py',
             'src/farm_neat/training/v0_regret_records.py')
    return {**A.source_pins(), **{p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}}


def _plan(value, path):
    _fields(value, ('schema', 'action_schema_sha256', 'cohorts', 'regimes', 'land', 'market'), path)
    _schema(value['schema'], V.SCHEMA, path + '.schema')
    for field in ('cohorts', 'regimes', 'market'):
        _require(type(value[field]) is dict, 'NONCANONICAL_PLAN', path + '.' + field)
    try:
        plan = Plan.from_canonical(value)
        _require(_equal(plan.canonical(), value), 'NONCANONICAL_PLAN', path)
        for n in value['cohorts'].values():
            _int(n, path + '.cohorts', 1, 10000)
        for row in value['market'].values():
            _int(row[2], path + '.market.rank', 0, V.RANKS - 1)
        return plan
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        if isinstance(exc, RecordError):
            raise
        raise RecordError('NONCANONICAL_PLAN', path) from exc


def _prefix(value, path):
    _fields(value, ('plan', 'last_key', 'edit_count', 'terminated'), path)
    plan = _plan(value['plan'], path + '.plan')
    _int(value['last_key'], path + '.last_key', -1, V.N_KEYS - 1)
    _int(value['edit_count'], path + '.edit_count', 0, V.MAX_EDITS_PER_EPOCH)
    _bool(value['terminated'], path + '.terminated')
    try:
        return A.Prefix(plan, value['last_key'], value['edit_count'], value['terminated'])
    except ValueError as exc:
        raise RecordError('PREFIX_MISMATCH', path) from exc


def _memory(value, path, *, observation=None):
    _fields(value, MEMORY_FIELDS, path)
    _plan(value['plan'], path + '.plan')
    _text(value['game_id'], path + '.game_id')
    _number(value['labor_factor'], path + '.labor_factor')
    _require(value['labor_factor'] == 4.0 and value['hands_mode'] == 'daily', 'CONTINUATION_SETTINGS', path)
    _fields(value['belief'], V.PRODUCTS, path + '.belief', code='ACTOR_FIELDS')
    for p, flow in value['belief'].items():
        _number(flow, path + '.belief.' + p, 0, 0)
    _require(value['controller'] is None, 'UNSUPPORTED_CONTINUATION', path + '.controller')
    try:
        # Existing memory codec owns allocation/started/HOLD order and validity.
        return V0Continuation.from_dict(value).restore(None, observation=observation)
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise RecordError('CONTINUATION_MISMATCH', path) from exc


def _projection_observation(obs):
    """Reconstruct only supplied projection facts, never LAND history or names."""
    farm = dict(money=obs['cash'], unlocked_quadrants=copy.deepcopy(obs['quadrant_names']),
                hands=[{}] * obs['hands_today'], hires_today=obs['hires_today'], tiles=[[{}] * obs['occupied']])
    return dict(player=obs['player'], step=obs['step'], farms=[farm, farm],
                private=dict(shed=obs['stock'], seeds=obs['seeds']),
                market=dict(inventory=obs['inventory'], params=obs['params']),
                town=dict(unlocked_shops=obs['shops']))


def _actor(context):
    _safe(context)
    _fields(context, ('schema', 'observation', 'prefix', 'compiler_memory'), '$.actor_context', code='ACTOR_FIELDS')
    _schema(context['schema'], ACTOR_SCHEMA, '$.actor_context.schema')
    obs = context['observation']
    _fields(obs, OBS_FIELDS, '$.actor_context.observation', code='ACTOR_FIELDS')
    _int(obs['player'], 'observation.player', 0, 1)
    _int(obs['step'], 'observation.step', 0, V.LAST_STEP)
    _require(obs['step'] % 24 == 0, 'UNSUPPORTED_CONTEXT', 'observation.step')
    _number(obs['cash'], 'observation.cash', 0)
    for k, lo, hi in (('quadrants', 1, 4), ('hands_today', 0, V.HAND_CAP),
                       ('hires_today', 0, V.HAND_CAP), ('occupied', 0, 100)):
        _int(obs[k], 'observation.' + k, lo, hi)
    _require(type(obs['quadrant_names']) is list
             and obs['quadrant_names'] == list(V.QUADRANTS[:obs['quadrants']]),
             'ACTOR_FIELDS', 'observation.quadrant_names')
    for name, products in (('stock', V.PRODUCTS), ('seeds', A.CROPS), ('inventory', V.PRODUCTS)):
        _fields(obs[name], products, 'observation.' + name, code='ACTOR_FIELDS')
        for p, count in obs[name].items():
            _int(count, 'observation.' + name + '.' + p)
    if obs['params'] is not None:
        _fields(obs['params'], V.PRODUCTS, 'observation.params', code='ACTOR_FIELDS')
        for p, row in obs['params'].items():
            _fields(row, ('base', 'I0', 'T', 'below_func', 'above_func', 'below_target', 'above_target'),
                    'observation.params.' + p, code='ACTOR_FIELDS')
            for key in ('base', 'I0', 'T', 'below_target', 'above_target'):
                _number(row[key], 'observation.params.' + key, 0)
            _require(row['T'] > 0, 'NUMBER_RANGE', 'observation.params.T')
            for key in ('below_func', 'above_func'):
                _require(row[key] in A.SHAPES, 'ACTOR_FIELDS', 'observation.params.' + key)
    # Native shop reveals are a sequence sampled with replacement, not IDs.
    # Preserve multiplicity/order; _texts intentionally enforces uniqueness elsewhere.
    shops = _list(obs['shops'], 'observation.shops')
    for shop in shops:
        _text(shop, 'observation.shops')
    _require(all(s in SHOPS for s in shops), 'ACTOR_FIELDS', 'observation.shops')
    prefix = _prefix(context['prefix'], 'actor_context.prefix')
    mem = context['compiler_memory']
    _fields(mem, MEMORY_FIELDS - {'game_id'}, 'actor_context.compiler_memory', code='ACTOR_FIELDS')
    observation = _projection_observation(obs)
    agent = _memory({**mem, 'game_id': 'actor-context'}, 'actor_context.compiler_memory',
                    observation=observation)
    state = State(observation, land_commitment=agent.land_commitment)
    options = {**T.SETTINGS, 'hold_releases': agent.hold_releases}
    return state, prefix, options


def actor_context_key(context: dict) -> str:
    """Hash the entire validated actor allowlist, never evidence or raw alias history."""
    _actor(context)
    return canonical_sha256(context)


def validate_pair_v1_entry_prefix(context: dict, continuation: dict) -> tuple[State, A.Prefix, dict]:
    """Join an opening pair-v1 entry to its exact restored compiler continuation.

    This is shared by pair admission and pair-v1 scoring, NOT by generic _actor
    consumers: search may intentionally start at a partial prefix and multi-dawn
    recovery has its own schema. Nonempty carried Plans are valid entries.
    Compare the whole native Prefix payload (Plan, cursor, edit count, STOP),
    not just Plan hashes. Preserve every memory field/order except game_id, which
    is deliberately outside actor input. No repair, likelihood or evidence check
    occurs here; validate_pair still owns source/hash/evaluation admission.
    """
    state, prefix, options = _actor(context)
    _require(state.step == 0, 'UNSUPPORTED_CONTEXT', '$.actor_context.observation.step')
    _safe(continuation, '$.continuation')
    agent = _memory(continuation, '$.continuation',
                    observation=_projection_observation(context['observation']))
    _require(_equal({k: v for k, v in continuation.items() if k != 'game_id'}, context['compiler_memory']),
             'CONTINUATION_MISMATCH', '$.actor_context.compiler_memory')
    _require(_equal(prefix.payload(), A.Prefix(agent.plan).payload()),
             'ENTRY_PREFIX_MISMATCH', '$.actor_context.prefix')
    return state, prefix, options


def _proposal(value, identity, prefix, state, options, menus, *, direct):
    path = '$.learner' if direct else '$.challenger'
    common = ('raw_actions', 'prefix_class_trace', 'canonical_plan', 'continuation_signature',
              'sampling_provenance', 'path_sha256')
    extra = (('direct_draw_id', 'raw_behavior_logp') if direct else
             ('proposal_family', 'parent_proposal_ids', 'search_budget', 'selection_receipt', 'selection_receipt_sha256'))
    _fields(value, common + extra, path, optional=() if direct else ('raw_behavior_logp',))
    _hash_join(value['path_sha256'], canonical_sha256({k: v for k, v in value.items() if k != 'path_sha256'}),
               'PATH_HASH_MISMATCH', path)
    actions, rows = _list(value['raw_actions'], path + '.raw_actions'), _list(value['prefix_class_trace'], path + '.trace')
    _require(0 < len(actions) == len(rows) <= V.MAX_EDITS_PER_EPOCH + 1, 'INCOMPLETE_PATH', path)
    for action in actions:
        _int(action, path + '.raw_actions', 0, V.N_ACTIONS - 1)
    provenance = value['sampling_provenance']
    if direct:
        _text(value['direct_draw_id'], path + '.direct_draw_id')
        _fields(provenance, ('type', 'mode', 'checkpoint_sha256', 'temperature', 'raw_step_logps',
                             'rng_before_sha256', 'rng_after_sha256'), path + '.sampling_provenance')
        _require(provenance['type'] == 'direct_policy' and provenance['mode'] == 'sample_raw',
                 'NOT_DIRECT_SAMPLE', path)
        _hash_join(provenance['checkpoint_sha256'], identity['parent_checkpoint_sha256'], 'SOURCE_MISMATCH', path)
        _number(provenance['temperature'], path + '.temperature', 0)
        _require(provenance['temperature'] > 0, 'NUMBER_RANGE', path + '.temperature')
        for name in ('rng_before_sha256', 'rng_after_sha256'):
            _hash(provenance[name], path + '.' + name)
        logps = _list(provenance['raw_step_logps'], path + '.raw_step_logps')
        _require(len(logps) == len(actions), 'LIKELIHOOD_MISMATCH', path)
        for logp in logps:
            _number(logp, path + '.raw_step_logps', high=0)
        _number(value['raw_behavior_logp'], path + '.raw_behavior_logp', high=0)
        try:
            total_logp = math.fsum(logps)
        except OverflowError as exc:
            raise RecordError('LIKELIHOOD_MISMATCH', path + '.raw_step_logps') from exc
        _require(math.isclose(total_logp, value['raw_behavior_logp'], abs_tol=3e-5, rel_tol=2e-6),
                 'LIKELIHOOD_MISMATCH', path)
    else:
        _fields(provenance, ('type',), path + '.sampling_provenance', code='SEARCH_LIKELIHOOD')
        _require(provenance['type'] == 'search_selection' and value.get('raw_behavior_logp') is None,
                 'SEARCH_LIKELIHOOD', path)
        _text(value['proposal_family'], path + '.proposal_family')
        _texts(value['parent_proposal_ids'], path + '.parent_proposal_ids')
        budget = value['search_budget']
        _fields(budget, ('max_candidates', 'generated_candidates', 'cpu_seconds'), path + '.search_budget')
        _int(budget['max_candidates'], path + '.max_candidates', 1, 32)
        _int(budget['generated_candidates'], path + '.generated_candidates', 1, budget['max_candidates'])
        _number(budget['cpu_seconds'], path + '.cpu_seconds', 0)
        receipt = value['selection_receipt']
        _fields(receipt, ('schema', 'actor_context_sha256', 'parent_checkpoint_sha256', 'raw_actions_sha256',
                         'continuation_signature', 'pool_sha256', 'confirmation_registration_sha256'), path + '.selection')
        _schema(receipt['schema'], SELECTION_SCHEMA, path + '.selection.schema')
        _hash_join(value['selection_receipt_sha256'], canonical_sha256(receipt), 'SELECTION_MISMATCH', path)
        for key in ('actor_context_sha256', 'parent_checkpoint_sha256'):
            _hash_join(receipt[key], identity[key], 'SELECTION_MISMATCH', path + '.' + key)
        _hash_join(receipt['raw_actions_sha256'], canonical_sha256(actions), 'SELECTION_MISMATCH', path)
        _hash(receipt['pool_sha256'], path + '.pool_sha256')
        _require(_equal(receipt['continuation_signature'], value['continuation_signature']), 'SELECTION_MISMATCH', path)
    for i, (action, row) in enumerate(zip(actions, rows)):
        loc = f'{path}.prefix_class_trace[{i}]'
        _fields(row, ('schema', 'prefix', 'support', 'phi_sha256', 'raw_id', 'class_id', 'class_members',
                      'continuation', 'forced_stop', 'termination'), loc)
        _schema(row['schema'], PREFIX_SCHEMA, loc)
        _bool(row['forced_stop'], loc + '.forced_stop')
        _int(row['raw_id'], loc + '.raw_id', 0, V.N_ACTIONS - 1)
        _require(not prefix.terminated and _equal(row['prefix'], prefix.payload()), 'PREFIX_MISMATCH', loc)
        key = T._json(prefix.payload())
        if key not in menus:
            try:
                menus[key] = v0_menu_fast.build_menu(state, prefix, projection_kwargs=options, backend='fast')
            except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as exc:
                raise RecordError('INVALID_PROJECTION', loc) from exc
        menu = menus[key]
        _require(_equal(row['support'], list(menu.legal_ids)), 'STALE_SUPPORT', loc)
        _hash_join(row['phi_sha256'], canonical_sha256(menu.phi), 'STALE_SUPPORT', loc + '.phi')
        _require(action in menu.legal_ids and row['raw_id'] == action, 'ILLEGAL_ACTION', loc)
        index = menu.legal_ids.index(action)
        cls = next(c for c in menu.classes if index in c.indices)
        forced = prefix.edit_count == V.MAX_EDITS_PER_EPOCH
        termination = ('CAP' if forced else 'STOP') if action == V.STOP_ID else 'CONTINUE'
        _require(row['class_id'] == cls.class_id and _equal(row['class_members'], [menu.legal_ids[j] for j in cls.indices])
                 and _equal(row['continuation'], cls.successor.payload()) and row['forced_stop'] == forced
                 and row['termination'] == termination, 'PREFIX_MISMATCH', loc)
        if direct and len(menu.legal_ids) == 1:
            _require(logps[i] == 0, 'LIKELIHOOD_MISMATCH', loc + '.forced_or_singleton_logp')
        prefix = cls.successor  # Never feed the stored/teacher prefix into the next menu.
    _require(prefix.terminated, 'INCOMPLETE_PATH', path)
    _plan(value['canonical_plan'], path + '.canonical_plan')
    expected = T.continuation_signature(prefix.plan, prefix.last_key, prefix.edit_count, prefix.terminated)
    _require(_equal(value['canonical_plan'], prefix.plan.canonical()) and _equal(value['continuation_signature'], expected),
             'CONTINUATION_MISMATCH', path)


def _evaluation(value, comparison, document):
    path = '$.evaluation'
    _fields(value, ('mode', 'discovery_world_groups', 'confirmation_world_groups', 'seat_rows', 'engine_hash',
                   'opponent_hash', 'continuation_hash', 'complete', 'replay_verified', 'attempt_ids',
                   'confirmation_registration'), path)
    _require(value['mode'] in ('EXACT_REALIZATION', 'CONFIRMED_EXPECTATION', 'PREDICTED_ONLY', 'UNEVALUATED'),
             'UNSUPPORTED_EVIDENCE', path + '.mode')
    identity = document['identity']
    _hash_join(value['engine_hash'], identity['source_pins'][ENGINE_PATH], 'SOURCE_MISMATCH', path + '.engine_hash')
    _hash(value['opponent_hash'], path + '.opponent_hash')
    _hash_join(value['continuation_hash'], identity['continuation_sha256'], 'CONTINUATION_MISMATCH', path)
    _bool(value['complete'], path + '.complete'); _bool(value['replay_verified'], path + '.replay_verified')
    groups, partition, world_hashes = {}, {}, set()
    for name in ('discovery_world_groups', 'confirmation_world_groups'):
        for world in _list(value[name], path + '.' + name):
            _fields(world, ('source_group_id', 'world_sha256', 'snapshot_sha256', 'rng_sha256', 'seed'), path + '.' + name)
            group = _text(world['source_group_id'], path + '.' + name, 'MISSING_SOURCE_GROUP')
            _require(group not in groups, 'WORLD_GROUP_OVERLAP', path)
            for key in ('world_sha256', 'snapshot_sha256', 'rng_sha256'):
                _hash(world[key], path + '.' + key)
            _require(world['world_sha256'] not in world_hashes, 'DUPLICATE_WORLD', path)
            _int(world['seed'], path + '.seed')
            groups[group], partition[group] = world, name
            world_hashes.add(world['world_sha256'])
    attempts = _texts(value['attempt_ids'], path + '.attempt_ids')
    used, seen, deltas = set(), set(), {}
    rows = _list(value['seat_rows'], path + '.seat_rows')
    actor_seat = document['actor_context']['observation']['player']
    for row in rows:
        _fields(row, ('source_group_id', 'seat', 'actor_context_sha256', 'learner_wdl', 'challenger_wdl',
                      'complete', 'replay_verified', 'learner_attempt_id', 'challenger_attempt_id'), path + '.seat_rows')
        group = _text(row['source_group_id'], path + '.source_group_id', 'MISSING_SOURCE_GROUP')
        _require(group in groups and group not in seen, 'WORLD_ROW_MISMATCH', path)
        seen.add(group)
        _int(row['seat'], path + '.seat', 0, 1)
        _require(row['seat'] == actor_seat, 'SEAT_CONTEXT_MISMATCH', path)
        _hash_join(row['actor_context_sha256'], identity['actor_context_sha256'], 'ACTOR_CONTEXT_MISMATCH', path)
        for key in ('complete', 'replay_verified'):
            _bool(row[key], path + '.' + key)
        _require(not row['replay_verified'] or row['complete'], 'INCOMPLETE_EVIDENCE', path)
        for key in ('learner_attempt_id', 'challenger_attempt_id'):
            attempt = _text(row[key], path + '.' + key)
            _require(attempt in attempts and attempt not in used, 'ATTEMPT_MISMATCH', path)
            used.add(attempt)
        for key in ('learner_wdl', 'challenger_wdl'):
            if row['complete']:
                _int(row[key], path + '.' + key, -1, 1)
            else:
                _require(row[key] is None, 'CENSORED_OUTCOME', path)
        if row['complete']:
            deltas[group] = row['challenger_wdl'] - row['learner_wdl']
    _require(not value['complete'] or (bool(groups) and seen == set(groups) and len(deltas) == len(groups)),
             'INCOMPLETE_EVIDENCE', path)
    _require(not value['replay_verified'] or (value['complete'] and all(r['replay_verified'] for r in rows)),
             'INCOMPLETE_EVIDENCE', path)
    _fields(comparison, ('objective', 'mean_wdl_delta', 'lower_bound', 'alpha', 'independent_world_count',
                        'verdict', 'actor_training_admissible', 'reason_codes'), '$.comparison')
    _int(comparison['independent_world_count'], 'comparison.independent_world_count')
    _bool(comparison['actor_training_admissible'], 'comparison.actor_training_admissible')
    _require(comparison['objective'] == 'TERMINAL_WDL', 'UNSUPPORTED_OBJECTIVE', '$.comparison')
    _require(type(comparison['verdict']) is str and comparison['verdict'] in VERDICTS, 'UNSUPPORTED_VERDICT', '$.comparison')
    reasons = _texts(comparison['reason_codes'], 'comparison.reason_codes')
    mode = value['mode']
    chosen = {g: d for g, d in deltas.items() if mode != 'CONFIRMED_EXPECTATION' or partition[g] == 'confirmation_world_groups'}
    _require(comparison['independent_world_count'] == len(chosen), 'WORLD_COUNT_MISMATCH', '$.comparison')
    if chosen:
        _number(comparison['mean_wdl_delta'], 'comparison.mean_wdl_delta', -2, 2)
        mean = math.fsum(chosen.values()) / len(chosen)
        _require(math.isclose(comparison['mean_wdl_delta'], mean, abs_tol=1e-12, rel_tol=1e-12),
                 'COMPARISON_MISMATCH', '$.comparison')
        sign = 'CHALLENGER_BETTER' if mean > 0 else 'LEARNER_BETTER' if mean < 0 else 'TIE'
        _require(comparison['verdict'] not in ('CHALLENGER_BETTER', 'LEARNER_BETTER', 'TIE')
                 or comparison['verdict'] == sign, 'COMPARISON_MISMATCH', '$.comparison.verdict')
    else:
        _require(comparison['mean_wdl_delta'] is None, 'COMPARISON_MISMATCH', '$.comparison.mean_wdl_delta')
    registration = value['confirmation_registration']
    selected_reg = document['challenger']['selection_receipt']['confirmation_registration_sha256']
    if mode != 'CONFIRMED_EXPECTATION':
        _require(comparison['alpha'] is None and comparison['lower_bound'] is None and registration is None
                 and selected_reg is None, 'EVIDENCE_MODE_MISMATCH', '$.comparison')
    else:
        _fields(registration, ('schema', 'family_alpha', 'tested_pairs', 'alpha_pair', 'min_effect',
                    'confirmation_source_group_ids', 'learner_path_sha256', 'challenger_actions_sha256',
                    'independence_attested', 'compatible_contexts_attested', 'frozen_before_confirmation'),
                path + '.confirmation_registration')
        _schema(registration['schema'], REGISTRATION_SCHEMA, path + '.confirmation_registration')
        _hash_join(selected_reg, canonical_sha256(registration), 'REGISTRATION_MISMATCH', path)
        _hash_join(registration['learner_path_sha256'], document['learner']['path_sha256'], 'REGISTRATION_MISMATCH', path)
        _hash_join(registration['challenger_actions_sha256'], canonical_sha256(document['challenger']['raw_actions']),
                   'REGISTRATION_MISMATCH', path)
        expected_groups = _texts(registration['confirmation_source_group_ids'], path + '.confirmation_groups')
        _require(set(expected_groups) == {g for g in groups if partition[g] == 'confirmation_world_groups'}
                 and bool(expected_groups), 'REGISTRATION_MISMATCH', path)
        _number(registration['family_alpha'], path + '.family_alpha', 0, 0.05)
        _int(registration['tested_pairs'], path + '.tested_pairs', 1)
        alpha = _number(registration['alpha_pair'], path + '.alpha_pair', 0, 1)
        _require(alpha > 0 and alpha == registration['family_alpha'] / registration['tested_pairs'],
                 'REGISTRATION_MISMATCH', path)
        _number(registration['min_effect'], path + '.min_effect', 0, 2)
        for name in ('independence_attested', 'compatible_contexts_attested', 'frozen_before_confirmation'):
            _bool(registration[name], path + '.' + name)
        _number(comparison['alpha'], 'comparison.alpha', 0, 1)
        _require(comparison['alpha'] == alpha, 'REGISTRATION_MISMATCH', '$.comparison.alpha')
        if chosen and len(chosen) == len(expected_groups):
            bound = mean - math.sqrt(8 * -math.log(alpha) / len(chosen))
            _number(comparison['lower_bound'], 'comparison.lower_bound')
            _require(math.isclose(comparison['lower_bound'], bound, rel_tol=1e-12, abs_tol=1e-12),
                     'COMPARISON_MISMATCH', '$.comparison.lower_bound')
        else:
            _require(comparison['lower_bound'] is None, 'INCOMPLETE_EVIDENCE', '$.comparison.lower_bound')
    admissible = comparison['actor_training_admissible']
    if admissible:
        _require(mode == 'CONFIRMED_EXPECTATION' and value['complete'] and value['replay_verified']
                 and comparison['verdict'] == 'CHALLENGER_BETTER' and not reasons
                 and comparison['lower_bound'] is not None and comparison['lower_bound'] > registration['min_effect']
                 and all(registration[k] for k in ('independence_attested', 'compatible_contexts_attested', 'frozen_before_confirmation'))
                 and identity['split'] in ('confirmation', 'training'), 'ADMISSION_FORBIDDEN', '$.comparison')
    else:
        _require(bool(reasons), 'MISSING_REASON', '$.comparison')
    if mode in ('PREDICTED_ONLY', 'UNEVALUATED'):
        _require(not rows and not groups and not value['complete'] and not value['replay_verified'],
                 'EVIDENCE_MODE_MISMATCH', path)
    if not value['complete']:
        _require(comparison['verdict'] in ('INVALID', 'CENSORED', 'UNSUPPORTED', 'INDETERMINATE'),
                 'INCOMPLETE_EVIDENCE', '$.comparison.verdict')


def validate_pair(document: dict) -> dict:
    """Fail closed; no repairs, invented logps, game calls or evidence upgrades."""
    _safe(document)
    _fields(document, ('schema', 'identity', 'actor_context', 'continuation', 'learner', 'challenger',
                       'evaluation', 'comparison', 'content_sha256'), '$')
    _schema(document['schema'], SCHEMA, '$.schema')
    identity = document['identity']
    _fields(identity, ('pair_id', 'source_pins', 'protocol_sha256', 'parent_checkpoint_sha256',
              'actor_context_sha256', 'continuation_sha256', 'source_group_id', 'split', 'round_id'), '$.identity')
    for key in ('pair_id', 'round_id'):
        _text(identity[key], '$.identity.' + key)
    _text(identity['source_group_id'], '$.identity.source_group_id', 'MISSING_SOURCE_GROUP')
    _require(type(identity['split']) is str and identity['split'] in SPLITS, 'UNSUPPORTED_SPLIT', '$.identity.split')
    pins = source_pins()
    _require(_equal(identity['source_pins'], pins), 'SOURCE_MISMATCH', '$.identity.source_pins')
    _hash_join(identity['protocol_sha256'], pins[DESIGN_PATH], 'SOURCE_MISMATCH', '$.identity.protocol_sha256')
    _hash(identity['parent_checkpoint_sha256'], '$.identity.parent_checkpoint_sha256')
    state, prefix, options = validate_pair_v1_entry_prefix(document['actor_context'], document['continuation'])
    _hash_join(identity['actor_context_sha256'], canonical_sha256(document['actor_context']),
               'ACTOR_CONTEXT_MISMATCH', '$.identity.actor_context_sha256')
    continuation = document['continuation']
    _hash_join(identity['continuation_sha256'], canonical_sha256(continuation), 'CONTINUATION_MISMATCH', '$.continuation')
    menus = {}  # Call-local; arms may share equal prefixes, never stale cross-record state.
    _proposal(document['learner'], identity, prefix, state, options, menus, direct=True)
    _proposal(document['challenger'], identity, prefix, state, options, menus, direct=False)
    _evaluation(document['evaluation'], document['comparison'], document)
    _hash_join(document['content_sha256'], canonical_sha256({k: v for k, v in document.items() if k != 'content_sha256'}),
               'CONTENT_HASH_MISMATCH', '$.content_sha256')
    return copy.deepcopy(document)


def _loads(text):
    def constant(_):
        raise RecordError('NONFINITE_JSON', '$')
    def pairs(items):
        out = {}
        for key, value in items:
            _require(key not in out, 'DUPLICATE_JSON_KEY', '$')
            out[key] = value
        return out
    def integer(text):
        _require(len(text.lstrip('-')) <= 1234, 'JSON_LIMIT', '$')
        value = int(text)
        _require(value.bit_length() <= 4096, 'JSON_LIMIT', '$')
        return value
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_int=integer)
    except RecordError:
        raise
    except (ValueError, RecursionError) as exc:
        raise RecordError('MALFORMED_JSON', '$') from exc


def validate_jsonl(path: str | Path) -> dict:
    """File-level split/identity joins. No rows are emitted before total success."""
    count, ids, exposures, draws = 0, set(), {}, {}
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                _require(bool(line.strip()), 'EMPTY_RECORD', '$')
                pair = validate_pair(_loads(line))
                identity = pair['identity']
                _require(identity['pair_id'] not in ids, 'DUPLICATE_PAIR', '$.identity.pair_id')
                ids.add(identity['pair_id'])
                # Confirmation/training are exposed together, never monitor/final.
                split = 'training' if identity['split'] == 'confirmation' else identity['split']
                keys = [('source', identity['source_group_id'])]
                for field in ('discovery_world_groups', 'confirmation_world_groups'):
                    for w in pair['evaluation'][field]:
                        keys.extend((('source', w['source_group_id']), ('world', w['world_sha256'])))
                for key in keys:
                    _require(key not in exposures or exposures[key] == split, 'SPLIT_LEAKAGE', '$.identity.split')
                    exposures[key] = split
                draw = (identity['parent_checkpoint_sha256'], identity['round_id'], pair['learner']['direct_draw_id'])
                binding = (identity['actor_context_sha256'], pair['learner']['path_sha256'])
                _require(draw not in draws or draws[draw] == binding, 'DRAW_ID_REUSE', '$.learner.direct_draw_id')
                draws[draw] = binding
                count += 1
            except RecordError as exc:
                raise RecordError(exc.code, f'line {line_number}: {exc.path}') from exc
    _require(count > 0, 'EMPTY_INPUT', '$')
    return {'schema': SCHEMA + '-validation', 'status': 'VALID', 'validated_pairs': count,
            'unique_direct_draws': len(draws), 'games_executed': 0,
            'scope': 'SOURCE_PATH_AND_RECORD_CONSISTENCY_NOT_RUNTIME_OR_INDEPENDENCE_CERTIFICATION'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    validate = sub.add_parser('validate')
    validate.add_argument('--input', required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_jsonl(args.input)
    except RecordError as exc:
        print(json.dumps({'status': 'INVALID', 'reason_code': exc.code, 'path': exc.path}, sort_keys=True))
        return 2
    except (OSError, UnicodeError) as exc:
        print(json.dumps({'status': 'INVALID', 'reason_code': 'INPUT_IO_ERROR', 'path': type(exc).__name__}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
