"""Opening-only V0 matched continuations; execution identity, NEVER admission.

``evaluate_pair(request, *, runner)`` executes exactly two arms. ``verify_pair``
executes two NEW attempts and compares their semantic receipts, not timing/IDs.
Use ``make_request(validated_unevaluated_pair, world_id=..., opponent=...)`` to
freeze one registered fixed-shop world/seat, local sources, compiler memory and
RNG semantics. An injected runner receives a detached, bound branch dictionary
and returns BRANCH_RESULT_SCHEMA. Only the built-in fresh-process native_runner
can emit NATIVE_GAME evidence. Injected runners are trusted diagnostics, not a
sandbox or a way to authenticate third-party claims.

The native implementation calls the EXISTING ScenarioEnvironment and PolicyAgent;
no economics, action vocabulary, compiler or engine transition is implemented here.
It restores V0Continuation at opening and v0_policy's source-bound snapshot at
dawn 24. A fresh responsive opponent is called on each branch's own observations.
Only four named deterministic opponents are supported; arbitrary stochastic actor
RNG codecs and multi-dawn neural replanning are deliberately not implemented.

Engine RNG is day-keyed, not a persistent generator. Empty-tile-dependent weed
draws can shift its within-day stream. Receipts bind actual state transitions and
day keys, but DO NOT certify aligned exogenous draws. Fixed future shops, seeds,
world snapshots, outcomes and rival memory stay evaluator-only.

Record output is EXACT_REALIZATION, has zero actor-training admission, and retains
both cashes. Statistical confirmation, authenticating the original policy draw,
training, runtime benchmarks and promotion belong to other components.

CLI (CPU fixtures / maintainer orchestration; no automatic retries):
  python -m farm_neat.evaluation.v0_regret_pairs evaluate --request R.json --output E.json
  python -m farm_neat.evaluation.v0_regret_pairs verify --request R.json --result E.json --output V.json
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.epochs import DAWNS, TrajectoryPolicy
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import State
from farm_neat.evaluation import router_panel as panel, scenarios, v0_policy as P
from farm_neat.evaluation.v0_continuation import V0Continuation
from farm_neat.training import v0_regret_records as R

MODULE = 'farm_neat.evaluation.v0_regret_pairs'
SCHEMA = 'farm-v0-regret-execution-v1'
REQUEST_SCHEMA = SCHEMA + '-request'
BRANCH_SCHEMA = SCHEMA + '-branch'
BRANCH_RESULT_SCHEMA = SCHEMA + '-branch-result-v2'
ARMS = ('learner', 'challenger')
OPPONENTS = ('pass', 'router', 'k0006', 'tape0')
ROOT = P.ROOT
SCOPE = 'REALIZATION_AND_REPLAY_ONLY; FIXED_SHOP_WORLDS; NO_STATISTICAL_OR_NEURAL_STRENGTH_CLAIM'


class PairError(ValueError):
    """Stable refusal reason. Runtime failures are retained as invalid/incomplete arms."""
    def __init__(self, code, detail=''):
        self.code = code
        super().__init__(code + (': ' + detail if detail else ''))


def _require(ok, code, detail=''):
    if not ok:
        raise PairError(code, detail)


def _same(actual, expected, code):
    _require(P.wire(actual) == P.wire(expected), code)


def _closed(value, fields, code):
    _require(type(value) is dict and set(value) == set(fields), code)


def _bound(value, code):
    R._safe(value)
    _require(type(value) is dict and 'content_sha256' in value, code)
    _same(value, P.bind(value), code)


def _plain(value):
    return json.loads(P.wire(value))


def source_pins() -> dict:
    names = ('src/farm_neat/evaluation/v0_regret_pairs.py',
             'src/farm_neat/evaluation/router_panel.py',
             'src/farm_neat/evaluation/shop_router.py',
             'src/farm_neat/evaluation/k0006.py',
             'src/farm_neat/evaluation/opponents.py')
    return {**R.source_pins(), **P.source_pins(),
            **{p: panel.file_sha(ROOT / p) for p in names}}


def observation_facts(observation: dict) -> dict:
    """Existing State's closed public/own projection; no seed, rival private or label."""
    state = State(observation)
    facts = {k: copy.deepcopy(getattr(state, k)) for k in R.OBS_FIELDS}
    # State omits zero-count seed keys; the closed record requires all crop keys.
    facts['seeds'] = {crop: state.seeds.get(crop, 0) for crop in R.A.CROPS}
    return _plain(facts)


def _world(document):
    _closed(document, ('id', 'purpose', 'seed', 'shops', 'random_weeds', 'cohort'), 'WORLD_FIELDS')
    known = panel.world_registry()[document['id']]
    _require(panel.canonical(document) == panel.canonical(_plain(asdict(known))), 'WORLD_MISMATCH')
    return known


def _world_snapshot(env):
    """Exact ordered engine boundary, not an actor feature or a lossy Plan digest."""
    return _plain({'state': env.state, 'configuration': env.configuration,
                   'info': env.info, 'step': env.step_index, 'done': env.done,
                   'scenario_identity': env.scenario_identity})


def _rng_binding(world):
    return {'engine': 'DAY_KEYED_PYTHON_RANDOM', 'seed': world.seed,
            'key_recipe': '(seed * 1000003) XOR zero_based_day',
            'persistent_rng_state': None, 'fixed_shops': list(world.shops),
            'random_weeds': world.random_weeds,
            'weed_draw_alignment': 'ACTION_DEPENDENT_EMPTY_TILES',
            'common_exogenous_events': False,
            'own_policy': 'DETERMINISTIC_FROZEN_OPENING_THEN_KEEP',
            'opponent_policy': 'FRESH_DETERMINISTIC_REGISTERED_INSTANCE',
            'trace_scope': 'DAY_KEYS_AND_STATE_TRANSITIONS_NOT_DRAW_ALIGNMENT_CERTIFICATE'}


def _check_opening(record, env, seat):
    _require(type(seat) is int and seat in (0, 1), 'SEAT_MISMATCH')
    context = record['actor_context']; memory = record['continuation']
    _require(context['observation']['player'] == seat, 'SEAT_CONTEXT_MISMATCH')
    _require(panel.canonical(observation_facts(env.agent_observation(seat))) ==
             panel.canonical(context['observation']), 'ACTUAL_OPENING_CONTEXT_MISMATCH')
    prefix = context['prefix']
    _require(prefix['last_key'] == -1 and prefix['edit_count'] == 0 and not prefix['terminated'],
             'UNSUPPORTED_INITIAL_PREFIX', 'opening execution requires an epoch-entry prefix')
    _require(panel.canonical(prefix['plan']) == panel.canonical(memory['plan']),
             'INITIAL_PLAN_MEMORY_MISMATCH')
    agent = V0Continuation.from_dict(memory).restore(
        None, agent_cls=P.PolicyAgent, observation=env.agent_observation(seat))
    _require(memory['controller'] is None and agent.labor_factor == 4. and
             agent.hands_mode == 'daily' and set(agent.belief) == set(V.PRODUCTS) and
             all(type(v) in (int, float) and v == 0 for v in agent.belief.values()),
             'COMPILER_SETTINGS')
    for arm in ARMS:
        decision = P.validate_edits(agent.actor_state(env.agent_observation(seat), initialize=False), agent.plan,
                    record[arm]['raw_actions'], projection_kwargs=agent.projection_kw())
        _require(panel.canonical(decision.plan.canonical()) == panel.canonical(record[arm]['canonical_plan']),
                 'ACTUAL_SUCCESSOR_MISMATCH')


def make_request(record: dict, *, world_id: str, opponent='router', timeout_seconds=120.) -> dict:
    """Freeze ONE context/path pair. Constructing its opening executes zero turns."""
    record = R.validate_pair(record)
    _require(record['evaluation']['mode'] == 'UNEVALUATED', 'EVIDENCE_OVERWRITE_FORBIDDEN')
    _require(opponent in OPPONENTS, 'UNSUPPORTED_OPPONENT')
    _require(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds)
             and 0 < timeout_seconds <= 300., 'TIMEOUT_BUDGET')
    world = panel.world_registry()[world_id]
    rival = panel.load_actor(opponent)
    env = scenarios.ScenarioEnvironment(world.scenario())
    seat = record['actor_context']['observation']['player']
    _check_opening(record, env, seat)
    return P.bind({'schema': REQUEST_SCHEMA, 'record': record, 'world': _plain(asdict(world)),
        'world_sha256': world.sha256, 'seat': seat,
        'opponent': {'ref': opponent, 'identity': rival.identity},
        'source_pins': source_pins(), 'initial_world_sha256': P.wire_hash(_world_snapshot(env)),
        'rng_binding': _rng_binding(world), 'arms': list(ARMS),
        'timeout_seconds': float(timeout_seconds)})


def _validate_request(request):
    _bound(request, 'REQUEST_HASH_MISMATCH')
    _closed(request, ('schema', 'record', 'world', 'world_sha256', 'seat', 'opponent',
        'source_pins', 'initial_world_sha256', 'rng_binding', 'arms', 'timeout_seconds',
        'content_sha256'), 'REQUEST_FIELDS')
    _require(request['schema'] == REQUEST_SCHEMA, 'REQUEST_SCHEMA')
    _same(request['arms'], list(ARMS), 'DUPLICATE_OR_MISSING_ARMS')
    _require(panel.canonical(request['source_pins']) == panel.canonical(source_pins()), 'SOURCE_MISMATCH')
    _closed(request['opponent'], ('ref', 'identity'), 'OPPONENT_FIELDS')
    expected = make_request(request['record'], world_id=request['world']['id'],
                           opponent=request['opponent']['ref'], timeout_seconds=request['timeout_seconds'])
    for key in ('world', 'world_sha256', 'seat', 'opponent', 'initial_world_sha256', 'rng_binding'):
        _require(panel.canonical(request[key]) == panel.canonical(expected[key]), key.upper() + '_MISMATCH')
    return copy.deepcopy(request)


def actor_identity(request, arm):
    """Compiler + exact raw opening schedule; not a learned policy/checkpoint identity."""
    return panel.bound({'kind': 'V0_FROZEN_OPENING_THEN_KEEP',
        'compiler_memory': request['record']['continuation'],
        'raw_actions': request['record'][arm]['raw_actions'],
        'source_pins_sha256': panel.digest(request['source_pins'])})


def execution_binding(runner):
    native = runner is _NATIVE_RUNNER
    fp = panel._fingerprint(_native_game if native else runner)
    if native:
        fp['module'] = MODULE  # Stable across python -m's __main__ spelling.
    return panel.bound({'kind': 'NATIVE_GAME' if native else 'SYNTHETIC_DIAGNOSTIC',
        'runner': fp, 'python': sys.version,
        'trust': 'SOURCE_BOUND_TRUSTED_RUNNER_NOT_A_SANDBOX',
        'fresh_process_per_native_attempt': native})


def _branch(request, arm, execution, phase):
    return P.bind({'schema': BRANCH_SCHEMA, 'request': copy.deepcopy(request),
        'arm': arm, 'execution': copy.deepcopy(execution), 'phase': phase,
        'attempt_id': 'regret-' + uuid.uuid4().hex})


def _expected_policy(request, arm):
    return P._policy_snapshot(TrajectoryPolicy({0: request['record'][arm]['raw_actions'][:-1]}))


def _validate_response(branch, response):
    _bound(response, 'RESPONSE_HASH_MISMATCH')
    _closed(response, ('schema', 'branch_sha256', 'source_pins_sha256', 'row',
        'initial_world_sha256', 'actor_context_sha256', 'rng_binding', 'actions_by_seat',
        'continuation', 'epochs', 'rng_trace', 'final_observation', 'terminal_environment_sha256', 'content_sha256'),
        'RESPONSE_FIELDS')
    req = branch['request']; arm = branch['arm']; rec = req['record']; row = response['row']
    _require(response['schema'] == BRANCH_RESULT_SCHEMA, 'RESPONSE_SCHEMA')
    for key, expected in (
        ('branch_sha256', branch['content_sha256']),
        ('source_pins_sha256', panel.digest(req['source_pins'])),
        ('initial_world_sha256', req['initial_world_sha256']),
        ('actor_context_sha256', rec['identity']['actor_context_sha256']),
        ('rng_binding', req['rng_binding'])):
        _same(response[key], expected, key.upper() + '_MISMATCH')
    con = response['continuation']
    _closed(con, ('initial', 'initial_policy', 'final', 'final_policy', 'dawn24_checkpoint'), 'CONTINUATION_FIELDS')
    # JSON object key order is not codec state. Ordered allocation/started/HOLD
    # row lists remain ordered and therefore remain bound by canonical JSON.
    _require(panel.canonical(con['initial']) == panel.canonical(rec['continuation']),
             'INITIAL_COMPILER_MEMORY_MISMATCH')
    _same(con['initial_policy'], _expected_policy(req, arm), 'INITIAL_POLICY_MISMATCH')
    _same(con['final_policy'], con['initial_policy'], 'FUTURE_POLICY_MISMATCH')
    final_observation = response['final_observation']
    _require(type(final_observation) is dict and final_observation.get('player') == req['seat']
             and final_observation.get('step') == row['steps'], 'FINAL_OBSERVATION_BOUNDARY')
    final = V0Continuation.from_dict(con['final']).restore(None, observation=final_observation)
    _require(final.game_id == rec['continuation']['game_id'] and final.labor_factor == 4.
             and final.hands_mode == 'daily', 'FINAL_CONTINUATION_SETTINGS')
    _require(panel.canonical(final.belief) == panel.canonical(rec['continuation']['belief']),
             'FINAL_CONTINUATION_SETTINGS')
    rival = req['opponent']['identity']['content_sha256']; actor = actor_identity(req, arm)
    ids = [rival, rival]; ids[req['seat']] = actor['content_sha256']
    for key, expected in (('world', req['world']['id']), ('world_sha256', req['world_sha256']),
        ('seed', req['world']['seed']), ('seat', req['seat']), ('agent', arm),
        ('opponent', req['opponent']['ref']), ('actor_identities_by_seat', ids),
        ('actor_identity_hashes', {'agent': actor['content_sha256'], 'opponent': rival}),
        ('evidence_kind', branch['execution']['kind']),
        ('execution_sha256', branch['execution']['content_sha256'])):
        _same(row[key], expected, 'ROW_' + key.upper() + '_MISMATCH')
    _require(type(row['steps']) is int and 0 <= row['steps'] <= 719, 'STEPS')
    tapes = response['actions_by_seat']
    _require(type(tapes) is list and len(tapes) == 2 and all(type(t) is list for t in tapes), 'ACTION_TAPES')
    _same(row['actions_sha256'], panel.digest(tapes), 'ACTION_HASH_MISMATCH')
    _same(row['actions_by_seat_sha256'], [panel.digest(t) for t in tapes], 'SEAT_ACTION_HASH_MISMATCH')
    native = branch['execution']['kind'] == 'NATIVE_GAME'
    if not native:
        _same(row['physical_games_completed'], 0, 'SYNTHETIC_NATIVE_CLAIM')
    if row['status'] != 'COMPLETE':
        _require(row['status'] in ('TIMEOUT', 'CENSORED', 'CRASHED', 'PARTIAL', 'FAILED'), 'UNKNOWN_STATUS')
        # Preserve a malicious/buggy partial cash claim only in raw evidence, never as a label.
        _require(all(row[k] is None for k in ('cash', 'margin', 'wdl')), 'CENSORED_OUTCOME')
        _require(response['terminal_environment_sha256'] is None, 'CENSORED_OUTCOME')
        return None
    _require(all(len(t) == 719 for t in tapes), 'INCOMPLETE_ACTION_TAPES')
    validation = {'actors': {arm: actor, req['opponent']['ref']: req['opponent']['identity']},
                  'execution': branch['execution']}
    job = {'world': req['world']['id'], 'seat': req['seat'], 'agent': arm, 'opponent': req['opponent']['ref']}
    verdict = panel.validate_game(row, job, validation, _world(req['world']))
    _require(verdict['passed'], 'PANEL_RECEIPT_INVALID', verdict['error'] or '')
    R._hash(response['terminal_environment_sha256'], '$.terminal_environment_sha256')
    _require(panel.canonical(final.plan.canonical()) == panel.canonical(rec[arm]['canonical_plan']),
             'FINAL_PLAN_MISMATCH')
    if native:
        _same(row['physical_games_completed'], 1, 'NATIVE_NOT_COMPLETE')
        epochs = response['epochs']
        _same([e['step'] for e in epochs], list(DAWNS), 'DAWN_COVERAGE')
        for e in epochs:
            _same(e['raw_edits'], rec[arm]['raw_actions'] if e['step'] == 0 else [0], 'LATER_DAWN_NOT_KEEP')
            _require(panel.canonical(e['after_plan']) == panel.canonical(rec[arm]['canonical_plan']), 'EPOCH_PLAN_MISMATCH')
        cp = con['dawn24_checkpoint']
        _closed(cp, ('observation', 'snapshot'), 'MISSING_RESTART_CHECKPOINT')
        restored = P.restore_agent(cp['snapshot'], cp['observation'])
        _same(P._policy_snapshot(restored.policy), con['initial_policy'], 'RESTART_POLICY_MISMATCH')
        _require(cp['snapshot']['step'] == 24, 'RESTART_STEP')
        trace = response['rng_trace']
        _require(type(trace) is list and len(trace) == 29, 'RNG_TRACE_COVERAGE')
        for day, t in enumerate(trace):
            _closed(t, ('step', 'day', 'key', 'pre_state_sha256', 'post_state_sha256'), 'RNG_TRACE_FIELDS')
            _same([t['step'], t['day'], t['key']],
                  [24*(day+1), day, (req['world']['seed']*1000003) ^ day], 'RNG_DAY_KEY_MISMATCH')
            R._hash(t['pre_state_sha256'], '$.rng.pre'); R._hash(t['post_state_sha256'], '$.rng.post')
    seat = req['seat']; cash = row['cash']
    return P.bind({'cash_by_seat': cash, 'own_cash': cash[seat], 'rival_cash': cash[1-seat],
        'margin': row['margin'], 'wdl': {'WIN': 1, 'DRAW': 0, 'LOSS': -1}[row['wdl']],
        'environment_sha256': response['terminal_environment_sha256']})


def _run(branch, runner):
    argument = copy.deepcopy(branch)
    response = None
    try:
        response = runner(argument)
        _same(argument, branch, 'RUNNER_MUTATED_REQUEST')
        terminal = _validate_response(branch, response)
        status = 'COMPLETE' if terminal is not None else 'INCOMPLETE'
        error = None if terminal is not None else 'INCOMPLETE_BRANCH'
    except Exception as exc:
        terminal = None
        status = 'INCOMPLETE' if response is None else 'INVALID'
        error = type(exc).__name__ + ': ' + str(exc)
        try:
            R._safe(response)
        except (R.RecordError, RecursionError):
            response = None  # Unsafe values are not exported as supposedly safe JSON.
    return {'arm': branch['arm'], 'attempt_id': branch['attempt_id'], 'phase': branch['phase'],
            'branch_sha256': branch['content_sha256'], 'status': status,
            'terminal': terminal, 'error': error, 'response': copy.deepcopy(response)}


def _comparison(arms):
    a, b = arms
    result = {'objective': 'TERMINAL_WDL', 'wdl_delta': None, 'own_cash_delta': None,
        'rival_cash_delta': None, 'margin_delta': None, 'joint_actions_equal': None,
        'rival_actions_equal': None, 'terminal_equal': None,
        'realized_challenger_better': False, 'actor_training_admissible': False,
        'verdict': 'CENSORED'}
    if not all(x['status'] == 'COMPLETE' for x in arms):
        if any(x['status'] == 'INVALID' for x in arms): result['verdict'] = 'INVALID'
        return result
    at, bt = a['terminal'], b['terminal']; seat = a['response']['row']['seat']
    delta = bt['wdl'] - at['wdl']
    result.update(wdl_delta=delta, own_cash_delta=bt['own_cash']-at['own_cash'],
        rival_cash_delta=bt['rival_cash']-at['rival_cash'], margin_delta=bt['margin']-at['margin'],
        joint_actions_equal=a['response']['row']['actions_sha256'] == b['response']['row']['actions_sha256'],
        rival_actions_equal=a['response']['row']['actions_by_seat_sha256'][1-seat] ==
                           b['response']['row']['actions_by_seat_sha256'][1-seat],
        terminal_equal=at == bt, realized_challenger_better=delta > 0,
        verdict='CHALLENGER_BETTER' if delta > 0 else 'LEARNER_BETTER' if delta < 0 else 'TIE')
    return result


def _record_result(request, arms, replay, execution, comparison):
    rec = copy.deepcopy(request['record']); group = rec['identity']['source_group_id']
    complete = all(a['status'] == 'COMPLETE' for a in arms)
    verified = replay['status'] == 'VERIFIED'
    world = dict(source_group_id=group, world_sha256=request['world_sha256'],
        snapshot_sha256=request['initial_world_sha256'], rng_sha256=panel.digest(request['rng_binding']),
        seed=request['world']['seed'])
    rec['evaluation'].update(mode='EXACT_REALIZATION', discovery_world_groups=[world],
        confirmation_world_groups=[], complete=complete, replay_verified=verified,
        opponent_hash=request['opponent']['identity']['content_sha256'],
        seat_rows=[dict(source_group_id=group, seat=request['seat'],
            actor_context_sha256=rec['identity']['actor_context_sha256'],
            learner_wdl=arms[0]['terminal']['wdl'] if complete else None,
            challenger_wdl=arms[1]['terminal']['wdl'] if complete else None,
            complete=complete, replay_verified=verified,
            learner_attempt_id=arms[0]['attempt_id'], challenger_attempt_id=arms[1]['attempt_id'])],
        attempt_ids=[a['attempt_id'] for a in [*arms, *replay['arms']]])
    reasons = ['REALIZATION_ONLY']
    if execution['kind'] != 'NATIVE_GAME': reasons.append('SYNTHETIC_DIAGNOSTIC')
    if not complete: reasons.append('INCOMPLETE_OR_INVALID_PAIR')
    if not verified: reasons.append('REPLAY_' + replay['status'])
    rec['comparison'].update(mean_wdl_delta=comparison['wdl_delta'], lower_bound=None, alpha=None,
        independent_world_count=int(complete), verdict=comparison['verdict'],
        actor_training_admissible=False, reason_codes=reasons)
    rec['content_sha256'] = R.canonical_sha256({k: v for k, v in rec.items() if k != 'content_sha256'})
    return R.validate_pair(rec)


def _assemble(request, execution, arms, replay):
    comp = _comparison(arms); attempts = [*arms, *replay['arms']]
    native = execution['kind'] == 'NATIVE_GAME'
    return P.bind({'schema': SCHEMA, 'scope': SCOPE, 'request_sha256': request['content_sha256'],
        'execution': execution, 'arms': arms, 'replay': replay, 'comparison': comp,
        'counts': {'execution_attempts': len(attempts), 'replay_executions': len(replay['arms']),
            'native_starts': len(attempts) if native else 0,
            'native_completed': sum(a['status'] == 'COMPLETE' for a in attempts) if native else 0,
            'independent_native_worlds': 1 if native else 0, 'replay_independent_worlds_added': 0,
            'declared_world_groups': 1, 'direct_policy_draws_added': 0, 'optimizer_updates': 0},
        'record': _record_result(request, arms, replay, execution, comp)})


def evaluate_pair(request: dict, *, runner) -> dict:
    """Execute both arms once, preserving invalid/failing attempts; never retries."""
    request = _validate_request(request)
    execution = execution_binding(runner)
    arms = [_run(_branch(request, arm, execution, 'EVALUATE'), runner) for arm in ARMS]
    return _assemble(request, execution, arms, {'status': 'NOT_RUN', 'arms': [], 'mismatches': []})


def _replay_signature(arm):
    r = arm['response']
    if arm['status'] != 'COMPLETE' or r is None:
        return {'status': arm['status'], 'error': arm['error']}
    # All continuation, observations, source/RNG, per-seat actions and engine state
    # are compared. Only attempt bindings and nonsemantic wall timing are excluded.
    return {k: v for k, v in r.items() if k not in ('branch_sha256', 'content_sha256')}


def verify_pair(request: dict, result: dict, *, runner) -> dict:
    """Fresh execution of each arm. Zero additional independent worlds/draws."""
    request = _validate_request(request)
    _bound(result, 'RESULT_HASH_MISMATCH')
    _require(result.get('request_sha256') == request['content_sha256'], 'RESULT_REQUEST_MISMATCH')
    execution = execution_binding(runner)
    _same(result['execution'], execution, 'REPLAY_RUNNER_MISMATCH')
    _same(result['replay'], {'status': 'NOT_RUN', 'arms': [], 'mismatches': []}, 'REPLAY_ALREADY_ATTEMPTED')
    _require(type(result['arms']) is list and len(result['arms']) == 2 and
             [a.get('arm') for a in result['arms']] == list(ARMS), 'DUPLICATE_OR_MISSING_ARMS')
    _require(len({a['attempt_id'] for a in result['arms']}) == 2, 'DUPLICATE_ATTEMPTS')
    for arm in result['arms']:
        branch = P.bind({'schema': BRANCH_SCHEMA, 'request': request, 'arm': arm['arm'],
                         'execution': execution, 'phase': 'EVALUATE', 'attempt_id': arm['attempt_id']})
        _same(arm['branch_sha256'], branch['content_sha256'], 'RESULT_BRANCH_MISMATCH')
        _same(arm['phase'], 'EVALUATE', 'RESULT_PHASE_MISMATCH')
        if arm['status'] == 'COMPLETE':
            _same(arm['terminal'], _validate_response(branch, arm['response']), 'RESULT_TERMINAL_MISMATCH')
        else:
            _require(arm['terminal'] is None, 'CENSORED_OUTCOME')
    expected = _assemble(request, execution, result['arms'], result['replay'])
    _same(result, expected, 'RESULT_DERIVED_FIELDS_MISMATCH')
    fresh = [_run(_branch(request, arm, execution, 'REPLAY'), runner) for arm in ARMS]
    mismatches = []
    for original, replay in zip(result['arms'], fresh):
        if original['status'] != 'COMPLETE' or replay['status'] != 'COMPLETE':
            mismatches.append({'arm': original['arm'], 'reason': 'INCOMPLETE_OR_INVALID_BRANCH'})
        elif P.wire(_replay_signature(original)) != P.wire(_replay_signature(replay)):
            mismatches.append({'arm': original['arm'], 'reason': 'SEMANTIC_REPLAY_MISMATCH',
                               'original_sha256': P.wire_hash(_replay_signature(original)),
                               'replay_sha256': P.wire_hash(_replay_signature(replay))})
    return _assemble(request, execution, copy.deepcopy(result['arms']),
        {'status': 'MISMATCH' if mismatches else 'VERIFIED', 'arms': fresh, 'mismatches': mismatches})


def _native_game(branch):
    """Native child: one existing-engine game, same receipt contract as router_panel."""
    _bound(branch, 'BRANCH_HASH_MISMATCH')
    req = _validate_request(branch['request']); arm = branch['arm']
    _require(arm in ARMS, 'UNKNOWN_ARM')
    _same(branch['execution'], execution_binding(native_runner), 'NATIVE_EXECUTION_MISMATCH')
    world = _world(req['world']); env = scenarios.ScenarioEnvironment(world.scenario())
    _same(P.wire_hash(_world_snapshot(env)), req['initial_world_sha256'], 'ACTUAL_WORLD_MISMATCH')
    _check_opening(req['record'], env, req['seat'])
    policy = TrajectoryPolicy({0: req['record'][arm]['raw_actions'][:-1]})
    agent = V0Continuation.from_dict(req['record']['continuation']).restore(
        policy, agent_cls=P.PolicyAgent, observation=env.agent_observation(req['seat']))
    rival_spec = panel.load_actor(req['opponent']['ref'])
    _same(rival_spec.identity, req['opponent']['identity'], 'ACTUAL_OPPONENT_MISMATCH')
    rival = rival_spec.spawn()  # Not an action tape copied from the other arm.
    actor = actor_identity(req, arm); seat = req['seat']
    ids = [rival_spec.sha256, rival_spec.sha256]; ids[seat] = actor['content_sha256']
    initial = V0Continuation.capture(agent).to_dict(); initial_policy = P._policy_snapshot(agent.policy)
    tapes = [[], []]; epochs = []; rng_trace = []; cp = None
    row = {'world': world.id, 'world_sha256': world.sha256, 'seed': world.seed, 'seat': seat,
        'agent': arm, 'opponent': rival_spec.name,
        'actor_identity_hashes': {'agent': actor['content_sha256'], 'opponent': rival_spec.sha256},
        'actor_identities_by_seat': ids, 'status': 'FAILED', 'cash': None, 'margin': None,
        'wdl': None, 'error': None, 'steps': 0, 'physical_games_completed': 0,
        'min_rival_cash': None, 'rival_cash_trough': 3000.,
        'evidence_kind': 'NATIVE_GAME', 'execution_sha256': branch['execution']['content_sha256']}
    terminal_hash = None
    deadline = time.monotonic() + req['timeout_seconds']
    try:
        while not env.done:
            step = env.step_index; obs = env.agent_observation(seat)
            if step == 24:
                snapshot = P.snapshot_agent(agent, obs)
                cp = {'observation': _plain(obs), 'snapshot': snapshot}
                restored = P.restore_agent(json.loads(P.wire(snapshot)), obs)
                _same(P.agent_signature(restored), P.agent_signature(agent), 'IN_PROCESS_RESTART_MISMATCH')
                agent = restored
            calls = [panel._callable(rival), panel._callable(rival)]; calls[seat] = agent
            joint = [panel.public_guard(call, seat=s, deadline=deadline)(
                env.agent_observation(s), env.configuration) for s, call in enumerate(calls)]
            if step in DAWNS:
                e = agent.records[-1]
                epochs.append({'step': step, 'raw_edits': e['raw_edits'], 'after_plan': e['after_plan'],
                               'compiled_program_sha256': P.wire_hash(e['compiled_program']),
                               'refusals': e['refusals']})
            for s, action in enumerate(joint): tapes[s].append(_plain(action))
            rng_event = (step+1) % 24 == 0
            before = P.wire_hash(_world_snapshot(env)) if rng_event else None
            env.step(joint)
            if rng_event:
                day = step // 24
                rng_trace.append({'step': env.step_index, 'day': day,
                    'key': (world.seed*1000003) ^ day, 'pre_state_sha256': before,
                    'post_state_sha256': P.wire_hash(_world_snapshot(env))})
            row['rival_cash_trough'] = min(row['rival_cash_trough'],
                                          float(env.agent_observation(seat)['farms'][1-seat]['money']))
        _require(env.step_index == 719 and all(r.status == 'DONE' for r in env.state), 'INCOMPLETE_NATIVE_GAME')
        cash = [float(r.reward) for r in env.state]; margin = cash[seat]-cash[1-seat]
        _require(all(math.isfinite(c) for c in cash), 'NONFINITE_TERMINAL')
        row.update(status='COMPLETE', cash=cash, margin=margin, wdl=panel._wdl(margin),
                   min_rival_cash=cash[1-seat], physical_games_completed=1)
        terminal_hash = P.wire_hash(env.state)
    except Exception as exc:
        row.update(status='TIMEOUT' if isinstance(exc, TimeoutError) else 'FAILED',
                   error=type(exc).__name__ + ': ' + str(exc), cash=None, margin=None, wdl=None,
                   physical_games_completed=0)
    row.update(steps=env.step_index, actions_sha256=panel.digest(tapes),
               actions_by_seat_sha256=[panel.digest(t) for t in tapes])
    final_observation = env.agent_observation(seat)
    agent.actor_state(final_observation, initialize=False)
    return P.bind(_plain({'schema': BRANCH_RESULT_SCHEMA, 'branch_sha256': branch['content_sha256'],
        'source_pins_sha256': panel.digest(req['source_pins']), 'row': row,
        'initial_world_sha256': req['initial_world_sha256'],
        'actor_context_sha256': req['record']['identity']['actor_context_sha256'],
        'rng_binding': req['rng_binding'], 'actions_by_seat': tapes, 'final_observation': final_observation,
        'continuation': {'initial': initial, 'initial_policy': initial_policy,
            'final': V0Continuation.capture(agent).to_dict(), 'final_policy': P._policy_snapshot(agent.policy),
            'dawn24_checkpoint': cp}, 'epochs': epochs, 'rng_trace': rng_trace,
        'terminal_environment_sha256': terminal_hash}))


def native_runner(branch: dict) -> dict:
    """Hard-deadline fresh interpreter per arm/replay, with no cached outcomes."""
    with tempfile.TemporaryDirectory(prefix='v0-regret-pair-') as directory:
        path = Path(directory); request = path/'branch.json'; output = path/'response.json'
        request.write_text(P.wire(branch)+'\n', encoding='utf-8')
        inherited_pythonpath = [part for part in os.environ.get('PYTHONPATH', '').split(os.pathsep) if part]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
                   PYTHONPATH=os.pathsep.join([str(ROOT/'src'), *inherited_pythonpath]),
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        cmd = [sys.executable, '-B', '-m', MODULE, '_game', '--request', str(request), '--output', str(output)]
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=branch['request']['timeout_seconds']+10.)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.communicate()
            raise TimeoutError('fresh-process branch deadline; no retry')
        if proc.returncode:
            raise RuntimeError(f'native worker exit {proc.returncode}: {stderr[-3000:]} {stdout[-500:]}')
        return _read(output)


_NATIVE_RUNNER = native_runner


# Explicit opt-in experiment. None of the TERMINAL_WDL v1 functions above use it.
MARGIN_SCHEMA = 'farm-v0-fixed-shop-margin-v1'
MARGIN_OBJECTIVE = 'PAIRED_TERMINAL_MARGIN_V1'
MARGIN_REQUEST_SCHEMA = MARGIN_SCHEMA + '-request'
MARGIN_RESULT_SCHEMA = MARGIN_SCHEMA + '-execution'
MARGIN_BRANCH_SCHEMA = MARGIN_SCHEMA + '-branch'
MARGIN_RESPONSE_SCHEMA = MARGIN_SCHEMA + '-arm'
MARGIN_ARMS = ('old', 'new')
MARGIN_RULE = dict(version=1, world_count=16, seats=[0, 1], pair_count=32,
    primary_games=64, fresh_replay_games=64, minimum_positive_worlds=10,
    mean_delta_strictly_positive=True, minimum_seat_mean=0.,
    positive_pair_teacher_weight=1., nonpositive_pair_teacher_weight=0.,
    loss='OFF_POLICY_SUPERVISED_CLASS_PATH', promotion_objective='TERMINAL_WDL',
    statistical_support=False)
MARGIN_RIVAL = 'FRESH_LIVE_BC_GREEDY_CLASS_ALL_DAWNS'


def margin_source_pins():
    paths = ('src/farm_neat/training/v0_regret_admission.py',
             'src/farm_neat/training/v0_regret_train.py',
             'src/farm_neat/training/teachers/v0_regret_proposals.py')
    return {**source_pins(), **{p: panel.file_sha(ROOT / p) for p in paths}}


def margin_worlds() -> list[dict]:
    """The first sixteen registered pair64 tapes, chosen without reading outcomes."""
    return [_plain(asdict(w)) for w in panel.world_registry().worlds if w.cohort == 'pair64'][:16]


def _margin_world_set(documents):
    _require(type(documents) is list and len(documents) == 16, 'MARGIN_SIXTEEN_WORLDS_REQUIRED')
    ids, tapes = set(), set()
    for document in documents:
        _require(type(document) is dict, 'MARGIN_WORLD_FIELDS')
        _require(document.get('random_weeds') is False,
                 'MARGIN_ACTION_DEPENDENT_WEED_DRAWS_REFUSED')
        world = _world(document)
        tape = tuple(world.shops)
        _require(world.id not in ids and tape not in tapes, 'MARGIN_DUPLICATE_WORLD_OR_TAPE')
        ids.add(world.id); tapes.add(tape)
    return documents


def _margin_checkpoint(ref, *, device, production):
    _closed(ref, ('path', 'commit_sha256'), 'MARGIN_CHECKPOINT_FIELDS')
    _require(type(ref['path']) is str and bool(ref['path']), 'MARGIN_CHECKPOINT_PATH')
    R._hash(ref['commit_sha256'], '$.checkpoint.commit_sha256')
    info = P.inspect_committed_scorer(ROOT / ref['path'], ref['commit_sha256'],
                                      device=device, production=production)
    return dict(ref=copy.deepcopy(ref), identity=copy.deepcopy(info['identity']),
                training_source_pins=copy.deepcopy(info['training_identity'].get('source_pins', {})))


def _margin_load(checkpoint, *, device, production):
    ref = checkpoint['ref']
    loaded = P.load_committed_scorer(ROOT / ref['path'], ref['commit_sha256'],
                                     device=device, production=production)
    _same(loaded.identity, checkpoint['identity'], 'MARGIN_LOADED_CHECKPOINT_MISMATCH')
    return loaded


def _margin_context(observation, *, game_id):
    agent = P.PolicyAgent(TrajectoryPolicy({}), game_id=game_id, labor_factor=4.,
                          hands_mode='daily', belief={p: 0 for p in V.PRODUCTS})
    agent.actor_state(observation, initialize=False)
    memory = V0Continuation.capture(agent).to_dict()
    context = dict(schema=R.ACTOR_SCHEMA, observation=observation_facts(observation),
        prefix=R.A.Prefix(agent.plan).payload(),
        compiler_memory={k: copy.deepcopy(v) for k, v in memory.items() if k != 'game_id'})
    return context, memory


def _margin_sample(loaded, observation, memory, *, seed, draw_id):
    """Actual scorer call on the native opening State; no world/labels enter it."""
    from farm_neat.training.teachers import v0_regret_proposals as Q
    policy = loaded.policy('sample_raw', seed, temperature=1., opening_only=True)
    agent = V0Continuation.from_dict(memory).restore(policy, agent_cls=P.PolicyAgent,
                                                     observation=observation)
    state = agent.actor_state(observation, initialize=False)
    options = agent.projection_kw()
    before = P.wire_hash(policy.generator.get_state().tolist())
    actions = list(policy(state, agent.plan.copy(), P.Masks(state, agent.plan, **options)))
    after = P.wire_hash(policy.generator.get_state().tolist())
    trace = copy.deepcopy(policy.last_trace)
    geometry = Q._trace(state, R.A.Prefix(agent.plan), options, {}, actions)
    _require(geometry['status'] == 'REACHABLE', 'MARGIN_UNREACHABLE_SAMPLE')
    sample = {k: copy.deepcopy(geometry[k]) for k in
              ('prefix_class_trace', 'canonical_plan', 'continuation_signature')}
    sample.update(raw_actions=actions, direct_draw_id=draw_id,
        raw_behavior_logp=trace['raw_path_logp'], sampling_provenance=dict(
            type='direct_policy', mode='sample_raw', checkpoint_sha256=loaded.identity['commit_sha256'],
            temperature=1., raw_step_logps=[r['raw_logp'] for r in trace['steps']],
            rng_before_sha256=before, rng_after_sha256=after))
    sample['path_sha256'] = R.canonical_sha256(sample)
    return dict(sampler_seed=seed, sample=sample, decode_trace=trace)


def validate_margin_cell(cell, checkpoints, *, geometry=True, menu_cache=None):
    """Preserve both genuine draws; selected new targets never acquire on-policy status."""
    _bound(cell, 'MARGIN_CELL_HASH_MISMATCH')
    _closed(cell, ('world_id', 'world_sha256', 'seat', 'initial_world_sha256',
        'actor_context', 'actor_context_sha256', 'continuation', 'continuation_sha256',
        'opponent_seed', 'old', 'new', 'content_sha256'), 'MARGIN_CELL_FIELDS')
    _require(type(cell['seat']) is int and cell['seat'] in (0, 1), 'MARGIN_SEAT')
    _same(P.wire_hash(cell['actor_context']), cell['actor_context_sha256'], 'MARGIN_CONTEXT_HASH_MISMATCH')
    _same(P.wire_hash(cell['continuation']), cell['continuation_sha256'], 'MARGIN_COMPILER_HASH_MISMATCH')
    state, prefix, options = R.validate_pair_v1_entry_prefix(cell['actor_context'], cell['continuation'])
    _require(state.player == cell['seat'], 'MARGIN_CONTEXT_SEAT_MISMATCH')
    menus = {} if menu_cache is None else menu_cache
    for arm in MARGIN_ARMS:
        opening = cell[arm]
        _closed(opening, ('sampler_seed', 'sample', 'decode_trace'), 'MARGIN_OPENING_FIELDS')
        R._int(opening['sampler_seed'], '$.sampler_seed', 0, 2**63-1)
        sample, trace = opening['sample'], opening['decode_trace']
        if geometry:
            R._proposal(sample, dict(parent_checkpoint_sha256=checkpoints[arm]['identity']['commit_sha256']),
                        prefix, state, options, menus, direct=True)
        else:
            _same(sample['path_sha256'], R.canonical_sha256({k: v for k, v in sample.items()
                  if k != 'path_sha256'}), 'MARGIN_SAMPLE_HASH_MISMATCH')
        # Model manifests are canonical JSON metadata, unlike ordered engine maps.
        _same(R.canonical_sha256(trace['model_identity']),
              R.canonical_sha256(checkpoints[arm]['identity']['model_identity']), 'MARGIN_READOUT_MISMATCH')
        feature_version = ('v3' if trace['model_identity'].get('feature_schema_sha256') ==
                           R.A.FEATURE_SCHEMA_SHA256_V3 else 'v2')
        _require(trace['schema'] == R.A.TRACE_SCHEMA, 'MARGIN_TRACE_SCHEMA')
        _same(trace['state_sha256'], R.A._state_digest(state, feature_version), 'MARGIN_TRACE_STATE_MISMATCH')
        _same(trace['projection_sha256'], R.A._digest(R.A._projection_options(options)),
              'MARGIN_TRACE_PROJECTION_MISMATCH')
        _require(trace['class_path_not_final_plan_marginal'] is True, 'MARGIN_PATH_LIKELIHOOD_SCOPE')
        _require(trace['mode'] == 'sample_raw' and trace['temperature'] == 1.
                 and sample['sampling_provenance']['temperature'] == 1., 'MARGIN_SAMPLE_MODE')
        for actual, expected in ((trace['actions'], sample['raw_actions']),
                (trace['final_plan'], sample['canonical_plan']),
                (trace['initial_prefix'], cell['actor_context']['prefix']),
                (trace['raw_path_logp'], sample['raw_behavior_logp']),
                ([r['raw_logp'] for r in trace['steps']], sample['sampling_provenance']['raw_step_logps'])):
            _same(actual, expected, 'MARGIN_TRACE_SAMPLE_MISMATCH')
        _require(len(trace['steps']) == len(sample['prefix_class_trace']), 'MARGIN_TRACE_GEOMETRY_MISMATCH')
        for actual, expected in zip(trace['steps'], sample['prefix_class_trace']):
            _require(actual['schema'] == R.A.STEP_SCHEMA and actual['mode'] == 'sample_raw'
                     and actual['temperature'] == 1., 'MARGIN_TRACE_STEP_MODE')
            _same(actual['behavior_logp'], actual['raw_logp'], 'MARGIN_BEHAVIOR_LIKELIHOOD_MISMATCH')
            _same({k: actual[k] for k in expected if k != 'schema'},
                  {k: v for k, v in expected.items() if k != 'schema'}, 'MARGIN_TRACE_GEOMETRY_MISMATCH')
    return cell


def make_margin_request(*, objective, checkpoints, world_ids=None, sampler_seed=0,
                        device='cuda', production=True, timeout_seconds=900.) -> dict:
    """Authenticate all three commits and prepare all 64 openings BEFORE any game.

    No fitting or environment steps. This can load/sample production actors and
    therefore must be run by the maintainer under the registered SLURM runtime.
    """
    _require(objective == MARGIN_OBJECTIVE, 'MARGIN_OBJECTIVE_OPT_IN')
    _closed(checkpoints, ('old', 'new', 'opponent'), 'MARGIN_THREE_CHECKPOINTS_REQUIRED')
    _require(type(production) is bool, 'MARGIN_PRODUCTION_FLAG')
    _require(type(device) is str and (device == 'cpu' or device == 'cuda' or device.startswith('cuda:')),
             'MARGIN_DEVICE')
    R._int(sampler_seed, '$.sampler_seed', 0, 2**63-97)
    _require(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds)
             and 0 < timeout_seconds <= 3600, 'MARGIN_TIMEOUT')
    registry = panel.world_registry()
    documents = margin_worlds() if world_ids is None else [_plain(asdict(registry[w])) for w in world_ids]
    _margin_world_set(documents)
    frozen = {role: _margin_checkpoint(checkpoints[role], device=device, production=production)
              for role in ('old', 'new', 'opponent')}
    _require(frozen['opponent']['identity']['kind'] == 'BC', 'MARGIN_BC_ONLY_OPPONENT_REQUIRED')
    pins = margin_source_pins()
    loaded = {arm: _margin_load(frozen[arm], device=device, production=production) for arm in MARGIN_ARMS}
    worlds, cells = [], []
    for i, document in enumerate(documents):
        world = _world(document); env = scenarios.ScenarioEnvironment(world.scenario())
        snapshot = _world_snapshot(env)
        worlds.append(dict(world=document, world_sha256=world.sha256, initial_snapshot=snapshot,
            initial_world_sha256=P.wire_hash(snapshot), shop_tape_sha256=P.wire_hash(list(world.shops))))
        for seat in (0, 1):
            obs = env.agent_observation(seat)
            context, memory = _margin_context(obs, game_id=f'margin-{world.id}-{seat}')
            cell = dict(world_id=world.id, world_sha256=world.sha256, seat=seat,
                initial_world_sha256=P.wire_hash(snapshot), actor_context=context,
                actor_context_sha256=P.wire_hash(context), continuation=memory,
                continuation_sha256=P.wire_hash(memory), opponent_seed=sampler_seed+64+2*i+seat)
            for j, arm in enumerate(MARGIN_ARMS):
                cell[arm] = _margin_sample(loaded[arm], obs, memory,
                    seed=sampler_seed+2*i+seat, draw_id=f'margin-{world.id}-{seat}-{arm}')
            cells.append(P.bind(cell))
    request = P.bind(dict(schema=MARGIN_REQUEST_SCHEMA, objective=objective,
        objective_version=1, registration_stage='ALL_IDENTITIES_AND_OPENINGS_FROZEN_BEFORE_OUTCOMES',
        rule=copy.deepcopy(MARGIN_RULE), worlds=worlds, cells=cells, checkpoints=frozen,
        source_pins=pins, sampler_seed=sampler_seed, device=device, production=production,
        timeout_seconds=float(timeout_seconds), learner_policy='FROZEN_SAMPLED_OPENING_THEN_KEEP',
        opponent_policy=MARGIN_RIVAL, rng=dict(random_weeds=False, weed_draw_alignment='DISABLED',
            learner_stream='COMMON_INITIAL_STREAM_PER_WORLD_SEAT', common_fixed_shop_tapes=True, engine_rng='DAY_KEYED_PYTHON_RANDOM'),
        promotion_objective='TERMINAL_WDL', strength_claim=False))
    validate_margin_request(request)
    return request


def validate_margin_request(request, *, authenticate=True, geometry=True):
    _bound(request, 'MARGIN_REQUEST_HASH_MISMATCH')
    _closed(request, ('schema', 'objective', 'objective_version', 'registration_stage', 'rule',
        'worlds', 'cells', 'checkpoints', 'source_pins', 'sampler_seed', 'device', 'production',
        'timeout_seconds', 'learner_policy', 'opponent_policy', 'rng', 'promotion_objective',
        'strength_claim', 'content_sha256'), 'MARGIN_REQUEST_FIELDS')
    _require(request['schema'] == MARGIN_REQUEST_SCHEMA and request['objective'] == MARGIN_OBJECTIVE
             and type(request['objective_version']) is int and request['objective_version'] == 1,
             'MARGIN_OBJECTIVE_OPT_IN')
    _same(request['rule'], MARGIN_RULE, 'MARGIN_RULE_DRIFT')
    _same(request['source_pins'], margin_source_pins(), 'MARGIN_SOURCE_MISMATCH')
    _require(request['registration_stage'] == 'ALL_IDENTITIES_AND_OPENINGS_FROZEN_BEFORE_OUTCOMES'
             and request['promotion_objective'] == 'TERMINAL_WDL' and request['strength_claim'] is False,
             'MARGIN_REGISTRATION_SCOPE')
    _require(request['learner_policy'] == 'FROZEN_SAMPLED_OPENING_THEN_KEEP'
             and request['opponent_policy'] == MARGIN_RIVAL, 'MARGIN_POLICY_MODE')
    _same(request['rng'], dict(random_weeds=False, weed_draw_alignment='DISABLED',
        learner_stream='COMMON_INITIAL_STREAM_PER_WORLD_SEAT', common_fixed_shop_tapes=True, engine_rng='DAY_KEYED_PYTHON_RANDOM'), 'MARGIN_WEED_OR_SHOP_MISMATCH')
    _require(type(request['production']) is bool, 'MARGIN_PRODUCTION_FLAG')
    _require(type(request['timeout_seconds']) in (int, float) and
             math.isfinite(request['timeout_seconds']) and 0 < request['timeout_seconds'] <= 3600, 'MARGIN_TIMEOUT')
    R._int(request['sampler_seed'], '$.sampler_seed', 0, 2**63-97)
    _closed(request['checkpoints'], ('old', 'new', 'opponent'), 'MARGIN_THREE_CHECKPOINTS_REQUIRED')
    for role, checkpoint in request['checkpoints'].items():
        _closed(checkpoint, ('ref', 'identity', 'training_source_pins'), 'MARGIN_CHECKPOINT_DESCRIPTOR')
        _require(checkpoint['identity']['commit_sha256'] == checkpoint['ref']['commit_sha256'],
                 'MARGIN_CHECKPOINT_MISMATCH')
        if authenticate:
            _same(checkpoint, _margin_checkpoint(checkpoint['ref'], device=request['device'],
                  production=request['production']), 'MARGIN_CHECKPOINT_MISMATCH')
    _require(request['checkpoints']['opponent']['identity']['kind'] == 'BC', 'MARGIN_BC_ONLY_OPPONENT_REQUIRED')
    _margin_world_set([w['world'] for w in request['worlds']])
    by_world = {}
    for item in request['worlds']:
        _closed(item, ('world', 'world_sha256', 'initial_snapshot', 'initial_world_sha256',
                       'shop_tape_sha256'), 'MARGIN_WORLD_FIELDS')
        world = _world(item['world']); env = scenarios.ScenarioEnvironment(world.scenario())
        _same(item['world_sha256'], world.sha256, 'MARGIN_WORLD_MISMATCH')
        _same(item['initial_snapshot'], _world_snapshot(env), 'MARGIN_INITIAL_SNAPSHOT_MISMATCH')
        _same(item['initial_world_sha256'], P.wire_hash(item['initial_snapshot']), 'MARGIN_SNAPSHOT_HASH_MISMATCH')
        _same(item['shop_tape_sha256'], P.wire_hash(list(world.shops)), 'MARGIN_SHOP_MISMATCH')
        by_world[world.id] = (item, env)
    _require(type(request['cells']) is list and len(request['cells']) == 32, 'MARGIN_CELL_COVERAGE')
    expected = [(w['world']['id'], seat) for w in request['worlds'] for seat in (0, 1)]
    _same([(c['world_id'], c['seat']) for c in request['cells']], expected, 'MARGIN_CELL_ORDER_OR_COVERAGE')
    context_menus = {}  # local to this exact source-bound validation; never cached across requests
    for n, cell in enumerate(request['cells']):
        item, env = by_world[cell['world_id']]
        _same(cell['world_sha256'], item['world_sha256'], 'MARGIN_CELL_WORLD_MISMATCH')
        _same(cell['initial_world_sha256'], item['initial_world_sha256'], 'MARGIN_CELL_SNAPSHOT_MISMATCH')
        actual_context, actual_memory = _margin_context(env.agent_observation(cell['seat']),
            game_id=f"margin-{cell['world_id']}-{cell['seat']}")
        _same(cell['actor_context'], actual_context, 'MARGIN_ACTUAL_CONTEXT_MISMATCH')
        _same(cell['continuation'], actual_memory, 'MARGIN_COMPILER_MEMORY_MISMATCH')
        _same(cell['opponent_seed'], request['sampler_seed']+64+n, 'MARGIN_OPPONENT_STREAM_MISMATCH')
        for j, arm in enumerate(MARGIN_ARMS):
            _same(cell[arm]['sampler_seed'], request['sampler_seed']+n, 'MARGIN_LEARNER_STREAM_MISMATCH')
            _same(cell[arm]['sample']['direct_draw_id'], f"margin-{cell['world_id']}-{cell['seat']}-{arm}",
                  'MARGIN_DRAW_ID_MISMATCH')
        validate_margin_cell(cell, request['checkpoints'], geometry=geometry,
                             menu_cache=context_menus.setdefault(cell['actor_context_sha256'], {}))
    return request


def margin_execution_binding(runner):
    native = runner is _MARGIN_NATIVE_RUNNER
    fp = panel._fingerprint(_margin_native_game if native else runner)
    if native: fp['module'] = MODULE
    return P.bind(dict(kind='NATIVE_GAME' if native else 'SYNTHETIC_DIAGNOSTIC', runner=fp,
        python=sys.version, fresh_process_per_attempt=native,
        trust='SOURCE_BOUND_TRUSTED_RUNNER_NOT_A_SANDBOX'))


def _margin_branch(request, cell, arm, phase, execution, attempt_id=None):
    return P.bind(dict(schema=MARGIN_BRANCH_SCHEMA, request=request,
        cell_sha256=cell['content_sha256'], world_id=cell['world_id'], seat=cell['seat'],
        arm=arm, phase=phase, execution=execution, attempt_id=attempt_id or 'margin-'+uuid.uuid4().hex))


def _margin_bindings(request, cell, arm):
    world = next(w for w in request['worlds'] if w['world']['id'] == cell['world_id'])
    return dict(request_sha256=request['content_sha256'], cell_sha256=cell['content_sha256'],
        source_pins_sha256=P.wire_hash(request['source_pins']), world_sha256=cell['world_sha256'],
        initial_world_sha256=cell['initial_world_sha256'], actor_context_sha256=cell['actor_context_sha256'],
        continuation_sha256=cell['continuation_sha256'], shop_tape_sha256=world['shop_tape_sha256'],
        learner_checkpoint_sha256=P.wire_hash(request['checkpoints'][arm]),
        opponent_checkpoint_sha256=P.wire_hash(request['checkpoints']['opponent']),
        opening_sha256=P.wire_hash(cell[arm]), rng=copy.deepcopy(request['rng']),
        opponent_policy=MARGIN_RIVAL, opponent_seed=cell['opponent_seed'])


def _margin_check_response(branch, response):
    request = branch['request']
    cell = next(c for c in request['cells'] if c['content_sha256'] == branch['cell_sha256'])
    _bound(response, 'MARGIN_RESPONSE_HASH_MISMATCH')
    _closed(response, ('schema', 'branch_sha256', 'bindings', 'complete', 'steps', 'terminal',
        'actions_by_seat', 'shop_history', 'initial_memories', 'final_memories', 'final_policies',
        'publications', 'error', 'content_sha256'), 'MARGIN_RESPONSE_FIELDS')
    _require(response['schema'] == MARGIN_RESPONSE_SCHEMA, 'MARGIN_RESPONSE_SCHEMA')
    _same(response['branch_sha256'], branch['content_sha256'], 'MARGIN_RESPONSE_BRANCH_MISMATCH')
    _same(response['bindings'], _margin_bindings(request, cell, branch['arm']), 'MARGIN_EXECUTION_JOIN_MISMATCH')
    _require(type(response['complete']) is bool and type(response['steps']) is int
             and 0 <= response['steps'] <= 719, 'MARGIN_COMPLETION_FIELDS')
    actions = response['actions_by_seat']
    _require(type(actions) is list and len(actions) == 2 and
             all(type(t) is list and len(t) == response['steps'] for t in actions), 'MARGIN_ACTION_COVERAGE')
    for tape in actions:
        for action in tape:
            _require(type(action) is dict and set(action) == {'farmer', 'hands', 'market'} and
                     all(type(action[k]) is list for k in action), 'MARGIN_ACTION_FIELDS')
    for key in ('initial_memories', 'final_memories', 'final_policies', 'publications'):
        _require(type(response[key]) is list and len(response[key]) == 2, 'MARGIN_CONTROLLER_COVERAGE')
    seat = cell['seat']
    _same(response['initial_memories'][seat], cell['continuation'], 'MARGIN_INITIAL_COMPILER_MISMATCH')
    rival_memory = dict(cell['continuation'], game_id=f"margin-bc-{cell['world_id']}-{seat}")
    _same(response['initial_memories'][1-seat], rival_memory, 'MARGIN_INITIAL_RIVAL_MEMORY_MISMATCH')
    _same(response['final_policies'][seat], dict(kind='TRAJECTORY',
        schedule=[[0, cell[branch['arm']]['sample']['raw_actions'][:-1]]]), 'MARGIN_LEARNER_KEEP_POLICY_MISMATCH')
    rival = response['final_policies'][1-seat]
    _require(type(rival) is dict and rival.get('kind') == 'SCORER', 'MARGIN_LIVE_RIVAL_POLICY_REQUIRED')
    state = rival['state']
    _same(state['identity'], request['checkpoints']['opponent']['identity'], 'MARGIN_RIVAL_CHECKPOINT_MISMATCH')
    _require(state['mode'] == 'greedy_class' and type(state['seed']) is int
             and state['seed'] == cell['opponent_seed'] and state['temperature'] == 1.,
             'MARGIN_RIVAL_MODE_OR_STREAM_MISMATCH')
    _require(type(state['generator_state']) is list and bool(state['generator_state']) and
             all(type(x) is int and 0 <= x < 256 for x in state['generator_state']), 'MARGIN_RIVAL_RNG_STATE')
    for memory in response['final_memories']:
        V0Continuation.from_dict(memory)  # structural own-history/Plan join, not a state rebind
    for player, publications in enumerate(response['publications']):
        _require(type(publications) is list and [r['step'] for r in publications] ==
                 [d for d in DAWNS if d < response['steps']], 'MARGIN_PUBLICATION_COVERAGE')
        for publication in publications:
            _closed(publication, ('step', 'raw_actions', 'after_plan', 'compiled_program_sha256',
                'decode_trace_sha256', 'refusals'), 'MARGIN_PUBLICATION_FIELDS')
            R._hash(publication['compiled_program_sha256'], '$.compiled_program_sha256')
            R._hash(publication['decode_trace_sha256'], '$.decode_trace_sha256')
            if player == seat:
                _same(publication['raw_actions'], cell[branch['arm']]['sample']['raw_actions']
                      if publication['step'] == 0 else [V.STOP_ID], 'MARGIN_OPENING_THEN_KEEP_MISMATCH')
                _same(publication['after_plan'], cell[branch['arm']]['sample']['canonical_plan'],
                      'MARGIN_LEARNER_PLAN_MISMATCH')
            else:
                _require(publication['decode_trace_sha256'] != P.wire_hash(None), 'MARGIN_LIVE_BC_TRACE_REQUIRED')
    world = next(w for w in request['worlds'] if w['world']['id'] == cell['world_id'])['world']
    _same(response['shop_history'], world['shops'][:min(8, response['steps']//72)], 'MARGIN_ACTUAL_SHOP_MISMATCH')
    if response['complete']:
        _require(response['steps'] == 719 and response['error'] is None, 'MARGIN_CENSORED_ARM')
        terminal = response['terminal']
        _closed(terminal, ('cash', 'margin', 'wdl', 'environment_sha256'), 'MARGIN_TERMINAL_FIELDS')
        cash = terminal['cash']; seat = cell['seat']
        _require(type(cash) is list and len(cash) == 2 and
                 all(type(c) in (int, float) and math.isfinite(c) for c in cash), 'MARGIN_NONFINITE_CASH')
        margin = cash[seat]-cash[1-seat]
        _require(math.isfinite(margin), 'MARGIN_NONFINITE_MARGIN')
        _same(terminal['margin'], margin, 'MARGIN_CASH_LABEL_MISMATCH')
        _require(type(terminal['wdl']) is str and terminal['wdl'] == panel._wdl(margin), 'MARGIN_WDL_LABEL_MISMATCH')
        R._hash(terminal['environment_sha256'], '$.terminal.environment_sha256')
    else:
        _require(response['terminal'] is None and type(response['error']) is str and bool(response['error']),
                 'MARGIN_INCOMPLETE_TERMINAL_LABEL')
    return response


def _margin_attempt(request, cell, arm, phase, execution, runner, *, attempt_id=None):
    branch = _margin_branch(request, cell, arm, phase, execution, attempt_id)
    out = dict(attempt_id=branch['attempt_id'], world_id=cell['world_id'], seat=cell['seat'],
        arm=arm, phase=phase, status='INDETERMINATE', response=None, error=None)
    try:
        # Keep malformed/partial returns too. Never retry or censor an unfavorable arm.
        response = runner(copy.deepcopy(branch))
        R._safe(response)
        out['response'] = copy.deepcopy(response)
        _margin_check_response(branch, response)
        out['status'] = 'COMPLETE' if response['complete'] else 'INDETERMINATE'
        out['error'] = response['error']
    except Exception as exc:
        out['error'] = getattr(exc, 'code', type(exc).__name__) + ': ' + str(exc)
    return P.bind(out)


def _margin_semantics(response):
    return {k: v for k, v in response.items() if k not in ('branch_sha256', 'content_sha256')}


def _margin_result(request, execution, primary, replay):
    """Recompute all coverage, per-arm joins and exact replay; no trusted summaries."""
    _bound(execution, 'MARGIN_EXECUTION_HASH')
    native = execution.get('kind') == 'NATIVE_GAME'
    if native: _same(execution, margin_execution_binding(_MARGIN_NATIVE_RUNNER), 'MARGIN_NATIVE_RUNNER_MISMATCH')
    else: _require(execution.get('kind') == 'SYNTHETIC_DIAGNOSTIC'
                   and execution.get('fresh_process_per_attempt') is False, 'MARGIN_EXECUTION_KIND')
    cells = {(c['world_id'], c['seat']): c for c in request['cells']}
    expected = {(w, s, arm) for w, s in cells for arm in MARGIN_ARMS}
    maps, valid, attempts, reasons = {}, {}, set(), []
    for phase, rows in (('EVALUATE', primary), ('REPLAY', replay)):
        _require(type(rows) is list, 'MARGIN_ATTEMPT_LIST')
        by_key, good = {}, set()
        if len(rows) != 64: reasons.append('MARGIN_'+phase+'_MISSING_OR_EXTRA_ARMS')
        for row in rows:
            try:
                _bound(row, 'MARGIN_ATTEMPT_HASH')
                _closed(row, ('attempt_id', 'world_id', 'seat', 'arm', 'phase', 'status', 'response',
                               'error', 'content_sha256'), 'MARGIN_ATTEMPT_FIELDS')
                _require(type(row['seat']) is int and row['seat'] in (0, 1), 'MARGIN_ATTEMPT_SEAT')
                key = (row['world_id'], row['seat'], row['arm'])
                _require(key in expected and row['phase'] == phase, 'MARGIN_SUBSTITUTED_ARM')
                _require(key not in by_key, 'MARGIN_DUPLICATE_ARM')
                _require(type(row['attempt_id']) is str and row['attempt_id']
                         and row['attempt_id'] not in attempts, 'MARGIN_REUSED_ATTEMPT')
                by_key[key] = row; attempts.add(row['attempt_id'])
                cell = cells[key[:2]]
                branch = _margin_branch(request, cell, row['arm'], phase, execution, row['attempt_id'])
                _margin_check_response(branch, row['response'])
                _require(row['status'] == 'COMPLETE' and row['response']['complete'] is True
                         and row['error'] is None, 'MARGIN_CENSORED_OR_FAILED_ARM')
                good.add(key)
            except (ValueError, KeyError, TypeError, StopIteration, OverflowError) as exc:
                reasons.append(getattr(exc, 'code', 'MARGIN_MALFORMED_ARM'))
        if set(by_key) != expected: reasons.append('MARGIN_'+phase+'_COVERAGE')
        maps[phase], valid[phase] = by_key, good
    pairs = []
    for (world_id, seat), cell in cells.items():
        keys = [(world_id, seat, arm) for arm in MARGIN_ARMS]
        complete = all(k in valid['EVALUATE'] for k in keys)
        replayed = complete and all(k in valid['REPLAY'] and
            _margin_semantics(maps['EVALUATE'][k]['response']) == _margin_semantics(maps['REPLAY'][k]['response'])
            for k in keys)
        if complete and all(k in valid['REPLAY'] for k in keys) and not replayed:
            reasons.append('MARGIN_REPLAY_MISMATCH')
        terminals = [maps['EVALUATE'][k]['response']['terminal'] if k in valid['EVALUATE'] else None for k in keys]
        delta = terminals[1]['margin']-terminals[0]['margin'] if complete else None
        if delta is not None and not math.isfinite(delta):
            reasons.append('MARGIN_NONFINITE_DELTA'); complete = replayed = False; delta = None
        pairs.append(dict(world_id=world_id, seat=seat, cell_sha256=cell['content_sha256'],
            complete=complete, replay_verified=replayed, delta=delta,
            old_wdl=terminals[0]['wdl'] if terminals[0] else None,
            new_wdl=terminals[1]['wdl'] if terminals[1] else None,
            old_terminal=terminals[0], new_terminal=terminals[1],
            rival_actions_equal=(maps['EVALUATE'][keys[0]]['response']['actions_by_seat'][1-seat] ==
                                maps['EVALUATE'][keys[1]]['response']['actions_by_seat'][1-seat]) if complete else None))
    status = 'VERIFIED' if not reasons and all(p['replay_verified'] for p in pairs) else 'INDETERMINATE'
    if not replay and all(p['complete'] for p in pairs) and len(primary) == 64:
        status = 'UNVERIFIED'
    return P.bind(dict(schema=MARGIN_RESULT_SCHEMA, objective=MARGIN_OBJECTIVE,
        request_sha256=request['content_sha256'], execution=copy.deepcopy(execution), status=status,
        primary=copy.deepcopy(primary), replay=copy.deepcopy(replay), pairs=pairs,
        reason_codes=sorted(set(reasons)), counts=dict(primary_attempts=len(primary), replay_attempts=len(replay),
            primary_physical_games=len(valid['EVALUATE']) if native else 0,
            replay_physical_games=len(valid['REPLAY']) if native else 0,
            paired_worlds=16, paired_seats=32, replay_verified_pairs=sum(p['replay_verified'] for p in pairs)),
        promotion_objective='TERMINAL_WDL', statistical_support=False, strength_claim=False))


def validate_margin_result(request, result):
    _bound(result, 'MARGIN_RESULT_HASH')
    _require(result.get('schema') == MARGIN_RESULT_SCHEMA and result.get('objective') == MARGIN_OBJECTIVE,
             'MARGIN_RESULT_SCHEMA')
    _same(result.get('request_sha256'), request['content_sha256'], 'MARGIN_RESULT_REQUEST_MISMATCH')
    expected = _margin_result(request, result['execution'], result['primary'], result['replay'])
    _same(result, expected, 'MARGIN_RESULT_RECOMPUTATION_MISMATCH')
    return result


def _margin_run_phase(request, phase, execution, runner, journal=None):
    # CLI journals are exclusive and fsynced before AND after each attempt. A
    # killed job leaves its completed outcomes and an unclosed START, not a
    # deceptively complete panel. There is deliberately no retry/resume path.
    stream = None
    def record(value):
        if stream is not None:
            stream.write(P.wire(value)+'\n'); stream.flush(); os.fsync(stream.fileno())
    try:
        if journal is not None:
            target = Path(journal); target.parent.mkdir(parents=True, exist_ok=True)
            stream = target.open('x', encoding='utf-8')
            record(dict(schema=MARGIN_SCHEMA+'-journal', request_sha256=request['content_sha256'],
                        execution=execution, phase=phase))
        rows = []
        for cell in request['cells']:
            for arm in MARGIN_ARMS:
                attempt_id = 'margin-'+uuid.uuid4().hex
                record(dict(event='START', attempt_id=attempt_id, world_id=cell['world_id'],
                            seat=cell['seat'], arm=arm, phase=phase))
                row = _margin_attempt(request, cell, arm, phase, execution, runner, attempt_id=attempt_id)
                rows.append(row); record(dict(event='RESULT', attempt=row))
        return rows
    finally:
        if stream is not None: stream.close()


def evaluate_margin_experiment(request, *, runner=None, journal=None):
    runner = _MARGIN_NATIVE_RUNNER if runner is None else runner
    validate_margin_request(request)
    execution = margin_execution_binding(runner)
    primary = _margin_run_phase(request, 'EVALUATE', execution, runner, journal)
    return _margin_result(request, execution, primary, [])


def verify_margin_experiment(request, result, *, runner=None, journal=None):
    runner = _MARGIN_NATIVE_RUNNER if runner is None else runner
    validate_margin_request(request); validate_margin_result(request, result)
    _require(not result['replay'], 'MARGIN_FRESH_REPLAY_ALREADY_RECORDED')
    _same(result['execution'], margin_execution_binding(runner), 'MARGIN_REPLAY_RUNNER_MISMATCH')
    replay = _margin_run_phase(request, 'REPLAY', result['execution'], runner, journal)
    return _margin_result(request, result['execution'], result['primary'], replay)


def _margin_opponent(loaded, cell):
    # Never share a sampler, controller or action tape across learner arms.
    _require(loaded.identity['kind'] == 'BC', 'MARGIN_BC_ONLY_OPPONENT_REQUIRED')
    return P.PolicyAgent(loaded.policy('greedy_class', cell['opponent_seed'], opening_only=False),
        game_id=f"margin-bc-{cell['world_id']}-{cell['seat']}", labor_factor=4., hands_mode='daily',
        belief={p: 0 for p in V.PRODUCTS})


def _margin_joint_actions(agents, env, deadline):
    # Each opponent call sees THAT arm's current own/public observation. In
    # particular a different learner action may cause a different BC response.
    return [panel.public_guard(agent, seat=seat, deadline=deadline)(
            env.agent_observation(seat), env.configuration) for seat, agent in enumerate(agents)]


def _margin_native_game(branch):
    """One fresh existing-engine game. Rival is live, stateful and never a tape."""
    _bound(branch, 'MARGIN_BRANCH_HASH')
    _require(branch['schema'] == MARGIN_BRANCH_SCHEMA and branch['arm'] in MARGIN_ARMS
             and branch['phase'] in ('EVALUATE', 'REPLAY'), 'MARGIN_BRANCH_FIELDS')
    request = validate_margin_request(branch['request'], geometry=False)
    _same(branch['execution'], margin_execution_binding(_MARGIN_NATIVE_RUNNER), 'MARGIN_NATIVE_RUNNER_MISMATCH')
    cell = next(c for c in request['cells'] if c['content_sha256'] == branch['cell_sha256'])
    _require((branch['world_id'], branch['seat']) == (cell['world_id'], cell['seat']), 'MARGIN_BRANCH_CELL_MISMATCH')
    world = _world(next(w['world'] for w in request['worlds'] if w['world']['id'] == cell['world_id']))
    env = scenarios.ScenarioEnvironment(world.scenario()); seat = cell['seat']; arm = branch['arm']
    _same(P.wire_hash(_world_snapshot(env)), cell['initial_world_sha256'], 'MARGIN_ACTUAL_SNAPSHOT_MISMATCH')
    kwargs = dict(device=request['device'], production=request['production'])
    own = _margin_load(request['checkpoints'][arm], **kwargs)
    opening = _margin_sample(own, env.agent_observation(seat), cell['continuation'],
        seed=cell[arm]['sampler_seed'], draw_id=cell[arm]['sample']['direct_draw_id'])
    _same(opening, cell[arm], 'MARGIN_AUTHENTIC_OPENING_REPLAY_MISMATCH')
    rival = _margin_load(request['checkpoints']['opponent'], **kwargs)
    _require(rival.identity['kind'] == 'BC', 'MARGIN_BC_ONLY_OPPONENT_REQUIRED')
    learner = V0Continuation.from_dict(cell['continuation']).restore(
        TrajectoryPolicy({0: opening['sample']['raw_actions'][:-1]}), agent_cls=P.PolicyAgent,
        observation=env.agent_observation(seat))
    opponent = _margin_opponent(rival, cell)
    agents = [opponent, opponent]; agents[seat] = learner
    initial = [V0Continuation.capture(a).to_dict() for a in agents]
    tapes, publications = [[], []], [[], []]
    error, terminal = None, None
    deadline = time.monotonic()+request['timeout_seconds']
    try:
        while not env.done:
            step = env.step_index
            joint = _margin_joint_actions(agents, env, deadline)
            for s, action in enumerate(joint):
                tapes[s].append(_plain(action))
                if step in DAWNS:
                    record = agents[s].records[-1]
                    publications[s].append(dict(step=step, raw_actions=record['raw_edits'],
                        after_plan=record['after_plan'], compiled_program_sha256=P.wire_hash(record['compiled_program']),
                        decode_trace_sha256=P.wire_hash(record.get('decode_trace')), refusals=record['refusals']))
            env.step(joint)
        _require(env.step_index == 719 and all(r.status == 'DONE' for r in env.state), 'MARGIN_NATIVE_INCOMPLETE')
        cash = [float(r.reward) for r in env.state]
        _require(all(math.isfinite(c) for c in cash), 'MARGIN_NONFINITE_CASH')
        margin = cash[seat]-cash[1-seat]
        terminal = dict(cash=cash, margin=margin, wdl=panel._wdl(margin), environment_sha256=P.wire_hash(env.state))
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
    # Failed transition attempts may have emitted a joint action without completing
    # that step. Preserve it separately by refusing the response coverage join;
    # the raw response still survives in the outer INDETERMINATE attempt receipt.
    return P.bind(_plain(dict(schema=MARGIN_RESPONSE_SCHEMA, branch_sha256=branch['content_sha256'],
        bindings=_margin_bindings(request, cell, arm), complete=terminal is not None,
        steps=env.step_index, terminal=terminal, actions_by_seat=tapes,
        shop_history=env.agent_observation(seat)['town']['unlocked_shops'],
        initial_memories=initial, final_memories=[V0Continuation.capture(a).to_dict() for a in agents],
        final_policies=[P._policy_snapshot(a.policy) for a in agents], publications=publications, error=error)))


def margin_native_runner(branch):
    """Fresh interpreter and fresh BC controller per primary/replay, no retries."""
    with tempfile.TemporaryDirectory(prefix='v0-margin-') as directory:
        request, output = Path(directory)/'branch.json', Path(directory)/'response.json'
        request.write_text(P.wire(branch)+'\n', encoding='utf-8')
        inherited = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
            PYTHONPATH=os.pathsep.join([str(ROOT/'src'), *inherited]),
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        command = [sys.executable, '-B', '-m', MODULE, '_margin-game',
                   '--request', str(request), '--output', str(output)]
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=branch['request']['timeout_seconds']+120.)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.communicate()
            raise TimeoutError('fresh margin worker deadline; no retry')
        if proc.returncode:
            raise RuntimeError(f'margin worker exit {proc.returncode}: {stderr[-3000:]} {stdout[-500:]}')
        return _read(output)


_MARGIN_NATIVE_RUNNER = margin_native_runner


def _margin_write_json(path, value):
    """Exclusive exact-schema JSON, including lists and canonical admission seals."""
    target = Path(path)
    _require(not any(p.is_symlink() for p in (target, *target.parents)), 'MARGIN_SYMLINKED_OUTPUT')
    encoded = P.wire(value)  # refuses nonfinite/non-JSON values before creating a file
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        stream.write(encoded+'\n'); stream.flush(); os.fsync(stream.fileno())


def _read(path):
    return R._loads(Path(path).read_text(encoding='utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('request', 'evaluate', 'verify', '_game',
        'margin-worlds', 'margin-request', 'margin-evaluate', 'margin-verify', '_margin-game'))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--record', type=Path)
    parser.add_argument('--world', default='router-dairy')
    parser.add_argument('--opponent', choices=OPPONENTS, default='router')
    parser.add_argument('--result', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=120.)
    parser.add_argument('--objective', choices=(MARGIN_OBJECTIVE,))
    parser.add_argument('--checkpoints', type=Path, help='old/new/opponent refs: path and commit_sha256')
    parser.add_argument('--worlds', type=Path, help='preregistered JSON world documents or IDs')
    parser.add_argument('--sampler-seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--diagnostic', action='store_true', help='never admits experimental weights')
    parser.add_argument('--journal', type=Path, help='exclusive durable per-arm journal (margin only)')
    args = parser.parse_args(argv)
    try:
        _require(not args.output.exists(), 'OUTPUT_EXISTS')
        if args.command == 'request':
            value = make_request(_read(args.record), world_id=args.world,
                                 opponent=args.opponent, timeout_seconds=args.timeout)
        elif args.command == '_game':
            value = _native_game(_read(args.request))
        elif args.command == 'evaluate':
            value = evaluate_pair(_read(args.request), runner=native_runner)
        elif args.command == 'verify':
            value = verify_pair(_read(args.request), _read(args.result), runner=native_runner)
        elif args.command == 'margin-worlds':
            value = margin_worlds()
        elif args.command == 'margin-request':
            _require(args.objective == MARGIN_OBJECTIVE, 'MARGIN_OBJECTIVE_OPT_IN')
            documents = None if args.worlds is None else _read(args.worlds)
            world_ids = None if documents is None else [w['id'] if isinstance(w, dict) else w for w in documents]
            if documents is not None and all(isinstance(w, dict) for w in documents):
                _margin_world_set(documents)  # do not silently fix modified tapes/weed flags
            value = make_margin_request(objective=args.objective, checkpoints=_read(args.checkpoints),
                world_ids=world_ids, sampler_seed=args.sampler_seed, device=args.device,
                production=not args.diagnostic, timeout_seconds=args.timeout)
        elif args.command == '_margin-game':
            value = _margin_native_game(_read(args.request))
        elif args.command == 'margin-evaluate':
            value = evaluate_margin_experiment(_read(args.request),
                journal=args.journal or Path(str(args.output)+'.journal.jsonl'))
        else:
            value = verify_margin_experiment(_read(args.request), _read(args.result),
                journal=args.journal or Path(str(args.output)+'.journal.jsonl'))
        if args.command == 'margin-worlds':
            _margin_write_json(args.output, value)
        else:
            P.write_new(args.output, value)
        if args.command in ('margin-evaluate', 'margin-verify'):
            print(json.dumps({'output': str(args.output), 'counts': value['counts'],
                              'status': value['status'], 'reason_codes': value['reason_codes']}))
            return int(value['status'] != ('UNVERIFIED' if args.command == 'margin-evaluate' else 'VERIFIED'))
        if args.command in ('evaluate', 'verify'):
            print(json.dumps({'output': str(args.output), 'counts': value['counts'],
                'comparison': value['comparison'], 'replay_status': value['replay']['status']}))
            return int(not all(a['status'] == 'COMPLETE' for a in value['arms']) or
                       (args.command == 'verify' and value['replay']['status'] != 'VERIFIED'))
        return 0
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(json.dumps({'status': 'INVALID', 'error': type(exc).__name__ + ': ' + str(exc)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
