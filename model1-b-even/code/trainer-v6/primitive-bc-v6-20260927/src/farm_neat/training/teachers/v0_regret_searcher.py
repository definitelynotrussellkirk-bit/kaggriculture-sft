"""Bounded, restartable V0 discovery. Selection is NEVER teacher admission.

``improve(context, learner, *, proposer, evaluator, budget, protocol, seed)``
starts with a validated genuine direct draw. ``proposer(actor_context,
incumbent_geometry, *, budget, seed)`` returns ONE ordinary raw path, or a
stable UNKNOWN refusal. It receives no outcomes, protocol, hidden execution
context, or likelihoods. In iterative mode its incumbent can change after a
strict terminal-WDL improvement; one-shot-fixed always proposes from the learner.
Every complete path is replayed from the ORIGINAL supplied canonical Prefix,
with native masks/classes recomputed after every edit. No stopped Prefix is
reopened and no candidate is installed in the engine by this module.

``evaluator(request)`` is a trusted injected single-start executor. Requests bind
one preregistered discovery group, seat, opponent, starting continuation, actor
context, candidate, sources, and timeout. It returns a request-hash-bound result:
status, terminal_wdl (-1/0/1), fresh_terminal_execution, realization_verified,
replay_only, simulator_starts=1, simulator_steps, and optional diagnostics.
A complete fresh realized terminal execution alone contributes. Replays,
predictions, errors and partial groups do not. This is an integration contract,
not authentication of a callback's claims or a sandbox for its closure. Native
executors must enforce the one-start limit themselves; replays need separate
charged requests, not unreported work inside a callback. The CLI is fixture-only.

The frozen rule averages WDL within predeclared source families and weights
families equally. All candidates need ALL the same groups; exact ties keep the
incumbent. Cash is retained only as diagnostics. A hash cannot establish external
registration chronology, independence, or undisclosed held-out exposure.

Hard logical caps: iterations, attempted candidates (INCLUDING learner,
duplicates and refusals; <=32 for the existing record seam), discovery groups,
and attempted simulator starts. Wall time is active time, including validation,
proposal/replay/evaluation and checkpoint work up to the last accounting cut;
paused process downtime is excluded. Callback deadlines use POSIX main-thread
alarms and fail closed where unavailable/occupied. Trusted Python extensions
must not swallow the alarm or launch uncontrolled children; use an externally
bounded process for such executors. No wall-time-based optimization occurs.

Optional budget checkpoint_path enables atomic fsync/replace checkpoints under
an exclusive OS lock. Intent and start charges are durable BEFORE callbacks.
Resume never retries a charged in-flight call: unknown outcomes are retained as
CENSORED and their reserved deadline is conservatively charged (exact lost CPU/
wall work is explicitly UNKNOWN). Completed calls and incumbents are not rerolled.
pause_after_evaluations is an invocation-only checkpoint cadence, not a renewed
budget. The file, sources, protocol, learner and caps are joined on every resume.

``challenger`` has the EXISTING v0_regret_records selected-path schema, including
R.SELECTION_SCHEMA and a null confirmation registration. The collector must
freeze its disjoint registration, assemble real pair records, then call the
existing admission gate. This module cannot set an actor label. ``best`` and
``best_so_far`` remain useful when no challenger is certified or budget expires.
``compare_searches`` emits four independently budgeted controls with identical
DECLARED caps/groups. Actual and unused costs remain separate; a duplicate or
learner-only control does not pretend it spent the unused allowance.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import threading
import time

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.evaluation.v0_continuation import V0Continuation
from farm_neat.training import v0_regret_records as R
from farm_neat.training import v0_targets as T
from farm_neat.training.teachers import v0_regret_proposals as P

SCHEMA = 'farm-v0-regret-search-v1'
PROTOCOL_SCHEMA = 'farm-v0-regret-discovery-protocol-v1'
MODES = ('learner', 'one-shot-random', 'one-shot-fixed', 'iterative')
DEFAULT_BUDGET = dict(max_iterations=8, max_candidates=16, max_discovery_groups=4,
                      max_simulator_starts=64, max_wall_seconds=60.,
                      evaluation_timeout_seconds=5., max_path_edits=4)
GEOMETRY = ('candidate_id', 'raw_actions', 'prefix_class_trace', 'canonical_plan',
            'continuation_signature', 'full_continuation_signature')
SOLVER_FAMILIES = ('shop_response', 'production', 'market_loop')
SOURCE_PATHS = (*P.SOURCE_PATHS, 'src/farm_neat/training/teachers/v0_regret_searcher.py',
    'src/farm_neat/training/v0_regret_records.py', 'src/farm_neat/training/v0_regret_admission.py',
    'src/farm_neat/evaluation/v0_regret_pairs.py', 'src/farm_neat/strategy/types.py',
    *(f'src/farm_neat/strategy/{name}.py' for name in SOLVER_FAMILIES))


def _need(ok, code):
    if not ok:
        raise ValueError(code)


def _hash(value):
    return R.canonical_sha256(value)


def _seal(value, field='content_sha256'):
    value[field] = _hash({k: v for k, v in value.items() if k != field})
    return value


def _visible(candidate):
    return copy.deepcopy({k: candidate[k] for k in GEOMETRY})


def _limits(value):
    R._fields(value, (), '$.budget', optional=(*DEFAULT_BUDGET, 'checkpoint_path', 'pause_after_evaluations'))
    out = DEFAULT_BUDGET | {k: v for k, v in value.items() if k in DEFAULT_BUDGET}
    for key, lo, hi in (('max_iterations', 0, 31), ('max_candidates', 1, 32),
                       ('max_discovery_groups', 1, 256), ('max_simulator_starts', 0, 8192),
                       ('max_path_edits', 1, 16)):
        R._int(out[key], '$.budget.' + key, lo, hi)
    R._number(out['max_wall_seconds'], '$.budget.max_wall_seconds', 0, 86400)
    R._number(out['evaluation_timeout_seconds'], '$.budget.evaluation_timeout_seconds', 0, 86400)
    _need(out['evaluation_timeout_seconds'] > 0, 'INVALID_EVALUATION_TIMEOUT')
    path = value.get('checkpoint_path')
    if path is not None:
        R._text(path, '$.budget.checkpoint_path')
    pause = value.get('pause_after_evaluations')
    if pause is not None:
        R._int(pause, '$.budget.pause_after_evaluations', 1, 8192)
        _need(path is not None, 'PAUSE_REQUIRES_DURABLE_CHECKPOINT')
    return out, None if path is None else Path(path), pause


def _protocol(value, budget):
    R._safe(value)
    R._fields(value, ('schema', 'objective', 'aggregation', 'discovery_world_groups',
        'protected_world_groups', 'source_families', 'opponent_sha256', 'continuation_sha256',
        'proposer_sha256', 'evaluator_sha256', 'evidence_scope', 'execution_context'),
        '$.protocol', optional=('mode',))
    _need(value['schema'] == PROTOCOL_SCHEMA, 'PROTOCOL_SCHEMA')
    _need(value['objective'] == 'TERMINAL_WDL' and
          value['aggregation'] == 'EQUAL_SOURCE_FAMILY_MEAN_V1', 'FROZEN_WDL_RULE_REQUIRED')
    for key in ('opponent_sha256', 'continuation_sha256', 'proposer_sha256', 'evaluator_sha256'):
        R._hash(value[key], '$.protocol.' + key)
    _need(value['evidence_scope'] in ('INJECTED_FIXTURE', 'CALLER_ATTESTED_EXECUTION'), 'EVIDENCE_SCOPE')
    _need(type(value['execution_context']) is dict, 'EXECUTION_CONTEXT')
    groups = R._list(value['discovery_world_groups'], '$.protocol.discovery_world_groups')
    protected = R._list(value['protected_world_groups'], '$.protocol.protected_world_groups')
    _need(0 < len(groups) <= budget['max_discovery_groups'], 'DISCOVERY_GROUP_CAP')
    families = value['source_families']
    _need(type(families) is dict, 'SOURCE_FAMILIES_REQUIRED')
    identity_keys = ('source_group_id', 'world_sha256', 'snapshot_sha256', 'rng_sha256')
    seen = {key: set() for key in identity_keys}
    used_families = set()
    for phase, rows in (('discovery', groups), ('protected', protected)):
        for group in rows:
            R._fields(group, (*identity_keys, 'seed'), '$.protocol.world_group')
            R._text(group['source_group_id'], '$.protocol.source_group_id')
            for key in identity_keys[1:]:
                R._hash(group[key], '$.protocol.' + key)
            R._int(group['seed'], '$.protocol.seed', 0, 2**64 - 1)
            family = families.get(group['source_group_id'])
            R._text(family, '$.protocol.source_family')
            if phase == 'protected':
                _need(not any(group[k] in seen[k] for k in identity_keys) and family not in used_families,
                      'PROTECTED_DISCOVERY_OVERLAP')
            else:
                _need(not any(group[k] in seen[k] for k in identity_keys), 'DUPLICATE_DISCOVERY_GROUP')
                for key in identity_keys:
                    seen[key].add(group[key])
                used_families.add(family)
    mode = value.get('mode', 'iterative')
    _need(mode in MODES, 'UNSUPPORTED_MODE')
    return copy.deepcopy(value) | {'mode': mode}


@contextmanager
def _lock(path):
    if path is None:
        yield
        return
    # Advisory lock is held throughout this invocation and released on process death.
    try:
        import fcntl
    except ImportError as exc:
        raise ValueError('CHECKPOINT_LOCK_UNSUPPORTED') from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + '.lock').open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('CHECKPOINT_BUSY') from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write(path, state):
    if path is None:
        return
    payload = json.dumps(_seal(state), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class _Clock:
    def __init__(self):
        self.wall, self.cpu = time.monotonic(), time.process_time()

    def tick(self, state):
        now, cpu = time.monotonic(), time.process_time()
        delta = max(0., now - self.wall)
        state['counts']['wall_seconds_observed'] += delta
        state['counts']['wall_seconds_charged'] += delta
        state['counts']['cpu_seconds_observed'] += max(0., cpu - self.cpu)
        self.wall, self.cpu = now, cpu


class _Deadline(TimeoutError):
    pass


@contextmanager
def _deadline(seconds):
    _need(seconds > 0, 'WALL_CAP')
    _need(hasattr(signal, 'setitimer') and threading.current_thread() is threading.main_thread(),
          'CALLBACK_DEADLINE_UNSUPPORTED')
    _need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'CALLBACK_DEADLINE_OCCUPIED')
    old = signal.getsignal(signal.SIGALRM)
    def expired(_sig, _frame):
        raise _Deadline('CALLBACK_TIMEOUT')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, old)


def _full_candidate_continuation(context, state, final_plan, decoder):
    """One owner for the exact compiler/HOLD lift of a replayed successor."""
    agent = R._memory({**context['compiler_memory'], 'game_id': 'proposal-context'}, '$.compiler_memory')
    agent.bind_holds(state, agent.plan, final_plan)
    return dict(decoder=decoder, starting_compiler_memory_sha256=_hash(context['compiler_memory']),
                hold_releases_after_binding=V0Continuation.capture(agent).to_dict()['hold_releases'])


def _candidate(context, spec, state, prefix, options, menus, *, edit_limit):
    """Reuse the accepted syntax/native-replay owner; never trust supplied geometry."""
    R._safe(spec)
    R._fields(spec, ('proposal_family',), '$.candidate', optional=(
        'raw_actions', 'target_plan', 'status', 'reason_codes', 'provenance'))
    R._text(spec['proposal_family'], '$.candidate.proposal_family')
    if spec.get('status') == 'UNKNOWN':
        return None, spec.get('reason_codes') or ['UNSUPPORTED_SOLVER_PROPOSAL']
    actions = spec.get('raw_actions')
    screen = P._syntax_screen(prefix, actions)
    if screen['status'] == 'INVALID':
        return None, screen['reason_codes']
    if len(actions) - 1 > edit_limit:
        return None, ['PATH_SEARCH_BUDGET']
    replay = P._trace(state, prefix, options, menus, actions)
    if replay['status'] != 'REACHABLE':
        return None, replay['reason_codes']
    final = replay['final']
    if 'target_plan' in spec and not R._equal(spec['target_plan'], replay['canonical_plan']):
        return None, ['TARGET_MISMATCH']
    continuation = _full_candidate_continuation(context, state, final.plan, replay['continuation_signature'])
    candidate = dict(schema=P.CANDIDATE_SCHEMA,
        candidate_id=_hash([_hash(context), replay['canonical_plan'], continuation]),
        raw_actions=copy.deepcopy(actions), prefix_class_trace=replay['prefix_class_trace'],
        canonical_plan=replay['canonical_plan'], continuation_signature=replay['continuation_signature'],
        full_continuation_signature=continuation, proposal_family=spec['proposal_family'],
        provenance=copy.deepcopy(spec.get('provenance', {})), raw_behavior_logp=None,
        sampling_provenance={'type': 'search_selection'}, actor_training_admissible=False,
        discovery_score=None, reachability=dict(status='REACHABLE', target_exact=True,
            compiler_executed=False, scope='FROM_SUPPLIED_PREFIX_ONLY'))
    return candidate, []


def random_proposer(context, incumbent, *, budget, seed):
    state, prefix, options = R._actor(context)
    return dict(proposal_family='random_legal_control', raw_actions=P._random_path(
        state, prefix, options, {}, random.Random(seed), budget['max_path_edits']),
        provenance=dict(generator_seed=seed, sampling='UNIFORM_LEGAL_RAW_IDS'))


def fixed_proposer(context, incumbent, *, budget, seed):
    """Fixed round-robin of accepted mutation/class/random and strict solver adapters."""
    trial = budget['iteration']
    family = ('incumbent_mutation', 'theorem_conditioned', 'random_legal_control', *SOLVER_FAMILIES)[trial % 6]
    if family in SOLVER_FAMILIES:
        return solver_candidate(context, incumbent, family=family, budget=budget, seed=seed)
    if family == 'random_legal_control':
        return random_proposer(context, incumbent, budget=budget, seed=seed)
    if family == 'incumbent_mutation':
        actions, origin = P._mutate(incumbent['raw_actions'], random.Random(seed), trial // 6)
    else:
        state, prefix, options = R._actor(context)
        actions = P._random_path(state, prefix, options, {}, random.Random(seed),
                                 budget['max_path_edits'], classes=True)
        origin = {'evidence': P.PARENTS['prefix_classes'], 'sampling': 'UNIFORM_PREFIX_CLASSES'}
    return dict(proposal_family=family, raw_actions=actions, provenance=origin)


def lower_solver(context, incumbent, proposal):
    """Strict Proposal-to-V0 adapter. Estimates never select or certify a path."""
    from farm_neat.strategy.types import Proposal
    R._actor(context)
    _need(isinstance(proposal, Proposal), 'SOLVER_PROPOSAL_TYPE')
    family = proposal.module
    unknown = dict(proposal_family='solver:' + family, status='UNKNOWN',
        reason_codes=['UNSUPPORTED_SOLVER_FAMILY'], provenance={'solver_proposal': proposal.to_dict()})
    if family not in SOLVER_FAMILIES:
        return unknown
    if family == 'production':
        unknown['reason_codes'] = ['PRODUCTION_COHORT_TAPE_NOT_V0_PATH']
        return unknown
    if family == 'market_loop':
        unknown['reason_codes'] = ['MARKET_QUEUE_NOT_V0_PATH']
        return unknown
    payload = proposal.candidate
    required = {'kind', 'family', 'base_plan_sha256', 'edits', 'target_plan', 'ordered_shops', 'new_shops'}
    if type(payload) is not dict or set(payload) != required or payload.get('kind') != 'v0_edit':
        unknown['reason_codes'] = ['UNSUPPORTED_SHOP_RESPONSE_PAYLOAD']
        return unknown
    if payload['base_plan_sha256'] != R._plan(incumbent['canonical_plan'], '$.incumbent').digest():
        unknown['reason_codes'] = ['STALE_SOLVER_BASE_PLAN']
        return unknown
    if payload['ordered_shops'] != context['observation']['shops']:
        unknown['reason_codes'] = ['STALE_SOLVER_PUBLIC_SHOPS']
        return unknown
    edits = payload['edits']
    if (type(edits) is not list or not edits or edits[-1] != 0 or 0 in edits[:-1]
            or any(type(a) is not int or not 0 <= a < V.N_ACTIONS for a in edits)):
        unknown['reason_codes'] = ['MALFORMED_SOLVER_V0_EDITS']
        return unknown
    changed = {V.plan_key(a): a for a in incumbent['raw_actions'] if a != 0}
    for action in edits[:-1]:
        changed[V.plan_key(action)] = action
    spec = dict(proposal_family=unknown['proposal_family'], raw_actions=[changed[k] for k in sorted(changed)] + [0],
                target_plan=payload['target_plan'], provenance=unknown['provenance'])
    state, prefix, options = R._actor(context)
    candidate, reasons = _candidate(context, spec, state, prefix, options, {}, edit_limit=16)
    if candidate is None:
        unknown['reason_codes'] = reasons
        return unknown
    return spec


def solver_candidate(context, incumbent, *, family, budget, seed):
    """Invoke three accepted families on ONLY facts actually present in the actor schema.

    The projection record lacks tile geometry, carried custody and production
    history. They are omitted, not synthesized from counts. This makes refusal
    useful: market queues/cohort tapes must not become secret V0 actions.
    """
    from farm_neat.strategy.types import StrategyContext
    from farm_neat.strategy.shop_response import ShopResponse
    from farm_neat.strategy.production import ProductionReplanner
    from farm_neat.strategy.market_loop import MarketLoop
    R._actor(context)
    _need(family in SOLVER_FAMILIES, 'UNSUPPORTED_SOLVER_FAMILY')
    facts = copy.deepcopy(context['observation'])
    observation = dict(player=facts['player'], step=facts['step'],
        private={'shed': facts['stock'], 'seeds': facts['seeds']},
        market={'inventory': facts['inventory'], 'params': facts['params']},
        town={'unlocked_shops': facts['shops']})
    actor = StrategyContext(facts['step'], facts['player'], observation,
                            memory={'v0_plan': copy.deepcopy(incumbent['canonical_plan'])},
                            shops=tuple(facts['shops']))
    if family == 'shop_response':
        proposals = ShopResponse().propose(actor)
        reason = 'SHOP_RESPONSE_REQUIRES_NATIVE_ACTOR_GEOMETRY'
    elif family == 'production':
        proposals = ProductionReplanner(max_changes=1).propose(actor)
        reason = 'PRODUCTION_BOOK_NOT_IN_ACTOR_CONTEXT'
    else:
        proposals = MarketLoop(max_comparisons=1, max_unit_quotes=1).propose(actor)
        reason = 'MARKET_QUEUE_NOT_V0_PATH'
    if not proposals:
        return dict(proposal_family='solver:' + family, status='UNKNOWN', reason_codes=[reason],
                    provenance={'adapter': family, 'information_scope': 'ACTOR_FACTS_ONLY'})
    # No value/ranking information chooses the solver candidate.
    chosen = random.Random(seed).choice(sorted(proposals, key=lambda p: p.decision_key))
    return lower_solver(context, incumbent, chosen)


def _validate_response(request, response):
    R._safe(response)
    R._fields(response, ('request_sha256', 'status', 'terminal_wdl', 'fresh_terminal_execution',
        'realization_verified', 'replay_only', 'simulator_starts', 'simulator_steps'),
        '$.response', optional=('diagnostics',))
    _need(response['request_sha256'] == request['request_sha256'], 'EVALUATION_BINDING_MISMATCH')
    _need(response['simulator_starts'] == 1 and type(response['simulator_starts']) is int,
          'SINGLE_START_CONTRACT_VIOLATION')
    R._int(response['simulator_steps'], '$.response.simulator_steps', 0)
    for flag in ('fresh_terminal_execution', 'realization_verified', 'replay_only'):
        R._bool(response[flag], '$.response.' + flag)
    if response['status'] != 'COMPLETE':
        _need(response['status'] in ('FAILED', 'CENSORED', 'REPLAY_ONLY'), 'EVALUATION_STATUS')
        return response['status'], 'NONTERMINAL_OR_REPLAY_EVIDENCE'
    if not response['fresh_terminal_execution'] or not response['realization_verified'] or response['replay_only']:
        return 'INDETERMINATE', 'FRESH_REALIZED_TERMINAL_EXECUTION_REQUIRED'
    R._int(response['terminal_wdl'], '$.response.terminal_wdl', -1, 1)
    return 'COMPLETE', 'VERIFIED_DISCOVERY_EXECUTION_ONLY'


def _score(candidate_id, evaluations, protocol):
    rows = [e for e in evaluations if e['candidate_id'] == candidate_id]
    groups = protocol['discovery_world_groups']
    if len(rows) != len(groups) or any(e['status'] != 'COMPLETE' for e in rows):
        return None
    values = {}
    for row in rows:
        family = protocol['source_families'][row['group_id']]
        values.setdefault(family, []).append(row['response']['terminal_wdl'])
    return math.fsum(math.fsum(v)/len(v) for v in values.values()) / len(values)


def _lookup(state, candidate_id):
    return next(c for c in state['candidates'] if c['candidate_id'] == candidate_id)


def _trace_best(state, reason):
    best = _lookup(state, state['incumbent_id'])
    state['best_so_far'].append(dict(after_candidate_attempt=len(state['proposal_attempts']) - 1,
        attempted_simulator_starts=state['counts']['attempted_simulator_starts'],
        candidate_id=best['candidate_id'], score=best['discovery_score'], reason=reason))


def _finish_candidate(state, protocol):
    candidate = _lookup(state, state['current_id'])
    value = _score(candidate['candidate_id'], state['evaluations'], protocol)
    candidate['discovery_score'] = value
    attempt = state['proposal_attempts'][-1]
    if candidate['candidate_id'] == state['learner_id']:
        reason = 'LEARNER_CONTROL' if value is not None else 'INDETERMINATE_LEARNER_CONTROL'
        state['incumbent_transitions'].append(dict(candidate_id=candidate['candidate_id'],
            parent_candidate_id=None, score=value, iteration=0, reachable=True))
    else:
        best = _lookup(state, state['incumbent_id'])
        if value is not None and best['discovery_score'] is not None and value > best['discovery_score']:
            reason = 'VERIFIED_DISCOVERY_IMPROVEMENT'
            state['incumbent_transitions'].append(dict(candidate_id=candidate['candidate_id'],
                parent_candidate_id=state['incumbent_id'], score=value,
                iteration=state['counts']['iterations'], reachable=True))
            state['incumbent_id'] = candidate['candidate_id']
        elif value is None or best['discovery_score'] is None:
            reason = 'INDETERMINATE_NO_PROMOTION'
        elif value == best['discovery_score']:
            reason = 'WDL_TIE_KEEP_INCUMBENT'
        else:
            reason = 'WDL_LOSS_KEEP_INCUMBENT'
    attempt.update(status='EVALUATED', reason_codes=[reason], discovery_score=value)
    _trace_best(state, reason)
    state['current_id'] = None
    state['phase'] = 'PROPOSE'


def _finish(state, reason):
    if state['current_id'] is not None:
        state['proposal_attempts'][-1].update(status='CENSORED', reason_codes=[reason])
        _trace_best(state, reason)
    if not state['best_so_far']:
        _trace_best(state, reason)
    state.update(phase='DONE', stop_reason=reason)


def _result(state):
    best = _lookup(state, state['incumbent_id'])
    pool_hash = _hash(dict(binding=state['binding'], candidates=state['candidates'],
        proposal_attempts=state['proposal_attempts'], evaluations=state['evaluations']))
    receipt = dict(schema=R.SELECTION_SCHEMA, actor_context_sha256=_hash(state['actor_context']),
        parent_checkpoint_sha256=state['learner_draw']['sampling_provenance']['checkpoint_sha256'],
        raw_actions_sha256=_hash(best['raw_actions']), continuation_signature=best['continuation_signature'],
        pool_sha256=pool_hash, confirmation_registration_sha256=None)
    challenger = {k: copy.deepcopy(best[k]) for k in ('raw_actions', 'prefix_class_trace',
                   'canonical_plan', 'continuation_signature')}
    challenger.update(raw_behavior_logp=None, sampling_provenance={'type': 'search_selection'},
        proposal_family=best['proposal_family'], parent_proposal_ids=[state['learner_draw']['direct_draw_id']],
        search_budget=dict(max_candidates=state['budget']['max_candidates'],
            generated_candidates=state['counts']['candidates'], cpu_seconds=state['counts']['cpu_seconds_observed']),
        selection_receipt=receipt, selection_receipt_sha256=_hash(receipt))
    _seal(challenger, 'path_sha256')
    result = dict(schema=SCHEMA, status='SELECTION_ONLY' if state['phase'] == 'DONE' else 'PAUSED',
        stop_reason=state.get('stop_reason', 'PAUSE_AFTER_EVALUATIONS'), budget=state['budget'],
        source_pins=state['source_pins'], binding=state['binding'], mode=state['protocol']['mode'],
        learner_draw=state['learner_draw'], best=best, best_score=best['discovery_score'],
        challenger=challenger, actor_training_admissible=False, teacher_weight=0.,
        candidates=state['candidates'], proposal_attempts=state['proposal_attempts'],
        evaluations=state['evaluations'], incumbent_transitions=state['incumbent_transitions'],
        best_so_far=state['best_so_far'], counts=state['counts'],
        frozen_selection=dict(selection_receipt=receipt, selection_receipt_sha256=_hash(receipt),
            learner_path_sha256=state['learner_draw']['path_sha256'],
            full_continuation_signature=best['full_continuation_signature'],
            protocol_sha256=_hash(state['protocol']), discovery_only=True,
            frozen=state['phase'] == 'DONE', discovery_world_groups=state['protocol']['discovery_world_groups'],
            protected_world_groups=state['protocol']['protected_world_groups'],
            source_families=state['protocol']['source_families']),
        evidence_scope=state['protocol']['evidence_scope'],
        limits=['NO_ACTOR_ADMISSION', 'NO_SEARCH_BEHAVIOR_LIKELIHOOD', 'NO_NATIVE_STRENGTH_CLAIM',
                'CALLBACK_ATTESTATIONS_NOT_AUTHENTICATED', 'REPORTED_COST_NOT_AUTHENTICATED',
                'INTERRUPTED_WALL_COST_RESERVED_NOT_OBSERVED'])
    return copy.deepcopy(_seal(result))


MC_SCHEMA = 'farm-v0-regret-monte-carlo-v1'

MC_CACHE_SCHEMA = 'farm-v0-regret-exact-mc-cache-v1'


def _mc_sources():
    from farm_neat.evaluation import v0_regret_pairs as E
    return E.source_pins() | {p: hashlib.sha256((R.ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}


def _mc_cache_identity(request, candidate, *, learner, protocol, sources, execution, runner_state):
    """Exact, order-preserving execution identity, NOT a Plan-only/feature key.

    Caller is estimate_rollouts, AFTER _validate_request and _candidate replay.
    These accepted owners bind the actual opening snapshot, every canonical prefix,
    active/draft Plans, ordered started/HOLD/allocation rows and controller state.
    No unsupported nonopening restoration is introduced here. Native opponents are
    fresh deterministic registered instances: their initial memory is fixed by the
    pinned factory, world and RNG contract. Future mutable codecs need a new version.

    Native challenger aliases may merge only by the already-replayed exact Plan +
    full_continuation_signature used by _candidate (including STOP/edit count/key
    and bound HOLD state), under FROZEN_OPENING_THEN_KEEP. Learner syntax remains
    pinned so cross-candidate control receipts can be compared without rewriting.
    Injected runners also retain challenger syntax: arbitrary callbacks need not be
    alias-invariant. Their explicit runner_state must cover ALL behavior-affecting
    configuration/memory/RNG, and attest that trainer provenance, draw order, pool
    IDs and resource caps do not affect a completed simulation. This is the existing
    trusted-callback boundary, not authentication of a closure or its attestation.
    """
    native = execution['kind'] == 'NATIVE_GAME'
    if not native:
        _need(type(runner_state) is dict and set(runner_state) == {'semantics', 'memory'} and
              runner_state['semantics'] == 'FROZEN_V0_REQUEST_ONLY_V1' and
              type(runner_state['memory']) is dict, 'MC_CACHE_RUNNER_STATE_REQUIRED')
    else:
        _need(runner_state is None, 'MC_CACHE_NATIVE_STATE_IS_OWNER_DEFINED')
    R._safe(runner_state)
    rec = request['record']
    geometry = lambda c: {k: c[k] for k in ('canonical_plan', 'full_continuation_signature')}
    return copy.deepcopy(dict(schema=MC_CACHE_SCHEMA, source_pins=sources,
        paired_source_pins=request['source_pins'], execution=execution,
        protocol=protocol, actor_context=rec['actor_context'], continuation=rec['continuation'],
        record_identity={k: v for k, v in rec['identity'].items() if k != 'pair_id'},
        world=request['world'], world_sha256=request['world_sha256'],
        snapshot_sha256=request['initial_world_sha256'], seat=request['seat'],
        opponent=request['opponent'], rng_binding=request['rng_binding'],
        timeout_seconds=request['timeout_seconds'], runner_state=runner_state,
        learner=dict(successor=geometry(learner), raw_actions=learner['raw_actions']),
        challenger=dict(successor=geometry(candidate),
            raw_actions=None if native else candidate['raw_actions']),
        alias_rule='REPLAYED_FULL_SUCCESSOR_FROZEN_NATIVE_ONLY' if native else 'EXACT_RAW_PATH'))


def _mc_cache_sources_match(identity):
    from farm_neat.evaluation import v0_regret_pairs as E
    if E.P.wire(identity['source_pins']) != E.P.wire(_mc_sources()):
        return False
    # An injected executor's file may be outside the repository source set.
    pin = identity['execution']['runner']
    try:
        return hashlib.sha256((R.ROOT / pin['path']).read_bytes()).hexdigest() == pin['sha256']
    except OSError:
        return False


def _mc_verify_cache_identity(identity, request):
    """Join the execution key back to the validated original record, including HOLD."""
    from farm_neat.evaluation import v0_regret_pairs as E
    rec = request['record']; context = rec['actor_context']
    state, _, _ = R._actor(context)
    def geometry(arm):
        path = rec[arm]
        return dict(raw_actions=path['raw_actions'], canonical_plan=path['canonical_plan'],
            full_continuation_signature=_full_candidate_continuation(context, state,
                T.Plan.from_canonical(path['canonical_plan']), path['continuation_signature']))
    expected = _mc_cache_identity(request, geometry('challenger'), learner=geometry('learner'),
        protocol=identity['protocol'], sources=identity['source_pins'], execution=identity['execution'],
        runner_state=identity['runner_state'])
    E._same(identity, expected, 'MC_CACHE_IDENTITY_REQUEST_MISMATCH')


def _mc_verified_receipt(request, result, execution):
    """Verify all four ORIGINAL branch receipts; never execute/rebind an old run."""
    from farm_neat.evaluation import v0_regret_pairs as E
    E._bound(request, 'MC_CACHE_REQUEST_HASH')
    E._bound(result, 'MC_CACHE_RESULT_HASH')
    E._same(result['request_sha256'], request['content_sha256'], 'MC_CACHE_RECEIPT_REQUEST')
    E._same(result['execution'], execution, 'MC_CACHE_EXECUTION')
    replay = result['replay']
    _need(replay['status'] == 'VERIFIED' and replay['mismatches'] == [], 'MC_CACHE_UNVERIFIED')
    seen = set()
    for phase, rows in (('EVALUATE', result['arms']), ('REPLAY', replay['arms'])):
        _need(type(rows) is list and len(rows) == 2 and [r['arm'] for r in rows] == list(E.ARMS),
              'MC_CACHE_INCOMPLETE_ARMS')
        for row in rows:
            _need(row['status'] == 'COMPLETE' and row['phase'] == phase and
                  row['attempt_id'] not in seen, 'MC_CACHE_INCOMPLETE_OR_REUSED_ATTEMPT')
            seen.add(row['attempt_id'])
            branch = E.P.bind(dict(schema=E.BRANCH_SCHEMA, request=request, arm=row['arm'],
                execution=execution, phase=phase, attempt_id=row['attempt_id']))
            E._same(row['branch_sha256'], branch['content_sha256'], 'MC_CACHE_BRANCH_HASH')
            E._same(row['terminal'], E._validate_response(branch, row['response']), 'MC_CACHE_TERMINAL')
    for original, repeated in zip(result['arms'], replay['arms']):
        E._same(E._replay_signature(original), E._replay_signature(repeated), 'MC_CACHE_REPLAY_MISMATCH')
    E._same(result, E._assemble(request, execution, result['arms'], replay), 'MC_CACHE_ASSEMBLY')


class ExactMCCache:
    """Bounded process-local cache of VERIFIED pairs, consumed by estimate_rollouts.

    Pass one instance as simulation_cache across invocations. Default None keeps
    the uncached control. Pair granularity deliberately retains the learner and
    both fresh verification attempts; it does not pretend an unverified single arm
    is reusable. Hits return immutable ORIGINAL receipt/request bytes and the same
    observation ID. Four simulator starts are saved, zero observations are created.
    Physical savings are four only for original native receipts, never for fixtures.

    Entries are admitted only after source/branch/replay/control verification, and
    each lookup rechecks live sources. The SHA index is only a bucket: full ordered
    identity bytes also have to match, so a hash collision is a miss, not a merge.
    LRU eviction bounds stored UTF-8 payload+key bytes (not Python RSS). All outputs
    are detached. There is no disk loader, approximate key, restart or concurrency
    claim. Create separate instances for concurrently running estimators.
    """
    def __init__(self, *, max_entries=128, max_bytes=64*1024*1024):
        _need(type(max_entries) is int and max_entries > 0, 'MC_CACHE_ENTRY_CAP')
        _need(type(max_bytes) is int and max_bytes > 0, 'MC_CACHE_BYTE_CAP')
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._counts = dict(hits=0, misses=0, collisions=0, stores=0, rejections=0,
                            evictions=0, corruptions=0, invalidations=0)

    def stats(self):
        return dict(self._counts, entries=len(self._entries), stored_bytes=self._bytes,
                    max_entries=self.max_entries, max_bytes=self.max_bytes)

    @staticmethod
    def _index(wire):
        return hashlib.sha256(wire.encode()).hexdigest()

    @staticmethod
    def _control(identity):
        from farm_neat.evaluation import v0_regret_pairs as E
        return E.P.wire({k: v for k, v in identity.items() if k != 'challenger'})

    def _remove(self, key):
        entry = self._entries.pop(key)
        self._bytes -= entry['bytes']

    def _invalidate(self, identity):
        control = self._control(identity)
        for key, entry in list(self._entries.items()):
            if entry['control'] == control:
                self._remove(key)
                self._counts['invalidations'] += 1

    def _lookup(self, identity):
        from farm_neat.evaluation import v0_regret_pairs as E
        wire = E.P.wire(identity)
        key = self._index(wire), wire
        self._counts['collisions'] += int(any(k[0] == key[0] and k[1] != wire for k in self._entries))
        entry = self._entries.get(key)
        # Rechecking actual source bytes prevents stale successful executions from
        # becoming reusable even when a caller holds an old identity object.
        if not _mc_cache_sources_match(identity):
            entry = None
        if entry is not None:
            payload = entry['payload']
            if hashlib.sha256(payload).hexdigest() != entry['payload_sha256']:
                self._remove(key)
                self._counts['corruptions'] += 1
            else:
                self._entries.move_to_end(key)
                self._counts['hits'] += 1
                return json.loads(payload)
        self._counts['misses'] += 1
        return None

    def _store(self, identity, request, result):
        from farm_neat.evaluation import v0_regret_pairs as E
        try:
            _need(_mc_cache_sources_match(identity), 'MC_CACHE_STALE_SOURCE')
            E._same(request['source_pins'], E.source_pins(), 'MC_CACHE_STALE_PAIR_SOURCE')
            _mc_verified_receipt(request, result, identity['execution'])
            _mc_verify_cache_identity(identity, request)
        except (ValueError, KeyError, TypeError) as exc:
            self._counts['rejections'] += 1
            return dict(status='REJECTED', reason=str(exc))
        wire = E.P.wire(identity)
        key = self._index(wire), wire
        payload = E.P.wire(dict(request=request, result=result,
            observation_id=E.P.wire_hash(identity))).encode()
        control = self._control(identity)
        size = len(wire.encode()) + len(payload) + len(control.encode())
        if size > self.max_bytes:
            self._counts['rejections'] += 1
            return dict(status='REJECTED', reason='MC_CACHE_BYTE_CAP')
        if key in self._entries:
            # Retain the first verified observation and its original attempt IDs.
            return dict(status='ALREADY_STORED', reason=None)
        while len(self._entries) >= self.max_entries or self._bytes + size > self.max_bytes:
            self._remove(next(iter(self._entries)))
            self._counts['evictions'] += 1
        self._entries[key] = dict(payload=payload, payload_sha256=hashlib.sha256(payload).hexdigest(),
                                  control=control, bytes=size)
        self._bytes += size
        self._counts['stores'] += 1
        return dict(status='STORED', reason=None)


def _mc_statistics(values, families, *, delta=False):
    """One value per registered world; independent units are declared source families."""
    clustered = {}
    for group, value in values.items():
        clustered.setdefault(families[group], []).append(value)
    units = {k: math.fsum(v) / len(v) for k, v in sorted(clustered.items())}
    n = len(units)
    mean = math.fsum(units.values()) / n if n else None
    variance = math.fsum((v - mean)**2 for v in units.values()) / (n-1) if n > 1 else None
    result = dict(independent_group_n=n, terminal_group_n=len(values), utility_mean=mean,
        sample_variance=variance, standard_error=math.sqrt(variance/n) if n > 1 else None,
        utility_bounds=[-2, 2] if delta else [-1, 1], family_means=units,
        uncertainty='DESCRIPTIVE_FAMILY_CLUSTER_SE; NOT_A_CONFIRMATION_BOUND')
    if not delta:
        result['terminal_wdl_counts'] = {name: sum(v == u for v in values.values())
                                        for name, u in (('W', 1), ('D', 0), ('L', -1))}
    return result


def _mc_request(template, candidate, pool_hash, count):
    """Lower a replayed search geometry into the existing selected-path record seam."""
    from farm_neat.evaluation import v0_regret_pairs as E
    request = copy.deepcopy(template)
    record = request['record']
    selected = {k: copy.deepcopy(candidate[k]) for k in (
        'raw_actions', 'prefix_class_trace', 'canonical_plan', 'continuation_signature')}
    receipt = dict(schema=R.SELECTION_SCHEMA,
        actor_context_sha256=record['identity']['actor_context_sha256'],
        parent_checkpoint_sha256=record['identity']['parent_checkpoint_sha256'],
        raw_actions_sha256=_hash(selected['raw_actions']),
        continuation_signature=selected['continuation_signature'], pool_sha256=pool_hash,
        confirmation_registration_sha256=None)
    selected.update(raw_behavior_logp=None, sampling_provenance={'type': 'search_selection'},
        proposal_family=candidate['proposal_family'],
        parent_proposal_ids=[record['learner']['direct_draw_id']],
        search_budget=dict(max_candidates=count, generated_candidates=count, cpu_seconds=0.),
        selection_receipt=receipt, selection_receipt_sha256=_hash(receipt))
    record['challenger'] = _seal(selected, 'path_sha256')
    record['identity']['pair_id'] = 'mc-' + _hash([pool_hash, request['content_sha256'], candidate['candidate_id']])
    _seal(record)
    # Template validation and native _candidate replay were already completed.
    # Only the selected path/selection provenance change here. The paired kernel
    # validates each returned branch and assembles/validates the resulting record;
    # rebuilding the same projection menus for every candidate/world is redundant.
    return E.P.bind(request)


def estimate_rollouts(context: dict, candidates: list[dict], *, runner,
                      continuation_groups: list[dict], budget: dict, seed: int,
                      simulation_cache: ExactMCCache | None = None,
                      runner_state: dict | None = None) -> dict:
    """Estimate already-proposed paths using the existing paired execution kernel.

    ``context = {actor_context, protocol}`` uses improve's ACTOR_SCHEMA and frozen
    PROTOCOL_SCHEMA. ``continuation_groups`` contains ``{group_id, request}`` rows;
    request is an UNEVALUATED v0_regret_pairs.make_request result. Each request's
    world/snapshot/RNG, source pins, seat, full compiler memory and learner join the
    preregistration before ANY callback. Candidate zero is that frozen learner;
    candidates are ordinary {proposal_family, raw_actions} specs or existing
    searcher candidate dicts. Supplied geometry is checked, never trusted.

    Draw groups without replacement (seeded shuffle), then shuffle candidate order
    inside each group. Exact repeated group rows reuse THIS invocation's cached
    job, including failures, never increasing n or retrying. Source-family means
    are independent units; equal family weighting matches improve. Only whole
    common, fully observed source families enter ranking; ties retain learner first.
    Missing data are NOT losses, zeros, wins or precision. Each candidate also
    reports its available-case diagnostic statistics, separate from ranking.

    Each job executes learner/challenger, then both fresh replays, through the
    paired evaluator's source-bound branch validator, comparison and assembler.
    This uses its existing private kernel to stop BETWEEN physical starts; the
    public evaluate_pair/verify_pair calls start both arms unconditionally. Its
    executor—not this module—owns responsive-opponent restoration and V0 execution.
    Requests currently inherit that owner's opening/frozen-KEEP restriction;
    nonopening dawns are refused, not approximated. Seats need separate invocations
    with their own exact actor context; these are not extra independent worlds.

    Budget accepts existing searcher caps except checkpoint/pause options. Each
    attempted callback, including replay/timeout/crash, consumes one start BEFORE
    execution. Injected callbacks use improve's POSIX deadlines. Native execution
    reserves the owner's ten-second subprocess-cleanup grace and is never interrupted
    with SIGALRM (the owner must reap its child). A native per-start cap must exceed
    ten seconds. Callback trust limitations are those of improve.
    Native counts use the paired evaluator's actual runner identity; injected
    fixtures are always zero physical games. No implicit native executor, actor
    label or confirmation relaxation. Optional simulation_cache reuses only verified
    pairs across invocations; hits are the SAME observations, not fresh games.
    Injected caching requires runner_state={semantics: FROZEN_V0_REQUEST_ONLY_V1,
    memory: {...all behavior-affecting callback state...}}; diagnostic counters must
    not affect its outcome. Native fresh-policy state is defined by the paired
    owner (runner_state must be None). See ExactMCCache for alias/memory boundaries.
    Cached job.paired_result and job.receipt_request retain the ORIGINAL bindings;
    job.request is the current demand. Never pass a historical receipt off as a
    new evaluate_pair result or use its historical counts as this invocation's cost.
    This is process-local reuse, not a durable restart claim.
    Seed reproduces logical draws/order, not UUIDs or measured runtime. Standard
    errors describe predeclared family units; they are not sequential confidence
    bounds, independence certificates, or evidence that the selected policy wins.
    """
    from farm_neat.evaluation import v0_regret_pairs as E
    started, cpu = time.monotonic(), time.process_time()
    R._safe(context); R._safe(candidates); R._safe(continuation_groups)
    R._fields(context, ('actor_context', 'protocol'), '$.mc_context')
    _need('checkpoint_path' not in budget and 'pause_after_evaluations' not in budget,
          'MC_CHECKPOINT_NOT_SUPPORTED')
    caps, _, _ = _limits(budget)
    protocol = _protocol(context['protocol'], caps)
    R._int(seed, '$.seed', 0, 2**64-1)
    _need(callable(runner), 'CALLBACK_REQUIRED')
    actor = context['actor_context']
    state, prefix, options = R._actor(actor)
    _need(not prefix.terminated, 'AFTER_STOP')
    _need(type(candidates) is list and 0 < len(candidates) <= caps['max_candidates'], 'CANDIDATE_CAP')
    _need(all(type(c) is dict for c in candidates), 'CANDIDATE_FIELDS')
    _need(type(continuation_groups) is list and 0 < len(continuation_groups) <= 8192, 'MC_GROUP_ROWS')
    registry = {g['source_group_id']: g for g in protocol['discovery_world_groups']}
    execution = E.execution_binding(runner)
    native = execution['kind'] == 'NATIVE_GAME'
    _need(not native or protocol['evidence_scope'] != 'INJECTED_FIXTURE', 'FIXTURE_NATIVE_MISMATCH')
    _need(not native or caps['evaluation_timeout_seconds'] > 10., 'MC_NATIVE_DEADLINE_TOO_SMALL')
    _need(simulation_cache is None or isinstance(simulation_cache, ExactMCCache), 'MC_CACHE_TYPE')
    cache_before = simulation_cache.stats() if simulation_cache is not None else None
    sources = _mc_sources()
    templates, identities, duplicates = {}, {}, []
    control = coupling = None
    for row in continuation_groups:
        R._fields(row, ('group_id', 'request'), '$.continuation_group')
        gid = row['group_id']
        _need(gid in registry, 'UNREGISTERED_OR_PROTECTED_GROUP')
        if gid in templates:
            _need(E.P.wire(row['request']) == E.P.wire(templates[gid]), 'DUPLICATE_GROUP_BINDING_MISMATCH')
            duplicates.append(gid)
            continue
        req = E._validate_request(row['request'])
        rec = req['record']; registered = registry[gid]
        _need(rec['identity']['split'] == 'discovery' and rec['identity']['source_group_id'] == gid,
              'DISCOVERY_ONLY')
        _need(E.P.wire(rec['actor_context']) == E.P.wire(actor) and req['seat'] == actor['observation']['player'],
              'MC_ACTOR_SEAT_MISMATCH')
        _need(req['opponent']['identity']['content_sha256'] == protocol['opponent_sha256'],
              'MC_OPPONENT_MISMATCH')
        _need(_hash(rec['continuation']) == protocol['continuation_sha256'], 'MC_CONTINUATION_MISMATCH')
        actual = dict(source_group_id=gid, world_sha256=req['world_sha256'],
            snapshot_sha256=req['initial_world_sha256'], rng_sha256=_hash(req['rng_binding']), seed=req['world']['seed'])
        _need(R._equal(actual, registered), 'MC_GROUP_IDENTITY_MISMATCH')
        _need(R._equal(rec['learner']['raw_actions'], candidates[0].get('raw_actions')), 'MC_LEARNER_MISMATCH')
        if control is None:
            control = rec['learner']
        _need(R._equal(control, rec['learner']), 'MC_LEARNER_DRAW_MISMATCH')
        # Seeds/shops vary by world. The coupling *mechanism* and future policies do not.
        rule = {k: v for k, v in req['rng_binding'].items() if k not in ('seed', 'fixed_shops')}
        if coupling is None:
            coupling = rule
        _need(R._equal(rule, coupling), 'MC_RNG_COUPLING_MISMATCH')
        templates[gid] = req
        identities[gid] = dict(group_id=gid, **{k: v for k, v in actual.items() if k != 'source_group_id'},
            seat=req['seat'], actor_context_sha256=_hash(actor),
            continuation_sha256=_hash(rec['continuation']), opponent_sha256=protocol['opponent_sha256'],
            source_pins_sha256=_hash(sources), rng_coupling=copy.deepcopy(req['rng_binding']))
    _need(set(templates) == set(registry), 'MC_REGISTRATION_NOT_COVERED')
    _need(len({g['seed'] for g in registry.values()}) == len(registry), 'MC_DUPLICATE_SEED')

    pool, rejected, candidate_duplicates, menus = [], [], [], {}
    for i, spec in enumerate(candidates):
        _need(type(spec) is dict, 'CANDIDATE_FIELDS')
        try:
            minimal = {k: copy.deepcopy(spec[k]) for k in ('proposal_family', 'raw_actions', 'target_plan',
                       'status', 'reason_codes', 'provenance') if k in spec}
            candidate, reasons = _candidate(actor, minimal, state, prefix, options, menus,
                                            edit_limit=caps['max_path_edits'])
            if candidate is not None:
                _need(all(k not in spec or R._equal(spec[k], candidate[k]) for k in GEOMETRY),
                      'MC_SUPPLIED_GEOMETRY_MISMATCH')
        except (ValueError, KeyError, TypeError) as exc:
            candidate, reasons = None, [str(exc)]
        if candidate is None:
            _need(i != 0, 'MC_INVALID_LEARNER:' + ','.join(reasons))
            rejected.append(dict(input_index=i, specification=copy.deepcopy(spec), reasons=reasons))
        elif candidate['candidate_id'] in {c['candidate_id'] for c in pool}:
            candidate_duplicates.append(dict(input_index=i, candidate_id=candidate['candidate_id']))
        else:
            pool.append(candidate)
    pool_hash = _hash(dict(context=context, candidates=pool, sources=sources, seed=seed, budget=caps,
                           continuation_groups=continuation_groups, execution=execution))
    rng = random.Random(seed)
    groups = list(templates); rng.shuffle(groups)
    draws = []
    for gid in groups:
        order = list(range(len(pool))); rng.shuffle(order)
        draws.extend(dict(group_id=gid, candidate_id=pool[i]['candidate_id']) for i in order)
    draws.extend(dict(group_id=gid, candidate_id=c['candidate_id']) for gid in duplicates for c in pool)
    counts = dict(simulator_starts=0, physical_starts=0, replay_checks=0, cache_hits=0,
                  physical_completed=0, timeouts=0, crashes=0, invalid_responses=0,
                  simulation_cache_hits=0, simulator_starts_saved=0, physical_starts_saved=0,
                  cache_independent_groups_added=0)
    attempts, jobs, seen_jobs, observations, best_trace = [], [], {}, {c['candidate_id']: {} for c in pool}, []
    cache_identities = {}
    stop = None
    control_signatures, bad_controls = {}, set()
    family_groups = {}
    for gid in registry:
        family_groups.setdefault(protocol['source_families'][gid], set()).add(gid)

    def ranking():
        # No easy-world / survivor / incomplete-family winner. Available-case
        # summaries remain diagnostic; ranking requires every world of each unit.
        observed = set.intersection(*(set(v) for v in observations.values()))
        common = sorted(g for g in observed
                        if family_groups[protocol['source_families'][g]] <= observed)
        scores = {cid: _mc_statistics({g: v[g] for g in common}, protocol['source_families'])['utility_mean']
                  for cid, v in observations.items()}
        order = [c['candidate_id'] for c in pool]
        if common:
            order.sort(key=lambda cid: -scores[cid])  # stable ties retain learner/input order
        return common, order, scores

    def save_best():
        common, order, scores = ranking()
        best_trace.append(dict(after_job=len(jobs), simulator_starts=counts['simulator_starts'],
            candidate_id=order[0], score=scores[order[0]], common_group_ids=common))

    def observe(job, originals):
        cid, gid = job['candidate_id'], job['group_id']
        signature = E.P.wire(E._replay_signature(originals[0]))
        if gid in control_signatures and control_signatures[gid] != signature:
            bad_controls.add(gid)
        control_signatures.setdefault(gid, signature)
        if gid in bad_controls:
            for values in observations.values():
                values.pop(gid, None)
            for previous in jobs:
                if previous['group_id'] == gid and previous['status'] in ('COMPLETE', 'CACHE_HIT'):
                    previous.update(status='INVALID', reason='MC_CONTROL_CONTINUATION_MISMATCH')
            if simulation_cache is not None:
                simulation_cache._invalidate(cache_identities[(cid, gid)])
        else:
            observations[cid][gid] = originals[1]['terminal']['wdl']

    def attempt(request, arm, phase, job):
        nonlocal stop
        if stop is not None:
            return None
        if counts['simulator_starts'] >= caps['max_simulator_starts']:
            stop = 'SIMULATOR_START_CAP'; return None
        remaining = caps['max_wall_seconds'] - (time.monotonic()-started)
        if remaining <= 0:
            stop = 'WALL_CAP'; return None
        allowance = min(remaining, caps['evaluation_timeout_seconds'])
        if native:
            allowance = request['timeout_seconds'] + 10.
            if allowance > remaining:
                stop = 'WALL_CAP'; return None
        branch = E._branch(request, arm, execution, phase)
        event = dict(job_id=job['job_id'], candidate_id=job['candidate_id'], group_id=job['group_id'],
            arm=arm, phase=phase, branch=branch, event='PENDING', outcome=None,
            simulator_starts=1, physical_starts=int(native), timeout_seconds=allowance)
        attempts.append(event)  # Account BEFORE execution, even if the callback raises.
        counts['simulator_starts'] += 1
        counts['physical_starts'] += int(native)
        counts['replay_checks'] += int(phase == 'REPLAY')
        before = time.monotonic()
        def bounded(argument):
            try:
                if native:
                    # native_runner owns terminate/kill/wait on subprocess timeout.
                    # An outer SIGALRM would bypass that cleanup and orphan a game.
                    return runner(argument)
                with _deadline(allowance):
                    return runner(argument)
            except TimeoutError:
                event['event'] = 'TIMEOUT'
                raise
            except Exception:
                event['event'] = 'CRASHED'
                raise
        outcome = E._run(branch, bounded)
        event['outcome'] = outcome
        event['wall_seconds'] = time.monotonic()-before
        if event['event'] == 'PENDING':
            response = outcome['response']
            row = response.get('row') if isinstance(response, dict) else None
            status = row.get('status') if isinstance(row, dict) else None
            event['event'] = 'INVALID' if outcome['status'] == 'INVALID' else status or outcome['status']
        counts['timeouts'] += int(event['event'] == 'TIMEOUT')
        counts['crashes'] += int(event['event'] == 'CRASHED')
        counts['invalid_responses'] += int(outcome['status'] == 'INVALID')
        counts['physical_completed'] += int(native and outcome['status'] == 'COMPLETE')
        return outcome

    save_best()
    for draw in draws:
        cid, gid = draw['candidate_id'], draw['group_id']
        key = (cid, gid)
        job = dict(job_id=len(jobs), **draw, shared_identity=identities[gid],
                   status='NOT_STARTED', paired_result=None, reason=None)
        jobs.append(job)
        if key in seen_jobs:
            previous = seen_jobs[key]
            successful = previous['status'] in ('COMPLETE', 'CACHE_HIT')
            job.update(status='CACHE_HIT' if successful else 'DUPLICATE_SUPPRESSED',
                cache_of_job=previous['job_id'], reason=previous['reason'],
                paired_result=copy.deepcopy(previous['paired_result']),
                reuse_kind='DUPLICATE_JOB_NOT_NEW_EVIDENCE', reused_status=previous['status'])
            counts['cache_hits'] += 1  # legacy in-call suppression includes failures, not admission
            continue
        seen_jobs[key] = job
        if gid in bad_controls:
            # No additional branch from a contradicted control can enter statistics.
            job.update(status='INVALID', reason='MC_CONTROL_CONTINUATION_MISMATCH')
            continue
        candidate = next(c for c in pool if c['candidate_id'] == cid)
        req = _mc_request(templates[gid], candidate, pool_hash, len(pool))
        if native:
            req['timeout_seconds'] = min(req['timeout_seconds'], caps['evaluation_timeout_seconds'] - 10.)
            req = E.P.bind(req)
        job['request'] = req
        if simulation_cache is not None:
            identity = _mc_cache_identity(req, candidate, learner=pool[0], protocol=protocol,
                sources=sources, execution=execution, runner_state=runner_state)
            cache_identities[key] = identity
            job['cache_identity_sha256'] = E.P.wire_hash(identity)
            if time.monotonic()-started < caps['max_wall_seconds']:
                hit = simulation_cache._lookup(identity)
                if hit is not None:
                    job.update(status='CACHE_HIT', paired_result=hit['result'],
                        receipt_request=hit['request'], observation_id=hit['observation_id'],
                        reuse_kind='ORIGINAL_VERIFIED_OBSERVATION')
                    counts['cache_hits'] += 1
                    counts['simulation_cache_hits'] += 1
                    counts['simulator_starts_saved'] += 4
                    counts['physical_starts_saved'] += 4 * int(native)
                    observe(job, hit['result']['arms'])
                    save_best()
                    continue
        if stop is None and counts['simulator_starts'] >= caps['max_simulator_starts']:
            stop = 'SIMULATOR_START_CAP'
        if time.monotonic()-started >= caps['max_wall_seconds']:
            stop = 'WALL_CAP'
        if stop is not None:
            job['reason'] = stop
            continue
        originals, replay = [], []
        for arm in E.ARMS:
            row = attempt(req, arm, 'EVALUATE', job)
            if row is None:
                break
            originals.append(row)
        if len(originals) < 2:
            job.update(status='CENSORED' if originals else 'NOT_STARTED', reason=stop)
            save_best(); continue
        mismatch = []
        if all(a['status'] == 'COMPLETE' for a in originals):
            for arm in E.ARMS:
                row = attempt(req, arm, 'REPLAY', job)
                if row is None:
                    break
                replay.append(row)
            for original, repeated in zip(originals, replay):
                if original['status'] != 'COMPLETE' or repeated['status'] != 'COMPLETE':
                    mismatch.append(dict(arm=original['arm'], reason='INCOMPLETE_OR_INVALID_BRANCH'))
                elif E.P.wire(E._replay_signature(original)) != E.P.wire(E._replay_signature(repeated)):
                    mismatch.append(dict(arm=original['arm'], reason='SEMANTIC_REPLAY_MISMATCH'))
        status = ('MISMATCH' if mismatch else 'VERIFIED') if len(replay) == 2 else ('INCOMPLETE' if replay else 'NOT_RUN')
        result = E._assemble(req, execution, originals, dict(status=status, arms=replay, mismatches=mismatch))
        job.update(paired_result=result, status='COMPLETE' if status == 'VERIFIED' else 'CENSORED',
                   reason=None if status == 'VERIFIED' else stop or 'INCOMPLETE_OR_UNVERIFIED_PAIR')
        if job['status'] == 'COMPLETE':
            observe(job, originals)
        save_best()

    # Do not publish a successful cache entry until all same-world learner controls
    # in this invocation agree. A later failed control withdraws earlier hits too.
    if simulation_cache is not None:
        for job in jobs:
            key = job['candidate_id'], job['group_id']
            if job['status'] == 'COMPLETE' and job['group_id'] not in bad_controls:
                job['cache_admission'] = simulation_cache._store(
                    cache_identities[key], job['request'], job['paired_result'])
                job['observation_id'] = E.P.wire_hash(cache_identities[key])
        cache_after = simulation_cache.stats()
        counts['simulation_cache_misses'] = cache_after['misses'] - cache_before['misses']
        counts['simulation_cache_collisions'] = cache_after['collisions'] - cache_before['collisions']
        counts['simulation_cache_stores'] = cache_after['stores'] - cache_before['stores']
        counts['simulation_cache_rejections'] = cache_after['rejections'] - cache_before['rejections']

    common, order, scores = ranking()
    summaries = []
    for c in pool:
        cid = c['candidate_id']; own = [a for a in attempts if a['candidate_id'] == cid]
        summaries.append(dict(candidate_id=cid, geometry=_visible(c), proposal_family=c['proposal_family'],
            statistics=_mc_statistics(observations[cid], protocol['source_families']),
            ranking_statistics=_mc_statistics({g: observations[cid][g] for g in common}, protocol['source_families']),
            counts=dict(simulator_starts=len(own), physical_starts=sum(a['physical_starts'] for a in own),
                replay_checks=sum(a['phase'] == 'REPLAY' for a in own),
                censored_groups=len(registry)-len(observations[cid]),
                timeouts=sum(a['event'] == 'TIMEOUT' for a in own),
                crashes=sum(a['event'] == 'CRASHED' for a in own),
                invalid_responses=sum(a['outcome']['status'] == 'INVALID' for a in own)),
            actor_training_admissible=False))
    comparisons = []
    for i, a in enumerate(pool):
        for b in pool[i+1:]:
            aid, bid = a['candidate_id'], b['candidate_id']
            shared = sorted(set(observations[aid]) & set(observations[bid]))
            values = {g: observations[bid][g]-observations[aid][g] for g in shared}
            comparisons.append(dict(left_id=aid, right_id=bid,
                shared_identities=[identities[g] for g in shared],
                statistics=_mc_statistics(values, protocol['source_families'], delta=True),
                actor_training_admissible=False))
    first = pool[0]['candidate_id']
    winner = order[0] if common and scores[order[0]] > scores[first] else None
    counts.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu)
    return _seal(dict(schema=MC_SCHEMA, status='DISCOVERY_ONLY', stop_reason=stop or 'SCHEDULE_COMPLETE',
        seed=seed, budget=caps, source_pins=sources, binding=pool_hash, execution=execution,
        registration=protocol, draw_order=draws, candidates=summaries, rejected_candidates=rejected,
        duplicate_candidates=candidate_duplicates, comparisons=comparisons,
        attempts=attempts, jobs=jobs, counts=counts, best_so_far=best_trace,
        common_group_ids=common, ranking=order, best_id=order[0], winner_id=winner,
        actor_training_admissible=False, teacher_weight=0., confirmation_required=True,
        limitations=['DISCOVERY_ORDERING_ONLY', 'NO_NATIVE_STRENGTH_CLAIM',
            'PREDECLARATION_AND_INDEPENDENCE_CALLER_ATTESTED', 'NO_DURABLE_CACHE_OR_RESTART',
            'CACHE_HITS_ARE_ORIGINAL_OBSERVATIONS_NOT_NEW_EVIDENCE',
            'OPENING_PAIRED_EVALUATOR_SCOPE', 'ONE_EXACT_ACTOR_SEAT_CONTEXT_PER_INVOCATION',
            'COMMON_RNG_KEYS_NOT_COMMON_EXOGENOUS_EVENT_CERTIFICATE',
            'CALLBACK_ATTESTATIONS_NOT_AUTHENTICATED', 'BUDGET_CENSORING_CAN_BIAS_AVAILABLE_CASE_STATS']))


def improve(context: dict, learner: dict, *, proposer, evaluator, budget: dict,
            protocol: dict, seed: int) -> dict:
    """Run or resume one source-bound fixed search. No native executor is implicit."""
    clock = _Clock()
    caps, path, pause = _limits(budget)
    protocol = _protocol(protocol, caps)
    R._int(seed, '$.seed', 0, 2**64 - 1)
    _need(callable(proposer) and callable(evaluator), 'CALLBACK_REQUIRED')
    native_state, prefix, options = R._actor(context)
    _need(not prefix.terminated, 'AFTER_STOP')
    R._safe(learner)
    checkpoint = learner.get('sampling_provenance', {}).get('checkpoint_sha256')
    R._hash(checkpoint, '$.learner.checkpoint')
    menus = {}
    R._proposal(learner, {'parent_checkpoint_sha256': checkpoint}, prefix,
                native_state, options, menus, direct=True)
    sources = R.source_pins() | {p: hashlib.sha256((R.ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    binding = _hash(dict(context=context, learner=learner, protocol=protocol, budget=caps, seed=seed, sources=sources))
    with _lock(path):
        if path is not None and path.exists():
            state = R._loads(path.read_text(encoding='utf-8'))
            _need(type(state) is dict and state.get('content_sha256') == _hash(
                {k: v for k, v in state.items() if k != 'content_sha256'}), 'CHECKPOINT_HASH_MISMATCH')
            _need(state.get('binding') == binding, 'RESUME_BINDING_MISMATCH')
            if state['phase'] == 'DONE':
                return _result(state)
        else:
            base, reasons = _candidate(context, dict(proposal_family='learner', raw_actions=learner['raw_actions']),
                native_state, prefix, options, menus, edit_limit=16)
            _need(base is not None, 'INVALID_LEARNER_GEOMETRY:' + ','.join(reasons))
            state = dict(schema=SCHEMA+'-checkpoint', binding=binding, budget=caps, protocol=protocol,
                source_pins=sources, actor_context=copy.deepcopy(context), learner_draw=copy.deepcopy(learner),
                candidates=[base], learner_id=base['candidate_id'], incumbent_id=base['candidate_id'],
                current_id=base['candidate_id'], phase='EVALUATE',
                proposal_attempts=[dict(attempt_id=0, proposal_family='learner', status='EVALUATING',
                    candidate_id=base['candidate_id'], reason_codes=[], raw_actions=learner['raw_actions'])],
                evaluations=[], incumbent_transitions=[], best_so_far=[],
                counts=dict(iterations=0, candidates=1, attempted_simulator_starts=0,
                    completed_terminal_executions=0, simulator_steps_reported=0,
                    reported_simulator_starts=0, responses_with_start_report=0,
                    physical_games=0 if protocol['evidence_scope'] == 'INJECTED_FIXTURE' else None,
                    wall_seconds_observed=0., wall_seconds_charged=0.,
                    interrupted_reserved_seconds=0., cpu_seconds_observed=0.))
        def save():
            clock.tick(state)
            _write(path, state)
        # Any durable in-flight intent is consumed, never resubmitted after interruption.
        for event in state['evaluations']:
            if event['status'] == 'PENDING':
                event.update(status='CENSORED', reason_code='INTERRUPTED_UNKNOWN_OUTCOME',
                             observed_wall_seconds=None, observed_cpu_seconds=None)
                reservation = event['reserved_wall_seconds']
                state['counts']['interrupted_reserved_seconds'] += reservation
                state['counts']['wall_seconds_charged'] += reservation
        if state['phase'] == 'GENERATING':
            attempt = state['proposal_attempts'][-1]
            attempt.update(status='FAILED', reason_codes=['INTERRUPTED_PROPOSER_UNKNOWN_OUTPUT'])
            reservation = attempt['reserved_wall_seconds']
            state['counts']['wall_seconds_charged'] += reservation
            state['counts']['interrupted_reserved_seconds'] += reservation
            state['phase'] = 'PROPOSE'
            _trace_best(state, 'INTERRUPTED_PROPOSER_UNKNOWN_OUTPUT')
        save()
        calls = 0
        while state['phase'] != 'DONE':
            clock.tick(state)
            # Commit a fully observed score BEFORE applying a work-start deadline. A slow
            # checkpoint is not permission to lose a completed, already-paid-for winner.
            if state['phase'] == 'EVALUATE':
                candidate = _lookup(state, state['current_id'])
                done = [e for e in state['evaluations'] if e['candidate_id'] == candidate['candidate_id']]
                if len(done) == len(protocol['discovery_world_groups']):
                    _finish_candidate(state, protocol)
                    save()
                    continue
            remaining = caps['max_wall_seconds'] - state['counts']['wall_seconds_charged']
            if remaining <= 0:
                _finish(state, 'WALL_CAP'); break
            if state['phase'] == 'EVALUATE':
                if state['counts']['attempted_simulator_starts'] >= caps['max_simulator_starts']:
                    _finish(state, 'SIMULATOR_START_CAP'); break
                if pause is not None and calls >= pause:
                    save(); return _result(state)
                group = protocol['discovery_world_groups'][len(done)]
                timeout = min(remaining, caps['evaluation_timeout_seconds'])
                request = dict(schema=SCHEMA+'-request', attempt_id=_hash([binding, len(state['evaluations'])]),
                    purpose='DISCOVERY', group=group, actor_context=copy.deepcopy(context),
                    candidate=_visible(candidate), seat=context['observation']['player'],
                    opponent_sha256=protocol['opponent_sha256'], continuation_sha256=protocol['continuation_sha256'],
                    execution_context=protocol['execution_context'], source_pins=sources, timeout_seconds=timeout)
                _seal(request, 'request_sha256')
                event = dict(attempt_id=request['attempt_id'], request_sha256=request['request_sha256'],
                    request=request, candidate_id=candidate['candidate_id'], group_id=group['source_group_id'],
                    status='PENDING', reason_code='DURABLE_START_INTENT', reserved_wall_seconds=timeout,
                    response=None, observed_wall_seconds=None, observed_cpu_seconds=None)
                state['evaluations'].append(event)
                state['counts']['attempted_simulator_starts'] += 1
                save()  # The start is durably charged before any external execution.
                clock.tick(state)
                allowance = min(timeout, caps['max_wall_seconds'] - state['counts']['wall_seconds_charged'])
                started, cpu = time.monotonic(), time.process_time()
                start_violation = False
                try:
                    if allowance <= 0:
                        raise _Deadline('WALL_CAP_BEFORE_CALLBACK')
                    with _deadline(allowance):
                        output = evaluator(copy.deepcopy(request))
                        # Invalid JSON cannot be put into a sealed checkpoint. Safe but
                        # malformed responses ARE retained, before semantic validation.
                        event['response_type'] = type(output).__name__
                        R._safe(output)
                        event['response'] = copy.deepcopy(output)
                        if type(output) is dict:
                            reported = output.get('simulator_starts')
                            if type(reported) is int and reported >= 0:
                                state['counts']['reported_simulator_starts'] += reported
                                state['counts']['responses_with_start_report'] += 1
                            start_violation = type(reported) is not int or reported != 1
                            steps = output.get('simulator_steps')
                            if type(steps) is int and steps >= 0:
                                state['counts']['simulator_steps_reported'] += steps
                        event['status'], event['reason_code'] = _validate_response(request, output)
                except _Deadline:
                    event.update(status='CENSORED', reason_code=(
                        'WALL_CAP_BEFORE_CALLBACK' if allowance <= 0 else 'EVALUATOR_TIMEOUT'))
                except Exception as exc:
                    event.update(status='FAILED', reason_code=getattr(exc, 'code', str(exc) or type(exc).__name__),
                                 failure_type=type(exc).__name__)
                # BaseException deliberately leaves the previously saved PENDING intent.
                event['observed_wall_seconds'] = max(0., time.monotonic()-started)
                event['observed_cpu_seconds'] = max(0., time.process_time()-cpu)
                if event['status'] == 'COMPLETE':
                    state['counts']['completed_terminal_executions'] += 1
                calls += 1
                save()
                if start_violation:
                    # A trusted executor that reports breaking the one-start contract is
                    # stopped, not laundered into another in-budget attempt. Reported cost
                    # stays separate from reservations and is not authenticated here.
                    _finish(state, 'EXECUTOR_START_CONTRACT_VIOLATION')
                    break
            else:
                if protocol['mode'] == 'learner':
                    _finish(state, 'LEARNER_CONTROL_ONLY'); break
                if state['counts']['iterations'] >= caps['max_iterations']:
                    _finish(state, 'ITERATION_CAP'); break
                if state['counts']['candidates'] >= caps['max_candidates']:
                    _finish(state, 'CANDIDATE_CAP'); break
                parent_id = state['incumbent_id'] if protocol['mode'] == 'iterative' else state['learner_id']
                parent = _lookup(state, parent_id)
                iteration = state['counts']['iterations']
                proposal_seed = int(_hash([seed, iteration])[:16], 16)
                timeout = min(remaining, caps['evaluation_timeout_seconds'])
                attempt = dict(attempt_id=state['counts']['candidates'], status='GENERATING',
                    proposal_family='unknown', parent_candidate_id=parent_id, generator_seed=proposal_seed,
                    reason_codes=[], reserved_wall_seconds=timeout, iteration=iteration)
                state['proposal_attempts'].append(attempt)
                state['counts']['candidates'] += 1
                state['counts']['iterations'] += 1
                state['phase'] = 'GENERATING'
                save()
                clock.tick(state)
                allowance = min(timeout, caps['max_wall_seconds'] - state['counts']['wall_seconds_charged'])
                started, cpu = time.monotonic(), time.process_time()
                try:
                    if allowance <= 0:
                        raise _Deadline('WALL_CAP_BEFORE_CALLBACK')
                    generator = random_proposer if protocol['mode'] == 'one-shot-random' else proposer
                    with _deadline(allowance):
                        spec = generator(copy.deepcopy(context), _visible(parent),
                            budget=dict(max_candidates=1, max_path_edits=caps['max_path_edits'], iteration=iteration),
                            seed=proposal_seed)
                        R._safe(spec)
                        attempt['proposal'] = copy.deepcopy(spec)
                        candidate, reasons = _candidate(context, spec, native_state, prefix, options, menus,
                                                        edit_limit=caps['max_path_edits'])
                    attempt['proposal_family'] = spec['proposal_family']
                    if candidate is None:
                        attempt.update(status='UNKNOWN' if spec.get('status') == 'UNKNOWN' else 'REJECTED',
                                       reason_codes=reasons)
                    elif any(c['candidate_id'] == candidate['candidate_id'] for c in state['candidates']):
                        attempt.update(status='DUPLICATE', reason_codes=['DUPLICATE_EXACT_CONTINUATION'],
                                       duplicate_of=candidate['candidate_id'])
                    else:
                        state['candidates'].append(candidate)
                        state['current_id'] = candidate['candidate_id']
                        attempt.update(status='EVALUATING', reason_codes=[], candidate_id=candidate['candidate_id'])
                except _Deadline:
                    attempt.update(status='CENSORED', reason_codes=[
                        'WALL_CAP_BEFORE_CALLBACK' if allowance <= 0 else 'PROPOSER_TIMEOUT'])
                except Exception as exc:
                    attempt.update(status='FAILED', reason_codes=[getattr(exc, 'code', str(exc) or type(exc).__name__)],
                                   failure_type=type(exc).__name__)
                attempt['observed_wall_seconds'] = max(0., time.monotonic()-started)
                attempt['observed_cpu_seconds'] = max(0., time.process_time()-cpu)
                if attempt['status'] == 'EVALUATING':
                    state['phase'] = 'EVALUATE'
                else:
                    state['phase'] = 'PROPOSE'
                    _trace_best(state, attempt['reason_codes'][0])
                save()
        save()
        return _result(state)


def compare_searches(context, learner, *, proposer, evaluator, budget, protocol, seed):
    """Four controls; same predeclared cap PER arm, not a pooled renewable budget."""
    caps, path, pause = _limits(budget)
    _need(pause is None, 'COMPARISON_REQUIRES_COMPLETE_OR_EXHAUSTED_ARMS')
    traces = {}
    for mode in MODES:
        local = dict(budget)
        if path is not None:
            local['checkpoint_path'] = str(path.with_name(path.name + '.' + mode))
        traces[mode] = improve(context, learner, proposer=proposer, evaluator=evaluator,
            budget=local, protocol={**protocol, 'mode': mode}, seed=seed)
    return _seal(dict(schema=SCHEMA+'-controls', declared_budget=caps, traces=traces,
        total_start_cap=len(MODES)*caps['max_simulator_starts'],
        total_attempted_simulator_starts=sum(t['counts']['attempted_simulator_starts'] for t in traces.values()),
        cost_semantics='MATCHED_DECLARED_CAPS; ACTUAL_WORK_AND_UNUSED_ALLOWANCE_ARE_NOT_EQUAL',
        actor_training_admissible=False))


# Fixed-population orchestration for the injected tensor proposal boundary.
BATCH_REGISTRATION_SCHEMA = 'farm-v0-progressive-proposal-registration-v1'
BATCH_PROGRESS_SCHEMA = 'farm-v0-progressive-proposals-v1'


def register_batch_proposals(*, quotas, seed, stage_multipliers, gate,
        checkpoint_sha256, tensor_backend_sha256, provider_sha256,
        discovery_groups=(), protected_groups=(), callback_timeout_seconds=60., **limits):
    """Freeze a cheap probe and later incremental populations before results.

    stage_multipliers scale every family equally; their total attempts per queue
    state must fit the existing <=32 proposal seam. ANY_REACHABLE uses only exact
    generated geometry. DISCOVERY_POSITIVE requires declared complete replayed
    discovery evidence; it cannot consume confirmation/final outcomes. Hashes
    bind declarations/bytes, not external chronology or evidence authenticity.
    """
    quotas = P._batch_quotas(quotas)
    R._int(seed, '$.seed', 0, 2**64 - 1)
    _need(type(stage_multipliers) is list and 1 <= len(stage_multipliers) <= 4,
          'WIDENING_STAGES')
    for multiplier in stage_multipliers:
        R._int(multiplier, '$.stage_multipliers', 1, 32)
    _need(stage_multipliers[0] == 1 and stage_multipliers == sorted(stage_multipliers),
          'CHEAP_PROBE_FIRST')
    _need(sum(stage_multipliers) * sum(quotas.values()) <= 32, 'POPULATION_CAP')
    _need(gate in ('ANY_REACHABLE', 'DISCOVERY_POSITIVE'), 'WIDENING_GATE')
    for pin in (checkpoint_sha256, tensor_backend_sha256, provider_sha256):
        R._hash(pin, '$.identity')
    groups, protected = list(discovery_groups), list(protected_groups)
    for group in groups + protected:
        R._text(group, '$.source_group_id')
    _need(len(set(groups)) == len(groups) and len(set(protected)) == len(protected) and
          not set(groups).intersection(protected), 'PROTECTED_DISCOVERY_OVERLAP')
    _need(gate != 'DISCOVERY_POSITIVE' or bool(groups), 'DISCOVERY_GROUPS_REQUIRED')
    R._number(callback_timeout_seconds, '$.callback_timeout_seconds', 0.001, 600.)
    source_paths = set(SOURCE_PATHS) | set(R.source_pins()) | {
        'src/farm_neat/policy/v0_scorer.py', 'src/farm_neat/policy/v0_menu_fast.py',
        'src/farm_neat/training/teachers/v0_regret_budget.py', 'project/runtime/interpreter.toml'}
    return _seal(dict(schema=BATCH_REGISTRATION_SCHEMA, quotas=quotas, seed=seed,
        stage_multipliers=stage_multipliers, gate=gate, checkpoint_sha256=checkpoint_sha256,
        tensor_backend_sha256=tensor_backend_sha256, provider_sha256=provider_sha256,
        discovery_groups=groups, protected_groups=protected,
        callback_timeout_seconds=float(callback_timeout_seconds), limits=P._batch_limits(limits),
        source_pins={p: hashlib.sha256((R.ROOT / p).read_bytes()).hexdigest() for p in sorted(source_paths)},
        teacher_admission='UNCHANGED_NOT_CALLED'))


def _batch_registration(registration):
    R._fields(registration, ('schema', 'quotas', 'seed', 'stage_multipliers', 'gate',
        'checkpoint_sha256', 'tensor_backend_sha256', 'provider_sha256', 'discovery_groups',
        'protected_groups', 'callback_timeout_seconds', 'limits', 'source_pins',
        'teacher_admission', 'content_sha256'), '$.registration')
    expected = register_batch_proposals(**{k: registration[k] for k in (
        'quotas', 'seed', 'stage_multipliers', 'gate', 'checkpoint_sha256',
        'tensor_backend_sha256', 'provider_sha256', 'discovery_groups',
        'protected_groups', 'callback_timeout_seconds')}, **registration['limits'])
    _need(R._equal(registration, expected), 'REGISTRATION_OR_SOURCE_DRIFT')
    return expected


def _batch_signal(signal, state, registration):
    R._fields(signal, ('stage', 'queue_id', 'pool_sha256', 'candidate_id', 'phase',
        'source_group_id', 'complete', 'replay_verified', 'wdl_delta'), '$.discovery_signal')
    _need(signal['phase'] == 'discovery' and
          signal['source_group_id'] in registration['discovery_groups'] and
          signal['source_group_id'] not in registration['protected_groups'], 'PROTECTED_SIGNAL')
    R._int(signal['stage'], '$.stage', 0, len(registration['stage_multipliers']) - 1)
    _need(type(signal['complete']) is bool and type(signal['replay_verified']) is bool,
          'SIGNAL_COMPLETION_FLAGS')
    if signal['complete'] and signal['replay_verified']:
        R._number(signal['wdl_delta'], '$.wdl_delta', -2, 2)
    else:
        _need(signal['wdl_delta'] is None, 'CENSORED_SIGNAL_HAS_NO_WDL')
    receipt = next((r for r in state['stage_receipts'] if r['stage'] == signal['stage']), None)
    _need(receipt is not None and signal['pool_sha256'] == receipt['pool_sha256'], 'STALE_DISCOVERY_SIGNAL')
    _need(signal['candidate_id'] in receipt['candidate_ids'].get(signal['queue_id'], []),
          'SIGNAL_CANDIDATE_JOIN')
    return copy.deepcopy(signal)


def _merge_batch_population(states, population):
    lookup = {r['queue_id']: r for r in states}
    for incoming in population:
        target = lookup[incoming['queue_id']]
        # Full continuation includes decoder cursor and starting compiler/HOLD
        # memory. Same Plan alone never licenses deduplication.
        exact = {T._json([c['canonical_plan'], c['full_continuation_signature']]): c['candidate_id']
                 for c in target['candidates']}
        duplicate_ids = {}
        for candidate in incoming['candidates']:
            key = T._json([candidate['canonical_plan'], candidate['full_continuation_signature']])
            if key in exact:
                duplicate_ids[candidate['candidate_id']] = exact[key]
            else:
                exact[key] = candidate['candidate_id']
                target['candidates'].append(copy.deepcopy(candidate))
        for original in incoming['attempts']:
            attempt = copy.deepcopy(original)
            if attempt['status'] == 'RETAINED' and attempt['candidate_id'] in duplicate_ids:
                attempt.update(status='DUPLICATE', reason_codes=['DUPLICATE_SUCCESSOR'],
                               duplicate_of=duplicate_ids[attempt.pop('candidate_id')])
            target['attempts'].append(attempt)
        P._batch_counts(target)


def progressive_proposals(queue_states, *, registration, tensor_decoder=None, provider=None,
        checkpoint_path=None, pause_after_stages=None, discovery_signals=()):
    """Source-bound fixed widening with durable at-most-once population intent.

    Resume reuses completed rows verbatim. An interrupted charged population is
    censored in its entirety, never retried; completed-but-uncommitted work is
    explicitly lost/unknown. pause_after_stages is invocation cadence only.
    Discovery results enter through a closed allowlist, never the decoder.
    Workers own native evaluation/queue restoration and subsequent admission.
    """
    registration = _batch_registration(registration)
    rows = P._batch_rows(queue_states)
    path = None if checkpoint_path is None else Path(checkpoint_path)
    if pause_after_stages is not None:
        R._int(pause_after_stages, '$.pause_after_stages', 1, 4)
        _need(path is not None, 'PAUSE_REQUIRES_DURABLE_CHECKPOINT')
    _need(type(discovery_signals) in (tuple, list), 'DISCOVERY_SIGNALS_LIST')
    binding = _hash([rows, registration])
    with _lock(path):
        if path is not None and path.exists():
            state = R._loads(path.read_text(encoding='utf-8'))
            _need(state.get('content_sha256') == _hash({k: v for k, v in state.items()
                  if k != 'content_sha256'}), 'CHECKPOINT_HASH')
            _need(state.get('schema') == BATCH_PROGRESS_SCHEMA and state.get('binding_sha256') == binding,
                  'CHECKPOINT_BINDING')
        else:
            state = dict(schema=BATCH_PROGRESS_SCHEMA, binding_sha256=binding,
                registration=registration, status='RUNNING', next_stage=0,
                active_queue_ids=[r['queue_id'] for r in rows], pending=None,
                states=P._batch_empty(rows), stage_receipts=[], discovery_signals=[],
                physical_games=0, simulator_steps=0, actor_training_admissible=False)
            _write(path, state)
        # Validate even extraneous signals. A completed checkpoint must not
        # silently accept protected outcomes, stale evidence or outcome changes.
        for supplied in discovery_signals:
            signal = _batch_signal(supplied, state, registration)
            key = (signal['stage'], signal['queue_id'], signal['candidate_id'], signal['source_group_id'])
            old = next((s for s in state['discovery_signals'] if
                (s['stage'], s['queue_id'], s['candidate_id'], s['source_group_id']) == key), None)
            _need(old is None or R._equal(old, signal), 'DISCOVERY_SIGNAL_CHANGED')
            if old is None:
                _need(state['status'] != 'COMPLETE' and signal['stage'] == state['next_stage'] - 1,
                      'DECISION_ALREADY_FROZEN')
                state['discovery_signals'].append(signal)
        if state['status'] == 'COMPLETE':
            return copy.deepcopy(_seal(state))
        if state['pending'] is not None:
            pending = state['pending']
            active = [r for r in rows if r['queue_id'] in pending['queue_ids']]
            population = P._batch_empty(active)
            for job in P._batch_jobs(active, pending['quotas'], registration['seed'], pending['stage'],
                                     registration['content_sha256']):
                attempt = P._batch_attempt(job)
                attempt['reason_codes'] = ['CENSORED_IN_FLIGHT_POPULATION']
                population[job['row_index']]['attempts'].append(attempt)
            _merge_batch_population(state['states'], population)
            state.update(pending=None, status='COMPLETE', stop_reason='LOST_IN_FLIGHT',
                         active_queue_ids=[])
            _write(path, state)
            return copy.deepcopy(_seal(state))
        state['status'] = 'RUNNING'
        executed = 0
        total_stages = len(registration['stage_multipliers'])
        while state['next_stage'] < total_stages:
            stage = state['next_stage']
            if stage:
                prior = state['stage_receipts'][-1]
                if registration['gate'] == 'ANY_REACHABLE':
                    active_ids = [qid for qid, ids in prior['candidate_ids'].items() if ids]
                else:
                    # Require every preregistered candidate x discovery-group
                    # cell, including losses and censoring. Selecting only a
                    # favorable result after seeing outcomes is not inclusion.
                    signals = [s for s in state['discovery_signals'] if s['stage'] == stage - 1]
                    expected = {(qid, cid, group) for qid, ids in prior['candidate_ids'].items()
                        for cid in ids for group in registration['discovery_groups']}
                    observed = {(s['queue_id'], s['candidate_id'], s['source_group_id']) for s in signals}
                    if observed != expected:
                        state['status'] = 'AWAITING_DISCOVERY'
                        _write(path, state)
                        return copy.deepcopy(_seal(state))
                    active_ids = sorted({s['queue_id'] for s in signals if s['complete'] and
                        s['replay_verified'] and s['wdl_delta'] > 0})
                state['active_queue_ids'] = active_ids
            active = [r for r in rows if r['queue_id'] in state['active_queue_ids']]
            if not active:
                state.update(status='COMPLETE', stop_reason='WIDENING_GATE_CLOSED')
                break
            multiplier = registration['stage_multipliers'][stage]
            quotas = {k: v * multiplier for k, v in registration['quotas'].items()}
            state['pending'] = dict(stage=stage, queue_ids=[r['queue_id'] for r in active], quotas=quotas)
            _write(path, state)  # charge the entire fixed population BEFORE calls
            def bounded_decoder(session):
                if tensor_decoder is None:
                    raise ValueError('TENSOR_DECODER_UNAVAILABLE')
                with _deadline(registration['callback_timeout_seconds']):
                    tensor_decoder(session)
            def bounded_provider(requests):
                with _deadline(registration['callback_timeout_seconds']):
                    return provider(requests)
            limits = dict(registration['limits'])
            used_bytes = sum(r['counts']['record_bytes'] for r in state['stage_receipts'])
            limits['max_record_bytes'] = max(1, limits['max_record_bytes'] - used_bytes)
            result = P.propose_batch(active, quotas=quotas, seed=registration['seed'], stage=stage,
                tensor_decoder=bounded_decoder, provider=bounded_provider if provider else None,
                registration_sha256=registration['content_sha256'], **limits)
            _merge_batch_population(state['states'], result['states'])
            candidate_ids = {r['queue_id']: sorted({a.get('candidate_id', a.get('duplicate_of'))
                for a in r['attempts'] if a['status'] in ('RETAINED', 'DUPLICATE')}) for r in result['states']}
            state['stage_receipts'].append(dict(stage=stage, quotas=quotas,
                pool_sha256=result['content_sha256'], candidate_ids=candidate_ids,
                counts=result['counts'], gate_inputs=_hash(state['discovery_signals'])))
            state.update(pending=None, next_stage=stage + 1)
            executed += 1
            if stage + 1 == total_stages:
                state.update(status='COMPLETE', stop_reason='REGISTERED_POPULATION_EXHAUSTED')
            elif pause_after_stages is not None and executed >= pause_after_stages:
                state['status'] = 'PAUSED'
            _write(path, state)
            if state['status'] != 'RUNNING':
                break
        _write(path, state)
        return copy.deepcopy(_seal(state))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='validated pair supplying actor context/direct draw')
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--budget', type=Path, required=True)
    parser.add_argument('--fixture-outcomes', type=Path, required=True,
                        help='JSON raw-actions SHA256 -> WDL; diagnostic injection ONLY, never native')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args(argv)
    try:
        _need(not args.output.exists(), 'OUTPUT_EXISTS')
        pair = R.validate_pair(R._loads(args.input.read_text(encoding='utf-8')))
        _need(pair['identity']['split'] in ('discovery', 'training'), 'PROTECTED_INPUT_SPLIT')
        protocol = R._loads(args.protocol.read_text(encoding='utf-8'))
        _need(protocol['evidence_scope'] == 'INJECTED_FIXTURE', 'CLI_IS_FIXTURE_ONLY')
        budget = R._loads(args.budget.read_text(encoding='utf-8'))
        outcomes = R._loads(args.fixture_outcomes.read_text(encoding='utf-8'))
        def fixture(request):
            value = outcomes.get(_hash(request['candidate']['raw_actions']))
            return dict(request_sha256=request['request_sha256'], status='COMPLETE' if value is not None else 'FAILED',
                terminal_wdl=value, fresh_terminal_execution=value is not None,
                realization_verified=value is not None, replay_only=False,
                simulator_starts=1, simulator_steps=0, diagnostics={'injected_fixture': True})
        fn = compare_searches if args.compare else improve
        result = fn(pair['actor_context'], pair['learner'], proposer=fixed_proposer, evaluator=fixture,
                    budget=budget, protocol=protocol, seed=args.seed)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n')
        print(json.dumps({'schema': result['schema'], 'content_sha256': result['content_sha256'],
                          'actor_training_admissible': False}, sort_keys=True))
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({'status': 'INVALID', 'reason_code': getattr(exc, 'code', str(exc))}, sort_keys=True))
        return 2
    return 0


# Recovery is a distinct non-opening wire. The accepted opening record, cache,
# MC executor and trainer validators remain deliberately unchanged.
RECOVERY_SCHEMA = 'farm-v0-regret-recovery-v1'
RECOVERY_PROTOCOL = 'farm-v0-regret-recovery-protocol-v1'


def _recovery_protocol(snapshot, learner, protocol, caps):
    """Closed source/context/exposure joins, before either external callback.

    Registrants provide actual reached snapshots, not seeds to splice into a
    state. All source-family replicas must have the same actor-visible context
    AND exact own execution memory/law. Private world/rival/RNG differences stay
    evaluator-only. Hashes cannot authenticate the claimed replay custody or
    scientific independence; these remain explicit producer attestations.
    """
    from farm_neat.rollout import v0_regret_dawn as D
    Q = D.P
    D._bound(protocol, 'recovery protocol')
    D._closed(protocol, ('schema', 'objective', 'aggregation', 'discovery', 'confirmation',
        'protected', 'frozen_before_outcomes', 'proposer_sha256', 'evaluator_sha256',
        'evidence_scope', 'admission', 'content_sha256'), 'recovery protocol')
    _need(protocol['schema'] == RECOVERY_PROTOCOL, 'RECOVERY_PROTOCOL_SCHEMA')
    _need(protocol['objective'] == 'TERMINAL_WDL' and protocol['aggregation'] ==
          'EQUAL_SOURCE_FAMILY_MEAN_V1', 'FROZEN_WDL_RULE_REQUIRED')
    _need(protocol['frozen_before_outcomes'] is True, 'POST_OUTCOME_REGISTRATION')
    _need(protocol['evidence_scope'] in ('INJECTED_FIXTURE', 'NATIVE_GAME'), 'EVIDENCE_SCOPE')
    for key in ('proposer_sha256', 'evaluator_sha256'):
        R._hash(protocol[key], '$.protocol.'+key)
    D._bound(learner, 'recovery learner')
    D._closed(learner, ('snapshot_sha256', 'raw_actions', 'checkpoint_sha256', 'draw_id',
                       'origin', 'content_sha256'), 'recovery learner')
    _need(learner['snapshot_sha256'] == snapshot['content_sha256'], 'LEARNER_SNAPSHOT_MISMATCH')
    R._hash(learner['checkpoint_sha256'], '$.learner.checkpoint_sha256')
    R._text(learner['draw_id'], '$.learner.draw_id')
    def origin(value):
        D._closed(value, ('kind', 'receipt_sha256'), 'reached-state origin')
        _need(value['kind'] in ('LEGAL_REPLAY', 'LEARNER_REACHED'), 'REACHED_ORIGIN_REQUIRED')
        R._hash(value['receipt_sha256'], '$.origin.receipt_sha256')
    origin(learner['origin'])
    context = D.recovery_actor_context(snapshot)
    # Full memory is trainer-only. Do not merge states merely because Plan agrees.
    # game_id is a namespace, not behavior; all other controller fields are exact.
    def own_execution(snap):
        mem = copy.deepcopy(snap['active'])
        mem['game_id'] = 'recovery-context'
        if mem['controller'] is not None:
            mem['controller']['game_id'] = 'recovery-context'
        return Q.wire_hash(mem)
    own = own_execution(snapshot)
    worlds, rows, exposure, rooted = {}, {}, {}, False
    for phase, name in (('discovery', 'discovery'), ('confirmation', 'confirmation'),
                        ('protected_final', 'protected')):
        seq = protocol[name]
        _need(type(seq) is list and (bool(seq) or name == 'protected'), 'EMPTY_WORLD_SET')
        _need(len(seq) <= 8192, 'WORLD_REGISTRY_LIMIT')
        worlds[phase] = []
        local = set()
        for row in seq:
            D._closed(row, ('source_group_id', 'source_family_id', 'snapshot', 'origin'), 'recovery group')
            for key in ('source_group_id', 'source_family_id'):
                R._text(row[key], '$.group.'+key)
            origin(row['origin'])
            snap = row['snapshot']
            visible = D.recovery_actor_context(snap)  # verifies original exact sources
            _need(Q.wire(visible) == Q.wire(context), 'INCOMPATIBLE_ACTOR_CONTEXT')
            _need(own_execution(snap) == own, 'INCOMPATIBLE_EXECUTION_MEMORY')
            _need(snap['continuation'] == snapshot['continuation'] and
                  snap['agent_type'] == snapshot['agent_type'], 'CONTINUATION_LAW_MISMATCH')
            # Own future policy state is fixed, not secretly varied across replicas.
            _need(snap['actor_policy'] == snapshot['actor_policy'], 'ACTOR_POLICY_STATE_MISMATCH')
            world = dict(source_group_id=row['source_group_id'], source_family_id=row['source_family_id'],
                world_sha256=Q.wire_hash(snap['world']['scenario_identity']),
                snapshot_sha256=snap['content_sha256'],
                rng_sha256=Q.wire_hash([snap['engine_rng'], snap['actor_policy']['rng'],
                                      snap['opponent_policy']['rng']]),
                seed=snap['world']['scenario']['seed'])
            for field, value in world.items():
                token = (field, value)
                _need(token not in exposure or exposure[token] == phase, 'SELECTION_CONFIRMATION_OVERLAP')
                exposure[token] = phase
                if field != 'source_family_id':
                    _need(token not in local, 'DUPLICATE_REGISTERED_WORLD')
                    local.add(token)
            worlds[phase].append(world)
            rows[row['source_group_id']] = row
            if snap['content_sha256'] == snapshot['content_sha256']:
                _need(phase == 'discovery' and row['origin'] == learner['origin'], 'ROOT_ORIGIN_MISMATCH')
                rooted = True
    _need(rooted, 'ROOT_MUST_BE_REGISTERED_DISCOVERY')
    _need(len({w['source_family_id'] for w in worlds['discovery']}) <= caps['max_discovery_groups'],
          'WORLD_CAP_BELOW_REGISTRATION')
    a = protocol['admission']
    D._closed(a, ('family_id', 'round_id', 'family_alpha', 'tested_pairs', 'alpha_pair',
        'min_effect', 'n_world_groups', 'independence_attested', 'compatible_contexts_attested',
        'frozen_before_confirmation'), 'recovery admission')
    for key in ('family_id', 'round_id'):
        R._text(a[key], '$.admission.'+key)
    R._number(a['family_alpha'], '$.admission.family_alpha', 0, .05)
    # This bounded API conducts ONE fixed test. Larger cross-search families need
    # the normal multi-test collector, not a caller-chosen unverified denominator.
    _need(type(a['tested_pairs']) is int and a['tested_pairs'] == 1, 'SINGLE_FROZEN_RECOVERY_TEST_REQUIRED')
    R._number(a['alpha_pair'], '$.admission.alpha_pair', 0, .05)
    _need(0 < a['alpha_pair'] == a['family_alpha'], 'ALPHA_ALLOCATION_MISMATCH')
    R._number(a['min_effect'], '$.admission.min_effect', 0, 2)
    R._int(a['n_world_groups'], '$.admission.n_world_groups', 1, 8192)
    _need(a['n_world_groups'] == len({w['source_family_id'] for w in worlds['confirmation']}),
          'FIXED_SAMPLE_COUNT_MISMATCH')
    for key in ('independence_attested', 'compatible_contexts_attested', 'frozen_before_confirmation'):
        _need(a[key] is True, key.upper()+'_REQUIRED')
    return context, worlds, rows


def _recovery_candidate(snapshot, actions, context, menus, *, edit_limit):
    """Replay each candidate from the actual dawn, never an incumbent's cursor."""
    from farm_neat.rollout import v0_regret_dawn as D
    from farm_neat.policy.v0_scorer import Prefix
    env, agent = D.validate_snapshot(snapshot, continuation=snapshot['continuation'])
    state, prefix = D.P.State(env.agent_observation(snapshot['seat'])), Prefix(agent.plan)
    screen = P._syntax_screen(prefix, actions)
    _need(screen['status'] != 'INVALID', '|'.join(screen.get('reason_codes', ['INVALID_ACTIONS'])))
    _need(len(actions)-1 <= edit_limit, 'PATH_SEARCH_BUDGET')
    traced = P._trace(state, prefix, agent.projection_kw(), menus, actions)
    _need(traced['status'] == 'REACHABLE', '|'.join(traced.get('reason_codes', ['UNREACHABLE'])))
    # Controller/started/allocation stay exact. Publishing will use bind_holds
    # through the existing agent; here we retain its independently derived lift.
    agent.bind_holds(state, agent.plan, traced['final'].plan)
    full = dict(decoder=traced['continuation_signature'],
        starting_active_sha256=D.P.wire_hash(snapshot['active']),
        starting_actor_policy_sha256=D.P.wire_hash(snapshot['actor_policy']),
        hold_releases_after_binding=V0Continuation.capture(agent).to_dict()['hold_releases'])
    return dict(raw_actions=copy.deepcopy(actions), canonical_plan=traced['canonical_plan'],
        prefix_class_trace=traced['prefix_class_trace'],
        continuation_signature=full, discovery_score=None, discovery_statistics=None,
        reachability='EXACT_CANONICAL_DAWN_PREFIX', actor_training_admissible=False)


def _recovery_branch_result(value, request):
    """Verify receipt joins and terminal semantics, not merely a claimed score."""
    from farm_neat.rollout import v0_regret_dawn as D
    D._bound(value, 'recovery branch result')
    required = ('schema', 'request_sha256', 'attempt_id', 'initial_snapshot_sha256', 'proposal_sha256',
        'status', 'terminal_wdl', 'cash', 'actions_sha256', 'terminal_runtime_sha256', 'simulator_starts',
        'simulator_steps', 'physical_starts', 'evidence_scope', 'error', 'content_sha256')
    R._fields(value, required, '$.branch_result', optional=('cpu_seconds', 'elapsed_seconds'))
    _need(value['schema'] == 'farm-v0-regret-recovery-branch-result-v1', 'BRANCH_RESULT_SCHEMA')
    for key in ('attempt_id', 'evidence_scope'):
        _need(value[key] == request[key], 'BRANCH_RESULT_JOIN')
    _need(value['request_sha256'] == request['content_sha256'] and
          value['initial_snapshot_sha256'] == request['branch']['snapshot']['content_sha256'] and
          value['proposal_sha256'] == request['branch']['proposal']['content_sha256'], 'BRANCH_RESULT_JOIN')
    _need(type(value['simulator_starts']) is int and value['simulator_starts'] == 1, 'SINGLE_START_REQUIRED')
    physical = int(request['evidence_scope'] == 'NATIVE_GAME')
    _need(type(value['physical_starts']) is int and value['physical_starts'] == physical, 'PHYSICAL_SCOPE_MISMATCH')
    for key in ('actions_sha256', 'terminal_runtime_sha256'):
        R._hash(value[key], '$.branch_result.'+key)
    for key in ('cpu_seconds', 'elapsed_seconds'):
        if key in value:
            R._number(value[key], '$.branch_result.'+key, 0)
    remaining = V.LAST_STEP + 1 - request['branch']['snapshot']['step']
    R._int(value['simulator_steps'], '$.branch_result.simulator_steps', 0, remaining)
    _need(value['status'] in ('COMPLETE', 'CENSORED', 'FAILED'), 'BRANCH_STATUS')
    if value['status'] == 'COMPLETE':
        _need(value['error'] is None, 'COMPLETE_WITH_ERROR')
        # LAST_STEP is the last ACTION index; a terminal run reaches LAST_STEP+1.
        _need(value['simulator_steps'] == remaining, 'NONTERMINAL_RESULT')
        _need(type(value['cash']) is list and len(value['cash']) == 2, 'CASH_TUPLE_REQUIRED')
        for money in value['cash']:
            R._number(money, '$.branch_result.cash')
        seat = request['branch']['snapshot']['seat']
        x, y = value['cash'][seat], value['cash'][1-seat]
        _need(type(value['terminal_wdl']) is int and value['terminal_wdl'] == int(x > y)-int(x < y),
              'TERMINAL_WDL_MISMATCH')
    else:
        _need(value['terminal_wdl'] is None and value['cash'] is None, 'PARTIAL_OUTCOME_FORBIDDEN')
    return copy.deepcopy(value)


def search_recovery(snapshot: dict, learner: dict, *, proposer, evaluator,
                    budget: dict, protocol: dict, seed: int) -> dict:
    """Search exact learner-reached dawns; teach only disjoint positive WDL gain.

    Proposer: ``(actor_context, incumbent_geometry, *, budget, seed) ->`` one
    ``{proposal_family, raw_actions}`` (or status UNKNOWN with reason_codes).
    Evaluator: one bound request -> one fresh terminal branch result. Use
    ``v0_regret_dawn.execute_recovery_branch`` for explicitly authorized native
    runs, or an honest INJECTED_FIXTURE test adapter. Callbacks are trusted code.

    Each iteration nominates from the incumbent, then canonicalizes from the
    ORIGINAL current dawn. The accepted MC allocator schedules two-arm, complete
    discovery panels, including fresh exact control/challenger replays (eight
    starts/world). Strict expected WDL improvement updates the incumbent. No
    cache hit, partial family, replay or duplicate adds independent precision.
    All costs, failed attempts and losers survive in this receipt.

    The winner is frozen before ONE disjoint fixed-N confirmation (four starts/
    world). Its equal-family paired WDL LCB uses the ordinary admission rule
    ``mean - sqrt(8 * -log(alpha) / n) > min_effect`` and weight LCB/2 capped at1.
    A new schema is necessary: existing pair-v1/MC-cache/trainers explicitly
    reject non-opening controller memory. Do NOT coerce this to an opening row.
    This API does not load models, train, schedule jobs or authenticate history.
    No durable resume: callbacks are never retried within an invocation; callers
    must not rerun a confirmation until success. Use externally managed test IDs.
    """
    from farm_neat.rollout import v0_regret_dawn as D
    from farm_neat.training.teachers import v0_regret_budget as B
    Q = D.P
    started, cpu_started = time.monotonic(), time.process_time()
    caps, path, pause = _limits(budget)
    _need(path is None and pause is None, 'RECOVERY_CHECKPOINT_NOT_SUPPORTED')
    R._int(seed, '$.seed', 0, 2**63-1)
    _need(callable(proposer) and callable(evaluator), 'TRUSTED_CALLBACKS_REQUIRED')
    # Detached inputs; even a malicious callback cannot mutate the frozen sources.
    snapshot, learner, protocol = copy.deepcopy((snapshot, learner, protocol))
    context, worlds, rows = _recovery_protocol(snapshot, learner, protocol, caps)
    pins = {**D.source_pins(), **B.source_pins(),
            'src/farm_neat/training/teachers/v0_regret_searcher.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    seat = snapshot['seat']; contexts = {str(seat): Q.wire_hash(context)}
    menus, candidates, proposals, attempts, allocations, trace = {}, [], [], [], [], []
    counts = dict(attempted_candidates=1, iterations=0, attempted_simulator_starts=0,
                  physical_starts_reported=0, physical_starts_unknown=0, simulator_steps_reported=0)
    control_signatures, integrity_errors = {}, []
    def remaining():
        return max(0., caps['max_wall_seconds']-(time.monotonic()-started))
    def candidate(actions, limit, family):
        c = _recovery_candidate(snapshot, actions, context, menus, edit_limit=limit)
        c['paths'] = {str(seat): B.path_binding(c)}
        c['candidate_id'] = B.candidate_key(c['paths'], contexts)
        c['proposal_family'] = family
        return c
    # The genuine learner path is not restricted by the challenger search edit
    # budget; it still obeys native max16+STOP and exact current-prefix legality.
    direct = candidate(learner['raw_actions'], V.MAX_EDITS_PER_EPOCH, 'learner')
    candidates.append(direct); best = direct
    limits = B.register_budget(dict(schema=B.LIMITS_SCHEMA,
        comparison_id=Q.wire_hash([snapshot['content_sha256'], protocol['content_sha256'], seed]),
        arm='fixed', evidence_kind=protocol['evidence_scope'],
        parent_checkpoint_sha256=learner['checkpoint_sha256'], source_pins=B.source_pins(),
        source_group_id=next(w['source_group_id'] for w in worlds['discovery']
                             if w['snapshot_sha256'] == snapshot['content_sha256']),
        actor_context_sha256s=contexts, learner_path_sha256s={str(seat): Q.wire_hash(learner['raw_actions'])},
        opponent_sha256=Q.wire_hash(snapshot['continuation']['opponent_identity']), seats=[seat], worlds=worlds,
        caps=dict(candidate_attempts=max(2, caps['max_candidates']), edit_depth=caps['max_path_edits'],
            independent_discovery_groups=caps['max_discovery_groups'], native_starts=caps['max_simulator_starts'],
            cpu_seconds=max(.000001, caps['max_wall_seconds']), elapsed_seconds=max(.000001, caps['max_wall_seconds'])),
        seed=seed, cost_match_relative_tolerance=0., frozen_before_outcomes=True,
        budget_basis='PREDECLARED_FIXED', protected_final_exposed=False, retry_until_success=False))
    def signature(result):
        return Q.wire_hash({k: result[k] for k in ('initial_snapshot_sha256', 'proposal_sha256',
            'terminal_wdl', 'cash', 'actions_sha256', 'terminal_runtime_sha256', 'simulator_steps')})
    def one(group, arm, role, phase, selection):
        if counts['attempted_simulator_starts'] >= caps['max_simulator_starts'] or remaining() <= 0:
            return None
        snap = rows[group]['snapshot']
        # Revalidate against THIS exact state's current prefix, not a copied target.
        proposal = D.propose_dawn(snap, arm['raw_actions'], continuation=snap['continuation'])
        branch = D.fork_dawn(snap, proposal, continuation=snap['continuation'])
        timeout = min(remaining(), caps['evaluation_timeout_seconds'])
        if timeout <= 0:
            return None
        i = counts['attempted_simulator_starts']
        request = Q.bind(dict(schema='farm-v0-regret-recovery-branch-request-v1', branch=branch,
            phase=phase, role=role, attempt_id=Q.wire_hash([limits['content_sha256'], i, phase, role]),
            candidate_id=arm['candidate_id'], source_group_id=group,
            source_family_id=rows[group]['source_family_id'], protocol_sha256=protocol['content_sha256'],
            selection_sha256=selection, evaluator_sha256=protocol['evaluator_sha256'],
            evidence_scope=protocol['evidence_scope'], timeout_seconds=timeout))
        counts['attempted_simulator_starts'] += 1  # charge BEFORE callback
        before, proc = time.monotonic(), time.process_time()
        record = dict(request=request, response=None, error=None, validated=False)
        attempts.append(record)
        result = None
        try:
            with _deadline(timeout):
                raw = evaluator(copy.deepcopy(request))
            R._safe(raw)  # Unsafe objects are not allowed to poison the durable receipt.
            record['response'] = copy.deepcopy(raw)
            result = _recovery_branch_result(raw, request)
            record['validated'] = True
            counts['physical_starts_reported'] += result['physical_starts']
            counts['simulator_steps_reported'] += result['simulator_steps']
        except Exception as exc:
            record['error'] = type(exc).__name__ + ': ' + str(exc)
            counts['physical_starts_unknown'] += int(protocol['evidence_scope'] == 'NATIVE_GAME')
        record.update(cpu_seconds=time.process_time()-proc, elapsed_seconds=time.monotonic()-before)
        return result
    def paired(group, arm, phase, selection=None):
        begin = len(attempts)
        values = [one(group, c, role, phase, selection) for c, role in (
            (direct, 'learner'), (arm, 'challenger'), (direct, 'learner_replay'), (arm, 'challenger_replay'))]
        complete = all(v is not None and v['status'] == 'COMPLETE' for v in values)
        reason = None
        if complete and any(signature(values[i]) != signature(values[i+2]) for i in (0, 1)):
            complete, reason = False, 'REPLAY_MISMATCH'
        if complete:
            control = signature(values[0])
            if group in control_signatures and control_signatures[group] != control:
                complete, reason = False, 'CONTROL_MISMATCH'
            control_signatures[group] = control
        if reason is not None:
            integrity_errors.append(dict(phase=phase, source_group_id=group,
                                         candidate_id=arm['candidate_id'], reason=reason))
        used = attempts[begin:]
        cell = dict(source_group_id=group, source_family_id=rows[group]['source_family_id'],
            candidate_id=arm['candidate_id'], seat=seat, complete=complete,
            learner_wdl=values[0]['terminal_wdl'] if complete else None,
            challenger_wdl=values[1]['terminal_wdl'] if complete else None,
            learner_replay_sha256=signature(values[0]) if complete else None,
            simulator_starts=len(used),
            physical_starts=int(protocol['evidence_scope']=='NATIVE_GAME')*len(used),
            attempt_ids=[a['request']['attempt_id'] for a in used],
            reason=reason or (None if complete else 'CENSORED_OR_MISSING_OUTCOMES'))
        return cell
    stop_reason = 'ITERATION_CAP'
    for iteration in range(caps['max_iterations']):
        if len(proposals)+1 >= caps['max_candidates']:
            stop_reason = 'CANDIDATE_CAP'; break
        if remaining() <= 0:
            stop_reason = 'WALL_CAP'; break
        counts['iterations'] += 1; counts['attempted_candidates'] += 1
        incumbent_before = best['candidate_id']
        # No source hashes, controller, private state, outcomes or evidence in the
        # proposer input. Its own prefix geometry is recomputed, not teacher-fed.
        visible = {k: copy.deepcopy(best[k]) for k in ('raw_actions', 'canonical_plan', 'prefix_class_trace')}
        row = dict(iteration=iteration, incumbent_id=incumbent_before, status='FAILED',
                   response=None, candidate_id=None, reason_codes=[])
        proposals.append(row)
        try:
            with _deadline(min(remaining(), caps['evaluation_timeout_seconds'])):
                spec = proposer(copy.deepcopy(context), visible,
                                budget=dict(max_path_edits=caps['max_path_edits']), seed=seed+iteration)
            R._safe(spec)
            row['response'] = copy.deepcopy(spec)
            R._fields(spec, ('proposal_family',), '$.recovery_proposal',
                      optional=('raw_actions', 'status', 'reason_codes'))
            R._text(spec['proposal_family'], '$.recovery_proposal.proposal_family')
            _need(spec.get('status') != 'UNKNOWN', '|'.join(spec.get('reason_codes') or ['UNKNOWN_PROPOSAL']))
            c = candidate(spec.get('raw_actions'), caps['max_path_edits'], spec['proposal_family'])
            row['candidate_id'] = c['candidate_id']
            if any(x['candidate_id'] == c['candidate_id'] for x in candidates):
                row.update(status='DUPLICATE', reason_codes=['EXACT_CONTINUATION_DUPLICATE'])
                continue
            candidates.append(c); row['status'] = 'REACHABLE'
        except Exception as exc:
            row['reason_codes'] = [getattr(exc, 'code', str(exc))]
            continue
        # Each iterative candidate has the same preregistered panel and controls.
        # The accepted allocator owns family batches and uncertainty; no opening
        # records are forged to get through the opening-only MC executor/cache.
        local_spec = {k: copy.deepcopy(v) for k, v in limits.items() if k != 'content_sha256'}
        local_spec['caps']['native_starts'] = caps['max_simulator_starts']-counts['attempted_simulator_starts']
        local_spec['caps']['elapsed_seconds'] = max(.000001, remaining())
        registration = B.register_mc_allocation(B.register_budget(local_spec),
            [dict(candidate_id=x['candidate_id'], paths=x['paths']) for x in (direct, c)],
            mode='uniform', minimum_families=1)
        allocator = B.MonteCarloAllocator(registration)
        history = []; raw_cells = []
        while remaining() > 0:
            req = allocator.next_request()
            if req is None:
                break
            if req['reserved_start_slots'] > caps['max_simulator_starts']-counts['attempted_simulator_starts']:
                break
            begin, proc, count = time.monotonic(), time.process_time(), len(attempts)
            event_cells = []
            for key in req['cells']:
                cell = paired(key['source_group_id'], direct if key['candidate_id'] == direct['candidate_id'] else c,
                              'discovery')
                raw_cells.append(cell)
                event_cells.append({**key, **{k: cell[k] for k in ('learner_wdl', 'challenger_wdl',
                    'learner_replay_sha256', 'simulator_starts', 'physical_starts')},
                    'status': 'COMPLETE' if cell['complete'] else 'CENSORED' if cell['simulator_starts'] else 'NOT_STARTED'})
            event = B.bind(dict(schema=B.MC_ALLOCATION_EVENT_SCHEMA, request_sha256=req['content_sha256'],
                phase='discovery', cells=event_cells, cpu_seconds=time.process_time()-proc,
                elapsed_seconds=time.monotonic()-begin, producer_receipt_sha256=Q.wire_hash(attempts[count:])))
            allocator.observe(event); history.append(event)
        receipt = allocator.receipt()
        allocations.append(dict(registration=registration, history=history, receipt=receipt, cells=raw_cells))
        n = len({w['source_family_id'] for w in worlds['discovery']})
        stats = receipt['statistics']
        all_complete = all(stats[x['candidate_id']]['verified_family_n'] == n for x in (direct, c))
        for x in (direct, c):
            x['discovery_statistics'] = stats[x['candidate_id']]
            if all_complete:
                x['discovery_score'] = stats[x['candidate_id']]['utility_mean']
        improved = all_complete and c['discovery_score'] > best['discovery_score']
        if improved:
            best = c
        trace.append(dict(iteration=iteration, incumbent_before=incumbent_before, attempted=c['candidate_id'],
            incumbent_after=best['candidate_id'], strict_wdl_improvement=improved,
            all_registered_families_complete=all_complete, allocator_receipt_sha256=receipt['content_sha256']))
        if counts['attempted_simulator_starts'] >= caps['max_simulator_starts']:
            stop_reason = 'SIMULATOR_START_CAP'; break
    frozen = Q.bind(dict(schema='farm-v0-regret-recovery-selection-v1',
        snapshot_sha256=snapshot['content_sha256'], learner_sha256=learner['content_sha256'],
        protocol_sha256=protocol['content_sha256'], source_pins=pins,
        candidate_id=best['candidate_id'], raw_actions=best['raw_actions'], canonical_plan=best['canonical_plan'],
        continuation_signature=best['continuation_signature'], discovery_trace_sha256=Q.wire_hash(trace),
        discovery_evidence_sha256=Q.wire_hash(attempts), confirmation_registration_sha256=Q.wire_hash(worlds['confirmation']),
        budget=caps, seed=seed, actor_training_admissible=False))
    confirm = dict(schema='farm-v0-regret-recovery-confirmation-v1',
        selection_sha256=frozen['content_sha256'], cells=[], mean_wdl_delta=None, lower_bound=None,
        learner_utility_mean=None, challenger_utility_mean=None, n_world_groups=0,
        registered_world_groups=protocol['admission']['n_world_groups'],
        uncertainty='FIXED_N_BOUNDED_PAIRED_WDL_HOEFFDING_ONE_SIDED', reason_codes=[])
    if integrity_errors:
        confirm['reason_codes'] = ['REPLAY_OR_CONTROL_INTEGRITY_FAILURE']
    elif best['candidate_id'] == direct['candidate_id']:
        confirm['reason_codes'] = ['NO_DISCOVERY_WDL_ADVANTAGE']
    elif 4*len(worlds['confirmation']) > caps['max_simulator_starts']-counts['attempted_simulator_starts']:
        confirm['reason_codes'] = ['CONFIRMATION_BUDGET_BELOW_FIXED_N']
    elif remaining() <= 0:
        confirm['reason_codes'] = ['WALL_CAP']
    else:
        for world in worlds['confirmation']:
            confirm['cells'].append(paired(world['source_group_id'], best, 'confirmation', frozen['content_sha256']))
        families = {}
        for cell in confirm['cells']:
            families.setdefault(cell['source_family_id'], []).append(cell)
        complete = all(c['complete'] for c in confirm['cells'])
        confirm['n_world_groups'] = sum(all(c['complete'] for c in cells) for cells in families.values())
        if not complete:
            confirm['reason_codes'] = ['CENSORED_OR_MISSING_OUTCOMES']
        else:
            for arm in ('learner', 'challenger'):
                confirm[arm+'_utility_mean'] = math.fsum(
                    math.fsum(c[arm+'_wdl'] for c in cs)/len(cs) for cs in families.values())/len(families)
            delta = math.fsum(math.fsum(c['challenger_wdl']-c['learner_wdl'] for c in cs)/len(cs)
                              for cs in families.values())/len(families)
            # Identical to v0_regret_admission.admit_pair's fixed-N ordinary gate;
            # source-bound separately because pair-v1 correctly rejects dawns.
            confirm['mean_wdl_delta'] = delta
            confirm['lower_bound'] = delta-math.sqrt(8*-math.log(protocol['admission']['alpha_pair'])/len(families))
            if confirm['lower_bound'] <= protocol['admission']['min_effect']:
                confirm['reason_codes'] = ['NO_MEAN_WDL_ADVANTAGE' if delta == 0 else
                                          'REALIZED_LEARNER_BETTER' if delta < 0 else 'INSUFFICIENT_POWER']
    proven = not confirm['reason_codes'] and confirm['lower_bound'] is not None
    counts.update(cpu_seconds=time.process_time()-cpu_started, elapsed_seconds=time.monotonic()-started)
    result = dict(schema=RECOVERY_SCHEMA, status='PROVEN_RECOVERY' if proven else 'NO_PROVEN_RECOVERY',
        teacher_weight=min(2., max(0., confirm['lower_bound']))/2. if proven else 0.,
        actor_training_admissible=proven, statistical_support=proven, evidence_scope=protocol['evidence_scope'],
        record_compatibility='DAWN_RECOVERY_ONLY_NOT_OPENING_PAIR_V1', snapshot_sha256=snapshot['content_sha256'],
        learner=learner, actor_context=context, protocol=protocol, source_pins=pins,
        candidates=candidates, best=best, proposals=proposals, search_trace=trace, allocations=allocations,
        attempts=attempts, integrity_errors=integrity_errors, frozen_selection=frozen, confirmation=Q.bind(confirm), counts=counts,
        budget=caps, stop_reason=stop_reason, seed=seed,
        limitations=['CALLER_ATTESTS_REACHABILITY_INDEPENDENCE_AND_NO_UNDISCLOSED_EXPOSURE',
            'INJECTED_FIXTURE_OUTCOMES_ARE_NOT_NATIVE_RECOVERY_OR_POLICY_STRENGTH',
            'NO_DAWN_CACHE_REUSE_NO_DURABLE_RESUME_NO_AUTOMATIC_TRAINER_INTAKE',
            'DECLARED_COMPATIBLE_DAWN_POPULATION_NOT_ARBITRARY_HIDDEN_STATE_VALUE',
            'TRUSTED_CALLBACKS_MUST_NOT_SWALLOW_DEADLINES_OR_LAUNCH_UNCONTROLLED_CHILDREN'])
    return Q.bind(result)


if __name__ == '__main__':
    raise SystemExit(main())
