"""Bounded, one-dawn V0 correction oracle; no training, GPU, or frozen-agent edits.

The core takes an actor observation + current Plan + complete V0Agent snapshot,
validates every edit on its own prefix, and searches explicit one-key mutations.
It does NOT enumerate the large menu or change the executor. The shared V0Agent
uses the same actual-prefix admission contract; no additional dawn-entry gate
rejects support enabled by earlier canonical edits.

Native measurement uses router_panel.play_game in a NEW PROCESS per attempt.
Each case plays its baseline and admitted alternatives, then independently
replays BOTH baseline and selected decision. All attempts count; no resume or
retry after failure. Selection maximizes own terminal CASH, not margin or WDL.
Ties retain the original decision. This is best-in-measured-neighborhood only.
The suffix policy, world, rival, seat, source closure and budget are fixed before
outcomes. A decision changes at ONE dawn; future epochs use the same frozen
suffix policy. Native replay reconstructs the complete prefix from opening and
checks all V0Agent memory, including allocation, started instances, executor
publication/credit and HOLD bindings, before changing the decision.

Outputs include a conditional correction label, not automatic admission to the
opening-only v0_targets exporter. Actor inputs are allowlisted public projection
facts and own memory; world/seed/future shops, full replay context and measurements
are separate trainer_only data. The smoke uses a PRELOADED opening Plan, not a
claim of opening reachability. All worlds are exposed fixtures, not ladder proof.

Commands (new output paths only)::

  PYTHONPATH=src python3 -B -m farm_neat.training.teachers.v0_oracle smoke \
    --output work/runs/v0-oracle-smoke --max-games 4 --seconds 300
  PYTHONPATH=src python3 -B -m farm_neat.training.teachers.v0_oracle run \
    --request REQUEST.json --output work/runs/v0-oracle-case --max-games 4
  # MAINTAINER ONLY: 72 worlds x 2 seats x 4 rivals x 2 stubs x <=4 attempts.
  PYTHONPATH=src python3 -B -m farm_neat.training.teachers.v0_oracle panel \
    --output work/runs/v0-oracle-panel --max-games 4608 --seconds 86400

A custom request freezes a program (dawn, initial_plan, prior_actions, policy,
suffix_policy), registered world, seat, rival, explicit action_pool, max_trials
and max_candidates. Pool integers add/replace that key in the ORIGINAL decision;
None proposes deleting each original key. Every proposal restarts from the SAME
dawn and revalidates the complete canonical trace. KeepPolicy/TargetPolicy are
native stubs; learned callers can supply an already generated decision to the
core without changing the search or label format. No teacher prefixes are fed
into candidate execution. Rejected proposals and losing measured candidates stay
in the result. Native outcomes and arithmetic unit fixtures remain distinct.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from farm_neat.economy.orders import FullSeasonPlan
from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.epochs import KeepPolicy, TargetPolicy, V0Agent
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import Masks, State
from farm_neat.evaluation import router_panel as panel
from farm_neat.evaluation.v0_continuation import V0Continuation
from farm_neat.evaluation.common import bound, canonical, digest, equal, file_sha, read, write_new
from farm_neat.training.v0_targets import actor_observation

ROOT = panel.ROOT
MODULE = 'farm_neat.training.teachers.v0_oracle'
SCHEMA = 'farm-v0-dawn-oracle-v2'
SNAPSHOT_FIELDS = {'schema', 'continuation_schema', 'land_commitment', 'plan', 'allocation', 'started', 'hold_releases', 'controller',
                   'game_id', 'labor_factor', 'hands_mode', 'belief'}
RIVALS = ('router', 'k0006', 'tape0', 'pass')


def _integer(value, lo: int, hi: int, name: str) -> int:
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in [{lo},{hi}]')
    return value


def _plan(value: dict) -> Plan:
    from farm_neat.training.teachers.v0_plan_search import strict_plan
    return strict_plan(value)


def _holds(agent: V0Agent) -> list:
    return [[p, list(pair), release] for p, (pair, release) in sorted(agent.hold_releases.items())]


def snapshot_agent(agent: V0Agent) -> dict:
    """Oracle envelope over the canonical V0Agent-owned continuation memory.

    Policy/RNG remains separately frozen by the oracle request; records remain
    diagnostics.  V0Agent owns the mutable executor-memory field definitions.
    """
    memory = V0Continuation.capture(agent).to_dict()
    memory['continuation_schema'] = memory.pop('schema')
    return copy.deepcopy({'schema': SCHEMA + '-agent-memory', **memory})


def restore_agent(memory: dict, policy, *, observation=None) -> V0Agent:
    """Restore oracle memory through V0Agent's canonical continuation codec."""
    if not isinstance(memory, dict) or set(memory) != SNAPSHOT_FIELDS or memory['schema'] != SCHEMA + '-agent-memory':
        raise ValueError('closed V0Agent memory required')
    if memory['hands_mode'] not in ('daily', 'opening'):
        raise ValueError('unsupported hands mode')
    belief = memory['belief']
    if belief is not None and (not isinstance(belief, dict) or set(belief) - set(V.PRODUCTS)
                               or any(type(v) not in (int, float) or v != 0 for v in belief.values())):
        raise ValueError('oracle v1 requires zero-flow belief')
    payload = {key: copy.deepcopy(value) for key, value in memory.items() if key != 'schema'}
    payload['schema'] = payload.pop('continuation_schema')
    agent = V0Continuation.from_dict(payload).restore(policy, observation=observation)
    equal(snapshot_agent(agent), memory, 'memory round trip (duplicate/coerced entry)')
    return agent


def capture_dawn(agent: V0Agent, observation: dict) -> dict:
    """Capture at epoch entry, AFTER controller.observe and BEFORE policy edits.

    Direct unit callers can also capture an unstarted opening controller. Native
    callers are captured by DawnActor at the exact V0Agent epoch boundary.
    """
    step = _integer(observation['step'], 0, V.LAST_STEP, 'dawn step')
    if step % 24:
        raise ValueError('dawn boundary required')
    _integer(observation['player'], 0, 1, 'seat')
    return bound({'schema': SCHEMA + '-dawn', 'observation': copy.deepcopy(observation),
                  'memory': snapshot_agent(agent)})


def actor_input(dawn: dict) -> dict:
    """No raw-observation, game-ID, world or controller-provenance passthrough.

    Executor state remains in trainer_only replay context. This conservative
    feature boundary is NOT a sufficiency claim or an existing scorer schema.
    """
    memory = dawn['memory']
    return {'public_observation': actor_observation(State(dawn['observation'])),
            'current_plan': copy.deepcopy(memory['plan']),
            'own_memory': {k: copy.deepcopy(memory[k]) for k in
                           ('land_commitment', 'allocation', 'started', 'hold_releases', 'labor_factor', 'hands_mode', 'belief')}}


def _sequence(actions) -> list[int]:
    if not isinstance(actions, (list, tuple)) or not actions or actions[-1] != V.STOP_ID:
        raise ValueError('explicit final STOP required')
    actions = list(actions)
    if len(actions) > V.MAX_EDITS_PER_EPOCH + 1:
        raise ValueError('edit cap exceeded')
    last = -1
    for index, action in enumerate(actions):
        _integer(action, 0, V.N_ACTIONS - 1, 'action ID')
        if action == 0:
            if index != len(actions) - 1:
                raise ValueError('no actions after STOP')
        else:
            key = V.plan_key(action)
            if key <= last:
                raise ValueError('strict increasing canonical keys required; no normalization')
            last = key
    return actions


def trace_edits(dawn: dict, actions, *, require_execution: bool = True) -> dict:
    """Prove the exact requested path, not an approximately reached target.

    Each successor payload is the v0_scorer.Prefix payload, including STOP and
    edit-cap continuation. The class ID is its canonical JSON SHA256. Phi/menu
    materialization is deliberately not required for bounded candidate search.

    epoch_compatible certifies V0Agent's prefix-local EDIT admission contract,
    not successful compilation/publication or native realization. A returned
    trace has passed that contract, including exact successor/HOLD continuation.
    require_execution is retained for caller compatibility; both values now
    enforce the same strict path checks. False never bypasses prefix legality.
    """
    equal(dawn, bound(dawn), 'dawn content binding')
    if dawn.get('schema') != SCHEMA + '-dawn':
        raise ValueError('oracle dawn schema required')
    actions = _sequence(actions)
    state = State(dawn['observation'])
    if not 0 <= state.step <= V.LAST_STEP or state.step % 24:
        raise ValueError('dawn boundary required')
    agent = restore_agent(dawn['memory'], KeepPolicy(), observation=dawn['observation'])
    state = agent.actor_state(dawn['observation'], initialize=False)
    current = agent.plan.copy()
    initial_masks = Masks(state, current, **agent.projection_kw())
    rows, last, used = [], -1, 0
    for action in actions:
        prefix = {'plan': current.canonical(), 'last_key': last, 'edit_count': used, 'terminated': False}
        stopped = action == V.STOP_ID
        forced = stopped and used == V.MAX_EDITS_PER_EPOCH
        if not stopped:
            masks = initial_masks if used == 0 else Masks(state, current, **agent.projection_kw())
            if not masks.legal(action):
                raise ValueError(f'masked at actual prefix {used}: {V.describe(action)}')
            current = current.apply(action, masks.resolver)
            last, used = V.plan_key(action), used + 1
        successor = {'plan': current.canonical(), 'last_key': last, 'edit_count': used, 'terminated': stopped}
        rows.append({'action': action, 'description': V.describe(action), 'prefix': prefix,
                     'successor': successor, 'successor_class_id': digest(successor), 'forced_stop': forced})
    agent.bind_holds(state, agent.plan, current)
    return bound({'schema': SCHEMA + '-trace', 'dawn_sha256': dawn['content_sha256'],
                  'actions': actions, 'steps': rows, 'plan': current.canonical(),
                  'hold_releases': _holds(agent), 'epoch_compatible': True})


def _successor_key(trace: dict) -> str:
    # Equal Plans alone are insufficient: canonical decoder continuation and
    # compiler HOLD continuation must also agree. Raw action aliases are absent.
    return canonical([trace['steps'][-1]['successor'], trace['hold_releases']])


def neighborhood(dawn: dict, baseline_actions, action_pool, *, max_trials=64, max_candidates=2) -> dict:
    """Bounded one-key insertion/replacement/deletion, always from original input.

    None expands to one deletion per baseline key. Insertion may change the
    legality/resolution of later actions; trace_edits rebuilds each such prefix.
    No RNG, world metadata, measured rewards or held-out tuning chooses the pool.
    """
    _integer(max_trials, 1, 4096, 'max_trials')
    _integer(max_candidates, 1, 257, 'max_candidates')
    if not isinstance(action_pool, list) or len(action_pool) > 4096:
        raise ValueError('bounded explicit action pool required')
    baseline = trace_edits(dawn, baseline_actions)
    edits = {V.plan_key(a): a for a in baseline['actions'] if a != 0}
    proposals = []
    for action in action_pool:
        if action is None:
            for key in list(edits) or [None]:
                changed = dict(edits); changed.pop(key, None)
                proposals.append(('DELETE', key, [changed[k] for k in sorted(changed)] + [0]))
        else:
            _integer(action, 1, V.N_ACTIONS - 1, 'pool action ID')
            key = V.plan_key(action)
            changed = dict(edits); changed[key] = action
            proposals.append(('REPLACE' if key in edits else 'INSERT', key,
                              [changed[k] for k in sorted(changed)] + [0]))
    candidates, attempts = [baseline], []
    seen = {_successor_key(baseline): baseline['content_sha256']}
    for kind, key, actions in proposals:
        if len(attempts) >= max_trials or len(candidates) >= max_candidates:
            break
        row = {'kind': kind, 'key': key, 'actions': actions}
        try:
            trace = trace_edits(dawn, actions)
        except ValueError as error:
            row.update(status='REJECTED', error=str(error))
        else:
            identity = _successor_key(trace)
            if identity in seen:
                row.update(status='DUPLICATE', duplicate_of=seen[identity])
            else:
                seen[identity] = trace['content_sha256']
                candidates.append(trace)
                row.update(status='ADMITTED', candidate_sha256=trace['content_sha256'])
        attempts.append(row)
    return bound({'schema': SCHEMA + '-neighborhood', 'dawn_sha256': dawn['content_sha256'],
                  'candidates': candidates, 'proposals': attempts, 'trials': len(attempts),
                  'bounded_pool_size': len(proposals), 'max_trials': max_trials, 'max_candidates': max_candidates,
                  'pool_exhausted': len(attempts) == len(proposals), 'scope': 'EXPLICIT_ONE_KEY_NEIGHBORHOOD'})


def select_best(menu: dict, outcomes: list[dict]) -> dict:
    """Deterministic reducer. Arithmetic fixtures do not become native evidence."""
    equal(menu, bound(menu), 'neighborhood binding')
    candidates = menu['candidates']
    expected = [c['content_sha256'] for c in candidates]
    if len(outcomes) != len(expected) or [o.get('candidate_sha256') for o in outcomes] != expected:
        raise ValueError('exact ordered outcome per admitted candidate required')
    kinds = {o.get('evidence_kind') for o in outcomes}
    if len(kinds) != 1 or not kinds <= {'NATIVE_GAME', 'SYNTHETIC_DIAGNOSTIC'}:
        raise ValueError('one explicit nonmixed evidence kind required')
    for outcome in outcomes:
        if outcome.get('dawn_sha256') != menu['dawn_sha256']:
            raise ValueError('outcome dawn context mismatch')
        for field in ('own_cash', 'rival_cash'):
            value = outcome.get(field)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('finite own and rival cash required')
    index = max(range(len(outcomes)), key=lambda i: outcomes[i]['own_cash'])
    base, best = outcomes[0], outcomes[index]
    own_delta = best['own_cash'] - base['own_cash']
    rival_delta = best['rival_cash'] - base['rival_cash']
    return bound({'schema': SCHEMA + '-selection', 'selected_index': index,
                  'selected': candidates[index], 'baseline': candidates[0], 'outcomes': copy.deepcopy(outcomes),
                  'delta_own_cash': own_delta, 'delta_rival_cash': rival_delta,
                  'delta_margin': own_delta - rival_delta, 'evidence_kind': next(iter(kinds)),
                  'objective': 'OWN_TERMINAL_CASH', 'tie_rule': 'BASELINE_THEN_POOL_ORDER',
                  'scope': 'BEST_MEASURED_LEGAL_CORRECTION_FOUND_NOT_GLOBAL_OPTIMUM'})


def _policy(spec: dict):
    if spec == {'kind': 'keep'}:
        return KeepPolicy()
    if isinstance(spec, dict) and set(spec) == {'kind', 'target'} and spec['kind'] == 'target':
        return TargetPolicy(_plan(spec['target']))
    raise ValueError('closed KeepPolicy or TargetPolicy specification required')


def smoke_request(*, policy='keep') -> dict:
    """One later dawn with NONEMPTY allocation, executor and kept HOLD memory."""
    wheat = V.MODULE_BY_KEY['WHEAT:watered:b00']
    future = V.MODULE_BY_KEY['WHEAT:watered:b01']
    hold = (V.PRIMITIVES.index('HOLD'), V.HORIZONS.index('NEXT_DAWN'), 0)
    opening = Plan({wheat: 1}, market={V.PRODUCTS.index('WHEAT'): hold})
    target = opening.copy(); target.cohorts[future] = 1
    return {'schema': SCHEMA + '-request', 'world': 'router-dairy', 'seat': 0, 'rival': 'router',
            'program': {'dawn': 24, 'initial_plan': opening.canonical(), 'prior_actions': {},
                        'policy': {'kind': 'keep'} if policy == 'keep' else {'kind': 'target', 'target': target.canonical()},
                        'suffix_policy': {'kind': 'keep'}},
            'action_pool': [V.encode('PROD', future, V.DELTAS.index('PLUS_ONE'))] if policy == 'keep' else [None],
            'max_trials': 1, 'max_candidates': 2}


def validate_request(request: dict) -> dict:
    fields = {'schema', 'world', 'seat', 'rival', 'program', 'action_pool', 'max_trials', 'max_candidates'}
    if not isinstance(request, dict) or set(request) != fields or request['schema'] != SCHEMA + '-request':
        raise ValueError('closed oracle request required')
    if request['world'] not in {w.id for w in panel.world_registry().worlds} or request['rival'] not in RIVALS:
        raise ValueError('registered world and frozen rival required')
    _integer(request['seat'], 0, 1, 'seat')
    _integer(request['max_trials'], 1, 4096, 'max_trials')
    _integer(request['max_candidates'], 1, 257, 'max_candidates')
    if not isinstance(request['action_pool'], list) or len(request['action_pool']) > 4096:
        raise ValueError('bounded explicit action pool required')
    for action in request['action_pool']:
        if action is not None: _integer(action, 1, V.N_ACTIONS - 1, 'pool action')
    program = request['program']
    if not isinstance(program, dict) or set(program) != {'dawn', 'initial_plan', 'prior_actions', 'policy', 'suffix_policy'}:
        raise ValueError('closed actor program required; no hidden world dispatch')
    step = _integer(program['dawn'], 0, V.LAST_STEP, 'dawn')
    if step % 24: raise ValueError('dawn boundary required')
    _plan(program['initial_plan']); _policy(program['policy']); _policy(program['suffix_policy'])
    if not isinstance(program['prior_actions'], dict): raise ValueError('prior schedule mapping required')
    for key, actions in program['prior_actions'].items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ValueError('canonical decimal prior dawn required')
        when = int(key)
        if when >= step or when % 24: raise ValueError('prior actions must precede correction dawn')
        _sequence(actions)
    canonical(request)
    return copy.deepcopy(request)


def source_pins() -> dict:
    # Reuse the existing game's full transitive dependency closure, not a second
    # hand-maintained engine/rival list. Add this consumer and public-feature code.
    from farm_neat.training.teachers.v0_plan_search import source_pins as parent_pins
    pins = parent_pins()
    for name in ('src/farm_neat/training/teachers/v0_oracle.py', 'src/farm_neat/training/v0_targets.py',
                 'src/farm_neat/evaluation/v0_continuation.py'):
        pins[name] = file_sha(ROOT / name)
    return pins


class DawnActor(V0Agent):
    """Only chooses a policy at the boundary; super().epoch owns all economics."""
    def __init__(self, program: dict, decision: list[int] | None = None, expected_dawn: str | None = None):
        self.program = copy.deepcopy(program)
        self.decision = None if decision is None else _sequence(decision)
        self.expected_dawn = expected_dawn
        self.dawn_capture = self.dawn_trace = self.after_sha256 = None
        super().__init__(KeepPolicy(), plan=_plan(program['initial_plan']), labor_factor=4.,
                         hands_mode='daily', belief=None, game_id='v0-oracle-actor')
        self.target_policy = _policy(program['policy'])
        self.suffix_policy = _policy(program['suffix_policy'])
        self.identity = {'kind': SCHEMA + '-actor', 'program': self.program, 'decision': self.decision,
                         'labor_factor': 4., 'hands_mode': 'daily', 'belief': 'ZERO_FLOW', 'assets': source_pins()}

    def epoch(self, observation):
        step = observation['step']
        if step < self.program['dawn']:
            actions = self.program['prior_actions'].get(str(step), [0])
        elif step == self.program['dawn']:
            self.dawn_capture = capture_dawn(self, observation)
            if self.expected_dawn is not None:
                equal(self.dawn_capture['content_sha256'], self.expected_dawn, 'complete pre-edit dawn memory')
            masks = Masks(State(observation), self.plan, **self.projection_kw())
            actions = (list(self.target_policy(State(observation), self.plan, masks))
                       if self.decision is None else self.decision)
        else:
            masks = Masks(State(observation), self.plan, **self.projection_kw())
            actions = list(self.suffix_policy(State(observation), self.plan, masks))
        # All decoded schedules, including TargetPolicy stubs, must be executable
        # on their own actual prefix; no teacher-plan injection or silent masking.
        dawn = self.dawn_capture if step == self.program['dawn'] else capture_dawn(self, observation)
        trace = trace_edits(dawn, actions)
        self.policy = lambda state, plan, masks: list(actions)
        super().epoch(observation)
        equal(self.plan.canonical(), trace['plan'], 'executed Plan versus legal prefix trace')
        equal(_holds(self), trace['hold_releases'], 'executed HOLD bindings')
        if step == self.program['dawn']:
            self.dawn_trace = trace
            self.after_sha256 = digest(snapshot_agent(self))


def _native_play(task: dict, timeout: float) -> dict:
    """Runs only in the per-attempt child. World information never enters actor."""
    equal(source_pins(), task['source_pins'], 'source closure')
    if panel.play_game is not panel._NATIVE_PLAY_GAME:
        raise ValueError('substituted native runner refused')
    request = validate_request(task['request'])
    actors = []
    def factory():
        actor = DawnActor(request['program'], task['decision'], task['expected_dawn'])
        actors.append(actor)
        return actor
    spec = panel.register_actor('v0-dawn-oracle', factory)
    row = panel.play_game(panel.world_registry()[request['world']], spec, panel.load_actor(request['rival']),
                          request['seat'], timeout)
    if row['status'] != 'COMPLETE' or row['physical_games_completed'] != 1 or row['steps'] != 719:
        return bound({'schema': SCHEMA + '-attempt', 'game': row, 'status': 'FAILED', 'pid': os.getpid()})
    actor = actors[-1]
    if actor.dawn_capture is None or actor.dawn_trace is None:
        raise ValueError('correction dawn was not executed')
    return bound({'schema': SCHEMA + '-attempt', 'status': 'COMPLETE', 'game': row,
                  'dawn': actor.dawn_capture, 'trace': actor.dawn_trace, 'after_sha256': actor.after_sha256,
                  'continuation_sha256': digest(snapshot_agent(actor)), 'epoch_records_sha256': digest(actor.records),
                  'epoch_records': actor.records, 'pid': os.getpid(), 'source_pins': source_pins(),
                  'cpu_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime})


def replay_fingerprint(receipt: dict) -> str:
    game = receipt['game']
    return digest({'game': {k: game[k] for k in ('cash', 'margin', 'wdl', 'steps', 'actions_sha256', 'actions_by_seat_sha256')},
                   'dawn_sha256': receipt['dawn']['content_sha256'], 'trace_sha256': receipt['trace']['content_sha256'],
                   'after_sha256': receipt['after_sha256'], 'continuation_sha256': receipt['continuation_sha256'],
                   'epoch_records_sha256': receipt['epoch_records_sha256']})


def _journal(path: Path, value: dict):
    with path.open('a') as stream:
        stream.write(canonical(bound(value)) + '\n'); stream.flush(); os.fsync(stream.fileno())


def _fresh_attempt(output: Path, index: int, role: str, task: dict, timeout: float) -> dict:
    request_path, result_path = output / f'attempt-{index:03d}.request.json', output / f'attempt-{index:03d}.json'
    write_new(request_path, task)
    log = output / f'attempt-{index:03d}.log'
    _journal(output / 'attempts.jsonl', {'event': 'STARTED', 'index': index, 'role': role,
                                       'request_sha256': file_sha(request_path), 'native_attempts': 1})
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    command = [sys.executable, '-B', '-m', MODULE, '_play', '--request', str(request_path),
               '--output', str(result_path), '--seconds', str(timeout)]
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
            _journal(output / 'attempts.jsonl', {'event': 'FAILED', 'index': index, 'role': role,
                                               'error': 'hard process timeout; counted; no automatic retry'})
            raise TimeoutError(f'counted native attempt {index} exceeded deadline')
    if code != 0 or not result_path.is_file():
        _journal(output / 'attempts.jsonl', {'event': 'FAILED', 'index': index, 'role': role, 'exit_code': code})
        raise RuntimeError(f'counted native attempt {index} failed; inspect {log}')
    result = read(result_path)
    _journal(output / 'attempts.jsonl', {'event': result['status'], 'index': index, 'role': role,
                                       'receipt_sha256': file_sha(result_path), 'pid': result['pid']})
    if result['status'] != 'COMPLETE':
        raise RuntimeError(f'counted native attempt {index} failed: {result["game"].get("error")}')
    if result['pid'] == os.getpid() or result['pid'] != child.pid:
        raise ValueError('independent fresh process required')
    equal(result['source_pins'], task['source_pins'], 'measured source closure')
    if result['game']['evidence_kind'] != 'NATIVE_GAME':
        raise ValueError('native runner evidence required')
    return result


def _outcome(receipt: dict) -> dict:
    game = receipt['game']; seat = game['seat']
    return {'candidate_sha256': receipt['trace']['content_sha256'], 'dawn_sha256': receipt['dawn']['content_sha256'],
            'own_cash': game['cash'][seat], 'rival_cash': game['cash'][1 - seat], 'margin': game['margin'], 'wdl': game['wdl'],
            'actions_sha256': game['actions_sha256'], 'evidence_kind': 'NATIVE_GAME',
            'receipt_sha256': receipt['content_sha256']}


def run_native(request: dict, output: Path, *, max_games=4, seconds=300.) -> dict:
    """One bounded correction case. At most candidates+2 full-game attempts.

    This exclusive case directory is also the budget boundary. Existing output
    is refused, including failed/torn runs. There is no automatic resume/retry.
    """
    request = validate_request(request)
    _integer(max_games, 1, 100000, 'native budget')
    required = request['max_candidates'] + 2
    if max_games < required:
        raise ValueError(f'native budget must reserve {required} attempts including BOTH fresh replays')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 86400:
        raise ValueError('finite bounded seconds required')
    output = Path(output).absolute()
    if output != output.resolve() or not output.is_relative_to(ROOT / 'work/runs') or output.exists():
        raise ValueError('new non-symlink output under work/runs required')
    pins = source_pins()
    output.mkdir(parents=True)
    manifest = write_new(output / 'MANIFEST.json', {'schema': SCHEMA + '-run', 'request': request, 'source_pins': pins,
                'max_games': max_games, 'worst_case_attempts': required, 'seconds_budget': seconds,
                'scope': panel.SCOPE, 'retention': {'purpose': 'One-dawn correction and replay evidence',
                'dependencies': list(pins), 'remove_when': 'No correction-label intake or oracle replay depends on this case'}})
    started = time.monotonic(); receipts = []
    def attempt(role, decision, expected):
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 0: raise TimeoutError('case deadline exhausted; no new native attempt')
        if len(receipts) >= max_games: raise ValueError('native budget exhausted')
        task = {'schema': SCHEMA + '-play', 'request': request, 'decision': decision,
                'expected_dawn': expected, 'source_pins': pins}
        value = _fresh_attempt(output, len(receipts), role, task, min(300., remaining))
        receipts.append(value)
        return value
    baseline = attempt('BASELINE', None, None)
    dawn = baseline['dawn']
    menu = neighborhood(dawn, baseline['trace']['actions'], request['action_pool'],
                        max_trials=request['max_trials'], max_candidates=request['max_candidates'])
    write_new(output / 'neighborhood.json', menu)
    measurements = [baseline]
    for trace in menu['candidates'][1:]:
        row = attempt('CANDIDATE', trace['actions'], dawn['content_sha256'])
        equal(row['trace'], trace, 'candidate fresh execution trace')
        measurements.append(row)
    selection = select_best(menu, [_outcome(r) for r in measurements])
    baseline_replay = attempt('BASELINE_REPLAY', selection['baseline']['actions'], dawn['content_sha256'])
    selected_replay = attempt('SELECTED_REPLAY', selection['selected']['actions'], dawn['content_sha256'])
    equal(replay_fingerprint(baseline_replay), replay_fingerprint(baseline), 'fresh baseline replay')
    equal(replay_fingerprint(selected_replay), replay_fingerprint(measurements[selection['selected_index']]), 'fresh selected replay')
    replay_delta = _outcome(selected_replay)['own_cash'] - _outcome(baseline_replay)['own_cash']
    equal(replay_delta, selection['delta_own_cash'], 'fresh paired rest-of-game delta')
    label = write_new(output / 'correction-label.json', {
        'schema': SCHEMA + '-conditional-label', 'actor': actor_input(dawn),
        'target': {'actions': selection['selected']['actions'], 'plan': selection['selected']['plan'],
                   'prefix_trace': selection['selected']['steps']},
        'trainer_only': {'request': request, 'dawn': dawn, 'selection': selection,
                         'manifest_sha256': manifest['content_sha256'], 'source_pins': pins,
                         'replay_fingerprints': [replay_fingerprint(baseline_replay), replay_fingerprint(selected_replay)],
                         'admission': 'CONDITIONAL_ONE_DAWN_REACHABILITY_ONLY; NOT_OPENING_EXPORTER_ADMITTED',
                         'initial_plan_origin': 'EXPLICIT_PRELOAD; OPENING_REACHABILITY_NOT_CERTIFIED'}})
    result = write_new(output / 'RESULT.json', {'schema': SCHEMA + '-result', 'status': 'COMPLETE', 'selection': selection,
        'neighborhood_sha256': menu['content_sha256'], 'label_sha256': label['content_sha256'],
        'fresh_replay_verified': True, 'native_attempts': len(receipts), 'native_games_completed': len(receipts),
        'independent_comparisons': len(measurements), 'distinct_worlds': 1, 'dawn_step': request['program']['dawn'],
        'prefix_turns_per_attempt': request['program']['dawn'], 'rest_of_game_turns_per_attempt': 719 - request['program']['dawn'],
        'replays_are_not_additional_worlds': True, 'seconds': time.monotonic() - started,
        'child_cpu_seconds': sum(r['cpu_seconds'] for r in receipts), 'source_pins': pins,
        'manifest_sha256': manifest['content_sha256'], 'scope': panel.SCOPE})
    return result


def run_panel(output: Path, *, max_games=4, seconds=86400.) -> dict:
    """Fixed local correction checks, NOT another broad plan-search campaign.

    Both stubs, all registered worlds, both seats and every frozen rival are
    specified before results. No selection/retuning across cases or promotion.
    """
    requests = []
    for world in panel.world_registry().worlds:
        for seat in (0, 1):
            for rival in RIVALS:
                for policy in ('keep', 'target'):
                    request = smoke_request(policy=policy)
                    request.update(world=world.id, seat=seat, rival=rival)
                    requests.append(request)
    required = sum(r['max_candidates'] + 2 for r in requests)
    _integer(max_games, 1, 100000, 'panel budget')
    if max_games < required:
        raise ValueError(f'full fixed panel budget needs {required} attempts; no games started')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 86400:
        raise ValueError('finite bounded panel seconds required')
    output = Path(output).absolute()
    if output != output.resolve() or not output.is_relative_to(ROOT / 'work/runs') or output.exists():
        raise ValueError('new output under work/runs required')
    output.mkdir(parents=True)
    write_new(output / 'MANIFEST.json', {'schema': SCHEMA + '-panel', 'cases': requests, 'source_pins': source_pins(),
                                      'max_games': max_games, 'worst_case_attempts': required, 'seconds_budget': seconds})
    started = time.monotonic(); rows, used = [], 0
    for index, request in enumerate(requests):
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 0: raise TimeoutError('panel budget exhausted; retain completed cases; no automatic resume')
        result = run_native(request, output / f'case-{index:04d}', max_games=request['max_candidates'] + 2,
                            seconds=min(300., remaining))
        used += result['native_attempts']
        rows.append({'case': index, 'world': request['world'], 'seat': request['seat'], 'rival': request['rival'],
                     'policy': request['program']['policy']['kind'], 'native_attempts': result['native_attempts'],
                     'delta_own_cash': result['selection']['delta_own_cash'], 'delta_margin': result['selection']['delta_margin'],
                     'result_sha256': result['content_sha256']})
        _journal(output / 'completed-cases.jsonl', rows[-1])
    return write_new(output / 'RESULT.json', {'schema': SCHEMA + '-panel-result', 'cases': rows,
                                             'native_attempts': used, 'scope': panel.SCOPE})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('smoke', 'run', 'panel', '_play'))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-games', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=300.)
    args = parser.parse_args(argv)
    if args.command == '_play':
        if args.request is None or not math.isfinite(args.seconds) or not 0 < args.seconds <= 300:
            parser.error('bounded internal play request required')
        resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(args.seconds), math.ceil(args.seconds) + 1))
        task = read(args.request)
        result = _native_play(task, args.seconds)
        write_new(args.output, result)
        return 0 if result['status'] == 'COMPLETE' else 1
    if args.command == 'panel':
        result = run_panel(args.output, max_games=args.max_games, seconds=args.seconds)
    else:
        if args.command == 'run' and args.request is None: parser.error('--request required')
        request = smoke_request() if args.command == 'smoke' else read(args.request, binding=False)
        result = run_native(request, args.output, max_games=args.max_games, seconds=args.seconds)
    print(canonical({k: result[k] for k in ('status', 'native_attempts', 'fresh_replay_verified', 'child_cpu_seconds') if k in result}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
