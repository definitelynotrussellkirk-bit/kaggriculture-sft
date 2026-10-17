"""Standard-game reached V0 corpus producer; NOT a bypass of opening-only admission.

    PYTHONPATH=src python3 -B -m farm_neat.training.v0_corpus_worlds collect --output NEW_DIR
    PYTHONPATH=src python3 -B -m farm_neat.training.v0_corpus_worlds replay --dataset DIR --manifest-sha256 SHA --output NEW_JSON

Owns: a predeclared three-world train/dev/test collection, source-bound own
continuations, actor-only later-dawn rows, and zero-native-step replay. Reuses the
unchanged V0Agent, compiler, native engine, scorer menus, and corpus auditors.
Does NOT own: auditor/BC admission changes, policy strength, optimization, teacher
approval, opponent private state, or counterfactual outcomes.

The current accepted v0_corpus audit and v0_bc reader reject later dawns / own
memory. This producer therefore writes candidate-prefixes.jsonl, NEVER the native
EXPORTED schema or a passing CORPUS.json. An actual audit failure is retained, and
training_admitted stays false. No monkeypatch, relabelled dawn, synthetic cash,
zeroed memory, or changed loss weights can turn that failure into admission.

Runtime/observation wire order is preserved; hashes of compiler/controller state
are order-sensitive. Allocation, started instances and HOLD bindings live in the
source continuation; the scorer currently consumes only declared HOLD memory.
Raw observations, source identities, policies, seeds and outcomes are trainer-only.
The built-in teacher is a legal, public-state scripted coverage policy, NOT a
strong teacher. Refused proposals survive; observed terminal cash is not edit value.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import gzip
import hashlib
import json
import re
from pathlib import Path
import time
from typing import Any, Iterable

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.epochs import V0Agent
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import Masks, PHI_NAMES, State
from farm_neat.evaluation import v0_policy as E
from farm_neat.game.exact_engine import DEFAULT_CONFIGURATION, ExactKaggricultureEnvironment, PASS_ACTION
from farm_neat.policy import v0_menu_fast as F, v0_scorer as A
from farm_neat.training import v0_corpus as C, v0_targets as T

SCHEMA = 'farm-v0-reached-candidate-corpus-v1'
PLAN_SCHEMA = 'farm-v0-standard-world-collection-plan-v1'
ROW_SCHEMA = 'farm-v0-reached-candidate-prefix-v1'
CAPTURE_STEPS = tuple(24 * d for d in (1, 3, 6, 7, 12, 18, 21, 22, 23, 29))
BLOCKED = 'BLOCKED_ACCEPTED_AUDITOR_OPENING_ONLY'
KNOWN_AUDIT_ERRORS = ('opening corpus cannot invent continuation memory',
                      'only opening actor rows are admitted by this exporter')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _write(path: Path, value: Any) -> None:
    # Do not use the sorted-key exporter writer for runtime snapshots.
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(E.wire(value) + '\n')


def _read(path: Path) -> Any:
    return C._read(path)


def pins() -> dict[str, str]:
    return {**C.source_pins(), **E.source_pins(),
            'src/farm_neat/training/v0_corpus_worlds.py': T._file_sha(Path(__file__))}


def world_id(spec: dict) -> str:
    """Native initialization identity, independent of seat/teacher/run name.

    Unlike the opening auditor's conservative *setup* key, this names a complete
    seeded physical world, not a distinct opening actor context. Seed never enters
    an actor row. Copies, seats and alternative policies retain this group.
    """
    require(type(spec.get('seed')) is int, 'integer native seed required')
    require(spec.get('configuration') == DEFAULT_CONFIGURATION, 'exact standard configuration required')
    return T._digest({'seed': spec['seed'], 'configuration': spec['configuration']})


def default_plan() -> dict:
    groups = [('train', 1700007601, 0, ['WHEAT', 'COW']),
              ('dev', 1700007602, 1, ['CARROT', 'SHEEP']),
              ('test', 1700007603, 0, ['TOMATO', 'STRAWBERRY', 'MELON', 'GOOSE'])]
    return {'schema': PLAN_SCHEMA, 'split_id': 'v0-standard-reached-three-world-v1',
            'max_games': 3, 'cpu_seconds': 1800, 'capture_steps': list(CAPTURE_STEPS),
            'worlds': [{'name': split + '-world', 'split': split, 'seed': seed, 'seat': seat,
                        'families': families, 'configuration': copy.deepcopy(DEFAULT_CONFIGURATION)}
                       for split, seed, seat, families in groups],
            'opponent': 'PASS', 'selection': 'PREDECLARED; NO PAYOFF OR MODEL SELECTION',
            'sources': pins()}


def validate_plan(plan: dict) -> None:
    require(plan.get('schema') == PLAN_SCHEMA, 'collection plan schema required')
    require(type(plan.get('max_games')) is int and 1 <= plan['max_games'] <= 4, 'native game budget must be 1..4')
    require(type(plan.get('cpu_seconds')) is int and 1 <= plan['cpu_seconds'] <= 1800, 'CPU budget must be 1..1800 seconds')
    worlds = plan.get('worlds')
    require(isinstance(worlds, list) and 0 < len(worlds) <= plan['max_games'], 'world count exceeds game budget')
    require(plan.get('opponent') == 'PASS', 'this bounded producer owns only a fixed PASS rival')
    require(plan.get('sources') == pins(), 'collection source pins changed')
    require(isinstance(plan.get('split_id'), str) and bool(plan['split_id']), 'split identity required')
    steps = plan.get('capture_steps')
    require(isinstance(steps, list) and bool(steps) and
            all(type(s) is int and s in CAPTURE_STEPS for s in steps) and steps == sorted(set(steps)),
            'capture steps must be predeclared later-dawn boundaries')
    seen, names = set(), set()
    for world in worlds:
        wid = world_id(world)
        require(wid not in seen, 'physical world repeated: duplicate or split leakage')
        seen.add(wid)
        require(type(world.get('seat')) is int and world['seat'] in (0, 1), 'native seat required')
        require(world.get('split') in ('train', 'dev', 'test'), 'train/dev/test split required')
        name = world.get('name')
        require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name) is not None,
                'safe world name required')
        require(name not in names, 'duplicate world name')
        names.add(name)
        ReachedTeacher(world['families'])  # validate authored policy before initializing anything
    require({w['split'] for w in worlds} == {'train', 'dev', 'test'}, 'all three dataset splits required')


@dataclass
class Budget:
    max_games: int
    cpu_seconds: int
    started_at: float = 0.0
    games: int = 0

    def __post_init__(self):
        self.started_at = time.process_time()

    def check(self):
        if time.process_time() - self.started_at > self.cpu_seconds:
            raise TimeoutError('cooperative collection CPU budget exhausted; no corpus admission')

    def start_game(self):
        self.check()
        require(self.games < self.max_games, 'native game budget exhausted')
        self.games += 1


def _module(family: str, birth: int) -> int | None:
    return next((m['id'] for m in V.MODULES if m['producer'] == family and m['birth'] == birth
                 and m['policy'] in ('watered', 'fed_cared')), None)


def _market(product: str, primitive: str, horizon: str) -> int:
    return V.encode('MARKET', V.PRODUCTS.index(product), V.PRIMITIVES.index(primitive),
                    V.HORIZONS.index(horizon), 0)


class ReachedTeacher:
    """Small bounded public-state coverage policy, max four proposed edits/dawn.

    No native game access, seed, private rival state, future shop schedule, payoff
    search or target repair. A template is a proposal, not an admitted Plan. Each
    rejected action and its exact prefix are retained in last_decision.
    """
    def __init__(self, families: Iterable[str]):
        families = list(families)
        allowed = {m['producer'] for m in V.MODULES}
        require(0 < len(families) <= 4 and len(families) == len(set(families)) and
                set(families) <= allowed, 'one to four distinct native producer families required')
        self.identity = {'kind': 'REACHED_COVERAGE_V1', 'families': families}
        self.last_decision = None

    def snapshot_state(self):
        return {'last_decision': copy.deepcopy(self.last_decision)}

    def restore_state(self, value):
        require(isinstance(value, dict) and set(value) == {'last_decision'}, 'teacher continuation fields')
        self.last_decision = copy.deepcopy(value['last_decision'])

    def proposals(self, day: int) -> list[int]:
        families = self.identity['families']
        target = families[:1] if day in (0, 3, 18) else families[1:] if day == 1 else []
        actions = [V.encode('PROD', m, V.DELTAS.index('PLUS_ONE'))
                   for f in target if (m := _module(f, day)) is not None]
        first_product = V.MODULES[_module(families[0], 0)]['product']
        animal_product = next((m['product'] for m in V.MODULES
                               if m['producer'] in families and m['animal']), first_product)
        if day == 6:
            actions.append(_market(first_product, 'HOLD', 'NEXT_REVEAL'))
        if day == 12:
            actions.append(_market(first_product, 'HOLD', 'TERMINAL'))
        if day == 21:
            actions.append(_market(animal_product, 'HOLD', 'NEXT_DAWN'))
        if day == 23:
            actions.append(_market(animal_product, 'CLEAR', 'NOW'))
        if day == 29:
            actions.append(_market(first_product, 'CLEAR', 'NOW'))
        return sorted(actions, key=V.plan_key)

    def __call__(self, state, plan, masks):
        prefix, actions, attempts = plan.copy(), [], []
        for action in self.proposals(state.day):
            support = Masks(state, prefix, **masks.kw)
            selected = support.legal(action)
            attempts.append({'action': action, 'before_plan': prefix.canonical(), 'selected': selected,
                             'reason': None if selected else 'MASKED_AT_CURRENT_PREFIX'})
            if selected:
                prefix = prefix.apply(action, support.resolver)
                actions.append(action)
        actions.append(V.STOP_ID)
        self.last_decision = {'step': state.step, 'actions': actions, 'proposals': attempts,
                              'intended_plan': prefix.canonical()}
        return list(actions)


def memory_wire(holds: dict) -> dict:
    A._hold_memory({'hold_releases': holds})
    return {'hold_releases': [[p, *binding, step] for p, (binding, step) in sorted(holds.items())]}


def memory_from_wire(value: dict) -> dict:
    require(isinstance(value, dict) and set(value) == {'hold_releases'}, 'closed own memory fields required')
    rows = value['hold_releases']
    require(isinstance(rows, list) and all(isinstance(r, list) and len(r) == 4 for r in rows), 'HOLD memory rows required')
    require(all(type(x) is int for r in rows for x in r), 'integer HOLD memory required')
    holds = {p: ((primitive, horizon), step) for p, primitive, horizon, step in rows}
    require(len(holds) == len(rows), 'duplicate HOLD product')
    return {'hold_releases': A._hold_memory({'hold_releases': holds})}


def actor_state(public: dict) -> State:
    """Actor-only later-dawn adapter; no trainer observation used for scoring.

    Same explicit facts as T.actor_observation, but reconstructs the real dawn
    from remaining_days. This is NOT installed into the accepted BC reader.
    """
    require(public.get('schema') == T.ACTOR_SCHEMA, 'actor schema required')
    remaining = public.get('remaining_days')
    require(type(remaining) is int and 1 <= remaining <= V.SEASON_DAYS, 'integer remaining days required')
    s = State.__new__(State)
    s.player, s.day, s.hour = 0, V.SEASON_DAYS - remaining, 0
    s.step = 24 * s.day
    try:
        s.cash = public['cash']
        s.quadrants, s.hands_today = public['owned_quadrants'], public['hands_today']
        s.hires_today, s.occupied = public['hires_today'], public['occupied_sites']
        s.stock, s.seeds = copy.deepcopy(public['stock']), copy.deepcopy(public['seeds'])
        s.inventory, s.params = copy.deepcopy(public['market_inventory']), copy.deepcopy(public['market_params'])
        s.shops = copy.deepcopy(public['unlocked_shops'])
        require(T.actor_observation(s) == public, 'actor fields/calendar round-trip mismatch')
        A.actor_tokens(s, A.Prefix(Plan()), memory=None)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('invalid actor fields') from exc
    return s


def scorer_inputs(row: dict) -> dict:
    """Only actor/candidates -> the existing scorer. No trainer provenance crosses.

    A consumer may pass these explicitly to V0Scorer.score(..., memory=memory).
    This diagnostic adapter does not authorize a BC update on a blocked corpus.
    """
    actor = row['actor']
    require(set(actor) == {'public_observation', 'own_memory', 'prefix_plan', 'last_key', 'nonstop_edit_count'},
            'closed actor fields required')
    return {'state': actor_state(actor['public_observation']),
            'prefix': A.Prefix(Plan.from_canonical(actor['prefix_plan']), actor['last_key'], actor['nonstop_edit_count']),
            'memory': memory_from_wire(actor['own_memory']),
            'legal_ids': row['candidates']['legal_ids'], 'phi': row['candidates']['phi']}


def class_payload(c: A.SuccessorClass) -> dict:
    p = c.successor
    return {'indices': list(c.indices), 'successor_plan_sha256': p.plan.digest(),
            'continuation': T.continuation_signature(p.plan, p.last_key, p.edit_count, p.terminated)}


def prefix_rows(request: dict, *, backend: str = 'fast') -> list[dict]:
    """Rebuild native menus at each actual prefix, preserve exact targets and STOP.

    Prefix reconstruction alone is NOT native reachability/admission. collect and
    replay bind the request to an ordered native observation/action stream and to
    an exact own-agent continuation. Failures never substitute a different target.
    """
    state = State(request['observation'])
    initial = Plan.from_canonical(request['initial_plan'])
    memory = memory_from_wire(request['own_memory'])
    decision = E.validate_edits(state, initial, request['actions'], projection_kwargs={**T.SETTINGS, **memory})
    require(decision.plan.canonical() == request['intended_plan'], 'exact target mismatch; no target repair')
    identity = {f: request[f] for f in T.IDENTITY_FIELDS} | {'learner_seat': state.player, 'dawn': state.day}
    rows = []
    prefix = A.Prefix(initial)
    nontrivial = any(a != V.STOP_ID for a in decision.actions)
    for index, action in enumerate(decision.actions):
        menu = F.build_menu(state, prefix, backend=backend, projection_kwargs={**T.SETTINGS, **memory})
        require(action in menu.legal_ids, 'trace disagrees with actual prefix support')
        selected = next(c for c in menu.classes if menu.legal_ids.index(action) in c.indices)
        payload = class_payload(selected)
        forced = action == V.STOP_ID and prefix.edit_count == V.MAX_EDITS_PER_EPOCH
        row = {'schema': ROW_SCHEMA, 'identity': identity | {'prefix_index': index},
               'actor': {'public_observation': T.actor_observation(state), 'own_memory': copy.deepcopy(request['own_memory']),
                         'prefix_plan': prefix.plan.canonical(), 'last_key': prefix.last_key,
                         'nonstop_edit_count': prefix.edit_count},
               'candidates': {'legal_ids': list(menu.legal_ids), 'phi': [list(v) for v in menu.phi],
                              'classes': [class_payload(c) for c in menu.classes], 'forced_stop': forced,
                              'phi_names': list(PHI_NAMES)},
               'target': {'class_indices': payload['indices'], 'successor_plan_sha256': payload['successor_plan_sha256'],
                          'continuation': payload['continuation'],
                          'stop_kind': ('forced_cap' if forced else 'sampled') if action == 0 else 'not_stop',
                          'loss_weight': int(nontrivial and not forced)},
               'trainer_only': {'replay_action': action, 'source_request_sha256': E.wire_hash(request)}}
        # Exact actor-token and menu reconstruction from only the exported facts.
        decoded = scorer_inputs(row)
        require(A.actor_tokens(state, prefix, memory=memory) ==
                A.actor_tokens(decoded['state'], decoded['prefix'], memory=decoded['memory']), 'actor token mismatch')
        C._label(row)  # unchanged accepted class/continuation/weight check
        rows.append(row)
        prefix = selected.successor
    require(prefix.terminated and prefix.plan == decision.plan, 'exact final successor mismatch')
    return rows


def accepted_audit(rows: list[dict]) -> dict:
    """Invoke, never replace or monkeypatch, the accepted auditor.

    audit_rows skips unweighted controls. Probe context support too so an all-KEEP
    dataset cannot accidentally look supported. Both failures remain literal.
    """
    require(bool(rows), 'nonempty corpus required')
    snapshot = E.wire_hash(rows)
    error, audit = None, None
    try:
        audit = C.audit_rows(rows)
        for row in rows:
            C._context_keys(row)
    except ValueError as exc:
        error = str(exc)
        if error not in KNOWN_AUDIT_ERRORS:
            raise
    from farm_neat.training.v0_bc import actor_state as accepted_actor_state
    probe = {'status': 'SUPPORTED'}
    try:
        for row in rows:
            accepted_actor_state(row['actor']['public_observation'])
    except ValueError as exc:
        if str(exc) not in KNOWN_AUDIT_ERRORS:
            raise
        probe = {'status': 'BLOCKED', 'error': str(exc)}
    require(E.wire_hash(rows) == snapshot, 'auditor mutated corpus rows')
    blocked = error is not None or probe['status'] == 'BLOCKED'
    return {'status': BLOCKED if blocked else 'AUDITED', 'training_admitted': False,
            'error': error, 'calendar_probe': probe, 'accepted_audit': audit, 'rows_modified': 0,
            'auditor_source_sha256': T._file_sha(Path(C.__file__)),
            'conflict_counts': None if blocked else {k: audit[k]['conflicting_contexts'] for k in ('exporter', 'scorer_fp32')},
            'note': 'No conflict-free claim or native BC dataset until the accepted admission consumer supports this state.'}


def _teacher_from_snapshot(snapshot: dict) -> ReachedTeacher:
    require(snapshot['policy']['kind'] == 'EXTERNAL', 'coverage teacher continuation required')
    identity = snapshot['policy']['identity']
    require(identity.get('kind') == 'REACHED_COVERAGE_V1', 'unknown coverage teacher')
    return ReachedTeacher(identity['families'])


def restore_request_agent(request: dict) -> V0Agent:
    snapshot = request['continuation']
    agent = E.restore_agent(snapshot, request['observation'], policy=_teacher_from_snapshot(snapshot))
    require(agent.plan.canonical() == request['initial_plan'], 'request initial Plan/continuation mismatch')
    require(memory_wire(agent.hold_releases) == request['own_memory'], 'request HOLD memory/continuation mismatch')
    return agent


def replay_request(request: dict) -> dict:
    """One real continuation -> one exact native publication/action; zero engine steps."""
    agent = restore_request_agent(request)
    action = agent(copy.deepcopy(request['observation']))
    require(agent.policy.last_decision['actions'] == request['actions'], 'own-prefix teacher replay mismatch')
    require(agent.plan.canonical() == request['intended_plan'], 'replayed target mismatch')
    require(E.wire_hash(action) == request['verification']['action_sha256'], 'replayed executor action mismatch')
    require(E.agent_signature(agent) == request['verification']['post_agent_sha256'], 'replayed continuation mismatch')
    require(agent.controller.last_publication_sha256 == request['verification']['publication_sha256'], 'compiled publication mismatch')
    return {'trajectory_id': request['trajectory_id'], 'status': 'EXACT_OWN_CONTINUATION_REPLAYED',
            'prefixes': len(request['actions']), 'engine_steps': 0,
            'allocation_entries': len(request['continuation']['allocation']),
            'started_entries': len(request['continuation']['started']),
            'hold_bindings': len(request['continuation']['hold_releases']),
            **request['verification']}


def _split_plan(plan):
    return {'schema': C.PLAN_SCHEMA, 'split_id': plan['split_id'],
            'source_groups': {'world/' + world_id(w): w['split'] for w in plan['worlds']},
            'reserved_worlds': []}


def check_request_groups(requests: list[dict], plan: dict) -> None:
    """Accepted ancestry/leakage rules plus independently bound native world IDs."""
    C.check_groups(requests, _split_plan(plan))
    C.check_sources(requests)
    by_id = {world_id(w): w for w in plan['worlds']}
    for r in requests:
        world = by_id.get(r['physical_world_id'])
        require(world is not None, 'request world not in frozen plan')
        wid = world_id(world)
        require(r['source_group'] == 'world/' + wid and r['seed_group'] == 'seed/' + str(world['seed'])
                and r['split'] == world['split'] and r['provenance']['world_spec'] == world,
                'native world/source/seed/split binding mismatch')
        require(State(r['observation']).player == world['seat'], 'native seat binding mismatch')


def _coverage(requests: list[dict]) -> dict:
    # Preserve accepted family/dawn/plan-class/split definitions. Mark mechanical
    # reachability separately from training admission; these aren't admitted rows.
    records = [{'status': 'ADMITTED', 'reason': None,
                'identity': {**{f: r[f] for f in T.IDENTITY_FIELDS}, 'dawn': State(r['observation']).day},
                'reached_plan': r['intended_plan'], 'nonstop_edits': len(r['actions']) - 1}
               for r in requests]
    raw = C._coverage(requests, records)
    def rename(value):
        if isinstance(value, dict):
            return {('reached_replayed' if k == 'admitted' else 'reached_counts' if k == 'admitted_counts' else k): rename(v)
                    for k, v in value.items()}
        return value
    return {'scope': 'EXACT_REACHED_REPLAY_ONLY; NOT_TRAINING_ADMISSION', **rename(raw)}


def collect(plan: dict, output: str | Path) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    validate_plan(plan)
    budget, wall = Budget(plan['max_games'], plan['cpu_seconds']), time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'PLAN.json', plan)  # before any game or held-out observation
    all_requests, worlds = [], []
    try:
        for world in plan['worlds']:
            budget.start_game()
            wid, seat = world_id(world), world['seat']
            teacher = ReachedTeacher(world['families'])
            agent = V0Agent(teacher, labor_factor=4., hands_mode='daily', belief=copy.deepcopy(T.SETTINGS['belief']),
                            game_id='world/' + wid)
            env = ExactKaggricultureEnvironment(seed=world['seed'], configuration=world['configuration'])
            require({k: v for k, v in env.configuration.items() if k != 'seed'} == DEFAULT_CONFIGURATION,
                    'engine configuration not standard')
            tape_name = world['name'] + '-observations.jsonl.gz'
            requests, action_hash, obs_hash = [], hashlib.sha256(), hashlib.sha256()
            with (output / tape_name).open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as tape:
                while not env.done:
                    budget.check()
                    observation = env.agent_observation(seat)
                    step = env.step_index
                    before = E.snapshot_agent(agent, observation) if step in plan['capture_steps'] else None
                    joint = [copy.deepcopy(PASS_ACTION), copy.deepcopy(PASS_ACTION)]
                    joint[seat] = agent(copy.deepcopy(observation))  # configuration/hidden seed is NOT passed to actor
                    observation_bytes = (E.wire(observation) + '\n').encode()
                    action_bytes = (E.wire(joint) + '\n').encode()
                    obs_hash.update(observation_bytes); action_hash.update(action_bytes)
                    tape.write((E.wire({'step': step, 'observation': observation, 'joint_action': joint}) + '\n').encode())
                    if before is not None:
                        decision = teacher.last_decision
                        request = {'teacher_kind': 'explicit_native_v0', 'teacher_id': 'coverage/' + '+'.join(world['families']),
                                   'trajectory_id': wid + f'/seat{seat}/dawn{step // 24:02d}', 'physical_world_id': wid,
                                   'source_group': 'world/' + wid, 'seed_group': 'seed/' + str(world['seed']),
                                   'split': world['split'], 'observation': copy.deepcopy(observation),
                                   'initial_plan': before['plan'], 'intended_plan': agent.plan.canonical(),
                                   'actions': list(decision['actions']), 'continuation': before,
                                   'own_memory': memory_wire(memory_from_wire({'hold_releases': before['hold_releases']})['hold_releases']),
                                   'proposals': copy.deepcopy(decision['proposals']),
                                   'provenance': {'kind': 'NATIVE_REACHED_STANDARD_WORLD', 'world_spec': copy.deepcopy(world),
                                                  'trajectory_group': wid + '/seat' + str(seat),
                                                  'source_context_id': wid + '/step' + str(step),
                                                  'source_bindings': [{'path': p, 'sha256': h} for p, h in sorted(plan['sources'].items())],
                                                  'outcome_kind': 'OBSERVED_TRAJECTORY_ONLY', 'edit_outcome': None,
                                                  'economic_strength_admitted': False},
                                   'verification': {'native_observation_sha256': E.wire_hash(observation),
                                                    'native_prefix_actions_sha256': action_hash.hexdigest(),
                                                    'native_prefix_observations_sha256': obs_hash.hexdigest(),
                                                    'action_sha256': E.wire_hash(joint[seat]),
                                                    'post_agent_sha256': E.agent_signature(agent),
                                                    'publication_sha256': agent.controller.last_publication_sha256}}
                        replay_request(request)
                        requests.append(request)
                    env.step(joint)
            require(env.done and env.step_index == V.LAST_STEP + 1, 'complete standard game required')
            terminal_obs = env.agent_observation(seat)
            terminal = {'observation_sha256': E.wire_hash(terminal_obs), 'agent_sha256': E.agent_signature(agent),
                        'cash_by_seat': [float(f['money']) for f in terminal_obs['farms']]}
            for r in requests:
                r['provenance']['observed_terminal_cash_by_seat'] = terminal['cash_by_seat']
            receipt = {'physical_world_id': wid, 'spec': world, 'steps': env.step_index, 'terminal': terminal,
                       'joint_actions_sha256': action_hash.hexdigest(), 'observations_sha256': obs_hash.hexdigest(),
                       'tape': {'path': tape_name, 'sha256': T._file_sha(output / tape_name)},
                       'captured_dawns': len(requests), 'policy_decision_count': len(agent.records),
                       'epoch_refusals': sum(r['rejected'] for r in agent.records),
                       'epoch_unplaced': sum(r['unplaced'] for r in agent.records)}
            worlds.append(receipt)
            all_requests.extend(requests)
            _write(output / (world['name'] + '-RECEIPT.json'), receipt)
        check_request_groups(all_requests, plan)
        _write(output / 'REQUESTS.json', {'schema': SCHEMA, 'requests': all_requests})
        _write(output / 'WORLDS.json', worlds)
        rows, replays = [], []
        with (output / 'candidate-prefixes.jsonl').open('x') as stream:
            for request in all_requests:
                budget.check()
                replays.append(replay_request(request))
                sequence = prefix_rows(request)
                rows.extend(sequence)
                for row in sequence:
                    stream.write(E.wire(row) + '\n')
        audit = accepted_audit(rows)
        _write(output / 'ACCEPTED-AUDIT.json', audit)
        _write(output / 'REPLAY.json', {'schema': SCHEMA, 'requests': replays, 'engine_steps': 0})
        _write(output / 'COVERAGE.json', _coverage(all_requests))
        report = {'schema': SCHEMA, 'status': 'REACHED_CANDIDATES_REPLAYED_ADMISSION_BLOCKED',
                  'training_admitted': False, 'accepted_auditor_status': audit['status'], 'sources': pins(),
                  'plan_sha256': T._file_sha(output / 'PLAN.json'),
                  'counts': {'physical_worlds': len(worlds), 'native_games_started': budget.games,
                             'native_games_completed': len(worlds), 'native_joint_steps': sum(w['steps'] for w in worlds),
                             'trajectories': len(all_requests), 'nontrivial': sum(len(r['actions']) > 1 for r in all_requests),
                             'prefixes': len(rows), 'weighted_prefixes': sum(r['target']['loss_weight'] for r in rows),
                             'training_admitted_trajectories': 0, 'model_calls': 0, 'optimizer_updates': 0},
                  'limits': ['Auditor/BC opening-only gate is unchanged; zero conflicts is NOT established.',
                             'Three worlds are exposed, predeclared dataset splits, not a held-out strength panel.',
                             'Coverage teacher is not a strong teacher; no paired helpful/harmful edit outcomes.',
                             'Allocation/started/controller memory retained for exact replay, not encoded as new scorer features.'],
                  'cpu_seconds': time.process_time() - budget.started_at, 'wall_seconds': time.perf_counter() - wall,
                  'files': {p.name: {'sha256': T._file_sha(p), 'bytes': p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}}
        _write(output / 'MANIFEST.json', report)  # last: interrupted/failing runs are not datasets
        return report
    except Exception as exc:
        _write(output / 'FAILURE.json', {'schema': SCHEMA, 'error_type': type(exc).__name__, 'error': str(exc),
                                        'native_games_started': budget.games, 'native_games_completed': len(worlds),
                                        'training_admitted': False, 'cpu_seconds': time.process_time() - budget.started_at})
        raise


def _documents(directory: Path, expected_sha256: str):
    require(T._file_sha(directory / 'MANIFEST.json') == expected_sha256, 'manifest hash mismatch')
    manifest = _read(directory / 'MANIFEST.json')
    require(manifest['schema'] == SCHEMA and manifest['sources'] == pins(), 'reached corpus schema/source mismatch')
    require(manifest['training_admitted'] is False, 'candidate corpus cannot self-admit')
    required = {'PLAN.json', 'REQUESTS.json', 'WORLDS.json', 'candidate-prefixes.jsonl',
                'ACCEPTED-AUDIT.json', 'REPLAY.json', 'COVERAGE.json'}
    require(required <= set(manifest['files']), 'all consumed dataset members must be hash-bound')
    for name, binding in manifest['files'].items():
        path = directory / name
        require(path.parent == directory and path.is_file() and not path.is_symlink(), 'unsafe dataset member')
        require(path.stat().st_size == binding['bytes'] and T._file_sha(path) == binding['sha256'], 'dataset member hash mismatch: ' + name)
    require(manifest['plan_sha256'] == manifest['files']['PLAN.json']['sha256'], 'plan hash binding mismatch')
    return manifest, _read(directory / 'PLAN.json'), _read(directory / 'REQUESTS.json')['requests']


def replay_dataset(directory: str | Path, expected_sha256: str, *, backend='fast') -> dict:
    """Fresh-process corpus replay, no native initialization or env.step.

    Each full own action stream is regenerated on the retained native observations.
    Each selected dawn is also independently restored BEFORE observe/publication.
    The stream is exogenous replay evidence, not an alternative native trajectory.
    """
    directory = Path(directory)
    manifest, plan, requests = _documents(directory, expected_sha256)
    validate_plan(plan); check_request_groups(requests, plan)
    by_world = {world_id(w): w for w in plan['worlds']}
    request_by_cut = {(r['physical_world_id'], r['observation']['step']): r for r in requests}
    require(len(request_by_cut) == len(requests), 'duplicate native cut')
    replayed, steps = [], 0
    receipts = _read(directory / 'WORLDS.json')
    require(len(receipts) == len(by_world) and {r['physical_world_id'] for r in receipts} == set(by_world),
            'all predeclared worlds must have one receipt')
    require(len(requests) == len(by_world) * len(plan['capture_steps']), 'all predeclared dawns required')
    for receipt in receipts:
        tape = receipt['tape']
        require(tape['path'] in manifest['files'] and
                manifest['files'][tape['path']]['sha256'] == tape['sha256'], 'tape not bound by manifest')
        wid, world = receipt['physical_world_id'], receipt['spec']
        require(by_world.get(wid) == world, 'world receipt/plan mismatch')
        agent = V0Agent(ReachedTeacher(world['families']), labor_factor=4., hands_mode='daily',
                        belief=copy.deepcopy(T.SETTINGS['belief']), game_id='world/' + wid)
        action_hash, obs_hash = hashlib.sha256(), hashlib.sha256()
        count = 0
        with gzip.open(directory / receipt['tape']['path'], 'rt') as tape:
            from farm_neat.training.v0_bc import _json_loads
            for line in tape:
                row = _json_loads(line)
                require(row['step'] == count == row['observation']['step'], 'noncontiguous native observation stream')
                request = request_by_cut.get((wid, count))
                if request is not None:
                    require(E.snapshot_agent(agent, row['observation']) == request['continuation'], 'native ancestry/continuation mismatch')
                    require(E.wire_hash(row['observation']) == E.wire_hash(request['observation']), 'native cut observation mismatch')
                action = agent(copy.deepcopy(row['observation']))
                require(E.wire_hash(action) == E.wire_hash(row['joint_action'][world['seat']]), 'uninterrupted own action replay mismatch')
                require(row['joint_action'][1 - world['seat']] == PASS_ACTION, 'rival is not frozen PASS')
                action_hash.update((E.wire(row['joint_action']) + '\n').encode())
                obs_hash.update((E.wire(row['observation']) + '\n').encode())
                if request is not None:
                    require(action_hash.hexdigest() == request['verification']['native_prefix_actions_sha256'] and
                            obs_hash.hexdigest() == request['verification']['native_prefix_observations_sha256'], 'cut prefix stream hash mismatch')
                    replayed.append(replay_request(request))
                count += 1
        require(count == receipt['steps'] == V.LAST_STEP + 1, 'full own replay length mismatch')
        require(action_hash.hexdigest() == receipt['joint_actions_sha256'] and
                obs_hash.hexdigest() == receipt['observations_sha256'], 'full replay stream hash mismatch')
        require(E.agent_signature(agent) == receipt['terminal']['agent_sha256'], 'terminal controller/Plan replay mismatch')
        steps += count
    from farm_neat.training.v0_bc import _json_loads
    rows = [_json_loads(s) for s in (directory / 'candidate-prefixes.jsonl').read_text().splitlines()]
    expected = []
    for r in requests:
        expected.extend(prefix_rows(r, backend=backend))
    require(rows == expected, 'full prefix support/phi/class/target replay mismatch')
    require(len(replayed) == len(requests), 'not every reached cut was replayed')
    audit = accepted_audit(rows)
    require(audit == _read(directory / 'ACCEPTED-AUDIT.json'), 'accepted audit replay mismatch')
    return {'schema': SCHEMA, 'status': 'EXACT_REPLAY_PASSED_ADMISSION_STILL_BLOCKED', 'training_admitted': False,
            'manifest_sha256': expected_sha256, 'sources': pins(), 'native_games_started': 0, 'native_engine_steps': 0,
            'own_actions_replayed': steps, 'trajectories_replayed': len(replayed), 'prefix_rows_rebuilt': len(rows),
            'accepted_auditor_status': audit['status'], 'backend': backend}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('collect'); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p = sub.add_parser('replay'); p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=('fast', 'reference'), default='fast')
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    if args.command == 'collect':
        result = collect(default_plan() if args.plan is None else _read(args.plan), args.output)
        print(E.wire({k: result[k] for k in ('status', 'counts', 'cpu_seconds', 'wall_seconds')}))
        print('MANIFEST_SHA256=' + T._file_sha(args.output / 'MANIFEST.json'))
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        result = replay_dataset(args.dataset, args.manifest_sha256, backend=args.backend)
        _write(args.output, result)
        print(E.wire({k: v for k, v in result.items() if k != 'sources'}))
    # A successful diagnostic collection/replay is not a passing admission gate.
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
