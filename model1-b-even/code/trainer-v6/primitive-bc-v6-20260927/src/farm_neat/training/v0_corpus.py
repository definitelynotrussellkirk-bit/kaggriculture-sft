"""Conflict-gated V0 corpora, composed over the UNCHANGED accepted exporter.

    PYTHONPATH=src python3 -B -m farm_neat.training.v0_corpus fixture32 --output NEW_DIR
    PYTHONPATH=src python3 -B -m farm_neat.training.v0_corpus control --output NEW_DIR
    PYTHONPATH=src python3 -B -m farm_neat.training.v0_corpus export --input REQUESTS.json --plan PLAN.json --output NEW_DIR
    PYTHONPATH=src python3 -B -m farm_neat.training.v0_corpus audit --dataset DIR --manifest-sha256 SHA --output NEW_JSON
    PYTHONPATH=src python3 -B -m farm_neat.training.v0_corpus replay --dataset DIR --manifest-sha256 SHA --output NEW_JSON

Owns: source/lineage checks, whole-trajectory conflict diagnostics, a predeclared
32-case configured-opening CPU fixture, and source-bound replay/coverage receipts.
Does NOT own: action IDs, masks, compiler economics, automatic teacher approval,
training, free-running inference, or later-dawn continuation admission.

fixture32 varies the native startingMoney INITIAL CONDITION, not engine source or
actor features. It is deliberately nonstandard, disposable overfit plumbing: 31
of its 32 initial budgets differ from the competition default. It is NOT a
32-world standard-game corpus. No step/run is called, no economic payoff is known,
and no claim of useful teacher decisions or family transfer is made. The 256-case
standard-world/later-dawn expansion remains open. Eight dev/test configurations
are reserved before export and are NOT initialized, scored, or used to tune labels.

The accepted MANIFEST.json and three JSONLs are retained byte-for-byte in their
native format for v0_bc.load_dataset. CORPUS.json is a separate last-written gate;
its absence or nonpassing status means this is not a conflict-free corpus. Never
infer the corpus gate from the underlying exporter's EXPORTED status alone.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.compiler import compile_opening, unplaced
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import Masks, State
from farm_neat.game.exact_engine import DEFAULT_CONFIGURATION, ExactKaggricultureEnvironment
from farm_neat.training import v0_targets as T

SCHEMA = 'farm-v0-conflict-gated-corpus-v1'
PLAN_SCHEMA = 'farm-v0-corpus-split-plan-v1'
FIXTURE_SPLIT = 'v0-configured-opening-conflict-free-32-v1'
ROOT = T.ROOT


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(T._json(value) + '\n')


def _read(path: Path) -> Any:
    # Use the learner's duplicate-key / nonfinite rejection, not a looser reader.
    from farm_neat.training.v0_bc import _json_read
    return _json_read(path)


def source_pins() -> dict[str, str]:
    from farm_neat.policy import v0_scorer as A
    return {**T.source_identity()['sources'], **A.source_pins(),
            'src/farm_neat/training/v0_corpus.py': T._file_sha(Path(__file__)),
            'src/farm_neat/training/v0_bc.py': T._file_sha(ROOT / 'src/farm_neat/training/v0_bc.py')}


def native_opening_key(spec: dict) -> str:
    """Conservative OPENING setup identity; hidden seeds cannot split equal setups.

    Not a claim that seeded complete games have identical weed/shop realizations.
    IDs/seats/seeds do not make fresh opening actor facts. All supplied configuration
    fields (including initial endowment) remain part of this setup identity.
    """
    _require(set(spec) == {'seed', 'configuration'}, 'complete native initialization required')
    _require(type(spec['seed']) is int and isinstance(spec['configuration'], dict), 'invalid native initialization')
    config = {**DEFAULT_CONFIGURATION, **spec['configuration']}
    config.pop('seed', None)
    return T._digest({'kind': 'native-opening-setup', 'configuration': config})


def _scenario_key(spec: dict) -> str:
    from farm_neat.evaluation.scenarios import validate
    spec = validate(spec)
    value = {'shops': spec['shops'], 'random_weeds': spec['random_weeds']}
    if spec['random_weeds']:
        value['seed'] = spec['seed']
    return T._digest(value)


def _group_tokens(request: dict) -> list[tuple[str, str]]:
    tokens = [(f, request[f]) for f in ('physical_world_id', 'source_group', 'seed_group')]
    provenance = request.get('provenance', {})
    parents = provenance.get('parent_source_groups', [])
    _require(isinstance(parents, list) and all(isinstance(p, str) and p for p in parents), 'invalid ancestry')
    tokens += [('source_group', p) for p in parents]
    for field in ('source_context_id', 'trajectory_group'):
        if field in provenance:
            _require(isinstance(provenance[field], str) and bool(provenance[field]), 'invalid ancestry')
            tokens.append((field, provenance[field]))
    if provenance.get('transformation'):
        _require(bool(parents) or bool(provenance.get('source_context_id')), 'transformed context requires ancestry')
    if 'scenario' in provenance:
        tokens.append(('scenario_schedule', _scenario_key(provenance['scenario'])))
    if 'native_initialization' in provenance:
        tokens.append(('native_opening_setup', native_opening_key(provenance['native_initialization'])))
    return tokens


def check_groups(requests: list[dict], plan: dict | None) -> None:
    """Shared ancestry tokens enforce connected-component splits, including copies.

    This is stronger than trusting renamed physical_world_id strings. It cannot
    detect an undeclared source transformation; source verification remains needed.
    """
    T._check_groups(requests, plan['split_id'] if plan else 'unplanned-audit')
    seen: dict[tuple[str, str], str] = {}
    for request in requests:
        for field in T.IDENTITY_FIELDS:
            _require(isinstance(request.get(field), str) and bool(request[field]), f'nonempty {field} required')
        for token in _group_tokens(request):
            _require(token not in seen or seen[token] == request['split'], f'split leakage in {token[0]}: {token[1]}')
            seen[token] = request['split']
    if plan is None:
        return
    _require(plan.get('schema') == PLAN_SCHEMA, 'corpus split plan schema required')
    reserved = plan.get('reserved_worlds', [])
    reserved_ids = {r['physical_world_id'] for r in reserved}
    reserved_tokens = set()
    for world in reserved:
        if 'native_initialization' in world:
            reserved_tokens.add(('native_opening_setup', native_opening_key(world['native_initialization'])))
        if 'scenario' in world:
            reserved_tokens.add(('scenario_schedule', _scenario_key(world['scenario'])))
    for request in requests:
        _require(request['physical_world_id'] not in reserved_ids and
                 not (set(_group_tokens(request)) & reserved_tokens), 'reserved world cannot enter export')
        _require(plan['source_groups'].get(request['source_group']) == request['split'], 'split disagrees with frozen plan')


def check_sources(requests: list[dict]) -> None:
    checked = set()
    for request in requests:
        bindings = request.get('provenance', {}).get('source_bindings', [])
        _require(isinstance(bindings, list), 'source bindings must be a list')
        for binding in bindings:
            _require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}, 'source binding fields required')
            rel, sha = binding['path'], binding['sha256']
            _require(isinstance(rel, str) and isinstance(sha, str), 'invalid source binding')
            path = (ROOT / rel).resolve()
            _require(not Path(rel).is_absolute() and path.is_relative_to(ROOT.resolve()), 'source must stay inside repository')
            if (rel, sha) not in checked:
                _require(path.is_file() and T._file_sha(path) == sha, f'source hash mismatch: {rel}')
                checked.add((rel, sha))


def verify_native_origin(request: dict) -> dict:
    """Reinitialize claimed native fixture observations; never trust edited cash.

    For general imported requests with no initialization spec, report ATTESTED,
    not independently reached. Later-dawn origin needs separate trajectory evidence.
    """
    provenance = request.get('provenance', {})
    spec = provenance.get('native_initialization')
    if spec is None:
        return {'status': 'CALLER_ATTESTED_ONLY', 'game_steps': 0}
    native_opening_key(spec)
    env = ExactKaggricultureEnvironment(seed=spec['seed'], configuration=spec['configuration'])
    seat = State(request['observation']).player
    _require(seat in (0, 1), 'invalid initialization seat')
    actual = env.agent_observation(seat)
    _require(actual == request['observation'], 'native initialization observation mismatch')
    _require(env.step_index == 0, 'initialization unexpectedly advanced a game')
    return {'status': 'NATIVE_INITIALIZATION_REPLAYED', 'observation_sha256': T._digest(actual), 'game_steps': 0}


def replay_request(request: dict, verified: dict) -> dict:
    """Independent prefix loop and fresh compile; no cached menu / teacher injection.

    Replays the requested trace, NOT a neural decoder or a game executor. Both exact
    Plan and continuation cursor/cap/STOP must match, not merely compiled behavior.
    """
    state = State(copy.deepcopy(request['observation']))
    current = Plan.from_canonical(request['initial_plan'])
    _require(verified['status'] == 'ADMITTED' and state.step == 0 and current == Plan(), 'replay requires admitted opening')
    _require(len(request['actions']) == len(verified['prefixes']) > 0, 'replay length mismatch')
    last, used, stopped = -1, 0, False
    for action, expected in zip(request['actions'], verified['prefixes']):
        _require(not stopped and type(action) is int and 0 <= action < V.N_ACTIONS, 'replay invalid action/STOP')
        _require(action == 0 or (used < V.MAX_EDITS_PER_EPOCH and V.plan_key(action) > last), 'replay cursor/cap mismatch')
        masks = Masks(state, current, **T.SETTINGS)
        _require(masks.legal(action), 'replay illegal prefix action')
        before = {'plan': current.canonical(), 'last_key': last, 'edit_count': used}
        current = current.apply(action, masks.resolver)
        if action:
            last, used = V.plan_key(action), used + 1
        else:
            stopped = True
        continuation = T.continuation_signature(current, last, used, stopped)
        _require(expected == {**before, 'action': action, 'successor_plan_sha256': current.digest(),
                              'continuation': continuation}, 'replay prefix mismatch')
    _require(stopped and current.canonical() == request['intended_plan'] == verified['reached_plan'], 'replay target mismatch')
    program, projection, allocation = compile_opening(current, copy.deepcopy(request['observation']), hands_mode='daily', **T.SETTINGS)
    _require(projection.feasible and not unplaced(allocation), 'replay compiler refusal')
    _require(T._digest(program) == verified['verification']['program_sha256'], 'replay compiler hash mismatch')
    return {'status': 'REPLAYED', 'trajectory_id': request['trajectory_id'], 'game_steps': 0,
            'prefixes_checked': len(request['actions']), 'reached_plan_sha256': current.digest(),
            'program_sha256': T._digest(program), 'source_request_sha256': T._digest(request)}


def _context_keys(row: dict) -> dict[str, str]:
    """No label, identity, source, hidden seed, or raw representative enters a key.

    The second key includes the actual CPU FP32 nonlinear preprocessing of actor
    tokens AND candidate phi. Distinct raw floats can round to identical scorer
    inputs. This certifies only absence of identical-input contradictory labels,
    not that trained weights distinguish all remaining contexts.
    """
    import torch
    from farm_neat.policy import v0_scorer as A
    from farm_neat.training.v0_bc import actor_state
    actor, candidates = row['actor'], row['candidates']
    prefix = A.Prefix(Plan.from_canonical(actor['prefix_plan']), actor['last_key'], actor['nonstop_edit_count'])
    _require(actor['own_memory'] == {}, 'opening corpus cannot invent continuation memory')
    tokens = A.actor_tokens(actor_state(actor['public_observation']), prefix, memory=None)

    def normalized_sha(values):
        values = torch.as_tensor(values, dtype=torch.float32, device='cpu')
        _require(bool(torch.isfinite(values).all()), 'nonfinite scorer inputs')
        scaled = values.sign() * values.abs().log1p() / 10
        return hashlib.sha256(scaled.contiguous().numpy().tobytes()).hexdigest()

    encoded = {'token_ids': [t[0] for t in tokens], 'known': [t[2] for t in tokens],
               'values': normalized_sha([t[1] for t in tokens]),
               'legal_ids': candidates['legal_ids'], 'phi': normalized_sha(candidates['phi'])}
    return {'exporter': T._digest({'actor': actor, 'legal_ids': candidates['legal_ids']}),
            'scorer_fp32': T._digest(encoded)}


def _label(row: dict) -> tuple[int, ...]:
    indices = tuple(row['target']['class_indices'])
    selected = next((c for c in row['candidates']['classes'] if tuple(c['indices']) == indices), None)
    _require(selected is not None, 'target must be one complete successor class')
    _require(all(row['target'][k] == selected[k] for k in ('successor_plan_sha256', 'continuation')), 'target continuation mismatch')
    weight = row['target']['loss_weight']
    _require(type(weight) in (int, float) and weight in (0, 1), 'native 0/1 loss weight required')
    return indices


def audit_rows(rows: Iterable[dict]) -> dict:
    """Audit weighted class targets, not raw representative IDs. Never edits rows.

    Use audit_dataset for hash-verified exports, and the native BC validator to
    recheck source semantics. This low-level audit alone is not data admission.
    """
    buckets = {name: defaultdict(lambda: defaultdict(list)) for name in ('exporter', 'scorer_fp32')}
    total = weighted = 0
    for row in rows:
        total += 1
        label = _label(row)
        if not row['target']['loss_weight']:
            continue
        weighted += 1
        reference = {'trajectory_id': row['identity']['trajectory_id'],
                     'prefix_index': row['identity']['prefix_index'], 'split': row['identity']['split']}
        for name, key in _context_keys(row).items():
            buckets[name][key][label].append(reference)
    result = {'rows': total, 'weighted_rows': weighted, 'unweighted_rows': total - weighted}
    for name, contexts in buckets.items():
        conflicts, floor = [], 0.0
        for key, labels in sorted(contexts.items()):
            if len(labels) < 2:
                continue
            n = sum(map(len, labels.values()))
            contribution = n * math.log(n) - sum(len(v) * math.log(len(v)) for v in labels.values())
            floor += contribution
            conflicts.append({'context_sha256': key, 'minimum_class_nll_nats': contribution,
                              'targets': [{'class_indices': list(k), 'rows': v} for k, v in sorted(labels.items())]})
        result[name] = {'weighted_contexts': len(contexts), 'conflicting_contexts': len(conflicts),
                        'minimum_summed_class_nll_nats': floor, 'conflicts': conflicts}
    return result


def _dataset_documents(directory: Path, expected_sha256: str) -> tuple[dict, dict]:
    _require(T._file_sha(directory / 'MANIFEST.json') == expected_sha256, 'manifest hash mismatch')
    manifest = _read(directory / 'MANIFEST.json')
    _require(manifest['status'] == 'EXPORTED', 'underlying export count gate failed')
    for k, value in T.source_identity().items():
        _require(manifest.get(k) == value, 'dataset source/schema mismatch: ' + k)
    names = {'prefixes.jsonl', 'trajectories.jsonl', 'rejections.jsonl'}
    _require(set(manifest['files']) == names, 'native dataset members required')
    from farm_neat.training.v0_bc import _json_loads
    documents = {}
    for name in sorted(names):
        path, pin = directory / name, manifest['files'][name]
        _require(path.stat().st_size == pin['bytes'] and T._file_sha(path) == pin['sha256'], 'dataset member hash mismatch: ' + name)
        documents[name] = [_json_loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    return manifest, documents


def audit_dataset(directory: str | Path, *, expected_sha256: str) -> dict:
    _, documents = _dataset_documents(Path(directory), expected_sha256)
    return audit_rows(documents['prefixes.jsonl'])


def replay_dataset(directory: str | Path, *, expected_sha256: str) -> dict:
    """Fresh-process-friendly replay of every bound request, including rejections.

    Unlike an input-hash audit, this runs the native reachability and compiler
    checks again. It does not rebuild every candidate phi or run a model/game.
    """
    directory = Path(directory)
    manifest, documents = _dataset_documents(directory, expected_sha256)
    document = _read(directory / 'REQUESTS.json')
    requests = document['requests']
    _require(document['schema'] == T.REQUEST_SCHEMA and document['split_id'] == manifest['split_id'], 'request schema mismatch')
    _require(T._digest(requests) == manifest['requests_sha256'], 'request hash mismatch')
    check_sources(requests)
    check_groups(requests, _read(directory / 'PLAN.json'))
    records = documents['trajectories.jsonl'] + documents['rejections.jsonl']
    by_id = {r['identity']['trajectory_id']: r for r in records}
    _require(len(by_id) == len(records) == len(requests), 'replay record count mismatch')
    replays, origins, rejections = [], [], []
    for request in requests:
        origin = verify_native_origin(request)
        origins.append({'trajectory_id': request['trajectory_id'], **origin})
        verified = T.verify_request(request)
        if verified['status'] == 'ADMITTED':
            verified.pop('examples')
        _require(verified == by_id[request['trajectory_id']], 'native record replay mismatch')
        if verified['status'] == 'ADMITTED':
            replays.append(replay_request(request, verified))
        else:
            rejections.append({'trajectory_id': request['trajectory_id'], 'reason': verified['reason'],
                               'residual': verified['residual']})
    return {'schema': SCHEMA, 'status': 'REPLAYED', 'dataset_manifest_sha256': expected_sha256,
            'records': replays, 'origins': origins, 'rejections': rejections,
            'game_steps': 0, 'sources': source_pins()}


def select_compatible(rows: Iterable[dict]) -> dict:
    """Deterministic WHOLE-trajectory greedy selection, reported but never applied.

    No optimality claim: it is a reproducible baseline, not a maximum independent
    set solver. An explicit caller must construct and re-export the chosen subset.
    An epoch cannot be repaired by deleting just its conflicting first prefix.
    """
    grouped = defaultdict(list)
    for row in rows:
        _label(row)
        if row['target']['loss_weight']:
            grouped[row['identity']['trajectory_id']].append(row)
    assignments, kept, dropped = {}, [], []
    for tid, sequence in sorted(grouped.items()):
        local, collisions = {}, []
        for row in sequence:
            label = tuple(row['target']['class_indices'])
            for kind, key in _context_keys(row).items():
                token = (kind, key)
                old = local.get(token, assignments.get(token))
                if old is not None and old[0] != label:
                    collisions.append({'kind': kind, 'context_sha256': key, 'prefix_index': row['identity']['prefix_index'],
                                       'conflicts_with': old[1]})
                local[token] = (label, tid)
        if collisions:
            dropped.append({'trajectory_id': tid, 'whole_trajectory': True, 'conflicts': collisions})
        else:
            kept.append(tid)
            assignments.update(local)
    return {'method': 'LEXICOGRAPHIC_WHOLE_TRAJECTORY_GREEDY', 'optimality_claim': False,
            'kept_trajectory_ids': kept, 'dropped': dropped, 'applied': False}


def fixture_requests() -> tuple[list[dict], dict]:
    """Declared teacher templates, fresh native initial conditions, no target repair.

    Only actions/targets are reused as policy templates; source observations are
    not copied/rotated. Full native observations are regenerated independently.
    The jump teacher keeps default cash because its intended count is source-bound.
    All other intended Plans are preserved literally and must pass admission.
    """
    templates = T.smoke_requests()
    teachers = [r for r in templates if not r['trajectory_id'].startswith(('reject-', 'stop-'))]
    teachers.sort(key=lambda r: (r['trajectory_id'] != 'prefix-alias-jump', r['trajectory_id']))
    _require(len(teachers) == 32, 'accepted smoke template count changed')
    source = {'path': 'src/farm_neat/training/v0_targets.py',
              'sha256': T._file_sha(ROOT / 'src/farm_neat/training/v0_targets.py')}
    requests, groups, separation = [], {}, []

    def make(template, index, role):
        spec = {'seed': 1_700_003_000 + index, 'configuration': {'startingMoney': 3000 + 100 * index}}
        env = ExactKaggricultureEnvironment(seed=spec['seed'], configuration=spec['configuration'])
        seat = State(template['observation']).player
        obs = env.agent_observation(seat)
        world = native_opening_key(spec)
        request = {k: copy.deepcopy(template[k]) for k in ('teacher_kind', 'actions', 'initial_plan', 'intended_plan')}
        request.update(teacher_id='configured-native/' + template['trajectory_id'],
                       trajectory_id=f'fixture-{index:02d}/' + template['trajectory_id'],
                       physical_world_id=world, source_group='configured-opening/' + world,
                       seed_group='opening-setup/' + world, split='smoke_train', observation=obs,
                       provenance={'kind': 'CONFIGURED_NATIVE_OPENING_DIAGNOSTIC', 'role': role,
                                   'native_initialization': spec, 'template_id': template['trajectory_id'],
                                   'source_bindings': [source], 'observed_outcome': None,
                                   'economic_strength_admitted': False, 'standard_initial_balance': index == 0})
        requests.append(request)
        groups[request['source_group']] = request['split']
        if role == 'teacher':
            separation.append({'template_id': template['trajectory_id'], 'trajectory_id': request['trajectory_id'],
                               'original_actor_sha256': T._digest(T.actor_observation(State(template['observation']))),
                               'configured_actor_sha256': T._digest(T.actor_observation(State(obs))),
                               'native_starting_money': 3000 + 100 * index})

    for i, template in enumerate(teachers):
        make(template, i, 'teacher')
    for template in templates:
        if template['trajectory_id'].startswith(('reject-', 'stop-')):
            make(template, 0, 'negative_control' if template['trajectory_id'].startswith('reject-') else 'stop_control')
    reserved = []
    for i in range(32, 40):
        spec = {'seed': 1_700_003_000 + i, 'configuration': {'startingMoney': 3000 + 100 * i}}
        reserved.append({'physical_world_id': native_opening_key(spec), 'native_initialization': spec,
                         'split': 'dev' if i < 36 else 'test', 'opponent': 'pass',
                         'status': 'RESERVED_NOT_INITIALIZED_OR_EVALUATED'})
    plan = {'schema': PLAN_SCHEMA, 'split_id': FIXTURE_SPLIT, 'source_groups': groups,
            'reserved_worlds': reserved, 'sources': source_pins(),
            'scope': 'CONFIGURED_OPENING_FIXTURE_ONLY; NOT STANDARD-WORLD DEVELOPMENT CORPUS',
            'context_separation': separation, 'template_targets_modified': 0,
            'selection': 'PREDECLARED TEMPLATE ORDER; NO MODEL OR PAYOFF SELECTION'}
    return requests, plan


def _coverage(requests: list[dict], records: list[dict]) -> dict:
    coverage = {axis: {} for axis in ('by_dawn', 'by_economic_family', 'by_plan_class', 'by_split')}
    transfer = defaultdict(Counter)
    for request, record in zip(requests, records):
        plan = Plan.from_canonical(request['intended_plan'])
        families = sorted({V.MODULES[m]['producer'] for m in plan.cohorts})
        plan_class = ('single:' if len(families) == 1 else 'mixed:') + '+'.join(families) if families else 'nonproduction'
        admitted = record['status'] == 'ADMITTED'
        flags = {'requested': 1, 'admitted': int(admitted), 'rejected': int(not admitted),
                 'unreachable_by_supplied_trace': int(record['reached_plan'] != request['intended_plan']),
                 'unsupported': int(record['reason'] in ('OPENING_ONLY', 'NOT_NATIVE_V0')),
                 'replayed': int(admitted)}
        for axis, values in (('by_dawn', [str(record['identity']['dawn'])]),
                             ('by_economic_family', families or ['NONE']), ('by_plan_class', [plan_class]),
                             ('by_split', [request['split']])):
            for value in values:
                coverage[axis].setdefault(value, Counter()).update(flags)
        if admitted and record['nonstop_edits']:
            for family in families:
                transfer[family][request['split']] += 1
    coverage['family_transfer'] = {'admitted_counts': dict(transfer), 'model_transfer_measured': False,
                                    'dev_test_examples': sum(r['status'] == 'ADMITTED' and r['identity']['split'] in ('dev', 'test') for r in records)}
    coverage['unreachable_scope'] = 'Supplied trace failed to reach exact Plan; no global infeasibility claim.'
    return coverage


def export_corpus(requests: Iterable[dict], output: str | Path, *, plan: dict,
                  expected_nontrivial: int = 32) -> dict:
    """Commit native export + separate gate. Failures never cause label substitution.

    The split plan is written before admission/export. Existing paths are refused;
    an interrupted run has no CORPUS.json. Unreached requests and all residuals stay
    in the accepted exporter's rejections.jsonl. No automatic subset is applied.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    _require(type(expected_nontrivial) is int and expected_nontrivial > 0, 'positive exact trajectory count required')
    requests = sorted(copy.deepcopy(list(requests)), key=lambda r: r['trajectory_id'])
    check_groups(requests, plan)
    check_sources(requests)
    _require(plan.get('sources') == source_pins(), 'corpus split-plan source mismatch')
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'PLAN.json', plan)
    _write(output / 'REQUESTS.json', {'schema': T.REQUEST_SCHEMA, 'split_id': plan['split_id'], 'requests': requests})
    records, replays, origins = [], [], []
    for request in requests:
        origin = verify_native_origin(request)
        origins.append({'trajectory_id': request['trajectory_id'], **origin})
        record = T.verify_request(request)
        records.append(record)
        if record['status'] == 'ADMITTED':
            replays.append(replay_request(request, record))
    # Do not rewrite the frozen exporter or its schema. Count is our outer gate;
    # the complete native export remains loadable even for a decisive negative.
    native = output / 'native'
    manifest = T.export_dataset(requests, native, split_id=plan['split_id'])
    for path in native.iterdir():
        path.rename(output / path.name)
    native.rmdir()
    manifest_sha = T._file_sha(output / 'MANIFEST.json')
    audit = audit_dataset(output, expected_sha256=manifest_sha)
    _, documents = _dataset_documents(output, manifest_sha)
    count_ok = manifest['counts']['nontrivial'] == expected_nontrivial
    conflict_ok = all(audit[k]['conflicting_contexts'] == 0 for k in ('exporter', 'scorer_fp32'))
    passed = count_ok and conflict_ok
    replay_doc = {'schema': SCHEMA, 'records': replays, 'origins': origins}
    _write(output / 'REPLAY.json', replay_doc)
    _write(output / 'AUDIT.json', audit)
    selection = select_compatible(documents['prefixes.jsonl'])
    _write(output / 'SELECTION.json', selection)
    teacher_requests = [r for r, v in zip(requests, records) if v['status'] == 'ADMITTED' and v['nonstop_edits']]
    report = {'schema': SCHEMA, 'status': 'CONFLICT_FREE_FIXTURE' if passed else 'CORPUS_GATE_FAILED',
              'expected_nontrivial': expected_nontrivial, 'count_gate': count_ok, 'conflict_gate': conflict_ok,
              'dataset_manifest_sha256': manifest_sha, 'counts': manifest['counts'], 'audit': audit,
              'coverage': _coverage(requests, records), 'rejection_reasons': manifest['rejection_reasons'],
              'replay': {'trajectories': len(replays), 'prefixes': sum(r['prefixes_checked'] for r in replays),
                         'exact_plan_and_program_checks': len(replays), 'native_initializations_verified': sum(o['status'] == 'NATIVE_INITIALIZATION_REPLAYED' for o in origins)},
              'exposure': {'physical_games_started': 0, 'physical_games_completed': 0, 'game_steps': 0,
                           'model_calls': 0, 'optimizer_updates': 0,
                           'distinct_teacher_opening_actor_states': len({T._digest(T.actor_observation(State(r['observation']))) for r in teacher_requests}),
                           'nondefault_balance_teachers': sum(State(r['observation']).cash != DEFAULT_CONFIGURATION['startingMoney'] for r in teacher_requests),
                           'later_dawn_trajectories': 0, 'heldout_worlds_initialized': 0},
              'sources': source_pins(), 'context_separation': plan.get('context_separation', []),
              'suggested_selection_applied': False, 'suggested_whole_trajectory_drops': len(selection['dropped']),
              'files': {name: {'sha256': T._file_sha(output / name), 'bytes': (output / name).stat().st_size}
                        for name in ('PLAN.json', 'REQUESTS.json', 'REPLAY.json', 'AUDIT.json', 'SELECTION.json', 'MANIFEST.json')},
              'limits': ['Zero identical-input conflicts is necessary, not sufficient, for neural 32/32 reconstruction.',
                         'Configured initial balances are fixture inputs, not standard-game exposure or policy improvement.',
                         'The standard-world/later-dawn 256 corpus remains open; do not bulk-generate more openings.',
                         'Sampled STOP stays weighted; cap STOP and whole KEEP controls retain native zero weights.',
                         'No executor realization, helpful/harmful payoff labels, WDL, or family transfer measured.',
                         'Held-out configurations are reserved, not dev/test trajectory coverage.',
                         'Replay is native prefix + compile replay, not own-prefix neural decoding.']}
    _write(output / 'CORPUS.json', report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('fixture32', 'control', 'export'):
        p = sub.add_parser(name)
        p.add_argument('--output', type=Path, required=True)
        if name == 'export':
            p.add_argument('--input', type=Path, required=True)
            p.add_argument('--plan', type=Path, required=True)
            p.add_argument('--expected-nontrivial', type=int, default=32)
    for name in ('audit', 'replay'):
        p = sub.add_parser(name)
        p.add_argument('--dataset', type=Path, required=True)
        p.add_argument('--manifest-sha256', required=True)
        p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    try:
        if args.command == 'replay':
            result = replay_dataset(args.dataset, expected_sha256=args.manifest_sha256)
            _write(args.output, result)
            passed = True
        elif args.command == 'audit':
            result = audit_dataset(args.dataset, expected_sha256=args.manifest_sha256)
            _write(args.output, result)
            passed = all(result[k]['conflicting_contexts'] == 0 for k in ('exporter', 'scorer_fp32'))
        else:
            expected = 32
            if args.command == 'fixture32':
                requests, plan = fixture_requests()
            elif args.command == 'control':
                requests = T.smoke_requests()
                plan = {'schema': PLAN_SCHEMA, 'split_id': 'v0-original-smoke-conflict-control-v1',
                        'source_groups': {r['source_group']: r['split'] for r in requests},
                        'reserved_worlds': [], 'sources': source_pins()}
            else:
                document, plan = _read(args.input), _read(args.plan)
                _require(document['schema'] == T.REQUEST_SCHEMA and document['split_id'] == plan['split_id'], 'bound request/plan schema mismatch')
                requests, expected = document['requests'], args.expected_nontrivial
            report = export_corpus(requests, args.output, plan=plan, expected_nontrivial=expected)
            result = {k: report[k] for k in ('status', 'counts', 'dataset_manifest_sha256', 'exposure', 'replay')}
            result['conflicts'] = {k: report['audit'][k]['conflicting_contexts'] for k in ('exporter', 'scorer_fp32')}
            result['corpus_sha256'] = T._file_sha(args.output / 'CORPUS.json')
            passed = report['status'] == 'CONFLICT_FREE_FIXTURE'
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
