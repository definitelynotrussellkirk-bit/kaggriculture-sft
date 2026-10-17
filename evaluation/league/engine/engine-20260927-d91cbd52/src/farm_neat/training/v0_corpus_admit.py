"""Additive, fail-closed admission of source-bound reached V0 cuts.

    PYTHONPATH=src python3 -B -m farm_neat.training.v0_corpus_admit admit \
        --dataset DIR --manifest-sha256 SHA --output NEW_DIR [--legacy-codec-import]

Owns continuation/prefix validation, conflict diagnostics and a reached-only data
 gate. Does not change the opening auditor/BC reader, train, collect native games,
 or certify teacher strength. ``validate_trajectory`` and ``audit_rows`` are
 low-level checks, NOT proof of native origin; only ``admit_dataset`` also checks
 the manifest, split ancestry and complete retained own-action streams.

Each output prefix carries V0Continuation memory in trainer_only. Scorers receive
 only ``scorer_inputs(row)`` (public facts, canonical prefix, declared HOLD memory,
 legal IDs and phi). Allocation/started/controller/source/teacher identities do
 not enter either conflict key. ``load_rows`` refuses blocked or changed gates.
 ``load_training_dataset`` retains full rows, separating trainable trajectories
 from validated zero-weight controls. ``accumulate_reached_gradients`` computes
 the mean of weighted class-PATH sums with the actual HOLD memory, never a row
 mean. It clears/accumulates gradients, but owns no optimizer, sampler, checkpoint
 or free-running evaluation. The existing opening BC trainer is NOT wired to this
 consumer; its accumulator discards HOLD memory. Do NOT pass these rows to it.

The optional historical import accepts ONLY the retained pre-consolidation
evaluator -> exact evaluator+codec transition, with its exact v1 scorer source
where present. It changes source metadata on an in-memory COPY, never original
evidence or memory fields. Both identities are reported, and all actions,
controller hashes and prefixes must still replay.
 Unrelated source drift, lossy/duplicate memory or replay differences refuse.
 The superseded 6995309125/ac1686ee pair is deliberately NOT a second transition:
 re-collect it with current sources, rather than guessing lost insertion order.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.projection import State
from farm_neat.evaluation import v0_policy as E
from farm_neat.evaluation.v0_continuation import V0Continuation
from farm_neat.policy import v0_scorer as A
from farm_neat.training import v0_corpus as C, v0_corpus_worlds as W, v0_targets as T

SCHEMA = 'farm-v0-reached-admission-v1'
ROW_SCHEMA = 'farm-v0-reached-admitted-prefix-v1'
KEYS = ('exporter', 'scorer_fp32')
REACHED_OBJECTIVE = 'MEAN_TRAINABLE_TRAJECTORIES_SUM_WEIGHTED_PREFIX_CLASS_NLL'
TRAINING_SCHEMA = 'farm-v0-reached-class-path-consumer-v1'
EVALUATOR = 'src/farm_neat/evaluation/v0_policy.py'
CODEC = 'src/farm_neat/evaluation/v0_continuation.py'
SCORER = 'src/farm_neat/policy/v0_scorer.py'
REALIZATION = 'src/farm_neat/evaluation/realization.py'
LEGACY_EVALUATOR_SHA256 = '9d8c46c63c5e7325d77975abb136eac02bb5476c04f4f1ed2efa504577985acb'
CURRENT_EVALUATOR_SHA256 = '715511d62df068863f13b93768c56233754c2caf99b4a951feafaf966ef23d71'
CODEC_SHA256 = '6237540422398a5025d611ef9503515f50140694334c4594a1fd8fb64d0b4c27'
LEGACY_SCORER_SHA256 = 'b4944a3d24fcb8a68cac23a5f12197eb9f6eb2a095decf344b17e9b33f33ce81'
CURRENT_SCORER_SHA256 = '54030ee8444304cee15f66eb34fa38d44bd9965017b1bc6d79dd37391a87f4fc'
LEGACY_REALIZATION_SHA256 = '3ceeac8610ef8835540a116d41f9cb6e3c857afe7ec1f09d3ed9310c80fd233d'
CURRENT_REALIZATION_SHA256 = 'a65d9531880274a9da652fd32bc2f3bf156736769e57f89e5fe2317da526960a'
LEGACY_COPY = ('work/runs/reviewer-play/v0-executor-consolidate/historical/'
               'v0_policy_before_consolidate.py')
LEGACY_SCORER_COPY = ('work/runs/reviewer-play/v0-fit-diagnosis/historical/'
                      'v0_scorer_v1.py')
_require = C._require


def source_pins() -> dict[str, str]:
    return {**W.pins(), 'src/farm_neat/training/v0_corpus_admit.py': T._file_sha(Path(__file__))}


def _same(a: Any, b: Any) -> bool:
    # Unlike Python ==, JSON equality does not identify True with 1 or 4 with 4.0.
    # Runtime snapshot order is separately checked by E.bind/E.wire_hash.
    return T._digest(a) == T._digest(b)


def _source_transition(original: dict, current: dict, allow_legacy_import: bool) -> dict:
    _require(isinstance(original, dict), 'source pins required')
    if original == current:
        return {'mode': 'CURRENT_SOURCES', 'changes': []}
    # The superseded codec sorted live rows. A source rebind cannot recover an
    # order already lost by capture; no paired ancestry proof accompanies these
    # newer envelopes. Keep the one retained import, not a transitive allowlist.
    superseded = dict(current)
    superseded[EVALUATOR] = '6995309125ccf63d74a08e65f998877e1b910c12614e3e735066f6a996969e2d'
    superseded[CODEC] = 'ac1686ee058b8a99d98943958d16fc416ed55c230ff8d3d48dcf888113bba4f6'
    _require(original != superseded,
             'source pins changed: superseded evaluator/codec pair requires re-collection; no second import')
    expected = {k: v for k, v in current.items() if k != CODEC}
    expected[EVALUATOR] = LEGACY_EVALUATOR_SHA256
    expected_v1 = {**expected, SCORER: LEGACY_SCORER_SHA256}
    # The shipped reached cuts also predate the additive custody/cash reporting
    # helpers in realization.py. Those helpers do not participate in V0
    # restoration or menu construction. Admit that one exact extra transition,
    # while retaining the original two transitions for cuts captured after the
    # reporting-only change. Any other realization drift still refuses.
    expected_realization = {**expected, REALIZATION: LEGACY_REALIZATION_SHA256}
    expected_v1_realization = {**expected_v1, REALIZATION: LEGACY_REALIZATION_SHA256}
    mode = ('EXACT_RETAINED_EVALUATOR_IMPORT' if original == expected else
            'EXACT_RETAINED_EVALUATOR_AND_SCORER_IMPORT' if original == expected_v1 else
            'EXACT_RETAINED_EVALUATOR_AND_REALIZATION_IMPORT'
            if original == expected_realization else
            'EXACT_RETAINED_EVALUATOR_SCORER_AND_REALIZATION_IMPORT'
            if original == expected_v1_realization else None)
    _require(allow_legacy_import and mode is not None and
             current.get(EVALUATOR) == CURRENT_EVALUATOR_SHA256 and
             current.get(CODEC) == CODEC_SHA256 and current.get(SCORER) == CURRENT_SCORER_SHA256 and
             current.get(REALIZATION) == CURRENT_REALIZATION_SHA256,
             'source pins changed; only the exact opt-in legacy evaluator/codec/scorer import is supported')
    copies = [(LEGACY_COPY, LEGACY_EVALUATOR_SHA256)]
    if mode in ('EXACT_RETAINED_EVALUATOR_AND_SCORER_IMPORT',
                'EXACT_RETAINED_EVALUATOR_SCORER_AND_REALIZATION_IMPORT'):
        copies.append((LEGACY_SCORER_COPY, LEGACY_SCORER_SHA256))
    for name, sha in copies:
        historical = T.ROOT / name
        _require(historical.is_file() and T._file_sha(historical) == sha,
                 'legacy source preimage unavailable or changed: ' + name)
    return {'mode': mode, 'historical_copy': LEGACY_COPY,
            'historical_copies': [name for name, _ in copies],
            'changes': [{'path': k, 'original_sha256': original.get(k), 'current_sha256': current.get(k)}
                        for k in sorted(set(original) | set(current)) if original.get(k) != current.get(k)]}


def _import_snapshot(snapshot: dict, allow_legacy_import: bool) -> tuple[dict, dict]:
    _require(isinstance(snapshot, dict), 'continuation object required')
    _require(snapshot == E.bind(snapshot), 'continuation hash mismatch')
    transition = _source_transition(snapshot.get('source_pins'), E.source_pins(), allow_legacy_import)
    imported = copy.deepcopy(snapshot)
    imported['source_pins'] = E.source_pins()
    imported = E.bind(imported)
    return imported, transition


def restore_cut(request: dict, *, allow_legacy_import: bool = False):
    """Restore a source-bound pre-observe cut, never infer missing continuation.

    Returns (fresh V0Agent, import receipt). This validates a cut, not its native
    ancestry. The codec is reached through the unchanged evaluator restore path.
    """
    try:
        _require('continuation' in request, 'missing continuation')
        snapshot, transition = _import_snapshot(request['continuation'], allow_legacy_import)
        _require(snapshot['agent_type'] == 'V0Agent', 'reached producer requires V0Agent continuation')
        agent = E.restore_agent(snapshot, request['observation'], policy=W._teacher_from_snapshot(snapshot))
        _require(_same(agent.plan.canonical(), request['initial_plan']), 'initial Plan/continuation mismatch')
        _require(_same(W.memory_wire(agent.hold_releases), request['own_memory']),
                 'request HOLD memory/continuation mismatch')
        _require(agent.labor_factor == T.SETTINGS['labor_factor'] and agent.hands_mode == 'daily'
                 and _same(agent.belief, T.SETTINGS['belief']), 'reached bootstrap settings mismatch')
        _require(request['verification']['native_observation_sha256'] == E.wire_hash(request['observation']),
                 'native observation receipt mismatch')
        memory = V0Continuation.capture(agent).to_dict()
        roundtrip = V0Continuation.from_dict(memory).restore(None, observation=request['observation'])
        _require(E.wire_hash(V0Continuation.capture(roundtrip).to_dict()) == E.wire_hash(memory),
                 'continuation did not round trip exactly')
        return agent, {'source_transition': transition,
                       'original_snapshot_sha256': E.wire_hash(request['continuation']),
                       'imported_snapshot_sha256': E.wire_hash(snapshot),
                       'codec_memory_sha256': E.wire_hash(memory)}
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError('malformed reached continuation: ' + str(exc)) from exc


def _replay_cut(request: dict, agent) -> None:
    action = agent(copy.deepcopy(request['observation']))
    _require(agent.policy.last_decision['actions'] == request['actions'], 'own-prefix teacher replay mismatch')
    _require(_same(agent.plan.canonical(), request['intended_plan']), 'replayed exact target mismatch')
    expected = request['verification']
    _require(E.wire_hash(action) == expected['action_sha256'], 'replayed executor action mismatch')
    _require(E.agent_signature(agent) == expected['post_agent_sha256'], 'replayed continuation signature mismatch')
    _require(agent.controller.last_publication_sha256 == expected['publication_sha256'], 'compiled publication mismatch')


def validate_trajectory(request: dict, rows: Iterable[dict], *, backend: str = 'fast',
                        allow_legacy_import: bool = False) -> tuple[list[dict], dict]:
    """Replay one exact cut/teacher chain and validate every prefix, even controls.

    Returns new codec-carrying rows and a receipt, without changing input rows.
    A passed synthetic/unit cut is still not native-reached data admission.
    """
    rows = list(rows)
    agent, imported = restore_cut(request, allow_legacy_import=allow_legacy_import)
    memory = V0Continuation.capture(agent).to_dict()  # BEFORE publication, not successor memory
    _replay_cut(request, agent)
    expected = W.prefix_rows(request, backend=backend)
    _require(len(rows) == len(expected) > 0, 'missing/extra prefix rows')
    result = []
    for index, (row, native) in enumerate(zip(rows, expected)):
        _require(_same(row, native), f'prefix {index} support/phi/classes/target/identity mismatch')
        new = copy.deepcopy(row)
        new['schema'] = ROW_SCHEMA
        new['trainer_only']['continuation'] = copy.deepcopy(memory)
        # Also check the actual scorer's actor-only menu, not only the full-observation menu.
        inputs = scorer_inputs(new)
        menu = W.F.build_menu(inputs['state'], inputs['prefix'], backend=backend,
                             projection_kwargs={**T.SETTINGS, **inputs['memory']})
        _require(_same(list(menu.legal_ids), row['candidates']['legal_ids']) and
                 _same([list(p) for p in menu.phi], row['candidates']['phi']) and
                 _same([W.class_payload(c) for c in menu.classes], row['candidates']['classes']),
                 f'prefix {index} actor-only scorer menu mismatch')
        result.append(new)
    return result, {'status': 'EXACT_CUT_AND_PREFIXES_VALIDATED', 'trajectory_id': request['trajectory_id'],
                    'source_request_sha256': E.wire_hash(request), 'prefixes_checked': len(result),
                    'engine_steps': 0, 'backend': backend, **imported}


def scorer_inputs(row: dict) -> dict:
    """Closed reached-only adapter. Does not expose codec/controller/provenance."""
    try:
        _require(row['schema'] == ROW_SCHEMA, 'reached codec-carrying row required')
        memory = row['trainer_only']['continuation']
        _require(memory.get('land_commitment') is None,
                 'LAND_ROW_REQUIRES_VERSIONED_ACTOR_OBSERVATION_JOIN; no inferred ownership')
        agent = V0Continuation.from_dict(memory).restore(None)
        E._validate_memory(agent)  # includes completeness of retained HOLD bindings
        inputs = W.scorer_inputs(row)
        state = inputs['state']
        identity = row['identity']
        _require(type(identity['dawn']) is int and identity['dawn'] == state.day,
                 'row dawn/actor calendar mismatch')
        _require(type(identity['learner_seat']) is int and identity['learner_seat'] in (0, 1), 'row seat required')
        E._boundary(agent, {'step': state.step, 'player': identity['learner_seat']})
        _require(_same(W.memory_wire(agent.hold_releases), row['actor']['own_memory']),
                 'row HOLD memory/continuation mismatch')
        _require(agent.labor_factor == T.SETTINGS['labor_factor'] and agent.hands_mode == 'daily'
                 and _same(agent.belief, T.SETTINGS['belief']), 'row bootstrap settings mismatch')
        return inputs
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError('malformed reached row continuation: ' + str(exc)) from exc


def context_keys(row: dict) -> dict[str, str]:
    """Same exporter and CPU FP32 log1p keys as v0_corpus, with real HOLD memory.

    Low-level diagnostic: call validate_trajectory for labels/menus/source binding.
    Restores even zero-weight rows; trainer-only data never enters either key.
    """
    import torch
    inputs = scorer_inputs(row)
    tokens = A.actor_tokens(inputs['state'], inputs['prefix'], memory=inputs['memory'])

    def normalized_sha(values):
        values = torch.as_tensor(values, dtype=torch.float32, device='cpu')
        _require(bool(torch.isfinite(values).all()), 'nonfinite scorer inputs')
        scaled = values.sign() * values.abs().log1p() / 10
        return hashlib.sha256(scaled.contiguous().numpy().tobytes()).hexdigest()

    encoded = {'token_ids': [t[0] for t in tokens], 'known': [t[2] for t in tokens],
               'values': normalized_sha([t[1] for t in tokens]), 'legal_ids': inputs['legal_ids'],
               'phi': normalized_sha(inputs['phi'])}
    return {'exporter': T._digest({'actor': row['actor'], 'legal_ids': inputs['legal_ids']}),
            'scorer_fp32': T._digest(encoded)}


def audit_rows(rows: Iterable[dict]) -> dict:
    """v0_corpus-compatible counts, full colliding row names and class-NLL floor.

    Same 0/1 objective and n log n - sum n_c log n_c contribution as the opening
    audit. No row selection, weight repair or conflict hiding is performed.
    """
    buckets = {name: defaultdict(lambda: defaultdict(list)) for name in KEYS}
    total = weighted = 0
    for row in rows:
        total += 1
        label = C._label(row)
        keys = context_keys(row)  # validate continuations before the zero-weight check
        if not row['target']['loss_weight']:
            continue
        weighted += 1
        reference = {k: row['identity'][k] for k in ('trajectory_id', 'prefix_index', 'split')}
        for name, key in keys.items():
            buckets[name][key][label].append(reference)
    result = {'rows': total, 'weighted_rows': weighted, 'unweighted_rows': total - weighted}
    for name, contexts in buckets.items():
        conflicts, floor = [], 0.
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


def _preflight(directory: Path, expected_sha256: str) -> tuple[dict, list[dict]]:
    _require(not directory.is_symlink() and not (directory / 'MANIFEST.json').is_symlink(), 'unsafe source directory/manifest')
    _require(T._file_sha(directory / 'MANIFEST.json') == expected_sha256, 'manifest hash mismatch')
    manifest = C._read(directory / 'MANIFEST.json')
    _require(manifest['schema'] == W.SCHEMA and manifest['training_admitted'] is False and
             manifest['status'] == 'REACHED_CANDIDATES_REPLAYED_ADMISSION_BLOCKED', 'reached candidate manifest required')
    required = {'PLAN.json', 'REQUESTS.json', 'WORLDS.json', 'candidate-prefixes.jsonl',
                'ACCEPTED-AUDIT.json', 'REPLAY.json', 'COVERAGE.json'}
    _require(required <= set(manifest['files']), 'all consumed members must be hash-bound')
    missing = []
    for name, pin in manifest['files'].items():
        _require(isinstance(name, str) and name not in ('', '.', '..') and '/' not in name and '\\' not in name,
                 'unsafe dataset member: ' + str(name))
        _require(isinstance(pin, dict) and set(pin) == {'bytes', 'sha256'} and type(pin['bytes']) is int
                 and pin['bytes'] >= 0 and isinstance(pin['sha256'], str) and len(pin['sha256']) == 64,
                 'invalid dataset member pin: ' + name)
        path = directory / name
        _require(not path.is_symlink(), 'unsafe dataset member: ' + name)
        if not path.exists():
            missing.append({'path': name, **pin})
        else:
            _require(path.is_file() and path.stat().st_size == pin['bytes'] and T._file_sha(path) == pin['sha256'],
                     'dataset member hash mismatch: ' + name)
    _require(manifest['plan_sha256'] == manifest['files']['PLAN.json']['sha256'], 'plan hash binding mismatch')
    return manifest, missing


def _check_groups(plan: dict, requests: list[dict], allow_legacy_import: bool) -> None:
    _source_transition(plan['sources'], W.pins(), allow_legacy_import)
    imported_plan = copy.deepcopy(plan)
    imported_plan['sources'] = W.pins()
    W.validate_plan(imported_plan)  # same plan/source shape checks after the explicit source join
    imported_requests, seen_ids, seen_cuts = [], set(), set()
    expected_bindings = [{'path': p, 'sha256': h} for p, h in sorted(plan['sources'].items())]
    for request in requests:
        tid = request['trajectory_id']
        cut = (request['physical_world_id'], request['observation']['player'], request['observation']['step'])
        _require(tid not in seen_ids, 'duplicate trajectory: ' + tid)
        _require(cut not in seen_cuts, 'duplicate reached cut: ' + repr(cut))
        seen_ids.add(tid); seen_cuts.add(cut)
        bindings = request['provenance']['source_bindings']
        _require(_same(sorted(bindings, key=lambda b: b['path']), expected_bindings), 'request source bindings mismatch: ' + tid)
        migrated = copy.deepcopy(request)
        migrated['provenance']['source_bindings'] = [{'path': p, 'sha256': h} for p, h in sorted(W.pins().items())]
        imported_requests.append(migrated)
    W.check_request_groups(imported_requests, imported_plan)  # unchanged ancestry/seat/world/seed/split checks
    expected_cuts = {(W.world_id(w), w['seat'], s) for w in plan['worlds'] for s in plan['capture_steps']}
    _require(seen_cuts == expected_cuts, 'missing/extra predeclared reached cuts')


def _replay_ancestry(directory: Path, manifest: dict, plan: dict, requests: list[dict],
                     allow_legacy_import: bool, check_budget) -> dict:
    """Exogenous retained observation replay, NEVER environment initialization/step.

    Source identity conversion is explicit; all policy/memory/observation/action
    fields remain exact. Does not independently re-simulate native physics.
    """
    from farm_neat.training.v0_bc import _json_loads
    by_world = {W.world_id(w): w for w in plan['worlds']}
    by_cut = {(r['physical_world_id'], r['observation']['step']): r for r in requests}
    receipts = C._read(directory / 'WORLDS.json')
    _require(len(receipts) == len(by_world) and {r['physical_world_id'] for r in receipts} == set(by_world),
             'one receipt per predeclared world required')
    checked = steps = 0
    for receipt in receipts:
        wid, world = receipt['physical_world_id'], receipt['spec']
        _require(by_world.get(wid) == world, 'world receipt/plan mismatch')
        tape = receipt['tape']
        _require(tape['path'] in manifest['files'] and manifest['files'][tape['path']]['sha256'] == tape['sha256'],
                 'observation tape not bound by manifest')
        agent = W.V0Agent(W.ReachedTeacher(world['families']), labor_factor=4., hands_mode='daily',
                          belief=copy.deepcopy(T.SETTINGS['belief']), game_id='world/' + wid)
        action_hash, obs_hash = hashlib.sha256(), hashlib.sha256()
        count = 0
        with gzip.open(directory / tape['path'], 'rt') as stream:
            for line in stream:
                check_budget()
                row = _json_loads(line)
                _require(row['step'] == count == row['observation']['step'], 'noncontiguous native observation stream')
                request = by_cut.get((wid, count))
                if request is not None:
                    imported, _ = _import_snapshot(request['continuation'], allow_legacy_import)
                    _require(E.wire_hash(E.snapshot_agent(agent, row['observation'])) == E.wire_hash(imported),
                             'native ancestry/continuation mismatch: ' + request['trajectory_id'])
                    _require(E.wire_hash(row['observation']) == E.wire_hash(request['observation']), 'native cut observation mismatch')
                joint = [copy.deepcopy(W.PASS_ACTION), copy.deepcopy(W.PASS_ACTION)]
                joint[world['seat']] = agent(copy.deepcopy(row['observation']))
                _require(E.wire_hash(joint) == E.wire_hash(row['joint_action']), f'native action stream mismatch: {wid}/{count}')
                action_hash.update((E.wire(joint) + '\n').encode())
                obs_hash.update((E.wire(row['observation']) + '\n').encode())
                if request is not None:
                    _require(action_hash.hexdigest() == request['verification']['native_prefix_actions_sha256'] and
                             obs_hash.hexdigest() == request['verification']['native_prefix_observations_sha256'], 'cut prefix stream hash mismatch')
                    checked += 1
                count += 1
        _require(count == receipt['steps'] == V.LAST_STEP + 1, 'full own replay length mismatch')
        _require(action_hash.hexdigest() == receipt['joint_actions_sha256'] and obs_hash.hexdigest() == receipt['observations_sha256'],
                 'full replay stream hash mismatch')
        _require(E.agent_signature(agent) == receipt['terminal']['agent_sha256'], 'terminal controller/Plan replay mismatch')
        steps += count
    _require(checked == len(requests), 'not every reached cut replayed')
    return {'status': 'EXACT_RETAINED_OWN_STREAMS_REPLAYED', 'own_actions_replayed': steps,
            'cuts_replayed': checked, 'native_games_started': 0, 'native_engine_steps': 0,
            'physics_replayed': False, 'origin': 'HASH_BOUND_PRODUCER_EVIDENCE'}


def admit_dataset(directory: str | Path, expected_sha256: str, output: str | Path, *, backend: str = 'fast',
                  allow_legacy_import: bool = False, cpu_seconds: int = 1800) -> dict:
    """Write a new last-written CORPUS gate. Any failure keeps admission false.

    Missing source members are named with original hashes; unknown conflict
    counts/floors are null, NEVER fabricated zero. No existing evidence is edited.
    """
    _require(backend in ('fast', 'reference'), 'unknown menu backend')
    _require(type(cpu_seconds) is int and 1 <= cpu_seconds <= 2400, 'CPU budget must be 1..2400 seconds')
    output, directory = Path(output), Path(directory)
    _require(not any(p.is_symlink() for p in (output, *output.parents)), 'symlinked output')
    output.mkdir(parents=True, exist_ok=False)
    start = time.process_time()
    def check_budget():
        _require(time.process_time() - start <= cpu_seconds, 'admission CPU budget exhausted')
    report = {'schema': SCHEMA, 'status': 'BLOCKED_VALIDATION', 'training_admitted': False,
              'source_manifest_sha256': expected_sha256, 'sources': source_pins(), 'backend': backend,
              'legacy_import_requested': allow_legacy_import, 'audit': None, 'failures': [], 'missing_evidence': [],
              'native_games_started': 0, 'native_engine_steps': 0, 'model_calls': 0, 'optimizer_updates': 0,
              'bc_reader_compatible': False, 'files': {},
              'consumer': 'load_training_dataset(..., split=...) -> accumulate_reached_gradients(model, data.trajectories, microbatch_size=N); opening v0_bc.Trainer is not connected'}
    stage = 'preflight'
    try:
        manifest, missing = _preflight(directory, expected_sha256)
        report['missing_evidence'] = missing
        report['source_transition'] = _source_transition(manifest['sources'], W.pins(), allow_legacy_import)
        if missing:
            report['status'] = 'BLOCKED_MISSING_EVIDENCE'
        else:
            from farm_neat.training.v0_bc import _json_loads
            stage = 'source groups'
            plan = C._read(directory / 'PLAN.json')
            _require(plan['sources'] == manifest['sources'], 'manifest/plan sources mismatch')
            document = C._read(directory / 'REQUESTS.json')
            _require(document['schema'] == W.SCHEMA, 'request document schema mismatch')
            requests = document['requests']
            _check_groups(plan, requests, allow_legacy_import)
            rows = [_json_loads(s) for s in (directory / 'candidate-prefixes.jsonl').read_text().splitlines()]
            by_id = {r['trajectory_id']: [] for r in requests}
            for row in rows:
                tid = row['identity']['trajectory_id']
                _require(tid in by_id, 'prefix belongs to an absent trajectory: ' + tid)
                by_id[tid].append(row)
            counts = manifest['counts']
            _require(counts['trajectories'] == len(requests) and counts['prefixes'] == len(rows) and
                     counts['weighted_prefixes'] == sum(r['target']['loss_weight'] for r in rows), 'source corpus count mismatch')
            validated, receipts = [], []
            for request in requests:
                check_budget(); stage = 'trajectory ' + request['trajectory_id']
                try:
                    new, receipt = validate_trajectory(request, by_id[request['trajectory_id']], backend=backend,
                                                       allow_legacy_import=allow_legacy_import)
                    validated.extend(new); receipts.append(receipt)
                except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                    report['failures'].append({'stage': 'trajectory', 'trajectory_id': request['trajectory_id'], 'reason': str(exc)})
            _require(not report['failures'], 'one or more reached trajectories refused; no partial admission')
            stage = 'conflict audit'; check_budget()
            report['audit'] = audit_rows(validated)
            report['audit_by_split'] = {s: audit_rows(r for r in validated if r['identity']['split'] == s)
                                        for s in ('train', 'dev', 'test')}
            stage = 'native ancestry replay'
            report['replay'] = _replay_ancestry(directory, manifest, plan, requests, allow_legacy_import, check_budget)
            _require(_same(W.accepted_audit(rows), C._read(directory / 'ACCEPTED-AUDIT.json')),
                     'unchanged opening-auditor control mismatch')
            stage = 'write gate'; check_budget()
            # Self-contained consumer rows/requests; source tapes stay in the producer dataset.
            W._write(output / 'REQUESTS.json', document)
            W._write(output / 'CUT-REPLAY.json', {'receipts': receipts, 'scope': report['replay']})
            W._write(output / 'SOURCE-MANIFEST.json', manifest)
            with (output / 'prefixes.jsonl').open('x') as stream:
                for row in validated:
                    stream.write(E.wire(row) + '\n')
            report['counts'] = {'trajectories': len(requests), 'prefixes': len(validated),
                                'weighted_prefixes': report['audit']['weighted_rows']}
            report['training_admitted'] = all(report['audit'][k]['conflicting_contexts'] == 0 for k in KEYS)
            report['status'] = 'ADMITTED_REACHED_CORPUS' if report['training_admitted'] else 'BLOCKED_CONFLICTS'
            report['files'] = {p.name: {'sha256': T._file_sha(p), 'bytes': p.stat().st_size}
                               for p in sorted(output.iterdir()) if p.is_file()}
    except (ValueError, OSError, KeyError, TypeError, IndexError, AttributeError) as exc:
        report['training_admitted'] = False
        report['status'] = 'BLOCKED_VALIDATION'
        report['failures'].append({'stage': stage, 'reason': str(exc)})
    report['cpu_seconds'] = time.process_time() - start
    W._write(output / 'CORPUS.json', report)  # last; missing/blocked report never grants admission
    return report


def load_rows(directory: str | Path, expected_sha256: str, *, split: str) -> list[dict]:
    """Explicit reached-only consumer. Revalidates all trajectories before split selection.

    This is deliberately not a Dataset accepted by the unchanged opening BC loop.
    Use load_training_dataset and accumulate_reached_gradients for the explicit
    memory-aware, per-trajectory/STOP-weighted consumer.
    """
    from farm_neat.training.v0_bc import _json_loads
    directory = Path(directory)
    _require(T._file_sha(directory / 'CORPUS.json') == expected_sha256, 'corpus gate hash mismatch')
    gate = C._read(directory / 'CORPUS.json')
    _require(gate.get('schema') == SCHEMA and gate.get('training_admitted') is True and
             gate.get('status') == 'ADMITTED_REACHED_CORPUS', 'reached corpus not admitted')
    _require(gate['sources'] == source_pins(), 'admission sources changed')
    _require(split in ('train', 'dev', 'test'), 'explicit reached split required')
    expected_names = {'REQUESTS.json', 'CUT-REPLAY.json', 'SOURCE-MANIFEST.json', 'prefixes.jsonl'}
    _require(set(gate['files']) == expected_names, 'closed reached dataset members required')
    for name, pin in gate['files'].items():
        p = directory / name
        _require(p.is_file() and not p.is_symlink() and p.stat().st_size == pin['bytes'] and T._file_sha(p) == pin['sha256'],
                 'admitted member hash mismatch: ' + name)
    rows = [_json_loads(s) for s in (directory / 'prefixes.jsonl').read_text().splitlines()]
    requests = C._read(directory / 'REQUESTS.json')['requests']
    by_id = {r['trajectory_id']: [] for r in requests}
    _require(len(by_id) == len(requests), 'duplicate admitted trajectory')
    for row in rows:
        # Validate the stored codec BEFORE stripping it for source reconstruction.
        # Even zero-weight rows must retain precise malformed/HOLD/dawn refusals.
        scorer_inputs(row)
        tid = row['identity']['trajectory_id']
        _require(tid in by_id, 'admitted row without source request')
        by_id[tid].append(row)
    for request in requests:
        sequence = by_id[request['trajectory_id']]
        candidates = copy.deepcopy(sequence)
        for row in candidates:
            row['schema'] = W.ROW_SCHEMA
            del row['trainer_only']['continuation']
        rebuilt, _ = validate_trajectory(request, candidates, backend=gate['backend'],
                                         allow_legacy_import=gate['legacy_import_requested'])
        _require(_same(rebuilt, sequence), 'admitted continuation/prefix mismatch')
    audit = audit_rows(rows)
    _require(_same(audit, gate['audit']) and all(audit[k]['conflicting_contexts'] == 0 for k in KEYS), 'admitted conflict audit mismatch')
    return [row for row in rows if row['identity']['split'] == split]


@dataclass(frozen=True)
class ReachedDataset:
    """Separate from v0_bc.Dataset: complete rows keep replay state trainer-side.

    Only load_training_dataset establishes admission. Tuples group complete paths
    without length normalization; controls are retained but not sampled for fit.
    Bind this identity AND source_pins() in any future checkpoint integration.
    """
    corpus_sha256: str
    split: str
    trajectories: tuple[tuple[dict, ...], ...]
    controls: tuple[tuple[dict, ...], ...]
    validation: dict

    @property
    def binding(self) -> dict:
        return {'schema': TRAINING_SCHEMA, 'corpus_sha256': self.corpus_sha256,
                'split': self.split, 'objective': REACHED_OBJECTIVE,
                'trajectory_ids': [p[0]['identity']['trajectory_id'] for p in self.trajectories],
                'control_trajectory_ids': [p[0]['identity']['trajectory_id'] for p in self.controls],
                'rows_sha256': E.wire_hash([*self.trajectories, *self.controls])}


def _training_path(path: Sequence[dict]) -> list[dict]:
    """Cheap batch guard after full gate admission, not a native-origin proof.

    load_rows already reconstructs each menu with prefix-local legality. Here we
    recheck the continuation, canonical chain and objective weights before ANY
    scoring or gradient mutation, including for zero-weight control/cap rows.
    """
    _require(bool(path), 'nonempty reached training path required')
    inputs, previous = [], None
    try:
        first = path[0]
        identity = {k: v for k, v in first['identity'].items() if k != 'prefix_index'}
        memory_sha = None
        for index, row in enumerate(path):
            x = scorer_inputs(row)  # Full codec/boundary check even when weight is zero.
            C._label(row)
            _require(type(row['identity']['prefix_index']) is int and row['identity']['prefix_index'] == index
                     and _same({k: v for k, v in row['identity'].items() if k != 'prefix_index'}, identity),
                     'reached training prefix identity/order mismatch')
            memory = row['trainer_only']['continuation']
            current_sha = E.wire_hash(memory)
            if index == 0:
                memory_sha = current_sha
                _require(x['prefix'].edit_count == 0 and x['prefix'].last_key == -1
                         and _same(x['prefix'].plan.canonical(), memory['plan']),
                         'reached training path must start at its exact continuation prefix')
            else:
                _require(current_sha == memory_sha, 'reached training continuation changed within path')
                actual = T.continuation_signature(x['prefix'].plan, x['prefix'].last_key,
                                                   x['prefix'].edit_count, False)
                _require(_same(actual, previous), 'reached training canonical prefix chain mismatch')
            target = row['target']
            previous = target['continuation']
            stopped = previous['terminated']
            _require(type(stopped) is bool and stopped == (index == len(path) - 1),
                     'reached training path needs exactly one final STOP')
            forced = x['prefix'].edit_count == V.MAX_EDITS_PER_EPOCH
            kind = ('forced_cap' if forced else 'sampled') if stopped else 'not_stop'
            _require(row['candidates']['forced_stop'] is forced and target['stop_kind'] == kind
                     and _same(target['loss_weight'], int(len(path) > 1 and not forced)),
                     'reached training STOP/loss weight mismatch')
            inputs.append(x)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError('malformed reached training path: ' + str(exc)) from exc
    return inputs


def load_training_dataset(directory: str | Path, expected_sha256: str, *, split: str) -> ReachedDataset:
    """Gate -> complete source-validated trajectories, with all controls retained.

    All source trajectories and splits are validated by load_rows BEFORE split
    selection. No Dataset cast or monkeypatch of the opening-only trainer occurs.
    """
    rows = load_rows(directory, expected_sha256, split=split)
    groups = defaultdict(list)
    for row in rows:
        groups[row['identity']['trajectory_id']].append(row)
    active, controls = [], []
    for _, sequence in sorted(groups.items()):
        _training_path(sequence)
        destination = active if any(r['target']['loss_weight'] for r in sequence) else controls
        destination.append(tuple(sequence))
    return ReachedDataset(expected_sha256, split, tuple(active), tuple(controls),
                          {'selected_rows': len(rows),
                           'selected_weighted_rows': sum(r['target']['loss_weight'] for r in rows),
                           'trainable_trajectories': len(active), 'control_trajectories': len(controls),
                           'all_source_splits_revalidated': True, 'native_games': 0, 'optimizer_updates': 0})


def weighted_path_nll(logits, targets, weights, *, valid=None):
    """Sum exact class NLLs with exported 0/1 weights; sampled STOP is weighted.

    Forced-cap STOP and whole KEEP/STOP controls have zero weight. Even these
    rows are passed to class_nll for target/padding validity, not silently dropped.
    This is a likelihood helper, NOT data admission or a trajectory mean.
    """
    import torch
    _require(len(logits) == len(targets) == len(weights) > 0, 'aligned nonempty weighted path required')
    masks = [None] * len(logits) if valid is None else valid
    _require(len(masks) == len(logits), 'aligned prefix validity masks required')
    _require(all(type(w) in (int, float) and w in (0, 1) for w in weights), 'native 0/1 loss weight required')
    return torch.stack([A.class_nll(z, c, valid=m) * w
                        for z, c, w, m in zip(logits, targets, weights, masks)]).sum()


def accumulate_reached_gradients(model: A.V0Scorer, trajectories: Sequence[Sequence[dict]], *,
                                 microbatch_size: int) -> float:
    """Memory-aware BC gradients on complete paths from load_training_dataset.

    Objective = sum_t sum_prefix weight * class_NLL / #trainable_trajectories.
    One global denominator survives a short microbatch tail. Unweighted controls
    are checked, not scored or counted; forced STOP is checked and zero weighted.
    Scoring sees only state, canonical prefix, support, phi and declared HOLD
    memory. All other continuation/provenance stays trainer-side. No optimizer
    step, source migration, sampling, checkpoint or free-running evaluation occurs.
    """
    import torch
    _require(bool(trajectories) and type(microbatch_size) is int and microbatch_size > 0,
             'nonempty reached batch and positive microbatch required')
    prepared = [(path, _training_path(path)) for path in trajectories]
    active = [(path, inputs) for path, inputs in prepared if any(r['target']['loss_weight'] for r in path)]
    _require(bool(active), 'at least one trainable reached trajectory required')
    model.zero_grad(set_to_none=True)
    total = 0.
    for start in range(0, len(active), microbatch_size):
        chunk = active[start:start + microbatch_size]
        inputs = [x for _, path_inputs in chunk for x in path_inputs]
        pack = model.score_batch([x['state'] for x in inputs], [x['prefix'] for x in inputs],
                                 [x['legal_ids'] for x in inputs], [x['phi'] for x in inputs],
                                 memories=[x['memory'] for x in inputs])
        offset, losses = 0, []
        for path, _ in chunk:
            end = offset + len(path)
            losses.append(weighted_path_nll(list(pack.logits[offset:end]),
                          [r['target']['class_indices'] for r in path],
                          [r['target']['loss_weight'] for r in path], valid=list(pack.valid[offset:end])))
            offset = end
        loss = torch.stack(losses).sum() / len(active)
        _require(bool(torch.isfinite(loss)), 'nonfinite reached class-path loss')
        loss.backward()
        total += float(loss.detach())
    _require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
             'nonfinite reached class-path gradient')
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    admit = commands.add_parser('admit')
    admit.add_argument('--dataset', type=Path, required=True)
    admit.add_argument('--manifest-sha256', required=True)
    admit.add_argument('--output', type=Path, required=True)
    admit.add_argument('--backend', choices=('fast', 'reference'), default='fast')
    admit.add_argument('--legacy-codec-import', action='store_true')
    admit.add_argument('--cpu-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    try:
        report = admit_dataset(args.dataset, args.manifest_sha256, args.output, backend=args.backend,
                               allow_legacy_import=args.legacy_codec_import, cpu_seconds=args.cpu_seconds)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + '\n')
    print(E.wire({k: report[k] for k in ('status', 'training_admitted', 'audit', 'failures', 'missing_evidence')}))
    return 0 if report['training_admitted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
