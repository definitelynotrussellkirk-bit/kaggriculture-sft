"""Append-only exact search attempts, WDL incumbents, and regret-gated export.

Commit ATTEMPT *before* spending its charged budget; append OUTCOME separately.
An unresolved attempt is CENSORED_PENDING, never a zero-payoff example. Charges
are conservative cumulative amounts; a result may increase but never refund them.
SQLite FULL-synchronous transactions serialize writers. Rows and their hash chain
are immutable; best is a derived index, with first-committed incumbents winning
WDL ties; no challenger is selected unless it strictly beats the frozen control.
Repeated canonical event bytes are idempotent; conflicting IDs refuse.

Inputs are trainer-only. start binds the full exact snapshot, own continuation,
RNG, world, rival, seat and actor context, not an approximate neighbor. A suffix
contains all explicit dawn edits plus its declared tail. Illegal/unfinished V0
paths and public primitives can be archived, but cannot inherit reachability.

The concrete exact-result adapter consumes existing v0_regret_pairs request and
result files. It checks their byte identities, source closure, original/fresh
attempts, response validation and derived fields WITHOUT calling any game runner.
Only native, complete, freshly replayed pairs enter best. The captured-suffix
adapter restores the counterfactual codec's BEFORE_BOTH_ACTORS dawn cut and runs
strict multi-epoch V0 paths through the existing agent/compiler/executor. Sparse
schedules explicitly KEEP at omitted dawns; arbitrary captured-policy tails refuse.
Teacher export still calls v0_regret_admission.admit_pair and requires custody for
EVERY cell. That owner refuses live controllers, so suffix receipts cannot be
cast into opening teachers or used to train automatically.

Hashes authenticate bytes and observable joins, NOT whether a producer really
ran a game, its registration chronology, or undisclosed holdout exposure. Retain
and authenticate producer files outside this archive. No replay tapes, weights,
secrets or full world snapshots are copied into the journal. Missing/stale evidence
fails closed. SQLite corruption is detected, not silently repaired; a separately
retained verify head receipt is needed to detect replacement by an older whole DB.

Read-only/append commands use no games, GPU, training, deletion or compaction.
The explicit maintainer-only execute-suffix --native command charges four suffix
starts before running two arms and two fresh-process replays from a retained cut.
It requires PYTHONHASHSEED=0, the same runtime as capture, an immutable request,
a new evaluation ID and an unused output path. It never retries a charged ID.

Commands (JSON stdout):
  python -m farm_neat.training.v0_search_archive append --archive A.sqlite --input EVENT.json
  python -m farm_neat.training.v0_search_archive verify --archive A.sqlite
  python -m farm_neat.training.v0_search_archive best --archive A.sqlite
  python -m farm_neat.training.v0_search_archive export-teachers --archive A.sqlite \
      --input BUNDLE.json --protocol PROTOCOL.json
  PYTHONHASHSEED=0 python -m farm_neat.training.v0_search_archive execute-suffix \
      --native --archive A.sqlite --request R.json --output RESULT.json --evaluation-id E1
All commands accept --evidence-root (default repository root). JSON inputs and
referenced files must be bounded, data-only, local files without symlink traversal.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sqlite3
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.evaluation import v0_regret_pairs as E
from farm_neat.training import v0_regret_admission as admission
from farm_neat.training import v0_regret_records as R
from farm_neat.strategy import counterfactual as C

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'farm-v0-search-archive-v1'
CANDIDATE_SCHEMA = SCHEMA + '-candidate'
EVENT_SCHEMA = SCHEMA + '-event'
MAX_EVENT_BYTES = 512 * 1024
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
ZERO = '0' * 64
STATUSES = ('COMPLETE', 'ILLEGAL', 'DUPLICATE', 'FAILED', 'CENSORED', 'REFUSED', 'UNSUPPORTED')
CHARGES = ('cpu_seconds', 'native_starts', 'candidate_attempts')
START_FIELDS = ('snapshot_sha256', 'continuation_sha256', 'actor_context_sha256',
                'world_sha256', 'rng_sha256', 'opponent_sha256', 'source_group_id', 'seat', 'step')
DIAGNOSTICS = ('own_cash_delta', 'margin_delta', 'learned_value')
LIMITS = ('BYTE_CUSTODY_NOT_EXTERNAL_EXECUTION_AUTHENTICATION',
          'NO_FRESH_GAMES_OR_POLICY_STRENGTH_MEASUREMENT',
          'CAPTURED_SUFFIX_BEST_SUPPORTED_TEACHER_SCHEMA_REMAINS_OPENING_ONLY',
          'CALLER_OWNS_PREDECLARATION_AND_LIFETIME_EXPOSURE')


class ArchiveError(ValueError):
    """Stable refusal code, without echoing private input bytes."""


def need(ok: bool, code: str) -> None:
    if not ok:
        raise ArchiveError(code)


def wire(value: Any) -> bytes:
    R._safe(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def digest(value: Any) -> str:
    return hashlib.sha256(wire(value)).hexdigest()


def seal(value: dict) -> dict:
    result = copy.deepcopy({k: v for k, v in value.items() if k != 'content_sha256'})
    return {**result, 'content_sha256': digest(result)}


def closed(value: Any, fields, code: str) -> None:
    need(type(value) is dict and set(value) == set(fields), code)


def text(value: Any, code: str) -> str:
    need(type(value) is str and 0 < len(value) <= 256 and bool(value.strip()), code)
    return value


def hashed(value: dict, fields, schema: str, code: str) -> None:
    closed(value, (*fields, 'content_sha256'), code + '_FIELDS')
    need(value['schema'] == schema and wire(value) == wire(seal(value)), code + '_HASH_OR_SCHEMA')


def safe_path(path: str | Path) -> Path:
    path = Path(path).absolute()
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'SYMLINK_PATH')
    return path


def read_json(path: str | Path, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> dict:
    p = safe_path(path)
    need(p.is_file(), 'MISSING_ARTIFACT')
    with p.open('rb') as stream:
        data = stream.read(max_bytes + 1)
    need(len(data) <= max_bytes, 'ARTIFACT_TOO_LARGE')
    try:
        value = R._loads(data.decode('utf-8'))
    except UnicodeError as exc:
        raise ArchiveError('MALFORMED_JSON') from exc
    R._safe(value)
    need(type(value) is dict, 'JSON_OBJECT_REQUIRED')
    return value


def _reference(ref: dict) -> None:
    closed(ref, ('path', 'sha256', 'bytes'), 'ARTIFACT_FIELDS')
    name = ref['path']
    need(type(name) is str and 0 < len(name) <= 1024 and '\\' not in name,
         'ARTIFACT_PATH')
    p = PurePosixPath(name)
    need(not p.is_absolute() and str(p) == name and
         all(part not in ('', '.', '..') for part in p.parts), 'ARTIFACT_PATH')
    R._hash(ref['sha256'], '$.artifact.sha256')
    R._int(ref['bytes'], '$.artifact.bytes', 1, MAX_ARTIFACT_BYTES)


def artifact_ref(path: str | Path, *, root: str | Path = ROOT) -> dict:
    root = safe_path(root).resolve()
    p = safe_path(path)
    need(p.is_file() and p.resolve().is_relative_to(root), 'ARTIFACT_PATH')
    with p.open('rb') as stream:
        data = stream.read(MAX_ARTIFACT_BYTES + 1)
    need(0 < len(data) <= MAX_ARTIFACT_BYTES, 'ARTIFACT_TOO_LARGE')
    result = dict(path=p.resolve().relative_to(root).as_posix(),
                  sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    _reference(result)
    return result


def _artifact_bytes(ref: dict, *, root: Path) -> bytes:
    _reference(ref)
    p = safe_path(root / ref['path'])
    need(p.resolve().is_relative_to(root.resolve()), 'ARTIFACT_PATH')
    need(p.is_file(), 'MISSING_ARTIFACT')
    with p.open('rb') as stream:
        data = stream.read(ref['bytes'] + 1)
    need(len(data) == ref['bytes'] and hashlib.sha256(data).hexdigest() == ref['sha256'],
         'ARTIFACT_MISMATCH')
    return data


def read_artifact(ref: dict, *, root: Path) -> dict:
    data = _artifact_bytes(ref, root=root)
    try:
        value = R._loads(data.decode('utf-8'))
    except UnicodeError as exc:
        raise ArchiveError('MALFORMED_JSON') from exc
    R._safe(value)
    need(type(value) is dict, 'JSON_OBJECT_REQUIRED')
    return value


def source_pins() -> dict[str, str]:
    names = ('src/farm_neat/training/v0_search_archive.py',
             'src/farm_neat/training/v0_regret_admission.py',
             'src/farm_neat/strategy/counterfactual.py')
    return {**E.source_pins(), **C.source_pins(), **{p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                 for p in names}}


def _suffix(suffix: dict) -> None:
    closed(suffix, ('kind', 'epochs', 'tail'), 'SUFFIX_FIELDS')
    need(suffix['kind'] in ('V0', 'PUBLIC_PRIMITIVES'), 'SUFFIX_KIND')
    need(suffix['tail'] in ('KEEP_TO_TERMINAL', 'CAPTURED_CONTINUATION'), 'SUFFIX_TAIL')
    epochs = suffix['epochs']
    need(type(epochs) is list and 1 <= len(epochs) <= 30, 'SUFFIX_EPOCHS')
    previous = -1
    for epoch in epochs:
        closed(epoch, ('step', 'raw_actions', 'canonical_plan'), 'EPOCH_FIELDS')
        R._int(epoch['step'], '$.epoch.step', 0, 718)
        need(epoch['step'] > previous, 'EPOCH_ORDER')
        previous = epoch['step']
        actions = epoch['raw_actions']
        need(type(actions) is list and len(actions) <= V.MAX_EDITS_PER_EPOCH + 1, 'ACTION_LIMIT')
        if suffix['kind'] == 'V0':
            need(epoch['step'] % 24 == 0, 'V0_DAWN_REQUIRED')
            # Out-of-vocabulary and incomplete proposals must remain archivable.
            need(all(type(a) is int and -(2**31) <= a < 2**31 for a in actions), 'RAW_ACTION_TYPE')
            if epoch['canonical_plan'] is not None:
                R._plan(epoch['canonical_plan'], '$.epoch.canonical_plan')
        else:
            need(epoch['canonical_plan'] is None, 'PUBLIC_NOT_CANONICAL_PLAN')
            for action in actions:
                closed(action, ('farmer', 'hands', 'market'), 'PUBLIC_ACTION_FIELDS')
                need(all(type(action[k]) is list for k in action), 'PUBLIC_ACTION_TYPE')


def validate_candidate(candidate: dict) -> dict:
    R._safe(candidate)
    hashed(candidate, ('schema', 'start', 'start_artifact', 'suffix', 'proposal_family',
        'policy_sha256', 'source_pins', 'comparison'), CANDIDATE_SCHEMA, 'CANDIDATE')
    start = candidate['start']
    closed(start, START_FIELDS, 'START_FIELDS')
    for name in START_FIELDS:
        if name.endswith('_sha256'):
            R._hash(start[name], '$.start.' + name)
    R._int(start['seat'], '$.start.seat', 0, 1)
    R._int(start['step'], '$.start.step', 0, 718)
    text(start['source_group_id'], 'SOURCE_GROUP')
    _reference(candidate['start_artifact'])
    _suffix(candidate['suffix'])
    need(candidate['suffix']['epochs'][0]['step'] == start['step'], 'SUFFIX_START')
    text(candidate['proposal_family'], 'PROPOSAL_FAMILY')
    R._hash(candidate['policy_sha256'], '$.policy_sha256')
    pins = candidate['source_pins']
    need(type(pins) is dict and 0 < len(pins) <= 512, 'SOURCE_PINS')
    for name, sha in pins.items():
        text(name, 'SOURCE_PATH'); R._hash(sha, '$.source_pins')
    comp = candidate['comparison']
    closed(comp, ('objective', 'control_path_sha256', 'control_suffix_sha256',
                  'protocol_sha256', 'split', 'round_id'), 'COMPARISON_FIELDS')
    need(comp['objective'] == 'TERMINAL_WDL', 'WDL_ONLY')
    for name in ('control_path_sha256', 'control_suffix_sha256', 'protocol_sha256'):
        R._hash(comp[name], '$.comparison.' + name)
    need(comp['split'] in R.SPLITS, 'SPLIT')
    text(comp['round_id'], 'ROUND_ID')
    need(len(wire(candidate)) <= MAX_EVENT_BYTES, 'CANDIDATE_TOO_LARGE')
    return copy.deepcopy(candidate)


def _opening_suffix(record: dict, arm: str) -> dict:
    return dict(kind='V0', epochs=[dict(step=record['actor_context']['observation']['step'],
        raw_actions=copy.deepcopy(record[arm]['raw_actions']),
        canonical_plan=copy.deepcopy(record[arm]['canonical_plan']))], tail='KEEP_TO_TERMINAL')


def candidate_from_request(request: dict, *, request_ref: dict) -> dict:
    """Construct a source-bound candidate from the existing exact opening request.

    Construction and validation execute zero game turns. The external request
    retains the full original context/continuation; only its byte reference is stored.
    """
    request = E._validate_request(request)
    rec = request['record']; identity = rec['identity']
    candidate = seal(dict(schema=CANDIDATE_SCHEMA,
        start=dict(snapshot_sha256=request['initial_world_sha256'],
            continuation_sha256=identity['continuation_sha256'],
            actor_context_sha256=identity['actor_context_sha256'],
            world_sha256=request['world_sha256'], rng_sha256=E.panel.digest(request['rng_binding']),
            opponent_sha256=request['opponent']['identity']['content_sha256'],
            source_group_id=identity['source_group_id'], seat=request['seat'],
            step=rec['actor_context']['observation']['step']), start_artifact=copy.deepcopy(request_ref),
        suffix=_opening_suffix(rec, 'challenger'), proposal_family=rec['challenger']['proposal_family'],
        policy_sha256=identity['parent_checkpoint_sha256'], source_pins=source_pins(),
        comparison=dict(objective='TERMINAL_WDL', control_path_sha256=rec['learner']['path_sha256'],
            control_suffix_sha256=digest(_opening_suffix(rec, 'learner')),
            protocol_sha256=identity['protocol_sha256'], split=identity['split'], round_id=identity['round_id'])))
    return validate_candidate(candidate)


def state_key(candidate: dict) -> str:
    """Full exact state AND runtime source identity; never an approximate bucket."""
    return digest(dict(start=candidate['start'], source_pins=candidate['source_pins']))


def comparison_key(candidate: dict) -> str:
    return digest(candidate['comparison'])


def _charge(value: dict) -> None:
    closed(value, CHARGES, 'CHARGED_FIELDS')
    R._number(value['cpu_seconds'], '$.charged.cpu_seconds', 0, 1e12)
    R._int(value['native_starts'], '$.charged.native_starts', 0, 2**31 - 1)
    R._int(value['candidate_attempts'], '$.charged.candidate_attempts', 1, 2**31 - 1)


def attempt_event(evaluation_id: str, candidate: dict, *, charged: dict) -> dict:
    return seal(dict(schema=EVENT_SCHEMA, kind='ATTEMPT', evaluation_id=evaluation_id,
                     candidate=copy.deepcopy(candidate), charged=copy.deepcopy(charged)))


def outcome_event(evaluation_id: str, candidate_sha256: str, *, status: str,
                  charged: dict, evidence: dict | None = None,
                  reason_codes: list[str] | None = None, duplicate_of: str | None = None,
                  diagnostics: dict | None = None, witnesses: list[dict] | None = None) -> dict:
    return seal(dict(schema=EVENT_SCHEMA, kind='OUTCOME', evaluation_id=evaluation_id,
        candidate_sha256=candidate_sha256, status=status, charged=copy.deepcopy(charged),
        evidence=copy.deepcopy(evidence), reason_codes=copy.deepcopy(reason_codes or []),
        duplicate_of=duplicate_of, diagnostics=copy.deepcopy(diagnostics or dict.fromkeys(DIAGNOSTICS)),
        witnesses=copy.deepcopy(witnesses or [])))


def _event(event: dict) -> None:
    R._safe(event)
    need(type(event) is dict and event.get('kind') in ('ATTEMPT', 'OUTCOME'), 'EVENT_KIND')
    common = ('schema', 'kind', 'evaluation_id', 'charged')
    extra = ('candidate',) if event['kind'] == 'ATTEMPT' else (
        'candidate_sha256', 'status', 'evidence', 'reason_codes', 'duplicate_of', 'diagnostics', 'witnesses')
    hashed(event, common + extra, EVENT_SCHEMA, 'EVENT')
    text(event['evaluation_id'], 'EVALUATION_ID'); _charge(event['charged'])
    need(len(wire(event)) <= MAX_EVENT_BYTES, 'EVENT_TOO_LARGE')
    if event['kind'] == 'ATTEMPT':
        validate_candidate(event['candidate'])
        return
    R._hash(event['candidate_sha256'], '$.candidate_sha256')
    need(event['status'] in STATUSES, 'OUTCOME_STATUS')
    reasons = event['reason_codes']
    need(type(reasons) is list and len(reasons) <= 32, 'REASON_CODES')
    for reason in reasons:
        text(reason, 'REASON_CODE')
    need(len(reasons) == len(set(reasons)), 'DUPLICATE_REASON')
    need(event['status'] == 'COMPLETE' or bool(reasons), 'MISSING_REASON')
    need(event['status'] != 'COMPLETE' or event['evidence'] is not None, 'MISSING_EXACT_EVIDENCE')
    if event['evidence'] is not None:
        closed(event['evidence'], ('request', 'result'), 'EVIDENCE_FIELDS')
        for ref in event['evidence'].values():
            _reference(ref)
    need(type(event['witnesses']) is list and len(event['witnesses']) <= 8, 'WITNESS_LIMIT')
    for ref in event['witnesses']:
        _reference(ref)
    if event['duplicate_of'] is not None:
        text(event['duplicate_of'], 'DUPLICATE_TARGET')
    need((event['status'] == 'DUPLICATE') == (event['duplicate_of'] is not None), 'DUPLICATE_TARGET')
    closed(event['diagnostics'], DIAGNOSTICS, 'DIAGNOSTIC_FIELDS')
    for value in event['diagnostics'].values():
        if value is not None:
            R._number(value, '$.diagnostics')


def _paired_facts(candidate: dict, request: dict, result: dict) -> dict:
    """Read-only adapter to the existing pair owner; no evaluate/verify runner call."""
    expected = candidate_from_request(request, request_ref=candidate['start_artifact'])
    need(wire(candidate) == wire(expected), 'EXACT_CANDIDATE_REQUEST_MISMATCH')
    E._bound(result, 'RESULT_HASH_MISMATCH')
    closed(result, ('schema', 'scope', 'request_sha256', 'execution', 'arms', 'replay',
                    'comparison', 'counts', 'record', 'content_sha256'), 'PAIR_RESULT_FIELDS')
    need(result['schema'] == E.SCHEMA and result['request_sha256'] == request['content_sha256'],
         'RESULT_REQUEST_MISMATCH')
    execution = result['execution']
    E._same(execution, E.panel.bound(execution), 'EXECUTION_HASH_MISMATCH')
    closed(execution, ('kind', 'runner', 'python', 'trust', 'fresh_process_per_native_attempt',
                       'content_sha256'), 'EXECUTION_FIELDS')
    need(execution['kind'] in ('NATIVE_GAME', 'SYNTHETIC_DIAGNOSTIC'), 'EXECUTION_KIND')
    native = execution['kind'] == 'NATIVE_GAME'
    if native:
        E._same(execution, E.execution_binding(E.native_runner), 'NATIVE_SOURCE_MISMATCH')
    replay = result['replay']
    closed(replay, ('status', 'arms', 'mismatches'), 'REPLAY_FIELDS')
    need(replay['status'] in ('NOT_RUN', 'VERIFIED', 'MISMATCH'), 'REPLAY_STATUS')
    need(type(replay['arms']) is list and len(replay['arms']) in (0, 2), 'REPLAY_ARMS')
    need((replay['status'] == 'NOT_RUN') == (not replay['arms']), 'REPLAY_COVERAGE')
    ids, executions = set(), {}
    for phase, arms in (('EVALUATE', result['arms']), ('REPLAY', replay['arms'])):
        if phase == 'REPLAY' and not arms:
            continue
        need(type(arms) is list and len(arms) == 2 and
             [a.get('arm') for a in arms] == list(E.ARMS), 'PAIR_ARMS')
        for arm in arms:
            closed(arm, ('arm', 'attempt_id', 'phase', 'branch_sha256', 'status',
                         'terminal', 'error', 'response'), 'ARM_FIELDS')
            attempt_id = text(arm['attempt_id'], 'EXECUTION_ID')
            need(attempt_id not in ids, 'REUSED_EXECUTION_ID'); ids.add(attempt_id)
            need(arm['phase'] == phase, 'EXECUTION_PHASE')
            branch = E.P.bind(dict(schema=E.BRANCH_SCHEMA, request=request, arm=arm['arm'],
                                 execution=execution, phase=phase, attempt_id=attempt_id))
            need(arm['branch_sha256'] == branch['content_sha256'], 'BRANCH_BINDING')
            need(arm['status'] in ('COMPLETE', 'INCOMPLETE', 'INVALID'), 'ARM_STATUS')
            if arm['status'] == 'COMPLETE':
                terminal = E._validate_response(branch, arm['response'])
                need(terminal is not None and arm['error'] is None, 'COMPLETE_ARM_REQUIRED')
                E._same(arm['terminal'], terminal, 'TERMINAL_MISMATCH')
            else:
                need(arm['terminal'] is None, 'CENSORED_OUTCOME')
            executions[attempt_id] = digest(arm)
    mismatches = []
    if replay['arms']:
        for original, fresh in zip(result['arms'], replay['arms']):
            if original['status'] != 'COMPLETE' or fresh['status'] != 'COMPLETE':
                mismatches.append(dict(arm=original['arm'], reason='INCOMPLETE_OR_INVALID_BRANCH'))
            elif E.P.wire(E._replay_signature(original)) != E.P.wire(E._replay_signature(fresh)):
                mismatches.append(dict(arm=original['arm'], reason='SEMANTIC_REPLAY_MISMATCH',
                    original_sha256=E.P.wire_hash(E._replay_signature(original)),
                    replay_sha256=E.P.wire_hash(E._replay_signature(fresh))))
    expected_replay = dict(status=('MISMATCH' if mismatches else 'VERIFIED') if replay['arms'] else 'NOT_RUN',
                           arms=replay['arms'], mismatches=mismatches)
    E._same(replay, expected_replay, 'REPLAY_DERIVED_FIELDS_MISMATCH')
    expected_result = E._assemble(request, execution, result['arms'], expected_replay)
    E._same(result, expected_result, 'RESULT_DERIVED_FIELDS_MISMATCH')
    complete = all(a['status'] == 'COMPLETE' for a in result['arms'])
    return dict(native=native, complete=complete, fresh_replay=replay['status'] == 'VERIFIED',
        learner_wdl=result['arms'][0]['terminal']['wdl'] if complete else None,
        challenger_wdl=result['arms'][1]['terminal']['wdl'] if complete else None,
        wdl_delta=result['comparison']['wdl_delta'],
        primary_attempt_ids=[a['attempt_id'] for a in result['arms']],
        replay_attempt_ids=[a['attempt_id'] for a in replay['arms']],
        execution_identities=executions, record_sha256=result['record']['content_sha256'],
        request_sha256=request['content_sha256'], result_sha256=result['content_sha256'])


# This adapter deliberately does not manufacture regret records: that owner's
# current memory schema refuses live controllers. Suffixes can become exact best
# candidates, but remain trainer-only until an owning admission schema supports
# the entire continuation. The opening adapter/export contract remains unchanged.
SUFFIX_SCHEMA = SCHEMA + '-captured-suffix'
SUFFIX_REQUEST_SCHEMA = SUFFIX_SCHEMA + '-request'
SUFFIX_BRANCH_SCHEMA = SUFFIX_SCHEMA + '-branch'
SUFFIX_RESPONSE_SCHEMA = SUFFIX_SCHEMA + '-response'
SUFFIX_REQUEST_FIELDS = ('schema', 'snapshot', 'start', 'control', 'challenger',
    'source_pins', 'policy_sha256', 'protocol_sha256', 'split', 'round_id',
    'proposal_family', 'timeout_seconds')


def _captured_start(snapshot_ref: dict, *, evidence_root: Path, source_group_id: str):
    """Restore the full evaluator cut, NOT the projected geometry witness."""
    data = read_artifact(snapshot_ref, root=evidence_root)
    snapshot = C.Snapshot.from_dict(data)
    runtime = C.restore(snapshot)
    env, actors, seat = runtime.environment, runtime.actors, runtime.seat
    step = env.step_index
    need(type(step) is int and step in range(0, 719, 24), 'CAPTURED_DAWN_REQUIRED')
    need(type(actors[seat]) is C.V0Agent, 'CAPTURED_V0_REQUIRED')
    # A claimed BEFORE_BOTH_ACTORS cut cannot already have observed, published or
    # emitted either V0 actor's current action. Preserve unsettled predecessor
    # receipts so the first real observe settles them against the actual branch.
    for s, actor in enumerate(actors):
        obs = env.agent_observation(s)
        need(all(type(obs.get(k)) is int for k in ('step', 'day', 'hour', 'player'))
             and obs['step'] == step and obs['day'] == step // 24
             and obs['hour'] == 0 and obs['player'] == s, 'CAPTURED_OBSERVATION_PHASE')
        if type(actor) is not C.V0Agent:
            continue  # Other supported actors' state belongs to the exact codec.
        controller = actor.controller
        if step == 0:
            need(controller is None and not actor.records, 'CAPTURED_OPENING_PHASE')
        else:
            need(controller is not None, 'MISSING_CAPTURED_CONTROLLER')
            executor = controller.executor
            pending = executor.pending
            need(controller.publication_steps == tuple(range(0, 719, 24)), 'CAPTURED_PUBLICATION_CLOCK')
            need(controller.committed_step == step - 24 and executor.stage_step == step - 1
                 and executor.player == s and type(pending) is dict
                 and pending.get('step') == step - 1
                 and 'settled_observation_sha256' not in pending, 'CAPTURED_PRE_ACTION_PHASE')
            need(all(type(row) is dict and type(row.get('step')) is int and row['step'] < step
                     for row in actor.records), 'CAPTURED_RECORD_PREFIX')
    need(C.capture(env, actors, seat=seat).sha256 == snapshot.sha256, 'CAPTURED_ROUNDTRIP_DRIFT')
    own = actors[seat]
    # Hash the actor allowlist separately, but never restore from it. It excludes
    # seed, future shops and rival private/continuation state.
    dawn = C.v0_oracle.capture_dawn(own, env.agent_observation(seat))
    world = dict(kind=data['environment']['kind'], configuration=C._pack(env.configuration),
                 info=C._pack(env.info), scenario_identity=getattr(env, 'scenario_identity', None))
    rng = dict(python=data['python'], engine_info=C._pack(env.info),
               configuration=C._pack(env.configuration),
               actor_state_sha256=[digest(actor) for actor in data['actors']])
    start = dict(snapshot_sha256=snapshot.sha256,
        continuation_sha256=digest(data['actors'][seat]),
        actor_context_sha256=digest(C.v0_oracle.actor_input(dawn)),
        world_sha256=digest(world), rng_sha256=digest(rng),
        opponent_sha256=digest(data['actors'][1-seat]), source_group_id=source_group_id,
        seat=seat, step=step)
    return start, runtime


def make_suffix_request(*, snapshot_ref: dict, control: dict, challenger: dict,
        source_group_id: str, policy_sha256: str, protocol_sha256: str, split: str,
        round_id: str, proposal_family: str, timeout_seconds: float = 120.,
        evidence_root: str | Path = ROOT) -> dict:
    """Freeze one exact dawn and two complete suffixes; executes ZERO transitions.

    Each explicit epoch includes strict raw actions with a final STOP and its exact
    requested Plan. Omitted dawns are KEEP, including every dawn after the last
    explicit epoch. CAPTURED_CONTINUATION is intentionally not an execution mode.
    Use ordinary candidate/ATTEMPT rows to retain unsupported/illegal proposals.
    """
    root = safe_path(evidence_root).resolve()
    start, _ = _captured_start(snapshot_ref, evidence_root=root, source_group_id=source_group_id)
    request = seal(dict(schema=SUFFIX_REQUEST_SCHEMA, snapshot=copy.deepcopy(snapshot_ref),
        start=start, control=copy.deepcopy(control), challenger=copy.deepcopy(challenger),
        source_pins=source_pins(), policy_sha256=policy_sha256, protocol_sha256=protocol_sha256,
        split=split, round_id=round_id, proposal_family=proposal_family, timeout_seconds=timeout_seconds))
    _validate_suffix_request(request, evidence_root=root)
    return request


def _validate_suffix_request(request: dict, *, evidence_root: Path):
    hashed(request, SUFFIX_REQUEST_FIELDS, SUFFIX_REQUEST_SCHEMA, 'SUFFIX_REQUEST')
    need(request['source_pins'] == source_pins(), 'SUFFIX_SOURCE_DRIFT')
    R._number(request['timeout_seconds'], '$.timeout_seconds', .01, 3600.)
    for key in ('policy_sha256', 'protocol_sha256'):
        R._hash(request[key], '$.' + key)
    for key in ('round_id', 'proposal_family'):
        text(request[key], key.upper())
    need(request['split'] in R.SPLITS, 'SPLIT')
    closed(request['start'], START_FIELDS, 'START_FIELDS')
    text(request['start']['source_group_id'], 'SOURCE_GROUP')
    expected, runtime = _captured_start(request['snapshot'], evidence_root=evidence_root,
        source_group_id=request['start']['source_group_id'])
    need(wire(request['start']) == wire(expected), 'SUFFIX_START_DRIFT')
    for name in ('control', 'challenger'):
        suffix = request[name]; _suffix(suffix)
        need(suffix['kind'] == 'V0' and suffix['tail'] == 'KEEP_TO_TERMINAL', 'UNSUPPORTED_EXACT_SUFFIX')
        need(suffix['epochs'][0]['step'] == expected['step'], 'SUFFIX_START')
        for epoch in suffix['epochs']:
            C.v0_oracle._sequence(epoch['raw_actions'])  # No sorting/truncation/STOP injection.
            need(epoch['canonical_plan'] is not None, 'MISSING_EXACT_PLAN')
        # Only first-dawn geometry is available without execution. Later epochs
        # MUST be rechecked against their own actual states inside the runner.
        own = runtime.actors[runtime.seat]
        dawn = C.v0_oracle.capture_dawn(own, runtime.environment.agent_observation(runtime.seat))
        first = C.v0_oracle.trace_edits(dawn, suffix['epochs'][0]['raw_actions'])
        need(wire(first['plan']) == wire(suffix['epochs'][0]['canonical_plan']), 'FIRST_PLAN_MISMATCH')
    return runtime


def candidate_from_suffix_request(request: dict, *, request_ref: dict,
                                  evidence_root: str | Path = ROOT) -> dict:
    _validate_suffix_request(request, evidence_root=safe_path(evidence_root).resolve())
    candidate = seal(dict(schema=CANDIDATE_SCHEMA, start=copy.deepcopy(request['start']),
        start_artifact=copy.deepcopy(request_ref), suffix=copy.deepcopy(request['challenger']),
        proposal_family=request['proposal_family'], policy_sha256=request['policy_sha256'],
        source_pins=copy.deepcopy(request['source_pins']), comparison=dict(objective='TERMINAL_WDL',
            control_path_sha256=digest(request['control']), control_suffix_sha256=digest(request['control']),
            protocol_sha256=request['protocol_sha256'], split=request['split'], round_id=request['round_id'])))
    return validate_candidate(candidate)


def suffix_execution_binding(runner) -> dict:
    native = runner is _NATIVE_SUFFIX_RUNNER
    fp = E.panel._fingerprint(_suffix_game if native else runner)
    if native:
        fp['module'] = 'farm_neat.training.v0_search_archive'
    return E.panel.bound(dict(kind='NATIVE_GAME' if native else 'SYNTHETIC_DIAGNOSTIC',
        runner=fp, python=sys.version, trust='SOURCE_BOUND_TRUSTED_RUNNER_NOT_A_SANDBOX',
        fresh_process_per_native_attempt=native))


def _suffix_branch(request, arm, execution, phase, attempt_id=None):
    return seal(dict(schema=SUFFIX_BRANCH_SCHEMA, request=copy.deepcopy(request), arm=arm,
        execution=copy.deepcopy(execution), phase=phase,
        attempt_id=attempt_id or 'suffix-' + uuid.uuid4().hex))


def _geometry_dawn(agent, observation):
    """Small prefix witness only. Full controller lives in the start capture.

    The exact pre-publication memory hash binds the omitted controller; this
    projected witness is never an exact-state restore or a teacher record.
    """
    dawn = C.v0_oracle.capture_dawn(agent, observation)
    exact_memory_sha256 = digest(dawn['memory'])
    dawn['memory']['controller'] = None
    # Snapshot observations may be engine AttrDicts. Normalize only the JSON
    # evidence boundary, not the restored live engine/controller objects.
    return E._plain(C.v0_oracle.bound(dawn)), exact_memory_sha256


def _suffix_game(branch: dict, *, evidence_root: str | Path = ROOT) -> dict:
    """One existing-engine suffix, called in a fresh native worker by the CLI.

    Tests may replace the environment transition with an explicit data-only
    fixture, always under SYNTHETIC_DIAGNOSTIC execution binding.
    """
    hashed(branch, ('schema', 'request', 'arm', 'execution', 'phase', 'attempt_id'),
           SUFFIX_BRANCH_SCHEMA, 'SUFFIX_BRANCH')
    req = branch['request']; arm = branch['arm']
    need(arm in E.ARMS and branch['phase'] in ('EVALUATE', 'REPLAY'), 'SUFFIX_BRANCH_ROLE')
    runtime = _validate_suffix_request(req, evidence_root=safe_path(evidence_root).resolve())
    env, actors, seat = runtime.environment, runtime.actors, runtime.seat
    own = actors[seat]
    suffix = req['control' if arm == 'learner' else 'challenger']
    schedule = {e['step']: e for e in suffix['epochs']}
    policy = C.TrajectoryPolicy({step: e['raw_actions'][:-1] for step, e in schedule.items()})
    own.policy = policy  # Both arms replace only own future policy, AFTER exact restore.
    epochs, rng_events = [], []
    action_hashes = [hashlib.sha256(), hashlib.sha256()]
    joint_hash = hashlib.sha256()
    status, reason, terminal = 'INCOMPLETE', None, None
    deadline = time.monotonic() + req['timeout_seconds']
    start_step = env.step_index
    try:
        while not env.done:
            step = env.step_index
            need(start_step <= step <= 718, 'SUFFIX_EXECUTION_CLOCK')
            if time.monotonic() >= deadline:
                raise TimeoutError('suffix deadline')
            observations = [env.agent_observation(s) for s in (0, 1)]
            joint = []
            for s in (0, 1):
                obs = observations[s]
                if s != seat or step % 24:
                    joint.append(C._act(actors[s], obs, env.configuration)); continue
                expected = schedule.get(step)
                actions = expected['raw_actions'] if expected is not None else [V.STOP_ID]
                witness = {}
                def decision(state, plan, masks):
                    # V0Agent has already called controller.observe exactly as in
                    # ordinary play. Validate canonical-prefix support HERE.
                    dawn, exact_hash = _geometry_dawn(own, obs)
                    trace = C.v0_oracle.trace_edits(dawn, actions)
                    if expected is not None:
                        need(wire(trace['plan']) == wire(expected['canonical_plan']), 'EPOCH_PLAN_MISMATCH')
                    witness.update(step=step, geometry_dawn=dawn, exact_memory_sha256=exact_hash,
                        raw_actions=list(actions), trace_sha256=trace['content_sha256'],
                        after_plan=trace['plan'], hold_releases=trace['hold_releases'])
                    return list(actions)
                own.policy = decision
                try:
                    action = C._act(own, obs, env.configuration)
                finally:
                    own.policy = policy
                need(bool(witness) and wire(own.plan.canonical()) == wire(witness['after_plan']),
                     'EXECUTED_PLAN_MISMATCH')
                memory = C.v0_oracle.snapshot_agent(own)
                need(memory['hold_releases'] == witness.pop('hold_releases'), 'EXECUTED_HOLD_MISMATCH')
                witness.update(after_memory_sha256=digest(memory),
                    publication_sha256=own.controller.last_publication_sha256,
                    compiler_rejected=own.records[-1]['rejected'])
                epochs.append(witness)
                joint.append(action)
            for s, action in enumerate(joint):
                action_hashes[s].update(wire(dict(step=step, action=action)) + b'\n')
            before = digest(C.snapshot_environment(env)) if (step + 1) % 24 == 0 else None
            env.step(joint)
            need(env.step_index == step + 1, 'SUFFIX_TRANSITION_CLOCK')
            joint_hash.update(wire(dict(step=step, actions=joint,
                cash=[float(f['money']) for f in env.agent_observation(seat)['farms']])) + b'\n')
            if before is not None:
                rng_events.append(dict(step=step + 1, key=(env.info['seed'] * 1000003) ^ (step // 24),
                    pre_state_sha256=before, post_state_sha256=digest(C.snapshot_environment(env))))
        need(env.step_index == 719 and all(r.status == 'DONE' for r in env.state), 'SUFFIX_NOT_TERMINAL')
        if time.monotonic() >= deadline:
            raise TimeoutError('suffix deadline')
        cash = [float(r.reward) for r in env.state]
        need(all(math.isfinite(c) for c in cash) and
             cash == [float(f['money']) for f in env.agent_observation(seat)['farms']], 'TERMINAL_CASH_MISMATCH')
        margin = cash[seat] - cash[1-seat]
        terminal = seal(dict(cash_by_seat=cash, own_cash=cash[seat], rival_cash=cash[1-seat],
            margin=margin, wdl=(margin > 0) - (margin < 0),
            environment_sha256=digest(C.snapshot_environment(env))))
        status = 'COMPLETE'
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ArchiveError) else type(exc).__name__
    return seal(dict(schema=SUFFIX_RESPONSE_SCHEMA, branch_sha256=branch['content_sha256'],
        start_sha256=digest(req['start']), suffix_sha256=digest(suffix),
        source_pins_sha256=digest(req['source_pins']), status=status, reason=reason,
        steps=env.step_index - start_step, terminal=terminal, epochs=epochs,
        row=dict(seat=seat, actions_sha256=joint_hash.hexdigest(),
                 actions_by_seat_sha256=[h.hexdigest() for h in action_hashes]),
        rng_events=rng_events, final_actor_sha256=[digest(C.snapshot_actor(a)) for a in actors]))


def _validate_suffix_response(branch, response, *, rng_seed=None):
    hashed(response, ('schema', 'branch_sha256', 'start_sha256', 'suffix_sha256',
        'source_pins_sha256', 'status', 'reason', 'steps', 'terminal', 'epochs', 'row',
        'rng_events', 'final_actor_sha256'), SUFFIX_RESPONSE_SCHEMA, 'SUFFIX_RESPONSE')
    req = branch['request']; suffix = req['control' if branch['arm'] == 'learner' else 'challenger']
    for key, expected in (('branch_sha256', branch['content_sha256']), ('start_sha256', digest(req['start'])),
            ('suffix_sha256', digest(suffix)), ('source_pins_sha256', digest(req['source_pins']))):
        need(response[key] == expected, 'SUFFIX_RESPONSE_BINDING')
    start = req['start']['step']; seat = req['start']['seat']
    R._int(response['steps'], '$.steps', 0, 719-start)
    need(response['status'] in ('COMPLETE', 'INCOMPLETE'), 'SUFFIX_RESPONSE_STATUS')
    closed(response['row'], ('seat', 'actions_sha256', 'actions_by_seat_sha256'), 'SUFFIX_ROW_FIELDS')
    need(response['row']['seat'] == seat, 'SUFFIX_RESPONSE_SEAT')
    R._hash(response['row']['actions_sha256'], '$.actions_sha256')
    for values in (response['row']['actions_by_seat_sha256'], response['final_actor_sha256']):
        need(type(values) is list and len(values) == 2, 'SUFFIX_SEAT_HASHES')
        for value in values:
            R._hash(value, '$.seat_hash')
    need(type(response['epochs']) is list and len(response['epochs']) <= 30, 'SUFFIX_EPOCH_COVERAGE')
    complete = response['status'] == 'COMPLETE'
    if not complete:
        need(response['terminal'] is None and response['reason'] is not None, 'SUFFIX_CENSORED_OUTCOME')
        return None
    need(response['steps'] == 719-start and response['reason'] is None, 'SUFFIX_NOT_TERMINAL')
    need([e['step'] for e in response['epochs']] == list(range(start, 719, 24)), 'SUFFIX_EPOCH_COVERAGE')
    schedule = {e['step']: e for e in suffix['epochs']}
    carried = None
    for epoch in response['epochs']:
        closed(epoch, ('step', 'geometry_dawn', 'exact_memory_sha256', 'raw_actions', 'trace_sha256',
            'after_plan', 'after_memory_sha256', 'publication_sha256', 'compiler_rejected'), 'SUFFIX_EPOCH_FIELDS')
        dawn = epoch['geometry_dawn']; step = epoch['step']; expected = schedule.get(step)
        need(dawn['observation']['step'] == step and dawn['observation']['player'] == seat
             and dawn['memory']['controller'] is None, 'SUFFIX_GEOMETRY_CONTEXT')
        if carried is None:
            need(digest(C.v0_oracle.actor_input(dawn)) == req['start']['actor_context_sha256'],
                 'SUFFIX_INITIAL_PREFIX_DRIFT')
        else:
            need(wire(dawn['memory']['plan']) == wire(carried), 'SUFFIX_PREFIX_DRIFT')
        need(epoch['raw_actions'] == (expected['raw_actions'] if expected else [V.STOP_ID]), 'SUFFIX_ACTION_DRIFT')
        trace = C.v0_oracle.trace_edits(dawn, epoch['raw_actions'])
        need(trace['content_sha256'] == epoch['trace_sha256'] and
             wire(trace['plan']) == wire(epoch['after_plan']), 'SUFFIX_TRACE_DRIFT')
        if expected:
            need(wire(trace['plan']) == wire(expected['canonical_plan']), 'SUFFIX_PLAN_DRIFT')
        for key in ('exact_memory_sha256', 'after_memory_sha256', 'publication_sha256'):
            R._hash(epoch[key], '$.epoch.' + key)
        R._int(epoch['compiler_rejected'], '$.compiler_rejected', 0)
        carried = epoch['after_plan']
    events = response['rng_events']
    need(type(events) is list and [e['step'] for e in events] == list(range(start+24, 719, 24)), 'SUFFIX_RNG_COVERAGE')
    for event in events:
        closed(event, ('step', 'key', 'pre_state_sha256', 'post_state_sha256'), 'SUFFIX_RNG_FIELDS')
        R._int(event['key'], '$.rng.key', 0)
        if rng_seed is not None:
            need(event['key'] == (rng_seed * 1000003) ^ ((event['step'] - 1) // 24), 'SUFFIX_RNG_KEY_DRIFT')
        R._hash(event['pre_state_sha256'], '$.rng.pre'); R._hash(event['post_state_sha256'], '$.rng.post')
    t = response['terminal']
    closed(t, ('cash_by_seat', 'own_cash', 'rival_cash', 'margin', 'wdl', 'environment_sha256',
               'content_sha256'), 'SUFFIX_TERMINAL_FIELDS')
    need(wire(t) == wire(seal(t)), 'SUFFIX_TERMINAL_HASH')
    cash = t['cash_by_seat']
    need(type(cash) is list and len(cash) == 2, 'SUFFIX_TERMINAL_CASH')
    for value in cash:
        R._number(value, '$.cash')
    margin = cash[seat] - cash[1-seat]
    need(t['own_cash'] == cash[seat] and t['rival_cash'] == cash[1-seat] and t['margin'] == margin
         and type(t['wdl']) is int and t['wdl'] == (margin > 0) - (margin < 0), 'SUFFIX_TERMINAL_DERIVATION')
    R._hash(t['environment_sha256'], '$.terminal.environment')
    return t


def _run_suffix(branch, runner, root):
    argument = copy.deepcopy(branch); response = None
    try:
        response = runner(argument, evidence_root=root)
        need(wire(argument) == wire(branch), 'RUNNER_MUTATED_REQUEST')
        terminal = _validate_suffix_response(branch, response)
        status, error = ('COMPLETE', None) if terminal is not None else ('INCOMPLETE', 'INCOMPLETE_SUFFIX')
    except Exception as exc:
        terminal = None; status = 'INCOMPLETE' if response is None else 'INVALID'
        error = str(exc) if isinstance(exc, ArchiveError) else type(exc).__name__
        try:
            R._safe(response)
        except (ValueError, RecursionError):
            response = None
    return dict(arm=branch['arm'], attempt_id=branch['attempt_id'], phase=branch['phase'],
        branch_sha256=branch['content_sha256'], status=status, terminal=terminal, error=error, response=response)


def _suffix_signature(arm):
    return dict(status=arm['status'], terminal=arm['terminal'],
        response={k: v for k, v in (arm['response'] or {}).items() if k not in ('branch_sha256', 'content_sha256')})


def _suffix_replay(arms, fresh):
    mismatches = []
    for a, b in zip(arms, fresh):
        if a['status'] != 'COMPLETE' or b['status'] != 'COMPLETE':
            mismatches.append(dict(arm=a['arm'], reason='INCOMPLETE_OR_INVALID_BRANCH'))
        elif wire(_suffix_signature(a)) != wire(_suffix_signature(b)):
            mismatches.append(dict(arm=a['arm'], reason='SEMANTIC_REPLAY_MISMATCH'))
    return dict(status=('MISMATCH' if mismatches else 'VERIFIED') if fresh else 'NOT_RUN',
                arms=fresh, mismatches=mismatches)


def _assemble_suffix(request, execution, arms, replay):
    attempts = [*arms, *replay['arms']]; native = execution['kind'] == 'NATIVE_GAME'
    return seal(dict(schema=SUFFIX_SCHEMA, request_sha256=request['content_sha256'], execution=execution,
        arms=arms, replay=replay, comparison=E._comparison(arms), counts=dict(
            execution_attempts=len(attempts), native_starts=len(attempts) if native else 0,
            native_completed=sum(a['status'] == 'COMPLETE' for a in attempts) if native else 0,
            replay_executions=len(replay['arms']), independent_worlds=1, replay_independent_worlds_added=0,
            actor_training_examples=0, optimizer_updates=0)))


def evaluate_suffix_pair(request: dict, *, runner, evidence_root: str | Path = ROOT) -> dict:
    root = safe_path(evidence_root).resolve()
    _validate_suffix_request(request, evidence_root=root)
    execution = suffix_execution_binding(runner)
    arms = [_run_suffix(_suffix_branch(request, arm, execution, 'EVALUATE'), runner, root) for arm in E.ARMS]
    return _assemble_suffix(request, execution, arms, _suffix_replay(arms, []))


def verify_suffix_pair(request: dict, result: dict, *, runner, evidence_root: str | Path = ROOT) -> dict:
    root = safe_path(evidence_root).resolve()
    _validate_suffix_request(request, evidence_root=root)
    _checked_suffix_result(request, result, evidence_root=root)
    need(result['execution'] == suffix_execution_binding(runner), 'SUFFIX_REPLAY_RUNNER_DRIFT')
    need(result['replay']['status'] == 'NOT_RUN', 'SUFFIX_ALREADY_REPLAYED')
    fresh = [_run_suffix(_suffix_branch(request, arm, result['execution'], 'REPLAY'), runner, root) for arm in E.ARMS]
    return _assemble_suffix(request, result['execution'], result['arms'], _suffix_replay(result['arms'], fresh))


def _checked_suffix_result(request, result, *, evidence_root):
    hashed(result, ('schema', 'request_sha256', 'execution', 'arms', 'replay', 'comparison', 'counts'),
           SUFFIX_SCHEMA, 'SUFFIX_RESULT')
    need(result['request_sha256'] == request['content_sha256'], 'SUFFIX_RESULT_REQUEST')
    execution = result['execution']
    closed(execution, ('kind', 'runner', 'python', 'trust', 'fresh_process_per_native_attempt',
                      'content_sha256'), 'EXECUTION_FIELDS')
    need(execution == E.panel.bound(execution), 'EXECUTION_HASH_MISMATCH')
    need(execution['kind'] in ('NATIVE_GAME', 'SYNTHETIC_DIAGNOSTIC'), 'EXECUTION_KIND')
    if execution['kind'] == 'NATIVE_GAME':
        need(execution == suffix_execution_binding(_NATIVE_SUFFIX_RUNNER), 'NATIVE_SOURCE_MISMATCH')
    replay = result['replay']
    closed(replay, ('status', 'arms', 'mismatches'), 'REPLAY_FIELDS')
    need(type(replay['arms']) is list and len(replay['arms']) in (0, 2), 'REPLAY_ARMS')
    cut = read_artifact(request['snapshot'], root=evidence_root)
    rng_seed = C._unpack(cut['environment']['data'])['info']['seed']
    identities = {}
    for phase, arms in (('EVALUATE', result['arms']), ('REPLAY', replay['arms'])):
        if phase == 'REPLAY' and not arms:
            continue
        need(type(arms) is list and len(arms) == 2 and [a['arm'] for a in arms] == list(E.ARMS), 'PAIR_ARMS')
        for arm in arms:
            closed(arm, ('arm', 'attempt_id', 'phase', 'branch_sha256', 'status', 'terminal', 'error', 'response'), 'ARM_FIELDS')
            attempt_id = text(arm['attempt_id'], 'EXECUTION_ID')
            need(attempt_id not in identities and arm['phase'] == phase, 'REUSED_EXECUTION_ID_OR_PHASE')
            branch = _suffix_branch(request, arm['arm'], execution, phase, attempt_id)
            need(arm['branch_sha256'] == branch['content_sha256'], 'BRANCH_BINDING')
            need(arm['status'] in ('COMPLETE', 'INCOMPLETE', 'INVALID'), 'ARM_STATUS')
            if arm['status'] == 'COMPLETE':
                terminal = _validate_suffix_response(branch, arm['response'], rng_seed=rng_seed)
                need(terminal is not None and wire(arm['terminal']) == wire(terminal) and arm['error'] is None,
                     'SUFFIX_TERMINAL_MISMATCH')
            else:
                need(arm['terminal'] is None, 'CENSORED_OUTCOME')
            identities[attempt_id] = digest(arm)
    expected_replay = _suffix_replay(result['arms'], replay['arms'])
    need(wire(replay) == wire(expected_replay), 'SUFFIX_REPLAY_DERIVED_DRIFT')
    need(wire(result) == wire(_assemble_suffix(request, execution, result['arms'], expected_replay)),
         'SUFFIX_RESULT_DERIVED_DRIFT')
    return identities


def _suffix_facts(candidate, request, result, root):
    expected = candidate_from_suffix_request(request, request_ref=candidate['start_artifact'], evidence_root=root)
    need(wire(candidate) == wire(expected), 'EXACT_CANDIDATE_REQUEST_MISMATCH')
    identities = _checked_suffix_result(request, result, evidence_root=root)
    complete = all(a['status'] == 'COMPLETE' for a in result['arms'])
    return dict(adapter='CAPTURED_SUFFIX', native=result['execution']['kind'] == 'NATIVE_GAME',
        complete=complete, fresh_replay=result['replay']['status'] == 'VERIFIED',
        learner_wdl=result['arms'][0]['terminal']['wdl'] if complete else None,
        challenger_wdl=result['arms'][1]['terminal']['wdl'] if complete else None,
        wdl_delta=result['comparison']['wdl_delta'],
        primary_attempt_ids=[a['attempt_id'] for a in result['arms']],
        replay_attempt_ids=[a['attempt_id'] for a in result['replay']['arms']],
        execution_identities=identities, record_sha256=None,
        request_sha256=request['content_sha256'], result_sha256=result['content_sha256'])


def native_suffix_runner(branch: dict, *, evidence_root: str | Path = ROOT) -> dict:
    """Hard deadline; fresh interpreter per charged arm. No automatic retries."""
    need(os.environ.get('PYTHONHASHSEED') == '0', 'NATIVE_SUFFIX_REQUIRES_PYTHONHASHSEED_0')
    with tempfile.TemporaryDirectory(prefix='v0-exact-suffix-') as directory:
        folder = Path(directory); inp = folder / 'branch.json'; output = folder / 'response.json'
        inp.write_bytes(wire(branch) + b'\n')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
            PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), *filter(None, os.environ.get('PYTHONPATH', '').split(os.pathsep))]),
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [sys.executable, '-B', '-m', 'farm_neat.training.v0_search_archive', '_suffix-game',
            '--input', str(inp), '--output', str(output), '--evidence-root', str(safe_path(evidence_root).resolve())]
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)
        try:
            proc.communicate(timeout=branch['request']['timeout_seconds'] + 10.)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.communicate()
            raise TimeoutError('fresh suffix worker deadline; no retry')
        need(proc.returncode == 0, 'NATIVE_SUFFIX_WORKER_FAILED')
        return read_json(output)


_NATIVE_SUFFIX_RUNNER = native_suffix_runner


def _write_suffix_artifact(path, value):
    path = safe_path(path)
    data = wire(value) + b'\n'
    need(len(data) <= MAX_ARTIFACT_BYTES, 'ARTIFACT_TOO_LARGE')
    with path.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def execute_suffix_to_archive(archive, *, request_path: str | Path,
        output_path: str | Path, evaluation_id: str, runner) -> dict:
    """Reserve four attempts BEFORE execution; a restart never silently retries.

    This is the smallest maintainer gate: two arms plus two fresh replays from
    one already-retained cut. Capture generation and statistical admission are
    separate owners. Every run gets a new evaluation_id and immutable result path.
    """
    root = archive.evidence_root
    request = read_json(request_path)
    ref = artifact_ref(request_path, root=root)
    candidate = candidate_from_suffix_request(request, request_ref=ref, evidence_root=root)
    output = safe_path(output_path)
    need(output.parent.is_dir() and output.resolve().is_relative_to(root) and not output.exists(), 'OUTPUT_PATH')
    # append's idempotence is not execution authorization. Pending IDs are spent.
    need(all(r['evaluation_id'] != evaluation_id for r in archive.entries()), 'ATTEMPT_ALREADY_CHARGED')
    native = runner is _NATIVE_SUFFIX_RUNNER
    budget = dict(cpu_seconds=4 * (request['timeout_seconds'] + 10.),
                  native_starts=4 if native else 0, candidate_attempts=1)
    reservation = archive.append(attempt_event(evaluation_id, candidate, charged=budget))
    need(reservation['inserted'], 'ATTEMPT_ALREADY_CHARGED')
    try:
        primary = evaluate_suffix_pair(request, runner=runner, evidence_root=root)
        result = verify_suffix_pair(request, primary, runner=runner, evidence_root=root)
        _write_suffix_artifact(output, result)
        evidence = dict(request=ref, result=artifact_ref(output, root=root))
        complete = all(a['status'] == 'COMPLETE' for a in result['arms'])
        event = outcome_event(evaluation_id, candidate['content_sha256'], charged=budget,
            status='COMPLETE' if complete else 'CENSORED', evidence=evidence,
            reason_codes=[] if complete else ['INCOMPLETE_OR_INVALID_SUFFIX'])
        archive.append(event)
    except Exception as exc:
        # A hard process kill leaves the precommitted ATTEMPT pending instead.
        archive.append(outcome_event(evaluation_id, candidate['content_sha256'], charged=budget,
            status='FAILED', reason_codes=['SUFFIX_EXECUTION_FAILED', type(exc).__name__]))
        raise
    return seal(dict(schema=SUFFIX_SCHEMA + '-run', status='COMPLETE' if complete else 'CENSORED',
        evaluation_id=evaluation_id, result=artifact_ref(output, root=root),
        replay=result['replay']['status'], charged=budget, teacher_export='NOT_ADMITTED'))


_SQL = '''
CREATE TABLE events (
 sequence INTEGER PRIMARY KEY,
 evaluation_id TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('ATTEMPT','OUTCOME')),
 payload BLOB NOT NULL,
 facts BLOB NOT NULL,
 previous_sha256 TEXT NOT NULL,
 row_sha256 TEXT NOT NULL UNIQUE,
 UNIQUE(evaluation_id,kind));
CREATE TABLE executions (attempt_id TEXT PRIMARY KEY, identity_sha256 TEXT NOT NULL);
CREATE TRIGGER events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'append-only'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'append-only'); END;
CREATE TRIGGER executions_no_update BEFORE UPDATE ON executions BEGIN SELECT RAISE(ABORT,'append-only'); END;
CREATE TRIGGER executions_no_delete BEFORE DELETE ON executions BEGIN SELECT RAISE(ABORT,'append-only'); END;
'''


class Archive:
    """One local append-only journal. No connection or mutable rows escape the API."""

    def __init__(self, path: str | Path, *, evidence_root: str | Path = ROOT):
        self.path = safe_path(path)
        self.evidence_root = safe_path(evidence_root).resolve()
        need(self.evidence_root.is_dir(), 'EVIDENCE_ROOT')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not tables:
                    # execute individually: executescript implicitly commits an open transaction.
                    for statement in _SQL.strip().split(';\n'):
                        if statement.strip():
                            db.execute(statement.rstrip(';'))
                    db.execute('PRAGMA user_version=1')
                self._schema(db)
                db.commit()
            except BaseException:
                db.rollback(); raise
        # Durable creation of the directory entry, not just database pages.
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @contextmanager
    def _connect(self):
        safe_path(self.path)
        db = sqlite3.connect(self.path, timeout=30., isolation_level=None)
        try:
            db.execute('PRAGMA synchronous=FULL')
            yield db
        finally:
            db.close()

    @staticmethod
    def _schema(db):
        need(db.execute('PRAGMA user_version').fetchone()[0] == 1, 'ARCHIVE_VERSION')
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        need(tables == {'events', 'executions'}, 'ARCHIVE_TABLES')
        triggers = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        need(triggers == {'events_no_update', 'events_no_delete',
                          'executions_no_update', 'executions_no_delete'}, 'ARCHIVE_TRIGGERS')
        # Names alone would accept a replaced no-op trigger. Compare the exact
        # schema emitted above, ignoring only formatting and terminal semicolons.
        normalize = lambda sql: ' '.join(sql.strip().rstrip(';').split())
        expected = {}
        for statement in _SQL.strip().split(';\n'):
            tokens = statement.split()
            expected[(tokens[1].lower(), tokens[2])] = normalize(statement)
        actual = {(kind, name): normalize(sql) for kind, name, sql in db.execute(
            "SELECT type,name,sql FROM sqlite_master WHERE type IN ('table','trigger')")}
        need(actual == expected, 'ARCHIVE_SCHEMA_DEFINITION')

    @staticmethod
    def _check_charge(facts, event):
        if facts and facts['native']:
            need(event['charged']['native_starts'] >= len(facts['execution_identities']),
                 'NATIVE_START_UNDERCHARGE')

    def _read_start(self, candidate):
        value = read_artifact(candidate['start_artifact'], root=self.evidence_root)
        if value.get('schema') == SUFFIX_REQUEST_SCHEMA:
            read_artifact(value['snapshot'], root=self.evidence_root)
        return value

    def _facts(self, candidate, event):
        if event['evidence'] is None:
            return None
        refs = event['evidence']
        need(refs['request']['sha256'] == candidate['start_artifact']['sha256'] and
             refs['request']['bytes'] == candidate['start_artifact']['bytes'], 'START_ARTIFACT_MISMATCH')
        request = read_artifact(refs['request'], root=self.evidence_root)
        result = read_artifact(refs['result'], root=self.evidence_root)
        facts = (_suffix_facts(candidate, request, result, self.evidence_root)
                 if request.get('schema') == SUFFIX_REQUEST_SCHEMA else _paired_facts(candidate, request, result))
        need(event['status'] != 'COMPLETE' or facts['complete'], 'INCOMPLETE_RESULT')
        return facts

    def append(self, event: dict) -> dict:
        """Atomic, idempotent canonical bytes; caller commits ATTEMPT before work."""
        _event(event)
        event = copy.deepcopy(event)
        payload = wire(event)
        # Validate external evidence before taking the writer lock. It is rehashed
        # on later verification/selection, so byte drift cannot silently promote.
        facts = None
        if event['kind'] == 'ATTEMPT':
            self._read_start(event['candidate'])
        else:
            with self._connect() as db:
                self._schema(db)
                old = db.execute('SELECT sequence,payload,row_sha256 FROM events WHERE evaluation_id=? AND kind=?',
                                 (event['evaluation_id'], event['kind'])).fetchone()
                if old is not None:
                    need(bytes(old[1]) == payload, 'CONFLICTING_EVENT')
                    return dict(sequence=old[0], row_sha256=old[2], inserted=False)
                parent = db.execute("SELECT payload FROM events WHERE evaluation_id=? AND kind='ATTEMPT'",
                                    (event['evaluation_id'],)).fetchone()
            need(parent is not None, 'MISSING_ATTEMPT')
            attempt = R._loads(bytes(parent[0]).decode()); _event(attempt)
            candidate = attempt['candidate']
            need(event['candidate_sha256'] == candidate['content_sha256'], 'CANDIDATE_MISMATCH')
            facts = self._facts(candidate, event)
            self._check_charge(facts, event)
            for ref in event['witnesses']:
                _artifact_bytes(ref, root=self.evidence_root)
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                self._schema(db)
                old = db.execute('SELECT sequence,payload,row_sha256 FROM events WHERE evaluation_id=? AND kind=?',
                                 (event['evaluation_id'], event['kind'])).fetchone()
                if old is not None:
                    need(bytes(old[1]) == payload, 'CONFLICTING_EVENT')
                    db.rollback()
                    return dict(sequence=old[0], row_sha256=old[2], inserted=False)
                if event['kind'] == 'OUTCOME':
                    parent = db.execute("SELECT payload FROM events WHERE evaluation_id=? AND kind='ATTEMPT'",
                                        (event['evaluation_id'],)).fetchone()
                    need(parent is not None, 'MISSING_ATTEMPT')
                    attempt = R._loads(bytes(parent[0]).decode())
                    candidate = attempt['candidate']
                    need(event['candidate_sha256'] == candidate['content_sha256'], 'CANDIDATE_MISMATCH')
                    need(all(event['charged'][k] >= attempt['charged'][k] for k in CHARGES), 'BUDGET_DECREASE')
                    if event['duplicate_of'] is not None:
                        target = db.execute("SELECT payload FROM events WHERE evaluation_id=? AND kind='ATTEMPT'",
                                            (event['duplicate_of'],)).fetchone()
                        need(target is not None and event['duplicate_of'] != event['evaluation_id'], 'DUPLICATE_TARGET')
                        other = R._loads(bytes(target[0]).decode())['candidate']
                        need(wire(other) == wire(candidate), 'DUPLICATE_TARGET_MISMATCH')
                    # Parent/event identities are rechecked under the writer lock;
                    # expensive evidence validation ran against the immutable parent.
                    if facts:
                        for attempt_id, identity in facts['execution_identities'].items():
                            old_id = db.execute('SELECT identity_sha256 FROM executions WHERE attempt_id=?',
                                                (attempt_id,)).fetchone()
                            need(old_id is None or old_id[0] == identity, 'CONFLICTING_EXECUTION_ID')
                            if old_id is None:
                                db.execute('INSERT INTO executions VALUES (?,?)', (attempt_id, identity))
                last = db.execute('SELECT sequence,row_sha256 FROM events ORDER BY sequence DESC LIMIT 1').fetchone()
                sequence, previous = (last[0] + 1, last[1]) if last else (0, ZERO)
                row_hash = digest(dict(sequence=sequence, previous_sha256=previous, event=event, facts=facts))
                db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)',
                    (sequence, event['evaluation_id'], event['kind'], payload, wire(facts), previous, row_hash))
                db.commit()
            except BaseException:
                db.rollback(); raise
        return dict(sequence=sequence, row_sha256=row_hash, inserted=True)

    def _read(self, *, verify_evidence: bool = True):
        with self._connect() as db:
            db.execute('BEGIN')
            self._schema(db)
            need(db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'SQLITE_INTEGRITY')
            rows = list(db.execute('SELECT sequence,evaluation_id,kind,payload,facts,previous_sha256,row_sha256 '
                                   'FROM events ORDER BY sequence'))
            stored_executions = dict(db.execute('SELECT attempt_id,identity_sha256 FROM executions'))
            db.commit()
        previous, attempts, executions = ZERO, {}, {}
        for n, (sequence, evaluation_id, kind, payload, raw_facts, prev, row_hash) in enumerate(rows):
            event = R._loads(bytes(payload).decode()); _event(event)
            facts = R._loads(bytes(raw_facts).decode())
            need(sequence == n and prev == previous and evaluation_id == event['evaluation_id']
                 and kind == event['kind'] and bytes(payload) == wire(event) and bytes(raw_facts) == wire(facts),
                 'JOURNAL_CHAIN_OR_BYTES')
            need(row_hash == digest(dict(sequence=n, previous_sha256=prev, event=event, facts=facts)),
                 'JOURNAL_HASH')
            previous = row_hash
            if kind == 'ATTEMPT':
                need(evaluation_id not in attempts and facts is None, 'DUPLICATE_ATTEMPT')
                if verify_evidence:
                    self._read_start(event['candidate'])
                attempts[evaluation_id] = dict(evaluation_id=evaluation_id, sequence=n,
                    candidate=event['candidate'], charged=event['charged'], status='CENSORED_PENDING',
                    outcome=None, facts=None)
                continue
            need(evaluation_id in attempts and attempts[evaluation_id]['outcome'] is None, 'OUTCOME_ORDER')
            row = attempts[evaluation_id]
            need(event['candidate_sha256'] == row['candidate']['content_sha256'], 'CANDIDATE_MISMATCH')
            need(all(event['charged'][k] >= row['charged'][k] for k in CHARGES), 'BUDGET_DECREASE')
            if event['duplicate_of'] is not None:
                target = attempts.get(event['duplicate_of'])
                need(target is not None and target is not row and
                     wire(target['candidate']) == wire(row['candidate']), 'DUPLICATE_TARGET')
            if verify_evidence:
                need(wire(self._facts(row['candidate'], event)) == wire(facts), 'STORED_FACTS_MISMATCH')
                for ref in event['witnesses']:
                    _artifact_bytes(ref, root=self.evidence_root)
            self._check_charge(facts, event)
            if facts:
                for attempt_id, identity in facts['execution_identities'].items():
                    need(attempt_id not in executions or executions[attempt_id] == identity, 'CONFLICTING_EXECUTION_ID')
                    executions[attempt_id] = identity
            row.update(charged=event['charged'], status=event['status'], outcome=event,
                       facts=facts, outcome_sequence=n)
        need(executions == stored_executions, 'EXECUTION_INDEX_MISMATCH')
        return list(attempts.values()), len(rows), previous

    def entries(self, *, verify_evidence: bool = True) -> list[dict]:
        """Detached trainer-only rows, including every failure and pending charge."""
        return self._read(verify_evidence=verify_evidence)[0]

    def verify(self) -> dict:
        rows, count, head = self._read()
        counts = {}
        for row in rows:
            counts[row['status']] = counts.get(row['status'], 0) + 1
        return seal(dict(schema=SCHEMA + '-verification', status='VERIFIED', event_count=count,
            attempt_count=len(rows), status_counts=counts,
            charged={k: math.fsum(r['charged'][k] for r in rows) if k == 'cpu_seconds'
                     else sum(r['charged'][k] for r in rows) for k in CHARGES},
            head_sha256=head, limitations=list(LIMITS)))

    @staticmethod
    def _eligible(row):
        f = row['facts']; c = row['candidate']
        return (row['status'] == 'COMPLETE' and f is not None and f['native'] and
                f['complete'] and f['fresh_replay'] and c['suffix']['kind'] == 'V0')

    def best(self, *, exact_state_key: str | None = None, paired_comparison_key: str | None = None) -> dict:
        """Strict improvements over the frozen control; ties retain that control.

        No result means keep the declared control, not select a losing proposal.
        No cash/value tie-break or cross-cell join is performed.
        """
        rows, _, head = self._read()
        winners, controls = {}, {}
        current_pins = source_pins()
        for row in sorted(rows, key=lambda r: r.get('outcome_sequence', r['sequence'])):
            c = row['candidate']; skey, ckey = state_key(c), comparison_key(c)
            if exact_state_key is not None and skey != exact_state_key:
                continue
            if paired_comparison_key is not None and ckey != paired_comparison_key:
                continue
            if not self._eligible(row):
                continue
            need(c['source_pins'] == current_pins, 'STALE_SOURCE')
            key = (skey, ckey)
            control_wdl = row['facts']['learner_wdl']
            need(key not in controls or controls[key] == control_wdl, 'CONTROL_WDL_CONFLICT')
            controls[key] = control_wdl
            if row['facts']['wdl_delta'] <= 0:
                continue
            incumbent = winners.get(key)
            # Identical control path and exact start: terminal WDL maximizers and
            # paired deltas have the same order. Never compare margin or predictions.
            if incumbent is None or row['facts']['wdl_delta'] > incumbent['facts']['wdl_delta']:
                winners[key] = row
        result = [dict(state_key=k[0], comparison_key=k[1], evaluation_id=row['evaluation_id'],
                       candidate=copy.deepcopy(row['candidate']), facts=copy.deepcopy(row['facts']))
                  for k, row in sorted(winners.items())]
        return seal(dict(schema=SCHEMA + '-best', objective='TERMINAL_WDL',
                         fallback='KEEP_DECLARED_CONTROL', head_sha256=head,
                         best=result, limitations=list(LIMITS)))

    def export_teachers(self, bundle: dict, *, protocol: dict) -> dict:
        """Delegate admission; require exact, complete archive custody for all cells.

        The admission receipt and weight remain trainer-only siblings. Do not
        replace each seat's comparison statistic with the clustered gate statistic.
        This is export only: no model, loss, optimizer, or new world execution.
        """
        decision = admission.admit_pair(bundle, protocol=protocol)
        out = dict(schema=SCHEMA + '-teacher-export', status='REFUSED', admission=decision,
                   teachers=[], reason_codes=[], limitations=list(LIMITS))
        if not decision['actor_training_admissible'] or decision['teacher_weight'] <= 0:
            out['reason_codes'] = ['ADMISSION_OWNER_REFUSED']
            return seal(out)
        records = [bundle] if bundle.get('schema') == R.SCHEMA else bundle['records']
        records = list({p['content_sha256']: p for p in records}.values())
        rows, _, head = self._read()
        eligible = [r for r in rows if self._eligible(r) and
                    r['facts'].get('adapter', 'OPENING_PAIR') == 'OPENING_PAIR']
        current_pins = source_pins()
        used_ids = set()
        teachers = []
        for p in records:
            p = R.validate_pair(p)
            identity = p['identity']; ev = p['evaluation']
            worlds = {w['source_group_id']: w for phase in admission.PHASES for w in ev[phase]}
            custody = []
            for cell in ev['seat_rows']:
                world = worlds[cell['source_group_id']]
                expected_start = dict(snapshot_sha256=world['snapshot_sha256'],
                    continuation_sha256=identity['continuation_sha256'],
                    actor_context_sha256=identity['actor_context_sha256'], world_sha256=world['world_sha256'],
                    rng_sha256=world['rng_sha256'], opponent_sha256=ev['opponent_hash'],
                    source_group_id=cell['source_group_id'], seat=cell['seat'], step=0)
                pair_ids = [cell['learner_attempt_id'], cell['challenger_attempt_id']]
                matches = []
                for row in eligible:
                    c, f = row['candidate'], row['facts']
                    if (c['start'] == expected_start and c['source_pins'] == current_pins and
                        c['policy_sha256'] == identity['parent_checkpoint_sha256'] and
                        c['comparison']['control_path_sha256'] == p['learner']['path_sha256'] and
                        c['comparison']['control_suffix_sha256'] == digest(_opening_suffix(p, 'learner')) and
                        c['comparison']['protocol_sha256'] == identity['protocol_sha256'] and
                        c['comparison']['round_id'] == identity['round_id'] and
                        c['comparison']['split'] in ('discovery', 'confirmation', 'training') and
                        c['suffix'] == _opening_suffix(p, 'challenger') and
                        f['primary_attempt_ids'] == pair_ids and
                        [f['learner_wdl'], f['challenger_wdl']] == [cell['learner_wdl'], cell['challenger_wdl']]):
                        matches.append(row)
                if not matches:
                    out['reason_codes'] = ['MISSING_EXACT_FRESH_ARCHIVE_CUSTODY']
                    return seal(out)
                row = min(matches, key=lambda r: r['outcome_sequence'])
                all_ids = row['facts']['primary_attempt_ids'] + row['facts']['replay_attempt_ids']
                if len(all_ids) != 4 or len(set(all_ids)) != 4 or used_ids.intersection(all_ids):
                    out['reason_codes'] = ['TEACHER_EXECUTION_ID_REUSE']
                    return seal(out)
                used_ids.update(all_ids)
                custody.append(dict(source_group_id=cell['source_group_id'],
                                    evaluation_id=row['evaluation_id'],
                                    evidence=copy.deepcopy(row['outcome']['evidence'])))
            teachers.append(dict(record=p, teacher_weight=decision['teacher_weight'],
                                 admission_sha256=decision['content_sha256'], exact_custody=custody))
        out.update(status='EXPORTED', teachers=teachers, archive_head_sha256=head)
        return seal(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('append', 'verify', 'best', 'export-teachers'):
        p = commands.add_parser(name)
        p.add_argument('--archive', required=True)
        p.add_argument('--evidence-root', default=str(ROOT))
        if name in ('append', 'export-teachers'):
            p.add_argument('--input', required=True)
        if name == 'best':
            p.add_argument('--state-key'); p.add_argument('--comparison-key')
        if name == 'export-teachers':
            p.add_argument('--protocol', required=True)
    p = commands.add_parser('execute-suffix')
    p.add_argument('--archive', required=True)
    p.add_argument('--evidence-root', default=str(ROOT))
    p.add_argument('--request', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--evaluation-id', required=True)
    p.add_argument('--native', action='store_true', help='Explicit maintainer-only four-start authorization')
    p = commands.add_parser('_suffix-game')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--evidence-root', default=str(ROOT))
    args = parser.parse_args(argv)
    try:
        if args.command == '_suffix-game':
            branch = read_json(args.input)
            need(branch['execution'] == suffix_execution_binding(_NATIVE_SUFFIX_RUNNER), 'NATIVE_SOURCE_MISMATCH')
            _write_suffix_artifact(args.output, _suffix_game(branch, evidence_root=args.evidence_root))
            return 0
        if args.command == 'execute-suffix':
            need(args.native, 'EXPLICIT_NATIVE_SUFFIX_AUTHORIZATION_REQUIRED')
            need(os.environ.get('PYTHONHASHSEED') == '0', 'NATIVE_SUFFIX_REQUIRES_PYTHONHASHSEED_0')
        if args.command not in ('append', 'execute-suffix'):
            need(safe_path(args.archive).is_file(), 'MISSING_ARCHIVE')
        archive = Archive(args.archive, evidence_root=args.evidence_root)
        if args.command == 'execute-suffix':
            result = execute_suffix_to_archive(archive, request_path=args.request, output_path=args.output,
                evaluation_id=args.evaluation_id, runner=_NATIVE_SUFFIX_RUNNER)
        elif args.command == 'append':
            result = archive.append(read_json(args.input, max_bytes=MAX_EVENT_BYTES))
        elif args.command == 'verify':
            result = archive.verify()
        elif args.command == 'best':
            result = archive.best(exact_state_key=args.state_key, paired_comparison_key=args.comparison_key)
        else:
            result = archive.export_teachers(read_json(args.input), protocol=read_json(args.protocol))
        print(wire(result).decode())
        return 0 if result.get('status') != 'REFUSED' else 2
    except (ArchiveError, R.RecordError, E.PairError, sqlite3.Error, OSError, ValueError,
            TypeError, KeyError, IndexError, OverflowError, RecursionError) as exc:
        print(wire(dict(schema=SCHEMA + '-refusal', status='REFUSED',
                        reason=getattr(exc, 'code', str(exc) if isinstance(exc, ArchiveError) else type(exc).__name__))).decode())
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
