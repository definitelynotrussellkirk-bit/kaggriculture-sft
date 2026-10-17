"""Terminal sequence-Q corpus: exact replay custody, actor-only finite menus.

No Bellman bootstrap, invented behavior likelihood, cash reward or teacher export.
Native collection uses the existing four-start (pair + fresh pair) archive owner.
All public writes are exclusive; an interrupted charged attempt is never retried.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.projection import State
from farm_neat.evaluation.v0_continuation import V0Continuation
from farm_neat.policy.v0_scorer import Prefix
from farm_neat.training import v0_regret_records as R
from farm_neat.training import v0_search_archive as A
from farm_neat.training.teachers import v0_regret_proposals as P
from farm_neat.training.teachers import v0_regret_searcher as S

ROOT = A.ROOT
SCHEMA = 'farm-v0-terminal-sequence-q-v1'
TAIL = 'KEEP_TO_TERMINAL'
TRAIN_SPLITS = ('discovery', 'training')
SPLITS = (*TRAIN_SPLITS, 'confirmation', 'monitor', 'final')
SOURCE_PATHS = (
    'src/farm_neat/training/v0_sequence_q_data.py',
    'src/farm_neat/training/v0_sequence_q.py',
    'src/farm_neat/training/v0_sequence_q_select.py',
    'src/farm_neat/policy/v0_scorer.py',
    'src/farm_neat/training/teachers/v0_regret_proposals.py',
    'src/farm_neat/training/teachers/v0_regret_searcher.py',
)


class QDataError(ValueError):
    """Machine-readable refusal; never a censored game's made-up outcome."""


def need(ok: bool, code: str) -> None:
    if not ok:
        raise QDataError(code)


# Preserve ordered continuation mappings; archive's own hashes remain its business.
def wire(value: Any) -> str:
    R._safe(value)
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':'))


def digest(value: Any) -> str:
    return hashlib.sha256(wire(value).encode()).hexdigest()


def seal(value: dict) -> dict:
    result = copy.deepcopy({k: v for k, v in value.items() if k != 'content_sha256'})
    return {**result, 'content_sha256': digest(result)}


def check(value: dict, schema: str, fields: tuple[str, ...]) -> None:
    need(type(value) is dict and set(value) == {'schema', 'content_sha256', *fields}, 'CLOSED_FIELDS')
    need(value['schema'] == schema and wire(value) == wire(seal(value)), 'HASH_OR_SCHEMA')


def source_pins() -> dict:
    return {**A.source_pins(), **{p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                for p in SOURCE_PATHS if (ROOT / p).is_file()}}


def read_json(path: str | Path) -> dict:
    return A.read_json(path)


def write_new(path: str | Path, value: dict) -> dict:
    # A snapshot owns its sha256; Q records own content_sha256. Never re-seal
    # foreign schemas or reorder insertion-sensitive continuation mappings.
    path = A.safe_path(path)
    payload = (wire(value) + '\n').encode('utf-8')
    need(len(payload) <= 32 * 1024 * 1024, 'ARTIFACT_SIZE_CAP')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return value


def actor_context(observation: dict, agent) -> dict:
    """Projection only. Full controller/policy/RNG stays in the exact cut.

    Allocation/started/HOLD order is retained. No projected context is a restore
    artifact, teacher pair, or state-sufficiency claim.
    """
    state = State(observation)
    facts = json.loads(wire({k: getattr(state, k) for k in sorted(R.OBS_FIELDS)}))
    memory = V0Continuation.capture(agent).to_dict()
    memory.pop('game_id'); memory['controller'] = None
    context = dict(schema=R.ACTOR_SCHEMA, observation=facts,
                   prefix=Prefix(agent.plan).payload(), compiler_memory=memory)
    R._actor(context)
    return context


def candidate(context: dict, actions: list[int], family: str = 'supplied', *, cache=None) -> dict:
    state, prefix, options = R._actor(context)
    found, reasons = S._candidate(context, dict(proposal_family=family, raw_actions=actions),
                                 state, prefix, options, {} if cache is None else cache,
                                 edit_limit=V.MAX_EDITS_PER_EPOCH)
    need(found is not None, 'ILLEGAL_SEQUENCE:' + ','.join(reasons))
    return {k: copy.deepcopy(found[k]) for k in ('candidate_id', 'raw_actions',
                                                'canonical_plan', 'proposal_family')}


def propose_menu(context: dict, incumbent_actions: list[int], *, seed: int,
                 max_candidates: int = 16, max_path_edits: int = 4,
                 supplied: list[dict] | None = None) -> dict:
    """Hard attempt cap, including duplicates/invalid proposals; never retry-until-valid.

    Reuses native proposal geometry. Optional supplied paths are explicitly
    labelled supplied, not fabricated samples from a missing policy checkpoint.
    """
    R._int(seed, '$.seed', 0, 2**63 - 1)
    R._int(max_candidates, '$.max_candidates', 1, 32)
    R._int(max_path_edits, '$.max_path_edits', 1, 16)
    state, prefix, options = R._actor(context)
    need(not prefix.terminated, 'AFTER_STOP')
    cache = {}; incumbent = candidate(context, incumbent_actions, 'incumbent', cache=cache)
    candidates = [incumbent]; seen = {incumbent['candidate_id']}
    attempts = [dict(family='incumbent', status='RETAINED', candidate_id=incumbent['candidate_id'])]
    supplied = [] if supplied is None else supplied
    need(type(supplied) is list and len(supplied) <= 32, 'SUPPLIED_CAP')
    for spec in supplied:
        R._fields(spec, ('raw_actions',), '$.supplied', optional=('proposal_family',))
        need(spec.get('proposal_family') not in ('current_policy', 'policy_sample', 'learner'),
             'POLICY_POOL_RECEIPT_REQUIRED')
    rng = random.Random(seed)
    for i in range(1, max_candidates):
        if i <= len(supplied):
            spec = supplied[i - 1]
            R._fields(spec, ('raw_actions',), '$.supplied', optional=('proposal_family',))
            actions, family = spec['raw_actions'], spec.get('proposal_family', 'supplied')
        elif i % 2:
            family = 'random_legal_control'
            actions = P._random_path(state, prefix, options, cache, rng, max_path_edits)
        else:
            family = 'incumbent_mutation'
            actions, _ = P._mutate(incumbent_actions, rng, (i - 2) // 2)
        try:
            need(type(actions) is list and len(actions) - 1 <= max_path_edits, 'PATH_BUDGET')
            c = candidate(context, actions, family, cache=cache)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            attempts.append(dict(family=family, status='ILLEGAL', reason=str(exc), raw_actions=actions))
            continue
        if c['candidate_id'] in seen:
            attempts.append(dict(family=family, status='DUPLICATE', candidate_id=c['candidate_id']))
        else:
            seen.add(c['candidate_id']); candidates.append(c)
            attempts.append(dict(family=family, status='RETAINED', candidate_id=c['candidate_id']))
    return seal(dict(schema=SCHEMA + '-menu', actor_context=copy.deepcopy(context),
                     incumbent_id=incumbent['candidate_id'], candidates=candidates,
                     attempted=len(attempts), attempts=attempts, seed=seed, tail=TAIL))


def menu_from_pool(pool: dict, *, incumbent_actions: list[int] | None = None) -> dict:
    """Import a source-bound unselected pool, preserving its real direct-draw receipt.

    The receipt's geometry/likelihood joins are checked by their existing owner;
    this does not independently attest a checkpoint execution. No new draws or
    games are made, and the current-policy candidate is never invented.
    """
    need(type(pool) is dict and pool.get('schema') == P.SCHEMA
         and pool.get('status') == 'UNSELECTED_POOL', 'POLICY_POOL_SCHEMA')
    body = {k: v for k, v in pool.items() if k != 'pool_sha256'}
    need(pool.get('pool_sha256') == R.canonical_sha256(body), 'POLICY_POOL_HASH')
    expected = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in P.SOURCE_PATHS}
    need(pool['source_pins'] == R.source_pins() and pool['generator_source_pins'] == expected,
         'POLICY_POOL_SOURCE_DRIFT')
    context = pool['actor_context']; state, prefix, options = R._actor(context)
    need(pool['actor_context_sha256'] == R.canonical_sha256(context), 'POLICY_ACTOR_JOIN')
    cache = {}
    R._proposal(pool['learner_draw'], {'parent_checkpoint_sha256': pool['parent_checkpoint_sha256']},
                prefix, state, options, cache, direct=True)
    direct = candidate(context, pool['learner_draw']['raw_actions'], 'current_policy', cache=cache)
    incumbent = candidate(context, [0] if incumbent_actions is None else incumbent_actions,
                          'incumbent', cache=cache)
    candidates = []
    for original in pool['candidates']:
        family = 'current_policy' if original['candidate_id'] == direct['candidate_id'] else original['proposal_family']
        c = candidate(context, original['raw_actions'], family, cache=cache)
        need(c['candidate_id'] == original['candidate_id']
             and c['canonical_plan'] == original['canonical_plan'], 'POLICY_CANDIDATE_JOIN')
        candidates.append(c)
    ids = {c['candidate_id'] for c in candidates}
    need(direct['candidate_id'] in ids and incumbent['candidate_id'] in ids, 'POLICY_OR_INCUMBENT_MISSING')
    attempted = pool['counts']['proposed']
    need(type(attempted) is int and attempted == len(pool['attempts']), 'POLICY_ATTEMPT_JOIN')
    result = seal(dict(schema=SCHEMA + '-menu', actor_context=copy.deepcopy(context),
        incumbent_id=incumbent['candidate_id'], candidates=candidates, attempted=attempted,
        attempts=copy.deepcopy(pool['attempts']), seed=pool['seed'], tail=TAIL))
    return validate_menu(result)


def validate_menu(menu: dict) -> dict:
    check(menu, SCHEMA + '-menu', ('actor_context', 'incumbent_id', 'candidates',
                                   'attempted', 'attempts', 'seed', 'tail'))
    need(menu['tail'] == TAIL, 'CONTINUATION_DRIFT')
    R._int(menu['seed'], '$.seed', 0, 2**63 - 1)
    need(type(menu['candidates']) is list and 1 <= len(menu['candidates']) <= 32, 'MENU_CAP')
    need(type(menu['attempted']) is int and len(menu['candidates']) <= menu['attempted'] <= 32
         and type(menu['attempts']) is list and len(menu['attempts']) == menu['attempted'], 'ATTEMPT_COUNT')
    cache = {}; ids = set()
    for c in menu['candidates']:
        R._fields(c, ('candidate_id', 'raw_actions', 'canonical_plan', 'proposal_family'), '$.candidate')
        expected = candidate(menu['actor_context'], c['raw_actions'], c['proposal_family'], cache=cache)
        need(wire(expected) == wire(c), 'CANDIDATE_GEOMETRY_OR_IDENTITY')
        need(c['candidate_id'] not in ids, 'DUPLICATE_CANDIDATE')
        ids.add(c['candidate_id'])
    need(menu['incumbent_id'] in ids, 'MISSING_INCUMBENT')
    return menu


def suffix(c: dict, step: int) -> dict:
    return dict(kind='V0', epochs=[dict(step=step, raw_actions=c['raw_actions'],
                                       canonical_plan=c['canonical_plan'])], tail=TAIL)


def physical_world_id(runtime) -> str:
    """Cluster cuts/seats of one exogenous world; never encode current actor/RNG state.

    Configuration and declared scenario include the initial seed/schedule/rules.
    Full mutable environment/actor custody remains in start/snapshot hashes.
    """
    env = runtime.environment
    need(type(env.info.get('seed')) is int, 'PHYSICAL_WORLD_SEED_REQUIRED')
    return digest(dict(kind=type(env).__name__, configuration=A.C._pack(env.configuration),
                       seed=env.info['seed'], scenario_identity=getattr(env, 'scenario_identity', None)))


def prepare(snapshot: str | Path, output: str | Path, *, evidence_root: str | Path,
            source_group_id: str, family_id: str, split: str = 'training', seed: int = 0,
            max_candidates: int = 16, max_path_edits: int = 4,
            incumbent_actions: list[int] | None = None, supplied: list[dict] | None = None,
            timeout_seconds: float = 120., proposal_pool: dict | None = None) -> dict:
    """Freeze one menu and all requests before any suffix evaluation."""
    root, out = A.safe_path(evidence_root).resolve(), A.safe_path(output)
    need(split in SPLITS, 'SPLIT')
    A.text(family_id, 'FAMILY_REQUIRED'); A.text(source_group_id, 'GROUP_REQUIRED')
    need(out.resolve().is_relative_to(root) and not out.exists(), 'NEW_OUTPUT_REQUIRED')
    ref = A.artifact_ref(snapshot, root=root)
    start, runtime = A._captured_start(ref, evidence_root=root, source_group_id=source_group_id)
    context = actor_context(runtime.environment.agent_observation(runtime.seat), runtime.actors[runtime.seat])
    if proposal_pool is None:
        menu = propose_menu(context, [0] if incumbent_actions is None else incumbent_actions,
                            seed=seed, max_candidates=max_candidates, max_path_edits=max_path_edits, supplied=supplied)
    else:
        need(supplied is None, 'POOL_AND_SUPPLIED_ARE_EXCLUSIVE')
        menu = menu_from_pool(proposal_pool, incumbent_actions=incumbent_actions)
        need(wire(menu['actor_context']) == wire(context), 'POLICY_POOL_CUT_JOIN')
        # An imported frozen pool owns its original generation budget and seed.
        seed = menu['seed']; max_candidates = proposal_pool['budget']['max_candidates']
        max_path_edits = proposal_pool['budget']['max_path_edits']
    control = next(c for c in menu['candidates'] if c['candidate_id'] == menu['incumbent_id'])
    protocol = digest(dict(schema=SCHEMA, tail=TAIL, seed=seed, max_candidates=max_candidates,
                           max_path_edits=max_path_edits))
    policy = digest(dict(kind='FROZEN_INCUMBENT_PATH_NOT_MODEL_CHECKPOINT', actions=control['raw_actions']))
    trainer = dict(snapshot=ref, start=start, family_id=family_id, split=split,
                   world_id=physical_world_id(runtime), source_group_id=source_group_id,
                   policy_sha256=policy, protocol_sha256=protocol)
    requests = []
    challengers = [c for c in menu['candidates'] if c != control] or [control]
    # Validate every request before creating an output directory.
    values = [A.make_suffix_request(snapshot_ref=ref, control=suffix(control, start['step']),
              challenger=suffix(c, start['step']), source_group_id=source_group_id,
              policy_sha256=policy, protocol_sha256=protocol, split=split,
              round_id='terminal-sequence-q', proposal_family=c['proposal_family'],
              timeout_seconds=timeout_seconds, evidence_root=root) for c in challengers]
    out.mkdir(parents=True, exist_ok=False)
    for i, (c, request) in enumerate(zip(challengers, values)):
        path = out / f'request-{i:03d}.json'; write_new(path, request)
        requests.append(dict(candidate_id=c['candidate_id'], evaluation_id='q-' + request['content_sha256'],
            request=A.artifact_ref(path, root=root), result_path=str((out / f'result-{i:03d}.json').relative_to(root))))
    manifest = seal(dict(schema=SCHEMA + '-prepared', menu=menu, trainer=trainer, requests=requests,
                         source_pins=source_pins(), required_native_starts=4 * len(requests),
                         proposal_pool=copy.deepcopy(proposal_pool)))
    write_new(out / 'PREPARED.json', manifest)
    return manifest


def validate_prepared(value: dict, *, evidence_root: str | Path) -> dict:
    check(value, SCHEMA + '-prepared', ('menu', 'trainer', 'requests', 'source_pins', 'required_native_starts', 'proposal_pool'))
    need(value['source_pins'] == source_pins(), 'SOURCE_DRIFT')
    menu = validate_menu(value['menu']); t = value['trainer']; root = A.safe_path(evidence_root).resolve()
    R._fields(t, ('snapshot', 'start', 'family_id', 'split', 'world_id', 'source_group_id',
                 'policy_sha256', 'protocol_sha256'), '$.trainer')
    need(t['split'] in SPLITS, 'SPLIT'); A.text(t['family_id'], 'FAMILY_REQUIRED')
    start, runtime = A._captured_start(t['snapshot'], evidence_root=root, source_group_id=t['source_group_id'])
    need(wire(start) == wire(t['start']) and physical_world_id(runtime) == t['world_id'], 'CUT_JOIN')
    context = actor_context(runtime.environment.agent_observation(runtime.seat), runtime.actors[runtime.seat])
    need(wire(context) == wire(menu['actor_context']), 'ACTOR_CUT_JOIN')
    by_id = {c['candidate_id']: c for c in menu['candidates']}
    if value['proposal_pool'] is None:
        need(not any(c['proposal_family'] in ('current_policy', 'policy_sample', 'learner')
                     for c in menu['candidates']), 'POLICY_POOL_RECEIPT_REQUIRED')
    else:
        imported = menu_from_pool(value['proposal_pool'],
                                  incumbent_actions=by_id[menu['incumbent_id']]['raw_actions'])
        need(wire(imported) == wire(menu), 'POLICY_POOL_MENU_JOIN')
    expected_ids = set(by_id) - {menu['incumbent_id']} or {menu['incumbent_id']}
    need(type(value['requests']) is list and len(value['requests']) == len(expected_ids)
         and {r['candidate_id'] for r in value['requests']} == expected_ids, 'REQUEST_MENU_JOIN')
    need(value['required_native_starts'] == 4 * len(value['requests']), 'START_COUNT')
    for r in value['requests']:
        R._fields(r, ('candidate_id', 'evaluation_id', 'request', 'result_path'), '$.request')
        req = A.read_artifact(r['request'], root=root)
        A._validate_suffix_request(req, evidence_root=root)
        need(req['start'] == start and req['snapshot'] == t['snapshot'] and req['split'] == t['split']
             and req['policy_sha256'] == t['policy_sha256'] and req['protocol_sha256'] == t['protocol_sha256'], 'REQUEST_CUT_JOIN')
        need(req['control'] == suffix(by_id[menu['incumbent_id']], start['step'])
             and req['challenger'] == suffix(by_id[r['candidate_id']], start['step']), 'REQUEST_ACTION_JOIN')
        need(r['evaluation_id'] == 'q-' + req['content_sha256'], 'EVALUATION_ID_JOIN')
        result = A.safe_path(root / r['result_path'])
        need(result.resolve().is_relative_to(root), 'RESULT_ESCAPE')
    return value


def execute(prepared: dict, *, archive_path: str | Path, evidence_root: str | Path,
            max_native_starts: int, native: bool = False, runner=None) -> dict:
    """Resume skips every charged ID (including failed/pending); no hidden retry.

    max_native_starts is a cumulative cap for the archive, not a fresh per-call
    allowance. Primary and replay starts are included. Capture is a separate job.
    """
    need(type(max_native_starts) is int and max_native_starts >= 0, 'START_BUDGET')
    root = A.safe_path(evidence_root).resolve()
    validate_prepared(prepared, evidence_root=root)
    runner = A.native_suffix_runner if runner is None else runner
    is_native = runner is A.native_suffix_runner
    need(not is_native or native, 'EXPLICIT_NATIVE_AUTHORIZATION_REQUIRED')
    need(not is_native or os.environ.get('PYTHONHASHSEED') == '0', 'PYTHONHASHSEED_0_REQUIRED')
    archive = A.Archive(archive_path, evidence_root=root)
    spent = {r['evaluation_id'] for r in archive.entries()}
    required = 4 * sum(r['evaluation_id'] not in spent for r in prepared['requests']) if is_native else 0
    need(archive.verify()['charged']['native_starts'] + required <= max_native_starts, 'NATIVE_START_BUDGET_EXCEEDED')
    completed, skipped = [], []
    for row in prepared['requests']:
        if row['evaluation_id'] in spent:
            skipped.append(row['evaluation_id']); continue
        completed.append(A.execute_suffix_to_archive(archive,
            request_path=root / row['request']['path'], output_path=root / row['result_path'],
            evaluation_id=row['evaluation_id'], runner=runner))
    return seal(dict(schema=SCHEMA + '-execution', completed=completed, skipped_charged=skipped,
                     archive=archive.verify(), counts_include_capture=False))


def build_row(prepared: dict, *, evidence_root: str | Path) -> dict:
    """Reopen raw custody and derive labels; missing/incomplete/replay-mismatch is null."""
    root = A.safe_path(evidence_root).resolve()
    validate_prepared(prepared, evidence_root=root)
    menu = prepared['menu']; outcomes = {c['candidate_id']: None for c in menu['candidates']}
    status = {k: 'NOT_RUN' for k in outcomes}; evidence = []; kinds = set(); starts = 0
    for r in prepared['requests']:
        path = root / r['result_path']
        if not path.exists():
            evidence.append(dict(candidate_id=r['candidate_id'], status='NOT_RUN')); continue
        req = A.read_artifact(r['request'], root=root); result = read_json(path)
        A._checked_suffix_result(req, result, evidence_root=root)
        kinds.add(result['execution']['kind']); starts += result['counts']['native_starts']
        verified = result['replay']['status'] == 'VERIFIED'
        for arm, key in zip(result['arms'], (menu['incumbent_id'], r['candidate_id'])):
            st = 'COMPLETE_VERIFIED' if verified and arm['status'] == 'COMPLETE' else 'CENSORED'
            if st == 'COMPLETE_VERIFIED':
                z = arm['terminal']['wdl']
                need(type(z) is int and z in (-1, 0, 1), 'TERMINAL_WDL_REQUIRED')
                need(outcomes[key] is None or outcomes[key] == z, 'CONTROL_OUTCOME_CONFLICT')
                outcomes[key] = z; status[key] = st
            elif outcomes[key] is None:
                status[key] = st
        evidence.append(dict(candidate_id=r['candidate_id'], status='VERIFIED' if verified else 'CENSORED',
                             request=r['request'], result=A.artifact_ref(path, root=root)))
    kind = 'NATIVE_GAME' if kinds == {'NATIVE_GAME'} else 'SYNTHETIC_DIAGNOSTIC'
    t = prepared['trainer']
    return seal(dict(schema=SCHEMA + '-row', menu=menu, labels=outcomes, statuses=status,
        provenance=dict(kind=kind, world_id=t['world_id'], family_id=t['family_id'], split=t['split'],
                        prepared=prepared, evidence=evidence, native_starts=starts)))


def diagnostic_row(menu: dict, labels: list[int], *, world_id: str, family_id: str | None = None,
                   split: str = 'training') -> dict:
    """Explicit diagnostic fixture constructor, impossible to upgrade by a flag."""
    validate_menu(menu)
    need(type(labels) is list and len(labels) == len(menu['candidates'])
         and all(type(z) is int and z in (-1, 0, 1) for z in labels), 'TERMINAL_WDL_REQUIRED')
    need(split in SPLITS, 'SPLIT')
    values = {c['candidate_id']: z for c, z in zip(menu['candidates'], labels)}
    return seal(dict(schema=SCHEMA + '-row', menu=copy.deepcopy(menu), labels=values,
        statuses={k: 'DIAGNOSTIC' for k in values}, provenance=dict(kind='SYNTHETIC_DIAGNOSTIC',
        world_id=world_id, family_id=family_id or world_id, split=split,
        prepared=None, evidence=[], native_starts=0)))


def validate_row(row: dict, *, evidence_root: str | Path | None = None) -> dict:
    check(row, SCHEMA + '-row', ('menu', 'labels', 'statuses', 'provenance'))
    validate_menu(row['menu']); p = row['provenance']
    R._fields(p, ('kind', 'world_id', 'family_id', 'split', 'prepared', 'evidence', 'native_starts'), '$.provenance')
    need(p['split'] in SPLITS and p['kind'] in ('SYNTHETIC_DIAGNOSTIC', 'NATIVE_GAME'), 'ROW_SCOPE')
    A.text(p['world_id'], 'WORLD_REQUIRED'); A.text(p['family_id'], 'FAMILY_REQUIRED')
    need(type(p['native_starts']) is int and p['native_starts'] >= 0 and type(p['evidence']) is list,
         'ROW_PROVENANCE')
    need(all(st in ('NOT_RUN', 'COMPLETE_VERIFIED', 'CENSORED', 'DIAGNOSTIC')
             for st in row['statuses'].values()), 'ROW_STATUS')
    keys = {c['candidate_id'] for c in row['menu']['candidates']}
    need(type(row['labels']) is dict and set(row['labels']) == keys
         and type(row['statuses']) is dict and set(row['statuses']) == keys, 'LABEL_MENU_ALIGNMENT')
    for k, z in row['labels'].items():
        need(z is None or type(z) is int and z in (-1, 0, 1), 'TERMINAL_WDL_REQUIRED')
        need((z is not None) == (row['statuses'][k] in ('COMPLETE_VERIFIED', 'DIAGNOSTIC')), 'CENSORED_LABEL')
    if p['kind'] == 'NATIVE_GAME':
        need(evidence_root is not None and p['prepared'] is not None, 'NATIVE_CUSTODY_REQUIRED')
        need(wire(row) == wire(build_row(p['prepared'], evidence_root=evidence_root)), 'NATIVE_LABEL_OR_EVIDENCE_DRIFT')
    return row


def validate_splits(rows: list[dict]) -> None:
    """Seats/cuts/replays are clustered by physical world and declared source family."""
    seen = {'world_id': {}, 'family_id': {}}
    for row in rows:
        p = row['provenance']; split = p['split']
        need(split in SPLITS, 'SPLIT')
        # Discovery + training are one development pool; none may overlap holdout.
        bucket = 'development' if split in TRAIN_SPLITS else split
        for key, groups in seen.items():
            val = p[key]
            need(val not in groups or groups[val] == bucket, 'SPLIT_OVERLAP:' + key)
            groups[val] = bucket


def corpus_stats(rows: list[dict]) -> dict:
    worlds = {r['provenance']['world_id'] for r in rows}
    contrasting = {r['provenance']['world_id'] for r in rows
                   if len({z for z in r['labels'].values() if z is not None}) >= 2}
    labels = [z for r in rows for z in r['labels'].values()]
    return dict(menus=len(rows), independent_worlds=len(worlds), contrasting_worlds=len(contrasting),
        loss_rows=labels.count(-1), draw_rows=labels.count(0), win_rows=labels.count(1),
        censored_rows=sum(z is None for z in labels), labelled_candidates=sum(z is not None for z in labels))


def training_gate(rows: list[dict], *, diagnostic: bool, min_contrasting_worlds: int = 32,
                  evidence_root: str | Path | None = None) -> dict:
    need(type(diagnostic) is bool and type(min_contrasting_worlds) is int and min_contrasting_worlds >= 1, 'GATE_CONFIG')
    need(type(rows) is list and bool(rows), 'EMPTY_CORPUS')
    validate_splits(rows)
    for row in rows:
        need(row['provenance']['split'] in TRAIN_SPLITS, 'PROTECTED_ROWS_IN_TRAINING')
        if not diagnostic:
            need(row['provenance']['kind'] == 'NATIVE_GAME', 'NATIVE_CORPUS_REQUIRED')
        validate_row(row, evidence_root=evidence_root)
    stats = corpus_stats(rows)
    need(stats['labelled_candidates'] > 0, 'NO_TERMINAL_LABELS')
    if not diagnostic:
        need(stats['contrasting_worlds'] >= min_contrasting_worlds, 'INSUFFICIENT_WDL_CONTRAST')
    return stats


def capture_native(*, output: str | Path, step: int, seed: int, scenario: str = 'router-dairy',
                   seat: int = 0, opponent: str = 'router', native: bool = False,
                   timeout_seconds: float = 120., opening_actions: list[int] | None = None) -> dict:
    """Actually advance the bundled engine to a reached BEFORE_BOTH_ACTORS cut."""
    need(native and os.environ.get('PYTHONHASHSEED') == '0', 'EXPLICIT_NATIVE_AND_HASHSEED_REQUIRED')
    need(type(step) is int and step in range(0, 719, 24) and type(seat) is int and seat in (0, 1), 'CUT_CLOCK_OR_SEAT')
    need(type(seed) is int and 0 <= seed < 2**63, 'SEED')
    need(opponent in ('keep', 'router'), 'RESPONSIVE_OPPONENT_REQUIRED')
    need(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds) and 0 < timeout_seconds <= 3600, 'DEADLINE')
    out = A.safe_path(output); need(not out.exists(), 'NEW_OUTPUT_REQUIRED')
    from farm_neat.strategy import counterfactual as C
    from farm_neat.evaluation.scenarios import ScenarioEnvironment, catalog
    from farm_neat.economy.v0.epochs import V0Agent, KeepPolicy
    from farm_neat.evaluation.shop_router import ShopRouterAdapter
    actions = [0] if opening_actions is None else opening_actions
    C.v0_oracle._sequence(actions)
    spec = copy.deepcopy(catalog()[scenario]); spec['seed'] = seed
    receipt_path = out.with_name(out.stem + '-capture-attempt.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    write_new(receipt_path, seal(dict(schema=SCHEMA + '-capture-attempt', seed=seed, scenario=scenario,
        seat=seat, requested_step=step, native_starts_charged=1, opponent=opponent,
        source_pins=source_pins(), opening_actions=actions, status='CHARGED_BEFORE_EXECUTION')))
    began = time.monotonic(); env = ScenarioEnvironment(spec)
    own = V0Agent(C.TrajectoryPolicy({0: actions[:-1]}), game_id='q-capture-own', labor_factor=4.0, hands_mode='daily', belief={p: 0 for p in V.PRODUCTS})
    other = ShopRouterAdapter() if opponent == 'router' else V0Agent(KeepPolicy(), game_id='q-capture-rival', labor_factor=4.0, hands_mode='daily', belief={p: 0 for p in V.PRODUCTS})
    actors = [None, None]; actors[seat] = own; actors[1 - seat] = other
    joint_hash = hashlib.sha256()
    while env.step_index < step:
        need(time.monotonic() - began <= timeout_seconds, 'CAPTURE_TIMEOUT')
        joint = [C._act(actors[s], env.agent_observation(s), env.configuration) for s in (0, 1)]
        joint_hash.update(wire(joint).encode()); env.step(joint)
    snapshot = C.capture(env, actors, seat=seat).to_dict()
    write_new(out, snapshot)
    receipt = seal(dict(schema=SCHEMA + '-capture-result', status='REACHED', step=step,
        physical_prefix_starts=1, native_steps=step, completed_games=0,
        elapsed_seconds=time.monotonic() - began, snapshot_sha256=snapshot['sha256'],
        joint_actions_sha256=joint_hash.hexdigest(), source_attempt=str(receipt_path)))
    write_new(out.with_name(out.stem + '-capture-result.json'), receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('capture')
    p.add_argument('--output', required=True); p.add_argument('--step', type=int, required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--scenario', default='router-dairy')
    p.add_argument('--seat', type=int, choices=(0, 1), default=0)
    p.add_argument('--opponent', choices=('keep', 'router'), default='router')
    p.add_argument('--native', action='store_true'); p.add_argument('--opening-actions')
    p = commands.add_parser('prepare')
    p.add_argument('--snapshot', required=True); p.add_argument('--output', required=True)
    p.add_argument('--source-group-id', required=True); p.add_argument('--family-id', required=True)
    p.add_argument('--split', choices=SPLITS, default='training'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-candidates', type=int, default=16); p.add_argument('--max-path-edits', type=int, default=4)
    p.add_argument('--incumbent-actions'); p.add_argument('--supplied'); p.add_argument('--pool')
    p.add_argument('--evidence-root', default=str(ROOT))
    p = commands.add_parser('execute')
    p.add_argument('--prepared', required=True); p.add_argument('--archive', required=True)
    p.add_argument('--max-native-starts', type=int, required=True); p.add_argument('--native', action='store_true')
    p.add_argument('--evidence-root', default=str(ROOT))
    p = commands.add_parser('corpus')
    p.add_argument('--prepared', nargs='+', required=True); p.add_argument('--output', required=True)
    p.add_argument('--evidence-root', default=str(ROOT))
    args = parser.parse_args(argv)
    try:
        if args.command == 'capture':
            result = capture_native(output=args.output, step=args.step, seed=args.seed, scenario=args.scenario,
                seat=args.seat, opponent=args.opponent, native=args.native,
                opening_actions=None if args.opening_actions is None else read_json(args.opening_actions)['raw_actions'])
        elif args.command == 'prepare':
            result = prepare(args.snapshot, args.output, evidence_root=args.evidence_root,
                source_group_id=args.source_group_id, family_id=args.family_id, split=args.split,
                seed=args.seed, max_candidates=args.max_candidates, max_path_edits=args.max_path_edits,
                incumbent_actions=None if args.incumbent_actions is None else read_json(args.incumbent_actions)['raw_actions'],
                supplied=None if args.supplied is None else read_json(args.supplied)['proposals'],
                proposal_pool=None if args.pool is None else read_json(args.pool))
        elif args.command == 'execute':
            result = execute(read_json(args.prepared), archive_path=args.archive, evidence_root=args.evidence_root,
                             max_native_starts=args.max_native_starts, native=args.native)
        else:
            rows = [build_row(read_json(p), evidence_root=args.evidence_root) for p in args.prepared]
            validate_splits(rows)
            result = seal(dict(schema=SCHEMA + '-corpus', rows=rows, stats=corpus_stats(rows), source_pins=source_pins()))
            write_new(args.output, result)
        print(wire(result)); return 0
    except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
        print(wire(dict(status='REFUSED', reason=str(exc), schema=SCHEMA))); return 2


if __name__ == '__main__':
    raise SystemExit(main())
