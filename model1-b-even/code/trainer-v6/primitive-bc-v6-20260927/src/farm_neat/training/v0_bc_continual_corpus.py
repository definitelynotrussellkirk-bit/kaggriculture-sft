"""Versioned append-only V0 behavior-cloning corpus assembly.

Command surface::

    PYTHONPATH=src python3 -B -m farm_neat.training.v0_bc_continual_corpus reserve \
        --input RESERVATIONS.json --output NEW_EXPOSURE.json

    PYTHONPATH=src python3 -B -m farm_neat.training.v0_bc_continual_corpus assemble \
        --source SOURCE.json [--source SOURCE2.json ...] --output NEW_DIR \
        [--registry EXPOSURE.json --registry-sha256 SHA] \
        [--parent OLD_DIR --parent-manifest-sha256 SHA] \
        [--conflicts reject|source-separate] [--base-minimum PROD=8 ...]

    PYTHONPATH=src python3 -B -m farm_neat.training.v0_bc_continual_corpus audit \
        --corpus DIR --manifest-sha256 SHA

Owns only corpus composition and lifetime exposure accounting. It consumes already
replay-capable V0 prefix rows and never changes action IDs, legality, compiler or
BC loss. Three source kinds are explicit: verified base teachers, searched/stronger
teachers, and own-prefix corrections. Every source row keeps world/seat/opponent,
checkpoint/cycle/plan-family ancestry plus enough row identity/trainer state to
materialize the exact original replay row.

Deduplication is deliberately narrow: within one corpus partition, rows collapse
only when the exact actor context/menu and exact successor class agree. Per-source
memberships remain attached, so dedup never erases ancestry. Conflicting labels at
the same actor context are either rejected or retained in SEPARATED.jsonl and kept
out of the uniform-BC rows. Conflicts across distinct partitions are reported as
partition-separated, never merged.

EXPOSURE.json is the lifetime registry. It binds nested physical-world,
(world, opponent family+digest), and (world, opponent family+digest, plan family)
groups. A physical world can never cross train/dev/protected_eval partitions; the
finer groups retain opponent/checkpoint-family novelty without weakening that split. Protected-evaluation groups must be reserved
with ``reserve`` before their first protected row is assembled. A later cycle sees
an existing registry key as known, so an inspected group cannot become "novel" by
renaming a cycle or source.

``sampling_weight`` is metadata only. The declared objective remains uniform
successor-class BC; this module never multiplies a loss by corpus sampling weight.
Terminal outcomes, correction effects and search traces belong only in each source
entry's ``trainer_only`` object, which is never part of actor_context().
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from functools import lru_cache
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.training import v0_corpus as C
from farm_neat.training import v0_corpus_admit as A
from farm_neat.training import v0_corpus_worlds as W
from farm_neat.training import v0_targets as T

SCHEMA = 'farm-v0-bc-continual-corpus-v1'
RECORD_SCHEMA = 'farm-v0-bc-continual-record-v1'
SEPARATED_SCHEMA = 'farm-v0-bc-continual-separated-v1'
SOURCE_SCHEMA = 'farm-v0-bc-continual-source-v1'
REGISTRY_SCHEMA = 'farm-v0-bc-exposure-registry-v1'
RESERVATION_SCHEMA = 'farm-v0-bc-protected-reservations-v1'
CONFLICT_SCHEMA = 'farm-v0-bc-continual-conflicts-v1'
COVERAGE_SCHEMA = 'farm-v0-bc-continual-coverage-v1'
OBJECTIVE = 'UNCHANGED_UNIFORM_SUCCESSOR_CLASS_BC; SAMPLING_WEIGHT_METADATA_ONLY'

BASE = 'verified_base_teacher'
SEARCH = 'search_teacher'
CORRECTION = 'own_prefix_correction'
SOURCE_KINDS = (BASE, SEARCH, CORRECTION)
PARTITIONS = ('train', 'dev', 'protected_eval')
CONFLICT_POLICIES = ('reject', 'source-separate')
ACTION_FAMILIES = ('PROD', 'SERVICE', 'LAND', 'MARKET', 'STOP')
ROOT = T.ROOT


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return T._file_sha(path)


def _write_json(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(_json(value) + '\n')


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(_json(row) + '\n')


def _read(path: Path) -> Any:
    return C._read(path)


def _safe_text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,191}', value) is not None,
             f'invalid {field}')
    return value


def _sha(value: Any, field: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None, f'{field} sha256 required')
    return value


def source_pins() -> dict[str, str]:
    paths = [
        Path(__file__),
        ROOT / 'src/farm_neat/training/v0_corpus.py',
        ROOT / 'src/farm_neat/training/v0_corpus_admit.py',
        ROOT / 'src/farm_neat/training/v0_corpus_worlds.py',
        ROOT / 'src/farm_neat/training/v0_targets.py',
        ROOT / 'src/farm_neat/economy/v0/plan.py',
        ROOT / 'src/farm_neat/policy/v0_scorer.py',
        ROOT / 'src/farm_neat/policy/v0_menu_fast.py',
    ]
    return {str(p.relative_to(ROOT)): _file_sha(p) for p in paths}


def _expected_candidates(row: dict) -> dict:
    """Recompute one prefix menu from actor-visible state plus declared own memory."""
    actor = row['actor']
    _require(set(actor) == {'public_observation', 'own_memory', 'prefix_plan', 'last_key', 'nonstop_edit_count'},
             'closed actor fields required')
    plan = Plan.from_canonical(actor['prefix_plan'])
    if row['schema'] == T.SCHEMA:
        from farm_neat.training.v0_bc import actor_state
        _require(actor['own_memory'] == {}, 'opening row cannot carry continuation memory')
        state = actor_state(actor['public_observation'])
        return T.prefix_support(state, plan, actor['last_key'], actor['nonstop_edit_count'])
    if row['schema'] == A.ROW_SCHEMA:
        from farm_neat.policy import v0_menu_fast as F
        inputs = A.scorer_inputs(row)
        menu = F.build_menu(inputs['state'], inputs['prefix'], backend='fast',
                            projection_kwargs={**T.SETTINGS, **inputs['memory']})
        return {'legal_ids': list(menu.legal_ids), 'phi': [list(p) for p in menu.phi],
                'classes': [W.class_payload(c) for c in menu.classes],
                'forced_stop': inputs['prefix'].edit_count == V.MAX_EDITS_PER_EPOCH,
                'phi_names': list(W.PHI_NAMES)}
    raise ValueError('unsupported V0 prefix row schema: ' + str(row.get('schema')))


def validate_row(row: dict) -> dict:
    """Fail closed on stale support, unreachable target class or trainer->actor bleed.

    This is a prefix replay check, not a native-world origin certificate. Producers
    remain responsible for their stronger admission gates. The assembler refuses to
    repair a row and preserves the producer replay identity unchanged. Repeated exact
    rows share only this pure validation result; changed bytes always miss the cache.
    """
    _require(isinstance(row, dict), 'V0 prefix row object required')
    return copy.deepcopy(_validate_row_cached(_json(row)))


@lru_cache(maxsize=512)
def _validate_row_cached(encoded: str) -> dict:
    row = json.loads(encoded)
    _require(set(row) == {'schema', 'identity', 'actor', 'candidates', 'target', 'trainer_only'},
             'closed V0 prefix row fields required')
    _require(row['schema'] in (T.SCHEMA, A.ROW_SCHEMA), 'accepted V0 prefix row schema required')
    identity = row['identity']
    _require(isinstance(identity, dict), 'row identity required')
    for field in ('physical_world_id', 'trajectory_id'):
        _safe_text(identity.get(field), 'row ' + field)
    _require(type(identity.get('learner_seat')) is int and identity['learner_seat'] in (0, 1), 'row learner seat required')
    _require(type(identity.get('dawn')) is int and 0 <= identity['dawn'] < V.SEASON_DAYS, 'row dawn required')
    _require(type(identity.get('prefix_index')) is int and identity['prefix_index'] >= 0, 'row prefix index required')
    _require(isinstance(row['trainer_only'], dict), 'row trainer_only object required')

    expected = _expected_candidates(row)
    _require(_digest(expected) == _digest(row['candidates']), 'stale or changed prefix support/menu')
    label = C._label(row)
    target = row['target']
    selected = next(c for c in row['candidates']['classes'] if tuple(c['indices']) == label)
    terminated = selected['continuation']['terminated']
    forced = row['candidates']['forced_stop']
    expected_stop = ('forced_cap' if forced else 'sampled') if terminated else 'not_stop'
    _require(target['stop_kind'] == expected_stop, 'target STOP/cap semantics mismatch')
    _require(target['successor_plan_sha256'] == target['continuation']['plan_sha256'], 'target Plan digest mismatch')
    indices = target['class_indices']
    _require(indices and all(type(i) is int and 0 <= i < len(row['candidates']['legal_ids']) for i in indices),
             'target class indices outside current prefix menu')
    families = sorted({V.decode(row['candidates']['legal_ids'][i])[0] for i in indices})
    _require(set(families) <= set(ACTION_FAMILIES), 'unknown V0 economic action family')
    actor_payload = {'actor': row['actor'], 'candidates': row['candidates']}
    class_payload = {k: copy.deepcopy(target[k]) for k in ('class_indices', 'successor_plan_sha256', 'continuation')}
    return {'actor_context_sha256': _digest(actor_payload), 'target_class_sha256': _digest(class_payload),
            'actor_payload': actor_payload, 'target_class': class_payload, 'economic_families': families,
            'loss_weight': target['loss_weight'], 'stop_kind': target['stop_kind']}


def actor_context(value: dict) -> dict:
    """Return only data allowed to determine the V0 actor/scorer input and menu."""
    if value.get('schema') == RECORD_SCHEMA:
        return {'actor': copy.deepcopy(value['actor']), 'candidates': copy.deepcopy(value['candidates'])}
    checked = validate_row(value)
    return copy.deepcopy(checked['actor_payload'])


def _row_parts(row: dict) -> tuple[dict, dict, dict]:
    return copy.deepcopy(row['actor']), copy.deepcopy(row['candidates']), copy.deepcopy(row['target'])


def _replay_identity(row: dict, checked: dict) -> dict:
    trainer = row['trainer_only']
    return {'row_sha256': _digest(row), 'row_schema': row['schema'],
            'trajectory_id': row['identity']['trajectory_id'], 'prefix_index': row['identity']['prefix_index'],
            'source_request_sha256': trainer.get('source_request_sha256'),
            'continuation_sha256': None if 'continuation' not in trainer else _digest(trainer['continuation']),
            'actor_context_sha256': checked['actor_context_sha256'],
            'target_class_sha256': checked['target_class_sha256']}


def materialize_row(record: dict, membership: dict | None = None) -> dict:
    """Recreate one exact producer row from the deduplicated common example."""
    _require(record.get('schema') == RECORD_SCHEMA and record['memberships'], 'continual record required')
    membership = record['memberships'][0] if membership is None else membership
    row = {'schema': membership['row_schema'], 'identity': copy.deepcopy(membership['row_identity']),
           'actor': copy.deepcopy(record['actor']), 'candidates': copy.deepcopy(record['candidates']),
           'target': copy.deepcopy(record['target']), 'trainer_only': copy.deepcopy(membership['row_trainer_only'])}
    _require(_digest(row) == membership['replay_identity']['row_sha256'], 'membership no longer materializes producer row')
    validate_row(row)
    return row


def uniform_bc_rows(records: Iterable[dict], *, partition: str = 'train') -> list[dict]:
    """One deduplicated row per actor-context/class; sampling weights are ignored."""
    _require(partition in PARTITIONS, 'known corpus partition required')
    result = []
    for record in sorted((r for r in records if r['partition'] == partition),
                         key=lambda r: (r['actor_context_sha256'], r['target_class_sha256'])):
        result.append(materialize_row(record))
    return result


def empty_registry() -> dict:
    return {'schema': REGISTRY_SCHEMA, 'world_groups': {}, 'pair_groups': {}, 'source_groups': {}}


def _group_payload(ancestry: dict, *, level: str) -> dict:
    _require(level in ('world', 'pair', 'source'), 'known exposure group level required')
    value = {'physical_world_id': ancestry['physical_world_id']}
    if level in ('pair', 'source'):
        value.update({'opponent_family': ancestry['opponent_family'], 'opponent_digest': ancestry['opponent_digest']})
    if level == 'source':
        value['plan_family'] = ancestry['plan_family']
    return value


def _group_specs():
    return (('world_groups', 'world'), ('pair_groups', 'pair'), ('source_groups', 'source'))


def _validate_registry(registry: dict) -> dict:
    _require(isinstance(registry, dict) and set(registry) == {'schema', 'world_groups', 'pair_groups', 'source_groups'} and
             registry['schema'] == REGISTRY_SCHEMA, 'exposure registry schema required')
    for name, level in _group_specs():
        _require(isinstance(registry[name], dict), 'registry groups must be maps')
        for key, row in registry[name].items():
            _sha(key, 'registry group')
            payload_fields = set(_group_payload(row, level=level))
            expected_keys = payload_fields | {'partition', 'first_cycle_id', 'seen_cycle_ids', 'source_ids', 'reserved', 'inspected'}
            _require(isinstance(row, dict) and set(row) == expected_keys, 'closed registry group fields required')
            _safe_text(row['physical_world_id'], 'physical_world_id'); _safe_text(row['first_cycle_id'], 'first_cycle_id')
            if level in ('pair', 'source'):
                _safe_text(row['opponent_family'], 'opponent_family'); _sha(row['opponent_digest'], 'opponent_digest')
            if level == 'source':
                _safe_text(row['plan_family'], 'plan_family')
            _require(row['partition'] in PARTITIONS, 'registry partition required')
            _require(type(row['reserved']) is bool and type(row['inspected']) is bool, 'registry booleans required')
            _require(isinstance(row['seen_cycle_ids'], list) and row['seen_cycle_ids'] == sorted(set(row['seen_cycle_ids'])) and
                     all(isinstance(x, str) and x for x in row['seen_cycle_ids']), 'registry cycle list required')
            _require(isinstance(row['source_ids'], list) and row['source_ids'] == sorted(set(row['source_ids'])) and
                     all(isinstance(x, str) and x for x in row['source_ids']), 'registry source list required')
            payload = _group_payload(row, level=level)
            _require(_digest(payload) == key, 'registry group digest mismatch')
    return copy.deepcopy(registry)


def _touch(registry: dict, ancestry: dict, partition: str, *, source_id: str | None,
           reserved: bool, inspected: bool) -> tuple[bool, bool, bool]:
    novel = []
    for name, level in _group_specs():
        payload = _group_payload(ancestry, level=level)
        key = _digest(payload)
        current = registry[name].get(key)
        novel.append(current is None)
        if current is None:
            current = {**payload, 'partition': partition, 'first_cycle_id': ancestry['cycle_id'],
                       'seen_cycle_ids': [], 'source_ids': [], 'reserved': False, 'inspected': False}
            registry[name][key] = current
        else:
            _require(current['partition'] == partition,
                     f'exposure partition leakage for {name[:-1]} {key}: {current["partition"]} -> {partition}')
        current['seen_cycle_ids'] = sorted(set(current['seen_cycle_ids']) | {ancestry['cycle_id']})
        if source_id is not None:
            current['source_ids'] = sorted(set(current['source_ids']) | {source_id})
        current['reserved'] = bool(current['reserved'] or reserved)
        current['inspected'] = bool(current['inspected'] or inspected)
    return novel[0], novel[1], novel[2]


def _reservation_group(value: dict, cycle_id: str) -> dict:
    _require(isinstance(value, dict) and set(value) == {'physical_world_id', 'opponent_family', 'opponent_digest', 'plan_family'},
             'closed protected reservation group fields required')
    result = copy.deepcopy(value) | {'cycle_id': cycle_id}
    _safe_text(result['physical_world_id'], 'physical_world_id'); _safe_text(result['opponent_family'], 'opponent_family')
    _sha(result['opponent_digest'], 'opponent_digest'); _safe_text(result['plan_family'], 'plan_family')
    return result


def reserve_registry(registry: dict | None, reservation: dict) -> dict:
    """Return a new registry value; never mutates or overwrites its input file."""
    result = empty_registry() if registry is None else _validate_registry(registry)
    _require(isinstance(reservation, dict) and set(reservation) == {'schema', 'reservation_id', 'cycle_id', 'groups'} and
             reservation['schema'] == RESERVATION_SCHEMA, 'protected reservation document schema required')
    _safe_text(reservation['reservation_id'], 'reservation_id'); cycle = _safe_text(reservation['cycle_id'], 'cycle_id')
    _require(isinstance(reservation['groups'], list) and reservation['groups'], 'nonempty protected groups required')
    for raw in sorted(reservation['groups'], key=_json):
        ancestry = _reservation_group(raw, cycle)
        _touch(result, ancestry, 'protected_eval', source_id=None, reserved=True, inspected=False)
    return _validate_registry(result)


def _registry_extends(base: dict, candidate: dict) -> None:
    base, candidate = _validate_registry(base), _validate_registry(candidate)
    for name, _ in _group_specs():
        for key, old in base[name].items():
            _require(key in candidate[name], 'lifetime exposure registry dropped a prior group')
            new = candidate[name][key]
            fixed = set(old) - {'seen_cycle_ids', 'source_ids', 'reserved', 'inspected'}
            _require(all(new[k] == old[k] for k in fixed), 'lifetime exposure registry rewrote prior group identity')
            _require(set(old['seen_cycle_ids']) <= set(new['seen_cycle_ids']) and set(old['source_ids']) <= set(new['source_ids']),
                     'lifetime exposure registry forgot prior exposure')
            _require((not old['reserved'] or new['reserved']) and (not old['inspected'] or new['inspected']),
                     'lifetime exposure registry cleared prior state')


def _validate_ancestry(value: dict, row: dict) -> dict:
    fields = {'physical_world_id', 'learner_seat', 'opponent_family', 'opponent_digest', 'checkpoint_digest',
              'cycle_id', 'plan_family'}
    _require(isinstance(value, dict) and set(value) == fields, 'closed ancestry fields required')
    result = copy.deepcopy(value)
    _safe_text(result['physical_world_id'], 'physical_world_id'); _safe_text(result['opponent_family'], 'opponent_family')
    _sha(result['opponent_digest'], 'opponent_digest'); _sha(result['checkpoint_digest'], 'checkpoint_digest')
    _safe_text(result['cycle_id'], 'cycle_id'); _safe_text(result['plan_family'], 'plan_family')
    _require(type(result['learner_seat']) is int and result['learner_seat'] in (0, 1), 'learner seat required')
    _require(result['physical_world_id'] == row['identity']['physical_world_id'], 'ancestry/world differs from producer row')
    _require(result['learner_seat'] == row['identity']['learner_seat'], 'ancestry/seat differs from producer row')
    return result


def validate_source(document: dict) -> dict:
    _require(isinstance(document, dict) and set(document) == {'schema', 'source_id', 'source_kind', 'rows'} and
             document['schema'] == SOURCE_SCHEMA, 'continual source document schema required')
    source_id = _safe_text(document['source_id'], 'source_id')
    _require(document['source_kind'] in SOURCE_KINDS, 'explicit continual source kind required')
    _require(isinstance(document['rows'], list), 'source rows must be a list')
    normalized = {'schema': SOURCE_SCHEMA, 'source_id': source_id, 'source_kind': document['source_kind'], 'rows': []}
    for entry in document['rows']:
        _require(isinstance(entry, dict) and set(entry) == {'row', 'partition', 'ancestry', 'sampling_weight', 'trainer_only'},
                 'closed continual source row fields required')
        _require(entry['partition'] in PARTITIONS, 'continual partition required')
        weight = entry['sampling_weight']
        _require(type(weight) in (int, float) and not isinstance(weight, bool) and math.isfinite(weight) and weight > 0,
                 'positive finite sampling_weight metadata required')
        _require(isinstance(entry['trainer_only'], dict), 'source trainer_only provenance required')
        row = copy.deepcopy(entry['row'])
        checked = validate_row(row)
        ancestry = _validate_ancestry(entry['ancestry'], row)
        normalized['rows'].append({'row': row, 'partition': entry['partition'], 'ancestry': ancestry,
                                   'sampling_weight': float(weight), 'trainer_only': copy.deepcopy(entry['trainer_only']),
                                   '_checked': checked})
    return normalized


def _membership(source: dict, entry: dict, source_document_sha256: str) -> dict:
    row, checked = entry['row'], entry['_checked']
    return {'source_id': source['source_id'], 'source_kind': source['source_kind'],
            'source_document_sha256': source_document_sha256, 'sampling_weight': entry['sampling_weight'],
            'ancestry': copy.deepcopy(entry['ancestry']), 'dawn': row['identity']['dawn'],
            'economic_families': list(checked['economic_families']),
            'replay_identity': _replay_identity(row, checked), 'row_schema': row['schema'],
            'row_identity': copy.deepcopy(row['identity']), 'row_trainer_only': copy.deepcopy(row['trainer_only']),
            'source_trainer_only': copy.deepcopy(entry['trainer_only'])}


def _record(entry: dict, membership: dict) -> dict:
    checked = entry['_checked']; actor, candidates, target = _row_parts(entry['row'])
    return {'schema': RECORD_SCHEMA, 'partition': entry['partition'],
            'actor_context_sha256': checked['actor_context_sha256'], 'target_class_sha256': checked['target_class_sha256'],
            'actor': actor, 'candidates': candidates, 'target': target,
            'economic_families': list(checked['economic_families']), 'memberships': [membership]}


def _membership_key(value: dict) -> str:
    return _digest(value)


def _merge_memberships(record: dict, other: dict) -> None:
    _require(record['partition'] == other['partition'] and record['actor_context_sha256'] == other['actor_context_sha256'] and
             record['target_class_sha256'] == other['target_class_sha256'], 'record merge identity mismatch')
    _require(_digest(record['actor']) == _digest(other['actor']) and _digest(record['candidates']) == _digest(other['candidates']),
             'actor context hash collision')
    _require({k: record['target'][k] for k in ('loss_weight', 'stop_kind')} ==
             {k: other['target'][k] for k in ('loss_weight', 'stop_kind')}, 'intrinsic target metadata disagreement')
    _require(_digest(record['target']) == _digest(other['target']), 'target class hash collision or target metadata drift')
    by_key = {_membership_key(m): m for m in record['memberships']}
    for m in other['memberships']:
        by_key.setdefault(_membership_key(m), m)
    record['memberships'] = [by_key[k] for k in sorted(by_key)]


def _validate_record(record: dict) -> None:
    expected = {'schema', 'partition', 'actor_context_sha256', 'target_class_sha256', 'actor', 'candidates', 'target',
                'economic_families', 'memberships'}
    _require(isinstance(record, dict) and set(record) == expected and record['schema'] == RECORD_SCHEMA,
             'continual record schema required')
    _require(record['partition'] in PARTITIONS and record['memberships'], 'continual record partition/membership required')
    _require(_digest({'actor': record['actor'], 'candidates': record['candidates']}) == record['actor_context_sha256'],
             'continual actor-context digest mismatch')
    target_class = {k: record['target'][k] for k in ('class_indices', 'successor_plan_sha256', 'continuation')}
    _require(_digest(target_class) == record['target_class_sha256'], 'continual target-class digest mismatch')
    _require(record['economic_families'] == sorted(set(record['economic_families'])), 'economic families must be sorted unique')
    for membership in record['memberships']:
        _require(membership['ancestry']['physical_world_id'] == membership['row_identity']['physical_world_id'] and
                 membership['ancestry']['learner_seat'] == membership['row_identity']['learner_seat'], 'membership ancestry mismatch')
        row = materialize_row(record, membership)
        checked = validate_row(row)
        _require(checked['actor_context_sha256'] == record['actor_context_sha256'] and
                 checked['target_class_sha256'] == record['target_class_sha256'] and
                 checked['economic_families'] == record['economic_families'], 'materialized membership semantic mismatch')


def _read_jsonl(path: Path) -> list[dict]:
    from farm_neat.training.v0_bc import _json_loads
    return [_json_loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def load_corpus(directory: str | Path, expected_manifest_sha256: str) -> dict:
    directory = Path(directory)
    _require(directory.is_dir() and not directory.is_symlink(), 'safe continual corpus directory required')
    manifest_path = directory / 'MANIFEST.json'
    _require(_file_sha(manifest_path) == expected_manifest_sha256, 'continual corpus manifest hash mismatch')
    manifest = _read(manifest_path)
    _require(manifest.get('schema') == SCHEMA and manifest.get('status') == 'ASSEMBLED', 'assembled continual corpus required')
    _require(manifest.get('source_pins') == source_pins(), 'continual corpus source pins changed')
    expected_files = {'ROWS.jsonl', 'SEPARATED.jsonl', 'CONFLICTS.json', 'EXPOSURE.json', 'COVERAGE.json'}
    _require(set(manifest.get('files', {})) == expected_files, 'closed continual corpus members required')
    for name, pin in manifest['files'].items():
        path = directory / name
        _require(path.is_file() and not path.is_symlink() and set(pin) == {'bytes', 'sha256'} and
                 path.stat().st_size == pin['bytes'] and _file_sha(path) == pin['sha256'], 'continual member hash mismatch: ' + name)
    rows, separated = _read_jsonl(directory / 'ROWS.jsonl'), _read_jsonl(directory / 'SEPARATED.jsonl')
    for record in rows:
        _validate_record(record)
    for item in separated:
        _require(item.get('schema') == SEPARATED_SCHEMA and set(item) == {'schema', 'reason', 'record'},
                 'separated row schema required')
        _validate_record(item['record'])
    exposure = _validate_registry(_read(directory / 'EXPOSURE.json'))
    conflicts, coverage = _read(directory / 'CONFLICTS.json'), _read(directory / 'COVERAGE.json')
    _require(_digest(rows) == manifest['rows_sha256'] and _digest(exposure) == manifest['exposure_sha256'] and
             _digest(coverage) == manifest['coverage_sha256'], 'continual manifest content digest mismatch')
    return {'manifest': manifest, 'rows': rows, 'separated': separated, 'exposure': exposure,
            'conflicts': conflicts, 'coverage': coverage}


def _coverage(records: Sequence[dict], *, base_minimum: dict[str, int], registry_delta: dict,
              separated: Sequence[dict], conflicts: dict) -> dict:
    memberships = [m for r in records for m in r['memberships']]
    by_source = Counter(m['source_kind'] for m in memberships)
    by_dawn = Counter(str(m['dawn']) for m in memberships)
    by_opp = Counter(m['ancestry']['opponent_digest'] for m in memberships)
    by_partition = Counter(r['partition'] for r in records)
    by_family = Counter()
    for record in records:
        for membership in record['memberships']:
            for family in record['economic_families']:
                by_family[family] += 1
    base_counts = Counter()
    for record in records:
        if record['partition'] != 'train' or not any(m['source_kind'] == BASE for m in record['memberships']):
            continue
        for family in record['economic_families']:
            base_counts[family] += 1
    missing = {family: required - base_counts.get(family, 0) for family, required in base_minimum.items()
               if base_counts.get(family, 0) < required}
    weights = [m['sampling_weight'] for m in memberships]
    return {'schema': COVERAGE_SCHEMA, 'objective': OBJECTIVE, 'records': len(records), 'memberships': len(memberships),
            'by_partition_records': dict(sorted(by_partition.items())), 'by_source_kind': dict(sorted(by_source.items())),
            'by_dawn': dict(sorted(by_dawn.items(), key=lambda x: int(x[0]))),
            'by_economic_family': dict(sorted(by_family.items())), 'by_opponent_digest': dict(sorted(by_opp.items())),
            'base_minimum': dict(sorted(base_minimum.items())), 'base_train_records_by_economic_family': dict(sorted(base_counts.items())),
            'base_retention_passed': not missing, 'base_retention_missing': missing,
            'sampling_weight_metadata': {'count': len(weights), 'min': min(weights) if weights else None,
                                         'max': max(weights) if weights else None,
                                         'used_by_objective': False},
            'registry_delta': registry_delta, 'source_separated_records': len(separated),
            'conflict_contexts': conflicts['conflict_contexts'],
            'strength_result': None, 'native_games_started': 0, 'optimizer_updates': 0}


def _validate_base_minimum(value: dict[str, int] | None) -> dict[str, int]:
    value = {} if value is None else dict(value)
    _require(all(k in ACTION_FAMILIES and type(v) is int and v >= 0 for k, v in value.items()),
             'base minimum must map V0 economic family to nonnegative integer')
    return dict(sorted(value.items()))


def assemble(source_documents: Sequence[dict], output: str | Path, *, registry: dict | None = None,
             parent: str | Path | None = None, parent_manifest_sha256: str | None = None,
             conflict_policy: str = 'reject', base_minimum: dict[str, int] | None = None) -> dict:
    """Create one immutable corpus version. Existing parent/registry bytes are read-only."""
    _require(conflict_policy in CONFLICT_POLICIES, 'known conflict policy required')
    requested_base_minimum = None if base_minimum is None else dict(base_minimum)
    output = Path(output)
    _require(not output.exists() and not any(p.is_symlink() for p in output.parents), 'new non-symlink output required')

    parent_data = None
    if parent is not None:
        _require(parent_manifest_sha256 is not None, 'parent manifest sha256 required')
        parent_data = load_corpus(parent, parent_manifest_sha256)
    elif parent_manifest_sha256 is not None:
        raise ValueError('parent path required with parent manifest sha256')

    if parent_data is not None and requested_base_minimum is None:
        base_minimum = _validate_base_minimum(parent_data['manifest']['base_minimum'])
    else:
        base_minimum = _validate_base_minimum(requested_base_minimum)
    if parent_data is not None:
        prior_minimum = _validate_base_minimum(parent_data['manifest']['base_minimum'])
        _require(all(base_minimum.get(family, 0) >= count for family, count in prior_minimum.items()),
                 'child corpus cannot weaken parent base-coverage minimum')

    if registry is None:
        exposure = empty_registry() if parent_data is None else copy.deepcopy(parent_data['exposure'])
    else:
        exposure = _validate_registry(registry)
        if parent_data is not None:
            _registry_extends(parent_data['exposure'], exposure)
    input_registry = copy.deepcopy(exposure)
    known_world_before, known_pair_before, known_source_before = (set(exposure[name]) for name in ('world_groups', 'pair_groups', 'source_groups'))

    raw_documents = [copy.deepcopy(d) for d in source_documents]
    normalized = [validate_source(d) for d in raw_documents]
    ids = [d['source_id'] for d in normalized]
    _require(len(ids) == len(set(ids)), 'source_id repeated across input documents')
    raw_by_id = {d['source_id']: raw for d, raw in zip(normalized, raw_documents)}
    normalized.sort(key=lambda d: d['source_id'])
    new_bindings = [{'source_id': d['source_id'], 'source_kind': d['source_kind'],
                     'document_sha256': _digest(raw_by_id[d['source_id']])} for d in normalized]
    source_sha = {b['source_id']: b['document_sha256'] for b in new_bindings}
    binding_by_id = {b['source_id']: copy.deepcopy(b) for b in ([] if parent_data is None else parent_data['manifest']['source_documents'])}
    for binding in new_bindings:
        old_binding = binding_by_id.get(binding['source_id'])
        _require(old_binding is None or old_binding == binding, 'source_id reused with changed kind or document bytes')
        binding_by_id[binding['source_id']] = binding
    source_bindings = [binding_by_id[k] for k in sorted(binding_by_id)]

    parent_rows = [] if parent_data is None else copy.deepcopy(parent_data['rows'])
    parent_separated = [] if parent_data is None else copy.deepcopy(parent_data['separated'])
    for row in parent_rows:
        _validate_record(row)

    new_by_key: dict[tuple[str, str, str], dict] = {}
    internal_labels: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    new_world, new_pair, new_source, inspected_reserved = set(), set(), set(), set()
    for source in normalized:
        for entry in source['rows']:
            ancestry, partition = entry['ancestry'], entry['partition']
            world_key = _digest(_group_payload(ancestry, level='world'))
            pair_key = _digest(_group_payload(ancestry, level='pair'))
            source_key = _digest(_group_payload(ancestry, level='source'))
            if partition == 'protected_eval':
                _require(world_key in input_registry['world_groups'] and pair_key in input_registry['pair_groups'] and
                         source_key in input_registry['source_groups'] and input_registry['world_groups'][world_key]['reserved'] and
                         input_registry['pair_groups'][pair_key]['reserved'] and input_registry['source_groups'][source_key]['reserved'],
                         'protected evaluation group must be reserved before first protected row')
                inspected_reserved.add(source_key)
            world_novel, pair_novel, source_novel = _touch(exposure, ancestry, partition, source_id=source['source_id'],
                                                            reserved=False, inspected=True)
            if world_novel:
                new_world.add(world_key)
            if pair_novel:
                new_pair.add(pair_key)
            if source_novel:
                new_source.add(source_key)
            checked = entry['_checked']; membership = _membership(source, entry, source_sha[source['source_id']])
            rec = _record(entry, membership)
            key = (partition, checked['actor_context_sha256'], checked['target_class_sha256'])
            internal_labels[(source['source_id'], partition, checked['actor_context_sha256'])].add(checked['target_class_sha256'])
            if key in new_by_key:
                _merge_memberships(new_by_key[key], rec)
            else:
                new_by_key[key] = rec
    bad_internal = [k for k, labels in internal_labels.items() if len(labels) > 1]
    _require(not bad_internal, 'one source contains conflicting target classes at the same actor context: ' + repr(bad_internal[0] if bad_internal else None))

    active: dict[tuple[str, str, str], dict] = {}
    for record in parent_rows:
        key = (record['partition'], record['actor_context_sha256'], record['target_class_sha256'])
        _require(key not in active, 'parent contains duplicate active record identity')
        active[key] = record
    parent_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for partition, actor_sha, target_sha in active:
        parent_labels[(partition, actor_sha)].add(target_sha)
        _require(len(parent_labels[(partition, actor_sha)]) == 1, 'parent contains active conflicting target classes')

    new_contexts: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for (partition, actor_sha, target_sha), rec in new_by_key.items():
        new_contexts[(partition, actor_sha)][target_sha] = rec

    conflict_rows, separated = [], parent_separated
    for context in sorted(new_contexts):
        labels = set(new_contexts[context]) | parent_labels.get(context, set())
        if len(labels) == 1:
            target_sha = next(iter(labels)); key = (*context, target_sha)
            rec = new_contexts[context].get(target_sha)
            if rec is not None:
                if key in active:
                    _merge_memberships(active[key], rec)
                else:
                    active[key] = rec
            continue
        detail = {'partition': context[0], 'actor_context_sha256': context[1],
                  'target_class_sha256': sorted(labels),
                  'parent_target_class_sha256': sorted(parent_labels.get(context, set())),
                  'new_sources': sorted({m['source_id'] for r in new_contexts[context].values() for m in r['memberships']})}
        conflict_rows.append({'status': 'REJECTED' if conflict_policy == 'reject' else 'SOURCE_SEPARATED', **detail})
        if conflict_policy == 'reject':
            raise ValueError('conflicting target classes at actor context ' + context[1])
        inherited = parent_labels.get(context, set())
        for target_sha, rec in sorted(new_contexts[context].items()):
            key = (*context, target_sha)
            if inherited and target_sha in inherited:
                _merge_memberships(active[key], rec)
            else:
                separated.append({'schema': SEPARATED_SCHEMA, 'reason': 'CONFLICTING_TARGET_CLASS_NOT_IN_UNIFORM_BC',
                                  'record': rec})

    # Same exact actor input can legitimately appear in different source partitions;
    # surface differing labels rather than merging across the split boundary.
    by_actor_partition: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for (partition, actor_sha, target_sha), _ in active.items():
        by_actor_partition[actor_sha][partition].add(target_sha)
    for item in separated:
        rec = item['record']; by_actor_partition[rec['actor_context_sha256']][rec['partition']].add(rec['target_class_sha256'])
    for actor_sha, partitions in sorted(by_actor_partition.items()):
        labels = {label for values in partitions.values() for label in values}
        if len(partitions) > 1 and len(labels) > 1:
            conflict_rows.append({'status': 'PARTITION_SEPARATED', 'actor_context_sha256': actor_sha,
                                  'partitions': {k: sorted(v) for k, v in sorted(partitions.items())}})

    records = [active[k] for k in sorted(active)]
    separated = sorted(separated, key=lambda x: (x['record']['partition'], x['record']['actor_context_sha256'],
                                                  x['record']['target_class_sha256'], _digest(x)))
    for record in records:
        _validate_record(record)
    conflicts = {'schema': CONFLICT_SCHEMA, 'policy': conflict_policy, 'conflict_contexts': len(conflict_rows),
                 'rows': conflict_rows}
    registry_delta = {'novel_world_groups': len(new_world), 'novel_pair_groups': len(new_pair),
                      'novel_source_groups': len(new_source), 'new_world_group_sha256': sorted(new_world),
                      'new_pair_group_sha256': sorted(new_pair), 'new_source_group_sha256': sorted(new_source),
                      'newly_inspected_reserved_source_groups': sorted(inspected_reserved),
                      'known_world_groups_before': len(known_world_before), 'known_pair_groups_before': len(known_pair_before),
                      'known_source_groups_before': len(known_source_before)}
    coverage = _coverage(records, base_minimum=base_minimum, registry_delta=registry_delta,
                         separated=separated, conflicts=conflicts)
    _require(coverage['base_retention_passed'], 'declared base-coverage stratum would be violated: ' + repr(coverage['base_retention_missing']))

    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / 'ROWS.jsonl', records)
    _write_jsonl(output / 'SEPARATED.jsonl', separated)
    _write_json(output / 'CONFLICTS.json', conflicts)
    _write_json(output / 'EXPOSURE.json', _validate_registry(exposure))
    _write_json(output / 'COVERAGE.json', coverage)
    files = {name: {'bytes': (output / name).stat().st_size, 'sha256': _file_sha(output / name)}
             for name in ('ROWS.jsonl', 'SEPARATED.jsonl', 'CONFLICTS.json', 'EXPOSURE.json', 'COVERAGE.json')}
    manifest = {'schema': SCHEMA, 'status': 'ASSEMBLED', 'objective': OBJECTIVE, 'source_pins': source_pins(),
                'action_schema_sha256': V.ACTION_SCHEMA_SHA256, 'parent_manifest_sha256': parent_manifest_sha256,
                'input_registry_sha256': _digest(input_registry), 'source_documents': source_bindings,
                'conflict_policy': conflict_policy, 'base_minimum': base_minimum,
                'counts': {'records': len(records), 'memberships': sum(len(r['memberships']) for r in records),
                           'source_separated_records': len(separated)},
                'rows_sha256': _digest(records), 'exposure_sha256': _digest(exposure), 'coverage_sha256': _digest(coverage),
                'files': files,
                'limits': ['No native games, model calls, optimizer updates or strength measurement are performed.',
                           'sampling_weight is corpus metadata and is not consumed by the unchanged uniform BC objective.',
                           'Native-world/teacher strength remains the responsibility of source admission and later evaluation.']}
    _write_json(output / 'MANIFEST.json', manifest)
    return {'manifest': manifest, 'manifest_sha256': _file_sha(output / 'MANIFEST.json'), 'coverage': coverage,
            'exposure': exposure, 'records': records, 'separated': separated, 'conflicts': conflicts}


def audit_corpus(directory: str | Path, expected_manifest_sha256: str) -> dict:
    data = load_corpus(directory, expected_manifest_sha256)
    coverage = data['coverage']
    _require(coverage['objective'] == OBJECTIVE and coverage['base_retention_passed'], 'continual coverage gate mismatch')
    return {'status': 'VERIFIED', 'manifest_sha256': expected_manifest_sha256,
            'records': len(data['rows']), 'memberships': sum(len(r['memberships']) for r in data['rows']),
            'source_separated_records': len(data['separated']), 'conflict_contexts': data['conflicts']['conflict_contexts'],
            'native_games_started': 0, 'optimizer_updates': 0, 'strength_result': None}


def _parse_base(values: Sequence[str]) -> dict[str, int]:
    result = {}
    for value in values:
        try:
            family, raw = value.split('=', 1); count = int(raw)
        except Exception as exc:
            raise ValueError('base minimum must be FAMILY=N') from exc
        _require(family not in result, 'duplicate base minimum family')
        result[family] = count
    return _validate_base_minimum(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    reserve = sub.add_parser('reserve')
    reserve.add_argument('--input', type=Path, required=True); reserve.add_argument('--output', type=Path, required=True)
    reserve.add_argument('--registry', type=Path); reserve.add_argument('--registry-sha256')
    assemble_p = sub.add_parser('assemble')
    assemble_p.add_argument('--source', type=Path, action='append', required=True); assemble_p.add_argument('--output', type=Path, required=True)
    assemble_p.add_argument('--registry', type=Path); assemble_p.add_argument('--registry-sha256')
    assemble_p.add_argument('--parent', type=Path); assemble_p.add_argument('--parent-manifest-sha256')
    assemble_p.add_argument('--conflicts', choices=CONFLICT_POLICIES, default='reject')
    assemble_p.add_argument('--base-minimum', action='append', default=[], metavar='FAMILY=N')
    audit = sub.add_parser('audit')
    audit.add_argument('--corpus', type=Path, required=True); audit.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'reserve':
            _require(not args.output.exists(), 'reservation output already exists')
            current = None
            if args.registry is not None:
                _require(args.registry_sha256 is not None and _file_sha(args.registry) == args.registry_sha256,
                         'input registry hash mismatch')
                current = _read(args.registry)
            else:
                _require(args.registry_sha256 is None, '--registry-sha256 requires --registry')
            result = reserve_registry(current, _read(args.input)); _write_json(args.output, result)
            print(_json({'status': 'RESERVED', 'registry_sha256': _file_sha(args.output),
                         'world_groups': len(result['world_groups']), 'pair_groups': len(result['pair_groups']),
                         'source_groups': len(result['source_groups'])}))
            return 0
        if args.command == 'assemble':
            current = None
            if args.registry is not None:
                _require(args.registry_sha256 is not None and _file_sha(args.registry) == args.registry_sha256,
                         'input registry hash mismatch')
                current = _read(args.registry)
            else:
                _require(args.registry_sha256 is None, '--registry-sha256 requires --registry')
            result = assemble([_read(p) for p in args.source], args.output, registry=current, parent=args.parent,
                              parent_manifest_sha256=args.parent_manifest_sha256, conflict_policy=args.conflicts,
                              base_minimum=_parse_base(args.base_minimum))
            print(_json({'status': 'ASSEMBLED', 'manifest_sha256': result['manifest_sha256'],
                         'records': result['coverage']['records'], 'memberships': result['coverage']['memberships'],
                         'source_separated_records': result['coverage']['source_separated_records'],
                         'strength_result': None}))
            return 0
        print(_json(audit_corpus(args.corpus, args.manifest_sha256)))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, TypeError, IndexError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
