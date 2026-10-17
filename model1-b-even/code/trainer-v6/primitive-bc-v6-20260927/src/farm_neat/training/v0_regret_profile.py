"""Read-only, identity-aware cost accounting for one V0 regret-teaching round.

``profile_round(receipt)`` consumes farm-v0-regret-profile-input-v1. Timings are
host monotonic seconds relative to a SINGLE round origin, never GPU kernel time.
Each timing event has event_id, component, phase, status and either start_seconds /
end_seconds or duration_seconds (unplaced). Optional parent_id expresses inclusive
nesting, not a dependency. IDs are global within the round: repeated identical
exports are idempotent; conflicting exports refuse.

Separate execution/draw/cache identities prevent counting summaries, replays or
cache reuse as fresh observations. Admissions are sealed receipts from the existing
v0_regret_admission gate, counted by (protocol_sha256, test_id), not by seat. This
module checks their reporting consistency; it DOES NOT replace the admission gate,
authenticate a producer, prove independent worlds, or generate training labels.

RoundRecorder supplies optional observer-only spans to existing orchestration.
``profile`` reads JSON and launches nothing. ``diagnostic`` measures CPU accounting
work only, with no torch, checkpoint, native engine or training dependency. Existing
components without positioned timing events remain explicitly unmeasured.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import threading
import time
from typing import Any, Iterator

INPUT_SCHEMA = 'farm-v0-regret-profile-input-v1'
OUTPUT_SCHEMA = 'farm-v0-regret-profile-v1'
ADMISSION_SCHEMA = 'farm-v0-regret-admission-receipt-v1'
PHASES = ('collection', 'proposal_generation', 'solver_candidate', 'external_teacher',
          'model_scoring', 'compilation', 'mc_discovery', 'cache_lookup',
          'adaptive_allocation', 'confirmation', 'replay_verification',
          'data_transfer', 'optimizer', 'training', 'evaluation', 'other')
COMPONENTS = ('collector', 'searcher', 'trainer', 'evaluator')
SCOPES = ('SYNTHETIC', 'CPU_DIAGNOSTIC', 'PRODUCTION')
FAILURES = {'FAILED', 'CRASHED', 'TIMEOUT', 'CENSORED', 'INVALID', 'PENDING', 'UNKNOWN'}
RESOURCE_FIELDS = ('candidate_batch_size', 'cpu_workers', 'peak_ram_bytes', 'peak_device_bytes', 'device')


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def _wire(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('utf-8')


def _sha(value: Any) -> str:
    return hashlib.sha256(_wire(value)).hexdigest()


def _seal(value: dict) -> dict:
    value['content_sha256'] = _sha(value)
    return value


def _number(value: Any, name: str) -> float:
    _need(type(value) in (float, int) and math.isfinite(value) and value >= 0, 'INVALID_NUMBER:'+name)
    return float(value)


def _text(value: Any, name: str) -> str:
    _need(isinstance(value, str) and bool(value.strip()), 'INVALID_TEXT:'+name)
    return value


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _unique(rows: list, key: str) -> tuple[list[dict], int]:
    _need(isinstance(rows, list), 'ROWS_MUST_BE_LIST:'+key)
    found = {}; duplicates = 0
    for row in rows:
        _need(isinstance(row, dict), 'ROW_MUST_BE_OBJECT:'+key)
        identity = _text(row.get(key), key)
        if identity in found:
            _need(_wire(found[identity]) == _wire(row), 'IDENTITY_CONFLICT:'+key+':'+identity)
            duplicates += 1
        else:
            found[identity] = row
    return list(found.values()), duplicates


def _union(intervals: list[tuple[float, float]]) -> float:
    result = 0.; right = None
    for start, end in sorted(intervals):
        if right is None or start >= right:
            result += end-start
        elif end > right:
            result += end-right
        right = end if right is None else max(right, end)
    return result


def _timings(events: list[dict], wall: float, negative_ids: set[str]) -> tuple[dict, dict, dict]:
    phases = {p: dict(event_count=0, inclusive_wall_seconds=None, exclusive_wall_seconds=0.,
                     summed_span_seconds=0., unplaced_span_seconds=0., failed_events=0)
              for p in PHASES}
    placed = {}; ancestors = {}; unplaced = 0
    for event in events:
        phase = event.get('phase'); _need(phase in PHASES, 'UNKNOWN_PHASE:'+str(phase))
        _need(event.get('component') in COMPONENTS, 'UNKNOWN_COMPONENT')
        _text(event.get('status'), 'status')
        entry = phases[phase]; entry['event_count'] += 1
        entry['failed_events'] += event['status'] in FAILURES
        positioned = 'start_seconds' in event or 'end_seconds' in event
        if positioned:
            _need('duration_seconds' not in event, 'AMBIGUOUS_DURATION')
            start = _number(event.get('start_seconds'), 'start_seconds')
            end = _number(event.get('end_seconds'), 'end_seconds')
            _need(start <= end <= wall, 'SPAN_OUTSIDE_ROUND_OR_REVERSED')
            placed[event['event_id']] = (start, end, event)
            duration = end-start
        else:
            _need(event.get('parent_id') is None, 'UNPLACED_PARENT')
            duration = _number(event.get('duration_seconds'), 'duration_seconds')
            entry['unplaced_span_seconds'] += duration
            unplaced += 1
        entry['summed_span_seconds'] += duration
    for identity, (start, end, event) in placed.items():
        chain = set(); parent = event.get('parent_id')
        while parent is not None:
            _need(parent != identity and parent not in chain and parent in placed, 'INVALID_PARENT:'+identity)
            chain.add(parent)
            ps, pe, row = placed[parent]
            _need(ps <= start <= end <= pe, 'NONCONTAINING_PARENT:'+identity)
            parent = row.get('parent_id')
        ancestors[identity] = chain
    for phase in PHASES:
        intervals = [(s,e) for s,e,v in placed.values() if v['phase'] == phase]
        if intervals:
            phases[phase]['inclusive_wall_seconds'] = _union(intervals)
    # Sweep endpoints; suppress inclusive ancestors, retain overlap rather than
    # guessing which concurrent worker "owns" shared wall time.
    transitions = defaultdict(lambda: [set(), set()])
    for identity, (start, end, _) in placed.items():
        if end > start:
            transitions[start][1].add(identity); transitions[end][0].add(identity)
    transitions[0.]; transitions[wall]
    active = set(); concurrent = unattributed = 0.; overlap = Counter()
    points = sorted(transitions)
    for i, start in enumerate(points[:-1]):
        remove, add = transitions[start]; active.difference_update(remove); active.update(add)
        duration = points[i+1]-start
        suppressed = set().union(*(ancestors[v] for v in active)) if active else set()
        leaves = active-suppressed
        active_phases = {placed[v][2]['phase'] for v in leaves}
        if len(active_phases) == 1:
            phases[next(iter(active_phases))]['exclusive_wall_seconds'] += duration
        elif active_phases:
            concurrent += duration; overlap[' + '.join(sorted(active_phases))] += duration
        else:
            unattributed += duration
    exclusive = math.fsum(v['exclusive_wall_seconds'] for v in phases.values())
    def selected(predicate):
        return _union([(s,e) for s,e,v in placed.values() if predicate(v)])
    costs = dict(
        verification_active_wall_seconds=selected(lambda v: v['phase']=='replay_verification'),
        verification_unplaced_span_seconds=phases['replay_verification']['unplaced_span_seconds'],
        confirmation_active_wall_seconds=selected(lambda v: v['phase']=='confirmation'),
        failed_active_wall_seconds=selected(lambda v: v['status'] in FAILURES),
        negative_search_active_wall_seconds=selected(lambda v: v.get('test_id') in negative_ids and
            v['phase'] in ('proposal_generation','solver_candidate','external_teacher','model_scoring',
                           'compilation','mc_discovery','adaptive_allocation','cache_lookup')),
        negative_search_attribution='Only spans explicitly joined by test_id; shared/unjoined costs are not apportioned.')
    timing = dict(end_to_end_seconds=wall, exclusive_wall_seconds=exclusive,
        concurrent_wall_seconds=concurrent, unattributed_wall_seconds=unattributed,
        reconciled_seconds=math.fsum((exclusive, concurrent, unattributed)),
        concurrent_phase_sets_seconds=dict(sorted(overlap.items())), unplaced_events=unplaced,
        units='HOST_MONOTONIC_SECONDS',
        scope='Inclusive and summed-span columns are non-additive; only exclusive + concurrent + unattributed reconciles.')
    _need(math.isclose(timing['reconciled_seconds'], wall, abs_tol=1e-9), 'TIMING_RECONCILIATION')
    return phases, timing, costs


def _admissions(rows: list[dict]) -> tuple[int, list[dict], set[str], int]:
    seen = {}; useful = 0; rejected = []; negative = set(); duplicates = 0
    for row in rows:
        _need(isinstance(row, dict) and row.get('schema') == ADMISSION_SCHEMA, 'ADMISSION_SCHEMA')
        _need(row.get('content_sha256') == _sha({k:v for k,v in row.items() if k != 'content_sha256'}),
              'ADMISSION_HASH_MISMATCH')
        # Invalid gate receipts can lack a test/protocol identity. They still
        # carry a costed refusal, but cannot be used to infer a tested opportunity.
        key = (row.get('protocol_sha256'), row.get('test_id'))
        if not all(key):
            key = ('INVALID', row['content_sha256'])
        if key in seen:
            _need(seen[key] == row['content_sha256'], 'ADMISSION_IDENTITY_CONFLICT')
            duplicates += 1; continue
        seen[key] = row['content_sha256']
        c = row.get('counts', {})
        n = c.get('confirmation_pairs')
        valid = (row.get('status') == 'ADMITTED' and row.get('verdict') == 'CHALLENGER_BETTER'
            and row.get('objective') == 'TERMINAL_WDL' and row.get('actor_training_admissible') is True
            and row.get('statistical_support') is True and _hash(row.get('protocol_sha256'))
            and _hash(row.get('input_sha256')) and isinstance(row.get('test_id'), str)
            and not row.get('reason_codes') and type(n) is int and n > 0
            and type(c.get('complete_pairs')) is int and type(c.get('replay_verified_pairs')) is int
            and c['complete_pairs'] == n and c['replay_verified_pairs'] == n
            and type(row.get('independent_world_count')) is int and row['independent_world_count'] > 0
            and row['independent_world_count'] == row.get('registered_world_count'))
        for field in ('teacher_weight', 'mean_wdl_delta', 'lower_bound'):
            value = row.get(field)
            valid = valid and type(value) in (int,float) and math.isfinite(value) and value > 0
        if valid:
            useful += 1
        else:
            rejected.append(dict(test_id=row.get('test_id'), status=row.get('status'),
                reason='NO_VERIFIED_USEFUL_CORRECTION', producer_reasons=row.get('reason_codes', []),
                content_sha256=row['content_sha256']))
            if row.get('test_id') is not None:
                negative.add(row['test_id'])
    return useful, rejected, negative, duplicates


def _counts(executions: list[dict], draws: list[dict], cache: list[dict]) -> dict:
    native = replay = failed = 0; families = set(); native_families = set(); worlds = set()
    unknown_groups = 0; group_families = {}; by_phase = defaultdict(set); starts_by_phase = Counter()
    for event in executions:
        _need(event.get('evidence_kind') in ('NATIVE_GAME','INJECTED_FIXTURE'), 'EXECUTION_EVIDENCE_KIND')
        _need(type(event.get('is_replay')) is bool, 'EXECUTION_REPLAY_FLAG')
        _need(event.get('phase') in PHASES, 'EXECUTION_PHASE')
        _text(event.get('status'), 'execution.status')
        native += event['evidence_kind'] == 'NATIVE_GAME'
        starts_by_phase[event['phase']] += 1
        replay += event['is_replay']; failed += event['status'] in FAILURES
        if not event['is_replay']:
            group = event.get('source_group_id'); family = event.get('independent_group_id')
            if group is not None: worlds.add(_text(group, 'source_group_id'))
            if family is None:
                unknown_groups += 1
            else:
                if group is not None:
                    _need(group not in group_families or group_families[group] == family, 'GROUP_FAMILY_CONFLICT')
                    group_families[group] = family
                families.add(_text(family, 'independent_group_id'))
                by_phase[event['phase']].add(family)
                if event['evidence_kind'] == 'NATIVE_GAME': native_families.add(family)
    for row in draws:
        _need(row.get('status') in ('RETURNED','FAILED','PENDING','UNKNOWN'), 'DIRECT_DRAW_STATUS')
    for row in cache:
        _need(row.get('outcome') in ('HIT','MISS'), 'CACHE_OUTCOME')
    return dict(simulator_starts=len(executions), physical_simulator_starts=native,
        fixture_simulator_starts=len(executions)-native, replay_starts=replay,
        failed_simulator_starts=failed, source_world_groups=len(worlds),
        simulator_starts_by_phase=dict(sorted(starts_by_phase.items())),
        independent_groups_by_phase={k:len(v) for k,v in sorted(by_phase.items())},
        independent_continuation_groups=len(families), native_independent_continuation_groups=len(native_families),
        base_starts_without_independent_group=unknown_groups, replay_independent_groups_added=0,
        direct_draw_attempts=len(draws), successful_direct_draws=sum(r['status']=='RETURNED' for r in draws),
        cache_hits=sum(r['outcome']=='HIT' for r in cache), cache_misses=sum(r['outcome']=='MISS' for r in cache))


def _resources(raw: dict) -> dict:
    _need(isinstance(raw, dict), 'RESOURCES_OBJECT_REQUIRED')
    result = {k:raw.get(k) for k in RESOURCE_FIELDS}
    for key in RESOURCE_FIELDS[:-1]:
        v = result[key]
        _need(v is None or (type(v) is int and v >= (1 if key in RESOURCE_FIELDS[:2] else 0)),
              'INVALID_RESOURCE:'+key)
    result['memory_scope'] = raw.get('memory_scope', 'UNSPECIFIED')
    result['missing'] = [k for k in RESOURCE_FIELDS if result[k] is None]
    return result


def _recommend(receipt: dict, phases: dict, timing: dict, resources: dict) -> dict | None:
    if receipt['evidence_scope'] != 'PRODUCTION': return None
    if receipt.get('component_receipts') and receipt.get('coverage', {}).get('timings_complete') is not True:
        return None
    pins = receipt.get('source_pins', {}); runtime = receipt.get('runtime', {})
    if not pins or not all(_hash(v) for v in pins.values()): return None
    if any(runtime.get(k) != v for k,v in dict(python='3.12.3',torch='2.10.0+cu128',numpy='2.4.3').items()):
        return None
    if not runtime.get('slurm_job_id') or not _hash(runtime.get('storage_receipt_sha256')): return None
    if any(resources[k] is None for k in ('candidate_batch_size','cpu_workers','peak_ram_bytes')): return None
    if resources.get('device') != 'cpu' and resources.get('peak_device_bytes') is None: return None
    # A long outer collection/train timer alone is not a diagnosed subcomponent.
    candidates = {k:v for k,v in phases.items() if k not in ('collection','training','evaluation','other')}
    phase = max(candidates, key=lambda k:candidates[k]['exclusive_wall_seconds'])
    seconds = candidates[phase]['exclusive_wall_seconds']
    other = max((v['exclusive_wall_seconds'] for k,v in phases.items() if k != phase), default=0.)
    unresolved = timing['concurrent_wall_seconds']+timing['unattributed_wall_seconds']
    if seconds <= max(other, unresolved) or timing['unplaced_events']: return None
    actions = {
        'proposal_generation':'Profile candidate generation at the recorded batch size before changing search depth.',
        'solver_candidate':'Profile the dominant solver provider without pruning additional legal proposals.',
        'external_teacher':'Measure provider latency and useful-candidate yield before changing call budgets.',
        'model_scoring':'Compare prefix-correct candidate batching/cache hits at fixed support and exact outputs.',
        'compilation':'Measure repeated identical compilation keys before introducing any cache.',
        'mc_discovery':'Compare depth/allocation at the same physical-start budget and independent confirmation gate.',
        'cache_lookup':'Profile verified cache lookup/serialization; do not weaken cache identity checks.',
        'adaptive_allocation':'Profile allocation bookkeeping at the recorded candidate and group counts.',
        'confirmation':'Measure correction yield per fixed confirmation group; do not relax admission.',
        'replay_verification':'Profile exact replay/restoration; preserve the verification gate.',
        'data_transfer':'Measure synchronized transfer/batching separately before changing placement.',
        'optimizer':'Capture device-kernel timing before changing optimizer or loss semantics.'}
    return dict(phase=phase, measured_exclusive_wall_seconds=seconds,
        scope='HOST_WALL_NOT_GPU_KERNEL_TIME', action=actions[phase],
        qualification='Conditional on producer/runtime provenance; receipt hashes are not authentication.')


def _expand_components(receipt: dict) -> tuple[dict, list[str]]:
    """Adapt existing receipts without adding their overlapping aggregate totals.

    Monte Carlo attempts carry physical execution identities; use the actual
    branch attempt_id when present, otherwise a caller-owned invocation/index.
    Native IDs are never replaced by a hash of an outcome. Other component totals
    remain reports, NOT additional events. No implicit actor/provenance passthrough.
    """
    result = copy.deepcopy(receipt); summaries = result.setdefault('source_summaries', [])
    aggregate_only = []; seen = {}
    for item in result.get('component_receipts', []):
        _need(isinstance(item, dict) and item.get('component') in COMPONENTS, 'COMPONENT_RECEIPT')
        identity = _text(item.get('invocation_id'), 'invocation_id')
        raw = item.get('receipt'); _need(isinstance(raw, dict), 'COMPONENT_RECEIPT_OBJECT')
        digest = _sha(raw)
        if identity in seen:
            _need(seen[identity] == digest, 'COMPONENT_INVOCATION_CONFLICT')
            continue
        seen[identity] = digest
        if 'content_sha256' in raw:
            _need(raw['content_sha256'] == _sha({k:v for k,v in raw.items() if k!='content_sha256'}),
                  'COMPONENT_RECEIPT_HASH_MISMATCH')
        counts = raw.get('counts', {})
        _need(isinstance(counts, dict), 'COMPONENT_COUNTS_OBJECT')
        summary = dict(invocation_id=identity, component=item['component'], schema=raw.get('schema'),
            source_receipt_sha256=digest, producer_content_sha256=raw.get('content_sha256'),
            reported_counts=copy.deepcopy(counts), reported_costs=copy.deepcopy(raw.get('costs', {})),
            aggregate_policy='NEVER_ADDED_TO_EVENT_COUNTS_OR_END_TO_END_TIME')
        for key in ('optimizer_updates','global_update','physical_game_executions',
                    'native_starts_including_failures','independent_world_groups'):
            if key in raw: summary[key] = raw[key]
        summaries.append(summary)
        if item['component'] == 'collector':
            result.setdefault('admissions', []).extend(copy.deepcopy(raw.get('admissions', [])))
        if item['component'] == 'searcher' and raw.get('schema') == 'farm-v0-regret-monte-carlo-v1':
            kind = raw.get('execution', {}).get('kind')
            _need(kind in ('NATIVE_GAME','SYNTHETIC_DIAGNOSTIC','INJECTED_FIXTURE'), 'MC_EXECUTION_KIND')
            native = kind == 'NATIVE_GAME'; attempts = raw.get('attempts', [])
            _need(isinstance(attempts,list), 'MC_ATTEMPTS_LIST')
            _need(type(counts.get('simulator_starts')) is int and counts['simulator_starts']==len(attempts)
                  and type(counts.get('physical_starts')) is int
                  and counts['physical_starts']==int(native)*len(attempts), 'MC_START_COUNT_MISMATCH')
            families = raw.get('registration', {}).get('source_families', {})
            for index, attempt in enumerate(attempts):
                _need(type(attempt.get('simulator_starts')) is int and attempt['simulator_starts']==1
                      and type(attempt.get('physical_starts')) is int
                      and attempt['physical_starts']==int(native), 'MC_ATTEMPT_START_COUNT')
                _need(attempt.get('phase') in ('EVALUATE','REPLAY'), 'MC_ATTEMPT_PHASE')
                phase = 'replay_verification' if attempt['phase']=='REPLAY' else 'mc_discovery'
                attempt_id = attempt.get('branch', {}).get('attempt_id') or f'{identity}/attempt/{index}'
                result.setdefault('executions', []).append(dict(execution_id=attempt_id,
                    phase=phase, evidence_kind='NATIVE_GAME' if native else 'INJECTED_FIXTURE',
                    is_replay=attempt['phase']=='REPLAY', source_group_id=attempt.get('group_id'),
                    independent_group_id=families.get(attempt.get('group_id')), status=attempt['event']))
                if 'wall_seconds' in attempt:
                    result.setdefault('timing_events', []).append(dict(event_id=attempt_id+'/wall',
                        component='searcher', phase=phase, duration_seconds=attempt['wall_seconds'],
                        status=attempt['event']))
            for job in raw.get('jobs', []):
                if job.get('status') == 'CACHE_HIT':
                    result.setdefault('cache_events', []).append(dict(
                        cache_event_id=f"{identity}/cache/{job['job_id']}",outcome='HIT'))
            summary['identity_accounting'] = 'MC_ATTEMPTS_RECONCILED_NO_NEW_CACHE_WORLDS'
            summary['cache_misses'] = counts.get('simulation_cache_misses')
        else:
            aggregate_only.append(identity)
            summary['identity_accounting'] = 'AGGREGATE_ONLY_NOT_A_PHYSICAL_START_LEDGER'
            if item['component']=='searcher' and raw.get('schema')=='farm-v0-regret-provider-candidates-v1':
                if 'elapsed_seconds' in counts:
                    result.setdefault('timing_events', []).append(dict(event_id=identity+'/provider',
                        component='searcher',phase='external_teacher',duration_seconds=counts['elapsed_seconds'],
                        status='COMPLETE'))
    return result, aggregate_only


def profile_round(receipt: dict) -> dict:
    """Return a sealed cost report, without mutating data or launching work.

    Duration-only observations are exposed but never packed into an invented
    timeline. Empty identity lists mean zero recorded events, not proof that an
    incompletely instrumented producer performed no work: coverage is explicit.
    """
    _need(isinstance(receipt, dict) and receipt.get('schema') == INPUT_SCHEMA, 'PROFILE_INPUT_SCHEMA')
    _wire(receipt)  # Refuse non-JSON/NaN evidence before arithmetic or hashing.
    _text(receipt.get('round_id'), 'round_id')
    _need(receipt.get('evidence_scope') in SCOPES, 'EVIDENCE_SCOPE')
    original_sha256 = _sha(receipt)
    receipt, aggregate_only = _expand_components(receipt)
    wall = _number(receipt.get('end_to_end_seconds'), 'end_to_end_seconds')
    rows = {}; dedup = {}
    for field,key in (('timing_events','event_id'),('executions','execution_id'),
                      ('direct_draws','draw_id'),('cache_events','cache_event_id')):
        rows[field], dedup[field] = _unique(receipt.get(field, []), key)
    useful, rejected, negative, dedup['admissions'] = _admissions(receipt.get('admissions', []))
    phases, timing, costs = _timings(rows['timing_events'], wall, negative)
    counts = _counts(rows['executions'], rows['direct_draws'], rows['cache_events'])
    resources = _resources(receipt.get('resources', {}))
    rate = useful/wall if wall > 0 else None
    claimed = receipt.get('claimed_verified_useful_corrections_per_second')
    if claimed is not None:
        claimed = _number(claimed, 'claimed_rate')
        _need(rate is not None and math.isclose(claimed, rate, rel_tol=1e-9, abs_tol=1e-12), 'THROUGHPUT_CLAIM_REJECTED')
    result = dict(schema=OUTPUT_SCHEMA, round_id=receipt['round_id'], evidence_scope=receipt['evidence_scope'],
        input_sha256=original_sha256, phases=phases, timing=timing, costs=costs, counts=counts,
        totals=dict(physical_simulator_starts=counts['physical_simulator_starts']
            if receipt.get('coverage', {}).get('executions_complete') is True and not aggregate_only else None,
            scope='TOTAL_ONLY_WITH_EXPLICIT_COMPLETE_EXECUTION_IDENTITY_COVERAGE'),
        resources=resources, deduplicated=dedup, rejected_admissions=rejected,
        throughput=dict(verified_useful_corrections=useful,
            verified_useful_corrections_per_second=rate, denominator_seconds=wall,
            status='UNDEFINED_ZERO_DURATION' if wall == 0 else 'NO_USEFUL_CORRECTIONS' if useful == 0 else 'RECORDED_CORRECTIONS',
            evidence_scope=receipt['evidence_scope'], unit='UNIQUE_ADMITTED_PROTOCOL_TEST_NOT_SEATS_OR_TRAINING_EPOCHS'),
        coverage=dict(aggregate_only_invocations=aggregate_only, missing_event_collections=[k for k in (*rows,'admissions') if k not in receipt],
            unmeasured_phases=[k for k,v in phases.items() if v['inclusive_wall_seconds'] is None],
            producer_coverage=copy.deepcopy(receipt.get('coverage', {}))),
        source_pins=copy.deepcopy(receipt.get('source_pins', {})),
        runtime=copy.deepcopy(receipt.get('runtime', {})),
        source_summaries=copy.deepcopy(receipt.get('source_summaries', [])),
        recommendation=_recommend(receipt, phases, timing, resources),
        limitations=['READ_ONLY_NOT_AN_ADMISSION_OR_TRAINING_GATE',
            'HOST_WALL_NOT_CUDA_KERNEL_TIME_NO_GPU_SPEED_ESTIMATE',
            'WORLD_INDEPENDENCE_AND_EVENT_COMPLETENESS_ARE_PRODUCER_ATTESTATIONS',
            'HASHES_CHECK_INTEGRITY_NOT_AUTHENTICITY',
            'SOURCE_SUMMARIES_ARE_NOT_ADDED_TO_EVENT_COUNTS',
            'NO_POLICY_STRENGTH_OR_EXPECTED_IMPROVEMENT_CLAIM'])
    return _seal(result)


# Tournament orchestration is deliberately a read-only extension of the profiler.
# Searchers own discovery, the canonical proposer owns reachability, and admit_pair
# owns labels. This seam freezes joins/costs and never implements a second searcher.
TOURNAMENT_RECIPE_SCHEMA = 'farm-v0-regret-tournament-recipe-v1'
TOURNAMENT_SCHEMA = 'farm-v0-regret-tournament-registration-v1'
TOURNAMENT_FROZEN_SCHEMA = 'farm-v0-regret-tournament-discovery-v1'
TOURNAMENT_ARMS = ('flat_fixed_n', 'iterative_fixed', 'uniform_mc', 'adaptive_mc',
                   'external_teacher_mc', 'value_guided_mc')
TOURNAMENT_MODES = ('physical_starts', 'wall_time')
TOURNAMENT_STRATA = ('opening', 'recovery')
TOURNAMENT_RULE = 'ADMITTED_YIELD_THEN_WDL_PER_TOTAL_SECOND_STRICTLY_BEAT_FLAT_V1'
_T_RECIPE_FIELDS = ('schema', 'tournament_id', 'evidence_scope', 'frozen_before_outcomes',
    'confirmation_unexposed', 'final_unexposed', 'phase_costs_disjoint',
    'parent_checkpoint_sha256', 'opponent_sha256', 'source_pins', 'family_alpha',
    'seeds', 'selection_rule', 'arms', 'budgets', 'draws')
_T_LIMITS = [
    'REPORTING_ONLY_NEVER_AN_ACTOR_LABEL_OR_POLICY_PROMOTION',
    'EXISTING_PROPOSER_OWNS_PREFIX_REACHABILITY_SEARCHER_OWNS_DISCOVERY',
    'EXISTING_ADMISSION_GATE_IS_RERUN_NOT_REPLACED_BY_PROFILE_FLAGS',
    'HASHES_CANNOT_PROVE_CHRONOLOGY_INDEPENDENCE_OR_ABSENCE_OF_UNDISCLOSED_EXPOSURE',
    'KEEP_REGISTRATION_AND_DISCOVERY_LOCK_IN_EXTERNAL_CUSTODY_BEFORE_CONFIRMATION',
    'FIXTURE_START_EQUIVALENTS_ARE_NOT_PHYSICAL_NATIVE_GAMES',
    'CACHE_SAVINGS_ARE_PRODUCER_ATTESTED_NEVER_SUBTRACTED_FROM_ACTUAL_COST',
    'STRATA_AND_BUDGET_MODES_HAVE_SEPARATE_DESCRIPTIVE_DECISIONS_NOT_A_POOLED_STRENGTH_TEST',
    'CONFIRMED_MEANS_ARE_DESCRIPTIVE_ON_OBSERVED_TESTS_NO_ARM_DIFFERENCE_SIGNIFICANCE_CLAIM',
    'SHARED_FROZEN_DRAW_WALL_IS_CHARGED_TO_EACH_ARM_NOT_SUMMED_AS_CAMPAIGN_WALL',
    'STARTS_AND_CPU_COVER_SEARCH_CONFIRMATION_REPLAY_NOT_SUNK_DRAW_COLLECTION_STARTS_OR_CPU',
]


def _t_fields(value: Any, fields: tuple | list, code: str, optional: tuple = ()) -> None:
    _need(type(value) is dict and set(fields) <= set(value) <= set(fields) | set(optional), code)


def _t_int(value: Any, name: str, minimum: int = 0) -> int:
    _need(type(value) is int and value >= minimum, 'TOURNAMENT_INTEGER:'+name)
    return value


def _t_bound(value: dict, schema: str) -> None:
    _need(type(value) is dict and value.get('schema') == schema, 'TOURNAMENT_SCHEMA')
    _need(value.get('content_sha256') == _sha({k:v for k,v in value.items() if k!='content_sha256'}),
          'TOURNAMENT_HASH_MISMATCH')


def _t_validate_recipe(recipe: dict) -> None:
    _t_fields(recipe, _T_RECIPE_FIELDS, 'TOURNAMENT_RECIPE_FIELDS')
    _wire(recipe)
    _need(recipe['schema'] == TOURNAMENT_RECIPE_SCHEMA, 'TOURNAMENT_RECIPE_SCHEMA')
    _text(recipe['tournament_id'], 'tournament_id')
    _need(recipe['evidence_scope'] in ('SYNTHETIC', 'PRODUCTION'), 'TOURNAMENT_SCOPE')
    for key in ('frozen_before_outcomes','confirmation_unexposed','final_unexposed','phase_costs_disjoint'):
        _need(recipe[key] is True, 'TOURNAMENT_ATTESTATION:'+key)
    for key in ('parent_checkpoint_sha256','opponent_sha256'):
        _need(_hash(recipe[key]), 'TOURNAMENT_HASH:'+key)
    pins = recipe['source_pins']
    _need(type(pins) is dict and bool(pins) and all(isinstance(k,str) and k and _hash(v) for k,v in pins.items()),
          'TOURNAMENT_SOURCE_PINS')
    _need(0 < _number(recipe['family_alpha'],'family_alpha') <= .05, 'TOURNAMENT_ALPHA')
    _need(recipe['selection_rule'] == TOURNAMENT_RULE, 'TOURNAMENT_SELECTION_RULE')
    _t_fields(recipe['seeds'], ('proposal','allocation','execution_order'), 'TOURNAMENT_SEEDS')
    for key,value in recipe['seeds'].items(): _t_int(value,key)
    _need(type(recipe['arms']) is list and len(recipe['arms']) == len(TOURNAMENT_ARMS), 'TOURNAMENT_ARMS')
    _need({a.get('arm_id') for a in recipe['arms'] if isinstance(a,dict)} == set(TOURNAMENT_ARMS), 'TOURNAMENT_ARMS')
    for arm in recipe['arms']:
        _t_fields(arm, ('arm_id','available','unavailable_reason','implementation_sha256',
            'provider_id','model_id','model_sha256','settings'), 'TOURNAMENT_ARM_FIELDS')
        _need(type(arm['available']) is bool, 'TOURNAMENT_AVAILABILITY')
        if not arm['available']:
            _need(arm['arm_id']=='external_teacher_mc', 'REQUIRED_ARM_UNAVAILABLE')
            _text(arm['unavailable_reason'],'unavailable_reason')
        else:
            _need(arm['unavailable_reason'] is None, 'AVAILABLE_ARM_HAS_REFUSAL')
        _need(_hash(arm['implementation_sha256']), 'TOURNAMENT_IMPLEMENTATION_PIN')
        for key in ('provider_id','model_id'): _text(arm[key],key)
        _need(_hash(arm['model_sha256']) or (arm['model_id']=='none' and arm['model_sha256'] is None),
              'TOURNAMENT_MODEL_PIN')
        _need(type(arm['settings']) is dict, 'TOURNAMENT_ARM_SETTINGS')
        _t_int(arm['settings'].get('max_proposals'),'max_proposals',1)
    _t_fields(recipe['budgets'], TOURNAMENT_MODES, 'TOURNAMENT_BUDGET_MODES')
    for mode,cap in recipe['budgets'].items():
        _t_fields(cap, ('physical_start_cap','wall_seconds_cap','wall_match_tolerance_seconds'), 'TOURNAMENT_BUDGET')
        _t_int(cap['physical_start_cap'],'physical_start_cap',1)
        wall = _number(cap['wall_seconds_cap'],'wall_seconds_cap')
        tolerance = _number(cap['wall_match_tolerance_seconds'],'wall_match_tolerance_seconds')
        _need(wall > 0 and tolerance <= wall*.01, 'TOURNAMENT_WALL_TOLERANCE')
    draws = recipe['draws']
    _need(type(draws) is list and 0 < len(draws) <= 256, 'TOURNAMENT_DRAWS')
    _need({d.get('stratum') for d in draws if isinstance(d,dict)} == set(TOURNAMENT_STRATA), 'TOURNAMENT_STRATA')
    draw_ids = set(); direct_ids = set(); exposure = {}; descriptors = {}
    for draw in draws:
        _t_fields(draw, ('draw_id','stratum','seats','source_group_id','actor_context_sha256s',
            'continuation_sha256s','learner_path_sha256s','direct_draw_ids','collection_seconds',
            'recovery_snapshot_sha256','worlds'), 'TOURNAMENT_DRAW_FIELDS')
        identity = _text(draw['draw_id'],'draw_id')
        _need(identity not in draw_ids, 'DUPLICATE_LEARNER_DRAW'); draw_ids.add(identity)
        _text(draw['source_group_id'],'source_group_id')
        _need(draw['seats'] == [0,1] and all(type(s) is int for s in draw['seats']), 'TOURNAMENT_BOTH_SEATS_REQUIRED')
        for key in ('actor_context_sha256s','continuation_sha256s','learner_path_sha256s','direct_draw_ids'):
            _t_fields(draw[key], ('0','1'), 'TOURNAMENT_SEAT_BINDINGS')
            for value in draw[key].values():
                if key=='direct_draw_ids':
                    _text(value,key); _need(value not in direct_ids,'DUPLICATE_DIRECT_DRAW'); direct_ids.add(value)
                else: _need(_hash(value), 'TOURNAMENT_DRAW_HASH:'+key)
        _number(draw['collection_seconds'],'collection_seconds')
        _need(_hash(draw['recovery_snapshot_sha256']) if draw['stratum']=='recovery'
              else draw['recovery_snapshot_sha256'] is None, 'RECOVERY_SNAPSHOT_BINDING')
        _t_fields(draw['worlds'], ('discovery','confirmation','protected_final'), 'TOURNAMENT_WORLD_SPLITS')
        for phase,worlds in draw['worlds'].items():
            _need(type(worlds) is list and bool(worlds), 'TOURNAMENT_EMPTY_WORLD_SPLIT')
            local = set()
            for world in worlds:
                _t_fields(world, ('source_group_id','source_family_id','world_sha256','snapshot_sha256','rng_sha256','seed'),
                          'TOURNAMENT_WORLD_FIELDS')
                for key in ('source_group_id','source_family_id'): _text(world[key],key)
                for key in ('world_sha256','snapshot_sha256','rng_sha256'):
                    _need(_hash(world[key]), 'TOURNAMENT_WORLD_HASH')
                _t_int(world['seed'],'world.seed')
                gid = world['source_group_id']
                _need(gid not in descriptors or descriptors[gid] == world, 'SPLIT_WORLD_DESCRIPTOR_CONFLICT')
                descriptors[gid] = world
                # The starting visible state may legitimately be shared; RNG,
                # world/seed and independent-family exposure may not cross splits.
                for key in ('source_group_id','source_family_id','world_sha256','rng_sha256','seed'):
                    token = (key,world[key])
                    _need(token not in exposure or exposure[token]==phase, 'SPLIT_OVERLAP:'+key)
                    exposure[token] = phase
                    if key != 'source_family_id':
                        _need(token not in local, 'SPLIT_DUPLICATE_WORLD:'+key); local.add(token)


def register_search_tournament(recipe: dict) -> dict:
    """Freeze a trainer-only recipe BEFORE outcomes. Performs no search or I/O.

    Both seats and opening/recovery strata are explicit. World descriptors and
    source families are split-checked globally, including aliases across draws.
    Model/provider/algorithm settings and both total-cost caps are immutable.
    Chronology/exposure flags require external custody; sealing is not proof.
    """
    _t_validate_recipe(recipe)
    result = copy.deepcopy(recipe)
    result.update(schema=TOURNAMENT_SCHEMA, recipe_sha256=_sha(recipe))
    return _seal(result)


def _t_registration(registration: dict) -> None:
    _t_bound(registration, TOURNAMENT_SCHEMA)
    raw = {k:v for k,v in registration.items() if k not in ('content_sha256','recipe_sha256')}
    raw['schema'] = TOURNAMENT_RECIPE_SCHEMA
    _t_validate_recipe(raw)
    _need(registration.get('recipe_sha256') == _sha(raw), 'TOURNAMENT_RECIPE_HASH_MISMATCH')


def search_tournament_jobs(registration: dict) -> list[dict]:
    """Discovery-only execution slots for the maintainer's existing searchers.

    No confirmation/final descriptors are passed through this interface. These
    are not actor inputs: the searcher must still give its proposer ONLY the
    actor-visible context, never this evaluator/job envelope.
    """
    import random
    _t_registration(registration)
    jobs = []
    for mode in TOURNAMENT_MODES:
        for draw in registration['draws']:
            for arm in registration['arms']:
                if not arm['available']: continue
                visible_draw = {k:v for k,v in draw.items() if k!='worlds'}
                jobs.append(_seal(dict(schema='farm-v0-regret-tournament-job-v1', phase='discovery',
                    cell_id=mode+':'+draw['draw_id']+':'+arm['arm_id'], mode=mode, stratum=draw['stratum'],
                    draw_id=draw['draw_id'], arm_id=arm['arm_id'],
                    registration_sha256=registration['content_sha256'], learner_draw_sha256=_sha(visible_draw),
                    learner_draw=copy.deepcopy(visible_draw), parent_checkpoint_sha256=registration['parent_checkpoint_sha256'],
                    opponent_sha256=registration['opponent_sha256'], source_pins=copy.deepcopy(registration['source_pins']),
                    worlds=copy.deepcopy(draw['worlds']['discovery']), arm=copy.deepcopy(arm),
                    total_cost_cap=copy.deepcopy(registration['budgets'][mode]), seeds=copy.deepcopy(registration['seeds']),
                    cost_scope='DISCOVERY_PLUS_CONFIRMATION_PLUS_REPLAY_PLUS_SHARED_DRAW_CHARGE')))
    random.Random(registration['seeds']['execution_order']).shuffle(jobs)
    return jobs


def _t_cells(rows: list[dict], jobs: list[dict]) -> dict[str,dict]:
    values,_ = _unique(rows,'cell_id')
    result = {r['cell_id']:r for r in values}
    _need(set(result) == {j['cell_id'] for j in jobs}, 'TOURNAMENT_CELL_SET_MISMATCH')
    return result


def _t_profile(raw: dict, *, registration: dict, draw: dict, job: dict, phase: str,
               used_executions: set[str]) -> tuple[dict,dict]:
    _need(raw.get('evidence_scope') == registration['evidence_scope'], 'TOURNAMENT_PROFILE_SCOPE')
    _need(raw.get('round_id') == job['cell_id']+'/'+phase, 'TOURNAMENT_PROFILE_ID')
    expanded,aggregate = _expand_components(raw)
    _need(not aggregate and expanded.get('coverage',{}).get('executions_complete') is True
          and expanded.get('coverage',{}).get('phase_cost_complete') is True, 'TOURNAMENT_COST_COVERAGE')
    _need(not expanded.get('direct_draws'), 'TOURNAMENT_USE_FROZEN_DRAW_NOT_NEW_SAMPLES')
    if phase=='discovery': _need(not expanded.get('admissions'), 'DISCOVERY_ADMISSION_LEAK')
    report = profile_round(raw)
    events,_ = _unique(expanded.get('executions',[]),'execution_id')
    worlds = {w['source_group_id']:w for w in draw['worlds'][phase]}
    kind = 'INJECTED_FIXTURE' if registration['evidence_scope']=='SYNTHETIC' else 'NATIVE_GAME'
    families = set()
    for event in events:
        identity = event['execution_id']
        _need(identity not in used_executions, 'TOURNAMENT_EXECUTION_REUSE:'+identity)
        used_executions.add(identity)
        _need(event['evidence_kind']==kind, 'TOURNAMENT_EXECUTION_SCOPE')
        group = event.get('source_group_id')
        _need(group in worlds, 'TOURNAMENT_EXECUTION_SPLIT')
        _need(event.get('independent_group_id')==worlds[group]['source_family_id'], 'TOURNAMENT_GROUP_FAMILY')
        _need(event['phase'] == ('replay_verification' if event['is_replay'] else
              'mc_discovery' if phase=='discovery' else 'confirmation'), 'TOURNAMENT_EXECUTION_PHASE')
        if not event['is_replay']: families.add(event['independent_group_id'])
    savings = 0
    cache,_ = _unique(expanded.get('cache_events',[]),'cache_event_id')
    for event in cache:
        if event['outcome']=='HIT':
            # A cache hit is not a universally fixed number of avoided starts.
            if 'avoided_simulator_starts' in event:
                _need(_hash(event.get('cached_receipt_sha256')), 'TOURNAMENT_CACHE_BINDING')
                savings += _t_int(event['avoided_simulator_starts'],'avoided_simulator_starts')
    return report, dict(executions=events, admissions=expanded.get('admissions',[]),
        families=families, cache_savings=savings)


def _t_proposals(row: dict, draw: dict, job: dict) -> dict:
    from farm_neat.training.teachers.v0_regret_budget import candidate_key
    proposals,_ = _unique(row['proposals'],'proposal_id')
    _need(len(proposals) <= job['arm']['settings']['max_proposals'], 'TOURNAMENT_PROPOSAL_CAP')
    reachable = 0; seen = set()
    for proposal in proposals:
        _t_fields(proposal, ('proposal_id','status','paths','producer_receipt_sha256'), 'TOURNAMENT_PROPOSAL_FIELDS')
        _need(_hash(proposal['producer_receipt_sha256']), 'TOURNAMENT_PROPOSAL_SOURCE')
        _need(proposal['status'] in ('RETAINED','DUPLICATE','ILLEGAL','FAILED','CENSORED','REJECTED'), 'TOURNAMENT_PROPOSAL_STATUS')
        if proposal['status'] in ('RETAINED','DUPLICATE'):
            key = candidate_key(proposal['paths'],draw['actor_context_sha256s'])
            _need((key in seen)==(proposal['status']=='DUPLICATE'), 'TOURNAMENT_PROPOSAL_DUPLICATE')
            seen.add(key); reachable += 1
    selected = row['selected']
    if selected is not None:
        _t_fields(selected, ('candidate_id','paths','selection_receipt_sha256s','source_receipt_sha256'),
                  'TOURNAMENT_SELECTION_FIELDS')
        _need(_hash(selected['source_receipt_sha256']), 'TOURNAMENT_SELECTION_SOURCE')
        key = candidate_key(selected['paths'],draw['actor_context_sha256s'])
        _need(key==selected['candidate_id'] and key in seen, 'TOURNAMENT_UNREACHABLE_SELECTION')
        _t_fields(selected['selection_receipt_sha256s'], ('0','1'), 'TOURNAMENT_SELECTION_SEATS')
        _need(all(_hash(v) for v in selected['selection_receipt_sha256s'].values()), 'TOURNAMENT_SELECTION_HASH')
    return dict(proposals=len(proposals),reachable_candidates=reachable,unique_candidates=len(seen))


def freeze_search_tournament(registration: dict, discoveries: list[dict]) -> dict:
    """Lock ALL discovery outcomes before releasing any confirmation job.

    Retains failed/no-winner cells and raw profiles, not only winners. Exact
    proposal Plan+continuation identity reuses candidate_key; prefix legality is
    still the producer's responsibility. Candidate selection is NEVER repeated.
    """
    jobs = search_tournament_jobs(registration); rows = _t_cells(discoveries,jobs)
    draws = {d['draw_id']:d for d in registration['draws']}; used = set()
    for job in jobs:
        row = rows[job['cell_id']]; draw = draws[job['draw_id']]
        _t_fields(row, ('cell_id','job_sha256','profile','cpu_seconds','proposals','selected'),
                  'TOURNAMENT_DISCOVERY_FIELDS', optional=('diagnostics',))
        _need(row['job_sha256']==job['content_sha256'], 'TOURNAMENT_JOB_HASH_MISMATCH')
        _number(row['cpu_seconds'],'cpu_seconds')
        _t_profile(row['profile'],registration=registration,draw=draw,job=job,phase='discovery',used_executions=used)
        _t_proposals(row,draw,job)
    return _seal(dict(schema=TOURNAMENT_FROZEN_SCHEMA, registration_sha256=registration['content_sha256'],
        discoveries=[copy.deepcopy(rows[j['cell_id']]) for j in jobs],
        selection_use='EXISTING_DISCOVERY_SELECTION_ONLY_NO_CONFIRMATION_TUNING'))


def _t_frozen(registration: dict, frozen: dict) -> None:
    _t_bound(frozen,TOURNAMENT_FROZEN_SCHEMA)
    _need(frozen.get('registration_sha256')==registration['content_sha256'], 'TOURNAMENT_FROZEN_REGISTRATION')
    _need(freeze_search_tournament(registration,frozen['discoveries'])==frozen, 'TOURNAMENT_FROZEN_CONTENT')


def _t_admission_sources(row: dict, selected: dict | None, *, registration: dict, draw: dict,
                         details: dict, discovery_details: dict, opportunity_count: int) -> list[dict]:
    """Rerun the actual gate, then join its protocol to the locked tournament cell."""
    sources = row['admission_sources']
    _need(type(sources) is list, 'TOURNAMENT_ADMISSION_SOURCES')
    if selected is None:
        _need(not sources and not details['admissions'] and not details['executions'], 'CONFIRMATION_WITHOUT_SELECTION')
        return []
    from farm_neat.training import v0_regret_admission as A
    by_hash = {}; computed = {}; expected_alpha = registration['family_alpha']/opportunity_count
    # The canonical gate retains BOTH discovery and confirmation outcomes.
    # Charge their original phase ledgers; neither replay nor an exported
    # receipt may manufacture another independent world or a free start.
    executions = {e['execution_id']:e for e in discovery_details['executions']+details['executions']}
    replays = defaultdict(list)
    for event in executions.values():
        if event['is_replay']: replays[event.get('replay_of')].append(event)
    for source in sources:
        _t_fields(source, ('pair','protocol'), 'TOURNAMENT_ADMISSION_SOURCE_FIELDS')
        identity = _sha(source)
        if identity in by_hash: continue
        by_hash[identity] = True
        protocol = source['protocol']
        receipt = A.admit_pair(source['pair'],protocol=protocol)
        _need(receipt['protocol_sha256'] and receipt['test_id'], 'TOURNAMENT_INVALID_ADMISSION_SOURCE')
        _need(receipt['alpha'] is not None and 0 < receipt['alpha'] <= expected_alpha,
              'TOURNAMENT_ADMISSION_ALPHA_ALLOCATION')
        tests = [t for t in protocol['tests'] if t['test_id']==receipt['test_id']]
        _need(len(tests)==1, 'TOURNAMENT_ADMISSION_TEST')
        test = tests[0]
        _need(test['seats']==draw['seats'], 'TOURNAMENT_ADMISSION_SEATS')
        for phase,key in (('discovery','discovery_world_groups'),('confirmation','confirmation_world_groups')):
            expected = [{k:v for k,v in w.items() if k!='source_family_id'} for w in draw['worlds'][phase]]
            _need(test[key]==expected, 'TOURNAMENT_ADMISSION_WORLD_SPLIT')
            for world in draw['worlds'][phase]:
                _need(test['source_families'].get(world['source_group_id'])==world['source_family_id'],
                      'TOURNAMENT_ADMISSION_FAMILY')
        _need(len(test['seat_locks'])==len(draw['seats']), 'TOURNAMENT_ADMISSION_LOCKS')
        for lock in test['seat_locks']:
            seat = str(lock['seat'])
            _need(seat in selected['paths'], 'TOURNAMENT_SELECTION_SEAT')
            expected = dict(actor_context_sha256=draw['actor_context_sha256s'][seat],
                continuation_sha256=draw['continuation_sha256s'][seat],
                parent_checkpoint_sha256=registration['parent_checkpoint_sha256'],
                source_group_id=draw['source_group_id'],direct_draw_id=draw['direct_draw_ids'][seat],
                learner_path_sha256=draw['learner_path_sha256s'][seat],
                challenger_actions_sha256=selected['paths'][seat]['raw_actions_sha256'],
                selection_receipt_sha256=selected['selection_receipt_sha256s'][seat],
                opponent_hash=registration['opponent_sha256'])
            _need(all(lock.get(k)==v for k,v in expected.items()), 'TOURNAMENT_SELECTION_LOCK_MISMATCH')
        # Actions alone cannot authenticate a Plan/continuation summary. Join the
        # exact challenger payload too, using the existing path-binding adapter.
        from farm_neat.training.teachers.v0_regret_budget import path_binding
        records = source['pair'].get('records',[source['pair']])
        for record in records:
            seat = str(record['actor_context']['observation']['player'])
            _need(path_binding(record['challenger'])==selected['paths'][seat], 'TOURNAMENT_SELECTION_PATH_MISMATCH')
        for outcome in receipt.get('retained_outcomes',[]):
            for arm in ('learner','challenger'):
                attempt = outcome[arm+'_attempt_id']
                _need(attempt in executions and not executions[attempt]['is_replay'], 'TOURNAMENT_ADMISSION_START_CHARGE')
                event = executions[attempt]
                _need(not outcome['complete'] or event['status']=='COMPLETE', 'TOURNAMENT_ADMISSION_COMPLETE_CHARGE')
                _need(event['source_group_id']==outcome['source_group_id'], 'TOURNAMENT_ADMISSION_GROUP_CHARGE')
                _need(event['phase']==('mc_discovery' if outcome['phase']=='discovery' else 'confirmation'),
                      'TOURNAMENT_ADMISSION_PHASE_CHARGE')
                if outcome['replay_verified']:
                    _need(any(r['source_group_id']==event['source_group_id'] and r['status']=='COMPLETE'
                              for r in replays[attempt]), 'TOURNAMENT_ADMISSION_REPLAY_CHARGE')
        computed[receipt['content_sha256']] = receipt
    # One opportunity can yield at most one selected, fixed-N test. Duplicate
    # exports are idempotent; multiple looks/protocols cannot allocate new alpha.
    _need(len(computed)<=1, 'TOURNAMENT_MULTIPLE_CONFIRMATION_LOOKS')
    supplied = {a.get('content_sha256'):a for a in details['admissions']}
    _need(supplied==computed, 'TOURNAMENT_ADMISSION_SOURCE_RECEIPT_MISMATCH')
    return list(computed.values())


def summarize_search_tournament(registration: dict, frozen: dict, confirmations: list[dict]) -> dict:
    """Produce all arm rows and separate mode/stratum engineering decisions.

    Inputs are a pre-outcome recipe, the frozen discovery file and per-cell
    non-overlapping confirmation profiles plus actual admission bundles/protocols.
    It reruns profile_round and admit_pair, not claimed summary numerators. Equal
    actual costs are checked PER DRAW, so opposite imbalances cannot cancel.
    No function here runs a game, changes an actor, or opens protected-final data.
    """
    _t_frozen(registration,frozen)
    jobs = search_tournament_jobs(registration)
    discoveries = _t_cells(frozen['discoveries'],jobs); confirms = _t_cells(confirmations,jobs)
    draws = {d['draw_id']:d for d in registration['draws']}
    used = set(); cells = []
    for job in jobs:
        draw = draws[job['draw_id']]; discovery = discoveries[job['cell_id']]; confirm = confirms[job['cell_id']]
        _t_fields(confirm, ('cell_id','discovery_sha256','profile','cpu_seconds','admission_sources'),
                  'TOURNAMENT_CONFIRMATION_FIELDS')
        _need(confirm['discovery_sha256']==frozen['content_sha256'], 'TOURNAMENT_DISCOVERY_LOCK_MISMATCH')
        first,a = _t_profile(discovery['profile'],registration=registration,draw=draw,job=job,
                              phase='discovery',used_executions=used)
        second,b = _t_profile(confirm['profile'],registration=registration,draw=draw,job=job,
                               phase='confirmation',used_executions=used)
        admissions = _t_admission_sources(confirm,discovery['selected'],registration=registration,draw=draw,
                                          details=b,discovery_details=a,opportunity_count=len(jobs))
        useful,_,_,_ = _admissions(admissions)
        # Signed WDL uses -1/0/+1. Divide a paired delta by TWO to report
        # delta(P(win)+.5P(draw)); never call +2 a two-point win-probability gain.
        completed = [r for r in admissions if r.get('mean_wdl_delta') is not None
            and r.get('independent_world_count')==r.get('registered_world_count')
            and r.get('counts',{}).get('confirmation_pairs',0)>0
            and r['counts'].get('complete_pairs')==r['counts']['confirmation_pairs']
            and r['counts'].get('replay_verified_pairs')==r['counts']['confirmation_pairs']
            and r['status'] not in ('INVALID','CENSORED')]
        counts = {k:first['counts'][k]+second['counts'][k] for k in (
            'physical_simulator_starts','fixture_simulator_starts','replay_starts','failed_simulator_starts',
            'cache_hits','cache_misses')}
        wall = math.fsum((draw['collection_seconds'],first['timing']['end_to_end_seconds'],second['timing']['end_to_end_seconds']))
        cpu = _number(discovery['cpu_seconds'],'cpu_seconds')+_number(confirm['cpu_seconds'],'cpu_seconds')
        cells.append(dict(cell_id=job['cell_id'],draw_id=job['draw_id'],mode=job['mode'],stratum=job['stratum'],arm_id=job['arm_id'],
            **_t_proposals(discovery,draw,job), **counts,
            family_ids=sorted(a['families']|b['families']),
            reported_cache_start_savings=a['cache_savings']+b['cache_savings'],
            discovery_winner=int(discovery['selected'] is not None), confirmed_tests=len(completed),
            confirmation_pending=discovery['selected'] is not None and not admissions,
            confirmed_positive_tests=sum(r['mean_wdl_delta']>0 for r in completed),
            admitted_corrections=useful, confirmed_deltas=[r['mean_wdl_delta']/2 for r in completed],
            admitted_expected_wdl_gain=math.fsum(r['mean_wdl_delta']/2 for r in admissions if _admissions([r])[0]),
            admission_statuses=[dict(status=r['status'],reason_codes=r['reason_codes'],content_sha256=r['content_sha256']) for r in admissions],
            total_cost=dict(end_to_end_seconds=wall,search_and_confirmation_cpu_seconds=cpu,shared_draw_charge_seconds=draw['collection_seconds'],
                discovery_seconds=first['timing']['end_to_end_seconds'],confirmation_and_replay_seconds=second['timing']['end_to_end_seconds']),
            profile_sha256s=[first['content_sha256'],second['content_sha256']]))
    rows = []; panels = []
    for mode in TOURNAMENT_MODES:
        cap = registration['budgets'][mode]
        for stratum in TOURNAMENT_STRATA:
            panel_cells = [c for c in cells if c['mode']==mode and c['stratum']==stratum]
            mismatches = []; overages = []
            for draw_id in sorted({c['draw_id'] for c in panel_cells}):
                same = [c for c in panel_cells if c['draw_id']==draw_id]
                for c in same:
                    starts = c['physical_simulator_starts'] if registration['evidence_scope']=='PRODUCTION' else c['fixture_simulator_starts']
                    if starts > cap['physical_start_cap'] or c['total_cost']['end_to_end_seconds'] > cap['wall_seconds_cap']:
                        overages.append(c['cell_id'])
                amounts = [(c['physical_simulator_starts'] if registration['evidence_scope']=='PRODUCTION'
                    else c['fixture_simulator_starts']) if mode=='physical_starts' else c['total_cost']['end_to_end_seconds'] for c in same]
                tolerance = 0 if mode=='physical_starts' else cap['wall_match_tolerance_seconds']
                if max(amounts)-min(amounts)>tolerance:
                    mismatches.append(dict(draw_id=draw_id,minimum=min(amounts),maximum=max(amounts)))
            panel_rows = []
            for arm in registration['arms']:
                selected = [c for c in panel_cells if c['arm_id']==arm['arm_id']]
                out = dict(mode=mode,stratum=stratum,arm_id=arm['arm_id'],status='RECORDED' if arm['available'] else 'UNAVAILABLE',
                    unavailable_reason=arm['unavailable_reason'],draw_count=len(selected))
                for key in ('proposals','reachable_candidates','unique_candidates','physical_simulator_starts','replay_starts',
                    'failed_simulator_starts','cache_hits','cache_misses','reported_cache_start_savings','confirmed_tests',
                    'confirmed_positive_tests','admitted_corrections'):
                    out[key] = sum(c[key] for c in selected) if selected else None
                out['fixture_start_equivalents'] = sum(c['fixture_simulator_starts'] for c in selected) if selected else None
                out['independent_continuation_groups'] = len({f for c in selected for f in c['family_ids']}) if selected else None
                out['no_clear_winner_rate'] = sum(1-c['discovery_winner'] for c in selected)/len(selected) if selected else None
                out['discovery_winner_rate'] = 1-out['no_clear_winner_rate'] if selected else None
                cost = {key:math.fsum(c['total_cost'][key] for c in selected) if selected else None
                    for key in ('end_to_end_seconds','search_and_confirmation_cpu_seconds','shared_draw_charge_seconds','discovery_seconds','confirmation_and_replay_seconds')}
                out['total_cost'] = cost
                deltas = [d for c in selected for d in c['confirmed_deltas']]
                out['paired_expected_wdl_delta'] = math.fsum(deltas)/len(deltas) if deltas else None
                out['expected_delta_observed_tests'] = len(deltas)
                out['expected_delta_scope'] = 'OBSERVED_COMPLETE_FIXED_TESTS_EQUAL_TEST_WEIGHT_NOT_IMPUTED_UNOBSERVED_TESTS'
                wall = cost['end_to_end_seconds']
                out['paired_expected_wdl_delta_per_total_second'] = math.fsum(deltas)/wall if wall and deltas else None
                out['admitted_corrections_per_total_second'] = out['admitted_corrections']/wall if wall else None
                out['admitted_expected_wdl_gain_per_total_second'] = math.fsum(c['admitted_expected_wdl_gain'] for c in selected)/wall if wall else None
                out['admission_statuses'] = [s for c in selected for s in c['admission_statuses']]
                rows.append(out); panel_rows.append(out)
            eligible = [r for r in panel_rows if r['status']=='RECORDED']
            def score(r):
                return (r['admitted_corrections_per_total_second'] or 0., r['paired_expected_wdl_delta_per_total_second'] or 0.)
            control = next(r for r in eligible if r['arm_id']=='flat_fixed_n')
            best = max(eligible,key=score)
            preferred = None
            if overages: decision = 'HOLD_COST_CAP_EXCEEDED'
            elif mismatches: decision = 'HOLD_UNMATCHED_COST'
            elif any(c['confirmation_pending'] for c in panel_cells): decision = 'HOLD_MISSING_CONFIRMATION'
            elif not any(r['admitted_corrections'] for r in eligible): decision = 'NO_ADMITTED_CORRECTIONS'
            elif score(best) <= score(control): decision = 'KEEP_FLAT_CONTROL'
            elif sum(score(r)==score(best) for r in eligible)>1: decision = 'HOLD_TIED_SMART_ARMS'
            else: decision = 'PREFER_SMART_ARM_FOR_PILOT'; preferred = best['arm_id']
            panels.append(dict(mode=mode,stratum=stratum,rule_decision=decision,preferred_arm=preferred,
                cost_matched=not mismatches and not overages,actual_cost_mismatches=mismatches,cap_overages=overages,
                comparison_unit='PER_DRAW_TOTAL_COST_NOT_CAP_OR_PANEL_AVERAGE',
                start_unit='PHYSICAL_NATIVE_STARTS' if registration['evidence_scope']=='PRODUCTION' else 'FIXTURE_START_EQUIVALENTS'))
    return _seal(dict(schema='farm-v0-regret-tournament-report-v1',registration_sha256=registration['content_sha256'],
        discovery_sha256=frozen['content_sha256'],confirmations_sha256=_sha(confirmations),evidence_scope=registration['evidence_scope'],
        rows=rows,panels=panels,cells=cells,selection_rule=TOURNAMENT_RULE,
        decision='SYNTHETIC_ONLY' if registration['evidence_scope']=='SYNTHETIC' else 'REQUIRES_SEARCH_DISABLED_DIRECT_POLICY_GATE',
        actor_training_admissible=False,native_games_executed_by_reporter=0,limitations=list(_T_LIMITS)))


def search_tournament_pilot_matrix(registration: dict, *, frozen: dict | None = None) -> dict:
    """Machine-readable work matrix for the existing V0-REGRET-PILOT task.

    Not a launcher. Preserves the design's flat 176+176 control unchanged; extra
    tournament modes/strata are separately budgeted. No confirmation jobs exist
    until ALL discoveries are frozen, and only the frozen candidate is supplied.
    """
    jobs = search_tournament_jobs(registration); confirmations = []
    draws = {d['draw_id']:d for d in registration['draws']}
    if frozen is not None:
        _t_frozen(registration,frozen)
        by_cell = {r['cell_id']:r for r in frozen['discoveries']}
        for job in jobs:
            row = by_cell[job['cell_id']]
            if row['selected'] is None: continue
            draw = draws[job['draw_id']]
            confirmations.append(dict(cell_id=job['cell_id'],phase='confirmation',
                discovery_sha256=frozen['content_sha256'],**copy.deepcopy(row['selected']),
                worlds=copy.deepcopy(draw['worlds']['confirmation']),seats=draw['seats'],
                admission_alpha_max=registration['family_alpha']/len(jobs),
                admission_consumer='farm_neat.training.v0_regret_admission.admit_pair',
                budget_is_remaining_of=job['total_cost_cap'],new_budget_authorized=False))
    design = Path(__file__).resolve().parents[3]/'project/configs/v0-regret-teacher.json'
    pilot = json.loads(design.read_text())['pilot_recipe']
    return _seal(dict(schema='farm-v0-regret-tournament-pilot-matrix-v1',consumer='V0-REGRET-PILOT',
        registration_sha256=registration['content_sha256'],design_sha256=hashlib.sha256(design.read_bytes()).hexdigest(),
        status='SYNTHETIC_NOT_EXECUTABLE' if registration['evidence_scope']=='SYNTHETIC' else 'MAINTAINER_RUNTIME_AND_CUSTODY_REQUIRED',
        discovery_jobs=jobs,confirmation_jobs=confirmations,unavailable_arms=[copy.deepcopy(a) for a in registration['arms'] if not a['available']],
        flat_control={k:pilot[k] for k in ('learner_draws','discovery_world_groups_per_pair','confirmation_world_groups_per_pair',
            'seats','base_execution_upper_bound','verification_replay_execution_upper_bound','total_simulator_execution_upper_bound')},
        max_optimizer_updates_per_learning_arm=pilot['max_optimizer_updates_per_arm'],
        no_signal='SKIP_TEACHER_FIT_WHEN_NO_EXISTING_GATE_ADMISSIONS',
        direct_rl_compute_controls=[dict(mode=m,stratum=s,total_cost_cap=copy.deepcopy(registration['budgets'][m]),
            ownership='V0-REGRET-PILOT_REGISTERS_GENUINE_DIRECT_RL_ROLLOUT_AND_MONITOR_BUDGETS_SEPARATELY')
            for m in TOURNAMENT_MODES for s in TOURNAMENT_STRATA],
        final_policy_gate=dict(consumer='farm_neat.evaluation.v0_regret_evaluate',search_enabled=False,teacher_prefixes=False,
            objective='TERMINAL_WDL',protected_final_exposure_authorized=False),
        source_adapters=dict(flat_fixed_n='v0_regret_proposals.propose',iterative_fixed='v0_regret_searcher.improve',
            uniform_mc='v0_regret_searcher.estimate_rollouts',adaptive_mc='v0_regret_budget.MonteCarloAllocator',
            external_teacher_mc='v0_regret_proposals.teacher_proposer + estimate_rollouts',
            value_guided_mc='v0_regret_budget.MonteCarloAllocator with frozen value ordering',
            recovery='v0_regret_searcher.search_recovery; exact learner-reached snapshot required'),
        required_runtime_bindings=copy.deepcopy(pilot['required_runtime_bindings']),
        limitations=list(_T_LIMITS)))


class RoundRecorder:
    """Small observer for existing collector/searcher/trainer/evaluator callers.

    All clock reads are CPU host clocks; no device synchronization is inserted.
    Separate threads have separate nesting. No actor state or callback result is
    changed. Events are durable only when the caller publishes finish() output.
    """
    def __init__(self, *, round_id: str, evidence_scope: str, resources: dict,
                 source_pins: dict | None = None, runtime: dict | None = None):
        _need(evidence_scope in SCOPES, 'EVIDENCE_SCOPE')
        self._start = time.perf_counter(); self._lock = threading.Lock()
        self._local = threading.local(); self._active = 0; self._finished = None
        self._receipt = dict(schema=INPUT_SCHEMA, round_id=round_id, evidence_scope=evidence_scope,
            timing_events=[], executions=[], direct_draws=[], cache_events=[], admissions=[],
            resources=copy.deepcopy(resources), source_pins=copy.deepcopy(source_pins or {}),
            runtime=copy.deepcopy(runtime or {}))

    @contextmanager
    def span(self, component: str, phase: str, *, test_id: str | None = None) -> Iterator[dict]:
        _need(component in COMPONENTS and phase in PHASES, 'INVALID_SPAN_CATEGORY')
        stack = getattr(self._local, 'stack', [])
        with self._lock:
            _need(self._finished is None, 'RECORDER_ALREADY_FINISHED')
            event = dict(event_id='span-'+str(len(self._receipt['timing_events'])), component=component,
                phase=phase, start_seconds=time.perf_counter()-self._start,
                parent_id=stack[-1] if stack else None, status='COMPLETE', thread_id=threading.get_ident())
            if test_id is not None: event['test_id'] = test_id
            self._receipt['timing_events'].append(event); self._active += 1
        self._local.stack = stack+[event['event_id']]
        try:
            yield event
        except BaseException:
            event['status'] = 'FAILED'
            raise
        finally:
            with self._lock:
                event['end_seconds'] = time.perf_counter()-self._start
                self._active -= 1
            self._local.stack = stack

    def record(self, collection: str, event: dict) -> None:
        """Append a producer-owned identity, never derive a start from a cache hit."""
        _need(collection in ('executions','direct_draws','cache_events','admissions'), 'UNKNOWN_COLLECTION')
        with self._lock:
            _need(self._finished is None, 'RECORDER_ALREADY_FINISHED')
            self._receipt[collection].append(copy.deepcopy(event))

    def finish(self) -> dict:
        with self._lock:
            _need(not self._active, 'RECORDER_HAS_OPEN_SPANS')
            if self._finished is None:
                self._receipt['end_to_end_seconds'] = time.perf_counter()-self._start
                self._finished = copy.deepcopy(self._receipt)
            return copy.deepcopy(self._finished)


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _wire(value)+b'\n'
    with path.open('xb') as stream:
        stream.write(data)


def cpu_diagnostic(output: Path, *, repetitions: int = 100) -> dict:
    """Measure accounting overhead only. Synthetic timings are not teaching work."""
    _need(type(repetitions) is int and 1 <= repetitions <= 10000, 'DIAGNOSTIC_REPETITIONS')
    output.mkdir(parents=True, exist_ok=False)
    resources = dict(candidate_batch_size=1, cpu_workers=1, device='cpu', peak_device_bytes=None,
        peak_ram_bytes=None, memory_scope='PROCESS_LIFETIME_RUSAGE_HIGH_WATER_NOT_CHILDREN')
    recorder = RoundRecorder(round_id='cpu-accounting-diagnostic', evidence_scope='CPU_DIAGNOSTIC',
        resources=resources, source_pins={'src/farm_neat/training/v0_regret_profile.py':
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    synthetic = dict(schema=INPUT_SCHEMA, round_id='diagnostic-fixture', evidence_scope='SYNTHETIC',
        end_to_end_seconds=10., timing_events=[dict(event_id='a',component='searcher',
            phase='proposal_generation',start_seconds=0.,end_seconds=3.,status='COMPLETE')],
        executions=[], direct_draws=[], cache_events=[], admissions=[], resources=resources)
    cpu = time.process_time()
    with recorder.span('evaluator', 'other'):
        for _ in range(repetitions): profile_round(synthetic)
    raw = recorder.finish()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    raw['resources']['peak_ram_bytes'] = int(peak if sys.platform == 'darwin' else peak*1024)
    raw['diagnostic'] = dict(repetitions=repetitions, cpu_seconds=time.process_time()-cpu,
        workload='PROFILE_ACCOUNTING_ONLY_NOT_TEACHING', python=platform.python_version(),
        native_games=0, optimizer_updates=0, model_calls=0)
    report = profile_round(raw)
    _write_new(output/'EVENTS.json', raw); _write_new(output/'PROFILE.json', report)
    return report


def _json_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        _need(key not in result, 'DUPLICATE_JSON_KEY:'+key)
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('profile', help='Read an existing event receipt; launches no workload')
    p.add_argument('--receipt', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('diagnostic', help='CPU accounting overhead only; no simulator/model/checkpoint')
    p.add_argument('--out', type=Path, required=True); p.add_argument('--repetitions', type=int, default=100)
    for command, fields in (
        ('tournament-register', ('recipe',)),
        ('tournament-freeze', ('registration','discoveries')),
        ('tournament-matrix', ('registration','frozen')),
        ('tournament-report', ('registration','frozen','confirmations'))):
        p = sub.add_parser(command, help='Read-only frozen search-tournament accounting')
        p.add_argument('--out', type=Path, required=True)
        for field in fields:
            p.add_argument('--'+field, type=Path,
                           required=not (command=='tournament-matrix' and field=='frozen'))
    args = parser.parse_args(argv)
    try:
        if args.command == 'profile':
            data = args.receipt.read_bytes(); raw = json.loads(data, object_pairs_hook=_json_object)
            report = profile_round(raw)
            report.pop('content_sha256')
            report['input_file_sha256'] = hashlib.sha256(data).hexdigest(); _seal(report)
            _write_new(args.out, report)
        elif args.command == 'diagnostic':
            report = cpu_diagnostic(args.out, repetitions=args.repetitions)
        else:
            def read_arg(name):
                path = getattr(args, name, None)
                return json.loads(path.read_bytes(), object_pairs_hook=_json_object) if path else None
            if args.command == 'tournament-register':
                report = register_search_tournament(read_arg('recipe'))
            elif args.command == 'tournament-freeze':
                report = freeze_search_tournament(read_arg('registration'), read_arg('discoveries'))
            elif args.command == 'tournament-matrix':
                report = search_tournament_pilot_matrix(read_arg('registration'), frozen=read_arg('frozen'))
            else:
                report = summarize_search_tournament(read_arg('registration'), read_arg('frozen'), read_arg('confirmations'))
            _write_new(args.out, report)
        print(json.dumps(dict(status='WRITTEN', content_sha256=report['content_sha256'],
                              evidence_scope=report.get('evidence_scope','READ_ONLY_TOURNAMENT')), sort_keys=True))
        return 0
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        print(json.dumps(dict(status='REFUSED', error=str(error)), sort_keys=True), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
