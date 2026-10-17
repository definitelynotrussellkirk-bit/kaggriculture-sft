"""Bounded, durable reached-dawn search queue. No games, GPU calls or training.

``make_candidate`` wraps the EXISTING exact v0_regret_dawn snapshot and derives
its recovery actor allowlist; it never invents a second continuation codec.
``SearchQueue`` owns enqueue/deduplication, fair selection and charged leases.
The worker must call charge and receive execute_permission=True BEFORE work.
That permission is issued only on the first durable charge acknowledgment.
Lost acknowledgment is conservative: a started batch is never automatically
repeated, even after expiry. An unstarted expired reservation may be leased anew.

Costs are PREPAID UPPER BOUNDS, not measured consumption. All allocated units are
charged at start (including failed/censored/interrupted batches), with no refunds.
Optional measured_costs on completion are diagnostics and must fit the allocation.
One candidate is one bounded search batch, not one native game or optimizer step.

Custody is an immutable, order-sensitive, hash-linked sequence of event files.
Cooperating local POSIX processes serialize through flock on a stable lock inode;
fsync + exclusive hard-link publication commits each event atomically. Interrupted
*.pending files are retained/reported, never replayed as committed events. No
filesystem scanning daemon, database, scheduler, GPU launcher or trainer exists.
This requires a local filesystem with working flock/link/fsync; network mounts,
uncooperative writers, malicious rollback and hidden producer state are NOT
certified. Hashes bind bytes, not checkpoint authenticity or actual reachability.

The producer attests LEARNER_REACHED or LEGAL_REPLAY with a source receipt; this
queue validates source/actor/continuation joins, not the native reaching trajectory.
Optional enqueue(producer_receipt=bytes) retains sealed V0 producer JSON under
receipts/<raw-sha256>.json before publishing a new v2 origin with exact byte size.
Each fresh fold rechecks the bytes/hash/seal. Historical v1 origins remain
HASH_ONLY_UNRETAINED; source pinning is unchanged (no historical source rebind).
inspect/resume expose private producer rows/counts once per distinct byte object,
separately from per-origin provenance, queue duplicate/result counts and charges.
V1 supports exact NON-OPENING, pre-publication dawn snapshots supported by the
existing codec; arbitrary intraday/opening records fail rather than lose memory.
Private snapshot, controller, policy RNG, source names, outcomes/priority inputs
never appear in actor_inputs(). Actor features intentionally exclude information
excluded by recovery_actor_context; they are not a state-sufficiency theorem.

Source/population integer quotas form a fixed weighted round-robin over their
Cartesian cells. With continually eligible backlog each cell is visited within
one wheel traversal; priority only ranks inside a cell. Finite budget, lease
expiry and uncooperative workers cannot promise infinite service. There is no
automatic quota adaptation or teacher authority. Completed exact matches retain
proposal + receipt; approximate neighbors return ONLY untrusted proposal actions,
never inherited value/proof. Downstream prefix replay/WDL admission stays mandatory.

CLI (all operations are finite; identity/source registration is immutable)::

    python -m farm_neat.training.v0_search_queue schema
    python -m farm_neat.training.v0_search_queue init --queue Q --config CONFIG.json
    python -m farm_neat.training.v0_search_queue enqueue --queue Q --identity ID.json \
        --candidate C.json --request-id enqueue-1 --now-ms 0
    python -m farm_neat.training.v0_search_queue lease --queue Q --identity ID.json \
        --worker worker-1 --ttl-ms 60000 --request-id lease-1 --now-ms 1
    python -m farm_neat.training.v0_search_queue charge --queue Q --identity ID.json \
        --worker worker-1 --lease-id LEASE --request-id charge-1 --now-ms 2

Persist a request ID before calling. Retrying the IDENTICAL command is idempotent
but never reissues execution permission; changing its arguments is refused. Time
is an explicit nondecreasing integer millisecond clock shared by workers, not
an ambient clock sampled during replay. Production callers supply UTC epoch ms.
A duplicate request is recognized before checking the present clock, so a delayed
acknowledgment retry remains safe. Read-only inspect does not expire leases; use
resume (or a new successful mutating command) to commit expiry transitions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

SCHEMA = 'farm-v0-search-queue-v1'
IDENTITY_SCHEMA = SCHEMA + '-identity'
CONFIG_SCHEMA = SCHEMA + '-config'
CANDIDATE_SCHEMA = SCHEMA + '-candidate'
ORIGIN_SCHEMA = SCHEMA + '-origin'  # historical hash-only attestation
RECEIPT_ORIGIN_SCHEMA = 'farm-v0-search-queue-v2-origin'
EVENT_SCHEMA = SCHEMA + '-event'
RESULT_SCHEMA = SCHEMA + '-result'
REASONS = ('loss_recovery', 'uncertainty', 'high_confidence_regret', 'novelty',
           'known_winner_mutation', 'adversarial_failure')
COSTS = ('candidate_attempts', 'native_starts', 'cpu_milliseconds', 'gpu_milliseconds')
STATUSES = ('READY', 'LEASED', 'STARTED', 'COMPLETE', 'CENSORED')
MAX_RECORD_BYTES = 4 * 1024 * 1024
MAX_STORE_BYTES = 256 * 1024 * 1024
MAX_EVENTS = 10000
MAX_CANDIDATES = 2048
MAX_TTL_MS = 24 * 60 * 60 * 1000
ROOT = Path(__file__).resolve().parents[3]


class QueueError(ValueError):
    """Stable refusal code/location, without dumping actor or evaluator data."""
    def __init__(self, code: str, at: str = '$'):
        self.code, self.at = code, at
        super().__init__(f'{code}: {at}')


def _need(ok, code, at='$'):
    if not ok:
        raise QueueError(code, at)


def _fields(value, keys, at='$'):
    _need(type(value) is dict and set(value) == set(keys), 'FIELDS_MISMATCH', at)


def _integer(value, at, low=0, high=10**15):
    _need(type(value) is int and low <= value <= high, 'INTEGER_RANGE', at)
    return value


def _text(value, at):
    _need(type(value) is str and 0 < len(value.strip()) <= 256, 'INVALID_TEXT', at)


def _hash(value, at):
    _need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None,
          'INVALID_HASH', at)


def _safe(value):
    stack, count = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        count += 1
        _need(depth <= 80 and count <= 300000, 'JSON_LIMIT')
        if item is None or type(item) in (str, bool):
            continue
        if type(item) is int:
            _need(item.bit_length() <= 4096, 'JSON_LIMIT')
        elif type(item) is float:
            _need(math.isfinite(item), 'NONFINITE_JSON')
        elif type(item) is dict:
            _need(all(type(k) is str for k in item), 'UNSAFE_JSON')
            stack.extend((v, depth + 1) for v in item.values())
        elif type(item) is list:
            stack.extend((v, depth + 1) for v in item)
        else:
            raise QueueError('UNSAFE_JSON')


def wire(value: Any) -> str:
    """Order-sensitive native wire; NEVER sort engine/controller mappings."""
    _safe(value)
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(wire(value).encode()).hexdigest()


def bind(value: dict) -> dict:
    _need(type(value) is dict, 'INVALID_OBJECT')
    body = copy.deepcopy({k: v for k, v in value.items() if k != 'content_sha256'})
    return {**body, 'content_sha256': digest(body)}


def _bound(value, schema):
    _safe(value)
    _need(type(value) is dict and value.get('schema') == schema, 'UNSUPPORTED_SCHEMA')
    _hash(value.get('content_sha256'), 'content_sha256')
    _need(value['content_sha256'] == bind(value)['content_sha256'], 'CONTENT_HASH_MISMATCH')


def source_pins() -> dict:
    """Existing semantic closures plus this queue and its budget input; no weights."""
    from farm_neat.rollout import v0_regret_dawn as D
    from farm_neat.training import v0_regret_records as R
    paths = ('src/farm_neat/training/v0_search_queue.py',
             'src/farm_neat/training/teachers/v0_regret_budget.py')
    return dict(sorted({**R.source_pins(), **D.source_pins(),
        **{p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}}.items()))


def make_identity(snapshot: dict, *, checkpoint_sha256: str) -> dict:
    """Bind a caller-verified checkpoint and the exact existing policy identity.

    The checkpoint hash is an explicit producer assertion, not a weight-file audit.
    ScorerPolicy owns its declared (possibly richer) checkpoint/source identity.
    """
    value = bind(dict(schema=IDENTITY_SCHEMA,
                      policy_sha256=digest(snapshot['actor_policy']['identity']),
                      checkpoint_sha256=checkpoint_sha256, source_pins=source_pins()))
    _identity(value)
    return value


def _identity(value):
    _bound(value, IDENTITY_SCHEMA)
    _fields(value, ('schema', 'policy_sha256', 'checkpoint_sha256', 'source_pins', 'content_sha256'))
    for key in ('policy_sha256', 'checkpoint_sha256'):
        _hash(value[key], key)
    _need(value['source_pins'] == source_pins(), 'STALE_SOURCE')


def _costs(value, *, attempt=False):
    _fields(value, COSTS, 'budget')
    for name in COSTS:
        _integer(value[name], name, low=1 if attempt and name == 'candidate_attempts' else 0)
    return value


def _zero():
    return dict.fromkeys(COSTS, 0)


def _add(left, right, sign=1):
    return {key: left[key] + sign * right[key] for key in COSTS}


def _fits(cost, remaining):
    return all(cost[k] <= remaining[k] for k in COSTS)


def make_config(identity: dict, *, budget: dict, source_quotas: dict, populations: dict,
                priority_weights: dict, evidence_kind: str, max_candidates=512,
                max_events=4096, max_bytes=128 * 1024 * 1024) -> dict:
    config = bind(dict(schema=CONFIG_SCHEMA, identity=identity, budget=budget,
        source_quotas=source_quotas, populations=populations, priority_weights=priority_weights,
        evidence_kind=evidence_kind, max_candidates=max_candidates, max_events=max_events,
        max_bytes=max_bytes))
    return validate_config(config)


def validate_config(value: dict) -> dict:
    _bound(value, CONFIG_SCHEMA)
    _fields(value, ('schema', 'identity', 'budget', 'source_quotas', 'populations',
                   'priority_weights', 'evidence_kind', 'max_candidates', 'max_events',
                   'max_bytes', 'content_sha256'))
    _identity(value['identity']); _costs(value['budget'], attempt=True)
    _need(value['evidence_kind'] in ('INJECTED_FIXTURE', 'NATIVE_GAME'), 'EVIDENCE_KIND')
    _fields(value['source_quotas'], REASONS, 'source_quotas')
    _fields(value['priority_weights'], REASONS, 'priority_weights')
    for reason in REASONS:
        _integer(value['source_quotas'][reason], 'source_quotas.' + reason, 1, 32)
        _integer(value['priority_weights'][reason], 'priority_weights.' + reason, 0, 10**6)
    populations = value['populations']
    _need(type(populations) is dict and 1 <= len(populations) <= 32, 'POPULATIONS')
    for name, quota in populations.items():
        _text(name, 'population'); _integer(quota, 'population_quota', 1, 32)
    _need(sum(value['source_quotas'].values()) * sum(populations.values()) <= 4096, 'QUOTA_WHEEL_LIMIT')
    _integer(value['max_candidates'], 'max_candidates', 1, MAX_CANDIDATES)
    _integer(value['max_events'], 'max_events', 4, MAX_EVENTS)
    _integer(value['max_bytes'], 'max_bytes', MAX_RECORD_BYTES, MAX_STORE_BYTES)
    return copy.deepcopy(value)


def _snapshot_actor(snapshot, identity):
    from farm_neat.rollout import v0_regret_dawn as D
    _identity(identity)
    try:
        actor = D.recovery_actor_context(snapshot)  # validates full exact snapshot; no steps
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise QueueError('INVALID_DAWN_SNAPSHOT') from exc
    _need(digest(snapshot['actor_policy']['identity']) == identity['policy_sha256'], 'STALE_POLICY')
    return actor


def _exact(snapshot, actor, identity):
    return dict(identity=identity, actor_context=actor, snapshot=snapshot)


def _fingerprint(snapshot, actor, identity):
    # Separate hook enables deliberate collision tests without breaking event hashing.
    return digest(_exact(snapshot, actor, identity))


def make_candidate(snapshot: dict, *, identity: dict, origin: dict, reason: str,
                   population: str, priority_inputs: dict, budget: dict) -> dict:
    actor = _snapshot_actor(snapshot, identity)
    candidate = bind(dict(schema=CANDIDATE_SCHEMA,
        fingerprint=_fingerprint(snapshot, actor, identity), identity=identity,
        actor_context=actor, snapshot=snapshot, origin=origin, reason=reason,
        population=population, priority_inputs=priority_inputs, budget=budget))
    return validate_candidate(candidate)


def validate_candidate(value: dict) -> dict:
    _bound(value, CANDIDATE_SCHEMA)
    _fields(value, ('schema', 'fingerprint', 'identity', 'actor_context', 'snapshot',
                   'origin', 'reason', 'population', 'priority_inputs', 'budget', 'content_sha256'))
    actor = _snapshot_actor(value['snapshot'], value['identity'])
    _need(wire(actor) == wire(value['actor_context']), 'ACTOR_INPUT_MISMATCH')
    _hash(value['fingerprint'], 'fingerprint')
    _need(value['fingerprint'] == _fingerprint(value['snapshot'], actor, value['identity']),
          'FINGERPRINT_MISMATCH')
    origin = value['origin']
    _need(type(origin) is dict, 'REACHED_ORIGIN_REQUIRED', 'origin')
    retained = origin.get('schema') == RECEIPT_ORIGIN_SCHEMA
    _fields(origin, ('schema', 'kind', 'evidence_kind', 'split', 'source_group_id', 'source_family_id',
                     'replay_id', 'receipt_sha256', 'snapshot_sha256', 'checkpoint_sha256')
                     + (('receipt_nbytes',) if retained else ()), 'origin')
    if retained:
        _integer(origin['receipt_nbytes'], 'origin.receipt_nbytes', 1, MAX_RECORD_BYTES)
    _need(origin['schema'] in (ORIGIN_SCHEMA, RECEIPT_ORIGIN_SCHEMA)
          and origin['kind'] in ('LEARNER_REACHED', 'LEGAL_REPLAY'),
          'REACHED_ORIGIN_REQUIRED')
    _need(origin['evidence_kind'] in ('INJECTED_FIXTURE', 'NATIVE_GAME'), 'EVIDENCE_KIND')
    _need(origin['split'] == 'discovery', 'PROTECTED_SOURCE')
    for name in ('source_group_id', 'source_family_id', 'replay_id'):
        _text(origin[name], 'origin.' + name)
    for name in ('receipt_sha256', 'snapshot_sha256', 'checkpoint_sha256'):
        _hash(origin[name], 'origin.' + name)
    _need(origin['snapshot_sha256'] == value['snapshot']['content_sha256'], 'ORIGIN_STATE_MISMATCH')
    _need(origin['checkpoint_sha256'] == value['identity']['checkpoint_sha256'], 'ORIGIN_POLICY_MISMATCH')
    _need(value['reason'] in REASONS, 'UNKNOWN_REASON')
    _text(value['population'], 'population')
    _fields(value['priority_inputs'], REASONS, 'priority_inputs')
    for reason, number in value['priority_inputs'].items():
        _integer(number, 'priority_inputs.' + reason, 0, 10**6)
    _costs(value['budget'], attempt=True)
    _need(len(wire(value).encode()) < MAX_RECORD_BYTES - 8192, 'JSON_LIMIT')
    return copy.deepcopy(value)


def actor_inputs(candidate: dict) -> dict:
    """ONLY the existing recovery actor payload. Never hand a queue row to a model."""
    return validate_candidate(candidate)['actor_context']


def _source_key(candidate):
    origin, snapshot = candidate['origin'], candidate['snapshot']
    return (origin['receipt_sha256'], origin['replay_id'], snapshot['seat'], snapshot['step'])


def _same_candidate(left, right):
    return wire(_exact(left['snapshot'], left['actor_context'], left['identity'])) == wire(
        _exact(right['snapshot'], right['actor_context'], right['identity']))


def _wheel(config):
    return [(reason, population)
        for reason in REASONS
        for population in sorted(config['populations'])
        for _ in range(config['source_quotas'][reason] * config['populations'][population])]


def _state(config):
    return dict(config=config, rows={}, source_index={}, requests={}, charged=_zero(),
                producer_receipts={}, receipt_store_bytes=0,
                enqueue_counts=dict(ENQUEUED=0, DUPLICATE=0),
                cursor=0, now_ms=0, sequence=0, head=None, bytes=0)


def _reserved(state):
    out = _zero()
    for row in state['rows'].values():
        if row['status'] == 'LEASED':
            out = _add(out, row['candidate']['budget'])
    return out


def _remaining(state):
    return _add(_add(state['config']['budget'], state['charged'], -1), _reserved(state), -1)


def _producer_custody(state):
    """Private diagnostic projection; count each retained byte object only once.

    Root attempts are the producer's accounting surface. Nested provider receipts
    stay in the original bytes, not recursively counted a second time. No attempt
    count, status, producer counter or origin is a queue budget debit or actor input.
    """
    counts, origins = {}, []
    for receipt in state['producer_receipts'].values():
        for status, count in receipt['attempt_status_counts'].items():
            counts[status] = counts.get(status, 0) + count
    for key, row in state['rows'].items():
        for origin in row['sources']:
            retained = origin['schema'] == RECEIPT_ORIGIN_SCHEMA
            origins.append(dict(candidate_id=key, origin=copy.deepcopy(origin),
                custody='RETAINED' if retained else 'HASH_ONLY_UNRETAINED'))
    retained_count = sum(r['custody'] == 'RETAINED' for r in origins)
    return dict(retained_receipt_count=len(state['producer_receipts']),
        retained_attempt_count=sum(counts.values()), attempt_status_counts=counts,
        retained_origin_count=retained_count, hash_only_origin_count=len(origins) - retained_count,
        origins=origins, receipts=copy.deepcopy(state['producer_receipts']),
        count_semantics='DISTINCT_RECEIPT_BYTES_ROOT_ATTEMPTS_ONLY; HASH_ONLY_ATTEMPTS_UNKNOWN')


def _summary(state):
    rows = {}
    result_counts = dict(SOLVED=0, FAILED=0, EXHAUSTED=0)
    counts = dict.fromkeys(STATUSES, 0)
    for key, row in state['rows'].items():
        status = row['status']; counts[status] += 1
        candidate = row['candidate']
        if row['result'] is not None:
            result_counts[row['result']['status']] += 1
        rows[key] = dict(status=status, reason=candidate['reason'], population=candidate['population'],
            source_group_id=candidate['origin']['source_group_id'], source_count=len(row['sources']),
            remaining_budget=copy.deepcopy(candidate['budget']) if status == 'READY' else _zero(),
            lease_id=None if row['lease'] is None else row['lease']['lease_id'],
            worker=None if row['lease'] is None else row['lease']['worker'],
            result_status=None if row['result'] is None else row['result']['status'],
            censor_reason=row['censor_reason'])
    return dict(schema=SCHEMA + '-summary', counts=counts, candidates=rows,
        producer_custody=_producer_custody(state),
        enqueue_counts=copy.deepcopy(state['enqueue_counts']), result_counts=result_counts,
        charged_budget=copy.deepcopy(state['charged']), reserved_budget=_reserved(state),
        remaining_budget=_remaining(state), sequence=state['sequence'], head_sha256=state['head'],
        clock_ms=state['now_ms'], teacher_authority=False, execute_permission=False)


def _expire(state, now):
    expired = []
    for key, row in state['rows'].items():
        if row['status'] in ('LEASED', 'STARTED') and row['lease']['expires_ms'] <= now:
            if row['status'] == 'LEASED':
                row['status'] = 'READY'
            else:
                row['status'], row['censor_reason'] = 'CENSORED', 'STARTED_LEASE_EXPIRED'
            expired.append(dict(candidate_id=key, lease_id=row['lease']['lease_id'], status=row['status']))
    return expired


def _owned(state, payload):
    _text(payload['worker'], 'worker'); _hash(payload['lease_id'], 'lease_id')
    for row in state['rows'].values():
        lease = row['lease']
        if (row['status'] in ('LEASED', 'STARTED') and lease is not None
                and lease['lease_id'] == payload['lease_id'] and lease['worker'] == payload['worker']):
            return row
    raise QueueError('LEASE_NOT_OWNED')


def _result(value, row):
    _fields(value, ('schema', 'status', 'raw_actions', 'receipt_sha256', 'measured_costs'), 'result')
    _need(value['schema'] == RESULT_SCHEMA, 'UNSUPPORTED_SCHEMA')
    _need(value['status'] in ('SOLVED', 'FAILED', 'EXHAUSTED'), 'RESULT_STATUS')
    _hash(value['receipt_sha256'], 'result.receipt_sha256')
    if value['measured_costs'] is not None:
        _costs(value['measured_costs'])
        _need(_fits(value['measured_costs'], row['candidate']['budget']), 'MEASUREMENT_EXCEEDS_ALLOCATION')
    if value['status'] == 'SOLVED':
        from farm_neat.rollout import v0_regret_dawn as D
        snapshot = row['candidate']['snapshot']
        try:
            D.propose_dawn(snapshot, value['raw_actions'], continuation=snapshot['continuation'])
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise QueueError('INVALID_PROPOSAL') from exc
    else:
        _need(value['raw_actions'] is None, 'FAILED_RESULT_HAS_PROPOSAL')
    return copy.deepcopy(value)


PAYLOAD_FIELDS = {
    'enqueue': ('now_ms', 'candidate'),
    'lease': ('now_ms', 'worker', 'ttl_ms'),
    'charge': ('now_ms', 'worker', 'lease_id'),
    'complete': ('now_ms', 'worker', 'lease_id', 'result'),
    'censor': ('now_ms', 'worker', 'lease_id', 'reason'),
    'resume': ('now_ms',),
}


def _apply(state, command, payload, sequence, *, candidate_validator=None, receipt_loader=None):
    """Deterministic single-event fold. Failed calls discard this in-memory state."""
    _need(command in PAYLOAD_FIELDS, 'UNKNOWN_COMMAND')
    _fields(payload, PAYLOAD_FIELDS[command], 'payload')
    now = _integer(payload['now_ms'], 'now_ms')
    _need(now >= state['now_ms'], 'CLOCK_REGRESSION')
    expired = _expire(state, now)
    state['now_ms'] = now
    config = state['config']
    out = dict(teacher_authority=False, execute_permission=False, expired=expired)
    if command == 'enqueue':
        candidate = (candidate_validator or validate_candidate)(payload['candidate'])
        _need(wire(candidate['identity']) == wire(config['identity']), 'STALE_IDENTITY')
        _need(candidate['origin']['evidence_kind'] == config['evidence_kind'], 'EVIDENCE_KIND')
        _need(candidate['population'] in config['populations'], 'UNREGISTERED_POPULATION')
        _need(_fits(candidate['budget'], config['budget']), 'CANDIDATE_EXCEEDS_TOTAL_BUDGET')
        key, origin = candidate['fingerprint'], candidate['origin']
        source_key = _source_key(candidate)
        if source_key in state['source_index']:
            _need(state['source_index'][source_key] == key, 'SOURCE_REPLAY_CONFLICT')
        if key in state['rows']:
            row = state['rows'][key]
            _need(_same_candidate(row['candidate'], candidate), 'HASH_COLLISION')
            old = row['candidate']['origin']
            _need(all(old[n] == origin[n] for n in ('source_group_id', 'source_family_id')),
                  'SOURCE_GROUP_CONFLICT')
            # Origin field ordering is not native world/controller ordering.
            if origin not in row['sources']:
                row['sources'].append(origin)
            status = 'DUPLICATE'
        else:
            _need(len(state['rows']) < config['max_candidates'], 'CANDIDATE_LIMIT')
            state['rows'][key] = dict(candidate=candidate, sources=[origin], status='READY',
                                     enqueue_sequence=sequence, lease=None, result=None, censor_reason=None)
            status = 'ENQUEUED'
        if origin['schema'] == RECEIPT_ORIGIN_SCHEMA:
            sha256 = origin['receipt_sha256']
            if sha256 not in state['producer_receipts']:
                _need(receipt_loader is not None, 'PRODUCER_RECEIPT_REQUIRED')
                state['producer_receipts'][sha256] = receipt_loader(origin)
            _need(state['producer_receipts'][sha256]['nbytes'] == origin['receipt_nbytes'],
                  'PRODUCER_RECEIPT_SIZE_MISMATCH', 'origin.receipt_nbytes')
        state['source_index'][source_key] = key
        state['enqueue_counts'][status] += 1
        out.update(status=status, candidate_id=key)
    elif command == 'lease':
        _text(payload['worker'], 'worker')
        _integer(payload['ttl_ms'], 'ttl_ms', 1, MAX_TTL_MS)
        wheel, available = _wheel(config), _remaining(state)
        selected = None
        for offset in range(len(wheel)):
            position = (state['cursor'] + offset) % len(wheel)
            reason, population = wheel[position]
            eligible = [(key, row) for key, row in state['rows'].items()
                        if row['status'] == 'READY' and row['candidate']['reason'] == reason
                        and row['candidate']['population'] == population
                        and _fits(row['candidate']['budget'], available)]
            if eligible:
                def rank(item):
                    key, row = item
                    score = sum(row['candidate']['priority_inputs'][r] * config['priority_weights'][r]
                                for r in REASONS)
                    return (-score, row['enqueue_sequence'], key)
                key, row = min(eligible, key=rank)
                state['cursor'] = (position + 1) % len(wheel)
                selected = key, row
                break
        if selected is None:
            out.update(status='EMPTY', remaining_budget=available)
        else:
            key, row = selected
            lease_id = digest(dict(candidate_id=key, worker=payload['worker'],
                                   sequence=sequence, expires_ms=now + payload['ttl_ms']))
            row['lease'] = dict(lease_id=lease_id, worker=payload['worker'],
                                expires_ms=now + payload['ttl_ms'])
            row['status'] = 'LEASED'
            c = row['candidate']
            out.update(status='LEASED', candidate_id=key, **row['lease'], reason=c['reason'],
                       population=c['population'], allocated_budget=copy.deepcopy(c['budget']))
    elif command in ('charge', 'complete', 'censor'):
        row = _owned(state, payload)
        if command == 'charge':
            _need(row['status'] == 'LEASED', 'ALREADY_STARTED')
            # Debit the entire bounded allocation BEFORE exposing execution permission.
            state['charged'] = _add(state['charged'], row['candidate']['budget'])
            row['status'] = 'STARTED'
            out.update(status='STARTED', execute_permission=True, charged_budget=copy.deepcopy(state['charged']))
        elif command == 'complete':
            _need(row['status'] == 'STARTED', 'NOT_STARTED')
            row['result'] = _result(payload['result'], row)
            row['status'] = 'COMPLETE'
            out.update(status='COMPLETE', result_status=row['result']['status'])
        else:
            _text(payload['reason'], 'censor_reason')
            row['status'], row['censor_reason'] = 'CENSORED', payload['reason']
            out.update(status='CENSORED', charged_budget=copy.deepcopy(state['charged']))
        out.update(candidate_id=row['candidate']['fingerprint'], lease_id=payload['lease_id'])
    else:
        out.update(_summary(state), expired=expired)
    return out


def _no_symlinks(path):
    _need(not any(p.is_symlink() for p in (path, *path.parents)), 'SYMLINK')


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(directory: Path, sequence: int, data: bytes):
    """Commit once. A crash may leave pending bytes; NEVER truncate/repair history."""
    _need(len(data) <= MAX_RECORD_BYTES, 'JSON_LIMIT')
    target = directory / f'{sequence:012d}.json'
    fd, pending = tempfile.mkstemp(prefix=f'.{sequence:012d}-', suffix='.pending', dir=directory)
    # Keep partial/full pending bytes on ANY exception for forensic recovery.
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, target)  # exclusive publication; never overwrite an event
    _sync_dir(directory)
    os.unlink(pending)
    _sync_dir(directory)


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def read_json(path: str | Path) -> dict:
    path = Path(path)
    _no_symlinks(path)
    try:
        with path.open('rb') as stream:
            raw = stream.read(MAX_RECORD_BYTES + 1)
        _need(len(raw) <= MAX_RECORD_BYTES, 'JSON_LIMIT')
        result = json.loads(raw, object_pairs_hook=_object_pairs)
        _safe(result)
        _need(type(result) is dict, 'INVALID_OBJECT')
        return result
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise QueueError('INVALID_JSON') from exc


def _producer_receipt(raw: bytes) -> dict:
    """Validate V0 canonical JSON seals without executing any producer code.

    Existing pools use pool_sha256; provider receipts use content_sha256. These
    are SORTED compact JSON seals (unlike the queue's order-sensitive event wire).
    Exact input bytes, including whitespace/order, have their own SHA256 in the
    versioned origin. Receipt authenticity/reachability is still a producer claim.
    """
    _need(type(raw) is bytes, 'PRODUCER_RECEIPT_BYTES_REQUIRED')
    _need(0 < len(raw) <= MAX_RECORD_BYTES, 'PRODUCER_RECEIPT_SIZE_LIMIT')
    try:
        value = json.loads(raw, object_pairs_hook=_object_pairs)
        _safe(value)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise QueueError('PRODUCER_RECEIPT_INVALID_JSON') from exc
    _need(type(value) is dict, 'PRODUCER_RECEIPT_OBJECT_REQUIRED')
    _text(value.get('schema'), 'producer_receipt.schema')
    seals = [k for k in ('content_sha256', 'pool_sha256') if k in value]
    _need(len(seals) == 1, 'PRODUCER_RECEIPT_SEAL_REQUIRED')
    seal = seals[0]
    _hash(value[seal], 'producer_receipt.' + seal)
    from farm_neat.training import v0_regret_records as R
    expected = R.canonical_sha256({k: v for k, v in value.items() if k != seal})
    _need(value[seal] == expected, 'PRODUCER_RECEIPT_SEAL_MISMATCH')
    attempts = value.get('attempts')
    _need(type(attempts) is list, 'PRODUCER_ATTEMPTS_REQUIRED')
    seen, counts = set(), {}
    for index, row in enumerate(attempts):
        at = f'producer_receipt.attempts[{index}]'
        _need(type(row) is dict, 'PRODUCER_ATTEMPT_INVALID', at)
        attempt_id = row.get('attempt_id')
        if type(attempt_id) is int:
            _integer(attempt_id, at + '.attempt_id')
        else:
            _text(attempt_id, at + '.attempt_id')
        _need(attempt_id not in seen, 'PRODUCER_ATTEMPT_ID_DUPLICATE', at)
        seen.add(attempt_id)
        _text(row.get('status'), at + '.status')
        counts[row['status']] = counts.get(row['status'], 0) + 1
    return dict(schema=value['schema'], nbytes=len(raw), seal_field=seal,
                attempts=attempts, attempt_status_counts=counts)


def read_producer_receipt(path: str | Path) -> bytes:
    """Bounded, data-only bytes for enqueue's producer_receipt keyword/CLI."""
    path = Path(path)
    _no_symlinks(path)
    at = 'receipts/' + path.name
    try:
        # O_NONBLOCK prevents a substituted FIFO from blocking before fstat.
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            _need(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'PRODUCER_RECEIPT_NOT_FILE', at)
            raw = stream.read(MAX_RECORD_BYTES + 1)
    except FileNotFoundError as exc:
        raise QueueError('PRODUCER_RECEIPT_MISSING', at) from exc
    except OSError as exc:
        raise QueueError('PRODUCER_RECEIPT_UNREADABLE', at) from exc
    _need(len(raw) <= MAX_RECORD_BYTES, 'PRODUCER_RECEIPT_SIZE_LIMIT', at)
    return raw


def _verify_receipt_bytes(raw, origin):
    at = 'receipts/' + origin['receipt_sha256'] + '.json'
    _need(len(raw) == origin['receipt_nbytes'], 'PRODUCER_RECEIPT_SIZE_MISMATCH', at)
    _need(hashlib.sha256(raw).hexdigest() == origin['receipt_sha256'],
          'PRODUCER_RECEIPT_HASH_MISMATCH', at)


def _publish_receipt(directory: Path, origin: dict, raw: bytes):
    """Same exclusive/fsynced publication as events; never overwrite an object."""
    target = directory / (origin['receipt_sha256'] + '.json')
    _no_symlinks(target)
    if target.exists():
        existing = read_producer_receipt(target)
        _verify_receipt_bytes(existing, origin)
        _need(existing == raw, 'PRODUCER_RECEIPT_HASH_COLLISION')
        _sync_dir(directory)  # also finish an earlier lost acknowledgment
        return
    fd, pending = tempfile.mkstemp(prefix='.' + origin['receipt_sha256'] + '-',
                                   suffix='.pending', dir=directory)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, target)
    _sync_dir(directory)
    os.unlink(pending)
    _sync_dir(directory)


class SearchQueue:
    """One immutable-policy, bounded local queue; mutating methods hold one lock."""

    def __init__(self, path: str | Path, identity: dict):
        # Do NOT resolve symlinks away before checking them.
        self.path = Path(os.path.abspath(path))
        self.identity = copy.deepcopy(identity)
        self._validated = {}  # bounded exact-byte validation memo, never persisted/features
        self._validated_bytes = 0

    @classmethod
    def create(cls, path: str | Path, config: dict) -> 'SearchQueue':
        config = validate_config(config)
        queue = cls(path, config['identity'])
        _no_symlinks(queue.path)
        queue.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        with queue._lock():
            events = queue.path / 'events'
            _no_symlinks(events)
            events.mkdir(exist_ok=True, mode=0o700)
            _sync_dir(queue.path); _sync_dir(queue.path.parent)
            if any(events.glob('*.json')):
                state, _, _ = queue._load()
                _need(wire(state['config']) == wire(config), 'CONFIG_CONFLICT')
                return queue
            event = bind(dict(schema=EVENT_SCHEMA, sequence=0, previous_event_sha256=None,
                              request_id='__init__', command='init', payload=config))
            _publish(events, 0, (wire(event) + '\n').encode())
        return queue

    @contextmanager
    def _lock(self):
        _no_symlinks(self.path)
        _need(self.path.is_dir(), 'QUEUE_NOT_INITIALIZED')
        lock = self.path / '.lock'
        _no_symlinks(lock)
        fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _checked_candidate(self, value):
        """Memoize semantic validation, NOT identity: every event byte is rehashed.

        _load checks the complete runtime source closure before this is called.
        External actor/opponent codec files are checked again even on a cache hit.
        At most 64 entries / 8 MiB of serialized input; no cross-process cache and
        no trusted hash-only hits. A fresh process performs all codec validation.
        """
        encoded = wire(value)
        key = value.get('content_sha256') if type(value) is dict else None
        cached = self._validated.get(key)
        if cached is not None and cached[0] == encoded:
            from farm_neat.rollout import v0_regret_dawn as D
            try:
                for name in ('actor_policy', 'opponent_policy'):
                    D._check_identity(value['snapshot'][name]['identity'])
            except (ValueError, KeyError, TypeError, OSError) as exc:
                raise QueueError('STALE_POLICY_SOURCE') from exc
            return copy.deepcopy(cached[1])
        checked = validate_candidate(value)
        if key in self._validated:
            self._validated_bytes -= len(self._validated.pop(key)[0])
        self._validated[key] = (encoded, checked)
        self._validated_bytes += len(encoded)
        while len(self._validated) > 64 or self._validated_bytes > 8 * 1024 * 1024:
            self._validated_bytes -= len(self._validated.pop(next(iter(self._validated)))[0])
        return copy.deepcopy(checked)

    def _receipt_inventory(self):
        """Account for orphan/pending bytes too; never delete/recover them silently."""
        directory = self.path / 'receipts'
        _no_symlinks(directory)
        if not directory.exists():
            return {}
        _need(directory.is_dir(), 'PRODUCER_RECEIPT_STORE_NOT_DIRECTORY')
        members = list(directory.iterdir())
        _need(len(members) <= MAX_EVENTS * 2, 'PRODUCER_RECEIPT_LIMIT')
        sizes = {}
        for path in members:
            _no_symlinks(path)
            _need(re.fullmatch(r'[0-9a-f]{64}\.json', path.name) is not None
                  or path.name.endswith('.pending'), 'UNKNOWN_CUSTODY_FILE', 'receipts/' + path.name)
            info = path.stat()
            _need(stat.S_ISREG(info.st_mode), 'PRODUCER_RECEIPT_NOT_FILE', 'receipts/' + path.name)
            sizes[path.name] = info.st_size
        return sizes

    def _read_receipt(self, origin):
        raw = read_producer_receipt(self.path / 'receipts' / (origin['receipt_sha256'] + '.json'))
        _verify_receipt_bytes(raw, origin)
        return _producer_receipt(raw)

    def _prepare_receipt(self, payload, raw):
        # Validate the original binding BEFORE creating an explicitly stronger
        # new origin. Never repair an invalid candidate or change old events.
        candidate = self._checked_candidate(payload['candidate'])
        receipt = _producer_receipt(raw)
        origin = candidate['origin']
        _need(hashlib.sha256(raw).hexdigest() == origin['receipt_sha256'],
              'PRODUCER_RECEIPT_HASH_MISMATCH', 'origin.receipt_sha256')
        if origin['schema'] == RECEIPT_ORIGIN_SCHEMA:
            _verify_receipt_bytes(raw, origin)
        else:
            candidate['origin'] = dict(origin, schema=RECEIPT_ORIGIN_SCHEMA, receipt_nbytes=len(raw))
            candidate = bind(candidate)
        return dict(payload, candidate=candidate), receipt

    def _load(self):
        """Verify/fold all committed events. No caches can outlive source identity."""
        _identity(self.identity)
        directory = self.path / 'events'
        _no_symlinks(directory)
        _need(directory.is_dir(), 'QUEUE_NOT_INITIALIZED')
        members = list(directory.iterdir())
        _need(len(members) <= MAX_EVENTS * 2, 'EVENT_LIMIT')
        paths = sorted(p for p in members if p.suffix == '.json')
        orphans = sorted(p.name for p in members if p.name.endswith('.pending'))
        _need(all(p.suffix == '.json' or p.name.endswith('.pending') for p in members),
              'UNKNOWN_CUSTODY_FILE')
        _need(paths and len(paths) <= MAX_EVENTS, 'EVENT_LIMIT')
        receipt_files = self._receipt_inventory()
        state, replies = None, {}
        for sequence, path in enumerate(paths):
            _need(path.name == f'{sequence:012d}.json', 'EVENT_SEQUENCE')
            _no_symlinks(path)
            event = read_json(path)
            _bound(event, EVENT_SCHEMA)
            _fields(event, ('schema', 'sequence', 'previous_event_sha256', 'request_id',
                            'command', 'payload', 'content_sha256'), 'event')
            _need(type(event['sequence']) is int and event['sequence'] == sequence, 'EVENT_SEQUENCE')
            _text(event['request_id'], 'request_id')
            if sequence == 0:
                _need(event['command'] == 'init' and event['request_id'] == '__init__'
                      and event['previous_event_sha256'] is None, 'INVALID_GENESIS')
                config = validate_config(event['payload'])
                _need(wire(config['identity']) == wire(self.identity), 'STALE_IDENTITY')
                state = _state(config)
                state['receipt_store_bytes'] = sum(receipt_files.values())
            else:
                _need(event['previous_event_sha256'] == state['head'], 'EVENT_CHAIN')
                _need(event['request_id'] not in state['requests'] and event['request_id'] != '__init__',
                      'DUPLICATE_REQUEST_ID')
                result = _apply(state, event['command'], event['payload'], sequence,
                                candidate_validator=self._checked_candidate, receipt_loader=self._read_receipt)
                state['requests'][event['request_id']] = wire([event['command'], event['payload']])
                replies[event['request_id']] = result
            state['sequence'], state['head'] = sequence, event['content_sha256']
            state['bytes'] += path.stat().st_size
            _need(sequence < state['config']['max_events'] and state['bytes'] <= state['config']['max_bytes'],
                  'EVENT_LIMIT')
        _need(state['bytes'] + state['receipt_store_bytes'] <= state['config']['max_bytes'], 'STORE_LIMIT')
        referenced = {key + '.json' for key in state['producer_receipts']}
        orphans.extend('receipts/' + name for name in sorted(set(receipt_files) - referenced))
        return state, replies, orphans

    def _transact(self, command, payload, request_id, *, producer_receipt=None):
        _text(request_id, 'request_id')
        _need(request_id != '__init__', 'RESERVED_REQUEST_ID')
        with self._lock():
            state, replies, orphans = self._load()
            receipt = None
            if command == 'enqueue':
                if producer_receipt is not None:
                    payload, receipt = self._prepare_receipt(payload, producer_receipt)
                else:
                    origin = payload['candidate'].get('origin', {}) if type(payload['candidate']) is dict else {}
                    _need(not (type(origin) is dict and origin.get('schema') == RECEIPT_ORIGIN_SCHEMA),
                          'PRODUCER_RECEIPT_REQUIRED')
            serialized = wire([command, payload])
            if request_id in state['requests']:
                _need(serialized == state['requests'][request_id], 'REQUEST_ID_CONFLICT')
                result = copy.deepcopy(replies[request_id])
                result.update(replayed_request=True, execute_permission=False,
                              head_sha256=state['head'], orphan_files=orphans)
                if result.get('candidate_id') in state['rows']:
                    result['current_candidate_status'] = state['rows'][result['candidate_id']]['status']
                return result
            sequence = state['sequence'] + 1
            _need(sequence < state['config']['max_events'], 'EVENT_LIMIT')
            result = _apply(state, command, payload, sequence,
                            candidate_validator=self._checked_candidate,
                            receipt_loader=(lambda origin: receipt) if receipt is not None else None)
            event = bind(dict(schema=EVENT_SCHEMA, sequence=sequence,
                              previous_event_sha256=state['head'], request_id=request_id,
                              command=command, payload=copy.deepcopy(payload)))
            raw = (wire(event) + '\n').encode()
            extra_receipt_bytes = 0
            if receipt is not None:
                origin = payload['candidate']['origin']
                target = self.path / 'receipts' / (origin['receipt_sha256'] + '.json')
                if not target.exists():
                    extra_receipt_bytes = len(producer_receipt)
            total_bytes = state['bytes'] + state['receipt_store_bytes'] + extra_receipt_bytes + len(raw)
            _need(total_bytes <= state['config']['max_bytes'], 'STORE_LIMIT')
            unfinished = sum(row['status'] in ('READY', 'LEASED', 'STARTED')
                             for row in state['rows'].values())
            # Every authorized or queued candidate must retain room for a final
            # COMPLETE/CENSORED custody event. Never return execute permission
            # from a journal that cannot durably record its terminal result.
            _need(sequence + 1 + unfinished <= state['config']['max_events'],
                  'CUSTODY_CAPACITY')
            _need(total_bytes + 8192 * unfinished <= state['config']['max_bytes'],
                  'CUSTODY_CAPACITY')
            if receipt is not None:
                directory = self.path / 'receipts'
                _no_symlinks(directory)
                directory.mkdir(exist_ok=True, mode=0o700)
                _sync_dir(self.path)
                # Receipt file and directory are durable BEFORE its event exists.
                _publish_receipt(directory, origin, producer_receipt)
                orphan_name = 'receipts/' + origin['receipt_sha256'] + '.json'
                orphans = [name for name in orphans if name != orphan_name]
            _publish(self.path / 'events', sequence, raw)  # permission is still LOCAL, not returned
            state['head'], state['sequence'] = event['content_sha256'], sequence
            if command == 'resume':
                result.update(_summary(state))
            result.update(replayed_request=False, head_sha256=state['head'], sequence=sequence,
                          orphan_files=orphans)
            return copy.deepcopy(result)

    def enqueue(self, candidate, *, request_id, now_ms, producer_receipt: bytes | None = None):
        """Append one origin; optionally retain the sealed producer's EXACT bytes.

        The compatible keyword verifies origin.receipt_sha256 against raw bytes
        and promotes only this new v1-origin event to RECEIPT_ORIGIN_SCHEMA. V2
        adds receipt_nbytes and always requires bytes, including request retries.
        No bytes on a v1 call remains HASH_ONLY_UNRETAINED, even when another
        origin retained that digest. Caller objects and historical events stay
        unchanged. Receipt objects/counts are private evaluator diagnostics.
        """
        return self._transact('enqueue', dict(now_ms=now_ms, candidate=copy.deepcopy(candidate)),
                              request_id, producer_receipt=producer_receipt)

    def get_producer_receipt(self, receipt_sha256: str) -> bytes:
        """Original private producer bytes, only for a committed retained origin."""
        _hash(receipt_sha256, 'receipt_sha256')
        with self._lock():
            state, _, _ = self._load()
            _need(receipt_sha256 in state['producer_receipts'], 'PRODUCER_RECEIPT_NOT_RETAINED')
            raw = read_producer_receipt(self.path / 'receipts' / (receipt_sha256 + '.json'))
            _verify_receipt_bytes(raw, dict(receipt_sha256=receipt_sha256,
                receipt_nbytes=state['producer_receipts'][receipt_sha256]['nbytes']))
            return raw

    def lease(self, worker, *, request_id, now_ms, ttl_ms):
        return self._transact('lease', dict(now_ms=now_ms, worker=worker, ttl_ms=ttl_ms), request_id)

    def charge(self, lease_id, worker, *, request_id, now_ms):
        return self._transact('charge', dict(now_ms=now_ms, worker=worker, lease_id=lease_id), request_id)

    def complete(self, lease_id, worker, result, *, request_id, now_ms):
        return self._transact('complete', dict(now_ms=now_ms, worker=worker, lease_id=lease_id,
                                             result=copy.deepcopy(result)), request_id)

    def censor(self, lease_id, worker, reason, *, request_id, now_ms):
        return self._transact('censor', dict(now_ms=now_ms, worker=worker, lease_id=lease_id, reason=reason), request_id)

    def resume(self, *, request_id, now_ms):
        return self._transact('resume', dict(now_ms=now_ms), request_id)

    def inspect(self, *, expected_head=None):
        with self._lock():
            state, _, orphans = self._load()
            if expected_head is not None:
                _hash(expected_head, 'expected_head')
                _need(expected_head == state['head'], 'HEAD_MISMATCH')
            return {**_summary(state), 'orphan_files': orphans}

    def get_candidate(self, candidate_id):
        """Private worker/evaluator payload. Use actor_inputs() for the model."""
        _hash(candidate_id, 'candidate_id')
        with self._lock():
            state, _, _ = self._load()
            _need(candidate_id in state['rows'], 'UNKNOWN_CANDIDATE')
            return copy.deepcopy(state['rows'][candidate_id]['candidate'])

    def lookup_exact(self, snapshot):
        actor = _snapshot_actor(snapshot, self.identity)
        fingerprint = _fingerprint(snapshot, actor, self.identity)
        with self._lock():
            state, _, _ = self._load()
            row = state['rows'].get(fingerprint)
            if row is None:
                return None
            candidate = row['candidate']
            _need(wire(_exact(snapshot, actor, self.identity)) == wire(
                _exact(candidate['snapshot'], candidate['actor_context'], candidate['identity'])), 'HASH_COLLISION')
            result = row['result']
            if row['status'] != 'COMPLETE' or result['status'] != 'SOLVED':
                return None
            return dict(match='EXACT', candidate_id=fingerprint, raw_actions=copy.deepcopy(result['raw_actions']),
                        receipt_sha256=result['receipt_sha256'], teacher_authority=False,
                        optimality_certified=False, execute_permission=False)

    def neighbors(self, snapshot, *, limit=8):
        """On-demand public-fact L1 ranking, NOT an embedding/feature store/cache proof."""
        _integer(limit, 'limit', 1, 32)
        actor = _snapshot_actor(snapshot, self.identity)
        facts = actor['observation']
        scales = {'cash': 1000., 'step': 24., 'quadrants': 1., 'hands_today': 1., 'occupied': 10.}
        with self._lock():
            state, _, _ = self._load()
            found = []
            for key, row in state['rows'].items():
                if row['status'] != 'COMPLETE' or row['result']['status'] != 'SOLVED':
                    continue
                other = row['candidate']['actor_context']['observation']
                distance = math.fsum(abs(facts[k] - other[k]) / scale for k, scale in scales.items())
                _need(math.isfinite(distance), 'NONFINITE_DISTANCE')
                found.append((distance, key, row['result']['raw_actions']))
            return [dict(match='APPROXIMATE_PROPOSAL_ONLY', candidate_id=key,
                         raw_actions=copy.deepcopy(actions), public_distance=distance,
                         requires_revalidation=True, teacher_authority=False, execute_permission=False)
                    for distance, key, actions in sorted(found)[:limit]]


def schema() -> dict:
    return dict(schema=SCHEMA, candidate_schema=CANDIDATE_SCHEMA, config_schema=CONFIG_SCHEMA,
        identity_schema=IDENTITY_SCHEMA, origin_schema=ORIGIN_SCHEMA, result_schema=RESULT_SCHEMA,
        receipt_origin_schema=RECEIPT_ORIGIN_SCHEMA,
        producer_receipt='OPTIONAL_EXACT_BYTES; CANONICAL_CONTENT_SHA256_OR_POOL_SHA256_SEAL',
        commands=['init', *PAYLOAD_FIELDS, 'inspect', 'lookup', 'neighbors', 'get'],
        reasons=list(REASONS), costs=list(COSTS), budget_semantics='PREPAID_UPPER_BOUNDS_NO_REFUNDS',
        clock='EXPLICIT_NONDECREASING_INTEGER_MILLISECONDS', actor='EXISTING_RECOVERY_ACTOR_ALLOWLIST',
        source_quotas='POSITIVE_INTEGER_WEIGHTED_CELLS_REASON_X_POPULATION',
        max_record_bytes=MAX_RECORD_BYTES, max_events=MAX_EVENTS, max_candidates=MAX_CANDIDATES,
        max_ttl_ms=MAX_TTL_MS, teacher_authority=False,
        reachability='SOURCE_RECEIPT_ATTESTATION_NOT_NATIVE_REPLAY',
        supported_boundary='NON_OPENING_DAWN_BEFORE_OBSERVE_AND_PUBLICATION')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('schema')
    init = sub.add_parser('init')
    init.add_argument('--queue', required=True); init.add_argument('--config', required=True)
    for name in (*PAYLOAD_FIELDS, 'inspect', 'lookup', 'neighbors', 'get'):
        p = sub.add_parser(name)
        p.add_argument('--queue', required=True); p.add_argument('--identity', required=True)
        if name in PAYLOAD_FIELDS:
            p.add_argument('--request-id', required=True)
            p.add_argument('--now-ms', type=int, required=True)
        if name == 'enqueue':
            p.add_argument('--candidate', required=True)
            p.add_argument('--producer-receipt', help='sealed producer JSON; retain exact original bytes')
        if name in ('lease', 'charge', 'complete', 'censor'):
            p.add_argument('--worker', required=True)
        if name == 'lease':
            p.add_argument('--ttl-ms', type=int, required=True)
        if name in ('charge', 'complete', 'censor'):
            p.add_argument('--lease-id', required=True)
        if name == 'complete':
            p.add_argument('--result', required=True)
        if name == 'censor':
            p.add_argument('--reason', required=True)
        if name in ('lookup', 'neighbors'):
            p.add_argument('--snapshot', required=True)
        if name == 'neighbors':
            p.add_argument('--limit', type=int, default=8)
        if name == 'get':
            p.add_argument('--candidate-id', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'schema':
            result = schema()
        elif args.command == 'init':
            queue = SearchQueue.create(args.queue, read_json(args.config))
            result = queue.inspect()
        else:
            queue = SearchQueue(args.queue, read_json(args.identity))
            common = dict(request_id=args.request_id, now_ms=args.now_ms) if args.command in PAYLOAD_FIELDS else {}
            if args.command == 'enqueue':
                result = queue.enqueue(read_json(args.candidate), **common, producer_receipt=(
                    read_producer_receipt(args.producer_receipt) if args.producer_receipt else None))
            elif args.command == 'lease':
                result = queue.lease(args.worker, ttl_ms=args.ttl_ms, **common)
            elif args.command == 'charge':
                result = queue.charge(args.lease_id, args.worker, **common)
            elif args.command == 'complete':
                result = queue.complete(args.lease_id, args.worker, read_json(args.result), **common)
            elif args.command == 'censor':
                result = queue.censor(args.lease_id, args.worker, args.reason, **common)
            elif args.command == 'resume':
                result = queue.resume(**common)
            elif args.command == 'inspect':
                result = queue.inspect()
            elif args.command == 'get':
                result = queue.get_candidate(args.candidate_id)
            elif args.command == 'lookup':
                result = queue.lookup_exact(read_json(args.snapshot))
            else:
                result = queue.neighbors(read_json(args.snapshot), limit=args.limit)
        print(wire(result))
        return 0
    except (QueueError, OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(wire(dict(status='REFUSED', reason_code=getattr(exc, 'code', type(exc).__name__),
                        execute_permission=False, teacher_authority=False)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
