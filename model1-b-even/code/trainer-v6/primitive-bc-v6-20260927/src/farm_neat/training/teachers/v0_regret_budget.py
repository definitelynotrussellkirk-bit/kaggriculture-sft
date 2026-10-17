"""V0 search-budget measurement and versioned adaptive discovery allocation.

The original fixed/random v1 API below remains unchanged. The separate
``register_mc_allocation`` / ``MonteCarloAllocator`` API allocates paired discovery
batches; ``compare_mc_allocations`` replays uniform/adaptive histories at equal
caps. It never executes games, changes prefix legality or admits actor labels.

``register_budget(spec)`` seals a PRE-OUTCOME fixed/random registration.
``summarize_budget(history, *, limits)`` folds an ordered, hash-linked event ledger.
``compare_budgets(control, candidate)`` RECOMPUTES two {limits, history} run bundles;
caller-supplied summary numerators cannot certify a matched-cost comparison.
``pilot_budget_table()`` emits unexecuted, equal-cap fixed/random recipes.
``schema()`` documents the searcher's JSON event wire contract.

Each history concerns ONE learner source/context/seat tuple and frozen direct draw.
Proposal payloads retain all attempts, including DUPLICATE/ILLEGAL/FAILED/CENSORED.
Evaluation payloads carry the existing regret-record seat_rows, not invented scalar
Q scores. Seats average inside worlds, then worlds inside preregistered independent
source families. All registered cells must finish before a batch has a WDL mean.
WDL uses the existing -1/0/+1 convention: paired gains range from -2 to +2, NOT
win-probability percentage points. A selection freezes the discovery winner before
confirmation, and an admission payload supplies the actual bundle and protocol to
the existing admit_pair gate. The gate is rerun, not an incoming admissible flag.
The output is measurement/provenance only and is never a replacement actor label.

Candidate identity binds exact Plan AND decoder continuation AND actor context
(which contains starting compiler memory); raw aliases are not separate candidates.
Edit depth is SEARCH LINEAGE depth, not native action count or the 16-edit V0 cap.
The canonical proposer/record validator still owns reachability and prefix legality.
Independent counts describe declared source families, not scientifically certified
independence. Source, replay and chronology attestations require external custody;
a hash cannot prove unobserved exposure or prevent an entirely rewritten history.

Costs are incremental charges, including failed attempts, cache lookup and replay.
Elapsed seconds are non-overlapping elapsed timeline increments attributed by the
collector, NOT summed overlapping worker runtimes. Native starts are physical branch
starts; replays cost starts without adding independent groups. No events are retried.
INJECTED_FIXTURE costs may be synthetic and are never native measurement evidence.
Malformed histories are REFUSED with forensic costs retained; do not train from them.

CLI, read-only inputs, exclusive output creation::

    python -m farm_neat.training.teachers.v0_regret_budget schema
    python -m farm_neat.training.teachers.v0_regret_budget pilot-table
    python -m farm_neat.training.teachers.v0_regret_budget summarize \
        --history HISTORY.json --limits LIMITS.json --output SUMMARY.json
    python -m farm_neat.training.teachers.v0_regret_budget compare \
        --control CONTROL_RUN.json --candidate CANDIDATE_RUN.json --output COMPARE.json

The adaptive CLI consumes a closed {registration, history} run bundle::

    python -m farm_neat.training.teachers.v0_regret_budget allocate --run RUN.json
    python -m farm_neat.training.teachers.v0_regret_budget compare-allocations \
        --control UNIFORM.json --candidate ADAPTIVE.json

The private hypothesis-level API composes the existing uniform MC allocator:
``register_belief_mc_from_record`` joins canonical frozen actor masses/geometry;
``BeliefMonteCarloAllocator`` schedules mass, uniform, uncertainty or decision-impact
batches; ``belief_mc_allocation_event`` revalidates raw MC receipts. Its terminal
WIN-PROBABILITY diagnostic is versioned separately from existing WDL admission.
``allocate-beliefs --run RUN.json`` replays one history; ``compare-beliefs --runs
FOUR_RUNS.json`` requires four completed, source-matched histories. No hypotheses
are silently renormalized when UNKNOWN cannot be simulated.

No games, neural inference, model files, optimizer steps or external scheduling are called.
At intake replace this module's planned entrypoint with the CLI; shared registries
are deliberately outside this reviewer's write ownership.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

LIMITS_SCHEMA = 'farm-v0-regret-budget-registration-v1'
EVENT_SCHEMA = 'farm-v0-regret-budget-event-v1'
SUMMARY_SCHEMA = 'farm-v0-regret-budget-summary-v1'
ROOT = Path(__file__).resolve().parents[4]
COSTS = ('native_starts', 'cpu_seconds', 'elapsed_seconds')
CAPS = ('candidate_attempts', 'edit_depth', 'independent_discovery_groups', *COSTS)
PHASES = ('discovery', 'confirmation', 'protected_final')
WORLD_FIELDS = ('source_group_id', 'source_family_id', 'world_sha256', 'snapshot_sha256', 'rng_sha256', 'seed')
ROW_FIELDS = ('source_group_id', 'seat', 'actor_context_sha256', 'learner_wdl', 'challenger_wdl',
              'complete', 'replay_verified', 'learner_attempt_id', 'challenger_attempt_id')
PATH_FIELDS = ('raw_actions_sha256', 'canonical_plan_sha256', 'continuation_signature_sha256')
LIMIT_FIELDS = ('schema', 'comparison_id', 'arm', 'evidence_kind', 'parent_checkpoint_sha256',
                'source_pins', 'source_group_id', 'actor_context_sha256s', 'learner_path_sha256s',
                'opponent_sha256', 'seats', 'worlds', 'caps', 'seed', 'cost_match_relative_tolerance',
                'frozen_before_outcomes', 'budget_basis', 'protected_final_exposed', 'retry_until_success')
EVENT_FIELDS = ('schema', 'event_id', 'sequence', 'previous_event_sha256', 'registration_sha256',
                'parent_checkpoint_sha256', 'source_pins_sha256', 'kind', 'candidate_id', 'costs', 'payload')
PAYLOAD_FIELDS = {
    'proposal': ('status', 'proposal_family', 'edit_depth', 'parent_candidate_id', 'paths',
                 'reason_codes', 'producer_receipt_sha256'),
    'evaluation': ('phase', 'mode', 'seat_rows', 'reuse_of', 'reuse_kind', 'diagnostic_score', 'producer_receipt_sha256'),
    'selection': ('discovery_event_id', 'stop_reason', 'confirmation_protocol_sha256', 'test_id'),
    'admission': ('pair', 'protocol'),
}
LIMITATIONS = [
    'MEASUREMENT_ONLY_NO_ADAPTIVE_ALLOCATION_OR_ACTOR_LABELS',
    'HASHES_DO_NOT_AUTHENTICATE_CHRONOLOGY_INDEPENDENCE_OR_UNDISCLOSED_EXPOSURE',
    'FIXTURE_COSTS_AND_OUTCOMES_ARE_NOT_NATIVE_POLICY_STRENGTH',
    'NO_COMPLETE_CASE_SELECTION_ON_CENSORED_WORLD_BATCHES',
]


class BudgetError(ValueError):
    """Stable refusal code and location; never silently repairs a record."""
    def __init__(self, code: str, at: str = '$'):
        self.code, self.at = code, at
        super().__init__(f'{code}: {at}')


def _need(ok, code, at='$'):
    if not ok:
        raise BudgetError(code, at)


def _fields(value, names, at):
    _need(type(value) is dict and set(value) == set(names), 'FIELDS_MISMATCH', at)


def _number(value, at, low=0, high=1e15, integer=False):
    _need(type(value) is int if integer else type(value) in (int, float), 'INVALID_NUMBER', at)
    try:
        good = math.isfinite(value) and low <= value <= high
    except OverflowError:
        good = False
    _need(good, 'INVALID_NUMBER', at)
    return value


def _text(value, at):
    _need(type(value) is str and 0 < len(value.strip()) <= 1024, 'INVALID_STRING', at)


def _hash(value, at):
    _need(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'INVALID_HASH', at)


def _safe(value, depth=0):
    _need(depth <= 64, 'JSON_LIMIT')
    if value is None or type(value) in (str, bool):
        return
    if type(value) in (int, float):
        # Do not cap finite producer numerics (e.g. counters) before their validator.
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        _need(finite, 'NONFINITE_JSON')
    elif type(value) in (list, dict):
        if type(value) is dict:
            _need(all(type(k) is str for k in value), 'UNSAFE_JSON')
        for child in (value.values() if type(value) is dict else value):
            _safe(child, depth + 1)
    else:
        raise BudgetError('UNSAFE_JSON')


def _wire(value):
    _safe(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(_wire(value).encode()).hexdigest()


def bind(value: dict) -> dict:
    """Detached canonical content binding. A digest is identity, not authentication."""
    _need(type(value) is dict, 'INVALID_OBJECT')
    out = copy.deepcopy({k: v for k, v in value.items() if k != 'content_sha256'})
    out['content_sha256'] = digest(out)
    return out


def _bound(value, code):
    _need(type(value) is dict and value.get('content_sha256') == bind(value)['content_sha256'], code)


def source_pins() -> dict:
    """Reuse the records' semantic source closure, plus this component's inputs."""
    from farm_neat.training import v0_regret_records as R
    paths = ('src/farm_neat/training/teachers/v0_regret_budget.py',
             'src/farm_neat/training/teachers/v0_regret_proposals.py',
             'src/farm_neat/evaluation/v0_regret_pairs.py',
             'src/farm_neat/training/v0_regret_admission.py',
             'src/farm_neat/evaluation/v0_regret_evaluate.py')
    return {**R.source_pins(), **{p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}}


def path_binding(proposal: dict) -> dict:
    """Adapter for actual proposer/record proposals; does not certify reachability."""
    return dict(raw_actions_sha256=digest(proposal['raw_actions']),
                canonical_plan_sha256=digest(proposal['canonical_plan']),
                continuation_signature_sha256=digest(proposal['continuation_signature']))


def candidate_key(paths: dict, actor_context_sha256s: dict) -> str:
    _need(type(paths) is dict and set(paths) == set(actor_context_sha256s), 'SEAT_PATH_MISMATCH')
    for seat, path in paths.items():
        _fields(path, PATH_FIELDS, 'paths.' + seat)
        for field in PATH_FIELDS:
            _hash(path[field], 'paths.' + field)
    semantic = {s: {k: p[k] for k in PATH_FIELDS if k != 'raw_actions_sha256'} for s, p in paths.items()}
    return digest(dict(actor_context_sha256s=actor_context_sha256s, successors=semantic))


def _registration(limits):
    _fields(limits, (*LIMIT_FIELDS, 'content_sha256'), 'limits')
    _bound(limits, 'REGISTRATION_HASH_MISMATCH')
    _need(limits['schema'] == LIMITS_SCHEMA, 'REGISTRATION_SCHEMA')
    _need(limits['arm'] in ('fixed', 'random'), 'ADAPTIVE_ALLOCATION_FORBIDDEN')
    _need(limits['evidence_kind'] in ('NATIVE_GAME', 'INJECTED_FIXTURE'), 'EVIDENCE_KIND')
    for f in ('comparison_id', 'source_group_id'):
        _text(limits[f], f)
    for f in ('parent_checkpoint_sha256', 'opponent_sha256'):
        _hash(limits[f], f)
    _need(limits['source_pins'] == source_pins(), 'SOURCE_MISMATCH')
    _need(limits['protected_final_exposed'] is False, 'PROTECTED_FINAL_EXPOSURE')
    _need(limits['frozen_before_outcomes'] is True, 'POST_OUTCOME_BUDGET')
    _need(limits['budget_basis'] == 'PREDECLARED_FIXED', 'PROTECTED_BUDGET_SELECTION')
    _need(limits['retry_until_success'] is False, 'RETRY_UNTIL_SUCCESS_FORBIDDEN')
    _number(limits['seed'], 'seed', integer=True)
    _number(limits['cost_match_relative_tolerance'], 'cost_match_relative_tolerance', high=.25)
    seats = limits['seats']
    _need(type(seats) is list and all(type(s) is int for s in seats) and seats in ([0], [1], [0, 1]), 'SEATS')
    for f in ('actor_context_sha256s', 'learner_path_sha256s'):
        _fields(limits[f], [str(s) for s in seats], f)
        for value in limits[f].values():
            _hash(value, f)
    caps = limits['caps']
    _fields(caps, CAPS, 'caps')
    for k in CAPS:
        _number(caps[k], 'caps.' + k, integer=k not in ('cpu_seconds', 'elapsed_seconds'))
    _need(caps['candidate_attempts'] > 0 and caps['elapsed_seconds'] > 0, 'EMPTY_BUDGET')
    _fields(limits['worlds'], PHASES, 'worlds')
    exposure, maps = {}, {}
    for phase in PHASES:
        worlds = limits['worlds'][phase]
        _need(type(worlds) is list and (bool(worlds) or phase == 'protected_final'), 'EMPTY_WORLD_SET', phase)
        local, phase_map = set(), {}
        for w in worlds:
            _fields(w, WORLD_FIELDS, 'world')
            for k in ('source_group_id', 'source_family_id'):
                _text(w[k], k)
            for k in ('world_sha256', 'snapshot_sha256', 'rng_sha256'):
                _hash(w[k], k)
            _number(w['seed'], 'world.seed', integer=True)
            for key in ('source_group_id', 'source_family_id', 'world_sha256', 'seed'):
                token = (key, w[key])
                _need(token not in exposure or exposure[token] == phase, 'SPLIT_OVERLAP', key)
                exposure[token] = phase
                if key != 'source_family_id':
                    _need(token not in local, 'DUPLICATE_WORLD', key)
                    local.add(token)
            phase_map[w['source_group_id']] = w
        maps[phase] = phase_map
    families = {w['source_family_id'] for w in maps['discovery'].values()}
    _need(len(families) <= caps['independent_discovery_groups'], 'WORLD_CAP_BELOW_REGISTRATION')
    return maps


def register_budget(spec: dict) -> dict:
    """Freeze caps/sources before outcomes; no quotas are inferred from data."""
    limits = bind(spec)
    _registration(limits)
    return limits


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _row_signature(rows, *, attempts=True):
    omitted = {'replay_verified'} | (set() if attempts else {'learner_attempt_id', 'challenger_attempt_id'})
    return sorted([{k: v for k, v in row.items() if k not in omitted} for row in rows],
                  key=lambda r: (r['source_group_id'], r['seat']))


def _batch(payload, candidate, limits, worlds, evaluations, attempts):
    """Validate before mutating the fold, returning explicit fresh/reused evidence."""
    phase = payload['phase']
    _need(phase not in ('final', 'protected_final', 'monitor'), 'PROTECTED_FINAL_EXPOSURE')
    _need(phase in ('discovery', 'confirmation'), 'UNKNOWN_PHASE')
    mode = payload['mode']
    _need(mode in ('EXACT_REALIZATION', 'CONFIRMED_EXPECTATION', 'PREDICTED_ONLY', 'CASH_ONLY'), 'EVIDENCE_MODE')
    _hash(payload['producer_receipt_sha256'], 'producer_receipt_sha256')
    rows = payload['seat_rows']
    _need(type(rows) is list, 'INVALID_ROWS')
    if mode in ('PREDICTED_ONLY', 'CASH_ONLY'):
        _need(not rows and phase == 'discovery' and payload['reuse_of'] is None and payload['reuse_kind'] is None,
              'NON_WDL_EVIDENCE_HAS_OUTCOMES')
        _number(payload['diagnostic_score'], 'diagnostic_score', low=-1e15)
        return dict(phase=phase, candidate_id=candidate, gain=None, rows=[], fresh_attempts={},
                    groups=set(), families=set(), reuse_kind=None, diagnostic=True)
    _need(payload['diagnostic_score'] is None, 'MIXED_WDL_PREDICTOR')
    _need((phase == 'discovery' and mode == 'EXACT_REALIZATION') or
          (phase == 'confirmation' and mode == 'CONFIRMED_EXPECTATION'), 'EVIDENCE_MODE')
    cells, fresh, groups, families = {}, {}, set(), set()
    for row in rows:
        _fields(row, ROW_FIELDS, 'seat_row')
        group, seat = row['source_group_id'], row['seat']
        _text(group, 'source_group_id')
        _need(group not in worlds['protected_final'], 'PROTECTED_FINAL_EXPOSURE')
        _need(group in worlds[phase], 'UNREGISTERED_WORLD')
        _need(type(seat) is int and seat in limits['seats'], 'UNREGISTERED_SEAT')
        _need(row['actor_context_sha256'] == limits['actor_context_sha256s'][str(seat)], 'ACTOR_CONTEXT_MISMATCH')
        _need(type(row['complete']) is bool and type(row['replay_verified']) is bool, 'INVALID_BOOLEAN')
        _need(not row['replay_verified'] or row['complete'], 'CENSORED_REPLAY')
        _need((group, seat) not in cells, 'DUPLICATE_OUTCOME')
        cells[group, seat] = row
        for arm in ('learner', 'challenger'):
            wdl = row[arm + '_wdl']
            if row['complete']:
                _need(type(wdl) is int and wdl in (-1, 0, 1), 'NON_WDL_OUTCOME')
            else:
                _need(wdl is None, 'CENSORED_OUTCOME')
            identity = row[arm + '_attempt_id']
            _text(identity, 'attempt_id')
            sig = (arm, group, seat, row['actor_context_sha256'], wdl, row['complete'],
                   candidate if arm == 'challenger' else None)
            _need(identity not in fresh, 'ATTEMPT_ID_REUSE')
            _need(identity not in attempts or attempts[identity] == sig, 'ATTEMPT_ID_CONFLICT')
            fresh[identity] = sig
        groups.add(group)
        families.add(worlds[phase][group]['source_family_id'])
    reuse = payload['reuse_kind']
    if reuse is not None:
        _need(reuse in ('cache', 'replay') and payload['reuse_of'] in evaluations, 'UNKNOWN_REUSE')
        old = evaluations[payload['reuse_of']]
        _need(old['candidate_id'] == candidate and old['phase'] == phase and not old['diagnostic'], 'REUSE_JOIN_MISMATCH')
        _need(_row_signature(rows, attempts=reuse == 'cache') ==
              _row_signature(old['rows'], attempts=reuse == 'cache'), 'REUSE_OUTCOME_MISMATCH')
        if reuse == 'cache':
            _need(set(fresh) <= set(attempts), 'CACHE_NEW_EXECUTION')
        else:
            _need(not set(fresh) & set(attempts), 'REPLAY_NOT_FRESH')
        gain = old['gain']
    else:
        _need(payload['reuse_of'] is None, 'UNKNOWN_REUSE')
        expected = {(g, s) for g in worlds[phase] for s in limits['seats']}
        complete = set(cells) == expected and all(r['complete'] for r in rows)
        clustered = {}
        if complete:
            for group, world in worlds[phase].items():
                value = math.fsum(cells[group, s]['challenger_wdl'] - cells[group, s]['learner_wdl']
                                  for s in limits['seats']) / len(limits['seats'])
                clustered.setdefault(world['source_family_id'], []).append(value)
        gain = math.fsum(math.fsum(v) / len(v) for v in clustered.values()) / len(clustered) if clustered else None
    return dict(phase=phase, candidate_id=candidate, gain=gain, rows=copy.deepcopy(rows),
                fresh_attempts={k: v for k, v in fresh.items() if k not in attempts},
                groups=groups, families=families, reuse_kind=reuse, diagnostic=False)


def _admission(payload, selection, candidate, limits, worlds, batches):
    """Reuse the existing statistical and native-prefix validator, not incoming flags."""
    from farm_neat.training import v0_regret_admission as A
    pair, protocol = payload['pair'], payload['protocol']
    _need(digest(protocol) == selection['confirmation_protocol_sha256'], 'ADMISSION_PROTOCOL_MISMATCH')
    receipt = A.admit_pair(pair, protocol=protocol)
    _need(receipt['status'] != 'INVALID', 'ADMISSION_INVALID:' + ','.join(receipt['reason_codes']))
    _need(receipt['test_id'] == selection['test_id'], 'ADMISSION_TEST_MISMATCH')
    test = next(t for t in protocol['tests'] if t['test_id'] == selection['test_id'])
    _need(test['seats'] == limits['seats'], 'ADMISSION_SEAT_MISMATCH')
    for phase in ('discovery', 'confirmation'):
        expected = {g: {k: v for k, v in w.items() if k != 'source_family_id'} for g, w in worlds[phase].items()}
        _need({w['source_group_id']: w for w in test[phase + '_world_groups']} == expected, 'ADMISSION_WORLD_MISMATCH')
        _need(all(test['source_families'][g] == w['source_family_id'] for g, w in worlds[phase].items()),
              'ADMISSION_FAMILY_MISMATCH')
        retained = [{k: v for k, v in r.items() if k not in ('phase', 'pair_id')}
                    for r in receipt['retained_outcomes'] if r['phase'] == phase]
        _need(_row_signature(retained) == _row_signature(batches[phase]['rows']), 'ADMISSION_OUTCOME_MISMATCH')
    records = pair['records'] if pair.get('schema') == A.BUNDLE_SCHEMA else [pair]
    for record in records:
        identity = record['identity']; seat = str(record['actor_context']['observation']['player'])
        _need(identity['parent_checkpoint_sha256'] == limits['parent_checkpoint_sha256'] and
              identity['source_group_id'] == limits['source_group_id'] and
              identity['actor_context_sha256'] == limits['actor_context_sha256s'][seat] and
              record['learner']['path_sha256'] == limits['learner_path_sha256s'][seat] and
              record['evaluation']['opponent_hash'] == limits['opponent_sha256'], 'ADMISSION_SOURCE_MISMATCH')
        _need(path_binding(record['challenger']) == candidate['paths'][seat], 'ADMISSION_PATH_MISMATCH')
    gain = receipt['mean_wdl_delta']
    complete_gain = batches['confirmation']['gain']
    if complete_gain is None:
        # The gate retains a descriptive complete-case mean for censored bundles.
        # Keep that receipt, but never promote it to a complete-batch frontier gain.
        _need(receipt['status'] != 'ADMITTED', 'CENSORED_ADMISSION')
    else:
        _need(gain is not None and math.isclose(gain, complete_gain, abs_tol=1e-12), 'ADMISSION_GAIN_MISMATCH')
    return receipt


def summarize_budget(history: list[dict], *, limits: dict) -> dict:
    """Pure sequential fold; refuses bad joins but retains every parseable cost."""
    totals = dict(events=0, candidate_attempts=0, unique_candidates=0, max_edit_depth=0,
                  evaluation_batches=0, diagnostic_batches=0, cache_hits=0, replay_batches=0,
                  independent_discovery_groups=0, independent_confirmation_groups=0,
                  discovery_worlds=0, confirmation_worlds=0, native_starts=0, cpu_seconds=0., elapsed_seconds=0.)
    out = dict(schema=SUMMARY_SCHEMA, status='VALID', limits=copy.deepcopy(limits), history_sha256=None,
               totals=totals, frontier=[], best_so_far_discovery_gain=0., best_candidate_id=None,
               discovery_selected_gain=None, confirmed_gain=None, admitted_corrections=0,
               admissions=[], refusals=[], strata={'source': {}, 'depth': {}, 'refusal': {}},
               adaptive_allocation_enabled=False, limitations=list(LIMITATIONS))
    def refuse(code, index=None, at='$'):
        out['refusals'].append(dict(code=code, event_index=index, at=at))
        tally(code)
    def tally(code):
        strata = out['strata']['refusal']
        strata[code] = strata.get(code, 0) + 1
    def stratum(family, depth, metric, value=1, *, maximum=False):
        for dimension, key in (('source', family), ('depth', str(depth))):
            group = out['strata'][dimension].setdefault(key, dict(
                native_starts=0, cpu_seconds=0., elapsed_seconds=0., best_so_far_discovery_gain=0.))
            group[metric] = max(group.get(metric, 0), value) if maximum else group.get(metric, 0) + value

    def charge_strata(family, depth, cost):
        for metric in COSTS:
            stratum(family, depth, metric, cost[metric])
    valid_registration = True
    try:
        worlds = _registration(limits)
    except (BudgetError, ValueError, TypeError, KeyError) as exc:
        refuse(getattr(exc, 'code', 'INVALID_REGISTRATION'), at=getattr(exc, 'at', 'limits'))
        valid_registration = False
    if type(history) is not list:
        refuse('HISTORY_NOT_LIST')
        history = []
    try:
        out['history_sha256'] = digest(history)
    except (BudgetError, ValueError, TypeError) as exc:
        refuse(getattr(exc, 'code', 'UNSAFE_HISTORY'))
    candidates, evaluations, phase_batches, attempts, ids = {}, {}, {}, {}, set()
    groups = {'discovery': set(), 'confirmation': set()}
    families = {'discovery': set(), 'confirmation': set()}
    selection = None
    previous = limits.get('content_sha256') if type(limits) is dict else None
    admission_cache, admitted_ids = {}, set()
    for index, event in enumerate(history):
        totals['events'] += 1
        kind = event.get('kind') if type(event) is dict else None
        if kind == 'proposal':
            totals['candidate_attempts'] += 1
        cost_ok = True
        try:
            cost = event['costs']
            _fields(cost, COSTS, 'costs')
            for key in COSTS:
                try:
                    _number(cost[key], 'costs.' + key, integer=key == 'native_starts')
                    totals[key] += cost[key]
                except BudgetError as exc:
                    # One malformed field must not erase other physical costs.
                    refuse(exc.code, index, exc.at)
                    cost_ok = False
        except (BudgetError, KeyError, TypeError) as exc:
            refuse(getattr(exc, 'code', 'MISSING_COST'), index, 'costs')
            cost_ok = False
        try:
            _need(valid_registration and cost_ok, 'UNUSABLE_EVENT')
            _fields(event, (*EVENT_FIELDS, 'content_sha256'), 'event')
            _need(event['registration_sha256'] == limits['content_sha256'], 'BUDGET_REGISTRATION_MISMATCH')
            _bound(event, 'EVENT_HASH_MISMATCH')
            _need(event['schema'] == EVENT_SCHEMA, 'EVENT_SCHEMA')
            _need(type(event['sequence']) is int and event['sequence'] == index, 'EVENT_ORDER')
            _need(event['previous_event_sha256'] == previous, 'HISTORY_CHAIN_MISMATCH')
            _text(event['event_id'], 'event_id')
            _need(event['event_id'] not in ids, 'DUPLICATE_EVENT_ID')
            ids.add(event['event_id'])
            _need(event['parent_checkpoint_sha256'] == limits['parent_checkpoint_sha256'], 'PARENT_MISMATCH')
            _need(event['source_pins_sha256'] == digest(limits['source_pins']), 'SOURCE_MISMATCH')
            _need(kind in PAYLOAD_FIELDS, 'EVENT_KIND')
            payload, key = event['payload'], event['candidate_id']
            _fields(payload, PAYLOAD_FIELDS[kind], 'payload')
            if kind == 'proposal':
                family, depth = payload['proposal_family'], payload['edit_depth']
                _text(family, 'proposal_family'); _number(depth, 'edit_depth', integer=True)
                stratum(family, depth, 'candidate_attempts')
                charge_strata(family, depth, cost)
                totals['max_edit_depth'] = max(totals['max_edit_depth'], depth)
                _hash(payload['producer_receipt_sha256'], 'producer_receipt_sha256')
                _need(type(payload['reason_codes']) is list and all(type(s) is str and s for s in payload['reason_codes']), 'REASON_CODES')
                status = payload['status']
                _need(status in ('RETAINED', 'DUPLICATE', 'ILLEGAL', 'FAILED', 'CENSORED', 'REJECTED'), 'PROPOSAL_STATUS')
                _need(selection is None, 'PROPOSAL_AFTER_SELECTION')
                parent = payload['parent_candidate_id']
                _need((parent is None and depth == 0) or
                      (parent in candidates and depth == candidates[parent]['edit_depth'] + 1), 'LINEAGE_DEPTH_MISMATCH')
                if status in ('RETAINED', 'DUPLICATE'):
                    _need(key == candidate_key(payload['paths'], limits['actor_context_sha256s']), 'CANDIDATE_ID_MISMATCH')
                    if status == 'DUPLICATE':
                        _need(key in candidates, 'UNKNOWN_DUPLICATE')
                    else:
                        _need(key not in candidates, 'UNDECLARED_DUPLICATE')
                        candidates[key] = copy.deepcopy(payload)
                        totals['unique_candidates'] += 1
                for reason in payload['reason_codes']:
                    tally(reason)
                if status != 'RETAINED' and not payload['reason_codes']:
                    tally(status)
            else:
                _need(type(key) is str and key in candidates, 'UNKNOWN_CANDIDATE')
                candidate = candidates[key]
                family, depth = candidate['proposal_family'], candidate['edit_depth']
                charge_strata(family, depth, cost)
                if kind == 'evaluation':
                    batch = _batch(payload, key, limits, worlds, evaluations, attempts)
                    phase, reuse = batch['phase'], batch['reuse_kind']
                    if phase == 'confirmation':
                        _need(selection is not None and selection['candidate_id'] == key, 'CONFIRMATION_BEFORE_SELECTION')
                    elif selection is not None and reuse is None:
                        raise BudgetError('DISCOVERY_AFTER_SELECTION')
                    if not batch['diagnostic'] and reuse is None:
                        _need((key, phase) not in phase_batches, 'FIXED_BATCH_RETRY_FORBIDDEN')
                    if reuse == 'cache':
                        _need(cost['native_starts'] == 0, 'CACHE_NATIVE_STARTS')
                    elif limits['evidence_kind'] == 'NATIVE_GAME':
                        _need(cost['native_starts'] >= len(batch['fresh_attempts']), 'UNDERREPORTED_NATIVE_STARTS')
                    attempts.update(batch['fresh_attempts'])
                    evaluations[event['event_id']] = batch
                    if reuse:
                        totals['cache_hits' if reuse == 'cache' else 'replay_batches'] += 1
                    elif batch['diagnostic']:
                        totals['diagnostic_batches'] += 1
                        tally('NON_WDL_DIAGNOSTIC')
                    else:
                        phase_batches[key, phase] = batch
                        totals['evaluation_batches'] += 1
                        stratum(family, depth, phase + '_batches')
                        groups[phase].update(batch['groups']); families[phase].update(batch['families'])
                        totals[phase + '_worlds'] = len(groups[phase])
                        totals['independent_' + phase + '_groups'] = len(families[phase])
                        gain = batch['gain']
                        if gain is not None:
                            stratum(family, depth, phase + '_gain_sum', gain)
                            if phase == 'discovery':
                                stratum(family, depth, 'best_so_far_discovery_gain', gain, maximum=True)
                        if gain is None:
                            tally('CENSORED_OR_MISSING_OUTCOMES')
                        elif phase == 'discovery' and gain > out['best_so_far_discovery_gain']:
                            out.update(best_so_far_discovery_gain=gain, best_candidate_id=key)
                        elif phase == 'discovery' and gain <= 0:
                            tally('TIE' if gain == 0 else 'LEARNER_BETTER')
                        if phase == 'confirmation':
                            out['confirmed_gain'] = gain
                elif kind == 'selection':
                    _need(selection is None, 'SELECTION_ALREADY_FROZEN')
                    stops = {'CANDIDATE_CAP': totals['candidate_attempts'] >= limits['caps']['candidate_attempts'],
                             'NATIVE_START_CAP': totals['native_starts'] >= limits['caps']['native_starts'] > 0,
                             'TIME_CAP': totals['elapsed_seconds'] >= limits['caps']['elapsed_seconds']}
                    _need(payload['stop_reason'] in stops and stops[payload['stop_reason']], 'EARLY_OUTCOME_DEPENDENT_STOP')
                    if payload['stop_reason'] == 'CANDIDATE_CAP':
                        _need(all((c, 'discovery') in phase_batches for c in candidates),
                              'UNACCOUNTED_DISCOVERY_CANDIDATES')
                    _need(payload['discovery_event_id'] in evaluations, 'UNSEEN_DISCOVERY')
                    ev = evaluations[payload['discovery_event_id']]
                    _need(ev['candidate_id'] == key and ev['phase'] == 'discovery' and not ev['diagnostic'] and
                          ev['gain'] is not None and ev['gain'] > 0 and key == out['best_candidate_id'], 'NOT_DISCOVERY_WINNER')
                    _hash(payload['confirmation_protocol_sha256'], 'confirmation_protocol_sha256')
                    _text(payload['test_id'], 'test_id')
                    selection = dict(payload, candidate_id=key)
                    out['discovery_selected_gain'] = ev['gain']
                elif kind == 'admission':
                    _need(selection is not None and selection['candidate_id'] == key and
                          (key, 'confirmation') in phase_batches, 'ADMISSION_BEFORE_CONFIRMATION')
                    token = digest(payload)
                    if token not in admission_cache:
                        receipt = _admission(payload, selection, candidate, limits, worlds,
                            {p: phase_batches[key, p] for p in ('discovery', 'confirmation')})
                        admission_cache[token] = receipt
                    receipt = admission_cache[token]
                    ident = (receipt['protocol_sha256'], receipt['test_id'])
                    duplicate = any(a['protocol_sha256'] == ident[0] and a['test_id'] == ident[1] for a in out['admissions'])
                    out['admissions'].append({k: receipt[k] for k in ('status', 'test_id', 'protocol_sha256',
                        'content_sha256', 'mean_wdl_delta', 'lower_bound', 'teacher_weight', 'reason_codes')})
                    out['admissions'][-1]['duplicate_read'] = duplicate
                    if receipt['status'] == 'ADMITTED' and ident not in admitted_ids:
                        admitted_ids.add(ident)
                        out['admitted_corrections'] += 1
                        stratum(family, depth, 'admitted_corrections')
                    for reason in receipt['reason_codes']:
                        tally(reason)
        except (BudgetError, ValueError, KeyError, TypeError, StopIteration, OverflowError) as exc:
            refuse(getattr(exc, 'code', 'MALFORMED_EVENT'), index, getattr(exc, 'at', 'event'))
        previous = event.get('content_sha256') if type(event) is dict else None
        if valid_registration:
            for metric, cap in limits['caps'].items():
                actual = totals['max_edit_depth' if metric == 'edit_depth' else metric]
                if actual > cap:
                    refuse('CAP_EXCEEDED', index, metric)
        out['frontier'].append(dict(event_index=index, **totals,
            **{k: out[k] for k in ('best_so_far_discovery_gain', 'best_candidate_id', 'discovery_selected_gain',
                                   'confirmed_gain', 'admitted_corrections')}))
    out['rates'] = dict(admitted_corrections_per_candidate=_ratio(out['admitted_corrections'], totals['candidate_attempts']),
                        admitted_corrections_per_native_start=_ratio(out['admitted_corrections'], totals['native_starts']),
                        admitted_corrections_per_elapsed_second=_ratio(out['admitted_corrections'], totals['elapsed_seconds']),
                        native_starts_per_admitted_correction=_ratio(totals['native_starts'], out['admitted_corrections']))
    out['status'] = 'REFUSED' if out['refusals'] else 'VALID' if history else 'NO_DATA'
    # Invalid/nonfinite inputs are not echoed into a purported finite JSON receipt.
    try:
        return bind(out)
    except (BudgetError, ValueError, TypeError):
        out['limits'] = None
        return bind(out)


def compare_budgets(control: dict, candidate: dict) -> dict:
    """Descriptive matched-cost contrast. Refuses summary-only or edited run data."""
    out = dict(schema='farm-v0-regret-budget-comparison-v1', status='REFUSED', refusals=[], deltas={},
               decision='DESCRIPTIVE_ONLY_NO_ADAPTIVE_ALLOCATION')
    def refuse(code, at='$'):
        out['refusals'].append(dict(code=code, at=at))
    summaries = []
    for name, run in (('control', control), ('candidate', candidate)):
        try:
            _fields(run, ('limits', 'history'), name)
            summary = summarize_budget(run['history'], limits=run['limits'])
            summaries.append(summary)
            out[name + '_summary_sha256'] = summary['content_sha256']
            if summary['status'] == 'REFUSED':
                refuse('INVALID_' + name.upper() + '_RUN')
                out['refusals'].extend(dict(r, run=name) for r in summary['refusals'])
        except (BudgetError, ValueError, TypeError) as exc:
            refuse(getattr(exc, 'code', 'INVALID_RUN'), name)
    if len(summaries) != 2 or out['refusals']:
        return bind(out)
    a, b = (s['limits'] for s in summaries)
    joins = {'parent_checkpoint_sha256': 'PARENT_MISMATCH', 'source_pins': 'SOURCE_MISMATCH',
             'source_group_id': 'SOURCE_GROUP_MISMATCH', 'worlds': 'SOURCE_GROUP_MISMATCH',
             'actor_context_sha256s': 'ACTOR_CONTEXT_MISMATCH', 'learner_path_sha256s': 'LEARNER_DRAW_MISMATCH',
             'opponent_sha256': 'OPPONENT_MISMATCH', 'seats': 'SEAT_MISMATCH', 'caps': 'CAP_MISMATCH',
             'comparison_id': 'COMPARISON_MISMATCH', 'evidence_kind': 'EVIDENCE_KIND_MISMATCH',
             'cost_match_relative_tolerance': 'COST_MATCH_RULE_MISMATCH'}
    for field, code in joins.items():
        if a[field] != b[field]:
            refuse(code, field)
    # Seed/arm may differ prospectively; all exposure and matching rules may not.
    if not out['refusals']:
        for field in ('candidate_attempts', *COSTS):
            left, right = (s['totals'][field] for s in summaries)
            tolerance = 0 if field in ('native_starts', 'candidate_attempts') else a['cost_match_relative_tolerance']
            if abs(left - right) > tolerance * max(abs(left), abs(right)):
                refuse('ACTUAL_COST_MISMATCH', field)
    if not out['refusals']:
        out['status'] = 'NO_DATA' if any(s['status'] == 'NO_DATA' for s in summaries) else 'MATCHED'
        for metric in ('best_so_far_discovery_gain', 'discovery_selected_gain', 'confirmed_gain', 'admitted_corrections'):
            left, right = (s[metric] for s in summaries)
            out['deltas'][metric] = right - left if left is not None and right is not None else None
        out['matched_caps'] = copy.deepcopy(a['caps'])
        out['evidence_kind'] = a['evidence_kind']
    return bind(out)


def pilot_budget_table() -> dict:
    """Zero-data recipes anchored to the existing 2-draw, 4/16-world pilot template.

    Challenger attempts are seat-tuples per draw, not retries until N legal winners.
    The learner is a frozen baseline execution, not a new challenger attempt. The
    aggregate cap spans two separate learner-source histories. A collector must
    bind those histories/world lists; this table is NOT an executable registration.
    """
    design_path = ROOT / 'project/configs/v0-regret-teacher.json'
    pilot = json.loads(design_path.read_text())['pilot_recipe']
    draws, seats = pilot['learner_draws'], len(pilot['seats'])
    discovery, confirm = pilot['discovery_world_groups_per_pair'], pilot['confirmation_world_groups_per_pair']
    rows = []
    for n in (1, 2, 4, 8):
        for depth in (1, 2, 4):
            base = draws * seats * (discovery * (1 + n) + confirm * 2)
            caps = dict(candidate_attempts=draws * n, edit_depth=depth, independent_discovery_groups=draws * discovery,
                        native_starts=2 * base, cpu_seconds=1800., elapsed_seconds=1800.)
            recipes = {arm: dict(arm=arm, caps=copy.deepcopy(caps), seed=17,
                         retry_until_success=False, stop='FIRST_PREREGISTERED_CAP_NO_REPLACEMENT_ATTEMPTS',
                         selection='MAX_COMPLETE_DISCOVERY_WDL_STABLE_FIRST_POSITIVE',
                         outcome_use='NONE_WHEN_SETTING_CAPS') for arm in ('fixed', 'random')}
            rows.append(dict(challengers_per_draw=n, edit_depth=depth, learner_draws=draws,
                             discovery_groups_per_draw=discovery, confirmation_groups_per_draw=confirm,
                             base_native_start_cap=base, replay_native_start_cap=base,
                             total_native_start_cap=2 * base, measured_candidate_attempts=0,
                             measured_correction_yield=None, measured_gain=None, **recipes))
    return bind(dict(schema='farm-v0-regret-pilot-budget-table-v1', status='NOT_RUN',
        design_sha256=hashlib.sha256(design_path.read_bytes()).hexdigest(), rows=rows,
        adaptive_allocation_enabled=False, selected_recipe=None,
        scope='MAINTAINER_REGISTRATION_TEMPLATE; 1800_SECONDS_IS_A_CAP_NOT_A_RUNTIME_ESTIMATE',
        required_bindings=['parent_checkpoint', 'source_pins', 'learner_source/context/draw', 'opponent',
                           'world/source-family partitions', 'pre-outcome registration custody',
                           'cost attribution', 'independent confirmation protocol'], limitations=list(LIMITATIONS)))


def schema() -> dict:
    """Machine-readable closed field contract; semantic joins are checked by the fold."""
    return dict(schema='farm-v0-regret-budget-wire-contract-v1', registration_schema=LIMITS_SCHEMA,
        registration_fields=[*LIMIT_FIELDS, 'content_sha256'], event_schema=EVENT_SCHEMA,
        event_fields=[*EVENT_FIELDS, 'content_sha256'], payload_fields={k: list(v) for k, v in PAYLOAD_FIELDS.items()},
        world_fields=list(WORLD_FIELDS), seat_row_fields=list(ROW_FIELDS), path_fields=list(PATH_FIELDS),
        cost_fields=list(COSTS), cap_fields=list(CAPS),
        semantics=dict(hash='SHA256_SORTED_COMPACT_FINITE_JSON_EXCLUDING_OWN_content_sha256',
            initial_previous_event='registration.content_sha256', sequence='ZERO_BASED_APPEND_ONLY',
            increment_costs='NONOVERLAPPING_ELAPSED_CHARGES; PHYSICAL_STARTS; CPU_SECONDS',
            edit_depth='SEARCH_LINEAGE_NOT_NATIVE_ACTION_COUNT', WDL='LOSS=-1,DRAW=0,WIN=1',
            discovery_selection='MAX_COMPLETE_WDL_GAIN_STABLE_FIRST_POSITIVE',
            confirmation='ONLY_PREVIOUSLY_FROZEN_DISCOVERY_WINNER',
            admission='EXISTING_admit_pair_RERUN_WITH_EXACT_BUNDLE_PROTOCOL_AND_LEDGER_JOINS',
            comparison_input='TWO_RAW_{limits,history}_BUNDLES_NOT_SUMMARIES',
            strata_source='PROPOSAL_FAMILY_NOT_INDEPENDENT_WORLD_FAMILY',
            strata_costs='INCLUDES_REPLAY_FAILURES_AND_REFUSED_EVENTS_WITH_IDENTIFIABLE_SOURCE_DEPTH',
            selection_at_candidate_cap='EVERY_RETAINED_CANDIDATE_HAS_A_COMPLETE_OR_CENSORED_DISCOVERY_BATCH',
            exhausted_global_start_or_time_cap='NO_FURTHER_POSITIVE_COST_EVENT_IS_AUTHORIZED'),
        adapters=dict(proposals='path_binding(actual_proposal); every pool.attempts row becomes one proposal event; '
                               'pool.attempt_id is producer provenance, not a new independent world',
                      evaluation='existing regret-record evaluation.seat_rows, partitioned by registered phase',
                      admission='existing confirmation bundle + full admission protocol'), limitations=list(LIMITATIONS))


# This versioned discovery lane deliberately leaves fixed v1 accounting unchanged.
MC_ALLOCATION_SCHEMA = 'farm-v0-regret-mc-allocation-registration-v1'
MC_ALLOCATION_EVENT_SCHEMA = 'farm-v0-regret-mc-allocation-event-v1'
MC_ALLOCATION_RESULT_SCHEMA = 'farm-v0-regret-mc-allocation-result-v1'
_MC_ALGORITHM = 'PAIRED_FAMILY_HOEFFDING_ELIMINATION_V1'
_MC_KEY_FIELDS = ('candidate_id', 'source_group_id', 'seat')
_MC_CELL_FIELDS = (*_MC_KEY_FIELDS, 'status', 'learner_wdl', 'challenger_wdl',
                   'learner_replay_sha256', 'simulator_starts', 'physical_starts')
_MC_STATUSES = ('COMPLETE', 'TIMEOUT', 'CENSORED', 'CRASHED', 'INVALID',
                'CACHE_HIT', 'REPLAY_ONLY', 'NOT_STARTED')
_MC_LIMITATIONS = [
    'DISCOVERY_SELECTION_ONLY; DISJOINT_CONFIRMATION_STILL_REQUIRED',
    'PREDECLARATION_REPLAY_AND_INDEPENDENT_FAMILY_CUSTODY_ARE_CALLER_RESPONSIBILITIES',
    'SIMULATOR_START_SLOTS_ARE_PHYSICAL_START_EQUIVALENTS; FIXTURES_HAVE_ZERO_PHYSICAL_GAMES',
    'CENSORED_FAMILIES_HAVE_WORST_CASE_BOUNDS; AVAILABLE_CASE_MEANS_ARE_DIAGNOSTIC_ONLY',
    'COMMON_RNG_KEYS_DO_NOT_CERTIFY_COMMON_EXOGENOUS_EVENTS',
    'PURE_SCHEDULER_NOT_AN_EXECUTOR_OR_DURABLE_CRASH_RECOVERY_SERVICE',
]


def register_mc_allocation(limits: dict, candidates: list[dict], *, mode: str,
                           alpha: float = .05, minimum_families: int = 8) -> dict:
    """Freeze a discovery-only allocation rule before observing any WDL.

    Reuses v1's source/context/path identities, disjoint world registry and caps;
    ``limits.arm`` must remain fixed: that INNER object only supplies identity and
    resource limits, NOT the allocation rule. The OUTER mode is uniform/adaptive.
    All caps in this API are DISCOVERY caps; confirmation is separately budgeted.
    ``candidates[0]`` is the frozen learner, followed by exact successor/path
    bindings from ``path_binding``/``candidate_key``. Reachability remains owned
    by the existing proposer/MC estimator, not these hashes.

    Each batch is one whole preregistered source family, all worlds/seats and all
    active candidates, including learner. The existing MC estimator needs four
    starts per cell: learner + challenger + both fresh replays. Families follow a
    seed-hashed order independent of outcomes. No replenishment, trees or retries.
    """
    _registration(limits)
    _need(limits['arm'] == 'fixed', 'MC_BASE_MUST_BE_FIXED')
    _need(mode in ('uniform', 'adaptive'), 'MC_ALLOCATION_MODE')
    _number(alpha, 'alpha', low=1e-12, high=.25)
    _number(minimum_families, 'minimum_families', low=1, high=8192, integer=True)
    _need(type(candidates) is list and 2 <= len(candidates) <= min(32, limits['caps']['candidate_attempts']),
          'MC_CANDIDATE_CAP')
    ids = set()
    for candidate in candidates:
        _fields(candidate, ('candidate_id', 'paths'), 'candidate')
        cid = candidate_key(candidate['paths'], limits['actor_context_sha256s'])
        _need(candidate['candidate_id'] == cid, 'CANDIDATE_ID_MISMATCH')
        _need(cid not in ids, 'MC_DUPLICATE_CANDIDATE')
        ids.add(cid)
    worlds = limits['worlds']['discovery']
    _need(len(worlds) <= 8192, 'MC_WORLD_LIMIT')
    # The fixed-budget join omits snapshot/RNG overlap; the adaptive lane also
    # refuses those identities across phases, before any external work is issued.
    for field in ('source_group_id', 'source_family_id', 'world_sha256', 'snapshot_sha256', 'rng_sha256', 'seed'):
        exposed = {w[field] for w in worlds}
        _need(not any(w[field] in exposed for phase in ('confirmation', 'protected_final')
                      for w in limits['worlds'][phase]), 'MC_PROTECTED_OVERLAP', field)
    families = {w['source_family_id'] for w in worlds}
    order = sorted(families, key=lambda f: (digest([_MC_ALGORITHM, limits['seed'], f]), f))
    return bind(dict(schema=MC_ALLOCATION_SCHEMA, limits=limits, candidates=candidates,
                     mode=mode, alpha=alpha, minimum_families=minimum_families,
                     algorithm=_MC_ALGORITHM, family_order=order))


class MonteCarloAllocator:
    """Deterministic, replayable allocator over a closed discovery-only event wire.

    ``next_request()`` returns a detached, hash-bound whole-family request (or
    None). Persist it before external execution, run only its cells with the
    accepted MC estimator and call ``observe(event)`` once. A request reserves
    four start slots per cell BEFORE execution; a censored/timed-out attempt is
    charged, never retried. If actual lost costs are unknown, submit CENSORED
    cells charged at their four-start reservation, not a fresh request. This
    class does not execute callbacks or provide transactional crash persistence.
    ``history`` replays exactly those requests; no teacher/final data fields exist.

    Statistical rule, preregistered for both arms:
    For K candidates, G family looks, M=K(K-1)/2 unordered comparisons, a paired
    family WDL difference lies in [-2,2]. Use radius
    sqrt(8*log(2*M*G/alpha)/n). The two-sided bounded-mean tail is at most
    2*exp(-n*r*r/8); union over M*G covers all possible elimination/stopping looks.
    Validity assumes independent common-mean family units (or uniform sampling
    without replacement from the frozen bounded family panel). Families/seats
    are NOT authenticated independent by their names. For a missing family,
    bound its unknown difference by [-2,2] in the mean before adding the radius.
    Thus censoring may depend on WDL without complete-case selection. Verified n
    excludes misses, caches and replay-only rows; allocated n counts latent slots,
    never pretends the missing values were observations.

    Adaptive mode eliminates A only when UCB(A-B)<0 for an active B/learner.
    Uniform mode uses the identical rule only at final selection and never prunes
    its schedule. A positive-mean guess is not a winner: the surviving winner must
    have LCB>0 versus learner and every other live candidate. Previously eliminated
    candidates retain their simultaneous dominance witnesses in the trace.
    """
    def __init__(self, registration: dict, *, history=()):
        _fields(registration, ('schema', 'limits', 'candidates', 'mode', 'alpha',
            'minimum_families', 'algorithm', 'family_order', 'content_sha256'), 'mc_registration')
        _bound(registration, 'MC_REGISTRATION_HASH')
        expected = register_mc_allocation(registration['limits'], registration['candidates'],
            mode=registration['mode'], alpha=registration['alpha'],
            minimum_families=registration['minimum_families'])
        _need(registration == expected, 'MC_REGISTRATION_MISMATCH')
        self._reg = copy.deepcopy(registration)
        self._limits = self._reg['limits']
        self._ids = [c['candidate_id'] for c in self._reg['candidates']]
        self._active = list(self._ids)
        self._values = {cid: {} for cid in self._ids}
        self._history, self._trace, self._eliminated = [], [], []
        self._totals = dict(simulator_starts=0, physical_starts=0, cpu_seconds=0., elapsed_seconds=0.,
                            cache_hits=0, replay_only_cells=0, censored_cells=0, completed_cells=0)
        self._families = {f: sorted([w for w in self._limits['worlds']['discovery']
                                    if w['source_family_id'] == f], key=lambda w: w['source_group_id'])
                          for f in self._reg['family_order']}
        _need(type(history) in (list, tuple), 'MC_HISTORY_TYPE')
        for event in history:
            self.observe(event)

    def _stop(self):
        if len(self._active) == 1:
            return 'NO_PLAUSIBLE_CHALLENGER'
        if len(self._history) == len(self._families):
            return 'FAMILY_PANEL_EXHAUSTED'
        caps = self._limits['caps']
        if any(self._totals[k] >= caps[k] for k in ('cpu_seconds', 'elapsed_seconds')):
            return 'TIME_CAP'
        family = self._reg['family_order'][len(self._history)]
        needed = 4 * len(self._families[family]) * len(self._limits['seats']) * len(self._active)
        if self._totals['simulator_starts'] + needed > caps['native_starts']:
            return 'START_CAP_NO_WHOLE_FAMILY'
        return None

    def next_request(self) -> dict | None:
        if self._stop() is not None:
            return None
        index = len(self._history)
        family = self._reg['family_order'][index]
        cells = [dict(candidate_id=cid, source_group_id=w['source_group_id'], seat=seat)
                 for w in self._families[family] for seat in self._limits['seats'] for cid in self._active]
        return bind(dict(schema='farm-v0-regret-mc-allocation-request-v1',
            registration_sha256=self._reg['content_sha256'], sequence=index, family_id=family,
            previous_event_sha256=self._history[-1]['content_sha256'] if self._history else self._reg['content_sha256'],
            phase='discovery', cells=cells, reserved_start_slots=4 * len(cells),
            remaining_start_slots=self._limits['caps']['native_starts'] - self._totals['simulator_starts']))

    def _comparison(self, left, right):
        a, b = self._values[left], self._values[right]
        common = sorted(set(a) & set(b))
        values = [a[f] - b[f] for f in common if a[f] is not None and b[f] is not None]
        n, observed = len(common), len(values)
        total = math.fsum(values)
        radius = math.sqrt(8 * math.log(len(self._ids) * (len(self._ids)-1) *
                           len(self._families) / self._reg['alpha']) / n) if n else None
        lo = max(-2., (total-2*(n-observed))/n-radius) if n else -2.
        hi = min(2., (total+2*(n-observed))/n+radius) if n else 2.
        return dict(left_id=left, right_id=right, allocated_family_n=n, verified_family_n=observed,
            censored_family_n=n-observed, mean_wdl_delta=total/observed if observed else None,
            lower_bound=lo, upper_bound=hi, radius=radius,
            uncertainty='SIMULTANEOUS_DISCOVERY_BOUND_WITH_WORST_CASE_MISSINGNESS')

    def observe(self, event: dict) -> None:
        """Consume one complete requested batch, validating before any mutation.

        Low-level events are source-bound producer attestations, not certificates
        of execution. Prefer ``mc_allocation_event`` to derive them from the raw MC
        estimator receipts. A rejected event must be retained by the collector
        with its costs; it authorizes no retry. Observing a corrected *receipt*
        does not authorize rerunning an already charged physical attempt.
        """
        req = self.next_request()
        _need(req is not None, 'MC_ALLOCATION_ALREADY_STOPPED')
        _fields(event, ('schema', 'request_sha256', 'phase', 'cells', 'cpu_seconds',
                        'elapsed_seconds', 'producer_receipt_sha256', 'content_sha256'), 'mc_event')
        _bound(event, 'MC_EVENT_HASH')
        _need(event['schema'] == MC_ALLOCATION_EVENT_SCHEMA, 'MC_EVENT_SCHEMA')
        _need(event['phase'] == 'discovery', 'MC_DISCOVERY_ONLY')
        _need(event['request_sha256'] == req['content_sha256'], 'MC_REQUEST_MISMATCH')
        _hash(event['producer_receipt_sha256'], 'producer_receipt_sha256')
        for key in ('cpu_seconds', 'elapsed_seconds'):
            _number(event[key], key)
        _need(type(event['cells']) is list and len(event['cells']) == len(req['cells']), 'MC_MISSING_CELLS')
        costs = dict.fromkeys(self._totals, 0)
        by_candidate = {cid: [] for cid in self._active}
        controls = {}
        for cell, key in zip(event['cells'], req['cells']):
            _fields(cell, _MC_CELL_FIELDS, 'mc_cell')
            _need({k: cell[k] for k in _MC_KEY_FIELDS} == key, 'MC_UNREQUESTED_CELL')
            status = cell['status']
            _need(status in _MC_STATUSES, 'MC_CELL_STATUS')
            starts = _number(cell['simulator_starts'], 'simulator_starts', high=4, integer=True)
            physical = _number(cell['physical_starts'], 'physical_starts', high=4, integer=True)
            _need(physical == (starts if self._limits['evidence_kind'] == 'NATIVE_GAME' else 0),
                  'MC_PHYSICAL_START_MISMATCH')
            _need(starts == 4 if status == 'COMPLETE' else
                  starts == 0 if status in ('CACHE_HIT', 'NOT_STARTED') else starts > 0,
                  'MC_UNCHARGED_OR_FALSE_FRESH_SAMPLE')
            costs['simulator_starts'] += starts; costs['physical_starts'] += physical
            costs['cache_hits'] += int(status == 'CACHE_HIT')
            costs['replay_only_cells'] += int(status == 'REPLAY_ONLY')
            costs['completed_cells' if status == 'COMPLETE' else 'censored_cells'] += 1
            if status == 'COMPLETE':
                for field in ('learner_wdl', 'challenger_wdl'):
                    _need(type(cell[field]) is int and cell[field] in (-1, 0, 1), 'MC_NON_WDL')
                _hash(cell['learner_replay_sha256'], 'learner_replay_sha256')
                identity = (cell['source_group_id'], cell['seat'])
                control = (cell['learner_wdl'], cell['learner_replay_sha256'])
                _need(identity not in controls or controls[identity] == control, 'MC_CONTROL_MISMATCH')
                controls[identity] = control
                if cell['candidate_id'] == self._ids[0]:
                    _need(cell['learner_wdl'] == cell['challenger_wdl'], 'MC_LEARNER_SELF_MISMATCH')
                value = cell['challenger_wdl']
            else:
                _need(all(cell[k] is None for k in ('learner_wdl', 'challenger_wdl', 'learner_replay_sha256')),
                      'MC_UNVERIFIED_OUTCOME')
                value = None
            by_candidate[cell['candidate_id']].append(value)
        _need(costs['simulator_starts'] <= req['reserved_start_slots'], 'MC_START_RESERVATION_EXCEEDED')
        # CPU/wall excess is charged and stops subsequent allocation. It cannot
        # create precision or authorize another callback. Physical cap cannot be
        # overrun by an event satisfying the four-start-per-cell contract.
        costs.update(cpu_seconds=event['cpu_seconds'], elapsed_seconds=event['elapsed_seconds'])
        for key, value in costs.items():
            self._totals[key] += value
        self._history.append(copy.deepcopy(event))
        for cid, values in by_candidate.items():
            self._values[cid][req['family_id']] = (math.fsum(values)/len(values)
                                                   if all(v is not None for v in values) else None)
        eliminated = []
        if self._reg['mode'] == 'adaptive' and len(self._history) >= self._reg['minimum_families']:
            for cid in self._active[1:]:
                witnesses = [self._comparison(cid, other) for other in self._active if other != cid]
                witness = min(witnesses, key=lambda row: (row['upper_bound'], self._ids.index(row['right_id'])))
                if witness['upper_bound'] < 0:
                    eliminated.append(dict(candidate_id=cid, against_id=witness['right_id'],
                        after_family_n=len(self._history), reason='PAIRED_UPPER_BOUND_BELOW_ZERO', **witness))
            removed = {e['candidate_id'] for e in eliminated}
            self._active = [cid for cid in self._active if cid not in removed]
            self._eliminated.extend(eliminated)
        self._trace.append(dict(sequence=req['sequence'], request_sha256=req['content_sha256'],
            event_sha256=event['content_sha256'], family_id=req['family_id'],
            allocated_candidates=list(by_candidate), active_candidates=list(self._active),
            simulator_starts=self._totals['simulator_starts'], eliminated=eliminated))

    def receipt(self) -> dict:
        """Detached result; RUNNING receipts cannot be frozen into selections."""
        stop = self._stop()
        comparisons = [self._comparison(a, b) for i, a in enumerate(self._ids) for b in self._ids[i+1:]]
        winner = None
        if stop is not None and len(self._history) >= self._reg['minimum_families']:
            for cid in self._active[1:]:
                if all(self._comparison(cid, other)['lower_bound'] > 0 for other in self._active if other != cid):
                    winner = cid
                    break
        stats = {}
        for cid, families in self._values.items():
            values = [v for v in families.values() if v is not None]
            n = len(values); mean = math.fsum(values)/n if n else None
            variance = math.fsum((v-mean)**2 for v in values)/(n-1) if n > 1 else None
            stats[cid] = dict(allocated_family_n=len(families), verified_family_n=n,
                censored_family_n=len(families)-n, utility_mean=mean, sample_variance=variance,
                standard_error=math.sqrt(variance/n) if n > 1 else None,
                mean_scope='AVAILABLE_CASE_DIAGNOSTIC_ONLY',
                versus_learner=self._comparison(cid, self._ids[0]) if cid != self._ids[0] else None)
        selection = None
        if winner:
            selection = bind(dict(schema='farm-v0-regret-mc-frozen-selection-v1',
                registration_sha256=self._reg['content_sha256'], history_sha256=digest(self._history),
                candidate_id=winner, paths=next(c['paths'] for c in self._reg['candidates'] if c['candidate_id'] == winner),
                parent_checkpoint_sha256=self._limits['parent_checkpoint_sha256'],
                actor_context_sha256s=self._limits['actor_context_sha256s'],
                learner_path_sha256s=self._limits['learner_path_sha256s'],
                confirmation_registration_sha256=None, actor_training_admissible=False))
        return bind(dict(schema=MC_ALLOCATION_RESULT_SCHEMA,
            status='RUNNING' if stop is None else 'SELECTED_FOR_CONFIRMATION' if winner else 'NO_CLEAR_WINNER',
            stop_reason=stop, registration_sha256=self._reg['content_sha256'], history_sha256=digest(self._history),
            mode=self._reg['mode'], learner_id=self._ids[0], winner_id=winner,
            active_candidates=self._active, totals=self._totals, statistics=stats, comparisons=comparisons,
            trace=self._trace, eliminations=self._eliminated, frozen_selection=selection,
            next_request=self.next_request(),
            unused_start_slots=self._limits['caps']['native_starts']-self._totals['simulator_starts'],
            time_cap_exceeded=any(self._totals[k] > self._limits['caps'][k] for k in ('cpu_seconds','elapsed_seconds')),
            actor_training_admissible=False, teacher_weight=0., confirmation_required=True,
            limitations=list(_MC_LIMITATIONS)))


def compare_mc_allocations(uniform: dict, adaptive: dict) -> dict:
    """Recompute raw histories; match caps/inputs, report actual cost differences.

    Unlike the fixed-budget comparison, equality of declared caps does not assert
    equality of actual costs (e.g. no plausible winner or partial final family).
    Never select a mode from confirmation/final outcomes; those fields are refused.
    """
    for run in (uniform, adaptive):
        _fields(run, ('registration', 'history'), 'mc_run')
    a, b = uniform['registration'], adaptive['registration']
    _need(a['mode'] == 'uniform' and b['mode'] == 'adaptive', 'MC_COMPARISON_ARMS')
    _need({k: v for k, v in a.items() if k not in ('mode', 'content_sha256')} ==
          {k: v for k, v in b.items() if k not in ('mode', 'content_sha256')}, 'MC_COMPARISON_MISMATCH')
    left = MonteCarloAllocator(a, history=uniform['history']).receipt()
    right = MonteCarloAllocator(b, history=adaptive['history']).receipt()
    _need(left['status'] != 'RUNNING' and right['status'] != 'RUNNING', 'MC_COMPARISON_UNFINISHED')
    return bind(dict(schema='farm-v0-regret-mc-allocation-comparison-v1', status='MATCHED_CAPS',
        physical_start_cap=a['limits']['caps']['native_starts'], uniform=left, adaptive=right,
        actual_start_costs_equal=left['totals']['simulator_starts'] == right['totals']['simulator_starts'],
        actual_physical_starts_equal=left['totals']['physical_starts'] == right['totals']['physical_starts'],
        decision='DISCOVERY_ONLY_NOT_A_PLAYING_STRENGTH_OR_CONFIRMATION_RESULT'))


def mc_allocation_event(request: dict, results: list[dict], *, registration: dict) -> dict:
    """Derive ONE allocation event from existing ``estimate_rollouts`` receipts.

    Supply one MC receipt per registered seat, restricted to the request's family
    and active candidates, learner first. No calls to a runner are made here.
    Revalidates source/world/context/frozen-draw/path joins and each physical
    attempt with the paired evaluator's validator. WDL is recomputed from raw
    terminal responses and fresh semantic replay, NEVER from MC ranking, means,
    uncertainty, predictor scores or producer ``winner_id``/``VERIFIED`` flags.
    All failed attempts stay charged. A structurally unjoinable receipt raises:
    the caller must retain it and its costs, not retry physical work.

    This adapter currently consumes unique jobs (the allocator never requests a
    duplicate family). External cache/replay-only evidence may use the closed
    low-level event wire but cannot increment verified family n. It deliberately
    does not implement the neighboring cache task or execute a native campaign.
    """
    from farm_neat.evaluation import v0_regret_pairs as E
    from farm_neat.training.teachers import v0_regret_searcher as S
    allocator = MonteCarloAllocator(registration)  # validates current source pins
    limits = allocator._limits
    _bound(request, 'MC_REQUEST_HASH')
    _need(request['registration_sha256'] == registration['content_sha256'] and request['phase'] == 'discovery',
          'MC_REQUEST_MISMATCH')
    _need(type(results) is list and len(results) == len(limits['seats']), 'MC_SEAT_RECEIPTS')
    wanted = {(c['candidate_id'], c['source_group_id'], c['seat']) for c in request['cells']}
    _need(len(wanted) == len(request['cells']), 'MC_DUPLICATE_REQUEST_CELL')
    registered = {w['source_group_id']: w for w in limits['worlds']['discovery']}
    wanted_groups = {g for _, g, _ in wanted}
    wanted_ids = {c for c, _, _ in wanted}
    sources = E.source_pins() | {p: hashlib.sha256((S.R.ROOT/p).read_bytes()).hexdigest() for p in S.SOURCE_PATHS}
    records, seen_attempts, seen_seats = {}, set(), set()
    elapsed = cpu = 0.
    result_fields = ('schema', 'status', 'stop_reason', 'seed', 'budget', 'source_pins', 'binding', 'execution',
        'registration', 'draw_order', 'candidates', 'rejected_candidates', 'duplicate_candidates',
        'comparisons', 'attempts', 'jobs', 'counts', 'best_so_far', 'common_group_ids', 'ranking',
        'best_id', 'winner_id', 'actor_training_admissible', 'teacher_weight', 'confirmation_required',
        'limitations', 'content_sha256')
    for result in results:
        _fields(result, result_fields, 'mc_receipt')
        _bound(result, 'MC_RECEIPT_HASH')
        _need(result['schema'] == S.MC_SCHEMA and result['status'] == 'DISCOVERY_ONLY', 'MC_DISCOVERY_ONLY')
        _need(result['source_pins'] == sources, 'MC_SOURCE_MISMATCH')
        _need(result['actor_training_admissible'] is False and result['teacher_weight'] == 0.
              and result['confirmation_required'] is True, 'MC_LABEL_FORBIDDEN')
        protocol = S._protocol(result['registration'], S._limits(result['budget'])[0])
        expected_worlds = [{k: v for k, v in registered[g].items() if k != 'source_family_id'}
                           for g in sorted(wanted_groups)]
        _need(sorted(protocol['discovery_world_groups'], key=lambda w: w['source_group_id']) == expected_worlds,
              'MC_DISCOVERY_WORLD_MISMATCH')
        _need(all(protocol['source_families'][g] == registered[g]['source_family_id'] for g in wanted_groups),
              'MC_FAMILY_MISMATCH')
        _need(protocol['opponent_sha256'] == limits['opponent_sha256'], 'MC_OPPONENT_MISMATCH')
        native = result['execution']['kind'] == 'NATIVE_GAME'
        _need(native == (limits['evidence_kind'] == 'NATIVE_GAME'), 'MC_EVIDENCE_KIND_MISMATCH')
        _need(result['jobs'] and not result['rejected_candidates'] and not result['duplicate_candidates'],
              'MC_INCOMPLETE_OR_DUPLICATE_POOL')
        seat = result['jobs'][0]['shared_identity']['seat']
        _need(type(seat) is int and seat in limits['seats'] and seat not in seen_seats, 'MC_SEAT_MISMATCH')
        seen_seats.add(seat)
        local = {}
        for candidate in result['candidates']:
            matches = [c['candidate_id'] for c in registration['candidates']
                       if c['paths'][str(seat)] == path_binding(candidate['geometry'])]
            _need(len(matches) == 1 and candidate['candidate_id'] not in local, 'MC_PATH_MISMATCH')
            local[candidate['candidate_id']] = matches[0]
        _need(list(local.values())[0] == registration['candidates'][0]['candidate_id']
              and len(local) == len(wanted_ids) and set(local.values()) == wanted_ids, 'MC_REQUESTED_POOL_MISMATCH')
        raw_attempts = result['attempts']
        _need(result['counts']['simulator_starts'] == len(raw_attempts) and
              result['counts']['physical_starts'] == int(native)*len(raw_attempts), 'MC_START_COUNT_MISMATCH')
        for key in ('cpu_seconds', 'wall_seconds'):
            _number(result['counts'][key], key)
        cpu += result['counts']['cpu_seconds']; elapsed += result['counts']['wall_seconds']
        used = set()
        for job in result['jobs']:
            _need(job['candidate_id'] in local, 'MC_UNKNOWN_CANDIDATE')
            cid, gid = local[job['candidate_id']], job['group_id']
            cell_key = (cid, gid, seat)
            _need(cell_key in wanted and cell_key not in records, 'MC_UNREQUESTED_OR_DUPLICATE_JOB')
            attempts = [a for a in raw_attempts if a['job_id'] == job['job_id']]
            _need(len(attempts) <= 4, 'MC_EXCESS_JOB_STARTS')
            _need([(a['arm'], a['phase']) for a in attempts] ==
                  [('learner','EVALUATE'), ('challenger','EVALUATE'), ('learner','REPLAY'), ('challenger','REPLAY')][:len(attempts)],
                  'MC_ATTEMPT_ORDER')
            terminal, rebuilt = [], []
            req = job.get('request')
            if attempts:
                req = E._validate_request(req)
                rec = req['record']
                _need(rec['identity']['split'] == 'discovery' and rec['identity']['source_group_id'] == gid,
                      'MC_DISCOVERY_ONLY')
                _need(req['seat'] == seat and rec['identity']['actor_context_sha256'] ==
                      limits['actor_context_sha256s'][str(seat)] and rec['identity']['parent_checkpoint_sha256'] ==
                      limits['parent_checkpoint_sha256'], 'MC_CONTEXT_MISMATCH')
                _need(rec['learner']['path_sha256'] == limits['learner_path_sha256s'][str(seat)] and
                      path_binding(rec['learner']) == registration['candidates'][0]['paths'][str(seat)],
                      'MC_LEARNER_DRAW_MISMATCH')
                _need(path_binding(rec['challenger']) == next(c['paths'][str(seat)] for c in registration['candidates']
                      if c['candidate_id'] == cid), 'MC_PATH_MISMATCH')
                actual = dict(source_group_id=gid, world_sha256=req['world_sha256'],
                    snapshot_sha256=req['initial_world_sha256'], rng_sha256=digest(req['rng_binding']), seed=req['world']['seed'])
                _need(actual == {k: v for k,v in registered[gid].items() if k != 'source_family_id'}, 'MC_WORLD_MISMATCH')
                _need(digest(rec['continuation']) == protocol['continuation_sha256'], 'MC_CONTINUATION_MISMATCH')
                _need(req['opponent']['identity']['content_sha256'] == limits['opponent_sha256'], 'MC_OPPONENT_MISMATCH')
            for a in attempts:
                branch = a['branch']
                E._bound(branch, 'MC_BRANCH_HASH')
                _need(branch['request'] == req and branch['execution'] == result['execution'] and
                      branch['arm'] == a['arm'] and branch['phase'] == a['phase'], 'MC_BRANCH_MISMATCH')
                aid = branch['attempt_id']
                _need(aid not in seen_attempts, 'MC_REUSED_ATTEMPT')
                seen_attempts.add(aid); used.add(id(a))
                _need(a['simulator_starts'] == 1 and a['physical_starts'] == int(native), 'MC_UNCHARGED_START')
                response = a['outcome']['response']
                # Invalid raw responses are censoring, not zero-cost absence and
                # not trustworthy terminal evidence even if producer flags say so.
                try:
                    value = E._validate_response(branch, response)
                except (ValueError, KeyError, TypeError):
                    value = None
                terminal.append(value)
                rebuilt.append(dict(status='COMPLETE' if value is not None else 'INVALID',
                                    response=response, error=None))
            complete = (len(terminal) == 4 and all(t is not None for t in terminal) and
                all(E.P.wire(E._replay_signature(rebuilt[i])) == E.P.wire(E._replay_signature(rebuilt[i+2]))
                    for i in (0,1)))
            status = 'COMPLETE' if complete else 'CENSORED' if attempts else 'NOT_STARTED'
            records[cell_key] = dict(candidate_id=cid, source_group_id=gid, seat=seat, status=status,
                learner_wdl=terminal[0]['wdl'] if complete else None,
                challenger_wdl=terminal[1]['wdl'] if complete else None,
                learner_replay_sha256=E.P.wire_hash(E._replay_signature(rebuilt[0])) if complete else None,
                simulator_starts=len(attempts), physical_starts=int(native)*len(attempts))
        _need(len(used) == len(raw_attempts), 'MC_UNACCOUNTED_ATTEMPTS')
    _need(set(records) == wanted, 'MC_MISSING_CELLS')
    return bind(dict(schema=MC_ALLOCATION_EVENT_SCHEMA, request_sha256=request['content_sha256'],
        phase='discovery', cells=[records[tuple(k[f] for f in _MC_KEY_FIELDS)] for k in request['cells']],
        cpu_seconds=cpu, elapsed_seconds=elapsed, producer_receipt_sha256=digest(results)))


def allocation_selection_receipt(result: dict, proposal: dict, *, seat: int) -> dict:
    """Map a frozen winner + exact producer geometry to existing R.SELECTION_SCHEMA.

    This is not admission. The collector must independently freeze and join the
    normal disjoint confirmation registration, assemble actual paired records and
    run ``admit_pair``. No confidence, allocated count or discovery result enters
    confirmation n. The canonical proposer/records validator still replays paths.
    """
    from farm_neat.training import v0_regret_records as R
    _bound(result, 'MC_RESULT_HASH')
    _need(result.get('schema') == MC_ALLOCATION_RESULT_SCHEMA and
          result.get('status') == 'SELECTED_FOR_CONFIRMATION' and result.get('winner_id') is not None,
          'MC_NO_FROZEN_WINNER')
    selected = result['frozen_selection']
    _bound(selected, 'MC_SELECTION_HASH')
    _need(selected['candidate_id'] == result['winner_id'] and selected['registration_sha256'] ==
          result['registration_sha256'] and selected['history_sha256'] == result['history_sha256'], 'MC_SELECTION_MISMATCH')
    _need(type(seat) is int and str(seat) in selected['paths'], 'MC_SEAT_MISMATCH')
    _need(path_binding(proposal) == selected['paths'][str(seat)], 'MC_SELECTED_PATH_MISMATCH')
    return dict(schema=R.SELECTION_SCHEMA, actor_context_sha256=selected['actor_context_sha256s'][str(seat)],
        parent_checkpoint_sha256=selected['parent_checkpoint_sha256'], raw_actions_sha256=digest(proposal['raw_actions']),
        continuation_signature=copy.deepcopy(proposal['continuation_signature']),
        pool_sha256=selected['content_sha256'], confirmation_registration_sha256=None)


# Hypothesis-level composition. The original fixed/random and candidate-level
# MC wires above are unchanged; their cost, path and replay validators still own
# each whole-family batch. These outputs are private discovery evidence ONLY.
BELIEF_MC_SCHEMA = 'farm-v0-belief-mc-registration-v1'
BELIEF_MC_EVENT_SCHEMA = 'farm-v0-belief-mc-event-v1'
BELIEF_MC_RESULT_SCHEMA = 'farm-v0-belief-mc-result-v1'
BELIEF_MC_MODES = ('mass', 'uniform', 'uncertainty', 'decision_impact')
_BELIEF_OBJECTIVE = 'MIXTURE_TERMINAL_WIN_PROBABILITY; DISCOVERY_DIAGNOSTIC_ONLY'


def register_belief_mc_allocation(hypotheses: list[dict], *, mode: str, caps: dict,
        seed: int, actor_payload_sha256: str, belief_sha256: str, pair_bounds=(),
        stop_when_certified: bool = False, alpha: float = .05) -> dict:
    """Freeze a private allocation over {hypothesis_id, mass, allocation} rows.

    ``allocation`` is an EXISTING uniform ``register_mc_allocation`` result, or
    None for an unexecutable/UNKNOWN hypothesis. All rows share frozen candidates,
    learner, actor contexts and seats; each can bind a different hypothetical
    controller/panel. Masses are fixed, must sum to one, and are NEVER renormalized
    over available rows. Low-level identity/mass declarations need producer custody;
    prefer ``register_belief_mc_from_record`` for the canonical actor-record join.

    Optional pair_bounds are source-bound PER-FAMILY SUPPORT claims on
    mean_world,seat(1[left wins]-1[right wins]), in [-1,1], not cash, WDL utility,
    estimated mean intervals or fitted priors. Without a proof use [-1,1] (default).
    A false bound invalidates conclusions even before a violation is observed.
    Neither a digest nor checking sampled values proves a universal support claim.

    Caps charge simulator-start slots (physical-start equivalents in fixtures),
    including both fresh replays, across ALL hypotheses. No executor is called.
    stop_when_certified=False spends matched caps for controlled comparisons;
    True may stop without sampling a tail whose worst case cannot change selection.
    This module does not update beliefs, freeze teacher selections or admit labels.
    """
    _need(mode in BELIEF_MC_MODES, 'BELIEF_ALLOCATION_MODE')
    _fields(caps, COSTS, 'belief_caps')
    for k in COSTS:
        _number(caps[k], k, integer=k == 'native_starts')
    _number(seed, 'seed', integer=True)
    _number(alpha, 'alpha', low=1e-12, high=.25)
    _need(type(stop_when_certified) is bool, 'BELIEF_STOP_FLAG')
    for key, value in (('actor_payload_sha256', actor_payload_sha256), ('belief_sha256', belief_sha256)):
        _hash(value, key)
    _need(type(hypotheses) is list and 1 <= len(hypotheses) <= 64, 'BELIEF_HYPOTHESIS_LIMIT')
    seen, live = set(), []
    for row in hypotheses:
        _fields(row, ('hypothesis_id', 'mass', 'allocation'), 'hypothesis')
        h = row['hypothesis_id']
        _text(h, 'hypothesis_id'); _number(row['mass'], 'mass', high=1.)
        _need(h not in seen, 'BELIEF_DUPLICATE_HYPOTHESIS'); seen.add(h)
        if row['allocation'] is not None:
            # Reuse the existing complete registration/semantic-source validator.
            child = MonteCarloAllocator(row['allocation'])
            _need(child._reg['mode'] == 'uniform', 'BELIEF_CHILD_MUST_BE_UNIFORM')
            live.append(child._reg)
    _need(abs(math.fsum(h['mass'] for h in hypotheses)-1.) <= 1e-12, 'BELIEF_MASS_SUM')
    _need(bool(live), 'BELIEF_NO_EXECUTABLE_PANEL')
    first = live[0]
    shared = ('comparison_id', 'evidence_kind', 'parent_checkpoint_sha256', 'source_pins',
              'source_group_id', 'actor_context_sha256s', 'learner_path_sha256s', 'seats')
    for child in live[1:]:
        _need(child['candidates'] == first['candidates'], 'BELIEF_CANDIDATE_MISMATCH')
        _need(all(child['limits'][k] == first['limits'][k] for k in shared), 'BELIEF_CONTEXT_MISMATCH')
    # A discovery panel in one hypothetical controller must not expose any other
    # controller's confirmation/final panel. No cross-hypothesis n is pooled.
    for field in WORLD_FIELDS:
        exposed = {w[field] for c in live for w in c['limits']['worlds']['discovery']}
        _need(not any(w[field] in exposed for c in live for phase in ('confirmation', 'protected_final')
                      for w in c['limits']['worlds'][phase]), 'BELIEF_PROTECTED_OVERLAP', field)
    ids = [c['candidate_id'] for c in first['candidates']]
    pairs = {(a,b) for i,a in enumerate(ids) for b in ids[i+1:]}
    _need(type(pair_bounds) in (list, tuple), 'BELIEF_BOUND_TYPE')
    bound_keys = set()
    for row in pair_bounds:
        _fields(row, ('hypothesis_id', 'left_id', 'right_id', 'lower_bound', 'upper_bound',
                      'evidence_sha256'), 'belief_bound')
        key = (row['hypothesis_id'], row['left_id'], row['right_id'])
        _need(key[0] in seen and key[1:] in pairs, 'BELIEF_BOUND_AXIS')
        _need(key not in bound_keys, 'BELIEF_DUPLICATE_BOUND'); bound_keys.add(key)
        for k in ('lower_bound', 'upper_bound'):
            _number(row[k], k, low=-1., high=1.)
        _need(row['lower_bound'] <= row['upper_bound'], 'BELIEF_REVERSED_BOUND')
        _hash(row['evidence_sha256'], 'evidence_sha256')
    return bind(dict(schema=BELIEF_MC_SCHEMA, source_pins=source_pins(), objective=_BELIEF_OBJECTIVE,
        hypotheses=sorted(hypotheses, key=lambda h: h['hypothesis_id']), mode=mode, caps=caps,
        seed=seed, alpha=alpha, actor_payload_sha256=actor_payload_sha256, belief_sha256=belief_sha256,
        pair_bounds=sorted(pair_bounds, key=lambda b: (b['hypothesis_id'], b['left_id'], b['right_id'])),
        stop_when_certified=stop_when_certified))


def register_belief_mc_from_record(record: dict, allocations: dict, **options) -> dict:
    """Read-only adapter from FrozenBeliefRecord's exact actor and mass_used.

    Each allocation must use the actor's single seat/context and exactly its V0
    candidate geometries. Private controller identities remain solely inside the
    allocations. No outcomes or trainer_only fields enter the returned actor;
    indeed this function never returns or mutates an actor. The supplied mapping
    declares how hypotheses condition controllers, not the true opponent identity.
    """
    from farm_neat.training import v0_belief_records as BR, v0_regret_records as R
    doc = BR.validate_record(record)
    actor = doc['actor']; belief = actor['belief']
    _fields(allocations, [h['hypothesis_id'] for h in belief['hypotheses']], 'hypothesis_allocations')
    expected = [path_binding(c) for c in actor['candidates']]
    for allocation in allocations.values():
        if allocation is None:
            continue
        limits = allocation['limits']
        _need(len(limits['seats']) == 1, 'BELIEF_RECORD_SINGLE_SEAT')
        seat = str(limits['seats'][0])
        _need(limits['actor_context_sha256s'] == {seat: actor['context_sha256']}, 'BELIEF_CONTEXT_MISMATCH')
        actual = [c['paths'][seat] for c in allocation['candidates']]
        _need(sorted(map(_wire, expected)) == sorted(map(_wire, actual)), 'BELIEF_CANDIDATE_MISMATCH')
    return register_belief_mc_allocation([
        dict(hypothesis_id=h['hypothesis_id'], mass=h[belief['mass_used']], allocation=allocations[h['hypothesis_id']])
        for h in belief['hypotheses']], actor_payload_sha256=doc['actor_payload_sha256'],
        belief_sha256=R.canonical_sha256(belief), **options)


class BeliefMonteCarloAllocator:
    """Deterministic, history-replayable composition of the existing MC allocator.

    Per-hypothesis pair differences use whole preregistered families, not cells or
    replay attempts, as sampling units. For support [a,b], n allocated families
    and k verified differences with sum S, intervals are
      [ (S+(n-k)*a)/n-r, (S+(n-k)*b)/n+r ] intersect [a,b],
      r=(b-a)*sqrt(log(2*M*sum_h G_h/alpha)/(2*n)).
    Union over every pair/hypothesis/possible prefix look makes adaptive scheduling
    and optional stopping safe under independent common-mean bounded families
    (or uniform sampling without replacement from each frozen finite panel).
    Missing outcomes retain worst cases; independent n counts only verified data.
    Cross-hypothesis dependence is allowed: mixture bounds are weighted sums, not
    a falsely independent pooled SEM. Mass/model uncertainty is NOT estimated.

    Allocation scores are heuristics, never confidence certificates:
      mass/uniform: D'Hondt priorities per charged start slot;
      uncertainty: estimated one-step reduction in weighted mean variance per
        reserved start, with a decaying variance floor (zero samples != certainty);
      decision_impact: optimistic sign-flip value from resolving one conditional
        interval with other contributions at their diagnostic midpoints, divided
        by (allocated_n+1)*cost; fall back to uncertainty if no interval is pivotal.
    Thus this is a bounded impact/VOI PROXY, not a Bayesian optimal VOI proof.
    Structural zero-effect strata can get no samples. Defaults assume no such
    structure. If all priorities vanish and fixed spend was requested, mass is a
    cost-filling fallback; early-stop mode avoids this unnecessary work.

    best_guess_id ranks baseline-relative means, using support/missing midpoints
    ONLY as an explicitly imputed diagnostic. certified_winner_id instead requires
    strict simultaneous lower bounds against EVERY candidate. Neither admits a
    teacher, changes V0 likelihoods, nor replaces disjoint confirmation WDL gates.
    """
    def __init__(self, registration: dict, *, history=()):
        fields = ('schema', 'source_pins', 'objective', 'hypotheses', 'mode', 'caps', 'seed', 'alpha',
                  'actor_payload_sha256', 'belief_sha256', 'pair_bounds', 'stop_when_certified', 'content_sha256')
        _fields(registration, fields, 'belief_registration'); _bound(registration, 'BELIEF_REGISTRATION_HASH')
        kwargs = {k:v for k,v in registration.items() if k not in ('schema','content_sha256','source_pins','objective')}
        _need(registration == register_belief_mc_allocation(**kwargs), 'BELIEF_REGISTRATION_MISMATCH')
        self._reg = copy.deepcopy(registration)
        self._children = {h['hypothesis_id']: MonteCarloAllocator(h['allocation'])
                          for h in self._reg['hypotheses'] if h['allocation'] is not None}
        self._mass = {h['hypothesis_id']: h['mass'] for h in self._reg['hypotheses']}
        first = next(iter(self._children.values()))
        self._ids = list(first._ids)
        self._pairs = [(a,b) for i,a in enumerate(self._ids) for b in self._ids[i+1:]]
        self._support = {(h,a,b): (-1.,1.) for h in self._mass for a,b in self._pairs}
        for row in self._reg['pair_bounds']:
            self._support[row['hypothesis_id'], row['left_id'], row['right_id']] = (row['lower_bound'],row['upper_bound'])
        self._values = {h: [] for h in self._mass}
        self._history = []
        self._totals = dict.fromkeys(first._totals, 0)
        self._log = math.log(2*len(self._pairs)*sum(len(c._families) for c in self._children.values())/self._reg['alpha'])
        _need(type(history) in (list,tuple), 'BELIEF_HISTORY_TYPE')
        for event in history:
            self.observe(event)

    def _conditional(self, h, a, b):
        low, high = self._support[h,a,b]
        rows = self._values[h]
        # observe rejects material support violations before mutation. Clamp only
        # its <=1e-12 arithmetic tolerance (e.g. 1 - 2/3 vs 1/3); otherwise a
        # degenerate, valid support can produce an inverted floating interval.
        values = [max(low, min(high, r[a]-r[b])) for r in rows
                  if r[a] is not None and r[b] is not None]
        n, k = len(rows), len(values)
        total = math.fsum(values)
        radius = (high-low)*math.sqrt(self._log/(2*n)) if n else None
        lo = max(low, (total+(n-k)*low)/n-radius) if n else low
        hi = min(high, (total+(n-k)*high)/n+radius) if n else high
        mean = total/k if k else None
        variance = math.fsum((x-mean)**2 for x in values)/(k-1) if k > 1 else None
        return dict(left_id=a, right_id=b, allocated_family_n=n, verified_family_n=k,
            censored_family_n=n-k, available_case_mean=mean, sample_variance=variance,
            available_case_sem=math.sqrt(variance/k) if k > 1 else None,
            point_delta=(total+(n-k)*(low+high)/2)/n if n else (low+high)/2,
            lower_bound=lo, upper_bound=hi, radius=radius, support=[low,high])

    def _statistics(self):
        per_h = {h: [self._conditional(h,a,b) for a,b in self._pairs] for h in self._mass}
        mixture = []
        for i,(a,b) in enumerate(self._pairs):
            mixture.append(dict(left_id=a, right_id=b,
                **{key: math.fsum(self._mass[h]*per_h[h][i][key] for h in self._mass)
                   for key in ('point_delta', 'lower_bound', 'upper_bound')}))
        bounds = {}
        for r in mixture:
            bounds[r['left_id'],r['right_id']] = r['lower_bound']
            bounds[r['right_id'],r['left_id']] = -r['upper_bound']
        winner = next((a for a in self._ids if all(bounds[a,b] > 0 for b in self._ids if b != a)), None)
        scores = {self._ids[0]: 0.}
        scores.update({r['right_id']: -r['point_delta'] for r in mixture if r['left_id'] == self._ids[0]})
        guess = max(self._ids, key=lambda a: scores[a])  # stable frozen-candidate tie break
        return per_h, mixture, winner, guess

    def _request(self, stats):
        per_h, mixture, winner, _ = stats
        caps = self._reg['caps']
        if self._reg['stop_when_certified'] and winner is not None:
            return None, 'SELECTION_CERTIFIED'
        if any(self._totals[k] >= caps[k] for k in ('cpu_seconds', 'elapsed_seconds')):
            return None, 'TIME_CAP'
        possible = []
        for h, child in self._children.items():
            if self._mass[h] == 0:
                continue
            req = child.next_request()
            if req is None or self._totals['simulator_starts']+req['reserved_start_slots'] > caps['native_starts']:
                continue
            cost = req['reserved_start_slots']; mass = self._mass[h]
            charged = child._totals['simulator_starts']
            uncertainty, impact = 0., 0.
            for row, mixed in zip(per_h[h], mixture):
                n, k = row['allocated_family_n'], row['verified_family_n']
                width = row['support'][1]-row['support'][0]
                variance = max(row['sample_variance'] or 0., width*width/(12*(k+1)))
                uncertainty = max(uncertainty, mass*mass*variance/((n+1)*(n+2)*cost))
                other = mixed['point_delta']-mass*row['point_delta']
                flip = min(max(0., other+mass*row['upper_bound']), max(0., -other-mass*row['lower_bound']))
                impact = max(impact, flip/((n+1)*cost))
            mode = self._reg['mode']
            priority = (mass if mode == 'mass' else 1.)/(charged+cost)
            if mode in ('uncertainty','decision_impact'):
                priority = uncertainty
            possible.append(dict(hypothesis_id=h, priority=priority, impact=impact, request=req))
        if not possible:
            return None, 'NO_AFFORDABLE_UNVISITED_FAMILY'
        if self._reg['mode'] == 'decision_impact' and any(x['impact'] > 0 for x in possible):
            for x in possible:
                x['priority'] = x['impact']
        if all(x['priority'] == 0 for x in possible):
            for x in possible:
                h = x['hypothesis_id']
                x['priority'] = self._mass[h]/(self._children[h]._totals['simulator_starts']+x['request']['reserved_start_slots'])
        chosen = min(possible, key=lambda x: (-x['priority'], digest([self._reg['seed'],x['hypothesis_id']])))
        return bind(dict(schema='farm-v0-belief-mc-request-v1', registration_sha256=self._reg['content_sha256'],
            sequence=len(self._history), previous_event_sha256=self._history[-1]['content_sha256'] if self._history else self._reg['content_sha256'],
            hypothesis_id=chosen['hypothesis_id'], request=chosen['request'],
            priorities=[{k:v for k,v in x.items() if k != 'request'} for x in possible])), None

    def next_request(self) -> dict | None:
        return self._request(self._statistics())[0]

    def observe(self, event: dict) -> None:
        """Validate all costs and support before mutation; consume no family twice.

        Persist rejected producer receipts WITH costs outside this object; rejection
        is not permission to rerun physical work. For raw estimator results use
        belief_mc_allocation_event, which reuses mc_allocation_event's receipt join.
        """
        req = self.next_request()
        _need(req is not None, 'BELIEF_ALLOCATION_STOPPED')
        _fields(event, ('schema','request_sha256','event','content_sha256'), 'belief_event')
        _bound(event, 'BELIEF_EVENT_HASH')
        _need(event['schema'] == BELIEF_MC_EVENT_SCHEMA and event['request_sha256'] == req['content_sha256'],
              'BELIEF_REQUEST_MISMATCH')
        child_event = event['event']; h = req['hypothesis_id']; child = self._children[h]
        _fields(child_event, ('schema','request_sha256','phase','cells','cpu_seconds','elapsed_seconds',
                             'producer_receipt_sha256','content_sha256'), 'mc_event')
        _bound(child_event, 'MC_EVENT_HASH')
        _need(type(child_event['cells']) is list and len(child_event['cells']) == len(req['request']['cells']), 'MC_MISSING_CELLS')
        by_id = {c: [] for c in self._ids}
        for row, key in zip(child_event['cells'], req['request']['cells']):
            _fields(row, _MC_CELL_FIELDS, 'mc_cell')
            _need({k:row[k] for k in _MC_KEY_FIELDS} == key, 'MC_UNREQUESTED_CELL')
            value = None
            if row['status'] == 'COMPLETE':
                _need(type(row['challenger_wdl']) is int and row['challenger_wdl'] in (-1,0,1), 'MC_NON_WDL')
                value = float(row['challenger_wdl'] == 1)
            by_id[row['candidate_id']].append(value)
        sample = {c: math.fsum(xs)/len(xs) if all(x is not None for x in xs) else None for c,xs in by_id.items()}
        for a,b in self._pairs:
            if sample[a] is not None and sample[b] is not None:
                lo,hi = self._support[h,a,b]
                _need(lo-1e-12 <= sample[a]-sample[b] <= hi+1e-12, 'BELIEF_BOUND_VIOLATION')
        before = dict(child._totals)
        child.observe(child_event)  # existing validate-before-mutation transaction
        for key in self._totals:
            self._totals[key] += child._totals[key]-before[key]
        self._values[h].append(sample)
        self._history.append(copy.deepcopy(event))

    def receipt(self) -> dict:
        per_h, mixture, winner, guess = stats = self._statistics()
        req, stop = self._request(stats)
        return bind(dict(schema=BELIEF_MC_RESULT_SCHEMA, objective=_BELIEF_OBJECTIVE,
            status='RUNNING' if stop is None else 'STOPPED', stop_reason=stop,
            registration_sha256=self._reg['content_sha256'], history_sha256=digest(self._history),
            actor_payload_sha256=self._reg['actor_payload_sha256'], belief_sha256=self._reg['belief_sha256'],
            mode=self._reg['mode'], best_guess_id=guess, certified_winner_id=winner,
            best_guess_scope='DIAGNOSTIC_BASELINE_RELATIVE; MISSING_SUPPORT_MIDPOINT_IMPUTATION',
            comparisons=mixture, per_hypothesis={h: dict(mass=self._mass[h], executable=h in self._children,
                allocated_family_n=len(self._values[h]), comparisons=per_h[h]) for h in self._mass},
            unavailable_mass=math.fsum(self._mass[h] for h in self._mass if h not in self._children),
            totals=self._totals, unused_start_slots=self._reg['caps']['native_starts']-self._totals['simulator_starts'],
            next_request=req, actor_training_admissible=False, teacher_weight=0., confirmation_required=True,
            limitations=list(_MC_LIMITATIONS)+[
                'CONDITIONAL_ON_FIXED_MASSES_MODELS_AND_TRUTHFUL_SUPPORT; NO_CALIBRATION_OR_MISSPECIFICATION_GUARANTEE',
                'WIN_PROBABILITY_DIAGNOSTIC_IS_NOT_THE_EXISTING_TERMINAL_WDL_ADMISSION_OBJECTIVE',
                'VARIANCE_IS_PREDICTIVE_SPREAD; SEM_AND_BOUNDS_MEASURE_SAMPLING_ERROR',
                'NO_POOLED_INDEPENDENT_N_ACROSS_HYPOTHESES; NO_REPLAY_OR_CACHE_PRECISION',
                'IMPACT_IS_A_BUDGETING_PROXY_NOT_OPTIMAL_BAYESIAN_VALUE_OF_INFORMATION']))


def belief_mc_allocation_event(request: dict, results: list[dict], *, registration: dict) -> dict:
    """Wrap the existing raw MC-estimator adapter, not a caller-supplied score.

    observe still authenticates the complete outer request against current history.
    Reused/censored cells retain the existing MC semantics and incur no extra n.
    """
    _bound(request, 'BELIEF_REQUEST_HASH')
    child = next((r['allocation'] for r in registration['hypotheses']
                  if r['hypothesis_id'] == request['hypothesis_id']), None)
    _need(child is not None, 'BELIEF_UNEXECUTABLE_HYPOTHESIS')
    return bind(dict(schema=BELIEF_MC_EVENT_SCHEMA, request_sha256=request['content_sha256'],
        event=mc_allocation_event(request['request'], results, registration=child)))


def compare_belief_mc_allocations(runs: list[dict]) -> dict:
    """Recompute all four histories; distinguish matching caps from matching spend.

    No fixture oracle/error label is accepted here. A controlled study computes
    selection error externally against its frozen generating laws, never feeds
    those laws into the allocator, and retains all seeds/losers/uncertainty.
    """
    _need(type(runs) is list and len(runs) == 4, 'BELIEF_FOUR_METHODS_REQUIRED')
    results, controls = [], []
    for run in runs:
        _fields(run, ('registration','history'), 'belief_run')
        results.append(BeliefMonteCarloAllocator(run['registration'], history=run['history']).receipt())
        controls.append({k:v for k,v in run['registration'].items() if k not in ('mode','content_sha256')})
    _need({r['mode'] for r in results} == set(BELIEF_MC_MODES), 'BELIEF_FOUR_METHODS_REQUIRED')
    _need(all(c == controls[0] for c in controls), 'BELIEF_COMPARISON_MISMATCH')
    _need(all(r['status'] == 'STOPPED' for r in results), 'BELIEF_COMPARISON_UNFINISHED')
    starts = [r['totals']['simulator_starts'] for r in results]
    physical = [r['totals']['physical_starts'] for r in results]
    equal = len(set(starts)) == len(set(physical)) == 1
    native = next(h['allocation'] for h in controls[0]['hypotheses'] if h['allocation'])['limits']['evidence_kind'] == 'NATIVE_GAME'
    return bind(dict(schema='farm-v0-belief-mc-comparison-v1',
        status=('MATCHED_PHYSICAL_STARTS' if native else 'MATCHED_START_SLOTS_FIXTURE') if equal else 'MATCHED_CAPS_UNEQUAL_SPEND',
        equal_spend=equal, cap=controls[0]['caps']['native_starts'], simulator_starts=starts,
        physical_starts=physical, results=results, actor_training_admissible=False, teacher_weight=0.))


def _load(path):
    def pairs(items):
        out = {}
        for k, v in items:
            _need(k not in out, 'DUPLICATE_JSON_KEY', k)
            out[k] = v
        return out
    result = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs)
    _safe(result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('schema', 'pilot-table', 'summarize', 'compare', 'register', 'allocate', 'compare-allocations', 'allocate-beliefs', 'compare-beliefs'):
        sub = commands.add_parser(name)
        sub.add_argument('--output', type=Path)
        if name == 'summarize':
            sub.add_argument('--history', required=True, type=Path)
            sub.add_argument('--limits', required=True, type=Path)
        elif name in ('compare', 'compare-allocations'):
            sub.add_argument('--control', required=True, type=Path)
            sub.add_argument('--candidate', required=True, type=Path)
        elif name == 'compare-beliefs':
            sub.add_argument('--runs', required=True, type=Path)
        elif name == 'register':
            sub.add_argument('--input', required=True, type=Path)
        elif name in ('allocate', 'allocate-beliefs'):
            sub.add_argument('--run', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'summarize':
            result = summarize_budget(_load(args.history), limits=_load(args.limits))
        elif args.command == 'compare':
            result = compare_budgets(_load(args.control), _load(args.candidate))
        elif args.command == 'register':
            result = register_budget(_load(args.input))
        elif args.command == 'allocate-beliefs':
            run = _load(args.run)
            _fields(run, ('registration', 'history'), 'belief_run')
            result = BeliefMonteCarloAllocator(run['registration'], history=run['history']).receipt()
        elif args.command == 'compare-beliefs':
            result = compare_belief_mc_allocations(_load(args.runs))
        elif args.command == 'allocate':
            run = _load(args.run)
            _fields(run, ('registration', 'history'), 'mc_run')
            result = MonteCarloAllocator(run['registration'], history=run['history']).receipt()
        elif args.command == 'compare-allocations':
            result = compare_mc_allocations(_load(args.control), _load(args.candidate))
        else:
            result = schema() if args.command == 'schema' else pilot_budget_table()
        text = _wire(result) + '\n'
        if args.output:
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(text)
        else:
            print(text, end='')
        return 2 if result.get('status') == 'REFUSED' else 0
    except (BudgetError, ValueError, OSError, TypeError, KeyError) as exc:
        print(json.dumps(dict(status='REFUSED', reason_code=getattr(exc, 'code', type(exc).__name__))))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
