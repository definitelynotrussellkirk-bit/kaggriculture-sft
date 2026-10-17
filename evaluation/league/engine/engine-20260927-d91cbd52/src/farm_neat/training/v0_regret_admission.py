"""Fixed-N terminal-WDL teaching gate; no games, gradients, or evidence upgrades.

``admit_pair(pair, *, protocol)`` consumes an existing v1 seat record or a
``farm-v0-regret-confirmation-bundle-v1`` {test_id, records} envelope. The latter
only groups existing records; it does not change their closed wire schema.
A frozen ``farm-v0-regret-admission-protocol-v1`` enumerates the ENTIRE family of
M tests, per-seat path/context/selection locks, exact discovery/confirmation
world descriptors, source-family membership and fixed independent-family N.
The plain design JSON is not a completed experimental registration.

Records are validated by v0_regret_records, including exact own-prefix support.
Seats are averaged within physical worlds, then dependent worlds within their
PREDECLARED source family. Each independent family has equal weight. Exact
record replays are idempotent; conflicting replays and reused attempt IDs refuse.
Only a complete, replay-attested fixed sample may obtain an LCB/teacher weight.
The gate's receipt is trainer-only and MUST NOT replace a single-seat record's
comparison (which describes a different statistic). It never changes an input.

Hashes and true attestations cannot prove world independence, external replay,
registration chronology, or invisible exposure in another file/round. The caller
must retain/authenticate the frozen family and prevent fresh alpha allocations
for repeated looks. No stateless function can establish that history. This gate
checks all observable joins and fails closed when required declarations are absent.

CLI (read-only inputs; JSON receipt to stdout):
    PYTHONPATH=src python3 -B -m farm_neat.training.v0_regret_admission admit \
        --input BUNDLE.json --protocol PROTOCOL.json
    PYTHONPATH=src python3 -B -m farm_neat.training.v0_regret_admission power-table
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from farm_neat.training import v0_regret_records as R

PROTOCOL_SCHEMA = 'farm-v0-regret-admission-protocol-v1'
BUNDLE_SCHEMA = 'farm-v0-regret-confirmation-bundle-v1'
RECEIPT_SCHEMA = 'farm-v0-regret-admission-receipt-v1'
LOCK_FIELDS = ('seat', 'pair_id', 'actor_context_sha256', 'continuation_sha256',
    'parent_checkpoint_sha256', 'source_group_id', 'direct_draw_id',
    'learner_path_sha256', 'challenger_actions_sha256', 'selection_receipt_sha256', 'opponent_hash')
PHASES = ('discovery_world_groups', 'confirmation_world_groups')
ATTESTATIONS = ('frozen_before_confirmation', 'independence_attested', 'compatible_contexts_attested')
LIMITATIONS = (
    'CONDITIONAL_ON_AUTHENTIC_REPLAY_AND_PREDECLARATION_ATTESTATIONS',
    'INDEPENDENCE_AND_NO_UNDISCLOSED_EXPOSURE_NOT_PROVED_BY_HASHES',
    'NO_GAME_GRADIENT_OR_POLICY_STRENGTH_MEASUREMENT',
    'FIXED_PLAN_SEAT_TUPLE_EXPECTATION_NOT_PER_HIDDEN_SEED_LABEL',
)


class _Refusal(Exception):
    def __init__(self, code: str, status: str = 'INVALID'):
        self.code, self.status = code, status
        super().__init__(code)


def _need(ok: bool, code: str, status: str = 'INVALID') -> None:
    if not ok:
        raise _Refusal(code, status)


def _seal(value: dict) -> dict:
    value['content_sha256'] = R.canonical_sha256(value)
    return value


def _protocol(protocol: dict) -> dict[str, dict]:
    """Validate every registered opportunity, not only the submitted winner."""
    R._fields(protocol, ('schema', 'design_sha256', 'family_id', 'round_id', 'family_alpha',
        'tested_pairs', 'alpha_pair', 'min_effect', *ATTESTATIONS, 'tests', 'content_sha256'), '$.protocol')
    R._schema(protocol['schema'], PROTOCOL_SCHEMA, '$.protocol.schema')
    R._hash_join(protocol['content_sha256'], R.canonical_sha256(
        {k: v for k, v in protocol.items() if k != 'content_sha256'}), 'PROTOCOL_HASH_MISMATCH', '$.protocol')
    R._hash_join(protocol['design_sha256'], R.source_pins()[R.DESIGN_PATH], 'SOURCE_MISMATCH', '$.protocol.design')
    for key in ('family_id', 'round_id'):
        R._text(protocol[key], '$.protocol.' + key)
    family = R._number(protocol['family_alpha'], '$.protocol.family_alpha', 0, 0.05)
    count = R._int(protocol['tested_pairs'], '$.protocol.tested_pairs', 1)
    tests = R._list(protocol['tests'], '$.protocol.tests')
    _need(len(tests) == count, 'FAMILY_SIZE_MISMATCH')
    alpha = R._number(protocol['alpha_pair'], '$.protocol.alpha_pair', 0, 1)
    _need(alpha > 0 and alpha == family / count, 'ALPHA_ALLOCATION_MISMATCH')
    R._number(protocol['min_effect'], '$.protocol.min_effect', 0, 2)
    for key in ATTESTATIONS:
        R._bool(protocol[key], '$.protocol.' + key)
        _need(protocol[key], key.upper() + '_REQUIRED', 'INDETERMINATE')
    found, draws, pair_ids = {}, set(), set()
    exposure, world_family, group_identity = {}, {}, {}
    for test in tests:
        loc = '$.protocol.tests[]'
        R._fields(test, ('test_id', 'n_world_groups', 'seats', *PHASES, 'source_families', 'seat_locks'), loc)
        test_id = R._text(test['test_id'], loc + '.test_id')
        _need(test_id not in found, 'DUPLICATE_TEST')
        found[test_id] = test
        n = R._int(test['n_world_groups'], loc + '.n_world_groups', 1)
        seats = R._list(test['seats'], loc + '.seats')
        for seat in seats:
            R._int(seat, loc + '.seats[]', 0, 1)
        _need(seats in ([0], [1], [0, 1]), 'INVALID_SEATS')
        families = test['source_families']
        R._require(type(families) is dict, 'INVALID_OBJECT', loc + '.source_families')
        for key, value in families.items():
            R._text(key, loc + '.source_families.key')
            R._text(value, loc + '.source_families.value')
        local_groups, local_worlds, confirm_families = set(), set(), set()
        for phase in PHASES:
            worlds = R._list(test[phase], loc + '.' + phase)
            _need(bool(worlds), 'EMPTY_REGISTERED_WORLD_SET', 'INDETERMINATE')
            for world in worlds:
                R._fields(world, ('source_group_id', 'world_sha256', 'snapshot_sha256', 'rng_sha256', 'seed'), loc + '.world')
                group = R._text(world['source_group_id'], loc + '.world.source_group_id')
                for key in ('world_sha256', 'snapshot_sha256', 'rng_sha256'):
                    R._hash(world[key], loc + '.world.' + key)
                R._int(world['seed'], loc + '.world.seed')
                _need(group in families, 'MISSING_SOURCE_FAMILY')
                family_id, wid = families[group], world['world_sha256']
                # All tests share this exposure join. Testing order cannot hide a leak.
                for key in (('group', group), ('world', wid), ('family', family_id)):
                    _need(key not in exposure or exposure[key] == phase, 'SELECTION_CONFIRMATION_OVERLAP')
                    exposure[key] = phase
                _need(wid not in world_family or world_family[wid] == family_id, 'WORLD_FAMILY_MISMATCH')
                world_family[wid] = family_id
                identity = (R.canonical_sha256(world), family_id)
                _need(group not in group_identity or group_identity[group] == identity, 'WORLD_IDENTITY_MISMATCH')
                group_identity[group] = identity
                _need(group not in local_groups and wid not in local_worlds, 'DUPLICATE_REGISTERED_WORLD')
                local_groups.add(group); local_worlds.add(wid)
                if phase == 'confirmation_world_groups':
                    confirm_families.add(family_id)
        _need(set(families) == local_groups, 'SOURCE_FAMILY_COVERAGE_MISMATCH')
        _need(n == len(confirm_families), 'FIXED_SAMPLE_COUNT_MISMATCH')
        locks = R._list(test['seat_locks'], loc + '.seat_locks')
        _need(len(locks) == len(seats), 'SEAT_LOCK_MISMATCH')
        for seat, lock in zip(seats, locks):
            R._fields(lock, LOCK_FIELDS, loc + '.seat_locks[]')
            R._int(lock['seat'], loc + '.seat', 0, 1)
            _need(lock['seat'] == seat, 'SEAT_LOCK_MISMATCH')
            for key in LOCK_FIELDS:
                if key.endswith('sha256') or key == 'opponent_hash':
                    R._hash(lock[key], loc + '.lock.' + key)
                elif key != 'seat':
                    R._text(lock[key], loc + '.lock.' + key)
            _need(lock['pair_id'] not in pair_ids, 'PAIR_ID_REUSED')
            pair_ids.add(lock['pair_id'])
            draw = (lock['parent_checkpoint_sha256'], protocol['round_id'], lock['direct_draw_id'])
            _need(draw not in draws, 'DRAW_REUSED')
            draws.add(draw)
    return found


def _lock(record: dict) -> dict:
    i = record['identity']
    return dict(seat=record['actor_context']['observation']['player'], pair_id=i['pair_id'],
        actor_context_sha256=i['actor_context_sha256'], continuation_sha256=i['continuation_sha256'],
        parent_checkpoint_sha256=i['parent_checkpoint_sha256'], source_group_id=i['source_group_id'],
        direct_draw_id=record['learner']['direct_draw_id'], learner_path_sha256=record['learner']['path_sha256'],
        challenger_actions_sha256=R.canonical_sha256(record['challenger']['raw_actions']),
        selection_receipt_sha256=record['challenger']['selection_receipt_sha256'],
        opponent_hash=record['evaluation']['opponent_hash'])


def _records(pair: dict, tests: dict, out: dict) -> tuple[list[dict], dict]:
    R._require(type(pair) is dict, 'INVALID_OBJECT', '$.pair')
    if pair.get('schema') == R.SCHEMA:
        records = [pair]
        R._require(type(pair.get('identity')) is dict, 'INVALID_OBJECT', '$.pair.identity')
        matches = [t for t in tests.values() if any(
            lock['pair_id'] == pair['identity'].get('pair_id') for lock in t['seat_locks'])]
        _need(len(matches) == 1, 'UNREGISTERED_PAIR')
        test = matches[0]
    else:
        R._fields(pair, ('schema', 'test_id', 'records'), '$.bundle')
        R._schema(pair['schema'], BUNDLE_SCHEMA, '$.bundle.schema')
        R._text(pair['test_id'], '$.bundle.test_id')
        _need(pair['test_id'] in tests, 'UNREGISTERED_TEST')
        records = R._list(pair['records'], '$.bundle.records')
        test = tests[pair['test_id']]
    out['test_id'] = test['test_id']
    out['registered_world_count'] = test['n_world_groups']
    unique, payloads = {}, set()
    for record in records:
        digest = R.canonical_sha256(record)
        if digest in payloads:
            out['duplicate_records_ignored'] += 1
            continue
        try:
            record = R.validate_pair(record)
        except OverflowError as exc:
            # The v1 dependency accepts large JSON integers before alpha division.
            # Refuse at our boundary; do not modify another task's validator.
            raise _Refusal('NUMERIC_RECORD_OVERFLOW') from exc
        pair_id = record['identity']['pair_id']
        _need(pair_id not in unique, 'CONFLICTING_REPLAY')
        unique[pair_id] = record
        payloads.add(digest)
    out['pair_content_sha256s'] = [p['content_sha256'] for _, p in sorted(unique.items())]
    # Never allow a partial seat tuple to self-certify its favorable arm.
    required = {row['pair_id'] for row in test['seat_locks']}
    _need(set(unique) <= required, 'UNREGISTERED_PAIR')
    _need(set(unique) == required, 'MISSING_REGISTERED_SEAT', 'INDETERMINATE')
    return [unique[lock['pair_id']] for lock in test['seat_locks']], test


def _retain(records: list[dict], test: dict, out: dict) -> None:
    """Retain both phases and every sign, without copying actor-visible features."""
    discovery = {w['source_group_id'] for w in test['discovery_world_groups']}
    for record in records:
        for row in record['evaluation']['seat_rows']:
            phase = 'discovery' if row['source_group_id'] in discovery else 'confirmation'
            out['retained_outcomes'].append(dict(pair_id=record['identity']['pair_id'],
                phase=phase, **copy.deepcopy(row)))
    out['upstream_decisions'] = [dict(pair_id=p['identity']['pair_id'],
        comparison=copy.deepcopy(p['comparison'])) for p in records]


def _summarize(records: list[dict], test: dict, out: dict) -> None:
    rows = [r for r in out['retained_outcomes'] if r['phase'] == 'confirmation']
    by_cell, attempts = {}, set()
    # Includes discovery IDs, so a control cannot be relabeled as a new execution.
    for row in out['retained_outcomes']:
        for key in ('learner_attempt_id', 'challenger_attempt_id'):
            _need(row[key] not in attempts, 'ATTEMPT_ID_REUSE')
            attempts.add(row[key])
    for row in rows:
        key = (row['source_group_id'], row['seat'])
        _need(key not in by_cell, 'DUPLICATE_OUTCOME')
        by_cell[key] = row
    counts = dict(confirmation_pairs=len(rows), complete_pairs=sum(r['complete'] for r in rows),
        replay_verified_pairs=sum(r['replay_verified'] for r in rows),
        learner_wdl=dict(win=0, draw=0, loss=0), challenger_wdl=dict(win=0, draw=0, loss=0),
        paired_sign=dict(challenger_better=0, tie=0, learner_better=0),
        unique_learner_attempts=len({r['learner_attempt_id'] for r in rows}),
        unique_challenger_attempts=len({r['challenger_attempt_id'] for r in rows}))
    for row in rows:
        if row['complete']:
            for arm in ('learner', 'challenger'):
                counts[arm + '_wdl'][{1: 'win', 0: 'draw', -1: 'loss'}[row[arm + '_wdl']]] += 1
            delta = row['challenger_wdl'] - row['learner_wdl']
            counts['paired_sign']['challenger_better' if delta > 0 else 'learner_better' if delta < 0 else 'tie'] += 1
    out['counts'] = counts
    families = {}
    for world in test['confirmation_world_groups']:
        group = world['source_group_id']
        seat_rows = [by_cell.get((group, seat)) for seat in test['seats']]
        complete = all(r is not None and r['complete'] for r in seat_rows)
        delta = math.fsum(r['challenger_wdl'] - r['learner_wdl'] for r in seat_rows) / len(seat_rows) if complete else None
        family = test['source_families'][group]
        families.setdefault(family, []).append(dict(source_group_id=group, world_sha256=world['world_sha256'],
            delta=delta, complete=complete))
    values = []
    for family, worlds in sorted(families.items()):
        complete = all(w['complete'] for w in worlds)
        delta = math.fsum(w['delta'] for w in worlds) / len(worlds) if complete else None
        out['world_clusters'].append(dict(source_family_id=family, worlds=worlds, delta=delta, complete=complete))
        if complete:
            values.append(delta)
    out['independent_world_count'] = len(values)
    if values:
        out['mean_wdl_delta'] = math.fsum(values) / len(values)
        mean = out['mean_wdl_delta']
        out['realized_verdict'] = 'CHALLENGER_BETTER' if mean > 0 else 'LEARNER_BETTER' if mean < 0 else 'TIE'
    _need(len(values) == test['n_world_groups'] and all(p['evaluation']['complete'] for p in records),
          'CENSORED_OR_MISSING_OUTCOMES', 'CENSORED')
    _need(all(p['evaluation']['replay_verified'] for p in records)
          and all(r['replay_verified'] for r in out['retained_outcomes']),
          'REPLAY_NOT_VERIFIED', 'INDETERMINATE')


def admit_pair(pair: dict, *, protocol: dict) -> dict:
    """Return a sealed, fail-closed trainer receipt, including zero-weight refusals.

    No upstream admission flag is trusted as a gate. No predictions/cash estimates
    are consumed. The returned positive support concerns the registered seat-tuple
    mean, NOT each seat separately, not general policy Q, and not monotonic fitting.
    """
    out = dict(schema=RECEIPT_SCHEMA, status='INDETERMINATE', objective='TERMINAL_WDL',
        verdict='INDETERMINATE', realized_verdict='INDETERMINATE', mean_wdl_delta=None,
        lower_bound=None, alpha=None, independent_world_count=0, registered_world_count=None,
        actor_training_admissible=False, statistical_support=False, teacher_weight=0.0,
        reason_codes=[], test_id=None, protocol_sha256=None, input_sha256=None,
        pair_content_sha256s=[], duplicate_records_ignored=0, world_clusters=[],
        retained_outcomes=[], upstream_decisions=[], counts={}, limitations=list(LIMITATIONS))
    try:
        R._safe(pair); R._safe(protocol)
        out['input_sha256'] = R.canonical_sha256(pair)
        out['protocol_sha256'] = R.canonical_sha256(protocol)
        tests = _protocol(protocol)
        out['alpha'] = protocol['alpha_pair']
        records, test = _records(pair, tests, out)
        _retain(records, test, out)
        for record, lock in zip(records, test['seat_locks']):
            identity, ev = record['identity'], record['evaluation']
            _need(identity['split'] in ('confirmation', 'training'), 'NONTRAINING_SPLIT')
            _need(identity['round_id'] == protocol['round_id'], 'ROUND_MISMATCH')
            _need(_lock(record) == lock, 'FROZEN_LOCK_MISMATCH')
            _need(ev['mode'] == 'CONFIRMED_EXPECTATION',
                  'REALIZATION_ONLY' if ev['mode'] == 'EXACT_REALIZATION' else 'NO_CONFIRMED_EVIDENCE', 'INDETERMINATE')
            for phase in PHASES:
                expected = {w['source_group_id']: w for w in test[phase]}
                actual = {w['source_group_id']: w for w in ev[phase]}
                _need(actual == expected, 'REGISTERED_WORLD_MISMATCH')
            reg = ev['confirmation_registration']
            for key in ('family_alpha', 'tested_pairs', 'alpha_pair', 'min_effect', *ATTESTATIONS):
                _need(reg[key] == protocol[key], 'REGISTRATION_MISMATCH')
        _summarize(records, test, out)
        n, mean, alpha = out['independent_world_count'], out['mean_wdl_delta'], out['alpha']
        bound = mean - math.sqrt(8 * -math.log(alpha) / n)
        out['lower_bound'] = bound
        if bound > protocol['min_effect']:
            out.update(status='ADMITTED', verdict='CHALLENGER_BETTER', statistical_support=True,
                actor_training_admissible=True, teacher_weight=min(2.0, max(0.0, bound)) / 2.0)
        elif mean == 0:
            out.update(status='TIE', verdict='TIE', reason_codes=['NO_MEAN_WDL_ADVANTAGE'])
        elif mean < 0:
            out.update(status='LEARNER_BETTER', verdict='LEARNER_BETTER', reason_codes=['REALIZED_LEARNER_BETTER'])
        else:
            out['reason_codes'] = ['INSUFFICIENT_POWER']
    except R.RecordError as exc:
        out.update(status='INVALID', verdict='INVALID', reason_codes=[exc.code], invalid_path=exc.path)
    except _Refusal as exc:
        out.update(status=exc.status, verdict=exc.status if exc.status in R.VERDICTS else 'INDETERMINATE',
                   reason_codes=[exc.code])
    return _seal(out)


def power_cost_table() -> dict:
    """Analytic pilot planning, not a benchmark, executed games, or promised yield.

    Minimum observed-mean N solves the STRICT LCB inequality. The distinct 80%
    power column is a sufficient distribution-free bound conditional on an actual
    expected advantage at least the stated effect and independent bounded groups.
    Costs are fresh fixed-N registrations, NOT optional extension of observed runs.
    """
    design = json.loads((R.ROOT / R.DESIGN_PATH).read_text())
    pilot, admission = design['pilot_recipe'], design['admission']
    draws, seats = pilot['learner_draws'], len(pilot['seats'])
    n0, discovery = pilot['confirmation_world_groups_per_pair'], pilot['discovery_world_groups_per_pair']
    shortlist = design['candidate_pool']['max_discovery_challengers']
    alpha, effect_min = admission['family_alpha'] / draws, admission['min_effect']
    log_alpha, log_beta = -math.log(alpha), -math.log(0.2)
    radius = math.sqrt(8 * log_alpha / n0)
    def cost(n):
        return 2 * draws * seats * (discovery * (1 + shortlist) + 2 * n)
    rows = []
    for effect in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        gap = effect - effect_min
        required = math.floor(8 * log_alpha / gap**2) + 1 if gap > 0 else None
        n80 = math.ceil(8 * (math.sqrt(log_alpha) + math.sqrt(log_beta))**2 / gap**2) if gap > 0 else None
        power_floor = 1 - math.exp(-n0 * (gap - radius)**2 / 8) if gap > radius else 0.0
        rows.append(dict(assumed_mean_delta=effect, lcb_at_pilot_n_if_observed_mean=effect - radius,
            minimum_n_for_observed_mean=required, sufficient_n_for_80_percent_power=n80,
            distribution_free_power_lower_bound_at_pilot_n=power_floor,
            all_arm_starts_with_full_replay=cost(required) if required else None,
            all_arm_starts_for_80_percent_power_bound=cost(n80) if n80 else None))
    return _seal(dict(schema='farm-v0-regret-power-cost-v1', scope='ANALYTIC_NOT_EMPIRICAL_POWER_OR_RUNTIME',
        design_sha256=R.source_pins()[R.DESIGN_PATH],
        pilot=dict(tested_pairs=draws, seats=pilot['seats'], alpha_pair=alpha,
            confirmation_world_groups_per_pair=n0, min_effect=effect_min, hoeffding_radius=radius,
            base_execution_upper_bound=cost(n0) // 2, verification_replay_execution_upper_bound=cost(n0) // 2,
            total_simulator_execution_upper_bound=cost(n0)), rows=rows,
        limits=['No positive-yield promise; all ties always have zero teaching weight.',
            'Power lower bounds assume independent bounded families with the stated expected mean.',
            'Observed-mean thresholds are not power estimates.',
            'Costs count simulator executions including replay, not independent observations or seconds.',
            'A costed larger N requires a fresh predeclared round/error allocation, not repeated looks.']))


# This experiment is intentionally disjoint from TERMINAL_WDL admission v1.
MARGIN_ADMISSION_SCHEMA = 'farm-v0-fixed-shop-margin-v1-admission'
MARGIN_CORPUS_SCHEMA = 'farm-v0-fixed-shop-margin-v1-supervised-corpus'
MARGIN_TRAINING_SCHEMA = 'farm-v0-fixed-shop-margin-v1-supervised-target'


def margin_activation_rule(rows: list[dict]) -> dict:
    """Pure, descriptive reducer. Passing this rule ALONE grants no training weight.

    Exactly sixteen two-seat clusters; no row deletion, confidence bound, optional
    sample extension, cash tie-break in WDL, or checkpoint-promotion authority.
    Authentication, preregistration, native provenance and fresh replay are checked
    separately by admit_margin_experiment before this reducer can enable teaching.
    """
    out = dict(status='INDETERMINATE', rule_passed=False, mean_margin_delta=None,
        seat_means={}, positive_worlds=0, world_means=[], retained_pairs=copy.deepcopy(rows),
        wdl={arm: dict(win=0, draw=0, loss=0) for arm in ('old', 'new')},
        statistical_support=False, promotion_objective='TERMINAL_WDL', reason_codes=[])
    try:
        R._safe(rows)
        _need(type(rows) is list and len(rows) == 32, 'MARGIN_MISSING_PAIRS')
        cells, worlds = {}, set()
        for row in rows:
            _need(type(row) is dict, 'MARGIN_PAIR_FIELDS')
            world, seat = row['world_id'], row['seat']
            _need(type(world) is str and world and type(seat) is int and seat in (0, 1),
                  'MARGIN_WORLD_SEAT')
            key = (world, seat)
            _need(key not in cells, 'MARGIN_DUPLICATE_PAIR')
            cells[key] = row; worlds.add(world)
            _need(row['complete'] is True and row['replay_verified'] is True,
                  'MARGIN_INCOMPLETE_OR_REPLAY_FAILURE')
            _need(type(row['delta']) in (int, float) and math.isfinite(row['delta']),
                  'MARGIN_NONFINITE_DELTA')
            for arm in ('old', 'new'):
                wdl = row[arm + '_wdl']
                _need(type(wdl) is str and wdl in ('LOSS', 'DRAW', 'WIN'), 'MARGIN_WDL_LABEL')
                out['wdl'][arm][wdl.lower()] += 1
        _need(len(worlds) == 16 and set(cells) == {(w, s) for w in worlds for s in (0, 1)},
              'MARGIN_WORLD_SEAT_COVERAGE')
        out['seat_means'] = {str(s): math.fsum(cells[w, s]['delta'] / 16 for w in sorted(worlds))
                             for s in (0, 1)}
        out['world_means'] = [dict(world_id=w, mean=math.fsum(cells[w, s]['delta'] / 2 for s in (0, 1)))
                              for w in sorted(worlds)]
        out['mean_margin_delta'] = math.fsum(r['mean'] / 16 for r in out['world_means'])
        out['positive_worlds'] = sum(r['mean'] > 0 for r in out['world_means'])
        for valid, code in ((out['mean_margin_delta'] > 0, 'MARGIN_NONPOSITIVE_MEAN'),
                           (out['positive_worlds'] >= 10, 'MARGIN_FEWER_THAN_TEN_POSITIVE_WORLDS'),
                           (all(x >= 0 for x in out['seat_means'].values()), 'MARGIN_SEAT_REVERSAL')):
            if not valid: out['reason_codes'].append(code)
        out.update(status='RULE_PASSED' if not out['reason_codes'] else 'NOT_ACTIVATED',
                   rule_passed=not out['reason_codes'])
    except (_Refusal, R.RecordError, KeyError, TypeError, ValueError, OverflowError) as exc:
        out.update(status='INDETERMINATE', rule_passed=False,
                   reason_codes=[getattr(exc, 'code', 'MARGIN_MALFORMED_PAIRS')])
    return out


def admit_margin_experiment(request: dict, result: dict) -> dict:
    """Recompute all joins; synthetic or incomplete evidence always has zero weight.

    The bounded v1 teacher weight is 1 for strictly positive paired rows AFTER
    panel activation, 0 otherwise. Negative/tied rows are retained, not censored.
    Both checkpoint samples keep their authentic original likelihoods in request;
    selected new paths are explicitly off-policy supervised class-path targets.
    """
    from farm_neat.evaluation import v0_regret_pairs as E
    out = dict(schema=MARGIN_ADMISSION_SCHEMA, objective=E.MARGIN_OBJECTIVE,
        objective_version=1, status='INDETERMINATE', activated=False,
        actor_training_admissible=False, teacher_weight=0.0, statistical_support=False,
        promotion_objective='TERMINAL_WDL', strength_claim=False, request_sha256=None,
        result_sha256=None, rule=None, training_records=[], reason_codes=[],
        limitations=['PREREGISTERED_BOUNDED_EXPERIMENT_NOT_A_CONFIDENCE_THEOREM',
                     'NO_POLICY_PROMOTION_OR_ORIGINAL_RULE_WDL_IMPROVEMENT_CLAIM',
                     'MARGIN_SELECTED_PATHS_ARE_OFF_POLICY_SUPERVISED_TARGETS'])
    try:
        R._safe(request); R._safe(result)
        E.validate_margin_request(request)
        E.validate_margin_result(request, result)
        out.update(request_sha256=request['content_sha256'], result_sha256=result['content_sha256'])
        rule = margin_activation_rule(result['pairs'])
        out['rule'] = rule
        native = (result['execution'] == E.margin_execution_binding(E.margin_native_runner)
                  and request['production'] is True
                  and all(not c['identity']['diagnostic'] for c in request['checkpoints'].values()))
        complete = (result['status'] == 'VERIFIED' and
                    result['counts']['primary_physical_games'] == 64 and
                    result['counts']['replay_physical_games'] == 64)
        activated = native and complete and rule['rule_passed']
        out.update(status='ACTIVATED' if activated else 'NOT_ACTIVATED' if native and complete else 'INDETERMINATE',
                   activated=activated, actor_training_admissible=activated,
                   teacher_weight=1.0 if activated else 0.0)
        out['reason_codes'] = list(result['reason_codes']) + list(rule['reason_codes'])
        if not native: out['reason_codes'].append('MARGIN_SYNTHETIC_OR_DIAGNOSTIC_NOT_ADMISSIBLE')
        if not complete: out['reason_codes'].append('MARGIN_COMPLETE_NATIVE_REPLAY_REQUIRED')
        by_cell = {(c['world_id'], c['seat']): c for c in request['cells']}
        for pair in result['pairs']:
            cell = by_cell[(pair['world_id'], pair['seat'])]
            delta = pair['delta']
            direction = None if delta is None else 1 if delta > 0 else -1 if delta < 0 else 0
            out['training_records'].append(dict(schema=MARGIN_TRAINING_SCHEMA,
                objective=E.MARGIN_OBJECTIVE, objective_version=1,
                cell_sha256=cell['content_sha256'], world_id=pair['world_id'], seat=pair['seat'],
                signed_margin_delta=delta, signed_margin_direction=direction,
                teacher_weight=1.0 if activated and direction == 1 else 0.0,
                loss='OFF_POLICY_SUPERVISED_CLASS_PATH', target_arm='new',
                target_path_sha256=cell['new']['sample']['path_sha256'],
                learner_path_sha256=cell['old']['sample']['path_sha256'],
                target_behavior_logp=None, on_policy=False))
    except (E.PairError, R.RecordError, ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        out.update(status='INDETERMINATE', activated=False, actor_training_admissible=False,
                   teacher_weight=0.0, training_records=[],
                   reason_codes=[getattr(exc, 'code', 'MARGIN_EVIDENCE_REJECTED')])
    return _seal(out)


def margin_training_corpus(request: dict, result: dict) -> dict:
    """Explicit, separate consumer envelope; never a TERMINAL_WDL pair record."""
    return dict(schema=MARGIN_CORPUS_SCHEMA, request=copy.deepcopy(request),
                result=copy.deepcopy(result), admission=admit_margin_experiment(request, result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    admit = sub.add_parser('admit')
    admit.add_argument('--input', required=True)
    admit.add_argument('--protocol', required=True)
    sub.add_parser('power-table')
    for command in ('margin-admit', 'margin-corpus'):
        margin = sub.add_parser(command)
        margin.add_argument('--request', required=True)
        margin.add_argument('--result', required=True)
        margin.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in ('margin-admit', 'margin-corpus'):
            from farm_neat.evaluation import v0_regret_pairs as E
            request = R._loads(Path(args.request).read_text(encoding='utf-8'))
            execution = R._loads(Path(args.result).read_text(encoding='utf-8'))
            value = (admit_margin_experiment(request, execution) if args.command == 'margin-admit'
                     else margin_training_corpus(request, execution))
            E._margin_write_json(Path(args.output), value)
            admission = value if args.command == 'margin-admit' else value['admission']
            print(json.dumps(dict(output=args.output, status=admission['status'],
                activated=admission['activated'], reason_codes=admission['reason_codes']), sort_keys=True))
            return 2 if admission['status'] == 'INDETERMINATE' else 0
        result = power_cost_table() if args.command == 'power-table' else admit_pair(
            R._loads(Path(args.input).read_text(encoding='utf-8')),
            protocol=R._loads(Path(args.protocol).read_text(encoding='utf-8')))
    except R.RecordError as exc:
        result = dict(status='INVALID', reason_codes=[exc.code], invalid_path=exc.path,
                      actor_training_admissible=False, teacher_weight=0.0)
    except (OSError, UnicodeError) as exc:
        result = dict(status='INVALID', reason_codes=['INPUT_IO_ERROR'], invalid_path=type(exc).__name__,
                      actor_training_admissible=False, teacher_weight=0.0)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 2 if result.get('status') == 'INVALID' else 0


if __name__ == '__main__':
    raise SystemExit(main())
