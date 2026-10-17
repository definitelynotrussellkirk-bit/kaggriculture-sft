"""Private ex-ante decision credit over existing V0 pairs and belief tensors.

``make_credit(pair, tensor, expected_actor_sha256=..., admission_protocol=None)``
returns a closed, source-bound DecisionCreditRecord document. ``DecisionCreditRecord``
freezes its serialized bytes; ``attach_credit`` adds it ONLY to trainer_only on a
NEW FrozenBeliefRecord. No actor, loss, optimizer, compiler or admission is changed.

The reference is the SAVED actor's mass_used and conditional LOSS/DRAW/WIN laws:
E[WDL(challenger)-WDL(learner)] with WDL=(-1,0,+1). Neither one realized result nor
an empirical tensor average overwrites that reference. Win and draw probability
deltas are separate. Variance/quantiles/tails are diagnostics, never risk penalties.
Marginal action laws do NOT identify a paired-delta law. Only matched tensor cells
supply an empirical paired law; it is not silently treated as an ex-ante forecast.

Outcome evidence reuses v0_belief_outcomes validation, deduplication and complete
source-family coverage. Every declared family/group for BOTH actions and ALL live
hypotheses is required. A missing/censored/conflicting/unreplayed cell makes the
whole tensor comparison unavailable: do not select a favorable complete subset.
Families have equal weight, groups equal weight within family, hypotheses their
frozen mass. Kish effective n counts families, not H cells/replays/aliases. The
family-mean standard error is descriptive and assumes declared independent families;
no IID independence, model calibration or registration chronology is certified.

Positive conditional gains stay OPPONENT_CONDITIONED unless BOTH the frozen full
belief mixture and the full matched tensor mixture remain positive. That broader
tag describes exactly the declared finite distribution, NOT a transfer certificate.
Teacher scope ALWAYS remains the original opponent-conditioned admission scope.
Entropy/concentration/UNKNOWN mass are uncalibrated belief-confidence diagnostics,
not precision estimates or probabilities that the belief is correct.

No supplied admission flag or scalar is trusted. When a protocol is supplied, the
existing fixed-N admit_pair gate is recomputed from the original single-seat pair.
Only positive ex-ante credit with complete/replayed evidence, a bound opponent and
non-held-out tensor sources can forward its unchanged weight. No new scalar weight
formula is adopted. Multi-seat admission requires its owning tuple consumer: this
single-context v1 does not apportion a tuple's statistical support to one seat.
Synthetic fixtures may exercise positive weights conditionally; they are labelled
and are NOT production training or native playing-strength evidence.

Wire v1 embeds original inputs as an ORDER-PRESERVING JSON string, like the outcome
tensor. Read validation rebuilds every derived field; resigning edited credit is
insufficient. Save the actor digest before outcomes; a hash cannot prove chronology.
Source pins identify local bytes, not honesty of external data or attestations.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

from farm_neat.training import v0_belief_records as B
from farm_neat.training import v0_belief_outcomes as O
from farm_neat.training import v0_regret_admission as A
from farm_neat.training import v0_regret_records as R

SCHEMA = 'farm-v0-belief-decision-credit-v1'
SCOPE = 'TRAINER_EVALUATOR_ONLY; NO_ACTOR_EXPORT_OR_AUTOMATIC_LOSS'
VALUES = (-1, 0, 1)
QUANTILES = (.05, .25, .5, .75, .95)
TAIL_MASS = .1
MAX_INPUT_BYTES = 128 * 1024 * 1024
RecordError = R.RecordError


def _need(ok: bool, code: str) -> None:
    R._require(ok, code, '$.decision_credit')


def _seal(value: dict) -> dict:
    body = {k: v for k, v in value.items() if k != 'content_sha256'}
    return {**body, 'content_sha256': R.canonical_sha256(body)}


def source_pins() -> dict[str, str]:
    paths = ('src/farm_neat/training/v0_belief_credit.py',
             'src/farm_neat/training/v0_regret_admission.py')
    return O.source_pins() | {p: hashlib.sha256((R.ROOT / p).read_bytes()).hexdigest() for p in paths}


def _distribution(law: dict[int, float]) -> dict:
    """Finite predictive/empirical law diagnostics; no scalar risk objective."""
    mean = math.fsum(x * p for x, p in law.items())
    variance = math.fsum(p * (x - mean)**2 for x, p in law.items())
    support = sorted((x, p) for x, p in law.items() if p > 0)
    quantiles = {}
    for q in QUANTILES:
        cumulative = 0.
        for x, p in support:
            cumulative = math.fsum((cumulative, p))
            if cumulative >= q:
                quantiles[str(q)] = x
                break
        else:
            quantiles[str(q)] = support[-1][0]
    remaining, tail = TAIL_MASS, 0.
    for x, p in support:
        take = min(remaining, p)
        tail += take * x
        remaining -= take
        if remaining <= 0:
            break
    return dict(mean=mean, variance=variance, quantiles=quantiles,
                lower_tail_mass=TAIL_MASS, lower_tail_mean=tail / TAIL_MASS)


def _forecasts(actor: dict, left: str, right: str) -> tuple[dict, dict]:
    belief = actor['belief']
    rows = belief['hypotheses']
    weights = {h['hypothesis_id']: h[belief['mass_used']] for h in rows}
    per_h, missing = {}, []
    for h in rows:
        hid = h['hypothesis_id']
        a, b = h['conditional_wdl'][left], h['conditional_wdl'][right]
        available = a is not None and b is not None
        if not available and weights[hid] > 0:
            missing.append(hid)
        per_h[hid] = dict(mass=weights[hid], learner_wdl=copy.deepcopy(a),
            challenger_wdl=copy.deepcopy(b), expected_wdl_delta=(
                (b[2] - b[0]) - (a[2] - a[0])) if available else None,
            expected_win_probability_delta=b[2]-a[2] if available else None,
            expected_draw_probability_delta=b[1]-a[1] if available else None,
            scope='OPPONENT_CONDITIONED')
    result = dict(basis='FROZEN_ACTOR_MASS_AND_CONDITIONAL_WDL',
        mass_used=belief['mass_used'], weights=weights, per_hypothesis=per_h,
        missing_hypothesis_ids=missing, learner=None, challenger=None,
        paired_delta_variance=None, paired_delta_quantiles=None,
        paired_law_status='NOT_IDENTIFIED_BY_MARGINAL_ACTION_LAWS',
        expected_wdl_delta=None, expected_win_probability_delta=None,
        expected_draw_probability_delta=None)
    if not missing:
        for arm in ('learner', 'challenger'):
            law = [math.fsum(weights[h] * p[arm+'_wdl'][k]
                             for h, p in per_h.items() if weights[h] > 0) for k in range(3)]
            stats = _distribution(dict(zip(VALUES, law)))
            within, between = [], []
            for h, p in per_h.items():
                if weights[h] <= 0:
                    continue
                hs = _distribution(dict(zip(VALUES, p[arm+'_wdl'])))
                within.append(weights[h] * hs['variance'])
                between.append(weights[h] * (hs['mean']-stats['mean'])**2)
            result[arm] = dict(wdl=law, **stats, within_hypothesis_variance=math.fsum(within),
                               between_hypothesis_mean_variance=math.fsum(between))
        for key in ('expected_wdl_delta', 'expected_win_probability_delta', 'expected_draw_probability_delta'):
            result[key] = math.fsum(weights[h]*p[key] for h, p in per_h.items() if weights[h] > 0)
    entropy = -math.fsum(w*math.log(w) for w in weights.values() if w > 0)
    confidence = dict(mass_used=belief['mass_used'], entropy_nats=entropy,
        normalized_entropy=entropy / math.log(len(weights)) if len(weights) > 1 else 0.,
        maximum_mass=max(weights.values()), effective_hypothesis_count=math.exp(entropy),
        unknown_mass=math.fsum(weights[h['hypothesis_id']] for h in rows if h['kind'] == 'UNKNOWN'),
        floor=belief['p_min'], calibrated=False, calibration_evidence=None,
        interpretation='CONCENTRATION_NOT_CORRECTNESS_OR_EFFECTIVE_SAMPLE_SIZE')
    return result, confidence


def _outcomes(t: dict, left: str, right: str, weights: dict) -> tuple[dict, dict]:
    """t is already rebuilt by the owning validator. Never drop incomplete families."""
    index = O._index(t)
    keys = [(c, h) for c in (left, right) for h, w in weights.items() if w > 0]
    complete = O._coverage(t, index, keys)
    families = sorted({g['source_family_id'] for g in t['groups']})
    missing = []
    for g in t['groups']:
        gid = g['continuation_group_sha256']
        for c, h in keys:
            cell = index.get((c, h, gid), {})
            if not cell.get('eligible', False):
                missing.append(dict(candidate_sha256=c, hypothesis_id=h, continuation_group_sha256=gid,
                                    status=cell.get('status', 'UNAVAILABLE')))
    ready = bool(families) and not missing and set(complete) == set(families)
    out = dict(status='COMPLETE' if ready else 'UNAVAILABLE',
        declared_family_count=len(families), complete_family_count=len(complete),
        source_family_ids=families, continuation_group_count=len(t['groups']),
        missing_cells=missing, mean_wdl_delta=None, family_wdl_deltas=None,
        paired_delta_law=None, paired_delta_variance=None, paired_delta_quantiles=None,
        paired_lower_tail_mean=None, evidence_kind=(
            'RETAINED_NATIVE_RECEIPTS' if t['counts']['native_starts'] else 'SYNTHETIC_DIAGNOSTIC'),
        input_execution_counts=copy.deepcopy(t['counts']),
        comparison_coverage='ALL_DECLARED_FAMILIES_BOTH_ACTIONS_ALL_LIVE_HYPOTHESES',
        independence=t['independence'], coupling=t['coupling'])
    uncertainty = dict(effective_n=None, effective_n_basis='KISH_EQUAL_SOURCE_FAMILIES',
        certified_independent_n=None, family_mean_standard_error=None,
        standard_error_assumption='DECLARED_INDEPENDENT_SOURCE_FAMILIES_NOT_CERTIFIED',
        confidence_interval=None, forecast_error_bound=None, forecast_calibrated=False)
    if ready:
        n = len(families)
        family_values, delta_terms = {}, {x: [] for x in range(-2, 3)}
        for family, groups in complete.items():
            values = []
            for g in groups:
                gid = g['continuation_group_sha256']
                for h, w in weights.items():
                    if w <= 0:
                        continue
                    d = index[right, h, gid]['wdl'] - index[left, h, gid]['wdl']
                    values.append(w*d / len(groups))
                    delta_terms[d].append(w / n / len(groups))
            family_values[family] = math.fsum(values)
        law = {x: math.fsum(terms) for x, terms in delta_terms.items()}
        stats = _distribution(law)
        mean = math.fsum(family_values.values()) / n
        out.update(mean_wdl_delta=mean, family_wdl_deltas=family_values,
            paired_delta_law=[dict(delta=x, mass=p) for x,p in law.items()],
            paired_delta_variance=stats['variance'], paired_delta_quantiles=stats['quantiles'],
            paired_lower_tail_mean=stats['lower_tail_mean'])
        uncertainty.update(effective_n=n, family_mean_standard_error=(
            math.sqrt(math.fsum((x-mean)**2 for x in family_values.values()) / (n*(n-1))) if n > 1 else None))
    return out, uncertainty


def _candidate_key(pair: dict, arm: str) -> str:
    return R.canonical_sha256(dict(context_sha256=pair['identity']['actor_context_sha256'],
        canonical_plan=pair[arm]['canonical_plan'], continuation_signature=pair[arm]['continuation_signature']))


def make_credit(pair: dict, tensor: dict, *, expected_actor_sha256: str,
                admission_protocol: dict | None = None) -> dict:
    """Produce diagnostics for any valid pair; weight only through unchanged admission.

    expected_actor_sha256 is a separately saved pre-outcome anchor, not a digest
    taken from an untrusted tensor after seeing its labels. All inputs are copied.
    """
    R._hash(expected_actor_sha256, '$.expected_actor_sha256')
    p = R.validate_pair(pair)
    t = O.validate_tensor(tensor, expected_actor_sha256=expected_actor_sha256)
    original = R._loads(t['inputs_wire'])
    actor = original['belief_record']['actor']
    _need(R._equal(p['actor_context'], actor['context']), 'CREDIT_ACTOR_CONTEXT_MISMATCH')
    left, right = (_candidate_key(p, arm) for arm in ('learner', 'challenger'))
    _need(left in t['axes']['candidate_sha256'] and right in t['axes']['candidate_sha256'],
          'CREDIT_CANDIDATE_MISMATCH')
    R._safe(admission_protocol)
    forecasts, confidence = _forecasts(actor, left, right)
    outcome, uncertainty = _outcomes(t, left, right, forecasts['weights'])
    admission = (A.admit_pair(p, protocol=admission_protocol) if admission_protocol is not None else
        dict(status='NOT_REQUESTED', actor_training_admissible=False, teacher_weight=0.,
             reason_codes=['NO_ADMISSION_PROTOCOL']))
    expected = forecasts['expected_wdl_delta']
    complete_pair = p['evaluation']['complete'] and p['evaluation']['replay_verified']
    resolved = expected is not None and outcome['status'] == 'COMPLETE' and complete_pair
    positive_hs = sorted(h for h, f in forecasts['per_hypothesis'].items()
                         if f['expected_wdl_delta'] is not None and f['expected_wdl_delta'] > 0)
    survived = bool(resolved and expected > 0 and outcome['mean_wdl_delta'] > 0)
    bound_hs = sorted(b['hypothesis_id'] for b in original['hypothesis_bindings']
                     if b['opponent_sha256'] == p['evaluation']['opponent_hash'])
    reasons = []
    if not resolved:
        reasons.append('UNRESOLVED_COMPARISON')
    if expected is not None and expected <= 0:
        reasons.append('NONPOSITIVE_EXPECTED_WDL')
    if admission['status'] != 'ADMITTED':
        reasons.append('ADMISSION_' + admission['status'])
    if t['split'] in ('monitor', 'final'):
        reasons.append('HELDOUT_OUTCOME_SPLIT')
    if not bound_hs:
        reasons.append('UNBOUND_ADMISSION_OPPONENT')
    eligible = not reasons
    inputs = dict(pair=p, tensor=t, admission_protocol=copy.deepcopy(admission_protocol))
    wire = O.E.P.wire(inputs)  # preserve order-sensitive legacy receipt hashes
    _need(len(wire.encode('utf-8')) <= MAX_INPUT_BYTES, 'CREDIT_INPUT_BYTE_LIMIT')
    quality = ('UNRESOLVED' if expected is None else 'POSITIVE_EXPECTED_WDL' if expected > 0 else
               'NEGATIVE_EXPECTED_WDL' if expected < 0 else 'NEUTRAL_EXPECTED_WDL')
    return _seal(dict(schema=SCHEMA, scope=SCOPE, source_pins=source_pins(), inputs_wire=wire,
        actor_payload_sha256=t['actor_payload_sha256'], pair_sha256=p['content_sha256'],
        tensor_sha256=t['content_sha256'], belief_sha256=t['belief_sha256'],
        learner_candidate_sha256=left, challenger_candidate_sha256=right,
        status='RESOLVED' if resolved else 'UNRESOLVED', objective='TERMINAL_WDL',
        wdl_order=['LOSS','DRAW','WIN'], wdl_values=list(VALUES), decision_quality=quality,
        expected_wdl_delta=expected,
        expected_win_probability_delta=forecasts['expected_win_probability_delta'],
        expected_draw_probability_delta=forecasts['expected_draw_probability_delta'],
        forecast=forecasts, outcome_evidence=outcome, uncertainty=uncertainty,
        belief_confidence=confidence, realized_comparison=copy.deepcopy(p['comparison']),
        opponent_scope=dict(tag='DECLARED_BELIEF_DISTRIBUTION' if survived else 'OPPONENT_CONDITIONED',
            declared_distribution_sha256=R.canonical_sha256(forecasts['weights']),
            positive_hypothesis_ids=positive_hs, admitted_opponent_hypothesis_ids=bound_hs,
            improvement_survives_broader_distribution=survived,
            statistical_generalization_established=False, teacher_scope='OPPONENT_CONDITIONED',
            interpretation='FINITE_FROZEN_FORECAST_AND_MATCHED_TENSOR; NOT_UNIVERSAL_OR_CALIBRATED_TRANSFER'),
        admission=admission, teacher_weight=admission['teacher_weight'] if eligible else 0.,
        actor_training_admissible=eligible, weight_reason_codes=reasons,
        scalar_weighting='FORWARD_EXISTING_ADMISSION_WEIGHT_UNCHANGED; NO_VARIANCE_OR_TAIL_PENALTY',
        native_games_executed=0, limitations=[
            'FROZEN_FORECAST_NOT_PROVEN_TRUE_OR_CALIBRATED',
            'HASHES_DO_NOT_AUTHENTICATE_PRODUCER_CHRONOLOGY_OR_INDEPENDENCE',
            'SINGLE_ACTOR_CONTEXT_NO_MULTI_SEAT_SUPPORT_APPORTIONMENT',
            'NO_TRAINING_UPDATE_OR_NATIVE_STRENGTH_MEASUREMENT']))


def validate_credit(document: dict, *, expected_actor_sha256: str) -> dict:
    """Recompute from original pairs/protocol/tensor, not caller-supplied summaries."""
    R._safe(document)
    _need(type(document) is dict and document.get('schema') == SCHEMA, 'CREDIT_SCHEMA')
    _need(R._equal(document, _seal(document)), 'CREDIT_HASH_MISMATCH')
    _need(R._equal(document.get('source_pins'), source_pins()), 'CREDIT_SOURCE_MISMATCH')
    wire = document.get('inputs_wire')
    _need(type(wire) is str and len(wire.encode('utf-8')) <= MAX_INPUT_BYTES, 'CREDIT_INPUT_BYTE_LIMIT')
    inputs = R._loads(wire)
    R._fields(inputs, ('pair', 'tensor', 'admission_protocol'), '$.decision_credit.inputs')
    rebuilt = make_credit(**inputs, expected_actor_sha256=expected_actor_sha256)
    _need(R._equal(document, rebuilt), 'CREDIT_DERIVED_MISMATCH')
    return rebuilt


@dataclass(frozen=True, init=False)
class DecisionCreditRecord:
    """Immutable private serialized document, with detached data-only projections."""
    _wire: str

    def __init__(self, document: dict, *, expected_actor_sha256: str):
        checked = validate_credit(document, expected_actor_sha256=expected_actor_sha256)
        object.__setattr__(self, '_wire', B._wire(checked))

    def to_dict(self) -> dict:
        return R._loads(self._wire)

    @property
    def actor_payload_sha256(self) -> str:
        return self.to_dict()['actor_payload_sha256']


def attach_credit(record: B.FrozenBeliefRecord, credit: dict) -> B.FrozenBeliefRecord:
    """No new actor inputs. Never overwrite another decision's private credit."""
    if not isinstance(record, B.FrozenBeliefRecord):
        raise TypeError('record must be a FrozenBeliefRecord')
    private = record.to_dict()['trainer_only']
    _need('decision_credit' not in private, 'CREDIT_ALREADY_ATTACHED')
    checked = validate_credit(credit, expected_actor_sha256=record.actor_payload_sha256)
    return record.with_trainer({**private, 'decision_credit': checked})


def read_credit(path: str | Path, *, expected_actor_sha256: str) -> dict:
    return validate_credit(R._loads(Path(path).read_text(encoding='utf-8')),
                           expected_actor_sha256=expected_actor_sha256)


def write_credit(path: str | Path, document: dict, *, expected_actor_sha256: str) -> None:
    checked = validate_credit(document, expected_actor_sha256=expected_actor_sha256)
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(B._wire(checked) + '\n')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    validate = sub.add_parser('validate')
    validate.add_argument('--input', required=True)
    validate.add_argument('--actor-sha256', required=True)
    create = sub.add_parser('make')
    create.add_argument('--pair', required=True)
    create.add_argument('--tensor', required=True)
    create.add_argument('--admission-protocol')
    create.add_argument('--actor-sha256', required=True)
    create.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'validate':
            out = read_credit(args.input, expected_actor_sha256=args.actor_sha256)
        else:
            load = lambda p: R._loads(Path(p).read_text(encoding='utf-8'))
            out = make_credit(load(args.pair), load(args.tensor), expected_actor_sha256=args.actor_sha256,
                              admission_protocol=load(args.admission_protocol) if args.admission_protocol else None)
            write_credit(args.output, out, expected_actor_sha256=args.actor_sha256)
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, OverflowError) as exc:
        print(f'decision-credit refused: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(dict(schema=SCHEMA+'-validation', status='PASS', credit_status=out['status'],
                         content_sha256=out['content_sha256'], native_games_executed=0), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
