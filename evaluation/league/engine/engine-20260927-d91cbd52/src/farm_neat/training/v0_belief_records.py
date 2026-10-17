"""Source-bound, actor-time multi-belief sidecars (no model, games or training).

``make_candidate(context, raw_actions)`` reuses native own-prefix replay.
``make_record(context, belief, candidates)`` seals a detached closed JSON record.
``validate_record`` checks it, including local source pins and both digests.
``FrozenBeliefRecord`` stores immutable JSON bytes; ``with_trainer`` returns a NEW
record, retaining exactly the original actor payload/digest. Save the original
actor digest before evaluation and pass it as ``expected_actor_sha256`` on read.
Hashes alone do not authenticate producer chronology, public provenance or truth.

The legacy actor context is NOT extended: this is a separately versioned sidecar.
The current bridge remains opening-only, labor_factor=4, daily HANDS, zero-flow,
no intraday controller. Candidate identity includes context/starting memory and
exact final Plan AND decoder continuation; raw aliases/traces remain auditable.
There are no teacher labels, private likelihoods or simulator-private fields in
the actor allowlist. Trainer-only finite JSON is isolated, never inspected by actor logic.
Public hypothetical model definitions are NOT actual opponent/controller identity.

Belief rows carry raw prior, positive initial mass, finite cumulative public log
likelihood, posterior, constrained output, and nullable per-candidate W/D/L laws
in LOSS/DRAW/WIN order. Posterior is independently checked in log space. A prior
strength and optional initial-odds cap are DECLARATIONS, not an implemented prior
training/transform algorithm. Uniform initialization has strength zero. A preloaded
producer must bind its prior provenance; raw prior and initial mass stay separate.
``lower_bound_only`` certifies feasibility of a reported constrained vector, NOT a
particular projection algorithm. ``two_hypothesis_clamp`` checks the accepted toy
transform exactly. No output floor is fed back into the posterior. No missing law
is replaced by zero or renormalized away; UNKNOWN is an optional residual expert,
not a calibrated probability of an unseen opponent family.

Distinctness: duplicate model definitions/parameters always refuse. Exact equal
complete predictive profiles on the same declared public probe panel also refuse
when the panel covers every candidate intervention. Missing coverage/conditionals
without an observed difference is UNRESOLVED, not evidence of distinctness or permission to merge recurrent state.
Different WDL means alone, slot IDs, embeddings and marginals are not certificates.
A distinct panel is only a finite-panel diagnostic, never global latent identity.

Bounds: 64 hypotheses, 32 candidates, 256 probes, 256 categories per public probe;
JSON's depth/integer limits are inherited from v0_regret_records. Probability sums
use 1e-12 absolute tolerance; posterior/transform joins use 1e-10 relative
tolerance with NO absolute epsilon. Individual floors, positive support and
K*p_min feasibility (checked using the exact input float ratio) are strict. No epsilon repair is applied.
Only unordered hypothesis/candidate/probe ROWS are canonicalized. Raw actions,
probability categories, compiler allocation/started/HOLD lists retain their order.

The accepted SPEC's detailed CONTRACT.json is omitted from this source packet.
This wire v1 implements its retained report/acceptance and calibration fixtures;
byte/schema parity with that omitted sketch is NOT claimed. It changes no legacy
record, compiler, loss, admission or scorer interface. Downstream records/head/
outcome consumers can use the explicit sidecar without passing it to the actor.

    PYTHONPATH=src python3 -B -m farm_neat.training.v0_belief_records \\
        validate --input RECORD.json --actor-sha256 SAVED_ACTOR_SHA256
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
from typing import Any

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.training import v0_regret_records as R
from farm_neat.training.teachers import v0_regret_proposals as P

SCHEMA = 'farm-v0-belief-record-v1'
ACTOR_SCHEMA = 'farm-v0-belief-actor-v1'
CANDIDATE_SCHEMA = 'farm-v0-belief-candidate-v1'
DISTINCTNESS_RULE = 'EXACT_COMPLETE_PUBLIC_PROBE_PROFILE_V1'
MAX_HYPOTHESES, MAX_CANDIDATES, MAX_PROBES = 64, 32, 256
RecordError = R.RecordError
canonical_sha256 = R.canonical_sha256
_SOURCE_PATHS = (
    'src/farm_neat/training/v0_belief_records.py',
    'src/farm_neat/strategy/opponent.py',
    'tests/fixtures/v0_belief_calibration.json',
    'work/runs/review-package/acceptance/V0-BELIEF-SPEC.json',
    'work/runs/review-package/acceptance/V0-BELIEF-CALIBRATION-FIXTURES.json',
    'work/runs/reviewer-play/v0-belief-spec/RESULT.md',
    'work/runs/reviewer-play/v0-belief-distinctness-study/RESULT.md',
    'work/runs/reviewer-play/v0-belief-unknown-reserve/RESULT.md',
)


def source_pins() -> dict[str, str]:
    """Local allowlisted files only; never open record-supplied paths or URLs."""
    return {**R.source_pins(), **{p: hashlib.sha256((R.ROOT / p).read_bytes()).hexdigest()
                                for p in _SOURCE_PATHS + P.SOURCE_PATHS}}


def _closed(value: Any, fields: tuple | list | set, path: str) -> None:
    R._fields(value, fields, path, code='ACTOR_FIELDS')


def _enum(value: Any, values: tuple, path: str, code: str) -> None:
    R._require(type(value) is str and value in values, code, path)


def _float(value: Any, path: str, low=None, high=None) -> float:
    R._number(value, path, low, high)
    value = float(value)
    return 0.0 if value == 0 else value  # one canonical signed-zero encoding


def _close(left: float, right: float) -> bool:
    # Absolute tolerances would silently accept arbitrary positive replacements
    # of small but live hypotheses (e.g. 1e-30 in place of 1e-20).
    return math.isclose(left, right, abs_tol=0., rel_tol=1e-10)


def _probabilities(value: Any, size: int, path: str, *, positive=False) -> list[float]:
    R._list(value, path)
    R._require(len(value) == size, 'PROBABILITY_ARITY', path)
    values = [_float(x, path, 0, 1) for x in value]
    R._require(not positive or all(x > 0 for x in values), 'POSITIVE_SUPPORT', path)
    R._require(math.isclose(math.fsum(values), 1., rel_tol=0., abs_tol=1e-12),
               'PROBABILITY_NORMALIZATION', path)
    return values


def _context(context: dict):
    state, prefix, options = R._actor(context)
    R._require(state.step == 0, 'UNSUPPORTED_CONTEXT', '$.actor.context.observation.step')
    return state, prefix, options


def _candidate(context, actions, state, prefix, options, menus):
    R._safe(actions)
    screen = P._syntax_screen(prefix, actions)
    R._require(screen['status'] != 'INVALID', 'INVALID_CANDIDATE_PATH', '$.candidate.raw_actions')
    try:
        trace = P._trace(state, prefix, options, menus, actions)
    except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as exc:
        raise RecordError('CANDIDATE_REPLAY_FAILED', '$.candidate') from exc
    R._require(trace['status'] == 'REACHABLE' and trace['final'].terminated,
               'UNREACHABLE_CANDIDATE', '$.candidate')
    geometry = {key: trace[key] for key in ('prefix_class_trace', 'canonical_plan', 'continuation_signature')}
    key = canonical_sha256(dict(context_sha256=canonical_sha256(context),
                                canonical_plan=geometry['canonical_plan'],
                                continuation_signature=geometry['continuation_signature']))
    return dict(schema=CANDIDATE_SCHEMA, candidate_sha256=key, raw_actions=copy.deepcopy(actions), **geometry)


def make_candidate(context: dict, raw_actions: list[int]) -> dict:
    """Build actual prefix-local geometry; no compilation or value certificate."""
    state, prefix, options = _context(context)
    return _candidate(context, raw_actions, state, prefix, options, {})


def _probes(value, context, context_key, candidate_ids):
    path = '$.actor.belief.distinctness'
    _closed(value, ('rule', 'probes'), path)
    R._require(value['rule'] == DISTINCTNESS_RULE, 'DISTINCTNESS_RULE', path)
    rows = R._list(value['probes'], path + '.probes')
    R._require(len(rows) <= MAX_PROBES, 'PROBE_LIMIT', path)
    seen, result = set(), []
    descriptor_fields = ('context_sha256', 'prefix_sha256', 'candidate_sha256', 'horizon_steps',
                         'target', 'outcome_count', 'definition_sha256')
    for row in rows:
        _closed(row, ('probe_id',) + descriptor_fields, path + '.probes')
        R._hash_join(row['context_sha256'], context_key, 'ACTOR_CONTEXT_MISMATCH', path)
        R._hash_join(row['prefix_sha256'], canonical_sha256(context['prefix']), 'PREFIX_MISMATCH', path)
        R._require(row['candidate_sha256'] is None or (type(row['candidate_sha256']) is str
                   and row['candidate_sha256'] in candidate_ids),
                   'UNKNOWN_CANDIDATE', path)
        R._int(row['horizon_steps'], path + '.horizon_steps', 1, V.LAST_STEP)
        _enum(row['target'], ('PUBLIC_EVENT', 'TERMINAL_WDL'), path, 'PREDICTION_TARGET')
        R._int(row['outcome_count'], path + '.outcome_count', 2, 256)
        if row['target'] == 'TERMINAL_WDL':
            R._require(row['outcome_count'] == 3 and row['candidate_sha256'] in candidate_ids
                       and row['horizon_steps'] == V.LAST_STEP - context['observation']['step'],
                       'PREDICTION_TARGET', path)
        R._hash(row['definition_sha256'], path + '.definition_sha256')
        R._hash_join(row['probe_id'], canonical_sha256({k: row[k] for k in descriptor_fields}),
                     'PROBE_HASH_MISMATCH', path)
        R._require(row['probe_id'] not in seen, 'DUPLICATE_PROBE', path)
        seen.add(row['probe_id']); result.append(copy.deepcopy(row))
    return sorted(result, key=lambda r: r['probe_id'])


def _belief(value, context, context_key, candidate_ids):
    path = '$.actor.belief'
    _closed(value, ('k', 'p_min', 'floor_transform', 'mass_used', 'prior', 'public_evidence',
                    'distinctness', 'hypotheses'), path)
    k = R._int(value['k'], path + '.k', 1, MAX_HYPOTHESES)
    floor = _float(value['p_min'], path + '.p_min', 0, 1)
    numerator, denominator = floor.as_integer_ratio()
    R._require(k * numerator <= denominator, 'INFEASIBLE_FLOOR', path + '.p_min')
    _enum(value['mass_used'], ('posterior_mass', 'constrained_mass'), path, 'MASS_SELECTOR')
    _enum(value['floor_transform'], ('none', 'two_hypothesis_clamp', 'lower_bound_only'), path, 'FLOOR_TRANSFORM')
    rows = R._list(value['hypotheses'], path + '.hypotheses')
    R._require(len(rows) == k, 'HYPOTHESIS_COUNT', path)
    prior = value['prior']
    _closed(prior, ('mode', 'strength', 'odds_cap', 'source_sha256'), path + '.prior')
    _enum(prior['mode'], ('uniform', 'preloaded'), path + '.prior', 'PRIOR_MODE')
    strength = _float(prior['strength'], path + '.prior.strength', 0)
    odds_cap = None if prior['odds_cap'] is None else _float(prior['odds_cap'], path + '.prior.odds_cap', 1)
    if prior['mode'] == 'uniform':
        R._require(strength == 0 and prior['source_sha256'] is None and odds_cap == 1., 'UNIFORM_PRIOR', path)
    else:
        R._require(strength > 0, 'PRIOR_STRENGTH', path)
        R._hash(prior['source_sha256'], path + '.prior.source_sha256')
    evidence = value['public_evidence']
    _closed(evidence, ('context_sha256', 'through_step', 'producer_sha256', 'source_sha256'), path + '.public_evidence')
    R._hash_join(evidence['context_sha256'], context_key, 'ACTOR_CONTEXT_MISMATCH', path)
    R._int(evidence['through_step'], path + '.public_evidence.through_step')
    R._require(evidence['through_step'] <= context['observation']['step'], 'FUTURE_EVIDENCE', path)
    for key in ('producer_sha256', 'source_sha256'): R._hash(evidence[key], path + '.public_evidence.' + key)
    probes = _probes(value['distinctness'], context, context_key, candidate_ids)
    probe_ids = {p['probe_id'] for p in probes}
    seen, result, unknowns = set(), [], 0
    mass_fields = ('raw_prior', 'initial_mass', 'posterior_mass', 'constrained_mass')
    fields = ('hypothesis_id', 'kind', 'model', 'provenance', *mass_fields, 'log_evidence',
              'conditional_wdl', 'probe_predictions')
    for h in rows:
        loc = path + '.hypotheses'
        _closed(h, fields, loc)
        _enum(h['kind'], ('MODEL', 'UNKNOWN'), loc, 'HYPOTHESIS_KIND')
        unknowns += h['kind'] == 'UNKNOWN'
        _closed(h['model'], ('definition_sha256', 'parameters_sha256'), loc + '.model')
        for key, val in h['model'].items(): R._hash(val, loc + '.model.' + key)
        hid = canonical_sha256(h['model'])
        R._hash_join(h['hypothesis_id'], hid, 'HYPOTHESIS_HASH_MISMATCH', loc)
        R._require(hid not in seen, 'DUPLICATE_HYPOTHESIS', loc); seen.add(hid)
        provenance = h['provenance']
        _closed(provenance, ('basis', 'source_sha256', 'as_of_step'), loc + '.provenance')
        R._require(provenance['basis'] == 'PUBLIC_HYPOTHETICAL_MODEL', 'PRIVATE_PROVENANCE', loc)
        R._hash(provenance['source_sha256'], loc + '.provenance.source_sha256')
        R._int(provenance['as_of_step'], loc + '.provenance.as_of_step')
        R._require(provenance['as_of_step'] <= context['observation']['step'], 'FUTURE_EVIDENCE', loc)
        out = copy.deepcopy(h)
        for key in mass_fields: out[key] = _float(h[key], loc + '.' + key, 0, 1)
        out['log_evidence'] = _float(h['log_evidence'], loc + '.log_evidence', high=0)
        _closed(h['conditional_wdl'], candidate_ids, loc + '.conditional_wdl')
        out['conditional_wdl'] = {key: None if law is None else _probabilities(law, 3, loc + '.conditional_wdl')
                                  for key, law in h['conditional_wdl'].items()}
        _closed(h['probe_predictions'], probe_ids, loc + '.probe_predictions')
        out['probe_predictions'] = {}
        for probe in probes:
            law = h['probe_predictions'][probe['probe_id']]
            law = None if law is None else _probabilities(law, probe['outcome_count'], loc + '.probe_predictions')
            out['probe_predictions'][probe['probe_id']] = law
            if probe['target'] == 'TERMINAL_WDL':
                R._require(law == out['conditional_wdl'][probe['candidate_sha256']], 'CONDITIONAL_PROBE_MISMATCH', loc)
        result.append(out)
    R._require(unknowns <= 1, 'MULTIPLE_UNKNOWN', path)
    vectors = {key: _probabilities([r[key] for r in result], k, path + '.' + key, positive=key != 'raw_prior')
               for key in mass_fields}
    initial, posterior, constrained = (vectors[key] for key in ('initial_mass', 'posterior_mass', 'constrained_mass'))
    if prior['mode'] == 'uniform':
        R._require(all(_close(v, 1/k) for v in initial + vectors['raw_prior']), 'UNIFORM_PRIOR', path)
    if odds_cap is not None:
        R._require(max(initial)/min(initial) <= odds_cap or _close(max(initial)/min(initial), odds_cap),
                   'PRIOR_ODDS_CAP', path)
    # Subtract the common evidence offset BEFORE adding log prior: a very large
    # equal negative likelihood must not round away the initial odds.
    shift = max(h['log_evidence'] for h in result)
    weights = [math.log(p) + (h['log_evidence'] - shift) for p, h in zip(initial, result)]
    peak = max(weights)
    unnormalized = [math.exp(w-peak) for w in weights]
    R._require(all(p > 0 for p in unnormalized), 'POSTERIOR_UNREPRESENTABLE', path)
    normalizer = math.fsum(unnormalized)
    expected = [p/normalizer for p in unnormalized]
    R._require(all(p > 0 for p in expected), 'POSTERIOR_UNREPRESENTABLE', path)
    R._require(all(_close(p, q) for p, q in zip(posterior, expected)), 'POSTERIOR_EVIDENCE_MISMATCH', path)
    R._require(all(p >= floor for p in constrained), 'FLOOR_VIOLATION', path)
    transform = value['floor_transform']
    if transform == 'none':
        R._require(floor == 0 and all(_close(p, q) for p, q in zip(posterior, constrained)), 'FLOOR_TRANSFORM', path)
    elif transform == 'two_hypothesis_clamp':
        R._require(k == 2, 'FLOOR_TRANSFORM', path)
        # Clamp each normalized coordinate symmetrically. Computing the second
        # as 1-first cancels a tiny live tail and makes acceptance row-order dependent.
        expected_constrained = [min(1-floor, max(floor, p)) for p in posterior]
        R._require(all(_close(p, q) for p, q in zip(constrained, expected_constrained)),
                   'FLOOR_TRANSFORM', path)
    out = copy.deepcopy(value)
    out.update(p_min=floor, hypotheses=sorted(result, key=lambda h: h['hypothesis_id']),
               prior={**prior, 'strength': strength, 'odds_cap': odds_cap},
               distinctness=dict(rule=DISTINCTNESS_RULE, probes=probes))
    _distinctness(out, candidate_ids)  # rejects provable on-panel duplicate rows
    return out


def _distinctness(belief, candidate_ids):
    probes, rows = belief['distinctness']['probes'], belief['hypotheses']
    coverage = {p['candidate_sha256'] for p in probes} >= set(candidate_ids)
    distinct, unresolved = 0, 0
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            pairs = [(a['probe_predictions'][p['probe_id']], b['probe_predictions'][p['probe_id']]) for p in probes]
            different = any(x is not None and y is not None and x != y for x, y in pairs)
            complete = bool(pairs) and coverage and all(x is not None and y is not None for x, y in pairs)
            R._require(not (complete and not different), 'DUPLICATE_PREDICTIVE_PROFILE', '$.actor.belief.distinctness')
            if different: distinct += 1
            else: unresolved += 1
    return dict(status='UNRESOLVED' if unresolved else 'DISTINCT_ON_DECLARED_PANEL',
                distinct_pairs=distinct, unresolved_pairs=unresolved,
                candidate_intervention_coverage=coverage, global_identity_certified=False)


def _actor(value, pins):
    R._safe(value)
    _closed(value, ('schema', 'source_sha256', 'context', 'context_sha256', 'candidates', 'belief',
                    'selected_candidate_sha256'), '$.actor')
    R._schema(value['schema'], ACTOR_SCHEMA, '$.actor.schema')
    R._hash_join(value['source_sha256'], canonical_sha256(pins), 'SOURCE_MISMATCH', '$.actor.source_sha256')
    state, prefix, options = _context(value['context'])
    context_key = canonical_sha256(value['context'])
    R._hash_join(value['context_sha256'], context_key, 'ACTOR_CONTEXT_MISMATCH', '$.actor.context_sha256')
    rows = R._list(value['candidates'], '$.actor.candidates')
    R._require(0 < len(rows) <= MAX_CANDIDATES, 'CANDIDATE_LIMIT', '$.actor.candidates')
    candidates, seen, menus = [], set(), {}
    for candidate in rows:
        _closed(candidate, ('schema', 'candidate_sha256', 'raw_actions', 'prefix_class_trace', 'canonical_plan',
                            'continuation_signature'), '$.actor.candidates')
        expected = _candidate(value['context'], candidate['raw_actions'], state, prefix, options, menus)
        R._require(R._equal(candidate, expected), 'CANDIDATE_REPLAY_MISMATCH', '$.actor.candidates')
        key = expected['candidate_sha256']
        R._require(key not in seen, 'DUPLICATE_CANDIDATE', '$.actor.candidates'); seen.add(key)
        candidates.append(expected)
    selected = value['selected_candidate_sha256']
    R._require(selected is None or (type(selected) is str and selected in seen), 'UNKNOWN_CANDIDATE', '$.actor.selected_candidate_sha256')
    b = _belief(value['belief'], value['context'], context_key, seen)
    return {**copy.deepcopy(value), 'candidates': sorted(candidates, key=lambda c: c['candidate_sha256']), 'belief': b}


def make_record(context: dict, belief: dict, candidates: list[dict], *,
                selected_candidate_sha256: str | None = None, trainer_only: dict | None = None) -> dict:
    """Seal one decision; safe inputs are copied, not mutated or auto-repaired."""
    trainer_only = {} if trainer_only is None else trainer_only
    for value in (context, belief, candidates, trainer_only): R._safe(value)
    R._require(type(trainer_only) is dict, 'INVALID_OBJECT', '$.trainer_only')
    pins = source_pins()
    actor = _actor(dict(schema=ACTOR_SCHEMA, source_sha256=canonical_sha256(pins), context=context,
                        context_sha256=canonical_sha256(context), candidates=candidates, belief=belief,
                        selected_candidate_sha256=selected_candidate_sha256), pins)
    record = dict(schema=SCHEMA, source_pins=pins, actor=actor, actor_payload_sha256=canonical_sha256(actor),
                  trainer_only=copy.deepcopy(trainer_only))
    return {**record, 'content_sha256': canonical_sha256(record)}


def validate_record(document: dict, *, expected_actor_sha256: str | None = None) -> dict:
    """Detached canonical record or stable RecordError; never upgrade evidence."""
    R._safe(document)
    R._fields(document, ('schema', 'source_pins', 'actor', 'actor_payload_sha256', 'trainer_only', 'content_sha256'), '$')
    R._schema(document['schema'], SCHEMA, '$.schema')
    pins = source_pins()
    R._require(R._equal(document['source_pins'], pins), 'SOURCE_MISMATCH', '$.source_pins')
    actor = _actor(document['actor'], pins)
    R._require(type(document['trainer_only']) is dict, 'INVALID_OBJECT', '$.trainer_only')
    key = canonical_sha256(actor)
    R._hash_join(document['actor_payload_sha256'], key, 'ACTOR_HASH_MISMATCH', '$.actor_payload_sha256')
    if expected_actor_sha256 is not None:
        R._hash_join(expected_actor_sha256, key, 'FROZEN_ACTOR_MISMATCH', '$.actor_payload_sha256')
    out = {**copy.deepcopy(document), 'actor': actor}
    R._hash_join(out['content_sha256'], canonical_sha256({k: v for k, v in out.items() if k != 'content_sha256'}),
                 'CONTENT_HASH_MISMATCH', '$.content_sha256')
    return out


def actor_payload_key(actor: dict) -> str:
    """Only accepts the closed actor payload, not a whole evaluator record."""
    return canonical_sha256(_actor(actor, source_pins()))


def actor_payload(document: dict) -> dict:
    return validate_record(document)['actor']


def distinctness_status(document: dict) -> dict:
    actor = actor_payload(document)
    return _distinctness(actor['belief'], {c['candidate_sha256'] for c in actor['candidates']})


def _wire(document: dict) -> str:
    return json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)


def dumps(document: dict) -> str:
    return _wire(validate_record(document)) + '\n'


def loads(text: str, *, expected_actor_sha256: str | None = None) -> dict:
    return validate_record(R._loads(text), expected_actor_sha256=expected_actor_sha256)


def read_record(path: str | Path, *, expected_actor_sha256: str | None = None) -> dict:
    return loads(Path(path).read_text(encoding='utf-8'), expected_actor_sha256=expected_actor_sha256)


def write_record(path: str | Path, document: dict) -> None:
    """Validate before creating the file; existing actor-time snapshots refuse overwrite."""
    text = dumps(document)
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(text)


@dataclass(frozen=True, init=False)
class FrozenBeliefRecord:
    """Owns immutable serialized bytes; every returned dict is detached."""
    _wire: str

    def __init__(self, document: dict):
        object.__setattr__(self, '_wire', _wire(validate_record(document)))

    @classmethod
    def from_document(cls, document: dict) -> FrozenBeliefRecord:
        return cls(document)

    def to_dict(self) -> dict:
        return R._loads(self._wire)

    def actor_payload(self) -> dict:
        return self.to_dict()['actor']

    @property
    def actor_payload_sha256(self) -> str:
        return self.to_dict()['actor_payload_sha256']

    def with_trainer(self, trainer_only: dict) -> FrozenBeliefRecord:
        """Attach/replace private evidence on a NEW record, never revise the decision."""
        R._safe(trainer_only)
        R._require(type(trainer_only) is dict, 'INVALID_OBJECT', '$.trainer_only')
        out = self.to_dict(); out['trainer_only'] = copy.deepcopy(trainer_only)
        out['content_sha256'] = canonical_sha256({k: v for k, v in out.items() if k != 'content_sha256'})
        validate_record(out, expected_actor_sha256=self.actor_payload_sha256)
        return type(self)(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('validate')
    command.add_argument('--input', required=True)
    command.add_argument('--actor-sha256', default=None)
    args = parser.parse_args(argv)
    try:
        document = read_record(args.input, expected_actor_sha256=args.actor_sha256)
    except (RecordError, OSError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(_wire(dict(schema='farm-v0-belief-validation-v1', status='PASS',
                     actor_payload_sha256=document['actor_payload_sha256'],
                     content_sha256=document['content_sha256'], native_games=0)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
