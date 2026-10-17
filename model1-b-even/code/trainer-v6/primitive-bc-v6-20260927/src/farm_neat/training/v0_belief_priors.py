"""Opt-in actor-side loading of approved, frozen hypothesis prior bundles.

``load_prior_bundle(bytes_or_path, expected_bundle_sha256=..., ... )`` requires
INDEPENDENT byte, registry and caller-approved support pins. An upstream offline auditor owns
source-row provenance/count reconstruction and admission; this loader never reads
bundle-supplied corpus/model paths, evaluator records or current-game labels. A
self-hash is not approval. Changing both trusted pins and data defeats that trust
boundary. The caller owns executable-catalog admission BEFORE approving the support pin;
this loader verifies exact pinned metadata, not artifact availability or executability.
A display tape ID is not an opponent identity or an executable tape certificate.

Consumes the accepted ``farm-v0-prior-bundle-v1`` shape, including its smoothed
static-tape counts, optional UNKNOWN base allocation, temperature and initial odds
cap. Rechecks schema, support/model/descriptor identities, public-source metadata,
count totals and every initialization operand. Zero raw counts remain visible.
IDs outside the caller-approved support refuse; no runtime intersection, silent uniform
fallback or inferred UNKNOWN mapping. Approved offline retired-to-UNKNOWN metadata
is allowed only in a new support version. No mid-game migration is implemented.

``initialize_belief(raw_model_logits, hypothesis_ids=..., kinds=..., prior=...)``
joins by content identity and uses the existing belief_masses implementation. The
frozen temperature gives log-odds strength 1/T, NOT N+alpha. The cap is applied
once, at initialization. To change strength, approve a new bundle/recipe; no
unrecorded runtime override or annealing. UNKNOWN's base mass is smoothing, not a
posterior floor. Optional output floors remain the existing head's responsibility.

The active path requires a closed public selector, a caller-pinned FULL actor
snapshot hash, and game-start time. The projection cannot establish that the
supplied snapshot or arbitrary model tensors really are actor-visible: the caller
owns capture, chronology and the public feature encoder. Model logits remain
UNCALIBRATED scores, not certified public likelihoods. The initializer itself is not
an event updater; never supply previous posterior logits plus replayed evidence as
fresh model scores.

No prior, or enabled=False, delegates exactly to the unchanged uniform head path.
Uniform bundles likewise export the existing uniform record semantics (strength=0,
source=null), retaining the approved corpus only in the audit sidecar.

The opt-in initialize_public_flow_belief bridge initializes the existing public
updater with the same frozen smoothed prior/temperature/cap and ZERO evidence. It
returns (updater, initial_state, initialization_audit); only subsequent typed public
observations add likelihood. Corpus counts stay in the audit, never interval counts.
This is fixed-prior Bayes, not annealing: the accepted prior-strength study supports
conditional log-odds budgets, not a universal strength or a wire-v1 decay schedule.
There is no production actor wiring, checkpoint I/O, training, selection or
objective change.

PriorBeliefOutput preserves raw frequency, smoothed numerical head input, actual
prior log contribution, raw and centered model scores, initial/posterior/constrained
masses separately. record_fields() restores the bundle's RAW frequency field for
the existing belief-record wire; prediction.raw_prior is the positive SMOOTHED
operand used by the head. Final actor-record sealing remains the existing validator.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from torch import Tensor
    from farm_neat.policy.v0_belief import BeliefHeadOutput
    from farm_neat.policy.v0_belief_update import BayesianFlowUpdater, FlowRegimeLaw, OnlineBeliefState

SCHEMA = 'farm-v0-prior-initialization-v1'
SCHEMA_SHA256 = '0675c4f07a7890adb0b714bba19a77e4f2dd97e7269817a63203d66254d88cee'
MAX_BYTES = 2_000_000
DENIED_PARTITIONS = frozenset({'discovery', 'confirmation', 'monitor', 'final', 'held_out', 'evaluation', 'test'})
PUBLIC_BASES = {'training': 'PUBLIC_TRAINING_BEHAVIOR', 'reference': 'PUBLIC_REFERENCE'}

# Exact bundle definition from the accepted provenance SCHEMA.json. Factoring
# repeated shapes keeps the actor module independent of disposable work/ files.
def _object(fields, required=None):
    return dict(type='object', properties=fields, required=list(fields) if required is None else required,
                additionalProperties=False)


def _array(item, lo, hi, *, unique=False):
    return dict(type='array', items=item, minItems=lo, maxItems=hi, **({'uniqueItems': True} if unique else {}))


def _map(item):
    return dict(type='object', patternProperties={'^[0-9a-f]{64}$': item},
                additionalProperties=False, minProperties=1, maxProperties=64)


_HASH = dict(type='string', pattern='^[0-9a-f]{64}$')
_ID = dict(type='string', minLength=1, maxLength=256, pattern='^[A-Za-z0-9_:./@-]+$')
_INT = dict(type='integer', minimum=0, maximum=9007199254740991)
_VERSION = dict(type='integer', minimum=1)
_NUMBER = dict(type='number')
_NONNEG = dict(type='number', minimum=0)
_PROB = dict(type='number', minimum=0, maximum=1)
_POS_PROB = dict(type='number', exclusiveMinimum=0, maximum=1)
_NULL_HASH = dict(anyOf=[_HASH, dict(type='null')])
_PARTITION = dict(enum=['training', 'reference', 'discovery', 'confirmation', 'monitor', 'final', 'held_out', 'evaluation', 'test'])
_BASIS = dict(enum=['PUBLIC_REFERENCE', 'PUBLIC_TRAINING_BEHAVIOR', 'TRAINER_PRIVATE_IDENTITY',
                   'REALIZED_OUTCOME', 'FUTURE_RANDOMNESS', 'HIDDEN_STATE'])
_CONTEXT_FIELDS = dict(player=dict(type='integer', minimum=0, maximum=1),
                       step=dict(type='integer', minimum=0, maximum=718),
                       unlocked_shops=_array(_ID, 0, 32))
_CONTEXT = _object(_CONTEXT_FIELDS, required=[])
_UNKNOWN = _object(dict(hypothesis_id=_NULL_HASH, base_mass=_PROB))
_MODEL = _object(dict(definition_sha256=_HASH, parameters_sha256=_HASH))
_HYPOTHESIS = _object(dict(kind=dict(enum=['MODEL', 'UNKNOWN']),
    family=dict(enum=['STATIC_TAPE', 'OPPONENT_MODEL', 'WORLD_LAW', 'UNKNOWN']),
    tape_id=dict(anyOf=[_ID, dict(type='null')]), model=_MODEL,
    definition_artifact=_ID, parameters_artifact=_ID))
_DESCRIPTOR = _object(dict(path=_ID, sha256=_HASH,
    kind=dict(enum=['corpus', 'model_definition', 'model_parameters', 'likelihood_producer']),
    partition=_PARTITION, basis=_BASIS, frozen_at=_INT))
_RECIPE = _object(dict(mode=dict(enum=['uniform', 'preloaded']),
    source_ids=_array(_ID, 1, 64, unique=True), support_ids=_array(_HASH, 1, 64, unique=True),
    support_version=_VERSION, previous_bundle_sha256=_NULL_HASH,
    retired_to_unknown=_array(_HASH, 0, 64, unique=True),
    context_keys=_array(dict(enum=['player', 'step', 'unlocked_shops']), 0, 3, unique=True),
    context_selector=_CONTEXT,
    smoothing=_object(dict(method=dict(enum=['dirichlet_base']), total_pseudo_count=_NONNEG)),
    temperature=dict(type='number', exclusiveMinimum=0), initial_odds_cap=dict(type='number', minimum=1),
    unknown=_UNKNOWN))
_INITIALIZATION = _object(dict(counts=_map(_INT), raw_frequency=_map(_PROB), raw_prior=_map(_PROB),
    smoothed_mass=_map(_POS_PROB), tempered_logits=_map(_NUMBER), logits=_map(_NUMBER),
    mass=_map(_POS_PROB), log_mass=_map(_NUMBER), effective_prior_count=_NONNEG))
_CORPUS_SOURCE = _object(dict(source_id=_ID, sha256=_HASH, corpus_id=_ID,
    window=_object(dict(start_inclusive=_INT, end_exclusive=_INT)),
    partition=_PARTITION, basis=_BASIS, total_groups=_INT, matched_groups=_INT))
BUNDLE_SCHEMA = _object(dict(schema=dict(enum=['farm-v0-prior-bundle-v1']),
    registry_sha256=_HASH, registry_id=_ID, frozen_at=_INT, builder_sha256=_HASH, schema_sha256=_HASH,
    excluded_partitions=_array(_PARTITION, 1, 9, unique=True),
    excluded_corpus_ids=_array(_ID, 0, 10000, unique=True), excluded_group_ids=_array(_ID, 0, 10000, unique=True),
    recipe=_RECIPE,
    source_bindings=dict(type='object', additionalProperties=_DESCRIPTOR,
                         minProperties=1, maxProperties=256, propertyNames=_ID),
    support=_object(dict(version=_VERSION, previous_bundle_sha256=_NULL_HASH,
                         hypotheses=_map(_HYPOTHESIS), sha256=_HASH)),
    corpus=_object(dict(effective_sample_count=_INT,
        count_unit=dict(enum=['ONE_SOURCE_GROUP_NOT_CURRENT_GAME_OBSERVATION']),
        groups_sha256=_HASH, sources=_array(_CORPUS_SOURCE, 1, 64))),
    initialization=_INITIALIZATION,
    unknown=_object(dict(hypothesis_id=_NULL_HASH, base_mass=_PROB, initial_mass=_PROB,
        semantics=dict(enum=['UNMODELED_RESIDUAL_EXPERT_NOT_TRUTH_PROBABILITY']))),
    content_sha256=_HASH))
_CONTEXT_SCHEMA = _object(_CONTEXT_FIELDS)
SUPPORT_VALIDATION = 'CALLER_APPROVED_SUPPORT_PIN_ONLY'


class PriorError(ValueError):
    """Fail-closed bundle, identity, context or numerical refusal."""


def _require(condition, code, detail=''):
    if not condition:
        raise PriorError(f'{code}: {detail}')


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash(value, name):
    _require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), name)


def _finite(value, depth=0):
    _require(depth <= 40, 'JSON_DEPTH')
    if type(value) is float:
        _require(math.isfinite(value), 'NONFINITE_JSON')
    elif type(value) is int:
        _require(abs(value) <= 9007199254740991, 'JSON_INTEGER_RANGE')
    elif type(value) is dict:
        for child in value.values(): _finite(child, depth+1)
    elif type(value) is list:
        for child in value: _finite(child, depth+1)
    else:
        _require(value is None or type(value) in (str, bool), 'JSON_TYPE')


def _json(raw):
    _require(type(raw) is bytes and len(raw) <= MAX_BYTES, 'JSON_BYTES_LIMIT')
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'DUPLICATE_JSON_KEY', key)
            result[key] = value
        return result
    def constant(value):
        raise PriorError('NONFINITE_JSON: '+value)
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, PriorError): raise
        raise PriorError('INVALID_JSON') from exc
    _finite(value)
    return value


# Only the keywords in the frozen bundle/context schemas are supported. This is
# not a general JSON Schema engine. A schema extension must fail closed until its
# behavior is implemented and tested. JSON size/depth is bounded before this walk;
# object/array bounds are checked before walking their children.
_SHAPE_KEYWORDS = frozenset({
    'type', 'enum', 'anyOf', 'properties', 'required', 'additionalProperties',
    'patternProperties', 'propertyNames', 'minProperties', 'maxProperties',
    'items', 'minItems', 'maxItems', 'uniqueItems', 'pattern', 'minLength',
    'maxLength', 'minimum', 'maximum', 'exclusiveMinimum',
})


def _shape(value: Any, schema: dict, path: tuple = (), depth: int = 0) -> None:
    """Validate the exact retained schema subset using only the standard library."""
    location = '/'.join(map(str, path)) or '<root>'
    def check(condition, detail):
        _require(condition, 'SCHEMA', location + ': ' + detail)
    _require(depth <= 40, 'JSON_DEPTH')
    _require(not set(schema).difference(_SHAPE_KEYWORDS), 'SCHEMA_DEFINITION', location)
    if 'anyOf' in schema:
        matched = False
        for choice in schema['anyOf']:
            try:
                _shape(value, choice, path, depth + 1)
            except PriorError:
                continue
            matched = True
            break
        check(matched, 'no anyOf alternative matched')
    if 'enum' in schema:
        # Every enum in the frozen schemas consists solely of strings.
        check(type(value) is str and value in schema['enum'], 'invalid enum value')
    kind = schema.get('type')
    if kind is not None:
        is_number = type(value) in (int, float)
        types = {
            'object': type(value) is dict, 'array': type(value) is list,
            'string': type(value) is str, 'null': value is None,
            'number': is_number,
            # JSON Schema integers include 1.0, but exclude True and False.
            'integer': type(value) is int or (type(value) is float and value.is_integer()),
        }
        _require(kind in types, 'SCHEMA_DEFINITION', 'unsupported type')
        check(types[kind], 'expected ' + kind)
    if type(value) in (int, float):
        for bound in ('minimum', 'maximum', 'exclusiveMinimum'):
            if bound in schema:
                limit = schema[bound]
                good = value >= limit if bound == 'minimum' else value <= limit if bound == 'maximum' else value > limit
                check(good, bound + ' violated')
    if type(value) is str:
        check(len(value) >= schema.get('minLength', 0), 'string too short')
        check(len(value) <= schema.get('maxLength', MAX_BYTES), 'string too long')
        if 'pattern' in schema:
            check(re.search(schema['pattern'], value) is not None, 'pattern mismatch')
    if type(value) is list:
        check(len(value) >= schema.get('minItems', 0), 'too few items')
        check(len(value) <= schema.get('maxItems', MAX_BYTES), 'too many items')
        if 'items' in schema:
            for index, item in enumerate(value):
                _shape(item, schema['items'], (*path, index), depth + 1)
        if schema.get('uniqueItems'):
            # Unique arrays in this frozen schema are strings only; item validation
            # above precedes this linear-time check (up to 10,000 exclusion IDs).
            check(len({_wire(item) for item in value}) == len(value), 'duplicate items')
    if type(value) is dict:
        check(all(type(key) is str for key in value), 'non-string object key')
        check(len(value) >= schema.get('minProperties', 0), 'too few properties')
        check(len(value) <= schema.get('maxProperties', MAX_BYTES), 'too many properties')
        check(set(schema.get('required', ())) <= set(value), 'missing required property')
        properties = schema.get('properties', {})
        patterns = schema.get('patternProperties', {})
        for key, child in value.items():
            if 'propertyNames' in schema:
                _shape(key, schema['propertyNames'], (*path, key), depth + 1)
            matched = False
            if key in properties:
                _shape(child, properties[key], (*path, key), depth + 1)
                matched = True
            for pattern, child_schema in patterns.items():
                if re.search(pattern, key):
                    _shape(child, child_schema, (*path, key), depth + 1)
                    matched = True
            if not matched:
                extra = schema.get('additionalProperties', True)
                check(extra is not False, 'unexpected property ' + key)
                if type(extra) is dict:
                    _shape(child, extra, (*path, key), depth + 1)


def _initialization(recipe, counts):
    """Same fixed-prior arithmetic as the accepted offline provenance auditor."""
    n, k = sum(counts.values()), len(counts)
    raw = {key: count/n for key, count in counts.items()}
    if recipe['mode'] == 'uniform':
        mass = {key: 1/k for key in counts}; logits = {key: 0. for key in counts}
        return dict(counts=counts, raw_frequency=raw, raw_prior=mass, smoothed_mass=mass,
            tempered_logits=logits, logits=logits, mass=mass,
            log_mass={key: math.log(p) for key, p in mass.items()}, effective_prior_count=0.)
    alpha, temperature = recipe['smoothing']['total_pseudo_count'], recipe['temperature']
    uid, reserve = recipe['unknown']['hypothesis_id'], recipe['unknown']['base_mass']
    base = {key: (reserve if key == uid else (1-reserve)/(k-1) if uid else 1/k) for key in counts}
    denominator = n+alpha
    _require(math.isfinite(denominator) and denominator > 0, 'NUMERICAL_RANGE')
    smooth = {key: (count+alpha*base[key])/denominator for key, count in counts.items()}
    _require(all(math.isfinite(p) and p > 0 for p in smooth.values()), 'ZERO_SUPPORT')
    logs = {key: math.log(p) for key, p in smooth.items()}; peak = max(logs.values())
    tempered = {key: (value-peak)/temperature for key, value in logs.items()}
    _require(all(math.isfinite(v) for v in tempered.values()), 'NUMERICAL_RANGE')
    capped = {key: max(value, -math.log(recipe['initial_odds_cap'])) for key, value in tempered.items()}
    weights = {key: math.exp(value) for key, value in capped.items()}; total = math.fsum(weights.values())
    mass = {key: value/total for key, value in weights.items()}
    _require(all(math.isfinite(p) and p > 0 for p in mass.values()), 'ZERO_SUPPORT')
    return dict(counts=counts, raw_frequency=raw, raw_prior=raw, smoothed_mass=smooth,
        tempered_logits=tempered, logits=capped, mass=mass,
        log_mass={key: math.log(p) for key, p in mass.items()}, effective_prior_count=denominator)


def _validate(raw, bundle_pin, registry_pin, support_pin):
    for value, name in ((bundle_pin, 'BUNDLE_HASH'), (registry_pin, 'REGISTRY_HASH'), (support_pin, 'SUPPORT_HASH')):
        _hash(value, name)
    _require(type(raw) is bytes and len(raw) <= MAX_BYTES, 'JSON_BYTES_LIMIT')
    _require(_sha(raw) == bundle_pin, 'BUNDLE_HASH')
    b = _json(raw); _shape(b, BUNDLE_SCHEMA)
    _require(_sha(_wire({k: v for k, v in b.items() if k != 'content_sha256'})) == b['content_sha256'], 'CONTENT_HASH')
    _require(b['registry_sha256'] == registry_pin, 'REGISTRY_HASH')
    _require(b['schema_sha256'] == SCHEMA_SHA256, 'SCHEMA_HASH')
    r, s, init = b['recipe'], b['support'], b['initialization']
    rows, bindings = s['hypotheses'], b['source_bindings']
    _require(s['sha256'] == support_pin == _sha(_wire(rows)), 'SUPPORT_HASH')
    keys = set(rows)
    _require(set(r['support_ids']) == keys, 'SUPPORT_IDS')
    _require(set(r['context_keys']) == set(r['context_selector']), 'CONTEXT_KEYS')
    _require(s['version'] == r['support_version'] and s['previous_bundle_sha256'] == r['previous_bundle_sha256'], 'SUPPORT_VERSION')
    _require((s['version'] == 1) == (s['previous_bundle_sha256'] is None), 'SUPPORT_VERSION')
    _require(not keys.intersection(r['retired_to_unknown']) and (s['version'] > 1 or not r['retired_to_unknown']), 'RETIRED_SUPPORT')
    _require(DENIED_PARTITIONS <= set(b['excluded_partitions']), 'INCOMPLETE_EXCLUSIONS')
    for aid, d in bindings.items():
        _require(d['partition'] in PUBLIC_BASES and d['partition'] not in b['excluded_partitions'], 'EXCLUDED_PARTITION', aid)
        _require(d['basis'] == PUBLIC_BASES[d['partition']], 'PRIVATE_SOURCE', aid)
        _require(d['frozen_at'] <= b['frozen_at'], 'UNFROZEN_SOURCE', aid)
        _require(':' not in d['path'] and all(p not in ('', '.', '..') for p in d['path'].split('/')), 'UNSAFE_SOURCE_PATH', aid)
    used = set(r['source_ids']); unknowns = []
    for hid, h in rows.items():
        _require(_sha(_wire(h['model'])) == hid, 'HYPOTHESIS_HASH', hid)
        for field, kind, model_key in [('definition_artifact', 'model_definition', 'definition_sha256'),
                                       ('parameters_artifact', 'model_parameters', 'parameters_sha256')]:
            aid = h[field]; used.add(aid)
            _require(aid in bindings and bindings[aid]['kind'] == kind, 'MODEL_ARTIFACT', aid)
            _require(bindings[aid]['sha256'] == h['model'][model_key], 'MODEL_HASH', hid)
        if h['kind'] == 'UNKNOWN':
            unknowns.append(hid)
            _require(h['family'] == 'UNKNOWN' and h['tape_id'] is None, 'UNKNOWN_KIND')
        else:
            _require(h['family'] != 'UNKNOWN' and ((h['family'] == 'STATIC_TAPE') == (h['tape_id'] is not None)), 'TAPE_ID')
    uid, reserve = r['unknown']['hypothesis_id'], r['unknown']['base_mass']
    _require(unknowns == ([] if uid is None else [uid]), 'UNKNOWN_SUPPORT')
    _require((reserve == 1 if len(rows) == 1 else 0 < reserve < 1) if uid else reserve == 0 and not r['retired_to_unknown'], 'UNKNOWN_BASE_MASS')
    _require(used == set(bindings), 'SOURCE_BINDINGS')
    sources = b['corpus']['sources']; source_ids = [x['source_id'] for x in sources]
    _require(len(source_ids) == len(set(source_ids)) and set(source_ids) == set(r['source_ids']), 'SOURCE_IDS')
    _require(len({x['sha256'] for x in sources}) == len(sources), 'DUPLICATE_SOURCE')
    for source in sources:
        aid = source['source_id']; d = bindings[aid]
        _require(d['kind'] == 'corpus' and all(source[k] == d[k] for k in ('sha256', 'partition', 'basis')), 'CORPUS_BINDING', aid)
        _require(source['corpus_id'] not in b['excluded_corpus_ids'], 'EXCLUDED_CORPUS', aid)
        lo, hi = source['window']['start_inclusive'], source['window']['end_exclusive']
        _require(lo < hi and source['matched_groups'] <= source['total_groups'] <= hi-lo, 'CORPUS_WINDOW', aid)
    for name, value in init.items():
        if name != 'effective_prior_count': _require(set(value) == keys, 'INITIALIZATION_SUPPORT', name)
    n = sum(init['counts'].values())
    _require(0 < n == b['corpus']['effective_sample_count'] == sum(x['matched_groups'] for x in sources), 'COUNT_TOTAL')
    if r['mode'] == 'uniform': _require(r['initial_odds_cap'] == 1, 'UNIFORM_POLICY')
    expected = _initialization(r, {h: init['counts'][h] for h in sorted(keys)})
    for name, value in expected.items():
        pairs = [(value, init[name])] if name == 'effective_prior_count' else [(value[h], init[name][h]) for h in keys]
        _require(all(math.isclose(a, z, rel_tol=1e-12, abs_tol=0.) for a, z in pairs), 'INITIALIZATION_RECOMPUTE', name)
    u = b['unknown']
    _require(u['hypothesis_id'] == uid and u['base_mass'] == reserve, 'UNKNOWN_METADATA')
    _require(u['initial_mass'] == (init['mass'][uid] if uid else 0.), 'UNKNOWN_METADATA')
    return b


@dataclass(frozen=True)
class FrozenBeliefPrior:
    """Immutable approved bytes; ordinary construction always validates the trust join."""
    _bytes: bytes
    bundle_sha256: str
    registry_sha256: str
    support_sha256: str

    def __post_init__(self):
        _validate(self._bytes, self.bundle_sha256, self.registry_sha256, self.support_sha256)

    def to_bytes(self) -> bytes:
        return self._bytes

    def to_dict(self) -> dict:
        return json.loads(self._bytes)

    @property
    def source_id(self) -> str:
        return self.to_dict()['registry_id']

    @property
    def content_sha256(self) -> str:
        return self.to_dict()['content_sha256']


def load_prior_bundle(source: bytes | str | Path, *, expected_bundle_sha256: str,
                      expected_registry_sha256: str, expected_support_sha256: str) -> FrozenBeliefPrior:
    """Load ONLY the chosen bundle; expected pins must come from independent approval.

    No referenced corpus/model/likelihood artifact is opened or executed here.
    Upstream source auditing and executable-catalog admission remain caller-owned.
    Repinning unsupported metadata is NOT an executability check; audit output
    explicitly labels this boundary CALLER_APPROVED_SUPPORT_PIN_ONLY.
    """
    if type(source) is bytes:
        raw = source
    elif isinstance(source, (str, Path)):
        try:
            with Path(source).open('rb') as stream:
                raw = stream.read(MAX_BYTES+1)
        except OSError as exc:
            raise PriorError('BUNDLE_READ: '+str(exc)) from exc
    else:
        raise TypeError('source must be frozen JSON bytes or an explicit local bundle path')
    return FrozenBeliefPrior(raw, expected_bundle_sha256, expected_registry_sha256, expected_support_sha256)


@dataclass(frozen=True)
class PriorBeliefOutput:
    """Differentiable output plus immutable source/capture metadata, not a sealed record."""
    prediction: BeliefHeadOutput
    prior: FrozenBeliefPrior | None
    applied: bool
    actor_context_sha256: str | None
    _context_bytes: bytes | None
    game_started_at: int | None

    @property
    def raw_model_logits(self):
        return self.prediction.unconstrained_logits

    @property
    def model_logit_contribution(self):
        return self.prediction.log_evidence

    @property
    def prior_logit_contribution(self):
        return self.prediction.initial_mass.log()

    @property
    def initial_mass(self):
        return self.prediction.initial_mass

    @property
    def posterior_mass(self):
        return self.prediction.posterior_mass

    @property
    def constrained_mass(self):
        return self.prediction.constrained_mass

    @property
    def smoothed_prior(self):
        return self.prediction.raw_prior

    @property
    def raw_prior(self):
        if not self.applied:
            return self.prediction.raw_prior
        import torch
        values = self.prior.to_dict()['initialization']['raw_prior']
        return torch.tensor([values[h] for h in self.prediction.hypothesis_ids],
                            device=self.raw_model_logits.device, dtype=torch.float64).expand_as(self.initial_mass)

    def record_fields(self, *, batch_index: int | None = None, mass_used: str = 'posterior_mass') -> dict:
        fields = self.prediction.record_fields(batch_index=batch_index, mass_used=mass_used)
        if self.applied:
            values = self.prior.to_dict()['initialization']['raw_prior']
            for row in fields['hypotheses']: row['raw_prior'] = values[row['hypothesis_id']]
        return fields

    def to_dict(self, *, batch_index: int | None = None) -> dict:
        """Detached audit sidecar for one context; no private labels or hidden source reads."""
        fields = self.record_fields(batch_index=batch_index)
        def vector(tensor):
            if tensor.ndim == 2: tensor = tensor[batch_index]
            return tensor.detach().cpu().tolist()
        source = None
        if self.prior is not None:
            b = self.prior.to_dict()
            source = dict(id=b['registry_id'], bundle_sha256=self.prior.bundle_sha256,
                content_sha256=b['content_sha256'], registry_sha256=b['registry_sha256'],
                support_sha256=b['support']['sha256'], support_validation=SUPPORT_VALIDATION,
                recipe=b['recipe'],
                effective_prior_count=b['initialization']['effective_prior_count'],
                frozen_initialization=b['initialization'], unknown=b['unknown'])
        result = dict(schema=SCHEMA, applied=self.applied, source=source,
            actor_context_sha256=self.actor_context_sha256,
            public_context=None if self._context_bytes is None else json.loads(self._context_bytes),
            game_started_at=self.game_started_at, hypothesis_ids=list(self.prediction.hypothesis_ids),
            raw_model_logits=vector(self.raw_model_logits), model_logit_contribution=vector(self.model_logit_contribution),
            smoothed_prior=vector(self.smoothed_prior), prior_logit_contribution=vector(self.prior_logit_contribution),
            belief=fields, calibration_status='UNMEASURED')
        result['content_sha256'] = _sha(_wire(result))
        return result


def initialize_belief(logits: Tensor, *, hypothesis_ids: Sequence[str], kinds: Sequence[str] | None = None,
                      prior: FrozenBeliefPrior | None = None, enabled: bool = True,
                      public_context: dict | None = None, actor_context_sha256: str | None = None,
                      game_started_at: int | None = None, p_min: float = 0.,
                      floor_transform: str = 'none') -> PriorBeliefOutput:
    """Initialize from a fixed approved bundle, or preserve the exact no-prior path.

    A batch must share the SAME public-context/capture pins and catalog; independent
    actor contexts need separate calls/receipts. This function scores no model and
    never reads an actor record or trainer dictionary. No posterior is fed back.
    """
    from farm_neat.policy.v0_belief import belief_masses
    _require(type(enabled) is bool, 'ENABLED_BOOL')
    _require(prior is None or isinstance(prior, FrozenBeliefPrior), 'PRIOR_TYPE')
    kwargs = dict(hypothesis_ids=hypothesis_ids, kinds=kinds, p_min=p_min, floor_transform=floor_transform)
    if prior is None or not enabled:
        prediction = belief_masses(logits, **kwargs)
        return PriorBeliefOutput(prediction, prior if not enabled else None, False, None, None, None)
    b = prior.to_dict(); rows = b['support']['hypotheses']; keys = tuple(hypothesis_ids)
    _require(len(keys) == len(rows) and set(keys) == set(rows), 'SUPPORT_IDS')
    tags = ('MODEL',)*len(keys) if kinds is None else tuple(kinds)
    _require(len(tags) == len(keys) and all(tag == rows[h]['kind'] for h, tag in zip(keys, tags)), 'SUPPORT_KINDS')
    _hash(actor_context_sha256, 'ACTOR_CONTEXT_HASH')
    _finite(public_context); _shape(public_context, _CONTEXT_SCHEMA)
    _require(all(public_context[k] == value for k, value in b['recipe']['context_selector'].items()), 'CONTEXT_SELECTOR')
    _require(type(game_started_at) is int and 0 <= game_started_at <= 9007199254740991
             and b['frozen_at'] <= game_started_at, 'FROZEN_BEFORE_GAME')
    if b['recipe']['mode'] == 'preloaded':
        import torch
        _require(isinstance(logits, torch.Tensor), 'LOGITS_TENSOR')
        strength = 1/b['recipe']['temperature']
        _require(math.isfinite(strength) and strength > 0, 'PRIOR_STRENGTH')
        kwargs.update(raw_prior=torch.tensor([b['initialization']['smoothed_mass'][h] for h in keys],
                      dtype=torch.float64, device=logits.device), prior_strength=strength,
                      prior_odds_cap=b['recipe']['initial_odds_cap'], prior_source_sha256=b['content_sha256'])
    prediction = belief_masses(logits, **kwargs)
    # Guard the adapter join, including finite precision differences across kernels.
    import torch
    expected = torch.tensor([b['initialization']['mass'][h] for h in keys],
                            dtype=torch.float64, device=logits.device).expand_as(prediction.initial_mass)
    _require(bool(torch.isclose(prediction.initial_mass, expected, rtol=1e-10, atol=0.).all()), 'HEAD_INITIALIZATION_RECOMPUTE')
    return PriorBeliefOutput(prediction, prior, b['recipe']['mode'] == 'preloaded',
                            actor_context_sha256, _wire(public_context), game_started_at)


def initialize_public_flow_belief(product: str, hypothesis_ids: Sequence[str],
                                  regime_laws: Sequence[FlowRegimeLaw], *,
                                  player: int, through_step: int = 0,
                                  kinds: Sequence[str] | None = None,
                                  prior: FrozenBeliefPrior | None = None,
                                  enabled: bool = True, public_context: dict | None = None,
                                  actor_context_sha256: str | None = None,
                                  game_started_at: int | None = None, p_min: float = 0.,
                                  floor_transform: str = 'none'
                                  ) -> tuple[BayesianFlowUpdater, OnlineBeliefState, PriorBeliefOutput]:
    """Join an approved prior to the existing CPU public-event updater once.

    Returns ``(updater, state, initialization_audit)``. The actor owns the full
    capture hash/chronology, catalog admission and stationary public likelihood
    table; this bridge certifies none of their statistical accuracy. With an active
    bundle, the closed public-context player/step must equal the returned state's
    origin. Recreate this fixed configuration and use updater.restore_state for
    continuation; never restart initialization to consume the same history twice.
    Reset the observer's legitimate own memory separately at a new game.

    No raw logits, previous posterior, source rows, trainer records, pseudo-events
    or runtime strength overrides are accepted. The frozen recipe applies once:
    q_n = softmax(log(initial_mass) + cumulative_public_log_likelihood).
    Evidence-free initialization is not a neural prediction. Offline N+alpha is
    smoothing provenance, not live effective sample size or extra evidence.

    Preserve initialization_audit.to_dict() next to the state: it retains bundle,
    registry, support and actor-capture pins, raw frequencies (including zero),
    smoothing operands and the fixed recipe. The updater wire's raw_prior remains
    the positive SMOOTHED numerical input, as in its existing direct API; the
    loader audit separately retains raw frequency. Their initial and posterior
    masses are identical. Do not replace likelihoods with a posterior to reconcile
    those two explicitly different raw-prior fields.

    No-prior/enabled=False and approved uniform bundles use the EXACT original
    updater options/state path. Diagnostics are obtained separately through
    updater.contribution_diagnostics(state); nothing is added to actor history.
    """
    import torch
    from farm_neat.policy.v0_belief_update import BayesianFlowUpdater

    if isinstance(hypothesis_ids, (str, bytes)):
        raise TypeError('hypothesis_ids must be a sequence')
    keys = tuple(hypothesis_ids)
    tags = None if kinds is None else tuple(kinds)
    initialization = initialize_belief(
        torch.zeros(len(keys), dtype=torch.float64, device='cpu'),
        hypothesis_ids=keys, kinds=tags, prior=prior, enabled=enabled,
        public_context=public_context, actor_context_sha256=actor_context_sha256,
        game_started_at=game_started_at, p_min=p_min, floor_transform=floor_transform)
    if prior is not None and enabled:
        # initialize_belief has already checked the closed public context shape.
        _require(public_context['player'] == player and public_context['step'] == through_step,
                 'UPDATE_CONTEXT', 'public selector and state origin must agree')
    prediction = initialization.prediction
    kwargs = {}
    if initialization.applied:
        kwargs = dict(raw_prior=prediction.raw_prior.tolist(),
                      prior_strength=prediction.prior_strength,
                      prior_odds_cap=prediction.prior_odds_cap,
                      prior_source_sha256=prediction.prior_source_sha256)
    updater = BayesianFlowUpdater(product, keys, regime_laws, kinds=tags,
                                  p_min=p_min, floor_transform=floor_transform, **kwargs)
    state = updater.initialize(player=player, through_step=through_step)
    _require(_wire(state.record_fields()) == _wire(prediction.record_fields()),
             'PUBLIC_UPDATE_INITIALIZATION_RECOMPUTE')
    return updater, state, initialization
