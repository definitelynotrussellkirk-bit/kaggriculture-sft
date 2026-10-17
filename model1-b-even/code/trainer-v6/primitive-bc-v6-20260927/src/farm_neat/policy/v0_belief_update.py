"""Bounded online Bayes over a declared catalog of PUBLIC flow-sign models.

Usage::

    updater = BayesianFlowUpdater(product, hypothesis_ids, regime_laws)
    state = updater.initialize(player=seat, through_step=opening_step)
    result = updater.update(state, opponent_model.observe(actor_context))
    state = result.state
    saved = state.to_dict()                 # JSON, not pickle
    state = updater.restore_state(saved)    # exact numerical continuation

The existing strategy.opponent.OpponentModel owns observation projection and
passive-product certification. This module consumes its typed answer, NEVER a
raw evaluator/trainer dictionary. It reads no policy_family, rival rank, hidden
identity, outcome, future shops or arbitrary callback. A product is fixed BEFORE
observations; do not multiply separate product posteriors as if independent.

Each public model declares P(sign(E_eff) | ordered published shops, demand,
model). Columns are negative/zero/positive. E_eff is historical displacement,
NOT requested trades, gross fills, rival cash or policy identity. Zero does not
identify PASS. Amounts beyond sign, other products, farm state, inventory level,
within-queue order and longer temporal dependence are excluded from this model.
The conditional-independence/stationarity assumption is explicit, not established
for real opponents. Caller-supplied likelihoods and prior provenance are NOT
certified by their hashes; indistinguishable hypotheses are neither merged nor
pruned. UNKNOWN is an optional declared expert, not calibrated missing-family mass.

Update uses cumulative public log likelihood plus the ORIGINAL transformed prior
through v0_belief.belief_masses. Output floors and posterior odds caps are never
recycled. Probabilities must have positive representable support; incompatible
zero laws/underflow refuse atomically, with no epsilon or hidden forgetting.
Unsupported, missing, confounded or noncontiguous intervals contribute NO evidence.
All rows must share a supported regime; missing laws are not renormalized away.

The watermark deduplicates rolling histories. Gaps not present in the observer's
bounded history are counted as uncovered steps, never invented observations.
Repeated identical current observations are idempotent; stale/conflicting times,
seat changes, overlapping intervals and model/config changes refuse. Reset BOTH
this state and OpponentModel's legitimate own memory at a new game. No secret
world/game key is needed or inferred. The hash chain binds delivered public
observation boundaries; batch delivery has the same numerical posterior as
incremental delivery but not necessarily the same boundary hash.

record_fields() is the accepted head's numerical export for a NEW sidecar; it is
not a sealed BeliefRecord. Join by public hypothesis hash, retain model/evidence
provenance and nullable WDL conditionals, and call v0_belief_records.make_record.
Do not widen the opening-only regret bridge or confuse this state with compiler
product-flow .belief. This module chooses no actions and changes no V0 semantics.
The consumer owns EXTERNAL snapshot_state/restore_state wiring and own-prefix
integration. Frozen bundles enter through
training.v0_belief_priors.initialize_public_flow_belief; that bridge starts with ZERO current-game evidence, never prior-seeded model scores.
contribution_diagnostics(state) exports additive prior/public log-odds in a detached
sidecar; it changes neither the wire-v1 state nor actor history. No universal cap
is selected here. Prior influence is a finite initial log-odds budget, not an
observation count. A sufficiently diagnostic cumulative Bayes factor can exceed
it; aliases, omitted truth, stale evidence after drift and float64 underflow remain
explicit limitations. prior_strength=0 disables the raw prior and canonicalizes
all its options to the exact existing uniform state/configuration identity.

Fixed prior/table, up to 64 hypotheses, 256 regimes, O(K*new intervals)
work (plus bounded history validation) and O(K) persistent state; no native games,
training, GPU, filesystem/network reads, likelihood fitting or recurrence search.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Sequence

import torch

from farm_neat.game.constants import MAX_SHOP_INSTANCES, PRODUCTS, SHOPS
from farm_neat.policy.v0_belief import belief_masses
from farm_neat.strategy.opponent import FlowSample, OpponentState
from farm_neat.strategy.types import Evidence, LAST_STEP

SCHEMA = 'farm-v0-public-flow-bayes-state-v1'
MODEL_SCHEMA = 'farm-v0-public-flow-sign-likelihood-v1'
_HASH = re.compile(r'[0-9a-f]{64}\Z')
_STATE_FIELDS = frozenset((
    'schema', 'updater_sha256', 'player', 'origin_step', 'through_step',
    'observed_intervals', 'scored_intervals', 'skipped_intervals', 'uncovered_steps',
    'predictive_log_score', 'history_sha256', 'public_fingerprint', 'belief',
    'content_sha256'))


def _wire(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def content_sha256(value) -> str:
    """Hash the entire finite JSON value. Establishes identity, not authenticity."""
    return hashlib.sha256(_wire(value).encode()).hexdigest()


def _integer(value, name, low=0, high=LAST_STEP):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric data')


def _hash(value, name):
    if type(value) is not str or not _HASH.fullmatch(value):
        raise ValueError(f'{name} must be a lowercase SHA256')


def _regime(shops, demand):
    shops, demand = tuple(shops), tuple(demand)
    if (len(shops) > MAX_SHOP_INSTANCES
            or any(type(s) is not str or s not in SHOPS for s in shops)):
        raise ValueError('invalid ordered public shop regime')
    if (len(demand) != len(PRODUCTS)
            or any(type(d) is not int or not 0 <= d <= 2*MAX_SHOP_INSTANCES+1 for d in demand)):
        raise ValueError('invalid public demand vector')
    return shops, demand


@dataclass(frozen=True)
class FlowRegimeLaw:
    """One public regime, K rows, three strictly positive sign probabilities.

    Row order is exactly the updater's hypothesis_ids order. Laws are declarations,
    not fitted or calibrated here. A full missing regime is neutral; a partial K
    table is invalid. All collections are defensively frozen.
    """
    shops: tuple[str, ...]
    demand: tuple[int, ...]
    probabilities: tuple[tuple[float, float, float], ...]

    def __post_init__(self):
        shops, demand = _regime(self.shops, self.demand)
        rows = tuple(tuple(row) for row in self.probabilities)
        if not 1 <= len(rows) <= 64:
            raise ValueError('law needs 1..64 hypothesis rows')
        for row in rows:
            if len(row) != 3:
                raise ValueError('law needs negative/zero/positive columns')
            for value in row:
                _number(value, 'likelihood')
                if not 0 < value <= 1:
                    raise ValueError('likelihood must be positive; no epsilon repair')
            if abs(math.fsum(row)-1.) > 1e-12:
                raise ValueError('likelihood row must sum to one within 1e-12')
        object.__setattr__(self, 'shops', shops)
        object.__setattr__(self, 'demand', demand)
        object.__setattr__(self, 'probabilities', tuple(tuple(float(x) for x in r) for r in rows))

    def to_dict(self) -> dict:
        return dict(shops=list(self.shops), demand=list(self.demand),
                    probabilities=[list(row) for row in self.probabilities])


@dataclass(frozen=True)
class OnlineBeliefState:
    """Immutable JSON capsule. Only updater.initialize/restore_state seal it.

    update revalidates even a caller-constructed capsule. Returned dictionaries
    are detached; state contains no mutable tensor or hidden recurrent object.
    """
    _json: str

    def to_dict(self) -> dict:
        return json.loads(self._json)

    def record_fields(self) -> dict:
        return self.to_dict()['belief']

    @property
    def posterior_mass(self) -> tuple[float, ...]:
        return tuple(r['posterior_mass'] for r in self.record_fields()['hypotheses'])

    @property
    def constrained_mass(self) -> tuple[float, ...]:
        return tuple(r['constrained_mass'] for r in self.record_fields()['hypotheses'])

    @property
    def calibration_status(self) -> str:
        return 'UNMEASURED'


@dataclass(frozen=True)
class BeliefUpdate:
    state: OnlineBeliefState
    events: tuple[dict, ...]  # detached audit output, NOT recurrent state


@dataclass(frozen=True, init=False)
class BayesianFlowUpdater:
    """Fixed public likelihood catalog. No call-time model selection or callbacks.

    raw_prior and its transform/floor options use the accepted head semantics.
    Public model hashes are associations, not numerical features or truth labels.
    Keep the table independent of which private family the evaluator selected.
    """
    product: str
    hypothesis_ids: tuple[str, ...]
    _laws: object
    _options_json: str
    updater_sha256: str

    def __init__(self, product: str, hypothesis_ids: Sequence[str],
                 regime_laws: Sequence[FlowRegimeLaw], *, kinds=None,
                 raw_prior=None, prior_strength=1., prior_odds_cap=None,
                 prior_source_sha256=None, p_min=0., floor_transform='none'):
        if product not in PRODUCTS:
            raise ValueError('product must be a public market product')
        if isinstance(hypothesis_ids, (str, bytes)):
            raise TypeError('hypothesis_ids must be a sequence')
        keys = tuple(hypothesis_ids)
        _number(prior_strength, 'prior_strength')
        if prior_strength < 0:
            raise ValueError('prior_strength must be nonnegative')
        if prior_strength == 0:
            # Explicit OFF switch, not a new transform or a zero-power posterior.
            # Canonicalize before serialization/head calls so even identity and
            # replay bytes match the pre-existing no-prior path exactly.
            raw_prior, prior_source_sha256, prior_odds_cap = None, None, None
            prior_strength = 1.
        options = dict(kinds=None if kinds is None else list(kinds),
                       raw_prior=None if raw_prior is None else list(raw_prior),
                       prior_strength=prior_strength, prior_odds_cap=prior_odds_cap,
                       prior_source_sha256=prior_source_sha256, p_min=p_min,
                       floor_transform=floor_transform)
        if raw_prior is not None:
            for value in options['raw_prior']:
                _number(value, 'raw_prior')
        object.__setattr__(self, 'product', product)
        object.__setattr__(self, 'hypothesis_ids', keys)
        object.__setattr__(self, '_options_json', _wire(options))
        # Reuse the existing owner for catalog, prior, positive support and floors.
        initial = self._head([0.]*len(keys))
        laws = tuple(regime_laws)
        if not 1 <= len(laws) <= 256:
            raise ValueError('provide 1..256 public regime laws')
        table = {}
        for law in laws:
            if type(law) is not FlowRegimeLaw or len(law.probabilities) != len(keys):
                raise ValueError('each regime must cover all hypothesis rows')
            key = law.shops, law.demand
            if key in table:
                raise ValueError('duplicate public regime')
            table[key] = law
        object.__setattr__(self, '_laws', MappingProxyType(table))
        identity = dict(schema=MODEL_SCHEMA, product=product,
                        hypothesis_ids=list(keys), initial=initial.record_fields(),
                        options=options, regimes=[table[key].to_dict() for key in sorted(table)])
        object.__setattr__(self, 'updater_sha256', content_sha256(identity))

    def _head(self, log_evidence):
        options = json.loads(self._options_json)
        if options['raw_prior'] is not None:
            options['raw_prior'] = torch.tensor(options['raw_prior'], dtype=torch.float64, device='cpu')
        with torch.inference_mode():
            return belief_masses(torch.tensor(log_evidence, dtype=torch.float64, device='cpu'),
                                 hypothesis_ids=self.hypothesis_ids, **options)

    @staticmethod
    def _seal(data):
        body = {k:v for k,v in data.items() if k != 'content_sha256'}
        return OnlineBeliefState(_wire(dict(body, content_sha256=content_sha256(body))))

    def initialize(self, *, player: int, through_step: int = 0) -> OnlineBeliefState:
        """Start before consuming any intervals, at an explicit actor observation step."""
        _integer(player, 'player', high=1)
        _integer(through_step, 'through_step')
        origin = dict(schema=SCHEMA, updater_sha256=self.updater_sha256,
                      player=player, origin_step=through_step)
        return self._seal(dict(origin, through_step=through_step, observed_intervals=0,
            scored_intervals=0, skipped_intervals=0, uncovered_steps=0,
            predictive_log_score=0., history_sha256=content_sha256(origin),
            public_fingerprint=None, belief=self._head([0.]*len(self.hypothesis_ids)).record_fields()))

    def restore_state(self, document: dict) -> OnlineBeliefState:
        """Validate closed JSON, config binding, counters and accepted-head arithmetic.

        Hashes/numerical consistency cannot authenticate external chronology or
        prior provenance. Save/restore both this capsule and the observer's memory.
        """
        if type(document) is not dict or set(document) != _STATE_FIELDS:
            raise ValueError('state must contain exactly the closed public state fields')
        text = _wire(document)
        if len(text) > 131072:
            raise ValueError('state exceeds bounded JSON size')
        data = json.loads(text)
        if data['schema'] != SCHEMA or data['updater_sha256'] != self.updater_sha256:
            raise ValueError('state schema/updater identity mismatch')
        _hash(data['content_sha256'], 'content_sha256')
        if data['content_sha256'] != content_sha256({k:v for k,v in data.items() if k != 'content_sha256'}):
            raise ValueError('state content hash mismatch')
        _hash(data['history_sha256'], 'history_sha256')
        if data['public_fingerprint'] is not None:
            _hash(data['public_fingerprint'], 'public_fingerprint')
        _integer(data['player'], 'player', high=1)
        for name in ('origin_step', 'through_step', 'observed_intervals', 'scored_intervals',
                     'skipped_intervals', 'uncovered_steps'):
            _integer(data[name], name)
        elapsed = data['through_step'] - data['origin_step']
        if (elapsed < 0 or data['scored_intervals'] + data['skipped_intervals'] != data['observed_intervals']
                or data['observed_intervals'] + data['uncovered_steps'] > elapsed):
            raise ValueError('state interval counters disagree with its time span')
        _number(data['predictive_log_score'], 'predictive_log_score')
        if data['predictive_log_score'] > 1e-12:
            raise ValueError('predictive log score cannot be positive')
        belief = data['belief']
        if (type(belief) is not dict or type(belief.get('hypotheses')) is not list
                or len(belief['hypotheses']) != len(self.hypothesis_ids)):
            raise ValueError('state belief rows do not match catalog')
        evidence = []
        for row in belief['hypotheses']:
            if type(row) is not dict or 'log_evidence' not in row:
                raise ValueError('invalid state belief row')
            value = row['log_evidence']
            _number(value, 'log_evidence')
            evidence.append(value)
        if max(evidence) != 0.:
            raise ValueError('cumulative relative log_evidence must have maximum zero')
        expected = self._head(evidence).record_fields()
        # Canonical JSON equality also rejects bool-for-number and extra-key tricks.
        if _wire(belief) != _wire(expected):
            raise ValueError('state belief numerical fields disagree with accepted head')
        return OnlineBeliefState(text)

    def contribution_diagnostics(self, state: OnlineBeliefState, *,
                                 reference_hypothesis_id: str | None = None) -> dict:
        """Audit fixed prior vs authentic accumulated evidence, in natural-log units.

        For each h and reference r, posterior log odds are exactly the model's
        log(pi_h/pi_r) + L_h - L_r (up to arithmetic rounding). These are NOT
        additive probability shares or an estimate of independent sample size.
        The prior span is bounded by the configured cap when present; the public
        term is not capped or attenuated. Output floors are reported separately.
        This validated, detached sidecar adds nothing to recurrent/wire-v1 state.
        """
        if type(state) is not OnlineBeliefState:
            raise TypeError('diagnostics require OnlineBeliefState')
        data = self.restore_state(state.to_dict()).to_dict()
        reference = (self.hypothesis_ids[0] if reference_hypothesis_id is None
                     else reference_hypothesis_id)
        if reference not in self.hypothesis_ids:
            raise ValueError('reference_hypothesis_id must be in the live catalog')
        rows = data['belief']['hypotheses']
        log_prior = [math.log(row['initial_mass']) for row in rows]
        evidence = [row['log_evidence'] for row in rows]
        ref = self.hypothesis_ids.index(reference)
        prior = data['belief']['prior']
        result = dict(schema='farm-v0-public-flow-prior-diagnostics-v1',
            state_sha256=data['content_sha256'], updater_sha256=self.updater_sha256,
            reference_hypothesis_id=reference, through_step=data['through_step'],
            scored_intervals=data['scored_intervals'], prior=prior,
            mechanism='FIXED_INITIAL_LOG_ODDS', prior_coefficient=1.,
            prior_log_odds_span_nats=max(log_prior)-min(log_prior),
            configured_log_odds_cap_nats=(None if prior['odds_cap'] is None
                                          else math.log(prior['odds_cap'])),
            public_log_evidence_span_nats=max(evidence)-min(evidence),
            hypotheses=[], calibration_status='UNMEASURED')
        for row, lp, ll in zip(rows, log_prior, evidence):
            prior_odds, public_odds = lp-log_prior[ref], ll-evidence[ref]
            result['hypotheses'].append(dict(
                hypothesis_id=row['hypothesis_id'], kind=row['kind'],
                initial_mass=row['initial_mass'], log_prior_contribution=lp,
                log_evidence=ll, prior_log_odds_vs_reference=prior_odds,
                public_log_bayes_factor_vs_reference=public_odds,
                posterior_log_odds_vs_reference=prior_odds+public_odds,
                posterior_mass=row['posterior_mass'], constrained_mass=row['constrained_mass']))
        result['content_sha256'] = content_sha256(result)
        return result

    def _project(self, row: FlowSample, certified_calendar: bool) -> dict:
        """Closed public projection. Do not copy arbitrary typed-answer annotations."""
        regime = None if row.shops is None or None in row.demand_vector else _regime(row.shops, row.demand_vector)
        sign = None
        if row.end_step != row.start_step+1:
            status = 'NONCONTIGUOUS_OBSERVATIONS'
        elif not certified_calendar:
            status = 'CALENDAR_UNCERTIFIED'
        elif regime is None:
            status = 'PUBLISHED_PREFIX_UNCERTIFIED'
        elif row.own_order_ranks is None:
            status = 'OWN_ACTION_UNCERTIFIED'
        elif row.own_order_ranks:
            status = 'OWN_SAME_PRODUCT_REQUEST'
        elif row.effect.kind is not Evidence.EXACT or row.reason is not None:
            # Closed reason tags, not evaluator-provided arbitrary text.
            status = (row.reason if row.reason in ('MISSING_PUBLIC_INVENTORY',
                      'EFFECT_NOT_EXACTLY_REPRESENTABLE') else 'UNCERTIFIED_FLOW')
        else:
            value = row.effect.value
            _number(value, 'exact flow effect')
            if int(value) != value:
                raise ValueError('exact flow effect must be integral')
            sign = (value > 0) - (value < 0)
            status = 'SCORED' if regime in self._laws else 'UNSUPPORTED_REGIME'
        return dict(start_step=row.start_step, end_step=row.end_step, product=self.product,
                    shops=None if regime is None else list(regime[0]),
                    demand=None if regime is None else list(regime[1]), sign=sign,
                    status=status, log_likelihoods=None, predictive_log_likelihood=None)

    def update(self, previous: OnlineBeliefState, observation: OpponentState) -> BeliefUpdate:
        """Consume newly arrived public intervals once, returning a NEW immutable state.

        The existing posterior is the next predictive mixture; raw cumulative
        likelihood is retained against the original prior. All work is local so
        any refusal (including posterior underflow) leaves previous unchanged.
        """
        if type(previous) is not OnlineBeliefState or type(observation) is not OpponentState:
            raise TypeError('update requires OnlineBeliefState and typed public OpponentState')
        data = self.restore_state(previous.to_dict()).to_dict()
        now, seat = observation.event.step, observation.event.player
        _integer(now, 'observation step')
        _integer(seat, 'observation player', high=1)
        if type(observation.default_calendar) is not bool:
            raise ValueError('default_calendar must be a bool')
        _hash(observation.public_fingerprint, 'public_fingerprint')
        if seat != data['player']:
            raise ValueError('opponent observation seat mismatch')
        if now < data['through_step']:
            raise ValueError('stale public observation')
        if now == data['through_step']:
            if data['public_fingerprint'] not in (None, observation.public_fingerprint):
                raise ValueError('conflicting public observation at same step')
            if data['public_fingerprint'] is not None:
                return BeliefUpdate(previous, ())
        if len(observation.history) > LAST_STEP * len(PRODUCTS):
            raise ValueError('public history exceeds bounded observer support')
        rows = []
        for row in observation.history:
            if type(row) is not FlowSample:
                raise TypeError('public history requires typed FlowSample rows')
            if row.product != self.product:
                continue
            _integer(row.start_step, 'interval start')
            _integer(row.end_step, 'interval end')
            if row.end_step > now or row.end_step <= row.start_step:
                raise ValueError('invalid/future public interval')
            if row.end_step > data['through_step']:
                rows.append(row)
        rows.sort(key=lambda r: (r.start_step, r.end_step))
        cursor, events = data['through_step'], []
        for row in rows:
            if row.start_step < cursor:
                raise ValueError('overlap/duplicate public intervals or watermark straddle')
            data['uncovered_steps'] += row.start_step-cursor
            cursor = row.end_step
            event = self._project(row, observation.default_calendar)
            data['observed_intervals'] += 1
            if event['status'] == 'SCORED':
                law = self._laws[(tuple(event['shops']), tuple(event['demand']))]
                increments = [math.log(p[event['sign']+1]) for p in law.probabilities]
                old = data['belief']['hypotheses']
                terms = [math.log(r['posterior_mass']) + ll for r,ll in zip(old, increments)]
                maximum = max(terms)
                predictive = maximum + math.log(math.fsum(math.exp(t-maximum) for t in terms))
                accumulated = [r['log_evidence'] + ll for r,ll in zip(old, increments)]
                data['belief'] = self._head(accumulated).record_fields()
                data['scored_intervals'] += 1
                data['predictive_log_score'] += predictive
                event['log_likelihoods'] = increments
                event['predictive_log_likelihood'] = predictive
            else:
                data['skipped_intervals'] += 1
            events.append(event)
        data['uncovered_steps'] += now-cursor
        data['through_step'] = now
        data['public_fingerprint'] = observation.public_fingerprint
        data['history_sha256'] = content_sha256(dict(previous=data['history_sha256'],
            through_step=now, public_fingerprint=observation.public_fingerprint, events=events))
        return BeliefUpdate(self._seal(data), tuple(events))
