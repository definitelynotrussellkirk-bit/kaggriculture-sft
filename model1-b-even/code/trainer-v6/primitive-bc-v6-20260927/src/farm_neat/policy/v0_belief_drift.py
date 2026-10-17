"""Public-sign drift evidence, not a production policy or opponent identity model.

The predictive alternative Q is a normalized fixed-share mixture of the frozen
public catalog and a past-only discounted-Dirichlet residual. Alarm on
log Q(x_1:t) - max_k log P_k(x_1:t) >= log(1/alpha_epoch). Under ANY fixed
catalog law, Q/P_k is a nonnegative likelihood-ratio martingale; the statistic is
bounded above by it. Alpha spending across detector restarts bounds the chance
of any false alarm, conditional on the declared laws being correct on scored
observations. The statistic itself need not be a martingale or drift probability.

A trigger restarts ONLY the detector likelihood epoch. Adaptive prediction keeps
its learned public state. Fallback is a measurable latch, not an action override:
its consumer still owns the V0 plan and the normal publication boundary. Nothing
here observes a private identity/change point, future draw or terminal outcome.
A residual weight is predictive responsibility, not probability of a new family.

The core expects certified signs; public_adapter.py owns the narrow bridge from
the accepted public-flow updater. Config tables must be fixed before observations.
Missing/scoring selection must itself satisfy the null's conditional-law claim.
All arithmetic is log-domain for likelihoods, O(K) per scored sign, bounded by
719 joint turns. This module uses only the Python standard library.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
import re
from typing import Any

SCHEMA = 'farm-v0-public-drift-state-v1'
CONFIG_SCHEMA = 'farm-v0-public-drift-config-v1'
_HASH = re.compile(r'[0-9a-f]{64}\Z')


def wire(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(wire(value).encode()).hexdigest()


def _num(x: Any, name: str) -> float:
    if type(x) not in (float, int) or not math.isfinite(x):
        raise ValueError(f'{name} must be finite numeric data')
    return float(x)


def _int(x: Any, name: str, low: int = 0, high: int = 719) -> int:
    if type(x) is not int or not low <= x <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return x


def _hash(x: Any, name: str) -> str:
    if type(x) is not str or _HASH.fullmatch(x) is None:
        raise ValueError(f'{name} must be lowercase SHA256')
    return x


def _probabilities(xs, name: str, size: int, *, positive: bool = False) -> tuple[float, ...]:
    if not isinstance(xs, (tuple, list)) or len(xs) != size:
        raise ValueError(f'{name} has wrong size')
    out = tuple(_num(x, name) for x in xs)
    if any(x < 0 or x > 1 or (positive and x == 0) for x in out):
        raise ValueError(f'{name} must have valid probability support')
    if abs(math.fsum(out)-1.) > 1e-12:
        raise ValueError(f'{name} must sum to one')
    return out


@dataclass(frozen=True)
class DriftConfig:
    hypothesis_ids: tuple[str, ...]
    # Frozen (public regime hash, K by 3 sign-likelihood table) rows.
    regimes: tuple[tuple[str, tuple[tuple[float, ...], ...]], ...]
    false_alarm_budget: float = .05
    hazard: float = .04
    residual_mass: float = .1
    residual_decay: float = .03
    residual_pseudocount: float = 1.
    hold_scored_observations: int = 12

    def __post_init__(self):
        if not isinstance(self.hypothesis_ids, (tuple,list)):
            raise ValueError('hypothesis_ids must be a sequence')
        ids = tuple(_hash(x, 'hypothesis id') for x in self.hypothesis_ids)
        if not 1 <= len(ids) <= 64 or len(set(ids)) != len(ids):
            raise ValueError('need 1..64 unique public hypothesis ids')
        if not isinstance(self.regimes, (tuple,list)) or not 1 <= len(self.regimes) <= 256:
            raise ValueError('need 1..256 frozen public regimes')
        regimes = []
        for entry in self.regimes:
            if not isinstance(entry,(tuple,list)) or len(entry) != 2:
                raise ValueError('invalid regime entry')
            key, table = entry
            _hash(key, 'regime')
            if not isinstance(table,(tuple,list)) or len(table) != len(ids):
                raise ValueError('regime table does not match the catalog')
            rows = tuple(_probabilities(row, 'likelihood', 3, positive=True) for row in table)
            regimes.append((key, rows))
        if len({key for key,_ in regimes}) != len(regimes):
            raise ValueError('duplicate regime')
        object.__setattr__(self, 'hypothesis_ids', ids)
        object.__setattr__(self, 'regimes', tuple(sorted(regimes)))
        for name in ('false_alarm_budget','hazard','residual_mass','residual_decay'):
            x = _num(getattr(self,name), name)
            if not 0 < x < 1:
                raise ValueError(f'{name} must be strictly between zero and one')
            object.__setattr__(self,name,x)
        x = _num(self.residual_pseudocount, 'residual_pseudocount')
        if not 1e-6 <= x <= 1e6:
            raise ValueError('residual_pseudocount outside bounded support')
        object.__setattr__(self,'residual_pseudocount',x)
        _int(self.hold_scored_observations,'hold_scored_observations',1)

    def to_dict(self) -> dict:
        return json.loads(wire(dict(schema=CONFIG_SCHEMA, **asdict(self))))

    @classmethod
    def from_dict(cls, doc: dict) -> 'DriftConfig':
        expected = {f.name for f in fields(cls)} | {'schema'}
        if type(doc) is not dict or set(doc) != expected or doc['schema'] != CONFIG_SCHEMA:
            raise ValueError('invalid config fields or schema')
        return cls(**{k:v for k,v in doc.items() if k != 'schema'})


@dataclass(frozen=True)
class DriftState:
    monitor_sha256: str
    origin_step: int
    through_step: int
    scored: int
    unscored_steps: int
    detector_scored: int
    null_log_likelihood: tuple[float, ...]
    alternative_log_likelihood: float
    adaptive_weights: tuple[float, ...]
    residual_counts: tuple[float, float, float]
    alarm_count: int
    last_alarm_step: int | None
    cooldown_remaining: int
    last_alarm: bool
    last_log_evidence: float | None
    last_threshold: float | None
    last_log_predictive: float | None
    history_sha256: str


class DriftMonitor:
    def __init__(self, config: DriftConfig):
        if type(config) is not DriftConfig:
            raise TypeError('requires immutable DriftConfig')
        self.config = config
        self.monitor_sha256 = content_hash(config.to_dict())
        self._tables = dict(config.regimes)
        self._k = len(config.hypothesis_ids)
        self._prior = ((1.-config.residual_mass)/self._k,)*self._k + (config.residual_mass,)

    def epoch_alpha(self, epoch: int) -> float:
        _int(epoch,'epoch',0,1000000)
        return self.config.false_alarm_budget / ((epoch+1)*(epoch+2))

    def initialize(self, *, through_step: int = 0) -> DriftState:
        _int(through_step,'through_step')
        return DriftState(self.monitor_sha256, through_step, through_step, 0, 0, 0,
            (0.,)*self._k, 0., self._prior, (0.,0.,0.), 0, None, 0, False,
            None,None,None,content_hash({'monitor':self.monitor_sha256,'origin':through_step}))

    def _check(self, s: DriftState) -> None:
        if type(s) is not DriftState:
            raise TypeError('requires immutable DriftState')
        if s.monitor_sha256 != self.monitor_sha256:
            raise ValueError('monitor identity mismatch')
        for name in ('origin_step','through_step','scored','unscored_steps',
                     'detector_scored','alarm_count','cooldown_remaining'):
            _int(getattr(s,name),name)
        if (s.origin_step > s.through_step or s.scored+s.unscored_steps != s.through_step-s.origin_step
                or s.alarm_count > s.scored or s.detector_scored > s.scored
                or s.cooldown_remaining > self.config.hold_scored_observations):
            raise ValueError('state counters inconsistent')
        _probabilities(s.adaptive_weights,'adaptive_weights',self._k+1,positive=True)
        if type(s.null_log_likelihood) is not tuple or len(s.null_log_likelihood) != self._k:
            raise ValueError('invalid null likelihood vector')
        for x in (*s.null_log_likelihood, s.alternative_log_likelihood):
            if _num(x,'log likelihood') > 1e-12:
                raise ValueError('log likelihood cannot be positive')
        if type(s.residual_counts) is not tuple or len(s.residual_counts) != 3:
            raise ValueError('invalid residual counts')
        for x in s.residual_counts:
            if not 0 <= _num(x,'residual counts') <= 1./self.config.residual_decay+1e-8:
                raise ValueError('residual counts out of bounds')
        if type(s.last_alarm) is not bool:
            raise ValueError('last_alarm must be bool')
        if s.last_alarm_step is None:
            if s.alarm_count or s.cooldown_remaining or s.last_alarm:
                raise ValueError('missing alarm step')
        else:
            _int(s.last_alarm_step,'last_alarm_step',s.origin_step+1,s.through_step)
            if not s.alarm_count or (s.last_alarm and s.last_alarm_step != s.through_step):
                raise ValueError('inconsistent alarm metadata')
        for name in ('last_log_evidence','last_threshold','last_log_predictive'):
            x = getattr(s,name)
            if x is not None: _num(x,name)
        if s.scored == 0 and any(getattr(s,n) is not None for n in (
                'last_log_evidence','last_threshold','last_log_predictive')):
            raise ValueError('unscored state has likelihood evidence')
        if s.last_log_predictive is not None and s.last_log_predictive > 1e-12:
            raise ValueError('predictive log probability cannot be positive')
        if s.detector_scored == 0 and (s.alternative_log_likelihood != 0.
                or any(s.null_log_likelihood)):
            raise ValueError('empty detector epoch has nonzero likelihood')
        _hash(s.history_sha256,'history_sha256')

    def predictive_weights(self, s: DriftState) -> tuple[float, ...]:
        self._check(s)
        h = self.config.hazard
        return tuple((1.-h)*w+h*p for w,p in zip(s.adaptive_weights,self._prior))

    def residual_prediction(self, s: DriftState) -> tuple[float, ...]:
        self._check(s)
        eta = self.config.residual_pseudocount
        den = math.fsum(s.residual_counts)+3*eta
        return tuple((c+eta)/den for c in s.residual_counts)

    def forecast(self, s: DriftState, regime: str) -> tuple[float, ...]:
        self._check(s)
        if regime not in self._tables:
            raise ValueError('unsupported public regime')
        weights = self.predictive_weights(s)
        table = self._tables[regime] + (self.residual_prediction(s),)
        return tuple(math.fsum(w*r[j] for w,r in zip(weights,table)) for j in range(3))

    def step(self, s: DriftState, *, through_step: int, sign: int | None,
             regime: str | None = None) -> DriftState:
        """Consume one certified sign at its endpoint, or a neutral watermark.

        A gap counts as unscored time. It cannot consume the fallback hold or
        supply invented observations. Immutable input stays intact on any refusal.
        Decisions for this interval MUST have used the previous state's forecast.
        """
        self._check(s)
        _int(through_step,'through_step',s.through_step+1)
        gap = through_step-s.through_step
        if sign is None:
            if regime is not None:
                raise ValueError('neutral observation has no scored regime')
            return replace(s,through_step=through_step,unscored_steps=s.unscored_steps+gap,
                last_alarm=False,history_sha256=content_hash([s.history_sha256,through_step,None]))
        if type(sign) is not int or sign not in (-1,0,1):
            raise ValueError('sign must be certified integer -1, 0, or 1')
        if regime not in self._tables:
            raise ValueError('unsupported public regime')
        table = self._tables[regime] + (self.residual_prediction(s),)
        weights = self.predictive_weights(s)
        j = sign+1
        terms = tuple(w*row[j] for w,row in zip(weights,table))
        q = math.fsum(terms)
        if not 0 < q <= 1.:
            raise ValueError('nonrepresentable predictive probability; no epsilon repair')
        posterior = tuple(v/q for v in terms)
        _probabilities(posterior,'updated weights',self._k+1,positive=True)
        ll = tuple(old+math.log(row[j]) for old,row in zip(s.null_log_likelihood,table[:-1]))
        logq = math.log(q)
        alt = s.alternative_log_likelihood+logq
        loge = alt-max(ll)
        threshold = -math.log(self.epoch_alpha(s.alarm_count))
        alarm = loge >= threshold
        counts = tuple((1.-self.config.residual_decay)*c+int(z==j)
                       for z,c in enumerate(s.residual_counts))
        nxt = DriftState(self.monitor_sha256,s.origin_step,through_step,s.scored+1,
            s.unscored_steps+gap-1,0 if alarm else s.detector_scored+1,
            (0.,)*self._k if alarm else ll,0. if alarm else alt,posterior,counts,
            s.alarm_count+int(alarm),through_step if alarm else s.last_alarm_step,
            self.config.hold_scored_observations if alarm else max(0,s.cooldown_remaining-1),
            alarm,loge,threshold,logq,
            content_hash([s.history_sha256,through_step,sign,regime]))
        self._check(nxt)
        return nxt

    def export_state(self, s: DriftState) -> dict:
        self._check(s)
        body = dict(schema=SCHEMA,**asdict(s))
        return json.loads(wire(dict(body,content_sha256=content_hash(body))))

    def restore_state(self, doc: dict) -> DriftState:
        expected = {f.name for f in fields(DriftState)} | {'schema','content_sha256'}
        if type(doc) is not dict or set(doc) != expected:
            raise ValueError('state must have exactly the closed fields')
        if len(wire(doc)) > 65536:
            raise ValueError('state exceeds bounded JSON size')
        body = {k:v for k,v in doc.items() if k != 'content_sha256'}
        if body['schema'] != SCHEMA or doc['content_sha256'] != content_hash(body):
            raise ValueError('state schema/content hash mismatch')
        data = {k:v for k,v in body.items() if k != 'schema'}
        for name in ('null_log_likelihood','adaptive_weights','residual_counts'):
            if type(data[name]) is not list:
                raise ValueError('serialized vectors must be JSON lists')
            data[name] = tuple(data[name])
        s = DriftState(**data)
        self._check(s)
        return s

    def evidence(self, s: DriftState) -> dict:
        """Closed actor-safe evidence for a future meta-selector, not a sidecar.

        No V0 plan is selected/published here. Confidence/fallback indicators do
        not certify a worst-case value or a calibrated terminal winning probability.
        """
        self._check(s)
        return dict(schema='farm-v0-public-drift-evidence-v1',
            monitor_sha256=self.monitor_sha256,through_step=s.through_step,
            scored_observations=s.scored,unscored_steps=s.unscored_steps,
            alarm=s.last_alarm,alarm_count=s.alarm_count,last_alarm_step=s.last_alarm_step,
            log_evidence=s.last_log_evidence,log_alarm_threshold=s.last_threshold,
            next_epoch_alpha=self.epoch_alpha(s.alarm_count),
            fallback_required=s.cooldown_remaining>0,
            fallback_scored_observations_remaining=s.cooldown_remaining,
            hypothesis_ids=list(self.config.hypothesis_ids),
            adaptive_predictive_weights=list(self.predictive_weights(s)),
            residual_predictive_probabilities=list(self.residual_prediction(s)),
            residual_semantics='PREDICTIVE_RESPONSIBILITY_NOT_FAMILY_PROBABILITY',
            null_scope='FIXED_DECLARED_PUBLIC_LAW',calibration_status='UNMEASURED',
            history_sha256=s.history_sha256)


# The preceding 16,110 bytes are the retained drift.py VERBATIM, SHA256
# eea6a94a7de3440b44794031e626efd61bee672a2dde506572f22c5133dc285b.
# The historical public_adapter.py is absent. Everything BELOW is a new,
# explicitly versioned source bridge, not a recovery of that missing artifact.
BRIDGE_SCHEMA = 'farm-v0-public-belief-drift-bridge-v1'
RETAINED_MONITOR_SHA256 = 'eea6a94a7de3440b44794031e626efd61bee672a2dde506572f22c5133dc285b'


@dataclass(frozen=True)
class PublicBeliefState:
    """Immutable combined updater/monitor/root-selector capsule; detached JSON only.

    This is NOT the complete actor: OpponentModel's legitimate own-action memory,
    compiler/controller continuation and caller RNG remain owned by the caller.
    Restore those as well before asking the observer for its next typed answer.
    """
    _json: str

    def to_dict(self) -> dict:
        return json.loads(self._json)


def public_observation_sha256(observation, *, product: str) -> str:
    """Digest the closed projection actually consumed, not private annotations.

    The typed answer comes from the accepted OpponentModel. Its public_fingerprint
    binds CURRENT inventory, not all historical rows; this additional digest can
    bind a delivery to a separately retained source receipt. A digest is identity,
    NOT authentication of the producer, the conditional law or the own memory.
    """
    from farm_neat.game.constants import PRODUCTS
    from farm_neat.strategy.opponent import FlowSample, OpponentState
    if type(observation) is not OpponentState or product not in PRODUCTS:
        raise TypeError('typed public OpponentState and public product required')
    if len(observation.history) > 718 * len(PRODUCTS):
        raise ValueError('public history exceeds bounded support')
    rows = []
    for row in observation.history:
        if type(row) is not FlowSample:
            raise TypeError('typed public FlowSample required')
        if row.product == product:
            rows.append(dict(start=row.start_step, end=row.end_step,
                effect=row.effect.to_dict(), demand=row.demand.to_dict(),
                ranks=row.own_order_ranks, reason=row.reason, shops=row.shops,
                demand_vector=row.demand_vector))
    return content_hash(dict(schema=BRIDGE_SCHEMA+'-observation', product=product,
        step=observation.event.step, player=observation.event.player,
        public_fingerprint=observation.public_fingerprint,
        default_calendar=observation.default_calendar, history=rows))


class PublicBeliefBridge:
    """Default-off composition; one canonical Bayes state, independent monitor.

    Frozen monitor tables are derived from the accepted updater's existing _laws
    owner (there is no public table export); row order and likelihoods are exact.
    No new prior, floor, null law or regime selection is fitted here. Explicitly
    setting enabled=True enables only this offline interface, not any actor.

    update() calls the owner ONCE per delivered packet; the owner scores each new
    interval once. Rolling old history is ignored; repeated/out-of-order deliveries
    refuse instead of the standalone owner's idempotent repeat convention. Scored
    rows must be certified passive exact flow. Certified active-product intervals
    and unsupported regimes are neutral; absent histories are coverage gaps.
    Uncertified rows refuse atomically, not silently fabricate missing evidence.

    Monitor restart clears ONLY its likelihood epoch. Adaptive weights, residual
    memory, updater history/prior/posterior and original root choice are retained.
    bind_public_drift in the existing meta owner exports current evidence but keeps
    ROOT_ONLY selection: post-opening alarms cannot reselect complete controllers.
    That unchanged live-selection composition branch is explicitly unsupported.

    State/delivery hashes detect changed bytes, not adversarially re-signed truth.
    Use expected_sha256 / expected_public_sha256 from a trusted receipt when an
    external integrity boundary is required. No opaque dict from an evaluator is
    accepted. Single-product sign law and 718-step owner horizon remain in force.
    """
    def __init__(self, updater, *, enabled: bool = False,
                 monitor_config: DriftConfig | None = None):
        from farm_neat.policy.v0_belief_update import BayesianFlowUpdater
        if type(updater) is not BayesianFlowUpdater:
            raise TypeError('accepted BayesianFlowUpdater required')
        if type(enabled) is not bool:
            raise ValueError('enabled must be bool')
        regimes = tuple((self.regime_key(law.shops, law.demand), law.probabilities)
                        for law in updater._laws.values())
        config = (DriftConfig(updater.hypothesis_ids, regimes)
                  if monitor_config is None else monitor_config)
        if (type(config) is not DriftConfig
                or config.hypothesis_ids != updater.hypothesis_ids
                or dict(config.regimes) != dict(regimes)):
            raise ValueError('monitor must use the exact updater catalog and laws')
        self.updater, self.monitor, self.enabled = updater, DriftMonitor(config), enabled
        self.bridge_sha256 = content_hash(dict(schema=BRIDGE_SCHEMA,
            updater=updater.updater_sha256, monitor=self.monitor.monitor_sha256,
            retained_monitor=RETAINED_MONITOR_SHA256, enabled=enabled))

    @staticmethod
    def regime_key(shops, demand) -> str:
        return content_hash(dict(shops=list(shops), demand=list(demand)))

    def _enabled(self):
        if not self.enabled:
            raise ValueError('DEFAULT_OFF: explicitly enable the diagnostic bridge')

    def _seal(self, updater: dict, monitor: dict, selector=None) -> PublicBeliefState:
        body = dict(schema=BRIDGE_SCHEMA, bridge_sha256=self.bridge_sha256,
                    updater=updater, monitor=monitor, selector=selector)
        return PublicBeliefState(wire(dict(body, content_sha256=content_hash(body))))

    def initialize(self, *, player: int, through_step: int = 0) -> PublicBeliefState:
        u = self.updater.initialize(player=player, through_step=through_step)
        m = self.monitor.initialize(through_step=through_step)
        return self._seal(u.to_dict(), self.monitor.export_state(m))

    def restore_state(self, document: dict, *, expected_sha256: str | None = None) -> PublicBeliefState:
        """Validate both owners, cross-clock/scored counts and original root draw.

        Reconstruction recomputes the root decision only to verify its original
        inputs; it never draws again or installs an initial belief over live state.
        """
        fields = {'schema','bridge_sha256','updater','monitor','selector','content_sha256'}
        if type(document) is not dict or set(document) != fields:
            raise ValueError('combined continuation must have closed fields')
        text = wire(document)
        if len(text) > 524288:
            raise ValueError('combined continuation exceeds bounded JSON size')
        doc = json.loads(text)
        digest = content_hash({k:v for k,v in doc.items() if k != 'content_sha256'})
        if (doc['schema'] != BRIDGE_SCHEMA or doc['bridge_sha256'] != self.bridge_sha256
                or doc['content_sha256'] != digest):
            raise ValueError('combined continuation identity/content hash mismatch')
        if expected_sha256 is not None:
            _hash(expected_sha256, 'expected_sha256')
            if digest != expected_sha256:
                raise ValueError('combined continuation trusted digest mismatch')
        u = self.updater.restore_state(doc['updater']).to_dict()
        m = self.monitor.restore_state(doc['monitor'])
        if (u['origin_step'] != m.origin_step or u['through_step'] != m.through_step
                or u['scored_intervals'] != m.scored
                or m.unscored_steps != u['through_step']-u['origin_step']-u['scored_intervals']):
            raise ValueError('combined updater/monitor clock or coverage mismatch')
        if doc['selector'] is not None:
            self._restore_selector(doc['selector'], player=u['player'], origin=u['origin_step'])
        return PublicBeliefState(text)

    def _validated(self, state):
        if type(state) is not PublicBeliefState:
            raise TypeError('combined PublicBeliefState required')
        return self.restore_state(state.to_dict()).to_dict()

    def update(self, previous: PublicBeliefState, observation, *,
               expected_public_sha256: str | None = None) -> PublicBeliefState:
        """Transactional immutable update: any owner/monitor refusal returns nothing."""
        from farm_neat.strategy.types import Evidence
        from farm_neat.evaluation.market_flow_labels import demand_at_step
        from farm_neat.game.constants import PRODUCTS, MAX_MARKET_ORDERS
        self._enabled()
        data = self._validated(previous)
        digest = public_observation_sha256(observation, product=self.updater.product)
        if expected_public_sha256 is not None:
            _hash(expected_public_sha256, 'expected_public_sha256')
            if digest != expected_public_sha256:
                raise ValueError('public delivery digest mismatch')
        before = data['updater']['through_step']
        now = observation.event.step
        _int(now, 'public step', 0, 718)
        if now <= before:
            raise ValueError('duplicate/out-of-order public delivery')
        if observation.default_calendar is not True:
            raise ValueError('uncertified public calendar')
        cursor = before
        for row in observation.history:
            if row.product != self.updater.product or row.end_step <= before:
                continue
            # The standalone owner sorts. This boundary instead refuses disorder.
            if row.start_step < cursor:
                raise ValueError('duplicate/out-of-order public intervals')
            cursor = row.end_step
            if (row.end_step != row.start_step+1 or row.shops is None
                    or row.own_order_ranks is None or None in row.demand_vector):
                raise ValueError('uncertified public interval')
            ranks = row.own_order_ranks
            if (type(ranks) is not tuple
                    or any(type(r) is not int or not 0 <= r < MAX_MARKET_ORDERS for r in ranks)
                    or ranks != tuple(sorted(set(ranks)))):
                raise ValueError('uncertified own order ranks')
            demand = demand_at_step(row.shops, row.start_step)
            if (tuple(demand[p] for p in PRODUCTS) != tuple(row.demand_vector)
                    or row.demand.kind is not Evidence.EXACT
                    or row.demand.value != demand[self.updater.product]):
                raise ValueError('uncertified public demand')
            if not row.own_order_ranks and (row.effect.kind is not Evidence.EXACT or row.reason is not None):
                raise ValueError('uncertified passive public effect')
        result = self.updater.update(self.updater.restore_state(data['updater']), observation)
        monitor = self.monitor.restore_state(data['monitor'])
        for event in result.events:
            if event['status'] == 'SCORED':
                monitor = self.monitor.step(monitor, through_step=event['end_step'],
                    sign=event['sign'], regime=self.regime_key(event['shops'], event['demand']))
            elif event['status'] in ('OWN_SAME_PRODUCT_REQUEST', 'UNSUPPORTED_REGIME'):
                monitor = self.monitor.step(monitor, through_step=event['end_step'], sign=None)
            else:
                raise ValueError('uncertified public evidence: '+event['status'])
        if monitor.through_step < now:
            monitor = self.monitor.step(monitor, through_step=now, sign=None)
        return self._seal(result.state.to_dict(), self.monitor.export_state(monitor), data['selector'])

    def evidence(self, state: PublicBeliefState) -> dict:
        """Closed detached actor evidence; UNKNOWN and adaptive residual are distinct.

        Uncovered time cannot discharge the monitor's scored-observation hold.
        Historical alarms are diagnostics, NOT hidden switch counts. With no
        current latch, an old alarm alone is not a permanent mismatch assertion.
        """
        data = self._validated(state)
        u, m = data['updater'], self.monitor.restore_state(data['monitor'])
        monitor = self.monitor.evidence(m)
        body = dict(schema=BRIDGE_SCHEMA+'-evidence', bridge_sha256=self.bridge_sha256,
            player=u['player'], through_step=u['through_step'],
            history_sha256=content_hash([u['history_sha256'], m.history_sha256]),
            canonical_belief=u['belief'], scored_observations=m.scored,
            unscored_steps=m.unscored_steps, skipped_intervals=u['skipped_intervals'],
            uncovered_steps=u['uncovered_steps'], alarm=monitor['alarm'],
            alarm_count=monitor['alarm_count'], fallback_required=monitor['fallback_required'],
            fallback_scored_observations_remaining=m.cooldown_remaining,
            adaptive_predictive_weights=monitor['adaptive_predictive_weights'],
            residual_semantics=monitor['residual_semantics'],
            selection_scope='ROOT_ONLY_COMPLETE_CONTROLLERS', calibration_status='UNMEASURED')
        return dict(body, content_sha256=content_hash(body))

    def commit_root(self, state: PublicBeliefState, context, forecast, uncertainty, *,
                    uniform: float, mode: str = 'META', config=None) -> PublicBeliefState:
        """Select/draw once; all future updates carry the exact original root capsule."""
        from farm_neat.policy import v0_belief_meta as M
        self._enabled()
        data = self._validated(state)
        if data['selector'] is not None:
            raise ValueError('root selector already committed')
        if data['updater']['through_step'] != 0:
            raise ValueError('ROOT_ONLY: post-opening drift cannot reselect controllers')
        config = M.SelectorConfig() if config is None else config
        f, u = M.bind_public_drift(context, forecast, uncertainty, bridge=self, state=state)
        decision = M.select_response(context, f, u, remaining_steps=f.horizon_steps,
                                     mode=mode, config=config)
        selector = dict(context=context, forecast=asdict(f), uncertainty=asdict(u),
            config=asdict(config), mode=mode, decision=decision.to_dict(),
            choice=M.commit_root(decision, uniform))
        return self._seal(data['updater'], data['monitor'], selector)

    def _restore_selector(self, saved, *, player, origin):
        from farm_neat.policy import v0_belief_meta as M
        if (type(saved) is not dict or set(saved) !=
                {'context','forecast','uncertainty','config','mode','decision','choice'}
                or origin != 0):
            raise ValueError('invalid original root selector capsule')
        try:
            f = dict(saved['forecast'])
            f['registry'] = M.ResponseRegistry(**f['registry'])
            f['hypotheses'] = tuple(M.HypothesisForecast(**h) for h in f['hypotheses'])
            forecast = M.ResponseForecast(**f)
            uncertainty = M.BeliefUncertainty(**saved['uncertainty'])
            # An INITIAL capsule is only a validation reference; live Bayes is never reset.
            initial = self.initialize(player=player)
            ff, uu = M.bind_public_drift(saved['context'], forecast, uncertainty,
                                        bridge=self, state=initial)
            decision = M.select_response(saved['context'], ff, uu,
                remaining_steps=forecast.horizon_steps, mode=saved['mode'],
                config=M.SelectorConfig(**saved['config']))
            if wire(saved['decision']) != wire(decision.to_dict()):
                raise ValueError('original root decision mismatch')
            M.restore_root_choice(saved['choice'], decision=decision)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('invalid original root selector continuation') from exc
