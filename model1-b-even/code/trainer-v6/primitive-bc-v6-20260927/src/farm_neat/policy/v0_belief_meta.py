"""Bounded public-only response selection; no fitting, rollout or policy activation.

The accepted frontier is a lottery over TWO COMPLETE controllers, chosen ONCE at
step 0: ROBUST=0, LIGHT_EXPLOIT=1/4, AGGRESSIVE_EXPLOIT=1 probability of the exploit
endpoint. These names do not certify minimax safety. No per-turn mixing, logits,
V0 IDs, prefixes, compiler state or teacher admission are changed here.

``select_response(context, forecast, uncertainty, remaining_steps=...)`` validates
context through the existing regret actor allowlist. Forecasts/uncertainty are
closed immutable public-only values, bound to that context, history and horizon.
It accepts neither an evaluator record nor the actual opponent family. Model IDs
and source hashes associate rows; their contents are never numerical features.
The producer remains responsible for provenance/calibration: hashes do NOT prove
that supplied probabilities/error bounds came from public information.

The objective is expected terminal score P(WIN)+0.5*P(DRAW), an affine transform
of the registered terminal -1/0/+1 WDL reward, NOT a wealth/risk-shaped objective.
POSTERIOR maximizes its declared mean; META additionally requires a positive
lower gain under declared conditional-gain error and model misspecification.
An unrestricted residual model with gain -1 is ALWAYS in the uncertainty set,
including when the catalog has no UNKNOWN row. Missing MODEL laws refuse
model-based exploitation; UNKNOWN gets a pessimistic -1, never an invented law.

Model-TV radius and finite-sample gain-error bound are SEPARATE declarations,
not predictive variances. Defaults express ignorance (1 and 2), so default META
is robust. Entropy cutoffs are explicitly experimental, uncalibrated safety
controls: >=.9 -> robust; <=.5 with TV<=.1 and UNKNOWN<=.05 -> aggressive; otherwise
positive-lower-gain -> light. Entropy is not a confidence interval or independent
sample count. Predictive within/between variances are diagnostics, never penalties.
A confidently wrong model with falsely tiny bounds can still choose a losing arm;
no public-only rule can distinguish identical histories using hidden truth.
Even calibrated probabilities can select a losing arm on a rare realized family;
a mixture-expectation lower bound is not a family-wise or pathwise guarantee.

``commit_root(decision, own_uniform)`` emits a JSON choice to persist alongside
both the selected controller's full continuation and caller-owned sampler state.
``restore_root_choice(saved, decision=original_decision)`` verifies the same choice
without another draw. This is NOT a codec for either controller's internal state.
Do not reselect inside the game. Non-root contexts/horizon mismatches refuse.

O(K log K) selector arithmetic, O(K) storage, 1<=K<=64; no search/model calls,
files, network, games or sampling internally. Existing context validation adds
its bounded prefix/continuation validation cost. No production adapter is enabled.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Sequence

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.training import v0_regret_records as R

SCHEMA = 'farm-v0-belief-meta-v1'
ARMS = (('ROBUST', 0.), ('LIGHT_EXPLOIT', .25), ('AGGRESSIVE_EXPLOIT', 1.))
MODES = ('ALWAYS_ROBUST', 'ALWAYS_AGGRESSIVE', 'POSTERIOR', 'META')
MAX_HYPOTHESES = 64
_HASH = re.compile(r'[0-9a-f]{64}\Z')


def _number(value, name: str, low: float = 0., high: float = 1.) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite numeric data in [{low},{high}]')
    return 0. if value == 0 else float(value)


def _hash(value, name: str) -> None:
    if type(value) is not str or not _HASH.fullmatch(value):
        raise ValueError(f'{name} must be a lowercase SHA256')


def _integer(value, name: str, low: int = 0, high: int = V.LAST_STEP) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low},{high}]')


def _law(value, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if type(value) not in (list, tuple) or len(value) != 3:
        raise ValueError(f'{name} needs LOSS/DRAW/WIN probabilities')
    law = tuple(_number(x, name) for x in value)
    if abs(math.fsum(law)-1.) > 1e-12:
        raise ValueError(f'{name} probabilities must sum to one; no renormalization')
    return law


@dataclass(frozen=True)
class ResponseRegistry:
    """Frozen endpoint bindings to the frontier; identity is not a strength claim.

    Supply hashes of the actual complete policies and accepted frontier evidence.
    No paths are read, no controller is instantiated, no caller hash authenticated.
    """
    robust_id: str
    exploit_id: str
    robust_sha256: str
    exploit_sha256: str
    frontier_sha256: str

    def __post_init__(self):
        for name in ('robust_id', 'exploit_id'):
            value = getattr(self, name)
            if type(value) is not str or not value.strip() or len(value) > 128:
                raise ValueError('endpoint IDs need 1..128 nonblank characters')
        for name in ('robust_sha256', 'exploit_sha256', 'frontier_sha256'):
            _hash(getattr(self, name), name)
        if self.robust_id == self.exploit_id or self.robust_sha256 == self.exploit_sha256:
            raise ValueError('response endpoints must be distinct frozen policies')


@dataclass(frozen=True)
class HypothesisForecast:
    """Actor-time predictions, NEVER post-game endpoint observations.

    A MODEL has positive mass and nullable conditional WDL laws. UNKNOWN may have
    nullable laws too, but META treats its gain as -1 regardless of their optimism.
    """
    hypothesis_id: str
    mass: float
    robust_wdl: tuple[float, float, float] | None
    exploit_wdl: tuple[float, float, float] | None
    kind: str = 'MODEL'

    def __post_init__(self):
        _hash(self.hypothesis_id, 'hypothesis_id')
        mass = _number(self.mass, 'mass')
        if mass == 0.:
            raise ValueError('hypothesis mass must be positive; preserve explicit live support')
        if self.kind not in ('MODEL', 'UNKNOWN'):
            raise ValueError('kind must be MODEL or UNKNOWN')
        object.__setattr__(self, 'mass', mass)
        object.__setattr__(self, 'robust_wdl', _law(self.robust_wdl, 'robust_wdl'))
        object.__setattr__(self, 'exploit_wdl', _law(self.exploit_wdl, 'exploit_wdl'))


@dataclass(frozen=True)
class ResponseForecast:
    context_sha256: str
    history_sha256: str
    producer_sha256: str
    through_step: int
    horizon_steps: int
    registry: ResponseRegistry
    hypotheses: tuple[HypothesisForecast, ...]
    mass_basis: str = 'posterior_mass'

    def __post_init__(self):
        for name in ('context_sha256', 'history_sha256', 'producer_sha256'):
            _hash(getattr(self, name), name)
        _integer(self.through_step, 'through_step')
        _integer(self.horizon_steps, 'horizon_steps')
        if self.mass_basis not in ('posterior_mass', 'constrained_mass'):
            raise ValueError('explicit posterior_mass or constrained_mass basis required')
        if type(self.registry) is not ResponseRegistry:
            raise TypeError('registry must be a closed ResponseRegistry')
        if type(self.hypotheses) not in (list, tuple):
            raise TypeError('hypotheses must be a bounded list/tuple, not a record')
        rows = tuple(self.hypotheses)
        if not 1 <= len(rows) <= MAX_HYPOTHESES:
            raise ValueError('forecast needs 1..64 hypotheses')
        if any(type(row) is not HypothesisForecast for row in rows):
            raise TypeError('forecast rows must be closed HypothesisForecast values')
        if len({h.hypothesis_id for h in rows}) != len(rows):
            raise ValueError('duplicate hypothesis identity')
        if sum(h.kind == 'UNKNOWN' for h in rows) > 1:
            raise ValueError('at most one UNKNOWN row')
        if abs(math.fsum(h.mass for h in rows)-1.) > 1e-12:
            raise ValueError('masses must sum to one; missing mass is not renormalized')
        # Canonicalize association rows only, never the WDL category order.
        object.__setattr__(self, 'hypotheses', tuple(sorted(rows, key=lambda h: h.hypothesis_id)))


@dataclass(frozen=True)
class BeliefUncertainty:
    """Actor-time, public-evidence-bound declarations; not estimator inference.

    model_tv_radius bounds TV distance from the nominal categorical posterior,
    allowing mass on an unrestricted residual model. gain_error_bound is a uniform
    one-sided error on each MODEL's estimated terminal-score gain, in [0,2].
    The bound concerns the posterior-mixture expectation, NOT every realized
    family or root draw, and is conditional on BOTH declarations being valid.
    No CI level is invented. public_model_mismatch must be derived from public evidence, never a
    comparison to private truth. Defaults cannot license an exploit.
    """
    context_sha256: str
    history_sha256: str
    producer_sha256: str
    through_step: int
    model_tv_radius: float = 1.
    gain_error_bound: float = 2.
    public_model_mismatch: bool = False

    def __post_init__(self):
        for name in ('context_sha256', 'history_sha256', 'producer_sha256'):
            _hash(getattr(self, name), name)
        _integer(self.through_step, 'through_step')
        object.__setattr__(self, 'model_tv_radius', _number(self.model_tv_radius, 'model_tv_radius'))
        object.__setattr__(self, 'gain_error_bound', _number(self.gain_error_bound, 'gain_error_bound', 0., 2.))
        if type(self.public_model_mismatch) is not bool:
            raise ValueError('public_model_mismatch must be Boolean')


@dataclass(frozen=True)
class SelectorConfig:
    """Versioned diagnostic thresholds, not fitted/calibrated production defaults."""
    robust_entropy: float = .9
    aggressive_entropy: float = .5
    aggressive_tv_radius: float = .1
    aggressive_unknown_mass: float = .05
    gain_epsilon: float = 1e-12

    def __post_init__(self):
        for name in ('robust_entropy', 'aggressive_entropy', 'aggressive_tv_radius',
                     'aggressive_unknown_mass', 'gain_epsilon'):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if not self.aggressive_entropy < self.robust_entropy:
            raise ValueError('aggressive entropy threshold must be below robust threshold')


def contamination_lower_gain(masses: Sequence[float], gains: Sequence[float], radius: float) -> float:
    """Exact finite TV-ball lower gain with an additional mass-zero, gain=-1 state.

    Every gain is >=-1. The optimal adversary transfers up to ``radius`` mass from
    largest gains to that unrestricted state. The construction permits unseen
    models, unlike a closed-catalog TV ball. No outcome/hidden identity is read.
    """
    if type(masses) not in (tuple, list) or type(gains) not in (tuple, list):
        raise ValueError('bounded masses/gains lists required')
    if len(masses) != len(gains) or not 1 <= len(masses) <= MAX_HYPOTHESES:
        raise ValueError('one gain per hypothesis required')
    p = tuple(_number(x, 'mass') for x in masses)
    g = tuple(_number(x, 'gain', -1., 1.) for x in gains)
    if abs(math.fsum(p)-1.) > 1e-12:
        raise ValueError('masses must sum to one')
    budget = _number(radius, 'radius')
    penalties = []
    for gain, mass in sorted(zip(g, p), reverse=True):
        moved = min(mass, budget)
        penalties.append(moved*(gain+1.))
        budget = max(0., budget-moved)
    value = math.fsum(mass*gain for mass, gain in zip(p, g))-math.fsum(penalties)
    return max(-1., min(1., value))  # floating range only, not probability repair


def _score(law: tuple[float, float, float]) -> float:
    return math.fsum((.5*law[1], law[2]))


def _variance(rows: tuple[HypothesisForecast, ...], endpoint: str) -> tuple[float, float]:
    means = [_score(getattr(h, endpoint)) for h in rows]
    overall = math.fsum(h.mass*mean for h, mean in zip(rows, means))
    within = math.fsum(h.mass*math.fsum(p*(value-mean)**2
                      for p, value in zip(getattr(h, endpoint), (0., .5, 1.)))
                      for h, mean in zip(rows, means))
    between = math.fsum(h.mass*(mean-overall)**2 for h, mean in zip(rows, means))
    return within, between


@dataclass(frozen=True)
class ResponseDecision:
    context_sha256: str
    history_sha256: str
    input_sha256: str
    registry: ResponseRegistry
    remaining_steps: int
    mode: str
    arm: str
    root_exploit_probability: float
    reason: str
    entropy: float
    unknown_mass: float
    posterior_score_gain: float | None
    lower_score_gain: float | None
    robust_within_variance: float | None
    robust_between_variance: float | None
    exploit_within_variance: float | None
    exploit_between_variance: float | None

    @property
    def safety_certified(self) -> bool:
        return False

    def to_dict(self) -> dict:
        return dict(schema=SCHEMA, **asdict(self), safety_certified=False,
                    selection_scope='ROOT_ONLY_COMPLETE_CONTROLLERS',
                    score='P(WIN)+0.5*P(DRAW)', calibration_status='UNMEASURED')


def select_response(context: dict, forecast: ResponseForecast, uncertainty: BeliefUncertainty, *,
                    remaining_steps: int, mode: str = 'META',
                    config: SelectorConfig = SelectorConfig()) -> ResponseDecision:
    """Read only an existing closed actor context plus public-only typed estimates.

    This is a decision, not a sampled policy/controller. Always controls ignore
    model quality by design, but share context/source/horizon/type validity gates.
    POSTERIOR/META retain missing-law and UNKNOWN limits instead of renormalizing.
    """
    if type(forecast) is not ResponseForecast or type(uncertainty) is not BeliefUncertainty:
        raise TypeError('closed public forecast and uncertainty required, not evaluator records')
    if type(config) is not SelectorConfig:
        raise TypeError('closed SelectorConfig required')
    if type(mode) is not str or mode not in MODES:
        raise ValueError('mode must be ALWAYS_ROBUST, ALWAYS_AGGRESSIVE, POSTERIOR or META')
    key = R.actor_context_key(context)  # canonical owner validates every allowed field
    step = context['observation']['step']
    if (key != forecast.context_sha256 or key != uncertainty.context_sha256
            or forecast.history_sha256 != uncertainty.history_sha256
            or step != forecast.through_step or step != uncertainty.through_step):
        raise ValueError('forecast/uncertainty public context/history/time binding mismatch')
    _integer(remaining_steps, 'remaining horizon')
    if remaining_steps != V.LAST_STEP-step or forecast.horizon_steps != remaining_steps:
        raise ValueError('remaining horizon must equal the bound terminal horizon')
    if step != 0:
        raise ValueError('ROOT_ONLY: retain the original choice; no measured midgame-switch contract')
    rows = forecast.hypotheses
    masses = tuple(h.mass for h in rows)
    entropy = 0. if len(rows) == 1 else -math.fsum(p*math.log(p) for p in masses)/math.log(len(rows))
    entropy = max(0., min(1., entropy))
    unknown = math.fsum(h.mass for h in rows if h.kind == 'UNKNOWN')
    missing = any(h.kind == 'MODEL' and (h.robust_wdl is None or h.exploit_wdl is None) for h in rows)
    mean = lower = None
    variances = (None,)*4
    if not missing:
        gains = tuple(-1. if h.kind == 'UNKNOWN' else _score(h.exploit_wdl)-_score(h.robust_wdl) for h in rows)
        corrected = tuple(-1. if h.kind == 'UNKNOWN' else max(-1., gain-uncertainty.gain_error_bound)
                          for h, gain in zip(rows, gains))
        lower = contamination_lower_gain(masses, corrected, uncertainty.model_tv_radius)
        if unknown == 0.:
            mean = math.fsum(p*gain for p, gain in zip(masses, gains))
            variances = (*_variance(rows, 'robust_wdl'), *_variance(rows, 'exploit_wdl'))
    arm, reason = 'ROBUST', 'ALWAYS_ROBUST'
    if mode == 'ALWAYS_AGGRESSIVE':
        arm, reason = 'AGGRESSIVE_EXPLOIT', 'ALWAYS_AGGRESSIVE'
    elif mode in ('POSTERIOR', 'META'):
        if missing:
            reason = 'MISSING_CONDITIONAL_LAW'
        elif mode == 'POSTERIOR':
            if mean is not None and mean > config.gain_epsilon:
                arm, reason = 'AGGRESSIVE_EXPLOIT', 'POSITIVE_POSTERIOR_GAIN'
            else:
                reason = 'UNKNOWN_OR_NONPOSITIVE_POSTERIOR_GAIN'
        elif uncertainty.public_model_mismatch:
            reason = 'PUBLIC_MODEL_MISMATCH'
        elif entropy >= config.robust_entropy:
            reason = 'HIGH_ENTROPY'
        elif lower <= config.gain_epsilon:
            reason = 'NONPOSITIVE_LOWER_GAIN'
        elif (entropy <= config.aggressive_entropy
              and uncertainty.model_tv_radius <= config.aggressive_tv_radius
              and unknown <= config.aggressive_unknown_mass):
            arm, reason = 'AGGRESSIVE_EXPLOIT', 'POSITIVE_LOWER_GAIN_LOW_UNCERTAINTY'
        else:
            arm, reason = 'LIGHT_EXPLOIT', 'POSITIVE_LOWER_GAIN_GUARDED'
    fingerprint = R.canonical_sha256(dict(schema=SCHEMA, forecast=asdict(forecast),
                                         uncertainty=asdict(uncertainty), config=asdict(config),
                                         mode=mode, remaining_steps=remaining_steps))
    return ResponseDecision(key, forecast.history_sha256, fingerprint, forecast.registry,
                            remaining_steps, mode, arm, dict(ARMS)[arm], reason, entropy,
                            unknown, mean, lower, *variances)


def commit_root(decision: ResponseDecision, uniform: float) -> dict:
    """Choose once with caller's independent own uniform; no RNG hidden in this API."""
    if type(decision) is not ResponseDecision:
        raise TypeError('ResponseDecision required')
    uniform = _number(uniform, 'uniform')
    if uniform == 1.:
        raise ValueError('uniform must be in [0,1)')
    if decision.arm not in dict(ARMS) or decision.root_exploit_probability != dict(ARMS)[decision.arm]:
        raise ValueError('invalid registered root arm')
    registry = decision.registry
    exploit = uniform < decision.root_exploit_probability
    value = dict(schema=SCHEMA+'-root-choice', decision_sha256=R.canonical_sha256(decision.to_dict()),
                 registry_sha256=R.canonical_sha256(asdict(registry)), uniform=uniform,
                 root_exploit_probability=decision.root_exploit_probability,
                 endpoint_id=registry.exploit_id if exploit else registry.robust_id,
                 endpoint_sha256=registry.exploit_sha256 if exploit else registry.robust_sha256)
    return dict(value, content_sha256=R.canonical_sha256(value))


def restore_root_choice(saved: dict, *, decision: ResponseDecision) -> dict:
    """Verify original decision/draw/endpoint exactly; caller restores controller state.

    No current-state forecast is accepted here; replay must use the ORIGINAL
    actor-time decision. Hashes establish identity, not external authenticity.
    """
    if type(saved) is not dict or 'uniform' not in saved:
        raise ValueError('invalid root continuation')
    try:
        expected = commit_root(decision, saved['uniform'])
        R._safe(saved)
        same = R.canonical_sha256(saved) == R.canonical_sha256(expected)
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid root continuation') from exc
    if not same:
        raise ValueError('root continuation differs from the frozen original decision')
    return expected


def forecast_from_actor(actor: dict, *, registry: ResponseRegistry) -> ResponseForecast:
    """Bridge a validated public-only V0 belief sidecar WITHOUT changing its wire.

    This optional bridge requires BOTH registry endpoint IDs to be reachable
    candidates in the accepted actor sidecar, and the registry's endpoint source
    digests to equal their full candidate payload hashes. Existing native scripted
    Router/K0006 IDs do not meet that condition: they use explicit public forecasts
    through the core API, never a fabricated V0 reachability claim.

    Existing owner replays prefix-local candidate geometry and continuation. This
    adds no policies, legality rules, missing laws or posterior-floor recycling.
    Candidate continuations remain opening edits then KEEP, as owned by the sidecar.
    Callers must not pass the whole trainer/evaluator record here.
    """
    from farm_neat.training import v0_belief_records as B

    if type(registry) is not ResponseRegistry:
        raise TypeError('closed ResponseRegistry required')
    B.actor_payload_key(actor)  # reject extra/private fields; verify native prefix replay
    candidates = {c['candidate_sha256']: c for c in actor['candidates']}
    for key, expected in ((registry.robust_id, registry.robust_sha256),
                          (registry.exploit_id, registry.exploit_sha256)):
        if key not in candidates or R.canonical_sha256(candidates[key]) != expected:
            raise ValueError('registry endpoint does not match the validated V0 candidate')
    belief = actor['belief']
    basis = belief['mass_used']
    rows = tuple(HypothesisForecast(h['hypothesis_id'], h[basis],
                                   h['conditional_wdl'][registry.robust_id],
                                   h['conditional_wdl'][registry.exploit_id], h['kind'])
                 for h in belief['hypotheses'])
    evidence = belief['public_evidence']
    step = actor['context']['observation']['step']
    return ResponseForecast(actor['context_sha256'], evidence['source_sha256'],
                            evidence['producer_sha256'], evidence['through_step'],
                            V.LAST_STEP-step, registry, rows, basis)


def bind_public_drift(context: dict, forecast: ResponseForecast,
                      uncertainty: BeliefUncertainty, *, bridge, state):
    """Bind current closed drift/coverage evidence without widening ROOT_ONLY.

    This is a diagnostic input bridge, NOT an online controller-switch contract.
    Caller forecasts retain nullable public conditional laws and must already use
    the exact canonical posterior (or explicitly declared output-floor mass).
    Adaptive monitor weights/residual responsibility never replace those masses.
    Coverage gaps restore ignorance bounds; a current alarm hold sets mismatch.
    Neither signal grants safety or identifies a hidden family. Non-root inputs
    can be bound for evaluation, but select_response still refuses them unchanged.
    """
    from dataclasses import replace
    from farm_neat.policy.v0_belief_drift import PublicBeliefBridge, PublicBeliefState
    if type(bridge) is not PublicBeliefBridge or type(state) is not PublicBeliefState:
        raise TypeError('closed public bridge and combined state required')
    if type(forecast) is not ResponseForecast or type(uncertainty) is not BeliefUncertainty:
        raise TypeError('closed public forecast and uncertainty required')
    key = R.actor_context_key(context)
    evidence = bridge.evidence(state)
    step = context['observation']['step']
    if (context['observation']['player'] != evidence['player']
            or key != forecast.context_sha256 or key != uncertainty.context_sha256
            or step != forecast.through_step or step != uncertainty.through_step
            or step != evidence['through_step']
            or forecast.history_sha256 != uncertainty.history_sha256
            or forecast.history_sha256 != evidence['history_sha256']
            or forecast.horizon_steps != V.LAST_STEP-step):
        raise ValueError('drift forecast/uncertainty context/history/time binding mismatch')
    rows = {r['hypothesis_id']: r for r in evidence['canonical_belief']['hypotheses']}
    if set(rows) != {h.hypothesis_id for h in forecast.hypotheses}:
        raise ValueError('forecast catalog differs from canonical belief')
    for h in forecast.hypotheses:
        row = rows[h.hypothesis_id]
        if h.kind != row['kind'] or h.mass != row[forecast.mass_basis]:
            raise ValueError('forecast must retain canonical belief mass and UNKNOWN kind')
    missing = evidence['unscored_steps'] > 0
    return forecast, replace(uncertainty,
        model_tv_radius=1. if missing else uncertainty.model_tv_radius,
        gain_error_bound=2. if missing else uncertainty.gain_error_bound,
        public_model_mismatch=uncertainty.public_model_mismatch or evidence['fallback_required'])
