"""Offline frozen-menu PROD/SERVICE compatibility law. Not a production decoder.

Inputs: a valid V0 Plan, verified frozen non-STOP menus keyed by V.plan_key,
finite base-context logits, and a remaining edit cap (0..16). The caller owns
state/mask construction, expired-key filtering and subtracting prior edits from
the cap. Never give this law teacher prefixes during own-prefix evaluation.

For module m, let P be all PROD weights, S all SERVICE weights, and C the PROD
weights that leave a cohort present. Replace (1+P*z)(1+S*z) by
    F_m(z) = 1 + (P+S)*z + C*S*z**2.
Other keys retain 1+W*z. Z sums coefficients through the cap. Counting degree
TWO for the joint choice is essential. All marginals are derivatives of log Z.

With the original Resolver (which never decreases a count), only DROP_ALL and
MINUS_ONE from count one remove a cohort. This is independent of other modules'
PROD edits; ALL PROD keys still precede ALL SERVICE keys in native execution.
Custom decreasing resolvers are outside this contract.

Outputs: log_z, log_probability, marginals, nll_gradient, exact integer counts,
and a canonical action-id sampler. Raw aliases remain separate; STOP is a
forced unscored terminator and is NOT returned by sample. No masks are changed,
no successor classes are merged, and no trainer/accepted normalizer is called.

This removes missing-cohort interpreter failures within the supplied frozen
support. It is NOT a fresh-prefix legality, joint-funding, compiled-realization,
final-Plan likelihood, existing autoregressive-law or playing-strength claim.
Arithmetic identities are exact over reals; evaluation uses binary64.
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan

SCHEMA = 'farm-v0-frozen-editset-module-compat-v1'


def _logsumexp(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values:
        return -math.inf
    high = max(values)
    return high + math.log(math.fsum(math.exp(x - high) for x in values))


def _probabilities(log_weights: Iterable[float]) -> tuple[float, ...]:
    weights = tuple(log_weights)
    high = max(weights)
    scaled = tuple(math.exp(x - high) for x in weights)
    total = math.fsum(scaled)
    return tuple(x / total for x in scaled)


@dataclass(frozen=True)
class _Lane:
    """One conditional categorical choice, retaining every raw action id."""
    actions: tuple[int, ...]
    probabilities: tuple[float, ...]
    log_weight: float

    @classmethod
    def create(cls, actions: Iterable[int], logits: Mapping[int, float]) -> '_Lane':
        actions = tuple(sorted(actions))
        weights = tuple(logits[a] for a in actions)
        return cls(actions, _probabilities(weights), _logsumexp(weights))


@dataclass(frozen=True)
class _Factor:
    keys: tuple[int, ...]
    # Degree 0 has no lane; degree 1 chooses any singleton; degree 2 independently
    # chooses one surviving PROD and one SERVICE. No pair enumeration is needed.
    lanes: tuple[tuple[_Lane, ...], ...]
    log_coefficients: tuple[float, ...]
    counts: tuple[int, ...]

    @classmethod
    def create(cls, keys: tuple[int, ...], singletons: Iterable[int],
               surviving_prod: Iterable[int], services: Iterable[int],
               logits: Mapping[int, float]) -> '_Factor':
        lanes: list[tuple[_Lane, ...]] = [(), (_Lane.create(singletons, logits),)]
        surviving_prod, services = tuple(surviving_prod), tuple(services)
        if surviving_prod and services:
            lanes.append((_Lane.create(surviving_prod, logits), _Lane.create(services, logits)))
        logs = tuple(math.fsum(l.log_weight for l in choices) for choices in lanes)
        if any(not math.isfinite(x) for x in logs):
            raise ValueError('factor log partition exceeds finite floating-point range')
        counts = tuple(math.prod(len(l.actions) for l in choices) for choices in lanes)
        return cls(keys, tuple(lanes), logs, counts)


class CompatibilitySetLaw:
    """Capped additive-logit law on cohort-compatible frozen action-id sets.

    Polynomial DP and all-marginal arithmetic cost O(A + F*K), with storage
    O(A + F*K); deterministic input sorting additionally costs O(A*log(A)).
    A is the supplied action count, F the number of module/other-key factors and
    K the remaining edit cap. Each polynomial has degree at most two.

    The low-level constructor validates ids/keys/logits and local cohort/service
    preconditions, NOT complete economic legality. Call Masks on the actual
    current prefix first. A subset menu is permitted but must be labeled as such.
    """

    def __init__(self, groups: Mapping[int, Mapping[int, float]], plan: Plan,
                 cap: int = V.MAX_EDITS_PER_EPOCH):
        if type(cap) is not int or not 0 <= cap <= V.MAX_EDITS_PER_EPOCH:
            raise ValueError('remaining edit cap must be an integer in 0..16')
        if not isinstance(plan, Plan) or not isinstance(groups, Mapping):
            raise ValueError('a V0 Plan and a mapping of frozen groups are required')
        if any(type(k) is not int or not 0 <= k < V.N_KEYS for k in groups):
            raise ValueError('frozen V0 plan keys are required')
        self.cap = cap
        self.base_plan_sha256 = plan.digest()
        copied, logits, action_key, erasing = {}, {}, {}, set()
        for key in sorted(groups):
            if not isinstance(groups[key], Mapping):
                raise ValueError('each frozen key must map action ids to logits')
            group = {}
            for action, value in groups[key].items():
                if type(action) is not int or action == V.STOP_ID or V.plan_key(action) != key:
                    raise ValueError('non-STOP action ids must match their frozen V0 key')
                try:
                    value = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError('finite logits required; remove masked actions') from exc
                if not math.isfinite(value):
                    raise ValueError('finite logits required; remove masked actions')
                fields = V.decode(action)
                if fields[0] == 'PROD':
                    count, delta = plan.cohorts.get(fields[1], 0), V.DELTAS[fields[2]]
                    if delta in ('DROP_ALL', 'MINUS_ONE') and count == 0:
                        raise ValueError('frozen deletion menu requires an existing cohort')
                    if delta == 'DROP_ALL' or (delta == 'MINUS_ONE' and count == 1):
                        erasing.add(action)
                elif fields[0] == 'SERVICE':
                    module, regime = fields[1:]
                    if (module not in plan.cohorts or not V.regime_legal(module, regime)
                            or plan.regime(module) == regime):
                        raise ValueError('SERVICE must be locally legal in the frozen base plan')
                group[action] = value
                logits[action], action_key[action] = value, key
            if group:
                copied[key] = MappingProxyType(dict(sorted(group.items())))
        self.groups = MappingProxyType(copied)
        self.logits = MappingProxyType(logits)
        self.action_key = MappingProxyType(action_key)
        self._erasing = frozenset(erasing)
        factors = []
        for module in range(V.N_MODULES):
            keys = tuple(k for k in (module, V.N_MODULES + module) if k in copied)
            if not keys:
                continue
            production, services = copied.get(module, {}), copied.get(V.N_MODULES + module, {})
            factors.append(_Factor.create(keys, (*production, *services),
                (a for a in production if a not in erasing), services, logits))
        for key in sorted(k for k in copied if k >= V.KEY_LAND):
            factors.append(_Factor.create((key,), copied[key], (), (), logits))
        self._factors = tuple(factors)
        # H[i,r] = log weight of choices from factor i onward using AT MOST r edits.
        self._suffix = [[0.] * (cap + 1) for _ in range(len(factors) + 1)]
        for i in range(len(factors) - 1, -1, -1):
            for r in range(cap + 1):
                self._suffix[i][r] = _logsumexp(
                    log_c + self._suffix[i + 1][r - degree]
                    for degree, log_c in enumerate(factors[i].log_coefficients) if degree <= r)
        self.log_z = self._suffix[0][cap]
        if not math.isfinite(self.log_z):
            raise ValueError('log partition exceeds finite floating-point range')

    def _target(self, actions: Iterable[int]) -> tuple[int, ...]:
        actions = tuple(actions)
        if len(actions) > self.cap:
            raise ValueError('target exceeds remaining edit cap')
        selected = {}
        for action in actions:
            if type(action) is not int or action not in self.action_key:
                raise ValueError('target contains a masked/unknown action or STOP')
            key = self.action_key[action]
            if key in selected:
                raise ValueError('target repeats a plan key')
            selected[key] = action
        for key, action in selected.items():
            if action in self._erasing and key + V.N_MODULES in selected:
                raise ValueError('incompatible PROD/SERVICE: production removes the cohort')
        return actions

    def supports(self, actions: Iterable[int]) -> bool:
        """Membership in this frozen compatible SET law, not prefix-local legality."""
        try:
            self._target(actions)
        except ValueError:
            return False
        return True

    def log_probability(self, actions: Iterable[int]) -> float:
        """Accept any set serialization; reject invalid sets, aliases are not merged."""
        chosen = self._target(actions)
        return math.fsum(self.logits[a] for a in chosen) - self.log_z

    def factor_polynomials(self) -> list[dict]:
        """Inspectable degree coefficients: integer counts and log weighted sums."""
        return [{'keys': list(f.keys), 'counts': list(f.counts),
                 'log_coefficients': list(f.log_coefficients)} for f in self._factors]

    def count_by_size(self) -> list[int]:
        """Exact integer count of labeled compatible sets of each size 0..cap."""
        counts = [1] + [0] * self.cap
        for factor in self._factors:
            counts = [sum(c * counts[j-d] for d, c in enumerate(factor.counts) if d <= j)
                      for j in range(self.cap + 1)]
        return counts

    def _degrees(self, i: int, remaining: int) -> tuple[float, ...]:
        return _probabilities(log_c + self._suffix[i + 1][remaining - degree]
            for degree, log_c in enumerate(self._factors[i].log_coefficients) if degree <= remaining)

    def marginals(self) -> dict[int, float]:
        """All action inclusion probabilities / log-Z derivatives in one pass."""
        out = {a: 0. for a in self.logits}
        masses = [0.] * (self.cap + 1)
        masses[self.cap] = 1.
        for i, factor in enumerate(self._factors):
            after = [0.] * (self.cap + 1)
            degree_mass = [0.] * len(factor.lanes)
            for remaining, mass in enumerate(masses):
                if not mass:
                    continue
                for degree, probability in enumerate(self._degrees(i, remaining)):
                    move = mass * probability
                    after[remaining - degree] += move
                    degree_mass[degree] += move
            for degree, choices in enumerate(factor.lanes):
                for lane in choices:
                    for action, probability in zip(lane.actions, lane.probabilities):
                        out[action] += degree_mass[degree] * probability
            masses = after
        return out

    def nll_gradient(self, target: Iterable[int]) -> dict[int, float]:
        """Cotangent for this law only; NOT an autoregressive actor gradient."""
        selected = set(self._target(target))
        return {a: probability - (a in selected) for a, probability in self.marginals().items()}

    def sample(self, rng: random.Random) -> tuple[int, ...]:
        """One draw, original canonical key order, no STOP; caller owns execution."""
        def draw(probabilities):
            threshold, cumulative = rng.random(), 0.
            for index, probability in enumerate(probabilities):
                cumulative += probability
                if threshold < cumulative:
                    return index
            return len(probabilities) - 1

        remaining, chosen = self.cap, []
        for i, factor in enumerate(self._factors):
            degree = draw(self._degrees(i, remaining))
            for lane in factor.lanes[degree]:
                chosen.append(lane.actions[draw(lane.probabilities)])
            remaining -= degree
        return tuple(sorted(chosen, key=V.plan_key))
