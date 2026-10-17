"""Shared contract for non-neural strategy modules: a context in, proposals out.

Modules are written independently. Each reads a StrategyContext built from the actor's
own observation and memory and returns Proposal objects; a coordinator, never the
module, decides what executes. A quantity nobody knows is UNKNOWN, not a guessed number.
Own cash and margin are separate fields because own-cash gain is not margin gain.
Nothing here schedules work, plays games, scores proposals or reads evaluator data.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

LAST_STEP = 718
CLAIM = re.compile(r'[0-9a-f]{64}')
MODULE = re.compile(r'[a-z][a-z0-9_]{0,31}')
# Top-level keys that only an evaluator can know; their presence in a context is a leak.
EVALUATOR_ONLY = frozenset({'seed', 'world', 'world_id', 'scenario', 'future_shops', 'rival_private',
                            'rival_memory', 'opponent_private', 'labels'})


class Evidence(str, Enum):
    """Strength of a quantity, strongest first."""
    EXACT = 'EXACT'          # proved or computed exactly while the guards hold
    BOUND = 'BOUND'          # certified interval
    MEASURED = 'MEASURED'    # observed in recorded games; a sample, not a guarantee
    HEURISTIC = 'HEURISTIC'  # uncertified rule
    UNKNOWN = 'UNKNOWN'


_ORDER = list(Evidence)


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + ' must be a finite number')
    return float(value)


@dataclass(frozen=True)
class Estimate:
    """A dollar or unit quantity with its evidence.

    UNKNOWN carries no numbers; BOUND carries low and/or high (None is unbounded on that
    side); every other kind carries one value.
    """
    kind: Evidence = Evidence.UNKNOWN
    value: float | None = None
    low: float | None = None
    high: float | None = None

    def __post_init__(self):
        object.__setattr__(self, 'kind', Evidence(self.kind))
        if self.kind is Evidence.UNKNOWN:
            if (self.value, self.low, self.high) != (None, None, None):
                raise ValueError('UNKNOWN carries no numbers')
        elif self.kind is Evidence.BOUND:
            if self.value is not None or (self.low is None and self.high is None):
                raise ValueError('BOUND needs low or high and no point value')
            low = None if self.low is None else _finite(self.low, 'low')
            high = None if self.high is None else _finite(self.high, 'high')
            if low is not None and high is not None and low > high:
                raise ValueError('BOUND needs low <= high')
            object.__setattr__(self, 'low', low); object.__setattr__(self, 'high', high)
        else:
            if self.low is not None or self.high is not None:
                raise ValueError(self.kind.value + ' carries one value, not an interval')
            object.__setattr__(self, 'value', _finite(self.value, 'value'))

    @classmethod
    def exact(cls, value): return cls(Evidence.EXACT, value)
    @classmethod
    def measured(cls, value): return cls(Evidence.MEASURED, value)
    @classmethod
    def heuristic(cls, value): return cls(Evidence.HEURISTIC, value)
    @classmethod
    def bound(cls, low=None, high=None): return cls(Evidence.BOUND, None, low, high)

    @property
    def known(self) -> bool:
        return self.kind is not Evidence.UNKNOWN

    def to_dict(self) -> dict:
        return {'kind': self.kind.value, **{k: getattr(self, k) for k in ('value', 'low', 'high')
                                            if getattr(self, k) is not None}}

    @classmethod
    def from_dict(cls, data: Mapping) -> 'Estimate':
        if not isinstance(data, Mapping) or set(data) - {'kind', 'value', 'low', 'high'} or 'kind' not in data:
            raise ValueError('estimate needs kind and optional value/low/high')
        return cls(data['kind'], data.get('value'), data.get('low'), data.get('high'))


UNKNOWN = Estimate()


def difference(a: Estimate, b: Estimate) -> Estimate:
    """a - b with the weaker evidence of the two; a point minus a bound is UNKNOWN."""
    kind = _ORDER[max(_ORDER.index(a.kind), _ORDER.index(b.kind))]
    if kind is Evidence.UNKNOWN:
        return UNKNOWN
    if kind is Evidence.EXACT:
        return Estimate.exact(a.value - b.value)
    if kind is Evidence.BOUND:
        lo_a, hi_a = (a.value, a.value) if a.kind is Evidence.EXACT else (a.low, a.high)
        lo_b, hi_b = (b.value, b.value) if b.kind is Evidence.EXACT else (b.low, b.high)
        low = None if lo_a is None or hi_b is None else lo_a - hi_b
        high = None if hi_a is None or lo_b is None else hi_a - lo_b
        return UNKNOWN if low is None and high is None else Estimate.bound(low, high)
    if Evidence.BOUND in (a.kind, b.kind):
        return UNKNOWN
    return Estimate(kind, a.value - b.value)


def _strings(values, label, *, sort=False) -> tuple[str, ...]:
    if isinstance(values, str) or not all(isinstance(v, str) and v.strip() for v in values):
        raise ValueError(label + ' must be a sequence of nonempty strings')
    values = tuple(values)
    if len(set(values)) != len(values):
        raise ValueError('duplicate ' + label)
    return tuple(sorted(values)) if sort else values


def _parents(values) -> tuple[str, ...]:
    """Ledger claim hashes or repository-relative measurement receipt paths."""
    values = _strings(values, 'parents', sort=True)
    for value in values:
        if not CLAIM.fullmatch(value) and (value.startswith('/') or '..' in value.split('/') or '\\' in value):
            raise ValueError('parent must be a full claim hash or a repository-relative path: ' + value)
    return values


def _json(value, label):
    try:
        text = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(label + ' must be finite JSON') from error
    return json.loads(text)


@dataclass(frozen=True)
class Requirements:
    """What executing a proposal needs. Cash is spent; reserve is held, not spent."""
    cash: Estimate = UNKNOWN
    reserve: Estimate = UNKNOWN
    inventory: Mapping[str, Estimate] = field(default_factory=dict)
    market_slots: Estimate = UNKNOWN
    worker_hours: Estimate = UNKNOWN
    land_quadrants: Estimate = UNKNOWN
    window: tuple[int, int] | None = None  # first and last joint step it occupies

    def __post_init__(self):
        for name in ('cash', 'reserve', 'market_slots', 'worker_hours', 'land_quadrants'):
            if not isinstance(getattr(self, name), Estimate):
                raise ValueError(name + ' must be an Estimate')
        if not all(isinstance(k, str) and k and isinstance(v, Estimate) for k, v in self.inventory.items()):
            raise ValueError('inventory maps product names to Estimates')
        object.__setattr__(self, 'inventory', MappingProxyType(dict(sorted(self.inventory.items()))))
        if self.window is not None:
            first, last = self.window
            if not all(type(s) is int for s in (first, last)) or not 0 <= first <= last <= LAST_STEP:
                raise ValueError('window must be steps 0 <= first <= last <= 718')
            object.__setattr__(self, 'window', (first, last))

    def to_dict(self) -> dict:
        out = {name: getattr(self, name).to_dict()
               for name in ('cash', 'reserve', 'market_slots', 'worker_hours', 'land_quadrants')}
        out['inventory'] = {k: v.to_dict() for k, v in self.inventory.items()}
        out['window'] = list(self.window) if self.window else None
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> 'Requirements':
        names = ('cash', 'reserve', 'market_slots', 'worker_hours', 'land_quadrants')
        if set(data) != {*names, 'inventory', 'window'}:
            raise ValueError('closed requirements fields required')
        return cls(**{n: Estimate.from_dict(data[n]) for n in names},
                   inventory={k: Estimate.from_dict(v) for k, v in data['inventory'].items()},
                   window=tuple(data['window']) if data['window'] is not None else None)


@dataclass(frozen=True)
class Proposal:
    """One candidate change from one module. The coordinator decides; the module proposes.

    touches names every resource the change competes for (e.g. "cash", "market:WHEAT",
    "land", "labor"), known amount or not, so conflicts are visible before scoring.
    guards must hold at execution; invalidated_by lists events after which the module must
    recompute it. Evidence of the proposal is the evidence of its margin delta.
    """
    module: str
    decision_key: str
    candidate: Any
    touches: tuple[str, ...]
    requirements: Requirements = field(default_factory=Requirements)
    own_cash_delta: Estimate = UNKNOWN
    rival_cash_delta: Estimate = UNKNOWN
    margin_delta: Estimate = UNKNOWN
    guards: tuple[str, ...] = ()
    invalidated_by: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.module, str) or not MODULE.fullmatch(self.module):
            raise ValueError('module must be a short lowercase name')
        if not isinstance(self.decision_key, str) or not self.decision_key or any(c.isspace() for c in self.decision_key):
            raise ValueError('decision_key must be a nonempty string without whitespace')
        object.__setattr__(self, 'candidate', _json(self.candidate, 'candidate'))
        touches = _strings(self.touches, 'touches', sort=True)
        if not touches:
            raise ValueError('a proposal touches at least one resource')
        object.__setattr__(self, 'touches', touches)
        if not isinstance(self.requirements, Requirements):
            raise ValueError('requirements must be Requirements')
        for name in ('own_cash_delta', 'rival_cash_delta', 'margin_delta'):
            if not isinstance(getattr(self, name), Estimate):
                raise ValueError(name + ' must be an Estimate')
        object.__setattr__(self, 'guards', _strings(self.guards, 'guards'))
        object.__setattr__(self, 'invalidated_by', _strings(self.invalidated_by, 'invalidated_by'))
        object.__setattr__(self, 'parents', _parents(self.parents))
        implied = difference(self.own_cash_delta, self.rival_cash_delta)
        if implied.kind is Evidence.EXACT and self.margin_delta != implied:
            raise ValueError('exact own and rival deltas fix the margin delta')

    @property
    def key(self) -> tuple[str, str]:
        return self.module, self.decision_key

    @property
    def evidence(self) -> Evidence:
        return self.margin_delta.kind

    def to_dict(self) -> dict:
        return {'module': self.module, 'decision_key': self.decision_key, 'candidate': self.candidate,
                'touches': list(self.touches), 'requirements': self.requirements.to_dict(),
                'own_cash_delta': self.own_cash_delta.to_dict(), 'rival_cash_delta': self.rival_cash_delta.to_dict(),
                'margin_delta': self.margin_delta.to_dict(), 'guards': list(self.guards),
                'invalidated_by': list(self.invalidated_by), 'parents': list(self.parents)}

    @classmethod
    def from_dict(cls, data: Mapping) -> 'Proposal':
        fields = {'module', 'decision_key', 'candidate', 'touches', 'requirements', 'own_cash_delta',
                  'rival_cash_delta', 'margin_delta', 'guards', 'invalidated_by', 'parents'}
        if set(data) != fields:
            raise ValueError('closed proposal fields required')
        return cls(data['module'], data['decision_key'], data['candidate'], tuple(data['touches']),
                   Requirements.from_dict(data['requirements']),
                   *(Estimate.from_dict(data[n]) for n in ('own_cash_delta', 'rival_cash_delta', 'margin_delta')),
                   tuple(data['guards']), tuple(data['invalidated_by']), tuple(data['parents']))


class Verdict(str, Enum):
    FEASIBLE = 'FEASIBLE'
    INFEASIBLE = 'INFEASIBLE'
    UNKNOWN = 'UNKNOWN'


@dataclass(frozen=True)
class Feasibility:
    """A capacity answer about one proposal: a witness when feasible, a cut when not."""
    proposal: tuple[str, str]
    verdict: Verdict
    witness: Any = None
    certificate: Any = None
    marginal: Requirements = field(default_factory=Requirements)
    parents: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'verdict', Verdict(self.verdict))
        if not (isinstance(self.proposal, tuple) and len(self.proposal) == 2 and all(isinstance(p, str) and p for p in self.proposal)):
            raise ValueError('proposal is its (module, decision_key) key')
        object.__setattr__(self, 'witness', None if self.witness is None else _json(self.witness, 'witness'))
        object.__setattr__(self, 'certificate', None if self.certificate is None else _json(self.certificate, 'certificate'))
        if (self.verdict is Verdict.FEASIBLE) != (self.witness is not None):
            raise ValueError('a witness goes with FEASIBLE and only FEASIBLE')
        if (self.verdict is Verdict.INFEASIBLE) != (self.certificate is not None):
            raise ValueError('a certificate goes with INFEASIBLE and only INFEASIBLE')
        object.__setattr__(self, 'parents', _parents(self.parents))


@dataclass(frozen=True)
class Reserve:
    """Liquidity at one step: bank cash split into protected and discretionary parts.

    Each commitment is a funding witness {key, due_step, amount}; protected cash is what
    those need in chronological order, not a constant buffer.
    """
    step: int
    cash: float
    protected: Estimate
    discretionary: Estimate
    commitments: tuple[Mapping[str, Any], ...] = ()
    parents: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.step) is not int or not 0 <= self.step <= LAST_STEP:
            raise ValueError('step must be 0..718')
        object.__setattr__(self, 'cash', _finite(self.cash, 'cash'))
        for item in self.commitments:
            if (not isinstance(item, Mapping) or set(item) != {'key', 'due_step', 'amount'}
                    or type(item['due_step']) is not int or not self.step <= item['due_step'] <= LAST_STEP):
                raise ValueError('commitment needs key, due_step (now..718) and amount')
            _finite(item['amount'], 'commitment amount')
        object.__setattr__(self, 'commitments', tuple(_json(dict(c), 'commitment') for c in self.commitments))
        if self.protected.kind is Evidence.EXACT and self.discretionary.kind is Evidence.EXACT:
            if abs(self.protected.value + self.discretionary.value - self.cash) > 1e-9:
                raise ValueError('exact protected and discretionary cash must add up to bank cash')
        object.__setattr__(self, 'parents', _parents(self.parents))


class Outcome(str, Enum):
    WIN = 'WIN'
    DRAW = 'DRAW'
    LOSS = 'LOSS'


@dataclass(frozen=True)
class SuffixResult:
    """What happened when a proposal (None: the baseline) was adopted and the game played out."""
    proposal: tuple[str, str] | None
    status: str  # COMPLETE, PROPOSAL_FAILED or ERROR
    own_cash: float | None = None
    rival_cash: float | None = None
    failure: str | None = None
    realization: Any = None

    def __post_init__(self):
        if self.status not in ('COMPLETE', 'PROPOSAL_FAILED', 'ERROR'):
            raise ValueError('status is COMPLETE, PROPOSAL_FAILED or ERROR')
        if self.status == 'COMPLETE':
            object.__setattr__(self, 'own_cash', _finite(self.own_cash, 'own_cash'))
            object.__setattr__(self, 'rival_cash', _finite(self.rival_cash, 'rival_cash'))
            if self.failure is not None:
                raise ValueError('a complete suffix has no failure')
        elif not (isinstance(self.failure, str) and self.failure.strip()):
            raise ValueError('a failed suffix names its failure')
        object.__setattr__(self, 'realization', None if self.realization is None else _json(self.realization, 'realization'))

    @property
    def margin(self) -> float | None:
        return None if self.status != 'COMPLETE' else self.own_cash - self.rival_cash

    @property
    def outcome(self) -> Outcome | None:
        if self.margin is None:
            return None
        return Outcome.WIN if self.margin > 0 else Outcome.LOSS if self.margin < 0 else Outcome.DRAW


@dataclass(frozen=True)
class StrategyContext:
    """Actor-legal input for one decision point.

    shops is the ordered published-shop prefix; count vectors may be derived from it but
    never replace it (ordered history can decide the continuation). derived holds what
    other modules published for this step, keyed by module name.
    """
    step: int
    player: int
    observation: Mapping[str, Any]
    memory: Mapping[str, Any] = field(default_factory=dict)
    shops: tuple[str, ...] = ()
    derived: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if type(self.step) is not int or not 0 <= self.step <= LAST_STEP:
            raise ValueError('step must be 0..718')
        if self.player not in (0, 1) or type(self.player) is not int:
            raise ValueError('player must be 0 or 1')
        for name in ('observation', 'memory', 'derived'):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(name + ' must be a mapping')
            leaked = EVALUATOR_ONLY & set(value)
            if leaked:
                raise ValueError(f'evaluator-only keys in {name}: {sorted(leaked)}')
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        # Shops are drawn with replacement: a repeated name is a real second instance.
        if isinstance(self.shops, str) or not all(isinstance(v, str) and v.strip() for v in self.shops):
            raise ValueError('shops must be a sequence of nonempty shop names')
        object.__setattr__(self, 'shops', tuple(self.shops))

    def with_derived(self, module: str, value: Any) -> 'StrategyContext':
        if not MODULE.fullmatch(module) or module in self.derived:
            raise ValueError('publish once per module name per context')
        return replace(self, derived={**self.derived, module: value})


@dataclass(frozen=True)
class StepView:
    """Everything one decision step produced, handed from the module pipeline to the coordinator.

    proposals from every module that ran; feasibility answers keyed by proposal key; the
    liquidity reserve when one was computed; notes name modules that abstained and why.
    """
    context: StrategyContext
    proposals: tuple[Proposal, ...] = ()
    feasibility: Mapping[tuple[str, str], Feasibility] = field(default_factory=dict)
    reserve: Reserve | None = None
    notes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.context, StrategyContext):
            raise ValueError('context must be a StrategyContext')
        proposals = tuple(self.proposals)
        if any(not isinstance(p, Proposal) for p in proposals):
            raise ValueError('proposals must be Proposal objects')
        if len({p.key for p in proposals}) != len(proposals):
            raise ValueError('duplicate proposal key in one step')
        object.__setattr__(self, 'proposals', proposals)
        keys = {p.key for p in proposals}
        for key, answer in self.feasibility.items():
            if not isinstance(answer, Feasibility) or answer.proposal != key or key not in keys:
                raise ValueError('feasibility answers must match a proposal of this step')
        object.__setattr__(self, 'feasibility', MappingProxyType(dict(self.feasibility)))
        if self.reserve is not None and (not isinstance(self.reserve, Reserve) or self.reserve.step != self.context.step):
            raise ValueError('reserve must be a Reserve for this step')
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in self.notes.items()):
            raise ValueError('notes map module names to reasons')
        object.__setattr__(self, 'notes', MappingProxyType(dict(self.notes)))


@runtime_checkable
class StrategyModule(Protocol):
    name: str

    def propose(self, context: StrategyContext) -> Sequence[Proposal]: ...


def checked_proposals(module: StrategyModule, context: StrategyContext) -> tuple[Proposal, ...]:
    """Run one module and refuse output that is not its own or repeats a decision key."""
    proposals = tuple(module.propose(context))
    if any(not isinstance(p, Proposal) or p.module != module.name for p in proposals):
        raise ValueError(f'{module.name} returned a proposal that is not its own')
    keys = [p.decision_key for p in proposals]
    if len(keys) != len(set(keys)):
        raise ValueError(f'{module.name} repeated a decision key')
    return proposals


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', action='store_true', help='Print one example proposal as JSON')
    args = parser.parse_args(argv)
    if args.example:
        example = Proposal('market', 'sell:WHEAT:216', {'orders': [['SELL', 'WHEAT', 4]]}, ('cash', 'market:WHEAT'),
                           Requirements(inventory={'WHEAT': Estimate.exact(4)}, market_slots=Estimate.exact(1), window=(216, 216)),
                           own_cash_delta=Estimate.bound(40, 64), guards=('shed WHEAT >= 4',),
                           invalidated_by=('rival WHEAT sale before 216',))
        print(json.dumps(example.to_dict(), indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
