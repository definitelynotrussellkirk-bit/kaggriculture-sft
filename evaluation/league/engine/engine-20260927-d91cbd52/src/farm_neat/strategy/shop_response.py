"""Public shop events -> demand context, conditional values and V0 edit proposals.

Consumer contract
-----------------
``ShopResponse.update(ctx)`` returns a ``ShopUpdate``. Feed ``update.context`` to
other modules in this decision and persist ``update.memory`` as the actor's own
memory. Observed and handled prefixes are separate: an intraday event or missing
plan remains pending until a valid dawn response can be generated. Inputs are not mutated. ``propose(ctx)`` implements StrategyModule but
cannot persist an observation; a coordinator needing event detection uses update.

* The native ``observation.town.unlocked_shops`` is the authoritative ordered
  prefix when present. Nonempty ``ctx.shops`` must agree. The empty ctx.shops
  fallback is necessary because the frozen shared type mistakenly forbids repeats.
  No deduplication, count-based route selection or invented future shops occurs.
* Optional ``memory['v0_plan']`` is a canonical Plan. Every active HOLD also needs
  its existing ``memory['hold_releases']`` binding (the V0Agent format, integer or
  JSON-string product indices). These inputs belong to the coordinator; never to us.
* Optional ``derived['production_replan']['marginal_modules']`` lists V0 module
  keys to revalue. Without it, existing plan cohorts are revalued. This is a local
  adapter contract, not an assumed implementation of the parallel production module.
* ``derived['shop_response']`` is the typed ShopView. Demand is exact only for the
  explicitly configured clocks and published prefix. ``solo_*`` prices are exact
  **conditional, isolated sale** calculations with no intervening player flow.
  Producer ``projection_*`` numbers are the EXISTING V0 full-service, zero-flow,
  known-shop projection, not native cash, feasibility, optimality or margin.
  Actual own/rival/margin deltas remain UNKNOWN throughout.

Owns event detection, demand clocks and revaluation. Does not own market order
execution, funding, capacity, production add/retire selection, rival forecasting,
Router memory or proposal adoption. MARKET_HOLD and MARKET_CLEAR are alternatives
from the same exact Plan, each one edit plus STOP. Revalidate at the current prefix
before composition; an old legality result is not a successor-prefix mask.

Cost: at most 8 shops, 9 products, 2 market alternatives/product, and a declared
producer-probe budget (16 by default). No simulation/search/network/model calls.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from farm_neat.economy.heuristic.market import sell_revenue
from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import DEFAULT_LABOR_FACTOR, Masks, Projection, State
from farm_neat.game.constants import PRODUCTS, SHOPS, TOWN_CENTER_PRODUCTS, MAX_SHOP_INSTANCES
from farm_neat.strategy.types import Estimate, UNKNOWN, Proposal, Requirements, StrategyContext, LAST_STEP

NAME = 'shop_response'
MEMORY_SCHEMA = 'farm-shop-response-memory-v1'
PARENTS = (
    'ebdcf16a430cad8b54514c97fc2759f724d35f2b2b672326689e2a9781e6a99b',
    '3b9bcaa7088dc4dbec0c61e985c2518a5e3091a1051cbae7a4248ac2052640d3',
    '875f9a758a5f6d193865c274aad70564c8feec29b64492bacf7080c4590064ad',
)


def _int(value: Any, label: str, *, minimum: int | None = 0, maximum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f'{label} is not an integer in the required range')
    return value


def _shops(values: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or len(values) > MAX_SHOP_INSTANCES:
        raise ValueError('published shops must be an ordered list of at most eight instances')
    if any(not isinstance(s, str) or s not in SHOPS for s in values):
        raise ValueError('unknown shop name in published prefix')
    return tuple(values)  # Instances, including repeated names, not a set.


def _ticks(start: int, end: int, interval: int) -> int:
    """Number of multiples of interval in [start, end)."""
    return 0 if end <= start else (end - 1) // interval - (start - 1) // interval


@dataclass(frozen=True)
class DemandClock:
    """Explicit game configuration, not a learned step-number rule."""
    turns_per_day: int = 24
    reveal_days: int = 3
    shop_interval: int = 4
    center_interval: int = 24
    last_step: int = LAST_STEP

    def __post_init__(self):
        for name in ('turns_per_day', 'reveal_days', 'shop_interval', 'center_interval'):
            _int(getattr(self, name), name, minimum=1)
        _int(self.last_step, 'last_step', maximum=LAST_STEP)

    @property
    def reveal_interval(self) -> int:
        return self.turns_per_day * self.reveal_days

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ('turns_per_day', 'reveal_days', 'shop_interval', 'center_interval', 'last_step')}


@dataclass(frozen=True)
class ShopDemand:
    step: int
    ordered_shops: tuple[str, ...]
    clock: DemandClock = field(default_factory=DemandClock)
    clock_certified: bool = True

    def __post_init__(self):
        _int(self.step, 'step', maximum=LAST_STEP)
        if not isinstance(self.clock, DemandClock) or type(self.clock_certified) is not bool:
            raise ValueError('ShopDemand requires a DemandClock and boolean certification')
        object.__setattr__(self, 'ordered_shops', _shops(self.ordered_shops))
        expected = min(MAX_SHOP_INSTANCES, self.step // self.clock.reveal_interval)
        object.__setattr__(self, 'clock_certified', self.clock_certified and
                           self.step <= self.clock.last_step and len(self.ordered_shops) == expected)

    @property
    def shop_counts(self) -> Mapping[str, int]:
        return MappingProxyType({s: self.ordered_shops.count(s) for s in SHOPS})

    @property
    def shop_rate(self) -> Mapping[str, int]:
        return MappingProxyType({p: sum((2 if len(SHOPS[s]) == 1 else 1)
                                       for s in self.ordered_shops if p in SHOPS[s]) for p in PRODUCTS})

    @property
    def next_reveal(self) -> int | None:
        step = (len(self.ordered_shops) + 1) * self.clock.reveal_interval
        return step if len(self.ordered_shops) < MAX_SHOP_INSTANCES and step <= self.clock.last_step else None

    @property
    def exact_before(self) -> int:
        """Exclusive demand-step boundary; sale at this step precedes its demand."""
        return (self.next_reveal or self.clock.last_step + 1) if self.clock_certified else self.step

    @property
    def next_demand(self) -> Mapping[str, int | None]:
        out = {}
        for p, rate in self.shop_rate.items():
            candidates = []
            if rate:
                candidates.append((self.step // self.clock.shop_interval + 1) * self.clock.shop_interval)
            if p in TOWN_CENTER_PRODUCTS:
                candidates.append((self.step // self.clock.center_interval + 1) * self.clock.center_interval)
            out[p] = min(candidates) if candidates and min(candidates) <= self.clock.last_step else None
        return MappingProxyType(out)

    def between(self, start: int, end: int, *, shops: Sequence[str] | None = None) -> dict[str, int]:
        """Known-shop/center withdrawals in [start,end), excluding unobserved shops.

        This component remains calculable past the next reveal; it is NOT the total
        then. Use total_between for a certified total. Publication slot times retain
        the temporal information that a current histogram cannot recover.
        """
        _int(start, 'start', maximum=self.clock.last_step + 1)
        _int(end, 'end', minimum=start, maximum=self.clock.last_step + 1)
        prefix = self.ordered_shops if shops is None else _shops(shops)
        if self.ordered_shops[:len(prefix)] != prefix:
            raise ValueError('demand control must be an ordered sub-prefix')
        out = {p: _ticks(start, end, self.clock.center_interval) if p in TOWN_CENTER_PRODUCTS else 0 for p in PRODUCTS}
        for slot, shop in enumerate(prefix, 1):
            ticks = _ticks(max(start, slot * self.clock.reveal_interval), end, self.clock.shop_interval)
            for p in SHOPS[shop]:
                out[p] += ticks * (2 if len(SHOPS[shop]) == 1 else 1)
        return out

    def at(self, step: int) -> dict[str, int]:
        return self.between(step, step + 1)

    def total_between(self, start: int, end: int) -> dict[str, Estimate]:
        known = self.between(start, end)
        return {p: Estimate.exact(q) if self.clock_certified and end <= self.exact_before else UNKNOWN
                for p, q in known.items()}

    @property
    def cumulative_before(self) -> Mapping[str, int]:
        return MappingProxyType(self.between(0, self.step))


@dataclass(frozen=True)
class InventoryValue:
    product: str
    shed_units: Estimate = UNKNOWN
    carried_units: Estimate = UNKNOWN
    sale_step: int | None = None
    solo_now: Estimate = UNKNOWN
    solo_without_new_shops: Estimate = UNKNOWN
    solo_with_known_shops: Estimate = UNKNOWN
    known_shop_uplift: Estimate = UNKNOWN
    own_cash_delta: Estimate = UNKNOWN
    rival_cash_delta: Estimate = UNKNOWN
    margin_delta: Estimate = UNKNOWN
    reason: str = ''


@dataclass(frozen=True)
class ProducerValue:
    module: str
    mode: str = 'UNKNOWN'
    projection_before: float | None = None
    projection_after: float | None = None
    projection_change: float | None = None
    includes_coproducts: bool = True
    own_cash_delta: Estimate = UNKNOWN
    rival_cash_delta: Estimate = UNKNOWN
    margin_delta: Estimate = UNKNOWN
    reason: str = ''


@dataclass(frozen=True)
class ShopView:
    status: str
    demand: ShopDemand | None
    new_shops: tuple[str, ...] = ()
    inventory: Mapping[str, InventoryValue] = field(default_factory=dict)
    producers: Mapping[str, ProducerValue] = field(default_factory=dict)
    pending_shops: tuple[str, ...] = ()
    continuation: Estimate = UNKNOWN  # Never reconstructed from the histogram.
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        for key in ('inventory', 'producers'):
            object.__setattr__(self, key, MappingProxyType(dict(getattr(self, key))))


@dataclass(frozen=True)
class ShopUpdate:
    context: StrategyContext
    memory: Mapping[str, Any]
    proposals: tuple[Proposal, ...]

    @property
    def view(self) -> ShopView:
        return self.context.derived[NAME]


class ShopResponse:
    name = NAME

    def __init__(self, *, clock: DemandClock | None = None, max_producer_probes: int = 16,
                 max_sale_units: int = 100):
        self.clock = clock or DemandClock()
        self.max_producer_probes = _int(max_producer_probes, 'max_producer_probes', maximum=V.N_MODULES)
        self.max_sale_units = _int(max_sale_units, 'max_sale_units', maximum=10000)

    def propose(self, context: StrategyContext) -> tuple[Proposal, ...]:
        return self.update(context).proposals

    def _prefix(self, context: StrategyContext) -> tuple[str, ...] | None:
        obs = context.observation
        for key, expected in (('step', context.step), ('player', context.player)):
            if key in obs and (type(obs[key]) is not int or obs[key] != expected):
                raise ValueError(f'observation {key} and context {key} disagree')
        town = obs.get('town')
        native = _shops(town['unlocked_shops']) if isinstance(town, Mapping) and 'unlocked_shops' in town else None
        explicit = _shops(context.shops) if context.shops else None
        if explicit is not None and native is not None and explicit != native:
            raise ValueError('context shops and native published shops disagree')
        return native if native is not None else explicit

    def update(self, context: StrategyContext) -> ShopUpdate:
        prefix = self._prefix(context)
        memory = dict(context.memory)  # Other modules own their objects; never clone or mutate them.
        if prefix is None:
            view = ShopView('UNKNOWN', None, reasons=('ordered published prefix unavailable',))
            return ShopUpdate(context.with_derived(NAME, view), memory, ())
        previous = context.memory.get(NAME)
        old_prefix: tuple[str, ...] = ()
        handled_prefix: tuple[str, ...] = ()
        if previous is not None:
            if not isinstance(previous, Mapping) or previous.get('schema') != MEMORY_SCHEMA:
                raise ValueError('invalid shop-response memory schema')
            old_step = _int(previous.get('step'), 'saved step', maximum=self.clock.last_step)
            if old_step > context.step:
                raise ValueError('shop-response time moved backwards; restore the correct actor memory')
            if previous.get('clock') != self.clock.to_dict():
                raise ValueError('shop-response clock changed across continuation')
            old_prefix = _shops(previous.get('ordered_shops'))
            if prefix[:len(old_prefix)] != old_prefix:
                raise ValueError('published prefix was rewritten or truncated')
            handled_prefix = _shops(previous.get('handled_shops', ()))
            if old_prefix[:len(handled_prefix)] != handled_prefix:
                raise ValueError('handled shops must be a sub-prefix of observed shops')
        pending_shops = prefix[len(handled_prefix):]
        new_shops = prefix[len(old_prefix):]
        expected = min(MAX_SHOP_INSTANCES, context.step // self.clock.reveal_interval)
        clock_ok = context.step <= self.clock.last_step and len(prefix) == expected
        for key, expected_value in (('day', context.step // self.clock.turns_per_day),
                                    ('hour', context.step % self.clock.turns_per_day)):
            if key in context.observation and context.observation[key] != expected_value:
                clock_ok = False
        demand = ShopDemand(context.step, prefix, self.clock, clock_ok)
        reasons = []
        if not clock_ok:
            reasons.append('published count/phase does not certify configured clock')
        values = self._inventory(context, demand, old_prefix)
        plan, holds, state, missing = self._plan_input(context, prefix)
        reasons.extend(missing)
        producers = self._producers(context, plan, holds, state, old_prefix, clock_ok)
        proposals: tuple[Proposal, ...] = ()
        if pending_shops and clock_ok and plan is not None and state is not None:
            if context.step % self.clock.turns_per_day:
                reasons.append('plan edits require a new-day dawn; retain active plan')
            elif self.clock != DemandClock():
                reasons.append('V0 projection requires default clock; demand context still available')
            else:
                proposals = self._market_proposals(context, plan, holds, state, pending_shops)
                handled_prefix = prefix
        if clock_ok:
            # Do not acknowledge invalid observations: a corrected retry must still
            # see the shop event, rather than swallowing it behind stale memory.
            memory[NAME] = {'schema': MEMORY_SCHEMA, 'step': context.step,
                            'ordered_shops': list(prefix), 'handled_shops': list(handled_prefix),
                            'clock': self.clock.to_dict()}
        status = 'UNKNOWN' if not clock_ok else 'NEW_SHOPS' if new_shops else 'UNCHANGED'
        view = ShopView(status, demand, new_shops, values, producers, pending_shops=pending_shops, reasons=tuple(reasons))
        return ShopUpdate(context.with_derived(NAME, view), memory, proposals)

    def _inventory(self, context: StrategyContext, demand: ShopDemand,
                   old_prefix: tuple[str, ...]) -> dict[str, InventoryValue]:
        obs = context.observation
        private = obs.get('private') or {}
        shed, carried = private.get('shed'), private.get('inventories')
        market = obs.get('market') or {}
        inventory = market.get('inventory') or {}
        params = market.get('params')
        sale_step = min(demand.exact_before, self.clock.last_step)
        forecast_ok = demand.clock_certified and sale_step >= context.step
        before = demand.between(context.step, sale_step, shops=old_prefix) if forecast_ok else {}
        after = demand.between(context.step, sale_step) if forecast_ok else {}
        out = {}
        for p in PRODUCTS:
            n = _int(shed.get(p, 0), 'shed units') if isinstance(shed, Mapping) else None
            cargo = None
            if isinstance(carried, (tuple, list)) and all(isinstance(i, Mapping) for i in carried):
                cargo = sum(_int(i.get(p, 0), 'carried units') for i in carried)
            args = {'product': p, 'shed_units': Estimate.exact(n) if n is not None else UNKNOWN,
                    'carried_units': Estimate.exact(cargo) if cargo is not None else UNKNOWN,
                    'sale_step': sale_step if forecast_ok else None}
            reason = 'conditional solo sale; no intervening own/rival flow; shed custody only'
            if n is None or p not in inventory:
                out[p] = InventoryValue(**args, reason='missing shed or market inventory')
                continue
            if n > self.max_sale_units:
                out[p] = InventoryValue(**args, reason='sale valuation budget exhausted')
                continue
            inv = _int(inventory[p], 'market inventory', minimum=None)
            if params is not None and (not isinstance(params, Mapping) or p not in params):
                out[p] = InventoryValue(**args, reason='missing product price parameters')
                continue
            now = sell_revenue(p, inv, n, params)
            args['solo_now'] = Estimate.exact(now)
            if forecast_ok:
                control = sell_revenue(p, inv - before[p], n, params)
                changed = sell_revenue(p, inv - after[p], n, params)
                args.update(solo_without_new_shops=Estimate.exact(control),
                            solo_with_known_shops=Estimate.exact(changed),
                            known_shop_uplift=Estimate.exact(changed - control))
            out[p] = InventoryValue(**args, reason=reason)
        return out

    def _plan_input(self, context: StrategyContext, prefix: tuple[str, ...]):
        data = context.memory.get('v0_plan')
        if data is None:
            return None, {}, None, ['canonical V0 plan unavailable; no executable edit proposed']
        plan = Plan.from_canonical(data)
        obs = context.observation
        private, market = obs.get('private') or {}, obs.get('market') or {}
        farms = obs.get('farms')
        if (not isinstance(farms, (tuple, list)) or len(farms) <= context.player
                or not isinstance(farms[context.player], Mapping)):
            return plan, {}, None, ['own farm observation missing']
        farm = farms[context.player]
        required = {'money', 'unlocked_quadrants', 'hands', 'hires_today', 'tiles'}
        if (not required <= farm.keys() or not {'shed', 'inventories', 'seeds'} <= private.keys()
                or not set(PRODUCTS) <= (market.get('inventory') or {}).keys()):
            return plan, {}, None, ['incomplete actor state for canonical legality/projection']
        raw_holds = context.memory.get('hold_releases', {})
        if not isinstance(raw_holds, Mapping):
            raise ValueError('hold_releases must be the V0Agent binding mapping')
        holds = {}
        for p, overlay in plan.market.items():
            if V.PRIMITIVES[overlay[0]] != 'HOLD':
                continue
            value = raw_holds.get(p, raw_holds.get(str(p)))
            if value is None:
                return plan, {}, None, [f'missing existing HOLD release for {V.PRODUCTS[p]}']
            if (not isinstance(value, (tuple, list)) or len(value) != 2
                    or not isinstance(value[0], (tuple, list)) or tuple(value[0]) != overlay[:2]):
                raise ValueError('HOLD binding does not match active Plan overlay')
            _int(value[1], 'HOLD release', maximum=LAST_STEP)
            holds[p] = (tuple(value[0]), value[1])
        # Never rewrite the caller's observation or drop the repeated prefix.
        visible = dict(obs, step=context.step, player=context.player,
                       town={'unlocked_shops': list(prefix)})
        return plan, holds, State(visible), []

    def _producers(self, context, plan, holds, state, old_prefix, clock_ok) -> dict[str, ProducerValue]:
        producer = context.derived.get('production_replan')
        requested = producer.get('marginal_modules') if isinstance(producer, Mapping) else None
        if requested is None:
            requested = [V.MODULES[m]['key'] for m in sorted(plan.cohorts)] if plan is not None else []
        if (not isinstance(requested, (tuple, list)) or len(requested) > V.N_MODULES
                or any(not isinstance(k, str) or k not in V.MODULE_BY_KEY for k in requested)
                or len(set(requested)) != len(requested)):
            raise ValueError('marginal_modules must be unique known V0 module keys')
        out = {}
        states, bases = None, None
        for i, key in enumerate(requested):
            unavailable = None
            if plan is None or state is None or not clock_ok or self.clock != DemandClock():
                unavailable = 'canonical plan/state/continuation or default clock unavailable'
            elif i >= self.max_producer_probes:
                unavailable = 'producer probe budget exhausted'
            if unavailable:
                out[key] = ProducerValue(key, reason=unavailable)
                continue
            module = V.MODULE_BY_KEY[key]
            born = V.MODULES[module]['birth'] < state.day
            if born and module not in plan.cohorts:
                out[key] = ProducerValue(key, reason='cannot add a producer born before the current day')
                continue
            if states is None:
                control = copy.copy(state)
                control.shops = list(old_prefix)
                states = (control, state)
                bases = [self._nominal(s, plan, holds) for s in states]
            trial = plan.copy()
            if born:
                # A valuation counterfactual, NOT a legal retirement of a born animal.
                trial.cohorts[module] -= 1
                if trial.cohorts[module] == 0:
                    trial.cohorts.pop(module); trial.regimes.pop(module, None)
            else:
                trial.cohorts[module] = trial.cohorts.get(module, 0) + 1
            sign = -1 if born else 1
            before, after = [sign * (self._nominal(s, trial, holds) - base)
                             for s, base in zip(states, bases)]
            out[key] = ProducerValue(key, 'KEEP_ONE' if born else 'ADD_ONE', before, after, after - before,
                                     reason='V0 full-service/known-shops/zero-flow projection only; no realization or rival certificate')
        return out

    @staticmethod
    def _nominal(state: State, plan: Plan, holds) -> float:
        p = Projection(state, plan, hold_releases=holds, labor_factor=DEFAULT_LABOR_FACTOR)
        return sum(row['net'] for row in p.cash_rows)

    def _market_proposals(self, context, plan, holds, state, new_shops) -> tuple[Proposal, ...]:
        affected = {p for s in new_shops for p in SHOPS[s]}
        masks = Masks(state, plan, hold_releases=holds, labor_factor=DEFAULT_LABOR_FACTOR)
        proposals = []
        base_digest = plan.digest()
        for product, p in enumerate(V.PRODUCTS):
            if p not in affected:
                continue
            # Only inventory/tape-bearing products need market-response alternatives.
            if not state.stock[p] and not any(masks.base.tapes.prod[p][state.day:]):
                continue
            for family, primitive, horizon in (('MARKET_HOLD', 'HOLD', 'NEXT_DAWN'),
                                                ('MARKET_CLEAR', 'CLEAR', 'NOW')):
                action = V.encode('MARKET', product, V.PRIMITIVES.index(primitive), V.HORIZONS.index(horizon), 0)
                if not masks.legal(action):
                    continue
                target = plan.apply(action, masks.resolver)
                proposals.append(Proposal(
                    NAME, f'{context.step}:{family}:{p}',
                    {'kind': 'v0_edit', 'family': family, 'base_plan_sha256': base_digest,
                     'edits': [action, V.STOP_ID], 'target_plan': target.canonical(),
                     'ordered_shops': list(state.shops), 'new_shops': list(new_shops)},
                    ('cash', f'inventory:{p}', f'market:{p}', f'plan:market:{p}'),
                    # Adoption is a point guard; persistent resource occupancy is uncertified.
                    Requirements(),
                    guards=(f'current dawn step == {context.step}', f'current canonical Plan digest == {base_digest}',
                            'ordered published prefix unchanged', 'recompute canonical masks at adoption prefix',
                            'preserve all other HOLD continuation bindings', 'coordinator certifies funding and capacity'),
                    invalidated_by=('new shop', 'plan or continuation changed', 'own inventory changed',
                                    'market flow', 'next step'), parents=PARENTS))
        return tuple(proposals)
