"""Opt-in, prefix-local acceleration of the reference V0 menu constructor.

Use ``build_menu(..., backend='fast')`` (default here) or ``backend='reference'``
for rollback. This module does not switch or monkey-patch existing consumers.
The original scorer owns input validation and exact successor partitioning.

All mutable caches belong to ONE invocation: state, Plan, projection options,
HOLD bindings and cursor are never cached across calls. Candidate projections
are keyed by exact plan values, not actor tokens or a hash. Arithmetic order in
cash accumulation and phi reductions is unchanged; no vectorized float sums,
approximate frontiers, economic rules or action IDs are introduced.
"""
from __future__ import annotations

from farm_neat.economy.v0 import projection as P
from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.policy import v0_scorer as reference


def _plan_key(plan: Plan) -> tuple:
    # Preserve explicit defaults too: they can distinguish canonical Plans.
    return (tuple(sorted(plan.cohorts.items())), tuple(sorted(plan.regimes.items())),
            plan.land, tuple(sorted(plan.market.items())))


class _Context:
    """Call-owned memoization; no cache key needs an implicit external state."""

    def __init__(self, state: P.State, options: dict):
        self.state, self.options = state, options
        self.projections = {}
        self.resolved = {}
        self.prices = {}
        # Exactly the generator used by Projection._cash_rows, evaluated once
        # per (product, day) instead of once per candidate projection.
        self.demand = {
            d: {p: sum(state.demand(p, step) for step in
                       range(24 * d, min(24 * d + 24, P.LAST_STEP + 1)))
                for p in V.PRODUCTS}
            for d in range(state.day, P.DAYS)
        }

    def project(self, plan: Plan) -> '_Projection':
        key = _plan_key(plan)
        if key not in self.projections:
            self.projections[key] = _Projection(self, plan)
        return self.projections[key]

    def price(self, product: str, inventory: int) -> int:
        key = (product, inventory)
        if key not in self.prices:
            self.prices[key] = P.price_at(product, inventory, self.state.params)
        return self.prices[key]

    def resolve(self, plan: Plan, module: int, delta: str) -> int:
        key = (_plan_key(plan), module, delta)
        if key in self.resolved:
            return self.resolved[key]
        current = plan.cohorts.get(module, 0)
        if V.MODULES[module]['birth'] < self.state.day:
            return current
        base = self.project(plan)
        signature = (tuple(base.hands), tuple(sorted(base.land.items())))
        best = current
        # Same stopping conditions, order, bounds and first-kink convention as
        # Resolver.__call__. NEXT_BINDING and MAX_FEASIBLE share trial objects,
        # not a guessed monotonicity/closed form or an opening-state result.
        for n in range(current + 1, current + 1 + 4 * V.SITES_PER_QUADRANT):
            trial = plan.copy()
            trial.cohorts[module] = n
            projected = self.project(trial)
            if not projected.feasible:
                break
            if delta == 'NEXT_BINDING' and (
                    tuple(projected.hands), tuple(sorted(projected.land.items()))) != signature:
                if best == current:
                    best = n
                break
            best = n
        self.resolved[key] = best
        return best

    def clear(self):
        # Projection -> Context -> projections would otherwise wait for cyclic
        # GC. Menus contain only value Plans/Prefixes, never this context.
        self.projections.clear()
        self.resolved.clear()
        self.prices.clear()
        self.demand.clear()


class _Projection(P.Projection):
    """Reference tapes, land, hands, binding and price mechanics; hoisted demand."""

    def __init__(self, context: _Context, plan: Plan):
        self.context = context
        super().__init__(context.state, plan, **context.options)

    def _price(self, product: str, inventory: int) -> int:
        return self.context.price(product, inventory)

    def _cash_rows(self) -> list[dict]:
        # Deliberately retain reference sequencing, including stock/input order,
        # last-day truncation, HOLD resolution, floor sales and daily net order.
        # Only loop-invariant overlay days and public daily demand are hoisted.
        s, t = self.state, self.tapes
        inv, stock = dict(s.inventory), dict(s.stock)
        rows = []
        held = {p: 0 for p in V.PRODUCTS}
        sell_days = {}
        for index, p in enumerate(V.PRODUCTS):
            primitive, horizon, _rank = self.plan.overlay(index)
            sell_days[p] = self._overlay_sell_day(p, primitive, horizon)
        for d in range(s.day, P.DAYS):
            wage = P.hire_bill(self.hands[d]) if self.hands[d] is not None else P.hire_bill(V.HAND_CAP)
            land = sum(V.LAND_PRICES[j - 1] for j, step in self.land_steps.items() if step // 24 == d)
            fixed = t.fixed[d]
            if d > s.day:
                for p in V.PRODUCTS:
                    stock[p] += t.prod[p][d - 1]
            needs = {'WHEAT': t.wheat[d], 'FERTILIZER': t.fert[d]}
            inputs = 0
            for p, need in needs.items():
                short = need - stock[p]
                if short > 0:
                    cost, inv[p] = self.buy_cost(p, inv[p], short)
                    inputs += cost
                    stock[p] += short
                stock[p] -= need
            receipts = 0
            for p in V.PRODUCTS:
                sell_day = sell_days[p]
                if stock[p] > 0 and (sell_day is None or d >= sell_day):
                    value, inv[p] = self.sell_value(p, inv[p], stock[p])
                    receipts += value
                    stock[p] = 0
                elif stock[p] > 0:
                    held[p] = stock[p]
            net = receipts - wage - land - fixed - inputs
            rows.append({'day': d, 'wage': wage, 'land': land, 'fixed': fixed, 'inputs': inputs,
                         'receipts': receipts, 'net': net, 'hands': self.hands[d]})
            for p in V.PRODUCTS:
                drift = self.context.demand[d][p] - self.belief[p]
                inv[p] = max(0, inv[p] - drift)
        self.held = held
        return rows


class _Masks(P.Masks):
    def __init__(self, context: _Context, plan: Plan):
        self.context = context
        self.state, self.plan, self.kw = context.state, plan, context.options
        self.resolver = context.resolve
        self.base = context.project(plan)
        self._plus_one, self._frontiers = {}, {}

    def plus_one_feasible(self, module: int) -> bool:
        if module not in self._plus_one:
            trial = self.plan.copy()
            trial.cohorts[module] = trial.cohorts.get(module, 0) + 1
            self._plus_one[module] = self.context.project(trial).feasible
        return self._plus_one[module]

    def legal(self, action: int) -> bool:
        fields = V.decode(action)
        if fields[0] != 'LAND':
            return super().legal(action)
        timing = V.LAND_TIMINGS[fields[1]]
        if self.state.quadrants >= 4 or timing == 'LAST_PROFITABLE' or self.plan.land == fields[1]:
            return False
        if timing == 'NEXT_BINDING' and not self.base.land:
            return False
        trial = self.plan.copy()
        trial.land = fields[1]
        return self.context.project(trial).feasible

    def prefix_actions(self, prefix: reference.Prefix) -> tuple[int, ...]:
        # Only an enumeration optimization: the full base Plan/projection above
        # still includes every earlier edit. Later support is recomputed here.
        if prefix.edit_count == V.MAX_EDITS_PER_EPOCH:
            return (V.STOP_ID,)
        out = [V.STOP_ID]
        for module in range(V.N_MODULES):
            if V.KEY_PROD_BASE + module > prefix.last_key:
                for delta in range(len(V.DELTAS)):
                    action = V.encode('PROD', module, delta)
                    if self.legal(action):
                        out.append(action)
            if V.KEY_SERVICE_BASE + module > prefix.last_key:
                for regime in range(len(V.REGIMES)):
                    action = V.encode('SERVICE', module, regime)
                    if self.legal(action):
                        out.append(action)
        if V.KEY_LAND > prefix.last_key:
            for timing in range(len(V.LAND_TIMINGS)):
                action = V.encode('LAND', timing)
                if self.legal(action):
                    out.append(action)
        for product in range(len(V.PRODUCTS)):
            if V.KEY_MARKET_BASE + product <= prefix.last_key:
                continue
            for primitive in range(len(V.PRIMITIVES)):
                if V.PRIMITIVES[primitive] in ('FRONTRUN', 'PIGGYBACK'):
                    continue
                for horizon in range(len(V.HORIZONS)):
                    for rank in range(len(self.frontier(product, primitive, horizon)) or 1):
                        action = V.encode('MARKET', product, primitive, horizon, rank)
                        if self.legal(action):
                            out.append(action)
        return tuple(sorted(out))


def _phi(state: P.State, plan: Plan, action: int, masks: _Masks) -> list[float]:
    """Same scalar formula/order as projection.phi; share its trial projection."""
    base = masks.base
    fields = V.decode(action)
    new = plan.apply(action, masks.resolver) if fields[0] != 'STOP' else plan
    after = masks.context.project(new) if fields[0] != 'STOP' else base
    out = {name: 0.0 for name in P.PHI_NAMES}
    out['reserve_delta'] = after.reserve - base.reserve
    out['reserve_after'] = after.reserve
    out['cash_slack'] = state.cash - after.reserve
    out['peak_hands_delta'] = after.peak_hands() - base.peak_hands()
    out['wages_delta'] = after.wages() - base.wages()
    out['sites_delta'] = max(after.tapes.occ) - max(base.tapes.occ)
    out['sites_after'] = max(after.tapes.occ)
    out['wheat_delta'] = sum(after.tapes.wheat) - sum(base.tapes.wheat)
    out['fertilizer_delta'] = sum(after.tapes.fert) - sum(base.tapes.fert)
    out['output_delta'] = sum(sum(v) for v in after.tapes.prod.values()) - sum(sum(v) for v in base.tapes.prod.values())
    if fields[0] == 'LAND' and after.land_steps:
        j = min(after.land_steps)
        out['land_price'] = V.LAND_PRICES[j - 1]
        out['land_step_delta'] = after.land_steps[j] - state.step
    if fields[0] == 'MARKET':
        product, primitive, horizon, rank = fields[1:]
        points = masks.frontier(product, primitive, horizon)
        if rank < len(points):
            out['market_reserve'], out['market_slots'], out['market_margin'] = (
                points[rank]['reserve'], points[rank]['slots'], points[rank]['margin'])
    if fields[0] == 'PROD':
        out['edits_days_to_birth'] = V.MODULES[fields[1]]['birth'] - state.day
    out['binding_' + (after.binding or 'none')] = 1.0
    return [float(out[name]) for name in P.PHI_NAMES]


def build_menu(state: P.State, prefix_plan: reference.Prefix | Plan, *,
               projection_kwargs=None, backend: str = 'fast') -> reference.Menu:
    """Drop-in exact Menu, with explicit rollback and no process-global switch.

    Callers must not mutate state/options concurrently during a call (also a
    requirement of the reference). Subsequent calls always start fresh, even
    when reusing and mutating the same State object or HOLD mapping.
    """
    if backend == 'reference':
        return reference.build_menu(state, prefix_plan, projection_kwargs=projection_kwargs)
    if backend != 'fast':
        raise ValueError("backend must be 'fast' or 'reference'")
    prefix = reference._prefix(prefix_plan)
    options = reference._projection_options(projection_kwargs)
    context = _Context(state, options)
    try:
        masks = _Masks(context, prefix.plan)
        ids = masks.prefix_actions(prefix)
        rows = [_phi(state, prefix.plan, action, masks) for action in ids]
        return reference.partition_menu(prefix, ids, rows, masks.resolver)
    finally:
        context.clear()
