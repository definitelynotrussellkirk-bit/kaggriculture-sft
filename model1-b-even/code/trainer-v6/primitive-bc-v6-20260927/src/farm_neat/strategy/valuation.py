"""Actor-only, supported continuation values for production and endgame.

Consumers
---------
``ctx.with_derived('production_value', PortfolioValuator(...))`` supplies the
accepted ProductionValuator protocol. analyze() additionally returns its dated
pooled settlement word and support. This is a CONDITIONAL cash-flow enclosure:
outputs must be delivered before that day's inputs, future internal inputs are
reserved, surplus sells in the last market phase of each day. It is NOT a route,
storage, funding or full-service certificate. Fixed acquisition/land remain the
production module's responsibility. Unknown wages or rival cash stay UNKNOWN.
A wage quote must be keyed by cost_key(context, tape), not shared across changes.

``public_suffix_model(ctx, baseline_orders, proposals)`` supplies endgame's
PublicSuffixModel or None when its support is missing. Its finite public law is
market-only: all physical actions PASS, own dated SELL/BUY_PRODUCT orders, native
nightly carried-stock deposits, and a CERTIFIED public rival PASS law. All eight
shop instances must already be published. Weeds/biology cannot change bank cash
under this restricted action word; we never simulate or guess their hidden RNG.
The baseline is the supplied word, NOT K0006. Exactness is conditional on this law,
not an observation-based identification of the opponent and not playing strength.

OpponentModel laws are retained in full as sensitivity diagnostics. Historical
passive E_eff is neither an active-actor order word nor rival cash. Its price
sensitivity is HEURISTIC, never a certified support bound or a terminal value.
The generic production enclosure does NOT use that empirical support to prune.

Actor calls do no historical/evaluator/network I/O, native env.step or search.
Only bundled pinned engine code is loaded; no evaluator snapshot is restored.
Limits are explicit in ValuationBudget. calibration_table is an OFFLINE join of
already-produced predictions to SuffixResults; it never passes outcomes back into
an actor. No fitter, evaluator snapshot, hidden stock or future shop is stored.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from farm_neat.game.constants import (
    ANIMALS, BUYABLE_PRODUCTS, DEFAULTS, MAX_MARKET_ORDERS, MAX_SHOP_INSTANCES,
    PRODUCTS, SHOPS,
)
from farm_neat.game.exact_engine import load_engine
from farm_neat.evaluation.market_flow_labels import demand_at_step
from farm_neat.strategy.endgame import EndgameGuards, PublicSuffixModel, context_key
from farm_neat.strategy.opponent import FlowLaw, OpponentModel, OpponentState
from farm_neat.strategy.production import PortfolioTape, ProductionBook, ProductionValue, published_shops
from farm_neat.strategy.types import (
    EVALUATOR_ONLY, Estimate, Evidence, Proposal, Requirements, StrategyContext,
    SuffixResult, UNKNOWN, difference,
)

PARENTS = (
    '56089f5537253080d7902bee52d35fc5eca38ff2cf83c9d9830428b007349d84',
    '3ce373846de04a17d5da8c33676e52e796bab468988aa4b585f80fc6a8203ac4',
    '55a108a84f0820284a29723230a74cda1038f3e8b536c03467326a2c746dc765',
    '4c1d9de9c0c7d10ccf28d381b4fbb79118a8ed39ea9bc912beda6cfd8b5f6fc1',
    '4a74a8c92963b7f11d10005c4d6a0d6b58193e92471cb61e224d8817f124a1a6',
)
TERMINAL = 719
DAYS = 30
WORD_SCHEMA = 'farm-public-market-word-v1'
_FORBIDDEN = EVALUATOR_ONLY | {'snapshot', 'evaluator_snapshot', 'terminal_labels', 'hidden_stock'}
# Public capacity/order limit, not a forecast of rival requested or filled orders.
_PHASE_UNITS = DEFAULTS['shedCapacity'] * MAX_MARKET_ORDERS


def _wire(value):
    if isinstance(value, ProductionBook):
        return _wire(value.to_dict())
    if isinstance(value, Mapping):
        if _FORBIDDEN.intersection(value):
            raise ValueError('evaluator-only data in actor input')
        if any(type(k) is not str for k in value):
            raise ValueError('actor input needs string mapping keys')
        return {k: _wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire(v) for v in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('actor input must be finite JSON or a ProductionBook')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def actor_key(context: StrategyContext) -> str:
    """Full actor-state binding; not a cash mark or a minimal-state assertion.

    Equivalent to endgame.context_key on JSON contexts; ProductionBook is also
    supported for production. Derived callbacks/evaluator data are never hashed
    as if they were actor state. Only explicitly named publications are consumed.
    """
    return _hash(_wire(dict(step=context.step, player=context.player,
                           observation=context.observation, memory=context.memory, shops=context.shops)))


def tape_key(tape: PortfolioTape) -> str:
    return _hash(tape.to_dict())


def cost_key(context: StrategyContext, tape: PortfolioTape) -> str:
    return _hash({'actor': actor_key(context), 'tape': tape_key(tape)})


def _count(n, label):
    if type(n) is not int or n < 0:
        raise ValueError(label + ' must be a nonnegative integer')
    return n


def _stock(value):
    if not isinstance(value, Mapping) or set(value) - set(PRODUCTS + list(ANIMALS)):
        raise ValueError('unsupported/missing own custody')
    return {p: _count(n, 'stock') for p, n in value.items()}


def _public(context):
    key = actor_key(context)
    obs = context.observation
    for k, expected in (('step', context.step), ('player', context.player),
                        ('day', context.step // 24), ('hour', context.step % 24)):
        if type(obs.get(k)) is not int or obs[k] != expected:
            raise ValueError('context and observation identity differ')
    shops = published_shops(context)
    if len(shops) != min(MAX_SHOP_INSTANCES, context.step // 72):
        raise ValueError('incomplete or not-yet-published shop prefix')
    market = obs.get('market', {})
    if market.get('params') not in (None, load_engine().MARKET_PARAMS):
        raise ValueError('nondefault public market parameters are unsupported')
    inventory = market.get('inventory')
    if not isinstance(inventory, Mapping) or set(inventory) != set(PRODUCTS) or any(type(n) is not int for n in inventory.values()):
        raise ValueError('complete integer public market inventory required')
    return key, shops, dict(inventory)


@dataclass(frozen=True)
class ValuationBudget:
    max_unit_quotes: int = 100_000
    max_suffix_actions: int = 191
    max_candidates: int = 32

    def __post_init__(self):
        for k, n in vars(self).items():
            _count(n, k)
        if self.max_unit_quotes > 1_000_000 or self.max_suffix_actions > TERMINAL or self.max_candidates > 256:
            raise ValueError('valuation budget exceeds portable limits')


@dataclass(frozen=True)
class CashValue:
    own_cash: Estimate = UNKNOWN
    rival_cash: Estimate = UNKNOWN
    support: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.own_cash, Estimate) or not isinstance(self.rival_cash, Estimate):
            raise ValueError('cash values must be Estimates')
        object.__setattr__(self, 'support', MappingProxyType(_wire(self.support)))

    @property
    def margin(self):
        return difference(self.own_cash, self.rival_cash)

    def to_dict(self):
        return {'own_cash': self.own_cash.to_dict(), 'rival_cash': self.rival_cash.to_dict(),
                'margin': self.margin.to_dict(), 'support': copy.deepcopy(dict(self.support))}


def unconditioned_value(context: StrategyContext) -> CashValue:
    """A cut state alone does not determine policy-conditional terminal cash."""
    return CashValue(support={'actor_key': actor_key(context), 'start_step': context.step,
                             'terminal_step': TERMINAL, 'published_shops': list(published_shops(context)),
                             'scope': 'UNCONDITIONED_CONTINUATION_NOT_IDENTIFIED',
                             'parents': list(PARENTS)})


@dataclass(frozen=True)
class PortfolioValuation:
    value: ProductionValue
    schedule: tuple[dict, ...] = ()
    support: Mapping = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()

    def to_dict(self):
        return {'value': self.value.to_dict(), 'schedule': list(self.schedule),
                'support': dict(self.support), 'diagnostics': list(self.diagnostics)}


def _validate_tape(context, tape):
    if not isinstance(tape, PortfolioTape) or context.step % 24 or tape.from_day != context.step // 24:
        raise ValueError('matching dawn PortfolioTape required')
    if set(tape.outputs) != set(PRODUCTS) or set(tape.inputs) - set(BUYABLE_PRODUCTS):
        raise ValueError('complete product outputs and buyable-only inputs required')
    for row in (*tape.outputs.values(), *tape.inputs.values(), tape.fixed_cash, tape.actions, tape.sites):
        if len(row) != DAYS or any(type(n) is not int or n < 0 for n in row):
            raise ValueError('dated tape rows need 30 nonnegative integer entries')
    if any(any(row[:tape.from_day]) for row in (*tape.outputs.values(), *tape.inputs.values(), tape.fixed_cash)):
        raise ValueError('elapsed flows must not be charged again')
    private = context.observation.get('private', {})
    if not isinstance(private.get('inventories'), (tuple, list)):
        raise ValueError('own carried inventories required')
    pooled = dict.fromkeys(PRODUCTS, 0)
    for bag in (private.get('shed'), *private['inventories']):
        for p, n in _stock(bag).items():
            if p in pooled: pooled[p] += n
    if any(type(n) is not int or n < 0 for n in tape.stock.values()) or set(tape.stock) - set(PRODUCTS):
        raise ValueError('invalid projected stock')
    if any(tape.stock.get(p, 0) != pooled[p] for p in PRODUCTS):
        raise ValueError('projected stock is not current own custody')


def _settlements(tape):
    # Minimum starting inventory needed AFTER day d to avoid later avoidable buys.
    reserves = {p: [0] * DAYS for p in PRODUCTS}
    for p in PRODUCTS:
        need = 0
        for day in range(DAYS - 1, tape.from_day - 1, -1):
            reserves[p][day] = need
            need = max(0, need + tape.inputs.get(p, (0,) * DAYS)[day] - tape.outputs[p][day])
    stock = {p: tape.stock.get(p, 0) for p in PRODUCTS}
    rows = []
    for day in range(tape.from_day, DAYS):
        buys, sells = {}, {}
        for p in PRODUCTS:
            stock[p] += tape.outputs[p][day]
            need = tape.inputs.get(p, (0,) * DAYS)[day]
            bought = max(0, need - stock[p])
            if bought: buys[p] = bought
            stock[p] += bought - need
            sold = max(0, stock[p] - reserves[p][day])
            if sold: sells[p] = sold
            stock[p] -= sold
        if buys or sells:
            rows.append({'step': min(24 * day + 23, TERMINAL - 1), 'buys': buys, 'sells': sells})
    return tuple(rows)


def _demand_bounds(shops, step):
    known = demand_at_step(shops, step)
    missing = max(0, min(MAX_SHOP_INSTANCES, step // 72) - len(shops))
    upper = dict(known)
    if step % 4 == 0 and missing:
        for p in PRODUCTS:
            weight = max((2 if len(items) == 1 else 1) if p in items else 0 for items in SHOPS.values())
            upper[p] += missing * weight
    return known, upper


def _cash_enclosure(context, shops, inventory, schedule, budget):
    """Conservative per-unit monotone interval propagation, not a mean-flow model.

    A rival can fill at most shedCapacity units per market order. We deliberately
    re-allow that full phase budget at every own unit boundary and after the own
    word: this OVER-encloses interleavings, including $1 sales that do not restore
    inventory after buys. Never assert that one net E_eff describes a mixed phase.
    Nonbuyable products have no negative rival displacement. Town uncertainty uses
    the maximum per-instance demand over public shop TYPES, not hidden shop draws.
    """
    engine = load_engine()
    lo, hi = dict(inventory), dict(inventory)
    receipt_lo = receipt_hi = cost_lo = cost_hi = quotes = 0
    by_step = {row['step']: row for row in schedule}

    def widen(product):
        if product in BUYABLE_PRODUCTS: lo[product] -= _PHASE_UNITS
        hi[product] += _PHASE_UNITS

    for step in range(context.step, max(by_step, default=context.step - 1) + 1):
        for p in PRODUCTS: widen(p)
        row = by_step.get(step, {})
        for side in ('buys', 'sells'):
            for p, count in row.get(side, {}).items():
                for _ in range(count):
                    if quotes >= budget.max_unit_quotes:
                        raise ValueError('unit-quote budget exhausted; no partial value')
                    quotes += 1
                    widen(p)
                    if side == 'buys':
                        lo[p] -= 1; hi[p] -= 1
                        cost_lo += engine.market_price(p, hi[p]); cost_hi += engine.market_price(p, lo[p])
                    else:
                        low_price = engine.market_price(p, hi[p]); high_price = engine.market_price(p, lo[p])
                        receipt_lo += low_price; receipt_hi += high_price
                        lo[p] += high_price > 1; hi[p] += low_price > 1
        for p in PRODUCTS: widen(p)
        d_lo, d_hi = _demand_bounds(shops, step)
        for p in PRODUCTS:
            lo[p] -= d_hi[p]; hi[p] -= d_lo[p]
    receipts = Estimate.exact(0) if not any(r['sells'] for r in schedule) else Estimate.bound(receipt_lo, receipt_hi)
    costs = Estimate.exact(0) if not any(r['buys'] for r in schedule) else Estimate.bound(cost_lo, cost_hi)
    return receipts, costs, quotes


def flow_sensitivity(context, product, quantity, law: FlowLaw, *, budget=None):
    """Shifted-inventory sale sensitivity, NOT an identified simultaneous fill.

    Reuses the entire historical marginal law. The interval of scenario receipts
    is diagnostic support only, not Estimate.BOUND and not exhaustive future risk.
    """
    budget = budget or ValuationBudget()
    _, _, inventory = _public(context)
    _count(quantity, 'quantity')
    if product not in PRODUCTS or not isinstance(law, FlowLaw):
        raise ValueError('known product and FlowLaw required')
    if quantity * len(law.counts) > budget.max_unit_quotes:
        raise ValueError('sensitivity unit-quote budget exhausted')
    engine = load_engine()
    rows = []
    mean = 0.0
    for effect, probability in law.atoms:
        inv, cash = inventory[product] + effect, 0
        for _ in range(quantity):
            price = engine.market_price(product, inv); cash += price; inv += price > 1
        rows.append({'effect': effect, 'cash': cash, 'mass': [probability.numerator, probability.denominator]})
        mean += float(probability) * cash
    estimate = Estimate.heuristic(mean) if rows else UNKNOWN
    return {'product': product, 'quantity': quantity, 'law': law.to_dict(), 'law_sha256': _hash(law.to_dict()),
            'estimate': estimate.to_dict(), 'scenarios': rows,
            'scenario_range': [min(r['cash'] for r in rows), max(r['cash'] for r in rows)] if rows else None,
            'certified_future_support': UNKNOWN.to_dict(), 'rival_cash': UNKNOWN.to_dict(),
            'scope': 'SHIFTED_INVENTORY_SENSITIVITY_NOT_ACTIVE_ORDER_OR_TERMINAL_VALUE'}


class PortfolioValuator:
    """Complete-portfolio quote protocol. Missing labor/cash laws are not zero.

    labor_costs is a caller-owned certificate map, keyed by cost_key(ctx, tape).
    Its truth is a producer responsibility. Alternatively, labor_certificates
    accepts computed LaborCertificate objects from the conditional scheduling
    producer. These additionally bind ordered custody and preserve their support,
    refusal diagnostics and parents. Neither path establishes executable profit.
    """
    def __init__(self, *, budget=None, labor_costs=None, labor_parents=(), labor_certificates=()):
        # Local import avoids a cycle: the producer reuses this contract's keys.
        from farm_neat.strategy.labor_certificate import LaborCertificate

        self.budget = budget or ValuationBudget()
        costs = dict(labor_costs or {})
        if any(not isinstance(k, str) or len(k) != 64 or not isinstance(v, Estimate) for k, v in costs.items()):
            raise ValueError('labor costs require binding hashes and Estimates')
        if costs and (not labor_parents or any(not isinstance(p, str) or not p for p in labor_parents)):
            raise ValueError('labor certificates require source parents')
        if any(x is not None and x < 0 for v in costs.values() for x in (v.value, v.low, v.high)):
            raise ValueError('whole-portfolio labor cash must be nonnegative')
        self.labor_costs = MappingProxyType(costs)
        self.labor_parents = tuple(labor_parents)
        certificates = {}
        for certificate in labor_certificates:
            if not isinstance(certificate, LaborCertificate):
                raise ValueError('computed labor certificates must be LaborCertificate objects')
            if certificate.cost_key in certificates or certificate.cost_key in costs:
                raise ValueError('ambiguous labor certificates for the same context and tape')
            certificates[certificate.cost_key] = certificate
        self.labor_certificates = MappingProxyType(certificates)

    def quote(self, context, tape):
        return self.analyze(context, tape).value

    def analyze(self, context, tape):
        # Leakage is a programming error, never silently ignored as unavailable data.
        key = actor_key(context)
        support = {'actor_key': key, 'start_step': context.step, 'terminal_step': TERMINAL,
                   'scope': 'CONDITIONAL_POOLED_DAILY_SETTLEMENT', 'tape_sha256': tape_key(tape),
                   'rival_cash': UNKNOWN.to_dict(), 'parents': list(PARENTS)}
        if not tape.history_certified or tape.stock is None:
            return PortfolioValuation(ProductionValue(), support=support,
                                      diagnostics=('OWN_PROJECTION_OR_CUSTODY_UNCERTIFIED',))
        try:
            _, shops, inventory = _public(context)
            _validate_tape(context, tape)
            schedule = _settlements(tape)
            receipts, inputs, quotes = _cash_enclosure(context, shops, inventory, schedule, self.budget)
        except ValueError as error:
            return PortfolioValuation(ProductionValue(), support=support, diagnostics=(str(error),))
        support.update(published_shops=list(shops), unit_quotes=quotes,
                       rival_flow_scope='PUBLIC_CAPACITY_ENCLOSURE_OVERCOUNTS_INTERLEAVINGS',
                       future_shops_scope='ALL_PUBLIC_SHOP_TYPES_NOT_REALIZED_DRAWS')
        forecast = context.derived.get('opponent')
        if forecast is not None:
            fresh = OpponentModel().observe(context)
            if (not isinstance(forecast, OpponentState) or not forecast.default_calendar
                    or forecast.public_fingerprint != fresh.public_fingerprint
                    or forecast.event.step != context.step or forecast.event.player != context.player):
                raise ValueError('stale or incompatible public opponent publication')
            support['public_flow_laws'] = {p: forecast.beliefs[p].to_dict() for p in PRODUCTS}
            support['public_flow_use'] = 'DIAGNOSTIC_ONLY_NOT_EXHAUSTIVE_OR_JOINT_FUTURE_SUPPORT'
        labor_binding = cost_key(context, tape)
        labor = self.labor_costs.get(labor_binding, UNKNOWN)
        labor_parents = self.labor_parents if labor.known else ()
        certificate = self.labor_certificates.get(labor_binding)
        if certificate is not None:
            support['labor_certificate'] = certificate.summary()
            if certificate.matches(context, tape):
                labor, labor_parents = certificate.labor_cash, certificate.parents
            else:
                labor, labor_parents = UNKNOWN, ()
                support['labor_certificate_binding'] = 'REFUSED'
        guards = (
            'default engine-valid state, prices, order cap and shed capacity',
            'entire dated full-service portfolio, starting custody and co-products are realized',
            'all daily outputs delivered before daily inputs; reserve future internal inputs before selling surplus',
            'daily settlement is the final market phase; every requested fill is funded and delivered without overflow',
            'labor certificate is bound to this actor and entire tape; missing wages remain UNKNOWN',
            'fixed acquisition and land are charged separately by production; no other own cash flows',
            'cash enclosure is conditional, not capacity, route, funding, global optimum or K0006 evidence',
        )
        parents = tuple(sorted(set(PARENTS + labor_parents)))
        value = ProductionValue(receipts, inputs, labor, UNKNOWN, guards, parents)
        notes = ('RIVAL_CASH_LAW_UNKNOWN',) + (() if labor.known else ('LABOR_CASH_UNKNOWN',))
        if certificate is not None and not labor.known:
            notes += certificate.diagnostics or ('LABOR_CERTIFICATE_NOT_APPLICABLE',)
        return PortfolioValuation(value, schedule, support, notes)


def _orders_word(context, orders, budget):
    if not isinstance(orders, Mapping) or len(orders) > budget.max_suffix_actions:
        raise ValueError('dated market-order mapping required within budget')
    rows, quotes = [], 0
    for step, orders_here in sorted(orders.items()):
        if type(step) is not int or not context.step <= step < TERMINAL:
            raise ValueError('market word contains an elapsed/nonterminal action step')
        if not isinstance(orders_here, (tuple, list)) or len(orders_here) > MAX_MARKET_ORDERS:
            raise ValueError('market word exceeds per-phase order limit')
        parsed = []
        for order in orders_here:
            if not isinstance(order, (tuple, list)) or len(order) != 3:
                raise ValueError('market-only word requires SELL/BUY_PRODUCT triples')
            op, p, n = order
            if (op not in ('SELL', 'BUY_PRODUCT') or p not in PRODUCTS
                    or (op == 'BUY_PRODUCT' and p not in BUYABLE_PRODUCTS) or type(n) is not int or n <= 0):
                raise ValueError('unsupported market-only order')
            quotes += n
            if quotes > budget.max_unit_quotes:
                raise ValueError('market word exceeds unit-quote budget')
            parsed.append([op, p, n])
        if parsed: rows.append({'step': step, 'orders': parsed})
    return {'schema': WORD_SCHEMA, 'context_key': actor_key(context), 'start_step': context.step,
            'terminal_step': TERMINAL, 'orders': rows, 'physical_actions': 'PASS_ENTIRE_SUFFIX'}


def market_proposal(context, decision_key, orders, *, budget=None):
    word = _orders_word(context, orders, budget or ValuationBudget())
    products = sorted({o[1] for row in word['orders'] for o in row['orders']})
    return Proposal('valuation', decision_key, word,
                    ('cash', 'labor') + tuple('market:' + p for p in products),
                    Requirements(window=(context.step, TERMINAL - 1)),
                    guards=('same full actor state', 'physical PASS and declared market word replace entire own suffix'),
                    invalidated_by=('any actor context or public-law change',), parents=PARENTS)


def _suffix_cash(context, word, proposal, budget):
    """Native monetary components on own state only; no environment or rival private."""
    engine = load_engine()
    _, shops, inventory = _public(context)
    own = context.observation['farms'][context.player]
    rival = context.observation['farms'][1 - context.player]
    cash, rival_cash = own['money'], rival['money']
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (cash, rival_cash)):
        raise ValueError('invalid public bank cash')
    private = copy.deepcopy(context.observation['private'])
    _stock(private['shed'])
    if not isinstance(private['inventories'], list) or len(private['inventories']) != len(own['hands']) + 1:
        raise ValueError('complete own worker custody required')
    for bag in private['inventories']: _stock(bag)
    if sum(private['shed'].values()) > DEFAULTS['shedCapacity']:
        raise ValueError('initial shed exceeds native capacity')
    farm, market = {'money': cash}, {'inventory': inventory}
    by_step = {row['step']: row['orders'] for row in word['orders']}
    quotes = 0
    for step in range(context.step, TERMINAL):
        for op, product, count in by_step.get(step, ()):
            for _ in range(count):
                if quotes >= budget.max_unit_quotes:
                    raise ValueError('suffix unit-quote budget exhausted')
                quotes += 1
                inv = market['inventory'][product] - (op == 'BUY_PRODUCT')
                price = engine.market_price(product, inv)
                if not engine._commit_unit(op, product, price, farm, private, market, DEFAULTS['shedCapacity']):
                    return SuffixResult(proposal, 'PROPOSAL_FAILED', failure='requested market fill failed at step ' + str(step))
        demand = demand_at_step(shops, step)
        for p in PRODUCTS: market['inventory'][p] -= demand[p]
        if (step + 1) % 24 == 0:
            engine._drop_inventories_to_shed(private, DEFAULTS['shedCapacity'])
            private['inventories'] = [{}]
    return SuffixResult(proposal, 'COMPLETE', farm['money'], rival_cash,
                        realization={'context_key': context_key(context), 'start_step': context.step,
                                     'terminal_step': TERMINAL, 'actions': TERMINAL - context.step,
                                     'unit_quotes': quotes, 'word_sha256': _hash(word),
                                     'scope': 'EXACT_CONDITIONAL_PUBLIC_MARKET_WORD_NOT_K0006',
                                     'ending_shed': private['shed'], 'ending_carry': private['inventories']})


def _custody_order_key(context):
    # Native nightly overflow follows inventory insertion order, unlike the
    # shared sorted-JSON context hash. Bind this value-relevant order separately.
    bags = context.observation['private']['inventories']
    if not isinstance(bags, (list, tuple)):
        raise ValueError('complete ordered own custody required')
    return _hash([list(_stock(bag).items()) for bag in bags])


def public_suffix_model(context, baseline_orders, candidates: Sequence[Proposal], *, budget=None):
    """Return the accepted endgame adapter, or None for uncertified support.

    Only a context-bound EndgameGuards.rival_pass=EXACT(1) with provenance can
    establish the public rival law. Observed inactivity and empirical flow atoms
    cannot. This function cannot authenticate the scientific truth of a caller's
    certificate. No actor or evaluator callback is accepted/captured.
    """
    budget = budget or ValuationBudget()
    key, shops, _ = _public(context)
    # The accepted endgame key requires JSON memory, not typed ProductionBook.
    if key != context_key(context):
        raise ValueError('incompatible full actor context key')
    guard = context.derived.get('endgame_guards')
    if (not isinstance(guard, EndgameGuards) or guard.context_key != key or guard.through_step != TERMINAL - 1
            or guard.rival_pass != Estimate.exact(1) or not guard.parents
            or len(shops) != MAX_SHOP_INSTANCES or TERMINAL - context.step > budget.max_suffix_actions):
        return None
    baseline_text = _json(_orders_word(context, baseline_orders, budget))
    custody_key = _custody_order_key(context)
    candidates = tuple(candidates)
    if any(not isinstance(p, Proposal) for p in candidates):
        raise ValueError('candidates must be Proposal objects')
    if len(candidates) > budget.max_candidates or len({p.key for p in candidates}) != len(candidates):
        raise ValueError('candidate budget exceeded or duplicate keys')
    allowed = {}
    clean = []
    for p in candidates:
        _wire(p.to_dict())
        payload = p.candidate
        if not isinstance(payload, dict) or payload.get('schema') != WORD_SCHEMA:
            raise ValueError('only declared market-word proposals are supported')
        rows = payload.get('orders')
        if not isinstance(rows, list) or any(not isinstance(r, dict) or set(r) != {'step', 'orders'} for r in rows):
            raise ValueError('malformed market word')
        if len({r['step'] for r in rows}) != len(rows):
            raise ValueError('duplicate market-word step')
        canonical = _orders_word(context, {r['step']: r['orders'] for r in rows}, budget)
        if payload != canonical:
            raise ValueError('market word is stale, incomplete or noncanonical')
        allowed[p.key] = _json(p.to_dict())
        clean.append(Proposal.from_dict(p.to_dict()))
    # Immutable canonical words only. In particular, no captured context, guard,
    # controller object, environment, hidden state or caller callback is retained.
    allowed = MappingProxyType(allowed)

    def evaluate(branch, proposal):
        pkey = None if proposal is None else proposal.key
        try:
            if actor_key(branch) != key or context_key(branch) != key:
                raise ValueError('stale actor continuation')
            if _custody_order_key(branch) != custody_key:
                raise ValueError('stale ordered custody continuation')
            if proposal is None:
                word = json.loads(baseline_text)
            elif pkey not in allowed or _json(proposal.to_dict()) != allowed[pkey]:
                return SuffixResult(pkey, 'PROPOSAL_FAILED', failure='unknown or changed candidate')
            else:
                word = json.loads(allowed[pkey])['candidate']
            return _suffix_cash(branch, word, pkey, budget)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            return SuffixResult(pkey, 'ERROR', failure=str(error))

    return PublicSuffixModel(key, 'PUBLIC_RIVAL_PASS__OWN_MARKET_WORD__ALL_SHOPS_PUBLISHED',
                             evaluate, tuple(clean), tuple(sorted(set(PARENTS + guard.parents))),
                             restoration_exact=True)


def calibration_table(predictions: Mapping[str, CashValue], observations: Sequence[tuple[str, SuffixResult]]):
    """Offline only: evaluate all supplied rows, retain misses, failures and abstentions.

    Coverage here is literal point/interval containment, not statistical coverage
    certification. Historical/component fixtures are not held-out playing evidence.
    Repeated keys intentionally join indistinguishable actor predictions to distinct
    evaluator outcomes. We NEVER maximize or average by hidden state for the actor.
    """
    fields = ('own_cash', 'rival_cash', 'margin')
    counts = {k: dict(observed=0, predicted=0, covered=0, missed=0, unknown=0) for k in fields}
    rows = []
    for key, realized in observations:
        if key not in predictions or not isinstance(realized, SuffixResult):
            raise ValueError('each observation needs a precomputed prediction and SuffixResult')
        predicted = predictions[key]
        row = {'key': key, 'predicted': predicted.to_dict(), 'status': realized.status,
               'failure': realized.failure, 'realized': None, 'covered': {}}
        if realized.status == 'COMPLETE':
            row['realized'] = {**{k: getattr(realized, k) for k in fields}, 'outcome': realized.outcome.value}
            for k in fields:
                value, actual = getattr(predicted, k), getattr(realized, k)
                counts[k]['observed'] += 1
                covered = None
                if value.kind is Evidence.UNKNOWN:
                    counts[k]['unknown'] += 1
                else:
                    counts[k]['predicted'] += 1
                    if value.kind is Evidence.BOUND:
                        covered = ((value.low is None or actual >= value.low) and (value.high is None or actual <= value.high))
                    else:
                        covered = actual == value.value
                    counts[k]['covered' if covered else 'missed'] += 1
                row['covered'][k] = covered
        rows.append(row)
    return {'schema': 'farm-continuation-calibration-v1', 'coverage': counts, 'rows': rows,
            'scope': 'OFFLINE_LITERAL_CONTAINMENT_NOT_FITTED_OR_HELD_OUT_COVERAGE',
            'failed_suffixes': sum(r['status'] != 'COMPLETE' for r in rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', action='store_true', help='Print supported inputs and limitations; execute no games')
    args = parser.parse_args(argv)
    if args.contract:
        print(json.dumps({'production_provider': 'PortfolioValuator.quote(context, PortfolioTape)',
                          'endgame_provider': 'public_suffix_model(context, baseline_orders, candidates)',
                          'production_evidence': 'CONDITIONAL_BOUND; missing labor/rival cash UNKNOWN',
                          'endgame_evidence': 'EXACT only under certified rival PASS, published shops and market-only word',
                          'native_games': 0, 'budget': vars(ValuationBudget()), 'parents': PARENTS}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
