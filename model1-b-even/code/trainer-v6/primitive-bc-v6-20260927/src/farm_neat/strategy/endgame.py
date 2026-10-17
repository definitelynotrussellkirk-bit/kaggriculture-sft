"""Guarded, actor-only endgame proposals; never executes or changes the fallback.

EndgameModule.solve(context) owns two consumers of the shared Proposal contract:
* On the last day, a Held--Karp/subset partition search routes existing workers and
  a budgeted number of funded hires to static collection jobs, then DROP/SELLs.
  The product-budget DP bounds receipts under certified no-arrival rival supply.
* Earlier terminal windows can consume a PublicSuffixModel in derived['counterfactual'].
  Only full terminal results from a declared actor-only deterministic model qualify;
  cut cash, sampled/hidden-world oracle results and failed branches are not values.

The last-day family excludes beneficial WATER/FERTILIZE, crop decay, overflow,
non-product custody and existing service commitments. Such states return UNKNOWN,
not a guessed rescue. It needs current EndgameGuards in derived['endgame_guards'];
absence of a certificate is NOT evidence that a rival or a commitment is absent.
These local integration types are proposals for the coordinator, not changes to types.py.

Result.status describes terminal cash evidence, NOT global optimality or strength.
The route menu is exact within its stated fixed-worker/initial-PASS-hire family;
optimal_cash separately bounds the unrestricted static collection problem. All cash
changes on this path are against PASS for the entire suffix, NOT against K0006.
The coordinator must preserve K0006 unless paired evaluation justifies adoption.
An action word is a witness, not permission to execute after a guard changes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from farm_neat.game.constants import ANIMALS, CROPS, PRODUCTS, SHOPS, shed_access_tiles
from farm_neat.game.exact_engine import DEFAULT_CONFIGURATION, EXPECTED_ENGINE_SHA256, load_engine
from farm_neat.strategy.types import (
    EVALUATOR_ONLY, LAST_STEP, Evidence, Estimate, Proposal, Requirements,
    StrategyContext, SuffixResult, UNKNOWN, difference,
)

LAST_DAY = 'd9bbf452b0ac0f1c67140006069d6b8300d5f59aab051bd6bb93d9d1f94c0af7'
H8 = '3ce373846de04a17d5da8c33676e52e796bab468988aa4b585f80fc6a8203ac4'
CALENDAR = '2a8da9a0c65979ebdc8118e22366f3af8c0828b55764de230277ecb45bfe1c9b'
CASH_BOUNDARY = '56089f5537253080d7902bee52d35fc5eca38ff2cf83c9d9830428b007349d84'
OBJECTIVES = 'ff7a551831610eaf372bedebbcd661184c898cbc3daebe2f6a829f634e32273e'
PARENTS = (LAST_DAY, H8, CALENDAR, CASH_BOUNDARY, OBJECTIVES)
TERMINAL_STEP = LAST_STEP + 1
TURNS = DEFAULT_CONFIGURATION['turnsPerDay']
SHEDS = tuple(shed_access_tiles())


@dataclass(frozen=True)
class Horizon:
    actions: int
    end_step: int
    terminal: bool


def horizon(step: int, days: int = 8) -> Horizon:
    """Actual joint actions, not a fictitious final refresh or day-29 hour-23 action."""
    if type(step) is not int or not 0 <= step <= LAST_STEP or type(days) is not int or days <= 0:
        raise ValueError('invalid step or day horizon')
    end = min(TERMINAL_STEP, step + days * TURNS)
    return Horizon(end - step, end, end == TERMINAL_STEP)


@dataclass(frozen=True)
class Budget:
    """Hard work limits, not economic rules or K0006 tuning constants."""
    max_jobs: int = 10
    max_workers: int = 10
    max_new_hires: int = 3
    max_nodes: int = 2_000_000
    max_stock: int = 100
    max_evaluations: int = 8  # Includes the baseline; each callback gets remaining actions.
    max_suffix_actions: int = 191

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < 0:
                raise ValueError(name + ' must be a nonnegative integer')
        if self.max_jobs > 16 or self.max_workers > 32 or self.max_new_hires > 10 or self.max_stock > 100:
            raise ValueError('budget exceeds portable bounded-search limits')


@dataclass(frozen=True)
class EndgameGuards:
    """Public/own-memory certificates, bound to the full actor context and horizon.

    commitments_clear=EXACT(1) releases ALL affected labor/land/cash obligations.
    no_rival_arrivals=EXACT(1) excludes rival production, buys and custody arrivals
    into sellable stock until terminal. rival_stock is a certified total sales cap,
    not a guessed private inventory and not automatically equal to shed capacity.
    rival_pass=EXACT(1) additionally fixes all rival orders/actions to PASS. A named
    public scenario law can certify that; seeing an idle farm cannot.
    Parents identify the producer proof/receipt. The module validates the shape and
    binding, not the scientific truth of the producer certificate.
    """
    context_key: str
    through_step: int
    commitments_clear: Estimate = UNKNOWN
    no_rival_arrivals: Estimate = UNKNOWN
    rival_stock: Estimate = UNKNOWN
    rival_pass: Estimate = UNKNOWN
    parents: tuple[str, ...] = ()


@dataclass(frozen=True)
class PublicSuffixModel:
    """Adapter supplied through derived, without importing another worker's module.

    evaluate(context, proposal_or_None) -> SuffixResult uses ONLY actor context and
    a declared deterministic public continuation law. It restores each branch from
    the SAME full state including own controller memory. No captured rival-private
    snapshot or realized hidden future may be closed over by this callback.
    COMPLETE realization must bind context_key, start_step and terminal_step=719.
    This contract is deliberately narrower than a general evaluator: the latter's
    hidden-world results belong in training/evaluation, never in this actor input.
    Callback resource use/restoration is the adapter's responsibility. This module
    bounds callback count and suffix length, but cannot preempt an arbitrary callable.
    """
    context_key: str
    law: str
    evaluate: Callable[[StrategyContext, Proposal | None], SuffixResult]
    candidates: tuple[Proposal, ...]
    parents: tuple[str, ...] = ()
    input_scope: str = 'ACTOR_ONLY_DETERMINISTIC'
    restoration_exact: bool = False


@dataclass(frozen=True)
class EndgameResult:
    status: Evidence
    reason: str
    proposals: tuple[Proposal, ...] = ()
    terminal_cash: Estimate = UNKNOWN
    optimal_cash: Estimate = UNKNOWN
    stats: Mapping = field(default_factory=dict)
    failures: tuple[str, ...] = ()


class _Uncertified(ValueError):
    pass


class _Fuel:
    def __init__(self, limit):
        self.limit, self.nodes = limit, 0

    def spend(self):
        if self.nodes >= self.limit:
            raise _Uncertified('search node budget exhausted; no exact claim')
        self.nodes += 1


def _wire(value):
    if isinstance(value, Mapping):
        if EVALUATOR_ONLY & set(value):
            raise _Uncertified('evaluator-only nested actor input')
        if not all(isinstance(k, str) for k in value):
            raise _Uncertified('actor input requires JSON string keys')
        return {k: _wire(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wire(v) for v in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise _Uncertified('actor context is not finite JSON')


def context_key(context: StrategyContext) -> str:
    """Includes ordered shops and full own memory, never derived certificates recursively."""
    payload = _wire({'step': context.step, 'player': context.player,
                     'observation': context.observation, 'memory': context.memory,
                     'shops': context.shops})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _yes(value):
    return value == Estimate.exact(1)


def _count(value, name):
    if type(value) is not int or value < 0:
        raise _Uncertified(name + ' must be a nonnegative integer')
    return value


def _stock(value):
    if not isinstance(value, Mapping) or set(value) - (set(PRODUCTS) | set(ANIMALS)):
        raise _Uncertified('non-product or missing custody')
    result = {}
    for product, count in value.items():
        _count(count, 'stock')
        if product not in PRODUCTS and count:
            raise _Uncertified('positive unsellable animal custody')
        if count:
            result[product] = count
    return result


def _add(*stocks):
    result = {}
    for stock in stocks:
        for p, n in stock.items():
            result[p] = result.get(p, 0) + n
    return {p: n for p, n in result.items() if n}


def _sell(product, inventory, count):
    engine = load_engine()
    cash = 0
    for _ in range(count):
        price = engine.market_price(product, inventory)
        cash += price
        inventory += int(price > 1)  # $1-floor sales do not increase supply.
    return cash, inventory


def stock_envelope(market: Mapping[str, int], own: Mapping[str, int], rival_units: int,
                   *, _fuel=None) -> Estimate:
    """Product-separable prior-stock DP: a certified interval, NOT a future-flow forecast.

    Rival all-before is the adverse relaxation of lockstep selling; no rival supply
    is the friendly endpoint. With no arrivals and a total backed sales cap, town
    depletion before terminal cannot make an earlier sale worse than this ordering.
    The bounds are sharp for the declared allocation/prior-sale relaxation, not a
    reachability claim for every allocation in a real opponent's private history.
    """
    _count(rival_units, 'rival stock cap')
    if rival_units > 100:
        raise ValueError('stock DP cap exceeds 100; increase only with a new resource contract')
    own = _stock(own)
    if sum(own.values()) > 100:
        raise ValueError('terminal own stock exceeds shed capacity')
    fuel = _fuel or _Fuel(1_000_000)
    dp = [0] + [-math.inf] * rival_units
    friendly = 0
    for product, count in sorted(own.items()):
        inventory = market[product]
        if type(inventory) is not int:
            raise ValueError('market inventory must be an integer (possibly negative)')
        base, _ = _sell(product, inventory, count)
        friendly += base
        damage = []
        for rival in range(rival_units + 1):
            _, after = _sell(product, inventory, rival)
            value, _ = _sell(product, after, count)
            damage.append(base - value)
        nxt = [-math.inf] * (rival_units + 1)
        for used, value in enumerate(dp):
            if value == -math.inf:
                continue
            for extra in range(rival_units - used + 1):
                fuel.spend()
                nxt[used + extra] = max(nxt[used + extra], value + damage[extra])
        dp = nxt
    return Estimate.bound(friendly - max(dp), friendly)


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _home(pos):
    return min(SHEDS, key=lambda p: (_distance(pos, p), SHEDS.index(p)))


def _walk(start, end):
    commands = []
    x, y = start
    tx, ty = end
    commands.extend([('EAST',)] * max(0, tx - x))
    commands.extend([('WEST',)] * max(0, x - tx))
    commands.extend([('SOUTH',)] * max(0, ty - y))
    commands.extend([('NORTH',)] * max(0, y - ty))
    return commands


@dataclass(frozen=True)
class _Job:
    xy: tuple[int, int]
    command: str
    product: str
    units: int


def _jobs(farm):
    jobs = []
    for y, row in enumerate(farm['tiles']):
        for x, tile in enumerate(row):
            if not isinstance(tile, Mapping):
                continue
            units = _count(tile.get('yield_units', 0), 'tile yield')
            if tile.get('kind') == 'PLANT':
                crop = CROPS[tile['crop']]
                age = LAST_STEP // TURNS - tile['planted_day']
                # Fertilizing then watering can increase a one-time yield too.
                bonus = (not crop['ongoing'] and not tile['watered_today']
                         and (crop['max_yield_day'] + 1) // 2 <= age <= crop['max_yield_day']
                         and units < crop['max_yield'])
                if bonus:
                    raise _Uncertified('beneficial WATER/FERTILIZE outside static collection guard')
                if units and age >= crop['first_yield_day']:
                    expiry = tile['max_lifespan_step']
                    if 0 <= expiry <= LAST_STEP:
                        raise _Uncertified('crop decay invalidates static route jobs')
                    jobs.append(_Job((x, y), 'HARVEST', tile['crop'], units))
            elif 'animal' in tile:
                if units:
                    jobs.append(_Job((x, y), 'HARVEST', ANIMALS[tile['animal']]['product'], units))
                if tile.get('fertilizer_available'):
                    jobs.append(_Job((x, y), 'COLLECT_FERTILIZER', 'FERTILIZER', 1))
    return tuple(jobs)


def _routes(start, slots, jobs, fuel):
    """Held--Karp shortest route per job subset, including every service and final DROP."""
    routes = {0: ()}
    if slots < _distance(start, _home(start)) + 1:
        return routes
    states = {}
    for j, job in enumerate(jobs):
        cost = _distance(start, job.xy) + 1
        if cost + _distance(job.xy, _home(job.xy)) + 1 <= slots:
            fuel.spend()
            states[(1 << j, j)] = (cost, (j,))
    for mask in range(1, 1 << len(jobs)):
        ends = [(j, states[(mask, j)]) for j in range(len(jobs)) if (mask, j) in states]
        for j, (cost, path) in ends:
            if mask not in routes or (cost + _distance(jobs[j].xy, _home(jobs[j].xy)), path) < routes[mask][0]:
                routes[mask] = ((cost + _distance(jobs[j].xy, _home(jobs[j].xy)), path), path)
            for k, nxt in enumerate(jobs):
                if mask & (1 << k):
                    continue
                fuel.spend()
                new_cost = cost + _distance(jobs[j].xy, nxt.xy) + 1
                if new_cost + _distance(nxt.xy, _home(nxt.xy)) + 1 > slots:
                    continue
                key = (mask | (1 << k), k)
                value = (new_cost, path + (k,))
                if key not in states or value < states[key]:
                    states[key] = value
    return {mask: (() if mask == 0 else value[1]) for mask, value in routes.items()}


def _route_word(start, slots, path, jobs, cargo):
    if not path and not cargo:
        return [('PASS',)] * slots
    cursor, commands = start, []
    for j in path:
        job = jobs[j]
        commands.extend(_walk(cursor, job.xy)); commands.append((job.command,))
        cursor = job.xy
    commands.extend(_walk(cursor, _home(cursor)))
    if len(commands) + 1 > slots:
        return [('PASS',)] * slots  # Only an unreachable original empty-subset cargo route.
    return commands + [('PASS',)] * (slots - len(commands) - 1) + [('DROP',)]


class EndgameModule:
    name = 'endgame'

    def __init__(self, budget: Budget | None = None, configuration: Mapping | None = None):
        self.budget = budget or Budget()
        self.configuration = dict(DEFAULT_CONFIGURATION)
        if configuration is not None:
            self.configuration.update(configuration)

    def propose(self, context: StrategyContext) -> Sequence[Proposal]:
        return self.solve(context).proposals

    def solve(self, context: StrategyContext) -> EndgameResult:
        fuel = _Fuel(self.budget.max_nodes)
        try:
            key = context_key(context)
            obs = context.observation
            if any(obs.get(k) != v for k, v in (('step', context.step), ('player', context.player),
                                                ('day', context.step // TURNS), ('hour', context.step % TURNS))):
                raise _Uncertified('context and observation clocks/seats disagree')
            if self.configuration != DEFAULT_CONFIGURATION:
                raise _Uncertified('non-default configuration lacks a terminal proof')
            if context.step // TURNS == LAST_STEP // TURNS:
                return self._last_day(context, key, fuel)
            return self._suffix(context, key)
        except (_Uncertified, KeyError, TypeError, IndexError, ValueError) as error:
            return EndgameResult(Evidence.UNKNOWN, str(error), stats={'nodes': fuel.nodes})

    def _last_day(self, context, key, fuel):
        guard = context.derived.get('endgame_guards')
        if not isinstance(guard, EndgameGuards) or guard.context_key != key or guard.through_step != LAST_STEP or not guard.parents:
            raise _Uncertified('missing/stale full-horizon endgame certificates')
        if not _yes(guard.commitments_clear):
            raise _Uncertified('committed service/cash not certified clear; preserve fallback')
        if not _yes(guard.no_rival_arrivals):
            raise _Uncertified('no rival arrivals not certified; shed capacity alone is insufficient')
        supply = guard.rival_stock
        cap = supply.value if supply.kind is Evidence.EXACT else supply.high if supply.kind is Evidence.BOUND else None
        if cap is None or cap < 0 or cap != int(cap) or cap > self.budget.max_stock:
            raise _Uncertified('missing or over-budget certified rival total sales cap')
        cap = int(cap)
        if _yes(guard.rival_pass) and cap != 0:
            raise _Uncertified('contradictory PASS and nonzero rival stock certificates')
        obs = context.observation
        if tuple(obs['town']['unlocked_shops']) != context.shops or any(s not in SHOPS for s in context.shops):
            raise _Uncertified('ordered published-shop prefix mismatch')
        farm = copy.deepcopy(obs['farms'][context.player])
        if len(farm['tiles']) != 10 or any(len(row) != 10 for row in farm['tiles']):
            raise _Uncertified('nonstandard board')
        starts = [tuple(farm['farmer']), *map(tuple, farm['hands'])]
        if any(len(p) != 2 or any(type(x) is not int or not 0 <= x < 10 for x in p) for p in starts):
            raise _Uncertified('invalid worker positions')
        if len(starts) > self.budget.max_workers:
            raise _Uncertified('worker search budget exceeded')
        custody = [_stock(s) for s in obs['private']['inventories']]
        if len(custody) != len(starts):
            raise _Uncertified('worker/custody count mismatch')
        shed = _stock(obs['private']['shed'])
        jobs = _jobs(farm)
        if len(jobs) > self.budget.max_jobs:
            raise _Uncertified('static collection job budget exceeded')
        all_stock = _add(shed, *custody, *({j.product:j.units} for j in jobs))
        if sum(all_stock.values()) > self.budget.max_stock:
            raise _Uncertified('possible terminal overflow requires a different sale/storage schedule')
        money = farm['money']
        if isinstance(money, bool) or not isinstance(money, (int,float)) or not math.isfinite(money) or money < 0:
            raise _Uncertified('invalid bank cash')
        _count(farm['hires_today'], 'hires_today')
        if obs['market'].get('params') not in (None, load_engine().MARKET_PARAMS):
            raise _Uncertified('non-default public market parameters lack a quote certificate')
        market = dict(obs['market']['inventory'])
        if set(market) != set(PRODUCTS) or any(type(n) is not int for n in market.values()):
            raise _Uncertified('full public integer market inventory required')
        # Native order: sale precedes current-step consumption, so terminal-step demand
        # cannot improve the terminal sale quote. No new shops publish on this suffix.
        for step in range(context.step, LAST_STEP):
            if step % DEFAULT_CONFIGURATION['townShopSellInterval'] == 0:
                for shop in context.shops:
                    for p in SHOPS[shop]:
                        market[p] -= 2 if len(SHOPS[shop]) == 1 else 1
            if step % DEFAULT_CONFIGURATION['townCenterSellInterval'] == 0:
                for p in PRODUCTS:
                    if p != 'FERTILIZER':
                        market[p] -= 1
        def revenue(stock):
            return sum(_sell(p, market[p], q)[0] for p,q in stock.items())
        n = len(jobs)
        stock_by_mask = [{}]
        for mask in range(1, 1 << n):
            bit = mask & -mask
            j = jobs[bit.bit_length()-1]
            stock_by_mask.append(_add(stock_by_mask[mask ^ bit], {j.product:j.units}))
        slots = TERMINAL_STEP - context.step
        best = None
        menu_masks = 0
        # Buying labor is a bounded menu, never a fixed three-hire rule. Existing units
        # PASS on that first turn so post-unit-phase spawn geometry is exactly modeled.
        for hires in range(min(self.budget.max_new_hires, self.budget.max_workers-len(starts)) + 1):
            if hires and slots < 2:
                break
            virtual = copy.deepcopy(farm)
            own = copy.deepcopy(obs['private'])
            cost = 0
            for _ in range(hires):
                charge = load_engine()._hire_cost(virtual['hires_today'])
                if virtual['money'] < charge:
                    break
                load_engine()._do_hire(virtual, own, 10)
                cost += charge
            if len(virtual['hands']) != len(farm['hands']) + hires:
                break
            positions = [tuple(virtual['farmer']), *map(tuple,virtual['hands'])]
            delay = int(hires > 0)
            available = slots - delay
            reachable_carry = [s if _distance(p,_home(p))+1 <= available else {}
                               for p,s in zip(positions,custody + [{}]*hires)]
            base = _add(shed,*reachable_carry)
            allocations = {0: ()}
            for p in positions:
                routes = _routes(p, available, jobs, fuel)
                after = {}
                for used, paths in allocations.items():
                    for mask, path in routes.items():
                        fuel.spend()
                        if not used & mask:
                            union = used | mask
                            # Given total stock fits, worker identity/path has no price
                            # effect: keep one shortest constructive witness per union.
                            if union not in after:
                                after[union] = paths + (path,)
                allocations = after
            menu_masks += len(allocations)
            for mask, paths in allocations.items():
                fuel.spend()
                stock = _add(base, stock_by_mask[mask])
                gain = revenue(stock) - cost
                rank = (gain, -hires, -mask)
                if best is None or rank > best[0]:
                    best = (rank, hires, cost, positions, reachable_carry, paths, stock)
        if best is None:
            raise _Uncertified('no evaluated route menu')
        _, hires, cost, positions, cargo, paths, stock = best
        env = stock_envelope(market, stock, cap, _fuel=fuel)
        own_delta = Estimate.exact(env.high-cost) if cap == 0 else Estimate.bound(env.low-cost, env.high-cost)
        terminal = Estimate.exact(money+own_delta.value) if cap == 0 else Estimate.bound(money+own_delta.low,money+own_delta.high)
        attainable = terminal.value if cap == 0 else terminal.low
        relaxed = money + revenue(all_stock)
        optimal = Estimate.exact(attainable) if attainable == relaxed else Estimate.bound(attainable,relaxed)
        stats = {'nodes':fuel.nodes,'jobs':len(jobs),'route_menu_masks':menu_masks,'hires':hires,
                 'native_games':0,'evaluations':0}
        if (own_delta.value if cap == 0 else own_delta.high) <= 0:
            return EndgameResult(terminal.kind,'PASS: no cash-positive collection in bounded menu',
                                 terminal_cash=terminal,optimal_cash=optimal,stats=stats)
        delay = int(hires > 0)
        words = [[('PASS',)]*delay + _route_word(p,slots-delay,path,jobs,c)
                 for p,path,c in zip(positions,paths,cargo)]
        actions = []
        for offset in range(slots):
            orders = [['HIRE'] for _ in range(hires)] if offset == 0 else []
            if offset == slots-1:
                # Preserve frozen final-liquidation exposure ordering without importing it.
                ordered = sorted(stock, key=lambda p: (-load_engine().market_price(p, market[p]) * stock[p],
                                                        list(PRODUCTS).index(p)))
                orders += [['SELL',p,stock[p]] for p in ordered]
            hand_count = len(starts)-1 if hires and offset == 0 else len(positions)-1
            actions.append({'farmer':list(words[0][offset]),
                            'hands':[list(words[i+1][offset]) for i in range(hand_count)],'market':orders})
        rival_delta = Estimate.exact(0) if _yes(guard.rival_pass) else UNKNOWN
        parents = tuple(sorted(set(PARENTS + guard.parents)))
        guards = ('default pinned engine/configuration', 'day has no remaining refresh',
                  'static yields: no decay or beneficial WATER/FERTILIZE',
                  'all terminal stock fits the shed; no discarded DROP units',
                  'all affected service/cash commitments certified clear',
                  'no rival arrivals; total backed rival sales <= '+str(cap),
                  'context_key='+key)
        if _yes(guard.rival_pass):
            guards += ('rival PASS certified for the complete suffix',)
        touches = ('cash','labor','land') + tuple('market:'+p for p in stock) + tuple('inventory:'+p for p in stock)
        candidate = {'family':'collection_liquidation','start_step':context.step,'terminal_step':TERMINAL_STEP,
                     'actions':actions,'context_key':key,'control':'PASS_ENTIRE_SUFFIX',
                     'scope':'static collection; existing workers or funded initial-PASS hires within budget',
                     'terminal_cash':terminal.to_dict(),'optimal_cash':optimal.to_dict(),
                     'engine_sha256':EXPECTED_ENGINE_SHA256,'budget':vars(self.budget),
                     'execution':'coordinator revalidates guards; witness is not unconditional adoption'}
        req = Requirements(cash=Estimate.exact(cost),reserve=Estimate.exact(0),
                           inventory={p:Estimate.exact(q) for p,q in shed.items()},
                           market_slots=Estimate.exact(len(stock)+hires),
                           worker_hours=Estimate.exact(sum(c[0]!='PASS' for word in words for c in word)),
                           land_quadrants=Estimate.exact(0),window=(context.step,LAST_STEP))
        p = Proposal(self.name,'collect:'+key[:16],candidate,touches,req,own_delta,rival_delta,
                     difference(own_delta,rival_delta),guards,
                     ('any actor observation or own memory change','rival flow/commitment certificate invalidation'),parents)
        return EndgameResult(terminal.kind,'conditional static collection; no K0006 improvement claimed',
                             (p,),terminal,optimal,stats)

    def _suffix(self, context, key):
        model = context.derived.get('counterfactual')
        if TERMINAL_STEP-context.step > self.budget.max_suffix_actions or not horizon(context.step).terminal:
            raise _Uncertified('window stops before terminal: cut cash is not continuation value')
        if not isinstance(model, PublicSuffixModel) or model.context_key != key or not model.parents:
            raise _Uncertified('complete actor-only suffix model not available or stale')
        if (model.input_scope != 'ACTOR_ONLY_DETERMINISTIC' or model.restoration_exact is not True
                or not isinstance(model.law, str) or not model.law.strip() or not callable(model.evaluate)):
            raise _Uncertified('suffix model law/restoration not certified')
        if self.budget.max_evaluations < 1:
            raise _Uncertified('no baseline evaluation budget')
        if not all(isinstance(p, Proposal) for p in model.candidates):
            raise _Uncertified('suffix candidates must be Proposal objects')
        keys = [p.key for p in model.candidates]
        if len(keys) != len(set(keys)):
            raise _Uncertified('duplicate candidate keys')
        calls, failures = 0, []
        def evaluate(p):
            nonlocal calls
            calls += 1
            try:
                # Give the adapter fresh actor data and a fresh candidate each time. Derived
                # objects (including this callable) are not actor state or replay memory.
                branch = StrategyContext(context.step, context.player, _wire(context.observation),
                                         _wire(context.memory), context.shops)
                branch_proposal = None if p is None else Proposal.from_dict(p.to_dict())
                result = model.evaluate(branch, branch_proposal)
            except Exception as error:
                failures.append(type(error).__name__+': '+str(error))
                return None
            if not isinstance(result,SuffixResult) or result.status != 'COMPLETE':
                failures.append('failed suffix: '+str(getattr(result,'failure','invalid result')))
                return None
            r = result.realization
            if (result.proposal != (None if p is None else p.key) or not isinstance(r,dict)
                    or r.get('context_key') != key or r.get('start_step') != context.step
                    or r.get('terminal_step') != TERMINAL_STEP):
                failures.append('suffix is not bound to this complete terminal continuation')
                return None
            if context_key(branch) != key:
                failures.append('evaluator mutated actor inputs')
                return None
            return result
        baseline = evaluate(None)
        if baseline is None:
            return EndgameResult(Evidence.UNKNOWN,'baseline lacks valid complete terminal result',
                                 stats={'evaluations':calls},failures=tuple(failures))
        def rank(r):
            # Competitive result first; margin is an explicitly declared tie-break,
            # never a claim of improved WDL when both complete branches lose.
            return ((r.margin>0)-(r.margin<0),r.margin)
        best, chosen = baseline, None
        for p in model.candidates[:max(0,self.budget.max_evaluations-1)]:
            result = evaluate(p)
            if result is not None and rank(result)>rank(best):
                best,chosen=result,p
        stats={'evaluations':calls,'unexamined_candidates':len(model.candidates)-(calls-1),
               'failed_candidates':len(failures),'native_games':None,
               'resource_note':'callback resource use is adapter-owned, not observed here'}
        if chosen is None:
            return EndgameResult(Evidence.BOUND,'KEEP: no evaluated WDL/margin improvement',
                                 terminal_cash=Estimate.exact(baseline.own_cash),stats=stats,failures=tuple(failures))
        own=Estimate.exact(best.own_cash-baseline.own_cash)
        rival=Estimate.exact(best.rival_cash-baseline.rival_cash)
        candidate={'family':'complete_suffix','adopt':chosen.to_dict(),'context_key':key,
                   'law':model.law,'search_status':'BOUND','objective':['terminal_WDL','margin_tiebreak'],
                   'control':{'own_cash':baseline.own_cash,'rival_cash':baseline.rival_cash,
                              'margin':baseline.margin,'outcome':baseline.outcome.value},
                   'result':{'own_cash':best.own_cash,'rival_cash':best.rival_cash,
                             'margin':best.margin,'outcome':best.outcome.value},
                   'start_step':context.step,'terminal_step':TERMINAL_STEP}
        p=Proposal(self.name,'suffix:'+key[:16],candidate,chosen.touches,chosen.requirements,
                   own,rival,difference(own,rival),chosen.guards+('actor-only deterministic law='+model.law,
                   'each branch restores full own continuation memory','complete terminal record; no cut mark'),
                   tuple(sorted(set(chosen.invalidated_by+('any actor context change',)))),
                   tuple(sorted(set(PARENTS+chosen.parents+model.parents))))
        return EndgameResult(Evidence.BOUND,'bounded candidate search; exact deltas conditional on declared public model',
                             (p,),Estimate.exact(best.own_cash),stats=stats,failures=tuple(failures))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizon',type=int,help='Print the exact eight-day endpoint for an acting step')
    args=parser.parse_args(argv)
    if args.horizon is not None:
        print(json.dumps(vars(horizon(args.horizon)),sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
