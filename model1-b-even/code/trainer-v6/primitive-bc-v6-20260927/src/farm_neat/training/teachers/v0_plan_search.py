"""Bounded canonical V0 Plan search on the existing router_panel game evaluator.

register freezes train/held-out worlds, opponents, seats, sources and TOTAL budget.
run resumes deterministically; every game is followed by an independent process
replay. Both attempts count, even for identical plans. Unknown/torn attempts fail
closed, never silently replay. A substituted test runner is SYNTHETIC_DIAGNOSTIC.

Plans are PRELOADED into V0Agent(labor_factor=4, hands_mode='daily', KeepPolicy).
They are NOT certified reachable from the empty V0 decoder and are NOT BC labels.
$60,753 is historical opening-only compiled_daily evidence, not this route's cash.
Projection infeasibility is diagnostic: the retained baseline exceeds its hand
bound but compiles. Actual refused commands/unplaced instances prevent promotion.

All registered worlds have unrevealed shops at opening, so there is ONE public
world class. No hidden shop, seed or world-ID dispatch reaches the Plan actor.
Selection uses complete paired TRAIN margins only, freezes before held-out play,
and reports every held-out pair against the same baseline and K0006 controls.
Fixed exposed worlds are not ladder strength. This is a search subset consumer of
router_panel.play_game/validate_game, NOT its separate full 80-row-regression gate.

Example four-game reviewer smoke (two train games plus their two replays):
  PYTHONPATH=src python3 -B -m farm_neat.training.teachers.v0_plan_search register \
    --output work/runs/v0-plan-search-smoke --train router-dairy \
    --heldout pair-03-04 --seats 0 --opponents router --trials 1 --max-games 4
  PYTHONPATH=src python3 -B -m farm_neat.training.teachers.v0_plan_search run \
    --output work/runs/v0-plan-search-smoke --max-games 4

For a full independent router_panel gate, set FARM_V0_SEARCH_SELECTION to the
frozen selection.json and use module:searched_actor with module:baseline_actor.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import sys
import time
from typing import Callable

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.epochs import KeepPolicy, V0Agent
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import (
    DEMAND_SPEC,
    DEMAND_STOP,
    Masks,
    Projection,
    PublicDemandHistory,
    State,
    frontier as market_frontier,
)
from farm_neat.evaluation import router_panel as panel
from farm_neat.game.constants import PRICE_FLOOR, SHOPS, TOWN_CENTER_PRODUCTS
from farm_neat.evaluation.common import bound, canonical, digest, equal, file_sha, read, write_new

ROOT = panel.ROOT
SCHEMA = 'farm-v0-plan-search-v1'
MODULE = 'farm_neat.training.teachers.v0_plan_search'
ROUNDTRIP = 'work/runs/reviewer-play/action-space-v0/ROUNDTRIP.json'
ROUNDTRIP_SHA = '0c78d1bb74183895df1e485f42748abd3514e630abe55ecf3a62735a83f35754'
BASELINE_SHA = '76b1a17279847c4018379b7c81b667f7e5d3427bace85205df8d087cf71b6726'
ROUTE = 'SEEDED_V0_AGENT_DAILY_KEEP'
FAMILIES = ('PROD', 'SERVICE', 'LAND', 'MARKET')
PROPOSAL_ORDERS = ('control', 'market_capture')
MARKET_HORIZONS = ('near', 'mid', 'late')
CRN_SCHEDULES = 'project/configs/v0-search-crn-schedules.json'
CRN_SCHEDULES_SHA = '9301fd90d2fde7577d84f33bd6e680548ae160d21649e950fce9ef582ee978a9'
FINAL_SELECTION_AUTHORITY = 'FRESH_ORIGINAL_RULE_PAIRED_TERMINAL_WDL_REQUIRED'
ACTOR_MARKET_INPUTS = (
    'public_observation', 'public_demand_history', 'known_remaining',
    'expected_unrevealed', 'uniform_unrevealed_shop_prior',
    'remaining_plan_production',
)
SAFE_ACTOR_MARKET_FEATURES = frozenset(ACTOR_MARKET_INPUTS)
FORBIDDEN_ACTOR_MARKET_FEATURES = frozenset({
    'future_shops', 'shop_history', 'world_seed', 'seed', 'schedule',
    'schedule_sha256', 'endpoint_inventory', 'outcomes', 'wdl',
})


def strict_plan(value: dict) -> Plan:
    """Do not let Plan's permissive int coercion admit malformed search inputs."""
    if not isinstance(value, dict) or set(value) != {
            'schema', 'action_schema_sha256', 'cohorts', 'regimes', 'land', 'market'}:
        raise ValueError('exact canonical Plan fields required')
    if not isinstance(value['cohorts'], dict) or any(
            type(n) is not int or not 1 <= n <= 100 for n in value['cohorts'].values()):
        raise ValueError('canonical positive integer cohort count in [1,100] required')
    try:
        plan = Plan.from_canonical(value)
        equal(plan.canonical(), value, 'canonical Plan round trip')
        if any(not V.regime_legal(m, plan.regime(m)) for m in plan.cohorts):
            raise ValueError('unsupported service regime')
        if plan.land == V.LAND_TIMINGS.index('LAST_PROFITABLE'):
            raise ValueError('reserved land timing')
        for overlay in value['market'].values():
            if not isinstance(overlay, list) or type(overlay[2]) is not int:
                raise ValueError('canonical market overlay required')
        if any(V.PRIMITIVES[a] in ('FRONTRUN', 'PIGGYBACK') for a, _, _ in plan.market.values()):
            raise ValueError('reserved market primitive')
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('invalid canonical Plan') from error
    return plan


def retained_baseline() -> tuple[Plan, dict]:
    equal(file_sha(ROOT / ROUNDTRIP), ROUNDTRIP_SHA, 'retained baseline source')
    item = read(ROOT / ROUNDTRIP, binding=False)['portfolios']['greedy_own_39']['by_labor_factor']['4.0']
    plan = strict_plan(item['plan'])
    equal(plan.digest(), BASELINE_SHA, 'retained baseline Plan')
    return plan, {'path': ROUNDTRIP, 'sha256': ROUNDTRIP_SHA, 'portfolio': 'greedy_own_39',
                  'historical_own_cash': 60753, 'historical_route': 'compiled_daily',
                  'historical_scope': 'router-dairy; frozen Router; both seats; attestation, not this run',
                  'execution_route': ROUTE, 'training_admission': False}


def source_pins() -> dict:
    """Pin this consumer and its existing compiler, executor, engine and opponents."""
    paths = {Path(__file__).resolve(), ROOT / ROUNDTRIP, ROOT / CRN_SCHEDULES, panel.scenarios.CATALOG,
             Path(panel.__file__), Path(panel.scenarios.__file__), Path(panel.exact_engine.__file__),
             panel.exact_engine.ENGINE_PATH, panel.exact_engine.ENGINE_CONFIG_PATH}
    for name in ('common.py', 'opponents.py', 'shop_router.py', 'k0006.py'):
        paths.add(ROOT / 'src/farm_neat/evaluation' / name)
    for folder in ('game', 'economy', 'evaluation/shop_router_0909', 'evaluation/k0006_v43'):
        paths.update(p for p in (ROOT / 'src/farm_neat' / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.json', '.toml'))
    return {p.relative_to(ROOT).as_posix(): file_sha(p) for p in sorted(paths)}


def check_sources(pins: dict):
    equal(source_pins(), pins, 'source closure (changed, missing or added dependency)')


class PlanActor(V0Agent):
    """Fresh actual V0Agent; no trainer registry/world metadata is given to it."""
    def __init__(self, plan: Plan):
        plan = strict_plan(plan.canonical())
        super().__init__(KeepPolicy(), plan=plan, labor_factor=4.0, hands_mode='daily',
                         belief=None, game_id='v0-search-' + plan.digest()[:16])
        self.identity = {'kind': ROUTE, 'plan': plan.canonical(), 'plan_sha256': plan.digest(),
                         'labor_factor': 4.0, 'hands_mode': 'daily', 'belief': 'ZERO_FLOW',
                         'training_admission': False, 'assets': source_pins()}


def baseline_actor():
    """Factory accepted by the unmodified full router_panel command."""
    return PlanActor(retained_baseline()[0])


def searched_actor():
    """Full-panel factory; only the frozen canonical Plan enters the actor."""
    name = os.environ.get('FARM_V0_SEARCH_SELECTION')
    if not name:
        raise ValueError('FARM_V0_SEARCH_SELECTION must name a frozen selection.json')
    selection = read(Path(name))
    if selection.get('schema') != SCHEMA + '-selection':
        raise ValueError('source-bound V0 search selection required')
    if selection.get('evidence_kind') != 'NATIVE_GAME':
        raise ValueError('native NATIVE_GAME selection required for the full panel factory')
    check_sources(selection['source_pins'])
    actor = PlanActor(strict_plan(selection['plan']))
    actor.identity['selection_sha256'] = selection['content_sha256']
    actor.identity['assets'][str(Path(name).resolve())] = file_sha(name)
    return actor


def actor_spec(plan: dict | None):
    if plan is None:
        return panel.load_actor('k0006')
    value = strict_plan(plan)
    return panel.register_actor('v0-' + value.digest(), lambda: PlanActor(value))


def opening_observation(world: str):
    return panel.scenarios.ScenarioEnvironment(panel.world_registry()[world].scenario()).agent_observation(0)


def screen_plan(plan: Plan, observation: dict) -> dict:
    """A public opening compile, not a physical game or feasibility certificate."""
    try:
        strict_plan(plan.canonical())
        if observation.get('step') != 0:
            raise ValueError('search preloads opening Plans only')
        state = State(observation)
        for product, overlay in plan.market.items():
            before = plan.copy(); before.market.pop(product, None)
            if not Masks(state, before, labor_factor=4.0).legal(V.encode('MARKET', product, *overlay)):
                raise ValueError('masked initial market overlay')
        actor = PlanActor(plan)
        actor(copy.deepcopy(observation), {})  # no env.step: ZERO native turns
        record = actor.records[0]
        status = 'COMPILES' if not record['rejected'] and not record['unplaced'] else 'UNCOMPILABLE'
        return {'status': status, 'plan_sha256': plan.digest(), 'projection_binding': record['binding'],
                'reserve': record['reserve'], 'commands': record['commands'],
                'rejected_commands': record['rejected'], 'unplaced': record['unplaced'],
                'records_sha256': digest(actor.records), 'physical_games': 0}
    except (ValueError, TypeError, KeyError, IndexError) as error:
        return {'status': 'UNCOMPILABLE', 'plan_sha256': plan.digest(), 'error': str(error), 'physical_games': 0}



def _prior_shop_rate(product: str) -> float:
    """Expected shop withdrawals per active unrevealed draw at one four-step tick."""
    return math.fsum(2 if len(items) == 1 else 1
                     for items in SHOPS.values() if product in items) / len(SHOPS)


def _expected_demand_at(state: State, product: str, step: int) -> tuple[float, float]:
    """Known public and prior-only demand created at one future transition."""
    known = 0.0
    if step % 4 == 0:
        known += sum((2 if len(SHOPS[shop]) == 1 else 1)
                     for shop in state.shops if product in SHOPS[shop])
    if step % 24 == 0 and product in TOWN_CENTER_PRODUCTS:
        known += 1.0
    expected = 0.0
    if step % 4 == 0:
        future_active = sum(reveal <= step
                            for reveal in DEMAND_SPEC['reveal_steps'][len(state.shops):])
        expected = future_active * _prior_shop_rate(product)
    return known, expected


def _expected_sell_value(projection: Projection, product: str, inventory: float,
                         quantity: int) -> tuple[float, float]:
    """Apply the exact integer sale rule with a fractional mean-path interpolation.

    The integer endpoints use ``Projection.sell_value`` exactly.  Interpolating
    between them is an accounting proxy for fractional expected inventory, not
    the exact expectation of stochastic market revenue.
    """
    if type(quantity) is not int or quantity < 0:
        raise ValueError('nonnegative integer sale quantity required')
    lo, hi = math.floor(inventory), math.ceil(inventory)
    lo_value, lo_after = projection.sell_value(product, lo, quantity)
    if lo == hi:
        return float(lo_value), float(lo_after)
    hi_value, hi_after = projection.sell_value(product, hi, quantity)
    weight = inventory - lo
    return ((1.0 - weight) * lo_value + weight * hi_value,
            (1.0 - weight) * lo_after + weight * hi_after)


def _fractional_sell_value(projection: Projection, product: str, inventory: float,
                           quantity: float) -> float:
    """Integer-rule marginal revenue with a linear fractional-unit proxy."""
    if not isinstance(quantity, (int, float)) or not math.isfinite(quantity) or quantity < 0:
        raise ValueError('nonnegative finite sale quantity required')
    whole = math.floor(quantity)
    value, after = _expected_sell_value(projection, product, inventory, whole)
    fraction = quantity - whole
    if fraction:
        next_value, _ = _expected_sell_value(projection, product, after, 1)
        value += fraction * next_value
    return value


def _planned_sale_events(state: State, projection: Projection) -> dict[str, list[tuple[int, int]]]:
    """Compiler-aligned own-stock offers; projected supply, never realized sales.

    Daily production/input tapes determine how much *owned* stock can reach the
    market.  Overlay timing follows the compiler contract: CLEAR is continuously
    active, HOLD releases at its bound, SELL emits the selected frontier quantity
    then releases the remainder at the next known demand, and CARRY's purchased
    units are deliberately not relabelled as production supply.
    """
    events = {product: [] for product in V.PRODUCTS}

    def emit(product: str, step: int, quantity: int):
        if quantity <= 0 or step >= DEMAND_STOP:
            return
        if events[product] and events[product][-1][0] == step:
            prior_step, prior_quantity = events[product][-1]
            events[product][-1] = (prior_step, prior_quantity + quantity)
        else:
            events[product].append((step, quantity))

    for product_index, product in enumerate(V.PRODUCTS):
        primitive, horizon, rank = projection.plan.overlay(product_index)
        kind = V.PRIMITIVES[primitive]
        partial_step = partial_quantity = None
        continuous_step = None
        if kind == 'CLEAR':
            continuous_step = state.step
        elif kind == 'HOLD':
            release = projection.hold_release(product_index)
            continuous_step = state.step if release is None else max(state.step, release)
        elif kind in ('SELL', 'CARRY'):
            target = state.horizon_step(product_index, horizon)
            points = market_frontier(state, projection, product_index, primitive, horizon)
            if target is None or rank >= len(points):
                continuous_step = state.step  # compiler's invalid/absent-horizon fallback
            else:
                quantity = int(points[rank]['q'] or 0)
                if kind == 'SELL':
                    partial_step, partial_quantity = target, quantity
                    later = state.next_demand(product, target) or min(target + 1, V.LAST_STEP)
                    continuous_step = later if later > target else None
                elif quantity:
                    # CARRY buys q at target and starts a blanket sale at target+1.  Only
                    # stock already owned by the plan is credited here; the bought q is not.
                    continuous_step = min(target + 1, V.LAST_STEP)
                else:
                    continuous_step = state.step
        else:  # reserved primitives are rejected by strict_plan; preserve compiler fallback.
            continuous_step = state.step

        stock = int(state.stock[product])
        if state.hour == 0:
            if product == 'WHEAT':
                stock = max(0, stock - projection.tapes.wheat[state.day])
            elif product == 'FERTILIZER':
                stock = max(0, stock - projection.tapes.fert[state.day])

        dawns = {}
        for day in range(state.day + 1, V.SEASON_DAYS):
            quantity = projection.tapes.prod[product][day - 1]
            if product == 'WHEAT':
                quantity -= projection.tapes.wheat[day]
            elif product == 'FERTILIZER':
                quantity -= projection.tapes.fert[day]
            dawns[24 * day] = quantity

        timeline = set(dawns)
        if partial_step is not None:
            timeline.add(partial_step)
        if continuous_step is not None:
            timeline.add(continuous_step)
        active = False
        for step in sorted(timeline):
            if step in dawns:
                stock = max(0, stock + dawns[step])
            if partial_step == step and partial_quantity:
                quantity = min(stock, partial_quantity)
                emit(product, step, quantity)
                stock -= quantity
            if continuous_step == step:
                active = True
            if active and stock:
                emit(product, step, stock)
                stock = 0
    return events


def _take_supply(lots: deque[list[float]], quantity: float) -> tuple[float, float]:
    """FIFO expected candidate units and their exact attributed marginal revenue."""
    taken = revenue = 0.0
    left = quantity
    while left > 1e-12 and lots:
        units, value = lots[0]
        use = min(units, left)
        share = 0.0 if units == 0 else value * use / units
        taken += use; revenue += share; left -= use
        units -= use; value -= share
        if units <= 1e-12:
            lots.popleft()
        else:
            lots[0] = [units, value]
    return taken, revenue


def market_capture_objective(plan: Plan, observation: dict, *,
                             demand_history: PublicDemandHistory | None = None) -> dict:
    """Actor-visible proposal score for expected demand capture; never a label or selector.

    The only demand inputs are the current public shop prefix, its cumulative public-memory
    cursor and the accepted uniform prior over still-unrevealed shop identities. Candidate
    stock is the plan's projected remaining production, not actual sales. Exact marginal
    revenue uses Projection.sell_value's sequential floor semantics; no nominal price is
    substituted. Fractional expected inventory follows one linearly interpolated mean path,
    not an exact stochastic revenue expectation. The returned score may order sibling
    proposals and has no promotion path.
    """
    plan = strict_plan(plan.canonical())
    state = State(observation, demand_history=demand_history)
    features = state.market_features(require_history=True)
    projection = Projection(state, plan, labor_factor=4.0)
    sale_events = _planned_sale_events(state, projection)
    edges = (state.step, min(DEMAND_STOP, state.step + 72),
             min(DEMAND_STOP, state.step + 240), DEMAND_STOP)
    products = {}
    total_capture = total_demand = total_revenue = total_surplus = total_missed = 0.0
    total_score = 0.0
    for product in V.PRODUCTS:
        public = features[product]
        horizons = []
        for index, name in enumerate(MARKET_HORIZONS):
            known = public['known_remaining'][index]
            prior = public['expected_unrevealed'][index]
            horizons.append({'name': name, 'start_step': edges[index], 'end_step': edges[index + 1],
                             'known_remaining_demand': known,
                             'prior_expected_unrevealed_demand': prior,
                             'expected_remaining_demand': known + prior,
                             'planned_player_supply': 0,
                             'market_replenishing_supply': 0.0,
                             'candidate_capture': 0.0, 'missed_demand': 0.0,
                             'exact_marginal_sale_revenue': 0.0})
        by_step = defaultdict(int)
        for step, quantity in sale_events[product]:
            by_step[step] += quantity
        inventory = float(state.inventory[product])
        lots: deque[list[float]] = deque()
        planned = accepted = capture = missed = revenue = captured_revenue = 0.0
        floor_blocked = floor_revenue = missed_value = 0.0
        for step in range(state.step, DEMAND_STOP):
            bucket = 0 if step < edges[1] else (1 if step < edges[2] else 2)
            quantity = by_step.get(step, 0)
            if quantity:
                value, after = _expected_sell_value(projection, product, inventory, quantity)
                moved = min(float(quantity), max(0.0, after - inventory))
                blocked = float(quantity) - moved
                blocked_value = blocked * PRICE_FLOOR
                accepted_value = max(0.0, value - blocked_value)
                planned += quantity; accepted += moved; revenue += value
                floor_blocked += blocked; floor_revenue += blocked_value
                horizons[bucket]['planned_player_supply'] += quantity
                horizons[bucket]['market_replenishing_supply'] += moved
                horizons[bucket]['exact_marginal_sale_revenue'] += value
                if moved > 1e-12:
                    lots.append([moved, accepted_value])
                inventory = after
            known, prior = _expected_demand_at(state, product, step)
            demand = known + prior
            if demand:
                got, got_value = _take_supply(lots, demand)
                short = demand - got
                capture += got; captured_revenue += got_value; missed += short
                horizons[bucket]['candidate_capture'] += got
                horizons[bucket]['missed_demand'] += short
                if short > 1e-12:
                    missed_value += _fractional_sell_value(projection, product, inventory, short)
                inventory -= demand  # demand is created even when public inventory is unavailable
        surplus = math.fsum(lot[0] for lot in lots)
        surplus_value = math.fsum(lot[1] for lot in lots)
        expected_remaining = math.fsum(h['expected_remaining_demand'] for h in horizons)
        # Captured value is rewarded; exact value of late surplus, floor-blocked supply and
        # demand that lacked timely supply are symmetric opportunity costs.
        score = captured_revenue - surplus_value - floor_revenue - missed_value
        products[product] = {
            'cumulative_demand_created': public['total_demand_created'],
            'created_by_phase': public['created_by_phase'],
            'remaining_expected_demand': expected_remaining,
            'planned_player_supply': planned,
            'market_replenishing_supply': accepted,
            'candidate_capture': capture, 'missed_demand': missed,
            'likely_unsold_surplus': surplus, 'floor_blocked_supply': floor_blocked,
            'exact_marginal_sale_revenue': revenue,
            'captured_marginal_revenue': captured_revenue,
            'surplus_marginal_value': surplus_value,
            'missed_marginal_value': missed_value,
            'score': score, 'horizons': horizons,
            'current_market_inventory': state.inventory[product],
            'actual_player_sales': None, 'actual_player_supply': None,
            'endpoint_inventory': None,
        }
        total_capture += capture; total_demand += expected_remaining; total_revenue += revenue
        total_surplus += surplus + floor_blocked; total_missed += missed; total_score += score
    return bound({
        'schema': SCHEMA + '-market-objective-v1', 'plan_sha256': plan.digest(),
        'step': state.step, 'score': total_score,
        'rank_key': [total_score, total_capture, total_revenue, -total_surplus, -total_missed],
        'expected_remaining_demand': total_demand, 'candidate_capture': total_capture,
        'candidate_capture_fraction': total_capture / total_demand if total_demand else 1.0,
        'planned_player_supply': math.fsum(products[p]['planned_player_supply'] for p in V.PRODUCTS),
        'likely_unsold_surplus': total_surplus, 'missed_demand': total_missed,
        'exact_marginal_sale_revenue': total_revenue, 'products': products,
        'demand_semantics': 'SCHEDULED_WITHDRAWALS_REGARDLESS_OF_INVENTORY',
        'supply_semantics': 'PLAN_PROJECTED_NOT_ACTUAL_PLAYER_SALES',
        'capture_semantics': 'ACCOUNTING_PROXY_NOT_OBSERVED_OR_OPPONENT_ATTRIBUTED_SALES',
        'fractional_expectation_semantics':
            'LINEAR_MEAN_PATH_PROXY_NOT_EXACT_STOCHASTIC_REVENUE',
        'actor_visible_inputs': list(ACTOR_MARKET_INPUTS),
        'uses_future_shop_identity': False, 'teacher_authority': False,
        'promotion_authority': False, 'selection_authority': 'NONE_PROPOSAL_ORDER_ONLY',
    })

def propose(parent: Plan, observation: dict, *, trial: int, seed: int,
            seen: set[str], max_proposals: int = 64,
            proposal_order: str = 'control') -> dict:
    """One canonical-key edit; optional market score orders siblings only.

    `control` preserves the historical first legal/compilable proposal. The
    `market_capture` arm evaluates the same bounded, seeded action prefix and
    orders its admitted siblings by the actor-visible objective. Neither route
    can promote a plan; the exact paired terminal evaluator remains authoritative.
    """
    if proposal_order not in PROPOSAL_ORDERS:
        raise ValueError('proposal_order must be control or market_capture')
    family = FAMILIES[trial % len(FAMILIES)]
    rng = random.Random(f'{seed}:{trial}:{parent.digest()}')
    masks = Masks(State(observation), parent, labor_factor=4.0)
    actions = []
    if family == 'PROD':
        existing = sorted(parent.cohorts, key=lambda m: (-parent.cohorts[m], m))
        actions = [V.encode('PROD', m, d) for m in existing for d in range(len(V.DELTAS))]
        newcomers = rng.sample(list(range(V.N_MODULES)), min(12, V.N_MODULES))
        actions += [V.encode('PROD', m, d) for m in newcomers for d in (2, 3, 4)]
        rng.shuffle(actions)
        if trial == 0 and existing:
            first = V.encode('PROD', existing[0], V.DELTAS.index('MINUS_ONE'))
            actions = [first] + [a for a in actions if a != first]
    elif family == 'SERVICE':
        actions = [V.encode('SERVICE', m, r) for m in sorted(parent.cohorts) for r in range(len(V.REGIMES))]
        rng.shuffle(actions)
    elif family == 'LAND':
        actions = [V.encode('LAND', t) for t in range(len(V.LAND_TIMINGS))]
        rng.shuffle(actions)
    else:
        for product in range(len(V.PRODUCTS)):
            for primitive in range(len(V.PRIMITIVES)):
                for horizon in range(len(V.HORIZONS)):
                    ranks = range(len(masks.frontier(product, primitive, horizon)) or 1)
                    actions += [V.encode('MARKET', product, primitive, horizon, rank) for rank in ranks]
        rng.shuffle(actions)
    result = {'trial': trial, 'family': family, 'parent_sha256': parent.digest(),
              'proposal_order': proposal_order, 'status': 'EXHAUSTED', 'attempts': [],
              'teacher_authority': False, 'promotion_authority': False,
              'selection_authority': 'EXACT_PAIRED_TERMINAL_ONLY'}
    admitted = []
    local_seen = set(seen)
    for position, action in enumerate(actions[:max_proposals]):
        attempt = {'action': action, 'description': V.describe(action)}
        if not masks.legal(action):
            result['attempts'].append({**attempt, 'status': 'MASKED'}); continue
        child = parent.apply(action, masks.resolver)
        child_sha = child.digest()
        if child_sha in local_seen:
            result['attempts'].append({**attempt, 'status': 'DUPLICATE'}); continue
        local_seen.add(child_sha)
        screen = screen_plan(child, observation)
        if screen['status'] != 'COMPILES':
            result['attempts'].append({**attempt, 'status': 'UNCOMPILABLE', 'screen': screen}); continue
        if proposal_order == 'control':
            result['attempts'].append({**attempt, 'status': 'ADMITTED', 'plan_sha256': child_sha})
            result.update(status='ADMITTED', action=action, plan=child.canonical(), screen=screen)
            break
        objective = market_capture_objective(child, observation)
        row = {**attempt, 'status': 'ADMITTED', 'plan_sha256': child_sha,
               'objective': objective}
        result['attempts'].append(row)
        key = tuple(objective.get('rank_key', [objective['score']])) + (-position,)
        admitted.append((key, action, child, screen, objective))
    if proposal_order == 'market_capture' and admitted:
        _key, action, child, screen, objective = max(admitted, key=lambda row: row[0])
        result.update(status='ADMITTED', action=action, plan=child.canonical(), screen=screen,
                      objective=objective)
        for attempt in result['attempts']:
            if attempt.get('plan_sha256') == child.digest():
                attempt['selected_by_objective'] = True
                break
    return result

def _execution(runner):
    return panel.execution_binding(panel._NATIVE_PLAY_GAME if runner is native_game else runner)


def _positive(value, name, *, integer=False, upper=None):
    if (type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value)
            or value <= 0 or (upper is not None and value > upper)):
        raise ValueError(f'positive bounded {name} required')



def _crn_config() -> dict:
    path = ROOT / CRN_SCHEDULES
    equal(file_sha(path), CRN_SCHEDULES_SHA, 'fixed V0 search CRN schedules')
    config = read(path, binding=False)
    if (config.get('schema') != 'farm-v0-search-crn-schedules-v1'
            or config.get('rules', {}).get('discovery', {}).get('explicit_shops') != 'REQUIRED_IDENTICAL_PER_WORLD'
            or config.get('rules', {}).get('confirmation', {}).get('explicit_shops') != 'FORBIDDEN'):
        raise ValueError('accepted V0 search CRN schedule contract required')
    return config


def market_comparison_protocol(*, game_budget: int | None = None) -> dict:
    """Freeze equal-budget objective/control discovery and fresh WDL confirmation.

    Discovery shop suffixes live only in evaluator jobs. The protocol object must
    never be passed to market_capture_objective, propose, or an actor.
    """
    config = _crn_config()
    base = []
    for world in config['worlds']['discovery']:
        for seat in config['seats']:
            base.append({'world_id': world['world_id'], 'seed': world['seed'],
                         'schedule_sha256': world['schedule_sha256'],
                         'shops': list(world['shops']), 'seat': seat})
    if game_budget is None:
        game_budget = len(base)
    _positive(game_budget, 'market comparison game_budget', integer=True, upper=1000000)
    jobs = []
    for index in range(game_budget):
        cell = copy.deepcopy(base[index % len(base)])
        jobs.append({'job_index': index, 'cycle': index // len(base), **cell})
    confirmation_rule = config['rules']['confirmation']
    confirmation = [
        {'world_id': world['world_id'], 'seed': world['seed'], 'seat': seat,
         'environment': confirmation_rule['environment'], 'explicit_shops': False}
        for world in config['worlds']['confirmation'] for seat in config['seats']
    ]
    return bound({
        'schema': SCHEMA + '-market-comparison-v1',
        'crn_config': {'path': CRN_SCHEDULES, 'sha256': CRN_SCHEDULES_SHA,
                       'schema': config['schema']},
        'information_boundary': {
            'proposal_inputs': list(ACTOR_MARKET_INPUTS),
            'future_shops': 'EVALUATOR_ONLY', 'world_seed': 'EVALUATOR_ONLY',
            'outcomes': 'EVALUATOR_ONLY',
            'protocol_must_not_enter_actor_or_proposer': True,
        },
        'discovery': {
            'rules': 'FIXED_EXPLICIT_SHOP_CRN_SEARCH_ONLY',
            'game_budget_per_arm': game_budget,
            'base_cells': len(base),
            'arms': {
                'control': {'proposal_order': 'control', 'jobs': copy.deepcopy(jobs)},
                'market_capture': {'proposal_order': 'market_capture', 'jobs': copy.deepcopy(jobs)},
            },
            'metrics': ['realized_total_demand_created', 'candidate_capture_fraction',
                        'candidate_diversity', 'wdl_by_market_prefix_stratum'],
        },
        'confirmation': {
            'rules': 'FRESH_ORIGINAL_RULE_NO_EXPLICIT_SHOPS',
            'cells': confirmation,
            'environment': confirmation_rule['environment'],
            'explicit_shops': False,
            'selection_authority': 'FRESH_ORIGINAL_RULE_PAIRED_TERMINAL_WDL',
            'candidate_confirmed_when': 'SUM_WDL_ORDINAL_DELTA_OVER_FROZEN_CONTROL_GT_0',
        },
        'kill_rules': [
            'KILL_HIDDEN_OR_FUTURE_FEATURE_LEAK',
            'KILL_IF_MARKET_CAPTURE_CONFIRMED_IMPROVEMENTS_PER_1000_NOT_ABOVE_CONTROL',
        ],
        'teacher_authority': False, 'promotion_authority': False,
    })


def _timing_name(day: int) -> str:
    if day <= 9:
        return 'early'
    if day <= 19:
        return 'mid'
    return 'late'


def _realized_market_metrics(shop_history: list[str], supply_events: list[dict]) -> dict:
    """Evaluator-only scheduled-demand accounting from market-supply receipts.

    Coverage is a FIFO accounting proxy.  It is neither an observed sale ledger nor
    attribution against the opponent's competing supply.
    """
    if (not isinstance(shop_history, list) or len(shop_history) != 8
            or any(type(shop) is not str or shop not in SHOPS for shop in shop_history)):
        raise ValueError('complete eight-shop realized history required')
    events = defaultdict(float)
    for event in supply_events:
        if (not isinstance(event, dict) or set(event) != {'step', 'product', 'units'}
                or type(event['step']) is not int or not 0 <= event['step'] < DEMAND_STOP
                or event['product'] not in V.PRODUCTS
                or type(event['units']) not in (int, float)
                or not math.isfinite(event['units']) or event['units'] < 0):
            raise ValueError('exact nonnegative actual market supply event required')
        events[(event['step'], event['product'])] += float(event['units'])
    buffers = {product: 0.0 for product in V.PRODUCTS}
    created = {product: 0.0 for product in V.PRODUCTS}
    captured = {product: 0.0 for product in V.PRODUCTS}
    shop_by_timing = {(product, timing): 0.0 for product in V.PRODUCTS
                      for timing in ('early', 'mid', 'late')}
    for step in range(DEMAND_STOP):
        for product in V.PRODUCTS:
            buffers[product] += events[(step, product)]
        active = shop_history[:sum(reveal <= step for reveal in DEMAND_SPEC['reveal_steps'])]
        for product in V.PRODUCTS:
            demand = 0
            shop_demand = 0
            if step % 4 == 0:
                shop_demand = sum((2 if len(SHOPS[shop]) == 1 else 1)
                                  for shop in active if product in SHOPS[shop])
                demand += shop_demand
            if step % 24 == 0 and product in TOWN_CENTER_PRODUCTS:
                demand += 1
            if demand:
                created[product] += demand
                take = min(buffers[product], demand)
                captured[product] += take
                buffers[product] -= take
            if shop_demand:
                shop_by_timing[(product, _timing_name(step // 24))] += shop_demand
    prefixes = {}
    timing_ends = {'early': 10 * 24, 'mid': 20 * 24, 'late': DEMAND_STOP}
    for timing, end in timing_ends.items():
        count = sum(reveal < end for reveal in DEMAND_SPEC['reveal_steps'])
        prefixes[timing] = list(shop_history[:count])
    total_created = math.fsum(created.values())
    total_captured = math.fsum(captured.values())
    return {'created_by_product': created, 'captured_by_product': captured,
            'realized_total_demand_created': total_created,
            'candidate_capture': total_captured,
            'candidate_capture_fraction': total_captured / total_created if total_created else 1.0,
            'capture_semantics': 'ACCOUNTING_PROXY_NOT_OBSERVED_OR_OPPONENT_ATTRIBUTED_SALES',
            'remaining_candidate_supply': math.fsum(buffers.values()),
            'shop_demand_by_timing': {f'{product}:{timing}': value
                                      for (product, timing), value in shop_by_timing.items()},
            'observed_prefix_by_timing': prefixes}


def summarize_market_comparison(protocol: dict, records: list[dict]) -> dict:
    """Strict offline reducer; it never runs games and never feeds evaluator data to actors."""
    equal(protocol, bound(protocol), 'market comparison protocol binding')
    if protocol.get('schema') != SCHEMA + '-market-comparison-v1':
        raise ValueError('market comparison protocol required')
    if not isinstance(records, list):
        raise ValueError('comparison records list required')
    arms = ('control', 'market_capture')
    discovery_rows = {arm: {} for arm in arms}
    confirmation_rows = {arm: defaultdict(dict) for arm in arms}
    leaks = set()
    for row in records:
        if not isinstance(row, dict) or row.get('arm') not in arms:
            raise ValueError('comparison arm required')
        features = row.get('actor_visible_features', [])
        if (not isinstance(features, list) or not features
                or any(type(name) is not str for name in features)):
            raise ValueError('nonempty actor_visible_features string list required')
        feature_set = set(features)
        leaks.update(feature_set & FORBIDDEN_ACTOR_MARKET_FEATURES)
        leaks.update(feature_set - SAFE_ACTOR_MARKET_FEATURES)
        if row.get('phase') == 'discovery':
            required = ('job_index', 'world_id', 'seed', 'schedule_sha256', 'seat',
                        'candidate_sha256', 'status', 'wdl', 'shop_history',
                        'actual_market_supply_events')
            if any(key not in row for key in required) or row['status'] != 'COMPLETE' \
                    or row['wdl'] not in ('WIN', 'DRAW', 'LOSS'):
                raise ValueError('complete discovery receipt required')
            index = row['job_index']
            if type(index) is not int or index in discovery_rows[row['arm']]:
                raise ValueError('unique discovery job_index required per arm')
            discovery_rows[row['arm']][index] = row
        elif row.get('phase') == 'confirmation':
            required = ('world_id', 'seed', 'seat', 'candidate_sha256', 'status', 'wdl',
                        'control_wdl', 'environment', 'explicit_shops')
            if any(key not in row for key in required) or row['status'] != 'COMPLETE' \
                    or row['wdl'] not in ('WIN', 'DRAW', 'LOSS') \
                    or row['control_wdl'] not in ('WIN', 'DRAW', 'LOSS'):
                raise ValueError('complete paired confirmation receipt required')
            if (row['environment'] != protocol['confirmation']['environment']
                    or row['explicit_shops'] is not False):
                raise ValueError('fresh original-rule confirmation with no explicit shops required')
            key = (row['world_id'], row['seed'], row['seat'])
            target = confirmation_rows[row['arm']][row['candidate_sha256']]
            if key in target:
                raise ValueError('unique confirmation cell required')
            target[key] = row
        else:
            raise ValueError('discovery or confirmation phase required')
    arm_summary = {}
    strata = defaultdict(lambda: {'WIN': 0, 'DRAW': 0, 'LOSS': 0, 'games': 0,
                                  'shop_demand_units': 0.0})
    discovery_candidates = {}
    for arm in arms:
        jobs = protocol['discovery']['arms'][arm]['jobs']
        expected = {job['job_index']: job for job in jobs}
        if set(discovery_rows[arm]) != set(expected):
            raise ValueError('complete equal-budget discovery jobs required for both arms')
        candidates = set(); total_created = total_capture = 0.0
        for index in sorted(expected):
            row, job = discovery_rows[arm][index], expected[index]
            if (row['world_id'], row['seed'], row['schedule_sha256'], row['seat']) != (
                    job['world_id'], job['seed'], job['schedule_sha256'], job['seat']):
                raise ValueError('discovery job/seed/schedule/cell mismatch')
            equal(row['shop_history'], job['shops'], 'realized fixed discovery shop history')
            metrics = _realized_market_metrics(row['shop_history'], row['actual_market_supply_events'])
            total_created += metrics['realized_total_demand_created']
            total_capture += metrics['candidate_capture']
            candidates.add(row['candidate_sha256'])
            for product in V.PRODUCTS:
                for timing in ('early', 'mid', 'late'):
                    demand = metrics['shop_demand_by_timing'][f'{product}:{timing}']
                    if demand <= 0:
                        continue
                    prefix = tuple(metrics['observed_prefix_by_timing'][timing])
                    key = (arm, product, timing, prefix)
                    strata[key][row['wdl']] += 1
                    strata[key]['games'] += 1
                    strata[key]['shop_demand_units'] += demand
        discovery_candidates[arm] = candidates
        arm_summary[arm] = {
            'games': len(expected), 'realized_total_demand_created': total_created,
            'candidate_capture': total_capture,
            'candidate_capture_fraction': total_capture / total_created if total_created else 1.0,
            'candidate_diversity': {'unique_candidates': len(candidates),
                                    'candidate_rows': len(expected),
                                    'unique_fraction': len(candidates) / len(expected)},
        }
    rows = []
    for (arm, product, timing, prefix), counts in sorted(strata.items()):
        rows.append({'arm': arm, 'product': product, 'timing': timing,
                     'observed_prefix': list(prefix), **counts})
    any_confirmation = any(confirmation_rows[arm] for arm in arms)
    rates = {arm: None for arm in arms}
    confirmed = {arm: None for arm in arms}
    if any_confirmation:
        cells = {(cell['world_id'], cell['seed'], cell['seat'])
                 for cell in protocol['confirmation']['cells']}
        ordinal = {'LOSS': 0, 'DRAW': 1, 'WIN': 2}
        confirmed = {}
        for arm in arms:
            if set(confirmation_rows[arm]) != discovery_candidates[arm]:
                raise ValueError('every discovered candidate requires fresh confirmation')
            wins = 0
            for candidate in sorted(discovery_candidates[arm]):
                by_cell = confirmation_rows[arm][candidate]
                if set(by_cell) != cells:
                    raise ValueError('complete fresh original-rule confirmation panel required')
                delta = sum(ordinal[row['wdl']] - ordinal[row['control_wdl']]
                            for row in by_cell.values())
                wins += int(delta > 0)
            confirmed[arm] = wins
        budget = protocol['discovery']['game_budget_per_arm']
        rates = {arm: 1000.0 * confirmed[arm] / budget for arm in arms}
        status = 'CONFIRMATION_COMPLETE'
        if leaks:
            decision = 'KILL_HIDDEN_OR_FUTURE_FEATURE_LEAK'
        elif rates['market_capture'] <= rates['control']:
            decision = 'KILL_NO_CONFIRMED_RATE_GAIN'
        else:
            decision = 'KEEP_CONFIRMED_RATE_GAIN'
    else:
        status = 'DISCOVERY_COMPLETE_CONFIRMATION_NOT_RUN'
        decision = 'KILL_HIDDEN_OR_FUTURE_FEATURE_LEAK' if leaks else 'NOT_MEASURED'
    return bound({
        'schema': SCHEMA + '-market-comparison-summary-v1',
        'protocol_sha256': protocol['content_sha256'], 'status': status,
        'arms': arm_summary, 'wdl_by_market_prefix_stratum': rows,
        'confirmed_improved_candidates': confirmed,
        'improvements_per_1000_games': rates,
        'feature_leaks': sorted(leaks), 'proxy_decision': decision,
        'selection_authority': 'FRESH_ORIGINAL_RULE_PAIRED_TERMINAL_WDL',
        'endpoint_inventory_used_as_metric': False,
        'demand_created_semantics': 'SCHEDULED_TOWN_CENTER_PLUS_SHOP_WITHDRAWALS_REGARDLESS_OF_INVENTORY',
        'capture_semantics': 'ACCOUNTING_PROXY_NOT_OBSERVED_OR_OPPONENT_ATTRIBUTED_SALES',
        'fractional_expectation_semantics':
            'LINEAR_MEAN_PATH_PROXY_NOT_EXACT_STOCHASTIC_REVENUE',
        'teacher_authority': False, 'promotion_authority': False,
    })

def _safe(path):
    if any(p.is_symlink() for p in (Path(path), *Path(path).parents)):
        raise ValueError('symlinked output is refused')


def _sync_file(path):
    with Path(path).open('rb') as f:
        os.fsync(f.fileno())


def register(output: Path, *, train=None, heldout=None, seats=(0, 1),
             opponents=('router', 'k0006', 'tape0', 'pass'), trials=16, seed=20260917,
             max_games=4, cpu_seconds=1800., timeout=120., max_proposals=64,
             proposal_order: str = 'control', runner: Callable | None = None) -> dict:
    """Immutable preregistration. No native game is started by this function."""
    runner = native_game if runner is None else runner
    if proposal_order not in PROPOSAL_ORDERS:
        raise ValueError('proposal_order must be control or market_capture')
    registry = panel.world_registry()
    train = list(train) if train is not None else [w.id for w in registry.worlds if w.cohort == 'catalog']
    heldout = list(heldout) if heldout is not None else [w.id for w in registry.worlds if w.cohort == 'pair64']
    if not train or not heldout or set(train) & set(heldout):
        raise ValueError('nonempty disjoint train and heldout worlds required')
    if len(set(train)) != len(train) or len(set(heldout)) != len(heldout):
        raise ValueError('duplicate split world')
    if set(train + heldout) - {w.id for w in registry.worlds}:
        raise ValueError('unknown registered world')
    if (not seats or any(type(s) is not int or s not in (0, 1) for s in seats)
            or len(set(seats)) != len(seats)):
        raise ValueError('unique seats 0/1 required')
    if (not opponents or len(set(opponents)) != len(opponents)
            or set(opponents) - {'router', 'k0006', 'tape0', 'pass'}):
        raise ValueError('unique frozen opponents required')
    for val, name, cap in ((trials, 'trials', 10000), (max_games, 'max_games', 100000),
                           (max_proposals, 'max_proposals', 4096)):
        _positive(val, name, integer=True, upper=cap)
    _positive(cpu_seconds, 'cpu_seconds'); _positive(timeout, 'timeout', upper=300)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('integer seed in [0,2**63) required')
    plan, provenance = retained_baseline()
    opening = opening_observation(train[0])
    if State(opening).shops:
        raise ValueError('unrevealed opening world class required')
    screen = screen_plan(plan, opening)
    if screen['status'] != 'COMPILES':
        raise ValueError('baseline fails opening V0Agent compilation')
    manifest = bound({'schema': SCHEMA, 'split': {'train': train, 'heldout': heldout},
                      'registry': registry.document(), 'world_class': 'ALL_UNREVEALED_OPENINGS',
                      'seats': list(seats), 'opponents': list(opponents), 'trials': trials, 'seed': seed,
                      'max_proposals': max_proposals, 'proposal_order': proposal_order,
                      'proposal_objective': {'name': 'EXPECTED_MARKET_CAPTURE_V1',
                          'actor_visible_only': True, 'teacher_authority': False,
                          'promotion_authority': False, 'selection_authority': 'NONE'},
                      'selection_authority': 'PAIRED_TERMINAL_GAME_RESULT_ONLY',
                      'selection_scope': 'DISCOVERY_ONLY',
                      'final_selection_authority': FINAL_SELECTION_AUTHORITY,
                      'promotion_authority': False,
                      'crn_market_comparison': {'config': CRN_SCHEDULES,
                          'config_sha256': CRN_SCHEDULES_SHA,
                          'required_for_objective_vs_control_claim': True},
                      'budget': {'max_games': max_games,
                          'cpu_seconds': float(cpu_seconds), 'timeout_seconds': float(timeout)},
                      'repeats': 2, 'baseline': plan.canonical(), 'baseline_provenance': provenance,
                      'baseline_screen': screen, 'source_pins': source_pins(), 'python': sys.version,
                      'execution': _execution(runner),
                      'method': 'paired terminal-game hill climb; family round-robin; optional proposal-only market rank; incumbent wins ties',
                      'evaluation_route': ROUTE, 'training_admission': False,
                      'scope': panel.SCOPE, 'full_router_panel_regression_gate': 'NOT_RUN_BY_THIS_SEARCH'})
    output = Path(output); _safe(output); output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'manifest.json', manifest); _sync_file(output / 'manifest.json')
    with (output / 'events.jsonl').open('x') as f:
        f.flush(); os.fsync(f.fileno())
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return manifest


def append_event(output, manifest, events, event, **payload):
    record = bound({**payload, 'event': event, 'index': len(events),
                    'previous': events[-1]['content_sha256'] if events else manifest['content_sha256']})
    path = Path(output) / 'events.jsonl'; _safe(path)
    with path.open('a') as f:
        f.write(canonical(record) + '\n'); f.flush(); os.fsync(f.fileno())
    events.append(record)
    return record


def read_events(output, manifest):
    path = Path(output) / 'events.jsonl'; _safe(path)
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('torn journal; no automatic repair')
    events = []; previous = manifest['content_sha256']
    for i, line in enumerate(raw.splitlines()):
        try:
            event = json.loads(line)
            equal(event, bound(event), 'journal content binding')
            equal(event['index'], i, 'journal index'); equal(event['previous'], previous, 'journal chain')
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError('invalid journal; no automatic repair') from error
        events.append(event); previous = event['content_sha256']
    return events


def make_request(manifest, job, plan):
    actor = actor_spec(plan); rival = panel.load_actor(job['opponent'])
    job = {**job, 'agent': actor.name}
    validation = {'actors': {actor.name: actor.identity, rival.name: rival.identity},
                  'execution': manifest['execution']}
    return bound({'schema': SCHEMA + '-game', 'manifest_sha256': manifest['content_sha256'],
                  'source_pins': manifest['source_pins'], 'world_registry_sha256': manifest['registry']['content_sha256'],
                  'job': job, 'plan': plan, 'validation_plan': validation,
                  'timeout_seconds': manifest['budget']['timeout_seconds']})


def _worker(request, output):
    """Private child command; only the public-guarded harness calls the actors."""
    if request.get('schema') != SCHEMA + '-game':
        raise ValueError('search game request required')
    check_sources(request['source_pins'])
    registry = panel.world_registry()
    equal(registry.sha256, request['world_registry_sha256'], 'game registry')
    actor, rival = actor_spec(request['plan']), panel.load_actor(request['job']['opponent'])
    execution = panel.execution_binding(panel.play_game)
    equal(execution, request['validation_plan']['execution'], 'native worker execution')
    equal(actor.identity, request['validation_plan']['actors'][actor.name], 'worker actor')
    equal(rival.identity, request['validation_plan']['actors'][rival.name], 'worker opponent')
    job = request['job']
    row = panel.play_game(registry[job['world']], actor, rival, job['seat'], request['timeout_seconds'])
    records = getattr(actor._last, 'records', [])
    diagnostics = {'epochs': records, 'records_sha256': digest(records),
                   'rejected_commands': sum(r['rejected'] for r in records),
                   'unplaced': sum(r['unplaced'] for r in records),
                   'final_plan_sha256': actor._last.plan.digest() if request['plan'] is not None else None}
    response = bound({'request_sha256': request['content_sha256'], 'row': row,
                      'diagnostics': diagnostics, 'cpu_seconds': time.process_time()})
    write_new(output, response); _sync_file(output)


def native_game(request, *, request_path: Path, output_path: Path):
    """Each call starts a FRESH interpreter; replay is never a cached action tape."""
    # Run-local paths are not actor inputs and are not mutable global state.
    path, out = request_path, output_path
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
               PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, '-B', '-m', MODULE, '_game', '--request', str(path), '--output', str(out)]
    proc = subprocess.Popen(command, env=env, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True, text=True)
    try:
        stdout, stderr = proc.communicate(timeout=request['timeout_seconds'] + 10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        raise TimeoutError('fresh-process game timeout; attempt retained, no retry')
    if proc.returncode:
        raise RuntimeError(f'fresh-process worker exit {proc.returncode}: {stderr[-4000:]} {stdout[-1000:]}')
    return read(out)


def replay_signature(response):
    row = response['row']
    return {'row': {k: row.get(k) for k in ('world_sha256', 'seed', 'seat', 'actor_identity_hashes',
            'status', 'cash', 'margin', 'wdl', 'steps', 'actions_sha256')},
            'diagnostics': response['diagnostics']}


def _validate_response(request, response):
    equal(response, bound(response), 'response binding')
    equal(response['request_sha256'], request['content_sha256'], 'request/response binding')
    validation = panel.validate_game(response['row'], request['job'], request['validation_plan'],
                                     panel.world_registry()[request['job']['world']])
    if not validation['passed']:
        raise ValueError('failed game retained: ' + validation['error'])
    _positive(response['cpu_seconds'] + 1e-12, 'game cpu seconds')
    if request['validation_plan']['execution']['kind'] != 'NATIVE_GAME':
        equal(response['row']['physical_games_completed'], 0, 'synthetic physical games')
    elif request['plan'] is not None:
        diag = response['diagnostics']
        equal(diag['final_plan_sha256'], strict_plan(request['plan']).digest(), 'exact final Plan')
        equal([r['step'] for r in diag['epochs']], list(range(0, 719, 24)), '30 daily V0 epochs')
        equal(diag['records_sha256'], digest(diag['epochs']), 'V0 epoch binding')
        if any(r['edits'] != ['STOP'] or r['plan'] != diag['final_plan_sha256'] for r in diag['epochs']):
            raise ValueError('KEEP-only Plan execution required')


def _pairs(candidate: list[dict], control: list[dict]):
    def index(rows):
        values = {}
        for row in rows:
            key = (row['world'], row['seat'], row['opponent'])
            if key in values or row['status'] != 'COMPLETE':
                raise ValueError('complete unique paired cells required')
            values[key] = row
        return values
    a, b = index(candidate), index(control)
    if not a or set(a) != set(b):
        raise ValueError('complete matching paired world/seat/opponent cells required')
    cells = []
    for key in sorted(a):
        x, y = a[key], b[key]
        equal(x['world_sha256'], y['world_sha256'], 'paired physical world')
        equal(x['actor_identity_hashes']['opponent'], y['actor_identity_hashes']['opponent'], 'paired opponent')
        cells.append({'world': key[0], 'seat': key[1], 'opponent': key[2],
                      'candidate_cash': x['cash'][key[1]], 'control_cash': y['cash'][key[1]],
                      'candidate_margin': x['margin'], 'control_margin': y['margin'],
                      'delta_margin': x['margin'] - y['margin'],
                      'delta_own_cash': x['cash'][key[1]] - y['cash'][key[1]],
                      'candidate_wdl': x['wdl'], 'control_wdl': y['wdl']})
    return {'mean_delta_margin': math.fsum(c['delta_margin'] for c in cells) / len(cells), 'cells': cells}


class _Pause(Exception):
    pass


def run_search(output: Path, *, max_games=4, runner: Callable | None = None) -> dict:
    """Resume only committed attempts. max_games is per invocation AND total-capped."""
    _positive(max_games, 'max_games', integer=True, upper=100000)
    runner = native_game if runner is None else runner
    output = Path(output); _safe(output)
    manifest = read(output / 'manifest.json')
    if manifest.get('schema') != SCHEMA:
        raise ValueError('registered search manifest required')
    check_sources(manifest['source_pins'])
    equal(manifest['execution'], _execution(runner), 'execution runner on resume')
    equal(manifest['python'], sys.version, 'resume interpreter')
    equal(manifest['registry'], panel.world_registry().document(), 'registered worlds')
    _safe(output / 'run.lock')
    with (output / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('search run already locked') from error
        return _run_locked(output, manifest, max_games, runner)


def _run_locked(output, manifest, per_call, runner):
    events = read_events(output, manifest)
    pending = None; completed = {}; started = []
    for event in events:
        if event['event'] == 'STARTED':
            if pending is not None:
                raise ValueError('multiple uncommitted starts; completion UNKNOWN')
            pending = event; started.append(event)
        elif event['event'] == 'RESULT':
            if pending is None:
                raise ValueError('result without write-ahead start')
            equal(event['attempt_sha256'], pending['content_sha256'], 'result/attempt')
            equal(event['job'], pending['job'], 'result job')
            request = read(output / pending['request_path'])
            equal(request['content_sha256'], pending['request_sha256'], 'stored request')
            equal(request['job'], pending['job'], 'stored job')
            equal(request['manifest_sha256'], manifest['content_sha256'], 'request manifest')
            response = read(output / event['response_path'])
            equal(response, event['response'], 'stored response')
            _validate_response(request, response)
            key = digest(event['job'])
            if key in completed:
                raise ValueError('duplicate attempted job')
            completed[key] = (request, response)
            pending = None
    if pending is not None:
        raise ValueError('uncommitted game attempt: completion UNKNOWN; no automatic replay')
    initial_attempts = len(started)
    parent_cpu0 = time.process_time()
    historical_cpu = sum(e.get('controller_cpu_seconds', 0.) for e in events)
    game_cpu = sum(response['cpu_seconds'] for _, response in completed.values())
    selection = None; heldout = None; decisions = []
    screens = {e['trial']: e for e in events if e['event'] == 'PROPOSAL'}
    existing_decisions = {e['trial']: e for e in events if e['event'] == 'DECISION'}

    def cpu_left():
        return manifest['budget']['cpu_seconds'] - historical_cpu - game_cpu - (time.process_time() - parent_cpu0)

    def evaluate(plan, *, role, split):
        nonlocal game_cpu
        rows = []; diagnostics = []
        for world in manifest['split'][split]:
            for opponent in manifest['opponents']:
                for seat in manifest['seats']:
                    responses = []
                    for repeat in range(2):
                        job = {'role': role, 'split': split, 'world': world, 'seat': seat,
                               'opponent': opponent, 'repeat': repeat}
                        request = make_request(manifest, job, plan)
                        job = request['job']; key = digest(job)
                        if key in completed:
                            old_request, response = completed[key]
                            equal(request, old_request, 'deterministic resumed request')
                        else:
                            if len(started) >= manifest['budget']['max_games']:
                                raise _Pause('BUDGET_EXHAUSTED')
                            if len(started) - initial_attempts >= per_call:
                                raise _Pause('PARTIAL')
                            if cpu_left() < manifest['budget']['timeout_seconds'] + 15:
                                raise _Pause('CPU_BUDGET_EXHAUSTED')
                            # No automatic retry from this point: even a failed worker counts.
                            number = len(started)
                            reqpath = f'games/{number:06d}-request.json'
                            outpath = f'games/{number:06d}-response.json'
                            attempt = append_event(output, manifest, events, 'STARTED', job=job,
                                request_sha256=request['content_sha256'], request_path=reqpath)
                            started.append(attempt)
                            write_new(output / reqpath, request); _sync_file(output / reqpath)
                            response = (runner(request, request_path=output / reqpath, output_path=output / outpath)
                                        if runner is native_game else runner(request))
                            equal(response, bound(response), 'runner response binding')
                            if not (output / outpath).exists():
                                write_new(output / outpath, response); _sync_file(output / outpath)
                            append_event(output, manifest, events, 'RESULT', job=job,
                                attempt_sha256=attempt['content_sha256'], response=response, response_path=outpath)
                            _validate_response(request, response)
                            completed[key] = (request, response)
                            game_cpu += response['cpu_seconds']
                        responses.append(response)
                    equal(replay_signature(responses[0]), replay_signature(responses[1]), 'fresh-process replay cash/actions/controller')
                    rows.append(responses[0]['row']); diagnostics.append(responses[0]['diagnostics'])
        return rows, diagnostics

    baseline = manifest['baseline']; incumbent = strict_plan(baseline)
    winner_role = 'baseline'; seen = {incumbent.digest()}
    status = 'COMPLETE'
    try:
        base_rows, _ = evaluate(baseline, role='baseline', split='train')
        best_rows = base_rows
        opening = opening_observation(manifest['split']['train'][0])
        for trial in range(manifest['trials']):
            if cpu_left() <= 2:
                raise _Pause('CPU_BUDGET_EXHAUSTED')
            proposal = propose(incumbent, opening, trial=trial, seed=manifest['seed'], seen=seen,
                               max_proposals=manifest['max_proposals'],
                               proposal_order=manifest['proposal_order'])
            if trial in screens:
                equal(screens[trial]['proposal'], proposal, 'deterministic mutation on resume')
            else:
                screens[trial] = append_event(output, manifest, events, 'PROPOSAL', trial=trial, proposal=proposal)
            if proposal['status'] != 'ADMITTED':
                continue
            candidate = strict_plan(proposal['plan']); seen.add(candidate.digest())
            role = f'trial-{trial:06d}'
            rows, diags = evaluate(proposal['plan'], role=role, split='train')
            paired = _pairs(rows, best_rows)
            realizable = all(d['rejected_commands'] == 0 and d['unplaced'] == 0 for d in diags)
            take = realizable and paired['mean_delta_margin'] > 0
            decision = {'trial': trial, 'role': role, 'plan_sha256': candidate.digest(),
                        'control_role': winner_role, 'paired': paired, 'selected': take,
                        'runtime_rejected': not realizable,
                        'proposal_order': manifest['proposal_order'],
                        'proposal_objective_used_for_selection': False,
                        'selection_authority': 'PAIRED_TERMINAL_GAME_RESULT_ONLY',
                        'selection_scope': 'DISCOVERY_ONLY',
                        'final_selection_authority': FINAL_SELECTION_AUTHORITY,
                        'promotion_authority': False}
            if trial in existing_decisions:
                equal(existing_decisions[trial]['decision'], decision, 'deterministic selection on resume')
            else:
                append_event(output, manifest, events, 'DECISION', trial=trial, decision=decision)
            decisions.append(decision)
            if take:
                incumbent, best_rows, winner_role = candidate, rows, role
        train_pair = _pairs(best_rows, base_rows)
        selection = bound({'schema': SCHEMA + '-selection', 'manifest_sha256': manifest['content_sha256'],
                           'source_pins': manifest['source_pins'], 'world_class': manifest['world_class'],
                           'plan': incumbent.canonical(), 'plan_sha256': incumbent.digest(),
                           'winner_role': winner_role, 'train_delta_margin': train_pair['mean_delta_margin'],
                           'train_pairs': train_pair['cells'], 'proposal_order': manifest['proposal_order'],
                           'proposal_objective_used_for_selection': False,
                           'selection_authority': 'PAIRED_TERMINAL_GAME_RESULT_ONLY',
                           'selection_scope': 'DISCOVERY_ONLY',
                           'final_selection_authority': FINAL_SELECTION_AUTHORITY,
                           'promotion_authority': False,
                           'training_admission': False,
                           'execution_route': ROUTE, 'evidence_kind': manifest['execution']['kind']})
        if (output / 'selection.json').exists():
            equal(read(output / 'selection.json'), selection, 'frozen selection before heldout')
        else:
            write_new(output / 'selection.json', selection); _sync_file(output / 'selection.json')
        frozen = [e for e in events if e['event'] == 'SELECTION']
        if frozen:
            equal([e['selection_sha256'] for e in frozen], [selection['content_sha256']], 'selection event')
        else:
            append_event(output, manifest, events, 'SELECTION', selection_sha256=selection['content_sha256'])
        # No branch after this point may change incumbent/selection.
        held_base, _ = evaluate(baseline, role='baseline', split='heldout')
        held_best, _ = evaluate(incumbent.canonical(), role='selected', split='heldout')
        held_k0006, _ = evaluate(None, role='champion', split='heldout')
        over_base, over_k = _pairs(held_best, held_base), _pairs(held_best, held_k0006)
        heldout = {'delta_over_baseline': over_base['mean_delta_margin'],
                   'delta_over_k0006': over_k['mean_delta_margin'],
                   'baseline_pairs': over_base['cells'], 'k0006_pairs': over_k['cells'],
                   'world_class': manifest['world_class'],
                   'interpretation': 'Paired margin differences versus a common frozen opponent, not head-to-head win rates.'}
    except _Pause as pause:
        status = str(pause)
    finally:
        append_event(output, manifest, events, 'CPU', controller_cpu_seconds=time.process_time() - parent_cpu0)
    attempts = [e for e in events if e['event'] == 'STARTED']
    results = [e for e in events if e['event'] == 'RESULT']
    counts = {'mutation_trials': len(screens), 'proposals': 0, 'masked': 0, 'uncompilable': 0,
              'duplicates': 0, 'admitted': 0, 'rejected': 0, 'runtime_rejected': 0}
    for event in screens.values():
        proposal = event['proposal']
        counts['rejected'] += int(proposal['status'] != 'ADMITTED')
        for attempt in proposal['attempts']:
            counts['proposals'] += 1
            key = {'MASKED': 'masked', 'UNCOMPILABLE': 'uncompilable', 'DUPLICATE': 'duplicates', 'ADMITTED': 'admitted'}[attempt['status']]
            counts[key] += 1
    counts['runtime_rejected'] = sum(e['decision']['runtime_rejected'] for e in events if e['event'] == 'DECISION')
    native = manifest['execution']['kind'] == 'NATIVE_GAME'
    verified = 0
    for request, response in completed.values():
        if request['job']['repeat'] == 1:
            original = {**request['job'], 'repeat': 0}
            if digest(original) in completed:
                equal(replay_signature(completed[digest(original)][1]), replay_signature(response), 'all reported replays')
                verified += 1
    report = bound({'schema': SCHEMA + '-report', 'manifest_sha256': manifest['content_sha256'],
                    'status': status, 'evidence_kind': manifest['execution']['kind'], 'counts': counts,
                    'attempt_records': len(attempts), 'physical_games_attempted': len(attempts) if native else 0,
                    'synthetic_attempts': 0 if native else len(attempts),
                    'physical_games_completed': sum(e['response']['row']['physical_games_completed'] for e in results) if native else 0,
                    'replay_pairs_verified': verified, 'selection': selection, 'heldout': heldout,
                    'budget': manifest['budget'], 'game_cpu_seconds': game_cpu,
                    'controller_cpu_seconds': sum(e.get('controller_cpu_seconds', 0.) for e in events),
                    'scope': panel.SCOPE, 'training_admission': False,
                    'selection_scope': 'DISCOVERY_ONLY',
                    'final_selection_authority': FINAL_SELECTION_AUTHORITY,
                    'promotion_authority': False,
                    'limits': ['Seeded Plans, not empty-decoder-reachable labels.',
                        'Historical $60,753 opening-only cash is not assumed for daily V0Agent.',
                        'No held-out delta until ALL registered cells and fresh-process replays complete.',
                        'Attempt limits include controls and replays; unknown starts block all continuation.',
                        'No full router_panel legacy-gate certificate, ladder claim or promotion authority.',
                        'Fresh original-rule paired terminal-WDL confirmation is required for final selection.']})
    write_new(output / f'checkpoint-{len(events):06d}.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    reg = commands.add_parser('register', help='freeze split, sources and TOTAL budget; no games')
    reg.add_argument('--output', type=Path, required=True)
    reg.add_argument('--train', nargs='+'); reg.add_argument('--heldout', nargs='+')
    reg.add_argument('--seats', nargs='+', type=int, default=[0, 1])
    reg.add_argument('--opponents', nargs='+', default=['router', 'k0006', 'tape0', 'pass'])
    reg.add_argument('--trials', type=int, default=16); reg.add_argument('--seed', type=int, default=20260917)
    reg.add_argument('--max-games', type=int, default=4)
    reg.add_argument('--cpu-seconds', type=float, default=1800.)
    reg.add_argument('--timeout', type=float, default=120.)
    reg.add_argument('--max-proposals', type=int, default=64)
    reg.add_argument('--proposal-order', choices=PROPOSAL_ORDERS, default='control')
    run = commands.add_parser('run', help='run/resume a committed prefix; controls/replays count')
    run.add_argument('--output', type=Path, required=True); run.add_argument('--max-games', type=int, default=4)
    worker = commands.add_parser('_game', help=argparse.SUPPRESS)
    worker.add_argument('--request', type=Path, required=True); worker.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args(argv)); command = args.pop('command')
    try:
        if command == '_game':
            request = read(args['request'])
            limit = math.ceil(request['timeout_seconds']) + 5
            resource.setrlimit(resource.RLIMIT_CPU, (limit, limit + 1))
            _worker(request, args['output']); return 0
        result = register(**args) if command == 'register' else run_search(**args)
        print(json.dumps(result, sort_keys=True, indent=2), flush=True)
        return 0 if command == 'register' or result['status'] == 'COMPLETE' else 3
    except (ValueError, OSError, RuntimeError, TimeoutError, KeyError) as error:
        print(f'REFUSED: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    # -m must use the same factory module identity as the imported API and replay child.
    from importlib import import_module
    raise SystemExit(import_module(MODULE).main())
