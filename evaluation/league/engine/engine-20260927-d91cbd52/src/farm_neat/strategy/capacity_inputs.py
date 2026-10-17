"""Actor-only, fail-closed producers for the accepted conditional capacity checker.

``publish(context, proposals)`` is the optional ``capacity_inputs`` publisher of
strategy.pipeline. Call it last, on the final context, before checking proposals.
``explain_capacity_inputs`` names every unattested condition; UNKNOWN is useful.

Supported, deliberately small execution families under the PINNED DEFAULT rules:
* An owner-issued local FEED window (hours 21->22 or 22 alone) may bind one
  previously published service to one current worker/carried wheat. It uses the
  existing chain calendar, preserves every baseline effect, and requires actual
  own receipts to advance. See commitments.issue_service_window. Its descriptor
  is NOT adopted; the actual due-prefix rescue uses the unchanged adoption path.
* ``native_turn_proposal`` declares the ENTIRE immediate native action word, not
  a production portfolio. Existing workers may PASS, move or WATER a distinct,
  owned dry plant. Market orders may sell owned shed products OR make fixed-price
  seed/animal/land/hire purchases. All requested units must execute; this is not
  the weaker assertion that the engine will accept a possibly silent no-op.
* Existing farm-market-loop-v1 proposals may replace the entire CURRENT market
  queue. memory['capacity_base_action'] = {'step': now, 'action': native_action}
  must match their control queue. Its worker phase is replayed on OWN copies with
  the pinned native component, including atomic PLANT validation. A declared
  POST_WORKER observation only accepts an inert base worker word, never double
  applies nontrivial unit effects. No worker turn
  is lent to the market proposal. Replaced orders are not double-counted.

The immediate native-word family owns the whole current word; the market family
preserves all current worker commitments. Neither assumes future commitments
disappeared or certifies any later window. Original production remainders and V0 shop edits have
no full native lowering here and stay UNKNOWN. A land purchase certificate does
NOT certify a farming expansion, future watering, harvest, profitability, proposal
selection, or eligibility for the strategy adopter. Objective deltas are untouched.

Input excludes rival private state, seed, future shops and trainer provenance.
Only the own farm is inspected; the accepted checker binds the whole supplied
actor context to detect any drift. No game, search, policy or forecasting call.
Rules are the bundled standard competition configuration; optional actor memory
``capacity_rules`` must agree with DEFAULT_CONFIGURATION if supplied. Nondefault
rules, malformed/incomplete observations and unsupported words remain UNKNOWN.
Budget: <=128 existing workers, <=128 supplied worker commands, <=64 order ranks,
<100,000 requested units per rank (native escape guard);
fixed work O(board area + workers + ranks), independent of purchased quantities.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from farm_neat.game.constants import ANIMALS, CROPS, LAND_ORDER, LAND_PRICES, MOVES, PRODUCTS, hire_cost
from farm_neat.game.exact_engine import DEFAULT_CONFIGURATION, EXPECTED_ENGINE_SHA256, load_engine
from farm_neat.strategy.capacity import CapacityCase, CONDITIONS, CHAIN_CALENDAR
from farm_neat.strategy.types import Estimate, Proposal, Requirements, StrategyContext, UNKNOWN

SCHEMA = 'farm-capacity-native-turn-v1'
SCOPE = 'FULL_IMMEDIATE_NATIVE_TURN'
Z = Estimate.exact(0)
MAX_WORKERS = 128
MAX_RANKS = 64


class _Unproved(ValueError):
    """Incomplete or unsupported actor data, not an infeasibility certificate."""


def _count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise _Unproved('quantities must be nonnegative native integers')
    return value


def _point(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise _Unproved('missing worker position')
    x, y = map(_count, value)
    if x >= 10 or y >= 10:
        raise _Unproved('worker position is outside the standard board')
    return x, y


def _snapshot(ctx: StrategyContext) -> tuple[dict, dict]:
    obs = ctx.observation
    if obs.get('step') != ctx.step or obs.get('player') != ctx.player:
        raise _Unproved('observation identity/time mismatch')
    if obs.get('day') != ctx.step // 24 or obs.get('hour') != ctx.step % 24:
        raise _Unproved('missing or mismatched standard day/hour')
    rules = ctx.memory.get('capacity_rules', {})
    if not isinstance(rules, Mapping) or any(k not in DEFAULT_CONFIGURATION or v != DEFAULT_CONFIGURATION[k]
                                             for k, v in rules.items()):
        raise _Unproved('nondefault or unknown rules are not certified')
    try:
        # Do not inspect the other farm, even though its public projection is visible.
        farm = copy.deepcopy(dict(obs['farms'][ctx.player]))
        private = copy.deepcopy(dict(obs['private']))
        money = farm['money']
        if type(money) not in (int, float) or not math.isfinite(money) or money < 0:
            raise _Unproved('cash is not a finite nonnegative native balance')
        _count(farm['hires_today'])
        if farm['hires_today'] > 128:
            raise _Unproved('hire-cost arithmetic budget exceeded')
        land = farm['unlocked_quadrants']
        if land not in [['NW'] + LAND_ORDER[:n] for n in range(4)]:
            raise _Unproved('land ownership is not a native sequential prefix')
        grid = farm['tiles']
        if len(grid) != 10 or any(not isinstance(r, list) or len(r) != 10 for r in grid):
            raise _Unproved('missing standard 10x10 tile map')
        for y, row in enumerate(grid):
            for x, tile in enumerate(row):
                quadrant = ('N' if y < 5 else 'S') + ('W' if x < 5 else 'E')
                if (tile == 'LOCKED') != (quadrant not in land):
                    raise _Unproved('tiles and quadrant ownership disagree')
                if tile is not None and tile != 'LOCKED' and not isinstance(tile, dict):
                    raise _Unproved('unknown native tile representation')
        if not isinstance(farm['hands'], list) or len(farm['hands']) + 1 > MAX_WORKERS:
            raise _Unproved('missing workers or extraction worker budget exceeded')
        for p in [farm['farmer'], *farm['hands']]:
            _point(p)
        inventories = private['inventories']
        if not isinstance(inventories, list) or len(inventories) != len(farm['hands']) + 1:
            raise _Unproved('worker custody inventory is incomplete')
        for stock in [private['shed'], private['seeds'], *inventories]:
            if not isinstance(stock, Mapping) or any(not isinstance(k, str) for k in stock):
                raise _Unproved('stock map is incomplete')
            for n in stock.values():
                _count(n)
        # Native shed/seeds contain these zero-valued keys too. Missing is not zero.
        if not set(PRODUCTS + list(ANIMALS)) <= private['shed'].keys() or not CROPS.keys() <= private['seeds'].keys():
            raise _Unproved('shed/seed snapshot omits canonical inventory entries')
        if sum(private['shed'].values()) > 100:
            raise _Unproved('shed exceeds the standard capacity')
    except (KeyError, IndexError, TypeError) as error:
        raise _Unproved('own farm/private snapshot is incomplete') from error
    return farm, private


def _word(value: Any) -> dict:
    if not isinstance(value, Mapping) or set(value) != {'farmer', 'hands', 'market'}:
        raise _Unproved('full native word must explicitly contain farmer, hands, market only')
    a = copy.deepcopy(dict(value))
    if not isinstance(a['farmer'], list) or not isinstance(a['hands'], list) or not isinstance(a['market'], list):
        raise _Unproved('native command/queue lists are required')
    if len(a['hands']) + 1 > MAX_WORKERS or len(a['market']) > MAX_RANKS:
        raise _Unproved('native word extraction budget exceeded')
    if any(not isinstance(c, list) for c in [a['farmer'], *a['hands'], *a['market']]):
        raise _Unproved('each native command/order must be a list')
    return a


def _requirements(ctx: StrategyContext, a: dict, farm: dict, *, market_only=False) -> Requirements:
    """Exact necessary full-fill budgets only when there are no sale/purchase cycles."""
    cash = 0; inventory: Counter = Counter(); land = 0; hires = farm['hires_today']
    sells = buys = False
    for order in a['market']:
        if not order:  # Empty ranks still consume a market slot.
            continue
        op = order[0]
        if op == 'BUY_LAND' and len(order) == 1:
            index = len(farm['unlocked_quadrants']) - 1 + land
            if index < len(LAND_PRICES):
                cash += LAND_PRICES[index]
            land += 1; buys = True
        elif op == 'HIRE' and len(order) == 1:
            cash += hire_cost(hires); hires += 1; buys = True
        elif op in ('BUY_SEED', 'BUY_ANIMAL', 'SELL') and len(order) == 3:
            item, n = order[1:]
            if not isinstance(item, str) or _count(n) == 0:
                raise _Unproved('only positive integral full-fill orders are lowered')
            if n >= 100_000:
                raise _Unproved('native 100k-iteration market escape prevents a full-fill proof')
            if op == 'SELL' and item in PRODUCTS:
                inventory[item] += n; sells = True
            elif op == 'BUY_SEED' and item in CROPS:
                cash += CROPS[item]['seed'] * n; buys = True
            elif op == 'BUY_ANIMAL' and item in ANIMALS:
                cash += ANIMALS[item]['cost'] * n; inventory['space:shed'] += n; buys = True
            else:
                raise _Unproved('unsupported native item or order')
        else:
            raise _Unproved('dynamic-price, malformed or unsupported order has no necessary prefix budget')
    if sells and buys:
        raise _Unproved('sale-funded spending/reused headroom needs an exact cash/material prefix proof')
    hours = 0 if market_only else sum(bool(c) and c != ['PASS'] for c in [a['farmer'], *a['hands']])
    return Requirements(cash=Estimate.exact(cash), reserve=Z,
                        inventory={k: Estimate.exact(n) for k, n in inventory.items()},
                        market_slots=Estimate.exact(max((i + 1 for i, o in enumerate(a['market']) if o), default=0)), worker_hours=Estimate.exact(hours),
                        land_quadrants=Estimate.exact(land), window=(ctx.step, ctx.step))


def _unit_chains(ctx: StrategyContext, a: dict, farm: dict) -> list[dict]:
    positions = [farm['farmer'], *farm['hands']]
    chains = []; wet_sites = set()
    for idx, command in enumerate([a['farmer'], *a['hands']]):
        if command == ['PASS'] or command == []:
            continue
        if idx >= len(positions):
            raise _Unproved('requested worker is absent; hires happen after the unit phase')
        if len(command) != 1 or not isinstance(command[0], str):
            raise _Unproved('only PASS, single moves and distinct WATER services are lowered')
        op = command[0]; x, y = _point(positions[idx])
        if op in MOVES:
            dx, dy = MOVES[op]
            if not (0 <= x + dx < 10 and 0 <= y + dy < 10):
                raise _Unproved('requested move would silently no-op at board edge')
        elif op == 'WATER':
            tile = farm['tiles'][y][x]
            if not isinstance(tile, dict) or tile.get('kind') != 'PLANT' or tile.get('crop') not in CROPS:
                raise _Unproved('WATER needs an owned live crop at the actual worker position')
            if tile.get('watered_today') is not False or (x, y) in wet_sites:
                raise _Unproved('WATER is already satisfied or coupled with another worker service')
            if not CROPS[tile['crop']]['ongoing']:
                _count(tile.get('planted_day'))
                _count(tile.get('yield_units'))
                if type(tile.get('fertilized_until_day')) is not int:
                    raise _Unproved('WATER has incomplete nonongoing crop bonus fields')
            wet_sites.add((x, y))
        else:
            raise _Unproved('unsupported worker operation needs material/biology/custody lowering')
        worker = 'farmer' if idx == 0 else f'hand:{idx - 1}'
        chains.append({'worker': worker, 'entry': ctx.step, 'until': ctx.step + 1,
                       'duties': [{'id': f'{worker}:{op}:{x},{y}', 'duration': 1,
                                   'release': ctx.step, 'deadline': ctx.step + 1, 'marginal': True}]})
    return chains


def _background_units(ctx: StrategyContext, a: dict, farm: dict, private: dict) -> None:
    """Exact own-only phase, including the interpreter's ALL-request seed gate."""
    actions = [a['farmer'], *a['hands']]
    demand = Counter(c[1] for c in actions if len(c) >= 2 and c[0] == 'PLANT' and isinstance(c[1], str))
    blocked = {crop for crop, n in demand.items() if n > private['seeds'].get(crop, 0)}
    engine = load_engine()  # verifies pinned bytes; no env initialization/step/seed
    try:
        for idx, command in enumerate(actions):
            if len(command) >= 2 and command[0] == 'PLANT' and command[1] in blocked:
                command = ['PASS']
            engine._apply_unit_action(farm, private, idx, command, 10, ctx.step // 24, 24, 100)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as error:
        raise _Unproved('background native unit phase could not be reproduced exactly') from error


def native_turn_proposal(ctx: StrategyContext, *, module: str, decision_key: str, action: Mapping) -> Proposal:
    """Declare a new immediate FULL-word leaf; never rewrite an existing portfolio.

    Unsupported actions still produce a Proposal with UNKNOWN requirements so
    diagnostics can say why lowering stopped. This constructor does not execute.
    """
    a = _word(action)
    try:
        farm, _ = _snapshot(ctx)
        required = _requirements(ctx, a, farm)
    except _Unproved:
        required = Requirements(window=(ctx.step, ctx.step))
    return Proposal(module, decision_key,
                    {'schema': SCHEMA, 'scope': SCOPE, 'step': ctx.step, 'action': a},
                    ('cash', 'land', 'labor', 'inventory', 'market_slots'), required,
                    invalidated_by=('actor_context_changed',), parents=(CHAIN_CALENDAR,))


def _extract(ctx: StrategyContext, proposal: Proposal) -> tuple[CapacityCase, dict[str, str]]:
    if isinstance(proposal.candidate, Mapping) and proposal.candidate.get('schema') in (
            'farm-commitment-service-window-v1', 'farm-production-feed-rescue-v1'):
        return _declared_case(ctx, proposal)
    missing = {name: 'no complete native lowering for this proposal/window'
               for name in CONDITIONS['chains']}
    model = {'kind': 'chains', 'chains': []}
    available = Requirements(window=proposal.requirements.window)
    c = proposal.candidate
    try:
        if not isinstance(c, Mapping):
            raise _Unproved('proposal candidate is not a native lowering')
        market_only = c.get('schema') == 'farm-market-loop-v1'
        if c.get('schema') == SCHEMA:
            if c.get('scope') != SCOPE or c.get('step') != ctx.step:
                raise _Unproved('native word scope or observation step mismatch')
            a = _word(c.get('action'))
        elif market_only:
            key = str(ctx.step)
            orders, control = c.get('orders'), c.get('control_orders')
            if (not isinstance(orders, Mapping) or not isinstance(control, Mapping) or
                    set(orders) - {key} or set(control) - {key}):
                raise _Unproved('only one current full market queue/control is lowered')
            a = _word({'farmer': ['PASS'], 'hands': [], 'market': orders.get(key, [])})
        else:
            raise _Unproved('full production remainders and V0 edits need an executor-native lowering')
        if proposal.requirements.window != (ctx.step, ctx.step):
            raise _Unproved('only the immediate native turn has an exact net envelope')
        farm, private = _snapshot(ctx)
        if market_only:
            base = ctx.memory.get('capacity_base_action')
            if not isinstance(base, Mapping) or base.get('step') != ctx.step:
                raise _Unproved('resources: missing current capacity_base_action worker/control footprint')
            baseline = _word(base.get('action'))
            if baseline['market'] != control.get(str(ctx.step), []):
                raise _Unproved('resources: base action does not match the replaced control market queue')
            phase = ctx.memory.get('market_loop', {})
            if (isinstance(phase, Mapping) and phase.get('phase') == 'POST_WORKER' and
                    any(c and c != ['PASS'] for c in [baseline['farmer'], *baseline['hands']])):
                raise _Unproved('resources: post-worker observation cannot replay a nontrivial worker phase twice')
            _background_units(ctx, baseline, farm, private)
        available = Requirements(cash=Estimate.exact(farm['money']), reserve=Z,
            inventory={**{k: Estimate.exact(n) for k, n in private['shed'].items()},
                       'space:shed': Estimate.exact(100 - sum(private['shed'].values()))},
            market_slots=Estimate.exact(10), worker_hours=Estimate.exact(0 if market_only else 1 + len(farm['hands'])),
            land_quadrants=Estimate.exact(4 - len(farm['unlocked_quadrants'])), window=(ctx.step, ctx.step))
        missing.pop('resources')
        expected = _requirements(ctx, a, farm, market_only=market_only)
        # Refuse arbitrary caller "proofs" and over/understated necessary costs.
        # Exact equality is conservative: unknown/bounded upstream fields are not
        # upgraded in place. A future producer may attest additional bound forms.
        if proposal.requirements != expected:
            raise _Unproved('requirements: declared requirements differ from exact full-fill native extraction')
        missing.pop('requirements')
        try:
            chains = [] if market_only else _unit_chains(ctx, a, farm)
        except _Unproved as error:
            # Unsupported units may DROP/PICKUP/BUILD before market. The current
            # shed is NOT a valid upper bound on their post-worker stock. Since
            # the frozen checker tests cuts BEFORE model conditions, withdraw
            # common attestations too, not just the positive schedule proof.
            missing['requirements'] = str(error)
            missing['resources'] = str(error)
            for name in missing:
                missing[name] = str(error)
        else:
            model['chains'] = chains
            # Each is justified by a separate restriction above, not caller data:
            # one named existing worker / fixed one-turn service / now-only window /
            # survives until phase end / no shared service or material / native
            # non-temporal guards / entire queue + entire declared worker family.
            for name in ('fixed_chains', 'anchored_durations', 'occupancy_windows',
                         'continuous_workers', 'independent_resources',
                         'non_temporal_prerequisites', 'complete_coverage'):
                missing.pop(name, None)
    except _Unproved as error:
        for name in missing:
            missing[name] = str(error)
    # Binding is also the central recursively checked actor-only data boundary.
    # It raises on hidden keys and executable objects rather than hiding leaks.
    evidence = hashlib.sha256(json.dumps({'model': model, 'available': available.to_dict(),
        'expected_engine': EXPECTED_ENGINE_SHA256}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    certs = {name: f'capacity_inputs:v1:{name}:{evidence}' for name in sorted(CONDITIONS['chains'] - missing.keys())}
    case = CapacityCase.bind(ctx, proposal, available=available, model=model, certificates=certs,
                             parents=(CHAIN_CALENDAR,))
    return case, dict(sorted(missing.items()))


def build_capacity_case(context: StrategyContext, proposal: Proposal) -> CapacityCase:
    """Build a bound case; missing extraction facts never become guessed zeroes."""
    return _extract(context, proposal)[0]


def explain_capacity_inputs(context: StrategyContext, proposal: Proposal) -> dict:
    """Return portable case plus named missing attestations (not objective values)."""
    case, missing = _extract(context, proposal)
    return {'schema': 'farm-capacity-inputs-diagnostic-v1', 'case': case.to_dict(), 'missing': missing,
            'scope': 'immediate word, current market replacement, or owner-issued local service; physical capacity only'}


def publish(context: StrategyContext, proposals: Iterable[Proposal]) -> dict[tuple[str, str], CapacityCase]:
    """Drop-in final input publisher for the accepted module pipeline."""
    cases = {}
    for proposal in proposals:
        if proposal.key in cases:
            raise ValueError('duplicate proposal key in capacity input publication')
        cases[proposal.key] = build_capacity_case(context, proposal)
    return cases


def declared_service_proposal(context: StrategyContext) -> Proposal:
    """One future local service, NOT arbitrary free time or a new production plan.

    Install commitments.issue_service_window's result in own memory under
    declared_service_window first. This proposal is capacity-only. At the actual
    due prefix, regenerate LateFeedRescue and use service_window_guard with the
    existing NativeLowerer/Session; do not adopt this future-window descriptor.
    """
    from farm_neat.strategy.commitments import SERVICE_WINDOW_SCHEMA, validate_service_window
    w = context.memory.get('declared_service_window')
    check = validate_service_window(context, w)
    if check['status'] != 'READY':
        raise ValueError(check.get('reason', 'no remaining declared service'))
    return Proposal('declared_window', 'feed:' + w['window_id'][:20],
        dict(schema=SERVICE_WINDOW_SCHEMA, window_id=w['window_id']), tuple(w['touches']),
        Requirements(cash=Z, reserve=Z, inventory={'WHEAT': Estimate.exact(1)},
                     market_slots=Z, worker_hours=Estimate.exact(1), land_quadrants=Z,
                     window=(context.step, w['due_step'])),
        invalidated_by=('commitment revision', 'own receipt or full parent drift',
                        'worker/animal generation', 'protected resources', 'price/shop support'),
        parents=(CHAIN_CALENDAR,))


def _declared_case(context: StrategyContext, proposal: Proposal) -> tuple[CapacityCase, dict[str, str]]:
    """Use the existing chain representation; permission/action proof stay separate."""
    from farm_neat.strategy.commitments import (SERVICE_WINDOW_SCHEMA, validate_service_window,
                                                service_window_guard)
    from farm_neat.strategy.rescue import SCHEMA as RESCUE_SCHEMA
    from farm_neat.strategy.adoption import RESCUE_AUTHORIZATION
    missing = {name: 'owner-issued declared service window unavailable' for name in CONDITIONS['chains']}
    model = dict(kind='chains', chains=[])
    available = Requirements(window=proposal.requirements.window)
    try:
        w = context.memory.get('declared_service_window')
        check = validate_service_window(context, w)
        if check['status'] != 'READY':
            raise _Unproved(check.get('reason', 'declared window complete/expired'))
        c = proposal.candidate
        if c.get('schema') == SERVICE_WINDOW_SCHEMA:
            expected = declared_service_proposal(context)
            if proposal.to_dict() != expected.to_dict():
                raise _Unproved('exact owner service descriptor and requirements required')
        elif c.get('schema') == RESCUE_SCHEMA:
            if service_window_guard(context, proposal, 'ADMIT').get(RESCUE_AUTHORIZATION) is not True:
                raise _Unproved('current rescue differs from the previously authorized service')
        else:
            raise _Unproved('unsupported declared-window consumer')
        duties = []
        for turn in w['turns'][check['index']:]:
            commands = [turn['action']['farmer'], *turn['action']['hands']]
            if commands[w['worker']] != ['PASS']:
                duties.append(dict(id=f"owned:{w['window_id']}:{turn['step']}", duration=1,
                    release=turn['step'], deadline=turn['step'] + 1,
                    marginal=turn['step'] == w['due_step']))
        model['chains'] = [dict(worker=f"day:{w['worker_generation']['day']}:unit:{w['worker']}",
            entry=context.step, until=w['due_step'] + 1, duties=duties)]
        # Only the allocated carried unit is fungible to THIS exact FEED proposal.
        # No shed wheat, pooled PASS time, future receipts or discretionary money.
        available = Requirements(cash=Z, reserve=Z, inventory={'WHEAT': Estimate.exact(1)},
            market_slots=Z, worker_hours=Estimate.exact(1), land_quadrants=Z,
            window=proposal.requirements.window)
        missing.clear()
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        missing = {name: str(error) for name in CONDITIONS['chains']}
    certs = {} if missing else {name: 'commitments:service-window:' + w['window_id'] + ':' + name
                                for name in CONDITIONS['chains']}
    return CapacityCase.bind(context, proposal, available=available, model=model,
                             certificates=certs, parents=(CHAIN_CALENDAR,)), missing
