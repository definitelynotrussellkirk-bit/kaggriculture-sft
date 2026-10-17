"""Guarded proposal execution over a frozen K0006 continuation.

Coordinator API
--------------
Session(base, context, proposal, guard=..., lowerer=...) owns only its proposal's
named action fields. ``act`` calls the SAME base once on EVERY real observation,
even while overriding. ``observe(post_observation, own_receipt)`` must follow each
intervention. Failure latches PROPOSAL_FAILED; subsequent act calls return only
base actions. COMPLETE here means the bound execution word was realized, NOT that
an economic value, full production tape, or playing-strength claim was proved.

MarketAdapter consumes the accepted market_loop queues (empty ranks preserved;
removed control entries become empty queues). EndgameAdapter consumes the full
collection/liquidation action word or delegates an accepted complete_suffix.
ProductionAdapter also lowers the unchanged intraday LateFeedRescue schema to
ONE sparse FEED, never an economic Plan edit. Like dawn production, callers
must supply a lowerer; this schema permits only the checked NativeLowerer. Bare
compile does not bypass the dawn guard or enable rescue. It regenerates the exact
proposal from current own state/parent action/reservations at compile and execution. Its
Guard must additionally attest RESCUE_AUTHORIZATION at ADMIT and EXECUTE: the
existing active plan/commitment permits this exact conditional service. The
callback owns that truth; availability, survival, and a generic all-guards answer
are NOT authorization. Missing permission refuses. All actor memory (including
any supplied commitment snapshot) is bound by the existing context digest.
After actual own FEED receipts pass, the session observes the remaining base-only
turn through dawn via confirm_rescue. It emits no extra intervention; missing or
contradictory observations permanently fail. COMPLETE certifies only observed
execution/survival, not native payoff, reachable V0 labels or policy strength.

ProductionAdapter and ShopResponseAdapter otherwise consume plan deltas through
lowerer(context, proposal) -> Feasibility. NativeLowerer is the built-in checked
implementation for explicit count-one production ADD bindings and same-dawn
shop-response action prefixes; unsupported coverage stays UNKNOWN. See its
contract below. External lowerer callbacks remain trusted producer extensions. FEASIBLE must carry bind_execution's
source/context/proposal-bound action witness and EXACT candidate delta. A capacity
schedule is NOT such a lowering. Missing routes/ownership/continuation fail closed;
there is deliberately no invented mapping from K0006's sites to production cohorts.
Shop edits additionally require a full own V0 continuation in memory['v0_agent']
and are checked by the existing prefix tracer, including STOP/cap and retained HOLD.

A guard(context, proposal, phase) returns {every proposal.guard: True}. Missing,
false, UNKNOWN, stale or malformed answers fail. The producer owns the scientific
truth of these attestations; strings are NEVER eval'd and a hash is not a proof.
Optional context_factory(observation) refreshes legitimate own memory/publications.
It receives only the own observation, never an engine, seed or rival-private state.
The lowerer owns coverage and worker-effect postconditions; the adapter owns exact
binding, emission, namespace preservation and native fill checks. Every intervention
passes joint_preflight, including whole-field lowerings; report['preflights'] binds
one successful check to each emitted override. Empty adoption does not project or
change the base's actions.

Counterfactual API
------------------
evaluate(snapshot, proposal, continuation=K0006Continuation(...)) delegates to the
UNCHANGED counterfactual.evaluate. It adds serial, read-only native realization
monitoring using the existing market_tap, including the final transition. No game
phase is reimplemented and each interpreter invocation is delegated exactly once.
The monitor receives only actual own commits and own worker effects. It is an
evaluator facility, NOT an actor-side prediction. Without native_realization, a
nonempty direct K0006Continuation.adopt fails: first-action emission alone cannot
certify fills or a later final-turn order. A live coordinator may instead provide
its own equally exact own-receipt transport to Session.observe.

All progress/receipts are local to the session. No accepted modules, dispatch,
engine economics, frozen K0006 bytes or V0 action IDs are changed. Budgets: <=719
patches, <=10 orders/turn, <=100 units/order; no search or automatic native games.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

from farm_neat.evaluation.k0006 import K0006Adapter
from farm_neat.game.exact_engine import DEFAULT_CONFIGURATION, EXPECTED_ENGINE_SHA256, load_engine
from farm_neat.strategy import counterfactual as cf
from farm_neat.strategy.rescue import (SCHEMA as FEED_RESCUE_SCHEMA, LateFeedRescue,
                                      confirm_rescue)
from farm_neat.strategy.types import EVALUATOR_ONLY, Feasibility, Proposal, StrategyContext, Verdict

SCHEMA = 'farm-strategy-adoption-v1'
WITNESS = 'farm-strategy-execution-v1'
LAST_STEP = 718
RESCUE_AUTHORIZATION = 'active plan authorizes this exact intraday FEED rescue'
Guard = Callable[[StrategyContext, Proposal, str], Mapping[str, bool]]
Lowerer = Callable[[StrategyContext, Proposal], Feasibility]


def _wire(value):
    """Normalize actor-owned contract values; reject nested evaluator leakage."""
    if hasattr(value, 'to_dict') and callable(value.to_dict):
        return _wire(value.to_dict())
    if isinstance(value, Mapping):
        if EVALUATOR_ONLY & set(value):
            raise cf.ProposalFailed('evaluator-only key in actor execution input')
        if any(type(k) not in (str, int) for k in value):
            raise cf.ProposalFailed('execution input needs string/integer mapping keys')
        keys = [str(k) for k in value]
        if len(keys) != len(set(keys)):
            raise cf.ProposalFailed('ambiguous serialized mapping keys')
        return {str(k): _wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        json.dumps(value, allow_nan=False)
        return value
    raise cf.ProposalFailed('unsupported actor execution input: ' + type(value).__name__)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(_wire(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def context_digest(context: StrategyContext) -> str:
    # Derived callbacks/certificates are deliberately not serialized. Guard producers
    # must revalidate them, rather than treating callback identity as scientific proof.
    return digest(dict(step=context.step, player=context.player, observation=context.observation,
                       memory=context.memory, shops=context.shops))


def _step(step):
    if type(step) is not int or not 0 <= step <= LAST_STEP:
        raise cf.ProposalFailed('patch step must be an integer in 0..718')
    return step


def _orders(rows):
    engine = load_engine()
    domains = {'SELL': engine.PRODUCTS, 'BUY_PRODUCT': ('WHEAT', 'FERTILIZER'),
               'BUY_SEED': engine.CROPS, 'BUY_ANIMAL': engine.ANIMALS}
    if not isinstance(rows, list) or len(rows) > 10:
        raise cf.ProposalFailed('native market queue needs at most ten explicit ranks')
    for row in rows:
        if not isinstance(row, list):
            raise cf.ProposalFailed('native order must be a list')
        if not row:
            continue
        if row[0] in ('HIRE', 'BUY_LAND') and len(row) == 1:
            continue
        if (len(row) != 3 or row[0] not in domains or row[1] not in domains[row[0]]
                or type(row[2]) is not int or not 1 <= row[2] <= 100):
            raise cf.ProposalFailed('unsupported/malformed native order or quantity outside 1..100')
    return copy.deepcopy(rows)


def _fields(value, *, empty=False):
    if (not isinstance(value, dict) or set(value) - {'farmer', 'hands', 'market'}
            or (not value and not empty)):
        raise cf.ProposalFailed('patch names only farmer, hands and/or market')
    if 'market' in value:
        _orders(value['market'])
    if 'farmer' in value and (not isinstance(value['farmer'], list) or not value['farmer']):
        raise cf.ProposalFailed('farmer action needs a nonempty native array')
    if 'hands' in value and (not isinstance(value['hands'], list)
                            or any(not isinstance(a, list) or not a for a in value['hands'])):
        raise cf.ProposalFailed('hands need explicit native action arrays')
    return _wire(value)


@dataclass(frozen=True)
class Patch:
    """One proposal-named action-field replacement, not a step-dependent strategy rule.

    control matches only the named preimage fields. after maps JSON pointers in the
    post-turn OWN observation to required values. Non-market production effects
    require producer postconditions; a native emission is not production completion.
    """
    step: int
    changes: dict = field(default_factory=dict)
    control: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    before: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)  # native unit index; zero is farmer
    effects: dict = field(default_factory=dict)  # exact per-unit native effect receipts
    sales: dict = field(default_factory=dict)  # replace only this product's SELL ranks
    needs: dict = field(default_factory=dict)  # lower bounds in the actual own observation

    def __post_init__(self):
        _step(self.step)
        object.__setattr__(self, 'changes', _fields(copy.deepcopy(self.changes), empty=True))
        object.__setattr__(self, 'control', _fields(copy.deepcopy(self.control), empty=True))
        for name in ('before', 'after', 'needs'):
            value = getattr(self, name)
            if not isinstance(value, dict) or any(not isinstance(k, str) or not k.startswith('/') for k in value):
                raise cf.ProposalFailed(name + ' checks need own-observation JSON pointers')
            if name == 'needs' and any(type(v) is not int or v < 0 for v in value.values()):
                raise cf.ProposalFailed('resource lower bounds must be nonnegative integers')
            object.__setattr__(self, name, _wire(value))
        units = _wire(self.units)
        if (not isinstance(units, dict) or any(not k.isdecimal() or str(int(k)) != k
                or not isinstance(v, list) or not v for k, v in units.items())):
            raise cf.ProposalFailed('units maps canonical native indices to actions')
        if any(('farmer' if k == '0' else 'hands') in self.changes for k in units):
            raise cf.ProposalFailed('ambiguous whole-field and sparse unit ownership')
        object.__setattr__(self, 'units', units)
        effects = _wire(self.effects)
        if not isinstance(effects, dict) or not set(effects) <= set(units):
            raise cf.ProposalFailed('worker effects must name owned unit indices')
        object.__setattr__(self, 'effects', effects)
        sales = _wire(self.sales)
        if not isinstance(sales, dict) or ('market' in self.changes and sales):
            raise cf.ProposalFailed('ambiguous whole-market and product sale ownership')
        for product, rows in sales.items():
            if product not in load_engine().PRODUCTS or any(not row or row[:2] != ['SELL', product]
                                                           for row in _orders(rows)):
                raise cf.ProposalFailed('sparse sales own only the named product SELL orders')
        object.__setattr__(self, 'sales', sales)
        if not (self.changes or units or sales):
            raise cf.ProposalFailed('patch must own an action field, unit or product sale')

    def to_dict(self):
        return copy.deepcopy(asdict(self))


def bind_execution(context, proposal, patches, *, delta, parents):
    """Producer helper, not a completeness proof. Supply only a checked full lowering."""
    patches = tuple(patches)
    if not parents or not all(isinstance(p, str) and p for p in parents):
        raise cf.ProposalFailed('lowering requires explicit producer provenance')
    if not all(isinstance(p, Patch) for p in patches):
        raise cf.ProposalFailed('lowering requires Patch objects')
    return dict(schema=WITNESS, proposal_sha256=digest(proposal.to_dict()),
                context_sha256=context_digest(context), delta=_wire(delta),
                patches=[p.to_dict() for p in patches], parents=list(parents))


def _lower(context, proposal, lowerer, delta):
    if lowerer is None:
        raise cf.ProposalFailed('plan delta requires a bound complete action lowerer; no routes invented')
    answer = lowerer(context, Proposal.from_dict(proposal.to_dict()))
    if (not isinstance(answer, Feasibility) or answer.proposal != proposal.key
            or answer.verdict is not Verdict.FEASIBLE):
        raise cf.ProposalFailed('lowerer did not certify this proposal FEASIBLE: '
                                + str(getattr(lowerer, 'report', {})))
    value = answer.witness
    if (not isinstance(value, dict) or set(value) != {'schema', 'proposal_sha256', 'context_sha256',
                                                    'delta', 'patches', 'parents'}
            or value['schema'] != WITNESS or value['proposal_sha256'] != digest(proposal.to_dict())
            or value['context_sha256'] != context_digest(context) or value['delta'] != _wire(delta)
            or not value['parents']):
        raise cf.ProposalFailed('lowering witness does not bind the exact proposal, context and delta')
    if not isinstance(value['patches'], list) or not value['patches']:
        raise cf.ProposalFailed('nonempty plan delta requires a nonempty execution witness')
    return tuple(Patch(**row) for row in value['patches'])


class MarketAdapter:
    """Full, rank-preserving native queue replacement, with K0006 preimage guards."""
    def compile(self, context, proposal, lowerer=None):
        c = proposal.candidate
        if c.get('schema') != 'farm-market-loop-v1':
            raise cf.ProposalFailed('accepted market_loop schema required')
        if c.get('status') == 'KEEP_BASELINE' and c.get('family') == 'PASS' and c.get('orders') == {}:
            return ()
        if c.get('status') != 'CERTIFIED':
            raise cf.ProposalFailed('uncertified/forecast/UNKNOWN market candidate is not executable')
        def schedule(value):
            if not isinstance(value, dict):
                raise cf.ProposalFailed('explicit market schedule and control required')
            result = {}
            for key, queue in value.items():
                if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
                    raise cf.ProposalFailed('canonical decimal market step required')
                result[_step(int(key))] = _orders(queue)
            return result
        candidate, control = schedule(c.get('orders')), schedule(c.get('control_orders'))
        # Omitted control entries represent canceled queues, not an invitation to
        # leave their K0006 actions running. Gap steps are checked by Session too.
        return tuple(Patch(s, {'market': candidate.get(s, [])}, {'market': control.get(s, [])})
                     for s in sorted(candidate.keys() | control.keys()))


class EndgameAdapter:
    def compile(self, context, proposal, lowerer=None):
        from farm_neat.strategy.endgame import context_key
        c = proposal.candidate
        if c.get('context_key') != context_key(context) or c.get('start_step') != context.step:
            raise cf.ProposalFailed('endgame actor context binding changed')
        if c.get('terminal_step') != LAST_STEP + 1:
            raise cf.ProposalFailed('endgame must bind the actual terminal boundary')
        if c.get('family') == 'complete_suffix':
            inner = Proposal.from_dict(c['adopt'])
            if inner.module == 'endgame':
                raise cf.ProposalFailed('recursive endgame delegation is not supported')
            if not set(inner.guards) <= set(proposal.guards) or inner.requirements != proposal.requirements:
                raise cf.ProposalFailed('endgame delegation cannot drop inner guards or resource requirements')
            return adapter_for(inner).compile(context, inner, lowerer)
        if c.get('family') != 'collection_liquidation' or c.get('engine_sha256') != EXPECTED_ENGINE_SHA256:
            raise cf.ProposalFailed('source-bound collection/liquidation witness required')
        actions = c.get('actions')
        if not isinstance(actions, list) or len(actions) != LAST_STEP + 1 - context.step:
            raise cf.ProposalFailed('endgame word must cover each remaining native action')
        if any(set(a) != {'farmer', 'hands', 'market'} for a in actions):
            raise cf.ProposalFailed('endgame witness needs complete native actions')
        return tuple(Patch(context.step + i, action) for i, action in enumerate(actions))


def _is_feed_rescue(proposal):
    return (proposal.module == 'production' and isinstance(proposal.candidate, dict)
            and proposal.candidate.get('schema') == FEED_RESCUE_SCHEMA)


def _validate_feed_rescue(context, proposal):
    """Re-use the unchanged producer's bounded guards, including ALL parent units.

    Exact wire equality also binds requirements, claims, guards and provenance;
    matching a target or merely copying a candidate hash is insufficient.
    """
    candidates = LateFeedRescue().propose(context)
    if not any(digest(p.to_dict()) == digest(proposal.to_dict()) for p in candidates):
        raise cf.ProposalFailed('rescue proposal no longer matches current actor/parent/resource bindings')


class ProductionAdapter:
    def compile(self, context, proposal, lowerer=None):
        from farm_neat.strategy.production import ProductionBook
        c = proposal.candidate
        if _is_feed_rescue(proposal) and lowerer is not None:
            if type(lowerer) is not NativeLowerer:
                raise cf.ProposalFailed('rescue requires the checked NativeLowerer, not a generic action patch')
            return _lower(context, proposal, lowerer, c)
        if c.get('schema') != 'farm-production-proposal-v1' or context.step % 24:
            raise cf.ProposalFailed('accepted dawn production delta required')
        book = context.memory.get('production')
        if not isinstance(book, ProductionBook):
            raise cf.ProposalFailed('production adoption requires the original own production book')
        snapshot = digest(dict(step=context.step, player=context.player,
                               observation=dict(context.observation), shops=context.shops, book=book.to_dict()))
        if c.get('snapshot_sha256') != snapshot or c.get('published_shops') != list(context.shops):
            raise cf.ProposalFailed('production book/observation/shop binding changed')
        if c.get('kind') == 'KEEP' and c.get('changes') == {}:
            return ()
        if c.get('kind') not in ('ADD', 'RETIRE'):
            raise cf.ProposalFailed('unsupported production delta')
        return _lower(context, proposal, lowerer, c['changes'])


class ShopResponseAdapter:
    def compile(self, context, proposal, lowerer=None):
        from farm_neat.economy.v0.epochs import KeepPolicy
        from farm_neat.training.teachers import v0_oracle
        c = proposal.candidate
        if c.get('kind') != 'v0_edit' or context.step % 24:
            raise cf.ProposalFailed('shop_response needs a dawn V0 edit')
        if c.get('ordered_shops') != list(context.shops):
            raise cf.ProposalFailed('shop_response ordered shop prefix changed')
        memory = context.memory.get('v0_agent')
        if memory is None:
            raise cf.ProposalFailed('full own V0 continuation required; Plan alone loses HOLD/controller state')
        agent = v0_oracle.restore_agent(copy.deepcopy(memory), KeepPolicy())
        if agent.plan.digest() != c.get('base_plan_sha256'):
            raise cf.ProposalFailed('shop_response base canonical Plan changed')
        if agent.controller is not None:
            agent.controller.observe(copy.deepcopy(dict(context.observation)))
        elif context.step:
            raise cf.ProposalFailed('later dawn requires a retained own controller')
        trace = v0_oracle.trace_edits(v0_oracle.capture_dawn(agent, dict(context.observation)), c.get('edits'))
        if trace['plan'] != c.get('target_plan'):
            raise cf.ProposalFailed('shop_response target is not exactly prefix reached')
        return _lower(context, proposal, lowerer,
                      dict(target_plan=trace['plan'], hold_releases=trace['hold_releases'], edits=trace['actions']))


class ActionAdapter:
    def compile(self, context, proposal, lowerer=None):
        c = proposal.candidate
        if set(c) != {'kind', 'action'} or c['kind'] != 'action':
            raise cf.ProposalFailed('unsupported raw action candidate')
        return (Patch(context.step, c['action']),)


def adapter_for(proposal):
    families = {'market_loop': MarketAdapter, 'endgame': EndgameAdapter,
                'production': ProductionAdapter, 'shop_response': ShopResponseAdapter}
    if proposal.module in families:
        return families[proposal.module]()
    if isinstance(proposal.candidate, dict) and proposal.candidate.get('kind') == 'action':
        return ActionAdapter()
    raise cf.ProposalFailed('no adoption adapter for this proposal family')


def _pointer(value, pointer):
    for key in pointer.split('/')[1:]:
        key = key.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _requested(queue):
    values = Counter()
    for row in _orders(queue):
        if row:
            values[row[0], row[1] if len(row) > 1 else None] += row[2] if len(row) > 2 else 1
    return values


def _stock_diff(before, after):
    return {k: after.get(k, 0) - before.get(k, 0) for k in sorted(before.keys() | after.keys())
            if after.get(k, 0) != before.get(k, 0)}


def _unit_effect(engine, farm, private, unit, action, day):
    """Delegate one OWN native subphase and record the same receipt as market_tap.

    No environment, market, random seed, rival state, or interpreter is constructed.
    The native interpreter's cross-worker PLANT gate is applied by _project_units.
    """
    pos = engine._farmer_position(farm, unit)
    if pos is None:
        raise cf.ProposalFailed('bound native worker does not exist')
    position = list(pos)
    tile = copy.deepcopy(farm['tiles'][pos[1]][pos[0]])
    def stocks():
        carried = Counter()
        for inv in private['inventories']:
            carried.update(inv)
        return dict(carried=dict(carried), seeds=dict(private['seeds']), shed=dict(private['shed']))
    before = stocks()
    engine._apply_unit_action(farm, private, unit, action, 10, day, 24, 100)
    if action[0] in ('PASS', 'NORTH', 'SOUTH', 'EAST', 'WEST'):
        return None
    after = stocks()
    return dict(unit=unit, engine_received_action=copy.deepcopy(action), position=position,
                delta={k: _stock_diff(before[k], after[k]) for k in before}, tile_before=tile,
                tile_after=copy.deepcopy(farm['tiles'][position[1]][position[0]]))


def _project_units(observation, action):
    """Exact deterministic own worker subphase, including atomic PLANT cancellation."""
    engine = load_engine()
    farm = copy.deepcopy(observation['farms'][observation['player']])
    private = copy.deepcopy(observation['private'])
    acts = [action.get('farmer', ['PASS']), *action.get('hands', [])]
    demand = Counter(a[1] for a in acts if len(a) >= 2 and a[0] == 'PLANT')
    blocked = {p for p, n in demand.items() if n > private['seeds'].get(p, 0)}
    effects = {}
    for unit, act in enumerate(acts):
        if unit > len(farm['hands']):
            continue  # Reference ignores nonexistent hand orders.
        allowed = ['PASS'] if len(act) >= 2 and act[0] == 'PLANT' and act[1] in blocked else act
        event = _unit_effect(engine, farm, private, unit, allowed, observation['step'] // 24)
        if event is not None:
            effects[str(unit)] = event
    return farm, private, effects


def _finish_own_turn(farm, private, step):
    """Deterministic own boundary only; random weeds/shops are not predicted.

    Markets do not mutate an existing bound worker's carried inventory or site.
    Actual receipts remain necessary, especially for a dawn random weed event.
    """
    engine = load_engine()
    engine._decay_plants(farm, step)
    if (step + 1) % 24 == 0:
        engine._daily_refresh_plants(farm, step // 24, 24)
        engine._daily_refresh_animals(farm, step // 24)
        engine._drop_inventories_to_shed(private, 100)
        farm['farmer'] = list(engine._default_spawn(10))
        farm['hands'], farm['hires_today'], private['inventories'] = [], 0, [{}]


def _merge_patch(baseline, patch, observation):
    result = copy.deepcopy(baseline)
    result.update(copy.deepcopy(patch.changes))
    for key, action in patch.units.items():
        unit = int(key)
        if unit > len(observation['farms'][observation['player']]['hands']):
            raise cf.ProposalFailed('bound worker disappeared; no new worker identity invented')
        if unit == 0:
            result['farmer'] = copy.deepcopy(action)
        else:
            hands = result.setdefault('hands', [])
            while len(hands) < unit:
                hands.append(['PASS'])
            hands[unit - 1] = copy.deepcopy(action)
    if patch.sales:
        queue = _orders(result.get('market', []))
        for product, orders in sorted(patch.sales.items()):
            if orders and any(row and row[:2] in (['BUY_PRODUCT', product], ['BUY_ANIMAL', product])
                              for row in queue):
                raise cf.ProposalFailed('compiled sale conflicts with a same-product baseline acquisition')
            ranks = [i for i, row in enumerate(queue) if row and row[:2] == ['SELL', product]]
            for i in ranks:
                queue[i] = []  # Never delete ranks and move an unrelated lockstep order.
            available = ranks + [i for i, row in enumerate(queue) if not row and i not in ranks]
            for order in orders:
                if available:
                    queue[available.pop(0)] = copy.deepcopy(order)
                elif len(queue) < 10:
                    queue.append(copy.deepcopy(order))
                else:
                    raise cf.ProposalFailed('no unowned market rank for compiled sale; unrelated orders retained')
        result['market'] = queue
    return result


def joint_preflight(observation, action, patch, configuration=None):
    """Check the complete merged OWN worker word before an override is emitted.

    Inputs are the current own observation, merged native action, its bound Patch
    and the execution configuration. No seed, engine snapshot, rival-private data
    or future shop state is used. This delegates native worker subphases, including
    atomic PLANT cancellation, then checks the final own tile/worker endpoint.
    Whole-field and sparse lowerings use the same check; neither may bypass it.

    Returns an action/observation/patch-bound audit record, or raises ProposalFailed.
    It is NOT a market-fill, random-weed, valuation or full-plan certificate. Market
    and other postconditions still require Session.observe's actual own receipts.
    Unowned K0006 actions are never repaired or suppressed to satisfy a proposal.
    """
    if not isinstance(patch, Patch) or observation.get('step') != patch.step:
        raise cf.ProposalFailed('joint preflight requires the current patch step')
    config = configuration or {}
    for key in ('episodeSteps', 'turnsPerDay', 'boardSize', 'maxMarketOrdersPerTurn',
                'shedCapacity', 'farmHandCostMult'):
        if config.get(key, DEFAULT_CONFIGURATION[key]) != DEFAULT_CONFIGURATION[key]:
            raise cf.ProposalFailed('native lowerer requires pinned default execution configuration')
    farm, private, effects = _project_units(observation, action)
    for unit, expected in patch.effects.items():
        if effects.get(unit) != expected:
            raise cf.ProposalFailed('actual joint worker word cannot realize certified native effect')
    for product, orders in patch.sales.items():
        if sum(row[2] for row in orders) > private['shed'].get(product, 0):
            raise cf.ProposalFailed('compiled sale lacks actual post-worker shed stock')
    # A later, preserved worker can undo an earlier owned effect. Check the joint
    # endpoint after deterministic day-end processing, not just the per-unit event.
    _finish_own_turn(farm, private, patch.step)
    projected = dict(observation)
    projected['farms'] = list(observation['farms'])
    projected['farms'][observation['player']] = farm
    projected['private'] = private
    own_prefix = '/farms/' + str(observation['player']) + '/'
    checked = []
    for pointer, expected in patch.after.items():
        if pointer.startswith((own_prefix, '/private/inventories/')):
            if _pointer(projected, pointer) != expected:
                raise cf.ProposalFailed('joint worker endpoint conflicts at ' + pointer)
            checked.append(pointer)
    return dict(schema='farm-strategy-joint-preflight-v1', step=patch.step,
                scope='OWN_WORKER_ENDPOINT_NOT_MARKET_FILL_OR_VALUE',
                observation_sha256=digest(observation), action_sha256=digest(action),
                patch_sha256=digest(patch.to_dict()), checked_after=sorted(checked),
                checked_effects=sorted(patch.effects), checked_sales=sorted(patch.sales))


def _reservation_request(context, cohort_key):
    requests = context.memory.get('production_reservations', {})
    if not isinstance(requests, Mapping):
        raise cf.ProposalFailed('RESERVATION_AUTHORITY: production_reservations must be a mapping')
    return requests.get(cohort_key)


class _ProductionReservation:
    """Proposal-local consumption ledger, not a second commitment owner.

    NativeLowerer supplies the exact word. The existing DECLARED-WINDOW authority
    reader owns FullSeasonPlan publication/revision/obligation validation; its
    current claim reader and coordinator namespace rule own competing claims.
    We never issue a service-window certificate for this longer production word.
    That owner's current authorization envelope is DAILY FEED only. Other future
    consumables refuse instead of acquiring invented economic permission.

    A current shed floor covers EVERY outstanding pickup, with no credit for
    future buys, harvests, deposits or sales returning cash. Worker custody and
    actual FEED are already exact Patch postconditions. Only verified receipts
    advance the balances. A competing parent action expires this reservation;
    Session returns the unmodified parent, never cancels its orders or buys feed.
    The caller still owns completeness of rescue_reservations and real receipts.
    """
    def __init__(self, context, proposal, patches, cohort, binding):
        from farm_neat.strategy.commitments import _service_authority
        self.cohort, self.binding = cohort.key, copy.deepcopy(binding)
        self.request = copy.deepcopy(_reservation_request(context, cohort.key))
        self.rules = _wire(context.memory.get('capacity_rules', {}))
        if (not isinstance(self.request, dict) or set(self.request) != {'obligation_id', 'revision'}
                or not isinstance(self.request['obligation_id'], str)
                or type(self.request['revision']) is not int):
            raise cf.ProposalFailed('RESERVATION_AUTHORITY: explicit obligation_id/revision binding required')
        unit = str(binding['unit'])
        self.feeds = {p.step: -p.effects[unit]['delta']['carried']['WHEAT'] for p in patches
                      if p.units.get(unit) == ['FEED']}
        self.pickups = {p.step: p.needs['/private/shed/WHEAT'] for p in patches
                        if '/private/shed/WHEAT' in p.needs}
        if cohort.producer not in load_engine().ANIMALS or not self.feeds:
            raise cf.ProposalFailed('RESERVATION_AUTHORITY_UNSUPPORTED: owner currently authorizes DAILY FEED only')
        try:
            self.authority = _service_authority(context, self.request['obligation_id'], max(self.feeds))
            first = _service_authority(context, self.request['obligation_id'], min(self.feeds))
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            raise cf.ProposalFailed('RESERVATION_AUTHORITY: ' + str(error)) from error
        command = self.authority['command']
        if (first != self.authority or self.request['revision'] != self.authority['revision']
                or command['animal'] != cohort.producer
                or [command['scope']['x'], command['scope']['y']] != binding['tile']):
            raise cf.ProposalFailed('RESERVATION_AUTHORITY: revision/species/tile differs from owner obligation')
        x, y = binding['tile']
        self.touches = ('labor', 'land', 'unit:' + unit, 'inventory:WHEAT',
                        f'animal:{x}:{y}', f'tile:{x}:{y}')
        self.status, self.reason = 'ACTIVE', None
        self.next_step, self.last_step = context.step, patches[-1].step
        self.remaining = sum(self.feeds.values())
        self.identity = dict(schema='farm-adoption-consumable-reservation-v1',
            proposal_sha256=digest(proposal.to_dict()), patches_sha256=digest([p.to_dict() for p in patches]),
            context_sha256=context_digest(context), authority=self.authority, cohort=self.cohort,
            binding=self.binding, touches=list(self.touches), feeds=self.feeds, pickups=self.pickups,
            start_step=context.step, last_step=self.last_step,
            scope='PREAUTHORIZED_DAILY_FEED_STOCK_NOT_SEASON_CAPACITY_OR_VALUE')
        self.reservation_id = digest(self.identity)
        self.validate(context)

    def to_dict(self):
        return _wire(dict(self.identity, reservation_id=self.reservation_id,
            status=self.status, reason=self.reason, next_step=self.next_step,
            remaining={'WHEAT': self.remaining}, shed_required={'WHEAT': self.shed_required(self.next_step)}))

    def shed_required(self, step):
        return sum(n for due, n in self.pickups.items() if due >= step)

    def _stock(self, private, step):
        available = private['shed'].get('WHEAT', 0)
        if type(available) is not int or available < self.shed_required(step):
            raise cf.ProposalFailed('RESERVATION_CONFLICT: remaining WHEAT shed pickups are not covered')

    def validate(self, context):
        from farm_neat.strategy.commitments import _service_authority, _service_claims
        from farm_neat.strategy.coordinator import resource_conflicts
        if self.status != 'ACTIVE' or context.step != self.next_step:
            raise cf.ProposalFailed('RESERVATION_EXPIRED: stale or unobserved reservation prefix')
        try:
            authority = _service_authority(context, self.request['obligation_id'], max(self.feeds))
            claims = _service_claims(context)
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            raise cf.ProposalFailed('RESERVATION_AUTHORITY: ' + str(error)) from error
        if (authority != self.authority
                or _reservation_request(context, self.cohort) != self.request
                or not isinstance(context.memory.get('production_bindings'), Mapping)
                or context.memory['production_bindings'].get(self.cohort) != self.binding
                or _wire(context.memory.get('capacity_rules', {})) != self.rules):
            raise cf.ProposalFailed('RESERVATION_REVISION_DRIFT: owner publication/obligation/binding changed')
        conflicts = resource_conflicts(self.touches, claims)
        if conflicts:
            raise cf.ProposalFailed('RESERVATION_CONFLICT: protected claims ' + ', '.join(conflicts))
        self._stock(context.observation['private'], context.step)

    def preflight(self, observation, action):
        # Do not forecast a market fill or credit a BUY. All submitted SELL units
        # must fit outside the reserve even when another order might replenish it.
        _, private, _ = _project_units(observation, action)
        sales = sum(row[2] for row in _orders(action.get('market', []))
                    if row and row[:2] == ['SELL', 'WHEAT'])
        private['shed']['WHEAT'] = private['shed'].get('WHEAT', 0) - sales
        self._stock(private, observation['step'] + 1)

    def observe(self, after):
        # Called ONLY after Session has checked all exact worker effects and
        # postconditions against the actual own transport receipt.
        if after['step'] != self.next_step + 1:
            raise cf.ProposalFailed('RESERVATION_EXPIRED: noncontiguous consumption receipt')
        self._stock(after['private'], after['step'])
        self.remaining -= self.feeds.get(self.next_step, 0)
        self.next_step += 1
        if self.next_step > self.last_step:
            self.status = 'COMPLETE'

    def expire(self, reason):
        self.status, self.reason = 'EXPIRED', str(reason)


def _reserve_production(context, proposal, patches, cohort, binding):
    """Require owner-bound reservations for future consumables; opt in on day 29.

    Select the existing owner's obligation in memory['production_reservations']
    as {cohort_key: {'obligation_id': str, 'revision': int}}. The published active
    plan must observe EVERY real prefix through context_factory. Supply the
    existing complete current rescue_reservations (other claims, excluding this
    session's own touches); no new competing-claim registry is introduced.
    """
    future = any(p.step // 24 > context.step // 24 and
        (any(k.startswith('/private/seeds/') for k in p.needs)
         or any(n < 0 and product in ('WHEAT', 'FERTILIZER')
                for e in p.effects.values() for product, n in e['delta']['carried'].items()))
        for p in patches)
    _reservation_request(context, cohort.key)  # Validate the mapping even when no entry is present.
    requested = cohort.key in context.memory.get('production_reservations', {})
    if not future and not requested:
        return None  # Existing single-day fixtures/ownership and trusted lowerers are unchanged.
    return _ProductionReservation(context, proposal, patches, cohort, binding)


class NativeLowerer:
    """Checked, actor-only construction, not an arbitrary witness-signing callback.

    Production ADD: one explicitly bound cohort instance, count=1, owned empty
    tile, existing assets only. memory['production_bindings'][cohort_key] is
    {'unit': native_index, 'tile': [x,y]}. This reserves that worker for this
    cohort's remaining lifetime; it does NOT infer a K0006 site assignment.
    Farmer bindings survive the native daily reset. Hand bindings are supported
    only when no reset remains. Exact engine subphases construct every move,
    pickup, placement, service and harvest. The FULL added module's dated output,
    input and service-attempt tape must equal project_cohorts. Native receipts,
    tile/position/inventory checks and resource guards enforce it in the real game.
    Current-day animal/feed prerequisites are collected on one shed visit before
    routing to the site; quantities are only the missing carried amounts. This
    shortens shared-stock exposure. Future consumables additionally require the
    revision-bound _reserve_production contract; the existing owner currently
    authorizes DAILY FEED only. A conflicting preserved worker/sale expires the
    reservation before the override rather than silently changing base behavior.
    Other units, markets and K0006's namespace are untouched. Future shop/weed,
    asset contention or route changes invalidate rather than silently replan.
    No cash/future purchases, land acquisition, multi-instance routing or RETIRE
    continuation ownership is guessed: those cases return UNKNOWN.

    Shop response: execute the real prefix-traced V0 compiler on an independent
    complete own continuation. Lower only the changed product's native SELLs,
    with unrelated order ranks preserved. The accepted producer explicitly
    invalidates on 'next step': coverage is its DAWN ACTION PREFIX, not a claim
    that K0006 thereafter follows the target V0 policy. The report retains only
    the traced Plan/HOLD edit result, NOT the shadow compiler's unexecuted full
    continuation. A coordinator must revalidate its own live state. No caller may
    relabel this prefix as a whole-season V0 behavior-equivalence certificate.

    No games/search are launched. Budget: <=719 native own-unit subphase checks,
    one V0 dawn compilation. One instance per concurrent request (report is local).
    """
    def __init__(self):
        self.report = {}
        self.reservation = None

    def __call__(self, context, proposal):
        self.report = {}
        self.reservation = None
        parents = ('src/farm_neat/strategy/adoption.py',
                   'src/farm_neat/game/kaggriculture_engine_reference.py')
        try:
            context_digest(context)
            if _is_feed_rescue(proposal):
                patches, delta = self._feed_rescue(context, proposal), proposal.candidate
                self.report = dict(scope='INTRADAY_FEED_CONDITIONAL_ON_ACTIVE_PLAN_AUTHORIZATION',
                                   own_cash_delta='UNKNOWN', rival_cash_delta='UNKNOWN',
                                   margin_delta='UNKNOWN', WDL='UNKNOWN')
            elif proposal.module == 'production':
                patches, delta = self._production(context, proposal)
            elif proposal.module == 'shop_response':
                patches, delta = self._shop(context, proposal)
            else:
                raise cf.ProposalFailed('NativeLowerer owns production and shop_response only')
            witness = bind_execution(context, proposal, patches, delta=delta, parents=parents)
            self.report.update(verdict='FEASIBLE', patches=len(patches),
                               engine_sha256=EXPECTED_ENGINE_SHA256)
            return Feasibility(proposal.key, Verdict.FEASIBLE, witness=witness, parents=parents)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.report = dict(verdict='UNKNOWN', reason=str(error))
            return Feasibility(proposal.key, Verdict.UNKNOWN, parents=parents)

    def _feed_rescue(self, context, proposal):
        _validate_feed_rescue(context, proposal)
        c, obs = proposal.candidate, context.observation
        unit, (x, y) = c['actor'], c['target']
        control = copy.deepcopy(c['control_action'])
        units = {str(unit): ['FEED']}
        # Construct from the FULL emitted word, not an isolated FEED worker.
        action = _merge_patch(control, Patch(context.step, units=units), obs)
        farm, private, effects = _project_units(obs, action)
        effect = effects[str(unit)]
        if (effect['engine_received_action'] != ['FEED']
                or effect['delta']['carried'].get('WHEAT') != -1
                or effect['tile_after'].get('fed_today') is not True):
            raise cf.ProposalFailed('rescue FEED has no exact carried-wheat/native service effect')
        _finish_own_turn(farm, private, context.step)
        prefix = '/farms/' + str(context.player)
        tile = prefix + '/tiles/' + str(y) + '/' + str(x)
        position = prefix + ('/farmer' if unit == 0 else '/hands/' + str(unit - 1))
        inventory = '/private/inventories/' + str(unit)
        before = {tile: obs['farms'][context.player]['tiles'][y][x],
                  position: c['target'], inventory: obs['private']['inventories'][unit]}
        after = {tile: farm['tiles'][y][x]}
        if (context.step + 1) % 24:
            after.update({position: c['target'], inventory: private['inventories'][unit]})
        else:
            # Native dawn retires hands and drops inventories; never invent a
            # surviving hand index or demand pre-dawn wheat custody after reset.
            after.update({prefix + '/farmer': farm['farmer'], prefix + '/hands': farm['hands'],
                          '/private/inventories': private['inventories']})
        return (Patch(context.step, units=units, control=control, before=before, after=after,
                      effects={str(unit): effect}, needs={inventory + '/WHEAT': 1}),)

    def _shop(self, context, proposal):
        from farm_neat.economy.v0 import vocabulary as V
        from farm_neat.economy.v0.epochs import KeepPolicy
        from farm_neat.economy.v0.plan import Plan
        from farm_neat.training.teachers import v0_oracle
        c = proposal.candidate
        if (context.step % 24 or c.get('kind') != 'v0_edit'
                or c.get('family') not in ('MARKET_CLEAR', 'MARKET_HOLD')
                or c.get('ordered_shops') != list(context.shops)
                or 'next step' not in proposal.invalidated_by):
            raise cf.ProposalFailed('shop lowering requires the accepted dawn/next-step response boundary')
        agent = v0_oracle.restore_agent(copy.deepcopy(context.memory['v0_agent']), KeepPolicy())
        before = agent.plan.copy()
        if before.digest() != c.get('base_plan_sha256'):
            raise cf.ProposalFailed('shop base continuation changed')
        target = Plan.from_canonical(c['target_plan'])
        if (target.cohorts, target.regimes, target.land) != (before.cohorts, before.regimes, before.land):
            raise cf.ProposalFailed('shop lowering cannot own production or land changes')
        changed = [i for i in range(len(V.PRODUCTS)) if before.market.get(i) != target.market.get(i)]
        if len(changed) != 1:
            raise cf.ProposalFailed('one changed shop product required')
        product = changed[0]
        expected_kind = c['family'].removeprefix('MARKET_')
        if V.PRIMITIVES[target.overlay(product)[0]] != expected_kind:
            raise cf.ProposalFailed('shop family does not match reached overlay')
        # The existing default adopter performs prefix tracing inside V0Agent's
        # observed epoch and checks Plan/HOLD equality and compiler refusals.
        inner = Proposal('native_lowerer', 'shop', {'kind': 'v0_edits', 'actions': c['edits']},
                         ('plan:market:' + V.PRODUCTS[product],))
        handle = cf.DefaultContinuation().adopt(agent, context, inner)
        action = handle.first_action(agent, copy.deepcopy(dict(context.observation)), DEFAULT_CONFIGURATION)
        if agent.plan.canonical() != target.canonical() or agent.records[-1]['unplaced']:
            raise cf.ProposalFailed('shop target not reached or compiler left instances unplaced')
        continuation = v0_oracle.snapshot_agent(agent)
        delta = dict(target_plan=target.canonical(), hold_releases=continuation['hold_releases'],
                     edits=list(c['edits']))
        # DefaultContinuation has already canonicalized the edit sequence. Bind
        # the same canonical actions as ShopResponseAdapter, rather than raw aliases.
        restored = v0_oracle.restore_agent(copy.deepcopy(context.memory['v0_agent']), KeepPolicy())
        if restored.controller is not None:
            restored.controller.observe(copy.deepcopy(dict(context.observation)))
        trace = v0_oracle.trace_edits(v0_oracle.capture_dawn(restored, dict(context.observation)), c['edits'])
        delta['edits'] = trace['actions']
        if delta['hold_releases'] != trace['hold_releases']:
            raise cf.ProposalFailed('shop lowering changed retained HOLD continuation')
        name = V.PRODUCTS[product]
        sales = [row for row in action['market'] if row and row[:2] == ['SELL', name]]
        self.report = dict(scope='DAWN_ACTION_PREFIX_INVALIDATED_NEXT_STEP', valid_through=context.step,
                           next_action='Revalidate before any next-step override; do not install a teacher prefix.',
                           edit_result=copy.deepcopy(delta), compiled_action=action,
                           continuation_scope='SHADOW_COMPILE_ONLY_NOT_ACTUAL_V0_CONTINUATION')
        return (Patch(context.step, sales={name: sales}),), delta

    def _production(self, context, proposal):
        from farm_neat.strategy.production import Cohort, ProductionBook, project_cohorts
        from farm_neat.evaluation.unit_economics import _one_shot_harvest_age
        c = proposal.candidate
        if c.get('schema') != 'farm-production-proposal-v1' or c.get('kind') != 'ADD' or context.step % 24:
            raise cf.ProposalFailed('checked production lowering currently supports dawn ADD, not RETIRE')
        book = context.memory['production']
        if not isinstance(book, ProductionBook):
            raise cf.ProposalFailed('original ProductionBook required')
        expected_snapshot = digest(dict(step=context.step, player=context.player,
            observation=dict(context.observation), shops=context.shops, book=book.to_dict()))
        if c['snapshot_sha256'] != expected_snapshot or c['published_shops'] != list(context.shops):
            raise cf.ProposalFailed('production source binding changed')
        request = next((e for e in book.expansions if e.key == c['request_key']), None)
        if (request is None or request.land_quadrants or len(request.cohorts) != 1
                or request.cohorts[0].count != 1
                or c['changes'] != {'add': [x.to_dict() for x in request.cohorts]}):
            raise cf.ProposalFailed('one explicitly requested count-one cohort on already owned land required')
        cohort = request.cohorts[0]
        if cohort.birth_day < context.step // 24:
            raise cf.ProposalFailed('cannot construct an unobserved past birth')
        baseline = project_cohorts(context, book.cohorts)
        target = project_cohorts(context, book.cohorts + request.cohorts)
        if digest(baseline.to_dict()) != c['baseline_sha256'] or target.to_dict() != c['tape']:
            raise cf.ProposalFailed('candidate full production tape changed')
        if sum(target.fixed_cash) != sum(baseline.fixed_cash):
            raise cf.ProposalFailed('future producer purchases need a separate funded acquisition binding')
        tape = project_cohorts(context, (cohort,))
        binding = context.memory['production_bindings'][cohort.key]
        if not isinstance(binding, dict) or set(binding) != {'unit', 'tile'}:
            raise cf.ProposalFailed('explicit cohort unit/tile ownership binding required')
        unit, tile = binding['unit'], binding['tile']
        if (type(unit) is not int or unit < 0 or not isinstance(tile, list) or len(tile) != 2
                or any(type(v) is not int or not 0 <= v < 10 for v in tile)):
            raise cf.ProposalFailed('invalid native worker/site binding')
        if unit and context.step // 24 != LAST_STEP // 24:
            raise cf.ProposalFailed('hand identity does not survive native dawn reset; future hires uncertified')
        engine = load_engine()
        obs = copy.deepcopy(dict(context.observation))
        own, private = obs['farms'][context.player], obs['private']
        if len(own['tiles']) != 10 or any(len(row) != 10 for row in own['tiles']):
            raise cf.ProposalFailed('native binding requires the default board')
        if unit > len(own['hands']) or len(private['inventories']) <= unit:
            raise cf.ProposalFailed('bound worker missing from actual own farm')
        if own['tiles'][tile[1]][tile[0]] is not None:
            raise cf.ProposalFailed('ADD needs an explicitly owned empty site; no baseline asset retired')
        patches, outputs, inputs, services = [], {}, {}, [0] * 30
        for product in engine.PRODUCTS:
            outputs[product] = [0] * 30
        for product in ('WHEAT', 'FERTILIZER'):
            inputs[product] = [0] * 30
        step = context.step
        prefix = '/farms/' + str(context.player)
        position_path = prefix + ('/farmer' if unit == 0 else '/hands/' + str(unit - 1))
        tile_path = prefix + '/tiles/' + str(tile[1]) + '/' + str(tile[0])
        inventory_path = '/private/inventories/' + str(unit)
        def pos(): return engine._farmer_position(own, unit)
        def current_tile(): return own['tiles'][tile[1]][tile[0]]
        def emit(action, *, service=False):
            nonlocal step
            day = step // 24
            if step > LAST_STEP or day != active_day:
                raise cf.ProposalFailed('bound route/service word exceeds its native day budget')
            before = {position_path: list(pos()), tile_path: copy.deepcopy(current_tile()),
                      inventory_path: copy.deepcopy(private['inventories'][unit])}
            needs = {}
            if action[0] == 'PLANT':
                needs['/private/seeds/' + action[1]] = 1
            if action[0] == 'PICKUP':
                needs['/private/shed/' + action[1]] = action[2]
                if private['shed'].get(action[1], 0) < action[2]:
                    raise cf.ProposalFailed('RESERVATION_CONFLICT: unfunded native pickup: ' + action[1])
            effect = _unit_effect(engine, own, private, unit, action, day)
            if service:
                services[day] += 1
                change = effect['delta']['carried']
                for product in outputs:
                    outputs[product][day] += max(0, change.get(product, 0))
                for product in inputs:
                    inputs[product][day] += max(0, -change.get(product, 0))
            _finish_own_turn(own, private, step)
            after = {position_path: list(pos()), tile_path: copy.deepcopy(current_tile()),
                     inventory_path: copy.deepcopy(private['inventories'][unit])}
            patches.append(Patch(step, before=before, after=after, units={str(unit): action},
                                 effects={} if effect is None else {str(unit): effect}, needs=needs))
            step += 1
        def route(destination):
            for axis, positive, negative in ((0, 'EAST', 'WEST'), (1, 'SOUTH', 'NORTH')):
                while pos()[axis] != destination[axis]:
                    emit([positive if pos()[axis] < destination[axis] else negative])
        def supply(*requirements):
            # Gather this service visit's prerequisites together. Routing to the
            # site before collecting feed needlessly exposes it to base sales
            # and repeats the shed trip. Actual pickup guards remain mandatory.
            missing = [(product, max(0, quantity - private['inventories'][unit].get(product, 0)))
                       for product, quantity in requirements]
            if any(quantity for _, quantity in missing):
                home = min(engine._shed_access_tiles(10), key=lambda xy:
                           (abs(pos()[0] - xy[0]) + abs(pos()[1] - xy[1]), xy))
                route(home)
                for product, quantity in missing:
                    if quantity:
                        emit(['PICKUP', product, quantity])
            route(tile)
        name = cohort.producer
        crop = engine.CROPS.get(name)
        for active_day in range(context.step // 24, 30):
            end = min((active_day + 1) * 24, LAST_STEP + 1)
            if active_day >= cohort.birth_day:
                if crop:
                    route(tile)
                    t = current_tile()
                    spent = (isinstance(t, dict) and t.get('kind') == 'PLANT'
                             and 0 <= t.get('max_lifespan_step', -1) <= active_day * 24)
                    if isinstance(t, dict) and (t.get('kind') == 'WEED' or spent):
                        emit(['DIG'], service=True)
                    if current_tile() is None:
                        emit(['PLANT', name], service=True)
                    t = current_tile()
                    if not isinstance(t, dict) or t.get('crop') != name:
                        raise cf.ProposalFailed('native PLANT did not establish the requested cohort')
                    age = active_day - t['planted_day']
                    fertilized = cohort.policy == 'watered_fertilized'
                    productive = (age >= crop['first_yield_day'] - 1 if crop['ongoing'] else
                                  (crop['max_yield_day'] + 1) // 2 <= age <= crop['max_yield_day'])
                    if fertilized and t.get('fertilized_until_day', -1) < active_day and productive:
                        supply(('FERTILIZER', 1)); emit(['FERTILIZE'], service=True)
                    emit(['WATER'], service=True)
                    if crop['ongoing']:
                        if age >= crop['first_yield_day'] and current_tile().get('yield_units', 0) > 0:
                            emit(['HARVEST'], service=True)
                    elif age >= _one_shot_harvest_age(crop, 2 if fertilized else 1):
                        emit(['HARVEST'], service=True)
                        emit(['PLANT', name], service=True)
                        emit(['WATER'], service=True)
                else:
                    requirements = ((name, 1), ('WHEAT', 1)) if current_tile() is None else (('WHEAT', 1),)
                    supply(*requirements)
                    if current_tile() is None:
                        emit(['BUILD_' + engine.ANIMALS[name]['structure']])
                        emit(['PLACE', name])
                    if not isinstance(current_tile(), dict) or current_tile().get('animal') != name:
                        raise cf.ProposalFailed('native PLACE did not establish the requested animal')
                    for op in ('FEED', 'CARE', 'COLLECT_FERTILIZER'):
                        emit([op], service=True)
                    if current_tile().get('yield_units', 0) > 0:
                        emit(['HARVEST'], service=True)
            while step < end:
                emit(['PASS'])  # Explicitly bound worker; other K0006 workers remain free.
        if (outputs != {p: list(v) for p, v in tape.outputs.items()}
                or inputs != {p: list(v) for p, v in tape.inputs.items()} or services != list(tape.actions)):
            raise cf.ProposalFailed('routed native service/output/input word differs from full requested module tape')
        self.reservation = _reserve_production(context, proposal, patches, cohort, binding)
        self.report = dict(scope='COMPLETE_CONDITIONAL_COUNT_ONE_ADD_WORD',
                           cohort=cohort.to_dict(), binding=copy.deepcopy(binding),
                           outputs=outputs, inputs=inputs, service_attempts=services,
                           own_cash_delta='UNKNOWN', rival_cash_delta='UNKNOWN', margin_delta='UNKNOWN',
                           excludes='future shops, rival private data, purchase funding, unrelated base commitments')
        if self.reservation is not None:
            self.report['reservation'] = self.reservation.to_dict()
        return tuple(patches), c['changes']


class Session:
    """One intervention; retained base instance and a permanently latched failure."""
    def __init__(self, base, context, proposal=None, *, guard: Guard | None = None,
                 lowerer: Lowerer | None = None, context_factory=None):
        self.base, self._call = base, base.act
        self.guard, self.context_factory, self.initial = guard, context_factory, context
        self.proposal = None
        self.reservation = None
        self.patches, self.pending, self.next_step = {}, None, context.step
        self._rescue_trace, self._rescue_action, self._rescue_waiting_step = None, None, None
        self.report = dict(schema=SCHEMA, status='ACTIVE', failure=None, overrides=[], verified=[], preflights=[],
                           scope='BOUND_EXECUTION_WORD_NOT_POLICY_STRENGTH', proposal=None)
        try:
            if proposal is None:
                self.report['status'] = 'COMPLETE'
                return
            self.proposal = Proposal.from_dict(copy.deepcopy(proposal.to_dict()))
            self.execution = self.proposal
            if (self.proposal.module == 'endgame'
                    and isinstance(self.proposal.candidate, dict)
                    and self.proposal.candidate.get('family') == 'complete_suffix'):
                self.execution = Proposal.from_dict(self.proposal.candidate['adopt'])
            if not isinstance(self.proposal.candidate, dict):
                raise cf.ProposalFailed('typed candidate object required')
            self.report.update(proposal=list(self.proposal.key), proposal_sha256=digest(self.proposal.to_dict()),
                               context_sha256=context_digest(context))
            patches = adapter_for(self.proposal).compile(context, self.proposal, lowerer)
            if isinstance(lowerer, NativeLowerer):
                self.report['lowering'] = copy.deepcopy(lowerer.report)
                self.reservation = copy.deepcopy(lowerer.reservation)
                if self.reservation is not None:
                    self.report['reservation'] = self.reservation.to_dict()
            if len(patches) > 719 or len({p.step for p in patches}) != len(patches):
                raise cf.ProposalFailed('execution word needs <=719 unique patch steps')
            window = self.proposal.requirements.window
            for patch in patches:
                if patch.step < context.step or (window and not window[0] <= patch.step <= window[1]):
                    raise cf.ProposalFailed('patch outside current/proposal execution window')
            self.patches = {p.step: Patch(**p.to_dict()) for p in patches}
            self.report['patches_sha256'] = digest([p.to_dict() for p in patches])
            if not patches:
                self.report['status'] = 'COMPLETE'
                return
            self._guards(context, 'ADMIT')
        except Exception as error:
            self._fail(error)

    @property
    def status(self):
        return self.report['status']

    def _fail(self, error):
        if self.status != 'PROPOSAL_FAILED':
            self.report.update(status='PROPOSAL_FAILED', failure=str(error))
        self.pending = None
        self._rescue_waiting_step = None
        if self.reservation is not None:
            self.reservation.expire(error)
            self.report['reservation'] = self.reservation.to_dict()

    def _context(self, observation):
        obs = copy.deepcopy(dict(observation))
        _wire(obs)
        if obs.get('player') != self.initial.player:
            raise cf.ProposalFailed('own seat changed')
        if self.context_factory:
            result = self.context_factory(copy.deepcopy(obs))
            if (not isinstance(result, StrategyContext) or result.step != obs['step']
                    or result.player != obs['player'] or dict(result.observation) != dict(obs)):
                raise cf.ProposalFailed('context factory changed the actual actor observation')
            return result
        return StrategyContext(obs['step'], obs['player'], obs, self.initial.memory,
                               tuple(obs.get('town', {}).get('unlocked_shops', ())), self.initial.derived)

    def _guards(self, context, phase):
        if context.shops != self.initial.shops:
            raise cf.ProposalFailed('ordered published shop prefix changed; replan required')
        rescue = _is_feed_rescue(self.execution)
        if not self.proposal.guards and not rescue:
            return
        if self.guard is None:
            raise cf.ProposalFailed('proposal guards are UNKNOWN without an explicit guard producer')
        # Defensive proposal copy: a guard cannot change the retained candidate.
        answers = self.guard(context, Proposal.from_dict(self.proposal.to_dict()), phase)
        if rescue and (not isinstance(answers, Mapping) or answers.get(RESCUE_AUTHORIZATION) is not True):
            raise cf.ProposalFailed('existing active-plan rescue authorization absent or UNKNOWN')
        if not isinstance(answers, Mapping) or any(answers.get(g) is not True for g in self.proposal.guards):
            raise cf.ProposalFailed('execution guard failed or remains UNKNOWN')
        if rescue:
            if phase == 'EXECUTE':
                _validate_feed_rescue(context, self.execution)
            self.report.setdefault('rescue_authorization', []).append(dict(
                phase=phase, context_sha256=context_digest(context),
                proposal_sha256=digest(self.execution.to_dict()),
                scope='EXTERNAL_ACTOR_GUARD_ATTESTATION_NOT_INFERRED_FROM_FEASIBILITY'))

    def act(self, observation, configuration=None):
        config = {k: copy.deepcopy(v) for k, v in (configuration or {}).items() if k in DEFAULT_CONFIGURATION}
        # The base advances even on intervention/failure turns, never a fresh K0006.
        baseline = self._call(copy.deepcopy(observation if isinstance(observation, dict)
                                           else dict(observation)), config)
        if self.status != 'ACTIVE':
            return baseline
        try:
            if observation.get('step') != self.next_step:
                raise cf.ProposalFailed('noncontiguous/repeated actor observation')
            self.next_step += 1
            if self.pending is not None or self._rescue_waiting_step is not None:
                raise cf.ProposalFailed('previous intervention has no verified realization receipt')
            context = self._context(observation)
            if (context.step == self.initial.step
                    and context_digest(context) != self.report['context_sha256']):
                raise cf.ProposalFailed('actor context changed after admission')
            if self._rescue_trace is not None:
                # Observation-only continuation: the one authorized patch is gone.
                if digest(observation) != digest(self._rescue_trace[-1]):
                    raise cf.ProposalFailed('rescue follow-up observation changed')
                self._rescue_waiting_step = context.step
                return baseline
            if self.reservation is not None:
                self.reservation.validate(context)
            self._guards(context, 'EXECUTE')
            # A market proposal certifies its whole window, including empty gaps.
            if self.execution.module == 'market_loop':
                controls = self.execution.candidate.get('control_orders', {})
                if baseline.get('market', []) != controls.get(str(context.step), []):
                    raise cf.ProposalFailed('K0006 baseline market queue changed')
            patch = self.patches.get(context.step)
            if patch is None:
                return baseline
            for key, value in patch.control.items():
                if baseline.get(key) != value:
                    raise cf.ProposalFailed('named K0006 action preimage changed: ' + key)
            for pointer, value in patch.before.items():
                if _pointer(observation, pointer) != value:
                    raise cf.ProposalFailed('native lowering precondition changed at ' + pointer)
            for pointer, value in patch.needs.items():
                actual = _pointer(observation, pointer)
                if type(actual) is not int or actual < value:
                    label = 'RESERVATION_CONFLICT: ' if self.execution.module == 'production' and not _is_feed_rescue(self.execution) else ''
                    raise cf.ProposalFailed(label + 'native lowering resource shortfall at ' + pointer)
            result = _merge_patch(baseline, patch, observation)
            if self.reservation is not None:
                self.reservation.preflight(observation, result)
            preflight = joint_preflight(observation, result, patch, config)
            if self.proposal.module == 'endgame':
                for key in ('episodeSteps', 'turnsPerDay', 'boardSize', 'maxMarketOrdersPerTurn',
                            'shedCapacity', 'farmHandCostMult'):
                    if config.get(key, DEFAULT_CONFIGURATION.get(key)) != DEFAULT_CONFIGURATION.get(key):
                        raise cf.ProposalFailed('endgame requires default pinned execution configuration')
            self.pending = (patch, copy.deepcopy(dict(observation)), copy.deepcopy(result))
            self.report['preflights'].append(preflight)
            self.report['overrides'].append(dict(step=patch.step, fields=sorted(set(patch.changes) | ({'market'} if patch.sales else set())
                                                       | {('farmer' if u == '0' else 'hands') for u in patch.units}),
                                                baseline=copy.deepcopy(baseline), emitted=copy.deepcopy(result)))
            return result
        except Exception as error:
            self._fail(error)
            return baseline

    def observe(self, after, receipt):
        """Consume an exact own native receipt, not a quote or net-stock inference."""
        if self.status != 'ACTIVE':
            return
        if self.pending is None:
            if self._rescue_trace is not None:
                try:
                    self._observe_rescue(after, receipt)
                except Exception as error:
                    self._fail(error)
            return
        patch, before, action = self.pending
        try:
            _wire(after)
            if after.get('step') != patch.step + 1 or after.get('player') != self.initial.player:
                raise cf.ProposalFailed('realization receipt has wrong post-turn/seat binding')
            if not isinstance(receipt, Mapping):
                raise cf.ProposalFailed('actual own native realization receipt required')
            if 'market' in patch.changes or patch.sales:
                got = Counter()
                rows = receipt.get('commits')
                if not isinstance(rows, list):
                    raise cf.ProposalFailed('actual own native commits required')
                for row in rows:
                    if patch.sales and (row.get('op') != 'SELL' or row.get('item') not in patch.sales):
                        continue
                    if (type(row.get('units')) is not int or row['units'] < 0
                            or row.get('failed_unit_attempts', 0)):
                        raise cf.ProposalFailed('native market order failed or partially filled')
                    got[row['op'], row.get('item')] += row['units']
                # Native orders never overfill. Equality of each op/item total,
                # zero failed attempts and exact queue equality certify every rank.
                requested = _requested([row for row in action['market']
                                        if not patch.sales or (row and row[:1] == ['SELL'] and row[1] in patch.sales)])
                if +got != +requested:
                    raise cf.ProposalFailed('realized native order quantities differ from the candidate')
            if patch.effects:
                effects = receipt.get('unit_effects')
                if not isinstance(effects, list):
                    raise cf.ProposalFailed('actual native worker effects required')
                for unit, expected in patch.effects.items():
                    got = [row for row in effects if row.get('unit') == int(unit)]
                    if got != [expected]:
                        raise cf.ProposalFailed('realized native worker effect differs for unit ' + unit)
            for pointer, value in patch.after.items():
                if _pointer(after, pointer) != value:
                    raise cf.ProposalFailed('realized worker/plan effect differs at ' + pointer)
            if self.execution.module == 'endgame':
                self._endgame_workers(patch, before, after, receipt)
            if self.reservation is not None:
                self.reservation.observe(after)
                self.report['reservation'] = self.reservation.to_dict()
            self.report['verified'].append(dict(step=patch.step, receipt=_wire(receipt),
                                                 after_sha256=digest(after)))
            self.pending = None
            del self.patches[patch.step]
            if not self.patches:
                if _is_feed_rescue(self.execution):
                    self._rescue_trace, self._rescue_action = [before], action
                    self._rescue_waiting_step = patch.step
                    self._observe_rescue(after, receipt)
                else:
                    self.report['status'] = 'COMPLETE'
        except Exception as error:
            self._fail(error)

    def _observe_rescue(self, after, receipt):
        """Consume actual transport only; predictions never enter confirmation.

        The first turn has already passed generic exact worker receipt checks.
        A possible second turn is the unchanged base, not a second FEED patch.
        """
        _wire(after)
        if (not isinstance(receipt, Mapping) or self._rescue_waiting_step is None
                or after.get('step') != self._rescue_waiting_step + 1
                or after.get('player') != self.initial.player):
            raise cf.ProposalFailed('rescue requires an acted, contiguous own-observation receipt')
        trace = self._rescue_trace + [copy.deepcopy(dict(after))]
        result = confirm_rescue(self.execution, trace, emitted_action=self._rescue_action)
        self.report['rescue'] = result
        if result['status'] not in ('PENDING_DAWN', 'SURVIVAL_OBSERVED'):
            raise cf.ProposalFailed('rescue confirmation failed: ' + result['reason'])
        self._rescue_trace, self._rescue_waiting_step = trace, None
        self.report.setdefault('rescue_observations', []).append(dict(
            step=after['step'], after_sha256=digest(after), receipt_sha256=digest(receipt)))
        if result['confirmed']:
            self.report['status'] = 'COMPLETE'

    def _endgame_workers(self, patch, before, after, receipt):
        own = self.initial.player
        movements = {'NORTH': (0, -1), 'SOUTH': (0, 1), 'WEST': (-1, 0), 'EAST': (1, 0)}
        acts = [patch.changes['farmer'], *patch.changes['hands']]
        starts = [before['farms'][own]['farmer'], *before['farms'][own]['hands']]
        ends = [after['farms'][own]['farmer'], *after['farms'][own]['hands']]
        effects = {r['unit']: r for r in receipt.get('unit_effects', [])}
        for i, action in enumerate(acts):
            op = action[0]
            if op == 'PASS':
                continue
            if op in movements:
                dx, dy = movements[op]
                if i >= len(starts) or i >= len(ends) or ends[i] != [starts[i][0] + dx, starts[i][1] + dy]:
                    raise cf.ProposalFailed('endgame move not realized')
                continue
            event = effects.get(i)
            if event is None or event.get('engine_received_action') != action:
                raise cf.ProposalFailed('endgame worker action not realized')
            delta = event.get('delta', {})
            if not any(delta.values()) and event.get('tile_before') == event.get('tile_after'):
                raise cf.ProposalFailed('endgame worker action had no requested effect')
            if op == 'DROP':
                lost = -sum(delta.get('carried', {}).values())
                stored = sum(delta.get('shed', {}).values())
                if lost <= 0 or lost != stored:
                    raise cf.ProposalFailed('endgame DROP lost or failed to deliver candidate stock')


class K0006Continuation:
    """counterfactual.Adoption adapter; never replaces frozen globals/closures."""
    def __init__(self, *, guard=None, lowerer=None, context_factory=None):
        self.guard, self.lowerer, self.context_factory = guard, lowerer, context_factory
        self.session = None
        self._monitored = False
        self._restore = None

    def adopt(self, actor, context, proposal):
        if type(actor) is not K0006Adapter:
            raise cf.ProposalFailed('these adapters require a frozen K0006 base')
        if self.session is not None:
            raise cf.ProposalFailed('one continuation per suffix; create a fresh adapter')
        if self.context_factory:
            built = self.context_factory(copy.deepcopy(dict(context.observation)))
            if (not isinstance(built, StrategyContext) or built.step != context.step
                    or built.player != context.player or dict(built.observation) != dict(context.observation)):
                raise cf.ProposalFailed('context factory must preserve actual actor observation')
            context = built
        self.session = Session(actor, context, proposal, guard=self.guard, lowerer=self.lowerer,
                               context_factory=self.context_factory)
        if self.session.status == 'PROPOSAL_FAILED':
            raise cf.ProposalFailed(self.session.report['failure'])
        if self.session.status == 'COMPLETE':
            return cf.Adoption(realization=self.session.report)
        if not self._monitored:
            self.session._fail('nonempty counterfactual adoption requires native_realization monitoring')
            raise cf.ProposalFailed(self.session.report['failure'])
        had_act, old_act = 'act' in vars(actor), vars(actor).get('act')
        def restore_method():
            if had_act:
                actor.act = old_act
            else:
                vars(actor).pop('act', None)
        self._restore = restore_method
        def strict(observation, configuration=None):
            action = self.session.act(observation, configuration)
            if self.session.status == 'PROPOSAL_FAILED':
                restore_method()
                raise cf.ProposalFailed(self.session.report['failure'])
            return action
        def first(own, observation, configuration):
            if own is not actor:
                raise cf.ProposalFailed('adoption called on a different actor')
            actor.act = strict
            return strict(observation, configuration)
        return cf.Adoption(first_action=first, realization=self.session.report)

    def close(self):
        if self._restore:
            self._restore()
            self._restore = None


@contextmanager
def native_realization(continuation):
    """Serial evaluator-only hook; original interpreter and taps delegate once.

    The shared counterfactual codec captures only base actors, not an in-flight
    Session. Capture before adoption, or after closing; mid-adoption checkpoints
    need a future explicit codec rather than silently dropping pending work.
    """
    from farm_neat.evaluation.realization import market_tap
    engine = load_engine()
    original = engine.interpreter
    if continuation._monitored or getattr(original, '_adoption_monitor', False):
        raise RuntimeError('nested/concurrent native adoption monitors are unsupported')
    def interpreter(state, env):
        session = continuation.session
        if session is None or (session.pending is None and session._rescue_waiting_step is None):
            return original(state, env)
        step = session.pending[0].step if session.pending is not None else session._rescue_waiting_step
        seat = session.initial.player
        with market_tap(env, seat) as tap:
            result = original(state, env)
        after = env.agent_observation(seat)
        after['step'] = step + 1  # env.step increments only after interpreter returns.
        receipt = dict(commits=list(tap['commits'].values()), unit_effects=tap['unit_effects'])
        session.observe(after, receipt)
        if session.status == 'PROPOSAL_FAILED':
            raise cf.ProposalFailed(session.report['failure'])
        if session.status == 'COMPLETE':
            continuation.close()
        return result
    interpreter._adoption_monitor = True
    engine.interpreter = interpreter
    continuation._monitored = True
    try:
        yield continuation
    finally:
        engine.interpreter = original
        continuation._monitored = False
        continuation.close()


def evaluate(snapshot, proposal=None, continuation=None, *, max_steps=719, seconds=120., trace=None):
    """Existing exact evaluator plus own realization monitoring; no alternate game loop."""
    continuation = continuation or K0006Continuation()
    with native_realization(continuation):
        result = cf.evaluate(snapshot, proposal, continuation, max_steps=max_steps, seconds=seconds, trace=trace)
    if continuation.session is not None:
        report = dict(result.realization or {})
        report['execution'] = copy.deepcopy(continuation.session.report)
        if result.status == 'COMPLETE' and continuation.session.status != 'COMPLETE':
            return cf.SuffixResult(result.proposal, 'PROPOSAL_FAILED', failure='execution word not fully verified',
                                   realization=report)
        return replace(result, realization=report)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', action='store_true', help='Print public integration and fail-closed boundaries; no games')
    args = parser.parse_args(argv)
    if args.interface:
        print(json.dumps(dict(schema=SCHEMA, families=['market_loop', 'endgame', 'production', 'shop_response', 'action'],
                              coordinator='Session.act / Session.observe', counterfactual='evaluate',
                              lowerer='NativeLowerer() or Feasibility(FEASIBLE, witness=bind_execution(...))',
                              plan_routes='EXPLICIT_OWN_BINDINGS; UNSUPPORTED_REMAINDERS_UNKNOWN', games=0), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
