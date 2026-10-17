"""Read-only, source-bound intentions and guarded current emission of frozen K0006.

Contract: ``extract(context, adapter)`` reads the OWN selected route and OWN pending
state; it neither calls ``act`` nor steps an engine. ``capture_memory`` produces a
finite-JSON snapshot for ``analyze`` through memory['k0006_commitments']. ``publish``
adds cashflow and capacity_commitments to StrategyContext without overwriting.

The frozen policy's existing pure forward getter, not a second calendar strategy,
selects each retained action. EXACT describes stored intent / conditional fixed-price
cost, NEVER realized delivery. Reactive overlays, future shop routing, delayed jobs
and unsupported terminal replanning remain UNKNOWN. A guarded final liquidation
branch additionally publishes EXACT_CURRENT_EMISSION, computed by the frozen own-stock
and public quote-order helpers. Both pre- and post-callback cuts mean BEFORE the
native transition. Orders are not fills. Consequently K0006 scope_complete remains
false: route-intention extraction does NOT certify a discretionary-cash reserve
or spare worker capacity.

Separately, issue_service_window reads a previously published own FullSeasonPlan
obligation and issues one conditional same-day local FEED window. Completeness
there applies only to its exact two-turn-or-less protected envelope, never the
surrounding season. NativeLowerer and Session retain action/realization ownership.

Cashflow uses the accepted liquidity schema. Unknown-price debits have the explicit
interval [0, unbounded]; 0 is a lower bound, not an amount estimate. Unknown-price
sales cannot enter service_timeline (which requires a finite receipt upper bound), so
remain visible in entries AND unresolved_events, never invented zero receipts. Pending
purchase confirmations are retained as state, not charged again. The final quote
ordering reads only rival PUBLIC crop/animal/yield tiles, separately hash-bound; no
rival private state, evaluator input, search, rollout or policy modification is used. Work is bounded by at
most LAST_STEP+1 getter calls and the finite retained route/pending-state sizes.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
from collections.abc import Mapping
from typing import Any
from types import FunctionType

from farm_neat.evaluation.k0006 import K0006Adapter, MEMBERS
from farm_neat.strategy.types import EVALUATOR_ONLY, Estimate, LAST_STEP, StrategyContext, UNKNOWN

SCHEMA = 'farm-k0006-commitments-v1'
PARENTS = (
    '2fc085811f0bc32c13b2252c40539bef5fcab9a1786d61cc6f229b42c1e1fa66',
    '910a49326d8a301988b3814f49091e687ec36e82b7492727ffaf19aaf79940de',
    'aff1d19527368e4fe386a1727dd12949f235ecdf12e9854dd5ace0ded81bf6c6',
    '70f76587d293e49f1e8ea65284837ded5f941c4bbaacd81afdfe92c649f8288a',
)
# Only per-seat OWN-state tables, not global reports/probes/rollout namespaces.
OWN_TABLES = ('_V219_STATES', '_V231_STATES', '_V233_STATES', '_R51_INPUT_STATES',
              '_R70_STATES', '_R85_STATES', '_R95_RESERVES', '_R97_LAST',
              '_R124_STATES', '_R127_STATES', '_R128_STATES', '_R148_PENDING')
CASH_OPS = frozenset(('HIRE', 'BUY_LAND', 'BUY_SEED', 'BUY_ANIMAL', 'BUY_PRODUCT', 'SELL'))


def _require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _plain(value: Any) -> Any:
    """Copy selected actor data, encoding own integer/coordinate keys deterministically."""
    if isinstance(value, Mapping):
        _require(not EVALUATOR_ONLY.intersection(value), 'evaluator-only key in own commitment input')
        out = {}
        for key, item in value.items():
            _require(isinstance(key, (str, int, tuple)), 'unsupported own-state key')
            name = key if isinstance(key, str) else str(key) if isinstance(key, int) else _canonical(list(key))
            _require(name not in out, 'ambiguous own-state key encoding')
            out[name] = _plain(item)
        return out
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(x) for x in value), key=_canonical)
    if isinstance(value, (tuple, list)):
        return [_plain(x) for x in value]
    _require(value is None or isinstance(value, (str, bool, int, float)), 'unsupported own-state value')
    _canonical(value)  # reject NaN/infinity
    return value


def _own_observation(context: StrategyContext) -> dict:
    _require(isinstance(context, StrategyContext), 'StrategyContext required')
    obs = context.observation
    _require(obs.get('step') == context.step and obs.get('player') == context.player,
             'observation/context step or player mismatch')
    shops = obs.get('town', {}).get('unlocked_shops', [])
    _require(tuple(shops) == context.shops, 'published shops/context mismatch')
    try:
        own = obs['farms'][context.player]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError('own farm required') from error
    # Never copy the farms list here. The final quote rule has its own minimal
    # public-tile projection and binding, separate from the own-intention snapshot.
    return _plain(dict(step=context.step, player=context.player, farm=own,
                       private=obs.get('private', {}), market=obs.get('market', {}),
                       town=obs.get('town', {}), day=obs.get('day'), hour=obs.get('hour')))


def _public_rival_tiles(context: StrategyContext) -> list:
    """Only the public fields read by frozen _r37_quote_priority; no rival memory."""
    tiles = context.observation['farms'][1 - context.player]['tiles']
    return _plain([[{k: tile[k] for k in ('crop', 'animal', 'yield_units') if k in tile}
                    if isinstance(tile, Mapping) else None for tile in row] for row in tiles])


def _unknown_queue(reason: str) -> dict:
    # None is deliberately different from a known empty emitted market queue.
    return dict(status='UNKNOWN', evidence='UNKNOWN', orders=None, reason=reason,
                public_rival_sha256=None)


def _current_queue(context: StrategyContext, own: dict, parent: dict, native: dict | None,
                   previous: int | None) -> dict:
    """Bounded exact FINAL accepted-plan branch, not a general callback emulator.

    In the pinned source terminal_action receives its own baseline[6]. If the
    accepted plan is consecutive, non-abandoned, and matches the current OWN
    physical state, it returns _parent_liquidate regardless of the stored route.
    Guard out active later worker overlays; V231 cannot add milk to an already
    exhaustive projected-stock sale. Day-end/supply/advance wrappers are inactive
    at 718. The only remaining market rewrite is the frozen public quote sorter.

    No act/terminal_action/simulate call, shadow actor, engine step, or cached
    evaluator action. POST_CALLBACK uses retained plan facts, NOT a duplicate
    callback (which would reset the frozen policy's state).
    """
    if context.step != LAST_STEP:
        return _unknown_queue('CURRENT_EMISSION_OUTSIDE_SUPPORTED_TERMINAL_BRANCH')
    plan = parent.get('_TERMINAL_PLANS', {}).get(context.player)
    if (not native or native.get('route') != 2 or native.get('pending')
            or previous not in (LAST_STEP - 1, LAST_STEP)
            or not isinstance(plan, Mapping) or not plan.get('accepted')
            or plan.get('abandoned')):
        return _unknown_queue('CURRENT_EMISSION_REQUIRES_LIVE_ACCEPTED_TERMINAL_PLAN')
    cfg = parent['_IMPL'].chassis.cfg
    expected = dict(board_size=10, turns_per_day=24, shed_capacity=100, max_orders=10)
    if any(cfg.get(k) != v for k, v in expected.items()):
        return _unknown_queue('CURRENT_EMISSION_REQUIRES_DEFAULT_GEOMETRY_AND_CAPACITY')
    # Only compare the one relevant own-state cut. Do not copy planner simulations,
    # expected future states, other seats, or evaluator provenance into a snapshot.
    physical = ({k: v for k, v in own['farm'].items() if k != 'money'}, own['private'])
    try:
        if physical != plan['expected_states'][LAST_STEP - 712]:
            return _unknown_queue('CURRENT_EMISSION_TERMINAL_PHYSICAL_STATE_DIVERGED')
    except (KeyError, IndexError, TypeError):
        return _unknown_queue('CURRENT_EMISSION_TERMINAL_PLAN_STATE_UNAVAILABLE')
    checks = {
        '_V219_STATES': ('eligible', 'committed', 'workers', 'pending'),
        '_V233_STATES': ('committed', 'workers', 'pending', 'work'),
        '_R51_INPUT_STATES': ('workers', 'pending'),
        '_R70_STATES': ('roles', 'pending'),
        '_R128_STATES': ('swaps',),
        '_V231_STATES': ('pending_buy', 'pending_places'),
    }
    for name, fields in checks.items():
        state = parent.get(name, {}).get(context.player, {})
        if any(state.get(k) for k in fields):
            return _unknown_queue('CURRENT_EMISSION_ACTIVE_OVERLAY:' + name)
    try:
        tiles = _public_rival_tiles(context)
        # Fresh minimal actor-visible data. The frozen helper gets no context,
        # derived certificates, replay receipts, seeds, or opponent private state.
        farms = [{}, {}]
        farms[context.player] = own['farm']
        farms[1 - context.player] = {'tiles': tiles}
        obs = dict(step=context.step, player=context.player, farms=farms,
                   private=own['private'], market=own['market'])
        action = parent['_parent_liquidate'](own['farm'], own['private'], own['market']['prices'])
        if parent['_R37_QUOTE']:
            original = parent['_r37_reorder_sales']
            # Its only mutation is a diagnostic counter. Reuse the exact function
            # code with that counter detached; every economic helper stays frozen.
            isolated = FunctionType(original.__code__,
                {**original.__globals__, '_R37_STATS': {'quote_reordered_turns': 0}},
                original.__name__, original.__defaults__, original.__closure__)
            action = isolated(obs, action)
        orders = _plain(action['market'])
        _require(len(orders) <= cfg['max_orders'] and all(
            len(o) == 3 and o[0] == 'SELL' and o[1] in parent['PRODUCTS']
            and type(o[2]) is int and o[2] > 0 for o in orders),
            'unsupported terminal helper output')
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as error:
        return _unknown_queue('CURRENT_EMISSION_HELPER_UNAVAILABLE:' + type(error).__name__)
    return dict(status='EXACT', evidence='EXACT_CURRENT_EMISSION', orders=orders,
                reason='GUARDED_FROZEN_TERMINAL_LIQUIDATION_AND_PUBLIC_QUOTE_ORDER',
                public_rival_sha256=_hash(tiles),
                helper_sources=['_parent_liquidate', '_r37_reorder_sales'],
                realization=UNKNOWN.to_dict())


def _check_public_binding(context: StrategyContext, digest: str | None) -> None:
    if digest is not None:
        _require(_hash(_public_rival_tiles(context)) == digest, 'stale public quote-order observation')


def _emission_rows(context: StrategyContext, current: dict) -> list[dict]:
    if current['status'] != 'EXACT':
        # The existing terminal consumer reads cashflow.entries, not current_queue.
        # Fail closed even when there is no retained current row: [] would falsely
        # certify an empty current emission. Retained rows remain in their own label.
        return [dict(due_step=context.step, queue=0, source='CURRENT_EMISSION_UNKNOWN',
                     evidence='UNKNOWN', reason=current['reason'])]
    unknown = UNKNOWN.to_dict
    return [dict(id=f'k0006-emission:{context.player}:{context.step}:{rank}',
                 due_step=context.step, queue=rank, op=op, item=item,
                 quantity=Estimate.exact(quantity).to_dict(), original_quantity=quantity,
                 suppressed_quantity=0,
                 # This is the existing value_basis wire tag for K0006's queue.
                 # Keep it for compatibility; it is NOT the epistemic label.
                 source='OWN_RETAINED_ROUTE', evidence='EXACT_CURRENT_EMISSION',
                 source_semantics='LEGACY_K0006_POLICY_TAG_NOT_INTENT_EVIDENCE',
                 emission_due_step=Estimate.exact(context.step).to_dict(),
                 execution_due_step=unknown(), realization=unknown(), cash=unknown(),
                 cash_conditions=[], direction='RECEIPT')
            for rank, (op, item, quantity) in enumerate(current['orders'])]


def capture_memory(context: StrategyContext, actor: K0006Adapter) -> dict:
    """Detached own-memory snapshot at the current pre-action or post-action boundary.

    Post-action means after the callback but BEFORE the native transition. The stored
    tape remains an intention. The separate current queue is computed from frozen
    actor-side rules, never copied from a returned/evaluator action.
    The integrity digest prevents accidental staleness/tampering; it is not a proof of
    authenticity against a party able to recompute the digest.
    """
    _require(type(actor) is K0006Adapter, 'frozen K0006Adapter required')
    own = _own_observation(context)
    source = MEMBERS['main.py']
    _require(actor.identity.get('assets', {}).get('src/farm_neat/evaluation/k0006_v43/main.py') == source,
             'K0006 source identity mismatch')
    root = actor.actor.__globals__
    parent = root['_PARENT_NS']
    chassis = parent['_IMPL'].chassis
    native = chassis.players.get(context.player)
    previous = parent.get('_TERMINAL_PREVIOUS', {}).get(context.player)
    if native is not None:
        clock = native.get('last_step')
        # An accepted terminal overlay can leave the underlying chassis clock behind.
        fresh = clock in (context.step - 1, context.step) or previous in (context.step - 1, context.step)
        _require(fresh, 'stale policy step/boundary')
    else:
        _require(context.step == 0, 'uninitialized K0006 at noninitial step boundary')
    schedule = []
    if native is not None and native.get('route') is not None:
        obs = {'step': context.step, 'player': context.player}
        for due in range(context.step, LAST_STEP + 1):
            action = parent['_r128_future'](obs, due - context.step)
            _require(isinstance(action, dict), 'K0006 forward getter returned no action')
            schedule.append({'due_step': due, 'action': _plain(action)})
    pending = []
    for name in OWN_TABLES:
        table = parent.get(name, root.get(name, {}))
        if isinstance(table, Mapping) and context.player in table:
            pending.append({'source': name, 'state': _plain(table[context.player])})
    terminal = parent.get('_TERMINAL_PLANS', {}).get(context.player)
    if terminal is not None:
        # Do not leak/evaluate the planner's private simulator graph or baseline states.
        fields = ('accepted', 'abandoned', 'deviated', 'abort_counted')
        pending.append({'source': '_TERMINAL_PLANS',
                        'state': _plain({k: terminal[k] for k in fields if k in terminal}),
                        'detail': 'TERMINAL_OVERRIDE_REQUIRES_OBSERVATION_DEPENDENT_REPLAN'})
    data = dict(schema=SCHEMA, step=context.step, player=context.player, shops=list(context.shops),
                source_sha256=source, observation_sha256=_hash(own),
                boundary='UNINITIALIZED' if native is None else
                         'POST_CALLBACK' if previous == context.step or native.get('last_step') == context.step
                         else 'PRE_CALLBACK',
                native=_plain(native), schedule=schedule, pending=pending,
                fixed_prices=_plain({'BUY_SEED': parent['SEED_PRICE'], 'BUY_ANIMAL': parent['ANIMAL_COST']}))
    data['current_queue'] = _current_queue(context, own, parent, native, previous)
    data['snapshot_sha256'] = _hash(data)
    return data


@dataclass(frozen=True)
class CommitmentResult:
    status: str
    entries: list[dict]
    cashflow: dict
    capacity: dict
    pending: list[dict]
    reconciliations: list[dict]
    unresolved: list[str]
    provenance: dict

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(schema=SCHEMA, **vars(self)))

    def publish(self, context: StrategyContext) -> StrategyContext:
        _require(not {'cashflow', 'capacity_commitments'}.intersection(context.derived),
                 'commitment producer cannot publish over already supplied data')
        _require((context.step, context.player, list(context.shops)) ==
                 (self.cashflow['step'], self.cashflow['player'], self.cashflow['shops']),
                 'stale publish step/player/shops')
        _require(_hash(_own_observation(context)) == self.provenance['observation_sha256'],
                 'stale publish observation')
        _check_public_binding(context, self.provenance.get('public_rival_sha256'))
        return context.with_derived('cashflow', copy.deepcopy(self.cashflow)).with_derived(
            'capacity_commitments', copy.deepcopy(self.capacity))


def _resource(command: list) -> str | None:
    if command[0] == 'PLANT' and len(command) > 1:
        return 'seed:' + str(command[1])
    return {'FEED': 'product:WHEAT', 'FERTILIZE': 'product:FERTILIZER'}.get(command[0])


def analyze(context: StrategyContext) -> CommitmentResult:
    """Publish labelled intentions/current emission from a capture. No policy invocation."""
    own = _own_observation(context)
    memory = context.memory.get('k0006_commitments')
    _require(isinstance(memory, Mapping), 'k0006_commitments snapshot required')
    data = _plain(memory)
    _require(data.get('schema') == SCHEMA, 'unsupported commitment snapshot schema')
    _require(data.get('source_sha256') == MEMBERS['main.py'], 'K0006 source mismatch')
    digest = data.pop('snapshot_sha256', None)
    _require(digest == _hash(data), 'commitment snapshot digest/hash mismatch')
    _require((data['step'], data['player'], data['shops']) == (context.step, context.player, list(context.shops)),
             'stale commitment step/player/shops')
    _require(data['observation_sha256'] == _hash(own), 'stale commitment observation')
    current = data.get('current_queue', _unknown_queue('CURRENT_EMISSION_NOT_CAPTURED'))
    _check_public_binding(context, current.get('public_rival_sha256'))
    unknown = UNKNOWN.to_dict
    native = data['native'] or {}
    unresolved = ['REACTIVE_OVERLAYS_AND_FUTURE_ROUTE_SELECTION', 'PENDING_FULFILLMENT_AND_REPLANNING',
                  'FUTURE_DYNAMIC_PRICES_AND_MARKET_COMPETITION', 'CAPACITY_FEASIBILITY_NOT_CERTIFIED']
    if not data['schedule']:
        unresolved.append('UNINITIALIZED_ROUTE')
    sell_state = native.get('sell_state', {})
    debts = sell_state.get('r36_debts', {})
    entries, duties, events, unpriced, reconciliations = [], [], [], [], []
    resources = set()
    for planned in data['schedule']:
        due, action = planned['due_step'], planned['action']
        _require(type(due) is int and context.step <= due <= LAST_STEP, 'invalid remaining due step')
        # Frozen native suppression and r36 debt both subtract sequentially from sells.
        suppress = dict(sell_state.get('suppress', {})) if sell_state.get('due_step') == due else {}
        debt = dict(debts.get(str(due), {}))
        for queue, order in enumerate(action.get('market', [])):
            if not order:
                continue
            op = order[0]
            if op not in CASH_OPS:
                unresolved.append('UNSUPPORTED_MARKET_INTENTION:' + str(op))
                continue
            item = None if op in ('HIRE', 'BUY_LAND') else order[1]
            quantity = 1 if item is None else int(order[2])
            _require(quantity >= 0, 'negative retained order quantity')
            original = quantity
            if op == 'SELL':
                for name, remaining in (('native_suppression', suppress), ('r36_debt', debt)):
                    reduction = min(quantity, max(0, int(remaining.get(item, 0))))
                    quantity -= reduction
                    remaining[item] = max(0, int(remaining.get(item, 0)) - reduction)
                    if reduction:
                        reconciliations.append(dict(due_step=due, queue=queue, item=item,
                                                    quantity=reduction, source=name,
                                                    interpretation='RETAINED_REQUEST_REDUCTION_NOT_PROOF_OF_PRIOR_FILL'))
            amount = unknown()
            conditions = []
            if op in data['fixed_prices']:
                price = data['fixed_prices'][op].get(item)
                if price is not None:
                    amount = Estimate.exact(price * quantity).to_dict()
                    conditions = ['RETAINED_ORDER_EXECUTES_UNCHANGED', 'ALL_REQUESTED_UNITS_FILL']
            row = dict(id=f'k0006:{context.player}:{due}:{queue}', due_step=due, queue=queue,
                       op=op, item=item, quantity=Estimate.exact(quantity).to_dict(),
                       original_quantity=original, suppressed_quantity=original - quantity,
                       source='OWN_RETAINED_ROUTE', evidence='EXACT_STORED_INTENT',
                       execution_due_step=unknown(), realization=unknown(), cash=amount,
                       cash_conditions=conditions, direction='RECEIPT' if op == 'SELL' else 'SPEND')
            entries.append(row)
            if op == 'HIRE':
                resources.add('labor')
            elif op == 'BUY_LAND':
                resources.add('land')
            else:
                resources.add(('seed:' if op == 'BUY_SEED' else 'animal:' if op == 'BUY_ANIMAL'
                               else 'product:') + str(item))
            if op == 'SELL':
                unpriced.append(copy.deepcopy(row))
                continue
            # Even fixed-price full-fill amounts are only upper bounds on this retained
            # request, not upper bounds on all later reactive expenditures.
            event = dict(id=row['id'], group='own-retained-route', object_id=row['id'],
                         generation=0, seat=context.player, obligation=op,
                         authorization='OWN_RETAINED_ROUTE_CONDITIONAL',
                         prerequisites=['RETAINED_ORDER_EXECUTES_UNCHANGED'], visibility='ACTOR_VISIBLE',
                         kind='SPEND', basis='COMMITTED', earliest=[due, queue, 0], latest=[due, queue, 0],
                         amount_min_cents=0,
                         amount_max_cents=None if amount['kind'] == 'UNKNOWN' else round(amount['value'] * 100))
            events.append(event)
        commands = [action.get('farmer', ['PASS']), *action.get('hands', [])]
        for worker, command in enumerate(commands):
            if not command or command[0] == 'PASS':
                continue
            resource = _resource(command)
            if resource:
                resources.add(resource)
            duties.append(dict(due_step=due, worker=worker, op=command[0], command=command,
                               worker_turns=Estimate.exact(1).to_dict(), resource=resource,
                               evidence='EXACT_STORED_INTENT', execution_due_step=unknown(),
                               feasibility=unknown()))
    pending_duties = []
    for worker, queue in native.get('pending', {}).items():
        for position, pair in enumerate(queue):
            site, command = pair
            pending_duties.append(dict(worker=int(worker), queue=position, site=site, command=command,
                                       op=command[0], resource=_resource(command), due_step=unknown(),
                                       feasibility=unknown(),
                                       accounting='REPLACES_A_ROUTE_TURN_DO_NOT_ADD_TO_DUTIES'))
    pending = [dict(**row, cash=unknown(), realization=unknown(),
                    interpretation='OWN_STATE_NOT_AN_ADDITIONAL_CASH_EVENT') for row in data['pending']]
    provenance = dict(source_sha256=data['source_sha256'], snapshot_sha256=digest,
                      observation_sha256=data['observation_sha256'], boundary=data['boundary'],
                      selected_route=native.get('route'), parents=list(PARENTS),
                      getter_calls=len(data['schedule']),
                      reads='OWN_NAMESPACE_AND_OWN_OBSERVATION; GUARDED_CURRENT_PUBLIC_QUOTE_TILES',
                      public_rival_sha256=current.get('public_rival_sha256'),
                      excluded=['rival_private_state', 'future_randomness', 'evaluator_provenance', 'realized_fills'])
    # Preserve the old result.entries API as intentions. Only the current terminal
    # publication consumed by value_basis becomes actual emission (or UNKNOWN).
    published = copy.deepcopy(entries)
    if context.step == LAST_STEP:
        published = _emission_rows(context, current)
        events = []  # the supported current emission is SELL-only, not old spend intent
        unpriced = copy.deepcopy(published)
        if current['status'] != 'EXACT':
            unresolved.append(current['reason'])
    cashflow = dict(schema='farm-strategy-cashflow-v1', step=context.step, player=context.player,
                    shops=list(context.shops), groups=['own-retained-route'], scope_complete=False,
                    certified_prerequisites=[], protected_resources=sorted(resources), parents=list(PARENTS),
                    events=events, entries=published, unresolved_events=unpriced,
                    retained_intentions=copy.deepcopy(entries), current_queue=copy.deepcopy(current),
                    unresolved=sorted(set(unresolved)), snapshot_sha256=digest)
    capacity = dict(schema='farm-k0006-capacity-commitments-v1', step=context.step, player=context.player,
                    shops=list(context.shops), duties=duties, pending_duties=pending_duties,
                    scope_complete=False, available_worker_turns=unknown(),
                    existing_workers=Estimate.exact(len(own['farm'].get('hands', [])) + 1).to_dict(),
                    pending_overlays=copy.deepcopy(pending), parents=list(PARENTS))
    return CommitmentResult('PARTIAL' if entries else 'UNKNOWN', entries, cashflow, capacity,
                            pending, reconciliations, sorted(set(unresolved)), provenance)


def extract(context: StrategyContext, actor: K0006Adapter) -> CommitmentResult:
    """Convenience adapter. Does not mutate context, observation, namespace or policy."""
    snapshot = capture_memory(context, actor)
    return analyze(replace(context, memory={'k0006_commitments': snapshot}))


# This is an extension of this commitment owner, NOT of the frozen K0006 route.
SERVICE_WINDOW_SCHEMA = 'farm-commitment-service-window-v1'
SERVICE_WINDOW_SCOPE = 'ONE_PREAUTHORIZED_LOCAL_FEED_SAME_DAY'


def _service_authority(context: StrategyContext, obligation_id: str, due_step: int) -> dict:
    """Read an already PUBLISHED FullSeasonPlan, never a draft or permission flag.

    A legitimate own snapshot is the trust root, just as for capture_memory. Its
    digest is integrity, not authentication against someone fabricating own memory.
    The executor continues to own publication, cancellation, IDs and revision.
    """
    from farm_neat.economy.orders import FullSeasonPlan
    snapshot = _plain(context.memory.get('active_plan'))
    _require(isinstance(snapshot, dict), 'published active_plan snapshot required')
    plan = FullSeasonPlan.restore(snapshot)
    _require(plan.committed_step is not None and plan.committed_step < context.step,
             'service must have been authorized by an earlier publication, not a draft')
    _require(plan.executor.stage_step == context.step and plan.executor.player == context.player,
             'active plan must observe the current own step/seat')
    rows = [row for stages in plan.executor.order_stages.values() for row in stages
            if row['id'] == obligation_id]
    _require(len(rows) == 1, 'one live owner obligation ID required')
    row = rows[0]; c = row['command']
    _require(c['op'] == 'SET_ANIMAL' and c['feed'] == 'DAILY' and not row['clear_before'],
             'only an existing daily FEED obligation without replacement is supported')
    _require(c['start'] <= due_step <= (LAST_STEP if c['until'] == 'GAME_END' else c['until']),
             'service outside authorized obligation window')
    _require(c['scope']['kind'] == 'TILE', 'owner must resolve the obligation to one tile')
    # Terms and credits of ALL owner obligations stay bound, not just a caller ID.
    return dict(game_id=plan.game_id, revision=plan.revision,
                publication_sha256=plan.last_publication_sha256,
                committed_step=plan.committed_step, obligation_id=obligation_id,
                command=copy.deepcopy(c),
                obligations_sha256=_hash([plan.executor.order_stages, plan.executor.rule_stages]))


def _service_claims(context: StrategyContext) -> list[str]:
    value = context.derived.get('rescue_reservations')
    _require(isinstance(value, Mapping) and set(value) == {'step', 'touches'}
             and type(value['step']) is int and value['step'] == context.step,
             'complete current rescue_reservations required')
    touches = value['touches']
    _require(isinstance(touches, (list, tuple)) and all(isinstance(t, str) and t.strip() for t in touches),
             'explicit protected-resource claims required')
    return sorted(set(touches))


def _service_origin(context: StrategyContext) -> dict:
    """Minimal OWN snapshot needed to reproduce this two-turn certificate."""
    return dict(own=_own_observation(context), active_plan=_plain(context.memory.get('active_plan')),
                capacity_rules=_plain(context.memory.get('capacity_rules', {})),
                baseline_action=_plain(context.memory.get('baseline_action')),
                claims=_service_claims(context))


def _origin_context(origin: dict) -> StrategyContext:
    own = origin['own']; seat = own['player']
    farms = [None, None]; farms[seat] = own['farm']
    obs = dict(step=own['step'], day=own['day'], hour=own['hour'], player=seat,
               farms=farms, private=own['private'], market=own['market'], town=own['town'])
    return StrategyContext(own['step'], seat, obs, memory=dict(active_plan=origin['active_plan'],
        capacity_rules=origin['capacity_rules'], baseline_action=origin['baseline_action']),
        shops=tuple(own['town'].get('unlocked_shops', [])),
        derived={'rescue_reservations': {'step': own['step'], 'touches': origin['claims']}})


def _compile_service_window(context: StrategyContext, *, obligation_id: str, worker: int,
                            due_step: int, parent_words: list) -> dict:
    from farm_neat.strategy import adoption
    from farm_neat.strategy.capacity_inputs import _snapshot, _word
    from farm_neat.strategy.rescue import LateFeedRescue
    from farm_neat.game.constants import MOVES
    from farm_neat.game.exact_engine import EXPECTED_ENGINE_SHA256
    # Hours 21 -> 22 or 22 alone: service AND its exit precede the daily reset.
    # Waiting is never lent as labor: the only marginal duty is the final FEED.
    _require(type(due_step) is int and 0 <= due_step - context.step <= 1
             and due_step % 24 == 22 and 1 <= due_step // 24 < 29,
             'supported window is at most two same-day turns ending with hour-22 FEED')
    authority = _service_authority(context, obligation_id, due_step)
    farm, private = _snapshot(context)
    _require(type(worker) is int and 0 <= worker <= len(farm['hands']) < 25,
             'one currently owned worker required; no hires or hand resets')
    target = [authority['command']['scope'][k] for k in ('x', 'y')]
    _require([farm['farmer'], *farm['hands']][worker] == target,
             'local service only: movement/pickup/new custody is unsupported')
    _require(type(private['inventories'][worker].get('WHEAT')) is int
             and private['inventories'][worker]['WHEAT'] >= 1, 'one currently carried WHEAT required')
    x, y = target; tile = farm['tiles'][y][x]
    _require(isinstance(tile, dict) and tile.get('animal') == authority['command']['animal']
             and type(tile.get('placed_day')) is int and 0 <= tile['placed_day'] < due_step // 24,
             'owned earlier animal generation must match the active obligation')
    _require(isinstance(parent_words, list) and len(parent_words) == due_step - context.step + 1,
             'one complete final parent word for EVERY window turn required')
    parents = [_word(a) for a in parent_words]
    _require(parents[0] == context.memory.get('baseline_action'), 'current final parent word changed')
    claims = _service_claims(context)
    origin = _service_origin(context)
    turns = []; obs = dict(_origin_context(origin).observation); selected = None
    for index, parent in enumerate(parents):
        step = context.step + index
        _require(len(parent['hands']) == len(farm['hands']) and parent['market'] == [],
                 'explicit all-worker parent and empty market required; no prices/fills/purchases forecast')
        commands = [parent['farmer'], *parent['hands']]
        _require(all(len(c) == 1 and isinstance(c[0], str)
                     and c[0] in {'PASS', 'WATER', 'CARE', 'FEED', *MOVES} for c in commands),
                 'baseline scope excludes new biology, pickup, purchase, hire and harvest')
        live = StrategyContext(step, context.player, obs,
            memory={'baseline_action': parent}, shops=context.shops,
            derived={'rescue_reservations': {'step': step, 'touches': claims}})
        _require([obs['farms'][context.player]['farmer'],
                  *obs['farms'][context.player]['hands']][worker] == target,
                 'worker movement invalidates local window')
        action = copy.deepcopy(parent)
        if step == due_step:
            choices = [p for p in LateFeedRescue().propose(live)
                       if p.candidate['actor'] == worker and p.candidate['target'] == target]
            _require(len(choices) == 1, 'local FEED prerequisites, parent protection or resource claims failed')
            selected = choices[0]
            if worker == 0: action['farmer'] = ['FEED']
            else: action['hands'][worker - 1] = ['FEED']
        bf, bp, be = adoption._project_units(obs, parent)
        cf, cp, ce = adoption._project_units(obs, action)
        protected = set(be) | set(ce)
        if step == due_step: protected.discard(str(worker))
        _require(all(be.get(k) == ce.get(k) for k in protected),
                 'preserved baseline worker effects changed')
        adoption._finish_own_turn(bf, bp, step)
        adoption._finish_own_turn(cf, cp, step)
        # Full endpoint comparison, not just FEED success or scalar resource cuts.
        expected_farm, expected_private = copy.deepcopy(bf), copy.deepcopy(bp)
        if step == due_step:
            expected_farm['tiles'][y][x]['fed_today'] = True
            inv = expected_private['inventories'][worker]
            inv['WHEAT'] -= 1
            if inv['WHEAT'] == 0: del inv['WHEAT']  # exact native _inv_take convention
        _require((cf, cp) == (expected_farm, expected_private),
                 'joint resource/worker endpoint differs beyond the single authorized FEED')
        after = copy.deepcopy(obs)
        after['farms'][context.player], after['private'] = cf, cp
        after.update(step=step + 1, day=(step + 1) // 24, hour=(step + 1) % 24)
        entry = _own_observation(live)
        exit_state = _own_observation(StrategyContext(step + 1, context.player, after, shops=context.shops))
        turns.append(dict(step=step, parent=parent, action=action, entry=entry, exit=exit_state,
            protected_effects={k: be[k] for k in sorted(protected) if k in be},
            unit_effects=list(ce.values()),
            ordered_resources=dict(cash_before=bf['money'], cash_after=cf['money'],
                cash_debits=[], cash_receipts=[], market_orders=[],
                custody_before=copy.deepcopy(obs['private']['inventories']), custody_after=cp['inventories'],
                seeds_before=copy.deepcopy(obs['private']['seeds']), seeds_after=cp['seeds'],
                shed_before=copy.deepcopy(obs['private']['shed']), shed_after=cp['shed'])))
        obs = after
    result = dict(schema=SERVICE_WINDOW_SCHEMA, status='READY', scope=SERVICE_WINDOW_SCOPE,
        scope_complete=True, start_step=context.step, due_step=due_step, player=context.player,
        worker=worker, worker_generation=dict(day=context.step // 24, unit=worker,
            hands=len(farm['hands']), hires_today=farm['hires_today']),
        target=target, animal_generation=selected.candidate['identity'], authority=authority,
        origin=origin, parent_words=parents, turns=turns,
        touches=list(selected.touches), engine_sha256=EXPECTED_ENGINE_SHA256,
        rule='CONDITIONAL_FULL_PARENT_PREIMAGES; REVALIDATE_EACH_ACTUAL_OWN_RECEIPT',
        excluded=['new economic decisions', 'future PASS capacity', 'new production lifecycle',
                  'unobserved receipts', 'future shops', 'value', 'teacher admission'])
    result['window_id'] = _hash(result)
    result['receipts'] = []
    return result


def issue_service_window(context: StrategyContext, *, obligation_id: str, worker: int,
                         due_step: int, parent_words: list) -> dict:
    """Issue ONE conditional, fully replayed same-day local FEED obligation.

    Authorization comes from the earlier published own FullSeasonPlan snapshot in
    memory['active_plan'], not from parent PASS, available wheat or a caller label.
    parent_words are explicit protected control preimages, NOT predictions of a
    future callback. All effects/endpoints are checked now, then exact current
    parent/own observation/receipt equality is required at EVERY actual prefix.
    Current complete coordinator claims live in derived['rescue_reservations'].
    Only one carried wheat and one dated worker turn are allocated; all other
    resources and baseline effects are protected. The surrounding season remains
    uncertified. Future-window capacity is not an executable action witness.

    Work: <=2 turns, <=25 workers, two own-only projections per turn. No policy
    callback, native interpreter/game, market, search or random generator is run.
    """
    try:
        return _compile_service_window(context, obligation_id=obligation_id, worker=worker,
                                       due_step=due_step, parent_words=parent_words)
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        return dict(schema=SERVICE_WINDOW_SCHEMA, status='UNKNOWN', scope_complete=False, reason=str(error))


def _service_receipt(turn: dict, after: dict, action: dict, receipt: Mapping) -> dict:
    _require(action == turn['action'], 'actual final emitted word differs from owned window')
    _require(after == turn['exit'], 'own receipt endpoint/step/day/price/shop support drift')
    _require(isinstance(receipt, Mapping) and receipt.get('commits') == []
             and receipt.get('unit_effects') == turn['unit_effects'],
             'complete successful own receipt for ALL worker effects required')
    return dict(step=turn['step'], action=_plain(action), receipt=_plain(receipt),
                before_sha256=_hash(turn['entry']), after_sha256=_hash(after))


def validate_service_window(context: StrategyContext, window: Mapping) -> dict:
    """Reproduce evidence, check live authorization and consume no speculative time.

    At a later prefix, a missing or failed actual OWN receipt is UNKNOWN even
    when the current snapshot happens to equal the projected endpoint. Call
    observe_service_window after the real receipt transport, not after a forecast.
    """
    try:
        _require(isinstance(window, Mapping) and window.get('status') in ('READY', 'COMPLETE'),
                 'owner-issued non-expired service window required')
        value = _plain(window)
        origin = _origin_context(value['origin'])
        rebuilt = _compile_service_window(origin, obligation_id=value['authority']['obligation_id'],
            worker=value['worker'], due_step=value['due_step'], parent_words=value['parent_words'])
        core = {k: v for k, v in value.items() if k not in ('receipts', 'status')}
        _require(core == {k: v for k, v in rebuilt.items() if k not in ('receipts', 'status')},
                 'window differs from independently rederived owner evidence')
        _require(_service_authority(context, value['authority']['obligation_id'], value['due_step'])
                 == value['authority'], 'commitment revision/publication/obligation drift')
        _require(context.player == value['player'] and _service_claims(context) == value['origin']['claims'],
                 'protected resources or actor seat changed')
        _require(_plain(context.memory.get('capacity_rules', {})) == value['origin']['capacity_rules'],
                 'execution rules changed')
        index = context.step - value['start_step']
        _require(0 <= index <= len(value['turns']), 'window step/day expired or noncontiguous')
        receipts = value['receipts']
        _require(isinstance(receipts, list) and len(receipts) == index,
                 'absent own receipt: future snapshot or PASS is not authorization to advance')
        for turn, proof in zip(value['turns'], receipts):
            _require(proof == _service_receipt(turn, turn['exit'], proof['action'], proof['receipt']),
                     'stale, conflicting or failed own receipt')
        expected = value['turns'][index]['entry'] if index < len(value['turns']) else value['turns'][-1]['exit']
        _require(_own_observation(context) == expected, 'full own entry/exit state or price/shop support drift')
        complete = index == len(value['turns'])
        _require(value['status'] == ('COMPLETE' if complete else 'READY'), 'window status/receipt disagreement')
        if not complete:
            _require(context.memory.get('baseline_action') == value['turns'][index]['parent'],
                     'actual final parent word changed')
        return dict(status='COMPLETE' if complete else 'READY', index=index, window_id=value['window_id'],
                    turn=None if complete else copy.deepcopy(value['turns'][index]))
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        return dict(status='UNKNOWN', reason=str(error))


def observe_service_window(context: StrategyContext, window: Mapping, *, emitted_action: Mapping,
                           after_observation: Mapping, receipt: Mapping) -> dict:
    """Transport one actual own receipt; failed advancement latches EXPIRED.

    An exact duplicate receipt is idempotent. Never call on projected evidence:
    the caller owns transport authenticity, as for adoption.Session.observe.
    Revalidation at the next prefix additionally requires an updated active-plan
    observation and actual final baseline word. This does not call that actor.
    """
    value = copy.deepcopy(dict(window))
    try:
        _require(value.get('status') in ('READY', 'COMPLETE'), 'expired window cannot be revived')
        index = context.step - value['start_step']
        _require(0 <= index < len(value['turns']), 'receipt outside window')
        after_ctx = StrategyContext(after_observation['step'], context.player, after_observation,
            shops=tuple(after_observation.get('town', {}).get('unlocked_shops', [])))
        proof = _service_receipt(value['turns'][index], _own_observation(after_ctx),
                                 _plain(emitted_action), receipt)
        if index < len(value['receipts']):
            _require(value['receipts'][index] == proof, 'conflicting duplicate receipt')
            return value
        check = validate_service_window(context, value)
        _require(check['status'] == 'READY', check.get('reason', 'window is not ready'))
        value['receipts'].append(proof)
        if len(value['receipts']) == len(value['turns']): value['status'] = 'COMPLETE'
        return value
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        value.update(status='EXPIRED', reason=str(error))
        return value


def service_window_guard(context: StrategyContext, proposal, phase: str) -> dict[str, bool]:
    """Concrete active-plan authorizer for the UNCHANGED FEED rescue adoption path.

    Only at the ACTUAL due prefix, after the earlier owned receipts have been
    verified, regenerate the existing rescue proposal. NativeLowerer still owns
    the bound action witness; Session still owns full-word preflight and actual
    realization/survival confirmation. No capacity verdict grants permission.
    """
    from farm_neat.strategy.adoption import RESCUE_AUTHORIZATION
    from farm_neat.strategy.rescue import LateFeedRescue
    window = context.memory.get('declared_service_window')
    checked = validate_service_window(context, window)
    if (phase not in ('ADMIT', 'EXECUTE') or checked['status'] != 'READY'
            or context.step != window['due_step']):
        return {RESCUE_AUTHORIZATION: False}
    matches = [p for p in LateFeedRescue().propose(context)
               if p.candidate['actor'] == window['worker'] and p.candidate['target'] == window['target']
               and p.to_dict() == proposal.to_dict()]
    if len(matches) != 1:
        return {RESCUE_AUTHORIZATION: False}
    return {**{g: True for g in proposal.guards}, RESCUE_AUTHORIZATION: True}
