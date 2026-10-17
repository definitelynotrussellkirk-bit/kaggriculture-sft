"""Commitment-backed cash reserves and bounded, advisory capital allocations.

Actor interface
---------------
``analyze(context)`` reads ONLY observation.farms[context.player].money and two
optional derived outputs. ``derived['cashflow']`` is a COMPLETE replacement for
``memory['cashflow']`` (never append both). Its JSON schema is::

    {schema: 'farm-strategy-cashflow-v1', step: int, player: int, shops: [str],
     groups: [str], events: [service_timeline cash event], scope_complete: bool,
     certified_prerequisites: [str], protected_resources: [str], parents?: [str]}

Events use the EXISTING cash_funding_bound schema, integer cents and
[step, own market queue, unit] moments. Include every remaining own obligation,
not sunk debits, forecasts of births, unselected proposals or rival cash. A current
snapshot must attest closure. Missing/stale scope means UNKNOWN, NOT zero reserve.
A guaranteed receipt needs every supplied prerequisite certified at this context;
otherwise its lower bound becomes zero. Certification is the supplying module's
responsibility (custody, fill, price, authorization and intervening events), not a
forecast upgraded here. ``protected_resources`` covers ALL noncash dependencies
of the funding schedule. Any proposal touching them requires a re-certified
replacement schedule, so this module abstains. Recompute after any state change.

``derived['liquidity_candidates']`` is::

    {proposals: Sequence[types.Proposal], joint_values_certified: bool = False}

These are a CURRENT menu, supplied by the coordinator. By default only one can be
selected; explicitly certified additive values permit bundles with disjoint noncash
``touches``. Cash needs are spent cash PLUS held reserve, conservatively ring-fenced
up front; no anticipated proposal receipts are credited or capital recycling inferred.
Return the reserve FIRST to quantity-generating modules: they must construct their
budget-constrained menu, not filter an unbudgeted optimum or ask us to resize orders.

Outputs/limits
--------------
The Reserve splits existing bank cash (never negative discretionary cash). ``required``
and ``shortfall`` retain unfunded commitments; clipping the split never hides a gap.
Each compact Reserve commitment carries a nominal upper debit if finite; for an
unbounded debit it carries only the known lower amount, with UNKNOWN discretionary
cash. The parallel ``funding['commitment_witnesses']`` states that distinction,
full amount bounds, native timing and conditional pessimistic prefix funding.

``funding['proposal_certificates']`` is the proposal-local cash screen.  It is
deliberately separate from Reserve: an incomplete global commitment schedule stays
UNKNOWN and is never promoted to spendable cash.  A proposal-local certificate may
still be complete in two monotone cases: (1) exact zero cash+reserve and no ``cash``
touch is FUNDED because it consumes none of the bank or commitment schedule; (2) a
positive certified lower requirement above current bank is NOT_FUNDED before any
future receipt can be credited. A third, live case is ``current_window_funding``:
the guarded terminal SELL queue is preserved and one fixed-price seed buy is
appended. Full gross debit plus held cash must fit the CURRENT bank, with zero
receipt credit, even if the proposal's forecast nets the cost to zero. This closes
only the proposal's current window, never the global Reserve. A source snapshot
revision, queue hash, actor-state/cash facts and economic proposal binding travel
with it; ``validate_current_window_funding`` must precede reuse. Other incomplete
source scope remains UNKNOWN. The new producer is opt-in via
``analyze(..., include_current_window=True)`` or the direct function: issuing it
must not implicitly adopt a new coordinator/policy gate.
With a complete source schedule, the proposal is FUNDED only when its capital upper
bound fits the guaranteed discretionary lower bound, and NOT_FUNDED only when its
capital lower bound exceeds the possible discretionary upper bound.  This follows the
module's existing conservative contract: proposal cash is ring-fenced at the current
step, not financed by anticipated proposal receipts.  Every witness names the declared
own commitments and same-step spends it is preserving.

Allocation decisions are advisory cash screens, NOT delivery/market/capacity or WDL
certificates. Own, rival and margin deltas stay on the original Proposal; holding
capital never reduces its value as though it were an expense. The finite search
maximizes certified margin floors, not win probability. Unknown-valued offers stay
visible and unallocated. Bound-valued offers may be safely funded, but their economic
shadow values are UNKNOWN. For an entirely exact eligible menu, ``shadow_value`` is
alternative margin foregone by occupying an offer's cash/resources; ``shadow_price``
is V(B+$1)-V(B) in dollars per dollar. These are conditional finite-menu diagnostics,
not smooth duals or claims of optimality outside that menu. Search is deterministic,
explicitly bounded (default 4096 subset visits), and fails closed on exhaustion.
No engine, executor, router, future shops, rival memory, or other strategy module runs.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any

from farm_neat.evaluation.service_timeline import cash_funding_bound, cents, compose_cash_segments
from farm_neat.game.constants import CROPS, MAX_MARKET_ORDERS, PRODUCTS
from farm_neat.strategy.types import EVALUATOR_ONLY, LAST_STEP, Estimate, Evidence, Proposal, Reserve, StrategyContext, UNKNOWN

CASH_PREFIX_PARENT = '910a49326d8a301988b3814f49091e687ec36e82b7492727ffaf19aaf79940de'
SCHEMA = 'farm-strategy-cashflow-v1'


@dataclass(frozen=True)
class Allocation:
    """One retained original offer. PASS is abstention, not infeasibility."""
    proposal: Proposal
    capital: Estimate = UNKNOWN
    decision: str = 'UNKNOWN'
    reason: str = ''
    shadow_value: Estimate = UNKNOWN

    def to_dict(self) -> dict:
        return dict(proposal=self.proposal.to_dict(), capital=self.capital.to_dict(),
                    decision=self.decision, reason=self.reason, shadow_value=self.shadow_value.to_dict())


@dataclass(frozen=True)
class LiquidityResult:
    reserve: Reserve
    required: Estimate
    shortfall: Estimate
    funding: Mapping[str, Any]
    allocations: tuple[Allocation, ...] = ()  # ranked, including abstentions
    selected: tuple[tuple[str, str], ...] = ()
    status: str = 'UNKNOWN'
    search_states: int = 0
    shadow_price: Estimate = UNKNOWN

    def to_dict(self) -> dict:
        r = self.reserve
        return dict(schema='farm-strategy-liquidity-v1',
                    reserve=dict(step=r.step, cash=r.cash, protected=r.protected.to_dict(),
                                 discretionary=r.discretionary.to_dict(), commitments=list(r.commitments),
                                 parents=list(r.parents)),
                    required=self.required.to_dict(), shortfall=self.shortfall.to_dict(),
                    funding=dict(self.funding), allocations=[a.to_dict() for a in self.allocations],
                    selected=[list(k) for k in self.selected], status=self.status,
                    search_states=self.search_states, shadow_price=self.shadow_price.to_dict(),
                    allocation_scope='CURRENT_CERTIFIED_MENU_ONLY; FUNDING_NOT_DELIVERY_OR_WDL')


def _require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def _strings(value: Any, label: str, *, unique: bool = True) -> tuple[str, ...]:
    _require(isinstance(value, (tuple, list)) and all(isinstance(x, str) and x.strip() for x in value),
             label + ' must be a list/tuple of nonempty strings')
    # Shops are drawn with replacement, so a repeated name is a real second instance.
    _require(not unique or len(set(value)) == len(value), 'duplicate ' + label)
    return tuple(value)


def _actor_only(value: Any) -> None:
    if isinstance(value, Mapping):
        _require(not (EVALUATOR_ONLY & set(value)), 'evaluator-only key in cashflow input')
        for v in value.values():
            _actor_only(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            _actor_only(v)


def _estimate(low: int, high: int | None) -> Estimate:
    return Estimate.exact(low / 100) if low == high else Estimate.bound(low / 100, None if high is None else high / 100)


def _cash(context: StrategyContext) -> int:
    _require(isinstance(context, StrategyContext), 'StrategyContext required')
    obs = context.observation
    for field in ('player', 'step'):
        if field in obs:
            _require(type(obs[field]) is int and obs[field] == getattr(context, field), 'observation/context ' + field + ' mismatch')
    try:
        money = obs['farms'][context.player]['money']
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError('current own farm money is required; no guessed cash') from error
    bank = cents(money)
    _require(bank >= 0, 'negative own cash')
    return bank


def _unknown(context: StrategyContext, bank: int, reason: str) -> LiquidityResult:
    return LiquidityResult(Reserve(context.step, bank / 100, UNKNOWN, UNKNOWN, parents=(CASH_PREFIX_PARENT,)),
                           UNKNOWN, UNKNOWN, {'status': 'UNKNOWN', 'reason': reason, 'commitment_witnesses': []})


def _funding(context: StrategyContext) -> tuple[LiquidityResult, frozenset[str]]:
    bank = _cash(context)
    source = 'derived' if 'cashflow' in context.derived else 'memory'
    supplied = getattr(context, source).get('cashflow')
    if supplied is None:
        return _unknown(context, bank, 'MISSING_CASHFLOW_SCOPE'), frozenset()
    _require(isinstance(supplied, Mapping), 'cashflow must be a mapping')
    _actor_only(supplied)
    try:
        data = json.loads(json.dumps(dict(supplied), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError('cashflow must be finite JSON') from error
    _require(data.get('schema') == SCHEMA, 'unsupported cashflow schema')
    _require(type(data.get('step')) is int and type(data.get('player')) is int, 'cashflow step/player required')
    shops = _strings(data.get('shops'), 'published shops', unique=False)
    if (data['step'], data['player'], shops) != (context.step, context.player, context.shops):
        return _unknown(context, bank, 'STALE_CASHFLOW_CONTEXT'), frozenset()
    # An exact emitted queue closes only a local action window. A caller cannot
    # promote this producer's intentionally incomplete horizon to a global Reserve.
    if 'current_queue' in data and data.get('scope_complete') is not False:
        result = _unknown(context, bank, 'CURRENT_EMISSION_DOES_NOT_CLOSE_GLOBAL_SCOPE')
        return replace(result, funding={**result.funding, 'scope_complete': False}), frozenset()
    resources = frozenset(_strings(data.get('protected_resources'), 'protected resources'))
    _require('cash' not in resources, 'cash is accounted numerically, not as a protected noncash resource')
    certified = set(_strings(data.get('certified_prerequisites', []), 'certified prerequisites'))
    events = data.get('events')
    _require(isinstance(events, list), 'cashflow events must be a list')
    snapshot_hash = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    downgraded = []
    for row in events:
        _require(isinstance(row, dict), 'cash event must be an object')
        _require(type(row.get('seat')) is int and row['seat'] == context.player, 'event must belong to own seat')
        early = row.get('earliest')
        _require(isinstance(early, list) and len(early) == 3 and type(early[0]) is int and early[0] >= context.step,
                 'remaining events cannot precede the current step; reconcile sunk/overdue commits')
        _require(row.get('basis') != 'REALIZED', 'remaining schedule cannot contain already REALIZED/sunk cash')
        if row.get('kind') == 'RECEIPT' and row.get('basis') == 'GUARANTEED':
            prerequisites = row.get('prerequisites')
            _require(isinstance(prerequisites, list) and all(isinstance(p, str) and p for p in prerequisites),
                     'receipt prerequisites must be named')
            if not prerequisites or not set(prerequisites) <= certified:
                # Do not hide a malformed original interval by replacing its
                # lower endpoint. Full canonical validation still runs below.
                low, high = row.get('amount_min_cents'), row.get('amount_max_cents')
                _require(type(low) is int and low >= 0, 'minimum flow must be nonnegative integer cents')
                _require(high is None or (type(high) is int and high >= low), 'invalid/reversed amount interval')
                row['basis'], row['amount_min_cents'] = 'FORECAST', 0
                downgraded.append(row.get('id'))
    report = cash_funding_bound(bank, events, groups=data.get('groups'),
                                scope_complete=data.get('scope_complete', False))
    report.update(snapshot_sha256=snapshot_hash, source=source, downgraded_receipts=sorted(downgraded),
                  protected_resources=sorted(resources))
    # Explain each debit using the same adverse ordering as the existing envelope.
    # Bounds are conditional; these are hypothetical cash paths, not realized fills.
    prefixes: dict[str, int] = {}
    if report['working_capital_upper_cents'] is not None:
        changes = []
        for e in events:
            if e['kind'] == 'SPEND':
                changes.append((tuple(e['earliest']), -e['amount_max_cents'], e['id']))
            elif e['latest'] is not None and e['latest'][0] <= 718:
                changes.append((tuple(e['latest']), e['amount_min_cents'], e['id']))
        segment = (0, 0)
        for when, delta, key in sorted(changes, key=lambda row: (row[0], row[1] >= 0, row[2])):
            segment = compose_cash_segments(segment, (delta, max(0, -delta)))
            prefixes[key] = segment[1]
    compact, witnesses = [], []
    for e in sorted((e for e in events if e['kind'] == 'SPEND'), key=lambda e: (e['latest'], e['id'])):
        low, high = e['amount_min_cents'], e['amount_max_cents']
        compact.append(dict(key=e['id'], due_step=e['latest'][0], amount=(low if high is None else high) / 100))
        witnesses.append(dict(key=e['id'], due_step=e['latest'][0], group=e['group'],
                              earliest=e['earliest'], latest=e['latest'], authorization=e['authorization'],
                              amount=_estimate(low, high).to_dict(),
                              compact_amount_basis='LOWER_ONLY' if high is None else 'EXACT' if low == high else 'UPPER_BOUND',
                              required_through_upper_cents=prefixes.get(e['id'])))
    report['commitment_witnesses'] = witnesses
    low, high = report['required_start_lower_cents'], report['required_start_upper_cents']
    required, shortfall = _estimate(low, high), _estimate(max(0, low-bank), None if high is None else max(0, high-bank))
    if not report['scope_complete']:
        required = shortfall = UNKNOWN
    if not report['scope_complete'] or high is None:
        protected = discretionary = UNKNOWN
    else:
        protected = _estimate(min(bank, low), min(bank, high))
        discretionary = _estimate(max(0, bank-high), max(0, bank-low))
    parents = tuple(dict.fromkeys((CASH_PREFIX_PARENT, *_strings(data.get('parents', []), 'parents'))))
    reserve = Reserve(context.step, bank / 100, protected, discretionary, tuple(compact), parents)
    return LiquidityResult(reserve, required, shortfall, report,
                           status='COMPLETE' if report['status'] != 'UNKNOWN' else 'UNKNOWN'), resources


def reserve(context: StrategyContext) -> Reserve:
    """Publish this BEFORE asking quantity generators for a budget-aware menu."""
    return _funding(context)[0].reserve


def _limits(value: Estimate) -> tuple[float | None, float | None]:
    if value.kind is Evidence.EXACT:
        return value.value, value.value
    if value.kind is Evidence.BOUND:
        return value.low, value.high
    return None, None


def _capital(p: Proposal) -> tuple[Estimate, int | None, int | None]:
    lo1, hi1 = _limits(p.requirements.cash)
    lo2, hi2 = _limits(p.requirements.reserve)
    for number in (lo1, hi1, lo2, hi2):
        if number is not None:
            _require(cents(number) >= 0, 'cash/reserve requirements must be nonnegative cents')
    low = None if lo1 is None or lo2 is None else cents(lo1) + cents(lo2)
    high = None if hi1 is None or hi2 is None else cents(hi1) + cents(hi2)
    if low is None:
        est = UNKNOWN if high is None else Estimate.bound(high=high / 100)
    else:
        est = _estimate(low, high)
    return est, low, high



CURRENT_WINDOW_SCHEMA = 'farm-strategy-current-window-funding-v1'
CURRENT_WINDOW_SCOPE = 'PROPOSAL_LOCAL_CURRENT_WINDOW'
# These producer limitations concern the surrounding horizon, not an extra owned
# debit. Any additional unresolved obligation is outside this proof's ownership.
_CURRENT_QUEUE_LIMITS = frozenset((
    'REACTIVE_OVERLAYS_AND_FUTURE_ROUTE_SELECTION', 'PENDING_FULFILLMENT_AND_REPLANNING',
    'FUTURE_DYNAMIC_PRICES_AND_MARKET_COMPETITION', 'CAPACITY_FEASIBILITY_NOT_CERTIFIED',
))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def current_window_funding(context: StrategyContext, proposal: Proposal, *,
                           scope: str = CURRENT_WINDOW_SCOPE) -> dict:
    """Certify an unchanged live terminal SELL queue plus one fixed-price seed buy.

    This is a funding proof, not a fill/value/adoption authorization. It consumes
    the commitment owner's freshly published cashflow (publish validates its actor
    cut), never a retained intention or forecast. The publication's snapshot hash
    is its queue revision; hashes bind supplied facts, not adversarial authenticity.

    The current producer proves exact emission only on step 718. Every baseline
    slot must be preserved. No other spend, worker change, unowned commitment or
    future window is supported. Cash already in the bank must cover the FULL seed
    cost and declared held reserve even when every SELL fails. Insufficient bank
    stays UNKNOWN: a sale might fund the buy, but that fill is not our evidence.
    Keep global scope_complete=false and revalidate before reusing this witness.
    Work is O(number of published rows + at most MAX_MARKET_ORDERS queue slots),
    plus hashing the actor-visible cut; no actor callback or engine is invoked.
    """
    row = dict(schema=CURRENT_WINDOW_SCHEMA, scope=CURRENT_WINDOW_SCOPE,
               status='UNKNOWN', reason='CURRENT_EMISSION_UNAVAILABLE', scope_complete=False,
               source_scope_complete=False, window=None, queue_revision=None, cash_facts={})
    try:
        _require(isinstance(context, StrategyContext) and isinstance(proposal, Proposal),
                 'CURRENT_WINDOW_REQUIRES_TYPED_CONTEXT_AND_PROPOSAL')
        row.update(proposal=list(proposal.key), window=(None if proposal.requirements.window is None
                                                       else list(proposal.requirements.window)))
        _require(scope == CURRENT_WINDOW_SCOPE, 'GLOBAL_SCOPE_NOT_CERTIFIED')
        _require(context.step == LAST_STEP and proposal.requirements.window == (LAST_STEP, LAST_STEP),
                 'ONLY_LIVE_TERMINAL_CURRENT_WINDOW_SUPPORTED')
        bank = _cash(context)
        source = 'derived' if 'cashflow' in context.derived else 'memory'
        flow = getattr(context, source).get('cashflow')
        _require(isinstance(flow, Mapping), 'CURRENT_EMISSION_UNAVAILABLE')
        _actor_only(flow)
        _require(flow.get('schema') == SCHEMA and flow.get('step') == context.step
                 and flow.get('player') == context.player and flow.get('shops') == list(context.shops),
                 'STALE_CURRENT_EMISSION_CONTEXT')
        _require(flow.get('scope_complete') is False, 'GLOBAL_SCOPE_NOT_CERTIFIED')
        revision = flow.get('snapshot_sha256')
        _require(isinstance(revision, str) and len(revision) == 64
                 and all(c in '0123456789abcdef' for c in revision), 'UNBOUND_QUEUE_REVISION')
        current = flow.get('current_queue')
        _require(isinstance(current, Mapping) and current.get('status') == 'EXACT'
                 and current.get('evidence') == 'EXACT_CURRENT_EMISSION'
                 and current.get('realization') == UNKNOWN.to_dict(), 'CURRENT_EMISSION_NOT_EXACT')
        orders = current.get('orders')
        _require(isinstance(orders, list) and len(orders) < MAX_MARKET_ORDERS
                 and all(isinstance(o, list) and len(o) == 3 and o[0] == 'SELL'
                         and o[1] in PRODUCTS and type(o[2]) is int and o[2] > 0 for o in orders),
                 'CURRENT_QUEUE_NOT_BOUNDED_SELL_ONLY')
        # A known-empty queue is valid; an unknown/missing queue is not empty.
        entries = flow.get('entries')
        _require(isinstance(entries, list) and len(entries) == len(orders),
                 'CURRENT_QUEUE_ENTRIES_MISMATCH')
        for rank, (entry, order) in enumerate(zip(entries, orders)):
            _require(isinstance(entry, Mapping)
                     and entry.get('id') == f'k0006-emission:{context.player}:{context.step}:{rank}'
                     and entry.get('source') == 'OWN_RETAINED_ROUTE'
                     and entry.get('evidence') == 'EXACT_CURRENT_EMISSION'
                     and entry.get('due_step') == context.step and entry.get('queue') == rank
                     and entry.get('op') == 'SELL' and entry.get('item') == order[1]
                     and entry.get('quantity') == Estimate.exact(order[2]).to_dict()
                     and entry.get('emission_due_step') == Estimate.exact(context.step).to_dict()
                     and entry.get('direction') == 'RECEIPT'
                     and entry.get('cash') == entry.get('realization') == UNKNOWN.to_dict(),
                     'CURRENT_QUEUE_ENTRIES_MISMATCH')
        _require(flow.get('groups') == ['own-retained-route'] and flow.get('events') == []
                 and flow.get('certified_prerequisites') == []
                 and flow.get('unresolved_events') == entries,
                 'UNOWNED_COMMITMENT_OR_RECEIPT_SCOPE')
        unresolved = _strings(flow.get('unresolved'), 'unresolved current queue scope')
        _require(set(unresolved) <= _CURRENT_QUEUE_LIMITS, 'UNOWNED_UNRESOLVED_COMMITMENT')
        protected = _strings(flow.get('protected_resources'), 'protected resources')
        _require('cash' not in protected, 'UNOWNED_PROTECTED_CASH')

        # Bind only actor data. Rival public tiles affect the producer's quote
        # ordering, but rival money/private inventory never enters the cash proof.
        obs = context.observation
        own = dict(step=context.step, player=context.player, farm=obs['farms'][context.player],
                   private=obs.get('private', {}), market=obs.get('market', {}),
                   town=obs.get('town', {}), day=obs.get('day'), hour=obs.get('hour'))
        _actor_only(own)
        _require(tuple(own['town'].get('unlocked_shops', [])) == context.shops,
                 'STALE_CURRENT_EMISSION_SHOPS')
        rival_tiles = [[{k: t[k] for k in ('crop', 'animal', 'yield_units') if k in t}
                        if isinstance(t, Mapping) else None for t in line]
                       for line in obs['farms'][1 - context.player]['tiles']]
        _require(current.get('public_rival_sha256') == _digest(rival_tiles),
                 'STALE_CURRENT_PUBLIC_QUEUE_ORDER')
        candidate = proposal.candidate
        _actor_only(candidate)
        _require(proposal.module == 'market_loop' and isinstance(candidate, Mapping)
                 and candidate.get('schema') == 'farm-market-loop-v1'
                 and candidate.get('scope') == 'CONDITIONAL_MARKET_WINDOW_NOT_TERMINAL_VALUE'
                 and candidate.get('execution') == 'NATIVE_MARKET_QUEUE_NOT_V0_RULES'
                 and candidate.get('status') in ('CERTIFIED', 'FORECAST_ONLY')
                 and candidate.get('family') != 'PASS' and candidate.get('refusals') in (None, []),
                 'UNOWNED_OR_UNEXECUTABLE_CURRENT_PROPOSAL')
        expected = {str(context.step): orders}
        _require(candidate.get('control_orders') == expected, 'CHANGED_CURRENT_CONTROL_QUEUE')
        proposed = candidate.get('orders')
        _require(isinstance(proposed, Mapping) and set(proposed) == {str(context.step)},
                 'PROPOSAL_HAS_UNOWNED_OR_FUTURE_ORDERS')
        child = proposed[str(context.step)]
        _require(isinstance(child, list) and len(child) == len(orders) + 1 and child[:-1] == orders,
                 'PROPOSAL_MUST_PRESERVE_QUEUE_AND_APPEND_ONE_ORDER')
        appended = child[-1]
        _require(isinstance(appended, list) and len(appended) == 3 and appended[0] == 'BUY_SEED'
                 and appended[1] in CROPS and type(appended[2]) is int and appended[2] > 0,
                 'FILL_OR_DYNAMIC_COST_NOT_CERTIFIED')
        item, quantity = appended[1:]
        _require(candidate.get('product') == item and candidate.get('quantity') == quantity,
                 'PROPOSAL_ORDER_IDENTITY_MISMATCH')
        req = proposal.requirements
        _require(req.worker_hours == req.land_quadrants == Estimate.exact(0)
                 and req.market_slots == Estimate.exact(len(child))
                 and set(proposal.touches) <= {'cash', 'market:' + item, 'market_slots', 'shed'},
                 'UNOWNED_NONMARKET_COMMITMENT')
        _require(not set(protected).intersection(proposal.touches),
                 'WOULD_INVALIDATE_COMMITMENT_FUNDING')
        _, declared_low, declared_high = _capital(proposal)
        spend_low, spend_high = _limits(req.cash)
        held_low, held_high = _limits(req.reserve)
        _require(declared_low is not None and declared_high is not None,
                 'UNKNOWN_CAPITAL_REQUIREMENT')
        unit_price = cents(CROPS[item]['seed'])
        gross = unit_price * quantity
        # The producer's peak_cash can be net of forecast SELL receipts. Never
        # inherit that zero. Honor any LARGER declared spending/held requirement.
        lower = max(gross, cents(spend_low)) + cents(held_low)
        upper = max(gross, cents(spend_high)) + cents(held_high)
        intent = proposal.to_dict()
        intent = {k: intent[k] for k in ('module', 'decision_key', 'candidate', 'touches', 'requirements')}
        row.update(source=source, queue_revision=revision, queue_sha256=_digest(orders),
                   cashflow_sha256=_digest(dict(flow)), actor_state_sha256=_digest(own),
                   public_queue_order_sha256=_digest(rival_tiles), proposal_intent_sha256=_digest(intent),
                   bank_cents=bank, capital=_estimate(lower, upper).to_dict(),
                   capital_lower_cents=lower, capital_upper_cents=upper,
                   protected_resources=list(protected), covered_commitments=[e['id'] for e in entries],
                   same_step_spends=[], unresolved_source_scope=list(unresolved),
                   cash_facts=dict(bank_cents=bank, baseline_spend_upper_cents=0,
                       gross_fixed_spend_cents=gross, fixed_unit_price_cents=unit_price, quantity=quantity,
                       declared_capital_lower_cents=declared_low, declared_capital_upper_cents=declared_high,
                       held_upper_cents=cents(held_high), required_ringfence_upper_cents=upper,
                       receipts_credited_cents=0, bank_after_ringfence_lower_cents=bank-upper),
                   excludes=['SELL_fills', 'future_receipts', 'unowned_commitments', 'global_reserve',
                             'delivery_capacity_value_or_WDL', 'rival_private_state'])
        _require(upper <= bank, 'CURRENT_BANK_INSUFFICIENT_WITHOUT_RECEIPTS')
        row.update(status='FUNDED', reason='LIVE_CURRENT_QUEUE_FIXED_SPEND_BANK_FUNDED', scope_complete=True)
        row['certificate_sha256'] = _digest(row)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as error:
        row.update(status='UNKNOWN', reason=str(error), scope_complete=False)
    return row


def validate_current_window_funding(context: StrategyContext, proposal: Proposal,
                                    certificate: Mapping) -> dict:
    """Re-derive at the current actor cut; stale/tampered receipts become UNKNOWN.

    Economic proposal bytes are bound, not downstream value/provenance additions.
    Never reuse a certificate across queue revisions, even at the same step/cash.
    This function does not authenticate a party able to forge the source publisher.
    """
    current = current_window_funding(context, proposal)
    if isinstance(certificate, Mapping) and dict(certificate) == current:
        return current
    return dict(schema=CURRENT_WINDOW_SCHEMA, scope=CURRENT_WINDOW_SCOPE,
                status='UNKNOWN', reason='STALE_OR_CHANGED_CURRENT_WINDOW_CERTIFICATE',
                scope_complete=False, source_scope_complete=False,
                window=current['window'], queue_revision=current['queue_revision'],
                current_reason=current['reason'])

def _proposal_funding(context: StrategyContext, result: LiquidityResult,
                      proposals: Sequence[Proposal], *, include_current_window: bool = False) -> tuple[list[dict], dict[str, int]]:
    """Return proposal-local funding certificates without widening global Reserve.

    These witnesses answer only whether the proposal's declared cash+reserve can be
    ring-fenced while preserving the supplied OWN commitment schedule.  They do not
    certify delivery, capacity, value, proposal execution ordering, rival behavior or
    replacement schedules for proposals that touch cash.  Missing/incomplete global
    commitments therefore remain UNKNOWN except for the monotone proofs and the guarded current-window proof documented
    above.  No evaluator-only state or future result is read here.
    """
    bank = cents(result.reserve.cash)
    source = 'derived' if 'cashflow' in context.derived else 'memory'
    supplied = getattr(context, source).get('cashflow')
    events: list[Mapping[str, Any]] = []
    unresolved: list[str] = []
    source_complete = False
    source_reason = result.funding.get('reason')
    if isinstance(supplied, Mapping):
        raw_events = supplied.get('events', [])
        if isinstance(raw_events, list):
            events = [e for e in raw_events if isinstance(e, Mapping)]
        raw_unresolved = supplied.get('unresolved', [])
        if isinstance(raw_unresolved, list):
            unresolved = sorted({str(x) for x in raw_unresolved if isinstance(x, str) and x})
        source_complete = supplied.get('scope_complete') is True and result.funding.get('scope_complete') is True

    protected_resources = frozenset(result.funding.get('protected_resources', ()))
    commitments = sorted(str(e.get('id')) for e in events
                         if e.get('kind') == 'SPEND' and isinstance(e.get('id'), str))
    same_step_spends = sorted(str(e.get('id')) for e in events
                              if e.get('kind') == 'SPEND' and isinstance(e.get('id'), str)
                              and isinstance(e.get('earliest'), (list, tuple))
                              and len(e['earliest']) == 3 and e['earliest'][0] == context.step)
    disc_low, disc_high = _limits(result.reserve.discretionary)
    guaranteed_discretionary = None if disc_low is None else cents(disc_low)
    possible_discretionary = None if disc_high is None else cents(disc_high)

    rows: list[dict] = []
    counts = {'FUNDED': 0, 'NOT_FUNDED': 0, 'UNKNOWN': 0}
    for p in sorted(proposals, key=lambda p: p.key):
        capital, low, high = _capital(p)
        window = None if p.requirements.window is None else list(p.requirements.window)
        row = {
            'proposal': list(p.key),
            'status': 'UNKNOWN',
            'reason': 'INCOMPLETE_COMMITMENT_SCOPE',
            'scope_complete': False,
            'scope': ('CURRENT_STEP_RINGFENCE_OF_PROPOSAL_DECLARED_CASH_PLUS_RESERVE; '
                      'PRESERVE_OWN_COMMITMENTS; NO_ANTICIPATED_PROPOSAL_RECEIPTS; '
                      'NO_DELIVERY_CAPACITY_VALUE_OR_WDL'),
            'source': source if isinstance(supplied, Mapping) else 'none',
            'source_scope_complete': source_complete,
            'source_reason': source_reason,
            'window': window,
            'bank_cents': bank,
            'capital': capital.to_dict(),
            'capital_lower_cents': low,
            'capital_upper_cents': high,
            'guaranteed_discretionary_cents': guaranteed_discretionary,
            'possible_discretionary_cents': possible_discretionary,
            'covered_commitments': commitments,
            'same_step_spends': same_step_spends,
            'protected_resources': sorted(protected_resources),
            'unresolved_source_scope': unresolved,
            'excludes': ['rival_private_state', 'future_randomness', 'evaluator_results',
                         'delivery_or_capacity', 'proposal_value'],
        }
        # A noncash zero-capital proposal cannot consume or displace any cash
        # commitment.  Its cash-funding proof is complete even if the commitment
        # producer cannot close the rest of the horizon.
        if protected_resources.intersection(p.touches):
            row['reason'] = 'WOULD_INVALIDATE_COMMITMENT_FUNDING; RECERTIFY_REPLACEMENT_SCHEDULE'
        elif high == 0 and low == 0 and 'cash' not in p.touches:
            row.update(status='FUNDED', reason='ZERO_CAPITAL_NONCASH_NONINTERFERENCE',
                       scope_complete=True)
        # Even before preserving a single commitment, current bank is an absolute
        # upper bound on cash available now.  This is a complete negative proof.
        elif low is not None and low > bank:
            row.update(status='NOT_FUNDED', reason='CAPITAL_LOWER_EXCEEDS_CURRENT_BANK',
                       scope_complete=True)
        elif source_complete:
            if result.funding.get('status') == 'INSUFFICIENT':
                if low is not None and low > 0:
                    row.update(status='NOT_FUNDED', reason='BASE_COMMITMENTS_ALREADY_IN_SHORTFALL',
                               scope_complete=True)
                else:
                    row['reason'] = 'BASE_COMMITMENTS_ALREADY_IN_SHORTFALL'
                    row['scope_complete'] = True
            elif high is None or low is None:
                row.update(reason='UNKNOWN_CAPITAL_REQUIREMENT', scope_complete=True)
            elif guaranteed_discretionary is not None and high <= guaranteed_discretionary:
                row.update(status='FUNDED', reason='CAPITAL_UPPER_WITHIN_GUARANTEED_DISCRETIONARY',
                           scope_complete=True)
            elif possible_discretionary is not None and low > possible_discretionary:
                row.update(status='NOT_FUNDED', reason='CAPITAL_LOWER_EXCEEDS_POSSIBLE_DISCRETIONARY',
                           scope_complete=True)
            else:
                row.update(reason='FUNDING_INTERVAL_OVERLAP', scope_complete=True)
        elif not isinstance(supplied, Mapping):
            row['reason'] = 'MISSING_CASHFLOW_SCOPE'
        elif result.funding.get('status') == 'INSUFFICIENT' and low is not None and low > 0:
            # An incomplete producer may still prove a hard lower-bound shortfall.
            row.update(status='NOT_FUNDED', reason='KNOWN_HARD_COMMITMENT_SHORTFALL',
                       scope_complete=True)
        if (include_current_window and row['status'] == 'UNKNOWN'
                and isinstance(supplied, Mapping) and 'current_queue' in supplied):
            local = current_window_funding(context, p)
            row['current_window_certificate'] = local
            row['reason'] = local['reason']
            if local['status'] == 'FUNDED':
                for field in ('status', 'reason', 'scope_complete', 'scope', 'window', 'queue_revision',
                              'capital', 'capital_lower_cents', 'capital_upper_cents', 'cash_facts',
                              'covered_commitments', 'same_step_spends'):
                    row[field] = local[field]
        counts[row['status']] += 1
        rows.append(row)
    return rows, counts


def analyze(context: StrategyContext, *, max_subsets: int = 4096,
            include_current_window: bool = False) -> LiquidityResult:
    """Return Reserve and advisory allocations; never execute or rewrite offers.

    ``include_current_window=True`` additionally screens against the live local
    proof. It is explicit opt-in so this certificate producer does not silently
    change the existing coordinator/policy's funding gates. Direct consumers may
    instead call current_window_funding and validate_current_window_funding.
    Neither path upgrades Reserve or makes analyze's allocations spend global cash.
    """
    _require(type(max_subsets) is int and 1 <= max_subsets <= 65536, 'max_subsets must be 1..65536')
    _require(type(include_current_window) is bool, 'include_current_window must be an explicit bool')
    result, protected_resources = _funding(context)
    menu = context.derived.get('liquidity_candidates', {})
    _require(isinstance(menu, Mapping), 'liquidity_candidates must be a mapping')
    proposals = menu.get('proposals', ())
    _require(isinstance(proposals, Sequence) and not isinstance(proposals, (str, bytes)) and
             len(proposals) <= 256 and all(isinstance(p, Proposal) for p in proposals), 'at most 256 typed Proposals required')
    _require(len({p.key for p in proposals}) == len(proposals), 'duplicate proposal key')
    additive = menu.get('joint_values_certified', False)
    _require(type(additive) is bool, 'joint_values_certified must be an explicit bool')
    proposals = sorted(proposals, key=lambda p: p.key)
    certificates, certificate_counts = _proposal_funding(
        context, result, proposals, include_current_window=include_current_window)
    result = replace(result, funding={**dict(result.funding),
                                     'proposal_certificates': certificates,
                                     'proposal_screen_counts': certificate_counts,
                                     'proposal_scope': ('PROPOSAL_LOCAL_ONLY; GLOBAL_RESERVE_REMAINS_UNKNOWN_WHEN_'
                                                        'COMMITMENT_SCOPE_IS_INCOMPLETE')})
    _, required_high = _limits(result.required)
    budget = None if required_high is None else max(0, cents(result.reserve.cash)-cents(required_high))
    # Do not add optional commitments while known mandatory funding is in deficit.
    if result.funding.get('status') == 'INSUFFICIENT':
        budget = None
    rows, eligible = [], []
    for p in proposals:
        capital, low, high = _capital(p)
        value_low, value_high = _limits(p.margin_delta)
        decision, reason = 'UNKNOWN', 'UNCERTIFIED_CAPITAL_OR_MARGIN'
        if value_high is not None and value_high <= 0:
            decision, reason = 'PASS', 'NO_POSITIVE_MARGIN'
        elif p.requirements.window is not None and p.requirements.window[0] < context.step:
            reason = 'STALE_PROPOSAL_WINDOW'
        elif protected_resources.intersection(p.touches):
            reason = 'WOULD_INVALIDATE_COMMITMENT_FUNDING; RECERTIFY_REPLACEMENT_SCHEDULE'
        elif budget is None:
            reason = 'COMMITMENT_FUNDING_UNCERTIFIED_OR_IN_DEFICIT'
        else:
            if low is not None and low > budget:
                decision, reason = 'PASS', 'CAPITAL_EXCEEDS_SAFE_BUDGET'
            elif high is not None and high <= budget and value_low is not None and value_low > 0:
                decision, reason = 'PASS', 'NOT_SELECTED_FROM_CERTIFIED_MENU'
            # V(B+$1) must include offers that become affordable at B+$1. They
            # are diagnostic-only until then; the winner is still constrained to B.
            if high is not None and high <= budget+100 and value_low is not None and value_low > 0:
                eligible.append((len(rows), high, value_low, frozenset(p.touches)-{'cash'}))
        rows.append(Allocation(p, capital, decision, reason))
    # Enumeration is tiny by design. Budget counts all visited subsets, not just
    # successful leaves, and has no time-dependent/random truncation.
    states: list[tuple[int, float, int, frozenset[str]]] = []
    total = (1 << len(eligible)) if additive else len(eligible)+1
    visited = min(total, max_subsets)
    for index in range(visited):
        mask = index if additive else (0 if index == 0 else 1 << (index-1))
        cost, score, resources = 0, 0.0, frozenset()
        for j, (_, capital, value, touches) in enumerate(eligible):
            if mask & (1 << j):
                if resources.intersection(touches):
                    break
                cost += capital; score += value; resources = resources | touches
        else:
            if cost <= (0 if budget is None else budget)+100:
                states.append((cost, score, mask, resources))
    if total > max_subsets:
        for i, *_ in eligible:
            rows[i] = replace(rows[i], decision='UNKNOWN', reason='SUBSET_BUDGET_EXHAUSTED')
        return replace(result, allocations=tuple(rows), status='BUDGET_LIMIT', search_states=visited)

    def best(cap: int, *, without: int = 0, blocked: frozenset[str] = frozenset(), empty_only: bool = False):
        allowed = [s for s in states if s[0] <= cap and not s[2] & without and not s[3].intersection(blocked)
                   and (not empty_only or s[2] == 0)]
        return max(allowed, key=lambda s: (s[1], -s[0], -s[2]), default=(0, 0.0, 0, frozenset()))

    b = budget or 0
    winner = best(b)
    all_exact = (not any(row.decision == 'UNKNOWN' for row in rows) and
                 all(rows[i].capital.kind is Evidence.EXACT and rows[i].proposal.margin_delta.kind is Evidence.EXACT
                     for i, *_ in eligible))
    selected = []
    for j, (i, capital, _, touches) in enumerate(eligible):
        shadow = UNKNOWN
        if all_exact and capital <= b:
            without = best(b, without=1 << j)[1]
            after = best(b-capital, without=1 << j, blocked=touches, empty_only=not additive)[1]
            shadow = Estimate.exact(without-after)
        if winner[2] & (1 << j):
            selected.append(rows[i].proposal.key)
            rows[i] = replace(rows[i], decision='ALLOCATE', reason='CERTIFIED_MENU_CASH_ALLOCATION')
        rows[i] = replace(rows[i], shadow_value=shadow)

    def rank(row: Allocation):
        floor = _limits(row.proposal.margin_delta)[0]
        if floor is None:
            return (1, 0, row.proposal.key)
        net = floor-(row.shadow_value.value if row.shadow_value.kind is Evidence.EXACT else 0)
        return (row.decision == 'UNKNOWN', -net, row.proposal.key)

    status = result.status
    if any(r.decision == 'UNKNOWN' for r in rows):
        status = 'UNKNOWN'
    shadow_price = Estimate.exact(best(b+100)[1]-winner[1]) if all_exact and budget is not None else UNKNOWN
    return replace(result, allocations=tuple(sorted(rows, key=rank)), selected=tuple(selected),
                   status=status, search_states=visited, shadow_price=shadow_price)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', action='store_true', help='Print a no-commitment actor fixture; zero games')
    args = parser.parse_args(argv)
    if args.example:
        flow = dict(schema=SCHEMA, step=0, player=0, shops=[], groups=['own'], events=[],
                    scope_complete=True, protected_resources=[], certified_prerequisites=[])
        context = StrategyContext(0, 0, {'farms': [{'money': 469}]}, derived={'cashflow': flow})
        print(json.dumps(analyze(context).to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
