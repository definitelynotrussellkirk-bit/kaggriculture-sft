"""Convert proposal cash to terminal deltas against the SAME frozen K0006 continuation.

Actor consumer: ``normalize_view(view, bridges=..., terminals=...)`` returns a new
StepView and audit Conversions for the existing Coordinator. Accepted producers,
Proposal, ValueBasis, engine and policies are unchanged. Run normalization BEFORE
capacity binding: changing value/guard/parent bytes invalidates old capacity proofs.
This adapter drops those proofs, not the resource requirements. It never adopts.

For each player separately, with module control R, proposed continuation P, window
endpoint h (exclusive), and frozen K0006 continuation K from the identical cut:

  C_P(T)-C_K(T) = [C_P(h)-C_R(h)]
                + [(C_P(T)-C_P(h))-(C_R(T)-C_R(h))]
                + [C_R(T)-C_K(T)],                       T = 719.

ContinuationBridge supplies the second (remainder) and third (reference) terms.
They are NOT optional zeros. Terminal PASS-relative endgame values need the third
term; their remainder must be EXACT zero. A market endpoint, even at T, does not
certify its queue is K0006, or that omitted worker/resource changes are absent.
Margin is own delta minus rival delta, conservatively without interval correlation.
Cash/reserve/shadow costs must not be subtracted again from complete cash deltas.

ValueJoin is a PRODUCER ATTESTATION, not a proof factory. Its guards must certify
full own continuation memory, matched rival law/support, actual realization and
all required economics. No live producer is invented here. Missing evidence is
UNKNOWN. Same context/K0006 but different comparison_law values are not rankable
unless this module contains an explicit, narrowly proved equivalence between the
two identifiers.  The shipped market-PASS and production-KEEP identifiers are one
such equivalence: both mean that child and control execute the literally unchanged
frozen-K0006 suffix.  The original identifiers are retained in Conversion rows;
the Coordinator is never asked to relabel them.  Module-specific resource guards
may differ. Constructors cannot verify scientific truth, public law sufficiency,
or inspect what a trusted Python producer closed over.

PublicTerminalPair is an alternative, actor-only deterministic full-suffix result;
it replaces, rather than adds to, the raw module estimate. ProductionTerminalPair
is narrower: it proves an exact terminal delta only when accepted production adoption
is an empty KEEP intervention, so child and frozen K0006 are behaviorally identical.
MarketTerminalPair is also symbolic and narrow: on the LAST action only, it proves
the delta of appending a fixed-price BUY_SEED after an unchanged SELL-only K0006
market queue published by the actor-side commitments producer.  This construction
does not need rival requests/fills because the appended order cannot change market
inventory and no later action exists.  Any earlier cut or unavailable current K0006
queue is refused instead of forecasting a continuation.
Non-identity production deltas retain source-bound, per-proposal execution and
continuation gaps in Conversion.terms; they never project terminal cash.
Captured-world native receipts are rejected by this actor path. calibration_row/table and
calibrate_snapshot are EVALUATOR ONLY: their MEASURED labels cannot be published
as a ValueBasis. Calibration misses and unvalued/failed rows are all retained.

No games on import, conversion, normalization, --registry or --example. Native
calibration is explicit and capped at four branch attempts, including the control.
This is conditional value plumbing and diagnostics, not demonstrated playing strength.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from farm_neat.game.constants import CROPS, MAX_MARKET_ORDERS
from farm_neat.strategy.coordinator import BASELINE_SHA256, TERMINAL_STEP, ValueBasis, context_key
from farm_neat.strategy import production
from farm_neat.strategy.types import (
    EVALUATOR_ONLY, Evidence, Estimate, Proposal, StepView, StrategyContext,
    SuffixResult, UNKNOWN, difference,
)

SCHEMA = 'farm-strategy-value-basis-v1'
CANONICAL_SCOPE = 'TERMINAL_OWN_RIVAL_CASH_DELTA_VS_FROZEN_K0006'
FIELDS = ('own_cash_delta', 'rival_cash_delta', 'margin_delta')
MODULE_ADAPTERS = MappingProxyType({
    'shop_response': MappingProxyType(dict(scope='SHOP_PROJECTION_NOT_NATIVE_CASH',
        guard='Needs an actor-only paired terminal producer; isolated quotes are not terminal returns.')),
    'market_loop': MappingProxyType(dict(scope='MARKET_WINDOW_VS_QUEUE',
        guard='Needs both paired post-window cash and source-control-minus-K0006 terminal cash, with resource closure.')),
    'liquidity': MappingProxyType(dict(scope='RESOURCE_ALLOCATION_NOT_VALUE',
        guard='Reserve and shadow prices are not payoffs; this module emits no economic Proposal.')),
    'production': MappingProxyType(dict(scope='PORTFOLIO_VS_BOOK_NOT_K0006',
        guard='Empty KEEP can use exact K0006 identity; non-identity production still needs a paired terminal continuation.')),
    'capacity': MappingProxyType(dict(scope='FEASIBILITY_NOT_VALUE',
        guard='A feasible witness or infeasible cut is not an own/rival cash value.')),
    'opponent': MappingProxyType(dict(scope='FLOW_NOT_VALUE',
        guard='Historical flow and heuristic forecasts are not terminal cash or exhaustive future support.')),
    'endgame': MappingProxyType(dict(scope='TERMINAL_VS_UNBOUND_CONTROL',
        guard='PASS or a public-model control must be rebased to the exact frozen K0006 continuation.')),
    'counterfactual': MappingProxyType(dict(scope='EVALUATOR_ONLY_MEASURED',
        guard='Native same-cut terminal differences are MEASURED calibration labels, never actor certificates.')),
})
_PROPOSAL_OWNERS = frozenset(('shop_response', 'market_loop', 'production', 'endgame'))
_FORBIDDEN = EVALUATOR_ONLY | {'snapshot_sha256', 'source_pins', 'environment', 'actors'}

# The production paired-terminal producer is intentionally smaller than a world
# simulator.  It may certify exact terminal equality only from a no-op adoption
# proof.  It never consumes ``context.derived`` or any evaluator outcome.
PRODUCTION_TERMINAL_BOUNDARY = MappingProxyType({
    'reads': (
        'current actor context step/player',
        'current actor-visible observation (including public rival fields)',
        'current ordered published shop prefix',
        "actor-owned memory['production'] ProductionBook",
        'the exact production Proposal bytes',
        'public production/adoption contracts for KEEP identity and ADD/RETIRE gap inspection',
    ),
    'may_not_read': (
        'rival private state',
        'future observations, shops, randomness, actions or market outcomes',
        'context.derived evaluator results or terminal rewards',
        'captured evaluator Snapshot/world/seed/source pins',
        'native SuffixResult receipts from a realized future branch',
        'PortfolioValuator immediate/window economics as terminal K0006 cash',
    ),
})
_PRODUCTION_TERMINAL_PARENTS = (
    'src/farm_neat/strategy/production.py',
    'src/farm_neat/strategy/adoption.py',
    '9ca6f02cecee48df1da42d8b4240b9cd57c0c22076cc2939986ec58975b0c48d',
)
_PRODUCTION_KEEP_LAW = 'production-empty-adoption-identical-frozen-k0006-v1'
_PRODUCTION_NONIDENTITY_REASON = (
    'PRODUCTION_NONIDENTITY_HAS_NO_ACTOR_SAFE_PAIRED_TERMINAL_CONTINUATION')
_MARKET_K0006_LAW = 'FROZEN_K0006_IDENTITY_CONTINUATION:' + BASELINE_SHA256

# Explicit semantic equivalence, local to the value consumer.  These identifiers
# were coined by different producers, but both prove the same physical comparison
# law: no intervention changes the frozen K0006 suffix.  Keeping the identifiers
# intact in Conversion rows preserves provenance while this class key permits a
# non-identity row using the market/K0006 law to rank beside a production KEEP.
COMPARISON_LAW_EQUIVALENCE = MappingProxyType({
    _MARKET_K0006_LAW: 'UNCHANGED_FROZEN_K0006_SUFFIX_V1',
    _PRODUCTION_KEEP_LAW: 'UNCHANGED_FROZEN_K0006_SUFFIX_V1',
})

MARKET_TERMINAL_BOUNDARY = MappingProxyType({
    'reads': (
        'current final-step actor context and public own/rival cash',
        'actor-side commitments cashflow publication for the current K0006 market queue',
        'the exact market_loop Proposal bytes',
        'public fixed seed prices and max-market-order engine constants',
    ),
    'may_not_read': (
        'rival private state or rival current request word',
        'future observations, shops, randomness, actions or market outcomes',
        'captured evaluator Snapshot/world/seed/source pins',
        'native terminal rewards or realized future branch cash',
        'market-loop conditional rival-flow estimates as terminal truth',
    ),
    'scope': (
        'only terminal action step 718; candidate must preserve the complete current '
        'K0006 SELL-only queue and append one BUY_SEED as the final market slot',
    ),
})
_MARKET_TERMINAL_PARENTS = (
    'src/farm_neat/strategy/market_loop.py',
    'src/farm_neat/strategy/commitments.py',
    'src/farm_neat/game/kaggriculture_engine_reference.py',
    BASELINE_SHA256,
)
_MARKET_TERMINAL_PROOF = 'FINAL_FIXED_SEED_APPEND_AFTER_UNCHANGED_K0006_SELL_QUEUE'


def _actor_json(value):
    if isinstance(value, Mapping):
        if _FORBIDDEN & set(value):
            raise ValueError('evaluator/private data in actor value evidence')
        if any(not isinstance(k, str) for k in value):
            raise ValueError('actor value evidence needs string JSON keys')
        return {k: _actor_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_actor_json(v) for v in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('finite actor JSON required')


def proposal_sha256(proposal: Proposal) -> str:
    return hashlib.sha256(json.dumps(proposal.to_dict(), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _proposal_actor_json(context: StrategyContext, proposal: Proposal) -> dict:
    """Validate proposal wire data without weakening evaluator snapshot bans.

    ``snapshot_sha256`` remains globally forbidden actor evidence.  Production is
    the sole narrow exception because its legacy field is recomputed here from the
    current actor observation, ordered shops and actor-owned ProductionBook.  A
    mismatched value is refused, and any other nested forbidden key is still caught
    by ``_actor_json``.  Thus evaluator Snapshot identities cannot be smuggled by
    merely reusing the field name.
    """
    data = proposal.to_dict()
    candidate = data.get('candidate')
    if (proposal.module == 'production' and isinstance(candidate, Mapping)
            and candidate.get('schema') == 'farm-production-proposal-v1'):
        actor_hash = candidate.get('snapshot_sha256')
        book_value = context.memory.get('production')
        if book_value is None:
            raise ValueError('production actor snapshot requires actor-owned ProductionBook memory')
        book = production.book_from_memory(book_value)
        expected = production.actor_snapshot_sha256(context, book)
        if actor_hash != expected:
            raise ValueError('production actor snapshot hash does not match current actor state')
        candidate = dict(candidate)
        candidate.pop('snapshot_sha256')
        data = {**data, 'candidate': candidate}
    return _actor_json(data)


def _negative(value: Estimate) -> Estimate:
    if value.kind is Evidence.UNKNOWN:
        return UNKNOWN
    if value.kind is Evidence.BOUND:
        return Estimate.bound(None if value.high is None else -value.high,
                              None if value.low is None else -value.low)
    return Estimate(value.kind, -value.value)


def add(left: Estimate, right: Estimate) -> Estimate:
    """Conservative sum using the shared evidence algebra; no bound/forecast mixing."""
    return difference(left, _negative(right))


@dataclass(frozen=True)
class CashChange:
    own: Estimate
    rival: Estimate

    def __post_init__(self):
        if not isinstance(self.own, Estimate) or not isinstance(self.rival, Estimate):
            raise TypeError('own and rival require explicit Estimates, including UNKNOWN')

    @property
    def margin(self):
        return difference(self.own, self.rival)

    def to_dict(self):
        return dict(zip(FIELDS, (self.own.to_dict(), self.rival.to_dict(), self.margin.to_dict())))

    def plus(self, other: CashChange) -> CashChange:
        return CashChange(add(self.own, other.own), add(self.rival, other.rival))


@dataclass(frozen=True)
class ValueJoin:
    context_key: str
    proposal_sha256: str
    comparison_law: str
    guards: tuple[str, ...]
    parents: tuple[str, ...]
    baseline_sha256: str = BASELINE_SHA256
    terminal_step: int = TERMINAL_STEP
    input_scope: str = 'ACTOR_ONLY_CERTIFIED'

    @classmethod
    def bind(cls, context, proposal, *, comparison_law, guards, parents):
        """Bind an independently justified producer assertion, NEVER infer zero terms."""
        result = cls(context_key(context), proposal_sha256(proposal), comparison_law,
                     tuple(guards), tuple(parents))
        result.validate(context, proposal)
        return result

    def validate(self, context, proposal):
        if (self.context_key != context_key(context)
                or self.proposal_sha256 != proposal_sha256(proposal)
                or self.baseline_sha256 != BASELINE_SHA256
                or type(self.terminal_step) is not int or self.terminal_step != TERMINAL_STEP
                or self.input_scope != 'ACTOR_ONLY_CERTIFIED'):
            raise ValueError('stale context/proposal, wrong K0006/horizon, or non-actor scope')
        if not isinstance(self.comparison_law, str) or not self.comparison_law.strip():
            raise ValueError('explicit common physical/continuation law required')
        if (not self.guards or not isinstance(self.guards, tuple)
                or any(not isinstance(s, str) or not s.strip() for s in self.guards)
                or len(set(self.guards)) != len(self.guards)):
            raise ValueError('nonempty distinct producer guards required')
        if not self.parents or not isinstance(self.parents, tuple):
            raise ValueError('independent producer parents required')
        # Reuse the consumer's existing parent and baseline validation.
        ValueBasis.bind(context, proposal, parents=self.parents)
        _actor_json(asdict(self))


@dataclass(frozen=True)
class ContinuationBridge:
    join: ValueJoin
    source_scope: str
    source_end_step: int  # EXCLUSIVE endpoint, unlike Requirements.window
    remainder: CashChange  # (P terminal-P endpoint) - (R terminal-R endpoint)
    reference: CashChange  # R terminal - K0006 terminal, from the same initial cut


@dataclass(frozen=True)
class PublicTerminalPair:
    join: ValueJoin
    control: SuffixResult  # None, frozen K0006 continuation
    child: SuffixResult    # exact proposal bytes identified by join


@dataclass(frozen=True)
class ProductionTerminalPair:
    """Symbolic exact pair: production adoption and K0006 are the same suffix.

    No terminal cash is guessed or embedded.  Validation re-proves that the bound
    production proposal is the accepted empty KEEP intervention at this exact cut.
    Therefore both own and rival terminal cash differences are identically zero for
    every possible future realization, including futures unavailable to the actor.
    """
    join: ValueJoin
    proof: str = 'EMPTY_PRODUCTION_ADOPTION_LEAVES_FROZEN_K0006_UNCHANGED'


@dataclass(frozen=True)
class MarketTerminalPair:
    """Symbolic exact pair for one actor-safe final-market non-identity class.

    Validation re-derives the fixed-price append from the proposal and the current
    actor-side K0006 commitments publication.  Absolute terminal cash and rival
    requests are intentionally absent: only the paired delta is claimed.
    """
    join: ValueJoin
    proof: str = _MARKET_TERMINAL_PROOF


@dataclass(frozen=True)
class TerminalRefusal:
    """Typed actor-side refusal retained by normalize_view for auditability."""
    reason: str
    producer: str = 'production_paired_terminal_v1'
    details: Mapping | None = None

    def __post_init__(self):
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError('terminal refusal reason must be nonempty')
        if self.producer != 'production_paired_terminal_v1':
            raise ValueError('unknown terminal refusal producer')
        if self.details is not None:
            if not isinstance(self.details, Mapping):
                raise ValueError('production refusal details must be an actor JSON mapping')
            object.__setattr__(self, 'details', _actor_json(self.details))


@dataclass(frozen=True)
class MarketTerminalRefusal:
    """Typed refusal for terminal market evidence; never a forecast."""
    reason: str
    producer: str = 'market_terminal_fixed_purchase_v1'

    def __post_init__(self):
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError('market terminal refusal reason must be nonempty')
        if self.producer != 'market_terminal_fixed_purchase_v1':
            raise ValueError('unknown market terminal refusal producer')


def source_scope(proposal: Proposal) -> str:
    c = proposal.candidate
    if proposal.module == 'market_loop':
        if (isinstance(c, dict) and c.get('schema') == 'farm-market-loop-v1'
                and c.get('scope') == 'CONDITIONAL_MARKET_WINDOW_NOT_TERMINAL_VALUE'):
            return 'MARKET_WINDOW_VS_QUEUE'
        return 'UNRECOGNIZED_MARKET_SCOPE'
    if proposal.module == 'endgame' and isinstance(c, dict):
        if c.get('family') == 'collection_liquidation' and c.get('control') == 'PASS_ENTIRE_SUFFIX':
            return 'TERMINAL_VS_PASS'
        if c.get('family') == 'complete_suffix':
            return 'TERMINAL_VS_PUBLIC_MODEL_CONTROL'
    return MODULE_ADAPTERS.get(proposal.module, {}).get('scope', 'UNSUPPORTED_MODULE')


def _json_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _validate_production_keep_identity(context: StrategyContext, proposal: Proposal) -> None:
    """Re-prove the exact empty-adoption contract from actor-visible bytes only."""
    if proposal.module != 'production':
        raise ValueError('production terminal identity requires a production proposal')
    if context.step % 24:
        raise ValueError('production terminal identity requires a dawn cut')
    candidate = proposal.candidate
    if not isinstance(candidate, Mapping) or candidate.get('schema') != 'farm-production-proposal-v1':
        raise ValueError('accepted production proposal schema required')
    # Recompute the legacy actor snapshot before inspecting the no-op shape.  This
    # rejects a proposal copied from another actor state while _FORBIDDEN remains
    # unchanged for every other evidence path.
    _proposal_actor_json(context, proposal)
    if candidate.get('kind') != 'KEEP' or candidate.get('changes') != {}:
        raise ValueError(_PRODUCTION_NONIDENTITY_REASON)
    if candidate.get('requirements_scope') != 'FULL_PRODUCTION_REMAINDER':
        raise ValueError('production KEEP does not bind the full remainder scope')
    shops = production.published_shops(context)
    if tuple(context.shops) != shops or candidate.get('published_shops') != list(shops):
        raise ValueError('production KEEP ordered shop prefix is not the current adoption prefix')
    book_value = context.memory.get('production')
    if book_value is None:
        raise ValueError('production KEEP requires actor-owned ProductionBook memory')
    book = production.book_from_memory(book_value)
    baseline = production.project_cohorts(context, book.cohorts)
    tape = baseline.to_dict()
    if candidate.get('tape') != tape or candidate.get('baseline_sha256') != _json_sha256(tape):
        raise ValueError('production KEEP is not the current book baseline tape')
    if candidate.get('minimum_purchases') != baseline.prefix_purchases():
        raise ValueError('production KEEP minimum purchases do not match the current baseline tape')


def _exact_quantity(value, label: str) -> int:
    """Read an exact nonnegative integral quantity from a commitments row."""
    try:
        estimate = value if isinstance(value, Estimate) else Estimate.from_dict(value)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(label + ' must be an exact integer') from error
    if (estimate.kind is not Evidence.EXACT or isinstance(estimate.value, bool)
            or not isinstance(estimate.value, (int, float))
            or not float(estimate.value).is_integer() or estimate.value < 0):
        raise ValueError(label + ' must be an exact nonnegative integer')
    return int(estimate.value)


def _current_k0006_sell_queue(context: StrategyContext) -> list[list]:
    """Reconstruct only the CURRENT retained queue from actor commitments.

    The commitments producer intentionally does not publish rival requests/fills.
    That is sufficient here because the supported child changes only a later fixed
    price private seed purchase.  Empty queue slots are not represented by the
    commitments schema, so any non-contiguous rank is refused rather than guessed.
    """
    flow = context.derived.get('cashflow')
    if not isinstance(flow, Mapping):
        raise ValueError('ACTOR_K0006_CURRENT_MARKET_QUEUE_UNAVAILABLE')
    if (flow.get('schema') != 'farm-strategy-cashflow-v1'
            or flow.get('step') != context.step
            or flow.get('player') != context.player
            or flow.get('shops') != list(context.shops)):
        raise ValueError('STALE_OR_UNBOUND_ACTOR_K0006_CASHFLOW')
    entries = flow.get('entries')
    if not isinstance(entries, list):
        raise ValueError('ACTOR_K0006_CURRENT_MARKET_QUEUE_UNAVAILABLE')
    rows = []
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get('due_step') != context.step:
            continue
        if entry.get('source') != 'OWN_RETAINED_ROUTE':
            raise ValueError('CURRENT_K0006_QUEUE_HAS_UNSUPPORTED_SOURCE')
        queue = entry.get('queue')
        if type(queue) is not int or queue < 0 or queue >= MAX_MARKET_ORDERS:
            raise ValueError('CURRENT_K0006_QUEUE_RANK_INVALID')
        op = entry.get('op')
        # A prior spend can make later seed fill depend on dynamic prices/fills.
        # Restrict the proof to receipts only; failed sells can only leave MORE
        # than the already-certified starting cash needed by the appended seed.
        if op != 'SELL':
            raise ValueError('CURRENT_K0006_QUEUE_NOT_SELL_ONLY')
        item = entry.get('item')
        quantity = _exact_quantity(entry.get('quantity'), 'current K0006 sell quantity')
        if not isinstance(item, str) or quantity <= 0:
            raise ValueError('CURRENT_K0006_SELL_ROW_INVALID')
        rows.append((queue, ['SELL', item, quantity]))
    rows.sort(key=lambda row: row[0])
    if [rank for rank, _ in rows] != list(range(len(rows))):
        raise ValueError('CURRENT_K0006_QUEUE_HAS_UNPUBLISHED_EMPTY_OR_MISSING_SLOT')
    return [row for _, row in rows]


def _validate_market_terminal_append(context: StrategyContext, proposal: Proposal) -> tuple[str, int, int]:
    """Prove a final-step fixed seed append without rival/future information."""
    if proposal.module != 'market_loop':
        raise ValueError('terminal fixed-purchase proof requires a market_loop proposal')
    if context.step != TERMINAL_STEP - 1:
        raise ValueError('MARKET_NONIDENTITY_NOT_TERMINAL_CUT_REQUIRES_FUTURE_ACTOR_INFORMATION')
    candidate = proposal.candidate
    if (not isinstance(candidate, Mapping)
            or candidate.get('schema') != 'farm-market-loop-v1'
            or candidate.get('scope') != 'CONDITIONAL_MARKET_WINDOW_NOT_TERMINAL_VALUE'
            or candidate.get('execution') != 'NATIVE_MARKET_QUEUE_NOT_V0_RULES'
            or candidate.get('status') not in ('CERTIFIED', 'FORECAST_ONLY')
            or candidate.get('family') == 'PASS'
            or candidate.get('refusals') not in (None, [])):
        raise ValueError('accepted executable non-PASS market proposal required')
    if proposal.requirements.window != (context.step, context.step):
        raise ValueError('terminal market proposal must cover only the final action')
    baseline = _current_k0006_sell_queue(context)
    expected_control = {str(context.step): baseline}
    control = candidate.get('control_orders')
    orders = candidate.get('orders')
    if control != expected_control or not isinstance(orders, Mapping):
        raise ValueError('proposal control queue is not the actor-published current K0006 queue')
    if set(orders) != {str(context.step)} or not isinstance(orders[str(context.step)], list):
        raise ValueError('candidate must publish exactly one final-step market queue')
    child = orders[str(context.step)]
    if len(baseline) >= MAX_MARKET_ORDERS or len(child) != len(baseline) + 1 or child[:-1] != baseline:
        raise ValueError('candidate must append exactly one final market slot after unchanged K0006')
    appended = child[-1]
    if (not isinstance(appended, list) or len(appended) != 3
            or appended[0] != 'BUY_SEED' or appended[1] not in CROPS
            or type(appended[2]) is not int or appended[2] <= 0):
        raise ValueError('only a positive fixed-price BUY_SEED append is supported')
    item, quantity = appended[1], appended[2]
    if candidate.get('product') != item or candidate.get('quantity') != quantity:
        raise ValueError('market proposal product/quantity do not match appended seed order')
    price = int(CROPS[item]['seed'])
    farms = context.observation.get('farms')
    if (not isinstance(farms, (list, tuple)) or len(farms) != 2
            or not isinstance(farms[context.player], Mapping)):
        raise ValueError('current public own cash unavailable')
    cash = farms[context.player].get('money')
    if (isinstance(cash, bool) or not isinstance(cash, (int, float))
            or not math.isfinite(cash) or cash < price * quantity):
        raise ValueError('appended seed purchase is not actor-certified to fill')
    # MarketLoop requirements are source-owned.  Check only the slot fact needed
    # by this proof; do not promote its conditional cash/rival estimates.
    slots = proposal.requirements.market_slots
    if slots != Estimate.exact(len(child)):
        raise ValueError('market-slot requirement does not match appended queue')
    return item, quantity, price


def market_terminal(context: StrategyContext, proposal: Proposal):
    """Actor-safe paired terminal evidence for one final-market intervention.

    The paired control is the actor-published current frozen-K0006 SELL queue.
    The child preserves that queue byte-for-byte and appends a private fixed-price
    seed purchase.  Because the append is last, cannot change market inventory,
    and there is no next action, rival private state/request/fill and all future
    observations are irrelevant to the exact terminal cash delta.
    """
    if not isinstance(context, StrategyContext) or not isinstance(proposal, Proposal):
        raise TypeError('StrategyContext and Proposal required')
    _actor_json(dict(observation=context.observation, memory=context.memory, shops=context.shops))
    try:
        _proposal_actor_json(context, proposal)
        item, quantity, price = _validate_market_terminal_append(context, proposal)
    except (TypeError, ValueError, AttributeError) as error:
        return MarketTerminalRefusal(str(error))
    join = ValueJoin.bind(
        context, proposal, comparison_law=_MARKET_K0006_LAW,
        guards=(
            'current K0006 market queue comes from actor-side commitments at this exact cut',
            'candidate preserves every K0006 market row and appends one final BUY_SEED slot',
            'all preserved K0006 rows are SELL receipts, so they cannot consume seed funding',
            f'appended {item} seed purchase is fixed-price {price} x {quantity} and actor-funded',
            'BUY_SEED changes only own cash/seeds, not market inventory or rival cash',
            'step is terminal action 718; no future observation or rival response exists',
        ),
        parents=_MARKET_TERMINAL_PARENTS,
    )
    return MarketTerminalPair(join)


def market_terminal_evidence(view: StepView) -> dict:
    """StrategyPipeline ``value_evidence`` map for proved market terminals only.

    Unsupported market rows are intentionally omitted rather than published as
    terminal evidence, so this map can be merged with an existing market-PASS
    bridge without supplying two producers for the same proposal.  Call
    :func:`market_terminal` directly when a typed refusal is required for audit.
    """
    if not isinstance(view, StepView):
        raise TypeError('StepView required')
    terminals = {}
    for proposal in view.proposals:
        if proposal.module != 'market_loop':
            continue
        terminal = market_terminal(view.context, proposal)
        if isinstance(terminal, MarketTerminalPair):
            terminals[proposal.key] = terminal
    return {'terminals': terminals}


def production_continuation_gap(context: StrategyContext, proposal: Proposal) -> dict:
    """Diagnose a canonical ADD/RETIRE without valuing or adopting it.

    Inputs are the current actor observation, ordered shops, own book/ownership
    memory and original proposal. Derived services, native snapshots and future
    receipts are never read. The existing replanner owns the conditional tape;
    the existing NativeLowerer owns executable feasibility. This helper owns only
    the missing continuation-input report consumed by ``production_terminal`` and
    ``Conversion.terms``. A feasible lowerer is NOT a paired terminal producer.

    Budget: one bounded (<=256-change) replan and one checked lowering (<=719 own
    unit subphases), no game/search/adoption. Signed tape deltas are conditional
    resource demands, NOT paid receipts, realizable output or a RETIRE refund.
    """
    from farm_neat.strategy.adoption import NativeLowerer

    if not isinstance(context, StrategyContext) or not isinstance(proposal, Proposal):
        raise TypeError('StrategyContext and Proposal required')
    _actor_json(dict(observation=context.observation, memory=context.memory, shops=context.shops))
    _proposal_actor_json(context, proposal)
    candidate = proposal.candidate
    shape_error = 'production continuation diagnostic requires the current canonical proposal shape'
    if (proposal.module != 'production' or context.step % 24
            or not isinstance(candidate, Mapping) or candidate.get('kind') not in ('ADD', 'RETIRE')):
        raise ValueError(shape_error)
    book = production.book_from_memory(context.memory.get('production'))
    # Clear ALL derived services: valuation callbacks are neither necessary nor
    # allowed inputs to this structural comparison. Keep the original actor key.
    typed = replace(context, memory={**context.memory, 'production': book}, derived={})
    canonical = next((p for p in production.ProductionReplanner(max_changes=256).propose(typed)
                      if p.candidate['kind'] == candidate['kind']
                      and p.candidate.get('request_key') == candidate.get('request_key')), None)
    fields = ('schema', 'kind', 'request_key', 'changes', 'baseline_sha256',
              'snapshot_sha256', 'published_shops', 'requirements_scope', 'tape', 'minimum_purchases')
    if (canonical is None or any(candidate.get(k) != canonical.candidate.get(k) for k in fields)
            or proposal.requirements != canonical.requirements):
        raise ValueError(shape_error)
    baseline = production.project_cohorts(typed, book.cohorts).to_dict()
    target = canonical.candidate['tape']

    def signed(before, after):
        return [[day, b - a] for day, (a, b) in enumerate(zip(before, after)) if a != b]

    delta = {k: signed(baseline[k], target[k]) for k in ('fixed_cash', 'actions', 'sites')}
    for name in ('outputs', 'inputs'):
        delta[name] = {product: values for product in target[name]
                       if (values := signed(baseline[name][product], target[name][product]))}
    delta['scope'] = 'CONDITIONAL_FULL_SERVICE_TAPE_DIFFERENCE_NOT_TERMINAL_CASH_OR_REFUND'
    lowerer = NativeLowerer()
    lowerer(typed, proposal)
    execution = {'verdict': lowerer.report['verdict']}
    if execution['verdict'] == 'UNKNOWN':
        execution['reason'] = lowerer.report['reason']
    else:
        execution['patches'] = lowerer.report['patches']
        execution['scope'] = 'CONDITIONAL_OWN_UNIT_TAPE_ONLY_NOT_PAIRED_TERMINAL'

    missing = []
    costs = [[day, value] for day, value in delta['fixed_cash'] if value > 0]
    if costs:
        missing.append(dict(code='FUNDED_ACQUISITION_BINDING',
            category='EXECUTION_BINDING', conditional_cash_by_day=costs,
            need='Dated producer purchases, paid funding and custody bound to the live queue; '
                 'current money or a module seed-cost debit alone does not execute an acquisition.'))
    if candidate['kind'] == 'ADD':
        ownership = context.memory.get('production_bindings', {})
        unbound = []
        for cohort in candidate['changes']['add']:
            binding = ownership.get(cohort['key']) if isinstance(ownership, Mapping) else None
            if (not isinstance(binding, dict) or set(binding) != {'unit', 'tile'}
                    or type(binding['unit']) is not int or binding['unit'] < 0
                    or not isinstance(binding['tile'], list) or len(binding['tile']) != 2
                    or any(type(v) is not int or not 0 <= v < 10 for v in binding['tile'])):
                unbound.append(cohort['key'])
        if unbound:
            missing.append(dict(code='COHORT_UNIT_TILE_OWNERSHIP', category='ACTOR_MEMORY',
                cohorts=unbound,
                need="Actor-owned memory.production_bindings[cohort] = {unit, tile}, including "
                     'actual worker/site availability; no K0006 allocation may be guessed.'))
    else:
        missing.append(dict(code='RETIRE_EXECUTION_OWNERSHIP', category='EXECUTION_BINDING',
            retirement=candidate['changes']['retire'],
            need='An owned cancellation/replacement-stop intervention and subsequent resource '
                 'release policy. The checked lowerer supports ADD only; removed future '
                 'tape costs are not cash refunds or cancellation of unrelated K0006 work.'))
    positive_inputs = {product: [[day, q] for day, q in values if q > 0]
                       for product, values in delta['inputs'].items() if any(q > 0 for _, q in values)}
    if positive_inputs:
        missing.append(dict(code='INPUT_CUSTODY_AND_FILLS', category='EXECUTION_CONTINUATION',
            conditional_units_by_day=positive_inputs,
            need='Real deliveries, reservations against base sales and funded input fills at '
                 'each service; a pooled minimum-purchases total does not establish custody.'))
    missing.extend((
        dict(code='PAIRED_REALIZATION_AND_SALE_POLICY', category='CONTINUATION_CONTRACT',
             need='The complete adopted own policy versus unchanged frozen K0006 from this '
                  'same cut, including guard failures, lost base work, routing, wages, custody, '
                  'sales and continuation memory. Conditional output is not terminal cash.'),
        dict(code='RESPONSIVE_K0006_CONTINUATION_SUPPORT', category='FUTURE_SUPPORT_OR_LAW',
             need='Evaluate the rival K0006 on each branch observation and bind a sound common '
                  'support/law for future rival fills. Neither a fixed rival action tape nor '
                  'unobserved private state from an evaluator may supply actor terminal evidence.'),
    ))
    if context.step // 24 < 29:
        missing.append(dict(code='PUBLIC_DISTURBANCE_SUPPORT', category='FUTURE_SUPPORT_OR_LAW',
            published_shops=list(context.shops),
            need='Account for possible future weeds and ordered shop publications with correct '
                 'branch coupling. The current prefix does not specify the future; identical '
                 'world seeds do not certify identical shops when occupied tiles change RNG use.'))
    return _actor_json(dict(schema='farm-production-continuation-gap-v1',
        input_scope='ACTOR_ONLY_DIAGNOSTIC_NOT_VALUE', context_key=context_key(context),
        proposal_sha256=proposal_sha256(proposal), key=list(proposal.key),
        step=context.step, seat=context.player, kind=candidate['kind'],
        request_key=candidate['request_key'], comparison_baseline_sha256=BASELINE_SHA256,
        terminal_step=TERMINAL_STEP, execution_check=execution,
        conditional_delta=delta, missing_facts=missing,
        limit='Missing current execution bindings are witnessed; future-support requirements '
              'are not a proof that every possible actor-safe terminal model is impossible.'))


def production_terminal(context: StrategyContext, proposal: Proposal):
    """Produce an actor-safe terminal pair or an explicit refusal for production.

    This is deliberately not a forecast.  The only currently provable terminal
    comparison is the accepted KEEP no-op: adoption emits no intervention, so the
    child suffix is literally the same frozen K0006 continuation from the same cut.
    ADD/RETIRE may have useful conditional portfolio economics, but without future
    public/rival outcomes and a complete paired continuation those are not terminal
    values and remain UNKNOWN.
    """
    if not isinstance(context, StrategyContext) or not isinstance(proposal, Proposal):
        raise TypeError('StrategyContext and Proposal required')
    # Validate actor inputs even on refusals.  Never inspect context.derived here.
    _actor_json(dict(observation=context.observation, memory=context.memory, shops=context.shops))
    try:
        _validate_production_keep_identity(context, proposal)
    except (TypeError, ValueError, AttributeError) as error:
        if str(error) == _PRODUCTION_NONIDENTITY_REASON:
            try:
                return TerminalRefusal(str(error), details=production_continuation_gap(context, proposal))
            except (TypeError, ValueError, AttributeError, KeyError) as gap_error:
                return TerminalRefusal(str(gap_error))
        return TerminalRefusal(str(error))
    join = ValueJoin.bind(
        context, proposal, comparison_law=_PRODUCTION_KEEP_LAW,
        guards=(
            'proposal is the current actor-bound production KEEP with empty changes',
            'KEEP tape and minimum purchases equal the current ProductionBook baseline',
            'accepted production adoption emits no intervention patches',
            'child and control therefore execute the identical frozen K0006 suffix',
            'no immediate production valuation is treated as terminal cash',
        ),
        parents=_PRODUCTION_TERMINAL_PARENTS,
    )
    return ProductionTerminalPair(join)


def production_terminal_evidence(view: StepView) -> dict:
    """StrategyPipeline value_evidence callback for the production family only."""
    if not isinstance(view, StepView):
        raise TypeError('StepView required')
    terminals = {p.key: production_terminal(view.context, p)
                 for p in view.proposals if p.module == 'production'}
    return {'terminals': terminals}


def _production_pair_value(context, proposal, pair):
    if not isinstance(pair, ProductionTerminalPair) or not isinstance(pair.join, ValueJoin):
        raise ValueError('typed ProductionTerminalPair required')
    pair.join.validate(context, proposal)
    if pair.proof != 'EMPTY_PRODUCTION_ADOPTION_LEAVES_FROZEN_K0006_UNCHANGED':
        raise ValueError('unknown production terminal identity proof')
    _validate_production_keep_identity(context, proposal)
    if pair.join.comparison_law != _PRODUCTION_KEEP_LAW:
        raise ValueError('production terminal identity comparison law mismatch')
    zero = Estimate.exact(0)
    return CashChange(zero, zero), {
        'proof': pair.proof,
        'control': 'FROZEN_K0006_FROM_SAME_STATE',
        'child': 'IDENTICAL_FROZEN_K0006_SUFFIX_AFTER_EMPTY_PRODUCTION_ADOPTION',
        'absolute_terminal_cash': 'NOT_READ_OR_FABRICATED',
    }


def _market_pair_value(context, proposal, pair):
    if not isinstance(pair, MarketTerminalPair) or not isinstance(pair.join, ValueJoin):
        raise ValueError('typed MarketTerminalPair required')
    pair.join.validate(context, proposal)
    if pair.proof != _MARKET_TERMINAL_PROOF:
        raise ValueError('unknown market terminal proof')
    item, quantity, price = _validate_market_terminal_append(context, proposal)
    if pair.join.comparison_law != _MARKET_K0006_LAW:
        raise ValueError('market terminal K0006 comparison law mismatch')
    own = Estimate.exact(-price * quantity)
    rival = Estimate.exact(0)
    return CashChange(own, rival), {
        'proof': pair.proof,
        'control': 'ACTOR_PUBLISHED_CURRENT_FROZEN_K0006_QUEUE',
        'child': 'SAME_QUEUE_PLUS_FINAL_FIXED_PRICE_BUY_SEED',
        'appended_order': ['BUY_SEED', item, quantity],
        'unit_price': price,
        'own_cash_delta': -price * quantity,
        'rival_cash_delta': 0,
        'absolute_terminal_cash': 'NOT_READ_OR_FABRICATED',
        'information_boundary': dict(MARKET_TERMINAL_BOUNDARY),
    }


def _bridge_value(context, proposal, bridge):
    if not isinstance(bridge, ContinuationBridge) or not isinstance(bridge.join, ValueJoin):
        raise ValueError('typed ContinuationBridge required')
    bridge.join.validate(context, proposal)
    scope = source_scope(proposal)
    if scope not in ('MARKET_WINDOW_VS_QUEUE', 'TERMINAL_VS_PASS', 'TERMINAL_VS_PUBLIC_MODEL_CONTROL'):
        raise ValueError('module source has no usable cash-difference scope; needs a paired terminal producer')
    if bridge.source_scope != scope or type(bridge.source_end_step) is not int:
        raise ValueError('bridge source scope/endpoint mismatch')
    if not isinstance(bridge.remainder, CashChange) or not isinstance(bridge.reference, CashChange):
        raise ValueError('explicit own AND rival remainder/reference terms required')
    if scope == 'MARKET_WINDOW_VS_QUEUE':
        if proposal.requirements.window is None:
            raise ValueError('market proposal needs its exact inclusive source window')
        if proposal.requirements.window[0] != context.step:
            raise ValueError('market proposal does not begin at the current cut')
        end = proposal.requirements.window[1] + 1
    else:
        c = proposal.candidate
        if (c.get('context_key') != context_key(context) or c.get('start_step') != context.step
                or type(c.get('terminal_step')) is not int or c['terminal_step'] != TERMINAL_STEP):
            raise ValueError('endgame source is stale or nonterminal')
        end = TERMINAL_STEP
    if bridge.source_end_step != end or not context.step < end <= TERMINAL_STEP:
        raise ValueError('bridge must use the source exclusive endpoint')
    if end == TERMINAL_STEP and bridge.remainder != CashChange(Estimate.exact(0), Estimate.exact(0)):
        raise ValueError('terminal source has exactly zero post-terminal remainder')
    raw = CashChange(proposal.own_cash_delta, proposal.rival_cash_delta)
    return raw.plus(bridge.remainder).plus(bridge.reference), {
        'raw': raw.to_dict(), 'remainder': bridge.remainder.to_dict(),
        'reference': bridge.reference.to_dict(), 'source_end_step': end}


def _public_pair_value(context, proposal, pair):
    if not isinstance(pair, PublicTerminalPair) or not isinstance(pair.join, ValueJoin):
        raise ValueError('typed actor-only PublicTerminalPair required; evaluator tables are forbidden')
    pair.join.validate(context, proposal)
    expected = dict(context_key=context_key(context), start_step=context.step,
                    terminal_step=TERMINAL_STEP, baseline_sha256=BASELINE_SHA256)
    for result, key, sha in ((pair.control, None, None),
                             (pair.child, proposal.key, proposal_sha256(proposal))):
        if not isinstance(result, SuffixResult) or result.status != 'COMPLETE' or result.proposal != key:
            raise ValueError('complete correctly keyed public control and child required')
        r = result.realization
        if (not isinstance(r, dict) or set(r) != {*expected, 'proposal_sha256'}
                or any(r[k] != v for k, v in expected.items()) or r['proposal_sha256'] != sha
                or type(r['start_step']) is not int or type(r['terminal_step']) is not int):
            raise ValueError('public terminal binding mismatch or evaluator-only realization fields')
        _actor_json(r)
    value = CashChange(Estimate.exact(pair.child.own_cash-pair.control.own_cash),
                       Estimate.exact(pair.child.rival_cash-pair.control.rival_cash))
    return value, {'control': {'own_cash': pair.control.own_cash, 'rival_cash': pair.control.rival_cash},
                   'child': {'own_cash': pair.child.own_cash, 'rival_cash': pair.child.rival_cash}}


@dataclass(frozen=True)
class Conversion:
    original: Proposal
    proposal: Proposal
    context_key: str
    start_step: int
    player: int
    source_scope: str
    reason: str
    method: str
    comparison_law: str | None
    terms: Mapping
    basis: ValueBasis | None = None

    def to_dict(self):
        return dict(schema=SCHEMA, input_scope='ACTOR_ONLY_CERTIFIED',
                    scope=CANONICAL_SCOPE, context_key=self.context_key,
                    baseline_sha256=BASELINE_SHA256, start_step=self.start_step,
                    terminal_step=TERMINAL_STEP, player=self.player,
                    key=list(self.original.key), original_sha256=proposal_sha256(self.original),
                    proposal_sha256=proposal_sha256(self.proposal), source_scope=self.source_scope,
                    reason=self.reason, method=self.method, comparison_law=self.comparison_law,
                    raw={f: getattr(self.original, f).to_dict() for f in FIELDS},
                    converted={f: getattr(self.proposal, f).to_dict() for f in FIELDS},
                    terms=dict(self.terms), guards=list(self.proposal.guards), parents=list(self.proposal.parents),
                    basis=None if self.basis is None else _actor_json(self.basis.to_dict()))


def convert(context: StrategyContext, proposal: Proposal, *, bridge=None, terminal=None) -> Conversion:
    """Convert one proposal without mutating it. Missing/refused sources keep UNKNOWN."""
    key = context_key(context)  # includes own memory and ordered shops, not derived services
    _actor_json(dict(observation=context.observation, memory=context.memory))
    _proposal_actor_json(context, proposal)
    original = Proposal.from_dict(proposal.to_dict())
    normalized = replace(original, own_cash_delta=UNKNOWN, rival_cash_delta=UNKNOWN, margin_delta=UNKNOWN)
    scope = source_scope(original)
    reason, method, law, terms, basis = 'MISSING_CONTINUATION_BRIDGE', 'UNKNOWN', None, {}, None
    if original.module not in _PROPOSAL_OWNERS:
        reason = MODULE_ADAPTERS.get(original.module, {}).get('guard', 'UNSUPPORTED_MODULE')
    elif bridge is None and terminal is None:
        if original.module in ('production', 'shop_response'):
            reason = 'MISSING_PAIRED_TERMINAL_PRODUCER'
    else:
        try:
            if bridge is not None and terminal is not None:
                raise ValueError('one value producer required, not a bridge AND a terminal pair')
            if terminal is not None:
                if isinstance(terminal, TerminalRefusal) and terminal.details is not None:
                    # A refusal may retain diagnostics, never a ValueBasis. Recompute
                    # against these exact actor/proposal bytes, not a caller's claim.
                    if (terminal.reason != _PRODUCTION_NONIDENTITY_REASON
                            or terminal.details != production_continuation_gap(context, original)):
                        raise ValueError('stale or altered production continuation gap')
                    return Conversion(original, normalized, key, context.step, context.player,
                        scope, 'REFUSED: '+terminal.reason, 'UNKNOWN', None,
                        {'production_continuation_gap': _actor_json(terminal.details)}, None)
                if isinstance(terminal, (TerminalRefusal, MarketTerminalRefusal)):
                    raise ValueError(terminal.reason)
                if isinstance(terminal, ProductionTerminalPair):
                    value, terms = _production_pair_value(context, original, terminal)
                    j, method = terminal.join, 'PAIRED_TERMINAL_IDENTITY'
                elif isinstance(terminal, MarketTerminalPair):
                    value, terms = _market_pair_value(context, original, terminal)
                    j, method = terminal.join, 'PAIRED_TERMINAL_FIXED_PURCHASE'
                else:
                    value, terms = _public_pair_value(context, original, terminal)
                    j, method = terminal.join, 'PAIRED_PUBLIC_TERMINAL'
            else:
                value, terms = _bridge_value(context, original, bridge)
                j, method = bridge.join, 'RAW_PLUS_REMAINDER_PLUS_REFERENCE'
            law = j.comparison_law
            guards = tuple(dict.fromkeys((*original.guards, *j.guards,
                'canonical terminal value conditional on common law='+law)))
            normalized = replace(original, own_cash_delta=value.own, rival_cash_delta=value.rival,
                margin_delta=value.margin, guards=guards,
                invalidated_by=tuple(dict.fromkeys((*original.invalidated_by,
                    'any actor context, own continuation, or common-law assumption change'))),
                parents=tuple(sorted(set(original.parents+j.parents))))
            reason = 'CONVERTED' if value.margin.known else 'UNKNOWN_REQUIRED_CASH_TERM'
            if value.margin.known:
                basis = ValueBasis.bind(context, normalized, parents=normalized.parents)
        except (TypeError, ValueError, AttributeError) as error:
            reason = 'REFUSED: '+str(error)
            method, law, terms = 'UNKNOWN', None, {}
    return Conversion(original, normalized, key, context.step, context.player,
                      scope, reason, method, law, terms, basis)


def normalize_view(view: StepView, *, bridges: Mapping | None = None,
                   terminals: Mapping | None = None) -> tuple[StepView, tuple[Conversion, ...]]:
    """Install current compatible joins for Coordinator; never reuse old value/capacity proofs.

    Inputs are keyed by ORIGINAL Proposal.key. All original proposals remain in the
    menu, including UNKNOWN competitors. Rebind capacity against the returned view.
    Different conditional-law equivalence classes withhold all bases; choosing one
    by list order is unsafe.  Equivalence changes compatibility only, never the
    original Conversion.comparison_law identifier.
    """
    bridges, terminals = bridges or {}, terminals or {}
    keys = {p.key for p in view.proposals}
    if (set(bridges) | set(terminals)) - keys:
        raise ValueError('value producer refers to a proposal outside this StepView')
    rows = tuple(convert(view.context, p, bridge=bridges.get(p.key), terminal=terminals.get(p.key))
                 for p in view.proposals)
    laws = {r.comparison_law for r in rows if r.basis is not None}
    law_classes = {COMPARISON_LAW_EQUIVALENCE.get(law, law) for law in laws}
    bases = ({r.proposal.key: r.basis for r in rows if r.basis is not None}
             if len(law_classes) <= 1 else {})
    notes = dict(view.notes)
    notes['value_basis'] = ('INCOMPATIBLE_COMPARISON_LAWS; all basis publication withheld'
                            if len(law_classes) > 1 else '; '.join(sorted({r.reason for r in rows})))
    if len(laws) > 1 and len(law_classes) == 1:
        notes['value_basis'] += '; EXPLICIT_EQUIVALENT_COMPARISON_LAWS=' + ','.join(sorted(laws))
    if view.feasibility:
        notes['value_basis'] += '; REBIND_CAPACITY_AFTER_NORMALIZATION'
    derived = {**view.context.derived, 'coordinator_values': bases}
    ctx = replace(view.context, derived=derived)
    return StepView(ctx, tuple(r.proposal for r in rows), {}, view.reserve, notes), rows


def _terminal_record(result):
    return dict(status=result.status, own_cash=result.own_cash if result.status == 'COMPLETE' else None,
                rival_cash=result.rival_cash if result.status == 'COMPLETE' else None,
                margin=result.margin, outcome=None if result.outcome is None else result.outcome.value,
                failure=result.failure, realization=result.realization)


def calibration_row(state_id: str, conversion: Conversion, control: SuffixResult,
                    child: SuffixResult, *, snapshot_sha256: str) -> dict:
    """EVALUATOR ONLY receipt comparison; does not authenticate an unprovided snapshot.

    Use calibrate_snapshot for source/K0006 authentication and actual evaluation.
    Standalone use requires the caller's snapshot/source provenance. Realized-minus-
    predicted error is signed. Interval coverage is containment, not a new certificate.
    Point matches/misses include heuristics; they are not statistical confidence tests.
    """
    row = dict(schema=SCHEMA+'-calibration-row', input_scope='EVALUATOR_ONLY', state_id=state_id,
        snapshot_sha256=snapshot_sha256, baseline_sha256=BASELINE_SHA256,
        conversion=conversion.to_dict(), control=_terminal_record(control), child=_terminal_record(child),
        status='UNPAIRED', failure=None, realized_delta={f: UNKNOWN.to_dict() for f in FIELDS},
        error={f: None for f in FIELDS}, covered={f: None for f in FIELDS},
        guard_validation='NOT_INFERRED_FROM_OUTCOME; PRODUCER/ADOPTER_OWNS_GUARDS')
    if control.status != 'COMPLETE' or child.status != 'COMPLETE':
        row.update(status='FAILED', failure=control.failure or child.failure)
        return row
    a, b = control.realization, child.realization
    expected = dict(snapshot_sha256=snapshot_sha256, start_step=conversion.start_step,
                    end_step=TERMINAL_STEP, seat=conversion.player,
                    steps=TERMINAL_STEP-conversion.start_step)
    if (not isinstance(snapshot_sha256, str) or len(snapshot_sha256) != 64
            or any(c not in '0123456789abcdef' for c in snapshot_sha256)
            or control.proposal is not None or child.proposal != conversion.original.key
            or not isinstance(a, dict) or not isinstance(b, dict)
            or any(a.get(k) != x or b.get(k) != x for k, x in expected.items())):
        row['failure'] = 'wrong control/proposal/cut/seat, or incomplete terminal receipt'
        return row
    actual = CashChange(Estimate.measured(child.own_cash-control.own_cash),
                        Estimate.measured(child.rival_cash-control.rival_cash))
    row.update(status='COMPLETE', realized_delta=actual.to_dict())
    for name, actual_value in zip(FIELDS, (actual.own.value, actual.rival.value, actual.margin.value)):
        predicted = getattr(conversion.proposal, name)
        if predicted.kind is Evidence.BOUND:
            row['covered'][name] = ((predicted.low is None or actual_value >= predicted.low)
                                    and (predicted.high is None or actual_value <= predicted.high))
        elif predicted.known:
            row['error'][name] = actual_value-predicted.value
            row['covered'][name] = abs(row['error'][name]) <= 1e-9
    return row


def calibration_table(rows: Sequence[Mapping]) -> dict:
    rows = list(rows)
    def counts(subset):
        complete = [r for r in subset if r['status'] == 'COMPLETE']
        coverage = [r['covered']['margin_delta'] for r in complete]
        return dict(rows=len(subset), complete_pairs=len(complete), failed_pairs=len(subset)-len(complete),
            predicted_margins=sum(c is not None for c in coverage), covered_margins=sum(c is True for c in coverage),
            missed_margins=sum(c is False for c in coverage), unknown_margins=sum(c is None for c in coverage))
    return dict(schema=SCHEMA+'-calibration', input_scope='EVALUATOR_ONLY', scope=CANONICAL_SCOPE,
        counts=counts(rows), by_module={m: counts([r for r in rows if r['conversion']['key'][0] == m])
                                      for m in sorted({r['conversion']['key'][0] for r in rows})},
        rows=rows, interpretation='Conditional suffix diagnostics; misses retained; no actor publication or policy-strength claim.')


def calibrate_snapshot(snapshot, context: StrategyContext, conversions: Sequence[Conversion], *,
                       state_id: str, continuation_policy=None, max_suffixes: int = 4,
                       max_steps: int = 191, seconds: float = 120.) -> dict:
    """EVALUATOR ONLY: authenticate and replay one control plus <=3 proposed branches.

    Every evaluate invocation consumes a suffix attempt, including errors/refusals.
    The caller owns the campaign's cumulative budget across calls/processes; this
    per-call cap does not reset a ticket's total allowance. Native traces and the
    source-bound snapshot remain evaluator artifacts, never actor inputs. trace_jsonl
    retains the native hash's insertion-order bytes even through sorted JSON exports. Exact
    restoration authenticates the captured own K0006 implementation and memory.
    Matching actor observations does NOT establish sufficiency of producer guards.
    """
    from farm_neat.strategy import counterfactual as cf
    from farm_neat.evaluation.k0006 import K0006Adapter
    conversions = tuple(conversions)
    if (type(max_suffixes) is not int or not 1 <= max_suffixes <= 4
            or not conversions or len(conversions)+1 > max_suffixes):
        raise ValueError('explicit budget 1..4 must cover control AND every child attempt')
    if (type(max_steps) is not int or max_steps < 1 or type(seconds) not in (int, float)
            or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('positive finite native step/time budget required')
    if not isinstance(state_id, str) or not state_id.strip():
        raise ValueError('recorded state_id required')
    if len({r.original.key for r in conversions}) != len(conversions):
        raise ValueError('duplicate proposals: share recorded results for alternate forecasts')
    for row in conversions:
        if (row.context_key != context_key(context) or row.start_step != context.step
                or row.player != context.player):
            raise ValueError('conversion does not belong to this actor context')
    runtime = cf.restore(snapshot)  # validates source pins, Python, full actors and hash seed
    if type(runtime.actors[runtime.seat]) is not K0006Adapter:
        raise ValueError('calibration control must be the restored frozen K0006 actor')
    obs = runtime.environment.agent_observation(runtime.seat)
    if (runtime.seat != context.player or runtime.environment.step_index != context.step
            or obs != dict(context.observation)):
        raise ValueError('snapshot and actor context are not the same cut/seat/observation')
    traces, results = [], []
    for proposal in (None, *(r.original for r in conversions)):
        trace = []
        result = cf.evaluate(snapshot, proposal, continuation_policy=continuation_policy,
                             max_steps=max_steps, seconds=seconds, trace=trace)
        results.append(result)
        traces.append(dict(proposal_sha256=None if proposal is None else proposal_sha256(proposal),
                           receipt=_terminal_record(result), trace=trace,
                           trace_jsonl=''.join(cf._json(row)+'\n' for row in trace)))
    report = calibration_table([calibration_row(state_id, r, results[0], result,
        snapshot_sha256=snapshot.sha256) for r, result in zip(conversions, results[1:])])
    report.update(native_suffix_attempts=len(results), native_joint_steps=sum(len(t['trace']) for t in traces),
        max_suffixes=max_suffixes, max_steps_per_suffix=max_steps, seconds_per_suffix=seconds,
        traces=traces, snapshot_sha256=snapshot.sha256,
        baseline_validation='SOURCE_AND_MEMORY_AUTHENTICATED_BY_COUNTERFACTUAL_RESTORE',
        limit='Same recorded cut only; own/rival producer-law truth is not inferred from a successful replay.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', action='store_true', help='Print source scopes and missing producer guards; zero games')
    parser.add_argument('--example', action='store_true', help='Print the canonical formula and integration seam; zero games')
    args = parser.parse_args(argv)
    if args.registry:
        result = {k: dict(v) for k, v in MODULE_ADAPTERS.items()}
    elif args.example:
        result = dict(schema=SCHEMA, scope=CANONICAL_SCOPE, baseline_sha256=BASELINE_SHA256,
                      terminal_step=TERMINAL_STEP,
                      formula='raw_window_delta + paired_remainder_delta + source_control_minus_K0006_terminal',
                      consumer='Coordinator.decide(normalize_view(view, bridges=..., terminals=...)[0])',
                      native_games=0, default='UNKNOWN until independent producer evidence is supplied')
    else:
        parser.print_help()
        return 0
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
