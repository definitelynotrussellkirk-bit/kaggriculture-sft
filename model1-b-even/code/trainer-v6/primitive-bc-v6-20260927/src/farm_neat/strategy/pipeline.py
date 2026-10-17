"""Actor-only, per-step join of the accepted strategy modules.

Usage::

    pipeline = StrategyPipeline()
    view = pipeline.step(own_observation, memory=own_memory)  # -> types.StepView
    action = coordinator(view)                              # NOT owned here
    own_memory = pipeline.remember(view, action)             # actual action only
    # json.dumps(own_memory, allow_nan=False) is a complete pipeline continuation.

``run(StrategyContext)`` is the equivalent typed entrypoint. Each call builds a
fresh publication set; no stale proposal, valuation, reserve or certificate is
cached across steps. Own memory is JSON; only the accepted ProductionBook is
rehydrated temporarily for its existing consumer. No biology/commitment book is
inferred from a count, a plan, or the absence of an input.

KEY_REGISTRY is the sole derived-key registry, including missing producers.
Optional actor-only publishers have signature ``publish(context) -> value|None``
and can fill its INPUT keys. ``capacity_inputs`` additionally receives the final
proposal tuple: ``publish(context, proposals)``. It binds cases at that exact cut,
AFTER all other publications, and capacity runs last without changing the context.
Live value/suffix services cannot be hashed by the accepted capacity binder;
that specific upstream rejection stays UNKNOWN without removing the service.
A publisher is trusted application code, not JSON-loaded code. Its certificates,
resource budget and actor-only closure are the caller's responsibility. Missing
publishers are not implemented with fabricated certificates or zero defaults.

Order: current input publications -> shop observation/memory -> opponent history
-> reserve -> dawn production -> event-triggered market -> terminal-horizon
endgame -> advisory liquidity menu -> final proposal-specific capacity checks.
Shop/demand and opponent publications refresh every step. Search triggers are
observed changes, native calendar events and the accepted endgame action budget,
not a chronological economic policy. Default bundled calendar is a precondition.

Two deliberately non-equivalent joins stay UNKNOWN:
* opponent -> opponent_state: past effective displacement is NOT a certified
  future request/rank word, private stock, or counterfactual resource envelope.
* liquidity -> market: cash funding alone does NOT certify slot/capital shadow
  prices or protected inventory. Those fields remain UNKNOWN, not zero.

Production's embedded ``capacity_planner`` answer precedes its own complete
proposal. Feeding capacity back changes those proposal bytes and invalidates
CapacityCase's full-proposal binding. The final answers therefore live ONLY in
StepView.feasibility; the embedded pre-check remains UNKNOWN. No circular rebind.
Likewise accepted counterfactual.py is an evaluator, not an actor-side producer:
no snapshot/hidden-world execution is started here. A separately certified public
suffix law may supply endgame's narrower ``counterfactual`` input.

Limits: zero native games, no adoption, training or economic changes. Each called
module retains its declared constructor budget (96 historical transitions by
default, 16 shop probes, 32 production changes, 4096 market comparisons/2M unit
quotes, 4096 liquidity states, 4096 states per capacity case; endgame's own Budget).
Reports count calls, original proposals and reasons, not wins or useful actions.
The replay CLI projects compact public records; absent private custody stays absent.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from farm_neat.game.constants import ANIMALS, PRODUCTS
from farm_neat.strategy import capacity, endgame, liquidity, market_loop, opponent, production, shop_response
from farm_neat.strategy.types import (
    EVALUATOR_ONLY, LAST_STEP, Evidence, Estimate, Feasibility, Proposal,
    Requirements, StepView, StrategyContext, UNKNOWN,
)

MODULES = ('shop_response', 'opponent', 'liquidity', 'production', 'market_loop',
           'endgame', 'capacity', 'counterfactual')
MEMORY_SCHEMA = 'farm-strategy-pipeline-memory-v1'
BOOK_SCHEMA = production.MEMORY_BOOK_SCHEMA
_CAPACITY_BIND_ERROR = 'capacity inputs must be finite data, not executable objects'
_CAPACITY_BIND_UNKNOWN = 'UPSTREAM_CAPACITY_CONTEXT_REJECTS_EXECUTABLE_PROVIDERS'


def _key(producer, readers=(), description=''):
    return MappingProxyType(dict(producer=producer, readers=tuple(readers), description=description))


KEY_REGISTRY = MappingProxyType({
    'capacity_commitments': _key('INPUT', (), 'Read-only partial K0006 intentions; not a capacity certificate.'),
    'coordinator_values': _key('pipeline', (), 'Compatible terminal joins, normalized before capacity binding.'),
    'cashflow': _key('INPUT', ('liquidity',), 'Certified current own funding scope; memory.cashflow is a fallback.'),
    'production_replan': _key('INPUT', ('shop_response',), 'Explicit marginal V0 module menu, not an inferred plan.'),
    'production_value': _key('INPUT', ('production',), 'Actor-only complete-portfolio valuation provider.'),
    'capacity_inputs': _key('INPUT', ('capacity',), 'Cases bound at the final context/proposal cut.'),
    'endgame_guards': _key('INPUT', ('endgame',), 'Context-bound public/own terminal certificates.'),
    'counterfactual': _key('INPUT', ('endgame',), 'PublicSuffixModel only; not the evaluator Snapshot API.'),
    'capacity_planner': _key('DEFERRED', ('production',), 'Final answer is StepView.feasibility, never circular proposal feedback.'),
    'shop_response': _key('shop_response', ('shop_response',), 'Typed current ShopView; ordered published shops retained.'),
    'opponent': _key('opponent', ('opponent',), 'Typed historical public OpponentState.'),
    'opponent_state': _key('pipeline', ('market_loop',), 'Explicit UNKNOWN order-word adapter; never upgrades a FlowLaw.'),
    'liquidity': _key('pipeline', ('market_loop',), 'Reserve plus UNKNOWN unproduced shadow prices and inventory protection.'),
    'liquidity_candidates': _key('pipeline', ('liquidity',), 'This step\'s unmodified proposal menu, not certified additive.'),
    'liquidity_result': _key('liquidity', (), 'Typed advisory allocation result; reserve is also on StepView.'),
    'production': _key('production', (), 'Typed ProductionAnalysis when due; does not fabricate production_replan.'),
    'market_loop': _key('market_loop', (), 'Original proposal tuple when an event triggers a scan.'),
    'endgame': _key('endgame', (), 'Typed EndgameResult when remaining actions enter the search horizon.'),
})
INPUT_KEYS = frozenset(k for k, v in KEY_REGISTRY.items() if v['producer'] == 'INPUT')


def _actor_only(value: Any) -> None:
    """Reject named leaks recursively, including typed publication fields.

    Cannot authenticate a caller's certificate or inspect an opaque callback's
    closure. Such callbacks must obey the documented actor-only provider contract.
    """
    if isinstance(value, Mapping):
        leaked = EVALUATOR_ONLY.intersection(value)
        if leaked:
            raise ValueError('evaluator-only input: ' + ', '.join(sorted(leaked)))
        for item in value.values():
            _actor_only(item)
    elif is_dataclass(value):
        for f in fields(value):
            _actor_only(getattr(value, f.name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _actor_only(item)


def _validate_input(key: str, value: Any) -> None:
    _actor_only(value)
    if key == 'counterfactual' and (not isinstance(value, endgame.PublicSuffixModel)
                                    or value.input_scope != 'ACTOR_ONLY_DETERMINISTIC'):
        raise ValueError('counterfactual input must be an actor-only PublicSuffixModel, never an evaluator snapshot')


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(k, (str, int)) or isinstance(k, bool) for k in value):
            raise ValueError('memory needs JSON string/integer keys')
        if len({str(k) for k in value}) != len(value):
            raise ValueError('memory key collision after JSON encoding')
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('memory/observation must be finite JSON, not executable objects')


def _book(value: Any) -> production.ProductionBook:
    return production.book_from_memory(value)


def memory_to_json(memory: Mapping[str, Any]) -> dict:
    """Safe, closed codec for the one accepted typed module memory; preserve others."""
    if not isinstance(memory, Mapping):
        raise ValueError('own memory must be a mapping')
    _actor_only(memory)
    result = dict(memory)
    if 'production' in result:
        result['production'] = {'schema': BOOK_SCHEMA, **_book(result['production']).to_dict()}
    return _json(result)


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(_json(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _metrics():
    return dict(calls=0, proposals=0, comparable_values=0, unknown={}, abstentions={})


def _reason(row, kind, reason, count=1):
    if reason:
        row[kind][str(reason)] = row[kind].get(str(reason), 0) + count


class StrategyPipeline:
    """Stateless economic join; ``stats`` is diagnostic only, never policy state."""

    def __init__(self, *, history_limit: int = 96, publishers: Mapping[str, Callable] | None = None,
                 normalize_values: bool = False, value_evidence: Callable | None = None,
                 retire_inputs: tuple[str, ...] = ()):
        if type(normalize_values) is not bool or (value_evidence is not None and not callable(value_evidence)):
            raise ValueError('normalization flag must be bool and value evidence must be a callable')
        if value_evidence is not None and not normalize_values:
            raise ValueError('value evidence requires normalization')
        if (not isinstance(retire_inputs, tuple) or len(set(retire_inputs)) != len(retire_inputs)
                or any(key not in INPUT_KEYS for key in retire_inputs)):
            raise ValueError('retire_inputs must be distinct registered INPUT keys')
        self.normalize_values, self.value_evidence = normalize_values, value_evidence
        self.retire_inputs = retire_inputs
        self.publishers = dict(publishers or {})
        if set(self.publishers) - INPUT_KEYS or any(not callable(f) for f in self.publishers.values()):
            raise ValueError('publishers must map registered INPUT keys to actor-only callables')
        self.shop = shop_response.ShopResponse()
        self.opponent = opponent.OpponentModel(history_limit=history_limit, default_calendar=True)
        self.production = production.ProductionReplanner()
        self.market = market_loop.MarketLoop()
        self.endgame = endgame.EndgameModule()
        self.last_report: dict = {}
        self.stats = {'steps': 0, 'modules': {name: _metrics() for name in MODULES}, 'events': {}, 'missing_keys': {}}

    def step(self, observation: Mapping, memory: Mapping | None = None) -> StepView:
        if not isinstance(observation, Mapping):
            raise ValueError('own observation must be a mapping')
        town = observation.get('town', {})
        shops = town.get('unlocked_shops', ()) if isinstance(town, Mapping) else ()
        return self.run(StrategyContext(observation.get('step'), observation.get('player'), observation,
                                        {} if memory is None else memory, tuple(shops)))

    def _events(self, context):
        obs = context.observation
        old = context.memory.get('pipeline')
        signature = _hash({k: obs.get(k) for k in ('market', 'private', 'farms')})
        # Refresh for changed own authorizations/certificates as well as public resources.
        authorizations = _hash({k: v for k, v in context.memory.items()
                                if k not in ('opponent', 'shop_response', 'pipeline', production.SERVICE_RECEIPT_KEY)})
        events = []
        if old is not None:
            if (not isinstance(old, Mapping) or old.get('schema') != MEMORY_SCHEMA
                    or type(old.get('step')) is not int):
                raise ValueError('invalid pipeline continuation schema')
            if old.get('player') != context.player:
                raise ValueError('pipeline memory belongs to another player seat')
            if old['step'] > context.step:
                raise ValueError('pipeline time moved backwards; clear memory at a new game')
            prefix = tuple(old.get('shops', ()))
            if context.shops[:len(prefix)] != prefix:
                raise ValueError('published prefix was rewritten or truncated')
            if old['step'] == context.step:
                events.append('refresh')
            if signature != old.get('market_signature'):
                events.append('market_event')
            if authorizations != old.get('authorization_signature'):
                events.append('own_memory_changed')
        else:
            events.append('initial')
            prefix = ()
        if context.shops[len(prefix):]:
            events.append('new_shop')
        if context.step % self.shop.clock.turns_per_day == 0:
            events.append('dawn')
        if LAST_STEP + 1 - context.step <= self.endgame.budget.max_suffix_actions:
            events.append('late_horizon')
        if context.derived or self.publishers:
            events.append('fresh_inputs')
        memory = dict(context.memory)
        memory['pipeline'] = dict(schema=MEMORY_SCHEMA, step=context.step, player=context.player,
                                  shops=list(context.shops), market_signature=signature,
                                  authorization_signature=authorizations)
        return events, replace(context, memory=memory)

    def run(self, context: StrategyContext) -> StepView:
        """One actor step. Invalid input raises; absent certification is explicit UNKNOWN."""
        if not isinstance(context, StrategyContext):
            raise ValueError('StrategyContext required')
        if set(context.derived) - INPUT_KEYS:
            raise ValueError('caller may publish INPUT keys only; pipeline outputs cannot be reused')
        if set(context.derived).intersection(self.publishers):
            raise ValueError('one producer per input: both value and callback supplied')
        _actor_only(context)
        obs = _json(context.observation)
        for key in ('step', 'player'):
            if key in obs and (type(obs[key]) is not int or obs[key] != getattr(context, key)):
                raise ValueError('observation/context ' + key + ' mismatch')
        native = obs.get('town', {}).get('unlocked_shops')
        shops = tuple(native) if native is not None else context.shops
        if context.shops and context.shops != shops:
            raise ValueError('observation/context ordered shop prefix mismatch')
        incoming = {k: v for k, v in context.derived.items() if v is not None}
        for key, value in incoming.items():
            _validate_input(key, value)
        ctx = replace(context, observation=obs, shops=shops, memory=memory_to_json(context.memory), derived=incoming)
        events, ctx = self._events(ctx)
        rows = {name: _metrics() for name in MODULES}
        missing = {}
        proposals: list[Proposal] = []
        notes = {}

        def publish(key, value):
            nonlocal ctx
            _actor_only(value)
            if key in INPUT_KEYS:
                _validate_input(key, value)
            ctx = ctx.with_derived(key, value)

        def supplied(key, *args):
            if key in ctx.derived:
                return
            value = self.publishers[key](ctx, *args) if key in self.publishers else None
            if key == 'cashflow' and value is None:
                value = ctx.memory.get('cashflow')
            if value is not None:
                publish(key, value)
            else:
                missing[key] = 'MISSING_PRODUCER:' + key

        def collect(name, output):
            output = tuple(output)
            if any(not isinstance(p, Proposal) or p.module != name for p in output):
                raise ValueError(name + ' emitted an invalid/foreign proposal')
            if len({p.key for p in (*proposals, *output)}) != len(proposals) + len(output):
                raise ValueError('duplicate proposal key')
            proposals.extend(output)
            rows[name]['proposals'] += len(output)
            for p in output:
                for field in ('own_cash_delta', 'rival_cash_delta', 'margin_delta'):
                    if getattr(p, field).kind is Evidence.UNKNOWN:
                        _reason(rows[name], 'unknown', 'UNCERTIFIED_' + field.upper())
                if isinstance(p.candidate, Mapping):
                    for reason in p.candidate.get('refusals', ()):
                        _reason(rows[name], 'unknown', reason)
                    if p.candidate.get('status') == 'KEEP_BASELINE' or p.candidate.get('kind') == 'KEEP':
                        _reason(rows[name], 'abstentions', 'KEEP_BASELINE')
            return output

        supplied('production_replan')
        rows['shop_response']['calls'] += 1
        update = self.shop.update(ctx)
        ctx = replace(update.context, memory=memory_to_json(update.memory))
        collect('shop_response', update.proposals)
        for reason in update.view.reasons:
            _reason(rows['shop_response'], 'unknown', reason)
        for value in update.view.inventory.values():
            if value.shed_units == UNKNOWN or value.solo_now == UNKNOWN:
                _reason(rows['shop_response'], 'unknown', value.reason)
        for value in update.view.producers.values():
            if value.mode == 'UNKNOWN':
                _reason(rows['shop_response'], 'unknown', value.reason)
        if not update.proposals:
            _reason(rows['shop_response'], 'abstentions', 'NO_DAWN_SHOP_EDIT')
        if update.view.demand is not None:
            if any(update.view.demand.at(ctx.step).values()):
                events.append('demand_event')

        rows['opponent']['calls'] += 1
        ctx = self.opponent.publish(ctx)
        state = ctx.derived['opponent']
        latest = [r for r in state.history if r.end_step == ctx.step]
        if not latest:
            _reason(rows['opponent'], 'unknown', 'NO_OBSERVED_TRANSITION')
        for sample in latest:
            if sample.effect.kind is Evidence.UNKNOWN:
                _reason(rows['opponent'], 'unknown', sample.reason)
        _reason(rows['opponent'], 'unknown', 'FUTURE_ORDER_WORD_AND_RESOURCE_SUPPORT_UNCERTIFIED')
        publish('opponent_state', dict(step=ctx.step, kind=Evidence.UNKNOWN.value,
                                      reason='HISTORICAL_EFFECTIVE_FLOW_IS_NOT_A_FUTURE_ORDER_WORD',
                                      basis='derived.opponent; public history only'))

        supplied('cashflow')
        initial_menu = {'proposals': tuple(proposals), 'joint_values_certified': False}
        early_context = ctx.with_derived('liquidity_candidates', initial_menu)
        rows['liquidity']['calls'] += 1
        # Repeated shop instances are real (shops are drawn with replacement) and reach
        # liquidity unchanged; intake fixed its duplicate-prefix rejection.
        early = liquidity.analyze(early_context)
        if early.funding.get('reason'):
            _reason(rows['liquidity'], 'unknown', early.funding['reason'])
        publish('liquidity', dict(step=ctx.step, reserve=early.reserve,
                                 capital_shadow=UNKNOWN, slot_shadow=UNKNOWN,
                                 protected_inventory={p: UNKNOWN for p in (*PRODUCTS, *ANIMALS)}))
        _reason(rows['liquidity'], 'unknown', 'MARKET_SHADOW_PRICES_AND_INVENTORY_PROTECTION_UNPRODUCED')

        supplied('production_value')
        missing['capacity_planner'] = 'DEFERRED_TO_STEPVIEW_FEASIBILITY: full-proposal binding forbids circular feedback'
        if 'dawn' in events:
            rows['production']['calls'] += 1
            pmem = dict(ctx.memory)
            owned = pmem.get(production.BOOK_OWNER_KEY)
            if owned is not None:
                if (not isinstance(owned, Mapping)
                        or set(owned) != {'schema', 'mode'}
                        or owned.get('schema') != production.BOOK_OWNER_SCHEMA
                        or owned.get('mode') not in ('clean_farm_dawn_bootstrap', 'actor_service_receipts')):
                    raise ValueError('invalid actor production-book owner marker')
                # Actor-generated books are dawn snapshots, never rolling claims.
                # Rebuild them so verified_step advances only through fresh receipt
                # evidence and a clean bootstrap cannot survive into live history.
                pmem.pop('production', None)
            build_reason = None
            if 'production' not in pmem:
                build = production.produce_actor_book(replace(ctx, memory=pmem))
                if build.book is not None:
                    pmem['production'] = build.book
                    pmem[production.BOOK_OWNER_KEY] = {
                        'schema': production.BOOK_OWNER_SCHEMA,
                        'mode': ('actor_service_receipts' if build.book.cohorts
                                 else 'clean_farm_dawn_bootstrap'),
                    }
                else:
                    pmem.pop(production.BOOK_OWNER_KEY, None)
                    build_reason = build.reason
                # Persist only actor-visible state/provenance; the typed book is
                # rehydrated below for the existing ProductionReplanner consumer.
                ctx = replace(ctx, memory=memory_to_json(pmem))
                pmem = dict(ctx.memory)
            if 'production' in pmem:
                pmem['production'] = _book(pmem['production'])
            analysis = self.production.analyze(replace(ctx, memory=pmem))
            publish('production', analysis)
            collect('production', analysis.proposals)
            for reason in (() if build_reason is None else (build_reason,)):
                _reason(rows['production'], 'unknown', reason)
            for reason in analysis.diagnostics:
                _reason(rows['production'], 'unknown', reason)
        else:
            _reason(rows['production'], 'abstentions', 'NOT_DAWN')

        trigger = set(events) - {'late_horizon'}
        if trigger:
            rows['market_loop']['calls'] += 1
            result = collect('market_loop', self.market.propose(ctx))
            publish('market_loop', result)
        else:
            _reason(rows['market_loop'], 'abstentions', 'NO_MARKET_EVENT')

        supplied('counterfactual')
        supplied('endgame_guards')
        _reason(rows['counterfactual'], 'abstentions', 'EVALUATOR_ONLY: no actor-safe snapshot execution')
        if 'late_horizon' in events:
            rows['endgame']['calls'] += 1
            result = self.endgame.solve(ctx)
            publish('endgame', result)
            collect('endgame', result.proposals)
            if result.status is Evidence.UNKNOWN:
                _reason(rows['endgame'], 'unknown', result.reason)
            for failure in result.failures:
                _reason(rows['endgame'], 'unknown', failure)
        else:
            _reason(rows['endgame'], 'abstentions', 'OUTSIDE_TERMINAL_ACTION_BUDGET')

        publish('liquidity_candidates', {'proposals': tuple(proposals), 'joint_values_certified': False})
        if len(proposals) <= 256:
            rows['liquidity']['calls'] += 1
            allocation = liquidity.analyze(ctx)
            if allocation.funding.get('reason'):
                _reason(rows['liquidity'], 'unknown', allocation.funding['reason'])
            for offer in allocation.allocations:
                if offer.decision in ('UNKNOWN', 'PASS'):
                    _reason(rows['liquidity'], 'unknown' if offer.decision == 'UNKNOWN' else 'abstentions', offer.reason)
        else:
            allocation = early
            _reason(rows['liquidity'], 'unknown', 'ALLOCATION_NOT_RUN: menu exceeds 256; no silent truncation')
        publish('liquidity_result', allocation)

        # Some accepted inputs are executable services rather than evidence data.
        # Once every declared module consumer has run, an opt-in composition may
        # retire those services before the final value/capacity cut. This does not
        # turn a missing producer into a value: it only prevents an already-consumed
        # callback from poisoning CapacityCase's intentionally finite binding.
        retired = []
        if self.retire_inputs:
            derived = dict(ctx.derived)
            for key in self.retire_inputs:
                if key in derived:
                    derived.pop(key)
                    retired.append(key)
            if retired:
                ctx = replace(ctx, derived=derived)

        # The accepted value adapter changes proposal bytes. Normalize FIRST,
        # then ask the accepted capacity producer to bind the resulting context.
        # A value callback returns only the existing normalize_view keyword maps;
        # its scientific guards remain the producer's responsibility.
        conversions = ()
        raw_proposals = tuple(proposals)
        if self.normalize_values:
            from farm_neat.strategy.value_basis import normalize_view
            before = StepView(ctx, raw_proposals, reserve=allocation.reserve, notes=notes)
            evidence = self.value_evidence(before) if self.value_evidence is not None else {}
            if not isinstance(evidence, Mapping) or set(evidence) - {'bridges', 'terminals'}:
                raise ValueError('value evidence must contain only bridges/terminals maps')
            normalized, conversions = normalize_view(before, **evidence)
            ctx, proposals, notes = normalized.context, list(normalized.proposals), dict(normalized.notes)
            # Do not count withheld (incompatible-law) bases as comparable.
            bases = ctx.derived['coordinator_values']
            for row in conversions:
                if row.proposal.key in bases:
                    rows[row.proposal.module]['comparable_values'] += 1
                else:
                    reason = ('INCOMPATIBLE_COMPARISON_LAWS' if row.basis is not None else row.reason)
                    _reason(rows[row.proposal.module], 'unknown', reason)

        # ALL publication and continuation mutations precede the binding callback.
        try:
            supplied('capacity_inputs', tuple(proposals))
        except ValueError as error:
            # Exact accepted-binder failure only. Never swallow malformed resource
            # data or provider failures, and never omit services to make a proof bind.
            if str(error) != _CAPACITY_BIND_ERROR:
                raise
            missing['capacity_inputs'] = _CAPACITY_BIND_UNKNOWN
        # A validation-only batch snapshots the now-final data once. Publication
        # still binds each proposal independently, and no prepared state crosses
        # the callback above. Preserve the per-proposal legacy error boundary:
        # an executable input may reject a bound case while an absent case keeps
        # its ordinary UNKNOWN reason.
        try:
            capacity_details = capacity._explain_feasibilities(ctx, proposals)
        except ValueError as error:
            if str(error) != _CAPACITY_BIND_ERROR:
                raise
            capacity_details = None
        feasibility = {}
        for index, proposal in enumerate(proposals):
            rows['capacity']['calls'] += 1
            try:
                detail = (capacity_details[index] if capacity_details is not None
                          else capacity.explain_feasibility(ctx, proposal))
                data = detail['result']
                answer = Feasibility(proposal.key, data['verdict'], data['witness'], data['certificate'],
                                     Requirements.from_dict(data['marginal']), tuple(data['parents']))
                reason = detail['reason']
            except ValueError as error:
                if str(error) != _CAPACITY_BIND_ERROR:
                    raise
                answer = Feasibility(proposal.key, 'UNKNOWN', marginal=proposal.requirements,
                                     parents=proposal.parents)
                reason = _CAPACITY_BIND_UNKNOWN
            feasibility[proposal.key] = answer
            if answer.verdict.value == 'UNKNOWN':
                _reason(rows['capacity'], 'unknown', reason)
        if not proposals:
            _reason(rows['capacity'], 'abstentions', 'NO_PROPOSALS')
        for key, spec in KEY_REGISTRY.items():
            if spec['readers'] and key not in ctx.derived and key not in retired:
                missing.setdefault(key, 'MISSING_PRODUCER:' + key)
        for key, reason in missing.items():
            for reader in KEY_REGISTRY[key]['readers']:
                _reason(rows[reader], 'unknown', reason)
        for name, row in rows.items():
            reasons = [*row['unknown'], *row['abstentions']]
            if reasons:
                notes[name] = '; '.join(reasons)
        view = StepView(ctx, tuple(proposals), feasibility, allocation.reserve, notes)
        self.last_report = dict(step=ctx.step, player=ctx.player, events=sorted(set(events)),
                                missing_keys=missing, modules=rows,
                                retired_inputs=retired,
                                value_conversions=[r.to_dict() for r in conversions],
                                original_proposals=[p.to_dict() for p in raw_proposals])
        self.stats['steps'] += 1
        for event in set(events):
            self.stats['events'][event] = self.stats['events'].get(event, 0) + 1
        for key in missing:
            self.stats['missing_keys'][key] = self.stats['missing_keys'].get(key, 0) + 1
        for name, row in rows.items():
            total = self.stats['modules'][name]
            for field in ('calls', 'proposals', 'comparable_values'):
                total[field] += row[field]
            for kind in ('unknown', 'abstentions'):
                for reason, count in row[kind].items():
                    _reason(total, kind, reason, count)
        return view

    def remember(self, view: StepView, action: Mapping) -> dict:
        """Commit ONLY the actually submitted own action, after coordinator selection.

        Opponent-flow history and production service receipts share the same
        post-selection boundary: discarded proposals and the frozen shadow never
        reach either memory.  Receipt replay sees only this actor's pre-action
        observation plus the action that the coordinator actually returned.
        """
        if not isinstance(view, StepView):
            raise ValueError('StepView required')
        _actor_only(action)
        memory = self.opponent.remember(view.context, action)
        receipt_context = replace(view.context, memory=memory)
        memory[production.SERVICE_RECEIPT_KEY] = production.update_service_receipts(
            receipt_context, action)
        return memory_to_json(memory)


def composed_pipeline() -> StrategyPipeline:
    """Real accepted producers, including identity and final fixed-purchase value joins.

    The caller supplies current ``cashflow``/``capacity_commitments`` from
    commitments.extract(context, frozen_actor).publish(context), or they remain
    UNKNOWN. No production book, V0 plan, post-worker custody, rival-PASS law,
    non-identity K0006 reference rebase or labor certificate is inferred here.

    The consumed PortfolioValuator service is retired before capacity binding;
    missing capacity evidence still remains UNKNOWN.
    This factory launches zero native games and is not a strength candidate.
    """
    from farm_neat.strategy.capacity_inputs import publish
    from farm_neat.strategy.valuation import PortfolioValuator
    valuator = PortfolioValuator()
    return StrategyPipeline(normalize_values=True, value_evidence=composed_value_evidence,
                            retire_inputs=('production_value',), publishers={
        'production_value': lambda context: valuator,
        'capacity_inputs': publish,
    })


def composed_value_evidence(view: StepView) -> dict:
    """Join accepted actor-safe proofs to the common frozen-K0006 terminal basis.

    This composes existing value producers; it does not invent a continuation
    model, terminal schedule generator, or funding certificate:

    * production KEEP is delegated to ``value_basis.production_terminal``;
    * market PASS/KEEP_BASELINE is admitted only when every accepted identity/no-op
      invariant still holds;
    * the final-step unchanged-SELL-queue/fixed-seed-append class is delegated to
      ``value_basis.market_terminal_evidence``; its exact negative value remains
      comparable even when later funding/selection screens reject it;
    * unsupported interventions receive no invented bridge and remain UNKNOWN.

    The terminal producer omits unsupported rows, including PASS. Its publication
    is disjoint from the identity bridges; a proposal never gets two value sources.
    This callback does not make MarketLoop.propose generate new candidate families.

    When a production KEEP proof exists at the same cut, its comparison-law label is
    reused for market identity rows.  The physical law is the same in both cases --
    child and control execute the unchanged frozen-K0006 suffix -- and using one law
    prevents the value-basis consumer from treating two names for that same law as
    incompatible.  No numeric term is inferred from this label unification.
    """
    from farm_neat.strategy.coordinator import BASELINE_SHA256
    from farm_neat.strategy.value_basis import (
        CashChange, ContinuationBridge, ProductionTerminalPair, ValueJoin,
        market_terminal_evidence, production_terminal_evidence, source_scope,
    )

    if not isinstance(view, StepView):
        raise TypeError('StepView required')
    evidence = production_terminal_evidence(view)
    terminals = {**evidence.get('terminals', {}),
                 **market_terminal_evidence(view)['terminals']}
    production_laws = {
        terminal.join.comparison_law for terminal in terminals.values()
        if isinstance(terminal, ProductionTerminalPair)
    }
    if len(production_laws) > 1:
        raise ValueError('same-cut production identity producers disagree on comparison law')
    market_law = (next(iter(production_laws)) if production_laws
                  else 'FROZEN_K0006_IDENTITY_CONTINUATION:' + BASELINE_SHA256)
    zero = Estimate.exact(0)
    zero_change = CashChange(zero, zero)
    parents = (
        '5def61686e29c45cd3246e62dbbd200f3d4ac5fce6bf2531e436d19e92696aca',
        'a2d8c566d0bfbf17329377e4b4b438f3cf4147663bbcafab537b579a95280054',
        'work/runs/reviewer-play/solver-continuation-bridge/continuation_bridge.py',
        'src/farm_neat/strategy/market_loop.py',
        'src/farm_neat/strategy/value_basis.py',
    )
    bridges = {}
    for proposal in view.proposals:
        if proposal.module != 'market_loop':
            continue
        candidate = proposal.candidate
        if (not isinstance(candidate, Mapping)
                or candidate.get('schema') != 'farm-market-loop-v1'
                or candidate.get('status') != 'KEEP_BASELINE'
                or candidate.get('family') != 'PASS'
                or candidate.get('execution') != 'NATIVE_MARKET_QUEUE_NOT_V0_RULES'
                or candidate.get('quantity') != 0
                or candidate.get('orders') != {}
                or candidate.get('refusals') not in (None, [])):
            continue
        req = proposal.requirements
        if (req.cash != zero or req.reserve != zero or req.inventory
                or req.market_slots != zero or req.worker_hours != zero
                or req.land_quadrants != zero or req.window != (view.context.step, view.context.step)):
            continue
        if any(getattr(proposal, field) != zero
               for field in ('own_cash_delta', 'rival_cash_delta', 'margin_delta')):
            continue
        if source_scope(proposal) != 'MARKET_WINDOW_VS_QUEUE':
            continue
        join = ValueJoin.bind(
            view.context, proposal, comparison_law=market_law,
            guards=(
                'proposal is accepted market PASS/KEEP_BASELINE with exact zero resources and value',
                'PASS retains the continuously advanced frozen-K0006 baseline queue',
                'child and control therefore execute the identical frozen-K0006 suffix',
                'no evaluator/private/future input is used to value the identity continuation',
            ),
            parents=parents,
        )
        bridges[proposal.key] = ContinuationBridge(
            join=join, source_scope='MARKET_WINDOW_VS_QUEUE',
            source_end_step=view.context.step + 1,
            remainder=zero_change, reference=zero_change,
        )
    return {'bridges': bridges, 'terminals': terminals}


def replay_market_witness(path: str | Path, *, history_limit: int = 96) -> dict:
    """Both seats, every pre-action row of one compact recorded game; zero games.

    The loop knows only public money/market, the prefix published by that row and
    the focal seat's orders. It never forwards the archive's seed, teams, rewards,
    future rows/shops, signatures, or the other seat's orders into the pipeline.
    Memory is JSON round-tripped after EVERY actual action, not just at the end.
    """
    path = Path(path)
    with gzip.open(path, 'rt') as stream:
        data = json.load(stream)
    if data.get('schema') != 'kaggle-replay-market-witness-v1' or tuple(data['products']) != tuple(PRODUCTS):
        raise ValueError('unsupported compact market witness schema/product order')
    rows = data['rows']
    if [r['step'] for r in rows] != list(range(LAST_STEP + 2)):
        raise ValueError('complete recorded pre-action rows 0..718 plus terminal row 719 required')
    clocks = data['shops_unlocked']
    if not clocks or clocks[0] != [0, []] or any(a[0] >= b[0] for a, b in zip(clocks, clocks[1:])):
        raise ValueError('ordered public shop publication records required')
    started, cpu = time.perf_counter(), time.process_time()
    seats = []
    for player in (0, 1):
        pipe = StrategyPipeline(history_limit=history_limit)
        memory, prefix, publication = {}, [], 0
        digest = hashlib.sha256()
        for row in rows[:-1]:
            step = row['step']
            while publication < len(clocks) and clocks[publication][0] <= step:
                prefix = list(clocks[publication][1])
                publication += 1
            obs = dict(step=step, player=player, day=step // 24, hour=step % 24,
                       farms=[{'money': amount} for amount in row['money']],
                       market={'inventory': dict(zip(PRODUCTS, row['inventory']))},
                       town={'unlocked_shops': list(prefix)})
            view = pipe.step(obs, memory=memory)
            memory = json.loads(json.dumps(pipe.remember(view, {'market': row['orders'][player]}), allow_nan=False))
            digest.update(json.dumps({'step': step, 'proposals': [p.to_dict() for p in view.proposals],
                                      'notes': dict(view.notes), 'memory_sha256': _hash(memory)},
                                     sort_keys=True, separators=(',', ':')).encode())
        seats.append(dict(player=player, steps=len(rows) - 1, memory_roundtrips=len(rows) - 1,
                          trace_sha256=digest.hexdigest(), final_memory_sha256=_hash(memory),
                          stats=pipe.stats))
    return dict(schema='farm-strategy-pipeline-replay-v1', recorded_games=1, native_games=0,
                actor_steps=2 * (len(rows) - 1), terminal_rows_skipped=1, errors=[],
                observation_scope='PUBLIC_MARKET_AND_MONEY_ONLY; NO_PRIVATE_CUSTODY_OR_PLAN_IN_REPLAY',
                history_limit=history_limit, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                elapsed_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu, seats=seats)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, help='Compact recorded market witness (.json.gz), not a new game')
    parser.add_argument('--history-limit', type=int, default=96)
    parser.add_argument('--output', type=Path, help='Write replay receipt JSON')
    parser.add_argument('--registry', action='store_true')
    parser.add_argument('--example', action='store_true')
    args = parser.parse_args(argv)
    if args.registry:
        result = {key: dict(value) for key, value in KEY_REGISTRY.items()}
    elif args.replay:
        result = replay_market_witness(args.replay, history_limit=args.history_limit)
    elif args.example:
        pipe = StrategyPipeline(history_limit=args.history_limit)
        view = pipe.step({'step': 0, 'player': 0, 'day': 0, 'hour': 0,
                          'farms': [{'money': 3000}, {'money': 3000}],
                          'market': {'inventory': dict.fromkeys(PRODUCTS, 10000)},
                          'town': {'unlocked_shops': []}})
        result = {'report': pipe.last_report, 'proposals': [p.to_dict() for p in view.proposals]}
    else:
        parser.print_help()
        return 0
    text = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
