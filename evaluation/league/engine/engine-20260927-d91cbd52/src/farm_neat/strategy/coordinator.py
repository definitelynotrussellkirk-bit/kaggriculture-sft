"""One justified intervention over a frozen, continuously advanced K0006 shadow.

Coordinator.decide(StepView) is the module-facing entry point. It screens funding
and certified capacity cuts, records touched-resource conflicts, and compares only
terminal margin deltas against the SAME K0006 baseline. UNKNOWN is not zero.
Strict interval dominance is independent of module/input order; exact ties abstain.
A selection is conditional on the producer certificates and execution-time guards,
not a native playing-strength or universal optimality claim.

Integration seams (no imports of the parallel pipeline/adoption implementations):
* pipeline.step(StrategyContext) -> StepView, or an equivalent callable. The returned
  context may update own memory and derived, but not the actor observation/seat/shops.
* adopter.adopt(own_branch, context, Proposal) -> counterfactual.Adoption. It owns ALL
  runtime guards, invalidation and action realization. first_action is called ONCE;
  longer-lived hooks must be installed on that branch by the adapter. The branch is
  discarded at the proposal window's end or any reported failure. No adapter receives
  the frozen shadow. One active change locks further adoption, even if disjoint.
* derived['liquidity'].funding (or the same mapping) supplies protected_resources and
  funding status. StepView.reserve supplies the numeric split. Full-replacement
  proposals touching protected resources require a re-certified replacement schedule;
  this coordinator NEVER adds a full production remainder to existing obligations.
* derived['coordinator_values'][Proposal.key] = ValueBasis (or its JSON dict) attests
  that the ORIGINAL Proposal deltas are terminal vs frozen K0006 at this exact context.
  This is necessary: market_loop quotes a market window; endgame may quote versus PASS.
  A ValueBasis is a producer assertion, NOT a constructor that proves those facts.
* derived['counterfactual'] optionally supplies a PUBLIC deterministic suffix model:
  context_key, baseline_sha256, input_scope='ACTOR_ONLY_DETERMINISTIC',
  restoration_exact=True, law, parents, evaluate(context, proposal_or_None).
  It returns types.SuffixResult with realization binding context_key, baseline_sha256,
  start_step and terminal_step=719. Its law must include exact own continuation.
  These are the existing public suffix model semantics plus an explicit baseline join.
  The evaluator-only captured-world counterfactual service is NOT an actor model.
  Missing/hidden/stale models leave a source-bound request, never launch native games.

Budgets bound proposals, public-model calls/suffix length, and adoption attempts
(including failures) per game. coordinator_history is reserved own-memory state.
Public callbacks are trusted code; their own resource use cannot be preempted here.
No native games run on import, in decide(), or in the default --example entry point.

CoordinatorPolicy() defaults to explicit all-abstention. Inject the parallel modules
when integrated; absence is not silently replaced by guessed producer data. Its own
observation/action history hash is retained in memory. Value producers still own the
sufficiency of their supplied own-memory/continuation, never just the current cash.
records and optional record_sink retain decisions, emissions, and verification events.
Emission is not realization. finish(final_observation) checks a last-turn verify hook;
without it that check stays PENDING, including under the unchanged shared panel runner.

--panel uses the UNMODIFIED shared router harness. Optional --pipeline and --adopter
are no-argument module:factory references implementing the seams above. Every spawned
actor has a separate hash-chained JSONL stream next to the panel directory. The full
1232-game command is maintainer-only; reviewers must count ALL their native attempts.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping

from farm_neat.evaluation.common import bound, file_sha
from farm_neat.evaluation.k0006 import K0006Adapter, MEMBERS
from farm_neat.game.exact_engine import DEFAULT_CONFIGURATION
from farm_neat.strategy.counterfactual import (
    Adoption, DefaultContinuation, ProposalFailed, restore_actor, snapshot_actor, source_pins,
)
from farm_neat.strategy.types import (
    EVALUATOR_ONLY, Evidence, Estimate, Proposal, StepView, StrategyContext, SuffixResult, Verdict,
)

ROOT = Path(__file__).resolve().parents[3]
BASELINE_SHA256 = MEMBERS['main.py']
SCHEMA = 'farm-strategy-coordinator-v1'
TERMINAL_STEP = 719


def _wire(value):
    """Actor/receipt data only; do not stringify arbitrary objects or drop private keys."""
    if isinstance(value, Mapping):
        if EVALUATOR_ONLY & set(value):
            raise ValueError('evaluator-only nested actor input')
        if any(not isinstance(k, str) for k in value):
            raise ValueError('actor/receipt data needs string JSON keys')
        return {k: _wire(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wire(v) for v in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('finite actor/receipt JSON required')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def context_key(context: StrategyContext) -> str:
    """Same public projection as endgame.context_key, excluding recursive derived outputs."""
    return _hash(_wire(dict(step=context.step, player=context.player,
                           observation=context.observation, memory=context.memory, shops=context.shops)))


def _parents(values):
    values = tuple(values)
    if not values or any(not isinstance(v, str) or not v.strip() or v.startswith('/')
                         or '..' in v.split('/') or '\\' in v for v in values):
        raise ValueError('nonempty repository-relative parents required')
    return values


@dataclass(frozen=True)
class ValueBasis:
    """Missing value-scope join, local to this consumer; never changes Proposal estimates."""
    context_key: str
    proposal_sha256: str
    parents: tuple[str, ...]
    baseline_sha256: str = BASELINE_SHA256
    terminal_step: int = TERMINAL_STEP
    input_scope: str = 'ACTOR_ONLY_CERTIFIED'

    @classmethod
    def bind(cls, context, proposal, *, parents):
        """Bind a producer's assertion; caller must independently justify terminal scope."""
        return cls(context_key(context), _hash(proposal.to_dict()), _parents(parents))

    def to_dict(self):
        return asdict(self)

    def valid(self, context, proposal):
        try:
            _parents(self.parents)
            return (self.context_key == context_key(context)
                    and self.proposal_sha256 == _hash(proposal.to_dict())
                    and self.baseline_sha256 == BASELINE_SHA256
                    and type(self.terminal_step) is int and self.terminal_step == TERMINAL_STEP
                    and self.input_scope == 'ACTOR_ONLY_CERTIFIED')
        except (TypeError, ValueError):
            return False


def _limits(value):
    if value.kind is Evidence.EXACT:
        return value.value, value.value
    if value.kind is Evidence.BOUND:
        return value.low, value.high
    return None, None


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def resource_conflicts(left, right):
    """Literal names plus declared colon namespaces; no hidden family priority/alias table."""
    return sorted({a if len(a) <= len(b) else b for a in left for b in right
                   if a == b or a.startswith(b + ':') or b.startswith(a + ':')})


@dataclass
class Decision:
    reason: str
    selected: Proposal | None = None
    screening: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    dominance: list[dict] = field(default_factory=list)
    funding: dict = field(default_factory=dict)
    valuation: dict = field(default_factory=dict)
    counterfactual: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    def to_dict(self):
        return dict(schema=SCHEMA, reason=self.reason,
                    selected=None if self.selected is None else self.selected.to_dict(),
                    screening=self.screening, conflicts=self.conflicts, dominance=self.dominance,
                    funding=self.funding, valuation=self.valuation,
                    counterfactual=self.counterfactual, notes=self.notes)


def _funding(view):
    r, ctx = view.reserve, view.context
    report = {'reason': 'UNKNOWN_COMMITMENT_FUNDING', 'safe_discretionary': None}
    if r is None:
        return report, ()
    report.update(cash=r.cash, protected=r.protected.to_dict(), discretionary=r.discretionary.to_dict(),
                  commitments=list(r.commitments), parents=list(r.parents))
    try:
        cash = ctx.observation['farms'][ctx.player]['money']
        if type(cash) not in (int, float) or not math.isfinite(cash) or cash != r.cash or cash < 0:
            report['reason'] = 'RESERVE_CASH_MISMATCH'
            return report, ()
    except (KeyError, IndexError, TypeError):
        return report, ()
    protected_low, protected_high = _limits(r.protected)
    disc_low, disc_high = _limits(r.discretionary)
    if any(n is not None and n < 0 for n in (protected_low, protected_high, disc_low, disc_high)):
        report['reason'] = 'INVALID_RESERVE'
        return report, ()
    if protected_high is None or disc_low is None:
        return report, ()
    # The real pipeline publishes the final allocation separately from its
    # narrow market-facing reserve/shadow adapter. Retain the legacy seam only
    # when no final result was published.
    liquidity = ctx.derived.get('liquidity_result', ctx.derived.get('liquidity'))
    funding = _get(liquidity, 'funding', {})
    resources = _get(funding, 'protected_resources')
    if not isinstance(resources, (list, tuple)) or any(not isinstance(x, str) or not x for x in resources):
        report['reason'] = 'UNKNOWN_PROTECTED_RESOURCES'
        return report, ()
    if _get(funding, 'status') == 'INSUFFICIENT':
        report['reason'] = 'HARD_COMMITMENT_SHORTFALL'
        return report, tuple(resources)
    if protected_high > cash:
        report['reason'] = 'HARD_COMMITMENT_SHORTFALL'
        return report, tuple(resources)
    report.update(reason='CERTIFIED_FUNDING_SCREEN', safe_discretionary=min(disc_low, cash-protected_high),
                  protected_resources=list(resources))
    return report, tuple(resources)


def _proposal_funding(view: StepView, proposal: Proposal):
    """Return the accepted proposal-local funding certificate for this exact key.

    ``liquidity.analyze`` can sometimes prove a proposal FUNDED/NOT_FUNDED even
    while the global Reserve remains UNKNOWN (for example a zero-capital noncash
    no-op).  The old coordinator discarded that narrower proof and therefore
    screened every proposal as UNKNOWN_COMMITMENT_FUNDING.  Consume only the
    already-published certificate here; never infer missing cash scope locally.
    """
    liquidity = view.context.derived.get('liquidity_result')
    funding = _get(liquidity, 'funding', {})
    certificates = _get(funding, 'proposal_certificates')
    if certificates is None:
        return None
    if not isinstance(certificates, (list, tuple)):
        raise ValueError('liquidity proposal_certificates must be a sequence')
    matches = []
    for row in certificates:
        if not isinstance(row, Mapping):
            raise ValueError('proposal funding certificate must be a mapping')
        key = row.get('proposal')
        if (isinstance(key, (list, tuple)) and tuple(key) == proposal.key):
            matches.append(row)
    if len(matches) > 1:
        raise ValueError('duplicate proposal funding certificate')
    if not matches:
        return None
    row = matches[0]
    status = row.get('status')
    reason = row.get('reason')
    scope_complete = row.get('scope_complete')
    resources = row.get('protected_resources')
    if (status not in ('FUNDED', 'NOT_FUNDED', 'UNKNOWN')
            or not isinstance(reason, str) or not reason
            or type(scope_complete) is not bool
            or not isinstance(resources, (list, tuple))
            or any(not isinstance(x, str) or not x for x in resources)):
        raise ValueError('malformed proposal funding certificate')
    if status != 'UNKNOWN' and not scope_complete:
        raise ValueError('decisive proposal funding certificate must close its scope')
    bank_cents = row.get('bank_cents')
    capital_upper = row.get('capital_upper_cents')
    if (type(bank_cents) is not int or bank_cents < 0
            or (capital_upper is not None and (type(capital_upper) is not int or capital_upper < 0))
            or (status == 'FUNDED' and capital_upper is None)):
        raise ValueError('proposal funding certificate has invalid cents')
    cash = view.context.observation['farms'][view.context.player]['money']
    if type(cash) not in (int, float) or not math.isfinite(cash) or round(cash * 100) != bank_cents:
        raise ValueError('proposal funding certificate bank does not match current actor cash')
    return dict(status=status, reason=reason, scope_complete=scope_complete,
                capital_upper_cents=capital_upper, protected_resources=list(resources),
                source_scope_complete=row.get('source_scope_complete'),
                unresolved_source_scope=list(row.get('unresolved_source_scope', ())))


def _basis(context, proposal):
    values = context.derived.get('coordinator_values', {})
    item = values.get(proposal.key) if isinstance(values, Mapping) else None
    if isinstance(item, Mapping):
        try:
            item = ValueBasis(**item)
        except (TypeError, ValueError):
            return None
    return item if isinstance(item, ValueBasis) and item.valid(context, proposal) else None


def _value_record(proposal, basis):
    return {**{k: getattr(proposal, k).to_dict() for k in
               ('own_cash_delta', 'rival_cash_delta', 'margin_delta')},
            'basis': basis.to_dict(), 'scope': 'TERMINAL_VS_FROZEN_K0006; CONDITIONAL_ON_PRODUCER_CERTIFICATES'}


class Coordinator:
    """A bounded single-change selector, not a budget allocator or an evaluator sandbox."""
    def __init__(self, *, max_proposals=128, max_evaluations=9, max_suffix_actions=191):
        for name, value, lo, hi in (('max_proposals',max_proposals,1,1024),
                                    ('max_evaluations',max_evaluations,0,129),
                                    ('max_suffix_actions',max_suffix_actions,0,719)):
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'{name} must be an integer in {lo}..{hi}')
            setattr(self, name, value)

    def decide(self, view: StepView) -> Decision:
        if not isinstance(view, StepView):
            raise TypeError('one typed StepView required')
        ctx = view.context
        context_key(ctx)  # Reject non-JSON/hidden context before any callback.
        d = Decision('NO_PROPOSALS', notes=dict(view.notes))
        if not view.proposals:
            return d
        if len(view.proposals) > self.max_proposals:
            d.reason = 'PROPOSAL_BUDGET_EXHAUSTED'
            return d
        d.funding, protected = _funding(view)
        menu, values = [], {}
        for p in sorted(view.proposals, key=lambda p: p.key):
            row = dict(proposal=list(p.key), reason='ELIGIBLE_FOR_COMPARISON')
            d.screening.append(row)
            answer = view.feasibility.get(p.key)
            if answer is not None:
                row['feasibility'] = dict(verdict=answer.verdict.value, witness=answer.witness,
                                          certificate=answer.certificate, parents=list(answer.parents))
                if answer.verdict is Verdict.INFEASIBLE:
                    row['reason'] = 'CERTIFIED_INFEASIBLE'
                    continue
            if p.requirements.window is not None and not p.requirements.window[0] <= ctx.step <= p.requirements.window[1]:
                row['reason'] = 'OUTSIDE_EXECUTION_WINDOW'
                continue
            basis = _basis(ctx, p)
            if basis is not None:
                # Preserve certified values on rejected rows too. Funding and
                # selection order are unchanged; an early refusal must not hide
                # a successfully connected terminal producer from the audit.
                row['valuation'] = _value_record(p, basis)
            # Proposal-local funding is an integration refinement for proposals
            # already certified on the common terminal basis.  Do not let a narrow
            # cash proof turn an otherwise UNKNOWN-valued generic pipeline into a
            # different selection/counterfactual policy.
            local_funding = _proposal_funding(view, p) if basis is not None else None
            if local_funding is not None:
                row['funding_certificate'] = local_funding
                if local_funding['status'] == 'NOT_FUNDED':
                    row['reason'] = 'CERTIFIED_NOT_FUNDED:' + local_funding['reason']
                    continue
            protected_for_proposal = (tuple(local_funding['protected_resources'])
                                      if local_funding is not None and local_funding['status'] == 'FUNDED'
                                      else protected)
            if resource_conflicts(protected_for_proposal, p.touches):
                row['reason'] = 'PROTECTED_RESOURCE_REQUIRES_REPLACEMENT_CERTIFICATE'
                continue
            cash, held = _limits(p.requirements.cash), _limits(p.requirements.reserve)
            if any(n is not None and n < 0 for n in (*cash, *held)):
                row['reason'] = 'INVALID_NEGATIVE_CAPITAL'
                continue
            if local_funding is not None and local_funding['status'] == 'FUNDED':
                upper = local_funding['capital_upper_cents']
                row['capital_upper'] = None if upper is None else upper / 100
            else:
                if local_funding is not None and d.funding['safe_discretionary'] is None:
                    row['reason'] = 'UNKNOWN_PROPOSAL_FUNDING:' + local_funding['reason']
                    continue
                if d.funding['safe_discretionary'] is None:
                    row['reason'] = d.funding['reason']
                    continue
                if cash[1] is None or held[1] is None:
                    row['reason'] = 'UNKNOWN_CAPITAL_REQUIREMENT'
                    continue
                row['capital_upper'] = cash[1] + held[1]
                if row['capital_upper'] > d.funding['safe_discretionary']:
                    row['reason'] = 'CAPITAL_EXCEEDS_SAFE_DISCRETIONARY'
                    continue
            if basis is not None:
                lo, hi = _limits(p.margin_delta)
                if hi is not None and hi <= 0:
                    row['reason'] = 'NO_POSITIVE_TERMINAL_MARGIN'
                    continue
                values[p.key] = (lo, hi, basis)
            else:
                row['value_scope'] = 'UNKNOWN_TERMINAL_K0006_BASELINE'
            menu.append(p)
        for i, a in enumerate(menu):
            for b in menu[i+1:]:
                resources = resource_conflicts(a.touches, b.touches)
                if resources:
                    d.conflicts.append(dict(left=list(a.key),right=list(b.key),resources=resources))
        # A complete terminal value already includes the cost of the chosen remainder.
        # Do not subtract reserve again or compare unlike cash/worker units.
        dominated = set()
        for a in menu:
            low = values.get(a.key, (None, None, None))[0]
            if low is None:
                continue
            for b in menu:
                high = values.get(b.key, (None, None, None))[1]
                if a.key != b.key and high is not None and low > high:
                    dominated.add(b.key)
                    d.dominance.append(dict(better=list(a.key),worse=list(b.key),
                        margin_lower=low,margin_upper=high,rule='STRICT_TERMINAL_INTERVAL_DOMINANCE'))
        remaining = [p for p in menu if p.key not in dominated]
        for row in d.screening:
            if tuple(row['proposal']) in dominated:
                row['reason'] = 'DOMINATED_TERMINAL_MARGIN'
        if not remaining:
            d.reason = 'NO_JUSTIFIED_INTERVENTION'
            return d
        if len(remaining) == 1:
            p = remaining[0]
            low, _, basis = values.get(p.key, (None, None, None))
            if low is not None and low > 0:
                d.reason, d.selected = 'CERTIFIED_POSITIVE_TERMINAL_MARGIN', p
                d.valuation = _value_record(p, basis)
                return d
        d.reason = 'UNRESOLVED_VALUE_COMPARISON'
        d.counterfactual = dict(schema=SCHEMA+'-counterfactual-request', status='PENDING',
            context_key=context_key(ctx), baseline_sha256=BASELINE_SHA256,
            start_step=ctx.step, terminal_step=TERMINAL_STEP,
            proposals=[p.to_dict() for p in remaining], max_evaluations=self.max_evaluations,
            max_suffix_actions=self.max_suffix_actions, native_games_launched=0,
            purpose='PAIRED_TERMINAL_WDL_THEN_MARGIN; NO_HIDDEN_WORLD_ACTOR_INPUT')
        self._resolve_public(ctx, remaining, d)
        return d

    def _resolve_public(self, ctx, proposals, d):
        request = d.counterfactual
        if len(proposals)+1 > self.max_evaluations or TERMINAL_STEP-ctx.step > self.max_suffix_actions:
            request['status'] = 'BUDGET_EXHAUSTED'
            return
        model = ctx.derived.get('counterfactual')
        if model is None:
            return
        try:
            parents = _parents(_get(model,'parents',()))
            law = _get(model,'law')
            if (_get(model,'context_key') != context_key(ctx)
                    or _get(model,'baseline_sha256') != BASELINE_SHA256
                    or _get(model,'input_scope') != 'ACTOR_ONLY_DETERMINISTIC'
                    or _get(model,'restoration_exact') is not True
                    or not isinstance(law,str) or not law.strip()
                    or not callable(_get(model,'evaluate'))):
                raise ValueError('public suffix law/context/baseline/restoration not certified')
        except (TypeError, ValueError) as error:
            request.update(status='MODEL_REFUSED', failure=str(error))
            return
        request.update(status='RUNNING_PUBLIC_MODEL', evaluations=0, results=[],
                       callback_native_games='NOT_OBSERVED; PROVIDER_OWNS_RESOURCE_ACCOUNTING')
        results = []
        for p in [None, *proposals]:
            # No provider, evaluator envelope, or sibling branch state reaches this input.
            branch = StrategyContext(ctx.step,ctx.player,_wire(ctx.observation),_wire(ctx.memory),ctx.shops)
            request['evaluations'] += 1
            try:
                r = _get(model,'evaluate')(branch,None if p is None else Proposal.from_dict(p.to_dict()))
                expected = dict(context_key=context_key(ctx),start_step=ctx.step,
                                terminal_step=TERMINAL_STEP,baseline_sha256=BASELINE_SHA256)
                if (not isinstance(r,SuffixResult) or r.status != 'COMPLETE'
                        or r.proposal != (None if p is None else p.key)
                        or not isinstance(r.realization,dict)
                        or any(r.realization.get(k) != v for k,v in expected.items())
                        or context_key(branch) != context_key(ctx)):
                    raise ValueError('failed, mutated or unbound terminal suffix: '+str(getattr(r,'failure',None)))
                request['results'].append(dict(proposal=None if p is None else list(p.key),
                                               own_cash=r.own_cash,rival_cash=r.rival_cash,
                                               margin=r.margin,outcome=r.outcome.value))
                results.append(r)
            except Exception as error:
                request.update(status='FAILED',failure=f'{type(error).__name__}: {error}')
                return  # Failed branches are UNKNOWN, never zero; no partial-menu winner.
        request['status'] = 'COMPLETE_PUBLIC_MODEL'
        best_margin = max(r.margin for r in results)
        winners = [i for i,r in enumerate(results) if r.margin == best_margin]
        if len(winners) != 1 or winners[0] == 0:
            return
        i = winners[0]; control, child = results[0], results[i]
        d.reason, d.selected = 'PUBLIC_TERMINAL_COUNTERFACTUAL', proposals[i-1]
        d.valuation = dict(own_cash_delta=Estimate.exact(child.own_cash-control.own_cash).to_dict(),
                           rival_cash_delta=Estimate.exact(child.rival_cash-control.rival_cash).to_dict(),
                           margin_delta=Estimate.exact(child.margin-control.margin).to_dict(),
                           conditional_law=law, parents=list(parents),
                           input_scope='ACTOR_ONLY_DETERMINISTIC',baseline_sha256=BASELINE_SHA256,
                           baseline_outcome=control.outcome.value,selected_outcome=child.outcome.value)


def _component(component):
    if component is None:
        return {'kind':'ALL_ABSTAIN'}
    target = component if inspect.isfunction(component) else type(component)
    path = inspect.getsourcefile(target)
    return {'type': target.__module__+':'+target.__qualname__,
            'source_sha256': file_sha(path) if path else None,
            'declared': copy.deepcopy(getattr(component,'identity',None))}


def _action(value):
    if (not isinstance(value,dict) or set(value) != {'farmer','hands','market'}
            or not all(isinstance(value[k],list) for k in value) or not value['farmer']):
        raise ProposalFailed('adoption must emit farmer/hands/market action lists')
    _wire(value)
    return value


class CoordinatorPolicy:
    """Executable actor. Each fresh game needs a fresh instance, as in router_panel."""
    pending = False

    def __init__(self, *, pipeline=None, adopter=None, coordinator=None, max_adoptions=16, record_sink=None,
                 capture_commitments=False):
        if type(max_adoptions) is not int or not 0 <= max_adoptions <= 719:
            raise ValueError('max_adoptions must be 0..719')
        if type(capture_commitments) is not bool:
            raise ValueError('capture_commitments must be bool')
        self.capture_commitments = capture_commitments
        self.base = K0006Adapter()
        self.pipeline, self.adopter = pipeline, adopter or DefaultContinuation()
        self.coordinator = coordinator or Coordinator()
        self.max_adoptions, self.record_sink = max_adoptions, record_sink
        self.memory, self.records, self.verification_events = {}, [], []
        self.last_step, self.player, self.active, self.awaiting = None, None, None, None
        self.adoptions = self.adoption_attempts = 0
        assets = source_pins()
        assets.update({str(p.relative_to(ROOT)):file_sha(p) for p in (ROOT/'src/farm_neat/strategy').glob('*.py')})
        self.identity = bound(dict(schema=SCHEMA+'-actor',baseline=self.base.identity,assets=assets,
            pipeline=_component(pipeline),adopter=_component(self.adopter),
            selection=dict(vars(self.coordinator)),max_adoptions=max_adoptions,
            capture_commitments=capture_commitments,
            objective='CERTIFIED_TERMINAL_MARGIN; PUBLIC_SUFFIX_WDL_THEN_MARGIN; TIES_ABSTAIN',
            continuation='FROZEN_SHADOW_ADVANCED_ON_ACTUAL_OBSERVATIONS; ONE_ACTIVE_BRANCH'))

    def _emit(self, record):
        if self.record_sink is not None:
            self.record_sink(copy.deepcopy(record))

    def _verify(self, observation):
        pending, self.awaiting = self.awaiting, None
        if pending is None:
            return False
        adoption, before, record = pending
        try:
            failure = adoption.verify(before,copy.deepcopy(observation))
        except Exception as error:
            failure = f'{type(error).__name__}: {error}'
        record['verification'] = 'PROPOSAL_FAILED' if failure is not None else 'VERIFIED'
        event = dict(schema=SCHEMA+'-verification',step=observation['step'],
                     proposal=record['selected'],origin_step=record['step'],
                     verification=record['verification'],failure=failure)
        self.verification_events.append(event)
        self._emit(event)
        if failure is not None:
            self.active = None
            return True
        return False

    def act(self, observation, configuration=None):
        obs = _wire(observation)
        step, player = obs['step'], obs['player']
        if self.last_step is not None and (step != self.last_step+1 or player != self.player):
            raise ValueError('one actor requires consecutive steps in one seat; use a fresh instance per game')
        public = {k:copy.deepcopy(v) for k,v in (configuration or {}).items() if k in DEFAULT_CONFIGURATION}
        verification_failed = self._verify(obs)
        if self.active is not None and step > self.active['until']:
            self.active = None
        prior_memory = _wire(self.memory)
        ctx = StrategyContext(step,player,copy.deepcopy(obs),_wire(prior_memory),
                              tuple(obs.get('town',{}).get('unlocked_shops',())))
        pipeline_error, commitment_report = None, None
        try:
            if self.capture_commitments:
                from farm_neat.strategy.commitments import extract
                result = extract(ctx, self.base)  # own shadow, BEFORE its one act call
                ctx = result.publish(ctx)
                commitment_report = dict(status=result.status, scope_complete=result.cashflow['scope_complete'],
                    unresolved=list(result.unresolved), entries=len(result.entries),
                    provenance=copy.deepcopy(result.provenance))
            input_observation = _hash(_wire(ctx.observation))
            if self.pipeline is None:
                v = StepView(ctx,notes={'pipeline':'NOT_CONFIGURED; ALL_MODULES_ABSTAIN'})
            else:
                # Accepted StrategyPipeline.run takes a typed context; its
                # step convenience method takes an observation plus memory.
                call = getattr(self.pipeline, 'run', None)
                if call is None:
                    call = getattr(self.pipeline, 'step', self.pipeline)
                v = call(ctx)
            if (not isinstance(v,StepView) or v.context.step != step or v.context.player != player
                    or v.context.shops != ctx.shops or _hash(_wire(v.context.observation)) != input_observation
                    or _hash(_wire(ctx.observation)) != input_observation):
                raise ValueError('pipeline changed actor observation/seat/step/ordered shops')
            new_memory = _wire(v.context.memory)
            if new_memory.get('coordinator_history') != prior_memory.get('coordinator_history'):
                raise ValueError('pipeline must preserve coordinator_history own-memory binding')
            self.memory = new_memory
            if self.active is not None:
                d = Decision('ACTIVE_ADOPTION_LOCK',notes=dict(v.notes))
            elif verification_failed:
                d = Decision('REALIZATION_FAILED; NO_READOPTION_THIS_STEP',notes=dict(v.notes))
            elif self.adoption_attempts >= self.max_adoptions:
                d = Decision('ADOPTION_BUDGET_EXHAUSTED',notes=dict(v.notes))
            else:
                d = self.coordinator.decide(v)
        except Exception as error:
            pipeline_error = f'{type(error).__name__}: {error}'
            d = Decision('PIPELINE_OR_COORDINATOR_FAILED')
            self.memory = prior_memory
            v = StepView(StrategyContext(step,player,copy.deepcopy(obs),_wire(prior_memory),
                         tuple(obs.get('town',{}).get('unlocked_shops',()))))
            self.active = None
        record = d.to_dict()
        record.update(step=step,player=player,identity_sha256=self.identity['content_sha256'],
                      context_key=context_key(v.context),emission='FROZEN_K0006',
                      verification='NOT_REQUESTED',new_adoption=False)
        if commitment_report is not None:
            record['commitments'] = commitment_report
        if pipeline_error is None and hasattr(self.pipeline, 'last_report'):
            report = self.pipeline.last_report
            coverage = copy.deepcopy({k: report[k] for k in ('modules', 'events', 'missing_keys')})
            for module in coverage['modules'].values():
                module['executions'] = 0
            record['coverage'] = coverage
            record['value_conversions'] = copy.deepcopy(report.get('value_conversions', []))
            record['original_proposals'] = copy.deepcopy(report.get('original_proposals', []))
        branch, adoption, error = None, None, None
        if d.selected is not None:
            self.adoption_attempts += 1
            # Snapshot ONLY on selection (~4 MB for K0006); abstention never serializes it.
            # Adopters may fail after mutating their actor: the frozen shadow is never exposed.
            try:
                branch = restore_actor(snapshot_actor(self.base))
                adopter_context = StrategyContext(step,player,_wire(v.context.observation),
                    _wire(v.context.memory),v.context.shops,v.context.derived)
                adoption = self.adopter.adopt(branch,adopter_context,Proposal.from_dict(d.selected.to_dict()))
                if not isinstance(adoption,Adoption):
                    raise ProposalFailed('adapter must return the existing Adoption type')
                _wire(adoption.realization)
            except Exception as exception:
                error = f'{type(exception).__name__}: {exception}'
        # Always exactly one frozen call per actual observation, including successful changes.
        baseline_action = self.base.act(copy.deepcopy(obs),copy.deepcopy(public))
        action = baseline_action
        if pipeline_error is not None:
            record.update(emission='PIPELINE_FAILED',failure=pipeline_error)
        elif d.selected is not None:
            if error is None:
                try:
                    action = _action(adoption.first_action(branch,copy.deepcopy(obs),copy.deepcopy(public))
                                     if adoption.first_action is not None else branch.act(copy.deepcopy(obs),copy.deepcopy(public)))
                    record.update(emission='ADOPTED',new_adoption=True,realization=_wire(adoption.realization),
                                  selection_reason=d.reason)
                    if 'coverage' in record and d.selected.module in record['coverage']['modules']:
                        record['coverage']['modules'][d.selected.module]['executions'] += 1
                    self.adoptions += 1
                    until = d.selected.requirements.window[1] if d.selected.requirements.window else step
                    self.active = dict(actor=branch,proposal=d.selected,until=until,valuation=d.valuation)
                    if adoption.verify is not None:
                        record['verification'] = 'PENDING'
                        self.awaiting = (adoption,adopter_context,record)
                except Exception as exception:
                    error = f'{type(exception).__name__}: {exception}'
            if error is not None:
                action, self.active = baseline_action, None
                record.update(emission='PROPOSAL_FAILED',failure=error)
        elif self.active is not None:
            record.update(selected=self.active['proposal'].to_dict(),valuation=self.active['valuation'])
            try:
                action = _action(self.active['actor'].act(copy.deepcopy(obs),copy.deepcopy(public)))
                record['emission'] = 'ACTIVE_ADOPTION'
            except Exception as exception:
                action, self.active = baseline_action, None
                record.update(emission='PROPOSAL_FAILED',failure=f'{type(exception).__name__}: {exception}')
        record.update(action=copy.deepcopy(action),action_sha256=_hash(action),
                      baseline_action_sha256=_hash(baseline_action))
        # Observe only the action actually returned (including a failed adoption's
        # fallback), never the proposal or the frozen shadow's discarded action.
        remember = getattr(self.pipeline, 'remember', None)
        if pipeline_error is None and callable(remember):
            try:
                memory = _wire(remember(v, copy.deepcopy(action)))
                if memory.get('coordinator_history') != self.memory.get('coordinator_history'):
                    raise ValueError('remember must preserve coordinator_history')
                self.memory = memory
            except Exception as exception:
                # The action is already selected; a history failure is not a new
                # economic choice and must not silently rewrite its record.
                record['memory_failure'] = f'{type(exception).__name__}: {exception}'
        self.memory['coordinator_history'] = _hash(dict(previous=self.memory.get('coordinator_history'),
                                                       observation=obs,action=action))
        self.last_step, self.player = step, player
        self.records.append(record)
        self._emit(record)
        return action

    def coverage_receipt(self):
        """Aggregate actor-only composition coverage without inventing missing evidence.

        This is a review surface over records already produced by the live pipeline and
        coordinator. It does not evaluate proposals, inspect evaluator state, or change
        selection. Counts are additive across the consecutive prefix seen by this actor.
        """
        modules = {}
        events, missing_keys, missing_reasons = {}, {}, {}
        emissions, decisions, screening, conversions = {}, {}, {}, {}
        pipeline_failed_steps = 0

        def bump(table, key, amount=1):
            key = str(key)
            table[key] = table.get(key, 0) + amount

        for record in self.records:
            bump(emissions, record.get('emission', 'UNKNOWN_EMISSION'))
            bump(decisions, record.get('reason', 'UNKNOWN_DECISION'))
            if record.get('emission') == 'PIPELINE_FAILED':
                pipeline_failed_steps += 1
            for row in record.get('screening', ()):
                if isinstance(row, Mapping):
                    bump(screening, row.get('reason', 'UNKNOWN_SCREENING'))
            for row in record.get('value_conversions', ()):
                if isinstance(row, Mapping):
                    bump(conversions, row.get('reason', 'UNKNOWN_CONVERSION'))
            coverage = record.get('coverage')
            if not isinstance(coverage, Mapping):
                continue
            for name, row in coverage.get('modules', {}).items():
                if not isinstance(name, str) or not isinstance(row, Mapping):
                    continue
                target = modules.setdefault(name, dict(calls=0, proposals=0, comparable_values=0,
                                                       executions=0, unknown={}, abstentions={}))
                for key in ('calls', 'proposals', 'comparable_values', 'executions'):
                    value = row.get(key, 0)
                    if type(value) is int and value >= 0:
                        target[key] += value
                for key in ('unknown', 'abstentions'):
                    source = row.get(key, {})
                    if isinstance(source, Mapping):
                        for reason, count in source.items():
                            if type(count) is int and count >= 0:
                                bump(target[key], reason, count)
            event_rows = coverage.get('events', ())
            if isinstance(event_rows, Mapping):
                for name, count in event_rows.items():
                    if type(count) is int and count >= 0:
                        bump(events, name, count)
            elif isinstance(event_rows, (list, tuple)):
                for name in event_rows:
                    if isinstance(name, str):
                        bump(events, name)
            missing = coverage.get('missing_keys', {})
            if isinstance(missing, Mapping):
                for name, reason in missing.items():
                    bump(missing_keys, name)
                    if isinstance(reason, str):
                        bump(missing_reasons, reason)

        receipt = dict(
            schema=SCHEMA+'-coverage-v1',
            identity_sha256=self.identity['content_sha256'],
            baseline_sha256=BASELINE_SHA256,
            steps=len(self.records),
            modules={name: modules[name] for name in sorted(modules)},
            events={name: events[name] for name in sorted(events)},
            missing_keys={name: missing_keys[name] for name in sorted(missing_keys)},
            missing_reasons={name: missing_reasons[name] for name in sorted(missing_reasons)},
            emissions={name: emissions[name] for name in sorted(emissions)},
            decision_reasons={name: decisions[name] for name in sorted(decisions)},
            screening_reasons={name: screening[name] for name in sorted(screening)},
            conversion_reasons={name: conversions[name] for name in sorted(conversions)},
            adoptions=self.adoptions,
            adoption_attempts=self.adoption_attempts,
            pipeline_failed_steps=pipeline_failed_steps,
            input_scope='ACTOR_RECORDS_ONLY; EVALUATOR_PRIVATE_INPUTS_FORBIDDEN',
        )
        return bound(_wire(receipt))

    def finish(self, final_observation):
        """Optional terminal callback; without a final observation verification stays PENDING."""
        if self.last_step is None or final_observation.get('step') != self.last_step+1 or final_observation.get('player') != self.player:
            raise ValueError('final observation must follow the last emitted action in the same seat')
        self._verify(_wire(final_observation))
        self.active = None


def composed_policy():
    """Opt-in real composition with PARTIAL commitments and certified values only.

    Nonempty K0006Continuation adoption requires the accepted evaluator's
    native_realization monitor; without it that dependency fails closed. The
    current reached cuts abstain before adoption for named missing producers.
    Use the assignment's bounded replay to inspect this connectivity gate; a
    full shared-harness run is maintainer-only, not automatic on construction.
    """
    from farm_neat.strategy.pipeline import composed_pipeline
    from farm_neat.strategy.adoption import K0006Continuation, NativeLowerer
    return CoordinatorPolicy(pipeline=composed_pipeline(), capture_commitments=True,
                             adopter=K0006Continuation(lowerer=NativeLowerer()))


class JsonlRecorder:
    """Append-only game-side evidence; updates are separate events, never overwritten rows."""
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory,prefix='actor-',suffix='.jsonl',delete=False) as stream:
            self.path = Path(stream.name)
        self.previous, self.sequence = None, 0

    def __call__(self, record):
        envelope = bound(dict(sequence=self.sequence,previous_sha256=self.previous,event=record))
        with self.path.open('a') as stream:
            stream.write(json.dumps(envelope,sort_keys=True,allow_nan=False)+'\n')
        self.previous = envelope['content_sha256']
        self.sequence += 1


_FACTORY = None


def _load_factory(reference):
    if reference.count(':') != 1:
        raise ValueError('expected no-argument module:factory reference')
    module, name = reference.split(':')
    factory = getattr(importlib.import_module(module),name)
    if not callable(factory):
        raise ValueError('factory must be callable')
    return factory


def configured_policy():
    """Shared-harness no-argument factory, configured explicitly by this module's CLI."""
    if _FACTORY is None:
        raise ValueError('use CoordinatorPolicy for abstention or --panel to configure factories and records')
    pipeline, adopter, records = _FACTORY
    return CoordinatorPolicy(pipeline=None if pipeline is None else _load_factory(pipeline)(),
                             adopter=None if adopter is None else _load_factory(adopter)(),
                             record_sink=JsonlRecorder(records))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example',action='store_true')
    parser.add_argument('--panel',action='store_true')
    parser.add_argument('--pipeline',help='factory returning .step(context)->StepView')
    parser.add_argument('--adopter',help='factory returning .adopt(actor,context,proposal)->Adoption')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-games',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(argv)
    if args.panel:
        if args.output is None or bool(args.pipeline) != bool(args.adopter):
            parser.error('--panel needs --output and either BOTH --pipeline/--adopter or neither')
        # Under python -m, the harness imports the canonical module, not __main__.
        module = importlib.import_module('farm_neat.strategy.coordinator')
        module._FACTORY = (args.pipeline,args.adopter,args.output.with_name(args.output.name+'-actors'))
        from farm_neat.evaluation.router_panel import run_panel
        report = run_panel(args.output,candidate='farm_neat.strategy.coordinator:configured_policy',
                           parent='k0006',max_games=args.max_games,resume=args.resume)
        print(json.dumps(report,sort_keys=True,indent=2))
        return 0 if report['status']=='COMPLETE' else 1 if report['status']=='FAILED' else 3
    print(json.dumps(Coordinator().decide(StepView(StrategyContext(0,0,{'step':0,'player':0}))).to_dict(),indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
