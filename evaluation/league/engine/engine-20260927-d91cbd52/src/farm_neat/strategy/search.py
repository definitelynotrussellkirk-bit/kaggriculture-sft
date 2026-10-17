"""Bounded *offline* proposal-decision search and frozen actor-only selection rules.

Inputs: a pre-registered physical-world split, a current StepView, and exact
K0006 counterfactual cuts. Outputs: retained whole-menu trials (including losers
and UNKNOWN failures), then empirical rules over semantic proposal families.
The module owns selection, NOT proposal generation, economic sizing, capacity
proofs, adoption, restoration, or native game mechanics. Those owners stay intact.

search_menu is an arithmetic/fixture seam and ALWAYS marks its output SYNTHETIC.
search_snapshot uses the accepted exact counterfactual evaluator on registered
TRAIN worlds only. One unchanged control plus EVERY certified alternative must
fit the remaining budget before any branch starts. There is no arbitrary prefix
search, branch-and-bound based on unlike values, or hidden-world actor callback.
A failed/censored/unbound branch makes the whole menu incomplete, not a zero.

fit_rules groups by ordered published shops, seat, and the complete semantic menu
(module, producer family/kind, product, resource touches, guards, invalidators).
It searches decisions, not step numbers, quantities, thresholds or constants.
Ambiguous multiple sizes in one family abstain; the producer must disambiguate
its economic choices. Cases/seats are averaged WITHIN a physical world, then
worlds are equally weighted. The pre-registered minimum independent-world count
is required. Select unique positive paired mean WDL improvement first, then margin;
ties or a baseline winner keep K0006. Failed cases veto their entire menu cell.

These features are NOT proved sufficient for value. Rules are empirical,
context-guarded hypotheses, not exact guarantees or a global family priority list.
They transfer only when the same ordered shop prefix AND complete semantic menu
reappear and all current funding/capacity certificates re-verify. Own cash, rival
cash, and margin remain separate MEASURED values; Proposal estimates are NEVER
rewritten. No evaluator world/seed, snapshot, rival memory or future shop enters
an actor rule. Default selection abstains without trained evidence.

RuleSelector.decide(StepView) is the existing coordinator's injection seam: its
Decision supplies selected, valuation and to_dict(). Execution-time guards,
realization, and the continuously advanced frozen shadow remain coordinator/
adopter responsibilities. make_policy limits adoption attempts to ONE per game,
matching the one-change suffix estimand. It does not certify multi-change search.

CLI: register / empty / train / panel. Every native attempt is journaled BEFORE
execution, including failures. Outputs are exclusive; interrupted runs never
silently retry. panel reuses router_panel.play_game for paired complete games on
registered held-out worlds; it is not the historical 80-game harness gate and is
not ladder strength. --max-games includes BOTH roles, rivals and seats. Full
panels are maintainer-only. No games, installations or GPU work on import.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

from farm_neat.evaluation.k0006 import MEMBERS
from farm_neat.strategy.types import (EVALUATOR_ONLY, Evidence, Estimate, Proposal,
    StepView, StrategyContext, SuffixResult, UNKNOWN, Verdict)

SCHEMA = 'farm-strategy-search-v1'
BASELINE_SHA256 = MEMBERS['main.py']
FIELDS = ('own_cash_delta', 'rival_cash_delta', 'margin_delta')
FEATURE_SCHEMA = 'ordered-public-shops-and-complete-semantic-menu-v1'
ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN = EVALUATOR_ONLY | {'snapshot_sha256', 'environment', 'actors', 'source_pins'}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def seal(value):
    body = copy.deepcopy({k: v for k, v in value.items() if k != 'content_sha256'})
    return {**body, 'content_sha256': digest(body)}


def _verified(value, schema):
    if not isinstance(value, dict) or value.get('schema') != schema or seal(value) != value:
        raise ValueError('schema/hash binding mismatch: ' + schema)
    return value


def _actor_json(value):
    if isinstance(value, Mapping):
        if FORBIDDEN & set(value):
            raise ValueError('evaluator/private key in actor input')
        if any(not isinstance(k, str) for k in value):
            raise ValueError('actor JSON needs string keys')
        return {k: _actor_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_actor_json(v) for v in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('finite actor JSON required')


def _context_body(context):
    return _actor_json(dict(step=context.step, player=context.player,
        observation=context.observation, memory=context.memory, shops=context.shops))


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return value


def _source_pins():
    # Source identities are evaluator metadata, not actor observations.
    from farm_neat.strategy.counterfactual import source_pins
    pins = source_pins()
    paths = list((ROOT/'src/farm_neat/strategy').glob('*.py'))
    paths += [ROOT/'src/farm_neat/evaluation/router_panel.py',
              ROOT/'project/configs/scenarios.json']
    pins.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return pins


def register_split(*, train_ids=None, heldout_ids=None, min_worlds=2):
    """Register before search; default hash ordering splits the existing 72 worlds 36/36."""
    from farm_neat.evaluation.router_panel import world_registry
    registry = world_registry()
    if train_ids is None and heldout_ids is None:
        ids = [w.id for w in sorted(registry.worlds, key=lambda w: digest({'split': SCHEMA, 'world': w.sha256}))]
        train_ids, heldout_ids = ids[::2], ids[1::2]
    if not train_ids or not heldout_ids:
        raise ValueError('nonempty train and held-out worlds required')
    train_ids, heldout_ids = list(train_ids), list(heldout_ids)
    if len(set(train_ids)) != len(train_ids) or len(set(heldout_ids)) != len(heldout_ids):
        raise ValueError('duplicate world IDs')
    if set(train_ids) & set(heldout_ids):
        raise ValueError('train and held-out worlds must be disjoint')
    worlds = [registry[n] for n in train_ids+heldout_ids]
    if len({w.seed for w in worlds}) != len(worlds) or len({(w.seed,w.shops,w.random_weeds) for w in worlds}) != len(worlds):
        raise ValueError('physical worlds/seeds overlap')
    return seal(dict(schema=SCHEMA+'-registration', baseline_sha256=BASELINE_SHA256,
        registry_sha256=registry.sha256, train_ids=train_ids, heldout_ids=heldout_ids,
        world_hashes={w.id:w.sha256 for w in worlds}, min_worlds=_integer(min_worlds,'min_worlds',1,72),
        feature_schema=FEATURE_SCHEMA, source_pins=_source_pins(),
        objective='PAIRED_MEAN_WDL_THEN_MARGIN; EQUAL_WORLD_WEIGHTS; TIES_KEEP',
        scope='REGISTERED_EXPOSED_FIXED_WORLDS; HELD_OUT_FROM_THIS_SEARCH_ONLY'))


def validate_registration(registration):
    from farm_neat.evaluation.router_panel import world_registry
    r = _verified(registration, SCHEMA+'-registration')
    registry = world_registry()
    if r != register_split(train_ids=r['train_ids'], heldout_ids=r['heldout_ids'], min_worlds=r['min_worlds']):
        raise ValueError('registration source/physical-world binding changed')
    if r['registry_sha256'] != registry.sha256:
        raise ValueError('registered worlds changed')
    return registry


def _choice(p):
    raw = p.candidate
    # Production's actor-recomputed snapshot hash is verified by value_basis on conversion;
    # the choice key never reads it, so drop only that field and keep every other ban.
    if (p.module == 'production' and isinstance(raw, Mapping)
            and raw.get('schema') == 'farm-production-proposal-v1'):
        raw = {k: v for k, v in raw.items() if k != 'snapshot_sha256'}
    candidate = _actor_json(raw)
    if not isinstance(candidate, dict):
        return None
    family = candidate.get('family', candidate.get('kind'))
    if not isinstance(family, str) or not family.strip():
        return None
    # Exact guard text is deliberately not generalized or evaluated as Python.
    return dict(module=p.module, family=family,
        kind=candidate.get('kind'), schema=candidate.get('schema'), product=candidate.get('product'),
        touches=list(p.touches), guards=list(p.guards), invalidated_by=list(p.invalidated_by))


def _bound(estimate, side):
    if estimate.kind is Evidence.EXACT:
        return estimate.value
    if estimate.kind is Evidence.BOUND:
        return getattr(estimate, side)
    return None


def screen(view: StepView, *, verifier=None, max_proposals=128):
    """Re-use the accepted capacity checker; certificate constructors prove nothing."""
    if not isinstance(view, StepView):
        raise TypeError('StepView required')
    _context_body(view.context)
    if len(view.proposals) > _integer(max_proposals, 'max_proposals', 1, 1024):
        return (), [{'reason':'PROPOSAL_BUDGET_EXHAUSTED'}]
    if verifier is None:
        from farm_neat.strategy.capacity import verify_feasibility
        verifier = verify_feasibility
    # This selector replaces Coordinator in live play. Reuse the accepted funding
    # owner verbatim rather than maintaining a weaker copy of its safety gate.
    from farm_neat.strategy.coordinator import _funding as coordinator_funding, resource_conflicts
    funding, protected_resources = coordinator_funding(view)
    safe = funding['safe_discretionary']
    funding_reason = None if safe is not None else funding['reason']
    menu, audit = [], []
    for p in sorted(view.proposals, key=lambda p:p.key):
        reason = 'ELIGIBLE'
        family = _choice(p)
        f = view.feasibility.get(p.key)
        a,b = _bound(p.requirements.cash,'high'), _bound(p.requirements.reserve,'high')
        if family is None:
            reason = 'NO_SEMANTIC_FAMILY'
        elif f is None or f.verdict is not Verdict.FEASIBLE:
            reason = 'UNVERIFIED_FEASIBILITY'
        elif p.requirements.window is not None and not p.requirements.window[0] <= view.context.step <= p.requirements.window[1]:
            reason = 'OUTSIDE_EXECUTION_WINDOW'
        elif resource_conflicts(protected_resources, p.touches):
            reason = 'PROTECTED_RESOURCE_REQUIRES_REPLACEMENT_CERTIFICATE'
        elif funding_reason is not None:
            reason = funding_reason
        elif a is None or b is None or min(a,b) < 0:
            reason = 'UNKNOWN_OR_INVALID_CAPITAL'
        elif a+b > safe:
            reason = 'CAPITAL_EXCEEDS_SAFE_DISCRETIONARY'
        elif not verifier(view.context,p,f):
            reason = 'STALE_OR_UNVERIFIED_CAPACITY_CERTIFICATE'
        else:
            menu.append(p)
        audit.append(dict(proposal=list(p.key),reason=reason))
    return tuple(menu), audit


def _cell(view, menu):
    choices = sorted((_choice(p) for p in menu), key=_json)
    if len({digest(c) for c in choices}) != len(choices):
        return None
    return dict(feature_schema=FEATURE_SCHEMA, player=view.context.player,
        ordered_shops=list(view.context.shops), choices=choices)


def _validate_cell(cell, *, seat=None):
    if (not isinstance(cell, dict)
            or set(cell) != {'feature_schema','player','ordered_shops','choices'}
            or cell.get('feature_schema') != FEATURE_SCHEMA
            or type(cell.get('player')) is not int or cell['player'] not in (0,1)
            or (seat is not None and cell['player'] != seat)
            or not isinstance(cell.get('ordered_shops'), list)
            or any(not isinstance(x,str) or not x for x in cell['ordered_shops'])
            or not isinstance(cell.get('choices'), list)
            or any(not isinstance(x,dict) for x in cell['choices'])
            or cell['choices'] != sorted(cell['choices'], key=_json)
            or len({digest(x) for x in cell['choices']}) != len(cell['choices'])):
        raise ValueError('case/rule cell/seat/feature binding mismatch')
    return cell


def _validate_rule_model(model):
    rules = model.get('rules')
    if not isinstance(rules,list):
        raise ValueError('rules must be a list')
    seen=set()
    for rule in rules:
        if (not isinstance(rule,dict)
                or set(rule) != {'cell','cell_sha256','choice','evidence','alternatives','guards'}):
            raise ValueError('closed empirical rule fields required')
        cell=_validate_cell(rule['cell'])
        cell_hash=digest(cell)
        if rule['cell_sha256'] != cell_hash or cell_hash in seen:
            raise ValueError('rule cell hash/uniqueness binding mismatch')
        seen.add(cell_hash)
        alternatives=rule['alternatives']
        if (not isinstance(alternatives,list) or not alternatives
                or any(not isinstance(a,dict) or set(a)!={'choice','evidence'} for a in alternatives)):
            raise ValueError('rule alternatives/evidence binding mismatch')
        alt_choices=[a['choice'] for a in alternatives]
        if alt_choices != cell['choices']:
            raise ValueError('rule alternatives must exactly bind the semantic menu')
        matches=[a for a in alternatives if a['choice']==rule['choice']]
        if len(matches)!=1 or matches[0]['evidence'] != rule['evidence']:
            raise ValueError('rule choice/evidence binding mismatch')
        try:
            scores=[(0.,0.),*[(a['evidence']['paired_wdl_mean'],
                               Estimate.from_dict(a['evidence']['margin_delta']).value)
                              for a in alternatives]]
            if any(type(x) not in (int,float) or not math.isfinite(x) for score in scores for x in score):
                raise ValueError
        except (KeyError,TypeError,ValueError):
            raise ValueError('rule evidence score binding mismatch') from None
        best=max(scores); winners=[i for i,x in enumerate(scores) if x==best]
        selected=alternatives.index(matches[0])+1
        if len(winners)!=1 or winners[0]!=selected:
            raise ValueError('rule choice is not the unique stored-evidence winner')


@dataclass
class SearchBudget:
    max_evaluations: int = 4
    max_proposals: int = 128
    max_suffix_steps: int = 719
    seconds: float = 120.
    evaluations: int = field(default=0, init=False)
    started: float = field(default_factory=time.monotonic, init=False, repr=False)

    def __post_init__(self):
        _integer(self.max_evaluations,'max_evaluations',0,100000)
        _integer(self.max_proposals,'max_proposals',1,1024)
        _integer(self.max_suffix_steps,'max_suffix_steps',1,719)
        if type(self.seconds) not in (int,float) or not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError('positive finite cooperative seconds budget required')

    @property
    def remaining_seconds(self):
        return max(0., self.seconds-(time.monotonic()-self.started))

    def permits(self, count, start_step):
        return (self.evaluations+count <= self.max_evaluations
                and 719-start_step <= self.max_suffix_steps and self.remaining_seconds > 0)


def _result_dict(result):
    return asdict(result)


def _outcome(margin):
    return 1 if margin > 0 else -1 if margin < 0 else 0


def _paired(control, child):
    if control is None or child is None or control.status != 'COMPLETE' or child.status != 'COMPLETE':
        return {k:UNKNOWN.to_dict() for k in FIELDS}, None
    values = (child.own_cash-control.own_cash, child.rival_cash-control.rival_cash, child.margin-control.margin)
    return dict(zip(FIELDS,(Estimate.measured(v).to_dict() for v in values))), _outcome(child.margin)-_outcome(control.margin)


def _search(registration, world_id, view, evaluator, *, snapshot_sha256, budget, verifier,
            evidence_kind, event_sink=None):
    validate_registration(registration)
    if world_id not in registration['train_ids']:
        raise ValueError('only registered train worlds may enter search; held-out worlds are forbidden')
    if not isinstance(snapshot_sha256,str) or len(snapshot_sha256)!=64 or any(c not in '0123456789abcdef' for c in snapshot_sha256):
        raise ValueError('source-bound snapshot SHA256 required')
    menu, audit = screen(view,verifier=verifier,max_proposals=budget.max_proposals)
    cell = _cell(view, menu)
    row = dict(schema=SCHEMA+'-case', registration_sha256=registration['content_sha256'],
        world_id=world_id, world_sha256=registration['world_hashes'][world_id],
        snapshot_sha256=snapshot_sha256, context_sha256=digest(_context_body(view.context)),
        seat=view.context.player, start_step=view.context.step, cell=cell,
        screening=audit, evidence_kind=evidence_kind, control=None, alternatives=[],
        selected=None, status='NO_CERTIFIED_PROPOSALS', attempts=0)
    if cell is None:
        row['status']='AMBIGUOUS_SEMANTIC_MENU'
    elif not menu:
        pass
    elif not budget.permits(len(menu)+1,view.context.step):
        row['status']='BUDGET_EXHAUSTED'
    else:
        results=[]
        for p in (None,*menu):
            expected = None if p is None else p.key
            result = None
            if not budget.permits(1,view.context.step):
                result = SuffixResult(expected,'ERROR',failure='cooperative search budget exhausted')
            else:
                budget.evaluations += 1
                row['attempts'] += 1
                if event_sink:
                    event_sink(dict(event='START',world_id=world_id,snapshot_sha256=snapshot_sha256,
                                    proposal=expected,attempt=budget.evaluations))
                try:
                    branch = None if p is None else Proposal.from_dict(p.to_dict())
                    before = None if branch is None else digest(branch.to_dict())
                    result = evaluator(branch)
                    if budget.remaining_seconds <= 0:
                        raise TimeoutError("cooperative search deadline exceeded by evaluator")
                    if not isinstance(result,SuffixResult) or result.proposal != expected:
                        raise ValueError('wrong proposal in suffix result')
                    if branch is not None and digest(branch.to_dict()) != before:
                        raise ValueError('evaluator mutated its proposal input')
                    if result.status == 'COMPLETE':
                        r = result.realization or {}
                        fields = dict(snapshot_sha256=snapshot_sha256,seat=view.context.player,
                                      start_step=view.context.step,end_step=719)
                        if any(r.get(k)!=v for k,v in fields.items()) or not isinstance(r.get('trace_sha256'),str) or len(r['trace_sha256'])!=64:
                            raise ValueError('unbound/incomplete terminal suffix')
                except Exception as error:
                    result = SuffixResult(expected,'ERROR',failure=f'{type(error).__name__}: {error}')
                if event_sink:
                    event_sink(dict(event='RESULT',attempt=budget.evaluations,result=_result_dict(result)))
            results.append(result)
        control = results[0]
        row['control'] = _result_dict(control)
        for p,result in zip(menu,results[1:]):
            deltas,wdl = _paired(control,result)
            row['alternatives'].append(dict(proposal=p.to_dict(),choice=_choice(p),
                result=_result_dict(result),deltas=deltas,paired_wdl=wdl))
        if any(r.status!='COMPLETE' for r in results):
            row['status']='INCOMPLETE_MENU'
        else:
            row['status']='COMPLETE'
            scores=[(_outcome(r.margin),r.margin) for r in results]
            best=max(scores)
            winners=[i for i,score in enumerate(scores) if score==best]
            if len(winners)==1 and winners[0]!=0:
                row['selected']=list(menu[winners[0]-1].key)
    return seal(row)


def search_menu(registration, world_id, view, evaluator, *, snapshot_sha256, budget=None, verifier=None):
    """Injectable deterministic fixture evaluator, NEVER native playing-strength evidence."""
    return _search(registration,world_id,view,evaluator,snapshot_sha256=snapshot_sha256,
        budget=budget or SearchBudget(),verifier=verifier,evidence_kind='SYNTHETIC_DIAGNOSTIC')


def search_snapshot(registration, world_id, snapshot, view, *, budget=None, continuation_factory=None, event_sink=None):
    """Offline native search; every branch restores the SAME exact K0006 continuation."""
    from farm_neat.strategy import counterfactual as cf
    from farm_neat.evaluation.k0006 import K0006Adapter
    registry = validate_registration(registration)
    if world_id not in registration['train_ids']:
        raise ValueError('registered train cut required, never held-out')
    runtime = cf.restore(snapshot)
    env,seat = runtime.environment,runtime.seat
    if (getattr(env,'scenario',None)!=registry[world_id].scenario()
            or type(runtime.actors[seat]) is not K0006Adapter
            or type(runtime.actors[1-seat]) is not K0006Adapter):
        raise ValueError('registered scenario and frozen K0006 in BOTH seats required')
    actual = cf.actor_context(env.agent_observation(seat), runtime.actors[seat])
    if (view.context.player!=seat or view.context.step!=env.step_index
            or _actor_json(view.context.observation)!=_actor_json(actual.observation)
            or view.context.shops!=actual.shops):
        raise ValueError('StepView observation/seat/ordered-shops/cut mismatch')
    budget = budget or SearchBudget()
    def evaluate(p):
        if continuation_factory is None:
            from farm_neat.strategy.adoption import evaluate as native_evaluate
            return native_evaluate(snapshot,p,max_steps=budget.max_suffix_steps,
                                   seconds=max(1e-6,budget.remaining_seconds))
        return cf.evaluate(snapshot,p,None if p is None else continuation_factory(),
                           max_steps=budget.max_suffix_steps,seconds=max(1e-6,budget.remaining_seconds))
    return _search(registration,world_id,view,evaluate,snapshot_sha256=snapshot.sha256,budget=budget,
        verifier=None,evidence_kind='NATIVE_SUFFIX',event_sink=event_sink)


def _mean(values):
    return statistics.fmean(values) if values else None


def fit_rules(registration, rows, *, allow_synthetic=False):
    """Freeze empirical rules; only complete, common-menu TRAIN cases may supply values."""
    validate_registration(registration)
    rows = list(rows)
    groups,seen={},set()
    kinds=set()
    for row in rows:
        _verified(row,SCHEMA+'-case')
        if row['registration_sha256']!=registration['content_sha256'] or row['world_id'] not in registration['train_ids']:
            raise ValueError('unregistered or held-out row cannot enter train fitting')
        if row['world_sha256']!=registration['world_hashes'][row['world_id']]:
            raise ValueError('world binding mismatch')
        _validate_case_results(row)
        key=(row['world_id'],row['snapshot_sha256'],row['seat'],row['context_sha256'])
        if key in seen:
            raise ValueError('duplicate training cut')
        seen.add(key)
        kinds.add(row['evidence_kind'])
        if row['evidence_kind']!='NATIVE_SUFFIX' and not allow_synthetic:
            raise ValueError('synthetic diagnostic rows cannot train a native policy')
        if row['evidence_kind'] not in ('NATIVE_SUFFIX','SYNTHETIC_DIAGNOSTIC'):
            raise ValueError('unknown evidence kind')
        if row['cell'] is not None and row['cell']['choices']:
            groups.setdefault(digest(row['cell']),[]).append(row)
    rules=[]; rejected=[]
    for cell_hash,cases in sorted(groups.items()):
        worlds={r['world_id'] for r in cases}
        if any(r['status']!='COMPLETE' for r in cases):
            rejected.append(dict(cell_sha256=cell_hash,reason='INCOMPLETE_MENU_VETO'))
            continue
        if len(worlds)<registration['min_worlds']:
            rejected.append(dict(cell_sha256=cell_hash,reason='INSUFFICIENT_INDEPENDENT_WORLDS'))
            continue
        alternatives=[]
        for choice in cases[0]['cell']['choices']:
            by_world={w:[] for w in worlds}
            for row in cases:
                matches=[a for a in row['alternatives'] if a['choice']==choice]
                if len(matches)!=1:
                    raise ValueError('complete semantic menu does not match trial alternatives')
                a=matches[0]
                # Recompute from terminal receipts, never trust caller-provided rank/deltas.
                control=_decode_result(row['control']); child=_decode_result(a['result'])
                deltas,wdl=_paired(control,child)
                if deltas!=a['deltas'] or wdl!=a['paired_wdl']:
                    raise ValueError('derived trial values disagree with terminal results')
                by_world[row['world_id']].append([wdl,*[deltas[f]['value'] for f in FIELDS]])
            means=[[_mean([x[i] for x in values]) for i in range(4)] for values in by_world.values()]
            mean=[_mean([x[i] for x in means]) for i in range(4)]
            evidence=dict(independent_worlds=len(worlds),cuts=len(cases),paired_wdl_mean=mean[0],
                **{f:Estimate.measured(mean[i+1]).to_dict() for i,f in enumerate(FIELDS)},
                minimum_world_mean_margin=min(x[3] for x in means),
                comparison='ONE_DEVIATION_VS_SAME_CUT_FROZEN_K0006; EMPIRICAL_NOT_A_BOUND',
                case_receipt_hashes=sorted(r['content_sha256'] for r in cases))
            alternatives.append(dict(choice=choice,evidence=evidence))
        scores=[(0.,0.),*[(a['evidence']['paired_wdl_mean'],a['evidence']['margin_delta']['value']) for a in alternatives]]
        best=max(scores); winners=[i for i,x in enumerate(scores) if x==best]
        if len(winners)!=1 or winners[0]==0:
            rejected.append(dict(cell_sha256=cell_hash,reason='KEEP_OR_TIED',alternatives=alternatives))
            continue
        selected=alternatives[winners[0]-1]
        rules.append(dict(cell=cases[0]['cell'],cell_sha256=cell_hash,**selected,
                          alternatives=alternatives,guards=['CURRENT_CAPACITY_REVERIFIED',
                          'CURRENT_COMMITMENTS_FUNDED','EXACT_ORDERED_SHOPS_AND_FULL_SEMANTIC_MENU',
                          'PRODUCER_AND_EXECUTION_GUARDS_STILL_HOLD','AT_MOST_ONE_ADOPTION_PER_GAME']))
    kind='SYNTHETIC_DIAGNOSTIC' if 'SYNTHETIC_DIAGNOSTIC' in kinds else 'NATIVE_SUFFIX'
    return seal(dict(schema=SCHEMA+'-rules',registration_sha256=registration['content_sha256'],
        baseline_sha256=BASELINE_SHA256,feature_schema=FEATURE_SCHEMA,
        evidence_kind=kind,future_value=UNKNOWN.to_dict(),rules=rules,rejected_cells=rejected,
        status='NO_TRAINING_EVIDENCE' if not rows else 'FROZEN_EMPIRICAL_RULES' if rules else 'NO_SUPPORTED_RULE',
        training_case_count=len(rows),source_pins_digest=digest(registration['source_pins'])))


def _validate_case_results(row):
    """A content hash is not a same-cut, same-proposal, terminal equality check."""
    cell = row['cell']
    if cell is not None:
        _validate_cell(cell,seat=row['seat'])
    controls = [] if row['control'] is None else [(None, row['control'])]
    choices = []
    for alternative in row['alternatives']:
        p = Proposal.from_dict(alternative['proposal'])
        if alternative['choice'] != _choice(p):
            raise ValueError('trial semantic choice/proposal binding mismatch')
        choices.append(alternative['choice'])
        controls.append((p.key, alternative['result']))
    if row['status'] == 'COMPLETE':
        if row['control'] is None or not row['alternatives'] or row['cell'] is None:
            raise ValueError('complete menu needs control and every proposal suffix')
        if sorted(choices, key=_json) != row['cell']['choices']:
            raise ValueError('complete trial menu/proposal binding mismatch')
        if len({digest(c) for c in choices}) != len(choices):
            raise ValueError('duplicate semantic proposal')
    for expected, data in controls:
        result = _decode_result(data)
        if result.proposal != expected:
            raise ValueError('result proposal binding mismatch')
        if row['status'] == 'COMPLETE' and result.status != 'COMPLETE':
            raise ValueError('incomplete suffix labelled complete')
        if result.status == 'COMPLETE':
            binding = dict(snapshot_sha256=row['snapshot_sha256'], seat=row['seat'],
                           start_step=row['start_step'], end_step=719)
            realization = result.realization
            trace = realization.get('trace_sha256') if isinstance(realization, dict) else None
            if (not isinstance(realization, dict)
                    or any(realization.get(k) != v for k,v in binding.items())
                    or not isinstance(trace,str) or len(trace)!=64
                    or any(c not in '0123456789abcdef' for c in trace)):
                raise ValueError('same-cut terminal suffix/trace binding mismatch')


def _decode_result(row):
    return SuffixResult(None if row['proposal'] is None else tuple(row['proposal']),row['status'],
        row['own_cash'],row['rival_cash'],row['failure'],row['realization'])


@dataclass
class SearchDecision:
    reason: str
    selected: Proposal | None = None
    valuation: dict = field(default_factory=dict)
    screening: list = field(default_factory=list)

    def to_dict(self):
        return dict(schema=SCHEMA+'-decision',reason=self.reason,
                    selected=None if self.selected is None else self.selected.to_dict(),
                    valuation=self.valuation,screening=self.screening)


class RuleSelector:
    """Coordinator injection seam; never invokes an evaluator or reads a registry."""
    def __init__(self, model, *, allow_synthetic=False):
        _verified(model,SCHEMA+'-rules')
        if model['baseline_sha256']!=BASELINE_SHA256 or model['feature_schema']!=FEATURE_SCHEMA:
            raise ValueError('policy baseline/feature binding mismatch')
        if model['evidence_kind']!='NATIVE_SUFFIX' and not allow_synthetic:
            raise ValueError('synthetic diagnostic policy cannot enter native play')
        _validate_rule_model(model)
        self.model=_actor_json(copy.deepcopy(model))

    def decide(self,view,*,verifier=None):
        menu,audit=screen(view,verifier=verifier)
        if not menu:
            return SearchDecision('NO_PROPOSALS' if not view.proposals else 'NO_CERTIFIED_PROPOSALS',screening=audit)
        cell=_cell(view,menu)
        if cell is None:
            return SearchDecision('AMBIGUOUS_SEMANTIC_MENU',screening=audit)
        matches=[r for r in self.model['rules'] if r['cell']==cell]
        if len(matches)!=1:
            return SearchDecision('NO_SUPPORTED_RULE',screening=audit)
        rule=matches[0]
        candidates=[p for p in menu if _choice(p)==rule['choice']]
        if len(candidates)!=1:
            return SearchDecision('AMBIGUOUS_RULE',screening=audit)
        return SearchDecision('EMPIRICAL_TRAIN_RULE_NOT_A_VALUE_CERTIFICATE',candidates[0],
            dict(rule_sha256=digest(rule),model_sha256=self.model['content_sha256'],
                 future_value=UNKNOWN.to_dict(),**rule['evidence']),audit)


def summarize_holdout(rows):
    """Paired role deltas, complete-game W/D/L, and independent-world counts; no imputation."""
    groups={}; failed=0
    for row in rows:
        key=(row['world'],row['seat'],row['opponent'])
        roles=groups.setdefault(key,{})
        role=row['role']
        if role not in ('parent','candidate') or role in roles:
            raise ValueError('invalid or duplicate paired role')
        roles[role]=row
        failed+=row['status']!='COMPLETE'
    pairs=[]; counts={r:{'W':0,'D':0,'L':0} for r in ('parent','candidate')}
    for (world,seat,rival),roles in groups.items():
        for role,row in roles.items():
            if row['status']=='COMPLETE':
                cash=row['cash']
                if row['steps']!=719 or len(cash)!=2 or any(type(c) not in (int,float) or not math.isfinite(c) for c in cash):
                    raise ValueError('finite complete native game result required')
                counts[role][('D','W','L')[_outcome(cash[seat]-cash[1-seat])]]+=1
        if set(roles)!={'parent','candidate'} or any(r['status']!='COMPLETE' for r in roles.values()):
            continue
        p,c=roles['parent']['cash'],roles['candidate']['cash']
        own,rival_delta=c[seat]-p[seat],c[1-seat]-p[1-seat]
        pairs.append(dict(world=world,seat=seat,opponent=rival,own_cash_delta=own,
                          rival_cash_delta=rival_delta,margin_delta=own-rival_delta,
                          action_hash_equal=roles['parent']['actions_sha256']==roles['candidate']['actions_sha256']))
    world_means={w:_mean([p['margin_delta'] for p in pairs if p['world']==w]) for w in {p['world'] for p in pairs}}
    return dict(paired_count=len(pairs),independent_worlds=len(world_means),
        attempted_games=len(rows),failed_attempts=failed,unpaired_cells=len(groups)-len(pairs),
        paired_margin_mean=_mean([p['margin_delta'] for p in pairs]),
        equal_world_mean_margin=_mean(list(world_means.values())),
        own_cash_delta_mean=_mean([p['own_cash_delta'] for p in pairs]),
        rival_cash_delta_mean=_mean([p['rival_cash_delta'] for p in pairs]),
        candidate_wdl=counts['candidate'],parent_wdl=counts['parent'],pairs=pairs,
        confidence_interval=None,interval_note='NO_INTERVAL_CLAIM; COUNTS_AND_WORLD_MEANS_ONLY')


class AttemptJournal:
    """Append+flush+fsync before native work. New files only; no automatic retry/resume."""
    def __init__(self,path):
        self.file=Path(path).open('x',encoding='utf-8')
        self.previous=None
        self.index=0

    def append(self,event):
        record=seal(dict(schema=SCHEMA+'-journal',index=self.index,
                         previous_sha256=self.previous,**copy.deepcopy(event)))
        self.file.write(_json(record)+'\n'); self.file.flush(); os.fsync(self.file.fileno())
        self.previous=record['content_sha256']; self.index+=1
        return record

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.file.close()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        f.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())


def _factory(ref):
    if not isinstance(ref,str) or ref.count(':')!=1:
        raise ValueError('factory must be module:callable')
    module,name=ref.split(':')
    factory=getattr(importlib.import_module(module),name)
    if not callable(factory):
        raise ValueError('callable factory required')
    return factory


class _PipelineBridge:
    """Only the accepted pipeline's context-input spelling is adapted here."""
    def __init__(self,reference):
        import inspect
        constructor=_factory(reference)
        path=inspect.getsourcefile(constructor)
        if path is None:
            raise ValueError('pipeline factory must have inspectable source')
        self.inner=constructor()
        self.identity=dict(factory=reference,assets={str(Path(path).resolve()):
            hashlib.sha256(Path(path).read_bytes()).hexdigest()})

    def step(self,context):
        call=getattr(self.inner,'run',getattr(self.inner,'step',self.inner))
        return call(context)


def make_policy(model, *, pipeline_factory='farm_neat.strategy.pipeline:StrategyPipeline', adopter_factory=None):
    """Build an existing CoordinatorPolicy with this selector; actor inputs stay public."""
    from farm_neat.strategy.coordinator import CoordinatorPolicy
    from farm_neat.evaluation.common import bound
    selector=RuleSelector(model)
    if model['rules'] and adopter_factory is None:
        raise ValueError('nonempty empirical rules require an explicit guarded coordinator-compatible live adopter')
    if model['source_pins_digest']!=digest(_source_pins()):
        raise ValueError('trained policy source pins changed; re-register and re-evaluate')
    pipeline=None if pipeline_factory is None else _PipelineBridge(pipeline_factory)
    adopter=None if adopter_factory is None else _factory(adopter_factory)()
    actor=CoordinatorPolicy(pipeline=pipeline,adopter=adopter,coordinator=selector,max_adoptions=1)
    identity=dict(actor.identity)
    identity['objective']='EMPIRICAL_PAIRED_WDL_THEN_MARGIN; FUTURE_VALUES_UNKNOWN'
    identity['search_policy_sha256']=model['content_sha256']
    identity['search_policy_status']=model['status']
    assets=dict(identity['assets'])
    if pipeline is not None:
        assets.update(pipeline.identity['assets'])
    if adopter_factory is not None:
        import inspect
        path=inspect.getsourcefile(_factory(adopter_factory))
        if path is None:
            raise ValueError('adopter factory must have inspectable source')
        assets[str(Path(path).resolve())]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    identity['assets']=assets
    actor.identity=bound(identity)
    return actor


def configured_policy():
    """No-argument factory for the unmodified shared router_panel full-panel command.

    FARMNEAT_SEARCH_MODEL points at the frozen rules JSON. Optional
    FARMNEAT_SEARCH_PIPELINE and FARMNEAT_SEARCH_ADOPTER name trusted factories.
    Those files and configuration must remain fixed throughout a panel.
    """
    path=os.environ.get('FARMNEAT_SEARCH_MODEL')
    if not path:
        raise ValueError('FARMNEAT_SEARCH_MODEL must name a frozen rules JSON')
    return make_policy(_read(path),
        pipeline_factory=os.environ.get('FARMNEAT_SEARCH_PIPELINE','farm_neat.strategy.pipeline:StrategyPipeline'),
        adopter_factory=os.environ.get('FARMNEAT_SEARCH_ADOPTER'))


def run_holdout(registration, model, output, *, max_games=4, rivals=('k0006',), timeout=120.,
                pipeline_factory='farm_neat.strategy.pipeline:StrategyPipeline', adopter_factory=None):
    """Read-only held-out evaluation. It never fits/edits rules or selects a new policy."""
    from farm_neat.evaluation import router_panel as panel
    registry=validate_registration(registration)
    if panel.play_game is not panel._NATIVE_PLAY_GAME:
        raise ValueError('native panel runner was substituted; diagnostic results are not native evidence')
    RuleSelector(model)
    if model['registration_sha256']!=registration['content_sha256']:
        raise ValueError('policy/registration binding mismatch')
    _integer(max_games,'max_games',0,100000)
    if len(set(rivals))!=len(rivals) or not rivals or any(r not in ('k0006','router','tape0','pass') for r in rivals):
        raise ValueError('distinct registered frozen rival names required')
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or not 0<timeout<=300:
        raise ValueError('timeout must be in (0,300] seconds')
    # A complete world/rival comparison has both roles in both seats. Do not
    # silently turn a three-game allowance into a biased incomplete statistic.
    if max_games%4:
        raise ValueError('max_games must fund whole four-game paired world/rival cells')
    def candidate_factory():
        return make_policy(model,pipeline_factory=pipeline_factory,adopter_factory=adopter_factory)
    specs={'candidate':panel.register_actor('strategy_search',candidate_factory),
           'parent':panel.load_actor('k0006')}
    opponents={r:panel.load_actor(r) for r in rivals}
    jobs=[dict(world=w,opponent=r,seat=s,role=role)
          for w in registration['heldout_ids'] for r in rivals for s in (0,1)
          for role in ('parent','candidate')]
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    _write(output/'registration.json',registration)
    _write(output/'rules.json',model)
    _write(output/'plan.json',seal(dict(schema=SCHEMA+'-panel-plan',
        registration_sha256=registration['content_sha256'],policy_sha256=model['content_sha256'],
        jobs=jobs,execution=panel.execution_binding(panel.play_game),
        actors={k:v.identity for k,v in specs.items()},rivals={k:v.identity for k,v in opponents.items()},
        max_games=max_games,timeout_seconds=timeout,source_pins=_source_pins(),
        training_state=model['status'],scope='HELD_OUT_FROM_THIS_SEARCH; NOT_HISTORICAL_GATE_OR_LADDER')))
    rows=[]
    with AttemptJournal(output/'attempts.jsonl') as journal:
        for i,job in enumerate(jobs[:max_games]):
            journal.append(dict(event='START',attempt=i+1,job=job))
            row=panel.play_game(registry[job['world']],specs[job['role']],opponents[job['opponent']],job['seat'],timeout)
            row['role']=job['role']
            # Retain actual coordinator emissions/guard failures separately from W/D/L.
            if job['role']=='candidate':
                actor=specs['candidate']._last
                records=getattr(actor,'records',())
                row['selection_counts']={reason:sum(x['reason']==reason for x in records)
                    for reason in sorted({x['reason'] for x in records})}
                row['adoption_attempts']=getattr(actor,'adoption_attempts',None)
                row['adoptions']=getattr(actor,'adoptions',None)
                failures=[x for x in records if x.get('failure')]
                row['actor_failures']=[dict(step=x['step'],reason=x['reason'],failure=x['failure']) for x in failures]
            journal.append(dict(event='RESULT',attempt=i+1,job=job,result=row))
            rows.append(row)
    report=seal(dict(schema=SCHEMA+'-heldout',registration_sha256=registration['content_sha256'],
        policy_sha256=model['content_sha256'],training_state=model['status'],
        evidence_kind='NATIVE_GAME',games_planned=len(jobs),games_not_run=max(0,len(jobs)-len(rows)),
        result=summarize_holdout(rows),records=rows,
        interpretation='EVALUATION_ONLY; EMPTY_RULES_ARE_FALLBACK_CONTROL_NOT_LEARNED_STRENGTH'))
    _write(output/'summary.json',report)
    return report


def train_cuts(registration,cases,output,*,pipeline_factory,budget,continuation_factory=None):
    """Consume source-bound K0006 cuts; the pipeline gets only public context+own memory.

    cases: {schema: farm-strategy-search-v1-cuts, registration_sha256, cuts:[
      {world_id, snapshot_path, snapshot_sha256, memory: <actor-owned JSON, optional>}]}
    Each snapshot is the accepted counterfactual.Snapshot JSON. Prefix generation
    and independent evidence of actor memory/producer attestations belong to its
    producer, NOT this routine. Input paths never reach the actor-facing pipeline.
    """
    from farm_neat.strategy import counterfactual as cf
    validate_registration(registration)
    if (set(cases)!={'schema','registration_sha256','cuts'} or cases['schema']!=SCHEMA+'-cuts'
            or cases['registration_sha256']!=registration['content_sha256']):
        raise ValueError('closed pre-registered cuts manifest required')
    seen=set()
    for cut in cases['cuts']:
        if set(cut)-{'world_id','snapshot_path','snapshot_sha256','memory'} or not {'world_id','snapshot_path','snapshot_sha256'}<=set(cut):
            raise ValueError('closed cut fields required')
        if cut['world_id'] not in registration['train_ids']:
            raise ValueError('held-out or unregistered cut refused before reading snapshots')
        key=(cut['world_id'],cut['snapshot_sha256'])
        if key in seen:
            raise ValueError('duplicate cut')
        seen.add(key)
        _actor_json(cut.get('memory',{}))
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    _write(output/'registration.json',registration)
    _write(output/'input-cuts.json',cases)
    rows=[]
    with AttemptJournal(output/'attempts.jsonl') as journal:
        for cut in cases['cuts']:
            snapshot=cf.Snapshot.from_dict(_read(cut['snapshot_path']))
            if snapshot.sha256!=cut['snapshot_sha256']:
                raise ValueError('input snapshot hash mismatch')
            runtime=cf.restore(snapshot)
            ctx=cf.actor_context(runtime.environment.agent_observation(runtime.seat),runtime.actors[runtime.seat])
            ctx=StrategyContext(ctx.step,ctx.player,ctx.observation,
                _actor_json(cut.get('memory',dict(ctx.memory))),ctx.shops)
            pipeline=_PipelineBridge(pipeline_factory)
            view=pipeline.step(ctx)
            result=search_snapshot(registration,cut['world_id'],snapshot,view,budget=budget,
                continuation_factory=None if continuation_factory is None else _factory(continuation_factory),
                event_sink=journal.append)
            rows.append(result)
            _write(output/f'case-{len(rows):04d}.json',result)
    model=fit_rules(registration,rows)
    _write(output/'rules.json',model)
    _write(output/'summary.json',seal(dict(schema=SCHEMA+'-training',cases=len(rows),
        attempted_native_suffixes=budget.evaluations,status=model['status'],
        rules=len(model['rules']),policy_sha256=model['content_sha256'],
        states={status:sum(r['status']==status for r in rows) for status in sorted({r['status'] for r in rows})})))
    return model


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('register',help='write disjoint train/held-out world and source bindings; no games')
    p.add_argument('--train',nargs='+'); p.add_argument('--heldout',nargs='+')
    p.add_argument('--min-worlds',type=int,default=2); p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('empty',help='explicit untrained K0006 fallback control, never learned strength')
    p.add_argument('--registration',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('train',help='bounded same-cut native suffix trials on registered TRAIN worlds only')
    p.add_argument('--registration',type=Path,required=True); p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--max-games',type=int,default=4)
    p.add_argument('--seconds',type=float,default=300.); p.add_argument('--max-suffix-steps',type=int,default=719)
    p.add_argument('--pipeline-factory',default='farm_neat.strategy.pipeline:StrategyPipeline')
    p.add_argument('--continuation-factory'); p.add_argument('--maintainer',action='store_true')
    p=sub.add_parser('panel',help='complete paired held-out native games; never fit on these results')
    p.add_argument('--registration',type=Path,required=True); p.add_argument('--rules',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--max-games',type=int,default=4)
    p.add_argument('--rivals',nargs='+',default=['k0006']); p.add_argument('--timeout',type=float,default=120.)
    p.add_argument('--pipeline-factory',default='farm_neat.strategy.pipeline:StrategyPipeline')
    p.add_argument('--adopter-factory'); p.add_argument('--maintainer',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='register':
        value=register_split(train_ids=args.train,heldout_ids=args.heldout,min_worlds=args.min_worlds)
        _write(args.output,value)
    elif args.command=='empty':
        value=fit_rules(_read(args.registration),[]); _write(args.output,value)
    elif args.command=='train':
        if args.max_games>4 and not args.maintainer:
            raise ValueError('more than four native attempts is maintainer-only')
        value=train_cuts(_read(args.registration),_read(args.cases),args.output,
            pipeline_factory=args.pipeline_factory,
            budget=SearchBudget(max_evaluations=args.max_games,max_suffix_steps=args.max_suffix_steps,seconds=args.seconds),
            continuation_factory=args.continuation_factory)
    else:
        if args.max_games>4 and not args.maintainer:
            raise ValueError('more than four native attempts is maintainer-only')
        value=run_holdout(_read(args.registration),_read(args.rules),args.output,max_games=args.max_games,
            rivals=tuple(args.rivals),timeout=args.timeout,pipeline_factory=args.pipeline_factory,
            adopter_factory=args.adopter_factory)
    print(json.dumps({k:v for k,v in value.items() if k in ('schema','content_sha256','status','result')},sort_keys=True,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
