"""Recorded-game strategy census and pre-registered, source-bound suffix measurements.

``Census.observe(StepView, identity, report=...)`` is the reusable consumer. It
never selects actions, changes a module, or supplies a missing certificate. The
optional report is the existing pipeline's structured diagnostics; without it,
call counts are explicitly unavailable. Proposal counts, no-ops, refusal-only
diagnostics, interventions, typed answer fields, and module calls have SEPARATE
denominators. EXACT zero market PASS rows are not useful intervention coverage.

``scan_rows`` replays actual OWN observations/actions without native transitions.
Sparse archived cuts restart memory at every gap: they cannot stand in for the
missing transitions or own controller/cohort history. The built-in archive reader
consumes the shipped weakness witnesses, NOT post-market worker snapshots as
pre-action observations. New complete recordings use the existing native run
loop, frozen registered actors and public_guard, with a read-only recorder.

Sampling is deterministic, within module/decision/value/control-basis strata.
Different market-window and terminal objectives are never pooled into one score.
A sample is written BEFORE suffix evaluation. Failed/censored adoptions retain
UNKNOWN deltas; successful paired deltas are MEASURED, never actor certificates.
Reason rankings join exact sample/proposal hashes. Dollar rankings are diagnostic
associations, not proof that fixing one reason recovers the measured money; reasons
may overlap. Unmeasured reasons have null dollar values, NOT zero priority.

CLI examples (from repository root, PYTHONPATH=src, PYTHONHASHSEED=0)::

    python3 -B -m farm_neat.strategy.census scan --output work/runs/census
    python3 -B -m farm_neat.strategy.census probe --output work/runs/census-probe
    python3 -B -m farm_neat.strategy.census measure --plan PLAN.json --output NEW

``probe`` costs at most FOUR native starts: one complete recording and three
suffixes (control, one pre-selected no-op, fresh replay). It explicitly tests
connectivity only, not profitable intervention. Default pipeline missing producers
remain missing. The current accepted counterfactual codec supports K0006/K0006,
not Router, tape0 or PassActor. Never substitute those rivals or play a tape as
though it were a responsive-policy continuation. The full census recording command
is maintainer-only; previously recorded inputs can always be scanned with zero
native games. No GPU, training, coordinator, codec, or economic changes are owned.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from farm_neat.evaluation.common import bound, canonical, digest, equal, file_sha, read, write_new
from farm_neat.evaluation import router_panel
from farm_neat.strategy.pipeline import MODULES, StrategyPipeline
from farm_neat.strategy.types import Evidence, Estimate, LAST_STEP, Proposal, StepView, SuffixResult, UNKNOWN

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'farm-strategy-census-v1'
RECORD_SCHEMA = SCHEMA + '-record'
SELECTION_RULE = ('round-robin sorted module/decision/value/control strata; within each stratum '
                  'descending margin point or certified lower bound, stable sample ID; '
                  'unpriced rows excluded; no cross-basis numeric ranking')
CASH_FIELDS = ('own_cash_delta', 'rival_cash_delta', 'margin_delta')
WEAKNESS = ROOT / 'work/runs/reviewer-play/solver-lane/weakness/native'


def _kinds():
    return dict.fromkeys((e.value for e in Evidence), 0)


def _module():
    return dict(calls=0, calls_known_cuts=0, cuts_with_proposals=0, proposals=0,
                noops=0, diagnostics=0, interventions=0, proposal_evidence=_kinds(),
                answers={}, answer_evidence={}, answer_evidence_total=_kinds(), abstentions={})


def _identity(identity):
    if not isinstance(identity, Mapping):
        raise ValueError('record identity required')
    required = {'world', 'world_sha256', 'seat', 'agent', 'rival', 'record_sha256'}
    if not required.issubset(identity):
        raise ValueError('complete registered world/seat/record identity required')
    try:
        world = router_panel.world_registry()[identity['world']]
    except KeyError as error:
        raise ValueError('unregistered world') from error
    equal(identity['world_sha256'], world.sha256, 'registered world identity')
    if type(identity['seat']) is not int or identity['seat'] not in (0, 1):
        raise ValueError('seat must be 0 or 1')
    for name in ('record_sha256', 'world_sha256'):
        value = identity[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid ' + name)
    if identity['agent'] != 'k0006' or identity['rival'] not in ('k0006', 'router', 'tape0', 'pass'):
        raise ValueError('registered frozen actor/rival identities required')
    return copy.deepcopy(dict(identity))


def decision_type(proposal):
    c = proposal.candidate
    if not isinstance(c, Mapping):
        return 'UNCLASSIFIED'
    return str(c.get('family', c.get('kind', 'UNCLASSIFIED')))


def disposition(proposal):
    c = proposal.candidate
    if isinstance(c, Mapping):
        if c.get('status') == 'KEEP_BASELINE' or c.get('kind') == 'KEEP':
            return 'NOOP'
        if c.get('status') == 'UNKNOWN':
            return 'DIAGNOSTIC'
    return 'INTERVENTION'


def _estimate_fields(value, prefix='', seen=None):
    """Only current typed quantities, excluding proposal duplicates and old history."""
    if isinstance(value, Estimate):
        yield prefix, value
        return
    seen = set() if seen is None else seen
    if is_dataclass(value) or isinstance(value, (Mapping, tuple, list)):
        if id(value) in seen:
            return
        seen.add(id(value))
    if is_dataclass(value):
        items = ((f.name, getattr(value, f.name)) for f in fields(value))
    elif isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, (tuple, list)):
        items = ((str(i), v) for i, v in enumerate(value))
    else:
        return
    for key, item in items:
        if key in ('proposal', 'proposals', 'history', 'reserve'):
            continue
        yield from _estimate_fields(item, (prefix + '.' if prefix else '') + str(key), seen)


class Census:
    """Read-only StepView consumer. Identity/provenance stay outside actor context."""

    def __init__(self):
        self.modules = {m: _module() for m in MODULES}
        self.rows = []
        self._cuts = set()
        self._identities = {}
        self._decisions = {}
        self._reasons = Counter()
        self._reason_cuts = defaultdict(set)

    def observe(self, view: StepView, identity: Mapping, *, report=None):
        if not isinstance(view, StepView):
            raise ValueError('StepView required')
        identity = _identity(identity)
        if identity['seat'] != view.context.player:
            raise ValueError('record/view seat mismatch')
        cut_id = digest(dict(identity=identity, step=view.context.step))
        if cut_id in self._cuts:
            raise ValueError('duplicate recorded actor cut')
        if any(p.module not in self.modules for p in view.proposals):
            raise ValueError('unregistered proposal module')
        if report is not None:
            if (report.get('step'), report.get('player')) != (view.context.step, view.context.player):
                raise ValueError('stale pipeline report')
            if set(report.get('modules', {})) != set(MODULES):
                raise ValueError('complete pipeline module report required')
            for name, row in report['modules'].items():
                for key in ('calls', 'proposals'):
                    if type(row.get(key)) is not int or row[key] < 0:
                        raise ValueError('invalid report count')
                if row['proposals'] != sum(p.module == name for p in view.proposals):
                    raise ValueError('report/proposal count mismatch')
                for kind in ('unknown', 'abstentions'):
                    if not isinstance(row.get(kind), Mapping) or any(
                            not isinstance(k, str) or type(v) is not int or v < 1 for k, v in row[kind].items()):
                        raise ValueError('invalid structured report reasons')
        self._cuts.add(cut_id)
        self._identities[digest(identity)] = identity
        cut_reasons = Counter()
        if report is not None:
            for name, data in report['modules'].items():
                row = self.modules[name]
                row['calls'] += data['calls']
                row['calls_known_cuts'] += 1
                for reason, count in data['unknown'].items():
                    cut_reasons[name, reason] = count
                for reason, count in data['abstentions'].items():
                    row['abstentions'][reason] = row['abstentions'].get(reason, 0) + count
        proposals_here = Counter(p.module for p in view.proposals)
        for name in proposals_here:
            self.modules[name]['cuts_with_proposals'] += 1
        emitted, local_reasons = [], Counter()
        for p in view.proposals:
            candidate = p.candidate if isinstance(p.candidate, Mapping) else {}
            category, decision = disposition(p), decision_type(p)
            row = self.modules[p.module]
            row['proposals'] += 1
            row[{'NOOP': 'noops', 'DIAGNOSTIC': 'diagnostics', 'INTERVENTION': 'interventions'}[category]] += 1
            row['proposal_evidence'][p.evidence.value] += 1
            dkey = (p.module, decision)
            d = self._decisions.setdefault(dkey, dict(module=p.module, decision_type=decision, proposals=0,
                                                     dispositions=dict.fromkeys(('NOOP','DIAGNOSTIC','INTERVENTION'),0),
                                                     evidence={k: _kinds() for k in CASH_FIELDS}))
            d['proposals'] += 1
            d['dispositions'][category] += 1
            reasons = {(p.module, 'UNCERTIFIED_' + key.upper()) for key in CASH_FIELDS
                       if getattr(p, key).kind is Evidence.UNKNOWN}
            refusals = candidate.get('refusals', ())
            if isinstance(refusals, (list, tuple)):
                reasons.update((p.module, r) for r in refusals if isinstance(r, str) and r)
            f = view.feasibility.get(p.key)
            if f is not None and f.verdict.value == 'UNKNOWN':
                reasons.add(('capacity', 'UNCERTIFIED_FEASIBILITY'))
            for key in CASH_FIELDS:
                d['evidence'][key][getattr(p, key).kind.value] += 1
            for reason in reasons:
                local_reasons[reason] += 1
            proposal_sha = digest(p.to_dict())
            out = dict(sample_id=digest(dict(cut=cut_id, proposal=proposal_sha)), cut_id=cut_id,
                       identity=identity, step=view.context.step, proposal=p.to_dict(),
                       proposal_sha256=proposal_sha, decision_type=decision, disposition=category,
                       value_basis=str(candidate.get('scope', 'UNSPECIFIED_BASIS')),
                       control_basis=copy.deepcopy(candidate.get('control', candidate.get('baseline_sha256',
                                          candidate.get('control_orders', 'UNSPECIFIED_CONTROL')))),
                       unknown_reasons=[dict(module=m, reason=r) for m, r in sorted(reasons)])
            emitted.append(out)
        # Existing report occurrences are authoritative; do not count them again
        # just because the same diagnostic also appears on a proposal.
        for key, count in local_reasons.items():
            cut_reasons.setdefault(key, count)
        for key, count in cut_reasons.items():
            self._reasons[key] += count
            self._reason_cuts[key].add(cut_id)
        self.rows.extend(emitted)
        if view.reserve is not None:
            self._answer('liquidity', 'reserve.protected', view.reserve.protected)
            self._answer('liquidity', 'reserve.discretionary', view.reserve.discretionary)
        for f in view.feasibility.values():
            answers = self.modules['capacity']['answers']
            key = 'feasibility:' + f.verdict.value
            answers[key] = answers.get(key, 0) + 1
        for name, publication in view.context.derived.items():
            module = 'liquidity' if name == 'liquidity_result' else name
            if module not in self.modules or name in ('liquidity', 'counterfactual'):
                continue
            for field, estimate in _estimate_fields(publication):
                self._answer(module, field, estimate)
            if module == 'opponent':
                # Only newly arrived transitions. Old history is not fresh answers.
                for sample in getattr(publication, 'history', ()):
                    if sample.end_step == view.context.step:
                        self._answer(module, 'current_effect.' + sample.product, sample.effect)
            status = getattr(publication, 'status', None)
            if isinstance(status, Evidence):
                counts = self.modules[module]['answers']
                key = 'result_status:' + status.value
                counts[key] = counts.get(key, 0) + 1
        return emitted

    def _answer(self, module, field, estimate):
        counts = self.modules[module]['answer_evidence'].setdefault(field, _kinds())
        counts[estimate.kind.value] += 1
        self.modules[module]['answer_evidence_total'][estimate.kind.value] += 1

    def summary(self):
        identities = sorted(self._identities.values(), key=lambda i: (i['world'], i['rival'], i['seat'], i['record_sha256']))
        return copy.deepcopy(dict(schema=SCHEMA, actor_cuts=len(self._cuts),
            records=len(identities), registered_worlds=len({i['world'] for i in identities}),
            identities=identities, modules=self.modules,
            decisions=[self._decisions[k] for k in sorted(self._decisions)],
            unknown_reasons=[dict(module=m, reason=r, occurrences=n, cuts=len(self._reason_cuts[m, r]))
                             for (m, r), n in sorted(self._reasons.items(), key=lambda item: (-item[1], item[0]))],
            interpretation=dict(proposal_evidence='margin_delta kind, NOT proof of execution',
                answers='typed answer-field counts; not proposals or independent trials',
                calls='only known when pipeline report supplied; see calls_known_cuts',
                reason_occurrences='pipeline occurrence counts, including repeated calls; cuts are deduplicated',
                strength='NO_POLICY_STRENGTH_CLAIM; exposed fixed worlds; correlated cuts/seats')))


def missing_certificate(reason):
    """Small follow-on certificates, not asserted or implemented theorems."""
    text = reason.lower()
    if 'post-worker custody' in text or 'phase certificate' in text:
        return 'Certify own post-worker custody for the exact declared baseline joint action and market window.'
    if 'production_book' in text:
        return 'Export explicit own cohort IDs, birth/service history and requested expansions; do not infer a book from counts.'
    if 'production_replan' in text or 'canonical v0 plan' in text:
        return 'Supply a reachable current V0 plan and bounded marginal edit menu, preserving prefix/STOP semantics.'
    if 'cashflow' in text or 'funding' in text:
        return 'A complete dated own-commitment funding scope, including reactive unresolved debits, before declaring discretionary cash.'
    if 'shadow' in text or 'protection' in text:
        return 'Certify protected custody plus capital/market-slot opportunity cost for the declared replacement window.'
    if 'capacity' in text or 'feasibility' in text:
        return 'A final-context, full-proposal-bound capacity case; a feasible schedule is not an execution lowering.'
    if 'rival' in text or 'future_order' in text or 'flow' in text:
        return 'Actor-visible future order/resource support; historical effective displacement alone does not certify request/rank words.'
    if 'endgame' in text or 'full-horizon' in text or 'counterfactual' in text:
        return 'An actor-safe terminal continuation/commitment certificate; separately extend exact evaluator codecs for the unchanged rival.'
    if 'production_value' in text or 'cash_delta' in text or 'margin_delta' in text:
        return 'Common-control dated own/rival terminal-cash valuation with explicit continuation effects and guarded cancellation.'
    if 'observed_transition' in text:
        return 'Contiguous pre-action observations and actual own actions; omitted transitions cannot be imputed.'
    return 'Provide the exact producer certificate named by this diagnostic; retain UNKNOWN until validated.'


def rank_unknown_reasons(summary, proposal_rows, measurements):
    by_id = {r['sample_id']: r for r in proposal_rows}
    evidence, seen = defaultdict(list), set()
    for measurement in measurements:
        key = measurement['sample_id']
        if key in seen:
            raise ValueError('duplicate measured sample; replay is not an independent observation')
        seen.add(key)
        if key not in by_id or measurement['proposal_sha256'] != by_id[key]['proposal_sha256']:
            raise ValueError('measurement/proposal identity mismatch')
        value = Estimate.from_dict(measurement['deltas']['margin_delta'])
        # A control/no-op's zero does not measure money blocked by a certificate.
        if (by_id[key]['disposition'] != 'INTERVENTION'
                or measurement['status'] != 'COMPLETE' or value.kind is not Evidence.MEASURED):
            continue
        for reason in by_id[key]['unknown_reasons']:
            evidence[reason['module'], reason['reason']].append(value.value)
    ranks = []
    for row in summary['unknown_reasons']:
        values = evidence[row['module'], row['reason']]
        ranks.append(dict(**row, paired_margins=values, measured_samples=len(values),
                          value_status='MEASURED_ASSOCIATION_NOT_CAUSAL_CERTIFICATE' if values else 'UNMEASURED',
                          max_positive_paired_margin=max(0., max(values)) if values else None,
                          minimum_paired_margin=min(values) if values else None,
                          mean_paired_margin=sum(values)/len(values) if values else None,
                          smallest_missing_certificate=missing_certificate(row['reason'])))
    return sorted(ranks, key=lambda r: (r['max_positive_paired_margin'] is None,
                    -(r['max_positive_paired_margin'] or 0), -r['cuts'], r['module'], r['reason']))


def build_sample_plan(rows, *, limit, scan_plan_sha256, include_noops=False):
    """Freeze evaluator sampling before any suffix measurement.

    The plan is intentionally useful even when empty: an empty intervention plan is a
    source-bound BLOCKED result, not permission to silently substitute no-ops or
    uncertified diagnostics. ``include_noops`` exists only for explicit connectivity
    controls and is recorded in the plan.
    """
    if not isinstance(scan_plan_sha256, str) or len(scan_plan_sha256) != 64:
        raise ValueError('scan plan sha256 required')
    selection = select_sample(rows, limit=limit, include_noops=include_noops)
    selected = selection['selected']
    groups = {}
    for row in selected:
        key = row['cut_id']
        group = groups.setdefault(key, dict(cut_id=key, identity=copy.deepcopy(row['identity']),
                                            step=row['step'], sample_ids=[]))
        if group['identity'] != row['identity'] or group['step'] != row['step']:
            raise ValueError('cut ID aliases different source identities')
        group['sample_ids'].append(row['sample_id'])
    status = 'READY' if selected else ('BLOCKED_NO_CONTROL_SAMPLE' if include_noops
                                       else 'BLOCKED_NO_ACTIONABLE_INTERVENTION')
    return dict(schema=SCHEMA + '-sample-plan', status=status, scan_plan_sha256=scan_plan_sha256,
                selection_frozen_before_measurement=True, selection_rule=selection['selection_rule'],
                include_noops=include_noops, selected=copy.deepcopy(selected), strata=selection['strata'],
                unpriced=selection['unpriced'], cut_groups=[groups[k] for k in sorted(groups)],
                measurement_scope=('CONTROL_CONNECTIVITY_ONLY' if include_noops else
                                   'ACTIONABLE_INTERVENTIONS_ONLY'),
                blocked_reason=(None if selected else
                    ('no eligible no-op control at requested cuts' if include_noops else
                     'the recorded pipeline emitted no priced actionable intervention; keep UNKNOWN and wait for a certified proposal')))


def _sample_plan_index(plan):
    if not isinstance(plan, Mapping) or plan.get('schema') != SCHEMA + '-sample-plan':
        raise ValueError('sample plan schema')
    if plan.get('selection_frozen_before_measurement') is not True:
        raise ValueError('sample selection was not frozen before measurement')
    selected = plan.get('selected')
    if not isinstance(selected, list) or len({r.get('sample_id') for r in selected}) != len(selected):
        raise ValueError('unique sample-plan rows required')
    return {r['sample_id']: r for r in selected}


def select_sample(rows, *, limit, include_noops=False):
    if type(limit) is not int or limit < 0:
        raise ValueError('nonnegative sample limit required')
    groups, unpriced = defaultdict(list), 0
    for row in rows:
        if row['disposition'] == 'DIAGNOSTIC' or (row['disposition'] == 'NOOP' and not include_noops):
            continue
        e = Estimate.from_dict(row['proposal']['margin_delta'])
        score = e.low if e.kind is Evidence.BOUND else e.value
        if score is None:
            unpriced += 1
            continue
        key = (row['proposal']['module'], row['decision_type'], row['value_basis'], canonical(row['control_basis']))
        groups[key].append((score, row))
    for values in groups.values():
        values.sort(key=lambda item: (-item[0], item[1]['sample_id']))
    keys, selected, index = sorted(groups), [], 0
    while len(selected) < limit:
        batch = [groups[k][index][1] for k in keys if index < len(groups[k])]
        if not batch:
            break
        selected.extend(batch[:limit - len(selected)])
        index += 1
    return dict(selection_rule=SELECTION_RULE, include_noops=include_noops,
                strata=[dict(module=k[0], decision_type=k[1], value_basis=k[2], control_basis=json.loads(k[3]),
                             priced_candidates=len(groups[k])) for k in keys],
                unpriced=unpriced, selected=copy.deepcopy(selected))


def scan_rows(rows, identity, census, *, pipeline_factory=StrategyPipeline, on_view=None):
    identity = _identity(identity)
    steps = [r['observation'].get('step') for r in rows]
    if any(type(s) is not int or not 0 <= s <= LAST_STEP for s in steps):
        raise ValueError('only real pre-action rows 0..718 required')
    if steps != sorted(set(steps)):
        raise ValueError('strictly ordered unique pre-action records required')
    pipe, memory, previous, resets = pipeline_factory(), {}, None, 0
    for row in rows:
        observation = copy.deepcopy(row['observation'])
        if observation.get('player') != identity['seat']:
            raise ValueError('record observation seat mismatch')
        step = observation['step']
        if previous is None or step != previous + 1:
            memory, resets = {}, resets + 1
        view = pipe.step(observation, memory=memory)
        census.observe(view, identity, report=getattr(pipe, 'last_report', None))
        if on_view is not None:
            on_view(view, identity)
        memory = json.loads(canonical(pipe.remember(view, copy.deepcopy(row['action']))))
        previous = step
    return dict(pre_action_rows=len(rows), omitted_pre_action_rows=LAST_STEP + 1 - len(rows),
                memory_resets=resets, continuation='ACTUAL_OWN_ACTIONS_ONLY; RESET_AT_EVERY_GAP', native_games=0)


def _read_gzip(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    def bad(value):
        raise ValueError('nonfinite JSON')
    with gzip.open(path, 'rt') as stream:
        value = json.load(stream, object_pairs_hook=pairs, parse_constant=bad)
    equal(value, bound(value), 'record content hash')
    return value


def load_record(path):
    path = Path(path)
    sha = file_sha(path)
    value = _read_gzip(path)
    if value.get('schema') == RECORD_SCHEMA:
        identity = _identity({**value['identity'], 'record_sha256': sha})
        if value.get('status') != 'COMPLETE' or len(value.get('rows', [])) != LAST_STEP + 1:
            raise ValueError('complete native census record required; failure remains a failure')
        rows = value['rows']
        if [r['observation'].get('step') for r in rows] != list(range(LAST_STEP + 1)):
            raise ValueError('complete ordered census record required')
        traces = value.get('traces', [])
        if len(traces) != LAST_STEP + 1 or value.get('native_steps') != LAST_STEP + 1:
            raise ValueError('complete native trace required')
        for step, (row, trace) in enumerate(zip(rows, traces)):
            equal(trace['step'], step, 'native trace clock')
            equal(row['observation']['player'], identity['seat'], 'native record seat')
            equal(row['action'], trace['actions'][identity['seat']], 'record/trace focal action')
        equal(value['terminal_cash'], traces[-1]['cash'], 'native terminal cash')
        coverage = 'COMPLETE_RECORDED_PRE_ACTION_ROWS'
    elif value.get('schema') == 'farm-solver-weakness-reconstruction-v1-case':
        manifest = read(path.parent / 'manifest.json')
        equal(value['manifest_sha256'], manifest['content_sha256'], 'weakness manifest')
        case = value['case']; trace = value['trace']; harness = value['harness_row']
        equal(case, manifest['cases'][value['case_index']], 'registered case')
        if value.get('status') != 'COMPLETE' or not trace['done'] or trace['steps'] != LAST_STEP + 1:
            raise ValueError('complete weakness record required')
        equal(harness['world_sha256'], router_panel.world_registry()[case['world']].sha256, 'registered world')
        equal(harness['actions_sha256'], digest(trace['actions']), 'recorded action tapes')
        equal(harness['seat'], case['seat'], 'recorded seat')
        equal(harness['opponent'], case['opponent'], 'recorded rival')
        equal(trace['scenario_identity']['scenario'], router_panel.world_registry()[case['world']].scenario(), 'recorded scenario')
        # This witness ships five genuine pre-action observations. Unit before/after
        # snapshots occur in another game phase and are NEVER used to fill gaps.
        rows = []
        for snapshot in trace['snapshots']:
            step = snapshot['step']
            observation = snapshot['observations'][case['seat']]
            equal(observation['step'], step, 'snapshot pre-action clock')
            rows.append(dict(observation=copy.deepcopy(observation), action=copy.deepcopy(trace['actions'][case['seat']][step])))
        identity = _identity(dict(world=case['world'], world_sha256=harness['world_sha256'], seat=case['seat'],
                                  agent=case['agent'], rival=case['opponent'], record_sha256=sha))
        coverage = 'SPARSE_RECORDED_CUTS_NOT_FULL_GAME_CENSUS'
    else:
        raise ValueError('unsupported recorded observation schema')
    return dict(identity=identity, rows=copy.deepcopy(rows), coverage=coverage, source=str(path), source_sha256=sha)


def _source_pins():
    paths = [p for p in (ROOT / 'src/farm_neat/strategy').glob('*.py')]
    paths += [ROOT / 'src/farm_neat/evaluation/router_panel.py', ROOT / 'project/configs/scenarios.json']
    from farm_neat.strategy.counterfactual import source_pins
    # The accepted evaluator already owns engine, K0006 and continuation closure.
    return {**source_pins(), **{str(p.relative_to(ROOT)): file_sha(p) for p in sorted(paths)}}


def scan_records(paths, output, *, history_limit=96):
    output = Path(output)
    if output.exists():
        raise FileExistsError('new census output directory required')
    records = [load_record(p) for p in paths]
    if len({r['source_sha256'] for r in records}) != len(records):
        raise ValueError('duplicate input record bytes')
    output.mkdir(parents=True)
    started, cpu = time.monotonic(), time.process_time()
    scan_plan = write_new(output / 'SCAN_PLAN.json', dict(schema=SCHEMA + '-scan-plan',
        source_pins=_source_pins(), history_limit=history_limit, native_games=0,
        records=[dict(source=r['source'], sha256=r['source_sha256'], coverage=r['coverage']) for r in records]))
    census, coverage = Census(), []
    for record in records:
        stats = scan_rows(record['rows'], record['identity'], census,
                          pipeline_factory=lambda: StrategyPipeline(history_limit=history_limit))
        coverage.append(dict(identity=record['identity'], scope=record['coverage'], **stats))
    summary = census.summary()
    sample_plan = build_sample_plan(census.rows, limit=4, scan_plan_sha256=scan_plan['content_sha256'])
    sample_plan = write_new(output / 'SAMPLE_PLAN.json', sample_plan)
    summary.update(coverage=coverage, native_games=0, cpu_seconds=time.process_time()-cpu,
                   wall_seconds=time.monotonic()-started, history_limit=history_limit,
                   total_proposals=len(census.rows), intervention_sample=copy.deepcopy(sample_plan),
                   sample_plan_sha256=sample_plan['content_sha256'], measurement_status=sample_plan['status'],
                   unknown_ranking=rank_unknown_reasons(summary, census.rows, []))
    summary = write_new(output / 'CENSUS.json', summary)
    with (output / 'PROPOSALS.jsonl.gz').open('xb') as stream:
        stream.write(gzip.compress(('\n'.join(canonical(r) for r in census.rows)+'\n').encode(), mtime=0))
    lines = ['# Strategy module census', '',
             '| Module | Calls | Proposals | No-ops | Diagnostics | Interventions | EXACT | BOUND | MEASURED | HEURISTIC | UNKNOWN |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in summary['modules'].items():
        values = [row[k] for k in ('calls','proposals','noops','diagnostics','interventions')]
        values += [row['proposal_evidence'][e.value] for e in Evidence]
        lines.append('| ' + name + ' | ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', '## Typed estimate answers', '',
              '| Module | EXACT | BOUND | MEASURED | HEURISTIC | UNKNOWN |',
              '|---|---:|---:|---:|---:|---:|']
    for name, row in summary['modules'].items():
        lines.append('| ' + name + ' | ' + ' | '.join(str(row['answer_evidence_total'][e.value]) for e in Evidence) + ' |')
    lines += ['', '## UNKNOWN reasons', '', '| Module | Reason | Occurrences | Cuts | Smallest missing certificate |',
              '|---|---|---:|---:|---|']
    for row in summary['unknown_ranking']:
        reason = row['reason'].replace('|', r'\|')
        cert = row['smallest_missing_certificate'].replace('|', r'\|')
        lines.append(f"| {row['module']} | {reason} | {row['occurrences']} | {row['cuts']} | {cert} |")
    lines += ['', f"Actor cuts: {summary['actor_cuts']}; input records: {summary['records']}; registered worlds: {summary['registered_worlds']}.",
              f"Measurement sample: {summary['measurement_status']} ({len(sample_plan['selected'])} selected); SAMPLE_PLAN.json is frozen before suffix evaluation.",
              'Proposal evidence is margin evidence, not execution. Typed answer evidence has a separate denominator.',
              'No omitted observation or missing certificate is imputed. No native games are run by scan.',
              'UNKNOWN dollar priority is null until an exact sample/proposal measurement is joined.']
    (output / 'TABLE.md').write_text('\n'.join(lines)+'\n')
    return summary, census.rows


def snapshot_support(identity):
    return 'SUPPORTED_K0006_PAIR' if identity['agent'] == identity['rival'] == 'k0006' else 'UNSUPPORTED_RIVAL_CODEC:' + identity['rival']


def native_cost(*, record_games, samples, controls, replays):
    values = (record_games, samples, controls, replays)
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError('nonnegative integer native start counts required')
    return sum(values)


def check_native_budget(starts, *, maintainer=False):
    if type(starts) is not int or starts < 0:
        raise ValueError('nonnegative integer game allowance required')
    if starts > 4 and not maintainer:
        raise ValueError('reviewers may start at most four native games including records and replays')


def _started_receipts(root):
    """Persisted native-start charges under one campaign root, including failed runs."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(str(p.relative_to(root)) for p in root.rglob('*STARTED.json') if p.is_file())


def _campaign_budget(plan_path, plan, expected_cost):
    """Resolve an optional persisted campaign cap before opening a new measurement."""
    root_name, limit = plan.get('native_budget_root'), plan.get('max_total_native_starts')
    if root_name is None and limit is None:
        return None, None, []
    if root_name is None or limit is None:
        raise ValueError('native campaign budget needs both root and total-start limit')
    if not isinstance(root_name, str) or not root_name or Path(root_name).is_absolute() or '..' in Path(root_name).parts:
        raise ValueError('native budget root must be a safe relative path')
    if type(limit) is not int or limit < 1:
        raise ValueError('positive native campaign start limit required')
    root = (Path(plan_path).parent / root_name).resolve()
    receipts = _started_receipts(root)
    if len(receipts) + expected_cost > limit:
        raise ValueError(f'native campaign budget exhausted: {len(receipts)} already started + '
                         f'{expected_cost} requested > {limit}')
    return root, limit, receipts


def paired_result(control: SuffixResult, child: SuffixResult):
    from farm_neat.strategy.counterfactual import compare
    return {k: v.to_dict() for k, v in compare(control, child).items()}


def _validate_sample(row):
    identity = _identity(row['identity'])
    if type(row['step']) is not int or not 0 <= row['step'] <= LAST_STEP:
        raise ValueError('registered pre-action cut required')
    proposal = Proposal.from_dict(row['proposal'])
    equal(row['proposal_sha256'], digest(row['proposal']), 'registered proposal bytes')
    equal(row['cut_id'], digest(dict(identity=identity, step=row['step'])), 'registered cut ID')
    equal(row['sample_id'], digest(dict(cut=row['cut_id'], proposal=row['proposal_sha256'])), 'registered sample ID')
    equal(row['disposition'], disposition(proposal), 'registered disposition')
    equal(row['decision_type'], decision_type(proposal), 'registered decision type')


def write_registration(path, samples, *, snapshot_path, snapshot_sha256, source_pins,
                       replay=False, expected_control=None, sample_plan=None,
                       native_budget_root=None, max_total_native_starts=None):
    if not samples or len({r['sample_id'] for r in samples}) != len(samples):
        raise ValueError('nonempty unique measured samples required')
    if len({r['cut_id'] for r in samples}) != 1:
        raise ValueError('one exact snapshot cut per registration')
    for row in samples:
        _validate_sample(row)
    sample_plan_sha256 = None
    if sample_plan is not None:
        sample_plan = read(sample_plan) if isinstance(sample_plan, (str, Path)) else copy.deepcopy(sample_plan)
        index = _sample_plan_index(sample_plan)
        for row in samples:
            if row['sample_id'] not in index or index[row['sample_id']] != row:
                raise ValueError('registration sample is not byte-identical to the frozen sample plan')
        sample_plan_sha256 = sample_plan.get('content_sha256') or digest(sample_plan)
    if (native_budget_root is None) != (max_total_native_starts is None):
        raise ValueError('native campaign budget needs both root and total-start limit')
    if native_budget_root is not None:
        if (not isinstance(native_budget_root, str) or not native_budget_root
                or Path(native_budget_root).is_absolute() or '..' in Path(native_budget_root).parts):
            raise ValueError('native budget root must be a safe relative path')
        if type(max_total_native_starts) is not int or max_total_native_starts < 1:
            raise ValueError('positive native campaign start limit required')
    document = dict(schema=SCHEMA + '-registration', samples=copy.deepcopy(samples),
        snapshot_path=str(snapshot_path), snapshot_sha256=snapshot_sha256, source_pins=source_pins,
        sample_plan_sha256=sample_plan_sha256, selection_rule=SELECTION_RULE, selection_frozen_before_measurement=True,
        required_suffix_evaluations=1+len(samples)+int(replay), replay_first_child=bool(replay),
        expected_control=expected_control,
        scope='EXPOSED_FIXED_WORLD_CONDITIONAL_SUFFIX; NOOP_IS_CONTROL_ONLY; NO_STRENGTH_CLAIM')
    if native_budget_root is not None:
        document.update(native_budget_root=native_budget_root,
                        max_total_native_starts=max_total_native_starts)
    return write_new(path, document)


def measure_registered(plan_path, output, *, seconds=120., max_evaluations=4, continuation_factory=None):
    """Exact same-cut control and child branches; a fresh adopter per branch.

    A caller-supplied continuation_factory(proposal_dict) must depend only on actor-side
    proposal/own context; evaluator world labels are NOT passed to that callback.
    The CLI supplies no new guards, lowerers or missing certificates.
    """
    from farm_neat.strategy import counterfactual as cf, adoption
    plan_path, output = Path(plan_path).resolve(), Path(output)
    plan = read(plan_path)
    equal(plan['schema'], SCHEMA + '-registration', 'registration schema')
    if not plan['selection_frozen_before_measurement']:
        raise ValueError('unregistered measurement')
    expected_cost = 1 + len(plan['samples']) + int(plan['replay_first_child'])
    equal(plan['required_suffix_evaluations'], expected_cost, 'registered evaluation count')
    if type(max_evaluations) is not int or expected_cost > max_evaluations:
        raise ValueError('suffix evaluation budget is smaller than the frozen plan')
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive finite suffix time limit required')
    budget_root, budget_limit, prior_receipts = _campaign_budget(plan_path, plan, expected_cost)
    for name, sha in plan['source_pins'].items():
        equal(file_sha(ROOT / name), sha, 'measurement source pin ' + name)
    snap_path = plan_path.parent / plan['snapshot_path']
    raw = json.loads(gzip.decompress(snap_path.read_bytes())) if snap_path.suffix == '.gz' else json.loads(snap_path.read_text())
    snapshot = cf.Snapshot.from_dict(raw)
    equal(snapshot.sha256, plan['snapshot_sha256'], 'registered snapshot')
    runtime = cf.restore(snapshot)  # zero steps; refuse incompatible source/runtime first
    if not plan['samples'] or len({r['sample_id'] for r in plan['samples']}) != len(plan['samples']):
        raise ValueError('nonempty unique registered samples required')
    for row in plan['samples']:
        _validate_sample(row)
        identity = _identity(row['identity'])
        equal(row['proposal_sha256'], digest(row['proposal']), 'registered candidate')
        if snapshot_support(identity) != 'SUPPORTED_K0006_PAIR':
            raise ValueError(snapshot_support(identity))
        equal(runtime.environment.scenario, router_panel.world_registry()[identity['world']].scenario(), 'snapshot world')
        equal(runtime.seat, identity['seat'], 'snapshot seat')
        equal(runtime.environment.step_index, row['step'], 'snapshot pre-action cut')
        if any(type(a).__name__ != 'K0006Adapter' for a in runtime.actors):
            raise ValueError('declared frozen actors differ from exact snapshot actors')
    if output.exists():
        raise FileExistsError('new measurement output directory required; no automatic rerun')
    output.mkdir(parents=True)
    started, cpu = time.monotonic(), time.process_time()
    attempts, measurements = [], []

    def run(label, row=None):
        # STARTED persists before any native transition. Failed starts remain charged.
        if budget_root is not None and len(_started_receipts(budget_root)) >= budget_limit:
            raise ValueError('native campaign budget exhausted before ' + label)
        write_new(output / (label + '-STARTED.json'), dict(plan_sha256=plan['content_sha256'], label=label,
                  sample_id=None if row is None else row['sample_id']))
        trace = []
        if row is None:
            result = cf.evaluate(snapshot, None, seconds=seconds, trace=trace)
        else:
            p = Proposal.from_dict(row['proposal'])
            continuation = continuation_factory(copy.deepcopy(row['proposal'])) if continuation_factory else adoption.K0006Continuation()
            result = adoption.evaluate(snapshot, p, continuation, seconds=seconds, trace=trace)
        record = dict(label=label, result=asdict(result), trace=trace, plan_sha256=plan['content_sha256'],
                      native_steps=len(trace), native_start_charged=1)
        write_new(output / (label + '.json'), record)
        attempts.append(record)
        return result, trace

    control, control_trace = run('control')
    expected = plan.get('expected_control')
    control_matches = None if expected is None else (
        control.status == 'COMPLETE' and control.own_cash == expected['own_cash']
        and control.rival_cash == expected['rival_cash'] and control_trace == expected['trace'])
    # A failed identical-control check does not authorize candidate runs.
    if control.status == 'COMPLETE' and control_matches is not False:
        first = None
        for index, row in enumerate(plan['samples']):
            child, trace = run('child-' + str(index), row)
            if first is None:
                first = (child, trace)
            measurements.append(dict(sample_id=row['sample_id'], proposal_sha256=row['proposal_sha256'],
                identity=row['identity'], step=row['step'], disposition=row['disposition'],
                status=child.status, failure=child.failure, deltas=paired_result(control, child)))
        replay_equal = None
        if plan['replay_first_child']:
            replay, trace = run('fresh-replay', plan['samples'][0])
            replay_equal = (asdict(replay) == asdict(first[0]) and trace == first[1])
    else:
        replay_equal = None
    complete = (len(attempts) == expected_cost and all(r['result']['status'] == 'COMPLETE' for r in attempts)
                and control_matches is not False and replay_equal is not False)
    result = dict(schema=SCHEMA + '-measurement', status='COMPLETE' if complete else 'PARTIAL_OR_FAILED',
        plan_sha256=plan['content_sha256'], sample_plan_sha256=plan.get('sample_plan_sha256'),
        measurements=measurements, control_matches_recorded=control_matches,
        fresh_replay_equal=replay_equal, native_starts_charged=len(attempts), native_steps=sum(a['native_steps'] for a in attempts),
        native_complete_suffixes=sum(a['result']['status'] == 'COMPLETE' for a in attempts),
        cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-started,
        native_budget_root=plan.get('native_budget_root'), native_start_limit=budget_limit,
        native_starts_preexisting=len(prior_receipts), native_start_receipts_preexisting=prior_receipts,
        productive_interventions=sum(m['status']=='COMPLETE' and m['disposition']=='INTERVENTION' for m in measurements),
        limit='Measured no-ops do not establish module value. Uncertified/failed alternatives retain UNKNOWN, never zero.')
    return write_new(output / 'MEASUREMENTS.json', result)


def record_game(output, *, world='pair-06-00', seat=0, rival='k0006', save_cut=False, seconds=120.):
    """One complete game through existing env.run; read-only pre-action recorder."""
    from farm_neat.evaluation.scenarios import ScenarioEnvironment
    from farm_neat.strategy import counterfactual as cf
    world = router_panel.world_registry()[world]
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError('seat 0 or 1 required')
    if rival not in ('k0006','router','tape0','pass'):
        raise ValueError('registered rival required')
    if save_cut and rival != 'k0006':
        raise ValueError('UNSUPPORTED_RIVAL_CODEC:' + rival)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive recording time limit required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('new record directory required')
    agent_spec, rival_spec = router_panel.load_actor('k0006'), router_panel.load_actor(rival)
    actors = [rival_spec.spawn(), rival_spec.spawn()] if rival == 'k0006' else [None, None]
    if rival != 'k0006':
        actors[seat], actors[1-seat] = agent_spec.spawn(), rival_spec.spawn()
    identity = dict(world=world.id, world_sha256=world.sha256, seat=seat, agent='k0006', rival=rival,
                    actor_sha256=agent_spec.sha256, rival_sha256=rival_spec.sha256)
    output.mkdir(parents=True)
    plan = write_new(output / 'RECORD_PLAN.json', dict(schema=SCHEMA + '-record-plan', identity=identity,
        registry_sha256=router_panel.world_registry().sha256, source_pins=_source_pins(),
        actor_bindings=[a.identity for a in (agent_spec, rival_spec)],
        max_native_starts=1, cut_step=LAST_STEP if save_cut else None, scope='EXPOSED_REGISTERED_WORLD', seconds=seconds))
    write_new(output / 'STARTED.json', dict(plan_sha256=plan['content_sha256'], native_start_charged=1))
    started, cpu = time.monotonic(), time.process_time()
    observations, cut, result, error = [], [], None, None
    env = None
    try:
        env = ScenarioEnvironment(world.scenario())
        calls = [router_panel.public_guard(a.act if hasattr(a, 'act') else a,
                 seat=s, deadline=started+seconds) for s, a in enumerate(actors)]
        original_first = calls[0]
        def recording_first(observation, configuration):
            # Native run calls seat 0 first, BEFORE either actor has advanced.
            observations.append(copy.deepcopy(dict(env.agent_observation(seat))))
            if save_cut and env.step_index == LAST_STEP:
                cut.append(cf.capture(env, actors, seat=seat))
            return original_first(observation, configuration)
        calls[0] = recording_first
        result = env.run(calls, record_actions=True)
        if result.steps != LAST_STEP + 1 or result.statuses != ('DONE','DONE'):
            raise ValueError('complete native terminal boundary required')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    rows = [] if result is None else [dict(observation=o, action=result.actions[seat][i]) for i, o in enumerate(observations)]
    traces = [] if result is None else [dict(step=i, actions=[result.actions[0][i], result.actions[1][i]],
                cash=[result.cash_history[0][i+1], result.cash_history[1][i+1]]) for i in range(result.steps)]
    data = bound(dict(schema=RECORD_SCHEMA, identity=identity, plan_sha256=plan['content_sha256'],
        status='COMPLETE' if error is None else 'FAILED', error=error, rows=rows,
        terminal_cash=None if result is None else list(result.rewards), traces=traces,
        source_pins=_source_pins(), actor_bindings=[a.identity for a in (agent_spec, rival_spec)], native_starts_charged=1,
        native_steps=0 if env is None else env.step_index, cpu_seconds=time.process_time()-cpu,
        wall_seconds=time.monotonic()-started))
    with (output / 'RECORD.json.gz').open('xb') as stream:
        stream.write(gzip.compress(canonical(data).encode(), mtime=0))
    if cut:
        with (output / 'CUT.json.gz').open('xb') as stream:
            stream.write(gzip.compress(canonical(cut[0].to_dict()).encode(), mtime=0))
    return data, cut[0] if cut else None


def probe(output, *, world='pair-06-00', seat=0, seconds=120., history_limit=96):
    """Four-start diagnostic control; no certificate or productive candidate invented."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('new probe output directory required')
    check_native_budget(native_cost(record_games=1, samples=1, controls=1, replays=1))
    output.mkdir(parents=True)
    write_new(output / 'PROBE_PLAN.json', dict(schema=SCHEMA + '-probe-plan', world=world, seat=seat, rival='k0006',
        max_native_starts=4, roles=['record','control','top_noop_child','fresh_child_replay'],
        sample_scope='TOP_PRICED_NOOP_AT_RECORDED_FINAL_PRE_ACTION_CUT; NOT_A_PRODUCTIVE_MODULE_TEST',
        productive_candidates='do not substitute this smoke for missing guarded interventions', source_pins=_source_pins()))
    record, snapshot = record_game(output/'record', world=world, seat=seat, save_cut=True, seconds=seconds)
    if record['status'] != 'COMPLETE' or snapshot is None:
        return write_new(output/'RESULT.json', dict(schema=SCHEMA+'-probe', status='RECORD_FAILED',
                         native_starts_charged=1, error=record['error']))
    summary, rows = scan_records([output/'record/RECORD.json.gz'], output/'census', history_limit=history_limit)
    selected = select_sample([r for r in rows if r['step']==LAST_STEP and r['disposition']=='NOOP'], limit=1, include_noops=True)
    if not selected['selected']:
        return write_new(output/'RESULT.json', dict(schema=SCHEMA+'-probe', status='NO_NOOP_CANDIDATE', native_starts_charged=1))
    write_registration(output/'PLAN.json', selected['selected'], snapshot_path='record/CUT.json.gz',
        snapshot_sha256=snapshot.sha256, source_pins=_source_pins(), replay=True,
        expected_control=dict(own_cash=record['terminal_cash'][seat], rival_cash=record['terminal_cash'][1-seat],
                              trace=record['traces'][LAST_STEP:]),
        native_budget_root='.', max_total_native_starts=4)
    measured = measure_registered(output/'PLAN.json', output/'paired', seconds=seconds, max_evaluations=3)
    return write_new(output/'RESULT.json', dict(schema=SCHEMA+'-probe', status=measured['status'],
        native_starts_charged=1+measured['native_starts_charged'], measured=measured,
        census_sha256=summary['content_sha256'],
        unknown_ranking=rank_unknown_reasons(summary, rows, measured['measurements']),
        decision='CONNECTIVITY_CONTROL_ONLY; NONTRIVIAL_ADOPTION_AND_DOLLAR_RANKING_REMAIN_BLOCKED'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    scan = sub.add_parser('scan', help='Zero-native-game replay of recorded actor observations')
    scan.add_argument('--records', type=Path, nargs='+', default=sorted(WEAKNESS.glob('case-*.json.gz')))
    scan.add_argument('--output', type=Path, required=True)
    scan.add_argument('--history-limit', type=int, default=96)
    p = sub.add_parser('probe', help='Four-start K0006/K0006 no-op connectivity control, not module strength')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--world', default='pair-06-00')
    p.add_argument('--seat', type=int, choices=(0,1), default=0)
    p.add_argument('--seconds', type=float, default=120.)
    p.add_argument('--history-limit', type=int, default=96)
    m = sub.add_parser('measure', help='Execute already registered same-cut suffix samples')
    m.add_argument('--plan', type=Path, required=True)
    m.add_argument('--output', type=Path, required=True)
    m.add_argument('--seconds', type=float, default=120.)
    m.add_argument('--max-evaluations', type=int, default=4)
    r = sub.add_parser('record', help='Full registered-game recording; >4 starts is maintainer-only')
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--worlds', nargs='+', default=['pair-06-00'])
    r.add_argument('--rivals', nargs='+', choices=('k0006','router','tape0','pass'), default=['k0006'])
    r.add_argument('--seats', nargs='+', type=int, choices=(0,1), default=[0])
    r.add_argument('--max-games', type=int, default=1)
    r.add_argument('--maintainer', action='store_true')
    r.add_argument('--seconds', type=float, default=120.)
    args = parser.parse_args(argv)
    if args.command == 'scan':
        if not args.records:
            parser.error('recorded inputs required; none found')
        result, _ = scan_records(args.records, args.output, history_limit=args.history_limit)
        print(canonical({k:result[k] for k in ('actor_cuts','records','registered_worlds','total_proposals','native_games','cpu_seconds')}))
    elif args.command == 'probe':
        result = probe(args.output, world=args.world, seat=args.seat, seconds=args.seconds, history_limit=args.history_limit)
        print(canonical({k:result[k] for k in ('status','native_starts_charged')}))
        return 0 if result['status']=='COMPLETE' else 1
    elif args.command == 'measure':
        check_native_budget(args.max_evaluations)
        result = measure_registered(args.plan, args.output, seconds=args.seconds, max_evaluations=args.max_evaluations)
        print(canonical(result))
        return 0 if result['status']=='COMPLETE' else 1
    else:
        worlds = [w.id for w in router_panel.world_registry().worlds] if args.worlds == ['all'] else args.worlds
        for world in worlds:
            router_panel.world_registry()[world]
        jobs = [(w,r,s) for w in worlds for r in args.rivals for s in args.seats]
        if len(set(jobs)) != len(jobs):
            parser.error('duplicate native jobs')
        check_native_budget(len(jobs), maintainer=args.maintainer)
        if args.max_games < len(jobs):
            parser.error('explicit max-games must cover the entire registered job list; no silent truncation')
        if args.output.exists():
            raise FileExistsError('new recording campaign directory required')
        args.output.mkdir(parents=True)
        write_new(args.output/'PLAN.json', dict(schema=SCHEMA+'-record-campaign', jobs=jobs, max_games=args.max_games,
                  maintainer=args.maintainer, source_pins=_source_pins(), exposure='EXPOSED_REGISTERED_FIXED_WORLDS'))
        results=[]
        for i,(world,rival,seat) in enumerate(jobs):
            row,_=record_game(args.output/f'game-{i:04d}', world=world, rival=rival, seat=seat, seconds=args.seconds)
            results.append(dict(world=world, rival=rival, seat=seat, status=row['status'], error=row['error']))
            if row['status'] != 'COMPLETE':
                break
        result=write_new(args.output/'RESULT.json', dict(schema=SCHEMA+'-record-campaign-result', results=results,
                       native_starts_charged=len(results), complete=len(results)==len(jobs) and all(r['status']=='COMPLETE' for r in results)))
        print(canonical(result))
        return 0 if result['complete'] else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
