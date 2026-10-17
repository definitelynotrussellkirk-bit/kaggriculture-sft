"""Actor-only Q selection, honest finite-menu controls and exact confirmation.

Selection is a search decision, not a teacher record or deployment authorization.
Support/envelope/confidence guards are heuristics; only a separately executed,
source-bound, freshly replayed terminal comparison can confirm an improvement.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import time

import torch

from farm_neat.training import v0_sequence_q_data as D
from farm_neat.training import v0_sequence_q as Q
from farm_neat.training import v0_search_archive as A


def validate_support(support: dict) -> None:
    D.check(support, D.SCHEMA + '-support', ('action_world_counts', 'sequence_world_counts',
        'training_world_ids', 'training_family_ids', 'envelope', 'scope'))
    for key in ('action_world_counts', 'sequence_world_counts'):
        D.need(type(support[key]) is dict and all(type(n) is int and n >= 0 for n in support[key].values()), 'SUPPORT_COUNTS')
    for key in ('training_world_ids', 'training_family_ids'):
        D.need(type(support[key]) is list and support[key] == sorted(set(support[key]))
               and all(type(x) is str and x for x in support[key]), 'SUPPORT_IDENTITIES')
    D.need(set(support['envelope']) == {'cash', 'step'}, 'SUPPORT_ENVELOPE')
    for pair in support['envelope'].values():
        D.need(type(pair) is list and len(pair) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in pair)
               and pair[0] <= pair[1], 'SUPPORT_ENVELOPE')


def choose(pack: Q.PreparedMenu, *, incumbent_id: str, qvalues: list[float], support: dict,
           probabilities: list[list[float]] | None = None, min_world_support: int = 1,
           min_advantage: float = .05, min_confidence: float = .55,
           require_envelope: bool = True) -> dict:
    """Finite menu selection. Ties and every failed guard retain the incumbent."""
    D.need(isinstance(pack, Q.PreparedMenu) and incumbent_id in pack.candidate_ids, 'MISSING_INCUMBENT')
    validate_support(support)
    D.need(type(min_world_support) is int and min_world_support >= 1 and type(require_envelope) is bool, 'SUPPORT_GUARD')
    D.need(type(min_advantage) in (int, float) and math.isfinite(min_advantage) and min_advantage >= 0
           and type(min_confidence) in (int, float) and math.isfinite(min_confidence) and 0 <= min_confidence <= 1, 'SELECTION_GUARD')
    n = len(pack.candidate_ids)
    D.need(type(qvalues) is list and len(qvalues) == n
           and all(type(x) in (float, int) and math.isfinite(x) and -1.000001 <= x <= 1.000001 for x in qvalues), 'Q_VALUES')
    if probabilities is not None:
        D.need(type(probabilities) is list and len(probabilities) == n, 'PROBABILITY_ALIGNMENT')
        for row, value in zip(probabilities, qvalues):
            D.need(type(row) is list and len(row) == 3
                   and all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1 for x in row)
                   and abs(sum(row) - 1) <= 1e-5 and abs(row[2] - row[0] - value) <= 1e-5, 'PROBABILITY_Q_MISMATCH')
    baseline = qvalues[pack.candidate_ids.index(incumbent_id)]
    envelope = support['envelope']
    in_envelope = all(envelope[k][0] <= getattr(pack.state, k) <= envelope[k][1] for k in envelope)
    rows, eligible = [], []
    for i, (key, actions, value) in enumerate(zip(pack.candidate_ids, pack.actions, qvalues)):
        count = min(support['action_world_counts'].get(str(a), 0) for a in set(actions))
        confidence = None if probabilities is None else max(probabilities[i])
        reasons = []
        if key != incumbent_id:
            if count < min_world_support: reasons.append('LOW_ACTION_SUPPORT')
            if require_envelope and not in_envelope: reasons.append('OUTSIDE_TRAINING_ENVELOPE')
            if min_confidence > 0 and (confidence is None or confidence < min_confidence): reasons.append('LOW_OR_UNAVAILABLE_CONFIDENCE')
            if value <= baseline + min_advantage: reasons.append('NO_STRICT_PREDICTED_ADVANTAGE')
            if not reasons: eligible.append((value, key))
        rows.append(dict(candidate_id=key, q=value, probabilities=None if probabilities is None else probabilities[i],
                         min_action_world_support=count, confidence=confidence, guard_failures=reasons))
    selected = sorted(eligible, key=lambda item: (-item[0], item[1]))[0][1] if eligible else incumbent_id
    return dict(status='SELECTED' if eligible else 'ABSTAINED_TO_INCUMBENT', selected_id=selected,
        changed=selected != incumbent_id, incumbent_id=incumbent_id, scores=rows,
        guards=dict(min_world_support=min_world_support, min_advantage=min_advantage,
                    min_confidence=min_confidence, require_envelope=require_envelope),
        actor_training_admissible=False, native_games_run=0,
        limit='PREDICTION_AND_HEURISTIC_SUPPORT_ARE_NOT_CONFIRMED_WDL_OR_CALIBRATION')


def control(menu: dict, *, method: str, seed: int) -> dict:
    """Controls select legal geometry, never evaluator outcomes or likelihood fiction."""
    D.validate_menu(menu)
    D.need(method in ('random', 'mutation', 'current_policy', 'incumbent'), 'CONTROL_METHOD')
    D.need(type(seed) is int and 0 <= seed < 2**63, 'SEED')
    candidates = sorted(menu['candidates'], key=lambda c: c['candidate_id'])
    if method == 'incumbent':
        pool = [c for c in candidates if c['candidate_id'] == menu['incumbent_id']]
    elif method == 'mutation':
        pool = [c for c in candidates if c['proposal_family'] in ('incumbent_mutation', 'learner_mutation', 'keep_mutation')]
    elif method == 'current_policy':
        # Only a validated import of a direct-draw pool may use this family.
        pool = [c for c in candidates if c['proposal_family'] == 'current_policy']
    else:
        pool = candidates
    selected = random.Random(seed).choice(pool)['candidate_id'] if pool else menu['incumbent_id']
    return dict(status='SELECTED_CONTROL' if pool else 'UNAVAILABLE_CONTROL', selected_id=selected,
        incumbent_id=menu['incumbent_id'], changed=selected != menu['incumbent_id'],
        method=method, seed=seed, available_candidates=len(pool),
        control_scope='UNIFORM_FROZEN_LEGAL_MENU' if method == 'random' else 'FROZEN_DECLARED_FAMILY',
        actor_training_admissible=False, native_games_run=0)


def select(prepared: dict, *, evidence_root: str | Path, method: str = 'q', seed: int = 0,
           checkpoint: str | Path | None = None, checkpoint_sha256: str | None = None,
           diagnostic: bool = False, min_world_support: int = 1, min_advantage: float = .05,
           min_confidence: float = .55, require_envelope: bool = True) -> dict:
    D.validate_prepared(prepared, evidence_root=evidence_root)
    menu, p = prepared['menu'], prepared['trainer']
    D.need(method in ('q', 'random', 'mutation', 'current_policy', 'incumbent'), 'SELECTION_METHOD')
    began = time.monotonic(); support_sha = None
    if method == 'q':
        D.need(checkpoint is not None and checkpoint_sha256 is not None, 'CHECKPOINT_REQUIRED')
        model, metadata = Q.load_checkpoint(checkpoint, expected_sha256=checkpoint_sha256, allow_diagnostic=diagnostic)
        support = metadata['support']; validate_support(support)
        if p['split'] not in D.TRAIN_SPLITS:
            D.need(p['world_id'] not in support['training_world_ids']
                   and p['family_id'] not in support['training_family_ids'], 'PROTECTED_TRAINING_OVERLAP')
        pack = Q.prepare_menu(menu)
        with torch.inference_mode():
            out = model([pack])
        decision = choose(pack, incumbent_id=menu['incumbent_id'], qvalues=out.q.tolist(), support=support,
            probabilities=None if out.probabilities is None else out.probabilities.tolist(),
            min_world_support=min_world_support, min_advantage=min_advantage,
            min_confidence=min_confidence, require_envelope=require_envelope)
        diagnostic = metadata['diagnostic']; support_sha = support['content_sha256']
    else:
        decision = control(menu, method=method, seed=seed)
        checkpoint_sha256 = None
    return D.seal(dict(schema=D.SCHEMA + '-selection', prepared_sha256=prepared['content_sha256'],
        menu_sha256=menu['content_sha256'], method=method, checkpoint_sha256=checkpoint_sha256,
        diagnostic=diagnostic, decision=decision, source_pins=D.source_pins(), support_sha256=support_sha,
        wall_seconds=time.monotonic() - began, gpu_hours=0., selection_device='cpu',
        native_games_run=0, frozen_before_confirmation=True))


def confirmation_request(prepared: dict, selection: dict, *, evidence_root: str | Path) -> dict:
    D.validate_prepared(prepared, evidence_root=evidence_root)
    D.check(selection, D.SCHEMA + '-selection', ('prepared_sha256', 'menu_sha256', 'method',
        'checkpoint_sha256', 'diagnostic', 'decision', 'source_pins', 'support_sha256',
        'wall_seconds', 'gpu_hours', 'selection_device', 'native_games_run', 'frozen_before_confirmation'))
    D.need(selection['source_pins'] == D.source_pins() and selection['prepared_sha256'] == prepared['content_sha256']
           and selection['menu_sha256'] == prepared['menu']['content_sha256'], 'STALE_SELECTION')
    D.need(selection['decision']['status'] != 'UNAVAILABLE_CONTROL', 'UNAVAILABLE_CONTROL')
    D.need(selection['native_games_run'] == 0 and selection['frozen_before_confirmation'] is True, 'PRESELECTION_REQUIRED')
    by_id = {c['candidate_id']: c for c in prepared['menu']['candidates']}
    own, chosen = prepared['menu']['incumbent_id'], selection['decision']['selected_id']
    D.need(chosen in by_id and selection['decision']['incumbent_id'] == own, 'SELECTION_CANDIDATE_JOIN')
    t = prepared['trainer']
    return A.make_suffix_request(snapshot_ref=t['snapshot'], control=D.suffix(by_id[own], t['start']['step']),
        challenger=D.suffix(by_id[chosen], t['start']['step']), source_group_id=t['source_group_id'],
        policy_sha256=selection['checkpoint_sha256'] or t['policy_sha256'],
        protocol_sha256=D.digest(selection), split=t['split'], round_id='sequence-q-frozen-confirmation',
        proposal_family='sequence_q_' + selection['method'], evidence_root=evidence_root)


def confirm(prepared: dict, selection: dict, *, output: str | Path, evidence_root: str | Path,
            native: bool, max_native_starts: int, allow_diagnostic_native_smoke: bool = False,
            runner=None) -> dict:
    """Four fresh starts total (pair plus replay), including on abstention."""
    root = A.safe_path(evidence_root).resolve(); out = A.safe_path(output)
    D.need(not out.exists() and out.resolve().is_relative_to(root), 'NEW_OUTPUT_REQUIRED')
    runner = A.native_suffix_runner if runner is None else runner
    is_native = runner is A.native_suffix_runner
    D.need(not is_native or native, 'EXPLICIT_NATIVE_AUTHORIZATION_REQUIRED')
    D.need(not is_native or not selection['diagnostic'] or allow_diagnostic_native_smoke, 'DIAGNOSTIC_NATIVE_SMOKE_REQUIRES_EXPLICIT_FLAG')
    D.need(type(max_native_starts) is int and max_native_starts >= (4 if is_native else 0), 'NATIVE_START_BUDGET_EXCEEDED')
    if is_native:
        import os
        D.need(os.environ.get('PYTHONHASHSEED') == '0', 'PYTHONHASHSEED_0_REQUIRED')
    request = confirmation_request(prepared, selection, evidence_root=root)
    out.mkdir(parents=True, exist_ok=False)
    D.write_new(out / 'SELECTION.json', selection)
    D.write_new(out / 'REQUEST.json', request)
    archive = A.Archive(out / 'archive.sqlite', evidence_root=root)
    began = time.monotonic()
    run = A.execute_suffix_to_archive(archive, request_path=out / 'REQUEST.json', output_path=out / 'RESULT.json',
                                    evaluation_id='q-confirm-' + request['content_sha256'], runner=runner)
    result = D.read_json(out / 'RESULT.json')
    A._checked_suffix_result(request, result, evidence_root=root)
    verified = is_native and result['replay']['status'] == 'VERIFIED'
    primary = result['arms']
    improvement = bool(verified and all(a['status'] == 'COMPLETE' for a in primary)
                       and primary[1]['terminal']['wdl'] > primary[0]['terminal']['wdl'])
    starts = result['counts']['native_starts']
    report = D.seal(dict(schema=D.SCHEMA + '-confirmation', method=selection['method'],
        diagnostic=selection['diagnostic'], status=run['status'], replay=result['replay']['status'],
        strict_wdl_improvements=int(improvement), suffix_native_starts=starts,
        improvements_per_1000_suffix_starts=1000. * improvement / starts if starts else None,
        improvements_per_gpu_hour=None, gpu_hours=0.,
        note='CPU_SELECTION; GPU_RATE_UNDEFINED; CAPTURE_COST_SEPARATE; ONE_CUT_NOT_BROAD_STRENGTH',
        elapsed_seconds=time.monotonic() - began, source_group_id=prepared['trainer']['source_group_id'],
        world_id=prepared['trainer']['world_id'], selected_id=selection['decision']['selected_id'],
        request=A.artifact_ref(out / 'REQUEST.json', root=root), result=A.artifact_ref(out / 'RESULT.json', root=root),
        selection=A.artifact_ref(out / 'SELECTION.json', root=root), teacher_admitted=False))
    D.write_new(out / 'CONFIRMATION.json', report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('select')
    p.add_argument('--prepared', required=True); p.add_argument('--output', required=True)
    p.add_argument('--method', choices=('q', 'random', 'mutation', 'current_policy', 'incumbent'), default='q')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--checkpoint'); p.add_argument('--checkpoint-sha256')
    p.add_argument('--diagnostic', action='store_true'); p.add_argument('--min-world-support', type=int, default=1)
    p.add_argument('--min-advantage', type=float, default=.05); p.add_argument('--min-confidence', type=float, default=.55)
    p.add_argument('--allow-outside-envelope', action='store_true'); p.add_argument('--evidence-root', default=str(D.ROOT))
    p = commands.add_parser('confirm')
    p.add_argument('--prepared', required=True); p.add_argument('--selection', required=True); p.add_argument('--output', required=True)
    p.add_argument('--native', action='store_true'); p.add_argument('--max-native-starts', type=int, required=True)
    p.add_argument('--allow-diagnostic-native-smoke', action='store_true'); p.add_argument('--evidence-root', default=str(D.ROOT))
    args = parser.parse_args(argv)
    try:
        prepared = D.read_json(args.prepared)
        if args.command == 'select':
            D.need(not A.safe_path(args.output).exists(), 'NEW_OUTPUT_REQUIRED')
            result = select(prepared, evidence_root=args.evidence_root, method=args.method, seed=args.seed,
                checkpoint=args.checkpoint, checkpoint_sha256=args.checkpoint_sha256, diagnostic=args.diagnostic,
                min_world_support=args.min_world_support, min_advantage=args.min_advantage,
                min_confidence=args.min_confidence, require_envelope=not args.allow_outside_envelope)
            D.write_new(args.output, result)
        else:
            result = confirm(prepared, D.read_json(args.selection), output=args.output, evidence_root=args.evidence_root,
                native=args.native, max_native_starts=args.max_native_starts,
                allow_diagnostic_native_smoke=args.allow_diagnostic_native_smoke)
        print(D.wire(result)); return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(D.wire(dict(schema=D.SCHEMA, status='REFUSED', reason=str(exc)))); return 2


if __name__ == '__main__':
    raise SystemExit(main())
