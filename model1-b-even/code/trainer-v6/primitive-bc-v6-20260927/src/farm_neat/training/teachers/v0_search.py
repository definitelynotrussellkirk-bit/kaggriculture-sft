"""Development-only reachable V0 teacher improvement, with paired KEEP controls.

This consumes v0_oracle's actual-prefix neighborhood, DawnActor and fresh-process
replay. It does not replace the compiler, enumerate an exhaustive solver, fit a
network, or import search4's own-cash/held-out ranking. Each case starts from an
EMPTY Plan and reconstructs explicit, legal prior dawn actions. The correction
and every later dawn use KEEP except for one candidate edit at the named dawn.
No preloaded portfolio is called reachable; no TargetPolicy injects future plans.

Objective: terminal WDL (WIN=1, DRAW=.5, LOSS=0). Ties retain KEEP, then pool order;
own cash, rival cash and margin are separate diagnostics, never tie breakers.
The measured failure class is exact frozen-teacher reproduction with all Router
losses (docs/training-state.md, V0-HELDOUT). Only that qualitative failure class
motivates a bounded production-addition / production-exit probe. No protected
outcome table or prospective world is loaded to choose actions or tune the pool.

Every admitted candidate AND KEEP is executed and replayed in independent child
processes, from opening to terminal. Losing/cash-harmful candidates stay in the
receipt. Only strict WDL gains with exact replay can become *review candidates*;
this module never writes a training corpus. Conditional target export and
maintainer admission remain separate, including later-dawn feature/schema checks.

Native commands (fresh paths only; <= 2*max_candidates attempts, including replays)::

  PYTHONPATH=src python3 -B -m farm_neat.training.teachers.v0_search probe \
    --output work/runs/v0-teacher-improve-probe --max-games 6 --seconds 300
  PYTHONPATH=src python3 -B -m farm_neat.training.teachers.v0_search run \
    --request REQUEST.json --output work/runs/v0-teacher-improve-next \
    --max-games 6 --seconds 300

A custom request is the closed oracle request plus split='development' and
objective='TERMINAL_WDL', with this module's schema, empty initial_plan and KEEP
policy/suffix_policy. Worlds must already belong to the exposed router_panel
registry. Pool and prior schedule are frozen before any outcome. Supply an
explicit runner only for portable fixtures: it is always SYNTHETIC_DIAGNOSTIC,
with zero physical-game counts and no teacher candidates. --charge-assignment
records each native start through the saved reviewer ticket before launching.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Callable

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.evaluation.common import bound, canonical, equal, file_sha, read, write_new
from farm_neat.training.teachers import v0_oracle as oracle

ROOT = oracle.ROOT
SCHEMA = 'farm-v0-teacher-search-v1'
OBJECTIVE = 'TERMINAL_WDL'
FAILURE_CLASS = 'EXACT_TEACHER_REPRODUCTION_WITH_ALL_ROUTER_LOSSES'
_NATIVE_ATTEMPT = oracle._fresh_attempt


def probe_request() -> dict:
    """Small prospective exposed probe; no outcome-conditioned proposal ranking."""
    wheat = V.MODULE_BY_KEY['WHEAT:watered:b00']
    strawberry = V.MODULE_BY_KEY['STRAWBERRY:watered:b01']
    return {'schema': SCHEMA + '-request', 'split': 'development', 'objective': OBJECTIVE,
            'world': 'router-dairy', 'seat': 0, 'rival': 'router',
            'program': {'dawn': 24, 'initial_plan': Plan().canonical(),
                        'prior_actions': {'0': [V.encode('PROD', wheat, V.DELTAS.index('PLUS_ONE')), 0]},
                        'policy': {'kind': 'keep'}, 'suffix_policy': {'kind': 'keep'}},
            'action_pool': [V.encode('PROD', strawberry, V.DELTAS.index('PLUS_ONE')),
                            V.encode('PROD', wheat, V.DELTAS.index('DROP_ALL'))],
            'max_trials': 2, 'max_candidates': 3}


def _oracle_request(request: dict) -> dict:
    return {**{k: copy.deepcopy(v) for k, v in request.items() if k not in ('split', 'objective')},
            'schema': oracle.SCHEMA + '-request'}


def validate_request(request: dict) -> dict:
    """Reject preloads, protected splits, unknown fields and hidden suffix targets.

    Prior actions are syntax-checked here and proven legal only by reconstructing
    the actual prefix in DawnActor. A source-bound request alone is NOT evidence
    of reachability, successful execution or policy strength.
    """
    fields = {'schema', 'split', 'objective', 'world', 'seat', 'rival', 'program',
              'action_pool', 'max_trials', 'max_candidates'}
    if (not isinstance(request, dict) or set(request) != fields
            or request.get('schema') != SCHEMA + '-request'):
        raise ValueError('closed teacher-search request required; no outcome inputs')
    if request['split'] != 'development' or request['objective'] != OBJECTIVE:
        raise ValueError('development-only terminal-WDL search required')
    base = oracle.validate_request(_oracle_request(request))
    if base['program']['initial_plan'] != Plan().canonical():
        raise ValueError('empty initial Plan required; a preload is not reachable evidence')
    if any(base['program'][name] != {'kind': 'keep'} for name in ('policy', 'suffix_policy')):
        raise ValueError('KEEP decision and KEEP suffix required; no teacher-prefix injection')
    return copy.deepcopy(request)


def source_pins() -> dict:
    pins = oracle.source_pins()
    pins['src/farm_neat/training/teachers/v0_search.py'] = file_sha(Path(__file__))
    return pins


def _wdl(own: float, rival: float) -> tuple[str, float]:
    return ('WIN', 1.) if own > rival else ('DRAW', .5) if own == rival else ('LOSS', 0.)


def rank_candidates(menu: dict, outcomes: list[dict]) -> dict:
    """Pure paired reducer, not an admission authority or a probability estimate."""
    equal(menu, bound(menu), 'neighborhood content binding')
    if menu.get('schema') != oracle.SCHEMA + '-neighborhood':
        raise ValueError('oracle neighborhood required')
    candidates = menu['candidates']
    if not candidates or candidates[0]['actions'] != [V.STOP_ID]:
        raise ValueError('explicit KEEP control required at candidate zero')
    for trace in candidates:
        equal(trace, bound(trace), 'trace content binding')
        equal(trace['dawn_sha256'], menu['dawn_sha256'], 'trace dawn')
    expected = [trace['content_sha256'] for trace in candidates]
    if (not isinstance(outcomes, list) or len(outcomes) != len(expected)
            or [o.get('candidate_sha256') for o in outcomes] != expected
            or len(set(expected)) != len(expected)):
        raise ValueError('exact ordered outcome per distinct admitted candidate required')
    kinds = {o.get('evidence_kind') for o in outcomes}
    if len(kinds) != 1 or not kinds <= {'NATIVE_GAME', 'SYNTHETIC_DIAGNOSTIC'}:
        raise ValueError('one explicit nonmixed evidence kind required')
    rows = []
    for outcome in outcomes:
        equal(outcome['dawn_sha256'], menu['dawn_sha256'], 'paired dawn')
        own, rival = outcome['own_cash'], outcome['rival_cash']
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (own, rival)):
            raise ValueError('finite nonnegative own and rival cash required')
        wdl, score = _wdl(own, rival)
        if 'wdl' in outcome: equal(outcome['wdl'], wdl, 'cash-derived WDL')
        if 'margin' in outcome: equal(outcome['margin'], own-rival, 'cash-derived margin')
        rows.append({**copy.deepcopy(outcome), 'wdl': wdl, 'wdl_score': score, 'margin': own-rival})
    keep = rows[0]
    for row in rows:
        row.update(delta_own_cash=row['own_cash']-keep['own_cash'],
                   delta_rival_cash=row['rival_cash']-keep['rival_cash'],
                   delta_margin=row['margin']-keep['margin'],
                   delta_wdl=row['wdl_score']-keep['wdl_score'],
                   competitive_harm=row['wdl_score'] < keep['wdl_score'],
                   cash_harm=row['own_cash'] < keep['own_cash'])
    selected = max(range(len(rows)), key=lambda i: rows[i]['wdl_score'])
    return bound({'schema': SCHEMA + '-selection', 'objective': OBJECTIVE,
                  'selected_index': selected, 'selected': candidates[selected], 'keep': candidates[0],
                  'improved': rows[selected]['delta_wdl'] > 0, 'pairs': rows,
                  'evidence_kind': next(iter(kinds)), 'tie_rule': 'KEEP_THEN_POOL_ORDER_NO_CASH_TIEBREAK',
                  'training_admission': False,
                  'scope': 'ONE_EXPOSED_DEVELOPMENT_CELL; NOT_POPULATION_STRENGTH'})


def _validate_attempt(value: dict, task: dict, *, evidence_kind: str) -> None:
    """Rebind execution to the requested path/context; do not trust outcome flags."""
    equal(value, bound(value), 'attempt content binding')
    equal(value['source_pins'], task['source_pins'], 'attempt source closure')
    game = value['game']; request = task['request']
    if (value['status'] != 'COMPLETE' or game['status'] != 'COMPLETE'
            or game['steps'] != 719 or game['evidence_kind'] != evidence_kind):
        raise ValueError('complete 719-step execution with declared evidence kind required')
    equal(game['physical_games_completed'], 1 if evidence_kind == 'NATIVE_GAME' else 0,
          'physical game accounting')
    equal(game['world'], request['world'], 'game world')
    equal(game['seat'], request['seat'], 'game seat')
    if evidence_kind == 'NATIVE_GAME':
        world = oracle.panel.world_registry()[request['world']]
        equal(game['world_sha256'], world.sha256, 'registered world identity')
        equal(game['actor_identity_hashes']['opponent'], oracle.panel.load_actor(request['rival']).sha256,
              'frozen opponent identity')
    dawn = value['dawn']
    equal(dawn['observation']['step'], request['program']['dawn'], 'requested dawn')
    equal(dawn['observation']['player'], request['seat'], 'dawn seat')
    if task['expected_dawn'] is not None:
        equal(dawn['content_sha256'], task['expected_dawn'], 'complete pre-edit dawn')
    equal(value['trace'], oracle.trace_edits(dawn, task['decision']), 'exact requested prefix trace')
    own, rival = game['cash'][request['seat']], game['cash'][1-request['seat']]
    if len(game['cash']) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
                                   for x in (own, rival)):
        raise ValueError('finite terminal cash pair required')
    equal(game['wdl'], _wdl(own, rival)[0], 'cash-derived WDL')
    equal(game['margin'], own-rival, 'cash-derived margin')


def run_search(request: dict, output: Path, *, max_games: int = 6, seconds: float = 300.,
               runner: Callable | None = None, charge_assignment: bool = False) -> dict:
    """One bounded case, fresh output only; no resume/retry or corpus mutation.

    runner(output,index,role,task,timeout) -> bound oracle-shaped attempt. A custom
    runner is ALWAYS diagnostic and must report zero physical games. The default
    native runner alone can produce review candidates, after ALL exact replays.
    A failure returns a durable FAILED result and an empty candidate list.
    """
    request = validate_request(request)
    oracle._integer(max_games, 1, 100000, 'attempt budget')
    required = 2 * request['max_candidates']
    if max_games < required:
        raise ValueError(f'attempt budget must reserve {required} including every KEEP/candidate replay')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 1800:
        raise ValueError('finite seconds budget in (0,1800] required')
    if type(charge_assignment) is not bool or (charge_assignment and runner is not None):
        raise ValueError('assignment charging is only for native execution')
    native = runner is None
    if native and oracle._fresh_attempt is not _NATIVE_ATTEMPT:
        raise ValueError('substituted native runner refused; supply an explicit diagnostic runner')
    kind = 'NATIVE_GAME' if native else 'SYNTHETIC_DIAGNOSTIC'
    execute = _NATIVE_ATTEMPT if native else runner
    output = Path(output).absolute()
    if output != output.resolve() or not output.is_relative_to(ROOT / 'work/runs') or output.exists():
        raise ValueError('new non-symlink output under work/runs required')
    pins = source_pins()
    # The existing oracle owns the child implementation and verifies its closure.
    # The parent additionally pins this search/reducer before and after each call.
    child_pins = oracle.source_pins() if native else pins
    run_id = uuid.uuid4().hex
    manifest = write_new(output / 'MANIFEST.json', {
        'schema': SCHEMA + '-run', 'run_id': run_id, 'request': request, 'source_pins': pins,
        'oracle_source_pins': child_pins, 'evidence_kind': kind, 'max_games': max_games,
        'worst_case_attempts': required, 'seconds_budget': seconds, 'objective': OBJECTIVE,
        'failure_class': FAILURE_CLASS,
        'failure_evidence': {'path': 'docs/training-state.md',
                             'use': 'QUALITATIVE_CLASS_ONLY; NO_PROTECTED_OUTCOME_INPUTS'},
        'training_admission': False,
        'retention': {'purpose': 'Development reachable-plan search and every paired replay',
                      'dependencies': list(pins),
                      'remove_when': 'No teacher-candidate intake or search replay depends on this case'}})
    start = time.monotonic(); parent_cpu = time.process_time()
    completed = []; started_count = 0; verified = []
    selection = None; menu = None; teachers = []; error = None

    def attempt(role: str, decision: list[int], expected: str | None) -> dict:
        nonlocal started_count
        left = seconds - (time.monotonic()-start)
        if left <= 0: raise TimeoutError('case deadline exhausted; no automatic retry')
        if started_count >= max_games: raise ValueError('case attempt budget exhausted')
        equal(source_pins(), pins, 'search source closure')
        if charge_assignment:
            command = [sys.executable, '-B', 'src/farm_neat/tooling/review_package.py', '--charge-native-start']
            charge = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            (output / f'charge-{started_count:03d}.json').write_text(charge.stdout)
        index = started_count
        task = {'schema': oracle.SCHEMA+'-play', 'request': _oracle_request(request),
                'decision': list(decision), 'expected_dawn': expected, 'source_pins': child_pins,
                'teacher_search_run_id': run_id, 'teacher_search_manifest_sha256': manifest['content_sha256']}
        oracle._journal(output/'search-attempts.jsonl', {
            'event': 'STARTED', 'index': index, 'role': role, 'evidence_kind': kind, 'run_id': run_id})
        started_count += 1  # count failures/torn launches, not only returned games
        value = execute(output, index, role, task, min(300., left))
        _validate_attempt(value, task, evidence_kind=kind)
        equal(source_pins(), pins, 'search source closure after execution')
        completed.append(value)
        oracle._journal(output/'search-attempts.jsonl', {
            'event': 'COMPLETE', 'index': index, 'role': role, 'receipt_sha256': value['content_sha256']})
        return value

    try:
        keep = attempt('KEEP', [0], None)
        dawn = keep['dawn']
        menu = oracle.neighborhood(dawn, [0], request['action_pool'],
                                   max_trials=request['max_trials'], max_candidates=request['max_candidates'])
        write_new(output/'neighborhood.json', menu)
        measurements = [keep]
        for trace in menu['candidates'][1:]:
            row = attempt('CANDIDATE', trace['actions'], dawn['content_sha256'])
            equal(row['trace'], trace, 'candidate menu trace')
            measurements.append(row)
        scores = []
        for row in measurements:
            score = oracle._outcome(row)
            score['evidence_kind'] = kind  # oracle's reducer helper assumes native; never inherit that for fixtures
            scores.append(score)
        selection = rank_candidates(menu, scores)
        write_new(output/'selection.json', selection)
        for index, measured in enumerate(measurements):
            replay = attempt('KEEP_REPLAY' if index == 0 else 'CANDIDATE_REPLAY',
                             measured['trace']['actions'], dawn['content_sha256'])
            equal(oracle.replay_fingerprint(replay), oracle.replay_fingerprint(measured),
                  'fresh paired replay including cash, both action streams and full continuation')
            verified.append({'candidate_index': index, 'primary_sha256': measured['content_sha256'],
                             'replay_sha256': replay['content_sha256'],
                             'fingerprint': oracle.replay_fingerprint(measured)})
        if native and selection['improved']:
            winner = selection['selected_index']; trace = menu['candidates'][winner]
            teachers.append(bound({
                'schema': SCHEMA+'-review-candidate', 'actor': oracle.actor_input(dawn),
                'target': {'actions': trace['actions'], 'plan': trace['plan'], 'prefix_trace': trace['steps']},
                'trainer_only': {'split': 'development', 'run_id': run_id,
                    'manifest_sha256': manifest['content_sha256'], 'source_pins': pins,
                    'opening_origin': 'EMPTY_PLAN_WITH_EXECUTED_PREFIX', 'request': request,
                    'dawn': dawn, 'keep_pair': verified[0], 'selected_pair': verified[winner],
                    'delta_wdl': selection['pairs'][winner]['delta_wdl'],
                    'admission': 'PENDING_MAINTAINER_REVIEW_AND_PREFIX_TARGET_EXPORT'},
                'training_admission': False}))
    except (ValueError, KeyError, IndexError, TypeError, RuntimeError, OSError,
            TimeoutError, subprocess.SubprocessError) as exc:
        error = f'{type(exc).__name__}: {exc}'
        teachers = []
        oracle._journal(output/'search-attempts.jsonl', {'event': 'FAILED', 'error': error, 'run_id': run_id})
    status = 'COMPLETE' if error is None else 'FAILED'
    return write_new(output/'RESULT.json', {
        'schema': SCHEMA+'-result', 'status': status, 'error': error, 'run_id': run_id,
        'manifest_sha256': manifest['content_sha256'], 'source_pins': pins,
        'evidence_kind': kind, 'selection': selection,
        'neighborhood_sha256': None if menu is None else menu['content_sha256'],
        'fresh_replay_verified': status == 'COMPLETE', 'replay_pairs_verified': len(verified),
        'replays': verified, 'native_attempts': started_count if native else 0,
        'native_games_completed': len(completed) if native else 0,
        'diagnostic_attempts': 0 if native else started_count,
        'distinct_worlds': 1 if native and completed else 0,
        'replays_are_not_additional_worlds': True,
        'seconds': time.monotonic()-start, 'controller_cpu_seconds': time.process_time()-parent_cpu,
        'child_cpu_seconds': sum(row['cpu_seconds'] for row in completed),
        'teacher_candidates': teachers, 'training_admission': False,
        'limits': ['No held-out/final outcomes read or tuned; registered exposed development cell only.',
                   'WDL ties, including cash-improved losses, do not create a teacher.',
                   'A review candidate is not admitted corpus data, a global optimum or neural strength.',
                   'One-key correction with a frozen KEEP suffix; no learned-policy suffix comparison.',
                   'Full replay includes controls and harmful candidates; no outcome cache or resume.']})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('probe', 'run'))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-games', type=int, default=6)
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--charge-assignment', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'run' and args.request is None: parser.error('--request required for run')
    if args.command == 'probe' and args.request is not None: parser.error('--request is only for run')
    try:
        request = probe_request() if args.command == 'probe' else read(args.request, binding=False)
        result = run_search(request, args.output, max_games=args.max_games, seconds=args.seconds,
                            charge_assignment=args.charge_assignment)
    except (ValueError, OSError) as error:
        print(f'REFUSED: {error}', file=sys.stderr)
        return 2
    print(canonical({k: result[k] for k in ('status', 'error', 'native_attempts', 'native_games_completed',
                                           'fresh_replay_verified', 'replay_pairs_verified', 'child_cpu_seconds')}))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
