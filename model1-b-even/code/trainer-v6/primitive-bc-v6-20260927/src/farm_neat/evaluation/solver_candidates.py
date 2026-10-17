"""Frozen non-neural factories and a ranking over unchanged router_panel runs.

No search, fitting, training, promotion, or scheduler mutation. Default candidate
selection is frozen before this ranking. The V0 plan was selected on the already
exposed router-dairy receipt matching this compiler, not on new ranking outcomes.
It is preseeded and kept, NOT a claim that a V0 decoder can reach it from empty.

The incumbent's builtin and module:factory identities are exactly equal. Four
child/parent panels supply five candidates: 4 * (80 + 72*2*4*2) = 4,928 physical
games, including the unchanged harness's regression gates and repeated parent
controls. The reducer uses 2,880 policy/world/seat/rival observations and checks
that repeated parent results agree. It does not silently skip native controls.

CPU commands (new output paths; no native game unless run/smoke is explicit):
  PYTHONPATH=src python3 -B -m farm_neat.evaluation.solver_candidates dry-run \
    --output work/runs/solver-champion-dry.json
  PYTHONPATH=src python3 -B -m farm_neat.evaluation.solver_candidates run \
    --output work/runs/solver-champion --max-games 4928
  PYTHONPATH=src python3 -B -m farm_neat.evaluation.solver_candidates rank \
    --run-dir work/runs/solver-champion --output work/runs/solver-ranking.json
  PYTHONPATH=src python3 -B -m farm_neat.evaluation.solver_candidates smoke \
    --output work/runs/solver-champion-smoke

run defaults to FOUR attempts TOTAL per invocation, not four per child. Resume
requires --resume with unchanged settings/source. Unknown starts block resume.
smoke uses four standalone shared-runner games, not a passed 80-row panel gate.
Fixed-world confidence intervals describe exposed fixtures, NOT ladder strength.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any

from farm_neat.evaluation import k0006, router_panel as panel, scenarios, shop_router
from farm_neat.evaluation.common import bound, digest, equal, file_sha, read, write_new

ROOT = panel.ROOT
MODULE = 'farm_neat.evaluation.solver_candidates'
SCHEMA = 'farm-solver-champion-v1'
TEACHER_PATH = scenarios.TEACHERS['teacher-v5-00']
TEACHER_SHA256 = '101a6b4ff0cf4601ba9d42b23a0f54d48637af58f429c39d294b8c9c0fbdd511'
PORTFOLIO_PATH = ROOT / 'work/runs/reviewer-play/v0-compiler-repairs/roundtrip/ROUNDTRIP.json'
PORTFOLIO_SHA256 = 'd13ec377a88cf4f7d540eaade5209af0e51244798d2e9a91a00a8b811fd409e9'
PORTFOLIO_NAME = 'greedy_own_39'
RIVALS = ['router', 'k0006', 'tape0', 'pass']

# Aliases, not new wrappers: the shared harness fingerprints the original factory.
# In particular the incumbent may reuse actually played builtin-parent rows.
k0006_factory = k0006.K0006Adapter
router_factory = shop_router.ShopRouterAdapter


def _assets(paths) -> dict[str, str]:
    return {panel._path_name(Path(p)): file_sha(p) for p in sorted(set(map(Path, paths)))}


def _executor_assets() -> dict[str, str]:
    """Conservative maintained executor closure; no model/weights are loaded."""
    paths = [p for p in (ROOT / 'src/farm_neat/economy').rglob('*')
             if p.is_file() and p.suffix in ('.py', '.json')]
    paths += [ROOT / 'src/farm_neat/game/constants.py', Path(scenarios.__file__),
              ROOT / 'src/farm_neat/evaluation/common.py', Path(__file__)]
    return _assets(paths)


def tape0_factory():
    actor = scenarios.FixedTape()
    actor.identity = bound({**actor.identity, 'assets': _assets([
        ROOT / 'src/farm_neat/evaluation/shop_router_0909/actions.json',
        Path(scenarios.__file__)]), 'training_admission': False})
    return actor


def teacher_v5_factory():
    equal(file_sha(TEACHER_PATH), TEACHER_SHA256, 'frozen teacher-v5-00 bytes')
    actor = scenarios.Program(read(TEACHER_PATH, binding=False)[0]['program'])
    actor.identity = bound({**actor.identity, 'kind': 'FROZEN_ECONOMIC_PROGRAM',
        'name': 'teacher-v5-00', 'hall_index': 0, 'training_admission': False,
        'assets': {**_executor_assets(), **_assets([TEACHER_PATH])}})
    return actor


def compiled_v0_factory():
    from farm_neat.economy.v0 import vocabulary as V
    from farm_neat.economy.v0.epochs import KeepPolicy, V0Agent
    from farm_neat.economy.v0.plan import Plan

    equal(file_sha(PORTFOLIO_PATH), PORTFOLIO_SHA256, 'retained portfolio receipt bytes')
    receipt = read(PORTFOLIO_PATH, binding=False)
    selection = receipt['portfolios'][PORTFOLIO_NAME]['by_labor_factor']['4.0']
    # Do not silently rerank old portfolios under an unmeasured compiler revision.
    compiler = 'src/farm_neat/economy/v0/compiler.py'
    equal(file_sha(ROOT / compiler), receipt['sources'][compiler], 'selected repaired compiler')
    plan = Plan.from_canonical(selection['plan'])
    belief = {p: 0 for p in V.PRODUCTS}
    actor = V0Agent(KeepPolicy(), plan=plan, labor_factor=4.0, hands_mode='daily',
                    belief=belief, game_id='solver-champion-compiled-v0')
    actor.identity = bound({'kind': 'FROZEN_V0_PLAN_KEEP_POLICY', 'portfolio': PORTFOLIO_NAME,
        'initial_plan': plan.canonical(), 'initial_plan_sha256': plan.digest(),
        'labor_factor': 4.0, 'hands_mode': 'daily', 'belief': belief,
        'initialization': 'PRESEEDED_PLAN_NOT_A_REACHABILITY_CLAIM',
        'selection': 'Best retained compiled_daily own cash at lambda 4 in source-matching receipt; '
                     'router-dairy historically exposed; no ranking-world retuning.',
        'execution': 'V0Agent recompiles at dawn; not the historical opening-only execution route.',
        'training_admission': False,
        'assets': {**_executor_assets(), **_assets([PORTFOLIO_PATH])}})
    return actor


def candidate_catalog() -> list[dict[str, Any]]:
    return [{'id': name, 'factory': MODULE + ':' + factory,
             'selection': selection} for name, factory, selection in (
        ('k0006', 'k0006_factory', 'Frozen provisional incumbent; no new selection.'),
        ('router', 'router_factory', 'Frozen ShopRouter 0909.'),
        ('tape0', 'tape0_factory', 'Frozen ShopRouter tape index 0; observation-blind.'),
        ('teacher-v5-00', 'teacher_v5_factory', 'Frozen teacher hall index 0; economic program.'),
        ('compiled-v0', 'compiled_v0_factory', 'Frozen ' + PORTFOLIO_NAME + '; source-matching retained selection.'))]


def make_manifest(*, bootstrap_samples=10000, bootstrap_seed=20260917, timeout=300.) -> dict:
    candidates = candidate_catalog()
    for candidate in candidates:
        candidate['factory_identity'] = panel.load_actor(candidate['factory']).identity
        candidate['evaluation_ref'] = 'k0006' if candidate['id'] == 'k0006' else candidate['factory']
        candidate['evaluation_identity'] = panel.load_actor(candidate['evaluation_ref']).identity
    equal(candidates[0]['factory_identity'], candidates[0]['evaluation_identity'],
          'incumbent module factory equals builtin parent')
    panels = []
    for candidate in candidates[1:]:
        plan, _, _ = panel.make_plan(candidate['evaluation_ref'], 'k0006',
            bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed, timeout=timeout)
        equal(plan['rivals'], RIVALS, 'exactly four shared ranking rivals')
        equal(plan['actors'][candidate['evaluation_ref']], candidate['evaluation_identity'], 'candidate identity')
        panels.append({'candidate': candidate['id'], 'directory': candidate['id'], 'manifest': plan})
    plan = panels[0]['manifest']
    for child in panels[1:]:
        for field in ('registry', 'rivals', 'bootstrap', 'execution', 'source_pins'):
            equal(child['manifest'][field], plan[field], 'shared ranking ' + field)
    n = len(plan['registry']['worlds']) * 2 * len(RIVALS)
    return bound({'schema': SCHEMA + '-manifest', 'status': 'NOT_YET_RUN', 'scope': panel.SCOPE,
        'source_sha256': file_sha(__file__), 'candidates': candidates, 'panels': panels,
        'registry': plan['registry'], 'rivals': RIVALS, 'seats': [0, 1],
        'bootstrap': plan['bootstrap'], 'execution': plan['execution'],
        'settings': {'bootstrap_samples': bootstrap_samples, 'bootstrap_seed': bootstrap_seed,
                     'timeout': float(timeout)},
        'games_planned': sum(p['manifest']['games_planned'] for p in panels),
        'logical_ranking_games': n * len(candidates), 'legacy_games': 80 * len(panels),
        'repeated_parent_games': n * (len(panels) - 1),
        'accepted_upgrades': [],
        'ranking_rule': {'resampling_unit': 'WORLD_MEAN_OVER_BOTH_SEATS_AND_ALL_FOUR_RIVALS',
            'metric': 'candidate cash margin minus other candidate cash margin on identical world/seat/rival',
            'dominance': 'lower endpoint of percentile 95% paired CI >= 0 against every other candidate',
            'strict_edge': 'lower endpoint >= 0 and upper endpoint > 0; zero ties do not eliminate either policy',
            'multiple_dominators': 'report tied dominators, never break ties by candidate order',
            'no_dominator': 'report candidates with no incoming strict edge',
            'multiplicity': 'Pairwise diagnostic 95% intervals; not simultaneous familywise coverage.',
            'objective': 'Terminal WDL remains the game objective; cash-margin ranking is ticket-specific.'},
        'ranking_table_template': [{'candidate': c['id'], 'status': 'NOT_YET_RUN', 'mean_margin': None,
            'world_bootstrap_ci95_vs_k0006': None, 'worst_world': None,
            'per_rival_wdl': {r: None for r in RIVALS}} for c in candidates],
        'limits': ['Exposed fixed worlds, not fresh held-out or ladder strength.',
                   'V0 seed-plan execution is not a reachable training target or a neural result.',
                   'No candidate is chosen or changed from ranking outcomes.',
                   'Hashes bind declared dependencies, not malicious undeclared code.']})


def _validated_rows(manifest: dict, rows: dict) -> dict:
    equal(manifest, bound(manifest), 'ranking manifest content binding')
    names = [c['id'] for c in manifest['candidates']]
    if set(rows) != set(names):
        raise ValueError('exactly the registered candidate rows required')
    registry = panel.WorldRegistry(panel.World(**{**w, 'shops': tuple(w['shops'])})
                                   for w in manifest['registry']['worlds'])
    wanted = {(w.id, r, s) for w in registry.worlds for r in manifest['rivals'] for s in manifest['seats']}
    result = {}
    for candidate in manifest['candidates']:
        ref = candidate['evaluation_ref']
        actors = dict(manifest['panels'][0]['manifest']['actors'])
        actors[ref] = candidate['evaluation_identity']
        validation_plan = {'actors': actors, 'execution': manifest['execution']}
        result[candidate['id']] = current = {}
        for row in rows[candidate['id']]:
            if type(row.get('seat')) is not int or row['seat'] not in (0, 1):
                raise ValueError('canonical integer seat 0 or 1 required')
            key = (row['world'], row['opponent'], row['seat'])
            if key not in wanted or key in current:
                raise ValueError('unexpected or duplicate world/rival/seat row')
            job = {'world': key[0], 'opponent': key[1], 'seat': key[2], 'agent': ref}
            validation = panel.validate_game(row, job, validation_plan, registry[key[0]])
            if not validation['passed']:
                raise ValueError('invalid ranking row: ' + str(validation))
            current[key] = row
        if set(current) != wanted:
            raise ValueError('complete matching world/seat/rival rows required')
    return result


def rank_rows(manifest: dict, rows: dict[str, list[dict]]) -> dict:
    """Pure offline reducer. Call rank_run to verify on-disk native journals first."""
    cells = _validated_rows(manifest, rows)
    names = sorted(cells)
    worlds = manifest['registry']['worlds']; rivals = manifest['rivals']
    settings = {'seed': manifest['bootstrap']['seed'], 'samples': manifest['bootstrap']['samples']}
    pairwise = {name: {} for name in names}
    for ai, a in enumerate(names):
        for b in names[ai + 1:]:
            for candidate, other in ((a, b), (b, a)):
                differences = []
                per_rival = {r: [] for r in rivals}
                for world in worlds:
                    by_rival = {r: [cells[candidate][world['id'], r, s]['margin'] -
                                     cells[other][world['id'], r, s]['margin'] for s in (0, 1)] for r in rivals}
                    value = statistics.mean(x for values in by_rival.values() for x in values)
                    differences.append({'world': world['id'], 'cohort': world['cohort'],
                                        'paired_margin': value, 'paired_margins_by_rival_seat': by_rival})
                    for r in rivals:
                        per_rival[r].append({'world': world['id'], 'paired_margin': statistics.mean(by_rival[r])})
                pairwise[candidate][other] = {
                    'paired_mean': statistics.mean(w['paired_margin'] for w in differences),
                    'world_bootstrap_ci95': panel.world_bootstrap([w['paired_margin'] for w in differences], **settings),
                    'worlds_count': len(worlds), 'worlds': differences,
                    'worst_world': min(differences, key=lambda w: (w['paired_margin'], w['world'])),
                    'per_rival': {r: {'paired_mean': statistics.mean(w['paired_margin'] for w in values),
                        'world_bootstrap_ci95': panel.world_bootstrap([w['paired_margin'] for w in values], **settings),
                        'worst_world': min(values, key=lambda w: (w['paired_margin'], w['world']))}
                        for r, values in per_rival.items()}}
    dominators = [a for a in names if all(c['world_bootstrap_ci95'][0] >= 0 for c in pairwise[a].values())]
    strict_edges = [[a, b] for a in names for b, c in pairwise[a].items()
                    if c['world_bootstrap_ci95'][0] >= 0 and c['world_bootstrap_ci95'][1] > 0]
    incomparable = [a for a in names if not any(b == a for _, b in strict_edges)]
    ranking = []
    for name in names:
        world_means = [{'world': w['id'], 'mean_margin': statistics.mean(
            cells[name][w['id'], r, s]['margin'] for r in rivals for s in (0, 1))} for w in worlds]
        ranking.append({'candidate': name, 'mean_margin': statistics.mean(w['mean_margin'] for w in world_means),
            'world_bootstrap_ci95_vs_k0006': [0., 0.] if name == 'k0006' else
                                            pairwise[name]['k0006']['world_bootstrap_ci95'],
            'worst_world': min(world_means, key=lambda w: (w['mean_margin'], w['world'])),
            'per_rival_wdl': {r: panel._counts(cells[name][w['id'], r, s]['margin']
                                              for w in worlds for s in (0, 1)) for r in rivals}})
    ranking.sort(key=lambda r: (-r['mean_margin'], r['candidate']))
    native = manifest['execution']['kind'] == 'NATIVE_GAME'
    return bound({'schema': SCHEMA + '-ranking', 'status': 'COMPLETE' if native else 'SYNTHETIC_DIAGNOSTIC',
        'scope': panel.SCOPE, 'manifest_sha256': manifest['content_sha256'], 'execution': manifest['execution'],
        'physical_games_completed': 0, 'accounting': 'Offline reducer starts zero games; source attempt counts added by rank_run.',
        'logical_ranking_games': manifest['logical_ranking_games'],
        'champion': dominators[0] if native and len(dominators) == 1 else None,
        'eligible_champions': dominators if native else [], 'diagnostic_dominators': dominators if not native else [],
        'comparison_decision': 'UNIQUE_DOMINATOR' if len(dominators) == 1 else
                               'TIED_DOMINATORS' if dominators else 'INCOMPARABLE',
        'incomparable_set': incomparable, 'strict_edges': strict_edges,
        'ranking': ranking, 'pairwise': pairwise, 'bootstrap': manifest['bootstrap'],
        'ranking_rule': manifest['ranking_rule'], 'promotion_authority': False,
        'source_identities': {c['id']: c['evaluation_identity'] for c in manifest['candidates']},
        'limits': manifest['limits']})


def control_signature(row: dict) -> str:
    """Repeat controls must agree on all evidence, excluding elapsed wall time."""
    return digest({k: v for k, v in row.items() if k != 'seconds'})


def _load_child(path: Path, expected: dict) -> tuple[list[dict], dict]:
    equal(read(path / 'manifest.json'), expected, 'child manifest (sources/worlds/actors/settings)')
    records = panel.read_journal(path / 'journal.jsonl', expected)
    attempts = panel.read_attempts(path / 'attempts.jsonl', expected)
    if len(records) != len(attempts):
        raise ValueError('uncommitted child attempt; completion UNKNOWN; no replay')
    oracle = panel.legacy_oracle()
    registry = panel.WorldRegistry(panel.World(**{**w, 'shops': tuple(w['shops'])})
                                   for w in expected['registry']['worlds'])
    for i, record in enumerate(records):
        validation = panel.validate_game(record['row'], record['job'], expected,
            registry[record['job']['world']], oracle['rows'][i] if i < 80 else None)
        equal(record['validation'], validation, 'child journal row validation')
    report = panel._report(expected, records, attempts)
    if report['status'] in ('COMPLETE', 'FAILED'):
        equal(read(path / 'summary.json'), report, 'child summary recomputed from journal')
    return records, report


def rank_run(run_dir: Path) -> dict:
    """Reverify every source, attempted game, result, regression gate and repeat."""
    run_dir = Path(run_dir)
    panel._safe_path(run_dir)
    manifest = read(run_dir / 'manifest.json')
    equal(make_manifest(**manifest['settings']), manifest, 'ranking manifest with current sources')
    rows = {c['id']: [] for c in manifest['candidates']}
    reference_controls = None
    inputs = []; native_count = synthetic_count = 0
    for child in manifest['panels']:
        directory = run_dir / 'panels' / child['directory']
        if not directory.is_dir():
            raise ValueError('all complete child panels required')
        records, report = _load_child(directory, child['manifest'])
        if report['status'] != 'COMPLETE' or report['legacy_gate']['matched'] != 80:
            raise ValueError('all complete child panels and exact 80-row gates required')
        if report['execution']['kind'] == 'NATIVE_GAME' and report['legacy_gate']['status'] != 'EXACT_80_ROW_REPLAY':
            raise ValueError('native ranking requires actual native regression gates')
        parent = [r['row'] for r in records if r['job']['phase'] == 'panel' and r['job']['role'] == 'parent']
        signature = [control_signature(row) for row in parent]
        if reference_controls is None:
            reference_controls = signature
            rows['k0006'] = parent
        else:
            equal(signature, reference_controls, 'repeated incumbent controls (cash/actions/identities)')
        rows[child['candidate']] = [r['row'] for r in records
            if r['job']['phase'] == 'panel' and r['job']['role'] == 'candidate']
        native_count += report['physical_games_completed']
        synthetic_count += report['synthetic_attempts']
        inputs.append({'candidate': child['candidate'], 'manifest_sha256': child['manifest']['content_sha256'],
            'journal_sha256': file_sha(directory / 'journal.jsonl'),
            'attempts_sha256': file_sha(directory / 'attempts.jsonl'),
            'summary_sha256': file_sha(directory / 'summary.json')})
    result = rank_rows(manifest, rows)
    result.update(source_panels=inputs, source_physical_games_completed=native_count,
                  source_synthetic_attempts=synthetic_count, repeated_parent_controls='EXACT_MATCH')
    return bound(result)


def ranking_markdown(result: dict) -> str:
    lines = ['# Non-neural policy ranking', '', f"Evidence: {result['status']}. {result['scope']}", '',
        f"Decision: {result['comparison_decision']}; champion: {result['champion'] or 'NOT_NAMED'}.",
        'Cash-margin order is not a ladder ranking or a promotion authorization.', '',
        '| Candidate | Mean margin | Paired 95% CI vs K0006 | Worst world (mean) | Router W/D/L | K0006 W/D/L | Tape0 W/D/L | PASS W/D/L |',
        '| --- | ---: | --- | --- | --- | --- | --- | --- |']
    for row in result['ranking']:
        ci = row['world_bootstrap_ci95_vs_k0006']; worst = row['worst_world']
        wdl = ['{wins}/{draws}/{losses}'.format(**row['per_rival_wdl'][r]) for r in RIVALS]
        lines.append(f"| {row['candidate']} | {row['mean_margin']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
                     f"{worst['world']} ({worst['mean_margin']:.3f}) | " + ' | '.join(wdl) + ' |')
    lines += ['', 'Incomparable / non-strictly-dominated set: ' + ', '.join(result['incomparable_set']) + '.',
              'The JSON contains all pairwise and per-rival intervals, world blocks, source hashes and raw W/D/L.', '']
    return '\n'.join(lines)


def run_ranking(output: Path, *, max_games=4, resume=False, bootstrap_samples=10000,
                bootstrap_seed=20260917, timeout=300.) -> dict:
    """Global bounded orchestration; all games and child journals belong to router_panel."""
    if type(max_games) is not int or not 1 <= max_games <= 100000:
        raise ValueError('positive bounded global max_games required')
    output = Path(output)
    panel._safe_path(output)
    manifest = make_manifest(bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed, timeout=timeout)
    if resume:
        if not output.is_dir():
            raise ValueError('resume requires an existing ranking directory')
    else:
        output.mkdir(parents=True, exist_ok=False)
    for name in ('run.lock', 'manifest.json', 'ranking.json', 'ranking.md'):
        panel._safe_path(output / name)
    with (output / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('ranking is already locked by another writer') from error
        if (output / 'ranking.json').exists():
            raise FileExistsError('existing final ranking is never overwritten')
        if resume:
            equal(read(output / 'manifest.json'), manifest, 'ranking resume manifest')
        else:
            write_new(output / 'manifest.json', manifest)
        # Validate EVERY already-started child before permitting one more game.
        reports = {}
        for child in manifest['panels']:
            directory = output / 'panels' / child['directory']
            panel._safe_path(directory)
            if directory.exists():
                _, reports[child['candidate']] = _load_child(directory, child['manifest'])
        remaining = max_games
        failed = any(r['status'] == 'FAILED' for r in reports.values())
        for child in manifest['panels']:
            if failed or remaining == 0:
                break
            prior = reports.get(child['candidate'])
            if prior and prior['status'] == 'COMPLETE':
                continue
            directory = output / 'panels' / child['directory']
            before = prior['attempt_records'] if prior else 0
            report = panel.run_panel(directory, candidate=child['manifest']['candidate'], parent='k0006',
                max_games=min(remaining, child['manifest']['games_planned'] - before),
                resume=directory.exists(), **manifest['settings'])
            spent = report['attempt_records'] - before
            if not 0 < spent <= remaining:
                raise ValueError('shared harness attempt accounting outside global budget')
            remaining -= spent
            reports[child['candidate']] = report
            failed = report['status'] == 'FAILED'
        complete = len(reports) == len(manifest['panels']) and all(r['status'] == 'COMPLETE' for r in reports.values())
        ranking = rank_run(output) if complete else None
        if ranking is not None:
            write_new(output / 'ranking.json', ranking)
            with (output / 'ranking.md').open('x') as stream:
                stream.write(ranking_markdown(ranking))
        result = bound({'schema': SCHEMA + '-progress',
            'status': ranking['status'] if ranking else 'FAILED' if failed else 'PARTIAL',
            'manifest_sha256': manifest['content_sha256'], 'attempt_records': sum(r['attempt_records'] for r in reports.values()),
            'attempts_this_invocation': max_games - remaining,
            'physical_games_completed': sum(r['physical_games_completed'] for r in reports.values()),
            'synthetic_attempts': sum(r['synthetic_attempts'] for r in reports.values()),
            'games_planned': manifest['games_planned'],
            'children': {n: {'status': r['status'], 'attempt_records': r['attempt_records'], 'failure': r['failure']}
                         for n, r in reports.items()}, 'ranking': ranking['content_sha256'] if ranking else None})
        write_new(output / f"progress-{result['attempt_records']:06d}.json", result)
        return result


def run_smoke(output: Path, *, timeout=120.) -> dict:
    """Exactly four preregistered adapter checks, at most four actual attempts."""
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError('bounded smoke timeout in (0,300] seconds required')
    output = Path(output)
    panel._safe_path(output)
    world = panel.world_registry()['router-dairy']
    refs = [MODULE + ':k0006_factory', 'k0006', MODULE + ':teacher_v5_factory', MODULE + ':compiled_v0_factory']
    actors = {ref: panel.load_actor(ref) for ref in [*refs, 'router']}
    execution = panel.execution_binding(panel.play_game)
    manifest = bound({'schema': SCHEMA + '-smoke', 'scope': panel.SCOPE, 'games_planned': 4,
        'actors': {ref: actor.identity for ref, actor in actors.items()}, 'execution': execution,
        'source_pins': _assets([Path(__file__), Path(panel.__file__), Path(scenarios.__file__),
            Path(k0006.__file__), Path(shop_router.__file__), Path(panel.exact_engine.__file__),
            panel.exact_engine.ENGINE_PATH, panel.exact_engine.ENGINE_CONFIG_PATH, scenarios.CATALOG]),
        'world': world.scenario(), 'world_sha256': world.sha256, 'timeout': float(timeout),
        'jobs': [{'world': world.id, 'seat': 0, 'agent': ref, 'opponent': 'router'} for ref in refs],
        'legacy_gate': 'NOT_RUN; SMOKE_IS_NOT_PANEL_ACCEPTANCE'})
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'manifest.json', manifest)
    rows = []; validations = []
    for index, job in enumerate(manifest['jobs']):
        equal(panel.execution_binding(panel.play_game), execution, 'smoke runner identity before attempt')
        start = write_new(output / f'attempt-{index}.json', {'event': 'STARTED', 'index': index,
            'manifest_sha256': manifest['content_sha256'], 'job': job})
        # Durable intent precedes the world; a killed game is UNKNOWN, never retried here.
        with (output / f'attempt-{index}.json').open('rb') as stream:
            os.fsync(stream.fileno())
        directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        row = panel.play_game(world, actors[job['agent']], actors['router'], 0, timeout)
        row['evidence_kind'] = execution['kind']; row['execution_sha256'] = execution['content_sha256']
        if execution['kind'] != 'NATIVE_GAME':
            row['physical_games_completed'] = 0
        validation = panel.validate_game(row, job, manifest, world)
        write_new(output / f'result-{index}.json', {'row': row, 'validation': validation,
            'attempt_sha256': start['content_sha256'], 'manifest_sha256': manifest['content_sha256']})
        rows.append(row); validations.append(validation)
        if not validation['passed']:
            break
    control = None
    if len(rows) >= 2 and all(v['passed'] for v in validations[:2]):
        control = {'paired_margin': rows[0]['margin'] - rows[1]['margin'],
                   'cash_equal': rows[0]['cash'] == rows[1]['cash'],
                   'joint_actions_equal': rows[0]['actions_sha256'] == rows[1]['actions_sha256']}
    passed = (len(rows) == 4 and all(v['passed'] for v in validations) and control ==
              {'paired_margin': 0., 'cash_equal': True, 'joint_actions_equal': True})
    native = execution['kind'] == 'NATIVE_GAME'
    result = bound({'schema': SCHEMA + '-smoke-summary',
        'status': 'FAILED' if not passed else 'COMPLETE' if native else 'SYNTHETIC_DIAGNOSTIC',
        'manifest_sha256': manifest['content_sha256'], 'execution': execution,
        'physical_games_attempted': len(rows) if native else 0,
        'physical_games_completed': sum(r['physical_games_completed'] for r in rows) if native else 0,
        'synthetic_attempts': 0 if native else len(rows), 'identical_control': control,
        'rows': rows, 'validations': validations, 'legacy_gate': manifest['legacy_gate'],
        'champion': None, 'interpretation': 'Adapter smoke only; no full-panel interval, ranking or promotion.'})
    write_new(output / 'summary.json', result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('dry-run', 'run', 'smoke', 'rank'):
        sub = commands.add_parser(name)
        sub.add_argument('--output', type=Path, required=True)
        if name in ('dry-run', 'run'):
            sub.add_argument('--bootstrap-samples', type=int, default=10000)
            sub.add_argument('--bootstrap-seed', type=int, default=20260917)
            sub.add_argument('--timeout', type=float, default=300.)
        if name == 'run':
            sub.add_argument('--max-games', type=int, default=4)
            sub.add_argument('--resume', action='store_true')
        if name == 'smoke':
            sub.add_argument('--timeout', type=float, default=120.)
        if name == 'rank':
            sub.add_argument('--run-dir', type=Path, required=True)
    args = vars(parser.parse_args(argv)); command = args.pop('command')
    output = args['output']
    try:
        if command == 'dry-run':
            output = args.pop('output')
            result = write_new(output, make_manifest(**args))
        elif command == 'run':
            result = run_ranking(**args)
        elif command == 'smoke':
            result = run_smoke(**args)
        else:
            result = write_new(args['output'], rank_run(args['run_dir']))
        print(json.dumps({'status': result['status'], 'content_sha256': result['content_sha256'],
                          'games_planned': result.get('games_planned'), 'output': str(output)},
                         sort_keys=True), flush=True)
    except (ValueError, FileExistsError, OSError, ImportError, AttributeError, KeyError) as error:
        print(f'REFUSED: {type(error).__name__}: {error}', file=sys.stderr)
        return 2
    if command == 'dry-run':
        return 0
    return 1 if result['status'] == 'FAILED' else 3 if result['status'] == 'PARTIAL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
