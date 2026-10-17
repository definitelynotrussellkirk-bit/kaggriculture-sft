"""D06: a default-off, claim-bound K0006 opening ablation, not a promoted agent.

The builder retains the frozen K0006 file as an exact byte prefix and appends
only metadata and, when explicitly selected, OPEN_UNITS=6. K0006's existing
agent function owns the standard-configuration/action/step guards and all
continuation state. No wrapper may reconstruct the parent on every action.

For the mandatory full panel, build first, set FARM_ROUTER_UPGRADE_ARTIFACT to
that explicit main.py, and use router_panel's ordinary module:factory interface.
The separate four-game smoke below grants NO legacy-gate or promotion certificate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable

from farm_neat.evaluation import k0006
from farm_neat.evaluation.common import bound, canonical, equal, file_sha, read, write_new

ROOT = Path(__file__).resolve().parents[4]
PARENT_PATH = k0006.FOLDER / 'main.py'
PARENT_SHA256 = '4757f3f5b28db8a2f4614bb08993a8a567a7d49bae1ac324fbcfbcc1af60e95e'
MAP_PATH = ROOT / 'work/runs/reviewer-play/router-fallback/knowledge-map/KNOWLEDGE-MAP.json'
MAP_SHA256 = 'db4acd279e0fb44ea17521857bbcf193f658eb0f92fc9793818764ef9d4350a3'
CLAIM = '3754740a86c54137638a77615a19ba8b9a539de0379d47d4b26f1e1f2d16eb7f'
PATCH = 'opening_claim_q6'
ARTIFACT_ENV = 'FARM_ROUTER_UPGRADE_ARTIFACT'
CANDIDATE_REF = 'farm_neat.evaluation.router_upgrade:packaged_factory'
SCHEMA = 'farm-router-upgrade-d06-v1'


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _name(path: Path) -> str:
    path = path.absolute()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _output_path(output) -> Path:
    if output is None or not str(output).strip():
        raise ValueError('explicit output path required')
    path = Path(output).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlinked output or ancestor refused')
    # Every currently frozen evaluation folder declares its provenance here.
    frozen = {p.parent.resolve() for p in (ROOT / 'src/farm_neat/evaluation').glob('*/PROVENANCE.json')}
    frozen.update({k0006.FOLDER.resolve(), (ROOT / 'src/farm_neat/evaluation/shop_router_0909').resolve()})
    if any(path.resolve().is_relative_to(folder) for folder in frozen):
        raise ValueError('output in frozen folder refused')
    return path


def render(patches: Iterable[str] = ()) -> tuple[bytes, dict]:
    """Deterministic source; quantities are read/validated against the owned map."""
    if isinstance(patches, (str, bytes)):
        raise ValueError('patches must be a sequence of registered switch names')
    selected = tuple(patches)
    if selected not in ((), (PATCH,)):
        raise ValueError('unknown or duplicate patch; only D06 is in this task')
    raw = PARENT_PATH.read_bytes()
    equal(_sha(raw), PARENT_SHA256, 'frozen parent bytes')
    equal(file_sha(MAP_PATH), MAP_SHA256, 'D06 knowledge map bytes')
    claim_path = ROOT / 'project/registries/search-ledger' / (CLAIM + '.json')
    equal(file_sha(claim_path), CLAIM, 'D06 quantity claim bytes')
    mapping = read(MAP_PATH, binding=False)
    row = next(r for r in mapping['decision_points'] if r['id'] == 'D06')
    equal(row['patch']['switch'], PATCH, 'map switch')
    constant, = row['patch']['new_tuning_constants']
    equal((constant['basis'], constant['claim'], constant['value']), ('CLAIM', CLAIM, 6), 'map quantity')
    metadata = {'schema': SCHEMA, 'parent_sha256': PARENT_SHA256,
                'knowledge_map_sha256': MAP_SHA256, 'map_row': 'D06', 'claim': CLAIM,
                'patches': list(selected), 'candidate_quantity': constant['value'],
                'promotion_authority': False}
    postlude = (b'\n\n# ROUTER-UPGRADE D06: additive wrapper configuration; upstream bytes unchanged.\n'
                b'# opening_claim_q6 is experimental, default OFF; no strength claim.\n'
                + ('ROUTER_UPGRADE_METADATA = ' + repr(metadata) + '\n').encode('utf-8'))
    if selected:
        postlude += b'OPEN_UNITS = 6\n'
    return raw + postlude, metadata


def build(output, *, patches: Iterable[str] = ()) -> dict:
    """Exclusive explicit output; never rewrites a frozen artifact or prior run."""
    path = _output_path(output)
    if path.suffix != '.py':
        raise ValueError('explicit .py output file required')
    sidecar = path.with_suffix('.build.json')
    for member in (path, sidecar):
        _output_path(member)
        if member.exists():
            raise FileExistsError(str(member))
    raw, metadata = render(patches)
    compile(raw, '<router-upgrade-preflight>', 'exec')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return write_new(sidecar, {**metadata, 'artifact': _name(path), 'artifact_sha256': _sha(raw),
                              'artifact_bytes': len(raw), 'builder_sha256': file_sha(__file__),
                              'parent_prefix_bytes': len(PARENT_PATH.read_bytes()),
                              'upstream_notices': 'All frozen parent bytes, including embedded Apache-2.0 notices, retained verbatim.'})


class PackagedActor:
    """Exec exactly the selected file once per game, without __file__ or path loads.

    Validation is evaluator-side. The resulting main.py needs neither this module
    nor the repository, knowledge map, build sidecar or external actions.json.
    """
    pending = False

    def __init__(self, artifact):
        if artifact is None or not str(artifact).strip():
            raise ValueError('explicit packaged artifact required')
        path = Path(artifact).absolute()
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('symlinked artifact refused')
        raw = path.read_bytes()
        metadata = None
        for switches in ((), (PATCH,)):
            expected, candidate = render(switches)
            if raw == expected:
                metadata = candidate
                break
        if metadata is None:
            raise ValueError('artifact is not an exact default-off or D06 build of the pinned parent')
        self.identity = bound({**metadata, 'kind': 'EXACT_PACKAGED_SOURCE_EXEC',
            'artifact_sha256': _sha(raw),
            'assets': {_name(path): _sha(raw), _name(Path(__file__)): file_sha(__file__),
                       'src/farm_neat/evaluation/common.py': file_sha(ROOT / 'src/farm_neat/evaluation/common.py'),
                       'src/farm_neat/evaluation/k0006.py': file_sha(k0006.__file__),
                       _name(PARENT_PATH): PARENT_SHA256, _name(MAP_PATH): MAP_SHA256,
                       'project/registries/search-ledger/' + CLAIM + '.json': CLAIM},
            'actor_inputs': 'normal observation and allowlisted public configuration only',
            'runtime_assets': 'self-contained artifact only; remaining assets bind evaluator validation'})
        self.namespace = {'__name__': 'router_upgrade_' + _sha(raw)[:16]}
        exec(compile(raw, str(path), 'exec'), self.namespace)
        self.actor = self.namespace['agent']

    def act(self, observation, configuration=None):
        return self.actor(observation, configuration)


def packaged_factory() -> PackagedActor:
    """router_panel module:factory; no implicit default path or wrapper fallback."""
    artifact = os.environ.get(ARTIFACT_ENV)
    if not artifact:
        raise ValueError(f'explicit artifact required: set {ARTIFACT_ENV}')
    return PackagedActor(artifact)


class _ArtifactFactory:
    """Immutable path selection for a smoke, avoiding process-global env changes."""
    def __init__(self, artifact):
        self.artifact = Path(artifact).absolute()

    def __call__(self):
        return PackagedActor(self.artifact)


def smoke_plan(artifact, *, rival='router', timeout=300.):
    """Four registered cells; no full-panel runner/gate is bypassed or modified."""
    from farm_neat.evaluation import router_panel as panel
    if rival not in ('router', 'k0006'):
        raise ValueError('D06 smoke rival must be router or k0006')
    base, actors, registry = panel.make_plan('k0006', 'k0006', timeout=timeout)
    actors[CANDIDATE_REF] = panel.register_actor(CANDIDATE_REF, _ArtifactFactory(artifact))
    declared = actors[CANDIDATE_REF].identity['declared']
    if declared['patches'] != [PATCH]:
        raise ValueError('D06 smoke requires the explicit q6 candidate artifact')
    jobs = [{'phase': 'smoke', 'role': role, 'world': 'router-dairy', 'seat': seat,
             'agent': CANDIDATE_REF if role == 'candidate' else 'k0006', 'opponent': rival}
            for seat in (0, 1) for role in ('parent', 'candidate')]
    plan = bound({'schema': SCHEMA + '-smoke', 'scope': 'EXPOSED_ONE_WORLD_DIAGNOSTIC_NOT_ACCEPTANCE_PANEL',
        'candidate': CANDIDATE_REF, 'parent': 'k0006', 'patches': [PATCH], 'map_row': 'D06', 'claim': CLAIM,
        'actors': {name: actor.identity for name, actor in actors.items()},
        'registry': registry.document(), 'source_pins': {**base['source_pins'], _name(Path(__file__)): file_sha(__file__)},
        'execution': base['execution'], 'engine': base['engine'], 'python': sys.version,
        'timeout_seconds': float(timeout), 'jobs': jobs, 'games_planned': 4,
        'legacy_gate': 'NOT_RUN_NO_BYPASS_OR_CERTIFICATE', 'promotion_authority': False,
        'bootstrap_interval': None, 'held_out_used': False,
        'unrun_required_coverage': 'Other rival plus both seats; mandatory 80 controls; complete 72-world registered panel.'})
    return plan, actors, registry


def diagnostic_decision(differences) -> str:
    values = list(differences)
    if not values:
        return 'NOT_EVALUATED_KEEP_DISABLED'
    if all(v < 0 for v in values):
        return 'REJECT_FOR_DEPLOYMENT_NEGATIVE_SMOKE'
    if all(v > 0 for v in values):
        return 'POSITIVE_SMOKE_ONLY_KEEP_DISABLED'
    return 'INCONCLUSIVE_SMOKE_KEEP_DISABLED'


def _durable_json(path, value):
    document = write_new(path, value)
    with Path(path).open('rb') as stream:
        os.fsync(stream.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return document


def smoke(output, *, artifact, rival='router', timeout=300.) -> dict:
    """Exactly four attempts maximum; no retries/resume and no CI from one world.

    Durable starts count even on interruption. An existing directory is refused;
    the caller remains responsible for the cross-invocation four-game allowance.
    The candidate is the packaged file, not a different wrapper implementation.
    """
    from farm_neat.evaluation import router_panel as panel
    out = _output_path(output)
    plan, actors, registry = smoke_plan(artifact, rival=rival, timeout=timeout)
    out.mkdir(parents=True, exist_ok=False)
    _durable_json(out / 'manifest.json', plan)
    oracle = panel.legacy_oracle()
    expected = {(r['agent'], r['opponent'], r['world'], r['seat']): r for r in oracle['rows']}
    records = []; started = 0; start_wall = time.monotonic(); start_cpu = time.process_time()
    for index, job in enumerate(plan['jobs']):
        equal(panel.execution_binding(panel.play_game), plan['execution'], 'native smoke runner')
        if plan['execution']['kind'] != 'NATIVE_GAME':
            raise ValueError('smoke requires the unmodified native harness runner')
        attempt = _durable_json(out / f'{index:02d}-start.json', {
            'manifest_sha256': plan['content_sha256'], 'index': index, 'job': job, 'event': 'STARTED'})
        started += 1
        row = panel.play_game(registry[job['world']], actors[job['agent']], actors[job['opponent']],
                              job['seat'], timeout)
        validation = panel.validate_game(row, job, plan, registry[job['world']])
        control_expected = expected.get((job['agent'], rival, job['world'], job['seat']))
        control_replay = (panel.validate_game(row, job, plan, registry[job['world']], control_expected)
                          if job['role'] == 'parent' and control_expected else None)
        record = _durable_json(out / f'{index:02d}-result.json', {
            'manifest_sha256': plan['content_sha256'], 'attempt_sha256': attempt['content_sha256'],
            'job': job, 'row': row, 'validation': validation, 'historical_control_replay': control_replay})
        records.append(record)
        print(canonical({'index': index, 'role': job['role'], 'seat': job['seat'], 'status': row['status'],
                         'cash': row['cash'], 'margin': row['margin'], 'min_rival_cash': row['min_rival_cash']}), flush=True)
        if not validation['passed'] or (control_replay and not control_replay['passed']):
            break
    pairs = []
    for seat in (0, 1):
        by_role = {r['job']['role']: r['row'] for r in records
                   if r['job']['seat'] == seat and r['validation']['passed']}
        if set(by_role) == {'candidate', 'parent'}:
            a, b = by_role['candidate'], by_role['parent']
            pairs.append({'seat': seat, 'world': 'router-dairy', 'rival': rival,
                'candidate_own_cash': a['cash'][seat], 'candidate_rival_cash': a['cash'][1-seat],
                'parent_own_cash': b['cash'][seat], 'parent_rival_cash': b['cash'][1-seat],
                'candidate_margin': a['margin'], 'parent_margin': b['margin'],
                'delta_own_cash': a['cash'][seat] - b['cash'][seat],
                'delta_rival_cash': a['cash'][1-seat] - b['cash'][1-seat],
                'paired_margin': a['margin'] - b['margin'], 'candidate_wdl': a['wdl'], 'parent_wdl': b['wdl']})
    valid = len(records) == 4 and all(r['validation']['passed'] and
        (r['historical_control_replay'] is None or r['historical_control_replay']['passed']) for r in records)
    complete_rows = [r['row'] for r in records if r['row']['status'] == 'COMPLETE']
    candidate_rows = [r['row'] for r in records if r['job']['role'] == 'candidate' and r['row']['status'] == 'COMPLETE']
    return _durable_json(out / 'summary.json', {'schema': SCHEMA + '-smoke-summary',
        'manifest_sha256': plan['content_sha256'], 'status': 'COMPLETE_DIAGNOSTIC' if valid else 'FAILED',
        'evidence_kind': 'NATIVE_GAME', 'physical_games_attempted': started,
        'physical_games_completed': sum(r['row']['physical_games_completed'] for r in records),
        'pairs': pairs, 'bootstrap_interval': None, 'independent_worlds': 1 if valid else 0,
        'min_rival_cash': min((r['min_rival_cash'] for r in complete_rows), default=None),
        'candidate_min_rival_cash': min((r['min_rival_cash'] for r in candidate_rows), default=None),
        'rival_cash_trough': min((r['rival_cash_trough'] for r in complete_rows), default=None),
        'decision': diagnostic_decision([p['paired_margin'] for p in pairs]) if valid else 'FAILED_KEEP_DISABLED',
        'kept_patches': [], 'promotion_authority': False, 'legacy_gate': plan['legacy_gate'],
        'historical_control_rows_matched': sum(bool(r['historical_control_replay'] and r['historical_control_replay']['passed']) for r in records),
        'wall_seconds': time.monotonic() - start_wall, 'cpu_seconds': time.process_time() - start_cpu,
        'limits': ['One exposed world is not a confidence interval or a ladder estimate.',
                   'Four-game smoke does not open or replace the full harness legacy gate.',
                   'No keep decision without the complete paired full-world criterion.',
                   'Other frozen rival and full panel NOT YET RUN.']})


def probes(output) -> dict:
    """Small adverse native market fixtures; zero initialized worlds/full games."""
    from farm_neat.evaluation import sale_order, scenarios
    from farm_neat.game import exact_engine
    out = _output_path(output)
    tape_path = ROOT / 'src/farm_neat/evaluation/shop_router_0909/actions.json'
    tape_word = read(tape_path, binding=False)[0][0]['market']
    rivals = {'router_tape0': tape_word,
              'k0006_q50': [['BUY_PRODUCT', 'WHEAT', 50], ['SELL', 'WHEAT', 50]],
              'wrong_forecast_q1': [['BUY_PRODUCT', 'WHEAT', 1], ['SELL', 'WHEAT', 1]], 'pass': []}
    inventory = {p: 10000 for p in exact_engine.load_engine().PRODUCTS}
    rows = []
    for rival, word in rivals.items():
        for seat in (0, 1):
            results = {}
            for quantity in (6, 50):
                order = [['BUY_PRODUCT', 'WHEAT', quantity], ['SELL', 'WHEAT', quantity]]
                results[str(quantity)] = sale_order.native_outcome(inventory, {}, {}, order, word, seat=seat)
            a, b = results['6'], results['50']
            rows.append({'rival': rival, 'rival_word': word, 'seat': seat, 'q6': a, 'q50': b,
                         'paired_margin': a['margin'] - b['margin'],
                         'delta_own_cash': a['own_cash'] - b['own_cash'],
                         'delta_rival_cash': a['other_cash'] - b['other_cash']})
    return _durable_json(out, {'schema': SCHEMA + '-market-probes', 'map_row': 'D06', 'claim': CLAIM,
        'evidence_kind': 'ISOLATED_NATIVE_MARKET_PHASE', 'physical_games_attempted': 0,
        'physical_games_completed': 0, 'native_phase_calls': 16, 'rows': rows,
        'fixtures': {'inventory_per_product': 10000, 'cash_per_seat': 3000, 'starting_sheds': 'empty'},
        'sources': {_name(Path(__file__)): file_sha(__file__), _name(tape_path): file_sha(tape_path),
                    _name(Path(sale_order.__file__)): file_sha(sale_order.__file__),
                    _name(exact_engine.ENGINE_PATH): file_sha(exact_engine.ENGINE_PATH)},
        'interpretation': 'Local component only. Both cash accounts retained; no continuation-equivalence or strength claim.'})
