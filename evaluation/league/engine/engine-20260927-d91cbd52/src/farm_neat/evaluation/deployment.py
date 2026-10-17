"""Build a source-bound complete-policy tar.gz and cold CPU full-game preflight.

Use the pinned interpreter and storage supervisor. No optimizer or uploader is
included. CPU quantization is a new deployment profile with no parity claim.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
TAPE_SCHEMA = 'farmneat-fixed-action-tape-deployment-v1'
RESPONSIVE_SCHEMA = 'farmneat-k0006-responsive-script-deployment-v1'
RESPONSIVE_KIND = 'FROZEN_RESPONSIVE_SCRIPTED_POLICY'
# One frozen actor, not an arbitrary-script interface. Hashes come from K0006's
# canonical identity and retained source-copy/panel receipts, not archive claims.
RESPONSIVE_ASSETS = {
    'main.py': '4757f3f5b28db8a2f4614bb08993a8a567a7d49bae1ac324fbcfbcc1af60e95e',
    'PROVENANCE.json': 'fa559183bc2370822f53dd2a97452d8e5005bba20fb029fa6aca58e52f1b4ad5'}
RESPONSIVE_IDENTITY = 'bd79399c4fd6f0f11d42a8d18c9986a7629128ca66cbbdb8d3514e2609d352b0'
RESPONSIVE_MEMBERS = {*RESPONSIVE_ASSETS, 'DEPLOYMENT.json', 'MANIFEST.json'}
RESPONSIVE_PANEL = 'work/runs/reviewer-play/router-fallback/K0006-PANEL.json'
RESPONSIVE_PANEL_SHA = '57ef9ea5c7aeb1dc212a15e113747d8b7ae00d4db1a560098fb02d9fe65cb577'
RESPONSIVE_CATALOG_SHA = '85611189bfd2b1790cb7443e2f0771b208b33eac47d9d17184774f4abd48f4b3'
RESPONSIVE_CELLS = tuple((world, seat) for world in ('router-no-yarn', 'router-yarn') for seat in (0, 1))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def checked_name(name):
    p = PurePosixPath(name)
    if not name or name == '.' or p.is_absolute() or str(p) != name or any(v in ('..', '.') for v in p.parts) or '\\' in name:
        raise ValueError('canonical archive relative name required')
    return name


def source_members():
    """Static runtime import closure plus exact files read by existing identities."""
    from farm_neat.tooling import reachability
    from farm_neat.policy import network, fast_profile, plan_observation
    from farm_neat.evaluation import deployment_codec
    known = reachability.modules()
    pinned = {**network.sources(), **fast_profile.identity()['source_pins'],
              **plan_observation.identity()['source_pins'], **deployment_codec.decoder_identity()['source_pins']}
    pending = ['farm_neat.evaluation.deployment_runtime']
    pending += [reachability.module_name(ROOT/p) for p in pinned if p.startswith('src/') and p.endswith('.py')]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen or name not in known:
            continue
        seen.add(name)
        for target in reachability.imports(known[name]) | {name.rsplit('.', 1)[0]}:
            while target:
                if target in known:
                    pending.append(target)
                    break
                target = target.rsplit('.', 1)[0] if '.' in target else ''
    paths = set(pinned) | {known[n].relative_to(ROOT).as_posix() for n in seen}
    # The inherited policy identity includes these source-loaded engine assets.
    if 'farm_neat.game.exact_engine' in seen:
        paths.update(('src/farm_neat/game/kaggriculture_engine_reference.py', 'src/farm_neat/game/kaggriculture.json'))
    for p in paths:
        f = ROOT / checked_name(p)
        if f.is_symlink() or not f.is_file() or f.resolve() != f:
            raise ValueError('regular runtime dependency required: '+p)
    return {p: sha(ROOT/p) for p in sorted(paths)}


def main_source(settings_sha):
    return ('''"""Full farmNEAT economic actor; CPU FP32 with quantized expert storage."""
import os as _os
import sys as _sys
from pathlib import Path as _Path
_os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    _os.environ[_key] = '1'
_root = _Path('/kaggle_simulations/agent')
if not (_root / 'DEPLOYMENT.json').is_file():
    _root = _Path(__file__).resolve().parent if '__file__' in globals() else _Path.cwd()
_sys.path.insert(0, str(_root / 'src'))
from farm_neat.evaluation.deployment_runtime import Agent as _Agent
''' + f'_resident = _Agent(_root, expected_settings_sha256={settings_sha!r})\n'
        + 'def agent(observation, configuration=None):\n    return _resident(observation, configuration)\n').encode()


def tape_main_source(settings_sha):
    return ('''"""Fixed external tape baseline; no neural inference or responsive economic edits."""
import copy
import hashlib
import json
from pathlib import Path
import sys
root = Path('/kaggle_simulations/agent')
if not (root / 'DEPLOYMENT.json').is_file():
    root = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
''' + f"assert hashlib.sha256((root / 'DEPLOYMENT.json').read_bytes()).hexdigest() == {settings_sha!r}\n"
        + '''settings = json.loads((root / 'DEPLOYMENT.json').read_text())
assert hashlib.sha256((root / 'TAPE.json').read_bytes()).hexdigest() == settings['tape_sha256']
tape = json.loads((root / 'TAPE.json').read_text())
assert len(tape) == 719
def agent(observation, configuration=None):
    step = observation['step']
    if type(step) is not int or not 0 <= step < len(tape):
        raise ValueError('exact nonterminal tape step required')
    if step == 0:
        print(json.dumps({'schema': settings['schema'], 'event': 'EPISODE_START',
            'seat': observation['player'], 'tape_sha256': settings['tape_sha256'],
            'tape_index': settings['tape_index'], 'learned_policy': False}, sort_keys=True), file=sys.stderr, flush=True)
    return copy.deepcopy(tape[step])
''').encode()


def write_tar(path, members):
    """One deterministic regular-file archive writer for all actor profiles."""
    with Path(path).open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w|') as archive:
            for name, value in sorted(members.items()):
                info = tarfile.TarInfo(checked_name(name)); info.mode, info.mtime = 0o644, 0
                info.size = value.stat().st_size if isinstance(value, Path) else len(value)
                with value.open('rb') if isinstance(value, Path) else io.BytesIO(value) as stream:
                    archive.addfile(info, stream)


def build_tape(*, output, tape_index=0):
    from farm_neat.evaluation import shop_router
    output = Path(output).absolute()
    if output.exists() or not output.is_relative_to(ROOT / 'kagglesub') or output.resolve() != output:
        raise ValueError('new output in canonical kagglesub required')
    snapshot = shop_router.identity()
    source = shop_router.FOLDER / 'actions.json'
    source_sha = sha(source)
    tapes = json.loads(source.read_text())
    if type(tape_index) is not int or not 0 <= tape_index < 13 or len(tapes) != 13:
        raise ValueError('one declared frozen tape index required')
    tape = tapes[tape_index]
    if len(tape) != 719:
        raise ValueError('complete original tape required')
    tape_bytes = encoded(tape)
    settings = {'schema': TAPE_SCHEMA, 'actor_kind': 'FIXED_EXTERNAL_ACTION_TAPE',
        'tape_index': tape_index, 'steps': 719, 'tape_sha256': hashlib.sha256(tape_bytes).hexdigest(),
        'source_actions_sha256': source_sha, 'snapshot': snapshot, 'learned_policy': False,
        'transformation': 'Select one original tape, serialize its unchanged actions; no wrapper, resampling or economic edits.'}
    payload = {'TAPE.json': tape_bytes, 'DEPLOYMENT.json': encoded(settings),
        'LICENSE.txt': (shop_router.FOLDER / 'LICENSE.txt').read_bytes(),
        'PROVENANCE.json': (shop_router.FOLDER / 'PROVENANCE.json').read_bytes(),
        'NOTICE.txt': b'Action tape from yhay81/shop-router-0909, Apache-2.0. See LICENSE.txt and PROVENANCE.json.\nModified packaging: select one unchanged tape; farmNEAT supplies the static playback entry point.\n'}
    settings_sha = hashlib.sha256(payload['DEPLOYMENT.json']).hexdigest()
    payload['main.py'] = tape_main_source(settings_sha)
    payload['MANIFEST.json'] = encoded({'schema': 'farmneat-deployment-archive-v1',
        'actor_kind': 'FIXED_EXTERNAL_ACTION_TAPE', 'deployment_sha256': settings_sha,
        'tape_sha256': settings['tape_sha256'], 'tape_index': tape_index, 'learned_policy': False,
        'members': {n: {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()} for n, b in payload.items()}})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        write_tar(temporary, payload)
        if sum(map(len, payload.values())) > 1_000_000 or sha(source) != source_sha:
            raise ValueError('bounded unchanged tape payload required')
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result = {'schema': 'farmneat-fixed-tape-build-v1', 'archive': str(output.relative_to(ROOT)),
        'archive_sha256': sha(output), 'archive_bytes': output.stat().st_size,
        'expanded_bytes': sum(map(len, payload.values())), 'tape_sha256': settings['tape_sha256'],
        'tape_index': tape_index, 'source_actions_sha256': source_sha,
        'builder_sha256': sha(Path(__file__)), 'learned_policy': False, 'preflight': 'REQUIRED',
        'hosted_acceptance': 'UNVERIFIED', 'retention': {'purpose': 'fixed tape deployment baseline',
            'remove_when': 'superseded without active baseline or hosted-feedback dependencies'}}
    Path(str(output) + '.json').write_bytes(encoded(result))
    return result



def _responsive_settings(identity):
    body = {k: v for k, v in identity.items() if k != 'content_sha256'}
    if (identity.get('content_sha256') != RESPONSIVE_IDENTITY or
            hashlib.sha256(encoded(body)[:-1]).hexdigest() != RESPONSIVE_IDENTITY):
        raise ValueError('canonical K0006 actor identity required')
    return {'schema': RESPONSIVE_SCHEMA, 'actor_kind': RESPONSIVE_KIND,
        'actor_identity': identity, 'builder_sha256': sha(Path(__file__)),
        'source_sha256': RESPONSIVE_ASSETS, 'learned_policy': False,
        'observation_blind_tape': False, 'transformation': 'NONE; exact frozen main.py and provenance bytes',
        'runtime_dependencies': {'standard_library_only': True, 'third_party': []}}


def _responsive_manifest(payload, settings):
    return {'schema': 'farmneat-deployment-archive-v1', 'actor_kind': RESPONSIVE_KIND,
        'deployment_sha256': hashlib.sha256(payload['DEPLOYMENT.json']).hexdigest(),
        'actor_identity_sha256': RESPONSIVE_IDENTITY, 'builder_sha256': settings['builder_sha256'],
        'members': {n: {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
                    for n, b in payload.items() if n != 'MANIFEST.json'}}


def _check_responsive_payload(payload):
    """Validate from trusted profile pins, not a self-asserted inventory alone.

    The caller's expected archive SHA authenticates MANIFEST.json itself. Builder
    equality deliberately requires rebuilding after deployment.py changes.
    """
    if set(payload) != RESPONSIVE_MEMBERS or sum(map(len, payload.values())) > 1_000_000:
        raise ValueError('closed bounded K0006 inventory required')
    if any(hashlib.sha256(payload[n]).hexdigest() != h for n, h in RESPONSIVE_ASSETS.items()):
        raise ValueError('unchanged K0006 source/provenance bytes required')
    try:
        settings = _responsive_settings(json.loads(payload['DEPLOYMENT.json'])['actor_identity'])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('complete K0006 deployment identity required') from error
    # Byte comparison also rejects duplicate keys and ambiguous metadata encodings.
    if payload['DEPLOYMENT.json'] != encoded(settings):
        raise ValueError('responsive settings or builder identity mismatch')
    if payload['MANIFEST.json'] != encoded(_responsive_manifest(payload, settings)):
        raise ValueError('responsive closed manifest identity mismatch')
    return settings


def _responsive_directory(root):
    root = Path(root).absolute()
    if root.resolve() != root or not root.is_dir():
        raise ValueError('regular responsive archive directory required')
    paths = list(root.iterdir())
    if {p.name for p in paths} != RESPONSIVE_MEMBERS or any(
            p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError('closed regular responsive archive inventory required')
    if sum(p.stat().st_size for p in paths) > 1_000_000:
        raise ValueError('bounded responsive archive required')
    payload = {p.name: p.read_bytes() for p in paths}
    return payload, _check_responsive_payload(payload)


def build_responsive(*, output):
    """Package only canonical K0006; no behavior wrapper, model or installer."""
    from farm_neat.evaluation import k0006
    output = Path(output).absolute()
    if (output.exists() or output.resolve() != output or
            not output.is_relative_to(ROOT/'kagglesub') or Path(str(output)+'.json').exists()
            or Path(str(output)+'.json').is_symlink()):
        raise ValueError('new output in canonical kagglesub required')
    settings = _responsive_settings(k0006.identity())
    source = ROOT/'src/farm_neat/evaluation/k0006_v43'
    payload = {}
    for name in RESPONSIVE_ASSETS:
        path = source/name
        if path.resolve() != path or not path.is_file():
            raise ValueError('regular frozen K0006 source required')
        payload[name] = path.read_bytes()
    payload['DEPLOYMENT.json'] = encoded(settings)
    payload['MANIFEST.json'] = encoded(_responsive_manifest(payload, settings))
    _check_responsive_payload(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    if temporary.exists() or temporary.is_symlink():
        raise ValueError('new responsive build temporary required')
    try:
        write_tar(temporary, payload)
        if (temporary.stat().st_size > 1_000_000 or settings['builder_sha256'] != sha(__file__) or
                any(sha(source/n) != h for n, h in RESPONSIVE_ASSETS.items())):
            raise ValueError('bounded unchanged responsive inputs required')
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result = {'schema': 'farmneat-responsive-script-build-v1', 'actor_kind': RESPONSIVE_KIND,
        'archive': str(output.relative_to(ROOT)), 'archive_sha256': sha(output),
        'archive_bytes': output.stat().st_size, 'expanded_bytes': sum(map(len, payload.values())),
        'actor_identity_sha256': RESPONSIVE_IDENTITY, 'builder_sha256': settings['builder_sha256'],
        'runtime_dependencies': settings['runtime_dependencies'], 'learned_policy': False,
        'preflight': 'REQUIRED', 'hosted_acceptance': 'UNVERIFIED',
        'retention': {'purpose': 'unchanged responsive competition-insurance artifact',
            'remove_when': 'superseded without active fallback or preflight dependencies'}}
    Path(str(output)+'.json').write_bytes(encoded(result))
    return result


@contextmanager
def responsive_imports(imported):
    """Measure/check imports only during frozen actor execution, not evaluation.

    This is a dependency assertion on authenticated code, not a sandbox for
    arbitrary untrusted Python. The fresh -I -S process supplies isolation.
    """
    original = builtins.__import__
    def checked(name, globals=None, locals=None, fromlist=(), level=0):
        package = (globals or {}).get('__package__', '') if level else name
        top = package.split('.')[0]
        if top not in sys.stdlib_module_names and top not in sys.builtin_module_names:
            raise ValueError('standard-library actor dependency required: '+name)
        imported.add(top)
        return original(name, globals, locals, fromlist, level)
    builtins.__import__ = checked
    try:
        yield
    finally:
        builtins.__import__ = original


def cold_responsive(root):
    """Import exact entrypoint bytes before admitting any project/evaluator path."""
    payload, _ = _responsive_directory(root)
    if (not sys.flags.isolated or not sys.flags.no_site or
            any(n == 'farm_neat' or n.startswith('farm_neat.') for n in sys.modules)):
        raise ValueError('fresh isolated -I -S process before project imports required')
    imported, before = set(), set(sys.modules)
    namespace = {'__name__': 'k0006_' + RESPONSIVE_IDENTITY[:16]}
    started = time.perf_counter()
    with responsive_imports(imported):
        exec(compile(payload['main.py'], 'main.py', 'exec'), namespace)
    cold = time.perf_counter()-started
    nonstdlib = sorted({n.split('.')[0] for n in set(sys.modules)-before}
                       - sys.stdlib_module_names - set(sys.builtin_module_names))
    if nonstdlib or not callable(namespace.get('agent')):
        raise ValueError('standalone standard-library responsive agent required')
    return namespace['agent'], {'cold_seconds_including_import': cold,
        'actor_imports': sorted(imported), 'third_party_actor_imports': nonstdlib,
        'isolated': bool(sys.flags.isolated), 'no_site': bool(sys.flags.no_site),
        'project_imported_before_actor': False, 'python': sys.version.split()[0]}


def responsive_controls():
    """The four exposed source-receipt cells, frozen before any archive game."""
    for name, expected in ((RESPONSIVE_PANEL, RESPONSIVE_PANEL_SHA),
                           ('project/configs/scenarios.json', RESPONSIVE_CATALOG_SHA)):
        path = ROOT/name
        if path.resolve() != path or sha(path) != expected:
            raise ValueError('responsive preflight source control changed: '+name)
    panel = json.loads((ROOT/RESPONSIVE_PANEL).read_bytes())
    if sha(ROOT/'src/farm_neat/evaluation/scenarios.py') != panel['sources']['scenarios']:
        raise ValueError('responsive source-control scenario implementation changed')
    if panel['identities']['k0006'] != RESPONSIVE_IDENTITY:
        raise ValueError('responsive panel actor mismatch')
    cases = []
    for world, seat in RESPONSIVE_CELLS:
        rows = [r for r in panel['rows'] if (r['agent'], r['opponent'], r['world'], r['seat'])
                == ('k0006', 'router', world, seat)]
        if len(rows) != 1:
            raise ValueError('one exact retained source control per cell required')
        expected = {k: rows[0][k] for k in ('actions_sha256', 'cash', 'status', 'margin', 'wdl')}
        cases.append({'world': world, 'seat': seat, 'expected': expected,
            'opponent_identity_sha256': panel['identities']['router'],
            'expected_result_sha256': hashlib.sha256(encoded(expected)[:-1]).hexdigest()})
    return cases

def build(checkpoint, *, model_sha256, output, command_budget, temperature, lane_seeds):
    from farm_neat.policy import network, fast_profile, plan_observation
    from farm_neat.evaluation import deployment_codec as codec, deployment_runtime as runtime
    import torch
    output = Path(output).absolute()
    if output.exists() or not output.is_relative_to(ROOT/'kagglesub') or output.resolve() != output:
        raise ValueError('new output in canonical kagglesub required')
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_relative_to(ROOT/'work'):
        raise ValueError('retained local checkpoint required')
    model_path = checkpoint/'model.pt'
    settings = {'schema': runtime.SCHEMA, 'decision_steps': list(runtime.STEPS),
        'command_budget': command_budget, 'temperature': temperature, 'diagnostic_lane_seeds': lane_seeds,
        'sampling_rng': 'PCG64_FRESH_OS_ENTROPY_PER_EPISODE', 'diagnostics': runtime.diagnostic_profile()}
    runtime.validate_settings(settings)
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only export required')
    torch.set_num_threads(1)
    graph, original = network.load_model(model_path, device='cpu', expected_file_sha256=model_sha256)
    if original['optimizer_steps'] < 1 or original['architecture'] != json.loads(json.dumps(network.architecture())):
        raise ValueError('trained full-size retained actor required')
    pins = source_members()
    payload, manifest = codec.encode_state(graph.state_dict(), original,
        original_checkpoint_sha256=model_sha256, expected_identity=codec.identity())
    del graph
    byte_members = {'MODEL.bin': payload, 'MODEL.bin.json': encoded(manifest)}
    settings.update(deployment_sha256=manifest['content_sha256'], original_model_sha256=model_sha256,
        optimizer_age_provenance=original['optimizer_steps'], fast_profile=fast_profile.identity(),
        plan_adapter=plan_observation.identity(),
        numerical_profile='GROUP64_EXPERT_INT8_STORAGE_DECODED_FP32_FAST_CPU_B1_BUCKET32_PCG64',
        cpu_cuda_parity_claim=False, original_quantized_parity_claim=False,
        member_sha256={**pins, **{n:hashlib.sha256(b).hexdigest() for n,b in byte_members.items()}})
    byte_members['DEPLOYMENT.json'] = encoded(settings)
    byte_members['main.py'] = main_source(hashlib.sha256(byte_members['DEPLOYMENT.json']).hexdigest())
    members = {**{n: ROOT/n for n in pins}, **byte_members}
    inventory = {n: {'sha256': sha(v) if isinstance(v, Path) else hashlib.sha256(v).hexdigest(),
                     'bytes': v.stat().st_size if isinstance(v, Path) else len(v)} for n,v in members.items()}
    byte_members = None
    members['MANIFEST.json'] = encoded({'schema': 'farmneat-deployment-archive-v1', 'members': inventory,
        'original_model_sha256': model_sha256, 'deployment_sha256': manifest['content_sha256'],
        'archive_cap_bytes': codec.ARCHIVE_CAP_BYTES, 'hosted_acceptance': 'UNVERIFIED',
        'retention': {'purpose': 'current candidate submission', 'remove_when': 'successor verified and no active submission dependency'}})
    expanded = sum(v.stat().st_size if isinstance(v, Path) else len(v) for v in members.values())
    if expanded > codec.ARCHIVE_CAP_BYTES:
        raise ValueError('expanded deployment exceeds conservative 100MB cap')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    try:
        write_tar(temporary, members)
        if temporary.stat().st_size > codec.ARCHIVE_CAP_BYTES:
            raise ValueError('compressed deployment exceeds conservative 100MB cap')
        if sha(model_path) != model_sha256 or any(sha(ROOT/n) != h for n,h in pins.items()):
            raise ValueError('checkpoint or inference closure changed during build')
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result = {'schema': 'farmneat-deployment-build-v1', 'archive': str(output.relative_to(ROOT)),
        'archive_sha256': sha(output), 'archive_bytes': output.stat().st_size, 'expanded_bytes': expanded,
        'members': len(members), 'checkpoint': str(checkpoint.relative_to(ROOT)),
        'original_model_sha256': model_sha256, 'deployment_sha256': manifest['content_sha256'],
        'optimizer_age': original['optimizer_steps'], 'command_budget': command_budget, 'temperature': temperature,
        'preflight': 'REQUIRED', 'hosted_acceptance': 'UNVERIFIED', 'tensor_encoding': manifest['sizing'],
        'diagnostics': runtime.diagnostic_profile()}
    Path(str(output)+'.json').write_bytes(encoded(result))
    return result


def extract(archive_path, expected_sha, destination):
    """Authenticate all regular members and refuse unsafe/extra/missing entries."""
    if sha(archive_path) != expected_sha or Path(archive_path).stat().st_size > 100_000_000:
        raise ValueError('bounded exact archive required')
    destination = Path(destination).absolute()
    if destination.resolve() != destination:
        raise ValueError('non-symlink extraction destination required')
    if any(destination.iterdir()):
        raise ValueError('empty extraction destination required')
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) > 150 or len({m.name for m in members}) != len(members) or sum(m.size for m in members) > 100_000_000:
            raise ValueError('bounded unique member inventory required')
        for m in members:
            checked_name(m.name)
            if not m.isfile() or m.size < 0:
                raise ValueError('regular archive members only')
        m = archive.getmember('MANIFEST.json')
        if m.size > 2_000_000:
            raise ValueError('bounded manifest required')
        manifest = json.load(archive.extractfile(m))
        expected = manifest['members']
        if set(expected) | {'MANIFEST.json'} != {m.name for m in members}:
            raise ValueError('archive inventory mismatch')
        names = {m.name for m in members}
        if manifest.get('actor_kind') == RESPONSIVE_KIND or names == RESPONSIVE_MEMBERS:
            if sum(m.size for m in members) > 1_000_000:
                raise ValueError('bounded responsive archive required')
            _check_responsive_payload({m.name: archive.extractfile(m).read() for m in members})
        for m in members:
            content = archive.extractfile(m).read()
            if m.name != 'MANIFEST.json' and expected[m.name] != {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}:
                raise ValueError('archive member identity mismatch')
            p = destination/m.name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(content)
    return manifest



def responsive_worker(root, *, scenario, seed, seat, out):
    """One archive game plus a shadow adapter on identical public inputs.

    Only the archive actions advance the world. The control uses a fresh private
    namespace and deep-copied inputs; it does not start a second engine.
    """
    import copy
    import resource
    if (scenario, seat) not in RESPONSIVE_CELLS or type(seat) is not int:
        raise ValueError('one preregistered responsive world/seat cell required')
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise ValueError('new responsive worker receipt required')
    actor, metrics = cold_responsive(root)
    metrics['rss_after_cold_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    # Only now expose canonical project code to the evaluator, never the archive.
    sys.path.insert(0, str(ROOT/'src'))
    from farm_neat.evaluation import scenarios, k0006, shop_router
    from farm_neat.evaluation.common import digest
    from farm_neat.game.exact_engine import assert_bundled_engine_identity
    cases = responsive_controls()
    case = next(c for c in cases if (c['world'], c['seat']) == (scenario, seat))
    spec = scenarios.catalog()[scenario]
    if spec['seed'] != seed or spec['opponent'] != 'router':
        raise ValueError('unchanged responsive scenario seed and rival required')
    if shop_router.identity()['content_sha256'] != case['opponent_identity_sha256']:
        raise ValueError('frozen responsive preflight rival changed')
    control = k0006.K0006Adapter()
    if control.identity['content_sha256'] != RESPONSIVE_IDENTITY:
        raise ValueError('canonical responsive adapter changed')
    imported = set(metrics['actor_imports'])
    times, pairs, witnesses = [], [], []
    def measured(observation, configuration):
        actual_obs, actual_cfg = copy.deepcopy(observation), copy.deepcopy(configuration)
        expected_obs, expected_cfg = copy.deepcopy(observation), copy.deepcopy(configuration)
        started = time.perf_counter()
        with responsive_imports(imported):
            action = actor(actual_obs, actual_cfg)
        times.append(time.perf_counter()-started)
        expected = control.act(expected_obs, expected_cfg)
        if not isinstance(action, dict):
            raise ValueError('responsive action must satisfy the object-type game schema')
        encoded(action)  # Reject non-JSON and non-finite output before the engine.
        pair = {'step': observation['step'], 'archive_sha256': digest(action),
                'adapter_sha256': digest(expected), 'equal': action == expected}
        pairs.append(pair)
        if not pair['equal']:
            witnesses.append({'step': observation['step'], 'observation_sha256': digest(observation),
                              'archive_action': action, 'adapter_action': expected})
            raise ValueError('responsive action mismatch at step '+str(observation['step']))
        return action
    _, settings = _responsive_directory(root)
    row = {'schema': 'farmneat-responsive-script-cold-game-v1', 'complete': False,
        'world': scenario, 'seed': seed, 'seat': seat, 'opponent': 'router',
        'physical_games_started': 1, 'candidate_identity': control.identity,
        'deployment_sha256': sha(Path(root)/'DEPLOYMENT.json'),
        'builder_sha256': settings['builder_sha256'], 'runtime': metrics,
        'engine': assert_bundled_engine_identity(),
        'native_engine_backend': 'BUNDLED_AUTHORITATIVE_PYTHON_INTERPRETER_NOT_CPP',
        'historical_control': case, 'learned_policy': False, 'gameplay_strength_claim': False}
    # Keep a charged-start receipt even if the evaluator fails before returning.
    out.write_bytes(encoded(row))
    started = time.perf_counter()
    result = scenarios.play(spec, measured, seat, timeout=120)
    elapsed = time.perf_counter()-started
    expected = case['expected']
    actual = {'actions_sha256': result.get('actions_sha256'),
        'cash': result.get('terminal_cash_by_seat'), 'status': result['status'],
        'margin': result.get('margin'), 'wdl': result.get('wdl')}
    matches = {k: actual[k] == expected[k] for k in expected}
    result_hash = digest(actual)
    metrics.update(actor_imports=sorted(imported))
    row.update({k: v for k, v in result.items() if k != 'actions_by_seat'})
    row.update(action_calls=len(times), action_digest_pairs=pairs, mismatch_witnesses=witnesses,
        action_parity_count=sum(p['equal'] for p in pairs), historical_matches=matches,
        terminal_result_sha256=result_hash, terminal_result_hash_scope='CANONICAL_RETAINED_PANEL_FIELDS',
        action_total_seconds=sum(times), action_max_seconds=max(times, default=0),
        action_p95_seconds=sorted(times)[int(.95*(len(times)-1))] if times else None,
        overage_used_at_1s=sum(max(0, t-1) for t in times),
        game_wall_seconds_including_shadow_and_evaluator=elapsed,
        rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        measurement_scope='actor timings include dependency guard, exclude shadow/evaluator; peak RSS includes all',
        action_validity_scope='object type, finite JSON, exact interpreter; no economic-fill or silent-no-op certification')
    row['complete'] = (result['status'] == 'COMPLETE' and result.get('physical_steps') == 719
        and len(pairs) == 719 and all(p['equal'] for p in pairs) and all(matches.values())
        and result_hash == case['expected_result_sha256'])
    _responsive_directory(root)  # Fail on archive drift during the game as well.
    out.write_bytes(encoded(row))
    if not row['complete']:
        raise ValueError('responsive archive parity or retained result mismatch')
    return row


def _preflight_responsive(archive, *, archive_sha256, extracted, manifest, out, timeout):
    """Existing preflight's single K0006 profile; fixed four cells, no search."""
    if not 0 < timeout <= 300:
        raise ValueError('bounded responsive worker timeout required (0,300] seconds')
    cases = responsive_controls()
    specs = {s['id']: s for s in json.loads((ROOT/'project/configs/scenarios.json').read_bytes())['scenarios']}
    result = {'schema': 'farmneat-responsive-script-preflight-v1', 'complete': False,
        'archive_sha256': archive_sha256, 'archive_bytes': Path(archive).stat().st_size,
        'expanded_bytes': sum(p.stat().st_size for p in extracted.iterdir()),
        'deployment_sha256': manifest['deployment_sha256'], 'builder_sha256': sha(__file__),
        'actor_identity_sha256': RESPONSIVE_IDENTITY, 'games_complete': False,
        'parity_complete': False, 'games': [], 'worker_launches': 0,
        'preregistered_cells': cases, 'maximum_worker_launches': 4,
        'reference_panel': {'path': RESPONSIVE_PANEL, 'sha256': RESPONSIVE_PANEL_SHA},
        'catalog_sha256': RESPONSIVE_CATALOG_SHA,
        'runtime_dependencies': {'standard_library_only': True, 'third_party': []},
        'scope': 'COLD_K0006_ARCHIVE_BOTH_SEATS_TWO_EXPOSED_ROUTER_SCENARIOS; NO_NEW_STRENGTH_CLAIM',
        'learned_policy': False, 'hosted_acceptance': 'UNVERIFIED',
        'retention': {'purpose': 'unchanged responsive fallback archive parity',
            'remove_when': 'superseded without active fallback or preflight dependencies'}}
    out.write_bytes(encoded(result))  # Freeze all four controls before any worker.
    for index, case in enumerate(cases):
        receipt = extracted.parent/f'responsive-{index}.json'
        args = [sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()), 'worker',
            '--root', str(extracted), '--scenario', case['world'], '--seat', str(case['seat']),
            '--seed', str(specs[case['world']]['seed']), '--out', str(receipt)]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': '', 'PYTHONDONTWRITEBYTECODE': '1'}
        result['worker_launches'] += 1
        out.write_bytes(encoded(result))
        started = time.perf_counter()
        game = None
        try:
            completed = subprocess.run(args, cwd=extracted, env=env, capture_output=True,
                                       text=True, timeout=timeout, check=False)
            if receipt.is_file():
                game = json.loads(receipt.read_bytes())
                game.update(process_stdout=completed.stdout, process_stderr=completed.stderr,
                    process_returncode=completed.returncode, process_wall_seconds=time.perf_counter()-started)
                result['games'].append(game)
            if completed.returncode or game is None:
                raise ValueError('responsive cold worker failed: '+completed.stderr[-3000:])
            if (not game.get('complete') or (game.get('world'), game.get('seat')) != (case['world'], case['seat'])
                    or game.get('deployment_sha256') != manifest['deployment_sha256']
                    or game.get('terminal_result_sha256') != case['expected_result_sha256']):
                raise ValueError('responsive cold worker evidence mismatch')
        except Exception as error:
            # Timeout/parse failures must not discard an already-written start/game.
            if game is None and receipt.is_file():
                try:
                    result['games'].append(json.loads(receipt.read_bytes()))
                except (ValueError, OSError):
                    result['invalid_worker_receipt'] = receipt.read_text(errors='replace')
            result['error'] = f'{type(error).__name__}: {error}'
            out.write_bytes(encoded(result))
            raise
        out.write_bytes(encoded(result))
        print(f"Cold responsive preflight {case['world']} seat {case['seat']} matched", flush=True)
    result['games_complete'] = len(result['games']) == 4 and all(
        g['status'] == 'COMPLETE' and g['physical_steps'] == 719 for g in result['games'])
    result['parity_complete'] = len(result['games']) == 4 and all(
        g['action_calls'] == g['action_parity_count'] == 719 and all(g['historical_matches'].values())
        for g in result['games'])
    result['complete'] = result['games_complete'] and result['parity_complete']
    result['physical_games_started'] = sum(g.get('physical_games_started', 0) for g in result['games'])
    result['physical_steps'] = sum(g.get('physical_steps', 0) for g in result['games'])
    out.write_bytes(encoded(result))
    if not result['complete']:
        raise ValueError('all four responsive archive parity cells required')
    return result

def worker(root, *, seed, seat, out, sampler_seed=None, scenario=None):
    # This worker starts in a fresh interpreter. Import archived agent first;
    # simulation is source-loaded independently, never supplied to the actor.
    import runpy
    import importlib.util
    import resource
    settings = json.loads((Path(root) / 'DEPLOYMENT.json').read_text())
    if settings.get('schema') == RESPONSIVE_SCHEMA or scenario is not None:
        if sampler_seed is not None:
            raise ValueError('responsive K0006 has no sampler')
        return responsive_worker(root, scenario=scenario, seed=seed, seat=seat, out=out)
    if settings.get('schema') == TAPE_SCHEMA:
        if sampler_seed is not None:
            raise ValueError('fixed tape has no sampler')
        return tape_worker(root, seed=seed, seat=seat, out=out)
    started = time.perf_counter()
    main = runpy.run_path(str(Path(root)/'main.py'))
    cold = time.perf_counter()-started
    if sampler_seed is not None:
        main['_resident'].use_diagnostic_sampler_seed(sampler_seed)
    module_path = ROOT/'src/farm_neat/game/exact_engine.py'
    spec = importlib.util.spec_from_file_location('_deployment_preflight_engine', module_path)
    engine = importlib.util.module_from_spec(spec); sys.modules[spec.name] = engine; spec.loader.exec_module(engine)
    env = engine.ExactKaggricultureEnvironment(seed=seed)
    times, active = [], 0
    for step in range(719):
        started = time.perf_counter()
        action = main['agent'](env.agent_observation(seat), env.configuration)
        times.append(time.perf_counter()-started)
        active += action != engine.PASS_ACTION
        joint = [engine.PASS_ACTION, engine.PASS_ACTION]; joint[seat] = action
        env.step(joint)
    own = env.agent_observation(seat)['farms']
    result = {'seed': seed, 'seat': seat, 'opponent': 'PASS', 'physical_games': 1,
        'sampler_seed': main['_resident'].episode_sampler_seed,
        'sampler_seed_source': 'OS_ENTROPY' if sampler_seed is None else 'EXPLICIT_DIAGNOSTIC_OVERRIDE',
        'physical_steps': env.step_index, 'done': env.done, 'own_cash': own[seat]['money'],
        'opponent_cash': own[1-seat]['money'], 'active_turns': active,
        'cold_seconds_including_import': cold, 'action_max_seconds': max(times),
        'action_total_seconds': sum(times), 'overage_used_at_1s': sum(max(0,t-1) for t in times),
        'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'decisions': main['_resident'].decisions, 'gameplay_strength_claim': False,
        'diagnostic_summary': (main['_resident'].diagnostics.summary()
            if getattr(main['_resident'], 'diagnostics', None) is not None else {'supported': False}),
        'runtime': {'python': sys.version.split()[0], 'torch': sys.modules['torch'].__version__,
                    'numpy': sys.modules['numpy'].__version__},
        'engine': engine.assert_bundled_engine_identity()}
    Path(out).write_bytes(encoded(result))


def tape_worker(root, *, seed, seat, out):
    import copy
    import importlib.util
    import resource
    import runpy
    started = time.perf_counter()
    main = runpy.run_path(str(Path(root) / 'main.py'))
    cold = time.perf_counter() - started
    module_path = ROOT / 'src/farm_neat/game/exact_engine.py'
    spec = importlib.util.spec_from_file_location('_tape_preflight_engine', module_path)
    engine = importlib.util.module_from_spec(spec); sys.modules[spec.name] = engine; spec.loader.exec_module(engine)
    env = engine.ExactKaggricultureEnvironment(seed=seed)
    actions, times = [], []
    for step in range(719):
        started = time.perf_counter()
        action = main['agent'](env.agent_observation(seat), env.configuration)
        times.append(time.perf_counter() - started)
        actions.append(copy.deepcopy(action))
        joint = [engine.PASS_ACTION, engine.PASS_ACTION]; joint[seat] = action
        env.step(joint)
    farms = env.agent_observation(seat)['farms']
    Path(out).write_bytes(encoded({'seed': seed, 'seat': seat, 'opponent': 'PASS',
        'physical_games': 1, 'physical_steps': env.step_index, 'done': env.done,
        'own_cash': farms[seat]['money'], 'opponent_cash': farms[1-seat]['money'],
        'emitted_actions': len(actions), 'emitted_tape_sha256': hashlib.sha256(encoded(actions)).hexdigest(),
        'cold_seconds_including_import': cold, 'action_max_seconds': max(times),
        'action_total_seconds': sum(times), 'overage_used_at_1s': sum(max(0, t-1) for t in times),
        'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'runtime': {'python': sys.version.split()[0], 'third_party_dependencies': False},
        'engine': engine.assert_bundled_engine_identity(), 'learned_policy': False,
        'gameplay_strength_claim': False}))


def check_tape_diagnostics(stderr, game, manifest):
    try:
        events = [json.loads(line) for line in stderr.splitlines() if line.strip()]
    except ValueError:
        return {'complete': False, 'error': 'invalid startup diagnostic'}
    expected = {'schema': TAPE_SCHEMA, 'event': 'EPISODE_START', 'seat': game['seat'],
        'tape_sha256': manifest['tape_sha256'], 'tape_index': manifest['tape_index'], 'learned_policy': False}
    return {'complete': events == [expected], 'events': events, 'event_count': len(events)}


def check_diagnostics(stderr, game):
    """Verify exact emitted programs separately from complete-game execution."""
    from farm_neat.evaluation import deployment_runtime as runtime
    events, errors, validation_errors = [], [], []
    for line in stderr.splitlines():
        if not line.strip():
            continue
        try:
            events.append(runtime.decode_diagnostic(line))
        except Exception as error:
            errors.append(type(error).__name__)
    publications = [e for e in events if e.get('event') == 'PUBLICATION']
    expected = game['decisions']
    complete = (len(events) == 6 and events[0].get('event') == 'EPISODE_START'
        and [e.get('sequence') for e in events] == list(range(6))
        and [e.get('step') for e in publications] == list(runtime.STEPS)
        and not errors and game.get('diagnostic_summary', {}).get('complete') is True
        and len(publications) == len(expected))
    if complete:
        for event, decision in zip(publications, expected):
            try:
                matches = (event['program'] == decision['program'] and event['admission'] == decision['admission']
                    and event['behavior_joint_logp'] == decision['joint_logp']
                    and event['input_sha256'] == decision['input_sha256']
                    and event['program_sha256'] == runtime.action_space.digest(event['program'])
                    and event['sampler_seed'] == game['sampler_seed']
                    and all(event.get(k) == events[0].get(k) for k in ('model_sha256', 'deployment_sha256', 'settings_sha256', 'runtime', 'seat', 'sampler_seed')))
            except (KeyError, TypeError, ValueError) as error:
                validation_errors.append(f'{type(error).__name__}: {error}')
                matches = False
            if not matches:
                complete = False
    return {'complete': complete, 'event_count': len(events), 'parse_errors': errors,
            'validation_errors': validation_errors,
            'stderr_bytes': len(stderr.encode()), 'events': events,
            'scope': 'EXACT_LOGGED_PROGRAMS_AND_ADMISSION; SEPARATE_FROM_GAMEPLAY_SUCCESS'}


def worker_environment(extracted):
    """Keep the active NumPy installation available without exposing our source tree."""
    import numpy as np
    numpy_site = Path(np.__file__).resolve().parent.parent
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': '',
            'PYTHONPATH': os.pathsep.join((str(Path(extracted)/'src'), str(numpy_site))),
            'PYTHONDONTWRITEBYTECODE': '1'}


def preflight(archive, *, archive_sha256, out, seed_base, timeout, sampler_seed_base=None):
    out = Path(out).absolute()
    if out.exists() or not out.is_relative_to(ROOT/'work/runs'):
        raise ValueError('new run preflight receipt required')
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = ROOT/'work/scratch'; scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='deployment-', dir=scratch) as tmp:
        extracted = Path(tmp)/'agent'; extracted.mkdir()
        manifest = extract(archive, archive_sha256, extracted)
        if manifest.get('actor_kind') == RESPONSIVE_KIND:
            if sampler_seed_base is not None:
                raise ValueError('responsive K0006 has no sampler seed')
            return _preflight_responsive(archive, archive_sha256=archive_sha256,
                extracted=extracted, manifest=manifest, out=out, timeout=timeout)
        is_tape = manifest.get('actor_kind') == 'FIXED_EXTERNAL_ACTION_TAPE'
        if is_tape and sampler_seed_base is not None:
            raise ValueError('fixed tape has no sampler seed')
        result = {'schema': 'farmneat-deployment-preflight-v1', 'archive_sha256': archive_sha256,
            'deployment_sha256': manifest['deployment_sha256'], 'complete': False,
            'games_complete': False, 'diagnostics_complete': False, 'games': [],
            'scope': 'COLD_CPU_ARCHIVE_COMPLETE_GAMES_BOTH_SEATS_VS_PASS; NOT_HOSTED_OR_STRENGTH_CERTIFICATION',
            'retention': {'purpose': 'exact artifact runtime and gameplay preflight', 'remove_when': 'artifact superseded and no unresolved deployment dependency'}}
        if is_tape:
            result.update(schema='farmneat-fixed-tape-preflight-v1', learned_policy=False)
        for seat in (0, 1):
            receipt = Path(tmp)/f'seat-{seat}.json'
            env = ({**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': '', 'PYTHONDONTWRITEBYTECODE': '1'}
                   if is_tape else worker_environment(extracted))
            args = [sys.executable, '-B', *(['-S'] if is_tape else []), str(Path(__file__).resolve()), 'worker', '--root', str(extracted),
                    '--seed', str(seed_base+seat), '--seat', str(seat), '--out', str(receipt)]
            if sampler_seed_base is not None:
                args += ['--sampler-seed', str(sampler_seed_base+seat)]
            try:
                completed = subprocess.run(args, cwd=extracted, env=env, capture_output=True, text=True, timeout=timeout, check=False)
                if completed.returncode or not receipt.is_file():
                    raise RuntimeError(f'cold worker exit {completed.returncode}: '+completed.stderr[-3000:])
                game = json.loads(receipt.read_text())
                result['games'].append(game)
                game['diagnostic_verification'] = (check_tape_diagnostics(completed.stderr, game, manifest)
                    if is_tape else check_diagnostics(completed.stderr, game))
                print(f'Cold complete-game preflight seat {seat} finished', flush=True)
            except Exception as error:
                result['error'] = f'{type(error).__name__}: {error}'
                out.write_bytes(encoded(result)); raise
            out.write_bytes(encoded(result))
        result['games_complete'] = all(g['done'] and g['physical_steps'] == 719 and
            (g['emitted_actions'] == 719 and g['emitted_tape_sha256'] == manifest['tape_sha256']
             if is_tape else len(g['decisions']) == 5) for g in result['games'])
        result['diagnostics_complete'] = all(g['diagnostic_verification']['complete'] for g in result['games'])
        result['complete'] = result['games_complete'] and result['diagnostics_complete']
        if not result['games_complete']:
            result['error'] = 'complete both-seat declared-profile games required'
        elif not result['diagnostics_complete']:
            result['error'] = 'complete startup and five-publication diagnostics required for both seats'
        out.write_bytes(encoded(result))
        if not result['complete']:
            raise ValueError(result['error'])
        return result


def compare(archive, *, archive_sha256, checkpoint, out):
    """Two zero-step actor-input witnesses; compare original and decoded policy.

    This measures opening sensitivity to quantization on CPU, not GPU parity or
    game-wide equivalence. Both graphs use the same input and sampler stream.
    """
    from farm_neat.evaluation import deployment_runtime as runtime
    from farm_neat.policy import network, fast_profile, plan_observation
    from farm_neat.economy.orders import FullSeasonPlan
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    import torch
    out = Path(out).absolute()
    if out.exists() or not out.is_relative_to(ROOT/'work/runs'):
        raise ValueError('new comparison receipt required')
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='deployment-compare-', dir=ROOT/'work/scratch') as tmp:
        root = Path(tmp)
        extract(archive, archive_sha256, root)
        settings = json.loads((root/'DEPLOYMENT.json').read_text())
        original, _ = network.load_model(Path(checkpoint)/'model.pt', device='cpu',
            expected_file_sha256=settings['original_model_sha256'])
        original.eval()
        decoded = runtime.load_graph(root, settings)
        encoders = [fast_profile.FastEncoder(g, expected_identity=settings['fast_profile']) for g in (original, decoded)]
        env = ExactKaggricultureEnvironment(seed=9161000)
        rows = []
        for seat in (0,1):
            obs = env.agent_observation(seat)
            controller = FullSeasonPlan('comparison', publication_steps=runtime.STEPS); controller.observe(obs)
            packed = plan_observation.pack(obs, controller, env.configuration, expected_identity=settings['plan_adapter'])
            tensors = runtime.collate_one(packed)
            values, probabilities, programs = [], [], []
            with torch.inference_mode():
                for graph, encoder in zip((original, decoded), encoders):
                    context, value, _ = encoder(*tensors)
                    logp, _, _, _, _ = graph.factor(context[0], 'op', network.actions.OPS, settings['temperature'], as_of_step=0)
                    probabilities.append(logp.exp().tolist()); values.append(float(value[0]))
                    record, _, _ = fast_profile.sample_batched(graph, context,
                        [fast_profile.LaneStream(settings['diagnostic_lane_seeds'][seat])],
                        max_commands=settings['command_budget'], temperature=settings['temperature'], as_of_step=0)
                    programs.append(record[0]['program'])
            rows.append({'seat': seat, 'input_sha256': packed['content_sha256'], 'first_op_choices': list(network.actions.OPS),
                'original_probabilities': probabilities[0], 'deployment_probabilities': probabilities[1],
                'first_op_max_abs_difference': max(abs(a-b) for a,b in zip(*probabilities)),
                'original_value': values[0], 'deployment_value': values[1],
                'sampled_program_equal': programs[0] == programs[1],
                'original_program': programs[0], 'deployment_program': programs[1]})
        result = {'schema': 'farmneat-deployment-opening-comparison-v1', 'archive_sha256': archive_sha256,
            'original_model_sha256': settings['original_model_sha256'], 'deployment_sha256': settings['deployment_sha256'],
            'physical_games': 0, 'physical_steps': 0, 'initial_states': 2, 'seed': 9161000,
            'temperature': settings['temperature'], 'command_budget': settings['command_budget'], 'rows': rows,
            'scope': 'CPU_SAME_INPUT_AND_SAMPLER_OPENING_WITNESSES_ONLY; NO_GAMEWIDE_OR_CUDA_PARITY_CLAIM',
            'retention': {'purpose': 'quantized actor behavior witness', 'remove_when': 'artifact superseded without unresolved quantization issue'}}
        out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(encoded(result))
        return result



def audit_v0(*, output, probe_import=False, timeout=30):
    """Default-off source admission; a blocked audit is never an archive manifest.

    The accepted V0 loader is not reimplemented here. Its scorer currently
    imports/hash-pins training files, so no training-isolated archive is emitted.
    The opt-in import probe is source-only, not a synthetic model replacement.
    """
    from farm_neat.evaluation import v0_deployment_runtime as runtime
    if type(probe_import) is not bool:
        raise ValueError('explicit boolean import-probe selection required')
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('symlinked audit output refused')
    if output.exists():
        raise ValueError('new audit output required; existing files are never overwritten')
    source_audit = runtime.audit_sources(ROOT)
    loader_contract = runtime.audit_loader_contract(ROOT)
    for name in source_audit['members'].keys() & loader_contract['members'].keys():
        if source_audit['members'][name] != loader_contract['members'][name]:
            raise ValueError('source changed between V0 admission stages: ' + name)
    auditor_sources = {}
    for path in (Path(__file__), Path(runtime.__file__)):
        path = runtime._regular(path)
        data = path.read_bytes()
        auditor_sources[path.relative_to(ROOT).as_posix()] = {
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    report = {'schema': runtime.SCHEMA, 'status': 'BLOCKED',
        'scope': 'SOURCE_ADMISSION_ONLY_NOT_V0_DEPLOYMENT', 'archive_produced': False,
        'auditor_sources': auditor_sources,
        'checkpoint_loaded': False, 'native_transitions': 0, 'source_audit': source_audit, 'loader_contract': loader_contract,
        'blockers': ['TRAINING_FREE_RUNTIME_NOT_AVAILABLE' if source_audit['forbidden'] else
                     'ARCHIVE_RUNTIME_AND_PARITY_NOT_IMPLEMENTED'],
        'not_run': ['AUTHENTICATED_CHECKPOINT_EXPORT', 'ARCHIVE_COLD_LOAD',
                    'OPENING_AND_RESUMED_PLAN_ACTION_SAMPLER_PARITY', 'NATIVE_FULL_GAME_PARITY',
                    'PRODUCTION_PLATFORM_QUALIFICATION', 'PROMOTION_OR_SUBMISSION']}
    if source_audit['missing']:
        report['blockers'].append('MISSING_SOURCE_DEPENDENCIES')
    if loader_contract['forbidden']:
        report['blockers'].append('COMMITTED_LOADER_REQUIRES_TRAINING_SOURCES')
    if loader_contract['original_envelope_required']:
        report['blockers'].append('COMMITTED_LOADER_REQUIRES_ORIGINAL_STATE_ENVELOPE')
    if loader_contract['missing'] or loader_contract['unresolved']:
        report['blockers'].append('COMMITTED_LOADER_CONTRACT_UNRESOLVED')
    if probe_import:
        report['import_probe'] = runtime.probe_imports(ROOT, timeout=timeout)
    report['content_sha256'] = hashlib.sha256(encoded(report)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(encoded(report)); stream.flush(); os.fsync(stream.fileno())
    return report


def prepare_v0(checkpoint, *, commit_sha256, request, max_bytes=100_000_000):
    """Default-off authenticated model-only export; no output or archive claim."""
    from farm_neat.evaluation import v0_deployment_runtime as runtime
    return runtime.prepare_export(checkpoint, commit_sha256, request, max_bytes=max_bytes)


def build_v0(checkpoint, *, commit_sha256, request, output, max_bytes=100_000_000, timeout=30):
    """Build only after original-COMMIT authentication and training-free closure.

    Derived model bytes never masquerade as a re-signed native checkpoint. A
    blocked source closure discards the private staging payload and emits no file.
    """
    from farm_neat.evaluation import v0_deployment_runtime as runtime
    runtime._limit(max_bytes); runtime._timeout(timeout)
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new non-symlinked V0 output required; never overwrite current/rollback')
    if output.is_relative_to(Path(checkpoint).absolute()) or any(
            part.lower() in ('current', 'rollback', 'champion') for part in output.parts):
        raise ValueError('V0 output must not mutate checkpoint/protected paths')
    prepared = prepare_v0(checkpoint, commit_sha256=commit_sha256, request=request, max_bytes=max_bytes)
    closure = runtime.archive_closure(ROOT, prepared['settings'])
    if closure['forbidden'] or closure['missing']:
        raise runtime.V0DeploymentBlocked({**prepared['receipt'], 'status': 'BLOCKED',
            'blockers': (['TRAINING_FREE_RUNTIME_SOURCE_CLOSURE_REQUIRED'] if closure['forbidden'] else [])
                        + (['MISSING_RUNTIME_SOURCE_DEPENDENCIES'] if closure['missing'] else []),
            'closure': closure,
            'not_run': ['ARCHIVE_COLD_LOAD', 'ISOLATED_OPENING_AND_RESUMED_PARITY',
                        'NATIVE_FULL_GAME_PARITY', 'PRODUCTION_PLATFORM_QUALIFICATION']})
    return runtime.publish_archive(ROOT, prepared, closure, output, max_bytes=max_bytes, timeout=timeout,
                                   writer=write_tar)


def cold_probe_v0(archive, *, archive_sha256, output, max_bytes=100_000_000, timeout=30):
    """Opt-in isolated CPU cold-load probe. No games, uploads or promotion."""
    from farm_neat.evaluation import v0_deployment_runtime as runtime
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new non-symlinked cold-probe output required')
    report = runtime.probe_archive(archive, archive_sha256, max_bytes=max_bytes, timeout=timeout)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the actor-identity order that the cold-worker join authenticated.
    wire = json.dumps(report, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()
    with output.open('xb') as stream:
        stream.write(wire); stream.flush(); os.fsync(stream.fileno())
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    vb = sub.add_parser('build-v0', help='opt-in diagnostic native V0 derived archive; source gate may BLOCK')
    vb.add_argument('--checkpoint', required=True); vb.add_argument('--commit-sha256', required=True)
    vb.add_argument('--request', required=True); vb.add_argument('--request-sha256', required=True)
    vb.add_argument('--output', required=True); vb.add_argument('--max-bytes', type=int, default=100_000_000)
    vb.add_argument('--timeout', type=float, default=30)
    vc = sub.add_parser('cold-probe-v0', help='opt-in isolated diagnostic V0 archive load, no native games')
    vc.add_argument('--archive', required=True); vc.add_argument('--archive-sha256', required=True)
    vc.add_argument('--output', required=True); vc.add_argument('--max-bytes', type=int, default=100_000_000)
    vc.add_argument('--timeout', type=float, default=30)
    v = sub.add_parser('audit-v0', help='source-only V0 isolation audit; never builds an archive')
    v.add_argument('--output', required=True)
    v.add_argument('--probe-import', action='store_true', help='two fresh isolated CPU source imports, no weights')
    v.add_argument('--timeout', type=float, default=30)
    r = sub.add_parser('build-responsive'); r.add_argument('--output', required=True)
    t = sub.add_parser('build-tape'); t.add_argument('--output', required=True)
    t.add_argument('--tape-index', type=int, default=0)
    b = sub.add_parser('build'); b.add_argument('--checkpoint', required=True)
    b.add_argument('--model-sha256', required=True); b.add_argument('--output', required=True)
    b.add_argument('--command-budget', type=int, required=True); b.add_argument('--temperature', type=float, required=True)
    b.add_argument('--lane-seeds', type=int, nargs=2, default=[9162000, 9162001])
    f = sub.add_parser('preflight'); f.add_argument('--archive', required=True); f.add_argument('--archive-sha256', required=True)
    f.add_argument('--out', required=True); f.add_argument('--seed-base', type=int, default=9160000)
    f.add_argument('--timeout', type=int, default=300)
    f.add_argument('--sampler-seed-base', type=int, help='explicit diagnostic override; default tests actual live OS entropy')
    c = sub.add_parser('compare'); c.add_argument('--archive', required=True); c.add_argument('--archive-sha256', required=True)
    c.add_argument('--checkpoint', required=True); c.add_argument('--out', required=True)
    w = sub.add_parser('worker'); w.add_argument('--root', required=True); w.add_argument('--seed', type=int, required=True)
    w.add_argument('--seat', type=int, choices=(0,1), required=True); w.add_argument('--out', required=True)
    w.add_argument('--sampler-seed', type=int)
    w.add_argument('--scenario', choices=('router-no-yarn', 'router-yarn'))
    a = p.parse_args(argv)
    if a.mode != 'worker':
        sys.path.insert(0, str(ROOT/'src'))
    if a.mode in ('build-v0', 'cold-probe-v0'):
        from farm_neat.evaluation import v0_deployment_runtime as runtime
        try:
            if a.mode == 'build-v0':
                request = runtime.read_request(a.request, a.request_sha256)
                report = build_v0(a.checkpoint, commit_sha256=a.commit_sha256, request=request,
                                  output=a.output, max_bytes=a.max_bytes, timeout=a.timeout)
            else:
                report = cold_probe_v0(a.archive, archive_sha256=a.archive_sha256, output=a.output,
                                       max_bytes=a.max_bytes, timeout=a.timeout)
        except runtime.V0DeploymentBlocked as error:
            print(json.dumps(error.report, indent=2, allow_nan=False))
            raise SystemExit(2)
        except (ValueError, OSError) as error:
            print(json.dumps({'status': 'REFUSED', 'archive_produced': False,
                              'error_type': type(error).__name__, 'error': str(error)}, indent=2))
            raise SystemExit(2)
        print(json.dumps(report, indent=2, allow_nan=False))
        if report.get('status') == 'BLOCKED':
            raise SystemExit(2)
    elif a.mode == 'audit-v0':
        report = audit_v0(output=a.output, probe_import=a.probe_import, timeout=a.timeout)
        print(json.dumps(report, indent=2))
        raise SystemExit(2)  # BLOCKED is not a successful deployment receipt.
    elif a.mode == 'build-responsive':
        print(json.dumps(build_responsive(output=a.output), indent=2))
    elif a.mode == 'build-tape':
        print(json.dumps(build_tape(output=a.output, tape_index=a.tape_index), indent=2))
    elif a.mode == 'build':
        print(json.dumps(build(a.checkpoint, model_sha256=a.model_sha256, output=a.output, command_budget=a.command_budget,
                               temperature=a.temperature, lane_seeds=a.lane_seeds), indent=2))
    elif a.mode == 'preflight':
        r = preflight(a.archive, archive_sha256=a.archive_sha256, out=a.out, seed_base=a.seed_base, timeout=a.timeout,
                      sampler_seed_base=a.sampler_seed_base)
        print(json.dumps({k:v for k,v in r.items() if k != 'games'}, indent=2))
    elif a.mode == 'compare':
        r = compare(a.archive, archive_sha256=a.archive_sha256, checkpoint=a.checkpoint, out=a.out)
        print(json.dumps({**{k:v for k,v in r.items() if k != 'rows'},
                         'first_op_max_abs_difference': [x['first_op_max_abs_difference'] for x in r['rows']],
                         'sampled_program_equal': [x['sampled_program_equal'] for x in r['rows']]}, indent=2))
    else:
        worker(a.root, seed=a.seed, seat=a.seat, out=a.out, sampler_seed=a.sampler_seed, scenario=a.scenario)


if __name__ == '__main__':
    main()
