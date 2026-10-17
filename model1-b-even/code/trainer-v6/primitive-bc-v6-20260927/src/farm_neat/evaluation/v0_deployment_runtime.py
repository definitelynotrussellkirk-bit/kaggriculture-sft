"""Default-off V0 derived export, source admission and diagnostic archive guards.

Original checkpoints are authenticated through the accepted bridge; only model
state and its actor identity are copied. No scorer, decoder or continuation codec
is replaced. The derived format is NOT a re-signed native COMMIT.

Current scorer imports and source pins require training files, so archive builds
remain BLOCKED. Source-import audits, same-checkout derived-model parity, isolated
archive load, native parity and production qualification are distinct gates.
Nothing in this module trains or advances an engine. Source audits are not proofs
of dynamic import closure; prospective archive success is not playing strength.
"""
from __future__ import annotations

import ast
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SCHEMA = 'farm-v0-cold-source-admission-v1'
ENTRIES = ('farm_neat.policy.v0_scorer', 'farm_neat.evaluation.v0_policy')
FORBIDDEN = ('farm_neat.training', 'farm_neat.rollout')
ASSETS = ('src/farm_neat/economy/v0/modules.json',
          'src/farm_neat/game/kaggriculture.json',
          'src/farm_neat/game/kaggriculture_engine_reference.py',
          'project/configs/scenarios.json')


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('regular non-symlinked source required: ' + str(path))
    return path


def _module(path, root):
    parts = path.relative_to(root/'src').with_suffix('').parts
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


def _eager_nodes(node):
    # Class suites execute now; function bodies do not. Branches are conservative
    # (both arms), so the cold-import probe distinguishes a real import failure.
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _eager_nodes(child)


def _targets(path, name, known, *, tree=None):
    if tree is None:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    package = name if path.name == '__init__.py' else name.rpartition('.')[0]
    targets, missing, external = set(), set(), set()
    for node in _eager_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = alias.name
                if target == 'farm_neat' or target.startswith('farm_neat.'):
                    (targets if target in known else missing).add(target)
                else:
                    external.add(target.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level:
                parents = package.split('.')
                if node.level > len(parents):
                    raise ValueError('relative import leaves package: ' + name)
                base = '.'.join(parents[:len(parents)-node.level+1] + ([base] if base else []))
            if base == 'farm_neat' or base.startswith('farm_neat.'):
                (targets if base in known else missing).add(base)
                # "from module import function" is not a missing submodule.
                targets.update(base+'.'+a.name for a in node.names if base+'.'+a.name in known)
            elif base and base != '__future__':
                external.add(base.split('.')[0])
    parent = name.rpartition('.')[0]
    if parent in known:
        targets.add(parent)
    return targets, missing, external


def audit_sources(root):
    """Return actual eager import chains and hashes; never import torch/training.

    No strings, identities or code are edited to make a closure admissible.
    A caller must still authenticate a real committed scorer with v0_policy's
    accepted loader and implement/prove the archive runtime after this boundary.
    """
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)) or not (root/'src').is_dir():
        raise ValueError('canonical source root required')
    known = {_module(p, root): p for p in sorted((root/'src/farm_neat').rglob('*.py'))}
    pending = deque((name, [name]) for name in ENTRIES)
    seen, members, forbidden, missing, external = set(), {}, [], [], set()
    while pending:
        name, chain = pending.popleft()
        if name in seen:
            continue
        seen.add(name)
        if name not in known:
            missing.append({'module': name, 'chain': chain})
            continue
        path = _regular(known[name]); relative = path.relative_to(root).as_posix()
        members[relative] = {'sha256': _sha(path), 'bytes': path.stat().st_size}
        if any(name == prefix or name.startswith(prefix+'.') for prefix in FORBIDDEN):
            forbidden.append({'module': name, 'path': relative, 'chain': chain,
                              'reason': 'TRAINING_OR_ROLLOUT_FILE_IN_EAGER_RUNTIME_CLOSURE'})
        targets, absent, dependencies = _targets(path, name, known)
        external.update(dependencies)
        missing.extend({'module': target, 'chain': [*chain, target]} for target in sorted(absent))
        pending.extend((target, [*chain, target]) for target in sorted(targets) if target not in seen)
    for relative in ASSETS:
        path = root/relative
        if not path.exists():
            missing.append({'asset': relative, 'chain': ['DECLARED_ACTOR_ASSET', relative]})
            continue
        path = _regular(path)
        members[relative] = {'sha256': _sha(path), 'bytes': path.stat().st_size}
    return {'schema': SCHEMA, 'status': 'BLOCKED' if forbidden or missing else 'STATIC_CHECK_ONLY',
            'deployment_ready': False, 'entries': list(ENTRIES),
            'members': dict(sorted(members.items())), 'forbidden': forbidden, 'missing': missing,
            'external_import_roots': sorted(external),
            'limitations': ['EAGER_IMPORT_LOWER_BOUND_NOT_FULL_DYNAMIC_CLOSURE',
                           'NO_CHECKPOINT_AUTHENTICATION_OR_MODEL_CONSTRUCTION',
                           'CLEAN_STATIC_IMPORTS_DO_NOT_CERTIFY_ARCHIVE_READINESS']}



LOADER_SOURCE = 'src/farm_neat/evaluation/v0_policy.py'
LOADER_ENTRIES = ('load_committed_scorer', 'inspect_committed_scorer')
# Narrow, literal witnesses from the accepted loader, not an alternative loader
# or a general Python control-flow proof. Changed expressions require re-review.
LOADER_CONSTRAINTS = {
    'load_state_hash': ('load_committed_scorer', "sha != info['commit']['state_sha256']"),
    'state_training_identity': ('load_committed_scorer',
                                "state.get('identity') != info['training_identity']"),
    'inspect_state_hash': ('inspect_committed_scorer',
                           "file_sha(path) != commit.get('state_sha256')"),
    'inspect_state_size': ('inspect_committed_scorer',
                           "path.stat().st_size != commit.get('state_bytes')"),
    'runtime_identity': ('inspect_committed_scorer',
                         "identity.get('runtime') != BC.runtime_identity(device)"),
}


def audit_loader_contract(root):
    """Expose call-time dependencies omitted by the eager-import lower bound.

    Read/hash the same source bytes; never import an owner or open a checkpoint.
    Only the two named accepted entrypoints are inspected. Import branches are
    conservative; transitively called helpers/dynamic imports are not analyzed.
    A clean/stub/refactored source is UNRESOLVED, never archive admission.
    """
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)) or not (root/'src').is_dir():
        raise ValueError('canonical source root required')
    known = {_module(p, root): p for p in sorted((root/'src/farm_neat').rglob('*.py'))}
    path = root/LOADER_SOURCE
    members, forbidden, missing, definitions = {}, [], [], {}
    text = ''
    if path.exists() or path.is_symlink():
        raw = _regular(path).read_bytes()
        text = raw.decode('utf-8')
        members[LOADER_SOURCE] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        tree = ast.parse(text, filename=LOADER_SOURCE)
        for name in LOADER_ENTRIES:
            found = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == name]
            if len(found) == 1:
                definitions[name] = found[0]
    for name in LOADER_ENTRIES:
        if name not in definitions:
            missing.append({'entrypoint': name, 'path': LOADER_SOURCE,
                            'reason': 'MISSING_OR_AMBIGUOUS_LOADER_ENTRYPOINT'})
            continue
        suite = ast.Module(body=definitions[name].body, type_ignores=[])
        for node in _eager_nodes(suite):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            direct = ast.Module(body=[node], type_ignores=[])
            targets, absent, _ = _targets(path, 'farm_neat.evaluation.v0_policy', known, tree=direct)
            missing.extend({'entrypoint': name, 'module': target, 'line': node.lineno}
                           for target in sorted(absent))
            for target in sorted(targets):
                dependency = _regular(known[target])
                relative = dependency.relative_to(root).as_posix()
                data = dependency.read_bytes()
                members[relative] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                if any(target == p or target.startswith(p+'.') for p in FORBIDDEN):
                    forbidden.append({'entrypoint': name, 'module': target, 'path': relative,
                                      'source_path': LOADER_SOURCE, 'line': node.lineno,
                                      'end_line': node.end_lineno, 'phase': 'CALL_TIME_IMPORT',
                                      'source_excerpt': ast.get_source_segment(text, node)})
    witnesses = {}
    for key, (name, expression) in LOADER_CONSTRAINTS.items():
        expected = ast.dump(ast.parse(expression, mode='eval').body)
        function = definitions.get(name)
        nodes = () if function is None else _eager_nodes(
            ast.Module(body=function.body, type_ignores=[]))
        match = next((node for node in nodes
                      if isinstance(node, ast.Compare) and ast.dump(node) == expected), None)
        witnesses[key] = {'status': 'UNRESOLVED' if match is None else 'FOUND_LITERAL_CONSTRAINT',
                          'entrypoint': name, 'source_path': LOADER_SOURCE}
        if match is not None:
            witnesses[key].update(line=match.lineno, end_line=match.end_lineno,
                                  source_excerpt=ast.get_source_segment(text, match))
    unresolved = [key for key, row in witnesses.items() if row['status'] == 'UNRESOLVED']
    envelope = True if all(witnesses[key]['status'] == 'FOUND_LITERAL_CONSTRAINT'
                          for key in ('load_state_hash', 'state_training_identity',
                                      'inspect_state_hash', 'inspect_state_size')) else None
    return {'schema': 'farm-v0-committed-loader-source-admission-v1',
            'status': 'BLOCKED' if forbidden or missing or envelope else 'UNRESOLVED',
            'deployment_ready': False, 'entries': list(LOADER_ENTRIES),
            'members': dict(sorted(members.items())), 'forbidden': forbidden, 'missing': missing,
            'contract_witnesses': witnesses, 'original_envelope_required': envelope,
            'unresolved': unresolved,
            'limitations': ['CALL_TIME_DIRECT_IMPORT_LOWER_BOUND_NOT_TRANSITIVE_CLOSURE',
                           'NOT_A_COMPLETE_CONTROL_FLOW_OR_EXPORT_PROOF',
                           'NO_CHECKPOINT_READ_OR_AUTHENTICATION_OR_MODEL_CONSTRUCTION',
                           'NO_OPTIMIZER_OR_PRIVATE_PROVENANCE_SANITIZATION',
                           'ABSENT_CONSTRAINTS_REQUIRE_REVIEW_NOT_DEPLOYMENT_ADMISSION']}

_PROBE = r'''
import importlib, json, pathlib, site, sys, traceback
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root/'src'))
result = {'isolated': bool(sys.flags.isolated), 'user_site_enabled': site.ENABLE_USER_SITE,
          'scope': 'SOURCE_IMPORT_ONLY_NO_MODEL_OR_CHECKPOINT', 'native_transitions': 0}
try:
    scorer = importlib.import_module('farm_neat.policy.v0_scorer')
    result.update(import_status='IMPORTED', model_source_pins=scorer.source_pins())
except Exception as error:
    result.update(import_status='IMPORT_FAILED', error_type=type(error).__name__,
                  missing_module=getattr(error, 'name', None), error=str(error),
                  traceback=traceback.format_exc())
loaded = {name: str(pathlib.Path(m.__file__).resolve())
          for name, m in sorted(sys.modules.items())
          if name.startswith('farm_neat') and getattr(m, '__file__', None)}
result['imported_package_files'] = loaded
result['all_package_files_in_supplied_tree'] = all(
    pathlib.Path(path).is_relative_to(root) for path in loaded.values())
result['loaded_training_modules'] = [name for name in loaded if name.startswith('farm_neat.training')]
print(json.dumps(result, sort_keys=True))
sys.exit(0 if result['import_status'] == 'IMPORTED' else 2)
'''


def _run_import(root, home, timeout):
    # The current interpreter's installed torch/numpy remain available. -I
    # ignores PYTHONPATH and user-site; the child's CWD/HOME are not the checkout.
    env = {'HOME': str(home), 'PATH': os.path.dirname(sys.executable),
           'CUDA_VISIBLE_DEVICES': '', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    try:
        proc = subprocess.run([sys.executable, '-I', '-B', '-c', _PROBE, str(root)],
                              cwd=home, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        return {'import_status': 'TIMEOUT', 'timeout_seconds': timeout,
                'stdout': str(error.stdout or ''), 'stderr': str(error.stderr or '')}
    try:
        result = json.loads(proc.stdout)
    except (ValueError, TypeError):
        return {'import_status': 'PROBE_FAILED', 'returncode': proc.returncode,
                'stdout': proc.stdout, 'stderr': proc.stderr}
    result.update(returncode=proc.returncode, stderr=proc.stderr)
    return result


def probe_imports(root, *, timeout=30):
    """Paired complete-tree versus training-free import, in fresh -I processes.

    This is NOT the archive cold-probe acceptance test. No archive is manufactured
    and no unavailable runtime is replaced by a stub, reduced graph or codec.
    """
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise ValueError('finite import-probe timeout in (0, 120] seconds required')
    root = Path(root).absolute()
    audit = audit_sources(root)
    excluded = {row['path'] for row in audit['forbidden']}
    with tempfile.TemporaryDirectory(prefix='farm-v0-cold-source-') as temporary:
        home = Path(temporary); isolated = home/'isolated'; isolated.mkdir()
        copied = {}
        for relative, identity in audit['members'].items():
            if relative in excluded:
                continue
            source = _regular(root/relative)
            data = source.read_bytes()
            if {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} != identity:
                raise ValueError('source changed during isolated copy: '+relative)
            destination = isolated/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            copied[relative] = identity
        control = _run_import(root, home, timeout)
        cold = _run_import(isolated, home, timeout)
        missing_module = cold.get('missing_module')
        blocked = (control.get('import_status') == 'IMPORTED'
                   and cold.get('import_status') == 'IMPORT_FAILED'
                   and isinstance(missing_module, str)
                   and (missing_module == 'farm_neat.training'
                        or missing_module.startswith('farm_neat.training.')))
        return {'status': 'ISOLATION_BLOCKER_REPRODUCED' if blocked else 'INCONCLUSIVE',
                'control': control, 'training_free': cold, 'copied_members': copied,
                'excluded_training_files': sorted(excluded), 'archive_produced': False,
                'limitations': ['SOURCE_IMPORT_ONLY; NOT_ARCHIVE_OR_BEHAVIOR_PARITY',
                               'TWO_FRESH_CPU_PROCESSES; ZERO_MODEL_CONSTRUCTIONS']}

# The producer authenticates the ORIGINAL COMMIT with the accepted bridge. The
# following model-only format is a DERIVED inference artifact, not a native
# checkpoint, a re-signed COMMIT, an optimizer continuation, or a quantization.
REQUEST_SCHEMA = 'farm-v0-cold-build-request-v1'
MODEL_SCHEMA = 'farm-v0-derived-inference-model-v1'
ARCHIVE_SCHEMA = 'farm-v0-derived-inference-archive-v1'
ARTIFACT_KIND = 'V0_DERIVED_INFERENCE_NOT_NATIVE_COMMIT'
MAX_BYTES = 100_000_000
MAX_MEMBERS = 512


class V0DeploymentBlocked(ValueError):
    """A reviewable failure receipt; never a successful deployment manifest."""
    def __init__(self, report):
        self.report = report
        super().__init__('V0 deployment BLOCKED: ' + ', '.join(report['blockers']))


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _identity(value):
    return {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}


def _hex(value):
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _limit(value):
    if type(value) is not int or not 1 <= value <= MAX_BYTES:
        raise ValueError('integer byte budget in [1, 100000000] required')
    return value


def _closed(value, fields, label):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError('closed ' + label + ' fields required')


def _json_read(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key: ' + key)
            out[key] = value
        return out
    def constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('valid UTF-8 JSON required') from error



def _timeout(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 120:
        raise ValueError('finite cold-probe timeout in (0,120] required')
    return value


def read_request(path, expected_sha256):
    """Authenticate the closed settings request before touching a checkpoint."""
    if not _hex(expected_sha256):
        raise ValueError('explicit request SHA256 required')
    with _regular(path).open('rb') as stream:
        raw = stream.read(2_000_001)
    if len(raw) > 2_000_000 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('request hash/size mismatch')
    return validate_request(_json_read(raw))


def _bounded_configuration(config):
    maxima = dict(width=256, layers=16, heads=128, experts=8, top_k=2,
                  expert_hidden=256, branches=4, conv_size=29)
    if (type(config) is not dict or set(config) != {*maxima, 'rms_epsilon'}
            or any(type(config[k]) is not int or not 1 <= config[k] <= high
                   for k, high in maxima.items())
            or type(config['rms_epsilon']) is not float
            or not math.isfinite(config['rms_epsilon']) or config['rms_epsilon'] <= 0):
        raise ValueError('bounded diagnostic configuration required')
    return config


def validate_request(request):
    """Closed first profile: diagnostic x86_64 CPU FP32, explicit two-seat RNG.

    Do not infer a runtime, schema, sampler or compiler setting from defaults.
    Production/other devices are refusals until a separate qualification exists.
    No import of torch, a trainer, a checkpoint or a native engine occurs here.
    """
    _closed(request, ('schema', 'kind', 'diagnostic', 'publication', 'sampler',
                     'compiler', 'numerics', 'runtime', 'action_schema_sha256',
                     'feature_schema_sha256'), 'V0 build request')
    if request['schema'] != REQUEST_SCHEMA or request['kind'] not in ('BC', 'RL', 'REGRET'):
        raise ValueError('native BC/RL/REGRET build request required; no legacy relabeling')
    if request['diagnostic'] is not True:
        raise ValueError('production V0 archive profile is NOT QUALIFIED; diagnostic=True required')
    if request['numerics'] != 'FP32_NO_AUTOCAST_NO_TF32':
        raise ValueError('unsupported V0 numerical profile; no codec/quantization conversion')
    if request['publication'] not in ('all_dawns', 'opening_only_later_keep'):
        raise ValueError('explicit supported dawn publication required')
    sampler = request['sampler']
    _closed(sampler, ('mode', 'seeds', 'temperature'), 'sampler')
    seeds, temperature = sampler['seeds'], sampler['temperature']
    if (sampler['mode'] not in ('greedy_class', 'sample_raw') or type(seeds) is not list
            or len(seeds) != 2 or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds)
            or seeds[0] == seeds[1] or type(temperature) not in (int, float)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('explicit mode, distinct two-seat seeds and positive finite temperature required')
    _closed(request['compiler'], ('labor_factor', 'hands_mode', 'belief'), 'compiler')
    compiler = request['compiler']
    if (type(compiler['labor_factor']) not in (int, float) or compiler['labor_factor'] != 4.
            or compiler['hands_mode'] != 'daily' or compiler['belief'] is not None):
        raise ValueError('first V0 archive profile pins labor_factor=4, daily hands and zero-flow belief')
    for key in ('action_schema_sha256', 'feature_schema_sha256'):
        if not _hex(request[key]):
            raise ValueError('explicit ' + key + ' required')
    runtime = request['runtime']
    _closed(runtime, ('runtime_key', 'machine', 'python', 'executable', 'torch', 'numpy',
                      'device', 'threads', 'interop_threads', 'deterministic', 'matmul_precision',
                      'tf32_matmul', 'tf32_cudnn', 'environment'), 'CPU runtime identity')
    if (runtime['device'] != 'cpu' or runtime['runtime_key'] != 'training'
            or runtime['machine'] != 'x86_64'):
        raise ValueError('only explicit diagnostic x86_64 CPU runtime supported; no cross-runtime remap')
    if (any(type(runtime[k]) is not str or not runtime[k] for k in
            ('python', 'executable', 'torch', 'numpy'))
            or any(type(runtime[k]) is not int or runtime[k] < 1 for k in ('threads', 'interop_threads'))
            or runtime['deterministic'] is not True or runtime['tf32_matmul'] is not False
            or runtime['tf32_cudnn'] is not False or runtime['matmul_precision'] != 'highest'):
        raise ValueError('explicit deterministic FP32 runtime identity required')
    env = runtime['environment']
    _closed(env, ('CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'),
            'runtime environment')
    if any(v is not None and type(v) is not str for v in env.values()):
        raise ValueError('runtime environment values must be strings or null')
    _encoded(request)  # Reject non-JSON objects; do not normalize or drop fields.
    return request


def _cpu_runtime_facts():
    """CPU subset of the accepted runtime join, without importing its trainer.

    Fixture tests compare the full mapping against v0_bc.runtime_identity. No
    CUDA/aarch64 profile is inferred. The producer must already have passed the
    original bridge's runtime join; this checks the derived consumer independently.
    """
    import platform
    import numpy as np
    import torch
    return {'runtime_key': 'training', 'machine': platform.machine(),
            'python': '.'.join(map(str, sys.version_info[:3])),
            'executable': str(Path(sys.executable).resolve()), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'device': 'cpu', 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'matmul_precision': torch.get_float32_matmul_precision(),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'environment': {k: os.environ.get(k) for k in
                ('CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}


def _validate_settings(settings):
    _closed(settings, ('schema', 'artifact_kind', 'request', 'identity', 'model'), 'model settings')
    if settings['schema'] != MODEL_SCHEMA or settings['artifact_kind'] != ARTIFACT_KIND:
        raise ValueError('derived V0 inference settings required; not a native COMMIT')
    request = validate_request(settings['request'])
    identity = settings['identity']
    _closed(identity, ('kind', 'commit_sha256', 'state_sha256', 'model_identity',
                       'training_identity_sha256', 'runtime_sha256', 'diagnostic'), 'actor identity')
    if identity['kind'] != request['kind'] or identity['diagnostic'] is not True:
        raise ValueError('checkpoint kind/diagnostic identity mismatch')
    if any(not _hex(identity[k]) for k in ('commit_sha256', 'state_sha256', 'training_identity_sha256', 'runtime_sha256')):
        raise ValueError('complete authenticated actor identities required')
    model = identity['model_identity']
    if (type(model) is not dict or model.get('diagnostic') is not True
            or model.get('action_schema_sha256') != request['action_schema_sha256']
            or model.get('feature_schema_sha256') != request['feature_schema_sha256']
            or model.get('numerics') != request['numerics']):
        raise ValueError('action/feature/model numerical identity mismatch')
    if identity['runtime_sha256'] != hashlib.sha256(_encoded(request['runtime'])).hexdigest():
        raise ValueError('explicit runtime does not match authenticated checkpoint runtime')
    _closed(settings['model'], ('sha256', 'bytes'), 'model bytes identity')
    if not _hex(settings['model']['sha256']):
        raise ValueError('model SHA256 required')
    _limit(settings['model']['bytes'])
    _encoded(settings)
    return settings



def encode_settings(settings):
    """Validate, then retain actor-identity insertion order on the transport wire.

    The accepted sampler/continuation hash includes the nested actor identity.
    Sorting immutable-looking settings changes that wire even with equal values.
    Runtime-digest joins still use the owner's separate canonical sorted codec.
    """
    _validate_settings(settings)
    return json.dumps(settings, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()


def prepare_export(directory, commit_sha256, request, *, max_bytes=MAX_BYTES):
    """Authenticate once through the bridge, then export ONLY its inference state.

    This in-memory preparation is useful independently of the blocked source
    closure. It never writes an output, changes the original COMMIT, constructs a
    Trainer, or calls torch.load on a pruned/re-signed native checkpoint.
    """
    import copy
    import io
    validate_request(request)
    _limit(max_bytes)
    if not _hex(commit_sha256):
        raise ValueError('explicit lowercase original COMMIT SHA256 required')
    directory = Path(directory).absolute()
    commit, state = _regular(directory/'COMMIT.json'), _regular(directory/'state.pt')
    if commit.stat().st_size > 2_000_000 or state.stat().st_size > max_bytes:
        raise ValueError('checkpoint input exceeds explicit byte budget')
    from farm_neat.evaluation import v0_policy as P
    import torch
    # Inspect through the same owner before any model allocation. This does not
    # reinterpret native identity or touch private corpus/optimizer channels.
    inspected = P.inspect_committed_scorer(directory, commit_sha256, device='cpu', production=False)
    _bounded_configuration(inspected['identity']['model_identity']['config'])
    # The bridge owns original COMMIT, state, source, schema and runtime joins.
    loaded = P.load_committed_scorer(directory, commit_sha256, device='cpu', production=False)
    request = copy.deepcopy(request)
    identity = copy.deepcopy(loaded.identity)
    model_state = loaded.scorer.state_dict()
    model_identity = loaded.scorer.get_extra_state()
    if model_identity != identity['model_identity'] or model_state.get('_extra_state') != model_identity:
        raise ValueError('bridge scorer/actor identity mismatch')
    copied = {}
    for name, value in model_state.items():
        if name == '_extra_state':
            copied[name] = copy.deepcopy(model_identity)
        elif (type(name) is str and isinstance(value, torch.Tensor)
              and value.device.type == 'cpu' and torch.isfinite(value).all().item()):
            copied[name] = value.detach().contiguous().clone()
        else:
            raise ValueError('native finite CPU model tensor required: ' + str(name))
    if sum(v.numel()*v.element_size() for v in copied.values() if isinstance(v, torch.Tensor)) > max_bytes:
        raise ValueError('model tensor bytes exceed explicit budget')
    buffer = io.BytesIO()
    torch.save(copied, buffer)
    data = buffer.getvalue()
    if len(data) > max_bytes:
        raise ValueError('serialized model exceeds explicit byte budget')
    settings = {'schema': MODEL_SCHEMA, 'artifact_kind': ARTIFACT_KIND,
                'request': request, 'identity': identity, 'model': _identity(data)}
    settings_wire = encode_settings(settings)
    # No incidental source mutation during the copy is admitted as the old commit.
    if _sha(commit) != commit_sha256 or _sha(state) != identity['state_sha256']:
        raise ValueError('original checkpoint changed during inference export')
    return {'model_bytes': data, 'settings': settings, 'settings_bytes': settings_wire,
            'receipt': {'schema': MODEL_SCHEMA, 'status': 'MODEL_ONLY_PREPARED_NOT_DEPLOYABLE',
                'checkpoint_authenticated': True, 'model_only_prepared': True,
                'original_commit_sha256': commit_sha256, 'original_state_sha256': identity['state_sha256'],
                'model': _identity(data), 'settings': _identity(settings_wire),
                'archive_produced': False, 'native_transitions': 0, 'optimizer_updates': 0,
                'scope': 'DIAGNOSTIC_CPU_DERIVED_BYTES_ONLY_NOT_COLD_OR_PRODUCTION_PARITY'}}


def load_model_payload(data, settings):
    """Strict reader for the derived artifact, reusing the ORIGINAL V0Scorer.

    The caller authenticates the settings via the outer archive hash. This is not
    an original-COMMIT loader and does not assert that source isolation passed.
    Global RNG is preserved; the bridge's CommittedScorer owns sampler creation.
    """
    import io
    import copy
    import torch
    _validate_settings(settings)
    if type(data) is not bytes or _identity(data) != settings['model']:
        raise ValueError('derived model hash/size mismatch')
    from farm_neat.evaluation import v0_policy as P
    from farm_neat.policy.v0_scorer import V0Scorer
    from farm_neat.policy.qwen_hybrid import Config
    P.inference_runtime('cpu')
    if _cpu_runtime_facts() != settings['request']['runtime']:
        raise ValueError('derived model runtime identity mismatch; no cross-runtime reinterpretation')
    identity = settings['identity']['model_identity']
    # Allocate only the bounded diagnostic graph; the constructor/schema join is
    # the original owner, not a copied feature encoder or alternate action head.
    config = _bounded_configuration(identity.get('config'))
    with torch.random.fork_rng(devices=[]):
        scorer = V0Scorer(Config(**config), diagnostic=True)
    if scorer.get_extra_state() != identity:
        raise ValueError('derived model source/action/feature identity mismatch')
    state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or set(state) != set(scorer.state_dict()):
        raise ValueError('complete model-only state inventory required')
    if state.get('_extra_state') != identity:
        raise ValueError('derived model extra-state identity mismatch')
    for name, tensor in state.items():
        if name != '_extra_state' and (not isinstance(tensor, torch.Tensor)
                or not torch.isfinite(tensor).all().item()):
            raise ValueError('finite model-only tensors required: ' + str(name))
    scorer.load_state_dict(state, strict=True)
    scorer.eval().requires_grad_(False)
    return P.CommittedScorer(scorer, copy.deepcopy(settings['identity']))


def make_agent(loaded, settings, seat):
    """Keep all publication, economics, prefix and sampling in accepted owners."""
    _validate_settings(settings)
    if type(seat) is not int or seat not in (0, 1) or loaded.identity != settings['identity']:
        raise ValueError('source-identical loaded scorer and explicit seat required')
    from farm_neat.evaluation import v0_policy as P
    request = settings['request']
    sampler = request['sampler']
    policy = loaded.policy(sampler['mode'], sampler['seeds'][seat], temperature=sampler['temperature'],
                           opening_only=request['publication'] == 'opening_only_later_keep')
    return P.PolicyAgent(policy, game_id='v0-derived-seat-' + str(seat), **request['compiler'])


def archive_closure(root, settings):
    """Conservative all-literal-import closure plus unchanged identity assets.

    Unlike the retained eager audit this includes function bodies. It is still
    not a proof about arbitrary dynamic imports: a fresh-process cold probe is a
    separate gate. No module is rewritten, relocated, stubbed or monkeypatched.
    """
    root = Path(root).absolute()
    _validate_settings(settings)
    known = {_module(p, root): p for p in sorted((root/'src/farm_neat').rglob('*.py'))}
    pins = settings['identity']['model_identity']['source_pins']
    from farm_neat.evaluation import v0_policy as P
    for name, digest in P.source_pins().items():
        if name in pins and pins[name] != digest:
            raise ValueError('conflicting model/runtime source pins: ' + name)
    pins = {**pins, **P.source_pins()}
    entries = (*ENTRIES, 'farm_neat.evaluation.v0_deployment_runtime')
    pending = deque((name, [name]) for name in entries)
    members, seen, forbidden, missing, external = {}, set(), [], [], set()
    while pending:
        name, chain = pending.popleft()
        if name in seen:
            continue
        seen.add(name)
        if name not in known:
            missing.append({'module': name, 'chain': chain})
            continue
        path = _regular(known[name]); raw = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        members[relative] = _identity(raw)
        if any(name == p or name.startswith(p+'.') for p in FORBIDDEN):
            forbidden.append({'module': name, 'path': relative, 'chain': chain})
        tree = ast.parse(raw, filename=relative)
        # _targets on import-only nodes visits function imports without executing.
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        targets, absent, dependencies = _targets(path, name, known,
            tree=ast.Module(body=imports, type_ignores=[]))
        external.update(dependencies)
        missing.extend({'module': m, 'chain': [*chain, m]} for m in sorted(absent))
        pending.extend((n, [*chain, n]) for n in sorted(targets) if n not in seen)
    for relative in sorted(set(ASSETS) | set(pins)):
        path = _regular(root/relative)
        raw = path.read_bytes()
        if relative in pins and hashlib.sha256(raw).hexdigest() != pins[relative]:
            raise ValueError('model/runtime source pin changed during export: ' + relative)
        members[relative] = _identity(raw)
        if relative.startswith(('src/farm_neat/training/', 'src/farm_neat/rollout/')) and not any(
                f['path'] == relative for f in forbidden):
            forbidden.append({'path': relative, 'chain': ['IDENTITY_ASSET', relative]})
    return {'members': dict(sorted(members.items())), 'forbidden': forbidden, 'missing': missing,
            'external_import_roots': sorted(external),
            'scope': 'ALL_LITERAL_IMPORTS_AND_PINNED_ASSETS_NOT_DYNAMIC_CLOSURE_PROOF'}

PROBE_INPUT_SCHEMA = 'farm-v0-offline-public-observation-probe-v1'


def offline_witness(loaded, settings, inputs):
    """Own-generated prefixes and fresh sampler/ordered-memory restoration.

    Observations are supplied test inputs, NOT a simulated rollout. The API has
    no teacher Plan, prefix, action, or externally manufactured continuation input.
    It calls the accepted compiler/executor but never calls an engine transition.
    """
    import copy
    from farm_neat.evaluation import v0_policy as P
    from farm_neat.economy.v0.plan import Plan
    _closed(inputs, ('schema', 'seat', 'observations'), 'offline probe input')
    observations, seat = inputs['observations'], inputs['seat']
    if (inputs['schema'] != PROBE_INPUT_SCHEMA or type(seat) is not int or seat not in (0, 1)
            or type(observations) is not list or len(observations) not in (25, 49)):
        raise ValueError('explicit seat and 25 or 49 consecutive public probe observations required')
    previous = -1
    for observation in observations:
        if (not isinstance(observation, dict) or type(observation.get('step')) is not int
                or observation['step'] != previous + 1 or observation.get('player') != seat):
            raise ValueError('consecutive same-seat observations required')
        previous = observation['step']
    if observations[0]['step'] != 0:
        raise ValueError('probe must generate its own opening prefix from step zero')
    agent = make_agent(loaded, settings, seat)
    rows, resumes, nontrivial = [], [], False
    for index, observation in enumerate(observations):
        restored = None
        if index and observation['step'] in P.DAWNS:
            cut = P.snapshot_agent(agent, observation)
            nontrivial |= cut['plan'] != Plan().canonical()
            # P.wire preserves nested dict insertion order. Never use the sorted
            # JSON canonical-hash codec for continuation or sampler state.
            restored = P.restore_agent(json.loads(P.wire(cut)), observation,
                                       policy=make_agent(loaded, settings, seat).policy)
            if P.wire(P.snapshot_agent(restored, observation)) != P.wire(cut):
                raise ValueError('order-losing continuation round trip')
        action = agent(copy.deepcopy(observation))
        if restored is not None:
            resumed_action = restored(copy.deepcopy(observation))
            check = {'step': observation['step'], 'exact_plan_equality': agent.plan == restored.plan,
                     'compiled_action_equality': P.wire(action) == P.wire(resumed_action),
                     'sampler_equality': P.wire(agent.policy.snapshot_state()) == P.wire(restored.policy.snapshot_state()),
                     'ordered_continuation_equality': P.agent_signature(agent) == P.agent_signature(restored)}
            resumes.append(check)
            if not all(value for key, value in check.items() if key != 'step'):
                raise ValueError('own-prefix resumed Plan/action/sampler/continuation divergence')
        trace = getattr(agent.policy, 'last_trace', None)
        rows.append({'step': observation['step'], 'compiled_action': action,
                     'exact_plan': agent.plan.canonical(),
                     'raw_edits': None if trace is None else trace['actions'],
                     'trace_sha256': P.wire_hash(trace),
                     'sampler_sha256': P.wire_hash(agent.policy.snapshot_state()),
                     'ordered_continuation_sha256': P.agent_signature(agent)})
    return {'schema': PROBE_INPUT_SCHEMA, 'scope': 'SUPPLIED_OBSERVATIONS_DIAGNOSTIC_NOT_NATIVE_OR_COLD_PARITY',
            'seat': seat, 'rows': rows, 'resume_checks': resumes,
            'nontrivial_own_resume': bool(nontrivial), 'native_transitions': 0, 'teacher_prefixes': False}


def _name(name):
    from pathlib import PurePosixPath
    if type(name) is not str or len(name) > 500:
        raise ValueError('bounded canonical archive name required')
    p = PurePosixPath(name)
    if (not name or name == '.' or p.is_absolute() or str(p) != name
            or any(v in ('..', '.') for v in p.parts) or '\\' in name or '\0' in name):
        raise ValueError('canonical relative archive name required')
    return name


def read_archive_bytes(archive, expected_sha256, *, max_bytes=MAX_BYTES):
    """Bound gzip expansion before parsing tar; verify before any extraction.

    This low-level byte reader does NOT admit a V0 model or execute an entrypoint.
    Compressed bytes, expanded tar bytes, member count and names are bounded.
    """
    import gzip
    import io
    import tarfile
    _limit(max_bytes)
    if not _hex(expected_sha256):
        raise ValueError('explicit archive SHA256 required')
    path = _regular(archive)
    with path.open('rb') as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('archive hash/size mismatch or explicit byte budget exceeded')
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as zipped:
            expanded = zipped.read(max_bytes + 1)
        if len(expanded) > max_bytes:
            raise ValueError('expanded archive exceeds explicit byte budget')
        result = {}
        with tarfile.open(fileobj=io.BytesIO(expanded), mode='r:') as tar:
            for member in tar:
                name = _name(member.name)
                if (len(result) >= MAX_MEMBERS or name in result or not member.isfile()
                        or member.size < 0 or member.size > max_bytes):
                    raise ValueError('bounded unique regular archive members required')
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError('regular member data missing')
                data = stream.read(max_bytes + 1)
                if len(data) != member.size:
                    raise ValueError('archive member size mismatch')
                result[name] = data
        return result
    except (OSError, EOFError, tarfile.TarError) as error:
        raise ValueError('valid bounded gzip/tar archive required') from error


def main_source(settings_sha256):
    if not _hex(settings_sha256):
        raise ValueError('settings SHA256 required')
    return ("# Diagnostic V0 inference artifact; no production qualification.\n"
            "import os, sys\nfrom pathlib import Path\n"
            "os.environ['CUDA_VISIBLE_DEVICES'] = ''\n"
            "root = Path(__file__).resolve().parent\n"
            "sys.path.insert(0, str(root/'src'))\n"
            "from farm_neat.evaluation.v0_deployment_runtime import Resident\n"
            f"agent = Resident(root, {settings_sha256!r})\n").encode()


def _validate_members(members):
    if 'MANIFEST.json' not in members or len(members['MANIFEST.json']) > 2_000_000:
        raise ValueError('bounded V0 manifest required')
    manifest = _json_read(members['MANIFEST.json'])
    _closed(manifest, ('schema', 'artifact_kind', 'scope', 'settings_sha256', 'members'), 'V0 archive manifest')
    if (manifest['schema'] != ARCHIVE_SCHEMA or manifest['artifact_kind'] != ARTIFACT_KIND
            or manifest['scope'] != 'DIAGNOSTIC_CPU_ARCHIVE_NOT_PRODUCTION_QUALIFIED'):
        raise ValueError('explicit diagnostic derived V0 archive required')
    inventory = manifest['members']
    if type(inventory) is not dict or set(inventory) | {'MANIFEST.json'} != set(members):
        raise ValueError('complete exact archive inventory required')
    for name, expected in inventory.items():
        _name(name)
        _closed(expected, ('sha256', 'bytes'), 'archive member identity')
        if expected != _identity(members[name]):
            raise ValueError('archive member hash/size mismatch: ' + name)
        if name.startswith(('src/farm_neat/training/', 'src/farm_neat/rollout/')):
            raise ValueError('training/rollout sources forbidden in V0 archive: ' + name)
        allowed = (name in {'MODEL.pt', 'SETTINGS.json', 'main.py', *ASSETS}
                   or name.startswith('src/farm_neat/') and name.endswith('.py'))
        if not allowed:
            raise ValueError('incidental/private archive member refused: ' + name)
    required = {'MODEL.pt', 'SETTINGS.json', 'main.py', *ASSETS,
                'src/farm_neat/__init__.py', 'src/farm_neat/evaluation/__init__.py',
                'src/farm_neat/evaluation/v0_deployment_runtime.py',
                'src/farm_neat/evaluation/v0_policy.py', 'src/farm_neat/policy/v0_scorer.py',
                'src/farm_neat/policy/qwen_hybrid.py'}
    if not required <= set(inventory):
        raise ValueError('missing runtime dependencies: ' + ', '.join(sorted(required - set(inventory))))
    settings = _json_read(members['SETTINGS.json'])
    _validate_settings(settings)
    if (manifest['settings_sha256'] != hashlib.sha256(members['SETTINGS.json']).hexdigest()
            or _identity(members['MODEL.pt']) != settings['model']):
        raise ValueError('archive settings/model identity mismatch')
    if members['main.py'] != main_source(manifest['settings_sha256']):
        raise ValueError('archive entrypoint identity mismatch')
    for name, digest in settings['identity']['model_identity']['source_pins'].items():
        if name not in inventory or inventory[name]['sha256'] != digest:
            raise ValueError('model source identity dependency missing or changed: ' + name)
    return manifest, settings


def validate_archive(archive, expected_sha256, *, max_bytes=MAX_BYTES):
    """Authenticate data/source inventory only. No model/cold/strength claim."""
    members = read_archive_bytes(archive, expected_sha256, max_bytes=max_bytes)
    manifest, settings = _validate_members(members)
    return members, manifest, settings


def _directory(root, settings_sha256):
    root = Path(root).absolute()
    _regular(root/'MANIFEST.json')
    names = [p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()]
    if len(names) > MAX_MEMBERS:
        raise ValueError('bounded runtime directory inventory required')
    members, total = {}, 0
    for name in names:
        path = _regular(root/_name(name))
        total += path.stat().st_size
        if total > MAX_BYTES:
            raise ValueError('runtime directory exceeds byte budget')
        members[name] = path.read_bytes()
    manifest, settings = _validate_members(members)
    if not _hex(settings_sha256) or manifest['settings_sha256'] != settings_sha256:
        raise ValueError('runtime settings identity mismatch')
    return members, manifest, settings



def configure_runtime(request):
    """Initialize the explicit diagnostic CPU profile for the actual entrypoint.

    A probe must not do setup that main.py/Resident omits. Version, executable and
    environment mismatches remain refusals; only the requested numerical/thread
    settings are applied. No seed, model, trainer or native transition is used.
    Interop reconfiguration after parallel work requires a fresh process.
    """
    validate_request(request)
    expected = request['runtime']
    if {key: os.environ.get(key) for key in expected['environment']} != expected['environment']:
        raise ValueError('entrypoint runtime environment mismatch; start with the pinned environment')
    actual = _cpu_runtime_facts()
    fixed = ('runtime_key', 'machine', 'python', 'executable', 'torch', 'numpy', 'device', 'environment')
    if any(actual[key] != expected[key] for key in fixed):
        raise ValueError('entrypoint runtime identity mismatch; no cross-runtime reinterpretation')
    import torch
    from farm_neat.evaluation import v0_policy as P
    try:
        if torch.get_num_interop_threads() != expected['interop_threads']:
            torch.set_num_interop_threads(expected['interop_threads'])
        if torch.get_num_threads() != expected['threads']:
            torch.set_num_threads(expected['threads'])
    except RuntimeError as error:
        raise ValueError('entrypoint thread initialization requires a fresh process') from error
    P.inference_runtime('cpu')
    actual = _cpu_runtime_facts()
    if actual != expected:
        raise ValueError('entrypoint numerical/runtime identity mismatch after initialization')
    return actual


class Resident:
    """Two own-prefix CPU actors; the shared V0 policy owns all economic choices.

    Create a new resident for a new physical game. Continuations use the existing
    order-preserving policy/agent codec, not this module's sorted settings JSON.
    """
    def __init__(self, root, settings_sha256):
        members, _, self.settings = _directory(root, settings_sha256)
        configure_runtime(self.settings['request'])
        self.loaded = load_model_payload(members['MODEL.pt'], self.settings)
        self.agents = {}

    def _seat(self, observation):
        seat = observation.get('player') if isinstance(observation, dict) else None
        if type(seat) is not int or seat not in (0, 1):
            raise ValueError('explicit observation player seat required')
        if seat not in self.agents:
            self.agents[seat] = make_agent(self.loaded, self.settings, seat)
        return seat

    def __call__(self, observation, configuration=None):
        return self.agents[self._seat(observation)](observation)

    def snapshot(self, observation):
        from farm_neat.evaluation import v0_policy as P
        return P.snapshot_agent(self.agents[self._seat(observation)], observation)

    def restore(self, payload, observation):
        from farm_neat.evaluation import v0_policy as P
        seat = self._seat(observation)
        fresh = make_agent(self.loaded, self.settings, seat)
        self.agents[seat] = P.restore_agent(payload, observation, policy=fresh.policy)


COLD_WORKER_SCHEMA = 'farm-v0-cold-entrypoint-worker-v1'
COLD_SUCCESS = 'COLD_ENTRYPOINT_LOADED_DIAGNOSTIC'

_COLD_ARCHIVE = r'''
import importlib.abc, json, pathlib, runpy, site, sys
root = pathlib.Path(sys.argv[1]).resolve()
settings_sha = sys.argv[2]
sys.path.insert(0, str(root/'src'))
class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p+'.') for p in ('farm_neat.training', 'farm_neat.rollout')):
            raise ImportError('training/rollout module forbidden in cold V0 runtime: ' + fullname)
sys.meta_path.insert(0, NoTraining())
result = {'schema': 'farm-v0-cold-entrypoint-worker-v1',
          'isolated': bool(sys.flags.isolated), 'user_site_enabled': site.ENABLE_USER_SITE,
          'native_transitions': 0, 'production_qualified': False}
try:
    from farm_neat.evaluation import v0_deployment_runtime as R
    _, manifest, settings = R._directory(root, settings_sha)
    namespace = runpy.run_path(str(root/'main.py'), run_name='__farm_v0_archive_entrypoint__')
    resident = namespace.get('agent')
    if not isinstance(resident, R.Resident):
        raise ValueError('archive main.py did not expose the V0 Resident')
    result.update(status='COLD_ENTRYPOINT_LOADED_DIAGNOSTIC', model_loaded=True,
                  entrypoint_loaded=True, identity=resident.loaded.identity,
                  settings_sha256=manifest['settings_sha256'],
                  entrypoint_sha256=manifest['members']['main.py']['sha256'],
                  runtime=R._cpu_runtime_facts())
except Exception as error:
    result.update(status='BLOCKED', model_loaded=False, entrypoint_loaded=False,
                  error_type=type(error).__name__, error=str(error))
loaded = {n: str(pathlib.Path(m.__file__).resolve()) for n, m in sorted(sys.modules.items())
          if n.startswith('farm_neat') and getattr(m, '__file__', None)}
result['all_package_files_in_archive'] = bool(loaded) and all(pathlib.Path(p).is_relative_to(root) for p in loaded.values())
result['training_modules'] = [n for n in loaded if n.startswith(('farm_neat.training', 'farm_neat.rollout'))]
if not result['isolated'] or result['user_site_enabled'] or not result['all_package_files_in_archive'] or result['training_modules']:
    result.update(status='BLOCKED', model_loaded=False, entrypoint_loaded=False, error='cold source isolation failed')
# Identity order participates in the actor's sampler/continuation wire. Do not
# sort this transport: value-equal identities can still have different wires.
print(json.dumps(result, ensure_ascii=True, separators=(',', ':'), allow_nan=False))
sys.exit(0 if result['status'] == 'COLD_ENTRYPOINT_LOADED_DIAGNOSTIC' else 2)
'''


def _admit_cold_result(report, manifest, settings):
    """Join a complete worker message to authenticated inputs before any PASS.

    This is a protocol check, not a sandbox against hostile Python. The archive
    itself must already be authenticated and structurally admitted. A zero exit
    code or status string alone is not evidence that its entrypoint loaded.
    """
    _closed(report, ('schema', 'status', 'model_loaded', 'entrypoint_loaded',
                     'settings_sha256', 'entrypoint_sha256', 'identity', 'runtime',
                     'isolated', 'user_site_enabled', 'all_package_files_in_archive',
                     'training_modules', 'native_transitions', 'production_qualified'),
            'cold entrypoint worker result')
    required_bools = {'model_loaded': True, 'entrypoint_loaded': True, 'isolated': True,
                      'user_site_enabled': False, 'all_package_files_in_archive': True,
                      'production_qualified': False}
    if (report['schema'] != COLD_WORKER_SCHEMA or report['status'] != COLD_SUCCESS
            or any(report[key] is not value for key, value in required_bools.items())
            or type(report['native_transitions']) is not int or report['native_transitions'] != 0
            or type(report['training_modules']) is not list or report['training_modules']):
        raise ValueError('complete isolated diagnostic entrypoint-load evidence required')
    if (report['settings_sha256'] != manifest['settings_sha256']
            or report['entrypoint_sha256'] != manifest['members']['main.py']['sha256']
            or _encoded(report['runtime']) != _encoded(settings['request']['runtime'])):
        raise ValueError('cold worker settings/entrypoint/runtime identity mismatch')
    def ordered(value):
        return json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    if ordered(report['identity']) != ordered(settings['identity']):
        raise ValueError('cold worker actor identity/order mismatch')
    return report


def probe_archive(archive, expected_sha256, *, max_bytes=MAX_BYTES, timeout=30):
    """Fresh -I CPU load from an authenticated temporary tree, outside checkout.

    Execute main.py/Resident, with no probe-only torch initialization. This is
    an entrypoint LOAD gate, not a complete-game or opening/resumed parity result.
    """
    _timeout(timeout)
    members, manifest, settings = validate_archive(archive, expected_sha256, max_bytes=max_bytes)
    with tempfile.TemporaryDirectory(prefix='farm-v0-archive-cold-') as temporary:
        home = Path(temporary); root = home/'actor'; root.mkdir()
        for name, data in members.items():
            path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        env = {'HOME': str(home), 'TMPDIR': str(home), 'PATH': os.path.dirname(sys.executable),
               'CUDA_VISIBLE_DEVICES': ''}
        env.update({k: v for k, v in settings['request']['runtime']['environment'].items() if v is not None})
        try:
            proc = subprocess.run([sys.executable, '-I', '-B', '-c', _COLD_ARCHIVE, str(root),
                                   manifest['settings_sha256']], cwd=home, env=env, capture_output=True,
                                  text=True, timeout=timeout)
            worker_result = None
            try:
                worker_result = _json_read(proc.stdout)
                if proc.returncode != 0:
                    raise ValueError('cold entrypoint worker returned nonzero exit status')
                report = _admit_cold_result(worker_result, manifest, settings)
            except ValueError as error:
                report = {'status': 'BLOCKED', 'model_loaded': False, 'entrypoint_loaded': False,
                          'error': str(error), 'worker_result': worker_result}
            report.update(returncode=proc.returncode, stderr=proc.stderr)
        except subprocess.TimeoutExpired:
            report = {'status': 'BLOCKED', 'model_loaded': False, 'entrypoint_loaded': False, 'error': 'cold-probe timeout'}
    return {**report, 'schema': ARCHIVE_SCHEMA, 'archive_sha256': expected_sha256,
            'settings_sha256': manifest['settings_sha256'], 'native_transitions': 0,
            'scope': 'ISOLATED_DIAGNOSTIC_ENTRYPOINT_LOAD_NOT_PLAN_OR_NATIVE_PARITY',
            'worker_schema': COLD_WORKER_SCHEMA,
            'deployment_ready': False, 'production_qualified': False}


def publish_archive(root, prepared, closure, output, *, max_bytes, timeout, writer):
    """Private staging -> bounded readback -> isolated cold load -> no-clobber link.

    The current source never reaches this stage: model/runtime source pins still
    require forbidden training files. Keeping staging transactional is important
    when the read-only upstream runtime eventually supplies a valid closure.
    """
    import os
    _limit(max_bytes)
    output, root = Path(output).absolute(), Path(root).absolute()
    payload = {'MODEL.pt': prepared['model_bytes'], 'SETTINGS.json': encode_settings(prepared['settings'])}
    if (payload['SETTINGS.json'] != prepared['settings_bytes']
            or _identity(payload['SETTINGS.json']) != prepared['receipt']['settings']):
        raise ValueError('settings changed after authenticated export')
    settings_sha = hashlib.sha256(payload['SETTINGS.json']).hexdigest()
    payload['main.py'] = main_source(settings_sha)
    for name, identity in closure['members'].items():
        data = _regular(root/_name(name)).read_bytes()
        if _identity(data) != identity:
            raise ValueError('source changed before archive staging: ' + name)
        payload[name] = data
    if sum(map(len, payload.values())) > max_bytes:
        raise ValueError('archive source/model/settings exceed explicit byte budget')
    manifest = {'schema': ARCHIVE_SCHEMA, 'artifact_kind': ARTIFACT_KIND,
                'scope': 'DIAGNOSTIC_CPU_ARCHIVE_NOT_PRODUCTION_QUALIFIED',
                'settings_sha256': settings_sha,
                'members': {name: _identity(data) for name, data in sorted(payload.items())}}
    payload['MANIFEST.json'] = _encoded(manifest)
    _validate_members(payload)
    # No output directories are created on admission failures. Publication uses
    # a same-filesystem hard link, which refuses existing destinations atomically.
    if not output.parent.is_dir():
        raise ValueError('existing V0 output parent required for atomic publication')
    with tempfile.TemporaryDirectory(prefix='.v0-staging-', dir=output.parent) as temp:
        staged = Path(temp)/'actor.tar.gz'
        writer(staged, payload)
        digest = _sha(staged)
        cold = probe_archive(staged, digest, max_bytes=max_bytes, timeout=timeout)
        if (cold.get('status') != COLD_SUCCESS or cold.get('model_loaded') is not True
                or cold.get('entrypoint_loaded') is not True):
            raise V0DeploymentBlocked({'status': 'BLOCKED', 'archive_produced': False,
                                       'blockers': ['COLD_ENTRYPOINT_LOAD_FAILED'], 'cold_probe': cold})
        if any(p.is_symlink() for p in (output, *output.parents)):
            raise ValueError('symlinked publication path refused')
        with staged.open('rb') as stream:
            os.fsync(stream.fileno())
        os.link(staged, output)
    return {'schema': ARCHIVE_SCHEMA, 'status': 'BUILT_DIAGNOSTIC_NOT_QUALIFIED',
            'archive_produced': True, 'archive_sha256': digest, 'archive_bytes': output.stat().st_size,
            'settings_sha256': settings_sha, 'model': prepared['settings']['model'],
            'cold_probe': cold, 'deployment_ready': False, 'production_qualified': False,
            'not_run': ['ISOLATED_OPENING_AND_RESUMED_PARITY', 'NATIVE_FULL_GAME_PARITY',
                        'PRODUCTION_PLATFORM_QUALIFICATION', 'PROMOTION_OR_SUBMISSION']}
