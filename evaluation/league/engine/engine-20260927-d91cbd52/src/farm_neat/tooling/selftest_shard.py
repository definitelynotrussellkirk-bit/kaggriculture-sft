"""Duration-balanced portable review tests, using review_package's unchanged runner.

Plan and run K of N, then merge exactly N complete, source-bound receipts. Only
shard 1 runs the advertised checks; every portable test file runs exactly once.
This module owns scheduling/receipts, not test selection, test execution, exclusions
or the source manifest. A receipt is integrity-checked evidence, not a signature.
A file is indivisible: sharding cannot make a >300-second file finish in <300 s.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT / 'src'))

from farm_neat.tooling import review_package as package

DEFAULT_TIMINGS = 'work/runs/reviewer-play/sanity-review7/full_self_test_timings.csv'
PLAN_SCHEMA = 'farm-review-self-test-shard-plan-v1'
RECEIPT_SCHEMA = 'farm-review-self-test-shard-receipt-v1'


def _hash(value) -> str:
    return hashlib.sha256(package.review_intake.canonical(value)).hexdigest()


def seal(value: dict) -> dict:
    """Bind receipt bytes excluding its own digest. Does not certify execution."""
    body = {key: item for key, item in value.items() if key != 'receipt_sha256'}
    return {**body, 'receipt_sha256': _hash(body)}


def _seconds(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('duration must be a finite nonnegative number')
    return float(value)


def partition(files: list[str], durations: dict[str, float], count: int,
              *, check_seconds: float = 0) -> list[list[str]]:
    """Deterministic LPT: largest predicted file first, lowest loaded bin first.

    Checks form one indivisible bundle owned by bin 1. Millisecond integer loads,
    lexical file ties and bin-index ties make partitions independent of input order.
    Unknown files use the median supplied duration, not exclusion or zero work.
    """
    if not files or len(files) != len(set(files)):
        raise ValueError('test selection must be nonempty and unique')
    if type(count) is not int or not 1 <= count <= len(files):
        raise ValueError('shard count must be between 1 and the portable file count')
    for name in files:
        package.layout.safe_name(name)
    values = {name: _seconds(value) for name, value in durations.items()}
    fallback = statistics.median(values.values()) if values else 1.0
    weights = {name: round(1000 * values.get(name, fallback)) for name in files}
    bins: list[list[str]] = [[] for _ in range(count)]
    loads = [round(1000 * _seconds(check_seconds))] + [0] * (count - 1)
    for name in sorted(files, key=lambda item: (-weights[item], item)):
        index = min(range(count), key=lambda i: (loads[i], len(bins[i]), i))
        bins[index].append(name)
        loads[index] += weights[name]
    return [sorted(part) for part in bins]


def _snapshot(root: Path) -> dict[str, str | None]:
    """Bind maintained source (including additions) and all shipped input bytes.

    Unlisted generated work is excluded: progress files and parallel test outputs
    are not source. Original manifest bytes and manifest-listed work stay bound.
    Never rewrite the manifest or use inventory(), which refuses candidate drift.
    """
    names = {name for name in package.ROOT_FILES if (root / name).exists()}
    for tree in package.TREES:
        base = package.layout.checked_path(root, tree)
        if base.exists():
            for path in base.rglob('*'):
                if set(path.relative_to(base).parts) & package.SKIP_DIRS:
                    continue
                if path.is_symlink():
                    raise ValueError(f'symlink in source snapshot: {path}')
                if path.is_file():
                    names.add(path.relative_to(root).as_posix())
    manifest = package.layout.checked_path(root, package.MANIFEST)
    if manifest.is_file():
        names.add(package.MANIFEST)
        names.update(package.review_intake.parse_json(manifest.read_bytes())['members'])
    return {name: package.digest(path.read_bytes()) if path.is_file() else None
            for name in sorted(names)
            for path in [package.layout.checked_path(root, name)]}


def _timings(root: Path, name: str, files: list[str], checks: list[dict]) -> tuple[dict, dict, float]:
    raw = package.layout.checked_path(root, name).read_bytes()
    rows = csv.DictReader(io.StringIO(raw.decode('utf-8')))
    if not {'kind', 'name', 'seconds'} <= set(rows.fieldnames or []):
        raise ValueError('timings CSV requires kind,name,seconds columns')
    observed: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row['kind'], row['name'])
        if key[0] not in ('test', 'check') or key in observed:
            raise ValueError('timings need unique test/check rows')
        observed[key] = _seconds(float(row['seconds']))
    known = {name: observed['test', name] for name in files if ('test', name) in observed}
    fallback = statistics.median(known.values()) if known else 1.0
    durations = {name: known.get(name, fallback) for name in files}
    check_seconds = sum(observed.get(('check', check['name']), fallback) for check in checks)
    return ({'path': name, 'sha256': package.digest(raw), 'fallback_seconds': fallback},
            durations, check_seconds)


def build_plan(root: Path, count: int, *, timings: str | None = None) -> dict:
    """Read-only plan over precisely the canonical portable selection."""
    root = Path(root).resolve()
    spec = tomllib.loads(package.layout.checked_path(root, package.CONTEXT_SPEC).read_text())
    files, excluded, deselected = package.portable_tests(root, spec)
    checks = spec.get('checks', [])
    if len({row['name'] for row in checks}) != len(checks):
        raise ValueError('advertised check names must be unique')
    profile, durations, check_seconds = _timings(root, timings or DEFAULT_TIMINGS, files, checks)
    bins = partition(files, durations, count, check_seconds=check_seconds)
    snapshot = _snapshot(root)
    measured = {row['name'] for row in csv.DictReader(io.StringIO(
        package.layout.checked_path(root, profile['path']).read_text())) if row['kind'] == 'test'}
    plan = {'schema': PLAN_SCHEMA, 'algorithm': 'lpt-milliseconds-checks-on-1-lexical-ties-v1',
            'count': count, 'tree_sha256': _hash(snapshot),
            'source_manifest_sha256': snapshot.get(package.MANIFEST),
            'files': files, 'checks': checks, 'maintainer_only': excluded, 'deselected': deselected,
            'timings': profile, 'estimated_files': sorted(set(files) - measured),
            'estimated_longest_file_seconds': max(durations.values()),
            'shards': [{'index': i, 'files': part, 'checks': checks if i == 1 else [],
                        'estimated_serial_seconds': round(sum(durations[n] for n in part)
                                                          + (check_seconds if i == 1 else 0), 3)}
                       for i, part in enumerate(bins, 1)]}
    return {**plan, 'plan_sha256': _hash(plan)}


def _runtime(env: dict[str, str]) -> dict:
    versions = {}
    for name in ('pytest', 'numpy', 'torch'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    fixture = env.get('FARMNEAT_TEST_SOURCE_ROOT')
    fixture_manifest = Path(fixture) / package.MANIFEST if fixture else None
    return {'python': sys.version.split()[0], 'implementation': sys.implementation.name,
            'platform': platform.system(), 'versions': versions,
            'environment': {name: env.get(name) for name in (
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'PYTEST_ADDOPTS', 'PYTHONHASHSEED')},
            'source_fixture_manifest_sha256': package.digest(fixture_manifest.read_bytes())
            if fixture_manifest and fixture_manifest.is_file() else None}


def _target(root: Path, path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    name = path.relative_to(root).as_posix()
    if not name.startswith('work/'):
        raise ValueError('generated receipts and plans must be written under work/')
    target = package.layout.checked_path(root, name)
    manifest = package.layout.checked_path(root, package.MANIFEST)
    if name == package.MANIFEST or (manifest.is_file() and name in
            package.review_intake.parse_json(manifest.read_bytes())['members']):
        raise ValueError('refusing to overwrite a shipped input with generated output')
    return target


def _failed(checks: list[dict], tests: list[dict]) -> list[str]:
    return ([row['name'] for row in checks if row['returncode']]
            + [row['path'] for row in tests if row['returncode']])


def run_shard(root: Path, *, index: int, count: int, receipt_path: Path,
              jobs: int = 1, timeout: int = 1800, timings: str | None = None) -> dict:
    """Run one shard through the canonical runners, with atomic partial receipts.

    A runner timeout/interruption remains incomplete, not an invented pytest failure
    or PASS. Completed receipts are never silently overwritten; use a new run folder.
    """
    root = Path(root).resolve()
    plan = build_plan(root, count, timings=timings)
    if type(index) is not int or not 1 <= index <= count:
        raise ValueError('shard index must be in 1..N')
    if type(jobs) is not int or jobs < 1 or type(timeout) is not int or timeout < 1:
        raise ValueError('jobs and per-command timeout must be positive integers')
    selected = plan['shards'][index - 1]
    env = package._portable_environment(root)
    if env.get('PYTEST_ADDOPTS', '').strip():
        raise ValueError('unset PYTEST_ADDOPTS: inherited options can silently change test coverage')
    destination = _target(root, receipt_path)
    before = _snapshot(root)
    if _hash(before) != plan['tree_sha256']:
        raise ValueError('source changed while constructing shard plan')
    started = time.monotonic()
    row = {'schema': RECEIPT_SCHEMA, 'state': 'RUNNING', 'verdict': None,
           'plan': plan, 'index': index, 'runtime': _runtime(env),
           'workers': jobs, 'timeout_seconds': timeout, 'seconds': 0.0,
           'checks': [], 'tests': [], 'running': [], 'not_yet_run': selected['files'],
           'failed': [], 'shipped_files_changed': [], 'changed_during_run': [],
           'commands_executed': 0}
    package._atomic_json(destination, seal(row), replace=False)

    def checkpoint(tests, running, not_yet_run):
        row.update(tests=list(tests), running=list(running), not_yet_run=list(not_yet_run),
                   seconds=round(time.monotonic() - started, 3),
                   commands_executed=len(row['checks']) + len(tests))
        package._atomic_json(destination, seal(row), replace=True)

    try:
        spec = {'checks': selected['checks']}
        if selected['checks']:
            row['checks'] = package._run_checks(root, spec, env, timeout)
        checkpoint([], [], selected['files'])
        tests = package._run_test_files(root, selected['files'], plan['deselected'], env,
                                       jobs=jobs, timeout=timeout, on_progress=checkpoint)
        after = _snapshot(root)
        changes = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
        drift = package._manifest_drift(root)
        failed = _failed(row['checks'], tests)
        row.update(state='COMPLETE', verdict='FAIL' if failed or drift or changes else 'PASS',
                   tests=tests, failed=failed, shipped_files_changed=drift, changed_during_run=changes,
                   running=[], not_yet_run=[], seconds=round(time.monotonic() - started, 3),
                   commands_executed=len(row['checks']) + len(tests))
        final = seal(row)
        package._atomic_json(destination, final, replace=True)
        return final
    except BaseException as error:
        row.update(state='INTERRUPTED', verdict=None, seconds=round(time.monotonic() - started, 3),
                   error={'type': type(error).__name__, 'message': str(error)})
        package._atomic_json(destination, seal(row), replace=True)
        raise


def _validate_rows(rows: list[dict], expected: list[str], field: str) -> None:
    if not isinstance(rows, list) or [r.get(field) for r in rows if isinstance(r, dict)] != expected:
        raise ValueError(f'receipt {field} rows do not exactly match the assigned coverage')
    if len(rows) != len(expected):
        raise ValueError('invalid result rows')
    for row in rows:
        if type(row.get('returncode')) is not int:
            raise ValueError('result returncode must be an integer')
        _seconds(row.get('seconds'))
        if field == 'path' and (not isinstance(row.get('result'), str)
                or not isinstance(row.get('skipped'), list)
                or not all(isinstance(v, str) for v in row['skipped'])):
            raise ValueError('test rows must preserve result and skipped summaries')


def merge_receipts(root: Path, paths: list[Path]) -> dict:
    """Validate exact coverage/source/runtime, then produce canonical verdict fields.

    Missing/stale/incomplete/duplicate receipts raise ValueError, never PASS. File
    failures retain canonical sorted-selection order, advertised failures precede
    them, and manifest drift fails exactly as it does in review_package.self_test.
    Timing is reported as the sum and maximum of shard wall times, not a fictitious
    single-run wall time. Identical machinery cannot promise nondeterministic tests
    return identical outcomes on different hardware or different runs.
    """
    root = Path(root).resolve()
    if not paths:
        raise ValueError('no shard receipts')
    receipts = []
    for path in paths:
        path = Path(path)
        raw = (path if path.is_absolute() else root / path).read_bytes()
        if len(raw) > 8_000_000:
            raise ValueError('shard receipt exceeds 8 MB')
        row = package.review_intake.parse_json(raw)
        if not isinstance(row, dict) or row != seal(row):
            raise ValueError('shard receipt digest mismatch')
        if row.get('schema') != RECEIPT_SCHEMA or row.get('state') != 'COMPLETE':
            raise ValueError('only COMPLETE shard receipts may be merged')
        receipts.append(row)
    first = receipts[0]
    try:
        plan = build_plan(root, first['plan']['count'], timings=first['plan']['timings']['path'])
        if len(receipts) != plan['count']:
            raise ValueError('need exactly N shard receipts')
        indices = [r['index'] for r in receipts]
        if any(type(i) is not int for i in indices) or sorted(indices) != list(range(1, plan['count'] + 1)):
            raise ValueError('duplicate or missing shard index')
        drift = package._manifest_drift(root)
        ordered = sorted(receipts, key=lambda r: r['index'])
        for row in ordered:
            if row['plan'] != plan:
                raise ValueError('source, selection or timing plan differs from current tree')
            if row['runtime'] != first['runtime']:
                raise ValueError('mixed shard runtimes')
            selected = plan['shards'][row['index'] - 1]
            _validate_rows(row['tests'], selected['files'], 'path')
            _validate_rows(row['checks'], [c['name'] for c in selected['checks']], 'name')
            if row['running'] or row['not_yet_run'] or row['changed_during_run']:
                raise ValueError('shard has pending work or source changed during execution')
            if row['shipped_files_changed'] != drift:
                raise ValueError('receipt manifest drift differs from current tree')
            failed = _failed(row['checks'], row['tests'])
            if row['failed'] != failed or row['verdict'] != ('FAIL' if failed or drift else 'PASS'):
                raise ValueError('receipt verdict is inconsistent with its result rows')
            if (type(row['commands_executed']) is not int
                    or row['commands_executed'] != len(row['checks']) + len(row['tests'])):
                raise ValueError('receipt command count is inconsistent')
            _seconds(row['seconds'])
        tests_by_path = {r['path']: r for shard in ordered for r in shard['tests']}
        tests = [tests_by_path[name] for name in plan['files']]
        checks = ordered[0]['checks']
        failed = _failed(checks, tests)
        return {'schema': 'farm-review-self-test-v1', 'status': 'FAIL' if failed or drift else 'PASS',
                'meaning': 'Merged exact source-bound shards of the canonical portable runner; no scientific replay.',
                'interpreter': first['runtime']['python'], 'runtime': first['runtime'],
                'workers': max(row['workers'] for row in ordered),
                'seconds': round(sum(row['seconds'] for row in ordered), 3),
                'timing_semantics': 'seconds is sum of shard wall seconds, not observed parallel elapsed time',
                'shard_seconds': [{'index': row['index'], 'seconds': row['seconds']} for row in ordered],
                'max_shard_seconds': max(row['seconds'] for row in ordered),
                'checks': checks, 'test_files': len(tests), 'tests': tests, 'failed': failed,
                'shipped_files_changed': drift, 'maintainer_only': plan['maintainer_only'],
                'deselected': plan['deselected'], 'commands_executed': len(checks) + len(tests),
                'plan_sha256': plan['plan_sha256'],
                'shard_receipt_sha256': [row['receipt_sha256'] for row in ordered],
                'limitations': ['Only the advertised portable coverage, not maintainer-only coverage.',
                                'Receipts are integrity-checked records, not authenticated proof of execution.',
                                'File-level sharding cannot shorten one indivisible slow test file.']}
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError(f'malformed shard receipt: {error}') from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true', help='Print the deterministic partition; run nothing')
    mode.add_argument('--run', type=int, metavar='K', help='Run shard K (1-based) of --shards N')
    mode.add_argument('--merge', nargs='+', type=Path, metavar='RECEIPT', help='Merge all N completed receipts')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--shards', type=int, default=12)
    parser.add_argument('--timings', help='Repository-relative kind,name,seconds CSV (default: bound parent evidence)')
    parser.add_argument('--receipt', type=Path, help='New work/ receipt path; required with --run')
    parser.add_argument('--output', type=Path, help='Write plan or merged result under work/')
    parser.add_argument('--jobs', type=int, default=1, help='Concurrent files inside this shard (default 1)')
    parser.add_argument('--timeout', type=int, default=1800, help='Canonical per-command timeout, not a shard deadline')
    args = parser.parse_args(argv)
    if args.run is not None and (args.receipt is None or args.output is not None):
        parser.error('--run requires --receipt and does not use --output')
    if args.run is None and args.receipt is not None:
        parser.error('--receipt requires --run')
    try:
        root = args.root.resolve()
        if args.plan:
            result = build_plan(root, args.shards, timings=args.timings)
        elif args.run is not None:
            result = run_shard(root, index=args.run, count=args.shards, receipt_path=args.receipt,
                               jobs=args.jobs, timeout=args.timeout, timings=args.timings)
        else:
            if args.timings:
                parser.error('--merge reads the timing profile named by the receipts')
            result = merge_receipts(root, args.merge)
        if args.output:
            package._atomic_json(_target(root, args.output), result, replace=False)
        if args.run is not None:
            shown = {k: result[k] for k in ('state', 'verdict', 'index', 'seconds', 'failed', 'shipped_files_changed')}
            shown.update(test_files=len(result['tests']), receipt=str(args.receipt))
        elif args.output:
            shown = {k: result[k] for k in ('status', 'plan_sha256', 'test_files', 'failed', 'max_shard_seconds') if k in result}
            shown['output'] = str(args.output)
        else:
            shown = result
        print(json.dumps(shown, indent=2, allow_nan=False))
        return int(result.get('status', result.get('verdict')) == 'FAIL')
    except (ValueError, OSError, tomllib.TOMLDecodeError) as error:
        print(f'Shard error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
