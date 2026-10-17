"""Current project/ policy path for the explicitly source-pinned storage guard.

The default changes only configuration's literal metadata filename in private
function globals. External corpus commands use an owner-declared separate mount
and budget. The run --retry-scan-timeout option can suspend its
own child group and require one fresh complete scan after a periodic timeout.
The separate --pause-for-storage-check mode pauses before every periodic scan;
neither mode relaxes admission or the total deadline. Historical v1 accounting
and the current v2 free-space check retain separate declared semantics. The termination repair is
explicitly source-pinned below; never retry a job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace

from farm_neat.training import storage as original
from farm_neat.training.project_paths import project_file

# Review patch: explicitly pin the TERM/KILL and finally-cleanup repair below.
# This does not rewrite checkpoint manifests or bypass checkpoint source checks.
ORIGINAL_SHA256 = "2d92bdfbb688bd7244a772041885d44c6b48afec7841f309c82e4d2175002178"


def bindings():
    if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("storage layout adapter requires exact original guard source")
    code = original.configuration.__code__
    if code.co_consts.count("PROJECT-STORAGE.json") != 1:
        raise ValueError("storage configuration filename seam changed")
    scope = dict(vars(original))
    for name, value in vars(original).items():
        if isinstance(value, FunctionType) and value.__globals__ is vars(original):
            clone = FunctionType(value.__code__, scope, value.__name__, value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            scope[name] = clone
    def rebound(value):
        if isinstance(value, FunctionType) and value.__globals__ is vars(original):
            return scope[value.__name__]
        return value
    for name, value in vars(original).items():
        if isinstance(value, FunctionType) and value.__globals__ is vars(original):
            scope[name].__defaults__ = (tuple(rebound(v) for v in value.__defaults__)
                                       if value.__defaults__ else None)
            scope[name].__kwdefaults__ = ({k: rebound(v) for k, v in value.__kwdefaults__.items()}
                                         if value.__kwdefaults__ else None)

    def configuration(root):
        target = project_file(root, "PROJECT-STORAGE.json", legacy=True)
        literal = target.relative_to(Path(root).absolute()).as_posix()
        derived = code.replace(co_consts=tuple(literal if x == "PROJECT-STORAGE.json" else x for x in code.co_consts))
        return FunctionType(derived, scope, "configuration")(root)

    scope["configuration"] = configuration
    raw_run = scope["run"]

    def serialized_run(command, *, terminate=scope["_terminate"], **kwargs):
        inspect = kwargs.pop("inspect", scope["status"])
        def checked(root, *, reserve_bytes, scan_timeout):
            # This callback executes AFTER the original COMMAND.lock acquisition.
            # A crashed panel cannot free its durable reservation by losing a PID.
            if reserve_bytes and _panel_pending(root):
                raise ValueError("PANEL_RESERVATION_HELD: explicit safe panel close required")
            return inspect(root, reserve_bytes=reserve_bytes, scan_timeout=scan_timeout)
        return raw_run(command, inspect=checked, terminate=terminate, **kwargs)

    scope["run"] = serialized_run
    scope["raw_run"] = raw_run
    return SimpleNamespace(**scope)


def _signal_owned_group(child, number):
    """Only the live group leader returned by our start_new_session spawn."""
    if child.poll() is not None or os.getpgid(child.pid) != child.pid:
        raise ProcessLookupError("storage child no longer owns its fresh group")
    os.killpg(child.pid, number)


def run_with_scan_recheck(bound, command, *, signal_group=_signal_owned_group, **kwargs):
    """Opt-in, once per invocation: stop work, fully remeasure, then resume or kill.

    Never recovers admission failure, measured overflow, unreadability or a second
    timeout. The original total command deadline includes both scans and the pause.
    A partial count never authorizes SIGCONT; no child command is relaunched.
    The original supervisor's finally/TERM/KILL path owns all failed-paused cleanup.
    """
    clock = kwargs.get("clock", time.monotonic)
    inspect = kwargs.pop("inspect", bound.status)
    spawn = kwargs.pop("spawn", subprocess.Popen)
    started, child, attempts = clock(), None, []

    def tracked_spawn(*args, **options):
        nonlocal child
        child = spawn(*args, **options)
        return child

    def checked(root, *, reserve_bytes, scan_timeout):
        first = inspect(root, reserve_bytes=reserve_bytes, scan_timeout=scan_timeout)
        errors = first.get("errors", [])
        timed_out = (first.get("status") == "INCOMPLETE_SCAN"
            and first.get("complete") is False and first.get("admissible") is False
            and first.get("allocated_bytes", 0) <= first.get("policy", {}).get("limit_bytes", original.LIMIT)
            and errors and all(row.get("error") ==
                "storage scan deadline exceeded; partial scan is not admission" for row in errors)
            and all(row.get("error") == "storage scan deadline exceeded; partial scan is not admission"
                for attempt in first.get("scan_attempts", []) for row in attempt.get("errors", [])))
        remaining = kwargs["timeout_seconds"] - (clock() - started)
        if (not timed_out or reserve_bytes != 0 or child is None or attempts
                or remaining <= 0 or child.poll() is not None):
            return first
        event = {"failed_scan": first, "stop_requested": False, "resumed": False}
        attempts.append(event)
        signal_group(child, signal.SIGSTOP)
        event["stop_requested"] = True
        remaining = kwargs["timeout_seconds"] - (clock() - started)
        if remaining <= 0:
            event["reason"] = "TOTAL_DEADLINE_EXHAUSTED"
            return first
        second = inspect(root, reserve_bytes=0, scan_timeout=min(scan_timeout, remaining))
        event["fresh_scan"] = second
        if (second.get("complete") is True and second.get("admissible") is True
                and clock() - started < kwargs["timeout_seconds"]):
            signal_group(child, signal.SIGCONT)
            event["resumed"] = True
            return second
        event["reason"] = "FRESH_SCAN_REFUSED_OR_TOTAL_DEADLINE_EXHAUSTED"
        return second if not second.get("admissible") else first

    result = bound.run(command, inspect=checked, spawn=tracked_spawn, **kwargs)
    result["scan_timeout_recovery"] = {
        "schema": "farm-storage-paused-scan-recheck-v1", "max_attempts": 1,
        "policy": "SIGSTOP_OWN_GROUP_THEN_COMPLETE_IN_BUDGET_SCAN_BEFORE_SIGCONT",
        "job_restarts": 0, "attempts": attempts}
    return result


def run_with_paused_scans(bound, command, *, signal_group=_signal_owned_group, **kwargs):
    """Pause only the owned live group during each periodic complete scan.

    Admission and final scans are ordinary scans. Failed scans leave the group
    stopped for the original supervisor's bounded cleanup. Time spent paused is
    included in the original command deadline; no scan or job is retried here.
    """
    clock = kwargs.get("clock", time.monotonic)
    inspect = kwargs.pop("inspect", bound.status)
    spawn = kwargs.pop("spawn", subprocess.Popen)
    started, child, pauses = clock(), None, []

    def tracked_spawn(*args, **options):
        nonlocal child
        child = spawn(*args, **options)
        return child

    def checked(root, *, reserve_bytes, scan_timeout):
        if reserve_bytes != 0 or child is None or child.poll() is not None:
            return inspect(root, reserve_bytes=reserve_bytes, scan_timeout=scan_timeout)
        event = {"stop_requested": False, "resumed": False}
        pauses.append(event)
        tick = clock()
        try:
            signal_group(child, signal.SIGSTOP)
        except ProcessLookupError:
            if child.poll() is None:
                raise
            event["reason"] = "CHILD_EXITED_BEFORE_PAUSE"
            return inspect(root, reserve_bytes=reserve_bytes, scan_timeout=scan_timeout)
        event["stop_requested"] = True
        remaining = max(.001, kwargs["timeout_seconds"] - (clock() - started))
        measured = inspect(root, reserve_bytes=0, scan_timeout=min(scan_timeout, remaining))
        event["scan"] = measured
        event["paused_seconds"] = clock() - tick
        if (measured.get("complete") is True and measured.get("admissible") is True
                and clock() - started < kwargs["timeout_seconds"]):
            signal_group(child, signal.SIGCONT)
            event["resumed"] = True
        else:
            event["reason"] = "SCAN_REFUSED_OR_TOTAL_DEADLINE_EXHAUSTED"
        return measured

    result = bound.run(command, inspect=checked, spawn=tracked_spawn, **kwargs)
    result["periodic_scan_pauses"] = {
        "schema": "farm-storage-pause-before-periodic-scan-v1",
        "policy": "SIGSTOP_OWN_GROUP_BEFORE_SCAN; SIGCONT_ONLY_AFTER_COMPLETE_ADMISSION",
        "job_restarts": 0, "scan_retries": 0, "pauses": pauses}
    return result


def main(argv=None, *, root=original.ROOT):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("panel-freeze", "panel-commands", "panel-admit", "panel-member",
                            "panel-resume", "panel-close", "panel-release-censored"):
        return panel_main(args, root=root)
    if args[:1] in (["external-check"], ["external-run"]):
        try:
            return external_main(args, root=root)
        except (ValueError, OSError) as error:
            print(json.dumps({"status": "REFUSED", "error": str(error), "hard_quota": False}))
            return 2
        except KeyboardInterrupt as error:
            print(json.dumps({"status": "INTERRUPTED", "error": str(error), "hard_quota": False}))
            return 130
    # The child argv after -- is opaque: never consume a similarly named flag.
    split = args.index("--") if "--" in args else len(args)
    prefix = args[:split]
    bound = bindings()
    modes = {"--retry-scan-timeout": run_with_scan_recheck,
             "--pause-for-storage-check": run_with_paused_scans}
    selected = [option for option in modes if option in prefix]
    if len(selected) > 1:
        raise ValueError("choose one storage scan mode")
    if selected:
        option = selected[0]
        if prefix.count(option) != 1 or not prefix or prefix[0] != "run":
            raise ValueError(option + " requires exactly one storage run option")
        args.remove(option)
        scope = bound.main.__globals__
        scope["run"] = lambda command, **kwargs: modes[option](bound, command, **kwargs)
    return bound.main(args, root=root)


def external_policy(root):
    """Owner-declared corpus only; fail closed if its separate mount is absent."""
    value = bindings().configuration(root).get("external_corpus")
    if not isinstance(value, dict):
        raise ValueError("no external corpus configured")
    mount, corpus = original.absolute(value["mount"]), original.absolute(value["root"])
    if (not os.path.ismount(mount) or corpus == mount or mount not in corpus.parents
            or os.stat(mount).st_dev == os.stat(root).st_dev):
        raise ValueError("corpus requires an available mount on a different filesystem")
    if (corpus.exists() and (not corpus.is_dir()
                            or os.stat(corpus).st_dev != os.stat(mount).st_dev)):
        raise ValueError("corpus directory must reside on its declared mount")
    if (original.natural(value["limit_bytes"]) == 0
            or original.natural(value["minimum_free_bytes"]) == 0
            or value.get("included_in_workspace_budget") is not False
            or value.get("included_in_review_zip") is not False):
        raise ValueError("explicit bounded external corpus policy required")
    return value


def external_status(root=original.ROOT, *, reserve_bytes=0, scan_timeout=60):
    """Separate corpus accounting and its own declared limit."""
    policy = external_policy(root)
    reserve_bytes = original.natural(reserve_bytes)
    measured = original.scan([policy["root"]], timeout_seconds=scan_timeout)
    free = shutil.disk_usage(policy["mount"]).free
    projected = measured["allocated_bytes"] + reserve_bytes
    admissible = (measured["complete"] and projected <= policy["limit_bytes"]
                  and free - reserve_bytes >= policy["minimum_free_bytes"])
    state = ("INCOMPLETE_SCAN" if not measured["complete"] else
             "OVER_BUDGET" if projected > policy["limit_bytes"] else
             "LOW_MOUNT_FREE_SPACE" if free - reserve_bytes < policy["minimum_free_bytes"] else
             "WITHIN_BUDGET")
    return {"schema": "farm-external-corpus-storage-status-v1", "policy": policy,
            **measured, "status": state, "admissible": admissible,
            "projected_bytes": projected, "requested_reserve_bytes": reserve_bytes,
            "mount_free_bytes": free, "hard_quota": False, "automatic_deletion": False,
            "reservation_acquired": False, "included_in_workspace_budget": False}


def external_main(args, *, root=original.ROOT):
    parser = argparse.ArgumentParser(description="Separate-mount public replay/SFT corpus; workspace cap unchanged")
    parser.add_argument("mode", choices=("external-check", "external-run"))
    parser.add_argument("--reserve-bytes", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600)
    # Split first so the child's flags remain opaque to this parser.
    split = args.index("--") if "--" in args else len(args)
    opts = parser.parse_args(args[:split])
    command = args[split + 1:] if split < len(args) else []
    if opts.mode == "external-check":
        if command:
            parser.error("external-check does not execute a command")
        result = external_status(root, reserve_bytes=opts.reserve_bytes)
        code = 0 if result["admissible"] else 2
    else:
        corpus = external_policy(root)["root"]
        def inspect(_corpus, **kwargs):
            return external_status(root, **kwargs)
        result = original.run(command, root=corpus, reserve_bytes=opts.reserve_bytes,
                              timeout_seconds=opts.timeout, inspect=inspect)
        code = result["exit_code"]
    print(json.dumps(result, indent=2))
    return code


# Fixed panel admission. All locks and journals live on the SAME execution root.
# No scheduler is implemented here. SLURM starts one local supervisor per member;
# the exact frozen original still owns every member's TERM/CONT/KILL/wait loop.
PANEL_SCHEMA = 'farm-storage-panel-v1'
PANEL_ROOT = Path('/mnt/ai/scratch/farmNEAT/source')
PANEL_NODES = {'gpu-4090b': 'gpu', 'gpu-skynet': 'gpu', 'gpu-spark': 'spark'}
PANEL_CANONICAL_RESERVE = 5_000_000_000
PANEL_SOURCES = ('RUN.py', 'src/farm_neat/training/storage.py',
                 'src/farm_neat/training/storage_layout.py',
                 'src/farm_neat/training/project_paths.py',
                 'project/PROJECT-STORAGE.json', 'project/runtime/interpreter.toml')
PANEL_RECORD_LIMIT = 1_048_576


class _PanelDecision(ValueError):
    def __init__(self, status):
        super().__init__(status)
        self.status = status


def _panel_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _panel_read(path):
    """Small, no-symlink, duplicate-key-free metadata; never load workload payloads."""
    path = original.absolute(path)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate panel metadata key: ' + key)
            result[key] = value
        return result
    import stat
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('panel metadata must be a regular file')
        body = stream.read(PANEL_RECORD_LIMIT + 1)
    if len(body) > PANEL_RECORD_LIMIT:
        raise ValueError('panel metadata size limit exceeded')
    value = json.loads(body, object_pairs_hook=pairs,
                       parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite panel metadata')))
    if not isinstance(value, dict):
        raise ValueError('panel metadata must be an object')
    return value, hashlib.sha256(body).hexdigest()


def _panel_mkdir(path, *, exist_ok=True):
    """Persist every newly created directory link, not merely its file contents."""
    path = original.absolute(path)
    if not path.parent.exists():
        _panel_mkdir(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if not exist_ok or not path.is_dir():
            raise
    fd = os.open(path.parent, original.FLAGS)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _panel_new(path, value):
    """Exclusive durable record. Torn writes remain censored, never overwritten."""
    path = original.absolute(path)
    body = _panel_bytes(value)
    if len(body) > PANEL_RECORD_LIMIT:
        raise ValueError('panel receipt size limit exceeded')
    with path.open('xb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, original.FLAGS)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(body).hexdigest()


def _panel_rel(value):
    if (type(value) is not str or not value or '\\' in value
            or Path(value).is_absolute() or any(p in ('', '.', '..') for p in value.split('/'))):
        raise ValueError('canonical root-relative panel path required')
    return value


def _panel_sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('lowercase SHA256 required')
    return value


def _panel_pins(rows):
    if type(rows) is not list or len(rows) > 4096:
        raise ValueError('bounded explicit panel input pins required')
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != {'path', 'sha256'}:
            raise ValueError('pin requires exactly path and sha256')
        name = _panel_rel(row['path'])
        _panel_sha(row['sha256'])
        if name in seen:
            raise ValueError('duplicate panel input pin')
        seen.add(name)
    return seen


def _panel_manifest(path, digest, root):
    root = original.absolute(root)
    value, actual = _panel_read(path)
    if actual != _panel_sha(digest):
        raise ValueError('panel manifest SHA256 mismatch')
    return _panel_validate(value, root)


def _panel_validate(value, root):
    required = {'schema', 'execution_root', 'panel_id', 'deadline_unix', 'reserve_bytes',
                'canonical_reserve_bytes', 'interval_seconds', 'scan_timeout', 'source_pins', 'members'}
    if set(value) != required or value['schema'] != PANEL_SCHEMA:
        raise ValueError('unsupported panel schema/fields')
    if value['execution_root'] != str(root):
        raise ValueError('panel requires the exact shared execution root')
    tag = value['panel_id']
    if (type(tag) is not str or not 1 <= len(tag) <= 64
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in tag)):
        raise ValueError('bounded literal panel_id required')
    if original.natural(value['canonical_reserve_bytes']) != PANEL_CANONICAL_RESERVE:
        raise ValueError('panel canonical reserve field changed')
    import math
    for key in ('deadline_unix', 'interval_seconds', 'scan_timeout'):
        number = value[key]
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError('finite positive panel deadline/interval required')
    if value['interval_seconds'] > 60 or value['scan_timeout'] > 60:
        raise ValueError('panel interval and scan timeout must be at most60 seconds')
    sources = _panel_pins(value['source_pins'])
    if not set(PANEL_SOURCES) <= sources:
        raise ValueError('panel must pin RUN.py, guard, adapter, paths, budget and runtime')
    pins = {p['path']: p['sha256'] for p in value['source_pins']}
    if pins['src/farm_neat/training/storage.py'] != ORIGINAL_SHA256:
        raise ValueError('panel requires the exact original storage guard')
    members = value['members']
    if type(members) is not list or len(members) != 3:
        raise ValueError('panel admits exactly three distinct members')
    seen, outputs, total = set(), [], 0
    for member in members:
        if type(member) is not dict or set(member) != {'id', 'node', 'partition', 'required_machine',
                'timeout_seconds', 'peak_growth_bytes', 'output', 'argv', 'inputs'}:
            raise ValueError('unsupported panel member fields')
        node = member['node']
        if (type(node) is not str or node not in PANEL_NODES or member['id'] != node
                or member['partition'] != PANEL_NODES[node] or node in seen):
            raise ValueError('duplicate, renamed or non-panel member refused')
        seen.add(node)
        if member['required_machine'] not in ('x86_64', 'aarch64'):
            raise ValueError('explicit member runtime machine required')
        seconds = member['timeout_seconds']
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 86400
                or seconds / value['interval_seconds'] > 10000):
            raise ValueError('bounded member timeout/scan count required')
        growth = original.natural(member['peak_growth_bytes'])
        if growth == 0:
            raise ValueError('positive per-member peak growth required')
        total += growth
        output = _panel_rel(member['output'])
        if not output.startswith('work/runs/') or output == 'work/runs':
            raise ValueError('member output must be below work/runs')
        target = original.absolute(root / output)
        for other in outputs:
            if target == other or target in other.parents or other in target.parents:
                raise ValueError('overlapping panel member outputs refused')
        outputs.append(target)
        args = member['argv']
        if (type(args) is not list or not args or any(type(a) is not str or not a or '\0' in a for a in args)
                or args.count('{output}') != 1):
            raise ValueError('literal argv requires exactly one separate {output} argument')
        inputs = _panel_pins(member['inputs']) | sources
        for name in inputs:
            path = original.absolute(root / name)
            if path == target or target in path.parents:
                raise ValueError('panel outputs may not contain frozen inputs/source')
    all_inputs = sources | {row['path'] for m in members for row in m['inputs']}
    for output in outputs:
        if any((root / name) == output or output in (root / name).parents for name in all_inputs):
            raise ValueError('panel outputs may not contain any members frozen inputs/source')
    if seen != set(PANEL_NODES) or original.natural(value['reserve_bytes']) < total:
        raise ValueError('aggregate reservation must cover all three member growth estimates')
    if (bindings().configuration(root)['schema'] == original.SCHEMA
            and value['reserve_bytes'] + PANEL_CANONICAL_RESERVE > original.LIMIT):
        raise ValueError('panel exceeds aggregate decimal60GB cap')
    return value


def _panel_verify_pins(root, rows, *, deadline=None):
    """Hash explicitly declared regular inputs incrementally; source freezes are cooperative."""
    import stat
    for row in rows:
        path = original.absolute(Path(root) / row['path'])
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError('panel pin must name a regular file')
            digest = hashlib.sha256()
            while True:
                if deadline is not None and time.time() >= deadline:
                    raise ValueError('panel deadline expired while verifying pins')
                data = os.read(fd, 1_048_576)
                if not data:
                    break
                digest.update(data)
            if digest.hexdigest() != row['sha256']:
                raise ValueError('panel input/source pin mismatch: ' + row['path'])
        finally:
            os.close(fd)


def _panel_home(root, digest):
    return original.absolute(Path(root) / 'work/cache/storage/panels' / _panel_sha(digest))


def _panel_lock(root):
    """Use the original shared COMMAND.lock, not a node-local or second-root lock."""
    from contextlib import contextmanager
    @contextmanager
    def locked():
        controls = original.absolute(Path(root) / 'work/cache/storage')
        _panel_mkdir(controls)
        fd = os.open(original.absolute(controls / 'COMMAND.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            original.fcntl.flock(fd, original.fcntl.LOCK_EX | original.fcntl.LOCK_NB)
            yield
        finally:
            os.close(fd)
    return locked()


def _panel_completion(root, manifest, digest, member):
    output = original.absolute(Path(root) / member['output'])
    start, start_sha = _panel_read(output / 'STARTED.json')
    done, done_sha = _panel_read(output / 'COMPLETED.json')
    for row in (start, done):
        if row.get('panel_sha256') != digest or row.get('member') != member['id']:
            raise ValueError('panel member receipt identity mismatch')
    if done.get('started_sha256') != start_sha:
        raise ValueError('panel completion does not bind its exact start')
    return done, done_sha


def _panel_not_started(root, digest, member, abort_sha, *, create=False):
    """Bind a peer-aborted cell that never acquired its STARTED intent."""
    output = original.absolute(Path(root) / member['output'])
    if ((output / 'STARTED.json').exists() or (output / 'COMPLETED.json').exists()
            or (output / 'payload').exists() or (output / 'stdout.log').exists()
            or (output / 'stderr.log').exists()):
        raise ValueError('not-started cell has member execution evidence')
    path = output / 'NOT_STARTED.json'
    expected = dict(schema='farm-storage-panel-not-started-v1',
                    status='NOT_STARTED_PEER_ABORT', panel_sha256=digest,
                    abort_sha256=abort_sha, member=member['id'], node=member['node'],
                    command_started=False, cleanup_confirmed=True, new_starts=0)
    if create and not path.exists():
        receipt_sha = _panel_new(path, expected)
        return expected, receipt_sha
    row, receipt_sha = _panel_read(path)
    if row != expected:
        raise ValueError('not-started panel receipt identity mismatch')
    return row, receipt_sha


def _panel_released(root, home):
    path = original.absolute(home / 'RELEASED.json')
    if not path.exists():
        return False
    released, _ = _panel_read(path)
    admission, admission_sha = _panel_read(home / 'ADMISSION.json')
    manifest = _panel_manifest(home / 'MANIFEST.json', home.name, root)
    status = released.get('status')
    abort_path = home / 'ABORT.json'
    if (released.get('schema') != 'farm-storage-panel-release-v1'
            or released.get('panel_sha256') != home.name or released.get('admission_sha256') != admission_sha
            or status not in ('RELEASED', 'RELEASED_CENSORED')
            or admission.get('panel_sha256') != home.name):
        raise ValueError('invalid/censored panel release')
    if status == 'RELEASED' and abort_path.exists():
        raise ValueError('invalid/censored panel release')
    abort_sha = None
    if status == 'RELEASED_CENSORED':
        if not abort_path.exists() or released.get('automatic_replay') is not False:
            raise ValueError('invalid censored panel release')
        _abort, abort_sha = _panel_read(abort_path)
        if released.get('abort_sha256') != abort_sha:
            raise ValueError('censored panel release abort changed')
    hashes, not_started = {}, {}
    censored = 0
    for member in manifest['members']:
        output = original.absolute(Path(root) / member['output'])
        if not (output / 'STARTED.json').exists():
            if status != 'RELEASED_CENSORED':
                raise ValueError('successful release requires every member completion')
            _row, terminal_sha = _panel_not_started(root, home.name, member, abort_sha)
            not_started[member['id']] = terminal_sha
            continue
        done, digest = _panel_completion(root, manifest, home.name, member)
        allowed = (('COMPLETED', 'BLOCKED_CELL') if status == 'RELEASED'
                   else ('COMPLETED', 'BLOCKED_CELL', 'CENSORED'))
        if done.get('status') not in allowed or done.get('cleanup_confirmed') is not True:
            raise ValueError('failed member cannot release a panel reservation')
        censored += done.get('status') == 'CENSORED'
        hashes[member['id']] = digest
    if status == 'RELEASED_CENSORED' and not censored:
        raise ValueError('censored release requires a censored member')
    if released.get('completed_sha256') != hashes:
        raise ValueError('panel release completion hashes changed')
    if released.get('not_started_sha256', {}) != not_started:
        raise ValueError('panel release not-started hashes changed')
    return True


def _panel_pending(root):
    base = original.absolute(Path(root) / 'work/cache/storage/panels')
    if not base.exists():
        return []
    homes = list(base.iterdir())
    if len(homes) > 1024:
        raise ValueError('panel journal needs maintainer retention review')
    pending = []
    for home in sorted(homes):
        original.absolute(home)
        _panel_sha(home.name)
        if not home.is_dir():
            raise ValueError('invalid panel journal entry')
        if not _panel_released(root, home):
            pending.append(home.name)
    return pending


def _panel_measure(manifest, root, inspect, *, growth=0, scan_timeout=None):
    # The legacy capped protocol reserved canonical headroom. The current
    # free-space policy needs only the declared growth estimate.
    policy = bindings().configuration(root)
    limit = policy['limit_bytes']
    padding = (max(0, limit - (original.LIMIT - PANEL_CANONICAL_RESERVE))
               if policy['schema'] == original.SCHEMA else 0)
    return inspect(root, reserve_bytes=padding + growth,
                   scan_timeout=manifest['scan_timeout'] if scan_timeout is None else scan_timeout)


def _panel_output_measure(manifest, root, admission, *, scan_timeout):
    """Monitor only the three declared outputs under an admitted aggregate reserve.

    Admission has already measured the complete project and reserved every member's
    declared peak growth. Repeating that whole-tree scan independently on all three
    nodes overloads the shared filesystem and can censor an otherwise in-budget
    payload. Live checks therefore measure the only paths these cooperative jobs may
    write. Explicit close still performs one fresh complete-project scan.
    """
    outputs = [original.absolute(Path(root) / member['output']) for member in manifest['members']]
    measured = original.scan(outputs, timeout_seconds=scan_timeout)
    admission_scan = admission['admission_scan']
    policy = admission_scan['policy']
    if policy['schema'] == original.FREE_SCHEMA:
        physical = bindings().status(root, reserve_bytes=0)
        observed = original.natural(measured.get('allocated_bytes', 0))
        reserve = original.natural(manifest['reserve_bytes'])
        complete = measured.get('complete') is True and physical['complete'] is True
        within_reserve = observed <= reserve
        admissible = complete and within_reserve and physical['admissible'] is True
        status = ('INCOMPLETE_CHECK' if not complete else
                  'OUTPUT_RESERVE_EXCEEDED' if not within_reserve else
                  physical['status'])
        return {**measured, 'schema': 'farm-storage-panel-output-status-v2',
                'root': str(root), 'policy': policy,
                'observed_output_allocated_bytes': observed,
                'declared_output_reserve_bytes': reserve,
                'physical_free_space': physical['filesystems'],
                'requested_reserve_bytes': 0, 'status': status,
                'admissible': admissible, 'complete': complete,
                'reservation_acquired': False, 'hard_quota': False,
                'automatic_deletion': False, 'monitor_scope': 'DECLARED_PANEL_OUTPUTS',
                'limits': ('Only declared panel outputs are scanned for growth. '
                           'Physical free space is checked on every configured filesystem.')}
    limit = original.natural(policy['limit_bytes'])
    base = original.natural(admission_scan['allocated_bytes'])
    observed = original.natural(measured.get('allocated_bytes', 0))
    reserve = original.natural(manifest['reserve_bytes'])
    padding = max(0, limit - (original.LIMIT - PANEL_CANONICAL_RESERVE))
    projected = base + observed + padding
    complete = measured.get('complete') is True
    within_reserve = observed <= reserve
    within_budget = projected <= limit
    status = ('INCOMPLETE_SCAN' if not complete else
              'OUTPUT_RESERVE_EXCEEDED' if not within_reserve else
              'OVER_BUDGET' if not within_budget else 'WITHIN_BUDGET')
    return {**measured, 'schema': 'farm-storage-panel-output-status-v1',
            'root': str(root), 'policy': policy, 'allocated_bytes': observed,
            'observed_output_allocated_bytes': observed,
            'admission_allocated_bytes': base, 'declared_output_reserve_bytes': reserve,
            'projected_bytes': projected, 'remaining_bytes': limit - projected,
            'requested_reserve_bytes': 0, 'status': status,
            'admissible': complete and within_reserve and within_budget,
            'reservation_acquired': False, 'hard_quota': False,
            'automatic_deletion': False, 'monitor_scope': 'DECLARED_PANEL_OUTPUTS',
            'limits': ('Admission and explicit closure scan the complete project. Live monitoring '
                       'covers only source-pinned cooperative member outputs; off-path writers are '
                       'prohibited and are detected only by closure.')}


def panel_admit(path, digest, *, root=original.ROOT, inspect=None):
    """One aggregate reservation. This never starts a member or reads workload inputs."""
    root = original.absolute(root)
    try:
        with _panel_lock(root):
            if _panel_pending(root):
                return dict(status='PANEL_RESERVATION_HELD', command_started=False, exit_code=2)
            manifest = _panel_manifest(path, digest, root)
            home = _panel_home(root, digest)
            if home.exists():
                return dict(status='STALE_PANEL_IDENTITY', command_started=False, exit_code=2)
            if time.time() >= manifest['deadline_unix']:
                raise ValueError('stale panel deadline')
            measured = _panel_measure(manifest, root, inspect or bindings().status, growth=manifest['reserve_bytes'])
            if measured.get('admissible') is not True or measured.get('complete') is not True:
                return dict(status='ADMISSION_REFUSED', command_started=False, exit_code=2, scan=measured)
            _panel_verify_pins(root, manifest['source_pins'], deadline=manifest['deadline_unix'])
            for member in manifest['members']:
                if original.absolute(root / member['output']).exists():
                    raise ValueError('member output already exists; never overwrite/reuse it')
            # An interrupted publication leaves a durable unresolved directory.
            _panel_mkdir(home, exist_ok=False)
            # Copy exact supplied bytes; the manifest SHA is not a serialization hash.
            data = original.absolute(path).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('panel manifest changed during admission')
            with (home / 'MANIFEST.json').open('xb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            for member in manifest['members']:
                _panel_mkdir(root / member['output'], exist_ok=False)
            row = dict(schema='farm-storage-panel-admission-v1', status='ADMITTED', panel_sha256=digest,
                       execution_root=str(root), reserve_bytes=manifest['reserve_bytes'],
                       canonical_reserve_bytes=PANEL_CANONICAL_RESERVE, admission_scan=measured,
                       created_unix=time.time(), command_started=False, hard_quota=False,
                       automatic_reservation_reuse=False, exit_code=0)
            _panel_new(home / 'ADMISSION.json', row)
            return row
    except BlockingIOError:
        return dict(status='COOPERATIVE_LAUNCHER_BUSY', command_started=False, exit_code=2)


def _panel_node_context():
    import platform
    import socket
    node = os.environ.get('SLURMD_NODENAME')
    if not node:
        raise ValueError('member must run on its exact SLURM-assigned node')
    return dict(node=node, hostname=socket.gethostname().split('.')[0], machine=platform.machine(),
                partition=os.environ.get('SLURM_JOB_PARTITION'),
                job_id=os.environ.get('SLURM_JOB_ID'), job_name=os.environ.get('SLURM_JOB_NAME', ''),
                restart_count=int(os.environ.get('SLURM_RESTART_COUNT', '0')))


def _panel_abort(home, digest, reason):
    try:
        _panel_new(home / 'ABORT.json', dict(schema='farm-storage-panel-abort-v1',
                   panel_sha256=digest, reason=str(reason)[:4096], reservation_held=True))
    except FileExistsError:
        pass  # First adverse witness wins. Never replace it with a later success.


def panel_member(path, digest, member_id, *, root=original.ROOT, identity=None,
                 inspect=None, spawn=None, terminate=None):
    """Source-bound delegation of ONE reservation to exactly three local supervisors.

    Only the original guard's flock seam is adapted: hold the canonical mutex for
    durable membership enrollment, then release it. Its deadline/scan/process-group
    loop is reused verbatim. STARTED is an at-most-once intent, not proof of spawn.
    Missing/failed completions retain the aggregate reservation after process death.
    """
    root = original.absolute(root)
    bound = bindings()
    manifest = _panel_manifest(path, digest, root)
    selected = [m for m in manifest['members'] if m['id'] == member_id]
    if not selected:
        raise ValueError('non-panel member refused')
    member = selected[0]
    home = _panel_home(root, digest)
    output = original.absolute(root / member['output'])
    command = [str(output / 'payload') if arg == '{output}' else arg for arg in member['argv']]
    state = dict(claimed=False, spawn_attempted=False, spawned=False, cleaned=False)
    logs = []
    injected_inspect = inspect
    spawn = spawn or subprocess.Popen
    terminate = terminate or original._terminate

    def enroll(fd, flags):
        # Concurrent members may arrive at the short metadata critical section
        # together. Bounded mutex waiting happens BEFORE intent/input/spawn, not
        # by retrying a started cell. An unrelated ordinary owner still refuses.
        until = time.monotonic() + min(manifest['scan_timeout'], member['timeout_seconds'])
        while True:
            try:
                original.fcntl.flock(fd, flags)
                break
            except BlockingIOError:
                if (not (home / 'ADMISSION.json').exists() or (home / 'ABORT.json').exists()
                        or time.monotonic() >= until or time.time() >= manifest['deadline_unix']):
                    raise
                time.sleep(min(.01, max(0., until-time.monotonic())))
        try:
            if _panel_pending(root) != [digest] or (home / 'ABORT.json').exists():
                raise _PanelDecision('PANEL_RESERVATION_HELD_OR_CENSORED')
            admission, admission_sha = _panel_read(home / 'ADMISSION.json')
            if admission.get('panel_sha256') != digest or admission.get('execution_root') != str(root):
                raise ValueError('panel admission identity mismatch')
            _panel_manifest(home / 'MANIFEST.json', digest, root)
            if (output / 'STARTED.json').exists():
                raise _PanelDecision('ALREADY_STARTED')
            if time.time() >= manifest['deadline_unix']:
                raise _PanelDecision('STALE_PANEL_IDENTITY')
            who = _panel_node_context() if identity is None else identity
            if (who.get('node') != member['node'] or who.get('partition') != member['partition']
                    or type(who.get('job_id')) is not str or not who['job_id'].isdigit() or int(who['job_id']) <= 0
                    or not who.get('job_name', '').startswith('farmneat-') or who.get('restart_count') != 0):
                raise ValueError('wrong member node/job/partition or requeued job refused')
            for other in manifest['members']:
                prior = original.absolute(root / other['output'] / 'STARTED.json')
                if prior.exists() and _panel_read(prior)[0].get('identity', {}).get('job_id') == who['job_id']:
                    raise ValueError('a panel job identity may claim only one member')
            _panel_verify_pins(root, manifest['source_pins'], deadline=manifest['deadline_unix'])
            start = dict(schema='farm-storage-panel-start-v1', status='STARTED', panel_sha256=digest,
                         admission_sha256=admission_sha, member=member_id, node=member['node'], identity=who,
                         argv=command, output=member['output'], created_unix=time.time(),
                         meaning='DURABLE_AT_MOST_ONCE_INTENT_BEFORE_INPUT_READ_OR_SPAWN')
            state.update(started_sha256=_panel_new(output / 'STARTED.json', start),
                         admission_sha256=admission_sha, claimed=True)
            if who.get('machine') != member['required_machine']:
                raise _PanelDecision('BLOCKED_CELL')
        finally:
            original.fcntl.flock(fd, original.fcntl.LOCK_UN)

    def measure(root, *, reserve_bytes, scan_timeout):
        # The original runner requires a nonzero declared reserve, but members do
        # NOT acquire another one. Only measured bytes + canonical5GB are tested.
        if (home / 'ABORT.json').exists():
            raise ValueError('panel peer failed; aggregate reservation remains censored')
        if _panel_read(home / 'ADMISSION.json')[1] != state['admission_sha256']:
            raise ValueError('panel admission changed during member supervision')
        if time.time() >= manifest['deadline_unix']:
            raise ValueError('panel deadline expired')
        timeout = min(scan_timeout, max(.001, manifest['deadline_unix'] - time.time()))
        if injected_inspect is not None:
            return _panel_measure(manifest, root, injected_inspect, scan_timeout=timeout)
        admission, _ = _panel_read(home / 'ADMISSION.json')
        return _panel_output_measure(manifest, root, admission, scan_timeout=timeout)

    def start(command, **kwargs):
        _panel_verify_pins(root, member['inputs'], deadline=manifest['deadline_unix'])
        if (home / 'ABORT.json').exists():
            raise ValueError('panel peer failed before spawn')
        for name in ('stdout.log', 'stderr.log'):
            logs.append(original.absolute(output / name).open('xb'))
        state['spawn_attempted'] = True
        process = spawn(command, **kwargs, stdout=logs[0], stderr=logs[1])
        state['spawned'] = True
        return process

    def cleanup(child):
        terminate(child)
        state['cleaned'] = True

    # Source-pinned adaptation, not a second implementation of the guard loop.
    scope = dict(bound.raw_run.__globals__)
    scope['fcntl'] = SimpleNamespace(flock=enroll, LOCK_EX=original.fcntl.LOCK_EX, LOCK_NB=original.fcntl.LOCK_NB)
    runner = FunctionType(bound.raw_run.__code__, scope, 'panel_local_supervisor', bound.raw_run.__defaults__)
    runner.__kwdefaults__ = dict(bound.raw_run.__kwdefaults__ or {})
    detail, error, status = {}, None, 'CENSORED'
    try:
        detail = runner(command, root=root, reserve_bytes=manifest['reserve_bytes'],
                        timeout_seconds=max(.001, min(member['timeout_seconds'], manifest['deadline_unix']-time.time())),
                        interval_seconds=manifest['interval_seconds'], scan_timeout=manifest['scan_timeout'],
                        inspect=measure, spawn=start, terminate=cleanup)
        if injected_inspect is None:
            detail['monitor_scope'] = 'DECLARED_PANEL_OUTPUTS'
        if state['claimed']:
            _panel_verify_pins(root, manifest['source_pins'], deadline=manifest['deadline_unix'])
        status = ('COMPLETED' if detail.get('status') == 'COMMAND_EXITED' and detail.get('exit_code') == 0
                  else 'CENSORED' if state['claimed'] else detail['status'])
    except _PanelDecision as exc:
        status = exc.status
    except (Exception, KeyboardInterrupt) as exc:
        if not state['claimed']:
            raise
        error = f'{type(exc).__name__}: {exc}'
        status = 'CENSORED'
    finally:
        for stream in logs:
            stream.close()
    result = dict(schema='farm-storage-panel-completion-v1', panel_sha256=digest, member=member_id,
                  node=member['node'], status=status, command_started=state['spawned'],
                  cleanup_confirmed=state['cleaned'] or not state['spawn_attempted'],
                  new_starts=int(state['claimed']), hard_quota=False, automatic_reservation_reuse=False,
                  supervision=detail, error=error, exit_code=0 if status in ('COMPLETED', 'BLOCKED_CELL', 'ALREADY_STARTED') else 2)
    if state['claimed']:
        result['started_sha256'] = state['started_sha256']
        if status not in ('COMPLETED', 'BLOCKED_CELL') or result['cleanup_confirmed'] is not True:
            _panel_abort(home, digest, error or detail.get('status', status))
        _panel_new(output / 'COMPLETED.json', result)
    return result


def panel_resume(path, digest, *, root=original.ROOT):
    """Read-only resume inventory. Never starts even a previously unstarted member."""
    root = original.absolute(root)
    manifest = _panel_manifest(path, digest, root)
    home = _panel_home(root, digest)
    rows = []
    for member in manifest['members']:
        output = original.absolute(root / member['output'])
        row = dict(member=member['id'], node=member['node'], status='NOT_STARTED')
        if (output / 'STARTED.json').exists():
            row['status'] = 'CENSORED_NO_COMPLETION'
            if (output / 'COMPLETED.json').exists():
                try:
                    row['status'] = _panel_completion(root, manifest, digest, member)[0]['status']
                except (ValueError, OSError):
                    row['status'] = 'CENSORED_INVALID_RECEIPT'
        rows.append(row)
    return dict(schema='farm-storage-panel-resume-v1', panel_sha256=digest, members=rows,
                new_starts=0, reservation_held=home.exists() and not _panel_released(root, home),
                automatic_reservation_reuse=False, exit_code=0)


def panel_close(path, digest, *, root=original.ROOT, inspect=None):
    """Explicit close only: all three safe terminal receipts AND a fresh complete scan."""
    root = original.absolute(root)
    try:
        with _panel_lock(root):
            manifest = _panel_manifest(path, digest, root)
            home = _panel_home(root, digest)
            if _panel_released(root, home):
                return dict(status='ALREADY_RELEASED', new_starts=0, exit_code=0)
            if _panel_pending(root) != [digest] or (home / 'ABORT.json').exists():
                raise ValueError('censored or unrelated panel; reservation remains held')
            hashes = {}
            for member in manifest['members']:
                done, done_sha = _panel_completion(root, manifest, digest, member)
                if done.get('status') not in ('COMPLETED', 'BLOCKED_CELL') or done.get('cleanup_confirmed') is not True:
                    raise ValueError('incomplete/failed member; reservation remains held')
                hashes[member['id']] = done_sha
            measured = _panel_measure(manifest, root, inspect or bindings().status)
            if measured.get('complete') is not True or measured.get('admissible') is not True:
                _panel_abort(home, digest, 'FINAL_SCAN_REFUSED')
                raise ValueError('incomplete/over-budget final scan; reservation remains held')
            row = dict(schema='farm-storage-panel-release-v1', status='RELEASED', panel_sha256=digest,
                       admission_sha256=_panel_read(home / 'ADMISSION.json')[1], completed_sha256=hashes,
                       final_scan=measured, new_starts=0, exit_code=0)
            _panel_new(home / 'RELEASED.json', row)
            return row
    except BlockingIOError:
        return dict(status='COOPERATIVE_LAUNCHER_BUSY', new_starts=0, exit_code=2)


def panel_release_censored(path, digest, *, root=original.ROOT, inspect=None):
    """Release only fully terminal, cleanup-confirmed failed panels; never replay them."""
    root = original.absolute(root)
    try:
        with _panel_lock(root):
            manifest = _panel_manifest(path, digest, root)
            home = _panel_home(root, digest)
            if _panel_released(root, home):
                return dict(status='ALREADY_RELEASED', new_starts=0, exit_code=0)
            if _panel_pending(root) != [digest] or not (home / 'ABORT.json').exists():
                raise ValueError('censored panel with one held reservation required')
            _abort, abort_sha = _panel_read(home / 'ABORT.json')
            hashes, not_started, censored = {}, {}, 0
            for member in manifest['members']:
                output = original.absolute(root / member['output'])
                if not (output / 'STARTED.json').exists():
                    _row, terminal_sha = _panel_not_started(
                        root, digest, member, abort_sha, create=True)
                    not_started[member['id']] = terminal_sha
                    continue
                done, done_sha = _panel_completion(root, manifest, digest, member)
                if done.get('status') not in ('COMPLETED', 'BLOCKED_CELL', 'CENSORED'):
                    raise ValueError('all panel members need terminal receipts')
                if done.get('cleanup_confirmed') is not True:
                    raise ValueError('all panel members need confirmed cleanup')
                censored += done.get('status') == 'CENSORED'
                hashes[member['id']] = done_sha
            if not censored:
                raise ValueError('censored release requires a censored member')
            measured = _panel_measure(manifest, root, inspect or bindings().status)
            if measured.get('complete') is not True or measured.get('admissible') is not True:
                raise ValueError('incomplete/over-budget censored-release scan')
            _admission, admission_sha = _panel_read(home / 'ADMISSION.json')
            row = dict(schema='farm-storage-panel-release-v1', status='RELEASED_CENSORED',
                       panel_sha256=digest, admission_sha256=admission_sha,
                       abort_sha256=abort_sha, completed_sha256=hashes,
                       not_started_sha256=not_started,
                       final_scan=measured, automatic_replay=False, new_starts=0, exit_code=0)
            _panel_new(home / 'RELEASED.json', row)
            return row
    except BlockingIOError:
        return dict(status='COOPERATIVE_LAUNCHER_BUSY', new_starts=0, exit_code=2)


def panel_freeze(request_path, output_path, *, root=original.ROOT, valid_seconds=3600):
    """Offline, standard-library-only metadata freeze; never load weights or submit jobs.

    The request supplies exact producer-owned argv/input hashes. The freezer adds
    source hashes, fixed execution identity, one reserve and a finite expiry. Copy
    these exact bytes and sources to the replica; do not re-freeze active work.
    """
    root = original.absolute(root)
    import math
    if (type(valid_seconds) not in (int, float) or not math.isfinite(valid_seconds)
            or not 0 < valid_seconds <= 86400):
        raise ValueError('freeze lifetime must be finite and within one day')
    with _panel_lock(root):
        if _panel_pending(root):
            raise ValueError('PANEL_RESERVATION_HELD: do not refresh an active source freeze')
        request, _ = _panel_read(request_path)
        if (not {'panel_id', 'members'} <= set(request) or set(request) -
                {'panel_id', 'members', 'reserve_bytes', 'interval_seconds', 'scan_timeout', 'source_paths'}):
            raise ValueError('request requires panel_id/members and only documented optional fields')
        extra = request.get('source_paths', [])
        if type(extra) is not list or any(type(x) is not str for x in extra):
            raise ValueError('source_paths must be an explicit list of root-relative source files')
        names = set(PANEL_SOURCES) | {_panel_rel(x) for x in extra}
        # Pin the code actually staged, not only the six control entrypoints.
        names.update(x.relative_to(root).as_posix() for x in (root / 'src').rglob('*')
                     if x.suffix in ('.py', '.json') and x.is_file())
        if len(names) > 4096:
            raise ValueError('bounded explicit panel source pins required')
        pins = []
        for name in sorted(names):
            path = original.absolute(root / name)
            if not path.is_file() or path.stat().st_size > 16_777_216:
                raise ValueError('source freeze requires regular source files, not workload tensors')
            pins.append(dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        members = request['members']
        if type(members) is not list or any(type(m) is not dict for m in members):
            raise ValueError('explicit member objects required')
        try:
            reserve = sum(original.natural(m['peak_growth_bytes']) for m in members) + 1_048_576
        except KeyError as exc:
            raise ValueError('member peak_growth_bytes required') from exc
        manifest = dict(schema=PANEL_SCHEMA, execution_root=str(PANEL_ROOT), panel_id=request['panel_id'],
                        deadline_unix=time.time() + valid_seconds,
                        reserve_bytes=request.get('reserve_bytes', reserve),
                        canonical_reserve_bytes=PANEL_CANONICAL_RESERVE,
                        interval_seconds=request.get('interval_seconds', 10),
                        scan_timeout=request.get('scan_timeout', 60), source_pins=pins, members=members)
        _panel_validate(manifest, PANEL_ROOT)
        _panel_mkdir(original.absolute(output_path).parent)
        digest = _panel_new(output_path, manifest)
        return dict(status='FROZEN', panel_sha256=digest, manifest=str(output_path),
                    execution_root=str(PANEL_ROOT), command_started=False, new_starts=0, exit_code=0)


def panel_commands(path, digest, *, staged_manifest=None):
    """Render exact argv only. This function never invokes sbatch, ssh or srun.

    Outside the replica, callers must state the staged root-relative manifest path.
    A relative input already identifies that path; never guess from a basename.
    """
    import math
    import shlex
    manifest = _panel_manifest(path, digest, PANEL_ROOT)
    if staged_manifest is None:
        input_path = Path(path)
        if input_path.is_absolute():
            try:
                staged_manifest = input_path.relative_to(PANEL_ROOT).as_posix()
            except ValueError as exc:
                raise ValueError('offline command rendering requires --staged-manifest') from exc
        else:
            staged_manifest = input_path.as_posix()
    staged_manifest = _panel_rel(staged_manifest)
    def front(operation, *extra):
        return ['python3', '-B', 'RUN.py', 'storage', operation, '--manifest', staged_manifest,
                '--sha256', digest, *extra]
    members = []
    for member in manifest['members']:
        minutes = math.ceil((member['timeout_seconds'] + 2*manifest['scan_timeout'] + 30) / 60)
        argv = ['sbatch', '--parsable', '--no-requeue', '--partition=' + member['partition'],
                '--nodelist=' + member['node'], '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--mem=32G',
                '--time=' + str(minutes), '--job-name=farmneat-panel-' + digest[:12] + '-' + member['id'],
                '--chdir=' + str(PANEL_ROOT), '--output=' + str(PANEL_ROOT / member['output'] / 'slurm-%j.log'),
                '--gres=gpu:1' if member['node']=='gpu-spark' else '--gres=gpu:4090:1',
                '--wrap', shlex.join(front('panel-member', '--member', member['id']))]
        members.append(dict(member=member['id'], sbatch_argv=argv, shell=shlex.join(argv)))
    return dict(schema='farm-storage-panel-commands-v1', panel_sha256=digest, cwd=str(PANEL_ROOT),
                admit_argv=front('panel-admit'), members=members, resume_argv=front('panel-resume'),
                close_argv=front('panel-close'), new_starts=0, command_started=False, exit_code=0,
                precondition='Canonical aggregate <=60GB; canonical files <=5GB; exact pinned replica frozen')


def panel_main(args, *, root=original.ROOT):
    """RUN.py storage owns preparation and launch. Launch has no root/auth override."""
    parser = argparse.ArgumentParser(description='Frozen three-member aggregate storage admission')
    parser.add_argument('operation', choices=('panel-freeze', 'panel-commands', 'panel-admit',
                                             'panel-member', 'panel-resume', 'panel-close',
                                             'panel-release-censored'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sha256')
    parser.add_argument('--member', choices=tuple(PANEL_NODES))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--valid-seconds', type=float, default=3600)
    parser.add_argument('--staged-manifest')
    opts = parser.parse_args(args)
    try:
        if opts.operation == 'panel-freeze':
            if not opts.request or opts.sha256 or opts.member or opts.staged_manifest:
                raise ValueError('panel-freeze requires --request; not a launch/digest/member operation')
            result = panel_freeze(opts.request, opts.manifest, root=root, valid_seconds=opts.valid_seconds)
        else:
            if not opts.sha256 or opts.request or opts.valid_seconds != 3600:
                raise ValueError('operation requires --sha256; freeze options are not launch options')
            if opts.operation == 'panel-commands':
                if opts.member:
                    raise ValueError('panel-commands renders all three members')
                result = panel_commands(opts.manifest, opts.sha256, staged_manifest=opts.staged_manifest)
            else:
                if original.absolute(root) != PANEL_ROOT or opts.staged_manifest:
                    raise ValueError('panel CLI requires the exact shared execution root: ' + str(PANEL_ROOT))
                if opts.operation == 'panel-member':
                    if not opts.member:
                        raise ValueError('panel-member requires --member')
                    result = panel_member(opts.manifest, opts.sha256, opts.member, root=root)
                else:
                    if opts.member:
                        raise ValueError('--member is only valid for panel-member')
                    action = {'panel-admit': panel_admit, 'panel-resume': panel_resume,
                              'panel-close': panel_close,
                              'panel-release-censored': panel_release_censored}[opts.operation]
                    result = action(opts.manifest, opts.sha256, root=root)
        code = result['exit_code']
    except (ValueError, OSError) as exc:
        result, code = dict(status='REFUSED', error=str(exc), command_started=False, hard_quota=False), 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return code
