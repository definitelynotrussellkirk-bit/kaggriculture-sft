"""Cooperative storage supervision; never deletes project data.

The current v2 policy checks free space on each declared filesystem, not total
project bytes. Historical v1 policies count all root entries, including hidden
files and staging. Symlinks are never followed and hard links count once by
(device,inode). Neither mode is an OS quota or an atomic snapshot.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import shutil
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
LIMIT = 60_000_000_000
SCHEMA = 'farm-project-storage-budget-v1'
FREE_SCHEMA = 'farm-project-storage-free-space-v2'
FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def natural(value):
    if type(value) is not int or value < 0:
        raise ValueError('nonnegative integer bytes required')
    return value


def absolute(path):
    path = Path(path)
    if '..' in path.parts:
        raise ValueError('parent traversal refused')
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlinked root/control path refused')
    return path


def configuration(root):
    path = absolute(root) / 'PROJECT-STORAGE.json'
    if path.is_symlink():
        raise ValueError('symlinked storage policy refused')
    if not path.exists():
        return {'schema': SCHEMA, 'limit_bytes': LIMIT, 'additional_roots': [],
                'accounting': 'ALLOCATED_UNIQUE_INODE_BYTES', 'automatic_deletion': False}
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError('duplicate storage policy key')
            value[key] = item
        return value
    value = json.loads(path.read_text(), object_pairs_hook=pairs)
    if value.get('schema') == SCHEMA:
        if not 0 < natural(value.get('limit_bytes')) <= LIMIT:
            raise ValueError('legacy storage policy cannot exceed decimal60GB')
        if value.get('accounting') != 'ALLOCATED_UNIQUE_INODE_BYTES':
            raise ValueError('unsupported legacy storage accounting policy')
    elif value.get('schema') == FREE_SCHEMA:
        if value.get('limit_bytes') is not None:
            raise ValueError('free-space policy has no project byte limit')
        if natural(value.get('minimum_free_bytes')) == 0:
            raise ValueError('positive minimum filesystem free space required')
        if value.get('accounting') != 'FILESYSTEM_FREE_BYTES':
            raise ValueError('unsupported free-space accounting policy')
    else:
        raise ValueError('unsupported storage policy schema')
    if value.get('automatic_deletion') is not False:
        raise ValueError('unsupported storage deletion policy')
    if type(value.get('additional_roots')) is not list or any(type(p) is not str or not p for p in value['additional_roots']):
        raise ValueError('explicit additional Farm root paths required')
    return value


def free_space_status(root, roots, policy, reserve_bytes, started):
    """Check physical headroom on every distinct filesystem, without a size cap."""
    filesystems, errors, seen = [], [], set()
    minimum = policy['minimum_free_bytes']
    for raw in roots:
        try:
            path = absolute(raw)
            device = os.stat(path, follow_symlinks=False).st_dev
            if device in seen:
                continue
            seen.add(device)
            usage = shutil.disk_usage(path)
            filesystems.append({
                'path': str(path), 'device': device,
                'total_bytes': usage.total, 'used_bytes': usage.used,
                'free_bytes': usage.free,
                'projected_free_bytes': usage.free - reserve_bytes,
                'minimum_free_bytes': minimum,
                'admissible': usage.free - reserve_bytes >= minimum,
            })
        except (OSError, ValueError) as error:
            errors.append({'path': str(raw), 'error': str(error)})
    complete = not errors
    admissible = complete and all(row['admissible'] for row in filesystems)
    return {
        'schema': 'farm-project-storage-free-space-status-v2', 'root': str(root),
        'policy': policy, 'filesystems': filesystems, 'errors': errors,
        'complete': complete, 'elapsed_seconds': time.monotonic() - started,
        'allocated_bytes': None, 'projected_bytes': None,
        'requested_reserve_bytes': reserve_bytes,
        'remaining_bytes': min((row['free_bytes'] - minimum for row in filesystems), default=None),
        'status': ('INCOMPLETE_CHECK' if not complete else
                   'WITHIN_FREE_SPACE' if admissible else 'LOW_DISK_FREE_SPACE'),
        'admissible': admissible, 'reservation_acquired': False,
        'hard_quota': False, 'atomic_snapshot': False, 'automatic_deletion': False,
        'scan_attempts': [],
        'limits': ('Physical free space is checked cooperatively, not reserved by the OS; '
                   'filesystem quotas and concurrent unguarded writers may differ.'),
    }


def scan(roots, *, timeout_seconds=60):
    """No file bodies read; fd-relative O_NOFOLLOW descent rejects rename races."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('finite positive scan timeout required')
    started = time.monotonic()
    seen = set()
    result = {'allocated_bytes': 0, 'logical_bytes': 0, 'logical_file_bytes': 0,
              'file_count': 0, 'unique_file_count': 0, 'directory_count': 0,
              'symlink_count': 0, 'other_count': 0, 'hardlink_duplicate_files': 0,
              'unique_inodes': 0, 'errors': [], 'roots': [str(p) for p in roots]}
    def deadline():
        if time.monotonic()-started > timeout_seconds:
            raise TimeoutError('storage scan deadline exceeded; partial scan is not admission')
    def account(info):
        regular = stat.S_ISREG(info.st_mode)
        if regular: result['file_count'] += 1
        elif stat.S_ISDIR(info.st_mode): result['directory_count'] += 1
        elif stat.S_ISLNK(info.st_mode): result['symlink_count'] += 1
        else: result['other_count'] += 1
        key = (info.st_dev, info.st_ino)
        if key in seen:
            if regular: result['hardlink_duplicate_files'] += 1
            return False
        seen.add(key)
        if not hasattr(info, 'st_blocks'):
            raise ValueError('allocated-byte accounting unavailable on this filesystem')
        result['allocated_bytes'] += info.st_blocks * 512
        result['logical_bytes'] += info.st_size
        if regular:
            result['unique_file_count'] += 1
            result['logical_file_bytes'] += info.st_size
        return True
    def walk(fd, label):
        deadline()
        with os.scandir(fd) as entries:
            for entry in entries:
                deadline()
                name = label / entry.name
                try:
                    info = entry.stat(follow_symlinks=False)
                    if not account(info):
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        child = os.open(entry.name, FLAGS, dir_fd=fd)
                        try:
                            opened = os.fstat(child)
                            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                                raise ValueError('directory changed during no-follow scan')
                            walk(child, name)
                        finally:
                            os.close(child)
                except (OSError, ValueError) as error:
                    if isinstance(error, TimeoutError): raise
                    result['errors'].append({'path': str(name), 'error': str(error)})
    for raw in roots:
        try:
            path = absolute(raw)
            fd = os.open(path, FLAGS)
            try:
                if account(os.fstat(fd)):
                    walk(fd, path)
            finally:
                os.close(fd)
        except (OSError, ValueError) as error:
            result['errors'].append({'path': str(raw), 'error': str(error)})
            if isinstance(error, TimeoutError): break
    result.update(unique_inodes=len(seen), complete=not result['errors'], elapsed_seconds=time.monotonic()-started)
    return result


def status(root=ROOT, *, reserve_bytes=0, scan_timeout=60):
    started = time.monotonic()
    root = absolute(root)
    reserve_bytes = natural(reserve_bytes)
    policy = configuration(root)
    roots = [root]
    for name in policy['additional_roots']:
        path = Path(name)
        roots.append(absolute(path if path.is_absolute() else root / path))
    if policy['schema'] == FREE_SCHEMA:
        return free_space_status(root, roots, policy, reserve_bytes, started)
    measured = scan(roots, timeout_seconds=scan_timeout)
    attempts = []
    def retain_attempt(row):
        return {key: row[key] for key in ('allocated_bytes', 'complete', 'errors', 'elapsed_seconds')}
    attempts.append(retain_attempt(measured))
    # A concurrent unlink can invalidate one walk. Never admit that partial
    # count: permit one entirely fresh walk inside the ORIGINAL time budget.
    # Persistent unreadability, deadline or an over-limit complete scan refuse.
    remaining = scan_timeout - (time.monotonic() - started)
    if not measured['complete'] and remaining > 0:
        measured = scan(roots, timeout_seconds=remaining)
        attempts.append(retain_attempt(measured))
    measured = {**measured, 'elapsed_seconds': time.monotonic() - started}
    projected = measured['allocated_bytes'] + reserve_bytes
    admissible = measured['complete'] and projected <= policy['limit_bytes']
    return {'schema': 'farm-project-storage-status-v1', 'root': str(root), 'policy': policy,
            **measured, 'scan_attempts': attempts,
            'requested_reserve_bytes': reserve_bytes, 'projected_bytes': projected,
            'remaining_bytes': policy['limit_bytes']-measured['allocated_bytes'],
            'status': 'WITHIN_BUDGET' if admissible else ('INCOMPLETE_SCAN' if not measured['complete'] else 'OVER_BUDGET'),
            'admissible': admissible, 'reservation_acquired': False,
            'hard_quota': False, 'atomic_snapshot': False, 'automatic_deletion': False,
            'limits': 'No symlink target traversal; inode-dedup, not extent-dedup. Unlinked-open files and undeclared external temps are not measurable here. Concurrent unguarded writers may grow after scan.'}


def _terminate(process, *, grace_seconds=2.0):
    """TERM then KILL only a child launched with start_new_session=True.

    The leader may already have exited while its same-group children remain.
    Do not use process.terminate()/kill(), which signal only the leader. Detached
    sessions are deliberately outside this cooperative supervisor's scope.
    """
    if not math.isfinite(grace_seconds) or grace_seconds < 0:
        raise ValueError('finite nonnegative termination grace required')
    pgid = process.pid
    if type(pgid) is not int or pgid <= 1 or pgid == os.getpgrp():
        raise ValueError('refuse to signal an unowned process group')
    # A live leader must still own the private group created by our spawn.
    if process.poll() is None:
        try:
            if os.getpgid(pgid) != pgid:
                raise ValueError('storage child is not its own process-group leader')
        except ProcessLookupError:
            process.poll()

    def send(number):
        try:
            os.killpg(pgid, number)
            return True
        except ProcessLookupError:
            return False

    if send(signal.SIGTERM):
        # A scan-recheck may have stopped this group. Let TERM handlers run.
        send(signal.SIGCONT)
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            process.poll()  # Reap the direct child if it has exited.
            if not send(0):
                break
            time.sleep(min(.02, max(0.0, deadline-time.monotonic())))
        else:
            send(signal.SIGKILL)
    process.wait(timeout=max(1.0, grace_seconds))


def run(command, *, root=ROOT, reserve_bytes, timeout_seconds, interval_seconds=10,
        scan_timeout=60, inspect=status, spawn=subprocess.Popen, terminate=_terminate,
        clock=time.monotonic):
    """Serialize compliant launchers; monitor/kill ONLY this child process group.

    The reserve is an estimate of total extra peak allocation, not an OS-enforced
    lease. Periodic scanning can detect, but cannot prevent, brief overshoot.
    """
    root = absolute(root)
    if natural(reserve_bytes) == 0:
        raise ValueError('run requires a positive explicit peak-growth reserve')
    if not command or any(type(arg) is not str for arg in command):
        raise ValueError('explicit command argv required')
    for value in (timeout_seconds, interval_seconds, scan_timeout):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('finite positive deadline/interval required')
    if interval_seconds > 60 or scan_timeout > 60:
        raise ValueError('scan interval and scan timeout must each be at most60 seconds')
    controls = root / 'work/cache/storage'
    absolute(controls)
    controls.mkdir(parents=True, exist_ok=True)
    lock_path = absolute(controls / 'COMMAND.lock')
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    child = None
    cleaned = False
    previous_signals = {}
    started = clock()
    scans = []
    report = {'schema': 'farm-project-storage-command-v1', 'command': command, 'root': str(root),
              'reserve_bytes': reserve_bytes, 'timeout_seconds': timeout_seconds,
              'interval_seconds': interval_seconds, 'hard_quota': False,
              'cooperative_serialization': True, 'automatic_deletion': False}
    def measure(reserve):
        row = inspect(root, reserve_bytes=reserve, scan_timeout=min(scan_timeout, max(.001, timeout_seconds-(clock()-started))))
        scans.append({key: row[key] for key in ('status', 'allocated_bytes', 'projected_bytes', 'admissible', 'complete', 'elapsed_seconds', 'errors', 'scan_attempts') if key in row})
        return row
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {**report, 'status': 'COOPERATIVE_LAUNCHER_BUSY', 'exit_code': 2, 'command_started': False}
        def interrupted(number, frame):
            raise KeyboardInterrupt('storage owner received signal ' + str(number))
        for number in (signal.SIGTERM, signal.SIGINT):
            previous_signals[number] = signal.signal(number, interrupted)
        admission = measure(reserve_bytes)
        if not admission['admissible']:
            return {**report, 'status': 'ADMISSION_REFUSED', 'exit_code': 2, 'command_started': False, 'scans': scans}
        if clock()-started >= timeout_seconds:
            return {**report, 'status': 'DEADLINE_BEFORE_LAUNCH', 'exit_code': 124, 'command_started': False, 'scans': scans}
        child = spawn(command, cwd=root, start_new_session=True)
        report.update(command_started=True, child_pid=child.pid)
        reason = 'COMMAND_EXITED'
        while True:
            remaining = timeout_seconds-(clock()-started)
            if remaining <= 0:
                reason = 'TIMEOUT'
                break
            try:
                child.wait(timeout=min(interval_seconds, remaining))
                break
            except subprocess.TimeoutExpired:
                if clock()-started >= timeout_seconds:
                    reason = 'TIMEOUT'
                    break
                if not measure(0)['admissible']:
                    reason = 'STORAGE_LIMIT_OR_SCAN_FAILURE'
                    break
        # Clean any same-session descendants even when the leader already exited.
        terminate(child)
        cleaned = True
        code = child.returncode
        if reason == 'COMMAND_EXITED':
            if not measure(0)['admissible']:
                reason = 'STORAGE_LIMIT_OR_SCAN_FAILURE'
        return {**report, 'status': reason, 'exit_code': 124 if reason == 'TIMEOUT' else (2 if reason == 'STORAGE_LIMIT_OR_SCAN_FAILURE' else (code if code >= 0 else 128-code)),
                'child_exit_code': code, 'elapsed_seconds': clock()-started, 'scans': scans,
                'monitor_scope': 'Only this fresh process group; detached sessions and preexisting writers are outside enforcement.'}
    finally:
        try:
            if child is not None and not cleaned:
                terminate(child)
        finally:
            # Even a cleanup failure must not leak handlers or the admission lock.
            try:
                for number, handler in previous_signals.items():
                    signal.signal(number, handler)
            finally:
                os.close(fd)


def main(argv=None, *, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=root)
    commands = parser.add_subparsers(dest='operation', required=True)
    for name in ('status', 'check'):
        sub = commands.add_parser(name)
        sub.add_argument('--reserve-bytes', type=int, default=0)
        sub.add_argument('--scan-timeout', type=float, default=60)
    sub = commands.add_parser('run')
    sub.add_argument('--reserve-bytes', type=int, required=True)
    sub.add_argument('--timeout', type=float, required=True)
    sub.add_argument('--interval', type=float, default=10)
    sub.add_argument('--scan-timeout', type=float, default=60)
    sub.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.operation == 'run':
            command = args.command[1:] if args.command[:1] == ['--'] else args.command
            result = run(command, root=args.root, reserve_bytes=args.reserve_bytes, timeout_seconds=args.timeout,
                         interval_seconds=args.interval, scan_timeout=args.scan_timeout)
            code = result['exit_code']
        else:
            result = status(args.root, reserve_bytes=args.reserve_bytes, scan_timeout=args.scan_timeout)
            code = 0 if result['admissible'] else 2
    except (ValueError, OSError) as error:
        result, code = {'status': 'REFUSED', 'error': str(error), 'hard_quota': False}, 2
    except KeyboardInterrupt as error:
        result, code = {'status': 'INTERRUPTED', 'error': str(error), 'hard_quota': False}, 130
    print(json.dumps(result, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
