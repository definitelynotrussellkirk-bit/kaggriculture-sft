"""Bounded single-node search: reached queue -> batched proposals -> exact archive.

No model is trained and no teacher is admitted here. Recovery actor facts are
adapted ONLY for proposal geometry. Exact jobs restore separately authenticated
full counterfactual captures; a projected actor record is never an exact cut.

One reached-state lease is active at a time. Its <=32 stochastic proposals are
batched through the existing whole-session decoder; at most ``workers`` exact
CPU pairs are in flight. This deliberately small first-node implementation does
not promise overlapping GPU work across reached states or multi-node scheduling.

Every queue call has a durable, replayable intent. Every exact job has an archive
ATTEMPT before dispatch, an exclusive execution claim, and an immutable OUTCOME
spool before completion notification. Charged jobs are never resubmitted. A crash
without a durable outcome is explicitly censored/unknown, not assigned a payoff.
Native children inherit a lifetime flock so a restarted parent can distinguish
live work from a dead claimed start even before the child writes its PID.

Run ``python -m farm_neat.training.v0_search_worker --help``. Production requires
one allocated 4090, the pinned runtime, ordinary RUN.py storage run, authentic
checkpoint bytes and a manifest of matched full cuts. Tests inject outcomes and
executors; there is no CPU fallback for production inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

from farm_neat.training import v0_search_archive as A
from farm_neat.training import v0_search_queue as Q
from farm_neat.training import v0_regret_records as R
from farm_neat.training.teachers import v0_regret_proposals as P
from farm_neat.training.teachers import v0_regret_searcher as S

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'farm-v0-search-worker-v1'
MAX_JSON_BYTES = 64 * 1024**2


class WorkerError(ValueError):
    """A local preflight/custody error; never permission to reroll."""


class NativeWorkerLost(WorkerError):
    """An exact coordinator died; detached native arms need allocation cleanup."""


def need(ok: bool, code: str) -> None:
    if not ok:
        raise WorkerError(code)


def read(path: str | Path) -> dict:
    path = A.safe_path(path)
    need(path.is_file(), 'MISSING_INPUT: ' + str(path))
    need(path.stat().st_size <= MAX_JSON_BYTES, 'WORKER_JSON_CAP')
    return A.read_json(path, max_bytes=MAX_JSON_BYTES)


def publish(path: str | Path, value: dict, *, replace: bool = False) -> dict:
    """fsync + atomic publication; immutable evidence retries require exact bytes."""
    path = A.safe_path(path)
    data = A.wire(value) + b'\n'
    need(len(data) <= MAX_JSON_BYTES, 'WORKER_JSON_CAP')
    if path.exists() and not replace:
        need(path.read_bytes() == data, 'IMMUTABLE_FILE_CONFLICT: ' + str(path))
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        Q._sync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return value


def sealed_read(path: Path) -> dict:
    value = read(path)
    need(value.get('content_sha256') == A.seal(value)['content_sha256'], 'WORKER_CHECKPOINT_HASH')
    return value


@contextmanager
def exclusive(path: Path):
    path = A.safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkerError('WORKER_BUSY') from exc
        yield fd
    finally:
        os.close(fd)


def job_busy(path: Path) -> bool:
    try:
        with exclusive(path):
            return False
    except WorkerError:
        return True


def proposal_context(recovery: dict) -> dict:
    """Explicit geometry-only bridge, not a recovery/teacher schema conversion.

    The original recovery projection remains in the queue. ``controller=None``
    denotes that the proposer has NO controller, not an assertion that the real
    actor has none. All existing own facts, ordered HOLD memory and prefix IDs
    are copied unchanged except that the projection's documented default
    ``belief=None`` is made explicit as the all-zero product vector. This is the
    same normalization used by economy.v0.projection.Projection; nonzero or
    unsupported memory is refused by the existing proposal owner. The original
    None-valued actor record and its hash remain in the queue. Native execution
    revalidates against the full cut.
    """
    Q._fields(recovery, ('schema', 'observation', 'prefix', 'compiler_memory'))
    need(recovery['schema'] == 'farm-v0-regret-recovery-actor-v2', 'RECOVERY_ACTOR_SCHEMA')
    result = copy.deepcopy(recovery)
    result['schema'] = R.ACTOR_SCHEMA
    result['compiler_memory']['controller'] = None
    if result['compiler_memory']['belief'] is None:
        from farm_neat.economy.v0 import vocabulary as V
        result['compiler_memory']['belief'] = dict.fromkeys(V.PRODUCTS, 0)
    R._actor(result)
    return result


@dataclass(frozen=True)
class Limits:
    workers: int = 6
    max_states: int = 16
    seconds: float = 600.
    drain_seconds: float = 20.
    poll_seconds: float = .05
    lease_ms: int = 3_600_000

    def validate(self):
        need(type(self.workers) is int and 1 <= self.workers <= 64, 'WORKER_COUNT')
        need(type(self.max_states) is int and 1 <= self.max_states <= 2048, 'STATE_CAP')
        for k in ('seconds', 'drain_seconds', 'poll_seconds'):
            v = getattr(self, k)
            need(type(v) in (float, int) and math.isfinite(v) and v > 0, 'FINITE_' + k.upper())
        need(type(self.lease_ms) is int and 1 <= self.lease_ms <= Q.MAX_TTL_MS, 'LEASE_TTL')
        need(self.lease_ms > 1000 * (self.seconds + self.drain_seconds + 5), 'LEASE_SHORTER_THAN_SEGMENT')
        return self


def confirmation_summary(cells: list[dict], *, expected: list[dict], alpha: float,
                         min_effect: float) -> dict:
    """Existing recovery fixed-N, equal-family paired WDL Hoeffding semantics.

    This reports statistical support, NOT teacher-schema admission. Replays and
    two seats in one world family never increase independent sample size.
    """
    need(type(alpha) in (int, float) and math.isfinite(alpha) and 0 < alpha < 1, 'CONFIRMATION_ALPHA')
    need(type(min_effect) in (int, float) and math.isfinite(min_effect) and 0 <= min_effect < 2,
         'CONFIRMATION_MIN_EFFECT')
    ids = [c['source_group_id'] for c in expected]
    need(len(ids) == len(set(ids)), 'CONFIRMATION_DUPLICATE_GROUP')
    out = dict(schema=SCHEMA + '-confirmation', statistical_support=False,
        actor_training_admissible=False, teacher_export='NOT_ADMITTED',
        n_world_groups=0, mean_wdl_delta=None, lower_bound=None,
        uncertainty='FIXED_N_BOUNDED_PAIRED_WDL_HOEFFDING_ONE_SIDED', reason_codes=[])
    found = {c['source_group_id']: c for c in cells}
    if not ids or len(found) != len(cells) or set(found) != set(ids):
        return out | {'reason_codes': ['CENSORED_OR_MISSING_OUTCOMES']}
    families: dict[str, list[float]] = {}
    native = True
    for meta in expected:
        cell = found[meta['source_group_id']]
        f = cell.get('facts')
        if cell.get('status') != 'COMPLETE' or not f or not f['complete'] or not f['fresh_replay']:
            return out | {'reason_codes': ['CENSORED_OR_MISSING_OUTCOMES']}
        need(cell['source_family_id'] == meta['source_family_id'], 'CONFIRMATION_FAMILY_DRIFT')
        delta = f['challenger_wdl'] - f['learner_wdl']
        need(math.isfinite(delta) and -2 <= delta <= 2 and delta == f['wdl_delta'], 'CONFIRMATION_WDL')
        native = native and f['native']
        families.setdefault(meta['source_family_id'], []).append(delta)
    means = [math.fsum(values) / len(values) for values in families.values()]
    delta = math.fsum(means) / len(means)
    lower = delta - math.sqrt(8 * -math.log(alpha) / len(means))
    out.update(n_world_groups=len(means), mean_wdl_delta=delta, lower_bound=lower)
    if not native:
        out['reason_codes'] = ['INJECTED_OUTCOMES_NOT_NATIVE_SUPPORT']
    elif lower <= min_effect:
        out['reason_codes'] = ['NO_MEAN_WDL_ADVANTAGE' if delta == 0 else
                               'REALIZED_LEARNER_BETTER' if delta < 0 else 'INSUFFICIENT_POWER']
    else:
        out['statistical_support'] = True
    return out


def _failure(job: dict, code: str, status: str = 'FAILED') -> dict:
    return A.outcome_event(job['evaluation_id'], job['candidate']['content_sha256'],
                          status=status, charged=job['charged'], reason_codes=[code])


def execute_spooled_job(job_path: str | Path, evaluator: Callable[[dict], dict]) -> dict:
    """Trusted injected/native seam. Outcome bytes precede completion notification."""
    path = Path(job_path)
    job = sealed_read(path)
    claim = path.parent / 'EXECUTION.claim'
    need(not claim.exists(), 'EXACT_EXECUTION_ALREADY_CLAIMED')
    # Exclusive create is execution permission; immutable publish idempotence is not.
    with claim.open('x') as stream:
        json.dump({'pid': os.getpid(), 'started_ns': time.time_ns()}, stream)
        stream.flush(); os.fsync(stream.fileno())
    Q._sync_dir(path.parent)
    start, cpu = time.monotonic(), time.process_time()
    try:
        event = evaluator(copy.deepcopy(job))
        A._event(event)
        need(event['kind'] == 'OUTCOME' and event['evaluation_id'] == job['evaluation_id']
             and event['candidate_sha256'] == job['candidate']['content_sha256'], 'EVALUATOR_EVENT_JOIN')
    except Exception as exc:
        event = _failure(job, 'EXACT_WORKER_FAILED_' + type(exc).__name__)
    result = A.seal(dict(schema=SCHEMA + '-spool', job_sha256=job['content_sha256'], event=event,
        elapsed_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != 'darwin' else 1)))
    publish(path.parent / 'OUTCOME.json', result)
    return result


def evaluate_native(job: dict) -> dict:
    request = A.read_artifact(job['request'], root=Path(job['evidence_root']))
    first = A.evaluate_suffix_pair(request, runner=A.native_suffix_runner, evidence_root=job['evidence_root'])
    result = A.verify_suffix_pair(request, first, runner=A.native_suffix_runner, evidence_root=job['evidence_root'])
    path = Path(job['result_path'])
    publish(path, result)
    complete = all(a['status'] == 'COMPLETE' for a in result['arms'])
    return A.outcome_event(job['evaluation_id'], job['candidate']['content_sha256'],
        status='COMPLETE' if complete else 'CENSORED', charged=job['charged'],
        evidence={'request': job['request'], 'result': A.artifact_ref(path, root=job['evidence_root'])},
        reason_codes=[] if complete else ['INCOMPLETE_OR_INVALID_SUFFIX'])


class _ProcessFuture:
    def __init__(self, process, log):
        self.process, self.log = process, log

    def done(self):
        return self.process.poll() is not None

    def result(self):
        code = self.process.wait()
        self.log.close()
        if code != 0:
            raise NativeWorkerLost('EXACT_PROCESS_EXIT_' + str(code))


def _descendants(pid: int) -> list[int]:
    """Snapshot only this owned worker's Linux descendants, deepest first."""
    children: dict[int, list[int]] = {}
    for member in Path('/proc').iterdir():
        if member.name.isdigit():
            try:
                stat = (member / 'stat').read_text().rsplit(')', 1)[1].split()
                children.setdefault(int(stat[1]), []).append(int(member.name))
            except (OSError, ValueError, IndexError):
                continue
    def visit(parent):
        result = []
        for child in children.get(parent, []):
            result.extend(visit(child)); result.append(child)
        return result
    return visit(pid)


class SubprocessExecutor:
    """Small local exact-pair launcher; not a scheduler or a persistent daemon."""
    def __init__(self):
        self.futures: list[_ProcessFuture] = []
        self.cleanup_uncertain = False

    def submit(self, job_path: Path):
        guard = job_path.parent / 'execution.lock'
        fd = os.open(guard, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        log = None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            log = (job_path.parent / 'worker.log').open('ab')
            env = dict(os.environ, FARM_SEARCH_GUARD_FD=str(fd), PYTHONDONTWRITEBYTECODE='1')
            process = subprocess.Popen([sys.executable, '-B', '-m', 'farm_neat.training.v0_search_worker', '_evaluate',
                '--job', str(job_path)], env=env, cwd=ROOT, stdout=log, stderr=log,
                pass_fds=(fd,))
            future = _ProcessFuture(process, log)
            self.futures.append(future)
            return future
        except BaseException:
            if log is not None:
                log.close()
            raise
        finally:
            os.close(fd)  # child keeps this open-file-description lock until exit

    def shutdown(self):
        for future in self.futures:
            process = future.process
            if process.poll() is None:
                # Native archive arms create their own sessions. Kill known owned
                # descendants explicitly, not unrelated process groups/jobs.
                # Freeze the owned exact coordinator before enumerating its
                # children; it cannot start a new detached archive arm mid-scan.
                try:
                    os.kill(process.pid, signal.SIGSTOP)
                except ProcessLookupError:
                    self.cleanup_uncertain = True
                for pid in _descendants(process.pid) + [process.pid]:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=5)
            elif process.returncode:
                self.cleanup_uncertain = True
            future.log.close()


class SearchWorker:
    """One bounded producer with a backpressured exact consumer pool.

    bridge owns full-cut request construction; executor.submit(job_path) starts
    execute_spooled_job and returns a done()/result() handle. Both are injected
    in tests. Model input always comes from the queue's original allowlist.
    """
    def __init__(self, *, queue, archive, directory, registration, bridge,
                 tensor_decoder, executor, limits=Limits(), clock=time.monotonic,
                 utc_ms=lambda: time.time_ns() // 1_000_000):
        self.queue, self.archive = queue, archive
        self.directory = A.safe_path(directory).resolve()
        need(self.directory.is_relative_to(archive.evidence_root), 'WORK_OUTSIDE_EVIDENCE_ROOT')
        self.registration = S._batch_registration(registration)
        self.bridge, self.decoder, self.executor = bridge, tensor_decoder, executor
        self.limits, self.clock, self.utc_ms = limits.validate(), clock, utc_ms
        self.path = self.directory / 'STATE.json'
        self.inflight: dict[str, Any] = {}
        self.stop_requested = False
        self.deadline: float | None = None
        self.binding = A.digest(dict(schema=SCHEMA, queue=str(queue.path.resolve()),
            queue_identity=queue.identity, archive=str(archive.path.resolve()),
            registration=self.registration, bridge_identity=bridge.identity))
        self.state: dict = {}
        self.active: dict | None = None

    def stop(self):
        self.stop_requested = True

    def _save(self):
        if self.active is not None:
            publish(self.directory / 'rounds' / self.active['candidate_id'] / 'ROUND.json',
                    A.seal(self.active), replace=True)
        queue_state = self.queue.inspect()
        self.state['queue_offset'] = queue_state['sequence']
        self.state['queue_head_sha256'] = queue_state['head_sha256']
        entries, offset, head = self.archive._read(verify_evidence=False)
        self.state['archive_offset'], self.state['archive_head_sha256'] = offset, head
        publish(self.path, A.seal(self.state), replace=True)

    def _stopping(self):
        return self.stop_requested or (self.deadline is not None and self.clock() >= self.deadline)

    def _now(self):
        return max(int(self.utc_ms()), self.queue.inspect()['clock_ms'])

    def _queue_call(self, name, args):
        """Arguments (including explicit clock) persist BEFORE the side effect."""
        pending = self.state.get('queue_call')
        if pending is None:
            n = self.state['queue_calls']
            pending = dict(name=name, args={**args, 'request_id': self.state['worker'] + f'-q{n}',
                                          'now_ms': self._now()})
            # Queue idempotence compares ordered JSON bytes. Canonicalize the
            # replayable intent BEFORE its first call, not only when writing it.
            # Otherwise restart sorts nested result keys and changes the request.
            pending = json.loads(A.wire(pending))
            self.state['queue_call'] = pending
            self.state['queue_calls'] += 1
            self._save()
        need(pending['name'] == name, 'UNRESOLVED_QUEUE_CALL')
        return getattr(self.queue, name)(**pending['args'])

    def _queue_done(self):
        self.state['queue_call'] = None
        self._save()

    def _start(self):
        response = self._queue_call('lease', {'worker': self.state['worker'], 'ttl_ms': self.limits.lease_ms})
        if response['status'] == 'EMPTY':
            self._queue_done()
            return False
        cid = response['candidate_id']
        path = self.directory / 'rounds' / cid / 'ROUND.json'
        if path.exists():
            self.active = sealed_read(path)
            need(self.active['lease']['lease_id'] == response['lease_id'], 'REUSED_REACHED_STATE')
        else:
            self.active = dict(schema=SCHEMA + '-round', candidate_id=cid, lease=response,
                status='LEASED', phase='discovery', jobs={}, proposal=None, selection=None,
                confirmation=None, queue_result=None, error=None, prepared_seconds=0.)
        self.state['active'] = cid
        self._queue_done()
        return True

    def _charge(self):
        a = self.active
        candidate = self.queue.get_candidate(a['candidate_id'])
        # Missing actual inputs/cost allocation fail before any proposal/GPU work.
        if not a.get('row'):
            a['row'] = self.bridge.row(candidate)
            a['discovery'] = self.bridge.cells(candidate, 'discovery')
            a['confirmation_cells'] = self.bridge.cells(candidate, 'confirmation')
            self.bridge.validate(candidate, self.registration, a['discovery'], a['confirmation_cells'])
            self._save()
        response = self._queue_call('charge', {'worker': self.state['worker'], 'lease_id': a['lease']['lease_id']})
        a['status'] = 'STARTED' if response['execute_permission'] else 'LOST_CHARGE_ACK'
        self._queue_done()
        return response['execute_permission']

    def _population(self, signals=()):
        a = self.active
        path = self.directory / 'rounds' / a['candidate_id'] / 'PROPOSALS.json'
        begin = self.clock()
        result = S.progressive_proposals([a['row']], registration=self.registration,
            tensor_decoder=self.decoder, checkpoint_path=path, pause_after_stages=1,
            discovery_signals=signals)
        a['prepared_seconds'] += self.clock()-begin
        a['proposal'] = result
        self._save()
        return result

    def _add_jobs(self, phase, proposals, cells):
        a = self.active
        candidate = self.queue.get_candidate(a['candidate_id'])
        for proposal in proposals:
            for cell in cells:
                eid = A.digest([self.binding, a['candidate_id'], phase,
                                proposal['candidate_id'], cell['source_group_id']])
                if eid in a['jobs']:
                    continue
                if self._stopping():
                    return False
                folder = self.directory / 'rounds' / a['candidate_id'] / 'jobs' / eid
                folder.mkdir(parents=True, exist_ok=True)
                # Request construction is deterministic, validates geometry and
                # executes no native turns. A failed preparation is retained.
                try:
                    spec = self.bridge.prepare(candidate, proposal, cell, phase, folder, eid)
                    job = A.seal(dict(schema=SCHEMA + '-job', evaluation_id=eid,
                        evidence_root=str(self.archive.evidence_root),
                        candidate=spec['candidate'], request=spec['request'], charged=spec['charged'],
                        result_path=str(folder / 'RESULT.json')))
                    publish(folder / 'JOB.json', job)
                    status, error = 'READY', None
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    status, error = 'PREPARATION_FAILED', type(exc).__name__ + ': ' + str(exc)
                a['jobs'][eid] = dict(evaluation_id=eid, proposal_id=proposal['candidate_id'],
                    phase=phase, source_group_id=cell['source_group_id'], source_family_id=cell['source_family_id'],
                    status=status, error=error, path=str(folder / 'JOB.json'), facts=None,
                    archive_status=None, elapsed_seconds=None, cpu_seconds=None, peak_rss_bytes=None)
                self._save()
        return True

    def _consume(self, row, event, spool=None):
        self.archive.append(event)
        entry = next(r for r in self.archive.entries(verify_evidence=False)
                     if r['evaluation_id'] == row['evaluation_id'])
        row.update(status='ARCHIVED', archive_status=entry['status'], facts=entry['facts'])
        if spool:
            for k in ('elapsed_seconds', 'cpu_seconds', 'peak_rss_bytes'):
                row[k] = spool[k]
        self._save()

    def _recover_job(self, row, entries):
        eid = row['evaluation_id']
        if row['status'] in ('ARCHIVED', 'PREPARATION_FAILED'):
            return
        path = Path(row['path']); spool_path = path.parent / 'OUTCOME.json'
        entry = entries.get(eid)
        if spool_path.exists():
            job, spool = sealed_read(path), sealed_read(spool_path)
            need(spool['job_sha256'] == job['content_sha256'], 'SPOOL_JOB_JOIN')
            need(entry is not None, 'SPOOL_WITHOUT_PRECHARGE')
            self._consume(row, spool['event'], spool)
        elif entry and entry.get('outcome'):
            row.update(status='ARCHIVED', archive_status=entry['status'], facts=entry['facts'])
            self._save()
        elif entry and eid not in self.inflight and not job_busy(path.parent / 'execution.lock'):
            self._consume(row, _failure(sealed_read(path), 'INTERRUPTED_CHARGED_START_UNKNOWN', 'CENSORED'))

    def _pump(self, *, launch=True):
        a = self.active
        entries = {r['evaluation_id']: r for r in self.archive.entries(verify_evidence=False)}
        for row in a['jobs'].values():
            self._recover_job(row, entries)
        for eid, future in list(self.inflight.items()):
            if not future.done():
                continue
            row = a['jobs'][eid]
            try:
                future.result()
            except Exception as exc:
                if isinstance(exc, NativeWorkerLost):
                    self.stop_requested = True
                    self.state['requires_allocation_cleanup'] = True
                if row['status'] != 'ARCHIVED':
                    self._consume(row, _failure(sealed_read(Path(row['path'])),
                                                'EXECUTOR_FAILED_' + type(exc).__name__))
            del self.inflight[eid]
            if row['status'] != 'ARCHIVED':
                self._recover_job(row, {r['evaluation_id']: r for r in self.archive.entries(verify_evidence=False)})
        if not launch:
            return
        for row in a['jobs'].values():
            if row['status'] != 'READY' or len(self.inflight) >= self.limits.workers or self._stopping():
                continue
            path = Path(row['path']); job = sealed_read(path)
            reserved = self.archive.append(A.attempt_event(job['evaluation_id'], job['candidate'], charged=job['charged']))
            row['status'] = 'DISPATCHED'
            self._save()  # charge and dispatch intent precede all execution
            if not reserved['inserted']:
                continue  # append idempotence is never a second grant
            try:
                self.inflight[row['evaluation_id']] = self.executor.submit(path)
                self.state['peak_inflight'] = max(self.state['peak_inflight'], len(self.inflight))
            except Exception as exc:
                self._consume(row, _failure(job, 'DISPATCH_FAILED_' + type(exc).__name__))
            self._save()

    def _jobs_done(self, phase):
        return all(r['status'] in ('ARCHIVED', 'PREPARATION_FAILED')
                   for r in self.active['jobs'].values() if r['phase'] == phase)

    def _signals(self):
        a = self.active
        receipt = a['proposal']['stage_receipts'][-1]
        result = []
        for cid in receipt['candidate_ids'][a['candidate_id']]:
            for group in self.registration['discovery_groups']:
                row = next(r for r in a['jobs'].values() if r['phase'] == 'discovery'
                           and r['proposal_id'] == cid and r['source_group_id'] == group)
                f = row['facts']
                complete = row['archive_status'] == 'COMPLETE' and bool(f) and f['complete'] and f['native']
                verified = complete and f['fresh_replay']
                result.append(dict(stage=receipt['stage'], queue_id=a['candidate_id'],
                    pool_sha256=receipt['pool_sha256'], candidate_id=cid, phase='discovery',
                    source_group_id=group, complete=bool(complete), replay_verified=bool(verified),
                    wdl_delta=f['wdl_delta'] if verified else None))
        return result

    def _select(self):
        a = self.active
        scores, controls = [], {}
        for proposal in a['proposal']['states'][0]['candidates']:
            rows = [r for r in a['jobs'].values() if r['phase'] == 'discovery'
                    and r['proposal_id'] == proposal['candidate_id']]
            complete = len(rows) == len(a['discovery'])
            families = {}
            for row in rows:
                f = row['facts']
                if not (row['archive_status'] == 'COMPLETE' and f and f['complete'] and f['fresh_replay'] and f['native']):
                    complete = False; continue
                key = row['source_group_id']
                need(key not in controls or controls[key] == f['learner_wdl'], 'DISCOVERY_CONTROL_WDL_CONFLICT')
                controls[key] = f['learner_wdl']
                families.setdefault(row['source_family_id'], []).append(f['wdl_delta'])
            if complete and families:
                score = math.fsum(math.fsum(v)/len(v) for v in families.values())/len(families)
                if score > 0:
                    scores.append((score, proposal['candidate_id'], proposal))
        if not scores:
            a['selection'] = {'status': 'KEEP_DECLARED_CONTROL', 'raw_behavior_logp': None,
                              'class_behavior_logp': None, 'actor_training_admissible': False}
        else:
            # Stable ID resolves equal WDL, independent of callback completion order.
            score, _, proposal = sorted(scores, key=lambda x: (-x[0], x[1]))[0]
            a['selection'] = A.seal(dict(status='FROZEN_BEFORE_CONFIRMATION',
                candidate_id=proposal['candidate_id'], raw_actions=proposal['raw_actions'],
                canonical_plan=proposal['canonical_plan'], discovery_mean_wdl_delta=score,
                evidence_sha256=A.digest(a['jobs']), confirmation_registration_sha256=A.digest(a['confirmation_cells']),
                raw_behavior_logp=None, class_behavior_logp=None, actor_training_admissible=False,
                original_sampling_provenance=copy.deepcopy(proposal.get('sampling_provenance'))))
        self._save()  # freeze BEFORE even constructing confirmation jobs

    def _ack(self):
        a = self.active
        if a['queue_result'] is None:
            solved = a.get('selection', {}).get('status') == 'FROZEN_BEFORE_CONFIRMATION' if a.get('selection') else False
            status = 'SOLVED' if solved else 'FAILED' if a['error'] else 'EXHAUSTED'
            receipt = A.seal(dict(schema=SCHEMA + '-state-result', candidate_id=a['candidate_id'],
                selection=a['selection'], confirmation=a['confirmation'], jobs=a['jobs'],
                proposal_sha256=a['proposal']['content_sha256'] if a['proposal'] else None,
                error=a['error'], teacher_export='NOT_ADMITTED'))
            path = self.directory / 'rounds' / a['candidate_id'] / 'RECEIPT.json'
            publish(path, receipt)
            a['queue_result'] = dict(schema=Q.RESULT_SCHEMA, status=status,
                raw_actions=a['selection']['raw_actions'] if solved else None,
                receipt_sha256=receipt['content_sha256'], measured_costs=None)
            self._save()
        pending = self.state.get('queue_call')
        if pending is not None and pending['name'] in ('complete', 'censor'):
            # A prior process may have committed the terminal event but died
            # before consuming its response. Replay the exact original intent
            # even if inspect now shows COMPLETE/CENSORED; never strand it.
            self._queue_call(pending['name'], {})
            a['status'] = 'FINISHED'
            self._queue_done()
        row = self.queue.inspect()['candidates'][a['candidate_id']]
        if row['status'] in ('LEASED', 'STARTED') and row['worker'] == self.state['worker']:
            if row['status'] == 'LEASED' or a['status'] == 'LOST_CHARGE_ACK':
                self._queue_call('censor', dict(worker=self.state['worker'], lease_id=a['lease']['lease_id'],
                                               reason=a['error'] or 'INTERRUPTED_CHARGE_ACK'))
            else:
                self._queue_call('complete', dict(worker=self.state['worker'], lease_id=a['lease']['lease_id'],
                                                 result=a['queue_result']))
            a['status'] = 'FINISHED'
            self._queue_done()
        else:
            a['status'] = 'FINISHED' if row['status'] == 'COMPLETE' else 'QUEUE_CENSORED_RESULTS_RETAINED'
        self.state['completed'].append(a['candidate_id']) if a['candidate_id'] not in self.state['completed'] else None
        self.state['active'] = None
        self._save()
        self.active = None

    def _advance(self):
        a = self.active
        if a['status'] in ('LEASED', 'LOST_CHARGE_ACK'):
            if a['status'] == 'LOST_CHARGE_ACK' or not self._charge():
                a['error'] = 'INTERRUPTED_CHARGE_ACK'; self._ack(); return
        if a['queue_result'] is not None:
            self._ack(); return
        if a['proposal'] is None:
            self._population()
        proposals = a['proposal']['states'][0]['candidates']
        if not self._add_jobs('discovery', proposals, a['discovery']):
            return
        self._pump()
        if not self._jobs_done('discovery'):
            return
        if a['proposal']['status'] != 'COMPLETE':
            signals = self._signals() if self.registration['gate'] == 'DISCOVERY_POSITIVE' else ()
            self._population(signals)
            return
        if a['selection'] is None:
            self._select()
        if a['selection']['status'] == 'FROZEN_BEFORE_CONFIRMATION' and a['confirmation_cells']:
            chosen = next(p for p in proposals if p['candidate_id'] == a['selection']['candidate_id'])
            if not self._add_jobs('confirmation', [chosen], a['confirmation_cells']):
                return
            self._pump()
            if not self._jobs_done('confirmation'):
                return
            admission = self.bridge.admission(self.queue.get_candidate(a['candidate_id']))
            cells = [{**r, 'status': r['archive_status']} for r in a['jobs'].values() if r['phase'] == 'confirmation']
            a['confirmation'] = confirmation_summary(cells, expected=a['confirmation_cells'],
                                                      alpha=admission['alpha_pair'], min_effect=admission['min_effect'])
        else:
            a['confirmation'] = {'statistical_support': False, 'actor_training_admissible': False,
                                 'reason_codes': ['NO_DISCOVERY_ADVANTAGE_OR_REGISTERED_CONFIRMATION']}
        self._ack()

    def run(self) -> dict:
        self.directory.mkdir(parents=True, exist_ok=True)
        with exclusive(self.directory / 'worker.lock'):
            if self.path.exists():
                self.state = sealed_read(self.path)
                if isinstance(self.executor, SubprocessExecutor):
                    require_previous_allocation_closed(self.state)
                need(self.state['binding_sha256'] == self.binding, 'WORKER_BINDING_DRIFT')
                # Validate retained offsets against a prefix, not merely a newer
                # sequence count. A missing/replaced archive cannot be resumed.
                entries, count, head = self.archive._read(verify_evidence=True)
                need(count >= self.state['archive_offset'], 'ARCHIVE_ROLLBACK')
                retained = self.state['archive_offset']
                with self.archive._connect() as db:
                    prefix = db.execute('SELECT row_sha256 FROM events WHERE sequence=?',
                                        (retained-1,)).fetchone() if retained else None
                need((prefix[0] if prefix else A.ZERO) == self.state['archive_head_sha256'],
                     'ARCHIVE_HEAD_DRIFT')
                info = self.queue.inspect()
                need(info['sequence'] >= self.state['queue_offset'], 'QUEUE_ROLLBACK')
                prefix = Q.read_json(self.queue.path/'events'/f"{self.state['queue_offset']:012d}.json")
                need(prefix['content_sha256'] == self.state['queue_head_sha256'], 'QUEUE_HEAD_DRIFT')
            else:
                self.state = dict(schema=SCHEMA, binding_sha256=self.binding, worker='search-' + self.binding[:16],
                    queue_calls=0, queue_call=None, active=None, completed=[], peak_inflight=0,
                    elapsed_seconds=0., queue_wait_seconds=0., segment_count=0)
                self._save()
            if self.state['active']:
                self.active = sealed_read(self.directory / 'rounds' / self.state['active'] / 'ROUND.json')
            start = self.clock()
            self.deadline = start + self.limits.seconds
            self.state['segment_count'] += 1
            self.state['segment_active'] = True
            if isinstance(self.executor, SubprocessExecutor):
                self.state['allocation_id'] = os.environ['SLURM_JOB_ID']
                self.state['requires_allocation_cleanup'] = False
            self._save()
            begun = 0
            reason = 'STATE_CAP'
            try:
                while self.clock()-start < self.limits.seconds and not self.stop_requested:
                    if self.active is None:
                        if begun >= self.limits.max_states:
                            break
                        before = self.clock()
                        available = self._start()
                        self.state['queue_wait_seconds'] += self.clock()-before
                        if not available:
                            reason = 'QUEUE_EMPTY'; break
                        begun += 1
                    self._advance()
                    if self.active is not None and self.inflight:
                        time.sleep(self.limits.poll_seconds)
                else:
                    reason = 'SIGNAL_STOP' if self.stop_requested else 'SEGMENT_DEADLINE'
            except (WorkerError, Q.QueueError) as exc:
                # Evidence/identity errors stop this edge, not other workers.
                if self.active is not None:
                    self.active['error'] = str(exc)
                reason = 'LOCAL_PREFLIGHT_OR_CUSTODY_ERROR'
                raise
            finally:
                deadline = self.clock() + self.limits.drain_seconds
                while self.active is not None and self.inflight and self.clock() < deadline:
                    self._pump(launch=False)
                    if self.inflight:
                        time.sleep(self.limits.poll_seconds)
                self.executor.shutdown()
                if getattr(self.executor, 'cleanup_uncertain', False):
                    self.state['requires_allocation_cleanup'] = True
                self.inflight.clear()
                if self.active is not None:
                    self._pump(launch=False)  # collect durable spools; censor dead charged starts
                self.state['elapsed_seconds'] += self.clock()-start
                self.state['last_stop_reason'] = ('NATIVE_WORKER_LOST_ALLOCATION_CLEANUP_REQUIRED'
                    if self.state.get('requires_allocation_cleanup') else reason)
                self.state['segment_active'] = False
                self._save()
                self.deadline = None
            return self.report()

    def report(self) -> dict:
        rounds = [sealed_read(p) for p in sorted((self.directory / 'rounds').glob('*/ROUND.json'))]
        attempts, unique, jobs = [], 0, []
        for row in rounds:
            if row['proposal']:
                population = row['proposal']['states'][0]
                attempts.extend(population['attempts']); unique += len(population['candidates'])
            jobs.extend(row['jobs'].values())
        wall = self.state['elapsed_seconds']
        rate = lambda n: n / wall if wall > 0 else None
        completed = [j for j in jobs if j['status'] == 'ARCHIVED']
        exact = [j for j in completed if j['facts'] and j['facts']['native'] and j['facts']['complete'] and j['facts']['fresh_replay']]
        confirmations = sum(bool((r.get('confirmation') or {}).get('statistical_support')) for r in rounds)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != 'darwin' else 1)
        return A.seal(dict(schema=SCHEMA + '-metrics', evidence_kind=self.bridge.evidence_kind,
            elapsed_seconds=wall, proposal_attempts=len(attempts), proposals_per_second=rate(len(attempts)),
            candidate_family_counts=dict(Counter(a['proposal_family'] for a in attempts)),
            proposal_status_counts=dict(Counter(a['status'] for a in attempts)), unique_legal_candidates=unique,
            exact_evaluation_attempts=sum(j['status'] != 'READY' and j['status'] != 'PREPARATION_FAILED' for j in jobs),
            archived_outcomes=len(completed), outcome_status_counts=dict(Counter(j['archive_status'] for j in completed)),
            native_complete_replayed_pairs=len(exact), exact_evaluations_per_second=rate(len(exact)),
            replay_verified_positive_pairs=sum(j['facts']['wdl_delta'] > 0 for j in exact),
            confirmed_wdl_improvements=confirmations, confirmed_wdl_improvements_per_elapsed_hour=rate(confirmations * 3600),
            gpu_allocation_seconds=None, gpu_hour_rates=None, gpu_accounting='NOT_MEASURED_USE_ALLOCATION_ACCOUNTING',
            queue_wait_seconds=self.state['queue_wait_seconds'], peak_inflight=self.state['peak_inflight'],
            coordinator_peak_rss_bytes=peak, largest_worker_peak_rss_bytes=max([j['peak_rss_bytes'] or 0 for j in jobs] or [0]),
            peak_memory_scope='PER_PROCESS_HIGH_WATER_NOT_SIMULTANEOUS_NODE_SUM',
            measured_worker_cpu_seconds=math.fsum(j['cpu_seconds'] or 0. for j in jobs),
            worker_cpu_scope='EXACT_COORDINATOR_PROCESS_ONLY_EXCLUDES_NATIVE_GRANDCHILDREN',
            queue_offset=self.state['queue_offset'], archive_offset=self.state['archive_offset'],
            archive_head_sha256=self.state['archive_head_sha256'], completed_states=len(self.state['completed']),
            active_state=self.state['active'], stop_reason=self.state.get('last_stop_reason'),
            teacher_export='NOT_ADMITTED', actor_training_admissible=False,
            requires_allocation_cleanup=self.state.get('requires_allocation_cleanup', False),
            limitations=['NO_POLICY_STRENGTH_CLAIM', 'UNKNOWN_INTERRUPTED_WORK_IS_NOT_A_ZERO_PAYOFF',
                         'ONE_REACHED_STATE_AT_A_TIME', 'HARD_KILLED_SEGMENT_TIME_NOT_RECONSTRUCTED']))


class NativeBridge:
    """Frozen full-cut manifest, deliberately separate from actor-visible rows.

    Manifest schema:
      {schema: 'farm-v0-search-worker-v1-inputs', states: {QUEUE_FINGERPRINT: {
        discovery: [{snapshot: artifact_ref, source_group_id: str,
                     source_family_id: str, control: exact_suffix}],
        confirmation: [same cell shape], known_suffixes: [[raw_ids...]],
        admission: {alpha_pair: float, min_effect: float,
                    independence_attested: bool, compatible_contexts_attested: bool,
                    frozen_before_confirmation: bool}}}}

    The first discovery cut must round-trip to the EXACT queued snapshot, not
    just equal public facts. Additional cuts must have equal actor projections,
    and preserve their own full worlds, controllers, opponent states and RNG.
    Source grouping/independence are explicit producer attestations, not a claim
    that matching hashes prove real-world independence or prior non-exposure.
    """
    evidence_kind = 'NATIVE_GAME'

    def __init__(self, manifest: dict, *, root: Path, timeout_seconds: float = 30.):
        Q._fields(manifest, ('schema', 'states'))
        need(manifest['schema'] == SCHEMA + '-inputs' and type(manifest['states']) is dict, 'EXACT_INPUT_MANIFEST')
        need(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds)
             and .01 <= timeout_seconds <= 3600, 'EXACT_TIMEOUT')
        self.manifest, self.root, self.timeout = copy.deepcopy(manifest), Path(root).resolve(), float(timeout_seconds)
        self.identity = {'manifest_sha256': A.digest(manifest), 'timeout_seconds': self.timeout,
                         'execution': 'NATIVE_CAPTURED_SUFFIX_KEEP_TO_TERMINAL',
                         'worker_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self._validated: set[str] = set()

    def spec(self, candidate):
        cid = candidate['fingerprint']
        need(cid in self.manifest['states'], 'MISSING_MATCHED_FULL_CUT_FOR_QUEUE_STATE: ' + cid)
        spec = self.manifest['states'][cid]
        Q._fields(spec, ('discovery', 'confirmation', 'known_suffixes', 'admission'))
        need(type(spec['discovery']) is list and 1 <= len(spec['discovery']) <= 64, 'DISCOVERY_CELL_CAP')
        need(type(spec['confirmation']) is list and len(spec['confirmation']) <= 4096, 'CONFIRMATION_CELL_CAP')
        return spec

    def cells(self, candidate, phase):
        need(phase in ('discovery', 'confirmation'), 'EVALUATION_PHASE')
        return copy.deepcopy(self.spec(candidate)[phase])

    def admission(self, candidate):
        return copy.deepcopy(self.spec(candidate)['admission'])

    def _cut(self, cell):
        from farm_neat.rollout import v0_regret_dawn as D
        Q._fields(cell, ('snapshot', 'source_group_id', 'source_family_id', 'control'))
        for name in ('source_group_id', 'source_family_id'):
            Q._text(cell[name], name)
        _, live = A._captured_start(cell['snapshot'], evidence_root=self.root,
                                     source_group_id=cell['source_group_id'])
        own, other = live.actors[live.seat], live.actors[1-live.seat]
        try:
            continuation = D.make_continuation(own, other, seat=live.seat)
            cut = D.capture_dawn(live.environment, own, other, continuation=continuation)
        except (ValueError, TypeError) as exc:
            raise WorkerError('FULL_CUT_NOT_IN_QUEUE_CODEC_INTERSECTION: ' + str(exc)) from exc
        return cut

    def row(self, candidate):
        spec = self.spec(candidate)
        first = spec['discovery'][0]
        cut = self._cut(first)
        # Draft is non-active, trainer-only data; retain it exactly in the roundtrip.
        if candidate['snapshot']['draft'] is not None:
            from farm_neat.rollout import v0_regret_dawn as D
            cut = D.P.bind({**cut, 'draft': copy.deepcopy(candidate['snapshot']['draft'])})
        need(Q.wire(cut) == Q.wire(candidate['snapshot']), 'QUEUE_FULL_CAPTURE_MISMATCH')
        need(first['source_group_id'] == candidate['origin']['source_group_id']
             and first['source_family_id'] == candidate['origin']['source_family_id'], 'QUEUE_SOURCE_GROUP_MISMATCH')
        visible = Q.actor_inputs(candidate)
        return dict(queue_id=candidate['fingerprint'], snapshot_sha256=candidate['snapshot']['content_sha256'],
                    split='discovery', actor_context=proposal_context(visible),
                    incumbent_actions=copy.deepcopy(first['control']['epochs'][0]['raw_actions']),
                    known_suffixes=copy.deepcopy(spec['known_suffixes']))

    def validate(self, candidate, registration, discovery, confirmation):
        need(candidate['origin']['evidence_kind'] == 'NATIVE_GAME', 'FIXTURE_QUEUE_IS_NOT_PRODUCTION')
        need(registration['checkpoint_sha256'] == candidate['identity']['checkpoint_sha256'], 'PROPOSAL_CHECKPOINT_JOIN')
        groups = [c['source_group_id'] for c in discovery + confirmation]
        need(len(groups) == len(set(groups)), 'DUPLICATE_OR_OVERLAPPING_WORLD_GROUP')
        d_families = {c['source_family_id'] for c in discovery}
        c_families = {c['source_family_id'] for c in confirmation}
        need(not d_families & c_families, 'PROTECTED_FAMILY_OVERLAP')
        if registration['gate'] == 'DISCOVERY_POSITIVE':
            need(set(registration['discovery_groups']) == {c['source_group_id'] for c in discovery}, 'DISCOVERY_GRID_JOIN')
        need(not set(registration['protected_groups']) & {c['source_group_id'] for c in discovery}, 'PROTECTED_DISCOVERY_GROUP')
        admission = self.admission(candidate)
        Q._fields(admission, ('alpha_pair', 'min_effect', 'independence_attested',
                              'compatible_contexts_attested', 'frozen_before_confirmation'))
        confirmation_summary([], expected=[], alpha=admission['alpha_pair'], min_effect=admission['min_effect'])
        if confirmation:
            need(all(admission[k] is True for k in ('independence_attested', 'compatible_contexts_attested',
                                                    'frozen_before_confirmation')), 'CONFIRMATION_ATTESTATIONS_REQUIRED')
        original = Q.actor_inputs(candidate)
        cuts = []
        for cell in discovery + confirmation:
            cut = self._cut(cell)
            from farm_neat.rollout import v0_regret_dawn as D
            need(D.recovery_actor_context(cut) == original, 'INCOMPATIBLE_DAWN_ACTOR_CONTEXT')
            cuts.append(cell['snapshot']['sha256'])
            # Validate complete control suffix at its own exact start. Later
            # epochs are rechecked by the existing executor, never repaired.
            A.make_suffix_request(snapshot_ref=cell['snapshot'], control=cell['control'], challenger=cell['control'],
                source_group_id=cell['source_group_id'], policy_sha256=registration['checkpoint_sha256'],
                protocol_sha256=A.digest(self.manifest), split='discovery', round_id='worker-preflight',
                proposal_family='control-preflight', timeout_seconds=self.timeout, evidence_root=self.root)
        need(len(cuts) == len(set(cuts)), 'REUSED_EXACT_CUT_AS_INDEPENDENT_WORLD')
        proposals = sum(registration['quotas'].values()) * sum(registration['stage_multipliers'])
        starts = 4 * (proposals * len(discovery) + len(confirmation))
        required = dict(candidate_attempts=proposals, native_starts=starts,
            cpu_milliseconds=math.ceil(1000 * starts * (self.timeout+10)),
            gpu_milliseconds=math.ceil(1000 * len(registration['stage_multipliers']) * registration['callback_timeout_seconds']))
        need(all(candidate['budget'][k] >= v for k, v in required.items()),
             'QUEUE_ALLOCATION_TOO_SMALL: required=' + json.dumps(required, sort_keys=True))
        self._validated.add(candidate['fingerprint'])

    def prepare(self, candidate, proposal, cell, phase, directory, evaluation_id):
        from farm_neat.rollout import v0_regret_dawn as D
        cut = self._cut(cell)
        replayed = D.propose_dawn(cut, list(proposal['raw_actions']), continuation=cut['continuation'])
        need(replayed['canonical_plan'] == proposal['canonical_plan'], 'PROPOSAL_EXACT_PLAN_MISMATCH')
        challenger = dict(kind='V0', epochs=[dict(step=cut['step'], raw_actions=replayed['raw_actions'],
                            canonical_plan=replayed['canonical_plan'])], tail='KEEP_TO_TERMINAL')
        request = A.make_suffix_request(snapshot_ref=cell['snapshot'], control=cell['control'], challenger=challenger,
            source_group_id=cell['source_group_id'], policy_sha256=candidate['identity']['checkpoint_sha256'],
            protocol_sha256=A.digest(self.manifest), split=phase, round_id=self.identity['manifest_sha256'],
            proposal_family=proposal['proposal_family'], timeout_seconds=self.timeout, evidence_root=self.root)
        path = directory / 'REQUEST.json'
        publish(path, request)
        ref = A.artifact_ref(path, root=self.root)
        exact = A.candidate_from_suffix_request(request, request_ref=ref, evidence_root=self.root)
        return dict(candidate=exact, request=ref,
                    charged=dict(candidate_attempts=1, native_starts=4, cpu_seconds=4*(self.timeout+10)))


def preflight_paths(args) -> tuple[Q.SearchQueue, NativeBridge]:
    """Local inputs only: no checkpoint load, GPU allocation or queue mutation."""
    for label, path in (('QUEUE_IDENTITY', args.identity), ('EXACT_INPUT_MANIFEST', args.inputs),
                        ('CHECKPOINT_COMMIT', Path(args.checkpoint)/'COMMIT.json'),
                        ('CHECKPOINT_STATE', Path(args.checkpoint)/'state.pt')):
        need(A.safe_path(path).is_file(), 'MISSING_' + label + ': ' + str(path))
    need(Path(args.queue).is_dir(), 'MISSING_QUEUE: ' + str(args.queue))
    Q._hash(args.commit_sha256, 'commit_sha256')
    need(hashlib.sha256((Path(args.checkpoint)/'COMMIT.json').read_bytes()).hexdigest() == args.commit_sha256,
         'CHECKPOINT_COMMIT_HASH_MISMATCH')
    adapter = ROOT / 'src/farm_neat/policy/v0_search_batch.py'
    need(adapter.is_file(), 'MISSING_ADAPTER: farm_neat.policy.v0_search_batch.make_tensor_decoder')
    from farm_neat.policy.v0_search_batch import make_tensor_decoder
    need(callable(make_tensor_decoder), 'MISSING_WHOLE_SESSION_TENSOR_DECODER')
    queue = Q.SearchQueue(args.queue, read(args.identity))
    queue.inspect()
    need(queue.identity['checkpoint_sha256'] == args.commit_sha256, 'QUEUE_CHECKPOINT_COMMIT_JOIN')
    bridge = NativeBridge(read(args.inputs), root=ROOT, timeout_seconds=args.branch_timeout)
    for path in (Path(args.work_dir), Path(args.archive)):
        need(A.safe_path(path).resolve().is_relative_to(ROOT / 'work'), 'OUTPUT_MUST_BE_IN_PROJECT_WORK')
    return queue, bridge


def require_previous_allocation_closed(state: dict) -> None:
    """A killed exact parent can leave detached archive arms inside its SLURM job.

    Lifetime locks cover live exact parents, not reparented native grandchildren.
    A hard-interrupted/uncertain native segment therefore cannot resume in the
    same allocation or while the former allocation is live/unverifiable. Do not
    silently infer remote cleanup from a missing process or a quiet log.
    """
    if not (state.get('segment_active') or state.get('requires_allocation_cleanup')):
        return
    old = state.get('allocation_id')
    need(type(old) is str and old.isdigit(), 'MISSING_PREVIOUS_ALLOCATION_ID')
    need(old != os.environ.get('SLURM_JOB_ID'), 'NEW_ALLOCATION_REQUIRED_AFTER_NATIVE_INTERRUPTION')
    try:
        text = subprocess.run(['sacct', '-j', old, '--noheader', '--parsable2',
            '--format=JobIDRaw,State'], capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkerError('PREVIOUS_ALLOCATION_CLEANUP_UNVERIFIED') from exc
    rows = [line.split('|') for line in text.splitlines()]
    matching = [row[1].split()[0].rstrip('+') for row in rows if len(row) >= 2 and row[0] == old and row[1]]
    # NODE_FAIL/BOOT_FAIL are deliberately excluded: unreachable is not cleaned.
    need(len(matching) == 1 and matching[0] in
         {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'DEADLINE'},
         'PREVIOUS_ALLOCATION_NOT_TERMINAL_OR_CLEANUP_UNKNOWN')


def require_native_allocation(workers: int, *, check_gpu=True):
    """No desktop inference, CPU production fallback, or another environment."""
    need(bool(os.environ.get('SLURM_JOB_ID')), 'SLURM_ALLOCATION_REQUIRED')
    need(os.environ.get('PYTHONHASHSEED') == '0', 'PYTHONHASHSEED_0_REQUIRED')
    need(os.environ.get('SLURM_JOB_NUM_NODES', '1') == '1', 'SINGLE_NODE_ONLY')
    need(int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) >= workers + 1, 'INSUFFICIENT_ALLOCATED_CPUS')
    need(bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'ALLOCATED_CUDA_DEVICE_REQUIRED')
    try:
        report = subprocess.run(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'],
                                capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkerError('CANNOT_VERIFY_SLURM_ALLOCATION') from exc
    fields = dict(word.split('=', 1) for word in report.split() if '=' in word)
    need(fields.get('JobState') == 'RUNNING' and fields.get('NumNodes') == '1', 'ACTIVE_SINGLE_NODE_ALLOCATION_REQUIRED')
    need(('gres/gpu:4090=1' in fields.get('AllocTRES', '').split(',')
          or 'gres/gpu:4090:1' in fields.get('TresPerNode', '').split(','))
         and not any(v.startswith('gres/gpu=') and v != 'gres/gpu=1' for v in fields.get('AllocTRES', '').split(',')),
         'SINGLE_4090_ALLOCATION_REQUIRED')
    # The ordinary storage mutex must be held by an ancestor launcher, not just
    # another job elsewhere. runtime.py run can be an intervening parent.
    ancestors, pid = [], os.getppid()
    for _ in range(16):
        try:
            cmd = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
            ancestors.append(cmd)
            pid = int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if pid <= 1:
            break
    need(any(b'storage' in cmd and b'run' in cmd for cmd in ancestors), 'ORDINARY_STORAGE_RUN_REQUIRED')
    lock = ROOT / 'work/cache/storage/COMMAND.lock'
    need(lock.is_file() and job_busy(lock), 'STORAGE_ADMISSION_MUTEX_NOT_HELD')
    if check_gpu:
        from farm_neat.training.v0_bc import require_production_runtime
        require_production_runtime()
        import torch
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'ONE_CUDA_DEVICE_REQUIRED')
        need('4090' in torch.cuda.get_device_name(0), 'RTX_4090_REQUIRED')
    return fields


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'run'):
        sub = commands.add_parser(name)
        for flag in ('queue', 'identity', 'inputs', 'checkpoint', 'commit-sha256', 'work-dir', 'archive'):
            sub.add_argument('--' + flag, required=True)
        sub.add_argument('--seconds', type=float, default=600.)
        sub.add_argument('--workers', type=int, default=6)
        sub.add_argument('--max-states', type=int, default=16)
        sub.add_argument('--branch-timeout', type=float, default=30.)
        sub.add_argument('--drain-seconds', type=float, default=20.)
        sub.add_argument('--seed', type=int, default=39)
        sub.add_argument('--registration', help='Existing frozen progressive registration; default creates 1,2 widening.')
    worker = commands.add_parser('_evaluate', help=argparse.SUPPRESS)
    worker.add_argument('--job', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == '_evaluate':
            need('FARM_SEARCH_GUARD_FD' in os.environ, 'OWNED_EXACT_WORKER_REQUIRED')
            os.fstat(int(os.environ['FARM_SEARCH_GUARD_FD']))
            need(bool(os.environ.get('SLURM_JOB_ID')), 'SLURM_ALLOCATION_REQUIRED')
            execute_spooled_job(args.job, evaluate_native)
            return 0
        limits = Limits(workers=args.workers, max_states=args.max_states, seconds=args.seconds,
                        drain_seconds=args.drain_seconds,
                        lease_ms=min(Q.MAX_TTL_MS, max(3_600_000, math.ceil((args.seconds+args.drain_seconds+60)*1000)))).validate()
        queue, bridge = preflight_paths(args)
        registration = read(args.registration) if args.registration else S.register_batch_proposals(
            quotas={f: (2 if f == 'policy_sample' else 1 if f in ('random_legal_control', 'incumbent_mutation') else 0)
                    for f in P.BATCH_FAMILIES}, seed=args.seed, stage_multipliers=[1, 2], gate='ANY_REACHABLE',
            checkpoint_sha256=args.commit_sha256,
            tensor_backend_sha256=hashlib.sha256((ROOT/'src/farm_neat/policy/v0_search_batch.py').read_bytes()).hexdigest(),
            provider_sha256=A.digest('no-external-provider'), callback_timeout_seconds=30.)
        S._batch_registration(registration)
        need(registration['tensor_backend_sha256'] == hashlib.sha256(
            (ROOT/'src/farm_neat/policy/v0_search_batch.py').read_bytes()).hexdigest(),
            'TENSOR_DECODER_SOURCE_JOIN')
        need(registration['quotas']['solver_provider'] == 0, 'EXTERNAL_PROVIDER_NOT_CONFIGURED')
        for cid, row in queue.inspect()['candidates'].items():
            if row['status'] in ('READY', 'LEASED', 'STARTED'):
                candidate = queue.get_candidate(cid)
                bridge.row(candidate)
                bridge.validate(candidate, registration, bridge.cells(candidate, 'discovery'), bridge.cells(candidate, 'confirmation'))
        if args.command == 'preflight':
            print(json.dumps({'status': 'LOCAL_INPUTS_READY_NOT_RUNTIME_OR_GPU_VALIDATED',
                              'bridge': bridge.identity, 'registration_sha256': registration['content_sha256']}))
            return 0
        allocation = require_native_allocation(limits.workers)
        from farm_neat.evaluation.v0_policy import load_committed_scorer
        from farm_neat.policy.v0_search_batch import make_tensor_decoder
        committed = load_committed_scorer(args.checkpoint, args.commit_sha256, device='cuda:0', production=True)
        archive = A.Archive(args.archive, evidence_root=ROOT)
        coordinator = SearchWorker(queue=queue, archive=archive, directory=args.work_dir,
            registration=registration, bridge=bridge, tensor_decoder=make_tensor_decoder(committed.scorer),
            executor=SubprocessExecutor(), limits=limits)
        old = {sig: signal.signal(sig, lambda *_: coordinator.stop()) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            result = coordinator.run()
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
        result['slurm_allocation'] = {k: allocation.get(k) for k in ('JobId', 'NodeList', 'AllocTRES', 'StartTime')}
        publish(Path(args.work_dir)/f'METRICS-{coordinator.state["segment_count"]:04d}.json', A.seal(result))
        print(json.dumps(A.seal(result), sort_keys=True))
        return 2 if coordinator.state.get('requires_allocation_cleanup') else 0
    except (WorkerError, Q.QueueError, A.ArchiveError, ValueError, OSError) as exc:
        print(json.dumps({'status': 'LOCAL_PREFLIGHT_OR_CUSTODY_ERROR', 'error': str(exc)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
