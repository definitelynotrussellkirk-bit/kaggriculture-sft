"""Bounded visible-information V0 challenger pools; no scoring, games or labels.

``propose(context, learner, *, budget, seed, archive)`` consumes the closed actor
context and direct learner proposal from ``v0_regret_records.validate_pair``.
It revalidates them. ``archive`` is an ordered list of pair records; both arms
are reusable paths, regardless of their outcomes. Source/context/checkpoint
joins and the existing full record validator precede reuse. Validation is not
an authenticity claim. Only valid substitutions of evaluator-only evidence are
invariant; corrupt records are still rejected. No evaluator field selects a path.

Budget keys: max_candidates (1..32, default 32), max_path_edits (1..16,
default 4), max_archive_records (0..32, default 8). EVERY attempted path, including
invalid/duplicate paths, consumes a candidate slot. There are no retry-until-valid
loops. Ordinary sources have no wall-clock-dependent decisions; the optional
external arm has explicit transport/validation deadlines. The edit sub-budget limits synthesized
paths, not the supplied learner/KEEP/archive controls. All native economic
families remain in the random legal support; a finite pool is not exhaustive.

The oracle's one-key insertion/replacement/deletion and strict sequence contract
are adapted to arbitrary native Prefix cursors. Unlike preloaded Plan search,
every survivor is replayed through native prefix-local menus. Theorem proposals
sample successor classes at the ACTUAL prefix, using their canonical raw member;
raw-uniform controls retain all native aliases. Cash objectives in the old tools
are untouched and never called. screen_plan cannot certify placement from the
record's projection-only facts: its adapter returns UNKNOWN, never infeasible.

Candidate records are UNSELECTED geometry, not complete pair/challenger records.
They intentionally lack selection/confirmation receipts and all behavior logps.
The untouched ``learner_draw`` alone retains the genuine direct-draw likelihood.
The paired evaluator consumes candidate raw_actions/trace/Plan and the pool's
actor_context, then supplies selection and evaluation evidence downstream.
``full_continuation_signature`` binds the exact starting compiler memory, final
decoder cursor and native HOLD bindings BEFORE publication. It is NOT a compiled
post-publication snapshot or evidence of executable realization.

CLI (one fully validated pair JSON, optional list of archive pair JSON records)::

    PYTHONPATH=src python3 -B -m farm_neat.training.teachers.v0_regret_proposals \
        --input PAIR.json --output POOL.json --max-candidates 32 --seed 37

No inputs/receipts are overwritten. Intake should replace this module's planned
entrypoint registration with this command; this reviewer does not edit registries.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
import selectors
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
import random

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.evaluation.v0_continuation import V0Continuation
from farm_neat.policy import v0_menu_fast as fast
from farm_neat.training import v0_regret_records as R
from farm_neat.training import v0_targets as T
from farm_neat.training.teachers import v0_oracle as oracle
from farm_neat.training.teachers import v0_plan_search as plan_search

SCHEMA = 'farm-v0-regret-proposal-pool-v1'
CANDIDATE_SCHEMA = 'farm-v0-regret-unselected-candidate-v1'
DEFAULT_BUDGET = dict(max_candidates=32, max_path_edits=4, max_archive_records=8)
PARENTS = {
    'prefix_classes': '10de976c1c73a81763f88f7dc5d714f66415af41ba43816be76db1a0cdbab948',
    'canonical_keys': 'db53d65ae335bfce005a1e18d8429960c78bb013ce4da095cab47e28779f2c59',
    'replay': '6b4d15f4332d6763a446296f1d086a3fcd1cdbad2d1ef21e011fe9eff38ff8c6',
    'roundtrip_boundary': '1fbc98d99acd7224706af8018b0965ec825920f7bb603344a7bc3d9319696ea0',
}
SOURCE_PATHS = ('src/farm_neat/training/teachers/v0_regret_proposals.py',
                'src/farm_neat/training/teachers/v0_oracle.py',
                'src/farm_neat/training/teachers/v0_plan_search.py')
SOURCES = ('learner_mutation', 'keep_mutation', 'theorem_conditioned', 'random_legal_control')


def _budget(value: dict) -> dict:
    R._fields(value, (), '$.budget', optional=DEFAULT_BUDGET)
    result = DEFAULT_BUDGET | value
    for key, maximum in (('max_candidates', 32), ('max_path_edits', 16), ('max_archive_records', 32)):
        R._int(result[key], '$.budget.' + key, 0 if key == 'max_archive_records' else 1, maximum)
    return result


def _syntax_screen(prefix, actions) -> dict:
    """Only apply proved syntactic cuts; other feasibility stays UNKNOWN."""
    conditions = dict(canonical_key_order_applies=True, native_edit_cap_applies=True,
                      prefix_terminated=prefix.terminated)
    reason = None
    if type(actions) is not list or not actions:
        reason = 'MISSING_STOP'
    elif any(type(a) is not int or not 0 <= a < V.N_ACTIONS for a in actions):
        reason = 'INVALID_ACTION'
    elif prefix.terminated:
        reason = 'AFTER_STOP'
    elif actions[-1] != V.STOP_ID:
        reason = 'MISSING_STOP'
    elif V.STOP_ID in actions[:-1]:
        reason = 'AFTER_STOP'
    elif len(actions) - 1 + prefix.edit_count > V.MAX_EDITS_PER_EPOCH:
        reason = 'EDIT_CAP'
    else:
        keys = [prefix.last_key] + [V.plan_key(a) for a in actions[:-1]]
        if any(a >= b for a, b in zip(keys, keys[1:])):
            reason = 'NONCANONICAL_KEY'
        else:
            # Reuse the oracle's fixed-lattice/sequence contract, not its cash selector.
            oracle._sequence(actions)
    return dict(status='INVALID' if reason else 'UNKNOWN', conditions=conditions,
                reason_codes=[reason or 'NATIVE_PREFIX_REPLAY_REQUIRED'],
                evidence=PARENTS['canonical_keys'])


def screen_actions(context: dict, actions: list[int]) -> dict:
    """Cheap conditional key/cap screen; UNKNOWN must proceed to native replay."""
    _state, prefix, _options = R._actor(context)
    return _syntax_screen(prefix, actions)


def _plan_screen(plan) -> dict:
    # Adapt plan-search's strict Plan input, without inventing a native observation
    # from occupied counts, or calling its preload route a decoder round trip.
    # The records contract admits counts beyond the legacy preload screen's
    # domain. A failed legacy precondition is not proof of native infeasibility.
    try:
        plan_search.strict_plan(plan.canonical())
        legacy_domain = True
    except ValueError:
        legacy_domain = False
    return dict(status='UNKNOWN', plan_sha256=plan.digest(), physical_games=0,
                reason_codes=['EXACT_PLACEMENT_GEOMETRY_NOT_IN_ACTOR_CONTEXT'],
                adapter='v0_plan_search.screen_plan',
                conditions=dict(exact_geometry_available=False, legacy_preload_domain=legacy_domain),
                scope='NO_COMPILE_OR_VALUE_CERTIFICATE')


def _menu(state, prefix, options, cache):
    key = T._json(prefix.payload())
    if key not in cache:
        cache[key] = fast.build_menu(state, prefix, projection_kwargs=options, backend='fast')
    return cache[key]


def _trace(state, prefix, options, cache, actions) -> dict:
    """Native target geometry; never advance from a supplied teacher prefix."""
    rows = []
    for index, action in enumerate(actions):
        menu = _menu(state, prefix, options, cache)
        if action not in menu.legal_ids:
            return dict(status='ILLEGAL', failure_index=index, reason_codes=['ILLEGAL_PREFIX_ACTION'],
                        prefix_class_trace=rows, reached_plan=prefix.plan.canonical())
        position = menu.legal_ids.index(action)
        cls = next(c for c in menu.classes if position in c.indices)
        forced = prefix.edit_count == V.MAX_EDITS_PER_EPOCH
        rows.append(dict(schema=R.PREFIX_SCHEMA, prefix=prefix.payload(), support=list(menu.legal_ids),
                         phi_sha256=R.canonical_sha256(menu.phi), raw_id=action, class_id=cls.class_id,
                         class_members=[menu.legal_ids[i] for i in cls.indices],
                         continuation=cls.successor.payload(), forced_stop=forced,
                         termination=('CAP' if forced else 'STOP') if action == 0 else 'CONTINUE'))
        prefix = cls.successor
    return dict(status='REACHABLE', prefix_class_trace=rows, canonical_plan=prefix.plan.canonical(),
                continuation_signature=T.continuation_signature(prefix.plan, prefix.last_key,
                                                                 prefix.edit_count, prefix.terminated),
                final=prefix)


def _random_path(state, prefix, options, cache, rng, limit, *, classes=False) -> list[int]:
    actions = []
    while len(actions) < limit and not prefix.terminated:
        menu = _menu(state, prefix, options, cache)
        if classes:
            # Parent theorem applies only to THIS menu's exact canonical prefix.
            cls = rng.choice(menu.classes)
            action = min(menu.legal_ids[i] for i in cls.indices)
        else:
            action = rng.choice(menu.legal_ids)
            index = menu.legal_ids.index(action)
            cls = next(c for c in menu.classes if index in c.indices)
        actions.append(action)
        prefix = cls.successor
    if not prefix.terminated:
        actions.append(V.STOP_ID)  # voluntary budget STOP unless native count==16
    return actions


def _mutate(actions: list[int], rng, trial: int) -> tuple[list[int], dict]:
    """oracle.neighborhood's one-key insertion/replacement/deletion, bounded once.

    We cannot invoke neighborhood directly at a resumed Prefix: its API resets
    last_key/edit_count. This lift preserves the original input and delegates all
    changed-suffix legality to _trace; it never accepts an entry-mask shortcut.
    """
    changed = {V.plan_key(a): a for a in actions if a != 0}
    if trial % 5 == 4 and changed:
        key = rng.choice(sorted(changed))
        del changed[key]
        operation = dict(kind='DELETE', key=key, action=None)
    else:
        family = plan_search.FAMILIES[trial % len(plan_search.FAMILIES)]
        if family == 'PROD':
            action = V.encode(family, rng.randrange(V.N_MODULES), rng.randrange(len(V.DELTAS)))
        elif family == 'SERVICE':
            # Existing production keys get a useful service challenge; no family
            # is suppressed when none exist (the resulting masked attempt stays).
            modules = [V.decode(a)[1] for a in changed.values() if V.decode(a)[0] == 'PROD']
            action = V.encode(family, rng.choice(modules) if modules else rng.randrange(V.N_MODULES),
                              rng.randrange(len(V.REGIMES)))
        elif family == 'LAND':
            action = V.encode(family, rng.randrange(len(V.LAND_TIMINGS)))
        else:
            action = V.encode(family, rng.randrange(len(V.PRODUCTS)), rng.randrange(len(V.PRIMITIVES)),
                              rng.randrange(len(V.HORIZONS)), rng.randrange(V.RANKS))
        key = V.plan_key(action)
        operation = dict(kind='REPLACE' if key in changed else 'INSERT', key=key, action=action)
        changed[key] = action
    return [changed[k] for k in sorted(changed)] + [0], operation


def _archive_paths(archive, context_key, pins, checkpoint):
    """Stable-index, unranked import. Do not propagate hashes of hidden evidence."""
    for index, document in enumerate(archive):
        reason = None
        identity = document.get('identity', {}) if type(document) is dict else {}
        if type(document) is not dict or type(identity) is not dict:
            reason = 'INVALID_ARCHIVE_IDENTITY'
        elif identity.get('source_pins') != pins:
            reason = 'STALE_ARCHIVE_SOURCE'
        elif identity.get('actor_context_sha256') != context_key:
            reason = 'STALE_ARCHIVE_CONTEXT'
        elif identity.get('parent_checkpoint_sha256') != checkpoint:
            reason = 'STALE_ARCHIVE_CHECKPOINT'
        elif identity.get('split') in ('monitor', 'final'):
            reason = 'PROTECTED_ARCHIVE_SPLIT'
        else:
            try:
                document = R.validate_pair(document)
            except R.RecordError as exc:
                reason = 'INVALID_ARCHIVE_' + exc.code
        if reason:
            yield None, None, dict(archive_index=index), reason
            continue
        for arm in ('learner', 'challenger'):
            row = document[arm]
            geometry = {k: row[k] for k in ('raw_actions', 'canonical_plan', 'continuation_signature')}
            yield row['raw_actions'], row['canonical_plan'], dict(
                archive_index=index, arm=arm, geometry_sha256=R.canonical_sha256(geometry)), None


def propose(context: dict, learner: dict, *, budget: dict, seed: int, archive: list[dict],
            provider: dict | None = None, provider_budget: dict | None = None) -> dict:
    """Capped auditable pool; ordinary RNG is deterministic, provider timing is not."""
    budget = _budget(budget)
    R._int(seed, '$.seed', 0, 2**64 - 1)
    R._list(archive, '$.archive')
    R._require(len(archive) <= budget['max_archive_records'], 'ARCHIVE_BUDGET', '$.archive')
    R._safe(learner)
    R._safe(archive)
    state, prefix, options = R._actor(context)
    R._require(not prefix.terminated, 'AFTER_STOP', '$.actor_context.prefix')
    R._require(type(learner) is dict, 'INVALID_OBJECT', '$.learner')
    provenance = learner.get('sampling_provenance', {})
    checkpoint = provenance.get('checkpoint_sha256') if type(provenance) is dict else None
    R._hash(checkpoint, '$.learner.checkpoint_sha256')
    cache = {}
    R._proposal(learner, {'parent_checkpoint_sha256': checkpoint}, prefix, state, options, cache, direct=True)
    context_key = R.canonical_sha256(context)
    pins = R.source_pins()
    generator_pins = {p: hashlib.sha256((R.ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    rng = random.Random(seed)
    attempts, candidates, seen = [], [], {}
    counts = dict(proposed=0, legal=0, unique=0, reachable=0, retained=0, physical_games=0, simulator_steps=0)
    archive_visited = set()
    archive_arms_visited = set()

    def consider(actions, family, provenance, wanted=None, exclusion=None, compiler_check=None):
        if len(attempts) >= budget['max_candidates']:
            return False
        number = len(attempts)
        attempt = dict(attempt_id=number, proposal_family=family, provenance=copy.deepcopy(provenance),
                       raw_actions=copy.deepcopy(actions), status='REJECTED', reason_codes=[])
        attempts.append(attempt)
        counts['proposed'] += 1
        if family == 'archive':
            archive_visited.add(provenance['archive_index'])
            # A rejected record excludes both arms; a valid partial import must
            # still expose the not-yet-generated arm when the hard cap is hit.
            for arm in (('learner', 'challenger') if exclusion else (provenance['arm'],)):
                archive_arms_visited.add((provenance['archive_index'], arm))
        if exclusion:
            attempt['reason_codes'] = [exclusion]
            return True
        screen = _syntax_screen(prefix, actions)
        attempt['conditional_screen'] = screen
        if screen['status'] == 'INVALID':
            attempt['reason_codes'] = screen['reason_codes']
            return True
        replay = _trace(state, prefix, options, cache, actions)
        if replay['status'] != 'REACHABLE':
            attempt.update(replay)
            attempt['status'] = 'REJECTED'
            return True
        counts['legal'] += 1
        final = replay.pop('final')
        attempt['prefix_class_trace'] = replay['prefix_class_trace']
        if wanted is not None and not R._equal(wanted, replay['canonical_plan']):
            attempt.update(reason_codes=['TARGET_MISMATCH'], intended_plan=copy.deepcopy(wanted),
                           reached_plan=replay['canonical_plan'], residual=T._residual(
                               R._plan(wanted, '$.target'), final.plan))
            return True
        counts['reachable'] += 1
        # Preserve compiler memory and native HOLD renewal semantics without an
        # epoch/publication (which would need real geometry and owned allocation).
        agent = R._memory({**context['compiler_memory'], 'game_id': 'proposal-context'}, '$.compiler_memory')
        agent.bind_holds(state, agent.plan, final.plan)
        holds = V0Continuation.capture(agent).to_dict()['hold_releases']
        continuation = dict(decoder=replay['continuation_signature'],
                            starting_compiler_memory_sha256=R.canonical_sha256(context['compiler_memory']),
                            hold_releases_after_binding=holds)
        exact = T._json([replay['canonical_plan'], continuation])
        if exact in seen:
            attempt.update(status='DUPLICATE', reason_codes=['DUPLICATE_SUCCESSOR'], duplicate_of=seen[exact])
            return True
        candidate_id = R.canonical_sha256([context_key, replay['canonical_plan'], continuation])
        seen[exact] = candidate_id
        candidate = dict(schema=CANDIDATE_SCHEMA, candidate_id=candidate_id, proposal_family=family,
                         parent_proposal_ids=[learner['direct_draw_id']], provenance=copy.deepcopy(provenance),
                         raw_actions=list(actions), prefix_class_trace=replay['prefix_class_trace'],
                         canonical_plan=replay['canonical_plan'], continuation_signature=replay['continuation_signature'],
                         full_continuation_signature=continuation, raw_behavior_logp=None,
                         sampling_provenance={'type': 'search_selection'}, actor_training_admissible=False,
                         reachability=dict(status='REACHABLE', target_exact=True, compiler_executed=False,
                                           scope='FROM_SUPPLIED_PREFIX_ONLY'),
                         screen=_plan_screen(final.plan))
        if compiler_check is not None:
            candidate['compiler_check'] = copy.deepcopy(compiler_check)
            candidate['reachability']['compiler_executed'] = True
        candidates.append(candidate)
        counts['unique'] += 1
        counts['retained'] += 1
        attempt.update(status='RETAINED', candidate_id=candidate_id)
        return True

    consider(learner['raw_actions'], 'learner', {'direct_draw_id': learner['direct_draw_id']}, learner['canonical_plan'])
    consider([0], 'keep', {'meaning': 'STOP_ON_CURRENT_PREFIX'}, prefix.plan.canonical())
    provider_receipt = None
    if provider is not None:
        _configuration, external_caps = _provider_config(provider, provider_budget or {})
        external_caps['max_candidates'] = min(external_caps['max_candidates'], budget['max_candidates'] - len(attempts))
        external_caps['max_path_edits'] = min(external_caps['max_path_edits'], budget['max_path_edits'])
        provider_receipt = provider_candidates(context, provider=provider, budget=external_caps, seed=seed)
        retained = {c['candidate_id']: c for c in provider_receipt['candidates']}
        for attempt in provider_receipt['attempts']:
            candidate = retained.get(attempt.get('candidate_id', attempt.get('duplicate_of')))
            if candidate is None:
                consider(None, 'external_teacher', attempt['provenance'], exclusion=attempt['reason_codes'][0])
            else:
                consider(candidate['raw_actions'], 'external_teacher', attempt['provenance'],
                         candidate['canonical_plan'], compiler_check=candidate['compiler_check'])
    # Round-robin reserves access to synthesis even with a long archive. Lazy
    # archive validation is bounded by remaining candidate slots, not its length.
    archive_iter = iter(_archive_paths(archive, context_key, pins, checkpoint))
    archive_done = not archive
    trial = 0
    while len(attempts) < budget['max_candidates']:
        if not archive_done:
            try:
                actions, wanted, origin, reason = next(archive_iter)
            except StopIteration:
                archive_done = True
            else:
                consider(actions, 'archive', origin, wanted, reason)
        for family in SOURCES:
            if len(attempts) >= budget['max_candidates']:
                break
            origin = dict(trial=trial, generator_seed=seed)
            if family in ('learner_mutation', 'keep_mutation'):
                parent_actions = learner['raw_actions'] if family == 'learner_mutation' else [0]
                actions, operation = _mutate(parent_actions, rng, trial)
                origin.update(operation=operation, adapter='v0_oracle.neighborhood')
                if len(actions) - 1 > budget['max_path_edits']:
                    consider(actions, family, origin, exclusion='PATH_SEARCH_BUDGET')
                    continue
            else:
                class_mode = family == 'theorem_conditioned'
                actions = _random_path(state, prefix, options, cache, rng, budget['max_path_edits'], classes=class_mode)
                origin['sampling'] = 'UNIFORM_PREFIX_CLASSES' if class_mode else 'UNIFORM_LEGAL_RAW_IDS'
                if class_mode:
                    origin.update(evidence=PARENTS['prefix_classes'], conditions='EXACT_CURRENT_PREFIX_MENU_ONLY')
            consider(actions, family, origin)
        trial += 1
    result = dict(schema=SCHEMA, status='UNSELECTED_POOL', actor_context=copy.deepcopy(context),
                  actor_context_sha256=context_key, parent_checkpoint_sha256=checkpoint, source_pins=pins,
                  generator_source_pins=generator_pins, parent_evidence=PARENTS.copy(), seed=seed, budget=budget,
                  learner_draw=copy.deepcopy(learner), candidates=candidates, attempts=attempts, counts=counts,
                  deferred_archive_indices=[i for i in range(len(archive)) if i not in archive_visited],
                  deferred_reason='CANDIDATE_CAP',
                  deferred_archive_arms=[dict(archive_index=i, arm=arm, reason_code='CANDIDATE_CAP')
                                         for i in range(len(archive)) for arm in ('learner', 'challenger')
                                         if (i, arm) not in archive_arms_visited],
                  count_semantics='proposed=attempts; legal=complete_native_paths; reachable=exact_targets_before_dedup; '
                                  'unique=unique_reachable_Plan_plus_continuation; retained=unique; no games',
                  limits=['BOUNDED_SEARCH_NOT_OPTIMALITY', 'NO_COMPILE_OR_TERMINAL_VALUE_CERTIFICATE',
                          'NO_TEACHER_ADMISSION', 'NO_SEARCH_BEHAVIOR_LIKELIHOOD'])
    if provider_receipt is not None:
        result['provider_receipt'] = provider_receipt
    result['pool_sha256'] = R.canonical_sha256(result)
    return copy.deepcopy(result)


# External transports are trusted maintainer-supplied commands, never model output.
# JSON stdin is the ENTIRE data seam; no record, evaluation or learner logp is sent.
PROVIDER_SCHEMA = 'farm-v0-regret-provider-request-v1'
PROVIDER_RESULT_SCHEMA = 'farm-v0-regret-provider-candidates-v1'
PROVIDER_DEFAULTS = dict(max_calls=1, max_candidates=4, max_path_edits=4,
    max_request_bytes=262144, max_response_bytes=262144,
    max_wall_seconds=30., call_timeout_seconds=5.)


def _provider_config(provider, budget):
    R._safe(provider)
    R._fields(provider, ('identity', 'command', 'config', 'prompt'), '$.provider')
    R._text(provider['identity'], '$.provider.identity')
    R._text(provider['prompt'], '$.provider.prompt')
    R._require(type(provider['config']) is dict, 'INVALID_OBJECT', '$.provider.config')
    command = R._list(provider['command'], '$.provider.command')
    R._require(0 < len(command) <= 64, 'PROVIDER_COMMAND', '$.provider.command')
    for word in command:
        R._text(word, '$.provider.command')
        R._require('\0' not in word, 'PROVIDER_COMMAND', '$.provider.command')
    R._fields(budget, (), '$.provider_budget', optional=PROVIDER_DEFAULTS)
    caps = PROVIDER_DEFAULTS | budget
    for key, low, high in (('max_calls', 0, 32), ('max_candidates', 0, 32),
            ('max_path_edits', 0, 16), ('max_request_bytes', 1, 1048576),
            ('max_response_bytes', 1, 1048576)):
        R._int(caps[key], '$.provider_budget.' + key, low, high)
    for key in ('max_wall_seconds', 'call_timeout_seconds'):
        R._number(caps[key], '$.provider_budget.' + key, 0, 3600)
    R._require(caps['call_timeout_seconds'] > 0, 'PROVIDER_TIMEOUT_RANGE', '$.provider_budget')
    return copy.deepcopy(provider), caps


class _ProviderTimeout(TimeoutError):
    pass


@contextmanager
def _provider_timebox(seconds):
    """Bound parsing/replay as well as I/O; preserve an earlier searcher alarm.

    POSIX main-thread only, fail closed otherwise. This bounds active work, not
    OS scheduling/kill/reap latency. It is not a sandbox for command code or a
    guarantee that killing a client cancels a remote server's billed generation.
    """
    R._require(os.name == 'posix' and hasattr(signal, 'setitimer') and
               threading.current_thread() is threading.main_thread(),
               'PROVIDER_DEADLINE_UNSUPPORTED', '$.provider')
    before = time.monotonic()
    delay, interval = signal.getitimer(signal.ITIMER_REAL)
    # An enclosing searcher already owns the shorter deadline. Do not swallow
    # its exception or renew its budget, including during transport cleanup.
    if delay and delay <= seconds:
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_sig, _frame):
        raise _ProviderTimeout('PROVIDER_WALL_CAP')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous)
        if delay:
            signal.setitimer(signal.ITIMER_REAL, max(1e-6, delay - (time.monotonic() - before)), interval)


def _provider_exchange(command, payload, *, timeout, byte_limit, receipt):
    """Bound BOTH stdout/stderr before JSON parsing; kill the process group.

    A temporary stdin file avoids pipe deadlocks when a provider writes before
    reading. At most byte_limit retained bytes plus one overflow sentinel are
    read. Report the sentinel as a lower bound, never a total output estimate.
    No shell, pickle, executable response, inherited stdin or error-text echo.
    """
    started = time.monotonic()
    output, error = bytearray(), bytearray()
    child = None
    try:
        with tempfile.TemporaryFile() as source, selectors.DefaultSelector() as streams:
            source.write(payload); source.seek(0)
            child = subprocess.Popen(command, stdin=source, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=True, close_fds=True)
            receipt['process_started'] = True
            for stream, name in ((child.stdout, 'stdout'), (child.stderr, 'stderr')):
                os.set_blocking(stream.fileno(), False)
                streams.register(stream, selectors.EVENT_READ, name)
            while streams.get_map():
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise _ProviderTimeout('PROVIDER_TIMEOUT')
                for key, _event in streams.select(remaining):
                    room = byte_limit - len(output) - len(error)
                    chunk = os.read(key.fd, min(8192, room + 1))
                    if not chunk:
                        streams.unregister(key.fileobj)
                        continue
                    target = output if key.data == 'stdout' else error
                    target.extend(chunk[:room])
                    if len(chunk) > room:
                        receipt['response_limit_exceeded'] = True
                        raise R.RecordError('PROVIDER_RESPONSE_BYTE_CAP', '$.provider.response')
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise _ProviderTimeout('PROVIDER_TIMEOUT')
            try:
                receipt['exit_code'] = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise _ProviderTimeout('PROVIDER_TIMEOUT') from exc
            R._require(receipt['exit_code'] == 0, 'PROVIDER_EXIT', '$.provider')
            return bytes(output)
    finally:
        if child is not None:
            # Also retire children that kept running after the immediate parent.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            child.stdout.close(); child.stderr.close()
        receipt.update(elapsed_seconds=max(0., time.monotonic() - started),
            response_bytes=len(output) + len(error), stdout_bytes=len(output), stderr_bytes=len(error),
            stdout_sha256=hashlib.sha256(output).hexdigest(), stderr_sha256=hashlib.sha256(error).hexdigest(),
            response_bytes_observed_lower_bound=len(output) + len(error) + int(receipt['response_limit_exceeded']))


def _target_path(state, prefix, options, menus, wanted, limit):
    """Exact one-key lowering, never preloading or an approximate trajectory.

    Inspect only the earliest differing key's native classes, then rebuild at
    its successor. Failure is a bounded lowering refusal, not global optimality
    or multi-dawn unreachability. No edit order/alias-history enumeration.
    """
    actions = []
    while prefix.plan != wanted:
        R._require(prefix.edit_count < V.MAX_EDITS_PER_EPOCH, 'EDIT_CAP', '$.target_plan')
        R._require(len(actions) < limit, 'PATH_SEARCH_BUDGET', '$.target_plan')
        changes = []
        for attr, family in (('cohorts', 'PROD'), ('regimes', 'SERVICE'), ('market', 'MARKET')):
            current, goal = getattr(prefix.plan, attr), getattr(wanted, attr)
            for item in sorted(set(current) | set(goal)):
                if current.get(item) != goal.get(item):
                    action = V.encode(family, item, 0, 0, 0) if family == 'MARKET' else V.encode(family, item, 0)
                    changes.append((V.plan_key(action), attr, item, goal.get(item)))
        if prefix.plan.land != wanted.land:
            changes.append((V.plan_key(V.encode('LAND', 0)), 'land', None, wanted.land))
        key, attr, item, value = min(changes)
        R._require(key > prefix.last_key, 'TARGET_UNREACHABLE', '$.target_plan')
        menu = _menu(state, prefix, options, menus)
        choices = []
        for cls in menu.classes:
            action = min(menu.legal_ids[i] for i in cls.indices)
            if action == 0 or V.plan_key(action) != key:
                continue
            field = getattr(cls.successor.plan, attr)
            if (field if item is None else field.get(item)) == value:
                choices.append((action, cls.successor))
        R._require(bool(choices), 'TARGET_UNREACHABLE', '$.target_plan')
        action, prefix = min(choices, key=lambda x: x[0])
        actions.append(action)
    return actions + [0]


def _provider_candidate(context, spec, state, prefix, options, menus, *, limit, provenance):
    # Lazy import avoids the searcher's existing P import cycle; it owns the
    # exact candidate/continuation identity shared with the immediate consumer.
    from farm_neat.training.teachers import v0_regret_searcher as searcher
    from farm_neat.economy.v0 import compiler as C
    from farm_neat.economy.v0.projection import Projection

    R._safe(spec)
    R._fields(spec, (), '$.provider.candidate', optional=('raw_actions', 'target_plan', 'prefix_supports'),
              code='PROVIDER_CANDIDATE_FIELDS')
    R._require('raw_actions' in spec or 'target_plan' in spec, 'MISSING_PATH_OR_PLAN', '$.provider.candidate')
    wanted = R._plan(spec['target_plan'], '$.target_plan') if 'target_plan' in spec else None
    actions = spec.get('raw_actions') if 'raw_actions' in spec else _target_path(
        state, prefix, options, menus, wanted, limit)
    screen = _syntax_screen(prefix, actions)
    R._require(screen['status'] != 'INVALID', screen['reason_codes'][0], '$.provider.candidate')
    R._require(len(actions) - 1 <= limit, 'PATH_SEARCH_BUDGET', '$.provider.candidate')
    replay = _trace(state, prefix, options, menus, actions)
    R._require(replay['status'] == 'REACHABLE', 'ILLEGAL_PREFIX_ACTION', '$.provider.candidate')
    if 'prefix_supports' in spec:
        R._require(R._equal(spec['prefix_supports'], [r['support'] for r in replay['prefix_class_trace']]),
                   'STALE_SUPPORT', '$.provider.candidate')
    # Canonicalize aliases ONLY after actual-prefix replay. Never sort/collapse
    # an invalid raw edit sequence or silently append STOP to a supplied path.
    actions = [min(row['class_members']) for row in replay['prefix_class_trace']]
    lowered = dict(proposal_family='external_teacher', raw_actions=actions, provenance=provenance)
    if wanted is not None:
        lowered['target_plan'] = wanted.canonical()
    candidate, reasons = searcher._candidate(context, lowered, state, prefix, options, menus, edit_limit=limit)
    R._require(candidate is not None, reasons[0] if reasons else 'INVALID_CANDIDATE', '$.provider.candidate')
    # Ordinary compiler preview using the existing own-memory codec. The closed
    # context lacks real placement geometry: this is NOT executor realization.
    final = R._plan(candidate['canonical_plan'], '$.candidate.plan')
    agent = R._memory({**context['compiler_memory'], 'game_id': 'proposal-context'}, '$.compiler_memory')
    agent.bind_holds(state, agent.plan, final)
    projection = Projection(state, final, **agent.projection_kw())
    allocation = C.allocate_sites(final, projection.land_steps, agent.allocation,
                                  owned=state.quadrants, now=state.step)
    dropped = sorted({tile for key, tile in agent.allocation.items() if allocation.get(key) != tile})
    started = dict(agent.started)
    for key, tile in allocation.items():
        if C.window(key[0], final.regime(key[0]))[0] <= state.step:
            started[tile] = key
    replaced = {tile for key, tile in allocation.items() if started.get(tile, key) != key}
    try:
        program = C.compile_epoch(state, projection, allocation, hands_mode=agent.hands_mode,
            fresh=state.step == 0, dropped=dropped, replace_tiles=replaced)
        R._require(projection.feasible and not C.unplaced(allocation), 'COMPILER_INFEASIBLE', '$.provider.candidate')
    except ValueError as exc:
        if isinstance(exc, R.RecordError):
            raise
        raise R.RecordError('COMPILER_REFUSAL', '$.provider.candidate') from exc
    finally:
        C.allocation_unplaced.pop(id(allocation), None)
    candidate['compiler_check'] = dict(status='COMPILED', scope='PROJECTION_CONTEXT_ONLY',
        program=program, program_sha256=R.canonical_sha256(program), executor_checked=False,
        physical_geometry_verified=False, projection_feasible=True, unplaced=[], game_steps=0)
    candidate['reachability']['compiler_executed'] = True
    return candidate


def provider_candidates(context: dict, *, provider: dict, budget: dict, seed: int) -> dict:
    """Run a capped proposal-only arm; return all charged attempts and survivors.

    provider = {identity, command: argv, config: static JSON object, prompt: str}.
    Each command reads one PROVIDER_SCHEMA request from stdin and writes ONE
    JSON object: {candidates: [{raw_actions: [id,...,0], target_plan?: canonical
    Plan, prefix_supports?: [legal_ids,...]} | {target_plan: canonical Plan}],
    usage?: {input_tokens: int, output_tokens: int}}. JSON text and structured
    solver output use the same safe codec. Unknown fields (including confidence,
    admission, confirmation and logp) refuse; nothing evaluates response code.

    Budgets are invocation totals, not per-call renewals. Every parsed candidate
    (including duplicates/rejections) costs one slot; a failed/oversized/empty
    response costs one refusal slot. Calls charge before spawn. Bytes bound total
    submitted requests and retained stdout+stderr, with a one-byte overflow probe.
    Call counts are verified local cost; token usage is optional UNVERIFIED data.
    A wall deadline covers transport, parsing, lowering and compiler checks; only
    kill/reap/final receipt accounting can overrun. No retries until validity.

    The trusted command can access its OS account/files/environment. The wire
    allowlist is not OS isolation; maintainers must provide a sandboxed/stateless
    adapter without hidden-data access and server-side spend/cancellation limits.
    """
    provider, caps = _provider_config(provider, budget)
    R._int(seed, '$.provider_seed', 0, 2**64 - 1)
    started, cpu = time.monotonic(), time.process_time()
    result = dict(schema=PROVIDER_RESULT_SCHEMA, status='UNSELECTED_PROVIDER_POOL', budget=caps,
        provider_identity=provider['identity'], provider_spec_sha256=R.canonical_sha256(provider),
        seed=seed, calls=[], attempts=[], candidates=[], reason_codes=[],
        counts=dict(calls=0, attempted_candidates=0, request_bytes=0, response_bytes=0,
                    physical_games=0, simulator_steps=0), actor_training_admissible=False, teacher_weight=0,
        limits=['PROPOSALS_NOT_LABELS', 'TOKEN_USAGE_UNVERIFIED', 'NO_EXECUTOR_REALIZATION'])
    calls, attempts, candidates, counts = (result[k] for k in ('calls', 'attempts', 'candidates', 'counts'))
    seen, menus = {}, {}
    active_call, active_attempt = None, None

    def refusal(reason, origin=None):
        row = dict(attempt_id=len(attempts), proposal_family='external_teacher', status='REJECTED',
                   reason_codes=[reason], provenance=origin or {}, raw_actions=None)
        attempts.append(row); counts['attempted_candidates'] += 1
        return row

    def execute():
        nonlocal active_call, active_attempt
        state, prefix, options = R._actor(context)
        result['actor_context_sha256'] = R.canonical_sha256(context)
        if prefix.terminated:
            result['reason_codes'] = ['AFTER_STOP']; return
        for index in range(caps['max_calls']):
            available = caps['max_candidates'] - len(attempts)
            if available <= 0:
                break
            request = dict(schema=PROVIDER_SCHEMA, actor_context=copy.deepcopy(context),
                actor_context_sha256=result['actor_context_sha256'], provider_identity=provider['identity'],
                config=provider['config'], prompt=provider['prompt'],
                prompt_sha256=R.canonical_sha256(provider['prompt']),
                provider_seed=int(R.canonical_sha256([seed, index])[:16], 16), call_index=index,
                limits=dict(max_candidates=available, max_path_edits=caps['max_path_edits'],
                            max_response_bytes=caps['max_response_bytes'] - counts['response_bytes']))
            payload = (T._json(request) + '\n').encode('utf-8')
            if len(payload) > caps['max_request_bytes'] - counts['request_bytes']:
                refusal('PROVIDER_REQUEST_BYTE_CAP'); break
            remaining_bytes = caps['max_response_bytes'] - counts['response_bytes']
            if remaining_bytes <= 0:
                refusal('PROVIDER_RESPONSE_BYTE_CAP'); break
            receipt = dict(call_id=index, provider_identity=provider['identity'],
                config_sha256=R.canonical_sha256(provider['config']), prompt_sha256=request['prompt_sha256'],
                command_sha256=R.canonical_sha256(provider['command']), request_sha256=R.canonical_sha256(request),
                provider_seed=request['provider_seed'], status='CHARGED', charged_calls=1,
                request_bytes=len(payload), response_bytes=0, response_limit_exceeded=False,
                process_started=False, usage_reported=None, usage_verified=False)
            active_call = receipt; calls.append(receipt)
            counts['calls'] += 1; counts['request_bytes'] += len(payload)
            origin = dict(provider_spec_sha256=result['provider_spec_sha256'], call_id=index,
                          request_sha256=receipt['request_sha256'])
            try:
                raw = _provider_exchange(provider['command'], payload,
                    timeout=caps['call_timeout_seconds'], byte_limit=remaining_bytes, receipt=receipt)
                document = R._loads(raw.decode('utf-8'))
                R._safe(document)
                R._fields(document, ('candidates',), '$.provider.response', optional=('usage',),
                          code='PROVIDER_RESPONSE_FIELDS')
                specs = R._list(document['candidates'], '$.provider.response.candidates')
                R._require(bool(specs), 'EMPTY_PROVIDER_RESPONSE', '$.provider.response')
                R._require(len(specs) <= available, 'PROVIDER_CANDIDATE_CAP', '$.provider.response')
                if 'usage' in document:
                    R._fields(document['usage'], ('input_tokens', 'output_tokens'), '$.provider.usage')
                    for key, value in document['usage'].items():
                        R._int(value, '$.provider.usage.' + key, 0, 2**63 - 1)
                    receipt['usage_reported'] = document['usage']
                receipt['status'] = 'COMPLETE'
            except (R.RecordError, UnicodeError, OSError, _ProviderTimeout) as exc:
                reason = (str(exc) if isinstance(exc, _ProviderTimeout) else
                    getattr(exc, 'code', 'PROVIDER_ENCODING' if isinstance(exc, UnicodeError) else 'PROVIDER_SPAWN_ERROR'))
                receipt.update(status='REFUSED', reason_codes=[reason])
                refusal(reason, origin)
                if isinstance(exc, _ProviderTimeout) and str(exc) == 'PROVIDER_WALL_CAP':
                    result['reason_codes'] = ['PROVIDER_WALL_CAP']; return
                continue
            finally:
                counts['response_bytes'] += receipt['response_bytes']
                active_call = None
            for item, spec in enumerate(specs):
                origin = {**origin, 'response_index': item, 'response_candidate_sha256': R.canonical_sha256(spec)}
                row = refusal('VALIDATION_PENDING', origin)
                row['submitted_candidate'] = copy.deepcopy(spec)
                active_attempt = row
                try:
                    candidate = _provider_candidate(context, spec, state, prefix, options, menus,
                        limit=caps['max_path_edits'], provenance=origin)
                    row['raw_actions'] = candidate['raw_actions']
                    cid = candidate['candidate_id']
                    if cid in seen:
                        row.update(status='DUPLICATE', reason_codes=['DUPLICATE_SUCCESSOR'], duplicate_of=cid)
                    else:
                        seen[cid] = candidate; candidates.append(candidate)
                        row.update(status='RETAINED', reason_codes=[], candidate_id=cid)
                except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError) as exc:
                    if isinstance(exc, _ProviderTimeout):
                        raise
                    row['reason_codes'] = [getattr(exc, 'code', 'PROVIDER_CANDIDATE_INVALID')]
                finally:
                    if row['reason_codes'] != ['VALIDATION_PENDING']:
                        active_attempt = None
        result['reason_codes'] = ['PROVIDER_CANDIDATE_CAP' if len(attempts) >= caps['max_candidates']
                                  else 'PROVIDER_CALL_CAP']

    try:
        if caps['max_wall_seconds'] == 0:
            # Validate caller data even with no transport allowance.
            R._actor(context)
            result['reason_codes'] = ['PROVIDER_WALL_CAP']
        else:
            with _provider_timebox(caps['max_wall_seconds']):
                execute()
    except _ProviderTimeout:
        result['reason_codes'] = ['PROVIDER_WALL_CAP']
        unfinished = active_attempt or next((row for row in attempts
            if row['reason_codes'] == ['VALIDATION_PENDING']), None)
        if unfinished is not None:
            unfinished.update(status='REJECTED', reason_codes=['PROVIDER_WALL_CAP'])
        elif len(attempts) < caps['max_candidates']:
            refusal('PROVIDER_WALL_CAP')
    except R.RecordError as exc:
        if exc.code != 'PROVIDER_DEADLINE_UNSUPPORTED':
            raise
        result['reason_codes'] = [exc.code]
    # The asynchronous deadline can land between append and receipt commit.
    # After restoring the timer, expose only committed survivors and derive
    # counters from the charged records, never partially updated counters.
    committed = {row.get('candidate_id') for row in attempts if row['status'] == 'RETAINED'}
    candidates[:] = [candidate for candidate in candidates if candidate['candidate_id'] in committed]
    counts.update(calls=len(calls), attempted_candidates=len(attempts),
                  request_bytes=sum(row['request_bytes'] for row in calls),
                  response_bytes=sum(row['response_bytes'] for row in calls),
                  retained=len(candidates), elapsed_seconds=max(0., time.monotonic() - started),
                  cpu_seconds=max(0., time.process_time() - cpu))
    # Source pins are audit metadata, NOT another input to the external provider.
    result['source_pins'] = R.source_pins() | {p: hashlib.sha256((R.ROOT / p).read_bytes()).hexdigest()
        for p in (*SOURCE_PATHS, 'src/farm_neat/training/teachers/v0_regret_searcher.py')}
    result['content_sha256'] = R.canonical_sha256(result)
    return result


def teacher_proposer(provider: dict, *, provider_budget: dict | None = None, fallback: str = 'fixed'):
    """Existing searcher callback; one charged provider attempt then fixed fallback.

    Each search iteration has its own receipt and <=1 call/attempt, so searcher's
    max_iterations also bounds total provider calls. No persistent retry/cache
    state to lose on search restart. The fallback has the original seed/iteration
    and never consumes provider RNG. Its extra proposal is explicit in provenance.
    """
    from farm_neat.training.teachers import v0_regret_searcher as S
    provider, caps = _provider_config(provider, provider_budget or {})
    R._require(fallback in ('fixed', 'random', *S.SOLVER_FAMILIES), 'PROVIDER_FALLBACK', '$.fallback')
    def proposer(context, incumbent, *, budget, seed):
        local = caps | dict(max_calls=min(1, caps['max_calls']), max_candidates=min(1, caps['max_candidates']),
                           max_path_edits=min(caps['max_path_edits'], budget['max_path_edits']))
        receipt = provider_candidates(context, provider=provider, budget=local, seed=seed)
        if receipt['candidates']:
            c = receipt['candidates'][0]
            spec = dict(proposal_family='external_teacher', raw_actions=c['raw_actions'],
                        target_plan=c['canonical_plan'], provenance={'compiler_check': c['compiler_check']})
        elif fallback == 'fixed':
            spec = S.fixed_proposer(context, incumbent, budget=budget, seed=seed)
        elif fallback == 'random':
            spec = S.random_proposer(context, incumbent, budget=budget, seed=seed)
        else:
            spec = S.solver_candidate(context, incumbent, family=fallback, budget=budget, seed=seed)
        # Keep the sealed receipt intact: dropping its candidates would make its
        # content hash unverifiable and erase rejected/duplicate witnesses.
        spec.setdefault('provenance', {})['teacher_injection'] = dict(
            receipt=receipt, fallback_attempted=not bool(receipt['candidates']))
        return spec
    return proposer


# Batched proposal boundary. The existing scalar pool above remains unchanged.
# No production per-token model loop lives here: a caller-owned batch inference
# driver receives one BatchClassPaths session per population chunk. It submits
# complete prefix waves to advance(), never one scalar scorer call per token.
BATCH_SCHEMA = 'farm-v0-batch-proposals-v1'
BATCH_FAMILIES = ('policy_sample', 'random_legal_control', 'incumbent_mutation',
                  'solver_provider', 'known_suffix_mutation')
BATCH_LIMITS = dict(max_batch_rows=64, max_tensor_bytes=256 * 1024**2,
                    max_record_bytes=64 * 1024**2, max_path_edits=16)


def _batch_quotas(quotas):
    R._fields(quotas, BATCH_FAMILIES, '$.quotas')
    for family in BATCH_FAMILIES:
        R._int(quotas[family], '$.quotas.' + family, 0, 32)
    R._require(1 <= sum(quotas.values()) <= 32, 'POPULATION_CAP', '$.quotas')
    return dict(quotas)


def _batch_limits(values):
    R._fields(values, (), '$.batch_limits', optional=BATCH_LIMITS)
    out = BATCH_LIMITS | values
    for name, lo, hi in (('max_batch_rows', 1, 256), ('max_tensor_bytes', 1, 2**32),
                         ('max_record_bytes', 1, 2**30), ('max_path_edits', 1, 16)):
        R._int(out[name], '$.' + name, lo, hi)
    return out


def batch_memory_estimate(rows=64, support_width=V.N_ACTIONS):
    """Conservative working-tensor allowance, NOT model/allocator/host RSS.

    Includes padded phi/IDs, class maps, FP64 logits/normalizers/scatters/CDF
    and transient masks with padding. All proposals share at most ONE prefix
    frame. Model parameters, transformer activations/KV and native Python menu
    objects remain separately profiled integration costs, not a claimed bound.
    """
    from farm_neat.policy.v0_scorer import PHI_NAMES
    R._int(rows, '$.rows', 1, 256)
    R._int(support_width, '$.support_width', 1, V.N_ACTIONS)
    per_entry = 128 + 4 * len(PHI_NAMES)
    return dict(rows=rows, support_width=support_width, bytes_per_entry=per_entry,
                working_tensor_bytes=rows * (support_width * per_entry + 1024),
                model_and_allocator_bytes='NOT_INCLUDED', host_menu_bytes='NOT_INCLUDED')


def _batch_rows(queue_states):
    R._list(queue_states, '$.queue_states')
    R._require(0 < len(queue_states) <= 64, 'QUEUE_BATCH_CAP', '$.queue_states')
    result, ids = [], set()
    for row in queue_states:
        R._fields(row, ('queue_id', 'snapshot_sha256', 'split', 'actor_context',
                        'incumbent_actions', 'known_suffixes'), '$.queue_state')
        R._text(row['queue_id'], '$.queue_id')
        R._hash(row['snapshot_sha256'], '$.snapshot_sha256')
        R._require(row['split'] == 'discovery', 'PROTECTED_QUEUE_SPLIT', '$.split')
        R._require(row['queue_id'] not in ids, 'DUPLICATE_QUEUE_ID', '$.queue_id')
        ids.add(row['queue_id'])
        state, prefix, options = R._actor(row['actor_context'])
        R._require(not prefix.terminated, 'AFTER_STOP', '$.prefix')
        # These are actor-visible path geometries only, not archive outcomes.
        R._list(row['known_suffixes'], '$.known_suffixes')
        R._require(len(row['known_suffixes']) <= 32, 'SUFFIX_CAP', '$.known_suffixes')
        for actions in [row['incumbent_actions'], *row['known_suffixes']]:
            screen = _syntax_screen(prefix, actions)
            R._require(screen['status'] != 'INVALID', screen['reason_codes'][0], '$.seed_path')
        result.append(copy.deepcopy(row))
    return result


def _batch_jobs(rows, quotas, seed, stage, namespace=None):
    jobs = []
    for index, row in enumerate(rows):
        binding = R.canonical_sha256(row)
        for family in BATCH_FAMILIES:
            for slot in range(quotas[family]):
                stream = R.canonical_sha256(['v0-batch-counter-v1', namespace, seed, binding, stage, family, slot])
                jobs.append(dict(row_index=index, proposal_family=family, stage=stage, slot=slot,
                                 rng_stream_sha256=stream, attempt_id=stream,
                                 generator_seed=int(stream[:16], 16)))
    return jobs


def _batch_attempt(job):
    return {k: copy.deepcopy(v) for k, v in job.items() if k != 'row_index'} | dict(
        status='CENSORED', reason_codes=['NOT_EXECUTED'], raw_actions=None,
        class_behavior_logp=None, raw_behavior_logp=None)


def _batch_empty(rows):
    return [dict(queue_id=row['queue_id'], snapshot_sha256=row['snapshot_sha256'],
                 actor_context_sha256=R.canonical_sha256(row['actor_context']),
                 attempts=[], candidates=[], counts={}) for row in rows]


def _batch_counts(row):
    row['counts'] = dict(attempted=len(row['attempts']), retained=len(row['candidates']),
        duplicates=sum(a['status'] == 'DUPLICATE' for a in row['attempts']),
        illegal=sum(a['status'] == 'ILLEGAL' for a in row['attempts']),
        censored=sum(a['status'] == 'CENSORED' for a in row['attempts']))
    return row


class PrefixTensorFrame:
    """One exact current-prefix wave; supports are native, not supplied targets.

    score_inputs() is the only scorer feature view: own/public State, canonical
    Prefix, native support/phi and compiler memory. Queue snapshot identity,
    source groups, proposal outcomes and RNG keys are not model features.
    """
    def __init__(self, indices, menus, session):
        self.indices, self.menus = tuple(indices), tuple(menus)
        self.shape = (len(indices), max(len(m.legal_ids) for m in menus))
        self._session = session

    def score_inputs(self):
        return dict(states=[self._session.states[i] for i in self.indices],
            prefix_plans=[m.prefix for m in self.menus],
            legal_ids=[m.legal_ids for m in self.menus], phi=[m.phi for m in self.menus],
            memories=[{'hold_releases': copy.deepcopy(self._session.options[i]['hold_releases'])}
                      for i in self.indices])


class BatchClassPaths:
    """Injected whole-sequence driver seam with vectorized class sampling.

    The trusted, source-pinned driver calls frame(), scores ALL live rows once,
    then advance(logits). Only this owner chooses/advances canonical classes.
    No raw-history cache or teacher prefix is accepted. Driver state may not
    condition logits on erased aliases; prefix-only scoring is required by
    K-V0-REPLAY-LUMPABILITY. The CPU reference driver is deliberately test-only.
    Calling advance with an unfinished frame or returning early never invents
    a successful path. Request chunks are disposable on callback failure.
    """
    def __init__(self, jobs, rows, limits):
        self.jobs, self.limits = copy.deepcopy(jobs), dict(limits)
        self.contexts = [copy.deepcopy(rows[j['row_index']]['actor_context']) for j in jobs]
        actors = [R._actor(c) for c in self.contexts]
        self.states = [a[0] for a in actors]
        self.prefixes = [a[1] for a in actors]
        self.options = [a[2] for a in actors]
        self.paths = [dict(raw_actions=[], class_step_logps=[], status='ACTIVE',
                           stop_origin=None, reason_codes=[]) for _ in jobs]
        self._frame = None
        self.max_rows_observed = self.max_support_observed = self.max_tensor_estimate = 0
        self.tensor_advances = 0

    def frame(self):
        if self._frame is not None:
            return self._frame
        indices, menus = [], []
        for i, (prefix, path) in enumerate(zip(self.prefixes, self.paths)):
            if path['status'] != 'ACTIVE':
                continue
            if prefix.edit_count == V.MAX_EDITS_PER_EPOCH:
                path['raw_actions'].append(V.STOP_ID)
                path['class_step_logps'].append(0.)
                path.update(status='COMPLETE', stop_origin='NATIVE_CAP')
                continue
            if len(path['raw_actions']) >= self.limits['max_path_edits']:
                if self.jobs[i]['proposal_family'] == 'policy_sample':
                    path.update(status='CENSORED', reason_codes=['PATH_SEARCH_BUDGET'])
                else:
                    path['raw_actions'].append(V.STOP_ID)
                    path.update(status='COMPLETE', stop_origin='BUDGET_CONTROL')
                continue
            # No stale entry mask, raw-history memo or unbounded cross-path cache.
            menu = fast.build_menu(self.states[i], prefix, projection_kwargs=self.options[i], backend='fast')
            indices.append(i); menus.append(menu)
        if not indices:
            return None
        self._frame = PrefixTensorFrame(indices, menus, self)
        n, k = self._frame.shape
        estimate = batch_memory_estimate(n, k)['working_tensor_bytes']
        R._require(estimate <= self.limits['max_tensor_bytes'], 'TENSOR_BYTE_CAP', '$.frame')
        self.max_rows_observed = max(self.max_rows_observed, n)
        self.max_support_observed = max(self.max_support_observed, k)
        self.max_tensor_estimate = max(self.max_tensor_estimate, estimate)
        return self._frame

    def advance(self, logits):
        """One batched tensor draw; FP64 class aggregation and one host transfer.

        Policy mass sums ALL raw aliases of the exact successor. Controls are
        raw-uniform before canonicalization and deliberately have no behavior
        likelihood. Padding never enters a normalizer. RNG is a stable 52-bit
        open-interval uniform per (source-bound path stream, prefix counter).
        """
        import torch
        from farm_neat.policy.v0_scorer import masked_log_probs
        frame = self._frame
        R._require(frame is not None, 'NO_ACTIVE_FRAME', '$.advance')
        R._require(isinstance(logits, torch.Tensor) and logits.shape == frame.shape and
                   logits.dtype in (torch.float32, torch.float64), 'LOGITS_SHAPE_DTYPE', '$.logits')
        n, k = frame.shape
        # Lists here describe one whole ragged native wave, not network calls.
        mapping, enabled, random_rows, uniforms = [], [], [], []
        for index, menu in zip(frame.indices, frame.menus):
            classes = [0] * k
            for ci, cls in enumerate(menu.classes):
                for position in cls.indices:
                    classes[position] = ci
            mapping.append(classes)
            enabled.append([True] * len(menu.legal_ids) + [False] * (k - len(menu.legal_ids)))
            random_rows.append(self.jobs[index]['proposal_family'] == 'random_legal_control')
            key = R.canonical_sha256([self.jobs[index]['rng_stream_sha256'], len(self.paths[index]['raw_actions'])])
            uniforms.append((int(key[:13], 16) + .5) / 2**52)
        device = logits.device
        with torch.no_grad():
            valid = torch.tensor(enabled, dtype=torch.bool, device=device)
            class_index = torch.tensor(mapping, dtype=torch.long, device=device)
            controls = torch.tensor(random_rows, dtype=torch.bool, device=device)
            values = logits.detach().double()
            values = torch.where(controls[:, None], torch.zeros_like(values), values)
            raw = masked_log_probs(values, valid)
            mass = torch.zeros_like(raw).scatter_add_(1, class_index, raw.exp().masked_fill(~valid, 0.))
            cdf = mass.cumsum(-1)
            u = torch.tensor(uniforms, dtype=torch.float64, device=device)
            selected = (cdf <= u[:, None]).sum(-1)
            class_counts = torch.tensor([len(m.classes) for m in frame.menus], device=device)
            selected = torch.minimum(selected, class_counts - 1)
            logp = mass.gather(1, selected[:, None]).squeeze(1).log()
            R._require(bool(torch.isfinite(logp).all()), 'NONFINITE_CLASS_LOGP', '$.sample')
            chosen = torch.stack((selected.double(), logp), -1).cpu().tolist()
        for index, menu, (ci, lp) in zip(frame.indices, frame.menus, chosen):
            cls = menu.classes[int(ci)]
            action = min(menu.legal_ids[i] for i in cls.indices)
            path = self.paths[index]
            path['raw_actions'].append(action)
            path['class_step_logps'].append(lp)
            self.prefixes[index] = cls.successor
            if cls.successor.terminated:
                path.update(status='COMPLETE', stop_origin='SAMPLED')
        self.tensor_advances += 1
        self._frame = None


def propose_batch(queue_states, *, quotas, seed, tensor_decoder=None, provider=None,
                  stage=0, registration_sha256=None, max_batch_rows=64, max_tensor_bytes=256 * 1024**2,
                  max_record_bytes=64 * 1024**2, max_path_edits=16):
    """Produce one fixed, unselected population for many exact queue identities.

    queue row: queue_id, snapshot_sha256, split='discovery', actor_context,
    incumbent_actions, known_suffixes (ordered action paths; no labels). Private
    snapshot bytes remain in the queue/evaluator owner; this boundary cannot
    attest snapshot restoration. Each path is from its supplied current prefix,
    not an isolated action or a whole multi-dawn policy. Later-day scheduling
    belongs to the sequence/worker owner.

    tensor_decoder(session) owns a whole batched driver invocation. provider
    accepts a list of {actor_context, incumbent_actions, seed, max_path_edits}
    and returns equally many strict V0 action-path specs, UNKNOWN or CENSORED.
    Providers can reuse solver_candidate/provider_candidates; they must lower
    to V0, not send public engine action labels. No provider is silently retried
    or replaced. External wall/process isolation remains the existing trusted
    worker's responsibility; progressive_proposals adds its bounded deadline.
    """
    from farm_neat.training.teachers import v0_regret_searcher as S
    rows, quotas = _batch_rows(queue_states), _batch_quotas(quotas)
    R._int(seed, '$.seed', 0, 2**64 - 1)
    R._int(stage, '$.stage', 0, 31)
    if registration_sha256 is not None:
        R._hash(registration_sha256, '$.registration_sha256')
    limits = _batch_limits(dict(max_batch_rows=max_batch_rows, max_tensor_bytes=max_tensor_bytes,
                              max_record_bytes=max_record_bytes, max_path_edits=max_path_edits))
    per_row = batch_memory_estimate(1)['working_tensor_bytes']
    R._require(max_tensor_bytes >= per_row, 'TENSOR_BYTE_CAP', '$.max_tensor_bytes')
    chunk_size = min(max_batch_rows, max_tensor_bytes // per_row)
    jobs, states = _batch_jobs(rows, quotas, seed, stage, registration_sha256), _batch_empty(rows)
    attempts = {j['attempt_id']: _batch_attempt(j) for j in jobs}
    seen = [{} for _ in rows]
    record_bytes = 0
    metrics = dict(tensor_decoder_calls=0, provider_calls=0, tensor_advances=0,
                   max_rows_observed=0, max_support_observed=0, max_tensor_estimate=0,
                   record_bytes=0, effective_batch_rows=chunk_size, physical_games=0, simulator_steps=0)

    def retain(job, actions, *, path=None, provenance=None, reason=None, status='CENSORED'):
        nonlocal record_bytes
        attempt = attempts[job['attempt_id']]
        attempt.update(raw_actions=copy.deepcopy(actions), provenance=copy.deepcopy(provenance or {}))
        if reason is not None:
            attempt.update(status=status, reason_codes=[reason])
            return
        context = rows[job['row_index']]['actor_context']
        actor, prefix, options = R._actor(context)
        try:
            candidate, reasons = S._candidate(context, dict(proposal_family=job['proposal_family'],
                    raw_actions=actions), actor, prefix, options, {}, edit_limit=max_path_edits)
        except (ValueError, TypeError, KeyError) as exc:
            candidate, reasons = None, [getattr(exc, 'code', 'INVALID_PROPOSAL')]
        if candidate is None:
            attempt.update(status='ILLEGAL', reason_codes=reasons)
            return
        # Canonical members preserve exact class continuation, never raw history.
        canonical = [min(t['class_members']) for t in candidate['prefix_class_trace']]
        if actions != canonical:
            candidate, _ = S._candidate(context, dict(proposal_family=job['proposal_family'],
                raw_actions=canonical), actor, prefix, options, {}, edit_limit=max_path_edits)
        attempt['canonical_actions'] = canonical
        candidate.update(stop_origin=path['stop_origin'] if path else
                         ('NATIVE_CAP' if candidate['prefix_class_trace'][-1]['forced_stop'] else 'PROPOSED'),
                         class_behavior_logp=None)
        if path is not None and job['proposal_family'] == 'policy_sample':
            # This is the sampled class path, NOT raw-ID or marginal final-Plan
            # probability, and it is not a direct learner/admitted teacher row.
            candidate['class_behavior_logp'] = sum(path['class_step_logps'])
            candidate['sampling_provenance'] = dict(type='policy_class_sample', mode='sample_class',
                class_step_logps=list(path['class_step_logps']), rng_stream_sha256=job['rng_stream_sha256'],
                class_kernel='CANONICAL_PREFIX_ONLY', temperature=1.)
            attempt['class_behavior_logp'] = candidate['class_behavior_logp']
            attempt['sampling_provenance'] = copy.deepcopy(candidate['sampling_provenance'])
        identity = T._json([candidate['canonical_plan'], candidate['full_continuation_signature']])
        index = job['row_index']
        if identity in seen[index]:
            attempt.update(status='DUPLICATE', reason_codes=['DUPLICATE_SUCCESSOR'],
                           duplicate_of=seen[index][identity])
            return
        candidate['provenance'] = dict(attempt_id=job['attempt_id'], stage=stage,
                                       generator_seed=job['generator_seed'])
        size = len(T._json(candidate).encode('utf-8'))
        if record_bytes + size > max_record_bytes:
            attempt.update(status='CENSORED', reason_codes=['RECORD_BYTE_CAP'])
            return
        record_bytes += size
        seen[index][identity] = candidate['candidate_id']
        states[index]['candidates'].append(candidate)
        attempt.update(status='RETAINED', reason_codes=[], candidate_id=candidate['candidate_id'])

    # Jobs are processed in population order so retention/dedup is independent
    # of chunk size. Decoding in mixed chunks batches the two sampling families.
    for offset in range(0, len(jobs), chunk_size):
        chunk = jobs[offset:offset + chunk_size]
        sampling = [j for j in chunk if j['proposal_family'] in BATCH_FAMILIES[:2]]
        decoded = {}
        if sampling:
            session = BatchClassPaths(sampling, rows, limits)
            try:
                # A native forced-STOP-only chunk needs no model/driver call.
                if session.frame() is not None:
                    if tensor_decoder is None:
                        raise ValueError('TENSOR_DECODER_UNAVAILABLE')
                    metrics['tensor_decoder_calls'] += 1
                    tensor_decoder(session)
                for job, path in zip(sampling, session.paths):
                    decoded[job['attempt_id']] = path
            except Exception as exc:
                code = getattr(exc, 'code', None) or ('CALLBACK_TIMEOUT' if isinstance(exc, TimeoutError)
                                                      else 'TENSOR_DECODER_FAILURE')
                for job, path in zip(sampling, session.paths):
                    # A failed chunk remains charged even when some rows had
                    # completed before failure: no cherry-picking survivors.
                    decoded[job['attempt_id']] = path | dict(status='CENSORED', reason_codes=[code])
            for key in ('tensor_advances',):
                metrics[key] += getattr(session, key)
            for key in ('max_rows_observed', 'max_support_observed', 'max_tensor_estimate'):
                metrics[key] = max(metrics[key], getattr(session, key))
        external = [j for j in chunk if j['proposal_family'] == 'solver_provider']
        supplied = {}
        if external:
            requests = [dict(actor_context=copy.deepcopy(rows[j['row_index']]['actor_context']),
                incumbent_actions=list(rows[j['row_index']]['incumbent_actions']),
                seed=j['generator_seed'], max_path_edits=max_path_edits) for j in external]
            try:
                if provider is None:
                    result = [dict(status='CENSORED', reason_codes=['PROVIDER_UNAVAILABLE']) for _ in requests]
                else:
                    metrics['provider_calls'] += 1
                    result = provider(requests)
                R._require(type(result) is list and len(result) == len(requests), 'PROVIDER_BATCH_ALIGNMENT', '$.provider')
                supplied = dict(zip((j['attempt_id'] for j in external), result))
            except Exception as exc:
                supplied = {j['attempt_id']: dict(status='CENSORED', reason_codes=[getattr(exc, 'code',
                            'PROVIDER_BATCH_FAILURE')]) for j in external}
        for job in chunk:
            family, row = job['proposal_family'], rows[job['row_index']]
            if family in BATCH_FAMILIES[:2]:
                path = decoded[job['attempt_id']]
                if path['status'] != 'COMPLETE':
                    retain(job, path['raw_actions'], reason=(path['reason_codes'] or ['INCOMPLETE_BATCH_DRIVER'])[0],
                           provenance={'partial_class_step_logps': list(path['class_step_logps'])
                                       if family == 'policy_sample' else []})
                else:
                    retain(job, path['raw_actions'], path=path)
            elif family == 'solver_provider':
                spec = supplied[job['attempt_id']]
                try:
                    R._safe(spec)
                    R._fields(spec, (), '$.provider_spec', optional=('raw_actions', 'status', 'reason_codes'))
                    R._require(spec.get('status') in (None, 'CENSORED', 'UNKNOWN'),
                               'PROVIDER_STATUS', '$.provider_spec')
                    if spec.get('status') in ('CENSORED', 'UNKNOWN'):
                        reasons = spec.get('reason_codes') or ['PROVIDER_UNKNOWN']
                        R._require(type(reasons) is list and all(type(x) is str for x in reasons),
                                   'PROVIDER_REASONS', '$.provider_spec')
                        retain(job, spec.get('raw_actions'), reason=reasons[0])
                    else:
                        retain(job, spec.get('raw_actions'))
                except (ValueError, TypeError, KeyError):
                    retain(job, None, reason='PROVIDER_SPEC_INVALID', status='ILLEGAL')
            else:
                base = row['incumbent_actions']
                provenance = {}
                if family == 'known_suffix_mutation':
                    if not row['known_suffixes']:
                        retain(job, None, reason='NO_KNOWN_SUFFIX')
                        continue
                    origin = job['slot'] % len(row['known_suffixes'])
                    base = row['known_suffixes'][origin]
                    provenance['suffix_index'] = origin
                actions, mutation = _mutate(base, random.Random(job['generator_seed']), stage * 32 + job['slot'])
                retain(job, actions, provenance=provenance | {'mutation': mutation})
    for job in jobs:
        states[job['row_index']]['attempts'].append(attempts[job['attempt_id']])
    for row in states:
        _batch_counts(row)
    metrics['record_bytes'] = record_bytes
    metrics['attempted'] = len(jobs)
    result = dict(schema=BATCH_SCHEMA, status='UNSELECTED_BATCH', seed=seed, stage=stage,
                  registration_sha256=registration_sha256,
                  quotas=quotas, states=states, counts=metrics, limits=limits,
                  evidence_scope='INJECTED_TENSOR_DRIVER',
                  source_pins=R.source_pins() | {p: hashlib.sha256((R.ROOT / p).read_bytes()).hexdigest()
                      for p in (SOURCE_PATHS[0], 'src/farm_neat/training/teachers/v0_regret_searcher.py')},
                  limitations=['NO_TEACHER_ADMISSION', 'NO_NATIVE_GAMES_OR_STRENGTH',
                               'SNAPSHOT_RESTORATION_NOT_ATTESTED', 'MODEL_MEMORY_NOT_INCLUDED'])
    result['content_sha256'] = R.canonical_sha256(result)
    return result

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--provider', type=Path, help='trusted command/config/prompt JSON; never response code')
    parser.add_argument('--provider-budget', type=Path, help='provider invocation caps JSON')
    parser.add_argument('--seed', type=int, default=0)
    for key, default in DEFAULT_BUDGET.items():
        parser.add_argument('--' + key.replace('_', '-'), type=int, default=default)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError('output already exists')
        pair = R.validate_pair(R._loads(args.input.read_text(encoding='utf-8')))
        # Protected evaluation rows must not be mined for training proposals.
        R._require(pair['identity']['split'] not in ('monitor', 'final'), 'PROTECTED_INPUT_SPLIT', '$.identity.split')
        archive = [] if args.archive is None else R._loads(args.archive.read_text(encoding='utf-8'))
        pool = propose(pair['actor_context'], pair['learner'],
                       budget={k: getattr(args, k) for k in DEFAULT_BUDGET}, seed=args.seed, archive=archive,
                       provider=None if args.provider is None else R._loads(args.provider.read_text(encoding='utf-8')),
                       provider_budget=None if args.provider_budget is None else R._loads(args.provider_budget.read_text(encoding='utf-8')))
        # Fully serialize before creating the output; exclusive creation refuses
        # concurrent overwrite, even if another writer appeared after the check.
        payload = T._json(pool) + '\n'
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(payload)
        print(json.dumps(dict(status=pool['status'], pool_sha256=pool['pool_sha256'], counts=pool['counts']), sort_keys=True))
    except (ValueError, OSError, UnicodeError) as exc:
        print(json.dumps(dict(status='INVALID', reason_code=getattr(exc, 'code', type(exc).__name__)), sort_keys=True))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
