"""Fixed-family V0 regret collection; separate exact-dawn recovery corpus. Never trains.

Public API: register_plan(spec); collect_round(plan, *, sampler, proposer,
evaluator, output). ``sampler(context, *, draw_id, seed,
parent_checkpoint_sha256)`` returns the existing genuine direct-draw record.
``proposer`` has v0_regret_proposals.propose's signature. ``evaluator`` is the
single-branch runner accepted by v0_regret_pairs.evaluate_pair/verify_pair.
Callbacks are trusted code, not a sandbox; their source fingerprints are frozen
before the first draw. Mutable callback configuration belongs in the registered
spec's callback_config, not an unrecorded closure. Native execution is possible
only through that module's native_runner; injected runners remain diagnostics.
No implicit native runner or policy/checkpoint load exists here.

The spec registers tests (one seat tuple per opportunity), visible contexts and
exact continuation, direct draw IDs/seeds, discovery/confirmation registry IDs,
source families, frozen parent, proposal/start budgets and alpha allocation.
The entire family is selected and locked before ANY confirmation starts. No
unseen candidate is a loss. Equal discovery WDL keeps the learner. All proposal
attempts and both winning and losing paths survive in the immutable object store.

Journal order is INTENT -> LAUNCH (charged/fsynced) -> RESULT. After interruption,
an INTENT can launch; a LAUNCH without RESULT becomes a retained unknown, NEVER
a retry. This deliberately offers at-most-once launches, not an impossible
exactly-once guarantee over arbitrary external side effects. An uncertain launch
is conservatively charged. A hash-chain tear fails closed without truncating it.
Each output directory is one round and must be reused for resume. Cross-directory
copying, hidden exposure and authentic direct sampling require caller custody.

The paired owner creates transient call IDs. The adapter below supplies stable
physical job IDs and rebinds only transport envelopes on an exact semantic cache
hit. Original branches/responses are retained. Learner controls are shared across
discovery challengers, not counted twice. Replays have separate jobs and add no
worlds. Search receipts replace flat selection only: a frozen selected path gets
a paired discovery audit (charged to the base/replay caps) before fresh disjoint
confirmation. Original search starts are charged separately; their attestations
never substitute for native replay rows. The existing response validator checks BOTH transport and original
execution receipts. Native evidence is never inferred from a callback claim.

CLI, explicit trusted factory returning {sampler, proposer, evaluator}:
  python -m farm_neat.rollout.v0_regret_collect plan --spec SPEC.json --output PLAN.json
  python -m farm_neat.rollout.v0_regret_collect run --plan PLAN.json --output RUN --callbacks module:factory
  python -m farm_neat.rollout.v0_regret_collect replay --plan PLAN.json --output RUN --callbacks module:factory
``replay`` cannot launch new samples, proposals or base games. It verifies retained
base results, then rebuilds derived exports. Exhausted budgets are immutable;
more confirmation requires a NEW prospectively registered family, not extending
an observed sample. Atomic exports retain older partial snapshots. No registry
or task state is modified; intake must register this real command separately.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import tempfile

from farm_neat.evaluation import router_panel as panel, scenarios, v0_regret_pairs as E
from farm_neat.training import v0_regret_records as R, v0_regret_admission as A
from farm_neat.training.teachers import v0_regret_proposals as P, v0_regret_searcher as S

SCHEMA = 'farm-v0-regret-collection-v1'
MODULE = 'farm_neat.rollout.v0_regret_collect'
PHASES = ('discovery', 'confirmation')
GEOMETRY = ('raw_actions', 'prefix_class_trace', 'canonical_plan', 'continuation_signature')


def need(condition, code):
    if not condition:
        raise ValueError(code)


def digest(value):
    return R.canonical_sha256(value)


def bind(value, field='content_sha256'):
    value = copy.deepcopy(value)
    value[field] = digest({k: v for k, v in value.items() if k != field})
    return value


def bound(value, code, field='content_sha256'):
    R._safe(value)
    need(type(value) is dict and value.get(field) == bind(value, field)[field], code)


def wire(value):
    # Nested engine receipts intentionally have order-sensitive hashes.
    return E.P.wire(value).encode()


def source_pins():
    names = (*S.SOURCE_PATHS, 'src/farm_neat/rollout/v0_regret_collect.py')
    return E.source_pins() | {p: hashlib.sha256((R.ROOT/p).read_bytes()).hexdigest() for p in names}


def _descriptor(world):
    env = scenarios.ScenarioEnvironment(world.scenario())  # opening only, zero steps
    return dict(source_group_id=world.id, world_sha256=world.sha256,
        snapshot_sha256=E.P.wire_hash(E._world_snapshot(env)),
        rng_sha256=panel.digest(E._rng_binding(world)), seed=world.seed)


def register_plan(spec: dict) -> dict:
    """Validate/register before sampling. Uses existing worlds; runs zero turns."""
    R._safe(spec)
    fields = ('round_id', 'family_id', 'parent_checkpoint_sha256', 'opponent', 'tests',
        'archive', 'family_alpha', 'min_effect', 'independence_attested',
        'compatible_contexts_attested', 'proposal_budget', 'max_discovery_challengers',
        'timeout_seconds', 'budget')
    R._fields(spec, fields, '$.spec', optional=('callback_config',))
    for k in ('round_id', 'family_id'):
        R._text(spec[k], '$.spec.'+k)
    R._hash(spec['parent_checkpoint_sha256'], '$.spec.parent')
    need(spec['opponent'] in E.OPPONENTS, 'UNSUPPORTED_OPPONENT')
    R._number(spec['family_alpha'], '$.alpha', 0., .05)
    need(spec['family_alpha'] > 0, 'ALPHA_MUST_BE_POSITIVE')
    R._number(spec['min_effect'], '$.min_effect', 0., 2.)
    for k in ('independence_attested', 'compatible_contexts_attested'):
        R._bool(spec[k], '$.'+k)
    P._budget(spec['proposal_budget'])
    R._int(spec['max_discovery_challengers'], '$.max_discovery_challengers', 1, 2)
    R._number(spec['timeout_seconds'], '$.timeout_seconds', 0., 300.)
    need(spec['timeout_seconds'] > 0, 'TIMEOUT_MUST_BE_POSITIVE')
    R._fields(spec['budget'], ('max_base_starts', 'max_replay_starts', 'max_search_starts'), '$.budget')
    for k, v in spec['budget'].items():
        R._int(v, '$.budget.'+k, 0, 100000)
    tests = R._list(spec['tests'], '$.tests')
    need(0 < len(tests) <= 32, 'PAIR_OPPORTUNITY_CAP')
    registry = panel.world_registry()
    worlds, exposure, families, ids, draws = {}, {}, {}, set(), set()
    contexts = {}
    for test in tests:
        R._fields(test, ('test_id', 'context_group_id', 'split', 'seats',
            'discovery_world_ids', 'confirmation_world_ids', 'source_families'), '$.test')
        R._text(test['test_id'], '$.test_id'); R._text(test['context_group_id'], '$.context_group')
        need(test['test_id'] not in ids, 'DUPLICATE_TEST')
        ids.add(test['test_id'])
        need(test['split'] in ('training', 'confirmation'), 'PROTECTED_OR_NONTRAINING_SPLIT')
        seats = test['seats']
        need(type(seats) is list and [s['seat'] for s in seats] in ([0], [1], [0, 1]), 'SEATS')
        for seat in seats:
            R._fields(seat, ('seat', 'actor_context', 'continuation', 'draw_id',
                'sample_seed', 'proposal_seed', 'search'), '$.seat')
            R._int(seat['seat'], '$.seat.seat', 0, 1)
            ctx = seat['actor_context']; state, prefix, _ = R._actor(ctx)
            need(state.step == 0 and ctx['observation']['player'] == seat['seat'] and
                 prefix.last_key == -1 and prefix.edit_count == 0 and not prefix.terminated,
                 'OPENING_EPOCH_ENTRY_REQUIRED')
            R._memory(seat['continuation'], '$.continuation',
                      observation=R._projection_observation(ctx['observation']))
            need({k: v for k, v in seat['continuation'].items() if k != 'game_id'} ==
                 ctx['compiler_memory'], 'CONTINUATION_MISMATCH')
            R._text(seat['draw_id'], '$.draw_id')
            need(seat['draw_id'] not in draws, 'DRAW_ID_REUSE')
            draws.add(seat['draw_id'])
            for k in ('sample_seed', 'proposal_seed'):
                R._int(seat[k], '$.'+k, 0, 2**64-1)
            group_key = (test['context_group_id'], seat['seat'])
            need(group_key not in contexts or contexts[group_key] == digest(ctx), 'CONTEXT_GROUP_DRIFT')
            contexts[group_key] = digest(ctx)
        local = set()
        for phase in PHASES:
            names = test[phase+'_world_ids']
            need(type(names) is list and 0 < len(names) <= 256, 'WORLD_GROUP_CAP')
            for name in names:
                need(name not in local, 'DUPLICATE_OR_OVERLAPPING_WORLD')
                local.add(name)
                world = registry[name]
                family = test['source_families'].get(name)
                R._text(family, '$.source_family')
                need(name not in families or families[name] == family, 'SOURCE_FAMILY_DRIFT')
                families[name] = family
                for key in (('world', world.sha256), ('seed', world.seed), ('family', family)):
                    need(key not in exposure or exposure[key] == phase, 'DISCOVERY_CONFIRMATION_OVERLAP')
                    exposure[key] = phase
                if name not in worlds:
                    worlds[name] = _descriptor(world)
                env = scenarios.ScenarioEnvironment(world.scenario())
                for seat in seats:
                    need(E.observation_facts(env.agent_observation(seat['seat'])) ==
                         seat['actor_context']['observation'], 'ACTUAL_OPENING_CONTEXT_MISMATCH')
        need(set(test['source_families']) == local, 'SOURCE_FAMILY_COVERAGE')
        for seat in seats:
            search = seat['search']
            if search is not None:
                R._fields(search, ('receipt_path', 'protocol', 'budget', 'seed'), '$.search')
                R._text(search['receipt_path'], '$.search.receipt_path')
                caps, _, _ = S._limits(search['budget'])
                need(caps == search['budget'], 'SEARCH_RUNTIME_FIELDS_FORBIDDEN')
                protocol = S._protocol(search['protocol'], caps)
                need(protocol['discovery_world_groups'] == [worlds[w] for w in test['discovery_world_ids']]
                     and protocol['protected_world_groups'] == [worlds[w] for w in test['confirmation_world_ids']]
                     and protocol['source_families'] == test['source_families']
                     and protocol['continuation_sha256'] == digest(seat['continuation'])
                     and protocol['opponent_sha256'] == panel.load_actor(spec['opponent']).sha256,
                     'SEARCH_DISCOVERY_REGISTRATION_MISMATCH')
    archive = R._list(spec['archive'], '$.archive')
    need(len(archive) <= P._budget(spec['proposal_budget'])['max_archive_records'], 'ARCHIVE_BUDGET')
    for row in archive:
        rec = R.validate_pair(row)
        need(rec['identity']['split'] not in ('monitor', 'final'), 'PROTECTED_ARCHIVE')
    starts = sum(len(t['seats']) * (len(t['discovery_world_ids']) *
                 (1+spec['max_discovery_challengers']) + 2*len(t['confirmation_world_ids'])) for t in tests)
    search_starts = sum(seat['search']['budget']['max_simulator_starts']
                        for test in tests for seat in test['seats'] if seat['search'] is not None)
    need(search_starts <= spec['budget']['max_search_starts'], 'UNREGISTERED_SEARCH_START_ALLOWANCE')
    return bind(dict(schema=SCHEMA+'-plan', spec=spec, source_pins=source_pins(), worlds=worlds,
        search_discovery_upper_bound=search_starts, flat_base_upper_bound=starts, flat_replay_upper_bound=starts,
        opening_only=True, objective='TERMINAL_WDL'))


def _atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def _lock(output):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output/'LOCK').open('a+b') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('COLLECTION_BUSY') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class Paused(BaseException):
    """Control flow must escape paired evaluator's ordinary failure handling."""


class Journal:
    """Hash-chained append-only metadata plus immutable content-addressed payloads."""
    def __init__(self, root):
        self.root = root; self.path = root/'JOURNAL.jsonl'
        self.events = []; self.by_key = {}
        if self.path.exists():
            for line in self.path.read_bytes().splitlines(keepends=True):
                need(line.endswith(b'\n'), 'TORN_JOURNAL_REQUIRES_CUSTODY_REVIEW')
                row = R._loads(line.decode())
                bound(row, 'JOURNAL_HASH_MISMATCH')
                need(row['seq'] == len(self.events) and row['previous'] ==
                     (self.events[-1]['content_sha256'] if self.events else None), 'JOURNAL_CHAIN_MISMATCH')
                self._index(row)

    def _index(self, row):
        self.events.append(row)
        self.by_key.setdefault(row['key'], {})[row['event']] = row['payload']

    def append(self, event, key, payload):
        need(event not in self.by_key.get(key, {}), 'DUPLICATE_JOURNAL_EVENT')
        row = bind(dict(seq=len(self.events), previous=self.events[-1]['content_sha256'] if self.events else None,
                        event=event, key=key, payload=payload))
        with self.path.open('ab') as f:
            f.write(wire(row)+b'\n'); f.flush(); os.fsync(f.fileno())
        self._index(row)
        return payload

    def put(self, value):
        key = E.P.wire_hash(value); path = self.root/'objects'/(key+'.json.gz')
        if not path.exists():
            _atomic(path, gzip.compress(wire(value), mtime=0))
        else:
            need(wire(self.get(key)) == wire(value), 'OBJECT_COLLISION')
        return key

    def get(self, key):
        R._hash(key, '$.object')
        value = R._loads(gzip.decompress((self.root/'objects'/(key+'.json.gz')).read_bytes()).decode())
        need(E.P.wire_hash(value) == key, 'OBJECT_HASH_MISMATCH')
        return value

    def once(self, event, key, value):
        prior = self.by_key.get(key, {}).get(event)
        if prior is None:
            self.append(event, key, dict(object=self.put(value)))
        else:
            need(wire(self.get(prior['object'])) == wire(value), 'FROZEN_ARTIFACT_DRIFT')
        return copy.deepcopy(value)

    def call(self, key, *, kind, argument, callback, permit=True, limit=None, lane=None):
        saved = self.by_key.get(key, {})
        intent = dict(kind=kind, argument_sha256=digest(argument), lane=lane)
        if 'INTENT' not in saved:
            self.append('INTENT', key, intent); saved = self.by_key[key]
        need(saved['INTENT'] == intent, 'JOB_ID_ARGUMENT_DRIFT')
        if 'RESULT' in saved:
            return self.get(saved['RESULT']['object'])
        if 'LAUNCH' in saved:
            result = dict(status='UNKNOWN', error='INTERRUPTED_UNKNOWN_OUTCOME', value=None)
        else:
            if not permit:
                raise Paused('REPLAY_CANNOT_LAUNCH_'+kind.upper())
            if limit is not None:
                charged = sum(e['event'] == 'LAUNCH' and e['payload'].get('lane') == lane for e in self.events)
                if charged >= limit:
                    raise Paused('BUDGET_EXHAUSTED_'+lane.upper())
            self.append('LAUNCH', key, dict(kind=kind, lane=lane))
            arg = copy.deepcopy(argument)
            try:
                value = callback(arg)
                need(wire(arg) == wire(argument), 'CALLBACK_MUTATED_INPUT')
                R._safe(value)
                result = dict(status='RETURNED', error=None, value=value)
            except Exception as exc:
                result = dict(status='FAILED', error=type(exc).__name__+': '+str(exc), value=None)
        self.append('RESULT', key, dict(kind=kind, lane=lane, object=self.put(result)))
        return result


def _candidate_challenger(candidate, learner, context, pool_hash, budget):
    value = {k: copy.deepcopy(candidate[k]) for k in GEOMETRY}
    value.update(raw_behavior_logp=None, sampling_provenance=dict(type='search_selection'),
        proposal_family=candidate['proposal_family'], parent_proposal_ids=[learner['direct_draw_id']],
        search_budget=dict(max_candidates=budget['max_candidates'],
                           generated_candidates=budget['max_candidates'], cpu_seconds=0.))
    selection = dict(schema=R.SELECTION_SCHEMA, actor_context_sha256=digest(context),
        parent_checkpoint_sha256=learner['sampling_provenance']['checkpoint_sha256'],
        raw_actions_sha256=digest(value['raw_actions']), continuation_signature=value['continuation_signature'],
        pool_sha256=pool_hash, confirmation_registration_sha256=None)
    value.update(selection_receipt=selection, selection_receipt_sha256=digest(selection))
    return bind(value, 'path_sha256')


def _record(plan, test, seat, learner, challenger, *, suffix='selected'):
    spec = plan['spec']; ctx = seat['actor_context']; memory = seat['continuation']
    return bind(dict(schema=R.SCHEMA,
        identity=dict(pair_id=test['test_id']+'-seat-'+str(seat['seat'])+'-'+suffix,
            source_pins=R.source_pins(), protocol_sha256=R.source_pins()[R.DESIGN_PATH],
            parent_checkpoint_sha256=spec['parent_checkpoint_sha256'], actor_context_sha256=digest(ctx),
            continuation_sha256=digest(memory), source_group_id=test['context_group_id'],
            split=test['split'], round_id=spec['round_id']), actor_context=ctx, continuation=memory,
        learner=learner, challenger=challenger,
        evaluation=dict(mode='UNEVALUATED', discovery_world_groups=[], confirmation_world_groups=[], seat_rows=[],
            engine_hash=R.source_pins()[R.ENGINE_PATH], opponent_hash=panel.load_actor(spec['opponent']).sha256,
            continuation_hash=digest(memory), complete=False, replay_verified=False, attempt_ids=[],
            confirmation_registration=None),
        comparison=dict(objective='TERMINAL_WDL', mean_wdl_delta=None, lower_bound=None, alpha=None,
            independent_world_count=0, verdict='INDETERMINATE', actor_training_admissible=False,
            reason_codes=['UNEVALUATED'])))


class _Runner:
    """Exact branch cache with honest transport/execution separation."""
    def __init__(self, journal, plan, evaluator, test_id, replay_only):
        self.j = journal; self.plan = plan; self.evaluator = evaluator
        self.test_id = test_id; self.replay_only = replay_only
        self.execution = E.execution_binding(evaluator)
        self.phase = 'discovery'

    def key(self, branch):
        req = branch['request']; arm = branch['arm']
        return 'game-'+digest(dict(plan=self.plan['content_sha256'], test=self.test_id,
            world=req['world_sha256'], initial_world=req['initial_world_sha256'],
            continuation_wire_sha256=E.P.wire_hash(req['record']['continuation']),
            seat=req['seat'], phase=branch['phase'], arm=arm,
            actor=E.actor_identity(req, arm), execution=self.execution,
            actor_context=req['record']['identity']['actor_context_sha256'],
            rng=req['rng_binding'], timeout=req['timeout_seconds']))

    def _actual_branch(self, branch):
        value = copy.deepcopy(branch)
        value.update(attempt_id=self.key(branch), execution=self.execution)
        return E.P.bind(value)

    def __call__(self, branch):
        key = self.key(branch); actual = self._actual_branch(branch)
        lane = 'replay' if branch['phase'] == 'REPLAY' else 'base'
        # The unselected partner and pair receipt do not affect this arm. Retain
        # original request; equality of the semantic key authorizes envelope reuse.
        canonical_argument = dict(job_id=key, phase=self.phase)
        if key in self.j.by_key and 'INTENT' in self.j.by_key[key]:
            canonical_argument = self.j.get(self.j.by_key[key]['BRANCH']['object'])['argument']
        if 'BRANCH' not in self.j.by_key.get(key, {}):
            self.j.once('BRANCH', key, dict(branch=actual, argument=canonical_argument))
        original = self.j.get(self.j.by_key[key]['BRANCH']['object'])['branch']
        need(self.key(original) == key, 'CACHED_BRANCH_SEMANTICS_DRIFT')
        def execute(_):
            argument = copy.deepcopy(original)
            response = self.evaluator(argument)
            need(wire(argument) == wire(original), 'EVALUATOR_MUTATED_BRANCH')
            E._validate_response(original, response)
            return response
        result = self.j.call(key, kind='branch', argument=canonical_argument,
            callback=execute, permit=not self.replay_only or lane == 'replay',
            limit=self.plan['spec']['budget']['max_'+lane+'_starts'], lane=lane)
        if actual['content_sha256'] != original['content_sha256']:
            self.j.once('CACHE', 'cache-'+digest([key, actual['content_sha256']]),
                        dict(job_id=key, request_sha256=branch['request']['content_sha256']))
        if result['status'] != 'RETURNED':
            raise RuntimeError(result['error'])
        raw = copy.deepcopy(result['value']); raw['branch_sha256'] = actual['content_sha256']; raw = E.P.bind(raw)
        E._validate_response(actual, raw)
        # Existing evaluate_pair sees a diagnostic transport callback. Authentic
        # native response is checked above and restored by normalize(), not forged.
        raw['branch_sha256'] = branch['content_sha256']
        raw['row'].update(execution_sha256=branch['execution']['content_sha256'],
                          evidence_kind=branch['execution']['kind'], physical_games_completed=0)
        return E.P.bind(raw)

    def normalize(self, request, result):
        def arm(value):
            branch = E.P.bind(dict(schema=E.BRANCH_SCHEMA, request=request, arm=value['arm'],
                execution=result['execution'], phase=value['phase'], attempt_id=value['attempt_id']))
            actual = self._actual_branch(branch); key = self.key(actual)
            saved = self.j.get(self.j.by_key[key]['RESULT']['object'])
            def response(_):
                if saved['status'] != 'RETURNED':
                    raise RuntimeError(saved['error'])
                raw = copy.deepcopy(saved['value']); raw['branch_sha256'] = actual['content_sha256']
                return E.P.bind(raw)
            return E._run(actual, response)
        arms = [arm(a) for a in result['arms']]
        replay = copy.deepcopy(result['replay'])
        replay['arms'] = [arm(a) for a in replay['arms']]
        return E._assemble(request, self.execution, arms, replay)


# The internal transport is the same source when imported or run via python -m.
# Canonicalize only its callable identity; external evaluator pins stay strict.
_Runner.__call__.__module__ = MODULE


def _pair(j, plan, runner, record, world, phase):
    request = E.make_request(record, world_id=world, opponent=plan['spec']['opponent'],
                             timeout_seconds=plan['spec']['timeout_seconds'])
    key = 'pair-'+request['content_sha256']
    runner.phase = phase
    saved = j.by_key.get(key, {})
    if 'BASE' in saved:
        base = j.get(saved['BASE']['object'])
    else:
        base = E.evaluate_pair(request, runner=runner)
        j.once('BASE', key, base)
    if 'VERIFIED' in j.by_key.get(key, {}):
        return j.get(j.by_key[key]['VERIFIED']['object'])
    verified = E.verify_pair(request, base, runner=runner)
    result = runner.normalize(request, verified)
    j.once('VERIFIED', key, result)
    return result


def _validated_pool(pool, plan, seat, learner):
    bound(pool, 'POOL_HASH_MISMATCH', 'pool_sha256')
    need(pool['schema'] == P.SCHEMA and pool['status'] == 'UNSELECTED_POOL', 'POOL_SCHEMA')
    need(pool['actor_context'] == seat['actor_context'] and pool['learner_draw'] == learner and
         pool['parent_checkpoint_sha256'] == plan['spec']['parent_checkpoint_sha256'] and
         pool['seed'] == seat['proposal_seed'] and pool['budget'] == P._budget(plan['spec']['proposal_budget']) and
         pool['source_pins'] == R.source_pins() and pool['generator_source_pins'] ==
         {p: hashlib.sha256((R.ROOT/p).read_bytes()).hexdigest() for p in P.SOURCE_PATHS}, 'POOL_BINDING_MISMATCH')
    need(len(pool['attempts']) == pool['counts']['proposed'] <= pool['budget']['max_candidates'], 'POOL_COUNT')
    state, prefix, options = R._actor(seat['actor_context']); menus = {}; ids = set()
    for candidate in pool['candidates']:
        need(candidate['candidate_id'] not in ids, 'DUPLICATE_CANDIDATE')
        ids.add(candidate['candidate_id'])
        challenger = _candidate_challenger(candidate, learner, seat['actor_context'], pool['pool_sha256'], pool['budget'])
        R._proposal(challenger, dict(parent_checkpoint_sha256=plan['spec']['parent_checkpoint_sha256'],
            actor_context_sha256=digest(seat['actor_context'])), prefix, state, options, menus, direct=False)
    need(any(c['raw_actions'] == learner['raw_actions'] for c in pool['candidates']), 'MISSING_LEARNER_CONTROL')
    return pool


def _preselected(j, plan, test, seat, learner):
    search = seat['search']; path = Path(search['receipt_path'])
    if not path.is_file():
        raise Paused('PRESELECTED_RECEIPT_NOT_AVAILABLE')
    value = R._loads(path.read_text()); bound(value, 'SEARCH_RECEIPT_HASH_MISMATCH')
    caps = search['budget']; protocol = S._protocol(search['protocol'], caps)
    sources = R.source_pins() | {p: hashlib.sha256((R.ROOT/p).read_bytes()).hexdigest() for p in S.SOURCE_PATHS}
    expected = digest(dict(context=seat['actor_context'], learner=learner, protocol=protocol,
                           budget=caps, seed=search['seed'], sources=sources))
    frozen = value['frozen_selection']; selected = value['challenger']
    need(value['schema'] == S.SCHEMA and value['status'] == 'SELECTION_ONLY' and
         value['actor_training_admissible'] is False and value['teacher_weight'] == 0 and
         value['evidence_scope'] == protocol['evidence_scope'] and value['mode'] == protocol['mode'] and
         value['binding'] == expected and value['source_pins'] == sources and
         value['learner_draw'] == learner and value['budget'] == caps and
         frozen['frozen'] is True and frozen['discovery_only'] is True and
         frozen['protocol_sha256'] == digest(protocol) and
         frozen['learner_path_sha256'] == learner['path_sha256'] and
         frozen['discovery_world_groups'] == protocol['discovery_world_groups'] and
         frozen['protected_world_groups'] == protocol['protected_world_groups'] and
         frozen['source_families'] == protocol['source_families'], 'SEARCH_BINDING_MISMATCH')
    receipt = selected['selection_receipt']
    need(receipt['confirmation_registration_sha256'] is None and
         frozen['selection_receipt'] == receipt and
         frozen['selection_receipt_sha256'] == digest(receipt) and
         receipt['pool_sha256'] == digest(dict(binding=value['binding'], candidates=value['candidates'],
             proposal_attempts=value['proposal_attempts'], evaluations=value['evaluations'])),
         'SEARCH_STALE_OR_ALREADY_CONFIRMED')
    state, prefix, options = R._actor(seat['actor_context'])
    R._proposal(selected, dict(parent_checkpoint_sha256=plan['spec']['parent_checkpoint_sha256'],
        actor_context_sha256=digest(seat['actor_context'])), prefix, state, options, {}, direct=False)
    need(all(selected[k] == value['best'][k] for k in GEOMETRY), 'SEARCH_BEST_PATH_MISMATCH')
    need(value['best_score'] == S._score(value['best']['candidate_id'], value['evaluations'], protocol),
         'SEARCH_SCORE_MISMATCH')
    count = value['counts']['attempted_simulator_starts']
    R._int(count, '$.search.starts', 0, caps['max_simulator_starts'])
    need(count == len(value['evaluations']), 'SEARCH_START_COUNT_MISMATCH')
    groups = {g['source_group_id']: g for g in protocol['discovery_world_groups']}
    candidates = {c['candidate_id']: c for c in value['candidates']}
    need(len(candidates) == len(value['candidates']) and
         candidates.get(value['best']['candidate_id']) == value['best'] and
         frozen['full_continuation_signature'] == value['best']['full_continuation_signature'],
         'SEARCH_CANDIDATE_BINDING_MISMATCH')
    cells = set()
    for index, event in enumerate(value['evaluations']):
        request = event['request']; group = event['group_id']; candidate = event['candidate_id']
        bound(request, 'SEARCH_REQUEST_HASH_MISMATCH', 'request_sha256')
        need(group in groups and candidate in candidates and (candidate, group) not in cells,
             'SEARCH_EVALUATION_COVERAGE_MISMATCH')
        cells.add((candidate, group))
        need(request['schema'] == S.SCHEMA+'-request' and request['purpose'] == 'DISCOVERY' and
             request['execution_context'] == protocol['execution_context'] and
             request['group'] == groups[group] and
             request['candidate'] == S._visible(candidates[candidate]) and
             request['actor_context'] == seat['actor_context'] and request['seat'] == seat['seat'] and
             request['opponent_sha256'] == protocol['opponent_sha256'] and
             request['continuation_sha256'] == protocol['continuation_sha256'] and
             request['source_pins'] == sources and
             event['attempt_id'] == request['attempt_id'] == digest([expected, index]) and
             event['request_sha256'] == request['request_sha256'], 'SEARCH_EVALUATION_BINDING_MISMATCH')
        if event['status'] == 'COMPLETE':
            need(S._validate_response(request, event['response'])[0] == 'COMPLETE',
                 'SEARCH_COMPLETION_MISMATCH')
    key = 'search-'+seat['draw_id']
    charged = sum(e['payload'].get('starts', 0) for e in j.events if e['event'] == 'SEARCH')
    if 'SEARCH' not in j.by_key.get(key, {}):
        need(charged+count <= plan['spec']['budget']['max_search_starts'], 'SEARCH_START_BUDGET')
        j.append('SEARCH', key, dict(starts=count, receipt_sha256=value['content_sha256'], object=j.put(value)))
    else:
        need(j.by_key[key]['SEARCH']['receipt_sha256'] == value['content_sha256'], 'SEARCH_RECEIPT_DRIFT')
    return selected, value


def _mean(rows, test, arm):
    if len(rows) != len(test['discovery_world_ids']) or any(
            r['arms'][arm]['status'] != 'COMPLETE' or r['replay']['status'] != 'VERIFIED' for r in rows.values()):
        return None
    families = {}
    for world, row in rows.items():
        families.setdefault(test['source_families'][world], []).append(row['arms'][arm]['terminal']['wdl'])
    return math.fsum(math.fsum(v)/len(v) for v in families.values())/len(families)


def _selection(j, plan, test, seat, learner, pool, runner):
    key = 'selection-'+seat['draw_id']
    if 'FREEZE' in j.by_key.get(key, {}):
        frozen = j.get(j.by_key[key]['FREEZE']['object'])
        if seat['search'] is not None:
            _preselected(j, plan, test, seat, learner)  # stale/tampered file refuses even on resume
        if seat['search'] is None:
            return frozen
    elif seat['search'] is not None:
        selected, search = _preselected(j, plan, test, seat, learner)
        frozen = dict(challenger=selected, pool_sha256=pool['pool_sha256'], discovery={},
            winner_is_learner=selected['raw_actions'] == learner['raw_actions'],
            selection_provenance=dict(kind='ITERATIVE_SEARCH_SELECTION_ONLY', receipt_sha256=search['content_sha256']))
    else:
        candidates = [c for c in pool['candidates'] if c['raw_actions'] != learner['raw_actions']]
        candidates = candidates[:plan['spec']['max_discovery_challengers']]
        results = []; scores = []; learner_score = None
        for candidate in candidates:
            selected = _candidate_challenger(candidate, learner, seat['actor_context'], pool['pool_sha256'], pool['budget'])
            record = _record(plan, test, seat, learner, selected, suffix=candidate['candidate_id'])
            rows = {w: _pair(j, plan, runner, record, w, 'discovery') for w in test['discovery_world_ids']}
            results.append((selected, rows)); scores.append(_mean(rows, test, 1))
            if learner_score is None:
                learner_score = _mean(rows, test, 0)
        need(candidates or pool['candidates'], 'EMPTY_CANDIDATE_POOL')
        # Missing evidence never selects by a favorable observed subset.
        if candidates and (learner_score is None or any(s is None for s in scores)):
            raise Paused('INDETERMINATE_DISCOVERY_FAILURE')
        winner = next(c for c in pool['candidates'] if c['raw_actions'] == learner['raw_actions'])
        selected = _candidate_challenger(winner, learner, seat['actor_context'], pool['pool_sha256'], pool['budget'])
        best, rows, learner_wins = learner_score, {}, True
        for score, (challenger, observed) in zip(scores, results):
            if best is not None and score > best:
                best, selected, rows, learner_wins = score, challenger, observed, False
        frozen = dict(challenger=selected, pool_sha256=pool['pool_sha256'], discovery=rows,
            winner_is_learner=learner_wins, selection_provenance=dict(kind='FLAT_TERMINAL_WDL',
                learner_score=learner_score, candidate_scores=scores, untested_are_bad=False))
    j.once('FREEZE', key, frozen)
    if seat['search'] is not None:
        # Search traces contain terminal discovery attestations, not the paired
        # owner's exact engine/replay receipts. Audit the already-frozen path;
        # never reselect it on these results or manufacture missing native rows.
        record = _record(plan, test, seat, learner, frozen['challenger'], suffix='search-audit')
        frozen['discovery'] = {w: _pair(j, plan, runner, record, w, 'discovery')
                               for w in test['discovery_world_ids']}
        if any(_mean(frozen['discovery'], test, arm) is None for arm in (0, 1)):
            raise Paused('INDETERMINATE_SEARCH_AUDIT')
    return frozen


def _registration(plan, learner, challenger, test):
    spec = plan['spec']
    return dict(schema=R.REGISTRATION_SCHEMA, family_alpha=spec['family_alpha'], tested_pairs=len(spec['tests']),
        alpha_pair=spec['family_alpha']/len(spec['tests']), min_effect=spec['min_effect'],
        confirmation_source_group_ids=test['confirmation_world_ids'], learner_path_sha256=learner['path_sha256'],
        challenger_actions_sha256=digest(challenger['raw_actions']),
        independence_attested=spec['independence_attested'], compatible_contexts_attested=spec['compatible_contexts_attested'],
        frozen_before_confirmation=True)


def _aggregate(plan, test, base, rows):
    rec = copy.deepcopy(base); spec = plan['spec']; ev = rec['evaluation']
    reg = _registration(plan, rec['learner'], rec['challenger'], test)
    rec['challenger']['selection_receipt']['confirmation_registration_sha256'] = digest(reg)
    rec['challenger']['selection_receipt_sha256'] = digest(rec['challenger']['selection_receipt'])
    rec['challenger'] = bind(rec['challenger'], 'path_sha256')
    ev.update(mode='CONFIRMED_EXPECTATION', confirmation_registration=reg,
        discovery_world_groups=[plan['worlds'][w] for w in test['discovery_world_ids']],
        confirmation_world_groups=[plan['worlds'][w] for w in test['confirmation_world_ids']],
        seat_rows=[], attempt_ids=[])
    for world, result in rows.items():
        row = copy.deepcopy(result['record']['evaluation']['seat_rows'][0])
        row['source_group_id'] = world
        # Diagnostic replay is useful software evidence, never an actor-label attestation.
        row['replay_verified'] = bool(row['replay_verified'] and result['execution']['kind'] == 'NATIVE_GAME')
        ev['seat_rows'].append(row)
        ev['attempt_ids'].extend(result['record']['evaluation']['attempt_ids'])
    ev['complete'] = (len(rows) == len(ev['discovery_world_groups'])+len(ev['confirmation_world_groups'])
                      and all(row['complete'] for row in ev['seat_rows']))
    ev['replay_verified'] = ev['complete'] and all(row['replay_verified'] for row in ev['seat_rows'])
    chosen = [r['challenger_wdl']-r['learner_wdl'] for r in ev['seat_rows']
              if r['source_group_id'] in test['confirmation_world_ids'] and r['complete']]
    mean = math.fsum(chosen)/len(chosen) if chosen else None
    lower = mean-math.sqrt(8*-math.log(reg['alpha_pair'])/len(chosen)) if len(chosen) == len(test['confirmation_world_ids']) else None
    rec['comparison'].update(mean_wdl_delta=mean, lower_bound=lower, alpha=reg['alpha_pair'],
        independent_world_count=len(chosen), verdict='INDETERMINATE', actor_training_admissible=False,
        reason_codes=['AWAITING_ADMISSION' if ev['replay_verified'] else 'MISSING_NATIVE_VERIFIED_EVIDENCE'])
    # The rank-corpus guard consumes this marker, not the collector's counters.
    if any(result['execution']['kind'] == 'SYNTHETIC_DIAGNOSTIC' for result in rows.values()):
        rec['comparison']['reason_codes'].append('SYNTHETIC_DIAGNOSTIC')
    return R.validate_pair(bind(rec))


def _protocol(plan, pending):
    spec = plan['spec']; tests = []
    for test, seats in pending:
        locks = [A._lock(_aggregate(plan, test, item['record'], {})) for item in seats]
        tests.append(dict(test_id=test['test_id'], seats=[s['seat']['seat'] for s in seats],
            n_world_groups=len({test['source_families'][w] for w in test['confirmation_world_ids']}),
            discovery_world_groups=[plan['worlds'][w] for w in test['discovery_world_ids']],
            confirmation_world_groups=[plan['worlds'][w] for w in test['confirmation_world_ids']],
            source_families=test['source_families'], seat_locks=locks))
    return bind(dict(schema=A.PROTOCOL_SCHEMA, design_sha256=R.source_pins()[R.DESIGN_PATH],
        family_id=spec['family_id'], round_id=spec['round_id'], family_alpha=spec['family_alpha'],
        tested_pairs=len(spec['tests']), alpha_pair=spec['family_alpha']/len(spec['tests']), min_effect=spec['min_effect'],
        frozen_before_confirmation=True, independence_attested=spec['independence_attested'],
        compatible_contexts_attested=spec['compatible_contexts_attested'], tests=tests))


def _counts(j, plan):
    out = dict(base_starts=0, replay_starts=0, confirmation_base_starts=0, search_discovery_starts=0,
        native_starts=0, unique_worlds=0, replay_independent_worlds_added=0,
        direct_policy_draws=0, sampler_calls=0, proposal_calls=0, cache_hits=0, unknown_launches=0,
        failed_calls=0, optimizer_updates=0, pair_opportunities=len(plan['spec']['tests']))
    worlds = set()
    for key, events in j.by_key.items():
        if 'SEARCH' in events:
            out['search_discovery_starts'] += events['SEARCH']['starts']
        if 'CACHE' in events:
            out['cache_hits'] += 1
        if 'LAUNCH' not in events:
            continue
        kind = events['LAUNCH']['kind']
        if kind == 'branch':
            lane = events['LAUNCH']['lane']; out[lane+'_starts'] += 1
            obj = j.get(events['BRANCH']['object']); branch = obj['branch']
            worlds.add(branch['request']['world_sha256'])
            out['confirmation_base_starts'] += lane == 'base' and obj['argument']['phase'] == 'confirmation'
            out['native_starts'] += branch['execution']['kind'] == 'NATIVE_GAME'
        if kind == 'sample':
            out['sampler_calls'] += 1
        if kind == 'propose':
            out['proposal_calls'] += 1
        if 'RESULT' not in events:
            out['unknown_launches'] += 1
        else:
            result = j.get(events['RESULT']['object'])
            out['unknown_launches'] += result['status'] == 'UNKNOWN'
            out['failed_calls'] += result['status'] == 'FAILED'
            out['direct_policy_draws'] += kind == 'sample' and result['status'] == 'RETURNED'
    out['unique_worlds'] = len(worlds)
    return out


def _paired_corpus(j, plan, records, admissions, reasons):
    """Join one closed round from its own journal; never promote archive rows.

    Returns the trainer's existing closed corpus and report-only provenance.
    INDETERMINATE complete evidence (including diagnostic replay or low power)
    stays zero-weight. Missing seats/outcomes and invalid/censored evidence do
    not become a fixed fit buffer. This reads no checkpoint and invokes no
    sampler, proposer, evaluator, optimizer or replay.
    """
    need(not reasons, 'PAIRED_ROUND_PARTIAL')
    bound(plan, 'PAIRED_PLAN_HASH_MISMATCH')
    need(plan == register_plan(plan['spec']), 'PAIRED_PLAN_SOURCE_OR_PROTOCOL_DRIFT')
    saved = Journal(j.root)  # authenticate the durable chain, not a mutable caller view
    need(saved.events == j.events, 'PAIRED_JOURNAL_VIEW_DRIFT')
    need(len({(e['event'], e['key']) for e in saved.events}) == len(saved.events),
         'PAIRED_DUPLICATE_JOURNAL_EVENT')

    def retained(event, key):
        need(event in saved.by_key.get(key, {}), 'PAIRED_MISSING_' + event)
        return saved.get(saved.by_key[key][event]['object'])

    saved_plan = R._loads((j.root/'PLAN.json').read_text())
    need(wire(saved_plan) == wire(plan), 'PAIRED_SAVED_PLAN_DRIFT')
    need(retained('REGISTER', 'round')['plan_sha256'] == plan['content_sha256'],
         'PAIRED_REGISTER_PLAN_MISMATCH')
    protocol = retained('PROTOCOL', 'family')
    tests = A._protocol(protocol)
    # Reconstruct locks from the original draw/selection, not from export-side
    # pairs. Re-sealing a different family, parent or seat set cannot relabel it.
    pending = []
    for test in plan['spec']['tests']:
        items = []
        for seat in test['seats']:
            draw = retained('RESULT', 'sample-' + seat['draw_id'])
            need(draw['status'] == 'RETURNED', 'PAIRED_DRAW_NOT_RETURNED')
            learner = draw['value']
            need(learner['direct_draw_id'] == seat['draw_id'], 'PAIRED_DRAW_ID_MISMATCH')
            frozen = retained('FREEZE', 'selection-' + seat['draw_id'])
            items.append(dict(seat=seat, record=_record(plan, test, seat, learner, frozen['challenger'])))
        pending.append((test, items))
    need(protocol == _protocol(plan, pending), 'PAIRED_FROZEN_PROTOCOL_DRIFT')
    positions = {(e['event'], e['key']): e['seq'] for e in saved.events}
    freeze_at = positions['PROTOCOL', 'family']
    need(all(positions['FREEZE', 'selection-' + seat['draw_id']] < freeze_at
             for test in plan['spec']['tests'] for seat in test['seats']),
         'PAIRED_PROTOCOL_ORDER_MISMATCH')

    grouped = {test_id: {} for test_id in tests}
    for test_id, record in records:
        need(test_id in grouped, 'PAIRED_UNREGISTERED_TEST')
        row = R.validate_pair(record)  # rejected geometry is still real V0 geometry
        seat = row['actor_context']['observation']['player']
        need(seat not in grouped[test_id], 'PAIRED_DUPLICATE_SEAT')
        need(row['identity']['parent_checkpoint_sha256'] == plan['spec']['parent_checkpoint_sha256'],
             'PAIRED_REFERENCE_PARENT_MISMATCH')
        grouped[test_id][seat] = row
    supplied = {}
    for receipt in admissions:
        test_id = receipt['test_id']
        need(test_id in tests, 'PAIRED_UNREGISTERED_ADMISSION')
        need(test_id not in supplied, 'PAIRED_DUPLICATE_TEST')
        supplied[test_id] = receipt
    need(set(supplied) == set(tests), 'PAIRED_ADMISSION_COVERAGE_MISMATCH')
    need({key for key, events in saved.by_key.items() if 'ADMISSION' in events} == set(tests),
         'PAIRED_JOURNAL_ADMISSION_COVERAGE_MISMATCH')

    opportunities = []
    for test_id, test in tests.items():
        need(bool(grouped[test_id]), 'PAIRED_MISSING_TEST')
        need(set(grouped[test_id]) == set(test['seats']), 'PAIRED_SEAT_COVERAGE_MISMATCH')
        rows = [grouped[test_id][seat] for seat in test['seats']]
        need([A._lock(row) for row in rows] == test['seat_locks'], 'PAIRED_SEAT_LOCK_MISMATCH')
        pair = dict(schema=A.BUNDLE_SCHEMA, test_id=test_id, records=rows)
        admission = A.admit_pair(pair, protocol=protocol)
        need(admission == supplied[test_id], 'PAIRED_ADMISSION_RECEIPT_MISMATCH')
        need(admission == retained('ADMISSION', test_id), 'PAIRED_JOURNAL_ADMISSION_MISMATCH')
        need(positions['ADMISSION', test_id] > freeze_at, 'PAIRED_ADMISSION_ORDER_MISMATCH')
        need(admission['status'] not in ('INVALID', 'CENSORED', 'UNSUPPORTED'),
             'PAIRED_ADMISSION_REJECTED:' + ','.join(admission['reason_codes']))
        opportunities.append(dict(pair=pair, admission=admission))

    # Import the schema owner only on the completed path. No second format or
    # trainer validation exception is introduced; provenance lives in RESULT.
    from farm_neat.training.v0_regret_train import PAIRED_SCHEMA
    pins = source_pins() | {name: hashlib.sha256((R.ROOT/name).read_bytes()).hexdigest()
        for name in ('src/farm_neat/training/v0_regret_train.py',
                     'src/farm_neat/training/v0_regret_admission.py')}
    return dict(schema=PAIRED_SCHEMA, protocol=protocol, opportunities=opportunities), dict(
        status='READY', reason_codes=[], protocol_sha256=protocol['content_sha256'],
        registered_tests=len(tests), seat_context_opportunities=sum(len(t['seat_locks']) for t in tests.values()),
        parent_checkpoint_sha256=plan['spec']['parent_checkpoint_sha256'], source_pins=pins,
        plan_file_sha256=hashlib.sha256((j.root/'PLAN.json').read_bytes()).hexdigest(),
        journal_file_sha256=hashlib.sha256(j.path.read_bytes()).hexdigest())


def _export(j, plan, records, admissions, reasons, command):
    accepted_ids = {a['test_id'] for a in admissions if a.get('actor_training_admissible', False)}
    accepted, rejected = [], []
    for test_id, record in records:
        (accepted if test_id in accepted_ids else rejected).append(record)
    seen = {record['content_sha256'] for record in accepted+rejected}
    for events in j.by_key.values():
        retained = events.get('VERIFIED') or events.get('BASE') or events.get('DRAW_RECORD')
        if retained is not None:
            value = j.get(retained['object'])
            record = value.get('record', value)
            if record['content_sha256'] not in seen:
                rejected.append(record); seen.add(record['content_sha256'])
    # A refused join still writes the existing archive/resume outputs below.
    try:
        corpus, paired_export = _paired_corpus(j, plan, records, admissions, reasons)
    except (ValueError, A._Refusal, KeyError, TypeError, OSError, ImportError) as exc:
        code = getattr(exc, 'code', None) or (str(exc) if isinstance(exc, ValueError)
                else 'PAIRED_EVIDENCE_UNAVAILABLE:' + type(exc).__name__)
        corpus = None
        paired_export = dict(status='REFUSED', reason_codes=[code])
    snapshot = digest(dict(journal=j.events[-1]['content_sha256'], records=records, admissions=admissions, reasons=reasons, command=command, paired_export=paired_export))
    target = j.root/'exports'/snapshot
    artifacts = {}
    for name, rows in (('accepted', accepted), ('rejected', rejected), ('archive', accepted+rejected),
                       ('admissions', admissions)):
        path = target/(name+'.jsonl')
        data = b''.join(wire(r)+b'\n' for r in rows)
        if path.exists():
            need(path.read_bytes() == data, 'EXPORT_DRIFT')
        else:
            _atomic(path, data)
        artifacts[name] = str(path)
    if corpus is not None:
        path = target/'paired-corpus.json'
        data = wire(corpus)+b'\n'
        if path.exists():
            need(path.read_bytes() == data, 'EXPORT_DRIFT')
        else:
            _atomic(path, data)
        artifacts['paired_corpus'] = str(path)
    counts = _counts(j, plan)
    report = bind(dict(schema=SCHEMA, status='PARTIAL' if reasons else 'COMPLETE',
        plan_sha256=plan['content_sha256'], journal_head=j.events[-1]['content_sha256'],
        paired_export=paired_export, counts=counts, accepted_pairs=len(accepted_ids), admissions=admissions, artifacts=artifacts,
        artifact_sha256s={k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in artifacts.items()},
        reason_codes=reasons, unseen_candidates_are_bad=False, next_command=command['text'],
        next_command_kind=command['kind'], archive_records=len(accepted)+len(rejected),
        corpus_unit='SEAT_TUPLE_ADMISSION_JOIN_NOT_PER_SEAT_OVERRIDE',
        limits=['OPENING_ONLY_NO_MODEL_FIT', 'INJECTED_RUNNERS_NEVER_ADMIT_ACTOR_LABELS',
                'AT_MOST_ONCE_LAUNCH_UNKNOWN_IS_CHARGED_NOT_RETRIED',
                'HASHES_DO_NOT_AUTHENTICATE_CALLBACKS_OR_CROSS_DIRECTORY_CUSTODY',
                'FIXED_SHOP_REGISTERED_WORLDS_NOT_ORIGINAL_RULE_STRENGTH']))
    _atomic(target/'RESULT.json', wire(report)+b'\n')
    _atomic(j.root/'RESULT.json', wire(report)+b'\n')
    return report


def _collect(plan, *, sampler, proposer, evaluator, output, replay_only=False, callbacks_ref=None):
    bound(plan, 'PLAN_HASH_MISMATCH')
    need(plan == register_plan(plan['spec']), 'PLAN_SOURCE_OR_PROTOCOL_DRIFT')
    need(all(callable(c) for c in (sampler, proposer, evaluator)), 'CALLBACK_REQUIRED')
    output = Path(output).resolve(); spec = plan['spec']
    binding = dict(plan_sha256=plan['content_sha256'], execution=E.execution_binding(evaluator),
        callbacks={name: panel._fingerprint(call) for name, call in
                   [('sampler', sampler), ('proposer', proposer), ('evaluator', evaluator)]})
    with _lock(output):
        j = Journal(output)
        j.once('REGISTER', 'round', binding)
        if (output/'PLAN.json').exists():
            need(wire(R._loads((output/'PLAN.json').read_text())) == wire(plan), 'SAVED_PLAN_DRIFT')
        else:
            _atomic(output/'PLAN.json', wire(plan)+b'\n')
        records, admissions, reasons, pending = [], [], [], []
        try:
            for test in spec['tests']:
                items = []
                runner = _Runner(j, plan, evaluator, test['test_id'], replay_only)
                for seat in test['seats']:
                    ctx = seat['actor_context']
                    result = j.call('sample-'+seat['draw_id'], kind='sample',
                        argument=dict(context=ctx, draw_id=seat['draw_id'], seed=seat['sample_seed'],
                                      parent_checkpoint_sha256=spec['parent_checkpoint_sha256']),
                        callback=lambda a: sampler(a['context'], **{k: v for k, v in a.items() if k != 'context'}), permit=not replay_only)
                    if result['status'] != 'RETURNED':
                        raise Paused('LEARNER_DRAW_FAILED_NO_REROLL')
                    learner = result['value']; state, prefix, options = R._actor(ctx)
                    need(learner['direct_draw_id'] == seat['draw_id'], 'DIRECT_DRAW_ID_MISMATCH')
                    R._proposal(learner, dict(parent_checkpoint_sha256=spec['parent_checkpoint_sha256']),
                                prefix, state, options, {}, direct=True)
                    seed_candidate = {k: learner[k] for k in GEOMETRY}
                    seed_candidate['proposal_family'] = 'retained_learner'
                    control = _candidate_challenger(seed_candidate, learner, ctx,
                        digest(dict(unselected_learner_path=learner['path_sha256'])), dict(max_candidates=1))
                    seed_record = R.validate_pair(_record(plan, test, seat, learner, control, suffix='retained-learner'))
                    j.once('DRAW_RECORD', 'draw-record-'+seat['draw_id'], seed_record)
                    result = j.call('propose-'+seat['draw_id'], kind='propose',
                        argument=dict(context=ctx, learner=learner, budget=spec['proposal_budget'],
                                      seed=seat['proposal_seed'], archive=spec['archive']),
                        callback=lambda a: proposer(a['context'], a['learner'], **{k: v for k, v in a.items() if k not in ('context', 'learner')}), permit=not replay_only)
                    if result['status'] != 'RETURNED':
                        raise Paused('PROPOSER_FAILED_NO_REROLL')
                    pool = _validated_pool(result['value'], plan, seat, learner)
                    frozen = _selection(j, plan, test, seat, learner, pool, runner)
                    rec = _record(plan, test, seat, learner, frozen['challenger'])
                    R.validate_pair(rec)
                    items.append(dict(seat=seat, learner=learner, frozen=frozen, record=rec, runner=runner))
                    records.append((test['test_id'], rec))
                pending.append((test, items))
            protocol = _protocol(plan, pending)
            # Validate without weakening declarations. False attestations remain an
            # INDETERMINATE gate; no external games are justified by them.
            if spec['independence_attested'] and spec['compatible_contexts_attested']:
                A._protocol(protocol)
            j.once('PROTOCOL', 'family', protocol)
            records = []
            for test, items in pending:
                skip = all(item['frozen']['winner_is_learner'] for item in items)
                bundle = []
                for item in items:
                    rows = item['rows'] = copy.deepcopy(item['frozen']['discovery'])
                    if not skip:
                        for world in test['confirmation_world_ids']:
                            rows[world] = _pair(j, plan, item['runner'], item['record'], world, 'confirmation')
                    rec = _aggregate(plan, test, item['record'], rows)
                    records.append((test['test_id'], rec)); bundle.append(rec)
                admission = A.admit_pair(dict(schema=A.BUNDLE_SCHEMA, test_id=test['test_id'], records=bundle), protocol=protocol)
                admissions.append(admission)
                j.once('ADMISSION', test['test_id'], admission)
        except Paused as exc:
            reasons.append(str(exc))
            records = [(test['test_id'], _aggregate(plan, test, item['record'],
                        item.get('rows', item['frozen']['discovery'])))
                       for test, items in pending for item in items]
        if callbacks_ref:
            verb = 'run' if reasons else 'replay'
            command = dict(kind='CLI_SHELL', text=f'PYTHONPATH=src python3 -B -m {MODULE} {verb} --plan {shlex.quote(str(output/"PLAN.json"))} --output {shlex.quote(str(output))} --callbacks {shlex.quote(callbacks_ref)}')
        else:
            command = dict(kind='PYTHON_IN_CALLER_CALLBACK_SCOPE',
                text=f'collect_round(plan, sampler=sampler, proposer=proposer, evaluator=evaluator, output={str(output)!r})')
        return _export(j, plan, records, admissions, reasons, command)


def collect_round(plan: dict, *, sampler, proposer, evaluator, output) -> dict:
    """Run/resume the registered round; all decisions derive from durable jobs."""
    return _collect(plan, sampler=sampler, proposer=proposer, evaluator=evaluator, output=output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan'); p.add_argument('--spec', required=True); p.add_argument('--output', required=True)
    for name in ('run', 'replay'):
        p = sub.add_parser(name); p.add_argument('--plan', required=True)
        p.add_argument('--output', required=True); p.add_argument('--callbacks', required=True)
    p = sub.add_parser('recovery-harvest', help='explicit complete legal source replay; no search or training')
    p.add_argument('--recipe', required=True); p.add_argument('--output', required=True)
    p.add_argument('--callbacks', help='trusted module:factory returning actor_factory/opponent_factory')
    p = sub.add_parser('recovery-collect', help='replay verified sources and search only registered training sources')
    p.add_argument('--spec', required=True); p.add_argument('--output', required=True); p.add_argument('--callbacks', required=True)
    p = sub.add_parser('recovery-validate', help='data-only receipt admission; no callbacks, games or updates')
    p.add_argument('--corpus', required=True)
    for name in ('belief-freeze', 'belief-join'):
        p = sub.add_parser(name, help='assemble/freeze retained native belief receipts; zero callbacks/games')
        p.add_argument('--spec',required=True); p.add_argument('--spec-sha256',required=True)
        p.add_argument('--output',required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'belief-join':
            spec = _belief_join_read(dict(path=args.spec, sha256=args.spec_sha256), 'spec')
            population = join_belief_production_population(spec)
            result = write_belief_production_population(args.output, population)
        elif args.command == 'belief-freeze':
            path=Path(args.spec)
            need(hashlib.sha256(path.read_bytes()).hexdigest()==args.spec_sha256,'BELIEF_SPEC_FILE_HASH')
            spec=R._loads(path.read_text())
            need(type(spec) is dict and set(spec)=={'records','actor_map','parent_checkpoint_sha256',
                 'callback_provenance','protected_groups'},'BELIEF_SPEC_FIELDS')
            population=freeze_belief_production_population(**spec,diagnostic=False)
            result=write_belief_production_population(args.output,population)
        elif args.command.startswith('recovery-'):
            result = _recovery_cli(args)
        elif args.command == 'plan':
            result = register_plan(R._loads(Path(args.spec).read_text()))
            # Exclusive creation, not an implicit replacement registration.
            with Path(args.output).open('xb') as f:
                f.write(wire(result)+b'\n'); f.flush(); os.fsync(f.fileno())
        else:
            plan = R._loads(Path(args.plan).read_text())
            module, name = args.callbacks.split(':', 1)
            factory = getattr(importlib.import_module(module), name)
            cb = factory(copy.deepcopy(plan['spec'].get('callback_config', {})))
            need(set(cb) == {'sampler', 'proposer', 'evaluator'}, 'CALLBACK_FACTORY_FIELDS')
            result = _collect(plan, **cb, output=args.output, replay_only=args.command == 'replay', callbacks_ref=args.callbacks)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps(dict(status='INVALID', reason=str(exc), error_type=type(exc).__name__)))
        return 2



def _recovery_cli(args):
    from farm_neat.rollout import v0_regret_dawn as D
    def callbacks(config):
        if not args.callbacks: return {}
        module, name = args.callbacks.split(':', 1)
        return getattr(importlib.import_module(module), name)(copy.deepcopy(config))
    if args.command == 'recovery-validate':
        corpus = R._loads(Path(args.corpus).read_text())
        return dict(status='VALIDATED_NOT_TRAINED', rows=validate_recovery_corpus(corpus),
                    corpus_sha256=corpus['content_sha256'], optimizer_updates=0, games_run=0)
    # Exclusive INTENT before any potentially expensive callback/replay. Failed
    # attempts leave their file reserved: never mistake a retry for fresh data.
    output = Path(args.output)
    with output.open('xb') as handle:
        handle.write(b'{"status":"INTENT_NOT_A_CORPUS"}\n'); handle.flush(); os.fsync(handle.fileno())
        if args.command == 'recovery-harvest':
            recipe = R._loads(Path(args.recipe).read_text())
            _recovery_recipe(recipe)
            cb = callbacks({})
            need(set(cb) <= {'actor_factory','opponent_factory'}, 'RECOVERY_FACTORY_FIELDS')
            result = harvest_recovery_source(recipe, **cb)
        else:
            spec = R._loads(Path(args.spec).read_text())
            D._bound(spec, 'recovery collection spec')
            D._closed(spec, ('sources','requests','family_alpha','callback_config','content_sha256'), 'recovery collection spec')
            cb = callbacks(spec['callback_config'])
            need({'proposer','evaluator'} <= set(cb) <=
                 {'proposer','evaluator','actor_factory','opponent_factory'}, 'RECOVERY_FACTORY_FIELDS')
            result = collect_recovery(spec['sources'], requests=spec['requests'],
                family_alpha=spec['family_alpha'], **cb)
        payload = (D.P.wire(result)+'\n').encode()
        handle.seek(0); handle.write(payload); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
    return dict(status='RECOVERY_SOURCE_WRITTEN' if args.command == 'recovery-harvest' else 'RECOVERY_CORPUS_WRITTEN',
        path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        content_sha256=result['content_sha256'], optimizer_updates=0, strength_claim=False)


# Dawn recovery stays separate from opening pair-v1; its replay and source labels
# are trainer-only. No input in this section changes the native V0 vocabulary.
RECOVERY_SOURCE_SCHEMA = 'farm-v0-recovery-source-v1'
RECOVERY_CORPUS_SCHEMA = 'farm-v0-recovery-curriculum-v1'
RECOVERY_KINDS = ('NATURAL_BAD', 'LEGALLY_DEGRADED')


def _recovery_recipe(recipe):
    from farm_neat.rollout import v0_regret_dawn as D
    from farm_neat.economy.v0 import vocabulary as V
    D._bound(recipe, 'recovery recipe')
    D._closed(recipe, ('schema', 'source_id', 'source_family_id', 'split', 'kind',
        'scenario', 'seat', 'game_id', 'actor_policy', 'opponent_policy',
        'checkpoint_sha256', 'target_step', 'alternative', 'evidence_scope',
        'content_sha256'), 'recovery recipe')
    need(recipe['schema'] == RECOVERY_SOURCE_SCHEMA+'-recipe', 'RECOVERY_RECIPE_SCHEMA')
    for key in ('source_id', 'source_family_id', 'game_id'):
        R._text(recipe[key], '$.recipe.'+key)
    R._hash(recipe['checkpoint_sha256'], '$.recipe.checkpoint_sha256')
    need(recipe['split'] in ('training', 'heldout'), 'RECOVERY_SOURCE_SPLIT')
    need(recipe['kind'] in RECOVERY_KINDS, 'REPLAY_SOURCE_KIND_REQUIRED')
    need(type(recipe['seat']) is int and recipe['seat'] in (0, 1), 'RECOVERY_SEAT')
    step = recipe['target_step']
    need(type(step) is int and 0 < step <= V.LAST_STEP and step % 24 == 0, 'RECOVERY_DAWN_REQUIRED')
    D._check_policy(recipe['actor_policy']); D._check_policy(recipe['opponent_policy'], opponent=True)
    need(recipe['evidence_scope'] in ('INJECTED_FIXTURE', 'NATIVE_GAME'), 'RECOVERY_SCOPE')
    if recipe['evidence_scope'] == 'NATIVE_GAME':
        actor = recipe['actor_policy']
        need(actor['kind'] == 'SCORER', 'NATURAL_SOURCE_REQUIRES_DIRECT_SCORER')
        identity = actor['state']['state']['identity']
        need(identity.get('commit_sha256') == recipe['checkpoint_sha256'] and
             actor['state']['state']['mode'] == 'sample_raw', 'SOURCE_PARENT_OR_DIRECT_DRAW_MISMATCH')
    alternative = recipe['alternative']
    if recipe['kind'] == 'NATURAL_BAD':
        need(alternative is None, 'NATURAL_SOURCE_CANNOT_CONTAIN_INTERVENTION')
    else:
        D._closed(alternative, ('step', 'raw_actions'), 'legal earlier alternative')
        need(type(alternative['step']) is int and 0 < alternative['step'] < step and
             alternative['step'] % 24 == 0, 'EARLIER_LEGAL_DAWN_REQUIRED')
        need(type(alternative['raw_actions']) is list, 'CANONICAL_EDIT_PATH_REQUIRED')
    return recipe



def _recovery_selection(snapshot, terminal, opening_cash):
    from farm_neat.rollout import v0_regret_dawn as D
    from farm_neat.economy.v0 import vocabulary as V
    facts = D.recovery_actor_context(snapshot)['observation']
    deficit = max(0., opening_cash-facts['cash'])
    return dict(after_complete_game=True, bad_state=terminal['terminal_wdl'] < 0 or deficit > 0,
        rule='TERMINAL_LOSS_OR_OBSERVED_CASH_BELOW_OPENING_V1', cash_deficit_proxy=deficit,
        remaining_steps=V.LAST_STEP+1-snapshot['step'], observed_hands=facts['hands_today'],
        observed_stock_units=sum(facts['stock'].values()))


def harvest_recovery_source(recipe: dict, *, actor_factory=None, opponent_factory=None) -> dict:
    """Explicitly replay ONE whole game from its original scenario, then harvest.

    The recipe has no engine-state/clock/cash override. An optional earlier V0
    decision replaces the ordinary sampled decision, through PolicyAgent, not by
    editing state. The ordinary draw still advances its own RNG once. All later
    decisions are direct policy calls. A bad-state flag is assigned only AFTER a
    complete game: terminal loss OR an observed cash deficit from opening cash.
    The latter is a resource proxy, not evidence of a WDL recovery.

    Source/family/split labels and complete-game outcomes are never actor inputs.
    Factories are trusted, fresh policy codecs under the existing dawn contract.
    NATIVE_GAME use requires maintainer authorization/runtime; this API loads no
    checkpoints. Fixture policies must declare INJECTED_FIXTURE explicitly.
    """
    from farm_neat.rollout import v0_regret_dawn as D
    from farm_neat.economy.v0 import vocabulary as V
    from farm_neat.economy.v0.projection import Masks
    Q = D.P
    recipe = copy.deepcopy(_recovery_recipe(recipe))
    actor = D._restore_policy(recipe['actor_policy'], factory=actor_factory)
    rival = D._restore_policy(recipe['opponent_policy'], opponent=True, factory=opponent_factory)
    env = scenarios.ScenarioEnvironment(copy.deepcopy(recipe['scenario']))
    agent = Q.PolicyAgent(actor, labor_factor=4., belief=None, game_id=recipe['game_id'])
    seat = recipe['seat']
    initial_cash = Q.State(env.agent_observation(seat)).cash
    snapshot, intervention, tape, learner_actions = None, None, [], None
    while not env.done:
        step = env.step_index
        need(step <= V.LAST_STEP, 'RECOVERY_REPLAY_STEP_CAP')
        if step == recipe['target_step']:
            continuation = D.make_continuation(agent, rival, seat=seat)
            snapshot = D.capture_dawn(env, agent, rival, continuation=continuation)
        alternate = recipe['alternative']
        if alternate is not None and step == alternate['step']:
            state = Q.State(env.agent_observation(seat))
            original = list(actor(state, agent.plan.copy(), Masks(state, agent.plan.copy(), **agent.projection_kw())))
            chosen = Q.validate_edits(state, agent.plan, alternate['raw_actions'],
                                     projection_kwargs=agent.projection_kw())
            need(list(chosen.actions) != original and chosen.plan.canonical() != agent.plan.canonical(),
                 'DEGRADATION_MUST_CHANGE_EARLIER_DECISION')
            intervention = dict(step=step, ordinary_raw_actions=original,
                                alternative_raw_actions=list(chosen.actions))
            agent.policy = lambda *_: list(chosen.actions)
        try:
            own = agent(env.agent_observation(seat), copy.deepcopy(env.configuration))
        finally:
            agent.policy = actor
        if step == recipe['target_step']:
            learner_actions = list(agent.records[-1]['raw_edits'])
        other = (rival if callable(rival) else rival.act)(
            env.agent_observation(1-seat), copy.deepcopy(env.configuration))
        actions = [None, None]; actions[seat], actions[1-seat] = own, other
        tape.append(copy.deepcopy(actions)); env.step(actions)
    need(snapshot is not None and env.step_index == V.LAST_STEP+1 and
         all(r.status == 'DONE' for r in env.state), 'COMPLETE_SOURCE_GAME_REQUIRED')
    cash = [float(r.reward) for r in env.state]
    wdl = int(cash[seat] > cash[1-seat])-int(cash[seat] < cash[1-seat])
    terminal = dict(status='COMPLETE', steps=env.step_index, cash=cash, terminal_wdl=wdl,
                    terminal_world_sha256=Q.wire_hash(D._world_data(env)))
    return Q.bind(dict(schema=RECOVERY_SOURCE_SCHEMA, recipe=recipe, snapshot=snapshot,
        terminal=terminal, intervention=intervention, learner_raw_actions=learner_actions,
        joint_actions_sha256=Q.wire_hash(tape),
        selection=_recovery_selection(snapshot, terminal, initial_cash),
        source_pins=source_pins() | D.source_pins(),
        counts=dict(complete_source_replays=1, joint_steps=len(tape), optimizer_updates=0)))


def _check_recovery_source(source):
    from farm_neat.rollout import v0_regret_dawn as D
    from farm_neat.economy.v0 import vocabulary as V
    D._bound(source, 'recovery source')
    D._closed(source, ('schema', 'recipe', 'snapshot', 'terminal', 'intervention',
        'joint_actions_sha256', 'learner_raw_actions', 'selection', 'source_pins', 'counts', 'content_sha256'), 'recovery source')
    need(source['schema'] == RECOVERY_SOURCE_SCHEMA, 'RECOVERY_SOURCE_SCHEMA')
    recipe = _recovery_recipe(source['recipe'])
    need(source['source_pins'] == source_pins() | D.source_pins(), 'RECOVERY_SOURCE_PINS')
    snapshot = source['snapshot']
    D.validate_snapshot(snapshot, continuation=snapshot['continuation'])
    need(snapshot['step'] == recipe['target_step'] and snapshot['seat'] == recipe['seat'] and
         snapshot['world']['scenario'] == recipe['scenario'], 'SOURCE_SNAPSHOT_JOIN')
    term = source['terminal']
    need(term['status'] == 'COMPLETE' and term['steps'] == V.LAST_STEP+1 and
         len(term['cash']) == 2, 'COMPLETE_SOURCE_GAME_REQUIRED')
    x, y = term['cash'][recipe['seat']], term['cash'][1-recipe['seat']]
    need(term['terminal_wdl'] == int(x > y)-int(x < y), 'SOURCE_TERMINAL_WDL_MISMATCH')
    opening = scenarios.ScenarioEnvironment(copy.deepcopy(recipe['scenario']))
    opening_cash = D.P.State(opening.agent_observation(recipe['seat'])).cash
    need(source['selection'] == _recovery_selection(snapshot, term, opening_cash), 'RECOVERY_SELECTION_FACTS')
    need(source['counts'] == dict(complete_source_replays=1, joint_steps=V.LAST_STEP+1,
         optimizer_updates=0), 'RECOVERY_SOURCE_COUNTS')
    need((source['intervention'] is None) == (recipe['kind'] == 'NATURAL_BAD'), 'SOURCE_INTERVENTION_JOIN')
    return source


def _source_verification(source):
    from farm_neat.evaluation import v0_policy as Q
    return Q.bind(dict(schema=RECOVERY_SOURCE_SCHEMA+'-verification',
        status='EXACT_FRESH_LEGAL_REPLAY', source_sha256=source['content_sha256'],
        recipe_sha256=source['recipe']['content_sha256'],
        snapshot_sha256=source['snapshot']['content_sha256'],
        terminal_sha256=Q.wire_hash(source['terminal']),
        joint_actions_sha256=source['joint_actions_sha256'], source_pins=source['source_pins']))


def verify_recovery_source(source: dict, *, actor_factory=None, opponent_factory=None) -> dict:
    """Independent full replay detects even re-sealed snapshot/state mutations.

    A verification receipt is data custody evidence, not a cryptographic
    attestation against a producer that forges every receipt and source pin.
    """
    from farm_neat.evaluation import v0_policy as Q
    source = copy.deepcopy(_check_recovery_source(source))
    fresh = harvest_recovery_source(source['recipe'], actor_factory=actor_factory,
                                   opponent_factory=opponent_factory)
    need(Q.wire(fresh) == Q.wire(source), 'SOURCE_NOT_REACHED_BY_LEGAL_REPLAY')
    return _source_verification(source)


def _recovery_admission(result, source):
    """Recheck immutable selection, own-prefix geometry and disjoint paired WDL.

    The existing recovery gate shares the ordinary admission formula but is NOT
    an opening-v1 record. This validator checks the underlying four independently
    returned receipts per confirmation world, not a caller-provided weight.
    No native games, teacher-prefix decoding or source mutation occurs here.
    """
    from farm_neat.rollout import v0_regret_dawn as D
    from farm_neat.training.teachers import v0_regret_budget as B
    Q = D.P
    D._bound(result, 'recovery result')
    need(result['schema'] == S.RECOVERY_SCHEMA and
         result['record_compatibility'] == 'DAWN_RECOVERY_ONLY_NOT_OPENING_PAIR_V1', 'RECOVERY_RESULT_SCHEMA')
    snapshot = source['snapshot']; protocol = result['protocol']; learner = result['learner']
    context, worlds, rows = S._recovery_protocol(snapshot, learner, protocol, result['budget'])
    need(result['snapshot_sha256'] == snapshot['content_sha256'] and result['actor_context'] == context,
         'RECOVERY_CONTEXT_JOIN')
    need(learner['raw_actions'] == source['learner_raw_actions'] and
         learner['checkpoint_sha256'] == source['recipe']['checkpoint_sha256'] and
         learner['origin'] == dict(kind='LEARNER_REACHED' if source['recipe']['kind'] == 'NATURAL_BAD'
             else 'LEGAL_REPLAY', receipt_sha256=source['content_sha256']), 'RECOVERY_ORIGIN_JOIN')
    need(result['evidence_scope'] == protocol['evidence_scope'] == source['recipe']['evidence_scope'],
         'RECOVERY_EVIDENCE_SCOPE_JOIN')
    pins = D.source_pins() | B.source_pins() | {
        'src/farm_neat/training/teachers/v0_regret_searcher.py': hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest()}
    need(result['source_pins'] == pins, 'RECOVERY_SEARCH_SOURCE_MISMATCH')
    frozen = result['frozen_selection']; D._bound(frozen, 'recovery selection')
    need(frozen['snapshot_sha256'] == snapshot['content_sha256'] and
         frozen['learner_sha256'] == learner['content_sha256'] and
         frozen['protocol_sha256'] == protocol['content_sha256'] and
         frozen['candidate_id'] == result['best']['candidate_id'] and
         frozen['discovery_trace_sha256'] == Q.wire_hash(result['search_trace']) and
         frozen['source_pins'] == pins and frozen['budget'] == result['budget'] and
         frozen['seed'] == result['seed'] and frozen['confirmation_registration_sha256'] ==
             Q.wire_hash(worlds['confirmation']), 'RECOVERY_SELECTION_JOIN')
    for path, expected in ((learner['raw_actions'], result['candidates'][0]),
                           (frozen['raw_actions'], result['best'])):
        geometry = S._recovery_candidate(snapshot, path, context, {}, edit_limit=16)
        need(all(geometry[k] == expected[k] for k in ('raw_actions', 'canonical_plan',
             'prefix_class_trace', 'continuation_signature')), 'RECOVERY_STALE_PREFIX_OR_MEMORY')
        paths = {str(snapshot['seat']):B.path_binding(geometry)}
        need(expected['paths'] == paths and expected['candidate_id'] ==
             B.candidate_key(paths, {str(snapshot['seat']):Q.wire_hash(context)}), 'RECOVERY_CANDIDATE_BINDING')
    need(all(frozen[k] == result['best'][k] for k in ('raw_actions', 'canonical_plan',
         'continuation_signature')), 'RECOVERY_FROZEN_PATH_JOIN')
    confirmation = result['confirmation']; D._bound(confirmation, 'recovery confirmation')
    need(confirmation['selection_sha256'] == frozen['content_sha256'], 'RECOVERY_CONFIRMATION_JOIN')
    all_ids, by_group, branches = set(), {}, {}
    discovery_attempts = []
    for attempt in result['attempts']:
        request = attempt['request']; D._bound(request, 'recovery attempt request')
        aid = request['attempt_id']; need(aid not in all_ids, 'RECOVERY_DUPLICATE_ATTEMPT'); all_ids.add(aid)
        if request['phase'] == 'discovery':
            discovery_attempts.append(attempt); continue
        need(request['phase'] == 'confirmation' and request['source_group_id'] in
             {w['source_group_id'] for w in worlds['confirmation']}, 'RECOVERY_CONFIRMATION_EXPOSURE')
        group = request['source_group_id']; role = request['role']
        need(role in ('learner', 'challenger', 'learner_replay', 'challenger_replay'), 'RECOVERY_ROLE')
        need(request['source_family_id'] == rows[group]['source_family_id'] and
             request['protocol_sha256'] == protocol['content_sha256'] and
             request['selection_sha256'] == frozen['content_sha256'] and
             request['evaluator_sha256'] == protocol['evaluator_sha256'] and
             request['evidence_scope'] == protocol['evidence_scope'] and
             request['candidate_id'] == (result['candidates'][0]['candidate_id'] if role.startswith('learner')
                                          else frozen['candidate_id']), 'RECOVERY_ATTEMPT_JOIN')
        actions = learner['raw_actions'] if role.startswith('learner') else frozen['raw_actions']
        snap = rows[group]['snapshot']
        key = (group, tuple(actions))
        if key not in branches:
            proposal = D.propose_dawn(snap, actions, continuation=snap['continuation'])
            branches[key] = D.fork_dawn(snap, proposal, continuation=snap['continuation'])
        expected = branches[key]
        need(Q.wire(request['branch']) == Q.wire(expected), 'RECOVERY_WRONG_BRANCH')
        roles = by_group.setdefault(group, {})
        need(role not in roles, 'RECOVERY_DUPLICATE_ROLE')
        response = attempt['response']
        roles[role] = (S._recovery_branch_result(response, request)
                       if response is not None and attempt['validated'] else None)
    need(frozen['discovery_evidence_sha256'] == Q.wire_hash(discovery_attempts), 'RECOVERY_DISCOVERY_JOIN')
    values = {}
    for world in worlds['confirmation']:
        group = world['source_group_id']; roles = by_group.get(group, {})
        if len(roles) != 4 or any(r is None or r['status'] != 'COMPLETE' for r in roles.values()):
            continue
        signature = lambda r: {k: r[k] for k in ('initial_snapshot_sha256', 'proposal_sha256',
            'terminal_wdl', 'cash', 'actions_sha256', 'terminal_runtime_sha256', 'simulator_steps')}
        for arm in ('learner', 'challenger'):
            need(signature(roles[arm]) == signature(roles[arm+'_replay']), 'RECOVERY_FRESH_REPLAY_MISMATCH')
        values.setdefault(world['source_family_id'], []).append(
            (roles['learner']['terminal_wdl'], roles['challenger']['terminal_wdl']))
    registered = protocol['admission']['n_world_groups']
    complete = len(by_group) == len(worlds['confirmation']) and sum(map(len, values.values())) == len(worlds['confirmation'])
    mean = lower = None
    if complete:
        mean = math.fsum(math.fsum(c-l for l, c in pairs)/len(pairs) for pairs in values.values())/len(values)
        lower = mean-math.sqrt(8*-math.log(protocol['admission']['alpha_pair'])/registered)
        need(confirmation['n_world_groups'] == registered and confirmation['mean_wdl_delta'] == mean
             and confirmation['lower_bound'] == lower, 'RECOVERY_WDL_GATE_MISMATCH')
    proven = (complete and not result['integrity_errors'] and
              frozen['candidate_id'] != result['candidates'][0]['candidate_id'] and
              lower > protocol['admission']['min_effect'])
    weight = min(2., max(0., lower))/2 if proven else 0.
    need(result['status'] == ('PROVEN_RECOVERY' if proven else 'NO_PROVEN_RECOVERY') and
         result['teacher_weight'] == weight and result['actor_training_admissible'] is proven and
         result['statistical_support'] is proven, 'RECOVERY_ADMISSION_WEIGHT_MISMATCH')
    return dict(status=result['status'], teacher_weight=weight, lower_bound=lower,
                mean_wdl_delta=mean, n_world_groups=len(values), independent_confirmation=proven)


def collect_recovery(sources: list[dict], *, requests: dict, proposer, evaluator,
                     family_alpha: float = .05, actor_factory=None, opponent_factory=None) -> dict:
    """Verify sources, then execute each registered training recovery exactly once.

    Each source is freshly replayed before search. Held-out sources are retained
    without calling proposer/evaluator. IDs and disjoint source-family splits
    must be frozen before use. No automatic retries/resume or best-of-reruns.
    Callers persist partial failures; new runs require new prospective test IDs.
    """
    from farm_neat.rollout import v0_regret_dawn as D
    Q = D.P
    need(type(sources) is list and bool(sources), 'RECOVERY_SOURCES_REQUIRED')
    sources = copy.deepcopy(sources)
    training = [s for s in sources if s['recipe']['split'] == 'training']
    need(set(requests) == {s['recipe']['source_id'] for s in training}, 'RECOVERY_REQUEST_COVERAGE')
    R._number(family_alpha, '$.family_alpha', 0., .05); need(family_alpha > 0, 'RECOVERY_ALPHA')
    # Validate the complete split manifest before the first replay or search.
    shell = Q.bind(dict(schema=RECOVERY_CORPUS_SCHEMA, family_alpha=family_alpha,
        registered_source_ids=[s['recipe']['source_id'] for s in sources],
        rows=[dict(source=s, verification=_source_verification(s), recovery=None) for s in sources]))
    _recovery_splits(shell, protocols={sid:req['protocol'] for sid,req in requests.items()})
    for source in training:
        request = requests[source['recipe']['source_id']]
        need(request['protocol']['admission']['family_alpha'] == family_alpha/len(training),
             'RECOVERY_CROSS_SOURCE_ALPHA_ALLOCATION')
        need(source['selection']['bad_state'] is True, 'BAD_SOURCE_REQUIRED')
        S._recovery_protocol(source['snapshot'], request['learner'], request['protocol'], request['budget'])
    rows = []
    for source in sources:
        verification = verify_recovery_source(source, actor_factory=actor_factory, opponent_factory=opponent_factory)
        result = None
        if source['recipe']['split'] == 'training':
            request = requests[source['recipe']['source_id']]
            result = S.search_recovery(source['snapshot'], request['learner'], proposer=proposer,
                evaluator=evaluator, budget=request['budget'], protocol=request['protocol'], seed=request['seed'])
        rows.append(dict(source=source, verification=verification, recovery=result))
    corpus = Q.bind(dict(schema=RECOVERY_CORPUS_SCHEMA, family_alpha=family_alpha,
        registered_source_ids=shell['registered_source_ids'], rows=rows))
    validate_recovery_corpus(corpus)
    return corpus


def _recovery_splits(corpus, *, protocols=None):
    from farm_neat.evaluation import v0_policy as Q
    ids, exposures, families, search_exposures = set(), {}, {}, {}
    for row in corpus['rows']:
        source = _check_recovery_source(row['source']); recipe = source['recipe']
        sid, split = recipe['source_id'], recipe['split']
        need(sid not in ids, 'RECOVERY_DUPLICATE_SOURCE'); ids.add(sid)
        need(row['verification'] == _source_verification(source), 'RECOVERY_REPLAY_VERIFICATION_JOIN')
        if split == 'heldout':
            need(recipe['kind'] == 'NATURAL_BAD' and row['recovery'] is None, 'HELDOUT_MUST_BE_NATURAL_UNSEARCHED')
        family = recipe['source_family_id']
        need(family not in families or families[family] == split, 'RECOVERY_SOURCE_FAMILY_LEAKAGE')
        families[family] = split
        tokens = [('family', family), ('seed', recipe['scenario']['seed']),
                  ('world', Q.wire_hash(recipe['scenario']))]
        tagged = [(token, 'source') for token in tokens]
        protocol = (protocols or {}).get(sid)
        if protocol is None and row['recovery'] is not None:
            protocol = row['recovery']['protocol']
        if protocol is not None:
            for phase in ('discovery', 'confirmation', 'protected'):
                for group in protocol[phase]:
                    tags = [('family', group['source_family_id']),
                            ('seed', group['snapshot']['world']['scenario']['seed']),
                            ('world', Q.wire_hash(group['snapshot']['world']['scenario']))]
                    tagged.extend((tag, phase) for tag in tags)
        for token, phase in tagged:
            need(token not in exposures or exposures[token] == split, 'RECOVERY_SOURCE_FAMILY_LEAKAGE')
            exposures[token] = split
            # No earlier test may expose a later test's confirmation worlds,
            # even when both tests carry the broad 'training' label. Repeated
            # discovery is permitted; independent per-source confirmation is not.
            for owner, previous in search_exposures.get(token, set()):
                need(owner == sid or not ({phase, previous} & {'confirmation','protected'}),
                     'RECOVERY_CROSS_TEST_CONFIRMATION_EXPOSURE')
            search_exposures.setdefault(token, set()).add((sid, phase))
    need(ids == set(corpus['registered_source_ids']) and len(ids) == len(corpus['registered_source_ids']),
         'RECOVERY_REGISTERED_SOURCE_COVERAGE')
    return families


def validate_recovery_corpus(corpus: dict) -> list[dict]:
    """Data-only intake: recheck gates and custody; does not launch replay games."""
    from farm_neat.rollout import v0_regret_dawn as D
    D._bound(corpus, 'recovery corpus')
    D._closed(corpus, ('schema', 'family_alpha', 'registered_source_ids', 'rows', 'content_sha256'), 'recovery corpus')
    need(corpus['schema'] == RECOVERY_CORPUS_SCHEMA, 'RECOVERY_CORPUS_SCHEMA')
    R._number(corpus['family_alpha'], '$.family_alpha', 0., .05)
    need(corpus['family_alpha'] > 0, 'RECOVERY_ALPHA')
    _recovery_splits(corpus)
    n = sum(row['source']['recipe']['split'] == 'training' for row in corpus['rows'])
    result = []
    for row in corpus['rows']:
        D._closed(row, ('source', 'verification', 'recovery'), 'recovery curriculum row')
        source = row['source']; recipe = source['recipe']
        need(source['selection']['bad_state'] is True, 'BAD_SOURCE_REQUIRED')
        gate = dict(status='UNSEARCHED_HELDOUT', teacher_weight=0., lower_bound=None)
        if recipe['split'] == 'training':
            need(row['recovery'] is not None, 'MISSING_RECOVERY_TRACE')
            need(row['recovery']['protocol']['admission']['family_alpha'] == corpus['family_alpha']/n,
                 'RECOVERY_CROSS_SOURCE_ALPHA_ALLOCATION')
            gate = _recovery_admission(row['recovery'], source)
        selection = source['selection']
        result.append(dict(source_id=recipe['source_id'], source_family_id=recipe['source_family_id'],
            split=recipe['split'], kind=recipe['kind'], evidence_scope=recipe['evidence_scope'],
            parent_checkpoint_sha256=recipe['checkpoint_sha256'], gate=gate,
            stratum=dict(remaining_horizon='late' if selection['remaining_steps'] <= 192 else 'early',
                cash_deficit='positive' if selection['cash_deficit_proxy'] > 0 else 'none',
                observed_hands=selection['observed_hands'], observed_stock_units=selection['observed_stock_units'],
                opponent=D.P.wire_hash(source['snapshot']['continuation']['opponent_identity']),
                source_family=recipe['source_family_id'], source_kind=recipe['kind'], outcome=gate['status'])))
    return result


# Actor-time teacher-state sampling. Admission remains in v0_regret_admission.
BELIEF_CURRICULUM_SCHEMA = 'farm-v0-belief-curriculum-v1'
BELIEF_CURRICULUM_MODES = ('uniform', 'focused')
BELIEF_CURRICULUM_STRATA = ('ORDINARY', 'DISAGREEMENT_UNCALIBRATED',
    'HIGH_UNCERTAINTY_NO_DISAGREEMENT', 'UNAVAILABLE_FORECAST')


def belief_curriculum_features(record: dict, *, expected_actor_sha256: str) -> dict:
    """Only frozen public forecasts determine search priority; no labels enter.

    Disagreement is the probability that two independent hypothesis draws have
    disjoint optimal-action sets (ties with a common optimum are not conflict).
    The objective is the existing terminal -1/0/+1 WDL utility, NOT wealth/risk.
    P(win) disagreement is reported separately. Conditional marginal WDL laws do
    not identify a paired regret law: its variance remains unknown. Cauchy bounds
    on within-hypothesis difference variance plus between-hypothesis mean-delta
    variance are conservative diagnostics, not confidence intervals or rewards.
    No calibrated predictor is established by the present record codec.
    """
    from farm_neat.training import v0_belief_records as B
    actor = B.validate_record(record, expected_actor_sha256=expected_actor_sha256)['actor']
    belief = actor['belief']; rows = belief['hypotheses']
    masses = [h[belief['mass_used']] for h in rows]
    ids = sorted(c['candidate_sha256'] for c in actor['candidates'])
    entropy = -math.fsum(p * math.log(p) for p in masses if p > 0)
    normalized = entropy / math.log(len(rows)) if len(rows) > 1 else 0.
    missing = math.fsum(p for p, h in zip(masses, rows)
                       if p > 0 and any(h['conditional_wdl'][c] is None for c in ids))
    result = dict(actor_payload_sha256=expected_actor_sha256,
        actor_context_sha256=actor['context_sha256'], through_step=actor['context']['observation']['step'],
        entropy_nats=entropy, normalized_entropy=normalized, effective_k=math.exp(entropy),
        mass_used=belief['mass_used'], missing_forecast_mass=missing,
        decision_disagreement=None, win_probability_disagreement=None,
        regret_variance_lower_bound=None, regret_variance_upper_bound=None,
        paired_regret_variance=None, calibrated=False, calibration_evidence=None,
        teacher_weight=0., focus_score=0., stratum='UNAVAILABLE_FORECAST')
    if missing:
        return result  # never renormalize away unknown hypotheses
    means, variances, wins = [], [], []
    for h in rows:
        laws = [h['conditional_wdl'][c] for c in ids]
        mu = [p[2] - p[0] for p in laws]
        means.append(mu); wins.append([p[2] for p in laws])
        variances.append([max(0., p[0]+p[2]-m*m) for p, m in zip(laws, mu)])

    def conflict(values):
        best = [{i for i, v in enumerate(row) if max(row)-v <= 1e-12} for row in values]
        return math.fsum(2*masses[i]*masses[j] for i in range(len(rows))
                         for j in range(i) if best[i].isdisjoint(best[j]))

    lower, upper = 0., 0.
    for a in range(len(ids)):
        for b in range(a):
            ds = [row[a]-row[b] for row in means]
            mean = math.fsum(w*d for w, d in zip(masses, ds))
            between = math.fsum(w*(d-mean)**2 for w, d in zip(masses, ds))
            lo = between + math.fsum(w*(math.sqrt(v[a])-math.sqrt(v[b]))**2
                                    for w, v in zip(masses, variances))
            hi = between + math.fsum(w*(math.sqrt(v[a])+math.sqrt(v[b]))**2
                                    for w, v in zip(masses, variances))
            lower, upper = max(lower, lo), max(upper, hi)
    disagreement = conflict(means)
    # Versioned constants; not tuned against outcomes. Large outcome variance
    # with a shared best action remains an explicit negative/control stratum.
    high_uncertainty = normalized >= .8 or math.sqrt(upper) >= .75
    stratum = ('DISAGREEMENT_UNCALIBRATED' if disagreement > 1e-12 else
               'HIGH_UNCERTAINTY_NO_DISAGREEMENT' if high_uncertainty else 'ORDINARY')
    result.update(decision_disagreement=disagreement, win_probability_disagreement=conflict(wins),
        regret_variance_lower_bound=lower, regret_variance_upper_bound=upper, stratum=stratum,
        focus_score=disagreement*(1+normalized)*(1+math.sqrt(upper)/2))
    return result


def _belief_curriculum_coverage(features, selected):
    counts = {s: sum(f['stratum'] == s for f in features.values()) for s in BELIEF_CURRICULUM_STRATA}
    chosen = {s: sum(features[k]['stratum'] == s for k in selected) for s in BELIEF_CURRICULUM_STRATA}
    ordinary = counts['ORDINARY']
    return dict(population_size=len(features), selected_opportunities=len(selected),
        strata_population=counts, strata_selected=chosen, ordinary_selected=chosen['ORDINARY'],
        ordinary_population=ordinary, ordinary_coverage=chosen['ORDINARY']/ordinary if ordinary else None,
        ordinary_sample_fraction=chosen['ORDINARY']/len(selected),
        unique_actor_contexts=len({features[k]['actor_context_sha256'] for k in selected}),
        distinct_frozen_forecasts=len({features[k]['actor_payload_sha256'] for k in selected}),
        independent_world_count=None)


def register_belief_curriculum(collection_plan: dict, *, records: dict,
        expected_actor_sha256s: dict, sample_size: int, seed: int,
        optimizer_updates: int, max_focus_fraction: float = .5,
        ordinary_min_fraction: float = .25) -> dict:
    """Freeze both sampling arms BEFORE any collection, confirmation or fitting.

    Inputs: one existing collector plan, one opening/single-seat frozen belief
    record and separately retained pre-outcome actor hash per test_id. Private
    record payloads are discarded. Source-plan worlds/opponent/IDs are custody
    metadata, never features. Equal per-state discovery/confirmation allocation
    is required; archived search receipts cannot select these new states.
    The result directly drives collect_belief_curriculum and the unchanged fit
    adapter. Only the teacher-state distribution changes; direct-policy RL data,
    action likelihoods, teacher-admission gates and weights are not redefined.
    Hashes bind bytes; they cannot certify when an external producer froze them.
    """
    from fractions import Fraction
    from farm_neat.training import v0_belief_records as B
    bound(collection_plan, 'CURRICULUM_COLLECTION_PLAN_HASH')
    need(collection_plan == register_plan(collection_plan['spec']), 'CURRICULUM_COLLECTION_SOURCE_DRIFT')
    spec = copy.deepcopy(collection_plan['spec'])
    spec['tests'].sort(key=lambda t: t['test_id'])
    source = register_plan(spec)
    tests = spec['tests']; ids = [t['test_id'] for t in tests]
    need(type(records) is dict and type(expected_actor_sha256s) is dict and
         set(records) == set(expected_actor_sha256s) == set(ids), 'CURRICULUM_POPULATION_COVERAGE')
    R._int(sample_size, '$.curriculum.sample_size', 1, len(ids))
    R._int(seed, '$.curriculum.seed', 0, 2**63-1)
    R._int(optimizer_updates, '$.curriculum.optimizer_updates', 1, 2)
    for name, value in (('max_focus_fraction', max_focus_fraction),
                        ('ordinary_min_fraction', ordinary_min_fraction)):
        R._number(value, '$.curriculum.'+name, 0., .5)
    clean, features, sizes = {}, {}, set()
    for test in tests:
        need(test['split'] == 'training', 'CURRICULUM_TRAINING_ONLY')
        need(len(test['seats']) == 1, 'CURRICULUM_SINGLE_SEAT_ONLY')
        seat = test['seats'][0]; key = test['test_id']
        need(seat['search'] is None, 'CURRICULUM_NO_PREOBSERVED_SEARCH')
        sizes.add(tuple(len(test[p+'_world_ids']) for p in PHASES))
        row = B.validate_record(records[key], expected_actor_sha256=expected_actor_sha256s[key])
        need(row['actor']['context'] == seat['actor_context'], 'CURRICULUM_ACTOR_CONTEXT_MISMATCH')
        row['trainer_only'] = {}
        clean[key] = bind(row)
        features[key] = belief_curriculum_features(clean[key], expected_actor_sha256=expected_actor_sha256s[key])
    need(len(sizes) == 1, 'CURRICULUM_UNMATCHED_STATE_BUDGETS')
    discovery_n, confirmation_n = next(iter(sizes))
    per_state = discovery_n*(1+spec['max_discovery_challengers']) + 2*confirmation_n
    required = dict(max_base_starts=sample_size*per_state,
                    max_replay_starts=sample_size*per_state, max_search_starts=0)
    need(all(required[k] <= spec['budget'][k] for k in required), 'CURRICULUM_SOURCE_BUDGET_EXCEEDED')
    # Seeded hash ranking is stable under input permutation and never uses labels,
    # source-family/opponent names, private outcomes, or the population digest.
    order = sorted(ids, key=lambda k: (digest(['belief-curriculum-uniform-v1', seed, k]), k))
    uniform = order[:sample_size]
    ordinary = [k for k in order if features[k]['stratum'] == 'ORDINARY']
    controls = [k for k in order if features[k]['stratum'] == 'HIGH_UNCERTAINTY_NO_DISAGREEMENT']
    ordinary_n = min(len(ordinary), math.ceil(sample_size*Fraction(str(ordinary_min_fraction))))
    selected = ordinary[:ordinary_n]
    control_n = int(bool(controls) and sample_size >= 3 and len(selected) < sample_size)
    selected += controls[:control_n]
    focus_cap = min(int(sample_size*Fraction(str(max_focus_fraction))), sample_size-len(selected))
    focus_order = sorted((k for k in order if features[k]['focus_score'] > 0),
        key=lambda k: (-features[k]['focus_score'], order.index(k)))
    focused = focus_order[:focus_cap]
    selected += focused
    selected += [k for k in order if k not in selected][:sample_size-len(selected)]
    arms = {}
    for mode, chosen in zip(BELIEF_CURRICULUM_MODES, (uniform, selected)):
        chosen = sorted(chosen)
        arm_spec = copy.deepcopy(spec)
        arm_spec.update(round_id=spec['round_id']+':belief:'+mode,
            family_id=spec['family_id']+':belief:'+mode, budget=required,
            tests=[copy.deepcopy(t) for t in tests if t['test_id'] in chosen])
        arms[mode] = dict(selected_state_ids=chosen, focus_slots=len(focused) if mode == 'focused' else 0,
            coverage=_belief_curriculum_coverage(features, chosen), collection_plan=register_plan(arm_spec))
    return bind(dict(schema=BELIEF_CURRICULUM_SCHEMA, collection_plan=source,
        records=clean, expected_actor_sha256s=copy.deepcopy(expected_actor_sha256s), features=features,
        selection=dict(sample_size=sample_size, seed=seed, max_focus_fraction=max_focus_fraction,
            ordinary_min_fraction=ordinary_min_fraction),
        budget=dict(simulator_start_slots_per_arm=2*sample_size*per_state,
            optimizer_updates_per_arm=optimizer_updates, opportunity_slots_per_arm=sample_size),
        arms=arms, source_pins=source_pins() | B.source_pins(),
        objective='TERMINAL_WDL', teacher_weight_rule='UNCHANGED_REGRET_ADMISSION_ONLY',
        calibration_status='NOT_ESTABLISHED;DISAGREEMENT_IS_UNCALIBRATED',
        scope='OPENING_SINGLE_SEAT_TEACHER_STATE_SELECTION;NOT_DIRECT_RL_RESAMPLING',
        frozen_before_collection_required=True, strength_claim=False))


def validate_belief_curriculum(document: dict) -> dict:
    """Recompute features, selections, budgets and child plans, not just a hash."""
    bound(document, 'CURRICULUM_HASH_MISMATCH')
    need(document.get('schema') == BELIEF_CURRICULUM_SCHEMA, 'CURRICULUM_SCHEMA')
    expected = register_belief_curriculum(document['collection_plan'], records=document['records'],
        expected_actor_sha256s=document['expected_actor_sha256s'], **document['selection'],
        optimizer_updates=document['budget']['optimizer_updates_per_arm'])
    need(document == expected, 'CURRICULUM_DERIVED_OR_SOURCE_DRIFT')
    return expected


def collect_belief_curriculum(registration: dict, *, mode: str, sampler, proposer,
                             evaluator, output: str | Path) -> dict:
    """Persist the priority decision, then use the existing journaled collector.

    Same-directory resume cannot change the arm/population/seed. This does not
    launch fitting and has no implicit native runner. Callbacks retain the
    existing collector's trust and source-fingerprint requirements.
    """
    reg = validate_belief_curriculum(registration)
    need(mode in BELIEF_CURRICULUM_MODES, 'CURRICULUM_MODE')
    output = Path(output).resolve()
    frozen = wire(dict(registration=reg, mode=mode))+b'\n'
    # The collector uses this same lock; release it before its own acquisition.
    with _lock(output):
        path = output/'BELIEF-CURRICULUM.json'
        if path.exists():
            need(path.read_bytes() == frozen, 'CURRICULUM_RESUME_DRIFT')
        else:
            need(not (output/'JOURNAL.jsonl').exists(), 'CURRICULUM_LATE_REGISTRATION')
            _atomic(path, frozen)
    report = collect_round(reg['arms'][mode]['collection_plan'], sampler=sampler,
        proposer=proposer, evaluator=evaluator, output=output)
    return bind(dict(schema=BELIEF_CURRICULUM_SCHEMA+'-run', registration_sha256=reg['content_sha256'],
        mode=mode, output=str(output), collection=report))


def read_belief_curriculum_run(registration: dict, run: dict) -> tuple[dict, dict]:
    """Authenticate retained collection bytes/counts before trainer-side metrics.

    This checks the durable plan, journal, frozen preselection and artifact bytes.
    The trainer separately recomputes admission from the retained paired corpus.
    Native provenance still depends on the existing producer/replay contracts.
    """
    reg = validate_belief_curriculum(registration)
    bound(run, 'CURRICULUM_RUN_HASH')
    R._fields(run, ('schema', 'registration_sha256', 'mode', 'output', 'collection', 'content_sha256'), '$.curriculum_run')
    need(run['schema'] == BELIEF_CURRICULUM_SCHEMA+'-run' and
         run['registration_sha256'] == reg['content_sha256'] and
         run['mode'] in BELIEF_CURRICULUM_MODES, 'CURRICULUM_RUN_BINDING')
    root = Path(run['output']); report = run['collection']
    expected = reg['arms'][run['mode']]['collection_plan']
    need(R._loads((root/'BELIEF-CURRICULUM.json').read_text()) == dict(registration=reg, mode=run['mode']),
         'CURRICULUM_FROZEN_REGISTRATION_MISMATCH')
    need(R._loads((root/'PLAN.json').read_text()) == expected, 'CURRICULUM_SAVED_PLAN_MISMATCH')
    need(R._loads((root/'RESULT.json').read_text()) == report, 'CURRICULUM_SAVED_REPORT_MISMATCH')
    bound(report, 'CURRICULUM_REPORT_HASH')
    journal = Journal(root)
    need(report['plan_sha256'] == expected['content_sha256'] and
         report['journal_head'] == journal.events[-1]['content_sha256'] and
         report['counts'] == _counts(journal, expected), 'CURRICULUM_JOURNAL_MISMATCH')
    need(set(report['artifacts']) == set(report['artifact_sha256s']), 'CURRICULUM_ARTIFACT_SET')
    for key, path in report['artifacts'].items():
        p = Path(path)
        need(p.resolve().is_relative_to(root.resolve()) and p.is_file() and
             hashlib.sha256(p.read_bytes()).hexdigest() == report['artifact_sha256s'][key],
             'CURRICULUM_ARTIFACT_HASH')
    return reg, copy.deepcopy(report)


# Production belief sidecars. This is a read-only join of ORIGINAL callback
# receipts, not a simulator or a replacement for the native collection owner.
BELIEF_PRODUCTION_SCHEMA = 'farm-v0-belief-production-population-v1'
BELIEF_PRIVATE_FIELDS = {'outcome_tensor', 'mass_target', 'decision_credit',
                         'confidence', 'confidence_source_sha256'}


def belief_production_source_pins():
    from farm_neat.training import v0_belief_outcomes as O
    from farm_neat.training import v0_belief_credit as K
    names = ('src/farm_neat/rollout/v0_regret_collect.py',
             'src/farm_neat/training/v0_regret_train.py',
             'src/farm_neat/policy/v0_belief.py',
             'src/farm_neat/training/v0_belief_loss.py',
             'project/configs/v0-belief.json', 'project/runtime/interpreter.toml')
    return O.source_pins() | K.source_pins() | {
        p: hashlib.sha256((R.ROOT/p).read_bytes()).hexdigest() for p in names}


def _belief_producer(provenance, actor_map, diagnostic):
    need(type(provenance) is dict and set(provenance) ==
         {'execution', 'producer_source', 'config_sha256', 'actor_map_sha256'},
         'BELIEF_CALLBACK_PROVENANCE_REQUIRED')
    need(provenance['actor_map_sha256'] == digest(actor_map), 'BELIEF_CALLBACK_ACTOR_MAP_JOIN')
    R._hash(provenance['config_sha256'], '$.callback.config_sha256')
    source = provenance['producer_source']
    need(type(source) is dict and set(source) == {'path', 'sha256'}, 'BELIEF_CALLBACK_SOURCE_REQUIRED')
    R._hash(source['sha256'], '$.callback.source.sha256')
    need(type(source['path']) is str and source['path'], 'BELIEF_CALLBACK_SOURCE_REQUIRED')
    path = R.ROOT / source['path']
    need(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'],
         'BELIEF_CALLBACK_SOURCE_DRIFT')
    execution = provenance['execution']
    bound(execution, 'BELIEF_CALLBACK_EXECUTION_HASH')
    if not diagnostic:
        need(execution == E.execution_binding(E.native_runner), 'BELIEF_NATIVE_CALLBACK_REQUIRED')
    else:
        need(execution.get('kind') == 'SYNTHETIC_DIAGNOSTIC', 'BELIEF_DIAGNOSTIC_CALLBACK_REQUIRED')
    return execution


def freeze_belief_production_population(*, records: dict, actor_map: dict,
        parent_checkpoint_sha256: str, callback_provenance: dict,
        protected_groups: dict, diagnostic: bool = False) -> dict:
    """Freeze one finite, opening-only belief population; zero callbacks/games.

    ``records`` maps stable record IDs to the existing FrozenBeliefRecord wire.
    Each trainer_only subtree has exactly BELIEF_PRIVATE_FIELDS. ``mass_target``
    is {hypothesis_ids, probabilities, observed, source_sha256}; missing mass
    labels use probabilities=None, observed=False. It is a *trusted producer's*
    full-catalog label, not inferred from a hypothetical branch or a posterior.
    ``actor_map`` is the independently saved pre-outcome mapping of record ID to
    {actor_payload_sha256, parent_checkpoint_sha256}. Its custody/chronology and
    label truth remain the producer's responsibility, not something hashes prove.

    ``callback_provenance`` pins the native execution binding, producer source
    file, config hash and actor-map hash. Rebuild original outcome/credit receipts;
    do not accept an execution-kind declaration instead of native evidence.
    ``protected_groups`` has source_group_ids, world_sha256s, source_family_ids.
    Production requires a nonempty protected-world list and native source rows.
    UNKNOWN and unobserved conditionals remain present, never imputed/renormalized.
    Diagnostic fixtures may exercise the same join but cannot pass preflight.
    """
    from farm_neat.training import v0_belief_records as B, v0_belief_outcomes as O
    from farm_neat.training import v0_belief_credit as K
    need(type(actor_map) is dict and bool(actor_map), 'BELIEF_ACTOR_MAP_REQUIRED')
    need(type(records) is dict and records and set(records) == set(actor_map), 'BELIEF_ACTOR_MAP_COVERAGE')
    need(type(diagnostic) is bool, 'BELIEF_DIAGNOSTIC_FLAG')
    R._hash(parent_checkpoint_sha256, '$.parent_checkpoint_sha256')
    execution = _belief_producer(callback_provenance, actor_map, diagnostic)
    protection = {'source_group_ids', 'world_sha256s', 'source_family_ids'}
    need(type(protected_groups) is dict and set(protected_groups) == protection, 'BELIEF_PROTECTED_GROUPS_REQUIRED')
    for key in protection:
        values = protected_groups[key]
        need(type(values) is list and all(type(v) is str and v for v in values)
             and len(values) == len(set(values)), 'BELIEF_PROTECTED_GROUPS_INVALID')
    need(bool(protected_groups['world_sha256s']), 'BELIEF_PROTECTED_WORLDS_REQUIRED')
    for value in protected_groups['world_sha256s']:
        R._hash(value, '$.protected.world_sha256s')
    rows, world_families, raw_worlds, actor_worlds = [], {}, {}, set()
    for rid in sorted(records):
        R._text(rid, '$.record_id')
        anchor = actor_map[rid]
        need(type(anchor) is dict and set(anchor) ==
             {'actor_payload_sha256', 'parent_checkpoint_sha256'}, 'BELIEF_ACTOR_MAP_ROW')
        need(anchor['parent_checkpoint_sha256'] == parent_checkpoint_sha256, 'BELIEF_PARENT_MISMATCH')
        record = B.validate_record(records[rid], expected_actor_sha256=anchor['actor_payload_sha256'])
        actor, private = record['actor'], record['trainer_only']
        need(set(private) == BELIEF_PRIVATE_FIELDS, 'BELIEF_PRIVATE_LABEL_FIELDS')
        tensor = O.validate_tensor(private['outcome_tensor'], expected_actor_sha256=anchor['actor_payload_sha256'])
        original = R._loads(tensor['inputs_wire'])
        need(R._equal(original['belief_record']['actor'], actor), 'BELIEF_TENSOR_ACTOR_JOIN')
        need(tensor['split'] == 'training', 'BELIEF_TRAIN_SPLIT_REQUIRED')
        # MC import is deliberately left to its owner. Do not trust aggregate
        # NATIVE counters without recursively joining its retained job receipts.
        need(not original['mc_sources'], 'BELIEF_PAIRED_NATIVE_SOURCES_REQUIRED')
        need(bool(original['pair_sources']), 'BELIEF_NATIVE_SOURCES_REQUIRED')
        for source in original['pair_sources']:
            need(source['result']['execution'] == execution, 'BELIEF_CALLBACK_RECEIPT_JOIN')
            need(source['request']['record']['identity']['parent_checkpoint_sha256'] ==
                 parent_checkpoint_sha256, 'BELIEF_NATIVE_PARENT_JOIN')
        if not diagnostic:
            need(tensor['counts']['native_starts'] > 0 and all(
                a['evidence_kind'] == 'NATIVE_GAME' for a in tensor['attempts']), 'BELIEF_SYNTHETIC_RECORD_REFUSED')
        groups = tensor['groups']
        for group in groups:
            world, family = group['world_sha256'], group['source_family_id']
            need(world not in protected_groups['world_sha256s'] and
                 family not in protected_groups['source_family_ids'] and
                 not (set(group['source_group_ids']) & set(protected_groups['source_group_ids'])),
                 'BELIEF_PROTECTED_SOURCE_OVERLAP')
            # Same world across seats/hypotheses is *one* source, not fresh n.
            need(world_families.setdefault(world, family) == family, 'BELIEF_SOURCE_FAMILY_ALIAS')
            for raw in group['source_group_ids']:
                need(raw_worlds.setdefault(raw, world) == world, 'BELIEF_SOURCE_GROUP_ALIAS')
        # A world is one source even when represented by multiple seats or
        # hypotheses. Repeating the same actor/world under another record ID
        # would silently give that evidence extra optimizer weight.
        for world in {group['world_sha256'] for group in groups}:
            actor_world = (anchor['actor_payload_sha256'], world)
            need(actor_world not in actor_worlds, 'BELIEF_DUPLICATE_SOURCE_RECORD')
            actor_worlds.add(actor_world)
        hids = [h['hypothesis_id'] for h in actor['belief']['hypotheses']]
        target = private['mass_target']
        need(type(target) is dict and set(target) ==
             {'hypothesis_ids', 'probabilities', 'observed', 'source_sha256'}, 'BELIEF_MASS_TARGET_FIELDS')
        need(type(target['hypothesis_ids']) is list and len(target['hypothesis_ids']) == len(hids)
             and set(target['hypothesis_ids']) == set(hids), 'BELIEF_MASS_TARGET_CATALOG')
        need(type(target['observed']) is bool, 'BELIEF_MASS_TARGET_MASK')
        R._hash(target['source_sha256'], '$.mass_target.source_sha256')
        if target['observed']:
            B._probabilities(target['probabilities'], len(hids), '$.mass_target.probabilities')
        else:
            need(target['probabilities'] is None, 'BELIEF_MISSING_MASS_MUST_REMAIN_MISSING')
        R._number(private['confidence'], '$.confidence', 0., 1.)
        R._hash(private['confidence_source_sha256'], '$.confidence_source_sha256')
        credit = private['decision_credit']
        if credit is not None:
            credit = K.validate_credit(credit, expected_actor_sha256=anchor['actor_payload_sha256'])
            need(credit['tensor_sha256'] == tensor['content_sha256'], 'BELIEF_CREDIT_TENSOR_JOIN')
            pair = R._loads(credit['inputs_wire'])['pair']
            need(pair['identity']['parent_checkpoint_sha256'] == parent_checkpoint_sha256,
                 'BELIEF_CREDIT_PARENT_JOIN')
        hypothesis_groups = {h: sorted({c['continuation_group_sha256'] for c in tensor['cells']
            if c['hypothesis_id'] == h}) for h in hids}
        rows.append(dict(record_id=rid, actor_payload_sha256=anchor['actor_payload_sha256'],
            parent_checkpoint_sha256=parent_checkpoint_sha256, record=record,
            hypothesis_source_groups=hypothesis_groups,
            worlds=[dict(world_sha256=g['world_sha256'], world=g['world'], seat=g['seat'],
                source_family_id=g['source_family_id'], source_group_ids=g['source_group_ids'],
                continuation_group_sha256=g['continuation_group_sha256']) for g in groups]))
    return bind(dict(schema=BELIEF_PRODUCTION_SCHEMA,
        status='DIAGNOSTIC_FIXTURE_ONLY' if diagnostic else 'FROZEN_NATIVE_POPULATION_NOT_FITTED',
        diagnostic=diagnostic, source_pins=belief_production_source_pins(),
        parent_checkpoint_sha256=parent_checkpoint_sha256, actor_map=copy.deepcopy(actor_map),
        actor_map_sha256=digest(actor_map), callback_provenance=copy.deepcopy(callback_provenance),
        protected_groups=copy.deepcopy(protected_groups), rows=rows,
        record_count=len(rows), independent_world_count=len(world_families),
        independence='SOURCE_ALIASES_AND_DUPLICATE_ACTOR_WORLDS_REJECTED;SAMPLING_INDEPENDENCE_AND_CHRONOLOGY_NOT_CERTIFIED',
        objective='TERMINAL_WDL', native_games_executed=0, optimizer_updates=0,
        strength_claim=False))


BELIEF_NATIVE_JOIN_SCHEMA = 'farm-v0-belief-native-join-spec-v1'
BELIEF_NATIVE_JOIN_REFS = ('actor_map', 'actor_records', 'callback_provenance', 'protected_groups', 'label_rows')


def _belief_join_read(ref: dict, label: str) -> dict:
    """Read one immutable JSON reference once, preserving original receipt order."""
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'BELIEF_JOIN_INPUT_REFERENCE: ' + label)
    R._text(ref['path'], '$.belief_join.' + label + '.path')
    R._hash(ref['sha256'], '$.belief_join.' + label + '.sha256')
    data = (R.ROOT / ref['path']).read_bytes()
    need(len(data) <= 64 * 1024 * 1024, 'BELIEF_JOIN_INPUT_BYTE_LIMIT: ' + label)
    need(hashlib.sha256(data).hexdigest() == ref['sha256'], 'BELIEF_JOIN_INPUT_HASH: ' + label)
    value = R._loads(data.decode('utf-8'))
    need(type(value) is dict, 'BELIEF_JOIN_INPUT_OBJECT: ' + label)
    return value


def join_belief_production_population(spec: dict, *, diagnostic: bool = False) -> dict:
    """Assemble accepted tensor/credit records from frozen callback exports.

    No callback is imported/invoked and no checkpoint/model is loaded. ``spec``
    pins five JSON objects by file SHA256: actor_records, actor_map,
    callback_provenance, protected_groups and label_rows. Relative paths are
    repository-relative. Actor records/maps must be independently frozen BEFORE
    label exposure; their chronology and truth remain the trusted producer's
    responsibility. This join never manufactures either artifact.

    Actor records have an empty trainer_only subtree. Per-ID label rows contain
    hypothesis_bindings, original pair_sources, explicit mass_target/confidence/
    confidence_source_sha256 and decision_credit (null or {pair,
    admission_protocol}). Existing outcome and credit owners derive their own
    tensors/loss labels; no summary counters or self-predictions become truth.
    Missing mass and UNKNOWN conditionals retain the existing missing masks.

    The returned population retains all original file references. Validation
    reopens those files and recomputes this transaction, so an embedded copy is
    not a substitute for a missing original. CLI always uses production mode;
    diagnostic=True is for CPU fixtures, never a launch authorization.
    """
    from farm_neat.training import v0_belief_records as B, v0_belief_outcomes as O
    from farm_neat.training import v0_belief_credit as K
    R._safe(spec)
    fields = {'schema', 'parent_checkpoint_sha256', *BELIEF_NATIVE_JOIN_REFS}
    need(type(spec) is dict and set(spec) == fields and spec['schema'] == BELIEF_NATIVE_JOIN_SCHEMA,
         'BELIEF_JOIN_SPEC_FIELDS')
    need(type(diagnostic) is bool, 'BELIEF_DIAGNOSTIC_FLAG')
    parent = spec['parent_checkpoint_sha256']
    R._hash(parent, '$.parent_checkpoint_sha256')
    anchors = _belief_join_read(spec['actor_map'], 'actor_map')
    need(bool(anchors), 'BELIEF_ACTOR_MAP_REQUIRED')
    provenance = _belief_join_read(spec['callback_provenance'], 'callback_provenance')
    _belief_producer(provenance, anchors, diagnostic)
    # Actor custody is checked before even opening the outcome-label file.
    actors = _belief_join_read(spec['actor_records'], 'actor_records')
    need(bool(actors) and set(actors) == set(anchors), 'BELIEF_ACTOR_MAP_COVERAGE')
    for rid in sorted(actors):
        R._text(rid, '$.record_id')
        anchor = anchors[rid]
        need(type(anchor) is dict and set(anchor) ==
             {'actor_payload_sha256', 'parent_checkpoint_sha256'}, 'BELIEF_ACTOR_MAP_ROW')
        need(anchor['parent_checkpoint_sha256'] == parent, 'BELIEF_PARENT_MISMATCH')
        actors[rid] = B.validate_record(actors[rid], expected_actor_sha256=anchor['actor_payload_sha256'])
        need(actors[rid]['trainer_only'] == {}, 'BELIEF_JOIN_ACTOR_ONLY_REQUIRED')
    protected = _belief_join_read(spec['protected_groups'], 'protected_groups')
    labels = _belief_join_read(spec['label_rows'], 'label_rows')
    need(set(labels) == set(actors), 'BELIEF_JOIN_LABEL_COVERAGE')
    records = {}
    for rid in sorted(actors):
        row = labels[rid]
        need(type(row) is dict and set(row) == {'hypothesis_bindings', 'pair_sources',
             'mass_target', 'confidence', 'confidence_source_sha256', 'decision_credit'},
             'BELIEF_JOIN_LABEL_FIELDS')
        tensor = O.make_tensor(actors[rid], hypothesis_bindings=row['hypothesis_bindings'],
                               pair_sources=row['pair_sources'])
        credit = row['decision_credit']
        if credit is not None:
            need(type(credit) is dict and set(credit) == {'pair', 'admission_protocol'},
                 'BELIEF_JOIN_CREDIT_FIELDS')
            credit = K.make_credit(credit['pair'], tensor,
                expected_actor_sha256=anchors[rid]['actor_payload_sha256'],
                admission_protocol=credit['admission_protocol'])
        private = dict(outcome_tensor=tensor, decision_credit=credit,
            mass_target=copy.deepcopy(row['mass_target']), confidence=row['confidence'],
            confidence_source_sha256=row['confidence_source_sha256'])
        records[rid] = B.FrozenBeliefRecord(actors[rid]).with_trainer(private).to_dict()
    population = freeze_belief_production_population(records=records, actor_map=anchors,
        parent_checkpoint_sha256=parent, callback_provenance=provenance,
        protected_groups=protected, diagnostic=diagnostic)
    population['native_join_inputs'] = copy.deepcopy(spec)
    return bind(population)


def validate_belief_production_population(document: dict, *, production: bool = True) -> dict:
    bound(document, 'BELIEF_POPULATION_HASH')
    need(document.get('schema') == BELIEF_PRODUCTION_SCHEMA, 'BELIEF_POPULATION_SCHEMA')
    need(document.get('source_pins') == belief_production_source_pins(), 'BELIEF_POPULATION_SOURCE_DRIFT')
    need(not production or document.get('diagnostic') is False, 'BELIEF_SYNTHETIC_POPULATION_REFUSED')
    rows = document.get('rows')
    need(type(rows) is list and all(type(r) is dict and 'record_id' in r and 'record' in r for r in rows),
         'BELIEF_POPULATION_ROWS')
    records = {r['record_id']: r['record'] for r in rows}
    need(len(records) == len(rows), 'BELIEF_DUPLICATE_RECORD_ID')
    if 'native_join_inputs' in document:
        # Rebuild from independently pinned originals, not merely from embedded
        # derived labels. Missing/changed input bytes must refuse before fitting.
        rebuilt = join_belief_production_population(document['native_join_inputs'],
            diagnostic=document['diagnostic'])
    else:
        rebuilt = freeze_belief_production_population(records=records, actor_map=document['actor_map'],
            parent_checkpoint_sha256=document['parent_checkpoint_sha256'],
            callback_provenance=document['callback_provenance'], protected_groups=document['protected_groups'],
            diagnostic=document['diagnostic'])
    need(R._equal(rebuilt, document), 'BELIEF_POPULATION_DERIVED_DRIFT')
    return rebuilt


def write_belief_production_population(path, document):
    """Exclusive deterministic publication; same-byte resume is a read, never overwrite."""
    validate_belief_production_population(document, production=not document['diagnostic'])
    path = Path(path)
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'BELIEF_SYMLINK_OUTPUT')
    data = wire(document)+b'\n'
    if path.exists():
        need(path.read_bytes() == data, 'BELIEF_POPULATION_RESUME_DRIFT')
        return dict(status='RESUMED_IDENTICAL_POPULATION', sha256=hashlib.sha256(data).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    # An incomplete write is not a resumable manifest. Publish by an atomic,
    # no-clobber hardlink, retaining a concurrent author's different file.
    fd, tmp = tempfile.mkstemp(prefix='.belief-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            need(path.read_bytes() == data, 'BELIEF_POPULATION_RESUME_DRIFT')
    finally:
        os.unlink(tmp)
    return dict(status='FROZEN_POPULATION_WRITTEN', sha256=hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    raise SystemExit(main())
