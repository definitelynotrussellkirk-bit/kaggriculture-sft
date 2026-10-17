"""Fail-closed public-capsule -> V0 context bridge; never primitive-action BC.

PUBLIC_ANCHOR is a completely replayed public context, not a Plan. An explicitly
bound opening V0Continuation permits proposal generation; only an independently
verified supplied V0 path becomes PROPOSAL_ONLY. Later dawns are inventory only. ADMITTED_PAIR additionally requires independently replayed branches
and the unmodified disjoint regret admission gate. Neither an opened public
winner nor its primitive tape is ever a teacher action path.

The exact snapshot, seed, recorded suffix and identifying/outcome provenance are
trainer-only. ``proposal_context`` is the ONLY actor/proposer input accessor.
Serialized snapshots retain dictionary insertion order (including inventories)
and restore shared engine objects. Hashes bind bytes, not provenance honesty.
A capsule recheck replays all 719 joint actions and checks terminal state/cash;
its 1,440-original-observation verification remains the source producer's
attestation, because those observations are not in a capsule. Raw local replays
are additionally checked against all original actor boundaries by the BC verifier.

This module does not download, fit, invent checkpoints, register worlds, change
compiler semantics or supply a production branch runner. ``scan`` only replays
existing local capsules / complete raw JSON(.gz) episodes. ``scan-zip`` streams
one bounded episode at a time from an already-local daily archive, never extracts
it, and writes a compact JSONL index rather than accumulating the daily payload. Unknown file formats
and refused sources are counted, never silently called a complete daily harvest.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import io
import os
import stat
import zipfile
import zlib
from typing import Callable

from farm_neat.evaluation.common import bound, canonical, digest
from farm_neat.evaluation import public_episodes as public
from farm_neat.evaluation import v0_regret_pairs as paired
from farm_neat.evaluation.v0_continuation import V0Continuation
from farm_neat.game.exact_engine import (AttrDict, ExactKaggricultureEnvironment,
                                        assert_bundled_engine_identity)
from farm_neat.policy.v0_scorer import Prefix
from farm_neat.training import v0_regret_records as R
from farm_neat.training import v0_regret_admission as admission
from farm_neat.training import bc

SCHEMA = 'farm-v0-public-pair-bridge-v1'
DAWNS = tuple(range(0, 719, 24))
STATES = ('PUBLIC_ANCHOR', 'PROPOSAL_ONLY', 'ADMITTED_PAIR')
MAX_SOURCE_BYTES = 256 * 1024 * 1024
STATIC = 'RECORDED_STATIC_SUFFIX'
RESPONSIVE = 'NAMED_RESPONSIVE_POLICY'
FUTURE = 'FROZEN_V0_PATH_THEN_KEEP'
NONOPENING = 'NONOPENING_CONTINUATION_UNSUPPORTED'
STRATUM = 'RECORDED_STATIC_SUFFIX_OFF_POLICY'


class BridgeError(ValueError):
    """Stable code; no private/outcome-bearing exception payload."""
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _need(ok, code):
    if not ok:
        raise BridgeError(code)


def _same(a, b, code):
    _need(canonical(a) == canonical(b), code)


def _sealed(value, code='CONTENT_HASH_MISMATCH'):
    _need(type(value) is dict and value.get('content_sha256') == bound(value)['content_sha256'], code)


def _closed(value, keys, code):
    _need(type(value) is dict and set(value) == set(keys), code)


def _record_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except R.RecordError as exc:
        raise BridgeError(exc.code) from exc


def _selection(steps, seats):
    _need(type(steps) is list and bool(steps), 'INVALID_BOUNDARY')
    _need(all(type(t) is int for t in steps), 'INVALID_BOUNDARY')
    _need(len(set(steps)) == len(steps), 'DUPLICATE_BOUNDARY')
    _need(all(t in DAWNS for t in steps), 'INTRADAY_CUT')
    _need(type(seats) is list and bool(seats) and all(type(s) is int and s in (0, 1) for s in seats), 'INVALID_SEAT')
    _need(len(set(seats)) == len(seats), 'DUPLICATE_SEAT')
    return sorted(steps), sorted(seats)


def source_pins():
    paths = ('src/farm_neat/training/v0_public_preferences.py',
             'src/farm_neat/evaluation/public_episodes.py', 'src/farm_neat/training/bc.py',
             'src/farm_neat/training/v0_regret_admission.py',
             'src/farm_neat/training/teachers/v0_regret_proposals.py')
    return {**R.source_pins(), **{p: hashlib.sha256((R.ROOT/p).read_bytes()).hexdigest() for p in paths}}


def _read_episode(path: Path):
    path = Path(path)
    _need(path.is_file() and not path.is_symlink(), 'SOURCE_UNAVAILABLE')
    _need(path.stat().st_size <= MAX_SOURCE_BYTES, 'SOURCE_SIZE_LIMIT')
    return _episode_bytes(path.read_bytes(), compressed=path.suffix == '.gz')[:3]


def _episode_bytes(encoded: bytes, *, compressed: bool, limit: int = MAX_SOURCE_BYTES):
    _need(len(encoded) <= limit, 'SOURCE_SIZE_LIMIT')
    try:
        if compressed:
            with gzip.GzipFile(fileobj=io.BytesIO(encoded)) as stream:
                data = stream.read(limit + 1)
        else:
            data = encoded
        _need(len(data) <= limit, 'SOURCE_SIZE_LIMIT')
        value = R._loads(data.decode('utf-8'))
    except (ValueError, OSError, EOFError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, BridgeError):
            raise
        raise BridgeError('INVALID_SOURCE_JSON') from exc
    _need(type(value) is dict, 'UNSUPPORTED_SOURCE_FORMAT')
    file_hash = hashlib.sha256(encoded).hexdigest()
    if value.get('schema') == public.SCHEMA:
        return value, file_hash, 'CAPSULE_TERMINAL_REPLAY', len(data)
    if value.get('name') == 'kaggriculture':
        try:
            header = bc.verify_public_replay(value)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise BridgeError('INCOMPLETE_OR_MISMATCHED_RAW_REPLAY') from exc
        # No winner filtering and no call to public_winner_reports/primitive BC.
        cap = bound(dict(schema=public.SCHEMA, episode_id=header['episode_id'],
            seed=header['seed'], configuration=value['configuration'], rewards=value['rewards'],
            teams=value['info'].get('TeamNames'), focus={'seat': 0}, rival={},
            joint_actions=[[value['steps'][t][s]['action'] for s in (0, 1)] for t in range(1, 720)],
            engine=header['engine'], raw={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)},
            verified_actor_boundaries=1440))
        return cap, file_hash, 'RAW_ALL_ACTOR_BOUNDARIES_REVERIFIED', len(data)
    raise BridgeError('UNSUPPORTED_SOURCE_FORMAT')


def _snapshot(env):
    # A JSON string is intentional: sorted object serialization must not erase
    # iteration order of the exact runtime. This string never enters an actor.
    return json.dumps(dict(state=env.state, configuration=env.configuration,
        info=env.info, step=env.step_index, done=env.done), separators=(',', ':'), allow_nan=False)


def _restore_snapshot(snapshot):
    try:
        world = R._loads(snapshot)
        _closed(world, ('state','configuration','info','step','done'), 'SNAPSHOT_FIELDS')
        _need(type(world['step']) is int and world['step'] in DAWNS and world['done'] is False, 'INTRADAY_CUT')
        env = ExactKaggricultureEnvironment(seed=world['info']['seed'], configuration=world['configuration'])
        _same(dict(env.configuration), world['configuration'], 'CONFIGURATION_MISMATCH')
        rows = world['state']
        _need(type(rows) is list and len(rows) == 2, 'SNAPSHOT_FIELDS')
        env.state = []
        for seat, row in enumerate(rows):
            _closed(row, ('observation','action','reward','status','info'), 'SNAPSHOT_FIELDS')
            _need(row['status'] == 'ACTIVE' and row['observation']['player'] == seat, 'SNAPSHOT_SEAT_MISMATCH')
            env.state.append(AttrDict({**copy.deepcopy(row), 'observation':AttrDict(copy.deepcopy(row['observation']))}))
        for key in ('farms','market','town'):
            _same(rows[0]['observation'][key], rows[1]['observation'][key], 'SNAPSHOT_SHARED_STATE_MISMATCH')
            env.state[1].observation[key] = env.state[0].observation[key]
        env.configuration = AttrDict(copy.deepcopy(world['configuration']))
        env.info, env.step_index, env.done = copy.deepcopy(world['info']), world['step'], world['done']
        _need(_snapshot(env) == snapshot, 'SNAPSHOT_ROUNDTRIP_MISMATCH')
        return env
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        if isinstance(exc, BridgeError):
            raise
        raise BridgeError('INVALID_SNAPSHOT') from exc


def _opponent(value, cap, step, seat):
    if value is None:
        return dict(mode=STATIC, suffix=copy.deepcopy([j[1-seat] for j in cap['joint_actions'][step:]]),
                    field_strength_claim=False)
    _closed(value, ('mode','identity','continuation','field_strength_claim'), 'OPPONENT_PROTOCOL_FIELDS')
    _need(value['mode'] == RESPONSIVE and value['field_strength_claim'] is False, 'OPPONENT_PROTOCOL_MISMATCH')
    _closed(value['identity'], ('name','source_sha256'), 'OPPONENT_IDENTITY_MISMATCH')
    _record_call(R._text, value['identity']['name'], '$.opponent.name')
    _record_call(R._hash, value['identity']['source_sha256'], '$.opponent.source')
    _record_call(R._safe, value['continuation'])
    return copy.deepcopy(value)


def capsule_anchors(path: str | Path, *, steps: list[int], seats: list[int], opponent: dict | None = None) -> list[dict]:
    """All-or-nothing exact re-verification; cuts are fixed before any replay."""
    steps, seats = _selection(steps, seats)
    return _capsule_anchors(*_read_episode(Path(path)), steps=steps, seats=seats, opponent=opponent)


def _capsule_anchors(cap, file_hash, verification_kind, *, steps, seats, opponent=None):
    try:
        return _verified_anchors(cap, file_hash, verification_kind, steps=steps, seats=seats, opponent=opponent)
    except BridgeError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, RecursionError) as exc:
        raise BridgeError('INVALID_CAPSULE') from exc


def _verified_anchors(cap, file_hash, verification_kind, *, steps, seats, opponent=None):
    _sealed(cap, 'CAPSULE_HASH_MISMATCH')
    _need(type(cap.get('joint_actions')) is list and len(cap['joint_actions']) == 719
          and cap.get('verified_actor_boundaries') == 1440, 'INCOMPLETE_REPLAY')
    _same(cap.get('engine'), assert_bundled_engine_identity(), 'ENGINE_MISMATCH')
    _record_call(R._int, cap.get('episode_id'), '$.episode_id')
    _record_call(R._int, cap.get('seed'), '$.seed')
    _record_call(R._hash, cap.get('raw',{}).get('sha256'), '$.raw.sha256')
    _need(type(cap.get('configuration')) is dict and cap['configuration'].get('episodeSteps') == 720
          and cap['configuration'].get('turnsPerDay') == 24, 'CONFIGURATION_MISMATCH')
    cfg = cap['configuration']
    _need(all(cfg.get(k) == v for k,v in bc.features.CONFIG.items())
          and cfg.get('startingMoney') == 3000 and cfg.get('marketParams') == {}, 'CONFIGURATION_MISMATCH')
    env = ExactKaggricultureEnvironment(seed=cap['seed'], configuration=cfg)
    pending = []
    pins = source_pins()
    try:
        for step, joint in enumerate(cap['joint_actions']):
            _need(not env.done and env.step_index == step, 'INCOMPLETE_REPLAY')
            if step in steps:
                for seat in seats:
                    obs = env.agent_observation(seat)
                    facts = paired.observation_facts(obs)
                    snap = _snapshot(env)
                    rival = _opponent(opponent, cap, step, seat)
                    public_state = bc.actor_observation(obs)
                    public_state.pop('private', None)
                    pending.append(dict(schema=SCHEMA+'-anchor', state='PUBLIC_ANCHOR',
                        teacher_weight=0.0, seat=seat, step=step, observation=facts,
                        observation_sha256=digest(facts), public_state_sha256=digest(public_state),
                        snapshot_sha256=digest(snap), opponent_sha256=digest(rival),
                        trainer=dict(snapshot=snap, opponent=rival, source_pins=pins,
                            capsule_file_sha256=file_hash, capsule_content_sha256=cap['content_sha256'],
                            source_sha256=cap['raw']['sha256'], episode_id=cap['episode_id'],
                            seed=cap['seed'], provenance={k:copy.deepcopy(cap.get(k)) for k in
                                ('teams','focus','rival','rewards')},
                            verification=dict(kind=verification_kind, replayed_joint_actions=719,
                                source_attested_actor_boundaries=1440, terminal_verified=True))))
            _need(type(joint) is list and len(joint) == 2 and all(type(a) is dict for a in joint), 'INCOMPLETE_REPLAY')
            env.step(joint)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        if isinstance(exc, BridgeError):
            raise
        raise BridgeError('REPLAY_FAILED') from exc
    _need(env.done and env.step_index == 719 and all(row.status == 'DONE' for row in env.state), 'INCOMPLETE_REPLAY')
    _need([f['money'] for f in env.agent_observation(0)['farms']] == cap.get('rewards')
          and [row.reward for row in env.state] == cap.get('rewards'), 'TERMINAL_MISMATCH')
    return [bound(a) for a in pending]


def _anchor(anchor):
    _sealed(anchor)
    _need(anchor.get('schema') == SCHEMA+'-anchor' and anchor.get('state') == 'PUBLIC_ANCHOR'
          and anchor.get('teacher_weight') == 0, 'ANCHOR_STATE_MISMATCH')
    _selection([anchor['step']], [anchor['seat']])
    tr = anchor['trainer']
    _same(tr['source_pins'], source_pins(), 'SOURCE_MISMATCH')
    _need(tr['verification']['terminal_verified'] is True and tr['verification']['replayed_joint_actions'] == 719,
          'INCOMPLETE_REPLAY')
    _same(anchor['snapshot_sha256'], digest(tr['snapshot']), 'SNAPSHOT_HASH_MISMATCH')
    _same(anchor['opponent_sha256'], digest(tr['opponent']), 'OPPONENT_IDENTITY_MISMATCH')
    env = _restore_snapshot(tr['snapshot'])
    _need(env.step_index == anchor['step'] and env.info['seed'] == tr['seed'], 'SNAPSHOT_BOUNDARY_MISMATCH')
    if anchor['step'] == 0:
        initial = ExactKaggricultureEnvironment(seed=tr['seed'], configuration=env.configuration)
        _same(_snapshot(env), _snapshot(initial), 'INITIAL_STATE_MISMATCH')
    obs = env.agent_observation(anchor['seat'])
    _same(paired.observation_facts(obs), anchor['observation'], 'VISIBLE_OBSERVATION_MISMATCH')
    _same(digest(anchor['observation']), anchor['observation_sha256'], 'VISIBLE_OBSERVATION_MISMATCH')
    public_state = bc.actor_observation(obs); public_state.pop('private', None)
    _same(digest(public_state), anchor['public_state_sha256'], 'PUBLIC_STATE_MISMATCH')
    return env


def bind_continuation(anchor: dict, memory: dict, *, origin: str) -> dict:
    """Bind supplied memory; this NEVER infers memory from the public action tape.

    EXACT_V0_CONTINUATION is an upstream producer attestation, not something a
    hash proves. A missing/non-roundtripping controller is never reset to None.
    """
    _anchor(anchor)
    return bound(dict(schema=SCHEMA+'-continuation', snapshot_sha256=anchor['snapshot_sha256'],
        observation_sha256=anchor['observation_sha256'], seat=anchor['seat'], step=anchor['step'],
        source_pins_sha256=digest(anchor['trainer']['source_pins']), origin=origin, memory=copy.deepcopy(memory)))


def prepare_anchor(anchor: dict, *, continuation: dict | None = None) -> dict:
    """Return zero-weight readiness or a stable refusal; keep the valid anchor."""
    env = _anchor(anchor)
    out = dict(schema=SCHEMA+'-prepared', state='PUBLIC_ANCHOR', teacher_weight=0.0,
        anchor=copy.deepcopy(anchor), continuation_binding=copy.deepcopy(continuation),
        actor_context=None, actor_context_sha256=None, proposal_ready=False, pair_schema_compatible=False,
        pair_reason_codes=[] if anchor['step'] == 0 else [NONOPENING], reason_codes=[])
    try:
        _need(anchor['step'] == 0, NONOPENING)
        _need(continuation is not None, 'MISSING_V0_CONTINUATION')
        _sealed(continuation, 'CONTINUATION_HASH_MISMATCH')
        _closed(continuation, ('schema','snapshot_sha256','observation_sha256','seat','step',
            'source_pins_sha256','origin','memory','content_sha256'), 'CONTINUATION_FIELDS')
        _need(continuation['schema'] == SCHEMA+'-continuation', 'CONTINUATION_FIELDS')
        for k in ('snapshot_sha256','observation_sha256','seat','step'):
            _same(continuation[k], anchor[k], 'CONTINUATION_BINDING_MISMATCH')
        _same(continuation['source_pins_sha256'], digest(anchor['trainer']['source_pins']), 'SOURCE_MISMATCH')
        _need(continuation['origin'] in ('EXPLICIT_V0_INITIALIZATION','EXACT_V0_CONTINUATION'), 'CONTINUATION_ORIGIN')
        memory = continuation['memory']
        observation = env.agent_observation(anchor['seat'])
        _record_call(R._memory, memory, '$.continuation', observation=observation)
        agent = V0Continuation.from_dict(memory).restore(None, observation=observation)
        ctx = dict(schema=R.ACTOR_SCHEMA, observation=copy.deepcopy(anchor['observation']),
            prefix=Prefix(agent.plan).payload(), compiler_memory={k:copy.deepcopy(v) for k,v in memory.items() if k != 'game_id'})
        key = _record_call(R.actor_context_key, ctx)
        out.update(proposal_ready=True, actor_context=ctx, actor_context_sha256=key,
            pair_schema_compatible=anchor['step'] == 0,
            pair_reason_codes=[] if anchor['step'] == 0 else [NONOPENING])
    except BridgeError as exc:
        out['reason_codes'] = [exc.code]
    return bound(out)


def proposal_context(prepared: dict) -> dict:
    """Closed actor-only projection. Never pass a prepared/anchor envelope to a scorer."""
    _sealed(prepared)
    _need(prepared['anchor']['step'] == 0, NONOPENING)
    _need(prepared.get('state') == 'PUBLIC_ANCHOR' and prepared.get('proposal_ready') is True, 'NOT_PROPOSAL_READY')
    _record_call(R.actor_context_key, prepared['actor_context'])
    fresh = prepare_anchor(prepared['anchor'], continuation=prepared['continuation_binding'])
    _same(prepared, fresh, 'PREPARED_CONTEXT_MISMATCH')
    return copy.deepcopy(fresh['actor_context'])


def propose(prepared: dict, learner: dict, *, budget: dict, proposal_seed: int) -> dict:
    """Use the ordinary bounded generator; its seed is explicit, NOT the world seed."""
    from farm_neat.training.teachers import v0_regret_proposals as generator
    ctx = proposal_context(prepared)
    pool = _record_call(generator.propose, ctx, learner, budget=budget, seed=proposal_seed, archive=[])
    return bound(dict(schema=SCHEMA+'-proposals', state='PROPOSAL_ONLY', teacher_weight=0.0,
        pool=pool, trainer={'anchor_sha256':prepared['anchor']['content_sha256']},
        reason_codes=['UNSELECTED_PROPOSALS_NOT_TEACHERS']))


def actor_observation(env, seat):
    """Actor/compiler view, never the evaluator snapshot or its other-seat private state."""
    return bc.actor_observation(env.agent_observation(seat))


def actor_configuration(env):
    from farm_neat.game.exact_engine import DEFAULT_CONFIGURATION
    return {k:copy.deepcopy(v) for k,v in env.configuration.items()
            if k in DEFAULT_CONFIGURATION and k != 'seed'}


def _rng(prepared):
    return dict(engine='DAY_KEYED_PYTHON_RANDOM', seed=prepared['anchor']['trainer']['seed'],
        key_recipe='(seed * 1000003) XOR zero_based_day', persistent_rng_state=None,
        draw_alignment='ACTION_DEPENDENT_EMPTY_TILES', common_exogenous_events=False)


def _initial(prepared):
    ctx = proposal_context(prepared)
    a = prepared['anchor']
    return dict(snapshot_sha256=a['snapshot_sha256'], actor_context_sha256=digest(ctx),
        configuration_sha256=digest(R._loads(a['trainer']['snapshot'])['configuration']),
        continuation_sha256=digest(prepared['continuation_binding']['memory']),
        seat=a['seat'], step=a['step'], opponent_sha256=a['opponent_sha256'],
        rng_sha256=digest(_rng(prepared)), future_actor=FUTURE)


def _validate_path(prepared, path, arm, checkpoint):
    ctx = proposal_context(prepared)
    state, prefix, options = _record_call(R._actor, ctx)
    # Independent cache per path; no stored prefix is ever used to advance.
    _record_call(R._proposal, path,
        dict(parent_checkpoint_sha256=checkpoint, actor_context_sha256=digest(ctx)),
        prefix, state, options, {}, direct=arm == 'learner')


def bind_request(prepared: dict, path: dict, *, arm: str) -> dict:
    """Bind a supplied reachable request, not a public primitive-action label."""
    _need(arm in ('learner', 'challenger'), 'INVALID_ARM')
    checkpoint = (path.get('sampling_provenance', {}).get('checkpoint_sha256') if arm == 'learner'
                  else path.get('selection_receipt', {}).get('parent_checkpoint_sha256'))
    _record_call(R._hash, checkpoint, '$.checkpoint')
    ctx = proposal_context(prepared)
    if arm == 'challenger':
        _same(path.get('selection_receipt', {}).get('actor_context_sha256'), digest(ctx),
              'ACTOR_CONTEXT_MISMATCH')
    _validate_path(prepared, path, arm, checkpoint)
    return bound(dict(schema=SCHEMA+'-request', state='PROPOSAL_ONLY', teacher_weight=0.0,
        arm=arm, path=copy.deepcopy(path), actor_context_sha256=digest(ctx),
        trainer=dict(anchor_sha256=prepared['anchor']['content_sha256'], checkpoint_sha256=checkpoint)))


def _world(prepared):
    a = prepared['anchor']; tr = a['trainer']
    return dict(source_group_id='public-episode-'+str(tr['episode_id']),
        world_sha256=digest(dict(snapshot=a['snapshot_sha256'], seed=tr['seed'])),
        snapshot_sha256=a['snapshot_sha256'], rng_sha256=_initial(prepared)['rng_sha256'], seed=tr['seed'])


def _branch(prepared, paths, arm, phase, checkpoint, timeout):
    return bound(dict(schema=SCHEMA+'-branch', prepared=copy.deepcopy(prepared),
        initial=_initial(prepared), path=copy.deepcopy(paths[arm]), arm=arm, phase=phase,
        checkpoint_sha256=checkpoint, rng_binding=_rng(prepared), timeout_seconds=timeout))


def restore_branch(request: dict, *, opponent_factory=None):
    """Fresh exact environment and V0 compiler for EVERY attempt/replay.

    Runner callbacks are trusted code, not a sandbox. Their tape is subsequently
    checked against independently restored compiler and opponent instances. A
    responsive factory must expose the frozen {name, source_sha256} identity and
    accept its explicit continuation as its sole argument. No static suffix is
    silently substituted for a missing responsive policy.
    """
    from farm_neat.economy.v0.epochs import TrajectoryPolicy
    from farm_neat.evaluation.router_panel import _fingerprint
    _sealed(request, 'BRANCH_HASH_MISMATCH')
    _need(request.get('schema') == SCHEMA+'-branch' and request.get('arm') in ('learner','challenger'), 'BRANCH_FIELDS')
    p = request['prepared']; a = p['anchor']
    _same(request['initial'], _initial(p), 'BRANCH_IDENTITY_MISMATCH')
    _same(request['rng_binding'], _rng(p), 'RNG_COUPLING_MISMATCH')
    _need(a['step'] == 0, NONOPENING)
    _validate_path(p, request['path'], request['arm'], request['checkpoint_sha256'])
    env = _anchor(a)
    policy = TrajectoryPolicy({a['step']:request['path']['raw_actions'][:-1]})
    agent = V0Continuation.from_dict(p['continuation_binding']['memory']).restore(
        policy, observation=env.agent_observation(a['seat']))
    protocol = a['trainer']['opponent']
    if protocol['mode'] == STATIC:
        _closed(protocol, ('mode','suffix','field_strength_claim'), 'OPPONENT_PROTOCOL_FIELDS')
        _need(protocol['field_strength_claim'] is False and len(protocol['suffix']) == 719-a['step'],
              'OPPONENT_SUFFIX_MISMATCH')
        def rival(obs, cfg):
            return copy.deepcopy(protocol['suffix'][obs['step']-a['step']])
    else:
        _need(protocol['mode'] == RESPONSIVE and opponent_factory is not None, 'MISSING_RESPONSIVE_POLICY')
        _same(getattr(opponent_factory,'identity',None), protocol['identity'], 'OPPONENT_IDENTITY_MISMATCH')
        _same(_fingerprint(opponent_factory)['sha256'], protocol['identity']['source_sha256'], 'OPPONENT_SOURCE_MISMATCH')
        rival = opponent_factory(copy.deepcopy(protocol['continuation']))
        _need(callable(rival), 'MISSING_RESPONSIVE_POLICY')
    return env, agent, rival


def _replay_response(request, response, *, opponent_factory, instances):
    _sealed(response, 'RESPONSE_HASH_MISMATCH')
    _closed(response, ('schema','request_sha256','initial','status','joint_actions','content_sha256'), 'RESPONSE_FIELDS')
    _need(response['schema'] == SCHEMA+'-branch-response', 'RESPONSE_FIELDS')
    _same(response['request_sha256'], request['content_sha256'], 'BRANCH_IDENTITY_MISMATCH')
    _same(response['initial'], request['initial'], 'BRANCH_IDENTITY_MISMATCH')
    _need(response['status'] in ('COMPLETE','CENSORED'), 'BRANCH_STATUS')
    tape = response['joint_actions']
    _need(type(tape) is list and len(tape) <= 719-request['initial']['step'], 'BRANCH_STEP_BUDGET')
    env, agent, rival = restore_branch(request, opponent_factory=opponent_factory)
    _need(all(rival is not old for old in instances), 'SHARED_OPPONENT_INSTANCE')
    instances.append(rival)  # Keep references alive so object-id recycling is irrelevant.
    seat = request['initial']['seat']
    for joint in tape:
        _need(not env.done and type(joint) is list and len(joint) == 2, 'BRANCH_STEP_BUDGET')
        cfg = actor_configuration(env)
        own = agent(actor_observation(env,seat), cfg)
        other = rival(actor_observation(env,1-seat), cfg)
        _same(own, joint[seat], 'V0_COMPILER_ACTION_MISMATCH')
        _same(other, joint[1-seat], 'OPPONENT_ACTION_MISMATCH')
        env.step(joint)
    complete = env.done and env.step_index == 719 and all(r.status == 'DONE' for r in env.state)
    _need(complete == (response['status'] == 'COMPLETE'), 'CENSORED_BRANCH')
    cash = [float(row.reward) for row in env.state] if complete else None
    return bound(dict(initial=copy.deepcopy(request['initial']), complete=complete,
        wdl=int(cash[seat]>cash[1-seat])-int(cash[seat]<cash[1-seat]) if complete else None,
        cash=cash, actions_sha256=digest(tape), opponent_actions_sha256=digest([j[1-seat] for j in tape]),
        terminal_state_sha256=digest(_snapshot(env)),
        final_continuation_sha256=digest(V0Continuation.capture(agent).to_dict()), simulator_steps=len(tape)))


def compare(prepared: dict, learner: dict, challenger: dict, *, runner: Callable,
            opponent_factory=None, phase: str = 'discovery', timeout_seconds: float = 30.) -> dict:
    """One same-context pair plus independent replay of BOTH arms; always zero weight.

    The injected runner returns a bound {request_sha256, initial, status,
    joint_actions} response. All tape actions are re-executed through the exact
    engine AND checked against the frozen V0 compiler and rival law. Returned
    cash/predictions are not accepted. Full runs are retained for gate rechecks.
    This library function can be expensive; the scanner never invokes it.
    """
    from farm_neat.training.teachers.v0_regret_searcher import _deadline
    _need(phase in ('discovery','confirmation'), 'INVALID_PHASE')
    _record_call(R._number, timeout_seconds, '$.timeout', 0.001, 300.)
    ctx = proposal_context(prepared)
    _need(prepared['pair_schema_compatible'], NONOPENING)
    checkpoint = learner.get('sampling_provenance',{}).get('checkpoint_sha256')
    paths = dict(learner=learner, challenger=challenger)
    for arm,path in paths.items():
        _validate_path(prepared,path,arm,checkpoint)
    a = prepared['anchor']; tr = a['trainer']; initial = _initial(prepared)
    world = _world(prepared)
    out = dict(schema=SCHEMA+'-execution', state='PROPOSAL_ONLY', teacher_weight=0.0,
        phase=phase, source_group=world, source_family_id='public-source-'+tr['source_sha256'],
        prepared=copy.deepcopy(prepared), paths=copy.deepcopy(paths), checkpoint_sha256=checkpoint,
        actor_context_sha256=digest(ctx), initial=initial, source_pins=source_pins(),
        runner_binding=paired.execution_binding(runner), timeout_seconds=float(timeout_seconds),
        outcomes={}, runs=[], replay_verified=False, mean_wdl_delta=None, reason_codes=[], field_strength_claim=False)
    instances=[]
    try:
        for run_phase in ('execute','replay'):
            for arm in ('learner','challenger'):
                req = _branch(prepared,paths,arm,phase+'-'+run_phase,checkpoint,float(timeout_seconds))
                with _deadline(timeout_seconds):
                    response = runner(copy.deepcopy(req))
                    checked = _replay_response(req,response,opponent_factory=opponent_factory,instances=instances)
                out['runs'].append(dict(arm=arm, phase=run_phase, response=response, verified=checked))
                if run_phase == 'execute':
                    out['outcomes'][arm]=checked
                else:
                    _same(checked,out['outcomes'][arm],'REPLAY_MISMATCH')
        _need(all(v['complete'] for v in out['outcomes'].values()), 'CENSORED_BRANCH')
        out['replay_verified']=True
        out['mean_wdl_delta']=out['outcomes']['challenger']['wdl']-out['outcomes']['learner']['wdl']
        out['reason_codes']=['AWAITING_DISJOINT_ADMISSION']
    except (BridgeError, TimeoutError) as exc:
        out['reason_codes']=[exc.code if isinstance(exc,BridgeError) else 'BRANCH_TIMEOUT']
    except Exception as exc:
        out['reason_codes']=['BRANCH_RUNNER_FAILED']
        out['error_type']=type(exc).__name__
    return bound(out)



def recorded_suffix_binding(prepared_rows: list[dict]) -> dict:
    """Freeze a trainer-only per-world rival-tape identity BEFORE confirmation.

    Both arms still replay the identical tape within each world. Different
    public episodes need not share one tape. Put this binding's content_sha256
    in pair-v1 evaluation.opponent_hash AND its ordinary protocol seat lock;
    admission reconstructs the entire binding from independently checked arms.
    This is not an actor feature, a responsive policy, or proof of independence.
    """
    _need(type(prepared_rows) is list and bool(prepared_rows), 'MISSING_REPLAY_EVIDENCE')
    members, groups, sources = [], set(), set()
    for prepared in prepared_rows:
        proposal_context(prepared)
        a = prepared['anchor']; tr = a['trainer']
        _need(tr['opponent']['mode'] == STATIC, 'RECORDED_SUFFIX_REQUIRED')
        group = 'public-episode-'+str(tr['episode_id'])
        _need(group not in groups and tr['source_sha256'] not in sources,
              'DUPLICATE_EPISODE_SOURCE')
        groups.add(group); sources.add(tr['source_sha256'])
        members.append(dict(source_group_id=group, source_sha256=tr['source_sha256'],
            snapshot_sha256=a['snapshot_sha256'], opponent_sha256=a['opponent_sha256'],
            seat=a['seat'], step=a['step']))
    return bound(dict(schema=SCHEMA+'-suffix-cohort', stratum=STRATUM,
        members=sorted(members, key=lambda row: row['source_group_id']), field_strength_claim=False))


def admit_public_pair(pair: dict, *, protocol: dict, executions: list[dict], opponent_factory=None) -> dict:
    """Emit an unchanged, validated v1 pair ONLY after replay joins and ordinary admission.

    A single-seat v1 pair is the supported input, not an unrelated-state bundle.
    Distinct cuts/seats/files of one episode/raw source cannot masquerade as
    independent worlds. Static suffixes are an off-policy diagnostic/training
    stratum, NEVER a responsive opponent or field-strength claim. Positive receipt
    weight still requires the unchanged externally frozen disjoint gate.
    The opponent lock is either the predeclared recorded_suffix_binding hash
    or, for a uniform cohort only, the legacy single recorded-tape hash.
    """
    from farm_neat.training.teachers.v0_regret_searcher import _deadline
    out=dict(schema=SCHEMA+'-admission',state='PROPOSAL_ONLY',teacher_weight=0.0,
             pair=None, admission=None, reason_codes=[],field_strength_claim=False, stratum=STRATUM,
             recorded_suffix_binding=None)
    try:
        # Keep this bridge's explicit boundary even before generic v1 validation.
        context = pair.get('actor_context') if type(pair) is dict else None
        observation = context.get('observation') if type(context) is dict else None
        if type(observation) is dict and type(observation.get('step')) is int:
            _need(observation['step'] == 0, NONOPENING)
        pair=_record_call(R.validate_pair,pair)
        _need(type(executions) is list and bool(executions),'MISSING_REPLAY_EVIDENCE')
        seen_groups,seen_episodes,seen_sources=set(),set(),set()
        for ex in executions:
            _sealed(ex,'EXECUTION_HASH_MISMATCH')
            tr=ex['prepared']['anchor']['trainer']
            group=ex['source_group']['source_group_id']
            _need(group not in seen_groups and tr['episode_id'] not in seen_episodes
                  and tr['source_sha256'] not in seen_sources,'DUPLICATE_EPISODE_SOURCE')
            seen_groups.add(group);seen_episodes.add(tr['episode_id']);seen_sources.add(tr['source_sha256'])
        suffix_binding = recorded_suffix_binding([ex['prepared'] for ex in executions])
        tape_hashes = {row['opponent_sha256'] for row in suffix_binding['members']}
        allowed_locks = {suffix_binding['content_sha256']}
        if len(tape_hashes) == 1:
            allowed_locks.update(tape_hashes)  # Preserve the original single-tape wire contract.
        _need(pair['evaluation']['opponent_hash'] in allowed_locks, 'OPPONENT_IDENTITY_MISMATCH')
        out['recorded_suffix_binding'] = suffix_binding
        seen_groups,seen_episodes,seen_sources=set(),set(),set()
        groups={}
        for phase in ('discovery','confirmation'):
            for w in pair['evaluation'][phase+'_world_groups']:
                groups[w['source_group_id']]=(phase,w)
        for ex in executions:
            _sealed(ex,'EXECUTION_HASH_MISMATCH')
            _need(ex.get('schema')==SCHEMA+'-execution','EXECUTION_FIELDS')
            a=ex['prepared']['anchor']; tr=a['trainer']; group=ex['source_group']['source_group_id']
            _need(group not in seen_groups and tr['episode_id'] not in seen_episodes
                  and tr['source_sha256'] not in seen_sources,'DUPLICATE_EPISODE_SOURCE')
            seen_groups.add(group);seen_episodes.add(tr['episode_id']);seen_sources.add(tr['source_sha256'])
            _need(group in groups,'UNREGISTERED_REPLAY_WORLD')
            _same((ex['phase'],ex['source_group']),groups[group],'WORLD_REPLAY_MISMATCH')
            _same(ex['source_group'], _world(ex['prepared']), 'WORLD_REPLAY_MISMATCH')
            _same(ex['initial'], _initial(ex['prepared']), 'BRANCH_IDENTITY_MISMATCH')
            _same(ex['source_family_id'], 'public-source-'+tr['source_sha256'], 'SOURCE_FAMILY_MISMATCH')
            _same(ex['source_pins'],source_pins(),'SOURCE_MISMATCH')
            _same(proposal_context(ex['prepared']),pair['actor_context'],'ACTOR_CONTEXT_MISMATCH')
            _same(ex['prepared']['continuation_binding']['memory'],pair['continuation'],'CONTINUATION_MISMATCH')
            _same(ex['checkpoint_sha256'],pair['identity']['parent_checkpoint_sha256'],'CHECKPOINT_MISMATCH')
            _need(tr['opponent']['mode']==STATIC,'RECORDED_SUFFIX_REQUIRED')
            _need(ex['replay_verified'] is True and len(ex['runs'])==4,'REPLAY_NOT_VERIFIED')
            for arm in ('learner','challenger'):
                _same(ex['paths'][arm],pair[arm],'FROZEN_PATH_MISMATCH')
            # No caller-supplied replay_verified flag or terminal cash can bypass
            # re-executing the retained tape with independently restored compilers.
            instances=[]; checked_runs={}
            for run in ex['runs']:
                key=(run['arm'],run['phase'])
                _need(key not in checked_runs and key[0] in ('learner','challenger')
                      and key[1] in ('execute','replay'),'DUPLICATE_BRANCH')
                req=_branch(ex['prepared'],ex['paths'],key[0],ex['phase']+'-'+key[1],
                            ex['checkpoint_sha256'],ex['timeout_seconds'])
                _record_call(R._number,ex['timeout_seconds'],'$.timeout',0.001,300.)
                with _deadline(ex['timeout_seconds']):
                    checked=_replay_response(req,run['response'],opponent_factory=opponent_factory,instances=instances)
                _same(checked,run['verified'],'REPLAY_MISMATCH')
                _same(checked,ex['outcomes'][key[0]],'REPLAY_MISMATCH')
                _need(checked['complete'],'CENSORED_BRANCH')
                checked_runs[key]=req['content_sha256']
            delta = ex['outcomes']['challenger']['wdl'] - ex['outcomes']['learner']['wdl']
            _need(delta <= 0 or ex['outcomes']['challenger']['wdl'] == 1, 'NO_TERMINAL_WIN_IMPROVEMENT')
            row=next((r for r in pair['evaluation']['seat_rows'] if r['source_group_id']==group),None)
            _need(row is not None,'MISSING_REPLAY_ROW')
            for arm in ('learner','challenger'):
                _same(row[arm+'_wdl'],ex['outcomes'][arm]['wdl'],'TERMINAL_WDL_MISMATCH')
                _same(row[arm+'_attempt_id'],checked_runs[(arm,'execute')],'ATTEMPT_MISMATCH')
            matching=[t for t in protocol['tests'] if any(l['pair_id']==pair['identity']['pair_id'] for l in t['seat_locks'])]
            _need(len(matching)==1,'UNREGISTERED_PAIR')
            _same(matching[0]['source_families'].get(group),ex['source_family_id'],'SOURCE_FAMILY_MISMATCH')
        _need(seen_groups==set(groups),'MISSING_REPLAY_EVIDENCE')
        gate=admission.admit_pair(pair,protocol=protocol)
        out['admission']=gate
        _need(gate['actor_training_admissible'] and gate['teacher_weight']>0,
              gate['reason_codes'][0] if gate['reason_codes'] else 'ADMISSION_REFUSED')
        out.update(state='ADMITTED_PAIR',pair=_record_call(R.validate_pair,pair),teacher_weight=gate['teacher_weight'])
    except BridgeError as exc:
        out['reason_codes']=[exc.code]
    except TimeoutError:
        out['reason_codes']=['BRANCH_TIMEOUT']
    except (KeyError,TypeError,ValueError,IndexError):
        out['reason_codes']=['INVALID_EXECUTION_JSON']
    return bound(out)


def _scan_counts():
    return dict(files_seen=0, sources_verified=0, sources_refused=0, unsupported_files=0,
        duplicate_episodes=0, anchors_found=0, opening_anchors=0, nonopening_refused=0,
        proposal_ready=0, proposal_refused=0, admitted_pairs=0, native_games=0,
        recorded_transitions_reverified=0)


def _scan_row(source, decoded, *, steps, seats, counts, seen):
    cap, file_hash, kind = decoded
    _need(type(cap.get('raw')) is dict, 'INVALID_CAPSULE')
    episode, raw = cap.get('episode_id'), cap.get('raw', {}).get('sha256')
    _record_call(R._int, episode, '$.episode_id')
    _record_call(R._hash, raw, '$.raw.sha256')
    _need(episode not in seen[0] and raw not in seen[1] and file_hash not in seen[2],
          'DUPLICATE_EPISODE_SOURCE')
    anchors = _capsule_anchors(cap, file_hash, kind, steps=steps, seats=seats)
    for keys, value in zip(seen, (episode, raw, file_hash)):
        keys.add(value)
    counts['sources_verified'] += 1
    counts['recorded_transitions_reverified'] += 719
    counts['anchors_found'] += len(anchors)
    counts['opening_anchors'] += sum(a['step'] == 0 for a in anchors)
    counts['nonopening_refused'] += sum(a['step'] != 0 for a in anchors)
    counts['proposal_refused'] += len(anchors)  # Never invent a model/memory/path.
    return dict(source=str(source), episode_id=episode, source_sha256=raw,
        capsule_file_sha256=file_hash, state='PUBLIC_ANCHOR', verification_kind=kind,
        reason_codes=['MISSING_V0_CONTINUATION'],
        anchors=[dict(**{k:a[k] for k in ('seat','step','content_sha256','snapshot_sha256',
                                        'observation_sha256','opponent_sha256')},
                      reason_codes=['MISSING_V0_CONTINUATION'] if a['step'] == 0 else [NONOPENING])
                 for a in anchors])


def _scan_refusal(source, code, counts):
    if code == 'DUPLICATE_EPISODE_SOURCE':
        counts['duplicate_episodes'] += 1
    elif code == 'UNSUPPORTED_SOURCE_FORMAT':
        counts['unsupported_files'] += 1
    else:
        counts['sources_refused'] += 1
    return dict(source=str(source), state='REFUSED', reason_codes=[code])


def _scan_result(counts, steps, seats):
    return dict(schema=SCHEMA+'-scan', status='PASS' if counts['sources_verified'] and
        not (counts['sources_refused'] or counts['unsupported_files']) else 'REFUSED_OR_PARTIAL',
        steps=steps, seats=seats, counts=counts,
        direct_pair_extraction='CLOSED_NEGATIVE_WITHOUT_SUPPLIED_REACHABLE_V0_REQUESTS',
        limits=['OPENED_PUBLIC_CONTEXTS_NOT_HELDOUT_CONFIRMATION', 'NO_PRODUCTION_CHECKPOINT',
                'SOURCE_CAPSULE_OR_RAW_MEMBER_REQUIRED_TO_REHYDRATE_ANCHORS',
                'CAPSULE_ORIGINAL_ACTOR_BOUNDARIES_ATTESTED_NOT_REAVAILABLE',
                'NO_FIELD_STRENGTH_CLAIM'])


def scan(path: str | Path, *, steps: list[int], seats: list[int]) -> dict:
    """Read-only local scan; no proposals, branch games, or positive weights."""
    steps, seats = _selection(steps, seats)
    root = Path(path)
    _need(root.exists() and not root.is_symlink(), 'SOURCE_UNAVAILABLE')
    files = [root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file())
    counts, rows, seen = _scan_counts(), [], (set(), set(), set())
    for source in files:
        counts['files_seen'] += 1
        try:
            _need(source.name.endswith(('.json', '.json.gz')), 'UNSUPPORTED_SOURCE_FORMAT')
            rows.append(_scan_row(source, _read_episode(source), steps=steps, seats=seats, counts=counts, seen=seen))
        except BridgeError as exc:
            rows.append(_scan_refusal(source, exc.code, counts))
    return bound(dict(**_scan_result(counts, steps, seats), sources=rows))


def scan_zip(path: str | Path, *, index_out: str | Path, steps: list[int], seats: list[int],
             max_members: int = 100000, max_source_bytes: int = MAX_SOURCE_BYTES,
             max_total_bytes: int = 32 * 1024**3) -> dict:
    """Bounded read-only ZIP streaming; at most one decoded episode in memory.

    JSON and JSON.gz members use the same exact replay verifier as ``scan``.
    No members are extracted; each compact row is flushed to an exclusive-create
    JSONL index. Limits count ALL non-directory members and both transport and
    decompressed episode bytes; hitting a limit never claims complete coverage.
    ZIP central-directory metadata and duplicate identity sets remain in memory.
    """
    steps, seats = _selection(steps, seats)
    for value in (max_members, max_source_bytes, max_total_bytes):
        _need(type(value) is int and value > 0, 'INVALID_SCAN_BUDGET')
    _need(max_source_bytes <= MAX_SOURCE_BYTES, 'INVALID_SCAN_BUDGET')
    path, index_out = Path(path), Path(index_out)
    _need(not any(p.is_symlink() for p in (index_out, *index_out.parents)), 'UNSAFE_OUTPUT')
    _need(not index_out.exists(), 'OUTPUT_EXISTS')
    counts, seen, names = _scan_counts(), (set(), set(), set()), set()
    reasons, used, complete, index_hash = [], 0, True, hashlib.sha256()
    try:
        flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BridgeError('SOURCE_UNAVAILABLE') from exc
    try:
        with os.fdopen(descriptor, 'rb') as archive_stream:
            before = os.fstat(archive_stream.fileno())
            _need(stat.S_ISREG(before.st_mode), 'SOURCE_UNAVAILABLE')
            archive_sha256 = hashlib.file_digest(archive_stream, 'sha256').hexdigest()
            archive_stream.seek(0)
            with zipfile.ZipFile(archive_stream) as archive:
                index_out.parent.mkdir(parents=True, exist_ok=True)
                with index_out.open('xb') as output:
                    for member in archive.infolist():
                        if member.is_dir():
                            continue
                        if counts['files_seen'] == max_members:
                            reasons.append('ARCHIVE_MEMBER_BUDGET'); complete = False; break
                        if used + member.file_size > max_total_bytes:
                            reasons.append('ARCHIVE_BYTE_BUDGET'); complete = False; break
                        counts['files_seen'] += 1
                        source = str(path)+'!'+member.filename
                        budget_limited = False
                        try:
                            parts = PurePosixPath(member.filename).parts
                            mode = member.external_attr >> 16
                            _need(bool(parts) and not member.filename.startswith('/') and
                                  '..' not in parts and '\\' not in member.filename and
                                  ':' not in parts[0] and not stat.S_ISLNK(mode) and
                                  (not stat.S_IFMT(mode) or stat.S_ISREG(mode)) and
                                  not member.flag_bits & 1, 'UNSAFE_ARCHIVE_MEMBER')
                            _need(member.filename not in names, 'DUPLICATE_ARCHIVE_MEMBER')
                            names.add(member.filename)
                            _need(member.file_size <= max_source_bytes, 'SOURCE_SIZE_LIMIT')
                            _need(member.filename.endswith(('.json', '.json.gz')), 'UNSUPPORTED_SOURCE_FORMAT')
                            with archive.open(member) as stream:
                                encoded = stream.read(max_source_bytes + 1)
                            used += len(encoded)
                            remaining = max_total_bytes - used
                            limit = min(max_source_bytes, remaining) if member.filename.endswith('.gz') else max_source_bytes
                            budget_limited = limit < max_source_bytes
                            decoded = _episode_bytes(encoded, compressed=member.filename.endswith('.gz'), limit=limit)
                            if member.filename.endswith('.gz'):
                                used += decoded[3]  # Actual decoded bytes, NOT capsule raw-source attestation.
                            row = _scan_row(source, decoded[:3], steps=steps, seats=seats, counts=counts, seen=seen)
                        except BridgeError as exc:
                            code = exc.code
                            if code == 'SOURCE_SIZE_LIMIT' and budget_limited:
                                code = 'ARCHIVE_BYTE_BUDGET'
                                reasons.append(code); complete = False
                            row = _scan_refusal(source, code, counts)
                        except (OSError, EOFError, ValueError, zipfile.BadZipFile, RuntimeError, zlib.error):
                            row = _scan_refusal(source, 'INVALID_ARCHIVE_MEMBER', counts)
                        line = (canonical(bound(row))+'\n').encode()
                        output.write(line); output.flush(); index_hash.update(line)
                        if not complete:
                            break
            after = os.fstat(archive_stream.fileno())
            _need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                  (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'ARCHIVE_MUTATED')
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise BridgeError('INVALID_ARCHIVE') from exc
    result = _scan_result(counts, steps, seats)
    if not complete:
        result['status'] = 'REFUSED_OR_PARTIAL'
    return bound(dict(**result, archive_sha256=archive_sha256, index=str(index_out),
        index_sha256=index_hash.hexdigest(), complete_scan=complete, reason_codes=reasons,
        bytes_read=used, budget=dict(max_members=max_members, max_source_bytes=max_source_bytes,
                                    max_total_bytes=max_total_bytes)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('scan', 'scan-zip'):
        p = sub.add_parser(command, help='Reverify local episodes; emit only context anchors')
        p.add_argument('--input', type=Path, required=True)
        p.add_argument('--steps', type=int, nargs='+', default=[0])
        p.add_argument('--seats', type=int, nargs='+', default=[0,1])
        p.add_argument('--out', type=Path)
        if command == 'scan-zip':
            p.add_argument('--index-out', type=Path, required=True)
            p.add_argument('--max-members', type=int, default=100000)
            p.add_argument('--max-source-bytes', type=int, default=MAX_SOURCE_BYTES)
            p.add_argument('--max-total-bytes', type=int, default=32 * 1024**3)
    args = parser.parse_args(argv)
    try:
        if args.out:
            _need(not args.out.exists(), 'OUTPUT_EXISTS')
            _need(not any(p.is_symlink() for p in (args.out, *args.out.parents)), 'UNSAFE_OUTPUT')
        if args.command == 'scan-zip':
            _need(args.out is None or args.out.resolve() != args.index_out.resolve(), 'OUTPUT_PATH_COLLISION')
            result = scan_zip(args.input, index_out=args.index_out, steps=args.steps, seats=args.seats,
                max_members=args.max_members, max_source_bytes=args.max_source_bytes, max_total_bytes=args.max_total_bytes)
        else:
            result = scan(args.input, steps=args.steps, seats=args.seats)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open('x') as stream:
                stream.write(canonical(result)+'\n')
        print(canonical(result))
        return 0 if result['status'] == 'PASS' else 2
    except BridgeError as exc:
        print(canonical(dict(schema=SCHEMA+'-error', status='REFUSED', reason_codes=[exc.code])))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
