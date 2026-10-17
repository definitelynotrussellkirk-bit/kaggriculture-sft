"""Bounded public-opening regret preparation, not primitive-tape imitation.

The only cohort is the manifest's first four distinct episodes (discovery), then
sixteen (confirmation). Individual JSON files are byte-bound before any replay.
``receipt`` is the explicitly exposed, single-source CPU preparation exception;
it cannot create a corpus. ``run --execute`` is maintainer-only and never fits.

Replay, actor projection, continuation validation, proposal generation, genuine
raw sampling, paired replay, admission and training remain their existing owners.
All source/team/reward/seed/suffix data stays in trainer envelopes. A positive pair
is RECORDED_STATIC_SUFFIX_OFF_POLICY evidence, never expert V0 BC or field strength.
"""
from __future__ import annotations

import argparse
import copy
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import stat
from typing import Callable

from farm_neat.evaluation.common import bound, canonical, digest
from farm_neat.training import v0_public_preferences as B
from farm_neat.training import v0_regret_records as R

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'farm-v0-public-warmstart-v1'
MANIFEST_SHA256 = 'a8243cecbe223eb4828db6f42b7cc9d0e5f7e760ce42d1716b70c9892f47b9c5'
# Already-acquired, maintainer-supplied identities; never fetch or reselect them.
ACQUISITION_REF = dict(path='work/datasets/public-episodes/daily/SELECTION.json',
    sha256='2dda2c0f720fa951c787da4cc8a8d086f247f835b4e5a0aebfc8a42f70f4084a')
EXPOSED_EPISODE = 111424871
EXPOSED_SHA256 = 'b5391964a3b4a4d9caed8ca9eaf82e62cf2ff2ca3bb6f8ef98182c165bfa76b8'
EXPOSED_BYTES = 33870075
DISCOVERY = 4
CONFIRMATION = 16
EPISODES = DISCOVERY + CONFIRMATION
MANIFEST_FIELDS = ('episode_id', 'create_time', 'avg_score', 'min_score', 'sum_score',
                   'agent_count', 'size_bytes')
# Two attempts: the obligatory learner control and exactly one KEEP challenger.
# A duplicate/refused KEEP is retained; it does not purchase a replacement draw.
PROPOSAL_BUDGET = dict(max_candidates=2, max_path_edits=1, max_archive_records=0)
LIMITS = [B.STRATUM, 'RECORDED_RIVAL_DOES_NOT_RESPOND_TO_COUNTERFACTUAL_PLAY',
          'NO_PUBLIC_FIELD_STRENGTH_CLAIM', 'NO_EXPERT_V0_STRATEGY_OR_RAW_TAPE_BC',
          'INDEPENDENCE_AND_UNDISCLOSED_EXPOSURE_REQUIRE_MAINTAINER_ATTESTATION',
          'ZERO_ADMITTED_PAIRS_MEANS_NO_FIT_NO_PROMOTION_NO_RELABELING']


class WarmstartError(ValueError):
    """Stable refusal code with trainer-only, optional source diagnostics."""
    def __init__(self, code: str, details=None):
        self.code, self.details = code, details
        super().__init__(code)


def _need(condition, code, details=None):
    if not condition:
        raise WarmstartError(code, details)


def _sealed(value, schema):
    _need(type(value) is dict and value.get('schema') == schema, 'SCHEMA_MISMATCH')
    _need(value.get('content_sha256') == bound(value)['content_sha256'], 'CONTENT_HASH_MISMATCH')


def _hash(value):
    _need(isinstance(value, str) and len(value) == 64 and
          all(c in '0123456789abcdef' for c in value), 'INVALID_SHA256')
    return value


def _safe_path(path):
    path = Path(path)
    _need('..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
          'UNSAFE_PATH')
    return path


def _read_bytes(path, *, expected_bytes=None, expected_sha256=None,
                limit=B.MAX_SOURCE_BYTES):
    """Hash/read one regular file, from one descriptor; never scan a directory."""
    path = _safe_path(path)
    if expected_sha256 is not None:
        _hash(expected_sha256)
    if expected_bytes is not None:
        _need(type(expected_bytes) is int and 0 < expected_bytes <= limit, 'SOURCE_SIZE_LIMIT')
    try:
        flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
        with os.fdopen(os.open(path, flags), 'rb') as stream:
            before = os.fstat(stream.fileno())
            _need(stat.S_ISREG(before.st_mode), 'REGULAR_FILE_REQUIRED')
            _need(before.st_size <= limit, 'SOURCE_SIZE_LIMIT')
            _need(expected_bytes is None or before.st_size == expected_bytes, 'SOURCE_BYTES_MISMATCH')
            data = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        _need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
              (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
              'SOURCE_CHANGED_DURING_READ')
    except FileNotFoundError as exc:
        raise WarmstartError('MISSING_SELECTED_MEMBER', str(path)) from exc
    except OSError as exc:
        raise WarmstartError('SOURCE_UNAVAILABLE', str(path)) from exc
    _need(len(data) == before.st_size and len(data) <= limit, 'SOURCE_BYTES_MISMATCH')
    _need(expected_sha256 is None or hashlib.sha256(data).hexdigest() == expected_sha256,
          'SOURCE_HASH_MISMATCH', str(path))
    return data


def source_pins():
    paths = ('src/farm_neat/training/v0_public_warmstart.py',
             'src/farm_neat/rollout/v0_regret_native.py',
             'src/farm_neat/rollout/v0_regret_collect.py',
             'src/farm_neat/training/v0_regret_train.py',
             'src/farm_neat/training/v0_selfplay.py',
             'src/farm_neat/evaluation/v0_policy.py',
             'src/farm_neat/evaluation/v0_regret_pairs.py',
             'project/configs/v0-regret-teacher.json', 'project/runtime/interpreter.toml')
    return B.source_pins() | {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def select_manifest(manifest: str | Path, *, expected_sha256: str,
                    episode_dir: str | Path) -> dict:
    """Preserve declared descending avg_score order, including CSV tie order.

    The byte hash binds ties, ordering and membership. Scores are trainer-only;
    no replay reward, winner flag, filename glob or branch outcome selects a row.
    """
    manifest, episode_dir = _safe_path(manifest), _safe_path(episode_dir)
    data = _read_bytes(manifest, expected_sha256=expected_sha256, limit=2 * 1024 * 1024)
    try:
        reader = csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))
        _need(tuple(reader.fieldnames or ()) == MANIFEST_FIELDS, 'MANIFEST_FIELDS')
        rows, seen, previous = [], set(), None
        for rank, row in enumerate(reader, 1):
            _need(set(row) == set(MANIFEST_FIELDS) and all(v is not None for v in row.values()),
                  'MANIFEST_FIELDS')
            episode, size, agents = int(row['episode_id']), int(row['size_bytes']), int(row['agent_count'])
            score = Decimal(row['avg_score'])
            _need(episode > 0 and str(episode) == row['episode_id'] and agents == 2 and
                  0 < size <= B.MAX_SOURCE_BYTES and score.is_finite(), 'INVALID_MANIFEST_ROW')
            _need(episode not in seen, 'DUPLICATE_MANIFEST_EPISODE')
            _need(previous is None or score <= previous, 'MANIFEST_SCORE_ORDER')
            previous = score
            seen.add(episode)
            if rank <= EPISODES:
                rows.append(dict(rank=rank, episode_id=episode,
                    split='discovery' if rank <= DISCOVERY else 'confirmation',
                    path=str(episode_dir / f'{episode}.json'), bytes=size, avg_score=row['avg_score']))
        _need(len(rows) == EPISODES, 'MISSING_MANIFEST_MEMBERS')
    except (ValueError, UnicodeError, csv.Error, InvalidOperation, TypeError) as exc:
        if isinstance(exc, WarmstartError):
            raise
        raise WarmstartError('INVALID_MANIFEST') from exc
    _need(not any(r['episode_id'] == EXPOSED_EPISODE and r['split'] != 'discovery' for r in rows),
          'EXPOSED_SOURCE_NOT_CONFIRMATION')
    return bound(dict(schema=SCHEMA + '-selection', manifest=dict(path=str(manifest),
        bytes=len(data), sha256=expected_sha256), episode_dir=str(episode_dir),
        order='CSV_AVG_SCORE_DESCENDING_STABLE_TIES', discovery_count=DISCOVERY,
        confirmation_count=CONFIRMATION, episodes=rows))


def validate_selection(selected: dict) -> dict:
    _sealed(selected, SCHEMA + '-selection')
    ref = selected['manifest']
    expected = select_manifest(ref['path'], expected_sha256=ref['sha256'], episode_dir=selected['episode_dir'])
    _need(selected == expected, 'SELECTION_DRIFT')
    return expected


def _known_source(row):
    if row['episode_id'] == EXPOSED_EPISODE:
        _need(row['bytes'] == EXPOSED_BYTES and row['sha256'] == EXPOSED_SHA256,
              'EXPOSED_SOURCE_IDENTITY_MISMATCH')
        _need(row['split'] == 'discovery', 'EXPOSED_SOURCE_NOT_CONFIRMATION')


def _acquisition_rows(selected: dict, reference: dict) -> dict[int, str]:
    """Join a byte-pinned acquisition ledger to the unchanged 4/16 selection.

    An acquisition claim is not replay evidence. For the supplied daily manifest,
    only the supplied ledger hash is accepted, even when its path is relocated.
    This reads metadata only; source JSONs are still individually verified below.
    """
    _need(type(reference) is dict and set(reference) == {'path', 'sha256'},
          'ACQUISITION_REFERENCE_FIELDS')
    _hash(reference['sha256'])
    if selected['manifest']['sha256'] == MANIFEST_SHA256:
        _need(reference['sha256'] == ACQUISITION_REF['sha256'], 'ACQUISITION_IDENTITY_MISMATCH')
    data = _read_bytes(reference['path'], expected_sha256=reference['sha256'], limit=2 * 1024 * 1024)
    document = R._loads(data.decode('utf-8'))
    _need(type(document) is dict and document.get('manifest_sha256') == selected['manifest']['sha256'],
          'ACQUISITION_MANIFEST_MISMATCH')
    rows = document.get('episodes')
    _need(type(rows) is list and len(rows) == EPISODES, 'ACQUISITION_MEMBERS')
    hashes, seen = {}, set()
    for expected, row in zip(selected['episodes'], rows):
        _need(type(row) is dict and set(row) == {'episode_id', 'file', 'bytes', 'split', 'sha256'},
              'ACQUISITION_FIELDS')
        _need(row['episode_id'] == expected['episode_id'] and row['bytes'] == expected['bytes'] and
              row['split'] == expected['split'] and row['file'] == Path(expected['path']).name,
              'ACQUISITION_SELECTION_MISMATCH')
        value = _hash(row['sha256'])
        _known_source(row)
        _need(value not in seen, 'DUPLICATE_EPISODE_SOURCE')
        hashes[row['episode_id']] = value
        seen.add(value)
    _need(document.get('total_source_bytes') == sum(r['bytes'] for r in rows), 'ACQUISITION_BYTES_MISMATCH')
    return hashes


def pin_sources(selected: dict, *, acquisition: dict | None = None) -> dict:
    """Freeze twenty individual sources before replay, preserving known hashes.

    The supplied real manifest automatically joins its already-acquired ledger.
    Other manifests may supply a byte-pinned ledger explicitly. Without one, the
    original first-acquisition behavior is retained for new/diagnostic datasets.
    No source identity, split, missing member or failed checksum is substituted.
    """
    selected = validate_selection(selected)
    if acquisition is None and selected['manifest']['sha256'] == MANIFEST_SHA256:
        acquisition = copy.deepcopy(ACQUISITION_REF)
    expected_hashes = _acquisition_rows(selected, acquisition) if acquisition is not None else {}
    rows, failures, hashes = [], [], set()
    for member in selected['episodes']:
        try:
            data = _read_bytes(member['path'], expected_bytes=member['bytes'],
                               expected_sha256=expected_hashes.get(member['episode_id']))
            row = dict(member, sha256=hashlib.sha256(data).hexdigest())
            _known_source(row)
            _need(row['sha256'] not in hashes, 'DUPLICATE_EPISODE_SOURCE')
            hashes.add(row['sha256']); rows.append(row)
        except WarmstartError as exc:
            failures.append(dict(episode_id=member['episode_id'], path=member['path'], code=exc.code))
    if failures:
        raise WarmstartError(failures[0]['code'], failures)
    provenance = {} if acquisition is None else dict(acquisition=copy.deepcopy(acquisition))
    return bound(dict(schema=SCHEMA + '-sources', selection=selected, episodes=rows,
        source_pins=source_pins(), scope='ALL_SELECTED_INDIVIDUAL_RAW_JSON_BYTES_PINNED_BEFORE_REPLAY',
        **provenance))


def validate_sources(locked: dict, *, read_files: bool = True) -> dict:
    _sealed(locked, SCHEMA + '-sources')
    fields = {'schema', 'selection', 'episodes', 'source_pins', 'scope', 'content_sha256'}
    _need(set(locked) in (fields, fields | {'acquisition'}), 'SOURCE_LOCK_FIELDS')
    selected = validate_selection(locked['selection'])
    if selected['manifest']['sha256'] == MANIFEST_SHA256:
        _need('acquisition' in locked, 'ACQUISITION_REFERENCE_REQUIRED')
    expected_hashes = (_acquisition_rows(selected, locked['acquisition'])
                       if 'acquisition' in locked else {})
    _need(locked['source_pins'] == source_pins(), 'SOURCE_CODE_DRIFT')
    rows = locked['episodes']
    _need(type(rows) is list and len(rows) == EPISODES, 'MISSING_SELECTED_MEMBER')
    hashes = set()
    for expected, row in zip(selected['episodes'], rows):
        _need(type(row) is dict and set(row) == set(expected) | {'sha256'} and
              {k: row[k] for k in expected} == expected, 'SELECTION_DRIFT')
        _hash(row['sha256']); _known_source(row)
        _need(not expected_hashes or row['sha256'] == expected_hashes[row['episode_id']],
              'ACQUISITION_SOURCE_MISMATCH')
        _need(row['sha256'] not in hashes, 'DUPLICATE_EPISODE_SOURCE')
        hashes.add(row['sha256'])
    # Structural/dedup checks finish before reads, and all reads before replay.
    if read_files:
        for row in rows:
            _read_bytes(row['path'], expected_bytes=row['bytes'], expected_sha256=row['sha256'])
    return copy.deepcopy(locked)


def empty_continuation() -> dict:
    """The declared empty factor-4 / zero-flow initialization, never inferred."""
    from farm_neat.economy.v0.epochs import V0Agent
    from farm_neat.economy.v0 import vocabulary as V
    from farm_neat.evaluation.v0_continuation import V0Continuation
    return V0Continuation.capture(V0Agent(None, labor_factor=4.0, hands_mode='daily',
        belief={p: 0.0 for p in V.PRODUCTS}, game_id=SCHEMA + '-initial')).to_dict()


def prepare_opening(anchor: dict) -> dict:
    _need(anchor.get('step') == 0 and type(anchor.get('step')) is int, 'STEP_ZERO_REQUIRED')
    memory = empty_continuation()
    continuation = B.bind_continuation(anchor, memory, origin='EXPLICIT_V0_INITIALIZATION')
    prepared = B.prepare_anchor(anchor, continuation=continuation)
    _need(prepared['proposal_ready'] and prepared['pair_schema_compatible'],
          'OPENING_NOT_PROPOSAL_READY', prepared['reason_codes'])
    B.proposal_context(prepared)
    return prepared


def _prepare_source(row: dict, *, seats: list[int]) -> list[dict]:
    # Feed exactly the already-checked bytes into the existing decoder/verifier.
    # Reopening through capsule_anchors would create a second unchecked path read.
    data = _read_bytes(row['path'], expected_bytes=row['bytes'], expected_sha256=row['sha256'])
    cap, file_hash, kind, _ = B._episode_bytes(data, compressed=False)
    _need(kind == 'RAW_ALL_ACTOR_BOUNDARIES_REVERIFIED', 'INDIVIDUAL_RAW_JSON_REQUIRED')
    _need(cap['episode_id'] == row['episode_id'], 'EPISODE_ID_MISMATCH')
    _need(file_hash == row['sha256'] == cap['raw']['sha256'], 'SOURCE_HASH_MISMATCH')
    anchors = B._capsule_anchors(cap, file_hash, kind, steps=[0], seats=seats)
    return [prepare_opening(a) for a in anchors]


def prepare_cohort(locked: dict, *, seat: int = 0) -> list[dict]:
    """All 20 sources, one preregistered seat; no filtering for compatible wins."""
    _need(type(seat) is int and seat in (0, 1), 'INVALID_SEAT')
    locked = validate_sources(locked)  # ALL source pins before the first replay.
    prepared = []
    for row in locked['episodes']:
        prepared.extend(_prepare_source(row, seats=[seat]))
    _cohort_context(prepared)
    return prepared


def _cohort_context(prepared: list[dict]) -> dict:
    _need(type(prepared) is list and len(prepared) == EPISODES, 'EXACT_COHORT_REQUIRED')
    # Existing cohort binding rejects duplicate episode AND raw-source identities.
    B.recorded_suffix_binding(prepared)
    context = B.proposal_context(prepared[0])
    memory = empty_continuation()
    for row in prepared:
        _need(B.proposal_context(row) == context, 'INCOMPATIBLE_OPENING_CONTEXTS')
        _need(row['continuation_binding']['memory'] == memory, 'EXPLICIT_EMPTY_CONTINUATION_REQUIRED')
    return context


def _write_json(path: str | Path, value: dict) -> dict:
    path = _safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical(value) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())


def preparation_receipt(selected: dict, *, output: str | Path) -> dict:
    """Exactly the one exposed real episode; never a reduced confirmation cohort."""
    selected = validate_selection(selected)
    matches = [r for r in selected['episodes'] if r['episode_id'] == EXPOSED_EPISODE]
    _need(len(matches) == 1 and matches[0]['split'] == 'discovery', 'EXPOSED_DISCOVERY_INPUT_REQUIRED')
    row = dict(matches[0], sha256=EXPOSED_SHA256)
    _known_source(row)
    output = _safe_path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'SELECTION.json', selected)
    _write_json(output / 'SOURCE.json', bound(dict(schema=SCHEMA + '-receipt-source',
        manifest=selected['manifest'], member=row, source_pins=source_pins(),
        scope='EXPOSED_DISCOVERY_PREPARATION_ONLY_NOT_A_COHORT')))
    prepared = _prepare_source(row, seats=[0, 1])
    artifact = _write_json(output / 'PREPARED.json', bound(dict(schema=SCHEMA + '-prepared-receipt',
        prepared=prepared, limits=LIMITS)))
    missing = [r['episode_id'] for r in selected['episodes'] if not Path(r['path']).is_file()]
    result = bound(dict(schema=SCHEMA + '-preparation-receipt', status='PREPARED_EXPOSED_DISCOVERY_ONLY',
        source=row, manifest=selected['manifest'], selection_sha256=selected['content_sha256'],
        prepared=artifact, counts=dict(sources_verified=1, recorded_transitions=719,
            original_actor_boundaries=1440, opening_anchors=len(prepared), proposal_ready=len(prepared),
            replay_passes=2, recorded_transition_executions=1438,
            model_loads=0, direct_draws=0, challengers=0, branch_games=0, admitted_pairs=0,
            optimizer_updates=0), missing_selected_episode_ids=missing,
        reason_codes=['NO_COMPLETE_PINNED_20_EPISODE_COHORT', 'NO_FIXED_PARENT_DRAW_OR_BRANCH_COMPARISON'],
        hard_stop='NO_FIT_NO_PROMOTION_CONTINUE_ORDINARY_FRESH_SELFPLAY_SEARCH',
        anchors=[dict(seat=p['anchor']['seat'], step=0, state=p['state'], teacher_weight=p['teacher_weight'],
            actor_context_sha256=p['actor_context_sha256'],
            anchor_sha256=p['anchor']['content_sha256']) for p in prepared],
        limits=LIMITS, source_pins=source_pins()))
    _write_json(output / 'RESULT.json', result)
    return result


def _repo_relative(path: str | Path) -> str:
    path = _safe_path(path)
    try:
        return str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
    except ValueError as exc:
        raise WarmstartError('REPOSITORY_RELATIVE_PATH_REQUIRED') from exc


def make_sampler(prepared: dict, *, parent: dict, draw_id: str, seed: int,
                 output: str | Path, device: str = 'cpu', production: bool = True):
    """Existing NativeSampler owns the one cold load, raw decoder and draw journal.

    The production CLI has no diagnostic runtime bypass;
    production=False is solely a Python CPU test seam. NativeSampler still requires
    its existing explicit execution authorization, including on diagnostic calls.
    """
    from farm_neat.rollout import v0_regret_native as N
    context = B.proposal_context(prepared)
    config = N.build_config(dict(schema=N.SPEC_SCHEMA, producer_id='public-warmstart-' + draw_id,
        parent=copy.deepcopy(parent), device=device, production=production,
        output=_repo_relative(output), timeout_seconds=30.0,
        collector_draws=[dict(draw_id=draw_id, seed=seed, actor_context_sha256=digest(context),
                             continuation=prepared['continuation_binding']['memory'])],
        direct_plan=None, worlds=[]))
    return N.NativeSampler(config)


def sample_learner(prepared: dict, *, sampler: Callable, draw_id: str, seed: int,
                   parent_checkpoint_sha256: str) -> dict:
    """Call once with the existing actor-only accessor; never invent likelihoods."""
    context = B.proposal_context(prepared)
    learner = sampler(context, draw_id=draw_id, seed=seed,
                      parent_checkpoint_sha256=parent_checkpoint_sha256)
    _need(learner['direct_draw_id'] == draw_id and
          learner['sampling_provenance']['checkpoint_sha256'] == parent_checkpoint_sha256,
          'DIRECT_DRAW_IDENTITY_MISMATCH')
    B.bind_request(prepared, learner, arm='learner')
    return copy.deepcopy(learner)


def freeze_challenger(prepared: dict, learner: dict, *, proposal_seed: int) -> tuple[dict | None, dict]:
    """One actor-only KEEP challenger; duplicate/refused proposals are not rerolled.

    max_candidates=2 is the existing generator's learner-control slot plus one
    challenger slot, not two challengers. A sampled STOP therefore has no challenger.
    """
    from farm_neat.rollout import v0_regret_collect as C
    B.bind_request(prepared, learner, arm='learner')
    proposals = B.propose(prepared, learner, budget=PROPOSAL_BUDGET, proposal_seed=proposal_seed)
    pool = proposals['pool']
    _need(pool['budget'] == PROPOSAL_BUDGET and pool['counts']['proposed'] <= 2,
          'ONE_CHALLENGER_BUDGET')
    candidates = [c for c in pool['candidates'] if c['proposal_family'] != 'learner']
    _need(len(candidates) <= 1, 'ONE_CHALLENGER_BUDGET')
    if not candidates:
        return None, proposals
    challenger = C._candidate_challenger(candidates[0], learner, B.proposal_context(prepared),
                                         pool['pool_sha256'], pool['budget'])
    B.bind_request(prepared, challenger, arm='challenger')
    return challenger, proposals


def freeze_pair(prepared: list[dict], learner: dict, challenger: dict, *,
                family_id: str, independence_attested: bool) -> dict:
    """Freeze ONE pair and all 4/16 worlds before observing either arm's return.

    This is a wire adapter to the existing collector's frozen registration and
    protocol builders, not a second statistical gate. Source deduplication alone
    cannot prove independence: the maintainer must explicitly attest it.
    """
    from farm_neat.rollout import v0_regret_collect as C
    from farm_neat.training import v0_regret_admission as AD
    _need(independence_attested is True, 'INDEPENDENCE_ATTESTATION_REQUIRED')
    R._text(family_id, '$.family_id')
    ctx = _cohort_context(prepared)
    for arm, path in (('learner', learner), ('challenger', challenger)):
        B.bind_request(prepared[0], path, arm=arm)
    suffix = B.recorded_suffix_binding(prepared)
    worlds = [B._world(p) for p in prepared]
    memory = prepared[0]['continuation_binding']['memory']
    seat = ctx['observation']['player']
    pins = R.source_pins()
    base = bound(dict(schema=R.SCHEMA,
        identity=dict(pair_id=family_id+'-opening', source_pins=pins,
            protocol_sha256=pins[R.DESIGN_PATH],
            parent_checkpoint_sha256=learner['sampling_provenance']['checkpoint_sha256'],
            actor_context_sha256=digest(ctx), continuation_sha256=digest(memory),
            source_group_id=worlds[0]['source_group_id'], split='training', round_id=family_id),
        actor_context=ctx, continuation=memory, learner=copy.deepcopy(learner), challenger=copy.deepcopy(challenger),
        evaluation=dict(mode='UNEVALUATED', discovery_world_groups=[], confirmation_world_groups=[], seat_rows=[],
            engine_hash=pins[R.ENGINE_PATH], opponent_hash=suffix['content_sha256'],
            continuation_hash=digest(memory), complete=False, replay_verified=False, attempt_ids=[],
            confirmation_registration=None),
        comparison=dict(objective='TERMINAL_WDL', mean_wdl_delta=None, lower_bound=None, alpha=None,
            independent_world_count=0, verdict='INDETERMINATE', actor_training_admissible=False,
            reason_codes=['UNEVALUATED'])))
    R.validate_pair(base)
    seat_spec = dict(seat=seat, actor_context=ctx, continuation=memory, draw_id=learner['direct_draw_id'])
    test = dict(test_id='public-opening-seat-'+str(seat), seats=[seat_spec],
        discovery_world_ids=[w['source_group_id'] for w in worlds[:DISCOVERY]],
        confirmation_world_ids=[w['source_group_id'] for w in worlds[DISCOVERY:]],
        source_families={w['source_group_id']:'public-source-'+p['anchor']['trainer']['source_sha256']
                         for w,p in zip(worlds, prepared)})
    plan = dict(spec=dict(family_alpha=.05, min_effect=0., tests=[test], family_id=family_id,
        round_id=family_id, independence_attested=True, compatible_contexts_attested=True),
        worlds={w['source_group_id']:w for w in worlds})
    # No outcomes enter _aggregate here. In particular, never relabel a callback
    # as NATIVE_GAME to bypass the generic collector's diagnostic-only guard.
    pair = C._aggregate(plan, test, base, {})
    protocol = C._protocol(plan, [(test, [dict(seat=seat_spec, record=base)])])
    AD._protocol(protocol)
    return bound(dict(schema=SCHEMA+'-frozen', pair=pair, protocol=protocol,
        recorded_suffix_binding=suffix, source_pins=source_pins(), limits=LIMITS))


def _execution_pair(frozen: dict, executions: list[dict]) -> dict:
    """Encode retained bridge outcomes in pair-v1; admission still replays them.

    A bridge's verified tape is its own evidence, not the generic collector's
    native callback type. Preserve runner_binding unchanged in the evidence
    archive. Neither this adapter nor its flags authorize teacher weight.
    """
    from farm_neat.training import v0_regret_admission as AD
    _sealed(frozen, SCHEMA+'-frozen')
    _need(frozen['source_pins'] == source_pins(), 'SOURCE_MISMATCH')
    AD._protocol(frozen['protocol'])
    pair = R.validate_pair(frozen['pair'])
    ev = pair['evaluation']
    _need(not ev['seat_rows'], 'OUTCOMES_WERE_NOT_FROZEN_EMPTY')
    groups = {w['source_group_id']:(phase,w) for phase in ('discovery','confirmation')
              for w in ev[phase+'_world_groups']}
    seen = set()
    for ex in executions:
        _sealed(ex, B.SCHEMA+'-execution')
        group = ex['source_group']['source_group_id']
        _need(group in groups and group not in seen, 'DUPLICATE_OR_UNREGISTERED_EXECUTION')
        seen.add(group)
        _need((ex['phase'],ex['source_group']) == groups[group], 'WORLD_REPLAY_MISMATCH')
        _need(ex['paths'] == {arm:pair[arm] for arm in ('learner','challenger')}, 'FROZEN_PATH_MISMATCH')
        # A failed/unknown launch does not get an invented counterpart attempt.
        attempts = {run['arm']:run['response']['request_sha256'] for run in ex['runs']
                    if run['phase'] == 'execute'}
        if set(attempts) != {'learner','challenger'}:
            continue
        complete = all(ex['outcomes'].get(arm,{}).get('complete') is True for arm in attempts)
        row = dict(source_group_id=group, seat=pair['actor_context']['observation']['player'],
            actor_context_sha256=pair['identity']['actor_context_sha256'], complete=complete,
            replay_verified=complete and ex['replay_verified'] is True)
        for arm in attempts:
            row[arm+'_attempt_id'] = attempts[arm]
            row[arm+'_wdl'] = ex['outcomes'][arm]['wdl'] if complete else None
        ev['seat_rows'].append(row)
        ev['attempt_ids'].extend(attempts.values())
    ev['complete'] = len(ev['seat_rows']) == len(groups) and all(r['complete'] for r in ev['seat_rows'])
    ev['replay_verified'] = ev['complete'] and all(r['replay_verified'] for r in ev['seat_rows'])
    confirmed = [r['challenger_wdl']-r['learner_wdl'] for r in ev['seat_rows']
                 if groups[r['source_group_id']][0] == 'confirmation' and r['complete']]
    reg = ev['confirmation_registration']
    mean = math.fsum(confirmed)/len(confirmed) if confirmed else None
    # Pair-v1 requires this descriptive fixed-N summary. The unchanged ordinary
    # admission routine independently recomputes the bound and the actual gate.
    lower = mean-math.sqrt(8*-math.log(reg['alpha_pair'])/len(confirmed)) if len(confirmed) == CONFIRMATION else None
    pair['comparison'].update(mean_wdl_delta=mean, lower_bound=lower, alpha=reg['alpha_pair'],
        independent_world_count=len(confirmed), reason_codes=['AWAITING_PUBLIC_BRIDGE_ADMISSION'])
    return R.validate_pair(bound(pair))


def export_corpus(frozen: dict, executions: list[dict], *, output: str | Path) -> dict:
    """Retain every outcome, call the unchanged public gate, then trainer preflight.

    The current trainer rejects CENSORED/INVALID/UNSUPPORTED rows outright. Those
    remain in EVIDENCE, and their registered lock remains in the corpus denominator
    as a missing (zero-weight) opportunity. Valid complete losers/ties stay in the
    corpus. A refusal *before* bridge admission never becomes a positive row by
    calling the statistical gate alone.
    """
    from farm_neat.training import v0_regret_train as RT
    output = _safe_path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output/'FROZEN.json', frozen)
    # Persist even evidence that the wire validator subsequently refuses.
    raw = _write_json(output/'EXECUTIONS.json', dict(executions=executions))
    pair = _execution_pair(frozen, executions)
    bridge = B.admit_public_pair(pair, protocol=frozen['protocol'], executions=executions)
    gate = bridge.get('admission')
    opportunities = []
    if gate is not None and gate['status'] not in ('INVALID','CENSORED','UNSUPPORTED'):
        _need(not gate['actor_training_admissible'] or bridge['state'] == 'ADMITTED_PAIR',
              'PUBLIC_BRIDGE_DID_NOT_ADMIT')
        opportunities.append(dict(pair=pair, admission=gate))
    evidence = _write_json(output/'EVIDENCE.json', bound(dict(schema=SCHEMA+'-evidence',
        frozen_sha256=frozen['content_sha256'], pair=pair, bridge=bridge, executions=executions,
        raw_executions=raw, limits=LIMITS)))
    corpus = dict(schema=RT.PAIRED_SCHEMA, protocol=frozen['protocol'], opportunities=opportunities)
    ref = _write_json(output/'PAIRED.json', corpus)  # deliberately NOT bound: closed trainer schema
    active, preflight = RT._paired_data(dict(paired_corpus=ref,
        reference_parent=dict(commit_sha256=pair['identity']['parent_checkpoint_sha256'])))
    result = bound(dict(schema=SCHEMA+'-corpus-result', status='PREFLIGHTED_NOT_TRAINED',
        corpus=ref, evidence=evidence, bridge=bridge, preflight=preflight,
        admitted_pairs=len(active), hard_stop=not active, optimizer_updates=0,
        next_action='BOUNDED_REGRET_THEN_FRESH_SELFPLAY' if active else 'ORDINARY_FRESH_SELFPLAY_SEARCH_ONLY',
        limits=LIMITS))
    _write_json(output/'RESULT.json', result)
    return result


def fresh_worlds(seed_base: int, *, public_seeds: list[int]) -> list[int]:
    """The existing selfplay CLI's fixed 2x4 train / 4 evaluation allocation.

    Check both world seeds and the owner's per-cohort private stream offsets.
    This only checks the supplied exposure set, not all historical user runs.
    """
    _need(type(seed_base) is int and 0 <= seed_base < 2**63-2020000, 'INVALID_FRESH_SEED')
    worlds = [seed_base+i for i in range(8)] + [seed_base+100+i for i in range(4)]
    streams = {seed_base+1000000+cohort*10000+j for cohort in range(2) for j in range(8)}
    streams.update(seed_base+2000000+j for j in range(8))
    _need(not (set(worlds) | streams) & set(public_seeds), 'FRESH_SEED_OVERLAP')
    return worlds


def maintainer_recipe(result: dict, *, parent: dict, fresh_seed_base: int,
                      public_seeds: list[int]) -> str:
    """Print existing bounded CLI commands; never load, fit or run them here."""
    fresh_worlds(fresh_seed_base, public_seeds=public_seeds)
    parent_path = _repo_relative(parent['path'])
    _hash(parent['commit_sha256'])
    active_count = 0
    if result.get('corpus') is not None:
        from farm_neat.training import v0_regret_train as RT
        active, preflight = RT._paired_data(dict(paired_corpus=result['corpus'], reference_parent=parent))
        _need(preflight == result['preflight'], 'PREFLIGHT_BINDING_MISMATCH')
        active_count = len(active)
    _need(result['admitted_pairs'] == active_count and result['hard_stop'] == (not active_count),
          'ZERO_ADMISSION_STOP_MISMATCH')
    q = shlex.quote
    text = [
        '#!/usr/bin/env bash', 'set -euo pipefail',
        '# Maintainer only: registered runtime/environment, authorized GPU allocation.',
        '# Off-policy public regret is NOT BC or a field-strength claim. No promotion.',
        '# Confirm these prospectively frozen seeds have no other historical exposure.',
        ': "${TRAIN_PY:=work/venv/python}"',
        ': "${SLURM_JOB_ID:?Run only in an authorized training allocation}"',
        'export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"',
        'RUN_ROOT="${1:?Supply a NEW work/runs/... output directory}"',
        'case "$RUN_ROOT" in work/runs/*) ;; *) echo "work/runs output required" >&2; exit 2;; esac',
        'hashfile() { "$TRAIN_PY" -c \'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())\' "$1"; }',
        'mkdir "$RUN_ROOT"',
        'NEXT_PARENT='+q(parent_path), 'NEXT_COMMIT='+q(parent['commit_sha256']),
    ]
    if active_count:
        ref = result['corpus']
        text += [
            '# Required: EXISTING same-parent direct plan/corpus with their previously pinned hashes.',
            ': "${DIRECT_PLAN:?Existing direct RL plan required}"',
            ': "${DIRECT_PLAN_SHA256:?Pinned direct-plan SHA256 required}"',
            ': "${DIRECT_CORPUS:?Existing genuine direct corpus required}"',
            ': "${DIRECT_CORPUS_SHA256:?Pinned direct-corpus SHA256 required}"',
            '"$TRAIN_PY" -B -m farm_neat.training.v0_regret_train plan '
            '--direct-plan "$DIRECT_PLAN" --direct-plan-sha256 "$DIRECT_PLAN_SHA256" '
            '--direct-corpus "$DIRECT_CORPUS" --direct-corpus-sha256 "$DIRECT_CORPUS_SHA256" '
            '--paired-corpus '+q(str(ref['path']))+' --paired-corpus-sha256 '+q(ref['sha256'])+' '
            '--mode direct-rl+gated-pairwise --lambda-teacher 0.1 --beta 0.1 '
            '--device cuda --updates 2 --output "$RUN_ROOT/warmstart-plan.json"',
            '"$TRAIN_PY" -B -m farm_neat.training.v0_regret_train train '
            '--plan "$RUN_ROOT/warmstart-plan.json" --plan-sha256 "$(hashfile "$RUN_ROOT/warmstart-plan.json")" '
            '--output "$RUN_ROOT/warmstart-fit"',
            'NEXT_PARENT=$("$TRAIN_PY" -c \'import json,sys,hashlib; from pathlib import Path; '
            'r=json.load(open(sys.argv[1])); assert r["status"]=="COMMITTED_NOT_STRENGTH" and r["optimizer_updates"]>0; '
            'p=Path(r["last_checkpoint"]); assert hashlib.sha256((p/"COMMIT.json").read_bytes()).hexdigest()==r["commit_sha256"]; '
            'print(p)\' "$RUN_ROOT/warmstart-fit/RESULT.json")',
            'NEXT_COMMIT=$(hashfile "$NEXT_PARENT/COMMIT.json")',
        ]
    else:
        text += ['# ZERO_ADMITTED_PAIRS: NO_FIT_NO_PROMOTION. Continue ordinary fresh self-play/search from the original parent.']
    text += [
        '"$TRAIN_PY" -c \'import sys; from farm_neat.evaluation.v0_policy import inspect_committed_scorer; '
        'inspect_committed_scorer(sys.argv[1],sys.argv[2],device="cuda",production=True)\' "$NEXT_PARENT" "$NEXT_COMMIT"',
        '"$TRAIN_PY" -B -m farm_neat.training.v0_selfplay --prepare --parent "$NEXT_PARENT" '
        '--seed-base '+str(fresh_seed_base)+' --cohorts 2 --games-per-cohort 4 --evaluation-games 4 '
        '--output "$RUN_ROOT/fresh-plan.json"',
        '"$TRAIN_PY" -B -m farm_neat.training.v0_selfplay --plan "$RUN_ROOT/fresh-plan.json" '
        '--plan-sha256 "$(hashfile "$RUN_ROOT/fresh-plan.json")" --output "$RUN_ROOT/fresh-selfplay"',
    ]
    return '\n'.join(text)+'\n'


def _recorded_branch(request: dict) -> dict:
    """Small runner adapter; the existing bridge restores and verifies BOTH arms.

    Native execution is reachable only through the explicitly authorized CLI.
    The bridge owns compiler/opponent construction, deadlines and independent
    replay/admission. No callback-kind attestation is changed here.
    """
    env, actor, rival = B.restore_branch(request)
    seat = request['initial']['seat']
    tape = []
    while not env.done and len(tape) < 719-request['initial']['step']:
        cfg = B.actor_configuration(env)
        joint = [None, None]
        joint[seat] = actor(B.actor_observation(env, seat), cfg)
        joint[1-seat] = rival(B.actor_observation(env, 1-seat), cfg)
        tape.append(copy.deepcopy(joint))
        env.step(joint)
    return bound(dict(schema=B.SCHEMA+'-branch-response', request_sha256=request['content_sha256'],
        initial=request['initial'], status='COMPLETE' if env.done else 'CENSORED', joint_actions=tape))


def run_warmstart(locked: dict, *, parent: dict, output: str | Path, draw_seed: int,
                  proposal_seed: int, fresh_seed_base: int, independence_attested: bool,
                  execute: bool = False, seat: int = 0, device: str = 'cpu') -> dict:
    """Maintainer execution: one journaled parent draw, one challenger, one cohort.

    No fit occurs here. An interrupted output is never silently resumed, retried,
    or replaced. Missing/censored calls remain evidence, never new opportunities.
    """
    from farm_neat.rollout import v0_regret_native as N
    _need(execute is True and os.environ.get(N.AUTH_ENV) == '1', 'EXPLICIT_EXECUTION_AUTHORIZATION_REQUIRED')
    _need(independence_attested is True, 'INDEPENDENCE_ATTESTATION_REQUIRED')
    _need(type(draw_seed) is int and type(proposal_seed) is int and
          0 <= draw_seed < 2**63 and 0 <= proposal_seed < 2**63 and draw_seed != proposal_seed,
          'DISTINCT_FROZEN_DRAW_SEEDS_REQUIRED')
    _need(device in ('cpu','cuda'), 'INVALID_DEVICE')
    _need(set(parent) == {'path','commit_sha256'}, 'PARENT_FIELDS')
    parent = dict(path=_repo_relative(parent['path']), commit_sha256=_hash(parent['commit_sha256']))
    locked = validate_sources(locked)  # all 20 files bound before replay/output/model load
    output = _safe_path(output)
    _repo_relative(output)
    output.mkdir(parents=True, exist_ok=False)
    draw_id = 'opening-'+locked['content_sha256'][:16]+'-seat-'+str(seat)
    _write_json(output/'INTENT.json', bound(dict(schema=SCHEMA+'-intent', sources=locked,
        parent=parent, seat=seat, draw_id=draw_id, draw_seed=draw_seed, proposal_seed=proposal_seed,
        fresh_seed_base=fresh_seed_base, independence_attested=True, proposal_budget=PROPOSAL_BUDGET,
        branch_timeout_seconds=30., source_pins=source_pins(), limits=LIMITS)))
    launched = 0
    try:
        from farm_neat.evaluation.v0_policy import inspect_committed_scorer
        # Metadata/runtime preflight before costly public replay; NativeSampler
        # remains the sole cold tensor loader and authentic direct-draw owner.
        inspect_committed_scorer(ROOT/parent['path'],parent['commit_sha256'],device=device,production=True)
        prepared = prepare_cohort(locked, seat=seat)
        _write_json(output/'PREPARED.json', dict(prepared=prepared))
        public_seeds = [p['anchor']['trainer']['seed'] for p in prepared]+[draw_seed,proposal_seed]
        fresh_worlds(fresh_seed_base, public_seeds=public_seeds)
        sampler = make_sampler(prepared[0],parent=parent,draw_id=draw_id,seed=draw_seed,
                               output=output/'native-draw',device=device,production=True)
        learner = sample_learner(prepared[0],sampler=sampler,draw_id=draw_id,seed=draw_seed,
                                 parent_checkpoint_sha256=parent['commit_sha256'])
        _write_json(output/'LEARNER.json',learner)
        challenger, proposals = freeze_challenger(prepared[0],learner,proposal_seed=proposal_seed)
        _write_json(output/'PROPOSALS.json',proposals)
        if challenger is None:
            result = dict(schema=SCHEMA+'-corpus-result',status='NO_REACHABLE_DISTINCT_CHALLENGER',
                admitted_pairs=0,hard_stop=True,optimizer_updates=0,corpus=None,preflight=None,
                next_action='ORDINARY_FRESH_SELFPLAY_SEARCH_ONLY',limits=LIMITS)
        else:
            frozen = freeze_pair(prepared,learner,challenger,family_id=draw_id,
                                 independence_attested=independence_attested)
            _write_json(output/'FROZEN.json',frozen)  # durable before ANY branch outcome
            def runner(request):
                nonlocal launched
                branch = output/'branches'/request['content_sha256']
                branch.mkdir(parents=True,exist_ok=False)
                _write_json(branch/'INTENT.json',request)
                launched += 1
                response = _recorded_branch(request)
                _write_json(branch/'RESPONSE.json',response)
                return response
            executions = []
            for index,p in enumerate(prepared):
                ex = B.compare(p,frozen['pair']['learner'],frozen['pair']['challenger'],runner=runner,
                    phase='discovery' if index < DISCOVERY else 'confirmation',timeout_seconds=30.)
                _write_json(output/'comparisons'/f'{index+1:02d}.json',ex)
                executions.append(ex)  # retain losses/ties/refusals; never substitute or reroll
            result = export_corpus(frozen,executions,output=output/'corpus')
        commands = maintainer_recipe(result,parent=parent,fresh_seed_base=fresh_seed_base,public_seeds=public_seeds)
        with (output/'CONTINUE.sh').open('x') as stream:
            stream.write(commands)
        os.chmod(output/'CONTINUE.sh',0o755)
        result = bound(dict(result,commands=commands,branch_attempts_started=launched,
                            fresh_worlds=fresh_worlds(fresh_seed_base,public_seeds=public_seeds)))
        _write_json(output/'RESULT.json',result)
        return result
    except Exception as exc:
        _write_json(output/'FAILURE.json',bound(dict(schema=SCHEMA+'-failure',status='REFUSED_NOT_RETRIED',
            code=getattr(exc,'code',type(exc).__name__),error=str(exc),branch_attempts_started=launched,
            optimizer_updates=0,hard_stop=True,limits=LIMITS)))
        raise


def _read_json_ref(path, sha256):
    return R._loads(_read_bytes(path,expected_sha256=sha256).decode('utf-8'))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('select','receipt'):
        p = sub.add_parser(name)
        p.add_argument('--manifest',default='work/datasets/public-episodes/daily/manifest.csv')
        p.add_argument('--manifest-sha256',default=MANIFEST_SHA256)
        p.add_argument('--episode-dir',default='work/datasets/public-episodes/daily')
        p.add_argument('--output',required=True)
    p = sub.add_parser('pin')
    p.add_argument('--selection',required=True); p.add_argument('--selection-sha256',required=True)
    p.add_argument('--acquisition', help='already-acquired source ledger; supplied daily ledger is automatic')
    p.add_argument('--acquisition-sha256', help='previously pinned acquisition-ledger file SHA256')
    p.add_argument('--output',required=True)
    p = sub.add_parser('run',help='maintainer-only; actual replay/branches, never fits')
    p.add_argument('--sources',required=True); p.add_argument('--sources-sha256',required=True)
    p.add_argument('--parent',required=True); p.add_argument('--parent-sha256',required=True)
    p.add_argument('--output',required=True); p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--seat',type=int,choices=[0,1],default=0)
    for name in ('draw-seed','proposal-seed','fresh-seed-base'):
        p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--attest-independence',action='store_true',
        help='attest distinct source independence and no undisclosed confirmation exposure')
    p.add_argument('--execute',action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command in ('select','receipt'):
            selection = select_manifest(args.manifest,expected_sha256=args.manifest_sha256,episode_dir=args.episode_dir)
            result = (_write_json(args.output,selection) if args.command == 'select' else
                      preparation_receipt(selection,output=args.output))
        elif args.command == 'pin':
            _need(bool(args.acquisition) == bool(args.acquisition_sha256), 'ACQUISITION_REFERENCE_REQUIRED')
            acquisition = (dict(path=args.acquisition, sha256=args.acquisition_sha256)
                           if args.acquisition is not None else None)
            result = _write_json(args.output, pin_sources(
                _read_json_ref(args.selection,args.selection_sha256), acquisition=acquisition))
        else:
            # Reject before even opening a source lock on an unauthorized command.
            from farm_neat.rollout import v0_regret_native as N
            _need(args.execute and os.environ.get(N.AUTH_ENV) == '1','EXPLICIT_EXECUTION_AUTHORIZATION_REQUIRED')
            result = run_warmstart(_read_json_ref(args.sources,args.sources_sha256),
                parent=dict(path=args.parent,commit_sha256=args.parent_sha256),output=args.output,
                draw_seed=args.draw_seed,proposal_seed=args.proposal_seed,fresh_seed_base=args.fresh_seed_base,
                independence_attested=args.attest_independence,execute=args.execute,seat=args.seat,device=args.device)
            print(result['commands'],end='')
        print(canonical(result))
        return 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
        print(canonical(dict(status='REFUSED',code=getattr(exc,'code',type(exc).__name__),
            error=str(exc),details=getattr(exc,'details',None),optimizer_updates=0)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
