"""Source-bound all-later-dawn V0 sequence signal; not an opening-v1 corpus.

The experiment is the single accepted preregistration, not a tunable general
trainer. Native-free diagnostic interfaces are separate from production gates.
World IDs, outcomes, checkpoint/source provenance and admission weights are
trainer-only. No function here changes the game, compiler, action vocabulary,
shared trainer, or a checkpoint's historical identity.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DESIGN = 'work/runs/reviewer-play/v0-regret-next-signal'
PREREG_PATH = DESIGN + '/PREREGISTRATION.json'
ACCEPTANCE = 'work/runs/review-package/acceptance/V0-REGRET-NEXT-SIGNAL.json'
PREREG_SHA256 = '27523d61dc660ff262f41f09db4971d6ce17a8100b7ab941db0be087cac68c17'
SCHEMA = 'farm-v0-regret-sequence-v1'
DAWNS = tuple(range(0, 697, 24))
SPLITS = ('discovery', 'confirmation', 'monitor')
KEEP_LAW = 'PARENT_OPENING_THEN_KEEP_ALL_LATER_DAWNS'
SCOPES = ('DIAGNOSTIC_CPU', 'PRODUCTION_NOT_STRENGTH')


class SequenceError(ValueError):
    """Fail-closed, machine-readable refusal; never an inferred WDL outcome."""


def need(ok: bool, code: str) -> None:
    if not ok:
        raise SequenceError(code)


def wire(value: Any) -> str:
    # Preserve mapping order: executor inventory/continuation order is behavior.
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':'))


def digest(value: Any) -> str:
    return hashlib.sha256(wire(value).encode()).hexdigest()


def file_sha(path: str | Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def seal(value: dict) -> dict:
    body = {k: copy.deepcopy(v) for k, v in value.items() if k != 'content_sha256'}
    return {**body, 'content_sha256': digest(body)}


def fields(value: Any, expected, code: str) -> None:
    need(type(value) is dict and set(value) == set(expected), code)


def hashed(value: dict, expected, schema: str, code: str) -> None:
    fields(value, set(expected) | {'schema', 'content_sha256'}, code + '_FIELDS')
    need(value['schema'] == schema and value == seal(value), code + '_HASH_OR_SCHEMA')


def safe_path(path: str | Path) -> Path:
    path = Path(path).absolute()
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'SYMLINKED_PATH')
    return path


def write_new(path: str | Path, value: dict) -> dict:
    """Publish immutable JSON with the existing exclusive/fsynced convention."""
    from farm_neat.evaluation.v0_policy import write_new as write
    return write(safe_path(path), value)


def read_json(path: str | Path) -> dict:
    def unique(pairs):
        result = {}
        for k, v in pairs:
            need(k not in result, 'DUPLICATE_JSON_KEY')
            result[k] = v
        return result
    with safe_path(path).open() as stream:
        result = json.load(stream, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(SequenceError('NONFINITE_JSON')))
    need(type(result) is dict, 'JSON_OBJECT_REQUIRED')
    wire(result)
    return result


def approved_preregistration() -> dict:
    """Use accepted WAVE78 evidence, never rerun its historical registry join."""
    receipt = read_json(ROOT / ACCEPTANCE)
    need(receipt.get('decision') == 'ACCEPT' and receipt.get('outcome') == 'PASS'
         and receipt.get('task_id') == 'V0-REGRET-NEXT-SIGNAL', 'DESIGN_NOT_ACCEPTED')
    need(file_sha(ROOT / PREREG_PATH) == PREREG_SHA256
         and receipt.get('source_pins', {}).get(PREREG_PATH) == PREREG_SHA256,
         'PREREGISTRATION_SOURCE_MISMATCH')
    return read_json(ROOT / PREREG_PATH)


def source_pins() -> dict[str, str]:
    """Current adapter runtime closure; historical SOURCES.json is not rebound."""
    paths = set()
    for folder in ('economy', 'game', 'evaluation', 'policy', 'training'):
        base = ROOT / 'src/farm_neat' / folder
        paths.update(base.rglob('*.py'))
        paths.update(base.rglob('*.json'))
    paths.update(ROOT / p for p in (
        PREREG_PATH, ACCEPTANCE, DESIGN + '/BUILD-SPEC.md', DESIGN + '/SOURCES.json',
        'project/runtime/interpreter.toml', 'project/configs/scenarios.json',
        'src/farm_neat/tooling/runtime.py', 'RUN.py', 'project/PROJECT-STORAGE.json'))
    return {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(paths) if p.is_file()}


def _exposure(exposure: dict, prereg: dict, scope: str) -> None:
    fields(exposure, ('schema', 'complete', 'independent_original_generator', 'scope',
                      'ledger_sources', 'exposed', 'protected', 'joins'), 'EXPOSURE_FIELDS')
    need(exposure['schema'] == 'farm-v0-sequence-exposure-v1' and exposure['complete'] is True
         and exposure['independent_original_generator'] is True and exposure['scope'] == scope,
         'EXPOSURE_INCOMPLETE_OR_DISTRIBUTION_UNVERIFIED')
    pins = exposure['ledger_sources']
    need(type(pins) is dict and bool(pins), 'EXPOSURE_LEDGER_SOURCES_MISSING')
    for path, sha in pins.items():
        need(type(path) is str and type(sha) is str and len(sha) == 64
             and all(c in '0123456789abcdef' for c in sha), 'EXPOSURE_SOURCE_IDENTITY')
        if scope != 'DIAGNOSTIC_CPU':
            p = safe_path(ROOT / path)
            need(p.is_relative_to(ROOT) and p.is_file() and file_sha(p) == sha,
                 'EXPOSURE_SOURCE_MISSING_OR_STALE')
    expected = [(phase, w['family_id'], w['seed'])
                for phase in SPLITS for w in prereg['worlds'][phase]]
    joins = exposure['joins']
    need(type(joins) is list and len(joins) == 18, 'EXPOSURE_MISSING_WORLD')
    for row in joins:
        fields(row, ('phase', 'family_id', 'seed', 'source_family_id', 'world_sha256'), 'EXPOSURE_JOIN_FIELDS')
        need(type(row['seed']) is int and 0 <= row['seed'] < 2**63
             and type(row['source_family_id']) is str and bool(row['source_family_id'])
             and type(row['world_sha256']) is str and len(row['world_sha256']) == 64
             and all(c in '0123456789abcdef' for c in row['world_sha256']), 'EXPOSURE_JOIN_IDENTITY')
    need([(r['phase'], r['family_id'], r['seed']) for r in joins] == expected,
         'EXPOSURE_WORLD_TABLE_MISMATCH')
    for name in ('seed', 'family_id', 'source_family_id', 'world_sha256'):
        known = {r[name] for r in joins}
        need(len(known) == 18, 'EXPOSURE_SPLIT_OR_SOURCE_FAMILY_OVERLAP')
        for group in ('exposed', 'protected'):
            need(type(exposure[group]) is list, 'EXPOSURE_LEDGER_ROWS')
            for old in exposure[group]:
                fields(old, ('phase', 'family_id', 'seed', 'source_family_id', 'world_sha256'),
                       'EXPOSURE_LEDGER_ROW_FIELDS')
                need(old[name] not in known, 'EXPOSURE_PROTECTED_OR_PREVIOUSLY_EXPOSED')


def _launches(prereg: dict) -> list[dict]:
    rows = []
    for phase in SPLITS:
        for world in prereg['worlds'][phase]:
            for opponent in (('pass', 'router') if phase == 'monitor' else ('pass',)):
                for seat in (0, 1):
                    for arm in ('control', 'challenger'):
                        primary = len(rows)
                        for replica in ('primary', 'replay'):
                            body = dict(index=len(rows), phase=phase,
                                family_id=world['family_id'], seed=world['seed'], seat=seat,
                                opponent=opponent, arm=arm, replica=replica, primary_index=primary,
                                law=KEEP_LAW if phase != 'monitor' and arm == 'challenger' else 'UNRESTRICTED',
                                replay=replica == 'replay')
                            rows.append({**body, 'launch_id': digest(body)})
    return rows


def register_sequence(preregistration: dict, *, exposure: dict, diagnostic: bool = False) -> dict:
    """Freeze exactly the accepted 2/8/8 worlds, one proposal and 208 starts.

    Exposure completeness/independence is an auditor attestation, not a theorem
    inferred from unequal seeds. The production preflight authenticates its files.
    """
    need(type(diagnostic) is bool, 'EXPLICIT_DIAGNOSTIC_SCOPE')
    approved = approved_preregistration()
    need(preregistration == approved, 'PREREGISTRATION_CHANGED')
    scope = SCOPES[0 if diagnostic else 1]
    _exposure(exposure, approved, scope)
    launches = _launches(approved)
    counts = dict(
        discovery=sum(row['phase'] == 'discovery' for row in launches),
        confirmation=sum(row['phase'] == 'confirmation' for row in launches),
        monitor=sum(row['phase'] == 'monitor' for row in launches),
        primary=sum(not row['replay'] for row in launches),
        replay=sum(row['replay'] for row in launches),
        maximum_physical_starts=len(launches),
    )
    need(counts == {'discovery': 16, 'confirmation': 64, 'monitor': 128,
                    'primary': 104, 'replay': 104, 'maximum_physical_starts': 208},
         'REGISTRATION_START_ACCOUNTING')
    return seal(dict(schema=SCHEMA + '-registration', scope=scope,
        preregistration_sha256=PREREG_SHA256, preregistration=approved,
        acceptance_sha256=file_sha(ROOT / ACCEPTANCE), source_pins=source_pins(),
        exposure=copy.deepcopy(exposure), launches=launches,
        source_identity=copy.deepcopy(approved['source_identity']),
        parent=copy.deepcopy(approved['parent']),
        worlds=copy.deepcopy(approved['worlds']),
        proposal=copy.deepcopy(approved['proposal']),
        policy_law=dict(control='UNRESTRICTED_PARENT_SCORER', challenger=KEEP_LAW,
                        mode='greedy_class', labor_factor=4.0, hands_mode='daily',
                        flow_belief='ZERO'),
        counts=counts, scientific_execution='NOT_RUN'))


def validate_registration(registration: dict) -> dict:
    hashed(registration, ('scope', 'preregistration_sha256', 'preregistration',
                         'acceptance_sha256', 'source_pins', 'exposure', 'launches',
                         'source_identity', 'parent', 'worlds', 'proposal', 'policy_law',
                         'counts', 'scientific_execution'),
           SCHEMA + '-registration', 'REGISTRATION')
    need(registration['scope'] in SCOPES, 'REGISTRATION_SCOPE')
    approved = approved_preregistration()
    need(registration['preregistration'] == approved
         and registration['preregistration_sha256'] == PREREG_SHA256, 'PREREGISTRATION_CHANGED')
    need(registration['acceptance_sha256'] == file_sha(ROOT / ACCEPTANCE)
         and registration['source_pins'] == source_pins(), 'SOURCE_DRIFT')
    _exposure(registration['exposure'], approved, registration['scope'])
    need(registration['launches'] == _launches(approved), 'REGISTRATION_LAUNCH_MANIFEST_CHANGED')
    need(registration['source_identity'] == approved['source_identity']
         and registration['parent'] == approved['parent']
         and registration['worlds'] == approved['worlds']
         and registration['proposal'] == approved['proposal']
         and registration['scientific_execution'] == 'NOT_RUN', 'REGISTRATION_DESIGN_CHANGED')
    need(registration['counts'] == {
        'discovery': 16, 'confirmation': 64, 'monitor': 128,
        'primary': 104, 'replay': 104, 'maximum_physical_starts': 208,
    }, 'REGISTRATION_COUNTS_CHANGED')
    need(registration['policy_law'] == dict(
        control='UNRESTRICTED_PARENT_SCORER', challenger=KEEP_LAW,
        mode='greedy_class', labor_factor=4.0, hands_mode='daily', flow_belief='ZERO'),
        'REGISTRATION_POLICY_LAW_CHANGED')
    return registration


class SequenceJournal:
    """Append-only, hash-chained launch reservation and result journal.

    Every attempted launch is charged before calling the runner. Interrupted or
    failed launches cannot be retried automatically. A partial line is refused,
    never silently truncated. Advisory file locks serialize cooperating writers.
    """
    def __init__(self, path: str | Path, registration: dict):
        validate_registration(registration)
        self.path = safe_path(path)
        self.registration = copy.deepcopy(registration)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            self._read(stream)

    def _read(self, stream) -> list[dict]:
        stream.seek(0)
        text = stream.read()
        need(not text or text.endswith('\n'), 'JOURNAL_TRUNCATED')
        rows, pending, started = [], None, []
        for line in text.splitlines():
            try:
                row = json.loads(line)
                hashed(row, ('sequence', 'previous_sha256', 'registration_sha256', 'launch_index',
                             'status', 'result', 'error'), SCHEMA + '-journal', 'JOURNAL')
            except (ValueError, TypeError) as exc:
                raise SequenceError('JOURNAL_CORRUPT') from exc
            need(row['sequence'] == len(rows)
                 and row['previous_sha256'] == (rows[-1]['content_sha256'] if rows else None)
                 and row['registration_sha256'] == self.registration['content_sha256'], 'JOURNAL_CHAIN')
            index = row['launch_index']
            need(type(index) is int and 0 <= index < 208, 'JOURNAL_LAUNCH_INDEX')
            if row['status'] == 'STARTED':
                need(pending is None and index == len(started) and row['result'] is None
                     and row['error'] is None, 'JOURNAL_ORDER_OR_RETRY')
                started.append(index); pending = index
            else:
                need(row['status'] in ('FINISHED', 'FAILED') and pending == index,
                     'JOURNAL_RESULT_WITHOUT_START')
                need((row['status'] == 'FINISHED' and type(row['result']) is dict and row['error'] is None)
                     or (row['status'] == 'FAILED' and row['result'] is None and type(row['error']) is str
                         and bool(row['error'])), 'JOURNAL_RESULT_FIELDS')
                pending = None
            rows.append(row)
        return rows

    @property
    def events(self) -> list[dict]:
        with self.path.open('r') as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            return self._read(stream)

    @property
    def starts(self) -> int:
        return sum(r['status'] == 'STARTED' for r in self.events)

    def _append(self, job: dict, *, result=None, error=None, reserve=False) -> None:
        need(type(job) is dict and type(job.get('index')) is int
             and 0 <= job['index'] < 208
             and job == self.registration['launches'][job['index']], 'JOURNAL_FOREIGN_LAUNCH')
        with self.path.open('a+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            rows = self._read(stream)
            starts = [r for r in rows if r['status'] == 'STARTED']
            if reserve:
                need(not any(r['status'] == 'FAILED' for r in rows)
                     and (not rows or rows[-1]['status'] == 'FINISHED')
                     and job['index'] == len(starts), 'JOURNAL_RETRY_OR_SKIPPED_LAUNCH_REFUSED')
                need(len(starts) < 208, 'JOURNAL_START_BUDGET')
                status = 'STARTED'
            else:
                need(rows and rows[-1]['status'] == 'STARTED'
                     and rows[-1]['launch_index'] == job['index'], 'JOURNAL_NO_PENDING_START')
                need((result is not None) != (error is not None), 'JOURNAL_EXACTLY_ONE_RESULT')
                need((type(result) is dict and error is None)
                     or (result is None and type(error) is str and bool(error)),
                     'JOURNAL_RESULT_FIELDS')
                status = 'FAILED' if error is not None else 'FINISHED'
            row = seal(dict(schema=SCHEMA + '-journal', sequence=len(rows),
                previous_sha256=rows[-1]['content_sha256'] if rows else None,
                registration_sha256=self.registration['content_sha256'], launch_index=job['index'],
                status=status, result=result, error=error))
            stream.seek(0, os.SEEK_END)
            stream.write(wire(row) + '\n'); stream.flush(); os.fsync(stream.fileno())
        # Durably retain creation of the journal too.
        fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)

    def reserve(self, job: dict) -> None:
        self._append(job, reserve=True)

    def finish(self, job: dict, *, result: dict | None = None, error: str | None = None) -> None:
        self._append(job, result=result, error=error)


def policy_pair_bound(world_deltas: list[float]) -> dict:
    """One predeclared range-four policy-pair bound; seats/replays are not n."""
    need(type(world_deltas) is list and len(world_deltas) == 8
         and all(type(x) in (int, float) and math.isfinite(x) and -2 <= x <= 2 for x in world_deltas),
         'BOUND_REQUIRES_EIGHT_COMPLETE_WORLD_DELTAS')
    mean = math.fsum(world_deltas) / 8
    lower = mean - math.sqrt(8 * math.log(20) / 8)
    return dict(n=8, M=1, alpha=.05, mean=mean, lower_bound=lower,
                gate=lower > 0, weight=min(2., max(0., lower)) / 2)


ACTOR_FIELDS = ('player', 'step', 'cash', 'quadrants', 'hands_today', 'hires_today',
                'stock', 'seeds', 'inventory', 'params', 'shops', 'occupied',
                'quadrant_names', 'land_commitment')


def public_state(facts: dict):
    """Reconstruct only the existing scorer's public State allowlist.

    This is a versioned sequence context, NOT an opening-v1 actor validator.
    No raw observation, full controller snapshot, source or future schedule is
    ever handed to model.score. The native State methods remain authoritative.
    """
    from farm_neat.economy.v0 import vocabulary as V
    from farm_neat.economy.v0.projection import State
    from farm_neat.game.constants import SHOPS
    from farm_neat.policy import v0_scorer as A
    fields(facts, ACTOR_FIELDS, 'ACTOR_FIELDS')
    for key, low, high in (('player', 0, 1), ('step', 0, 696), ('quadrants', 1, 4),
                           ('hands_today', 0, V.HAND_CAP), ('hires_today', 0, V.HAND_CAP),
                           ('occupied', 0, 100)):
        need(type(facts[key]) is int and low <= facts[key] <= high, 'ACTOR_RANGE_' + key)
    need(facts['step'] in DAWNS, 'ACTOR_DAWN_REQUIRED')
    need(type(facts['cash']) in (int, float) and math.isfinite(facts['cash']) and facts['cash'] >= 0,
         'ACTOR_CASH')
    for key, names in (('stock', V.PRODUCTS), ('inventory', V.PRODUCTS), ('seeds', A.CROPS)):
        fields(facts[key], names, 'ACTOR_' + key)
        need(all(type(n) is int and n >= 0 for n in facts[key].values()), 'ACTOR_COUNT_' + key)
    params = facts['params']
    if params is not None:
        fields(params, V.PRODUCTS, 'ACTOR_MARKET_FIELDS')
        for row in params.values():
            fields(row, ('base', 'I0', 'T', 'below_func', 'above_func', 'below_target', 'above_target'),
                   'ACTOR_MARKET_PARAMETER_FIELDS')
            for key in ('base', 'I0', 'T', 'below_target', 'above_target'):
                need(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0,
                     'ACTOR_MARKET_PARAMETER_RANGE')
            need(row['T'] > 0 and row['below_func'] in A.SHAPES and row['above_func'] in A.SHAPES,
                 'ACTOR_MARKET_SHAPE')
    need(type(facts['shops']) is list and all(type(s) is str and s in SHOPS for s in facts['shops'])
         and len(facts['shops']) <= 8, 'ACTOR_SHOPS')
    state = State.__new__(State)
    for name in ACTOR_FIELDS:
        setattr(state, name, copy.deepcopy(facts[name]))
    state.day, state.hour = divmod(state.step, 24)
    need(facts['quadrant_names'] == list(V.QUADRANTS[:state.quadrants]),
         'ACTOR_ORDERED_QUADRANTS')
    state.quadrant_names = tuple(facts['quadrant_names'])
    if state.land_commitment is not None:
        from farm_neat.economy.v0.projection import LandCommitment
        LandCommitment.from_payload(state.land_commitment).validate_observation(state)
    return state


def public_facts(state) -> dict:
    """Explicit field selection, never vars(state) or a private observation."""
    from farm_neat.policy import v0_scorer as A
    value = {k: copy.deepcopy(getattr(state, k)) for k in ACTOR_FIELDS}
    value['quadrant_names'] = list(state.quadrant_names)
    value['seeds'] = {c: state.seeds.get(c, 0) for c in A.CROPS}
    public_state(value)
    return value


def _options_wire(plan, options: dict) -> dict:
    from farm_neat.economy.v0 import vocabulary as V
    from farm_neat.policy import v0_scorer as A
    need(type(options) is dict and not set(options) - {'labor_factor', 'belief', 'hold_releases'},
         'SEQUENCE_OPTIONS_FIELDS')
    normalized = A._projection_options(options)
    need(normalized['labor_factor'] == 4., 'SEQUENCE_COMPILER_FACTOR')
    holds = normalized['hold_releases']
    expected = {p for p in range(len(V.PRODUCTS)) if V.PRIMITIVES[plan.overlay(p)[0]] == 'HOLD'}
    need(set(holds) == expected, 'SEQUENCE_MISSING_OR_FOREIGN_HOLD_CUSTODY')
    for product, (overlay, release) in holds.items():
        need(tuple(overlay) == tuple(plan.overlay(product)[:2]) and type(release) is int
             and 0 <= release <= 718, 'SEQUENCE_STALE_HOLD_CUSTODY')
    return dict(labor_factor=4., belief=normalized['belief'],
                hold_releases=[[p, list(overlay), release] for p, (overlay, release) in holds.items()])


def _options_live(plan, value: dict) -> dict:
    fields(value, ('labor_factor', 'belief', 'hold_releases'), 'SEQUENCE_OPTIONS_FIELDS')
    need(type(value['hold_releases']) is list, 'SEQUENCE_HOLD_ROWS')
    holds = {}
    for row in value['hold_releases']:
        need(type(row) is list and len(row) == 3 and type(row[0]) is int
             and row[0] not in holds and type(row[1]) is list and len(row[1]) == 2,
             'SEQUENCE_HOLD_ROWS')
        holds[row[0]] = (tuple(row[1]), row[2])
    options = dict(labor_factor=value['labor_factor'], belief=value['belief'], hold_releases=holds)
    need(_options_wire(plan, options) == value, 'SEQUENCE_NONCANONICAL_OPTIONS')
    return options


def _geometry(state, plan, actions: list[int], options: dict, *, backend='fast'):
    from farm_neat.economy.v0 import vocabulary as V
    from farm_neat.policy import v0_scorer as A, v0_menu_fast as F
    need(backend in ('fast', 'reference'), 'SEQUENCE_MENU_BACKEND')
    need(type(actions) is list and 1 <= len(actions) <= 17
         and all(type(a) is int and 0 <= a < V.N_ACTIONS for a in actions)
         and actions[-1] == 0 and 0 not in actions[:-1], 'SEQUENCE_STOP_OR_CAP')
    prefix = A.Prefix(plan)
    rows, menus = [], []
    for action in actions:
        menu = F.build_menu(state, prefix, projection_kwargs=options, backend=backend)
        need(action in menu.legal_ids, 'SEQUENCE_UNREACHABLE_PREFIX_ACTION')
        index = menu.legal_ids.index(action)
        cls = next(c for c in menu.classes if index in c.indices)
        forced = prefix.edit_count == V.MAX_EDITS_PER_EPOCH
        rows.append(dict(prefix=prefix.payload(), support=list(menu.legal_ids), phi_sha256=digest(menu.phi),
                         raw_id=action, class_id=cls.class_id,
                         class_members=[menu.legal_ids[i] for i in cls.indices],
                         continuation=cls.successor.payload(), forced_stop=forced,
                         termination=('CAP' if forced else 'STOP') if action == 0 else 'CONTINUE'))
        menus.append((prefix, menu, cls))
        prefix = cls.successor
    need(prefix.terminated, 'SEQUENCE_UNTERMINATED_PATH')
    return rows, menus, prefix


def _score_menus(model, state, menus, options):
    import torch
    from farm_neat.policy import v0_scorer as A
    terms = []
    for prefix, menu, cls in menus:
        logits = model.score(state, prefix, menu.legal_ids, menu.phi,
                             memory={'hold_releases': options['hold_releases']})
        logps = A._raw_log_probs(logits, menu)
        terms.append(torch.logsumexp(logps[list(cls.indices)], 0).double())
    result = torch.stack(terms).sum()
    need(bool(torch.isfinite(result)), 'SEQUENCE_NONFINITE_LIKELIHOOD')
    return result


def record_path(actor: dict, carried_plan, actions: list[int], *, options: dict,
                reference_model, reference_identity: str) -> dict:
    """Record one observed actual path; does not execute or authenticate a game.

    The sequence collector must additionally join this to real per-dawn
    continuation/publication and replay provenance. This function deliberately
    cannot certify an arbitrary hand-authored context as a teacher trajectory.
    """
    import torch
    from farm_neat.policy import v0_scorer as A
    from farm_neat.training import v0_bc as BC
    need(type(reference_identity) is str and len(reference_identity) == 64
         and all(c in '0123456789abcdef' for c in reference_identity), 'REFERENCE_IDENTITY')
    need(reference_model.training is False, 'REFERENCE_MUST_BE_FROZEN_EVAL')
    state = public_state(actor)
    stored_options = _options_wire(carried_plan, options)
    live_options = _options_live(carried_plan, stored_options)
    rows, menus, final = _geometry(state, carried_plan, actions, live_options)
    device = next(reference_model.parameters()).device
    rng = BC._rng_state(device)
    try:
        with torch.no_grad():
            reference = float(_score_menus(reference_model, state, menus, live_options))
        need(BC._same_tree(rng, BC._rng_state(device)), 'REFERENCE_OBSERVER_CONSUMED_RNG')
    finally:
        BC._restore_rng(rng, device)
    return seal(dict(schema=SCHEMA + '-path', actor=copy.deepcopy(actor),
        initial_prefix=A.Prefix(carried_plan).payload(), raw_actions=list(actions),
        prefix_trace=rows, final_plan=final.plan.canonical(), options=stored_options,
        behavior_logp=None, reference_class_logp=reference, reference_identity=reference_identity,
        source_sha256=digest(source_pins())))


def path_logp(model, path: dict, *, carried_plan, step: int, backend='fast',
              check_reference=False):
    """Differentiable conditional successor-class path probability, not RL logp.

    carried_plan/step are mandatory independent custody inputs. The eventual
    corpus validator must supply them from executed continuation, never by
    trusting the same path's saved prefix. Saved menus/successors are witnesses;
    replay rebuilds every one from the current canonical prefix.
    """
    from farm_neat.policy import v0_scorer as A
    hashed(path, ('actor', 'initial_prefix', 'raw_actions', 'prefix_trace', 'final_plan',
                  'options', 'behavior_logp', 'reference_class_logp', 'reference_identity',
                  'source_sha256'), SCHEMA + '-path', 'SEQUENCE_PATH')
    need(path['source_sha256'] == digest(source_pins()), 'SEQUENCE_SOURCE_DRIFT')
    need(type(step) is int and step in DAWNS and path['actor'].get('step') == step,
         'SEQUENCE_STALE_DAWN')
    need(path['initial_prefix'] == A.Prefix(carried_plan).payload(), 'SEQUENCE_STALE_OR_EMPTY_PLAN_SUBSTITUTION')
    need(path['behavior_logp'] is None, 'SEQUENCE_TEACHER_HAS_NO_BEHAVIOR_LOGP')
    need(type(path['reference_class_logp']) in (int, float)
         and math.isfinite(path['reference_class_logp']) and path['reference_class_logp'] <= 0,
         'SEQUENCE_REFERENCE_LIKELIHOOD')
    need(type(path['reference_identity']) is str and len(path['reference_identity']) == 64
         and all(c in '0123456789abcdef' for c in path['reference_identity']), 'REFERENCE_IDENTITY')
    state = public_state(path['actor'])
    options = _options_live(carried_plan, path['options'])
    rows, menus, final = _geometry(state, carried_plan, path['raw_actions'], options, backend=backend)
    need(rows == path['prefix_trace'] and final.plan.canonical() == path['final_plan'],
         'SEQUENCE_STALE_SUPPORT_CLASS_OR_STOP')
    result = _score_menus(model, state, menus, options)
    if check_reference:
        need(math.isclose(float(result.detach()), path['reference_class_logp'], rel_tol=2e-6, abs_tol=3e-5),
             'SEQUENCE_REFERENCE_REPLAY_MISMATCH')
    return result

# The completed BUILD layers below extend the reviewed custody foundation.
PHASES = SPLITS
EXPECTED_FAMILIES = {"discovery": 2, "confirmation": 8, "monitor": 8}
ARMS = ("control", "challenger")
REPLAYS = (False, True)
WDL = {"WIN": 1, "DRAW": 0, "LOSS": -1}
CORPUS_SCHEMA = "farm-v0-regret-sequence-corpus-v1"
OBSERVED_CORPUS_SCHEMA = "farm-v0-regret-sequence-corpus-v2"
NATIVE_CORPUS_SCHEMA = "farm-v0-regret-sequence-native-corpus-v1"
TEACHER_SCHEMA = "farm-v0-regret-teacher-sequence-v1"
EPOCH_SCHEMA = "farm-v0-regret-sequence-epoch-v1"
STATE_FIELDS = set(ACTOR_FIELDS)
canonical = wire
_need = need
def _hash(value: Any, name: str) -> str:
    _need(isinstance(value, str) and len(value) == 64
          and all(char in "0123456789abcdef" for char in value), name + " SHA256 required")
    return value


def _bound(value: dict, name: str) -> None:
    _need(isinstance(value, dict) and value.get("content_sha256") == digest(
        {key: item for key, item in value.items() if key != "content_sha256"}),
        name + " content hash mismatch")


def _state(payload: dict):
    # Reuse the closed public-state owner, including nested product allowlists.
    need(type(payload) is dict and set(payload) == STATE_FIELDS,
         'closed actor state fields required; no future/private fields')
    state = public_state(payload)
    need(state.step in DAWNS[1:], 'later-dawn actor state required')
    return state


def _projection(payload: dict) -> dict:
    from farm_neat.economy.v0 import vocabulary as V

    _need(isinstance(payload, dict)
          and set(payload) == {"labor_factor", "belief", "hold_releases"},
          "closed actor projection fields required")
    _need(payload["labor_factor"] == 4.0
          and payload["belief"] == {product: 0 for product in V.PRODUCTS},
          "bootstrap projection settings changed")
    rows = payload["hold_releases"]
    _need(isinstance(rows, list), "HOLD rows required")
    holds = {}
    for row in rows:
        _need(isinstance(row, list) and len(row) == 4
              and all(type(value) is int for value in row), "closed HOLD row required")
        product, primitive, horizon, release = row
        _need(product not in holds and 0 <= product < len(V.PRODUCTS)
              and 0 <= primitive < len(V.PRIMITIVES) and 0 <= horizon < len(V.HORIZONS)
              and 0 <= release <= V.LAST_STEP, "valid distinct HOLD binding required")
        holds[product] = ((primitive, horizon), release)
    return {"labor_factor": 4.0, "belief": dict(payload["belief"]), "hold_releases": holds}


def _prefix(payload: dict):
    from farm_neat.economy.v0.plan import Plan
    from farm_neat.policy.v0_scorer import Prefix

    _need(isinstance(payload, dict)
          and set(payload) == {"plan", "last_key", "edit_count", "terminated"},
          "closed prefix fields required")
    try:
        return Prefix(Plan.from_canonical(payload["plan"]), payload["last_key"],
                      payload["edit_count"], payload["terminated"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid canonical prefix") from exc


def _validate_epoch(row: dict, expected_step: int, *, menu_builder, model=None, allow_empty=False) -> tuple[Any, Any, tuple[int, ...]]:
    """Rebuild one real first-choice STOP target on its actual carried Plan."""
    from farm_neat.economy.v0 import vocabulary as V
    from farm_neat.economy.v0.plan import Plan
    from farm_neat.policy import v0_scorer as A

    _need(isinstance(row, dict) and set(row) == {"schema", "step", "actor", "target", "custody"}
          and row["schema"] == EPOCH_SCHEMA and row["step"] == expected_step,
          "ordered closed epoch record required")
    actor = row["actor"]
    _need(isinstance(actor, dict) and set(actor) == {"state", "prefix", "projection"},
          "closed actor context required")
    state = _state(actor["state"])
    _need(state.step == expected_step, "actor/epoch step mismatch")
    prefix = _prefix(actor["prefix"])
    _need(prefix.last_key == -1 and prefix.edit_count == 0 and not prefix.terminated,
          "fresh prefix at each real dawn required")
    _need(allow_empty or prefix.plan != Plan(), "empty carried Plan substitution refused")
    options = _projection(actor["projection"])
    _options_wire(prefix.plan, options)  # Validate actual Plan/HOLD correspondence.
    menu = menu_builder(state, prefix, projection_kwargs=options)
    target = row["target"]
    target_fields = {"raw_id", "support", "phi_sha256", "class_id", "class_members",
                     "continuation", "forced_stop", "termination", "behavior_logp",
                     "reference_class_logp"}
    _need(isinstance(target, dict) and set(target) == target_fields, "closed STOP target fields required")
    _need(target["behavior_logp"] is None, "teacher behavior logp must be null")
    _need(target["raw_id"] == V.STOP_ID and target["forced_stop"] is False
          and target["termination"] == "STOP", "sampled first-choice STOP required")
    _need(tuple(target["support"]) == menu.legal_ids and target["phi_sha256"] == A._digest(menu.phi),
          "stale current-prefix support or phi")
    stop_index = menu.legal_ids.index(V.STOP_ID)
    cls = next((item for item in menu.classes if stop_index in item.indices), None)
    _need(cls is not None, "STOP successor class absent")
    _need(target["class_id"] == cls.class_id
          and target["class_members"] == [menu.legal_ids[index] for index in cls.indices]
          and target["continuation"] == cls.successor.payload(),
          "stale STOP successor class")
    _need(cls.successor.terminated and cls.successor.plan == prefix.plan,
          "STOP must retain actual carried Plan")
    custody = row["custody"]
    _need(isinstance(custody, dict) and set(custody) == {"before_plan_sha256", "after_plan_sha256",
          "hold_releases", "allocation_sha256", "controller_sha256"}, "closed custody fields required")
    _need(custody["before_plan_sha256"] == prefix.plan.digest()
          and custody["after_plan_sha256"] == cls.successor.plan.digest()
          and custody["hold_releases"] == actor["projection"]["hold_releases"],
          "stale Plan/HOLD custody")
    _hash(custody["allocation_sha256"], "allocation")
    _hash(custody["controller_sha256"], "controller")
    if model is None:
        saved = target["reference_class_logp"]
        _need(saved is None or (type(saved) in (int, float) and math.isfinite(saved)),
              "finite or explicitly unbound reference likelihood required")
    else:
        logits = model.score(state, prefix, menu.legal_ids, menu.phi,
                             memory={"hold_releases": options["hold_releases"]})
        actual = float(A.class_log_probs(logits, menu)[list(menu.classes).index(cls)].detach())
        saved = target["reference_class_logp"]
        _need(type(saved) in (int, float) and math.isfinite(saved)
              and math.isclose(saved, actual, rel_tol=2e-6, abs_tol=3e-5),
              "reference class likelihood mismatch")
    return state, menu, cls.indices


def validate_sequence_corpus(corpus: dict, registration: dict, *, menu_builder=None, model=None) -> dict:
    """Validate all four discovery sequences and 116 actual later-dawn targets.

    ``menu_builder`` is injectable only for source-only diagnostic fixtures.  A
    production caller omits it and therefore uses the exact fast V0 menu owner.
    """
    from farm_neat.economy.v0.plan import Plan
    from farm_neat.policy import v0_menu_fast

    validate_registration(registration)
    native = registration['scope'] == 'PRODUCTION_NOT_STRENGTH'
    if native:
        need(menu_builder is None and model is None, 'NATIVE_CORPUS_MODEL_OR_MENU_INJECTION_REFUSED')
    else:
        _diagnostic_execution_only(registration)
    _bound(corpus, "corpus")
    joined = corpus.get("schema") == OBSERVED_CORPUS_SCHEMA or native
    corpus_fields = {"schema", "registration_sha256", "parent_commit_sha256", "parent_state_sha256",
                     "source_closure_sha256", "split", "teacher_sequences", "content_sha256"}
    if native:
        corpus_fields.update(('collection', 'reference'))
        schemas = (NATIVE_CORPUS_SCHEMA,)
    else:
        if joined:
            corpus_fields.add("execution_evidence")
        schemas = (CORPUS_SCHEMA, OBSERVED_CORPUS_SCHEMA)
    _need(set(corpus) == corpus_fields and corpus["schema"] in schemas,
          "closed sequence corpus required")
    _need(corpus["registration_sha256"] == registration["content_sha256"],
          "foreign sequence registration")
    _need(corpus["parent_commit_sha256"] == registration["parent"]["commit_sha256"]
          and corpus["parent_state_sha256"] == registration["parent"]["state_sha256"],
          "stale parent checkpoint")
    need(corpus["source_closure_sha256"] == digest(registration['source_pins']),
         'CORPUS_SOURCE_CLOSURE_MISMATCH')
    _need(corpus["split"] == "discovery", "only discovery teachers may optimize")
    if native:
        _validate_native_corpus_join(corpus, registration)
    elif joined:
        _validate_observed_corpus_join(corpus, registration)
    sequences = corpus["teacher_sequences"]
    _need(isinstance(sequences, list) and len(sequences) == 4, "fixed sequence denominator four required")
    expected = {(world["family_id"], world["seed"], seat)
                for world in registration["worlds"]["discovery"] for seat in (0, 1)}
    observed = set()
    steps = list(range(24, 719, 24))
    builder = v0_menu_fast.build_menu if menu_builder is None else menu_builder
    rows_checked = empty = behavior = 0
    for sequence in sequences:
        seq_fields = {"schema", "sequence_id", "family_id", "seed", "seat", "arm", "replay",
                      "opening_trace_sha256", "game_receipt_sha256", "epochs"}
        _need(isinstance(sequence, dict) and set(sequence) == seq_fields
              and sequence["schema"] == TEACHER_SCHEMA, "closed teacher sequence required")
        identity = (sequence["family_id"], sequence["seed"], sequence["seat"])
        _need(identity in expected and identity not in observed, "discovery family/seat overlap or omission")
        observed.add(identity)
        _need(sequence["arm"] == "challenger" and sequence["replay"] is False,
              "primary challenger teacher required")
        _hash(sequence["opening_trace_sha256"], "opening trace")
        _hash(sequence["game_receipt_sha256"], "game receipt")
        epochs = sequence["epochs"]
        _need(isinstance(epochs, list) and len(epochs) == 29, "29 later-dawn rows required")
        for step, row in zip(steps, epochs):
            _validate_epoch(row, step, menu_builder=builder, model=model, allow_empty=joined)
            rows_checked += 1
            behavior += row["target"]["behavior_logp"] is not None
            empty += row["actor"]["prefix"]["plan"] == Plan().canonical()
    _need(observed == expected, "complete discovery family/seat join required")
    if native:
        _validate_native_reference(corpus, registration)
    return {
        "schema": SCHEMA + "-corpus-validation",
        "registration_sha256": registration["content_sha256"],
        "corpus_sha256": corpus["content_sha256"],
        "sequence_denominator": 4,
        "teacher_rows": rows_checked,
        "steps": steps,
        "behavior_logps": behavior,
        "empty_carried_plans": empty,
        "reference_likelihoods_bound": all(
            row["target"]["reference_class_logp"] is not None
            for sequence in sequences for row in sequence["epochs"]
        ),
        "native_games": 0,
        "validation_scope": ("NATIVE_DISCOVERY_CUSTODY" if native else
                             "DIAGNOSTIC_OBSERVED_CUSTODY" if joined else "DIAGNOSTIC_STRUCTURAL_ONLY"),
        "execution_custody_joined": joined,
        "observed_executions_checked": 16 if joined else 0,
        "native_replay_verified": False,
        "scientific_outcome": "NOT_RUN",
    }


EXECUTION_SCHEMA = "farm-v0-regret-sequence-execution-v1"
EXECUTION_FIELDS = {
    "schema", "registration_sha256", "launch", "status", "physical_games_started",
    "physical_games_completed", "physical_steps", "wdl", "own_actions_sha256",
    "joint_actions_sha256", "terminal_environment_sha256", "epochs_sha256",
    "actor_instance_sha256", "source_closure_sha256", "runtime_sha256",
    "initial_state_sha256", "error", "content_sha256",
}
REPLAY_EQUAL_FIELDS = (
    "wdl", "own_actions_sha256", "joint_actions_sha256", "terminal_environment_sha256",
    "epochs_sha256", "initial_state_sha256",
)


def execution_runtime_sha256(registration: dict) -> str:
    """Expected runtime; native summaries additionally require full producer evidence."""
    if registration['scope'] == 'PRODUCTION_NOT_STRENGTH':
        return digest(registration['parent']['runtime'])
    _diagnostic_execution_only(registration)
    return digest(dict(scope='DIAGNOSTIC_CPU', runner='INJECTED_NATIVE_FREE', native_games=0))


def _validate_execution(record: dict, registration: dict, launch: dict) -> None:
    native = registration['scope'] == 'PRODUCTION_NOT_STRENGTH'
    expected = EXECUTION_FIELDS - {'schema', 'content_sha256'}
    hashed(record, expected | ({'native_evidence'} if native else set()),
           NATIVE_EXECUTION_SCHEMA if native else EXECUTION_SCHEMA, 'EXECUTION')
    if native:
        evidence = record['native_evidence']
        fields(evidence, ('path', 'sha256', 'bytes'), 'NATIVE_EVIDENCE_FIELDS')
        path = safe_path(evidence['path'])
        need(path.is_file() and type(evidence['bytes']) is int
             and path.stat().st_size == evidence['bytes'] and file_sha(path) == evidence['sha256'],
             'NATIVE_EVIDENCE_MISSING_OR_CHANGED')
        observed = read_json(path)
        validate_native_execution(observed, registration)
        need(record == _native_execution_summary(observed, registration, path),
             'NATIVE_EXECUTION_SUMMARY_JOIN')
    need(record['registration_sha256'] == registration['content_sha256']
         and record['launch'] == launch, 'FOREIGN_EXECUTION_REGISTRATION_OR_LAUNCH')
    need(type(record['physical_games_started']) is int and record['physical_games_started'] == 1,
         'EXECUTION_START_COUNT')
    need(type(record['physical_games_completed']) is int
         and type(record['physical_steps']) is int and 0 <= record['physical_steps'] <= 719,
         'EXECUTION_STEP_OR_COMPLETION_COUNT')
    need(record['source_closure_sha256'] == digest(registration['source_pins']),
         'EXECUTION_SOURCE_CLOSURE_MISMATCH')
    need(record['runtime_sha256'] == execution_runtime_sha256(registration),
         'EXECUTION_RUNTIME_MISMATCH')
    if record['status'] == 'COMPLETE':
        need(record['physical_games_completed'] == 1 and record['physical_steps'] == 719
             and type(record['wdl']) is str and record['wdl'] in WDL and record['error'] is None,
             'COMPLETE_719_STEP_TERMINAL_WDL_REQUIRED')
        for name in ('own_actions_sha256', 'joint_actions_sha256', 'terminal_environment_sha256',
                     'epochs_sha256', 'actor_instance_sha256', 'initial_state_sha256'):
            _hash(record[name], name)
    else:
        need(record['status'] == 'FAILED' and record['physical_games_completed'] == 0
             and record['wdl'] is None, 'EXPLICIT_FAILED_LAUNCH_REQUIRED')
        fields(record['error'], ('type', 'message'), 'EXECUTION_ERROR_FIELDS')
        need(all(type(v) is str and v for v in record['error'].values()),
             'EXECUTION_ERROR_VALUES')


def _validate_replay(primary: dict, replay: dict) -> None:
    need(all(primary[field] == replay[field] for field in REPLAY_EQUAL_FIELDS),
         'exact replay mismatch')
    need(primary['actor_instance_sha256'] != replay['actor_instance_sha256'],
         'fresh replay actor identity required')
    if primary['schema'] == NATIVE_EXECUTION_SCHEMA or replay['schema'] == NATIVE_EXECUTION_SCHEMA:
        need(primary['schema'] == replay['schema'] == NATIVE_EXECUTION_SCHEMA, 'NATIVE_REPLAY_SCOPE_JOIN')
        observed = []
        for summary in (primary, replay):
            evidence = summary['native_evidence']
            path = safe_path(evidence['path'])
            need(file_sha(path) == evidence['sha256'], 'NATIVE_REPLAY_EVIDENCE_CHANGED')
            observed.append(read_json(path))
        left, right = observed
        need(right['launch']['replay'] and not left['launch']['replay']
             and right['launch']['primary_index'] == left['launch']['index'], 'NATIVE_REPLAY_LAUNCH_JOIN')
        for name in left:
            if name not in ('content_sha256', 'launch', 'actor_instance_sha256'):
                need(left[name] == right[name], 'NATIVE_EXACT_REPLAY_MISMATCH:' + name)


def _execution_rows(records: list[dict], registration: dict, phase: str) -> dict[str, dict]:
    validate_registration(registration)
    need(phase in PHASES, 'known phase required')
    expected = {row['launch_id']: row for row in registration['launches'] if row['phase'] == phase}
    need(type(records) is list and len(records) == len(expected), 'complete fixed launch set required')
    found, instances = {}, set()
    for record in records:
        need(type(record) is dict and type(record.get('launch')) is dict, 'EXECUTION_FIELDS')
        launch = record['launch']
        launch_id = launch.get('launch_id')
        need(type(launch_id) is str and launch_id in expected and launch_id not in found,
             'foreign, changed, or duplicate launch')
        _validate_execution(record, registration, expected[launch_id])
        need(record['status'] == 'COMPLETE', 'failed/incomplete fixed launch set')
        need(record['actor_instance_sha256'] not in instances, 'fresh actor instance per execution required')
        instances.add(record['actor_instance_sha256'])
        found[launch_id] = record
    by_index = {record['launch']['index']: record for record in found.values()}
    for record in found.values():
        if record['launch']['replay']:
            primary = by_index.get(record['launch']['primary_index'])
            need(primary is not None, 'primary/replay join required')
            _validate_replay(primary, record)
    return found


def _seat_deltas(found: dict[str, dict]) -> dict[tuple[str, int], float]:
    primary = [row for row in found.values() if not row["launch"]["replay"]]
    groups: dict[tuple[str, int], dict[str, dict]] = {}
    for row in primary:
        launch = row["launch"]
        groups.setdefault((launch["family_id"], launch["seat"]), {})[launch["arm"]] = row
    result = {}
    for key, pair in groups.items():
        _need(set(pair) == set(ARMS), "both policy arms required")
        result[key] = float(WDL[pair["challenger"]["wdl"]] - WDL[pair["control"]["wdl"]])
    return result


def discovery_gate(records: list[dict], registration: dict) -> dict:
    """The predeclared 16-start cheap falsifier; it never tunes the proposal."""
    found = _execution_rows(records, registration, "discovery")
    deltas = _seat_deltas(found)
    _need(len(deltas) == 4, "two discovery worlds and both seats required")
    values = [deltas[key] for key in sorted(deltas)]
    mean = math.fsum(values) / len(values)
    proceed = mean >= 1.0 and all(value >= 0 for value in values)
    return seal({
        "schema": SCHEMA + "-discovery-gate",
        "registration_sha256": registration["content_sha256"],
        "status": "PASS" if proceed else "CLOSED_NEGATIVE",
        "proceed": proceed,
        "seat_deltas": values,
        "mean_paired_wdl_gain": mean,
        "physical_starts": len(found),
        "n_world_families": 2,
        "proposal_count": 1,
    })


def sequence_admission(records: list[dict], registration: dict) -> dict:
    """Apply the fixed-n terminal-WDL policy-pair bound to confirmation only."""
    found = _execution_rows(records, registration, "confirmation")
    seats = _seat_deltas(found)
    worlds = sorted({family for family, _seat in seats})
    _need(len(worlds) == 8 and len(seats) == 16, "eight confirmation families and both seats required")
    deltas = [math.fsum(seats[(family, seat)] for seat in (0, 1)) / 2 for family in worlds]
    mean = math.fsum(deltas) / 8
    penalty = math.sqrt(8 * math.log(20) / 8)
    lower = mean - penalty
    admitted = lower > 0
    return seal({
        "schema": "farm-v0-sequence-policy-pair-admission-v1",
        "registration_sha256": registration["content_sha256"],
        "status": "ADMITTED" if admitted else "REJECTED",
        "metric": "TERMINAL_WDL",
        "n_world_families": 8,
        "world_deltas": deltas,
        "mean": mean,
        "alpha": 0.05,
        "penalty": penalty,
        "lower_bound": lower,
        "teacher_gate": int(admitted),
        "teacher_weight": min(2.0, max(0.0, lower)) / 2,
        "physical_starts": len(found),
        "primary_starts": sum(not row["launch"]["replay"] for row in found.values()),
        "replay_starts": sum(row["launch"]["replay"] for row in found.values()),
        "independent_replay_worlds": 0,
        "label_count": 1 if admitted else 0,
    })


def validate_sequence_admission(admission: dict, registration: dict) -> dict:
    """Recompute every frozen scalar; this remains diagnostic, not native proof."""
    validate_registration(registration)
    _diagnostic_execution_only(registration)
    hashed(admission, ('registration_sha256', 'status', 'metric', 'n_world_families',
        'world_deltas', 'mean', 'alpha', 'penalty', 'lower_bound', 'teacher_gate',
        'teacher_weight', 'physical_starts', 'primary_starts', 'replay_starts',
        'independent_replay_worlds', 'label_count'),
        'farm-v0-sequence-policy-pair-admission-v1', 'ADMISSION')
    bound = policy_pair_bound(admission['world_deltas'])
    expected = dict(registration_sha256=registration['content_sha256'],
        status='ADMITTED' if bound['gate'] else 'REJECTED', metric='TERMINAL_WDL',
        n_world_families=8, mean=bound['mean'], alpha=.05,
        penalty=math.sqrt(8 * math.log(20) / 8), lower_bound=bound['lower_bound'],
        teacher_gate=int(bound['gate']), teacher_weight=bound['weight'],
        physical_starts=64, primary_starts=32, replay_starts=32,
        independent_replay_worlds=0, label_count=int(bound['gate']))
    for name, value in expected.items():
        need(type(admission[name]) is type(value) and admission[name] == value,
             'ADMISSION_FROZEN_SCALAR_MISMATCH: ' + name)
    return admission


def _monitor_stratum(found: dict[str, dict], opponent: str) -> dict:
    primary = [row for row in found.values()
               if not row["launch"]["replay"] and row["launch"]["opponent"] == opponent]
    groups: dict[tuple[str, int], dict[str, dict]] = {}
    for row in primary:
        launch = row["launch"]
        groups.setdefault((launch["family_id"], launch["seat"]), {})[launch["arm"]] = row
    _need(len(groups) == 16 and all(set(pair) == set(ARMS) for pair in groups.values()),
          "complete monitor policy pairs required")
    seat_deltas = {
        key: float(WDL[pair["challenger"]["wdl"]] - WDL[pair["control"]["wdl"]])
        for key, pair in groups.items()
    }
    families = sorted({family for family, _ in seat_deltas})
    world_deltas = [math.fsum(seat_deltas[(family, seat)] for seat in (0, 1)) / 2
                    for family in families]
    bound = policy_pair_bound(world_deltas)
    return {
        "opponent": opponent,
        "n_world_families": 8,
        "seat_deltas": [seat_deltas[key] for key in sorted(seat_deltas)],
        "world_deltas": world_deltas,
        "minimum_world_delta": min(world_deltas),
        "minimum_seat_delta": min(seat_deltas.values()),
        "mean": bound["mean"],
        "lower_bound": bound["lower_bound"],
    }


def monitor_sequence(records: list[dict], registration: dict, *,
                     parent_checkpoint_sha256: str, child_checkpoint_sha256: str) -> dict:
    """Reduce the frozen unrestricted parent/child monitor without teacher inputs."""
    _hash(parent_checkpoint_sha256, "parent checkpoint")
    _hash(child_checkpoint_sha256, "child checkpoint")
    _need(parent_checkpoint_sha256 != child_checkpoint_sha256,
          "distinct parent and child checkpoints required")
    found = _execution_rows(records, registration, "monitor")
    strata = {name: _monitor_stratum(found, name) for name in ("pass", "router")}
    passed = (strata["pass"]["lower_bound"] > 0
              and strata["router"]["minimum_seat_delta"] >= 0)
    return seal({
        "schema": SCHEMA + "-monitor-result",
        "registration_sha256": registration["content_sha256"],
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "child_checkpoint_sha256": child_checkpoint_sha256,
        "status": "PASS" if passed else "CLOSED_NEGATIVE",
        "physical_starts": len(found),
        "strata": strata,
        "teacher_input": False,
        "search_enabled": False,
        "monitor_only": True,
        "strength_claim": False,
    })


def _json_read(path: Path) -> dict:
    return read_json(path)


def _write_new(path: Path, value: dict) -> None:
    write_new(path, value)


def _diagnostic_execution_only(registration: dict) -> None:
    """Keep source-only fixture consumers separate from native evidence.

    Production uses the separate cold-loader, charged-journal and frozen-child
    campaign paths. A diagnostic boolean or matching summary hashes cannot
    promote source-only fixtures into native execution evidence.
    """
    need(registration['scope'] == 'DIAGNOSTIC_CPU',
         'PRODUCTION_EXECUTION_REQUIRES_AUTHENTICATED_CAMPAIGN')


def _journal_snapshot(journal: SequenceJournal, *, stop_before: int | None = None
                      ) -> tuple[list[dict], list[dict], dict]:
    events = journal.events  # Always authenticate the ENTIRE chain, including cached runs.
    if stop_before is not None:
        events = [row for row in events if row['launch_index'] < stop_before]
    results, failures = [], []
    for event in events:
        if event['status'] == 'FINISHED':
            job = journal.registration['launches'][event['launch_index']]
            _validate_execution(event['result'], journal.registration, job)
            need(event['result']['status'] == 'COMPLETE', 'JOURNAL_FAILED_RESULT_NOT_FAILURE')
            results.append(event['result'])
        elif event['status'] == 'FAILED':
            failures.append(event)
    starts = sum(event['status'] == 'STARTED' for event in events)
    counts = dict(physical_starts=starts, completed=len(results), failed=len(failures),
                  unknown=starts - len(results) - len(failures),
                  native_games=starts if journal.registration['scope'] == 'PRODUCTION_NOT_STRENGTH' else 0)
    return events, results, counts


def _development_snapshot(registration: dict, journal: SequenceJournal, *,
                          stop_before: int | None = None) -> dict | None:
    """Derive durable status from records; a RESULT file is only a checked cache."""
    events, records, counts = _journal_snapshot(journal, stop_before=stop_before)
    need(all(row['launch_index'] < 80 for row in events), 'DEVELOPMENT_JOURNAL_FOREIGN_PHASE')
    discovery = admission = last = reason = None
    status = None
    try:
        # Detect a bad replay at its pair, rather than spending the rest of a stage.
        by_index = {row['launch']['index']: row for row in records}
        for index, replay in by_index.items():
            if replay['launch']['replay']:
                primary = by_index.get(replay['launch']['primary_index'])
                need(primary is not None, 'primary/replay join required')
                _validate_replay(primary, replay)
        for phase in ('discovery', 'confirmation'):
            phase_records = [row for row in records if row['launch']['phase'] == phase]
            expected = 16 if phase == 'discovery' else 64
            if len(phase_records) < expected:
                break
            if phase == 'discovery':
                discovery = discovery_gate(phase_records, registration)
                last = 'discovery'
                if not discovery['proceed']:
                    need(counts['physical_starts'] == 16, 'STARTS_AFTER_NEGATIVE_DISCOVERY')
                    status, reason = 'CLOSED_NEGATIVE', 'DISCOVERY_PROPOSAL_FAILED'
                    break
            else:
                admission = sequence_admission(phase_records, registration)
                last = 'confirmation'
                if admission['status'] == 'ADMITTED':
                    status, reason = 'READY_TO_FIT', 'CONFIRMATION_ADMITTED'
                else:
                    status, reason = 'CLOSED_NEGATIVE', 'CONFIRMATION_NOT_ADMITTED'
    except SequenceError as exc:
        status, reason = 'PARTIAL', 'EXECUTION_VALIDATION_FAILED: ' + str(exc)
    if counts['failed']:
        status, reason = 'PARTIAL', 'DURABLE_FAILED_LAUNCH_NO_AUTOMATIC_RETRY'
    elif counts['unknown']:
        status, reason = 'PARTIAL', 'UNKNOWN_DURABLE_LAUNCH_NO_AUTOMATIC_RETRY'
    if status is None:
        return None
    return seal(dict(schema=SCHEMA + '-collection-result',
        registration_sha256=registration['content_sha256'], status=status, reason=reason,
        last_complete_stage=last, counts=counts, discovery=discovery, admission=admission,
        journal_head_sha256=events[-1]['content_sha256'] if events else None,
        scope=registration['scope'], scientific_outcome=(
            'NOT_RUN' if registration['scope'] == 'DIAGNOSTIC_CPU' else 'NOT_EVALUATED')))


def collect_sequence(registration: dict, *, runner=None, output: str | Path,
                     native: bool = False, preflight: str | Path | None = None) -> dict:
    """Collect development with a persistent, cross-output production lease."""
    need(type(native) is bool, 'EXPLICIT_RUNNER_SCOPE')
    if native:
        need(runner is None, 'INJECTED_NATIVE_RUNNER_REFUSED')
        return run_sequence_campaign(registration, collection=output, preflight=preflight,
                                     stop_after='development')
    return _collect_sequence(registration, runner=runner, output=output, native=False, preflight=preflight)


def _collect_sequence(registration: dict, *, runner=None, output: str | Path,
                      native: bool = False, preflight: str | Path | None = None) -> dict:
    """Run the fixed development fixture through the one durable launch journal.

    The collector owns ordering, reservation, failure retention and fixed gates.
    The injected runner owns one source-only execution. Ordinary exceptions and
    invalid returns become durable FAILED starts; BaseException/process death
    leaves a charged STARTED entry and is never automatically retried. Native
    execution accepts no injected runner: it requires a guarded accepted preflight,
    an authenticated registered parent and the concrete original-rule backend.
    """
    validate_registration(registration)
    need(type(native) is bool, 'EXPLICIT_RUNNER_SCOPE')
    if native:
        need(runner is None, 'INJECTED_NATIVE_RUNNER_REFUSED')
        runner = NativeSequenceRunner(registration, output=output, preflight=preflight)
    else:
        need(callable(runner) and preflight is None, 'EXPLICIT_RUNNER_SCOPE')
        _diagnostic_execution_only(registration)
    output = safe_path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Lock the entire orchestration as well as individual journal appends. Two
    # callers must not share a launch even when each append is individually safe.
    lock_path = safe_path(output / 'COLLECTION.lock')
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SequenceError('COLLECTION_ALREADY_RUNNING') from exc
        registration_path = safe_path(output / 'REGISTRATION.json')
        if registration_path.exists():
            need(read_json(registration_path) == registration, 'COLLECTION_REGISTRATION_CHANGED')
        else:
            write_new(registration_path, registration)
        journal = SequenceJournal(output / 'JOURNAL.jsonl', registration)
        result_path = safe_path(output / 'RESULT.json')
        snapshot = _development_snapshot(registration, journal)
        if result_path.exists():
            need(snapshot is not None and read_json(result_path) == snapshot,
                 'CACHED_RESULT_JOURNAL_MISMATCH')
            return snapshot
        if snapshot is not None:
            write_new(result_path, snapshot)
            return snapshot
        while journal.starts < 80:
            job = registration['launches'][journal.starts]
            journal.reserve(job)  # The start is fsynced before user code runs.
            returned = None
            try:
                returned = runner(copy.deepcopy(job))
                validate_registration(registration)  # A runner may not mutate the source closure.
                _validate_execution(returned, registration, job)
                if returned['status'] == 'FAILED':
                    # Retain the literal failed execution, not only its message.
                    journal.finish(job, error=wire(dict(
                        code='RUNNER_REPORTED_FAILED', execution=returned)))
                else:
                    journal.finish(job, result=returned)
            except Exception as exc:
                error = dict(code='RUNNER_EXCEPTION_OR_INVALID_RECORD',
                             exception=type(exc).__name__, message=str(exc))
                # Hashes are not evidence of correctness; retain the rejected
                # JSON itself when serializable, so the maintainer can inspect it.
                try:
                    wire(returned)
                    error['returned_record'] = returned
                except (TypeError, ValueError, OverflowError):
                    error['returned_record_type'] = type(returned).__name__
                journal.finish(job, error=wire(error))
            snapshot = _development_snapshot(registration, journal)
            if snapshot is not None:
                write_new(result_path, snapshot)
                return snapshot
        raise AssertionError('complete development must produce a fixed-stage result')


def _examples(corpus: dict, *, menu_builder) -> list[tuple]:
    examples = []
    for sequence in corpus["teacher_sequences"]:
        for row in sequence["epochs"]:
            actor = row["actor"]
            state = _state(actor["state"])
            prefix = _prefix(actor["prefix"])
            options = _projection(actor["projection"])
            menu = menu_builder(state, prefix, projection_kwargs=options)
            target = row["target"]
            cls = next((item for item in menu.classes if item.class_id == target["class_id"]), None)
            _need(cls is not None and target["class_members"] == [menu.legal_ids[i] for i in cls.indices],
                  "training target class changed")
            examples.append((state, prefix, menu, cls.indices,
                             {"hold_releases": options["hold_releases"]}, row))
    _need(len(examples) == 116, "fixed four-by-29 teacher buffer required")
    return examples


def _batch(model, examples):
    return model.score_batch(
        [row[0] for row in examples],
        [row[1] for row in examples],
        [row[2].legal_ids for row in examples],
        [row[2].phi for row in examples],
        memories=[row[4] for row in examples],
    )


def bind_sequence_likelihoods(corpus: dict, registration: dict, model, *, menu_builder=None) -> dict:
    """Bind diagnostic STOP likelihoods; native references have a cold-loader API."""
    need(registration.get('scope') == 'DIAGNOSTIC_CPU', 'NATIVE_REFERENCE_REQUIRES_AUTHENTICATED_BINDER')
    import torch
    from farm_neat.policy import v0_menu_fast
    from farm_neat.policy import v0_scorer as A

    validation = validate_sequence_corpus(corpus, registration, menu_builder=menu_builder)
    _need(not validation["reference_likelihoods_bound"], "reference likelihoods already bound")
    builder = v0_menu_fast.build_menu if menu_builder is None else menu_builder
    result = copy.deepcopy(corpus)
    examples = _examples(result, menu_builder=builder)
    from farm_neat.training import v0_bc as BC
    was_training = model.training
    device = next(model.parameters()).device
    rng = BC._rng_state(device)
    try:
        model.eval()
        with torch.no_grad():
            scored = _batch(model, examples)
            for index, example in enumerate(examples):
                logp = -A.class_nll(scored.logits[index], example[3], valid=scored.valid[index])
                example[5]["target"]["reference_class_logp"] = float(logp)
            result = seal(result)
            # Replay under the same frozen eval mode, not restored train mode.
            checked = validate_sequence_corpus(result, registration, menu_builder=builder, model=model)
        _need(checked["reference_likelihoods_bound"], "reference likelihood binding failed")
        need(BC._same_tree(rng, BC._rng_state(device)), 'REFERENCE_OBSERVER_CONSUMED_RNG')
        return result
    finally:
        model.train(was_training)
        BC._restore_rng(rng, device)


def sequence_loss(model, corpus: dict, registration: dict, *, gate: int, weight: float,
                  menu_builder=None):
    """Exact teacher-only loss: ``gate*weight/4 * sum(116 STOP NLLs)``."""
    import torch
    from farm_neat.policy import v0_menu_fast
    from farm_neat.policy import v0_scorer as A

    _diagnostic_execution_only(registration)  # Native fitting uses its authenticated campaign transaction.
    validate_sequence_corpus(corpus, registration, menu_builder=menu_builder)
    _need(type(gate) is int and gate in (0, 1), "detached binary sequence gate required")
    _need(type(weight) in (int, float) and math.isfinite(weight) and 0 <= weight <= 1,
          "detached finite sequence weight required")
    builder = v0_menu_fast.build_menu if menu_builder is None else menu_builder
    examples = _examples(corpus, menu_builder=builder)
    scored = _batch(model, examples)
    losses = [A.class_nll(scored.logits[index], example[3], valid=scored.valid[index])
              for index, example in enumerate(examples)]
    scale = scored.logits.new_tensor(float(gate) * float(weight) / 4.0).detach()
    return torch.stack(losses).sum() * scale


FIT_SCHEMA = 'farm-v0-regret-sequence-fit-v2'


def _fit_optimizer(model):
    import torch
    return torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.0, foreach=False)


def _fit_identity(model, registration: dict, corpus: dict, admission: dict,
                  native_authorization: dict | None = None) -> dict:
    from farm_neat.training import v0_bc as BC
    value = dict(schema=FIT_SCHEMA + '-lineage', scope=registration['scope'],
        registration_sha256=registration['content_sha256'],
        corpus_sha256=corpus['content_sha256'], admission_sha256=admission['content_sha256'],
        registered_parent=copy.deepcopy(registration['parent']),
        model_identity=copy.deepcopy(model.get_extra_state()),
        parameter_order=[name for name, _ in model.named_parameters()],
        runtime=BC.runtime_identity(next(model.parameters()).device),
        maximum_updates=2, scientific_outcome='NOT_RUN')
    if registration['scope'] == 'PRODUCTION_NOT_STRENGTH':
        need(type(native_authorization) is dict, 'NATIVE_FIT_AUTHORIZATION_REQUIRED')
        value['native_authorization'] = copy.deepcopy(native_authorization)
    else:
        need(native_authorization is None, 'DIAGNOSTIC_NATIVE_AUTHORIZATION_REFUSED')
    return seal(value)


def _publish_fit_json(path: Path, value: dict) -> None:
    """Reuse exclusive/fsynced JSON and BC's atomic hard-link commit protocol."""
    from farm_neat.training import v0_bc as BC
    path = safe_path(path)
    pending = safe_path(path.with_name('.' + path.name + '.pending'))
    published = False
    try:
        write_new(pending, value)
        need(read_json(pending) == value, 'FIT_JSON_ROUNDTRIP')
        BC._fsync_dir(path.parent)
        os.link(pending, path)  # Exclusive publication; never replaces a parent.
        published = True
        BC._fsync_dir(path.parent)
        pending.unlink()
        BC._fsync_dir(path.parent)
    except BaseException:
        if published:
            path.unlink(missing_ok=True)
            BC._fsync_dir(path.parent)
        raise


def _fit_live(model, optimizer) -> dict:
    from farm_neat.training import v0_bc as BC
    return BC._cpu_tree(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        training=model.training, gradients={n: p.grad for n, p in model.named_parameters()},
        rng=BC._rng_state(next(model.parameters()).device)))


def _restore_fit_live(model, optimizer, value: dict) -> None:
    from farm_neat.training import v0_bc as BC
    model.load_state_dict(value['model'], strict=True)
    optimizer.load_state_dict(value['optimizer'])
    for name, parameter in model.named_parameters():
        grad = value['gradients'][name]
        parameter.grad = None if grad is None else grad.to(parameter.device).clone()
    model.train(value['training'])
    BC._restore_rng(value['rng'], next(model.parameters()).device)


def _validate_fit_state(state: dict, model, identity: dict, *, update: int,
                        parent_sha256: str | None) -> None:
    """Validate every load-sensitive field BEFORE changing the live model/RNG."""
    import torch
    from farm_neat.training import v0_bc as BC
    fields(state, ('schema', 'global_update', 'lineage_sha256', 'parent_manifest_sha256',
        'registration_sha256', 'corpus_sha256', 'admission_sha256', 'model_identity',
        'model', 'optimizer', 'optimizer_state_ids', 'rng', 'cursor', 'parameter_order',
        'model_training', 'runtime'), 'FIT_STATE_FIELDS')
    need(state['schema'] == FIT_SCHEMA + '-state', 'FIT_STATE_VERSION')
    need(type(state['global_update']) is int and state['global_update'] == update
         and 0 <= update <= 2, 'FIT_STATE_AGE')
    need(state['lineage_sha256'] == identity['content_sha256']
         and state['parent_manifest_sha256'] == parent_sha256, 'FIT_STATE_LINEAGE')
    need(all(state[key] == identity[key] for key in ('registration_sha256',
        'corpus_sha256', 'admission_sha256', 'model_identity', 'parameter_order', 'runtime')),
        'FIT_STATE_IDENTITY')
    need(state['cursor'] == dict(sequence_denominator=4, teacher_rows_per_update=116,
                                completed_updates=update), 'FIT_CURSOR')
    need(type(state['model_training']) is bool
         and (update == 0 or state['model_training']), 'FIT_MODEL_MODE')
    need(BC._finite_tree(state), 'FIT_NONFINITE_CONTINUATION')
    expected = model.state_dict()
    need(isinstance(state['model'], dict) and state['model'].keys() == expected.keys(),
         'FIT_MODEL_KEYS')
    for name, value in expected.items():
        actual = state['model'][name]
        if isinstance(value, torch.Tensor):
            need(isinstance(actual, torch.Tensor) and actual.shape == value.shape
                 and actual.dtype == value.dtype, 'FIT_MODEL_SHAPE_DTYPE')
        else:
            need(BC._same_tree(actual, value), 'FIT_MODEL_EXTRA_STATE')
    optim = state['optimizer']
    blank = _fit_optimizer(model).state_dict()
    need(type(optim) is dict and set(optim) == set(blank)
         and optim['param_groups'] == blank['param_groups'], 'FIT_ADAM_SETTINGS')
    ids = state['optimizer_state_ids']
    need(type(optim['state']) is dict and type(ids) is list
         and all(type(i) is int for i in ids) and ids == sorted(optim['state'])
         and bool(ids) == (update > 0), 'FIT_ADAM_INVENTORY')
    parameters = list(model.parameters())
    for index, moments in optim['state'].items():
        need(type(index) is int and 0 <= index < len(parameters), 'FIT_ADAM_PARAMETER_ID')
        fields(moments, ('step', 'exp_avg', 'exp_avg_sq'), 'FIT_ADAM_MOMENT_FIELDS')
        age = moments['step']
        need(isinstance(age, torch.Tensor) and age.numel() == 1
             and float(age).is_integer() and 0 < float(age) <= update, 'FIT_ADAM_AGE')
        for name in ('exp_avg', 'exp_avg_sq'):
            value = moments[name]
            need(isinstance(value, torch.Tensor) and value.shape == parameters[index].shape
                 and value.dtype == parameters[index].dtype, 'FIT_ADAM_MOMENT_SHAPE_DTYPE')
        need(bool((moments['exp_avg_sq'] >= 0).all()), 'FIT_ADAM_NEGATIVE_VARIANCE')
    fields(state['rng'], ('python', 'numpy', 'torch_cpu', 'torch_cuda'), 'FIT_RNG_FIELDS')
    device = next(model.parameters()).device
    before_rng = BC._rng_state(device)
    try:
        BC._restore_rng(state['rng'], device)
    except (ValueError, TypeError, RuntimeError, KeyError, IndexError) as exc:
        raise SequenceError('FIT_INVALID_RNG') from exc
    finally:
        BC._restore_rng(before_rng, device)


def _reserve_fit_step(output: Path, update: int, identity: dict,
                      parent_sha256: str | None) -> Path:
    from farm_neat.training import v0_bc as BC
    directory = safe_path(output / f'step-{update:08d}')
    need(not directory.exists(), 'FIT_INCOMPLETE_OR_ALREADY_COMMITTED_STEP')
    directory.mkdir(exist_ok=False)
    _publish_fit_json(directory / 'ATTEMPT.json', seal(dict(schema=FIT_SCHEMA + '-attempt',
        global_update=update, lineage_sha256=identity['content_sha256'],
        parent_manifest_sha256=parent_sha256)))
    BC._fsync_dir(output)  # Attempt is durable before the first gradient/Adam call.
    return directory


def _save_fit(model, optimizer, *, output: Path, update: int, registration: dict,
              corpus: dict, admission: dict, metrics: dict | None, identity: dict,
              parent_sha256: str | None) -> dict:
    import torch
    from farm_neat.training import v0_bc as BC
    validate_registration(registration)
    need(identity == _fit_identity(model, registration, corpus, admission,
         identity.get('native_authorization')), 'FIT_LIVE_IDENTITY_DRIFT')
    if identity['scope'] == 'PRODUCTION_NOT_STRENGTH':
        _verify_native_fit_authorization(registration, corpus, admission,
                                         identity['native_authorization'])
    directory = safe_path(output / f'step-{update:08d}')
    need(directory.is_dir() and (directory / 'ATTEMPT.json').is_file(), 'FIT_UNRESERVED_UPDATE')
    state = BC._cpu_tree(dict(schema=FIT_SCHEMA + '-state', global_update=update,
        lineage_sha256=identity['content_sha256'], parent_manifest_sha256=parent_sha256,
        registration_sha256=registration['content_sha256'], corpus_sha256=corpus['content_sha256'],
        admission_sha256=admission['content_sha256'], model_identity=model.get_extra_state(),
        model=model.state_dict(), optimizer=optimizer.state_dict(),
        optimizer_state_ids=sorted(optimizer.state_dict()['state']),
        rng=BC._rng_state(next(model.parameters()).device),
        cursor=dict(sequence_denominator=4, teacher_rows_per_update=116, completed_updates=update),
        parameter_order=[name for name, _ in model.named_parameters()],
        model_training=model.training, runtime=identity['runtime']))
    _validate_fit_state(state, model, identity, update=update, parent_sha256=parent_sha256)
    state_path = safe_path(directory / 'state.pt')
    with state_path.open('xb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    restored = torch.load(state_path, map_location='cpu', weights_only=True)
    need(BC._same_tree(state, restored), 'FIT_CHECKPOINT_ROUNDTRIP')
    _validate_fit_state(restored, model, identity, update=update, parent_sha256=parent_sha256)
    commit = seal(dict(schema=FIT_SCHEMA + '-commit', status='COMMITTED', global_update=update,
        lineage_sha256=identity['content_sha256'], parent_manifest_sha256=parent_sha256,
        state_sha256=file_sha(state_path), state_bytes=state_path.stat().st_size,
        registration_sha256=registration['content_sha256'], corpus_sha256=corpus['content_sha256'],
        admission_sha256=admission['content_sha256'], metrics=copy.deepcopy(metrics),
        scope=identity['scope'], scientific_outcome='NOT_RUN'))
    if identity['scope'] == 'PRODUCTION_NOT_STRENGTH':
        commit = seal({**commit, 'process_identity': _sequence_process_identity()})
    _publish_fit_json(directory / 'COMMIT.json', commit)
    return dict(directory=str(directory), commit_sha256=file_sha(directory / 'COMMIT.json'),
                state_sha256=commit['state_sha256'], metrics=copy.deepcopy(metrics))


def _load_fit_checkpoint(directory: Path, model, identity: dict, *, update: int,
                         parent_sha256: str | None) -> tuple[dict, str]:
    import torch
    directory = safe_path(directory)
    need(directory.is_dir() and not (directory / 'FAILURE.json').exists(),
         'FIT_INCOMPLETE_CHECKPOINT')
    commit = read_json(directory / 'COMMIT.json')
    hashed(commit, ('status', 'global_update', 'lineage_sha256', 'parent_manifest_sha256',
        'state_sha256', 'state_bytes', 'registration_sha256', 'corpus_sha256',
        'admission_sha256', 'metrics', 'scope', 'scientific_outcome')
        + (('process_identity',) if identity['scope'] == 'PRODUCTION_NOT_STRENGTH' else ()),
        FIT_SCHEMA + '-commit', 'FIT_COMMIT')
    need(commit['status'] == 'COMMITTED' and type(commit['global_update']) is int
         and commit['global_update'] == update and commit['scope'] == identity['scope']
         and commit['scientific_outcome'] == 'NOT_RUN', 'FIT_COMMIT_STATUS_AGE_SCOPE')
    if identity['scope'] == 'PRODUCTION_NOT_STRENGTH':
        _validate_sequence_process_identity(commit['process_identity'])
    need(commit['lineage_sha256'] == identity['content_sha256']
         and commit['parent_manifest_sha256'] == parent_sha256
         and all(commit[k] == identity[k] for k in ('registration_sha256',
             'corpus_sha256', 'admission_sha256')), 'FIT_COMMIT_LINEAGE')
    attempt = seal(dict(schema=FIT_SCHEMA + '-attempt', global_update=update,
        lineage_sha256=identity['content_sha256'], parent_manifest_sha256=parent_sha256))
    need(read_json(directory / 'ATTEMPT.json') == attempt, 'FIT_ATTEMPT_CUSTODY')
    metrics = commit['metrics']
    if update == 0:
        need(metrics is None, 'FIT_PARENT_METRICS')
    else:
        fields(metrics, ('loss', 'preclip_gradient_norm', 'parameter_delta_norm',
            'post_update_loss', 'sequence_denominator', 'teacher_rows', 'learning_rate'),
            'FIT_METRIC_FIELDS')
        need(all(type(metrics[k]) in (int, float) and math.isfinite(metrics[k])
                 for k in ('loss', 'preclip_gradient_norm', 'parameter_delta_norm', 'post_update_loss'))
             and metrics['preclip_gradient_norm'] > 0 and metrics['parameter_delta_norm'] > 0
             and metrics['sequence_denominator'] == 4 and metrics['teacher_rows'] == 116
             and metrics['learning_rate'] == 1e-5, 'FIT_METRIC_VALUES')
    path = safe_path(directory / 'state.pt')
    need(type(commit['state_bytes']) is int and commit['state_bytes'] > 0, 'FIT_STATE_SIZE')
    with path.open('rb') as stream:
        need(os.fstat(stream.fileno()).st_size == commit['state_bytes']
             and hashlib.file_digest(stream, 'sha256').hexdigest() == commit['state_sha256'],
             'FIT_STATE_PAYLOAD_MISMATCH')
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=True)
        stream.seek(0)
        need(hashlib.file_digest(stream, 'sha256').hexdigest() == commit['state_sha256'],
             'FIT_STATE_CHANGED_DURING_LOAD')
    _validate_fit_state(state, model, identity, update=update, parent_sha256=parent_sha256)
    return state, file_sha(directory / 'COMMIT.json')


def _resume_fit(model, optimizer, directory: Path, *, registration: dict,
                corpus: dict, admission: dict, output: Path, updates: int,
                native_authorization: dict | None = None) -> tuple[int, dict, str]:
    from farm_neat.training import v0_bc as BC
    directory = safe_path(directory)
    need(directory.parent == output, 'FIT_CROSS_OUTPUT_LINEAGE_REFUSED')
    identity = _fit_identity(model, registration, corpus, admission, native_authorization)
    need(read_json(output / 'LINEAGE.json') == identity, 'FIT_LINEAGE_IDENTITY')
    commit = read_json(directory / 'COMMIT.json')
    age = commit.get('global_update')
    need(type(age) is int and 0 <= age <= 2
         and directory.name == f'step-{age:08d}', 'FIT_RESUME_CURSOR_DIRECTORY')
    need(age + updates <= 2, 'two-update cap exceeded')
    if native_authorization is not None:
        need(age == updates == 1, 'NATIVE_EXACT_UPDATE1_RESUME_REQUIRED')
        need(commit.get('process_identity') != _sequence_process_identity(),
             'NATIVE_UPDATE2_REQUIRES_FRESH_PROCESS')
    expected = {'FIT.lock', 'LINEAGE.json', *(f'step-{i:08d}' for i in range(age + 1))}
    need({p.name for p in output.iterdir()} == expected, 'FIT_INCOMPLETE_OR_FOREIGN_LINEAGE')
    parent = None
    state = None
    for index in range(age + 1):
        state, parent = _load_fit_checkpoint(output / f'step-{index:08d}', model, identity,
                                             update=index, parent_sha256=parent)
    before = _fit_live(model, optimizer)
    try:
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        # Initialization and all preflight probes precede restoration of the RNG.
        model.train(state['model_training'])
        BC._restore_rng(state['rng'], next(model.parameters()).device)
    except BaseException:
        _restore_fit_live(model, optimizer, before)
        raise
    return age, identity, parent


def fit_sequence(model, corpus: dict, registration: dict, admission: dict, *, output: str | Path,
                 updates: int, resume: str | Path | None = None, menu_builder=None) -> dict:
    """Diagnostic transaction; production uses the cold-loader campaign front door."""
    need(registration.get('scope') == 'DIAGNOSTIC_CPU', 'NATIVE_FIT_REQUIRES_AUTHENTICATED_CAMPAIGN')
    validate_sequence_admission(admission, registration)
    need(getattr(model, 'diagnostic', None) is True
         and all(parameter.device.type == 'cpu' for parameter in model.parameters()),
         'DIAGNOSTIC_CPU_MODEL_REQUIRED_UNTIL_NATIVE_BUILD')
    return _fit_sequence_transaction(model, corpus, registration, admission, output=output,
        updates=updates, resume=resume, menu_builder=menu_builder)


def _fit_sequence_transaction(model, corpus: dict, registration: dict, admission: dict, *,
        output: str | Path, updates: int, resume: str | Path | None = None,
        menu_builder=None, native_authorization: dict | None = None) -> dict:
    """Shared non-forking model/Adam/RNG transaction, not a production public API.

    Native callers supply neither a model nor scalars: the campaign authenticates
    all 80 charged traces, derives admission, and cold-loads the registered parent.
    One native update per interpreter is mandatory. Failed attempts remain spent.
    """
    import torch
    from farm_neat.training import v0_bc as BC
    native = registration.get('scope') == 'PRODUCTION_NOT_STRENGTH'
    if native:
        need(menu_builder is None and updates == 1, 'NATIVE_FIXED_ONE_UPDATE_PER_PROCESS')
        _verify_native_fit_authorization(registration, corpus, admission, native_authorization)
    else:
        need(native_authorization is None, 'DIAGNOSTIC_NATIVE_AUTHORIZATION_REFUSED')
    need(admission.get('status') == 'ADMITTED' and admission.get('teacher_gate') == 1,
         'admitted fixed confirmation required')
    validation = validate_sequence_corpus(corpus, registration, menu_builder=menu_builder)
    need(validation['reference_likelihoods_bound'], 'parent reference likelihoods required')
    need(type(updates) is int and 1 <= updates <= 2, 'two-update cap exceeded')
    output = safe_path(output)
    if resume is not None:
        resume = safe_path(resume)
        need(resume.parent == output, 'FIT_CROSS_OUTPUT_LINEAGE_REFUSED')
        need(output.is_dir() and (resume / 'COMMIT.json').is_file(), 'FIT_MISSING_RESUME')
    output.mkdir(parents=True, exist_ok=True)
    with safe_path(output / 'FIT.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SequenceError('FIT_ALREADY_RUNNING') from exc
        optimizer = _fit_optimizer(model)
        if resume is None:
            need({p.name for p in output.iterdir()} == {'FIT.lock'},
                 'FIT_EXISTING_OR_INCOMPLETE_LINEAGE')
            identity = _fit_identity(model, registration, corpus, admission, native_authorization)
            need(BC._finite_tree(model.state_dict()), 'FIT_NONFINITE_PARENT')
            _publish_fit_json(output / 'LINEAGE.json', identity)
            _reserve_fit_step(output, 0, identity, None)
            parent = _save_fit(model, optimizer, output=output, update=0,
                registration=registration, corpus=corpus, admission=admission, metrics=None,
                identity=identity, parent_sha256=None)['commit_sha256']
            age = 0
        else:
            age, identity, parent = _resume_fit(model, optimizer, resume,
                registration=registration, corpus=corpus, admission=admission,
                output=output, updates=updates, native_authorization=native_authorization)
        checkpoints = []
        for _ in range(updates):
            before = _fit_live(model, optimizer)
            directory = _reserve_fit_step(output, age + 1, identity, parent)
            try:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                if native:
                    # One real prefix at a time: same summed objective without
                    # retaining 116 production-model autograd graphs in VRAM.
                    loss = _native_sequence_loss(model, corpus, admission, backward=True)
                else:
                    loss = sequence_loss(model, corpus, registration, gate=1,
                        weight=admission['teacher_weight'], menu_builder=menu_builder)
                    need(bool(torch.isfinite(loss)), 'nonfinite sequence loss')
                    loss.backward()
                squares = [p.grad.detach().double().square().sum() for p in model.parameters()
                           if p.grad is not None]
                grad_norm = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
                if native and grad_norm == 0:
                    raise SequenceError('NATIVE_ZERO_GRADIENT_CLOSED_NEGATIVE')
                need(math.isfinite(grad_norm) and grad_norm > 0,
                     'finite nonzero sequence gradient required')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True,
                                                foreach=False)
                optimizer.step()
                need(BC._finite_tree(model.state_dict()) and BC._finite_tree(optimizer.state_dict()),
                     'FIT_NONFINITE_POST_UPDATE_STATE')
                deltas = [(p.detach().double() - before['model'][name].to(p.device).double()).square().sum()
                          for name, p in model.named_parameters()]
                delta_norm = float(torch.stack(deltas).sum().sqrt())
                need(math.isfinite(delta_norm) and delta_norm > 0,
                     'finite nonzero parameter delta required')
                rng = BC._rng_state(next(model.parameters()).device)
                try:
                    model.eval()
                    with torch.no_grad():
                        post_loss = (_native_sequence_loss(model, corpus, admission, backward=False)
                            if native else sequence_loss(model, corpus, registration, gate=1,
                                weight=admission['teacher_weight'], menu_builder=menu_builder))
                    need(bool(torch.isfinite(post_loss)), 'nonfinite post-update sequence forward')
                    need(BC._same_tree(rng, BC._rng_state(next(model.parameters()).device)),
                         'FIT_POST_UPDATE_CHECK_CONSUMED_RNG')
                finally:
                    model.train()
                    BC._restore_rng(rng, next(model.parameters()).device)
                metrics = dict(loss=float(loss.detach()), preclip_gradient_norm=grad_norm,
                    parameter_delta_norm=delta_norm, post_update_loss=float(post_loss),
                    sequence_denominator=4, teacher_rows=116, learning_rate=1e-5)
                checkpoint = _save_fit(model, optimizer, output=output, update=age + 1,
                    registration=registration, corpus=corpus, admission=admission, metrics=metrics,
                    identity=identity, parent_sha256=parent)
            except BaseException as exc:
                _restore_fit_live(model, optimizer, before)
                failure = seal(dict(schema=FIT_SCHEMA + '-failure', status='PARTIAL',
                    attempted_update=age + 1, last_committed_update=age,
                    parent_manifest_sha256=parent, exception=type(exc).__name__, message=str(exc),
                    automatic_retry=False, scientific_outcome='NOT_RUN'))
                try:
                    _publish_fit_json(directory / 'FAILURE.json', failure)
                except OSError:
                    pass  # Reserved incomplete directory still prevents automatic retry.
                raise
            age += 1
            parent = checkpoint['commit_sha256']
            checkpoints.append(checkpoint)
        return seal(dict(schema=SCHEMA + '-fit-result', registration_sha256=registration['content_sha256'],
            corpus_sha256=corpus['content_sha256'], admission_sha256=admission['content_sha256'],
            lineage_sha256=identity['content_sha256'], global_update=age, checkpoints=checkpoints,
            monitor_ran=False, strength_claim=False))

OBSERVED_SCHEMA = SCHEMA + '-observed-execution-v2'
OBSERVED_EPOCH_SCHEMA = SCHEMA + '-observed-publication-v2'


class SequenceObserver:
    """Read-only evidence around the actual PolicyAgent publication owner.

    No policy wrapper, second decode, reconstructed Plan, menu builder or
    simulator is used. Class-likelihood enrichment belongs after execution;
    this observer never makes a model call. The captured phase is AFTER observe
    and BEFORE epoch publication, not a pre-observe restart snapshot.
    """
    def __init__(self):
        self.epochs: list[dict] = []
        self._attached = False

    def agent(self, policy, **kwargs):
        from farm_neat.evaluation.v0_policy import PolicyAgent, _policy_snapshot
        from farm_neat.evaluation.v0_continuation import V0Continuation
        from farm_neat.economy.v0.projection import State
        need(not self._attached, 'OBSERVER_REQUIRES_FRESH_AGENT')
        self._attached = True
        observer = self

        class ObservedPolicyAgent(PolicyAgent):
            def epoch(agent, observation):
                index = len(observer.epochs)
                need(index < len(DAWNS) and observation['step'] == DAWNS[index],
                     'OBSERVER_PUBLICATION_ORDER')
                state = agent.actor_state(observation)
                before = V0Continuation.capture(agent).to_dict()
                options = _options_wire(agent.plan, agent.projection_kw())
                policy_before = copy.deepcopy(_policy_snapshot(agent.policy))
                actor = public_facts(state)
                original = copy.deepcopy(observation)
                count = len(agent.records)
                super().epoch(observation)
                need(observation == original and len(agent.records) == count + 1,
                     'OBSERVER_PUBLICATION_MUTATED_INPUT_OR_COUNT')
                after = V0Continuation.capture(agent).to_dict()
                observer.epochs.append(dict(schema=OBSERVED_EPOCH_SCHEMA,
                    phase='AFTER_OBSERVE_BEFORE_PUBLICATION', step=observation['step'],
                    observation=original, actor=actor, projection=options, before=before,
                    after=after, policy_before=policy_before,
                    policy_after=copy.deepcopy(_policy_snapshot(agent.policy)),
                    publication=copy.deepcopy(agent.records[-1])))

        return ObservedPolicyAgent(policy, **kwargs)


def _environment_witness(environment) -> dict:
    """Evaluator-only exact state; never used as an actor/scorer feature.

    The selected original engine derives each day's random stream from its
    seed/day, rather than retaining a random.Random object. Full games must be
    reexecuted; this is not a newly invented midgame restoration codec.
    """
    return copy.deepcopy(dict(configuration=environment.configuration, info=environment.info,
        state=environment.state, step_index=environment.step_index, done=environment.done))


def _actor_witness(agent) -> dict:
    from farm_neat.evaluation.v0_continuation import V0Continuation
    from farm_neat.evaluation.v0_policy import _policy_snapshot
    return dict(memory=V0Continuation.capture(agent).to_dict(),
                policy=copy.deepcopy(_policy_snapshot(agent.policy)))


def observe_sequence(registration: dict, launch: dict, *, case: dict, policy,
                     environment, rival, diagnostic: bool, used_policies=()) -> dict:
    """Record one already-created environment via its existing step method.

    Diagnostic-only lower-level producer. The caller owns durable reservation
    and fresh factories. Native preflight, model joins and campaign integration
    are still required; no native authorization is implied. Source fixtures
    supply an observation stream with ZERO native engine invocations. Neither
    teacher inputs nor a starting Plan are accepted.
    """
    import time
    from farm_neat.evaluation.v0_policy import OriginalRuleCase
    from farm_neat.economy.v0 import vocabulary as V
    validate_registration(registration)
    _diagnostic_execution_only(registration)
    need(diagnostic is True, 'OBSERVED_NATIVE_EXECUTION_NOT_AUTHORIZED')
    need(type(launch) is dict and type(launch.get('index')) is int
         and 0 <= launch['index'] < len(registration['launches'])
         and launch == registration['launches'][launch['index']], 'OBSERVED_FOREIGN_LAUNCH')
    expected = OriginalRuleCase(launch['seed'], launch['seat'], launch['opponent']).document()
    need(case == expected, 'OBSERVED_ORIGINAL_CASE_REQUIRED')
    need(callable(policy) and callable(rival) and not any(policy is old for old in used_policies),
         'OBSERVED_FRESH_POLICY_REQUIRED')
    need(environment.step_index == 0 and environment.done is False,
         'OBSERVED_OPENING_ENVIRONMENT_REQUIRED')
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    need(not isinstance(environment, ExactKaggricultureEnvironment),
         'OBSERVED_NATIVE_EXECUTION_NOT_AUTHORIZED')
    return _observe_sequence_body(registration, launch, case=case, policy=policy,
        environment=environment, rival=rival, native=False)


def _observe_sequence_body(registration: dict, launch: dict, *, case: dict, policy,
                           environment, rival, native: bool) -> dict:
    """Shared recording loop; the public front doors own authorization/factories."""
    import time
    from farm_neat.economy.v0 import vocabulary as V
    observer = SequenceObserver()
    agent = observer.agent(policy, labor_factor=4., hands_mode='daily', belief=None,
                           game_id=case['game_id'])
    result = dict(schema=OBSERVED_SCHEMA, registration_sha256=registration['content_sha256'],
        source_closure_sha256=digest(registration['source_pins']), launch=copy.deepcopy(launch),
        case=copy.deepcopy(case), scope='DIAGNOSTIC_CPU', native_games=0,
        actor_instance_sha256=digest(dict(launch_id=launch['launch_id'], instance=id(policy))),
        initial=_environment_witness(environment), initial_actor=_actor_witness(agent),
        transitions=[], epochs=[], terminal=None, terminal_actor=None,
        terminal_observation=None, status='FAILED', wdl=None, error=None,
        scientific_outcome='NOT_RUN')
    if native:
        result.update(schema=NATIVE_OBSERVED_SCHEMA, scope='PRODUCTION_NOT_STRENGTH',
            native_games=1, scientific_outcome='NOT_EVALUATED',
            parent_identity=copy.deepcopy(policy.identity),
            runtime=copy.deepcopy(registration['parent']['runtime']))
    start = time.monotonic()
    try:
        while not environment.done:
            need(time.monotonic() - start < registration['preregistration']['resources']['per_game_timeout_seconds'],
                 'OBSERVED_GAME_DEADLINE')
            step, seat = environment.step_index, launch['seat']
            need(type(step) is int and step == len(result['transitions']) and 0 <= step < 719,
                 'OBSERVED_STEP_ORDER')
            observation = copy.deepcopy(environment.agent_observation(seat))
            other_observation = copy.deepcopy(environment.agent_observation(1 - seat))
            need(observation['step'] == step and observation['player'] == seat
                 and other_observation['step'] == step and other_observation['player'] == 1-seat,
                 'OBSERVED_STEP_OR_SEAT')
            before_environment = digest(_environment_witness(environment))
            before_actor = _actor_witness(agent)
            joint = [None, None]
            joint[seat] = copy.deepcopy(agent(observation, copy.deepcopy(environment.configuration)))
            joint[1-seat] = copy.deepcopy(rival(other_observation, copy.deepcopy(environment.configuration)))
            environment.step(copy.deepcopy(joint))
            need(environment.step_index == step + 1, 'OBSERVED_TRANSITION_DID_NOT_ADVANCE')
            # Capture post-transition own LAND fulfillment, including a purchase
            # on turn 718. The executor remains at its pre-observe boundary.
            agent.actor_state(environment.agent_observation(seat), initialize=False)
            after_actor = _actor_witness(agent)
            result['transitions'].append(dict(step=step, observation_sha256=digest(observation),
                rival_observation_sha256=digest(other_observation), joint=joint,
                before_environment_sha256=before_environment,
                after_environment_sha256=digest(_environment_witness(environment)),
                before_actor_sha256=digest(before_actor), after_actor_sha256=digest(after_actor),
                before_plan_sha256=digest(before_actor['memory']['plan']),
                after_plan_sha256=digest(after_actor['memory']['plan']),
                publication_sha256=(digest(observer.epochs[-1]) if step in DAWNS else None)))
        need(environment.step_index == V.LAST_STEP + 1 == len(result['transitions'])
             and [row['step'] for row in observer.epochs] == list(DAWNS), 'OBSERVED_COMPLETE_HORIZON_REQUIRED')
        if launch['law'] == KEEP_LAW:
            need(all(row['publication']['raw_edits'] == [V.STOP_ID]
                     for row in observer.epochs[1:]), 'OBSERVED_ALL_LATER_STOP_REQUIRED')
        final = copy.deepcopy(environment.agent_observation(launch['seat']))
        cash = [farm['money'] for farm in final['farms']]
        need(len(cash) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in cash),
             'OBSERVED_TERMINAL_CASH')
        difference = cash[launch['seat']] - cash[1-launch['seat']]
        result.update(status='COMPLETE', wdl='WIN' if difference > 0 else 'LOSS' if difference < 0 else 'DRAW')
    except Exception as exc:
        result.update(status='FAILED', wdl=None, error=dict(type=type(exc).__name__, message=str(exc)))
    finally:
        result.update(epochs=copy.deepcopy(observer.epochs), terminal=_environment_witness(environment),
                      terminal_actor=_actor_witness(agent),
                      terminal_observation=copy.deepcopy(environment.agent_observation(launch['seat'])))
    return seal(result)


# Content-addressed *structural* validation only. Source and full closed-schema
# hashes are checked afresh before every hit. Never cache native execution,
# parent loading, acceptance, file authentication, or scientific outcomes.
_OBSERVED_VALIDATION_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_OBSERVED_VALIDATION_CACHE_LIMIT = 256


def validate_observed_execution(record: dict, registration: dict) -> dict:
    """Validate a diagnostic observation; native records use their own schema."""
    return _validate_observed_execution(record, registration, native=False)


def _validate_observed_execution(record: dict, registration: dict, *, native: bool) -> dict:
    """Authenticate observed public context/custody against the ordered trace.

    Self-consistent JSON is not proof of native execution. The two schemas stay
    separate; independent native reexecution is a separate gate, and production
    data admission remains disabled until its training joins are implemented.
    """
    from farm_neat.economy.v0.plan import Plan
    from farm_neat.economy.v0.epochs import KeepPolicy
    from farm_neat.economy.v0.projection import State
    from farm_neat.evaluation.v0_policy import OriginalRuleCase, validate_edits, _validate_memory
    from farm_neat.evaluation.v0_continuation import V0Continuation
    validate_registration(registration)
    if not native:
        _diagnostic_execution_only(registration)
    names = ('registration_sha256', 'source_closure_sha256', 'launch', 'case', 'scope',
             'native_games', 'actor_instance_sha256', 'initial', 'initial_actor', 'transitions',
             'epochs', 'terminal', 'terminal_actor', 'terminal_observation', 'status', 'wdl',
             'error', 'scientific_outcome')
    if native:
        names += ('parent_identity', 'runtime')
        if record.get('launch', {}).get('phase') == 'monitor':
            names += ('monitor_freeze',)
    observed_schema = (NATIVE_MONITOR_OBSERVED_SCHEMA if native
        and record.get('launch', {}).get('phase') == 'monitor' else NATIVE_OBSERVED_SCHEMA)
    hashed(record, names, observed_schema if native else OBSERVED_SCHEMA, 'OBSERVED_EXECUTION')
    cache_key = (native, registration['content_sha256'], record['content_sha256'])
    cached = _OBSERVED_VALIDATION_CACHE.get(cache_key)
    if cached is not None:
        _OBSERVED_VALIDATION_CACHE.move_to_end(cache_key)
        return copy.deepcopy(cached)
    need(record['registration_sha256'] == registration['content_sha256']
         and record['source_closure_sha256'] == digest(registration['source_pins']),
         'OBSERVED_STALE_SOURCE_OR_REGISTRATION')
    need(record['scope'] == ('PRODUCTION_NOT_STRENGTH' if native else 'DIAGNOSTIC_CPU')
         and type(record['native_games']) is int and record['native_games'] == int(native)
         and record['scientific_outcome'] == ('NOT_EVALUATED' if native else 'NOT_RUN'),
         'OBSERVED_NATIVE_SCOPE_REQUIRED' if native else 'OBSERVED_DIAGNOSTIC_SCOPE_REQUIRED')
    launch = record['launch']
    need(type(launch) is dict and type(launch.get('index')) is int
         and 0 <= launch['index'] < len(registration['launches'])
         and launch == registration['launches'][launch['index']], 'OBSERVED_FOREIGN_LAUNCH')
    need(record['case'] == OriginalRuleCase(launch['seed'], launch['seat'], launch['opponent']).document(),
         'OBSERVED_ORIGINAL_CASE_REQUIRED')
    _hash(record['actor_instance_sha256'], 'observed actor')
    need(record['status'] == 'COMPLETE' and record['error'] is None and record['wdl'] in WDL,
         'OBSERVED_COMPLETE_EXECUTION_REQUIRED')
    rows, epochs = record['transitions'], record['epochs']
    need(type(rows) is list and len(rows) == 719 and type(epochs) is list
         and len(epochs) == 30, 'OBSERVED_COMPLETE_HORIZON_REQUIRED')
    for name, step, done in (('initial', 0, False), ('terminal', 719, True)):
        value = record[name]
        fields(value, ('configuration', 'info', 'state', 'step_index', 'done'), 'OBSERVED_ENVIRONMENT_FIELDS')
        need(value['step_index'] == step and value['done'] is done, 'OBSERVED_ENVIRONMENT_CLOCK')
    need(record['initial']['configuration'] == record['terminal']['configuration']
         and record['initial']['info'] == record['terminal']['info'], 'OBSERVED_ENVIRONMENT_IDENTITY_CHANGED')
    for name in ('initial_actor', 'terminal_actor'):
        fields(record[name], ('memory', 'policy'), 'OBSERVED_ACTOR_FIELDS')
        memory = record[name]['memory']
        observed = (epochs[0]['observation'] if name == 'initial_actor'
                    else record['terminal_observation'])
        restored = V0Continuation.from_dict(memory).restore(KeepPolicy(), observation=observed)
        _validate_memory(restored)
        need(memory['game_id'] == record['case']['game_id'], 'OBSERVED_ACTOR_GAME_ID')
    need(record['initial_actor']['memory']['plan'] == Plan().canonical()
         and record['initial_actor']['memory']['controller'] is None, 'OBSERVED_TEACHER_INJECTED_OPENING')
    epoch_map = {}
    for step, row in zip(DAWNS, epochs):
        fields(row, ('schema', 'phase', 'step', 'observation', 'actor', 'projection', 'before',
                     'after', 'policy_before', 'policy_after', 'publication'), 'OBSERVED_EPOCH_FIELDS')
        need(row['schema'] == OBSERVED_EPOCH_SCHEMA and row['step'] == step
             and row['phase'] == 'AFTER_OBSERVE_BEFORE_PUBLICATION', 'OBSERVED_EPOCH_PHASE')
        before, after = row['before'], row['after']
        before_agent = V0Continuation.from_dict(before).restore(KeepPolicy(), observation=row['observation'])
        state = before_agent.actor_state(row['observation'], initialize=False)
        need(row['actor'] == public_facts(state)
             and row['actor']['step'] == step and row['actor']['player'] == launch['seat'],
             'OBSERVED_ACTOR_CONTEXT')
        for memory in (before, after):
            restored = V0Continuation.from_dict(memory).restore(KeepPolicy(), observation=row['observation'])
            _validate_memory(restored)
            need(memory['game_id'] == record['case']['game_id']
                 and memory['labor_factor'] == 4. and memory['hands_mode'] == 'daily',
                 'OBSERVED_COMPILER_IDENTITY')
        carried = Plan.from_canonical(before['plan'])
        options = _options_live(carried, row['projection'])
        need(before['hold_releases'] == row['projection']['hold_releases'], 'OBSERVED_HOLD_CONTEXT')
        pub = row['publication']
        need(type(pub) is dict and pub.get('step') == step and pub.get('before_plan') == before['plan']
             and pub.get('after_plan') == after['plan'], 'OBSERVED_PUBLICATION_PLAN')
        decision = validate_edits(state, carried, pub['raw_edits'], projection_kwargs=options)
        need(decision.trace == pub['prefix_trace'] and decision.plan.canonical() == after['plan'],
             'OBSERVED_PREFIX_CUSTODY')
        need(type(pub['compiled_program']) is list and pub['commands'] == len(pub['compiled_program'])
             and type(pub['refusals']) is list and pub['rejected'] == len(pub['refusals']),
             'OBSERVED_COMPILED_PUBLICATION')
        need(after['controller']['committed_step'] == step
             and after['controller']['publication_steps'] == list(DAWNS), 'OBSERVED_SKIPPED_PUBLICATION')
        if step and launch['law'] == KEEP_LAW:
            need(pub['raw_edits'] == [0] and before['plan'] == after['plan']
                 and before['hold_releases'] == after['hold_releases'], 'OBSERVED_ALL_LATER_STOP_REQUIRED')
        epoch_map[step] = row
    previous_env, previous_actor = digest(record['initial']), digest(record['initial_actor'])
    previous_plan = digest(record['initial_actor']['memory']['plan'])
    for step, row in enumerate(rows):
        fields(row, ('step', 'observation_sha256', 'rival_observation_sha256', 'joint',
                     'before_environment_sha256', 'after_environment_sha256', 'before_actor_sha256',
                     'after_actor_sha256', 'before_plan_sha256', 'after_plan_sha256',
                     'publication_sha256'), 'OBSERVED_TRANSITION_FIELDS')
        need(row['step'] == step and row['before_environment_sha256'] == previous_env
             and row['before_actor_sha256'] == previous_actor
             and row['before_plan_sha256'] == previous_plan, 'OBSERVED_TRANSITION_CHAIN')
        need(type(row['joint']) is list and len(row['joint']) == 2
             and all(type(action) is dict for action in row['joint']), 'OBSERVED_JOINT_ACTIONS')
        if step in epoch_map:
            epoch = epoch_map[step]
            need(row['publication_sha256'] == digest(epoch)
                 and row['observation_sha256'] == digest(epoch['observation'])
                 and row['before_plan_sha256'] == digest(epoch['before']['plan'])
                 and row['after_plan_sha256'] == digest(epoch['after']['plan']),
                 'OBSERVED_PUBLICATION_TRACE_JOIN')
        else:
            need(row['publication_sha256'] is None and row['before_plan_sha256'] == row['after_plan_sha256'],
                 'OBSERVED_INTRADAY_PLAN_CHANGE')
        for name, value in row.items():
            if name.endswith('_sha256') and value is not None:
                _hash(value, name)
        previous_env, previous_actor, previous_plan = (row['after_environment_sha256'],
            row['after_actor_sha256'], row['after_plan_sha256'])
    need(previous_env == digest(record['terminal']) and previous_actor == digest(record['terminal_actor'])
         and previous_plan == digest(record['terminal_actor']['memory']['plan']), 'OBSERVED_TERMINAL_JOIN')
    final = record['terminal_observation']
    need(final['step'] == 719 and final['player'] == launch['seat'], 'OBSERVED_TERMINAL_OBSERVATION')
    cash = [farm['money'] for farm in final['farms']]
    need(len(cash) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in cash),
         'OBSERVED_TERMINAL_CASH')
    delta = cash[launch['seat']] - cash[1-launch['seat']]
    need(record['wdl'] == ('WIN' if delta > 0 else 'LOSS' if delta < 0 else 'DRAW'), 'OBSERVED_TERMINAL_WDL')
    result = seal(dict(schema=OBSERVED_SCHEMA + '-validation', record_sha256=record['content_sha256'],
        steps_checked=719, epochs_checked=30, native_replay_verified=False,
        scientific_outcome='NOT_EVALUATED' if native else 'NOT_RUN',
        scope='NATIVE_STRUCTURAL_ONLY' if native else 'DIAGNOSTIC_STRUCTURAL_ONLY'))
    _OBSERVED_VALIDATION_CACHE[cache_key] = copy.deepcopy(result)
    while len(_OBSERVED_VALIDATION_CACHE) > _OBSERVED_VALIDATION_CACHE_LIMIT:
        _OBSERVED_VALIDATION_CACHE.popitem(last=False)
    return result


def validate_observed_pair(primary: dict, replay: dict, registration: dict) -> dict:
    """Compare joint actions, observations, publications and exact custody.

    Equality validation is not native reexecution. Callers must retain the
    independent launch journal/factory evidence as well.
    """
    for row in (primary, replay):
        validate_observed_execution(row, registration)
    p, r = primary['launch'], replay['launch']
    need(p['replica'] == 'primary' and r['replica'] == 'replay'
         and r['primary_index'] == p['index'], 'OBSERVED_PRIMARY_REPLAY_JOIN')
    need(primary['actor_instance_sha256'] != replay['actor_instance_sha256'], 'OBSERVED_FRESH_REPLAY_REQUIRED')
    for key in primary:
        if key not in ('launch', 'actor_instance_sha256', 'content_sha256'):
            need(primary[key] == replay[key], 'OBSERVED_EXACT_REPLAY_MISMATCH:' + key)
    return seal(dict(schema=OBSERVED_SCHEMA + '-replay-validation', exact_replay=True,
        primary_sha256=primary['content_sha256'], replay_sha256=replay['content_sha256'],
        steps_checked=719, epochs_checked=30, native_replay_verified=False,
        scientific_outcome='NOT_RUN', scope='DIAGNOSTIC_SOURCE_ONLY'))


def _index_observed_discovery(records: list[dict], registration: dict, *, native: bool = False) -> list[dict]:
    """Validate all 16 discovery observations before extracting four teachers.

    Controls and replays are custody evidence, never extra supervision. The
    first day must be identical across the frozen policy pair. An external
    diagnostic policy identity still does NOT authenticate parent tensors.
    """
    validate_registration(registration)
    if native:
        need(registration['scope'] == 'PRODUCTION_NOT_STRENGTH', 'NATIVE_CORPUS_REGISTRATION_REQUIRED')
    else:
        _diagnostic_execution_only(registration)
    need(type(records) is list and len(records) == 16, 'DISCOVERY_EXECUTION_COMPLETE_SET_REQUIRED')
    expected = {row['launch_id']: row for row in registration['launches'][:16]}
    found, actors = {}, set()
    for row in records:
        need(type(row) is dict and type(row.get('launch')) is dict, 'DISCOVERY_EXECUTION_FIELDS')
        launch = row['launch']
        key = launch.get('launch_id')
        need(type(key) is str and key in expected and key not in found and launch == expected[key],
             'DISCOVERY_EXECUTION_FOREIGN_OR_DUPLICATE')
        actor = row.get('actor_instance_sha256')
        _hash(actor, 'discovery actor')
        need(actor not in actors, 'DISCOVERY_EXECUTION_FRESH_ACTORS_REQUIRED')
        actors.add(actor)
        found[key] = row
    return [found[job['launch_id']] for job in registration['launches'][:16]]


def _observed_discovery(records: list[dict], registration: dict, *, native: bool = False) -> list[dict]:
    """Fully validate discovery replays and the shared opening before labels."""
    ordered = _index_observed_discovery(records, registration, native=native)
    for index in range(0, 16, 2):
        if native:
            primary, replay = ordered[index:index + 2]
            for row in (primary, replay):
                validate_native_execution(row, registration)
            need(not primary['launch']['replay'] and replay['launch']['replay']
                 and replay['launch']['primary_index'] == primary['launch']['index']
                 and primary['actor_instance_sha256'] != replay['actor_instance_sha256'],
                 'NATIVE_CORPUS_FRESH_REPLAY_JOIN')
            for key in primary:
                if key not in ('launch', 'actor_instance_sha256', 'content_sha256'):
                    need(primary[key] == replay[key], 'NATIVE_CORPUS_EXACT_REPLAY_MISMATCH:' + key)
        else:
            validate_observed_pair(ordered[index], ordered[index + 1], registration)
    for index in range(0, 16, 4):
        control, teacher = ordered[index], ordered[index + 2]
        # Model publication adapters legitimately have different snapshots.
        # Compare actual public inputs/Plan/compiler/controller and emitted
        # actions; never require their implementation-specific RNG envelopes
        # to be identical across different policy laws.
        need(control['initial'] == teacher['initial'], 'DISCOVERY_POLICY_PAIR_INITIAL_MISMATCH')
        for key in ('actor', 'observation', 'projection', 'before', 'after'):
            need(control['epochs'][0][key] == teacher['epochs'][0][key],
                 'DISCOVERY_POLICY_PAIR_OPENING_MISMATCH:' + key)
        need(control['epochs'][0]['publication'] == teacher['epochs'][0]['publication'],
             'DISCOVERY_POLICY_PAIR_OPENING_MISMATCH:publication')
        for left, right in zip(control['transitions'][:24], teacher['transitions'][:24]):
            for key in ('step', 'observation_sha256', 'rival_observation_sha256', 'joint',
                        'before_environment_sha256', 'after_environment_sha256',
                        'before_plan_sha256', 'after_plan_sha256'):
                need(left[key] == right[key], 'DISCOVERY_POLICY_PAIR_DAY_ONE_MISMATCH:' + key)
    return [row for row in ordered if row['launch']['arm'] == 'challenger'
            and not row['launch']['replay']]


def _observed_teacher_identity(execution: dict) -> dict:
    job = execution['launch']
    return dict(schema=TEACHER_SCHEMA, sequence_id=job['launch_id'], family_id=job['family_id'],
        seed=job['seed'], seat=job['seat'], arm='challenger', replay=False,
        opening_trace_sha256=digest(execution['epochs'][0]),
        game_receipt_sha256=execution['content_sha256'])


def _observed_teacher_context(epoch: dict) -> tuple[dict, dict]:
    from farm_neat.economy.v0.plan import Plan
    from farm_neat.policy.v0_scorer import Prefix
    before, after = epoch['before'], epoch['after']
    plan = Plan.from_canonical(before['plan'])
    options = _options_live(plan, epoch['projection'])
    projection = dict(labor_factor=options['labor_factor'], belief=copy.deepcopy(options['belief']),
        hold_releases=[[product, *overlay, release]
                       for product, (overlay, release) in options['hold_releases'].items()])
    actor = dict(state=copy.deepcopy(epoch['actor']), prefix=Prefix(plan).payload(), projection=projection)
    custody = dict(before_plan_sha256=plan.digest(),
        after_plan_sha256=Plan.from_canonical(after['plan']).digest(),
        hold_releases=copy.deepcopy(projection['hold_releases']),
        allocation_sha256=digest(after['allocation']), controller_sha256=digest(after['controller']))
    return actor, custody


def _validate_teacher_execution_join(teachers: list[dict], actual: list[dict]) -> None:
    need(type(teachers) is list and len(teachers) == 4, 'OBSERVED_TEACHER_DENOMINATOR_FOUR')
    for teacher, execution in zip(teachers, actual):
        identity = _observed_teacher_identity(execution)
        need(type(teacher) is dict and set(teacher) == set(identity) | {'epochs'}
             and all(teacher[key] == value for key, value in identity.items()),
             'OBSERVED_TEACHER_EXECUTION_JOIN')
        rows = teacher['epochs']
        need(type(rows) is list and len(rows) == 29, 'OBSERVED_TEACHER_DAWN_COUNT')
        for row, epoch in zip(rows, execution['epochs'][1:]):
            actor, custody = _observed_teacher_context(epoch)
            need(type(row) is dict and row.get('step') == epoch['step']
                 and row.get('actor') == actor and row.get('custody') == custody,
                 'OBSERVED_TEACHER_CONTEXT_OR_CUSTODY_SUBSTITUTION')


def _validate_observed_corpus_join(corpus: dict, registration: dict) -> None:
    ordered = _index_observed_discovery(corpus['execution_evidence'], registration)
    actual = [row for row in ordered if row['launch']['arm'] == 'challenger'
              and not row['launch']['replay']]
    _validate_teacher_execution_join(corpus['teacher_sequences'], actual)
    # Reject cheap field substitutions first, but never admit a corpus without
    # authenticating every full execution and independent replay.
    _observed_discovery(corpus['execution_evidence'], registration)


def _teacher_sequences(actual: list[dict], builder) -> list[dict]:
    """One shared STOP target builder for observed diagnostic and native records."""
    from farm_neat.economy.v0 import vocabulary as V
    from farm_neat.policy import v0_scorer as A
    teachers = []
    for execution in actual:
        rows = []
        for epoch in execution['epochs'][1:]:
            actor, custody = _observed_teacher_context(epoch)
            state, prefix = _state(actor['state']), _prefix(actor['prefix'])
            menu = builder(state, prefix, projection_kwargs=_projection(actor['projection']))
            stop_index = menu.legal_ids.index(V.STOP_ID)
            cls = next(item for item in menu.classes if stop_index in item.indices)
            row = dict(schema=EPOCH_SCHEMA, step=epoch['step'], actor=actor, custody=custody,
                target=dict(raw_id=V.STOP_ID, support=list(menu.legal_ids), phi_sha256=A._digest(menu.phi),
                    class_id=cls.class_id, class_members=[menu.legal_ids[i] for i in cls.indices],
                    continuation=cls.successor.payload(), forced_stop=False, termination='STOP',
                    behavior_logp=None, reference_class_logp=None))
            _validate_epoch(row, epoch['step'], menu_builder=builder, allow_empty=True)
            rows.append(row)
        teachers.append({**_observed_teacher_identity(execution), 'epochs': rows})
    return teachers


def corpus_from_observed(records: list[dict], registration: dict, *, menu_builder=None) -> dict:
    """Build v2 teachers ONLY from independently replayed discovery executions.

    Rebuild each first-choice STOP class on its observed carried Plan. Full
    observation evidence remains trainer-only. No model or checkpoint is
    asserted by this producer: the existing likelihood binder is a separate
    step, and production admission is deliberately still refused.
    """
    from farm_neat.policy import v0_menu_fast
    from farm_neat.economy.v0 import vocabulary as V
    from farm_neat.policy import v0_scorer as A
    actual = _observed_discovery(records, registration)
    builder = v0_menu_fast.build_menu if menu_builder is None else menu_builder
    teachers = _teacher_sequences(actual, builder)
    return seal(dict(schema=OBSERVED_CORPUS_SCHEMA, registration_sha256=registration['content_sha256'],
        parent_commit_sha256=registration['parent']['commit_sha256'],
        parent_state_sha256=registration['parent']['state_sha256'],
        source_closure_sha256=digest(registration['source_pins']), split='discovery',
        teacher_sequences=teachers, execution_evidence=copy.deepcopy(records)))


def execution_from_observed(record: dict, registration: dict) -> dict:
    """Project a complete observed fixture into the existing journal schema.

    The caller MUST retain ``record`` as full evidence; this summary does not
    replace it. This does not reserve a launch, create a policy or execute a
    native game. The existing collector remains the sole journal owner.
    """
    validate_observed_execution(record, registration)
    joint = [row['joint'] for row in record['transitions']]
    summary = seal(dict(schema=EXECUTION_SCHEMA,
        registration_sha256=registration['content_sha256'], launch=copy.deepcopy(record['launch']),
        status='COMPLETE', physical_games_started=1, physical_games_completed=1,
        physical_steps=len(joint), wdl=record['wdl'],
        own_actions_sha256=digest([row[record['launch']['seat']] for row in joint]),
        joint_actions_sha256=digest(joint), terminal_environment_sha256=digest(record['terminal']),
        epochs_sha256=digest(record['epochs']), actor_instance_sha256=record['actor_instance_sha256'],
        source_closure_sha256=record['source_closure_sha256'],
        runtime_sha256=execution_runtime_sha256(registration),
        initial_state_sha256=digest(record['initial']), error=None))
    _validate_execution(summary, registration, record['launch'])
    return summary


MONITOR_FREEZE_SCHEMA = SCHEMA + '-diagnostic-monitor-freeze'


def _monitor_fit_assets(registration: dict, development: dict, fit_output: Path, model
                        ) -> tuple[dict, dict, dict]:
    """Read and authenticate the existing three-commit diagnostic lineage.

    The caller holds FIT.lock. The template is only a shape/runtime validator;
    its parameters, mode, gradients and application RNG are not modified. The
    local step-0 tensors, NOT the omitted registered CUDA parent, are monitored.
    """
    from farm_neat.training import v0_bc as BC
    need(getattr(model, 'diagnostic', None) is True
         and all(p.device.type == 'cpu' for p in model.parameters()),
         'MONITOR_DIAGNOSTIC_CPU_MODEL_REQUIRED')
    need(development is not None and development['status'] == 'READY_TO_FIT'
         and development['counts']['physical_starts'] == 80,
         'MONITOR_COMPLETE_ADMITTED_DEVELOPMENT_REQUIRED')
    admission = development['admission']
    validate_sequence_admission(admission, registration)
    identity = read_json(fit_output / 'LINEAGE.json')
    # The lineage binds the already-validated training corpus; monitor never
    # opens that corpus, scores a teacher prefix, or receives teacher examples.
    _hash(identity.get('corpus_sha256'), 'monitor corpus identity')
    need(identity == _fit_identity(model, registration,
        {'content_sha256': identity['corpus_sha256']}, admission), 'MONITOR_FIT_LINEAGE_JOIN')
    expected = {'FIT.lock', 'LINEAGE.json', *(f'step-{i:08d}' for i in range(3))}
    need({p.name for p in fit_output.iterdir()} == expected,
         'MONITOR_EXACT_UPDATE2_LINEAGE_REQUIRED')
    parent, loaded, pins, commits = None, {}, {}, {}
    for index in range(3):
        directory = fit_output / f'step-{index:08d}'
        state, parent = _load_fit_checkpoint(directory, model, identity,
                                            update=index, parent_sha256=parent)
        if index != 1:
            loaded[index] = state['model']
        commits[index] = dict(directory=str(directory), global_update=index,
            commit_sha256=parent, state_sha256=file_sha(directory / 'state.pt'))
        for name in ('ATTEMPT.json', 'COMMIT.json', 'state.pt'):
            path = safe_path(directory / name)
            pins[str(path)] = file_sha(path)
    pins[str(fit_output / 'LINEAGE.json')] = file_sha(fit_output / 'LINEAGE.json')
    need(not BC._same_tree(loaded[0], loaded[2]), 'MONITOR_CHILD_HAS_NO_PARAMETER_DELTA')
    frozen = seal(dict(schema=MONITOR_FREEZE_SCHEMA, scope='DIAGNOSTIC_CPU',
        registration_sha256=registration['content_sha256'],
        source_closure_sha256=digest(registration['source_pins']),
        development_journal_head_sha256=development['journal_head_sha256'],
        admission_sha256=admission['content_sha256'],
        corpus_sha256=identity['corpus_sha256'], lineage_sha256=identity['content_sha256'],
        fit_output=str(fit_output), parent=commits[0], child=commits[2], asset_pins=pins,
        registered_parent=copy.deepcopy(registration['parent']),
        production_parent_authenticated=False, teacher_input=False,
        policy_law='UNRESTRICTED', scientific_outcome='NOT_RUN'))
    return frozen, loaded[0], loaded[2]


def _monitor_snapshot(registration: dict, journal: SequenceJournal, frozen: dict) -> dict | None:
    """Derive a monitor result solely from the one durable charged journal."""
    events, records, counts = _journal_snapshot(journal)
    monitor_records = [r for r in records if r['launch']['phase'] == 'monitor']
    result, status, reason = None, None, None
    try:
        by_index = {row['launch']['index']: row for row in monitor_records}
        for replay in monitor_records:
            if replay['launch']['replay']:
                primary = by_index.get(replay['launch']['primary_index'])
                need(primary is not None, 'MONITOR_PRIMARY_REPLAY_JOIN')
                _validate_replay(primary, replay)
        if counts['failed']:
            status, reason = 'PARTIAL', 'DURABLE_FAILED_LAUNCH_NO_AUTOMATIC_RETRY'
        elif counts['unknown']:
            status, reason = 'PARTIAL', 'UNKNOWN_DURABLE_LAUNCH_NO_AUTOMATIC_RETRY'
        elif len(monitor_records) == 128:
            need(counts['physical_starts'] == counts['completed'] == 208, 'MONITOR_START_ACCOUNTING')
            result = monitor_sequence(monitor_records, registration,
                parent_checkpoint_sha256=frozen['parent']['commit_sha256'],
                child_checkpoint_sha256=frozen['child']['commit_sha256'])
            status, reason = result['status'], 'FROZEN_MONITOR_COMPLETE'
    except SequenceError as exc:
        status, reason = 'PARTIAL', 'MONITOR_REPLAY_VALIDATION_FAILED: ' + str(exc)
    if status is None:
        return None
    phase_starts = {phase: sum(row['status'] == 'STARTED'
        and registration['launches'][row['launch_index']]['phase'] == phase for row in events)
        for phase in SPLITS}
    return seal(dict(schema=SCHEMA + '-monitor-collection-result',
        registration_sha256=registration['content_sha256'], freeze_sha256=frozen['content_sha256'],
        scope=registration['scope'], status=status, reason=reason, counts=counts,
        phase_starts=phase_starts, monitor=result,
        journal_head_sha256=events[-1]['content_sha256'], scientific_outcome=(
            'NOT_RUN' if registration['scope'] == 'DIAGNOSTIC_CPU' else
            status if status in ('PASS', 'CLOSED_NEGATIVE') else 'PARTIAL'),
        production_parent_authenticated=frozen['production_parent_authenticated'], strength_claim=False))


def collect_sequence_monitor(registration: dict, *, runner, output: str | Path,
                             fit_output: str | Path, model) -> dict:
    """Finish the 128 source-only monitor callbacks after the same 80-start journal.

    ``runner(launch, policy)`` receives a fresh unrestricted ScorerPolicy backed
    by the authenticated immutable local step-0 or update-2 tensors. It receives
    no teacher Plan, corpus, target, optimizer, confirmation state or gate. The
    caller supplies a native-free runner: this API DOES NOT authorize engine
    games. Native production collection is deliberately still refused elsewhere.

    Both directory locks cover the whole call. The freeze is durably published
    before the first monitor reservation, cannot be replaced on resume, and all
    exceptions/failures/unknown starts retain their original budget charge.
    """
    from farm_neat.evaluation.v0_policy import CommittedScorer, ScorerPolicy
    from farm_neat.training import v0_bc as BC
    validate_registration(registration)
    _diagnostic_execution_only(registration)
    need(callable(runner), 'MONITOR_RUNNER_REQUIRED')
    supplied_registration = registration
    registration = copy.deepcopy(registration)
    output, fit_output = safe_path(output), safe_path(fit_output)
    need(output.is_dir() and fit_output.is_dir(), 'MONITOR_EXISTING_COLLECTION_AND_FIT_REQUIRED')
    # Do not create a new output or a new fit lineage as a fallback.
    need(read_json(output / 'REGISTRATION.json') == registration, 'MONITOR_REGISTRATION_JOIN')
    with safe_path(output / 'COLLECTION.lock').open('a+') as lock, \
            safe_path(fit_output / 'FIT.lock').open('r') as fit_lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fit_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SequenceError('MONITOR_COLLECTION_OR_FIT_ALREADY_RUNNING') from exc
        journal = SequenceJournal(output / 'JOURNAL.jsonl', registration)
        development = _development_snapshot(registration, journal, stop_before=80)
        need(development is not None and read_json(output / 'RESULT.json') == development,
             'MONITOR_DEVELOPMENT_JOURNAL_JOIN')
        frozen, parent_state, child_state = _monitor_fit_assets(registration, development,
                                                                fit_output, model)
        freeze_path = safe_path(output / 'MONITOR-FREEZE.json')
        if freeze_path.exists():
            need(read_json(freeze_path) == frozen, 'MONITOR_FREEZE_CHANGED')
        else:
            need(journal.starts == 80, 'MONITOR_FREEZE_REQUIRED_BEFORE_FIRST_START')
            _publish_fit_json(freeze_path, frozen)
        result_path = safe_path(output / 'MONITOR-RESULT.json')
        snapshot = _monitor_snapshot(registration, journal, frozen)
        if result_path.exists():
            need(snapshot is not None and read_json(result_path) == snapshot,
                 'MONITOR_CACHED_RESULT_JOURNAL_MISMATCH')
            return snapshot
        if snapshot is not None:
            _publish_fit_json(result_path, snapshot)
            return snapshot
        # No initialization or restore of the application's model/RNG. Inference
        # copies preserve the committed extra-state and parameter identities.
        factories, states = {}, {'control': parent_state, 'challenger': child_state}
        for arm, state in states.items():
            actor_model = copy.deepcopy(model)
            actor_model.load_state_dict(state, strict=True)
            actor_model.eval().requires_grad_(False)
            checkpoint = frozen['parent' if arm == 'control' else 'child']
            identity = dict(kind='SEQUENCE_DIAGNOSTIC', diagnostic=True,
                commit_sha256=checkpoint['commit_sha256'], state_sha256=checkpoint['state_sha256'],
                model_identity=copy.deepcopy(actor_model.get_extra_state()))
            factories[arm] = CommittedScorer(actor_model, identity)
        # Retaining sampler objects prevents identity/address reuse from being
        # mistaken for fresh instance custody in diagnostic callback records.
        policies = []
        while journal.starts < 208:
            validate_registration(registration)
            for path, sha in frozen['asset_pins'].items():
                need(file_sha(safe_path(path)) == sha, 'MONITOR_FROZEN_ASSET_DRIFT')
            job = registration['launches'][journal.starts]
            need(job['phase'] == 'monitor' and job['law'] == 'UNRESTRICTED', 'MONITOR_LAW')
            journal.reserve(job)
            returned = None
            try:
                factory = factories[job['arm']]
                # A replay resets the sampler to the primary's seed, not its own
                # launch index. This seed is sampler custody, never a world feature.
                policy = factory.policy('greedy_class', job['primary_index'])
                need(type(policy) is ScorerPolicy and all(policy is not old for old in policies),
                     'MONITOR_FRESH_UNRESTRICTED_POLICY_REQUIRED')
                policies.append(policy)
                identity = copy.deepcopy(policy.identity)
                returned = runner(copy.deepcopy(job), policy)
                need(supplied_registration == registration, 'MONITOR_REGISTRATION_MUTATED')
                validate_registration(registration)
                need(policy.identity == identity and policy.mode == 'greedy_class'
                     and policy.seed == job['primary_index'] and policy.temperature == 1.
                     and policy.scorer is factory.scorer and not policy.scorer.training
                     and all(not p.requires_grad for p in policy.scorer.parameters()),
                     'MONITOR_POLICY_IDENTITY_OR_LAW_CHANGED')
                for arm, source in factories.items():
                    need(BC._same_tree(source.scorer.state_dict(), states[arm]),
                         'MONITOR_FROZEN_MODEL_MUTATED')
                for path, sha in frozen['asset_pins'].items():
                    need(file_sha(safe_path(path)) == sha, 'MONITOR_FROZEN_ASSET_DRIFT')
                _validate_execution(returned, registration, job)
                if returned['status'] == 'FAILED':
                    journal.finish(job, error=wire(dict(code='RUNNER_REPORTED_FAILED', execution=returned)))
                else:
                    journal.finish(job, result=returned)
            except Exception as exc:
                error = dict(code='MONITOR_RUNNER_EXCEPTION_OR_INVALID_RECORD',
                             exception=type(exc).__name__, message=str(exc))
                try:
                    wire(returned)
                    error['returned_record'] = returned
                except (TypeError, ValueError, OverflowError):
                    error['returned_record_type'] = type(returned).__name__
                journal.finish(job, error=wire(error))
            snapshot = _monitor_snapshot(registration, journal, frozen)
            if snapshot is not None:
                _publish_fit_json(result_path, snapshot)
                return snapshot
        raise AssertionError('complete monitor must produce a fixed-stage result')


# Native production records are deliberately NOT diagnostic observed-v1 records.
NATIVE_OBSERVED_SCHEMA = SCHEMA + '-native-observed-execution-v2'
NATIVE_EXECUTION_SCHEMA = SCHEMA + '-native-execution-summary-v1'


def inspect_sequence_parent(registration: dict, *, require_state: bool = True) -> dict:
    """Authenticate the *registered* parent header; never select a fallback.

    Header-only inspection is portable and allocates no model. It explicitly
    does not authenticate/load omitted tensor bytes or authorize a campaign.
    Actual loading remains owned by v0_policy.load_committed_scorer.
    """
    from farm_neat.training import v0_bc as BC
    validate_registration(registration)
    need(registration['scope'] == 'PRODUCTION_NOT_STRENGTH', 'PARENT_PRODUCTION_REGISTRATION_REQUIRED')
    need(type(require_state) is bool, 'PARENT_EXPLICIT_STATE_CHECK_REQUIRED')
    parent = registration['parent']
    directory = safe_path(ROOT / parent['directory'])
    commit_path, state_path = safe_path(ROOT / parent['commit_path']), safe_path(ROOT / parent['state_path'])
    need(directory.is_relative_to(ROOT) and commit_path == directory / 'COMMIT.json'
         and state_path == directory / 'state.pt', 'PARENT_REGISTERED_PATHS')
    need(commit_path.is_file() and file_sha(commit_path) == parent['commit_sha256'], 'PARENT_COMMIT_MISSING_OR_CHANGED')
    commit = read_json(commit_path)
    need(commit.get('schema') == BC.COMMIT_SCHEMA and commit.get('status') == 'COMMITTED'
         and commit.get('scope') == 'V0_BC_NOT_STRENGTH', 'PARENT_COMMIT_KIND_OR_SCOPE')
    need(type(commit.get('global_update')) is int and commit['global_update'] == parent['global_update']
         and type(commit.get('state_bytes')) is int and commit['state_bytes'] == parent['state_bytes']
         and commit.get('state_sha256') == parent['state_sha256'], 'PARENT_COMMIT_REGISTERED_IDENTITY')
    identity = commit.get('identity', {})
    model = identity.get('model', {})
    need(identity.get('schema') == BC.SCHEMA and identity.get('runtime') == parent['runtime']
         and model.get('diagnostic') is False and model.get('config', {}).get('layers') == parent['n_layer']
         and model.get('action_schema_sha256') == parent['action_schema_sha256']
         and model.get('feature_schema_sha256') == parent['feature_schema_sha256'],
         'PARENT_RUNTIME_OR_MODEL_IDENTITY')
    try:
        BC.validate_model_manifest(model, production=True)
    except (ValueError, TypeError, KeyError) as exc:
        raise SequenceError('PARENT_MODEL_SOURCE_IDENTITY: ' + str(exc)) from exc
    present = state_path.is_file()
    if require_state:
        need(present, 'PARENT_STATE_MISSING')
        need(state_path.stat().st_size == parent['state_bytes']
             and file_sha(state_path) == parent['state_sha256'], 'PARENT_STATE_BYTES_OR_HASH')
    actor_identity = dict(kind='BC', commit_sha256=parent['commit_sha256'],
        state_sha256=parent['state_sha256'], model_identity=copy.deepcopy(model),
        training_identity_sha256=BC._digest(identity), runtime_sha256=BC._digest(identity['runtime']),
        diagnostic=False)
    return seal(dict(schema=SCHEMA + '-registered-parent-inspection-v1',
        registration_sha256=registration['content_sha256'], directory=str(directory),
        commit_sha256=parent['commit_sha256'], state_sha256=parent['state_sha256'],
        state_bytes=parent['state_bytes'], global_update=parent['global_update'],
        identity=actor_identity, state_present=present, state_bytes_verified=require_state,
        tensors_loaded=False, production_authorized=False))


def load_sequence_parent(registration: dict, *, device: str = 'cuda:0'):
    """Use the existing cold loader and authenticate its tensors against A's parent."""
    from farm_neat.evaluation.v0_policy import load_committed_scorer, CommittedScorer
    from farm_neat.training import v0_bc as BC
    inspected = inspect_sequence_parent(registration, require_state=True)
    need(device == registration['parent']['runtime']['device'], 'PARENT_RUNTIME_DEVICE_MISMATCH')
    loaded = load_committed_scorer(inspected['directory'], inspected['commit_sha256'],
                                  device=device, production=True)
    need(type(loaded) is CommittedScorer and loaded.identity == inspected['identity'],
         'PARENT_LOADED_IDENTITY_MISMATCH')
    model = loaded.scorer
    need(model.get_extra_state() == inspected['identity']['model_identity']
         and getattr(model, 'diagnostic', None) is False
         and sum(p.numel() for p in model.parameters()) == registration['parent']['parameter_count']
         and not model.training and all(not p.requires_grad and str(p.device) == device
                                        for p in model.parameters()), 'PARENT_LOADED_TENSOR_IDENTITY')
    need(BC.runtime_identity(next(model.parameters()).device) == registration['parent']['runtime'],
         'PARENT_LOADED_RUNTIME_MISMATCH')
    return loaded


def _authorize_native_collection(registration: dict, output: Path, preflight) -> None:
    """A ready JSON alone cannot authorize execution of an unaccepted BUILD.

    The READY handoff binds accepted BUILD/inert gates, source, exposure,
    outputs, script and runtime. Re-sealing READY cannot replace those joins.
    """
    need(preflight is not None, 'NATIVE_PREFLIGHT_REQUIRED')
    validate_registration(registration)
    need(registration['scope'] == 'PRODUCTION_NOT_STRENGTH', 'NATIVE_PRODUCTION_REGISTRATION_REQUIRED')
    report = read_json(preflight)
    need(report.get('schema') == PREFLIGHT_SCHEMA and report == seal(report)
         and report.get('status') == 'READY' and report.get('production_authorized') is True
         and report.get('blockers') == [] and report.get('software_blockers') == [],
         'NATIVE_PREFLIGHT_NOT_AUTHORIZED')
    need(report.get('registration_sha256') == registration['content_sha256']
         and report.get('source_pins') == registration['source_pins']
         and report.get('registered_parent') == registration['parent']
         and report.get('resources') == registration['preregistration']['resources']
         and report.get('collection_directory') == str(safe_path(output)), 'NATIVE_PREFLIGHT_IDENTITY_JOIN')
    pins = report.get('acceptance_pins', {})
    for task in ('V0-REGRET-SEQUENCE-BUILD', 'V0-REGRET-SEQUENCE-INERT-GATE'):
        name = 'work/runs/review-package/acceptance/' + task + '.json'
        path = safe_path(ROOT / name)
        need(path.is_file() and pins.get(name) == file_sha(path), 'NATIVE_ACCEPTED_GATE_REQUIRED:' + task)
        receipt = read_json(path)
        need(receipt.get('decision') == 'ACCEPT' and receipt.get('outcome') == 'PASS'
             and receipt.get('task_id') == task, 'NATIVE_ACCEPTED_GATE_REQUIRED:' + task)
        if task.endswith('BUILD'):
            module = 'src/farm_neat/training/v0_regret_sequence.py'
            need(receipt.get('source_pins', {}).get(module) == file_sha(ROOT / module),
                 'NATIVE_ACCEPTED_BUILD_SOURCE_MISMATCH')
    need(report.get('registration') == registration
         and report.get('fit_directory') == str(safe_path(ROOT / 'work/checkpoints/v0-regret-sequence-signal'))
         and report.get('campaign_owner_directory') == str(safe_path(CAMPAIGN_HOME)),
         'NATIVE_PREFLIGHT_CAMPAIGN_JOIN')
    script = report.get('launch_script', {})
    path = safe_path(script.get('path', ''))
    need(path == safe_path(preflight).with_suffix('.slurm.sh') and path.is_file()
         and script.get('production_authorized') is True and file_sha(path) == script.get('sha256')
         and path.stat().st_size == script.get('bytes'), 'NATIVE_PREFLIGHT_SCRIPT_JOIN')
    runtime = report.get('checks', {}).get('runtime', {})
    executable = safe_path(runtime.get('resolved_path', ''))
    need(executable.is_file() and file_sha(executable) == runtime.get('sha256'),
         'NATIVE_PINNED_RUNTIME_EXECUTABLE_DRIFT')
    from farm_neat.training import v0_bc as BC
    BC.require_production_runtime()
    need(executable == Path(sys.executable).resolve(), 'NATIVE_WORKER_INTERPRETER_MISMATCH')
    need(bool(os.environ.get('SLURM_JOB_ID')) and bool(os.environ.get('CUDA_VISIBLE_DEVICES')),
         'NATIVE_SLURM_CUDA_ALLOCATION_REQUIRED')


def validate_native_execution(record: dict, registration: dict) -> dict:
    """Check full native producer custody, never promote a diagnostic summary.

    Structural validation does not itself execute a replay or prove authenticity
    of externally fabricated JSON. The concrete producer + charged independent
    primary/replay launches provide that evidence; no native-strength claim is
    inferred merely from hashes or equal terminal WDL.
    """
    validate_registration(registration)
    need(registration['scope'] == 'PRODUCTION_NOT_STRENGTH', 'NATIVE_PRODUCTION_REGISTRATION_REQUIRED')
    need(type(record) is dict and record.get('schema') in
         (NATIVE_OBSERVED_SCHEMA, NATIVE_MONITOR_OBSERVED_SCHEMA), 'NATIVE_OBSERVED_SCHEMA_REQUIRED')
    header = inspect_sequence_parent(registration, require_state=False)
    launch = record.get('launch', {})
    expected_identity = header['identity']
    if launch.get('phase') == 'monitor':
        need(record['schema'] == NATIVE_MONITOR_OBSERVED_SCHEMA, 'NATIVE_MONITOR_SCHEMA_REQUIRED')
        frozen = _validate_native_monitor_freeze_reference(record.get('monitor_freeze'), registration)
        expected_identity = frozen['actor_identities'][launch['arm']]
    else:
        need(launch.get('phase') in ('discovery', 'confirmation')
             and record['schema'] == NATIVE_OBSERVED_SCHEMA, 'NATIVE_DEVELOPMENT_SCHEMA_REQUIRED')
    need(record.get('parent_identity') == expected_identity
         and record.get('runtime') == registration['parent']['runtime'], 'NATIVE_PARENT_OR_RUNTIME_JOIN')
    result = _validate_observed_execution(record, registration, native=True)
    expected = dict(identity=expected_identity, mode='greedy_class', seed=launch['primary_index'],
                    temperature=1., device=registration['parent']['runtime']['device'])
    if launch['law'] == KEEP_LAW:
        expected['publication'] = 'opening_only_later_keep'
    snapshots = [record['initial_actor']['policy'], record['terminal_actor']['policy']]
    for epoch in record['epochs']:
        snapshots.extend((epoch['policy_before'], epoch['policy_after']))
    for snapshot in snapshots:
        fields(snapshot, ('kind', 'state'), 'NATIVE_SCORER_SNAPSHOT_FIELDS')
        state = snapshot['state']
        need(snapshot['kind'] == 'SCORER' and type(state) is dict
             and set(state) == set(expected) | {'generator_state'}
             and all(state[key] == value for key, value in expected.items()),
             'NATIVE_PARENT_POLICY_LAW_OR_IDENTITY')
        # Greedy decoding never consumes the sampler, even at the parent opening.
        need(state['generator_state'] == snapshots[0]['state']['generator_state'],
             'NATIVE_GREEDY_SAMPLER_CHANGED')
    return result


def _native_execution_summary(record: dict, registration: dict, path: Path) -> dict:
    joint = [row['joint'] for row in record['transitions']]
    return seal(dict(schema=NATIVE_EXECUTION_SCHEMA,
        registration_sha256=registration['content_sha256'], launch=copy.deepcopy(record['launch']),
        status='COMPLETE', physical_games_started=1, physical_games_completed=1,
        physical_steps=len(joint), wdl=record['wdl'],
        own_actions_sha256=digest([row[record['launch']['seat']] for row in joint]),
        joint_actions_sha256=digest(joint), terminal_environment_sha256=digest(record['terminal']),
        epochs_sha256=digest(record['epochs']), actor_instance_sha256=record['actor_instance_sha256'],
        source_closure_sha256=record['source_closure_sha256'],
        runtime_sha256=execution_runtime_sha256(registration),
        initial_state_sha256=digest(record['initial']), error=None,
        native_evidence=dict(path=str(safe_path(path)), sha256=file_sha(path), bytes=path.stat().st_size)))


class NativeSequenceRunner:
    """Concrete original-rule development producer; no caller-supplied actors.

    One loaded, immutable registered parent; fresh ScorerPolicy or
    OpeningScorerPolicy, original environment and responsive rival on EACH
    charged launch. No teacher Plan, hidden state, alternative model, callback,
    seed override, rule override, or automatic retry is accepted.
    """
    def __init__(self, registration: dict, *, output: str | Path, preflight=None):
        output = safe_path(output)
        _authorize_native_collection(registration, output, preflight)
        parent = load_sequence_parent(registration)
        from farm_neat.training import v0_bc as BC
        self.registration, self.output, self.parent = copy.deepcopy(registration), output, parent
        self.identity = copy.deepcopy(parent.identity)
        self.model_state = BC._cpu_tree(parent.scorer.state_dict())
        self.policies = []  # Retain samplers so an address cannot masquerade as a new actor.

    def _check_frozen_parent(self, policy) -> None:
        from farm_neat.training import v0_bc as BC
        need(self.parent.identity == self.identity and policy.identity == self.identity
             and policy.scorer is self.parent.scorer and not policy.scorer.training
             and all(not p.requires_grad and p.grad is None for p in policy.scorer.parameters())
             and BC._same_tree(BC._cpu_tree(policy.scorer.state_dict()), self.model_state),
             'NATIVE_FROZEN_PARENT_MUTATED')

    def __call__(self, launch: dict) -> dict:
        from farm_neat.evaluation import v0_policy as owner, scenarios
        from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
        from farm_neat.training import v0_bc as BC
        registration = self.registration
        validate_registration(registration)
        journal_path = safe_path(self.output / 'JOURNAL.jsonl')
        need(journal_path.is_file(), 'NATIVE_DURABLE_RESERVATION_REQUIRED')
        journal = SequenceJournal(journal_path, registration)
        events = journal.events
        need(type(launch) is dict and type(launch.get('index')) is int
             and 0 <= launch['index'] < 80 and launch == registration['launches'][launch['index']],
             'NATIVE_DEVELOPMENT_LAUNCH_REQUIRED')
        need(bool(events) and events[-1]['status'] == 'STARTED'
             and events[-1]['launch_index'] == launch['index'], 'NATIVE_DURABLE_RESERVATION_REQUIRED')
        evidence_dir = safe_path(self.output / 'native-executions')
        evidence_dir.mkdir(parents=True, exist_ok=True)
        # A crash after reservation is never an automatic retry, including a
        # second process attempting the same durable STARTED entry.
        marker = evidence_dir / f"{launch['index']:03d}.ATTEMPT.json"
        _publish_fit_json(marker, seal(dict(schema=SCHEMA + '-native-attempt-v1',
            registration_sha256=registration['content_sha256'], launch=copy.deepcopy(launch),
            journal_head_sha256=events[-1]['content_sha256'])))
        case = owner.OriginalRuleCase(launch['seed'], launch['seat'], launch['opponent']).document()
        opening_only = launch['law'] == KEEP_LAW
        policy = self.parent.policy('greedy_class', launch['primary_index'], opening_only=opening_only)
        need(type(policy) is (owner.OpeningScorerPolicy if opening_only else owner.ScorerPolicy)
             and all(policy is not old for old in self.policies), 'NATIVE_FRESH_POLICY_REQUIRED')
        self.policies.append(policy)
        self._check_frozen_parent(policy)
        rival, rival_identity = scenarios.opponent(launch['opponent'])
        need(rival_identity == case['opponent_identity'], 'NATIVE_RIVAL_IDENTITY')
        environment = owner.environment_for_case(case)
        need(type(environment) is ExactKaggricultureEnvironment and environment.step_index == 0
             and environment.done is False, 'NATIVE_ORIGINAL_OPENING_ENVIRONMENT_REQUIRED')
        record = _observe_sequence_body(registration, launch, case=case, policy=policy,
                                        environment=environment, rival=rival, native=True)
        path = evidence_dir / f"{launch['index']:03d}.json"
        _publish_fit_json(path, record)  # Preserve failed and partial execution evidence too.
        need(record['status'] == 'COMPLETE', 'NATIVE_EXECUTION_FAILED:' + wire(record['error']))
        validate_registration(registration)
        self._check_frozen_parent(policy)
        validate_native_execution(record, registration)
        summary = _native_execution_summary(record, registration, path)
        _validate_execution(summary, registration, launch)
        return summary


# Native corpora retain file references to the charged discovery prefix rather
# than duplicating all 16 native traces. An append to the launch journal does not
# invalidate the frozen prefix; a rewrite, truncation or failed discovery does.
NATIVE_COLLECTION_SCHEMA = SCHEMA + '-native-corpus-collection-v1'
NATIVE_REFERENCE_SCHEMA = SCHEMA + '-native-reference-v1'


def _native_discovery_evidence(registration: dict, collection: str | Path) -> tuple[dict, list[dict]]:
    validate_registration(registration)
    need(registration['scope'] == 'PRODUCTION_NOT_STRENGTH', 'NATIVE_CORPUS_REGISTRATION_REQUIRED')
    directory = safe_path(collection)
    need(directory.is_dir() and (directory / 'REGISTRATION.json').is_file(),
         'NATIVE_CORPUS_EXISTING_COLLECTION_REQUIRED')
    registration_path = safe_path(directory / 'REGISTRATION.json')
    need(read_json(registration_path) == registration, 'NATIVE_CORPUS_COLLECTION_REGISTRATION_JOIN')
    journal_path = safe_path(directory / 'JOURNAL.jsonl')
    need(journal_path.is_file(), 'NATIVE_CORPUS_EXISTING_JOURNAL_REQUIRED')
    # Do not instantiate a missing journal: read-only corpus inspection must not
    # create an empty execution record. The whole chain is authenticated before
    # selecting its already-completed 32-event discovery prefix.
    journal = SequenceJournal(journal_path, registration)
    events = journal.events
    need(len(events) >= 32, 'NATIVE_CORPUS_COMPLETE_DISCOVERY_REQUIRED')
    prefix = events[:32]
    summaries, records = [], []
    for index, job in enumerate(registration['launches'][:16]):
        start, finish = prefix[2 * index:2 * index + 2]
        need(start['status'] == 'STARTED' and finish['status'] == 'FINISHED'
             and start['launch_index'] == finish['launch_index'] == index,
             'NATIVE_CORPUS_DISCOVERY_FAILURE_OR_MISSING_REPLAY')
        summary = finish['result']
        _validate_execution(summary, registration, job)
        need(summary['status'] == 'COMPLETE', 'NATIVE_CORPUS_COMPLETE_EXECUTION_REQUIRED')
        path = safe_path(summary['native_evidence']['path'])
        # Keep evidence under the original collection. Foreign matching files
        # are not interchangeable with the concrete producer's output paths.
        need(path == directory / 'native-executions' / f'{index:03d}.json',
             'NATIVE_CORPUS_FOREIGN_EVIDENCE_PATH')
        summaries.append(summary)
        records.append(read_json(path))
    for index in range(0, 16, 2):
        _validate_replay(summaries[index], summaries[index + 1])
    witness = seal(dict(schema=NATIVE_COLLECTION_SCHEMA, directory=str(directory),
        registration_file_sha256=file_sha(registration_path), discovery_event_count=32,
        discovery_prefix_sha256=digest(prefix),
        execution_manifest_sha256=digest(summaries)))
    return witness, records


def _validate_native_corpus_join(corpus: dict, registration: dict) -> None:
    value = corpus['collection']
    hashed(value, ('directory', 'registration_file_sha256', 'discovery_event_count',
                  'discovery_prefix_sha256', 'execution_manifest_sha256'),
           NATIVE_COLLECTION_SCHEMA, 'NATIVE_CORPUS_COLLECTION')
    need(type(value['directory']) is str and type(value['discovery_event_count']) is int
         and value['discovery_event_count'] == 32, 'NATIVE_CORPUS_COLLECTION_FIELDS')
    actual, records = _native_discovery_evidence(registration, value['directory'])
    need(value == actual, 'NATIVE_CORPUS_COLLECTION_PREFIX_OR_EVIDENCE_CHANGED')
    # These are the actually recorded teacher descendants, not a reconstruction
    # from Plan(), a fresh root fork, a summary hash, or a control's state.
    teachers = _observed_discovery(records, registration, native=True)
    _validate_teacher_execution_join(corpus['teacher_sequences'], teachers)


def corpus_from_native(registration: dict, *, collection: str | Path) -> dict:
    """Extract four discovery teachers from the native producer's charged files.

    This is a read-only corpus builder, not authorization to start a game or to
    optimize. Native fit/update-2 monitoring remain separate, fail-closed gates.
    Losing/tied complete discoveries can be retained but cannot bypass admission.
    Full native source/parent/runtime, replay, 719-step and 30-publication custody
    is revalidated; each of the 116 targets uses its actual carried Plan.
    """
    from farm_neat.policy import v0_menu_fast
    witness, records = _native_discovery_evidence(registration, collection)
    actual = _observed_discovery(records, registration, native=True)
    return seal(dict(schema=NATIVE_CORPUS_SCHEMA,
        registration_sha256=registration['content_sha256'],
        parent_commit_sha256=registration['parent']['commit_sha256'],
        parent_state_sha256=registration['parent']['state_sha256'],
        source_closure_sha256=digest(registration['source_pins']), split='discovery',
        teacher_sequences=_teacher_sequences(actual, v0_menu_fast.build_menu),
        collection=witness, reference=None))


def _reference_values(corpus: dict) -> list[float | None]:
    return [row['target']['reference_class_logp']
            for teacher in corpus['teacher_sequences'] for row in teacher['epochs']]


def _native_reference_scores(corpus: dict, registration: dict, *, device: str) -> tuple[list[float], dict]:
    """Cold-load the pinned parent, then replay both owners on all real prefixes.

    No model/menu injection is accepted by this interface. Loading and runtime
    authentication belong to the existing cold loader. The CPU tests replace
    that owner seam explicitly; their resulting temporary records are not native
    experiment evidence. Scoring is read-only and RNG-neutral after loading.
    """
    import torch
    from farm_neat.policy import v0_menu_fast, v0_scorer as A
    from farm_neat.training import v0_bc as BC
    loaded = load_sequence_parent(registration, device=device)
    model = loaded.scorer
    expected = inspect_sequence_parent(registration, require_state=False)['identity']
    need(loaded.identity == expected, 'NATIVE_REFERENCE_PARENT_IDENTITY_MISMATCH')
    need(not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters()),
         'NATIVE_REFERENCE_FROZEN_PARENT_REQUIRED')
    model_before = BC._cpu_tree(model.state_dict())
    rng = BC._rng_state(next(model.parameters()).device)
    values, worst = [], 0.0
    try:
        with torch.no_grad():
            for teacher in corpus['teacher_sequences']:
                for row in teacher['epochs']:
                    actor = row['actor']
                    state, prefix = _state(actor['state']), _prefix(actor['prefix'])
                    options = _projection(actor['projection'])
                    fast = v0_menu_fast.build_menu(state, prefix, projection_kwargs=options)
                    reference = A.build_menu(state, prefix, projection_kwargs=options)
                    need(fast.legal_ids == reference.legal_ids
                         and [(c.class_id, c.indices, c.successor.payload()) for c in fast.classes]
                         == [(c.class_id, c.indices, c.successor.payload()) for c in reference.classes],
                         'NATIVE_REFERENCE_MENU_CLASS_MISMATCH')
                    need(len(fast.phi) == len(reference.phi)
                         and all(len(a) == len(b) and all(math.isclose(x, y, rel_tol=2e-6, abs_tol=3e-5)
                             for x, y in zip(a, b)) for a, b in zip(fast.phi, reference.phi)),
                         'NATIVE_REFERENCE_PHI_MISMATCH')
                    likelihoods = []
                    for menu in (fast, reference):
                        cls = next((c for c in menu.classes if c.class_id == row['target']['class_id']), None)
                        need(cls is not None, 'NATIVE_REFERENCE_STOP_CLASS_MISSING')
                        logits = model.score(state, prefix, menu.legal_ids, menu.phi,
                                             memory={'hold_releases': options['hold_releases']})
                        logp = float(A.class_log_probs(logits, menu)[list(menu.classes).index(cls)])
                        need(math.isfinite(logp) and logp <= 3e-5, 'NATIVE_REFERENCE_NONFINITE_OR_POSITIVE_LOGP')
                        likelihoods.append(logp)
                    left, right = likelihoods
                    need(math.isclose(left, right, rel_tol=2e-6, abs_tol=3e-5),
                         'NATIVE_REFERENCE_BACKEND_LIKELIHOOD_MISMATCH')
                    worst = max(worst, abs(left - right))
                    values.append(left)
        need(len(values) == 116, 'NATIVE_REFERENCE_DENOMINATOR_FOUR_BY_29')
        need(BC._same_tree(model_before, BC._cpu_tree(model.state_dict()))
             and loaded.identity == expected and not model.training
             and all(not p.requires_grad and p.grad is None for p in model.parameters()),
             'NATIVE_REFERENCE_PARENT_MUTATED')
        need(BC._same_tree(rng, BC._rng_state(next(model.parameters()).device)),
             'NATIVE_REFERENCE_OBSERVER_CONSUMED_RNG')
    finally:
        # Scoring may fail; no caller-visible stochastic state leaks from it.
        BC._restore_rng(rng, next(model.parameters()).device)
    proof = seal(dict(schema=NATIVE_REFERENCE_SCHEMA,
        parent_commit_sha256=registration['parent']['commit_sha256'],
        parent_state_sha256=registration['parent']['state_sha256'],
        parent_identity_sha256=digest(expected),
        runtime_sha256=digest(registration['parent']['runtime']), rows_checked=116,
        fast_backend='farm_neat.policy.v0_menu_fast.build_menu',
        reference_backend='farm_neat.policy.v0_scorer.build_menu',
        max_abs_backend_logp_difference=worst, row_likelihoods_sha256=digest(values)))
    return values, proof


def _validate_native_reference(corpus: dict, registration: dict) -> None:
    saved, proof = _reference_values(corpus), corpus['reference']
    if proof is None:
        need(all(value is None for value in saved), 'NATIVE_REFERENCE_AUTHENTICATION_REQUIRED')
        return
    hashed(proof, ('parent_commit_sha256', 'parent_state_sha256', 'parent_identity_sha256',
        'runtime_sha256', 'rows_checked', 'fast_backend', 'reference_backend',
        'max_abs_backend_logp_difference', 'row_likelihoods_sha256'),
        NATIVE_REFERENCE_SCHEMA, 'NATIVE_REFERENCE')
    need(all(type(value) in (int, float) and math.isfinite(value) for value in saved)
         and proof['row_likelihoods_sha256'] == digest(saved), 'NATIVE_REFERENCE_ROWS_MISMATCH')
    # A re-sealed logp/proof is not sufficient. Replay against the actual parent
    # tensors and both menu owners on every validation of a bound native corpus.
    actual, expected = _native_reference_scores(corpus, registration,
                                               device=registration['parent']['runtime']['device'])
    need(proof == expected and all(math.isclose(a, b, rel_tol=2e-6, abs_tol=3e-5)
                                  for a, b in zip(saved, actual)), 'NATIVE_REFERENCE_REPLAY_MISMATCH')


def bind_native_sequence_likelihoods(corpus: dict, registration: dict, *, device: str = 'cuda:0') -> dict:
    """Bind only the protected production parent's real conditional class logps.

    No sampled-path behavior probability, direct-RL ratio, confirmation context,
    or monitor data is created. Input objects and execution files are not edited.
    This function does not enable the diagnostic trainer for production data.
    """
    need(registration.get('scope') == 'PRODUCTION_NOT_STRENGTH', 'NATIVE_REFERENCE_REGISTRATION_REQUIRED')
    need(device == registration['parent']['runtime']['device'], 'NATIVE_REFERENCE_RUNTIME_DEVICE_MISMATCH')
    checked = validate_sequence_corpus(corpus, registration)
    need(not checked['reference_likelihoods_bound'] and corpus['reference'] is None,
         'NATIVE_REFERENCE_ALREADY_BOUND')
    result = copy.deepcopy(corpus)
    values, proof = _native_reference_scores(result, registration, device=device)
    for row, value in zip((r for teacher in result['teacher_sequences'] for r in teacher['epochs']), values):
        row['target']['reference_class_logp'] = value
    result['reference'] = proof
    return seal(result)


CAMPAIGN_HOME = ROOT / 'work/runs/maintainer/v0-regret-sequence-signal/.campaign-owner'
CAMPAIGN_SCHEMA = SCHEMA + '-campaign-lease-v1'


@contextmanager
def sequence_campaign_lease(registration: dict, *, output: str | Path,
                            preflight: str | Path | None):
    """Serialize the single production campaign across *all* output directories.

    The fixed owner path is independent of registration/source hashes, output
    names and PIDs. Its immutable claim survives process death; changing any
    bound output, parent, registration or preflight cannot create a fresh lineage.
    This is a cooperating-process lease, not protection against manual deletion
    by the filesystem owner. Development, fit, monitor and replay all compose
    this lease; fresh workers inherit its same locked open-file description.
    """
    output = safe_path(output)
    _authorize_native_collection(registration, output, preflight)
    report = read_json(preflight)
    need(type(report.get('fit_directory')) is str, 'CAMPAIGN_FIT_DIRECTORY_REQUIRED')
    fit = safe_path(report['fit_directory'])
    need(fit != output and not fit.is_relative_to(output) and not output.is_relative_to(fit),
         'CAMPAIGN_OUTPUT_OVERLAP')
    claim = seal(dict(schema=CAMPAIGN_SCHEMA,
        preregistration_sha256=registration['preregistration_sha256'],
        registration_sha256=registration['content_sha256'],
        source_closure_sha256=digest(registration['source_pins']),
        parent_commit_sha256=registration['parent']['commit_sha256'],
        parent_state_sha256=registration['parent']['state_sha256'],
        preflight_sha256=file_sha(preflight), collection_directory=str(output), fit_directory=str(fit)))
    home = safe_path(CAMPAIGN_HOME)
    home.mkdir(parents=True, exist_ok=True)
    path = safe_path(home / 'CAMPAIGN.lock')
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SequenceError('CAMPAIGN_ALREADY_RUNNING') from exc
        identity = safe_path(home / 'CAMPAIGN.json')
        if identity.exists():
            need(read_json(identity) == claim, 'CAMPAIGN_FORK_REFUSED')
        else:
            _publish_fit_json(identity, claim)
        global _ACTIVE_CAMPAIGN_LEASE
        need(_ACTIVE_CAMPAIGN_LEASE is None, 'CAMPAIGN_NESTED_LEASE_REFUSED')
        _ACTIVE_CAMPAIGN_LEASE = (stream.fileno(), copy.deepcopy(claim))
        try:
            yield copy.deepcopy(claim)
        finally:
            _ACTIVE_CAMPAIGN_LEASE = None
        # Closing releases only the live lock. Never delete the durable claim.


def _preflight_launch_script(output: Path, details: dict, resources: dict, *,
                             authorized: bool = False) -> dict:
    """Render an immutable SLURM/storage handoff, without executing anything.

    Only a fully joined preflight enables the command. Missing gates retain an
    explicit exit-2 guard; editing it cannot bypass the native CLI's live checks.
    """
    import shlex
    from farm_neat.training import v0_bc as BC
    path = safe_path(output.with_suffix('.slurm.sh'))
    need(not path.exists(), 'PREFLIGHT_SCRIPT_ALREADY_EXISTS')
    interpreter = details.get('runtime', {}).get('resolved_path')
    collection = safe_path(output.parent / 'attempt-01')
    argv = None
    if interpreter:
        argv = [interpreter, '-B', str(ROOT / 'RUN.py'), 'storage', 'run',
            '--reserve-bytes', str(resources['new_retained_bytes_cap']),
            '--timeout', str(resources['wall_seconds_cap']), '--',
            interpreter, '-B', '-m', 'farm_neat.training.v0_regret_sequence', 'run',
            '--preflight', str(output), '--output', str(collection)]
    lines = [
        '#!/usr/bin/env bash',
        '# Source-only BLOCKED launch artifact; not accepted production authority.',
        '#SBATCH --job-name=farmneat-v0-regret-sequence',
        '#SBATCH --partition=gpu', '#SBATCH --gres=gpu:4090:1',
        '#SBATCH --cpus-per-task=8', '#SBATCH --mem=32G',
        '#SBATCH --time=01:30:00', '#SBATCH --no-requeue',
        'set -euo pipefail',
        ': "${SLURM_JOB_ID:?SLURM allocation required}"',
        ': "${CUDA_VISIBLE_DEVICES:?CUDA allocation required}"',
        "printf '%s\\n' 'PREFLIGHT_NOT_AUTHORIZED: required asset, runtime, exposure or acceptance gate is missing' >&2",
        'exit 2',
        '# Exact planned guarded argv below is unreachable in this BUILD version.',
        'unset PYTHONHOME PYTHONSTARTUP LD_PRELOAD',
        'export PYTHONNOUSERSITE=1',
        'export PYTHONDONTWRITEBYTECODE=1',
        'cd -- ' + shlex.quote(str(ROOT)),
    ]
    profile = details.get('runtime', {}).get('expected_profile', {})
    for key, value in sorted(profile.get('environment', {}).items()):
        need(key.isidentifier() and type(value) is str, 'PREFLIGHT_RUNTIME_ENVIRONMENT_FIELDS')
        lines.append('export ' + key + '=' + shlex.quote(value))
    pythonpath = os.pathsep.join([str(ROOT / 'src'), *([profile['pythonpath']] if profile.get('pythonpath') else [])])
    lines.append('export PYTHONPATH=' + shlex.quote(pythonpath))
    lines.append('exec ' + shlex.join(argv) if argv else '# No resolved pinned runtime; no command can be rendered.')
    if authorized:
        need(argv is not None, 'PREFLIGHT_EXECUTABLE_RUNTIME_REQUIRED')
        lines = [line for line in lines if line != 'exit 2'
                 and 'PREFLIGHT_NOT_AUTHORIZED:' not in line
                 and 'unreachable in this BUILD version' not in line]
        lines[1] = '# Source-bound accepted native campaign; authorization is rechecked by the CLI.'
    payload = ('\n'.join(lines) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    BC._fsync_dir(path.parent)
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                production_authorized=authorized, command_argv=argv,
                sbatch_argv=['sbatch', '--parsable', '--no-requeue', '--chdir', str(ROOT),
                             '--output', str(output.parent / 'slurm-%j.log'), str(path)])


PREFLIGHT_SCHEMA = SCHEMA + '-preflight-v3'


def preflight_sequence(preregistration: dict, *, output: str | Path,
                       exposure: dict | None = None, budget: dict | None = None) -> dict:
    """Immutable readiness report; no models, subprocesses or games are started.

    READY requires the exact parent/source/runtime, exposure attestation, both
    accepted maintainer gates and storage reserve. Native consumers recheck all
    authority and the current pinned runtime before allocating any worker.
    """
    import shutil
    from farm_neat.tooling import runtime
    approved = approved_preregistration()
    need(preregistration == approved, 'PREREGISTRATION_CHANGED')
    output = safe_path(output)
    need(not output.exists(), 'PREFLIGHT_OUTPUT_ALREADY_EXISTS')
    need(not safe_path(output.with_suffix('.slurm.sh')).exists(), 'PREFLIGHT_SCRIPT_ALREADY_EXISTS')
    blockers, details, acceptance_pins, registration = [], {}, {}, None
    parent = approved['parent']
    for kind in ('commit', 'state'):
        path = safe_path(ROOT / parent[kind + '_path'])
        details[kind] = dict(path=str(path), exists=path.is_file(), sha256=None, bytes=None)
        if not path.is_file():
            blockers.append('PARENT_' + kind.upper() + '_MISSING')
        else:
            details[kind].update(sha256=file_sha(path), bytes=path.stat().st_size)
            if details[kind]['sha256'] != parent[kind + '_sha256']:
                blockers.append('PARENT_' + kind.upper() + '_HASH_MISMATCH')
            if kind == 'state' and details[kind]['bytes'] != parent['state_bytes']:
                blockers.append('PARENT_STATE_SIZE_MISMATCH')
    # Model-source compatibility is independently checkable even when the
    # protected tensors/runtime/exposure join are missing from a source ZIP.
    from farm_neat.training import v0_bc as BC
    try:
        commit_path = safe_path(ROOT / parent['commit_path'])
        need(commit_path.is_file() and file_sha(commit_path) == parent['commit_sha256'],
             'PARENT_COMMIT_MISSING_OR_CHANGED')
        model_manifest = read_json(commit_path)['identity']['model']
        BC.validate_model_manifest(model_manifest, production=True)
        details['parent_model'] = dict(source_compatible=True, tensors_loaded=False)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        blockers.append('PARENT_MODEL_SOURCE_IDENTITY')
        details['parent_model'] = dict(source_compatible=False, tensors_loaded=False, reason=str(exc))
    try:
        spec = runtime.lock()
        section = runtime.training_section(spec)
        interpreter = runtime.interpreter(spec, section=section)
        details['runtime'] = dict(section=section, configured_path=str(interpreter),
            resolved_path=str(interpreter.resolve()), exists=interpreter.is_file(),
            expected_profile=copy.deepcopy(spec[section]), version_probe='NOT_RUN',
            sha256=file_sha(interpreter) if interpreter.is_file() else None)
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            blockers.append('PINNED_RUNTIME_MISSING')
    except (ValueError, OSError) as exc:
        blockers.append('PINNED_RUNTIME_UNRESOLVED')
        details['runtime'] = dict(error=str(exc), version_probe='NOT_RUN')
    if exposure is None:
        blockers.append('EXPOSURE_JOIN_REQUIRED')
    else:
        try:
            _exposure(exposure, approved, 'PRODUCTION_NOT_STRENGTH')
            details['exposure_sha256'] = digest(exposure)
            registration = register_sequence(approved, exposure=exposure, diagnostic=False)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            blockers.append('EXPOSURE_UNSAFE')
            details['exposure_error'] = str(exc)
    resources = approved['resources']
    if budget is not None and (type(budget) is not dict or budget != resources):
        blockers.append('RESOURCE_BUDGET_CHANGED')
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    project_bytes = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    reserve = resources['new_retained_bytes_cap']
    details['storage'] = dict(project_bytes=project_bytes, free_bytes=free,
        required_reserve_bytes=reserve, project_cap_bytes=resources['project_storage_cap_bytes'])
    if free < reserve or project_bytes + reserve > resources['project_storage_cap_bytes']:
        blockers.append('INSUFFICIENT_STORAGE_RESERVE')
    for task in ('V0-REGRET-SEQUENCE-BUILD', 'V0-REGRET-SEQUENCE-INERT-GATE'):
        name = 'work/runs/review-package/acceptance/' + task + '.json'
        path = safe_path(ROOT / name)
        try:
            receipt = read_json(path)
            need(receipt.get('decision') == 'ACCEPT' and receipt.get('outcome') == 'PASS'
                 and receipt.get('task_id') == task, 'ACCEPTED_GATE_REQUIRED:' + task)
            if task.endswith('BUILD'):
                module = 'src/farm_neat/training/v0_regret_sequence.py'
                need(receipt.get('source_pins', {}).get(module) == file_sha(ROOT / module),
                     'ACCEPTED_BUILD_SOURCE_MISMATCH')
            acceptance_pins[name] = file_sha(path)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            blockers.append('ACCEPTED_GATE_REQUIRED:' + task)
            details.setdefault('acceptance_errors', {})[task] = str(exc)
    try:
        from farm_neat.training.storage_layout import bindings
        guarded = bindings().status(ROOT, reserve_bytes=reserve, scan_timeout=5)
        details['declared_root_storage'] = guarded
        need(guarded['admissible'] is True
             and guarded['projected_bytes'] <= resources['project_storage_cap_bytes'],
             'DECLARED_ROOT_STORAGE_NOT_ADMISSIBLE')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        blockers.append('DECLARED_ROOT_STORAGE_NOT_ADMISSIBLE')
        details['declared_root_storage_error'] = str(exc)
    software_blockers = []
    authorized = not blockers and registration is not None
    launch_script = _preflight_launch_script(output, details, resources, authorized=authorized)
    report = seal(dict(schema=PREFLIGHT_SCHEMA, status='READY' if authorized else 'PARTIAL',
        preregistration_sha256=PREREG_SHA256, source_pins=source_pins(),
        registration=registration,
        registration_sha256=registration['content_sha256'] if registration is not None else None,
        registered_parent=copy.deepcopy(parent), resources=copy.deepcopy(resources),
        blockers=blockers, software_blockers=software_blockers, checks=details,
        acceptance_pins=acceptance_pins, production_authorized=authorized, launch_script=launch_script,
        collection_directory=str(safe_path(output.parent / 'attempt-01')),
        fit_directory=str(safe_path(ROOT / 'work/checkpoints/v0-regret-sequence-signal')),
        campaign_owner_directory=str(safe_path(CAMPAIGN_HOME)), native_games=0,
        scientific_outcome='NOT_RUN'))
    write_new(output, report)
    return report


# Native campaign integration. All public native front doors enter the SAME
# supervised campaign. Underscored workers are deliberately not CLI authorities.
NATIVE_FIT_AUTH_SCHEMA = SCHEMA + '-native-fit-authorization-v1'
NATIVE_MONITOR_FREEZE_SCHEMA = SCHEMA + '-native-monitor-freeze-v1'
NATIVE_MONITOR_OBSERVED_SCHEMA = SCHEMA + '-native-monitor-observed-execution-v1'


def _sequence_process_identity(pid: int | None = None) -> dict:
    """OS identity, not model/application RNG, distinguishes fresh interpreters."""
    pid = os.getpid() if pid is None else pid
    need(type(pid) is int and pid > 0, 'FIT_PROCESS_IDENTITY_PID')
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(stat[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def _validate_sequence_process_identity(value: dict) -> None:
    fields(value, ('pid', 'start_ticks', 'boot_id'), 'FIT_PROCESS_IDENTITY_FIELDS')
    need(type(value['pid']) is int and value['pid'] > 0
         and type(value['start_ticks']) is int and value['start_ticks'] >= 0
         and type(value['boot_id']) is str and len(value['boot_id']) == 36,
         'FIT_PROCESS_IDENTITY')


def _file_reference(path: str | Path) -> dict:
    path = safe_path(path)
    need(path.is_file(), 'NATIVE_REQUIRED_FILE_MISSING:' + str(path))
    return dict(path=str(path), sha256=file_sha(path), bytes=path.stat().st_size)


def _check_file_reference(value: dict, code: str) -> Path:
    fields(value, ('path', 'sha256', 'bytes'), code + '_FIELDS')
    path = safe_path(value['path'])
    need(type(value['bytes']) is int and value['bytes'] >= 0 and path.is_file()
         and path.stat().st_size == value['bytes'] and file_sha(path) == value['sha256'], code)
    return path


def _native_admission_context(registration: dict, collection: str | Path) -> tuple[dict, dict]:
    """Derive admission from the original 80 reservations and full native traces.

    A caller-provided ADMITTED object (even correctly re-sealed) is never evidence.
    Check the whole journal chain, its fixed development prefix, each concrete
    producer ATTEMPT and the original evidence location before returning scalars.
    Monitor appends do not change this immutable prefix or enlarge training data.
    """
    validate_registration(registration)
    need(registration['scope'] == 'PRODUCTION_NOT_STRENGTH', 'NATIVE_FIT_REGISTRATION_REQUIRED')
    collection = safe_path(collection)
    need(read_json(collection / 'REGISTRATION.json') == registration, 'NATIVE_FIT_REGISTRATION_JOIN')
    need((collection / 'JOURNAL.jsonl').is_file(), 'NATIVE_FIT_COMPLETE_80_STARTS_REQUIRED')
    journal = SequenceJournal(collection / 'JOURNAL.jsonl', registration)
    events = [event for event in journal.events if event['launch_index'] < 80]
    need(len(events) == 160 and all(events[2*i]['status'] == 'STARTED'
         and events[2*i+1]['status'] == 'FINISHED' for i in range(80)),
         'NATIVE_FIT_COMPLETE_80_STARTS_REQUIRED')
    development = _development_snapshot(registration, journal, stop_before=80)
    need(development is not None and development['status'] == 'READY_TO_FIT'
         and development['counts'] == dict(physical_starts=80, completed=80, failed=0,
                                           unknown=0, native_games=80)
         and read_json(collection / 'RESULT.json') == development,
         'NATIVE_FIT_ADMITTED_JOURNAL_JOIN')
    pins = {str(collection / name): file_sha(collection / name)
            for name in ('REGISTRATION.json', 'RESULT.json')}
    for index in range(80):
        start, finished = events[2*index:2*index+2]
        job = registration['launches'][index]
        record = finished['result']
        need(record['schema'] == NATIVE_EXECUTION_SCHEMA, 'NATIVE_FIT_NO_DIAGNOSTIC_PROMOTION')
        path = safe_path(collection / 'native-executions' / f'{index:03d}.json')
        need(record['native_evidence'] == _file_reference(path), 'NATIVE_FIT_EVIDENCE_LOCATION')
        attempt = path.with_name(f'{index:03d}.ATTEMPT.json')
        need(read_json(attempt) == seal(dict(schema=SCHEMA + '-native-attempt-v1',
            registration_sha256=registration['content_sha256'], launch=job,
            journal_head_sha256=start['content_sha256'])), 'NATIVE_FIT_RESERVATION_CUSTODY')
        pins[str(path)], pins[str(attempt)] = file_sha(path), file_sha(attempt)
    witness = seal(dict(schema=SCHEMA + '-native-admission-witness-v1',
        registration_sha256=registration['content_sha256'], collection_directory=str(collection),
        development_journal_head_sha256=events[-1]['content_sha256'],
        development_prefix_sha256=digest(events), admission_sha256=development['admission']['content_sha256'],
        discovery_starts=16, confirmation_starts=64, asset_pins=pins))
    return development, witness


def _native_fit_authorization(registration: dict, corpus: dict, development: dict,
                              witness: dict, preflight: str | Path) -> dict:
    collection = safe_path(witness['collection_directory'])
    header = inspect_sequence_parent(registration, require_state=True)
    report = read_json(preflight)
    return seal(dict(schema=NATIVE_FIT_AUTH_SCHEMA, scope='PRODUCTION_NOT_STRENGTH',
        registration_sha256=registration['content_sha256'],
        source_closure_sha256=digest(registration['source_pins']),
        collection_directory=str(collection), fit_directory=report['fit_directory'],
        preflight=_file_reference(preflight), parent_identity=header['identity'],
        corpus=_file_reference(collection / 'NATIVE-CORPUS.json'),
        corpus_sha256=corpus['content_sha256'], admission=copy.deepcopy(development['admission']),
        witness=copy.deepcopy(witness), fixed_updates=2, teacher_rows=116,
        sequence_denominator=4, confirmation_is_teacher=False, monitor_is_teacher=False))


def _verify_native_fit_authorization(registration: dict, corpus: dict, admission: dict,
                                   authorization: dict) -> None:
    """Cheap repeatable custody checks around mutation; deep replay precedes fit."""
    need(type(authorization) is dict, 'NATIVE_FIT_AUTHORIZATION_REQUIRED')
    hashed(authorization, ('scope', 'registration_sha256', 'source_closure_sha256',
        'collection_directory', 'fit_directory', 'preflight', 'parent_identity', 'corpus',
        'corpus_sha256', 'admission', 'witness', 'fixed_updates', 'teacher_rows',
        'sequence_denominator', 'confirmation_is_teacher', 'monitor_is_teacher'),
        NATIVE_FIT_AUTH_SCHEMA, 'NATIVE_FIT_AUTHORIZATION')
    collection = safe_path(authorization['collection_directory'])
    preflight = _check_file_reference(authorization['preflight'], 'NATIVE_FIT_PREFLIGHT_DRIFT')
    _authorize_native_collection(registration, collection, preflight)
    need(read_json(collection / 'FIT-AUTHORIZATION.json') == authorization,
         'NATIVE_FIT_DURABLE_AUTHORIZATION_REQUIRED')
    need(authorization['scope'] == 'PRODUCTION_NOT_STRENGTH'
         and authorization['registration_sha256'] == registration['content_sha256']
         and authorization['source_closure_sha256'] == digest(registration['source_pins'])
         and authorization['admission'] == admission
         and authorization['corpus_sha256'] == corpus['content_sha256']
         and authorization['fixed_updates'] == 2 and authorization['teacher_rows'] == 116
         and authorization['sequence_denominator'] == 4
         and authorization['confirmation_is_teacher'] is False
         and authorization['monitor_is_teacher'] is False, 'NATIVE_FIT_AUTHORIZATION_JOIN')
    path = _check_file_reference(authorization['corpus'], 'NATIVE_FIT_CORPUS_DRIFT')
    need(path == collection / 'NATIVE-CORPUS.json' and read_json(path) == corpus,
         'NATIVE_FIT_CORPUS_LOCATION')
    need(read_json(preflight)['fit_directory'] == authorization['fit_directory'],
         'NATIVE_FIT_OUTPUT_JOIN')
    events = SequenceJournal(collection / 'JOURNAL.jsonl', registration).events
    need(len(events) == 160 and events[-1]['status'] == 'FINISHED'
         and digest(events) == authorization['witness']['development_prefix_sha256'],
         'NATIVE_FIT_JOURNAL_CHANGED_OR_MONITOR_ALREADY_STARTED')
    for path, sha in authorization['witness']['asset_pins'].items():
        need(file_sha(safe_path(path)) == sha, 'NATIVE_FIT_CHARGED_EVIDENCE_DRIFT')
    need(authorization['parent_identity'] == inspect_sequence_parent(registration,
         require_state=True)['identity'], 'NATIVE_FIT_PARENT_DRIFT')


def _native_sequence_loss(model, corpus: dict, admission: dict, *, backward: bool):
    """Exact weighted sum, with fixed one-prefix microbatches (no denominator drift).

    Only four discovery teacher sequences are accepted by the upstream corpus
    validator. No reference/behavior ratio, confirmation input or opening target.
    This private primitive never performs admission or claims evidence on its own.
    """
    import torch
    from farm_neat.policy import v0_menu_fast, v0_scorer as A
    need(admission['teacher_gate'] == 1 and 0 < admission['teacher_weight'] <= 1,
         'NATIVE_FIT_POSITIVE_ADMISSION_REQUIRED')
    scale = float(admission['teacher_weight']) / 4.
    total = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)
    for example in _examples(corpus, menu_builder=v0_menu_fast.build_menu):
        scored = _batch(model, [example])
        loss = A.class_nll(scored.logits[0], example[3], valid=scored.valid[0]) * scale
        need(bool(torch.isfinite(loss)), 'nonfinite sequence loss')
        if backward:
            loss.backward()
        total += loss.detach().double()
    return total


def _native_fit_update(registration: dict, *, collection: str | Path,
                       preflight: str | Path, update: int) -> dict:
    """One authenticated update; only the supervised worker calls this entry."""
    from farm_neat.training import v0_bc as BC
    collection = safe_path(collection)
    _authorize_native_collection(registration, collection, preflight)
    need(type(update) is int and update in (1, 2), 'NATIVE_FIT_UPDATE_REQUIRED')
    development, witness = _native_admission_context(registration, collection)
    journal = SequenceJournal(collection / 'JOURNAL.jsonl', registration)
    need(journal.starts == 80, 'NATIVE_FIT_NO_MONITOR_FEEDBACK')
    corpus_path = collection / 'NATIVE-CORPUS.json'
    authorization_path = collection / 'FIT-AUTHORIZATION.json'
    if update == 1:
        need(not corpus_path.exists() and not authorization_path.exists(),
             'NATIVE_FIT_EXISTING_OR_INCOMPLETE_PREPARATION')
        corpus = bind_native_sequence_likelihoods(corpus_from_native(registration,
            collection=collection), registration)
        _publish_fit_json(corpus_path, corpus)
        authorization = _native_fit_authorization(registration, corpus, development, witness, preflight)
        _publish_fit_json(authorization_path, authorization)
    else:
        corpus, authorization = read_json(corpus_path), read_json(authorization_path)
        need(authorization == _native_fit_authorization(registration, corpus, development,
             witness, preflight), 'NATIVE_FIT_AUTHORIZATION_REPLAY')
    # Replays both real menu owners and true cold parent probabilities; no model
    # or menu callback is admitted through the production API.
    validate_sequence_corpus(corpus, registration)
    _verify_native_fit_authorization(registration, corpus, development['admission'], authorization)
    parent = load_sequence_parent(registration)
    need(parent.identity == authorization['parent_identity'], 'NATIVE_FIT_COLD_PARENT_JOIN')
    model = copy.deepcopy(parent.scorer).requires_grad_(True)
    fit = safe_path(authorization['fit_directory'])
    if update == 2:
        identity = read_json(fit / 'LINEAGE.json')
        root, _ = _load_fit_checkpoint(fit / 'step-00000000', model, identity,
                                       update=0, parent_sha256=None)
        need(BC._same_tree(root['model'], BC._cpu_tree(parent.scorer.state_dict())),
             'NATIVE_FIT_ROOT_IS_NOT_REGISTERED_PARENT')
        del root
    result = _fit_sequence_transaction(model, corpus, registration, development['admission'],
        output=fit, updates=1, resume=fit / 'step-00000001' if update == 2 else None,
        native_authorization=authorization)
    need(result['global_update'] == update, 'NATIVE_FIT_CURSOR')
    return result


def _native_monitor_fit_assets(registration: dict, development: dict, fit_output: Path,
                                parent, collection: Path) -> tuple[dict, dict]:
    """Authenticate update 2 without reading or scoring any teacher corpus."""
    from farm_neat.training import v0_bc as BC
    authorization_path = collection / 'FIT-AUTHORIZATION.json'
    authorization = read_json(authorization_path)
    identity = read_json(fit_output / 'LINEAGE.json')
    need(authorization == seal(authorization) and authorization['schema'] == NATIVE_FIT_AUTH_SCHEMA
         and authorization['registration_sha256'] == registration['content_sha256']
         and authorization['collection_directory'] == str(collection)
         and authorization['fit_directory'] == str(fit_output)
         and authorization['admission'] == development['admission']
         and authorization['witness']['development_journal_head_sha256'] == development['journal_head_sha256']
         and authorization['parent_identity'] == parent.identity, 'NATIVE_MONITOR_FIT_AUTHORIZATION_JOIN')
    need(identity == _fit_identity(parent.scorer, registration,
        {'content_sha256': authorization['corpus_sha256']}, development['admission'], authorization),
        'NATIVE_MONITOR_FIT_LINEAGE_JOIN')
    need({p.name for p in fit_output.iterdir()} ==
         {'FIT.lock', 'LINEAGE.json', *(f'step-{i:08d}' for i in range(3))},
         'MONITOR_EXACT_UPDATE2_LINEAGE_REQUIRED')
    previous, child_state, pins, commits = None, None, {}, []
    for index in range(3):
        directory = fit_output / f'step-{index:08d}'
        state, previous = _load_fit_checkpoint(directory, parent.scorer, identity,
                                               update=index, parent_sha256=previous)
        commit = read_json(directory / 'COMMIT.json')
        commits.append(commit)
        if index == 0:
            need(BC._same_tree(state['model'], BC._cpu_tree(parent.scorer.state_dict())),
                 'NATIVE_MONITOR_ROOT_IS_NOT_REGISTERED_PARENT')
        if index == 2:
            child_state = state['model']
        for name in ('ATTEMPT.json', 'COMMIT.json', 'state.pt'):
            path = directory / name
            pins[str(path)] = file_sha(path)
        del state
    need(commits[1]['process_identity'] != commits[2]['process_identity'],
         'NATIVE_UPDATE2_REQUIRES_FRESH_PROCESS')
    need(not BC._same_tree(child_state, BC._cpu_tree(parent.scorer.state_dict())),
         'MONITOR_CHILD_HAS_NO_PARAMETER_DELTA')
    for path in (fit_output / 'LINEAGE.json', authorization_path, collection / 'REGISTRATION.json',
                 collection / 'RESULT.json'):
        pins[str(path)] = file_sha(path)
    # The true registered parent is monitored, not an arbitrary local root.
    parent_identity = copy.deepcopy(parent.identity)
    child_identity = dict(kind='SEQUENCE_CHILD', diagnostic=False,
        commit_sha256=previous, state_sha256=commits[2]['state_sha256'],
        model_identity=copy.deepcopy(parent.scorer.get_extra_state()),
        lineage_sha256=identity['content_sha256'], global_update=2,
        registered_parent_commit_sha256=registration['parent']['commit_sha256'])
    frozen = seal(dict(schema=NATIVE_MONITOR_FREEZE_SCHEMA, scope='PRODUCTION_NOT_STRENGTH',
        registration_sha256=registration['content_sha256'],
        source_closure_sha256=digest(registration['source_pins']),
        development_journal_head_sha256=development['journal_head_sha256'],
        admission_sha256=development['admission']['content_sha256'],
        corpus_sha256=identity['corpus_sha256'], lineage_sha256=identity['content_sha256'],
        collection_directory=str(collection), fit_output=str(fit_output),
        parent=dict(directory=str(ROOT / registration['parent']['directory']),
            commit_sha256=registration['parent']['commit_sha256'],
            state_sha256=registration['parent']['state_sha256']),
        child=dict(directory=str(fit_output / 'step-00000002'), global_update=2,
            commit_sha256=previous, state_sha256=commits[2]['state_sha256']),
        actor_identities=dict(control=parent_identity, challenger=child_identity), asset_pins=pins,
        registered_parent=copy.deepcopy(registration['parent']), production_parent_authenticated=True,
        teacher_input=False, policy_law='UNRESTRICTED', scientific_outcome='NOT_EVALUATED'))
    return frozen, child_state


def _validate_native_monitor_freeze_reference(reference: dict, registration: dict) -> dict:
    path = _check_file_reference(reference, 'NATIVE_MONITOR_FREEZE_MISSING_OR_CHANGED')
    frozen = read_json(path)
    hashed(frozen, ('scope', 'registration_sha256', 'source_closure_sha256',
        'development_journal_head_sha256', 'admission_sha256', 'corpus_sha256', 'lineage_sha256',
        'collection_directory', 'fit_output', 'parent', 'child', 'actor_identities', 'asset_pins',
        'registered_parent', 'production_parent_authenticated', 'teacher_input', 'policy_law',
        'scientific_outcome'), NATIVE_MONITOR_FREEZE_SCHEMA, 'NATIVE_MONITOR_FREEZE')
    need(path == safe_path(frozen['collection_directory']) / 'MONITOR-FREEZE.json'
         and frozen['registration_sha256'] == registration['content_sha256']
         and frozen['source_closure_sha256'] == digest(registration['source_pins'])
         and frozen['registered_parent'] == registration['parent']
         and frozen['scope'] == 'PRODUCTION_NOT_STRENGTH'
         and frozen['production_parent_authenticated'] is True
         and frozen['teacher_input'] is False and frozen['policy_law'] == 'UNRESTRICTED'
         and frozen['scientific_outcome'] == 'NOT_EVALUATED', 'NATIVE_MONITOR_FREEZE_JOIN')
    for name, sha in frozen['asset_pins'].items():
        need(file_sha(safe_path(name)) == sha, 'MONITOR_FROZEN_ASSET_DRIFT')
    child_path = safe_path(frozen['fit_output']) / 'step-00000002'
    commit = read_json(child_path / 'COMMIT.json')
    need(frozen['child'] == dict(directory=str(child_path), global_update=2,
         commit_sha256=file_sha(child_path / 'COMMIT.json'), state_sha256=file_sha(child_path / 'state.pt'))
         and commit.get('global_update') == 2 and commit.get('lineage_sha256') == frozen['lineage_sha256'],
         'NATIVE_MONITOR_UPDATE2_COMMIT_JOIN')
    parent_identity = inspect_sequence_parent(registration, require_state=False)['identity']
    need(frozen['parent'] == dict(directory=str(ROOT / registration['parent']['directory']),
        commit_sha256=registration['parent']['commit_sha256'], state_sha256=registration['parent']['state_sha256'])
        and commit['scope'] == 'PRODUCTION_NOT_STRENGTH' and commit['state_sha256'] == frozen['child']['state_sha256']
        and commit['registration_sha256'] == registration['content_sha256'], 'NATIVE_MONITOR_PARENT_CHILD_COMMIT')
    child_identity = dict(kind='SEQUENCE_CHILD', diagnostic=False,
        commit_sha256=frozen['child']['commit_sha256'], state_sha256=frozen['child']['state_sha256'],
        model_identity=parent_identity['model_identity'], lineage_sha256=frozen['lineage_sha256'],
        global_update=2, registered_parent_commit_sha256=registration['parent']['commit_sha256'])
    need(frozen['actor_identities'] == dict(control=parent_identity, challenger=child_identity),
         'NATIVE_MONITOR_ACTOR_IDENTITY_JOIN')
    return frozen


class NativeMonitorRunner(NativeSequenceRunner):
    """Same original environment/evidence owner, but only frozen unrestricted actors."""
    def __init__(self, registration: dict, *, output: Path, preflight, frozen: dict,
                 parent, child_state):
        from farm_neat.evaluation.v0_policy import CommittedScorer
        from farm_neat.training import v0_bc as BC
        _authorize_native_collection(registration, output, preflight)
        self.registration, self.output = copy.deepcopy(registration), safe_path(output)
        self.frozen = copy.deepcopy(frozen)
        self.freeze_reference = _file_reference(self.output / 'MONITOR-FREEZE.json')
        child = copy.deepcopy(parent.scorer)
        child.load_state_dict(child_state, strict=True)
        child.eval().requires_grad_(False)
        self.factories = dict(control=parent, challenger=CommittedScorer(child,
                             copy.deepcopy(frozen['actor_identities']['challenger'])))
        self.states = {arm: BC._cpu_tree(factory.scorer.state_dict())
                       for arm, factory in self.factories.items()}
        self.policies = []

    def __call__(self, launch: dict) -> dict:
        from farm_neat.evaluation import v0_policy as owner, scenarios
        from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
        from farm_neat.training import v0_bc as BC
        registration = self.registration
        validate_registration(registration)
        need(type(launch) is dict and type(launch.get('index')) is int
             and 80 <= launch['index'] < 208 and launch == registration['launches'][launch['index']]
             and launch['phase'] == 'monitor' and launch['law'] == 'UNRESTRICTED', 'NATIVE_MONITOR_LAUNCH')
        need(_validate_native_monitor_freeze_reference(self.freeze_reference, registration) == self.frozen,
             'NATIVE_MONITOR_FREEZE_CHANGED')
        events = SequenceJournal(self.output / 'JOURNAL.jsonl', registration).events
        need(events and events[-1]['status'] == 'STARTED'
             and events[-1]['launch_index'] == launch['index'], 'NATIVE_DURABLE_RESERVATION_REQUIRED')
        directory = self.output / 'native-executions'
        directory.mkdir(exist_ok=True)
        _publish_fit_json(directory / f"{launch['index']:03d}.ATTEMPT.json",
            seal(dict(schema=SCHEMA + '-native-attempt-v1',
                registration_sha256=registration['content_sha256'], launch=copy.deepcopy(launch),
                journal_head_sha256=events[-1]['content_sha256'])))
        factory = self.factories[launch['arm']]
        policy = factory.policy('greedy_class', launch['primary_index'])
        need(type(policy) is owner.ScorerPolicy and all(policy is not old for old in self.policies),
             'NATIVE_FRESH_UNRESTRICTED_POLICY_REQUIRED')
        self.policies.append(policy)
        case = owner.OriginalRuleCase(launch['seed'], launch['seat'], launch['opponent']).document()
        rival, rival_identity = scenarios.opponent(launch['opponent'])
        need(rival_identity == case['opponent_identity'], 'NATIVE_RIVAL_IDENTITY')
        environment = owner.environment_for_case(case)
        need(type(environment) is ExactKaggricultureEnvironment and environment.step_index == 0
             and environment.done is False, 'NATIVE_ORIGINAL_OPENING_ENVIRONMENT_REQUIRED')
        record = _observe_sequence_body(registration, launch, case=case, policy=policy,
                                        environment=environment, rival=rival, native=True)
        record = seal({**record, 'schema': NATIVE_MONITOR_OBSERVED_SCHEMA,
                       'monitor_freeze': self.freeze_reference})
        path = directory / f"{launch['index']:03d}.json"
        _publish_fit_json(path, record)
        need(record['status'] == 'COMPLETE', 'NATIVE_EXECUTION_FAILED:' + wire(record['error']))
        for arm, actor in self.factories.items():
            need(actor.identity == self.frozen['actor_identities'][arm]
                 and not actor.scorer.training and all(not p.requires_grad and p.grad is None
                 for p in actor.scorer.parameters())
                 and BC._same_tree(BC._cpu_tree(actor.scorer.state_dict()), self.states[arm]),
                 'NATIVE_MONITOR_FROZEN_MODEL_MUTATED')
        validate_native_execution(record, registration)
        summary = _native_execution_summary(record, registration, path)
        _validate_execution(summary, registration, launch)
        return summary


def _collect_native_sequence_monitor(registration: dict, *, collection: str | Path,
                                     preflight: str | Path) -> dict:
    collection = safe_path(collection)
    _authorize_native_collection(registration, collection, preflight)
    fit = safe_path(read_json(preflight)['fit_directory'])
    need(collection.is_dir() and fit.is_dir(), 'MONITOR_EXISTING_COLLECTION_AND_FIT_REQUIRED')
    with (collection / 'COLLECTION.lock').open('a+') as lock, (fit / 'FIT.lock').open('r') as fit_lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fit_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SequenceError('MONITOR_COLLECTION_OR_FIT_ALREADY_RUNNING') from exc
        development, _ = _native_admission_context(registration, collection)
        parent = load_sequence_parent(registration)
        frozen, child_state = _native_monitor_fit_assets(registration, development, fit, parent, collection)
        journal = SequenceJournal(collection / 'JOURNAL.jsonl', registration)
        freeze_path = collection / 'MONITOR-FREEZE.json'
        if freeze_path.exists():
            need(read_json(freeze_path) == frozen, 'MONITOR_FREEZE_CHANGED')
        else:
            need(journal.starts == 80, 'MONITOR_FREEZE_REQUIRED_BEFORE_FIRST_START')
            _publish_fit_json(freeze_path, frozen)
        result_path = collection / 'MONITOR-RESULT.json'
        snapshot = _monitor_snapshot(registration, journal, frozen)
        if result_path.exists():
            need(snapshot is not None and read_json(result_path) == snapshot,
                 'MONITOR_CACHED_RESULT_JOURNAL_MISMATCH')
            return snapshot
        if snapshot is not None:
            _publish_fit_json(result_path, snapshot)
            return snapshot
        runner = NativeMonitorRunner(registration, output=collection, preflight=preflight,
                                     frozen=frozen, parent=parent, child_state=child_state)
        while journal.starts < 208:
            launch = registration['launches'][journal.starts]
            journal.reserve(launch)
            returned, failed = None, False
            try:
                returned = runner(copy.deepcopy(launch))
                _validate_execution(returned, registration, launch)
                if launch['replay']:
                    primary = journal.events[-2]
                    need(primary['status'] == 'FINISHED'
                         and primary['launch_index'] == launch['primary_index'], 'MONITOR_PRIMARY_REPLAY_JOIN')
                    _validate_replay(primary['result'], returned)
                journal.finish(launch, result=returned)
            except Exception as exc:
                failed = True
                journal.finish(launch, error=wire(dict(code='NATIVE_MONITOR_FAILED',
                    exception=type(exc).__name__, message=str(exc), returned_record=returned)))
            # Each new record and replay is checked immediately. Revalidate all
            # external evidence at completion/failure, not quadratically after
            # every successful launch (which repeatedly hashes large checkpoints).
            if failed or journal.starts == 208:
                snapshot = _monitor_snapshot(registration, journal, frozen)
                need(snapshot is not None, 'NATIVE_MONITOR_FINAL_SNAPSHOT_REQUIRED')
                _publish_fit_json(result_path, snapshot)
                return snapshot
        raise AssertionError('complete monitor must produce a fixed-stage result')


def fit_native_sequence(registration: dict, *, collection: str | Path,
                        preflight: str | Path | None) -> dict:
    """No model, scalars, teachers, alternate outputs or GPU bypass in this API."""
    return run_sequence_campaign(registration, collection=collection, preflight=preflight, stop_after='fit-2')


def collect_native_sequence_monitor(registration: dict, *, collection: str | Path,
                                    preflight: str | Path | None) -> dict:
    return run_sequence_campaign(registration, collection=collection, preflight=preflight)


CAMPAIGN_PHASES = ('development', 'fit-1', 'fit-2', 'monitor')
RESOURCE_SCHEMA = SCHEMA + '-campaign-resources-v1'
_ACTIVE_CAMPAIGN_LEASE = None


def _process_group_sample(pgid: int) -> tuple[dict, int]:
    """Linux /proc CPU counters and resident bytes for our owned process group."""
    ticks, page = os.sysconf('SC_CLK_TCK'), os.sysconf('SC_PAGE_SIZE')
    cpus, rss = {}, 0
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            row = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(row[2]) != pgid:
                continue
            cpus[(int(entry.name), int(row[19]))] = (int(row[11]) + int(row[12])) / ticks
            rss += max(0, int(row[21])) * page
        except (FileNotFoundError, ProcessLookupError):
            continue
    return cpus, rss


def _supervise_sequence_process(command: list[str], *, output: str | Path, wall_limit: float,
        cpu_limit: float, memory_limit: int, storage_check, pass_fds: tuple[int, ...] = (),
        activity_check=None) -> dict:
    """Bound and reap one fresh process group; no shell, retry or detached jobs.

    CPU includes threads and observed descendants, with wait4/rusage as a floor.
    RSS is sampled for the owned group; SLURM's fixed 32 GiB cgroup is the hard
    allocation limit. The outer existing storage guard and its declared-root
    accounting remain required. Timed-out/failed workers never lose their charge.
    """
    import resource
    import subprocess
    import time
    from farm_neat.training.storage import _terminate
    need(type(command) is list and bool(command) and all(type(x) is str for x in command),
         'RESOURCE_ARGV_REQUIRED')
    need(all(type(x) in (int, float) and math.isfinite(x) and x > 0
             for x in (wall_limit, cpu_limit)) and type(memory_limit) is int and memory_limit > 0,
         'RESOURCE_POSITIVE_LIMITS_REQUIRED')
    need(callable(storage_check), 'RESOURCE_STORAGE_OWNER_REQUIRED')
    storage_check()  # No allocation or output creation on an inadmissible reserve.
    output = safe_path(output)
    output.mkdir(parents=True, exist_ok=True)
    start, self0 = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    child0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    observed, peak_rss, reason, child, returncode = {}, 0, None, None, None
    next_storage = start
    with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                cwd=str(ROOT), env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                start_new_session=True, close_fds=True, pass_fds=pass_fds)
            while True:
                sample, rss = _process_group_sample(child.pid)
                for identity, seconds in sample.items():
                    observed[identity] = max(observed.get(identity, 0.), seconds)
                peak_rss = max(peak_rss, rss)
                now = time.monotonic()
                own = resource.getrusage(resource.RUSAGE_SELF)
                own_cpu = own.ru_utime + own.ru_stime - self0.ru_utime - self0.ru_stime
                if now - start >= wall_limit:
                    reason = 'RESOURCE_WALL_LIMIT'
                elif sum(observed.values()) + own_cpu >= cpu_limit:
                    reason = 'RESOURCE_CPU_LIMIT'
                elif rss > memory_limit:
                    reason = 'RESOURCE_MEMORY_LIMIT'
                if reason is not None:
                    break
                if activity_check is not None:
                    activity_check()
                if now >= next_storage:
                    storage_check()
                    next_storage = time.monotonic() + 1.
                returncode = child.poll()
                if returncode is not None:
                    break
                time.sleep(.025)
        except BaseException as exc:
            reason = str(exc) if isinstance(exc, SequenceError) else type(exc).__name__ + ': ' + str(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            if child is not None:
                # Includes same-group grandchildren after an early leader exit.
                _terminate(child, grace_seconds=.25)
                returncode = child.wait()
    end_child, end_self = resource.getrusage(resource.RUSAGE_CHILDREN), resource.getrusage(resource.RUSAGE_SELF)
    cpu = max(sum(observed.values()), end_child.ru_utime + end_child.ru_stime -
              child0.ru_utime - child0.ru_stime)
    own_cpu = end_self.ru_utime + end_self.ru_stime - self0.ru_utime - self0.ru_stime
    elapsed = time.monotonic() - start
    if reason is None and cpu + own_cpu >= cpu_limit:
        reason = 'RESOURCE_CPU_LIMIT'
    if reason is None and elapsed >= wall_limit:
        reason = 'RESOURCE_WALL_LIMIT'
    if reason is None:
        reason = 'PROCESS_COMPLETE' if returncode == 0 else 'PROCESS_EXIT_NONZERO'
    return seal(dict(schema=RESOURCE_SCHEMA + '-process', status='COMPLETE' if reason == 'PROCESS_COMPLETE' else 'FAILED',
        reason=reason, pid=child.pid if child is not None else None, returncode=returncode,
        wall_seconds=elapsed, cpu_seconds=cpu, supervisor_cpu_seconds=own_cpu,
        peak_group_rss_bytes=peak_rss, command_argv=list(command),
        automatic_retries=0, process_group_reaped=True))


def _campaign_storage_check(registration: dict, collection: Path, fit: Path, *, reserve: bool) -> dict:
    """Use the existing declared-root/inode-deduplicating guard, not a new quota."""
    import shutil
    from farm_neat.training.storage_layout import bindings
    owner, resources = bindings(), registration['preregistration']['resources']
    status = owner.status(ROOT, reserve_bytes=resources['new_retained_bytes_cap'] if reserve else 0,
                          scan_timeout=5)
    need(status['admissible'] is True and status['allocated_bytes'] <= resources['project_storage_cap_bytes'],
         'RESOURCE_PROJECT_STORAGE_LIMIT_OR_INCOMPLETE_SCAN')
    retained = owner.scan([path for path in (collection, fit) if path.exists()], timeout_seconds=5)
    need(retained['complete'] and retained['allocated_bytes'] <= resources['new_retained_bytes_cap'],
         'RESOURCE_RETAINED_STORAGE_LIMIT_OR_INCOMPLETE_SCAN')
    ancestor = collection.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    need(shutil.disk_usage(ancestor).free > (resources['new_retained_bytes_cap'] if reserve else 0),
         'RESOURCE_DISK_FREE_RESERVE')
    return dict(project_allocated_bytes=status['allocated_bytes'],
                retained_allocated_bytes=retained['allocated_bytes'])


def _campaign_phase_history(collection: Path, registration: dict) -> list[dict]:
    """No skipped, overwritten, failed or UNKNOWN phase can become a retry."""
    directory = collection / 'CAMPAIGN-PHASES'
    if not directory.exists():
        return []
    entries = sorted(directory.iterdir())
    need(len(entries) <= 4 and [entry.name for entry in entries] ==
         [f'{index:02d}-{name}' for index, name in enumerate(CAMPAIGN_PHASES[:len(entries)])],
         'CAMPAIGN_PHASE_ORDER_OR_FOREIGN_FILES')
    history, previous = [], None
    for index, entry in enumerate(entries):
        attempt_path, receipt_path = entry / 'ATTEMPT.json', entry / 'COMMIT.json'
        need(attempt_path.is_file(), 'CAMPAIGN_UNRESERVED_PHASE_DIRECTORY')
        attempt = read_json(attempt_path)
        need(attempt == seal(attempt) and attempt['schema'] == RESOURCE_SCHEMA + '-attempt'
             and attempt['phase'] == CAMPAIGN_PHASES[index]
             and attempt['registration_sha256'] == registration['content_sha256']
             and attempt['previous_phase_sha256'] == previous,
             'CAMPAIGN_PHASE_ATTEMPT_JOIN')
        need(receipt_path.is_file(), 'CAMPAIGN_UNKNOWN_PHASE_NO_AUTOMATIC_RETRY')
        receipt = read_json(receipt_path)
        need(receipt == seal(receipt) and receipt['schema'] == RESOURCE_SCHEMA + '-commit'
             and receipt['attempt_sha256'] == file_sha(attempt_path), 'CAMPAIGN_RESOURCE_COMMIT_JOIN')
        resource_path = _check_file_reference(receipt['resource'], 'CAMPAIGN_RESOURCE_RECEIPT_DRIFT')
        resource = read_json(resource_path)
        need(resource == seal(resource) and resource['schema'] == RESOURCE_SCHEMA + '-process'
             and all(type(resource.get(key)) in (int, float) and math.isfinite(resource[key])
                     and resource[key] >= 0 for key in ('cpu_seconds', 'supervisor_cpu_seconds', 'wall_seconds'))
             and resource['returncode'] == 0 and resource['reason'] == 'PROCESS_COMPLETE'
             and resource['automatic_retries'] == 0 and resource['process_group_reaped'] is True,
             'CAMPAIGN_RESOURCE_RECEIPT_HASH')
        need(resource['status'] == 'COMPLETE' and receipt['status'] == 'COMPLETE',
             'CAMPAIGN_FAILED_PHASE_NO_AUTOMATIC_RETRY')
        result_path = _check_file_reference(receipt['worker_result'], 'CAMPAIGN_WORKER_RESULT_DRIFT')
        result = read_json(result_path)
        need(result == seal(result) and result['registration_sha256'] == registration['content_sha256']
             and result['phase'] == CAMPAIGN_PHASES[index]
             and result['process_identity']['pid'] == resource['pid'], 'CAMPAIGN_WORKER_RESULT_JOIN')
        _validate_sequence_process_identity(result['process_identity'])
        need(type(result['gpu_elapsed_seconds']) in (int, float)
             and math.isfinite(result['gpu_elapsed_seconds']) and result['gpu_elapsed_seconds'] >= 0,
             'CAMPAIGN_GPU_MEASUREMENT')
        history.append(dict(receipt=receipt, resource=resource, worker=result))
        previous = file_sha(receipt_path)
    return history


def _campaign_game_watch(collection: Path, registration: dict):
    """Parent-side deadline also covers native calls that do not return to Python."""
    import time
    active = [None, None]
    def check():
        path = collection / 'JOURNAL.jsonl'
        if not path.exists():
            return
        # The writer fsyncs before native execution. A partial append is not an
        # execution authorization; the next poll sees its durable complete line.
        with path.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 65536))
            tail = stream.read()
        lines = tail.splitlines()
        if not tail.endswith(b'\n') or not lines:
            return
        try:
            event = json.loads(lines[-1])
        except (ValueError, UnicodeDecodeError):
            return
        if event.get('status') != 'STARTED':
            active[:] = [None, None]
        elif event['launch_index'] != active[0]:
            active[:] = [event['launch_index'], time.monotonic()]
        elif time.monotonic() - active[1] >= registration['preregistration']['resources']['per_game_timeout_seconds']:
            raise SequenceError('RESOURCE_GAME_DEADLINE: launch=' + str(active[0]))
    return check


def _run_campaign_phases(registration: dict, collection: Path, preflight: Path,
                         *, stop_after: str, manager_started_cpu: float | None = None) -> dict:
    """One manager holds the persistent lease across all fresh worker processes."""
    import resource
    import time
    from farm_neat.training import v0_bc as BC
    need(_ACTIVE_CAMPAIGN_LEASE is not None, 'CAMPAIGN_LIVE_LEASE_REQUIRED')
    descriptor, claim = _ACTIVE_CAMPAIGN_LEASE
    report, resources = read_json(preflight), registration['preregistration']['resources']
    fit = safe_path(report['fit_directory'])
    need(stop_after in CAMPAIGN_PHASES, 'CAMPAIGN_FIXED_PHASE_REQUIRED')
    baseline_path = collection / 'CAMPAIGN-RESOURCE-BASELINE.json'
    if not baseline_path.exists():
        need(not collection.exists() or not any(collection.iterdir()), 'CAMPAIGN_FOREIGN_EXISTING_COLLECTION')
        need(not fit.exists(), 'CAMPAIGN_FOREIGN_EXISTING_FIT')
        initial_storage = _campaign_storage_check(registration, collection, fit, reserve=True)
        collection.mkdir(parents=True, exist_ok=True)
        baseline = seal(dict(schema=RESOURCE_SCHEMA + '-baseline',
            registration_sha256=registration['content_sha256'], claim_sha256=claim['content_sha256'],
            started_monotonic=time.monotonic(), started_unix=time.time(),
            boot_id=_sequence_process_identity()['boot_id'], storage=initial_storage))
        _publish_fit_json(baseline_path, baseline)
    baseline = read_json(baseline_path)
    need(baseline == seal(baseline) and baseline['registration_sha256'] == registration['content_sha256']
         and baseline['claim_sha256'] == claim['content_sha256']
         and baseline['boot_id'] == _sequence_process_identity()['boot_id'], 'CAMPAIGN_RESOURCE_BASELINE_DRIFT')
    with _campaign_resource_session(registration, collection, started_cpu=manager_started_cpu) as session:
        return _run_campaign_phase_loop(registration, collection, preflight,
            stop_after=stop_after, session=session, baseline=baseline)


def _run_campaign_phase_loop(registration: dict, collection: Path, preflight: Path, *,
                             stop_after: str, session: dict, baseline: dict) -> dict:
    import time
    descriptor, claim = _ACTIVE_CAMPAIGN_LEASE
    report, resources = read_json(preflight), registration['preregistration']['resources']
    fit = safe_path(report['fit_directory'])
    history = _campaign_phase_history(collection, registration)
    # A completed negative/partial decision is final, not permission to advance.
    for row in history:
        _verify_campaign_phase_output(registration, collection, row['worker']['phase'], row['worker']['result'])
    if history and history[-1]['worker']['result'].get('status') in ('CLOSED_NEGATIVE', 'PARTIAL'):
        return history[-1]['worker']['result']
    while len(history) <= CAMPAIGN_PHASES.index(stop_after):
        index, phase = len(history), CAMPAIGN_PHASES[len(history)]
        used_cpu = (sum(row['resource']['cpu_seconds'] for row in history)
            + session['previous_cpu_seconds'] + _self_cpu_seconds() - session['started_cpu_seconds'])
        # Conservative allocation-time bound: phase wall time upper-bounds one
        # allocated GPU's elapsed use, including CPU waiting and synchronization.
        # CUDA event elapsed time is recorded separately, never called utilization.
        charged_gpu = sum(row['resource']['wall_seconds'] for row in history)
        wall_remaining = resources['wall_seconds_cap'] - (time.monotonic() - baseline['started_monotonic'])
        gpu_remaining = resources['gpu_seconds_cap'] - charged_gpu
        cpu_remaining = resources['aggregate_cpu_seconds_cap'] - used_cpu
        need(min(wall_remaining, gpu_remaining, cpu_remaining) > 0, 'CAMPAIGN_RESOURCE_BUDGET_EXHAUSTED')
        phase_dir = collection / 'CAMPAIGN-PHASES' / f'{index:02d}-{phase}'
        phase_dir.mkdir(parents=True, exist_ok=False)
        attempt = seal(dict(schema=RESOURCE_SCHEMA + '-attempt', phase=phase,
            registration_sha256=registration['content_sha256'], preflight_sha256=file_sha(preflight),
            claim_sha256=claim['content_sha256'], manager_process=_sequence_process_identity(),
            previous_phase_sha256=(file_sha(collection / 'CAMPAIGN-PHASES' /
                f'{index-1:02d}-{CAMPAIGN_PHASES[index-1]}' / 'COMMIT.json') if index else None)))
        _publish_fit_json(phase_dir / 'ATTEMPT.json', attempt)
        command = [report['checks']['runtime']['resolved_path'], '-B', '-m',
            'farm_neat.training.v0_regret_sequence', '_worker', '--preflight', str(preflight),
            '--output', str(collection), '--phase', phase, '--lease-fd', str(descriptor)]
        measured = _supervise_sequence_process(command, output=phase_dir,
            wall_limit=min(wall_remaining, gpu_remaining), cpu_limit=cpu_remaining,
            memory_limit=resources['memory_gib'] * 1024**3,
            storage_check=lambda: _campaign_storage_check(registration, collection, fit, reserve=False),
            activity_check=_campaign_game_watch(collection, registration), pass_fds=(descriptor,))
        _publish_fit_json(phase_dir / 'RESOURCE.json', measured)
        worker_path = phase_dir / 'WORKER-RESULT.json'
        success = measured['status'] == 'COMPLETE' and worker_path.is_file()
        receipt = seal(dict(schema=RESOURCE_SCHEMA + '-commit',
            status='COMPLETE' if success else 'FAILED', attempt_sha256=file_sha(phase_dir / 'ATTEMPT.json'),
            resource=_file_reference(phase_dir / 'RESOURCE.json'),
            worker_result=_file_reference(worker_path) if worker_path.is_file() else None))
        _publish_fit_json(phase_dir / 'COMMIT.json', receipt)
        if not success:
            journal_path = collection / 'JOURNAL.jsonl'
            events = SequenceJournal(journal_path, registration).events if journal_path.exists() else []
            return seal(dict(schema=SCHEMA + '-campaign-result', status='PARTIAL',
                reason=measured['reason'], last_phase=phase,
                physical_starts=sum(row['status'] == 'STARTED' for row in events),
                last_durable_event=events[-1]['content_sha256'] if events else None,
                resource_receipt=_file_reference(phase_dir / 'RESOURCE.json'),
                automatic_retry=False, scientific_outcome='PARTIAL', strength_claim=False))
        history = _campaign_phase_history(collection, registration)
        result = history[-1]['worker']['result']
        _verify_campaign_phase_output(registration, collection, phase, result)
        if result.get('status') in ('CLOSED_NEGATIVE', 'PARTIAL'):
            return result
    return history[-1]['worker']['result']



def _self_cpu_seconds() -> float:
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF)
    return value.ru_utime + value.ru_stime


@contextmanager
def _campaign_resource_session(registration: dict, collection: Path, *, started_cpu: float | None = None):
    """Persist manager CPU too; interruption is UNKNOWN, never a budget reset."""
    import time
    started_cpu = _self_cpu_seconds() if started_cpu is None else started_cpu
    started_wall = time.monotonic()
    home = collection / 'RESOURCE-SESSIONS'
    home.mkdir(exist_ok=True)
    entries = sorted(home.iterdir())
    need([entry.name for entry in entries] == [f'{index:04d}' for index in range(len(entries))],
         'CAMPAIGN_RESOURCE_SESSION_ORDER')
    previous, previous_cpu = None, 0.
    for entry in entries:
        need((entry / 'COMMIT.json').is_file(), 'CAMPAIGN_RESOURCE_SESSION_UNKNOWN_NO_AUTOMATIC_RETRY')
        receipt = read_json(entry / 'COMMIT.json')
        attempt = read_json(entry / 'ATTEMPT.json')
        need(receipt == seal(receipt) and attempt == seal(attempt)
             and receipt['schema'] == RESOURCE_SCHEMA + '-manager-commit'
             and attempt['schema'] == RESOURCE_SCHEMA + '-manager-attempt'
             and attempt['previous_sha256'] == previous
             and attempt['registration_sha256'] == registration['content_sha256']
             and receipt['attempt_sha256'] == file_sha(entry / 'ATTEMPT.json')
             and type(receipt['cpu_seconds']) in (int, float)
             and math.isfinite(receipt['cpu_seconds']) and receipt['cpu_seconds'] >= 0,
             'CAMPAIGN_RESOURCE_SESSION_CUSTODY')
        previous_cpu += receipt['cpu_seconds']
        previous = file_sha(entry / 'COMMIT.json')
    directory = home / f'{len(entries):04d}'
    directory.mkdir(exist_ok=False)
    _publish_fit_json(directory / 'ATTEMPT.json', seal(dict(schema=RESOURCE_SCHEMA + '-manager-attempt',
        registration_sha256=registration['content_sha256'], previous_sha256=previous,
        process_identity=_sequence_process_identity())))
    interrupted = False
    try:
        yield dict(previous_cpu_seconds=previous_cpu, started_cpu_seconds=started_cpu)
    except BaseException:
        interrupted = True
        raise
    finally:
        spent_cpu = _self_cpu_seconds() - started_cpu
        _publish_fit_json(directory / 'COMMIT.json', seal(dict(schema=RESOURCE_SCHEMA + '-manager-commit',
            attempt_sha256=file_sha(directory / 'ATTEMPT.json'), cpu_seconds=spent_cpu,
            wall_seconds=time.monotonic() - started_wall)))
        if not interrupted:
            # Include failed workers and the optional read-only replay as well.
            # Supervisor CPU is already in manager rusage: do not count it twice.
            cpu, gpu = previous_cpu + spent_cpu, 0.
            paths = list((collection / 'CAMPAIGN-PHASES').glob('*/RESOURCE.json'))
            if (collection / 'CAMPAIGN-REPLAY/RESOURCE.json').exists():
                paths.append(collection / 'CAMPAIGN-REPLAY/RESOURCE.json')
            for path in paths:
                measured = read_json(path)
                need(measured == seal(measured) and all(type(measured[key]) in (int, float)
                     and math.isfinite(measured[key]) and measured[key] >= 0
                     for key in ('cpu_seconds', 'wall_seconds')), 'CAMPAIGN_RESOURCE_ACCOUNTING')
                cpu += measured['cpu_seconds']
                gpu += measured['wall_seconds']
            resources = registration['preregistration']['resources']
            need(cpu < resources['aggregate_cpu_seconds_cap'], 'RESOURCE_CPU_LIMIT')
            need(gpu < resources['gpu_seconds_cap'], 'RESOURCE_GPU_ALLOCATION_TIME_LIMIT')
            baseline_path = collection / 'CAMPAIGN-RESOURCE-BASELINE.json'
            if baseline_path.exists():
                baseline = read_json(baseline_path)
                need(baseline['boot_id'] == _sequence_process_identity()['boot_id']
                     and time.monotonic() - baseline['started_monotonic'] < resources['wall_seconds_cap'],
                     'RESOURCE_WALL_LIMIT')


def _verify_campaign_phase_output(registration: dict, collection: Path, phase: str, result: dict) -> None:
    """Worker receipts are caches, not permission to ignore changed durable work."""
    need(type(result) is dict and result == seal(result), 'CAMPAIGN_PHASE_RESULT_HASH')
    if phase == 'development':
        journal = SequenceJournal(collection / 'JOURNAL.jsonl', registration)
        need(result == _development_snapshot(registration, journal, stop_before=80)
             and read_json(collection / 'RESULT.json') == result, 'CAMPAIGN_DEVELOPMENT_RESULT_DRIFT')
    elif phase.startswith('fit-'):
        update = int(phase[-1])
        authorization = read_json(collection / 'FIT-AUTHORIZATION.json')
        fit = safe_path(authorization['fit_directory'])
        if result.get('status') == 'CLOSED_NEGATIVE':
            failure = read_json(fit / f'step-{update:08d}' / 'FAILURE.json')
            need(result['reason'] == failure['message'] == 'NATIVE_ZERO_GRADIENT_CLOSED_NEGATIVE'
                 and not (fit / f'step-{update:08d}' / 'COMMIT.json').exists(),
                 'CAMPAIGN_NATIVE_ZERO_GRADIENT_JOIN')
            return
        need(result.get('global_update') == update and len(result.get('checkpoints', [])) == 1,
             'CAMPAIGN_FIT_CURSOR')
        checkpoint = result['checkpoints'][0]
        directory = fit / f'step-{update:08d}'
        commit = read_json(directory / 'COMMIT.json')
        need(checkpoint == dict(directory=str(directory), commit_sha256=file_sha(directory / 'COMMIT.json'),
                state_sha256=file_sha(directory / 'state.pt'), metrics=commit['metrics'])
             and commit['state_sha256'] == checkpoint['state_sha256']
             and result['registration_sha256'] == registration['content_sha256']
             and commit['scope'] == 'PRODUCTION_NOT_STRENGTH'
             and commit['global_update'] == update
             and commit['parent_manifest_sha256'] == file_sha(fit / f'step-{update-1:08d}' / 'COMMIT.json'),
             'CAMPAIGN_FIT_CHECKPOINT_DRIFT')
    else:
        need(phase == 'monitor', 'CAMPAIGN_PHASE')
        frozen = _validate_native_monitor_freeze_reference(
            _file_reference(collection / 'MONITOR-FREEZE.json'), registration)
        journal = SequenceJournal(collection / 'JOURNAL.jsonl', registration)
        need(_monitor_snapshot(registration, journal, frozen) == result
             and read_json(collection / 'MONITOR-RESULT.json') == result, 'CAMPAIGN_MONITOR_RESULT_DRIFT')

def run_sequence_campaign(registration: dict, *, collection: str | Path,
                          preflight: str | Path | None, stop_after: str = 'monitor') -> dict:
    """The sole executable native authority: one lease, fixed phases, no retries."""
    started_cpu = _self_cpu_seconds()
    collection = safe_path(collection)
    _authorize_native_collection(registration, collection, preflight)
    with sequence_campaign_lease(registration, output=collection, preflight=preflight):
        return _run_campaign_phases(registration, collection, safe_path(preflight),
                                    stop_after=stop_after, manager_started_cpu=started_cpu)


def _replay_campaign_evidence(registration: dict, collection: Path, preflight: Path) -> dict:
    """Read-only validation of the already charged primary/replay executions.

    `replay` never launches 104 extra games. Independent replay is part of the
    fixed 208-entry execution manifest and is performed by the run workers.
    """
    history = _campaign_phase_history(collection, registration)
    need(len(history) == 4, 'REPLAY_COMPLETE_FOUR_PHASE_CAMPAIGN_REQUIRED')
    development, _ = _native_admission_context(registration, collection)
    fit = safe_path(read_json(preflight)['fit_directory'])
    parent = load_sequence_parent(registration)
    frozen, _ = _native_monitor_fit_assets(registration, development, fit, parent, collection)
    need(read_json(collection / 'MONITOR-FREEZE.json') == frozen, 'REPLAY_FROZEN_CHILD_JOIN')
    journal = SequenceJournal(collection / 'JOURNAL.jsonl', registration)
    result = _monitor_snapshot(registration, journal, frozen)
    need(result is not None and result['status'] in ('PASS', 'CLOSED_NEGATIVE')
         and read_json(collection / 'MONITOR-RESULT.json') == result, 'REPLAY_MONITOR_RESULT_JOIN')
    return seal(dict(schema=SCHEMA + '-campaign-replay-result', status='VERIFIED',
        registration_sha256=registration['content_sha256'], monitor_result=result,
        charged_primary_starts=104, charged_replay_starts=104, new_physical_starts=0,
        new_optimizer_updates=0, teacher_input=False, strength_claim=False))


def replay_sequence_campaign(registration: dict, *, collection: str | Path,
                             preflight: str | Path | None) -> dict:
    """Guarded, resource-accounted, no-new-games replay of the immutable evidence."""
    started_cpu = _self_cpu_seconds()
    collection = safe_path(collection)
    _authorize_native_collection(registration, collection, preflight)
    with sequence_campaign_lease(registration, output=collection, preflight=preflight), \
            _campaign_resource_session(registration, collection, started_cpu=started_cpu) as session:
        # Replay is one additional validation phase, never a second experiment.
        # It spends the *remaining* original resource budget and cannot be retried.
        import time
        descriptor, claim = _ACTIVE_CAMPAIGN_LEASE
        history = _campaign_phase_history(collection, registration)
        need(len(history) == 4, 'REPLAY_COMPLETE_FOUR_PHASE_CAMPAIGN_REQUIRED')
        baseline = read_json(collection / 'CAMPAIGN-RESOURCE-BASELINE.json')
        need(baseline == seal(baseline) and baseline['claim_sha256'] == claim['content_sha256']
             and baseline['boot_id'] == _sequence_process_identity()['boot_id'], 'CAMPAIGN_RESOURCE_BASELINE_DRIFT')
        resources = registration['preregistration']['resources']
        remaining_wall = resources['wall_seconds_cap'] - (time.monotonic() - baseline['started_monotonic'])
        remaining_gpu = resources['gpu_seconds_cap'] - sum(row['resource']['wall_seconds'] for row in history)
        remaining_cpu = (resources['aggregate_cpu_seconds_cap']
            - sum(row['resource']['cpu_seconds'] for row in history)
            - session['previous_cpu_seconds'] - (_self_cpu_seconds() - session['started_cpu_seconds']))
        need(min(remaining_wall, remaining_gpu, remaining_cpu) > 0, 'CAMPAIGN_RESOURCE_BUDGET_EXHAUSTED')
        directory = collection / 'CAMPAIGN-REPLAY'
        directory.mkdir(exist_ok=False)
        attempt = seal(dict(schema=RESOURCE_SCHEMA + '-attempt', phase='replay',
            registration_sha256=registration['content_sha256'], preflight_sha256=file_sha(preflight),
            claim_sha256=claim['content_sha256'], manager_process=_sequence_process_identity(),
            previous_phase_sha256=file_sha(collection / 'CAMPAIGN-PHASES/03-monitor/COMMIT.json')))
        _publish_fit_json(directory / 'ATTEMPT.json', attempt)
        report = read_json(preflight)
        command = [report['checks']['runtime']['resolved_path'], '-B', '-m',
            'farm_neat.training.v0_regret_sequence', '_worker', '--preflight', str(safe_path(preflight)),
            '--output', str(collection), '--phase', 'replay', '--lease-fd', str(descriptor)]
        measured = _supervise_sequence_process(command, output=directory,
            wall_limit=min(remaining_wall, remaining_gpu), cpu_limit=remaining_cpu,
            memory_limit=resources['memory_gib'] * 1024**3, pass_fds=(descriptor,),
            storage_check=lambda: _campaign_storage_check(registration, collection,
                safe_path(report['fit_directory']), reserve=False))
        _publish_fit_json(directory / 'RESOURCE.json', measured)
        worker_path = directory / 'WORKER-RESULT.json'
        complete = measured['status'] == 'COMPLETE' and worker_path.is_file()
        _publish_fit_json(directory / 'COMMIT.json', seal(dict(schema=RESOURCE_SCHEMA + '-commit',
            status='COMPLETE' if complete else 'FAILED',
            attempt_sha256=file_sha(directory / 'ATTEMPT.json'),
            resource=_file_reference(directory / 'RESOURCE.json'),
            worker_result=_file_reference(worker_path) if worker_path.is_file() else None)))
        need(complete, 'REPLAY_RESOURCE_OR_PROCESS_FAILURE:' + measured['reason'])
        worker = read_json(worker_path)
        hashed(worker, ('phase', 'registration_sha256', 'process_identity', 'gpu_elapsed_seconds', 'result'),
               RESOURCE_SCHEMA + '-worker-result', 'REPLAY_WORKER_RESULT_JOIN')
        _validate_sequence_process_identity(worker['process_identity'])
        need(worker['phase'] == 'replay' and worker['registration_sha256'] == registration['content_sha256']
             and worker['process_identity']['pid'] == measured['pid']
             and type(worker['gpu_elapsed_seconds']) in (int, float)
             and math.isfinite(worker['gpu_elapsed_seconds']) and worker['gpu_elapsed_seconds'] >= 0,
             'REPLAY_WORKER_RESULT_JOIN')
        result = worker['result']
        hashed(result, ('status', 'registration_sha256', 'monitor_result', 'charged_primary_starts',
            'charged_replay_starts', 'new_physical_starts', 'new_optimizer_updates', 'teacher_input',
            'strength_claim'), SCHEMA + '-campaign-replay-result', 'REPLAY_RESULT_JOIN')
        need(result['status'] == 'VERIFIED' and result['registration_sha256'] == registration['content_sha256']
             and result['monitor_result'] == read_json(collection / 'MONITOR-RESULT.json')
             and result['teacher_input'] is False and result['strength_claim'] is False
             and all(type(result[key]) is int and result[key] == value for key, value in
                 (('charged_primary_starts', 104), ('charged_replay_starts', 104),
                  ('new_physical_starts', 0), ('new_optimizer_updates', 0))), 'REPLAY_RESULT_JOIN')
        return result


def _sequence_campaign_worker(preflight: Path, collection: Path, phase: str, lease_fd: int) -> dict:
    """Inherited-lease worker; a fabricated READY file alone is not authorization."""
    import torch
    report = read_json(preflight)
    registration = report['registration']
    _authorize_native_collection(registration, collection, preflight)
    need(phase in (*CAMPAIGN_PHASES, 'replay') and type(lease_fd) is int and lease_fd >= 3,
         'CAMPAIGN_INHERITED_LEASE_REQUIRED')
    actual, expected = os.fstat(lease_fd), (safe_path(CAMPAIGN_HOME) / 'CAMPAIGN.lock').stat()
    need((actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino), 'CAMPAIGN_LEASE_FD_IDENTITY')
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # The same inherited open-file description.
    directory = (collection / 'CAMPAIGN-REPLAY' if phase == 'replay' else
        collection / 'CAMPAIGN-PHASES' / f'{CAMPAIGN_PHASES.index(phase):02d}-{phase}')
    attempt = read_json(directory / 'ATTEMPT.json')
    claim = read_json(safe_path(CAMPAIGN_HOME) / 'CAMPAIGN.json')
    need(attempt == seal(attempt) and attempt['phase'] == phase
         and attempt['registration_sha256'] == registration['content_sha256']
         and attempt['preflight_sha256'] == file_sha(preflight)
         and attempt['claim_sha256'] == claim['content_sha256']
         and claim['collection_directory'] == str(collection)
         and attempt['manager_process'] == _sequence_process_identity(os.getppid())
         and not (directory / 'COMMIT.json').exists()
         and not (directory / 'WORKER-RESULT.json').exists(), 'CAMPAIGN_WORKER_PENDING_AUTHORITY')
    need(torch.cuda.device_count() == 1 and os.environ.get('SLURM_CPUS_PER_TASK') == '8',
         'CAMPAIGN_FIXED_GPU_CPU_ALLOCATION_REQUIRED')
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    try:
        if phase == 'development':
            result = _collect_sequence(registration, output=collection, native=True, preflight=preflight)
        elif phase in ('fit-1', 'fit-2'):
            result = _native_fit_update(registration, collection=collection, preflight=preflight,
                                        update=int(phase[-1]))
        elif phase == 'monitor':
            result = _collect_native_sequence_monitor(registration, collection=collection, preflight=preflight)
        else:
            result = _replay_campaign_evidence(registration, collection, preflight)
    except SequenceError as exc:
        if str(exc) != 'NATIVE_ZERO_GRADIENT_CLOSED_NEGATIVE':
            raise
        result = seal(dict(schema=SCHEMA + '-native-fit-negative', status='CLOSED_NEGATIVE',
            reason=str(exc), registration_sha256=registration['content_sha256'],
            attempted_update=int(phase[-1]), monitor_ran=False, strength_claim=False))
    finally:
        end.record()
        torch.cuda.synchronize()
    value = seal(dict(schema=RESOURCE_SCHEMA + '-worker-result', phase=phase,
        registration_sha256=registration['content_sha256'], process_identity=_sequence_process_identity(),
        gpu_elapsed_seconds=float(begin.elapsed_time(end)) / 1000., result=result))
    _publish_fit_json(directory / 'WORKER-RESULT.json', value)
    return value



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Source-bound fixed V0 sequence campaign; accepted native authority required.')
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('preflight', help='write immutable readiness report and guarded SLURM script')
    check.add_argument('--preregistration', type=Path, required=True)
    check.add_argument('--output', type=Path, required=True)
    check.add_argument('--exposure', type=Path)
    check.add_argument('--budget', type=Path)
    for name in ('run', 'replay', '_worker'):
        command = commands.add_parser(name, help='guarded native campaign' if name != '_worker' else argparse.SUPPRESS)
        command.add_argument('--preflight', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        if name == '_worker':
            command.add_argument('--phase', choices=(*CAMPAIGN_PHASES, 'replay'), required=True)
            command.add_argument('--lease-fd', type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'preflight':
            report = preflight_sequence(read_json(args.preregistration), output=args.output,
                exposure=read_json(args.exposure) if args.exposure else None,
                budget=read_json(args.budget) if args.budget else None)
            print(wire(dict(status=report['status'], output=str(args.output),
                blockers=report['blockers'], software_blockers=report['software_blockers'],
                scientific_outcome='NOT_RUN')))
            return 0 if report['production_authorized'] else 2
        preflight, collection = safe_path(args.preflight), safe_path(args.output)
        report = read_json(preflight)
        need(report.get('schema') == PREFLIGHT_SCHEMA and report == seal(report), 'PREFLIGHT_SCHEMA_OR_HASH')
        need(report.get('source_pins') == source_pins(), 'PREFLIGHT_SOURCE_DRIFT')
        need(report.get('status') == 'READY' and report.get('production_authorized') is True,
             'PREFLIGHT_NOT_AUTHORIZED')
        if args.command == '_worker':
            result = _sequence_campaign_worker(preflight, collection, args.phase, args.lease_fd)
        else:
            action = run_sequence_campaign if args.command == 'run' else replay_sequence_campaign
            result = action(report['registration'], collection=collection, preflight=preflight)
        print(wire(result))
        return 2 if result.get('status') == 'PARTIAL' else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # No false zero-game claim after a charged worker/attempt has begun.
        print(wire(dict(status='REFUSED', error=type(exc).__name__, reason=str(exc),
                        execution_accounting='CONSULT_DURABLE_CAMPAIGN_JOURNAL')), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
