"""Private sparse Y[candidate, hypothesis, continuation group], over EXISTING receipts.

``make_tensor(belief_record, hypothesis_bindings=..., pair_sources=..., mc_sources=...)``
validates the frozen actor, native prefix geometry, original paired receipts and MC
identities. It executes NO turns, callbacks, replay, search or training. Pair source:
``{hypothesis_id, source_family_id, request, result}``; MC source:
``{hypothesis_id, result}``. A binding is ``{hypothesis_id, opponent_sha256,
conditioning_source_sha256, semantics: 'DECLARED_CONTROLLER_CONDITIONING_V1'}``.
This is a PRIVATE producer declaration that a registered controller implements a
public hypothetical model, NOT an inferred true opponent class or proof of that
model. Missing bindings/outcomes stay unavailable (including UNKNOWN). No opponent
identity is inferred from a hypothesis slot or injected into the actor.

One tensor has ONE exact actor context/seat and one data split. Common-group keys
exclude intervention and opponent, but bind the world, snapshot, initial compiler
memory, execution backend and RNG contract. Same keys do NOT certify aligned
exogenous draws: the paired owner explicitly reports action-dependent weed draws.
Responsive opponent action tapes may differ across interventions. Within a cell,
contradictory deterministic receipts withdraw eligibility instead of adding precision.

``aggregate_tensor`` reports LOSS/DRAW/WIN laws, equal-weighted over complete
DECLARED source families (equal groups within family). Mixtures use the frozen
belief's ``mass_used`` on common complete families across all live hypotheses.
``compare_actions`` also intersects groups/families before subtracting WDL. There
is no cash tie-break, selector, confidence bound, calibration, independence proof,
regret admission or actor loss. Missing/censored/unverified evidence is not zero,
a loss, a renormalized mixture, or a reason to shrink the declared family.

Only verified complete pairs enter numeric summaries. Original outcomes and all
failed/partial attempts remain auditable. Replays and MC cache hits preserve original
attempts; aliases of one source world cannot be declared independent. MC candidate
IDs, original cached receipt requests, raw group IDs and source-family IDs survive.
MC's invalidated controls remain invalid. Unstarted jobs retain their registered
worlds; a one-arm budget interruption retains that attempt without fabricating a pair.

Wire v1 embeds ordered inputs as a JSON STRING for read-time revalidation. This
preserves order-sensitive paired hashes even inside a sorted FrozenBeliefRecord;
sorting decoded legacy receipts would destroy their original engine/replay identity.
It uses current local source pins; hashes do NOT authenticate producer chronology,
callback behavior, conditioning declarations or independent sampling. ``attach_tensor``
returns a NEW FrozenBeliefRecord with evidence only under trainer_only. Save the
actor digest before evaluation and supply expected_actor_sha256 when reading.
Opening/frozen-KEEP scope is inherited unchanged. Bounds: 4096 pair sources, 32 MC
sources, 64 MiB input JSON; this is a bounded offline evidence adapter, not a hot
learner tensor/device format. Source fixtures are not measured neural strength.

  PYTHONPATH=src python3 -B -m farm_neat.training.v0_belief_outcomes \\
      validate --input TENSOR.json --actor-sha256 SAVED_ACTOR_DIGEST
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

from farm_neat.evaluation import v0_regret_pairs as E
from farm_neat.training import v0_belief_records as B, v0_regret_records as R
from farm_neat.training.teachers import v0_regret_searcher as MC

SCHEMA = 'farm-v0-belief-outcome-tensor-v1'
SUMMARY_SCHEMA = 'farm-v0-belief-outcome-summary-v1'
CONDITIONING = 'DECLARED_CONTROLLER_CONDITIONING_V1'
SCOPE = 'TRAINER_EVALUATOR_ONLY; DESCRIPTIVE_MATCHED_WDL; NO_ADMISSION'
MAX_INPUT_BYTES = 64 * 1024 * 1024
RecordError = R.RecordError


def _need(ok, code):
    R._require(ok, code, '$.belief_outcomes')


def _same(actual, expected, code):
    _need(R._equal(actual, expected), code)


def _seal(value):
    return {**value, 'content_sha256': R.canonical_sha256(
        {k: v for k, v in value.items() if k != 'content_sha256'})}


def _canonical_bound(value, code):
    R._safe(value)
    _need(type(value) is dict and 'content_sha256' in value, code)
    _same(value, _seal(value), code)


def source_pins() -> dict:
    name = 'src/farm_neat/training/v0_belief_outcomes.py'
    return B.source_pins() | MC._mc_sources() | {
        name: hashlib.sha256((R.ROOT / name).read_bytes()).hexdigest()}


def _branch(request, execution, row, phase):
    R._fields(row, ('arm', 'attempt_id', 'phase', 'branch_sha256', 'status',
                    'terminal', 'error', 'response'), '$.attempt')
    _need(row['arm'] in E.ARMS and row['phase'] == phase, 'ATTEMPT_PHASE_OR_ARM')
    R._text(row['attempt_id'], '$.attempt.attempt_id')
    branch = E.P.bind(dict(schema=E.BRANCH_SCHEMA, request=request, arm=row['arm'],
                          execution=execution, phase=phase, attempt_id=row['attempt_id']))
    _same(row['branch_sha256'], branch['content_sha256'], 'ATTEMPT_BRANCH_MISMATCH')
    if row['status'] == 'COMPLETE':
        terminal = E._validate_response(branch, row['response'])
        _need(terminal is not None, 'COMPLETE_WITHOUT_TERMINAL')
        _same(row['terminal'], terminal, 'TERMINAL_MISMATCH')
        _need(row['error'] is None, 'COMPLETE_WITH_ERROR')
    else:
        _need(row['status'] in ('INCOMPLETE', 'INVALID') and row['terminal'] is None,
              'CENSORED_OUTCOME')
        R._text(row['error'], '$.attempt.error')
        if row['status'] == 'INCOMPLETE' and row['response'] is not None:
            _need(E._validate_response(branch, row['response']) is None, 'CENSORED_TERMINAL')
        # INVALID raw responses intentionally remain negative, never trusted labels.
    return branch


def _pair_receipt(request, result):
    E._bound(result, 'PAIR_HASH_MISMATCH')
    _need(result['schema'] == E.SCHEMA, 'PAIR_SCHEMA')
    _same(result['request_sha256'], request['content_sha256'], 'PAIR_REQUEST_MISMATCH')
    execution = result['execution']
    _canonical_bound(execution, 'EXECUTION_HASH_MISMATCH')
    _need(execution['kind'] in ('NATIVE_GAME', 'SYNTHETIC_DIAGNOSTIC'), 'EXECUTION_KIND')
    replay = result['replay']
    if replay['status'] == 'VERIFIED':
        # Accepted MC owner verifies all four original branch receipts and assembly.
        MC._mc_verified_receipt(request, result, execution)
    else:
        _same([r['arm'] for r in result['arms']], list(E.ARMS), 'PAIR_ARMS')
        _need(len(replay['arms']) <= 2, 'REPLAY_ARMS')
        _same([r['arm'] for r in replay['arms']], list(E.ARMS)[:len(replay['arms'])], 'REPLAY_ARMS')
        ids = []
        for phase, rows in (('EVALUATE', result['arms']), ('REPLAY', replay['arms'])):
            for row in rows:
                _branch(request, execution, row, phase)
                ids.append(row['attempt_id'])
        _need(len(ids) == len(set(ids)), 'REUSED_ATTEMPT')
        if not replay['arms']:
            _need(replay['status'] == 'NOT_RUN' and not replay['mismatches'], 'REPLAY_STATUS')
        elif len(replay['arms']) < 2:
            _need(replay['status'] == 'INCOMPLETE', 'REPLAY_STATUS')
        else:
            mismatched = {a['arm'] for a, b in zip(result['arms'], replay['arms'])
                if a['status'] != 'COMPLETE' or b['status'] != 'COMPLETE' or
                E.P.wire(E._replay_signature(a)) != E.P.wire(E._replay_signature(b))}
            _need(bool(mismatched) and replay['status'] == 'MISMATCH' and
                  {m['arm'] for m in replay['mismatches']} == mismatched, 'REPLAY_STATUS')
        _same(result, E._assemble(request, execution, result['arms'], replay), 'PAIR_DERIVED_MISMATCH')
    return execution


def _semantic(row):
    """Same candidate may be learner in one pair and challenger in another."""
    if row['status'] != 'COMPLETE':
        return None
    value = copy.deepcopy(E._replay_signature(row))
    value['row'].pop('agent')  # role name is not game state; keep every other field
    return E.P.wire_hash(value)


class _Join:
    """One bounded import transaction; no global caches or mutable actor state."""
    def __init__(self, record, bindings):
        self.actor = record['actor']
        self.candidates = {c['candidate_sha256']: c for c in self.actor['candidates']}
        self.hypotheses = {h['hypothesis_id']: h for h in self.actor['belief']['hypotheses']}
        self.bindings = {}
        for b in R._list(bindings, '$.hypothesis_bindings'):
            R._fields(b, ('hypothesis_id', 'opponent_sha256', 'conditioning_source_sha256',
                          'semantics'), '$.hypothesis_binding')
            _need(b['hypothesis_id'] in self.hypotheses, 'UNKNOWN_HYPOTHESIS')
            _need(b['hypothesis_id'] not in self.bindings, 'DUPLICATE_HYPOTHESIS_BINDING')
            _need(b['semantics'] == CONDITIONING, 'CONDITIONING_SEMANTICS')
            R._hash(b['opponent_sha256'], '$.opponent_sha256')
            R._hash(b['conditioning_source_sha256'], '$.conditioning_source_sha256')
            self.bindings[b['hypothesis_id']] = b
        self.groups, self.cells, self.attempts = {}, {}, {}
        self.requests, self.raw_groups, self.world_families = {}, {}, {}
        self.split = None
        self.cache_references = 0

    def request(self, request, hypothesis, family, execution):
        _need(hypothesis in self.bindings, 'UNKNOWN_OR_UNBOUND_HYPOTHESIS')
        R._text(family, '$.source_family_id')
        key = E.P.wire(request)  # full bytes as cache key, not an unchecked supplied digest
        if key not in self.requests:
            self.requests[key] = E._validate_request(request)
        req = self.requests[key]
        rec = req['record']
        _same(rec['actor_context'], self.actor['context'], 'ACTOR_CONTEXT_MISMATCH')
        _same(req['opponent']['identity']['content_sha256'],
              self.bindings[hypothesis]['opponent_sha256'], 'HYPOTHESIS_CONTROLLER_MISMATCH')
        split = rec['identity']['split']
        if self.split is None:
            self.split = split
        _need(split == self.split, 'SPLIT_MISMATCH')
        _canonical_bound(execution, 'EXECUTION_HASH_MISMATCH')
        _need(execution['kind'] in ('NATIVE_GAME', 'SYNTHETIC_DIAGNOSTIC'), 'EXECUTION_KIND')
        base = dict(world_sha256=req['world_sha256'], snapshot_sha256=req['initial_world_sha256'],
            rng_coupling=req['rng_binding'], seat=req['seat'],
            continuation_sha256=R.canonical_sha256(rec['continuation']),
            actor_context_sha256=rec['identity']['actor_context_sha256'],
            paired_source_pins_sha256=R.canonical_sha256(req['source_pins']),
            execution_sha256=execution['content_sha256'])
        gid = R.canonical_sha256(base)
        raw = rec['identity']['source_group_id']
        if raw in self.raw_groups:
            _same(self.raw_groups[raw], (gid, family), 'RAW_SOURCE_GROUP_MISMATCH')
        self.raw_groups[raw] = (gid, family)
        # One engine world cannot gain independent n via different raw group labels.
        # Omit backend/metadata IDs here: new diagnostic backends are not fresh worlds.
        world_key = R.canonical_sha256([req['initial_world_sha256'], req['rng_binding']])
        if world_key in self.world_families:
            _need(self.world_families[world_key] == family, 'SOURCE_FAMILY_ALIAS_MISMATCH')
        self.world_families[world_key] = family
        if gid not in self.groups:
            self.groups[gid] = dict(continuation_group_sha256=gid, **copy.deepcopy(base),
                world=copy.deepcopy(req['world']), source_family_id=family, source_group_ids=[])
        group = self.groups[gid]
        _need(group['source_family_id'] == family, 'SOURCE_FAMILY_MISMATCH')
        if raw not in group['source_group_ids']:
            group['source_group_ids'].append(raw)
        return gid

    def candidate(self, request, arm):
        path = request['record'][arm]
        key = R.canonical_sha256(dict(context_sha256=self.actor['context_sha256'],
            canonical_plan=path['canonical_plan'], continuation_signature=path['continuation_signature']))
        _need(key in self.candidates, 'UNKNOWN_CANDIDATE')
        # Request validator replayed this raw path on its actual current prefixes.
        for name in ('canonical_plan', 'continuation_signature'):
            _same(path[name], self.candidates[key][name], 'CANDIDATE_GEOMETRY_MISMATCH')
        return key

    def attempt(self, row, execution):
        key = row['attempt_id']
        descriptor = dict(attempt_id=key, phase=row['phase'], arm=row['arm'], status=row['status'],
            branch_sha256=row['branch_sha256'], response_sha256=R.canonical_sha256(row['response']),
            outcome_sha256=R.canonical_sha256(row), execution_sha256=execution['content_sha256'],
            evidence_kind=execution['kind'])
        if key in self.attempts:
            _same(self.attempts[key], descriptor, 'REUSED_ATTEMPT_MISMATCH')
        self.attempts[key] = descriptor

    def observation(self, request, row, *, hypothesis, gid, origin, result_sha256,
                    replay_status, eligible, withdrawn=False, mc_candidate_id=None,
                    mc_job_id=None, observation_id=None):
        cid = self.candidate(request, row['arm'])
        key = (cid, hypothesis, gid)
        value = self.cells.setdefault(key, dict(candidate_sha256=cid, hypothesis_id=hypothesis,
            continuation_group_sha256=gid, observations=[]))
        response = row['response'] if isinstance(row['response'], dict) else {}
        raw_row = response.get('row')
        raw_row = raw_row if isinstance(raw_row, dict) else {}
        hashes = raw_row.get('actions_by_seat_sha256')
        terminal = row['terminal'] if row['status'] == 'COMPLETE' else None
        obs = dict(attempt_id=row['attempt_id'], source_group_id=request['record']['identity']['source_group_id'],
            origin_sha256=origin, request_sha256=request['content_sha256'], paired_result_sha256=result_sha256,
            path_sha256=request['record'][row['arm']]['path_sha256'], raw_actions=copy.deepcopy(request['record'][row['arm']]['raw_actions']),
            opponent=copy.deepcopy(request['opponent']), arm=row['arm'], status=row['status'],
            raw_status=raw_row.get('status'), error=row['error'], replay_status=replay_status,
            wdl=None if terminal is None else terminal['wdl'],
            cash_by_seat=None if terminal is None else copy.deepcopy(terminal['cash_by_seat']),
            rival_actions_sha256=hashes[1-request['seat']] if isinstance(hashes, list) and len(hashes) == 2 else None,
            semantic_sha256=_semantic(row), eligible=bool(eligible and terminal is not None), withdrawn=withdrawn,
            mc_candidate_id=mc_candidate_id, mc_job_id=mc_job_id, observation_id=observation_id)
        # Duplicate source references retain input provenance but not a second observation.
        if obs not in value['observations']:
            value['observations'].append(obs)

    def pair(self, source, *, origin=None, permitted=True, withdrawn=False, mc_ids=None,
             mc_job_id=None, observation_id=None):
        req, result = source['request'], source['result']
        h, family = source['hypothesis_id'], source['source_family_id']
        gid = self.request(req, h, family, result['execution'])
        execution = _pair_receipt(req, result)
        for row in [*result['arms'], *result['replay']['arms']]:
            self.attempt(row, execution)
        for row in result['arms']:
            self.observation(req, row, hypothesis=h, gid=gid, origin=origin or result['content_sha256'],
                result_sha256=result['content_sha256'], replay_status=result['replay']['status'],
                eligible=permitted and result['replay']['status'] == 'VERIFIED', withdrawn=withdrawn,
                mc_candidate_id=(mc_ids or {}).get(row['arm']), mc_job_id=mc_job_id, observation_id=observation_id)

    def monte_carlo(self, source):
        R._fields(source, ('hypothesis_id', 'result'), '$.mc_source')
        h, result = source['hypothesis_id'], source['result']
        _canonical_bound(result, 'MC_HASH_MISMATCH')
        _need(result['schema'] == MC.MC_SCHEMA and result['status'] == 'DISCOVERY_ONLY', 'MC_SCHEMA')
        _need(result['actor_training_admissible'] is False and result['teacher_weight'] == 0.
              and result['confirmation_required'] is True, 'MC_ADMISSION')
        _same(result['source_pins'], MC._mc_sources(), 'MC_SOURCE_MISMATCH')
        protocol = MC._protocol(result['registration'], result['budget'])
        registry = {g['source_group_id']: g for g in protocol['discovery_world_groups']}
        families, execution = protocol['source_families'], result['execution']
        _canonical_bound(execution, 'MC_EXECUTION_HASH')
        state, prefix, options = R._actor(self.actor['context'])
        pool, menus = {}, {}
        for item in result['candidates']:
            geom = item['geometry']
            candidate, reasons = MC._candidate(self.actor['context'], dict(proposal_family=item['proposal_family'],
                raw_actions=geom['raw_actions']), state, prefix, options, menus,
                edit_limit=result['budget']['max_path_edits'])
            _need(candidate is not None and not reasons, 'MC_CANDIDATE_REPLAY')
            _same(geom, MC._visible(candidate), 'MC_GEOMETRY_MISMATCH')
            _same(item['candidate_id'], candidate['candidate_id'], 'MC_CANDIDATE_ID')
            _need(item['candidate_id'] not in pool, 'MC_DUPLICATE_CANDIDATE')
            pool[item['candidate_id']] = candidate
        _need(bool(pool), 'MC_EMPTY_POOL')
        learner = next(iter(pool))
        jobs = result['jobs']
        _same([dict(group_id=j['group_id'], candidate_id=j['candidate_id']) for j in jobs],
              result['draw_order'], 'MC_DRAW_JOIN')
        resolved, declared, event_ids = {}, set(), set()
        events = {}
        for event in result['attempts']:
            R._int(event['job_id'], '$.mc.attempt.job_id', 0, len(jobs)-1)
            events.setdefault(event['job_id'], []).append(event)
        for index, job in enumerate(jobs):
            _same(job['job_id'], index, 'MC_JOB_ORDER')
            raw, cid = job['group_id'], job['candidate_id']
            _need(raw in registry and cid in pool, 'MC_UNREGISTERED_JOB')
            declared.add(raw)
            duplicate = 'cache_of_job' in job
            if duplicate:
                parent = job['cache_of_job']
                _need(type(parent) is int and 0 <= parent < index, 'MC_DUPLICATE_REFERENCE')
                _same((raw, cid), (jobs[parent]['group_id'], jobs[parent]['candidate_id']), 'MC_DUPLICATE_REFERENCE')
                _same(job['paired_result'], jobs[parent]['paired_result'], 'MC_DUPLICATE_RECEIPT')
                req = resolved[parent]
                _need(not events.get(index), 'MC_DUPLICATE_HAS_STARTS')
            else:
                req = job.get('receipt_request', job.get('request'))
            resolved[index] = req
            if req is None:
                _need(job['status'] == 'INVALID' and job['reason'] == 'MC_CONTROL_CONTINUATION_MISMATCH'
                      and not events.get(index) and job['paired_result'] is None, 'MC_MISSING_REQUEST')
                continue
            gid = self.request(req, h, families[raw], execution)
            rec = req['record']
            actual = dict(source_group_id=raw, world_sha256=req['world_sha256'],
                snapshot_sha256=req['initial_world_sha256'], rng_sha256=R.canonical_sha256(req['rng_binding']),
                seed=req['world']['seed'])
            _same(actual, registry[raw], 'MC_GROUP_IDENTITY')
            _same(rec['identity']['source_group_id'], raw, 'MC_RAW_SOURCE_GROUP')
            _need(rec['identity']['split'] == 'discovery', 'MC_DISCOVERY_ONLY')
            _same(R.canonical_sha256(rec['continuation']), protocol['continuation_sha256'], 'MC_CONTINUATION')
            _same(req['opponent']['identity']['content_sha256'], protocol['opponent_sha256'], 'MC_OPPONENT')
            expected = dict(group_id=raw, **{k:v for k,v in actual.items() if k != 'source_group_id'},
                seat=req['seat'], actor_context_sha256=self.actor['context_sha256'],
                continuation_sha256=protocol['continuation_sha256'], opponent_sha256=protocol['opponent_sha256'],
                source_pins_sha256=R.canonical_sha256(result['source_pins']), rng_coupling=req['rng_binding'])
            _same(job['shared_identity'], expected, 'MC_SHARED_IDENTITY')
            for arm, candidate_id in (('learner', learner), ('challenger', cid)):
                self.candidate(req, arm)
                for name in ('raw_actions', 'canonical_plan', 'continuation_signature', 'prefix_class_trace'):
                    _same(rec[arm][name], pool[candidate_id][name], 'MC_PATH_JOIN')
            if 'receipt_request' in job:
                # Historical cache receipt is NEVER rebound to today's demand.
                demand = job['request']
                self.request(demand, h, families[raw], execution)
                for name in ('world', 'world_sha256', 'seat', 'opponent', 'source_pins',
                             'initial_world_sha256', 'rng_binding', 'timeout_seconds'):
                    _same(demand[name], req[name], 'MC_CACHE_DEMAND_MISMATCH')
                for name in ('actor_context', 'continuation', 'learner'):
                    _same(demand['record'][name], rec[name], 'MC_CACHE_DEMAND_MISMATCH')
                for name in ('raw_actions', 'canonical_plan', 'continuation_signature', 'prefix_class_trace'):
                    _same(demand['record']['challenger'][name], rec['challenger'][name], 'MC_CACHE_DEMAND_MISMATCH')
                withdrawn_hit = (job['status'] == 'INVALID' and
                                 job['reason'] == 'MC_CONTROL_CONTINUATION_MISMATCH')
                _need((job['status'] == 'CACHE_HIT' or withdrawn_hit) and not events.get(index),
                      'MC_CACHE_NEW_STARTS')
            if job['status'] == 'CACHE_HIT' or 'receipt_request' in job:
                self.cache_references += 1
            _need(job['status'] in ('COMPLETE', 'CACHE_HIT', 'INVALID', 'CENSORED',
                                   'NOT_STARTED', 'DUPLICATE_SUPPRESSED'), 'MC_JOB_STATUS')
            retained = []
            for event in events.get(index, []):
                _same((event['group_id'], event['candidate_id']), (raw, cid), 'MC_EVENT_JOB')
                row = event['outcome']
                branch = _branch(req, execution, row, event['phase'])
                _same(event['branch'], branch, 'MC_EVENT_BRANCH')
                _same(event['arm'], row['arm'], 'MC_EVENT_ARM')
                _need(row['attempt_id'] not in event_ids, 'MC_REUSED_EVENT')
                event_ids.add(row['attempt_id'])
                self.attempt(row, execution)
                retained.append(row)
            paired = job['paired_result']
            if paired is not None:
                _same(paired['execution'], execution, 'MC_PAIR_EXECUTION')
                if not duplicate and 'receipt_request' not in job:
                    _same(retained, [*paired['arms'], *paired['replay']['arms']], 'MC_EVENT_RECEIPT_JOIN')
                allowed = job['status'] in ('COMPLETE', 'CACHE_HIT')
                _need(not allowed or paired['replay']['status'] == 'VERIFIED', 'MC_UNVERIFIED_SUCCESS')
                self.pair(dict(hypothesis_id=h, source_family_id=families[raw], request=req, result=paired),
                    origin=result['content_sha256'], permitted=allowed, withdrawn=job['status'] == 'INVALID',
                    mc_ids={'learner': learner, 'challenger': cid}, mc_job_id=index,
                    observation_id=job.get('observation_id'))
            else:
                _need(job['status'] not in ('COMPLETE', 'CACHE_HIT'), 'MC_MISSING_PAIR')
                for row in retained:
                    if row['phase'] == 'EVALUATE':
                        self.observation(req, row, hypothesis=h, gid=gid, origin=result['content_sha256'],
                            result_sha256=None, replay_status='INCOMPLETE', eligible=False,
                            withdrawn=job['status'] == 'INVALID', mc_candidate_id=learner if row['arm'] == 'learner' else cid,
                            mc_job_id=index)
        _same(sorted(declared), sorted(registry), 'MC_REGISTRATION_COVERAGE')
        _same(len(event_ids), result['counts']['simulator_starts'], 'MC_START_COUNT')
        _same(sum(e['phase'] == 'REPLAY' for e in result['attempts']),
              result['counts']['replay_checks'], 'MC_REPLAY_COUNT')
        _same(len(event_ids) if execution['kind'] == 'NATIVE_GAME' else 0,
              result['counts']['physical_starts'], 'MC_PHYSICAL_COUNT')

    def finish(self):
        for group in self.groups.values():
            group['source_group_ids'].sort()
        for cell in self.cells.values():
            observations = cell['observations']
            observations.sort(key=lambda o: (o['attempt_id'], o['origin_sha256'], str(o['mc_job_id'])))
            signatures = {o['semantic_sha256'] for o in observations if o['semantic_sha256'] is not None}
            conflict = len(signatures) > 1
            withdrawn = any(o['withdrawn'] for o in observations)
            valid = [o for o in observations if o['eligible']]
            cell.update(eligible=bool(valid) and not conflict and not withdrawn,
                status='CONFLICT' if conflict else 'WITHDRAWN' if withdrawn else
                       'COMPLETE' if valid else 'UNVERIFIED_OR_CENSORED',
                wdl=valid[0]['wdl'] if valid and not conflict and not withdrawn else None)
        return dict(groups=[self.groups[k] for k in sorted(self.groups)],
            cells=[self.cells[k] for k in sorted(self.cells)],
            attempts=[self.attempts[k] for k in sorted(self.attempts)], split=self.split,
            counts=dict(unique_execution_attempts=len(self.attempts),
                replay_attempts=sum(a['phase'] == 'REPLAY' for a in self.attempts.values()),
                native_starts=sum(a['evidence_kind'] == 'NATIVE_GAME' for a in self.attempts.values()),
                cache_references=self.cache_references, replay_independent_groups_added=0,
                cache_independent_groups_added=0))


def make_tensor(belief_record: dict, *, hypothesis_bindings: list[dict],
                pair_sources: list[dict] | None = None, mc_sources: list[dict] | None = None,
                expected_actor_sha256: str | None = None) -> dict:
    """Read-only join. Conditioning and independence are explicit producer declarations."""
    record = B.validate_record(belief_record, expected_actor_sha256=expected_actor_sha256)
    pairs = [] if pair_sources is None else pair_sources
    mcs = [] if mc_sources is None else mc_sources
    inputs = dict(belief_record=record, hypothesis_bindings=hypothesis_bindings, pair_sources=pairs, mc_sources=mcs)
    R._safe(inputs)
    _need(len(R._list(pairs, '$.pair_sources')) <= 4096 and len(R._list(mcs, '$.mc_sources')) <= 32, 'SOURCE_LIMIT')
    _need(len(E.P.wire(inputs).encode()) <= MAX_INPUT_BYTES, 'INPUT_BYTE_LIMIT')
    join = _Join(record, hypothesis_bindings)
    for source in pairs:
        R._fields(source, ('hypothesis_id', 'source_family_id', 'request', 'result'), '$.pair_source')
        join.pair(source)
    for source in mcs:
        join.monte_carlo(source)
    actor = record['actor']
    return _seal(dict(schema=SCHEMA, scope=SCOPE, source_pins=source_pins(), inputs_wire=E.P.wire(inputs),
        actor_payload_sha256=record['actor_payload_sha256'], belief_sha256=R.canonical_sha256(actor['belief']),
        axes=dict(candidate_sha256=sorted(join.candidates), hypothesis_id=sorted(join.hypotheses)),
        **join.finish(), actor_training_admissible=False, teacher_weight=0.,
        independence='DECLARED_SOURCE_FAMILIES_NOT_STATISTICALLY_CERTIFIED',
        coupling='COMMON_KEYS_NOT_ALIGNED_EXOGENOUS_DRAW_CERTIFICATE'))


def validate_tensor(document: dict, *, expected_actor_sha256: str | None = None) -> dict:
    """Rebuild derived fields from original receipts; resigning a cell is insufficient."""
    R._safe(document)
    _need(type(document) is dict and document.get('schema') == SCHEMA, 'TENSOR_SCHEMA')
    _canonical_bound(document, 'TENSOR_HASH_MISMATCH')
    _same(document['source_pins'], source_pins(), 'TENSOR_SOURCE_MISMATCH')
    _need(type(document['inputs_wire']) is str and
          len(document['inputs_wire'].encode()) <= MAX_INPUT_BYTES, 'INPUT_BYTE_LIMIT')
    inputs = R._loads(document['inputs_wire'])
    R._fields(inputs, ('belief_record', 'hypothesis_bindings', 'pair_sources', 'mc_sources'), '$.inputs')
    expected = make_tensor(**inputs, expected_actor_sha256=expected_actor_sha256)
    _same(document, expected, 'TENSOR_DERIVED_MISMATCH')
    return expected


def _index(document):
    return {(c['candidate_sha256'], c['hypothesis_id'], c['continuation_group_sha256']): c
            for c in document['cells']}


def cell(document: dict, candidate_sha256: str, hypothesis_id: str, continuation_group_sha256: str) -> dict:
    """Explicit missing cell of the sparse tensor; never silently invent a zero."""
    t = validate_tensor(document)
    _need(candidate_sha256 in t['axes']['candidate_sha256'] and hypothesis_id in t['axes']['hypothesis_id'] and
          any(g['continuation_group_sha256'] == continuation_group_sha256 for g in t['groups']), 'UNKNOWN_AXIS')
    return _index(t).get((candidate_sha256, hypothesis_id, continuation_group_sha256),
        dict(candidate_sha256=candidate_sha256, hypothesis_id=hypothesis_id,
             continuation_group_sha256=continuation_group_sha256, status='UNAVAILABLE',
             eligible=False, wdl=None, observations=[]))


def _coverage(t, index, keys):
    families = {}
    for group in t['groups']:
        families.setdefault(group['source_family_id'], []).append(group)
    return {family: groups for family, groups in sorted(families.items()) if all(
        index.get((c, h, g['continuation_group_sha256']), {}).get('eligible', False)
        for g in groups for c, h in keys)}


def _stats(groups, laws):
    # laws is indexed by common group identity, each law is LOSS/DRAW/WIN.
    family_laws = {f: [math.fsum(laws[g['continuation_group_sha256']][k] for g in gs)/len(gs)
                       for k in range(3)] for f, gs in groups.items()}
    n = len(family_laws)
    wdl = [math.fsum(x[k] for x in family_laws.values())/n for k in range(3)] if n else None
    raw = sorted({s for gs in groups.values() for g in gs for s in g['source_group_ids']})
    return dict(wdl=wdl, utility_mean=None if wdl is None else wdl[2]-wdl[0],
        win_probability=None if wdl is None else wdl[2], independent_group_n=n,
        terminal_group_n=sum(map(len, groups.values())), source_family_ids=list(groups),
        source_group_ids=raw, continuation_group_ids=sorted(laws), family_wdl=family_laws,
        missing_policy='REQUIRE_COMPLETE_DECLARED_FAMILIES; NO_IMPUTATION_OR_RENORMALIZATION')


def _law(value):
    return [float(value == k) for k in (-1, 0, 1)]


def aggregate_tensor(document: dict) -> dict:
    """Descriptive matched mixture forecasts, not admission or policy selection."""
    t = validate_tensor(document)
    index = _index(t)
    belief = R._loads(t['inputs_wire'])['belief_record']['actor']['belief']
    mass_field = belief['mass_used']
    weights = {h['hypothesis_id']: h[mass_field] for h in belief['hypotheses']}
    result = {}
    for c in t['axes']['candidate_sha256']:
        per_h = {}
        for h in weights:
            groups = _coverage(t, index, [(c, h)])
            laws = {g['continuation_group_sha256']: _law(index[c, h, g['continuation_group_sha256']]['wdl'])
                    for gs in groups.values() for g in gs}
            per_h[h] = _stats(groups, laws)
        groups = _coverage(t, index, [(c, h) for h, mass in weights.items() if mass > 0])
        laws = {}
        for gs in groups.values():
            for g in gs:
                gid = g['continuation_group_sha256']
                laws[gid] = [math.fsum(mass * float(index[c, h, gid]['wdl'] == k)
                            for h, mass in weights.items() if mass > 0) for k in (-1, 0, 1)]
        mixture = _stats(groups, laws)
        mixture.update(weights=weights, mass_used=mass_field,
            unavailable_hypothesis_ids=sorted(h for h in weights if per_h[h]['wdl'] is None))
        result[c] = dict(per_hypothesis=per_h, mixture=mixture)
    return _seal(dict(schema=SUMMARY_SCHEMA, tensor_sha256=t['content_sha256'], candidates=result,
        wdl_order=['LOSS', 'DRAW', 'WIN'], actor_training_admissible=False, teacher_weight=0.,
        independence=t['independence'], scope=SCOPE))


def compare_actions(document: dict, left: str, right: str, *, hypothesis_id: str) -> dict:
    """Right-minus-left WDL, paired on the SAME complete families; no marginal subtraction."""
    t = validate_tensor(document)
    _need(left in t['axes']['candidate_sha256'] and right in t['axes']['candidate_sha256'] and
          hypothesis_id in t['axes']['hypothesis_id'], 'UNKNOWN_AXIS')
    index = _index(t)
    groups = _coverage(t, index, [(left, hypothesis_id), (right, hypothesis_id)])
    values = {f: math.fsum(index[right, hypothesis_id, g['continuation_group_sha256']]['wdl'] -
                          index[left, hypothesis_id, g['continuation_group_sha256']]['wdl'] for g in gs)/len(gs)
              for f, gs in groups.items()}
    n = len(values)
    return dict(left_candidate_sha256=left, right_candidate_sha256=right, hypothesis_id=hypothesis_id,
        mean_wdl_delta=math.fsum(values.values())/n if n else None, independent_group_n=n,
        family_wdl_deltas=values, source_family_ids=list(groups),
        source_group_ids=sorted({s for gs in groups.values() for g in gs for s in g['source_group_ids']}),
        continuation_group_ids=sorted(g['continuation_group_sha256'] for gs in groups.values() for g in gs),
        delta_bounds=[-2, 2], actor_training_admissible=False)


def attach_tensor(record: B.FrozenBeliefRecord, tensor: dict) -> B.FrozenBeliefRecord:
    private = record.to_dict()['trainer_only']
    _need('belief_outcome_tensor' not in private, 'TENSOR_ALREADY_ATTACHED')
    checked = validate_tensor(tensor, expected_actor_sha256=record.actor_payload_sha256)
    return record.with_trainer({**private, 'belief_outcome_tensor': checked})


def write_tensor(path: str | Path, document: dict) -> None:
    text = E.P.wire(validate_tensor(document)) + '\n'
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(text)


def read_tensor(path: str | Path, *, expected_actor_sha256: str | None = None) -> dict:
    return validate_tensor(R._loads(Path(path).read_text(encoding='utf-8')),
                           expected_actor_sha256=expected_actor_sha256)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'aggregate'))
    parser.add_argument('--input', required=True)
    parser.add_argument('--actor-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        t = read_tensor(args.input, expected_actor_sha256=args.actor_sha256)
        out = aggregate_tensor(t) if args.command == 'aggregate' else dict(
            schema=SCHEMA + '-validation', status='PASS', content_sha256=t['content_sha256'],
            actor_payload_sha256=t['actor_payload_sha256'], native_games_executed=0)
    except (ValueError, KeyError, TypeError, OSError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(out, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
