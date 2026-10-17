"""Build, cold-preflight and overage-simulate the JOINT EntityGraph Kaggle archive.

Owner: evaluation maintainer. CPU only; no uploader, optimizer or GPU path.

    build-joint --model PATH.pt --model-sha256 SHA --output ARCHIVE.tar.gz [--profile v1|v2 ...]
    preflight-joint --archive PATH --archive-sha256 SHA --out JSON [--emulate-cpu-count 2] [--slowdown R]
    simulate-overage --preflight JSON --slowdown 1,2,3 --bank 60

Runtime profiles, selected at build time and recorded in DEPLOYMENT.json:

  v1 (default) ships joint_deployment_runtime.Agent. That file is not edited, so a
     v1 build from a recorded receipt (archive d50d5aa7...) reproduces the archive
     bytes. v1 keeps its known defect: after an executor or observation exception
     it plays PASS for the rest of the episode.
  v2 ships AgentV2 from THIS module (the module is an archive member of v2 builds):
     K cached opening programs per seat published at step 0 without a model call
     (refused if the live step-0 input digest differs), a daemon model-load thread
     started at the first call, pre-dawn drafts sampled from the latest own-seat
     view and published through propose()/publish at the next dawn, a bank-guarded
     synchronous compute when no draft is ready, optional fast encode (fast_gdn +
     fast_moe) kept only if a load-time self-check passes, an optional slot95
     cached-entity decoder, and a fallback that repairs the executor clock instead
     of latching PASS.

The legacy pooled builder (deployment.py) is reused read-only for its source
closure, deterministic tar writer and authenticated extractor. The joint codec
blocker (no parameter_report.parameter_tensors) is handled by the runtime's
codec_view; the unchanged original manifest ships beside the payload.

This module imports only the standard library at import time: the worker runs
in a fresh process whose PYTHONPATH is the extracted archive plus the
torch/numpy site, so actor code comes only from the archive.
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_CAP_BYTES = 100_000_000
MEMBER_CAP = 150
PREFLIGHT_SCHEMA = 'farmneat-joint-deployment-preflight-v1'
WORKER_SCHEMA = 'farmneat-joint-deployment-cold-game-v1'
BUILD_SCHEMA = 'farmneat-joint-deployment-build-v1'
SIMULATION_SCHEMA = 'farmneat-joint-overage-simulation-v1'
ACT_TIMEOUT = 1.0
GAME_STEPS = 719

# ----- runtime profile v2 identities ------------------------------------------------------
PROFILES = ('v1', 'v2')
SCHEMA_V2 = 'farmneat-joint-cpu-deployment-v2'
RUNTIME_V2_SCHEMA = 'farmneat-joint-runtime-v2-options-v1'
LOG_SCHEMA_V2 = 'farmneat-joint-deployment-log-v2'
CACHE_SCHEMA = 'farmneat-joint-opening-cache-v1'
SELF_CHECK_SCHEMA = 'farmneat-joint-fast-encode-self-check-v1'
FALLBACK_V2 = 'LOG_EXCEPTION_REPAIR_EXECUTOR_CLOCK_THEN_LAST_PUBLISHED_PLAN_ELSE_PASS_NO_LATCH'
CACHE_MEMBER = 'OPENING-CACHE.json'
MODEL_MEMBERS = ('MODEL.bin', 'MODEL.bin.json', 'ORIGINAL-MODEL.json')
V2_MODULE = 'src/farm_neat/evaluation/joint_deployment.py'
FAST_GDN = 'work/runs/maintainer/fast-joint-arm/fast_gdn.py'
FAST_MOE = 'work/runs/maintainer/fast-joint-arm/fast_moe.py'
SLOT95_DECODER = 'work/runs/reviewer-play/joint-deploy-archive/joint_decode.py'
V2_SOURCES = (V2_MODULE, FAST_GDN, FAST_MOE, SLOT95_DECODER)
REFERENCE_DECODER = 'joint-reference-v1'
SLOT95_PROFILE = 'joint-frozen-entity-kv-v1'
INVARIANCE_SEEDS = (0, 1, 2, 3, 9160000, 9161000, 123456789, 2**31 - 1)
DEFAULT_CACHE_K = 16
DEFAULT_CACHE_SEED_BASE = 7_700_000
DEFAULT_DRAFT_HOUR = 12
DEFAULT_CALL_BUDGET = 0.85
FAST_TOLERANCE = 1e-4
SELF_CHECK_FACTORS = 16
OPENING_WAIT_SECONDS = 900.0
INERT_PROGRAM = ({'op': 'STOP'},)
WORKER_LOG_RESERVE_BYTES = 2500
# Harness-only speed emulation (preflight --slowdown). The runtime sleeps (R-1)x each
# measured compute segment and logs R in every START line; hosted runs leave it at 1.
EMULATED_SLOWDOWN = 1.0


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _external_imports(path):
    """farm_neat modules imported (at any depth) by a pinned file outside src/."""
    found = set()
    for node in ast.walk(ast.parse(Path(path).read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
            found.update(f'{node.module}.{alias.name}' for alias in node.names)
    return {n for n in found if n == 'farm_neat' or n.startswith('farm_neat.')}


def joint_source_members():
    """Legacy source_members plus the joint runtime closure and every entity_model pin."""
    from farm_neat.evaluation import deployment as legacy, joint_deployment_runtime as runtime
    from farm_neat.tooling import reachability
    entity = runtime.load_private('farmneat_joint_entity_model_build', ROOT/runtime.ENTITY_MODEL)
    pins = entity.sources()
    known = reachability.modules()
    pending = ['farm_neat.evaluation.joint_deployment_runtime', 'farm_neat.evaluation.deployment_codec',
               'farm_neat.training.bc']
    for name in (runtime.ENTITY_MODEL, runtime.ADAPTER):
        pending += sorted(_external_imports(ROOT/name))
    pending += [reachability.module_name(ROOT/p) for p in pins if p.startswith('src/') and p.endswith('.py')]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen or name not in known:
            continue
        seen.add(name)
        for target in reachability.imports(known[name]) | {name.rsplit('.', 1)[0]}:
            while target:
                if target in known:
                    pending.append(target)
                    break
                target = target.rsplit('.', 1)[0] if '.' in target else ''
    paths = set(legacy.source_members()) | set(pins) | {runtime.ENTITY_MODEL, runtime.ADAPTER}
    paths |= {known[n].relative_to(ROOT).as_posix() for n in seen}
    if 'farm_neat.game.exact_engine' in seen:
        paths.update(('src/farm_neat/game/kaggriculture_engine_reference.py', 'src/farm_neat/game/kaggriculture.json'))
    for p in paths:
        f = ROOT/legacy.checked_name(p)
        if f.is_symlink() or not f.is_file() or f.resolve() != f:
            raise ValueError('regular runtime dependency required: ' + p)
    return {p: sha(ROOT/p) for p in sorted(paths)}


def v2_source_members():
    """Files only v2 archives add: this module (the v2 runtime), fast encode, slot95 decoder.

    fast_gdn/fast_moe import only torch and farm_neat.policy.qwen_hybrid (already in
    the joint closure); joint_decode imports only torch.
    """
    from farm_neat.evaluation import deployment as legacy
    for p in V2_SOURCES:
        f = ROOT/legacy.checked_name(p)
        if f.is_symlink() or not f.is_file() or f.resolve() != f:
            raise ValueError('regular v2 runtime dependency required: ' + p)
    return {p: sha(ROOT/p) for p in V2_SOURCES}


def _options(command_budget, publication_steps):
    from farm_neat.evaluation import joint_deployment_runtime as runtime
    options = runtime.default_options()
    options['command_budget'] = command_budget
    options['publication_steps'] = None if publication_steps is None else list(publication_steps)
    return options


def runtime_v2_options(*, cache_k=DEFAULT_CACHE_K, cache_seed_base=DEFAULT_CACHE_SEED_BASE,
                       draft_hour=DEFAULT_DRAFT_HOUR, per_call_budget=DEFAULT_CALL_BUDGET, fast_encode=True,
                       slot95_decoder=False, cache_sha256=None):
    """The complete, versioned v2 option block; every value is explicit and bounded."""
    if type(cache_k) is not int or not 0 <= cache_k <= 64:
        raise ValueError('opening cache K must be an integer in [0, 64]')
    if type(cache_seed_base) is not int or not 0 <= cache_seed_base < 2**62:
        raise ValueError('explicit nonnegative opening seed base required')
    if type(draft_hour) is not int or not 1 <= draft_hour <= 23:
        raise ValueError('draft hour must be an integer in [1, 23]')
    if type(per_call_budget) not in (int, float) or not math.isfinite(per_call_budget) \
            or not 0 < per_call_budget < ACT_TIMEOUT:
        raise ValueError('per-call budget must be in (0, actTimeout): the bank is never charged by waiting')
    if type(fast_encode) is not bool or type(slot95_decoder) is not bool:
        raise ValueError('explicit boolean fast-encode and slot95 flags required')
    if cache_k and (type(cache_sha256) is not str or len(cache_sha256) != 64
                    or any(c not in '0123456789abcdef' for c in cache_sha256)):
        raise ValueError('opening cache content SHA-256 required')
    if not cache_k and cache_sha256 is not None:
        raise ValueError('an empty opening cache has no content SHA-256')
    return {'schema': RUNTIME_V2_SCHEMA,
        'model_load': 'DAEMON_THREAD_STARTED_AT_FIRST_CALL',
        'per_call_budget_seconds': float(per_call_budget),
        'per_call_rule': 'WAIT_FOR_BACKGROUND_WORK_AT_MOST_BUDGET_MINUS_OWN_WORK_THEN_RETURN_EXECUTOR_ACTION',
        'draft_hour': draft_hour,
        'draft_observation': 'LATEST_OWN_SEAT_CLEAN_VIEW_WHEN_THE_WORKER_STARTS_THE_DRAFT',
        'draft_as_of': 'NEXT_PUBLICATION_STEP',
        'draft_publication': 'PROPOSE_PUBLISH_REVALIDATION_AT_THE_NEXT_DAWN',
        'no_draft_at_dawn': 'KEEP_ACTIVE_PLAN_UNLESS_OVERAGE_GUARD_ALLOWS_SYNCHRONOUS_COMPUTE',
        'opening_cache': {'k': cache_k, 'seed_base': cache_seed_base,
                          'member': CACHE_MEMBER if cache_k else None, 'content_sha256': cache_sha256,
                          'key': 'SEAT_AND_PUBLIC_CONFIGURATION_SHA256',
                          'selection': 'EPISODE_SEED_MOD_K',
                          'refusal': 'LIVE_STEP0_INPUT_DIGEST_DIFFERS_THEN_SYNCHRONOUS_COMPUTE'},
        'opening_fallback': 'EXPLICIT_EMPTY_PROGRAM_IF_NO_OPENING_PROGRAM_IS_AVAILABLE',
        'fast_encode': {'requested': fast_encode, 'modules': [FAST_GDN, FAST_MOE], 'tolerance': FAST_TOLERANCE,
                        'self_check_factors': SELF_CHECK_FACTORS,
                        'rule': 'ON_ONLY_IF_LOAD_TIME_SELF_CHECK_PASSES_ELSE_OFF_AND_LOGGED'},
        'decoder': SLOT95_PROFILE if slot95_decoder else REFERENCE_DECODER,
        'decoder_source': SLOT95_DECODER if slot95_decoder else None,
        'decoder_parity_claim': not slot95_decoder}


def _v2_parameters(block):
    """Invert runtime_v2_options; malformed blocks raise ValueError."""
    try:
        cache = block['opening_cache']
        return {'cache_k': cache['k'], 'cache_seed_base': cache['seed_base'], 'draft_hour': block['draft_hour'],
                'per_call_budget': block['per_call_budget_seconds'],
                'fast_encode': block['fast_encode']['requested'],
                'slot95_decoder': block['decoder'] == SLOT95_PROFILE, 'cache_sha256': cache['content_sha256']}
    except (KeyError, TypeError) as error:
        raise ValueError('explicit current v2 runtime options required') from error


def validate_settings_v2(settings):
    """Return (effective profile, v2 options); refuse anything but the current v2 schema."""
    from farm_neat.economy import action_space
    from farm_neat.evaluation import joint_deployment_runtime as runtime
    if (type(settings) is not dict or settings.get('schema') != SCHEMA_V2 or settings.get('arm') != 'joint'
            or settings.get('runtime_profile') != 'v2'):
        raise ValueError('explicit current joint v2 deployment settings required')
    if settings.get('profile') != runtime.default_profile():
        raise ValueError('versioned default joint profile required; cheaper profiles are options')
    effective = runtime.effective_profile(settings['profile'], settings.get('options'))
    if settings.get('effective_profile') != effective:
        raise ValueError('declared effective profile differs from profile+options')
    budget = effective['command_budget']
    if type(budget) is not int or not 2 <= budget <= action_space.MAX_COMMANDS:
        raise ValueError('command budget including forced final STOP required')
    if not runtime._finite(effective['temperature']) or effective['temperature'] <= 0:
        raise ValueError('finite positive temperature required')
    guard = settings.get('overage_guard')
    expected = runtime.default_guard(guard.get('reserve_seconds', -1) if type(guard) is dict else -1)
    if type(guard) is not dict or guard != expected or not runtime._finite(guard['reserve_seconds']) \
            or guard['reserve_seconds'] < 0:
        raise ValueError('explicit current overage guard with a finite nonnegative reserve required')
    block = settings.get('runtime_v2')
    if type(block) is not dict or block != runtime_v2_options(**_v2_parameters(block)):
        raise ValueError('explicit current v2 runtime options required')
    for key, value in (('sampling_rng', runtime.SAMPLING_RNG), ('fallback', FALLBACK_V2),
                       ('log_schema', LOG_SCHEMA_V2), ('actor_kind', runtime.ACTOR_KIND)):
        if settings.get(key) != value:
            raise ValueError('joint v2 deployment contract field changed: ' + key)
    if type(settings.get('fixture')) is not bool or type(settings.get('member_sha256')) is not dict:
        raise ValueError('explicit fixture flag and member pins required')
    pins = settings['member_sha256']
    if any(p not in pins for p in V2_SOURCES) or (block['opening_cache']['k'] and CACHE_MEMBER not in pins):
        raise ValueError('v2 runtime members must be pinned')
    return effective, block


def v2_actor_description():
    return {'class': 'AgentV2', 'source': V2_MODULE, 'model_source': 'work/runs/reviewer-play/public-bc-entity-readout/entity_model.py',
            'view_source': 'work/runs/reviewer-play/public-winner-bc-freerun/checkpoint_adapter.py',
            'input_adapter': 'actor_view -> observation.pack(clean, None, cfg) -> encode -> sample (worker thread)',
            'controller': 'FullSeasonPlan propose/publish at dawns (cache, draft or guarded sync compute); act every step'}


# ----- opening cache and fast-encode self-check (shared by builder and v2 runtime) ------------

def step0_inputs(adapter, codec, engine, seeds=INVARIANCE_SEEDS):
    """Engine initial actor inputs for both seats over several world seeds.

    Returns {(seat, public_configuration_sha256): {input_sha256: (clean, cfg, packed)}}.
    The cache is valid only if every key has exactly one input digest.
    """
    found = {}
    for seed in seeds:
        env = engine.ExactKaggricultureEnvironment(seed=seed)
        for seat in (0, 1):
            clean = adapter.actor_view(env.agent_observation(seat), seat)
            cfg = adapter.public_configuration(env.configuration)
            packed = codec.pack(clean, None, cfg)
            found.setdefault((seat, adapter.digest(cfg)), {})[packed['content_sha256']] = (clean, cfg, packed)
    return found


def build_opening_cache(graph, *, adapter, codec, engine, k, seed_base, command_budget, temperature, identity,
                        seeds=INVARIANCE_SEEDS):
    """K sampled step-0 programs per seat from the deployed (decoded) graph; no program editing."""
    import torch
    found = step0_inputs(adapter, codec, engine, seeds)
    varying = {f'{seat}:{cfg[:12]}': len(v) for (seat, cfg), v in found.items() if len(v) != 1}
    if varying:
        raise ValueError(f'step-0 actor input varies with the world seed {varying}; build with opening cache K=0')
    entries = []
    for (seat, cfg_sha), variants in sorted(found.items()):
        (input_sha, (clean, cfg, packed)), = variants.items()
        batch = codec.collate([packed])
        samples = []
        with torch.no_grad():
            context, values, _ = graph.encode(*(torch.as_tensor(batch[n]) for n in ('rows', 'token_mask', 'knownness')))
            for index in range(k):
                seed = seed_base + index
                generator = torch.Generator(device='cpu').manual_seed(seed)
                records, _, _ = graph.sample(context, generator=generator, max_commands=command_budget,
                                             temperature=temperature, as_of_step=0)
                if len(records) != 1:
                    raise ValueError('exactly one sampled opening record required')
                behavior = records[0]
                samples.append({'index': index, 'seed': seed, 'program': behavior['program'],
                    'program_sha256': adapter.digest(behavior['program']), 'joint_logp': float(behavior['joint_logp']),
                    'factors': len(behavior['selected_logps']), 'record': behavior})
        entries.append({'seat': seat, 'configuration_sha256': cfg_sha, 'observation_sha256': adapter.digest(clean),
            'input_sha256': input_sha, 'observation': clean, 'configuration': cfg,
            'inventory_key_order': [list(inv) for inv in clean['private']['inventories']],
            'value_prediction': float(values[0]), 'entity_tokens': int(batch['token_mask'].sum()),
            'distinct_programs': len({s['program_sha256'] for s in samples}), 'samples': samples})
    digests = {str(e['seat']): e['input_sha256'] for e in entries}
    body = {'schema': CACHE_SCHEMA, **identity, 'k': k, 'seed_base': seed_base, 'temperature': temperature,
        'command_budget': command_budget, 'as_of_step': 0, 'encoder': 'REFERENCE',
        'invariance': {'engine': 'farm_neat.game.exact_engine.ExactKaggricultureEnvironment(seed).agent_observation(seat)',
                       'world_seeds': list(seeds), 'seats': [0, 1], 'input_sha256_by_seat': digests,
                       'seats_differ': len(set(digests.values())) > 1,
                       'rule': 'ONE_ACTOR_INPUT_DIGEST_PER_SEAT_AND_PUBLIC_CONFIGURATION_ACROSS_WORLD_SEEDS'},
        'entries': entries}
    return {**body, 'content_sha256': adapter.digest(body)}


def validate_cache(cache, *, adapter, settings, block, effective):
    """Exact cache bound to this deployment; returns {(seat, cfg_sha): entry}."""
    oc = block['opening_cache']
    body = {k: v for k, v in cache.items() if k != 'content_sha256'} if type(cache) is dict else None
    if (body is None or cache.get('schema') != CACHE_SCHEMA or cache.get('content_sha256') != adapter.digest(body)
            or cache['content_sha256'] != oc['content_sha256'] or cache.get('k') != oc['k']
            or cache.get('seed_base') != oc['seed_base']
            or cache.get('original_model_sha256') != settings['original_model_sha256']
            or cache.get('deployment_sha256') != settings['deployment_sha256']
            or cache.get('decoder') != block['decoder']
            or cache.get('temperature') != effective['temperature']
            or cache.get('command_budget') != effective['command_budget']):
        raise ValueError('opening cache is not bound to this deployment and profile')
    entries = {}
    for entry in cache['entries']:
        key = (entry['seat'], entry['configuration_sha256'])
        if key in entries or len(entry['samples']) != oc['k'] or [s['index'] for s in entry['samples']] != list(range(oc['k'])):
            raise ValueError('one complete K-sample cache entry per seat/configuration required')
        entries[key] = entry
    return entries


def _uninstall_fast(graph):
    from farm_neat.policy import qwen_hybrid as hybrid
    for module in graph.modules():
        if isinstance(module, (hybrid.BidirectionalGDN, hybrid.SparseMoE)) and 'forward' in vars(module):
            del module.forward
    for name in ('_fast_gdn_execution_arm', '_fast_gdn_chunk', '_fast_moe_arm'):
        if name in vars(graph):
            delattr(graph, name)


def _encode_and_walk(graph, tensors, record, factors):
    """Encode, then teacher-force the record's first `factors` factors and keep each legal logp."""
    context, values, _ = graph.encode(*tensors)
    state = context[0]
    budget, temperature, as_of = record['max_commands'], record['temperature'], record.get('as_of_step')
    logps, masks = [], []
    for slot, command in enumerate(record['commands']):
        for factor in command['factors']:
            if len(logps) >= factors:
                break
            logp, _, ids, role, mask = graph.factor(state, factor['path'], tuple(factor['choices']), temperature,
                stop_only=(slot == budget - 1 and factor['path'] == 'op'), as_of_step=as_of)
            logps.append(logp[mask])
            masks.append(mask.tolist())
            state = graph.advance(state, ids[factor['index']], role)
    return {'pooled': context.pooled, 'memory': context.memory, 'mask': context.mask, 'values': values,
            'logps': logps, 'masks': masks}


def _max_abs_diffs(reference, fast):
    mask = reference['mask'][..., None]
    return {'pooled': float((reference['pooled'] - fast['pooled']).abs().max()),
            'memory': float((reference['memory'] - fast['memory']).masked_fill(~mask, 0).abs().max()),
            'value': float((reference['values'] - fast['values']).abs().max()),
            'logp': max((float((a - b).abs().max()) for a, b in zip(reference['logps'], fast['logps'])), default=0.0)}


def fast_self_check(graph, *, gdn, moe, batch, record, tolerance=FAST_TOLERANCE, factors=SELF_CHECK_FACTORS):
    """Reference vs fast (fast_gdn+fast_moe) encode of one input, plus the first factors' logps.

    Leaves fast installed only if every max-abs difference is finite and <= tolerance
    and the legal masks are identical; otherwise the reference forwards are restored.
    """
    import torch
    started = time.perf_counter()
    tensors = [torch.as_tensor(batch[n]) for n in ('rows', 'token_mask', 'knownness')]
    diffs, masks_equal, error = {}, False, None
    with torch.no_grad():
        reference = _encode_and_walk(graph, tensors, record, factors)
        try:
            gdn.install_gdn(graph)
            moe.install_moe(graph)
            fast = _encode_and_walk(graph, tensors, record, factors)
            diffs = _max_abs_diffs(reference, fast)
            masks_equal = reference['masks'] == fast['masks'] and len(fast['logps']) == len(reference['logps'])
        except Exception as exc:  # a failed fast path is a failed self-check, not a crash
            error = f'{type(exc).__name__}: {str(exc)[:200]}'
    passed = (error is None and masks_equal and bool(diffs)
              and all(math.isfinite(v) and v <= tolerance for v in diffs.values()))
    if not passed:
        _uninstall_fast(graph)
    return {'schema': SELF_CHECK_SCHEMA, 'passed': passed, 'installed': passed, 'tolerance': tolerance,
            'factors_compared': len(reference['logps']), 'max_abs': diffs, 'masks_equal': masks_equal,
            'error': error, 'seconds': time.perf_counter() - started}


def build_joint(model, *, model_sha256, output, command_budget=None, publication_steps=None,
                reserve_seconds=20.0, fixture=False, profile='v1', opening_cache_k=DEFAULT_CACHE_K,
                opening_seed_base=DEFAULT_CACHE_SEED_BASE, draft_hour=DEFAULT_DRAFT_HOUR,
                per_call_budget=DEFAULT_CALL_BUDGET, fast_encode=True, slot95_decoder=False):
    """Encode a joint checkpoint and write a hash-pinned archive; no upload.

    fixture=True is Python/test-only: it admits tiny graphs and is recorded in the
    settings, the manifest and every START log line. profile='v1' (default) writes
    exactly the pre-v2 archive; the remaining keyword options exist only for v2.
    """
    from farm_neat.evaluation import deployment as legacy, deployment_codec as codec
    from farm_neat.evaluation import joint_deployment_runtime as runtime
    if profile not in PROFILES:
        raise ValueError('runtime profile v1 or v2 required')
    v2 = profile == 'v2'
    v2_arguments = {'cache_k': opening_cache_k, 'cache_seed_base': opening_seed_base, 'draft_hour': draft_hour,
                    'per_call_budget': per_call_budget, 'fast_encode': fast_encode, 'slot95_decoder': slot95_decoder}
    if v2:
        runtime_v2_options(**v2_arguments, cache_sha256='0'*64 if opening_cache_k else None)
    elif v2_arguments != {'cache_k': DEFAULT_CACHE_K, 'cache_seed_base': DEFAULT_CACHE_SEED_BASE,
                          'draft_hour': DEFAULT_DRAFT_HOUR, 'per_call_budget': DEFAULT_CALL_BUDGET,
                          'fast_encode': True, 'slot95_decoder': False}:
        raise ValueError('v2 runtime options require profile v2')
    if type(fixture) is not bool:
        raise ValueError('explicit boolean fixture flag required')
    output = Path(output).absolute()
    receipt = Path(str(output) + '.json')
    if output.exists() or receipt.exists() or output.is_symlink() or output.parent.resolve() != output.parent:
        raise ValueError('new non-symlinked archive output required')
    model = Path(model).absolute()
    sidecar = Path(str(model) + '.json')
    for path in (model, sidecar):
        if path.is_symlink() or not path.is_file():
            raise ValueError('regular checkpoint and manifest required: ' + str(path))
    sidecar_bytes = sidecar.read_bytes()
    original = json.loads(sidecar_bytes)
    if original.get('arm') != 'joint' or original.get('schema') != runtime.JOINT_SCHEMA:
        raise ValueError('JOINT EntityGraph checkpoint required; pooled/cached manifests refused')
    report = original.get('parameter_report', {})
    if not fixture and (report.get('total_trainable_parameters', 0) < 30_000_000 or report.get('layers', 0) < 14):
        raise ValueError('production >=30M/>=14-layer joint checkpoint required')
    entity = runtime.load_private('farmneat_joint_entity_model_build', ROOT/runtime.ENTITY_MODEL)
    runtime.check_original(original, entity)
    if not fixture and original.get('optimizer_steps', 0) < 1:
        raise ValueError('trained joint checkpoint required')
    if sha(model) != model_sha256:
        raise ValueError('model file SHA mismatch')
    settings = {'schema': runtime.SCHEMA, 'arm': 'joint', 'actor_kind': runtime.ACTOR_KIND,
        'actor': {'class': 'LiveBCActor', 'source': runtime.ADAPTER, 'model_source': runtime.ENTITY_MODEL,
                  'input_adapter': 'actor_view -> observation.pack(clean, None, cfg) -> encode -> sample',
                  'controller': 'FullSeasonPlan propose/publish at admitted dawns; controller(clean, cfg) every step'},
        'profile': runtime.default_profile(), 'options': _options(command_budget, publication_steps),
        'overage_guard': runtime.default_guard(reserve_seconds),
        'threads': {'rule': 'MIN(OS_CPU_COUNT, 4)', 'cap': runtime.THREAD_CAP},
        'sampling_rng': runtime.SAMPLING_RNG, 'fallback': runtime.FALLBACK, 'log_schema': runtime.LOG_SCHEMA,
        'fixture': fixture, 'member_sha256': {}}
    settings['effective_profile'] = runtime.effective_profile(settings['profile'], settings['options'])
    if v2:
        settings.update(schema=SCHEMA_V2, runtime_profile='v2', actor=v2_actor_description(), fallback=FALLBACK_V2,
                        log_schema=LOG_SCHEMA_V2, runtime_v2=runtime_v2_options(**v2_arguments,
                            cache_sha256='0'*64 if opening_cache_k else None),
                        member_sha256={p: '0'*64 for p in V2_SOURCES} | ({CACHE_MEMBER: '0'*64} if opening_cache_k else {}))
    validate = validate_settings_v2 if v2 else runtime.validate_settings
    validate(settings)
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only export required')
    graph, loaded = entity.load_model(model, device='cpu', expected_file_sha256=model_sha256, expected_arm='joint')
    if loaded != original:
        raise ValueError('checkpoint manifest changed while loading')
    payload, deployment = codec.encode_state(graph.state_dict(), runtime.codec_view(original),
        original_checkpoint_sha256=model_sha256, expected_identity=codec.identity())
    del graph
    pins = joint_source_members()
    if v2:
        pins = {**pins, **v2_source_members()}
    byte_members = {'MODEL.bin': payload, 'MODEL.bin.json': encoded(deployment),
                    runtime.ORIGINAL_MANIFEST: sidecar_bytes}
    settings.update(original_model_sha256=model_sha256,
        original_manifest_file_sha256=hashlib.sha256(sidecar_bytes).hexdigest(),
        original_manifest_content_sha256=original['content_sha256'],
        optimizer_steps=original['optimizer_steps'], parameter_report=original['parameter_report'],
        deployment_sha256=deployment['content_sha256'], tensor_encoding=deployment['sizing'],
        numerical_profile='GROUP64_EXPERT_INT8_STORAGE_DECODED_FP32_CPU; TORCH_GENERATOR_SAMPLING',
        cpu_cuda_parity_claim=False, original_quantized_parity_claim=False, hosted_acceptance='UNVERIFIED',
        member_sha256={**pins, **{n: hashlib.sha256(b).hexdigest() for n, b in byte_members.items()}})
    v2_build = None
    if v2:
        cache, v2_build = _v2_build_artifacts(settings, byte_members, entity, output=output, **v2_arguments)
        if cache is not None:
            byte_members[CACHE_MEMBER] = encoded(cache)
        settings['runtime_v2'] = runtime_v2_options(**v2_arguments,
            cache_sha256=None if cache is None else cache['content_sha256'])
        settings['member_sha256'] = {**pins, **{n: hashlib.sha256(b).hexdigest() for n, b in byte_members.items()}}
    validate(settings)
    byte_members['DEPLOYMENT.json'] = encoded(settings)
    settings_sha = hashlib.sha256(byte_members['DEPLOYMENT.json']).hexdigest()
    byte_members['main.py'] = main_source_v2(settings_sha) if v2 else runtime.main_source(settings_sha)
    members = {**{n: ROOT/n for n in pins}, **byte_members}
    inventory = {n: {'sha256': sha(v) if isinstance(v, Path) else hashlib.sha256(v).hexdigest(),
                     'bytes': v.stat().st_size if isinstance(v, Path) else len(v)} for n, v in members.items()}
    manifest = {'schema': 'farmneat-deployment-archive-v1',
        'actor_kind': runtime.ACTOR_KIND, 'members': inventory, 'original_model_sha256': model_sha256,
        'deployment_sha256': deployment['content_sha256'], 'settings_sha256': settings_sha, 'fixture': fixture,
        'archive_cap_bytes': ARCHIVE_CAP_BYTES, 'hosted_acceptance': 'UNVERIFIED'}
    if v2:
        manifest['runtime_profile'] = 'v2'
    members['MANIFEST.json'] = encoded(manifest)
    expanded = sum(v.stat().st_size if isinstance(v, Path) else len(v) for v in members.values())
    if len(members) > MEMBER_CAP or expanded > ARCHIVE_CAP_BYTES:
        raise ValueError(f'archive bounds exceeded: {len(members)} members, {expanded} expanded bytes')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    if temporary.exists() or temporary.is_symlink():
        raise ValueError('new build temporary required')
    try:
        legacy.write_tar(temporary, members)
        if temporary.stat().st_size > ARCHIVE_CAP_BYTES:
            raise ValueError('compressed archive exceeds 100,000,000 bytes')
        if sha(model) != model_sha256 or sidecar.read_bytes() != sidecar_bytes or \
                any(sha(ROOT/n) != h for n, h in pins.items()):
            raise ValueError('checkpoint or source closure changed during build')
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result = {'schema': BUILD_SCHEMA, 'archive': str(output), 'archive_sha256': sha(output),
        'archive_bytes': output.stat().st_size, 'expanded_bytes': expanded, 'members': len(members),
        'source_members': len(pins), 'model': str(model), 'original_model_sha256': model_sha256,
        'settings_sha256': settings_sha, 'deployment_sha256': deployment['content_sha256'],
        'optimizer_steps': original['optimizer_steps'],
        'parameters': original['parameter_report']['total_trainable_parameters'],
        'effective_profile': settings['effective_profile'], 'options': settings['options'],
        'overage_guard': settings['overage_guard'], 'tensor_encoding': deployment['sizing'], 'fixture': fixture,
        'builder_sha256': sha(__file__), 'runtime_sha256': pins['src/farm_neat/evaluation/joint_deployment_runtime.py'],
        'runtime_profile': profile, 'preflight': 'REQUIRED', 'hosted_acceptance': 'UNVERIFIED'}
    if v2:
        result.update(runtime_v2=settings['runtime_v2'], v2_build=v2_build)
    receipt.write_bytes(encoded(result))
    return result


def _v2_build_artifacts(settings, byte_members, entity, *, output, cache_k, cache_seed_base, draft_hour,
                        per_call_budget, fast_encode, slot95_decoder):
    """Decode the payload exactly as the runtime does, then sample the opening cache.

    The cache and the build-time self-check use the DEPLOYED (int8-decoded FP32)
    weights through the v1 runtime's load_graph, never the original checkpoint.
    """
    import torch
    from farm_neat.evaluation import joint_deployment_runtime as runtime
    from farm_neat.game import exact_engine as engine
    from farm_neat.policy import observation as codec
    adapter = runtime.load_private('farmneat_joint_checkpoint_adapter_build', ROOT/runtime.ADAPTER)
    with tempfile.TemporaryDirectory(prefix='joint-v2-build-', dir=output.parent) as tmp:
        for name in MODEL_MEMBERS:
            (Path(tmp)/name).write_bytes(byte_members[name])
        graph, original = runtime.load_graph(tmp, settings, entity)
    decoder = REFERENCE_DECODER
    if slot95_decoder:
        runtime.load_private('farmneat_joint_decode_build', ROOT/SLOT95_DECODER).install(graph, SLOT95_PROFILE)
        decoder = SLOT95_PROFILE
    effective = settings['effective_profile']
    identity = {'original_model_sha256': settings['original_model_sha256'],
                'deployment_sha256': settings['deployment_sha256'],
                'architecture_sha256': original['architecture']['content_sha256'], 'decoder': decoder}
    cache = None
    if cache_k:
        started = time.perf_counter()
        cache = build_opening_cache(graph, adapter=adapter, codec=codec, engine=engine, k=cache_k,
            seed_base=cache_seed_base, command_budget=effective['command_budget'],
            temperature=effective['temperature'], identity=identity)
        cache_seconds = time.perf_counter() - started
    summary = {'decoder': decoder,
        'opening_cache': None if cache is None else {'k': cache_k, 'content_sha256': cache['content_sha256'],
            'input_sha256_by_seat': cache['invariance']['input_sha256_by_seat'],
            'seats_differ': cache['invariance']['seats_differ'], 'world_seeds': cache['invariance']['world_seeds'],
            'distinct_programs_by_seat': {str(e['seat']): e['distinct_programs'] for e in cache['entries']},
            'build_seconds': cache_seconds},
        'fast_self_check_at_build': None}
    if fast_encode:
        if cache is not None:
            entry = cache['entries'][0]
            packed, record = codec.pack(entry['observation'], None, entry['configuration']), entry['samples'][0]['record']
        else:
            found = step0_inputs(adapter, codec, engine, seeds=(0,))
            (_, (_, _, packed)), = found[min(found)].items()
            with torch.no_grad():
                batch = codec.collate([packed])
                context, _, _ = graph.encode(*(torch.as_tensor(batch[n]) for n in ('rows', 'token_mask', 'knownness')))
                record = graph.sample(context, generator=torch.Generator(device='cpu').manual_seed(0),
                    max_commands=effective['command_budget'], temperature=effective['temperature'], as_of_step=0)[0][0]
        gdn = runtime.load_private('farmneat_joint_fast_gdn_build', ROOT/FAST_GDN)
        moe = runtime.load_private('farmneat_joint_fast_moe_build', ROOT/FAST_MOE)
        summary['fast_self_check_at_build'] = fast_self_check(graph, gdn=gdn, moe=moe, batch=codec.collate([packed]),
                                                              record=record)
    del graph
    return cache, summary


# ----- runtime profile v2 -------------------------------------------------------------------

def _emulated_slowdown():
    value = globals().get('EMULATED_SLOWDOWN', 1.0)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 1:
        raise ValueError('emulated slowdown must be a finite number >= 1')
    return float(value)


def _dist_version(name):
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        return None


def cgroup_cpu(base='/sys/fs/cgroup'):
    """(raw quota text, quota in CPUs or None); cgroup v2 cpu.max, else the v1 CFS files."""
    base = Path(base)
    try:
        raw = (base/'cpu.max').read_text().strip()
        quota, period = raw.split()[:2]
        return raw[:64], None if quota == 'max' else round(int(quota)/int(period), 3)
    except Exception:
        pass
    try:
        quota = int((base/'cpu'/'cpu.cfs_quota_us').read_text())
        period = int((base/'cpu'/'cpu.cfs_period_us').read_text())
        return f'v1:{quota} {period}', None if quota < 0 else round(quota/period, 3)
    except Exception:
        return None, None


def _error_text(error):
    return None if error is None else f'{type(error).__name__}: {str(error)[:200]}'


class _Job:
    """One model computation owned by the worker thread; the main thread only reads it after done."""
    __slots__ = ('kind', 'episode', 'dawn', 'fixed', 'queued_step', 'done', 'status', 'result', 'error',
                 'started', 'cancelled', 'published_step')

    def __init__(self, kind, episode, dawn, *, fixed=None, queued_step=None):
        self.kind, self.episode, self.dawn, self.fixed, self.queued_step = kind, episode, dawn, fixed, queued_step
        self.done = threading.Event()
        self.status, self.result, self.error, self.started = 'QUEUED', None, None, None
        self.cancelled, self.published_step = False, None

    def record(self):
        result = self.result or {}
        return {'kind': self.kind, 'dawn': self.dawn, 'queued_step': self.queued_step, 'status': self.status,
                'observation_step': result.get('observation_step'), 'seconds': result.get('seconds'),
                'raw_seconds': result.get('raw_seconds'), 'encode_seconds': result.get('encode_seconds'),
                'sample_seconds': result.get('sample_seconds'), 'published_step': self.published_step,
                'error': _error_text(self.error)}


class ModelWorker:
    """Daemon thread that loads the model once, runs the fast self-check, then serves jobs.

    Only this thread touches torch, the graph and each episode's generator.
    """

    def __init__(self, agent):
        self.agent = agent
        self.cond = threading.Condition()
        self.queue = collections.deque()
        self.events = collections.deque(maxlen=256)
        self.state, self.running, self.thread = 'NOT_STARTED', None, None
        self.graph = self.entity = self.original = None
        self.load = {'state': 'NOT_STARTED'}

    def start(self):
        with self.cond:
            if self.thread is not None:
                return False
            self.state = 'LOADING'
            self.load['state'] = 'LOADING'
            self.thread = threading.Thread(target=self._run, name='farmneat-joint-model', daemon=True)
            self.thread.start()
            return True

    def _busy(self):
        return self.state == 'LOADING' or self.running is not None or any(not j.cancelled for j in self.queue)

    def busy(self):
        with self.cond:
            return self._busy()

    def wait_idle(self, timeout):
        with self.cond:
            return self.cond.wait_for(lambda: not self._busy(), timeout)

    def wait_ready(self, timeout):
        with self.cond:
            return self.cond.wait_for(lambda: self.state in ('READY', 'FAILED'), timeout)

    def submit(self, job):
        with self.cond:
            self.queue.append(job)
            self.cond.notify_all()

    def cancel(self, job, status='CANCELLED'):
        """Cancel a job the worker has not started; returns whether it was cancelled."""
        with self.cond:
            if job.started is not None or job.done.is_set():
                return False
            job.cancelled, job.status = True, status
            try:
                self.queue.remove(job)
            except ValueError:
                pass
            job.done.set()
            self.cond.notify_all()
            return True

    def _run(self):
        try:
            self._load()
            state = 'READY'
        except BaseException as error:
            state = 'FAILED'
            self.load['error'] = _error_text(error)
            self.events.append({'w': 'LOAD_FAILED', 'err': self.load['error']})
        with self.cond:
            self.state = state
            self.load['state'] = state
            self.cond.notify_all()
        while True:
            with self.cond:
                self.cond.wait_for(lambda: bool(self.queue))
                job = self.queue.popleft()
                if job.cancelled:
                    continue
                self.running, job.started, job.status = job, time.perf_counter(), 'RUNNING'
            try:
                if state != 'READY':
                    raise RuntimeError('model unavailable: ' + str(self.load.get('error')))
                self._compute(job)
            except BaseException as error:
                job.error, job.status = error, 'FAILED'
                episode = job.episode
                if episode is not None:
                    episode.errors.append({'step': job.dawn, 'stage': job.kind, 'type': type(error).__name__,
                                           'message': str(error)[:300]})
                self.events.append({'w': job.kind + '_FAILED', 'dawn': job.dawn, 'err': _error_text(error)})
            finally:
                with self.cond:
                    self.running = None
                    job.done.set()
                    self.cond.notify_all()

    def _load(self):
        agent = self.agent
        runtime, settings = agent.rt, agent.settings
        started = time.perf_counter()
        runtime.verify_members(agent.root, {n: settings['member_sha256'][n] for n in MODEL_MEMBERS})
        import torch
        if torch.cuda.is_initialized() or torch.is_autocast_enabled('cpu'):
            raise ValueError('CPU FP32 deployment required')
        hardware = runtime.configure_threads(torch)
        entity = runtime.load_private('farmneat_joint_entity_model', agent.root/runtime.ENTITY_MODEL)
        graph, original = runtime.load_graph(agent.root, settings, entity)
        if agent.v2['decoder'] == SLOT95_PROFILE:
            runtime.load_private('farmneat_joint_decode', agent.root/SLOT95_DECODER).install(graph, SLOT95_PROFILE)
        raw = time.perf_counter() - started
        agent.emulate(raw)
        self.graph, self.entity, self.original = graph, entity, original
        self.load.update(seconds=time.perf_counter() - started, raw_seconds=raw, hardware=hardware,
                         torch_imported=str(torch.__version__), decoder=agent.v2['decoder'])
        self.events.append({'w': 'LOAD', 'sec': round(self.load['seconds'], 3), 'raw': round(raw, 3),
                            'th': hardware['threads'], 'torch': str(torch.__version__), 'dec': agent.v2['decoder'],
                            'rss_mb': agent.rt.rss_mb()})
        fast = {'requested': agent.v2['fast_encode']['requested'], 'installed': False}
        if fast['requested']:
            try:
                check = self._self_check(torch)
            except Exception as error:  # a broken self-check disables fast encode, never the model
                _uninstall_fast(graph)
                check = {'passed': False, 'installed': False, 'reason': 'SELF_CHECK_ERROR', 'error': _error_text(error)}
            fast.update(check or {'passed': False, 'reason': 'NO_SELF_CHECK_INPUT'})
            fast['installed'] = bool(check and check['installed'])
            self.events.append({'w': 'FAST', 'on': fast['installed'], 'pass': fast.get('passed'),
                                'why': fast.get('reason') or fast.get('error'),
                                'd': {k: float(f'{v:.3g}') for k, v in (fast.get('max_abs') or {}).items()},
                                'sec': round(fast.get('seconds') or 0.0, 3)})
        self.load['fast_encode'] = fast

    def _self_check(self, torch):
        agent = self.agent
        codec = agent.codec
        entry, record = agent.self_check_entry(), None
        if entry is not None:
            observation, configuration = entry['observation'], entry['configuration']
            record = entry['samples'][0]['record']
        else:
            episode = agent.ep
            opening = None if episode is None else episode.opening_input
            if opening is None:
                return None
            observation, configuration = opening
        packed = codec.pack(observation, None, configuration)
        if entry is not None and packed['content_sha256'] != entry['input_sha256']:
            return {'passed': False, 'installed': False, 'reason': 'CACHED_INPUT_DIGEST_MISMATCH'}
        batch = codec.collate([packed])
        if record is None:
            with torch.no_grad():
                context, _, _ = self.graph.encode(*(torch.as_tensor(batch[n]) for n in ('rows', 'token_mask', 'knownness')))
                record = self.graph.sample(context, generator=torch.Generator(device='cpu').manual_seed(0),
                    max_commands=agent.profile['command_budget'], temperature=agent.profile['temperature'],
                    as_of_step=0)[0][0]
        started = time.perf_counter()
        gdn = agent.rt.load_private('farmneat_joint_fast_gdn', agent.root/FAST_GDN)
        moe = agent.rt.load_private('farmneat_joint_fast_moe', agent.root/FAST_MOE)
        fast = agent.fast_encode_fields
        check = fast_self_check(self.graph, gdn=gdn, moe=moe, batch=batch, record=record,
                                tolerance=fast['tolerance'], factors=fast['self_check_factors'])
        agent.emulate(time.perf_counter() - started)
        check['seconds'] = time.perf_counter() - started
        check['input_sha256'] = packed['content_sha256']
        return check

    def _compute(self, job):
        agent, episode = self.agent, job.episode
        if episode is not agent.ep:
            job.status = 'STALE_EPISODE'
            return
        if job.fixed is not None:
            step, clean, cfg = job.fixed
        else:
            with episode.lock:
                step, clean, cfg = episode.latest
            if step >= job.dawn:
                job.status = 'STALE'
                return
        import torch
        started = time.perf_counter()
        codec = agent.codec
        packed = codec.pack(clean, None, cfg)
        batch = codec.collate([packed])
        if episode.generator is None:
            episode.generator = torch.Generator(device='cpu').manual_seed(episode.seed)
        generator = episode.generator
        rng_before = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
        with torch.no_grad():
            encode_started = time.perf_counter()
            context, values, _ = self.graph.encode(*(torch.as_tensor(batch[n]) for n in ('rows', 'token_mask', 'knownness')))
            sample_started = time.perf_counter()
            records, _, _ = self.graph.sample(context, generator=generator, max_commands=agent.profile['command_budget'],
                temperature=agent.profile['temperature'], as_of_step=job.dawn)
            sample_ended = time.perf_counter()
        if len(records) != 1:
            raise ValueError('exactly one sampled behavior record per decision required')
        behavior = records[0]
        raw = time.perf_counter() - started
        agent.emulate(raw)
        seconds = time.perf_counter() - started
        job.result = {'program': behavior['program'], 'program_sha256': agent.adapter.digest(behavior['program']),
            'joint_logp': float(behavior['joint_logp']), 'factors': len(behavior['selected_logps']),
            'value': float(values[0]), 'input_sha256': packed['content_sha256'],
            'record_sha256': agent.adapter.digest(behavior), 'observation_step': step, 'as_of_step': job.dawn,
            'entity_tokens': int(batch['token_mask'].sum()), 'encode_seconds': sample_started - encode_started,
            'sample_seconds': sample_ended - sample_started, 'raw_seconds': raw, 'seconds': seconds,
            'rng_before': rng_before,
            'rng_after': hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()}
        job.status = 'OK'
        episode.compute_seconds.append(seconds)
        self.events.append({'w': job.kind, 'dawn': job.dawn, 'from': step, 'sec': round(seconds, 3),
                            'enc': round(job.result['encode_seconds'], 3), 'smp': round(job.result['sample_seconds'], 3)})


class EpisodeV2:
    def __init__(self, index, seat, seed, steps, controller):
        self.index, self.seat, self.seed = index, seat, seed
        self.steps = tuple(steps)
        self.controller = controller
        self.lock = threading.Lock()
        self.latest = self.generator = self.opening = self.opening_input = None
        self.last_step = -1
        self.drafts = {}
        self.compute_seconds = []
        self.publications, self.skips, self.errors, self.repairs, self.calls = [], [], [], [], []
        self.log_failures = 0
        self.ended = False


class AgentV2:
    """Per-episode actor over one background-loaded joint graph; never raises to the host."""

    def __init__(self, root, *, expected_settings_sha256, started=None, entropy=os.urandom, clock=time.perf_counter):
        from farm_neat.evaluation import joint_deployment_runtime as runtime
        self.rt, self.clock, self.entropy = runtime, clock, entropy
        started = clock() if started is None else started
        self.root = Path(root).resolve()
        raw = (self.root/'DEPLOYMENT.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_settings_sha256:
            raise ValueError('deployment settings identity changed')
        self.settings = json.loads(raw)
        self.settings_sha256 = expected_settings_sha256
        self.profile, self.v2 = validate_settings_v2(self.settings)
        self.guard = self.settings['overage_guard']
        self.fast_encode_fields = self.v2['fast_encode']
        runtime.verify_members(self.root, {n: h for n, h in self.settings['member_sha256'].items()
                                           if n not in MODEL_MEMBERS})
        # Every module the main thread needs is imported here, before the worker thread exists.
        import numpy
        from farm_neat.training import bc  # noqa: F401  (actor_view imports it lazily)
        from farm_neat.economy import action_space, orders  # noqa: F401
        from farm_neat.policy import observation as codec
        self.codec, self.FullSeasonPlan = codec, orders.FullSeasonPlan
        adapter_path = self.root/runtime.ADAPTER
        if adapter_path.is_symlink() or not adapter_path.is_file() or adapter_path.resolve() != adapter_path:
            raise ValueError('regular archived actor source required: ' + runtime.ADAPTER)
        self.adapter = runtime.load_private('farmneat_joint_checkpoint_adapter', adapter_path)
        self.cache_entries = {}
        if self.v2['opening_cache']['k']:
            cache = json.loads((self.root/CACHE_MEMBER).read_bytes())
            self.cache_entries = validate_cache(cache, adapter=self.adapter, settings=self.settings, block=self.v2,
                                                effective=self.profile)
        self.slowdown = _emulated_slowdown()
        cpu_max, quota = cgroup_cpu()
        count = os.cpu_count() or 1
        try:
            affinity = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            affinity = None
        self.hardware = {'cpu_count': count, 'affinity': affinity, 'cpu_max': cpu_max, 'cpu_quota': quota,
                         'threads_planned': max(1, min(count, runtime.THREAD_CAP))}
        self.runtime_info = {'python': sys.version.split()[0], 'torch': _dist_version('torch'),
                             'numpy': str(numpy.__version__)}
        self.worker = ModelWorker(self)
        self.cold_seconds = clock() - started
        self.cold_reported = False
        self.episode_index = -1
        self.ep = None
        self.orphan_errors = []
        self._budget = runtime.DEFAULT_LOG_BYTES_PER_CALL
        self._call = {}

    # ----- small helpers -----------------------------------------------------------------
    def emulate(self, seconds):
        """Harness-only: stretch a measured compute segment to R x its duration."""
        if self.slowdown > 1 and seconds > 0:
            time.sleep((self.slowdown - 1.0)*seconds)

    def self_check_entry(self):
        if not self.cache_entries:
            return None
        seat = None if self.ep is None else self.ep.seat
        matching = [e for (s, _), e in sorted(self.cache_entries.items()) if s == seat]
        return (matching or [e for _, e in sorted(self.cache_entries.items())])[0]

    def wait_ready(self, timeout=None):
        """Start the model load if needed and wait for it (tests and diagnostics)."""
        self.worker.start()
        return self.worker.wait_ready(timeout)

    def _block(self, job, timeout):
        """Wait for a worker job inside a call; the wait is excluded from own-work emulation."""
        began = self.clock()
        try:
            return job.done.wait(timeout)
        finally:
            self._call['blocked'] = self._call.get('blocked', 0.0) + (self.clock() - began)

    def _write(self, value, *, optional=()):
        """One bounded ASCII JSON line; optional keys drop first; failures never propagate."""
        tag = self.rt.LOG_TAG
        try:
            value = {tag: 1, **value}
            line = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n'
            for key in optional:
                if len(line) <= self._budget:
                    break
                if key in value:
                    size = len(json.dumps(value[key], separators=(',', ':'), ensure_ascii=True, allow_nan=False))
                    value = {**{k: v for k, v in value.items() if k != key}, key + '_omitted_bytes': size}
                    line = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                                      allow_nan=False) + '\n'
            if len(line) > self._budget:
                raise ValueError('log line exceeds per-call budget')
            stream = sys.stderr  # the host may replace streams between calls
            stream.write(line)
            stream.flush()
            self._budget -= len(line)
            return True
        except Exception:
            if self.ep is not None:
                self.ep.log_failures += 1
            return False

    def _record_error(self, step, error, stage):
        target = self.ep.errors if self.ep is not None else self.orphan_errors
        record = {'step': step, 'stage': stage, 'type': type(error).__name__}
        if len(target) < self.rt.MAX_ERROR_DETAILS:
            record['message'] = str(error)[:300]
        target.append(record)
        return record

    # ----- episode control ---------------------------------------------------------------
    def _start_episode(self, seat):
        self.episode_index += 1
        self.ep = None
        seed = int.from_bytes(self.entropy(8), 'little') & ((1 << 63) - 1)
        steps = list(self.profile['publication_steps'])
        controller = self.FullSeasonPlan(f'public-bc-freerun-seat-{seat}', publication_steps=steps)
        ep = EpisodeV2(self.episode_index, seat, seed, steps, controller)
        self.ep = ep
        fast = self.v2['fast_encode']
        self._write({'e': 'S', 's': 0, 'schema': LOG_SCHEMA_V2, 'rp': 'v2', 'ep': ep.index, 'seat': seat, 'seed': seed,
            'seed_src': 'OS_ENTROPY', 'cold_s': None if self.cold_reported else round(self.cold_seconds, 3),
            'settings': self.settings_sha256, 'deployment': self.settings['deployment_sha256'],
            'model': self.settings['original_model_sha256'], 'fixture': self.settings['fixture'],
            **self.runtime_info, **self.hardware, 'dawns': steps, 'budget': self.profile['command_budget'],
            'temp': self.profile['temperature'],
            'options': {k: v for k, v in self.settings['options'].items() if k != 'schema'},
            'reserve': self.guard['reserve_seconds'], 'safety': self.guard['safety_factor'],
            'pcb': self.v2['per_call_budget_seconds'], 'dh': self.v2['draft_hour'],
            'k': self.v2['opening_cache']['k'], 'fast': fast['requested'], 'dec': self.v2['decoder'],
            'slow': self.slowdown, 'model_state': self.worker.state, 'rss_mb': self.rt.rss_mb()})
        self.cold_reported = True

    def _observe(self, ep, clean, step):
        """Observe; if the executor refuses (a lost/failed step), drop its receipt and retry once.

        The only state removed is the executor's pending-action receipt (its credit
        settlement for the last issued action). Orders, rules, stages, past segments
        and the published plan are unchanged; the drop is recorded.
        """
        try:
            ep.controller.observe(clean, return_snapshot=False)
            return None
        except Exception as first:
            pending = ep.controller.executor.pending
            if pending is None:
                raise
            state = ep.controller.snapshot()
            state['executor']['pending'] = None
            repaired = self.FullSeasonPlan.restore(state)
            repaired.observe(clean, return_snapshot=False)
            ep.controller = repaired
            record = {'step': step, 'dropped_pending_step': pending.get('step'),
                      'pending_was_settled': bool(pending.get('settled_observation_sha256')),
                      'reason': str(first)[:200]}
            ep.repairs.append(record)
            self._call['repair'] = record
            return record

    def _act(self, observation, configuration, started):
        step, seat = observation.get('step'), observation.get('player')
        if type(step) is not int or type(seat) is not int:
            raise ValueError('integer step and player required')
        if step == 0:
            self._start_episode(seat)
            self._call['started'] = True
        ep = self.ep
        if ep is None or ep.seat != seat:
            raise ValueError('no active episode for this seat')
        self._call['step'] = step
        clean = self.adapter.actor_view(observation, seat)
        cfg = self.adapter.public_configuration(configuration)
        with ep.lock:
            ep.latest = (step, clean, cfg)
        self._observe(ep, clean, step)
        kind, detail = 'T', {}
        if step in ep.steps:
            kind, detail = self._dawn(ep, step, clean, cfg, observation, configuration, started)
        if ep.controller.committed_step is None:
            raise ValueError('no published plan to execute')
        action = ep.controller.act(clean, cfg)
        ep.last_step = step
        self._schedule_draft(ep, step)
        return action, kind, detail

    # ----- dawns -------------------------------------------------------------------------
    def _opening(self, ep, clean, cfg):
        packed = self.codec.pack(clean, None, cfg)
        live = packed['content_sha256']
        ep.opening_input = (clean, cfg)
        refusal = None
        entry = self.cache_entries.get((ep.seat, self.adapter.digest(cfg)))
        if not self.cache_entries:
            refusal = 'CACHE_EMPTY'
        elif entry is None:
            refusal = 'CACHE_MISS'
        elif entry['input_sha256'] != live:
            refusal = 'CACHE_DIGEST_MISMATCH'
        if refusal is None:
            index = ep.seed % self.v2['opening_cache']['k']
            sample = entry['samples'][index]
            return {'source': 'CACHE', 'cache_index': index, 'cache_seed': sample['seed'], 'program': sample['program'],
                    'program_sha256': sample['program_sha256'], 'joint_logp': sample['joint_logp'],
                    'factors': sample['factors'], 'value': entry['value_prediction'], 'input_sha256': live,
                    'as_of_step': 0}
        self.worker.start()
        job = _Job('OPENING', ep, 0, fixed=(0, clean, cfg), queued_step=0)
        self.worker.submit(job)
        if not self._block(job, OPENING_WAIT_SECONDS):
            self.worker.cancel(job, 'TIMEOUT')
        if job.status == 'OK':
            return {**job.result, 'source': 'SYNC', 'refusal': refusal}
        return {'source': 'INERT', 'program': list(INERT_PROGRAM), 'refusal': refusal,
                'compute_status': job.status, 'compute_error': _error_text(job.error)}

    def _dawn_program(self, ep, step, clean, cfg, bank, act_timeout, started):
        job = ep.drafts.get(step)
        if job is not None and not job.done.is_set() and job.started is not None:
            remaining = self.v2['per_call_budget_seconds'] - (self.clock() - started)
            if remaining > 0:
                self._block(job, remaining)
        if job is not None and job.status == 'OK':
            return job, {'draft': 'READY'}
        if job is not None and job.started is None:
            self.worker.cancel(job, 'NOT_STARTED_BY_DAWN')
        status = 'NONE' if job is None else job.status
        if self.worker.state != 'READY':
            return None, {'reason': 'MODEL_' + ('FAILED' if self.worker.state == 'FAILED' else 'NOT_READY'),
                          'draft': status}
        publish, detail = self.rt.guard_decision(step=step, bank=bank, publication_seconds=ep.compute_seconds,
                                                  act_timeout=act_timeout, guard=self.guard)
        detail = {**detail, 'draft': status}
        if not publish:
            return None, detail
        basis = max(ep.compute_seconds) if ep.compute_seconds else self.guard['prior_publication_seconds']
        limit = self.guard['safety_factor']*basis + 1.0
        if job is not None and job.started is not None and not job.done.is_set():
            self._block(job, limit)
            if job.status == 'OK':
                return job, {**detail, 'draft': 'LATE'}
            if not job.done.is_set():
                return None, {**detail, 'reason': 'DRAFT_TIMEOUT'}
        sync = _Job('SYNC', ep, step, fixed=(step, clean, cfg), queued_step=step)
        self.worker.submit(sync)
        if not self._block(sync, limit):
            self.worker.cancel(sync, 'TIMEOUT')
            return None, {**detail, 'reason': 'COMPUTE_TIMEOUT'}
        if sync.status != 'OK':
            return None, {**detail, 'reason': 'COMPUTE_FAILED'}
        return sync, {**detail, 'sync': True}

    def _dawn(self, ep, step, clean, cfg, observation, configuration, started):
        bank = self.rt.read_bank(observation, self.guard)
        act_timeout = self.rt.read_act_timeout(configuration, self.guard)
        if step == 0:
            job, detail = None, {}
            try:
                result = self._opening(ep, clean, cfg)
            except Exception as error:  # the contract's explicit empty-program opening
                self._record_error(0, error, 'OPENING')
                result = {'source': 'INERT', 'program': list(INERT_PROGRAM), 'refusal': 'OPENING_ERROR',
                          'compute_error': _error_text(error)}
        else:
            try:
                job, detail = self._dawn_program(ep, step, clean, cfg, bank, act_timeout, started)
            except Exception as error:  # a dawn failure keeps the active plan; it is not a step failure
                self._record_error(step, error, 'DAWN')
                job, detail = None, {'reason': 'DAWN_ERROR'}
            if job is None:
                skip = {'step': step, 'bank': bank, **detail}
                skip.setdefault('reason', 'NO_DRAFT')
                ep.skips.append(skip)
                self._call['skip'] = skip
                return 'K', detail
            source = 'SYNC' if job.kind == 'SYNC' else ('DRAFT_LATE' if detail.get('draft') == 'LATE' else 'DRAFT')
            result = {**job.result, 'source': source, 'draft_queued_step': job.queued_step}
        try:
            record = self._publish(ep, step, clean, result)
        except Exception as error:
            self._record_error(step, error, 'PUBLISH')
            if step != 0:
                skip = {'step': step, 'bank': bank, 'reason': 'PUBLICATION_REFUSED', 'draft': result['source']}
                ep.skips.append(skip)
                self._call['skip'] = skip
                return 'K', skip
            record = self._publish(ep, 0, clean, {'source': 'INERT', 'program': list(INERT_PROGRAM),
                                                  'refusal': result.get('refusal'), 'publish_error': _error_text(error)})
        if job is not None:
            job.published_step = step
        self._call['pub'] = record
        return 'P', detail

    def _publish(self, ep, step, clean, result):
        proposal = ep.controller.propose(clean, result['program'])
        ep.controller = ep.controller.publish(clean, proposal)
        program = result['program']
        record = {'step': step, 'source': result['source'], 'program': program,
            'program_sha256': result.get('program_sha256') or self.adapter.digest(program),
            'commands': len(program), 'factors': result.get('factors'), 'joint_logp': result.get('joint_logp'),
            'value': result.get('value'), 'input_sha256': result.get('input_sha256'),
            'as_of_step': result.get('as_of_step'), 'observation_step': result.get('observation_step'),
            'draft_queued_step': result.get('draft_queued_step'), 'compute_seconds': result.get('seconds'),
            'encode_seconds': result.get('encode_seconds'), 'sample_seconds': result.get('sample_seconds'),
            'entity_tokens': result.get('entity_tokens'), 'cache_index': result.get('cache_index'),
            'cache_seed': result.get('cache_seed'), 'refusal': result.get('refusal'),
            'record_sha256': result.get('record_sha256'), 'rng_before': result.get('rng_before'),
            'rng_after': result.get('rng_after'), 'proposal_sha256': proposal['content_sha256'],
            'refused': [[r['command_index'], str(r['reason'])[:80]] for r in proposal['rejected']]}
        for key in ('compute_status', 'compute_error', 'publish_error'):
            if key in result:
                record[key] = result[key]
        ep.publications.append(record)
        if step == 0:
            ep.opening = {k: record[k] for k in ('source', 'cache_index', 'refusal', 'input_sha256', 'program_sha256')}
        return record

    def _schedule_draft(self, ep, step):
        nxt = next((s for s in ep.steps if s > step), None)
        if nxt is None or nxt in ep.drafts or self.worker.state == 'FAILED':
            return None
        if nxt - step > 24 - self.v2['draft_hour']:
            return None
        job = _Job('DRAFT', ep, nxt, queued_step=step)
        ep.drafts[nxt] = job
        self.worker.start()
        self.worker.submit(job)
        self._call['draft'] = nxt
        return nxt

    # ----- failure policy ----------------------------------------------------------------
    def _fallback(self, observation, configuration):
        """Never latch: advance this episode's clock, repair the executor, replay the active plan."""
        ep = self.ep
        step = observation.get('step') if isinstance(observation, dict) else None
        if ep is None or (step == 0 and not self._call.get('started')):
            return copy.deepcopy(self.rt.PASS), 'PASS_NO_EPISODE'
        if type(step) is not int:
            step = ep.last_step + 1
        self._call.setdefault('step', step)
        ep.last_step = step
        pending = ep.controller.executor.pending
        if pending and pending.get('step') == step:
            return copy.deepcopy(pending['action']), 'ISSUED_ACTION'
        try:
            clean = self.adapter.actor_view(observation, ep.seat)
            cfg = self.adapter.public_configuration(configuration)
        except Exception:
            return copy.deepcopy(self.rt.PASS), 'PASS_UNREADABLE_OBSERVATION'
        try:
            self._observe(ep, clean, step)
            if ep.controller.committed_step is None:
                return copy.deepcopy(self.rt.PASS), 'PASS_NO_PLAN'
            return ep.controller.act(clean, cfg), 'LAST_PUBLISHED_PLAN'
        except Exception as second:
            self._record_error(step, second, 'EXECUTOR')
            return copy.deepcopy(self.rt.PASS), 'PASS_EXECUTOR_ERROR'

    def __call__(self, observation, configuration=None):
        started = self.clock()
        try:
            self._budget = int((configuration or {}).get('maxLogLength', self.rt.DEFAULT_LOG_BYTES_PER_CALL))
        except Exception:
            self._budget = self.rt.DEFAULT_LOG_BYTES_PER_CALL
        self._call = {}
        error = mode = None
        kind, detail = 'T', {}
        try:
            action, kind, detail = self._act(observation, configuration, started)
        except Exception as exc:
            error, kind = exc, 'X'
            try:
                action, mode = self._fallback(observation, configuration)
            except Exception:
                action, mode = copy.deepcopy(self.rt.PASS), 'PASS_FALLBACK_FAILED'
        waited = 0.0
        try:
            self.emulate(self.clock() - started - self._call.get('blocked', 0.0))
            self.worker.start()  # the model load starts at the first call, after its own work
            remaining = self.v2['per_call_budget_seconds'] - (self.clock() - started)
            if remaining > 0 and self.worker.busy():
                began = self.clock()
                self.worker.wait_idle(remaining)
                waited = self.clock() - began
        except Exception:
            pass
        seconds = self.clock() - started
        try:
            self._account(observation, kind, detail, seconds, error, mode, waited)
        except Exception:
            pass
        return action

    # ----- accounting --------------------------------------------------------------------
    def _flush_worker_events(self, step):
        events = self.worker.events
        while events and self._budget > WORKER_LOG_RESERVE_BYTES:
            try:
                event = events.popleft()
            except IndexError:
                break
            self._write({'e': 'W', 's': step, **event})

    def _account(self, observation, kind, detail, seconds, error, mode, waited):
        ep = self.ep
        step = self._call.get('step')
        if step is None and isinstance(observation, dict):
            step = observation.get('step')
        bank = self.rt.read_bank(observation, self.guard) if isinstance(observation, dict) else None
        self._flush_worker_events(step)
        line = {'e': kind, 's': step, 'ms': round(seconds*1000, 1), 'b': None if bank is None else round(bank, 3)}
        if waited:
            line['w'] = round(waited*1000, 1)
        if self._call.get('blocked'):
            line['bw'] = round(self._call['blocked']*1000, 1)
        optional = ()
        if ep is not None:
            ep.calls.append([step, kind, seconds])
        pub, skip = self._call.get('pub'), self._call.get('skip')
        if pub is not None:
            line.update(src=pub['source'], psha=pub['program_sha256'], cmd=pub['commands'], fac=pub['factors'],
                lp=None if pub['joint_logp'] is None else round(pub['joint_logp'], 4),
                v=None if pub['value'] is None else round(pub['value'], 4), rf=pub['refused'], prog=pub['program'])
            if pub['cache_index'] is not None:
                line['ci'] = pub['cache_index']
            if pub['refusal'] is not None:
                line['cr'] = pub['refusal']
            if pub['compute_seconds'] is not None:
                line['csec'] = round(pub['compute_seconds'], 3)
            if pub['source'].startswith('DRAFT'):
                line.update(dq=pub['draft_queued_step'], dstep=pub['observation_step'])
            if kind == 'X':
                line['pub'] = 1
            optional = ('prog', 'rf')
        if skip is not None:
            line['why'] = skip['reason']
            line['dst'] = skip.get('draft')
            if 'predicted_charge' in skip:
                line.update(pc=round(skip['predicted_charge'], 3), basis=round(skip['basis_seconds'], 3),
                            rs=skip['reserve'])
            if kind == 'X':
                line['skip'] = 1
        if self._call.get('draft') is not None:
            line['ds'] = self._call['draft']
        repair = self._call.get('repair')
        if repair is not None:
            line['rp'] = repair['dropped_pending_step']
        if kind == 'X':
            record = self._record_error(step, error, 'ACT')
            record['mode'] = mode
            line.update(err=record['type'], fb=mode)
            if 'message' in record:
                line['msg'] = record['message'][:200]
        self._write(line, optional=optional)
        if ep is not None and step == self.rt.LAST_STEP and not ep.ended:
            ep.ended = True
            times = [c[2] for c in ep.calls]
            sources = collections.Counter(p['source'] for p in ep.publications)
            self._write({'e': 'Z', 's': step, 'ep': ep.index, 'calls': len(ep.calls), 'pubs': len(ep.publications),
                'src': dict(sources), 'skips': len(ep.skips), 'errs': len(ep.errors), 'repairs': len(ep.repairs),
                'drafts': sum(j.status == 'OK' for j in ep.drafts.values()),
                'tot_s': round(sum(times), 3), 'max_ms': round(max(times)*1000, 1),
                'max_ms_after_first': round(max(times[1:], default=0.0)*1000, 1),
                'model_state': self.worker.state, 'load_s': self.worker.load.get('seconds'),
                'fast': (self.worker.load.get('fast_encode') or {}).get('installed'),
                'log_failures': ep.log_failures, 'rss_mb': self.rt.rss_mb()})

    def summary(self):
        ep = self.ep
        load = copy.deepcopy(self.worker.load)
        hardware = {**self.hardware, 'threads': (load.get('hardware') or {}).get('threads', self.hardware['threads_planned'])}
        return {'runtime_profile': 'v2', 'episode': None if ep is None else ep.index,
            'seat': None if ep is None else ep.seat, 'sampler_seed': None if ep is None else ep.seed,
            'sampler_seed_source': 'OS_ENTROPY', 'cold_seconds': self.cold_seconds, 'runtime': self.runtime_info,
            'hardware': hardware, 'profile': self.profile, 'guard': self.guard, 'runtime_v2': self.v2,
            'fixture': self.settings['fixture'], 'emulated_slowdown': self.slowdown, 'model_load': load,
            'model_state': self.worker.state, 'opening': None if ep is None else ep.opening,
            'publications': [] if ep is None else ep.publications, 'skips': [] if ep is None else ep.skips,
            'drafts': [] if ep is None else [ep.drafts[d].record() for d in sorted(ep.drafts)],
            'repairs': [] if ep is None else ep.repairs,
            'errors': (self.orphan_errors + ([] if ep is None else ep.errors)),
            'calls': [] if ep is None else ep.calls, 'executor_failed': False,
            'log_failures': None if ep is None else ep.log_failures}


def main_source_v2(settings_sha256):
    """Archive-root main.py for v2: lazy light AgentV2 at the first call; PASS forever if construction fails."""
    return ('''"""farmNEAT JOINT EntityGraph actor, runtime profile v2 (background load, cached opening, pre-dawn drafts)."""
import os as _os
import sys as _sys
import time as _time
from pathlib import Path as _Path
_os.environ['CUDA_VISIBLE_DEVICES'] = ''
_root = _Path('/kaggle_simulations/agent')
if not (_root / 'DEPLOYMENT.json').is_file():
    _root = _Path(__file__).resolve().parent if '__file__' in globals() else _Path.cwd()
if str(_root / 'src') not in _sys.path:
    _sys.path.insert(0, str(_root / 'src'))
''' + f'_SETTINGS_SHA256 = {settings_sha256!r}\n' + '''_resident = None
_load_error = None


def _report(error):
    try:
        import json as _json
        import traceback as _traceback
        tail = ''.join(_traceback.format_exception(type(error), error, error.__traceback__)[-3:])[-1200:]
        _sys.stderr.write(_json.dumps({'fj': 1, 'e': 'F', 'err': type(error).__name__, 'msg': str(error)[:300],
            'tb': tail}, ensure_ascii=True, separators=(',', ':'), sort_keys=True) + '\\n')
        _sys.stderr.flush()
    except Exception:
        pass


def agent(observation, configuration=None):
    global _resident, _load_error
    if _resident is None and _load_error is None:
        _started = _time.perf_counter()
        try:
            from farm_neat.evaluation.joint_deployment import AgentV2 as _Agent
            _resident = _Agent(_root, expected_settings_sha256=_SETTINGS_SHA256, started=_started)
        except Exception as _error:
            _load_error = _error
            _report(_error)
    if _resident is None:
        return {'farmer': ['PASS'], 'hands': [], 'market': []}
    return _resident(observation, configuration)
''').encode()


# ----- cold preflight -------------------------------------------------------------------

def worker_environment(extracted):
    """Extracted archive src plus the torch/numpy site only; no repository source."""
    sites = []
    for name in ('numpy', 'torch'):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError('installed ' + name + ' required')
        site = str(Path(spec.origin).resolve().parent.parent)
        if site not in sites:
            sites.append(site)
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
            'PYTHONPATH': os.pathsep.join((str(Path(extracted)/'src'), *sites))}


def _module_origins(root, allowed_sites):
    root = Path(root).resolve()
    stdlib = {Path(p).resolve() for p in (sys.base_prefix, sys.prefix)}
    harness_file = Path(__file__).resolve()
    archive, site, foreign, harness = 0, 0, [], []
    for name, module in list(sys.modules.items()):
        origin = getattr(module, '__file__', None)
        if not origin:
            continue
        path = Path(origin).resolve()
        if path == harness_file or name == '_joint_preflight_engine':
            harness.append({'module': name, 'file': str(path)})
        elif path.is_relative_to(root):
            archive += 1
        elif any(path.is_relative_to(s) for s in allowed_sites):
            site += 1
        elif name.startswith('farm_neat') or not any(path.is_relative_to(s) for s in stdlib):
            foreign.append({'module': name, 'file': str(path)})
    return {'archive_modules': archive, 'site_modules': site, 'foreign': foreign, 'harness_modules': harness}


def _set_archived_slowdown(root, slowdown):
    """Harness-only: set EMULATED_SLOWDOWN on the ARCHIVED v2 runtime module before main.py loads it."""
    if type(slowdown) not in (int, float) or not math.isfinite(slowdown) or slowdown < 1:
        raise ValueError('slowdown must be a finite number >= 1')
    try:
        import farm_neat.evaluation.joint_deployment as archived
    except ImportError as error:
        raise ValueError('slowdown emulation requires a v2 archive (its runtime module)') from error
    if not Path(archived.__file__).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('slowdown hook must target the archived v2 runtime module')
    archived.EMULATED_SLOWDOWN = float(slowdown)
    return str(Path(archived.__file__).resolve())


def worker_joint(root, *, seed, seat, out, inject_bank, emulate_cpu_count=None, slowdown=None):
    """One cold complete game vs PASS: exec archived main.py like the host, time every call.

    emulate_cpu_count (harness only) makes os.cpu_count() return that value before
    main.py runs, so the unchanged runtime rule MIN(OS_CPU_COUNT, 4) picks the
    thread count of a smaller host. slowdown (harness only, v2 archives) makes the
    archived runtime sleep so every measured compute segment lasts R times longer.
    Both are recorded in the receipt.
    """
    import resource
    if emulate_cpu_count is not None:
        if type(emulate_cpu_count) is not int or emulate_cpu_count < 1:
            raise ValueError('positive integer emulated CPU count required')
        os.cpu_count = lambda: emulate_cpu_count
    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != script_dir]
    root = Path(root).resolve()
    out = Path(out)
    if out.exists():
        raise ValueError('new worker receipt required')
    sites = [Path(p).resolve() for p in os.environ.get('PYTHONPATH', '').split(os.pathsep)[1:] if p]
    namespace = {'__name__': 'main'}
    source = (root/'main.py').read_bytes()
    started = time.perf_counter()
    exec(compile(source, 'main.py', 'exec'), namespace)
    import_seconds = time.perf_counter() - started
    slowdown_module = None if slowdown is None else _set_archived_slowdown(root, slowdown)
    agent = namespace['agent']
    # The evaluator engine is loaded by path under a private name, never offered to the actor.
    spec = importlib.util.spec_from_file_location('_joint_preflight_engine', ROOT/'src/farm_neat/game/exact_engine.py')
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    env = engine.ExactKaggricultureEnvironment(seed=seed)
    bank = None if inject_bank is None else float(inject_bank)
    durations, banks, active, timeout_step = [], [], 0, None
    for step in range(GAME_STEPS):
        observation = env.agent_observation(seat)
        if bank is not None:
            observation['remainingOverageTime'] = bank
        banks.append(bank)
        began = time.perf_counter()
        action = agent(observation, env.configuration)
        elapsed = time.perf_counter() - began
        durations.append(elapsed)
        if bank is not None:
            if elapsed > ACT_TIMEOUT + bank and timeout_step is None:
                timeout_step = step
            bank = max(0.0, bank - max(0.0, elapsed - ACT_TIMEOUT))
        active += action != engine.PASS_ACTION
        joint = [engine.PASS_ACTION, engine.PASS_ACTION]
        joint[seat] = action
        env.step(joint)
    resident = namespace.get('_resident')
    farms = env.agent_observation(seat)['farms']
    summary = resident.summary() if resident is not None else {'load_error': repr(namespace.get('_load_error'))}
    origins = _module_origins(root, sites)
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    result = {'schema': WORKER_SCHEMA, 'seed': seed, 'seat': seat, 'opponent': 'PASS', 'physical_games': 1,
        'physical_steps': env.step_index, 'done': env.done, 'own_cash': farms[seat]['money'],
        'opponent_cash': farms[1-seat]['money'], 'active_turns': active,
        'main_exec_seconds': import_seconds, 'first_call_seconds_including_lazy_load': durations[0],
        'call_seconds': durations, 'bank_before_call': banks, 'injected_bank': inject_bank,
        'act_timeout': ACT_TIMEOUT, 'injected_bank_timeout_step': timeout_step, 'injected_bank_final': bank,
        'action_total_seconds': sum(durations), 'action_max_seconds': max(durations),
        'action_max_seconds_after_first': max(durations[1:]),
        'overage_at_1s_including_first_call': sum(max(0.0, t-ACT_TIMEOUT) for t in durations),
        'overage_at_1s_excluding_first_call': sum(max(0.0, t-ACT_TIMEOUT) for t in durations[1:]),
        'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'agent': summary, 'module_origins': origins, 'engine': engine.assert_bundled_engine_identity(),
        'emulated_cpu_count': emulate_cpu_count, 'emulated_slowdown': slowdown,
        'slowdown_module': slowdown_module, 'affinity': affinity, 'gameplay_strength_claim': False}
    out.write_bytes(encoded(result))
    return result


def check_log(stderr, game):
    """Every call has exactly one line; P/K lines (or X lines marked pub/skip) match recorded publications/skips."""
    events, noise = [], 0
    for line in stderr.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            noise += int(bool(line.strip()))
            continue
        if isinstance(value, dict) and value.get('fj') == 1:
            events.append(value)
        else:
            noise += 1
    calls = [e for e in events if e.get('e') in ('P', 'K', 'T', 'X')]
    agent = game.get('agent', {})
    publications = [p['step'] for p in agent.get('publications', [])]
    skips = [s['step'] for s in agent.get('skips', [])]
    starts = [e for e in events if e.get('e') == 'S']
    ends = [e for e in events if e.get('e') == 'Z']
    complete = (bool(events) and len(starts) == 1 and len(ends) == 1 and events[0].get('e') == 'S'
        and events[-1].get('e') == 'Z'
        and [e.get('s') for e in calls] == list(range(GAME_STEPS))
        and [e['s'] for e in calls if e['e'] == 'P' or (e['e'] == 'X' and e.get('pub'))] == publications
        and [e['s'] for e in calls if e['e'] == 'K' or (e['e'] == 'X' and e.get('skip'))] == skips
        and not any(e.get('e') == 'F' for e in events)
        and starts[0].get('seed') == agent.get('sampler_seed'))
    return {'complete': complete, 'events': len(events), 'call_lines': len(calls), 'non_log_lines': noise,
            'publication_lines': sum(e['e'] == 'P' for e in calls), 'skip_lines': sum(e['e'] == 'K' for e in calls),
            'fallback_lines': sum(e['e'] == 'X' for e in calls), 'worker_lines': sum(e.get('e') == 'W' for e in events),
            'stderr_bytes': len(stderr.encode()),
            'max_line_bytes': max((len(line) for line in stderr.splitlines()), default=0),
            'start': starts[0] if starts else None, 'end': ends[0] if ends else None}


def _pinning(emulate_cpu_count, pin_cpus):
    """CPUs to pin a worker to, and how (taskset if installed, else sched_setaffinity)."""
    if pin_cpus is None:
        if emulate_cpu_count is None:
            return None
        available = sorted(os.sched_getaffinity(0))
        if len(available) < emulate_cpu_count:
            raise ValueError('fewer available CPUs than the emulated count')
        pin_cpus = available[-emulate_cpu_count:]
    pin_cpus = [int(c) for c in pin_cpus]
    if not pin_cpus:
        return None
    if emulate_cpu_count is not None and len(pin_cpus) != emulate_cpu_count:
        raise ValueError('pinned CPU count must equal the emulated CPU count')
    return {'cpus': pin_cpus, 'method': 'taskset' if shutil.which('taskset') else 'sched_setaffinity'}


def preflight_joint(archive, *, archive_sha256, out, seed_base=9160000, timeout=1200, inject_bank=60.0,
                    seats=(0, 1), scratch=None, emulate_cpu_count=None, slowdown=None, pin_cpus=None):
    """Cold isolated games from the archive. With emulate_cpu_count, the worker is also pinned
    to that many CPUs (pin_cpus, default the last N of this process's affinity; [] disables)."""
    from farm_neat.evaluation import deployment as legacy, joint_deployment_runtime as runtime
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError('new preflight receipt required')
    if slowdown is not None and (type(slowdown) not in (int, float) or not math.isfinite(slowdown) or slowdown < 1):
        raise ValueError('slowdown must be a finite number >= 1')
    pinning = _pinning(emulate_cpu_count, pin_cpus)
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(scratch) if scratch is not None else ROOT/'work/scratch'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='joint-deployment-', dir=scratch) as tmp:
        extracted = Path(tmp)/'agent'
        extracted.mkdir()
        manifest = legacy.extract(archive, archive_sha256, extracted)
        if manifest.get('actor_kind') != runtime.ACTOR_KIND:
            raise ValueError('joint deployment archive required')
        settings = json.loads((extracted/'DEPLOYMENT.json').read_bytes())
        profile = settings.get('runtime_profile', 'v1')
        if slowdown is not None and profile != 'v2':
            raise ValueError('slowdown emulation requires a v2 archive')
        result = {'schema': PREFLIGHT_SCHEMA, 'archive': str(Path(archive).absolute()), 'archive_sha256': archive_sha256,
            'archive_bytes': Path(archive).stat().st_size, 'members': len(manifest['members']) + 1,
            'settings_sha256': manifest['settings_sha256'], 'deployment_sha256': manifest['deployment_sha256'],
            'original_model_sha256': manifest['original_model_sha256'], 'fixture': manifest['fixture'],
            'runtime_profile': profile, 'runtime_v2': settings.get('runtime_v2'),
            'effective_profile': settings['effective_profile'], 'overage_guard': settings['overage_guard'],
            'injected_bank': inject_bank, 'act_timeout': ACT_TIMEOUT, 'seed_base': seed_base,
            'host': {'cpu_count': os.cpu_count(), 'emulated_cpu_count': emulate_cpu_count,
                     'pinning': pinning, 'emulated_slowdown': slowdown, 'cpu_max': cgroup_cpu()[0],
                     'loadavg_start': os.getloadavg()},
            'complete': False, 'games': [],
            'scope': 'COLD_ISOLATED_CPU_ARCHIVE_COMPLETE_GAMES_VS_PASS; NOT_HOSTED_OR_STRENGTH_CERTIFICATION'}
        out.write_bytes(encoded(result))
        for seat in seats:
            receipt = Path(tmp)/f'seat-{seat}.json'
            args = [sys.executable, '-B', '-S', str(Path(__file__).resolve()), 'worker-joint', '--root', str(extracted),
                    '--seed', str(seed_base + seat), '--seat', str(seat), '--out', str(receipt)]
            if inject_bank is not None:
                args += ['--inject-bank', repr(float(inject_bank))]
            if emulate_cpu_count is not None:
                args += ['--emulate-cpu-count', str(int(emulate_cpu_count))]
            if slowdown is not None:
                args += ['--slowdown', repr(float(slowdown))]
            preexec = None
            if pinning is not None and pinning['method'] == 'taskset':
                args = ['taskset', '-c', ','.join(map(str, pinning['cpus']))] + args
            elif pinning is not None:
                cpus = set(pinning['cpus'])
                preexec = lambda: os.sched_setaffinity(0, cpus)  # noqa: E731
            started = time.perf_counter()
            try:
                completed = subprocess.run(args, cwd=extracted, env=worker_environment(extracted),
                                           capture_output=True, text=True, timeout=timeout, check=False,
                                           preexec_fn=preexec)
            except subprocess.TimeoutExpired as error:
                result['error'] = f'worker seat {seat} timeout after {timeout}s'
                result['games'].append({'seat': seat, 'timeout': True,
                    'stderr_tail': (error.stderr or b'')[-3000:].decode(errors='replace')
                    if isinstance(error.stderr, bytes) else (error.stderr or '')[-3000:]})
                out.write_bytes(encoded(result))
                raise ValueError(result['error']) from error
            game = json.loads(receipt.read_bytes()) if receipt.is_file() else {'seat': seat, 'receipt': 'MISSING'}
            game['process'] = {'returncode': completed.returncode, 'wall_seconds': time.perf_counter() - started,
                               'stdout_tail': completed.stdout[-2000:], 'stderr_tail': completed.stderr[-3000:]}
            game['log_verification'] = check_log(completed.stderr, game)
            game['stderr'] = completed.stderr
            result['games'].append(game)
            out.write_bytes(encoded(result))
            if completed.returncode or not receipt.is_file():
                result['error'] = f'worker seat {seat} exit {completed.returncode}'
                out.write_bytes(encoded(result))
                raise ValueError(result['error'] + ': ' + completed.stderr[-2000:])
        profile_steps = settings['effective_profile']['publication_steps']
        for game in result['games']:
            agent = game['agent']
            pubs = [p['step'] for p in agent['publications']]
            game['checks'] = {'complete_game': game['done'] and game['physical_steps'] == GAME_STEPS,
                'opening_published': pubs[:1] == [0],
                'dawns_accounted': sorted(pubs + [s['step'] for s in agent['skips']]) == profile_steps,
                'no_exceptions': not agent['errors'], 'log_complete': game['log_verification']['complete'],
                'archive_only_imports': not game['module_origins']['foreign']}
        result['games_complete'] = all(g['checks']['complete_game'] for g in result['games'])
        result['complete'] = all(all(g['checks'].values()) for g in result['games'])
        result['host']['loadavg_end'] = os.getloadavg()
        result['summary'] = [_game_summary(g) for g in result['games']]
        if not result['complete']:
            result['error'] = 'preflight checks failed'
        out.write_bytes(encoded(result))
        return result


def _game_summary(g):
    agent = g['agent']
    summary = {'seat': g['seat'], 'publications': len(agent['publications']),
        'skips': len(agent['skips']), 'errors': len(agent['errors']), 'own_cash': g['own_cash'],
        'opponent_cash': g['opponent_cash'], 'first_call_seconds': g['first_call_seconds_including_lazy_load'],
        'agent_cold_seconds': agent['cold_seconds'],
        'publication_seconds_mean': statistics.fmean(p['seconds'] for p in agent['publications'])
            if agent['publications'] and 'seconds' in agent['publications'][0] else None,
        'publication_seconds_max': max((p['seconds'] for p in agent['publications'] if 'seconds' in p), default=None),
        'overage_at_1s_including_first_call': g['overage_at_1s_including_first_call'],
        'rss_peak_bytes': g['rss_peak_bytes'], 'threads': agent['hardware']['threads'],
        'checks': g['checks']}
    if agent.get('runtime_profile') == 'v2':
        call_seconds = g['call_seconds']
        summary.update(runtime_profile='v2',
            publication_sources=dict(collections.Counter(p['source'] for p in agent['publications'])),
            skip_reasons=dict(collections.Counter(s['reason'] for s in agent['skips'])),
            drafts_ok=sum(d['status'] == 'OK' for d in agent['drafts']),
            draft_compute_seconds_max=max((d['seconds'] for d in agent['drafts'] if d['seconds'] is not None),
                                          default=None),
            repairs=len(agent['repairs']), model_load_seconds=agent['model_load'].get('seconds'),
            fast_encode=agent['model_load'].get('fast_encode'), opening=agent['opening'],
            calls_over_act_timeout=sum(t > ACT_TIMEOUT for t in call_seconds),
            action_max_seconds_after_first=g['action_max_seconds_after_first'],
            injected_bank_final=g['injected_bank_final'], emulated_slowdown=g.get('emulated_slowdown'),
            affinity=g.get('affinity'), cpu_max=agent['hardware'].get('cpu_max'))
    return summary


# ----- offline overage simulation -------------------------------------------------------

def simulate_game(call_seconds, kinds, dawns, *, slowdown, bank, guard, guarded, act_timeout=ACT_TIMEOUT,
                  charge_first_call=True, inner_seconds=None):
    """Apply the Kaggle charge rule to measured calls, optionally replaying the guard.

    call_seconds are host-side call times (the first includes the lazy Agent load)
    and are what the bank is charged. inner_seconds are the agent's own call times,
    which is what the runtime guard predicts from. Skipped dawns cost the median
    measured ordinary call; a dawn published in the simulation but not in the
    measurement costs the slowest measured publication. A call longer than
    actTimeout + bank is a host TIMEOUT; the simulation stops there.
    (v1 model: publication cost at the dawn. v2 moves compute off the dawn; use
    preflight-joint --slowdown for v2 instead.)
    """
    from farm_neat.evaluation import joint_deployment_runtime as runtime
    inner_seconds = call_seconds if inner_seconds is None else inner_seconds
    measured_pub = [t for t, k in zip(call_seconds, kinds) if k == 'P']
    ordinary = [t for i, (t, k) in enumerate(zip(call_seconds, kinds)) if k == 'T' and i]
    ordinary_cost = statistics.median(ordinary) if ordinary else 0.0
    slowest = max(measured_pub) if measured_pub else guard['prior_publication_seconds']
    remaining, seen, published, skipped, charged = float(bank), [], [], [], 0.0
    timeout_step = None
    for step, (seconds, kind) in enumerate(zip(call_seconds, kinds)):
        if step in dawns:
            publish = True
            if guarded:
                publish, _ = runtime.guard_decision(step=step, bank=remaining, publication_seconds=seen,
                                                    act_timeout=act_timeout, guard=guard)
            if publish:
                measured = kind in ('P', 'X')
                cost = (seconds if measured else slowest) * slowdown
                seen.append((inner_seconds[step] if measured else slowest) * slowdown)
                published.append(step)
            else:
                cost = ordinary_cost * slowdown
                skipped.append(step)
        else:
            cost = seconds * slowdown
        if step == 0 and not charge_first_call:
            continue
        if cost > act_timeout + remaining:
            timeout_step = step
            break
        charge = max(0.0, cost - act_timeout)
        remaining -= charge
        charged += charge
    return {'slowdown': slowdown, 'guarded': guarded, 'charge_first_call': charge_first_call,
            'timeout_step': timeout_step, 'survived': timeout_step is None, 'final_bank': remaining,
            'charged_seconds': charged, 'published_dawns': published, 'skipped_dawns': skipped,
            'publications': len(published), 'skips': len(skipped),
            'first_call_cost_seconds': call_seconds[0]*slowdown}


def simulate_overage(preflight, *, slowdowns, bank, out=None, guard_reserve=None):
    from farm_neat.evaluation import joint_deployment_runtime as runtime
    data = json.loads(Path(preflight).read_bytes())
    if data.get('schema') != PREFLIGHT_SCHEMA:
        raise ValueError('joint preflight receipt required')
    guard = dict(data['overage_guard'])
    if guard_reserve is not None:
        guard['reserve_seconds'] = float(guard_reserve)
    dawns = set(data['effective_profile']['publication_steps'])
    rows = []
    for game in data['games']:
        seconds = game['call_seconds']
        kinds, inner = [None]*len(seconds), list(seconds)
        for step, kind, own in game['agent']['calls']:
            kinds[step], inner[step] = kind, own
        for slowdown in slowdowns:
            for guarded in (False, True):
                for charge_first in (True, False):
                    row = simulate_game(seconds, kinds, dawns, slowdown=slowdown, bank=bank, guard=guard,
                                        guarded=guarded, act_timeout=data['act_timeout'],
                                        charge_first_call=charge_first, inner_seconds=inner)
                    rows.append({'seat': game['seat'], **row})
    result = {'schema': SIMULATION_SCHEMA, 'preflight': str(Path(preflight).absolute()),
        'preflight_sha256': sha(preflight), 'archive_sha256': data['archive_sha256'], 'bank': bank,
        'act_timeout': data['act_timeout'], 'guard': guard, 'slowdowns': list(slowdowns), 'rows': rows,
        'rule': 'EACH CALL CHARGES MAX(0, SECONDS*SLOWDOWN - ACT_TIMEOUT); SECONDS > ACT_TIMEOUT + BANK IS TIMEOUT',
        'scope': 'MEASURED LOCAL CALL TIMES SCALED UNIFORMLY; HOSTED SLOWDOWN AND FIRST-CALL CHARGING UNVERIFIED'}
    if out is not None:
        out = Path(out).absolute()
        if out.exists():
            raise ValueError('new simulation output required')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(encoded(result))
    return result


def _steps(text):
    return None if text is None else [int(v) for v in text.split(',') if v.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)
    b = sub.add_parser('build-joint', help='encode a joint checkpoint into a hash-pinned Kaggle archive')
    b.add_argument('--model', required=True)
    b.add_argument('--model-sha256', required=True)
    b.add_argument('--output', required=True)
    b.add_argument('--command-budget', type=int, help='option (default OFF): command budget instead of 8')
    b.add_argument('--publication-steps', help='option (default OFF): comma subset of the 30 dawns, starting 0')
    b.add_argument('--reserve-seconds', type=float, default=20.0, help='overage guard reserve (default 20)')
    b.add_argument('--profile', choices=PROFILES, default='v1',
                   help='runtime profile (default v1: the unchanged pre-v2 archive)')
    b.add_argument('--opening-cache-k', type=int, default=DEFAULT_CACHE_K,
                   help=f'v2: cached step-0 programs per seat (default {DEFAULT_CACHE_K}; 0 disables)')
    b.add_argument('--opening-seed-base', type=int, default=DEFAULT_CACHE_SEED_BASE,
                   help='v2: generator seed of cached program 0 (program i uses base+i)')
    b.add_argument('--draft-hour', type=int, default=DEFAULT_DRAFT_HOUR,
                   help=f'v2: hour from which the next dawn is drafted (default {DEFAULT_DRAFT_HOUR})')
    b.add_argument('--per-call-budget', type=float, default=DEFAULT_CALL_BUDGET,
                   help=f'v2: max seconds a call waits for background work (default {DEFAULT_CALL_BUDGET})')
    b.add_argument('--no-fast-encode', action='store_true', help='v2: never install fast_gdn/fast_moe')
    b.add_argument('--slot95-decoder', action='store_true',
                   help='v2: slot95 frozen-entity-KV decoder (declared non-equivalent; default OFF)')
    f = sub.add_parser('preflight-joint', help='cold isolated complete games vs PASS from the archive')
    f.add_argument('--archive', required=True)
    f.add_argument('--archive-sha256', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--seed-base', type=int, default=9160000)
    f.add_argument('--timeout', type=int, default=1200)
    f.add_argument('--inject-bank', type=float, default=60.0,
                   help='remainingOverageTime injected and charged at measured speed (default 60)')
    f.add_argument('--no-bank', action='store_true', help='do not inject remainingOverageTime')
    f.add_argument('--seats', default='0,1')
    f.add_argument('--emulate-cpu-count', type=int,
                   help='harness only: os.cpu_count() seen by the archived runtime (thread rule MIN(n, 4)); '
                        'the worker is also pinned to n CPUs')
    f.add_argument('--pin-cpus', help='CPUs for the pinned worker (default: last n of the current affinity)')
    f.add_argument('--no-pin', action='store_true', help='emulate the CPU count without pinning')
    f.add_argument('--slowdown', type=float,
                   help='harness only, v2 archives: stretch every measured compute segment to R x its duration')
    s = sub.add_parser('simulate-overage', help='Kaggle charge rule over measured preflight call times')
    s.add_argument('--preflight', required=True)
    s.add_argument('--slowdown', default='1,2,3')
    s.add_argument('--bank', type=float, default=60.0)
    s.add_argument('--reserve-seconds', type=float, help='replay the guard with another reserve')
    s.add_argument('--out')
    w = sub.add_parser('worker-joint', help=argparse.SUPPRESS)
    w.add_argument('--root', required=True)
    w.add_argument('--seed', type=int, required=True)
    w.add_argument('--seat', type=int, choices=(0, 1), required=True)
    w.add_argument('--out', required=True)
    w.add_argument('--inject-bank', type=float)
    w.add_argument('--emulate-cpu-count', type=int)
    w.add_argument('--slowdown', type=float)
    a = p.parse_args(argv)
    if a.mode == 'worker-joint':
        worker_joint(a.root, seed=a.seed, seat=a.seat, out=a.out, inject_bank=a.inject_bank,
                     emulate_cpu_count=a.emulate_cpu_count, slowdown=a.slowdown)
        return
    if str(ROOT/'src') not in sys.path:
        sys.path.insert(0, str(ROOT/'src'))
    if a.mode == 'build-joint':
        v2_options = {}
        if a.profile == 'v2':
            v2_options = dict(opening_cache_k=a.opening_cache_k, opening_seed_base=a.opening_seed_base,
                              draft_hour=a.draft_hour, per_call_budget=a.per_call_budget,
                              fast_encode=not a.no_fast_encode, slot95_decoder=a.slot95_decoder)
        print(json.dumps(build_joint(a.model, model_sha256=a.model_sha256, output=a.output,
            command_budget=a.command_budget, publication_steps=_steps(a.publication_steps),
            reserve_seconds=a.reserve_seconds, profile=a.profile, **v2_options), indent=2))
    elif a.mode == 'preflight-joint':
        pin = [] if a.no_pin else _steps(a.pin_cpus)
        r = preflight_joint(a.archive, archive_sha256=a.archive_sha256, out=a.out, seed_base=a.seed_base,
                            timeout=a.timeout, inject_bank=None if a.no_bank else a.inject_bank,
                            seats=tuple(_steps(a.seats)), emulate_cpu_count=a.emulate_cpu_count,
                            slowdown=a.slowdown, pin_cpus=pin)
        print(json.dumps({k: v for k, v in r.items() if k != 'games'}, indent=2))
        if not r['complete']:
            raise SystemExit(2)
    else:
        slowdowns = [float(v) for v in a.slowdown.split(',') if v.strip()]
        if not slowdowns or any(not math.isfinite(v) or v <= 0 for v in slowdowns):
            raise SystemExit('positive finite slowdowns required')
        r = simulate_overage(a.preflight, slowdowns=slowdowns, bank=a.bank, out=a.out, guard_reserve=a.reserve_seconds)
        keys = ('seat', 'slowdown', 'guarded', 'charge_first_call', 'survived', 'timeout_step', 'final_bank',
                'publications', 'skips', 'first_call_cost_seconds')
        print(json.dumps({**{k: v for k, v in r.items() if k != 'rows'},
                          'rows': [{k: row[k] for k in keys} for row in r['rows']]}, indent=2))


if __name__ == '__main__':
    main()
