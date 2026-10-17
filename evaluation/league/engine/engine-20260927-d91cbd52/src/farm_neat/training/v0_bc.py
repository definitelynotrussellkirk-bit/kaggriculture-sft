"""Native V0 class-path BC, with verified data and immutable continuation.

    PYTHONPATH=src python3 -B -m farm_neat.training.v0_bc model-manifest --output NEW_JSON
    PYTHONPATH=src python3 -B -m farm_neat.training.v0_bc validate --help
    PYTHONPATH=src python3 -B -m farm_neat.training.v0_bc train --help
    PYTHONPATH=src python3 -B -m farm_neat.training.v0_bc weights-transfer --help

Register this module in project/entrypoints.toml during maintainer intake.
Production train requires the pinned interpreter, the default 16-layer V0 model,
explicit dataset/model hashes, a finite GLOBAL update cap and a new output root.
Only an exact committed V0 continuation may resume. A separately permitted
weights-transfer may initialize a NEW lineage from authenticated model tensors only;
it is not resume, source equivalence, historical migration or a strength claim.
Diagnostic models are available to Python CPU unit tests,
not the production CLI. No NLL/gradient/checkpoint result is a strength claim.

The sole objective is trajectory-mean SUM of prefix successor-class NLL, including
sampled STOP. Forced cap STOP is a singleton support, hence probability one. Menus
are rebuilt by the accepted scorer at every distinct canonical prefix at dataset
validation, then used index-for-index by its score_batch/class_nll. The cache key
includes actor facts, canonical Plan/key/count, backend and current decoder source hashes;
it never stores a model-dependent tensor or aliases different prefixes.

A commit publishes COMMIT.json last, with an exclusive atomic hard link after
fsync and a weights_only round-trip. Incomplete directories cannot be parents.
Every update saves optimizer, parameter order, all used RNGs, shuffle order/cursor,
exposure count and global age. On failure the previous commit stays authoritative.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import pickle
import random
import stat
import sys
import tomllib
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import PHI_NAMES, State
from farm_neat.policy import v0_scorer as A
from farm_neat.policy import v0_menu_fast as F
from farm_neat.policy.qwen_hybrid import Config
from farm_neat.training import v0_targets as T

SCHEMA = 'farm-v0-class-path-bc-v1'
COMMIT_SCHEMA = SCHEMA + '-commit-v1'
WEIGHTS_TRANSFER_ORIGIN_SCHEMA = SCHEMA + '-weights-transfer-origin-v1'
WEIGHTS_TRANSFER_PERMIT_SCHEMA = SCHEMA + '-weights-transfer-permit-v1'
WEIGHTS_TRANSFER_INVENTORY_SCHEMA = SCHEMA + '-weights-transfer-tensor-inventory-v1'
WEIGHTS_TRANSFER_INITIALIZATION_SCHEMA = SCHEMA + '-weights-transfer-initialization-v1'
WEIGHTS_TRANSFER_OPERATION = 'MODEL_TENSORS_ONLY_NEW_NATIVE_V0_BC_LINEAGE'
WEIGHTS_TRANSFER_IMPLEMENTATION = 'v1'
WEIGHTS_TRANSFER_SOURCE_PATH = 'src/farm_neat/training/v0_bc.py'
# This admits a 60M-class FP32 scorer plus a full Adam-family native state while
# refusing unexpectedly huge payloads before torch.load can exhaust the process.
MAX_WEIGHTS_TRANSFER_STATE_BYTES = 8_000_000_000
MAX_WEIGHTS_TRANSFER_TENSOR_BYTES = 4_000_000_000
MAX_WEIGHTS_TRANSFER_JSON_BYTES = 2_000_000
OBJECTIVE = 'TRAJECTORY_MEAN_OF_WEIGHTED_SUMMED_PREFIX_CLASS_NLL_INCLUDING_SAMPLED_STOP;NO_AUXILIARIES'
RUNTIME_PATH = 'project/runtime/interpreter.toml'
DATA_FILES = ('prefixes.jsonl', 'trajectories.jsonl', 'rejections.jsonl')


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_loads(text: str) -> Any:
    def reject_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_pairs)


def _json_read(path: Path) -> Any:
    return _json_loads(path.read_text(encoding='utf-8'))


def _pinned_json(path: Path, expected_sha256: str) -> dict:
    _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64,
             'explicit SHA256 required')
    _require(file_sha(path) == expected_sha256, 'file hash mismatch: ' + str(path))
    value = _json_read(path)
    _require(isinstance(value, dict), 'JSON object required: ' + str(path))
    return value


def source_pins() -> dict[str, str]:
    return {**T.source_identity()['sources'], **A.source_pins(),
            'src/farm_neat/training/v0_bc.py': file_sha(__file__),
            RUNTIME_PATH: file_sha(ROOT / RUNTIME_PATH)}


def model_manifest(config: Config | None = None, *, diagnostic: bool = False,
                   readout: str = 'mean') -> dict:
    """Describe a fresh scorer without allocating a production model's tensors.

    Mean retains the historical manifest exactly (no readout key). Nondefault
    modes bind the pinned scorer's complete versioned extra-state identity; this
    is a fresh-lineage selector, never permission to reinterpret old tensors.
    """
    A._readout_mode(readout)
    config = Config() if config is None else config
    _require(isinstance(config, Config) and type(diagnostic) is bool, 'typed model configuration required')
    _require(diagnostic or config == Config(), 'production bootstrap pins the default 16-layer V0 config')
    identity = {'schema': A.SCHEMA, 'config': asdict(config), 'diagnostic': diagnostic,
        'action_schema_sha256': V.ACTION_SCHEMA_SHA256,
        'feature_schema_sha256': A.FEATURE_SCHEMA_SHA256, 'source_pins': A.source_pins(),
        'legacy_checkpoint_compatible': False, 'initialization': 'FRESH_TORCH_NO_WEIGHT_IMPORT',
        'numerics': 'FP32_NO_AUTOCAST_NO_TF32',
        'continuation': 'REENCODE_CANONICAL_PREFIX_NO_RAW_ALIAS_CACHE'}
    if readout != 'mean':
        # Keep this allocation-free declaration equal to V0Scorer.get_extra_state.
        # The focused join tests compare the full identities, not just mode names.
        identity['readout'] = {
            'schema': A.READOUT_SCHEMA, 'version': 'v1', 'mode': readout,
            'method': ('POOLED_GATED_MLP' if readout == 'pooled_control'
                       else 'MASKED_TOKEN_ATTENTION'),
            'context': ('MASKED_MEAN_ONLY' if readout == 'pooled_control'
                        else 'MASKED_ACTOR_TOKENS'),
            'candidate': 'ACTION_FIELDS_PLUS_PHI', 'residual': 'MASKED_MEAN'}
    return identity


def validate_model_manifest(value: dict, *, production: bool = False) -> None:
    try:
        _require(isinstance(value, dict), 'complete native V0 model manifest required')
        # Only absence of the whole key denotes historical mean. A partial/null
        # nondefault identity is not a request to fall back to mean.
        readout = value['readout']['mode'] if 'readout' in value else 'mean'
        expected = model_manifest(Config(**value['config']), diagnostic=value['diagnostic'],
                                  readout=readout)
        _require(value == expected, 'model/schema/source/decoder manifest mismatch')
        _require(not production or not value['diagnostic'], 'diagnostic model refused by production CLI')
    except (KeyError, TypeError) as exc:
        raise ValueError('complete native V0 model manifest required') from exc


def actor_state(public: dict) -> State:
    """Reconstruct ONLY exported actor facts, never a trainer replay observation.

    State's methods operate on these attributes. player is an unused neutral seat;
    the actual seat stays in trainer grouping. Event facts are round-trip checked
    against the native calendar. No fake coordinates or worker identities enter.
    """
    try:
        _require(public['schema'] == T.ACTOR_SCHEMA, 'unknown actor schema')
        _require(type(public['remaining_days']) is int and public['remaining_days'] == V.SEASON_DAYS,
                 'only opening actor rows are admitted by this exporter')
        s = State.__new__(State)
        s.player, s.day, s.hour, s.step = 0, 0, 0, 0
        s.cash = public['cash']
        s.quadrants, s.hands_today = public['owned_quadrants'], public['hands_today']
        # Opening exporter pins the default engine's ordered quadrant unlocks;
        # the admitted count uniquely determines this public ownership prefix.
        s.quadrant_names = tuple(V.QUADRANTS[:s.quadrants])
        s.land_commitment = None  # opening examples precede any publication
        s.hires_today, s.occupied = public['hires_today'], public['occupied_sites']
        s.stock, s.seeds = copy.deepcopy(public['stock']), copy.deepcopy(public['seeds'])
        s.inventory, s.params = copy.deepcopy(public['market_inventory']), copy.deepcopy(public['market_params'])
        s.shops = copy.deepcopy(public['unlocked_shops'])
        _require(T.actor_observation(s) == public, 'actor fact/event round-trip mismatch')
        A.actor_tokens(s, A.Prefix(Plan()), memory={})  # same finite/knownness checks as the actor
        return s
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('invalid actor-only observation') from exc


@lru_cache(maxsize=128)
def _menu(actor_json: str, decoder_source_sha256: str, *, backend: str = 'fast') -> A.Menu:
    # decoder_source_sha256 is intentionally part of the memo key even though
    # build_menu reads code, not that string. Changed sources invalidate the cache.
    actor = json.loads(actor_json)
    prefix = A.Prefix(Plan.from_canonical(actor['prefix_plan']), actor['last_key'], actor['nonstop_edit_count'])
    return F.build_menu(actor_state(actor['public_observation']), prefix, backend=backend,
                        projection_kwargs={**T.SETTINGS, 'hold_releases': {}})


def _class_payload(c: A.SuccessorClass) -> dict:
    p = c.successor
    return {'indices': list(c.indices), 'successor_plan_sha256': p.plan.digest(),
            'continuation': T.continuation_signature(p.plan, p.last_key, p.edit_count, p.terminated)}


@dataclass(frozen=True)
class Example:
    state: State
    menu: A.Menu
    target_indices: tuple[int, ...]


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    examples: tuple[Example, ...]


@dataclass(frozen=True)
class Dataset:
    manifest_sha256: str
    split: str
    trajectories: tuple[Trajectory, ...]
    validation: dict

    @property
    def binding(self) -> dict:
        return {'schema': T.SCHEMA, 'manifest_sha256': self.manifest_sha256, 'split': self.split,
                'trajectory_ids': [t.trajectory_id for t in self.trajectories]}


def load_dataset(directory: str | Path, expected_sha256: str, *, split: str,
                 backend: str = 'fast') -> Dataset:
    """Fail closed before any optimization; validate ALL rows, not just the fit split.

    Acceptance is the existing exporter's reachability/compiler check, not a new
    teacher-strength approval. Rejections stay excluded. Trainer replay observations
    are used only for admission and actor equality; networks receive actor_state.
    """
    backend = A._menu_backend(backend)
    directory = Path(directory)
    try:
        manifest = _pinned_json(directory / 'MANIFEST.json', expected_sha256)
        _require(isinstance(split, str) and bool(split), 'explicit split required')
        _require(manifest['status'] == 'EXPORTED', 'dataset not committed EXPORTED')
        identity = T.source_identity()
        for key, value in identity.items():
            _require(manifest.get(key) == value, 'dataset source/schema mismatch: ' + key)
        _require(set(manifest['files']) == set(DATA_FILES), 'exact V0 dataset files required')
        documents = {}
        for name in DATA_FILES:
            path, pin = directory / name, manifest['files'][name]
            _require(path.stat().st_size == pin['bytes'] and file_sha(path) == pin['sha256'],
                     'dataset member hash mismatch: ' + name)
            # JSONL uses the same duplicate-key/nonfinite rejection as manifests.
            lines = path.read_text(encoding='utf-8').splitlines()
            documents[name] = [_json_loads(line) for line in lines]
            _require(all(isinstance(row, dict) for row in documents[name]), 'JSONL objects required')
            canonical(documents[name])  # reject NaNs/Infs, including unconsumed metadata
        accepted, rejected = documents['trajectories.jsonl'], documents['rejections.jsonl']
        rows = documents['prefixes.jsonl']
        all_records = accepted + rejected
        ids = [r['identity']['trajectory_id'] for r in all_records]
        _require(len(ids) == len(set(ids)), 'duplicate trajectory ID')
        groups = [r['identity'] | {'status': r['status']} for r in sorted(all_records,
                  key=lambda r: r['identity']['trajectory_id'])]
        _require(groups == manifest['groups'] and _digest(groups) == manifest['groups_sha256'],
                 'dataset group identity mismatch')
        for field in ('physical_world_id', 'source_group', 'seed_group'):
            assigned = {}
            for g in groups:
                _require(all(isinstance(g[k], str) and g[k] for k in T.IDENTITY_FIELDS), 'group fields required')
                _require(g[field] not in assigned or assigned[g[field]] == g['split'], 'group split leakage: ' + field)
                assigned[g[field]] = g['split']
        _require(all(r['status'] == 'REJECTED' for r in rejected), 'invalid rejection record')
        by_id = {r['identity']['trajectory_id']: [] for r in accepted}
        for row in rows:
            _require(row['schema'] == T.SCHEMA, 'non-V0 prefix row')
            tid = row['identity']['trajectory_id']
            _require(tid in by_id, 'prefix belongs to an absent or rejected trajectory')
            by_id[tid].append(row)
        decoder_sha = _digest(A.source_pins())
        trajectories = []
        nontrivial = checked = 0
        for record in accepted:
            rid = record['identity']
            tid = rid['trajectory_id']
            _require(record['status'] == 'ADMITTED', 'unadmitted trajectory')
            request = {k: rid[k] for k in T.IDENTITY_FIELDS}
            request.update(teacher_kind=record['trainer_only']['teacher_kind'],
                observation=record['trainer_only']['replay_observation'],
                initial_plan=record['initial_plan'], intended_plan=record['intended_plan'],
                actions=record['actions'], provenance=record['trainer_only']['provenance'])
            verified = T.verify_request(request)
            # Source-request hashes attest the original request (which may contain
            # extra provenance); no reconstructed request is silently substituted.
            for key in ('status', 'reason', 'identity', 'actions', 'prefixes', 'nonstop_edits',
                        'initial_plan', 'intended_plan', 'reached_plan', 'reached_plan_sha256',
                        'residual', 'verification'):
                _require(record[key] == verified[key], 'native trajectory admission mismatch: ' + key)
            _require(verified['status'] == 'ADMITTED', 'native trajectory rejected')
            nontrivial += record['nonstop_edits'] > 0
            sequence = by_id[tid]
            _require(len(sequence) == len(record['prefixes']) > 0, 'missing/extra prefix rows')
            prefix = A.Prefix(Plan())
            examples = []
            for index, (row, native) in enumerate(zip(sequence, record['prefixes'])):
                _require(not prefix.terminated, 'row after STOP')
                expected_id = rid | {'prefix_index': index, 'split_id': manifest['split_id'],
                    **{k: identity[k] for k in ('source_bundle_sha256', 'action_schema_sha256',
                                               'compiler_sha256', 'feature_schema_sha256')}}
                _require(row['identity'] == expected_id, 'prefix identity/cursor mismatch')
                actor = row['actor']
                _require(set(actor) == {'public_observation', 'own_memory', 'prefix_plan', 'last_key', 'nonstop_edit_count'},
                         'actor field allowlist mismatch')
                _require(actor['own_memory'] == {}, 'opening exporter has no continuation memory')
                _require(actor['public_observation'] == T.actor_observation(State(request['observation'])),
                         'actor facts disagree with admitted observation')
                _require(actor['prefix_plan'] == prefix.plan.canonical()
                         and type(actor['last_key']) is int and actor['last_key'] == prefix.last_key
                         and type(actor['nonstop_edit_count']) is int and actor['nonstop_edit_count'] == prefix.edit_count,
                         'not the actual canonical prefix')
                state = actor_state(actor['public_observation'])
                menu = _menu(canonical(actor), decoder_sha, backend=backend)
                cand = row['candidates']
                expected_cand = {'legal_ids': list(menu.legal_ids), 'phi': [list(p) for p in menu.phi],
                    'classes': [_class_payload(c) for c in sorted(menu.classes, key=lambda c: c.indices[0])],
                    'forced_stop': prefix.edit_count == V.MAX_EDITS_PER_EPOCH, 'phi_names': list(PHI_NAMES)}
                _require(cand == expected_cand, 'prefix support/phi/classes mismatch')
                target = row['target']
                indices = tuple(target['class_indices'])
                _require(all(type(i) is int for i in indices), 'integer target class indices required')
                chosen = next((c for c in menu.classes if c.indices == indices), None)
                _require(chosen is not None, 'target is not one complete successor class')
                payload = _class_payload(chosen)
                stop_kind = ('forced_cap' if cand['forced_stop'] else 'sampled') if chosen.successor.terminated else 'not_stop'
                _require(target == {'class_indices': list(indices),
                    'successor_plan_sha256': payload['successor_plan_sha256'],
                    'continuation': payload['continuation'], 'stop_kind': stop_kind,
                    # The exporter unweights forced cap STOP and whole KEEP/STOP control trajectories.
                    'loss_weight': 0 if cand['forced_stop'] or record['nonstop_edits'] == 0 else 1},
                    'target/STOP/loss-weight mismatch')
                _require(payload['continuation'] == native['continuation'], 'target does not continue native trace')
                _require(row['trainer_only'] == {'replay_action': native['action'],
                    'source_request_sha256': record['trainer_only']['source_request_sha256']}, 'row provenance mismatch')
                _require(native['action'] in [menu.legal_ids[i] for i in indices], 'native representative outside target class')
                examples.append(Example(state, menu, indices))
                prefix = chosen.successor
                checked += 1
            _require(prefix.terminated and prefix.plan.canonical() == record['intended_plan'], 'missing STOP/unreached target')
            if rid['split'] == split and record['nonstop_edits'] > 0:
                # Controls stay validated but carry no loss: they would dilute the trajectory mean.
                trajectories.append(Trajectory(tid, tuple(examples)))
        expected_counts = {'requested': len(all_records), 'admitted': len(accepted), 'rejected': len(rejected),
            'nontrivial': nontrivial, 'stop_controls': len(accepted)-nontrivial, 'prefixes': len(rows)}
        _require(manifest['counts'] == expected_counts, 'dataset denominator/count mismatch')
        _require(manifest['expected_nontrivial'] is None or manifest['expected_nontrivial'] == nontrivial,
                 'nontrivial count gate mismatch')
        _require(bool(trajectories), 'selected training split is empty')
        return Dataset(expected_sha256, split, tuple(sorted(trajectories, key=lambda t: t.trajectory_id)),
                       {'prefixes_checked': checked, 'admitted': len(accepted), 'rejected': len(rejected),
                        'selected_trajectories': len(trajectories), 'game_steps': 0, 'games': 0})
    except (KeyError, TypeError, AttributeError, StopIteration, IndexError) as exc:
        raise ValueError('malformed native V0 dataset: ' + str(exc)) from exc


def path_nll(logits: Sequence[torch.Tensor], targets: Sequence[Sequence[int]], *,
             valid: Sequence[torch.Tensor] | None = None,
             weights: Sequence[float] | None = None) -> torch.Tensor:
    """Exact weighted class-PATH NLL; accepted scorer owns the categorical term.

    Weights only change the reduction over already-frozen prefix targets.  A weight
    of one reproduces the original objective exactly; no class, target or support is
    changed.
    """
    _require(len(logits) == len(targets) > 0, 'aligned nonempty prefix path required')
    masks = [None] * len(logits) if valid is None else valid
    values = [1.] * len(logits) if weights is None else list(weights)
    _require(len(masks) == len(logits) and len(values) == len(logits),
             'aligned prefix validity masks/weights required')
    _require(all(type(w) in (int, float) and math.isfinite(w) and w > 0 for w in values),
             'finite positive prefix weights required')
    return torch.stack([A.class_nll(z, c, valid=m) * w
                        for z, c, m, w in zip(logits, targets, masks, values)]).sum()


def accumulate_gradients(model: A.V0Scorer, trajectories: Sequence[Trajectory], *, microbatch_size: int,
                         first_prefix_weight: float = 1.) -> float:
    """One effective trajectory denominator, including the short microbatch tail.

    Re-encode each canonical prefix once in a ragged batch. MoE routing is consumed
    by the scorer only; no auxiliary routing/entropy/value/flow objective is added.
    """
    _require(bool(trajectories) and type(microbatch_size) is int and microbatch_size > 0,
             'nonempty batch and positive microbatch required')
    model.zero_grad(set_to_none=True)
    total = 0.
    for start in range(0, len(trajectories), microbatch_size):
        chunk = trajectories[start:start + microbatch_size]
        examples = [e for t in chunk for e in t.examples]
        _require(all(t.examples for t in chunk), 'empty trajectory')
        pack = model.score_batch([e.state for e in examples], [e.menu.prefix for e in examples],
            [e.menu.legal_ids for e in examples], [e.menu.phi for e in examples], memories=[{} for _ in examples])
        offset, losses = 0, []
        for t in chunk:
            size = len(t.examples)
            weights = [first_prefix_weight] + [1.] * (size - 1)
            losses.append(path_nll(list(pack.logits[offset:offset+size]),
                [e.target_indices for e in t.examples], valid=list(pack.valid[offset:offset+size]),
                weights=weights))
            offset += size
        loss = torch.stack(losses).sum() / len(trajectories)
        _require(bool(torch.isfinite(loss)), 'nonfinite class-path loss')
        loss.backward()
        total += float(loss.detach())
    _require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
             'nonfinite class-path gradient')
    return total


@dataclass(frozen=True)
class TrainConfig:
    max_updates: int = 512
    batch_size: int = 4
    microbatch_size: int = 1
    seed: int = 0
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    optimizer: str = 'adam'
    schedule: str = 'constant'
    weight_decay: float = 0.0
    first_prefix_weight: float = 1.0
    min_learning_rate_ratio: float = 0.1

    def __post_init__(self):
        for name, upper in (('max_updates', 65536), ('batch_size', 128), ('microbatch_size', 128)):
            value = getattr(self, name)
            _require(type(value) is int and 1 <= value <= upper, 'bounded positive ' + name + ' required')
        _require(type(self.seed) is int and 0 <= self.seed < 2**32, 'uint32 seed required')
        for name in ('learning_rate', 'max_grad_norm', 'first_prefix_weight'):
            value = getattr(self, name)
            _require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'finite positive ' + name + ' required')
        _require(type(self.weight_decay) in (int, float) and math.isfinite(self.weight_decay)
                 and self.weight_decay >= 0, 'finite nonnegative weight_decay required')
        _require(type(self.min_learning_rate_ratio) in (int, float)
                 and math.isfinite(self.min_learning_rate_ratio)
                 and 0 <= self.min_learning_rate_ratio <= 1,
                 'min_learning_rate_ratio in [0,1] required')
        _require(self.optimizer in ('adam', 'adamw'), 'supported optimizer required')
        _require(self.schedule in ('constant', 'cosine'), 'supported learning-rate schedule required')


def scheduled_learning_rate(config: TrainConfig, update_index: int) -> float:
    """Stateless LR schedule keyed only by committed global update age."""
    _require(isinstance(config, TrainConfig) and type(update_index) is int
             and 0 <= update_index < config.max_updates, 'valid global update index required')
    if config.schedule == 'constant' or config.max_updates == 1:
        return float(config.learning_rate)
    progress = update_index / (config.max_updates - 1)
    ratio = config.min_learning_rate_ratio + (1 - config.min_learning_rate_ratio) * (1 + math.cos(math.pi * progress)) / 2
    return float(config.learning_rate * ratio)


def _make_optimizer(model: A.V0Scorer, config: TrainConfig) -> torch.optim.Optimizer:
    kwargs = dict(lr=scheduled_learning_rate(config, 0), betas=(.9, .999), eps=1e-8,
                  weight_decay=float(config.weight_decay), foreach=False)
    if config.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), **kwargs)
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _optimizer_identity(config: TrainConfig) -> str:
    # Preserve the legacy default identity exactly.  Non-default optimization
    # choices are already fully bound by identity['config'].
    if config.optimizer == 'adam' and config.schedule == 'constant' and config.weight_decay == 0:
        return 'ADAM_CONSTANT_LR_NO_WEIGHT_DECAY_NO_SCALER'
    return ('ADAM' if config.optimizer == 'adam' else 'ADAMW') + '_' + config.schedule.upper() + '_LR_NO_SCALER'


def _training_runtime(machine: str) -> tuple[str | None, dict]:
    """Select from the lock, never infer a production profile from versions.

    An unregistered machine can still describe a diagnostic CPU fixture, but
    require_production_runtime refuses it. The runtime key names a numeric
    profile, not evidence that a diagnostic process passed production preflight.
    """
    lock = tomllib.loads((ROOT / RUNTIME_PATH).read_text())
    key = lock['training_by_machine'].get(machine)
    return key, lock[key] if key is not None else {}


def require_production_runtime() -> None:
    machine = platform.machine()
    key, spec = _training_runtime(machine)
    _require(key is not None, 'no pinned training runtime registered for machine: ' + machine)
    # Preserve x86_64's registered-link precedence. Other profiles have no
    # implicit fallback to that link (or to any interpreter with equal versions).
    pinned = ROOT / spec['link'] if spec.get('link') else Path(spec['path'])
    if not pinned.exists():
        pinned = Path(spec['path'])
    _require(Path(sys.executable).resolve() == pinned.resolve()
             and '.'.join(map(str, sys.version_info[:3])) == spec['python']
             and str(torch.__version__) == spec['torch'] and np.__version__ == spec['numpy'],
             'production requires the pinned training interpreter/torch/numpy: ' + key)
    if 'sha256' in spec:
        _require(pinned.is_file() and file_sha(pinned) == spec['sha256'],
                 'production requires the pinned training interpreter SHA256: ' + key)
    _require(all(os.environ.get(k) == v for k, v in spec['environment'].items()),
             'production requires all pinned runtime environment variables')
    if 'pythonpath' in spec:
        site = Path(spec['pythonpath']).resolve()
        entries = [Path(p).resolve() for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
        _require(site in entries, 'production requires the pinned PYTHONPATH site: ' + str(site))
        # Merely naming the site in PYTHONPATH does not rule out shadowing by
        # another same-version install, or imports made before an env change.
        _require(all(getattr(module, '__file__', None)
                     and Path(module.__file__).resolve().is_relative_to(site) for module in (torch, np)),
                 'production requires torch/numpy from the pinned PYTHONPATH site: ' + str(site))


def runtime_identity(device: torch.device) -> dict:
    machine = platform.machine()
    key, spec = _training_runtime(machine)
    environment_keys = ('CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
    if 'pythonpath' in spec:
        environment_keys += ('PYTHONPATH',)
    # Existing exact identity joins in save/resume also bind these two fields;
    # old unlabelled commits and cross-architecture continuations cannot mix.
    result = {'runtime_key': key, 'machine': machine,
        'python': '.'.join(map(str, sys.version_info[:3])), 'executable': str(Path(sys.executable).resolve()),
        'torch': str(torch.__version__), 'numpy': np.__version__, 'device': str(device),
        'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'deterministic': torch.are_deterministic_algorithms_enabled(),
        'matmul_precision': torch.get_float32_matmul_precision(),
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'environment': {k: os.environ.get(k) for k in environment_keys}}
    if device.type == 'cuda':
        result.update(cuda_version=torch.version.cuda, device_name=torch.cuda.get_device_name(device),
                      device_capability=list(torch.cuda.get_device_capability(device)))
    return result


def _rng_state(device: torch.device) -> dict:
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}


def _restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'cuda':
        _require(len(state['torch_cuda']) == torch.cuda.device_count(), 'device RNG topology mismatch')
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    else:
        _require(state['torch_cuda'] == [], 'CPU continuation cannot import CUDA RNG')


class Cursor:
    """Whole-trajectory shuffle; stop at an epoch's short tail, never pad/wrap it."""
    def __init__(self, size: int, seed: int):
        self.size = size
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.order: list[int] = []
        self.cursor, self.epoch = 0, 0

    def take(self, count: int) -> list[int]:
        if self.cursor == len(self.order):
            self.order = torch.randperm(self.size, generator=self.generator).tolist()
            self.cursor = 0
            self.epoch += 1
        result = self.order[self.cursor:self.cursor + count]
        self.cursor += len(result)
        return result

    def state_dict(self) -> dict:
        return {'size': self.size, 'order': list(self.order), 'cursor': self.cursor,
                'epoch': self.epoch, 'generator': self.generator.get_state()}

    def load_state_dict(self, value: dict) -> None:
        _require(set(value) == set(self.state_dict()), 'sampler fields mismatch')
        _require(type(value['size']) is int and value['size'] == self.size, 'sampler dataset size mismatch')
        epoch, cursor, order = value['epoch'], value['cursor'], value['order']
        _require(type(epoch) is int and epoch >= 0 and type(cursor) is int and 0 <= cursor <= self.size,
                 'invalid sampling age/cursor')
        _require((epoch == 0 and order == [] and cursor == 0) or
                 (epoch > 0 and isinstance(order, list) and all(type(x) is int for x in order)
                  and sorted(order) == list(range(self.size))), 'invalid sampler permutation')
        state = value['generator']
        _require(isinstance(state, torch.Tensor) and state.dtype == torch.uint8
                 and state.shape == self.generator.get_state().shape, 'invalid sampling RNG')
        self.generator.set_state(state)
        self.epoch, self.cursor, self.order = epoch, cursor, list(order)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return type(value)((k, _cpu_tree(v)) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def _finite_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def _same_tree(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same_tree(x, y) for x, y in zip(a, b))
    return a == b


def _same_json(a: Any, b: Any) -> bool:
    """JSON-exact equality that does not conflate bool with int."""
    try:
        return canonical(a) == canonical(b)
    except (TypeError, ValueError, OverflowError):
        return False


def _is_sha256(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _closed(value: Any, fields: Sequence[str], message: str) -> None:
    _require(type(value) is dict and set(value) == set(fields), message)


def _validate_blob(value: Any, *, maximum: int, message: str) -> None:
    _closed(value, ('sha256', 'bytes'), message + ' schema mismatch')
    _require(_is_sha256(value['sha256']) and type(value['bytes']) is int
             and 0 < value['bytes'] <= maximum, message + ' identity mismatch')


def _open_regular(path: Path, *, maximum: int, message: str) -> tuple[int, os.stat_result]:
    """Open one bounded non-symlink inode for race-resistant transfer admission."""
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(message + ' must be a regular non-symlink file') from exc
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode), message + ' must be a regular non-symlink file')
        _require(0 < info.st_size <= maximum,
                 f'{message} exceeds explicit resource limit ({maximum} bytes)')
        return fd, info
    except BaseException:
        os.close(fd)
        raise


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in
               ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def _stream_sha(stream) -> str:
    stream.seek(0)
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b''):
        h.update(chunk)
    return h.hexdigest()


def _pinned_regular_json(path: Path, expected_sha256: str, *, message: str,
                         maximum: int = MAX_WEIGHTS_TRANSFER_JSON_BYTES) -> tuple[dict, dict]:
    _require(_is_sha256(expected_sha256), message + ' explicit SHA256 trust pin required')
    fd, before = _open_regular(path, maximum=maximum, message=message)
    with os.fdopen(fd, 'rb') as stream:
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    _require(_same_stat(before, after) and len(raw) == before.st_size,
             message + ' changed while being authenticated')
    observed = hashlib.sha256(raw).hexdigest()
    _require(observed == expected_sha256, message + ' hash mismatch')
    try:
        value = _json_loads(raw.decode('utf-8'))
    except UnicodeDecodeError as exc:
        raise ValueError(message + ' must be UTF-8 JSON') from exc
    _require(type(value) is dict, message + ' JSON object required')
    return value, {'sha256': observed, 'bytes': len(raw)}


def _validate_source_pin_map(value: Any, message: str) -> None:
    _require(type(value) is dict and bool(value), message + ' required')
    for path, digest_value in value.items():
        _require(type(path) is str and bool(path) and not path.startswith('/')
                 and '\\' not in path and all(part not in ('', '.', '..') for part in path.split('/')),
                 message + ' contains unsafe path')
        _require(_is_sha256(digest_value), message + ' contains invalid SHA256')


def _validate_historical_model_manifest(value: Any) -> None:
    """Validate native scorer identity while allowing only source-pin history.

    The accepted transfer may cross a reviewed source transition. Architecture,
    action/feature schemas, readout, numerics and every other identity field remain
    exact. Unknown fields are therefore refused rather than silently ignored.
    """
    try:
        _require(type(value) is dict, 'complete historical native V0 model manifest required')
        _validate_source_pin_map(value['source_pins'], 'historical model source pins')
        readout = value['readout']['mode'] if 'readout' in value else 'mean'
        expected = model_manifest(Config(**value['config']), diagnostic=value['diagnostic'],
                                  readout=readout)
        expected['source_pins'] = copy.deepcopy(value['source_pins'])
        _require(_same_json(value, expected),
                 'historical model architecture/schema/readout identity mismatch')
    except (KeyError, TypeError) as exc:
        raise ValueError('complete historical native V0 model manifest required') from exc


def _require_transfer_compatible_models(origin: dict, destination: dict) -> None:
    _validate_historical_model_manifest(origin)
    validate_model_manifest(destination)
    old = copy.deepcopy(origin); old.pop('source_pins')
    new = copy.deepcopy(destination); new.pop('source_pins')
    _require(_same_json(old, new),
             'origin and destination model architecture/schema/readout manifest mismatch')


def _tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def model_tensor_inventory(state: Any) -> dict:
    """Measure every serialized model tensor, in state-dict order, including bytes."""
    _require(type(state) in (dict, OrderedDict) and '_extra_state' in state and len(state) > 1,
             'complete model-only state dictionary required')
    _require(type(state['_extra_state']) is dict, 'model extra-state identity required')
    entries = []
    total = 0
    for name, tensor in state.items():
        if name == '_extra_state':
            continue
        _require(type(name) is str and bool(name) and type(tensor) is torch.Tensor
                 and tensor.device.type == 'cpu' and tensor.layout == torch.strided,
                 'unsupported transfer tensor state: ' + str(name))
        _require(bool(torch.isfinite(tensor).all()),
                 'nonfinite transfer tensor: ' + name)
        raw = _tensor_raw_bytes(tensor)
        total += len(raw)
        _require(total <= MAX_WEIGHTS_TRANSFER_TENSOR_BYTES,
                 f'weights-transfer tensors exceed explicit resource limit '
                 f'({MAX_WEIGHTS_TRANSFER_TENSOR_BYTES} bytes)')
        entries.append({'name': name, 'dtype': str(tensor.dtype),
                        'shape': list(tensor.shape), 'stride': list(tensor.stride()),
                        'storage_offset': tensor.storage_offset(),
                        'data': {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}})
    _require(bool(entries), 'at least one transfer tensor required')
    return {'schema': WEIGHTS_TRANSFER_INVENTORY_SCHEMA, 'sha256': _digest(entries),
            'tensors': len(entries), 'bytes': total, 'entries': entries}


def _validate_tensor_inventory(value: Any) -> None:
    _closed(value, ('schema', 'sha256', 'tensors', 'bytes', 'entries'),
            'tensor inventory schema mismatch')
    _require(value['schema'] == WEIGHTS_TRANSFER_INVENTORY_SCHEMA
             and _is_sha256(value['sha256'])
             and type(value['tensors']) is int and value['tensors'] > 0
             and type(value['bytes']) is int
             and 0 < value['bytes'] <= MAX_WEIGHTS_TRANSFER_TENSOR_BYTES
             and type(value['entries']) is list
             and len(value['entries']) == value['tensors'],
             'tensor inventory identity mismatch')
    names = []
    total = 0
    for entry in value['entries']:
        _closed(entry, ('name', 'dtype', 'shape', 'stride', 'storage_offset', 'data'),
                'tensor inventory entry schema mismatch')
        _require(type(entry['name']) is str and bool(entry['name'])
                 and type(entry['dtype']) is str and bool(entry['dtype'])
                 and type(entry['shape']) is list
                 and all(type(x) is int and x >= 0 for x in entry['shape'])
                 and type(entry['stride']) is list
                 and len(entry['stride']) == len(entry['shape'])
                 and all(type(x) is int for x in entry['stride'])
                 and type(entry['storage_offset']) is int and entry['storage_offset'] >= 0,
                 'tensor inventory entry identity mismatch')
        _validate_blob(entry['data'], maximum=MAX_WEIGHTS_TRANSFER_TENSOR_BYTES,
                       message='tensor data')
        names.append(entry['name'])
        total += entry['data']['bytes']
    _require(len(set(names)) == len(names), 'duplicate tensor name in inventory')
    _require(total == value['bytes'] and _digest(value['entries']) == value['sha256'],
             'tensor inventory digest/count mismatch')


def _inventory_layout(value: dict) -> list[dict]:
    _validate_tensor_inventory(value)
    return [{k: (v['bytes'] if k == 'data' else copy.deepcopy(v))
             for k, v in entry.items()} for entry in value['entries']]


def _load_transfer_snapshot(path: Path) -> tuple[dict, dict]:
    """Load exactly the regular-file bytes whose hash is returned.

    Hashing before and after torch.load on the same open inode prevents path
    substitution or an in-place rewrite from authenticating one payload while
    extracting tensors from another.
    """
    fd, before = _open_regular(path, maximum=MAX_WEIGHTS_TRANSFER_STATE_BYTES,
                               message='origin state payload')
    with os.fdopen(fd, 'rb') as stream:
        first_sha = _stream_sha(stream)
        stream.seek(0)
        try:
            value = torch.load(stream, map_location='cpu', weights_only=True)
        except MemoryError as exc:
            raise ValueError('weights-transfer resource exhaustion while loading origin state') from exc
        except OSError as exc:
            if getattr(exc, 'errno', None) == 12:
                raise ValueError('weights-transfer resource exhaustion while loading origin state') from exc
            raise ValueError('malformed or unreadable origin state payload') from exc
        except RuntimeError as exc:
            text = str(exc).lower()
            if 'out of memory' in text or 'cannot allocate memory' in text:
                raise ValueError('weights-transfer resource exhaustion while loading origin state') from exc
            raise ValueError('malformed origin state payload') from exc
        except (pickle.UnpicklingError, EOFError, TypeError, ValueError) as exc:
            raise ValueError('malformed origin state payload') from exc
        after_load = os.fstat(stream.fileno())
        second_sha = _stream_sha(stream)
        after_hash = os.fstat(stream.fileno())
    _require(_same_stat(before, after_load) and _same_stat(before, after_hash)
             and first_sha == second_sha,
             'origin state payload changed while being authenticated')
    _require(type(value) is dict, 'native V0 origin state object required')
    _closed(value, ('schema', 'identity', 'parameter_order', 'model', 'optimizer', 'rng',
                    'sampler', 'global_update', 'samples_seen', 'model_training',
                    'optimizer_state_ids'), 'complete native V0 origin state required')
    return value, {'sha256': first_sha, 'bytes': before.st_size}


def _validate_training_identity_shape(identity: Any) -> None:
    _require(type(identity) is dict, 'native V0 training identity required')
    required = {'schema', 'objective', 'model', 'dataset', 'config', 'sources', 'runtime',
                'optimizer', 'loss_weighting'}
    _require(set(identity) in (required, required | {'initialization'}),
             'unknown native V0 training identity fields')
    _require(identity['schema'] == SCHEMA and identity['objective'] == OBJECTIVE,
             'native V0 training identity mismatch')
    _validate_historical_model_manifest(identity['model'])
    _validate_source_pin_map(identity['sources'], 'training source pins')
    _require(type(identity['dataset']) is dict and type(identity['config']) is dict
             and type(identity['runtime']) is dict and type(identity['optimizer']) is str
             and type(identity['loss_weighting']) is dict,
             'native V0 training identity fields malformed')


def _validate_origin_authentication(authentication: dict, commit: dict, snapshot: dict,
                                    commit_blob: dict, state_blob: dict) -> dict:
    _closed(authentication, ('schema', 'status', 'kind', 'scope', 'commit', 'state',
            'training_identity_sha256', 'training_source_pins', 'model_manifest',
            'runtime', 'tensor_inventory', 'authenticator'),
            'origin authentication schema mismatch')
    _require(authentication['schema'] == WEIGHTS_TRANSFER_ORIGIN_SCHEMA
             and authentication['status'] == 'AUTHENTICATED'
             and authentication['kind'] == 'NATIVE_V0_BC_COMMIT',
             'accepted native V0 origin authentication required')
    _validate_blob(authentication['commit'], maximum=2_000_000,
                   message='origin COMMIT authentication')
    _validate_blob(authentication['state'], maximum=MAX_WEIGHTS_TRANSFER_STATE_BYTES,
                   message='origin state authentication')
    _require(_same_json(authentication['commit'], commit_blob)
             and _same_json(authentication['state'], state_blob),
             'origin COMMIT/state changed after authentication')
    _closed(commit, ('schema', 'status', 'identity', 'state_sha256', 'state_bytes',
                     'global_update', 'samples_seen', 'parent_manifest_sha256',
                     'metrics', 'scope'), 'native V0 COMMIT schema mismatch')
    _require(commit['schema'] == COMMIT_SCHEMA and commit['status'] == 'COMMITTED',
             'not a committed native V0 origin')
    _require(commit['scope'] in ('DIAGNOSTIC_CPU_NOT_STRENGTH', 'V0_BC_NOT_STRENGTH')
             and authentication['scope'] == commit['scope'],
             'origin scope mismatch')
    _require(commit['state_sha256'] == state_blob['sha256']
             and commit['state_bytes'] == state_blob['bytes'],
             'origin state hash/size mismatch')
    identity = commit['identity']
    _validate_training_identity_shape(identity)
    _require(snapshot['schema'] == SCHEMA and _same_json(snapshot['identity'], identity)
             and snapshot['global_update'] == commit['global_update']
             and snapshot['samples_seen'] == commit['samples_seen']
             and snapshot['model_training'] is True,
             'origin state/COMMIT training identity mismatch')
    _require(_is_sha256(authentication['training_identity_sha256'])
             and authentication['training_identity_sha256'] == _digest(identity),
             'origin training identity digest mismatch')
    _validate_source_pin_map(authentication['training_source_pins'],
                             'authenticated training source pins')
    _require(_same_json(authentication['training_source_pins'], identity['sources'])
             and _same_json(authentication['model_manifest'], identity['model'])
             and _same_json(authentication['runtime'], identity['runtime']),
             'origin authentication/training identity mismatch')
    _closed(authentication['authenticator'], ('entrypoint', 'source_pins'),
            'origin authenticator schema mismatch')
    _require(authentication['authenticator']['entrypoint']
             == 'farm_neat.training.v0_bc.Trainer.resume',
             'strict native Trainer.resume origin authentication required')
    _validate_source_pin_map(authentication['authenticator']['source_pins'],
                             'origin authenticator source pins')
    _require(_same_json(authentication['authenticator']['source_pins'],
                        identity['sources']),
             'origin was not authenticated under its recorded frozen source closure')
    _require(type(snapshot['model']) in (dict, OrderedDict)
             and _same_json(snapshot['model'].get('_extra_state'), identity['model']),
             'origin model extra-state/training identity mismatch')
    measured = model_tensor_inventory(snapshot['model'])
    _validate_tensor_inventory(authentication['tensor_inventory'])
    _require(_same_json(authentication['tensor_inventory'], measured),
             'origin tensor inventory or tensor bytes changed after authentication')
    return measured


def _transfer_claims() -> dict:
    return {'behavior_equivalence': False, 'historical_migration': False,
            'resume': False, 'source_equivalence': False, 'strength': False}


def _transfer_implementation() -> dict:
    return {'version': WEIGHTS_TRANSFER_IMPLEMENTATION,
            'path': WEIGHTS_TRANSFER_SOURCE_PATH,
            'sha256': file_sha(__file__)}


def weights_transfer_permit(origin_authentication: dict, origin_authentication_sha256: str,
                            destination_model_manifest: dict, *, status: str = 'DRAFT') -> dict:
    """Build the closed permit document; its creation is not its authorization.

    Production admission still requires the separately retained hash of a permit
    whose status was independently changed to ACCEPTED_FOR_BC_WEIGHTS_TRANSFER.
    Diagnostic unit fixtures use the explicit FIXTURE_ONLY scope.
    """
    _require(_is_sha256(origin_authentication_sha256),
             'origin authentication SHA256 trust pin required')
    _closed(origin_authentication, ('schema', 'status', 'kind', 'scope', 'commit', 'state',
            'training_identity_sha256', 'training_source_pins', 'model_manifest',
            'runtime', 'tensor_inventory', 'authenticator'),
            'origin authentication schema mismatch')
    _require(origin_authentication['schema'] == WEIGHTS_TRANSFER_ORIGIN_SCHEMA
             and origin_authentication['status'] == 'AUTHENTICATED',
             'authenticated origin record required for permit')
    _validate_historical_model_manifest(origin_authentication['model_manifest'])
    _validate_tensor_inventory(origin_authentication['tensor_inventory'])
    validate_model_manifest(destination_model_manifest)
    _require(status in ('DRAFT', 'FIXTURE_ONLY', 'ACCEPTED_FOR_BC_WEIGHTS_TRANSFER'),
             'known transfer permit status required')
    return {'schema': WEIGHTS_TRANSFER_PERMIT_SCHEMA, 'status': status,
            'operation': WEIGHTS_TRANSFER_OPERATION,
            'origin_authentication_sha256': origin_authentication_sha256,
            'origin_commit_sha256': origin_authentication['commit']['sha256'],
            'origin_state_sha256': origin_authentication['state']['sha256'],
            'origin_model_manifest_sha256': _digest(origin_authentication['model_manifest']),
            'destination_model_manifest_sha256': _digest(destination_model_manifest),
            'source_transition': {
                'from': copy.deepcopy(origin_authentication['model_manifest']['source_pins']),
                'to': copy.deepcopy(destination_model_manifest['source_pins'])},
            'tensor_inventory_sha256': origin_authentication['tensor_inventory']['sha256'],
            'implementation': _transfer_implementation(), 'claims': _transfer_claims()}


def _validate_weights_transfer_permit(permit: dict, permit_sha256: str,
                                      authentication: dict, authentication_sha256: str,
                                      destination_model_manifest: dict) -> None:
    _require(_is_sha256(permit_sha256), 'transfer permit SHA256 trust pin required')
    _closed(permit, ('schema', 'status', 'operation', 'origin_authentication_sha256',
            'origin_commit_sha256', 'origin_state_sha256', 'origin_model_manifest_sha256',
            'destination_model_manifest_sha256', 'source_transition',
            'tensor_inventory_sha256', 'implementation', 'claims'),
            'transfer permit schema mismatch')
    _require(permit['schema'] == WEIGHTS_TRANSFER_PERMIT_SCHEMA
             and permit['operation'] == WEIGHTS_TRANSFER_OPERATION,
             'BC weights-transfer permit required')
    _require(permit['status'] in ('FIXTURE_ONLY', 'ACCEPTED_FOR_BC_WEIGHTS_TRANSFER'),
             'explicit accepted BC weights-transfer permit required')
    _require(permit['origin_authentication_sha256'] == authentication_sha256
             and permit['origin_commit_sha256'] == authentication['commit']['sha256']
             and permit['origin_state_sha256'] == authentication['state']['sha256']
             and permit['origin_model_manifest_sha256'] == _digest(authentication['model_manifest'])
             and permit['destination_model_manifest_sha256'] == _digest(destination_model_manifest)
             and permit['tensor_inventory_sha256'] == authentication['tensor_inventory']['sha256'],
             'transfer permit origin/destination binding mismatch')
    _closed(permit['source_transition'], ('from', 'to'),
            'transfer source-transition schema mismatch')
    _validate_source_pin_map(permit['source_transition']['from'], 'permit historical source pins')
    _validate_source_pin_map(permit['source_transition']['to'], 'permit destination source pins')
    _require(_same_json(permit['source_transition']['from'],
                        authentication['model_manifest']['source_pins'])
             and _same_json(permit['source_transition']['to'],
                            destination_model_manifest['source_pins']),
             'transfer permit source transition mismatch')
    _require(_same_json(permit['implementation'], _transfer_implementation()),
             'transfer implementation/source hash not permitted')
    _require(_same_json(permit['claims'], _transfer_claims()),
             'weights transfer cannot claim resume/source/behavior equivalence, migration or strength')
    if destination_model_manifest['diagnostic']:
        _require(permit['status'] == 'FIXTURE_ONLY',
                 'diagnostic transfer requires explicit FIXTURE_ONLY permit')
    else:
        _require(permit['status'] == 'ACCEPTED_FOR_BC_WEIGHTS_TRANSFER'
                 and not authentication['model_manifest']['diagnostic'],
                 'diagnostic-as-production origin or unaccepted production permit refused')


def _admit_weights_transfer(destination_model_manifest: dict, origin_directory: str | Path,
                            origin_commit_sha256: str,
                            origin_authentication: str | Path,
                            origin_authentication_sha256: str,
                            transfer_permit: str | Path,
                            transfer_permit_sha256: str) -> dict:
    """Authenticate every immutable input and tensor byte before destination allocation."""
    validate_model_manifest(destination_model_manifest)
    directory = Path(origin_directory)
    authentication_path = Path(origin_authentication)
    permit_path = Path(transfer_permit)
    authentication, _ = _pinned_regular_json(
        authentication_path, origin_authentication_sha256, message='origin authentication')
    permit, _ = _pinned_regular_json(
        permit_path, transfer_permit_sha256, message='weights-transfer permit')
    commit_path = directory / 'COMMIT.json'
    commit, commit_blob = _pinned_regular_json(
        commit_path, origin_commit_sha256, message='origin COMMIT')
    state_path = directory / 'state.pt'
    snapshot, state_blob = _load_transfer_snapshot(state_path)
    inventory = _validate_origin_authentication(
        authentication, commit, snapshot, commit_blob, state_blob)
    _validate_weights_transfer_permit(
        permit, transfer_permit_sha256, authentication,
        origin_authentication_sha256, destination_model_manifest)
    _require_transfer_compatible_models(authentication['model_manifest'],
                                        destination_model_manifest)
    model_state = snapshot.pop('model')
    del snapshot
    return {'model_state': model_state, 'tensor_inventory': inventory,
            'authentication': authentication, 'permit': permit,
            'origin_commit_sha256': origin_commit_sha256,
            'origin_state_sha256': state_blob['sha256'],
            'origin_authentication_sha256': origin_authentication_sha256,
            'transfer_permit_sha256': transfer_permit_sha256}


def _validate_weights_transfer_initialization(value: dict, base_identity: dict) -> None:
    _closed(value, ('schema', 'operation', 'origin_authentication_schema',
            'origin_authentication_sha256', 'origin_commit_sha256', 'origin_state_sha256',
            'origin_training_identity_sha256', 'transfer_permit_schema',
            'transfer_permit_sha256', 'tensor_inventory_sha256', 'tensor_count',
            'tensor_bytes', 'transfer_implementation', 'destination_identity_sha256',
            'destination_model_manifest_sha256', 'destination_sources_sha256', 'claims'),
            'weights-transfer initialization schema mismatch')
    _require(value['schema'] == WEIGHTS_TRANSFER_INITIALIZATION_SCHEMA
             and value['operation'] == WEIGHTS_TRANSFER_OPERATION
             and value['origin_authentication_schema'] == WEIGHTS_TRANSFER_ORIGIN_SCHEMA
             and value['transfer_permit_schema'] == WEIGHTS_TRANSFER_PERMIT_SCHEMA,
             'weights-transfer initialization identity mismatch')
    for field in ('origin_authentication_sha256', 'origin_commit_sha256',
                  'origin_state_sha256', 'origin_training_identity_sha256',
                  'transfer_permit_sha256', 'tensor_inventory_sha256',
                  'destination_identity_sha256', 'destination_model_manifest_sha256',
                  'destination_sources_sha256'):
        _require(_is_sha256(value[field]), 'invalid initialization SHA256: ' + field)
    _require(type(value['tensor_count']) is int and value['tensor_count'] > 0
             and type(value['tensor_bytes']) is int
             and 0 < value['tensor_bytes'] <= MAX_WEIGHTS_TRANSFER_TENSOR_BYTES,
             'invalid initialization tensor inventory summary')
    _require(_same_json(value['claims'], _transfer_claims()),
             'invalid weights-transfer initialization claims')
    _require(_same_json(value['transfer_implementation'], _transfer_implementation())
             and value['transfer_implementation']['sha256']
             == base_identity['sources'][WEIGHTS_TRANSFER_SOURCE_PATH],
             'weights-transfer initialization implementation/source mismatch')
    _require(value['destination_identity_sha256'] == _digest(base_identity)
             and value['destination_model_manifest_sha256'] == _digest(base_identity['model'])
             and value['destination_sources_sha256'] == _digest(base_identity['sources']),
             'weights-transfer initialization destination identity mismatch')


def _make_weights_transfer_initialization(admission: dict, base_identity: dict) -> dict:
    authentication = admission['authentication']
    inventory = admission['tensor_inventory']
    value = {'schema': WEIGHTS_TRANSFER_INITIALIZATION_SCHEMA,
        'operation': WEIGHTS_TRANSFER_OPERATION,
        'origin_authentication_schema': WEIGHTS_TRANSFER_ORIGIN_SCHEMA,
        'origin_authentication_sha256': admission['origin_authentication_sha256'],
        'origin_commit_sha256': admission['origin_commit_sha256'],
        'origin_state_sha256': admission['origin_state_sha256'],
        'origin_training_identity_sha256': authentication['training_identity_sha256'],
        'transfer_permit_schema': WEIGHTS_TRANSFER_PERMIT_SCHEMA,
        'transfer_permit_sha256': admission['transfer_permit_sha256'],
        'tensor_inventory_sha256': inventory['sha256'],
        'tensor_count': inventory['tensors'], 'tensor_bytes': inventory['bytes'],
        'transfer_implementation': copy.deepcopy(admission['permit']['implementation']),
        'destination_identity_sha256': _digest(base_identity),
        'destination_model_manifest_sha256': _digest(base_identity['model']),
        'destination_sources_sha256': _digest(base_identity['sources']),
        'claims': _transfer_claims()}
    _validate_weights_transfer_initialization(value, base_identity)
    return value


def _write_json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Trainer:
    """Fresh initialization, strict resume, or permitted tensor-only new lineage."""
    def __init__(self, data: Dataset, identity: dict, config: TrainConfig, *, device: str = 'cpu',
                 _initialization: dict | None = None):
        validate_model_manifest(identity)
        _require(isinstance(data, Dataset) and bool(data.trajectories), 'validated nonempty dataset required')
        _require(isinstance(config, TrainConfig), 'typed training configuration required')
        if not identity['diagnostic']:
            require_production_runtime()
        self.device = torch.device(device)
        _require(self.device.type in ('cpu', 'cuda'), 'CPU or CUDA FP32 runtime required')
        _require(not identity['diagnostic'] or self.device.type == 'cpu', 'diagnostic fixtures are CPU only')
        if self.device.type == 'cuda':
            _require(torch.cuda.is_available(), 'CUDA requested but unavailable')
            self.device = torch.device('cuda', torch.cuda.current_device() if self.device.index is None else self.device.index)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(config.seed)
        self.data, self.config = data, config
        # identity was authenticated above; there is no independent resume-mode override.
        self.model = A.V0Scorer(Config(**identity['config']), diagnostic=identity['diagnostic'],
                               readout=identity.get('readout', {}).get('mode', 'mean')).to(self.device)
        _require(self.model.get_extra_state() == identity, 'constructed model identity mismatch')
        self.model.train()
        self.optimizer = _make_optimizer(self.model, config)
        self.parameter_order = [name for name, _ in self.model.named_parameters()]
        self.identity = {'schema': SCHEMA, 'objective': OBJECTIVE, 'model': copy.deepcopy(identity),
            'dataset': data.binding, 'config': asdict(config), 'sources': source_pins(),
            'runtime': runtime_identity(self.device), 'optimizer': _optimizer_identity(config),
            'loss_weighting': {'first_prefix_weight': float(config.first_prefix_weight)}}
        if _initialization is not None:
            _validate_weights_transfer_initialization(_initialization, self.identity)
            self.identity['initialization'] = copy.deepcopy(_initialization)
        self.sampler = Cursor(len(data.trajectories), config.seed)
        self.global_update, self.samples_seen = 0, 0
        self.parent_sha256: str | None = None
        self.parent_directory: Path | None = None
        self.last_attempt: dict | None = None
        self._failed = False

    def step(self) -> dict:
        _require(not self._failed, 'failed trainer must resume its last committed parent')
        _require(self.global_update < self.config.max_updates, 'global update cap reached')
        try:
            indices = self.sampler.take(self.config.batch_size)
            batch = [self.data.trajectories[i] for i in indices]
            self.last_attempt = {'global_update': self.global_update + 1,
                                 'trajectory_ids': [t.trajectory_id for t in batch]}
            learning_rate = scheduled_learning_rate(self.config, self.global_update)
            for group in self.optimizer.param_groups:
                group['lr'] = learning_rate
            loss = accumulate_gradients(self.model, batch, microbatch_size=self.config.microbatch_size,
                                        first_prefix_weight=self.config.first_prefix_weight)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm,
                                                 error_if_nonfinite=True, foreach=False)
            self.optimizer.step()
            _require(_finite_tree(self.model.state_dict()) and _finite_tree(self.optimizer.state_dict()),
                     'nonfinite optimizer update')
            self.global_update += 1
            self.samples_seen += len(batch)
            return {'global_update': self.global_update, 'samples_seen': self.samples_seen,
                    'trajectory_ids': [t.trajectory_id for t in batch], 'loss': loss, 'grad_norm': float(norm),
                    'learning_rate': learning_rate}
        except BaseException:
            self._failed = True
            raise

    def snapshot(self) -> dict:
        _require(not self._failed, 'failed update is not a continuation')
        return _cpu_tree({'schema': SCHEMA, 'identity': self.identity,
            'parameter_order': self.parameter_order, 'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'optimizer_state_ids': sorted(self.optimizer.state_dict()['state']), 'rng': _rng_state(self.device),
            'sampler': self.sampler.state_dict(), 'global_update': self.global_update,
            'samples_seen': self.samples_seen, 'model_training': self.model.training})

    def _validate_snapshot(self, value: dict) -> None:
        _require(set(value) == {'schema', 'identity', 'parameter_order', 'model', 'optimizer', 'rng',
            'sampler', 'global_update', 'samples_seen', 'model_training', 'optimizer_state_ids'},
            'complete continuation fields required')
        _require(value['schema'] == SCHEMA and value['identity'] == self.identity, 'continuation identity mismatch')
        _require(value['parameter_order'] == self.parameter_order, 'optimizer parameter order mismatch')
        age, seen = value['global_update'], value['samples_seen']
        _require(type(age) is int and 0 <= age <= self.config.max_updates
                 and type(seen) is int and seen >= 0 and value['model_training'] is True,
                 'invalid training age/exposure/mode')
        probe = Cursor(len(self.data.trajectories), 0)
        probe.load_state_dict(value['sampler'])
        _require(probe.epoch == 0 or (probe.cursor > 0 and
                 (probe.cursor == probe.size or probe.cursor % self.config.batch_size == 0)),
                 'continuation must be at a complete batch boundary')
        expected_seen = 0 if probe.epoch == 0 else (probe.epoch-1)*probe.size + probe.cursor
        expected_age = 0 if probe.epoch == 0 else ((probe.epoch-1)*math.ceil(probe.size/self.config.batch_size)
                                                 + math.ceil(probe.cursor/self.config.batch_size))
        _require(seen == expected_seen and age == expected_age, 'sampler/global update/exposure join mismatch')
        _require(_finite_tree(value), 'nonfinite continuation tensor/value')
        expected_optimizer = _make_optimizer(self.model, self.config).state_dict()
        expected_lr = scheduled_learning_rate(self.config, max(age - 1, 0))
        for group in expected_optimizer['param_groups']:
            group['lr'] = expected_lr
        _require(set(value['optimizer']) == set(expected_optimizer)
                 and value['optimizer']['param_groups'] == expected_optimizer['param_groups'],
                 'optimizer settings/parameter IDs mismatch')
        params = list(self.model.parameters())
        active = value['optimizer_state_ids']
        _require(isinstance(active, list) and all(type(i) is int for i in active)
                 and active == sorted(value['optimizer']['state'])
                 and (bool(active) == (age > 0)), 'incomplete optimizer state inventory')
        for key, state in value['optimizer']['state'].items():
            _require(type(key) is int and 0 <= key < len(params) and set(state) == {'step', 'exp_avg', 'exp_avg_sq'},
                     'unexpected Adam-family state')
            _require(isinstance(state['step'], torch.Tensor) and state['step'].numel() == 1
                     and float(state['step']).is_integer() and 0 < float(state['step']) <= age,
                     'Adam-family step/global age mismatch')
            for name in ('exp_avg', 'exp_avg_sq'):
                _require(isinstance(state[name], torch.Tensor) and state[name].shape == params[key].shape
                         and state[name].dtype == params[key].dtype, 'Adam moment shape/dtype mismatch')
        _require(set(value['rng']) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'}, 'complete RNG state required')
        # Validate RNG restoration without changing the caller's current RNG.
        before = _rng_state(self.device)
        try:
            _restore_rng(value['rng'], self.device)
        finally:
            _restore_rng(before, self.device)

    def save(self, directory: str | Path, *, metrics: dict | None = None) -> dict:
        directory = Path(directory)
        _require(self.identity['sources'] == source_pins(), 'live sources changed during training')
        _require(self.identity['runtime'] == runtime_identity(self.device), 'live numerics/runtime changed')
        value = self.snapshot()
        self._validate_snapshot(value)
        directory.mkdir(parents=True, exist_ok=False)
        marker = directory / 'COMMIT.json'
        try:
            state_path = directory / 'state.pt'
            with state_path.open('xb') as stream:
                torch.save(value, stream)
                stream.flush(); os.fsync(stream.fileno())
            restored = torch.load(state_path, map_location='cpu', weights_only=True)
            _require(_same_tree(value, restored), 'checkpoint round-trip mismatch')
            self._validate_snapshot(restored)
            commit = {'schema': COMMIT_SCHEMA, 'status': 'COMMITTED', 'identity': self.identity,
                'state_sha256': file_sha(state_path), 'state_bytes': state_path.stat().st_size,
                'global_update': self.global_update, 'samples_seen': self.samples_seen,
                'parent_manifest_sha256': self.parent_sha256, 'metrics': metrics,
                'scope': 'DIAGNOSTIC_CPU_NOT_STRENGTH' if self.model.diagnostic else 'V0_BC_NOT_STRENGTH'}
            temporary = directory / '.COMMIT.pending'
            _write_json(temporary, commit)
            _require(_json_read(temporary) == commit, 'commit JSON round-trip mismatch')
            _fsync_dir(directory)
            os.link(temporary, marker)  # atomic publication, exclusive; never overwrite a parent
            _fsync_dir(directory)
            temporary.unlink()
            _fsync_dir(directory.parent)
            self.parent_sha256 = file_sha(marker)
            self.parent_directory = directory.resolve()
            return commit
        except BaseException:
            if marker.exists():
                marker.unlink()  # a reported failed publication cannot be selected as a parent
            self._failed = True
            raise

    @classmethod
    def from_weights_transfer(cls, data: Dataset, identity: dict, config: TrainConfig,
                              origin_directory: str | Path, origin_commit_sha256: str,
                              origin_authentication: str | Path,
                              origin_authentication_sha256: str,
                              transfer_permit: str | Path,
                              transfer_permit_sha256: str, *, device: str = 'cpu') -> Trainer:
        """Start a new lineage from authenticated model tensors only.

        Admission and complete origin tensor measurement happen before allocating the
        destination Trainer. The destination is then constructed normally under its
        current source/runtime/data/schedule. No origin optimizer, sampler, counters or
        RNG object is passed to the new lineage.
        """
        admission = _admit_weights_transfer(
            identity, origin_directory, origin_commit_sha256,
            origin_authentication, origin_authentication_sha256,
            transfer_permit, transfer_permit_sha256)
        trainer = cls(data, identity, config, device=device)
        permitted_implementation = admission['permit']['implementation']
        _require(_same_json(_transfer_implementation(), permitted_implementation)
                 and trainer.identity['sources'][WEIGHTS_TRANSFER_SOURCE_PATH]
                 == permitted_implementation['sha256'],
                 'weights-transfer implementation/source changed after admission')
        fresh_optimizer = _cpu_tree(trainer.optimizer.state_dict())
        fresh_sampler = _cpu_tree(trainer.sampler.state_dict())
        fresh_rng = _rng_state(trainer.device)
        _require(trainer.global_update == 0 and trainer.samples_seen == 0
                 and fresh_optimizer['state'] == {},
                 'destination Trainer did not start with fresh continuation state')

        origin_state = admission['model_state']
        # Inventory measures serialized CPU tensors, not live CUDA parameters.
        destination_state = _cpu_tree(trainer.model.state_dict())
        destination_inventory = model_tensor_inventory(destination_state)
        _require(_same_json(_inventory_layout(admission['tensor_inventory']),
                            _inventory_layout(destination_inventory)),
                 'origin/destination tensor names/order/shapes/dtypes mismatch')
        _require(set(origin_state) == set(destination_state)
                 and list(origin_state) == list(destination_state),
                 'origin/destination complete tensor key order mismatch')
        imported = OrderedDict()
        for name in destination_state:
            imported[name] = (copy.deepcopy(identity) if name == '_extra_state'
                              else origin_state[name])
        # V0Scorer preflights the complete key set and every shape/dtype before any
        # copy. The destination extra-state stays current; only tensors cross.
        trainer.model.load_state_dict(imported)
        del imported, destination_state, destination_inventory, origin_state
        admission.pop('model_state')
        observed = model_tensor_inventory(_cpu_tree(trainer.model.state_dict()))
        _require(_same_json(observed, admission['tensor_inventory']),
                 'weights-transfer tensor bytes changed during destination copy')
        _require(_same_tree(trainer.optimizer.state_dict(), fresh_optimizer)
                 and _same_tree(trainer.sampler.state_dict(), fresh_sampler)
                 and _same_tree(_rng_state(trainer.device), fresh_rng)
                 and trainer.global_update == 0 and trainer.samples_seen == 0,
                 'weights transfer imported or changed non-model continuation state')
        initialization = _make_weights_transfer_initialization(admission, trainer.identity)
        trainer.identity['initialization'] = initialization
        return trainer

    @classmethod
    def resume(cls, data: Dataset, identity: dict, config: TrainConfig, directory: str | Path,
               expected_sha256: str, *, device: str = 'cpu') -> Trainer:
        directory = Path(directory)
        commit = _pinned_json(directory / 'COMMIT.json', expected_sha256)
        _require(commit.get('schema') == COMMIT_SCHEMA and commit.get('status') == 'COMMITTED',
                 'not a committed native V0 checkpoint')
        # Refuse a different feature/model schema before allocating a fresh scorer
        # or resetting RNGs in __init__. The full continuation join below remains
        # mandatory; this is a fail-fast check, never a load-time rebind.
        _require(isinstance(commit.get('identity'), dict)
                 and commit['identity'].get('model') == identity,
                 'parent dataset/model/source/runtime/schedule mismatch')
        path = directory / 'state.pt'
        _require(path.stat().st_size == commit['state_bytes'] and file_sha(path) == commit['state_sha256'],
                 'checkpoint payload hash/size mismatch')
        initialization = (commit['identity'].get('initialization')
                          if isinstance(commit.get('identity'), dict) else None)
        trainer = cls(data, identity, config, device=device, _initialization=initialization)
        _require(commit['identity'] == trainer.identity, 'parent dataset/model/source/runtime/schedule mismatch')
        value = torch.load(path, map_location='cpu', weights_only=True)
        trainer._validate_snapshot(value)
        _require(value['global_update'] == commit['global_update'] and value['samples_seen'] == commit['samples_seen'],
                 'commit age/exposure mismatch')
        trainer.model.load_state_dict(value['model'])  # accepted scorer preflights identity, keys and dtypes
        trainer.optimizer.load_state_dict(value['optimizer'])
        trainer.sampler.load_state_dict(value['sampler'])
        trainer.global_update, trainer.samples_seen = value['global_update'], value['samples_seen']
        trainer.parent_sha256 = expected_sha256
        trainer.parent_directory = directory.resolve()
        trainer.model.train(value['model_training'])
        _restore_rng(value['rng'], trainer.device)  # LAST: initialization/load cannot consume resumed RNG
        return trainer


def authenticate_weights_transfer_origin(data: Dataset, identity: dict, config: TrainConfig,
                                         directory: str | Path, expected_sha256: str,
                                         *, device: str = 'cpu') -> dict:
    """Create an origin observation only after strict native resume succeeds.

    Run this under the origin's recorded frozen closure. The returned JSON is an
    observation, not self-authenticating authority; transfer still requires its
    separately retained file hash and an independently accepted transfer permit.
    """
    requested = torch.device(device)
    if requested.type == 'cuda' and requested.index is None and torch.cuda.is_available():
        requested = torch.device('cuda', torch.cuda.current_device())
    before = _rng_state(requested)
    try:
        # This is the accepted native authenticator. It verifies the full training
        # continuation; the later transfer never calls it under changed source.
        Trainer.resume(data, identity, config, directory, expected_sha256, device=device)
        directory = Path(directory)
        commit_path = directory / 'COMMIT.json'
        state_path = directory / 'state.pt'
        commit, commit_blob = _pinned_regular_json(
            commit_path, expected_sha256, message='origin COMMIT')
        snapshot, state_blob = _load_transfer_snapshot(state_path)
        inventory = model_tensor_inventory(snapshot['model'])
        record = {'schema': WEIGHTS_TRANSFER_ORIGIN_SCHEMA, 'status': 'AUTHENTICATED',
            'kind': 'NATIVE_V0_BC_COMMIT', 'scope': commit['scope'],
            'commit': commit_blob, 'state': state_blob,
            'training_identity_sha256': _digest(commit['identity']),
            'training_source_pins': copy.deepcopy(commit['identity']['sources']),
            'model_manifest': copy.deepcopy(commit['identity']['model']),
            'runtime': copy.deepcopy(commit['identity']['runtime']),
            'tensor_inventory': inventory,
            'authenticator': {'entrypoint': 'farm_neat.training.v0_bc.Trainer.resume',
                              'source_pins': copy.deepcopy(commit['identity']['sources'])}}
        _validate_origin_authentication(record, commit, snapshot, commit_blob, state_blob)
        return record
    finally:
        _restore_rng(before, requested)


def run_segment(trainer: Trainer, output: str | Path, *, updates: int) -> dict:
    """New immutable output namespace. Every completed update is a resume boundary.

    Production orchestration must use RUN.py storage run. Commits are never rotated
    here; retention of superseded source-bound parents belongs to that maintainer.
    """
    _require(type(updates) is int and 1 <= updates <= trainer.config.max_updates-trainer.global_update,
             'segment exceeds remaining global update cap')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = committed_age = trainer.global_update
    committed_seen, attempted = trainer.samples_seen, 0
    exposures = []
    last = trainer.parent_directory
    _write_json(output / 'RUN.json', {'identity': trainer.identity, 'updates': updates,
                                    'parent_manifest_sha256': trainer.parent_sha256})
    try:
        if start == 0:
            last = output / 'step-00000000'
            trainer.save(last)
        for _ in range(updates):
            # A disk-space refusal cannot turn a previously valid checkpoint into
            # an incomplete parent. Workspace quota is also enforced by RUN.py.
            import shutil
            tensors = list(trainer.model.parameters())
            estimate = 4 * sum(p.numel() * p.element_size() for p in tensors) + 16_000_000
            _require(shutil.disk_usage(output).free > estimate, 'insufficient checkpoint disk headroom')
            attempted += 1
            try:
                metric = trainer.step()
            finally:
                if trainer.last_attempt is not None:
                    exposures.append(copy.deepcopy(trainer.last_attempt))
            candidate = output / f'step-{trainer.global_update:08d}'
            trainer.save(candidate, metrics=metric)
            last = candidate
            committed_age, committed_seen = trainer.global_update, trainer.samples_seen
        result = {'schema': SCHEMA, 'status': 'COMMITTED', 'global_update': trainer.global_update,
            'attempted_updates': attempted, 'committed_updates': committed_age-start, 'rolled_back_updates': 0,
            'last_checkpoint': str(last),
            'commit_sha256': trainer.parent_sha256, 'scope': 'BC_PLUMBING_NOT_PLAYING_STRENGTH'}
        _write_json(output / 'RESULT.json', result)
        return result
    except BaseException as exc:
        _write_json(output / 'FAILURE.json', {'schema': SCHEMA, 'status': 'FAILED',
            'error': f'{type(exc).__name__}: {exc}', 'last_checkpoint': None if last is None else str(last),
            'last_commit_sha256': trainer.parent_sha256,
            'attempted_global_update': start + attempted, 'committed_global_update': committed_age,
            'committed_samples_seen': committed_seen, 'attempted_updates': attempted,
            'committed_updates': committed_age-start, 'rolled_back_updates': attempted-(committed_age-start),
            'exposures': exposures, 'dataset_manifest_sha256': trainer.data.manifest_sha256,
            'attempted_sampler': {k: v for k, v in trainer.sampler.state_dict().items() if k != 'generator'},
            'rule': 'Only a COMMIT.json with verified payload is a parent; resume it into a new output root'})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest='command', required=True)
    describe = subs.add_parser('model-manifest', help='write the current fresh production identity; no model allocation')
    describe.add_argument('--output', type=Path, required=True)
    describe.add_argument('--readout', choices=('mean', 'pooled_control', 'candidate_attention'),
                          default='mean', help='fresh lineage only; train/resume reads the authenticated manifest')
    for command in ('validate', 'train', 'weights-transfer'):
        p = subs.add_parser(command)
        p.add_argument('--dataset', type=Path, required=True)
        p.add_argument('--dataset-sha256', required=True)
        p.add_argument('--model-manifest', type=Path, required=True)
        p.add_argument('--model-sha256', required=True)
        p.add_argument('--split', required=True)
        p.add_argument('--menu-backend', choices=('fast', 'reference'), default='fast',
                       help='exact prefix-menu implementation; reference restores the old path')
        if command in ('train', 'weights-transfer'):
            p.add_argument('--output', type=Path, required=True)
            p.add_argument('--max-updates', type=int, required=True)
            p.add_argument('--batch-size', type=int, default=4)
            p.add_argument('--microbatch-size', type=int, default=1)
            p.add_argument('--seed', type=int, required=True)
            p.add_argument('--learning-rate', type=float, default=1e-4)
            p.add_argument('--max-grad-norm', type=float, default=1.)
            p.add_argument('--optimizer', choices=('adam', 'adamw'), default='adam')
            p.add_argument('--schedule', choices=('constant', 'cosine'), default='constant')
            p.add_argument('--weight-decay', type=float, default=0.)
            p.add_argument('--first-prefix-weight', type=float, default=1.)
            p.add_argument('--min-learning-rate-ratio', type=float, default=.1)
            p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
        if command == 'train':
            p.add_argument('--updates', type=int, required=True)
            p.add_argument('--resume', type=Path)
            p.add_argument('--resume-sha256')
        elif command == 'weights-transfer':
            p.add_argument('--origin', type=Path, required=True)
            p.add_argument('--origin-commit-sha256', '--origin-sha256',
                           dest='origin_commit_sha256', required=True)
            p.add_argument('--origin-authentication', type=Path, required=True)
            p.add_argument('--origin-authentication-sha256', required=True)
            p.add_argument('--transfer-permit', type=Path, required=True)
            p.add_argument('--transfer-permit-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'model-manifest':
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(args.output, model_manifest(readout=args.readout))
            print(canonical({'model_manifest': str(args.output), 'sha256': file_sha(args.output)}))
            return 0
        identity = _pinned_json(args.model_manifest, args.model_sha256)
        validate_model_manifest(identity, production=True)
        if args.command in ('train', 'weights-transfer'):
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError(args.output)
            cfg = TrainConfig(args.max_updates, args.batch_size, args.microbatch_size, args.seed,
                              args.learning_rate, args.max_grad_norm, args.optimizer, args.schedule,
                              args.weight_decay, args.first_prefix_weight, args.min_learning_rate_ratio)
            if args.command == 'train':
                _require((args.resume is None) == (args.resume_sha256 is None),
                         'resume path and hash required together')
                _require(type(args.updates) is int and 1 <= args.updates <= cfg.max_updates,
                         'finite segment cap required')
            require_production_runtime()  # fail before dataset work or production allocation
        data = load_dataset(args.dataset, args.dataset_sha256, split=args.split, backend=args.menu_backend)
        if args.command == 'validate':
            print(canonical({'status': 'VALIDATED', 'dataset': data.binding, 'checks': data.validation,
                             'model_manifest_sha256': args.model_sha256, 'training_executed': False}))
            return 0
        if args.command == 'weights-transfer':
            trainer = Trainer.from_weights_transfer(
                data, identity, cfg, args.origin, args.origin_commit_sha256,
                args.origin_authentication, args.origin_authentication_sha256,
                args.transfer_permit, args.transfer_permit_sha256, device=args.device)
            commit = trainer.save(args.output)
            print(canonical({'status': 'TRANSFER_INITIALIZED', 'checkpoint': str(args.output),
                'commit_sha256': file_sha(args.output / 'COMMIT.json'),
                'global_update': commit['global_update'], 'samples_seen': commit['samples_seen'],
                'optimizer_updates': 0, 'games': 0,
                'scope': 'WEIGHTS_ONLY_NEW_LINEAGE_NOT_RESUME_NOT_STRENGTH',
                'initialization': commit['identity']['initialization']}))
            return 0
        trainer = (Trainer(data, identity, cfg, device=args.device) if args.resume is None else
                   Trainer.resume(data, identity, cfg, args.resume, args.resume_sha256, device=args.device))
        print(canonical(run_segment(trainer, args.output, updates=args.updates)))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(canonical({'status': 'REFUSED', 'error': f'{type(exc).__name__}: {exc}'}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
