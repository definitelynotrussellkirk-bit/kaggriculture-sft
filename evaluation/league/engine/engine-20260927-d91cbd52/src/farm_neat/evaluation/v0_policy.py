"""Own-prefix V0 complete-game evaluation; CPU stubs and future scorers share it.

PolicyAgent keeps V0Agent's callable/continuation interface and delegates all
execution to V0Agent's prefix-local epoch and FullSeasonPlan executor. It adds
isolated actor inputs, strict canonical-path admission and diagnostic records;
no sorting, repair, resampling or target injection is permitted at this boundary.

Run the bounded two-world smoke (no torch/checkpoint required):
  PYTHONPATH=src python3 -B -m farm_neat.evaluation.v0_policy smoke --output NEW_DIR

The edit arm and its separately seeded compiled-target control run in lockstep
on ONE physical world. A target outcome is certified only if all 719 actions
match. KEEP plays a second physical world. This is an equivalence certificate,
not three independent games. At dawn22 an independent controller is restored
from serialized own continuation and checked for the remaining 191 actions.
The retained HOLD witness is a separate source-bound regression gate.

For independent multi-arm/model evaluation use evaluate_panel(Case[], Arm[], ...).
Factories take ONLY (decode_mode, sampler_seed), not hidden world information.
ScorerPolicy calls the existing free-running decoder unchanged. Checkpoint
loading is available through load_committed_scorer; pinned runtime/SLURM authorization
remains the caller's responsibility.
The evaluator never trains, selects checkpoints, promotes, or retries failures.

Outputs preserve dictionary insertion order: executor inventory order is behavior.
market_tap is process-global; run this evaluator serially in a dedicated process.
Deadlines are cooperative, not a sandbox against untrusted/blocking Python.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import dataclass
import hashlib
import inspect
from itertools import islice
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.epochs import DAWNS, KeepPolicy, TrajectoryPolicy, V0Agent
from farm_neat.economy.v0.plan import Plan, edit_trajectory
from farm_neat.economy.v0.projection import Masks, State
from farm_neat.economy.orders import FullSeasonPlan
from farm_neat.evaluation import realization, scenarios
from farm_neat.evaluation.v0_continuation import V0Continuation
from farm_neat.evaluation.common import file_sha

SCHEMA = 'farm-v0-policy-evaluation-v1'
SNAPSHOT_SCHEMA = 'farm-v0-policy-continuation-v2'
SCOPE = 'REGISTERED_EXPOSED_FIXED_WORLDS; NOT_HELDOUT_OR_NEURAL_IMPROVEMENT; NO_PROMOTION'
MODES = ('greedy_class', 'sample_raw')
AGENT_FIELDS = {'policy', 'labor_factor', 'hands_mode', 'belief', 'plan', 'allocation',
                'started', 'hold_releases', 'land_commitment', 'controller', 'game_id', 'records'}


def wire(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def wire_hash(value: Any) -> str:
    """Order-sensitive identity, including executor/engine inventory mappings."""
    return hashlib.sha256(wire(value).encode()).hexdigest()


def bind(value: dict) -> dict:
    body = {k: v for k, v in value.items() if k != 'content_sha256'}
    return {**body, 'content_sha256': wire_hash(body)}


def write_new(path: str | Path, value: dict) -> dict:
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlinked output')
    path.parent.mkdir(parents=True, exist_ok=True)
    result = bind(value)
    with path.open('x') as stream:
        stream.write(wire(result) + '\n'); stream.flush(); os.fsync(stream.fileno())
    return result


def source_pins() -> dict:
    paths = [*sorted((ROOT / 'src/farm_neat/economy').rglob('*.py')),
             ROOT / 'src/farm_neat/economy/v0/modules.json', Path(__file__).resolve()]
    paths += [ROOT / name for name in (
        'src/farm_neat/game/constants.py', 'src/farm_neat/game/exact_engine.py',
        'src/farm_neat/game/kaggriculture_engine_reference.py', 'src/farm_neat/game/kaggriculture.json',
        'src/farm_neat/evaluation/scenarios.py', 'src/farm_neat/evaluation/realization.py',
        'src/farm_neat/evaluation/common.py', 'src/farm_neat/evaluation/v0_continuation.py',
        'src/farm_neat/policy/v0_scorer.py',
        'project/configs/scenarios.json')]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(set(paths))}


def _options(value: dict | None = None) -> dict:
    result = {'labor_factor': 4., 'belief': None, 'hold_releases': {}}
    if value is not None:
        if set(value) - set(result):
            raise ValueError('unknown projection options')
        result.update(copy.deepcopy(value))
    factor = result['labor_factor']
    if type(factor) not in (int, float) or not math.isfinite(factor) or factor <= 0:
        raise ValueError('finite positive labor_factor required')
    belief = result['belief']
    if belief is not None and (not isinstance(belief, dict) or set(belief) - set(V.PRODUCTS)
                              or any(type(v) not in (int, float) or v != 0 for v in belief.values())):
        raise ValueError('bootstrap requires explicit zero-flow belief')
    return result


@dataclass
class ValidatedEdits:
    plan: Plan
    actions: tuple[int, ...]
    trace: list[dict]


def validate_edits(state: State, plan: Plan, edits, *, projection_kwargs=None) -> ValidatedEdits:
    """Validate chosen IDs only, rebuilding economic support at every own prefix.

    This is not a menu replacement: ScorerPolicy still uses build_menu for all
    legal IDs/phi/classes. A stub's selected transition has the exact same Plan,
    key cursor and STOP/cap continuation; no representative-ID likelihood claim.
    """
    actions = list(islice(iter(edits), V.MAX_EDITS_PER_EPOCH + 2))
    if any(type(a) is not int or not 0 <= a < V.N_ACTIONS for a in actions):
        raise ValueError('frozen integer action IDs required')
    if len(actions) > V.MAX_EDITS_PER_EPOCH + 1:
        raise ValueError('edit cap exceeded')
    if not actions or actions[-1] != V.STOP_ID or V.STOP_ID in actions[:-1]:
        raise ValueError('exactly one final STOP required; no ignored suffix')
    keys = [V.plan_key(a) for a in actions[:-1]]
    if keys != sorted(set(keys)):
        raise ValueError('strictly increasing unique plan key order required')
    if state.hour != 0 or not 0 <= state.step <= V.LAST_STEP:
        raise ValueError('dawn-only V0 edit publication')
    options = _options(projection_kwargs)
    prefix, last, count, trace = plan.copy(), -1, 0, []
    for action in actions:
        before = prefix.canonical()
        if action == V.STOP_ID:
            successor = prefix.copy()
        else:
            masks = Masks(state, prefix, **options)
            if not masks.legal(action):
                raise ValueError(f'prefix {count}: masked action {V.describe(action)}')
            successor = prefix.apply(action, masks.resolver)
        after_count = count + int(action != V.STOP_ID)
        after_key = last if action == V.STOP_ID else V.plan_key(action)
        signature = {'plan': successor.canonical(), 'last_key': after_key,
                     'edit_count': after_count, 'terminated': action == V.STOP_ID}
        trace.append({'raw_id': action, 'before_plan': before, 'last_key': last,
                      'edits_used': count, 'successor': signature,
                      'successor_sha256': wire_hash(signature),
                      'forced_stop': action == V.STOP_ID and count == V.MAX_EDITS_PER_EPOCH})
        prefix, last, count = successor, after_key, after_count
    return ValidatedEdits(prefix, tuple(actions), trace)


def target_edits(state: State, plan: Plan, target: Plan, *, projection_kwargs=None) -> ValidatedEdits:
    """A strict reference/stub constructor, NEVER called for a student target."""
    options = _options(projection_kwargs)
    masks = Masks(state, plan, **options)
    requested, _, _ = edit_trajectory(target, plan, masks.resolver)
    try:
        result = validate_edits(state, plan, requested, projection_kwargs=options)
    except ValueError as exc:
        raise ValueError(f'target not independently reachable: {exc}') from exc
    if result.plan != target:
        raise ValueError('target not independently reachable: exact Plan residual remains')
    return result


class ExactTargetPolicy:
    """Fixed target control; reachability is proved anew against its own current Plan."""
    def __init__(self, target: Plan):
        self.target = target.copy()

    def __call__(self, state, plan, masks):
        return target_edits(state, plan, self.target, projection_kwargs=masks.kw).actions


class ScorerPolicy:
    """Existing V0 scorer/decoder -> unchanged (state, plan, masks) callable.

    The caller supplies a verified checkpoint identity (or explicitly diagnostic
    scorer identity), loads the model, and owns its device/runtime authorization.
    No target Plan or teacher prefix is accepted by this adapter.
    """
    def __init__(self, scorer, *, identity: dict, mode='greedy_class', seed=0, temperature=1.):
        import torch
        if mode not in MODES or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('explicit decoding mode and sampler seed required')
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('positive finite temperature required')
        if not isinstance(identity, dict) or not identity:
            raise ValueError('verified model/source identity required')
        wire(identity)
        self.scorer = scorer.eval()
        self.identity = copy.deepcopy(identity)
        self.mode, self.seed, self.temperature = mode, seed, float(temperature)
        parameters = list(scorer.parameters())
        device = parameters[0].device if parameters else torch.device('cpu')
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.last_trace = None

    def __call__(self, state, plan, masks):
        import torch
        from farm_neat.policy.v0_scorer import decode_epoch
        if self.scorer.training:
            raise ValueError('evaluation scorer must remain in eval mode')
        with torch.inference_mode():
            result = decode_epoch(self.scorer, state, plan, generator=self.generator,
                                  mode=self.mode, temperature=self.temperature,
                                  projection_kwargs=copy.deepcopy(masks.kw))
        self.last_trace = copy.deepcopy(result.trace)
        return result.actions

    def snapshot_state(self):
        return {'identity': self.identity, 'mode': self.mode, 'seed': self.seed,
                'temperature': self.temperature, 'device': str(self.generator.device),
                'generator_state': self.generator.get_state().tolist()}

    def restore_state(self, payload):
        import torch
        own = self.snapshot_state()
        for key in ('identity', 'mode', 'seed', 'temperature', 'device'):
            if payload.get(key) != own[key]:
                raise ValueError('scorer continuation identity mismatch: ' + key)
        if set(payload) != set(own):
            raise ValueError('scorer continuation fields')
        self.generator.set_state(torch.tensor(payload['generator_state'], dtype=torch.uint8, device='cpu'))
        self.last_trace = None


class CheckpointCapabilityError(ValueError):
    """Missing/mismatched native assets are refusals, never synthetic fallbacks."""


def inference_runtime(device='cpu'):
    """Set the existing FP32 numerical contract, without seeding or training."""
    import torch
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise CheckpointCapabilityError('CPU or CUDA inference device required')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise CheckpointCapabilityError('CUDA runtime unavailable')
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return device


def inspect_committed_scorer(directory, expected_commit_sha256, *, device='cpu', production=True):
    """Authenticate inference assets without constructing a Trainer or loading data.

    Source and runtime joins are EXACT, including the original training device.
    A CPU remap of a CUDA commit is not authorized by this interface. Regret uses
    RL's COMMIT and state envelope, but retains its own identity and experiment.
    Optimizer/corpus provenance is authenticated as bytes, not resumed/validated
    as a training continuation. The full training identity stays evaluator-only.
    """
    from farm_neat.training import v0_bc as BC, v0_rl as RL
    if type(production) is not bool:
        raise CheckpointCapabilityError('explicit production/diagnostic scope required')
    directory = Path(directory)
    if any(p.is_symlink() for p in (directory, *directory.parents, directory/'COMMIT.json', directory/'state.pt')):
        raise CheckpointCapabilityError('symlinked checkpoint refused')
    device = inference_runtime(device)
    commit = BC._pinned_json(directory/'COMMIT.json', expected_commit_sha256)
    identity = commit.get('identity', {})
    schema = identity.get('schema') if isinstance(identity, dict) else None
    if commit.get('status') != 'COMMITTED':
        raise CheckpointCapabilityError('COMMITTED checkpoint required')
    if commit.get('schema') == BC.COMMIT_SCHEMA and schema == BC.SCHEMA:
        kind, state_schema, model = 'BC', BC.SCHEMA, identity.get('model')
        RL.inspect_bc_parent(directory, expected_commit_sha256)
        if identity.get('objective') != BC.OBJECTIVE:
            raise CheckpointCapabilityError('BC objective identity mismatch')
    elif commit.get('schema') == RL.COMMIT_SCHEMA:
        from farm_neat.training import v0_regret_train as RT
        if schema not in (RL.SCHEMA, RT.SCHEMA):
            raise CheckpointCapabilityError('native RL/regret identity required; no relabeling')
        kind, state_schema, model = ('RL' if schema == RL.SCHEMA else 'REGRET'), RL.SCHEMA, identity.get('model_identity')
        if identity.get('source_pins') != RL.source_pins() or identity.get('protocol') != RL.PROTOCOL:
            raise CheckpointCapabilityError('RL source/protocol identity mismatch')
        if kind == 'RL':
            if identity.get('arm', {}).get('id') not in (RL.ARM_PG, RL.ARM_ANCHOR) or 'adapter' in identity:
                raise CheckpointCapabilityError('RL arm identity mismatch')
        else:
            adapter = identity.get('adapter', {})
            experiment = adapter.get('experiment', {})
            # Authenticate the embedded fit identity, NOT its training data.
            # RT._validate_plan additionally loads the original direct plan;
            # requiring it here would turn inference into a training dependency.
            if (experiment.get('schema') != RT.PLAN_SCHEMA
                    or experiment.get('source_pins') != RT.source_pins()
                    or experiment.get('reductions') != RT.REDUCTIONS
                    or experiment.get('rl_protocol') != RL.PROTOCOL
                    or experiment.get('mode') not in RT.MODES
                    or experiment.get('strength_claim') is not False):
                raise CheckpointCapabilityError('regret experiment source/protocol identity mismatch')
            if (identity.get('arm') != {'id': experiment['mode'], 'bc_anchor': None}
                    or adapter.get('reference_parent') != experiment['reference_parent']
                    or adapter.get('reference_state_sha256') != identity.get('behavior_parent', {}).get('state_sha256')
                    or experiment['reference_parent']['commit_sha256'] != identity.get('behavior_parent', {}).get('commit_sha256')):
                raise CheckpointCapabilityError('regret adapter/reference identity mismatch')
    else:
        raise CheckpointCapabilityError('native BC/RL/regret COMMIT schema required')
    BC.validate_model_manifest(model, production=production)
    if not model['diagnostic']:
        BC.require_production_runtime()
    elif device.type != 'cpu':
        raise CheckpointCapabilityError('diagnostic checkpoints are CPU-only')
    if identity.get('runtime') != BC.runtime_identity(device):
        raise CheckpointCapabilityError('checkpoint runtime identity mismatch; no cross-runtime reinterpretation')
    path = directory/'state.pt'
    if (not path.is_file() or path.stat().st_size != commit.get('state_bytes')
            or file_sha(path) != commit.get('state_sha256')):
        raise CheckpointCapabilityError('checkpoint state hash/size mismatch')
    age = commit.get('global_update')
    if type(age) is not int or age < 0:
        raise CheckpointCapabilityError('checkpoint update identity required')
    expected_scope = ('DIAGNOSTIC_CPU_NOT_STRENGTH' if model['diagnostic'] else
                      'V0_BC_NOT_STRENGTH' if kind == 'BC' else 'V0_TERMINAL_WDL_PILOT_NOT_STRENGTH')
    if commit.get('scope') != expected_scope:
        raise CheckpointCapabilityError('checkpoint diagnostic/production scope mismatch')
    # Minimal actor identity: no teacher prefixes, corpus rows, world seeds, or
    # trainer-private metadata is passed into ScorerPolicy's callable interface.
    actor_identity = {'kind': kind, 'commit_sha256': expected_commit_sha256,
                      'state_sha256': commit['state_sha256'], 'model_identity': copy.deepcopy(model),
                      'training_identity_sha256': BC._digest(identity),
                      'runtime_sha256': BC._digest(identity['runtime']),
                      'diagnostic': model['diagnostic']}
    return {'identity': actor_identity, 'training_identity': copy.deepcopy(identity),
            'commit': commit, 'state_schema': state_schema, 'device': str(device)}


@dataclass(frozen=True)
class CommittedScorer:
    scorer: Any
    identity: dict

    def policy(self, mode, seed, *, temperature=1., opening_only=False):
        """A new sampler per call; only immutable inference tensors are shared."""
        cls = OpeningScorerPolicy if opening_only else ScorerPolicy
        return cls(self.scorer, identity=self.identity, mode=mode, seed=seed, temperature=temperature)


def load_committed_scorer(directory, expected_commit_sha256, *, device='cpu', production=True):
    """Inference-only cold load. No Trainer, optimizer construction, or corpus IO."""
    import torch
    from farm_neat.policy.v0_scorer import V0Scorer
    from farm_neat.policy.qwen_hybrid import Config
    from farm_neat.training import v0_bc as BC
    info = inspect_committed_scorer(directory, expected_commit_sha256, device=device, production=production)
    path = Path(directory)/'state.pt'
    # Hash and deserialize the same open file, then check it again; do not use an
    # unchecked second path lookup after authentication.
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        if sha != info['commit']['state_sha256']:
            raise CheckpointCapabilityError('checkpoint state changed before load')
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=True)
        stream.seek(0)
        if hashlib.file_digest(stream, 'sha256').hexdigest() != sha:
            raise CheckpointCapabilityError('checkpoint state changed during load')
    if (not isinstance(state, dict) or state.get('schema') != info['state_schema']
            or state.get('identity') != info['training_identity']
            or state.get('global_update') != info['commit']['global_update']
            or not isinstance(state.get('model'), dict)):
        raise CheckpointCapabilityError('checkpoint state/COMMIT identity mismatch')
    if info['identity']['kind'] == 'BC' and state.get('samples_seen') != info['commit'].get('samples_seen'):
        raise CheckpointCapabilityError('BC sample count mismatch')
    model_identity = info['identity']['model_identity']
    # Initialization must not consume the application's global sampler stream.
    with torch.random.fork_rng(devices=[]):
        # inspect_committed_scorer authenticated the entire versioned readout identity.
        # Missing key means historical mean, never an inferred mode from tensor shapes.
        model = V0Scorer(Config(**model_identity['config']), diagnostic=model_identity['diagnostic'],
                         readout=model_identity.get('readout', {}).get('mode', 'mean'))
    if model.get_extra_state() != model_identity or not BC._finite_tree(state['model']):
        raise CheckpointCapabilityError('model identity/nonfinite state mismatch')
    if state.get('parameter_order') != [name for name, _ in model.named_parameters()]:
        raise CheckpointCapabilityError('model parameter order mismatch')
    expected = model.state_dict()
    if set(expected) != set(state['model']):
        raise CheckpointCapabilityError('model state inventory mismatch')
    for name, tensor in expected.items():
        actual = state['model'][name]
        if isinstance(tensor, torch.Tensor) and (not isinstance(actual, torch.Tensor)
                or actual.shape != tensor.shape or actual.dtype != tensor.dtype):
            raise CheckpointCapabilityError('model tensor shape/dtype mismatch: ' + name)
    model.load_state_dict(state['model'], strict=True)
    model.to(info['device']).eval().requires_grad_(False)
    return CommittedScorer(model, copy.deepcopy(info['identity']))


class OpeningScorerPolicy(ScorerPolicy):
    """Registered comparison scope: own-prefix opening decode, later KEEP only."""
    def __call__(self, state, plan, masks):
        if state.step != 0:
            self.last_trace = None
            return (V.STOP_ID,)
        return super().__call__(state, plan, masks)

    def snapshot_state(self):
        return {**super().snapshot_state(), 'publication': 'opening_only_later_keep'}

    def restore_state(self, payload):
        if payload.get('publication') != 'opening_only_later_keep':
            raise ValueError('scorer continuation publication mismatch')
        super().restore_state(payload)


class _PublicationCapture:
    """Epoch-local diagnostics only; the real controller owns propose/publish.

    Capture the compiler's program and actual refusals without recompiling or
    persisting a wrapper in agent continuation. publish returns the real controller.
    """
    def __init__(self, controller):
        self.controller = controller

    def propose(self, observation, program):
        proposal = self.controller.propose(observation, program)
        self.program = copy.deepcopy(program)
        self.refusals = copy.deepcopy(proposal['rejected'])
        return proposal

    def publish(self, observation, proposal):
        return self.controller.publish(observation, proposal)


class PolicyAgent(V0Agent):
    """Strict evaluator admission around the shared prefix-local V0Agent epoch.

    The base agent owns plan/HOLD commit, allocation, compiler and publication.
    Validate before delegation: its normalizer intentionally accepts emissions
    this evaluator must reject. Sample once, then replay only those validated IDs.
    """
    def epoch(self, observation):
        state = self.actor_state(copy.deepcopy(observation))
        actor_plan = self.plan.copy()
        options = _options(self.projection_kw())
        actor_masks = Masks(state, actor_plan, **copy.deepcopy(options))
        before_input = wire_hash([vars(state), actor_plan.canonical(), actor_masks.kw])
        requested = list(islice(iter(self.policy(state, actor_plan, actor_masks)), V.MAX_EDITS_PER_EPOCH + 2))
        if wire_hash([vars(state), actor_plan.canonical(), actor_masks.kw]) != before_input:
            raise ValueError('policy mutated actor input instead of emitting edits')
        decision = validate_edits(self.actor_state(observation), self.plan, requested, projection_kwargs=options)
        before = self.plan.canonical()
        policy, controller = self.policy, self.controller
        capture = _PublicationCapture(controller)
        self.policy = lambda *_: decision.actions
        self.controller = capture
        try:
            super().epoch(observation)
        finally:
            self.policy = policy
            # On refusal there is no published replacement. Never leak the
            # diagnostics proxy into continuation, on success OR failure.
            if self.controller is capture:
                self.controller = controller
        record = self.records.pop()
        if self.plan != decision.plan:
            raise ValueError('validator/agent successor Plan disagreement')
        record.update({'before_plan': before, 'after_plan': self.plan.canonical(),
                       'raw_edits': list(decision.actions), 'prefix_trace': decision.trace,
                       'compiled_program': capture.program, 'refusals': capture.refusals})
        if isinstance(self.policy, ScorerPolicy) and self.policy.last_trace is not None:
            if self.policy.last_trace['final_plan'] != self.plan.canonical():
                raise ValueError('decoder/agent successor Plan disagreement')
            record['decode_trace'] = copy.deepcopy(self.policy.last_trace)
        self.records.append(record)


def _policy_snapshot(policy) -> dict:
    if type(policy) is KeepPolicy:
        return {'kind': 'KEEP'}
    if type(policy) is TrajectoryPolicy:
        return {'kind': 'TRAJECTORY', 'schedule': [[k, v] for k, v in policy.schedule.items()]}
    if type(policy) is ExactTargetPolicy:
        return {'kind': 'TARGET', 'target': policy.target.canonical()}
    if isinstance(policy, ScorerPolicy):
        return {'kind': 'SCORER', 'state': policy.snapshot_state()}
    if (isinstance(getattr(policy, 'identity', None), dict) and policy.identity
            and callable(getattr(policy, 'snapshot_state', None))
            and callable(getattr(policy, 'restore_state', None))):
        return {'kind': 'EXTERNAL', 'identity': copy.deepcopy(policy.identity),
                'state': copy.deepcopy(policy.snapshot_state())}
    raise ValueError('policy continuation unsupported; provide explicit identity and snapshot/restore_state')


def _restore_policy(payload: dict, supplied=None):
    kind = payload.get('kind')
    if kind == 'KEEP' and set(payload) == {'kind'}:
        return KeepPolicy()
    if kind == 'TRAJECTORY' and set(payload) == {'kind', 'schedule'}:
        rows = payload['schedule']
        if (not isinstance(rows, list) or any(not isinstance(r, list) or len(r) != 2
                or type(r[0]) is not int or r[0] not in DAWNS or not isinstance(r[1], list) for r in rows)
                or len({r[0] for r in rows}) != len(rows)):
            raise ValueError('trajectory policy rows')
        return TrajectoryPolicy(dict(rows))
    if kind == 'TARGET' and set(payload) == {'kind', 'target'}:
        return ExactTargetPolicy(Plan.from_canonical(payload['target']))
    if kind in ('SCORER', 'EXTERNAL') and supplied is not None:
        current = _policy_snapshot(supplied)
        if current['kind'] != kind or current.get('identity') != payload.get('identity'):
            raise ValueError('supplied policy identity mismatch')
        if set(current) != set(payload):
            raise ValueError('supplied policy fields')
        supplied.restore_state(copy.deepcopy(payload['state']))
        if _policy_snapshot(supplied) != payload:
            raise ValueError('policy state did not restore exactly')
        return supplied
    raise ValueError('policy continuation requires matching explicitly supplied model/actor')


def _memory(agent: V0Agent) -> dict:
    # Preserve row order in the explicitly versioned flat wire encoding. The shared codec
    # owns both the values AND row order; never take a second live-dict view.
    # Allocation None remains unsupported by this evaluator envelope.
    memory = V0Continuation.capture(agent).to_dict()
    return {'agent_type': type(agent).__name__, 'game_id': memory['game_id'], 'plan': memory['plan'],
            'continuation_schema': memory['schema'], 'land_commitment': memory['land_commitment'],
            'allocation': [[*key, *tile] for key, tile in memory['allocation']],
            'started': [[*tile, *key] for tile, key in memory['started']],
            'hold_releases': [[p, *overlay, release] for p, overlay, release in memory['hold_releases']],
            'controller': memory['controller'], 'labor_factor': memory['labor_factor'],
            'hands_mode': memory['hands_mode'], 'belief': memory['belief']}


def agent_signature(agent: V0Agent) -> str:
    """Ordered memory plus policy identity, not map/Plan/behavior equivalence.

    V2 includes explicit LAND memory and the shared codec schema. Historical
    post_agent_sha256 measurements are not relabelled or re-derived. Source-pinned
    envelopes require reviewed integration; their integrity guards are not bypassed.
    """
    return wire_hash({**_memory(agent), 'policy': _policy_snapshot(agent.policy)})


def _validate_memory(agent):
    _options(agent.projection_kw())
    if agent.hands_mode not in ('daily', 'constant'):
        raise ValueError('explicit hands_mode required')
    holds = {p for p in range(len(V.PRODUCTS)) if V.PRIMITIVES[agent.plan.overlay(p)[0]] == 'HOLD'}
    if set(agent.hold_releases) != holds:
        raise ValueError('every retained HOLD requires its original binding; no silent rebind')
    for p, (overlay, release) in agent.hold_releases.items():
        if tuple(overlay) != agent.plan.overlay(p)[:2] or type(release) is not int or not 0 <= release <= V.LAST_STEP:
            raise ValueError('invalid HOLD binding')
    for key, tile in agent.allocation.items():
        if key not in agent.plan.instances() or any(type(v) is not int or not 0 <= v < 10 for v in tile):
            raise ValueError('invalid allocation memory')
    for tile, key in agent.started.items():
        if (any(type(v) is not int or not 0 <= v < 10 for v in tile)
                or not 0 <= key[0] < V.N_MODULES or not 0 <= key[1] < 100):
            raise ValueError('invalid started-instance memory')


def _boundary(agent, observation):
    step = observation.get('step')
    controller = agent.controller
    if (type(step) is not int or step not in DAWNS[1:] or controller is None
            or controller.executor.stage_step != step - 1
            or controller.executor.player != observation.get('player')
            or controller.committed_step != step - 24 or tuple(controller.publication_steps) != DAWNS
            or controller.game_id != agent.game_id):
        raise ValueError('pre-dawn boundary BEFORE observe/publication required')
    return step


def snapshot_agent(agent: V0Agent, observation: dict) -> dict:
    if type(agent) not in (V0Agent, PolicyAgent) or set(vars(agent)) != AGENT_FIELDS:
        raise ValueError('unrecognized agent fields; review continuation sufficiency')
    step = _boundary(agent, observation)
    # Reconcile only own observed acquisition, before the executor observes. Never
    # recreate a missing authorization. Both live and restarted agents retain it.
    agent.actor_state(observation, initialize=False)
    _validate_memory(agent)
    return bind({'schema': SNAPSHOT_SCHEMA, 'phase': 'BEFORE_OBSERVE_AND_PUBLICATION',
                 'step': step, 'observation_sha256': wire_hash(observation), 'source_pins': source_pins(),
                 'policy': _policy_snapshot(agent.policy), **_memory(agent)})


def restore_agent(payload: dict, observation: dict, *, policy=None) -> V0Agent:
    fields = {'schema', 'phase', 'step', 'observation_sha256', 'source_pins', 'policy',
              'agent_type', 'game_id', 'plan', 'allocation', 'started', 'hold_releases', 'controller',
              'labor_factor', 'hands_mode', 'belief', 'continuation_schema', 'land_commitment', 'content_sha256'}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError('closed continuation fields required')
    if payload != bind(payload):
        raise ValueError('continuation hash mismatch')
    if payload['schema'] != SNAPSHOT_SCHEMA or payload['phase'] != 'BEFORE_OBSERVE_AND_PUBLICATION':
        raise ValueError('continuation schema or phase')
    if payload['source_pins'] != source_pins():
        raise ValueError('continuation source pins changed')
    if payload['observation_sha256'] != wire_hash(observation):
        raise ValueError('continuation observation mismatch')
    types = {'V0Agent': V0Agent, 'PolicyAgent': PolicyAgent}
    if payload['agent_type'] not in types:
        raise ValueError('unknown agent type')
    restored_policy = _restore_policy(payload['policy'], policy)

    def flat_rows(value, name):
        if (not isinstance(value, list) or any(not isinstance(row, list) or len(row) != 4
                or any(type(v) is not int for v in row) for row in value)):
            raise ValueError('invalid ' + name + ' rows')
        keys = [tuple(row[:2]) for row in value]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate ' + name + ' rows')

    flat_rows(payload['allocation'], 'allocation')
    flat_rows(payload['started'], 'started')
    flat_rows(payload['hold_releases'], 'HOLD')
    if len({row[0] for row in payload['hold_releases']}) != len(payload['hold_releases']):
        raise ValueError('duplicate HOLD product')
    memory = {
        'schema': payload['continuation_schema'],
        'land_commitment': copy.deepcopy(payload['land_commitment']),
        'plan': copy.deepcopy(payload['plan']),
        'allocation': [[row[:2], row[2:]] for row in payload['allocation']],
        'started': [[row[:2], row[2:]] for row in payload['started']],
        'hold_releases': [[row[0], row[1:3], row[3]] for row in payload['hold_releases']],
        'controller': copy.deepcopy(payload['controller']), 'game_id': payload['game_id'],
        'labor_factor': payload['labor_factor'], 'hands_mode': payload['hands_mode'],
        'belief': copy.deepcopy(payload['belief']),
    }
    agent = V0Continuation.from_dict(memory).restore(
        restored_policy, agent_cls=types[payload['agent_type']], observation=observation)
    _validate_memory(agent)
    if _boundary(agent, observation) != payload['step']:
        raise ValueError('continuation step mismatch')
    return agent


@dataclass(frozen=True)
class Case:
    scenario: str
    seat: int
    opponent: str | None = None
    game_id: str | None = None

    def document(self):
        if type(self.seat) is not int or self.seat not in (0, 1):
            raise ValueError('seat 0 or 1 required')
        spec = scenarios.catalog()[self.scenario]
        if self.opponent is not None:
            spec['opponent'] = self.opponent
        spec = scenarios.validate(spec)
        _, identity = scenarios.opponent(spec['opponent'])
        body = {'scenario': spec, 'seat': self.seat, 'opponent_identity': identity}
        key = wire_hash(body)
        return {'id': key, **body, 'game_id': self.game_id or 'v0-eval-' + key[:16]}


ORIGINAL_CASE_SCHEMA = 'farm-v0-original-rule-case-v1'


@dataclass(frozen=True)
class OriginalRuleCase:
    """Evaluator-only initialization; no catalog shops or weed substitutions."""
    seed: int
    seat: int
    opponent: str = 'router'
    configuration: dict | None = None
    game_id: str | None = None

    def document(self):
        from farm_neat.game.exact_engine import DEFAULT_CONFIGURATION, assert_bundled_engine_identity
        if type(self.seed) is not int or not 0 <= self.seed < 2**63 or type(self.seat) is not int or self.seat not in (0, 1):
            raise ValueError('explicit seed and seat required')
        config = copy.deepcopy(DEFAULT_CONFIGURATION if self.configuration is None else self.configuration)
        if config != DEFAULT_CONFIGURATION:
            raise ValueError('exact original-rule configuration required; no rule overrides')
        if self.opponent not in ('pass', 'router'):
            raise ValueError('original-rule bridge supports frozen PASS or Router')
        _, identity = scenarios.opponent(self.opponent)
        spec = {'schema': ORIGINAL_CASE_SCHEMA, 'seed': self.seed, 'configuration': config,
                'opponent': self.opponent, 'engine': assert_bundled_engine_identity()}
        body = {'scenario': spec, 'seat': self.seat, 'opponent_identity': identity}
        key = wire_hash(body)
        return {'id': key, **body, 'game_id': self.game_id or 'v0-original-' + key[:16]}


def environment_for_case(case):
    """Explicit dispatch inside the existing owner, without process monkeypatches."""
    if case['scenario'].get('schema') != ORIGINAL_CASE_SCHEMA:
        return scenarios.ScenarioEnvironment(case['scenario'])
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    spec = case['scenario']
    expected = OriginalRuleCase(spec['seed'], case['seat'], spec['opponent'],
                                spec['configuration'], case['game_id']).document()
    if case != expected:
        raise ValueError('original-rule Case binding changed')
    env = ExactKaggricultureEnvironment(seed=spec['seed'], configuration=copy.deepcopy(spec['configuration']))
    env.scenario_identity = bind({'schema': ORIGINAL_CASE_SCHEMA+'-environment',
                                  'initialization': copy.deepcopy(spec), 'case_id': case['id'],
                                  'rules': 'EXACT_ORIGINAL; NO_SHOP_OR_WEED_SUBSTITUTIONS'})
    return env


@dataclass(frozen=True)
class Arm:
    name: str
    factory: Callable[[str, int], Any]
    identity: dict
    kind: str = 'policy'
    initial_plan: Plan | None = None
    shadow: 'Arm | None' = None

    def document(self):
        if not self.name or not self.identity or self.kind not in ('policy', 'keep', 'compiled_target'):
            raise ValueError('named, identified policy/control arm required')
        if self.initial_plan is not None and self.kind != 'compiled_target':
            raise ValueError('only compiled-target control may start from a teacher Plan')
        source = inspect.getsourcefile(self.factory)
        return {'name': self.name, 'identity': copy.deepcopy(self.identity), 'kind': self.kind,
                'initial_plan': None if self.initial_plan is None else self.initial_plan.canonical(),
                'factory_source': None if source is None else {'path': _relative(source), 'sha256': file_sha(source)},
                'shadow': None if self.shadow is None else self.shadow.document()}


def _relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def registration(cases, arms, *, modes=('greedy_class',), sample_seeds=(0,), max_games=2,
                 labor_factor=4., hands_mode='daily', timeout_seconds=180., restart_steps=()) -> dict:
    cases, arms = tuple(cases), tuple(arms)
    if (not cases or not arms or not modes or len(set(modes)) != len(modes)
            or any(m not in MODES for m in modes)):
        raise ValueError('nonempty cases/arms and distinct supported decode strata required')
    if (not sample_seeds or any(type(s) is not int or not 0 <= s < 2**63 for s in sample_seeds)
            or len(set(sample_seeds)) != len(sample_seeds)):
        raise ValueError('distinct explicit sampler seeds required')
    _options({'labor_factor': labor_factor})
    if hands_mode not in ('daily', 'constant') or not 0 < timeout_seconds <= 1800:
        raise ValueError('bounded compiler/game settings required')
    if any(type(s) is not int or s not in DAWNS[1:] for s in restart_steps):
        raise ValueError('restart only at a real pre-dawn boundary')
    docs, controls = [c.document() for c in cases], [a.document() for a in arms]
    if len({c['id'] for c in docs}) != len(docs) or len({a.name for a in arms}) != len(arms):
        raise ValueError('duplicate physical case or arm name')
    jobs = []
    for case in docs:
        for mode in modes:
            for seed in ((0,) if mode == 'greedy_class' else sample_seeds):
                for arm in arms:
                    jobs.append({'index': len(jobs), 'case_id': case['id'], 'arm': arm.name,
                                 'mode': mode, 'sampler_seed': seed})
    if type(max_games) is not int or len(jobs) > max_games:
        raise ValueError(f'physical-game budget exceeded: {len(jobs)} requested, {max_games} allowed')
    return bind({'schema': SCHEMA + '-registration', 'scope': SCOPE, 'source_pins': source_pins(),
                 'opponent_sources': scenarios.sources(), 'cases': docs, 'arms': controls, 'jobs': jobs,
                 'compiler': {'labor_factor': labor_factor, 'hands_mode': hands_mode, 'belief': None},
                 'restart_steps': list(restart_steps), 'timeout_seconds': timeout_seconds,
                 'max_games': max_games, 'world_information_visibility': 'EVALUATOR_ONLY',
                 'policy_inputs': 'OWN_VISIBLE_State, OWN_CURRENT_Plan, OWN_PREFIX_Masks',
                 'shadow_scope': 'LOCKSTEP_ACTION_EQUIVALENCE; ZERO_EXTRA_PHYSICAL_GAMES',
                 'fresh_learning_games': 0, 'optimizer_updates': 0})


def compare_games(control: dict, child: dict) -> dict:
    result = {'exact_plan_equality': None, 'compiled_action_equality': None, 'joint_action_equality': None,
              'realization_equality': None, 'own_cash_delta': None, 'rival_cash_delta': None, 'margin_delta': None,
              'control_wdl': control.get('wdl'), 'child_wdl': child.get('wdl')}
    if control['status'] != 'COMPLETE' or child['status'] != 'COMPLETE':
        return result
    result.update(exact_plan_equality=[(r['step'], r['after_plan']) for r in control['epochs']]
                   == [(r['step'], r['after_plan']) for r in child['epochs']],
                  compiled_action_equality=wire(control['own_actions']) == wire(child['own_actions']),
                  joint_action_equality=wire(control['joint_actions']) == wire(child['joint_actions']),
                  realization_equality=control['realization'] == child['realization'],
                  own_cash_delta=child['own_final_cash'] - control['own_final_cash'],
                  rival_cash_delta=child['rival_final_cash'] - control['rival_final_cash'],
                  margin_delta=child['margin'] - control['margin'])
    return result


def _make_agent(arm: Arm, job, case, compiler, used):
    policy = arm.factory(job['mode'], job['sampler_seed'])
    if any(policy is prior for prior in used):
        raise ValueError('factory reused mutable policy across physical jobs')
    if isinstance(policy, ScorerPolicy) and (policy.mode != job['mode'] or policy.seed != job['sampler_seed']):
        raise ValueError('factory scorer disagrees with registered decoding stratum/seed')
    used.append(policy)
    agent = PolicyAgent(policy, plan=arm.initial_plan, game_id=case['game_id'], **compiler)
    return agent


def _stream_update(hasher, value):
    hasher.update((wire_hash(value) + '\n').encode())


def _shadow_row(name, start):
    return {'name': name, 'from_step': start, 'steps_checked': 0, 'actions_equal': True,
            'exact_plans_equal': True, 'first_divergence': None,
            'scope': 'LOCKSTEP_EQUIVALENCE_ON_ONE_PHYSICAL_WORLD', 'physical_games': 0}


def play_game(case: dict, arm: Arm, job: dict, registration_doc: dict, *, used_policies=None,
              retained_hold_gate=False) -> dict:
    """One physical game, optionally multiple NON-independent lockstep controls."""
    start, cpu_start = time.monotonic(), time.process_time()
    used = [] if used_policies is None else used_policies
    env = agent = None
    actions, joints, epochs, markets = [], [], [], []
    realization_rows, shadows, shadow_actors, checkpoints = [], [], [], []
    sold, receipts, spend, committed, attempts, unit_ops = (Counter() for _ in range(6))
    suffix_actions, suffix_obs = hashlib.sha256(), hashlib.sha256()
    gate_start = None
    row = {'schema': SCHEMA + '-game', 'job': copy.deepcopy(job), 'case_id': case['id'],
           'scenario': case['scenario'], 'seat': case['seat'], 'mode': job['mode'],
           'sampler_seed': job['sampler_seed'], 'world_seed': case['scenario']['seed'],
           'opponent_identity': case['opponent_identity'], 'compiler': registration_doc['compiler'],
           'registration_sha256': registration_doc['content_sha256'],
           'arm_identity_sha256': wire_hash(arm.identity),
           'scope': SCOPE, 'physical_games_started': 0, 'physical_games_completed': 0,
           'physical_steps': 0, 'status': 'FAILED', 'own_final_cash': None, 'rival_final_cash': None,
           'margin': None, 'wdl': None, 'no_automatic_retry': True}
    try:
        # Charge an attempted original-rule initialization even if it throws.
        if case['scenario'].get('schema') == ORIGINAL_CASE_SCHEMA:
            row['physical_games_started'] = 1
        env = environment_for_case(case)
        row['physical_games_started'] = 1
        row['scenario_identity'] = env.scenario_identity
        other, identity = scenarios.opponent(case['scenario']['opponent'])
        if identity != case['opponent_identity']:
            raise ValueError('frozen opponent identity changed')
        agent = _make_agent(arm, job, case, registration_doc['compiler'], used)
        row['policy_initial_state'] = _policy_snapshot(agent.policy)
        ps = row['policy_initial_state']
        row['gpu_seconds'] = (0. if ps['kind'] in ('KEEP', 'TRAJECTORY', 'TARGET')
                              or (ps['kind'] == 'SCORER' and ps['state']['device'] == 'cpu') else None)
        row['gpu_timing'] = 'CPU_POLICY' if row['gpu_seconds'] == 0 else 'NOT_MEASURED_BY_HARNESS'
        if arm.initial_plan is not None:
            proof = target_edits(State(env.agent_observation(case['seat'])), Plan(), arm.initial_plan,
                                 projection_kwargs=agent.projection_kw())
            row['compiled_target_reachability'] = proof.trace
        if arm.shadow is not None:
            shadow = _make_agent(arm.shadow, job, case, registration_doc['compiler'], used)
            if arm.shadow.initial_plan is not None:
                proof = target_edits(State(env.agent_observation(case['seat'])), Plan(), arm.shadow.initial_plan,
                                     projection_kwargs=shadow.projection_kw())
                row['shadow_target_reachability'] = proof.trace
            shadows.append(_shadow_row(arm.shadow.name, 0)); shadow_actors.append(shadow)
        with realization.market_tap(env, case['seat']) as tap:
            while not env.done:
                if time.monotonic() - start > registration_doc['timeout_seconds']:
                    raise TimeoutError('bounded V0 game deadline')
                step, seat = env.step_index, case['seat']
                obs = env.agent_observation(seat)
                if step in registration_doc['restart_steps']:
                    payload = snapshot_agent(agent, obs)
                    supplied = None
                    if payload['policy']['kind'] in ('SCORER', 'EXTERNAL'):
                        supplied = arm.factory(job['mode'], job['sampler_seed'])
                        if any(supplied is p for p in used):
                            raise ValueError('restart factory reused live policy')
                        used.append(supplied)
                    fresh = restore_agent(json.loads(wire(payload)), obs, policy=supplied)
                    checkpoints.append({'actor_observation': obs, 'actor': payload,
                                        'visibility': 'OWN_ACTOR_OBSERVATION_AND_OWN_CONTINUATION'})
                    shadows.append(_shadow_row(f'restart-{step}', step)); shadow_actors.append(fresh)
                if retained_hold_gate and step == 528:
                    gate_start = {'observation_sha256': wire_hash(obs), 'plan_sha256': agent.plan.digest(),
                                  'allocation': [[*k, *v] for k, v in agent.allocation.items()],
                                  'started': [[*k, *v] for k, v in agent.started.items()],
                                  'hold_releases': [[p, *a, t] for p, (a, t) in agent.hold_releases.items()],
                                  'controller_sha256': wire_hash(agent.controller.snapshot())}
                old_records = len(agent.records)
                action = agent(copy.deepcopy(obs), copy.deepcopy(env.configuration))
                if len(agent.records) - old_records != int(step % 24 == 0):
                    raise ValueError('dawn-only publication violated')
                if step % 24 == 0:
                    epochs.append(copy.deepcopy(agent.records[-1]))
                for index, shadow in enumerate(shadow_actors):
                    check = shadows[index]
                    if check['first_divergence'] is not None:
                        continue
                    shadow_action = shadow(copy.deepcopy(obs), copy.deepcopy(env.configuration))
                    check['steps_checked'] += 1
                    check['exact_plans_equal'] &= shadow.plan == agent.plan
                    if wire(shadow_action) != wire(action):
                        check['actions_equal'] = False
                        check['first_divergence'] = {'step': step, 'primary': action, 'shadow': shadow_action}
                    if check['name'].startswith('restart-') and agent_signature(shadow) != agent_signature(agent):
                        check['first_divergence'] = {'step': step, 'error': 'full continuation signature differs'}
                        check['actions_equal'] = False
                joint = [None, None]
                joint[seat] = action
                joint[1-seat] = other(env.agent_observation(1-seat), copy.deepcopy(env.configuration))
                tap['commits'], tap['pre_market'], tap['unit_effects'] = {}, None, []
                env.step(joint)
                actions.append(copy.deepcopy(action)); joints.append(copy.deepcopy(joint))
                if step >= 528:
                    _stream_update(suffix_actions, joint)
                    _stream_update(suffix_obs, env.agent_observation(seat))
                for (op, item), event in tap['commits'].items():
                    label = f'{op}:{item}'
                    committed[label] += event['units']
                    attempts[label] += event['units'] + event['failed_unit_attempts']
                    markets.append({'step': step, **copy.deepcopy(event)})
                    if op == 'SELL':
                        sold[item] += event['units']; receipts[item] += event['cash_delta']
                    else:
                        spend[label] -= event['cash_delta']
                for event in tap['unit_effects']:
                    unit_ops[event['engine_received_action'][0]] += 1
                    realization_rows.append({'step': step, **copy.deepcopy(event)})
        if env.step_index != V.LAST_STEP + 1 or len(epochs) != len(DAWNS):
            raise ValueError('not a complete 719-step, 30-dawn game')
        final = env.agent_observation(case['seat'])
        cash = [f['money'] for f in final['farms']]
        margin = cash[case['seat']] - cash[1-case['seat']]
        row.update(status='COMPLETE', physical_games_completed=1, own_final_cash=cash[case['seat']],
                   rival_final_cash=cash[1-case['seat']], margin=margin,
                   wdl='WIN' if margin > 0 else 'LOSS' if margin < 0 else 'DRAW',
                   terminal_cash_by_seat=cash, terminal_plan=agent.plan.canonical(),
                   terminal_controller_sha256=wire_hash(agent.controller.snapshot()),
                   terminal_environment_sha256=wire_hash(env.state), terminal_agent_signature=agent_signature(agent))
        for check in shadows:
            exact = check['actions_equal'] and check['steps_checked'] == V.LAST_STEP + 1 - check['from_step']
            check.update(complete_action_equivalence=exact, own_final_cash=cash[case['seat']] if exact else None,
                         rival_final_cash=cash[1-case['seat']] if exact else None,
                         margin=margin if exact else None, wdl=row['wdl'] if exact else None,
                         outcome_basis='DETERMINISTIC_IDENTICAL_JOINT_ACTION_STREAM' if exact else 'UNMEASURED')
        if retained_hold_gate:
            row['retained_hold_gate'] = _hold_gate(row, gate_start, suffix_actions.hexdigest(), suffix_obs.hexdigest(), markets)
    except Exception as exc:
        row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        # Any exception, including terminal verification, is a failed evaluation;
        # retain terminal diagnostics separately, never invent a successful arm.
        if row['status'] == 'COMPLETE':
            row['terminal_diagnostics_before_failure'] = {k: row[k] for k in ('own_final_cash', 'rival_final_cash', 'margin', 'wdl')}
        row.update(status='FAILED', physical_games_completed=0, own_final_cash=None, rival_final_cash=None, margin=None, wdl=None)
    row.update(physical_steps=0 if env is None else env.step_index, own_actions=actions, joint_actions=joints,
               own_actions_sha256=wire_hash(actions), joint_actions_sha256=wire_hash(joints), epochs=epochs,
               checkpoints=checkpoints, shadows=shadows, market_events=markets, unit_events=realization_rows,
               realization={'sold': dict(sold), 'receipts': dict(receipts), 'spend': dict(spend),
                            'committed_units': dict(committed), 'attempted_units': dict(attempts),
                            'executed_unit_operations': dict(unit_ops),
                            'publication_refusals': sum(r['rejected'] for r in epochs),
                            'unplaced_by_dawn': [{'step': r['step'], 'count': r['unplaced']} for r in epochs],
                            'scope': 'NATIVE_COMMITS_AND_UNIT_DELTAS; NOT_PROJECTED_FULFILLMENT'},
               wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu_start)
    row['checks_passed'] = (row['status'] == 'COMPLETE'
        and all(s.get('complete_action_equivalence', False) for s in shadows)
        and row.get('retained_hold_gate', {}).get('passed', True))
    return bind(row)


def _hold_gate(row, start, action_sha, observation_sha, markets):
    directory = ROOT / 'work/runs/reviewer-play/v0-backsolve/native'
    cut = json.loads((directory / 'CUT.json').read_text())
    reference = json.loads((directory / 'REFERENCE.json').read_text())
    actor = cut['actor']
    checks = {'witness_capsule_hash': actor['content_sha256'] == wire_hash({k: v for k, v in actor.items() if k != 'content_sha256'}),
              'witness_source_pins': all(file_sha(ROOT / p) == sha for p, sha in actor['source_pins'].items()),
              'cut_observation': start is not None and start['observation_sha256'] == actor['observation_sha256'],
              'cut_plan': start is not None and start['plan_sha256'] == Plan.from_canonical(actor['plan']).digest(),
              'cut_controller': start is not None and start['controller_sha256'] == wire_hash(actor['controller']),
              'action_stream_sha256': action_sha == reference['action_stream_sha256'],
              'observation_stream_sha256': observation_sha == reference['observation_stream_sha256'],
              'final_environment_sha256': row['terminal_environment_sha256'] == reference['final_environment_sha256'],
              'final_controller_sha256': row['terminal_controller_sha256'] == reference['final_controller_sha256'],
              'cash_by_seat': row['terminal_cash_by_seat'] == reference['cash_by_seat']}
    for key in ('allocation', 'started', 'hold_releases'):
        checks['cut_' + key] = start is not None and start[key] == actor[key]
    milk = [{'step': r['step'], 'units': r['units'], 'cash': r['cash_delta']} for r in markets
            if r['step'] >= 528 and r['op'] == 'SELL' and r['item'] == 'MILK' and r['units']]
    checks['milk_sales'] = milk == reference['milk_sales']
    return {'passed': all(checks.values()), 'checks': checks, 'suffix_steps': 191,
            'first_milk_sale': milk[0] if milk else None, 'extra_native_games': 0,
            'witness_files': {str(p.relative_to(ROOT)): file_sha(p) for p in (directory / 'CUT.json', directory / 'REFERENCE.json')},
            'scope': 'EXECUTED_SUFFIX_COMPARED_TO_FROZEN_REFERENCE; NOT_RECOMPUTED_ADVERSE_ARM'}


def evaluate_panel(cases, arms, output, *, modes=('greedy_class',), sample_seeds=(0,), max_games=2,
                   labor_factor=4., hands_mode='daily', timeout_seconds=180., restart_steps=(),
                   retained_hold_gate=False) -> dict:
    arms = tuple(arms)
    plan = registration(cases, arms, modes=modes, sample_seeds=sample_seeds, max_games=max_games,
                        labor_factor=labor_factor, hands_mode=hands_mode, timeout_seconds=timeout_seconds,
                        restart_steps=restart_steps)
    plan = bind({**plan, 'retained_hold_gate': bool(retained_hold_gate)})
    output = Path(output)
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('symlinked output')
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'PLAN.json', plan)
    rows, receipts, used = [], [], []
    lookup = {c['id']: c for c in plan['cases']}
    actors = {a.name: a for a in arms}
    with (output / 'JOURNAL.jsonl').open('x') as journal:
        def emit(value):
            journal.write(wire(bind(value)) + '\n'); journal.flush(); os.fsync(journal.fileno())
        for job in plan['jobs']:
            emit({'event': 'START', 'job': job, 'registration_sha256': plan['content_sha256']})
            row = play_game(lookup[job['case_id']], actors[job['arm']], job, plan, used_policies=used,
                            retained_hold_gate=retained_hold_gate and job['arm'] == 'edits')
            name = f"game-{job['index']:04d}.json"
            write_new(output / name, row)
            receipts.append({'path': name, 'sha256': file_sha(output / name), 'content_sha256': row['content_sha256']})
            emit({'event': 'RESULT', 'job': job, 'status': row['status'], 'receipt': receipts[-1]})
            rows.append(row)
    pairs = []
    for mode in modes:
        for case in plan['cases']:
            seeds = (0,) if mode == 'greedy_class' else sample_seeds
            for seed in seeds:
                group = [r for r in rows if r['case_id'] == case['id'] and r['mode'] == mode and r['sampler_seed'] == seed]
                for child in group[1:]:
                    pairs.append({'case_id': case['id'], 'mode': mode, 'sampler_seed': seed,
                                  'control': group[0]['job']['arm'], 'child': child['job']['arm'],
                                  **compare_games(group[0], child)})
    source_unchanged = source_pins() == plan['source_pins'] and scenarios.sources() == plan['opponent_sources']
    result = {'schema': SCHEMA, 'scope': SCOPE, 'registration_sha256': plan['content_sha256'],
              'status': 'PASS' if source_unchanged and all(r['checks_passed'] for r in rows) else 'FAIL',
              'sources_unchanged': source_unchanged, 'requested_games': len(plan['jobs']), 'attempted_jobs': len(rows),
              'physical_games_started': sum(r['physical_games_started'] for r in rows),
              'physical_games_completed': sum(r['physical_games_completed'] for r in rows),
              'failed_jobs': sum(r['status'] != 'COMPLETE' for r in rows),
              'physical_steps': sum(r['physical_steps'] for r in rows),
              'shadow_controllers': sum(len(r['shadows']) for r in rows), 'extra_shadow_physical_games': 0,
              'cpu_seconds': sum(r['cpu_seconds'] for r in rows), 'wall_seconds_games': sum(r['wall_seconds'] for r in rows),
              'fresh_learning_games': 0, 'optimizer_updates': 0,
              'gpu_seconds': 0. if all(r.get('gpu_seconds') == 0 for r in rows) else None,
              'games': receipts, 'paired_comparisons': pairs,
              'outcomes': [{k: r[k] for k in ('job', 'status', 'checks_passed', 'own_final_cash', 'rival_final_cash',
                           'margin', 'wdl', 'shadows')} for r in rows]}
    return write_new(output / 'RESULT.json', result)


def smoke_arms():
    cow = V.MODULE_BY_KEY['COW:fed_cared:b01']
    plus = V.encode('PROD', cow, V.DELTAS.index('PLUS_ONE'))
    hold = V.encode('MARKET', V.PRODUCTS.index('MILK'), V.PRIMITIVES.index('HOLD'), V.HORIZONS.index('NEXT_DAWN'), 0)
    identity = {'kind': 'SCRIPTED_CPU_STUB', 'schedule': [[0, [plus]], [504, [hold]]], 'not_neural': True}
    target = Arm('compiled_target', lambda mode, seed: TrajectoryPolicy({504: [hold]}),
                 {'kind': 'REACHABLE_COMPILED_TARGET', 'hold_step': 504}, kind='compiled_target', initial_plan=Plan({cow: 1}))
    return (Arm('edits', lambda mode, seed: TrajectoryPolicy({0: [plus], 504: [hold]}), identity, shadow=target),
            Arm('keep', lambda mode, seed: KeepPolicy(), {'kind': 'KEEP_STOP', 'not_neural': True}, kind='keep'))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke',))
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=MODES, default='greedy_class')
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--max-games', type=int, default=2)
    parser.add_argument('--timeout-seconds', type=float, default=180.)
    args = parser.parse_args(argv)
    result = evaluate_panel([Case('router-dairy', 0, 'pass', 'backsolve-bcbffb5d-seat0')], smoke_arms(), args.output,
                            modes=(args.mode,), sample_seeds=(args.sample_seed,), max_games=args.max_games,
                            timeout_seconds=args.timeout_seconds, restart_steps=(528,), retained_hold_gate=True)
    print(wire({k: v for k, v in result.items() if k not in ('outcomes',)}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
