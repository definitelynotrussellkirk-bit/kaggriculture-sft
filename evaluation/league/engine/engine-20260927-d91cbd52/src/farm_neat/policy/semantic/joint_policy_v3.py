"""PopulationV2 plus post-edit and commitment context; unchanged semantic PPO.

New80-wide representation, not a relabelled V1 action vocabulary. CONTROL and
JOINT have identical geometry; only CONTROL's added columns are zero. Migration
accepts a true population64 parent and preserves its old columns/moments/age.
No collector, optimization loop or execution policy is implemented here.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import types

import numpy as np

from farm_neat.training.schedule_model import bound, file_sha, tensor_sha
from farm_neat.training.schedule import validate_settings
from farm_neat.policy.semantic import policy_v1 as original
from farm_neat.policy.semantic import population_v2 as population
from farm_neat.policy.semantic import postedit_policy_v2 as postedit
from farm_neat.policy.semantic import commitment_context as commitment

SCHEMA = "farm-semantic-joint-population-actor-v3"
ORDER_SCHEMA = "farm-semantic-joint-population-orders-v3"
MODES = ("CONTROL", "JOINT")
WIDTH = 80
ACTIONS, ACTION_IDS, ACTION_SCHEMA = population.ACTIONS, population.ACTION_IDS, population.ACTION_SCHEMA
SHAPES = {**population.SHAPES, "shared.weight": (32, WIDTH)}
TRAINABLE = population.TRAINABLE
MISSING_ESTIMATE = -2.0  # Reserved sentinel; valid normalized estimates are[-1,1].
COMMITMENT_FEATURES = (
    "before_wheat_sale_reserve_div100", "after_wheat_sale_reserve_div100",
    "after_remaining_wheat_cycles_div12", "after_held_wheat_minus_feed_reserve_div24",
    "after_due_unfed_cows_div12", "after_optimistic_feed_slack_div24_or_missing",
    "after_due_cows_without_current_feed_source_div12",
    "after_optimistic_crop_deadline_slack_div29_or_missing",
)
REPRESENTATION_SCHEMA = bound({
    "schema": "farm-semantic-joint-population-context-v3", "width": WIDTH,
    "population_columns": [0, 63], "postedit_columns": [63, 71],
    "commitment_columns": [71, 79], "action_id_column": 79,
    "postedit_features": list(postedit.FEATURES),
    "commitment_features": list(COMMITMENT_FEATURES),
    "missing_estimate": MISSING_ESTIMATE,
    "valid_estimate_range": [-1.0, 1.0],
    "missing_feed_meaning": "NO_DUE_COW_WITH_A_CURRENT_HELD_FEED_ROUTE; DUE_AND_SOURCE_COUNTS_REMAIN_EXPLICIT",
    "missing_deadline_meaning": "NO_UNFINISHED_CROP_COMMITMENT",
    "heuristics_are_not_feasibility_certificates": True,
    "control_added_columns": "ZERO63_TO78", "modes": list(MODES),
})


def sources():
    return {**population.sources(),
        "src/farm_neat/policy/semantic/postedit_policy_v2.py": file_sha(postedit.__file__),
        "src/farm_neat/policy/semantic/commitment_context.py": file_sha(commitment.__file__),
        "src/farm_neat/policy/semantic/joint_policy_v3.py": file_sha(__file__)}


def _bind(function, **scope):
    result = types.FunctionType(function.__code__, {**function.__globals__, **scope},
                                argdefs=function.__defaults__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class JointOrders(population.PopulationOrders):
    """Only save/restore the feature mode; execution remains PopulationOrders."""
    def __init__(self, *, hands=2, mode="JOINT"):
        if mode not in MODES:
            raise ValueError("explicit CONTROL or JOINT mode required")
        super().__init__(hands=hands)
        self.joint_feature_mode = mode

    def snapshot(self):
        return {**super().snapshot(), "schema": ORDER_SCHEMA,
                "joint_feature_mode": self.joint_feature_mode}

    @classmethod
    def restore(cls, state):
        if state.get("schema") != ORDER_SCHEMA or state.get("joint_feature_mode") not in MODES:
            raise ValueError("versioned joint order state and mode required")
        old = copy.deepcopy(state)
        old["schema"] = population.ORDER_SCHEMA
        mode = old.pop("joint_feature_mode")
        result = population.PopulationOrders.restore.__func__(cls, old)
        result.joint_feature_mode = mode
        return result


def initial_orders(mode="JOINT"):
    orders = JointOrders(hands=2, mode=mode)
    for item in ("WHEAT", "MILK", "FERTILIZER"):
        orders.sell_when(item, price_above=0)
    return orders


apply_command = population.apply_command


def _normalized(value, scale):
    return float(np.clip(value/scale, -1.0, 1.0))


def _estimate(value, scale):
    return MISSING_ESTIMATE if value is None else _normalized(value, scale)


def commitment_block(observation, orders, index, allowed):
    facts = commitment.candidate_commitment_facts(observation, orders, index,
        apply_command=apply_command, allowed=allowed)
    before, after = facts["before"], facts["after"]
    return np.asarray([
        _normalized(before["sale_reserves"]["WHEAT"], 100),
        _normalized(after["sale_reserves"]["WHEAT"], 100),
        _normalized(after["remaining_wheat_cycles"], 12),
        _normalized(after["held_wheat_minus_standing_feed_reserve"], 24),
        _normalized(after["due_unfed_cows"], 12),
        _estimate(after["optimistic_worst_feed_slack_turns"], 24),
        _normalized(after["due_cows_without_current_feed_source"], 12),
        _estimate(after["optimistic_worst_crop_deadline_slack_days"], 29),
    ], dtype=np.float32)


def pack(observation, orders):
    mode = getattr(orders, "joint_feature_mode", None)
    if mode not in MODES:
        raise ValueError("controller requires explicit joint feature mode")
    old, mask = population.pack(observation, orders)
    rows = np.zeros((29, WIDTH), np.float32)
    rows[:, :63], rows[:, 79] = old[:, :63], old[:, 63]
    if mode == "JOINT":
        for index, command in enumerate(ACTIONS):
            allowed = bool(mask[index])
            rows[index, 63:71] = postedit.post_edit_block(
                orders, command if allowed else {"kind": "NO_EDIT"})
            rows[index, 71:79] = commitment_block(observation, orders, index, allowed)
    return validate_inputs(rows, mask)


def validate_inputs(rows, mask):
    try:
        rows, mask = _bind(original.validate_inputs, WIDTH=WIDTH)(rows, mask)
    except ValueError as exc:
        raise ValueError("exact joint29x80 rows and population mask required") from exc
    # Sentinels occur only in the two declared estimate slots, never as values
    # that could be confused with a valid clipped estimate or an unknown mask.
    for index in (76, 78):
        values = rows[:, index]
        if not np.all((values == MISSING_ESTIMATE) | ((values >= -1) & (values <= 1))):
            raise ValueError("explicit known estimate or missing sentinel required")
    for start, end in ((71, 76), (77, 78)):
        if np.any(np.abs(rows[:, start:end]) > 1):
            raise ValueError("normalized commitment facts required")
    return rows, mask


_distribution = _bind(original.Policy.distribution, validate_inputs=validate_inputs)


class Policy(original.Policy):
    def distribution(self, rows, mask):
        if (self.manifest.get("schema") != SCHEMA
                or self.manifest.get("joint_feature_mode") not in MODES
                or self.manifest.get("representation_schema_sha256") != REPRESENTATION_SCHEMA["content_sha256"]):
            raise ValueError("joint policy representation/mode required")
        rows, mask = validate_inputs(rows, mask)
        if self.manifest["joint_feature_mode"] == "CONTROL" and np.any(rows[:, 63:79]):
            raise ValueError("CONTROL policy requires zero added feature block")
        return _distribution(self, rows, mask)


torch_graph = _bind(original.torch_graph, WIDTH=WIDTH)


def save_model(path, weights, metadata):
    if metadata.get("joint_feature_mode") not in MODES:
        raise ValueError("closed joint feature mode required")
    return _bind(original.save_model, SHAPES=SHAPES, WIDTH=WIDTH, SCHEMA=SCHEMA,
        ACTION_SCHEMA=ACTION_SCHEMA, sources=sources)(path, weights,
        {**metadata, "representation_schema_sha256": REPRESENTATION_SCHEMA["content_sha256"]})


def load_model(path):
    manifest, weights = _bind(original.load_model, SHAPES=SHAPES, SCHEMA=SCHEMA,
        ACTION_SCHEMA=ACTION_SCHEMA, sources=sources)(path)
    if (manifest.get("joint_feature_mode") not in MODES
            or manifest.get("candidate_width") != WIDTH
            or manifest.get("representation_schema_sha256") != REPRESENTATION_SCHEMA["content_sha256"]
            or any(weights[k].shape != shape or weights[k].dtype != np.float32
                   or not np.isfinite(weights[k]).all() for k, shape in SHAPES.items())):
        raise ValueError("joint checkpoint representation/mode/tensors mismatch")
    return manifest, weights


def fit(model_path, episodes, output_model, **kwargs):
    """Same PPO bytecode; caller must verify fresh games AND episode tensors."""
    manifest, _ = load_model(model_path)
    result = _bind(original.fit, load_model=load_model, save_model=save_model,
        Policy=Policy, validate_inputs=validate_inputs, torch_graph=torch_graph,
        sources=sources, ACTION_SCHEMA=ACTION_SCHEMA, ACTION_IDS=ACTION_IDS,
        TRAINABLE=TRAINABLE)(model_path, episodes, output_model, **kwargs)
    return bound({**result, "schema": "farm-semantic-joint-population-ppo-fit-v3",
        "representation_schema_sha256": REPRESENTATION_SCHEMA["content_sha256"],
        "joint_feature_mode": manifest["joint_feature_mode"]})


def _widen(value):
    """Preserve every population64 coefficient, insert sixteen zero columns."""
    if isinstance(value, np.ndarray):
        result = np.zeros((32, WIDTH), dtype=value.dtype)
    else:
        result = value.new_zeros((32, WIDTH))
    result[:, :63], result[:, 79] = value[:, :63], value[:, 63]
    return result


def migrate(model, optimizer, output, mode):
    """Explicit zero-update population64→joint80 migration, never V1 parity.

    An absent optimizer is allowed only for a true age0/rollout0 population model.
    Existing optimizer moments are relocated, not reset, including action-ID
    column63→79. New zero-moment columns represent only newly introduced features.
    """
    if mode not in MODES:
        raise ValueError("CONTROL or JOINT mode required")
    manifest, weights = population.load_model(model)
    age, iteration = manifest["optimizer_steps"], manifest["rollout_iteration"]
    if type(age) is not int or age < 0 or type(iteration) is not int or iteration < 0:
        raise ValueError("nonnegative exact population age required")
    output = Path(output)
    if output.exists():
        raise FileExistsError("new joint migration directory required")
    checkpoint = None
    if optimizer is None:
        if age != 0 or iteration != 0:
            raise ValueError("trained population parent requires exact optimizer; no age reset")
    else:
        import torch
        checkpoint = torch.load(optimizer, map_location="cpu", weights_only=True)
        if (checkpoint.get("schema") != "farm-economic-ppo-optimizer-v2"
                or checkpoint.get("model_sha256") != file_sha(model)
                or checkpoint.get("global_step") != age
                or checkpoint.get("last_rollout_iteration") != iteration):
            raise ValueError("exact population model/optimizer/age required")
        validate_settings(checkpoint["settings"])
        state = checkpoint["optimizer"]
        groups = state["param_groups"]
        if len(groups) != 1 or len(groups[0]["params"]) != len(TRAINABLE):
            raise ValueError("maintained Adam parameter ordering required")
        ids = groups[0]["params"]
        if len(set(ids)) != len(ids) or (age and set(state["state"]) != set(ids)):
            raise ValueError("complete optimizer parameter state required")
        if set(state["state"]) - set(ids):
            raise ValueError("foreign optimizer parameter state")
        for key, parameter in zip(TRAINABLE, ids, strict=True):
            if parameter not in state["state"]:
                continue  # Properly untrained Adam state is empty.
            moment = state["state"][parameter]
            if (set(moment) != {"step", "exp_avg", "exp_avg_sq"}
                    or moment["step"].numel() != 1 or moment["step"].item() != age
                    or any(moment[k].shape != population.SHAPES[key]
                           or not torch.isfinite(moment[k]).all() for k in ("exp_avg", "exp_avg_sq"))):
                raise ValueError("optimizer moment shape/step mismatch")
        # Validate local RNG encoding without changing either process RNG.
        rng = np.random.default_rng(0)
        rng.bit_generator.state = json.loads(checkpoint["numpy_rng_state"])
        if (checkpoint["torch_rng_state"].dtype != torch.uint8
                or checkpoint["torch_rng_state"].shape != torch.get_rng_state().shape
                or not isinstance(checkpoint["recent_episode_ids"], list)):
            raise ValueError("exact optimizer RNG/history state required")
        try:
            torch.Generator().set_state(checkpoint["torch_rng_state"])
        except RuntimeError as exc:
            raise ValueError("invalid optimizer Torch RNG encoding") from exc
        checkpoint = copy.deepcopy(checkpoint)
        moments = checkpoint["optimizer"]["state"]
        if ids[0] in moments:
            for key in ("exp_avg", "exp_avg_sq"):
                moments[ids[0]][key] = _widen(moments[ids[0]][key])
    changed = {k: v.copy() for k, v in weights.items()}
    changed["shared.weight"] = _widen(weights["shared.weight"])
    output.mkdir(parents=True)
    child = output/"model.npz"
    save_model(child, changed, {**manifest, "joint_feature_mode": mode,
        "migration_parent_sha256": file_sha(model),
        "migration": "POPULATION64_TO_JOINT80_PRESERVE_OLD_COLUMNS_MOMENTS_RNG_AGE",
        "canonical_submission": False})
    optimizer_sha = None
    if checkpoint is not None:
        checkpoint["model_sha256"] = file_sha(child)
        with (output/"optimizer.pt").open("xb") as stream:
            torch.save(checkpoint, stream)
        optimizer_sha = file_sha(output/"optimizer.pt")
    receipt = bound({"schema": "farm-semantic-joint-population-migration-v3",
        "mode": mode, "sources": sources(),
        "representation_schema_sha256": REPRESENTATION_SCHEMA["content_sha256"],
        "parent_model_sha256": file_sha(model),
        "parent_optimizer_sha256": None if optimizer is None else file_sha(optimizer),
        "model_sha256": file_sha(child), "optimizer_sha256": optimizer_sha,
        "parent_tensor_sha256": tensor_sha(weights), "tensor_sha256": tensor_sha(changed),
        "old_column_mapping": "0:63->0:63;63->79", "new_zero_columns": [63, 79],
        "optimizer_steps": 0, "global_optimizer_steps": age, "rollout_iteration": iteration,
        "optimizer_state_present": checkpoint is not None,
        "behavior_parity": "NUMERICAL_PARITY_REQUIRES_SEPARATE_TESTS_NOT_A_STRENGTH_CLAIM"})
    with (output/"MIGRATION.json").open("x") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)+"\n")
    return receipt
