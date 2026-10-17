"""Candidate post-edit persistent-state features for the semantic command policy.

This is a source-versioned local ablation over the existing 29 commands and PPO.
It makes NO_EDIT and setter candidates comparable in terms of the persistent scalar
plan that would exist *after* the choice.  The executor, masks, terminal reward,
network shape and action meanings are unchanged.  V1 remains independently runnable.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import types

import numpy as np

from farm_neat.training.schedule_model import bound, file_sha, tensor_sha
from farm_neat.policy.semantic import policy_v1 as base
from farm_neat.policy.semantic import train_orders as original

ROOT = base.ROOT
SCHEMA = "farm-semantic-postedit-actor-v2"
MODES = ("CONTROL", "POST_EDIT")
SOURCE_PATHS = (*base.SOURCE_PATHS, "src/farm_neat/policy/semantic/postedit_policy_v2.py",
                "src/farm_neat/policy/semantic/train_orders.py")
FEATURES = (
    "post_hands_target_div4",
    "delta_hands_target_div4",
    "post_wheat_sale_threshold_div320",
    "delta_wheat_sale_threshold_div320",
    "post_milk_sale_threshold_div320",
    "delta_milk_sale_threshold_div320",
    "post_fertilizer_sale_threshold_div320",
    "delta_fertilizer_sale_threshold_div320",
)
ACTION_IDS, ACTIONS, ACTION_SCHEMA = base.ACTION_IDS, base.ACTIONS, base.ACTION_SCHEMA
TRAINABLE, SHAPES = base.TRAINABLE, base.SHAPES
Policy, torch_graph, apply_command = base.Policy, base.torch_graph, base.apply_command


def sources():
    return {name: file_sha(ROOT / name) for name in SOURCE_PATHS}


def _function(function, **scope):
    result = types.FunctionType(function.__code__, {**function.__globals__, **scope},
                                argdefs=function.__defaults__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def _thresholds(orders):
    return {item: orders.sale_rules.get(item, {"price_above": 0})["price_above"]
            for item in ("WHEAT", "MILK", "FERTILIZER")}


def post_edit_block(orders, command):
    """Actor-visible resulting scalar plan, not profitability or future execution.

    NO_EDIT and non-scalar commands retain all four scalar settings.  HANDS changes
    only the daily hand target; SELL changes only the selected product threshold.
    Signed deltas make replacement versus retention explicit without masking it.
    """
    before_hands = orders.hands
    before = _thresholds(orders)
    after_hands = command["target"] if command["kind"] == "HANDS" else before_hands
    after = dict(before)
    if command["kind"] == "SELL":
        after[command["item"]] = command["threshold"]
    return np.asarray([
        after_hands / 4,
        (after_hands - before_hands) / 4,
        after["WHEAT"] / 320,
        (after["WHEAT"] - before["WHEAT"]) / 320,
        after["MILK"] / 320,
        (after["MILK"] - before["MILK"]) / 320,
        after["FERTILIZER"] / 320,
        (after["FERTILIZER"] - before["FERTILIZER"]) / 320,
    ], dtype=np.float32)


def pack(observation, orders):
    rows, mask = base.pack(observation, orders)
    mode = getattr(orders, "postedit_feature_mode", None)
    if mode not in (*MODES, "LEGACY"):
        raise ValueError("explicit post-edit feature mode required")
    if np.any(rows[:, 55:63]):
        raise ValueError("formerly unused feature columns changed")
    if mode == "POST_EDIT":
        for i, command in enumerate(ACTIONS):
            rows[i, 55:63] = post_edit_block(orders, command)
    return rows, mask


def validate_inputs(rows, mask):
    return base.validate_inputs(rows, mask)


def save_model(path, weights, metadata):
    if metadata.get("postedit_feature_mode") not in MODES:
        raise ValueError("closed source-versioned feature mode required")
    return _function(base.save_model, SCHEMA=SCHEMA, sources=sources)(path, weights,
        {**metadata, "postedit_feature_names": FEATURES})


def load_model(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 1_000_000:
        raise ValueError("small regular semantic checkpoint required")
    with np.load(path, allow_pickle=False) as archive:
        schema = json.loads(archive["__manifest_json__"].tobytes())["schema"]
    if schema == base.SCHEMA:
        return base.load_model(path)
    manifest, weights = _function(base.load_model, SCHEMA=SCHEMA, sources=sources)(path)
    if (manifest.get("postedit_feature_mode") not in MODES or
            manifest.get("postedit_feature_names") != list(FEATURES)):
        raise ValueError("post-edit input meaning mismatch")
    return manifest, weights


def fit(model_path, episodes, output_model, **kwargs):
    manifest, _ = load_model(model_path)
    if manifest["schema"] != SCHEMA:
        raise ValueError("migrate explicitly before post-edit fitting")
    result = _function(base.fit, load_model=load_model, save_model=save_model,
        Policy=Policy, validate_inputs=validate_inputs, sources=sources)(
            model_path, episodes, output_model, **kwargs)
    return bound({**result, "schema": "farm-semantic-postedit-ppo-fit-v2",
                  "postedit_feature_mode": manifest["postedit_feature_mode"]})


def migrate(model, optimizer, output, mode):
    """Zero-update identity migration preserving optimizer moments/RNG/global age."""
    import torch
    if mode not in MODES:
        raise ValueError("CONTROL or POST_EDIT required")
    manifest, weights = base.load_model(model)
    checkpoint = torch.load(optimizer, map_location="cpu", weights_only=True)
    if (checkpoint["model_sha256"] != file_sha(model)
            or checkpoint["global_step"] != manifest["optimizer_steps"]
            or checkpoint["last_rollout_iteration"] != manifest["rollout_iteration"]):
        raise ValueError("exact parent model/optimizer age required")
    state = checkpoint["optimizer"]["state"][0]
    if (state["exp_avg"].shape != weights["shared.weight"].shape
            or any(torch.count_nonzero(state[k][:, 55:63]) for k in ("exp_avg", "exp_avg_sq"))):
        raise ValueError("unused-column moments are not zero; no silent reset")
    output = Path(output)
    if output.exists():
        raise FileExistsError("new migration directory required")
    changed = {k: v.copy() for k, v in weights.items()}
    changed["shared.weight"][:, 55:63] = 0
    output.mkdir(parents=True)
    child = output / "model.npz"
    save_model(child, changed, {**manifest, "postedit_feature_mode": mode,
        "migration_parent_sha256": file_sha(model),
        "migration": "ZERO_PREVIOUSLY_INACTIVE_COLUMNS_KEEP_ALL_MOMENTS_RNG_AGE",
        "canonical_submission": False})
    new_checkpoint = copy.deepcopy(checkpoint)
    new_checkpoint["model_sha256"] = file_sha(child)
    with (output / "optimizer.pt").open("xb") as stream:
        torch.save(new_checkpoint, stream)
    receipt = bound({"schema": "farm-semantic-postedit-migration-v2", "mode": mode,
        "sources": sources(), "parent_model_sha256": file_sha(model),
        "parent_optimizer_sha256": file_sha(optimizer), "model_sha256": file_sha(child),
        "optimizer_sha256": file_sha(output / "optimizer.pt"),
        "parent_tensor_sha256": tensor_sha(weights), "tensor_sha256": tensor_sha(changed),
        "only_zeroed_inactive_columns": [55, 63], "optimizer_state_values_unchanged": True,
        "optimizer_steps": 0, "global_optimizer_steps": checkpoint["global_step"],
        "behavior_parity": "REQUIRED_BY_SEPARATE_COMPLETE_GAME_TEST_NOT_ASSUMED_BY_MIGRATION"})
    original.write(output / "MIGRATION.json", receipt)
    return receipt


def play(job):
    modes = []
    for path in job["models"]:
        if path == "FIXED_COMMANDS":
            modes.append("LEGACY")
        else:
            manifest, _ = load_model(path)
            modes.append(manifest.get("postedit_feature_mode", "LEGACY"))
    neural_modes = iter(mode for mode, path in zip(modes, job["models"]) if path != "FIXED_COMMANDS")

    def initial_orders():
        controller = base.initial_orders()
        controller.postedit_feature_mode = next(neural_modes)
        return controller

    facade = types.SimpleNamespace(Policy=Policy, load_model=load_model, pack=pack,
        initial_orders=initial_orders, apply_command=apply_command, ACTIONS=ACTIONS)
    scope = {"binding": facade}
    if job.get("opponent") == "DEFAULT_PLAN_V03":
        raise NotImplementedError('the semantic benchmark harness was not migrated')
        if (type(job.get("seat")) is not int or job["seat"] not in (0, 1)
                or job["models"][1-job["seat"]] != "FIXED_COMMANDS"):
            raise ValueError("explicit Default Plan opponent seat required")
        scope["fixed_opponent"] = DefaultPlanAdapter
    result = _function(original.play, **scope)(job)
    if job.get("opponent") == "DEFAULT_PLAN_V03":
        POLICY_ID = None
        other = 1-job["seat"]
        result["record"]["model_sha256"][other] = "DEFAULT_PLAN_V03:" + POLICY_ID
        result["episodes"][other]["behavior_model_sha256"] = "NOT_A_NEURAL_TRAINING_EPISODE"
    return result


def pool_play(jobs, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(play, jobs))
