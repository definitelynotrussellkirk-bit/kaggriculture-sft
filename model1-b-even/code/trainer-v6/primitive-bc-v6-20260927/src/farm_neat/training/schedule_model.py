"""Versioned actor-critic weights; standalone inference needs only NumPy.

The old outcome head initializes an unbounded scalar preference. This is an
explicit new policy initialization, not an assertion of old-policy parity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "farm-economic-actor-critic-model-v2"
NUMERICS_LEGACY = "NUMPY_FP32_GRAPH_FP64_SOFTMAX_V1"
NUMERICS_CANONICAL = "FP64_FORWARD_FP32_PARAMETERS_V1"


def numerical_contract(manifest):
    """Missing means the frozen original graph, never an implicit migration."""
    value = manifest.get("numerics", NUMERICS_LEGACY)
    if value not in (NUMERICS_LEGACY, NUMERICS_CANONICAL):
        raise ValueError("unsupported actor-critic numerical contract")
    return value


ACTION_IDS = (0, 16, 17, 18, 25, 26, 27)
SHAPES = {
    "public_mean": (166,), "public_scale": (166,),
    "tail_mean": (136,), "tail_scale": (136,),
    "public_linear.weight": (32, 166), "public_linear.bias": (32,),
    "candidate_linear.weight": (64, 168), "candidate_linear.bias": (64,),
    "actor_head.weight": (1, 64), "actor_head.bias": (1,),
    "value_head.weight": (1, 64), "value_head.bias": (1,),
}
NORMALIZATION = ("public_mean", "public_scale", "tail_mean", "tail_scale")
TRAINABLE = tuple(k for k in SHAPES if k not in NORMALIZATION)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def bound(value):
    value = {k: v for k, v in value.items() if k != "content_sha256"}
    return {**value, "content_sha256": hashlib.sha256(canonical(value).encode()).hexdigest()}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(weights):
    digest = hashlib.sha256()
    for key, array in sorted(weights.items()):
        digest.update(canonical([key, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_weights(weights):
    if set(weights) != set(SHAPES):
        raise ValueError("exact v2 actor-critic tensor set required")
    for name, shape in SHAPES.items():
        array = weights[name]
        if (array.shape != shape or array.dtype != np.float32
                or not np.isfinite(array).all()):
            raise ValueError("invalid v2 tensor " + name)
    if any(np.any(weights[name] <= 0) for name in ("public_scale", "tail_scale")):
        raise ValueError("positive normalization scales required")


def save_model(path, weights, metadata):
    validate_weights(weights)
    numerical_contract(metadata)
    temperature = metadata.get("temperature")
    if (type(temperature) not in (float, int) or not np.isfinite(temperature)
            or temperature <= 0):
        raise ValueError("fixed positive policy temperature required")
    manifest = bound({**metadata, "schema": SCHEMA,
        "candidate_width": 302, "action_ids": list(ACTION_IDS),
        "tensor_sha256": tensor_sha(weights),
        "trainable_parameters": sum(weights[k].size for k in TRAINABLE),
        "policy_objective": "TERMINAL_WIN_DRAW_LOSS_PLUS1_ZERO_MINUS1",
        "normalization_frozen": True, "population_strength_established": False})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **weights,
            __manifest_json__=np.frombuffer(canonical(manifest).encode(), dtype=np.uint8))
    return manifest


def load_model(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("small regular v2 model required")
    import zipfile
    with zipfile.ZipFile(path) as archive:
        if sum(member.file_size for member in archive.infolist()) > 8 * 1024 * 1024:
            raise ValueError("oversized uncompressed model")
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != {*SHAPES, "__manifest_json__"}:
            raise ValueError("closed unique v2 model members required")
        encoded = archive["__manifest_json__"]
        if encoded.dtype != np.uint8 or encoded.ndim != 1:
            raise ValueError("byte manifest required")
        manifest = json.loads(encoded.tobytes().decode())
        weights = {key: archive[key].copy() for key in SHAPES}
    validate_weights(weights)
    if (manifest != bound(manifest) or manifest.get("schema") != SCHEMA
            or manifest.get("candidate_width") != 302
            or manifest.get("action_ids") != list(ACTION_IDS)
            or manifest.get("tensor_sha256") != tensor_sha(weights)):
        raise ValueError("v2 model identity or interface mismatch")
    temperature = manifest.get("temperature")
    if (type(temperature) not in (float, int) or not np.isfinite(temperature)
            or temperature <= 0):
        raise ValueError("invalid registered temperature")
    numerical_contract(manifest)
    return manifest, weights


def initialize_from_legacy(legacy_npz_path, output_npz_path, *, temperature=0.1):
    """Copy current features; initialize actor with WIN minus LOSS raw logits."""
    with np.load(legacy_npz_path, allow_pickle=False) as archive:
        old = {key: archive[key].copy() for key in archive.files if key != "__manifest_json__"}
    weights = {key: old[key].copy() for key in SHAPES if key in old}
    if old["outcome_head.weight"].shape != (3, 64) or old["outcome_head.bias"].shape != (3,):
        raise ValueError("current three-outcome initialization required")
    weights["actor_head.weight"] = (old["outcome_head.weight"][2:3] - old["outcome_head.weight"][0:1]).copy()
    weights["actor_head.bias"] = (old["outcome_head.bias"][2:3] - old["outcome_head.bias"][0:1]).copy()
    weights["value_head.weight"] = np.zeros((1, 64), dtype=np.float32)
    weights["value_head.bias"] = np.zeros(1, dtype=np.float32)
    return save_model(output_npz_path, weights, {
        "stage": "ACTOR_CRITIC_INITIALIZATION_NO_RL_UPDATES", "optimizer_steps": 0,
        "rollout_iteration": 0,
        "initialization_model_sha256": file_sha(legacy_npz_path), "temperature": float(temperature),
        "initialization": "COPY_CURRENT_TRUNKS_WIN_MINUS_LOSS_RAW_LOGITS_ZERO_CRITIC",
        "legacy_greedy_parity_claim": False,
    })
