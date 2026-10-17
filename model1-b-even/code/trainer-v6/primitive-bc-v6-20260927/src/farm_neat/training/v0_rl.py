"""Bounded terminal-WDL policy-gradient pilot for the native V0 decoder.

This module deliberately implements only the first V0 RL experiment described by
``docs/contracts/training.md``:

* one policy decision at the opening (step 0), then the frozen executor/continuation;
* direct ``sample_raw`` decoding from the native V0 scorer;
* undiscounted complete-game terminal W/D/L in ``{-1, 0, +1}``;
* within-registered-group mean-centred advantages, with NO standard-deviation
  normalization, critic, shaping, margin reward, flow head or future target;
* clipped importance policy gradient on the exact sampled RAW-ID path likelihood;
* a tiny fixed update budget and an Adam reset at the RL objective boundary;
* an optional BC anchor only as a separately named sibling arm, never hidden in
  the default arm.

The raw/class distinction is intentional.  ``v0_scorer`` records both raw action
and successor-class probability because classes are useful for supervised V0 BC.
For RL a behavior trace is a sequence of sampled raw IDs.  Unless a particular
raw-history decoder has separately established the required lumpability, replacing
its raw behavior likelihood by class mass changes the sampled distribution.  This
module therefore never uses ``class_path_logp`` in its policy-gradient ratio.

No game runner is hidden here.  ``plan`` produces a source-bound finite collection
and matched evaluation registration; ``validate`` admits complete rollout groups;
``train`` is the maintainer-side fit command and requires the pinned production
runtime for non-diagnostic checkpoints.  External review can exercise all contracts
on CPU fixtures without playing games or training production weights.

Example command surface (all files are immutable/pinned by SHA256)::

    PYTHONPATH=src python3 -B -m farm_neat.training.v0_rl plan \
        --registration REGISTRATION.json --registration-sha256 SHA --output PLAN.json

    PYTHONPATH=src python3 -B -m farm_neat.training.v0_rl validate \
        --plan PLAN.json --plan-sha256 SHA --rollouts ROLLOUTS.json --rollouts-sha256 SHA

    # maintainer only, pinned runtime, frozen BC parent named by PLAN.json
    PYTHONPATH=src python3 -B -m farm_neat.training.v0_rl train \
        --plan PLAN.json --plan-sha256 SHA --rollouts ROLLOUTS.json --rollouts-sha256 SHA \
        --arm pg_only --output NEW_DIR --updates 2 --device cuda

    # later measurement gate; descriptive complete-game WDL only, no promotion
    PYTHONPATH=src python3 -B -m farm_neat.training.v0_rl eval-diagnostics \
        --plan PLAN.json --plan-sha256 SHA --results EVAL.json --results-sha256 SHA
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.policy import v0_scorer as A
from farm_neat.policy.qwen_hybrid import Config
from farm_neat.training import v0_bc as BC
from farm_neat.training import v0_targets as T


SCHEMA = "farm-v0-terminal-wdl-pg-pilot-v1"
REGISTRATION_SCHEMA = SCHEMA + "-registration-v1"
PLAN_SCHEMA = SCHEMA + "-plan-v1"
ROLLOUT_SCHEMA = SCHEMA + "-rollouts-v1"
EVAL_SCHEMA = SCHEMA + "-evaluation-results-v1"
COMMIT_SCHEMA = SCHEMA + "-commit-v1"
RESULT_SCHEMA = SCHEMA + "-result-v1"

# Frozen experiment semantics.  Changes to any of these are a new pilot protocol,
# not a runtime option hidden behind a command-line default.
PROTOCOL = {
    "objective": "UNDISCOUNTED_COMPLETE_GAME_TERMINAL_WDL_MINUS1_0_PLUS1",
    "decision_scope": "OPENING_STEP_0_ONLY;ONE_CANONICAL_SAMPLE_RAW_TRACE;FROZEN_CONTINUATION_AFTER_OPENING",
    "likelihood": "EXACT_SAMPLED_RAW_ID_PATH;CLASS_PATH_LOGP_FOR_DIAGNOSTICS_ONLY_NOT_PG",
    "advantage": "WITHIN_REGISTERED_GROUP_RETURN_MINUS_GROUP_MEAN;NO_STD_NORMALIZATION;TIED_GROUP_ZERO",
    "actor_loss": "MEAN_COMPLETE_GAME_CLIPPED_IMPORTANCE_POLICY_GRADIENT",
    "clip_ratio": 0.20,
    "entropy_coefficient": 0.0,
    "value_loss": None,
    "bootstrap": None,
    "discount": 1.0,
    "optimizer": "ADAM_RESET_AT_RL_OBJECTIVE_BOUNDARY;BETAS_0.9_0.999;EPS_1E-8;NO_WEIGHT_DECAY",
    "learning_rate": 1.0e-5,
    "max_grad_norm": 1.0,
    "optimizer_updates": 2,
    "temperature": 1.0,
    "menu_backend": "fast",
    "behavior_decode_mode": "sample_raw",
    "gradient_denominator": "ALL_ADMITTED_COMPLETE_GAMES_IN_ALL_ADMITTED_GROUPS",
    "invalid_or_censored": "ENTIRE_REGISTERED_GROUP_EXCLUDED_FROM_LEARNING",
    "optimizer_transition": "RESET_MOMENTS;DO_NOT_INHERIT_BC_ADAM_MOMENTS",
}

MAX_TRAIN_GROUPS = 16
MAX_GROUP_SIZE = 8
MAX_TRAIN_GAMES = 64
MAX_EVAL_CASES = 16
MAX_PLANNED_NATIVE_GAMES = 96
MAX_BC_ANCHOR_TRAJECTORIES = 64

ARM_PG = "pg_only"
ARM_ANCHOR = "pg_plus_bc_anchor"
BASELINE_ARM = "frozen_bc_baseline"


# ---------------------------------------------------------------------------
# canonical/source helpers

def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_loads(text: str) -> Any:
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_pairs)


def _pinned_json(path: str | Path, expected_sha256: str) -> dict:
    path = Path(path)
    _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64,
             "explicit SHA256 required")
    _require(path.is_file() and file_sha(path) == expected_sha256, "file hash mismatch: " + str(path))
    value = _json_loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "JSON object required: " + str(path))
    return value


def _write_json_new(path: str | Path, value: dict) -> None:
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlinked output")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "lowercase SHA256 required for " + label)
    return value


def _nonempty(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), "nonempty " + label + " required")
    return value


def _repo_relative(value: Any, label: str) -> str:
    value = _nonempty(value, label)
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts, label + " must be repository-relative")
    return path.as_posix()


def source_pins() -> dict[str, str]:
    # Bind the ticket's authoritative contracts/inputs in addition to the live scorer closure.
    # A plan collected under changed reward, policy, action-space or executor semantics is a
    # different experiment and must be registered again rather than resumed opportunistically.
    names = {
        "docs/contracts/training.md": ROOT / "docs/contracts/training.md",
        "docs/contracts/policy.md": ROOT / "docs/contracts/policy.md",
        "docs/guides/action-space-v0.md": ROOT / "docs/guides/action-space-v0.md",
        "src/farm_neat/economy/v0/epochs.py": ROOT / "src/farm_neat/economy/v0/epochs.py",
        "src/farm_neat/training/program_bc.py": ROOT / "src/farm_neat/training/program_bc.py",
        "src/farm_neat/training/v0_bc.py": ROOT / "src/farm_neat/training/v0_bc.py",
        "src/farm_neat/training/v0_rl.py": Path(__file__),
        "src/farm_neat/training/v0_targets.py": ROOT / "src/farm_neat/training/v0_targets.py",
    }
    return {**A.source_pins(), **{name: file_sha(path) for name, path in names.items()}}


def _bind(value: dict) -> dict:
    body = {k: v for k, v in value.items() if k != "content_sha256"}
    return {**body, "content_sha256": digest(body)}


def _verify_bound(value: dict, schema: str) -> None:
    _require(value.get("schema") == schema, "wrong schema")
    saved = value.get("content_sha256")
    _sha256(saved, label="content_sha256")
    _require(saved == digest({k: v for k, v in value.items() if k != "content_sha256"}),
             "bound content hash mismatch")


# ---------------------------------------------------------------------------
# frozen BC parent and pilot registration

def inspect_bc_parent(directory: str | Path, expected_commit_sha256: str) -> dict:
    """Verify the immutable BC parent without allocating its model tensors."""
    directory = Path(directory)
    commit_path = directory / "COMMIT.json"
    commit = _pinned_json(commit_path, expected_commit_sha256)
    _require(commit.get("schema") == BC.COMMIT_SCHEMA and commit.get("status") == "COMMITTED",
             "frozen native V0 BC COMMIT required")
    identity = commit.get("identity")
    _require(isinstance(identity, dict) and identity.get("schema") == BC.SCHEMA,
             "native V0 BC checkpoint identity required")
    model_identity = identity.get("model")
    BC.validate_model_manifest(model_identity)
    _require(model_identity["source_pins"] == A.source_pins(),
             "BC parent scorer/decoder sources differ from current V0 decoder")
    _require(identity.get("sources") == BC.source_pins(),
             "BC parent training source closure differs from current source")
    state_path = directory / "state.pt"
    _require(state_path.is_file()
             and state_path.stat().st_size == commit.get("state_bytes")
             and file_sha(state_path) == commit.get("state_sha256"),
             "BC parent payload hash/size mismatch")
    return {
        "commit_sha256": expected_commit_sha256,
        "commit_path": str(commit_path),
        "state_sha256": commit["state_sha256"],
        "state_bytes": commit["state_bytes"],
        "global_update": commit["global_update"],
        "samples_seen": commit["samples_seen"],
        "model_identity": copy.deepcopy(model_identity),
        "bc_identity_sha256": digest(identity),
    }


def _anchor_spec(value: Any) -> dict | None:
    if value is None:
        return None
    _require(isinstance(value, dict)
             and set(value) == {"dataset_path", "manifest_sha256", "split", "coefficient"},
             "bc_anchor must be null or a closed dataset/split/coefficient object")
    result = {
        "dataset_path": _repo_relative(value["dataset_path"], "BC anchor dataset_path"),
        "manifest_sha256": _sha256(value["manifest_sha256"], label="BC anchor manifest"),
        "split": _nonempty(value["split"], "BC anchor split"),
        "coefficient": value["coefficient"],
    }
    coefficient = result["coefficient"]
    _require(type(coefficient) in (int, float) and math.isfinite(coefficient) and 0.0 < coefficient <= 1.0,
             "BC anchor coefficient must be finite in (0, 1]")
    result["coefficient"] = float(coefficient)
    return result


def _world(row: dict, *, group: bool) -> dict:
    expected = ({"group_id", "world_id", "world_sha256", "seat", "opponent_id", "sample_seeds"}
                if group else {"case_id", "world_id", "world_sha256", "seat", "opponent_id", "sampler_seed"})
    _require(isinstance(row, dict) and set(row) == expected, "closed train/evaluation world registration required")
    key = "group_id" if group else "case_id"
    result = {
        key: _nonempty(row[key], key),
        "world_id": _nonempty(row["world_id"], "world_id"),
        "world_sha256": _sha256(row["world_sha256"], label="world"),
        "seat": row["seat"],
        "opponent_id": _nonempty(row["opponent_id"], "opponent_id"),
    }
    _require(type(result["seat"]) is int and result["seat"] in (0, 1), "seat must be 0 or 1")
    if group:
        seeds = row["sample_seeds"]
        _require(isinstance(seeds, list) and 2 <= len(seeds) <= MAX_GROUP_SIZE
                 and len(set(seeds)) == len(seeds)
                 and all(type(s) is int and 0 <= s < 2**32 for s in seeds),
                 "each train group needs 2..8 distinct uint32 sample seeds")
        result["sample_seeds"] = list(seeds)
    else:
        seed = row["sampler_seed"]
        _require(type(seed) is int and 0 <= seed < 2**32, "evaluation sampler_seed must be uint32")
        result["sampler_seed"] = seed
    return result


def validate_registration(value: dict, parent: dict) -> dict:
    """Close every pilot choice that can otherwise drift between collection and fit."""
    expected = {"schema", "behavior_checkpoint", "opponents", "train_groups",
                "evaluation_cases", "seed", "bc_anchor"}
    _require(isinstance(value, dict) and set(value) == expected and value.get("schema") == REGISTRATION_SCHEMA,
             "closed V0 RL registration required")
    behavior = value["behavior_checkpoint"]
    _require(isinstance(behavior, dict) and set(behavior) == {"path", "commit_sha256"},
             "behavior_checkpoint path and commit_sha256 required")
    behavior = {"path": _repo_relative(behavior["path"], "behavior checkpoint path"),
                "commit_sha256": _sha256(behavior["commit_sha256"], label="behavior checkpoint")}
    _require(behavior["commit_sha256"] == parent["commit_sha256"], "inspected BC parent hash differs from registration")

    opponents = value["opponents"]
    _require(isinstance(opponents, list) and 1 <= len(opponents) <= 8, "1..8 frozen opponents required")
    closed_opponents = []
    for row in opponents:
        _require(isinstance(row, dict) and set(row) == {"id", "sha256"}, "closed opponent identity required")
        closed_opponents.append({"id": _nonempty(row["id"], "opponent id"),
                                 "sha256": _sha256(row["sha256"], label="opponent")})
    _require(len({o["id"] for o in closed_opponents}) == len(closed_opponents), "distinct opponent ids required")
    opponent_ids = {o["id"] for o in closed_opponents}

    groups = value["train_groups"]
    _require(isinstance(groups, list) and 1 <= len(groups) <= MAX_TRAIN_GROUPS,
             "bounded nonempty train_groups required")
    closed_groups = [_world(row, group=True) for row in groups]
    _require(len({g["group_id"] for g in closed_groups}) == len(closed_groups), "distinct train group ids required")
    _require(all(g["opponent_id"] in opponent_ids for g in closed_groups), "unknown train opponent")
    train_games = sum(len(g["sample_seeds"]) for g in closed_groups)
    _require(train_games <= MAX_TRAIN_GAMES, "pilot train collection exceeds 64 complete games")

    cases = value["evaluation_cases"]
    _require(isinstance(cases, list) and 1 <= len(cases) <= MAX_EVAL_CASES,
             "bounded nonempty evaluation_cases required")
    closed_cases = [_world(row, group=False) for row in cases]
    _require(len({c["case_id"] for c in closed_cases}) == len(closed_cases), "distinct evaluation case ids required")
    _require(all(c["opponent_id"] in opponent_ids for c in closed_cases), "unknown evaluation opponent")

    train_worlds = {(g["world_sha256"], g["seat"], g["opponent_id"]) for g in closed_groups}
    eval_worlds = {(c["world_sha256"], c["seat"], c["opponent_id"]) for c in closed_cases}
    _require(train_worlds.isdisjoint(eval_worlds), "training and later evaluation worlds must be disjoint")

    seed = value["seed"]
    _require(type(seed) is int and 0 <= seed < 2**32, "pilot seed must be uint32")
    anchor = _anchor_spec(value["bc_anchor"])
    return {"schema": REGISTRATION_SCHEMA, "behavior_checkpoint": behavior,
            "opponents": closed_opponents, "train_groups": closed_groups,
            "evaluation_cases": closed_cases, "seed": seed, "bc_anchor": anchor}


def build_plan(registration: dict, parent: dict, *, registration_sha256: str) -> dict:
    registration = validate_registration(registration, parent)
    _sha256(registration_sha256, label="registration")
    arms = [{"id": ARM_PG, "parent": BASELINE_ARM, "bc_anchor": None}]
    if registration["bc_anchor"] is not None:
        arms.append({"id": ARM_ANCHOR, "parent": BASELINE_ARM,
                     "bc_anchor": copy.deepcopy(registration["bc_anchor"])})

    collection = []
    for group in registration["train_groups"]:
        for sample_seed in group["sample_seeds"]:
            collection.append({"job_id": f"collect:{group['group_id']}:{sample_seed}",
                "group_id": group["group_id"], "world_id": group["world_id"],
                "world_sha256": group["world_sha256"], "seat": group["seat"],
                "opponent_id": group["opponent_id"], "sampler_seed": sample_seed,
                "arm": BASELINE_ARM, "mode": "sample_raw", "complete_game_required": True})
    evaluation = []
    for case in registration["evaluation_cases"]:
        for arm in [BASELINE_ARM, *[a["id"] for a in arms]]:
            evaluation.append({"job_id": f"eval:{case['case_id']}:{arm}", "case_id": case["case_id"],
                "world_id": case["world_id"], "world_sha256": case["world_sha256"], "seat": case["seat"],
                "opponent_id": case["opponent_id"], "sampler_seed": case["sampler_seed"],
                "arm": arm, "mode": "sample_raw", "complete_game_required": True})
    planned = len(collection) + len(evaluation)
    _require(planned <= MAX_PLANNED_NATIVE_GAMES, "predeclared pilot exceeds bounded native-game registration")

    mixture = {o["id"]: sum(len(g["sample_seeds"]) for g in registration["train_groups"]
                             if g["opponent_id"] == o["id"]) for o in registration["opponents"]}
    body = {
        "schema": PLAN_SCHEMA,
        "scope": "IMPLEMENTED_HARNESS_AND_FUTURE_MAINTAINER_PILOT;NO_STRENGTH_RESULT_YET",
        "registration_sha256": registration_sha256,
        "protocol": copy.deepcopy(PROTOCOL),
        "seed": registration["seed"],
        "behavior_checkpoint": copy.deepcopy(registration["behavior_checkpoint"]),
        "behavior_checkpoint_state_sha256": parent["state_sha256"],
        "behavior_model_identity": copy.deepcopy(parent["model_identity"]),
        "opponents": copy.deepcopy(registration["opponents"]),
        "opponent_game_counts": mixture,
        "train_groups": copy.deepcopy(registration["train_groups"]),
        "evaluation_cases": copy.deepcopy(registration["evaluation_cases"]),
        "arms": arms,
        "collection_jobs": collection,
        "evaluation_jobs": evaluation,
        "planned_native_games": planned,
        "source_pins": source_pins(),
        "scorer_source_pins": A.source_pins(),
        "strength_claim": False,
    }
    return _bind(body)


def load_plan(path: str | Path, expected_sha256: str) -> dict:
    value = _pinned_json(path, expected_sha256)
    _verify_bound(value, PLAN_SCHEMA)
    _require(value.get("protocol") == PROTOCOL, "pilot protocol drift")
    _require(value.get("source_pins") == source_pins(), "V0 RL source closure changed since planning")
    _require(value.get("scorer_source_pins") == A.source_pins(), "V0 scorer source closure changed since planning")
    return value


# ---------------------------------------------------------------------------
# rollout admission and exact group advantages

@dataclass(frozen=True)
class Rollout:
    game_id: str
    group_id: str
    sampler_seed: int
    actor: dict
    trace: dict
    old_logp: float
    reward: float
    advantage: float
    world_id: str
    world_sha256: str
    seat: int
    opponent_id: str


@dataclass(frozen=True)
class RolloutDataset:
    plan_sha256: str
    rollout_sha256: str
    groups: tuple[tuple[Rollout, ...], ...]
    binding: dict
    diagnostics: dict

    @property
    def games(self) -> tuple[Rollout, ...]:
        return tuple(game for group in self.groups for game in group)


def terminal_wdl(own_cash: Any, rival_cash: Any) -> tuple[str, float]:
    _require(type(own_cash) in (int, float) and type(rival_cash) in (int, float)
             and math.isfinite(own_cash) and math.isfinite(rival_cash), "finite terminal cash required")
    if own_cash > rival_cash:
        return "WIN", 1.0
    if own_cash < rival_cash:
        return "LOSS", -1.0
    return "DRAW", 0.0


def _trace_old_logp(trace: dict, expected_model: dict) -> float:
    _require(isinstance(trace, dict) and trace.get("schema") == A.TRACE_SCHEMA, "native V0 decode trace required")
    _require(trace.get("model_identity") == expected_model, "rollout model identity differs from frozen BC parent")
    _require(trace.get("mode") == PROTOCOL["behavior_decode_mode"]
             and trace.get("temperature") == PROTOCOL["temperature"], "pilot requires direct sample_raw temperature-1 traces")
    _require(trace.get("initial_prefix") == A.Prefix(Plan()).payload(), "pilot trace must start from empty opening Plan")
    steps = trace.get("steps")
    _require(isinstance(steps, list) and bool(steps), "nonempty STOP-terminated trace required")
    raw_sum = 0.0
    actions = []
    for i, step in enumerate(steps):
        _require(isinstance(step, dict) and step.get("schema") == A.STEP_SCHEMA
                 and step.get("mode") == "sample_raw" and step.get("temperature") == 1.0,
                 f"prefix {i}: sample_raw step required")
        raw, behavior = step.get("raw_logp"), step.get("behavior_logp")
        _require(type(raw) in (int, float) and type(behavior) in (int, float)
                 and math.isfinite(raw) and math.isfinite(behavior) and raw <= 1e-12 and behavior <= 1e-12,
                 f"prefix {i}: finite nonpositive raw behavior logp required")
        _require(math.isclose(raw, behavior, rel_tol=2e-6, abs_tol=2e-6),
                 f"prefix {i}: behavior likelihood is not the sampled raw-ID likelihood")
        if step.get("forced_stop"):
            _require(abs(raw) <= 1e-12 and abs(behavior) <= 1e-12,
                     "forced administrative STOP must have log probability zero")
        raw_sum += float(raw)
        actions.append(step.get("raw_id"))
    _require(actions == trace.get("actions") and actions[-1] == V.STOP_ID,
             "trace actions/STOP do not match step records")
    saved = trace.get("raw_path_logp")
    _require(type(saved) in (int, float) and math.isfinite(saved)
             and math.isclose(float(saved), raw_sum, rel_tol=2e-6, abs_tol=3e-5),
             "raw path logp does not equal the sampled step likelihoods")
    _require(trace.get("class_path_not_final_plan_marginal") is True,
             "trace must retain the V0 class-path boundary declaration")
    return float(saved)


def _expected_collection(plan: dict) -> dict[tuple[str, int], dict]:
    return {(job["group_id"], job["sampler_seed"]): job for job in plan["collection_jobs"]}


def _validate_game(row: dict, job: dict, plan: dict) -> tuple[dict | None, str | None]:
    """Return normalized complete row or an explicit non-learning reason."""
    try:
        expected = {"game_id", "group_id", "world_id", "world_sha256", "seat", "opponent_id", "sampler_seed",
                    "status", "checks_passed", "censored", "physical_steps", "own_final_cash", "rival_final_cash",
                    "wdl", "actor", "initial_plan", "decode_trace", "behavior_checkpoint_sha256"}
        _require(isinstance(row, dict) and set(row) == expected, "closed rollout row required")
        for key in ("group_id", "world_id", "world_sha256", "seat", "opponent_id", "sampler_seed"):
            _require(row[key] == job[key], "collection registration mismatch: " + key)
        _require(row["behavior_checkpoint_sha256"] == plan["behavior_checkpoint"]["commit_sha256"],
                 "behavior checkpoint mismatch")
        if row["status"] != "COMPLETE" or row["checks_passed"] is not True or row["censored"] is not False:
            return None, "INVALID_OR_CENSORED_COMPLETE_GAME_GATE"
        if row["physical_steps"] != V.LAST_STEP + 1:
            return None, "INCOMPLETE_PHYSICAL_GAME"
        wdl, reward = terminal_wdl(row["own_final_cash"], row["rival_final_cash"])
        if row["wdl"] != wdl:
            return None, "TERMINAL_WDL_CASH_MISMATCH"
        actor = row["actor"]
        # Actor reconstruction enforces the opening-only allowlist and rejects trainer-only/future/rival-private additions.
        state = BC.actor_state(actor)
        if state.step != 0:
            return None, "NON_OPENING_ACTOR_STATE"
        _require(Plan.from_canonical(row["initial_plan"]) == Plan(), "pilot requires empty opening Plan")
        old = _trace_old_logp(row["decode_trace"], plan["behavior_model_identity"])
        # Header state/projection binding is model-independent and catches a trace copied to another opening.
        _require(row["decode_trace"].get("state_sha256") == A._state_digest(state), "trace state digest mismatch")
        _require(row["decode_trace"].get("projection_sha256") == A._digest(A._projection_options(T.SETTINGS)),
                 "trace projection options differ from frozen opening pilot")
        return {**row, "reward": reward, "old_logp": old}, None
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return None, f"MALFORMED_OR_UNREPLAYABLE:{type(exc).__name__}:{exc}"


def validate_rollout_document(document: dict, *, plan: dict, plan_sha256: str, rollout_sha256: str) -> RolloutDataset:
    _verify_bound(plan, PLAN_SCHEMA)
    _sha256(plan_sha256, label="plan")
    _sha256(rollout_sha256, label="rollouts")
    _require(isinstance(document, dict)
             and set(document) == {"schema", "plan_sha256", "behavior_checkpoint_sha256", "scorer_source_pins", "games"}
             and document.get("schema") == ROLLOUT_SCHEMA,
             "closed V0 RL rollout document required")
    _require(document["plan_sha256"] == plan_sha256, "rollouts bind a different plan file")
    _require(document["behavior_checkpoint_sha256"] == plan["behavior_checkpoint"]["commit_sha256"],
             "rollouts bind a different behavior checkpoint")
    _require(document["scorer_source_pins"] == plan["scorer_source_pins"], "collection scorer sources changed")
    rows = document["games"]
    _require(isinstance(rows, list), "games list required")
    expected = _expected_collection(plan)
    by_key: dict[tuple[str, int], dict] = {}
    duplicate = []
    unregistered = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("rollout rows must be objects")
        key = (row.get("group_id"), row.get("sampler_seed"))
        if key not in expected:
            unregistered.append(key)
            continue
        if key in by_key:
            duplicate.append(key)
        else:
            by_key[key] = row
    _require(not unregistered, "unregistered rollout rows: " + repr(unregistered[:4]))
    _require(not duplicate, "duplicate rollout rows: " + repr(duplicate[:4]))

    group_specs = {g["group_id"]: g for g in plan["train_groups"]}
    admitted_groups: list[tuple[Rollout, ...]] = []
    excluded: dict[str, list[str]] = {}
    invalid_games = []
    group_diagnostics = []
    all_rewards = []
    nonzero_advantages = 0
    for group_id, spec in group_specs.items():
        normalized = []
        reasons = []
        for sample_seed in spec["sample_seeds"]:
            key = (group_id, sample_seed)
            if key not in by_key:
                reasons.append(f"MISSING_REGISTERED_GAME:{sample_seed}")
                continue
            clean, reason = _validate_game(by_key[key], expected[key], plan)
            if clean is None:
                reasons.append(f"{sample_seed}:{reason}")
                invalid_games.append({"group_id": group_id, "sampler_seed": sample_seed, "reason": reason})
            else:
                normalized.append(clean)
        # One censored/invalid/missing member excludes the WHOLE registered group.  We do not silently
        # shrink K and then recompute a different group baseline.
        if reasons or len(normalized) != len(spec["sample_seeds"]):
            excluded[group_id] = reasons or ["REGISTERED_GROUP_CARDINALITY_CHANGED"]
            continue
        actor_hashes = {digest(r["actor"]) for r in normalized}
        if len(actor_hashes) != 1:
            excluded[group_id] = ["GROUP_ACTOR_VISIBLE_OPENING_MISMATCH"]
            continue
        rewards = [r["reward"] for r in normalized]
        mean = sum(rewards) / len(rewards)
        advantages = [float(r - mean) for r in rewards]
        # Algebraic ties become literal zeros instead of epsilon/noise-created signal.
        if max(rewards) == min(rewards):
            advantages = [0.0] * len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        group_rows = tuple(Rollout(
            game_id=_nonempty(r["game_id"], "game_id"), group_id=group_id, sampler_seed=r["sampler_seed"],
            actor=copy.deepcopy(r["actor"]), trace=copy.deepcopy(r["decode_trace"]), old_logp=r["old_logp"],
            reward=r["reward"], advantage=a, world_id=r["world_id"], world_sha256=r["world_sha256"],
            seat=r["seat"], opponent_id=r["opponent_id"]
        ) for r, a in zip(normalized, advantages))
        admitted_groups.append(group_rows)
        all_rewards.extend(rewards)
        nonzero_advantages += sum(a != 0.0 for a in advantages)
        group_diagnostics.append({"group_id": group_id, "denominator": len(group_rows),
                                  "wdl_returns": rewards, "mean": mean, "variance": variance,
                                  "nonzero_advantages": sum(a != 0.0 for a in advantages)})

    _require(bool(admitted_groups), "no complete registered rollout group is eligible for learning")
    return_mean = sum(all_rewards) / len(all_rewards)
    return_variance = sum((r - return_mean) ** 2 for r in all_rewards) / len(all_rewards)
    diagnostics = {
        "registered_groups": len(group_specs), "admitted_groups": len(admitted_groups),
        "excluded_groups": len(excluded), "registered_games": len(expected),
        "rows_present": len(rows), "admitted_games": len(all_rewards), "invalid_games": invalid_games,
        "group_exclusions": excluded, "groups": group_diagnostics,
        "return_mean": return_mean, "return_variance": return_variance,
        "nonzero_advantages": nonzero_advantages,
        "learning_signal": ("STOP_ZERO_RETURN_VARIANCE" if return_variance == 0.0 else
                            "STOP_NO_WITHIN_GROUP_ADVANTAGE" if nonzero_advantages == 0 else "READY"),
        "denominator": len(all_rewards),
        "denominator_semantics": PROTOCOL["gradient_denominator"],
    }
    binding = {"schema": ROLLOUT_SCHEMA + "-binding", "plan_sha256": plan_sha256,
               "rollout_sha256": rollout_sha256, "behavior_checkpoint_sha256": document["behavior_checkpoint_sha256"],
               "admitted_group_ids": [g[0].group_id for g in admitted_groups],
               "admitted_game_ids": [r.game_id for g in admitted_groups for r in g],
               "group_denominators": {g[0].group_id: len(g) for g in admitted_groups},
               "rejected_group_ids": sorted(excluded), "source_pins": source_pins()}
    return RolloutDataset(plan_sha256, rollout_sha256, tuple(admitted_groups), binding, diagnostics)


def load_rollouts(path: str | Path, expected_sha256: str, *, plan: dict, plan_sha256: str) -> RolloutDataset:
    document = _pinned_json(path, expected_sha256)
    return validate_rollout_document(document, plan=plan, plan_sha256=plan_sha256,
                                     rollout_sha256=expected_sha256)


# ---------------------------------------------------------------------------
# exact replay / PG math

def replay_raw_logp(model, rollout: Rollout, *, verify_behavior: bool) -> torch.Tensor:
    """Replay the actual sampled raw trace on freshly rebuilt prefix-local menus.

    ``A.replay_epoch`` also returns a successor-class path logp; intentionally do
    not use it here.  The returned tensor is ``EpochResult.raw_logp`` only.
    """
    state = BC.actor_state(copy.deepcopy(rollout.actor))
    result = A.replay_epoch(model, state, Plan(), copy.deepcopy(rollout.trace),
                            projection_kwargs=copy.deepcopy(T.SETTINGS),
                            verify_logps=verify_behavior, backend=PROTOCOL["menu_backend"])
    _require(tuple(result.actions) == tuple(rollout.trace["actions"]), "replay action sequence changed")
    if verify_behavior:
        _require(math.isclose(float(result.raw_logp.detach()), rollout.old_logp, rel_tol=2e-6, abs_tol=3e-5),
                 "frozen behavior model does not reproduce stored raw-path likelihood")
    return result.raw_logp


def importance_ratio(current_logp: torch.Tensor, old_logp: float) -> torch.Tensor:
    _require(isinstance(current_logp, torch.Tensor) and current_logp.numel() == 1,
             "scalar current raw-path logp required")
    _require(type(old_logp) in (int, float) and math.isfinite(old_logp), "finite frozen behavior logp required")
    ratio = torch.exp(current_logp.double() - float(old_logp))
    _require(bool(torch.isfinite(ratio)), "nonfinite policy importance ratio")
    return ratio


def clipped_pg_loss(current_logp: torch.Tensor, old_logp: float, advantage: float,
                    *, clip_ratio: float = PROTOCOL["clip_ratio"]) -> tuple[torch.Tensor, torch.Tensor]:
    _require(type(advantage) in (int, float) and math.isfinite(advantage), "finite detached advantage required")
    _require(type(clip_ratio) in (int, float) and 0.0 < clip_ratio < 1.0, "valid clip ratio required")
    ratio = importance_ratio(current_logp, old_logp)
    a = ratio.new_tensor(float(advantage)).detach()
    unclipped = ratio * a
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * a
    return -torch.minimum(unclipped, clipped), ratio


def behavior_parity(model, data: RolloutDataset) -> dict:
    was_training = bool(model.training)
    model.eval()
    errors = []
    with torch.no_grad():
        for rollout in data.games:
            got = replay_raw_logp(model, rollout, verify_behavior=True)
            errors.append(abs(float(got) - rollout.old_logp))
    model.train(was_training)
    return {"games": len(errors), "max_abs_raw_logp_error": max(errors, default=0.0),
            "likelihood": PROTOCOL["likelihood"]}


def ratio_diagnostics(model, data: RolloutDataset) -> dict:
    was_training = bool(model.training)
    model.eval()
    ratios, deltas = [], []
    with torch.no_grad():
        for rollout in data.games:
            current = replay_raw_logp(model, rollout, verify_behavior=False)
            ratio = importance_ratio(current, rollout.old_logp)
            ratios.append(float(ratio))
            deltas.append(float(current) - rollout.old_logp)
    model.train(was_training)
    clipped = sum(r < 1.0-PROTOCOL["clip_ratio"] or r > 1.0+PROTOCOL["clip_ratio"] for r in ratios)
    return {"count": len(ratios), "ratio_min": min(ratios), "ratio_max": max(ratios),
            "ratio_mean": sum(ratios)/len(ratios), "clipped_fraction": clipped/len(ratios),
            "sample_mean_old_minus_new_logp": -sum(deltas)/len(deltas)}


def _anchor_loss(model: A.V0Scorer, data: BC.Dataset) -> torch.Tensor:
    losses = []
    for trajectory in data.trajectories:
        terms = []
        for example in trajectory.examples:
            logits = model.score(example.state, example.menu.prefix, example.menu.legal_ids,
                                 example.menu.phi, memory={})
            terms.append(A.class_nll(logits, example.target_indices))
        _require(bool(terms), "BC anchor contains empty trajectory")
        losses.append(torch.stack(terms).sum())
    _require(bool(losses) and len(losses) <= MAX_BC_ANCHOR_TRAJECTORIES,
             "BC anchor must contain 1..64 complete trajectories")
    return torch.stack(losses).mean()


def _signal_decision(data: RolloutDataset) -> dict:
    status = data.diagnostics["learning_signal"]
    return {"status": status,
            "action": ("RUN_PREDECLARED_PG" if status == "READY" else
                       "STOP_AND_DIAGNOSE_WITHOUT_REWARD_CHANGE"),
            "return_variance": data.diagnostics["return_variance"],
            "nonzero_advantages": data.diagnostics["nonzero_advantages"]}


# ---------------------------------------------------------------------------
# source-bound trainer and immutable continuation

def _load_bc_parent_model(directory: str | Path, expected_commit_sha256: str, *, device: torch.device):
    parent = inspect_bc_parent(directory, expected_commit_sha256)
    commit = _pinned_json(Path(directory) / "COMMIT.json", expected_commit_sha256)
    state = torch.load(Path(directory) / "state.pt", map_location="cpu", weights_only=True)
    _require(isinstance(state, dict) and state.get("schema") == BC.SCHEMA
             and state.get("identity") == commit["identity"]
             and isinstance(state.get("model"), dict), "complete BC parent state required")
    identity = parent["model_identity"]
    if not identity["diagnostic"]:
        BC.require_production_runtime()
    model = A.V0Scorer(Config(**identity["config"]), diagnostic=identity["diagnostic"]).to(device)
    _require(model.get_extra_state() == identity, "BC parent model identity cannot be reconstructed")
    model.load_state_dict(state["model"], strict=True)
    return model, parent


def _rng_state(device: torch.device) -> dict:
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}


def _restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        _require(len(state["torch_cuda"]) == torch.cuda.device_count(), "device RNG topology mismatch")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    else:
        _require(state["torch_cuda"] == [], "CPU continuation cannot import CUDA RNG")


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def _finite_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(v) for v in value)
    if type(value) is float:
        return math.isfinite(value)
    return True


def _same_tree(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_tree(x, y) for x, y in zip(a, b))
    return a == b


def _runtime_identity(device: torch.device) -> dict:
    # Same numerical surface as BC; no autocast/TF32 is introduced by this pilot.
    return BC.runtime_identity(device)


def _arm(plan: dict, arm_id: str) -> dict:
    matches = [a for a in plan["arms"] if a["id"] == arm_id]
    _require(len(matches) == 1 and arm_id in (ARM_PG, ARM_ANCHOR), "declared child arm required")
    return matches[0]


def load_anchor(plan: dict, arm_id: str) -> BC.Dataset | None:
    arm = _arm(plan, arm_id)
    spec = arm["bc_anchor"]
    if spec is None:
        return None
    path = ROOT / spec["dataset_path"]
    data = BC.load_dataset(path, spec["manifest_sha256"], split=spec["split"], backend=PROTOCOL["menu_backend"])
    _require(len(data.trajectories) <= MAX_BC_ANCHOR_TRAJECTORIES, "BC anchor exceeds bounded trajectory cap")
    return data


class Trainer:
    """Exact fixed-rollout PPO pilot initialized from one frozen committed V0 BC parent."""
    def __init__(self, plan: dict, plan_sha256: str, data: RolloutDataset, arm_id: str, *, device: str = "cpu",
                 anchor_data: BC.Dataset | None = None):
        _verify_bound(plan, PLAN_SCHEMA)
        _require(data.plan_sha256 == plan_sha256, "trainer data/plan mismatch")
        arm = _arm(plan, arm_id)
        _require((arm["bc_anchor"] is None) == (anchor_data is None), "BC anchor data must match the declared arm")
        if anchor_data is not None:
            _require(anchor_data.manifest_sha256 == arm["bc_anchor"]["manifest_sha256"]
                     and anchor_data.split == arm["bc_anchor"]["split"], "BC anchor binding mismatch")
        self.device = torch.device(device)
        _require(self.device.type in ("cpu", "cuda"), "CPU or CUDA device required")
        if self.device.type == "cuda":
            _require(torch.cuda.is_available(), "CUDA requested but unavailable")
            self.device = torch.device("cuda", torch.cuda.current_device() if self.device.index is None else self.device.index)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        random.seed(plan["seed"]); np.random.seed(plan["seed"]); torch.manual_seed(plan["seed"])
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(plan["seed"])
        parent_dir = ROOT / plan["behavior_checkpoint"]["path"]
        self.model, parent = _load_bc_parent_model(parent_dir, plan["behavior_checkpoint"]["commit_sha256"], device=self.device)
        _require(parent["model_identity"] == plan["behavior_model_identity"], "planned behavior model changed")
        self.data, self.plan, self.plan_sha256 = data, copy.deepcopy(plan), plan_sha256
        self.arm, self.anchor_data = copy.deepcopy(arm), anchor_data
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=PROTOCOL["learning_rate"],
                                          betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False)
        self.parameter_order = [name for name, _ in self.model.named_parameters()]
        self.model.eval()
        self.behavior_parity = behavior_parity(self.model, data)
        self.model.train()
        self.signal = _signal_decision(data)
        self.identity = {
            "schema": SCHEMA, "protocol": copy.deepcopy(PROTOCOL), "plan_sha256": plan_sha256,
            "rollouts": copy.deepcopy(data.binding), "arm": copy.deepcopy(arm),
            "behavior_parent": {"commit_sha256": parent["commit_sha256"], "state_sha256": parent["state_sha256"]},
            "model_identity": copy.deepcopy(parent["model_identity"]), "source_pins": source_pins(),
            "runtime": _runtime_identity(self.device),
            "optimizer": PROTOCOL["optimizer"],
            "anchor": None if anchor_data is None else {"manifest_sha256": anchor_data.manifest_sha256,
                                                         "split": anchor_data.split,
                                                         "trajectories": len(anchor_data.trajectories)},
        }
        self.global_update = 0
        self.parent_sha256: str | None = None
        self.parent_directory: Path | None = None
        self._failed = False

    def step(self) -> dict:
        _require(not self._failed, "failed trainer must resume its last committed parent")
        _require(self.signal["status"] == "READY",
                 self.signal["status"] + ": stop and diagnose; reward/objective changes require a new reviewed plan")
        _require(self.global_update < PROTOCOL["optimizer_updates"], "pilot optimizer update cap reached")
        try:
            self.model.train(); self.optimizer.zero_grad(set_to_none=True)
            n = len(self.data.games)
            pg_total = 0.0
            ratios = []
            for rollout in self.data.games:
                current = replay_raw_logp(self.model, rollout, verify_behavior=False)
                loss, ratio = clipped_pg_loss(current, rollout.old_logp, rollout.advantage)
                (loss / n).backward()
                pg_total += float(loss.detach()) / n
                ratios.append(float(ratio.detach()))
            anchor_value = 0.0
            if self.anchor_data is not None:
                anchor = _anchor_loss(self.model, self.anchor_data)
                coefficient = self.arm["bc_anchor"]["coefficient"]
                (coefficient * anchor).backward()
                anchor_value = float(anchor.detach())
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), PROTOCOL["max_grad_norm"],
                                                  error_if_nonfinite=True, foreach=False)
            self.optimizer.step()
            _require(_finite_tree(self.model.state_dict()) and _finite_tree(self.optimizer.state_dict()),
                     "nonfinite RL optimizer update")
            self.global_update += 1
            after = ratio_diagnostics(self.model, self.data)
            return {"global_update": self.global_update, "complete_game_denominator": n,
                    "pg_loss": pg_total, "bc_anchor_loss": anchor_value,
                    "bc_anchor_coefficient": 0.0 if self.anchor_data is None else self.arm["bc_anchor"]["coefficient"],
                    "grad_norm": float(norm), "pre_step_ratio_min": min(ratios), "pre_step_ratio_max": max(ratios),
                    "post_step_ratios": after, "return_variance": self.data.diagnostics["return_variance"],
                    "nonzero_advantages": self.data.diagnostics["nonzero_advantages"]}
        except BaseException:
            self._failed = True
            raise

    def snapshot(self) -> dict:
        _require(not self._failed, "failed update is not a continuation")
        return _cpu_tree({"schema": SCHEMA, "identity": self.identity, "parameter_order": self.parameter_order,
                          "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                          "optimizer_state_ids": sorted(self.optimizer.state_dict()["state"]),
                          "rng": _rng_state(self.device), "global_update": self.global_update,
                          "model_training": self.model.training})

    def _validate_snapshot(self, value: dict) -> None:
        _require(isinstance(value, dict) and set(value) == {"schema", "identity", "parameter_order", "model",
                 "optimizer", "optimizer_state_ids", "rng", "global_update", "model_training"},
                 "complete V0 RL continuation fields required")
        _require(value["schema"] == SCHEMA and value["identity"] == self.identity, "RL continuation identity mismatch")
        _require(value["parameter_order"] == self.parameter_order, "RL optimizer parameter order mismatch")
        age = value["global_update"]
        _require(type(age) is int and 0 <= age <= PROTOCOL["optimizer_updates"] and value["model_training"] is True,
                 "invalid RL update age/mode")
        expected = self.optimizer.state_dict()
        _require(set(value["optimizer"]) == set(expected)
                 and value["optimizer"]["param_groups"] == expected["param_groups"],
                 "RL optimizer settings/parameter ids mismatch")
        active = value["optimizer_state_ids"]
        _require(isinstance(active, list) and all(type(i) is int for i in active)
                 and active == sorted(value["optimizer"]["state"])
                 and (bool(active) == (age > 0)), "RL optimizer state inventory/age mismatch")
        params = list(self.model.parameters())
        for key, state in value["optimizer"]["state"].items():
            _require(type(key) is int and 0 <= key < len(params)
                     and set(state) == {"step", "exp_avg", "exp_avg_sq"}, "unexpected Adam state")
            _require(isinstance(state["step"], torch.Tensor) and state["step"].numel() == 1
                     and float(state["step"]).is_integer() and 0 < float(state["step"]) <= age,
                     "Adam step/global RL age mismatch")
            for name in ("exp_avg", "exp_avg_sq"):
                _require(isinstance(state[name], torch.Tensor) and state[name].shape == params[key].shape
                         and state[name].dtype == params[key].dtype, "Adam moment shape/dtype mismatch")
        _require(_finite_tree(value), "nonfinite RL continuation")
        _require(set(value["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}, "complete RNG state required")
        before = _rng_state(self.device)
        try:
            _restore_rng(value["rng"], self.device)
        finally:
            _restore_rng(before, self.device)

    def save(self, directory: str | Path, *, metrics: dict | None = None) -> dict:
        _require(self.identity["source_pins"] == source_pins(), "live V0 RL sources changed during training")
        _require(self.identity["runtime"] == _runtime_identity(self.device), "live RL runtime changed")
        value = self.snapshot(); self._validate_snapshot(value)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        marker = directory / "COMMIT.json"
        try:
            state_path = directory / "state.pt"
            with state_path.open("xb") as stream:
                torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
            restored = torch.load(state_path, map_location="cpu", weights_only=True)
            _require(_same_tree(value, restored), "RL checkpoint round-trip mismatch")
            self._validate_snapshot(restored)
            commit = {"schema": COMMIT_SCHEMA, "status": "COMMITTED", "identity": self.identity,
                      "state_sha256": file_sha(state_path), "state_bytes": state_path.stat().st_size,
                      "global_update": self.global_update, "parent_manifest_sha256": self.parent_sha256,
                      "metrics": metrics, "scope": ("DIAGNOSTIC_CPU_NOT_STRENGTH" if self.model.diagnostic
                                                     else "V0_TERMINAL_WDL_PILOT_NOT_STRENGTH")}
            temporary = directory / ".COMMIT.pending"
            _write_json_new(temporary, commit)
            _fsync_dir(directory)
            os.link(temporary, marker); _fsync_dir(directory)
            temporary.unlink(); _fsync_dir(directory.parent)
            self.parent_sha256 = file_sha(marker); self.parent_directory = directory.resolve()
            return commit
        except BaseException:
            if marker.exists():
                marker.unlink()
            self._failed = True
            raise

    @classmethod
    def resume(cls, plan: dict, plan_sha256: str, data: RolloutDataset, arm_id: str,
               directory: str | Path, expected_sha256: str, *, device: str = "cpu",
               anchor_data: BC.Dataset | None = None) -> "Trainer":
        directory = Path(directory)
        commit = _pinned_json(directory / "COMMIT.json", expected_sha256)
        _require(commit.get("schema") == COMMIT_SCHEMA and commit.get("status") == "COMMITTED",
                 "committed V0 RL parent required")
        state_path = directory / "state.pt"
        _require(state_path.stat().st_size == commit.get("state_bytes")
                 and file_sha(state_path) == commit.get("state_sha256"), "RL parent payload hash/size mismatch")
        trainer = cls(plan, plan_sha256, data, arm_id, device=device, anchor_data=anchor_data)
        _require(commit["identity"] == trainer.identity, "RL parent source/rollout/arm/runtime mismatch")
        value = torch.load(state_path, map_location="cpu", weights_only=True)
        trainer._validate_snapshot(value)
        _require(value["global_update"] == commit["global_update"], "RL commit age mismatch")
        trainer.model.load_state_dict(value["model"], strict=True)
        trainer.optimizer.load_state_dict(value["optimizer"])
        trainer.global_update = value["global_update"]
        trainer.parent_sha256 = expected_sha256; trainer.parent_directory = directory.resolve()
        trainer.model.train(value["model_training"])
        _restore_rng(value["rng"], trainer.device)
        return trainer


def run_segment(trainer: Trainer, output: str | Path, *, updates: int) -> dict:
    _require(type(updates) is int and 1 <= updates <= PROTOCOL["optimizer_updates"] - trainer.global_update,
             "segment exceeds remaining predeclared RL update budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json_new(output / "RUN.json", {"schema": SCHEMA + "-run-v1", "identity": trainer.identity,
                                           "updates": updates, "signal": trainer.signal,
                                           "parent_manifest_sha256": trainer.parent_sha256})
    if trainer.signal["status"] != "READY":
        result = {"schema": RESULT_SCHEMA, "status": trainer.signal["status"], "decision": trainer.signal,
                  "optimizer_updates": 0, "strength_claim": False,
                  "rule": "stop and diagnose; do not alter reward/objective without a new reviewed plan"}
        _write_json_new(output / "RESULT.json", result)
        return result
    start = trainer.global_update
    last = trainer.parent_directory
    history = []
    try:
        if start == 0:
            last = output / "step-00000000"
            trainer.save(last, metrics={"behavior_parity": trainer.behavior_parity, "signal": trainer.signal})
        for _ in range(updates):
            metric = trainer.step(); history.append(metric)
            candidate = output / f"step-{trainer.global_update:08d}"
            trainer.save(candidate, metrics=metric); last = candidate
        result = {"schema": RESULT_SCHEMA, "status": "COMMITTED_NOT_STRENGTH",
                  "arm": trainer.arm["id"], "global_update": trainer.global_update,
                  "optimizer_updates": trainer.global_update-start, "last_checkpoint": str(last),
                  "commit_sha256": trainer.parent_sha256, "behavior_parity": trainer.behavior_parity,
                  "signal": trainer.signal, "history": history,
                  "complete_game_wdl_measurement_run": False, "strength_claim": False}
        _write_json_new(output / "RESULT.json", result)
        return result
    except BaseException as exc:
        _write_json_new(output / "FAILURE.json", {"schema": RESULT_SCHEMA, "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}", "last_checkpoint": None if last is None else str(last),
            "last_commit_sha256": trainer.parent_sha256, "attempted_updates": len(history)+1,
            "committed_global_update": trainer.global_update, "history": history,
            "rule": "resume only the last immutable COMMIT; never adopt a partial update"})
        raise


# ---------------------------------------------------------------------------
# later complete-game WDL measurement diagnostics (no ranking/promotion)

def evaluation_diagnostics(document: dict, *, plan: dict, plan_sha256: str, results_sha256: str) -> dict:
    _sha256(results_sha256, label="evaluation results")
    _require(isinstance(document, dict) and set(document) == {"schema", "plan_sha256", "games"}
             and document.get("schema") == EVAL_SCHEMA and document.get("plan_sha256") == plan_sha256,
             "closed evaluation result document for this plan required")
    expected = {job["job_id"]: job for job in plan["evaluation_jobs"]}
    rows = document["games"]
    _require(isinstance(rows, list), "evaluation games list required")
    seen, invalid = {}, []
    for row in rows:
        _require(isinstance(row, dict)
                 and set(row) == {"job_id", "status", "checks_passed", "censored", "physical_steps",
                                  "own_final_cash", "rival_final_cash", "wdl"},
                 "closed evaluation row required")
        job_id = row["job_id"]
        _require(job_id in expected and job_id not in seen, "unknown/duplicate evaluation job")
        seen[job_id] = row
        if row["status"] != "COMPLETE" or row["checks_passed"] is not True or row["censored"] is not False \
                or row["physical_steps"] != V.LAST_STEP + 1:
            invalid.append({"job_id": job_id, "reason": "INVALID_CENSORED_OR_INCOMPLETE"})
            continue
        wdl, _ = terminal_wdl(row["own_final_cash"], row["rival_final_cash"])
        if row["wdl"] != wdl:
            invalid.append({"job_id": job_id, "reason": "WDL_CASH_MISMATCH"})
    missing = sorted(set(expected) - set(seen))
    if invalid or missing:
        return {"schema": EVAL_SCHEMA + "-diagnostics", "status": "INCOMPLETE_MEASUREMENT",
                "invalid": invalid, "missing": missing, "strength_claim": False,
                "decision": "fix/complete the predeclared jobs; do not impute or retry-select outcomes"}
    score = {"WIN": 1.0, "DRAW": 0.5, "LOSS": 0.0}
    by_arm = {}
    for job_id, row in seen.items():
        arm = expected[job_id]["arm"]
        d = by_arm.setdefault(arm, {"WIN": 0, "DRAW": 0, "LOSS": 0, "scores": []})
        d[row["wdl"]] += 1; d["scores"].append(score[row["wdl"]])
    summaries = {arm: {"games": len(v["scores"]), "win": v["WIN"], "draw": v["DRAW"], "loss": v["LOSS"],
                       "mean_wdl_score": sum(v["scores"])/len(v["scores"])} for arm, v in by_arm.items()}
    pairs = []
    for case in plan["evaluation_cases"]:
        baseline_id = f"eval:{case['case_id']}:{BASELINE_ARM}"
        base = score[seen[baseline_id]["wdl"]]
        for arm in [a["id"] for a in plan["arms"]]:
            child = score[seen[f"eval:{case['case_id']}:{arm}"]["wdl"]]
            pairs.append({"case_id": case["case_id"], "child_arm": arm,
                          "baseline_wdl": seen[baseline_id]["wdl"],
                          "child_wdl": seen[f"eval:{case['case_id']}:{arm}"]["wdl"],
                          "paired_wdl_score_delta": child-base})
    return {"schema": EVAL_SCHEMA + "-diagnostics", "status": "COMPLETE_DESCRIPTIVE_WDL",
            "results_sha256": results_sha256, "arms": summaries, "paired": pairs,
            "measurement": "COMPLETE_GAME_WDL_ONLY", "strength_claim": False,
            "promotion_decision": None}


# ---------------------------------------------------------------------------
# CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--registration-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("validate")
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--plan-sha256", required=True)
    p.add_argument("--rollouts", type=Path, required=True); p.add_argument("--rollouts-sha256", required=True)

    p = sub.add_parser("train")
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--plan-sha256", required=True)
    p.add_argument("--rollouts", type=Path, required=True); p.add_argument("--rollouts-sha256", required=True)
    p.add_argument("--arm", choices=(ARM_PG, ARM_ANCHOR), required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--updates", type=int, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--resume", type=Path); p.add_argument("--resume-sha256")

    p = sub.add_parser("eval-diagnostics")
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--plan-sha256", required=True)
    p.add_argument("--results", type=Path, required=True); p.add_argument("--results-sha256", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            registration = _pinned_json(args.registration, args.registration_sha256)
            behavior = registration.get("behavior_checkpoint", {})
            path = ROOT / _repo_relative(behavior.get("path"), "behavior checkpoint path")
            parent = inspect_bc_parent(path, _sha256(behavior.get("commit_sha256"), label="behavior checkpoint"))
            plan = build_plan(registration, parent, registration_sha256=args.registration_sha256)
            _write_json_new(args.output, plan)
            print(canonical({"status": "PLANNED_NOT_RUN", "path": str(args.output),
                             "sha256": file_sha(args.output), "planned_native_games": plan["planned_native_games"],
                             "strength_claim": False}))
            return 0
        plan = load_plan(args.plan, args.plan_sha256)
        if args.command == "validate":
            data = load_rollouts(args.rollouts, args.rollouts_sha256, plan=plan, plan_sha256=args.plan_sha256)
            print(canonical({"status": "VALIDATED_NOT_TRAINED", "binding": data.binding,
                             "diagnostics": data.diagnostics, "strength_claim": False}))
            return 0
        if args.command == "eval-diagnostics":
            document = _pinned_json(args.results, args.results_sha256)
            print(canonical(evaluation_diagnostics(document, plan=plan, plan_sha256=args.plan_sha256,
                                                   results_sha256=args.results_sha256)))
            return 0
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        _require((args.resume is None) == (args.resume_sha256 is None), "resume path and hash required together")
        # Production CLI is intentionally not a back door for diagnostic CPU training.
        _require(plan["behavior_model_identity"]["diagnostic"] is False,
                 "diagnostic checkpoint refused by maintainer train command")
        data = load_rollouts(args.rollouts, args.rollouts_sha256, plan=plan, plan_sha256=args.plan_sha256)
        anchor = load_anchor(plan, args.arm)
        trainer = (Trainer(plan, args.plan_sha256, data, args.arm, device=args.device, anchor_data=anchor)
                   if args.resume is None else
                   Trainer.resume(plan, args.plan_sha256, data, args.arm, args.resume, args.resume_sha256,
                                  device=args.device, anchor_data=anchor))
        result = run_segment(trainer, args.output, updates=args.updates)
        print(canonical(result))
        return 0 if result["status"] in ("COMMITTED_NOT_STRENGTH", "STOP_ZERO_RETURN_VARIANCE",
                                          "STOP_NO_WITHIN_GROUP_ADVANTAGE") else 2
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(canonical({"status": "REFUSED", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
