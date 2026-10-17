"""Explicit, source-versioned 2..16-world full-economic cohort execution.

The intended producer shape is sixteen worlds / thirty-two actor lanes. Each
daily batch uses world-major, seat-minor ordering and its actual longest entity
roster. Sampling remains scalar with independent lane generators. This is NOT a
B2 receipt or admission. Fresh reconstruction reuses the maintained mechanical
checker through private function bindings; no shared module is monkeypatched.

``witness`` is CPU-only and returns freshly reconstructed fitting objects in RAM.
It replays recorded programs, not a newly responsive opponent. ``verify_graph``
requires the caller to load and bind the exact model; it verifies the declared
cohort numerical execution, not a caller-provided "passed" flag. No optimizer,
seed allocation, automatic partial restart or training credit is owned here.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from farm_neat.rollout import rollout as base
from farm_neat.policy import observation as features
from farm_neat.economy.orders import EconomicOrders
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment

SCHEMA = "farm-full-economic-batched-rollout-v1"
PLAN_SCHEMA, WORLD_PLAN_SCHEMA = SCHEMA + "-plan", SCHEMA + "-world-plan"
GAME_SCHEMA, EPISODE_SCHEMA = SCHEMA + "-game", SCHEMA + "-episode"
MODE = "FRESH_SAMPLED_FULL_ECONOMIC_COHORT"
BATCH_MODE = "WORLD_MAJOR_SEAT_MINOR_DYNAMIC_PADDING_SCALAR_LANE_RNG"
STEPS, ATOL = base.STEPS, base.ATOL
bound, digest, file_sha = base.bound, base.digest, base.file_sha
_sealed, _sha = base._sealed, base._sha
_IMPORTED_SHA = file_sha(__file__)


def _equal(actual, expected, label):
    """Physical/structural equality includes nested scalar types, not bool==1."""
    if type(actual) is not type(expected):
        raise ValueError(label + " type mismatch")
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise ValueError(label + " keys mismatch")
        for key in actual:
            _equal(actual[key], expected[key], label + "/" + str(key))
    elif isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError(label + " length mismatch")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _equal(left, right, label + f"/{index}")
    elif actual != expected:
        raise ValueError(label + " mismatch")


def sources():
    # Changing this pin list changes the source identity of every future checkpoint; intended.
    names = ("src/farm_neat/rollout/batched_rollout.py",
             "tests/contracts/test_batched_rollout.py", "src/farm_neat/training/moe_lab.py")
    return {**base.sources(), **{name: file_sha(ROOT / name) for name in names}}


def attest(pins):
    if (file_sha(__file__) != _IMPORTED_SHA or not isinstance(pins, dict)
            or any(pins.get(k) != v for k, v in sources().items())):
        raise ValueError("cohort source closure changed or missing")
    base.attest(pins)


def _registration(seeds, rng, budget, cohort_id, environment):
    if (not isinstance(seeds, list) or not 2 <= len(seeds) <= 16
            or any(type(v) is not int or not 0 <= v <= base.actions.MAX_INTEGER for v in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("2..16 distinct explicit physical world seeds required")
    if (not isinstance(rng, list) or len(rng) != 2 * len(seeds)
            or any(type(v) is not int or not 0 <= v < 2**63 for v in rng)
            or len(set(rng)) != len(rng)):
        raise ValueError("one distinct explicit sampler seed per ordered lane required")
    base.actions._integer(budget, 2, base.actions.MAX_COMMANDS, "command budget")
    if not isinstance(cohort_id, str) or not cohort_id or len(cohort_id) > 200:
        raise ValueError("explicit bounded cohort identity required")
    if (not isinstance(environment, dict) or environment.get("dtype") != "FLOAT32"
            or environment.get("tf32") is not False or environment.get("autocast") is not False):
        raise ValueError("explicit FP32 no-TF32/no-autocast profile required")


def make_plan(manifest, model_sha256, *, seeds, actor_rng_seeds, command_budget,
              cohort_id, environment, source_pins=None):
    """Pure registration: no checkpoint tensor loading, world or GPU execution."""
    policy = base.policy_module()
    _sealed(manifest)
    _equal(manifest["schema"], policy.SCHEMA, "model schema")
    _equal(manifest["architecture"], json.loads(json.dumps(policy.architecture())), "full model architecture")
    _equal(manifest["source_pins"], policy.sources(), "model sources")
    _registration(seeds, actor_rng_seeds, command_budget, cohort_id, environment)
    pins = sources() if source_pins is None else source_pins
    attest(pins)
    plan = bound({"schema": PLAN_SCHEMA, "mode": MODE, "cohort_id": cohort_id,
        "seeds": seeds, "actor_rng_seeds": actor_rng_seeds, "worlds": len(seeds),
        "lanes": 2 * len(seeds), "lane_order": [[w, s] for w in range(len(seeds)) for s in (0, 1)],
        "command_budget": command_budget, "temperature": 1.0,
        "model_sha256": _sha(model_sha256), "model_manifest_sha256": manifest["content_sha256"],
        "model_optimizer_age": manifest["optimizer_steps"], "rollout_iteration": manifest["rollout_iteration"] + 1,
        "architecture_sha256": manifest["architecture"]["content_sha256"], "numerics": policy.NUMERICS,
        "source_pins": pins, "source_sha256": digest(pins), "environment": environment,
        "decision_steps": list(STEPS), "physical_steps_per_world": 719,
        "neural_batch": BATCH_MODE, "cublas_workspace_config": ":4096:8",
        "terminal_training_admission": False})
    validate_plan(plan)
    return plan


def validate_plan(plan):
    _sealed(plan)
    _registration(plan["seeds"], plan["actor_rng_seeds"], plan["command_budget"],
                  plan["cohort_id"], plan["environment"])
    policy = base.policy_module()
    for key, expected in (("schema", PLAN_SCHEMA), ("mode", MODE), ("worlds", len(plan["seeds"])),
            ("lanes", 2 * len(plan["seeds"])), ("neural_batch", BATCH_MODE),
            ("lane_order", [[w, s] for w in range(len(plan["seeds"])) for s in (0, 1)]),
            ("temperature", 1.0), ("decision_steps", list(STEPS)), ("physical_steps_per_world", 719),
            ("cublas_workspace_config", ":4096:8"),
            ("terminal_training_admission", False), ("source_sha256", digest(plan["source_pins"])),
            ("numerics", policy.NUMERICS), ("architecture_sha256", policy.architecture()["content_sha256"])):
        _equal(plan[key], expected, "cohort " + key)
    _sha(plan["model_sha256"]); _sha(plan["model_manifest_sha256"])
    base.actions._integer(plan["model_optimizer_age"], label="model age")
    base.actions._integer(plan["rollout_iteration"], 1, base.actions.MAX_INTEGER, "rollout age")
    attest(plan["source_pins"])
    return plan


def world_plan(plan, index):
    """Derived NEW-schema identity; never an old B2 compatibility receipt."""
    if type(index) is not int or not 0 <= index < plan["worlds"]:
        raise ValueError("registered world index required")
    common = ("command_budget", "temperature", "model_sha256", "model_manifest_sha256",
        "model_optimizer_age", "rollout_iteration", "architecture_sha256", "numerics",
        "source_pins", "source_sha256", "environment", "decision_steps", "neural_batch",
        "terminal_training_admission")
    return bound({**{key: plan[key] for key in common}, "schema": WORLD_PLAN_SCHEMA,
        "mode": MODE, "cohort_plan_sha256": plan["content_sha256"], "world_index": index,
        "game_id": plan["cohort_id"] + f"-world{index:03d}", "seed": plan["seeds"][index],
        "actor_rng_seeds": plan["actor_rng_seeds"][2 * index:2 * index + 2], "physical_steps": 719})


def _bind(function, **changes):
    return FunctionType(function.__code__, {**function.__globals__, **changes},
                        function.__name__, function.__defaults__, function.__closure__)


# Only this module's private callable objects change schema bindings.
capture_decision = _bind(base.capture_decision, SCHEMA=SCHEMA)
capture_turn = _bind(base.capture_turn, SCHEMA=SCHEMA)
finish = _bind(base.finish, GAME_SCHEMA=GAME_SCHEMA)


def batch_inputs(plan, step, packs):
    """Return exact collated arrays plus a content-bound original cohort shape."""
    if step not in STEPS or len(packs) != plan["lanes"]:
        raise ValueError("complete registered daily cohort required")
    batch = features.collate(packs)
    arrays = {}
    for key in ("rows", "token_mask", "knownness"):
        value = np.ascontiguousarray(batch[key])
        arrays[key] = {"dtype": value.dtype.str, "shape": list(value.shape),
                       "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    identity = bound({"schema": SCHEMA + "-batch", "plan_sha256": plan["content_sha256"],
        "step": step, "lane_order": plan["lane_order"], "neural_batch": BATCH_MODE,
        "input_sha256s": list(batch["content_sha256s"]),
        "token_counts": [len(p["rows"]) for p in packs], "arrays": arrays})
    return batch, identity


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON member")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def _read(path):
    return _json(Path(path).read_bytes())


def _read_pinned(path, expected=None):
    data = Path(path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if expected is not None:
        _equal(sha, _sha(expected), "external/payload file identity")
    return _json(data), sha


def _check_numerical_environment(plan, graph=None):
    from farm_neat.training import moe_lab as lab
    if (not sys.dont_write_bytecode or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or (graph is not None and next(graph.parameters()).device.type != "cuda")):
        raise ValueError("actual CUDA cold -B :4096:8 execution required; no CPU fallback")
    _equal(lab.device_profile(), plan["environment"], "actual cohort numerical environment")


def _load_graph(model_path, plan):
    _check_numerical_environment(plan)
    graph, manifest = base.policy_module().load_model(model_path, device="cuda",
        expected_file_sha256=plan["model_sha256"])
    for key, expected in (("content_sha256", plan["model_manifest_sha256"]),
            ("optimizer_steps", plan["model_optimizer_age"]), ("rollout_iteration", plan["rollout_iteration"] - 1)):
        _equal(manifest[key], expected, "actual model " + key)
    graph.eval()
    return graph


def _new_output(output):
    output = Path(output).absolute()
    if output != output.resolve() or not output.is_relative_to(ROOT / "work/runs") or output.exists():
        raise ValueError("exclusive new output below work/runs required; partial restart refused")
    output.mkdir(parents=True)
    return output


def collect(output, model_path, plan):
    """Execute one explicitly registered cohort; failure preserves partial raw."""
    import torch
    validate_plan(plan)
    # Refuse an occupied target before allocating the graph or creating worlds.
    target = Path(output).absolute()
    if target != target.resolve() or not target.is_relative_to(ROOT / "work/runs") or target.exists():
        raise ValueError("exclusive new output below work/runs required; partial restart refused")
    graph = _load_graph(model_path, plan)
    output = _new_output(output)
    _write(output / "PLAN.json", plan)
    started = time.perf_counter()
    envs, controllers, batches, world_results = [], [], [], []
    phase = {key: 0.0 for key in ("observe_pack", "collate_hash", "encode_sample", "lower_step", "capture_write")}
    step = -1
    try:
        device = next(graph.parameters()).device
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in plan["actor_rng_seeds"]]
        registrations = [world_plan(plan, w) for w in range(plan["worlds"])]
        for seed in plan["seeds"]:
            envs.append(ExactKaggricultureEnvironment(seed=seed))
            controllers.append([EconomicOrders(), EconomicOrders()])
        with ExitStack() as stack:
            streams = []
            for w in range(plan["worlds"]):
                folder = output / f"world-{w:03d}"
                folder.mkdir()
                streams.append(stack.enter_context(gzip.open(folder / "turns.jsonl.gz", "xt", encoding="utf-8")))
            for step in range(719):
                tick = time.perf_counter()
                observations, packs = [], []
                for w, env in enumerate(envs):
                    if env.done or env.step_index != step:
                        raise ValueError(f"world{w} left registered synchronous horizon at step{step}")
                    obs = [env.agent_observation(s) for s in (0, 1)]
                    observations.append(obs)
                    for s in (0, 1):
                        controllers[w][s].observe(obs[s])
                        if step in STEPS:
                            packs.append(features.pack(obs[s], controllers[w][s], env.configuration))
                phase["observe_pack"] += time.perf_counter() - tick
                programs, calls = [None] * plan["lanes"], [None] * plan["lanes"]
                if step in STEPS:
                    tick = time.perf_counter()
                    batch, identity = batch_inputs(plan, step, packs)
                    batches.append(identity)
                    phase["collate_hash"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    with torch.no_grad():
                        context, values, _ = graph.encode(*(torch.as_tensor(batch[k], device=device)
                            for k in ("rows", "token_mask", "knownness")))
                        for lane, (w, s) in enumerate(plan["lane_order"]):
                            programs[lane] = graph.sample(context[lane:lane + 1], generator=generators[lane],
                                max_commands=plan["command_budget"], temperature=plan["temperature"])[0][0]
                            call = capture_decision(registrations[w], step, s, packs[lane], programs[lane],
                                float(values[lane]), controllers[w][s].snapshot())
                            calls[lane] = bound({**call, "lane_index": lane,
                                "cohort_batch_sha256": identity["content_sha256"]})
                    phase["encode_sample"] += time.perf_counter() - tick
                for w, env in enumerate(envs):
                    tick = time.perf_counter()
                    rejected = []
                    if step in STEPS:
                        for s in (0, 1):
                            for command in programs[2 * w + s]["program"]:
                                try:
                                    controllers[w][s].apply(command)
                                except ValueError as exc:
                                    rejected.append({"step": step, "seat": s, "command": command, "reason": str(exc)})
                    joint = [controllers[w][s](observations[w][s], env.configuration) for s in (0, 1)]
                    env.step(joint)
                    phase["lower_step"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    row = capture_turn(step, observations[w], programs[2*w:2*w+2] if step in STEPS else None,
                        calls[2*w:2*w+2] if step in STEPS else None, joint,
                        [env.agent_observation(s) for s in (0, 1)], controllers[w], rejected)
                    streams[w].write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                    phase["capture_write"] += time.perf_counter() - tick
                if step in STEPS:
                    print(json.dumps({"step": step, "worlds": plan["worlds"],
                        "model_decisions": len(batches) * plan["lanes"]}), flush=True)
        for w, env in enumerate(envs):
            result = finish(registrations[w], env, controllers[w],
                turns_sha256=file_sha(output / f"world-{w:03d}/turns.jsonl.gz"))
            _write(output / f"world-{w:03d}/RESULT.json", result)
            world_results.append({"world_index": w, "result_sha256": result["content_sha256"],
                "result_file_sha256": file_sha(output / f"world-{w:03d}/RESULT.json"),
                "turns_sha256": result["turns_sha256"]})
        _write(output / "COHORTS.json", bound({"schema": SCHEMA + "-cohorts",
            "plan_sha256": plan["content_sha256"], "batches": batches}))
        _equal(file_sha(model_path), plan["model_sha256"], "unchanged behavior model")
        _equal(_read(output / "PLAN.json"), plan, "unchanged plan")
        attest(plan["source_pins"])
        result = bound({"schema": SCHEMA + "-collection", "status": "COMPLETE_PENDING_INDEPENDENT_ADMISSION",
            "plan_sha256": plan["content_sha256"], "model_sha256": plan["model_sha256"],
            "source_sha256": plan["source_sha256"], "world_results": world_results,
            "cohorts_file_sha256": file_sha(output / "COHORTS.json"), "cohorts_sha256": digest(batches),
            "physical_games": plan["worlds"], "physical_steps": 719 * plan["worlds"],
            "model_decisions": len(STEPS) * plan["lanes"], "optimizer_updates": 0,
            "responsive_reference_games": 0, "terminal_training_admission": False,
            "seconds": time.perf_counter() - started, "host_wall_phase_seconds": phase})
        _write(output / "RESULT.json", result)  # The cohort is committed last, never world-by-world.
        return result
    except BaseException as exc:
        try:
            _write(output / "FAILURE.json", {"schema": SCHEMA + "-failure", "plan_sha256": plan["content_sha256"],
                "step": step, "world_steps": [e.step_index for e in envs], "error_type": type(exc).__name__,
                "message": str(exc), "terminal_training_admission": False, "optimizer_updates": 0})
        except OSError:
            pass  # Preserve the original failure even when a full disk prevents its receipt.
        raise


def _reconstruct_world(plan, index, result, rows):
    registration = world_plan(plan, index)
    def check(actual):
        _equal(actual, registration, "derived cohort world plan")
    checker = _bind(base._reconstruct, SCHEMA=SCHEMA, PLAN_SCHEMA=WORLD_PLAN_SCHEMA,
        GAME_SCHEMA=GAME_SCHEMA, EPISODE_SCHEMA=EPISODE_SCHEMA, _validate_plan=check,
        attest=attest, _equal=_equal, finish=finish, ExactKaggricultureEnvironment=ExactKaggricultureEnvironment,
        EconomicOrders=EconomicOrders)
    return checker(registration, result, rows)


def cohort_groups(plan, episodes):
    """Validate flat fitting identities and recover the ORIGINAL daily cohorts."""
    if not isinstance(episodes, list) or len(episodes) != plan["lanes"]:
        raise ValueError("every registered world/seat episode required")
    for lane, (w, s) in enumerate(plan["lane_order"]):
        episode = episodes[lane]
        for key, expected in (("schema", EPISODE_SCHEMA), ("seat", s), ("seed", plan["seeds"][w]),
                ("episode_id", world_plan(plan, w)["game_id"] + f"-seat{s}"),
                ("behavior_model_sha256", plan["model_sha256"]),
                ("model_manifest_sha256", plan["model_manifest_sha256"]),
                ("rollout_iteration", plan["rollout_iteration"]), ("source_sha256", plan["source_sha256"]),
                ("numerics", plan["numerics"])):
            _equal(episode[key], expected, "episode " + key)
        if (type(episode["terminal_reward"]) is not int or episode["terminal_reward"] not in (-1, 0, 1)
                or not isinstance(episode["calls"], list)
                or [c["step"] for c in episode["calls"]] != list(STEPS)):
            raise ValueError("all30 calls and actual terminal WDL required")
        if s == 1 and episode["terminal_reward"] != -episodes[lane - 1]["terminal_reward"]:
            raise ValueError("opposing world rewards must be zero-sum")
    groups = []
    for ordinal, step in enumerate(STEPS):
        calls = [e["calls"][ordinal] for e in episodes]
        packs = []
        for lane, call in enumerate(calls):
            _sealed(call)
            for key, expected in (("schema", SCHEMA + "-decision"), ("step", step),
                    ("seat", lane % 2), ("lane_index", lane)):
                _equal(call[key], expected, "daily call " + key)
            base._trace(call["behavior"], plan)
            _equal(call["old_log_probability"], call["behavior"]["joint_logp"], "actual joint logp")
            base._finite(call["old_value"])
            packs.append(base.unpack(call["input"]))
        _, identity = batch_inputs(plan, step, packs)
        for call in calls:
            _equal(call["cohort_batch_sha256"], identity["content_sha256"], "original cohort padding and lane hashes")
        groups.append({"calls": calls, "returns": [e["terminal_reward"] for e in episodes], "batch": identity})
    return groups


def witness(folder, *, expected_plan_file_sha256):
    """Fresh CPU worlds; no model, newly sampled program or duplicated raw file."""
    folder = Path(folder)
    watched = {folder / "PLAN.json": _sha(expected_plan_file_sha256)}
    plan, _ = _read_pinned(folder / "PLAN.json", expected_plan_file_sha256)
    validate_plan(plan)
    result, result_file_sha = _read_pinned(folder / "RESULT.json"); _sealed(result)
    watched[folder / "RESULT.json"] = result_file_sha
    for key, expected in (("schema", SCHEMA + "-collection"), ("status", "COMPLETE_PENDING_INDEPENDENT_ADMISSION"),
            ("plan_sha256", plan["content_sha256"]), ("model_sha256", plan["model_sha256"]),
            ("source_sha256", plan["source_sha256"]), ("physical_games", plan["worlds"]),
            ("physical_steps", plan["worlds"] * 719), ("model_decisions", plan["lanes"] * len(STEPS)),
            ("optimizer_updates", 0), ("responsive_reference_games", 0), ("terminal_training_admission", False)):
        _equal(result[key], expected, "complete cohort " + key)
    if len(result["world_results"]) != plan["worlds"] or (folder / "FAILURE.json").exists():
        raise ValueError("partial/failed cohort is not admissible")
    watched[folder / "COHORTS.json"] = result["cohorts_file_sha256"]
    shapes, _ = _read_pinned(folder / "COHORTS.json", result["cohorts_file_sha256"]); _sealed(shapes)
    _equal(shapes["schema"], SCHEMA + "-cohorts", "cohort shape schema")
    _equal(shapes["plan_sha256"], plan["content_sha256"], "cohort shape plan")
    episodes, witnesses = [], []
    for w, member in enumerate(result["world_results"]):
        _equal(member["world_index"], w, "ordered world membership")
        child = folder / f"world-{w:03d}"
        for leaf, expected in (("RESULT.json", member["result_file_sha256"]), ("turns.jsonl.gz", member["turns_sha256"])):
            watched[child / leaf] = _sha(expected)
            _equal(file_sha(child / leaf), expected, "exact producer " + leaf)
        world_result, _ = _read_pinned(child / "RESULT.json", member["result_file_sha256"])
        _equal(world_result["content_sha256"], member["result_sha256"], "world result content")
        _equal(world_result["turns_sha256"], member["turns_sha256"], "world raw identity")
        location = [-1]
        def records(stream):
            for ordinal, line in enumerate(stream):
                location[0] = ordinal
                yield _json(line)
        try:
            with gzip.open(child / "turns.jsonl.gz", "rt", encoding="utf-8") as stream:
                checked = _reconstruct_world(plan, w, world_result, records(stream))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"world{w} step{location[0]} physical witness: {exc}") from exc
        episodes.extend(checked["episodes"])
        witnesses.append(checked["engine_witness"])
    groups = cohort_groups(plan, episodes)
    identities = [g["batch"] for g in groups]
    _equal(identities, shapes["batches"], "fresh original complete cohorts")
    _equal(digest(identities), result["cohorts_sha256"], "original cohorts digest")
    for path, expected in watched.items():
        _equal(file_sha(path), expected, "unchanged source-bound producer")
    attest(plan["source_pins"])
    return {"plan": plan, "episodes": episodes, "engine_witness": bound({
        "schema": SCHEMA + "-engine-witness", "plan_sha256": plan["content_sha256"],
        "producer_result_sha256": result["content_sha256"], "producer_plan_file_sha256": expected_plan_file_sha256,
        "producer_result_file_sha256": watched[folder / "RESULT.json"],
        "training_batch_sha256": digest({"episodes": episodes}), "cohorts_sha256": digest(identities),
        "world_witnesses": witnesses, "reconstructed_physical_games": plan["worlds"],
        "physical_steps": 719 * plan["worlds"], "decision_calls": len(STEPS) * plan["lanes"],
        "terminal_cash": [w["terminal_cash"] for w in witnesses], "responsive_reference_games": 0,
        "method": "FRESH_ENGINE_RECORDED_PROGRAMS_EXACT_EXECUTOR_INPUTS_ACTIONS_SUCCESSORS",
        "terminal_training_admission": False})}


def verify_graph(graph, plan, episodes):
    """Reproduce all original cohorts and private sampler streams, without fit.

    Exact checkpoint loading / file identity is the caller's responsibility.
    This numerical check alone never constitutes physical/training admission.
    """
    import torch
    validate_plan(plan)
    _check_numerical_environment(plan, graph)
    _equal(base.policy_module().architecture(graph.config)["content_sha256"],
           plan["architecture_sha256"], "actual graph architecture")
    groups = cohort_groups(plan, episodes)
    device = next(graph.parameters()).device
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in plan["actor_rng_seeds"]]
    graph.eval()
    max_logp = max_value = 0.0
    with torch.no_grad():
        for step, group in zip(STEPS, groups, strict=True):
            calls = group["calls"]
            batch, _ = batch_inputs(plan, step, [base.unpack(c["input"]) for c in calls])
            context, values, _ = graph.encode(*(torch.as_tensor(batch[k], device=device)
                for k in ("rows", "token_mask", "knownness")))
            logp, entropy = graph.replay(context, [c["behavior"] for c in calls])
            if not bool(torch.isfinite(entropy).all()):
                raise ValueError("nonfinite replay entropy")
            for lane, call in enumerate(calls):
                original = call["behavior"]
                sampled = graph.sample(context[lane:lane + 1], generator=generators[lane],
                    max_commands=plan["command_budget"], temperature=plan["temperature"])[0][0]
                base._trace(sampled, plan)
                for key in ("schema", "architecture_sha256", "action_schema_sha256", "temperature",
                        "max_commands", "program", "commands", "sampling_masks", "training_admission"):
                    _equal(sampled[key], original[key], f"step{step} lane{lane} sampled " + key)
                errors = [abs(base._finite(a) - base._finite(b)) for a, b in
                    zip(sampled["selected_logps"], original["selected_logps"], strict=True)]
                logp_error = max([abs(base._finite(float(logp[lane])) - call["old_log_probability"]),
                    abs(sampled["joint_logp"] - call["old_log_probability"]), *errors])
                value_error = abs(base._finite(float(values[lane])) - call["old_value"])
                if max(logp_error, value_error) > ATOL:
                    raise ValueError(f"step{step} lane{lane} frozen cohort logp/value numerical mismatch")
                max_logp, max_value = max(max_logp, logp_error), max(max_value, value_error)
    attest(plan["source_pins"])
    return bound({"schema": SCHEMA + "-numerical-witness", "plan_sha256": plan["content_sha256"],
        "model_sha256": plan["model_sha256"], "model_manifest_sha256": plan["model_manifest_sha256"],
        "training_batch_sha256": digest({"episodes": episodes}),
        "cohorts_sha256": digest([g["batch"] for g in groups]), "calls": plan["lanes"] * len(STEPS),
        "numeric_atol": ATOL, "max_logp_error": max_logp, "max_value_error": max_value,
        "method": "EXACT_ORIGINAL_COHORT_GRAPH_REPLAY_PLUS_SCALAR_PRIVATE_LANE_SAMPLERS",
        "terminal_training_admission": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("collect")
    c.add_argument("--plan", required=True); c.add_argument("--plan-sha256", required=True)
    c.add_argument("--model", required=True); c.add_argument("--output", required=True)
    c.add_argument("--execute", action="store_true")
    w = commands.add_parser("witness")
    w.add_argument("--producer", required=True); w.add_argument("--plan-sha256", required=True)
    w.add_argument("--include-episodes", action="store_true", help="IPC only: do not retain duplicate derived JSON")
    args = parser.parse_args()
    if args.command == "collect":
        _equal(file_sha(args.plan), _sha(args.plan_sha256), "external plan file")
        plan = validate_plan(_read(args.plan))
        result = collect(args.output, args.model, plan) if args.execute else {
            "status": "REGISTERED_NOT_EXECUTED", "plan_sha256": plan["content_sha256"],
            "physical_games": 0, "optimizer_updates": 0}
    else:
        result = witness(args.producer, expected_plan_file_sha256=args.plan_sha256)
        if not args.include_episodes:
            result = {"plan": result["plan"], "engine_witness": result["engine_witness"]}
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
