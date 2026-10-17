"""Plan-first replay-BC free-running evaluation, task-local to slot 81.

prepare reads metadata only; run is an explicit maintainer native-game operation.
The injectable Python backend is for reduced software fixtures, never strength.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from checkpoint_adapter import (
    INPUT_ADAPTER, LiveBCActor, artifact_path, actor_view, bound, canonical, check_bound,
    digest, load_frozen, observation_identity, pin_checkpoint, public_configuration,
    read_json, sha256_file, validate_binding, verify_artifacts, write_new,
)

PLAN_SCHEMA = "public-winner-bc-freerun-plan-v1"
RESULT_SCHEMA = "public-winner-bc-freerun-result-v1"
OPPONENTS = ("PASS", "FIRST_BC", "SHOP_ROUTER")
MAX_CASES = 192
NATIVE_STEPS = 719


def describe_execution(root: Path = ROOT) -> dict:
    """Bound source/asset reads only. Does not load weights or start an engine."""
    from farm_neat.policy import network as policy, observation as codec
    from farm_neat.training import bc
    from farm_neat.evaluation import shop_router
    from farm_neat.game.exact_engine import assert_bundled_engine_identity
    root = Path(root).resolve()
    pins = {**policy.sources(), **bc.sources(), **shop_router.sources()}
    paths = list((root / "src/farm_neat/economy").rglob("*.py"))
    paths += [root / name for name in (
        "src/farm_neat/game/exact_engine.py", "src/farm_neat/game/kaggriculture_engine_reference.py",
        "src/farm_neat/game/kaggriculture.json", "src/farm_neat/evaluation/scenarios.py",
        "src/farm_neat/tooling/runtime.py", "project/runtime/interpreter.toml",
        "work/runs/reviewer-play/public-winner-bc-freerun/freerun.py",
        "work/runs/reviewer-play/public-winner-bc-freerun/checkpoint_adapter.py")]
    for path in paths:
        pins[str(path.relative_to(root))] = sha256_file(path)
    return bound({"schema": "public-winner-bc-execution-v1", "kind": "NATIVE",
        "source_pins": pins, "engine": assert_bundled_engine_identity(),
        "router_identity": shop_router.identity(), "input_adapter": INPUT_ADAPTER,
        "representation_schema_sha256": codec.REPRESENTATION_SCHEMA_SHA256,
        "input_memory": "NONE_AS_IN_BC_PACK_STEP; ACTUAL_EXECUTOR_MEMORY_NOT_FABRICATED_AS_LABELS",
        "decode": "Graph.encode -> Graph.sample(as_of_step=live_step) -> FullSeasonPlan.propose/publish -> unchanged executor",
        "numerical_profile": "FP32_REFERENCE_B1_DYNAMIC_ENTITIES; SAMPLED_TEMPERATURE; NOT_FAST_OR_TRAINING_REPLAY",
        "training_admission": False, "teacher_inputs": False,
        "implicit_economic_actions": False, "checkpoint_promotion": False})


def _integer(value: Any, low: int, high: int, name: str) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def validate_spec(spec: dict) -> dict:
    required = {"schema", "world_seeds", "sampler_seed", "opponents", "rules", "controls"}
    if not isinstance(spec, dict) or set(spec) != required or spec["schema"] != "public-winner-bc-comparison-spec-v1":
        raise ValueError("closed explicit prospective comparison spec required")
    seeds = spec["world_seeds"]
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 8 or len(set(seeds)) != len(seeds):
        raise ValueError("1..8 distinct prospective world seeds required")
    for seed in seeds:
        _integer(seed, 0, 2**31-1, "world seed")
    _integer(spec["sampler_seed"], 0, 2**63-1, "sampler seed")
    if spec["opponents"] != list(OPPONENTS):
        raise ValueError("preserve PASS, FIRST_BC and SHOP_ROUTER as separate ordered strata")
    rules = spec["rules"]
    if not isinstance(rules, list) or not 2 <= len(rules) <= 4:
        raise ValueError("original rules plus 1..3 explicit fixed-shop strata required")
    if len({r.get("id") for r in rules}) != len(rules):
        raise ValueError("distinct rule stratum IDs required")
    originals = 0
    from farm_neat.economy.heuristic.constants import SHOPS
    for rule in rules:
        if not isinstance(rule.get("id"), str) or not rule["id"] or not all(c.isalnum() or c in "-_" for c in rule["id"]):
            raise ValueError("simple nonempty rule ID required")
        if rule.get("kind") == "ORIGINAL":
            if set(rule) != {"id", "kind"}:
                raise ValueError("original-rule stratum cannot override game economics")
            originals += 1
        elif rule.get("kind") == "FIXED_SHOP":
            if (set(rule) != {"id", "kind", "shops", "random_weeds"} or type(rule["random_weeds"]) is not bool
                    or not isinstance(rule["shops"], list) or len(rule["shops"]) != 8
                    or any(shop not in SHOPS for shop in rule["shops"])):
                raise ValueError("eight legal fixed shops and an explicit weed setting required")
        else:
            raise ValueError("unknown prospective rule stratum")
    if originals != 1:
        raise ValueError("exactly one original-rule stratum required")
    controls = spec["controls"]
    if not isinstance(controls, dict) or set(controls) != {
            "publication_steps", "command_budget", "temperature", "seconds_per_game", "trace_byte_limit", "device"}:
        raise ValueError("closed inference/resource controls required")
    if controls["device"] not in ("cpu", "cuda"):
        raise ValueError("prospective inference device must be cpu or cuda")
    steps = controls["publication_steps"]
    if (not isinstance(steps, list) or not steps or steps[0] != 0 or len(steps) > 30
            or any(type(s) is not int or not 0 <= s < 719 or s % 24 for s in steps)
            or steps != sorted(set(steps))):
        raise ValueError("publication must occur at explicit ordered dawns beginning at opening")
    _integer(controls["command_budget"], 2, 256, "command budget including STOP")
    for key, low, high in (("temperature", 0.01, 100), ("seconds_per_game", 0.01, 3600)):
        value = controls[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"finite bounded {key} required")
    _integer(controls["trace_byte_limit"], 1000, 1_000_000_000, "trace byte limit")
    if len(seeds) * len(rules) * 3 * 2 * 2 > MAX_CASES:
        raise ValueError(f"prospective pilot exceeds {MAX_CASES} physical cases")
    return copy.deepcopy(spec)


def _stream(base: int, pair: dict, role: str) -> int:
    return int(digest({"base": base, "pair": pair, "stream": role})[:16], 16) % 2**63


def make_plan(spec: dict, models: dict, execution: dict) -> dict:
    """No checkpoint path access: even absent training returns can be pre-bound."""
    spec = validate_spec(spec)
    if not isinstance(models, dict) or set(models) != {"old", "new", "first_bc"}:
        raise ValueError("old/new/frozen first-BC checkpoint bindings required")
    for model in models.values():
        validate_binding(model)
    if models["old"]["model_sha256"] == models["new"]["model_sha256"]:
        raise ValueError("old and new must be distinct checkpoints; relabelled substitution refused")
    if len({m["architecture_sha256"] for m in models.values()}) != 1:
        raise ValueError("same full-economic architecture required for the controlled BC comparison")
    check_bound(execution, "execution identity")
    if (execution.get("schema") != "public-winner-bc-execution-v1"
            or execution.get("input_adapter") != INPUT_ADAPTER):
        raise ValueError("explicit observed-BC live input adapter identity required")
    cases, used_streams = [], set()
    for rule in spec["rules"]:
        for world_seed in spec["world_seeds"]:
            for opponent in OPPONENTS:
                for seat in (0, 1):
                    pair = {"rule_id": rule["id"], "world_seed": world_seed, "opponent": opponent, "seat": seat}
                    candidate_seed = _stream(spec["sampler_seed"], pair, "candidate")
                    opponent_seed = _stream(spec["sampler_seed"], pair, "opponent") if opponent == "FIRST_BC" else None
                    for stream in (candidate_seed, opponent_seed):
                        if stream is not None:
                            if stream in used_streams:
                                raise ValueError("sampler stream collision; select a new prospective base seed")
                            used_streams.add(stream)
                    identity = ({"kind": "PASS"} if opponent == "PASS" else
                        {"kind": "FROZEN_FIRST_BC", "binding": copy.deepcopy(models["first_bc"])} if opponent == "FIRST_BC" else
                        copy.deepcopy(execution["router_identity"]))
                    pair_id = digest(pair)
                    # Fixed alternating arm order. It does not adapt to observed outcomes.
                    order = ("old", "new") if (len(cases)//2) % 2 == 0 else ("new", "old")
                    for role in order:
                        case = {**pair, "pair_id": pair_id, "model": role,
                            "model_binding_sha256": models[role]["content_sha256"],
                            "opponent_identity": identity, "sampler_seed": candidate_seed,
                            "opponent_sampler_seed": opponent_seed}
                        cases.append({**copy.deepcopy(case), "case_id": digest(case)})
    return bound({"schema": PLAN_SCHEMA, "scientific_status": "NOT_RUN", "spec": spec,
        "models": copy.deepcopy(models), "execution": copy.deepcopy(execution), "cases": cases,
        "native_steps_per_complete_game": NATIVE_STEPS, "primary": "TERMINAL_W_D_L_BY_STRATUM",
        "secondary": "NEW_MINUS_OLD_PAIRED_TERMINAL_CASH_MARGIN",
        "pairing": "SAME_RULE_WORLD_OPPONENT_SEAT_AND_SAMPLER_STREAMS; BOTH_SEATS_ARE_ONE_WORLD_CLUSTER",
        "pooling": "NEVER_POOL_OPPONENTS_OR_RULES; NEVER_CLAIM_SELF_PLAY_50_PERCENT_AS_STRENGTH",
        "no_automatic_retry": True, "training_admission": False})


def validate_plan(plan: dict) -> dict:
    check_bound(plan, "prospective plan")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("explicit prospective plan schema required")
    expected = make_plan(plan["spec"], plan["models"], plan["execution"])
    if plan != expected:
        raise ValueError("prospective plan cases/settings drift; create a new plan before results")
    return plan


def runtime_guard(root: Path, device: str) -> dict:
    """Maintainer profile only; no install, interpreter guessing or CUDA fallback."""
    import platform
    import tomllib
    from farm_neat.tooling import runtime
    root = Path(root).resolve()
    lock = tomllib.loads((root / "project/runtime/interpreter.toml").read_text())
    section = runtime.training_section(lock)
    profile = lock[section]
    expected = Path(profile["path"])
    linked = root / profile["link"] if profile.get("link") else expected
    selected = linked if linked.exists() else expected
    if not selected.is_file() or Path(sys.executable).resolve() != selected.resolve():
        raise ValueError(f"registered training interpreter required: {selected}; active {sys.executable}")
    if profile.get("sha256") and sha256_file(selected.resolve()) != profile["sha256"]:
        raise ValueError("registered interpreter SHA mismatch")
    if not sys.dont_write_bytecode:
        raise ValueError("run the pinned interpreter with -B")
    import numpy as np
    import torch
    actual = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__)}
    if any(actual[key] != profile[key] for key in actual):
        raise ValueError(f"registered numerical versions required: actual {actual}")
    if "pythonpath" in profile:
        site = Path(profile["pythonpath"]).resolve()
        if any(not Path(module.__file__).resolve().is_relative_to(site) for module in (np, torch)):
            raise ValueError("registered numerical package site required")
    if any(os.environ.get(key) != value for key, value in profile["environment"].items()):
        raise ValueError("registered interpreter environment variables required; use tooling/runtime.py run")
    if device == "cuda" and (not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()):
        raise ValueError("explicit CUDA evaluation requires the existing SLURM allocation; no CPU fallback")
    if device not in ("cpu", "cuda"):
        raise ValueError("explicit inference device required")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if torch.is_autocast_enabled(device):
        raise ValueError("FP32 no-autocast evaluation required")
    return {**actual, "runtime_section": section, "device": device, "dtype": "FLOAT32", "tf32": False,
            "autocast": False, "deterministic_algorithms": True, "threads": 1,
            "decoder": "REFERENCE_GRAPH_SAMPLE_B1", "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


class NativeBackend:
    kind = "NATIVE"
    expected_steps = NATIVE_STEPS

    def __init__(self, *, root: Path = ROOT, device: str = "cpu"):
        self.root, self.device = Path(root).resolve(), device
        self.profile, self.plan = None, None

    def preflight(self, plan: dict) -> dict:
        if self.device != plan["spec"]["controls"]["device"]:
            raise ValueError("device differs from prospective plan; create a new plan before results")
        self.profile = runtime_guard(self.root, self.device)
        if plan["execution"] != describe_execution(self.root):
            raise ValueError("prospectively bound execution sources/assets/input adapter changed")
        self.plan = plan
        return {role: load_frozen(binding, root=self.root, device=self.device)
                for role, binding in plan["models"].items()}

    def new_environment(self, case: dict, rule: dict):
        if rule["kind"] == "ORIGINAL":
            from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
            return ExactKaggricultureEnvironment(seed=case["world_seed"])
        from farm_neat.evaluation.scenarios import ScenarioEnvironment
        # ScenarioEnvironment injects only the scheduled shops/weed setting.
        # Its validated legacy 'opponent' field is unused: this driver supplies
        # the exact separately bound actors, and never calls scenarios.play.
        return ScenarioEnvironment({"id": rule["id"], "purpose": "PROSPECTIVE_REPLAY_BC_COMPARISON",
            "seed": case["world_seed"], "shops": copy.deepcopy(rule["shops"]),
            "random_weeds": rule["random_weeds"], "opponent": "pass"})

    def new_actor(self, loaded, **kwargs):
        return LiveBCActor(loaded, **kwargs)

    def new_router(self):
        from farm_neat.evaluation.shop_router import ShopRouterAdapter
        actor = ShopRouterAdapter()
        if actor.identity != self.plan["execution"]["router_identity"]:
            raise ValueError("frozen ShopRouter identity substitution refused")
        return actor.act

    def finalize(self, plan, loaded):
        if describe_execution(self.root) != plan["execution"]:
            raise ValueError("execution sources/assets changed during evaluation")
        for binding in plan["models"].values():
            verify_artifacts(binding, self.root)


class TraceLimitError(RuntimeError):
    pass


class TraceStore:
    def __init__(self, path: Path, limit: int):
        self.stream = gzip.open(path, "xt")
        self.limit, self.bytes, self.exhausted = limit, 0, False

    def emit(self, row: dict) -> None:
        text = canonical(row) + "\n"
        size = len(text.encode())
        if self.bytes + size > self.limit:
            self.exhausted = True
            raise TraceLimitError("prospective uncompressed trace byte budget exhausted; no silent trace dropping")
        self.stream.write(text)
        self.bytes += size

    def flush(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _terminal(env, expected_steps: int) -> tuple[list[float], list[str]]:
    if not env.done or env.step_index != expected_steps:
        raise ValueError(f"incomplete/censored game: expected {expected_steps} transitions, got {env.step_index}")
    statuses = [row["status"] for row in env.state]
    rewards = [row["reward"] for row in env.state]
    if statuses != ["DONE", "DONE"] or len(rewards) != 2:
        raise ValueError("both terminal statuses must be DONE; error/inactive games are not W/D/L")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in rewards):
        raise ValueError("finite nonnegative exact terminal cash required")
    for seat in (0, 1):
        if [f["money"] for f in env.agent_observation(seat)["farms"]] != rewards:
            raise ValueError("terminal reward/public-cash/seat disagreement")
    return [float(x) for x in rewards], statuses


def _play_case(case, rule, plan, loaded, backend, trace):
    from farm_neat.game.exact_engine import PASS_ACTION
    started = time.monotonic()
    env = None
    row = {"case": copy.deepcopy(case), "mode": backend.kind, "status": "FAILED",
        "model_identity": copy.deepcopy(plan["models"][case["model"]]),
        "opponent_identity": copy.deepcopy(case["opponent_identity"]),
        "environment_start_attempted": False, "physical_games_started": 0,
        "physical_games_completed": 0, "physical_steps": 0, "no_automatic_retry": True}
    def emit(event):
        trace.emit({**event, "case_id": case["case_id"]})
    def deadline():
        if time.monotonic() - started > plan["spec"]["controls"]["seconds_per_game"]:
            raise TimeoutError("prospective per-game deadline exceeded; no automatic retry")
    try:
        emit({"type": "CASE_STARTED", "case": case})
        controls = plan["spec"]["controls"]
        candidate = backend.new_actor(loaded[case["model"]], seat=case["seat"],
            sampler_seed=case["sampler_seed"], controls=copy.deepcopy(controls), emit=emit)
        if case["opponent"] == "FIRST_BC":
            other = backend.new_actor(loaded["first_bc"], seat=1-case["seat"],
                sampler_seed=case["opponent_sampler_seed"], controls=copy.deepcopy(controls), emit=emit)
        elif case["opponent"] == "SHOP_ROUTER":
            other = backend.new_router()
        else:
            other = lambda obs, cfg: copy.deepcopy(PASS_ACTION)
        actors = [None, None]
        actors[case["seat"]], actors[1-case["seat"]] = candidate, other
        deadline()
        row["environment_start_attempted"] = True
        env = backend.new_environment(copy.deepcopy(case), copy.deepcopy(rule))
        row["physical_games_started"] = 1
        if env.step_index != 0 or env.done:
            raise ValueError("fresh unadvanced environment required")
        while not env.done and env.step_index < backend.expected_steps:
            deadline()
            step = env.step_index
            # Only the engine's actor-facing accessors cross into any actor.
            observations = [actor_view(env.agent_observation(seat), seat) for seat in (0, 1)]
            identities = [observation_identity(obs) for obs in observations]
            cfg = public_configuration(env.configuration)
            actions = []
            for seat, actor in enumerate(actors):
                actions.append(actor(copy.deepcopy(observations[seat]), copy.deepcopy(cfg)))
                deadline()
            emit({"type": "ACTION_SUBMITTED", "step": step, "actor_observation_identities": identities,
                  "actions_by_seat": copy.deepcopy(actions)})
            env.step(actions)
            if env.step_index != step + 1:
                raise ValueError("environment clock must advance exactly one physical transition")
            row["physical_steps"] = env.step_index
            emit({"type": "TURN_ADVANCED", "step": env.step_index})
            deadline()
        cash, statuses = _terminal(env, backend.expected_steps)
        margin = cash[case["seat"]] - cash[1-case["seat"]]
        row.update(status="COMPLETE", physical_games_completed=1,
            terminal_cash_by_seat=cash, terminal_statuses=statuses, margin=margin,
            wdl="W" if margin > 0 else "L" if margin < 0 else "D",
            match_score=1.0 if margin > 0 else 0.0 if margin < 0 else 0.5)
        emit({"type": "CASE_COMPLETE", "terminal_cash_by_seat": cash, "wdl": row["wdl"]})
    except (Exception, KeyboardInterrupt) as error:
        # If even the final trace could not be retained, do not admit the game.
        row.update(status="FAILED", physical_games_completed=0,
                   error={"type": type(error).__name__, "message": str(error)},
                   interrupted=isinstance(error, KeyboardInterrupt))
        for key in ("terminal_cash_by_seat", "terminal_statuses", "margin", "wdl", "match_score"):
            row.pop(key, None)
        if env is not None:
            row["physical_steps"] = env.step_index
        if not trace.exhausted:
            try:
                emit({"type": "CASE_FAILED", "error": row["error"], "physical_steps": row["physical_steps"]})
            except TraceLimitError:
                pass  # Failure row is durably retained in the independent game journal.
    row["seconds"] = time.monotonic() - started
    return row


def summarize(plan: dict, rows: list[dict], *, mode: str, errors: list[dict], runtime_profile=None) -> dict:
    by_id = {r["case"]["case_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate physical case result; automatic retries are forbidden")
    planned_ids = {c["case_id"] for c in plan["cases"]}
    if not set(by_id).issubset(planned_ids):
        raise ValueError("result not in prospective plan")
    counts = {"planned": len(plan["cases"]), "attempted": len(rows),
        "started": sum(r["physical_games_started"] for r in rows),
        "complete": sum(r["status"] == "COMPLETE" for r in rows),
        "failed": sum(r["status"] == "FAILED" for r in rows),
        "not_started": len(plan["cases"]) - len(rows)}
    counts["native_game_start_attempts"] = sum(r["environment_start_attempted"] for r in rows) if mode == "NATIVE" else 0
    counts["native_games_completed"] = counts["complete"] if mode == "NATIVE" else 0
    counts["fixture_games_completed"] = counts["complete"] if mode == "INJECTED_FIXTURE" else 0
    strata = []
    for rule in plan["spec"]["rules"]:
        for opponent in OPPONENTS:
            cases = [c for c in plan["cases"] if c["rule_id"] == rule["id"] and c["opponent"] == opponent]
            group = {"rule": copy.deepcopy(rule), "opponent": opponent,
                "opponent_identity": copy.deepcopy(cases[0]["opponent_identity"]), "models": {}}
            for role in ("old", "new"):
                intended = [c for c in cases if c["model"] == role]
                present = [by_id[c["case_id"]] for c in intended if c["case_id"] in by_id]
                complete = [r for r in present if r["status"] == "COMPLETE"]
                group["models"][role] = {"identity": copy.deepcopy(plan["models"][role]),
                    "planned_games": len(intended), "complete_games": len(complete),
                    "failed_games": sum(r["status"] == "FAILED" for r in present),
                    "not_started_games": len(intended) - len(present),
                    "wdl": {label: sum(r["wdl"] == label for r in complete) for label in ("W", "D", "L")},
                    "match_score": _mean([r["match_score"] for r in complete]),
                    "mean_cash_margin": _mean([r["margin"] for r in complete]),
                    "self_play": opponent == "FIRST_BC" and plan["models"][role]["model_sha256"] == plan["models"]["first_bc"]["model_sha256"]}
            pairs, missing = [], []
            for pair_id in dict.fromkeys(c["pair_id"] for c in cases):
                pair_cases = [c for c in cases if c["pair_id"] == pair_id]
                records = {c["model"]: by_id.get(c["case_id"]) for c in pair_cases}
                if any(records[role] is None or records[role]["status"] != "COMPLETE" for role in ("old", "new")):
                    missing.append(pair_id)
                    continue
                old, new = records["old"], records["new"]
                pairs.append({"pair_id": pair_id, "world_seed": old["case"]["world_seed"], "seat": old["case"]["seat"],
                    "sampler_seed": old["case"]["sampler_seed"], "opponent_sampler_seed": old["case"]["opponent_sampler_seed"],
                    "old_case_id": old["case"]["case_id"], "new_case_id": new["case"]["case_id"],
                    "margin_delta": new["margin"]-old["margin"], "match_score_delta": new["match_score"]-old["match_score"]})
            worlds = []
            for seed in plan["spec"]["world_seeds"]:
                seat_pairs = [p for p in pairs if p["world_seed"] == seed]
                if {p["seat"] for p in seat_pairs} == {0, 1}:
                    worlds.append({"world_seed": seed, "margin_delta": _mean([p["margin_delta"] for p in seat_pairs]),
                                   "match_score_delta": _mean([p["match_score_delta"] for p in seat_pairs])})
            group["paired"] = {"complete_seat_pairs": len(pairs), "complete_world_pairs": len(worlds),
                "expected_seat_pairs": 2*len(plan["spec"]["world_seeds"]), "missing_pair_ids": missing,
                "mean_margin_delta": _mean([p["margin_delta"] for p in pairs]),
                "mean_match_score_delta": _mean([p["match_score_delta"] for p in pairs]),
                "world_cluster_mean_margin_delta": _mean([w["margin_delta"] for w in worlds]),
                "world_cluster_mean_match_score_delta": _mean([w["match_score_delta"] for w in worlds]),
                "seat_pairs": pairs, "complete_world_pairs_detail": worlds,
                "uncertainty": "NO_INDEPENDENCE_OR_SIGNIFICANCE_CLAIM; SEAT_SWAPS_CLUSTERED_BY_WORLD"}
            strata.append(group)
    scientific = ("NOT_RUN" if mode != "NATIVE" or not counts["native_game_start_attempts"] else
                  "INVALID_IDENTITY_OR_EXECUTION" if errors else
                  "INCOMPLETE" if counts["failed"] or counts["not_started"] else "COMPLETE_PROSPECTIVE_COMPARISON")
    return {"schema": RESULT_SCHEMA, "plan_sha256": plan["content_sha256"], "mode": mode,
        "software_status": "CHECKS_OR_EXECUTION_FAILED" if errors or counts["failed"] or counts["not_started"] else "COMPLETED",
        "scientific_status": scientific, "counts": counts, "strata": strata,
        "errors": copy.deepcopy(errors), "runtime_profile": copy.deepcopy(runtime_profile),
        "strength_claim": False, "training_admission": False, "checkpoint_promotion": False,
        "scope": "Synthetic fixtures prove software only; native W/D/L stays stratified. No pooled self-play strength claim."}


def _atomic_result(path: Path, result: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_evaluation(plan: dict, output: Path, *, backend) -> dict:
    """Backend injection is explicit; the CLI only instantiates NativeBackend.

    Per-game failures are journaled and never retried or removed. Missing inputs
    stop execution before any world, but do not impede make_plan/CLI preparation.
    """
    validate_plan(plan)
    if backend.kind not in ("NATIVE", "INJECTED_FIXTURE"):
        raise ValueError("explicit native or injected-fixture backend identity required")
    if backend.kind == "NATIVE" and (not isinstance(backend, NativeBackend) or backend.expected_steps != NATIVE_STEPS):
        raise ValueError("native complete-game counts require the fixed native backend/horizon")
    _integer(backend.expected_steps, 1, NATIVE_STEPS, "backend horizon")
    output = Path(output)
    if output.resolve() != output.absolute():
        raise ValueError("output symlink redirection refused")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "PLAN.json", plan)
    rows, errors, loaded = [], [], None
    trace = TraceStore(output / "TRACE.jsonl.gz", plan["spec"]["controls"]["trace_byte_limit"])
    def finish(*, final=False):
        result = summarize(plan, rows, mode=backend.kind, errors=errors, runtime_profile=getattr(backend, "profile", None))
        result["trace_uncompressed_bytes"] = trace.bytes
        result["trace_budget_exhausted"] = trace.exhausted
        result["evidence_closed"] = final
        if not final and backend.kind == "NATIVE" and result["counts"]["native_game_start_attempts"]:
            result["scientific_status"] = "RUNNING_NOT_YET_CLOSED"
        if final:
            result["evidence_sha256"] = {name: sha256_file(output / name) if (output / name).is_file() else None
                for name in ("PLAN.json", "TRACE.jsonl.gz", "GAMES.jsonl")}
        result = bound(result)
        _atomic_result(output / "RESULT.json", result)
        return result
    try:
        # Record NOT_RUN even if a subsequent process/environment preflight fails.
        finish()
        loaded = backend.preflight(plan)
        if not isinstance(loaded, dict) or set(loaded) != {"old", "new", "first_bc"}:
            raise ValueError("all three explicitly bound checkpoints must load before native games")
        for role, model in loaded.items():
            if getattr(model, "binding", None) != plan["models"][role]:
                raise ValueError(f"checkpoint substitution refused for role {role}")
        rules = {r["id"]: r for r in plan["spec"]["rules"]}
        with (output / "GAMES.jsonl").open("x") as journal:
            for case in plan["cases"]:
                row = _play_case(case, rules[case["rule_id"]], plan, loaded, backend, trace)
                rows.append(row)
                journal.write(canonical(row) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                trace.flush()
                finish()
                if row.get("interrupted"):
                    errors.append({"type": "KeyboardInterrupt", "message": "interrupted case retained; remaining cases NOT_STARTED"})
                    break
                if trace.exhausted:
                    errors.append({"type": "TraceLimitError", "message": "stopped at the prospective trace budget; remaining cases NOT_STARTED"})
                    break
        backend.finalize(plan, loaded)
    except (Exception, KeyboardInterrupt) as error:
        errors.append({"type": type(error).__name__, "message": str(error)})
    finally:
        trace.close()
        result = finish(final=True)
    return result


def _native_output(path: Path) -> Path:
    path = path.absolute()
    if path != path.resolve() or not path.is_relative_to(ROOT / "work/runs"):
        raise ValueError("new non-symlink output under work/runs required")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pin = commands.add_parser("bind", help="read-only exact full-economic checkpoint binding; pinned runtime required")
    pin.add_argument("--model", type=Path, required=True)
    pin.add_argument("--name", required=True)
    pin.add_argument("--output", type=Path, required=True)
    pin.add_argument("--allow-remaining-window-transition", action="store_true",
                     help="explicitly opt into the one canonical source transition; never arbitrary rebind")
    prepare = commands.add_parser("prepare", help="freeze worlds/streams/opponents/settings without opening weights")
    prepare.add_argument("--spec", type=Path, required=True)
    for role in ("old", "new", "first-bc"):
        prepare.add_argument("--" + role, type=Path, required=True, help="exact checkpoint binding JSON, not model weights")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--device", choices=("cpu", "cuda"), default=None,
                         help="override spec device before freezing the prospective plan")
    inspect = commands.add_parser("inspect", help="validate a plan without checkpoint or native-game execution")
    inspect.add_argument("--plan", type=Path, required=True)
    run = commands.add_parser("run", help="MAINTAINER ONLY: execute the complete prospectively fixed comparison")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default=None,
                     help="must match the prospective plan; defaults to its bound device")
    run.add_argument("--allow-native-games", action="store_true", help="required explicit maintainer native-game opt-in")
    args = parser.parse_args(argv)
    try:
        if args.command == "bind":
            runtime_guard(ROOT, "cpu")
            value = pin_checkpoint(ROOT, args.model, name=args.name,
                allow_remaining_window_transition=args.allow_remaining_window_transition)
            write_new(_native_output(args.output), value)
            print(canonical(value))
        elif args.command == "prepare":
            models = {role: read_json(getattr(args, role)) for role in ("old", "new", "first_bc")}
            spec = read_json(args.spec)
            if args.device is not None:
                spec["controls"]["device"] = args.device
            plan = make_plan(spec, models, describe_execution(ROOT))
            write_new(_native_output(args.output), plan)
            print(canonical({"plan_sha256": plan["content_sha256"], "planned_games": len(plan["cases"]),
                             "scientific_status": "NOT_RUN", "weights_loaded": 0, "native_games_started": 0}))
        elif args.command == "inspect":
            plan = validate_plan(read_json(args.plan))
            print(canonical({"plan_sha256": plan["content_sha256"], "planned_games": len(plan["cases"]),
                             "world_seeds": plan["spec"]["world_seeds"], "device": plan["spec"]["controls"]["device"],
                             "scientific_status": "NOT_RUN"}))
        else:
            if not args.allow_native_games:
                parser.error("run is maintainer-only; --allow-native-games is required")
            plan = validate_plan(read_json(args.plan))
            device = args.device or plan["spec"]["controls"]["device"]
            result = run_evaluation(plan, _native_output(args.output),
                backend=NativeBackend(root=ROOT, device=device))
            print(canonical({"software_status": result["software_status"], "scientific_status": result["scientific_status"],
                             "counts": result["counts"], "errors": result["errors"]}))
            return 0 if result["scientific_status"] == "COMPLETE_PROSPECTIVE_COMPARISON" else 2
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
