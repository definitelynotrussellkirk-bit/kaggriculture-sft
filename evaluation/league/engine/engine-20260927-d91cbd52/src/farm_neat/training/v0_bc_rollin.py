"""Source-bound own-prefix V0 BC roll-in collection.

This module records the dawn states reached by the unchanged V0 PolicyAgent and
ScorerPolicy.  It does not teacher-force, repair a decoded path, train, promote a
checkpoint, or claim strength.

A cycle is an immutable registration plus one receipt per physical game.  An
interrupted cycle may be rerun against the same output directory: receipts that
already match the registration are deduplicated, an unmatched/mutated receipt is
fatal, and a failed game is evidence rather than an automatic retry.  Terminal
labels are attached only after a complete 719-step game.

Example (production checkpoint/runtime authorization remains maintainer-owned)::

  PYTHONPATH=src python3 -B -m farm_neat.training.v0_bc_rollin collect \\
    --cycle-id dev-2026-09-18 --output-dir work/runs/v0-rollin/dev-2026-09-18 \\
    --checkpoint-dir work/runs/.../checkpoint-000512 \\
    --checkpoint-sha256 <COMMIT.json sha256> \\
    --case router-dairy:0 --opponent pass --opponent router --device cpu
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.epochs import DAWNS
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import Masks, PHI_NAMES, State
from farm_neat.evaluation import realization, scenarios
from farm_neat.evaluation.common import file_sha
from farm_neat.evaluation import v0_policy as E
from farm_neat.policy.v0_scorer import Prefix, V0Scorer, partition_menu
from farm_neat.policy.qwen_hybrid import Config
from farm_neat.training import v0_targets as T

SCHEMA = "farm-v0-bc-rollin-v1"
REGISTRATION_SCHEMA = SCHEMA + "-registration"
GAME_SCHEMA = SCHEMA + "-game"
STATE_SCHEMA = SCHEMA + "-state"
SOURCE_GROUP_SCHEMA = SCHEMA + "-source-group"
INITIAL_CONTINUATION_SCHEMA = SCHEMA + "-initial-continuation"
SCOPE = "DEVELOPMENT_OWN_PREFIX_STATE_COLLECTION; NOT_STRENGTH_OR_TEACHER_ADMISSION"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(E.wire(value).encode()).hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"object JSON required: {path}")
    return value


def _safe_output(path: str | Path) -> Path:
    path = Path(path).resolve()
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlinked output forbidden")
    return path


def source_pins() -> dict[str, str]:
    """Source closure that controls collection semantics, not checkpoint weights."""
    from farm_neat.training import v0_bc
    pins = {**E.source_pins(), **scenarios.sources(), **v0_bc.source_pins()}
    pins[str(Path(__file__).resolve().relative_to(ROOT))] = file_sha(__file__)
    return dict(sorted(pins.items()))


def _physical_world(spec: dict) -> dict:
    """Opponent-independent physical world identity."""
    body = {
        "schema": "farm-v0-rollin-physical-world-v1",
        "seed": spec["seed"],
        "random_weeds": spec["random_weeds"],
        "shops": copy.deepcopy(spec["shops"]),
    }
    return {**body, "physical_world_id": _sha256_text(body)}


def actor_state(public: dict) -> State:
    """Reconstruct a dawn State from the explicit V0 actor-visible allowlist.

    Unlike the opening-only BC exporter adapter, roll-ins cover every dawn.  The
    absolute dawn follows exactly from remaining_days; no seed, opponent identity,
    future shop schedule, rival-private field or terminal label is reconstructed.
    """
    try:
        _require(public["schema"] == T.ACTOR_SCHEMA, "unknown actor observation schema")
        remaining = public["remaining_days"]
        _require(type(remaining) is int and 1 <= remaining <= V.SEASON_DAYS, "valid dawn remaining_days required")
        day = V.SEASON_DAYS - remaining
        s = State.__new__(State)
        s.player, s.day, s.hour, s.step = 0, day, 0, day * 24
        s.cash = public["cash"]
        s.quadrants, s.hands_today = public["owned_quadrants"], public["hands_today"]
        s.hires_today, s.occupied = public["hires_today"], public["occupied_sites"]
        s.stock, s.seeds = copy.deepcopy(public["stock"]), copy.deepcopy(public["seeds"])
        s.inventory, s.params = copy.deepcopy(public["market_inventory"]), copy.deepcopy(public["market_params"])
        s.shops = copy.deepcopy(public["unlocked_shops"])
        _require(T.actor_observation(s) == public, "actor observation round-trip mismatch")
        return s
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid actor-only observation") from exc


def _source_group(world_id: str, seat: int, opponent_digest: str, checkpoint_digest: str) -> tuple[str, dict]:
    body = {
        "schema": SOURCE_GROUP_SCHEMA,
        "physical_world_id": world_id,
        "seat": seat,
        "opponent_digest": opponent_digest,
        "checkpoint_digest": checkpoint_digest,
    }
    digest = _sha256_text(body)
    return "v0-rollin/" + digest, {**body, "source_group_digest": digest}


@dataclass(frozen=True)
class Checkpoint:
    scorer: V0Scorer
    provenance: dict


def load_checkpoint(directory: str | Path, expected_commit_sha256: str, *, device: str = "cpu") -> Checkpoint:
    """Load an immutable native V0 BC checkpoint without reconstructing a Trainer.

    The commit file, payload hash/size, embedded training identity, current source
    closure and model manifest all have to agree.  Optimizer state is intentionally
    ignored: roll-in is inference only.
    """
    import torch
    from farm_neat.training import v0_bc

    directory = Path(directory).resolve()
    _require(device in ("cpu", "cuda"), "device must be cpu or cuda")
    _require(len(expected_commit_sha256) == 64 and all(c in "0123456789abcdef" for c in expected_commit_sha256),
             "explicit lowercase checkpoint COMMIT sha256 required")
    commit_path, state_path = directory / "COMMIT.json", directory / "state.pt"
    _require(commit_path.is_file() and state_path.is_file(), "checkpoint COMMIT.json and state.pt required")
    _require(not commit_path.is_symlink() and not state_path.is_symlink(), "symlinked checkpoint forbidden")
    _require(file_sha(commit_path) == expected_commit_sha256, "checkpoint COMMIT sha256 mismatch")
    commit = _load_json(commit_path)
    _require(commit.get("schema") == v0_bc.COMMIT_SCHEMA and commit.get("status") == "COMMITTED",
             "committed native V0 BC checkpoint required")
    _require(state_path.stat().st_size == commit.get("state_bytes") and file_sha(state_path) == commit.get("state_sha256"),
             "checkpoint state hash/size mismatch")
    identity = commit.get("identity")
    _require(isinstance(identity, dict) and isinstance(identity.get("model"), dict), "checkpoint model identity required")
    v0_bc.validate_model_manifest(identity["model"])
    _require(identity.get("sources") == v0_bc.source_pins(), "checkpoint training source closure changed")
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, dict) and payload.get("identity") == identity, "checkpoint payload identity mismatch")
    _require(payload.get("global_update") == commit.get("global_update")
             and payload.get("samples_seen") == commit.get("samples_seen"), "checkpoint age/exposure mismatch")
    _require(isinstance(payload.get("model"), dict), "checkpoint model weights required")
    model_identity = identity["model"]
    scorer = V0Scorer(Config(**model_identity["config"]), diagnostic=model_identity["diagnostic"])
    scorer.load_state_dict(payload["model"])
    scorer.to(torch.device(device)).eval()
    _require(scorer.get_extra_state() == model_identity, "loaded scorer identity mismatch")
    provenance = {
        "checkpoint_digest": expected_commit_sha256,
        "checkpoint_path": str(directory),
        "state_sha256": commit["state_sha256"],
        "global_update": commit["global_update"],
        "samples_seen": commit["samples_seen"],
        "model_identity": copy.deepcopy(model_identity),
        "scope": commit.get("scope"),
    }
    return Checkpoint(scorer, provenance)


def register_cycle(*, cycle_id: str, checkpoint: dict, cases: Iterable[tuple[str, int]],
                   opponents: Iterable[str], mode: str = "greedy_class", sampler_seed: int = 0,
                   max_games: int = 4, labor_factor: float = 4.0, hands_mode: str = "daily",
                   timeout_seconds: float = 180.0) -> dict:
    """Bind worlds/seats/opponents/checkpoint before any game starts."""
    _require(isinstance(cycle_id, str) and cycle_id and len(cycle_id) <= 160, "nonempty bounded cycle id required")
    _require(isinstance(checkpoint, dict) and isinstance(checkpoint.get("checkpoint_digest"), str),
             "checkpoint provenance with digest required")
    checkpoint_digest = checkpoint["checkpoint_digest"]
    _require(len(checkpoint_digest) == 64, "checkpoint digest must be sha256")
    _require(mode in E.MODES and type(sampler_seed) is int and 0 <= sampler_seed < 2**63,
             "supported decode mode and explicit sampler seed required")
    _require(type(max_games) is int and 1 <= max_games <= 1024, "positive explicit game budget required")
    _require(type(labor_factor) in (int, float) and math.isfinite(labor_factor) and labor_factor > 0,
             "finite positive labor factor required")
    _require(hands_mode in ("daily", "constant") and 0 < timeout_seconds <= 1800,
             "bounded compiler/game settings required")
    case_list, opponent_list = list(cases), list(opponents)
    _require(case_list and opponent_list and len(opponent_list) == len(set(opponent_list)),
             "nonempty cases and distinct registered opponents required")
    catalog = scenarios.catalog()
    jobs, source_groups = [], {}
    for scenario_name, seat in case_list:
        _require(scenario_name in catalog and type(seat) is int and seat in (0, 1), "registered scenario and seat 0/1 required")
        base = catalog[scenario_name]
        world = _physical_world(base)
        for opponent_name in opponent_list:
            spec = copy.deepcopy(base)
            spec["opponent"] = opponent_name
            spec = scenarios.validate(spec)
            _, opponent_identity = scenarios.opponent(opponent_name)
            opponent_digest = E.wire_hash(opponent_identity)
            group_name, group = _source_group(world["physical_world_id"], seat, opponent_digest, checkpoint_digest)
            source_groups[group_name] = group
            job_body = {
                "scenario_name": scenario_name,
                "scenario": spec,
                "physical_world": world,
                "seat": seat,
                "opponent": opponent_name,
                "opponent_digest": opponent_digest,
                "source_group": group_name,
                "mode": mode,
                "sampler_seed": sampler_seed,
            }
            jobs.append({"index": len(jobs), "trajectory_id": _sha256_text(job_body), **job_body})
    keys = [(j["physical_world"]["physical_world_id"], j["seat"], j["opponent_digest"],
             checkpoint_digest, j["mode"], j["sampler_seed"]) for j in jobs]
    _require(len(keys) == len(set(keys)), "duplicate registered roll-in job")
    _require(len(jobs) <= max_games, f"physical-game budget exceeded: {len(jobs)} requested, {max_games} allowed")
    return E.bind({
        "schema": REGISTRATION_SCHEMA,
        "scope": SCOPE,
        "cycle_id": cycle_id,
        "checkpoint": copy.deepcopy(checkpoint),
        "source_pins": source_pins(),
        "scenario_sources": scenarios.sources(),
        "compiler": {"labor_factor": float(labor_factor), "hands_mode": hands_mode, "belief": None},
        "timeout_seconds": float(timeout_seconds),
        "max_games": max_games,
        "jobs": jobs,
        "source_groups": source_groups,
        "schema_partition": {
            "actor_visible": "explicit V0 actor allowlist + own Plan/projection inputs only",
            "behavior": "learner-selected prefix-local path and publication result",
            "trainer_only": "checkpoint/world/seat/opponent/cycle/continuation and terminal outcome",
        },
        "terminal_label_rule": "ONLY_COMPLETE_719_STEP_GAME",
        "restart_rule": "MATCHING_RECEIPT_DEDUP; FAILED_RECEIPT_IS_NOT_AUTOMATICALLY_RETRIED",
        "strength_claim": False,
    })


def _initial_continuation(agent: E.PolicyAgent) -> dict:
    policy = agent.policy
    _require(isinstance(policy, E.ScorerPolicy), "roll-in requires unchanged ScorerPolicy")
    body = {
        "schema": INITIAL_CONTINUATION_SCHEMA,
        "phase": "BEFORE_FIRST_OBSERVE",
        "game_id": agent.game_id,
        "plan": agent.plan.canonical(),
        "allocation": [],
        "started": [],
        "hold_releases": [],
        "controller": None,
        "labor_factor": agent.labor_factor,
        "hands_mode": agent.hands_mode,
        "belief": copy.deepcopy(agent.belief),
        "policy": {"kind": "SCORER", "state": policy.snapshot_state()},
    }
    return E.bind(body)


def _prefix_from_payload(value: dict) -> Prefix:
    _require(isinstance(value, dict) and set(value) == {"plan", "last_key", "edit_count", "terminated"},
             "closed prefix payload required")
    return Prefix(Plan.from_canonical(value["plan"]), value["last_key"], value["edit_count"], value["terminated"])


def _capture_path(observation: dict, before_plan: Plan, projection: dict, record: dict) -> dict:
    """Rebuild every menu from the learner's own saved prefix and verify trace joins."""
    trace = record.get("decode_trace")
    _require(isinstance(trace, dict) and trace.get("initial_prefix", {}).get("plan") == before_plan.canonical(),
             "ScorerPolicy own-prefix decode trace required")
    state = State(copy.deepcopy(observation))
    steps, previous = [], None
    for index, chosen in enumerate(trace.get("steps", [])):
        _require(isinstance(chosen, dict), "closed decoder step required")
        prefix = _prefix_from_payload(chosen["prefix"])
        _require(not prefix.terminated, "selected step cannot start terminated")
        if index == 0:
            _require(prefix.plan == before_plan and prefix.last_key == -1 and prefix.edit_count == 0,
                     "first decoder prefix must be current own Plan")
        else:
            _require(prefix.payload() == previous, "decoder prefix must equal its own prior selected continuation")
        # Recompute prefix-local legality from the canonical learner prefix, but
        # do not redo expensive phi projection: the scorer already computed phi
        # to decode.  Class identity depends only on exact continuation state.
        masks = Masks(state, prefix.plan, **copy.deepcopy(projection))
        legal_ids = ((V.STOP_ID,) if prefix.edit_count == V.MAX_EDITS_PER_EPOCH else
                     tuple(sorted(a for a in masks.legal_actions()
                                  if a == V.STOP_ID or V.plan_key(a) > prefix.last_key)))
        zeros = ((0.0,) * len(PHI_NAMES),) * len(legal_ids)
        menu = partition_menu(prefix, legal_ids, zeros, masks.resolver)
        _require(list(menu.legal_ids) == chosen["support"], "stale or noncanonical prefix support in decode trace")
        classes = [{
            "class_id": c.class_id,
            "members": [menu.legal_ids[i] for i in c.indices],
            "continuation": c.successor.payload(),
            "continuation_sha256": _sha256_text(c.successor.payload()),
        } for c in menu.classes]
        selected = next((c for c in menu.classes if c.class_id == chosen["class_id"]), None)
        _require(selected is not None and chosen["raw_id"] in [menu.legal_ids[i] for i in selected.indices],
                 "selected raw id/class not in freshly recomputed menu")
        _require(selected.successor.payload() == chosen["continuation"],
                 "selected continuation differs from fresh own-prefix class")
        steps.append({
            "index": index,
            "prefix": prefix.payload(),
            "prefix_plan_sha256": prefix.plan.digest(),
            "legal_successor_classes": classes,
            "selected": {
                "raw_id": chosen["raw_id"],
                "class_id": chosen["class_id"],
                "class_members": copy.deepcopy(chosen["class_members"]),
                "continuation": copy.deepcopy(chosen["continuation"]),
                "forced_stop": chosen["forced_stop"],
                "termination": chosen["termination"],
            },
        })
        previous = copy.deepcopy(chosen["continuation"])
    _require(steps and previous is not None and previous["terminated"], "decode path must end in STOP/cap")
    _require(previous["plan"] == trace["final_plan"] == record["after_plan"],
             "decoder path/final published Plan mismatch")
    return {
        "trace_schema": trace.get("schema"),
        "state_sha256": trace.get("state_sha256"),
        "projection_sha256": trace.get("projection_sha256"),
        "mode": trace.get("mode"),
        "temperature": trace.get("temperature"),
        "steps": steps,
        "selected_class_path": [s["selected"]["class_id"] for s in steps],
        "selected_raw_path": [s["selected"]["raw_id"] for s in steps],
        "final_plan_sha256": Plan.from_canonical(trace["final_plan"]).digest(),
        "class_path_logp": trace.get("class_path_logp"),
        "raw_path_logp": trace.get("raw_path_logp"),
        "class_path_not_final_plan_marginal": trace.get("class_path_not_final_plan_marginal"),
    }


def _failure_kind(exc: BaseException) -> str:
    text = str(exc).lower()
    if any(token in text for token in ("masked action", "integer action", "stop required", "edit cap", "key order")):
        return "ILLEGAL_POLICY_EMISSION"
    if any(token in text for token in ("disagreement", "divergence", "stale", "continuation differs", "prefix")):
        return "DIVERGENT_BEHAVIOR"
    return "DECISION_OR_EXECUTION_FAILURE"


def _state_row(*, observation: dict, before_plan: Plan, projection: dict, continuation: dict,
               record: dict | None, job: dict, registration_sha256: str, cycle_id: str,
               checkpoint_digest: str, failure: dict | None = None) -> dict:
    public_observation = T.actor_observation(State(copy.deepcopy(observation)))
    # Round-trip immediately so a saved row cannot silently depend on a raw field
    # outside the declared actor schema.
    _require(T.actor_observation(actor_state(public_observation)) == public_observation,
             "actor-only observation did not round-trip")
    actor = {
        "observation": public_observation,
        "observation_sha256": E.wire_hash(public_observation),
        "plan": before_plan.canonical(),
        "plan_sha256": before_plan.digest(),
        # These are the exact prefix-local projection inputs handed to Masks;
        # they are own configuration/memory, not rival or future state.
        "projection": copy.deepcopy(projection),
        "projection_sha256": E.wire_hash(projection),
    }
    trainer = {
        "cycle_registration_sha256": registration_sha256,
        "cycle_id": cycle_id,
        "trajectory_id": job["trajectory_id"],
        "source_group": job["source_group"],
        "physical_world_id": job["physical_world"]["physical_world_id"],
        "seat": job["seat"],
        "opponent_name": job["opponent"],
        "opponent_digest": job["opponent_digest"],
        "checkpoint_digest": checkpoint_digest,
        "native_observation_sha256": E.wire_hash(observation),
        "continuation": copy.deepcopy(continuation),
        "continuation_sha256": continuation["content_sha256"],
        "terminal_outcome": None,
    }
    behavior = None
    if record is not None:
        behavior = {
            "path": _capture_path(observation, before_plan, projection, record),
            "after_plan": copy.deepcopy(record["after_plan"]),
            "after_plan_sha256": Plan.from_canonical(record["after_plan"]).digest(),
            "publication": {
                "compiled_program_sha256": E.wire_hash(record["compiled_program"]),
                "refusals": copy.deepcopy(record["refusals"]),
                "rejected_count": record["rejected"],
                "unplaced_count": record["unplaced"],
            },
        }
    return {
        "schema": STATE_SCHEMA,
        "step": observation["step"],
        "actor_visible": actor,
        "behavior": behavior,
        "trainer_only": trainer,
        "failure": copy.deepcopy(failure),
    }


def _play_job(job: dict, registration: dict, scorer_factory: Callable[[], Any]) -> dict:
    """Execute one source-bound physical game and record its reached dawn states."""
    start, cpu_start = time.monotonic(), time.process_time()
    env = agent = None
    states, actions, joints, market_events, unit_events = [], [], [], [], []
    sold, receipts, spend, committed, attempts, unit_ops = (Counter() for _ in range(6))
    first_issue = None
    terminal_outcome = None
    status, failure = "FAILED", None
    try:
        env = scenarios.ScenarioEnvironment(job["scenario"])
        other, opponent_identity = scenarios.opponent(job["opponent"])
        _require(E.wire_hash(opponent_identity) == job["opponent_digest"], "registered opponent identity changed")
        scorer = scorer_factory()
        policy_identity = {
            "kind": "FROZEN_BC_CHECKPOINT",
            "checkpoint_digest": registration["checkpoint"]["checkpoint_digest"],
            "model_identity": copy.deepcopy(registration["checkpoint"].get("model_identity")),
            "target_or_teacher_input": False,
        }
        policy = E.ScorerPolicy(scorer, identity=policy_identity, mode=job["mode"], seed=job["sampler_seed"])
        compiler = registration["compiler"]
        agent = E.PolicyAgent(policy, labor_factor=compiler["labor_factor"], hands_mode=compiler["hands_mode"],
                              belief=compiler["belief"], game_id="v0-rollin-" + job["trajectory_id"][:16])
        with realization.market_tap(env, job["seat"]) as tap:
            while not env.done:
                if time.monotonic() - start > registration["timeout_seconds"]:
                    raise TimeoutError("bounded V0 roll-in game deadline")
                step, seat = env.step_index, job["seat"]
                observation = env.agent_observation(seat)
                before_plan = agent.plan.copy()
                projection = copy.deepcopy(agent.projection_kw())
                pending = None
                if step % 24 == 0:
                    if step == 0:
                        continuation = _initial_continuation(agent)
                    else:
                        continuation = E.snapshot_agent(agent, copy.deepcopy(observation))
                    pending = (copy.deepcopy(observation), before_plan, projection, continuation)
                old_records = len(agent.records)
                try:
                    action = agent(copy.deepcopy(observation), copy.deepcopy(env.configuration))
                except Exception as exc:
                    if pending is not None:
                        issue = {"kind": _failure_kind(exc), "step": step, "type": type(exc).__name__, "message": str(exc)}
                        states.append(_state_row(observation=pending[0], before_plan=pending[1], projection=pending[2],
                                                 continuation=pending[3], record=None, job=job,
                                                 registration_sha256=registration["content_sha256"],
                                                 cycle_id=registration["cycle_id"],
                                                 checkpoint_digest=registration["checkpoint"]["checkpoint_digest"],
                                                 failure=issue))
                        if first_issue is None:
                            first_issue = copy.deepcopy(issue)
                    raise
                if step % 24 == 0:
                    _require(len(agent.records) == old_records + 1, "dawn publication record missing")
                    record = copy.deepcopy(agent.records[-1])
                    try:
                        state_row = _state_row(
                            observation=pending[0], before_plan=pending[1], projection=pending[2],
                            continuation=pending[3], record=record, job=job,
                            registration_sha256=registration["content_sha256"],
                            cycle_id=registration["cycle_id"],
                            checkpoint_digest=registration["checkpoint"]["checkpoint_digest"],
                        )
                    except Exception as exc:
                        # A published action whose trace cannot be joined back to the exact
                        # learner prefix is evidence.  Preserve the reached actor state and
                        # first divergence, but never repair or relabel it as a usable path.
                        issue = {"kind": _failure_kind(exc), "step": step,
                                 "type": type(exc).__name__, "message": str(exc)}
                        states.append(_state_row(
                            observation=pending[0], before_plan=pending[1], projection=pending[2],
                            continuation=pending[3], record=None, job=job,
                            registration_sha256=registration["content_sha256"],
                            cycle_id=registration["cycle_id"],
                            checkpoint_digest=registration["checkpoint"]["checkpoint_digest"],
                            failure=issue,
                        ))
                        if first_issue is None:
                            first_issue = copy.deepcopy(issue)
                        raise
                    states.append(state_row)
                    if record["refusals"] and first_issue is None:
                        first_issue = {"kind": "PUBLICATION_REFUSAL", "step": step,
                                       "refusals": copy.deepcopy(record["refusals"])}
                else:
                    _require(len(agent.records) == old_records, "intraday publication violated")
                joint = [None, None]
                joint[seat] = action
                joint[1-seat] = other(env.agent_observation(1-seat), copy.deepcopy(env.configuration))
                tap["commits"], tap["pre_market"], tap["unit_effects"] = {}, None, []
                env.step(joint)
                actions.append(copy.deepcopy(action)); joints.append(copy.deepcopy(joint))
                for (op, item), event in tap["commits"].items():
                    label = f"{op}:{item}"
                    committed[label] += event["units"]
                    attempts[label] += event["units"] + event["failed_unit_attempts"]
                    market_events.append({"step": step, **copy.deepcopy(event)})
                    if op == "SELL":
                        sold[item] += event["units"]; receipts[item] += event["cash_delta"]
                    else:
                        spend[label] -= event["cash_delta"]
                for event in tap["unit_effects"]:
                    unit_ops[event["engine_received_action"][0]] += 1
                    unit_events.append({"step": step, **copy.deepcopy(event)})
        _require(env.step_index == V.LAST_STEP + 1 and len(states) == len(DAWNS),
                 "incomplete game cannot receive terminal labels")
        final = env.agent_observation(job["seat"])
        cash = [f["money"] for f in final["farms"]]
        own, rival = cash[job["seat"]], cash[1-job["seat"]]
        margin = own - rival
        terminal_outcome = {"own_final_cash": own, "rival_final_cash": rival, "margin": margin,
                            "wdl": "WIN" if margin > 0 else "LOSS" if margin < 0 else "DRAW"}
        for row in states:
            row["trainer_only"]["terminal_outcome"] = copy.deepcopy(terminal_outcome)
        status = "COMPLETE"
    except Exception as exc:
        failure = {"kind": _failure_kind(exc), "step": None if env is None else env.step_index,
                   "type": type(exc).__name__, "message": str(exc)}
        if first_issue is None:
            first_issue = copy.deepcopy(failure)
        terminal_outcome = None
        for row in states:
            row["trainer_only"]["terminal_outcome"] = None

    realization_doc = {
        "sold": dict(sold), "receipts": dict(receipts), "spend": dict(spend),
        "committed_units": dict(committed), "attempted_units": dict(attempts),
        "executed_unit_operations": dict(unit_ops),
        "publication_refusals": sum(len(r.get("behavior", {}).get("publication", {}).get("refusals", []))
                                     for r in states if r.get("behavior")),
        "scope": "NATIVE_COMMITS_AND_UNIT_DELTAS; NOT_PROJECTED_FULFILLMENT",
    }
    return E.bind({
        "schema": GAME_SCHEMA,
        "scope": SCOPE,
        "registration_sha256": registration["content_sha256"],
        "job": copy.deepcopy(job),
        "status": status,
        "physical_games_started": int(env is not None),
        "physical_games_completed": int(status == "COMPLETE"),
        "physical_steps": 0 if env is None else env.step_index,
        "states": states,
        "terminal_outcome": terminal_outcome,
        "first_issue": first_issue,
        "failure": failure,
        "realization": realization_doc,
        "market_events": market_events,
        "unit_events": unit_events,
        "own_actions_sha256": E.wire_hash(actions),
        "joint_actions_sha256": E.wire_hash(joints),
        "terminal_labels_on_incomplete": False,
        "wall_seconds": time.monotonic() - start,
        "cpu_seconds": time.process_time() - cpu_start,
        "strength_claim": False,
    })


def _journal(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(E.wire(E.bind(value)) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def collect_registered(registration: dict, output_dir: str | Path, scorer_factory: Callable[[], Any],
                       *, game_runner: Callable[[dict, dict, Callable[[], Any]], dict] = _play_job) -> dict:
    """Run/resume one immutable registration; matching receipts are deterministic dedup."""
    _require(registration == E.bind(registration) and registration.get("schema") == REGISTRATION_SCHEMA,
             "bound roll-in registration required")
    _require(registration["source_pins"] == source_pins() and registration["scenario_sources"] == scenarios.sources(),
             "roll-in source closure changed after registration")
    output = _safe_output(output_dir)
    plan_path = output / "PLAN.json"
    if output.exists():
        _require(output.is_dir() and plan_path.is_file(), "existing roll-in directory requires PLAN.json")
        existing = _load_json(plan_path)
        _require(existing == registration, "existing roll-in registration mismatch")
    else:
        output.mkdir(parents=True)
        E.write_new(plan_path, registration)
    receipts = []
    for job in registration["jobs"]:
        path = output / f"game-{job['index']:04d}.json"
        if path.exists():
            row = _load_json(path)
            _require(row == E.bind(row) and row.get("registration_sha256") == registration["content_sha256"]
                     and row.get("job") == job, "existing game receipt mismatch")
            receipts.append({"index": job["index"], "status": row["status"], "deduplicated": True,
                             "path": path.name, "sha256": file_sha(path), "content_sha256": row["content_sha256"]})
            continue
        _journal(output / "JOURNAL.jsonl", {"event": "START", "job": job,
                                             "registration_sha256": registration["content_sha256"]})
        row = game_runner(job, registration, scorer_factory)
        _require(row == E.bind(row) and row.get("registration_sha256") == registration["content_sha256"]
                 and row.get("job") == job, "game runner returned unbound/mismatched receipt")
        E.write_new(path, row)
        receipt = {"index": job["index"], "status": row["status"], "deduplicated": False,
                   "path": path.name, "sha256": file_sha(path), "content_sha256": row["content_sha256"]}
        receipts.append(receipt)
        _journal(output / "JOURNAL.jsonl", {"event": "RESULT", "job": job, "receipt": receipt})
    rows = [_load_json(output / r["path"]) for r in receipts]
    complete = sum(r["status"] == "COMPLETE" for r in rows)
    failed = len(rows) - complete
    result_body = {
        "schema": SCHEMA,
        "scope": SCOPE,
        "registration_sha256": registration["content_sha256"],
        "cycle_id": registration["cycle_id"],
        "requested_games": len(registration["jobs"]),
        "attempted_or_deduplicated_games": len(rows),
        "complete_games": complete,
        "failed_games": failed,
        "physical_games_started_this_or_prior_run": sum(r["physical_games_started"] for r in rows),
        "source_groups": sorted({j["source_group"] for j in registration["jobs"]}),
        "receipts": [{k: v for k, v in receipt.items() if k != "deduplicated"} for receipt in receipts],
        "terminal_label_rule": registration["terminal_label_rule"],
        "strength_claim": False,
        "status": "COMPLETE" if len(rows) == len(registration["jobs"]) else "INCOMPLETE",
    }
    result = E.bind(result_body)
    result_path = output / "RESULT.json"
    if result_path.exists():
        old = _load_json(result_path)
        _require(old == result, "existing RESULT disagrees with deterministic dedup replay")
        return old
    return E.write_new(result_path, result)


def _parse_case(value: str) -> tuple[str, int]:
    try:
        name, seat = value.rsplit(":", 1)
        seat_i = int(seat)
    except Exception as exc:
        raise argparse.ArgumentTypeError("case must be SCENARIO:SEAT") from exc
    if not name or seat_i not in (0, 1):
        raise argparse.ArgumentTypeError("case must be SCENARIO:0 or SCENARIO:1")
    return name, seat_i


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--cycle-id", required=True)
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--checkpoint-dir", required=True)
    collect.add_argument("--checkpoint-sha256", required=True)
    collect.add_argument("--case", action="append", type=_parse_case, required=True)
    collect.add_argument("--opponent", action="append", required=True)
    collect.add_argument("--mode", choices=E.MODES, default="greedy_class")
    collect.add_argument("--sampler-seed", type=int, default=0)
    collect.add_argument("--max-games", type=int, default=4)
    collect.add_argument("--labor-factor", type=float, default=4.0)
    collect.add_argument("--hands-mode", choices=("daily", "constant"), default="daily")
    collect.add_argument("--timeout-seconds", type=float, default=180.0)
    collect.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    checkpoint = load_checkpoint(args.checkpoint_dir, args.checkpoint_sha256, device=args.device)
    registration = register_cycle(cycle_id=args.cycle_id, checkpoint=checkpoint.provenance, cases=args.case,
                                  opponents=args.opponent, mode=args.mode, sampler_seed=args.sampler_seed,
                                  max_games=args.max_games, labor_factor=args.labor_factor,
                                  hands_mode=args.hands_mode, timeout_seconds=args.timeout_seconds)
    result = collect_registered(registration, args.output_dir, lambda: checkpoint.scorer)
    print(E.wire(result))
    return 0 if result["failed_games"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
