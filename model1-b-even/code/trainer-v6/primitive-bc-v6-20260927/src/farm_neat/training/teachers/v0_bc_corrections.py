"""Fixture-gated V0 BC corrections from learner-owned roll-in prefixes.

This build consumes an accepted ``v0_bc_rollin`` directory and a *synthetic
fixture oracle* document.  It never queries the native bounded search itself:
that is intentionally deferred until the separate V0-TEACHER-IMPROVE gate.

The correction root is always the exact dawn row reached by the learner.  Every
candidate is replayed from ``row['actor_visible']['plan']`` with that row's
projection inputs, recomputing prefix-local legality and successor classes after
each selected edit.  The saved learner path is independently replayed the same
way before it can be used as the paired control label.  There is no API for a
teacher prefix.

Only development fixture outcomes may select a correction.  Candidate outcome,
paired game identity, continuation, raw edit/search trace and oracle notes stay
under ``trainer_only``.  The supervised examples contain only a byte-identical
actor-visible input plus either the learner or corrected successor-class target.
A strict objective improvement is required; ties, harmful candidates and
UNKNOWN/no-certificate candidates are retained without becoming labels.

Command surface (fixture diagnostics only; no strength claim)::

  PYTHONPATH=src python3 -B -m farm_neat.training.teachers.v0_bc_corrections build \\
    --rollin-dir work/runs/v0-rollin/dev-cycle \\
    --fixture-oracle FIXTURE.json --output CORRECTIONS.json

``FIXTURE.json`` is a bound ``farm-v0-bc-correction-fixture-oracle-v1`` object::

  {"schema": ..., "selection_split": "development",
   "rollin_registration_sha256": "...", "objective": "MARGIN",
   "cases": [{"game_index": 0, "state_index": 0,
              "state_sha256": "...", "selection_reason": "DISAGREEMENT",
              "candidates": [
                {"candidate_id": "keep", "role": "KEEP", "actions": [0],
                 "certificate": "PAIRED_FIXTURE", "search_trace": {},
                 "game": {"identity": {"fixture_game_id": "...", ...},
                          "terminal_outcome": {...}}},
                {"candidate_id": "unknown", "role": "EDIT", "actions": [..., 0],
                 "certificate": "UNKNOWN", "search_trace": {},
                 "unknown_reason": "no fixture certificate"}]}]}

Candidate game identities have one free field, ``fixture_game_id``; every other
field is an exact context binding produced by :func:`pair_context`.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import Masks
from farm_neat.evaluation.common import file_sha
from farm_neat.evaluation import v0_policy as E
from farm_neat.policy.v0_scorer import Prefix
from farm_neat.training import v0_bc_rollin as R

SCHEMA = "farm-v0-bc-corrections-v1"
CASE_SCHEMA = SCHEMA + "-case"
TARGET_SCHEMA = SCHEMA + "-successor-class-target"
TRACE_SCHEMA = SCHEMA + "-replay-trace"
FIXTURE_SCHEMA = "farm-v0-bc-correction-fixture-oracle-v1"
OBJECTIVES = ("MARGIN", "WDL_SCORE")
SCOPE = "DEVELOPMENT_FIXTURE_CORRECTION_CONTRACT; REAL_BOUNDED_SEARCH_NOT_RUN; NOT_STRENGTH"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe(path: str | Path) -> Path:
    path = Path(path).resolve()
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlinked path forbidden")
    return path


def _load_object(path: str | Path) -> dict:
    path = _safe(path)
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"object JSON required: {path}")
    return value


def source_pins() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src/farm_neat/training/v0_bc_rollin.py",
        ROOT / "src/farm_neat/policy/v0_scorer.py",
        ROOT / "src/farm_neat/economy/v0/plan.py",
        ROOT / "src/farm_neat/economy/v0/projection.py",
        ROOT / "src/farm_neat/evaluation/v0_policy.py",
    )
    return {str(path.relative_to(ROOT)): file_sha(path) for path in paths}


def _validate_actor(actor: dict) -> None:
    fields = {"observation", "observation_sha256", "plan", "plan_sha256", "projection", "projection_sha256"}
    _require(isinstance(actor, dict) and set(actor) == fields, "closed roll-in actor-visible record required")
    _require(E.wire_hash(actor["observation"]) == actor["observation_sha256"], "actor observation hash mismatch")
    _require(Plan.from_canonical(actor["plan"]).digest() == actor["plan_sha256"], "actor Plan hash mismatch")
    _require(E.wire_hash(actor["projection"]) == actor["projection_sha256"], "actor projection hash mismatch")
    _require(set(actor["projection"]) == {"labor_factor", "belief", "hold_releases"},
             "unexpected projection inputs")
    R.actor_state(actor["observation"])


def _action_sequence(actions: Any) -> list[int]:
    _require(isinstance(actions, list) and actions and len(actions) <= V.MAX_EDITS_PER_EPOCH + 1,
             "bounded explicit action path required")
    last = -1
    for index, action in enumerate(actions):
        _require(type(action) is int and 0 <= action < V.N_ACTIONS, "integer V0 action id required")
        if action == V.STOP_ID:
            _require(index == len(actions) - 1, "STOP must terminate candidate path")
        else:
            key = V.plan_key(action)
            _require(key > last, "strict increasing canonical keys required")
            last = key
    _require(actions[-1] == V.STOP_ID, "explicit final STOP required")
    return list(actions)


def replay_actions(actor_visible: dict, actions: list[int]) -> dict:
    """Replay a raw representative path from the exact learner-owned dawn root.

    Raw IDs are retained only in this replay proof.  The exported target below is
    the successor-class trajectory, so an alias representative never becomes a CE
    label.  Prefix support is recomputed after every edit from the reached Plan.
    """
    _validate_actor(actor_visible)
    actions = _action_sequence(actions)
    state = R.actor_state(copy.deepcopy(actor_visible["observation"]))
    projection = copy.deepcopy(actor_visible["projection"])
    prefix = Prefix(Plan.from_canonical(actor_visible["plan"]), -1, 0, False)
    steps = []
    for index, action in enumerate(actions):
        _require(not prefix.terminated, "candidate action after terminated prefix")
        masks = Masks(state, prefix.plan, **copy.deepcopy(projection))
        before = prefix.payload()
        if action == V.STOP_ID:
            successor = Prefix(prefix.plan, prefix.last_key, prefix.edit_count, True)
        else:
            _require(prefix.edit_count < V.MAX_EDITS_PER_EPOCH, "non-STOP action after edit cap")
            _require(V.plan_key(action) > prefix.last_key, "candidate violates prefix key cursor")
            _require(masks.legal(action), f"candidate masked at own prefix {index}: {V.describe(action)}")
            successor = Prefix(prefix.plan.apply(action, masks.resolver), V.plan_key(action), prefix.edit_count + 1)
        after = successor.payload()
        signature = json.dumps(after, sort_keys=True, separators=(",", ":"), allow_nan=False)
        class_id = hashlib.sha256(signature.encode()).hexdigest()
        steps.append({
            "index": index,
            "prefix": before,
            "raw_id": action,
            "class_id": class_id,
            "continuation": copy.deepcopy(after),
            "forced_stop": action == V.STOP_ID and prefix.edit_count == V.MAX_EDITS_PER_EPOCH,
        })
        prefix = successor
    _require(prefix.terminated, "candidate did not terminate")
    body = {
        "schema": TRACE_SCHEMA,
        "actor_visible_sha256": E.wire_hash(actor_visible),
        "initial_plan_sha256": actor_visible["plan_sha256"],
        "actions": actions,
        "class_path": [step["class_id"] for step in steps],
        "steps": steps,
        "final_plan": prefix.plan.canonical(),
        "final_plan_sha256": prefix.plan.digest(),
    }
    return E.bind(body)


def target_from_trace(trace: dict) -> dict:
    _require(trace == E.bind(trace) and trace.get("schema") == TRACE_SCHEMA, "bound replay trace required")
    body = {
        "schema": TARGET_SCHEMA,
        "class_path": list(trace["class_path"]),
        "steps": [{
            "index": step["index"],
            "class_id": step["class_id"],
            "continuation": copy.deepcopy(step["continuation"]),
            "forced_stop": step["forced_stop"],
        } for step in trace["steps"]],
        "successor_plan": copy.deepcopy(trace["final_plan"]),
        "successor_plan_sha256": trace["final_plan_sha256"],
    }
    return E.bind(body)


def _verify_saved_learner_path(state_row: dict) -> tuple[dict, dict]:
    behavior = state_row.get("behavior")
    _require(isinstance(behavior, dict) and isinstance(behavior.get("path"), dict),
             "selected correction state requires a learner behavior path")
    saved = behavior["path"]
    raw = saved.get("selected_raw_path")
    trace = replay_actions(state_row["actor_visible"], raw)
    _require(trace["class_path"] == saved.get("selected_class_path"), "saved learner class path replay mismatch")
    _require(trace["final_plan_sha256"] == saved.get("final_plan_sha256"), "saved learner final Plan replay mismatch")
    _require(len(trace["steps"]) == len(saved.get("steps", [])), "saved learner trace length mismatch")
    for fresh, old in zip(trace["steps"], saved["steps"]):
        _require(fresh["prefix"] == old["prefix"], "saved learner prefix reset/divergence")
        selected = old["selected"]
        _require(fresh["raw_id"] == selected["raw_id"] and fresh["class_id"] == selected["class_id"]
                 and fresh["continuation"] == selected["continuation"], "saved learner step replay mismatch")
    return trace, target_from_trace(trace)


def _terminal(value: dict) -> dict:
    fields = {"own_final_cash", "rival_final_cash", "margin", "wdl"}
    _require(isinstance(value, dict) and set(value) == fields, "closed terminal outcome required")
    own, rival, margin = value["own_final_cash"], value["rival_final_cash"], value["margin"]
    _require(all(type(x) in (int, float) and math.isfinite(x) for x in (own, rival, margin)),
             "finite terminal cash/margin required")
    _require(math.isclose(margin, own - rival, rel_tol=0.0, abs_tol=1e-9), "terminal margin mismatch")
    expected = "WIN" if margin > 0 else "LOSS" if margin < 0 else "DRAW"
    _require(value["wdl"] == expected, "terminal WDL mismatch")
    return copy.deepcopy(value)


def _objective(outcome: dict, objective: str) -> float:
    if objective == "MARGIN":
        return float(outcome["margin"])
    if objective == "WDL_SCORE":
        return {"LOSS": 0.0, "DRAW": 0.5, "WIN": 1.0}[outcome["wdl"]]
    raise ValueError("unsupported correction objective")


def _rollin_game_identity(game: dict, state_row: dict, state_index: int) -> dict:
    trainer = state_row["trainer_only"]
    return E.bind({
        "kind": "ROLLIN_LEARNER_CONTROL",
        "game_content_sha256": game["content_sha256"],
        "trajectory_id": trainer["trajectory_id"],
        "state_index": state_index,
        "step": state_row["step"],
        "physical_world_id": trainer["physical_world_id"],
        "seat": trainer["seat"],
        "opponent_digest": trainer["opponent_digest"],
        "continuation_sha256": trainer["continuation_sha256"],
        "state_sha256": E.wire_hash(state_row),
        "actor_visible_sha256": E.wire_hash(state_row["actor_visible"]),
    })


def pair_context(game: dict, state_row: dict, state_index: int) -> dict:
    """Exact fields a fixture candidate game must share with learner control."""
    control = _rollin_game_identity(game, state_row, state_index)
    return {key: copy.deepcopy(control[key]) for key in (
        "physical_world_id", "seat", "opponent_digest", "continuation_sha256",
        "state_sha256", "actor_visible_sha256", "step")}


def _validate_candidate_game(game: dict, context: dict) -> tuple[dict, dict]:
    _require(isinstance(game, dict) and set(game) == {"identity", "terminal_outcome"},
             "closed fixture candidate game required")
    identity = game["identity"]
    _require(isinstance(identity, dict) and set(identity) == {"fixture_game_id", *context.keys()},
             "closed matched fixture game identity required")
    _require(isinstance(identity["fixture_game_id"], str) and identity["fixture_game_id"],
             "nonempty fixture game id required")
    for key, expected in context.items():
        _require(identity[key] == expected, f"candidate/control {key} mismatch")
    return E.bind(copy.deepcopy(identity)), _terminal(game["terminal_outcome"])


def load_rollin_batch(directory: str | Path) -> tuple[dict, dict, dict[int, dict]]:
    """Load a source-bound development roll-in and verify receipt/file joins."""
    directory = _safe(directory)
    _require(directory.is_dir(), "roll-in directory required")
    registration = _load_object(directory / "PLAN.json")
    result = _load_object(directory / "RESULT.json")
    _require(registration == E.bind(registration) and registration.get("schema") == R.REGISTRATION_SCHEMA,
             "bound V0 BC roll-in registration required")
    _require(registration.get("scope") == R.SCOPE, "development roll-in scope required")
    _require(registration.get("source_pins") == R.source_pins(), "roll-in source closure changed")
    _require(result == E.bind(result) and result.get("schema") == R.SCHEMA,
             "bound V0 BC roll-in result required")
    _require(result.get("registration_sha256") == registration["content_sha256"], "roll-in result/plan mismatch")
    jobs = {job["index"]: job for job in registration["jobs"]}
    games: dict[int, dict] = {}
    for receipt in result.get("receipts", []):
        _require(isinstance(receipt, dict) and set(receipt) == {"index", "status", "path", "sha256", "content_sha256"},
                 "closed roll-in receipt required")
        index = receipt["index"]
        _require(index in jobs and index not in games, "unknown/duplicate roll-in game index")
        _require(Path(receipt["path"]).name == receipt["path"], "roll-in receipt path must be local basename")
        path = _safe(directory / receipt["path"])
        _require(path.is_file() and file_sha(path) == receipt["sha256"], "roll-in receipt file hash mismatch")
        game = _load_object(path)
        _require(game == E.bind(game) and game.get("schema") == R.GAME_SCHEMA,
                 "bound V0 BC roll-in game required")
        _require(game["content_sha256"] == receipt["content_sha256"], "roll-in content receipt mismatch")
        _require(game.get("registration_sha256") == registration["content_sha256"] and game.get("job") == jobs[index],
                 "roll-in game registration/job mismatch")
        games[index] = game
    _require(len(games) == len(result.get("receipts", [])), "roll-in receipt join failure")
    return registration, result, games


def _process_case(game: dict, case: dict, objective: str) -> dict:
    fields = {"game_index", "state_index", "state_sha256", "selection_reason", "candidates"}
    _require(isinstance(case, dict) and set(case) == fields, "closed fixture case required")
    state_index = case["state_index"]
    _require(type(state_index) is int and 0 <= state_index < len(game.get("states", [])), "valid state index required")
    state_row = game["states"][state_index]
    _require(isinstance(state_row, dict) and state_row.get("schema") == R.STATE_SCHEMA, "roll-in state row required")
    _require(E.wire_hash(state_row) == case["state_sha256"], "fixture state preimage mismatch")
    _validate_actor(state_row["actor_visible"])
    trainer = state_row.get("trainer_only")
    _require(isinstance(trainer, dict) and trainer.get("continuation_sha256") == trainer.get("continuation", {}).get("content_sha256"),
             "exact bound learner continuation required")
    _require(trainer.get("terminal_outcome") is not None, "complete development learner control outcome required")
    control_outcome = _terminal(trainer["terminal_outcome"])
    if game.get("terminal_outcome") is not None:
        _require(_terminal(game["terminal_outcome"]) == control_outcome, "state/game learner terminal outcome mismatch")
    learner_trace, learner_target = _verify_saved_learner_path(state_row)
    control_identity = _rollin_game_identity(game, state_row, state_index)
    context = pair_context(game, state_row, state_index)
    candidates = case["candidates"]
    _require(isinstance(candidates, list) and candidates, "bounded fixture candidates required")
    _require(len(candidates) <= 257, "too many fixture candidates")
    seen_ids: set[str] = set()
    effects = []
    best: tuple[float, int, dict, dict] | None = None
    for order, candidate in enumerate(candidates):
        common = {"candidate_id", "role", "actions", "certificate", "search_trace"}
        _require(isinstance(candidate, dict), "fixture candidate object required")
        certificate = candidate.get("certificate")
        expected_fields = common | ({"game"} if certificate == "PAIRED_FIXTURE" else {"unknown_reason"})
        _require(set(candidate) == expected_fields, "closed fixture candidate fields required")
        cid = candidate["candidate_id"]
        _require(isinstance(cid, str) and cid and cid not in seen_ids, "unique nonempty candidate id required")
        seen_ids.add(cid)
        role = candidate["role"]
        _require(role in ("KEEP", "EDIT"), "fixture candidate role must be KEEP or EDIT")
        actions = _action_sequence(candidate["actions"])
        if role == "KEEP":
            _require(actions == [V.STOP_ID], "KEEP candidate must be STOP only")
        _require(isinstance(candidate["search_trace"], dict), "trainer-only fixture search trace object required")
        trace = replay_actions(state_row["actor_visible"], actions)
        target = target_from_trace(trace)
        row = {
            "candidate_id": cid,
            "role": role,
            "certificate": certificate,
            "target_sha256": target["content_sha256"],
            "status": None,
            "objective_delta": None,
            "pair": None,
            "search_trace": copy.deepcopy(candidate["search_trace"]),
            "replay_trace": trace,
        }
        if certificate == "UNKNOWN":
            _require(isinstance(candidate["unknown_reason"], str) and candidate["unknown_reason"],
                     "UNKNOWN candidate reason required")
            row["status"] = "UNKNOWN"
            row["unknown_reason"] = candidate["unknown_reason"]
        elif certificate == "PAIRED_FIXTURE":
            candidate_identity, outcome = _validate_candidate_game(candidate["game"], context)
            delta = _objective(outcome, objective) - _objective(control_outcome, objective)
            status = "IMPROVES" if delta > 0 else "HARMFUL" if delta < 0 else "TIE"
            row.update({
                "status": status,
                "objective_delta": delta,
                "pair": {
                    "evidence_kind": "SYNTHETIC_FIXTURE",
                    "control_game_identity": control_identity,
                    "candidate_game_identity": candidate_identity,
                    "control_terminal_outcome": copy.deepcopy(control_outcome),
                    "candidate_terminal_outcome": outcome,
                },
            })
            if status == "IMPROVES" and target["content_sha256"] != learner_target["content_sha256"]:
                key = (delta, -order)
                if best is None or key > (best[0], -best[1]):
                    best = (delta, order, target, row)
        else:
            raise ValueError("certificate must be PAIRED_FIXTURE or UNKNOWN")
        effects.append(row)

    chosen_target = None if best is None else best[2]
    chosen_id = None if best is None else best[3]["candidate_id"]
    actor_a = copy.deepcopy(state_row["actor_visible"])
    actor_b = copy.deepcopy(state_row["actor_visible"])
    _require(E.wire(actor_a) == E.wire(actor_b), "learner/corrected actor-visible bytes changed")
    supervision = {
        "learner": {"actor_visible": actor_a, "target": learner_target},
        "corrected": None if chosen_target is None else {"actor_visible": actor_b, "target": chosen_target},
    }
    body = {
        "schema": CASE_SCHEMA,
        "source": {
            "game_index": case["game_index"],
            "state_index": state_index,
            "state_sha256": case["state_sha256"],
            "trajectory_id": trainer["trajectory_id"],
            "source_group": trainer["source_group"],
            "step": state_row["step"],
        },
        "decision": "CORRECTION" if chosen_target is not None else "NO_VERIFIED_IMPROVEMENT",
        "selected_candidate_id": chosen_id,
        "objective": objective,
        "supervision": supervision,
        "actor_visible_sha256": E.wire_hash(state_row["actor_visible"]),
        "trainer_only": {
            "selection_reason": case["selection_reason"],
            "continuation": copy.deepcopy(trainer["continuation"]),
            "continuation_sha256": trainer["continuation_sha256"],
            "control_game_identity": control_identity,
            "control_terminal_outcome": control_outcome,
            "learner_replay_trace": learner_trace,
            "effects": effects,
        },
    }
    return E.bind(body)


def build_batch(rollin_dir: str | Path, fixture_oracle: dict) -> dict:
    """Build fixture-only corrections; zero verified improvements is valid output."""
    registration, result, games = load_rollin_batch(rollin_dir)
    _require(fixture_oracle == E.bind(fixture_oracle) and fixture_oracle.get("schema") == FIXTURE_SCHEMA,
             "bound fixture oracle document required")
    fields = {"schema", "selection_split", "rollin_registration_sha256", "objective", "cases", "content_sha256"}
    _require(set(fixture_oracle) == fields, "closed fixture oracle document required")
    _require(fixture_oracle["selection_split"] == "development",
             "held-out/final outcomes cannot select correction labels")
    _require(fixture_oracle["rollin_registration_sha256"] == registration["content_sha256"],
             "fixture/roll-in registration mismatch")
    objective = fixture_oracle["objective"]
    _require(objective in OBJECTIVES, "declared paired competitive objective required")
    cases = fixture_oracle["cases"]
    _require(isinstance(cases, list) and len(cases) <= 1024, "bounded fixture case list required")
    seen: set[tuple[int, int]] = set()
    rows = []
    for case in cases:
        game_index = case.get("game_index") if isinstance(case, dict) else None
        _require(type(game_index) is int and game_index in games, "fixture game index not in roll-in batch")
        key = (game_index, case.get("state_index"))
        _require(key not in seen, "duplicate fixture state selection")
        seen.add(key)
        rows.append(_process_case(games[game_index], case, objective))
    statuses = [effect["status"] for row in rows for effect in row["trainer_only"]["effects"]]
    body = {
        "schema": SCHEMA,
        "scope": SCOPE,
        "source_rollin": {
            "directory": str(_safe(rollin_dir)),
            "registration_sha256": registration["content_sha256"],
            "result_sha256": result["content_sha256"],
            "complete_games": result.get("complete_games"),
        },
        "fixture_oracle_sha256": fixture_oracle["content_sha256"],
        "selection_split": "development",
        "objective": objective,
        "cases": rows,
        "counts": {
            "selected_states": len(rows),
            "corrections": sum(row["decision"] == "CORRECTION" for row in rows),
            "no_verified_improvement": sum(row["decision"] == "NO_VERIFIED_IMPROVEMENT" for row in rows),
            "improves": statuses.count("IMPROVES"),
            "ties": statuses.count("TIE"),
            "harmful": statuses.count("HARMFUL"),
            "unknown": statuses.count("UNKNOWN"),
        },
        "source_pins": source_pins(),
        "strength_claim": False,
        "real_bounded_search": "NOT_RUN_WAITING_FOR_V0_TEACHER_IMPROVE",
    }
    return E.bind(body)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--rollin-dir", required=True)
    build.add_argument("--fixture-oracle", required=True)
    build.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    fixture = _load_object(args.fixture_oracle)
    result = build_batch(args.rollin_dir, fixture)
    E.write_new(_safe(args.output), result)
    print(json.dumps({"output": str(Path(args.output).resolve()), "content_sha256": result["content_sha256"],
                      "counts": result["counts"], "strength_claim": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
