"""Tiny source-bound witness for the first semantic divergence on a V0 edit path.

This module is deliberately not a fuzzer.  It localizes one disagreement between
an exact scorer-menu reference and one existing consumer path, records only the
first differing edit, greedily removes irrelevant earlier edits / continuation
state, and replays the reduced JSON witness in a fresh process.

The two production adapters are intentionally narrow:

* ``evaluator`` executes :func:`farm_neat.evaluation.v0_policy.validate_edits`;
* ``teacher`` executes :func:`farm_neat.training.teachers.v0_oracle.trace_edits`.

A single ``successor_class`` test-double injection exists solely to prove that
capture -> reduction -> fresh-process replay preserves the first differing
field.  It is provenance in the witness, never a normalization of a real
consumer result.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Literal

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.epochs import KeepPolicy
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import Masks, State
from farm_neat.evaluation.common import file_sha
from farm_neat.evaluation.v0_continuation import V0Continuation
from farm_neat.policy import v0_scorer as scorer
from farm_neat.training.v0_targets import actor_observation

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "farm-v0-divergence-witness"
VERSION = 1
CONSUMERS = frozenset({"evaluator", "teacher"})
TEST_DOUBLE_FIELDS = frozenset({None, "successor_class"})

WITNESS_FIELDS = frozenset({
    "schema", "version", "source_pins", "scenario", "seat",
    "public_observation_sha256", "continuation_memory_sha256", "continuation",
    "prefix", "candidate_action", "reference", "candidate",
    "first_differing_field", "replay_command",
})
SCENARIO_FIELDS = frozenset({"id", "consumer", "observation", "history_actions", "test_double"})
SEMANTIC_FIELDS = frozenset({"legality", "successor_class", "continuation"})
STOP_FIELDS = frozenset({"successor", "forced_stop", "termination"})
PREFIX_FIELDS = frozenset({"plan", "last_key", "edit_count", "terminated"})
DIFF_ORDER = (
    "legality",
    "successor_class",
    "continuation.successor",
    "continuation.forced_stop",
    "continuation.termination",
)


class NoDivergence(ValueError):
    """The supplied paths agree through the requested candidate action."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _json_copy(value: Any) -> Any:
    # Also proves the value is finite and JSON-safe.
    return json.loads(canonical(value))


def _read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key in divergence witness")
            result[key] = value
        return result

    def bad(value):
        raise ValueError("nonfinite JSON value in divergence witness")

    value = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=bad)
    if not isinstance(value, dict):
        raise ValueError("divergence witness object required")
    return value


def source_pins() -> dict[str, str]:
    """Pin the exact semantic implementation used by capture and replay.

    v0_policy.source_pins already closes over the economy / engine / evaluator
    sources on which prefix legality depends.  Add the teacher, actor-public
    projection and this witness implementation itself because they are direct
    replay code paths not owned by that closure.
    """
    from farm_neat.evaluation import v0_policy

    pins = dict(v0_policy.source_pins())
    for rel in (
        "src/farm_neat/training/teachers/v0_oracle.py",
        "src/farm_neat/training/v0_targets.py",
        "src/farm_neat/evaluation/v0_divergence.py",
    ):
        pins[rel] = file_sha(ROOT / rel)
    return dict(sorted(pins.items()))


def _continuation_dict(value: V0Continuation | dict, observation=None) -> dict:
    if isinstance(value, V0Continuation):
        memory = value.to_dict()
    elif isinstance(value, dict):
        memory = V0Continuation.from_dict(value).to_dict()
    else:
        raise TypeError("V0Continuation or closed continuation mapping required")
    # Restore is part of witness admission: malformed-but-shaped memory is not
    # allowed to become a replay boundary.
    V0Continuation.from_dict(memory).restore(KeepPolicy(), observation=observation)
    return memory


def _projection_kwargs(memory: dict, observation=None) -> dict:
    agent = V0Continuation.from_dict(memory).restore(KeepPolicy(), observation=observation)
    return agent.projection_kw()


def _prefix_from_payload(value: dict) -> scorer.Prefix:
    if not isinstance(value, dict) or set(value) != PREFIX_FIELDS:
        raise ValueError("closed canonical prefix required")
    prefix = scorer.Prefix(
        Plan.from_canonical(value["plan"]),
        value["last_key"],
        value["edit_count"],
        value["terminated"],
    )
    if prefix.payload() != value:
        raise ValueError("noncanonical prefix payload")
    return prefix


def _semantic(legality: bool, successor_class: str | None, continuation: dict | None) -> dict:
    if type(legality) is not bool:
        raise ValueError("boolean legality required")
    if not legality:
        if successor_class is not None or continuation is not None:
            raise ValueError("illegal action cannot carry successor semantics")
        return {"legality": False, "successor_class": None, "continuation": None}
    if (not isinstance(successor_class, str) or len(successor_class) != 64
            or any(c not in "0123456789abcdef" for c in successor_class)):
        raise ValueError("canonical successor-class sha256 required")
    if not isinstance(continuation, dict) or set(continuation) != STOP_FIELDS:
        raise ValueError("closed STOP/cap continuation required")
    successor = _prefix_from_payload(continuation["successor"]).payload()
    forced = continuation["forced_stop"]
    termination = continuation["termination"]
    if type(forced) is not bool or termination not in {"CONTINUE", "STOP", "CAP"}:
        raise ValueError("explicit STOP/cap continuation required")
    if forced != (termination == "CAP"):
        raise ValueError("forced_stop must mean CAP termination")
    if (termination == "CONTINUE") == successor["terminated"]:
        raise ValueError("termination and successor prefix disagree")
    if termination == "CAP" and not successor["terminated"]:
        raise ValueError("CAP must terminate the prefix")
    return {"legality": True, "successor_class": successor_class,
            "continuation": {"successor": successor, "forced_stop": forced,
                             "termination": termination}}


def _reference_step(state: State, prefix: scorer.Prefix, action: int, memory: dict, observation=None) -> dict:
    """Exact one-action menu semantics without materializing every phi row.

    This is the same support / Prefix partition used by v0_scorer.build_menu, but
    a divergence witness needs only one candidate.  Rebuilding hundreds of phi
    rows would turn a tiny reducer into the broad expensive check this task
    explicitly avoids.
    """
    if type(action) is not int or not 0 <= action < V.N_ACTIONS:
        raise ValueError("frozen integer candidate action required")
    if prefix.terminated:
        return _semantic(False, None, None)
    if action == V.STOP_ID:
        successor = scorer.Prefix(prefix.plan, prefix.last_key, prefix.edit_count, True)
    else:
        if prefix.edit_count == V.MAX_EDITS_PER_EPOCH or V.plan_key(action) <= prefix.last_key:
            return _semantic(False, None, None)
        masks = Masks(state, prefix.plan, **_projection_kwargs(memory, observation))
        if not masks.legal(action):
            return _semantic(False, None, None)
        successor = scorer.Prefix(prefix.plan.apply(action, masks.resolver),
                                  V.plan_key(action), prefix.edit_count + 1, False)
    payload = successor.payload()
    forced = action == V.STOP_ID and prefix.edit_count == V.MAX_EDITS_PER_EPOCH
    termination = "CAP" if forced else "STOP" if action == V.STOP_ID else "CONTINUE"
    return _semantic(True, digest(payload),
                     {"successor": payload, "forced_stop": forced, "termination": termination})


def _sequence_for_consumer(actions: tuple[int, ...]) -> list[int]:
    if not actions:
        raise ValueError("at least one candidate action required")
    if V.STOP_ID in actions[:-1]:
        raise ValueError("STOP may only be the candidate/final edit")
    sequence = list(actions)
    if sequence[-1] != V.STOP_ID:
        sequence.append(V.STOP_ID)
    return sequence


def _candidate_evaluator(observation: dict, memory: dict, actions: tuple[int, ...]) -> dict:
    from farm_neat.evaluation import v0_policy

    index = len(actions) - 1
    action = actions[-1]
    agent = V0Continuation.from_dict(memory).restore(KeepPolicy(), observation=observation)
    state = agent.actor_state(observation, initialize=False)
    plan = agent.plan.copy()
    try:
        validated = v0_policy.validate_edits(
            state, plan, _sequence_for_consumer(actions),
            projection_kwargs=_projection_kwargs(memory, observation),
        )
    except ValueError:
        # All prior actions were independently reference-admitted before this
        # function is called.  A failure here therefore belongs to this consumer
        # at or before the candidate; caller localization handles the first row.
        return _semantic(False, None, None)
    row = validated.trace[index]
    successor = row["successor"]
    forced = bool(row["forced_stop"])
    termination = "CAP" if forced else "STOP" if action == V.STOP_ID else "CONTINUE"
    return _semantic(True, digest(successor),
                     {"successor": successor, "forced_stop": forced, "termination": termination})


def _candidate_teacher(observation: dict, memory: dict, actions: tuple[int, ...]) -> dict:
    from farm_neat.training.teachers import v0_oracle

    index = len(actions) - 1
    action = actions[-1]
    try:
        agent = V0Continuation.from_dict(memory).restore(KeepPolicy(), observation=observation)
        dawn = v0_oracle.capture_dawn(agent, copy.deepcopy(observation))
        trace = v0_oracle.trace_edits(dawn, _sequence_for_consumer(actions))
    except ValueError:
        return _semantic(False, None, None)
    row = trace["steps"][index]
    successor = row["successor"]
    forced = bool(row["forced_stop"])
    termination = "CAP" if forced else "STOP" if action == V.STOP_ID else "CONTINUE"
    return _semantic(True, row["successor_class_id"],
                     {"successor": successor, "forced_stop": forced, "termination": termination})


def _candidate_step(consumer: str, observation: dict, memory: dict,
                    actions: tuple[int, ...]) -> dict:
    if consumer == "evaluator":
        return _candidate_evaluator(observation, memory, actions)
    if consumer == "teacher":
        return _candidate_teacher(observation, memory, actions)
    raise ValueError("consumer must be evaluator or teacher")


def _inject_test_double(value: dict, field: str | None) -> dict:
    result = copy.deepcopy(value)
    if field is None:
        return result
    if field != "successor_class" or not result["legality"]:
        raise ValueError("supported test double requires a legal successor_class")
    original = result["successor_class"]
    # Deterministic one-nibble corruption; never a normalization or collision.
    result["successor_class"] = ("0" if original[0] != "0" else "1") + original[1:]
    return result


def first_differing_field(reference: dict, candidate: dict) -> str | None:
    reference = _validate_semantic(reference)
    candidate = _validate_semantic(candidate)
    if reference["legality"] != candidate["legality"]:
        return "legality"
    if not reference["legality"]:
        return None
    if reference["successor_class"] != candidate["successor_class"]:
        return "successor_class"
    for name in ("successor", "forced_stop", "termination"):
        if reference["continuation"][name] != candidate["continuation"][name]:
            return "continuation." + name
    return None


def _validate_semantic(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != SEMANTIC_FIELDS:
        raise ValueError("closed V0 semantic record required")
    return _semantic(value["legality"], value["successor_class"], value["continuation"])


@dataclass(frozen=True)
class DivergenceCase:
    scenario_id: str
    consumer: Literal["evaluator", "teacher"]
    observation: dict
    continuation: dict
    history_actions: tuple[int, ...]
    candidate_action: int
    test_double: str | None = None

    def checked(self) -> "DivergenceCase":
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("nonempty scenario id required")
        if self.consumer not in CONSUMERS:
            raise ValueError("consumer must be evaluator or teacher")
        observation = _json_copy(self.observation)
        state = State(copy.deepcopy(observation))
        if state.hour != 0:
            raise ValueError("V0 divergence witness requires a dawn observation")
        memory = _continuation_dict(self.continuation, observation)
        history = tuple(self.history_actions)
        if (len(history) > V.MAX_EDITS_PER_EPOCH
                or any(type(a) is not int or not 0 <= a < V.N_ACTIONS or a == V.STOP_ID for a in history)
                or type(self.candidate_action) is not int or not 0 <= self.candidate_action < V.N_ACTIONS):
            raise ValueError("bounded non-STOP history plus one frozen candidate required")
        if self.test_double not in TEST_DOUBLE_FIELDS:
            raise ValueError("unknown test-double injection")
        return DivergenceCase(self.scenario_id, self.consumer, observation, memory,
                              history, self.candidate_action, self.test_double)


@dataclass(frozen=True)
class _Located:
    index: int
    prefix: dict
    action: int
    reference: dict
    candidate: dict
    field: str


def _advance(reference: dict) -> scorer.Prefix:
    if not reference["legality"]:
        raise ValueError("cannot advance an illegal reference path")
    return _prefix_from_payload(reference["continuation"]["successor"])


def locate_first(case: DivergenceCase) -> _Located:
    case = case.checked()
    memory = case.continuation
    agent = V0Continuation.from_dict(memory).restore(KeepPolicy(), observation=case.observation)
    state = agent.actor_state(case.observation, initialize=False)
    prefix = scorer.Prefix(Plan.from_canonical(memory["plan"]))
    requested = (*case.history_actions, case.candidate_action)
    for index, action in enumerate(requested):
        reference = _reference_step(state, prefix, action, memory, case.observation)
        candidate = _candidate_step(case.consumer, case.observation, memory, requested[:index + 1])
        if index == len(requested) - 1:
            candidate = _inject_test_double(candidate, case.test_double)
        field = first_differing_field(reference, candidate)
        if field is not None:
            return _Located(index, prefix.payload(), action, reference, candidate, field)
        if not reference["legality"]:
            raise NoDivergence("both paths reject the candidate; no later canonical prefix exists")
        prefix = _advance(reference)
    raise NoDivergence("reference and candidate semantics agree through the requested action")


def _truncate_to_first(case: DivergenceCase) -> DivergenceCase:
    case = case.checked()
    located = locate_first(case)
    requested = (*case.history_actions, case.candidate_action)
    test_double = case.test_double if located.index == len(requested) - 1 else None
    return replace(case, history_actions=tuple(requested[:located.index]),
                   candidate_action=requested[located.index], test_double=test_double)


def reduce_case(case: DivergenceCase) -> DivergenceCase:
    """Greedily remove irrelevant prior edits and continuation state.

    A reduction is accepted only when a fresh localization still finds the same
    first differing *field* at the retained candidate.  Required continuation
    keys remain present because V0Continuation is closed; collection/controller
    state is replaced by its empty value rather than silently dropped.
    """
    current = _truncate_to_first(case)
    target = locate_first(current).field

    changed = True
    while changed and current.history_actions:
        changed = False
        for index in range(len(current.history_actions)):
            history = current.history_actions[:index] + current.history_actions[index + 1:]
            trial = replace(current, history_actions=history)
            try:
                located = locate_first(trial)
            except (ValueError, NoDivergence):
                continue
            if (located.field == target and located.index == len(history)
                    and located.action == current.candidate_action):
                current = trial
                changed = True
                break

    empty_values = (
        ("allocation", []),
        ("started", []),
        ("hold_releases", []),
        ("controller", None),
        ("belief", None),
    )
    for field, empty in empty_values:
        if current.continuation[field] == empty:
            continue
        memory = copy.deepcopy(current.continuation)
        memory[field] = copy.deepcopy(empty)
        trial = replace(current, continuation=memory)
        try:
            checked = trial.checked()
            located = locate_first(checked)
        except (ValueError, NoDivergence):
            continue
        if (located.field == target and located.index == len(checked.history_actions)
                and located.action == current.candidate_action):
            current = checked
    return current.checked()


def _public_hash(observation: dict) -> str:
    return digest(actor_observation(State(copy.deepcopy(observation))))


def _replay_command(path: str | Path, continuation_path: str | Path | None = None) -> str:
    args = ["PYTHONPATH=src", "python3", "-B", "-m", "farm_neat.evaluation.v0_divergence",
            "replay", "--witness", str(path)]
    if continuation_path is not None:
        args.extend(["--continuation", str(continuation_path)])
    return " ".join(shlex.quote(part) for part in args)


def build_witness(case: DivergenceCase, *, replay_path: str | Path,
                  reduce: bool = True, inline_continuation: bool = True,
                  continuation_path: str | Path | None = None) -> dict:
    case = reduce_case(case) if reduce else _truncate_to_first(case)
    located = locate_first(case)
    memory = _continuation_dict(case.continuation, case.observation)
    state = State(copy.deepcopy(case.observation))
    if state.player not in (0, 1):
        raise ValueError("seat 0 or 1 required")
    if not inline_continuation and continuation_path is None:
        raise ValueError("external continuation path required when inline continuation is omitted")
    witness = {
        "schema": SCHEMA,
        "version": VERSION,
        "source_pins": source_pins(),
        "scenario": {
            "id": case.scenario_id,
            "consumer": case.consumer,
            "observation": _json_copy(case.observation),
            "history_actions": list(case.history_actions),
            "test_double": case.test_double,
        },
        "seat": state.player,
        "public_observation_sha256": _public_hash(case.observation),
        "continuation_memory_sha256": digest(memory),
        "continuation": memory if inline_continuation else None,
        "prefix": located.prefix,
        "candidate_action": located.action,
        "reference": located.reference,
        "candidate": located.candidate,
        "first_differing_field": located.field,
        "replay_command": _replay_command(replay_path, continuation_path),
    }
    return validate_witness(witness, verify_sources=True,
                            external_continuation=memory if not inline_continuation else None)


def evaluator_witness(*, scenario_id: str, observation: dict,
                      continuation: V0Continuation | dict, history_actions=(),
                      candidate_action: int, replay_path: str | Path,
                      test_double: str | None = None, reduce: bool = True,
                      inline_continuation: bool = True,
                      continuation_path: str | Path | None = None) -> dict:
    """Capture the evaluator validator's first disagreement with exact menu semantics."""
    case = DivergenceCase(scenario_id, "evaluator", observation, _continuation_dict(continuation, observation),
                          tuple(history_actions), candidate_action, test_double)
    return build_witness(case, replay_path=replay_path, reduce=reduce,
                         inline_continuation=inline_continuation,
                         continuation_path=continuation_path)


def teacher_witness(*, scenario_id: str, observation: dict,
                    continuation: V0Continuation | dict, history_actions=(),
                    candidate_action: int, replay_path: str | Path,
                    test_double: str | None = None, reduce: bool = True,
                    inline_continuation: bool = True,
                    continuation_path: str | Path | None = None) -> dict:
    """Capture the V0 oracle trace path's first disagreement with exact menu semantics."""
    case = DivergenceCase(scenario_id, "teacher", observation, _continuation_dict(continuation, observation),
                          tuple(history_actions), candidate_action, test_double)
    return build_witness(case, replay_path=replay_path, reduce=reduce,
                         inline_continuation=inline_continuation,
                         continuation_path=continuation_path)


def validate_witness(value: dict, *, verify_sources: bool,
                     external_continuation: dict | V0Continuation | None = None) -> dict:
    if not isinstance(value, dict) or set(value) != WITNESS_FIELDS:
        raise ValueError("closed divergence witness required")
    if value["schema"] != SCHEMA or value["version"] != VERSION:
        raise ValueError("unsupported divergence witness schema/version")
    if (not isinstance(value["source_pins"], dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value["source_pins"].items())):
        raise ValueError("source pin mapping required")
    if verify_sources and value["source_pins"] != source_pins():
        raise ValueError("divergence witness source pins no longer match tree")

    scenario = value["scenario"]
    if not isinstance(scenario, dict) or set(scenario) != SCENARIO_FIELDS:
        raise ValueError("closed divergence scenario required")
    if not isinstance(scenario["id"], str) or not scenario["id"].strip():
        raise ValueError("nonempty scenario id required")
    if scenario["consumer"] not in CONSUMERS or scenario["test_double"] not in TEST_DOUBLE_FIELDS:
        raise ValueError("unknown divergence consumer/test double")
    observation = _json_copy(scenario["observation"])
    state = State(copy.deepcopy(observation))
    if value["seat"] != state.player or type(value["seat"]) is not int or value["seat"] not in (0, 1):
        raise ValueError("witness seat/observation mismatch")
    if value["public_observation_sha256"] != _public_hash(observation):
        raise ValueError("public observation hash mismatch")
    history = scenario["history_actions"]
    if (not isinstance(history, list) or len(history) > V.MAX_EDITS_PER_EPOCH
            or any(type(a) is not int or a == V.STOP_ID or not 0 <= a < V.N_ACTIONS for a in history)):
        raise ValueError("closed earlier-edit history required")

    inline = value["continuation"]
    supplied = inline if inline is not None else external_continuation
    if supplied is None:
        raise ValueError("inline or explicitly supplied V0Continuation required for replay")
    memory = _continuation_dict(supplied, observation)
    if value["continuation_memory_sha256"] != digest(memory):
        raise ValueError("continuation memory hash mismatch")
    if inline is not None and _continuation_dict(inline, observation) != memory:
        raise ValueError("inline continuation mismatch")

    prefix = _prefix_from_payload(value["prefix"]).payload()
    action = value["candidate_action"]
    if type(action) is not int or not 0 <= action < V.N_ACTIONS:
        raise ValueError("frozen candidate action required")
    reference = _validate_semantic(value["reference"])
    candidate = _validate_semantic(value["candidate"])
    field = first_differing_field(reference, candidate)
    if field is None or value["first_differing_field"] != field or field not in DIFF_ORDER:
        raise ValueError("recorded first differing field is not exact")
    if not isinstance(value["replay_command"], str) or not value["replay_command"].strip():
        raise ValueError("exact replay command required")

    result = copy.deepcopy(value)
    result["scenario"]["observation"] = observation
    result["scenario"]["history_actions"] = list(history)
    result["prefix"] = prefix
    result["reference"] = reference
    result["candidate"] = candidate
    if inline is not None:
        result["continuation"] = memory
    return result


def replay_witness(value: dict, *, external_continuation: dict | V0Continuation | None = None) -> dict:
    witness = validate_witness(value, verify_sources=True, external_continuation=external_continuation)
    memory = witness["continuation"] if witness["continuation"] is not None else _continuation_dict(external_continuation, witness["scenario"]["observation"])
    scenario = witness["scenario"]
    case = DivergenceCase(
        scenario["id"], scenario["consumer"], scenario["observation"], memory,
        tuple(scenario["history_actions"]), witness["candidate_action"], scenario["test_double"],
    ).checked()
    located = locate_first(case)
    if (located.index != len(case.history_actions)
            or located.prefix != witness["prefix"]
            or located.action != witness["candidate_action"]
            or located.reference != witness["reference"]
            or located.candidate != witness["candidate"]
            or located.field != witness["first_differing_field"]):
        raise ValueError("fresh replay does not reproduce the recorded first divergence")
    return {
        "schema": "farm-v0-divergence-replay-v1",
        "status": "REPRODUCED",
        "scenario": scenario["id"],
        "consumer": scenario["consumer"],
        "history_edits": len(case.history_actions),
        "candidate_action": located.action,
        "prefix_sha256": digest(located.prefix),
        "first_differing_field": located.field,
    }


def write_witness(path: str | Path, witness: dict) -> dict:
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlinked witness output")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = validate_witness(witness, verify_sources=True)
    with path.open("x") as stream:
        stream.write(canonical(value) + "\n")
    return value


def replay_file(path: str | Path, *, continuation_path: str | Path | None = None) -> dict:
    witness = _read_json(Path(path))
    external = None
    if continuation_path is not None:
        external = _read_json(Path(continuation_path))
    return replay_witness(witness, external_continuation=external)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay one reduced source-bound V0 divergence witness")
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--witness", required=True, type=Path)
    replay.add_argument("--continuation", type=Path)
    args = parser.parse_args(argv)
    if args.command == "replay":
        print(canonical(replay_file(args.witness, continuation_path=args.continuation)))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
