"""Fallback non-neural teachers without admission-by-conversion.

This module is deliberately a *bridge* between two already-owned training surfaces:

* ``bc.inspect_step`` owns observed primitive SFT labels.  Those labels remain
  attempted primitive prefixes and are never promoted to latent V0 plans.
* ``v0_targets`` owns V0 successor-class rows.  This module only hands it an
  explicit native V0 edit trajectory *after* exact prefix/compiler verification
  and a same-opponent cash-margin gate.

A source season may be either a responsive Router-family harness game or a public
leader capsule.  Leader capsules always carry ``RECORDED_RIVAL_OPEN_LOOP``: the
recorded rival tape is a control, not a responsive opponent.  Splits are assigned
once to ``(physical_world_id, episode_id)`` and inherited by every step.

The inversion is intentionally honest about representation loss.  It proposes a
reachable opening Plan from observed producer instances, measures full-season
producer/product/sale-window residuals on candidate play, and returns every source
cohort that could not be represented exactly as a ``v0_inexpressible_gap``.  A
reachable approximation may be *evaluated*, but its V0 label is only the Plan it
actually reaches; an unrealized source target is never labeled.

No function here launches games.  Callers provide source/candidate traces and
terminal cash measured under the declared comparison protocol.  This keeps native
runs chargeable/auditable by the surrounding harness.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import Masks, State
from farm_neat.training import bc, v0_targets

SCHEMA = "farm-fallback-teacher-v1"
MANIFEST_SCHEMA = "farm-fallback-teacher-manifest-v1"
SIGNATURE_SCHEMA = "farm-fallback-season-signature-v1"
CANDIDATE_SCHEMA = "farm-fallback-v0-candidate-v1"
GATE_SCHEMA = "farm-fallback-v0-gate-v1"

RESPONSIVE = "RESPONSIVE_HARNESS"
OPEN_LOOP = "RECORDED_RIVAL_OPEN_LOOP"
SOURCE_KINDS = {"router": RESPONSIVE, "leader": OPEN_LOOP}
RESIDUAL_FAMILIES = ("producer_instances", "product_instances", "sale_windows")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"finite {name} required")
    return float(value)


def source_descriptor(*, kind: str, source_name: str, source_uri: str,
                      source_sha256: str, license: str, physical_world_id: str,
                      episode_id: str, learner_seat: int, opponent_identity: str,
                      comparison_protocol: str | None = None,
                      seed_group: str | None = None) -> dict:
    """Validate the identity/control needed by both objectives.

    ``kind='leader'`` is fail-closed to recorded-rival open loop.  ``kind='router'``
    is fail-closed to a responsive harness declaration; callers cannot relabel one
    protocol as the other after measuring it.
    """
    if kind not in SOURCE_KINDS:
        raise ValueError("source kind must be router or leader")
    required_protocol = SOURCE_KINDS[kind]
    comparison_protocol = required_protocol if comparison_protocol is None else comparison_protocol
    if comparison_protocol != required_protocol:
        raise ValueError(f"{kind} source requires {required_protocol}")
    if type(learner_seat) is not int or learner_seat not in (0, 1):
        raise ValueError("learner seat must be 0 or 1")
    values = {
        "source_name": source_name, "source_uri": source_uri, "license": license,
        "physical_world_id": physical_world_id, "episode_id": episode_id,
        "opponent_identity": opponent_identity,
    }
    if any(not isinstance(v, str) or not v for v in values.values()):
        raise ValueError("nonempty source/world/episode/opponent strings required")
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_sha256)):
        raise ValueError("lowercase source SHA256 required")
    seed_group = physical_world_id if seed_group is None else seed_group
    if not isinstance(seed_group, str) or not seed_group:
        raise ValueError("nonempty seed group required")
    value = {
        "kind": kind, "source_name": source_name, "source_uri": source_uri,
        "source_sha256": source_sha256, "license": license,
        "physical_world_id": physical_world_id, "episode_id": episode_id,
        "seed_group": seed_group, "learner_seat": learner_seat,
        "opponent_identity": opponent_identity,
        "comparison_protocol": comparison_protocol,
        "responsive": comparison_protocol == RESPONSIVE,
    }
    value["source_identity_sha256"] = _digest(value)
    return value


def leader_source(capsule: dict, *, source_uri: str, source_sha256: str,
                  license: str = "public Kaggle episode") -> dict:
    """Source descriptor for a verified public capsule; never responsive."""
    from farm_neat.evaluation.public_episodes import SCHEMA as CAPSULE_SCHEMA
    if capsule.get("schema") != CAPSULE_SCHEMA or len(capsule.get("joint_actions", [])) != 719:
        raise ValueError("complete leader capsule required")
    seat = capsule.get("focus", {}).get("seat")
    episode = capsule.get("episode_id")
    rival = capsule.get("rival", {})
    if type(episode) not in (int, str):
        raise ValueError("capsule episode identity required")
    opponent = f"{rival.get('team','UNKNOWN')}/{rival.get('submission_id','UNKNOWN')}"
    world = _digest({"seed": capsule.get("seed"), "configuration": capsule.get("configuration", {})})
    return source_descriptor(kind="leader", source_name="Majkel1337 public capsule",
        source_uri=source_uri, source_sha256=source_sha256, license=license,
        physical_world_id=world, episode_id=str(episode), learner_seat=seat,
        opponent_identity=opponent, comparison_protocol=OPEN_LOOP,
        seed_group=str(capsule.get("seed")))



def leader_capsule_material(capsule: dict, *, source_uri: str, source_sha256: str,
                            license: str = "public Kaggle episode") -> dict:
    """Restore one leader capsule into source trace/signature/cash material.

    This performs one complete exact-engine replay of the already-recorded game.
    Review-package callers must charge that native replay before invoking it.  The
    rival is not queried as a policy; the source remains recorded open-loop data.
    """
    from farm_neat.evaluation.public_episodes import restore
    src = leader_source(capsule, source_uri=source_uri, source_sha256=source_sha256, license=license)
    trace = list(restore(capsule))
    if len(trace) != 719:
        raise ValueError("complete 719-step leader restoration required")
    rewards = capsule.get("rewards")
    seat = src["learner_seat"]
    if not isinstance(rewards, list) or len(rewards) != 2:
        raise ValueError("capsule terminal rewards required")
    cash = [_finite_number(rewards[seat], "focus reward"),
            _finite_number(rewards[1 - seat], "rival reward")]
    return {"source": src, "steps": trace, "signature": season_signature(trace),
            "source_cash": cash, "comparison_protocol": OPEN_LOOP,
            "physical_game_steps": 719}


def measured_candidate(candidate: dict, steps: Sequence[tuple[int, dict, dict]],
                       terminal_cash: Sequence[float], source: dict, *,
                       comparison_protocol: str, opponent_identity: str) -> dict:
    """Normalize an externally measured Router/open-loop candidate for selection."""
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("fallback candidate required")
    if candidate.get("source_identity_sha256") != source.get("source_identity_sha256"):
        raise ValueError("candidate/source identity mismatch")
    if comparison_protocol != source.get("comparison_protocol"):
        raise ValueError("candidate/source comparison protocol mismatch")
    if opponent_identity != source.get("opponent_identity"):
        raise ValueError("candidate/source opponent identity mismatch")
    if len(terminal_cash) != 2:
        raise ValueError("[focus_cash, rival_cash] required")
    cash = [_finite_number(terminal_cash[0], "candidate focus cash"),
            _finite_number(terminal_cash[1], "candidate rival cash")]
    return {"candidate": candidate, "signature": season_signature(steps),
            "terminal_cash": cash, "comparison_protocol": comparison_protocol,
            "opponent_identity": opponent_identity}


def evaluate_leader_open_loop(capsule: dict, source: dict, candidate: dict) -> dict:
    """Run one reachable candidate against the capsule rival's recorded action tape.

    This is intentionally *not* a responsive counterfactual.  It performs one native
    719-step game and must be charged by review tooling before use.  The focus seat's
    recorded actions are discarded; only the rival seat's recorded actions are replayed.
    """
    from farm_neat.economy.v0.epochs import TrajectoryPolicy, V0Agent
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    if source.get("kind") != "leader" or source.get("comparison_protocol") != OPEN_LOOP:
        raise ValueError("leader recorded-rival source required")
    if candidate.get("source_identity_sha256") != source.get("source_identity_sha256"):
        raise ValueError("candidate/source identity mismatch")
    verification = v0_targets.verify_request(candidate["request"])
    if verification["status"] != "ADMITTED":
        raise ValueError("only exact reachable V0 candidates may be evaluated")
    seat = source["learner_seat"]
    if capsule.get("focus", {}).get("seat") != seat or len(capsule.get("joint_actions", [])) != 719:
        raise ValueError("capsule/source focus mismatch")
    edits = candidate["request"]["actions"]
    if not edits or edits[-1] != V.STOP_ID or any(a == V.STOP_ID for a in edits[:-1]):
        raise ValueError("canonical opening trajectory with final STOP required")
    policy = TrajectoryPolicy({0: edits[:-1]})
    actor = V0Agent(policy, labor_factor=4.0, hands_mode="daily",
                    belief={p: 0 for p in V.PRODUCTS},
                    game_id="fallback-open-loop/" + candidate["candidate_id"])
    env = ExactKaggricultureEnvironment(seed=capsule["seed"], configuration=capsule["configuration"])
    trace = []
    for step, recorded_joint in enumerate(capsule["joint_actions"]):
        obs = env.agent_observation(seat)
        action = actor(obs, env.configuration)
        trace.append((step, copy.deepcopy(obs), copy.deepcopy(action)))
        joint = copy.deepcopy(recorded_joint)
        joint[seat] = action
        env.step(joint)
    final = env.agent_observation(seat)
    cash_by_seat = [_finite_number(f["money"], "terminal cash") for f in final["farms"]]
    measurement = measured_candidate(candidate, trace,
        [cash_by_seat[seat], cash_by_seat[1 - seat]], source,
        comparison_protocol=OPEN_LOOP, opponent_identity=source["opponent_identity"])
    measurement["physical_game_steps"] = 719
    measurement["control"] = "RECORDED_RIVAL_ACTIONS_OPEN_LOOP; FOCUS_ACTIONS_REPLACED_BY_V0_AGENT"
    return measurement

def split_for_episode(physical_world_id: str, episode_id: str, split_id: str,
                      *, train: int = 80, validation: int = 10, test: int = 10) -> str:
    """Stable split for an entire world+episode group, never for individual steps."""
    if any(not isinstance(v, str) or not v for v in (physical_world_id, episode_id, split_id)):
        raise ValueError("world, episode and split identity required")
    if any(type(v) is not int or v < 0 for v in (train, validation, test)) or train + validation + test != 100:
        raise ValueError("integer split percentages must total 100")
    bucket = int(hashlib.sha256(f"{split_id}\0{physical_world_id}\0{episode_id}".encode()).hexdigest()[:8], 16) % 100
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"


def _tiles(observation: dict) -> Iterable[tuple[int, int, dict]]:
    seat = observation.get("player")
    farms = observation.get("farms", [])
    if type(seat) is not int or seat not in (0, 1) or len(farms) != 2:
        raise ValueError("actor observation with two farms required")
    for y, row in enumerate(farms[seat].get("tiles", [])):
        for x, tile in enumerate(row):
            if isinstance(tile, dict):
                yield y, x, tile


def season_signature(steps: Iterable[tuple[int, dict, dict]]) -> dict:
    """Observed producer instances and attempted sale windows for one focus seat.

    An instance key includes producer, birth day and tile coordinate so a persistent
    body is counted once across the season.  Product counts are the products implied
    by those observed producer instances.  Sale windows are attempted SELL quantity
    by product/day; fills are deliberately not inferred from attempts.
    """
    instances: set[tuple[str, int, int, int]] = set()
    sales: Counter[tuple[str, int]] = Counter()
    seen_steps: list[int] = []
    seat = None
    for step, observation, action in steps:
        if type(step) is not int or step < 0 or observation.get("step") != step:
            raise ValueError("trace step/observation clock mismatch")
        here = observation.get("player")
        if seat is None:
            seat = here
        if here != seat:
            raise ValueError("trace mixes focus seats")
        seen_steps.append(step)
        for y, x, tile in _tiles(observation):
            producer = tile.get("crop") or tile.get("animal")
            if producer is None:
                continue
            birth = tile.get("planted_day") if tile.get("crop") else tile.get("placed_day")
            if type(birth) is not int or not 0 <= birth < V.SEASON_DAYS:
                # A body with no trusted birth is still visible but cannot be mapped to V0.
                continue
            instances.add((producer, birth, y, x))
        if action is None:
            action = {}
        if not isinstance(action, dict):
            raise ValueError("submitted action object required")
        for order in action.get("market", []) or []:
            if (isinstance(order, list) and len(order) == 3 and order[0] == "SELL"
                    and order[1] in V.PRODUCTS and type(order[2]) is int and order[2] >= 0):
                sales[(order[1], step // 24)] += order[2]
    cohort_counts = Counter((producer, birth) for producer, birth, _y, _x in instances)
    product_counts = Counter()
    for (producer, _birth), count in cohort_counts.items():
        if producer in V.PRODUCTS:
            product = producer
        else:
            matches = {m["product"] for m in V.MODULES if m["producer"] == producer}
            if len(matches) != 1:
                raise ValueError(f"producer {producer!r} has no unique V0 product")
            product = next(iter(matches))
        product_counts[product] += count
    return {
        "schema": SIGNATURE_SCHEMA,
        "learner_seat": seat,
        "observed_steps": len(seen_steps),
        "step_min": min(seen_steps) if seen_steps else None,
        "step_max": max(seen_steps) if seen_steps else None,
        "producer_instances": [
            {"producer": p, "birth_day": b, "count": n}
            for (p, b), n in sorted(cohort_counts.items())
        ],
        "product_instances": {p: product_counts[p] for p in sorted(product_counts)},
        "sale_windows": {p: {str(day): sales[(p, day)] for day in range(V.SEASON_DAYS) if sales[(p, day)]}
                         for p in V.PRODUCTS if any(sales[(p, day)] for day in range(V.SEASON_DAYS))},
    }


def _counter_from_instances(signature: dict) -> Counter:
    if signature.get("schema") != SIGNATURE_SCHEMA:
        raise ValueError("fallback season signature required")
    out = Counter()
    for row in signature.get("producer_instances", []):
        producer, birth, count = row.get("producer"), row.get("birth_day"), row.get("count")
        if not isinstance(producer, str) or type(birth) is not int or type(count) is not int or count < 0:
            raise ValueError("malformed producer instance row")
        out[(producer, birth)] += count
    return out


def compare_signatures(source: dict, candidate: dict) -> dict:
    """Full signed residuals plus a bounded normalized L1 selection score."""
    s_inst, c_inst = _counter_from_instances(source), _counter_from_instances(candidate)
    s_prod, c_prod = Counter(source.get("product_instances", {})), Counter(candidate.get("product_instances", {}))
    inst = {f"{p}:b{b:02d}": s_inst[p, b] - c_inst[p, b]
            for p, b in sorted(set(s_inst) | set(c_inst)) if s_inst[p, b] != c_inst[p, b]}
    prod = {p: s_prod[p] - c_prod[p] for p in sorted(set(s_prod) | set(c_prod)) if s_prod[p] != c_prod[p]}
    sale = {}
    sale_l1 = 0
    source_sales, candidate_sales = source.get("sale_windows", {}), candidate.get("sale_windows", {})
    for product in sorted(set(source_sales) | set(candidate_sales)):
        days = set(source_sales.get(product, {})) | set(candidate_sales.get(product, {}))
        rows = {}
        for day in sorted(days, key=int):
            delta = int(source_sales.get(product, {}).get(day, 0)) - int(candidate_sales.get(product, {}).get(day, 0))
            if delta:
                rows[day] = delta
                sale_l1 += abs(delta)
        if rows:
            sale[product] = rows
    inst_l1, prod_l1 = sum(abs(v) for v in inst.values()), sum(abs(v) for v in prod.values())
    denom_inst = max(1, sum(s_inst.values()))
    denom_prod = max(1, sum(s_prod.values()))
    denom_sale = max(1, sum(int(q) for by_day in source_sales.values() for q in by_day.values()))
    components = {
        "producer_instances": inst_l1 / denom_inst,
        "product_instances": prod_l1 / denom_prod,
        "sale_windows": sale_l1 / denom_sale,
    }
    return {
        "families": list(RESIDUAL_FAMILIES),
        "producer_instances": inst,
        "product_instances": prod,
        "sale_windows": sale,
        "normalized_l1": components,
        "score": sum(components.values()),
    }


def _module_for_source(producer: str, birth: int) -> int | None:
    matches = [m for m in V.MODULES if m["producer"] == producer and m["birth"] == birth]
    if not matches:
        return None
    # Never infer fertilization from mere occupancy.  Plain watered is the least
    # latent crop assumption; animals have one fed_cared mode.
    matches.sort(key=lambda m: (m["policy"] != "watered" and not m["animal"], m["id"]))
    return matches[0]["id"]


def propose_opening_candidate(observation: dict, source_signature: dict, *,
                              source: dict, candidate_id: str,
                              split_id: str = "fallback-v1") -> dict:
    """Construct one reachable approximation without ever labeling the source target.

    Every observed source cohort is considered.  If its exact count is not the image
    of one legal native PROD edit at the live prefix, the gap is retained and the
    nearest *reachable* count may enter the candidate.  Once 16 keys are consumed,
    every remaining source cohort gets an explicit EDIT_CAP gap.
    """
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate id required")
    if source.get("learner_seat") != observation.get("player"):
        raise ValueError("source/observation learner seat mismatch")
    desired = _counter_from_instances(source_signature)
    rows = []
    gaps = []
    for (producer, birth), count in desired.items():
        module = _module_for_source(producer, birth)
        if module is None:
            gaps.append({"family": "producer_instances", "reason": "NO_V0_MODULE",
                         "producer": producer, "birth_day": birth, "source_count": count})
        else:
            rows.append((module, producer, birth, count))
    rows.sort()
    state, plan, edits, last_key = State(observation), Plan(), [], -1
    witnesses = []
    for module, producer, birth, wanted in rows:
        if len(edits) >= V.MAX_EDITS_PER_EPOCH:
            gaps.append({"family": "producer_instances", "reason": "EDIT_CAP",
                         "producer": producer, "birth_day": birth, "source_count": wanted,
                         "module": V.MODULES[module]["key"]})
            continue
        masks = Masks(state, plan, **v0_targets.SETTINGS)
        attempts = []
        reachable = []
        for delta in ("PLUS_ONE", "NEXT_BINDING", "MAX_FEASIBLE"):
            action = V.encode("PROD", module, V.DELTAS.index(delta))
            legal = V.plan_key(action) > last_key and masks.legal(action)
            row = {"action": action, "delta": delta, "legal": legal}
            if legal:
                successor = plan.apply(action, masks.resolver)
                got = successor.cohorts.get(module, 0)
                row["count"] = got
                reachable.append((abs(wanted - got), action, got, successor))
            attempts.append(row)
        if not reachable:
            gaps.append({"family": "producer_instances", "reason": "MASKED_OR_UNREACHABLE",
                         "producer": producer, "birth_day": birth, "source_count": wanted,
                         "module": V.MODULES[module]["key"], "attempts": attempts})
            continue
        reachable.sort(key=lambda row: (row[0], row[1]))
        _error, action, got, successor = reachable[0]
        if got != wanted:
            gaps.append({"family": "producer_instances", "reason": "COUNT_NOT_ONE_EDIT_REACHABLE",
                         "producer": producer, "birth_day": birth, "source_count": wanted,
                         "selected_reachable_count": got, "module": V.MODULES[module]["key"],
                         "attempts": attempts})
        edits.append(action)
        witnesses.append({"source": {"producer": producer, "birth_day": birth, "count": wanted},
                          "action": action, "description": V.describe(action),
                          "reached_count": got, "prefix_plan_sha256": plan.digest(),
                          "successor_plan_sha256": successor.digest()})
        plan, last_key = successor, V.plan_key(action)
    actions = edits + [V.STOP_ID]
    split = split_for_episode(source["physical_world_id"], source["episode_id"], split_id)
    request = {
        "teacher_kind": "explicit_native_v0",
        "teacher_id": f"fallback-inversion/{source['source_identity_sha256'][:12]}",
        "trajectory_id": candidate_id,
        "physical_world_id": source["physical_world_id"],
        "source_group": f"{source['kind']}:{source['physical_world_id']}:{source['episode_id']}",
        "seed_group": source["seed_group"], "split": split,
        "observation": copy.deepcopy(observation), "initial_plan": Plan().canonical(),
        "actions": actions, "intended_plan": plan.canonical(),
        "provenance": {
            "kind": "fallback_v0_inversion_candidate",
            "source_identity_sha256": source["source_identity_sha256"],
            "comparison_protocol": source["comparison_protocol"],
            "conversion_is_training_admission": False,
        },
    }
    verified = v0_targets.verify_request(request)
    if verified["status"] != "ADMITTED":
        gaps.append({"family": "v0_target", "reason": verified["reason"],
                     "detail": verified.get("detail"), "residual": verified.get("residual")})
    return {
        "schema": CANDIDATE_SCHEMA, "candidate_id": candidate_id,
        "source_identity_sha256": source["source_identity_sha256"],
        "split": split, "split_group": [source["physical_world_id"], source["episode_id"]],
        "request": request, "v0_verification": verified, "opening_witness": witnesses,
        "v0_inexpressible_gaps": gaps,
        "scope": "REACHABLE_OPENING_CANDIDATE_ONLY; CONVERSION_IS_NOT_ADMISSION",
    }


def primitive_sft_reports(steps: Sequence[tuple[int, dict, dict]], source: dict,
                          *, configuration: dict | None = None) -> list[dict]:
    """Run bc.inspect_step with one episode-level split; no V0 targets are mixed in."""
    reports = []
    if not steps:
        return reports
    # Compute once to fail closed on malformed group identity; all step reports
    # inherit this episode group in the outer manifest, never step-derived splits.
    split_for_episode(source["physical_world_id"], source["episode_id"], "fallback-v1")
    provenance = {
        "source_name": source["source_name"], "source_uri": source["source_uri"],
        "source_sha256": source["source_sha256"], "license": source["license"],
        "episode_id": source["episode_id"], "player": source["learner_seat"],
    }
    for i, (step, observation, action) in enumerate(steps):
        if observation.get("step") != step or observation.get("player") != source["learner_seat"]:
            raise ValueError("primitive SFT trace identity mismatch")
        successor = None
        if i + 1 < len(steps) and steps[i + 1][0] == step + 1:
            successor = steps[i + 1][1]
        report = bc.inspect_step(observation, action, provenance=provenance,
                                 successor=successor, configuration=configuration)
        # Keep the BC content seal byte-valid. Episode grouping/objective separation
        # lives in the outer fallback manifest rather than being injected into BC.
        reports.append(report)
    return reports


def cash_margin_gate(source: dict, source_cash: Sequence[float], candidate_cash: Sequence[float],
                     *, declared_min_fraction: float, candidate_protocol: str,
                     candidate_opponent_identity: str) -> dict:
    """Same-opponent outcome gate.  Non-positive source margins are rejected denominators."""
    if (type(declared_min_fraction) not in (int, float) or not math.isfinite(declared_min_fraction)
            or declared_min_fraction < 0):
        raise ValueError("finite nonnegative declared fraction required")
    if candidate_protocol != source["comparison_protocol"]:
        raise ValueError("candidate/source comparison protocol mismatch")
    if candidate_opponent_identity != source["opponent_identity"]:
        raise ValueError("candidate/source opponent identity mismatch")
    if len(source_cash) != 2 or len(candidate_cash) != 2:
        raise ValueError("[focus_cash, rival_cash] required")
    source_focus, source_rival = (_finite_number(v, "source cash") for v in source_cash)
    cand_focus, cand_rival = (_finite_number(v, "candidate cash") for v in candidate_cash)
    source_margin, candidate_margin = source_focus - source_rival, cand_focus - cand_rival
    rejected = None
    kept = None
    admitted = False
    if source_margin <= 0:
        rejected = {"reason": "NONPOSITIVE_SOURCE_CASH_MARGIN", "source_margin": source_margin,
                    "source_cash": [source_focus, source_rival]}
    else:
        kept = candidate_margin / source_margin
        admitted = kept >= declared_min_fraction
    return {
        "schema": GATE_SCHEMA, "declared_min_fraction": float(declared_min_fraction),
        "source_cash": [source_focus, source_rival], "candidate_cash": [cand_focus, cand_rival],
        "source_margin": source_margin, "candidate_margin": candidate_margin,
        "kept_cash_fraction": kept, "rejected_denominator": rejected,
        "cash_gate_passed": admitted,
        "comparison_protocol": source["comparison_protocol"],
        "opponent_identity": source["opponent_identity"],
    }


def select_inversion(source: dict, source_signature: dict, source_cash: Sequence[float],
                     candidate_measurements: Iterable[dict], *,
                     declared_min_fraction: float) -> dict:
    """Choose the best measured season reproduction, then apply exact V0 + cash gates.

    Each candidate measurement is ``{"candidate": propose_opening_candidate(...),
    "signature": season_signature(...), "terminal_cash": [focus, rival],
    "comparison_protocol": ..., "opponent_identity": ...}``.
    """
    rows = []
    for measurement in candidate_measurements:
        candidate = measurement.get("candidate")
        if not isinstance(candidate, dict) or candidate.get("schema") != CANDIDATE_SCHEMA:
            raise ValueError("fallback candidate measurement required")
        if candidate.get("source_identity_sha256") != source["source_identity_sha256"]:
            raise ValueError("candidate belongs to a different source")
        residual = compare_signatures(source_signature, measurement["signature"])
        gate = cash_margin_gate(source, source_cash, measurement["terminal_cash"],
            declared_min_fraction=declared_min_fraction,
            candidate_protocol=measurement["comparison_protocol"],
            candidate_opponent_identity=measurement["opponent_identity"])
        verification = v0_targets.verify_request(candidate["request"])
        rows.append({"candidate": candidate, "residual": residual, "cash_gate": gate,
                     "v0_verification": verification})
    if not rows:
        return {"selected": None, "candidates": [], "admitted": False,
                "reason": "NO_MEASURED_CANDIDATES"}
    rows.sort(key=lambda row: (row["residual"]["score"], row["candidate"]["candidate_id"]))
    selected = rows[0]
    admitted = (selected["v0_verification"]["status"] == "ADMITTED"
                and selected["cash_gate"]["cash_gate_passed"])
    reason = None
    if selected["v0_verification"]["status"] != "ADMITTED":
        reason = "V0_TARGET_REJECTED"
    elif selected["cash_gate"]["rejected_denominator"] is not None:
        reason = "REJECTED_DENOMINATOR"
    elif not selected["cash_gate"]["cash_gate_passed"]:
        reason = "CASH_FRACTION_GATE"
    return {"selected": selected, "candidates": rows, "admitted": admitted,
            "reason": reason,
            "selection": "MIN_FULL_SEASON_RESIDUAL_BEFORE_ADMISSION_GATE"}


def export_bundle(output: str | Path, *, source: dict, source_signature: dict,
                  source_cash: Sequence[float], candidate_measurements: Iterable[dict],
                  primitive_steps: Sequence[tuple[int, dict, dict]] = (),
                  configuration: dict | None = None, declared_min_fraction: float,
                  split_id: str = "fallback-v1") -> dict:
    """Write separated primitive SFT and gated V0-TARGETS artifacts.

    The outer MANIFEST is written last.  ``v0_targets/`` exists only when the best
    measured candidate passes *both* native V0 verification and the cash-margin gate.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    reports = primitive_sft_reports(list(primitive_steps), source, configuration=configuration)
    primitive_path = output / "primitive_sft.jsonl"
    with primitive_path.open("x", encoding="utf-8", newline="\n") as stream:
        for report in reports:
            stream.write(_json(report) + "\n")
    selection = select_inversion(source, source_signature, source_cash,
                                 candidate_measurements, declared_min_fraction=declared_min_fraction)
    target_manifest = None
    selected = selection.get("selected")
    if selection["admitted"]:
        request = copy.deepcopy(selected["candidate"]["request"])
        # The bundle split identity owns all rows; the request still carries the
        # episode-level split computed from exactly the same world+episode group.
        request["split"] = split_for_episode(source["physical_world_id"], source["episode_id"], split_id)
        target_manifest = v0_targets.export_dataset([request], output / "v0_targets", split_id=split_id)
        if target_manifest["counts"].get("admitted") != 1:
            raise RuntimeError("gated request was not admitted by V0-TARGETS")
    rejected_denominators = []
    for row in selection.get("candidates", []):
        rejected = row["cash_gate"].get("rejected_denominator")
        if rejected is not None:
            rejected_denominators.append({"candidate_id": row["candidate"]["candidate_id"], **rejected})
    gaps = [] if selected is None else copy.deepcopy(selected["candidate"].get("v0_inexpressible_gaps", []))
    manifest = {
        "schema": MANIFEST_SCHEMA, "source": copy.deepcopy(source),
        "source_signature_sha256": _digest(source_signature),
        "split_id": split_id,
        "split_group": {"physical_world_id": source["physical_world_id"],
                        "episode_id": source["episode_id"],
                        "split": split_for_episode(source["physical_world_id"], source["episode_id"], split_id)},
        "objectives": {
            "primitive_sft": {"file": "primitive_sft.jsonl", "reports": len(reports),
                              "schema": bc.SCHEMA, "shares_loss_with_v0_classes": False,
                              "training_admission": False},
            "v0_successor_classes": {"directory": "v0_targets" if target_manifest else None,
                                     "schema": v0_targets.SCHEMA,
                                     "shares_loss_with_primitive_sft": False,
                                     "admitted_trajectories": 1 if target_manifest else 0},
        },
        "selection": {
            "candidate_id": None if selected is None else selected["candidate"]["candidate_id"],
            "rule": selection.get("selection"), "admitted": selection["admitted"],
            "reason": selection.get("reason"),
            "residual_families": list(RESIDUAL_FAMILIES),
            "residual": None if selected is None else selected["residual"],
            "kept_cash_fraction": None if selected is None else selected["cash_gate"]["kept_cash_fraction"],
            "declared_min_fraction": declared_min_fraction,
            "comparison_protocol": source["comparison_protocol"],
            "responsive": source["responsive"],
        },
        "candidate_diagnostics": [
            {"candidate_id": row["candidate"]["candidate_id"],
             "residual_score": row["residual"]["score"],
             "residual_families": row["residual"]["families"],
             "kept_cash_fraction": row["cash_gate"]["kept_cash_fraction"],
             "rejected_denominator": row["cash_gate"]["rejected_denominator"],
             "cash_gate_passed": row["cash_gate"]["cash_gate_passed"],
             "v0_status": row["v0_verification"]["status"]}
            for row in selection.get("candidates", [])
        ],
        "rejected_denominators": rejected_denominators,
        "v0_inexpressible_gaps": gaps,
        "files": {"primitive_sft.jsonl": {"sha256": _file_sha(primitive_path),
                                            "bytes": primitive_path.stat().st_size}},
        "limits": [
            "Conversion/proposal is not training admission; V0-TARGETS plus the same-opponent cash gate own admission.",
            "Primitive SFT imitates observed attempted primitives only and has no latent V0 loss.",
            "Leader controls are recorded-rival open loop and are never responsive counterfactuals.",
            "The opening inversion does not infer fertilization/service regimes from occupancy.",
        ],
    }
    if target_manifest is not None:
        manifest["v0_targets_manifest_sha256"] = _file_sha(output / "v0_targets" / "MANIFEST.json")
    (output / "MANIFEST.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "RESPONSIVE", "OPEN_LOOP", "source_descriptor", "leader_source", "leader_capsule_material",
    "measured_candidate", "evaluate_leader_open_loop", "split_for_episode", "season_signature",
    "compare_signatures", "propose_opening_candidate", "primitive_sft_reports",
    "cash_margin_gate", "select_inversion", "export_bundle",
]
