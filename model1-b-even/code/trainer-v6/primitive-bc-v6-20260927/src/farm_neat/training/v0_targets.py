"""Native V0 prefix-example exporter (opening-only bootstrap, CPU, no game steps).

Commands (register the same module in project/entrypoints.toml during intake)::

    PYTHONPATH=src python3 -B -m farm_neat.training.v0_targets --smoke --output NEW_DIR
    PYTHONPATH=src python3 -B -m farm_neat.training.v0_targets --input REQUESTS.json --output NEW_DIR

Input: {"schema": REQUEST_SCHEMA, "split_id": str, "requests": [request, ...]}.
A request has teacher_kind="explicit_native_v0", teacher_id, trajectory_id,
physical_world_id, source_group, seed_group, split, observation, initial_plan,
actions (strict key order with explicit final STOP), intended_plan, provenance.
Plans use Plan.canonical(). The opening-only exporter requires step=0 and an empty
initial Plan; later dawns need allocation/started/HOLD memory and are not admitted.
Naming an input native is an explicit caller attestation, not teacher approval.

Outputs: prefixes.jsonl, trajectories.jsonl, rejections.jsonl and MANIFEST.json
(written last). Existing directories are refused. An incomplete directory has no
manifest and is never a dataset. All requests, including rejected ones, are bound;
a mismatched desired Plan is NEVER silently replaced by the smaller reached Plan.

The learner consumes only row['actor'] and row['candidates']. Target-class indices
address that row's sorted legal support; do not use raw replay_action as a CE label.
Classes include exact Plan/last key/edit count/termination. Phi is from the existing
projection.phi in PHI_NAMES order with explicit labor=4 and zero flow. Sampled STOP
is included in class-path NLL; forced cap STOP contributes zero. This is NOT a
final-Plan marginal or evidence of executor realization/teacher playing strength.

actor.public_observation is a versioned allowlist of projection facts (own-private
stock is legitimate own information), not the raw engine observation. It omits
coordinates, worker identities, absolute step tokens and trainer provenance. The
raw public/own observation is retained only under trainer_only for reproduction.
This representation is a bootstrap choice, not a state-sufficiency claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, OrderedDict
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.compiler import compile_opening, unplaced
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import Masks, PHI_NAMES, Resolver, State, phi

SCHEMA = "farm-v0-prefix-examples-v1"
REQUEST_SCHEMA = "farm-v0-trajectory-requests-v1"
ACTOR_SCHEMA = "farm-v0-public-projection-facts-v1"
SETTINGS = {"labor_factor": 4.0, "belief": {p: 0 for p in V.PRODUCTS}}
IDENTITY_FIELDS = ("teacher_id", "trajectory_id", "physical_world_id", "source_group", "seed_group", "split")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def continuation_signature(plan: Plan, last_key: int, edit_count: int, terminated: bool) -> dict:
    """Public decoder continuation; raw representative action/history is absent."""
    return {"plan_sha256": plan.digest(), "last_key": last_key,
            "edit_count": edit_count, "terminated": terminated}


class _MemoResolver(Resolver):
    """Only memoizes the existing resolver; no replacement economics or jump rule."""
    def __init__(self, state: State):
        super().__init__(state, **SETTINGS)
        self.cache: dict[tuple, int] = {}

    def __call__(self, plan: Plan, module: int, delta: str) -> int:
        key = (_json(plan.canonical()), module, delta)
        if key not in self.cache:
            self.cache[key] = super().__call__(plan, module, delta)
        return self.cache[key]


def actor_observation(state: State) -> dict:
    """Explicit allowlist: no arbitrary raw-observation or provenance passthrough."""
    return {
        "schema": ACTOR_SCHEMA, "cash": state.cash, "owned_quadrants": state.quadrants,
        "hands_today": state.hands_today, "hires_today": state.hires_today,
        "occupied_sites": state.occupied, "stock": dict(state.stock),
        "seeds": {p: state.seeds.get(p, 0) for p in sorted({m["producer"] for m in V.MODULES if not m["animal"]})},
        "market_inventory": dict(state.inventory),
        "market_params": None if not state.params else {
            p: {k: state.params[p][k] for k in ("base", "I0", "T", "below_func", "above_func", "below_target", "above_target")}
            for p in V.PRODUCTS},
        "unlocked_shops": list(state.shops), "remaining_days": V.SEASON_DAYS - state.day,
        "event_distances": {
            "demand": {p: None if (t := state.next_demand(p, state.step)) is None else t - state.step for p in V.PRODUCTS},
            "reveal": None if (t := state.next_reveal(state.step)) is None else t - state.step,
            "terminal": V.LAST_STEP - state.step,
        },
    }


def prefix_support(state: State, plan: Plan, last_key: int, edit_count: int) -> dict:
    """Exact sparse support and complete successor partition at ONE canonical prefix.

    Caller supplies a reachable decoder state. This function also supports isolated
    continuation fixtures; trajectory admission separately checks reachability.
    Groups are constructed by canonical Plan bytes, not just a digest. Identical
    Plans with different continuation states, including STOP/no-op, stay distinct.
    """
    if type(edit_count) is not int or not 0 <= edit_count <= V.MAX_EDITS_PER_EPOCH:
        raise ValueError("edit count outside V0 cap")
    if type(last_key) is not int or not -1 <= last_key < V.N_KEYS:
        raise ValueError("invalid last key")
    masks = Masks(state, plan, resolver=_MemoResolver(state), **SETTINGS)
    forced = edit_count == V.MAX_EDITS_PER_EPOCH
    legal = [0] if forced else [a for a in masks.legal_actions() if a == 0 or V.plan_key(a) > last_key]
    if len(legal) != len(set(legal)):
        raise ValueError("duplicate native legal IDs")
    legal.sort()
    classes: list[dict] = []
    group_index: dict[tuple, int] = {}
    features = []
    for i, action in enumerate(legal):
        terminal = action == V.STOP_ID
        successor = plan if terminal else plan.apply(action, masks.resolver)
        key = last_key if terminal else V.plan_key(action)
        used = edit_count if terminal else edit_count + 1
        exact = (_json(successor.canonical()), key, used, terminal)
        if exact not in group_index:
            group_index[exact] = len(classes)
            classes.append({"indices": [], "successor_plan_sha256": successor.digest(),
                            "continuation": continuation_signature(successor, key, used, terminal)})
        classes[group_index[exact]]["indices"].append(i)
        row = phi(state, plan, action, masks, **SETTINGS)
        if len(row) != len(PHI_NAMES) or not all(math.isfinite(v) for v in row):
            raise ValueError("invalid phi row")
        features.append(row)
    return {"legal_ids": legal, "phi": features, "classes": classes,
            "forced_stop": forced, "phi_names": list(PHI_NAMES)}


def _residual(wanted: Plan, reached: Plan) -> dict:
    """Include every Plan component, including changes not expressible in one epoch."""
    return {
        "cohorts": {V.MODULES[m]["key"]: wanted.cohorts.get(m, 0) - reached.cohorts.get(m, 0)
                    for m in sorted(wanted.cohorts.keys() | reached.cohorts.keys())
                    if wanted.cohorts.get(m, 0) != reached.cohorts.get(m, 0)},
        "regimes": {V.MODULES[m]["key"]: {"wanted": wanted.regime(m), "reached": reached.regime(m)}
                    for m in sorted(wanted.cohorts.keys() | reached.cohorts.keys()) if wanted.regime(m) != reached.regime(m)},
        "land": None if wanted.land == reached.land else {"wanted": wanted.land, "reached": reached.land},
        "market": {p: {"wanted": list(wanted.overlay(i)), "reached": list(reached.overlay(i))}
                   for i, p in enumerate(V.PRODUCTS) if wanted.overlay(i) != reached.overlay(i)},
    }


def verify_request(request: dict) -> dict:
    """Cheap reachability + compiler admission, before expensive candidate features.

    Returns explicit rejections with first failure and actual partial reached Plan.
    No edit_trajectory conversion, canonicalization, truncation, or game execution.
    Malformed serialization raises; well-formed but unadmissible targets are retained.
    """
    for field in IDENTITY_FIELDS:
        if not isinstance(request.get(field), str) or not request[field]:
            raise ValueError(f"nonempty {field} required")
    initial = Plan.from_canonical(request["initial_plan"])
    wanted = Plan.from_canonical(request["intended_plan"])
    state = State(request["observation"])
    actions = request["actions"]
    if not isinstance(actions, list) or any(type(a) is not int or not 0 <= a < V.N_ACTIONS for a in actions):
        raise ValueError("actions must be a list of native integer IDs")
    reached, last, used, stopped = initial.copy(), -1, 0, False
    prefixes = []
    verification = {"compiler_feasible": False, "unplaced": [], "game_steps": 0,
                    "executor_checked": False, "full_game_outcome": None}

    def result(reason=None, failure_index=None, detail=None):
        return {"status": "ADMITTED" if reason is None else "REJECTED", "reason": reason,
                "failure_index": failure_index, "detail": detail,
                "identity": {f: request[f] for f in IDENTITY_FIELDS} | {"learner_seat": state.player, "dawn": state.day},
                "actions": list(actions), "nonstop_edits": used,
                "initial_plan": initial.canonical(), "intended_plan": wanted.canonical(),
                "reached_plan": reached.canonical(), "reached_plan_sha256": reached.digest(),
                "residual": _residual(wanted, reached), "verification": verification,
                "examples": [], "prefixes": prefixes,
                "trainer_only": {"source_request_sha256": _digest(request),
                                 "replay_observation": copy.deepcopy(request["observation"]),
                                 "provenance": copy.deepcopy(request.get("provenance", {})),
                                 "teacher_kind": request.get("teacher_kind")}}

    if request.get("teacher_kind") != "explicit_native_v0":
        return result("NOT_NATIVE_V0")
    if state.step != 0 or initial != Plan():
        return result("OPENING_ONLY", detail="Later dawns/nonempty initial Plans need separately admitted compiler memory")
    for index, action in enumerate(actions):
        if stopped:
            return result("AFTER_STOP", index)
        if action != 0 and used == V.MAX_EDITS_PER_EPOCH:
            return result("EDIT_CAP", index)
        if action != 0 and V.plan_key(action) <= last:
            return result("NONCANONICAL_KEY", index)
        masks = Masks(state, reached, **SETTINGS)
        if not masks.legal(action):
            return result("ILLEGAL_PREFIX_ACTION", index)
        before = {"plan": reached.canonical(), "last_key": last, "edit_count": used}
        if action == 0:
            stopped = True
        else:
            reached = reached.apply(action, masks.resolver)
            last, used = V.plan_key(action), used + 1
        prefixes.append({**before, "action": action, "successor_plan_sha256": reached.digest(),
                         "continuation": continuation_signature(reached, last, used, stopped)})
    if not stopped:
        return result("MISSING_STOP", len(actions))
    if reached != wanted:
        return result("TARGET_MISMATCH", len(actions) - 1)
    try:
        program, projection, allocation = compile_opening(reached, request["observation"], hands_mode="daily", **SETTINGS)
    except ValueError as exc:
        return result("COMPILER_REFUSAL", detail=str(exc))
    verification.update({"compiler_feasible": projection.feasible, "unplaced": unplaced(allocation),
                         "program_sha256": _digest(program), "program_commands": len(program),
                         "binding": projection.binding, "prefix_legal": True, "target_exact": True})
    if not projection.feasible or verification["unplaced"]:
        return result("COMPILER_INFEASIBLE")
    return result()


def source_identity() -> dict:
    """Bind the source closure and rules used here, not just an informal version name."""
    paths = set((ROOT / "src/farm_neat/economy").rglob("*.py"))
    paths.update((ROOT / "src/farm_neat/game").glob("*.py"))
    paths.update([Path(__file__), V.TAPES_PATH, ROOT / "src/farm_neat/game/kaggriculture.json",
                  ROOT / "src/farm_neat/evaluation/scenarios.py", ROOT / "src/farm_neat/evaluation/common.py",
                  ROOT / "project/configs/scenarios.json",
                  ROOT / "docs/contracts/training.md", ROOT / "docs/guides/action-space-v0.md"])
    # project/review.toml is the task schedule, not data semantics: binding it made accepting this
    # task (a schedule edit) invalidate the accepted export.
    sources = {str(p.relative_to(ROOT)): _file_sha(p) for p in sorted(paths)}
    feature = {"actor_schema": ACTOR_SCHEMA, "phi_names": list(PHI_NAMES),
               "projection_sha256": sources["src/farm_neat/economy/v0/projection.py"],
               "actor_exporter_sha256": sources["src/farm_neat/training/v0_targets.py"], "settings": SETTINGS}
    return {"schema": SCHEMA, "action_schema": V.SCHEMA, "action_schema_sha256": V.ACTION_SCHEMA_SHA256,
            "sources": sources, "source_bundle_sha256": _digest(sources),
            "compiler_sha256": sources["src/farm_neat/economy/v0/compiler.py"],
            "feature_schema_sha256": _digest(feature), "feature_schema": feature,
            "compiler": {"labor_factor": 4.0, "hands_mode": "daily", "flow_belief": SETTINGS["belief"]},
            "max_nonstop_edits": V.MAX_EDITS_PER_EPOCH}


def _check_groups(requests: list[dict], split_id: str) -> None:
    if not isinstance(split_id, str) or not split_id:
        raise ValueError("explicit split identity required")
    ids = [r["trajectory_id"] for r in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trajectory IDs")
    for field in ("physical_world_id", "source_group", "seed_group"):
        groups = {}
        for request in requests:
            group, split = request[field], request["split"]
            if group in groups and groups[group] != split:
                raise ValueError(f"split leakage in {field}: {group}")
            groups[group] = split


def export_dataset(requests: Iterable[dict], output: str | Path, *, split_id: str,
                   expected_nontrivial: int | None = None) -> dict:
    """Deterministic JSONL dataset; no existing destination may be overwritten.

    A bounded LRU shares only byte-identical consumed State+Plan+key+count contexts.
    Changed prefixes always get fresh Masks, phi and exact successor classes. Cache
    is local to this export and never merges distinct raw observations via actor
    feature equality. Output does not depend on cache size or visitation history.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    requests = sorted(copy.deepcopy(list(requests)), key=lambda r: r["trajectory_id"])
    _check_groups(requests, split_id)
    identity = source_identity()
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter(requested=len(requests), admitted=0, rejected=0, nontrivial=0, stop_controls=0, prefixes=0)
    reasons, families, producers, regimes, primitives, land_timings = (Counter() for _ in range(6))
    groups, cache = [], OrderedDict()
    stop_counts = Counter(sampled=0, forced_cap=0)
    target_alias_rows, max_edits, alias_changes = 0, 0, 0
    reference_aliases: dict[str, dict[int, tuple[int, ...]]] = {}
    with (output / "prefixes.jsonl").open("w", encoding="utf-8", newline="\n") as examples, \
         (output / "trajectories.jsonl").open("w", encoding="utf-8", newline="\n") as accepted, \
         (output / "rejections.jsonl").open("w", encoding="utf-8", newline="\n") as rejected:
        for request in requests:
            verified = verify_request(request)
            groups.append(verified["identity"] | {"status": verified["status"]})
            if verified["status"] != "ADMITTED":
                counts["rejected"] += 1
                reasons[verified["reason"]] += 1
                rejected.write(_json(verified) + "\n")
                continue
            counts["admitted"] += 1
            counts["nontrivial" if verified["nonstop_edits"] else "stop_controls"] += 1
            max_edits = max(max_edits, verified["nonstop_edits"])
            state = State(request["observation"])
            opening_aliases = reference_aliases.setdefault(_json(vars(state)), {})
            for index, prefix in enumerate(verified["prefixes"]):
                plan = Plan.from_canonical(prefix["plan"])
                key = _json({"state": vars(state), "plan": prefix["plan"], "last_key": prefix["last_key"],
                             "edit_count": prefix["edit_count"], "settings": SETTINGS})
                if key not in cache:
                    cache[key] = prefix_support(state, plan, prefix["last_key"], prefix["edit_count"])
                    if len(cache) > 128:
                        cache.popitem(last=False)
                cache.move_to_end(key)
                support = cache[key]
                action = prefix["action"]
                if action not in support["legal_ids"]:
                    raise ValueError("admitted trace disagrees with prefix support")
                slot = support["legal_ids"].index(action)
                selected = next(c for c in support["classes"] if slot in c["indices"])
                if (selected["successor_plan_sha256"] != prefix["successor_plan_sha256"]
                        or selected["continuation"] != prefix["continuation"]):
                    raise ValueError("successor reconstruction mismatch")
                terminal = action == 0
                stop_kind = ("forced_cap" if support["forced_stop"] else "sampled") if terminal else "not_stop"
                if terminal:
                    stop_counts[stop_kind] += 1
                if len(selected["indices"]) > 1:
                    target_alias_rows += 1
                # A source-bound coverage diagnostic, not a global alias law.
                for c in support["classes"]:
                    members = tuple(support["legal_ids"][i] for i in c["indices"])
                    for a in members:
                        if prefix["edit_count"] == 0:
                            opening_aliases.setdefault(a, members)
                        elif a in opening_aliases and opening_aliases[a] != members:
                            alias_changes += 1
                fields = V.decode(action)
                families[fields[0]] += 1
                if fields[0] == "PROD":
                    producers[V.MODULES[fields[1]]["producer"]] += 1
                elif fields[0] == "SERVICE":
                    regimes[V.REGIMES[fields[2]]] += 1
                elif fields[0] == "LAND":
                    land_timings[V.LAND_TIMINGS[fields[1]]] += 1
                elif fields[0] == "MARKET":
                    primitives[V.PRIMITIVES[fields[2]]] += 1
                row = {
                    "schema": SCHEMA,
                    "identity": verified["identity"] | {"prefix_index": index,
                        "source_bundle_sha256": identity["source_bundle_sha256"],
                        "action_schema_sha256": V.ACTION_SCHEMA_SHA256,
                        "compiler_sha256": identity["compiler_sha256"],
                        "feature_schema_sha256": identity["feature_schema_sha256"], "split_id": split_id},
                    "actor": {"public_observation": actor_observation(state), "own_memory": {},
                              "prefix_plan": prefix["plan"], "last_key": prefix["last_key"],
                              "nonstop_edit_count": prefix["edit_count"]},
                    "candidates": support,
                    "target": {"class_indices": list(selected["indices"]),
                               "successor_plan_sha256": selected["successor_plan_sha256"],
                               "continuation": selected["continuation"], "stop_kind": stop_kind,
                               # KEEP/STOP controls are diagnostics, not teachers: at the opening they would
                               # teach STOP against every edit teacher (duplicate returns ee382d3e, 2e2ce843).
                               "loss_weight": 0 if support["forced_stop"] or not verified["nonstop_edits"] else 1},
                    "trainer_only": {"replay_action": action,
                                     "source_request_sha256": verified["trainer_only"]["source_request_sha256"]},
                }
                examples.write(_json(row) + "\n")
                counts["prefixes"] += 1
            verified.pop("examples")
            accepted.write(_json(verified) + "\n")
    passed = expected_nontrivial is None or counts["nontrivial"] == expected_nontrivial
    manifest = {
        **identity, "status": "EXPORTED" if passed else "COUNT_GATE_FAILED", "counts": dict(counts),
        "expected_nontrivial": expected_nontrivial, "rejection_reasons": dict(reasons),
        "split_id": split_id, "groups": groups, "groups_sha256": _digest(groups),
        "requests_sha256": _digest(requests),
        "coverage": {"action_families": dict(families), "production_families": dict(producers),
                     "service_regimes": dict(regimes), "market_primitives": dict(primitives),
                     "land_timings": dict(land_timings),
                     "uncovered_land_timings": sorted(set(V.LAND_TIMINGS) - land_timings.keys()),
                     "uncovered_action_families": sorted({"PROD", "SERVICE", "LAND", "MARKET", "STOP"} - families.keys()),
                     "uncovered_production_families": sorted({m["producer"] for m in V.MODULES} - producers.keys()),
                     "uncovered_service_regimes": sorted(set(V.REGIMES) - regimes.keys()),
                     "uncovered_market_primitives": sorted(set(V.PRIMITIVES) - primitives.keys()),
                     "stop": dict(stop_counts), "target_alias_rows": target_alias_rows,
                     "observed_prefix_alias_membership_changes": alias_changes, "max_nonstop_edits": max_edits},
        "files": {name: {"sha256": _file_sha(output / name), "bytes": (output / name).stat().st_size}
                  for name in ("prefixes.jsonl", "trajectories.jsonl", "rejections.jsonl")},
        "exposure": {"physical_worlds": len({r["physical_world_id"] for r in requests}),
                     "physical_games_completed": 0, "game_steps": 0, "model_calls": 0, "optimizer_updates": 0},
        "limits": ["Opening-only explicit native teachers; caller attestation is not independent provenance verification",
                   "Masks and compilation checked; executor realization and WDL not checked",
                   "No automatic admission of reached substitutes, legacy teachers, or held-out evaluation",
                   "Actor allowlist is a versioned bootstrap representation, not a sufficiency theorem",
                   "FRONTRUN/PIGGYBACK and LAST_PROFITABLE remain reserved/masked by existing code"],
    }
    # Cache counters are intentionally not dataset identity or semantic measurements.
    (output / "MANIFEST.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    return manifest


def smoke_requests() -> list[dict]:
    """32 declared native traces, 2 KEEP controls, 4 retained negative controls.

    One native router-dairy opening, both seats; no calls to step/run. These are
    disposable reachable plumbing teachers, not economic winners or legacy fits.
    Family coverage is explicit; unexercised regimes/market primitives stay visible.
    """
    from farm_neat.evaluation.scenarios import ScenarioEnvironment, catalog
    spec = catalog()["router-dairy"]
    env = ScenarioEnvironment(spec)
    observations = [env.agent_observation(seat) for seat in (0, 1)]
    world = _digest({"scenario": spec, "kind": "native-opening-only"})
    rows = []

    def add(name, actions, target, seat=0, teacher_kind="explicit_native_v0"):
        rows.append({"teacher_kind": teacher_kind, "teacher_id": "native-smoke/" + name,
                     "trajectory_id": name, "physical_world_id": world,
                     "source_group": "native-smoke/router-dairy", "seed_group": str(spec["seed"]),
                     "split": "smoke_train", "observation": copy.deepcopy(observations[seat]),
                     "initial_plan": Plan().canonical(), "actions": list(actions),
                     "intended_plan": target.canonical(),
                     "provenance": {"kind": "explicit_native_plumbing_teacher", "scenario": spec,
                                    "observed_outcome": None, "economic_strength_admitted": False}})

    modules = [m for m in V.MODULES if m["birth"] == 0]
    for n, m in enumerate(modules):
        for variant in (0, 1):
            module, product = m["id"], V.PRODUCTS.index(m["product"])
            regimes = {module: 2} if variant and not m["animal"] else {}
            overlay = (V.PRIMITIVES.index("HOLD"), V.HORIZONS.index("TERMINAL"), 0)
            actions = [V.encode("PROD", module, V.DELTAS.index("PLUS_ONE"))]
            if regimes:
                actions.append(V.encode("SERVICE", module, 2))
            actions += [V.encode("LAND", variant), V.encode("MARKET", product, *overlay), 0]
            add(f"native-{n:02d}-{variant}", actions, Plan({module: 1}, regimes, variant, {product: overlay}), variant)
    for n in range(6):
        a, b = modules[n]["id"], modules[n + 6]["id"]
        actions = [V.encode("PROD", m, V.DELTAS.index("PLUS_ONE")) for m in sorted((a, b))]
        actions += [V.encode("LAND", 1), 0]
        add(f"mixed-{n:02d}", actions, Plan({a: 1, b: 1}, land=1), n % 2)
    wheat = V.MODULE_BY_KEY["WHEAT:watered:b00"]
    jump = V.encode("PROD", wheat, V.DELTAS.index("MAX_FEASIBLE"))
    add("prefix-alias-jump", [jump, 0], Plan({wheat: 15}))
    cap_modules = [V.MODULE_BY_KEY[f"WHEAT:watered:b{i:02d}"] for i in range(16)]
    cap_actions = [V.encode("PROD", m, V.DELTAS.index("PLUS_ONE")) for m in cap_modules]
    add("cap-sixteen", cap_actions + [0], Plan({m: 1 for m in cap_modules}))
    add("stop-seat-0", [0], Plan())
    add("stop-seat-1", [0], Plan(), 1)
    cow = V.MODULE_BY_KEY["COW:fed_cared:b00"]
    add("reject-cow-five", [V.encode("PROD", cow, 2), 0], Plan({cow: 5}))
    late = V.MODULE_BY_KEY["WHEAT:watered:b04"]
    add("reject-stale-mask", [jump, V.encode("PROD", late, 2), 0], Plan({wheat: 15, late: 1}))
    seventeenth = V.MODULE_BY_KEY["WHEAT:watered:b16"]
    add("reject-cap-seventeen", cap_actions + [V.encode("PROD", seventeenth, 2), 0], Plan({m: 1 for m in cap_modules + [seventeenth]}))
    add("reject-legacy-conversion", [0], Plan(), teacher_kind="legacy_conversion")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="export the fixed 32 reachable native trajectories and controls")
    group.add_argument("--input", type=Path, help="explicit source-bound native request document")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; refusing overwrite")
    try:
        if args.smoke:
            requests, split_id, expected = smoke_requests(), "v0-bootstrap-smoke-32-v1", 32
        else:
            document = json.loads(args.input.read_text(encoding="utf-8"))
            if document.get("schema") != REQUEST_SCHEMA:
                raise ValueError("native request document schema required")
            requests, split_id, expected = document["requests"], document["split_id"], None
        manifest = export_dataset(requests, args.output, split_id=split_id, expected_nontrivial=expected)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"],
                      "coverage": manifest["coverage"], "rejections": manifest["rejection_reasons"],
                      "manifest_sha256": _file_sha(args.output / "MANIFEST.json")}, sort_keys=True, indent=2))
    return 0 if manifest["status"] == "EXPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
