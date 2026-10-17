"""The persistent canonical plan and its edit set.

P = (production cohorts, service regimes, land authorization, market overlays). A plan is
a value: edits return new plans. An epoch's edits form a set in canonical key order, each
key at most once before STOP; repeated edits of one key collapse to the final value.
Deltas that need a projection (NEXT_BINDING, MAX_FEASIBLE) take it from a resolver.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass

from farm_neat.economy.v0 import vocabulary as V

CLEAR_OVERLAY = (V.PRIMITIVES.index("CLEAR"), V.HORIZONS.index("NOW"), 0)


class Plan:
    __slots__ = ("cohorts", "regimes", "land", "market")

    def __init__(self, cohorts=None, regimes=None, land=None, market=None):
        self.cohorts = {int(k): int(v) for k, v in (cohorts or {}).items() if int(v) > 0}
        self.regimes = {int(k): int(v) for k, v in (regimes or {}).items() if int(k) in self.cohorts}
        self.land = None if land is None else int(land)
        self.market = {int(k): tuple(int(x) for x in v) for k, v in (market or {}).items()}
        for module in self.cohorts:
            V._check(module, V.N_MODULES)
        for regime in self.regimes.values():
            V._check(regime, len(V.REGIMES))
        if self.land is not None:
            V._check(self.land, len(V.LAND_TIMINGS))
        for product, (primitive, horizon, rank) in self.market.items():
            V._check(product, len(V.PRODUCTS)); V._check(primitive, len(V.PRIMITIVES))
            V._check(horizon, len(V.HORIZONS)); V._check(rank, V.RANKS)

    def canonical(self) -> dict:
        return {"schema": V.SCHEMA, "action_schema_sha256": V.ACTION_SCHEMA_SHA256,
                "cohorts": {V.MODULES[m]["key"]: n for m, n in sorted(self.cohorts.items())},
                "regimes": {V.MODULES[m]["key"]: V.REGIMES[r] for m, r in sorted(self.regimes.items())},
                "land": None if self.land is None else V.LAND_TIMINGS[self.land],
                "market": {V.PRODUCTS[p]: [V.PRIMITIVES[a], V.HORIZONS[h], k]
                           for p, (a, h, k) in sorted(self.market.items())}}

    @classmethod
    def from_canonical(cls, value: dict) -> "Plan":
        if not isinstance(value, dict) or value.get("schema") != V.SCHEMA or value.get("action_schema_sha256") != V.ACTION_SCHEMA_SHA256:
            raise ValueError("V0 plan schema required")
        if set(value) != {"schema", "action_schema_sha256", "cohorts", "regimes", "land", "market"}:
            raise ValueError("closed V0 plan fields required")
        return cls({V.MODULE_BY_KEY[k]: n for k, n in value["cohorts"].items()},
                   {V.MODULE_BY_KEY[k]: V.REGIMES.index(r) for k, r in value["regimes"].items()},
                   None if value["land"] is None else V.LAND_TIMINGS.index(value["land"]),
                   {V.PRODUCTS.index(p): (V.PRIMITIVES.index(a), V.HORIZONS.index(h), k)
                    for p, (a, h, k) in value["market"].items()})

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def copy(self) -> "Plan":
        return Plan(dict(self.cohorts), dict(self.regimes), self.land, dict(self.market))

    def __eq__(self, other):
        return isinstance(other, Plan) and self.canonical() == other.canonical()

    def regime(self, module: int) -> int:
        return self.regimes.get(module, 0)

    def overlay(self, product: int) -> tuple[int, int, int]:
        return self.market.get(product, CLEAR_OVERLAY)

    def instances(self):
        """(module, index) for every cohort instance: animals first, then birth, then module key."""
        rows = [(m, i) for m, n in self.cohorts.items() for i in range(n)]
        rows.sort(key=lambda r: (0 if V.MODULES[r[0]]["animal"] else 1, V.MODULES[r[0]]["birth"], V.MODULES[r[0]]["key"], r[1]))
        return rows

    def apply(self, action: int, resolver=None) -> "Plan":
        """One edit. `resolver(plan, module, delta_name)` returns the projected count."""
        fields = V.decode(action)
        kind = fields[0]
        new = self.copy()
        if kind == "STOP":
            return new
        if kind == "PROD":
            module, delta = fields[1], V.DELTAS[fields[2]]
            current = new.cohorts.get(module, 0)
            if delta == "DROP_ALL":
                count = 0
            elif delta == "MINUS_ONE":
                count = current - 1
            elif delta == "PLUS_ONE":
                count = current + 1
            else:
                if resolver is None:
                    raise ValueError(f"{delta} needs a projection resolver")
                count = int(resolver(new, module, delta))
            if count < 0:
                raise ValueError("cohort count below zero")
            if count:
                new.cohorts[module] = count
            else:
                new.cohorts.pop(module, None); new.regimes.pop(module, None)
        elif kind == "SERVICE":
            module, regime = fields[1], fields[2]
            if module not in new.cohorts:
                raise ValueError("SERVICE needs an existing cohort")
            if regime:
                new.regimes[module] = regime
            else:
                new.regimes.pop(module, None)
        elif kind == "LAND":
            new.land = fields[1]
        else:
            product, primitive, horizon, rank = fields[1:]
            if (primitive, horizon, rank) == CLEAR_OVERLAY:
                new.market.pop(product, None)
            else:
                new.market[product] = (primitive, horizon, rank)
        return new


def canonical_edit_set(actions: list[int]) -> list[int]:
    """Collapse an epoch's edits to one serialization: key order, last edit per key, STOP last.

    Edits after STOP are ignored. More than MAX_EDITS_PER_EPOCH distinct keys is refused.
    """
    last = {}
    for action in actions:
        if action == V.STOP_ID:
            break
        last[V.plan_key(action)] = action
    if len(last) > V.MAX_EDITS_PER_EPOCH:
        raise ValueError("more than MAX_EDITS_PER_EPOCH plan keys edited in one epoch")
    return [last[k] for k in sorted(last)] + [V.STOP_ID]


def apply_edits(plan: Plan, actions: list[int], resolver=None) -> Plan:
    for action in canonical_edit_set(actions):
        plan = plan.apply(action, resolver)
    return plan


def edit_trajectory(target: Plan, current: Plan | None = None, resolver=None) -> tuple[list[int], Plan, dict]:
    """The canonical edits that move `current` toward `target` in one epoch.

    Counts use PLUS_ONE / MINUS_ONE / DROP_ALL when they reach the target in one edit, else the
    jump (NEXT_BINDING or MAX_FEASIBLE) whose projected count is nearest the target without
    overshooting, else PLUS_ONE. Returns (edits, reached plan, residual counts by module key).
    """
    current = Plan() if current is None else current
    edits, residual = [], {}
    # Earliest births first: the per-epoch cap drops the keys later epochs can still edit.
    keys = sorted(set(current.cohorts) | set(target.cohorts), key=lambda m: (V.MODULES[m]["birth"], m))
    reached = current.copy()
    for module in keys:
        want, have = target.cohorts.get(module, 0), current.cohorts.get(module, 0)
        if want == have:
            continue
        if want == 0:
            choice = V.encode("PROD", module, V.DELTAS.index("DROP_ALL"))
        elif want == have - 1:
            choice = V.encode("PROD", module, V.DELTAS.index("MINUS_ONE"))
        elif want == have + 1:
            choice = V.encode("PROD", module, V.DELTAS.index("PLUS_ONE"))
        else:
            best = None
            if resolver is not None:
                for delta in ("NEXT_BINDING", "MAX_FEASIBLE"):
                    projected = int(resolver(reached, module, delta))
                    if have < projected <= want or want < projected < have:
                        if best is None or abs(want - projected) < abs(want - best[0]):
                            best = (projected, V.encode("PROD", module, V.DELTAS.index(delta)))
            choice = best[1] if best else V.encode("PROD", module, V.DELTAS.index("PLUS_ONE" if want > have else "MINUS_ONE"))
        edits.append(choice)
        reached = reached.apply(choice, resolver)
        got = reached.cohorts.get(module, 0)
        if got != want:
            residual[V.MODULES[module]["key"]] = want - got
    for module, regime in sorted(target.regimes.items()):
        if module in reached.cohorts and reached.regime(module) != regime:
            edits.append(V.encode("SERVICE", module, regime))
            reached = reached.apply(edits[-1])
    if target.land != current.land and target.land is not None:
        edits.append(V.encode("LAND", target.land))
        reached = reached.apply(edits[-1])
    for product in range(len(V.PRODUCTS)):
        if target.overlay(product) != reached.overlay(product):
            edits.append(V.encode("MARKET", product, *target.overlay(product)))
            reached = reached.apply(edits[-1])
    if len(edits) > V.MAX_EDITS_PER_EPOCH:
        residual["__edits_beyond_cap__"] = len(edits) - V.MAX_EDITS_PER_EPOCH
        kept = edits[:V.MAX_EDITS_PER_EPOCH]
        reached = apply_edits(current, kept, resolver)
        edits = kept
    return canonical_edit_set(edits), reached, residual


# Independently usable BUILD slice, NOT an extension of the active V0 action
# lattice/Plan codec. The full preserving-retirement contract has no engine
# implementation. Do not store this under an old Plan or checkpoint identity.
LIVESTOCK_CONTROL_SCHEMA = "farm-v0-born-livestock-control-v1"
LIVESTOCK_SERVICE_MODES = tuple(
    (f"FEED_{feed}_CARE_{care}", feed, care)
    for feed in ("DAILY", "WHEN_UNFED", "OFF") for care in ("DAILY", "OFF")
)
LIVESTOCK_RETIRE = "RETIRE_FUTURE_SERVICE_PRODUCTION"
LIVESTOCK_CONTROL_SCHEMA_SHA256 = hashlib.sha256(json.dumps({
    "schema": LIVESTOCK_CONTROL_SCHEMA, "service_modes": LIVESTOCK_SERVICE_MODES,
    "retire": LIVESTOCK_RETIRE,
    "identity": "GAME_OWNER_MODULE_INSTANCE_TILE_ACTUAL_PLACED_DAY",
    "retirement_capability": "UNSUPPORTED_WITH_PRESERVATION_IN_CURRENT_ENGINE",
}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
_CONTROL_FIELDS = frozenset({"schema", "control_schema_sha256", "game_id", "player",
                             "module", "authorized_step", "mode", "instances", "content_sha256"})


def _control_need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _control_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _control_observation(observation: dict) -> tuple[int, int, dict]:
    """Read only the current owner's public asset fields, never economics or rival state."""
    _control_need(isinstance(observation, dict), "LIVESTOCK_CONTROL_OBSERVATION")
    player, step = observation.get("player"), observation.get("step")
    _control_need(type(player) is int and player in (0, 1)
                  and type(step) is int and 0 <= step <= V.LAST_STEP,
                  "LIVESTOCK_CONTROL_OBSERVATION")
    farms = observation.get("farms")
    _control_need(isinstance(farms, list) and len(farms) == 2
                  and isinstance(farms[player], dict), "LIVESTOCK_CONTROL_OBSERVATION")
    farm = farms[player]
    quads, tiles = farm.get("unlocked_quadrants"), farm.get("tiles")
    _control_need(isinstance(quads, list) and bool(quads)
                  and all(type(q) is str and q in V.QUADRANTS for q in quads)
                  and len(set(quads)) == len(quads)
                  and isinstance(tiles, list) and len(tiles) == 10
                  and all(isinstance(row, list) and len(row) == 10 for row in tiles),
                  "LIVESTOCK_CONTROL_OBSERVATION")
    return player, step, farm


@dataclass(frozen=True)
class BornLivestockControl:
    """Explicit service intent bound to an entire *actual* already-born cohort.

    This closed data-only codec is staged separately from Plan and V0Continuation;
    neither silently accepts it under its existing schema. Policy code must bind
    an explicit mode. Merely restoring/validating this value never revises it.
    Instances are immutable (index, x, y, actual placed_day) tuples, ordered by
    cohort index. Nominal birth is not substituted for delayed actual placement.
    A digest detects changed bytes; it is not a signature proving policy choice.
    """

    game_id: str
    player: int
    module: int
    authorized_step: int
    mode: str
    instances: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self):
        _control_need(type(self.game_id) is str and bool(self.game_id.strip()),
                      "LIVESTOCK_CONTROL_GAME")
        _control_need(type(self.player) is int and self.player in (0, 1),
                      "LIVESTOCK_CONTROL_OWNER")
        _control_need(type(self.module) is int and 0 <= self.module < V.N_MODULES
                      and V.MODULES[self.module]["animal"], "LIVESTOCK_CONTROL_COHORT")
        _control_need(type(self.authorized_step) is int
                      and 0 <= self.authorized_step <= V.LAST_STEP
                      and self.authorized_step % 24 == 0, "LIVESTOCK_CONTROL_AUTHORIZATION")
        modes = tuple(row[0] for row in LIVESTOCK_SERVICE_MODES) + (LIVESTOCK_RETIRE,)
        _control_need(type(self.mode) is str and self.mode in modes, "LIVESTOCK_CONTROL_MODE")
        _control_need(type(self.instances) is tuple and 1 <= len(self.instances) <= 100,
                      "LIVESTOCK_CONTROL_INSTANCE")
        seen = set()
        birth = V.MODULES[self.module]["birth"]
        for expected, row in enumerate(self.instances):
            _control_need(type(row) is tuple and len(row) == 4
                          and all(type(v) is int for v in row), "LIVESTOCK_CONTROL_INSTANCE")
            index, x, y, placed_day = row
            _control_need(index == expected and 0 <= x < 10 and 0 <= y < 10
                          and (x, y) not in seen
                          and birth <= placed_day <= self.authorized_step // 24,
                          "LIVESTOCK_CONTROL_INSTANCE")
            seen.add((x, y))

    def payload(self) -> dict:
        body = {"schema": LIVESTOCK_CONTROL_SCHEMA,
                "control_schema_sha256": LIVESTOCK_CONTROL_SCHEMA_SHA256,
                "game_id": self.game_id, "player": self.player,
                "module": V.MODULES[self.module]["key"],
                "authorized_step": self.authorized_step, "mode": self.mode,
                "instances": [{"index": index, "tile": [x, y], "placed_day": day}
                              for index, x, y, day in self.instances]}
        return {**body, "content_sha256": _control_digest(body)}

    @classmethod
    def from_payload(cls, value: dict) -> "BornLivestockControl":
        _control_need(isinstance(value, dict)
                      and value.get("schema") == LIVESTOCK_CONTROL_SCHEMA
                      and value.get("control_schema_sha256") == LIVESTOCK_CONTROL_SCHEMA_SHA256,
                      "LIVESTOCK_CONTROL_SCHEMA")
        _control_need(set(value) == _CONTROL_FIELDS, "LIVESTOCK_CONTROL_FIELDS")
        body = {k: v for k, v in value.items() if k != "content_sha256"}
        try:
            valid_hash = value["content_sha256"] == _control_digest(body)
        except (ValueError, TypeError):
            valid_hash = False
        _control_need(valid_hash, "LIVESTOCK_CONTROL_HASH")
        key = value["module"]
        _control_need(type(key) is str and key in V.MODULE_BY_KEY,
                      "LIVESTOCK_CONTROL_COHORT")
        rows = value["instances"]
        _control_need(isinstance(rows, list), "LIVESTOCK_CONTROL_INSTANCE")
        instances = []
        for row in rows:
            _control_need(isinstance(row, dict) and set(row) == {"index", "tile", "placed_day"}
                          and isinstance(row["tile"], list) and len(row["tile"]) == 2,
                          "LIVESTOCK_CONTROL_INSTANCE")
            instances.append((row["index"], *row["tile"], row["placed_day"]))
        return cls(value["game_id"], value["player"], V.MODULE_BY_KEY[key],
                   value["authorized_step"], value["mode"], tuple(instances))

    @classmethod
    def bind(cls, plan: Plan, module: int, mode: str, observation: dict,
             allocation: dict, started: dict, *, game_id: str) -> "BornLivestockControl":
        player, step, farm = _control_observation(observation)
        _control_need(isinstance(plan, Plan) and type(module) is int
                      and 0 <= module < V.N_MODULES and V.MODULES[module]["animal"]
                      and module in plan.cohorts, "LIVESTOCK_CONTROL_COHORT")
        _control_need(isinstance(allocation, dict) and isinstance(started, dict),
                      "LIVESTOCK_CONTROL_ALLOCATION")
        count = plan.cohorts[module]
        _control_need(type(count) is int and 1 <= count <= 100, "LIVESTOCK_CONTROL_COHORT")
        rows = []
        for index in range(count):
            tile = allocation.get((module, index))
            _control_need(type(tile) is tuple and len(tile) == 2
                          and all(type(v) is int and 0 <= v < 10 for v in tile),
                          "LIVESTOCK_CONTROL_ALLOCATION")
            x, y = tile
            asset = farm["tiles"][y][x]
            _control_need(isinstance(asset, dict), "LIVESTOCK_CONTROL_ASSET")
            rows.append((index, x, y, asset.get("placed_day")))
        result = cls(game_id, player, module, step, mode, tuple(rows))
        result.validate_context(plan, observation, allocation, started, game_id=game_id)
        return result

    def validate_context(self, plan: Plan, observation: dict, allocation: dict,
                         started: dict, *, game_id: str) -> None:
        """Fail before scoring/publication; never rebind to a replacement or foreign animal."""
        # Revalidate direct instances as well as restored payloads.
        self.__post_init__()
        player, step, farm = _control_observation(observation)
        _control_need(game_id == self.game_id, "LIVESTOCK_CONTROL_GAME")
        _control_need(player == self.player, "LIVESTOCK_CONTROL_OWNER")
        _control_need(step >= self.authorized_step, "LIVESTOCK_CONTROL_AUTHORIZATION")
        _control_need(isinstance(plan, Plan) and type(plan.cohorts.get(self.module)) is int
                      and plan.cohorts[self.module] == len(self.instances)
                      and type(plan.regime(self.module)) is int and plan.regime(self.module) == 0,
                      "LIVESTOCK_CONTROL_COHORT")
        _control_need(isinstance(allocation, dict) and isinstance(started, dict),
                      "LIVESTOCK_CONTROL_ALLOCATION")
        # Dict equality alone aliases False with 0 and True with 1. Validate
        # identity pairs before comparing, and reject extra rows for this cohort.
        def pair(value):
            return type(value) is tuple and len(value) == 2 and all(type(v) is int for v in value)
        _control_need(all(pair(key) for key in allocation)
                      and all(pair(tile) and pair(key) for tile, key in started.items()),
                      "LIVESTOCK_CONTROL_ALLOCATION")
        expected = {(self.module, index): (x, y) for index, x, y, _day in self.instances}
        allocated = {key: tile for key, tile in allocation.items() if key[0] == self.module}
        begun = {tile: key for tile, key in started.items() if key[0] == self.module}
        _control_need(all(pair(tile) for tile in allocated.values()) and allocated == expected
                      and begun == {tile: key for key, tile in expected.items()},
                      "LIVESTOCK_CONTROL_ALLOCATION")
        producer = V.MODULES[self.module]["producer"]
        for index, x, y, placed_day in self.instances:
            key, tile = (self.module, index), (x, y)
            _control_need(allocation.get(key) == tile and started.get(tile) == key,
                          "LIVESTOCK_CONTROL_ALLOCATION")
            quadrant = ("N" if y < 5 else "S") + ("W" if x < 5 else "E")
            _control_need(quadrant in farm["unlocked_quadrants"], "LIVESTOCK_CONTROL_OWNER")
            asset = farm["tiles"][y][x]
            _control_need(isinstance(asset, dict) and asset.get("animal") == producer
                          and asset.get("kind") == V.ANIMALS[producer]["structure"]
                          and type(asset.get("placed_day")) is int
                          and asset["placed_day"] == placed_day, "LIVESTOCK_CONTROL_ASSET")
