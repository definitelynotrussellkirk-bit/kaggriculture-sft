"""Frozen V0 vocabulary: stable action ids, plan keys and the module table.

Ids never change meaning. Reserved choices (a regime or timing without a projection yet)
keep their id and stay masked until a theorem supplies the projection. Nothing here reads
game state; masks live in projection.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from farm_neat.game.constants import ANIMALS, CROPS, PRODUCTS as ENGINE_PRODUCTS

SCHEMA = "farm-v0-plan-actions-v1"
PRODUCTS = tuple(ENGINE_PRODUCTS)
SEASON_DAYS = 30
LAST_STEP = 718
# (producer, policy, first output day) in the order of K-LIFECYCLE-COLUMNS-302; the
# fertilized MELON mode is table-dominated by watered MELON and has no column.
MODES = (
    ("WHEAT", "watered", 4), ("WHEAT", "watered_fertilized", 4),
    ("CARROT", "watered", 3), ("CARROT", "watered_fertilized", 3),
    ("TOMATO", "watered", 8), ("TOMATO", "watered_fertilized", 8),
    ("STRAWBERRY", "watered", 10), ("STRAWBERRY", "watered_fertilized", 10),
    ("MELON", "watered", 10),
    ("GOOSE", "fed_cared", 4), ("COW", "fed_cared", 8), ("SHEEP", "fed_cared", 6),
)
DELTAS = ("DROP_ALL", "MINUS_ONE", "PLUS_ONE", "NEXT_BINDING", "MAX_FEASIBLE")
# STANDARD: the mode's tape (held-cap harvest for one-time crops, every event otherwise,
# replanted at the tape's own cadence to the season end). FIRST_ELIGIBLE: harvest a one-time
# crop at its first eligible yield (WHEAT, CARROT). ONE_CYCLE (and its first-eligible form):
# one plant lifecycle, then the tile is free; the tape is the STANDARD tape cut at its first
# replant. Animals have no exit: only STANDARD is legal for them.
REGIMES = ("STANDARD", "FIRST_ELIGIBLE", "ONE_CYCLE", "ONE_CYCLE_FIRST_ELIGIBLE")
LAND_TIMINGS = ("NOW", "NEXT_DAWN", "NEXT_BINDING", "LAST_PROFITABLE")
PRIMITIVES = ("CLEAR", "SELL", "HOLD", "CARRY", "FRONTRUN", "PIGGYBACK")
HORIZONS = ("NOW", "NEXT_DEMAND", "NEXT_REVEAL", "NEXT_DAWN", "TERMINAL")
RANKS = 101
QUADRANTS = ("NW", "NE", "SW", "SE")
LAND_PRICES = (1000, 2000, 4000)
SITES_PER_QUADRANT = 25
HAND_CAP = 10
MAX_EDITS_PER_EPOCH = 16


def _modules():
    rows = []
    for mode, (producer, policy, first) in enumerate(MODES):
        for birth in range(SEASON_DAYS):
            if birth + first > SEASON_DAYS - 1:
                continue
            animal = producer in ANIMALS
            rows.append({
                "id": len(rows), "mode": mode, "producer": producer, "policy": policy,
                "birth": birth, "first_output_day": first, "animal": animal,
                "product": ANIMALS[producer]["product"] if animal else producer,
                "key": f"{producer}:{policy}:b{birth:02d}",
                "default_regime": "every_event" if animal or CROPS[producer]["ongoing"] else "held_cap",
            })
    return tuple(rows)


MODULES = _modules()
MODULE_BY_KEY = {m["key"]: m["id"] for m in MODULES}
N_MODULES = len(MODULES)

STOP_ID = 0
PROD_BASE = 1
SERVICE_BASE = PROD_BASE + N_MODULES * len(DELTAS)
LAND_BASE = SERVICE_BASE + N_MODULES * len(REGIMES)
MARKET_BASE = LAND_BASE + len(LAND_TIMINGS)
N_ACTIONS = MARKET_BASE + len(PRODUCTS) * len(PRIMITIVES) * len(HORIZONS) * RANKS

# Plan keys: PROD module, SERVICE module, LAND, MARKET product; canonical order is the id.
KEY_PROD_BASE = 0
KEY_SERVICE_BASE = N_MODULES
KEY_LAND = 2 * N_MODULES
KEY_MARKET_BASE = KEY_LAND + 1
N_KEYS = KEY_MARKET_BASE + len(PRODUCTS)


def encode(kind: str, *fields: int) -> int:
    if kind == "STOP":
        return STOP_ID
    if kind == "PROD":
        module, delta = fields
        return PROD_BASE + _check(module, N_MODULES) * len(DELTAS) + _check(delta, len(DELTAS))
    if kind == "SERVICE":
        module, regime = fields
        return SERVICE_BASE + _check(module, N_MODULES) * len(REGIMES) + _check(regime, len(REGIMES))
    if kind == "LAND":
        (timing,) = fields
        return LAND_BASE + _check(timing, len(LAND_TIMINGS))
    if kind == "MARKET":
        product, primitive, horizon, rank = fields
        return MARKET_BASE + ((_check(product, len(PRODUCTS)) * len(PRIMITIVES) + _check(primitive, len(PRIMITIVES)))
                              * len(HORIZONS) + _check(horizon, len(HORIZONS))) * RANKS + _check(rank, RANKS)
    raise ValueError(f"unknown action kind {kind!r}")


def decode(action: int) -> tuple:
    if type(action) is not int or not 0 <= action < N_ACTIONS:
        raise ValueError("action id outside the frozen lattice")
    if action == STOP_ID:
        return ("STOP",)
    if action < SERVICE_BASE:
        module, delta = divmod(action - PROD_BASE, len(DELTAS))
        return ("PROD", module, delta)
    if action < LAND_BASE:
        module, regime = divmod(action - SERVICE_BASE, len(REGIMES))
        return ("SERVICE", module, regime)
    if action < MARKET_BASE:
        return ("LAND", action - LAND_BASE)
    rest, rank = divmod(action - MARKET_BASE, RANKS)
    rest, horizon = divmod(rest, len(HORIZONS))
    product, primitive = divmod(rest, len(PRIMITIVES))
    return ("MARKET", product, primitive, horizon, rank)


def plan_key(action: int) -> int | None:
    """The plan key an action edits (None for STOP); ids order the canonical edit set."""
    fields = decode(action)
    if fields[0] == "STOP":
        return None
    if fields[0] == "PROD":
        return KEY_PROD_BASE + fields[1]
    if fields[0] == "SERVICE":
        return KEY_SERVICE_BASE + fields[1]
    if fields[0] == "LAND":
        return KEY_LAND
    return KEY_MARKET_BASE + fields[1]


def describe(action: int) -> str:
    fields = decode(action)
    kind = fields[0]
    if kind == "STOP":
        return "STOP"
    if kind == "PROD":
        return f"PROD({MODULES[fields[1]]['key']}, {DELTAS[fields[2]]})"
    if kind == "SERVICE":
        return f"SERVICE({MODULES[fields[1]]['key']}, {REGIMES[fields[2]]})"
    if kind == "LAND":
        return f"LAND({LAND_TIMINGS[fields[1]]})"
    return f"MARKET({PRODUCTS[fields[1]]}, {PRIMITIVES[fields[2]]}, {HORIZONS[fields[3]]}, rank {fields[4]})"


def _check(value: int, count: int) -> int:
    if type(value) is not int or not 0 <= value < count:
        raise ValueError(f"field {value!r} outside 0..{count - 1}")
    return value


TABLE = {"schema": SCHEMA, "modes": MODES, "modules": [m["key"] for m in MODULES], "deltas": DELTAS,
         "regimes": REGIMES, "land_timings": LAND_TIMINGS, "primitives": PRIMITIVES, "horizons": HORIZONS,
         "ranks": RANKS, "products": PRODUCTS, "bases": {"prod": PROD_BASE, "service": SERVICE_BASE,
         "land": LAND_BASE, "market": MARKET_BASE, "actions": N_ACTIONS, "keys": N_KEYS}}
ACTION_SCHEMA_SHA256 = hashlib.sha256(json.dumps(TABLE, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

_TAPES = None
TAPES_PATH = Path(__file__).resolve().parent / "modules.json"


def tapes() -> dict:
    """Per module and regime, the nine length-30 daily tapes (units, inputs, cash, actions)."""
    global _TAPES
    if _TAPES is None:
        data = json.loads(TAPES_PATH.read_text())
        if data.get("schema") != "farm-v0-module-tapes-v1":
            raise ValueError("V0 module tape asset required")
        _TAPES = data
    return _TAPES


_ONE_CYCLE = {}
TAPE_FIELDS = ("primary", "manure", "wheat", "fertilizer", "fixed_cash", "seeds", "service_actions", "manure_actions", "occupancy")


def tape(module: int, regime: int) -> dict:
    """The tape row of one module under one regime; raises for an illegal regime."""
    m = MODULES[module]
    name = REGIMES[regime]
    first_eligible = name.endswith("FIRST_ELIGIBLE")
    if first_eligible and m["producer"] not in ("WHEAT", "CARROT"):
        raise KeyError(f"no first-eligible tape for {m['key']}")
    if name.startswith("ONE_CYCLE") and m["animal"]:
        raise KeyError(f"an animal has no exit: {m['key']}")
    row_regime = "first_eligible" if first_eligible else m["default_regime"]
    row = tapes()["rows"][f"{m['producer']}:{m['policy']}:{row_regime}:b{m['birth']:02d}"]
    if not name.startswith("ONE_CYCLE"):
        return row
    key = (module, regime)
    if key not in _ONE_CYCLE:
        # The lifecycle ends at the tape's first replant after the birth: that day keeps its
        # harvest (units, one HARVEST action, the site) and drops the replant (seed, cash,
        # watering); every later day is zero.
        birth = m["birth"]
        replant = next((d for d in range(birth + 1, SEASON_DAYS) if row["seeds"][d]), SEASON_DAYS)
        cut = {f: [v if d < replant else 0 for d, v in enumerate(row[f])] for f in TAPE_FIELDS}
        if replant < SEASON_DAYS and row["primary"][replant]:
            cut["primary"][replant] = row["primary"][replant]
            cut["occupancy"][replant] = 1
            cut["service_actions"][replant] = 1
        cut["product"] = row["product"]
        _ONE_CYCLE[key] = cut
    return _ONE_CYCLE[key]


def regime_legal(module: int, regime: int) -> bool:
    try:
        tape(module, regime)
    except KeyError:
        return False
    return True
