"""Actor-only, variable-length economic entities; NEW80-column meanings.

No policy/model/engine execution is performed. Token presence, field knownness
and a downstream policy's action-validity mask are three different contracts.
Values are scaled, not clipped; absent fields have zero storage AND false
knownness. Opponent private fields and diagnostic seed/future data never enter.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from farm_neat.economy.heuristic.constants import ANIMALS, CROPS, MARKET_PARAMS, PRODUCTS, SHOPS
from farm_neat.economy import action_space as commands

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "farm-full-economic-entity-observation-v1"
ORDER_SCHEMA = "farm-full-economic-orders-v1"
WIDTH = 80
ITEMS = tuple(PRODUCTS) + tuple(ANIMALS)
KINDS = ("GLOBAL", "FARM", "TILE", "CARRIER", "MARKET", "STOCK", "SHOP",
         "COMMAND", "ARGUMENT", "CREDIT", "CONTROL", "CONSTANT")
TILES = ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE")
SHAPES = ("linear", "sq", "sqrt", "log", "log10", "hinge")
PAYLOAD = 35
CONFIG = {"episodeSteps": 720, "boardSize": 10, "turnsPerDay": 24,
          "shedCapacity": 100, "maxMarketOrdersPerTurn": 10,
          "farmHandCostMult": 1, "weedSpawnChance": .005,
          "townShopUnlockInterval": 3, "townShopSellInterval": 4,
          "townCenterSellInterval": 24}
TILE_FIELDS = (("planted_day", 30), ("watered_today", 1),
    ("consecutive_unwatered", 30), ("yield_units", 100),
    ("max_lifespan_step", 720), ("fertilized_until_day", 30),
    ("placed_day", 30), ("consecutive_unfed", 30), ("fed_today", 1),
    ("cared_today", 1), ("fertilizer_available", 1), ("pending_care_bonus", 100))
FIELDS = {
    "GLOBAL": (("step", 720), ("day", 30), ("hour", 24), ("seat", 1),
               ("remaining_actions", 720), *((k, float(v) or 1) for k, v in CONFIG.items())),
    "FARM": (("cash", 10000), ("hires_today", 100), ("hands", 100),
             *((q, 1) for q in ("NW", "NE", "SW", "SE"))),
    "TILE": TILE_FIELDS,
    "CARRIER": (*(("carried:" + p, 100) for p in ITEMS),
                *(("key_rank:" + p, len(ITEMS)) for p in ITEMS),
                ("farmer", 1), ("shed_access", 1)),
    "MARKET": (("price", 1000), ("inventory", 10000), ("base", 1000),
               ("I0", 10000), ("T", 1000), ("below_target", 1), ("above_target", 1),
               *((side + ":" + s, 1) for side in ("below", "above") for s in SHAPES)),
    "STOCK": (("quantity", 100), ("shed", 1), ("seed", 1)),
    "SHOP": (("count", 8), *(("consumes:" + p, 1) for p in PRODUCTS)),
    "CONSTANT": (("seed_cost", 1000), ("animal_cost", 1000), ("first_yield_day", 30),
                 ("max_yield_day", 30), ("interval", 30), ("max_yield", 100),
                 ("ongoing", 1), ("max_held", 100),
                 *(("produces:" + p, 1) for p in PRODUCTS)),
}
ENUM_FIELDS = {"watering": commands.WATERING, "fertilizing": commands.FERTILIZING,
    "feed": commands.FEED, "care": commands.CARE, "manure": commands.MANURE,
    "stock_kind": commands.STOCK_KINDS, "cancel_kind": commands.CANCEL_KINDS}
FIELDS["COMMAND"] = (*(("op:"+op, 1) for op in commands.OPS),
    *(("region:"+r, 1) for r in commands.REGIONS), ("scope_count", 100),
    ("cycles", 100), ("cycles_unbounded", 1), ("start", 720), ("until", 720),
    ("game_end", 1), ("priority", 255), ("harvest_threshold", 100),
    ("target", 100), ("quantity", 100), ("reserve", 100),
    ("price_above", 1000), ("price_below", 1000), ("amount", 10000),
    *((name, max(1, len(choices)-1)) for name, choices in ENUM_FIELDS.items()),
    ("replacement", 1), ("is_rule", 1))
FIELDS["CREDIT"] = (("completed_cycles", 100), ("successful_harvests", 100),
    ("harvested_units", 100), ("cycle_harvests", 100), ("fertilized_day", 30),
    ("clear_before", 1), ("acquired", 100), ("birth_present", 1),
    ("birth_planted_day", 30), ("birth_placed_day", 30), ("birth_day", 30),
    *(("birth:"+p, 1) for p in ITEMS), ("scope_is_tile", 1),
    *(("scope_region:"+r, 1) for r in commands.REGIONS))
REPORT_FIELDS = ("step", "due_water_feed", "pending_jobs", "assigned_jobs",
    "unassigned_jobs", "market_slots_used", "shed_used", "cash_spent",
    "sale_revenue", "destroyed_units", "weeds", "animals_died", "animal_losses",
    "missed_services", "available_units")
FIELDS["CONTROL"] = (("commitments_known", 1), ("revision", 100), ("pending_present", 1), ("pending_step", 720),
    ("order_count", 100), ("rule_count", 100), ("event_count", 100),
    ("blocked_count", 100), *(("report:"+n, 10000 if n in ("cash_spent", "sale_revenue")
                              else 720 if n == "step" else 100) for n in REPORT_FIELDS))
assert all(len(fields) <= WIDTH-PAYLOAD for fields in FIELDS.values())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def sources():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
            ("src/farm_neat/policy/observation.py", "src/farm_neat/economy/heuristic/constants.py",
             "src/farm_neat/economy/action_space.py", "src/farm_neat/game/constants.py")}


def _number(value, name, *, integer=False, boolean=False, minimum=None):
    expected = (bool,) if boolean else (int,) if integer else (int, float)
    if type(value) not in expected or not math.isfinite(value):
        raise ValueError("typed finite " + name + " required")
    if minimum is not None and value < minimum:
        raise ValueError(name + " below minimum")
    return value


def _position(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("two coordinates required")
    if any(type(n) is not int or not 0 <= n < 10 for n in value):
        raise ValueError("in-bounds10x10 coordinates required")
    return tuple(value)


def _stock(value, vocabulary, *, complete=False):
    if not isinstance(value, dict) or set(value) - set(vocabulary):
        raise ValueError("typed stock with known items required")
    if complete and set(value) != set(vocabulary):
        raise ValueError("complete stock required")
    return {k: _number(v, k, integer=True, minimum=0) for k, v in value.items()}


class _Rows:
    def __init__(self):
        self.rows, self.known, self.ids = [], [], []

    def add(self, kind, identity, *, role=None, position=None, ordinal=None, item=None,
            tile_kind=None, values=None):
        row = np.zeros(WIDTH, np.float32)
        known = np.zeros(WIDTH, np.bool_)
        row[KINDS.index(kind)] = 1
        known[:len(KINDS)] = True
        if role is not None:
            row[12 + role] = 1
            known[12:14] = True
        if position is not None:
            row[14:16] = np.asarray(_position(position), np.float32) / 9
            known[14:16] = True
        if ordinal is not None:
            row[16] = _number(ordinal, "ordinal", integer=True, minimum=0) / 100
            known[16] = True
        if item is not None:
            row[17 + ITEMS.index(item)] = 1
            known[17:29] = True
        if tile_kind is not None:
            row[29 + TILES.index(tile_kind)] = 1
            known[29:35] = True
        values = values or {}
        if set(values) - {n for n, _ in FIELDS[kind]}:
            raise ValueError("unrepresented " + kind + " field")
        for i, (name, scale) in enumerate(FIELDS[kind], PAYLOAD):
            value = values.get(name)
            if value is not None:
                _number(value, name, boolean=isinstance(value, bool))
                row[i] = float(value) / scale
                known[i] = True
        if not np.isfinite(row).all():
            raise ValueError("feature exceeds finite FP32 range")
        self.rows.append(row)
        self.known.append(known)
        self.ids.append(identity)


def _world(observation, configuration, result):
    if not isinstance(observation, dict) or not isinstance(configuration, dict):
        raise ValueError("actor observation and configuration required")
    seat = _number(observation.get("player"), "seat", integer=True, minimum=0)
    if seat not in (0, 1):
        raise ValueError("seat must be0 or1")
    cfg = {k: configuration.get(k, default) for k, default in CONFIG.items()}
    for key, value in cfg.items():
        _number(value, key, integer=key != "weedSpawnChance", minimum=0)
    if cfg["boardSize"] != 10 or cfg["turnsPerDay"] != 24 or cfg["episodeSteps"] != 720:
        raise ValueError("explicit10x10/24/720 schema; custom geometry requires new version")
    if min(cfg[k] for k in ("shedCapacity", "maxMarketOrdersPerTurn",
                            "townShopUnlockInterval", "townShopSellInterval",
                            "townCenterSellInterval")) <= 0:
        raise ValueError("positive capacity and intervals required")
    step = _number(observation.get("step"), "step", integer=True, minimum=0)
    if step > 719 or observation.get("day") != step // 24 or observation.get("hour") != step % 24:
        raise ValueError("actor clock disagrees with physical step")
    _number(observation["day"], "day", integer=True)
    _number(observation["hour"], "hour", integer=True)
    result.add("GLOBAL", "global", values={"step": step, "day": step // 24,
        "hour": step % 24, "seat": seat, "remaining_actions": 719-step, **cfg})
    farms, private = observation.get("farms"), observation.get("private")
    if not isinstance(farms, list) or len(farms) != 2 or not isinstance(private, dict):
        raise ValueError("two public farms and own private state required")
    for role, farm in enumerate((farms[seat], farms[1-seat])):
        board = farm.get("tiles")
        if not isinstance(board, list) or len(board) != 10 or any(not isinstance(r, list) or len(r) != 10 for r in board):
            raise ValueError("complete public10x10 tile grid required")
        hands = farm.get("hands")
        if not isinstance(hands, list):
            raise ValueError("all public hand positions required")
        positions = [_position(farm.get("farmer")), *(_position(p) for p in hands)]
        land = farm.get("unlocked_quadrants")
        if not isinstance(land, list) or len(set(land)) != len(land) or set(land) - {"NW", "NE", "SW", "SE"}:
            raise ValueError("explicit unique unlocked quadrants required")
        result.add("FARM", f"farm/{role}", role=role, values={
            "cash": _number(farm.get("money"), "cash", minimum=0),
            "hires_today": _number(farm.get("hires_today"), "hires", integer=True, minimum=0),
            "hands": len(hands), **{q: q in land for q in ("NW", "NE", "SW", "SE")}})
        for y, line in enumerate(board):
            for x, tile in enumerate(line):
                if tile is None or tile == "LOCKED":
                    kind, fields, item = "EMPTY" if tile is None else "LOCKED", {}, None
                elif isinstance(tile, dict):
                    kind = tile.get("kind")
                    if kind not in TILES[2:]:
                        raise ValueError("unknown public tile kind")
                    item = tile.get("crop") if kind == "PLANT" else tile.get("animal")
                    if kind == "PLANT" and (item not in CROPS or tile.get("animal") is not None):
                        raise ValueError("known crop identity required")
                    if kind != "PLANT" and tile.get("crop") is not None:
                        raise ValueError("crop identity on nonplant")
                    if item is not None and kind != "PLANT" and (item not in ANIMALS or ANIMALS[item]["structure"] != kind):
                        raise ValueError("animal structure mismatch")
                    fields = {k: tile.get(k) for k, _ in TILE_FIELDS}
                    for key, value in fields.items():
                        if value is not None:
                            _number(value, key, boolean=key.endswith("_today") or key == "fertilizer_available", integer=not(key.endswith("_today") or key == "fertilizer_available"))
                else:
                    raise ValueError("invalid public tile")
                result.add("TILE", f"tile/{role}/{x},{y}", role=role,
                           position=(x, y), item=item, tile_kind=kind, values=fields)
        inventories = private.get("inventories") if role == 0 else None
        if role == 0 and (not isinstance(inventories, list) or len(inventories) != len(positions)):
            raise ValueError("every own carrier must have aligned private inventory")
        for index, position in enumerate(positions):
            values = {"farmer": index == 0, "shed_access": position in ((4,4),(4,5),(5,4),(5,5))}
            if role == 0:
                inv = _stock(inventories[index], ITEMS)
                ranks = {k: n for n, k in enumerate(inv, 1)}
                values.update({"carried:" + k: inv.get(k, 0) for k in ITEMS})
                values.update({"key_rank:" + k: ranks.get(k, 0) for k in ITEMS})
            result.add("CARRIER", f"carrier/{role}/{index}", role=role,
                       position=position, ordinal=index, values=values)
    for storage, vocabulary in (("shed", ITEMS), ("seeds", tuple(CROPS))):
        stock = _stock(private.get(storage), vocabulary)
        for item in vocabulary:
            result.add("STOCK", f"stock/{storage}/{item}", role=0, item=item,
                       values={"quantity": stock.get(item, 0), "shed": storage == "shed", "seed": storage == "seeds"})
    market = observation.get("market")
    if not isinstance(market, dict):
        raise ValueError("public market required")
    prices, inventory = market.get("prices"), market.get("inventory")
    if not isinstance(prices, dict) or set(prices) != set(PRODUCTS) or not isinstance(inventory, dict) or set(inventory) != set(PRODUCTS):
        raise ValueError("all nine public market products required")
    params = market.get("params")
    if params is None:
        overrides = configuration.get("marketParams", {})
        if not isinstance(overrides, dict) or set(overrides) - set(PRODUCTS):
            raise ValueError("known market overrides required")
        params = {item: {**MARKET_PARAMS[item], **overrides.get(item, {})} for item in PRODUCTS}
    if not isinstance(params, dict) or set(params) != set(PRODUCTS):
        raise ValueError("complete market curve identities required")
    for item in PRODUCTS:
        p = params[item]
        if not isinstance(p, dict) or set(p) != set(MARKET_PARAMS[item]) or p["below_func"] not in SHAPES or p["above_func"] not in SHAPES:
            raise ValueError("known complete market curve required")
        values = {k: _number(p[k], "curve:"+k) for k in ("base", "I0", "T", "below_target", "above_target")}
        values.update(price=_number(prices[item], "price", minimum=0),
                      inventory=_number(inventory[item], "inventory", integer=True))
        values.update({side + ":" + shape: p[side + "_func"] == shape for side in ("below", "above") for shape in SHAPES})
        result.add("MARKET", "market/"+item, item=item, values=values)
    shops = observation.get("town", {}).get("unlocked_shops")
    if shops is not None and (not isinstance(shops, list) or any(s not in SHOPS for s in shops)):
        raise ValueError("known public shop identities required")
    counts = Counter(shops) if shops is not None else None
    for index, (shop, products) in enumerate(SHOPS.items()):
        result.add("SHOP", "shop/"+shop, ordinal=index, values={
            "count": None if counts is None else counts[shop],
            **{"consumes:"+item: products.count(item)*(2 if len(products)==1 else 1) for item in PRODUCTS}})
    for item, data in (*CROPS.items(), *ANIMALS.items()):
        values = {k: data.get(k) for k, _ in FIELDS["CONSTANT"] if not k.startswith("produces:")}
        values.update(seed_cost=data.get("seed"), animal_cost=data.get("cost"))
        product = item if item in CROPS else data["product"]
        values.update({"produces:"+p: p == product for p in PRODUCTS})
        result.add("CONSTANT", "mechanics/"+item, item=item,
                   tile_kind="PLANT" if item in CROPS else data["structure"], values=values)
    return step


def _command(command, identity, ordinal, result, *, is_rule=False, replacement=False):
    c = commands.validate_command(command)
    values = {"op:"+op: c["op"] == op for op in commands.OPS}
    target = c["target"] if c["op"] == "CANCEL" else {}
    scope = c.get("scope", target.get("scope"))
    position = None
    if scope is not None:
        position = (scope["x"], scope["y"]) if scope["kind"] == "TILE" else None
        values.update({"region:"+r: scope.get("region") == r for r in commands.REGIONS})
        values["scope_count"] = 1 if position is not None else scope["count"]
    item = c.get("crop", c.get("animal", c.get("product", c.get("item"))))
    item = target.get("product", target.get("item", item))
    for name in ("start", "priority", "harvest_threshold", "quantity", "reserve", "price_above", "price_below", "amount"):
        if name in c:
            values[name] = c[name]
    if "target" in c and c["op"] != "CANCEL":
        values["target"] = c["target"]
    if "cycles" in c:
        values.update(cycles=c["cycles"], cycles_unbounded=c["cycles"] is None)
    if "until" in c:
        values.update(until=None if c["until"] == "GAME_END" else c["until"],
                      game_end=c["until"] == "GAME_END")
    enum_values = {k: c[k] for k in ("watering", "fertilizing", "feed", "care", "manure") if k in c}
    if "kind" in c:
        enum_values["stock_kind"] = c["kind"]
    if target:
        enum_values["cancel_kind"] = target["kind"]
        if "stock_kind" in target:
            enum_values["stock_kind"] = target["stock_kind"]
    values.update({name: ENUM_FIELDS[name].index(value) for name, value in enum_values.items()})
    values.update(replacement=replacement, is_rule=is_rule)
    result.add("COMMAND", identity, role=0, ordinal=ordinal,
               position=position, item=item, values=values)
    if c["op"] == "REPLACE":
        _command(c["replacement"], identity+"/replacement", ordinal, result,
                 is_rule=is_rule, replacement=True)


def _state(snapshot, step, result):
    if snapshot is None:
        # Public teachers do not expose their latent economic plan. Absence is
        # not a declaration of an empty plan, zero reserves or zero past credit.
        result.add("CONTROL", "control", role=0, values={"commitments_known": False})
        return
    if hasattr(snapshot, "snapshot"):
        snapshot = snapshot.snapshot()
    if not isinstance(snapshot, dict) or snapshot.get("schema") != ORDER_SCHEMA:
        raise ValueError("new full-economic order snapshot required; not legacy29 orders")
    expected = {"schema", "orders", "rules", "pending", "events", "last_report", "revision"}
    if set(snapshot) != expected:
        raise ValueError("closed full-economic snapshot fields required")
    orders, rules = snapshot["orders"], snapshot["rules"]
    if not isinstance(orders, dict) or not isinstance(rules, dict):
        raise ValueError("ordered commitment and rule maps required")
    ordinal = 0
    for group, mapping in (("orders", orders), ("rules", rules)):
        for key, row in mapping.items():
            if not isinstance(key, str) or not isinstance(row, dict) or "command" not in row:
                raise ValueError("typed command row required")
            allowed = {"command", "acquired"} if group == "rules" else {
                "command", "id", "completed_cycles", "successful_harvests", "harvested_units",
                "birth", "cycle_harvests", "fertilized_day", "clear_before", "scope_key"}
            if set(row) != allowed:
                raise ValueError("unrepresented or missing commitment/credit field")
            if group == "orders":
                if not isinstance(row["id"], str) or not row["id"]:
                    raise ValueError("stable commitment identity required")
                scope = row["command"].get("scope", {})
                if scope.get("kind") != "TILE" or key != f'{scope.get("x")},{scope.get("y")}':
                    raise ValueError("allocated order key must match TILE command")
                if row["command"].get("op") not in ("SET_CROP", "SET_ANIMAL", "CLEAR"):
                    raise ValueError("production command required in allocated order")
            else:
                c = commands.validate_command(row["command"])
                op = c["op"]
                if op not in ("SALE", "STOCK", "BUY", "LAND", "HANDS", "CASH_RESERVE"):
                    raise ValueError("persistent rule command required")
                expected_key = ("SALE:"+c["product"] if op == "SALE" else
                                f'{op}:{c["kind"]}:{c["item"]}' if op in ("STOCK", "BUY") else op)
                if key != expected_key:
                    raise ValueError("rule identity differs from command")
            identity = group+"/"+key
            _command(row["command"], identity+"/command", ordinal, result, is_rule=group == "rules")
            values = {k: row[k] for k, _ in FIELDS["CREDIT"] if k in row}
            for name, value in values.items():
                _number(value, name, boolean=name == "clear_before", integer=name != "clear_before",
                        minimum=-1 if name == "fertilized_day" else 0)
            if group == "orders":
                scope_key = row["scope_key"]
                if scope_key == "TILE:"+key:
                    region = None
                elif isinstance(scope_key, str) and scope_key.startswith("REGION:") and scope_key[7:] in commands.REGIONS:
                    region = scope_key[7:]
                else:
                    raise ValueError("explicit original scope group required")
                values.update(scope_is_tile=region is None,
                              **{"scope_region:"+r: region == r for r in commands.REGIONS})
                birth = row["birth"]
                values["birth_present"] = birth is not None
                if birth is not None:
                    if not isinstance(birth, dict) or set(birth) != {"crop", "planted_day"}:
                        raise ValueError("unrepresented birth identity")
                    item = birth["crop"]
                    if item not in CROPS or item != row["command"].get("crop"):
                        raise ValueError("birth item required")
                    values.update({"birth:"+p: p == item for p in ITEMS})
                    for name in ("day", "planted_day", "placed_day"):
                        if name in birth:
                            values["birth_"+name] = _number(birth[name], name, integer=True, minimum=0)
            result.add("CREDIT", identity+"/credit", role=0, ordinal=ordinal, values=values)
            ordinal += 1
    pending = snapshot["pending"]
    if pending is not None and (not isinstance(pending, dict) or type(pending.get("step")) is not int or not 0 <= pending["step"] <= step):
        raise ValueError("pending summary clock invalid")
    events, report = snapshot["events"], snapshot["last_report"]
    if not isinstance(events, list) or not isinstance(report, dict):
        raise ValueError("event list and compact report required")
    blocked = report.get("blocked")
    if blocked is not None and not isinstance(blocked, list):
        raise ValueError("blocked report must be a list")
    summary = {"commitments_known": True,
        "revision": _number(snapshot["revision"], "revision", integer=True, minimum=0),
        "pending_present": pending is not None, "pending_step": None if pending is None else pending["step"],
        "order_count": len(orders), "rule_count": len(rules), "event_count": len(events),
        "blocked_count": None if blocked is None else len(blocked)}
    for name in REPORT_FIELDS:
        value = report.get(name)
        if value is not None:
            summary["report:"+name] = _number(value, "report:"+name)
    result.add("CONTROL", "control", role=0, values=summary)


REPRESENTATION_SCHEMA = {"schema": SCHEMA, "width": WIDTH, "order_schema": ORDER_SCHEMA,
    "action_schema_sha256": commands.ACTION_SCHEMA_SHA256,
    "token_kinds": KINDS, "items": ITEMS, "tile_kinds": TILES,
    "base_columns": {"kind": [0,12], "self_opponent": [12,14], "xy_div9": [14,16],
                     "ordinal_div100": 16, "item": [17,29], "tile_kind": [29,35]},
    "fields_by_kind": FIELDS, "enums": ENUM_FIELDS,
    "payload_starts": PAYLOAD, "normalization": "DIVIDE_THEN_FP32_WITHOUT_CLIPPING",
    "knownness": "BOOL_PRESENT_OBSERVED_OR_DECLARED; FALSE_IS_NOT_KNOWN_ZERO",
    "token_mask": "ENTITY_PRESENT; NEVER_ACTION_LEGALITY", "unknown_storage": 0,
    "order": "GLOBAL; SELF_THEN_OPPONENT_FARM_TILES_CARRIERS; OWN_STOCK; MARKET; SHOPS; MECHANICS; ORDERED_COMMAND_CREDIT; CONTROL",
    "carrier_limit": None, "private_scope": "OWN_ONLY; OPPONENT_INVENTORY_UNKNOWN",
    "unknown_commitments": "orders=None emits CONTROL with commitments_known=False; counts, rules, reserves and credit stay unknown",
    "excluded": ["diagnostic_seed", "future_world", "opponent_private", "raw_pending_world", "raw_event_history"],
    "no_legacy_29x80_meaning_or_weight_parity": True}
REPRESENTATION_SCHEMA_SHA256 = digest(REPRESENTATION_SCHEMA)
_LOADED_SOURCES = sources()


def pack(observation, orders, configuration=None):
    """Return detached dynamicN entities; no caller input is mutated.

    ``orders`` is EconomicOrders, its snapshot, or None for an unknown teacher
    plan (not an invented empty plan). Externally pinned run
    guards own source-liveness checks; source metadata describes loaded code.
    """
    result = _Rows()
    step = _world(observation, {} if configuration is None else configuration, result)
    _state(orders, step, result)
    rows, known = np.stack(result.rows), np.stack(result.known)
    ids = tuple(result.ids)
    packed = {"schema": SCHEMA, "representation_schema_sha256": REPRESENTATION_SCHEMA_SHA256,
        "source_pins": dict(_LOADED_SOURCES), "source_sha256": digest(_LOADED_SOURCES),
        "rows": rows, "knownness": known, "token_mask": np.ones(len(ids), np.bool_),
        "token_ids": ids}
    packed["content_sha256"] = _content(packed)
    validate(packed)
    return packed


def _content(packed):
    h = hashlib.sha256()
    h.update(digest({k: packed[k] for k in ("schema", "representation_schema_sha256", "source_pins", "token_ids")}).encode())
    for k in ("rows", "knownness", "token_mask"):
        array = packed[k]
        h.update(str((k, array.shape, array.dtype.str)).encode())
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


def validate(packed):
    """Reject forged knownness/padding/source/layout; no action mask accepted."""
    if (not isinstance(packed, dict) or set(packed) != {"schema", "representation_schema_sha256",
            "source_pins", "source_sha256", "rows", "knownness", "token_mask", "token_ids", "content_sha256"}
            or packed["schema"] != SCHEMA or packed["representation_schema_sha256"] != REPRESENTATION_SCHEMA_SHA256
            or packed["source_pins"] != _LOADED_SOURCES or packed["source_sha256"] != digest(_LOADED_SOURCES)):
        raise ValueError("exact new representation/source schema required")
    rows, known, mask, ids = (packed[k] for k in ("rows", "knownness", "token_mask", "token_ids"))
    if (not all(isinstance(a, np.ndarray) for a in (rows, known, mask))
            or rows.dtype != np.float32 or rows.ndim != 2 or rows.shape[1] != WIDTH
            or known.shape != rows.shape or known.dtype != np.bool_
            or mask.shape != (len(rows),) or mask.dtype != np.bool_ or not len(rows)
            or not mask.all() or not isinstance(ids, tuple) or len(ids) != len(rows)
            or ids[0] != "global" or len(set(ids)) != len(ids)
            or not np.isfinite(rows).all() or np.any(rows[~known] != 0)
            or np.any(known[~mask]) or np.any(rows[~mask])
            or packed["content_sha256"] != _content(packed)):
        raise ValueError("exact finite entities/knownness/presence/content required")
    return packed


def collate(packs):
    """Pad only to this batch's longest roster; never truncate carriers/entities."""
    if not isinstance(packs, (list, tuple)) or not packs:
        raise ValueError("nonempty packed batch required")
    for packed in packs:
        validate(packed)
    length = max(len(p["rows"]) for p in packs)
    rows = np.zeros((len(packs), length, WIDTH), np.float32)
    known = np.zeros(rows.shape, np.bool_)
    mask = np.zeros(rows.shape[:2], np.bool_)
    for index, packed in enumerate(packs):
        n = len(packed["rows"])
        rows[index, :n], known[index, :n], mask[index, :n] = packed["rows"], packed["knownness"], packed["token_mask"]
    return {"schema": SCHEMA, "rows": rows, "knownness": known, "token_mask": mask,
            "token_ids": tuple(p["token_ids"] for p in packs),
            "content_sha256s": tuple(p["content_sha256"] for p in packs)}
