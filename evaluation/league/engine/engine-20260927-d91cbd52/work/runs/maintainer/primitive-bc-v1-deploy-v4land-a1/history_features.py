"""Per-seat causal history features for the primitive-action BC policy.

Reads only the actor-visible observation fields kept by
``farm_neat.training.bc.actor_observation`` (clock, both public farms, own private
shed/seeds/carried, public market, public town shops) plus this seat's OWN past wire
actions.  No rival private state, episode seed, future shop draw, future action or
outcome is read.  Everything is causal: the features for step t depend only on
observations 0..t and own actions 0..t-1 of the current episode.

Ideas adapted (not ported) from the old farm repo; see FEATURE-IDEAS.md for the table:
  /home/russ/Desktop/farm/train/spatiotemporal.py      sha256 9f61b2c6a2c32f52846ea3049f5dad3bb9d4c625af8f5ac658ed81c4ae9e04e6
  /home/russ/Desktop/farm/neural/current_neural_features.py sha256 5ba8ef1ee8a84d87f34628425c7d2122271e390965a1f6ada4c7b8da9b923c3f
  /home/russ/Desktop/farm/neural/current_neural_history_knownness.py sha256 fa7f7a53dd9070a089e79d0282887078cebbb0164e6a9974e5d5766ac9ad8082
  /home/russ/Desktop/farm/plan_player_bridge/opponent_projection.py sha256 aedbb64fae73a91f27dd1efbf998dc13d5b3fedc82077fa52f77e3b333faddd7
  /home/russ/Desktop/farm/agent/planning_trackers.py   sha256 b846c6e3500cb292bea67c23ecafed02067ff610b68ac285ab067f63e7e33eb1

Usage, once per engine step for one seat::

    hist = HistoryFeatures()            # or SeatHistories() for one callable, two seats
    out = hist.update(actor_obs, configuration)   # features for this step
    action = policy(pack(...), out)
    hist.record_action(action)          # the exact wire action submitted this step

Output ``HistoryOutput.blocks``: name -> (row_labels, field_names, values f32[R,F],
known bool[R,F]).  Values are divided by a fixed scale, never clipped; unknown values
are stored as 0 with known=False (observation.py convention).  Every block has at most
45 fields so one row fits one width-80 entity token (PAYLOAD=35).

Exactness: town consumption and ``trade_flow`` (net market supply change caused by
both players' trades) are exact given the visible shop list, except that a sale at the
$1 price floor adds no supply.  ``*_obtained``/``*_used``/``*_trade_est``/``stock_est``
are public-tile inferences with known blind spots (see FEATURE-IDEAS.md) and are
labelled ``est``.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque

import numpy as np

from farm_neat.economy.heuristic.constants import (
    ANIMALS, CROPS, LAND_PRICES, MARKET_PARAMS, PRODUCTS, SHOPS, TOWN_CENTER_PRODUCTS, fib,
)

SCHEMA = "primitive-bc-history-features-v1"
ITEMS = tuple(PRODUCTS) + tuple(ANIMALS)        # == farm_neat.policy.observation.ITEMS
N_ITEMS = len(ITEMS)
N_PRODUCTS = len(PRODUCTS)
ITEM_INDEX = {item: i for i, item in enumerate(ITEMS)}
PRODUCT_ROWS = np.arange(N_ITEMS) < N_PRODUCTS
LAGS = (1, 4, 24, 72)
MAX_LAG = max(LAGS)
UNIT_OPS = ("PASS", "NORTH", "SOUTH", "EAST", "WEST", "PLANT", "WATER", "HARVEST",
            "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "PICKUP", "PLACE", "DROP", "FEED",
            "CARE", "COLLECT_FERTILIZER", "DIG")
MARKET_OPS = ("SELL", "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL", "HIRE", "BUY_LAND")
DEFAULT_CONFIG = {"turnsPerDay": 24, "episodeSteps": 720, "shedCapacity": 100,
                  "maxMarketOrdersPerTurn": 10, "farmHandCostMult": 1,
                  "townShopSellInterval": 4, "townCenterSellInterval": 24,
                  "townShopUnlockInterval": 3}
PRODUCT_OF = {**{c: c for c in CROPS}, **{a: d["product"] for a, d in ANIMALS.items()}}
CAP_OF = {**{c: d["max_yield"] for c, d in CROPS.items()},
          **{a: d["max_held"] for a, d in ANIMALS.items()}}
ONGOING = {c for c, d in CROPS.items() if d["ongoing"]}
WHEAT, FERT = ITEM_INDEX["WHEAT"], ITEM_INDEX["FERTILIZER"]
BUYABLE = frozenset(("WHEAT", "FERTILIZER"))
PRICE_FLOOR = 1
_TILE_SCALE = 25.0

# ----------------------------------------------------------------------------- schema
# (name, scale); value stored = raw / scale.
GLOBAL_NOW = (
    ("phase.hour_sin", 1), ("phase.hour_cos", 1),
    ("phase.tick4_0", 1), ("phase.tick4_1", 1), ("phase.tick4_2", 1), ("phase.tick4_3", 1),
    ("phase.day_end_follows", 1), ("phase.final_action", 1),
    ("phase.day_mod3_0", 1), ("phase.day_mod3_1", 1), ("phase.day_mod3_2", 1),
    ("hist.from_start", 1), ("hist.depth", MAX_LAG), ("hist.reset_gap", 1),
    ("flag.cash_lead", 1), ("flag.cash_tie", 1), ("flag.shed_empty", 1),
    ("flag.shed_full", 1), ("flag.has_seed", 1), ("flag.has_carried", 1),
    ("own.at_cap_tiles", _TILE_SCALE), ("rival.at_cap_tiles", _TILE_SCALE),
    ("own.discard_risk_steps", 24),
    ("own.loaded_return_steps", 24), ("own.max_return_steps", 10), ("own.stranded_carried", 100),
    ("own.feed_cover_days", 10),
)
GLOBAL_HIST = (
    ("lag.value", MAX_LAG), ("lag.exact", 1), ("lag.age", 1), ("lag.day_ends", 3),
    ("d.own_cash", 1000), ("d.rival_cash", 1000), ("d.cash_lead", 1000),
    ("own.trade_cash", 1000), ("rival.trade_cash_est", 1000),
    ("d.own_units", 10), ("d.rival_units", 10), ("d.own_quadrants", 1), ("d.rival_quadrants", 1),
    ("d.own_plants", _TILE_SCALE), ("d.own_animals", _TILE_SCALE), ("d.own_weeds", _TILE_SCALE),
    ("d.rival_plants", _TILE_SCALE), ("d.rival_animals", _TILE_SCALE), ("d.rival_weeds", _TILE_SCALE),
    ("d.own_shed_total", 100), ("d.own_carried_total", 100), ("d.own_seed_total", 100),
)
PREV_ACTION = (
    ("prev.known", 1), ("prev.units_commanded", 10),
    *((f"prev.op.{op}", 10) for op in UNIT_OPS),
    ("prev.market_orders", 10), ("prev.hire_orders", 10), ("prev.land_orders", 1),
    ("prev.malformed", 10),
)
ITEM_NOW = (
    ("mkt.price_vs_base", 1), ("mkt.inv_vs_target", 1), ("mkt.scarce", 1),
    ("mkt.price_up1", 1), ("mkt.price_down1", 1),
    ("cum.town_consumed", 1000), ("cum.trade_flow", 1000), ("cum.floor_steps", 24),
    ("rival.obtained_cum_est", 100), ("rival.trade_cum_est", 100), ("rival.stock_est", 100),
    ("prev.sell", 100), ("prev.buy_product", 100), ("prev.buy_seed", 100),
    ("prev.buy_animal", 10), ("prev.pickup", 100), ("prev.place", 100), ("prev.plant", 10),
)
ITEM_HIST_PER_LAG = (
    ("d.mkt_inv", 100), ("d.price_rel", 1), ("trade_flow", 100), ("town_consumed", 100),
    ("d.own_holdings", 100), ("own.trade_est", 100), ("rival.trade_est", 100),
    ("rival.obtained_est", 10), ("d.rival_yield", 10),
)
ITEM_HIST = tuple((f"lag{lag}.{name}", scale) for lag in LAGS for name, scale in ITEM_HIST_PER_LAG)
BLOCKS = {
    "history_global_now": (("now",), GLOBAL_NOW),
    "history_global_lag": (tuple(f"lag{lag}" for lag in LAGS), GLOBAL_HIST),
    "history_prev_action": (("prev",), PREV_ACTION),
    "history_item_now": (ITEMS, ITEM_NOW),
    "history_item_lag": (ITEMS, ITEM_HIST),
}
assert all(len(fields) <= 45 for _, fields in BLOCKS.values())
SCHEMA_PAYLOAD = {"schema": SCHEMA, "items": ITEMS, "lags": LAGS, "unit_ops": UNIT_OPS,
                  "blocks": {k: {"rows": list(r), "fields": [list(f) for f in fs]}
                             for k, (r, fs) in BLOCKS.items()},
                  "normalization": "DIVIDE_THEN_FP32_WITHOUT_CLIPPING",
                  "knownness": "FALSE_MEANS_UNKNOWN_AND_STORED_ZERO"}
SCHEMA_SHA256 = hashlib.sha256(json.dumps(SCHEMA_PAYLOAD, sort_keys=True).encode()).hexdigest()
_SCALES = {name: np.asarray([s for _, s in fields], np.float64) for name, (_, fields) in BLOCKS.items()}
_FIELD_INDEX = {name: {f: i for i, (f, _) in enumerate(fields)} for name, (_, fields) in BLOCKS.items()}

# raw snapshot layouts
G_RAW = ("own_cash", "rival_cash", "own_units", "rival_units", "own_quads", "rival_quads",
         "own_hires", "rival_hires", "own_plants", "own_animals", "own_weeds",
         "rival_plants", "rival_animals", "rival_weeds",
         "own_shed_total", "own_carried_total", "own_seed_total")
GR = {n: i for i, n in enumerate(G_RAW)}
I_RAW = ("mkt_inv", "mkt_price", "own_shed", "own_carried", "own_seeds", "own_yield", "rival_yield")
IR = {n: i for i, n in enumerate(I_RAW)}
# cumulative per-item transition sums since history start
I_CUM = ("town_consumed", "trade_flow", "own_obtained", "own_used", "own_trade",
         "rival_obtained", "rival_used", "rival_trade", "rival_stock", "floor_steps")
IC = {n: i for i, n in enumerate(I_CUM)}
G_CUM = ("own_trade_cash", "rival_trade_cash", "shops_unknown_steps", "discard_risk_steps")
GC = {n: i for i, n in enumerate(G_CUM)}



def _market_params(observation, configuration):
    """Visible curve identities: observation params, else configuration overrides, else defaults."""
    params = (observation.get("market") or {}).get("params")
    if params is not None:
        return params
    overrides = (configuration.get("marketParams") if hasattr(configuration, "get") else None) or {}
    if not overrides:
        return MARKET_PARAMS
    return {p: {**MARKET_PARAMS[p], **overrides.get(p, {})} for p in PRODUCTS}


# history_global_lag field <- index into concat(delta G_RAW, delta G_CUM, [cash lead delta])
_LAG_SOURCE = {"d.own_cash": GR["own_cash"], "d.rival_cash": GR["rival_cash"],
               "d.cash_lead": len(G_RAW) + len(G_CUM),
               "own.trade_cash": len(G_RAW) + GC["own_trade_cash"],
               "rival.trade_cash_est": len(G_RAW) + GC["rival_trade_cash"],
               "d.own_units": GR["own_units"], "d.rival_units": GR["rival_units"],
               "d.own_quadrants": GR["own_quads"], "d.rival_quadrants": GR["rival_quads"],
               "d.own_plants": GR["own_plants"], "d.own_animals": GR["own_animals"],
               "d.own_weeds": GR["own_weeds"], "d.rival_plants": GR["rival_plants"],
               "d.rival_animals": GR["rival_animals"], "d.rival_weeds": GR["rival_weeds"],
               "d.own_shed_total": GR["own_shed_total"],
               "d.own_carried_total": GR["own_carried_total"],
               "d.own_seed_total": GR["own_seed_total"]}
_LAG_DST = np.asarray([_FIELD_INDEX["history_global_lag"][k] for k in _LAG_SOURCE])
_LAG_SRC = np.asarray(list(_LAG_SOURCE.values()))
assert set(_LAG_SOURCE) | {"lag.value", "lag.exact", "lag.age", "lag.day_ends"} == {n for n, _ in GLOBAL_HIST}


class HistoryOutput:
    """``blocks``: see module doc.  ``prev_tokens``: codec.encode_step labels of the own
    action submitted at step-1 (unit_tok/unit_qty/order_tok/order_qty) when codec.py is
    importable and that action was recorded, else None."""
    __slots__ = ("step", "blocks", "prev_tokens")

    def __init__(self, step, blocks, prev_tokens=None):
        self.step, self.blocks, self.prev_tokens = step, blocks, prev_tokens

    def flat(self):
        """Concatenate every block row-major in BLOCKS order: (names, values, known)."""
        names, values, known = [], [], []
        for block, (rows, fields) in BLOCKS.items():
            _, _, v, k = self.blocks[block]
            names.extend(f"{block}/{r}/{f}" for r in rows for f, _ in fields)
            values.append(v.reshape(-1))
            known.append(k.reshape(-1))
        return tuple(names), np.concatenate(values), np.concatenate(known)


def flat_names():
    return tuple(f"{block}/{r}/{f}" for block, (rows, fields) in BLOCKS.items()
                 for r in rows for f, _ in fields)


# ----------------------------------------------------------------------------- helpers
def _cfg(configuration):
    cfg = dict(DEFAULT_CONFIG)
    if configuration:
        for key in DEFAULT_CONFIG:
            value = configuration.get(key) if hasattr(configuration, "get") else None
            if value is not None:
                cfg[key] = value
    return cfg


def _tile_tuple(tile):
    """Compact immutable tile record of the public fields the transition inference reads.
    PLANT: (kind, crop, planted_day, yield, max_lifespan_step, fertilized_until_day)
    animal: (kind, animal, placed_day, yield, fed_today, consecutive_unfed, fertilizer_available)
    other: (kind,)"""
    if tile is None:
        return None
    if not isinstance(tile, dict):
        return "LOCKED"
    get = tile.get
    kind = get("kind")
    if kind == "PLANT":
        return ("PLANT", get("crop"), get("planted_day"), get("yield_units") or 0,
                get("max_lifespan_step", -1), get("fertilized_until_day", -1))
    animal = get("animal")
    if animal is not None:
        return (kind, animal, get("placed_day"), get("yield_units") or 0, bool(get("fed_today")),
                get("consecutive_unfed") or 0, bool(get("fertilizer_available")))
    return (kind,)


def _scan_farm(tiles, g, it, role, at_cap_out):
    """Tile tuples plus per-farm aggregates; fills the raw arrays in place."""
    out = []
    plants = animals = weeds = at_cap = 0
    yield_col = IR["own_yield" if role == 0 else "rival_yield"]
    append = out.append
    for line in tiles:
        for tile in line:
            if tile is None:
                append(None)
                continue
            if tile.__class__ is str:
                append("LOCKED")
                continue
            t = _tile_tuple(tile)
            append(t)
            kind = t[0]
            if kind == "PLANT":
                plants += 1
                row = ITEM_INDEX.get(t[1])
                if row is not None:
                    it[row, yield_col] += t[3]
                    at_cap += t[3] >= CAP_OF[t[1]]
            elif kind == "WEED":
                weeds += 1
            elif len(t) > 1:
                animals += 1
                product = PRODUCT_OF.get(t[1])
                if product is not None:
                    it[ITEM_INDEX[product], yield_col] += t[3]
                    at_cap += t[3] >= CAP_OF[t[1]]
    prefix = "own_" if role == 0 else "rival_"
    g[GR[prefix + "plants"]] = plants
    g[GR[prefix + "animals"]] = animals
    g[GR[prefix + "weeds"]] = weeds
    at_cap_out[role] = at_cap
    return out


def _removals(prev_tiles, now_tiles, step, day_end, obtained, used):
    """Public-tile inference of product units taken (harvest/collect) and used (feed/fertilize)
    during the transition step -> step+1 on one farm.  Adds into obtained/used in place."""
    for a, b in zip(prev_tiles, now_tiles):
        if a == b or a is None or a == "LOCKED" or len(a) == 1:
            continue
        kind, item, birth, y = a[0], a[1], a[2], a[3]
        same = isinstance(b, tuple) and len(b) > 1 and b[0] == kind and b[1] == item and b[2] == birth
        if kind == "PLANT":
            row = ITEM_INDEX.get(item)
            if row is None:
                continue
            mls = a[4] if a[4] is not None else -1
            decay = 1 if (mls >= 0 and step >= mls and (step - mls) % 2 == 0) else 0
            if same:
                if y > 0 and b[3] < y - decay:
                    obtained[row] += y            # HARVEST takes all stored yield
                if (b[5] if b[5] is not None else -1) > (a[5] if a[5] is not None else -1):
                    used[FERT] += 1               # FERTILIZE consumed one carried fertilizer
            elif y > 0:
                weed = isinstance(b, tuple) and b[0] == "WEED"
                if not weed:
                    if item not in ONGOING:       # one-shot harvest removes the plant (or DIG)
                        obtained[row] += y
                elif item in ONGOING and decay and y >= 2 and not day_end:
                    obtained[row] += y            # harvest to 0, then decay step weeds it
        else:
            product = PRODUCT_OF.get(item)
            if product is None or not same:
                continue
            if y > 0 and b[3] < y:
                obtained[ITEM_INDEX[product]] += y
            if not a[4]:
                fed = (b[5] == 0) if day_end else b[4]
                if fed:
                    used[WHEAT] += 1
            if a[6] and not day_end and not b[6]:
                obtained[FERT] += 1               # COLLECT_FERTILIZER


def _shape(func, x, T):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def market_price(params, inventory):
    """Engine quote for one product's visible curve (kaggriculture_engine_reference.market_price)."""
    base, i0, T = params["base"], params["I0"], params["T"]
    if inventory < i0:
        f = params["below_func"]
        price = base + params["below_target"] * base / _shape(f, T, T) * _shape(f, i0 - inventory, T)
    else:
        f = params["above_func"]
        price = base - params["above_target"] * base / _shape(f, T, T) * _shape(f, inventory - i0, T)
    return max(PRICE_FLOOR, int(round(price)))


def _supply_cap(params, inventory, sold):
    """How many of `sold` consecutive sales starting at `inventory` add supply (quote > $1)."""
    if sold <= 0:
        return 0
    sold = int(sold)
    if market_price(params, inventory + sold - 1) > PRICE_FLOOR:
        return sold
    lo, hi = 0, sold - 1                      # first j with quote at the floor
    while lo < hi:
        mid = (lo + hi) // 2
        if market_price(params, inventory + mid) > PRICE_FLOOR:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _own_supply(net, prev, flow, params):
    """Own net market-supply change per product: the own holdings identity (sold - bought),
    bounded by the own orders actually submitted, by the $1 floor (no supply added) and, for
    sell-only products, by the exact total flow of both players."""
    supply = np.zeros(N_ITEMS)
    req = prev.action[1] if prev.action is not None else None
    for i, p in enumerate(PRODUCTS):
        n = net[i]
        if req is None:
            sell, buy = math.inf, (math.inf if p in BUYABLE else 0.0)
        else:
            sell, buy = req[i, 0], req[i, 1]
        if buy == 0:
            sold = min(max(n, 0.0), sell)
            value = min(sold, _supply_cap(params[p], prev.it[i, IR["mkt_inv"]], sold))
            if p not in BUYABLE:
                value = min(value, max(flow[i], 0.0))
            supply[i] = value
        elif sell == 0:
            supply[i] = -min(max(-n, 0.0), buy)
        else:
            supply[i] = n
    return supply


def _own_day_end_collects(prev):
    """COLLECT_FERTILIZER in the hour-23 step is erased on the tile by the day-end refresh;
    count it from the own submitted action and the own public tile it stood on."""
    ops = prev.action[0]["unit_ops"]
    taken, n = set(), 0
    tiles = prev.tiles[0]
    for op, pos in zip(ops, prev.own_pos):
        if op != "COLLECT_FERTILIZER" or pos in taken:
            continue
        x, y = pos
        size = int(round(len(tiles) ** 0.5))
        t = tiles[y * size + x] if 0 <= x < size and 0 <= y < size else None
        if isinstance(t, tuple) and len(t) > 1 and t[0] != "PLANT" and t[6]:
            taken.add(pos)
            n += 1
    return n


class _Snap:
    __slots__ = ("step", "day", "hour", "g", "it", "tiles", "shops", "cum", "gcum", "at_cap",
                 "action", "own_pos", "params", "loads")

    def __init__(self):
        self.action = None


def _town_consumption(step, shops, cfg):
    cons = np.zeros(N_ITEMS, np.float64)
    if step % max(1, int(cfg["townShopSellInterval"])) == 0:
        for shop in shops:
            products = SHOPS.get(shop, ())
            mult = 2 if len(products) == 1 else 1
            for p in products:
                cons[ITEM_INDEX[p]] += mult
    if step % max(1, int(cfg["townCenterSellInterval"])) == 0:
        for p in TOWN_CENTER_PRODUCTS:
            cons[ITEM_INDEX[p]] += 1
    return cons


def _hire_cost(before, after, mult):
    return sum(mult * fib(n) for n in range(int(before), int(after)))


def _land_cost(before, after):
    return sum(LAND_PRICES[n - 1] for n in range(int(before), int(after))
               if 0 <= n - 1 < len(LAND_PRICES))


_CODEC = None


def _codec():
    """codec.py of this build, if importable (optional; counts never depend on it)."""
    global _CODEC
    if _CODEC is None:
        try:
            import codec as module
            _CODEC = module if hasattr(module, "encode_step") else False
        except Exception:
            _CODEC = False
    return _CODEC or None


def _parse_action(action, units, max_orders, cap=100):
    """Own wire action -> (global counts dict, per-item matrix [N_ITEMS, 7])."""
    ops = dict.fromkeys(UNIT_OPS, 0)
    per_item = np.zeros((N_ITEMS, 7), np.float64)  # sell, buy_product, buy_seed, buy_animal, pickup, place, plant
    malformed = market_orders = hires = lands = 0
    if not isinstance(action, dict):
        action = {}
    farmer = action.get("farmer", ["PASS"])
    hands = action.get("hands", [])
    hands = hands if isinstance(hands, list) else []
    commands = [farmer, *hands][:max(1, units)]
    commands += [["PASS"]] * (max(1, units) - len(commands))
    for cmd in commands:
        if not isinstance(cmd, (list, tuple)) or not cmd or cmd[0] not in ops:
            malformed += 1
            continue
        op = cmd[0]
        ops[op] += 1
        row = ITEM_INDEX.get(cmd[1]) if len(cmd) >= 2 and isinstance(cmd[1], str) else None
        if row is None:
            continue
        qty = 1
        if len(cmd) >= 3:
            try:
                qty = min(max(0, int(cmd[2])), cap)   # a PICKUP/PLACE moves at most the shed capacity
            except (TypeError, ValueError, OverflowError):
                qty = 0
        if op == "PICKUP":
            per_item[row, 4] += qty
        elif op == "PLACE":
            per_item[row, 5] += qty
        elif op == "PLANT" and cmd[1] in CROPS:
            per_item[row, 6] += 1
    market = action.get("market", [])
    market = market[:max_orders] if isinstance(market, list) else []
    for order in market:
        if not isinstance(order, (list, tuple)) or not order or order[0] not in MARKET_OPS:
            malformed += 1
            continue
        market_orders += 1
        op = order[0]
        if op == "HIRE":
            hires += 1
            continue
        if op == "BUY_LAND":
            lands += 1
            continue
        row = ITEM_INDEX.get(order[1]) if len(order) >= 3 and isinstance(order[1], str) else None
        try:
            qty = int(order[2]) if row is not None else 0
        except (TypeError, ValueError, OverflowError):
            qty = 0
        if row is None or qty <= 0:
            malformed += 1
            continue
        col = {"SELL": 0, "BUY_PRODUCT": 1, "BUY_SEED": 2, "BUY_ANIMAL": 3}[op]
        if op != "BUY_SEED":            # each unit needs shed stock or room: engine-identical cap
            qty = min(qty, cap)
        per_item[row, col] += qty
    unit_ops = [cmd[0] if isinstance(cmd, (list, tuple)) and cmd else None for cmd in commands]
    counts = {"units": len(commands), "ops": ops, "market_orders": market_orders,
              "hires": hires, "lands": lands, "malformed": malformed, "unit_ops": unit_ops}
    return counts, per_item


# applicability of prev per-item columns
_PREV_APPLICABLE = np.zeros((N_ITEMS, 7), np.bool_)
for _i, _item in enumerate(ITEMS):
    _PREV_APPLICABLE[_i, 0] = _item in PRODUCTS
    _PREV_APPLICABLE[_i, 1] = _item in ("WHEAT", "FERTILIZER")
    _PREV_APPLICABLE[_i, 2] = _item in CROPS
    _PREV_APPLICABLE[_i, 3] = _item in ANIMALS
    _PREV_APPLICABLE[_i, 4] = True
    _PREV_APPLICABLE[_i, 5] = True
    _PREV_APPLICABLE[_i, 6] = _item in CROPS


# ----------------------------------------------------------------------------- tracker
class HistoryFeatures:
    """Causal per-seat state.  One instance per seat; call update() once per step."""

    def __init__(self):
        self.seat = None
        self.reset()

    def reset(self):
        self._hist = deque(maxlen=MAX_LAG + 1)
        self._from_start = False
        self._gap = False

    # .......................................................... snapshot
    def _snapshot(self, obs, cfg):
        s = _Snap()
        step = int(obs["step"])
        s.step, s.day, s.hour = step, step // cfg["turnsPerDay"], step % cfg["turnsPerDay"]
        seat = int(obs["player"])
        farms = obs["farms"]
        own, rival = farms[seat], farms[1 - seat]
        g = np.zeros(len(G_RAW), np.float64)
        it = np.zeros((N_ITEMS, len(I_RAW)), np.float64)
        for prefix, farm in (("own_", own), ("rival_", rival)):
            g[GR[prefix + "cash"]] = float(farm.get("money", 0))
            g[GR[prefix + "units"]] = 1 + len(farm.get("hands") or ())
            g[GR[prefix + "quads"]] = len(farm.get("unlocked_quadrants") or ())
            g[GR[prefix + "hires"]] = int(farm.get("hires_today") or 0)
        s.own_pos = [tuple(own.get("farmer") or (4, 4)), *(tuple(h) for h in own.get("hands") or ())]
        at_cap = [0, 0]
        s.tiles = (_scan_farm(own.get("tiles") or (), g, it, 0, at_cap),
                   _scan_farm(rival.get("tiles") or (), g, it, 1, at_cap))
        s.at_cap = at_cap
        private = obs.get("private") or {}
        for item, n in (private.get("shed") or {}).items():
            row = ITEM_INDEX.get(item)
            if row is not None:
                it[row, IR["own_shed"]] = n
        for crop, n in (private.get("seeds") or {}).items():
            row = ITEM_INDEX.get(crop)
            if row is not None:
                it[row, IR["own_seeds"]] = n
        loads = []
        for inv in private.get("inventories") or ():
            load = 0
            for item, n in inv.items():
                row = ITEM_INDEX.get(item)
                if row is not None:
                    it[row, IR["own_carried"]] += n
                    load += n
            loads.append(load)
        s.loads = loads
        g[GR["own_shed_total"]] = it[:, IR["own_shed"]].sum()
        g[GR["own_carried_total"]] = it[:, IR["own_carried"]].sum()
        g[GR["own_seed_total"]] = it[:, IR["own_seeds"]].sum()
        market = obs.get("market") or {}
        inv, prices = market.get("inventory") or {}, market.get("prices") or {}
        for i, p in enumerate(PRODUCTS):
            it[i, IR["mkt_inv"]] = inv.get(p, 0)
            it[i, IR["mkt_price"]] = prices.get(p, 0)
        shops = (obs.get("town") or {}).get("unlocked_shops")
        s.shops = None if shops is None else tuple(shops)
        s.g, s.it = g, it
        return s

    # .......................................................... transition s -> s+1
    def _advance(self, prev, now, cfg):
        cum, gcum = prev.cum.copy(), prev.gcum.copy()
        day_end = (prev.step + 1) % cfg["turnsPerDay"] == 0
        if prev.shops is None:
            cons = np.zeros(N_ITEMS)
            gcum[GC["shops_unknown_steps"]] += 1
        else:
            cons = _town_consumption(prev.step, prev.shops, cfg)
        d_inv = now.it[:, IR["mkt_inv"]] - prev.it[:, IR["mkt_inv"]]
        flow = np.where(PRODUCT_ROWS, d_inv + cons, 0.0)
        obtained = [np.zeros(N_ITEMS), np.zeros(N_ITEMS)]
        used = [np.zeros(N_ITEMS), np.zeros(N_ITEMS)]
        for role in (0, 1):
            _removals(prev.tiles[role], now.tiles[role], prev.step, day_end, obtained[role], used[role])
        if day_end and prev.action is not None:
            obtained[0][FERT] += _own_day_end_collects(prev)
        holdings_now = now.it[:, IR["own_shed"]] + now.it[:, IR["own_carried"]]
        holdings_prev = prev.it[:, IR["own_shed"]] + prev.it[:, IR["own_carried"]]
        net = -(holdings_now - holdings_prev) + obtained[0] - used[0]      # sold - bought (+discards)
        floor = prev.it[:, IR["mkt_price"]] <= PRICE_FLOOR
        own_trade = _own_supply(net, prev, flow, prev.params)
        rival_trade = np.where(PRODUCT_ROWS, flow - own_trade, 0.0)
        cum[:, IC["town_consumed"]] += cons
        cum[:, IC["trade_flow"]] += flow
        cum[:, IC["own_obtained"]] += obtained[0]
        cum[:, IC["own_used"]] += used[0]
        cum[:, IC["own_trade"]] += own_trade
        cum[:, IC["rival_obtained"]] += obtained[1]
        cum[:, IC["rival_used"]] += used[1]
        cum[:, IC["rival_trade"]] += rival_trade
        # holdings are never negative: a negative running estimate reveals unobserved inflow
        cum[:, IC["rival_stock"]] = np.where(PRODUCT_ROWS, np.maximum(
            0.0, cum[:, IC["rival_stock"]] + obtained[1] - used[1] - rival_trade), 0.0)
        cum[:, IC["floor_steps"]] += np.where(PRODUCT_ROWS, floor, False)
        mult = int(cfg["farmHandCostMult"])
        for role, prefix in ((0, "own_"), (1, "rival_")):
            d_cash = now.g[GR[prefix + "cash"]] - prev.g[GR[prefix + "cash"]]
            # A hire in the hour-23 step is erased by the day-end reset (hires_today=0,
            # hands removed), so its cost is unobservable there and counted as 0.
            spend = 0.0 if day_end else _hire_cost(prev.g[GR[prefix + "hires"]], now.g[GR[prefix + "hires"]], mult)
            spend += _land_cost(prev.g[GR[prefix + "quads"]], now.g[GR[prefix + "quads"]])
            gcum[GC[prefix + "trade_cash"]] += d_cash + spend
        if now.g[GR["own_shed_total"]] >= cfg["shedCapacity"]:
            gcum[GC["discard_risk_steps"]] += 1
        now.cum, now.gcum = cum, gcum

    # .......................................................... public API
    def update(self, observation, configuration=None):
        cfg = _cfg(configuration)
        seat = int(observation["player"])
        if self.seat is None:
            self.seat = seat
        elif seat != self.seat:
            raise ValueError("one HistoryFeatures instance per seat; use SeatHistories")
        step = int(observation["step"])
        now = self._snapshot(observation, cfg)
        now.params = _market_params(observation, configuration)
        prev = self._hist[-1] if self._hist else None
        if prev is not None and step == prev.step:      # same step observed again: recompute it
            self._hist.pop()
            prev = self._hist[-1] if self._hist else None
            if prev is None:
                now.cum = np.zeros((N_ITEMS, len(I_CUM)), np.float64)
                now.gcum = np.zeros(len(G_CUM), np.float64)
        if prev is not None and step == prev.step + 1:
            self._advance(prev, now, cfg)
        elif prev is not None or not hasattr(now, "cum"):
            gap = prev is not None and step != 0
            self.reset()
            self._from_start = step == 0
            self._gap = gap
            now.cum = np.zeros((N_ITEMS, len(I_CUM)), np.float64)
            now.gcum = np.zeros(len(G_CUM), np.float64)
        self._hist.append(now)
        self._cfg = cfg
        self._last_obs = observation
        self._params = now.params
        return self._features()

    def record_action(self, action):
        """Record the exact wire action this seat submitted at the last updated step."""
        if not self._hist:
            raise ValueError("update() must precede record_action()")
        now = self._hist[-1]
        counts, per_item = _parse_action(action, int(now.g[GR["own_units"]]),
                                         int(self._cfg["maxMarketOrdersPerTurn"]),
                                         int(self._cfg["shedCapacity"]))
        tokens = None
        codec = _codec()
        if codec is not None:
            try:
                tokens = codec.encode_step(action, self._last_obs)
            except Exception:
                tokens = None
        now.action = (counts, per_item, tokens)

    # .......................................................... features
    def _features(self):
        hist, cfg = self._hist, self._cfg
        now = hist[-1]
        prev = hist[-2] if len(hist) >= 2 else None
        blocks = {}

        # global now
        v = np.zeros(len(GLOBAL_NOW), np.float64)
        k = np.ones(len(GLOBAL_NOW), np.bool_)
        f = _FIELD_INDEX["history_global_now"]
        tpd = cfg["turnsPerDay"]
        angle = 2 * math.pi * now.hour / tpd
        v[f["phase.hour_sin"]], v[f["phase.hour_cos"]] = math.sin(angle), math.cos(angle)
        v[f["phase.tick4_%d" % (now.step % 4)]] = 1
        v[f["phase.day_end_follows"]] = (now.step + 1) % tpd == 0 and now.step < cfg["episodeSteps"] - 2
        v[f["phase.final_action"]] = now.step == cfg["episodeSteps"] - 2
        v[f["phase.day_mod3_%d" % (now.day % 3)]] = 1
        v[f["hist.from_start"]] = self._from_start
        v[f["hist.depth"]] = len(hist) - 1
        v[f["hist.reset_gap"]] = self._gap
        g = now.g
        v[f["flag.cash_lead"]] = g[GR["own_cash"]] > g[GR["rival_cash"]]
        v[f["flag.cash_tie"]] = g[GR["own_cash"]] == g[GR["rival_cash"]]
        v[f["flag.shed_empty"]] = g[GR["own_shed_total"]] <= 0
        v[f["flag.shed_full"]] = g[GR["own_shed_total"]] >= cfg["shedCapacity"]
        v[f["flag.has_seed"]] = g[GR["own_seed_total"]] > 0
        v[f["flag.has_carried"]] = g[GR["own_carried_total"]] > 0
        v[f["own.at_cap_tiles"]], v[f["rival.at_cap_tiles"]] = now.at_cap
        v[f["own.discard_risk_steps"]] = now.gcum[GC["discard_risk_steps"]]
        # Return trips (planning_trackers transport.*/terminal.*): a loaded unit needs its
        # Manhattan distance to the nearest shed-access tile (locked tiles are traversable,
        # units never block) plus one PLACE/DROP; the same step's market can sell it.  Only
        # the final day has no day-end deposit, so only there can carried items be stranded.
        half = int(round(len(now.tiles[0]) ** 0.5)) // 2
        access = ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))
        last = cfg["episodeSteps"] - 2
        steps_left = last - now.step + 1
        final_day = (now.day + 1) * tpd - 1 > last
        total = longest = stranded = 0
        for (x, y), load in zip(now.own_pos, now.loads):
            if load <= 0:
                continue
            need = min(abs(x - ax) + abs(y - ay) for ax, ay in access) + 1
            total += need
            longest = max(longest, need)
            if final_day and need > steps_left:
                stranded += load
        v[f["own.loaded_return_steps"]] = total
        v[f["own.max_return_steps"]] = longest
        v[f["own.stranded_carried"]] = stranded
        n_animals = g[GR["own_animals"]]
        if n_animals > 0:
            wheat = now.it[WHEAT, IR["own_shed"]] + now.it[WHEAT, IR["own_carried"]]
            v[f["own.feed_cover_days"]] = wheat / n_animals
        else:
            k[f["own.feed_cover_days"]] = False
        blocks["history_global_now"] = self._pack("history_global_now", v[None], k[None])

        # global per lag
        rows = len(LAGS)
        fg = _FIELD_INDEX["history_global_lag"]
        vg = np.zeros((rows, len(GLOBAL_HIST)), np.float64)
        kg = np.zeros(vg.shape, np.bool_)
        fi = _FIELD_INDEX["history_item_lag"]
        vi = np.zeros((N_ITEMS, len(ITEM_HIST)), np.float64)
        ki = np.zeros(vi.shape, np.bool_)
        per = len(ITEM_HIST_PER_LAG)
        base_price = np.ones(N_ITEMS)
        base_price[:N_PRODUCTS] = [max(1e-9, float(self._params[p]["base"])) for p in PRODUCTS]
        for r, lag in enumerate(LAGS):
            then = hist[max(0, len(hist) - 1 - lag)]
            age = now.step - then.step
            vg[r, fg["lag.value"]] = lag
            vg[r, fg["lag.exact"]] = age == lag
            vg[r, fg["lag.age"]] = age / lag
            vg[r, fg["lag.day_ends"]] = now.day - then.day
            kg[r, :4] = True
            if age <= 0:
                continue
            dg = now.g - then.g
            dc = now.gcum - then.gcum
            source = np.concatenate((dg, dc, (dg[GR["own_cash"]] - dg[GR["rival_cash"]],)))
            vg[r, _LAG_DST] = source[_LAG_SRC]
            kg[r, _LAG_DST] = True
            di = now.it - then.it
            dci = now.cum - then.cum
            base = r * per
            shops_ok = dc[GC["shops_unknown_steps"]] == 0
            cols = (
                (di[:, IR["mkt_inv"]], PRODUCT_ROWS),
                (np.where(PRODUCT_ROWS, di[:, IR["mkt_price"]] / base_price, 0.0), PRODUCT_ROWS),
                (dci[:, IC["trade_flow"]], PRODUCT_ROWS & shops_ok),
                (dci[:, IC["town_consumed"]], PRODUCT_ROWS & shops_ok),
                (di[:, IR["own_shed"]] + di[:, IR["own_carried"]], np.ones(N_ITEMS, np.bool_)),
                (dci[:, IC["own_trade"]], PRODUCT_ROWS),
                (dci[:, IC["rival_trade"]], PRODUCT_ROWS & shops_ok),
                (dci[:, IC["rival_obtained"]], PRODUCT_ROWS),
                (di[:, IR["rival_yield"]], PRODUCT_ROWS),
            )
            for j, (value, known) in enumerate(cols):
                vi[:, base + j] = np.where(known, value, 0.0)
                ki[:, base + j] = known
        blocks["history_global_lag"] = self._pack("history_global_lag", vg, kg)
        blocks["history_item_lag"] = self._pack("history_item_lag", vi, ki)

        # prev action (own action submitted at step-1)
        fp = _FIELD_INDEX["history_prev_action"]
        vp = np.zeros(len(PREV_ACTION), np.float64)
        kp = np.zeros(len(PREV_ACTION), np.bool_)
        kp[fp["prev.known"]] = True
        action = prev.action if prev is not None else None
        if action is not None:
            counts = action[0]
            kp[:] = True
            vp[fp["prev.known"]] = 1
            vp[fp["prev.units_commanded"]] = counts["units"]
            for op, n in counts["ops"].items():
                vp[fp["prev.op." + op]] = n
            vp[fp["prev.market_orders"]] = counts["market_orders"]
            vp[fp["prev.hire_orders"]] = counts["hires"]
            vp[fp["prev.land_orders"]] = counts["lands"]
            vp[fp["prev.malformed"]] = counts["malformed"]
        blocks["history_prev_action"] = self._pack("history_prev_action", vp[None], kp[None])

        # item now
        fn = _FIELD_INDEX["history_item_now"]
        vn = np.zeros((N_ITEMS, len(ITEM_NOW)), np.float64)
        kn = np.zeros(vn.shape, np.bool_)
        it = now.it
        params = self._params
        for i, p in enumerate(PRODUCTS):
            pp = params[p]
            price, inv = it[i, IR["mkt_price"]], it[i, IR["mkt_inv"]]
            vn[i, fn["mkt.price_vs_base"]] = price / pp["base"] - 1
            vn[i, fn["mkt.inv_vs_target"]] = (inv - pp["I0"]) / pp["T"]
            vn[i, fn["mkt.scarce"]] = inv < pp["I0"]
        mk = PRODUCT_ROWS
        for name in ("mkt.price_vs_base", "mkt.inv_vs_target", "mkt.scarce",
                     "cum.town_consumed", "cum.trade_flow"):
            kn[:, fn[name]] = mk
        if prev is not None:
            dp = it[:, IR["mkt_price"]] - prev.it[:, IR["mkt_price"]]
            vn[:, fn["mkt.price_up1"]] = np.where(mk, dp > 0, 0)
            vn[:, fn["mkt.price_down1"]] = np.where(mk, dp < 0, 0)
            kn[:, fn["mkt.price_up1"]] = mk
            kn[:, fn["mkt.price_down1"]] = mk
        shops_ok = now.gcum[GC["shops_unknown_steps"]] == 0
        start_ok = self._from_start
        cum = now.cum
        if start_ok and shops_ok:
            vn[:, fn["cum.town_consumed"]] = np.where(mk, cum[:, IC["town_consumed"]], 0)
            vn[:, fn["cum.trade_flow"]] = np.where(mk, cum[:, IC["trade_flow"]], 0)
            vn[:, fn["cum.floor_steps"]] = np.where(mk, cum[:, IC["floor_steps"]], 0)
            stock = cum[:, IC["rival_stock"]]
            for name, value in (("rival.obtained_cum_est", cum[:, IC["rival_obtained"]]),
                                ("rival.trade_cum_est", cum[:, IC["rival_trade"]]),
                                ("rival.stock_est", stock)):
                vn[:, fn[name]] = np.where(mk, value, 0)
                kn[:, fn[name]] = mk
            kn[:, fn["cum.floor_steps"]] = mk
        else:
            kn[:, fn["cum.town_consumed"]] = False
            kn[:, fn["cum.trade_flow"]] = False
        if action is not None:
            c0 = fn["prev.sell"]
            vn[:, c0:c0 + 7] = np.where(_PREV_APPLICABLE, action[1], 0)
            kn[:, c0:c0 + 7] = _PREV_APPLICABLE
        blocks["history_item_now"] = self._pack("history_item_now", vn, kn)
        tokens = action[2] if action is not None and len(action) > 2 else None
        return HistoryOutput(now.step, blocks, tokens)

    @staticmethod
    def _pack(name, values, known):
        rows, fields = BLOCKS[name]
        out = np.where(known, values / _SCALES[name], 0.0).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("non-finite history feature in " + name)
        return rows, tuple(n for n, _ in fields), out, known.copy()


class SeatHistories:
    """One callable playing both seats keeps one causal tracker per seat."""

    def __init__(self):
        self._by_seat = {}

    def update(self, observation, configuration=None):
        seat = int(observation["player"])
        tracker = self._by_seat.setdefault(seat, HistoryFeatures())
        return tracker.update(observation, configuration)

    def record_action(self, seat, action):
        self._by_seat[int(seat)].record_action(action)


def episode_features(observations, actions=None, configuration=None):
    """Offline helper for dataset builders: the same causal code path as live play."""
    tracker = HistoryFeatures()
    out = []
    for t, obs in enumerate(observations):
        out.append(tracker.update(obs, configuration))
        if actions is not None and t < len(actions) and actions[t] is not None:
            tracker.record_action(actions[t])
    return out


def benchmark(observations, actions=None, configuration=None, repeat=1):
    """Mean/max milliseconds per update(+record_action) over a recorded episode."""
    times = []
    for _ in range(repeat):
        tracker = HistoryFeatures()
        for t, obs in enumerate(observations):
            begun = time.perf_counter()
            tracker.update(obs, configuration)
            if actions is not None:
                tracker.record_action(actions[t])
            times.append((time.perf_counter() - begun) * 1000.0)
    times = np.asarray(times)
    return {"steps": int(times.size), "mean_ms": float(times.mean()),
            "p99_ms": float(np.percentile(times, 99)), "max_ms": float(times.max())}
