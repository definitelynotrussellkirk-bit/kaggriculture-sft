"""Same-step causal prefix state for decoder queries (Astra feature review 2026-09-26, recommendation 1).

Before unit k decides, S_k is the exact own-unit state after the earlier units' commands of this step with a PASS suffix:
the engine's own _apply_unit_action from the actor-visible snapshot, with every PLANT of a crop whose requests so far
exceed the seeds held at step start replaced by PASS (the engine's atomic PLANT check). When a request first
over-subscribes a crop, the prefix is replayed from the snapshot, so earlier planting-dependent effects are undone.
Market orders run after all units: per product a stock interval [L, U] in the shed (SELL is exact; BUY_PRODUCT/BUY_ANIMAL
only raise U because the rival can change affordability). These are conditional facts, never masks.
Only the acting seat's farm, shed, seeds and inventories are read; nothing of the rival's private state.
"""
from __future__ import annotations

import copy
import sys
import types
from collections import Counter

import numpy as np

if "kaggle_environments.utils" not in sys.modules:       # the engine imports it only for episode initialisation
    try:
        import kaggle_environments.utils  # noqa: F401
    except ImportError:
        _ke, _ku = types.ModuleType("kaggle_environments"), types.ModuleType("kaggle_environments.utils")
        _ku.resolve_episode_seed = lambda env: 0
        sys.modules.setdefault("kaggle_environments", _ke)
        sys.modules["kaggle_environments.utils"] = _ku
from farm_neat.game import kaggriculture_engine_reference as E  # noqa: E402

CROP_ORDER = tuple(E.CROPS)                  # WHEAT, CARROT, TOMATO, STRAWBERRY, MELON
PRODUCTS = tuple(E.PRODUCTS)                 # 9 market products
MOVES = frozenset(E.FARMER_MOVES)
F_UP = 27
F_OP = 22
SHED_CAPACITY = 100
CASH_SCALARS = 31
CASH_THRESHOLD_NAMES = (*('seed_' + p for p in CROP_ORDER), *('animal_' + p for p in E.ANIMALS),
                        'product_WHEAT', 'product_FERTILIZER', 'hire', 'land', 'shed_room',
                        *('structure_' + p for p in E.ANIMALS), 'slot_room')
F_CASH = CASH_SCALARS + 3 * len(CASH_THRESHOLD_NAMES)


def _margin_triplet(available, required):
    """Finite signed distance to a threshold; absent thresholds use an explicit mask elsewhere."""
    z = (float(available) - float(required)) / max(1.0, float(required))
    return min(2.0, max(-2.0, z)), float(np.sign(z) * np.log1p(abs(z))), float(z >= 0)


class PrefixExecutor:
    def __init__(self, obs, shed_capacity=SHED_CAPACITY, configuration=None, cash_enabled=True):
        seat = int(obs["player"])
        self.day = int(obs["step"]) // 24
        self.configuration = configuration or {}
        self.cash_enabled = cash_enabled
        self.cap = int(self.configuration.get("shedCapacity", shed_capacity))
        self.board_size = int(self.configuration.get("boardSize", len(obs["farms"][seat]["tiles"])))
        self.hire_mult = int(self.configuration.get("farmHandCostMult", E.FARM_HAND_COST_MULT))
        farm = obs["farms"][seat]
        private = obs.get("private") or {}
        self.init_farm = copy.deepcopy(farm)
        self.init_private = {k: copy.deepcopy(private.get(k, d)) for k, d in (("shed", {}), ("seeds", {}),
                                                                             ("inventories", [{}]))}
        self.seeds0 = dict(self.init_private["seeds"])
        self.positions = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
        self.cmds, self.requests, self.blocked = [], Counter(), set()
        self.farm, self.private = copy.deepcopy(self.init_farm), copy.deepcopy(self.init_private)
        self.intervals = None
        self.orders = []
        self.init_market = copy.deepcopy(obs["market"])
        self.cash_farm = None
        self.cash_private = None
        self.cash_market = None
        self.cash_start = int(farm["money"])
        self.cash_estimate = 0
        self.bought_seeds = Counter()
        self.bought_animals = Counter()
        self.bought_hands = 0
        self.bought_land = 0

    # ------------------------------------------------------------------ units
    def _apply(self, idx, a):
        if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] in self.blocked:
            a = ["PASS"]
        E._apply_unit_action(self.farm, self.private, idx, a, len(self.farm["tiles"]), self.day, 24, self.cap)

    def apply_unit(self, k, a):
        if k != len(self.cmds):
            raise ValueError("units must be applied in order")
        a = a if isinstance(a, list) and a else ["PASS"]
        self.cmds.append(a)
        if len(a) >= 2 and a[0] == "PLANT":
            self.requests[a[1]] += 1
            if self.requests[a[1]] > self.seeds0.get(a[1], 0) and a[1] not in self.blocked:
                self.blocked.add(a[1])
                self.farm, self.private = copy.deepcopy(self.init_farm), copy.deepcopy(self.init_private)
                for i, b in enumerate(self.cmds):
                    self._apply(i, b)
                return
        self._apply(k, a)

    def unit_features(self, k):
        f = np.zeros(F_UP, np.float32)
        if k >= len(self.positions):
            return f
        x, y = self.positions[k]
        tile, start = self.farm["tiles"][y][x], self.init_farm["tiles"][y][x]
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            f[0] = 1
            cd = E.CROPS[tile["crop"]]
            f[5] = float(tile["watered_today"])
            f[10] = float(self.day - tile["planted_day"] >= cd["first_yield_day"] and tile["yield_units"] > 0)
        elif isinstance(tile, dict) and "animal" in tile:
            f[1] = 1
            f[6], f[7] = float(tile["fed_today"]), float(tile["cared_today"])
            f[8] = float(bool(tile.get("fertilizer_available")))
            f[10] = float(tile["yield_units"] > 0)
        elif isinstance(tile, dict):
            f[2] = 1
        elif tile is None:
            f[3] = 1
        else:
            f[4] = 1
        if isinstance(tile, dict):
            f[9] = tile.get("yield_units", 0) / 10
        f[11] = float(tile != start)
        f[12] = sum(1 for j in range(k) if self.positions[j] == (x, y) and self.cmds[j][0] not in MOVES
                    and self.cmds[j][0] != "PASS") / 4
        for i, c in enumerate(CROP_ORDER):
            f[13 + i] = (self.seeds0.get(c, 0) - self.requests.get(c, 0)) / 10
            f[18 + i] = float(c in self.blocked)
        shed = self.private["shed"]
        f[23] = (self.cap - sum(shed.values())) / self.cap
        f[24], f[25] = shed.get("WHEAT", 0) / 20, shed.get("FERTILIZER", 0) / 20
        f[26] = k / 25
        return f

    # ------------------------------------------------------------------ orders
    def finalize_units(self):
        if self.intervals is None:
            shed = self.private["shed"]
            self.intervals = {p: [int(shed.get(p, 0)), int(shed.get(p, 0))] for p in (*PRODUCTS, *E.ANIMALS)}
            if self.cash_enabled:
                self.cash_farm = copy.deepcopy(self.farm)
                self.cash_private = copy.deepcopy(self.private)
                self.cash_market = copy.deepcopy(self.init_market)

    @property
    def cash_money(self):
        self.finalize_units()
        return int(self.cash_farm["money"])

    @property
    def next_hire_cost(self):
        self.finalize_units()
        return E._hire_cost(int(self.cash_farm["hires_today"]), self.hire_mult)

    @property
    def next_land_cost(self):
        self.finalize_units()
        n = len(self.cash_farm["unlocked_quadrants"]) - 1
        return E.LAND_PRICES[n] if n < len(E.LAND_PRICES) else None

    def affordable_seed_count(self, crop):
        return max(0, self.cash_money // E.CROPS[crop]["seed"])

    def affordable_animal_count(self, animal):
        self.finalize_units()
        room = max(0, self.cap - sum(self.cash_private["shed"].values()))
        return min(room, max(0, self.cash_money // E.ANIMALS[animal]["cost"]))

    def affordable_product_count(self, product):
        self.finalize_units()
        room = max(0, self.cap - sum(self.cash_private["shed"].values()))
        money = self.cash_money
        inventory = int(self.cash_market["inventory"][product])
        params = self.cash_market.get("params")
        count = 0
        while count < room:
            price = E.market_price(product, inventory - 1, params)
            if money < price:
                break
            money -= price
            inventory -= 1
            count += 1
        return count

    def _apply_cash_order(self, a):
        if not isinstance(a, list) or not a:
            return
        if a[0] == "SELL":
            self.cash_estimate = 1
        parsed = E._parse_order(a)
        if parsed is None:
            return
        op = parsed["type"]
        if op == "HIRE":
            before = len(self.cash_farm["hands"])
            E._do_hire(self.cash_farm, self.cash_private, self.board_size, self.hire_mult)
            self.bought_hands += len(self.cash_farm["hands"]) - before
            return
        if op == "BUY_LAND":
            before = len(self.cash_farm["unlocked_quadrants"])
            E._do_buy_land(self.cash_farm, self.board_size)
            self.bought_land += len(self.cash_farm["unlocked_quadrants"]) - before
            return
        item = parsed["item"]
        for _ in range(parsed["remaining"]):
            inv = self.cash_market["inventory"]
            if op == "SELL" and item in PRODUCTS:
                price = E.market_price(item, inv[item], self.cash_market.get("params"))
            elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
                price = E.market_price(item, inv[item] - 1, self.cash_market.get("params"))
            elif op == "BUY_SEED" and item in E.CROPS:
                price = E.CROPS[item]["seed"]
            elif op == "BUY_ANIMAL" and item in E.ANIMALS:
                price = E.ANIMALS[item]["cost"]
            else:
                break
            if not E._commit_unit(op, item, price, self.cash_farm, self.cash_private, self.cash_market, self.cap):
                break
            if op == "BUY_SEED":
                self.bought_seeds[item] += 1
            elif op == "BUY_ANIMAL":
                self.bought_animals[item] += 1
        E._refresh_prices(self.cash_market)

    def apply_order(self, a):
        self.finalize_units()
        self.orders.append(a)
        if self.cash_enabled:
            self._apply_cash_order(a)
        if not isinstance(a, list) or len(a) < 3:
            return
        op, p = a[0], a[1]
        try:
            q = max(0, int(a[2]))
        except (TypeError, ValueError):
            return
        if op == "SELL" and p in PRODUCTS:
            lo, hi = self.intervals[p]
            self.intervals[p] = [max(0, lo - q), max(0, hi - q)]
        elif (op == "BUY_PRODUCT" and p in ("WHEAT", "FERTILIZER")) or (op == "BUY_ANIMAL" and p in E.ANIMALS):
            self.intervals[p][1] = min(self.cap, self.intervals[p][1] + q)

    def cash_features(self, k, max_orders=10):
        """Decision-time cash ladder under a no-rival-trade scenario; never a legality mask."""
        self.finalize_units()
        f = np.zeros(F_CASH, np.float32)
        money = self.cash_money
        land_cost = self.next_land_cost
        shed_room = max(0, self.cap - sum(self.cash_private["shed"].values()))
        structure = Counter(t.get("kind") for row in self.cash_farm["tiles"] for t in row
                            if isinstance(t, dict) and "animal" not in t)
        slots_left = max(0, max_orders - k)
        f[:8] = (money / 10000, (money - self.cash_start) / 10000, self.cash_estimate,
                 self.next_hire_cost / 10000, (land_cost or 0) / 10000, slots_left / 10,
                 self.bought_hands / 15, self.bought_land / 3)
        for i, crop in enumerate(CROP_ORDER):
            f[8 + i] = min(20, self.bought_seeds[crop]) / 20
            f[16 + i] = min(20, self.affordable_seed_count(crop)) / 20
        for i, animal in enumerate(E.ANIMALS):
            f[13 + i] = min(20, self.bought_animals[animal]) / 20
            f[21 + i] = min(10, self.affordable_animal_count(animal)) / 10
            f[28 + i] = min(25, structure[E.ANIMALS[animal]["structure"]]) / 25
        for i, product in enumerate(("WHEAT", "FERTILIZER")):
            f[24 + i] = self.affordable_product_count(product) / 100
        f[26] = float(money >= self.next_hire_cost)
        f[27] = float(land_cost is not None and money >= land_cost)
        thresholds = ([E.CROPS[c]["seed"] for c in CROP_ORDER]
                      + [E.ANIMALS[a]["cost"] for a in E.ANIMALS]
                      + [E.market_price(p, self.cash_market["inventory"][p] - 1,
                                        self.cash_market.get("params")) for p in ("WHEAT", "FERTILIZER")]
                      + [self.next_hire_cost, land_cost or 0])
        margins = [_margin_triplet(money, cost) for cost in thresholds[:-1]]
        margins.append(_margin_triplet(money, land_cost) if land_cost is not None else (0, 0, 0))
        margins.append(_margin_triplet(shed_room, 1))
        margins.extend(_margin_triplet(structure[E.ANIMALS[a]["structure"]], 1) for a in E.ANIMALS)
        margins.append(_margin_triplet(slots_left, 1))
        f[CASH_SCALARS:] = np.asarray(margins, np.float32).reshape(-1)
        return f

    def order_features(self, k, max_orders=10):
        self.finalize_units()
        f = np.zeros(F_OP, np.float32)
        for i, p in enumerate(PRODUCTS):
            lo, hi = self.intervals[p]
            f[i], f[9 + i] = lo / 100, (hi - lo) / 100
        f[18] = max(0, max_orders - k) / 10
        f[19] = float(any(isinstance(o, list) and o and o[0] in ("BUY_PRODUCT", "BUY_ANIMAL") for o in self.orders))
        f[20] = (self.cap - sum(lo for lo, _ in self.intervals.values())) / self.cap
        f[21] = len(self.blocked) / 5
        return f


def teacher_arrays(obs, wire_action, max_units, n_order_slots, *, configuration=None, include_cash=False):
    """Prefix features along a decoded (wire) action: unit rows [max_units, F_UP], order rows [n_order_slots, F_OP]."""
    pe = PrefixExecutor(obs, configuration=configuration, cash_enabled=include_cash)
    units = [wire_action.get("farmer", ["PASS"]), *wire_action.get("hands", [])]
    up = np.zeros((max_units, F_UP), np.float32)
    for k, a in enumerate(units):
        if k < max_units:
            up[k] = pe.unit_features(k)
        pe.apply_unit(k, a)
    op = np.zeros((n_order_slots, F_OP), np.float32)
    cash = np.zeros((n_order_slots, F_CASH), np.float32) if include_cash else None
    market = wire_action.get("market", [])
    for k in range(n_order_slots):
        op[k] = pe.order_features(k)
        if cash is not None:
            cash[k] = pe.cash_features(k, n_order_slots - 1)  # final decoder slot is forced STOP
        if k < len(market):
            pe.apply_order(market[k])
    if cash is not None:
        return up.astype(np.float16), op.astype(np.float16), cash.astype(np.float16)
    return up.astype(np.float16), op.astype(np.float16)
