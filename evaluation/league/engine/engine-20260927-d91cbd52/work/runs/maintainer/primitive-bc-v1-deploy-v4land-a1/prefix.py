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


class PrefixExecutor:
    def __init__(self, obs, shed_capacity=SHED_CAPACITY):
        seat = int(obs["player"])
        self.day = int(obs["step"]) // 24
        self.cap = shed_capacity
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

    def apply_order(self, a):
        self.finalize_units()
        self.orders.append(a)
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


def teacher_arrays(obs, wire_action, max_units, n_order_slots):
    """Prefix features along a decoded (wire) action: unit rows [max_units, F_UP], order rows [n_order_slots, F_OP]."""
    pe = PrefixExecutor(obs)
    units = [wire_action.get("farmer", ["PASS"]), *wire_action.get("hands", [])]
    up = np.zeros((max_units, F_UP), np.float32)
    for k, a in enumerate(units):
        if k < max_units:
            up[k] = pe.unit_features(k)
        pe.apply_unit(k, a)
    op = np.zeros((n_order_slots, F_OP), np.float32)
    market = wire_action.get("market", [])
    for k in range(n_order_slots):
        op[k] = pe.order_features(k)
        if k < len(market):
            pe.apply_order(market[k])
    return up.astype(np.float16), op.astype(np.float16)
