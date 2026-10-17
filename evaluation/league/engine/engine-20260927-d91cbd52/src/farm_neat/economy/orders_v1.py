"""Small persistent economic commands, not a new simulator or economic oracle.

Deliberate curriculum: unlocked tiles; wheat cycles; cows; daily hand targets;
single-unit observable-price sale rules. No economic enumeration or NPV gates.
The existing movement and ordered inventory projection implement the mechanics.
This experimental lane is NOT the deployed seven-edit neural policy.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from farm_neat.game.constants import ANIMALS, PRODUCTS, hire_cost, is_shed_adjacent
from farm_neat.economy.heuristic.policy import _MarketLedger, _haul_action, _project_unit_market_ledger
from farm_neat.economy.heuristic.state import View, manhattan, nearest_shed_tile, step_toward

SCHEMA = "farm-persistent-semantic-orders-v1"
SUPPORTED_CROPS = ("WHEAT",)
SUPPORTED_ANIMALS = ("COW",)
LAST_STEP = 718


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _site(tile):
    if not isinstance(tile, (list, tuple)) or len(tile) != 2:
        raise ValueError("tile must be [x, y]")
    return tuple(_integer(x, 0, 9, "coordinate") for x in tile)


class SemanticOrders:
    """One stateful policy/controller per seat. Reissuing a setter is idempotent.

    Cycles means successful whole-plant harvests since this tile order began,
    NOT additional cycles on every call. A weed never satisfies a crop cycle.
    Price rules test the current observation, not an unavailable limit-order API.
    """

    def __init__(self, *, hands=2):
        self.hands = _integer(hands, 0, 4, "daily hand target")
        self.orders = {}
        self.sale_rules = {}
        self.pending = None
        self.last_report = {}
        self.events = []

    def farm_tile(self, tile, crop="WHEAT", *, cycles=3):
        tile = _site(tile)
        if crop not in SUPPORTED_CROPS:
            raise ValueError("UNSUPPORTED_CROP: initial curriculum supports WHEAT")
        cycles = _integer(cycles, 1, 6, "total harvest cycles")
        key = f"{tile[0]},{tile[1]}"
        old = self.orders.get(key)
        if old and (old["kind"] != "CROP" or old["item"] != crop):
            raise ValueError("CONFLICTING_TILE_ORDER: no implicit disposal or replacement")
        self.orders[key] = {"kind": "CROP", "tile": list(tile), "item": crop,
                            "cycles": cycles, "completed": old["completed"] if old else 0}

    def keep_animal(self, tile, animal="COW", *, until="GAME_END"):
        tile = _site(tile)
        if animal not in SUPPORTED_ANIMALS or until != "GAME_END":
            raise ValueError("UNSUPPORTED_ANIMAL_ORDER: initial curriculum is COW until GAME_END")
        key = f"{tile[0]},{tile[1]}"
        proposed = {"kind": "ANIMAL", "tile": list(tile), "item": animal, "until": until}
        if key in self.orders and self.orders[key] != proposed:
            raise ValueError("CONFLICTING_TILE_ORDER: no implicit disposal or replacement")
        self.orders[key] = proposed

    def sell_when(self, item, *, price_above, reserve=0):
        if item not in PRODUCTS:
            raise ValueError("product required")
        if type(price_above) not in (float, int) or not math.isfinite(price_above) or price_above < 0:
            raise ValueError("finite nonnegative sale threshold required")
        self.sale_rules[item] = {"price_above": float(price_above),
                                 "reserve": _integer(reserve, 0, 100, "reserve")}

    def snapshot(self):
        return copy.deepcopy({"schema": SCHEMA, "hands": self.hands,
            "orders": self.orders, "sale_rules": self.sale_rules, "pending": self.pending,
            "last_report": self.last_report, "events": self.events})

    @classmethod
    def restore(cls, state):
        if state.get("schema") != SCHEMA:
            raise ValueError("explicit semantic-orders snapshot required")
        result = cls(hands=state["hands"])
        for key, order in state["orders"].items():
            if order["kind"] == "CROP":
                result.farm_tile(order["tile"], order["item"], cycles=order["cycles"])
                _integer(order["completed"], 0, 6, "completed cycles")
            elif order["kind"] == "ANIMAL":
                result.keep_animal(order["tile"], order["item"], until=order["until"])
            else:
                raise ValueError("unknown order kind")
            if key not in result.orders:
                raise ValueError("tile key mismatch")
            result.orders[key] = copy.deepcopy(order)
        for item, rule in state["sale_rules"].items():
            result.sell_when(item, **rule)
        for name in ("pending", "last_report", "events"):
            setattr(result, name, copy.deepcopy(state[name]))
        return result

    def _reconcile(self, v):
        if self.pending is None:
            return
        previous = self.pending
        if v.step != previous["step"] + 1:
            raise ValueError("consecutive observations required; restore the saved controller to resume")
        if previous.get("settled_observation_sha256"):
            if previous["settled_observation_sha256"] != digest(v.obs):
                raise ValueError("settled observation changed")
            return
        for attempted in previous["harvests"]:
            tile = v.tile(*attempted["tile"])
            # Farms are disjoint: only our recorded legal ripe HARVEST can remove
            # this plant into an empty tile. Weeds/failed requests cannot earn credit.
            if tile is None:
                order = self.orders[attempted["key"]]
                order["completed"] += 1
                self.events.append({"event": "CROP_CYCLE_COMPLETED", "step": previous["step"],
                                    "tile": attempted["tile"], "cycle": order["completed"]})
        previous["settled_observation_sha256"] = digest(v.obs)

    def observe(self, observation):
        """Settle one actual successor, including terminal, without issuing an action."""
        self._reconcile(View(observation))
        return {"observed_step": observation["step"], "orders": copy.deepcopy(self.orders),
                "events": copy.deepcopy(self.events), "last_dispatch": copy.deepcopy(self.last_report)}

    def _jobs(self, v):
        jobs, blocked = [], []
        def add(priority, order, op, need=None):
            jobs.append({"priority": priority, "tile": tuple(order["tile"]),
                         "op": op, "need": need})
        for order in self.orders.values():
            tile = v.tile(*order["tile"])
            if tile == "LOCKED":
                blocked.append({"tile": order["tile"], "cause": "LOCKED_SITE_NOT_IN_CURRICULUM"})
                continue
            if order["kind"] == "CROP":
                if order["completed"] >= order["cycles"]:
                    continue
                if tile is None:
                    # PLANT must be followed by a same-day WATER. The route is
                    # checked again per worker below; no profitability veto.
                    if v.hour < 22 and v.step < LAST_STEP:
                        add(4, order, ["PLANT", order["item"]])
                elif tile.get("kind") == "WEED":
                    add(3, order, ["DIG"])
                elif tile.get("kind") == "PLANT" and tile["crop"] == order["item"]:
                    age = v.day - tile["planted_day"]
                    if tile.get("yield_units", 0) > 0 and (age >= 4 or v.day == 29):
                        add(1, order, ["HARVEST"])
                    elif not tile.get("watered_today"):
                        add(0, order, ["WATER"])
                else:
                    blocked.append({"tile": order["tile"], "cause": "OCCUPIED_BY_OTHER_USE"})
            else:
                structure = ANIMALS[order["item"]]["structure"]
                if tile is None:
                    add(3, order, ["BUILD_" + structure])
                elif tile.get("kind") == "WEED":
                    add(3, order, ["DIG"])
                elif tile.get("kind") != structure or tile.get("animal") not in (None, order["item"]):
                    blocked.append({"tile": order["tile"], "cause": "OCCUPIED_BY_OTHER_USE"})
                elif not tile.get("animal"):
                    add(2, order, ["PLACE", order["item"]], order["item"])
                else:
                    if not tile.get("fed_today"):
                        add(0, order, ["FEED"], "WHEAT")
                    if tile.get("yield_units", 0):
                        add(1, order, ["HARVEST"])
                    if not tile.get("cared_today"):
                        add(5, order, ["CARE"])
                    if tile.get("fertilizer_available"):
                        add(6, order, ["COLLECT_FERTILIZER"])
        return jobs, blocked

    def _units(self, v, jobs):
        actions = [["PASS"] for _ in range(v.n_units)]
        free = set(range(v.n_units))
        available_shed = dict(v.shed)
        seeds = dict(v.seeds)
        claimed_tiles = set()
        routes = []
        for unit in sorted(free):
            inv, pos = v.unit_inv(unit), v.unit_pos(unit)
            produce = sum(n for item, n in inv.items() if item not in ANIMALS and item != "WHEAT")
            wheat = inv.get("WHEAT", 0)
            # Fixed logistics recipe, not an economic sale decision. Preserve
            # livestock cargo; use the existing capacity-safe unload helper.
            must_haul = (produce + wheat >= 8 or
                         (v.day == 29 and produce + wheat and
                          LAST_STEP - v.step <= manhattan(pos, nearest_shed_tile(pos, v)) + 2))
            if must_haul:
                actions[unit] = _haul_action(v, pos, inv)
                free.remove(unit)
        for job in sorted(jobs, key=lambda j: (j["priority"], j["tile"], j["op"])):
            if job["tile"] in claimed_tiles:
                continue
            candidates = []
            for unit in sorted(free):
                pos, inv = v.unit_pos(unit), v.unit_inv(unit)
                destination, action = job["tile"], job["op"]
                if job["need"] and not inv.get(job["need"], 0):
                    if available_shed.get(job["need"], 0) <= 0:
                        continue
                    destination = nearest_shed_tile(pos, v)
                    action = ["PICKUP", job["need"], 1]
                    distance = manhattan(pos, destination) + 1 + manhattan(destination, job["tile"])
                else:
                    distance = manhattan(pos, destination)
                if job["op"][0] == "PLANT":
                    if not seeds.get(job["op"][1], 0) or v.hour + distance + 1 >= 24 or v.step + distance + 1 > LAST_STEP:
                        continue
                candidates.append((distance, unit, destination, action))
            if not candidates:
                continue
            distance, unit, destination, op = min(candidates)
            pos = v.unit_pos(unit)
            actual = op if pos == destination else [step_toward(pos, destination, v)]
            actions[unit] = actual
            free.remove(unit)
            claimed_tiles.add(job["tile"])
            if actual[0] == "PICKUP":
                available_shed[actual[1]] -= actual[2]
            if actual[0] == "PLANT":
                seeds[actual[1]] -= 1
            routes.append({"unit": unit, "target": list(job["tile"]), "intent": job["op"],
                           "issued": actual, "minimum_travel_and_setup": distance})
        for unit in sorted(free):
            if sum(v.unit_inv(unit).values()):
                actions[unit] = _haul_action(v, v.unit_pos(unit), v.unit_inv(unit))
        projected = _project_unit_market_ledger(v, actions, shed_capacity=100, rewrite_unsafe_unloads=True)
        return projected, routes

    def _market(self, v, projected):
        ledger = _MarketLedger(v.money, v.market, projected["shed"], v.farm.get("hires_today", 0))
        stock = dict(projected["shed"])
        market, blocked = [], []
        cows = sum(o["kind"] == "ANIMAL" and v.tile(*o["tile"]) != "LOCKED" for o in self.orders.values())
        feed_reserve = 2 * cows  # explicit two-day supply recipe for standing animal service
        for item, rule in sorted(self.sale_rules.items()):
            reserve = max(rule["reserve"], feed_reserve if item == "WHEAT" else 0)
            if v.prices.get(item, 0) > rule["price_above"] and stock.get(item, 0) > reserve:
                order = ["SELL", item, 1]
                if ledger.commit(order):
                    market.append(order)
                    stock[item] -= 1
        def buy(op, item, count):
            for _ in range(count):
                if len(market) >= 10:
                    blocked.append({"cause": "MARKET_ORDER_SLOTS", "item": item})
                    return
                order = [op, item, 1]
                if ledger.commit(order) != 1:
                    blocked.append({"cause": "CASH_OR_SUPPLY_OR_SHED_CAPACITY", "item": item})
                    return
                market.append(order)
                if op != "BUY_SEED":
                    stock[item] = stock.get(item, 0) + 1
        carried_wheat = sum(inv.get("WHEAT", 0) for inv in projected["inventories"])
        buy("BUY_PRODUCT", "WHEAT", max(0, feed_reserve - stock.get("WHEAT", 0) - carried_wheat))
        while v.hour < 23 and ledger.hires_today < self.hands and len(market) < 10:
            if ledger.cash < hire_cost(ledger.hires_today):
                blocked.append({"cause": "HIRE_CASH", "next_hire_cost": hire_cost(ledger.hires_today)})
                break
            if not ledger.commit(["HIRE"]):
                break
            market.append(["HIRE"])
        actual_cows = sum(isinstance(v.tile(*o["tile"]), dict) and
                         v.tile(*o["tile"]).get("animal") == "COW"
                         for o in self.orders.values() if o["kind"] == "ANIMAL")
        cow_stock = stock.get("COW", 0) + sum(inv.get("COW", 0) for inv in projected["inventories"])
        # Count the current unit-phase placement so it cannot trigger a duplicate purchase.
        for pos, after in projected["tiles"].items():
            before = v.tile(*pos)
            actual_cows += int(isinstance(after, dict) and after.get("animal") == "COW")
            actual_cows -= int(isinstance(before, dict) and before.get("animal") == "COW")
        buy("BUY_ANIMAL", "COW", max(0, cows - actual_cows - cow_stock))
        needed_seeds = sum(o["kind"] == "CROP" and o["completed"] < o["cycles"] and
                           v.tile(*o["tile"]) is None for o in self.orders.values())
        buy("BUY_SEED", "WHEAT", max(0, needed_seeds - projected["seeds"].get("WHEAT", 0)))
        return market, blocked

    def act(self, observation, configuration=None):
        # The small curriculum deliberately fixes the ordinary board/time/storage
        # profile; it does not pretend to cover custom engine configurations.
        if configuration is not None:
            for key, expected in (("boardSize", 10), ("turnsPerDay", 24), ("episodeSteps", 720),
                                  ("shedCapacity", 100), ("maxMarketOrdersPerTurn", 10), ("farmHandCostMult", 1)):
                if configuration.get(key, expected) != expected:
                    raise ValueError("UNSUPPORTED_CONFIGURATION: " + key)
            if configuration.get("marketParams"):
                raise ValueError("UNSUPPORTED_CONFIGURATION: custom market parameters")
        v = View(observation)
        signature = digest(observation)
        if self.pending and v.step == self.pending["step"]:
            if self.pending.get("settled_observation_sha256"):
                raise ValueError("cannot return to an already settled previous turn")
            if signature != self.pending["observation_sha256"]:
                raise ValueError("same step with changed observation")
            return copy.deepcopy(self.pending["action"])
        self._reconcile(v)
        jobs, blocked = self._jobs(v)
        projected, routes = self._units(v, jobs)
        market, market_blocked = self._market(v, projected)
        actions = projected["actions"]
        action = {"farmer": actions[0], "hands": actions[1:], "market": market}
        harvests = []
        for unit, op in enumerate(actions):
            tile = v.tile(*v.unit_pos(unit))
            key = ",".join(map(str, v.unit_pos(unit)))
            if (op == ["HARVEST"] and key in self.orders and self.orders[key]["kind"] == "CROP"
                    and isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("yield_units", 0) > 0):
                harvests.append({"key": key, "tile": list(v.unit_pos(unit))})
        self.last_report = {"step": v.step, "daily_hand_target": self.hands, "available_units": v.n_units,
            "due_water_feed": sum(j["priority"] == 0 for j in jobs), "pending_jobs": len(jobs),
            "assigned_jobs": len(routes), "unassigned_jobs": len(jobs) - len(routes),
            "market_slots_used": len(market), "shed_used": v.shed_used(),
            "blocked": blocked + market_blocked, "routes": routes,
            "optimality": "NOT_CLAIMED_GREEDY_MECHANICAL_DISPATCH",
            "sale_condition": "OBSERVED_PRICE_STRICTLY_ABOVE_THRESHOLD_NOT_A_FILL_PRICE_GUARANTEE"}
        self.pending = {"step": v.step, "observation_sha256": signature, "action": copy.deepcopy(action),
                        "harvests": harvests}
        return action

    __call__ = act
