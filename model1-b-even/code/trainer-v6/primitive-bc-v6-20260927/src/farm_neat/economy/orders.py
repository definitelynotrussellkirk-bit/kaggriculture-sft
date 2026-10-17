"""Policy-owned economic requirements, lowered by a mechanical rolling dispatcher.

This is a NEW experimental interface, not a relabelled 29-action checkpoint.
All engine species/products participate. No NPV filter, fixed herd/crop mix,
implicit liquidation, hidden feed reserve, or economic-plan enumeration is used.
Market forecasts are own-queue projections, NOT opponent-proof fill guarantees.
"""
from __future__ import annotations

import copy
import math

from farm_neat.game.constants import ANIMALS, CROPS, LAND_PRICES, PRODUCTS, hire_cost
from farm_neat.economy.heuristic.policy import _MarketLedger, _haul_action, _project_unit_market_ledger
from farm_neat.economy.heuristic.state import View, manhattan, nearest_shed_tile, step_toward
from farm_neat.economy.orders_v1 import digest
from farm_neat.economy.action_space import validate_command

SCHEMA = "farm-full-economic-orders-v1"
LAST_STEP = 718
MARKET_ORDER_SLOTS = 10
# A turn carries at most MARKET_ORDER_SLOTS market orders and hands are cleared every
# night, so an unbounded HIRE batch spends the whole of the first turn of every day on
# itself -- exactly the turn at which that day's sales and input requisitions queue.
# `hire_cost` is priced by `hires_today`, not by the turn a hire lands in, so spreading
# the batch over consecutive turns costs nothing and the day still reaches its target.
HIRE_BATCH_LIMIT = 5


def _end(command):
    return LAST_STEP if command.get("until") == "GAME_END" else command.get("until", LAST_STEP)


def _active(command, step):
    return command.get("start", 0) <= step <= _end(command)


def _scope_sites(scope):
    if scope["kind"] == "TILE":
        return [(scope["x"], scope["y"])]
    region = scope["region"]
    positions = [(x, y) for y in range(10) for x in range(10)
                 if region == "ALL" or (("N" if y < 5 else "S") +
                                        ("W" if x < 5 else "E")) == region]
    # Placement is a deterministic logistics choice. Every cell can also be
    # addressed explicitly; occupancy/affordability does not hide an intent.
    positions.sort(key=lambda p: (min(manhattan(p, s) for s in
                                     ((4, 4), (5, 4), (4, 5), (5, 5))), p))
    return positions[:scope["count"]]


def _scope_key(scope):
    return (f"TILE:{scope['x']},{scope['y']}" if scope["kind"] == "TILE"
            else "REGION:" + scope["region"])


def _key(position):
    return f"{position[0]},{position[1]}"


def _rule_key(c):
    op = c["op"]
    if op == "SALE":
        return "SALE:" + c["product"]
    if op in ("STOCK", "BUY"):
        return f"{op}:{c['kind']}:{c['item']}"
    return op


class EconomicOrders:
    """Persistent requirements. Repeated setters do not restart credited work.

Cycles count plant lifecycles, not individual ongoing-product harvests. A
completed one-time harvest earns one cycle; a naturally exhausted ongoing
plant earns one only if it yielded an actual harvest. Weed death is a failure.
An ordinary setter cannot destroy an asset: CLEAR/REPLACE is explicit.
"""

    def __init__(self):
        self.orders, self.rules = {}, {}
        self.pending, self.last_report = None, {}
        self.events, self.revision = [], 0

    def apply(self, command):
        c = validate_command(command)
        if self.pending and not self.pending.get("settled_observation_sha256"):
            raise ValueError("observe the successor before editing an issued action")
        if c["op"] == "STOP":
            return
        before = digest([self.orders, self.rules])
        op = c["op"]
        if op in ("SET_CROP", "SET_ANIMAL", "REPLACE", "CLEAR"):
            replacing = op == "REPLACE"
            scope = c["scope"]
            if replacing:
                c = {**c["replacement"], "scope": scope}
            positions = _scope_sites(scope)
            group = _scope_key(scope)
            target_keys = {_key(p) for p in positions}
            # Validate the complete edit before changing any commitments.
            for p in positions:
                old = self.orders.get(_key(p))
                if old and op != "CLEAR" and not replacing:
                    oc = old["command"]
                    if (oc["op"], oc.get("crop"), oc.get("animal")) != (
                            c["op"], c.get("crop"), c.get("animal")):
                        raise ValueError("CONFLICTING_TILE_ORDER: use explicit REPLACE or CLEAR")
            for k, old in list(self.orders.items()):
                if old.get("scope_key") == group and k not in target_keys:
                    del self.orders[k]  # target reduction retires intent, not physical assets
            for p in positions:
                k = _key(p)
                command_at_site = {**c, "scope": {"kind": "TILE", "x": p[0], "y": p[1]}}
                old = self.orders.get(k)
                same_species = not replacing and old and (old["command"]["op"], old["command"].get("crop"),
                    old["command"].get("animal")) == (c["op"], c.get("crop"), c.get("animal"))
                state = copy.deepcopy(old) if same_species else {
                    "id": f"{self.revision + 1}:{k}", "completed_cycles": 0,
                    "successful_harvests": 0, "harvested_units": 0,
                    "birth": None, "cycle_harvests": 0, "fertilized_day": -1,
                    "clear_before": replacing}
                state.update(command=command_at_site, scope_key=group)
                self.orders[k] = state
        elif op == "CANCEL":
            target = c["target"]
            if target["kind"] == "PRODUCTION":
                for p in _scope_sites(target["scope"]):
                    self.orders.pop(_key(p), None)
            else:
                kind = target["kind"]
                key = kind
                if kind == "SALE":
                    key += ":" + target["product"]
                elif kind in ("STOCK", "BUY"):
                    key += ":" + target["stock_kind"] + ":" + target["item"]
                self.rules.pop(key, None)
        else:
            key = _rule_key(c)
            old = self.rules.get(key, {})
            self.rules[key] = {"command": c, "acquired": old.get("acquired", 0)}
        if digest([self.orders, self.rules]) != before:
            self.revision += 1

    def snapshot(self):
        return copy.deepcopy({"schema": SCHEMA, "orders": self.orders, "rules": self.rules,
            "pending": self.pending, "events": self.events, "last_report": self.last_report,
            "revision": self.revision})

    @classmethod
    def restore(cls, state):
        if set(state) != {"schema", "orders", "rules", "pending", "events", "last_report", "revision"} or state["schema"] != SCHEMA:
            raise ValueError("explicit full-economic snapshot required")
        result = cls()
        for k, order in state["orders"].items():
            c = validate_command(order["command"])
            if c["op"] not in ("SET_CROP", "SET_ANIMAL", "CLEAR") or c["scope"]["kind"] != "TILE":
                raise ValueError("snapshot production command")
            if _key(_scope_sites(c["scope"])[0]) != k:
                raise ValueError("snapshot tile key")
            for field in ("completed_cycles", "successful_harvests", "harvested_units", "cycle_harvests"):
                if type(order[field]) is not int or order[field] < 0:
                    raise ValueError("nonnegative realized counter required")
        for k, rule in state["rules"].items():
            if _rule_key(validate_command(rule["command"])) != k or type(rule["acquired"]) is not int or rule["acquired"] < 0:
                raise ValueError("snapshot rule identity/credit")
        for field in ("orders", "rules", "pending", "events", "last_report", "revision"):
            setattr(result, field, copy.deepcopy(state[field]))
        return result

    def _reconcile(self, v):
        p = self.pending
        if not p:
            return
        if v.step != p["step"] + 1:
            raise ValueError("consecutive observations required")
        signature = digest(v.obs)
        if p.get("settled_observation_sha256"):
            if signature != p["settled_observation_sha256"]:
                raise ValueError("settled observation changed")
            return
        # Validate the complete settlement BEFORE awarding any credit. A later
        # refused receipt must not leave earlier harvests/acquisitions credited.
        for receipt in p["harvests"]:
            order = self.orders.get(receipt["key"])
            if not order or order["id"] != receipt["order_id"]:
                raise ValueError("issued commitment changed before settlement")
            c = order["command"]
            if c["op"] == "SET_CROP" and not CROPS[c["crop"]]["ongoing"]:
                tile = v.tile(*receipt["tile"])
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    raise ValueError("one-time harvest was not realized")
        acquisitions = []
        for receipt in p["buys"]:
            kind, item = receipt["kind"], receipt["item"]
            after = v.seeds.get(item, 0) if kind == "SEED" else (
                v.shed.get(item, 0) + sum(i.get(item, 0) for i in v.inventories))
            acquired = after - receipt["before"]
            if receipt["rule_key"] not in self.rules or not 0 <= acquired <= receipt["requested"]:
                raise ValueError("ambiguous acquisition receipt; not credited")
            acquisitions.append((receipt, acquired))
        # Own unit-phase harvest legality is exact and independent of the
        # opponent. Successor holdings are separately replay-checked in tests;
        # these controller counters are NOT training-admission certificates.
        for receipt in p["harvests"]:
            order = self.orders.get(receipt["key"])
            if not order or order["id"] != receipt["order_id"]:
                raise ValueError("issued commitment changed before settlement")
            order["successful_harvests"] += 1
            order["harvested_units"] += receipt["units"]
            order["cycle_harvests"] += 1
            c = order["command"]
            if c["op"] == "SET_CROP" and not CROPS[c["crop"]]["ongoing"]:
                tile = v.tile(*receipt["tile"])
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    raise ValueError("one-time harvest was not realized")
                order["completed_cycles"] += 1
                order["birth"] = None
                order["cycle_harvests"] = 0
            self.events.append({"event": "HARVEST", "step": p["step"], **receipt})
        for k, order in self.orders.items():
            c = order["command"]
            if c["op"] != "SET_CROP":
                continue
            tile = v.tile(*_scope_sites(c["scope"])[0])
            birth = order["birth"]
            # Every lost plant retires its birth, including one-time crops
            # that die or decay without being harvested. Keeping that birth
            # makes the next legal replant look like an unexplained replacement.
            if birth and not (
                    isinstance(tile, dict) and tile.get("kind") == "PLANT" and
                    tile.get("crop") == birth["crop"] and
                    tile.get("planted_day") == birth["planted_day"]):
                final_age = CROPS[c["crop"]]["first_yield_day"] + 3 * CROPS[c["crop"]]["interval"]
                previous = p["pre_tiles"].get(k)
                harvested = sum(r["units"] for r in p["harvests"] if r["key"] == k)
                naturally_exhausted = (CROPS[c["crop"]]["ongoing"] and
                    isinstance(previous, dict) and
                    previous.get("max_lifespan_step", -1) >= 0 and
                    p["step"] >= previous["max_lifespan_step"] and
                    (p["step"] - previous["max_lifespan_step"]) % 2 == 0 and
                    previous.get("yield_units", 0) - harvested <= 1 and
                    p["step"] // 24 - birth["planted_day"] > final_age and
                    not any(a["key"] == k and a["op"] == "DIG" for a in p["operations"]))
                if naturally_exhausted and order["cycle_harvests"]:
                    order["completed_cycles"] += 1
                    self.events.append({"event": "ONGOING_CYCLE_COMPLETED", "step": p["step"], "key": k})
                else:
                    self.events.append({"event": "PLANT_LOST_WITHOUT_CYCLE_CREDIT", "step": p["step"], "key": k})
                order["birth"], order["cycle_harvests"] = None, 0
        for receipt, acquired in acquisitions:
            item = receipt["item"]
            rule = self.rules[receipt["rule_key"]]
            rule["acquired"] += acquired
            self.events.append({"event": "BUY_ACQUIRED", "step": p["step"],
                                "item": item, "units": acquired})
        p["settled_observation_sha256"] = signature

    def observe(self, observation):
        self._reconcile(View(observation))
        return self.snapshot()

    def _jobs(self, v):
        jobs, blocked = [], []
        # DEADLINE PRE-EMPTION. The engine destroys an asset at the daily refresh
        # when the neglect counter reaches two: `consecutive_unwatered >= 2` turns
        # a PLANT into a WEED and `consecutive_unfed >= 2` makes the animal escape
        # (the structure survives). Both counters are incremented for a day that
        # ends unwatered/unfed, so a site that already carries a counter of one
        # and is still untouched today is one missed day from total loss. That is
        # not a priority question the program can express -- a program that ranks
        # crops over animals is asking for an ordering among healthy sites, not
        # for its herd to starve -- so such a job outranks EVERY non-deadline job
        # regardless of the program's own priority. CARE is deliberately NOT in
        # this class: `cared_today` only accrues `pending_care_bonus` on a fed
        # production day and no engine rule destroys an uncared animal.
        def add(order, rank, op, need=None, deadline=False):
            c = order["command"]
            jobs.append({"priority": (0 if deadline else 1, c["priority"], rank),
                         "tile": _scope_sites(c["scope"])[0],
                         "op": op, "need": need, "key": _key(_scope_sites(c["scope"])[0]),
                         "deadline": bool(deadline)})
        for k, order in self.orders.items():
            c = order["command"]
            if not _active(c, v.step):
                continue
            pos = _scope_sites(c["scope"])[0]
            tile = v.tile(*pos)
            if tile == "LOCKED":
                blocked.append({"tile": list(pos), "cause": "LAND_NOT_UNLOCKED", "repair": "LAND target covering this site"})
                continue
            if c["op"] == "CLEAR" or order["clear_before"]:
                if isinstance(tile, dict) and "animal" in tile:
                    blocked.append({"tile": list(pos), "cause": "PLACED_ANIMAL_CANNOT_BE_CLEARED"})
                    continue
                if tile is not None:
                    add(order, 2, ["DIG"])
                    continue
                order["clear_before"] = False
                if c["op"] == "CLEAR":
                    continue
            if c["op"] == "SET_CROP":
                if c["cycles"] is not None and order["completed_cycles"] >= c["cycles"]:
                    continue
                if tile is None:
                    add(order, 4, ["PLANT", c["crop"]])
                elif tile.get("kind") == "WEED":
                    add(order, 2, ["DIG"])
                elif tile.get("kind") == "PLANT" and tile["crop"] == c["crop"]:
                    birth = {"crop": c["crop"], "planted_day": tile["planted_day"]}
                    if order["birth"] is None:
                        order["birth"] = birth
                    elif order["birth"] != birth:
                        raise ValueError("unaccounted plant replacement")
                    cd = CROPS[c["crop"]]
                    age = v.day - tile["planted_day"]
                    if c["fertilizing"] != "OFF" and order["fertilized_day"] != v.day and (
                            c["fertilizing"] == "DAILY" or tile.get("fertilized_until_day", -1) < v.day):
                        add(order, -1, ["FERTILIZE"], "FERTILIZER")
                    if c["watering"] != "OFF" and not tile.get("watered_today"):
                        # Fertilize-before-water can increase yield, but must
                        # not displace requested WATER at the final daily slot.
                        add(order, -2 if v.hour == 23 else 0, ["WATER"],
                            deadline=tile.get("consecutive_unwatered", 0) >= 1)
                    if age >= cd["first_yield_day"] and tile.get("yield_units", 0) >= max(1, c["harvest_threshold"]):
                        add(order, 1, ["HARVEST"])
                else:
                    blocked.append({"tile": list(pos), "cause": "OCCUPIED_BY_OTHER_USE", "repair": "explicit REPLACE or CLEAR"})
            elif c["op"] == "SET_ANIMAL":
                structure = ANIMALS[c["animal"]]["structure"]
                if tile is None:
                    add(order, 3, ["BUILD_" + structure])
                elif tile.get("kind") == "WEED":
                    add(order, 2, ["DIG"])
                elif tile.get("kind") != structure or tile.get("animal") not in (None, c["animal"]):
                    blocked.append({"tile": list(pos), "cause": "OCCUPIED_BY_OTHER_USE"})
                elif "animal" not in tile:
                    add(order, 2, ["PLACE", c["animal"]], c["animal"])
                else:
                    if c["feed"] != "OFF" and not tile.get("fed_today"):
                        add(order, 0, ["FEED"], "WHEAT",
                            deadline=tile.get("consecutive_unfed", 0) >= 1)
                    if tile.get("yield_units", 0) >= max(1, c["harvest_threshold"]):
                        add(order, 1, ["HARVEST"])
                    if c["care"] != "OFF" and not tile.get("cared_today"):
                        add(order, 5, ["CARE"])
                    if c["manure"] != "OFF" and tile.get("fertilizer_available"):
                        add(order, 6, ["COLLECT_FERTILIZER"])
        return jobs, blocked

    def _units(self, v, jobs):
        actions = [["PASS"] for _ in range(v.n_units)]
        free, claimed = set(range(v.n_units)), set()
        available, seeds = dict(v.shed), dict(v.seeds)
        routes, blocked = [], []
        for job in sorted(jobs, key=lambda j: j["priority"]):
            # ONE DUTY PER TILE PER TURN, deliberately, and NOT because the engine
            # requires it: `_apply_unit_action` accepts FEED and HARVEST from two units
            # standing on the same tile in the same turn (asserted against the engine in
            # tests/test_full_economic_orders_dispatch_v2.py). Keying this claim by
            # (tile, op) was implemented and measured on 2026-09-11 -- six seeds x four
            # programs, vs PASS and vs ShopRouter, in three op-set variants -- and lost
            # money on every variant: the freed unit clusters onto an already-served tile
            # instead of reaching an unserved one, because nearest-unit assignment has no
            # yield term to justify the cluster. The whole-tile claim is kept on evidence.
            if job["tile"] in claimed:
                blocked.append({"cause": "TILE_DUTY_SERIALIZED", "tile": list(job["tile"]),
                    "op": job["op"], "need": job["need"], "free_units": len(free)})
                continue
            candidates = []
            for unit in sorted(free):
                pos, inv = v.unit_pos(unit), v.unit_inv(unit)
                destination, action = job["tile"], job["op"]
                if job["need"] and not inv.get(job["need"], 0):
                    if not available.get(job["need"], 0):
                        continue
                    destination = nearest_shed_tile(pos, v)
                    action = ["PICKUP", job["need"], 1]
                distance = manhattan(pos, destination)
                if job["op"][0] == "PLANT":
                    c = self.orders[job["key"]]["command"]
                    if not seeds.get(job["op"][1], 0):
                        continue
                    if c["watering"] != "OFF" and (v.hour + distance + 1 >= 24 or v.step + distance + 1 > LAST_STEP):
                        continue
                # ONE INPUT UNIT PER TRIP. The engine's PICKUP moves `n` units at once and a
                # unit's inventory is uncapped, so batching several duties into one shed
                # trip is available and it does cut supply traffic hard: fetching
                # min(outstanding need / units, 6) took best_program.json from 376 supply
                # unit-turns a game to 295 (-22%), and to 184 (-51%) with the HIRE cap.
                # It also lost money doing it -- 24 seeds, best_program: -$3,592 vs PASS
                # and -$1,203 vs ShopRouter on own final cash -- because a carrier that
                # holds the turn's whole requirement is then the cheapest candidate for
                # every duty of that input, so the duties serialize behind one worker and
                # the other carriers walk to the shed for nothing. Measured 2026-09-11 and
                # reverted; see tests/test_full_economic_orders_dispatch_v2.py.
                # Prefer an already-equipped worker, then shortest completion route.
                setup = distance + (1 + manhattan(destination, job["tile"]) if job["need"] and action[0] == "PICKUP" else 0)
                candidates.append((setup, unit, destination, action))
            if not candidates:
                blocked.append({"cause": "UNASSIGNED_DUTY", "tile": list(job["tile"]),
                    "op": job["op"], "need": job["need"], "free_units": len(free)})
                continue
            distance, unit, destination, op = min(candidates)
            actual = op if v.unit_pos(unit) == destination else [step_toward(v.unit_pos(unit), destination, v)]
            actions[unit] = actual
            free.remove(unit)
            claimed.add(job["tile"])
            if actual[0] == "PICKUP":
                available[actual[1]] -= actual[2]
            elif actual[0] == "PLANT":
                seeds[actual[1]] -= 1
            routes.append({"unit": unit, "target": list(job["tile"]), "intent": job["op"],
                           "issued": actual, "minimum_travel_and_setup": distance})
        for unit in sorted(free):
            if sum(v.unit_inv(unit).values()):
                actions[unit] = _haul_action(v, v.unit_pos(unit), v.unit_inv(unit))
        projected = _project_unit_market_ledger(v, actions, shed_capacity=100, rewrite_unsafe_unloads=True)
        return projected, routes, blocked

    def _market(self, v, projected, jobs):
        reserve = self.rules.get("CASH_RESERVE", {}).get("command", {}).get("amount", 0)
        ledger = _MarketLedger(v.money - reserve, v.market, projected["shed"], v.farm.get("hires_today", 0))
        stock, seeds = dict(projected["shed"]), dict(projected["seeds"])
        carried = {item: sum(inv.get(item, 0) for inv in projected["inventories"]) for item in [*PRODUCTS, *ANIMALS]}
        candidates, blocked, receipts = [], [], []
        # One candidate per economic rule; queue priority is explicit policy input.
        for key, rule in self.rules.items():
            c = rule["command"]
            if not _active(c, v.step) or c["op"] == "CASH_RESERVE":
                continue
            candidates.append((c["priority"], key, c, rule))
        # Production requisitions buy precisely the currently necessary inputs,
        # not a hidden two-day reserve or lifetime stockpile. STOCK is independent.
        demands = {}
        for job in jobs:
            op = job["op"][0]
            if op == "PLANT":
                kind, item = "SEED", job["op"][1]
            elif job["need"]:
                kind, item = ("ANIMAL" if job["need"] in ANIMALS else "PRODUCT"), job["need"]
            else:
                continue
            # Current unit effects may already have fulfilled this prerequisite.
            after = projected["tiles"].get(job["tile"], v.tile(*job["tile"]))
            if ((op == "PLANT" and isinstance(after, dict) and after.get("crop") == item) or
                (op == "PLACE" and isinstance(after, dict) and after.get("animal") == item) or
                (op == "FEED" and isinstance(after, dict) and after.get("fed_today")) or
                (op == "FERTILIZE" and isinstance(after, dict) and after.get("fertilized_until_day", -1) >= v.day)):
                continue
            key = (kind, item)
            prior = demands.get(key, (0, 255))
            demands[key] = (prior[0] + 1, min(prior[1], job["priority"][1]))
        # REQUISITION RESERVE. A SALE rule and a production requisition for the same
        # product otherwise form a round trip: whichever the program ranks first --
        # and SALE is normally ranked first, because it is what pays for everything
        # else -- empties the shed, the requisition re-buys the same units next turn,
        # and the farm pays the market's buy/sell spread on feed that is never
        # delivered. What today's outstanding jobs still require, over and above what
        # units already carry, is therefore withheld from every SALE of that product
        # in addition to the rule's own `reserve`. This is a projection of THIS turn's
        # stated requirement, not a hidden strategic stockpile: it is zero as soon as
        # the jobs are satisfied, and it never withholds more than the jobs ask for.
        requisition_reserve = {}
        for (kind, item), (target, _priority) in demands.items():
            if kind == "SEED":
                continue  # seeds are a separate private pool, never sellable stock
            requisition_reserve[item] = max(0, target - carried.get(item, 0))
        for (kind, item), (target, priority) in demands.items():
            candidates.append((priority, f"PREREQUISITE:{kind}:{item}",
                {"op": "STOCK", "kind": kind, "item": item, "target": target, "price_below": math.inf}, {}))
        queue, buys_by_item, sold = [], set(), set()
        land_count = len(v.farm.get("unlocked_quadrants", ["NW"])) - 1
        for _, key, c, rule in sorted(candidates, key=lambda row: row[0]):
            op = c["op"]
            if len(queue) >= MARKET_ORDER_SLOTS:
                blocked.append({"cause": "MARKET_ORDER_SLOTS", "rule": key})
                continue
            if op == "SALE":
                item = c["product"]
                withheld = c["reserve"] + requisition_reserve.get(item, 0)
                quantity = min(c["quantity"], max(0, stock.get(item, 0) - withheld))
                if any(kind != "SEED" and bought_item == item for kind, bought_item in buys_by_item) or v.prices.get(item, 0) <= c["price_above"] or quantity <= 0:
                    if (quantity <= 0 and requisition_reserve.get(item, 0)
                            and stock.get(item, 0) > c["reserve"]):
                        blocked.append({"cause": "ACQUISITION_SALE_SERIALIZED",
                                        "rule": key, "item": item})
                    continue
                got = ledger.commit(["SELL", item, quantity])
                queue.append(["SELL", item, got])
                stock[item] -= got
                sold.add(item)
            elif op == "HANDS":
                batch = HIRE_BATCH_LIMIT if len(candidates) > 1 else MARKET_ORDER_SLOTS
                hired = 0
                while ledger.hires_today < c["target"] and len(queue) < 10 and hired < batch:
                    if not ledger.commit(["HIRE"]):
                        blocked.append({"cause": "HIRE_CASH", "rule": key, "next_hire_cost": hire_cost(ledger.hires_today)})
                        break
                    queue.append(["HIRE"])
                    hired += 1
                if hired >= batch and ledger.hires_today < c["target"]:
                    blocked.append({"cause": "MARKET_ORDER_SLOTS", "rule": key})
            elif op == "LAND":
                while land_count < c["target"] and len(queue) < 10:
                    if not ledger.commit(["BUY_LAND", LAND_PRICES[land_count]]):
                        blocked.append({"cause": "LAND_CASH", "rule": key, "cost": LAND_PRICES[land_count]})
                        break
                    queue.append(["BUY_LAND"])
                    land_count += 1
            elif op in ("STOCK", "BUY"):
                kind, item = c["kind"], c["item"]
                # A cumulative BUY is reconciled from independent own holdings;
                # do not net it with same-item sales or another acquisition rule.
                if (kind != "SEED" and item in sold) or (kind, item) in buys_by_item:
                    blocked.append({"cause": "ACQUISITION_SALE_SERIALIZED", "rule": key})
                    continue
                quote = (CROPS[item]["seed"] if kind == "SEED" else
                         ANIMALS[item]["cost"] if kind == "ANIMAL" else v.prices.get(item, 0))
                if quote >= c["price_below"]:
                    continue
                before = seeds.get(item, 0) if kind == "SEED" else stock.get(item, 0) + carried.get(item, 0)
                want = max(0, c["quantity"] - rule["acquired"]) if op == "BUY" else max(0, c["target"] - before)
                if not want:
                    continue
                if op == "BUY" and kind != "SEED" and v.hour == 23:
                    blocked.append({"cause": "ACQUISITION_RESET_ACCOUNTING_DEFERRED", "rule": key})
                    continue
                engine_op = {"SEED": "BUY_SEED", "ANIMAL": "BUY_ANIMAL", "PRODUCT": "BUY_PRODUCT"}[kind]
                # Flat-price seed quantities may be very large. Bound by cash
                # before calling the existing unitwise ledger, not a strategy cap.
                if kind == "SEED":
                    want = max(0, min(want, int(ledger.cash // CROPS[item]["seed"])))
                else:
                    want = min(want, ledger.room)
                got = ledger.commit([engine_op, item, want]) if want else 0
                if not got:
                    blocked.append({"cause": "CASH_OR_SHED_CAPACITY", "rule": key, "item": item})
                    continue
                queue.append([engine_op, item, got])
                buys_by_item.add((kind, item))
                if kind == "SEED":
                    seeds[item] = seeds.get(item, 0) + got
                else:
                    stock[item] = stock.get(item, 0) + got
                if op == "BUY":
                    receipts.append({"rule_key": key, "kind": kind, "item": item,
                                     "before": before, "requested": got})
        return queue, blocked, receipts

    def act(self, observation, configuration=None):
        if configuration is not None:
            for key, expected in (("boardSize", 10), ("turnsPerDay", 24), ("episodeSteps", 720),
                                  ("shedCapacity", 100), ("maxMarketOrdersPerTurn", 10), ("farmHandCostMult", 1)):
                if configuration.get(key, expected) != expected:
                    raise ValueError("unsupported engine configuration: " + key)
            if configuration.get("marketParams"):
                raise ValueError("custom market parameters need explicit projection binding")
        v = View(observation)
        if self.pending and v.step == self.pending["step"]:
            if self.pending.get("settled_observation_sha256") or digest(observation) != self.pending["observation_sha256"]:
                raise ValueError("same issued step changed or already settled")
            return copy.deepcopy(self.pending["action"])
        self._reconcile(v)
        jobs, blocked = self._jobs(v)
        projected, routes, dispatch_blocked = self._units(v, jobs)
        market, market_blocked, buys = self._market(v, projected, jobs)
        actions = projected["actions"]
        action = {"farmer": actions[0], "hands": actions[1:], "market": market}
        harvests, operations = [], []
        pre_tiles = {_key(p): copy.deepcopy(v.tile(*p)) for p in
                     (_scope_sites(o["command"]["scope"])[0] for o in self.orders.values())}
        for unit, op in enumerate(actions):
            pos = v.unit_pos(unit)
            k, tile = _key(pos), v.tile(*pos)
            if k not in self.orders:
                continue
            operations.append({"key": k, "op": op[0]})
            after = projected["tiles"].get(pos, tile)
            if op == ["FERTILIZE"] and isinstance(after, dict) and after.get("fertilized_until_day", -1) >= v.day:
                self.orders[k]["fertilized_day"] = v.day
            if op == ["HARVEST"] and isinstance(tile, dict) and tile.get("yield_units", 0) > 0:
                item = tile.get("crop") or ANIMALS.get(tile.get("animal"), {}).get("product")
                units = projected["inventories"][unit].get(item, 0) - v.unit_inv(unit).get(item, 0)
                if units > 0:
                    harvests.append({"key": k, "tile": list(pos), "order_id": self.orders[k]["id"], "units": units})
        self.last_report = {"step": v.step, "available_units": v.n_units, "pending_jobs": len(jobs),
            "assigned_jobs": len(routes), "unassigned_jobs": len(dispatch_blocked),
            "market_slots_used": len(market), "shed_used": v.shed_used(),
            "blocked": blocked + dispatch_blocked + market_blocked, "routes": routes,
            "due_water_feed": sum(j["op"][0] in ("WATER", "FEED") for j in jobs),
            "optimality": "NOT_CLAIMED_GREEDY_MECHANICAL_DISPATCH",
            "market_forecast": "OWN_QUEUE_ONLY_OPPONENT_CAN_CHANGE_FILLS",
            "sale_condition": "OBSERVED_TRIGGER_NOT_LIMIT_PRICE_GUARANTEE"}
        self.pending = {"step": v.step, "observation_sha256": digest(observation),
            "action": copy.deepcopy(action), "harvests": harvests, "buys": buys,
            "pre_tiles": pre_tiles, "operations": operations}
        return action

    __call__ = act


class FullSeasonPlan:
    """Whole-plan publication: immutable elapsed intent + live future + receipts.

    This owns publication, not route selection or a second game engine. A sampled
    edit program deterministically materializes a FULL candidate; only publish()
    activates it. Past credit snapshots are cumulative as-of values, never sums.
    The executor's event ledger retains actual realizations after cancellation.
    """

    SCHEMA = "farm-full-season-plan-v1"
    PROPOSAL_SCHEMA = "farm-full-season-plan-proposal-v1"
    STEPS = (0, 144, 288, 432, 576)

    def __init__(self, game_id, *, publication_steps=STEPS, seed_program=None):
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("explicit game/seat plan identity required")
        if (not isinstance(publication_steps, (list, tuple)) or not publication_steps
                or publication_steps[0] != 0 or any(type(s) is not int or not 0 <= s <= LAST_STEP
                    or s % 24 for s in publication_steps)
                or list(publication_steps) != sorted(set(publication_steps))):
            raise ValueError("ordered explicit morning publication steps required")
        self.game_id, self.publication_steps = game_id, tuple(publication_steps)
        self.executor = ScheduledEconomicOrders()
        self.committed_step, self.last_publication_sha256 = None, None
        self.past = []
        if seed_program is not None:
            from farm_neat.economy.action_space import validate_program
            for command in validate_program(seed_program):
                self.executor.apply(command)

    @property
    def orders(self): return self.executor.orders

    @property
    def rules(self): return self.executor.rules

    @property
    def last_report(self): return self.executor.last_report

    @property
    def revision(self): return self.executor.revision

    def snapshot(self):
        return copy.deepcopy({"schema": self.SCHEMA, "game_id": self.game_id,
            "publication_steps": list(self.publication_steps), "committed_step": self.committed_step,
            "last_publication_sha256": self.last_publication_sha256, "past": self.past,
            "executor": self.executor.snapshot()})

    @classmethod
    def restore(cls, state):
        fields = {"schema", "game_id", "publication_steps", "committed_step",
                  "last_publication_sha256", "past", "executor"}
        if not isinstance(state, dict) or set(state) != fields or state["schema"] != cls.SCHEMA:
            raise ValueError("closed full-season plan snapshot required")
        result = cls(state["game_id"], publication_steps=state["publication_steps"])
        result.executor = ScheduledEconomicOrders.restore(state["executor"])
        committed, previous, past = state["committed_step"], 0, state["past"]
        if committed is not None and (type(committed) is not int or committed not in result.publication_steps
                or committed > result.executor.stage_step):
            raise ValueError("full-plan publication clock")
        if not isinstance(past, list) or len(past) >= len(result.publication_steps):
            raise ValueError("bounded elapsed plan segments required")
        for segment in past:
            if (not isinstance(segment, dict) or set(segment) != {"from_step", "until_step", "as_of_step", "state"}
                    or any(type(segment[k]) is not int for k in ("from_step", "until_step", "as_of_step"))
                    or segment["from_step"] != previous or segment["until_step"] < previous
                    or segment["as_of_step"] != segment["until_step"] + 1
                    or segment["as_of_step"] not in result.publication_steps):
                raise ValueError("contiguous immutable elapsed plan prefix required")
            historic = segment["state"]
            if not isinstance(historic, dict) or set(historic) != {"order_stages", "rule_stages", "revision"}:
                raise ValueError("closed elapsed intentions and as-of credit required")
            skeleton = ScheduledEconomicOrders().snapshot()
            skeleton.update(copy.deepcopy(historic))
            ScheduledEconomicOrders.restore(skeleton)
            previous = segment["until_step"] + 1
        if previous != (committed or 0) or (committed is None and past):
            raise ValueError("elapsed prefix must end at the last publication")
        last = state["last_publication_sha256"]
        if (committed is None) != (last is None) or (last is not None and (
                not isinstance(last, str) or len(last) != 64 or any(c not in "0123456789abcdef" for c in last))):
            raise ValueError("accepted publication digest required")
        result.committed_step, result.last_publication_sha256 = committed, last
        result.past = copy.deepcopy(past)
        return result

    def observe(self, observation, *, return_snapshot=True):
        self.executor.observe(observation, return_snapshot=False)
        return self.snapshot() if return_snapshot else None

    def act(self, observation, configuration=None):
        if self.committed_step is None:
            raise ValueError("publish the opening plan before execution; seed/draft is not active")
        return self.executor.act(observation, configuration)

    __call__ = act

    def _ready(self, observation):
        step = self.executor._observation_step(observation)
        if step != self.executor.stage_step or self.executor.player != observation["player"]:
            raise ValueError("observe the actual state before planning")
        pending = self.executor.pending
        if pending is not None and pending.get("settled_observation_sha256") != digest(observation):
            raise ValueError("settle the actual successor before planning")
        if self.committed_step is None and step != 0:
            raise ValueError("a full-season plan cannot invent an unobserved opening")
        return step

    def propose(self, observation, program):
        """Pure draft; unchanged live object, world, RNG and physical clock.

        Backdated starts in a sampled program apply to the REMAINING interval
        only; the frozen prefix keeps the previously accepted economic terms.
        Past-only windows have no future effect and are explicitly refused.
        """
        from farm_neat.economy.action_space import validate_program
        from hashlib import sha256
        from pathlib import Path
        commands = validate_program(program)
        step = self._ready(observation)
        if self.committed_step is not None and step <= self.committed_step:
            raise ValueError("one publication per eligible morning; current plan already committed")
        staged, rejected = copy.deepcopy(self.executor), []
        for index, command in enumerate(commands):
            c = command.get("replacement", command)
            reason = None
            if "until" in c and _end(c) < step:
                reason = "PAST_ONLY_WINDOW: no remaining authorized execution"
            else:
                try:
                    staged.apply(command)
                except ValueError as exc:
                    if not str(exc).startswith("CONFLICTING_TILE_ORDER:"):
                        raise
                    reason = str(exc)
            if reason:
                rejected.append({"command": copy.deepcopy(command), "command_index": index,
                    "reason": reason, "transaction": "COMMAND_REFUSED_REMAINDER_PRESERVED"})
        past = copy.deepcopy(self.past)
        if self.committed_step is not None:
            historic = {"revision": self.revision}
            for group in ("order_stages", "rule_stages"):
                historic[group] = {key: [copy.deepcopy(row) for row in rows
                    if self.executor._start(row) < step and self.executor._end(row) >= self.committed_step]
                    for key, rows in getattr(self.executor, group).items()}
                historic[group] = {key: rows for key, rows in historic[group].items() if rows}
            past.append({"from_step": self.committed_step, "until_step": step-1,
                         "as_of_step": step, "state": historic})
        result = {"schema": self.PROPOSAL_SCHEMA, "game_id": self.game_id,
            "executor_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "base_sha256": digest(self.snapshot()), "base_revision": self.revision,
            "observation_sha256": digest(observation), "step": step,
            "program": commands, "rejected": rejected,
            "full_plan": {"horizon": [0, LAST_STEP], "past": past,
                "future_from": step, "executor": staged.snapshot()}}
        result["content_sha256"] = digest(result)
        return result

    def publish(self, observation, proposal):
        """Recompute/compare the whole candidate, then return its accepted state.

        A preview's own score cannot accept it. Exact same-morning retries return
        the current state, including an already-issued action, without resetting it.
        """
        if not isinstance(proposal, dict):
            raise ValueError("typed complete plan proposal required")
        payload = {k: v for k, v in proposal.items() if k != "content_sha256"}
        sha = proposal.get("content_sha256")
        if sha != digest(payload) or proposal.get("schema") != self.PROPOSAL_SCHEMA:
            raise ValueError("complete plan proposal digest/schema required")
        step = self.executor._observation_step(observation)
        if step not in self.publication_steps or step != self.executor.stage_step or proposal.get("step") != step:
            raise ValueError("full-plan publication only at registered current mornings")
        if proposal.get("game_id") != self.game_id or proposal.get("observation_sha256") != digest(observation):
            raise ValueError("stale or foreign game/seat/observation proposal")
        if sha == self.last_publication_sha256:
            return copy.deepcopy(self)
        expected = self.propose(observation, proposal.get("program"))
        if proposal != expected:
            raise ValueError("stale base, changed history/credit or edited complete candidate")
        result = copy.deepcopy(self)
        result.executor = ScheduledEconomicOrders.restore(proposal["full_plan"]["executor"])
        result.past = copy.deepcopy(proposal["full_plan"]["past"])
        result.committed_step, result.last_publication_sha256 = step, sha
        return result


class ScheduledEconomicOrders(EconomicOrders):
    """Opt-in temporal composition over the unchanged mechanical dispatcher.

    Disjoint windows coexist. A setter at an existing start updates that stage
    without resetting credit; overlapping production needs explicit REPLACE.
    REPLACE retires intersecting intentions at its start, not at publication.
    CANCEL retires every stage of its named target. No automatic stage is added.
    Old EconomicOrders snapshots/semantics remain unchanged and independently
    resumable; this snapshot must not be passed off as the old representation.
    """

    SCHEMA = "farm-full-economic-scheduled-orders-v2"

    def __init__(self):
        super().__init__()
        self.order_stages, self.rule_stages = {}, {}
        self.stage_step = 0
        self.player = None

    @staticmethod
    def _start(row):
        return row["command"].get("start", 0)

    @staticmethod
    def _end(row):
        # CLEAR has no policy-selected time fields. A future REPLACE may bound
        # its execution, but must not manufacture an invalid sampled command.
        return row.get("stage_until", _end(row["command"]))

    @staticmethod
    def _legacy_row(row):
        """Project scheduler-only metadata away for the unchanged v1 codec."""
        return {k: copy.deepcopy(v) for k, v in row.items() if k != "stage_until"}

    def _select(self, step):
        self.stage_step = step
        self.orders = {key: row for key, stages in self.order_stages.items()
                       for row in stages if self._start(row) <= step <= self._end(row)}
        self.rules = {key: row for key, stages in self.rule_stages.items()
                      for row in stages if self._start(row) <= step <= self._end(row)}

    def _install(self, mapping, key, row, *, replace=False, production=False):
        stages = mapping.setdefault(key, [])
        c = row["command"]
        start, end = c.get("start", 0), _end(c)
        same = next((old for old in stages if self._start(old) == start), None)
        if same is not None:
            old = same["command"]
            species_equal = (old["op"], old.get("crop"), old.get("animal")) == (
                c["op"], c.get("crop"), c.get("animal"))
            if production and not replace and not species_equal:
                raise ValueError("CONFLICTING_TILE_ORDER: use explicit REPLACE or CLEAR")
            # Reapplying an identical replacement cannot reset its credit or
            # request destruction again after the replacement has been built.
            if same["command"] == c and (not production or not replace or species_equal):
                return
            if not production or (species_equal and not replace):
                row = copy.deepcopy(same)
                row["command"] = c
        kept = []
        for old in stages:
            if old is same:
                continue
            old_start, old_end = self._start(old), self._end(old)
            if end < old_start or start > old_end:
                kept.append(old)
            elif replace:
                if old_start < start:
                    # Explicit replacement shortens the previous intention;
                    # its stable identity and already-realized credit survive.
                    previous = copy.deepcopy(old)
                    if previous["command"]["op"] == "CLEAR":
                        previous["stage_until"] = start - 1
                    else:
                        previous["command"]["until"] = start - 1
                    kept.append(previous)
            else:
                raise ValueError("CONFLICTING_TILE_ORDER: overlapping stage windows; revise the matching start or REPLACE")
        mapping[key] = sorted([*kept, row], key=self._start)

    def apply(self, command):
        c = validate_command(command)
        if self.pending and not self.pending.get("settled_observation_sha256"):
            raise ValueError("observe the successor before editing an issued action")
        if c["op"] == "STOP":
            return
        before = digest([self.order_stages, self.rule_stages])
        # Validate/compose in isolation: a multi-tile conflict must not leave a
        # partly installed region or shortened earlier stage behind.
        orders, rules = copy.deepcopy(self.order_stages), copy.deepcopy(self.rule_stages)
        op = c["op"]
        if op in ("SET_CROP", "SET_ANIMAL", "REPLACE", "CLEAR"):
            replacing = op == "REPLACE"
            scope = c["scope"]
            if replacing:
                c = copy.deepcopy(c["replacement"])
            positions = _scope_sites(scope)
            group = _scope_key(scope)
            start = c.get("start", 0)
            targets = {_key(p) for p in positions}
            # Region target reduction is scoped to this stage, never to all
            # future phases that happen to use the same region.
            for key, rows in list(orders.items()):
                orders[key] = [row for row in rows if not (
                    row["scope_key"] == group and self._start(row) == start and key not in targets)]
                if not orders[key]:
                    del orders[key]
            for position in positions:
                key = _key(position)
                at_site = {**c, "scope": {"kind": "TILE", "x": position[0], "y": position[1]}}
                row = {"command": at_site, "scope_key": group,
                    "id": f"{self.revision + 1}:{key}@{start}", "completed_cycles": 0,
                    "successful_harvests": 0, "harvested_units": 0, "birth": None,
                    "cycle_harvests": 0, "fertilized_day": -1, "clear_before": replacing}
                if op == "CLEAR":
                    orders.pop(key, None)
                self._install(orders, key, row, replace=replacing, production=True)
        elif op == "CANCEL":
            target = c["target"]
            if target["kind"] == "PRODUCTION":
                for position in _scope_sites(target["scope"]):
                    orders.pop(_key(position), None)
            else:
                key = target["kind"]
                if key == "SALE":
                    key += ":" + target["product"]
                elif key in ("STOCK", "BUY"):
                    key += ":" + target["stock_kind"] + ":" + target["item"]
                rules.pop(key, None)
        else:
            self._install(rules, _rule_key(c), {"command": c, "acquired": 0})
        if digest([orders, rules]) != before:
            self.order_stages, self.rule_stages = orders, rules
            self.revision += 1
            self._select(self.stage_step)

    def snapshot(self):
        return copy.deepcopy({"schema": self.SCHEMA,
            "order_stages": self.order_stages, "rule_stages": self.rule_stages,
            "stage_step": self.stage_step, "player": self.player, "pending": self.pending,
            "events": self.events, "last_report": self.last_report, "revision": self.revision})

    @staticmethod
    def _validate_row(row, *, production, key):
        fields = ({"command", "id", "scope_key", "completed_cycles", "successful_harvests",
                   "harvested_units", "birth", "cycle_harvests", "fertilized_day", "clear_before"}
                  if production else {"command", "acquired"})
        if not isinstance(row, dict) or set(row) not in (fields, fields | {"stage_until"}):
            raise ValueError("closed scheduled commitment/credit fields required")
        c = validate_command(row["command"])
        if "stage_until" in row and (not production or c["op"] != "CLEAR"
                or type(row["stage_until"]) is not int or not 0 <= row["stage_until"] < LAST_STEP):
            raise ValueError("scheduler cutoff belongs only to shortened CLEAR stages")
        if not production:
            if c["op"] not in ("SALE", "STOCK", "BUY", "LAND", "HANDS", "CASH_RESERVE"):
                raise ValueError("persistent scheduled rule required")
            return
        if type(row["clear_before"]) is not bool:
            raise ValueError("explicit clear_before boolean required")
        day = row["fertilized_day"]
        if type(day) is not int or not -1 <= day <= LAST_STEP // 24:
            raise ValueError("scheduled fertilization day")
        scope_key = row["scope_key"]
        regions = {"REGION:" + r for r in ("NW", "NE", "SW", "SE", "ALL")}
        if not isinstance(scope_key, str) or scope_key not in {"TILE:" + key, *regions}:
            raise ValueError("scheduled original scope identity")
        birth = row["birth"]
        if birth is not None and (not isinstance(birth, dict) or set(birth) != {"crop", "planted_day"}
                or c["op"] != "SET_CROP" or birth["crop"] != c["crop"]
                or type(birth["planted_day"]) is not int or not 0 <= birth["planted_day"] <= LAST_STEP // 24):
            raise ValueError("scheduled crop birth identity")

    @staticmethod
    def _validate_pending(pending, step):
        if pending is None:
            return
        fields = {"step", "observation_sha256", "action", "harvests", "buys", "pre_tiles", "operations"}
        if not isinstance(pending, dict) or set(pending) not in (fields, fields | {"settled_observation_sha256"}):
            raise ValueError("closed scheduled pending receipt required")
        settled = "settled_observation_sha256" in pending
        issued = pending["step"]
        if type(issued) is not int or not 0 <= issued <= LAST_STEP or issued + int(settled) != step:
            raise ValueError("scheduled receipt and execution clock disagree")
        for key in ("observation_sha256", "settled_observation_sha256"):
            if key in pending:
                value = pending[key]
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("scheduled observation digest")
        if any(not isinstance(pending[k], list) for k in ("harvests", "buys", "operations")):
            raise ValueError("scheduled receipt arrays required")
        if not isinstance(pending["pre_tiles"], dict):
            raise ValueError("scheduled pre-action tile map required")
        action = pending["action"]
        if not isinstance(action, dict) or set(action) != {"farmer", "hands", "market"} or any(
                not isinstance(action[k], list) for k in action):
            raise ValueError("scheduled issued action fields")

    @classmethod
    def restore(cls, state):
        expected = {"schema", "order_stages", "rule_stages", "stage_step", "player", "pending",
                    "events", "last_report", "revision"}
        if not isinstance(state, dict) or set(state) != expected or state["schema"] != cls.SCHEMA:
            raise ValueError("explicit scheduled-economic snapshot required")
        if type(state["stage_step"]) is not int or not 0 <= state["stage_step"] <= LAST_STEP + 1:
            raise ValueError("scheduled snapshot clock")
        if type(state["revision"]) is not int or state["revision"] < 0:
            raise ValueError("scheduled snapshot revision")
        player = state["player"]
        if (player is not None and (type(player) is not int or player not in (0, 1))) or (
                player is None and (state["pending"] is not None or state["stage_step"] != 0)):
            raise ValueError("scheduled snapshot seat binding required")
        cls._validate_pending(state["pending"], state["stage_step"])
        if not isinstance(state["events"], list) or not isinstance(state["last_report"], dict):
            raise ValueError("scheduled event list and execution report required")
        result = cls()
        ids = set()
        for group, target in (("order_stages", "orders"), ("rule_stages", "rules")):
            mapping = state[group]
            if not isinstance(mapping, dict):
                raise ValueError("scheduled snapshot stage map")
            for key, stages in mapping.items():
                if not isinstance(key, str) or not isinstance(stages, list) or not stages:
                    raise ValueError("nonempty scheduled stages required")
                previous_end = -1
                for row in stages:
                    cls._validate_row(row, production=target == "orders", key=key)
                    legacy = EconomicOrders().snapshot()
                    legacy[target] = {key: cls._legacy_row(row)}
                    EconomicOrders.restore(legacy)
                    start = cls._start(row)
                    if start <= previous_end:
                        raise ValueError("scheduled stages must be ordered and disjoint")
                    previous_end = cls._end(row)
                    if target == "orders":
                        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
                            raise ValueError("unique scheduled commitment identity required")
                        ids.add(row["id"])
        for field in expected - {"schema"}:
            setattr(result, field, copy.deepcopy(state[field]))
        result._select(result.stage_step)
        return result

    def model_stages(self):
        """Readable complete current/future/retained-credit input for the adapter.

        No future stage may be hidden by using only the active dispatcher maps.
        The existing v1 observation packer deliberately rejects this v2 snapshot.
        """
        return copy.deepcopy([{"kind": group, "key": key, "stage": i,
            "active": self._start(row) <= self.stage_step <= self._end(row), "state": row}
            for group, mapping in (("orders", self.order_stages), ("rules", self.rule_stages))
            for key, rows in mapping.items() for i, row in enumerate(rows)])

    def _jobs(self, v):
        jobs, blocked = super()._jobs(v)
        kept = []
        for job in jobs:
            if job["op"][0] == "PLANT":
                row = self.orders[job["key"]]
                c = row["command"]
                # Optimistic bound under this dispatcher's one-job-per-tile
                # cadence: market-bought seeds act NEXT turn; requested water
                # needs another turn that same day. No price/yield/profit test.
                earliest = v.step + int(not v.seeds.get(c["crop"], 0))
                water = c["watering"] != "OFF"
                if water and earliest % 24 == 23:
                    earliest += 1
                water_step = earliest + 1
                service = not water or any(
                    self._start(stage) <= water_step <= self._end(stage)
                    and stage["command"]["op"] == "SET_CROP"
                    and stage["command"]["crop"] == c["crop"]
                    and stage["command"]["watering"] != "OFF"
                    and not stage["clear_before"]
                    and (stage["command"]["cycles"] is None
                         or stage["completed_cycles"] < stage["command"]["cycles"])
                    for stage in self.order_stages[job["key"]])
                if earliest > self._end(row) or not service:
                    blocked.append({"cause": "UNASSIGNED_DUTY", "tile": list(job["tile"]),
                        "op": job["op"], "need": None, "free_units": v.n_units,
                        "repair": "plant/service window has no remaining dispatch slot; revise the window"})
                    continue
            kept.append(job)
        # Removing the impossible job also removes its automatic seed purchase.
        # Explicit BUY/STOCK rules and already-established crops are unchanged.
        return kept, blocked

    def _observation_step(self, observation):
        step, day, hour = (observation.get(k) for k in ("step", "day", "hour"))
        if (any(type(v) is not int for v in (step, day, hour))
                or not 0 <= step <= LAST_STEP + 1 or day != step // 24 or hour != step % 24):
            raise ValueError("consistent scheduled observation clock required")
        player = observation.get("player")
        if type(player) is not int or player not in (0, 1) or self.player not in (None, player):
            raise ValueError("scheduled observation belongs to a different or invalid seat")
        return step

    def observe(self, observation, *, return_snapshot=True):
        step = self._observation_step(observation)
        self._reconcile(View(observation))
        self.player = observation["player"]
        self._select(step)
        return self.snapshot() if return_snapshot else None

    def act(self, observation, configuration=None):
        step = self._observation_step(observation)
        if step > LAST_STEP:
            raise ValueError("no scheduled action after terminal or outside the game clock")
        if not self.pending or step != self.pending["step"]:
            self.observe(observation, return_snapshot=False)
        return super().act(observation, configuration)

    __call__ = act
