"""Shop plans with small, observation-based repairs. Python standard library only.

The 13 complete action tapes live in actions.json. This file contains every rule:
choose a plan after two shops, delay weed-blocked work within the current day,
bring some planned sales forward one turn, and liquidate on the final turn.

Sell timing and shed projection follow aurax7's public Reactive Router:
https://www.kaggle.com/code/aurax7/kaggriculture-reactive-router
The shop-pair routes and worker-local, same-day queues were developed here.
"""

import copy
import json
from collections import deque
from pathlib import Path

TURNS_PER_DAY = 24
ROUTE_STEP = 144
FINAL_PLAN_STEP = 648
LAST_STEP = 718
SHED_CAPACITY = 100
MAX_ORDERS = 10
PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
WEED_BLOCKED_WORK = {"PLANT", "BUILD_COOP", "BUILD_PASTURE"}
ANIMALS = {"GOOSE", "COW", "SHEEP"}

# Keys are the first two shops in their observed order; values index actions.json.
# All other pairs keep plan 0. Plan 1 is the previous yarn-market continuation.
# Plans 3..12 are the ten distinct continuations selected in the latest search.
SHOP_PLANS = {
    ("BAKERY", "YARN_STORE"): 3,
    ("BRUNCH_SPOT", "YARN_STORE"): 4,
    ("FARMERS_MARKET", "YARN_STORE"): 5,
    ("ICE_CREAM_SHOP", "YARN_STORE"): 6,
    ("PET_CAFE", "YARN_STORE"): 5,
    ("PIZZA_SHOP", "YARN_STORE"): 7,
    ("SMOOTHIE_SHOP", "YARN_STORE"): 8,
    ("YARN_STORE", "BAKERY"): 9,
    ("YARN_STORE", "BRUNCH_SPOT"): 9,
    ("YARN_STORE", "FARMERS_MARKET"): 1,
    ("YARN_STORE", "ICE_CREAM_SHOP"): 9,
    ("YARN_STORE", "PET_CAFE"): 10,
    ("YARN_STORE", "PIZZA_SHOP"): 6,
    ("YARN_STORE", "SMOOTHIE_SHOP"): 11,
    ("YARN_STORE", "YARN_STORE"): 12,
}


class FarmView:
    """Only the current own farm, private inventory, and public prices."""

    def __init__(self, observation):
        farm = observation["farms"][observation["player"]]
        private = observation["private"]
        self.tiles = farm["tiles"]
        self.positions = [farm["farmer"], *farm["hands"]]
        self.inventories = private["inventories"]
        self.shed = {item: max(0, int(qty)) for item, qty in private["shed"].items()}
        self.prices = observation["market"]["prices"]

    def inventory(self, worker):
        return self.inventories[worker] if worker < len(self.inventories) else {}

    def beside_shed(self, position):
        center = len(self.tiles) // 2
        return position[0] in (center - 1, center) and position[1] in (center - 1, center)


class DayState:
    """Per-player memory; queues expire at dawn and sales expire next turn."""

    def __init__(self):
        self.plan = 0
        self.last_step = -1
        self.day = -1
        self.queues = {}
        self.sale_due_step = -1
        self.advanced_sales = {}


def repair_weeds(action, view, state, step):
    """Insert DIG without consuming the blocked action; shift only this worker."""
    day = step // TURNS_PER_DAY
    if day != state.day:
        state.day = day
        state.queues.clear()  # Unfinished work never spills into tomorrow.

    workers = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    for worker in range(min(len(workers), len(view.positions))):
        queue = state.queues.setdefault(worker, deque())
        queue.append(list(workers[worker]))
        x, y = view.positions[worker]
        tile = view.tiles[y][x]
        blocked = (queue[0][0] in WEED_BLOCKED_WORK
                   and isinstance(tile, dict) and tile.get("kind") == "WEED")
        workers[worker] = ["DIG"] if blocked else queue.popleft()
    action["farmer"], action["hands"] = workers[0], workers[1:]


def projected_shed(action, view):
    """Estimate stock after this turn's nearby PICKUP, DROP and PLACE actions.

    Preserve worker and inventory order: limited shed capacity can make it matter.
    This is the qualified lightweight estimate, not a full game simulation.
    """
    stock = {item: view.shed.get(item, 0) for item in PRODUCTS}
    stock.update(view.shed)
    total = sum(stock.values())
    workers = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    for worker in range(min(len(workers), len(view.positions))):
        if not view.beside_shed(view.positions[worker]):
            continue
        work = workers[worker]
        operation = work[0] if work else "PASS"
        inventory = view.inventory(worker)
        if operation == "PICKUP" and len(work) >= 2 and work[1] in stock:
            quantity = max(0, int(work[2]) if len(work) >= 3 else 1)
            taken = min(stock[work[1]], quantity)
            stock[work[1]] -= taken
            total -= taken
        elif operation == "DROP":
            for item, held in inventory.items():
                added = min(max(0, int(held)), max(0, SHED_CAPACITY - total))
                if added > 0:
                    stock[item] = stock.get(item, 0) + added
                    total += added
        elif operation == "PLACE" and len(work) >= 2 and work[1] not in ANIMALS:
            item = work[1]
            quantity = max(0, int(work[2]) if len(work) >= 3 else 1)
            added = min(quantity, max(0, int(inventory.get(item, 0))),
                        max(0, SHED_CAPACITY - total))
            if added > 0:
                stock[item] = stock.get(item, 0) + added
                total += added
    return stock


def subtract_advanced_sales(action, state, step):
    """Remove quantities already requested one turn early, retaining order slots."""
    if state.sale_due_step == step:
        remaining = dict(state.advanced_sales)
        for order in action["market"]:
            if order and order[0] == "SELL" and len(order) >= 3:
                item = order[1]
                removed = min(max(0, int(order[2])), remaining.get(item, 0))
                if removed > 0:
                    order[2] = int(order[2]) - removed
                    remaining[item] -= removed
    state.advanced_sales = {}
    state.sale_due_step = -1


def advance_sales(action, view, state, tape, step):
    """Bring eligible sales from our next planned action forward by one turn."""
    next_step = step + 1
    if next_step > LAST_STEP or next_step % 72 == 0 or step % 4 == 0:
        return
    planned = {}
    for order in tape[next_step].get("market") or []:
        if order and order[0] == "SELL" and len(order) >= 3 and order[1] in PRODUCTS:
            item = order[1]
            planned[item] = planned.get(item, 0) + max(0, int(order[2]))
    already_selling = {order[1] for order in action["market"]
                       if order and order[0] == "SELL" and len(order) > 1}
    stock = projected_shed(action, view)
    for item in PRODUCTS:
        if item in ("WHEAT", "FERTILIZER") or item in already_selling:
            continue
        quantity = min(stock.get(item, 0), planned.get(item, 0))
        if quantity <= 0 or int(view.prices.get(item, 0)) < 2:
            continue
        if len(action["market"]) >= MAX_ORDERS:
            break
        action["market"].append(["SELL", item, quantity])
        state.advanced_sales[item] = quantity
    if state.advanced_sales:
        state.sale_due_step = next_step


def liquidate(view):
    """On the last turn, drop reachable inventory and sell the projected shed."""
    workers = [["DROP"] if view.beside_shed(pos) and view.inventory(worker) else ["PASS"]
               for worker, pos in enumerate(view.positions)]
    action = {"farmer": workers[0], "hands": workers[1:], "market": []}
    stock = projected_shed(action, view)
    action["market"] = [["SELL", item, stock[item]] for item in PRODUCTS if stock[item] > 0]
    action["market"].sort(key=lambda order: -int(view.prices.get(order[1], 0)) * order[2])
    return action


class Policy:
    def __init__(self, folder):
        self.tapes = json.loads((Path(folder) / "actions.json").read_text())
        if len(self.tapes) != 13 or any(len(tape) != LAST_STEP + 1 for tape in self.tapes):
            raise ValueError("Expected 13 complete, 719-turn action tapes")
        self.players = {}

    def act(self, observation):
        step, player = int(observation["step"]), int(observation["player"])
        state = self.players.get(player)
        if state is None or step <= state.last_step:
            state = self.players[player] = DayState()
        state.last_step = step

        if step == ROUTE_STEP:
            shops = observation["town"]["unlocked_shops"]
            state.plan = SHOP_PLANS.get(tuple(shops[:2]), 0)
        if step == FINAL_PLAN_STEP:
            state.plan = 2

        view = FarmView(observation)
        tape = self.tapes[state.plan]
        action = copy.deepcopy(tape[step])
        repair_weeds(action, view, state, step)
        subtract_advanced_sales(action, state, step)
        advance_sales(action, view, state, tape, step)
        action["market"] = action["market"][:MAX_ORDERS]
        return liquidate(view) if step == LAST_STEP else action


_POLICY = None


def agent(observation, configuration=None):
    global _POLICY
    if _POLICY is None:
        # Kaggle's source loader omits __file__, but retains the code filename.
        folder = Path(agent.__code__.co_filename).resolve().parent
        _POLICY = Policy(folder)
    return _POLICY.act(observation)
