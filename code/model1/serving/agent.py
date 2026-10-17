"""Primitive-BC actor: raw Kaggle observation -> actor view -> PolicyNet.act -> joint action.

``make_agent(model_path)`` returns a callable ``agent(observation, configuration)``:

* CPU only; the policy loads lazily on the first call (``load_ms`` is part of that call).
* The observation is reduced with ``farm_neat.training.bc.actor_observation`` (the same
  whitelist the replay pipeline uses) and the configuration with ``public_configuration``
  (the observation module's CONFIG keys plus ``marketParams``, as the frozen BC actor did).
* The thinking-field state is kept per (episode, seat): a new episode starts at step 0 or
  whenever the step does not advance; the state returned by ``act`` is fed back next step.
* Every returned action passes ``sanitize_action`` (plain str/int JSON, <= 10 orders).
  Any exception anywhere in the step returns PASS and counts a fallback.
* One compact stderr line per step when ``log`` is true:
  ``[pbc] step=<t> seat=<s> ms=<wall ms> fb=<0|1>`` (+ ``load_ms`` on the loading call,
  + ``err=<type>`` on a fallback).

Model specs accepted by ``load_policy``: ``stub[:seed]`` (torch-free random grammar policy
with the act() signature, for harness tests), ``random[:seed]`` (randomly initialised
``model.PolicyNet``), a ``primitive-bc-weights-v1`` file written by ``build_archive.py``,
or a training checkpoint (``model`` helper if it exposes one, else a torch dict holding a
config and a state dict).
"""
from __future__ import annotations

import copy
import inspect
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import step_search  # v9-decode: sequence-level step decoding (imported eagerly so archive tracing ships it)

HERE = Path(__file__).resolve().parent

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}
MAX_ORDERS = 10
WEIGHTS_FORMAT = "primitive-bc-weights-v1"
STATE_PARAMS = ("state", "field", "field_state", "think_state", "carry", "memory")
LOG_PREFIX = "[pbc]"


def pass_action():
    return copy.deepcopy(PASS_ACTION)


# ----------------------------------------------------------------------------- inputs
def public_configuration(configuration):
    """Declared public configuration vocabulary (seed and extras excluded)."""
    from farm_neat.policy.observation import CONFIG
    cfg = {} if configuration is None else configuration
    public = {key: copy.deepcopy(cfg.get(key, default)) for key, default in CONFIG.items()}
    public["marketParams"] = copy.deepcopy(cfg.get("marketParams", {}) or {})
    return public


def actor_view(observation):
    from farm_neat.training.bc import actor_observation
    obs = actor_observation(_plain(observation))
    inv = obs.get("private", {}).get("inventories")
    if isinstance(inv, list):  # replays (training) are key-sorted; live engine dicts are not
        obs["private"]["inventories"] = [dict(sorted(d.items())) if isinstance(d, dict) else d for d in inv]
    return obs


def _plain(value):
    """Kaggle Struct / AttrDict trees -> plain dict/list trees (no attribute magic)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# ----------------------------------------------------------------------------- output
def _int(value, what):
    if isinstance(value, bool):
        raise TypeError(f"{what}: bool is not a quantity")
    if isinstance(value, int):
        return int(value)
    # numpy / torch integer scalars
    if hasattr(value, "item") and hasattr(value, "dtype"):
        item = value.item()
        if isinstance(item, int) and not isinstance(item, bool):
            return int(item)
    raise TypeError(f"{what}: integer quantity required, got {type(value).__name__}")


QTY_ORDER_OPS = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL")


def _atom(value, what):
    return str(value) if isinstance(value, str) else _int(value, what)


def _unit(action):
    """[op] / [op, item] / [op, item, int]: the engine indexes item as a key and int()s the count."""
    if not isinstance(action, (list, tuple)) or not action or not isinstance(action[0], str):
        raise TypeError(f"unit action must be a nonempty [op, ...] list: {action!r}")
    if len(action) > 3:
        raise TypeError(f"unit action too long: {action!r}")
    out = [str(action[0])]
    if len(action) >= 2:
        if not isinstance(action[1], str):
            raise TypeError(f"unit argument must be a string: {action!r}")
        out.append(str(action[1]))
    if len(action) >= 3:
        out.append(_int(action[2], "unit quantity"))
    return out


def _order(order):
    """Engine market grammar; any other op (e.g. the codec's NOOP) is kept as an ignored slot.

    The engine skips an order it cannot parse, but the slot still counts toward the
    10-order cap and the per-index lockstep with the rival, so it is not dropped here.
    """
    if not isinstance(order, (list, tuple)) or not order or not isinstance(order[0], str):
        raise TypeError(f"market order must be a nonempty [op, ...] list: {order!r}")
    op = str(order[0])
    if op in ("HIRE", "BUY_LAND"):
        return [op]
    if op in QTY_ORDER_OPS:
        if len(order) != 3 or not isinstance(order[1], str):
            raise TypeError(f"market order must be [op, item, quantity]: {order!r}")
        return [op, str(order[1]), _int(order[2], "order quantity")]
    return [op, *(_atom(v, "order field") for v in order[1:])]


# Owner 2026-09-25: force the 4th land quadrant when conditions apply; 2026-09-26: "i'd prefer if we forced it to always buy
# LAND, at some point". Retried every step until the 4th quadrant is owned (land4b fired once, and a failed buy stayed failed).
LAND4 = {"cash": 6000, "day_lo": 9, "day_hi": 22, "require_full": False}
# Owner 2026-09-28: "force our model to buy a land from turns 10-15", "it has to choose one of those". Top 2880+ winners
# since 09-20 buy the 4th quadrant on day 10 (4,345 seats, 64% win) vs day 18 (82 seats, 24%). LAND_WINDOW replaces LAND4
# when "on": the 4th quadrant ($4,000) at the first step of days day_lo..day_hi-1 with cash >= price + reserve,
# and on day_hi as soon as cash >= price.
LAND_WINDOW = {"on": False, "day_lo": 10, "day_hi": 15, "reserve": 2000}
LAND_PRICES = (1000, 2000, 4000)


def land_rule(action, obs, cfg=None, slot=None):
    """Every step with 3 quadrants owned, cash >= LAND4["cash"] and day in [day_lo, day_hi] (and, when require_full, no free
    unlocked tile) appends BUY_LAND after the policy's market orders (replacing the last order when the list is full),
    unless the policy already ordered it. Any error leaves the policy's action unchanged."""
    try:
        seat = int(obs["player"])
        farm = obs["farms"][seat]
        day = int(obs.get("day", 0))
        owned = len(farm.get("unlocked_quadrants") or [])
        if LAND_WINDOW["on"]:
            if owned != 3 or not LAND_WINDOW["day_lo"] <= day <= LAND_WINDOW["day_hi"]:   # the 4th quadrant only
                return action
            price = LAND_PRICES[2]
            need = price if day == LAND_WINDOW["day_hi"] else price + LAND_WINDOW["reserve"]
            if float(farm.get("money", 0)) < need:
                return action
        else:
            if owned != 3 or not LAND4["day_lo"] <= day <= LAND4["day_hi"]:
                return action
            if float(farm.get("money", 0)) < LAND4["cash"]:
                return action
            if LAND4["require_full"] and any(t is None for row in farm["tiles"] for t in row):
                return action
        limit = min(MAX_ORDERS, max(1, int((cfg or {}).get("maxMarketOrdersPerTurn", MAX_ORDERS) or MAX_ORDERS)))
        market = list(action.get("market") or [])
        if any(isinstance(o, (list, tuple)) and o and o[0] == "BUY_LAND" for o in market):
            return action
        if len(market) < limit:
            market.append(["BUY_LAND"])
        else:
            market = market[:limit - 1] + [["BUY_LAND"]]
        return {**action, "market": market}
    except Exception:  # noqa: BLE001 - the rule must never break the policy's action
        return action


# Maintainer 2026-09-27 (model-2 design section 1, TERMINAL_SELL and SELL_OVERFLOW as fixed rules): at t=718 sell every shed
# product and drop purchases that cannot pay back; at dusk sell shed units that tonight's deposit would otherwise discard.
SELL_RULES = {"terminal": True, "overflow": True}
LAST_ACTION = 718
SHED_CAP = 100


def _post_unit_private(action, obs, E, raw_inventories=None):
    seat, step = int(obs["player"]), int(obs["step"])
    farm, private = copy.deepcopy(obs["farms"][seat]), copy.deepcopy(obs["private"])
    if isinstance(raw_inventories, list) and len(raw_inventories) == len(private.get("inventories") or []):
        private["inventories"] = [dict(d) for d in raw_inventories]   # engine deposit order, not the actor view's sorted keys
    cmds = [action.get("farmer", ["PASS"]), *(action.get("hands") or [])]
    demand = {}
    for c in cmds:
        if isinstance(c, list) and len(c) >= 2 and c[0] == "PLANT":
            demand[c[1]] = demand.get(c[1], 0) + 1
    blocked = {k for k, n in demand.items() if n > (private.get("seeds") or {}).get(k, 0)}
    for i, c in enumerate(cmds):
        a = ["PASS"] if isinstance(c, list) and len(c) >= 2 and c[0] == "PLANT" and c[1] in blocked else c
        E._apply_unit_action(farm, private, i, a, len(farm["tiles"]), step // 24, 24, SHED_CAP)
    return private


def _shed_after_orders(shed, market):
    shed = {k: int(v) for k, v in shed.items() if int(v) > 0}
    for o in market:
        if not isinstance(o, (list, tuple)) or len(o) < 3:
            continue
        n = int(o[2])
        if o[0] == "SELL":
            shed[o[1]] = shed.get(o[1], 0) - min(n, shed.get(o[1], 0))
        elif o[0] in ("BUY_PRODUCT", "BUY_ANIMAL"):
            shed[o[1]] = shed.get(o[1], 0) + min(n, max(0, SHED_CAP - sum(shed.values())))
    return shed


def _add_sells(market, sells, limit):
    """Raise an existing SELL of the item, else append one order per item while slots remain."""
    market = [list(o) if isinstance(o, (list, tuple)) else o for o in market]
    for item, n in sells:
        if n <= 0:
            continue
        # only a SELL after the last shed-filling BUY may be raised: a later BUY would take the room it frees
        last_buy = max((j for j, o in enumerate(market) if isinstance(o, list) and o and o[0] in ("BUY_PRODUCT", "BUY_ANIMAL")),
                       default=-1)
        hit = next((o for o in market[last_buy + 1:] if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] == item),
                   None)
        if hit is not None:
            hit[2] = int(hit[2]) + n
        elif len(market) < limit:
            market.append(["SELL", item, n])
        else:
            spare = [j for j, o in enumerate(market) if not (isinstance(o, list) and o and o[0] in ("SELL", "BUY_LAND"))]
            if spare:
                market[spare[-1]] = ["SELL", item, n]
    return market


def _quotes(obs, cfg, E):
    """Current quote and projected clearing quote P* per product (calc_features); P* = quote when unavailable."""
    market = obs.get("market") or {}
    inv = dict(market.get("inventory") or {})
    params = market.get("params") or None
    spot = {k: float(E.market_price(k, int(inv.get(k, 0)), params)) for k in E.PRODUCTS}
    proj = dict(spot)
    try:
        import calc_features as CF
        prod, _ = CF.raw(obs, cfg)
        for i, k in enumerate(CF.PRODUCTS):
            proj[k] = float(prod[i, CF._PC["projected_quote"]])
    except Exception:  # noqa: BLE001
        pass
    return spot, proj, inv, params


def _dusk_sales(lost, shed, obs, cfg, E):
    """Units to sell so tonight's deposit keeps the first k lost units (deposit order), k chosen to maximise
    sum(value of the saved units) - sum(holding premium P* - marginal quote of the units sold); the same product as a saved unit
    costs nothing (the stock is unchanged). Owner/Opus review 2026-09-27: compare current and projected quotes, not base prices."""
    spot, proj, inv, params = _quotes(obs, cfg, E)
    left = {k: n for k, n in shed.items() if k in E.PRODUCTS and n > 0}
    sold, gain, best, best_k, plan = {}, 0.0, 0.0, 0, []
    for item in lost:
        v = min(spot.get(item, 0.0), proj.get(item, spot.get(item, 0.0))) if item in spot else float(E.ANIMALS.get(item, {}).get("cost", 0))
        if left.get(item, 0) > 0:
            pick, cost = item, 0.0
        else:
            pick, cost = None, None
            for k, n in left.items():
                if n <= 0:
                    continue
                q = float(E.market_price(k, int(inv.get(k, 0)) + sold.get(k, 0), params))
                c = max(0.0, proj.get(k, q) - q)
                if cost is None or (c, -q) < (cost, -qk):
                    pick, cost, qk = k, c, q
        if pick is None:
            break
        left[pick] -= 1
        sold[pick] = sold.get(pick, 0) + 1
        plan.append(pick)
        gain += v - cost
        if gain > best:
            best, best_k = gain, len(plan)
    out = {}
    for k in plan[:best_k]:
        out[k] = out.get(k, 0) + 1
    return out


def sell_rule(action, obs, cfg=None, raw_observation=None):
    """Any error leaves the action unchanged."""
    try:
        step = int(obs["step"])
        terminal = SELL_RULES["terminal"] and step == LAST_ACTION
        dusk = SELL_RULES["overflow"] and step % 24 == 23 and step < LAST_ACTION
        if not (terminal or dusk):
            return action
        import prefix
        E = prefix.E
        limit = min(MAX_ORDERS, max(1, int((cfg or {}).get("maxMarketOrdersPerTurn", MAX_ORDERS) or MAX_ORDERS)))
        raw_inv = None
        if raw_observation is not None:
            raw_inv = _plain((raw_observation.get("private") or {}).get("inventories"))
        private = _post_unit_private(action, obs, E, raw_inv)
        products = tuple(E.PRODUCTS)
        if terminal:
            shed = {k: int(v) for k, v in private["shed"].items() if int(v) > 0 and k in products}
            market, have = [], set()
            for o in action.get("market") or []:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and shed.get(o[1], 0) > 0 and o[1] not in have:
                    market.append(["SELL", o[1], shed[o[1]]])
                    have.add(o[1])
            rest = sorted((k for k in shed if k not in have),
                          key=lambda k: -E.MARKET_PARAMS[k]["base"] * shed[k])
            market += [["SELL", k, shed[k]] for k in rest][:max(0, limit - len(market))]
            return {**action, "market": market}
        market = list(action.get("market") or [])
        shed = _shed_after_orders(private["shed"], market)
        room = max(0, SHED_CAP - sum(shed.values()))
        flat = [item for inv in private["inventories"] for item, n in inv.items() for _ in range(max(0, int(n)))]
        lost = flat[room:]
        if not lost:
            return action

        sold = _dusk_sales(lost, shed, obs, cfg, E)
        if not sold:
            return action
        order = sorted(sold.items(), key=lambda kv: -kv[1])
        return {**action, "market": _add_sells(market, order, limit)}
    except Exception:  # noqa: BLE001 - the rule must never break the policy's action
        return action


# Maintainer 2026-09-27, owner: "make model 2 prevent overflow earlier in the day". When the units already carry more than the
# shed can take tonight and the shed holds too few products to sell, the most-loaded unit that can still reach a shed-access tile
# by dusk walks there at the latest feasible step, drops its load, and the same step's market sells the overflow (the market
# runs after the units). Stateless: recomputed every step from the observation and model 1's proposal.
GUARD = {"on": False, "slack": 1, "keep": ("WHEAT", "FERTILIZER"), "from_hour": 20}   # off in A1: never fired in 10 pinned pairs
WORK_OPS = ("FEED", "WATER", "HARVEST", "CARE", "PLANT", "FERTILIZE", "COLLECT_FERTILIZER", "DIG", "BUILD_COOP", "BUILD_PASTURE")


def _unit_cmds(action):
    return [action.get("farmer", ["PASS"]), *(action.get("hands") or [])]


def _with_unit(action, idx, cmd):
    if idx == 0:
        return {**action, "farmer": cmd}
    hands = list(action.get("hands") or [])
    hands += [["PASS"] for _ in range(idx - len(hands))]   # omitted suffix commands are implicit PASS (PRO-RT-DEPLOY-0927 #5)
    hands[idx - 1] = cmd
    return {**action, "hands": hands}


def overflow_guard(action, obs, cfg=None, raw_observation=None, slot=None):
    """Guard v3 (Opus review 2026-09-27, FEEDBACK-OPUS.md section 6): from hour 20, when even selling every shed product at dusk
    cannot keep tonight's deposit within the shed (R = carried - WHEAT still needed for today's feeding + non-product shed - 100 > 0),
    movers (units whose proposal is a move or PASS) deposit at a shed-access tile, one PLACE per step, surplus WHEAT included;
    farther movers walk there at the latest feasible step, as many as cover R, committed until they arrive. No sale here: the
    hour-23 sell rule sells what was deposited. Any error leaves the action unchanged."""
    try:
        if not GUARD["on"]:
            return action
        step = int(obs["step"])
        hour, to_dusk = step % 24, 23 - step % 24
        if step >= 696 or hour < GUARD["from_hour"]:
            return action
        import prefix
        E = prefix.E
        seat = int(obs["player"])
        farm = obs["farms"][seat]
        board = len(farm["tiles"])
        access = [tuple(p) for p in E._shed_access_tiles(board)]
        raw_inv = None
        if raw_observation is not None:
            raw_inv = _plain((raw_observation.get("private") or {}).get("inventories"))
        market = list(action.get("market") or [])
        private = _post_unit_private(action, obs, E, raw_inv)
        shed = _shed_after_orders(private["shed"], market)
        carried = sum(max(0, int(n)) for inv in private["inventories"] for n in inv.values())
        unfed = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and "animal" in t and not t.get("fed_today"))
        wheat = sum(max(0, int(inv.get("WHEAT", 0))) for inv in private["inventories"])
        fixed = sum(n for k, n in shed.items() if k not in E.PRODUCTS)
        residual = carried - min(unfed, wheat) + fixed - SHED_CAP
        day = step // 24
        memo = slot.get("guard") if isinstance(slot, dict) else None
        if not memo or memo.get("day") != day:
            memo = {"day": day, "units": []}
        if residual <= 0:
            if isinstance(slot, dict):
                slot["guard"] = {"day": day, "units": []}
            return action
        positions = [tuple(farm["farmer"])] + [tuple(h) for h in farm.get("hands") or []]
        start_inv = raw_inv if isinstance(raw_inv, list) else (obs.get("private") or {}).get("inventories") or []
        invs = [start_inv[i] if i < len(start_inv) and isinstance(start_inv[i], dict) else {} for i in range(len(positions))]
        cmds = _unit_cmds(action)
        mover = [i < len(cmds) and isinstance(cmds[i], list) and bool(cmds[i]) and cmds[i][0] in ("PASS", "NORTH", "SOUTH", "EAST", "WEST")
                 for i in range(len(positions))]
        dist = [min(abs(p[0] - a[0]) + abs(p[1] - a[1]) for a in access) for p in positions]
        reserve = min(unfed, wheat)

        def depositable(i):
            inv = {k: int(n) for k, n in invs[i].items() if k in E.PRODUCTS and int(n) > 0}
            return inv

        new, covered, committed = action, 0, []
        order = sorted(range(len(positions)), key=lambda i: (i not in memo["units"], dist[i], -sum(depositable(i).values()), i))
        for i in order:
            if covered >= residual:
                break
            if not mover[i] or dist[i] > to_dusk:
                continue
            inv = depositable(i)
            if "WHEAT" in inv:
                keep = min(inv["WHEAT"], reserve)
                reserve -= keep
                inv["WHEAT"] -= keep
            load = sum(inv.values())
            if load <= 0:
                continue
            if dist[i] == 0:
                k = max(inv, key=inv.get)
                new = _with_unit(new, i, ["PLACE", k, inv[k]])       # PLACE keeps what does not fit
                covered += load
                continue
            if i not in memo["units"] and dist[i] + GUARD["slack"] < to_dusk:
                continue                    # not yet the latest feasible step for this unit
            p = positions[i]
            a = min(access, key=lambda a: (abs(p[0] - a[0]) + abs(p[1] - a[1]), a))
            dx, dy = a[0] - p[0], a[1] - p[1]
            move = (("EAST" if dx > 0 else "WEST") if abs(dx) >= abs(dy) and dx else ("SOUTH" if dy > 0 else "NORTH"))
            new = _with_unit(new, i, [move])
            committed.append(i)
            covered += load
        if isinstance(slot, dict):
            slot["guard"] = {"day": day, "units": committed}
        return new
    except Exception:  # noqa: BLE001 - the rule must never break the policy's action
        return action


def plant_clamp(action, obs):
    """The engine cancels every PLANT of a crop whose requests exceed the seeds held; keep the first ones that the seeds cover."""
    try:
        seeds = dict((obs.get("private") or {}).get("seeds") or {})
        cmds = _unit_cmds(action)
        demand = {}
        for c in cmds:
            if isinstance(c, list) and len(c) >= 2 and c[0] == "PLANT":
                demand[c[1]] = demand.get(c[1], 0) + 1
        over = {k for k, n in demand.items() if n > int(seeds.get(k, 0))}
        if not over:
            return action
        used, new = {}, action
        for i, c in enumerate(cmds):
            if isinstance(c, list) and len(c) >= 2 and c[0] == "PLANT" and c[1] in over:
                used[c[1]] = used.get(c[1], 0) + 1
                if used[c[1]] > int(seeds.get(c[1], 0)):
                    new = _with_unit(new, i, ["PASS"])
        return new
    except Exception:  # noqa: BLE001
        return action


TERMINAL_FROM = 700


def terminal_salvage(action, obs, raw_observation=None):
    """From t=700, units carrying products walk to a shed-access tile at the latest step that still reaches it by t=718 and
    deposit there; the t=718 sell-off then sells the shed."""
    try:
        step = int(obs["step"])
        if not TERMINAL_FROM <= step <= LAST_ACTION:
            return action
        import prefix
        E = prefix.E
        left = LAST_ACTION - step
        farm = obs["farms"][int(obs["player"])]
        access = [tuple(p) for p in E._shed_access_tiles(len(farm["tiles"]))]
        raw_inv = None
        if raw_observation is not None:
            raw_inv = _plain((raw_observation.get("private") or {}).get("inventories"))
        start_inv = raw_inv if isinstance(raw_inv, list) else (obs.get("private") or {}).get("inventories") or []
        positions = [tuple(farm["farmer"])] + [tuple(h) for h in farm.get("hands") or []]
        new = action
        for i, p in enumerate(positions):
            inv = start_inv[i] if i < len(start_inv) and isinstance(start_inv[i], dict) else {}
            prods = {k: int(n) for k, n in inv.items() if k in E.PRODUCTS and int(n) > 0}
            if not prods:
                continue
            d = min(abs(p[0] - a[0]) + abs(p[1] - a[1]) for a in access)
            if d == 0:
                only = all(k in E.PRODUCTS for k, n in inv.items() if int(n) > 0)
                trial = _with_unit(new, i, ["DROP"])
                before = _post_unit_private(_with_unit(new, i, ["PASS"]), obs, E, raw_inv)   # vs keeping the load
                after = _post_unit_private(trial, obs, E, raw_inv)
                total = lambda pr: sum(pr["shed"].values()) + sum(max(0, int(n)) for v in pr["inventories"] for n in v.values())
                if only and total(after) == total(before):
                    new = trial
                else:
                    k = max(prods, key=prods.get)
                    new = _with_unit(new, i, ["PLACE", k, prods[k]])
            elif d >= left:
                a = min(access, key=lambda a: (abs(p[0] - a[0]) + abs(p[1] - a[1]), a))
                dx, dy = a[0] - p[0], a[1] - p[1]
                move = (("EAST" if dx > 0 else "WEST") if abs(dx) >= abs(dy) and dx else ("SOUTH" if dy > 0 else "NORTH"))
                new = _with_unit(new, i, [move])
        return new
    except Exception:  # noqa: BLE001
        return action


# Maintainer 2026-09-27 (Fable review of model 2, section 2): whole-day hire cap as a paired rule arm. Hires cost fib(hires today)
# (the 12th hand $144, the 13th $233); our 12th+ hands cost $4,221/game vs $2,052 for matched opponents and $672 for DSM, and their
# productivity is unmeasured. HIREs beyond the cap become NOOP slots, keeping the lockstep order indices.
HIRE_CAP = {"on": False, "max_hires": 11}   # paired arm only; off in builds until measured


def hire_cap(action, obs):
    try:
        if not HIRE_CAP["on"]:
            return action
        farm = obs["farms"][int(obs["player"])]
        allowed = max(0, HIRE_CAP["max_hires"] - int(farm.get("hires_today", 0) or 0))
        out, n, cut = [], 0, False
        for o in action.get("market") or []:
            if isinstance(o, (list, tuple)) and o and o[0] == "HIRE":
                n += 1
                if n > allowed:
                    out.append(["NOOP"])
                    cut = True
                    continue
            out.append(o)
        return {**action, "market": out} if cut else action
    except Exception:  # noqa: BLE001
        return action


def sanitize_action(action):
    """Plain JSON action with only engine keys; raises on structural invalidity."""
    if not isinstance(action, dict):
        raise TypeError(f"action must be a dict, got {type(action).__name__}")
    hands = action.get("hands", [])
    market = action.get("market", [])
    if not isinstance(hands, (list, tuple)) or not isinstance(market, (list, tuple)):
        raise TypeError("hands and market must be lists")
    if len(market) > MAX_ORDERS:
        raise ValueError(f"{len(market)} market orders > {MAX_ORDERS}")
    return {"farmer": _unit(action.get("farmer", ["PASS"])),
            "hands": [_unit(a) for a in hands],
            "market": [_order(o) for o in market]}


# ----------------------------------------------------------------------------- stub
UNIT_OPS = ("PASS", "NORTH", "SOUTH", "EAST", "WEST", "PLANT", "WATER", "HARVEST",
            "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED",
            "COLLECT_FERTILIZER", "CARE", "DROP", "PICKUP", "PLACE")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")


class StubPolicy:
    """Torch-free random grammar policy with the PolicyNet.act() shape (harness tests only).

    ``act(obs, generator=None, temperature=1.0, state=None) -> (action, state)``; the
    state is a tiny carried dict so the per-episode/seat field plumbing is exercised.
    """

    kind = "stub"

    def __init__(self, seed=0, max_units=13):
        self.seed = int(seed)
        self.max_units = int(max_units)
        self.rng = random.Random(self.seed)

    def act(self, observation, generator=None, temperature=1.0, state=None):
        rng = generator if isinstance(generator, random.Random) else self.rng
        farm = observation["farms"][observation["player"]]
        n_units = min(self.max_units, 1 + len(farm.get("hands", [])))
        units = [self._unit(rng) for _ in range(n_units)]
        market = [self._order(rng) for _ in range(rng.randrange(0, 4))]
        carried = {"steps_seen": (state or {}).get("steps_seen", 0) + 1,
                   "last_step": observation.get("step")}
        return {"farmer": units[0], "hands": units[1:], "market": market}, carried

    @staticmethod
    def _unit(rng):
        op = rng.choice(UNIT_OPS)
        if op == "PLANT":
            return [op, rng.choice(CROPS)]
        if op in ("PICKUP", "PLACE"):
            return [op, rng.choice(PRODUCTS + ANIMALS), rng.randint(1, 5)]
        return [op]

    @staticmethod
    def _order(rng):
        op = rng.choice(("SELL", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "HIRE", "BUY_LAND"))
        if op in ("HIRE", "BUY_LAND"):
            return [op]
        item = {"SELL": PRODUCTS, "BUY_SEED": CROPS, "BUY_ANIMAL": ANIMALS,
                "BUY_PRODUCT": ("WHEAT", "FERTILIZER")}[op]
        return [op, rng.choice(item), rng.randint(1, 5)]

    def export(self):
        return {"format": WEIGHTS_FORMAT, "kind": "stub", "seed": self.seed, "max_units": self.max_units}


# ----------------------------------------------------------------------------- loading
def _model_module():
    import model  # primitive-bc/model.py
    return model


def default_config(model):
    for name in ("default_config", "DefaultConfig"):
        fn = getattr(model, name, None)
        if callable(fn):
            return fn()
    for name in ("PolicyConfig", "ModelConfig", "Config"):
        cls = getattr(model, name, None)
        if isinstance(cls, type):
            return cls()
    for name in ("DEFAULT_CONFIG", "CONFIG"):
        if isinstance(getattr(model, name, None), dict):
            return copy.deepcopy(getattr(model, name))
    return None


def _config_to_plain(config):
    import dataclasses
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config), type(config).__name__
    if isinstance(config, dict):
        return copy.deepcopy(config), "dict"
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return to_dict(), type(config).__name__
    raise TypeError(f"cannot serialise PolicyNet config of type {type(config).__name__}")


def _config_from_plain(model, plain, kind):
    if kind in (None, "dict"):
        return copy.deepcopy(plain)
    cls = getattr(model, kind, None)
    if cls is None:
        raise ValueError(f"model.py has no config class {kind}")
    from_dict = getattr(cls, "from_dict", None)
    return from_dict(plain) if callable(from_dict) else cls(**plain)


def net_config(net):
    for name in ("config", "cfg"):
        if hasattr(net, name):
            return getattr(net, name)
    raise AttributeError("PolicyNet exposes no .config/.cfg; cannot export it")


def random_policy(seed=0, config=None):
    import torch
    model = _model_module()
    torch.manual_seed(int(seed))
    config = default_config(model) if config is None else config
    net = model.PolicyNet(config) if config is not None else model.PolicyNet()
    return net.eval()


def _dequantize(entry, torch):
    if isinstance(entry, dict) and entry.get("q") is not None:
        q, scale = entry["q"], entry["scale"]
        return (q.float() * scale.float().view(-1, *([1] * (q.dim() - 1)))).reshape(entry["shape"])
    return entry.float() if entry.is_floating_point() else entry


def read_weights_file(path):
    """The ``primitive-bc-weights-v1`` blob at ``path``, or None when it is not one."""
    import torch
    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - not a weights_only-loadable file: a training checkpoint
        return None
    return blob if isinstance(blob, dict) and blob.get("format") == WEIGHTS_FORMAT else None


def policy_from_weights(blob):
    """``primitive-bc-weights-v1`` (fp32 / fp16 / int8 storage) -> policy in fp32 eval mode."""
    import torch
    if blob.get("kind") == "stub":
        return StubPolicy(blob["seed"], blob.get("max_units", 13))
    model = _model_module()
    config = _config_from_plain(model, blob["config"], blob.get("config_kind"))
    net = model.PolicyNet(config)
    net.load_state_dict({name: _dequantize(entry, torch) for name, entry in blob["tensors"].items()},
                        strict=True)
    return net.eval()


def load_checkpoint(path):
    """Training checkpoint -> policy: model.py's own loader first, then a generic torch dict."""
    import torch
    model = _model_module()
    loaders = [getattr(model, n, None) for n in ("load_policy", "load_checkpoint")]
    loaders += [getattr(model.PolicyNet, n, None) for n in ("load", "from_checkpoint")]
    first_error = None
    for fn in (f for f in loaders if callable(f)):
        try:
            net = fn(str(path))
            net = net[0] if isinstance(net, tuple) else net
            return net.eval()
        except Exception as error:  # noqa: BLE001 - try the next loader, report the first failure
            first_error = first_error or error
    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        config = next((blob[k] for k in ("config", "model_config", "cfg") if k in blob), None)
        state = next((blob[k] for k in ("model", "state_dict", "model_state", "weights") if k in blob), None)
        if state is None:
            raise ValueError(f"{path}: no state dict under model/state_dict/model_state/weights")
        if isinstance(config, dict):
            from_dict = getattr(getattr(model, "ModelConfig", None), "from_dict", None)
            config = from_dict(config) if callable(from_dict) else config
        net = model.PolicyNet(config) if config is not None else model.PolicyNet()
        net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in state.items()},
                            strict=True)
        return net.eval()
    except Exception:
        if first_error is not None:
            raise first_error
        raise


def is_stub_spec(spec):
    return str(spec).partition(":")[0] == "stub"


def load_policy(spec, *, seed=0):
    """Resolve a model spec to a policy object exposing ``act``."""
    spec = str(spec)
    head, _, tail = spec.partition(":")
    if head == "stub":
        return StubPolicy(int(tail) if tail else seed)
    if head == "random":
        return random_policy(int(tail) if tail else seed)
    path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(spec)
    blob = read_weights_file(path)
    return policy_from_weights(blob) if blob is not None else load_checkpoint(path)


def export_weights(policy, *, storage="fp16", int8_min_numel=4096):
    """Policy -> ``primitive-bc-weights-v1`` blob (dict for torch.save, weights_only-safe)."""
    if isinstance(policy, StubPolicy):
        return policy.export()
    import torch
    if storage not in ("fp32", "fp16", "int8"):
        raise ValueError("storage must be fp32, fp16 or int8")
    plain, kind = _config_to_plain(net_config(policy))
    # Auxiliary next-turn/opponent heads are supervised during training but not
    # read by the eval actor. Drop both their weights and their construction
    # flags so cold loading stays strict and the hosted archive has headroom.
    strip_aux = bool(plain.get("v6_aux", False))
    if strip_aux:
        plain = dict(plain)
        plain["v6_aux"] = False
        plain["v6_aux_turn"] = False
    tensors = {}
    for name, value in policy.state_dict().items():
        if strip_aux and name.startswith("aux_"):
            continue
        value = value.detach().cpu()
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):   # PRO-RT-DEPLOY-0927 #6
            raise ValueError(f"non-finite checkpoint tensor {name}")
        if value.is_floating_point() and not bool(torch.isfinite(value.float()).all()):   # PRO-RELEASE-R3-0928 #6: fp64 -> fp32
            raise ValueError(f"{name} overflows fp32")
        halved = value.is_floating_point() and (storage == "fp16" or (
            storage == "int8" and not (value.dim() >= 2 and value.numel() >= int8_min_numel)))
        if halved and not bool(torch.isfinite(value.half()).all()):
            raise ValueError(f"{name} overflows fp16 storage; build with --storage fp32")
        if not value.is_floating_point():
            tensors[name] = value.clone()
        elif storage == "int8" and value.dim() >= 2 and value.numel() >= int8_min_numel:
            flat = value.float().reshape(value.shape[0], -1)
            scale = flat.abs().amax(dim=1).clamp_min(1e-12) / 127.0
            q = torch.round(flat / scale[:, None]).clamp(-127, 127).to(torch.int8)
            tensors[name] = {"q": q.reshape(value.shape).contiguous(), "scale": scale.float(),
                             "shape": list(value.shape)}
        elif storage in ("fp16", "int8"):
            tensors[name] = value.half().clone()
        else:
            tensors[name] = value.float().clone()
    return {"format": WEIGHTS_FORMAT, "kind": "policynet", "storage": storage,
            "config": plain, "config_kind": kind, "tensors": tensors}


# ----------------------------------------------------------------------------- agent
class Agent:
    """Callable Kaggle agent. Never raises out of ``__call__``; PASS on any exception."""

    def __init__(self, model_path, *, seed=0, temperature=1.0, threads=1, log=True,
                 stream=None, policy=None, decode=None):
        self.model_path = str(model_path)
        self.decode = step_search.normalize(decode)
        self.decode_log = []        # per searched step: log-probs, chosen row, ms (bounded)
        self.search_errors = 0      # searched steps that fell back to the greedy decode
        self.search_error_log = []
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.threads = int(threads)
        self.log = bool(log)
        self.stream = stream
        self.policy = policy
        self.generator = None
        self._act_params = None
        self._states = {}           # seat -> {"last_step": int, "state": object}
        self.fallbacks = 0
        self.errors = []
        self.calls = 0
        self.ms = []
        self.load_ms = None
        self.last_fallback = False
        self.last_error = None
        self.load_error = None

    # -- loading
    def _ensure_loaded(self):
        if self.policy is not None and self._act_params is not None:
            return
        if self.load_error is not None:
            # One load attempt per process: a failed load must not re-pay its cost every step.
            raise RuntimeError(f"policy load failed earlier: {self.load_error}")
        begun = time.perf_counter()
        try:
            self._load()
        except Exception as error:
            self.load_error = f"{type(error).__name__}: {error}"
            self.load_ms = (time.perf_counter() - begun) * 1000.0
            raise
        self.load_ms = (time.perf_counter() - begun) * 1000.0

    def _load(self):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        if self.policy is None:
            if not is_stub_spec(self.model_path):
                import torch
                torch.set_num_threads(max(1, self.threads))
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass
            self.policy = load_policy(self.model_path, seed=self.seed)
        if isinstance(self.policy, StubPolicy):
            self.generator = random.Random(self.seed)
        else:
            import torch
            self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        act = getattr(self.policy, "act")
        try:
            self._act_params = tuple(inspect.signature(act).parameters)
        except (TypeError, ValueError):
            self._act_params = ()

    def _initial_state(self):
        for name in ("initial_state", "init_state", "new_state", "reset_state"):
            fn = getattr(self.policy, name, None)
            if callable(fn):
                return fn()
        return None

    def _state_for(self, seat, step):
        slot = self._states.get(seat)
        if slot is None or step == 0 or step != slot["last_step"] + 1:
            reset = getattr(self.policy, "reset", None)
            if slot is not None and callable(reset):
                reset()
            slot = self._states[seat] = {"last_step": -1, "state": self._initial_state()}
        return slot

    def _act(self, clean, cfg, state, *, v8_history=None, v8_long=None):
        params = self._act_params or ()
        kwargs = {}
        if "generator" in params:
            kwargs["generator"] = self.generator
        if "temperature" in params:
            kwargs["temperature"] = self.temperature
        if "configuration" in params:
            kwargs["configuration"] = cfg
        if "v8_history" in params and v8_history is not None:
            kwargs["v8_history"] = v8_history
        if "v8_long" in params and v8_long is not None:
            kwargs["v8_long"] = v8_long
        state_param = next((p for p in STATE_PARAMS if p in params), None)
        if state_param is not None:
            kwargs[state_param] = state
        searched = (not isinstance(self.policy, StubPolicy) and hasattr(self.policy, "encode")
                    and step_search.applies(self.decode, int(clean["step"])))
        out = None
        if searched:
            import torch
            info = self.decode_log if len(self.decode_log) < 5000 else None
            search_kwargs = dict(kwargs)
            search_state = search_kwargs.pop(state_param, None) if state_param is not None else None
            try:
                with torch.inference_mode():
                    out = step_search.act(self.policy, clean, self.decode, state=search_state, info=info,
                                          **search_kwargs)
            except Exception as error:  # noqa: BLE001 - a failed search plays this step's greedy decode instead
                # (review 2026-09-29: e.g. noop_unit_mask models, non-finite sampling logits). The greedy call below
                # recomputes the features from the same carried state; v8 compute_long is idempotent per step.
                out = None
                self.search_errors += 1
                if len(self.search_error_log) < 20:
                    self.search_error_log.append(f"step {clean.get('step')}: {type(error).__name__}: {error}"[:300])
        if out is None and not isinstance(self.policy, StubPolicy):
            import torch
            with torch.inference_mode():
                out = self.policy.act(clean, **kwargs)
        elif out is None:
            out = self.policy.act(clean, **kwargs)
        if isinstance(out, dict):
            return out, state
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], dict):
            return out[0], (out[1] if len(out) > 1 else state)
        action = getattr(out, "action", None)
        if isinstance(action, dict):
            return action, getattr(out, "state", state)
        raise TypeError(f"act() returned {type(out).__name__}, expected action dict or (action, state)")

    # -- call
    def step(self, observation, configuration=None):
        """One decision; raises on any error (``__call__`` wraps it with the fallback)."""
        self._ensure_loaded()
        clean = actor_view(observation)
        if clean.get("step") is None and "day" in clean and "hour" in clean:
            clean["step"] = int(clean["day"]) * 24 + int(clean["hour"])
        cfg = public_configuration(configuration)
        step, seat = int(clean["step"]), int(clean["player"])
        slot = self._state_for(seat, step)
        pc = getattr(self.policy, "config", None)
        needs_v8 = bool(pc and (getattr(pc, "x8_token_width", 0) or getattr(pc, "x8_unit_width", 0)
                                or getattr(pc, "lattice_channels", 0)))
        if needs_v8 and "v8_long" not in slot:
            from v8_gaps import GapsState
            slot["v8_long"] = GapsState()
            slot["v8_history"] = []
        slot.pop("pending_state", None)
        action, new_state = self._act(clean, cfg, slot["state"],
                                      v8_history=slot["v8_history"] if needs_v8 else None,
                                      v8_long=slot["v8_long"] if needs_v8 else None)
        slot["pending_state"] = new_state
        decoded_action = copy.deepcopy(action) if pc and getattr(pc, "v6_prev", False) else None
        action = plant_clamp(action, clean)
        action = land_rule(action, clean, cfg, slot)
        action = hire_cap(action, clean)
        action = overflow_guard(action, clean, cfg, observation, slot)
        action = terminal_salvage(action, clean, observation)
        action = sell_rule(action, clean, cfg, observation)
        action = sanitize_action(action)
        if decoded_action is not None and action != decoded_action:
            # The trainer's lagged action labels describe the submitted wire action,
            # not the neural decoder's proposal before these external rules.
            import data as _data
            if not getattr(new_state, "hist", ()) or int(new_state.hist[-1][0]) != step:
                raise ValueError("v6 action history is missing the current decision")
            saved = _data.encode_step_lenient(action, clean)
            entry = (step, *(saved[k] for k in
                             ("unit_tok", "unit_qty", "order_tok", "order_qty")))
            new_state = replace(new_state, hist=(*new_state.hist[:-1], entry))
        if needs_v8:
            slot["v8_history"] = [clean, *slot["v8_history"][:1]]
        slot["state"], slot["last_step"] = new_state, step
        slot.pop("pending_state", None)
        return action

    def _advance_fallback(self, observation, configuration):
        """Record the submitted PASS and visible step even when policy/rules fail."""
        clean = actor_view(observation)
        if clean.get("step") is None and "day" in clean and "hour" in clean:
            clean["step"] = int(clean["day"]) * 24 + int(clean["hour"])
        step, seat = int(clean["step"]), int(clean["player"])
        slot = self._states.get(seat)
        if slot is not None and step == slot["last_step"]:
            return
        if slot is None or step == 0 or step != slot["last_step"] + 1:
            slot = self._state_for(seat, step)
        pc = getattr(self.policy, "config", None)
        needs_v8 = bool(pc and (getattr(pc, "x8_token_width", 0) or getattr(pc, "x8_unit_width", 0)
                                or getattr(pc, "lattice_channels", 0)))
        if needs_v8:
            from v8_gaps import GapsState, compute_long
            slot.setdefault("v8_long", GapsState())
            slot.setdefault("v8_history", [])
            # compute_long is idempotent if act() already reached its feature extractor.
            compute_long(clean, public_configuration(configuration), (), slot["v8_long"])
            if not slot["v8_history"] or int(slot["v8_history"][0]["step"]) != step:
                slot["v8_history"] = [clean, *slot["v8_history"][:1]]
        pending = slot.pop("pending_state", None)
        state = pending if pending is not None else slot["state"]
        if pc and getattr(pc, "v6_prev", False):
            import data as _data
            saved = _data.encode_step_lenient(pass_action(), clean)
            entry = (step, *(saved[k] for k in ("unit_tok", "unit_qty", "order_tok", "order_qty")))
            hist = tuple(h for h in getattr(state, "hist", ()) if int(h[0]) < step)
            hist = (*hist, entry)[-2:]
            if state is None:
                from model import ThinkState
                state = ThinkState(None, step, hist)
            else:
                state = replace(state, field=getattr(state, "field", None) if pending is not None else None,
                                last_step=step, hist=hist)
        slot["state"], slot["last_step"] = state, step

    def __call__(self, observation, configuration=None):
        begun = time.perf_counter()
        loading = self.load_ms is None
        self.calls += 1
        self.last_fallback, self.last_error = False, None
        try:
            action = self.step(observation, configuration)
        except Exception as error:  # noqa: BLE001 - PASS fallback is the contract
            action = pass_action()
            self.fallbacks += 1
            self.last_fallback = True
            self.last_error = f"{type(error).__name__}: {error}"
            if len(self.errors) < 20:
                self.errors.append(self.last_error[:500])
            try:
                self._advance_fallback(observation, configuration)
            except Exception as carry_error:  # noqa: BLE001 - never replace the safe wire fallback
                if len(self.errors) < 20:
                    self.errors.append(f"fallback carry: {type(carry_error).__name__}: {carry_error}"[:500])
        ms = (time.perf_counter() - begun) * 1000.0
        self.ms.append(ms)
        if self.log:
            self._log(observation, ms, loading)
        return action

    def _log(self, observation, ms, loading):
        try:
            step = observation.get("step") if hasattr(observation, "get") else None
            seat = observation.get("player") if hasattr(observation, "get") else None
            extra = f" load_ms={self.load_ms:.0f}" if loading and self.load_ms is not None else ""
            if self.last_fallback:
                extra += f" err={self.last_error.split(':', 1)[0]}"
            print(f"{LOG_PREFIX} step={step} seat={seat} ms={ms:.1f} fb={int(self.last_fallback)}{extra}",
                  file=self.stream or sys.stderr, flush=True)
        except Exception:  # noqa: BLE001 - logging never breaks a step
            pass


def make_agent(model_path, *, seed=0, temperature=1.0, threads=1, log=True, stream=None, decode=None):
    """``model_path`` -> callable(observation, configuration) -> action (lazy CPU load)."""
    return Agent(model_path, seed=seed, temperature=temperature, threads=threads,
                 log=log, stream=stream, decode=decode)
