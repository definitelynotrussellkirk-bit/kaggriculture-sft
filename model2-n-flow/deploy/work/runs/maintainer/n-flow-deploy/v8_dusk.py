"""v8 feature family "dusk": dusk load and return-to-shed feasibility (own seat only).

Engine rules (src/farm_neat/game/kaggriculture_engine_reference.py, cited as E<line>):
  * One interpreter step s applies every own unit action (farmer, then hands in index order; E935-939), then the market
    (E941: SELL takes shed units only, E652-661), then town consumption and decay; when (s + 1) % 24 == 0 the day ends
    (E945-946). So a DROP and a SELL of the dropped units can both happen in step s.
  * Day end (E860-882): every carrier inventory, farmer first then hands, is dropped into the shed up to shedCapacity and
    the overflow is discarded (E843-857, E878); the farmer respawns at the shed and hands are dismissed (E879-882).
    The deposit ignores where the carrier stands.
  * DROP needs a shed-access tile (E343-357; tiles (4,4) (5,4) (4,5) (5,5), E132-135) and discards what exceeds the room;
    moves are one tile per step with no blocking (E323-331). A carrier at Manhattan distance d from the nearest access tile
    can DROP at the earliest in decision d counted from 0 (the current one), i.e. visible ETA d + 1 (flip.py convention).
  * The episode's last processed interpreter step is 718 (E960: DONE once step >= episodeSteps - 2). The day-29 day end
    would need step 719, so on day 29 nothing carried is ever deposited; goods not dropped and sold by step 718 are lost.

"Deadline" below = the last interpreter step whose actions still act before goods are lost: s = 24*day + 23 (the dusk
step) on days 0-28, s = 718 on day 29.  left = deadline - t + 1 = decisions still available, the current one included.

compute(actor_obs, configuration, token_ids, history=()) -> {"token": [N, WIDTH_TOKEN] f16, "unit": [25, WIDTH_UNIT] f16}.
Only the actor observation (own private state, public board and market) and the seat's own previous observations are read.
"""
from __future__ import annotations

import math

import numpy as np

import calc_features as CALC            # legacy schema (the base models); upstream used calc_features_v7
import codec
import flip                              # SHED tiles and _shed_dist (Manhattan to the nearest shed-access tile)
from farm_neat.economy.heuristic.constants import ANIMALS, CROPS, PRODUCTS

DAY = 24
FINAL_DECISION = 718                     # last interpreter step (E960); day 29 has no day end
MAX_UNITS = codec.MAX_UNITS              # 25 unit rows, codec.unit_positions order (row 0 = farmer)
ITEMS = tuple(PRODUCTS) + tuple(ANIMALS)  # shed/stock token order (observation.ITEMS)
CROP_ITEMS = ("CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMAL_PRODUCTS = ("EGG", "MILK", "WOOL")

GLOBAL_COLS = (
    "dusk_left",                 # decisions until the deadline, current included, /24
    "dusk_final_day",            # 1 on day 29: no day-end deposit remains
    "dusk_shed_fill",            # shed units (animals included) / capacity
    "dusk_shed_room",            # (capacity - fill) / 100
    "dusk_carried_units",        # all own carried units / 100
    "dusk_carried_value_spot",   # slog($ at spot quote, 1000)
    "dusk_carried_value_proj",   # slog($ at calc projected quote P*, 1000)  = total carried value at risk
    "dusk_lost_units_idle",      # units lost at the deadline if nothing changes / 100
    "dusk_lost_value_idle_min",  # slog($ spot, 1000), cheapest order within each carrier
    "dusk_lost_value_idle_max",  # slog($ spot, 1000), dearest order within each carrier
    "dusk_overflow_after_sellout",  # units lost even if the whole sellable shed is sold by the dusk step / 100
    "dusk_feasible_payload",     # units on carriers with return slack >= 0 / 100
    "dusk_late_payload",         # units on carriers with return slack < 0 / 100
    "dusk_overflow_unavoidable",  # lower bound: lost even if every feasible carrier deposits and all is sold / 100
    "dusk_n_late_loaded",        # loaded carriers with slack < 0 / 10
    "dusk_min_loaded_slack",     # min slack over loaded carriers / 24 in [-1, 1]; 1 if none loaded
    "dusk_pending_ready_units",  # harvest-ready own units some carrier can reach by the deadline / 100
    "dusk_pending_return_units",  # ... that some carrier can harvest AND drop by the deadline / 100
    "dusk_overflow_with_pending",  # max(0, fill + carried + pending_ready - capacity) / 100 (0 on day 29)
    "dusk_late_value_proj",      # slog($ P*, 1000) carried by carriers with slack < 0
)
ITEM_COLS = (
    "dusk_item_carried",         # this item carried by all own carriers / 100
    "dusk_item_lost_idle_min",   # min units of this item lost at the deadline if nothing changes / 100
    "dusk_item_lost_idle_max",   # max units ... (bounds over the unknown within-carrier insertion order) / 100
    "dusk_item_carried_late",    # this item on carriers with slack < 0 / 100
)
TOKEN_COLS = GLOBAL_COLS + ITEM_COLS
UNIT_COLS = (
    "dusk_u_carried",            # units carried / 100 (clip 4)
    "dusk_u_wheat",              # WHEAT / 100
    "dusk_u_fertilizer",         # FERTILIZER / 100
    "dusk_u_crop",               # CARROT+TOMATO+STRAWBERRY+MELON / 100
    "dusk_u_animal_product",     # EGG+MILK+WOOL / 100
    "dusk_u_live_animal",        # GOOSE+COW+SHEEP / 10
    "dusk_u_shed_eta",           # (Manhattan to nearest access tile + 1) / 24
    "dusk_u_return_slack",       # (left - shed_eta) / 24 in [-1, 1]
    "dusk_u_can_return",         # 1 if slack >= 0
    "dusk_u_lost_idle",          # units of this carrier lost at the deadline if nothing changes / 100 (exact total)
    "dusk_u_lost_value_min",     # slog($ spot, 100)
    "dusk_u_lost_value_max",     # slog($ spot, 100)
    "dusk_u_value_spot",         # slog(carried $ at spot quote, 100)
    "dusk_u_value_proj",         # slog(carried $ at P*, 100)  = value at risk
    "dusk_u_queue_room",         # (room at its deposit - load) / 100 in [-2, 1], all loaded carriers heading home now
    "dusk_u_detour_cost",        # min extra decisions of one harvest detour before dropping / 24; 1 if none fits
    "dusk_u_pending_return",     # ready units on tiles it can harvest and still drop by the deadline / 100
    "dusk_u_home_delta",         # +1 moved closer to the shed since the previous observation, -1 farther, 0 same/unknown
    "dusk_u_carried_delta",      # change of units carried since the previous observation / 10 in [-2, 2]
)
WIDTH_TOKEN = len(TOKEN_COLS)
WIDTH_UNIT = len(UNIT_COLS)
N_GLOBAL = len(GLOBAL_COLS)
_G = {n: i for i, n in enumerate(GLOBAL_COLS)}
_U = {n: i for i, n in enumerate(UNIT_COLS)}
_ANIMAL_VALUE = {a: float(ANIMALS[a]["cost"]) for a in ANIMALS}     # a lost animal loses its purchase price
_PQ = CALC._PC["projected_quote"]


def slog(x, scale):
    return math.copysign(math.log1p(abs(x) / scale), x) if x else 0.0


def deadline(t):
    """(deadline interpreter step, final_day) for the decision at step t."""
    day = t // DAY
    if day * DAY + DAY - 1 >= FINAL_DECISION + 1:     # day 29: the day end would need step 719
        return FINAL_DECISION, True
    return day * DAY + DAY - 1, False


def unit_values(actor_obs, configuration):
    """(spot, projected) $ per unit for every item: spot = current sell quote (market prices, E652-661 quote), projected =
    calc_features P*; animals at purchase cost in both."""
    prices = actor_obs["market"]["prices"]
    prod, _ = CALC.raw(actor_obs, configuration)
    spot, proj = dict(_ANIMAL_VALUE), dict(_ANIMAL_VALUE)
    for i, p in enumerate(CALC.PRODUCTS):
        spot[p] = float(prices[p])
        proj[p] = float(prod[i, _PQ])
    return spot, proj


def _lost_value_bounds(inv, lost, value):
    """($ of the cheapest `lost` units, $ of the dearest `lost` units) of one inventory."""
    if lost <= 0:
        return 0.0, 0.0
    units = sorted((value.get(k, 0.0), n) for k, n in inv.items() if n > 0)
    lo = hi = 0.0
    r = lost
    for v, n in units:
        take = min(n, r)
        lo += take * v
        r -= take
        if r <= 0:
            break
    r = lost
    for v, n in reversed(units):
        take = min(n, r)
        hi += take * v
        r -= take
        if r <= 0:
            break
    return lo, hi


def _ready_tiles(farm, day):
    """[(x, y, units)] own tiles a HARVEST empties now (E446-474): plants past first_yield_day with yield, animals with yield."""
    out = []
    for y, line in enumerate(farm["tiles"]):
        for x, tile in enumerate(line):
            if not isinstance(tile, dict):
                continue
            units = int(tile.get("yield_units", 0) or 0)
            if units <= 0:
                continue
            if tile.get("kind") == "PLANT":
                if day - int(tile["planted_day"]) < CROPS[tile["crop"]]["first_yield_day"]:
                    continue
            elif "animal" not in tile:
                continue
            out.append((x, y, units))
    return out


def _prev_same_day(history, t):
    """The previous observation if it is step t-1 of the same day (worker slots then name the same workers)."""
    if not history:
        return None
    prev = history[0]
    try:
        ts = int(prev["step"])
    except (KeyError, TypeError, ValueError):
        return None
    if ts != t - 1 or ts // DAY != t // DAY:
        return None
    return prev


def raw(actor_obs, configuration, history=()):
    """Unscaled facts (integers, $) for compute() and the engine tests."""
    t = int(actor_obs["step"])
    seat = int(actor_obs["player"])
    cap = int((configuration or {}).get("shedCapacity", 100) or 100)
    farm = actor_obs["farms"][seat]
    private = actor_obs.get("private") or {}
    shed = private.get("shed") or {}
    invs = [inv or {} for inv in (private.get("inventories") or [])]
    workers = np.asarray([farm["farmer"], *farm["hands"]], np.int64).reshape(-1, 2)
    n_w = len(workers)
    invs = invs + [{}] * max(0, n_w - len(invs))       # inventories are aligned with [farmer, *hands] (E173-174, E709)

    dl, final = deadline(t)
    left = dl - t + 1                                    # <= 0 only at the terminal boundary t = 719
    fill = sum(int(v) for v in shed.values())
    room0 = max(0, cap - fill)
    shed_animals = sum(int(shed.get(a, 0)) for a in ANIMALS)
    spot, proj = unit_values(actor_obs, configuration)

    load = np.asarray([sum(int(n) for n in inv.values() if n > 0) for inv in invs[:n_w]], np.int64).reshape(n_w)
    d_shed = flip._shed_dist(workers) if n_w else np.zeros(0, np.int64)   # Manhattan to the nearest access tile
    eta = d_shed + 1
    slack = left - eta
    feasible = slack >= 0

    # dusk deposit if nothing changes (E843-857: farmer first, hands in order, overflow discarded); day 29: all lost
    lost = np.zeros(n_w, np.int64)
    room = room0
    for k in range(n_w):
        if final:
            lost[k] = load[k]
        else:
            take = min(int(load[k]), room)
            lost[k] = load[k] - take
            room -= take
    item_carried = {k: 0 for k in ITEMS}
    item_lmin = {k: 0 for k in ITEMS}
    item_lmax = {k: 0 for k in ITEMS}
    item_late = {k: 0 for k in ITEMS}
    lv_lo, lv_hi, v_spot, v_proj = (np.zeros(n_w) for _ in range(4))
    for k in range(n_w):
        inv = invs[k]
        nk, lk = int(load[k]), int(lost[k])
        for item, n in inv.items():
            if n <= 0:
                continue
            v_spot[k] += n * spot.get(item, 0.0)
            v_proj[k] += n * proj.get(item, 0.0)
            if item in item_carried:
                item_carried[item] += n
                item_lmin[item] += max(0, lk - (nk - n))
                item_lmax[item] += min(n, lk)
                if not feasible[k]:
                    item_late[item] += n
        lv_lo[k], lv_hi[k] = _lost_value_bounds(inv, lk, spot)

    # every loaded carrier walks home now and DROPs on arrival (E343-357 discards what exceeds the room), no sales;
    # carriers that cannot arrive by the deadline deposit at the day end after them (E878), in index order.
    big = 10 ** 6
    order = sorted(range(n_w), key=lambda k: (int(d_shed[k]) if feasible[k] else big, k))
    queue_room = np.zeros(n_w, np.int64)
    room = room0
    for k in order:
        if final and not feasible[k]:
            queue_room[k] = -int(load[k])                # never deposited: all of it is lost
            continue
        queue_room[k] = room - int(load[k])
        room = max(0, room - int(load[k]))

    # harvest-ready units and one-tile harvest detours (walk, HARVEST, walk to the nearest access tile, DROP)
    ready = _ready_tiles(farm, t // DAY)
    detour_extra = np.full(n_w, -1, np.int64)            # -1 = no detour fits before the deadline
    detour_cell = [None] * n_w
    pend_ret = np.zeros(n_w, np.int64)
    pending_ready = pending_return = 0
    if ready and n_w:
        cells = np.asarray([(x, y) for x, y, _ in ready], np.int64)
        units = np.asarray([u for _, _, u in ready], np.int64)
        dwj = np.abs(workers[:, None, 0] - cells[None, :, 0]) + np.abs(workers[:, None, 1] - cells[None, :, 1])   # [W, C]
        djs = flip._shed_dist(cells)                                                                             # [C]
        route = dwj + 1 + djs[None, :] + 1               # visible ETA of the DROP that follows the HARVEST
        # HARVEST acts before decay in its own step. Only count decay in
        # [t, t + travel - 1]; an on-time route can otherwise collect nothing.
        expiry = np.asarray([farm["tiles"][y][x].get("max_lifespan_step", -1)
                             for x, y, _ in ready], np.int64)
        first_tick = np.maximum(t, expiry)
        first_tick += (first_tick - expiry) % 2
        last_before_harvest = t + dwj - 1
        decay = np.where(expiry[None, :] >= 0,
                         np.maximum(0, (last_before_harvest - first_tick[None, :]) // 2 + 1), 0)
        payload = np.maximum(0, units[None, :] - decay)
        ok = (route <= left) & (payload > 0)
        # Per-tile alternatives, NOT a simultaneously schedulable harvest plan.
        pending_ready = int(np.where(dwj + 1 <= left, payload, 0).max(0).sum())
        pending_return = int(np.where(ok, payload, 0).max(0).sum())
        extra = np.where(ok, route - eta[:, None], big)
        best = extra.argmin(1)
        for k in range(n_w):
            if ok[k, best[k]]:
                detour_extra[k] = int(extra[k, best[k]])
                detour_cell[k] = (int(cells[best[k], 0]), int(cells[best[k], 1]))
        pend_ret = (ok * payload).sum(1)

    # history: motion and load change of the same worker slot since t-1 (same day only)
    home_delta = np.zeros(n_w, np.int64)
    load_delta = np.zeros(n_w, np.int64)
    prev = _prev_same_day(history, t)
    if prev is not None:
        pfarm = prev["farms"][seat]
        pw = np.asarray([pfarm["farmer"], *pfarm["hands"]], np.int64).reshape(-1, 2)
        pinv = (prev.get("private") or {}).get("inventories") or []
        m = min(n_w, len(pw))
        if m:
            home_delta[:m] = np.sign(flip._shed_dist(pw[:m]) - d_shed[:m])
            for k in range(min(m, len(pinv))):
                load_delta[k] = int(load[k]) - sum(int(n) for n in (pinv[k] or {}).values() if n > 0)

    carried = int(load.sum())
    late_load = int(load[~feasible].sum())
    loaded = load > 0
    return {
        "t": t, "deadline": dl, "final": final, "left": left, "cap": cap, "fill": fill, "room": room0,
        "shed_animals": shed_animals, "carried": carried, "late_load": late_load,
        "lost_total": int(lost.sum()),
        "overflow_after_sellout": carried if final else max(0, carried + shed_animals - cap),
        "overflow_unavoidable": late_load if final else max(0, late_load + shed_animals - cap),
        "n_late_loaded": int((loaded & ~feasible).sum()),
        "min_loaded_slack": int(slack[loaded].min()) if loaded.any() else None,
        "pending_ready": pending_ready, "pending_return": pending_return,
        "overflow_with_pending": 0 if final else max(0, fill + carried + pending_ready - cap),
        "late_value_proj": float(v_proj[~feasible].sum()),
        "item_carried": item_carried, "item_lost_min": item_lmin, "item_lost_max": item_lmax, "item_late": item_late,
        "load": load, "d_shed": d_shed, "eta": eta, "slack": slack, "feasible": feasible, "lost": lost,
        "lost_value_min": lv_lo, "lost_value_max": lv_hi, "value_spot": v_spot, "value_proj": v_proj,
        "queue_room": queue_room, "detour_extra": detour_extra, "detour_cell": detour_cell, "pending_return_k": pend_ret,
        "home_delta": home_delta, "load_delta": load_delta, "invs": invs[:n_w], "spot": spot,
    }


def compute(actor_obs, configuration, token_ids, history=()):
    r = raw(actor_obs, configuration, history)
    g = np.zeros(N_GLOBAL, np.float64)
    g[_G["dusk_left"]] = max(0, r["left"]) / DAY
    g[_G["dusk_final_day"]] = float(r["final"])
    g[_G["dusk_shed_fill"]] = r["fill"] / max(1, r["cap"])
    g[_G["dusk_shed_room"]] = r["room"] / 100
    g[_G["dusk_carried_units"]] = r["carried"] / 100
    g[_G["dusk_carried_value_spot"]] = slog(float(r["value_spot"].sum()), 1000)
    g[_G["dusk_carried_value_proj"]] = slog(float(r["value_proj"].sum()), 1000)
    g[_G["dusk_lost_units_idle"]] = r["lost_total"] / 100
    g[_G["dusk_lost_value_idle_min"]] = slog(float(r["lost_value_min"].sum()), 1000)
    g[_G["dusk_lost_value_idle_max"]] = slog(float(r["lost_value_max"].sum()), 1000)
    g[_G["dusk_overflow_after_sellout"]] = r["overflow_after_sellout"] / 100
    g[_G["dusk_feasible_payload"]] = (r["carried"] - r["late_load"]) / 100
    g[_G["dusk_late_payload"]] = r["late_load"] / 100
    g[_G["dusk_overflow_unavoidable"]] = r["overflow_unavoidable"] / 100
    g[_G["dusk_n_late_loaded"]] = r["n_late_loaded"] / 10
    s = r["min_loaded_slack"]
    g[_G["dusk_min_loaded_slack"]] = 1.0 if s is None else min(1.0, max(-1.0, s / DAY))
    g[_G["dusk_pending_ready_units"]] = r["pending_ready"] / 100
    g[_G["dusk_pending_return_units"]] = r["pending_return"] / 100
    g[_G["dusk_overflow_with_pending"]] = r["overflow_with_pending"] / 100
    g[_G["dusk_late_value_proj"]] = slog(r["late_value_proj"], 1000)
    g = np.clip(g, -8.0, 8.0)

    token = np.zeros((len(token_ids), WIDTH_TOKEN), np.float32)
    for n, tid in enumerate(token_ids):
        if tid == "global":
            token[n, :N_GLOBAL] = g
        elif tid.startswith("stock/shed/"):
            item = tid[11:]
            if item in r["item_carried"]:
                token[n, N_GLOBAL:] = [min(4.0, r[key][item] / 100)
                                       for key in ("item_carried", "item_lost_min", "item_lost_max", "item_late")]

    unit = np.zeros((MAX_UNITS, WIDTH_UNIT), np.float32)
    for k in range(min(len(r["load"]), MAX_UNITS)):
        inv = r["invs"][k]
        u = unit[k]
        u[_U["dusk_u_carried"]] = min(4.0, r["load"][k] / 100)
        u[_U["dusk_u_wheat"]] = min(4.0, inv.get("WHEAT", 0) / 100)
        u[_U["dusk_u_fertilizer"]] = min(4.0, inv.get("FERTILIZER", 0) / 100)
        u[_U["dusk_u_crop"]] = min(4.0, sum(inv.get(c, 0) for c in CROP_ITEMS) / 100)
        u[_U["dusk_u_animal_product"]] = min(4.0, sum(inv.get(c, 0) for c in ANIMAL_PRODUCTS) / 100)
        u[_U["dusk_u_live_animal"]] = min(4.0, sum(inv.get(c, 0) for c in ANIMALS) / 10)
        u[_U["dusk_u_shed_eta"]] = min(4.0, r["eta"][k] / DAY)
        u[_U["dusk_u_return_slack"]] = min(1.0, max(-1.0, r["slack"][k] / DAY))
        u[_U["dusk_u_can_return"]] = float(r["feasible"][k])
        u[_U["dusk_u_lost_idle"]] = min(4.0, r["lost"][k] / 100)
        u[_U["dusk_u_lost_value_min"]] = slog(r["lost_value_min"][k], 100)
        u[_U["dusk_u_lost_value_max"]] = slog(r["lost_value_max"][k], 100)
        u[_U["dusk_u_value_spot"]] = slog(r["value_spot"][k], 100)
        u[_U["dusk_u_value_proj"]] = slog(r["value_proj"][k], 100)
        u[_U["dusk_u_queue_room"]] = min(1.0, max(-2.0, r["queue_room"][k] / 100))
        e = int(r["detour_extra"][k])
        u[_U["dusk_u_detour_cost"]] = 1.0 if e < 0 else min(e, DAY) / DAY
        u[_U["dusk_u_pending_return"]] = min(4.0, r["pending_return_k"][k] / 100)
        u[_U["dusk_u_home_delta"]] = r["home_delta"][k]
        u[_U["dusk_u_carried_delta"]] = min(2.0, max(-2.0, r["load_delta"][k] / 10))
    unit = np.clip(unit, -8.0, 8.0)
    return {"token": token.astype(np.float16), "unit": unit.astype(np.float16)}


def warm(configuration=None):
    CALC.warm(configuration)
