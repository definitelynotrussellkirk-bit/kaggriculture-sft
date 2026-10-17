"""Exact no-action forecasts ("flip scalars"), duty counts and worker reachability from the actor-visible view.

Rules: src/farm_neat/game/kaggriculture_engine_reference.py. The interpreter at step s applies both players' unit
actions, the market, town consumption, plant decay (every 2 steps from max_lifespan_step), and when (s + 1) % 24 == 0
the day end: plants (a plant not watered that day adds 1 to consecutive_unwatered and becomes WEED at 2; ongoing
crops produce), animals (same rule with feeding: escape at 2; production), weed spawn, hands dismissed, farmer
respawn, and a shop unlock when (day + 1) % 3 == 0 (at most 8). The last day end is at s = 695.
ETA: this player's decisions, the current one included, before the change is visible; a change applied at
interpreter step s is visible at step s + 1, so ETA = s + 1 - t. Moves are one tile per step and nothing blocks, and
tile actions act on the standing tile, so a worker reaches and acts on a tile at Manhattan distance d in d + 1
decisions (FEED also needs one WHEAT in the worker's inventory).

token_features(actor_obs, token_ids) -> (extra [N, WIDTH] float16, unit_extra [MAX_UNITS, UNIT_WIDTH] float16)
aligned with observation.pack's token_ids and codec.unit_positions. extra[:, :LIN] feed a zero-initialised linear
layer; extra[:, KIND_COL] (job kind code) and extra[:, DEADLINE_COL] (deadline ETA, 0 = none) are raw integers for
the decoder's reachability bias.
"""
from __future__ import annotations

import math

import numpy as np

from farm_neat.economy.heuristic.constants import ANIMALS, CROPS, MARKET_PARAMS, PRODUCTS, SHOPS
from farm_neat.economy.heuristic.market import price_at

DAY = 24
LAST_DAY_END_NEXT_DAY = 29          # the day end at interpreter step 695 is the last one (next_day 29)
LAST_STEP = 719
MAX_SHOPS = 8
SHOP_EVERY_DAYS = 3
SHOP_SELL_EVERY = 4
CENTER_SELL_EVERY = 24
SHED = ((4, 4), (5, 4), (4, 5), (5, 5))
_LOG720 = math.log1p(720)
_LOG20 = math.log1p(20)

# job kind codes (most urgent first when a tile has several)
NONE, WATER_TONIGHT, FEED_TONIGHT, HARVEST_DECAYING, WATER_TODAY, FEED_TODAY, HARVEST, COLLECT, FERTILIZER, \
    PLANT_EMPTY, WEED, CARE = range(12)
N_KINDS = 12
TONIGHT = (WATER_TONIGHT, FEED_TONIGHT)

G0, F0, T0, C0 = 0, 8, 30, 62          # column blocks: GLOBAL 7, FARM 22, TILE 20 + 12 kind one-hots, CARRIER 12
E0, M0 = 74, 87                        # FARM economics 13 (precision-safe), MARKET no-trade price path 9
J0 = 96                                # v2: per-tile multi-job bits (12 columns, Codex Task 47 finding 3)
S0 = 108                               # v3: PLANT->WATER startup certificate on empty own tiles (4 columns, Astra review)
N0 = 112                               # v3: tonight's overflow (2 columns, Astra adversarial review B)
LIN = 114
TILE_COLS = C0 - T0
NO_REACH = 999                         # reach_eta value for a job no worker can do (FEED with no WHEAT anywhere)
HORIZON_TICKS = 6
LAND_PRICES = (1000, 2000, 4000)
SHED_CAPACITY = 100
_LOG1E5 = math.log1p(1e5)
_LOG1E4 = math.log1p(1e4)
KIND_COL, DEADLINE_COL = LIN, LIN + 1
WIDTH = LIN + 2
UNIT_WIDTH = 12


def _log_eta(e):
    return 1.0 if e is None else math.log1p(max(0, e)) / _LOG720


def _reach(e):
    return 1.0 if e is None else math.log1p(max(0, e)) / _LOG20


def _day_actions_left(t):
    """Current decision included; the terminal boundary has no action or dusk."""
    return max(0, min(DAY - t % DAY, LAST_STEP - t))


def _counter_eta(done_today, counter, left):
    """Visible ETA of the day end at which a watering/feeding counter reaches 2 with no further action."""
    if done_today:
        return left + 2 * DAY
    return left if counter >= 1 else left + DAY


def _decay_tick(t, mls):
    """First interpreter step s >= t with s >= mls and (s - mls) even."""
    s = max(t, mls)
    return s + ((s - mls) % 2)


def _plant_no_action(tile, t, cd):
    """This tile alone under the engine's rules with no further action, up to 3 day ends ahead:
    (visible ETA of becoming WEED or None, visible ETA of the first yield decay tick or None)."""
    y, c, w = tile["yield_units"], tile["consecutive_unwatered"], bool(tile["watered_today"])
    mls, pd, fud = tile["max_lifespan_step"], tile["planted_day"], tile["fertilized_until_day"]
    first_decay = None
    s, end = t, min(t + 3 * DAY, LAST_STEP - 1)
    while s <= end:
        if mls >= 0 and s >= mls and (s - mls) % 2 == 0:
            y -= 1
            first_decay = s + 1 - t if first_decay is None else first_decay
            if y <= 0:
                return s + 1 - t, first_decay
        if (s + 1) % DAY == 0:
            day, was = s // DAY, w
            c, w = (0 if w else c + 1), False
            if c >= 2:
                return s + 1 - t, first_decay
            if cd["ongoing"]:
                ds = day + 1 - pd - cd["first_yield_day"]
                if ds >= 0 and ds % cd["interval"] == 0 and ds // cd["interval"] + 1 <= cd["max_yield"]:
                    y = min(cd["max_yield"], y + (2 if was and fud >= day else 1))
                    if ds // cd["interval"] + 1 == cd["max_yield"]:
                        mls = (day + 2) * DAY
        nxt = s + DAY if (s + 1) % DAY == 0 else s + (DAY - 1 - s % DAY)
        if mls >= 0:
            d = max(s + 1, mls)
            nxt = min(nxt, d + (d - mls) % 2)
        s = nxt
    return None, first_decay


def plant_forecast(tile, t):
    """Water deadline (counter rule), no-action WEED/decay, next production if kept watered (no harvest)."""
    day, left = t // DAY, DAY - t % DAY
    cd = CROPS[tile["crop"]]
    pd, y, fud = tile["planted_day"], tile["yield_units"], tile["fertilized_until_day"]
    w, c, mls = bool(tile["watered_today"]), tile["consecutive_unwatered"], tile["max_lifespan_step"]
    death = _counter_eta(w, c, left)
    death = death if t + death <= LAST_STEP else None
    out = {"death": death, "tonight": death == left, "tomorrow": death == left + DAY}
    out["gone"], out["decay"] = _plant_no_action(tile, t, cd)
    out["decay_active"] = 0 <= mls <= t
    out["harvest_eta"] = max(0, (pd + cd["first_yield_day"]) * DAY - t)
    out["harvest_ready"] = out["harvest_eta"] == 0 and y > 0
    out["fert_days"] = max(0, fud - day + 1)
    out["at_cap"] = y >= cd["max_yield"]
    out["next_eta"] = out["next_units"] = None
    out["window"], out["growth"] = False, 0
    if cd["ongoing"]:
        nd = day + 1
        while nd <= LAST_DAY_END_NEXT_DAY:
            ds = nd - pd - cd["first_yield_day"]
            if ds >= 0 and ds % cd["interval"] == 0 and ds // cd["interval"] + 1 <= cd["max_yield"]:
                out["next_eta"] = nd * DAY - t
                out["next_units"] = min(cd["max_yield"], y + (2 if fud >= nd - 1 else 1)) - y
                break
            nd += 1
    else:
        ws, age = (cd["max_yield_day"] + 1) // 2, day - pd
        out["window"] = ws <= age <= cd["max_yield_day"]
        first = max(day + int(w), pd + ws)
        last = min(pd + cd["max_yield_day"], (LAST_STEP - 1) // DAY)
        gain = sum(2 if fud >= d else 1 for d in range(first, last + 1))
        out["growth"] = max(0, min(cd["max_yield"] - y, gain))
    return out


def animal_forecast(tile, t):
    """Feeding deadline (counter rule; escape) and the next product if fed every day from now on (no care)."""
    day, left = t // DAY, DAY - t % DAY
    a = ANIMALS[tile["animal"]]
    f, c, y = bool(tile["fed_today"]), tile["consecutive_unfed"], tile["yield_units"]
    death = _counter_eta(f, c, left)
    death = death if t + death <= LAST_STEP else None
    out = {"death": death, "tonight": death == left, "tomorrow": death == left + DAY,
           "next_eta": None, "next_units": None, "at_cap": y >= a["max_held"], "overflow": False,
           "decay": None, "decay_active": False}
    out["gone"] = death
    pcb = tile.get("pending_care_bonus", 0) or 0
    nd = day + 1
    while nd <= LAST_DAY_END_NEXT_DAY:
        ds = nd - tile["placed_day"] - a["first_yield_day"]
        if ds >= 0 and ds % a["interval"] == 0:
            bonus = pcb if nd == day + 1 else pcb + (1 if tile.get("cared_today") else 0)
            out["next_eta"] = nd * DAY - t
            out["next_units"] = min(a["max_held"], y + 1 + bonus) - y
            out["overflow"] = y + 1 + bonus > a["max_held"]
            break
        nd += 1
    return out


def _jobs(farm, t, seeds_known, has_seeds):
    """Per cell (y, x): (kind, deadline_eta, forecast dict or None)."""
    left = DAY - t % DAY
    out = {}
    for yy, line in enumerate(farm["tiles"]):
        for xx, tile in enumerate(line):
            kind, deadline, fc = NONE, 0, None
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                fc = plant_forecast(tile, t)
                decaying_soon = fc["harvest_ready"] and fc["decay"] is not None and fc["decay"] <= left
                if fc["tonight"]:
                    kind, deadline = WATER_TONIGHT, left
                elif decaying_soon:
                    kind, deadline = HARVEST_DECAYING, fc["decay"]
                elif fc["tomorrow"]:
                    kind, deadline = WATER_TODAY, left + DAY
                elif fc["harvest_ready"]:
                    kind = HARVEST
                elif fc["window"] and not tile["watered_today"]:
                    kind = WATER_TODAY
            elif isinstance(tile, dict) and "animal" in tile:
                fc = animal_forecast(tile, t)
                if fc["tonight"]:
                    kind, deadline = FEED_TONIGHT, left
                elif fc["tomorrow"]:
                    kind, deadline = FEED_TODAY, left + DAY
                elif tile["yield_units"] > 0:
                    kind = COLLECT
                elif tile.get("fertilizer_available"):
                    kind = FERTILIZER
                elif not tile.get("cared_today"):
                    kind = CARE
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                kind = WEED
            elif tile is None and (has_seeds or not seeds_known):
                kind = PLANT_EMPTY
            out[(yy, xx)] = (kind, deadline, fc)
    return out


def _workers(farm):
    return np.asarray([farm["farmer"], *farm["hands"]], np.int64).reshape(-1, 2)


def _dist(workers, cells):
    """[W, C] Manhattan distances; workers [W, 2] (x, y), cells [C, 2] (x, y)."""
    return np.abs(workers[:, None, 0] - cells[None, :, 0]) + np.abs(workers[:, None, 1] - cells[None, :, 1])


def _shed_dist(workers):
    return _dist(workers, np.asarray(SHED, np.int64)).min(1)


def farm_view(farm, t, wheat=None, seeds_known=True, has_seeds=True, shed_wheat=True):
    """Everything for one farm: jobs per cell, per-cell reach, per-worker summaries, duty counts. wheat=None
    (rival: inventories unknown) takes every worker as able to feed directly."""
    left = _day_actions_left(t)
    jobs = _jobs(farm, t, seeds_known, has_seeds)
    workers = _workers(farm)
    cells = np.asarray([(x, y) for (y, x) in jobs], np.int64)
    kinds = np.asarray([jobs[(y, x)][0] for (x, y) in cells], np.int64)
    d = _dist(workers, cells)                                  # [W, 100]
    eta = d + 1
    if wheat is None:
        wheat = np.ones(len(workers), bool)
    feed = np.isin(kinds, (FEED_TONIGHT, FEED_TODAY))
    shed = np.asarray(SHED, np.int64)
    via = (_dist(workers, shed)[:, :, None] + _dist(shed, cells)[None]).min(1) + 2   # pick up WHEAT, walk, feed
    if not shed_wheat:
        via = np.full_like(via, 999)
    job_eta = np.where(feed[None, :] & ~wheat[:, None], via, eta)
    via_feed = via
    tonight = np.isin(kinds, TONIGHT)
    min_eta = job_eta.min(0)
    cell_index = {(int(x), int(y)): i for i, (x, y) in enumerate(cells)}
    view = {"jobs": jobs, "cells": cells, "kinds": kinds, "eta": eta, "job_eta": job_eta, "min_eta": min_eta,
            "n_reach": (job_eta <= left).sum(0), "tonight": tonight, "cell_index": cell_index, "workers": workers,
            "wheat": wheat, "left": left, "via_feed": via_feed}
    w = len(workers)
    per_worker = []
    closest = job_eta == min_eta[None, :]
    for k in range(w):
        row = job_eta[k]
        tn = np.where(tonight)[0]
        near_tn = int(tn[row[tn].argmin()]) if len(tn) else None
        any_job = np.where(kinds != NONE)[0]
        near_any = row[any_job].min() if len(any_job) else None
        here = cell_index.get((int(workers[k, 0]), int(workers[k, 1])))
        fj = np.where(feed)[0]
        per_worker.append([
            _reach(None if near_tn is None else row[near_tn]),
            float(near_tn is not None and row[near_tn] <= left),
            float((row[tn] <= left).sum()) / 10 if len(tn) else 0.0,
            _reach(near_any),
            float(here is not None and kinds[here] in (WATER_TONIGHT, FEED_TONIGHT, HARVEST_DECAYING, WATER_TODAY,
                                                         FEED_TODAY, HARVEST, COLLECT, FERTILIZER)),
            float(near_tn is not None and bool(closest[k, near_tn])),
            _reach(int(_shed_dist(workers[k:k + 1])[0]) + 1),
            float(wheat[k]),
            _reach(row[fj].min() if len(fj) else None),
            float(((row <= 3) & (kinds != NONE)).sum()) / 10,
            float(k == 0),
            left / DAY,
        ])
    view["per_worker"] = np.asarray(per_worker, np.float32).reshape(-1, UNIT_WIDTH)
    counts = {k: int((kinds == k).sum()) for k in range(N_KINDS)}
    plants = animals = watered = fed = harvest_units = decay_units = at_cap = collect_units = fert = window = grow = 0
    for (yy, xx), (kind, deadline, fc) in jobs.items():
        tile = farm["tiles"][yy][xx]
        if fc is None:
            continue
        if tile.get("kind") == "PLANT":
            plants += 1
            watered += bool(tile["watered_today"])
            if fc["harvest_ready"]:
                harvest_units += tile["yield_units"]
                if fc["decay"] is not None and fc["decay"] <= DAY:
                    decay_units += tile["yield_units"]
            window += bool(fc["window"] and not tile["watered_today"])
            grow += fc["growth"]
        else:
            animals += 1
            fed += bool(tile["fed_today"])
            at_cap += bool(fc["at_cap"])
            collect_units += tile["yield_units"]
            fert += bool(tile.get("fertilizer_available"))
    duties = counts[WATER_TONIGHT] + counts[FEED_TONIGHT]
    infeasible = int((tonight & (min_eta > left)).sum())
    move = int(min_eta[tonight].sum()) if tonight.any() else 0
    view["farm"] = [plants / 100, counts[WATER_TONIGHT] / 100, counts[WATER_TODAY] / 100, watered / 100,
                    animals / 100, counts[FEED_TONIGHT] / 100, counts[FEED_TODAY] / 100, fed / 100,
                    harvest_units / 100, decay_units / 100, at_cap / 100, collect_units / 100, fert / 100,
                    w / 25, w * left / 600, math.log1p(duties / max(1, w * left)),
                    move / 200, infeasible / 100, window / 100, grow / 100, counts[HARVEST_DECAYING] / 100,
                    counts[PLANT_EMPTY] / 100]
    return view


def global_features(obs):
    t = int(obs["step"])
    day, left = t // DAY, _day_actions_left(t)
    shops = len(obs.get("town", {}).get("unlocked_shops") or [])
    unlock = None
    if shops < MAX_SHOPS:
        nd = day + 1
        while nd <= LAST_DAY_END_NEXT_DAY:
            if nd % SHOP_EVERY_DAYS == 0:
                unlock = nd * DAY - t
                break
            nd += 1
    s_shop = t + (-t) % SHOP_SELL_EVERY
    s_center = t + (-t) % CENTER_SELL_EVERY
    return [left / DAY, _log_eta(unlock), float(unlock is not None), (MAX_SHOPS - shops) / MAX_SHOPS,
            _log_eta(s_shop + 1 - t if s_shop <= LAST_STEP - 1 else None),
            _log_eta(s_center + 1 - t if s_center <= LAST_STEP - 1 else None), float(day == LAST_STEP // DAY)]


def _tile_row(kind, deadline, fc, tile, view, cell, left):
    row = np.zeros(TILE_COLS, np.float32)
    if fc is not None:
        row[0] = _log_eta(fc["death"])
        row[1], row[2] = float(fc["tonight"]), float(fc["tomorrow"])
        row[3] = _log_eta(fc["decay"])
        row[4] = float(fc["decay_active"])
        row[5] = _log_eta(fc["gone"])
        row[6] = _log_eta(fc["next_eta"])
        row[7] = 0.0 if fc["next_units"] is None else fc["next_units"] / 4
        if "harvest_eta" in fc:
            row[8] = _log_eta(fc["harvest_eta"])
            row[9] = float(fc["harvest_ready"])
            row[10], row[11] = float(fc["window"]), fc["growth"] / 6
            row[12] = float(fc["fert_days"] > 0)
            row[13] = min(fc["fert_days"], 3) / 3
        row[14] = float(fc["at_cap"])
        row[15] = float(fc.get("overflow", False))
    i = view["cell_index"][cell]
    m = int(view["min_eta"][i])
    row[16] = _reach(m)
    row[17] = float(view["n_reach"][i]) / 10
    if kind in TONIGHT:
        row[18] = float(m <= left)
        row[19] = (left - m) / DAY
    row[20 + kind] = 1.0
    return row


def np_sign(v):
    return (v > 0) - (v < 0)


def _signed_log(v, scale=_LOG1E5):
    return math.copysign(math.log1p(abs(v)), v) / scale if v else 0.0


def hire_cost(hires_today, mult=1):
    a, b = 1, 1
    for _ in range(hires_today):
        a, b = b, a + b
    return mult * a


def overflow_tonight(obs):
    """Own carried units the next day-end deposit discards if nothing changes: max(0, carried - shed room); exact in
    total whatever the item order (per-item survival depends on the carrier's pickup order). The final step has no
    further night: whatever is still carried then is never deposited or sold."""
    t = int(obs["step"])
    private = obs.get("private") or {}
    carried = sum(sum((inv or {}).values()) for inv in (private.get("inventories") or []))
    room = SHED_CAPACITY - sum((private.get("shed") or {}).values())
    night = (t // DAY + 1) * DAY - 1 <= LAST_STEP - 1
    return [max(0, carried - room) / 100 if night else 0.0, float(not night) * carried / 100]


def economics(obs, role, mult=1):
    """Precision-safe boundaries, computed on raw integers before any scaling (FARM token of `role`)."""
    seat = int(obs["player"])
    me, other = obs["farms"][seat if role == 0 else 1 - seat], obs["farms"][1 - seat if role == 0 else seat]
    cash, lead = float(me["money"]), float(me["money"]) - float(other["money"])
    cost = hire_cost(int(me["hires_today"]), mult)
    n_aff, spent, k = 0, 0.0, int(me["hires_today"])
    while n_aff < 25 and spent + hire_cost(k, mult) <= cash:
        spent += hire_cost(k, mult)
        n_aff, k = n_aff + 1, k + 1
    owned = len(me.get("unlocked_quadrants") or [])
    land = LAND_PRICES[owned - 1] if 1 <= owned <= len(LAND_PRICES) else None
    row = [_signed_log(lead), float(lead > 0), float(lead == 0), math.log1p(cost) / _LOG1E4, float(cash >= cost),
           _signed_log(cash - cost), n_aff / 10, 1.0 if land is None else math.log1p(land) / _LOG1E4,
           float(land is not None and cash >= land), 0.0 if land is None else _signed_log(cash - land), 0.0, 0.0, 0.0]
    if role == 0:
        private = obs.get("private") or {}
        shed = sum((private.get("shed") or {}).values())
        carried = sum(sum((inv or {}).values()) for inv in (private.get("inventories") or []))
        seeds = sum((private.get("seeds") or {}).values())
        empty = sum(tile is None for line in me["tiles"] for tile in line)
        row[10], row[11], row[12] = (SHED_CAPACITY - shed) / SHED_CAPACITY, carried / 100, (seeds - empty) / 100
    return row


def _market_params(obs, configuration):
    params = (obs.get("market") or {}).get("params")
    if params is not None:
        return params
    over = (configuration or {}).get("marketParams") or {}
    return {item: {**MARKET_PARAMS[item], **over.get(item, {})} for item in PRODUCTS}


def price_path(obs, configuration=None, ticks=HORIZON_TICKS):
    """Known no-trade prefix, stopping before an unrevealed shop can consume."""
    t = int(obs["step"])
    market = obs["market"]
    params = _market_params(obs, configuration)
    inv = {item: int(market["inventory"][item]) for item in PRODUCTS}
    shops = (obs.get("town") or {}).get("unlocked_shops") or []
    out = {item: [] for item in PRODUCTS}
    s = t + (-t) % SHOP_SELL_EVERY
    known_until = LAST_STEP
    if len(shops) < MAX_SHOPS:
        next_unlock_day = (t // DAY // SHOP_EVERY_DAYS + 1) * SHOP_EVERY_DAYS
        if next_unlock_day <= LAST_DAY_END_NEXT_DAY:
            known_until = min(known_until, next_unlock_day * DAY)
    while s < known_until and len(out[PRODUCTS[0]]) < ticks:
        for shop in shops:
            products = SHOPS[shop]
            for item in products:
                inv[item] -= 2 if len(products) == 1 else 1
        if s % CENTER_SELL_EVERY == 0:
            for item in PRODUCTS:
                if item != "FERTILIZER":
                    inv[item] -= 1
        for item in PRODUCTS:
            out[item].append((s + 1 - t, price_at(item, inv[item], params)))
        s += SHOP_SELL_EVERY
    return out, params


def market_row(obs, item, path, params):
    now = int(obs["market"]["prices"][item])
    base = max(1.0, float(params[item]["base"]))
    steps = path[item]
    row = [0.0] * (LIN - M0)
    for k, (_, price) in enumerate(steps[:HORIZON_TICKS]):
        row[k] = (price - now) / base
    change = next((eta for eta, price in steps if price != now), None)
    row[6] = _log_eta(change)
    p = params[item]
    row[7] = (int(obs["market"]["inventory"][item]) - p["I0"]) / max(1.0, float(p["T"]))
    row[8] = float(np_sign(steps[0][1] - now)) if steps else 0.0
    return row


def job_bits(tile, t, own, has_seeds, fc):
    """Every action that would change this tile now (engine _apply_unit_action conditions), not one priority code; public
    conditions only, except PLANT on our own farm (seeds held). 12 columns."""
    day = t // DAY
    b = [0.0] * (S0 - J0)
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        b[0] = float(not tile["watered_today"])
        b[1] = float(bool(fc and fc["harvest_ready"]))
        b[6] = float(tile["fertilized_until_day"] < day + 2)
        b[8] = 1.0
    elif isinstance(tile, dict) and "animal" in tile:
        b[2] = float(not tile["fed_today"])
        b[3] = float(tile["yield_units"] > 0)
        b[4] = float(bool(tile.get("fertilizer_available")))
        b[5] = float(not tile.get("cared_today"))
    elif isinstance(tile, dict):
        b[8] = 1.0                                         # WEED or an empty COOP/PASTURE: DIG
    elif tile is None:
        b[7] = float(has_seeds or not own)                 # empty owned land: PLANT (rival seeds unknown)
    n = sum(b[:9])
    b[9], b[10] = n / 5, float(n >= 2)
    return b


def startup_rows(obs, farm, has_seeds):
    """{(x, y): [certified, slack/24, log-ETA, seeds held]} for empty owned tiles: the fastest PLANT then WATER by two
    present workers before tonight's refresh (planter u, waterer v; a later-index v may water in the same step). Exact for
    otherwise idle workers and one held seed; a timing fact, not a statement that the workers are free."""
    t = int(obs["step"])
    left = min((t // DAY + 1) * DAY - 1, LAST_STEP - 1) - t + 1
    refresh = (t // DAY + 1) * DAY - 1 <= LAST_STEP - 1
    cells = [(x, y) for y, line in enumerate(farm["tiles"]) for x, tile in enumerate(line) if tile is None]
    if not cells:
        return {}
    pos = np.asarray([farm["farmer"], *farm["hands"]], np.int64).reshape(-1, 2)
    xy = np.asarray(cells, np.int64)
    d = np.abs(pos[:, None, 0] - xy[None, :, 0]) + np.abs(pos[:, None, 1] - xy[None, :, 1])      # [U, T]
    before = np.vstack([np.full((1, d.shape[1]), 10 ** 6), np.minimum.accumulate(d, axis=0)[:-1]])  # min over u < v
    after = np.minimum.accumulate(d[::-1], axis=0)[::-1]                                          # min over u >= v
    best = 1 + np.min(np.maximum(d, np.minimum(before, after + 1)), axis=0)                       # 1 + max(dv, du + [v <= u])
    out = {}
    for (x, y), c in zip(cells, best.tolist()):
        ok = bool(refresh and has_seeds and c <= left)
        out[(x, y)] = [float(ok), max(-1.0, min(1.0, (left - c) / DAY)), _log_eta(c), float(has_seeds)]
    return out


def reach_matrix(view, farm, max_units):
    """[max_units, 100] int16, cell = y*10+x: decisions for each own worker to act on each own tile; FEED on an unfed
    animal needs WHEAT (carried, else a shed pickup), NO_REACH when there is none; 0 for absent units."""
    out = np.zeros((max_units, 100), np.int16)
    eta = view["eta"].copy()
    feedable = np.asarray([isinstance(farm["tiles"][y][x], dict) and "animal" in farm["tiles"][y][x]
                           and not farm["tiles"][y][x]["fed_today"] for (x, y) in view["cells"]], bool)
    via = view["via_feed"]
    job = np.where(feedable[None, :] & ~view["wheat"][:, None], via, eta)
    k = min(max_units, len(job))
    out[:k] = np.minimum(job[:k], NO_REACH)
    return out


def token_features(obs, token_ids, max_units=25, configuration=None):
    """(extra [N, WIDTH] float16, unit_extra [max_units, UNIT_WIDTH] float16, reach_eta [max_units, 100] int16) for
    observation.pack token_ids."""
    t = int(obs["step"])
    seat = int(obs["player"])
    left = _day_actions_left(t)
    private = obs.get("private") or {}
    farms = (obs["farms"][seat], obs["farms"][1 - seat])
    inv = private.get("inventories") or []
    own_wheat = np.asarray([bool((inv[k] or {}).get("WHEAT", 0) > 0) if k < len(inv) else False
                            for k in range(1 + len(farms[0]["hands"]))], bool)
    has_seeds = any(v > 0 for v in (private.get("seeds") or {}).values())
    shed_wheat = (private.get("shed") or {}).get("WHEAT", 0) > 0
    views = (farm_view(farms[0], t, own_wheat, True, has_seeds, shed_wheat), farm_view(farms[1], t, None, False, True))
    start = startup_rows(obs, farms[0], has_seeds)
    extra = np.zeros((len(token_ids), WIDTH), np.float32)
    g = global_features(obs)
    path, params = price_path(obs, configuration)
    mult = int((configuration or {}).get("farmHandCostMult", 1) or 1)
    for n, tid in enumerate(token_ids):
        if tid == "global":
            extra[n, G0:G0 + len(g)] = g
        elif tid.startswith("farm/"):
            f = views[int(tid[5])]["farm"]
            extra[n, F0:F0 + len(f)] = f
            e = economics(obs, int(tid[5]), mult)
            extra[n, E0:E0 + len(e)] = e
            if tid == "farm/0":
                extra[n, N0:LIN] = overflow_tonight(obs)
        elif tid.startswith("market/"):
            m = market_row(obs, tid[7:], path, params)
            extra[n, M0:M0 + len(m)] = m
        elif tid.startswith("tile/"):
            role = int(tid[5])
            x, y = (int(v) for v in tid[7:].split(","))
            kind, deadline, fc = views[role]["jobs"][(y, x)]
            tile = farms[role]["tiles"][y][x]
            extra[n, T0:C0] = _tile_row(kind, deadline, fc, tile, views[role], (x, y), left)
            extra[n, J0:S0] = job_bits(tile, t, role == 0, has_seeds, fc)
            if role == 0 and (x, y) in start:
                extra[n, S0:N0] = start[(x, y)]
            extra[n, KIND_COL] = kind
            extra[n, DEADLINE_COL] = deadline
        elif tid.startswith("carrier/"):
            role = int(tid[8])
            k = int(tid[10:])
            pw = views[role]["per_worker"]
            if k < len(pw):
                extra[n, C0:C0 + UNIT_WIDTH] = pw[k]
    unit_extra = np.zeros((max_units, UNIT_WIDTH), np.float32)
    pw = views[0]["per_worker"][:max_units]
    unit_extra[:len(pw)] = pw
    return extra.astype(np.float16), unit_extra.astype(np.float16), reach_matrix(views[0], farms[0], max_units)
