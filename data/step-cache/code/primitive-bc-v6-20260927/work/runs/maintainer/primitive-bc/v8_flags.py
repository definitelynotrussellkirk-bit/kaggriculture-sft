"""v8 flags: calendar tokens, shop-reveal timeline + demand belief, action-effect receipts, a short net product ledger
and binary state flags (owner 2026-09-28: "we could also add some extra binary flags and the like").

Standalone feature family (pbc-features-v8/CONTRACT.md). Pure function of this seat's actor observation at step t and
`history` = its own previous actor observations (most recent first, at most 2 used). No rival private state, no future,
no replay label, no own submitted action (the model gets those separately as v6 prev_unit/prev_order tokens).

Engine rules (src/farm_neat/game/kaggriculture_engine_reference.py, "E Lnnn"):
  * interpreter step s: own unit actions in worker order (farmer, then hands) E L935-939, then the market E L941
    (HIRE/BUY_LAND atomic, per-unit lockstep SELL/BUY), town consumption E L942 (shops when s % 4 == 0 E L736, town centre
    when s % 24 == 0 E L745), plant decay E L944, day end when (s + 1) % 24 == 0 E L945 (plants/animals refresh, weeds,
    carried goods deposited with overflow discarded, hands dismissed, farmer respawned, hires_today reset E L873-882, a shop
    appended when (day + 1) % 3 == 0 while fewer than 8 E L884-891). Actions exist for s = 0..718 E L960; last day end s = 695.
  * one action per worker per step; every tile action acts on the standing tile E L337, E L414; inventories are per worker
    E L292-296, so an inventory change is attributable to exactly one worker; tile-only effects (PLANT, BUILD, DIG, WATER,
    CARE) are attributable only when one unexplained worker stood on that tile.
  * shop k (0-based, in town.unlocked_shops order) is revealed at the day end of day 3(k+1)-1, i.e. visible from step
    72(k+1): every third day end appends exactly one shop until 8 (E L884-891); a draw is uniform over the 8 shop types
    (rng.choice(sorted(SHOPS)) E L891); a one-product shop consumes 2 per tick, others 1 per product (E L740-743).

compute(actor_obs, configuration, token_ids, history=()) -> {"token": [N, WIDTH_TOKEN] f16, "unit": [25, WIDTH_UNIT] f16}.
PROD_COLS go on market/<PRODUCT> tokens (first N_PROD columns), GLOB_COLS on "global" (last N_GLOB columns), zeros
elsewhere; UNIT_COLS rows follow codec.unit_positions (row 0 = farmer, then hands).
"""
from __future__ import annotations

import math

import numpy as np

from farm_neat.economy.heuristic.constants import ANIMALS, CROPS, LAND_PRICES, MARKET_PARAMS, PRODUCTS, SHOPS, fib
from farm_neat.economy.heuristic.market import price_at

DAY = 24
LAST_ACTION_STEP = 718                  # E L960: the step-718 actions are the last ones applied
LAST_DUSK_STEP = 695                    # (s + 1) % 24 == 0 with s <= 718
LAST_DAY = 29
MAX_UNITS = 25
MAX_SHOPS = 8
SHED_ACCESS = frozenset({(4, 4), (5, 4), (4, 5), (5, 5)})     # E L132-135 for boardSize 10
PRODUCER_OF = {"WHEAT": "WHEAT", "CARROT": "CARROT", "TOMATO": "TOMATO", "STRAWBERRY": "STRAWBERRY",
               "MELON": "MELON", "EGG": "GOOSE", "MILK": "COW", "WOOL": "SHEEP"}
BUYABLE = ("WHEAT", "FERTILIZER")      # BUY_PRODUCT E L598
SHOP_TYPES = tuple(sorted(SHOPS))
SHOP_MULT = {s: {p: (2 if len(ps) == 1 else 1) for p in ps} for s, ps in SHOPS.items()}   # E L741

PROD_COLS = (
    # H5/C1 net product ledger: last step (t-1 -> t) and last two steps (t-2 -> t); same-day steps only
    "lg1_produced", "lg1_consumed", "lg1_bought", "lg1_sold",
    "lg2_produced", "lg2_consumed", "lg2_bought", "lg2_sold",
    "lg1_seeds_bought",
    # state flags at current quotes
    "can_buy_producer", "n_producer_affordable", "can_buy_unit", "free_structure",
    # S8 season boundary per producer
    "producer_can_yield", "producer_full_schedule", "producer_pulses_left",
    # H7 shop-reveal timeline + C10 supported demand belief
    "shop_src_count", "shop_src_none", "shop_src_newest_age", "shop_src_first_age", "shop_src_new_last_unlock",
    "dem_p_new_source", "dem_future_hi", "dem_future_sd", "dem_supported_share",
)
GLOB_COLS = (
    # S8 calendar
    "cal_hour_sin", "cal_hour_cos", "cal_phase_night", "cal_phase_morning", "cal_phase_afternoon", "cal_phase_evening",
    "cal_dawn", "cal_dusk", "cal_final_action", "cal_no_dusk_ahead", "cal_day_ends_left", "cal_shop_unlock_tonight",
    "cal_shop_tick_now", "cal_center_tick_now",
    # H3 receipts, whole seat, last step
    "rc_valid", "rc2_valid", "rc_any_idle", "rc_n_idle", "rc_n_effect", "rc_n_ambiguous", "rc_n_new_hands",
    "rc_land_bought", "rc_goose_bought", "rc_cow_bought", "rc_sheep_bought", "rc_seeds_bought",
    "rc_cash_delta", "rc_trade_cash", "rc_hire_cash",
    # H5/C1 ledger, whole seat
    "lg_exact", "lg_discard", "lg2_trade_cash", "lg2_cash_delta", "labour_spent_today",
    # binary state flags
    "can_afford_hire", "can_afford_land", "q4_owned", "q4_affordable", "shed_full",
    "any_animal_unfed", "any_plant_unwatered", "any_loss_tonight",
)
UNIT_COLS = (
    "u_valid", "u_new_hand", "u_effect", "u_no_effect", "u_ambiguous", "u_moved",
    "u_plant", "u_water", "u_harvest", "u_feed", "u_care", "u_fertilize", "u_collect_fert", "u_place_animal",
    "u_build", "u_dig", "u_pickup", "u_deposit", "u_other_effect", "u_load_delta", "u_idle_streak", "u_undo_move",
)
N_PROD, N_GLOB = len(PROD_COLS), len(GLOB_COLS)
TOKEN_COLS = PROD_COLS + GLOB_COLS
WIDTH_TOKEN, WIDTH_UNIT = len(TOKEN_COLS), len(UNIT_COLS)
_P = {n: i for i, n in enumerate(PROD_COLS)}
_G = {n: i for i, n in enumerate(GLOB_COLS)}
_U = {n: i for i, n in enumerate(UNIT_COLS)}
_ROW = {p: i for i, p in enumerate(PRODUCTS)}

# per-unit effect codes (one action per worker per step)
MOVE, PLANT, WATER, HARVEST, FEED, CARE, FERTILIZE, COLLECT, PLACE_ANIMAL, BUILD, DIG, PICKUP, DEPOSIT, OTHER = range(14)
_CODE_COL = {MOVE: "u_moved", PLANT: "u_plant", WATER: "u_water", HARVEST: "u_harvest", FEED: "u_feed",
             CARE: "u_care", FERTILIZE: "u_fertilize", COLLECT: "u_collect_fert", PLACE_ANIMAL: "u_place_animal",
             BUILD: "u_build", DIG: "u_dig", PICKUP: "u_pickup", DEPOSIT: "u_deposit", OTHER: "u_other_effect"}


def slog(x, scale):
    return math.copysign(math.log1p(abs(x) / scale), x)


# ------------------------------------------------------------------ helpers
def _cfg(configuration):
    c = configuration or {}
    return {"shed": int(c.get("shedCapacity", 100) or 100), "mult": int(c.get("farmHandCostMult", 1) or 1),
            "unlock": max(1, int(c.get("townShopUnlockInterval", 3) or 3))}


def _params(obs, configuration):
    params = (obs.get("market") or {}).get("params")
    if params is not None:
        return params
    over = (configuration or {}).get("marketParams") or {}
    return {item: {**MARKET_PARAMS[item], **over.get(item, {})} for item in PRODUCTS}


def _own(obs):
    seat = int(obs["player"])
    return obs["farms"][seat], obs.get("private") or {}


def _positions(farm):
    return [tuple(farm["farmer"]), *(tuple(h) for h in farm.get("hands") or [])]


def _is_plant(t):
    return isinstance(t, dict) and t.get("kind") == "PLANT"


def _is_animal(t):
    return isinstance(t, dict) and "animal" in t


def _same_plant(a, b):
    return (_is_plant(a) and _is_plant(b) and a.get("crop") == b.get("crop")
            and a.get("planted_day") == b.get("planted_day"))


def _same_animal(a, b):
    return (_is_animal(a) and _is_animal(b) and a.get("animal") == b.get("animal")
            and a.get("placed_day") == b.get("placed_day"))


def _product(t):
    if _is_plant(t):
        return t.get("crop")
    if _is_animal(t):
        return ANIMALS[t["animal"]]["product"]
    return None


def _consecutive(prev, cur):
    """prev is the same seat's observation of step t-1 (clock-consecutive)."""
    return (isinstance(prev, dict) and prev.get("step") is not None and cur.get("step") is not None
            and int(prev["step"]) + 1 == int(cur["step"]) and prev.get("player") == cur.get("player"))


def _same_day_pair(prev, cur):
    """A t-1 -> t step with no day end in between (E L945: the day end at s with (s+1) % 24 == 0 resets hands,
    inventories, hires and tile day flags), so workers and inventories are comparable."""
    return _consecutive(prev, cur) and int(cur["step"]) % DAY != 0


def _classify_inventory(dinv, xy, tp, tc, day):
    """(code, alternative or None) for one non-moving worker whose inventory changed (engine _apply_unit_action
    E L312-530). Off the shed-access tiles an inventory change has one engine cause per item type (PICKUP, DROP and
    PLACE-to-shed need shed access, E L344, L359, L394); on them the tile change decides, and an alternative is given
    when two engine actions explain the same change."""
    pos = {i: d for i, d in dinv.items() if d > 0}
    neg = {i: -d for i, d in dinv.items() if d < 0}
    shed = xy in SHED_ACCESS
    if pos and not neg and len(pos) == 1:
        (item, n), = pos.items()
        y0 = int(tp.get("yield_units", 0) or 0) if isinstance(tp, dict) else 0
        if _product(tp) == item and y0 > 0:                       # E L446-473
            if not shed or (n >= y0 and not ((_same_plant(tp, tc) or _same_animal(tp, tc))
                                             and int(tc.get("yield_units", 0) or 0) > 0)):
                return HARVEST, None
        if item == "FERTILIZER" and n == 1 and _is_animal(tp) and tp.get("fertilizer_available"):   # E L515-522
            if not shed or (_same_animal(tp, tc) and not tc.get("fertilizer_available")):
                return COLLECT, None
        return (PICKUP if shed else OTHER), None                  # E L358-375
    if neg and not pos:
        if len(neg) == 1:
            (item, n), = neg.items()
            if item == "WHEAT" and n == 1 and (_is_animal(tp) or _is_animal(tc)):                  # E L505-513
                # the animal may have been placed by an earlier worker in this step (E L935-939 worker order)
                if not shed or (_is_animal(tc) and tc.get("fed_today")
                                and not (_same_animal(tp, tc) and tp.get("fed_today"))):
                    return FEED, None
            if item == "FERTILIZER" and n == 1 and _is_plant(tp):                                   # E L475-482
                if not shed:
                    return FERTILIZE, None
                if _same_plant(tp, tc) and int(tc.get("fertilized_until_day", -1)) > int(tp.get("fertilized_until_day", -1)):
                    return FERTILIZE, None
                if not _same_plant(tp, tc) or int(tc.get("fertilized_until_day", -1)) >= day + 2:
                    return FERTILIZE, DEPOSIT                     # re-fertilized or harvested after: no tile evidence
            if item in ANIMALS and n == 1:                         # E L384-392 (structure may be built this step)
                if not shed or (_is_animal(tc) and tc.get("animal") == item and not _same_animal(tp, tc)):
                    return PLACE_ANIMAL, None
        return (DEPOSIT if shed else OTHER), None                 # DROP E L343-356 / PLACE-to-shed E L394-409
    return OTHER, None


# transitions that can happen at most once per tile per step: FEED needs not fed_today (E L508), COLLECT needs
# fertilizer_available (E L518), HARVEST needs yield > 0 (E L449), PLACE needs a structure without animal (E L388)
_EXCLUSIVE = {FEED: DEPOSIT, COLLECT: PICKUP, HARVEST: PICKUP, PLACE_ANIMAL: DEPOSIT}


def _tile_events(tp, tc, day, harvest_gain):
    """(events, hidden) for one tile between t-1 and t, for workers with no inventory change: PLANT E L417-429,
    WATER E L431-444, DIG E L484-491, BUILD E L493-503, CARE E L524-530. harvest_gain = None if no worker harvested
    here, else units harvested minus the t-1 yield (> 0 proves a WATER before the harvest, E L440-443). hidden: an
    unwatered plant left the tile (harvest or decay to WEED, E L464-468, E L764-766), so a WATER may be invisible."""
    if tp == "LOCKED":
        return [], False
    harvested_here = harvest_gain is not None
    ev = []
    new_plant = _is_plant(tc) and tc.get("planted_day") == day and not _same_plant(tp, tc)
    new_structure = (isinstance(tc, dict) and tc.get("kind") in ("COOP", "PASTURE")
                     and not (isinstance(tp, dict) and tp.get("kind") == tc.get("kind")))
    removed = tp is not None and not harvested_here and not _is_animal(tp)
    if new_plant or new_structure:
        if removed and not (new_plant and _same_plant(tp, tc)):
            ev.append(DIG)
        ev.append(PLANT if new_plant else BUILD)
    elif tc is None and removed:
        ev.append(DIG)
    # a later worker acts on the tile an earlier worker left (E L935-939), e.g. PLANT then WATER in one step
    if _is_plant(tc) and tc.get("watered_today") and not (_same_plant(tp, tc) and tp.get("watered_today")):
        ev.append(WATER)
    if _is_animal(tc) and tc.get("cared_today") and not (_same_animal(tp, tc) and tp.get("cared_today")):
        ev.append(CARE)
    hidden = _is_plant(tp) and not tp.get("watered_today") and not _same_plant(tp, tc) and DIG not in ev
    if hidden and harvested_here and harvest_gain > 0 and WATER not in ev:
        ev.append(WATER)
        hidden = False
    return ev, hidden


def analyse_step(prev, cur, configuration=None):
    """One same-day step prev (t-1) -> cur (t) of one seat, or None when not comparable.

    Returns {"codes": per worker alive at t-1: list of effect codes ([] = no visible effect), "ambiguous": per worker
    bool, "n_prev": workers at t-1, "load_delta": per worker carried-unit change, and per-item flows (dicts):
    produced, consumed, picked, received (into the shed by deposits), market_net (shed change not explained by
    workers: + bought, - sold), seeds_bought; scalars: discard, exact, cash_delta, trade_cash, hire_cash, land_bought,
    hired, seed_cash, animal_cash}."""
    if not _same_day_pair(prev, cur):
        return None
    cfg = _cfg(configuration)
    fp, pp = _own(prev)
    fc, pc = _own(cur)
    day = int(prev["step"]) // DAY
    tiles_p, tiles_c = fp["tiles"], fc["tiles"]
    pos_p, pos_c = _positions(fp), _positions(fc)
    inv_p = list(pp.get("inventories") or [])
    inv_c = list(pc.get("inventories") or [])
    n_prev = len(pos_p)
    codes = [[] for _ in range(n_prev)]
    ambiguous = [False] * n_prev
    effect = [False] * n_prev
    load_delta = [0] * n_prev
    unexplained, inv_units = {}, {}
    exact = True
    for k in range(min(n_prev, len(pos_c))):
        a = inv_p[k] if k < len(inv_p) and isinstance(inv_p[k], dict) else {}
        b = inv_c[k] if k < len(inv_c) and isinstance(inv_c[k], dict) else {}
        load_delta[k] = sum(b.values()) - sum(a.values())
        if pos_c[k] != pos_p[k]:
            codes[k].append(MOVE)                                 # E L323-332; a move never changes inventory
            effect[k] = True
            continue
        dinv = {i: b.get(i, 0) - a.get(i, 0) for i in set(a) | set(b) if b.get(i, 0) != a.get(i, 0)}
        x, y = pos_p[k]
        if not dinv:
            unexplained.setdefault((x, y), []).append(k)
            continue
        code, alt = _classify_inventory(dinv, (x, y), tiles_p[y][x], tiles_c[y][x], day)
        inv_units[k] = [code, alt, dinv, (x, y), bool(b)]
    claims = {}
    for k, v in inv_units.items():
        if v[0] in _EXCLUSIVE:
            claims.setdefault((v[3], v[0]), []).append(k)
    for (xy, code), ks in claims.items():
        if len(ks) < 2:
            continue
        alt = _EXCLUSIVE[code] if xy in SHED_ACCESS else OTHER   # the others moved goods to/from the shed
        if len({tuple(sorted(inv_units[k][2].items())) for k in ks}) > 1:
            exact = False                                         # which worker did it changes the per-item flows
        for j, k in enumerate(ks):
            inv_units[k][0], inv_units[k][1] = (code, alt) if j == 0 else (alt, code)
    produced, consumed, picked, deposits, harvest_gain = {}, {}, {}, [], {}
    for k, (code, alt, dinv, xy, kept) in inv_units.items():
        codes[k] = [code] if alt is None else sorted({code, alt})
        ambiguous[k] = alt is not None
        effect[k] = True
        exact = exact and not (alt is not None and code not in _EXCLUSIVE and alt not in _EXCLUSIVE)
        if code == HARVEST:
            tp = tiles_p[xy[1]][xy[0]]
            harvest_gain[xy] = sum(dinv.values()) - (int(tp.get("yield_units", 0) or 0) if isinstance(tp, dict) else 0)
            for item, n in dinv.items():
                produced[item] = produced.get(item, 0) + n
        elif code == COLLECT:
            produced["FERTILIZER"] = produced.get("FERTILIZER", 0) + 1
        elif code == FEED:
            consumed["WHEAT"] = consumed.get("WHEAT", 0) + 1
        elif code == FERTILIZE:
            consumed["FERTILIZER"] = consumed.get("FERTILIZER", 0) + 1
        elif code == PICKUP:
            for item, n in dinv.items():
                picked[item] = picked.get(item, 0) + n
            deposits.append((k, "pickup", dict(dinv), True))
        elif code == DEPOSIT:
            deposits.append((k, "deposit", {i: -d for i, d in dinv.items()}, kept))
        elif code == OTHER:
            exact = False
    for (x, y), ks in unexplained.items():
        ev, hidden = _tile_events(tiles_p[y][x], tiles_c[y][x], day, harvest_gain.get((x, y)))
        if hidden:                                                # a WATER may have left no trace: unknown
            for k in ks:
                ambiguous[k] = True
                codes[k].extend(e for e in ev if e not in codes[k])
            continue
        if not ev:
            continue
        if len(ks) == 1 and len(ev) == 1:
            codes[ks[0]].append(ev[0])
            effect[ks[0]] = True
        else:
            for k in ks:
                ambiguous[k] = True
                codes[k].extend(e for e in ev if e not in codes[k])
                effect[k] = len(ev) >= len(ks)                  # every co-located worker did something
    # shed flows, worker order (E L935-939): pickups free room, deposits fill it; DROP discards the rest (E L346-355)
    shed_p = {i: int(v) for i, v in (pp.get("shed") or {}).items()}
    shed_c = {i: int(v) for i, v in (pc.get("shed") or {}).items()}
    room = cfg["shed"] - sum(shed_p.values())
    received, discard = {}, 0
    for k, kind, items, kept in sorted(deposits, key=lambda d: d[0]):
        total = sum(items.values())
        if kind == "pickup":
            room += total
            continue
        take = total if kept else min(total, max(0, room))      # PLACE keeps what does not fit; DROP discards it
        if take < total:
            discard += total - take
            exact = False
            left = take
            for i in sorted(items):                              # insertion order unknown: sorted-item approximation
                got = min(items[i], left)
                received[i] = received.get(i, 0) + got
                left -= got
        else:
            for i, n in items.items():
                received[i] = received.get(i, 0) + n
        room -= take
    items = set(shed_p) | set(shed_c) | set(received) | set(picked)
    market_net = {i: shed_c.get(i, 0) - shed_p.get(i, 0) - received.get(i, 0) + picked.get(i, 0) for i in items}
    market_net = {i: v for i, v in market_net.items() if v}
    # seeds: bought = change + planted (E L425-428: PLANT consumes one seed; seeds never pass through the shed)
    planted = {}
    for yy in range(len(tiles_c)):
        rc, rp = tiles_c[yy], tiles_p[yy]
        for xx in range(len(rc)):
            t = rc[xx]
            if _is_plant(t) and t.get("planted_day") == day and not _same_plant(rp[xx], t):
                planted[t["crop"]] = planted.get(t["crop"], 0) + 1
    sp, sc = pp.get("seeds") or {}, pc.get("seeds") or {}
    seeds_bought = {c: int(sc.get(c, 0)) - int(sp.get(c, 0)) + planted.get(c, 0) for c in CROPS}
    seeds_bought = {c: v for c, v in seeds_bought.items() if v}
    hired = max(0, int(fc.get("hires_today", 0)) - int(fp.get("hires_today", 0)))
    hire_cash = cfg["mult"] * sum(fib(i) for i in range(int(fp.get("hires_today", 0)), int(fp.get("hires_today", 0)) + hired))
    q_p, q_c = len(fp.get("unlocked_quadrants") or []), len(fc.get("unlocked_quadrants") or [])
    land_cash = sum(LAND_PRICES[j] for j in range(max(0, q_p - 1), max(0, q_c - 1)) if j < len(LAND_PRICES))
    seed_cash = sum(CROPS[c]["seed"] * n for c, n in seeds_bought.items())
    animal_cash = sum(ANIMALS[a]["cost"] * market_net.get(a, 0) for a in ANIMALS)
    cash_delta = float(fc["money"]) - float(fp["money"])
    trade_cash = cash_delta + seed_cash + animal_cash + hire_cash + land_cash      # = product receipts - WHEAT/FERT cost
    return {"codes": codes, "ambiguous": ambiguous, "effect": effect, "n_prev": n_prev, "load_delta": load_delta,
            "pos_prev": pos_p, "produced": produced, "consumed": consumed, "picked": picked, "received": received,
            "market_net": market_net, "seeds_bought": seeds_bought, "discard": discard, "exact": exact,
            "cash_delta": cash_delta, "trade_cash": trade_cash, "hire_cash": hire_cash, "land_bought": q_c - q_p,
            "hired": hired, "seed_cash": seed_cash, "animal_cash": animal_cash}


# ------------------------------------------------------------------ blocks
def _calendar(t, n_shops, cfg):
    day, hour = divmod(t, DAY)
    g = {}
    ang = 2.0 * math.pi * hour / DAY
    g["cal_hour_sin"], g["cal_hour_cos"] = math.sin(ang), math.cos(ang)
    g["cal_phase_night"], g["cal_phase_morning"] = float(hour < 6), float(6 <= hour < 12)
    g["cal_phase_afternoon"], g["cal_phase_evening"] = float(12 <= hour < 18), float(hour >= 18)
    g["cal_dawn"] = float(hour == 0)
    g["cal_dusk"] = float((t + 1) % DAY == 0 and t <= LAST_DUSK_STEP)
    g["cal_final_action"] = float(t == LAST_ACTION_STEP)
    g["cal_no_dusk_ahead"] = float(t > LAST_DUSK_STEP)
    first = max(0, -(-(t - (DAY - 1)) // DAY))                  # first day d with 24d + 23 >= t
    g["cal_day_ends_left"] = max(0, LAST_DUSK_STEP // DAY - first + 1) / LAST_DAY
    g["cal_shop_unlock_tonight"] = float(t <= LAST_DUSK_STEP and (day + 1) % cfg["unlock"] == 0 and n_shops < MAX_SHOPS)
    g["cal_shop_tick_now"] = float(t % 4 == 0 and t <= LAST_ACTION_STEP)
    g["cal_center_tick_now"] = float(t % DAY == 0 and t <= LAST_ACTION_STEP)
    return g


def _ticks_from(s0, every):
    """Interpreter steps s in [s0, 718] with s % every == 0 (town consumption ticks E L736, E L745)."""
    if s0 > LAST_ACTION_STEP:
        return 0
    return LAST_ACTION_STEP // every - (-(-s0 // every)) + 1


def _shop_block(t, shops, cfg):
    """Per product: H7 reveal timeline of the visible shops and C10 demand belief over the future draws."""
    n = len(shops)
    every = cfg["unlock"] * DAY
    out = {}
    tick_now = _ticks_from(t, 4)
    center = _ticks_from(t, DAY)
    future = [_ticks_from(every * (k + 1), 4) for k in range(n, MAX_SHOPS)]
    for p in PRODUCTS:
        ages = [(t - every * (k + 1)) / DAY for k, s in enumerate(shops) if p in SHOP_MULT[s]]
        mults = [SHOP_MULT[s].get(p, 0) for s in SHOP_TYPES]
        mean = sum(mults) / len(mults)
        var = sum((m - mean) ** 2 for m in mults) / len(mults)
        frac = sum(m > 0 for m in mults) / len(mults)
        visible = sum(SHOP_MULT[s].get(p, 0) for s in shops) * tick_now + (center if p != "FERTILIZER" else 0)
        expected = mean * sum(future)
        r = {"shop_src_count": len(ages) / MAX_SHOPS, "shop_src_none": float(not ages),
             "shop_src_newest_age": min(ages) / 30 if ages else 0.0, "shop_src_first_age": max(ages) / 30 if ages else 0.0,
             "shop_src_new_last_unlock": float(n > 0 and p in SHOP_MULT[shops[-1]]),
             "dem_p_new_source": 1.0 - (1.0 - frac) ** len(future),
             "dem_future_hi": slog(max(mults) * sum(future), 10.0),
             "dem_future_sd": slog(math.sqrt(var * sum(f * f for f in future)), 10.0),
             "dem_supported_share": visible / (visible + expected) if visible + expected > 0 else 0.0}
        out[p] = r
    return out


def _season(t, private):
    """Per producing product: can a producer started at the earliest step still yield before the end (S8 season
    boundary), full crop schedule, production events left. Crops: planted now if a seed is held, else next step
    (seed bought in this step's market, E L673-678, planted at t+1). Animals: placed now if carried, next step if in
    the shed, else t+2 (buy -> PICKUP -> PLACE). Production days: crops E L453, E L789-802; animals E L822-827; the
    last production day end is next_day 29."""
    seeds = private.get("seeds") or {}
    shed = private.get("shed") or {}
    carried = {}
    for inv in private.get("inventories") or []:
        for i, v in (inv or {}).items():
            carried[i] = carried.get(i, 0) + v
    out = {}
    for p, producer in PRODUCER_OF.items():
        if producer in CROPS:
            cd = CROPS[producer]
            start = t + (0 if seeds.get(producer, 0) > 0 else 1)
            d = start // DAY
            room = LAST_DAY - d - cd["first_yield_day"]
            can = room >= 0 and start <= LAST_ACTION_STEP
            if cd["ongoing"]:
                pulses = min(cd["max_yield"], room // cd["interval"] + 1) if can else 0
                full = d + cd["first_yield_day"] + cd["interval"] * (cd["max_yield"] - 1) <= LAST_DAY
            else:
                pulses = 1 if can else 0
                full = d + cd["max_yield_day"] <= LAST_DAY
        else:
            a = ANIMALS[producer]
            start = t + (0 if carried.get(producer, 0) > 0 else 1 if shed.get(producer, 0) > 0 else 2)
            d = start // DAY
            room = LAST_DAY - d - a["first_yield_day"]
            can = room >= 0 and start <= LAST_ACTION_STEP
            pulses = room // a["interval"] + 1 if can else 0
            full = False
        out[p] = {"producer_can_yield": float(can), "producer_full_schedule": float(full and can),
                  "producer_pulses_left": min(pulses, 30) / 10}
    return out


def _idle(step, k):
    """No visible effect of worker k in this step and nothing ambiguous: a PASS or a command that failed."""
    return not step["effect"][k] and not step["ambiguous"][k]


def compute(actor_obs, configuration, token_ids, history=()):
    """{"token": [len(token_ids), WIDTH_TOKEN] float16, "unit": [MAX_UNITS, WIDTH_UNIT] float16}."""
    cfg = _cfg(configuration)
    t = int(actor_obs["step"])
    farm, private = _own(actor_obs)
    history = list(history or ())[:2]
    h0 = history[0] if history else None
    h1 = history[1] if len(history) > 1 else None
    s1 = analyse_step(h0, actor_obs, configuration) if h0 is not None else None
    s2 = analyse_step(h1, h0, configuration) if (s1 is not None and h1 is not None) else None
    shops = list((actor_obs.get("town") or {}).get("unlocked_shops") or [])
    cash = float(farm["money"])
    shed_total = sum(int(v) for v in (private.get("shed") or {}).values())
    shed_full = shed_total >= cfg["shed"]
    params = _params(actor_obs, configuration)
    inventory = (actor_obs.get("market") or {}).get("inventory") or {}

    # -------- global
    g = _calendar(t, len(shops), cfg)
    g["rc_valid"] = float(s1 is not None)
    g["rc2_valid"] = float(s2 is not None)
    if h0 is not None and _consecutive(h0, actor_obs):
        fp = _own(h0)[0]
        g["rc_land_bought"] = float(len(farm.get("unlocked_quadrants") or []) > len(fp.get("unlocked_quadrants") or []))
        g["rc_cash_delta"] = slog(cash - float(fp["money"]), 100.0)
    if s1 is not None:
        n_idle = sum(1 for k in range(s1["n_prev"]) if _idle(s1, k))
        g["rc_any_idle"] = float(n_idle > 0)
        g["rc_n_idle"] = n_idle / MAX_UNITS
        g["rc_n_effect"] = sum(s1["effect"]) / MAX_UNITS
        g["rc_n_ambiguous"] = sum(s1["ambiguous"]) / MAX_UNITS
        g["rc_n_new_hands"] = max(0, len(_positions(farm)) - s1["n_prev"]) / 10
        for a, col in (("GOOSE", "rc_goose_bought"), ("COW", "rc_cow_bought"), ("SHEEP", "rc_sheep_bought")):
            g[col] = slog(max(0, s1["market_net"].get(a, 0)), 1.0)
        g["rc_seeds_bought"] = slog(sum(s1["seeds_bought"].values()), 1.0)
        g["rc_trade_cash"] = slog(s1["trade_cash"], 100.0)
        g["rc_hire_cash"] = slog(s1["hire_cash"], 100.0)
        g["lg_exact"] = float(s1["exact"])
        g["lg_discard"] = slog(s1["discard"], 1.0)
    if s2 is not None:
        g["lg2_trade_cash"] = slog(s1["trade_cash"] + s2["trade_cash"], 100.0)
        g["lg2_cash_delta"] = slog(s1["cash_delta"] + s2["cash_delta"], 100.0)
    hires = int(farm.get("hires_today", 0))
    g["labour_spent_today"] = slog(cfg["mult"] * sum(fib(i) for i in range(hires)), 100.0)
    g["can_afford_hire"] = float(cash >= cfg["mult"] * fib(hires))
    owned = len(farm.get("unlocked_quadrants") or [])
    g["can_afford_land"] = float(1 <= owned <= len(LAND_PRICES) and cash >= LAND_PRICES[owned - 1])
    g["q4_owned"] = float(owned >= 4)
    g["q4_affordable"] = float(owned == 3 and cash >= LAND_PRICES[2])
    g["shed_full"] = float(shed_full)
    unfed = unwatered = loss = False
    free = {"COOP": 0, "PASTURE": 0}
    tonight = t <= LAST_DUSK_STEP
    for row in farm["tiles"]:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                if not tile.get("watered_today"):
                    unwatered = True
                    loss = loss or int(tile.get("consecutive_unwatered", 0)) >= 1     # E L777-785
            elif "animal" in tile:
                if not tile.get("fed_today"):
                    unfed = True
                    loss = loss or int(tile.get("consecutive_unfed", 0)) >= 1         # E L813-820
            elif kind in free:
                free[kind] += 1
    g["any_animal_unfed"], g["any_plant_unwatered"] = float(unfed), float(unwatered)
    g["any_loss_tonight"] = float(loss and tonight)
    glob = np.asarray([g.get(c, 0.0) for c in GLOB_COLS], np.float32)

    # -------- per product
    shop = _shop_block(t, shops, cfg)
    season = _season(t, private)
    prod = np.zeros((len(PRODUCTS), N_PROD), np.float32)
    for i, p in enumerate(PRODUCTS):
        r = dict(shop[p])
        r.update(season.get(p, {}))
        if s1 is not None:
            net = s1["market_net"].get(p, 0)
            r["lg1_produced"] = slog(s1["produced"].get(p, 0), 1.0)
            r["lg1_consumed"] = slog(s1["consumed"].get(p, 0), 1.0)
            r["lg1_bought"], r["lg1_sold"] = slog(max(0, net), 1.0), slog(max(0, -net), 1.0)
            r["lg1_seeds_bought"] = slog(max(0, s1["seeds_bought"].get(p, 0)), 1.0) if p in CROPS else 0.0
        if s2 is not None:
            net = s1["market_net"].get(p, 0) + s2["market_net"].get(p, 0)
            r["lg2_produced"] = slog(s1["produced"].get(p, 0) + s2["produced"].get(p, 0), 1.0)
            r["lg2_consumed"] = slog(s1["consumed"].get(p, 0) + s2["consumed"].get(p, 0), 1.0)
            r["lg2_bought"], r["lg2_sold"] = slog(max(0, net), 1.0), slog(max(0, -net), 1.0)
        producer = PRODUCER_OF.get(p)
        if producer in CROPS:
            cost = CROPS[producer]["seed"]
            r["can_buy_producer"] = float(cash >= cost)                          # E L673-678
            r["n_producer_affordable"] = slog(math.floor(cash / cost), 1.0)
        elif producer in ANIMALS:
            cost = ANIMALS[producer]["cost"]
            r["can_buy_producer"] = float(cash >= cost and not shed_full)       # E L679-686
            r["n_producer_affordable"] = slog(math.floor(cash / cost), 1.0)
            r["free_structure"] = slog(free[ANIMALS[producer]["structure"]], 1.0)
        if p in BUYABLE and p in inventory:
            q = price_at(p, int(inventory[p]) - 1, params)                       # E L598-601, E L662-672
            r["can_buy_unit"] = float(cash >= q and not shed_full)
        prod[i] = [r.get(c, 0.0) for c in PROD_COLS]

    token = np.zeros((len(token_ids), WIDTH_TOKEN), np.float32)
    for n, tid in enumerate(token_ids):
        if tid == "global":
            token[n, N_PROD:] = glob
        elif tid.startswith("market/") and tid[7:] in _ROW:
            token[n, :N_PROD] = prod[_ROW[tid[7:]]]

    # -------- per unit (codec.unit_positions order)
    unit = np.zeros((MAX_UNITS, WIDTH_UNIT), np.float32)
    if s1 is not None:
        n_now = min(MAX_UNITS, len(_positions(farm)))
        for k in range(n_now):
            row = unit[k]
            if k >= s1["n_prev"]:
                row[_U["u_new_hand"]] = 1.0                                     # hired in the last market (E L702-709)
                continue
            codes, amb = s1["codes"][k], s1["ambiguous"][k]
            row[_U["u_valid"]] = 1.0
            row[_U["u_ambiguous"]] = float(amb)
            row[_U["u_effect"]] = float(s1["effect"][k])
            row[_U["u_no_effect"]] = float(_idle(s1, k))
            for c in codes:
                row[_U[_CODE_COL[c]]] = 1.0
            row[_U["u_load_delta"]] = slog(s1["load_delta"][k], 1.0)
            if s2 is not None and k < s2["n_prev"]:
                idle1, idle2 = _idle(s1, k), _idle(s2, k)
                row[_U["u_idle_streak"]] = (float(idle1) + float(idle1 and idle2)) / 2
                if MOVE in codes and MOVE in s2["codes"][k]:
                    row[_U["u_undo_move"]] = float(_positions(farm)[k] == s2["pos_prev"][k])
            elif _idle(s1, k):
                row[_U["u_idle_streak"]] = 0.5
    token16, unit16 = token.astype(np.float16), unit.astype(np.float16)
    if not (np.isfinite(token16).all() and np.isfinite(unit16).all()):
        raise ValueError("non-finite flags feature")
    return {"token": token16, "unit": unit16}
