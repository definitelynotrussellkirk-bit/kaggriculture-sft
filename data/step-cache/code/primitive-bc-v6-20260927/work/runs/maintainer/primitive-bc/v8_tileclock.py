"""tileclock (v8 feature family): per-TILE crop start clock, season cliffs, service value and fertilizer opportunity.

Review items: OPUS E3 (producer start clock) and S2/S3 (per-tile service value in units and $; value at risk), FABLE E6
(crop clock cliffs) and E9 (fertilizer economics), CODEX F1 (expose omitted theorem tile facts) and C4 (per-investment
ROI, fertilized variant, payback timing). Column definitions: COLUMNS-tileclock.md.

Rules. Engine = src/farm_neat/game/kaggriculture_engine_reference.py (sha256 bc8a5487...ee653e, the same bytes
theorem_features.py pins); line numbers refer to it.
  * PLANT (L215-226, L417-429): a new plant has consecutive_unwatered=1, yield 1 (one-shot) or 0 (ongoing), so it must be
    watered on its planting day.
  * WATER (L431-444): one-shot crops gain +1 (+2 if fertilized_until_day >= day) per watered day with
    (max_yield_day+1)//2 <= age <= max_yield_day, capped at max_yield.
  * HARVEST (L446-473): allowed when yield > 0 and age >= first_yield_day; a one-shot tile becomes empty.
  * FERTILIZE (L475-482): fertilized_until_day = max(old, day + 2) (three days incl. today).
  * FEED (L505-513) needs one WHEAT; CARE (L524-530); COLLECT_FERTILIZER (L515-522).
  * Decay (L752-766): from max_lifespan_step = (planted_day + max_yield_day + 1) * TPD (one-shot, L224) or the morning after
    the last ongoing production (L801-802), -1 yield every 2 steps.
  * Day end (L769-834, run when (step+1) % TPD == 0, L945): a plant unwatered today gets consecutive_unwatered+1 and becomes
    WEED at 2 (L777-785); ongoing crops produce on mornings next_day with days_since_first % interval == 0 and count <=
    max_yield, +2 only if watered today and fertilized_until_day >= today (L789-800). Animals: unfed -> escape at 2
    (L813-820); production +1 plus the pending care bonus only if fed today, and the bonus resets (L822-828); cared and
    fed today -> pending +1 (L829-830); fertilizer_available = True (L831).
  * The last day end is at step 695 (the game is DONE after step episodeSteps-2, L960), so LASTDAY = (episodeSteps-2)//TPD
    has no day end and no production follows it.
Season arithmetic follows theorem_features.py (season_formula L369-393, _plant_static L448-510, _animal_static L513-558);
test_tileclock.py checks svc_season_units against theorem_features' season_units_remaining and every clock/service column
against engine-function rollouts.

Plans (all day-granular, deterministic):
  full service  = WATER (plants) / FEED + CARE (animals) today if not done, and every later day before LASTDAY; ongoing crops
                  and animals are harvested every day; a one-shot crop is harvested on min(planted_day + max_yield_day,
                  LASTDAY) (horizon-adjusted: a partial final harvest counts). Current fertilizer only.
  skip          = no WATER / FEED / CARE today, full service from tomorrow; a tile that then dies tonight is harvested today
                  when the engine allows (salvage), otherwise its stored units are lost.
  fert now      = full service + one FERTILIZE today (before today's WATER).
  fert all      = full service + a FERTILIZE whenever an uncovered day would raise the season units: an ongoing crop's
                  production night, or a one-shot watering day from which plain waterings to the harvest day cannot reach
                  the cap (greedy interval cover; reaches the fertilize-every-day units with fewer fertilizers).

Only the actor observation is read (own private shed/carried FERTILIZER for one global; both public farms; market).
No history, rival private state, future shops or labels. $ values are at today's spot sell quote (the price impact of
selling many units is not modelled; units are given beside every $ column).
"""
from __future__ import annotations

import numpy as np

import theorem_features as TF

SCHEMA = "tileclock-v1"
CROPS, ANIMALS, PRODUCTS = TF.CROPS, TF.ANIMALS, TF.PRODUCTS
CROP_ORDER = tuple(CROPS)                                     # WHEAT, CARROT, TOMATO, STRAWBERRY, MELON
_PIDX = {p: i for i, p in enumerate(PRODUCTS)}
_FI, _WI = _PIDX["FERTILIZER"], _PIDX["WHEAT"]
LIN_CLIP = 8.0

# (field, transform, scale): lin -> clip(x/scale, -8, 8); slog -> sign(x) * log1p(|x|/scale)
CLOCK_FIELDS = (
    ("harvests", "lin", 4.0), ("units", "lin", 10.0), ("units_fert", "lin", 10.0), ("wait_loss", "lin", 10.0),
    ("days_to_cliff", "lin", 30.0), ("days_to_last_start", "lin", 30.0), ("first_harvest_days", "lin", 30.0),
    ("net_quote", "slog", 100.0), ("net_fert_quote", "slog", 100.0),
)
SVC_FIELDS = (
    ("season_units", "lin", 10.0), ("season_value", "slog", 100.0),
    ("gain_now_units", "lin", 10.0), ("gain_now_value", "slog", 100.0),
    ("skip_loss_units", "lin", 10.0), ("skip_loss_value", "slog", 100.0),
    ("care_gain_units", "lin", 10.0), ("care_gain_value", "slog", 100.0),
    ("fert_gain_units", "lin", 10.0), ("fert_gain_net_value", "slog", 100.0), ("fert_takes_tonight", "lin", 1.0),
    ("fert_upside_units", "lin", 10.0), ("collect_fert_value", "slog", 100.0), ("cap_loss_tonight_units", "lin", 10.0),
)
GLOBAL_FIELDS = (
    ("own_fert_held", "lin", 10.0), ("own_fert_collectable", "lin", 10.0), ("own_fert_gain_units_total", "lin", 10.0),
    ("own_fert_takes_tonight_total", "lin", 10.0), ("own_skip_loss_value_total", "slog", 100.0),
    ("rival_skip_loss_value_total", "slog", 100.0), ("own_care_gain_units_total", "lin", 10.0),
)
_SPEC = (tuple((f"tc_clk_{c.lower()}_{n}", k, s) for c in CROP_ORDER for n, k, s in CLOCK_FIELDS)
         + tuple((f"tc_svc_{n}", k, s) for n, k, s in SVC_FIELDS)
         + tuple((f"tc_g_{n}", k, s) for n, k, s in GLOBAL_FIELDS))
TOKEN_COLS: tuple[str, ...] = tuple(n for n, _, _ in _SPEC)
UNIT_COLS: tuple[str, ...] = ()
WIDTH_TOKEN = len(TOKEN_COLS)
WIDTH_UNIT = len(UNIT_COLS)
MAX_UNITS = 25
COL = {n: i for i, n in enumerate(TOKEN_COLS)}
N_CLK = len(CROP_ORDER) * len(CLOCK_FIELDS)
S0 = N_CLK                                  # first service column
G0 = S0 + len(SVC_FIELDS)                   # first global column
_SC = {n: i for i, (n, _, _) in enumerate(SVC_FIELDS)}
_GC = {n: i for i, (n, _, _) in enumerate(GLOBAL_FIELDS)}
_SCALE = np.asarray([s for _, _, s in _SPEC], np.float32)
_IS_SLOG = np.asarray([k == "slog" for _, k, _ in _SPEC], bool)


# ----------------------------------------------------------------------------- engine-rule simulators
def plant_sim(crop, pd, y, cu, w, fud, d, lastday, water0=True, fert0=False, fert_all=False):
    """One PLANT tile from the start of day d to LASTDAY. Returns (units harvested, fertilizers used,
    harvest events, first harvest day or -1). water0: WATER today (if not already); fert0: FERTILIZE today."""
    c = CROPS[crop]
    first, myd, cap, interval = c["first_yield_day"], c["max_yield_day"], c["max_yield"], c["interval"]
    nf = 0
    if not c["ongoing"]:
        ws = (myd + 1) // 2
        if d - pd > myd:                                   # decaying (L224, L752-766): harvest what is left now
            return (y, 0, 1, d) if y > 0 else (0, 0, 0, -1)
        h = min(pd + myd, lastday)
        if h - pd < first:                                 # no harvestable age before the end (L453)
            return 0, 0, 0, -1
        reach, y_reach = -1, -1
        for day in range(d, h + 1):
            watered = w if day == d else False
            if day == d and fert0:
                fud = max(fud, d + 2)                      # L481
                nf += 1
            if not watered and (water0 if day == d else True):
                in_win = ws <= day - pd <= myd             # L440-441
                # fertilize only when plain waterings from today to the harvest day cannot reach the cap
                if fert_all and in_win and fud < day and y + (h - day + 1) < cap:
                    fud = day + 2
                    nf += 1
                if in_win:
                    y = min(cap, y + (2 if fud >= day else 1))   # L442-443
                watered = True
            if day - pd >= first and y > y_reach:          # earliest harvestable day holding the final yield
                reach, y_reach = day, y
            if day == h:
                break
            if not watered and cu + 1 >= 2:                # dies tonight (L777-785): salvage harvest today if allowed
                return (y, nf, 1, day) if (day - pd >= first and y > 0) else (0, nf, 0, -1)
            cu = 0 if watered else cu + 1
        return y, nf, 1, reach                             # harvested on h; reach = cash date
    total = events = 0
    first_ev = -1
    for day in range(d, lastday + 1):
        total += y                                         # daily HARVEST (L446-466)
        y = 0
        if day >= lastday:                                 # no day end with a later action step
            break
        if day == d and fert0:
            fud = max(fud, d + 2)
            nf += 1
        watered = (w or water0) if day == d else True
        dsf = day + 1 - pd - first                         # L789-797
        event = dsf >= 0 and dsf % interval == 0 and dsf // interval + 1 <= cap
        if fert_all and event and watered and fud < day:
            fud = day + 2
            nf += 1
        cu = 0 if watered else cu + 1                      # L777-785
        if cu >= 2:
            break
        if event:
            y = min(cap, y + (2 if (watered and fud >= day) else 1))   # L799-800
            events += 1
            if first_ev < 0:
                first_ev = day + 1
    return total, nf, events, first_ev


def animal_sim(animal, pl, y, cu, fed, cared, pcb, d, lastday, feed0=True, care0=True):
    """One animal tile from the start of day d to LASTDAY with daily harvest. Returns (units harvested, FEEDs used)."""
    a = ANIMALS[animal]
    first, interval, cap = a["first_yield_day"], a["interval"], a["max_held"]
    total = feeds = 0
    for day in range(d, lastday + 1):
        total += y                                         # daily HARVEST (L469-472)
        y = 0
        if day >= lastday:                                 # feeding on the last day changes nothing
            break
        if day == d:
            f, ca = fed, cared
            if feed0 and not f:
                f = True
                feeds += 1
            ca = ca or care0
        else:
            f = ca = True
            feeds += 1
        cu = 0 if f else cu + 1                            # L813-820
        if cu >= 2:
            break
        dsf = day + 1 - pl - first
        if dsf >= 0 and dsf % interval == 0:               # L822-828
            y = min(cap, y + 1 + (pcb if f else 0))
            pcb = 0
        if ca and f:                                       # L829-830
            pcb += 1
    return total, feeds


# ----------------------------------------------------------------------------- per-day clock table
_CLOCK_CACHE: dict = {}


def clock_table(lastday):
    """Per crop, per start day p in [0, LASTDAY+1]: (units, units_fert, n_fert, harvests, first_harvest_day) for a single
    planting on an empty tile, watered daily (and fertilized on every useful day for the _fert values)."""
    tab = _CLOCK_CACHE.get(lastday)
    if tab is not None:
        return tab
    tab = {}
    for crop in CROP_ORDER:
        c = CROPS[crop]
        y0 = 0 if c["ongoing"] else 1                      # L223
        rows = []
        for p in range(0, lastday + 2):
            if p > lastday:
                rows.append((0, 0, 0, 0, -1))
                continue
            u, _, hv, fh = plant_sim(crop, p, y0, 1, False, -1, p, lastday, True, False, False)
            uf, nf, _, _ = plant_sim(crop, p, y0, 1, False, -1, p, lastday, True, False, True)
            rows.append((u, uf, nf, hv, fh))
        tab[crop] = rows
    _CLOCK_CACHE[lastday] = tab
    return tab


def clock_raw(d, lastday, quotes, fert_buy=0.0):
    """[N_CLK] raw (unscaled) clock values for a start on day d."""
    out = np.zeros(N_CLK, np.float64)
    tab = clock_table(lastday)
    k = len(CLOCK_FIELDS)
    for j, crop in enumerate(CROP_ORDER):
        rows = tab[crop]
        dd = min(max(d, 0), lastday + 1)
        u, uf, nf, hv, fh = rows[dd]
        u1 = rows[min(dd + 1, lastday + 1)][0]
        cliff = last_any = 0
        if hv > 0:
            p = dd
            while p + 1 <= lastday and rows[p + 1][3] == hv:
                p += 1
            cliff = p - dd + 1
            p = dd
            while p + 1 <= lastday and rows[p + 1][3] >= 1:
                p += 1
            last_any = p - dd + 1
        q = quotes[_PIDX[crop]]
        seed = CROPS[crop]["seed"]
        base = j * k
        out[base + 0] = hv
        out[base + 1] = u
        out[base + 2] = uf
        out[base + 3] = u - u1
        out[base + 4] = cliff
        out[base + 5] = last_any
        out[base + 6] = (fh - dd) if hv > 0 and fh >= 0 else 0
        out[base + 7] = u * q - seed
        out[base + 8] = uf * q - seed - nf * fert_buy
    return out


# ----------------------------------------------------------------------------- per-tile service facts (cached)
_TILE_CACHE: dict = {}


def _plant_facts(t, d, lastday):
    """(season, gain_now, skip_loss, fert_gain, fert_takes, fert_upside, cap_loss, product index, is_plant=1, feeds_saved=0,
    care_gain=0, fert_available=0)."""
    crop = t["crop"]
    pd, y, cu = t["planted_day"], t["yield_units"], t["consecutive_unwatered"]
    w = bool(t["watered_today"])
    fud = t.get("fertilized_until_day", -1)
    if fud is None:
        fud = -1
    key = ("P", crop, pd, y, cu, w, fud, d, lastday)
    hit = _TILE_CACHE.get(key)
    if hit is not None:
        return hit
    c = CROPS[crop]
    first, myd, cap, interval = c["first_yield_day"], c["max_yield_day"], c["max_yield"], c["interval"]
    full = plant_sim(crop, pd, y, cu, w, fud, d, lastday, True)[0]
    skip = plant_sim(crop, pd, y, cu, w, fud, d, lastday, False)[0]
    fnow = plant_sim(crop, pd, y, cu, w, fud, d, lastday, True, True)[0]
    fall = plant_sim(crop, pd, y, cu, w, fud, d, lastday, True, False, True)[0]
    age = d - pd
    gain_now = takes = cap_loss = 0
    if not c["ongoing"]:
        ws = (myd + 1) // 2
        harvestable_by_end = min(pd + myd, lastday) - pd >= first
        if not w and ws <= age <= myd and harvestable_by_end:           # theorem water_gain_now (TF L465-467)
            gain_now = min(cap, y + (2 if fud >= d else 1)) - y
            takes = int(fud < d and y <= cap - 2 and fnow > full)   # +1 on today's watering that the harvest keeps
    elif d < lastday:
        dsf = d + 1 - pd - first
        if dsf >= 0 and dsf % interval == 0 and dsf // interval + 1 <= cap:   # event tonight
            b_srv = 2 if fud >= d else 1
            dies = (not w) and cu + 1 >= 2
            cur = 0 if dies else (min(cap, y + (2 if (w and fud >= d) else 1)) - y)
            gain_now = (min(cap, y + b_srv) - y) - cur                    # TF gain_tonight_serviced - current (L493-498)
            takes = int(fud < d)
            cap_loss = max(0, y + b_srv - cap)            # TF overflow_next_event_serviced (L499); b_srv <= cap
    res = (full, gain_now, full - skip, fnow - full, takes, fall - full, cap_loss, _PIDX[crop], 1, 0, 0, 0)
    _TILE_CACHE[key] = res
    return res


def _animal_facts(t, d, lastday):
    animal = t["animal"]
    pl, y, cu = t["placed_day"], t["yield_units"], t["consecutive_unfed"]
    fed, cared = bool(t["fed_today"]), bool(t["cared_today"])
    pcb = t.get("pending_care_bonus", 0) or 0
    fa = bool(t.get("fertilizer_available", False))
    key = ("A", animal, pl, y, cu, fed, cared, pcb, fa, d, lastday)
    hit = _TILE_CACHE.get(key)
    if hit is not None:
        return hit
    a = ANIMALS[animal]
    first, interval, cap = a["first_yield_day"], a["interval"], a["max_held"]
    full, feeds_full = animal_sim(animal, pl, y, cu, fed, cared, pcb, d, lastday, True, True)
    skip, feeds_skip = animal_sim(animal, pl, y, cu, fed, cared, pcb, d, lastday, False, False)
    nocare = animal_sim(animal, pl, y, cu, fed, cared, pcb, d, lastday, True, False)[0] if not cared else full
    gain_now = cap_loss = 0
    if d < lastday:
        dsf = d + 1 - pl - first
        if dsf >= 0 and dsf % interval == 0:                              # event tonight (L822-823)
            dies = (not fed) and cu + 1 >= 2
            cur = 0 if dies else min(cap, y + 1 + (pcb if fed else 0)) - y
            gain_now = (min(cap, y + 1 + pcb) - y) - cur                  # TF L531-535
            cap_loss = max(0, y + 1 + pcb - cap) - max(0, 1 + pcb - cap)   # the part a HARVEST today would save
    res = (full, gain_now, full - skip, 0, 0, 0, cap_loss, _PIDX[a["product"]], 0, feeds_full - feeds_skip,
           full - nocare, int(fa))
    _TILE_CACHE[key] = res
    return res


# ----------------------------------------------------------------------------- token map
_TOKEN_MAP: dict = {}


def _token_map(token_ids):
    """(global index or -1, {(role, cell): token index})."""
    key = tuple(token_ids)
    hit = _TOKEN_MAP.get(key)
    if hit is not None:
        return hit
    g = -1
    tiles = {}
    for n, tid in enumerate(key):
        if tid == "global":
            g = n
        elif tid.startswith("tile/"):
            role = int(tid[5])
            xs, ys = tid[7:].split(",")
            tiles[(role, int(ys) * 10 + int(xs))] = n
    if len(_TOKEN_MAP) > 64:
        _TOKEN_MAP.clear()
    hit = _TOKEN_MAP[key] = (g, tiles)
    return hit


# ----------------------------------------------------------------------------- main entry
def _config(configuration):
    tpd = max(1, int(TF._setting(configuration, "turnsPerDay", 24)))
    episode = int(TF._setting(configuration, "episodeSteps", 720))
    return tpd, episode


def raw(actor_obs, configuration=None, token_ids=None):
    """Unscaled float64 [len(token_ids), WIDTH_TOKEN] (same layout as compute's token block)."""
    tpd, episode = _config(configuration)
    last = episode - 2
    lastday = last // tpd
    step = actor_obs.get("step")
    s = int(step) if step is not None else int(actor_obs["day"]) * tpd + int(actor_obs["hour"])
    d = s // tpd
    me = int(actor_obs["player"])
    farms = actor_obs["farms"]
    market = actor_obs["market"]
    curves = TF._curves(TF._market_params(actor_obs, configuration))
    inv = market["inventory"]
    quotes = [curves[i].quote(int(inv[p])) for i, p in enumerate(PRODUCTS)]
    fert_sell = quotes[_FI]
    fert_buy = curves[_FI].quote(int(inv["FERTILIZER"]) - 1)
    wheat_buy = curves[_WI].quote(int(inv["WHEAT"]) - 1)
    if len(_TILE_CACHE) > 50000:
        _TILE_CACHE.clear()
    g_idx, tok = _token_map(token_ids)
    out = np.zeros((len(token_ids), WIDTH_TOKEN), np.float64)
    clk = clock_raw(d, lastday, quotes, fert_buy=fert_buy)
    gl = np.zeros(len(GLOBAL_FIELDS), np.float64)
    skip_totals = [0.0, 0.0]
    for role, farm in ((0, farms[me]), (1, farms[1 - me])):
        facts = []
        empty = []
        best_fert = 0.0
        cell = 0
        for line in farm["tiles"]:
            for t in line:
                if t is None:
                    empty.append(cell)
                elif isinstance(t, dict):
                    kind = t.get("kind")
                    if kind == "PLANT":
                        f = _plant_facts(t, d, lastday)
                        facts.append((cell, f))
                        v = f[3] * quotes[f[7]]
                        if v > best_fert:
                            best_fert = v
                    elif "animal" in t:
                        facts.append((cell, _animal_facts(t, d, lastday)))
                cell += 1
        if role == 0:
            rows = [tok[(0, c)] for c in empty if (0, c) in tok]
            if rows:
                out[rows, :N_CLK] = clk
        fert_value = max(float(fert_sell), best_fert)
        for c, f in facts:
            n = tok.get((role, c))
            season, gain_now, skip_u, fert_u, takes, upside, cap_loss, pi, is_plant, feeds_saved, care_u, fa = f
            q = quotes[pi]
            skip_v = skip_u * q - feeds_saved * wheat_buy
            skip_totals[role] += skip_v
            if role == 0:
                gl[_GC["own_fert_gain_units_total"]] += fert_u
                gl[_GC["own_fert_takes_tonight_total"]] += takes
                gl[_GC["own_care_gain_units_total"]] += care_u
                gl[_GC["own_fert_collectable"]] += fa
            if n is None:
                continue
            row = out[n]
            row[S0 + _SC["season_units"]] = season
            row[S0 + _SC["season_value"]] = season * q
            row[S0 + _SC["gain_now_units"]] = gain_now
            row[S0 + _SC["gain_now_value"]] = gain_now * q
            row[S0 + _SC["skip_loss_units"]] = skip_u
            row[S0 + _SC["skip_loss_value"]] = skip_v
            row[S0 + _SC["care_gain_units"]] = care_u
            row[S0 + _SC["care_gain_value"]] = care_u * q
            row[S0 + _SC["cap_loss_tonight_units"]] = cap_loss
            if is_plant:
                row[S0 + _SC["fert_gain_units"]] = fert_u
                row[S0 + _SC["fert_gain_net_value"]] = fert_u * q - fert_sell
                row[S0 + _SC["fert_takes_tonight"]] = takes
                row[S0 + _SC["fert_upside_units"]] = upside
            elif fa:
                row[S0 + _SC["collect_fert_value"]] = fert_value
    if g_idx >= 0:
        private = actor_obs.get("private") or {}
        held = int((private.get("shed") or {}).get("FERTILIZER", 0))
        for inv_ in private.get("inventories") or ():
            if isinstance(inv_, dict):
                held += int(inv_.get("FERTILIZER", 0))
        gl[_GC["own_fert_held"]] = held
        gl[_GC["own_skip_loss_value_total"]] = skip_totals[0]
        gl[_GC["rival_skip_loss_value_total"]] = skip_totals[1]
        out[g_idx, G0:] = gl
    return out


def scale(raw_block):
    """Raw -> model scale: lin columns clip(x/scale, +-8); slog columns sign(x)*log1p(|x|/scale). float32."""
    x = np.asarray(raw_block, np.float64)
    lin = np.clip(x / _SCALE, -LIN_CLIP, LIN_CLIP)
    slog = np.sign(x) * np.log1p(np.abs(x) / _SCALE)
    return np.where(_IS_SLOG, slog, lin).astype(np.float32)


def compute(actor_obs, configuration, token_ids, history=()):
    """{"token": [len(token_ids), WIDTH_TOKEN] float16, "unit": [25, 0] float16}. history is accepted and unused."""
    block = scale(raw(actor_obs, configuration, token_ids))
    return {"token": block.astype(np.float16), "unit": np.zeros((MAX_UNITS, WIDTH_UNIT), np.float16)}


TF.warm()
