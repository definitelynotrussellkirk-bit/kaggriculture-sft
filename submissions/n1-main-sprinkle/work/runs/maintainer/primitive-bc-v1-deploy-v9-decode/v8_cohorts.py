"""v8 feature family "cohorts": production cohorts, scheduled receipts, working capital to first sale,
realizable equity, producer attractiveness, production/demand horizons and price trends.

Pure function of ONE seat's actor observation (training.bc.actor_observation whitelist) plus that seat's own
previous observations (`history`, most recent first). Nothing reads the rival's private state (shed, carried
goods, seeds), the episode seed, future shop identities, replay labels or outcomes. Rival columns use the
public board only; the rival's shed and carried goods are unknown and are NOT counted (not zero-filled into
any lead as if known: the equity columns say which side includes what).

Declared scenario (a schedule, not a behaviour forecast) = FULL SERVICE of every visible producer on both
farms: plants watered daily, animals fed and cared daily, every unit harvested on the day it becomes
available, animal fertilizer collected daily, no new fertilizer applied. Produced units are sold on the day
they become available ("sell-as-produced" market path); town demand (visible shops + town centre + the
theorem_features expected-new-shop prior) drains the market between days; animal feed is bought as WHEAT.

Engine = src/farm_neat/game/kaggriculture_engine_reference.py sha256 bc8a5487...653e (same bytes as
theorem_features.ENGINE_SHA256); line numbers refer to those bytes:
  crop/animal tables L11-24; price law L192-206 (+_shape L61-74); new plant L215-227 (one-shot yield 1,
  lifespan (pd+myd+1)*TPD, planting day counts unwatered); new animal L229-242; WATER one-shot window
  L431-444; HARVEST L446-473; FEED L505-513; COLLECT_FERTILIZER L515-522; CARE L524-529; market commit
  L652-688 ($1 sales add no inventory L658-660; BUY_PRODUCT quotes inv-1 L598-601); town demand L728-749;
  plant refresh (ongoing events, fertilizer bonus, death) L769-802; animal refresh (events, care bonus,
  fertilizer, escape) L805-834; shop unlock L886-892; last refresh at step (d+1)*TPD-1 <= episodeSteps-2.
Price law helpers (exact tables, floor inventory, visible-demand ticks, expected-new-shop rate) are reused
from theorem_features (TF._curves/_ticks/_SHOP_MULT/_EXPECTED_RATE/_CENTER/_harvest_age); nothing is
copied or edited there.

compute(actor_obs, configuration, token_ids, history=()) -> {"token": [N, WIDTH_TOKEN] f16, "unit": [25, 0] f16}
Product columns (PROD_COLS) sit on market/<PRODUCT> tokens, global columns (GLOB_COLS) on "global"; zeros
elsewhere. Column meanings: COLUMNS-cohorts.md.
"""
from __future__ import annotations

import numpy as np

import theorem_features as TF

CROPS, ANIMALS = TF.CROPS, TF.ANIMALS
PRODUCTS = tuple(TF.PRODUCTS)                 # WHEAT CARROT TOMATO STRAWBERRY MELON EGG MILK WOOL FERTILIZER
PRODUCERS = tuple(TF.PRODUCERS)               # WHEAT CARROT TOMATO STRAWBERRY MELON GOOSE COW SHEEP
PRODUCT_OF = TF.PRODUCT_OF
NPI = len(PRODUCTS)
PIDX = {p: i for i, p in enumerate(PRODUCTS)}
PROD_OF_R = tuple(PIDX[PRODUCT_OF[r]] for r in PRODUCERS)
R_OF_P = {PIDX[PRODUCT_OF[r]]: ri for ri, r in enumerate(PRODUCERS)}
FERT = PIDX["FERTILIZER"]
WHEAT = PIDX["WHEAT"]
MAX_UNITS = 25
BUCKETS = ("producing", "first_le2d", "first_3to5d", "first_6plusd", "no_output_left")
HWIN = ("1d", "3d", "7d", "end")
HDAYS = (1, 3, 7, 10 ** 6)
FAST_REL, SLOW_REL = 0.05, 0.01        # trend class thresholds: |implied quote change| per day / current quote
STRUCTURE = {a: ANIMALS[a]["structure"] for a in ANIMALS}

# (name, mode, scale): slog -> sign(x)*log1p(|x|/scale); lin -> x/scale (bounded by construction); clip -> clip(x/scale,-1,1)
PROD_COLS = (
    *((f"own_{b}", "slog", 1.0) for b in BUCKETS),
    *((f"rival_{b}", "slog", 1.0) for b in BUCKETS),
    *((f"own_units_{h}", "slog", 10.0) for h in HWIN),
    *((f"own_value_{h}", "slog", 100.0) for h in HWIN),
    *((f"rival_units_{h}", "slog", 10.0) for h in HWIN),
    *((f"rival_value_{h}", "slog", 100.0) for h in HWIN),
    ("wc_cash_to_first_sale", "slog", 100.0), ("wc_days_to_first_sale", "lin", 10.0),
    ("wc_first_sale_reachable", "lin", 1.0), ("wc_start_today", "lin", 1.0), ("wc_site_state", "lin", 1.0),
    ("wc_reserve_after", "slog", 100.0), ("wc_payback_days", "lin", 10.0), ("wc_net_to_end", "slog", 100.0),
    ("attract_ratio", "slog", 1.0), ("attract_rank", "lin", 8.0),
    ("count_own", "slog", 1.0), ("count_rival", "slog", 1.0), ("count_total", "slog", 1.0),
    *((f"vis_production_{h}", "slog", 10.0) for h in HWIN[:3]),
    *((f"demand_{h}", "slog", 10.0) for h in HWIN[:3]),
    *((f"delta_{h}", "slog", 10.0) for h in HWIN[:3]),
    *((f"delta_quote_{h}", "slog", 10.0) for h in HWIN[:3]),
    ("flow_production_rate", "slog", 1.0), ("flow_sell_pressure", "slog", 1.0), ("flow_demand_rate", "slog", 1.0),
    ("net_flow_per_day", "slog", 1.0), ("dq_per_day", "slog", 1.0), ("dq_rel_per_day", "clip", 1.0),
    ("trend_class", "lin", 2.0),
    ("obs_dq_1", "slog", 1.0), ("obs_dq_2", "slog", 1.0), ("obs_net_sales_1", "slog", 1.0),
    ("obs_net_sales_2", "slog", 1.0),
)
_RANKED = ("best", "second", "worst", "second_worst")
GLOB_COLS = (
    ("own_liquid_budget", "slog", 1000.0), ("own_obligations_3d", "slog", 100.0),
    ("own_held_value", "slog", 1000.0), ("own_board_value_end", "slog", 1000.0),
    ("own_feed_liability", "slog", 1000.0), ("own_cash_plus_pipeline", "slog", 1000.0),
    ("rival_board_value_end", "slog", 1000.0), ("rival_feed_liability", "slog", 1000.0),
    ("rival_cash_plus_pipeline", "slog", 1000.0), ("realizable_equity_lead", "slog", 1000.0),
    ("board_equity_lead", "slog", 1000.0), ("cash_equity_disagree", "lin", 1.0),
    *((f"pipeline_lead_{h}", "slog", 1000.0) for h in HWIN),
    *(c for k in _RANKED for c in (
        *((f"{k}_is_{r}", "lin", 1.0) for r in PRODUCERS), (f"{k}_ratio", "slog", 1.0),
        (f"{k}_count_own", "slog", 1.0), (f"{k}_count_rival", "slog", 1.0), (f"{k}_count_total", "slog", 1.0))),
    ("best_ratio_gap", "slog", 1.0),
    ("hist_lag1_avail", "lin", 1.0), ("hist_lag2_avail", "lin", 1.0),
)
N_PROD, N_GLOB = len(PROD_COLS), len(GLOB_COLS)
TOKEN_COLS = tuple("m." + n for n, _, _ in PROD_COLS) + tuple("g." + n for n, _, _ in GLOB_COLS)
UNIT_COLS: tuple = ()
WIDTH_TOKEN, WIDTH_UNIT = len(TOKEN_COLS), len(UNIT_COLS)
PC = {n: i for i, (n, _, _) in enumerate(PROD_COLS)}
GC = {n: i for i, (n, _, _) in enumerate(GLOB_COLS)}


def _scaler(cols):
    mode = np.array([{"slog": 0, "lin": 1, "clip": 2}[m] for _, m, _ in cols])
    scale = np.array([s for _, _, s in cols], np.float64)
    return mode, scale


_PMODE, _PSCALE = _scaler(PROD_COLS)
_GMODE, _GSCALE = _scaler(GLOB_COLS)


def scale(x, mode, sc):
    """Signed log / linear / clip per column (float64 in, float32 out)."""
    y = x / sc
    out = np.where(mode == 0, np.sign(y) * np.log1p(np.abs(y)), y)
    out = np.where(mode == 2, np.clip(y, -1.0, 1.0), out)
    return out.astype(np.float32)


# ----------------------------------------------------------------------------- per-tile schedules (cached)
_CACHE: dict = {}
_NEW: dict = {}


def _bucket(vec, producing, d):
    """0 producing, 1 first scheduled receipt in <=2 days, 2 in 3-5, 3 in 6+, 4 no scheduled units left."""
    nz = np.flatnonzero(vec)
    if not len(nz):
        return 4
    if producing:
        return 0
    delta = int(nz[0]) - d
    return 1 if delta <= 2 else (2 if delta <= 5 else 3)


def _plant_entry(t, d, lastday, nd):
    crop = t["crop"]
    c = CROPS[crop]
    first, myd, cap, interval = c["first_yield_day"], c["max_yield_day"], c["max_yield"], c["interval"]
    pd = int(t["planted_day"])
    w = bool(t.get("watered_today"))
    y = int(t.get("yield_units") or 0)
    fud = t.get("fertilized_until_day")
    fud = -1 if fud is None else int(fud)
    age = d - pd
    vec = np.zeros(nd)
    if not c["ongoing"]:
        # one-shot: +1 (+2 fertilized) per watered day in [pd+(myd+1)//2, pd+myd] (L431-444), harvest removes the
        # tile (L467-468); harvestable from age >= first_yield_day (L453); decays from (pd+myd+1)*TPD (L224, L752-766)
        if pd + first <= lastday:
            if age > myd:
                if y > 0:
                    vec[d] = y
            else:
                ws = (myd + 1) // 2
                yy, h = y, None
                for day in range(d, min(pd + myd, lastday) + 1):
                    if day >= pd + ws and not (day == d and w):
                        yy = min(cap, yy + (2 if fud >= day else 1))
                    if day >= pd + first and yy >= cap:
                        h = day
                        break
                if h is None:
                    h = min(pd + myd, lastday)
                if yy > 0:
                    vec[h] = yy
    else:
        # ongoing: event on morning e = pd+first+k*interval, k < max_yield (L789-802); +2 if watered and
        # fertilized on e-1 (L799-800); stored yield is harvestable now (L446-473)
        if y > 0:
            vec[d] += y
        for k in range(cap):
            e = pd + first + k * interval
            if d + 1 <= e <= lastday:
                vec[e] += 2 if fud >= e - 1 else 1
    producing = age >= first
    return (PIDX[crop], _bucket(vec, producing, d), vec, -1, None, None, crop)


def _animal_entry(t, d, lastday, nd):
    animal = t["animal"]
    a = ANIMALS[animal]
    first, interval, cap = a["first_yield_day"], a["interval"], a["max_held"]
    pl = int(t["placed_day"])
    fed = bool(t.get("fed_today"))
    y = int(t.get("yield_units") or 0)
    p = int(t.get("pending_care_bonus") or 0)
    fa = bool(t.get("fertilizer_available"))
    vec = np.zeros(nd)
    if y > 0:
        vec[d] += y
    # refresh of night n: event when (n+1-pl-first) >= 0 and % interval == 0: +1+pending (fed), cap max_held,
    # pending reset; then +1 pending if fed and cared (L822-830). Daily harvest keeps held at 0 before events.
    for night in range(d, lastday):
        dsf = night + 1 - pl - first
        if dsf >= 0 and dsf % interval == 0:
            vec[night + 1] += min(cap, 1 + p)
            p = 0
        p += 1
    fvec = np.zeros(nd)          # fertilizer_available after every surviving refresh (L831), one collect per day
    if fa:
        fvec[d] = 1.0
    fvec[d + 1:lastday + 1] = 1.0
    feed = np.zeros(nd)          # one WHEAT per day keeps consecutive_unfed at 0 (L505-513, L813-819)
    if d < lastday:
        feed[d] = 0.0 if fed else 1.0
        feed[d + 1:lastday] = 1.0
    producing = d >= pl + first
    fprod = d >= pl + 1
    return (PIDX[a["product"]], _bucket(vec, producing, d), vec, _bucket(fvec, fprod, d), fvec, feed, animal)


def _entry(t, d, lastday, nd):
    if t.get("kind") == "PLANT":
        key = ("P", t["crop"], t["planted_day"], bool(t.get("watered_today")), t.get("yield_units"),
               t.get("fertilized_until_day"), d, lastday)
        hit = _CACHE.get(key)
        if hit is None:
            hit = _CACHE[key] = _plant_entry(t, d, lastday, nd)
        return hit
    key = ("A", t["animal"], t["placed_day"], bool(t.get("fed_today")), t.get("yield_units"),
           t.get("pending_care_bonus"), bool(t.get("fertilizer_available")), d, lastday)
    hit = _CACHE.get(key)
    if hit is None:
        hit = _CACHE[key] = _animal_entry(t, d, lastday, nd)
    return hit


def _farm(farm, d, lastday, nd, sites=None):
    """(A [9, nd] units by availability day, F [nd] feed units, B [9, 5] bucket counts, C [9] producer counts).
    sites: dict of lists to fill with (x, y) of empty / weed / free COOP / free PASTURE tiles."""
    groups = {}
    for yy, line in enumerate(farm["tiles"]):
        for xx, t in enumerate(line):
            if isinstance(t, dict):
                kind = t.get("kind")
                if kind == "PLANT" or "animal" in t:
                    e = _entry(t, d, lastday, nd)
                    g = groups.get(id(e))
                    if g is None:
                        groups[id(e)] = [e, 1]
                    else:
                        g[1] += 1
                elif sites is not None and kind in sites:
                    sites[kind].append((xx, yy))
            elif t is None and sites is not None:
                sites["EMPTY"].append((xx, yy))
    A = np.zeros((NPI, nd))
    F = np.zeros(nd)
    B = np.zeros((NPI, 5))
    C = np.zeros(NPI)
    for e, c in groups.values():
        pi, b, vec, fb, fvec, feed, _ = e
        A[pi] += c * vec
        B[pi, b] += c
        C[pi] += c
        if fvec is not None:
            A[FERT] += c * fvec
            B[FERT, fb] += c
            F += c * feed
    return A, F, B, C


def _new_schedule(r, sd, lastday, nd):
    """One more producer r started on day sd under full service: (units, fert, feed, purchase $) by day."""
    key = (r, sd, lastday)
    hit = _NEW.get(key)
    if hit is not None:
        return hit
    u, fv, fd, cost = [0] * nd, [0] * nd, [0] * nd, [0.0] * nd
    if sd <= lastday:
        if r in CROPS:
            c = CROPS[r]
            first, myd, cap, interval, seed = (c["first_yield_day"], c["max_yield_day"], c["max_yield"],
                                               c["interval"], c["seed"])
            cost[sd] += seed
            if not c["ongoing"]:
                ws = (myd + 1) // 2
                cyc = TF._harvest_age(c, 1)
                upc = min(cap, 1 + max(0, cyc - ws + 1))
                t0 = sd
                while t0 + cyc <= lastday:          # full cycles, replant on harvest day
                    if t0 > sd:
                        cost[t0] += seed
                    u[t0 + cyc] += upc
                    t0 += cyc
                tail = lastday - t0                  # terminal partial cycle (theorem_features_v7)
                if t0 > sd and tail >= first:
                    cost[t0] += seed
                if tail >= first:
                    u[lastday] += min(cap, 1 + max(0, tail - ws + 1))
            else:
                cyc = first + (cap - 1) * interval + 1
                t0 = sd
                while t0 + first <= lastday:
                    if t0 > sd:
                        cost[t0] += seed
                    for i in range(cap):
                        e = t0 + first + i * interval
                        if e <= lastday:
                            u[e] += 1
                    t0 += cyc
        else:
            a = ANIMALS[r]
            first, interval, cap = a["first_yield_day"], a["interval"], a["max_held"]
            cost[sd] += a["cost"]
            k = 0
            while sd + first + k * interval <= lastday:
                u[sd + first + k * interval] += min(cap, 1 + (first - 1 if k == 0 else interval))
                k += 1
            for e in range(sd + 1, lastday + 1):
                fv[e] = 1
            for e in range(sd, lastday):
                fd[e] = 1
    if len(_NEW) > 4096:
        _NEW.clear()
    hit = _NEW[key] = (u, fv, fd, cost)
    return hit


# ----------------------------------------------------------------------------- helpers
def _cfg(configuration):
    g = lambda k, v: int(TF._setting(configuration, k, v))   # noqa: E731
    return dict(tpd=max(1, g("turnsPerDay", 24)), episode=g("episodeSteps", 720), cap=g("shedCapacity", 100),
                mult=g("farmHandCostMult", 1), unlock=max(1, g("townShopUnlockInterval", 3)),
                shop=max(1, g("townShopSellInterval", 4)), center=max(1, g("townCenterSellInterval", 24)),
                board=g("boardSize", 10))


def _step_of(obs, tpd):
    s = obs.get("step")
    return int(s) if s is not None else int(obs["day"]) * tpd + int(obs["hour"])


def _rates(shops):
    rate = [0] * NPI
    for shop in shops or ():
        for p, m in TF._SHOP_MULT[shop].items():
            rate[PIDX[p]] += m
    return rate


def _demand(a, b, rate, mornings, k):
    """Town units consumed at steps a..b (L728-749): visible shops, town centre, expected-new-shop prior."""
    if b < a:
        return [0.0] * NPI
    st = TF._ticks(a, b, k["shop"])
    ct = TF._ticks(a, b, k["center"])
    new = sum(TF._ticks(max(a, m * k["tpd"]), b, k["shop"]) for m in mornings)
    return [rate[i] * st + TF._CENTER[i] * ct + TF._EXPECTED_RATE[i] * new for i in range(NPI)]


def _consumed_at(step, shops, k):
    """Exact town consumption at one step with the given visible shops (L736-749)."""
    rate = _rates(shops)
    shop_tick = step % k["shop"] == 0
    cen_tick = step % k["center"] == 0
    return [rate[i] * shop_tick + TF._CENTER[i] * cen_tick for i in range(NPI)]


def _mind(src, dst):
    if not len(src) or not len(dst):
        return None
    s = np.asarray(src, np.int64)
    t = np.asarray(dst, np.int64)
    return int(np.abs(s[:, None, :] - t[None, :, :]).sum(-1).min())


# ----------------------------------------------------------------------------- main
def raw(actor_obs, configuration=None, history=()):
    """Unscaled (prod [9, N_PROD], glob [N_GLOB]) float64 and a detail dict (for tests)."""
    k = _cfg(configuration)
    tpd = k["tpd"]
    last = k["episode"] - 2
    lastday = last // tpd
    nd = lastday + 1
    s = _step_of(actor_obs, tpd)
    d = min(s // tpd, lastday)
    nj = lastday - d + 1
    day_end = min((d + 1) * tpd - 1, last)
    if len(_CACHE) > 20000:
        _CACHE.clear()
    me = int(actor_obs["player"])
    own, rival = actor_obs["farms"][me], actor_obs["farms"][1 - me]
    private = actor_obs.get("private") or {}
    shed = private.get("shed") or {}
    carried = {}
    for inv in private.get("inventories") or ():
        for item, n in (inv or {}).items():
            carried[item] = carried.get(item, 0) + int(n)
    curves = TF._curves(TF._market_params(actor_obs, configuration))
    inv0 = [int(actor_obs["market"]["inventory"][p]) for p in PRODUCTS]
    q0 = [curves[i].quote(inv0[i]) for i in range(NPI)]
    own_cash, riv_cash = float(own["money"]), float(rival["money"])

    sites = {"EMPTY": [], "WEED": [], "COOP": [], "PASTURE": []}
    A_o, F_o, B_o, C_o = _farm(own, d, lastday, nd, sites)
    A_r, F_r, B_r, C_r = _farm(rival, d, lastday, nd)
    held = [int(shed.get(p, 0)) + int(carried.get(p, 0)) for p in PRODUCTS]

    # ---- town demand per day d..lastday (Q2 demand law via TF helpers; shop unlock mornings as TF.compute)
    shops = (actor_obs.get("town") or {}).get("unlocked_shops") or []
    rate = _rates(shops)
    slots = max(0, TF.MAX_SHOP_INSTANCES - len(shops))
    mornings = []
    m = (d // k["unlock"] + 1) * k["unlock"]
    while slots > 0 and m * tpd - 1 <= last:
        mornings.append(m)
        slots -= 1
        m += k["unlock"]
    D = [_demand(max(s, e * tpd), min(e * tpd + tpd - 1, last), rate, mornings, k) for e in range(d, lastday + 1)]

    # ---- sell-as-produced market path (price law L192-206, commit L652-688, demand L728-749)
    Ao, Ar = A_o[:, d:].tolist(), A_r[:, d:].tolist()
    Fo, Fr = F_o[d:].tolist(), F_r[d:].tolist()
    ro = [[0.0] * nj for _ in range(NPI)]
    rr = [[0.0] * nj for _ in range(NPI)]
    rh = [0.0] * NPI
    post = [[0] * nj for _ in range(NPI)]
    fo, fr, wq = [0.0] * nj, [0.0] * nj, [0.0] * nj
    for i in range(NPI):
        cv = curves[i]
        floor = cv.floor
        I = float(inv0[i])
        ao, ar, roi, rri = Ao[i], Ar[i], ro[i], rr[i]
        for j in range(nj):
            o, r = int(ao[j]), int(ar[j])
            h = held[i] if j == 0 else 0
            tot = o + r + h
            ii = int(round(I))
            if tot > 0:
                R = cv.total(ii, ii + tot)
                roi[j] = R * o / tot
                rri[j] = R * r / tot
                if h:
                    rh[i] = R * h / tot
                if floor is None or ii + tot <= floor:
                    I += tot
                elif ii < floor:
                    I += floor - ii
                ii = int(round(I))
            post[i][j] = ii
            if i == WHEAT:
                nb = int(Fo[j] + Fr[j])
                if nb > 0:
                    C = cv.total(ii - nb, ii)
                    fo[j] = C * Fo[j] / nb
                    fr[j] = C * Fr[j] / nb
                    I -= nb
                    ii -= nb
                wq[j] = cv.quote(ii - 1)
            I -= D[j][i]

    prod = np.zeros((NPI, N_PROD))
    glob = np.zeros(N_GLOB)
    P = PC
    for i in range(NPI):
        row = prod[i]
        for b in range(5):
            row[P["own_" + BUCKETS[b]]] = B_o[i, b]
            row[P["rival_" + BUCKETS[b]]] = B_r[i, b]
        cu_o = np.cumsum(A_o[i, d:])
        cu_r = np.cumsum(A_r[i, d:])
        cv_o = np.cumsum(ro[i])
        cv_r = np.cumsum(rr[i])
        for hname, hd in zip(HWIN, HDAYS):
            j = min(hd, nj - 1)
            row[P["own_units_" + hname]] = cu_o[j]
            row[P["own_value_" + hname]] = cv_o[j]
            row[P["rival_units_" + hname]] = cu_r[j]
            row[P["rival_value_" + hname]] = cv_r[j]

    # ---- own liquidity and 3-day obligations
    wheat_cv = curves[WHEAT]
    liquid = own_cash + sum(curves[i].total(inv0[i], inv0[i] + int(shed.get(p, 0)))
                            for i, p in enumerate(PRODUCTS) if int(shed.get(p, 0)) > 0)
    feed3 = float(F_o[d:d + 3].sum())
    wheat_stock = int(shed.get("WHEAT", 0)) + int(carried.get("WHEAT", 0))
    feed_cash3 = max(0.0, feed3 - wheat_stock) * wheat_cv.quote(inv0[WHEAT] - 1)
    wages = k["mult"] * TF._FIB_PREFIX[min(int(own.get("hires_today", 0)), 159)]
    obligations = feed_cash3 + 3.0 * wages

    # ---- working capital, payback and attractiveness of one more producer
    units = [own["farmer"], *own.get("hands", [])]
    half = k["board"] // 2
    access = [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]
    # A route must use the SAME access square for arrival and departure.
    # Buying in the current market phase means PICKUP cannot act before s+1.
    pickup_eta = [(a, max(1, _mind(units, [a]))) for a in access]
    shed_used = sum(int(v) for v in shed.values())
    ratios = [0.0] * len(PRODUCERS)
    fcv = curves[FERT]
    for ri_, r in enumerate(PRODUCERS):
        i = PROD_OF_R[ri_]
        row = prod[i]
        if r in CROPS:
            if sites["EMPTY"]:
                dist, extra, site = _mind(units, sites["EMPTY"]), 0, 1.0
            elif sites["WEED"]:
                dist, extra, site = _mind(units, sites["WEED"]), 1, 0.5      # DIG first (L484-491)
            else:
                dist, extra, site = None, 0, 0.0
            # BUY_SEED fills in this step's market phase, PLANT from s+1, WATER the same day (L215-227, L783)
            today = dist is not None and s + max(1, dist + extra) + 1 <= day_end
        else:
            free = sites[STRUCTURE[r]]
            if free:
                tgt, extra, site = free, 0, 1.0
            elif sites["EMPTY"]:
                tgt, extra, site = sites["EMPTY"], 1, 0.5                   # BUILD_COOP/PASTURE (L493-503)
            elif sites["WEED"]:
                tgt, extra, site = sites["WEED"], 2, 0.5
            else:
                tgt, extra, site = [], 0, 0.0
            # BUY_ANIMAL lands in the shed (L679-688), PICKUP at a shed-access tile, walk, PLACE (L377-392)
            route = min((eta + _mind([a], tgt) + 1 + extra for a, eta in pickup_eta),
                        default=None) if tgt else None
            today = (route is not None and shed_used < k["cap"]
                     and s + route <= day_end)
        sd = d if today else d + 1
        u, fv, fd, cost = _new_schedule(r, sd, lastday, nd)
        cv = curves[i]
        post_i, post_f = post[i], post[FERT]
        first_sale = None
        cash_before = tot_rec = tot_cost = cum = 0.0
        payback = None
        for j in range(nj):
            e = d + j
            cj = cost[e] + fd[e] * wq[j]
            rj = 0.0
            if u[e]:
                rj += cv.total(post_i[j], post_i[j] + u[e])
            if fv[e]:
                rj += fcv.total(post_f[j], post_f[j] + fv[e])
            if first_sale is None:
                if u[e]:
                    first_sale = e
                else:
                    cash_before += cj
            tot_rec += rj
            tot_cost += cj
            cum += rj - cj
            if payback is None and e >= sd and tot_cost > 0 and cum >= 0:
                payback = j
        row[P["wc_cash_to_first_sale"]] = cash_before
        row[P["wc_days_to_first_sale"]] = (first_sale - d) if first_sale is not None else nj
        row[P["wc_first_sale_reachable"]] = float(first_sale is not None)
        row[P["wc_start_today"]] = float(today)
        row[P["wc_site_state"]] = site
        row[P["wc_reserve_after"]] = liquid - cash_before - obligations
        row[P["wc_payback_days"]] = payback if payback is not None else nj
        row[P["wc_net_to_end"]] = tot_rec - tot_cost
        ratios[ri_] = tot_rec / tot_cost if tot_cost > 0 else 0.0
        row[P["attract_ratio"]] = ratios[ri_]
        row[P["count_own"]] = C_o[i]
        row[P["count_rival"]] = C_r[i]
        row[P["count_total"]] = C_o[i] + C_r[i]
    order = sorted(range(len(PRODUCERS)), key=lambda x: (-ratios[x], x))
    for rank, ri_ in enumerate(order, 1):
        prod[PROD_OF_R[ri_], P["attract_rank"]] = rank

    # ---- visible production vs demand by horizon, flows and trends
    feed_all = F_o[d:] + F_r[d:]
    w72_end = min(s + 3 * tpd - 1, last)
    dem72 = _demand(s, w72_end, rate, mornings, k)
    hist_ok = [None, None]
    for lag in (1, 2):
        if len(history) >= lag and history[lag - 1] is not None:
            ho = history[lag - 1]
            try:
                if _step_of(ho, tpd) == s - lag:
                    hist_ok[lag - 1] = ho
            except (KeyError, TypeError, ValueError):
                pass
    cons = [None, None]
    if hist_ok[0] is not None:
        cons[0] = _consumed_at(s - 1, (hist_ok[0].get("town") or {}).get("unlocked_shops"), k)
    if hist_ok[0] is not None and hist_ok[1] is not None:
        c2 = _consumed_at(s - 2, (hist_ok[1].get("town") or {}).get("unlocked_shops"), k)
        cons[1] = [a + b for a, b in zip(cons[0], c2)]
    for i in range(NPI):
        row = prod[i]
        cv = curves[i]
        vis = A_o[i, d:] + A_r[i, d:]
        cp = np.cumsum(vis)
        cd = np.cumsum([D[j][i] for j in range(nj)])
        if i == WHEAT:
            cd = cd + np.cumsum(feed_all)
        for hname, hd in zip(HWIN[:3], HDAYS[:3]):
            j = min(hd, nj - 1)
            delta = cp[j] - cd[j]
            row[P["vis_production_" + hname]] = cp[j]
            row[P["demand_" + hname]] = cd[j]
            row[P["delta_" + hname]] = delta
            row[P["delta_quote_" + hname]] = cv.quote(inv0[i] + int(round(delta)))
        rate3 = float(vis[1:4].sum())
        pressure = float(vis[0]) + held[i]
        demand3 = dem72[i] + (float(feed_all[:3].sum()) if i == WHEAT else 0.0)
        net72 = rate3 + pressure - demand3
        dq3 = cv.quote(inv0[i] + int(round(net72))) - q0[i]
        dq_day = dq3 / 3.0
        rel = dq_day / max(1.0, float(q0[i]))
        row[P["flow_production_rate"]] = rate3 / 3.0
        row[P["flow_sell_pressure"]] = pressure
        row[P["flow_demand_rate"]] = demand3 / 3.0
        row[P["net_flow_per_day"]] = net72 / 3.0
        row[P["dq_per_day"]] = dq_day
        row[P["dq_rel_per_day"]] = rel
        row[P["trend_class"]] = (2 if rel >= FAST_REL else 1 if rel >= SLOW_REL else
                                 -2 if rel <= -FAST_REL else -1 if rel <= -SLOW_REL else 0)
        for lag in (1, 2):
            ho = hist_ok[lag - 1] if lag == 1 else (hist_ok[1] if hist_ok[0] is not None else None)
            if ho is None:
                continue
            hinv = int(ho["market"]["inventory"][PRODUCTS[i]])
            row[P[f"obs_dq_{lag}"]] = q0[i] - cv.quote(hinv)
            row[P[f"obs_net_sales_{lag}"]] = inv0[i] - hinv + cons[lag - 1][i]

    # ---- globals: equity, leads, attractiveness ranking
    G = GC
    held_value = sum(rh)
    own_board = float(sum(sum(x) for x in ro))
    riv_board = float(sum(sum(x) for x in rr))
    own_feed, riv_feed = sum(fo), sum(fr)
    own_cpp = own_cash + held_value + own_board - own_feed
    riv_cpp = riv_cash + riv_board - riv_feed
    lead = own_cpp - riv_cpp
    cash_lead = own_cash - riv_cash
    glob[G["own_liquid_budget"]] = liquid
    glob[G["own_obligations_3d"]] = obligations
    glob[G["own_held_value"]] = held_value
    glob[G["own_board_value_end"]] = own_board
    glob[G["own_feed_liability"]] = own_feed
    glob[G["own_cash_plus_pipeline"]] = own_cpp
    glob[G["rival_board_value_end"]] = riv_board
    glob[G["rival_feed_liability"]] = riv_feed
    glob[G["rival_cash_plus_pipeline"]] = riv_cpp
    glob[G["realizable_equity_lead"]] = lead
    glob[G["board_equity_lead"]] = own_cash + own_board - own_feed - riv_cpp
    glob[G["cash_equity_disagree"]] = 1.0 if (cash_lead > 0 > lead) else (-1.0 if cash_lead < 0 < lead else 0.0)
    rsum_o = np.cumsum(np.asarray(ro).sum(0))
    rsum_r = np.cumsum(np.asarray(rr).sum(0))
    fsum_o, fsum_r = np.cumsum(fo), np.cumsum(fr)
    for hname, hd in zip(HWIN, HDAYS):
        j = min(hd, nj - 1)
        glob[G["pipeline_lead_" + hname]] = (rsum_o[j] - fsum_o[j]) - (rsum_r[j] - fsum_r[j])
    picks = (order[0], order[1], order[-1], order[-2])
    for name, ri_ in zip(_RANKED, picks):
        i = PROD_OF_R[ri_]
        glob[G[f"{name}_is_{PRODUCERS[ri_]}"]] = 1.0
        glob[G[f"{name}_ratio"]] = ratios[ri_]
        glob[G[f"{name}_count_own"]] = C_o[i]
        glob[G[f"{name}_count_rival"]] = C_r[i]
        glob[G[f"{name}_count_total"]] = C_o[i] + C_r[i]
    glob[G["best_ratio_gap"]] = ratios[order[0]] - ratios[order[1]]
    glob[G["hist_lag1_avail"]] = float(hist_ok[0] is not None)
    glob[G["hist_lag2_avail"]] = float(hist_ok[0] is not None and hist_ok[1] is not None)
    detail = dict(d=d, s=s, nj=nj, A_own=A_o, A_riv=A_r, F_own=F_o, F_riv=F_r, B_own=B_o, B_riv=B_r,
                  D=D, post=post, ratios=ratios, liquid=liquid, obligations=obligations, held=held,
                  receipts_own=ro, receipts_riv=rr, feed_own=fo, feed_riv=fr, wheat_buy=wq)
    return prod, glob, detail


def compute(actor_obs, configuration, token_ids, history=()):
    prod, glob, _ = raw(actor_obs, configuration, history)
    sp = scale(prod, _PMODE, _PSCALE).astype(np.float16)
    sg = scale(glob, _GMODE, _GSCALE).astype(np.float16)
    if not (np.isfinite(sp).all() and np.isfinite(sg).all()):
        raise ValueError("non-finite cohorts feature")
    out = np.zeros((len(token_ids), WIDTH_TOKEN), np.float16)
    for n, tid in enumerate(token_ids):
        if tid == "global":
            out[n, N_PROD:] = sg
        elif tid.startswith("market/"):
            i = PIDX.get(tid[7:])
            if i is not None:
                out[n, :N_PROD] = sp[i]
    return {"token": out, "unit": np.zeros((MAX_UNITS, WIDTH_UNIT), np.float16)}


def warm(configuration=None):
    TF.warm(configuration)
