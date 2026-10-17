"""v8 feature family "gaps": the cheap, actor-visible inputs that game analysts keep asking for and no earlier family has
(GAPS.md lists every ask and where it is already covered). Contract: pbc-features-v8/CONTRACT.md.

Two entry points:

compute(actor_obs, configuration, token_ids, history=()) -> {"token": [N, WIDTH_TOKEN] f16, "unit": [25, 0] f16}
    Pure function of this seat's actor observation (history is accepted and ignored). Three blocks:
      c_*  cohort maturity per product, own and rival public board: mean age, mean first-output day, producer-days,
           remaining production events, held (harvestable now) yield and its spot value;
      mk_* shared-market first-order effect of ONE more unit in market inventory: quote drop, own cannibalisation,
           rival externality, relative value (own delta - rival delta);
      an_* own purchase-to-placement backlog;
      mr_* every cost and sale as a fraction of own current money (owner 2026-09-28), with money = 0 handling.
compute_long(actor_obs, configuration, token_ids, state) -> {"token": [N, WIDTH_LONG] f16, "unit": [25, 0] f16}
    Needs the whole-game history, so it carries an explicit per-seat GapsState (see its docstring for exactly what the
    deploy agent keeps). Blocks: running cash identity for both seats (land, labour, seeds, animals, trade), own
    per-product conservation and net account, rival market flow over the last 24/72 steps (I2), recent own income.

Engine = src/farm_neat/game/kaggriculture_engine_reference.py (E Lnnn): price law L192-206; HARVEST L446-473;
market commit L652-688 (SELL at a $1 quote adds no inventory L658-660; BUY_PRODUCT quotes inv-1 L598-601; BUY_SEED /
BUY_ANIMAL fail when money < price L673-688); hire L698-709 (fib wage); land L712-725; town consumption L728-749
(price-independent fixed units per tick); animal refresh and escape L805-834; day end L860-882 (carried goods
deposited, hands dismissed, hires_today reset).
Schedules (remaining production events, future units under full service) reuse v8_cohorts._entry, the per-tile
schedule the cohorts family already verified against the engine; own-step flows reuse v8_flags.analyse_step.
Column meanings: COLUMNS-gaps.md.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from farm_neat.economy.heuristic.constants import ANIMALS, CROPS, LAND_PRICES, SHOPS, fib
from farm_neat.economy.heuristic.market import price_at

import v8_cohorts as COH
import v8_flags as FLAGS

PRODUCTS = COH.PRODUCTS                      # WHEAT CARROT TOMATO STRAWBERRY MELON EGG MILK WOOL FERTILIZER
NP = len(PRODUCTS)
PIDX = COH.PIDX
FERT = PIDX["FERTILIZER"]
MAX_UNITS = 25
SPECIES_OF = {a["product"]: name for name, a in ANIMALS.items()}         # EGG -> GOOSE, MILK -> COW, WOOL -> SHEEP
STRUCTURE_OF = {name: a["structure"] for name, a in ANIMALS.items()}      # GOOSE -> COOP, COW/SHEEP -> PASTURE
RATIO_CAP = 1.0e6                                                        # ratio clip before log1p (<= 13.8)

# (name, mode, scale): mode slog = sign(x) log1p(|x|/s); lin = x/s clipped to [-lo, hi]; flag = 0/1; rlog = already
# log1p(min(ratio, RATIO_CAP)) (ratios are built by _ratio).
PROD_COLS = (
    ("c_own_mean_age", "lin", 10.0), ("c_rival_mean_age", "lin", 10.0),
    ("c_own_mean_first_day", "lin", 30.0), ("c_rival_mean_first_day", "lin", 30.0),
    ("c_own_producer_days", "slog", 10.0), ("c_rival_producer_days", "slog", 10.0),
    ("c_own_pulses_left", "slog", 1.0), ("c_rival_pulses_left", "slog", 1.0),
    ("c_own_held_units", "slog", 1.0), ("c_rival_held_units", "slog", 1.0),
    ("c_own_held_value", "slog", 100.0), ("c_rival_held_value", "slog", 100.0),
    ("mk_unit_impact", "slog", 1.0), ("mk_own_cannibal_1u", "slog", 10.0), ("mk_rival_ext_1u", "slog", 10.0),
    ("mk_rel_value_1u", "slog", 10.0),
    ("an_own_unplaced", "slog", 1.0), ("an_site_backlog", "slog", 1.0),
    ("mr_producer_unit", "rlog", 1.0), ("mr_producer_unit_cannot", "flag", 1.0),
    ("mr_producer_lot", "rlog", 1.0), ("mr_producer_lot_cannot", "flag", 1.0),
    ("mr_buy_unit", "rlog", 1.0), ("mr_buy_unit_cannot", "flag", 1.0),
    ("mr_buy_lot", "rlog", 1.0), ("mr_buy_lot_cannot", "flag", 1.0),
    ("mr_sell_unit", "rlog", 1.0), ("mr_sell_held", "rlog", 1.0),
)
GLOB_COLS = (
    ("g_rival_labour_today", "slog", 100.0), ("g_own_board_capital", "slog", 1000.0),
    ("g_rival_board_capital", "slog", 1000.0), ("g_held_value_lead", "slog", 1000.0),
    ("mr_money_zero", "flag", 1.0), ("mr_land", "rlog", 1.0), ("mr_hire", "rlog", 1.0), ("mr_shed_value", "rlog", 1.0),
)
TOKEN_COLS = tuple(n for n, _, _ in PROD_COLS) + tuple(n for n, _, _ in GLOB_COLS)
UNIT_COLS: tuple = ()
N_PROD, N_GLOB = len(PROD_COLS), len(GLOB_COLS)
WIDTH_TOKEN, WIDTH_UNIT = len(TOKEN_COLS), len(UNIT_COLS)

LONG_PROD_COLS = (
    ("L_own_produced", "slog", 10.0), ("L_own_consumed", "slog", 10.0), ("L_own_bought", "slog", 10.0),
    ("L_own_sold", "slog", 10.0), ("L_own_trade_account", "slog", 1000.0), ("L_own_capital_spent", "slog", 1000.0),
    ("L_own_net_account", "slog", 1000.0),
    ("L_rival_net_24", "slog", 1.0), ("L_rival_deficit_24", "slog", 1.0), ("L_rival_known_24", "lin", 1.0),
    ("L_rival_net_72", "slog", 1.0), ("L_rival_deficit_72", "slog", 1.0), ("L_rival_known_72", "lin", 1.0),
    ("L_rival_net_all", "slog", 10.0), ("L_sell_held_vs_inc24", "rlog", 1.0),
)
LONG_GLOB_COLS = (
    ("L_own_land", "slog", 1000.0), ("L_own_labour", "slog", 1000.0), ("L_own_seeds", "slog", 1000.0),
    ("L_own_animals", "slog", 1000.0), ("L_own_trade_net", "slog", 1000.0), ("L_own_trade_unattributed", "slog", 1000.0),
    ("L_rival_land", "slog", 1000.0), ("L_rival_labour", "slog", 1000.0), ("L_rival_seeds_seen", "slog", 1000.0),
    ("L_rival_animals_seen", "slog", 1000.0), ("L_rival_trade_residual", "slog", 1000.0),
    ("L_trade_lead", "slog", 1000.0), ("L_spend_lead", "slog", 1000.0),
    ("L_inc24", "slog", 1000.0), ("L_inc24_nonpos", "flag", 1.0),
    ("L_own_flow_known_frac", "lin", 1.0), ("L_contiguous", "flag", 1.0),
)
LONG_TOKEN_COLS = tuple(n for n, _, _ in LONG_PROD_COLS) + tuple(n for n, _, _ in LONG_GLOB_COLS)
N_LPROD, N_LGLOB = len(LONG_PROD_COLS), len(LONG_GLOB_COLS)
WIDTH_LONG = len(LONG_TOKEN_COLS)
WINDOWS = (24, 72)


def _scaler(cols):
    mode = np.array([{"slog": 0, "lin": 1, "flag": 2, "rlog": 3}[m] for _, m, _ in cols])
    return mode, np.array([s for _, _, s in cols], np.float64)


_PM, _PS = _scaler(PROD_COLS)
_GM, _GS = _scaler(GLOB_COLS)
_LPM, _LPS = _scaler(LONG_PROD_COLS)
_LGM, _LGS = _scaler(LONG_GLOB_COLS)


def _scale(x, mode, sc):
    """float64 raw -> float16 scaled. slog: sign*log1p(|x|/s); lin: x/s clipped to [-3, 3]; flag/rlog: as built."""
    y = x / sc
    out = np.where(mode == 0, np.sign(y) * np.log1p(np.abs(y)), x)
    out = np.where(mode == 1, np.clip(y, -3.0, 3.0), out)
    out = out.astype(np.float16)
    if not np.isfinite(out).all():
        raise ValueError("non-finite gaps feature")
    return out


def _ratio(amount, money):
    """log1p(min(amount / max(money, 1), RATIO_CAP)) for amount >= 0. money <= 0 divides by 1, so the column then carries
    log1p(amount) itself and mr_money_zero says the relative value is unbounded."""
    return math.log1p(min(max(0.0, float(amount)) / max(float(money), 1.0), RATIO_CAP))


_TOK = {"global": -1, **{f"market/{p}": i for i, p in enumerate(PRODUCTS)}}


def _place(token_ids, prod, glob, n_prod, width):
    """Product rows on market/<PRODUCT> tokens, the global row on "global", zeros elsewhere."""
    out = np.zeros((len(token_ids), width), np.float16)
    get = _TOK.get
    for n, tid in enumerate(token_ids):
        i = get(tid)
        if i is None:
            continue
        if i < 0:
            out[n, n_prod:] = glob
        else:
            out[n, :n_prod] = prod[i]
    return out


def _horizon(configuration):
    k = COH._cfg(configuration)
    lastday = (k["episode"] - 2) // k["tpd"]
    return k, lastday, lastday + 1


def _carried(private):
    out = {}
    for inv in private.get("inventories") or ():
        for item, n in (inv or {}).items():
            out[item] = out.get(item, 0) + int(n)
    return out


# ----------------------------------------------------------------------------------------------- per-step block
def _scan(farm, d, lastday, nd):
    """Per product (index PIDX): producers n, sum age, sum first-output day, producer-days, remaining production events,
    held units, future full-service units (held included); plus capital at input cost, animal count, free structures,
    empty tiles."""
    n = [0] * NP
    age = [0.0] * NP
    first = [0.0] * NP
    pdays = [0.0] * NP
    pulses = [0.0] * NP
    held = [0.0] * NP
    fut = [0.0] * NP
    capital = 0.0
    animals = 0
    free = {"COOP": 0, "PASTURE": 0}
    empty = 0
    memo = {}
    for row in farm["tiles"]:
        for t in row:
            if t is None:
                empty += 1
                continue
            if not isinstance(t, dict):
                continue
            kind = t.get("kind")
            if kind == "PLANT":
                c = CROPS[t["crop"]]
                pi = PIDX[t["crop"]]
                start = int(t["planted_day"])
                fd = start + c["first_yield_day"]
                y = int(t.get("yield_units") or 0)
                if y > 0 and (c["ongoing"] or d - start >= c["first_yield_day"]):
                    held[pi] += y                              # HARVEST gives y now (E L446-473)
                capital += c["seed"]
            elif "animal" in t:
                a = ANIMALS[t["animal"]]
                pi = PIDX[a["product"]]
                start = int(t["placed_day"])
                fd = start + a["first_yield_day"]
                held[pi] += int(t.get("yield_units") or 0)      # HARVEST on an animal gives y (E L469-472)
                held[FERT] += bool(t.get("fertilizer_available"))
                capital += a["cost"]
                animals += 1
            else:
                if kind in free:
                    free[kind] += 1
                continue
            n[pi] += 1
            age[pi] += d - start
            first[pi] += fd
            pdays[pi] += max(0, d - fd)
            e = COH._entry(t, d, lastday, nd)
            m = memo.get(id(e))
            if m is None:
                vec, fvec = e[2], e[4]
                m = memo[id(e)] = (float(np.count_nonzero(vec[d + 1:])), float(vec[d:].sum()),
                                   0.0 if fvec is None else float(fvec[d:].sum()))
            pulses[pi] += m[0]
            fut[pi] += m[1]
            fut[FERT] += m[2]
    return dict(n=n, age=age, first=first, pdays=pdays, pulses=pulses, held=held, fut=fut, capital=capital,
                animals=animals, free=free, empty=empty)


def raw(actor_obs, configuration=None):
    """Unscaled (prod [9, N_PROD], glob [N_GLOB]) float64."""
    k, lastday, nd = _horizon(configuration)
    if len(COH._CACHE) > 20000:                  # same bound cohorts applies, in case gaps runs without it
        COH._CACHE.clear()
    s = COH._step_of(actor_obs, k["tpd"])
    d = min(s // k["tpd"], lastday)
    me = int(actor_obs["player"])
    own, rival = actor_obs["farms"][me], actor_obs["farms"][1 - me]
    private = actor_obs.get("private") or {}
    shed = private.get("shed") or {}
    carried = _carried(private)
    market = actor_obs.get("market") or {}
    inv = {p: int(market["inventory"][p]) for p in PRODUCTS}
    params = FLAGS._params(actor_obs, configuration)
    quote = {p: float((market.get("prices") or {}).get(p, price_at(p, inv[p], params))) for p in PRODUCTS}
    money = float(own["money"])
    so, sr = _scan(own, d, lastday, nd), _scan(rival, d, lastday, nd)
    prod = np.zeros((NP, N_PROD))
    col = {c: i for i, (c, _, _) in enumerate(PROD_COLS)}
    for i, p in enumerate(PRODUCTS):
        r = prod[i]
        if i != FERT:
            for tag, sc in (("own", so), ("rival", sr)):
                if sc["n"][i]:
                    r[col[f"c_{tag}_mean_age"]] = sc["age"][i] / sc["n"][i]
                    r[col[f"c_{tag}_mean_first_day"]] = sc["first"][i] / sc["n"][i]
                r[col[f"c_{tag}_producer_days"]] = sc["pdays"][i]
                r[col[f"c_{tag}_pulses_left"]] = sc["pulses"][i]
        for tag, sc in (("own", so), ("rival", sr)):
            r[col[f"c_{tag}_held_units"]] = sc["held"][i]
            r[col[f"c_{tag}_held_value"]] = sc["held"][i] * quote[p]
        # one more unit in market inventory lowers every later quote by the law's slope (town demand is
        # price-independent, E L728-749, so the +1 persists); first order at today's inventory.
        slope = float(price_at(p, inv[p], params) - price_at(p, inv[p] + 1, params)) if price_at(p, inv[p], params) > 1 else 0.0
        own_units = so["fut"][i] + int(shed.get(p, 0)) + int(carried.get(p, 0))
        riv_units = sr["fut"][i]
        r[col["mk_unit_impact"]] = slope
        r[col["mk_own_cannibal_1u"]] = slope * own_units
        r[col["mk_rival_ext_1u"]] = slope * riv_units
        r[col["mk_rel_value_1u"]] = quote[p] - slope * own_units + slope * riv_units
        # money ratios
        if p in CROPS:
            cost = CROPS[p]["seed"]
            lot = cost * so["empty"]
        elif p in SPECIES_OF:
            sp = SPECIES_OF[p]
            cost = ANIMALS[sp]["cost"]
            unplaced = int(shed.get(sp, 0)) + int(carried.get(sp, 0))
            structure = STRUCTURE_OF[sp]
            need = sum(int(shed.get(a, 0)) + int(carried.get(a, 0)) for a in ANIMALS if STRUCTURE_OF[a] == structure)
            r[col["an_own_unplaced"]] = unplaced
            r[col["an_site_backlog"]] = need - so["free"][structure]
            lot = cost * so["free"][structure]
        else:
            cost = lot = 0
        if cost:
            r[col["mr_producer_unit"]] = _ratio(cost, money)
            r[col["mr_producer_unit_cannot"]] = float(cost > money)
            r[col["mr_producer_lot"]] = _ratio(lot, money)
            r[col["mr_producer_lot_cannot"]] = float(lot > money)
        if p in ("WHEAT", "FERTILIZER"):
            unit = price_at(p, inv[p] - 1, params)                      # BUY_PRODUCT quote, E L598-601
            r[col["mr_buy_unit"]] = _ratio(unit, money)
            r[col["mr_buy_unit_cannot"]] = float(unit > money)
            if p == "WHEAT":                                             # lot = one day of feed for own animals
                feed = sum(price_at(p, inv[p] - j, params) for j in range(1, so["animals"] + 1))
                r[col["mr_buy_lot"]] = _ratio(feed, money)
                r[col["mr_buy_lot_cannot"]] = float(feed > money)
        r[col["mr_sell_unit"]] = _ratio(quote[p], money)
        r[col["mr_sell_held"]] = _ratio(quote[p] * int(shed.get(p, 0)), money)
    glob = np.zeros(N_GLOB)
    gc = {c: i for i, (c, _, _) in enumerate(GLOB_COLS)}
    glob[gc["g_rival_labour_today"]] = k["mult"] * sum(fib(j) for j in range(int(rival.get("hires_today", 0))))
    glob[gc["g_own_board_capital"]] = so["capital"]
    glob[gc["g_rival_board_capital"]] = sr["capital"]
    glob[gc["g_held_value_lead"]] = sum((so["held"][i] - sr["held"][i]) * quote[p] for i, p in enumerate(PRODUCTS))
    glob[gc["mr_money_zero"]] = float(money <= 0)
    owned = len(own.get("unlocked_quadrants") or [])
    if 1 <= owned <= len(LAND_PRICES):
        glob[gc["mr_land"]] = _ratio(LAND_PRICES[owned - 1], money)
    glob[gc["mr_hire"]] = _ratio(k["mult"] * fib(int(own.get("hires_today", 0))), money)
    glob[gc["mr_shed_value"]] = _ratio(sum(quote[p] * int(shed.get(p, 0)) for p in PRODUCTS), money)
    return prod, glob


def compute(actor_obs, configuration, token_ids, history=()):
    prod, glob = raw(actor_obs, configuration)
    return {"token": _place(token_ids, _scale(prod, _PM, _PS), _scale(glob, _GM, _GS), N_PROD, WIDTH_TOKEN),
            "unit": np.zeros((MAX_UNITS, WIDTH_UNIT), np.float16)}


# ----------------------------------------------------------------------------------------------- long block
class _Seat:
    """Cumulative spend of one farm, from its public (rival) or private (own) view."""
    __slots__ = ("labour_closed", "day", "hires", "seeds", "animals", "seed_by", "animal_by")

    def __init__(self):
        self.labour_closed = 0.0      # wages of finished days
        self.day = 0
        self.hires = 0                # max hires_today seen on self.day
        self.seeds = 0.0              # own: seeds bought; rival: seeds planted (lower bound)
        self.animals = 0.0            # own: animals bought; rival: animals placed (lower bound)
        self.seed_by = [0.0] * NP     # per product index
        self.animal_by = [0.0] * NP

    def labour(self, farm, day, mult):
        if day != self.day:
            self.labour_closed += mult * sum(fib(j) for j in range(self.hires))
            self.day, self.hires = day, 0
        self.hires = max(self.hires, int(farm.get("hires_today", 0)))
        return self.labour_closed + mult * sum(fib(j) for j in range(self.hires))


class GapsState:
    """Per-seat carry for compute_long. What the deploy agent must keep: ONE instance per (episode, seat), created
    before step 0 (or reset automatically when step 0 or an earlier step arrives), passed to compute_long exactly once per
    observed step in step order. It holds:
      prev          the seat's previous actor observation (the full dict compute_long received, not a copy of parts);
      ring          deque(maxlen=72) of (rival_flow[9] int, known[9] bool) per interpreter step;
      trade_ring    deque(maxlen=24) of own trade cash per step (sales - product buys, from the cash identity);
      own, rival    _Seat accumulators (wages by day, seeds, animals, per-product capital);
      own flows     cumulative produced/consumed/bought/sold units per product, trade account per product,
                    unattributed trade, rival net units all game, counters of observed / flow-known steps;
      last          (step, output) of the last call, so a repeated call for the same step is idempotent.
    Memory: one observation (~35 KB as JSON) + ~2 KB of arrays. Missing steps (timeouts) are tolerated: the transition
    is marked unknown and L_contiguous drops to 0; it never raises."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev = None
        self.ring = deque(maxlen=max(WINDOWS))
        self.trade_ring = deque(maxlen=24)
        self.own, self.rival = _Seat(), _Seat()
        self.produced = [0.0] * NP
        self.consumed = [0.0] * NP
        self.bought = [0.0] * NP
        self.sold = [0.0] * NP
        self.account = [0.0] * NP
        self.unattributed = 0.0
        self.rival_all = [0.0] * NP
        self.trade_prev = 0.0
        self.n_steps = 0
        self.n_flow_known = 0
        self.contiguous = True
        self.last = None


def _animal_totals(farm, private):
    tot = {a: 0 for a in ANIMALS}
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict) and "animal" in t:
                tot[t["animal"]] += 1
    for a in ANIMALS:
        tot[a] += int((private.get("shed") or {}).get(a, 0))
    for inv in private.get("inventories") or ():
        for a in ANIMALS:
            tot[a] += int((inv or {}).get(a, 0))
    return tot


def _board_changes(fp, fc, prev_day):
    """New plantings per crop, new placements and escapes per species between two views of one farm (E L215-242,
    L813-819). A tile counts as planted when it holds a plant that is not the same plant (crop, planted_day) as before
    and was planted on or after the previous observation's day."""
    planted, placed, escaped = {}, {}, {}
    for yy, (rc, rp) in enumerate(zip(fc["tiles"], fp["tiles"])):
        for xx in range(len(rc)):
            a, b = rp[xx], rc[xx]
            if a is b or a == b:
                continue
            if FLAGS._is_plant(b) and not FLAGS._same_plant(a, b) and int(b["planted_day"]) >= prev_day:
                planted[b["crop"]] = planted.get(b["crop"], 0) + 1
            if FLAGS._is_animal(b) and not FLAGS._same_animal(a, b):
                placed[b["animal"]] = placed.get(b["animal"], 0) + 1
            if FLAGS._is_animal(a) and not FLAGS._is_animal(b):
                escaped[a["animal"]] = escaped.get(a["animal"], 0) + 1
    return planted, placed, escaped


def _demand(shops, s, k):
    """Town consumption at interpreter step s with the shops visible then (E L728-749)."""
    dem = [0] * NP
    if s % k["shop"] == 0:
        for shop in shops:
            prods = SHOPS[shop]
            for p in prods:
                dem[PIDX[p]] += 2 if len(prods) == 1 else 1
    if s % k["center"] == 0:
        for i, p in enumerate(PRODUCTS):
            if p != "FERTILIZER":
                dem[i] += 1
    return dem


def _update(state, obs, configuration, k):
    t = int(obs["step"])
    me = int(obs["player"])
    own, rival = obs["farms"][me], obs["farms"][1 - me]
    private = obs.get("private") or {}
    day = t // k["tpd"]
    prev = state.prev
    step_known = prev is not None and int(prev["step"]) == t - 1 and int(prev["player"]) == me
    state.n_steps += 1
    if prev is None:
        state.contiguous = t == 0
    elif not step_known:
        state.contiguous = False
    # --- spend accumulators (gap-tolerant: board and inventory differences)
    state.own.labour(own, day, k["mult"])
    state.rival.labour(rival, day, k["mult"])
    if prev is not None:
        pday = int(prev["step"]) // k["tpd"]
        fp_own, fp_riv = prev["farms"][me], prev["farms"][1 - me]
        pp = prev.get("private") or {}
        planted, _, escaped = _board_changes(fp_own, own, pday)
        sp, sc = pp.get("seeds") or {}, private.get("seeds") or {}
        for c in CROPS:
            nb = int(sc.get(c, 0)) - int(sp.get(c, 0)) + planted.get(c, 0)     # PLANT consumes one seed (E L425-428)
            if nb > 0:
                state.own.seeds += nb * CROPS[c]["seed"]
                state.own.seed_by[PIDX[c]] += nb * CROPS[c]["seed"]
        ta, tb = _animal_totals(fp_own, pp), _animal_totals(own, private)
        for a in ANIMALS:
            nb = tb[a] - ta[a] + escaped.get(a, 0)
            if nb > 0:
                state.own.animals += nb * ANIMALS[a]["cost"]
                state.own.animal_by[PIDX[ANIMALS[a]["product"]]] += nb * ANIMALS[a]["cost"]
        r_planted, r_placed, _ = _board_changes(fp_riv, rival, pday)
        for c, nb in r_planted.items():
            state.rival.seeds += nb * CROPS[c]["seed"]
        for a, nb in r_placed.items():
            state.rival.animals += nb * ANIMALS[a]["cost"]
    start = float(COH.TF._setting(configuration, "startingMoney", 3000))
    land_o = float(sum(LAND_PRICES[: max(0, len(own.get("unlocked_quadrants") or []) - 1)]))
    trade_net = (float(own["money"]) - start + land_o + state.own.labour(own, day, k["mult"])
                 + state.own.seeds + state.own.animals)
    trade = trade_net - state.trade_prev if prev is not None else trade_net
    state.trade_prev = trade_net
    # --- per-step own flows and rival flow
    if prev is None:
        if t > 0:
            state.unattributed += trade
        return trade_net, land_o, start
    if not step_known:
        state.unattributed += trade
        for _ in range(min(max(1, t - int(prev["step"])), max(WINDOWS))):
            state.ring.append((np.zeros(NP), np.zeros(NP, bool)))
            state.trade_ring.append(0.0)
        return trade_net, land_o, start
    state.trade_ring.append(trade)
    s = FLAGS.analyse_step(prev, obs, configuration) if t % k["tpd"] != 0 else None
    mn = None
    if s is not None:
        for p, v in s["produced"].items():
            if p in PIDX:
                state.produced[PIDX[p]] += v
        for p, v in s["consumed"].items():
            if p in PIDX:
                state.consumed[PIDX[p]] += v
        if s["exact"]:
            mn = {p: v for p, v in s["market_net"].items() if p in PIDX}
    if mn is None and abs(trade) < 1e-9:
        mn = {}                                   # no own trade this step (cash identity closes at 0)
    pinv = prev["market"]["inventory"]
    cinv = obs["market"]["inventory"]
    pparams = FLAGS._params(prev, configuration)
    known = np.zeros(NP, bool)
    flow = np.zeros(NP)
    if mn is not None:
        state.n_flow_known += 1
        dem = _demand((prev.get("town") or {}).get("unlocked_shops") or [], t - 1, k)
        est = 0.0
        for i, p in enumerate(PRODUCTS):
            v = mn.get(p, 0)
            if v > 0:
                state.bought[i] += v
            elif v < 0:
                state.sold[i] += -v
            total = int(cinv[p]) - int(pinv[p]) + dem[i]           # both seats' net units into inventory
            if v < 0:
                # A $1 sale adds no inventory (E L658-660), so own sales are displacement only if no own unit can
                # have sold at $1. The quote only falls as inventory rises; before town consumption the market held
                # cinv + dem, and within the step it can have been higher only by units bought later in the queue,
                # which land in a shed of capacity cap (own and rival: 2 * cap; only WHEAT/FERTILIZER can be bought).
                peak = int(cinv[p]) + dem[i] + (2 * k["cap"] if p in ("WHEAT", "FERTILIZER") else 0)
                if price_at(p, peak, pparams) <= 1:
                    continue
            known[i] = True
            flow[i] = total + v                                    # rival = total - own; own adds -market_net
        if len(mn) == 1:
            (p, _), = mn.items()
            state.account[PIDX[p]] += trade
        elif mn:
            for p, v in mn.items():
                q = price_at(p, int(pinv[p]) - 1, pparams) if v > 0 else float(prev["market"]["prices"][p])
                e = -v * q
                state.account[PIDX[p]] += e
                est += e
            state.unattributed += trade - est
        elif abs(trade) >= 1e-9:
            state.unattributed += trade
    else:
        state.unattributed += trade
    state.ring.append((flow, known))
    for i in range(NP):
        if known[i]:
            state.rival_all[i] += flow[i]
    return trade_net, land_o, start


def raw_long(actor_obs, configuration, state):
    """Update `state` with this step's observation and return unscaled (prod [9, N_LPROD], glob [N_LGLOB])."""
    k, lastday, nd = _horizon(configuration)
    t = int(actor_obs["step"])
    if state.prev is not None and (t == 0 or t < int(state.prev["step"])
                                   or int(state.prev["player"]) != int(actor_obs["player"])):
        state.reset()
    trade_net, land_o, start = _update(state, actor_obs, configuration, k)
    state.prev = actor_obs
    me = int(actor_obs["player"])
    own, rival = actor_obs["farms"][me], actor_obs["farms"][1 - me]
    private = actor_obs.get("private") or {}
    shed = private.get("shed") or {}
    day = t // k["tpd"]
    prod = np.zeros((NP, N_LPROD))
    col = {c: i for i, (c, _, _) in enumerate(LONG_PROD_COLS)}
    ring = list(state.ring)
    inc24 = float(sum(state.trade_ring))
    quote = (actor_obs.get("market") or {}).get("prices") or {}
    win = {}
    for w in WINDOWS:
        h = ring[-w:]
        if h:
            f = np.stack([x[0] for x in h])
            kn = np.stack([x[1] for x in h])
            c = np.cumsum(f, 0)
            win[w] = (c[-1], np.maximum(0.0, -np.minimum(0.0, c.min(0))), kn.mean(0))
        else:
            win[w] = (np.zeros(NP), np.zeros(NP), np.zeros(NP))
    for i, p in enumerate(PRODUCTS):
        r = prod[i]
        r[col["L_own_produced"]] = state.produced[i]
        r[col["L_own_consumed"]] = state.consumed[i]
        r[col["L_own_bought"]] = state.bought[i]
        r[col["L_own_sold"]] = state.sold[i]
        cap = state.own.seed_by[i] + state.own.animal_by[i]
        r[col["L_own_trade_account"]] = state.account[i]
        r[col["L_own_capital_spent"]] = cap
        r[col["L_own_net_account"]] = state.account[i] - cap
        for w in WINDOWS:
            r[col[f"L_rival_net_{w}"]] = win[w][0][i]
            r[col[f"L_rival_deficit_{w}"]] = win[w][1][i]
            r[col[f"L_rival_known_{w}"]] = win[w][2][i]
        r[col["L_rival_net_all"]] = state.rival_all[i]
        r[col["L_sell_held_vs_inc24"]] = _ratio(float(quote.get(p, 0)) * int(shed.get(p, 0)), inc24)
    glob = np.zeros(N_LGLOB)
    gc = {c: i for i, (c, _, _) in enumerate(LONG_GLOB_COLS)}
    labour_o = state.own.labour(own, day, k["mult"])
    labour_r = state.rival.labour(rival, day, k["mult"])
    land_r = float(sum(LAND_PRICES[: max(0, len(rival.get("unlocked_quadrants") or []) - 1)]))
    resid_r = float(rival["money"]) - start + land_r + labour_r + state.rival.seeds + state.rival.animals
    glob[gc["L_own_land"]] = land_o
    glob[gc["L_own_labour"]] = labour_o
    glob[gc["L_own_seeds"]] = state.own.seeds
    glob[gc["L_own_animals"]] = state.own.animals
    glob[gc["L_own_trade_net"]] = trade_net
    glob[gc["L_own_trade_unattributed"]] = state.unattributed
    glob[gc["L_rival_land"]] = land_r
    glob[gc["L_rival_labour"]] = labour_r
    glob[gc["L_rival_seeds_seen"]] = state.rival.seeds
    glob[gc["L_rival_animals_seen"]] = state.rival.animals
    glob[gc["L_rival_trade_residual"]] = resid_r
    glob[gc["L_trade_lead"]] = trade_net - resid_r
    glob[gc["L_spend_lead"]] = ((land_o + labour_o + state.own.seeds + state.own.animals)
                                - (land_r + labour_r + state.rival.seeds + state.rival.animals))
    glob[gc["L_inc24"]] = inc24
    glob[gc["L_inc24_nonpos"]] = float(inc24 <= 0)
    glob[gc["L_own_flow_known_frac"]] = state.n_flow_known / max(1, state.n_steps - 1)
    glob[gc["L_contiguous"]] = float(state.contiguous)
    return prod, glob


def compute_long(actor_obs, configuration, token_ids, state):
    """Long-history columns (see GapsState). Call once per step, in order; a repeated call for the same step returns the
    same arrays without updating the state again."""
    t = int(actor_obs["step"])
    if state.last is not None and state.last[0] == (t, int(actor_obs["player"])) and state.prev is not None \
            and int(state.prev["step"]) == t:
        prod, glob = state.last[1]
    else:
        prod, glob = raw_long(actor_obs, configuration, state)
        prod, glob = _scale(prod, _LPM, _LPS), _scale(glob, _LGM, _LGS)
        state.last = ((t, int(actor_obs["player"])), (prod, glob))
    return {"token": _place(token_ids, prod, glob, N_LPROD, WIDTH_LONG),
            "unit": np.zeros((MAX_UNITS, 0), np.float16)}
