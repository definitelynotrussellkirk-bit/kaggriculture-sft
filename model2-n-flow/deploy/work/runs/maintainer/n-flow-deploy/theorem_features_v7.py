"""Proven game facts as input features for ONE seat's actor-visible observation.

``compute(actor_obs, config)`` returns raw (unscaled) float32 arrays:

* ``global``    (len(GLOBAL_NAMES),)             clocks, cash, labour, land, shed, liquidation
* ``product``   (9, len(PRODUCT_NAMES))          rows in PRODUCTS order: quotes, k-unit receipts,
                                                 exact visible town demand, floor lock, stocks
* ``tile_own`` / ``tile_rival`` (100, len(TILE_NAMES))  row index y*10+x: crop/animal clocks
* ``structure`` (8, len(STRUCTURE_NAMES))        rows in PRODUCERS order (5 crop plants, then the
                                                 GOOSE/COW/SHEEP animal structures): unit economics
                                                 of starting that producer today, counts, stocks

``SCALES[block]`` holds a divisor per column for consumers that want O(1) inputs;
``normalized(features)`` applies it. Every column is documented in ``SPEC`` with its formula
and its engine line / theorem source. Engine = src/farm_neat/game/kaggriculture_engine_reference.py
(sha256 bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e); line numbers refer to
those bytes. The input is the ``training.bc.actor_observation`` whitelist: own private
shed/seeds/inventories, both public farms, market, visible shops. Nothing here reads the rival's
private state, the episode seed, future shop identities or outcomes. The two ``*_expected_new``
and ``*_max_new`` demand columns use only the rule-given uniform shop draw (a prior), never the
realised draw. Features, not constraints: nothing here masks an action.

Adapted sources (no farm-repo code is ported): season closed forms re-implement
src/farm_neat/evaluation/unit_economics.py ``formula`` (sha256 6a657b838c8e7c85b6d145a149c7811cbd30fd4c3384c2587c7af257cfdd6c8f);
the price function copies the engine's ``_shape``/``market_price`` arithmetic; tables come from
src/farm_neat/economy/heuristic/constants.py (sha256 1818d486b89a46844d5dde9c6d2801e7886f9ae85c12987bb229261abb76e850).
test_theorem_features.py asserts all three equal the engine / theorem module.
Cost: the first call builds exact price tables (~40 ms; ``warm()`` does it early); later calls
reuse a per-(tile, day) cache.
"""
from __future__ import annotations

import math

import numpy as np

from farm_neat.economy.heuristic.constants import (
    ANIMALS, CROPS, LAND_PRICES, MARKET_PARAMS, PRODUCTS, SHOPS)

ENGINE_SHA256 = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"
HINGE_GAIN = 8.0          # engine L58
PRICE_FLOOR = 1           # engine L206
MAX_SHOP_INSTANCES = 8    # engine L890
BUYABLE = ("WHEAT", "FERTILIZER")   # engine L600-601
PRODUCERS = tuple(CROPS) + tuple(ANIMALS)
PRODUCT_OF = {**{c: c for c in CROPS}, **{a: ANIMALS[a]["product"] for a in ANIMALS}}
K_SET = (1, 5, 10, 25, 50)
MIRROR_K = 10
TABLE_HALF_WIDTH = 2048   # exact price table I0 +- this; outside it prices are computed directly
_PIDX = {p: i for i, p in enumerate(PRODUCTS)}
_RIDX = {p: i for i, p in enumerate(PRODUCERS)}
_SHOP_NAMES = tuple(sorted(SHOPS))   # engine L892 draws rng.choice(sorted(SHOPS))
_SHOP_MULT = {s: {p: (2 if len(v) == 1 else 1) for p in v} for s, v in SHOPS.items()}   # L741
_EXPECTED_RATE = tuple(sum(_SHOP_MULT[s].get(p, 0) for s in _SHOP_NAMES) / len(_SHOP_NAMES) for p in PRODUCTS)
_MAX_RATE = tuple(max(_SHOP_MULT[s].get(p, 0) for s in _SHOP_NAMES) for p in PRODUCTS)
_CENTER = tuple(0 if p == "FERTILIZER" else 1 for p in PRODUCTS)   # engine L113, L745-747

E = "engine "
SPEC = {
    "global": (
        ("steps_remaining", 720, "LAST-s+1 with LAST=episodeSteps-2: joint actions left incl. this one", "K-CALENDAR-719; " + E + "L960"),
        ("days_remaining", 30, "LASTDAY-d+1, LASTDAY=LAST//turnsPerDay", "K-CALENDAR-719"),
        ("day_actions_remaining", 24, "min((d+1)*TPD-1, LAST)-s+1: actions before the next day-end refresh (or the end)", E + "L945, L960"),
        ("refresh_tonight", 1, "(d+1)*TPD-1 <= LAST: a day-end refresh follows today (false on the last day)", "K-CALENDAR-719; " + E + "L945"),
        ("shop_tick_now", 1, "s % townShopSellInterval == 0: visible shops consume after this step's market", E + "L736, L942"),
        ("steps_to_shop_tick", 4, "(-s) mod townShopSellInterval", E + "L736"),
        ("center_tick_now", 1, "s % townCenterSellInterval == 0", E + "L745"),
        ("steps_to_center_tick", 24, "(-s) mod townCenterSellInterval", E + "L745"),
        ("shops_visible", 8, "len(unlocked_shops)", E + "L886-892"),
        ("shop_unlocks_remaining", 8, "#{m: m%U==0, m>d, m*TPD-1<=LAST} capped at 8-shops_visible", E + "L886-892"),
        ("steps_to_next_shop_unlock", 72, "m_next*TPD-s for the first such morning m (0 if none)", E + "L886-892"),
        ("own_cash", 10000, "farms[me].money", E + "L652-688"),
        ("rival_cash", 10000, "farms[rival].money (public)", E + "L652-688"),
        ("cash_diff", 10000, "own_cash-rival_cash: the terminal objective's sign", "README Game rules"),
        ("own_next_hire_cost", 10, "mult*fib(hires_today), fib 1,1,2,3,5..", E + "L690-709"),
        ("rival_next_hire_cost", 10, "mult*fib(rival hires_today)", E + "L690-709"),
        ("own_hires_affordable", 30, "max k: sum_{i<k} mult*fib(h+i) <= cash (cap 60)", E + "L702-709"),
        ("own_wages_today", 100, "sum_{i<h} mult*fib(i)", E + "L702-709"),
        ("rival_wages_today", 100, "sum_{i<h_rival} mult*fib(i)", E + "L702-709"),
        ("hire_useful_actions", 24, "day_actions_remaining-1: a hand hired now acts from the next step until the refresh removes it", E + "L702-709, L880"),
        ("own_hands", 20, "len(farms[me].hands)", E + "L880"),
        ("rival_hands", 20, "len(farms[rival].hands)", E + "L880"),
        ("own_land_count", 4, "len(unlocked_quadrants)", E + "L712-725"),
        ("own_next_land_price", 4000, "LAND_PRICES[len-1] (0 if all owned)", E + "L712-725"),
        ("own_land_affordable", 1, "next land exists and cash >= price", E + "L712-725"),
        ("rival_land_count", 4, "len(rival unlocked_quadrants)", E + "L712-725"),
        ("rival_next_land_price", 4000, "LAND_PRICES[len-1] for the rival (0 if all owned)", E + "L712-725"),
        ("shed_used", 100, "sum(shed.values()) incl. animals", E + "L667, L682"),
        ("shed_room", 100, "shedCapacity-shed_used", E + "L667, L682"),
        ("carried_total", 100, "sum over own worker inventories", E + "L843-857"),
        ("dayend_overflow", 100, "refresh_tonight*max(0, shed_used+carried_total-cap): units the day-end deposit discards if nothing changes", E + "L843-857, L878"),
        ("own_fertilizer_held", 100, "shed+carried FERTILIZER", E + "L475-482"),
        ("own_fertilizer_collectable", 100, "#own animal tiles with fertilizer_available", E + "L515-522, L831"),
        ("rival_fertilizer_collectable", 100, "#rival animal tiles with fertilizer_available", E + "L515-522, L831"),
        ("own_seed_total", 100, "sum(seeds.values())", E + "L416-426"),
        ("own_shed_liquidation", 10000, "sum_p R_p(shed_p): cash from selling every shed product now, empty rival queue", "K-TIMED-RECEIPT-RECURRENCE; " + E + "L592-627, L652-660"),
        ("own_held_value_at_quote", 10000, "sum_p (shed_p+carried_p)*sell_quote_p (a mark, not receipts)", E + "L192-206"),
        ("liquidation_lead", 10000, "own_cash+own_shed_liquidation-rival_cash", "README Game rules"),
        ("own_unserviced_today", 100, "#own plants not watered / animals not fed today", E + "L431-434, L505-513"),
        ("own_dying_tonight", 100, "#own tiles with dies_tonight_if_unserviced", E + "L783, L817"),
        ("own_harvestable_units", 100, "sum own harvestable_now", E + "L446-473"),
        ("own_uncared_animals", 100, "#own animals not cared today", E + "L524-529"),
        ("rival_unserviced_today", 100, "same for the rival farm (public tiles)", E + "L431-434, L505-513"),
        ("rival_dying_tonight", 100, "same for the rival farm", E + "L783, L817"),
        ("rival_harvestable_units", 100, "same for the rival farm", E + "L446-473"),
        ("rival_uncared_animals", 100, "same for the rival farm", E + "L524-529"),
    ),
    "product": (
        ("inventory_offset", 1000, "market inventory - I0", E + "L192-206"),
        ("sell_quote", 100, "price(inv): the quote for the next unit sold", E + "L597"),
        ("buy_quote", 100, "price(inv-1) for WHEAT/FERTILIZER, else 0", E + "L598-601"),
        ("buyable", 1, "product in (WHEAT, FERTILIZER)", E + "L600"),
        *((f"sell_quote_after_{k}", 100, f"price(inv+{k}): quote after selling {k} units alone (floor units add nothing)", E + "L597, L659") for k in K_SET),
        *((f"sell_receipts_{k}", 1000, f"R({k})=sum_(j<{k}) price(inv+j), empty rival queue; exact because price is monotone and floored", "K-TIMED-RECEIPT-RECURRENCE; " + E + "L592-627, L652-660") for k in K_SET),
        *((f"buy_cost_{k}", 1000, f"C({k})=sum_(j=1..{k}) price(inv-j) (buyable only)", E + "L601, L662-671") for k in K_SET),
        ("buy_affordable_units", 100, "max k<=shed_room with C(k)<=cash (buyable only)", E + "L662-671"),
        ("floor_reachable", 1, "some glut inventory F quotes $1", "K-MARKET-DEEP-FLOOR-LOCK-9A7C2E4D"),
        ("floor_distance", 1000, "F-inv, F=min{I>=I0: price(I)=1} (0 if unreachable)", "K-MARKET-DEEP-FLOOR-LOCK-9A7C2E4D"),
        ("floor_locked", 1, "not buyable and inv >= F+demand_rest_visible+demand_rest_max_new: every later quote is $1", "K-MARKET-DEEP-FLOOR-LOCK-9A7C2E4D"),
        ("shed_qty", 100, "own shed stock (SELL reads the shed)", E + "L654-656"),
        ("carried_qty", 100, "own carried stock (needs a deposit before it can be sold)", E + "L343-356"),
        ("shed_receipts", 1000, "R(shed_qty)", E + "L592-627"),
        ("quote_after_shed_sale", 100, "price(inv+shed_qty)", E + "L597, L659"),
        ("mirror_receipts_10", 1000, "sum_(j<10) price(inv+2j): both seats SELL 10 in the same slot (both quoted before either commits)", E + "L585-627"),
        ("visible_rate", 10, "units removed per shop tick by visible shops (2 for single-product shops)", "Q2-DEMAND-LAW-3C0F0F84; " + E + "L736-743"),
        ("demand24_visible", 100, "rate*ticks4([s,s+23]) + center*ticks24([s,s+23]), window clipped at LAST", "Q2-DEMAND-LAW-3C0F0F84; " + E + "L728-749"),
        ("demand_rest_visible", 1000, "same over [s, LAST]", "Q2-DEMAND-LAW-3C0F0F84"),
        ("demand24_expected_new", 100, "E[rate of a uniform shop draw]*ticks after each future unlock in the window (prior, not the realised draw)", E + "L886-892"),
        ("demand_rest_expected_new", 1000, "same over [s, LAST]", E + "L886-892"),
        ("demand_rest_max_new", 1000, "max single-shop rate * ticks after each future unlock: upper bound", E + "L886-892"),
        ("quote_after_demand24", 100, "price(inv-demand24_visible): the quote in 24 steps with no trades and no unlock", "Q2-DEMAND-LAW-3C0F0F84"),
        ("own_field_stock", 100, "sum of yield_units on own tiles producing p (FERTILIZER: collectable count)", E + "L446-473"),
        ("rival_field_stock", 100, "same on the rival farm (public)", E + "L446-473"),
        ("own_output_tonight", 100, "sum own gain_tonight_current", E + "L769-834"),
        ("rival_output_tonight", 100, "sum rival gain_tonight_current", E + "L769-834"),
        ("own_producers", 100, "#own tiles producing p", ""),
        ("rival_producers", 100, "#rival tiles producing p", ""),
        ("own_season_units", 100, "sum own season_units_remaining", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
        ("rival_season_units", 100, "sum rival season_units_remaining", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
    ),
    "tile": (
        ("owned", 1, "tile != LOCKED", E + "L157"),
        ("plantable", 1, "owned empty tile", E + "L416-426"),
        ("weed", 1, "WEED tile (DIG clears it)", E + "L484-491"),
        ("structure_free", 1, "COOP/PASTURE without an animal", E + "L381-392"),
        ("age_days", 30, "d-planted_day or d-placed_day", E + "L438, L822"),
        ("service_needed_today", 1, "plant not watered_today / animal not fed_today", E + "L431-434, L505-513"),
        ("care_needed_today", 1, "animal not cared_today", E + "L524-529"),
        ("dies_tonight_if_unserviced", 1, "refresh_tonight and not serviced and missed+1>=2", E + "L775-785, L813-819"),
        ("day_ends_to_death_unserviced", 3, "3 if serviced today else 2-missed (consecutive_unwatered/unfed)", E + "L775-785, L813-819"),
        ("stored_yield", 6, "yield_units", E + "L446-473"),
        ("yield_room", 6, "cap-yield_units (max_yield / max_held)", E + "L443, L827"),
        ("harvestable_now", 6, "yield if yield>0 and (animal or age>=first_yield_day)", E + "L446-473"),
        ("water_gain_now", 2, "one-shot, not watered, (myd+1)//2<=age<=myd: min(cap,y+b)-y, b=2 if fertilized_until_day>=d else 1", E + "L431-444"),
        ("fert_active_today", 1, "fertilized_until_day >= d", E + "L442, L799"),
        ("fert_days_left", 3, "max(0, fertilized_until_day-d+1)", E + "L481"),
        ("next_event_days", 10, "e-d for the next production morning e<=LASTDAY (0 if none); crops e=pd+first+k*interval, k<4; animals e-pl-first>=0, %interval==0", E + "L789-802, L822-823"),
        ("event_tonight", 1, "next_event_days == 1", E + "L789-802, L822-823"),
        ("gain_tonight_current", 6, "tonight's refresh yield gain with today's flags as they are: plants min(cap,y+(2 if watered&fert else 1))-y; animals min(cap,y+1+(pcb if fed))-y; 0 if it dies", E + "L769-834"),
        ("gain_tonight_serviced", 6, "same assuming WATER/FEED today", E + "L769-834"),
        ("overflow_next_event_serviced", 6, "max(0, y+g-cap): units the cap discards at the next event under daily service if not harvested first; g uncapped (plants 1|2, animals 1+pcb+days of care before it)", E + "L800, L827"),
        ("events_remaining", 30, "#production mornings left <= LASTDAY", E + "L789-802, L822-823"),
        ("pending_care_bonus", 6, "pending_care_bonus", E + "L826-830"),
        ("pending_after_tonight_current", 6, "(0 if event tonight else pcb) + (fed and cared); 0 if it escapes; pcb when no refresh follows", E + "L824-830"),
        ("fertilizer_available_now", 1, "animal fertilizer_available", E + "L515-522"),
        ("fertilizer_available_tomorrow", 1, "animal survives tonight's refresh", E + "L831"),
        ("season_units_remaining", 30, "units harvestable by LASTDAY with daily service, current fertilizer only: one-shot y+bonus waters to cap; ongoing y+sum events(2 if fert on e-1); animals y+sum min(cap,1+pending)", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
        ("unit_distance", 18, "min Manhattan distance from this farm's farmer/hands (locked tiles are traversable)", E + "L321-330"),
        ("decay_scheduled", 1, "next decay step <= LAST (one-shot max_lifespan_step, ongoing (pd+first+3*interval+1)*TPD)", E + "L224, L752-766, L801-802"),
        ("steps_to_next_decay", 48, "t-s, t=mls if s<=mls else s+(s-mls)%2; decay applies after this step's actions, so HARVEST by then keeps every unit", E + "L752-766"),
        ("steps_to_decay_death", 48, "t+2*max(0,y-1)-s when mls is set: the step the tile turns to WEED with no further yield change", E + "L752-766"),
        ("season_value_at_quote", 1000, "season_units_remaining*sell_quote(product) (a mark)", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
    ),
    "structure": (
        ("own_count", 100, "#own tiles holding this producer", ""),
        ("rival_count", 100, "#rival tiles holding this producer", ""),
        ("own_start_stock", 100, "seeds (crop) or shed+carried animals", E + "L358-392"),
        ("start_cost", 500, "seed or animal price", E + "L11-24"),
        ("buyable_now", 100, "units a BUY order fills now: cash//cost (animals also <= shed_room)", E + "L673-688"),
        ("first_output_days", 12, "one-shot watered harvest age / first_yield_day", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
        ("season_units_watered", 100, "unit_economics.formula(producer, watered|fed_cared, days=LASTDAY-d+1)", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
        ("season_units_fertilized", 100, "unit_economics.formula(crop, watered_fertilized, ...) (animals: as watered)", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
        ("season_cycles", 30, "formula cycles (crops) / events (animals)", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
        ("season_value_at_quote", 10000, "season_units_watered*sell_quote(product)", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
        ("season_input_cost", 1000, "crops seed*cycles; animals cost+days*wheat buy_quote", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
        ("season_net_at_quote", 10000, "season_value_at_quote-season_input_cost", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
        ("wheat_needed", 30, "animals: days (1 WHEAT/day); crops 0", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
        ("fertilizer_output", 30, "animals: days-1 surviving refreshes; crops 0", E + "L831"),
        ("own_stored_total", 100, "sum yield_units on own tiles of this producer", ""),
        ("rival_stored_total", 100, "sum yield_units on rival tiles of this producer", ""),
        ("own_season_units", 100, "sum own tile season_units_remaining", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
        ("rival_season_units", 100, "sum rival tile season_units_remaining", "K-UNIT-ECONOMICS-ONE-TILE-v2"),
    ),
}
GLOBAL_NAMES = tuple(r[0] for r in SPEC["global"])
PRODUCT_NAMES = tuple(r[0] for r in SPEC["product"])
TILE_NAMES = tuple(r[0] for r in SPEC["tile"])
STRUCTURE_NAMES = tuple(r[0] for r in SPEC["structure"])
BLOCK_SPEC = {"global": "global", "product": "product", "tile_own": "tile", "tile_rival": "tile",
              "structure": "structure"}
SCALES = {block: np.asarray([r[1] for r in SPEC[kind]], np.float32) for block, kind in BLOCK_SPEC.items()}
NAMES = {block: tuple(r[0] for r in SPEC[kind]) for block, kind in BLOCK_SPEC.items()}
ROWS = {"product": tuple(PRODUCTS), "structure": PRODUCERS,
        "tile_own": tuple(f"{x},{y}" for y in range(10) for x in range(10)),
        "tile_rival": tuple(f"{x},{y}" for y in range(10) for x in range(10))}
_G = {n: i for i, n in enumerate(GLOBAL_NAMES)}
_P = {n: i for i, n in enumerate(PRODUCT_NAMES)}
_S = {n: i for i, n in enumerate(STRUCTURE_NAMES)}
_T = {n: i for i, n in enumerate(TILE_NAMES)}
_T_STATIC = _T["unit_distance"]   # columns [0, _T_STATIC) come from the per-(tile, day) cache


# ----------------------------------------------------------------------------- market math
def _shape(func, x, T=None):   # engine L61-74, byte-for-byte arithmetic
    x = max(0.0, x)
    if func == "linear": return x
    if func == "sq": return x * x
    if func == "sqrt": return math.sqrt(x)
    if func == "log": return math.log(1.0 + x)
    if func == "log10": return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(p, inventory):
    """Engine market_price (L192-206) for one product's parameter dict."""
    base, I0, T = p["base"], p["I0"], p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T, T)
        price = base + amp * _shape(f, I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T, T)
        price = base - amp * _shape(f, inventory - I0, T)
    return max(PRICE_FLOOR, int(round(price)))


def floor_inventory(p, cap=10_000_000):
    """Least glut inventory quoting $1 (price is non-increasing above I0), or None within cap."""
    I0 = p["I0"]
    if market_price(p, I0 + cap) > PRICE_FLOOR:
        return None
    lo, hi = I0, I0 + cap          # price(lo) may be > 1; price(hi) == 1
    if market_price(p, lo) == PRICE_FLOOR:
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if market_price(p, mid) == PRICE_FLOOR:
            hi = mid
        else:
            lo = mid
    return hi


class _Curve:
    """Exact price table and prefix sums around I0; direct computation outside."""
    __slots__ = ("p", "lo", "hi", "prices", "prefix", "floor")

    def __init__(self, p):
        self.p = p
        self.lo, self.hi = p["I0"] - TABLE_HALF_WIDTH, p["I0"] + TABLE_HALF_WIDTH
        self.prices = [market_price(p, i) for i in range(self.lo, self.hi + 1)]
        acc, prefix = 0, [0]
        for v in self.prices:
            acc += v
            prefix.append(acc)
        self.prefix = prefix
        self.floor = floor_inventory(p)

    def quote(self, inventory):
        if self.lo <= inventory <= self.hi:
            return self.prices[inventory - self.lo]
        return market_price(self.p, inventory)

    def stride_total(self, a, count, stride):
        """Sum of quotes at inventories a, a+stride, ... (count terms)."""
        end = a + stride * (count - 1)
        if self.lo <= a and end <= self.hi:
            return sum(self.prices[a - self.lo:end - self.lo + 1:stride])
        return sum(self.quote(a + stride * j) for j in range(count))

    def total(self, a, b):
        """Sum of quotes at inventories a..b-1."""
        if b <= a:
            return 0
        if self.lo <= a and b - 1 <= self.hi:
            return self.prefix[b - self.lo] - self.prefix[a - self.lo]
        return sum(self.quote(i) for i in range(a, b))


_CURVES: dict = {}


_DEFAULT_KEY = None


def _curves(params):
    if params is MARKET_PARAMS and _DEFAULT_KEY is not None:
        return _CURVES[_DEFAULT_KEY]
    key = tuple((item, tuple(sorted(params[item].items()))) for item in PRODUCTS)
    curves = _CURVES.get(key)
    if curves is None:
        if len(_CURVES) > 8:
            _CURVES.clear()
            globals()["_DEFAULT_KEY"] = None
        curves = _CURVES[key] = tuple(_Curve(params[item]) for item in PRODUCTS)
    if params is MARKET_PARAMS:
        globals()["_DEFAULT_KEY"] = key
    return curves


def warm(config=None):
    """Build the default price tables (about 40k exact quotes) before the first timed call."""
    _curves(_market_params({}, config))


# ----------------------------------------------------------------------------- configuration
def _setting(config, key, default):
    if config is None:
        return default
    try:
        value = config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)
    except Exception:
        value = default
    return default if value is None else value


def _market_params(obs, config):
    market = obs.get("market") or {}
    params = market.get("params")
    if params:
        return {item: {**MARKET_PARAMS[item], **(params.get(item) or {})} for item in PRODUCTS}
    overrides = _setting(config, "marketParams", None) or {}
    if not overrides:
        return MARKET_PARAMS
    return {item: {**MARKET_PARAMS[item], **(overrides.get(item) or {})} for item in PRODUCTS}


def _fib(n):   # engine L690-695
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


_FIB = tuple(_fib(n) for n in range(160))
_FIB_PREFIX = tuple(sum(_FIB[:n]) for n in range(161))


def _ticks(a, b, n):
    """#t in [a, b] with t % n == 0."""
    return b // n - (a - 1) // n if b >= a else 0


# ----------------------------------------------------------------------------- unit economics
def _harvest_age(c, per_water):   # unit_economics._one_shot_harvest_age
    start = (c["max_yield_day"] + 1) // 2
    for age in range(c["first_yield_day"], c["max_yield_day"] + 1):
        if 1 + per_water * max(0, age - start + 1) >= c["max_yield"]:
            return age
    return c["max_yield_day"]


def season_formula(producer, fertilized, days):
    """Idealized full cycles plus a harvestable terminal partial one-shot crop."""
    if producer in CROPS:
        c = CROPS[producer]
        pw = 2 if fertilized else 1
        if not c["ongoing"]:
            start = (c["max_yield_day"] + 1) // 2
            cycle = _harvest_age(c, pw)
            upc = min(c["max_yield"], 1 + pw * max(0, cycle - start + 1))
            cycles = max(0, (days - 1) // cycle)
            tail_age = days - 1 - cycles * cycle
            tail = (min(c["max_yield"], 1 + pw * max(0, tail_age - start + 1))
                    if tail_age >= c["first_yield_day"] else 0)
            return upc * cycles + tail, cycles + int(tail > 0), (cycle if cycles or not tail else tail_age)
        events, interval, first = c["max_yield"], c["interval"], c["first_yield_day"]
        cycle = first + (events - 1) * interval + 1
        total = p = cycles = 0
        while p + first <= days - 1:
            total += pw * sum(1 for i in range(events) if p + first + i * interval <= days - 1)
            cycles += 1
            p += cycle
        return total, cycles, first
    a = ANIMALS[producer]
    first, interval, cap = a["first_yield_day"], a["interval"], a["max_held"]
    event_days = [e for e in range(first, days) if (e - first) % interval == 0]
    units = sum(min(cap, 1 + (first - 1 if i == 0 else interval)) for i in range(len(event_days)))
    return units, len(event_days), first


_SEASON = {(r, f, n): season_formula(r, f, n) for r in PRODUCERS for f in (False, True) for n in range(0, 61)}


# ----------------------------------------------------------------------------- tiles
def _const_row(**values):
    row = [0.0] * _T_STATIC
    for k, v in values.items():
        row[_T[k]] = float(v)
    return tuple(row)


class _RowBank:
    """Distinct static tile rows stored once; a farm block is one fancy-index gather."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mat = np.zeros((512, _T_STATIC), np.float32)
        self.index = {}
        self.n = 0

    def id(self, row):
        i = self.index.get(row)
        if i is None:
            if self.n == len(self.mat):
                self.mat = np.concatenate([self.mat, np.zeros_like(self.mat)])
            self.mat[self.n] = row
            i = self.index[row] = self.n
            self.n += 1
        return i


_BANK = _RowBank()
_TILE_CACHE: dict = {}
_ROW_EMPTY = _ROW_LOCKED = _ROW_WEED = _ROW_STRUCT = 0
_NO_AUX = None


def _reset_caches():
    global _ROW_EMPTY, _ROW_LOCKED, _ROW_WEED, _ROW_STRUCT
    _TILE_CACHE.clear()
    _BANK.reset()
    _ROW_EMPTY = _BANK.id(_const_row(owned=1, plantable=1))
    _ROW_LOCKED = _BANK.id(_const_row())
    _ROW_WEED = _BANK.id(_const_row(owned=1, weed=1))
    _ROW_STRUCT = _BANK.id(_const_row(owned=1, structure_free=1))


_reset_caches()


def _plant_static(t, d, lastday, refresh, tpd):
    """Columns [0, _T_STATIC) + aux for a PLANT tile at day d."""
    crop = t["crop"]
    c = CROPS[crop]
    first, myd, cap, interval = c["first_yield_day"], c["max_yield_day"], c["max_yield"], c["interval"]
    pd = t["planted_day"]
    w = bool(t["watered_today"])
    cu = t["consecutive_unwatered"]
    y = t["yield_units"]
    fud = t.get("fertilized_until_day", -1)
    age = d - pd
    dies = refresh and not w and cu + 1 >= 2
    to_death = 3 if w else 2 - cu
    harvestable = y if (y > 0 and age >= first) else 0
    water_gain = 0
    fert_today = fud >= d
    if not c["ongoing"]:
        ws = (myd + 1) // 2
        if not w and ws <= age <= myd:
            water_gain = min(cap, y + (2 if fert_today else 1)) - y
        # season units: bonus waterings to the cap, harvest by min(pd+myd, lastday)
        if pd + first > lastday:
            season = 0
        elif age > myd:
            season = y
        else:
            season = y
            for day in range(max(d + (1 if w else 0), pd + ws), min(pd + myd, lastday) + 1):
                season = min(cap, season + (2 if fud >= day else 1))
        nxt = gain_cur = gain_srv = overflow = remaining = 0
        mls_eff = t["max_lifespan_step"]
    else:
        nxt = 0
        remaining = 0
        season = y
        b_next = 1
        for k in range(cap):
            e = pd + first + k * interval
            if d + 1 <= e <= lastday:
                b = 2 if fud >= e - 1 else 1
                if nxt == 0:
                    nxt = e - d
                    b_next = b
                remaining += 1
                season += b
        event_tonight = nxt == 1
        if event_tonight and not dies:
            gain_cur = min(cap, y + (2 if (w and fert_today) else 1)) - y
        else:
            gain_cur = 0
        gain_srv = (min(cap, y + (2 if fert_today else 1)) - y) if event_tonight else 0
        overflow = max(0, y + b_next - cap) if nxt else 0
        mls_eff = t["max_lifespan_step"]
        if mls_eff < 0:
            e3 = pd + first + (cap - 1) * interval
            mls_eff = (e3 + 1) * tpd if e3 <= lastday else -1
    row = (1.0, 0.0, 0.0, 0.0, float(age), float(not w), 0.0, float(dies), float(to_death), float(y),
           float(cap - y), float(harvestable), float(water_gain), float(fert_today),
           float(max(0, fud - d + 1)), float(nxt), float(nxt == 1), float(gain_cur), float(gain_srv),
           float(overflow), float(remaining), 0.0, 0.0, 0.0, 0.0, float(season))
    aux = (_PIDX[crop], _RIDX[crop], y, harvestable, gain_cur, season, not w, dies, False, False,
           mls_eff, t["max_lifespan_step"] >= 0)
    return row, aux


def _animal_static(t, d, lastday, refresh, tpd):
    animal = t["animal"]
    a = ANIMALS[animal]
    first, interval, cap = a["first_yield_day"], a["interval"], a["max_held"]
    pl = t["placed_day"]
    fed = bool(t["fed_today"])
    cared = bool(t["cared_today"])
    cu = t["consecutive_unfed"]
    y = t["yield_units"]
    pcb = t.get("pending_care_bonus", 0)
    fa = bool(t.get("fertilizer_available", False))
    age = d - pl
    dies = refresh and not fed and cu + 1 >= 2
    to_death = 3 if fed else 2 - cu
    e0 = pl + first
    e = e0 if d + 1 <= e0 else e0 + -(-(d + 1 - e0) // interval) * interval
    nxt = e - d if e <= lastday else 0
    event_tonight = nxt == 1
    if event_tonight and not dies:
        gain_cur = min(cap, y + 1 + (pcb if fed else 0)) - y
    else:
        gain_cur = 0
    gain_srv = (min(cap, y + 1 + pcb) - y) if event_tonight else 0
    if not refresh:
        pending_after = pcb
    elif dies:
        pending_after = 0
    else:
        pending_after = (0 if event_tonight else pcb) + (1 if (fed and cared) else 0)
    overflow = max(0, y + 1 + pcb + (nxt - 1) - cap) if nxt else 0
    # season under daily feed+care and daily harvest (held is 0 at every refresh)
    season, p, remaining = y, pcb, 0
    for night in range(d, lastday):
        dsf = night + 1 - pl - first
        if dsf >= 0 and dsf % interval == 0:
            season += min(cap, 1 + p)
            p = 0
            remaining += 1
        p += 1
    product = a["product"]
    row = (1.0, 0.0, 0.0, 0.0, float(age), float(not fed), float(not cared), float(dies),
           float(to_death), float(y), float(cap - y), float(y), 0.0, 0.0, 0.0, float(nxt),
           float(event_tonight), float(gain_cur), float(gain_srv), float(overflow), float(remaining),
           float(pcb), float(pending_after), float(fa), float(not dies and refresh), float(season))
    aux = (_PIDX[product], _RIDX[animal], y, y, gain_cur, season, not fed, dies, not cared, fa, -1, False)
    return row, aux


def _tile_static(t, d, lastday, refresh, tpd):
    kind = t.get("kind")
    if kind == "PLANT":
        key = ("P", t["crop"], t["planted_day"], t["watered_today"], t["consecutive_unwatered"],
               t["yield_units"], t["max_lifespan_step"], t.get("fertilized_until_day", -1), d, lastday, tpd, refresh)
        hit = _TILE_CACHE.get(key)
        if hit is None:
            row, aux = _plant_static(t, d, lastday, refresh, tpd)
            hit = _TILE_CACHE[key] = (_BANK.id(row), aux)
        return hit
    if "animal" in t:
        key = ("A", t["animal"], t["placed_day"], t["fed_today"], t["cared_today"], t["consecutive_unfed"],
               t["yield_units"], t.get("pending_care_bonus", 0), t.get("fertilizer_available", False), d, lastday, refresh)
        hit = _TILE_CACHE.get(key)
        if hit is None:
            row, aux = _animal_static(t, d, lastday, refresh, tpd)
            hit = _TILE_CACHE[key] = (_BANK.id(row), aux)
        return hit
    if kind == "WEED":
        return _ROW_WEED, _NO_AUX
    return _ROW_STRUCT, _NO_AUX


_XS = np.tile(np.arange(10), 10)
_YS = np.repeat(np.arange(10), 10)


def _farm_block(farm, d, s, lastday, last, refresh, tpd, quotes, prod_acc, struct_acc, glob_acc):
    """Returns the (100, T) array; accumulates per-product/per-producer/global aggregates."""
    if len(_TILE_CACHE) > 20000 or _BANK.n > 20000:
        _reset_caches()
    rows = []
    decay = []   # (index, mls_eff, y, mls_set)
    values = []  # (index, season, product_index)
    append = rows.append
    i = 0
    for line in farm["tiles"]:
        for t in line:
            if t is None:
                append(_ROW_EMPTY)
            elif t == "LOCKED":
                append(_ROW_LOCKED)
            else:
                row, aux = _tile_static(t, d, lastday, refresh, tpd)
                append(row)
                if aux is not None:
                    pi, ri, y, harvestable, gain_cur, season, unserviced, dies, uncared, fa, mls_eff, mls_set = aux
                    prod_acc[pi][0] += y
                    prod_acc[pi][1] += gain_cur
                    prod_acc[pi][2] += 1
                    prod_acc[pi][3] += season
                    struct_acc[ri][0] += 1
                    struct_acc[ri][1] += y
                    struct_acc[ri][2] += season
                    glob_acc[0] += unserviced
                    glob_acc[1] += dies
                    glob_acc[2] += harvestable
                    glob_acc[3] += uncared
                    glob_acc[4] += fa
                    if mls_eff >= 0:
                        decay.append((i, mls_eff, y, mls_set))
                    if season:
                        values.append((i, season, pi))
            i += 1
    block = np.zeros((100, len(TILE_NAMES)), np.float32)
    block[:, :_T_STATIC] = _BANK.mat[np.array(rows, np.intp)]
    units = [farm["farmer"], *farm.get("hands", [])]
    if len(units) == 1:
        ux, uy = units[0]
        block[:, _T["unit_distance"]] = np.abs(_XS - ux) + np.abs(_YS - uy)
    else:
        pos = np.asarray(units)
        block[:, _T["unit_distance"]] = (np.abs(_XS[:, None] - pos[None, :, 0])
                                         + np.abs(_YS[:, None] - pos[None, :, 1])).min(axis=1)
    c_sched, c_next, c_death, c_value = (_T["decay_scheduled"], _T["steps_to_next_decay"],
                                         _T["steps_to_decay_death"], _T["season_value_at_quote"])
    for idx, mls, y, mls_set in decay:
        nxt = mls if s <= mls else s + (s - mls) % 2
        if nxt <= last:
            block[idx, c_sched] = 1.0
            block[idx, c_next] = nxt - s
            if mls_set:
                block[idx, c_death] = nxt + 2 * max(0, y - 1) - s
    for idx, season, pi in values:
        block[idx, c_value] = season * quotes[pi]
    return block



# ----------------------------------------------------------------------------- main entry
def compute(actor_obs, config=None):
    """Theorem features for the observing seat. Pure function of the actor observation."""
    tpd = int(_setting(config, "turnsPerDay", 24))
    episode = int(_setting(config, "episodeSteps", 720))
    cap = int(_setting(config, "shedCapacity", 100))
    mult = int(_setting(config, "farmHandCostMult", 1))
    unlock_every = max(1, int(_setting(config, "townShopUnlockInterval", 3)))
    shop_every = max(1, int(_setting(config, "townShopSellInterval", 4)))
    center_every = max(1, int(_setting(config, "townCenterSellInterval", 24)))
    last = episode - 2
    lastday = last // tpd
    step = actor_obs.get("step")
    s = int(step) if step is not None else int(actor_obs["day"]) * tpd + int(actor_obs["hour"])
    d = s // tpd
    day_end = min((d + 1) * tpd - 1, last)
    refresh = (d + 1) * tpd - 1 <= last
    me = int(actor_obs["player"])
    farms = actor_obs["farms"]
    own, rival = farms[me], farms[1 - me]
    private = actor_obs.get("private") or {}
    shed = private.get("shed") or {}
    seeds = private.get("seeds") or {}
    carried = {}
    for inv in private.get("inventories") or ():
        for item, n in inv.items():
            carried[item] = carried.get(item, 0) + n
    market = actor_obs["market"]
    inventory = market["inventory"]
    curves = _curves(_market_params(actor_obs, config))
    inv = tuple(int(inventory[p]) for p in PRODUCTS)
    quotes = tuple(curves[i].quote(inv[i]) for i in range(9))
    own_cash = float(own["money"])
    rival_cash = float(rival["money"])
    shed_used = sum(shed.values())
    shed_room = max(0, cap - shed_used)
    carried_total = sum(carried.values())

    # ---- tiles (and aggregates)
    prod_own = [[0, 0, 0, 0] for _ in range(9)]
    prod_riv = [[0, 0, 0, 0] for _ in range(9)]
    str_own = [[0, 0, 0] for _ in range(8)]
    str_riv = [[0, 0, 0] for _ in range(8)]
    g_own = [0, 0, 0, 0, 0]
    g_riv = [0, 0, 0, 0, 0]
    tile_own = _farm_block(own, d, s, lastday, last, refresh, tpd, quotes, prod_own, str_own, g_own)
    tile_rival = _farm_block(rival, d, s, lastday, last, refresh, tpd, quotes, prod_riv, str_riv, g_riv)

    # ---- town demand (Q2-DEMAND-LAW)
    shops = (actor_obs.get("town") or {}).get("unlocked_shops") or []
    rate = [0] * 9
    for shop in shops:
        for p, m in _SHOP_MULT[shop].items():
            rate[_PIDX[p]] += m
    w_end = min(s + 23, last)
    shop24, shop_rest = _ticks(s, w_end, shop_every), _ticks(s, last, shop_every)
    cen24, cen_rest = _ticks(s, w_end, center_every), _ticks(s, last, center_every)
    slots = max(0, MAX_SHOP_INSTANCES - len(shops))
    mornings = []
    m = (d // unlock_every + 1) * unlock_every
    while slots > 0 and m * tpd - 1 <= last:
        mornings.append(m)
        slots -= 1
        m += unlock_every
    new24 = sum(_ticks(max(s, m * tpd), w_end, shop_every) for m in mornings)
    new_rest = sum(_ticks(max(s, m * tpd), last, shop_every) for m in mornings)

    # ---- products
    prows = []
    width = len(PRODUCT_NAMES)
    liquidation = 0
    held_value = 0
    P = _P
    for i, p in enumerate(PRODUCTS):
        cv, I = curves[i], inv[i]
        row = [0.0] * width
        prows.append(row)
        q = quotes[i]
        row[P["inventory_offset"]] = I - cv.p["I0"]
        row[P["sell_quote"]] = q
        buyable = p in BUYABLE
        row[P["buyable"]] = buyable
        for j, k in enumerate(K_SET):
            row[P["sell_quote_after_1"] + j] = cv.quote(I + k)
            row[P["sell_receipts_1"] + j] = cv.total(I, I + k)
        if buyable:
            row[P["buy_quote"]] = cv.quote(I - 1)
            for j, k in enumerate(K_SET):
                row[P["buy_cost_1"] + j] = cv.total(I - k, I)
            lo_k, hi_k = 0, shed_room
            while lo_k < hi_k:
                mid = (lo_k + hi_k + 1) // 2
                if cv.total(I - mid, I) <= own_cash:
                    lo_k = mid
                else:
                    hi_k = mid - 1
            row[P["buy_affordable_units"]] = lo_k
        d24 = rate[i] * shop24 + _CENTER[i] * cen24
        drest = rate[i] * shop_rest + _CENTER[i] * cen_rest
        max_new = _MAX_RATE[i] * new_rest
        if cv.floor is not None:
            row[P["floor_reachable"]] = 1
            row[P["floor_distance"]] = cv.floor - I
            row[P["floor_locked"]] = (not buyable) and I >= cv.floor + drest + max_new
        sq = shed.get(p, 0)
        row[P["shed_qty"]] = sq
        row[P["carried_qty"]] = carried.get(p, 0)
        r_shed = cv.total(I, I + sq)
        liquidation += r_shed
        held_value += (sq + carried.get(p, 0)) * q
        row[P["shed_receipts"]] = r_shed
        row[P["quote_after_shed_sale"]] = cv.quote(I + sq)
        row[P["mirror_receipts_10"]] = cv.stride_total(I, MIRROR_K, 2)
        row[P["visible_rate"]] = rate[i]
        row[P["demand24_visible"]] = d24
        row[P["demand_rest_visible"]] = drest
        row[P["demand24_expected_new"]] = _EXPECTED_RATE[i] * new24
        row[P["demand_rest_expected_new"]] = _EXPECTED_RATE[i] * new_rest
        row[P["demand_rest_max_new"]] = max_new
        row[P["quote_after_demand24"]] = cv.quote(I - d24)
        row[P["own_field_stock"]] = prod_own[i][0]
        row[P["rival_field_stock"]] = prod_riv[i][0]
        row[P["own_output_tonight"]] = prod_own[i][1]
        row[P["rival_output_tonight"]] = prod_riv[i][1]
        row[P["own_producers"]] = prod_own[i][2]
        row[P["rival_producers"]] = prod_riv[i][2]
        row[P["own_season_units"]] = prod_own[i][3]
        row[P["rival_season_units"]] = prod_riv[i][3]
    fi = _PIDX["FERTILIZER"]
    prows[fi][P["own_field_stock"]] = g_own[4]
    prows[fi][P["rival_field_stock"]] = g_riv[4]
    product = np.array(prows, np.float32)

    # ---- producers ("structure")
    srows = []
    width = len(STRUCTURE_NAMES)
    days = lastday - d + 1
    wheat_buy = curves[_PIDX["WHEAT"]].quote(inv[_PIDX["WHEAT"]] - 1)
    S = _S
    for r, name in enumerate(PRODUCERS):
        row = [0.0] * width
        srows.append(row)
        pi = _PIDX[PRODUCT_OF[name]]
        is_crop = name in CROPS
        cost = CROPS[name]["seed"] if is_crop else ANIMALS[name]["cost"]
        units, cycles, first_out = _SEASON[(name, False, days)] if days <= 60 else season_formula(name, False, days)
        units_f = (_SEASON[(name, True, days)] if days <= 60 else season_formula(name, True, days))[0] if is_crop else units
        row[S["own_count"]] = str_own[r][0]
        row[S["rival_count"]] = str_riv[r][0]
        row[S["own_start_stock"]] = seeds.get(name, 0) if is_crop else shed.get(name, 0) + carried.get(name, 0)
        row[S["start_cost"]] = cost
        affordable = int(own_cash // cost)
        row[S["buyable_now"]] = affordable if is_crop else min(affordable, shed_room)
        row[S["first_output_days"]] = first_out
        row[S["season_units_watered"]] = units
        row[S["season_units_fertilized"]] = units_f
        row[S["season_cycles"]] = cycles
        value = units * quotes[pi]
        input_cost = cost * cycles if is_crop else cost + days * wheat_buy
        row[S["season_value_at_quote"]] = value
        row[S["season_input_cost"]] = input_cost
        row[S["season_net_at_quote"]] = value - input_cost
        if not is_crop:
            row[S["wheat_needed"]] = days
            row[S["fertilizer_output"]] = max(0, days - 1)
        row[S["own_stored_total"]] = str_own[r][1]
        row[S["rival_stored_total"]] = str_riv[r][1]
        row[S["own_season_units"]] = str_own[r][2]
        row[S["rival_season_units"]] = str_riv[r][2]
    structure = np.array(srows, np.float32)

    # ---- global
    g = [0.0] * len(GLOBAL_NAMES)
    G = _G
    h_own, h_riv = int(own.get("hires_today", 0)), int(rival.get("hires_today", 0))
    g[G["steps_remaining"]] = max(0, last - s + 1)
    g[G["days_remaining"]] = lastday - d + 1
    g[G["day_actions_remaining"]] = max(0, day_end - s + 1)
    g[G["refresh_tonight"]] = refresh
    g[G["shop_tick_now"]] = s % shop_every == 0
    g[G["steps_to_shop_tick"]] = (-s) % shop_every
    g[G["center_tick_now"]] = s % center_every == 0
    g[G["steps_to_center_tick"]] = (-s) % center_every
    g[G["shops_visible"]] = len(shops)
    g[G["shop_unlocks_remaining"]] = len(mornings)
    g[G["steps_to_next_shop_unlock"]] = mornings[0] * tpd - s if mornings else 0
    g[G["own_cash"]] = own_cash
    g[G["rival_cash"]] = rival_cash
    g[G["cash_diff"]] = own_cash - rival_cash
    g[G["own_next_hire_cost"]] = mult * _FIB[h_own]
    g[G["rival_next_hire_cost"]] = mult * _FIB[h_riv]
    k, spent = 0, 0
    while k < 60 and spent + mult * _FIB[h_own + k] <= own_cash:
        spent += mult * _FIB[h_own + k]
        k += 1
    g[G["own_hires_affordable"]] = k
    g[G["own_wages_today"]] = mult * _FIB_PREFIX[h_own]
    g[G["rival_wages_today"]] = mult * _FIB_PREFIX[h_riv]
    g[G["hire_useful_actions"]] = max(0, day_end - s)
    g[G["own_hands"]] = len(own.get("hands", ()))
    g[G["rival_hands"]] = len(rival.get("hands", ()))
    for who, farm, cash in (("own", own, own_cash), ("rival", rival, rival_cash)):
        n_land = len(farm.get("unlocked_quadrants", ("NW",)))
        price = LAND_PRICES[n_land - 1] if n_land - 1 < len(LAND_PRICES) else 0
        g[G[who + "_land_count"]] = n_land
        g[G[who + "_next_land_price"]] = price
        if who == "own":
            g[G["own_land_affordable"]] = price > 0 and cash >= price
    g[G["shed_used"]] = shed_used
    g[G["shed_room"]] = shed_room
    g[G["carried_total"]] = carried_total
    g[G["dayend_overflow"]] = max(0, shed_used + carried_total - cap) if refresh else 0
    g[G["own_fertilizer_held"]] = shed.get("FERTILIZER", 0) + carried.get("FERTILIZER", 0)
    g[G["own_fertilizer_collectable"]] = g_own[4]
    g[G["rival_fertilizer_collectable"]] = g_riv[4]
    g[G["own_seed_total"]] = sum(seeds.values())
    g[G["own_shed_liquidation"]] = liquidation
    g[G["own_held_value_at_quote"]] = held_value
    g[G["liquidation_lead"]] = own_cash + liquidation - rival_cash
    for who, acc in (("own", g_own), ("rival", g_riv)):
        g[G[who + "_unserviced_today"]] = acc[0]
        g[G[who + "_dying_tonight"]] = acc[1]
        g[G[who + "_harvestable_units"]] = acc[2]
        g[G[who + "_uncared_animals"]] = acc[3]
    return {"global": np.array(g, np.float32), "product": product, "tile_own": tile_own, "tile_rival": tile_rival,
            "structure": structure}


def normalized(features):
    """Divide each block by SCALES (no clipping); returns new arrays."""
    return {k: (v / SCALES[k]).astype(np.float32) for k, v in features.items()}


def describe():
    """Name, scale, formula and source for every column, grouped by block."""
    return {kind: [dict(name=n, scale=sc, formula=f, source=src) for n, sc, f, src in rows]
            for kind, rows in SPEC.items()}
