"""Exact-numbers economics layer for the heuristic opponent.

Every strategic and tactical decision in `policy.py` is a dollar value computed here from
exact sources: the engine constants (`constants.py`, asserted against the shipped engine
by tests/), the market price-curve replica (`market.py`), and a `Forecast` of where each
product's market inventory is heading (estimated online from observed net drain -- town
demand plus opponent behaviour -- so prices are evaluated at the production day, not at
today's spot). No hand-set tiers, no magic quotas:

  * what to plant        -> argmax over crops of `crop_tile_value` (NPV of an empty tile)
  * whether to buy a goose -> `goose_npv` vs the crop NPV of the coop tile it occupies
  * whether to fertilize -> `fertilize_value` (bonus units x price vs the fertilizer sale)
  * which job a unit takes -> highest value density: value / (1 + walk distance)

The one calibrated constant is `ACTION_COST`, the shadow price of labour: the farm has
more valuable jobs than unit-actions, so every action is priced at the value of the job it
displaces. It is a tunable because it encodes scarcity, which the obs does not state.

Values are ranking signals, not oracles: closed-form, day-granularity, evaluated per turn
in microseconds (the 1s actTimeout is never at risk).
"""

from .constants import (
    ANIMALS,
    CROPS,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    hire_cost,
    one_time_cycle_harvest_age,
    one_time_units_without_fertilizer,
    water_bonus_window,
)
from .market import price_at, sell_revenue

#: Shadow price of one unit-action, in dollars. Calibrated against arena sweeps: the
#: season earns ~$45k over ~6.5k unit-actions. Swept indirectly through goose/fertilize
#: behaviour -- at $7 the goose flock sizes itself to ~6 and fertilizing stays selective.
ACTION_COST = 7.0

#: Estimated unit-actions per tile per one-time crop cycle beyond watering:
#: PLANT + HARVEST + one haul trip + ~4 turns of walking.
CYCLE_OVERHEAD_ACTIONS = 7.0

#: Daily unit-actions per animal: FEED + CARE + COLLECT + ~1 HARVEST + haul + walking.
ANIMAL_ACTIONS_PER_DAY = 6.5


class Forecast:
    """Market state plus estimated net inventory drain rates (signed units/day).

    rates[item] < 0 means the town + opponent are draining the product (price drifting
    up on scarcity); > 0 means supply is being dumped (price drifting down). Evaluated
    exactly through the price-curve replica: `price(item, days)` is what a unit fetches
    `days` from now if the observed net rate continues.

    supply[item] is OUR OWN unsold pipeline (shed stock, carried goods, expected harvests
    of standing crops). It phases in over ~10 days and is what stops the calculator from
    pricing every new tile as if it were the only seller: the tenth melon tile faces the
    glut the first nine will create.
    """

    def __init__(self, market, rates=None, supply=None):
        self.market = market or {}
        self.rates = rates or {}
        self.supply = supply or {}

    def inv(self, item, days=0.0):
        raw = self.market.get("inventory", {}).get(item, 10000)
        own = self._own_supply(item, days)
        # The engine permits market inventory to go negative when scarce stock is bought.
        # Preserve that legal state: clipping it aliases distinct high-scarcity prices.
        return raw + self.rates.get(item, 0.0) * days + own

    def price(self, item, days=0.0):
        return price_at(item, self.inv(item, days))

    def _own_supply(self, item, days):
        """Our own pipeline that has reached the market by `days` from now.

        A scalar entry is phased linearly over ten days, which is the original crude model. A
        DATED entry -- `{day_offset: units}` -- is summed exactly through the horizon, which is
        what a supply schedule actually is: the tenth melon does not glut the market on day 0,
        it gluts it on the day it is harvested. Review 2026-08-31 found the allocator injecting
        a whole season's output into day-0 inventory, which suppresses exactly the products
        whose curves collapse fastest (MILK, WOOL) and understates cows and sheep.
        """
        entry = self.supply.get(item)
        if entry is None:
            return 0.0
        if isinstance(entry, dict):
            return float(sum(n for offset, n in entry.items() if offset <= days))
        return float(entry) * min(1.0, days / 10.0)

    def avg_sell_price(self, item, units, days=0.0):
        """Average per-unit revenue for dumping `units`, including own price impact."""
        if units <= 0:
            return 0.0
        return sell_revenue(item, self.inv(item, days), units) / units

    def buy_price(self, item):
        """What the next unit bought costs (quoted at post-buy inventory)."""
        return price_at(item, self.inv(item) - 1)


# --- crops --------------------------------------------------------------------

def _one_time_tile_value(cd, crop, day, total_days, fc):
    """NPV of repeated one-time cycles on one tile, planted today."""
    # max_yield_day is the *end of the bonus window*, not always the useful cycle time.
    # Ordinary melon watering reaches the six-unit cap at age 10, two days earlier.
    grow = one_time_cycle_harvest_age(crop)
    cycle = grow + 2                    # + harvest/replant slack
    window = water_bonus_window(crop)
    window_days = window[1] - window[0] + 1 if window else 0
    units_per_cycle = one_time_units_without_fertilizer(crop)
    pre_window_survival_waters = len(range(0, window[0], 2)) if window else 0
    bonus_waters = units_per_cycle - 1
    waters = pre_window_survival_waters + bonus_waters
    cost = cd["seed"] + (waters + CYCLE_OVERHEAD_ACTIONS) * ACTION_COST
    last = total_days - 1               # final day has no end-of-day; harvest by day 29
    value = 0.0
    ready = day + grow
    while ready <= last:
        value += units_per_cycle * fc.avg_sell_price(
            crop, units_per_cycle, ready - day
        ) - cost
        ready += cycle
    if value == 0.0 and day + cd["first_yield_day"] <= last:
        # Partial cycle at season's end: take whatever the remaining window days give.
        window_end = min(window[1], last - day) if window else 0
        units = 1 + max(0, window_end - (window[0] if window else 0) + 1)
        units = min(cd["max_yield"], units)
        value = units * fc.avg_sell_price(crop, units, last - day) - cost
    return value


def _ongoing_tile_value(cd, crop, day, total_days, fc):
    """NPV of repeated ongoing plantings on one tile, planted today.

    An ongoing crop gets exactly `max_yield` productions (one unit each) starting at
    first_yield_day and every `interval` days, then decays; the tile is replanted.
    """
    prods = cd["max_yield"]
    span = cd["first_yield_day"] + (prods - 1) * cd["interval"]
    cycle = span + 3                    # + decay tail, DIG, replant
    waters = cycle / 2                  # survival only; ongoing bonus needs fertilizer
    cost = cd["seed"] + (waters + prods / 2 + CYCLE_OVERHEAD_ACTIONS) * ACTION_COST
    last = total_days - 1
    value = 0.0
    start = day
    while start + cd["first_yield_day"] <= last:
        fit = 1 + (last - (start + cd["first_yield_day"])) // cd["interval"]
        units = min(prods, fit)
        sell_day = start + cd["first_yield_day"] + (units - 1) * cd["interval"]
        value += units * fc.avg_sell_price(crop, units, sell_day - day) \
            - cost * (units / prods)
        start += cycle
    return value


def crop_tile_value(crop, day, total_days, fc):
    """NPV of planting `crop` on an empty tile today, over the rest of the season."""
    cd = CROPS[crop]
    if cd["ongoing"]:
        return _ongoing_tile_value(cd, crop, day, total_days, fc)
    return _one_time_tile_value(cd, crop, day, total_days, fc)


def best_crop(day, total_days, fc, crops=None):
    """(crop, value) with the highest tile NPV today."""
    ranked = sorted(((crop_tile_value(c, day, total_days, fc), c)
                     for c in (crops or CROPS)), reverse=True)
    return ranked[0][1], ranked[0][0]


def standing_plant_value(tile, day, total_days, fc):
    """What a live plant is still worth -- the loss an urgent WATER tonight prevents."""
    cd = CROPS.get(tile.get("crop"))
    if cd is None:
        return 0.0
    price = fc.price(tile["crop"])
    banked = tile.get("yield_units", 0) * price
    if cd["ongoing"]:
        prods_done = max(0, (day - tile.get("planted_day", 0)
                             - cd["first_yield_day"]) // cd["interval"] + 1)
        remaining_prods = max(0, cd["max_yield"] - prods_done)
        future = remaining_prods * price
    else:
        future = max(0, cd["max_yield"] - tile.get("yield_units", 0)) * price
    remaining_care = 3 * ACTION_COST   # a few more waters/harvest actions
    return banked + future - remaining_care


def fertilize_value(tile, day, fc, ongoing_gain=0.0):
    """Deprecated alias for `fertilizer_use_advantage` (the FERTILIZE job's own value)."""
    return fertilizer_use_advantage(tile, day, fc, ongoing_gain)


def water_bonus_value(tile, day, fc):
    """Tonight's watering bonus in dollars (+2 if the tile is fertilized, else +1)."""
    price = fc.price(tile.get("crop"))
    return (2 if tile.get("fertilized_until_day", -1) >= day else 1) * price


def fertilizer_use_advantage(tile, day, fc, ongoing_gain=0.0):
    """How much better USING one fertilizer on `tile` is than SELLING it.

    Extra yield from FERTILIZE's 3-day cover, minus the forgone sale. This is a NET
    advantage: its baseline is "sold", not "not owned". R5 found the two baselines were
    being mixed -- see `fertilizer_gross_marginal_value`.
    """
    cd = CROPS.get(tile.get("crop"))
    if cd is None:
        return 0.0
    if cd["ongoing"]:
        # An ongoing crop accrues +1 per scheduled production, or +2 if it is watered AND
        # fertilized that day -- so cover on a production day doubles that event's output.
        # This returned 0 unconditionally ("hard to time"), which is why 0% of our strawberry
        # is ever fertilized while it harvests at 1.00 unit against a cap of 4. `ongoing_gain`
        # prices it; 0.0 keeps the historical behaviour.
        if not ongoing_gain:
            return 0.0
        headroom = cd["max_yield"] - tile.get("yield_units", 0)
        if headroom <= 0:
            return 0.0
        age = day - tile.get("planted_day", 0)
        since = age - cd["first_yield_day"]
        if since < -2:                      # cover lasts three days; nothing to catch yet
            return 0.0
        # Productions landing inside the three-day cover window.
        interval = max(1, cd["interval"])
        hits = sum(1 for d in (age, age + 1, age + 2)
                   if d >= cd["first_yield_day"] and (d - cd["first_yield_day"]) % interval == 0)
        extra = min(hits, headroom)
        return float(ongoing_gain) * extra * fc.price(tile["crop"]) - fc.price("FERTILIZER")
    window = water_bonus_window(tile["crop"])
    if not window:
        return 0.0
    age = day - tile.get("planted_day", 0)
    headroom = cd["max_yield"] - tile.get("yield_units", 0)
    if headroom <= 0:
        return 0.0
    cover = sum(1 for d in (age, age + 1, age + 2) if window[0] <= d <= window[1])
    extra = min(cover, headroom)
    return extra * fc.price(tile["crop"]) - fc.price("FERTILIZER")


# --- animals ------------------------------------------------------------------

def animal_units_per_production(kind):
    """Units one scheduled production yields for a daily fed+cared animal.

    The engine adds the base unit plus the banked CARE bonus, and the bank grows by one for
    every fed+cared day since the previous production (engine `pending_care_bonus`): a goose
    yields 2 eggs a day, a cow 3 milk every 2 days, a sheep 4 wool every 3 days.
    """
    return 1 + ANIMALS[kind]["interval"]


def animal_daily_net(kind, fc, fertilize_use=0.0, horizon=0.0, gross_fertilizer=True,
                     action_cost=None):
    """Net $/day of one fed+cared animal of `kind`: its product stream (base + banked care,
    averaged over the production interval) + 1 fertilizer, minus feed and labour. Prices are
    evaluated `horizon` days out (mid-season by default callers). Every animal eats one
    wheat a day and takes the same service actions; only the product line differs."""
    a = ANIMALS[kind]
    per_production = animal_units_per_production(kind)
    product = per_production * fc.avg_sell_price(a["product"], per_production, horizon)
    product /= a["interval"]
    fert = fertilizer_gross_marginal_value(fc, fertilize_use, horizon, gross_fertilizer)
    feed = fc.buy_price("WHEAT")
    unit_action = ACTION_COST if action_cost is None else float(action_cost)
    return product + fert - feed - ANIMAL_ACTIONS_PER_DAY * unit_action


def goose_daily_net(fc, fertilize_use=0.0, horizon=0.0):
    """Net $/day of one fed+cared goose: 2 eggs (base + care) + 1 fertilizer, minus feed
    and labour. Prices evaluated `horizon` days out (mid-season by default callers)."""
    return animal_daily_net("GOOSE", fc, fertilize_use, horizon)


def animal_npv(kind, day, total_days, fc, fertilize_use=0.0, action_cost=None):
    """NPV of one more animal of `kind` bought and placed today (purchase price included)."""
    prod_start = day + ANIMALS[kind]["first_yield_day"]
    days = (total_days - 1) - prod_start + 1
    if days <= 0:
        return -float(ANIMALS[kind]["cost"])
    return days * animal_daily_net(kind, fc, fertilize_use, horizon=days / 2,
                                   action_cost=action_cost) \
        - ANIMALS[kind]["cost"]


def goose_npv(day, total_days, fc, fertilize_use=0.0):
    """NPV of one more goose bought and placed today (purchase price included)."""
    return animal_npv("GOOSE", day, total_days, fc, fertilize_use)


def animal_remaining_value(tile, day, total_days, fc, fertilize_use=0.0, action_cost=None):
    """Remaining NPV of a living animal -- the loss an urgent FEED tonight prevents.
    The purchase price is sunk, so this is just remaining production net of costs."""
    days = (total_days - 1) - day
    if days <= 0:
        return 0.0
    kind = tile.get("animal", "GOOSE")
    return days * animal_daily_net(kind, fc, fertilize_use, horizon=days / 2,
                                   action_cost=action_cost)


def care_value(tile, fc, assume_fed=False):
    """CARE banks +1 product, paid out on the next production day the animal is also fed.

    The engine's CARE op has no `fed_today` precondition -- only end-of-day's bank increment
    does -- so `assume_fed` prices the action for a caller that is arranging the FEED in the
    same turn. Without it the value is zero until feeding has already been observed.
    """
    if not (assume_fed or tile.get("fed_today")):
        return 0.0
    product = ANIMALS[tile["animal"]]["product"]
    return fc.price(product, 1)


def routine_feed_value(tile, fc, fertilize_use=0.0, action_cost=None):
    """A routine (non-urgent) FEED keeps the care bonus and the production chain alive."""
    return animal_daily_net(tile.get("animal", "GOOSE"), fc, fertilize_use,
                            action_cost=action_cost)


def herd_daily_net(animals, fc, fertilize_use=0.0):
    """Summed daily net of the living herd (tiles with an `animal`); the feed-security premium."""
    return sum(animal_daily_net(t.get("animal", "GOOSE"), fc, fertilize_use) for t in animals)


# --- the remaining named decisions -------------------------------------------

def fertilizer_sale_value(fc, horizon=0.0):
    """Gross cash from selling one FERTILIZER `horizon` days out."""
    return fc.price("FERTILIZER", horizon)


def fertilizer_gross_marginal_value(fc, use_advantage=0.0, horizon=0.0, gross=True):
    """What OWNING one more FERTILIZER is worth, baselined at not owning it.

    R5's P0.1: `fertilize_value` returns a net advantage of use over sale, so
    `max(sale, advantage)` compared two different baselines and understated the good case.
    If fertilizer sells for $100 and the best use adds $750 of crop, the advantage is $650
    and the correct marginal value is $750, not $650.
    """
    if not gross:   # legacy control: compares a gross sale against a net advantage
        return max(fertilizer_sale_value(fc, horizon), use_advantage)
    return fertilizer_sale_value(fc, horizon) + max(0.0, use_advantage)


def fertilizer_value(fc, use_value=0.0):
    """Deprecated alias for `fertilizer_gross_marginal_value`, kept for frozen baselines."""
    return fertilizer_gross_marginal_value(fc, use_value)


def goose_breakeven_day(total_days, fc, fertilize_use=0.0):
    """Last day a goose purchase still has positive NPV -- the computed buy cutoff."""
    daily = goose_daily_net(fc, fertilize_use)
    if daily <= 0:
        return -1
    days_needed = ANIMALS["GOOSE"]["cost"] / daily
    return int((total_days - 1) - ANIMALS["GOOSE"]["first_yield_day"] - days_needed + 1)


def last_profitable_plant_day(crop, total_days, fc):
    """Last day planting `crop` still has positive tile NPV (not merely reachable)."""
    for day in range(total_days - 1, -1, -1):
        if crop_tile_value(crop, day, total_days, fc) > 0:
            return day
    return -1


def hand_npv(hires_today, marginal_action_value):
    """One hired hand for a day: 24 unit-actions at the marginal action value, minus the
    fib hire cost. Positive while labour is the binding constraint, which is almost
    always -- the meaningful question is the marginal action value, not the hire cost."""
    return 24 * marginal_action_value - hire_cost(hires_today)


def town_daily_drain(town, turns_per_day=24, shop_interval=4, center_interval=24):
    """Units the town removes from the market per day, per product -- EXACTLY known.

    `town["unlocked_shops"]` is public and lists duplicates separately, because shops are drawn
    with replacement and each instance consumes independently.  Every instance takes
    `2 if len(products) == 1 else 1` of each of its products every `shop_interval` steps, and the
    town centre takes 1 of every non-fertilizer product every `center_interval` steps.  None of
    it ever replenishes.

    The drain-rate EMA has to discover all of this from inventory deltas that also contain both
    players' selling, and it cannot react to a shop unlock until several days after the fact.
    This is the deterministic half, available the turn the shop appears.
    """
    shops_per_day = turns_per_day / max(1, shop_interval)
    center_per_day = turns_per_day / max(1, center_interval)
    drain = {item: center_per_day for item in TOWN_CENTER_PRODUCTS}
    for shop in (town or {}).get("unlocked_shops", []) or []:
        products = SHOPS.get(shop)
        if not products:
            continue
        multiplier = 2 if len(products) == 1 else 1
        for item in products:
            drain[item] = drain.get(item, 0.0) + shops_per_day * multiplier
    return drain


def hold_premium(item, day, total_days, fc):
    """Expected one-day price drift from the town-centre drain alone (deterministic,
    never replenishes). The Forecast already folds the observed net drain into every
    price; this isolates the structural town-only component, as a reference value and
    aux target for the sell/hold decision."""
    per_day = 1.0  # flat one-unit daily town-center tick since kaggle-environments 1.32.6
    return price_at(item, max(0.0, fc.inv(item) - per_day)) - fc.price(item)


def land_npv(day, total_days, fc, n_tiles=25, walk_tax=0.25, price=1000):
    """One more quadrant: n_tiles of the best crop, minus the purchase price and a
    walking tax per tile (extra distance on every job, in action-cost units). Measured
    negative in arena (docs/ECONOMY.md) -- kept as a callable so the number exists."""
    _, tile_value = best_crop(day, total_days, fc)
    return n_tiles * tile_value * (1 - walk_tax) - price


def harvest_timing(tile, day, total_days, fc):
    """(harvest_now, harvest_wait) values for a one-time crop still below its cap.

    now:  bank the units today (marginal sale price) and restart the cycle sooner.
    wait: tonight's watered bonus (x2 if fertilized) at the same price, minus the day
    of cycle time it costs. The executor's _ready_to_harvest rules encode this; the
    callable exists so the comparison is explicit and learnable.
    """
    crop = tile.get("crop")
    units = tile.get("yield_units", 0)
    now = units * fc.avg_sell_price(crop, units)
    wait = water_bonus_value(tile, day, fc) - ACTION_COST
    return now, wait


# --- labour as a dual, not a constant -----------------------------------------

def action_shadow_price(job_values, capacity, *, floor=0.0, ceiling=None, default=ACTION_COST):
    """What one more unit-action is actually worth right now.

    `ACTION_COST = 7.0` is a single calibrated number applied to every state, and all six
    2026-08-31 review returns said the same thing about it: labour is a capacity dual, not a
    universal fact. R3: "it should become a state-dependent dual"; R5 warned that retuning the
    constant before the calendars are right would let it absorb several modelling errors at
    once.

    The dual is directly observable from the job surface we already build. Rank this turn's
    jobs by value and take the one just past what our units can serve: that is the work a
    marginal action would buy, and therefore what an action costs to spend elsewhere. When the
    farm has more labour than jobs -- which it does for 18.5% of its turns -- the marginal job
    is worth nothing and an action is free; when the board is dense it is worth far more
    than $7.

    `capacity` is unit-actions available this turn. Returns `default` when the surface is
    empty, so a caller with no information is no worse off than the constant.
    """
    values = sorted((float(v) for v in job_values), reverse=True)
    if not values:
        return default
    capacity = max(0, int(capacity))
    price = values[capacity] if capacity < len(values) else floor
    if ceiling is not None:
        price = min(price, float(ceiling))
    return max(float(floor), float(price))
