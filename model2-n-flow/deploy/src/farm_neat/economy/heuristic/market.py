"""Market price curve and sell-side planning.

The single most important market fact: both players share one inventory pool, and the premium
goods (STRAWBERRY, MELON, MILK, WOOL) use steep `above` curves that drive them to the $1 floor
after only a few hundred units of glut. Selling is therefore a *marginal revenue* problem, not a
"do I have stock" problem.
"""

import math

from .constants import MARKET_PARAMS, PRICE_FLOOR, PRODUCTS


HINGE_GAIN = 8.0


def shape(func, x, T=None):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def price_at(item, inventory, params=None):
    """Exact replica of the engine's market_price()."""
    p = (params or MARKET_PARAMS)[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / shape(f, T, T)
        price = base + amp * shape(f, I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / shape(f, T, T)
        price = base - amp * shape(f, inventory - I0, T)
    return max(PRICE_FLOOR, int(round(price)))


def sell_revenue(item, inventory, units, params=None):
    """Total revenue for dumping `units` now, accounting for our own price impact.

    Mirrors the engine: each unit is quoted at the pre-sell inventory, then inventory grows by 1
    (unless the price was already at the $1 floor, in which case supply does not grow).
    """
    total = 0
    inv = inventory
    for _ in range(units):
        p = price_at(item, inv, params)
        total += p
        if p > PRICE_FLOOR:
            inv += 1
    return total


def marginal_price(item, inventory, units, params=None):
    """Price the `units`-th additional unit would fetch."""
    inv = inventory
    for _ in range(max(0, units - 1)):
        p = price_at(item, inv, params)
        if p > PRICE_FLOOR:
            inv += 1
    return price_at(item, inv, params)


def units_above_floor(item, inventory, cap, params=None):
    """How many units we can sell before the marginal price reaches the engine's $1 floor."""
    return units_at_or_above_price(item, inventory, cap, PRICE_FLOOR + 1, params)


def units_at_or_above_price(item, inventory, cap, minimum_price, params=None):
    """Largest immediate sale whose every marginal unit meets ``minimum_price``.

    This is an exact bounded search over the engine's integer quote curve. It also mirrors the
    engine's rule that inventory stops increasing once a floor-priced unit is sold.
    """
    n = 0
    inv = inventory
    while n < cap:
        price = price_at(item, inv, params)
        if price < minimum_price:
            break
        n += 1
        if price > PRICE_FLOOR:
            inv += 1
    return n


def plan_sales(shed, market, budget_orders, min_price=2, keep=None, min_ratio=0.0):
    """Choose which shed items to sell this turn, best marginal revenue first.

    Returns a list of ["SELL", item, n] orders, at most `budget_orders` long.
    `keep` maps item -> units to retain (e.g. wheat reserved for animal feed).
    `min_ratio` refuses any sale below `min_ratio * base_price` -- the learned policy's lever for
    holding stock when the market is temporarily glutted.
    `min_price=1` is total liquidation: floor-priced stock still pays $1/unit and frees shed
    room, which matters in the endgame when unsold stock scores zero.
    """
    keep = keep or {}
    prices = market.get("prices", {})
    inventory = market.get("inventory", {})
    candidates = []
    # FERTILIZER is deliberately included: animals shed 1 free fertilizer/day each and the town
    # never consumes it, so collected fertilizer is pure sell-side revenue (~$100 base).
    for item in PRODUCTS:
        have = shed.get(item, 0) - keep.get(item, 0)
        if have <= 0:
            continue
        threshold = max(min_price, min_ratio * MARKET_PARAMS[item]["base"])
        price = prices.get(item, 0)
        if price < threshold:
            continue
        inv = inventory.get(item, MARKET_PARAMS[item]["I0"])
        # Total liquidation explicitly accepts the $1 floor. Otherwise solve the largest
        # quantity whose final marginal quote still satisfies the requested ratio/price floor.
        n = (
            have
            if threshold <= PRICE_FLOOR
            else units_at_or_above_price(item, inv, have, threshold)
        )
        if n <= 0:
            continue
        candidates.append((sell_revenue(item, inv, n), item, n))
    candidates.sort(reverse=True)
    return [["SELL", item, n] for _, item, n in candidates[:budget_orders]]
