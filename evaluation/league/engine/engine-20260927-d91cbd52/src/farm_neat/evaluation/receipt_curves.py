"""Exact conditional terminal-batch receipt curves.

The functions here deliberately stop at a declared market state.  They do not
claim that season receipts are a function of quantity alone: callers must supply
demand and any prior effective rival supply, and must keep timing/flow outside
this abstraction when those can change the market state.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import itertools
import math


def tick_count(start: int, stop: int, interval: int) -> int:
    """Number of multiples of ``interval`` in ``[start, stop)``."""
    if min(start, stop) < 0 or interval <= 0:
        raise ValueError("nonnegative endpoints and positive interval required")
    return max(0, (stop - 1) // interval - (start - 1) // interval)


def shop_contribution(engine, shop: str, product: str) -> int:
    """Per shop-demand tick units for one product under default shop rules."""
    products = engine.SHOPS[shop]
    if product not in products:
        return 0
    return 2 if len(products) == 1 else 1


def full_declared_demand(engine, shops: list[str], product: str, *, last_market: int = 718) -> int:
    """Evaluator-only total town/shop demand before ``last_market``."""
    if len(shops) != 8:
        raise ValueError("eight declared shops required")
    baseline = tick_count(0, last_market, 24)
    return baseline + sum(
        shop_contribution(engine, shop, product) * tick_count(72 * (j + 1), last_market, 4)
        for j, shop in enumerate(shops)
    )


def hidden_demand_distribution(
    engine,
    product: str,
    visible_shops: list[str],
    *,
    total_slots: int = 8,
    last_market: int = 718,
) -> dict:
    """Exact productwise demand distribution after a published shop prefix.

    Unpublished shop identities are iid uniform over the engine shop catalog.
    The return contains a Counter ``demand_multiplicity`` whose mass sums to
    ``len(engine.SHOPS) ** hidden_slots``.  Enumeration is over distinct
    contribution values, not joint shop identities, preserving multiplicity.
    """
    if len(visible_shops) > total_slots:
        raise ValueError("visible prefix longer than total slots")
    for shop in visible_shops:
        if shop not in engine.SHOPS:
            raise ValueError("unknown visible shop")
    slot_weights = [tick_count(72 * (j + 1), last_market, 4) for j in range(total_slots)]
    baseline = tick_count(0, last_market, 24)
    fixed = baseline + sum(
        shop_contribution(engine, shop, product) * slot_weights[j]
        for j, shop in enumerate(visible_shops)
    )
    counts = Counter(shop_contribution(engine, shop, product) for shop in sorted(engine.SHOPS))
    values = sorted(counts)
    hidden_slots = total_slots - len(visible_shops)
    totals = Counter()
    classes = 0
    for path in itertools.product(values, repeat=hidden_slots):
        classes += 1
        multiplicity = math.prod(counts[v] for v in path)
        demand = fixed + sum(v * slot_weights[len(visible_shops) + j] for j, v in enumerate(path))
        totals[demand] += multiplicity
    denominator = len(engine.SHOPS) ** hidden_slots
    if sum(totals.values()) != denominator:
        raise AssertionError("hidden demand probability mass mismatch")
    return {
        "visible_shops": list(visible_shops),
        "slot_weights": slot_weights,
        "baseline_demand": baseline,
        "contribution_multiplicity": dict(sorted(counts.items())),
        "contribution_path_classes": classes,
        "denominator": denominator,
        "demand_multiplicity": totals,
    }


def marginal_receipts(engine, product: str, demand: int, prior_effective_supply: int, qcap: int) -> list[int]:
    """Native quote for each sequential own sell unit in one terminal batch."""
    if min(demand, prior_effective_supply, qcap) < 0:
        raise ValueError("nonnegative demand/supply/quantity required")
    inventory = int(engine.MARKET_PARAMS[product]["I0"]) - demand + prior_effective_supply
    out = []
    for _ in range(qcap):
        price = int(engine.market_price(product, inventory))
        out.append(price)
        # Native floor-price fills do not increase market inventory.
        if price > 1:
            inventory += 1
    return out


def expected_marginal_numerators(
    engine,
    product: str,
    demand_multiplicity: Counter,
    prior_effective_supply: int,
    qcap: int,
) -> list[int]:
    """Exact numerators for expected marginal receipts over a finite distribution."""
    out = [0] * qcap
    for demand, multiplicity in demand_multiplicity.items():
        row = marginal_receipts(engine, product, int(demand), prior_effective_supply, qcap)
        for i, value in enumerate(row):
            out[i] += int(multiplicity) * value
    return out


def first_unit_below_average_cost(nums: list[int], denominator: int, cash: int, units: int) -> int | None:
    if denominator <= 0 or cash < 0 or units <= 0:
        raise ValueError("valid denominator/cash/units required")
    cost = Fraction(cash, units)
    for q, numerator in enumerate(nums, 1):
        if Fraction(numerator, denominator) < cost:
            return q
    return None


def first_nonpositive_producer_block(
    nums: list[int], denominator: int, units: int, cash: int, *, max_producers: int = 100
) -> tuple[int | None, Fraction | None]:
    """First n whose nth producer block has expected gross minus cash <= 0."""
    if denominator <= 0 or units <= 0 or cash < 0 or max_producers < 0:
        raise ValueError("valid denominator/units/cash/count required")
    if len(nums) < units * max_producers:
        raise ValueError("marginal receipt vector shorter than requested producer range")
    prefix = [0]
    for value in nums:
        prefix.append(prefix[-1] + value)
    for n in range(1, max_producers + 1):
        lo, hi = (n - 1) * units, n * units
        delta = Fraction(prefix[hi] - prefix[lo], denominator) - cash
        if delta <= 0:
            return n, delta
    return None, None


def producer_cut(
    engine,
    product: str,
    distribution: dict,
    prior_effective_supply: int,
    units: int,
    cash: int,
    *,
    max_producers: int = 100,
) -> dict:
    qcap = units * max_producers
    nums = expected_marginal_numerators(
        engine, product, distribution["demand_multiplicity"], prior_effective_supply, qcap
    )
    n, delta = first_nonpositive_producer_block(
        nums, distribution["denominator"], units, cash, max_producers=max_producers
    )
    return {
        "marginal_numerators": nums,
        "denominator": distribution["denominator"],
        "first_nonpositive_producer_block": n,
        "max_retained_producers": max_producers if n is None else n - 1,
        "first_nonpositive_block_net": delta,
        "first_unit_below_average_cost": first_unit_below_average_cost(
            nums, distribution["denominator"], cash, units
        ),
    }


def count_bounded_compositions(caps: dict[str, int], budget: int, order: tuple[str, ...] | list[str]) -> int:
    """Count integer vectors 0<=x_p<=cap_p with sum x_p<=budget."""
    if budget < 0 or any(caps[p] < 0 for p in order):
        raise ValueError("nonnegative caps/budget required")
    dp = [0] * (budget + 1)
    dp[0] = 1
    for product in order:
        nxt = [0] * (budget + 1)
        for used, ways in enumerate(dp):
            if not ways:
                continue
            for n in range(min(caps[product], budget - used) + 1):
                nxt[used + n] += ways
        dp = nxt
    return sum(dp)
