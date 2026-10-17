"""Day-zero funding screen for dated producer portfolios; a necessary cut, never a certificate.

Graduated from work/runs/reviewer-play/portfolio-knapsack-df951c02/funded_screen.py
(K-PORTFOLIO-ZERO-RECEIPT-CUT-DF951C02). For a portfolio whose modules produce
nothing on day zero, with empty starting stocks, no borrowing and no off-tape
trading, day zero must be paid from starting cash alone:

    cash >= acquisitions_0 + land_0 + sum_p sum_{k=1..B_p} P_p(ceiling_p - k)

where B_p are the input units bought on day zero and P_p the native quote at the
post-buy inventory, evaluated at a declared exogenous inventory ceiling (the most
optimistic price). Wages, travel and demand are omitted, so the bound is low: a
portfolio it rejects is unfunded; one it passes is not thereby feasible. The cut is
hereditary over an alphabet without day-zero output: adding a column cannot repair
it. It says nothing about later days, whose shortfalls complementary output can repair.
"""
from __future__ import annotations

LAND_CUMULATIVE = ((25, 0), (50, 1000), (75, 3000), (100, 7000))
UNKNOWN_UNBOUNDED_SUPPLY = "UNKNOWN_UNBOUNDED_SUPPLY"
UNKNOWN_PREFIX_OUTSIDE_CONTRACT = "UNKNOWN_PREFIX_OUTSIDE_CONTRACT"
REJECT_UNFUNDED = "REJECT_UNFUNDED"
FUNDED_PREFIX = "FUNDED_PREFIX_NOT_A_FEASIBILITY_CERTIFICATE"


def land_cost(sites: int) -> int:
    """Cumulative native land dollars for `sites` reserved sites (25 free, then 50/75/100)."""
    if type(sites) is not int or sites < 0:
        raise ValueError("nonnegative integer site count required")
    for cap, cost in LAND_CUMULATIVE:
        if sites <= cap:
            return cost
    raise ValueError("more than 100 sites cannot be reserved")


def purchase_cost(engine, product: str, units: int, ceiling: int) -> int:
    """Native BUY quotes for `units` sequential purchases starting at inventory `ceiling`."""
    if type(units) is not int or units < 0 or type(ceiling) is not int or ceiling < 0:
        raise ValueError("nonnegative integer units and ceiling required")
    return sum(int(engine.market_price(product, ceiling - k)) for k in range(1, units + 1))


def screen_no_receipt_day0(engine, *, acquisitions: int, sites: int, purchases: dict[str, int],
                           inventory_ceiling: dict[str, int] | None, starting_cash: int = 3000,
                           day_zero_output_units: int = 0, starting_stock_units: int = 0,
                           borrowing_or_off_tape_trade: bool = False) -> dict:
    """Return the status, the required dollars and the shortfall for one day-zero prefix."""
    for name, value in (("acquisitions", acquisitions), ("starting_cash", starting_cash),
                        ("day_zero_output_units", day_zero_output_units), ("starting_stock_units", starting_stock_units)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name}: nonnegative integer required")
    result = {"acquisitions": acquisitions, "land": land_cost(sites), "purchases": {}, "required": None,
              "shortfall": None, "starting_cash": starting_cash}
    if inventory_ceiling is None or any(p not in inventory_ceiling for p in purchases):
        result["status"] = UNKNOWN_UNBOUNDED_SUPPLY
        return result
    if day_zero_output_units or starting_stock_units or borrowing_or_off_tape_trade:
        result["status"] = UNKNOWN_PREFIX_OUTSIDE_CONTRACT
        return result
    for product, units in purchases.items():
        result["purchases"][product] = purchase_cost(engine, product, units, int(inventory_ceiling[product]))
    required = acquisitions + result["land"] + sum(result["purchases"].values())
    result["required"] = required
    result["shortfall"] = max(0, required - starting_cash)
    result["status"] = REJECT_UNFUNDED if required > starting_cash else FUNDED_PREFIX
    return result
