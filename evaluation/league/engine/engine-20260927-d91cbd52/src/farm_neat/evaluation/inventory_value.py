"""Capacity-coupled value envelopes for the maintained nightly-stock quotient.

This is a CPU component analysis, NOT a dispatcher, selling policy or WDL head.
The caller supplies separable value intervals indexed by admitted product count.
They can describe explicit sale revenue and a declared holding-value envelope.
Their validity is a caller assumption; exact optimization does not validate them.

PREFIX LEMMA. A capacity-limited ordering retains a subset of complete stacks and
at most one strictly partial stack. Conversely, put any such complete subset
first, its partial stack next, and the rejected stacks last: the native prefix
admits exactly those counts. Thus a product-subtotal DP with ONE GLOBAL split
flag gives exact extrema of the supplied lower/upper functions over arbitrary
whole-stack permutations. It does not assert that any route/rematch can realize
these permutations, nor that endpoints of market/holding intervals co-occur.

The unit-allocation relaxation drops stack indivisibility but keeps sum(A)=m;
independent product bounds drop that equality too. All three are returned so a
learner does not mistake simultaneous marginal maxima for attainable inventory.
Ordered deposit itself is always delegated to inventory_quotient.deposit_quotient.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from farm_neat.evaluation.common import equal, file_sha, read, write_new
from farm_neat.evaluation.inventory_quotient import deposit_quotient
from farm_neat.game.constants import PRODUCTS

SCHEMA = "farm-inventory-value-envelope-v1"
SCOPE = "CONDITIONAL_SEPARABLE_VALUE_ENVELOPE_NOT_ROUTE_FEASIBILITY_OR_WDL"


def _integer(value: object, label: str, *, signed: bool = False) -> int:
    if type(value) is not int or (not signed and value < 0):
        raise ValueError(label + ": integer" + ("" if signed else " >= 0") + " required")
    return value


def _interval(value: object, label: str, *, signed: bool = False) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(label + ": [lower, upper] required")
    lo, hi = (_integer(v, label, signed=signed) for v in value)
    if lo > hi:
        raise ValueError(label + ": lower exceeds upper")
    return [lo, hi]


class _BudgetExceeded(Exception):
    pass


class _Budget:
    def __init__(self, limit: int):
        self.limit = _integer(limit, "max_transitions")
        self.used = 0

    def tick(self):
        if self.used >= self.limit:
            raise _BudgetExceeded
        self.used += 1


def _stock_case(shed, inventories, capacity):
    result = deposit_quotient(shed, inventories, capacity)
    stacks = [(item, n) for inventory in inventories for item, n in inventory if n]
    items = sorted(set(shed) | set(result["carried_by_item"]))
    m = sum(result["admitted_by_item"].values())
    return result, stacks, items, m


def sale_value_tables(shed: dict, inventories: list, capacity: int,
                      sales: dict, price_bounds: dict,
                      holding_bounds: dict | None = None,
                      fill: str = "full") -> dict:
    """Compile a deliberately restricted, explicit SALE/HOLD value envelope.

    sales maps product -> {quantity, reserve}. Only active, trigger-eligible
    SALEs with available queue slots, no same-product acquisition, no other stock
    changes, and no unmet production requisitions are in scope. Reserve is the
    policy's own reserve; this helper does NOT insert requisition/strategic stock.
    Absence of SALE means HOLD. At most one SALE per product is represented.

    q(n)=min(quantity, max(0,n-reserve)); the canonical queue is tested separately.
    fill='full' means a valid stocked native SELL (own total <= native loop budget).
    fill='unknown' allows any f in [0,q]; it is a conservative external-window
    envelope, NOT a claim that the opponent can prevent an admitted native SELL.
    Conditional unit-price [l,u] and residual-stock value [hL,hU] give
      full: [l*q+hL*(n-q), u*q+hU*(n-q)]
      unknown: [hL*n+min(0,l-hL)*q, hU*n+max(0,u-hU)*q].
    Holding values default to zero: revenue-only, not a forecast of liquidation.
    At terminal time unsold inventory has ZERO bank-cash salvage in this engine.
    """
    quotient, _, items, m = _stock_case(shed, inventories, capacity)
    if fill not in ("full", "unknown"):
        raise ValueError("fill must be full or unknown")
    if not isinstance(sales, dict) or not isinstance(price_bounds, dict):
        raise ValueError("sales and price_bounds mappings required")
    holding_bounds = {} if holding_bounds is None else holding_bounds
    if not isinstance(holding_bounds, dict):
        raise ValueError("holding_bounds mapping required")
    if set(sales) - set(PRODUCTS):
        raise ValueError("SALE product not supported by native engine")
    if set(price_bounds) != set(sales):
        raise ValueError("price_bounds must bind exactly the explicit SALE products")
    if set(holding_bounds) - (set(items) | set(sales)):
        raise ValueError("holding bound for absent product")
    prices, holds = {}, {}
    for item, sale in sales.items():
        if not isinstance(sale, dict) or set(sale) != {"quantity", "reserve"}:
            raise ValueError("each sale requires exactly quantity and reserve")
        for key, value in sale.items():
            _integer(value, "sale " + key)
        prices[item] = _interval(price_bounds[item], "unit sale price")
        if prices[item][0] < 1:
            raise ValueError("native sale prices have a one-dollar floor")
    for item, interval in holding_bounds.items():
        holds[item] = _interval(interval, "holding unit value")
    tables = {}
    for item in items:
        n0 = shed.get(item, 0)
        max_a = min(m, quotient["carried_by_item"].get(item, 0))
        sale = sales.get(item, {"quantity": 0, "reserve": 0})
        lo, hi = prices.get(item, [0, 0])
        hl, hu = holds.get(item, [0, 0])
        rows = []
        for a in range(max_a + 1):
            n = n0 + a
            q = min(sale["quantity"], max(0, n - sale["reserve"]))
            if fill == "full":
                rows.append([lo*q + hl*(n-q), hi*q + hu*(n-q)])
            else:
                rows.append([hl*n + min(0, lo-hl)*q,
                             hu*n + max(0, hi-hu)*q])
        tables[item] = rows
    return tables


def _product_options(stacks, item, m, budget):
    # One witness per (subtotal, used_strict_split). Complete stacks before the
    # cutoff can be reordered freely, so their internal order is irrelevant.
    states = {(0, 0): ((), None)}
    for index, (product, n) in enumerate(stacks):
        if product != item:
            continue
        following = dict(states)  # reject this whole stack
        for (a, split), (whole, cutoff) in states.items():
            if a + n <= m:
                budget.tick()
                following.setdefault((a+n, split), (whole+(index,), cutoff))
            if not split:
                for r in range(1, min(n-1, m-a) + 1):
                    budget.tick()
                    following.setdefault((a+r, 1), (whole, (index, r)))
        states = following
    return states


def _optimize(items, tables, options, m, budget):
    # Each record keeps separate lower-minimizing and upper-maximizing paths.
    states = {(0, 0): (0, 0, (), ())}
    for item in items:
        following = {}
        for (used, split), (lo, hi, low_path, high_path) in states.items():
            for a, partial in options[item]:
                if used+a > m or split+partial > 1:
                    continue
                budget.tick()
                key = (used+a, split+partial)
                l, u = tables[item][a]
                low = (lo+l, low_path+((a, partial),))
                high = (hi+u, high_path+((a, partial),))
                previous = following.get(key)
                if previous is not None:
                    if previous[0] <= low[0]:
                        low = previous[0], previous[2]
                    if previous[1] >= high[0]:
                        high = previous[1], previous[3]
                following[key] = (low[0], high[0], low[1], high[1])
        states = following
    finals = [record for (used, _split), record in states.items() if used == m]
    if not finals:
        raise AssertionError("valid cargo always has at least one capacity prefix")
    low = min(finals, key=lambda r: r[0])
    high = max(finals, key=lambda r: r[1])
    return {"lower": low[0], "upper": high[1]}, low[2], high[3]


def _witness(items, selections, options, stacks, shed, capacity, value):
    whole, cutoffs, admissions = [], [], {}
    for item, (a, partial) in zip(items, selections):
        selected, cutoff = options[item][a, partial]
        whole.extend(selected)
        if cutoff is not None:
            cutoffs.append(cutoff)
        if a:
            admissions[item] = a
    assert len(cutoffs) <= 1
    cutoff_indices = [index for index, _r in cutoffs]
    used = set(whole + cutoff_indices)
    order = whole + cutoff_indices + [i for i in range(len(stacks)) if i not in used]
    # Use one hypothetical carrier per original whole stack: this is a permutation
    # witness only, NOT an assertion about feasible worker custody or rematching.
    inventories = [[[stacks[i][0], stacks[i][1]]] for i in order]
    decoded = deposit_quotient(shed, inventories, capacity)
    equal(decoded["admitted_by_item"], admissions, "DP witness native-prefix decoder")
    return {"flat_stack_indices": order, "admitted_by_item": admissions,
            "strictly_partial_stack": list(cutoffs[0]) if cutoffs else None,
            "value_envelope_endpoint": value,
            "route_feasibility": "NOT_CHECKED_ARBITRARY_STACK_PERMUTATION"}


def inventory_value_bounds(shed: dict, inventories: list, capacity: int,
                           values: dict, max_transitions: int = 2_000_000) -> dict:
    """Optimize product-additive interval tables under true stack-prefix coupling.

    values[p][a] is [lower,upper] for INITIAL shed[p]+a stock, not just cargo.
    There must be one entry for a=0..min(total cargo[p], admitted total). Tables
    may be nonlinear/nonmonotone and signed, but must contain exact integers.
    No claim of independent market endpoint realizability is made. UNKNOWN on
    exhausted transition budget retains only previously completed valid bounds.

    Runtime: O((K+P)*m^2) transitions in a conservative bound, rather than K!
    permutations, where K=positive stacks, P=products and m=admitted units. Work
    is pseudo-polynomial in capacity, not whole-game/economic-program search.
    """
    quotient, stacks, items, m = _stock_case(shed, inventories, capacity)
    if not isinstance(values, dict) or set(values) != set(items):
        raise ValueError("values must cover exactly all shed/cargo products")
    tables, bounds = {}, {}
    for item in items:
        maximum = min(m, quotient["carried_by_item"].get(item, 0))
        rows = values[item]
        if not isinstance(rows, (list, tuple)) or len(rows) != maximum + 1:
            raise ValueError("value table length mismatch for " + item)
        tables[item] = [_interval(r, item + " value", signed=True) for r in rows]
        bounds[item] = quotient["arbitrary_stack_order_bounds"].get(
            item, {"minimum": 0, "maximum": 0})
    independent = {"lower": 0, "upper": 0}
    known = {"lower": 0, "upper": 0}
    for item in items:
        b = bounds[item]
        independent["lower"] += min(tables[item][a][0] for a in range(b["minimum"], b["maximum"]+1))
        independent["upper"] += max(tables[item][a][1] for a in range(b["minimum"], b["maximum"]+1))
        l, u = tables[item][quotient["admitted_by_item"].get(item, 0)]
        known["lower"] += l
        known["upper"] += u
    budget = _Budget(max_transitions)
    result = {"schema": SCHEMA, "scope": SCOPE, "status": "COMPLETE",
              "deposit": quotient, "admitted_total": m, "positive_stacks": len(stacks),
              "known_order_envelope": known,
              "independent_product_envelope": independent,
              "capacity_relaxation_envelope": None,
              "arbitrary_stack_envelope": None,
              "price_and_holding_bounds_validated": False,
              "terminal_wdl": "NOT_INFERRED"}
    try:
        relaxed = {p: {(a, 0): None for a in range(bounds[p]["minimum"], bounds[p]["maximum"]+1)}
                   for p in items}
        interval, lo, hi = _optimize(items, tables, relaxed, m, budget)
        result["capacity_relaxation_envelope"] = {
            **interval, "lower_admissions": {p: a for p, (a, _) in zip(items, lo) if a},
            "upper_admissions": {p: a for p, (a, _) in zip(items, hi) if a},
            "stack_realizability": "NOT_GUARANTEED"}
        exact = {p: _product_options(stacks, p, m, budget) for p in items}
        interval, lo, hi = _optimize(items, tables, exact, m, budget)
        result["arbitrary_stack_envelope"] = {
            **interval,
            "lower_witness": _witness(items, lo, exact, stacks, shed, capacity, interval["lower"]),
            "upper_witness": _witness(items, hi, exact, stacks, shed, capacity, interval["upper"]),
            "tightness": "EXACT_EXTREMA_OF_SUPPLIED_ENVELOPES_OVER_ARBITRARY_STACK_ORDERS"}
    except _BudgetExceeded:
        result["status"] = "UNKNOWN"
        result["reason"] = "DP_TRANSITION_BUDGET_EXHAUSTED_NOT_INFEASIBLE_OR_HARMFUL"
    result["dp_transitions"] = budget.used
    result["max_transitions"] = budget.limit
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "verify"))
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON: shed, inventories, capacity, values (bound tables)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-transitions", type=int, default=2_000_000)
    args = parser.parse_args()
    data = read(args.input, binding=False)
    if not isinstance(data, dict) or set(data) != {"shed", "inventories", "capacity", "values"}:
        raise ValueError("exact shed/inventories/capacity/values input required")
    import farm_neat.evaluation.inventory_quotient as prefix
    result = {"schema": SCHEMA, "input_sha256": file_sha(args.input),
              "sources": {"inventory_value": file_sha(__file__), "inventory_quotient": file_sha(prefix.__file__)},
              "result": inventory_value_bounds(**data, max_transitions=args.max_transitions),
              "new_games": 0, "model_calls": 0, "optimizer_updates": 0}
    if args.mode == "audit":
        write_new(args.out, result)
    else:
        expected = read(args.out)
        expected.pop("content_sha256")
        equal(result, expected, "all capacity-coupled value fields")
    print(result["result"]["status"] + ": conditional stock/value analysis; no game or model.")


if __name__ == "__main__":
    main()
