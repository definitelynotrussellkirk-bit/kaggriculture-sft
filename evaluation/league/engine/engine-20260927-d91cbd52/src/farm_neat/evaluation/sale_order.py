"""Conditional sale-queue oracles; NOT a whole-game value model.

solve_sale_slots adds guarded mixed-queue screening on exact native phase snapshots.
It never changes production behavior; SELL-only solve retains its original contract.

Assumptions: one SELL entry per product per seat, <=9 own / <=10 other slots,
fixed market-phase stock, default engine pricing, and no BUY/HIRE/BUY_LAND.
Zero-fill entries retain their slots. One item's prices and fills cannot affect
another item's sales. Thus a product/slot reward matrix exactly decomposes own
cash AND opponent cash. Subset assignment needs n*2**(n-1) edges, not n! games.
The legacy solve() rejects mixed transactions, duplicates and price overrides.
solve_sale_slots() handles bounded mixed phases, not future deliveries or value.

OrderBelief receives only past public market/cash changes, own actions/stock,
and an EXPLICIT finite hypothesis set. Its ranges are support ranges, not
calibrated confidence intervals. Unknown quantities/stock must be represented
in that set. It is a component experiment, not a wired live opponent model.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from farm_neat.evaluation.common import read, write_new
from farm_neat.game.exact_engine import AttrDict, DEFAULT_CONFIGURATION, load_engine

PRODUCTS = tuple(load_engine().PRODUCTS)
SCHEMA = "farm-sale-order-assignment-v1"
Order = tuple[str, int]


def _integer(value, name: str, *, minimum=0, maximum=None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"UNSUPPORTED: {name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"UNSUPPORTED: {name} exceeds {maximum}")
    return value


def orders(raw: Sequence, *, limit=9) -> tuple[Order, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) > limit:
        raise ValueError("UNSUPPORTED: queue length")
    result = []
    for row in raw:
        if (not isinstance(row, (list, tuple)) or len(row) != 3 or
                row[0] != "SELL" or row[1] not in PRODUCTS):
            raise ValueError("UNSUPPORTED: only explicit product SELL entries")
        quantity = _integer(row[2], "requested units", maximum=100_000)
        result.append((row[1], quantity))
    if len({item for item, _ in result}) != len(result):
        raise ValueError("UNSUPPORTED: duplicate product would couple assignments")
    return tuple(result)


def wire(queue: Sequence[Order]) -> list[list]:
    return [["SELL", item, quantity] for item, quantity in queue]


def stock(raw: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(raw, Mapping) or set(raw) - set(PRODUCTS):
        raise ValueError("UNSUPPORTED: stock product keys")
    result = {item: _integer(raw.get(item, 0), "stock") for item in PRODUCTS}
    if sum(result.values()) > 100:
        raise ValueError("UNSUPPORTED: default shed capacity exceeded")
    return result


def inventory(raw: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(raw, Mapping) or set(raw) != set(PRODUCTS):
        raise ValueError("UNSUPPORTED: all nine public inventory fields required")
    # The native market may go negative after consumption. Do not clip it.
    if any(isinstance(v, bool) or not isinstance(v, int) for v in raw.values()):
        raise ValueError("UNSUPPORTED: market inventory must be integral")
    return dict(raw)


def item_effect(item: str, start: int, own_units: int, other_units: int,
                own_slot: int, other_slot: int) -> tuple[int, int, int]:
    """Cash in dollars and final inventory in units; quantities already clipped.

    Slot equality quotes BOTH units before either inventory increment. Price-$1
    sales fill and earn cash but never add market supply. No seat priority term.
    """
    price = load_engine().market_price
    cash = [0, 0]
    inv = start
    if own_slot == other_slot:
        groups = [(own_units, other_units)]
    elif own_slot < other_slot:
        groups = [(own_units, 0), (0, other_units)]
    else:
        groups = [(0, other_units), (own_units, 0)]
    for a, b in groups:
        for unit in range(max(a, b)):
            quote = price(item, inv)
            active = (unit < a, unit < b)
            for seat in (0, 1):
                cash[seat] += int(active[seat]) * quote
            if quote > 1:
                inv += sum(active)
    return cash[0], cash[1], inv


def outcome(market: Mapping, own_stock: Mapping, other_stock: Mapping,
            own_orders: Sequence, other_orders: Sequence) -> dict:
    market, a, b = inventory(market), stock(own_stock), stock(other_stock)
    own, other = orders(own_orders), orders(other_orders, limit=10)
    left = {p: (i, min(q, a[p])) for i, (p, q) in enumerate(own)}
    right = {p: (i, min(q, b[p])) for i, (p, q) in enumerate(other)}
    own_cash = other_cash = 0
    after, fill_a, fill_b = {}, {}, {}
    for p in PRODUCTS:
        i, qa = left.get(p, (0, 0))
        j, qb = right.get(p, (0, 0))
        ca, cb, inv = item_effect(p, market[p], qa, qb, i, j)
        own_cash += ca
        other_cash += cb
        after[p], fill_a[p], fill_b[p] = inv, qa, qb
    return {"own_cash": own_cash, "other_cash": other_cash,
            "margin": own_cash - other_cash, "inventory": after,
            "own_fills": fill_a, "other_fills": fill_b}


def native_outcome(market: Mapping, own_stock: Mapping, other_stock: Mapping,
                   own_orders: Sequence, other_orders: Sequence, *, seat=0,
                   starting_cash=(3000, 3000)) -> dict:
    """Evaluator-only isolated _process_market call; no initialized world/steps.

    Raw order rows are intentional: tests can expose unsupported mixed actions.
    Never pass its opponent-private arguments to an actor as observations.
    """
    if seat not in (0, 1):
        raise ValueError("seat must be zero or one")
    engine = load_engine()
    public = engine._new_market()
    public["inventory"] = inventory(market)
    engine._refresh_prices(public)
    farms = [engine._new_farm(10, starting_cash[0]),
             engine._new_farm(10, starting_cash[1])]
    privates = [engine._new_private(), engine._new_private()]
    privates[seat]["shed"].update(stock(own_stock))
    privates[1-seat]["shed"].update(stock(other_stock))
    before = [dict(p["shed"]) for p in privates]
    queues = [None, None]
    queues[seat], queues[1-seat] = own_orders, other_orders
    state = [AttrDict(observation=AttrDict(market=public, farms=farms,
                                         private=privates[s]),
                      action={"market": list(queues[s])}) for s in (0, 1)]
    env = AttrDict(configuration=AttrDict(DEFAULT_CONFIGURATION))
    engine._process_market(state, env)
    cash = [int(farms[s]["money"] - starting_cash[s]) for s in (0, 1)]
    return {"own_cash": cash[seat], "other_cash": cash[1-seat],
            "margin": cash[seat] - cash[1-seat],
            "inventory": public["inventory"],
            "own_fills": {p: before[seat][p] - privates[seat]["shed"][p] for p in PRODUCTS},
            "other_fills": {p: before[1-seat][p] - privates[1-seat]["shed"][p] for p in PRODUCTS}}


def assignment(matrix: Sequence[Sequence[int]]) -> tuple[int, tuple[int, ...], int]:
    """Maximize square product-by-slot matrix; ties keep lexicographic input order."""
    n = len(matrix)
    if n > 9 or any(len(row) != n for row in matrix):
        raise ValueError("bounded square matrix of size <=9 required")
    best = {0: (0, ())}
    edges = 0
    for mask in range(1 << n):
        value, path = best[mask]
        slot = mask.bit_count()
        for item in range(n):
            if mask & (1 << item):
                continue
            edges += 1
            key = mask | (1 << item)
            candidate = (value + matrix[item][slot], path + (item,))
            old = best.get(key)
            if old is None or candidate[0] > old[0] or (candidate[0] == old[0] and candidate[1] < old[1]):
                best[key] = candidate
    value, path = best[(1 << n) - 1]
    return value, path, edges


def solve(market: Mapping, own_stock: Mapping, other_stock: Mapping,
          own_orders: Sequence, other_orders: Sequence, *, objective="margin") -> dict:
    if objective not in ("own_cash", "margin"):
        raise ValueError("objective must be explicit own_cash or margin, not WDL")
    market, a, b = inventory(market), stock(own_stock), stock(other_stock)
    own, other = orders(own_orders), orders(other_orders, limit=10)
    right = {p: (i, min(q, b[p])) for i, (p, q) in enumerate(other)}
    matrix = []
    for p, requested in own:
        j, qb = right.get(p, (0, 0))
        row = []
        for slot in range(len(own)):
            ca, cb, _ = item_effect(p, market[p], min(requested, a[p]), qb, slot, j)
            row.append(ca if objective == "own_cash" else ca - cb)
        matrix.append(row)
    value, indices, edges = assignment(matrix)
    selected = wire([own[i] for i in indices])
    base = outcome(market, a, b, own_orders, other_orders)
    changed = outcome(market, a, b, selected, other_orders)
    constant = changed[objective] - value  # Opponent-only products: fixed, not zeroed.
    row_bound = sum(max(row) - row[i] for i, row in enumerate(matrix))
    return {"schema": SCHEMA, "objective": objective, "orders": selected,
            "baseline": base, "selected": changed,
            "oracle_gap": changed[objective] - base[objective],
            "row_relaxation_bound": row_bound, "matrix": matrix,
            "constant": constant, "matrix_cells": len(own)**2,
            "dp_edges": edges, "raw_permutations": math.factorial(len(own)),
            "scope": "CONDITIONAL_IMMEDIATE_MARKET_NOT_TERMINAL_VALUE"}


def public_signature(result: dict) -> tuple:
    """Only public cash deltas and public inventory; no opponent shed or orders."""
    return (result["own_cash"], result["other_cash"],
            tuple(result["inventory"][p] for p in PRODUCTS))


@dataclass(frozen=True)
class Hypothesis:
    """Assumed recurring requested order AND market-phase available stock."""
    label: str
    queue: tuple[Order, ...]
    available: tuple[int, ...]

    @classmethod
    def make(cls, label: str, queue: Sequence, available: Mapping) -> "Hypothesis":
        checked = stock(available)
        return cls(label, orders(queue, limit=10), tuple(checked[p] for p in PRODUCTS))

    def predict(self, market, own_stock, own_orders):
        return outcome(market, own_stock, dict(zip(PRODUCTS, self.available)),
                       own_orders, wire(self.queue))


class OrderBelief:
    """Finite version space, cold-start/ambiguity KEEP, detectable-change reset.

    Information is learned AFTER each decision. Singleton exploitation can lose
    on the first changed round; no guarantee holds outside the declared support.
    The fixed fallback is not a dominance guarantee against an arbitrary opponent.
    """
    def __init__(self, hypotheses: Sequence[Hypothesis]):
        if not 1 <= len(hypotheses) <= 720:
            raise ValueError("explicit support of 1..720 hypotheses required")
        if len({h.label for h in hypotheses}) != len(hypotheses):
            raise ValueError("unique labels required")
        self.pool = tuple(hypotheses)
        self.active = self.pool
        self.violations = 0
        self.observations = 0
        self.supported = True

    def decide(self, market, own_stock, own_orders, *, objective="margin") -> dict:
        orders(own_orders)
        if objective not in ("own_cash", "margin"):
            raise ValueError("invalid objective")
        mode = "COLD_START_KEEP" if not self.observations else "AMBIGUOUS_KEEP"
        selected = [list(o) for o in own_orders]
        point = None
        if not self.supported:
            mode = "OUTSIDE_SUPPORT_KEEP"
        if self.supported and self.observations and len(self.active) == 1:
            h = self.active[0]
            selected = solve(market, own_stock, dict(zip(PRODUCTS, h.available)),
                             own_orders, wire(h.queue), objective=objective)["orders"]
            point, mode = h.label, "IDENTIFIED_CONDITIONAL_ASSIGNMENT"
        predictions = [h.predict(market, own_stock, selected)[objective] for h in self.active]
        return {"orders": selected, "mode": mode, "point_forecast": point,
                "support_size": len(self.active),
                "score_range": [min(predictions), max(predictions)],
                "range_kind": "FINITE_ASSUMED_SUPPORT_NOT_CONFIDENCE_INTERVAL"}

    def observe(self, market_before, own_stock_before, chosen_orders, observed_signature) -> dict:
        def matches(pool):
            return tuple(h for h in pool if public_signature(
                h.predict(market_before, own_stock_before, chosen_orders)) == observed_signature)
        survivors = matches(self.active)
        violation = not survivors
        if violation:
            self.violations += 1
            survivors = matches(self.pool)
        outside = not survivors
        self.supported = not outside
        self.active = survivors or self.pool
        self.observations += 1
        return {"support_size": len(self.active), "detected_change_or_model_error": violation,
                "outside_declared_support": outside, "violations": self.violations}


# Mixed queues reuse the native market, not a second economic simulator. The only
# intervention is a permutation of existing SELL entries among existing SELL slots.
PHASE_KEYS = {"market", "farms", "privates", "queues", "configuration"}


def _phase_input(snapshot: Mapping) -> dict:
    """Validate a market-phase snapshot, AFTER workers and BEFORE town consumption.

    Both private states and both current queues are evaluator-only. Never obtain
    this snapshot by pretending an actor's pre-worker observation is post-worker.
    """
    import copy
    if not isinstance(snapshot, Mapping) or set(snapshot) != PHASE_KEYS:
        raise ValueError("UNSUPPORTED: exact native market-phase fields required")
    snap = copy.deepcopy(dict(snapshot))
    engine = load_engine()
    config = snap["configuration"]
    for key in ("boardSize", "shedCapacity", "maxMarketOrdersPerTurn", "farmHandCostMult"):
        _integer(config.get(key, DEFAULT_CONFIGURATION[key]), key, minimum=1,
                 maximum=100 if key != "farmHandCostMult" else 10000)
    if config.get("boardSize", 10) != 10 or config.get("maxMarketOrdersPerTurn", 10) != 10:
        raise ValueError("UNSUPPORTED: snapshot requires boardSize=10, maxMarketOrdersPerTurn=10")
    inventory(snap["market"]["inventory"])
    if any(len(snap[k]) != 2 for k in ("farms", "privates", "queues")):
        raise ValueError("UNSUPPORTED: exactly two seats required")
    domains = {"SELL": PRODUCTS, "BUY_PRODUCT": ("WHEAT", "FERTILIZER"),
               "BUY_SEED": engine.CROPS, "BUY_ANIMAL": engine.ANIMALS}
    for seat in (0, 1):
        farm, private, queue = snap["farms"][seat], snap["privates"][seat], snap["queues"][seat]
        if not isinstance(queue, list) or len(queue) > 10:
            raise ValueError("UNSUPPORTED: queues must contain at most ten entries")
        if (isinstance(farm["money"], bool) or not isinstance(farm["money"], (int, float))
                or not math.isfinite(farm["money"]) or farm["money"] < 0):
            raise ValueError("UNSUPPORTED: nonnegative finite cash required")
        _integer(farm["hires_today"], "hires_today", maximum=100)
        if not 1 <= len(farm["unlocked_quadrants"]) <= 4:
            raise ValueError("UNSUPPORTED: quadrant count")
        for name, domain in (("shed", (*PRODUCTS, *engine.ANIMALS)), ("seeds", engine.CROPS)):
            if set(private[name]) != set(domain):
                raise ValueError("UNSUPPORTED: complete native private " + name + " required")
            for amount in private[name].values():
                _integer(amount, name + " units")
        if sum(private["shed"].values()) > config.get("shedCapacity", 100):
            raise ValueError("UNSUPPORTED: initial shed exceeds capacity")
        for row in queue:
            if not isinstance(row, list):
                raise ValueError("UNSUPPORTED: explicit native order rows required")
            if not row:
                continue  # Native empty-row no-op occupies its original queue index.
            if row[0] in ("HIRE", "BUY_LAND") and len(row) == 1:
                continue
            if len(row) != 3 or row[0] not in domains or row[1] not in domains[row[0]]:
                raise ValueError("UNSUPPORTED: market order")
            _integer(row[2], "requested units", maximum=10000)
    return snap



def capture_market_phase(state, env) -> dict:
    """Evaluator hook: call at entry to native _process_market, never before workers.

    Does not execute a step or market call. Includes opponent-private truth for
    oracle checks; the returned object must never be sent to a policy predictor.
    """
    return _phase_input({
        "market": state[0].observation.market,
        "farms": state[0].observation.farms,
        "privates": [row.observation.private for row in state],
        "queues": [row.action.get("market", []) for row in state],
        "configuration": {key: env.configuration.get(key, value)
                          for key, value in DEFAULT_CONFIGURATION.items()},
    })


def native_phase(snapshot: Mapping, *, split_slots=True) -> dict:
    """Resolve only the market on a copy; preserve every zero-fill/duplicate slot.

    Split mode records actual fills per original queue entry. Calling the native
    market once per queue index is identical to its outer loop: no other engine
    phase occurs between calls. Both quotes and commit order remain native.
    Unsplit mode is the independent full-call parity control (fills=None).
    """
    snap = _phase_input(snapshot)
    engine = load_engine()
    farms, privates, market = snap["farms"], snap["privates"], snap["market"]
    initial_cash = [f["money"] for f in farms]
    states = [AttrDict(observation=AttrDict(farms=farms, private=privates[i], market=market),
                       action={"market": []}) for i in (0, 1)]
    env = AttrDict(configuration=AttrDict({**DEFAULT_CONFIGURATION, **snap["configuration"]}))
    fills = [[], []] if split_slots else None
    calls = 0
    if split_slots:
        for index in range(max(map(len, snap["queues"]), default=0)):
            before = []
            rows = [q[index] if index < len(q) else None for q in snap["queues"]]
            for i, row in enumerate(rows):
                if not row:
                    before.append(0)
                elif row[0] == "HIRE":
                    before.append(farms[i]["hires_today"])
                elif row[0] == "BUY_LAND":
                    before.append(len(farms[i]["unlocked_quadrants"]))
                else:
                    field = "seeds" if row[0] == "BUY_SEED" else "shed"
                    before.append(privates[i][field][row[1]])
                states[i].action = {"market": [] if row is None else [row]}
            engine._process_market(states, env)
            calls += 1
            for i, row in enumerate(rows):
                if row is None:
                    continue
                if not row:
                    fills[i].append(0)
                    continue
                if row[0] == "HIRE":
                    amount = farms[i]["hires_today"] - before[i]
                elif row[0] == "BUY_LAND":
                    amount = len(farms[i]["unlocked_quadrants"]) - before[i]
                else:
                    field = "seeds" if row[0] == "BUY_SEED" else "shed"
                    amount = privates[i][field][row[1]] - before[i]
                    if row[0] == "SELL":
                        amount = -amount
                fills[i].append(amount)
    else:
        for i in (0, 1):
            states[i].action = {"market": snap["queues"][i]}
        engine._process_market(states, env)
        calls = 1
    return {"market": market, "farms": farms, "privates": privates,
            "cash_delta": [farms[i]["money"] - initial_cash[i] for i in (0, 1)],
            "fills": fills, "native_market_calls": calls}


def mixed_certificate(snapshot: Mapping) -> dict:
    """Sufficient, not necessary, separation conditions; dollars and shed units.

    All buys must be fundable with pre-phase cash and fit without any sales.
    No product sold by either seat may be BUY_PRODUCT-bought by either seat.
    Then non-SELL outcomes are fixed and product-slot assignment remains exact.
    This deliberately rejects many safe states rather than inventing a proof.
    """
    snap = _phase_input(snapshot)
    e, config = load_engine(), snap["configuration"]
    queues = snap["queues"]
    sold = {r[1] for q in queues for r in q if r and r[0] == "SELL"}
    bought = {r[1] for q in queues for r in q if r and r[0] == "BUY_PRODUCT"}
    blockers, funding, capacity = [], [], []
    if snap["market"].get("params", e.MARKET_PARAMS) != e.MARKET_PARAMS:
        blockers.append("CUSTOM_PRICE_CURVES")
    if sold & bought:
        blockers.append("SAME_PRODUCT_BUY_SELL")
    total_buys = {p: sum(r[2] for q in queues for r in q
                        if r and r[0] == "BUY_PRODUCT" and r[1] == p) for p in bought}
    for seat, q in enumerate(queues):
        sales = [r[1] for r in q if r and r[0] == "SELL"]
        if len(set(sales)) != len(sales):
            blockers.append(f"DUPLICATE_SELL_SEAT{seat}")
        hires = snap["farms"][seat]["hires_today"]
        land = len(snap["farms"][seat]["unlocked_quadrants"]) - 1
        spend = incoming = 0
        for row in q:
            if not row:
                continue
            op = row[0]
            if op == "HIRE":
                spend += e._hire_cost(hires, config.get("farmHandCostMult", 1))
                hires += 1
            elif op == "BUY_LAND":
                if land < len(e.LAND_PRICES):
                    spend += e.LAND_PRICES[land]
                    land += 1
            elif op in ("BUY_PRODUCT", "BUY_ANIMAL", "BUY_SEED"):
                item, quantity = row[1:]
                if op == "BUY_PRODUCT":
                    # Native default prices are monotone nonincreasing in inventory.
                    # Ignore all sales and debit every requested buy by both seats.
                    quote = e.market_price(item, snap["market"]["inventory"][item] - total_buys[item])
                else:
                    quote = e.ANIMALS[item]["cost"] if op == "BUY_ANIMAL" else e.CROPS[item]["seed"]
                spend += quantity * quote
                incoming += quantity if op != "BUY_SEED" else 0
        funding.append(spend)
        used = sum(snap["privates"][seat]["shed"].values()) + incoming
        capacity.append(used)
        if snap["farms"][seat]["money"] < spend:
            blockers.append(f"CASH_DEPENDENT_SEAT{seat}")
        if used > config.get("shedCapacity", 100):
            blockers.append(f"CAPACITY_DEPENDENT_SEAT{seat}")
    return {"certified": not blockers, "blockers": blockers,
            "cash_upper_bound_dollars_by_seat": funding,
            "shed_upper_bound_units_by_seat": capacity,
            "meaning": "SUFFICIENT_SEPARATION_ONLY; failure is not infeasibility"}


def sale_slot_equivalence(snapshot: Mapping, *, seat: int) -> dict:
    """Sufficient exact no-op certificate, not merely a zero margin bound.

    Consumes an evaluator-only post-worker market snapshot. With the existing
    mixed separation certificate, compare (own dollars, other dollars, final
    inventory units) for each sold product in every permitted SELL slot. If each
    row is constant, ALL permutations have the same market, farms and private
    end states. Pinned purchases/hires and original-entry fills remain unchanged.
    This removes queue order only at this fixed phase, not timing/production.

    Failure is UNKNOWN or POSSIBLY_ORDER_SENSITIVE, never an inferred positive
    gap. A nonconstant row can still cancel against another under assignment.
    No native market calls, initialization, actor inputs or production changes.
    """
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError("seat must be zero or one")
    snap = _phase_input(snapshot)
    own, other = snap["queues"][seat], snap["queues"][1-seat]
    slots = tuple(i for i, row in enumerate(own) if row and row[0] == "SELL")
    result = {"schema": "farm-sale-slot-equivalence-v1", "seat": seat,
              "sell_slots": list(slots), "raw_permutations": math.factorial(len(slots)),
              "native_market_calls": 0, "physical_games": 0,
              "scope": "EVALUATOR_ONLY_FIXED_MARKET_PHASE"}
    if len(slots) < 2:
        return {**result, "status": "EXACT_NO_REORDER", "product_slot_evaluations": 0,
                "sensitive_products": []}
    certificate = mixed_certificate(snap)
    if not certificate["certified"]:
        return {**result, "status": "UNKNOWN_COUPLING", "certificate": certificate,
                "product_slot_evaluations": 0, "sensitive_products": None}
    right = {r[1]: (i, min(r[2], snap["privates"][1-seat]["shed"][r[1]]))
             for i, r in enumerate(other) if r and r[0] == "SELL"}
    sensitive = []
    for original in slots:
        _, item, requested = own[original]
        available = min(requested, snap["privates"][seat]["shed"][item])
        j, amount = right.get(item, (0, 0))
        effects = {item_effect(item, snap["market"]["inventory"][item],
                               available, amount, i, j) for i in slots}
        if len(effects) > 1:
            sensitive.append(item)
    return {**result, "certificate": certificate,
            "status": "POSSIBLY_ORDER_SENSITIVE" if sensitive else "EXACT_MARKET_EQUIVALENCE",
            "product_slot_evaluations": len(slots)**2,
            "sensitive_products": sensitive}


def solve_sale_slots(snapshot: Mapping, *, seat: int, objective="margin",
                     max_permutations=120) -> dict:
    """Exact current-phase oracle preserving own realized commitments, or UNKNOWN.

    Certified mixed phases use existing assignment. Otherwise enumerate <= budget
    permutations of SELL occurrence IDs ONLY; purchases and every other slot stay
    pinned. Reject candidates changing own per-entry fills OR noncash end state.
    Always report unconstrained cash/margin temptation separately. No plan emits.
    """
    import copy
    import hashlib
    from itertools import permutations
    from farm_neat.evaluation.common import canonical
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError("seat must be zero or one")
    if objective not in ("own_cash", "margin"):
        raise ValueError("objective must be own_cash or margin, not WDL")
    _integer(max_permutations, "max_permutations", minimum=1, maximum=720)
    snap = _phase_input(snapshot)
    own, other = snap["queues"][seat], snap["queues"][1-seat]
    slots = tuple(i for i, row in enumerate(own) if row and row[0] == "SELL")
    total = math.factorial(len(slots))
    certificate = mixed_certificate(snap)
    header = {"schema": "farm-sale-slot-oracle-v1", "objective": objective, "seat": seat,
              "sell_slots": list(slots), "raw_permutations": total,
              "certificate": certificate, "scope": "EVALUATOR_ONLY_IMMEDIATE_MARKET_NOT_WDL",
              "physical_games": 0, "actor_calls": 0}
    if not certificate["certified"] and total > max_permutations:
        return {**header, "status": "UNKNOWN_BUDGET", "oracle_gap": None,
                "native_market_calls": 0, "candidate_evaluations": 0,
                "reason": "No candidate executed; not an infeasibility or zero-gap result"}
    def score(result):
        a, b = result["cash_delta"][seat], result["cash_delta"][1-seat]
        return a if objective == "own_cash" else a-b
    def resources(result, who):
        return ({k: v for k, v in result["farms"][who].items() if k != "money"},
                result["privates"][who])
    baseline = native_phase(snap)
    calls, evaluated, rejected = baseline["native_market_calls"], 0, 0
    best = unconstrained = None
    digest = hashlib.sha256()
    matrix = []
    if certificate["certified"]:
        right = {r[1]: (i, min(r[2], snap["privates"][1-seat]["shed"][r[1]]))
                 for i, r in enumerate(other) if r and r[0] == "SELL"}
        for original in slots:
            _, item, quantity = own[original]
            quantity = min(quantity, snap["privates"][seat]["shed"][item])
            j, qb = right.get(item, (0, 0))
            row = []
            for slot in slots:
                ca, cb, _ = item_effect(item, snap["market"]["inventory"][item], quantity, qb, slot, j)
                row.append(ca if objective == "own_cash" else ca-cb)
            matrix.append(row)
        value, indices, edges = assignment(matrix)
        chosen = tuple(slots[i] for i in indices)
        candidates = [slots] if chosen == slots else [slots, chosen]
        predicted_gap = value - sum(matrix[i][i] for i in range(len(slots)))
        header.update(matrix_cells=len(slots)**2, dp_edges=edges,
                      search_method="CERTIFIED_PRODUCT_SLOT_ASSIGNMENT")
    else:
        candidates = permutations(slots)
        predicted_gap = None
        header["search_method"] = "BOUNDED_NATIVE_SELL_SLOT_ENUMERATION"
    for perm in candidates:
        changed = copy.deepcopy(snap)
        origin = list(range(len(own)))
        for destination, original in zip(slots, perm):
            changed["queues"][seat][destination] = copy.deepcopy(own[original])
            origin[destination] = original
        value = baseline if perm == slots else native_phase(changed)
        if perm != slots:
            calls += value["native_market_calls"]
        restored_fills = [0] * len(own)
        for i, original in enumerate(origin):
            restored_fills[original] = value["fills"][seat][i]
        preserved = (restored_fills == baseline["fills"][seat] and
                     resources(value, seat) == resources(baseline, seat))
        row = {"sell_occurrence_order": list(perm), "orders": changed["queues"][seat],
               "cash_delta_by_seat": value["cash_delta"], "score": score(value),
               "fills_by_original_entry": restored_fills,
               "other_fills": value["fills"][1-seat], "own_realization_preserved": preserved,
               "other_realization_preserved": (value["fills"][1-seat] == baseline["fills"][1-seat]
                    and resources(value, 1-seat) == resources(baseline, 1-seat))}
        evaluated += 1
        rejected += not preserved
        digest.update((canonical(row) + "\n").encode())
        if unconstrained is None or row["score"] > unconstrained["score"]:
            unconstrained = row
        if preserved and (best is None or row["score"] > best["score"]):
            best = row
        if certificate["certified"] and not (preserved and row["other_realization_preserved"]):
            raise AssertionError("certificate failed native realization control")
    gap = best["score"] - score(baseline)
    if predicted_gap is not None and gap != predicted_gap:
        raise AssertionError("assignment/native cash mismatch")
    return {**header, "status": "EXACT_WITHIN_DECLARED_INTERVENTION",
            "baseline_cash_delta_by_seat": baseline["cash_delta"],
            "baseline_fills_by_seat": baseline["fills"],
            "oracle_gap": gap, "unconstrained_gap": unconstrained["score"] - score(baseline),
            "selected": best, "unconstrained": unconstrained,
            "candidate_evaluations": evaluated, "preservation_rejections": rejected,
            "native_market_calls": calls, "candidate_rows_sha256": digest.hexdigest(),
            "decision": "KEEP" if gap == 0 else "POSITIVE_ORACLE_GAP_NOT_LIVE_PERMISSION"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objective", choices=("margin", "own_cash"), required=True)
    parser.add_argument("--mode", choices=("sell-only", "sell-slots"), default="sell-only")
    parser.add_argument("--seat", type=int, choices=(0, 1))
    parser.add_argument("--max-permutations", type=int, default=120)
    args = parser.parse_args()
    given = read(args.input, binding=False)
    if args.mode == "sell-slots":
        if args.seat is None:
            parser.error("--seat is required for a native snapshot")
        result = solve_sale_slots(given, seat=args.seat, objective=args.objective,
                                  max_permutations=args.max_permutations)
        write_new(args.out, result)
        print(f"{result['status']}: {result['oracle_gap']} dollars; no games run.")
        return
    required = {"market", "own_stock", "other_stock", "own_orders", "other_orders"}
    if set(given) != required:
        raise ValueError("exact market, both stocks and both queues required; evaluator-only input")
    result = solve(**given, objective=args.objective)
    write_new(args.out, {**result, "new_games": 0, "optimizer_updates": 0,
                         "actor_calls": 0, "native_component_calls": 0})
    print(f"Conditional {args.objective} gap: {result['oracle_gap']} dollars; no games run.")


if __name__ == "__main__":
    main()
