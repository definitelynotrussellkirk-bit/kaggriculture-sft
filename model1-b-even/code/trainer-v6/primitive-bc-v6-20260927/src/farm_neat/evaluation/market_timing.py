"""Bounded, evaluator-only sale-window screen around a FIXED flow schedule.

Not an actor, full-game simulator, automatic liquidation or production planner.
One explicitly authorized tranche changes time, never queue slot or total quantity.
Existing native market code owns quotations/commits; the deposit quotient owns
ordered custody counts; the cash-prefix reducer owns realized liquidity checks.

Conditional event-window identity: in a maximal run of event-free market turns,
move a single tranche to the first turn. Nothing else changes cash, custody,
market inventory, constraints or time value there, so state at the next event is
identical. Keep event turns AND their successors: consumption/night deposits are
AFTER market, while explicit deposits are BEFORE it. Dense events give no gain.
This removes empty time labels, NOT uncertainty, competing orders or logistics.

Default configuration only; <=48 turns, <=16 explicitly declared future scenarios.
Fixed BUYs are replayed, and missing fills disqualify a purported plan-preserving
improvement. Future deliveries/opponent orders are labels or explicit support,
never observed facts. Fixed-flow forecasts cease to be valid if funding/service
changes their production; return that dependency to the native plan evaluator.
This module temporarily instruments _commit_unit and is for serial offline use.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from pathlib import Path

from farm_neat.economy.heuristic.market import sell_revenue
from farm_neat.evaluation.common import file_sha, read, write_new
from farm_neat.evaluation.inventory_quotient import deposit_quotient
from farm_neat.evaluation.sale_order import inventory, _integer
from farm_neat.evaluation.service_timeline import cash_prefix_audit
from farm_neat.game.exact_engine import AttrDict, DEFAULT_CONFIGURATION, load_engine

SCHEMA = "farm-market-windows-v1"


def checked(raw: dict) -> dict:
    """Validate the CLOSED fixture/support schema; unsupported is never feasible."""
    if not isinstance(raw, dict):
        raise ValueError("case must be an object")
    required = {"first", "last", "keep", "product", "quantity", "slot", "market",
                "cash", "shed", "shops", "events", "visibility"}
    if set(raw) - required - {"reserve"} or required - set(raw):
        raise ValueError("exact market-window fields required")
    c = copy.deepcopy(raw)
    e = load_engine()
    for k in ("first", "last", "keep"):
        _integer(c[k], k, maximum=718)
    if not c["first"] <= c["keep"] <= c["last"] or c["last"] - c["first"] >= 48:
        raise ValueError("ordered horizon <=48 market turns required")
    if c["visibility"] not in ("SYNTHETIC_EVALUATOR_ONLY", "DECLARED_FUTURE_SUPPORT"):
        raise ValueError("future flow must be labelled, never asserted actor-visible")
    if c["product"] not in e.PRODUCTS:
        raise ValueError("authorized product required")
    _integer(c["quantity"], "authorized quantity", minimum=1, maximum=100)
    _integer(c["slot"], "fixed slot", maximum=9)
    c["market"] = inventory(c["market"])
    if (not isinstance(c["shops"], list) or len(c["shops"]) > 8 or
            any(s not in e.SHOPS for s in c["shops"])):
        raise ValueError("explicit current shops, including duplicates, required")
    if len(c["shops"]) < 8 and c["last"] >= (c["first"] // 72 + 1) * 72:
        raise ValueError("UNKNOWN future shop unlock; shorten this constant-shop horizon")
    c.setdefault("reserve", [0, 0])
    for key in ("cash", "shed", "reserve"):
        if not isinstance(c[key], list) or len(c[key]) != 2:
            raise ValueError("actor-relative two-seat " + key + " required")
    for k in ("cash", "reserve"):
        for v in c[k]:
            _integer(v, k)
    allowed = set(e.PRODUCTS) | set(e.ANIMALS)
    for stock in c["shed"]:
        if not isinstance(stock, dict) or set(stock) - allowed:
            raise ValueError("unsupported shed keys")
        for n in stock.values():
            _integer(n, "shed quantity")
        if sum(stock.values()) > 100:
            raise ValueError("default shed capacity exceeded")
    if not isinstance(c["events"], list) or len(c["events"]) > 48:
        raise ValueError("bounded explicit event list required")
    seen = set()
    for ev in c["events"]:
        if not isinstance(ev, dict) or set(ev) - {"step", "queues", "before", "after"}:
            raise ValueError("unsupported event fields")
        t = _integer(ev.get("step"), "event step")
        if t in seen or not c["first"] <= t <= c["last"]:
            raise ValueError("unique event steps within horizon required")
        seen.add(t)
        qs = ev.setdefault("queues", [[], []])
        if not isinstance(qs, list) or len(qs) != 2:
            raise ValueError("two fixed queues required")
        for role, q in enumerate(qs):
            if not isinstance(q, list) or len(q) > 10:
                raise ValueError("at most ten fixed queue slots")
            if role == 0 and len(q) > c["slot"] and q[c["slot"]]:
                raise ValueError("target slot must be empty; do not reorder fixed commitments")
            for row in q:
                if row == []:
                    continue  # Preserve a zero-fill slot, never compact queues.
                if not isinstance(row, list) or len(row) != 3:
                    raise ValueError("only unit transactions and empty slots supported")
                op, item, n = row
                domain = {"SELL": e.PRODUCTS, "BUY_PRODUCT": ("WHEAT", "FERTILIZER"),
                          "BUY_SEED": e.CROPS, "BUY_ANIMAL": e.ANIMALS}
                if op not in domain or item not in domain[op]:
                    raise ValueError("unsupported fixed transaction")
                _integer(n, "fixed requested quantity", maximum=100)
                if role == 0 and op == "SELL" and item == c["product"]:
                    raise ValueError("target must be one isolated authorized tranche")
        for phase in ("before", "after"):
            pair = ev.setdefault(phase, [[], []])
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("two ordered cargo lists required")
            if phase == "after" and any(pair) and t % 24 != 23:
                raise ValueError("after-market cargo is nightly only; retain native phase")
            for cargo in pair:
                if not isinstance(cargo, list) or len(cargo) > 40:
                    raise ValueError("bounded worker cargo required")
                # Validates stack insertion order, duplicate keys and integer counts.
                result = deposit_quotient({}, cargo)
                if set(result["carried_by_item"]) - allowed or sum(result["carried_by_item"].values()) > 200:
                    raise ValueError("unsupported or oversized cargo")
    c["events"].sort(key=lambda row: row["step"])
    return c


def demand(shops: list[str], step: int) -> dict:
    """Known current-shop demand ONLY; no seed or future shop sampling."""
    e, result = load_engine(), Counter()
    if step % 4 == 0:
        for shop in shops:
            products = e.SHOPS[shop]
            for p in products:
                result[p] += 2 if len(products) == 1 else 1
    if step % 24 == 0:
        result.update(e.TOWN_CENTER_PRODUCTS)
    return dict(result)


def _events(c: dict) -> set[int]:
    events = {ev["step"] for ev in c["events"]
              if any(any(row for row in q) for q in ev["queues"])
              or any(ev["before"]) or any(ev["after"])}
    events.update(t for t in range(c["first"], c["last"] + 1) if demand(c["shops"], t))
    return events


def event_windows(raw: dict) -> list[int]:
    c = checked(raw)
    events = _events(c)
    return sorted({c["first"], c["keep"], c["last"]} |
                  {t for x in events for t in (x, x + 1) if c["first"] <= t <= c["last"]})


def price_bounds(product: str, start_inventory: int, quantity: int,
                 net_flow_low: int, net_flow_high: int) -> dict:
    """Isolated-sale revenue bounds, dollars; net = above-$1 sales - buys - demand.

    Excludes concurrent same-product quotes, stock shortage, cash/storage needs,
    other-sale effects and opponent revenue. This is NOT a competitive bound.
    """
    if product not in load_engine().PRODUCTS:
        raise ValueError("unknown product")
    if any(type(v) is not int for v in (start_inventory, net_flow_low, net_flow_high)):
        raise ValueError("integral inventories/flows required")
    _integer(quantity, "quantity", maximum=100)
    if net_flow_low > net_flow_high:
        raise ValueError("ordered net-flow bounds required")
    now = sell_revenue(product, start_inventory, quantity)
    lo = sell_revenue(product, start_inventory + net_flow_high, quantity)
    hi = sell_revenue(product, start_inventory + net_flow_low, quantity)
    return {"now": now, "wait_min": lo, "wait_max": hi,
            "own_revenue_delta_range": [lo - now, hi - now],
            "scope": "CONDITIONAL_ISOLATED_QUOTE_NOT_PLAN_OR_MARGIN"}


def replay(raw: dict, sale_step: int, *, seat: int = 0, tranches: list | None = None) -> dict:
    """Replay native market COMPONENTS, never initialize or advance a full game.

    All inputs are actor-relative. tranches is an explicit fixed-total split for
    diagnostic frontier checks; event-window identity applies to one tranche.
    """
    c, e = checked(raw), load_engine()
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")
    pieces = [(sale_step, c["quantity"])] if tranches is None else tranches
    schedule = {}
    for t, n in pieces:
        _integer(t, "sale step")
        _integer(n, "tranche quantity", maximum=100)
        if t in schedule or not c["first"] <= t <= c["last"]:
            raise ValueError("distinct sale windows within horizon required")
        schedule[t] = n
    if sum(schedule.values()) != c["quantity"]:
        raise ValueError("explicit authorized total must be preserved")
    physical = [seat, 1 - seat]
    farms = [None, None]
    private = [None, None]
    for r, s in enumerate(physical):
        farms[s] = e._new_farm(10, c["cash"][r])
        private[s] = e._new_private()
        private[s]["shed"].update(c["shed"][r])
    public = e._new_market()
    public["inventory"] = dict(c["market"])
    e._refresh_prices(public)
    state = [AttrDict(observation=AttrDict(market=public, farms=farms,
                    private=private[s], town={"unlocked_shops": c["shops"]}), action={}) for s in (0, 1)]
    env = AttrDict(configuration=AttrDict(DEFAULT_CONFIGURATION))
    events = {ev["step"]: ev for ev in c["events"]}
    fills, failed, losses = [Counter(), Counter()], [Counter(), Counter()], [Counter(), Counter()]
    cash_deltas, target_fills = [[], []], 0
    counts = {"native_market_phases": 0, "native_deposit_checks": 0, "native_consumption_phases": 0}
    original = e._commit_unit

    def commit(op, item, price, farm, priv, market, capacity=100):
        nonlocal target_fills
        role = 0 if priv is private[seat] else 1
        before = farm["money"]
        ok = original(op, item, price, farm, priv, market, capacity)
        cash_deltas[role].append(farm["money"] - before)
        if role == 0 and op == "SELL" and item == c["product"]:
            target_fills += int(ok)
        else:
            (fills if ok else failed)[role][(t, op, item)] += 1
        return ok

    def deposit(pair):
        for role, cargo in enumerate(pair):
            if not cargo:
                continue
            priv = private[physical[role]]
            summary = deposit_quotient(priv["shed"], cargo)
            priv["inventories"] = [dict(stacks) for stacks in cargo]
            e._drop_inventories_to_shed(priv, 100)
            counts["native_deposit_checks"] += 1
            if priv["shed"] != summary["shed_after"]:
                raise AssertionError("ordered deposit projection/native mismatch")
            losses[role].update(summary["destroyed_by_item"])
            priv["inventories"] = [{}]

    # No event-free state changes in this declared model; skip those turns.
    e._commit_unit = commit
    try:
        for t in sorted(_events(c) | set(schedule)):
            ev = events.get(t, {"queues": [[], []], "before": [[], []], "after": [[], []]})
            deposit(ev["before"])
            qs = copy.deepcopy(ev["queues"])
            if schedule.get(t, 0):
                qs[0] += [[] for _ in range(c["slot"] + 1 - len(qs[0]))]
                qs[0][c["slot"]] = ["SELL", c["product"], schedule[t]]
            for role, s in enumerate(physical):
                state[s].action = {"market": qs[role]}
            if any(any(row for row in q) for q in qs):
                e._process_market(state, env)
                counts["native_market_phases"] += 1
            if demand(c["shops"], t):
                e._town_consume(env, state, t)
                counts["native_consumption_phases"] += 1
            deposit(ev["after"])
    finally:
        e._commit_unit = original
    cash = [farms[s]["money"] for s in physical]
    rank = "WIN" if cash[0] > cash[1] else "LOSS" if cash[0] < cash[1] else "DRAW"
    rows = lambda count: [[*k, n] for k, n in sorted(count.items())]
    return {"cash": cash, "margin": cash[0] - cash[1], "cash_order": rank,
            "wdl": rank if c["last"] == 718 else None,
            "wdl_scope": "INJECTED_TERMINAL_FLOW_SUFFIX_NOT_PLAYED_GAME" if c["last"] == 718 else "NONTERMINAL_NO_WDL",
            "market": public["inventory"], "shed": [private[s]["shed"] for s in physical],
            "seeds": [private[s]["seeds"] for s in physical],
            "target_filled": target_fills, "authorized_unfilled": c["quantity"] - target_fills,
            "fixed_fills": [rows(f) for f in fills], "fixed_failures": [rows(f) for f in failed],
            "destroyed": [dict(sorted(x.items())) for x in losses],
            "cash_prefix": [cash_prefix_audit(c["cash"][r], cash_deltas[r], c["reserve"][r]) for r in (0, 1)],
            "counts": counts, "scope": "CONDITIONAL_FIXED_FLOW_NOT_DELIVERY_OR_PRODUCTION_CERTIFICATE"}


def preserved(base: dict, changed: dict) -> bool:
    """Necessary local guards, NOT whole-plan feasibility or matching full state."""
    actual = {tuple(row[:3]): row[3] for row in changed["fixed_fills"][0]}
    return (changed["target_filled"] >= base["target_filled"]
            and all(actual.get(tuple(row[:3]), 0) >= row[3] for row in base["fixed_fills"][0])
            and all(n <= base["destroyed"][0].get(p, 0) for p, n in changed["destroyed"][0].items())
            and changed["cash_prefix"][0]["safe_on_supplied_path"])


def screen(support: list[dict], *, seat: int = 0) -> dict:
    """Finite-support robust screen; KEEP unless all local guards and cash/margin pass.

    No probabilities are inferred, and no recommendation holds outside support.
    Output is a diagnostic candidate, never an executable policy decision.
    """
    if not 1 <= len(support) <= 16:
        raise ValueError("explicit support of 1..16 cases required")
    cases = [checked(c) for c in support]
    common = ("first", "last", "keep", "product", "quantity", "slot")
    if any(any(c[k] != cases[0][k] for k in common) for c in cases):
        raise ValueError("same authorized edit/horizon across support required")
    windows = sorted(set().union(*(event_windows(c) for c in cases)))
    keep = cases[0]["keep"]
    baseline = [replay(c, keep, seat=seat) for c in cases]
    table = []
    totals = Counter()
    for r in baseline:
        totals.update(r["counts"])
    for step in windows:
        outcomes = baseline if step == keep else [replay(c, step, seat=seat) for c in cases]
        if step != keep:
            for r in outcomes:
                totals.update(r["counts"])
        own, other, margin, guards = [], [], [], []
        for b, x in zip(baseline, outcomes):
            own.append(x["cash"][0] - b["cash"][0])
            other.append(x["cash"][1] - b["cash"][1])
            margin.append(x["margin"] - b["margin"])
            guards.append(preserved(b, x))
        eligible = all(guards) and min(own) >= 0 and min(margin) >= 0
        table.append({"step": step, "own_delta": [min(own), max(own)],
                      "opponent_delta": [min(other), max(other)],
                      "margin_delta": [min(margin), max(margin)],
                      "local_guards": guards, "eligible": eligible})
    viable = [r for r in table if r["eligible"] and r["margin_delta"][0] > 0]
    best = max(viable, key=lambda r: (r["margin_delta"][0], -r["step"]), default=None)
    return {"schema": SCHEMA, "keep": keep, "candidate": best["step"] if best else keep,
            "decision": "SCOPED_CANDIDATE_ONLY" if best else "KEEP_NO_ROBUST_MARGIN_GAIN",
            "raw_windows": cases[0]["last"] - cases[0]["first"] + 1,
            "event_windows": windows, "support_size": len(cases), "rows": table,
            "counts": dict(totals), "full_games": 0,
            "limits": ["Ranges are supplied-support extrema, not confidence intervals.",
                       "No terminal WDL, unknown-tail value, real opportunity or actor observability is established.",
                       "Fixed exogenous flows may be invalidated by changed native logistics/funding."]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="JSON list of declared future cases")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seat", type=int, choices=(0, 1), default=0)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("refuse reused output")
    result = screen(read(args.input, binding=False), seat=args.seat)
    write_new(args.out, {"input_sha256": file_sha(args.input), "result": result})
    print(result["decision"], result["raw_windows"], "->", len(result["event_windows"]), "windows; zero games")


if __name__ == "__main__":
    main()
