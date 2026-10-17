"""Small exact MARKET-PHASE fixtures, not initialized worlds or game rollouts.

Calls the existing EconomicOrders market planner and the hash-verified engine's
market function on deliberately injected states. Records attempted/filled units
and ordered own cash changes. This characterizes (does not fix) own-only quote
risk. No evaluator, training, model, Kaggle or random world is initialized.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from farm_neat.economy.orders import EconomicOrders
from farm_neat.evaluation.service_timeline import cash_prefix_audit
from farm_neat.game.exact_engine import AttrDict, assert_bundled_engine_identity, load_engine


def case(*, seat: int = 0, money: int = 2520, reserve: int = 10,
         opponent: str = "BUY", seed_first: bool = False) -> dict:
    if seat not in (0, 1) or opponent not in {"BUY", "SELL", "PASS"}:
        raise ValueError("explicit seat and opponent fixture required")
    engine = load_engine()  # verified module only; not an environment constructor
    market = engine._new_market()
    executor = EconomicOrders()
    executor.apply({"op": "CASH_RESERVE", "amount": reserve})
    for kind, item, amount, priority in (("PRODUCT", "FERTILIZER", 24, 2 if seed_first else 1),
                                         ("SEED", "TOMATO", 1, 1 if seed_first else 2)):
        executor.apply({"op": "BUY", "kind": kind, "item": item, "quantity": amount,
                        "price_below": 1000, "start": 0, "until": "GAME_END", "priority": priority})
    view = SimpleNamespace(money=money, market=market, farm={"hires_today": 0, "unlocked_quadrants": ["NW"]},
                           prices=market["prices"], step=0, hour=0)
    projected = {"shed": {}, "seeds": {}, "inventories": [{}], "tiles": {}}
    queue, blockers, requisitions = executor._market(view, projected, [])
    from farm_neat.economy.heuristic.policy import _MarketLedger
    ledger = _MarketLedger(money - reserve, market, {}, 0)
    projected_units = [ledger.commit(order) for order in queue]
    opponent_queue = [] if opponent == "PASS" else [
        ["BUY_PRODUCT" if opponent == "BUY" else "SELL", "FERTILIZER", 24]]
    farms = [{"money": 100000} for _ in range(2)]
    farms[seat]["money"] = money
    private = [{"shed": {}, "seeds": {}} for _ in range(2)]
    if opponent == "SELL":
        private[1 - seat]["shed"]["FERTILIZER"] = 24
    state = [AttrDict(observation=AttrDict(farms=farms, market=market, private=private[i]),
                      action={"market": queue if i == seat else opponent_queue}) for i in range(2)]
    commits = []
    original = engine._commit_unit

    def tapped(op, item, price, farm, own_private, town, capacity=100):
        before = farm["money"]
        ok = original(op, item, price, farm, own_private, town, capacity)
        if farm is farms[seat]:
            commits.append({"op": op, "item": item, "price": price, "filled": bool(ok),
                            "cash_before": before, "cash_after": farm["money"],
                            "cash_delta": farm["money"] - before})
        return ok

    # Process-local hook, restored even on a failure; run serially, not in threads.
    engine._commit_unit = tapped
    try:
        engine._process_market(state, SimpleNamespace(configuration={}))
    finally:
        engine._commit_unit = original
    filled = Counter((r["op"], r["item"]) for r in commits if r["filled"])
    return {"seat": seat, "initial_cash": money, "reserve": reserve, "opponent": opponent,
            "seed_first": seed_first, "emitted_queue": queue, "opponent_queue": opponent_queue,
            "planner_blocked": blockers, "planner_acquisition_requests": requisitions,
            "own_only_projected_units": projected_units,
            "own_only_projected_cash": ledger.cash + reserve,
            "actual_cash": farms[seat]["money"], "actual_inventory": private[seat],
            "actual_fills": [{"op": op, "item": item, "units": n} for (op, item), n in sorted(filled.items())],
            "ordered_commits": commits,
            "cash_floor": cash_prefix_audit(money, [c["cash_delta"] for c in commits], reserve),
            "isolated_market_phases": 1, "initialized_worlds": 0,
            "scope": "INJECTED_MARKET_FIXTURE_NOT_FULL_GAME_OR_NEURAL_RESULT"}


def build() -> dict:
    rows = [case(seat=seat, **config) for seat in (0, 1) for config in (
        {"opponent": "PASS"}, {"opponent": "BUY"}, {"opponent": "SELL"},
        {"opponent": "BUY", "money": 2610, "reserve": 100},
        {"opponent": "BUY", "seed_first": True})]
    return {"schema": "farm-reserve-market-stress-v1", "engine": assert_bundled_engine_identity(),
            "cases": rows, "work_accounting": {"isolated_market_phases": len(rows),
                                                "initialized_worlds": 0, "full_games": 0},
            "interpretation": "Known own-only projection limitation reproduced, not repaired. No strategy is silently overridden."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build()
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.out.exists():
        if args.out.read_text() != payload:
            raise ValueError("retained stress result differs")
    else:
        args.out.write_text(payload)
    print(f"PASS: {len(result['cases'])} isolated exact market-phase fixtures; 0 initialized worlds / 0 games.")
    for row in result["cases"]:
        print(row["seat"], row["opponent"], "seed_first", row["seed_first"],
              "cash", row["own_only_projected_cash"], "->", row["actual_cash"],
              "seed", row["actual_inventory"]["seeds"].get("TOMATO", 0),
              "floor_safe", row["cash_floor"]["safe_on_supplied_path"])


if __name__ == "__main__":
    main()
