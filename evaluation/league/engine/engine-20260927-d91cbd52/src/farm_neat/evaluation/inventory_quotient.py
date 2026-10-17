"""Exact conditional nightly-deposit quotient; no engine, actor or game execution.

Let R=max(0, capacity-sum(shed.values())). Flatten the positive carried stacks
in native order: worker index first, then each inventory's insertion order.
For stack k=(item_k,n_k), let P_k=sum(n_j for j<k). Native admission is exactly
    a_k = min(n_k, max(0, R-P_k)).
The retained product vector is A_p=sum(a_k for item_k=p); lost product is T_p-A_p.
Proof: induction on stack order. Before capacity fills, all earlier positive
units are admitted; after it fills, every later admission is zero. The engine
clears every carried stack, including overflow. No per-unit expansion is needed.

Conditional on the immediate pre-deposit shed and ordered cargo, (shed,A,T) is
sufficient for final shed counts and loss counts. Positions, paths, asset states,
worker names and stack suffix order can be removed AFTER A is formed. This is
not a service, delivery, market, reset-position or full-game-state equivalence.
For fixed aggregate cargo and R, all stack orders have the same retained product
counts iff R=0, R>=sum(T), or there is at most one positive product. Otherwise the
order matters. For each product p, all-order admission bounds are
    max(0, min(R,sum(T)) - (sum(T)-T_p)) <= A_p <= min(R,T_p).
These bounds are individually tight under arbitrary whole-stack permutations;
they need not be jointly attainable or tight under a restricted worker rematch.

Wire format uses ordered PAIR ARRAYS for each inventory, never JSON objects.
A sorted-key serializer is harmless to these arrays but destroys mapping order.
The existing actor encoder already has carrier key_rank features; preserve them.
Future ordered cargo is a forecast/label, never information available at opening.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from farm_neat.evaluation.common import equal, file_sha, read, write_new

SCHEMA = "farm-nightly-inventory-quotient-v1"


def _quantity(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label}: nonnegative integer required")
    return value


def _item(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("nonempty item string required")
    return value


def ordered_inventories(inventories: Sequence[Mapping[str, int]]) -> list:
    """Capture native worker/key order BEFORE serialization; never mutates input."""
    if not isinstance(inventories, (list, tuple)):
        raise ValueError("worker-indexed inventory list required")
    result = []
    for inventory in inventories:
        if not isinstance(inventory, dict):
            raise ValueError("native inventory mapping required")
        result.append([[_item(k), _quantity(v, "carried quantity")]
                       for k, v in inventory.items()])
    return result


def deposit_quotient(shed: dict[str, int], inventories: list, capacity: int = 100) -> dict:
    """Exact per-product deposit/loss counts from explicit ordered stack arrays.

    Caller supplies the immediate PRE-DEPOSIT state (after the last market phase),
    not a stale turn-start snapshot. Initial over-capacity stock is permitted:
    native behavior is zero room, not deletion of stock already in the shed.
    Zero carried keys are ignored by the native deposit but remain in wire order.
    No inventory unit cap or currency value is invented. Runtime is O(stacks + items*log(items)),
    independent of quantities; neither the input nor canonical engine is modified.
    """
    capacity = _quantity(capacity, "capacity")
    if not isinstance(shed, dict):
        raise ValueError("shed mapping required")
    stock = {_item(k): _quantity(v, "shed quantity") for k, v in shed.items()}
    if not isinstance(inventories, (list, tuple)):
        raise ValueError("ordered worker inventory arrays required")
    stacks = []
    for inventory in inventories:
        if not isinstance(inventory, (list, tuple)):
            raise ValueError("ordered pair array required, not a mapping")
        seen = set()
        for pair in inventory:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("each stack must be [item, quantity]")
            item, count = _item(pair[0]), _quantity(pair[1], "carried quantity")
            if item in seen:
                raise ValueError("duplicate item in one worker inventory")
            seen.add(item)
            if count:
                stacks.append((item, count))
    room = max(0, capacity - sum(stock.values()))
    totals, admitted = Counter(), Counter()
    prefix = 0
    for item, count in stacks:
        got = min(count, max(0, room - prefix))
        if got:
            admitted[item] += got
        totals[item] += count
        prefix += count
    lost = {item: total - admitted[item] for item, total in sorted(totals.items())
            if total > admitted[item]}
    after = dict(stock)
    for item, count in admitted.items():
        after[item] = after.get(item, 0) + count
    admitted_total = min(room, prefix)
    bounds = {item: {"minimum": max(0, admitted_total - (prefix - total)),
                     "maximum": min(room, total)}
              for item, total in sorted(totals.items())}
    return {"schema": SCHEMA,
            "scope": "IMMEDIATE_PRE_DEPOSIT_COUNTS_ONLY_NOT_GAME_VALUE",
            "capacity": capacity, "room_before": room,
            "carried_by_item": dict(sorted(totals.items())),
            "admitted_by_item": dict(sorted(admitted.items())),
            "destroyed_by_item": lost, "shed_after": dict(sorted(after.items())),
            "all_carriers_empty_after": True,
            "arbitrary_stack_order_bounds": bounds,
            "aggregate_sufficient_for_all_stack_orders":
                room == 0 or room >= prefix or len(totals) <= 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "verify"))
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON: shed, inventories (ordered pair arrays), capacity")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    given = read(args.input, binding=False)
    if not isinstance(given, dict) or set(given) != {"shed", "inventories", "capacity"}:
        raise ValueError("exact shed/inventories/capacity input required")
    result = {"schema": SCHEMA, "input_sha256": file_sha(args.input),
              "source_sha256": file_sha(__file__), "result": deposit_quotient(**given),
              "new_games": 0, "model_calls": 0, "optimizer_updates": 0}
    if args.mode == "audit":
        write_new(args.out, result)
    else:
        expected = read(args.out)
        expected.pop("content_sha256")
        equal(result, expected, "all inventory quotient fields")
    print("Verified inventory accounting; no game, model or market executed.")


if __name__ == "__main__":
    main()
