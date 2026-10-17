"""Exact public-replay labels for strict-boundary rival market flow.

The supervised target is ``E_eff``: effective displacement of shared market
inventory by the rival during one market step.  Positive values add supply;
negative values remove supply.  Labels use the *next* public observation and are
therefore training/evaluator-only, never actor features.

This quotient is valid only when the focal seat is passive on the same product,
or when its own effective displacement is independently known and subtracted.
Concurrent own/rival cycles are a different target: phase aggregate ``E_eff`` is
not sufficient because market-order rank can change value.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
from typing import Iterable

from farm_neat.game.constants import (
    BUYABLE_PRODUCTS,
    MAX_SHOP_INSTANCES,
    PRODUCTS,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
)

SCHEMA = "market-passive-flow-labels-v1"
TARGET_SCOPE = "TRAINING_EVALUATOR_ONLY_NEXT_PUBLIC_OBSERVATION"


def demand_at_step(public_shop_prefix: Iterable[str], step: int) -> dict[str, int]:
    """Return deterministic town demand at ``step`` from the public shop prefix.

    Passing a longer authored shop sequence is also safe: only the prefix that
    could have unlocked by this step is consumed.  No hidden future shop is read.
    """
    if type(step) is not int or step < 0:
        raise ValueError("nonnegative integral step required")
    shops = tuple(public_shop_prefix)
    if any(shop not in SHOPS for shop in shops):
        raise ValueError("unknown shop in public prefix")
    unlocked = min(MAX_SHOP_INSTANCES, max(0, step // 72), len(shops))
    active = shops[:unlocked]
    out = {product: 0 for product in PRODUCTS}
    if step % 4 == 0:
        for shop in active:
            items = SHOPS[shop]
            multiplier = 2 if len(items) == 1 else 1
            for product in items:
                out[product] += multiplier
    if step % 24 == 0:
        for product in TOWN_CENTER_PRODUCTS:
            out[product] += 1
    return out


def passive_flow_label(
    inventory_now: int,
    inventory_next: int,
    demand: int,
    *,
    own_effect: int = 0,
) -> int:
    """Recover rival ``E_eff`` from one public inventory transition.

    Native accounting is ``I_next = I_now + own + rival - demand``.  A passive
    focal seat has ``own_effect == 0``.  The optional subtraction exists for
    evaluator controls where exact own effective flow is separately known.
    """
    values = (inventory_now, inventory_next, demand, own_effect)
    if any(type(value) is not int for value in values):
        raise ValueError("integral inventory, demand and effective flow required")
    if inventory_now < 0 or inventory_next < 0 or demand < 0:
        raise ValueError("nonnegative inventory and demand required")
    return inventory_next - inventory_now + demand - own_effect


def _signed_request(orders: object, product: str) -> tuple[bool, int]:
    if not isinstance(orders, list):
        raise ValueError("market orders must be a list")
    present = False
    signed = 0
    for order in orders:
        if not isinstance(order, list) or len(order) < 3 or order[1] != product:
            continue
        op = order[0]
        if op not in ("SELL", "BUY_PRODUCT"):
            continue
        quantity = order[2]
        if type(quantity) is not int or quantity < 0:
            raise ValueError("market request quantity must be a nonnegative integer")
        present = True
        signed += quantity if op == "SELL" else -quantity
    return present, signed


def _active_shops(witness: dict, step: int) -> tuple[str, ...]:
    active: Iterable[str] = ()
    schedule = witness.get("shops_unlocked")
    if not isinstance(schedule, list):
        raise ValueError("witness shops_unlocked list required")
    for row in schedule:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("shop unlock rows must be [step, shops]")
        unlock_step, shops = row
        if type(unlock_step) is not int or not isinstance(shops, list):
            raise ValueError("typed shop unlock row required")
        if unlock_step > step:
            break
        active = shops
    return tuple(active)


def extract_passive_labels(witness: dict) -> list[dict]:
    """Compile one compact public witness into strict-boundary ``E_eff`` rows.

    Raw rival request ranks/fills are deliberately not a target.  If the focal
    actor has any same-product SELL/BUY_PRODUCT request, that actor-product-step
    is excluded rather than mislabelled as a strict-boundary example.
    """
    if not isinstance(witness, dict):
        raise ValueError("market witness object required")
    if witness.get("schema") != "kaggle-replay-market-witness-v1":
        raise ValueError("kaggle-replay-market-witness-v1 required")
    episode = witness.get("episode")
    products = witness.get("products")
    rows = witness.get("rows")
    if type(episode) is not int or not isinstance(products, list) or not isinstance(rows, list):
        raise ValueError("typed episode/products/rows required")
    if set(products) != set(PRODUCTS) or len(products) != len(PRODUCTS):
        raise ValueError("complete product inventory vector required")
    pindex = {product: i for i, product in enumerate(products)}
    out: list[dict] = []
    for step in range(len(rows) - 1):
        before, after = rows[step], rows[step + 1]
        if int(before.get("step", -1)) != step or int(after.get("step", -1)) != step + 1:
            raise ValueError("contiguous zero-based witness rows required")
        before_inventory = before.get("inventory")
        after_inventory = after.get("inventory")
        orders = before.get("orders")
        if (
            not isinstance(before_inventory, list)
            or not isinstance(after_inventory, list)
            or len(before_inventory) != len(products)
            or len(after_inventory) != len(products)
            or not isinstance(orders, list)
            or len(orders) != 2
        ):
            raise ValueError("complete inventory and two-seat order rows required")
        active = _active_shops(witness, step)
        demand = demand_at_step(active, step)
        for actor in (0, 1):
            rival = 1 - actor
            for product in PRODUCTS:
                actor_present, _ = _signed_request(orders[actor], product)
                if actor_present:
                    continue
                rival_present, requested_signed = _signed_request(orders[rival], product)
                index = pindex[product]
                e_eff = passive_flow_label(
                    int(before_inventory[index]),
                    int(after_inventory[index]),
                    int(demand[product]),
                )
                out.append({
                    "episode": episode,
                    "step": step,
                    "actor": actor,
                    "product": product,
                    "e_eff": e_eff,
                    "demand": int(demand[product]),
                    "rival_request_present": rival_present,
                    "requested_signed": requested_signed,
                    "target_scope": TARGET_SCOPE,
                })
    return out


def validate_cycle_supervision(row: dict) -> dict:
    """Require rank/alignment and event size for concurrent-cycle supervision."""
    if not isinstance(row, dict) or not isinstance(row.get("alignment"), str) or not row["alignment"]:
        raise ValueError("concurrent cycle requires explicit market rank/alignment")
    event_size = row.get("event_size")
    if type(event_size) is not int or event_size <= 0:
        raise ValueError("concurrent cycle requires positive event_size target")
    return {"alignment": row["alignment"], "event_size": event_size}


def _read_witness(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def export_passive_labels(paths: Iterable[Path], output: Path, *, buyable_only: bool = False) -> dict:
    """Write training-ready JSONL rows and return a compact export summary."""
    selected = set(BUYABLE_PRODUCTS if buyable_only else PRODUCTS)
    labels: list[dict] = []
    for path in paths:
        labels.extend(row for row in extract_passive_labels(_read_witness(Path(path)))
                      if row["product"] in selected)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".gz":
        raw = output.open("wb")
        compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
        handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    else:
        handle = output.open("w", encoding="utf-8", newline="\n")
    with handle:
        for row in labels:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    controls = [row for row in labels if not row["rival_request_present"]]
    return {
        "schema": SCHEMA,
        "target": "E_eff",
        "target_scope": TARGET_SCOPE,
        "rows": len(labels),
        "products": sorted(selected),
        "zero_rows": sum(row["e_eff"] == 0 for row in labels),
        "both_passive_controls": len(controls),
        "both_passive_nonzero_mismatches": sum(row["e_eff"] != 0 for row in controls),
        "output": str(output),
        "cycle_target": "SEPARATE_RANK_ALIGNMENT_PLUS_EVENT_SIZE",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("witness", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--buyable-only", action="store_true")
    args = parser.parse_args()
    summary = export_passive_labels(args.witness, args.output, buyable_only=args.buyable_only)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
