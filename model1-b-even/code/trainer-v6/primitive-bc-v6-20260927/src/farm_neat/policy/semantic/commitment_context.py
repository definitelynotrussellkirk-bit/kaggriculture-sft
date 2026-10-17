"""Shared actor-visible commitment facts; no model-column or mask ownership.

Adapted from September8 D05 returns by gpt56sol-d05-context (archive9c729993)
and gpt-5.6-sol-d05-20260908 (archive561dcf1f). Their independent eight-column
bindings overlap SERVICE and populationV2, so they are not stacked or relabelled.
This helper preserves the useful reserve/delivery/deadline distinctions for the
next explicitly versioned consumer. It is NOT connected to the live learner.

Important repair: the maintained SELL edit resets reserve to0. Compute the actual
post-edit declaration rather than labelling that candidate with the old reserve.
Distances/deadlines below are optimistic heuristics, not feasibility certificates.
"""
from __future__ import annotations

import copy

from farm_neat.game.constants import shed_access_tiles
from farm_neat.economy.heuristic.state import manhattan

SCHEMA = "farm-semantic-commitment-facts-v1"
PRODUCTS = ("WHEAT", "MILK", "FERTILIZER")


def _feed_cost(observation, tile):
    """Best uncongested current-supply cost, or None if no held supply exists."""
    farm = observation["farms"][observation["player"]]
    private = observation["private"]
    positions = [farm["farmer"], *farm["hands"]]
    costs = [manhattan(pos, tile) + 1
             for pos, inv in zip(positions, private["inventories"], strict=True)
             if inv.get("WHEAT", 0) > 0]
    if private["shed"].get("WHEAT", 0) > 0:
        costs.extend(min(manhattan(pos, access) + 1 + manhattan(access, tile) + 1
                         for access in shed_access_tiles()) for pos in positions)
    return min(costs) if costs else None


def _crop_days(observation, order):
    remaining = max(0, order["cycles"] - order["completed"])
    if not remaining:
        return 0
    x, y = order["tile"]
    tile = observation["farms"][observation["player"]]["tiles"][y][x]
    first = 4
    if isinstance(tile, dict) and tile.get("crop") == "WHEAT":
        age = observation["day"] - tile["planted_day"]
        first = (0 if tile.get("yield_units", 0) > 0
                 and (age >= 4 or observation["day"] == 29) else max(0, 4-age))
    return first + 4 * max(0, remaining-1)


def commitment_facts(observation, orders):
    """Unnormalized quantities and explicitly labelled estimates, own view only.

    No fixed four-site normalization: requested population and materialized orders
    differ in populationV2. No turn advances, allocation, credit, or route dispatch.
    Held supply is not called deliverable: contention, future buying and opponent
    market effects are excluded from the optimistic current-supply route estimate.
    """
    own = observation["farms"][observation["player"]]
    private = observation["private"]
    crops = [o for o in orders.orders.values() if o["kind"] == "CROP"]
    cows = [o for o in orders.orders.values() if o["kind"] == "ANIMAL"]
    unlocked = [o for o in cows if own["tiles"][o["tile"][1]][o["tile"][0]] != "LOCKED"]
    requested = max(len(cows), getattr(orders, "requested_cows", 0))
    feed_reserve = 2 * len(unlocked)  # Same standing-order recipe as _market.
    shed = int(private["shed"].get("WHEAT", 0))
    carried = sum(int(inv.get("WHEAT", 0)) for inv in private["inventories"])
    remaining_turns = max(0, min(24-observation["hour"], 719-observation["step"]))
    due, slacks, unavailable, placed = 0, [], 0, 0
    for order in cows:
        x, y = order["tile"]
        tile = own["tiles"][y][x]
        if not isinstance(tile, dict) or tile.get("animal") != "COW":
            continue
        placed += 1
        if not tile.get("fed_today", False):
            due += 1
            cost = _feed_cost(observation, order["tile"])
            if cost is None:
                unavailable += 1
            else:
                slacks.append(remaining_turns-cost)
    days_left = max(0, 29-observation["day"])
    crop_slacks = [days_left-_crop_days(observation, o) for o in crops
                   if o["completed"] < o["cycles"]]
    reserves = {item: int(orders.sale_rules.get(item, {}).get("reserve", 0))
                for item in PRODUCTS}
    prices = observation["market"]["prices"]
    return {
        "schema": SCHEMA,
        "commitment_sites": len(orders.orders),
        "requested_cows": requested,
        "committed_cows": len(cows),
        "placed_cows": placed,
        "unallocated_cows": max(0, requested-len(cows)),
        "remaining_wheat_cycles": sum(max(0, o["cycles"]-o["completed"]) for o in crops),
        "sale_reserves": reserves,
        "effective_wheat_sale_reserve": max(reserves["WHEAT"], feed_reserve),
        "standing_two_day_feed_reserve": feed_reserve,
        "requested_two_day_feed_requirement": 2*requested,
        "shed_wheat": shed,
        "carried_wheat": carried,
        "held_wheat_minus_standing_feed_reserve": shed+carried-feed_reserve,
        "due_unfed_cows": due,
        "due_cows_without_current_feed_source": unavailable,
        "optimistic_worst_feed_slack_turns": min(slacks) if slacks else None,
        "optimistic_worst_crop_deadline_slack_days": min(crop_slacks) if crop_slacks else None,
        "sale_quote_exceeds_threshold": {
            item: bool(prices.get(item, 0) > rule["price_above"])
            for item, rule in sorted(orders.sale_rules.items())},
        "future_supply_or_competing_duties_verified": False,
    }


def candidate_commitment_facts(observation, orders, index, *, apply_command, allowed=True):
    """Join actual before/after declarations without choosing a policy schema.

    Supply the selected vocabulary's apply_command; do not assume that index3
    means COW under populationV2. Illegal candidates remain unchanged and labelled.
    Both physical observation and caller's persistent controller remain untouched.
    """
    if type(index) is not int or index < 0 or type(allowed) is not bool:
        raise ValueError("nonnegative candidate index and explicit boolean legality required")
    candidate = copy.deepcopy(orders)
    if allowed:
        apply_command(candidate, index)
    return {"candidate_index": index, "candidate_applied": allowed,
            "before": commitment_facts(observation, orders),
            "after": commitment_facts(observation, candidate)}
