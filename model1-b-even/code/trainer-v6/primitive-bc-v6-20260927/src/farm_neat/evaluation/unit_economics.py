"""Producers in terms of per-day inputs, labor, cash and output: one table, engine-checked.

Representation: each producer (five crops, three animals) under a declared service policy
is a vector of per-day requirements and outputs. The numbers come from the bundled engine's
own unit-action and day-end refresh functions applied to ONE isolated tile for a 30-day
season; closed-form expressions are asserted equal to those native counts in the tests.

Exchange rates then express one producer in terms of another: wheat tiles per fed animal,
service actions per producer, hands per set of producers, and gross value at a market price.
Nothing here is a route, a funding proof, a shared-market forecast or a WDL claim: travel,
custody, cash-flow timing and both farms' price impact are outside this one-tile boundary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from farm_neat.game.exact_engine import load_engine

SCHEMA = "farm-unit-economics-v2"
SEASON_DAYS = 30
TURNS_PER_DAY = 24
ACTIONS_PER_UNIT_PER_DAY = TURNS_PER_DAY  # nominal; travel and pickups are outside this boundary
CROP_POLICIES = ("watered", "watered_fertilized")
ANIMAL_POLICY = "fed_cared"


def engine():
    return load_engine()


def price(item: str, inventory: int) -> int:
    """The engine's quote at a market inventory (10000 is neutral)."""
    return int(engine().market_price(item, inventory))


def _one_shot_harvest_age(crop: dict, per_water: int) -> int:
    """Earliest allowed age at which the held cap is reached, else max_yield_day; replant the same day."""
    start = (crop["max_yield_day"] + 1) // 2
    for age in range(crop["first_yield_day"], crop["max_yield_day"] + 1):
        if 1 + per_water * max(0, age - start + 1) >= crop["max_yield"]:
            return age
    return crop["max_yield_day"]


def _fresh_farm(tile):
    return {"tiles": [[tile]], "farmer": [0, 0], "hands": [], "money": 0.0,
            "unlocked_quadrants": ["NW"], "hires_today": 0}


def simulate(producer: str, policy: str, days: int = SEASON_DAYS) -> dict:
    """Native one-tile season: counts of units harvested, wheat fed, fertilizer used/collected, actions."""
    E = engine()
    crops, animals = E.CROPS, E.ANIMALS
    private = {"inventories": [{"WHEAT": 10_000, "FERTILIZER": 10_000}],
               "seeds": {c: 10_000 for c in crops}}
    acts = {}
    harvested = 0
    if producer in crops:
        crop = crops[producer]
        farm = _fresh_farm(None)
        fertilize = policy == "watered_fertilized"
        harvest_age = _one_shot_harvest_age(crop, 2 if fertilize else 1)
        planted = None
        window_start = (crop["max_yield_day"] + 1) // 2

        def do(op, day):
            before = dict(private["inventories"][0])
            E._apply_unit_action(farm, private, 0, [op] if op != "PLANT" else ["PLANT", producer], 1, day, TURNS_PER_DAY)
            acts[op] = acts.get(op, 0) + 1
            return before

        for day in range(days):
            tile = farm["tiles"][0][0]
            spent = isinstance(tile, dict) and tile.get("kind") == "PLANT" and 0 <= tile.get("max_lifespan_step", -1) <= day * TURNS_PER_DAY
            if isinstance(tile, dict) and (tile.get("kind") == "WEED" or spent):
                do("DIG", day); tile = None          # a spent ongoing plant is dug the day after its last event
            if tile is None:
                do("PLANT", day); planted = day; tile = farm["tiles"][0][0]
            age = day - planted
            if fertilize and tile.get("fertilized_until_day", -1) < day:
                productive = (window_start <= age <= crop["max_yield_day"]) if not crop["ongoing"] else age >= crop["first_yield_day"] - 1
                if productive:
                    do("FERTILIZE", day)
            do("WATER", day)
            tile = farm["tiles"][0][0]
            if crop["ongoing"]:
                if age >= crop["first_yield_day"] and tile.get("yield_units", 0) > 0:
                    before = do("HARVEST", day); harvested += private["inventories"][0][producer] - before.get(producer, 0)
            elif age >= harvest_age:
                before = do("HARVEST", day); harvested += private["inventories"][0][producer] - before.get(producer, 0)
                do("PLANT", day); planted = day; do("WATER", day)  # replant the same day; planting day counts as unwatered
            for turn in range(TURNS_PER_DAY):          # native per-turn decay after the day's actions
                E._decay_plants(farm, day * TURNS_PER_DAY + turn)
            if day < days - 1:
                E._daily_refresh_plants(farm, day, TURNS_PER_DAY)
        product = producer
    else:
        animal = animals[producer]
        farm = _fresh_farm(E._new_animal(producer, 0))
        product = animal["product"]
        for day in range(days):
            for op in ("FEED", "CARE", "COLLECT_FERTILIZER"):
                E._apply_unit_action(farm, private, 0, [op], 1, day, TURNS_PER_DAY); acts[op] = acts.get(op, 0) + 1
            if farm["tiles"][0][0].get("yield_units", 0) > 0:
                before = dict(private["inventories"][0])
                E._apply_unit_action(farm, private, 0, ["HARVEST"], 1, day, TURNS_PER_DAY); acts["HARVEST"] = acts.get("HARVEST", 0) + 1
                harvested += private["inventories"][0][product] - before.get(product, 0)
            if day < days - 1:
                E._daily_refresh_animals(farm, day)
    inv = private["inventories"][0]
    is_crop = producer in crops
    return {"producer": producer, "policy": policy, "product": product, "days": days, "units": harvested,
            "wheat_fed": 0 if is_crop else 10_000 - inv.get("WHEAT", 10_000),
            "fertilizer_used": 10_000 - inv.get("FERTILIZER", 10_000) if is_crop else 0,
            "fertilizer_collected": 0 if is_crop else inv.get("FERTILIZER", 10_000) - 10_000,
            "actions": dict(sorted(acts.items())), "seeds_used": 10_000 - private["seeds"].get(producer, 10_000) if is_crop else 0}


def formula(producer: str, policy: str, days: int = SEASON_DAYS) -> dict:
    """Closed forms for the same season; asserted equal to simulate() in the tests."""
    E = engine()
    crops, animals = E.CROPS, E.ANIMALS
    if producer in crops:
        c = crops[producer]; fert = policy == "watered_fertilized"; per_water = 2 if fert else 1
        if not c["ongoing"]:
            start = (c["max_yield_day"] + 1) // 2
            cycle = _one_shot_harvest_age(c, per_water)
            units_per_cycle = min(c["max_yield"], 1 + per_water * max(0, cycle - start + 1))
            cycles = (days - 1) // cycle          # a harvest on day k*cycle needs k*cycle <= days-1
            return {"units": units_per_cycle * cycles, "cycle_days": cycle, "units_per_cycle": units_per_cycle,
                    "units_per_day": units_per_cycle / cycle, "first_output_day": cycle, "cycles": cycles}
        events = c["max_yield"]; interval = c["interval"]; first = c["first_yield_day"]
        cycle = first + (events - 1) * interval + 1   # weed appears the day after the last event
        total = 0; p = 0; cycles = 0
        while p + first <= days - 1:
            k = sum(1 for i in range(events) if p + first + i * interval <= days - 1)
            total += k * per_water; cycles += 1; p += cycle
        return {"units": total, "cycle_days": cycle, "units_per_cycle": events * per_water,
                "units_per_day": events * per_water / cycle, "first_output_day": first, "cycles": cycles}
    a = animals[producer]; first, interval, cap = a["first_yield_day"], a["interval"], a["max_held"]
    event_days = [d for d in range(first, days) if (d - first) % interval == 0]
    units = 0
    for i, d in enumerate(event_days):
        units += min(cap, 1 + (first - 1 if i == 0 else interval))  # pending care since the previous event
    return {"units": units, "events": len(event_days), "units_per_event_steady": min(cap, 1 + interval),
            "units_per_day": min(cap, 1 + interval) / interval, "first_output_day": first,
            "wheat_per_day": 1, "fertilizer_per_day": 1}


def table(days: int = SEASON_DAYS) -> dict:
    """Every producer under every declared policy, native counts joined with closed forms and prices."""
    E = engine()
    rows = []
    for name in E.CROPS:
        for pol in CROP_POLICIES:
            sim, f = simulate(name, pol, days), formula(name, pol, days)
            actions = sum(sim["actions"].values())
            rows.append({**sim, **{"formula_" + k: v for k, v in f.items()}, "acquisition_cost_per_cycle": E.CROPS[name]["seed"],
                         "cash_per_season": E.CROPS[name]["seed"] * sim["seeds_used"], "actions_per_day": actions / days,
                         "manure_actions_per_day": 0.0, "base_price": E.MARKET_PARAMS[name]["base"],
                         "gross_at_base_price_not_a_ranking_signal": E.MARKET_PARAMS[name]["base"] * sim["units"]})
    for name in E.ANIMALS:
        sim, f = simulate(name, ANIMAL_POLICY, days), formula(name, ANIMAL_POLICY, days)
        manure = sim["actions"].get("COLLECT_FERTILIZER", 0); actions = sum(sim["actions"].values()) - manure
        product = E.ANIMALS[name]["product"]
        rows.append({**sim, **{"formula_" + k: v for k, v in f.items()}, "acquisition_cost_per_cycle": E.ANIMALS[name]["cost"],
                     "cash_per_season": E.ANIMALS[name]["cost"] + E.MARKET_PARAMS["WHEAT"]["base"] * sim["wheat_fed"],
                     "actions_per_day": actions / days, "manure_actions_per_day": manure / days,
                     "base_price": E.MARKET_PARAMS[product]["base"],
                     "gross_at_base_price_not_a_ranking_signal": E.MARKET_PARAMS[product]["base"] * sim["units"]})
    # A policy that yields the same units with strictly more inputs or actions is dominated in the strategic space.
    for row in rows:
        row["dominated_by"] = None
        for other in rows:
            if other is row or other["producer"] != row["producer"] or other["units"] != row["units"]: continue
            if (other["fertilizer_used"] <= row["fertilizer_used"] and other["actions_per_day"] <= row["actions_per_day"]
                    and (other["fertilizer_used"], other["actions_per_day"]) != (row["fertilizer_used"], row["actions_per_day"])):
                row["dominated_by"] = other["policy"]
    wheat = next(r for r in rows if r["producer"] == "WHEAT" and r["policy"] == "watered")
    fib = [1, 1]
    while len(fib) < 16: fib.append(fib[-1] + fib[-2])
    exchange = {
        "wheat_tiles_per_fed_animal_day": round(1 / (wheat["units"] / days), 3),
        "note_wheat": "one fed animal eats 1 WHEAT/day; a watered WHEAT tile yields units/day as tabulated",
        "actions_per_unit_per_day_nominal": ACTIONS_PER_UNIT_PER_DAY,
        "hand_cost_per_day_by_count": {str(k): sum(fib[:k]) for k in range(0, 13)},
        "land_prices": list(E.LAND_PRICES), "shed_capacity_items": 100, "market_orders_per_turn": 10,
        "boundary": "one isolated tile; no travel, pickups, custody, funding timing, shared-market price impact or opponent",
        "value_note": "gross at base price is a unit conversion, not a ranking signal: fertilizer has no external demand and its late price can be $1; rank with a demand-aware receipt curve",
    }
    return {"schema": SCHEMA, "season_days": days, "rows": rows, "exchange": exchange,
            "engine_tables": {"CROPS": E.CROPS, "ANIMALS": E.ANIMALS, "MARKET_PARAMS": E.MARKET_PARAMS}}


def render(value: dict) -> str:
    lines = ["| producer | policy | product | units/season | units/day | first day | wheat/day | fert used | fert out | service actions/day | manure actions/day | cash/season | gross at base (not a rank) | dominated by |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    d = value["season_days"]
    for r in value["rows"]:
        lines.append(f"| {r['producer']} | {r['policy']} | {r['product']} | {r['units']} | {r['formula_units_per_day']:.2f} | {r['formula_first_output_day']} | "
                     f"{r['wheat_fed']/d:.2f} | {r['fertilizer_used']} | {r['fertilizer_collected']} | {r['actions_per_day']:.2f} | {r['manure_actions_per_day']:.2f} | {r['cash_per_season']} | {r['gross_at_base_price_not_a_ranking_signal']} | {r['dominated_by'] or ''} |")
    ex = value["exchange"]
    lines.append(f"\nwheat tiles per fed animal-day: {ex['wheat_tiles_per_fed_animal_day']}; hands cost/day by count: {ex['hand_cost_per_day_by_count']}")
    lines.append("boundary: " + ex["boundary"])
    return "\n".join(lines)


def verify(days: int = SEASON_DAYS) -> dict:
    mismatches = []
    for name in engine().CROPS:
        for pol in CROP_POLICIES:
            s, f = simulate(name, pol, days), formula(name, pol, days)
            if s["units"] != f["units"]: mismatches.append((name, pol, s["units"], f["units"]))
    for name in engine().ANIMALS:
        s, f = simulate(name, ANIMAL_POLICY, days), formula(name, ANIMAL_POLICY, days)
        if s["units"] != f["units"] or s["wheat_fed"] != days: mismatches.append((name, ANIMAL_POLICY, s["units"], f["units"]))
    return {"status": "PASS" if not mismatches else "FAIL", "mismatches": mismatches, "checked": 2 * len(engine().CROPS) + len(engine().ANIMALS)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("table", "verify"))
    p.add_argument("--out", type=Path)
    p.add_argument("--days", type=int, default=SEASON_DAYS)
    a = p.parse_args()
    if a.mode == "verify":
        result = verify(a.days); print(json.dumps(result)); return 0 if result["status"] == "PASS" else 1
    value = table(a.days)
    if a.out:
        with a.out.open("x") as stream: json.dump(value, stream, indent=1)
    print(render(value)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
