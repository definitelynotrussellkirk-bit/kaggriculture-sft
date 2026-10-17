"""Conditional producer-count screen; no actor, game runner, forecast or WDL solver.

Extends unit_economics' declared 30-day policies. A constant externally declared
price vector turns each tile into (actions, receipts, input costs). Identical
resource prefixes can then be merged. This is exact for THAT integer surrogate,
not an upper bound on native receipts, a feasible timetable or a funded plan.
All feed/fertilizer inputs are purchased externally in this relaxation. Diverting
byproduct to crop fertilizer or growing feed internally needs a coupled model.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import hashlib
from math import comb, lcm
from pathlib import Path
from typing import Mapping, Sequence

from farm_neat.evaluation import unit_economics as u
from farm_neat.evaluation.sale_order import item_effect, native_outcome

SCHEMA = "farm-product-frontier-v1"
ROOT = Path(__file__).resolve().parents[3]
CONTROL = "CONSTANT_BASE_PRICE_SURROGATE_NOT_A_PRICE_CAP"
OBSERVED = "RETROSPECTIVE_REALIZED_AVERAGES_NOT_MARGINAL_PRICES"


def integer(value: int, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [0,{maximum}]")
    return value


def prices(raw: Mapping | None = None) -> dict[str, Fraction]:
    """Exact rationals; require the entire product vector, never invent missing quotes."""
    names = tuple(u.engine().PRODUCTS)
    if raw is None:
        return {p: Fraction(u.engine().MARKET_PARAMS[p]["base"]) for p in names}
    if not isinstance(raw, Mapping) or set(raw) != set(names):
        raise ValueError("one explicit price for every product is required")
    result = {}
    for p, v in raw.items():
        if isinstance(v, (bool, float)):
            raise ValueError("use exact integer/string/Fraction prices, not floats/bools")
        q = Fraction(v)
        if q < 1 or q > 1_000_000 or q.denominator > 1_000_000:
            raise ValueError("price must lie in [1,1000000] with bounded denominator")
        result[p] = q
    return result


def vectors(native_table: dict | None = None) -> list[dict]:
    """No new biological formulas: reuse the parent's native counts intact."""
    table = u.table() if native_table is None else native_table
    rows = []
    for r in table["rows"]:
        animal = r["producer"] in u.engine().ANIMALS
        acquisition = (u.engine().ANIMALS[r["producer"]]["cost"] if animal
                       else u.engine().CROPS[r["producer"]]["seed"] * r["seeds_used"])
        rows.append({"id": r["producer"] + "/" + r["policy"],
                     "producer": r["producer"], "policy": r["policy"],
                     "product": r["product"], "units": r["units"],
                     "actions": sum(r["actions"].values()),
                     "wheat_input": r["wheat_fed"], "fertilizer_input": r["fertilizer_used"],
                     "fertilizer_output": r["fertilizer_collected"],
                     "acquisition": acquisition,
                     "first_acquisition": r["acquisition_cost_per_cycle"]})
    return sorted(rows, key=lambda x: x["id"])


def amounts(row: dict, p: Mapping) -> tuple[Fraction, Fraction, Fraction]:
    """Gross receipts, all-season procurement, contribution before hands/land."""
    revenue = row["units"] * p[row["product"]] + row["fertilizer_output"] * p["FERTILIZER"]
    procurement = (row["acquisition"] + row["wheat_input"] * p["WHEAT"]
                   + row["fertilizer_input"] * p["FERTILIZER"])
    return revenue, procurement, revenue - procurement


def rank(rows: Sequence[dict], p: Mapping) -> dict:
    metrics = {
        "receipts_per_tile_season": lambda r: amounts(r, p)[0],
        # (receipts/30)/(actions/30) = receipts/action, not receipts/(actions/day).
        "receipts_per_action": lambda r: amounts(r, p)[0] / r["actions"],
        "receipts_per_season_acquisition_dollar": lambda r: amounts(r, p)[0] / r["acquisition"],
        "receipts_per_first_acquisition_dollar": lambda r: amounts(r, p)[0] / r["first_acquisition"],
    }
    return {name: [r["id"] for r in sorted(rows, key=lambda r: (-fn(r), r["id"]))]
            for name, fn in metrics.items()}


def dominance(rows: Sequence[dict], p: Mapping) -> list[dict]:
    """Sufficient one-for-one substitute, only under unlimited fixed-price fills.

    Check land (one each), service actions, acquisitions and commodity input
    quantities separately. Never prune an economic family from the live policy.
    Output composition is NOT equivalent: this certificate ends when price,
    demand, feed custody, output usefulness or timing differs.
    """
    answer = []
    for a in rows:
        for b in rows:
            if a is b:
                continue
            ra, _, na = amounts(a, p); rb, _, nb = amounts(b, p)
            lower = all(a[k] <= b[k] for k in ("actions", "acquisition", "wheat_input", "fertilizer_input"))
            strict = ra > rb or any(a[k] < b[k] for k in ("actions", "acquisition", "wheat_input", "fertilizer_input"))
            if lower and ra >= rb and na >= nb and strict:
                answer.append({"substitute": a["id"], "dominated": b["id"],
                               "receipts_gain": str(ra-rb), "contribution_gain": str(na-nb)})
    return answer


def solve(rows: Sequence[dict], p: Mapping, *, tiles: int = 75, hands: int = 3,
          objective: str = "receipts", action_budget: int | None = None) -> dict:
    """Exact Pareto DP over (tile count, total service actions), with KEEP=empty.

    Unlimited repetitions of declared producer/policy rows. At each exact tile
    count retain only strictly increasing value versus actions. Higher-action
    lower-value prefixes cannot improve with ANY common additive continuation.
    Tie rule: maximum objective, then fewer tiles/actions, then lexicographically
    smallest count tuple in sorted row-ID order. No resource-price dual guessing.
    """
    integer(tiles, "tiles", 75); integer(hands, "hands", 12)
    if objective not in ("receipts", "contribution"):
        raise ValueError("objective must be receipts or contribution, never WDL")
    rows = sorted(rows, key=lambda r: r["id"])
    if not rows or len(rows) > 13 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("1..13 unique producer-policy rows required")
    for r in rows:
        if type(r["actions"]) is not int or r["actions"] <= 0:
            raise ValueError("positive integer service actions required")
    p = prices(p)
    nominal = 30 * 24 * (hands + 1)
    budget = nominal if action_budget is None else integer(action_budget, "action_budget", 100_000)
    weights = [amounts(r, p)[0 if objective == "receipts" else 2] for r in rows]
    scale = lcm(*(v.denominator for v in weights))
    weights = [int(v * scale) for v in weights]
    zero = (0,) * len(rows)
    frontier = {0: (0, zero)}
    # Best tuple: value, tile count, actions, counts. Empty plan is a real option.
    best = (0, 0, 0, zero)
    transitions = 0; retained = 1; peak = 1
    for t in range(1, tiles + 1):
        candidates = {}
        for a, (value, counts) in sorted(frontier.items()):
            for j, row in enumerate(rows):
                transitions += 1
                a2 = a + row["actions"]
                if a2 > budget:
                    continue
                v2 = value + weights[j]
                c2 = counts[:j] + (counts[j]+1,) + counts[j+1:]
                old = candidates.get(a2)
                if old is None or v2 > old[0] or (v2 == old[0] and c2 < old[1]):
                    candidates[a2] = (v2, c2)
        frontier = {}; high = None
        for a, (value, counts) in sorted(candidates.items()):
            if high is not None and value <= high:
                continue
            frontier[a] = (value, counts); high = value
            proposal = (value, t, a, counts)
            if value > best[0] or (value == best[0] and proposal[1:] < best[1:]):
                best = proposal
        retained += len(frontier); peak = max(peak, len(frontier))
        if not frontier:
            break
    value, used_tiles, used_actions, counts = best
    gross = sum((amounts(r, p)[0]*n for r, n in zip(rows, counts)), Fraction())
    purchases = sum((amounts(r, p)[1]*n for r, n in zip(rows, counts)), Fraction())
    fed = sum(r["wheat_input"]*n for r,n in zip(rows, counts))
    fib = [1, 1]
    while len(fib) < hands:
        fib.append(sum(fib[-2:]))
    return {"objective": objective, "objective_value": str(Fraction(value, scale)),
            "mix": {r["id"]: n for r,n in zip(rows, counts) if n},
            "tiles": used_tiles, "actions": used_actions, "action_budget": budget,
            "gross_receipts": str(gross), "procurement": str(purchases),
            "contribution_before_hands_land": str(gross-purchases),
            "wheat_to_purchase": fed,
            "fertilizer_to_purchase": sum(r["fertilizer_input"]*n for r,n in zip(rows, counts)),
            "fixed_hand_cost_30_days": 30*sum(fib[:hands]),
            "land_cost_if_unlocking_75_tiles": 3000,
            "hands": hands, "nominal_actions_per_unit_day": 24,
            "count_vectors_before_resource_screen": comb(tiles+len(rows), len(rows)),
            "candidate_transitions": transitions, "retained_labels": retained, "peak_labels": peak,
            "status": "EXACT_SURROGATE_OPTIMUM_NATIVE_FEASIBILITY_UNKNOWN"}


def observed_profile(root: Path) -> tuple[dict, dict]:
    witness = json.loads((root / "work/runs/reviewer-play/structure/RECEIPTS-router-dairy-teacher.json").read_text())
    r = witness["rows"][0]; p = prices(); measured = []
    for product, cash in r["receipts"].items():
        if r["sold"][product] > 0:
            p[product] = Fraction(cash, r["sold"][product]); measured.append(product)
    return p, {"scenario": witness["scenario"], "profile": OBSERVED,
               "observed_products": sorted(measured),
               "base_fallback_products": sorted(set(p)-set(measured)),
               "wool_receipts": r["receipts"]["WOOL"], "wool_units": r["sold"]["WOOL"]}


def native_checks(rows: Sequence[dict]) -> dict:
    """162 native isolated market phases and a two-arm glut adverse case; no games."""
    checked = 0; digest = hashlib.sha256()
    for product in u.engine().PRODUCTS:
        for inv in (9500, 10000, 10500):
            for quantity in (1, 34, 100):
                for seat in (0, 1):
                    market = {p: 10000 for p in u.engine().PRODUCTS}; market[product] = inv
                    got = native_outcome(market, {product: quantity}, {}, [["SELL", product, quantity]], [], seat=seat)
                    expected = item_effect(product, inv, quantity, 0, 0, 0)
                    assert (got["own_cash"], got["other_cash"], got["inventory"][product]) == expected
                    assert got["own_fills"][product] == quantity
                    digest.update((json.dumps([product,inv,quantity,seat,*expected],separators=(",",":"))+"\n").encode())
                    checked += 1
    controls = {}
    market = {p:10000 for p in u.engine().PRODUCTS}; market["WOOL"] = 10059
    for name in ("SHEEP", "MELON"):
        r = next(r for r in rows if r["producer"] == name)
        stock = {r["product"]: r["units"]}
        if r["fertilizer_output"]:
            stock["FERTILIZER"] = r["fertilizer_output"]
        got = native_outcome(market, stock, {}, [["SELL",p,n] for p,n in stock.items()], [])
        assert all(got["own_fills"][p] == n for p,n in stock.items())
        controls[name] = {"base_projection": str(amounts(r, prices())[0]),
                          "native_receipts": got["own_cash"], "native_fills": {p:n for p,n in got["own_fills"].items() if n},
                          "service_actions": r["actions"]}
    assert controls["SHEEP"]["native_receipts"] < controls["MELON"]["native_receipts"]
    return {"native_market_price_cases": checked, "native_adverse_arms": 2,
            "price_case_grid":{"products":list(u.engine().PRODUCTS),"inventories":[9500,10000,10500],
                               "quantities":[1,34,100],"seats":[0,1]},
            "checked_effects_sha256":digest.hexdigest(),
            "initialized_worlds": 0, "complete_games": 0,
            "glut_case": {"start_wool_inventory": 10059, "all_other_inventory":10000,
                          "opponent_orders": [], "controls": controls,
                          "limit":"Synthetic market-entry custody, not a production timetable or terminal game."}}


def report(root: Path = ROOT) -> dict:
    table = u.table(); all_rows = vectors(table)
    rows = [r for r in all_rows if r["policy"] != "watered_fertilized"]
    p = prices(); op, obs = observed_profile(root)
    scenarios = json.loads((root / "project/configs/scenarios.json").read_text())["scenarios"]
    catalog = [{"scenario": s["id"], "rank_profile": "base", "mix_profile": "base",
                "shop_specific_price_cap": "UNAVAILABLE_IN_PACKET",
                "schedule_visibility": "EVALUATOR_ONLY_NOT_AN_ACTOR_INPUT"} for s in scenarios]
    # This corrects attribution without guessing the realized price history.
    dairy = next(s for s in scenarios if s["id"] == obs["scenario"])
    obs["catalog_yarn_unlock_days"] = [3*(i+1) for i,s in enumerate(dairy["shops"]) if s == "YARN_STORE"]
    obs["is_no_yarn_scenario"] = not obs["catalog_yarn_unlock_days"]
    measurements = []
    for r in all_rows:
        gross, procurement, net = amounts(r,p)
        measurements.append({**r, "base_receipts":str(gross), "base_procurement":str(procurement),
                             "base_contribution":str(net), "base_receipts_per_action":str(gross/r["actions"]),
                             "base_receipts_per_acquisition_dollar":str(gross/r["acquisition"])})
    fertilization = []
    for r in all_rows:
        if r["policy"] == "watered_fertilized":
            b = next(b for b in rows if b["producer"] == r["producer"])
            fertilization.append({"producer":r["producer"], "extra_units":r["units"]-b["units"],
                                  "extra_actions":r["actions"]-b["actions"],
                                  "fertilizer_input":r["fertilizer_input"],
                                  "base_contribution_delta":str(amounts(r,p)[2]-amounts(b,p)[2])})
    observed_rows = [{"producer":r["producer"], "receipts":str(amounts(r,op)[0]),
                      "contribution":str(amounts(r,op)[2])} for r in rows]
    return {"schema": SCHEMA, "days":30, "rows":measurements, "scenarios":catalog,
            "prices": {"base":{k:str(v) for k,v in p.items()},
                       "observed_diagnostic":{k:str(v) for k,v in op.items()}},
            "price_profile_meanings":{"base":CONTROL, "observed_diagnostic":OBSERVED},
            "rankings":{"base":rank(rows,p), "observed_diagnostic":rank(rows,op)},
            "dominance":{"base":dominance(rows,p), "observed_diagnostic":dominance(rows,op)},
            "observed_rows":observed_rows, "wool_attribution":obs,
            "fertilization_deltas":fertilization,
            "terminal_slot_control":{"nominal_hands3":solve(rows,p,hands=3),
                                     "optimistic_actual_slots_hands3":solve(rows,p,hands=3,action_budget=719*4),
                                     "meaning":"Only removes the nonexistent final joint action; hiring/travel can further reduce capacity."},
            "mixes":{"base":[solve(rows,p,hands=h) for h in range(13)],
                     "observed_diagnostic":[solve(rows,op,hands=h) for h in (3,11)]},
            "native_biology":u.verify(), "native_checks":native_checks(rows),
            "limits":["Base prices are not a bound; no DEMAND-LAW price cap was available.",
                      "8 fixed policies; 5 extra fertilized vectors are sensitivity data, not silently selected.",
                      "Unlimited constant-price sales and external input purchases; no route, timing, custody, funding or market coupling proof.",
                      "719 real joint actions; 720 is the requested optimistic nominal envelope per all-season worker.",
                      "Observed means are retrospective evaluator diagnostics, not marginal prices or forecasts.",
                      "K04/K05 are attributed historical evidence, not replayed games."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("report", "solve"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--hands", type=int, default=3)
    parser.add_argument("--tiles", type=int, default=75)
    parser.add_argument("--objective", choices=("receipts","contribution"), default="receipts")
    parser.add_argument("--prices", type=Path, help="JSON full exact price vector; externally declared, never inferred from scenario name")
    args = parser.parse_args()
    # Refuse reused outputs BEFORE expensive work; no implicit overwrite/resume.
    if args.out and args.out.exists():
        parser.error("output exists; choose a fresh path")
    if args.mode == "report":
        result = report()
    else:
        p = prices(json.loads(args.prices.read_text())) if args.prices else prices()
        rows = [r for r in vectors() if r["policy"] != "watered_fertilized"]
        result = solve(rows,p,hands=args.hands,tiles=args.tiles,objective=args.objective)
    text = json.dumps(result, sort_keys=True, separators=(",",":")) + "\n"
    if args.out:
        with args.out.open("x") as f:
            f.write(text)
        print(json.dumps({"written":str(args.out),"bytes":len(text.encode()),"games":0}))
    else:
        print(text,end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
