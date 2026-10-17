"""Round trip: retained portfolios and Router tape 0 through the V0 vocabulary, natively checked.

    PYTHONPATH=src python3 -B src/farm_neat/economy/v0/roundtrip.py --out DIR

Portfolios: each retained knapsack plan is read as a V0 plan, compiled at the opening with
constant and daily HANDS, played against the frozen Router on router-dairy in both seats,
and rebuilt as a canonical edit trajectory over the dawn epochs (TargetPolicy) and played
again. Tape 0: the frozen tape plays both seats with an event tap; its births, placements,
land purchases, hires and sales are read as a plan (cohort chains at the module cadence);
the plan is compiled and played, and the trajectory is played. The report names, per
portfolio, the cash of every route and the residual the vocabulary could not express.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from farm_neat.economy.v0 import vocabulary as V  # noqa: E402
from farm_neat.economy.v0.compiler import compile_opening  # noqa: E402
from farm_neat.economy.v0.epochs import TargetPolicy, play, run  # noqa: E402
from farm_neat.economy.v0.plan import Plan  # noqa: E402
from farm_neat.economy.v0.projection import Projection, State  # noqa: E402
from farm_neat.evaluation.receipts import play as play_program  # noqa: E402
from farm_neat.evaluation.scenarios import FixedTape, ScenarioEnvironment, catalog  # noqa: E402

FINALISTS = ROOT / "work/runs/reviewer-play/portfolio-knapsack/FINALISTS.json"
RECORDED = ROOT / "work/runs/reviewer-play/portfolio-knapsack/games/RESULT.json"
SOURCES = ["src/farm_neat/economy/v0/vocabulary.py", "src/farm_neat/economy/v0/plan.py",
           "src/farm_neat/economy/v0/projection.py", "src/farm_neat/economy/v0/compiler.py",
           "src/farm_neat/economy/v0/epochs.py", "src/farm_neat/economy/v0/modules.json",
           "src/farm_neat/economy/orders.py", "src/farm_neat/game/kaggriculture_engine_reference.py",
           "src/farm_neat/evaluation/shop_router_0909/actions.json", "project/configs/scenarios.json"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def knapsack_plan(counts: dict) -> Plan:
    cohorts = {}
    for key, n in counts.items():
        producer, policy, _regime, birth = key.split(":")
        cohorts[V.MODULE_BY_KEY[f"{producer}:{policy}:{birth}"]] = n
    return Plan(cohorts)


def summary(row: dict) -> dict:
    return {k: row[k] for k in ("seat", "cash", "margin", "wdl", "receipts_total", "sold", "unlocked", "cash_min", "seconds")}


class TapeReader:
    """Reads one seat's native game as births, placements, land, hires and sales."""

    def __init__(self, seat: int):
        self.seat = seat
        self.births, self.placements, self.land_steps, self.hires = [], [], [], Counter()
        self.sales, self.fertilized, self.first_harvest = [], set(), {}

    def __call__(self, step, obs, action, after, tap):
        before_tiles = obs["farms"][self.seat]["tiles"]
        after_tiles = after["farms"][self.seat]["tiles"]
        for y, row in enumerate(after_tiles):
            for x, a in enumerate(row):
                b = before_tiles[y][x]
                if isinstance(a, dict) and a.get("kind") == "PLANT" and not (
                        isinstance(b, dict) and b.get("kind") == "PLANT" and b.get("crop") == a.get("crop")
                        and b.get("planted_day") == a.get("planted_day")):
                    self.births.append({"tile": (x, y), "crop": a["crop"], "day": int(a["planted_day"])})
                if isinstance(a, dict) and a.get("animal") and not (isinstance(b, dict) and b.get("animal")):
                    self.placements.append({"tile": (x, y), "animal": a["animal"], "day": step // 24})
        if len(after["farms"][self.seat].get("unlocked_quadrants", [])) > len(obs["farms"][self.seat].get("unlocked_quadrants", [])):
            self.land_steps.append(step)
        for (op, item), row in tap["commits"].items():
            if op == "HIRE":
                self.hires[step // 24] += row["units"]
            elif op == "SELL":
                self.sales.append((step, item, row["units"]))
        for e in tap["unit_effects"]:
            act = e["engine_received_action"]
            tile = e.get("tile_before")
            pos = tuple(e["position"]) if e.get("position") else None
            if not isinstance(tile, dict) or pos is None:
                continue
            if act[0] == "FERTILIZE" and tile.get("kind") == "PLANT":
                self.fertilized.add((pos, tile["crop"], int(tile["planted_day"])))
            if act[0] == "HARVEST" and tile.get("kind") == "PLANT":
                units = sum(v for v in e["delta"]["carried"].values() if v > 0)
                self.first_harvest.setdefault((pos, tile["crop"], int(tile["planted_day"])), units)

    def chains(self) -> tuple[list[dict], list[str]]:
        """Cohort chains: a same-crop birth on a tile at the chain's replant day continues it;
        any other birth on the tile ends the chain. A chain that ends before its first replant
        is one lifecycle (ONE_CYCLE regime, no exit edit); a longer chain keeps an exit day."""
        chains, current, unproductive = [], {}, []
        for birth in sorted(self.births, key=lambda b: (b["day"], b["tile"])):
            tile, crop, day = birth["tile"], birth["crop"], birth["day"]
            policy = "watered_fertilized" if (tile, crop, day) in self.fertilized else "watered"
            live = current.get(tile)
            if live is not None and live["crop"] == crop and live["policy"] == policy \
                    and V.tape(live["module"], live["regime"])["seeds"][day]:
                live["cycles"] += 1
                continue  # the module's own replanting cadence
            if live is not None:
                if live["cycles"] == 1:
                    live["regime"] += 2  # ONE_CYCLE form of the same harvest regime
                else:
                    live["exit"] = day
            key = f"{crop}:{policy}:b{day:02d}"
            if key not in V.MODULE_BY_KEY:
                unproductive.append(key)
                current.pop(tile, None)
                continue
            regime = 1 if crop in ("WHEAT", "CARROT") and self.first_harvest.get((tile, crop, day)) == 2 else 0
            chain = {"crop": crop, "policy": policy, "module": V.MODULE_BY_KEY[key], "regime": regime,
                     "birth": day, "exit": None, "tile": tile, "cycles": 1}
            chains.append(chain)
            current[tile] = chain
        for placement in self.placements:
            key = f"{placement['animal']}:fed_cared:b{placement['day']:02d}"
            if key in V.MODULE_BY_KEY:
                chains.append({"crop": placement["animal"], "policy": "fed_cared", "module": V.MODULE_BY_KEY[key],
                               "regime": 0, "birth": placement["day"], "exit": None, "tile": placement["tile"]})
            else:
                unproductive.append(key)
        return chains, unproductive

    def plan(self, day: int | None = None) -> tuple[Plan, dict]:
        """The plan at dawn `day` (every chain alive or future); day None means all chains."""
        chains, unproductive = self.chains()
        counts, regimes = Counter(), defaultdict(Counter)
        for c in chains:
            if day is not None and c["exit"] is not None and c["exit"] <= day:
                continue
            counts[c["module"]] += 1
            regimes[c["module"]][c["regime"]] += 1
        chosen = {m: c.most_common(1)[0][0] for m, c in regimes.items() if c.most_common(1)[0][0]}
        plan = Plan(dict(counts), chosen)
        ticks = sum(n for s, _p, n in self.sales if s % 4 == 0)
        total = sum(n for _s, _p, n in self.sales)
        residual = {"unproductive_births": sorted(Counter(unproductive).items()),
                    "chains": len(chains), "chains_with_exit": sum(1 for c in chains if c["exit"] is not None),
                    "one_cycle_chains": sum(1 for c in chains if V.REGIMES[c["regime"]].startswith("ONE_CYCLE")),
                    "plan_keys": len({(c["module"], c["regime"]) for c in chains}),
                    "sale_units_at_demand_ticks": ticks, "sale_units": total,
                    "land_purchase_steps": list(self.land_steps),
                    "hires_by_day": [self.hires.get(d, 0) for d in range(V.SEASON_DAYS)]}
        return plan, residual


def portfolio_routes(spec: dict, plan: Plan, opening: dict, seats=(0, 1), *, labor_factor: float = 1.0,
                     compiled: bool = True, target=None) -> dict:
    """Cash of every route of one plan: compiled at the opening (constant and daily HANDS) and
    rebuilt as an edit trajectory over the dawns (target may be a plan or a state -> plan map)."""
    out = {"plan": plan.canonical(), "labor_factor": labor_factor, "routes": {}}
    if compiled:
        for mode in ("constant", "daily"):
            program, proj, _alloc = compile_opening(plan, opening, hands_mode=mode, labor_factor=labor_factor)
            rows = [summary(play_program(spec, program, seat, budget=256)) for seat in seats]
            out["routes"][f"compiled_{mode}"] = {"commands": len(program), "reserve": proj.reserve, "hands": proj.hands,
                                                 "land_steps": proj.land_steps, "games": rows}
            if mode == "constant":
                out["program_sha256"] = hashlib.sha256(json.dumps(program, sort_keys=True).encode()).hexdigest()
    rows, residuals = [], {}
    for seat in seats:
        policy = TargetPolicy(plan if target is None else target)
        result, agent = run(spec, seat, policy, labor_factor=labor_factor)
        rows.append(summary(result))
        residuals[str(seat)] = {"reached": agent.plan.canonical(), "same_plan": agent.plan == plan,
                                "residual_epochs": len(policy.residuals), "last_residual": policy.residuals.get(696),
                                "first_residual": policy.residuals.get(0),
                                "epochs_changed": sum(1 for r in result["epochs"] if r["changed"])}
    out["routes"]["trajectory_daily"] = {"games": rows, "residual": residuals}
    return out


def tape_routes(spec: dict, seats=(0, 1), labor_factor: float = 1.0) -> dict:
    """Tape 0 read as a plan trajectory: chains appear at birth and exit when their tile is reused."""
    out = {"tape": {}}
    readers = {}
    for seat in seats:
        reader = TapeReader(seat)
        row = play(spec, FixedTape(), seat, capture_events=reader)
        readers[seat] = reader
        plan, residual = reader.plan()
        out["tape"][str(seat)] = {"game": summary(row), "plan": plan.canonical(), "residual": residual}
    reader = readers[seats[0]]
    plan, _ = reader.plan()
    opening = ScenarioEnvironment(spec).agent_observation(0)
    state = State(opening)
    proj = Projection(state, plan, labor_factor=labor_factor)
    out["tape"]["projection_all_chains"] = {"reserve": proj.reserve, "binding": proj.binding, "hands": proj.hands,
                                            "land_days": proj.land, "occupancy": proj.tapes.occ}
    out["tape"]["same_plan_both_seats"] = readers[seats[0]].plan()[0] == readers[seats[-1]].plan()[0]
    out.update(portfolio_routes(spec, plan, opening, seats, labor_factor=labor_factor, compiled=False,
                                target=lambda st: reader.plan(st.day)[0]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "work/runs/reviewer-play/action-space-v0")
    ap.add_argument("--scenario", default="router-dairy")
    ap.add_argument("--seats", default="0,1")
    ap.add_argument("--portfolios", default="all")
    ap.add_argument("--labor-factors", default="1.0")
    ap.add_argument("--skip-tape", action="store_true")
    a = ap.parse_args(argv)
    seats = tuple(int(s) for s in a.seats.split(","))
    factors = [float(x) for x in a.labor_factors.split(",")]
    spec = catalog()[a.scenario]
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "farm-v0-roundtrip-v1", "scenario": a.scenario, "seats": list(seats), "labor_factors": factors,
              "action_schema_sha256": V.ACTION_SCHEMA_SHA256, "actions": V.N_ACTIONS, "modules": V.N_MODULES,
              "sources": {p: sha(ROOT / p) for p in SOURCES}, "portfolios": {}, "games": 0}
    finalists = json.loads(FINALISTS.read_text())["finalists"]
    names = list(finalists) if a.portfolios == "all" else [n for n in a.portfolios.split(",") if n]
    recorded = {}
    if RECORDED.exists():
        for row in json.loads(RECORDED.read_text())["rows"]:
            recorded[(row["program"], row["seat"])] = row["cash"][row["seat"]]
    opening = ScenarioEnvironment(spec).agent_observation(0)
    for name in names:
        plan = knapsack_plan(finalists[name]["plan"])
        entry = {"recorded_native_cash": {str(s): recorded.get((name, s)) for s in seats},
                 "knapsack_relaxation_own_cash": finalists[name]["frozen_daily"]["own_cash"], "by_labor_factor": {}}
        for factor in factors:
            entry["by_labor_factor"][str(factor)] = portfolio_routes(spec, plan, opening, seats, labor_factor=factor)
            report["games"] += 3 * len(seats)
            print(name, factor, json.dumps({k: [g["cash"][g["seat"]] for g in v["games"]]
                                            for k, v in entry["by_labor_factor"][str(factor)]["routes"].items()}), flush=True)
        report["portfolios"][name] = entry
    if not a.skip_tape:
        report["tape0"] = {"by_labor_factor": {}}
        for factor in factors:
            t = tape_routes(spec, seats, labor_factor=factor)
            report["tape0"]["by_labor_factor"][str(factor)] = t
            report["games"] += 2 * len(seats)
            print("tape0", factor, json.dumps({k: [g["cash"][g["seat"]] for g in v["games"]] for k, v in t["routes"].items()}),
                  "tape own", [t["tape"][str(s)]["game"]["cash"][s] for s in seats], flush=True)
    (a.out / "ROUNDTRIP.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print("games", report["games"], "->", a.out / "ROUNDTRIP.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
