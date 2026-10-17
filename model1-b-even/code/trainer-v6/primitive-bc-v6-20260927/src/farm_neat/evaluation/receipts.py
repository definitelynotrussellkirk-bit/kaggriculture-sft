"""Receipts by product for one economic program on one catalog scenario, six seconds a game.

Maintained evaluator (graduated from the reviewer-play harness). Reuses ScenarioEnvironment,
Program, the frozen opponents and the native market tap; no second market or executor.
Per game: terminal cash and WDL, filled SELL units and receipts per product, spending by
market operation, unit-action counts, units harvested into custody, first planting day per
crop, the cash minimum, and optionally both seats' observations at one step (evaluator-only).
`--tap-opponent` records the rival's market fills instead of ours.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
from farm_neat.evaluation.scenarios import ScenarioEnvironment, Program, catalog, opponent
from farm_neat.evaluation import realization as r
from farm_neat.economy.action_space import validate_program


def play(spec, program, seat, budget=64, tap_opponent=False, capture_step=None):
    env = ScenarioEnvironment(spec)
    actor = Program(program, budget)
    other, _ = opponent(spec["opponent"])
    sold = [Counter(), Counter()]; receipts = [Counter(), Counter()]; spend = [Counter(), Counter()]
    acts = Counter(); harvested = Counter(); first_plant = {}; cash_min = [1e9, 1e9]; feed_fail = 0
    captured = None  # evaluator-only snapshot of BOTH seats' observations at the start of one step
    t0 = time.monotonic()
    # market_tap observes one farm; --tap-opponent records the rival's fills instead of ours.
    ts = 1 - seat if tap_opponent else seat
    with r.market_tap(env, ts) as tap:
        for step in range(719):
            obs = env.agent_observation(seat)
            if capture_step is not None and step == capture_step:
                captured = {"step": step, "day": step // 24, "observations": [env.agent_observation(0), env.agent_observation(1)],
                            "scope": "EVALUATOR_ONLY: both private sheds and carried goods are visible here; an actor sees only its own."}
            joint = [None, None]
            joint[seat] = actor(obs, env.configuration)
            joint[1 - seat] = other(env.agent_observation(1 - seat), env.configuration)
            tap["commits"], tap["pre_market"], tap["unit_effects"] = {}, None, []
            env.step(joint)
            actor.controller.observe(env.agent_observation(seat))
            for e in tap["unit_effects"]:
                a = e["engine_received_action"]; op = a[0]
                acts[op] += 1
                if op == "PLANT":
                    acts["PLANT:" + str(a[1])] += 1; first_plant.setdefault(str(a[1]), step // 24)
                if op in ("HARVEST", "COLLECT_FERTILIZER"):
                    for prod, dc in e["delta"]["carried"].items():
                        if dc > 0: harvested[prod] += dc
            for s_ in (0, 1):
                cash_min[s_] = min(cash_min[s_], env.agent_observation(seat)["farms"][s_]["money"])
            for (op, item), row in tap["commits"].items():
                if op == "SELL":
                    sold[ts][item] += row["units"]; receipts[ts][item] += round(row["cash_delta"])
                else:
                    spend[ts][f"{op}:{item}"] += -round(row["cash_delta"])
    final = env.agent_observation(seat)
    cash = [f["money"] for f in final["farms"]]
    margin = cash[seat] - cash[1 - seat]
    return {"seat": seat, "cash": cash, "margin": margin, "wdl": "WIN" if margin > 0 else "LOSS" if margin < 0 else "DRAW",
            "tapped_seat": ts, "sold": dict(sold[ts]), "receipts": dict(receipts[ts]), "receipts_total": sum(receipts[ts].values()),
            "spend": dict(spend[ts]), "spend_total": sum(spend[ts].values()),
            "unlocked": final["farms"][ts].get("unlocked_quadrants"), "actions": dict(acts), "harvested": dict(harvested),
            "first_plant_day": first_plant, "cash_min": cash_min, "captured_state": captured,
            "seconds": round(time.monotonic() - t0, 1)}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--scenario", required=True); p.add_argument("--program", required=True)
    p.add_argument("--teacher-index", type=int); p.add_argument("--seat", default="both"); p.add_argument("--out", required=True); p.add_argument("--tap-opponent", action="store_true"); p.add_argument("--capture-step", type=int)
    a = p.parse_args()
    spec = catalog()[a.scenario]
    prog = json.loads(Path(a.program).read_text())
    if a.teacher_index is not None: prog = prog[a.teacher_index]["program"]
    if prog[-1].get("op") != "STOP": prog = prog + [{"op": "STOP"}]
    validate_program(prog)
    seats = [0, 1] if a.seat == "both" else [int(a.seat)]
    rows = [play(spec, prog, s, tap_opponent=a.tap_opponent, capture_step=a.capture_step) for s in seats]
    out = {"scenario": a.scenario, "opponent": spec["opponent"], "program": a.program, "teacher_index": a.teacher_index, "rows": rows}
    with Path(a.out).open("x") as stream:
        json.dump(out, stream, indent=1)
    for row in rows:
        print(f"seat{row['seat']} {row['wdl']} cash={row['cash']} margin={row['margin']:.0f} receipts={row['receipts_total']} spend={row['spend_total']} land={row['unlocked']} {row['seconds']}s")
        print("   sold:", {k: v for k, v in sorted(row['sold'].items())}); print("   receipts:", {k: v for k, v in sorted(row['receipts'].items())})
        print("   spend:", {k: v for k, v in sorted(row['spend'].items())}); print("   harvested:", {k: v for k, v in sorted(row['harvested'].items())})
        print("   actions:", {k: v for k, v in sorted(row['actions'].items()) if not k.startswith(('NORTH','SOUTH','EAST','WEST'))}, "first_plant_day:", row['first_plant_day'], "cash_min:", row['cash_min'])
if __name__ == "__main__":
    raise SystemExit(main())
