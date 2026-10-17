"""Complete local games for the primitive-BC agent on the exact engine.

Our agent (in-process ``agent.make_agent`` or, with ``--archive``, a cold extracted Kaggle
archive in a fresh process via ``build_archive.ColdArchiveAgent``) plays full 719-step
games against PASS / SHOP_ROUTER / K0006 from the SHA-pinned opponent league, in either
seat.  Per game: per-step agent latency (p50/p95/max, first call separately because it
includes the lazy load), fallbacks, own/rival cash, W/D/L, and an engine no-op audit.

No-op audit: before every engine step, a deep copy of the pre-step state is advanced
with the engine's OWN functions (_apply_unit_action for farmer then hands, with the
engine's atomic PLANT seed check, then the market loop in its per-index lockstep order).
Unit status: pass | effective | ignored (state unchanged) | no_such_unit | plant_seed_block.
Order status: effective | partial (fewer units than requested) | ignored (zero units) |
unparsed (engine's _parse_order rejects it; the slot still counts).  Fidelity check: the
simulated post-market cash of both seats must equal the engine's cash after the real step
(cash changes only in the market phase); mismatching steps are counted per game.

``--shadow-driver K0006``: latency-only mode.  The league actor plays our seat; our agent
is timed on the same observations (so hand counts are realistic) and its actions are
discarded.  Reports latency by the acting seat's hand count.

v3 agents (agent.py): greedy decoding by default (``--temperature 0``), per-step deadline
budget ``--budget`` (s; halved by the agent when remainingOverageTime < 10 s), optional
``--background-load``; each game records the agent's summary (deadline hits, PASS steps
while loading, model steps, load ms) when the agent exposes one.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LEAGUE_DIR = ROOT / "work/runs/reviewer-play/opponent-league-adapters"
for _p in (ROOT / "src", HERE, LEAGUE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from farm_neat.game.exact_engine import ExactKaggricultureEnvironment  # noqa: E402

OPPONENTS = ("PASS", "SHOP_ROUTER", "K0006")
STEPS = 719


# ----------------------------------------------------------------------------- audit
def _is_pass(action):
    return isinstance(action, list) and len(action) >= 1 and action[0] == "PASS"


def _unit_label(action):
    if isinstance(action, list) and action and isinstance(action[0], str):
        return action[0]
    return "MALFORMED"


def _order_label(order):
    if isinstance(order, list) and order and isinstance(order[0], str):
        return order[0]
    return "MALFORMED"


def audit_step(env, actions):
    """Replay one step's action phase on copies with the engine's own functions.

    Returns (per-seat audit dicts, simulated post-market cash per seat).
    The engine is not advanced; the caller steps the real environment afterwards.
    """
    E = env.engine
    cfg = env.configuration
    tpd = max(1, int(E.get(cfg, "turnsPerDay", 24)))
    board = int(E.get(cfg, "boardSize", 10))
    shed_cap = int(E.get(cfg, "shedCapacity", 100))
    max_orders = max(1, int(E.get(cfg, "maxMarketOrdersPerTurn", 10)))
    hire_mult = int(E.get(cfg, "farmHandCostMult", E.FARM_HAND_COST_MULT))
    obs0 = env.state[0].observation
    step = int(E.get(obs0, "step", 0))
    day = step // tpd
    farms = [copy.deepcopy(dict(f)) for f in obs0.farms]
    privates = [copy.deepcopy(dict(env.state[p].observation.private)) for p in range(2)]
    market = copy.deepcopy(dict(obs0.market))
    audits = []
    # ---- unit phase, exactly the interpreter's normalization and order
    for i in range(2):
        units = []
        action = actions[i] if isinstance(actions[i], dict) else {}
        farmer_action = action.get("farmer", ["PASS"])
        hands_actions = action.get("hands", [])
        if not isinstance(hands_actions, list):
            hands_actions = []
        unit_actions = [farmer_action, *hands_actions]
        demand = Counter(a[1] for a in unit_actions
                         if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT")
        seeds = privates[i].get("seeds", {})
        blocked = {crop for crop, n in demand.items() if n > seeds.get(crop, 0)}
        for idx, a in enumerate(unit_actions):
            label = _unit_label(a)
            if _is_pass(a):
                units.append((label, "pass"))
                continue
            allowed = ["PASS"] if (isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT"
                                   and a[1] in blocked) else a
            exists = idx == 0 or idx - 1 < len(farms[i].get("hands", []))
            before = copy.deepcopy((farms[i], privates[i]))
            E._apply_unit_action(farms[i], privates[i], idx, allowed, board, day, tpd, shed_cap)
            changed = (farms[i], privates[i]) != before
            if changed:
                units.append((label, "effective"))
            elif not exists:
                units.append((label, "no_such_unit"))
            elif allowed is not a:
                units.append((label, "plant_seed_block"))
            else:
                units.append((label, "ignored"))
        audits.append({"units": units, "orders": []})
    # ---- market phase: _process_market replicated with per-order fill counts
    queues, records = [], []
    for i in range(2):
        action = actions[i] if isinstance(actions[i], dict) else {}
        m = action.get("market", []) if isinstance(action, dict) else []
        q = list(m) if isinstance(m, list) else []
        audits[i]["truncated_orders"] = max(0, len(q) - max_orders)
        q = q[:max_orders]
        queues.append(q)
        records.append([{"label": _order_label(o), "requested": 0, "filled": 0, "status": None}
                        for o in q])
    max_len = max((len(q) for q in queues), default=0)
    for k in range(max_len):
        states = []
        for i, q in enumerate(queues):
            st = None
            if k < len(q):
                st = E._parse_order(q[k])
                if st is None:
                    records[i][k]["status"] = "unparsed"   # engine skips it; the slot still counts
                else:
                    records[i][k]["requested"] = st.get("remaining", 1)
            states.append(st)
        for i, st in enumerate(states):
            if st is None:
                continue
            if st["type"] in ("HIRE", "BUY_LAND"):
                before = (farms[i]["money"], len(farms[i]["hands"]), len(farms[i]["unlocked_quadrants"]))
                if st["type"] == "HIRE":
                    E._do_hire(farms[i], privates[i], board, hire_mult)
                else:
                    E._do_buy_land(farms[i], board)
                after = (farms[i]["money"], len(farms[i]["hands"]), len(farms[i]["unlocked_quadrants"]))
                records[i][k]["filled"] = int(after != before)
                states[i] = None
        guard = 0
        while True:
            guard += 1
            if guard >= 100_000:
                break
            quoted = [None, None]
            for i, st in enumerate(states):
                if st is None or st["remaining"] <= 0:
                    continue
                op, item = st["type"], st["item"]
                if op == "SELL" and item in E.PRODUCTS:
                    quoted[i] = ("SELL", item, E.market_price(item, market["inventory"][item], market.get("params")), st)
                elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
                    quoted[i] = ("BUY_PRODUCT", item, E.market_price(item, market["inventory"][item] - 1, market.get("params")), st)
                elif op == "BUY_SEED" and item in E.CROPS:
                    quoted[i] = ("BUY_SEED", item, E.CROPS[item]["seed"], st)
                elif op == "BUY_ANIMAL" and item in E.ANIMALS:
                    quoted[i] = ("BUY_ANIMAL", item, E.ANIMALS[item]["cost"], st)
                else:
                    records[i][k]["status"] = "bad_item"
                    states[i] = None
            if all(x is None for x in quoted):
                break
            committed_any = False
            for i, x in enumerate(quoted):
                if x is None:
                    continue
                op, item, price, st = x
                if E._commit_unit(op, item, price, farms[i], privates[i], market, shed_cap):
                    st["remaining"] -= 1
                    records[i][k]["filled"] += 1
                    committed_any = True
                else:
                    states[i] = None
            if not committed_any:
                break
        E._refresh_prices(market)
    for i in range(2):
        for r in records[i]:
            if r["status"] is None:
                r["status"] = ("ignored" if r["filled"] == 0 else
                               "partial" if r["filled"] < r["requested"] else "effective")
        audits[i]["orders"] = [(r["label"], r["status"], r["requested"], r["filled"]) for r in records[i]]
    return audits, [float(f["money"]) for f in farms]


def _new_tally():
    return {"steps": 0, "unit_actions": Counter(), "units_by_op": {}, "orders": Counter(),
            "orders_by_op": {}, "requested_units": 0, "filled_units": 0,
            "truncated_orders": 0, "hands_issued": 0, "orders_issued": 0}


def _tally(t, audit):
    t["steps"] += 1
    t["hands_issued"] += max(0, len(audit["units"]) - 1)
    t["orders_issued"] += len(audit["orders"])
    t["truncated_orders"] += audit["truncated_orders"]
    for label, status in audit["units"]:
        t["unit_actions"][status] += 1
        t["units_by_op"].setdefault(label, Counter())[status] += 1
    for label, status, requested, filled in audit["orders"]:
        t["orders"][status] += 1
        t["orders_by_op"].setdefault(label, Counter())[status] += 1
        t["requested_units"] += requested
        t["filled_units"] += filled


def _plain_tally(t):
    out = dict(t)
    for key in ("unit_actions", "orders"):
        out[key] = dict(t[key])
    for key in ("units_by_op", "orders_by_op"):
        out[key] = {k: dict(v) for k, v in sorted(t[key].items())}
    nonpass = sum(v for k, v in t["unit_actions"].items() if k != "pass")
    ignored_units = nonpass - t["unit_actions"].get("effective", 0)
    out["unit_nonpass"] = nonpass
    out["unit_noops"] = ignored_units
    out["order_noops"] = sum(v for k, v in t["orders"].items() if k not in ("effective", "partial"))
    return out


# ----------------------------------------------------------------------------- stats
def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def latency_summary(ms):
    if not ms:
        return {"n": 0}
    return {"n": len(ms), "p50": round(percentile(ms, 0.5), 3), "p95": round(percentile(ms, 0.95), 3),
            "max": round(max(ms), 3), "mean": round(sum(ms) / len(ms), 3)}


# ----------------------------------------------------------------------------- game
class TimedAgent:
    """Wall-clock latency around one seat's decision (includes lazy load on step 0)."""

    def __init__(self, fn):
        self.fn = fn
        self.ms = []

    def __call__(self, observation, configuration):
        begun = time.perf_counter()
        action = self.fn(observation, configuration)
        self.ms.append((time.perf_counter() - begun) * 1000.0)
        return action


def play_game(agent, opponent, *, our_seat, world_seed, sampler_seed, league=None,
              audit=True, steps=STEPS):
    """One complete game: ``agent`` in ``our_seat`` vs a fresh league actor."""
    if our_seat not in (0, 1):
        raise ValueError("seat 0 or 1")
    if league is None:
        from league import League
        league = League()
    rival = league.new_actor(opponent, seat=1 - our_seat, sampler_seed=sampler_seed)
    ours = TimedAgent(agent)
    env = ExactKaggricultureEnvironment(seed=world_seed)
    seats = [None, None]
    seats[our_seat], seats[1 - our_seat] = ours, rival
    tallies = [_new_tally(), _new_tally()]
    mismatches = 0
    begun = time.perf_counter()
    while not env.done and env.step_index < steps:
        actions = [seats[p](env.agent_observation(p), env.configuration) for p in range(2)]
        if audit:
            audits, sim_cash = audit_step(env, actions)
        env.step(actions)
        if audit:
            real = [float(env.state[0].observation.farms[p]["money"]) for p in range(2)]
            mismatches += int(real != sim_cash)
            for p in range(2):
                _tally(tallies[p], audits[p])
    farms = env.state[0].observation.farms
    own, rival_cash = float(farms[our_seat]["money"]), float(farms[1 - our_seat]["money"])
    fallbacks = getattr(agent, "fallbacks", None)
    summary = getattr(agent, "summary", None)
    result = {
        "opponent": opponent, "our_seat": our_seat, "world_seed": world_seed,
        "sampler_seed": sampler_seed, "steps": env.step_index, "done": bool(env.done),
        "statuses": [str(s.status) for s in env.state],
        "own_cash": own, "rival_cash": rival_cash,
        "outcome": "W" if own > rival_cash else "L" if own < rival_cash else "D",
        "latency_ms": latency_summary(ours.ms),
        "latency_ms_after_first": latency_summary(ours.ms[1:]),
        "step_ms": [round(v, 3) for v in ours.ms],
        "first_step_ms": round(ours.ms[0], 3) if ours.ms else None,
        "fallbacks": fallbacks,
        "fallback_errors": list(getattr(agent, "errors", []))[:5],
        "game_seconds": round(time.perf_counter() - begun, 3),
        "agent_summary": summary() if callable(summary) else None,
    }
    if audit:
        result["audit_cash_mismatch_steps"] = mismatches
        result["noop_audit_ours"] = _plain_tally(tallies[our_seat])
        result["noop_audit_rival"] = _plain_tally(tallies[1 - our_seat])
    return result


def shadow_game(agent, driver, rival, *, seat, world_seed, sampler_seed, league=None, steps=STEPS):
    """Latency on realistic states: league actor ``driver`` plays ``seat`` against ``rival``;
    our agent is called on the same observation every step and its action is discarded."""
    if league is None:
        from league import League
        league = League()
    actors = [None, None]
    actors[seat] = league.new_actor(driver, seat=seat, sampler_seed=sampler_seed)
    actors[1 - seat] = league.new_actor(rival, seat=1 - seat, sampler_seed=sampler_seed + 1)
    env = ExactKaggricultureEnvironment(seed=world_seed)
    ms, hands = [], []
    while not env.done and env.step_index < steps:
        observations = [env.agent_observation(p) for p in range(2)]
        mine = copy.deepcopy(observations[seat])
        hands.append(len(mine["farms"][seat]["hands"]))
        begun = time.perf_counter()
        agent(mine, env.configuration)
        ms.append((time.perf_counter() - begun) * 1000.0)
        env.step([actors[p](observations[p], env.configuration) for p in range(2)])
    buckets = {}
    for label, lo, hi in (("0", 0, 0), ("1-3", 1, 3), ("4-7", 4, 7), ("8-11", 8, 11), ("12-15", 12, 15), ("16+", 16, 10**9)):
        sel = [m for m, h in zip(ms[1:], hands[1:]) if lo <= h <= hi]
        if sel:
            buckets[label] = latency_summary(sel)
    return {"mode": "shadow", "driver": driver, "opponent": rival, "our_seat": seat, "world_seed": world_seed,
            "steps": env.step_index, "max_hands": max(hands) if hands else 0,
            "first_step_ms": round(ms[0], 3) if ms else None,
            "latency_ms_after_first": latency_summary(ms[1:]), "latency_ms_by_hands": buckets,
            "step_ms": [round(v, 3) for v in ms], "hands": hands,
            "fallbacks": getattr(agent, "fallbacks", None),
            "fallback_errors": list(getattr(agent, "errors", []))[:5]}


def summarize(games):
    by = {}
    for g in games:
        key = f"{g['opponent']}/seat{g['our_seat']}"
        row = by.setdefault(key, {"games": 0, "W": 0, "D": 0, "L": 0, "own_cash": [], "rival_cash": [],
                                  "fallbacks": 0, "unit_nonpass": 0, "unit_noops": 0,
                                  "orders": 0, "order_noops": 0, "complete": 0, "_ms": [], "deadline_hits": 0})
        row["games"] += 1
        row[g["outcome"]] += 1
        row["complete"] += int(g["done"] and g["steps"] == STEPS)
        row["own_cash"].append(g["own_cash"])
        row["rival_cash"].append(g["rival_cash"])
        row["fallbacks"] += int(g["fallbacks"] or 0)
        row["deadline_hits"] += int((g.get("agent_summary") or {}).get("deadline_hits") or 0)
        row["_ms"].extend(g.get("step_ms", [])[1:])
        audit = g.get("noop_audit_ours")
        if audit:
            row["unit_nonpass"] += audit["unit_nonpass"]
            row["unit_noops"] += audit["unit_noops"]
            row["orders"] += audit["orders_issued"]
            row["order_noops"] += audit["order_noops"]
    for row in by.values():
        row["latency_ms_after_first"] = latency_summary(row.pop("_ms"))
    pooled = [v for g in games for v in g.get("step_ms", [])[1:]]
    first = [g["first_step_ms"] for g in games if g.get("first_step_ms") is not None]
    return {"cells": by, "games": len(games),
            "W": sum(g["outcome"] == "W" for g in games), "D": sum(g["outcome"] == "D" for g in games),
            "L": sum(g["outcome"] == "L" for g in games),
            "complete": sum(int(g["done"] and g["steps"] == STEPS) for g in games),
            "fallbacks": sum(int(g["fallbacks"] or 0) for g in games),
            "latency_ms_after_first_pooled": latency_summary(pooled),
            "first_step_ms": latency_summary(first)}


# ----------------------------------------------------------------------------- CLI
def _build_agent(args, sampler_seed):
    if args.archive:
        from build_archive import ColdArchiveAgent
        return ColdArchiveAgent(args.archive, threads=args.threads)
    import agent as agent_module
    return agent_module.make_agent(args.model, seed=sampler_seed, temperature=args.temperature,
                                   threads=args.threads, log=args.log_steps, budget_s=args.budget,
                                   background=args.background_load)


def _run_one(spec):
    args = argparse.Namespace(**spec["args"])
    agent = _build_agent(args, spec["sampler_seed"])
    try:
        if args.shadow_driver:
            return shadow_game(agent, args.shadow_driver, spec["opponent"], seat=spec["seat"],
                               world_seed=spec["world_seed"], sampler_seed=spec["sampler_seed"], steps=args.steps)
        return play_game(agent, spec["opponent"], our_seat=spec["seat"], world_seed=spec["world_seed"],
                         sampler_seed=spec["sampler_seed"], audit=not args.no_audit, steps=args.steps)
    finally:
        close = getattr(agent, "close", None)
        if close:
            close()


def _loadavg():
    try:
        return list(os.getloadavg())
    except OSError:
        return None


def _print_game(g):
    lat = g["latency_ms_after_first"]
    if g.get("mode") == "shadow":
        print(f"shadow {g['driver']} vs {g['opponent']:<12} seat{g['our_seat']} steps={g['steps']} "
              f"max_hands={g['max_hands']} first={g['first_step_ms']} p50={lat.get('p50')} p95={lat.get('p95')} "
              f"max={lat.get('max')} ms fallbacks={g['fallbacks']}", flush=True)
        return
    print(f"{g['opponent']:<12} seat{g['our_seat']} world{g['world_seed']} {g['outcome']} "
          f"own={g['own_cash']:.0f} rival={g['rival_cash']:.0f} first={g['first_step_ms']} "
          f"p50={lat.get('p50')} p95={lat.get('p95')} max={lat.get('max')} ms fallbacks={g['fallbacks']}",
          flush=True)


def summarize_shadow(games):
    pooled, by_hands = [], {}
    for g in games:
        pooled.extend(g["step_ms"][1:])
        for m, h in zip(g["step_ms"][1:], g["hands"][1:]):
            by_hands.setdefault(h, []).append(m)
    return {"games": len(games), "fallbacks": sum(int(g["fallbacks"] or 0) for g in games),
            "max_hands": max((g["max_hands"] for g in games), default=0),
            "latency_ms_after_first_pooled": latency_summary(pooled),
            "latency_ms_by_exact_hands": {str(h): latency_summary(v) for h, v in sorted(by_hands.items())}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="checkpoint path, 'random[:seed]' or 'stub'")
    src.add_argument("--archive", help="Kaggle tar.gz: play from a cold extracted copy in a fresh process")
    parser.add_argument("--opponents", default=",".join(OPPONENTS))
    parser.add_argument("--seats", default="0,1")
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--world-seed-base", type=int, default=20260923)
    parser.add_argument("--sampler-seed-base", type=int, default=7100)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (v3 default)")
    parser.add_argument("--budget", type=float, default=0.8, help="agent per-step deadline budget, seconds")
    parser.add_argument("--background-load", action="store_true", help="agent loads weights on a thread, PASS until ready")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1, help="parallel game processes (<= 8)")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--log-steps", action="store_true", help="agent per-step stderr lines")
    parser.add_argument("--shadow-driver", help="latency-only mode: this league actor plays our seat, "
                        "our agent is timed on the same observations and its actions are discarded")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 8:
        raise SystemExit("--jobs must be in 1..8")
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    specs = []
    for w in range(args.worlds):
        for opponent in args.opponents.split(","):
            for seat in (int(s) for s in args.seats.split(",")):
                specs.append({"opponent": opponent, "seat": seat, "world_seed": args.world_seed_base + w,
                              "sampler_seed": args.sampler_seed_base + 2 * w + seat, "args": vars(args)})
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    games = []
    if args.jobs == 1:
        for spec in specs:
            games.append(_run_one(spec))
            _print_game(games[-1])
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
            games = list(pool.map(_run_one, specs))
        for g in games:
            _print_game(g)
    summary = summarize_shadow(games) if args.shadow_driver else summarize(games)
    report = {"schema": "primitive-bc-play-v3", "argv": sys.argv[1:] if argv is None else list(argv),
              "threads": args.threads, "jobs": args.jobs, "load_average_at_end": _loadavg(),
              "games": games, "summary": summary}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report["summary"], sort_keys=True))
    return report


if __name__ == "__main__":
    main()
