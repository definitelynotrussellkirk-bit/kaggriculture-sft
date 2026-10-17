"""Evaluate plan programs in the real 2-player game vs ShopRouter (and vs PASS) on fixed seeds."""
import argparse, copy, json, os, sys, time
from pathlib import Path
from multiprocessing import Pool
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the src/ directory, so farm_neat imports work when run as a script
sys.path.insert(0, _SRC)
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment, PASS_ACTION, assert_bundled_engine_identity


def play_program(program, seed, seat, opponent, *, record_actions=False, timeout=60):
    """One exact complete game, editable economic program, no neural dependencies.

    This is the same opening-program game used by teacher evaluation. It is a
    scripted-policy sandbox, not the unimplemented human browser interface.
    """
    from farm_neat.economy.action_space import validate_program
    from farm_neat.economy.orders import FullSeasonPlan
    from farm_neat.evaluation.shop_router import ShopRouterAdapter
    if type(seed) is not int or not 0 <= seed < 2**31 or type(seat) is not int or seat not in (0, 1):
        raise ValueError("explicit nonnegative seed and seat0/1 required")
    if opponent not in ("pass", "router") or not 0 < timeout <= 300:
        raise ValueError("PASS/Router and timeout in (0,300] required")
    program = copy.deepcopy(program)
    if not program or program[-1].get("op") != "STOP":
        program.append({"op": "STOP"})
    program = validate_program(program)
    started = time.monotonic()
    env = ExactKaggricultureEnvironment(seed=seed)
    ctl = FullSeasonPlan(f"eval-{seed}-{seat}")
    obs = env.agent_observation(seat); ctl.observe(obs)
    proposal = ctl.propose(obs, program)
    ctl = ctl.publish(obs, proposal)
    other = ShopRouterAdapter() if opponent == "router" else None
    actions = []
    while not env.done:
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"single-game bound after {env.step_index} exposed transitions; seed{seed}")
        joint = [copy.deepcopy(PASS_ACTION), copy.deepcopy(PASS_ACTION)]
        joint[seat] = ctl(env.agent_observation(seat), env.configuration)
        if other is not None:
            joint[1-seat] = other.act(env.agent_observation(1-seat), env.configuration)
        env.step(joint)
        if record_actions: actions.append(copy.deepcopy(joint))
        ctl.observe(env.agent_observation(seat))
        if other is not None: other.observe(env.agent_observation(1-seat))
    farms = env.agent_observation(seat)["farms"]
    cash = [float(f["money"]) for f in farms]
    return {"seed": seed, "seat": seat, "opponent": opponent, "physical_games": 1,
        "physical_steps": env.step_index, "terminal_cash_by_seat": cash,
        "own_cash": cash[seat], "opponent_cash": cash[1-seat],
        "outcome": "WIN" if cash[seat] > cash[1-seat] else "LOSS" if cash[seat] < cash[1-seat] else "DRAW",
        "program": program, "opening_refusals": proposal["rejected"],
        "joint_actions": actions if record_actions else None,
        "engine": assert_bundled_engine_identity(),
        "opponent_identity": other.identity if other else {"kind": "PASS"},
        "seconds": time.monotonic()-started, "fresh_learning_games": 0, "optimizer_updates": 0}

def play(args):
    prog, seed, seat, opp = args
    try:
        result = play_program(prog, seed, seat, opp)
        return [result["own_cash"], result["opponent_cash"]]
    except Exception as e:
        return [-1.0, -1.0, repr(e)[:120]]


def try_game(argv):
    from farm_neat.evaluation.common import bound, file_sha
    parser = argparse.ArgumentParser(description="Play one complete CPU game with your economic program; no weights or Kaggle installation.")
    parser.add_argument("--program", type=Path, required=True, help="JSON command list, or a shipped teacher hall with --teacher-index")
    parser.add_argument("--teacher-index", type=int)
    parser.add_argument("--opponent", choices=("pass", "router"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--command-budget", type=int, required=True, help="Includes final STOP; the compared policy uses64")
    parser.add_argument("--out", type=Path, required=True, help="New receipt path under work/runs; no overwrites")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    out = args.out.absolute()
    if (".." in out.parts or out.suffix != ".json" or not out.is_relative_to(root/"work/runs")
            or any(p.is_symlink() for p in (out, *out.parents))):
        parser.error("receipt must be under this workspace's work/runs without symlinks")
    payload = json.loads(args.program.read_text())
    if args.teacher_index is not None:
        if not isinstance(payload, list) or not 0 <= args.teacher_index < len(payload): parser.error("teacher index outside hall")
        payload = payload[args.teacher_index]["program"]
    if not isinstance(payload, list) or any(not isinstance(c, dict) for c in payload): parser.error("JSON command list required")
    if not payload or payload[-1].get("op") != "STOP": payload = [*payload, {"op": "STOP"}]
    if not 1 <= args.command_budget <= 256 or len(payload) > args.command_budget: parser.error("program exceeds explicit budget including STOP")
    request = {"schema": "farmneat-reviewer-program-game-v1", "status": "REGISTERED",
        "physical_games_requested": 1,
        "seed": args.seed, "seat": args.seat, "opponent": args.opponent,
        "program_file_sha256": file_sha(args.program), "program": payload,
        "teacher_index": args.teacher_index, "command_budget": args.command_budget,
        "source_sha256": file_sha(Path(__file__)), "engine": assert_bundled_engine_identity(),
        "timeout_seconds": args.timeout, "retention": {"purpose": "One reviewer-controlled demonstration game",
        "dependencies": [], "remove_when": "Feedback is integrated and this game no longer supports an open question"}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream: stream.write(json.dumps(bound(request), indent=2)+"\n")
    try:
        result = play_program(payload, args.seed, args.seat, args.opponent, record_actions=True, timeout=args.timeout)
        receipt = bound({**request, **result, "status": "COMPLETE", "scope": "One exposed scripted-policy game; not neural evaluation or an unseen strength panel"})
    except Exception as error:
        receipt = bound({**request, "status": "FAILED", "error": f"{type(error).__name__}: {error}",
                         "seed_exposed": True, "physical_games_completed": 0, "no_automatic_retry": True,
                         "failure_accounting": "No verified complete result; exact initialized/partial work unknown. Seed conservatively exposed even if failure preceded initialization."})
        out.write_text(json.dumps(receipt, indent=2)+"\n")
        raise
    out.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k: receipt[k] for k in ("status", "seed", "seat", "opponent", "physical_steps", "own_cash", "opponent_cash", "outcome", "seconds")}, indent=2))

def main(path, n_seeds=8, workers=24, top=10):
    hall = json.load(open(path))
    progs = [h["program"] for h in hall[:top]]
    seeds = [4100000 + i for i in range(n_seeds)]
    jobs = [(p, s, s % 2, opp) for p in progs for s in seeds for opp in ("router", "pass")]
    with Pool(workers) as pool: res = pool.map(play, jobs)
    k = 0; out = []
    for i, p in enumerate(progs):
        rows = {"router": [], "pass": []}
        for s in seeds:
            for opp in ("router", "pass"):
                rows[opp].append(res[k]); k += 1
        r = [a for a in rows["router"] if len(a) == 2]; q = [a for a in rows["pass"] if len(a) == 2]
        errors = sum(1 for a in rows["router"] + rows["pass"] if len(a) > 2)
        if not r or not q: print(f"prog {i:2d} ALL GAMES ERRORED ({errors})"); continue
        wins = sum(1 for a in r if a[0] > a[1]); draws = sum(1 for a in r if a[0] == a[1])
        out.append({"i": i, "vs_router_own": sum(a[0] for a in r) / len(r), "vs_router_opp": sum(a[1] for a in r) / len(r),
                    "vs_router_wdl": [wins, draws, len(r) - wins - draws], "vs_pass_own": sum(a[0] for a in q) / len(q),
                    "errors": errors, "ops": [c["op"] for c in p], "program": p})
        print(f"prog {i:2d} len {len(p):2d} | vs ShopRouter own {out[-1]['vs_router_own']:8.0f} opp {out[-1]['vs_router_opp']:8.0f} W/D/L {out[-1]['vs_router_wdl']} | vs PASS own {out[-1]['vs_pass_own']:8.0f} | err {out[-1]['errors']}")
    json.dump(out, open(os.path.join(os.path.dirname(path), "eval_vs.json"), "w"), indent=1)

if __name__ == "__main__":
    if sys.argv[1:2] == ["--try-game"]: try_game(sys.argv[2:])
    else: main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8, top=int(sys.argv[3]) if len(sys.argv) > 3 else 10)
