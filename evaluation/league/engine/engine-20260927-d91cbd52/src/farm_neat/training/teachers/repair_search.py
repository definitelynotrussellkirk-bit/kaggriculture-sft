"""Counterfactual plan-repair search: DAgger-style labels at real decision states.

For a real mid-game decision state (seed, seat, opponent, the programs already
published at earlier decision steps) this reconstructs the exact environment,
FullSeasonPlan controller and opponent by replaying from step 0, enumerates
candidate EDIT programs against the standing plan, plays every branch to the end
and reports Delta = cash(edit) - cash(KEEP).

PAIRED SEED, NOT FUTURE-UNCERTAINTY-AVERAGED. Every branch of one state shares
the same seed and the same opponent, so Delta is a paired difference on ONE
hidden future. It is not an expectation over futures; a positive Delta means the
edit was better on that realization, not that it is better in expectation.

Two state sources:
  corpus    work/runs/evo-20260911/corpus/games-v1.jsonl.gz (evolved programs,
            live ShopRouter or PASS opponent, controller re-executed forward).
  campaign  a fast-campaign world directory (turns.jsonl.gz + RESULT.json). The
            recorded joint actions are replayed to reach step k; forward of k the
            controller is re-executed and the opponent's RECORDED actions are
            replayed OPEN LOOP (the frozen opponent does not react to the edit).

Outputs (work/runs/evo-20260911/repairs/):
  <tag>.states.jsonl      one line per evaluated state, every candidate + Delta
  <tag>.repairs.jsonl.gz  record_corpus_v2.py schema, best edit as the label
  <tag>.effects.jsonl.gz  all candidates, paired own/opponent/WDL labels and actor context
  <tag>.summary.json      aggregate tables

Read-only legacy audit (no reconstruction, games or data rewriting):
  repair_search.py audit states=TAG.states.jsonl repairs=TAG.repairs.jsonl.gz out=AUDIT.json

The selected repair corpus still maximizes own cash; it is not a WDL selector.
Use the effects stream (including harms, ties, refusals and failures) for edit
learning. It records realized single-future contrasts, not expected causal values.
"""
import copy, gzip, hashlib, json, math, os, random, re, statistics, sys, time
from pathlib import Path
from multiprocessing import Pool

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the src/ directory, so farm_neat imports work when run as a script
sys.path.insert(0, _SRC)

from farm_neat.game.exact_engine import ExactKaggricultureEnvironment, PASS_ACTION
from farm_neat.game.constants import ANIMALS, CROPS, PRODUCTS
from farm_neat.training.teachers import DATASETS, output_dir

STEPS = (0, 144, 288, 432, 576)
LAST_STEP = 718
ANIMAL_PRODUCT = {a: ANIMALS[a]["product"] for a in ANIMALS}
PRODUCT_ANIMAL = {v: k for k, v in ANIMAL_PRODUCT.items()}
CORPUS = str(DATASETS / "corpus/games-v1.jsonl.gz")
OUTDIR = str(output_dir("work/datasets/repairs"))  # generated repair records; never the source tree
STOP = [{"op": "STOP"}]


# ---------------------------------------------------------------- environment

def clone_env(env):
    """Byte-exact environment fork.

    The bundled interpreter keeps no RNG across steps (it rebuilds a
    random.Random from env.info['seed'] and the day), so the whole mutable
    world is env.state/step_index/done. The engine module itself is shared,
    which is why copy.deepcopy(env) fails and this exists.
    """
    fork = ExactKaggricultureEnvironment.__new__(ExactKaggricultureEnvironment)
    fork.engine = env.engine
    fork.configuration = copy.deepcopy(env.configuration)
    fork.info = copy.deepcopy(env.info)
    fork.done = env.done
    fork.step_index = env.step_index
    fork.state = copy.deepcopy(env.state)
    return fork


def make_opponent(kind):
    if kind == "router":
        from farm_neat.evaluation.shop_router import ShopRouterAdapter
        return ShopRouterAdapter()
    return None


# --------------------------------------------------------- state specification
# spec = {kind, seed, seat, opponent, programs {step: program}, actions|None,
#         program_id, label}

def corpus_specs(path=CORPUS, programs=10, seeds=4):
    """First `programs` program_ids x first `seeds` games of each (both opponents)."""
    by_pid, order = {}, []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            game = json.loads(line)
            if "error" in game:
                continue
            pid = game["program_id"]
            if pid not in by_pid:
                by_pid[pid] = []
                order.append(pid)
            by_pid[pid].append(game)
    specs = []
    for pid in order[:programs]:
        games = sorted(by_pid[pid], key=lambda g: g["seed"])[:seeds]
        for game in games:
            specs.append({"kind": "corpus", "seed": game["seed"], "seat": game["seat"],
                          "game_id": game["records"][0]["snapshot"]["game_id"],
                          "opponent": game["opponent"], "program_id": pid,
                          "programs": {r["step"]: r["program"] for r in game["records"]},
                          "actions": None,
                          "recorded_cash": game["cash"],
                          "label": f"p{pid}-s{game['seed']}-seat{game['seat']}-{game['opponent']}",
                          "recorded": {r["step"]: {"observation": r["observation"], "snapshot": r["snapshot"]}
                                       for r in game["records"]}})
    return specs


def campaign_specs(producer_dir, worlds=8):
    plan = json.load(open(os.path.join(producer_dir, "PLAN.json")))
    specs = []
    names = sorted(n for n in os.listdir(producer_dir) if n.startswith("world-"))
    for name in names[:worlds]:
        index = int(name.split("-")[1])
        world = os.path.join(producer_dir, name)
        rows = [json.loads(l) for l in gzip.open(os.path.join(world, "turns.jsonl.gz"), "rt")]
        seat = next(s for s in (0, 1) if rows[0]["decisions"][s])
        programs = {}
        for row in rows:
            decision = row["decisions"][seat] if row.get("decisions") else None
            if decision:
                programs[row["step"]] = decision["behavior"]["program"]
        result = json.load(open(os.path.join(world, "RESULT.json")))
        opponent_actions = [r["actions"][1 - seat] for r in rows]
        idle = all(a["farmer"] == ["PASS"] and not a["hands"] and not a["market"] for a in opponent_actions)
        specs.append({"kind": "campaign", "seed": plan["seeds"][index], "seat": seat,
                      "opponent": "replay-pass" if idle else "replay-active",
                      "program_id": index, "programs": programs,
                      "actions": [r["actions"] for r in rows],
                      "recorded_cash": result["cash"],
                      "label": f"{os.path.basename(producer_dir)}-{name}",
                      "recorded": {}})
    return specs


# ------------------------------------------------------------- reconstruction

def reconstruct(spec, step, game_id=None):
    """Replay from step 0 to `step`; return (env, ctl, opponent, observation, diag)."""
    from farm_neat.economy.orders import FullSeasonPlan
    seat = spec["seat"]
    env = ExactKaggricultureEnvironment(seed=spec["seed"])
    ctl = FullSeasonPlan(game_id or spec.get("game_id") or f"repair-{spec['label']}",
                         publication_steps=STEPS)
    opponent = make_opponent(spec["opponent"])
    actions = spec["actions"]
    diag = {"action_mismatch": 0, "steps": 0}
    while env.step_index < step:
        index = env.step_index
        obs = env.agent_observation(seat)
        if index in spec["programs"]:
            ctl.observe(obs)
            ctl = ctl.publish(obs, ctl.propose(obs, spec["programs"][index]))
        own = ctl(obs, env.configuration)
        if actions is not None:
            joint = copy.deepcopy(actions[index])
            if json.loads(json.dumps(own)) != joint[seat]:
                diag["action_mismatch"] += 1
            # replay the RECORDED joint action, not the re-executed one
        else:
            joint = [copy.deepcopy(PASS_ACTION), copy.deepcopy(PASS_ACTION)]
            joint[seat] = own
            if opponent is not None:
                joint[1 - seat] = opponent.act(env.agent_observation(1 - seat), env.configuration)
        env.step(joint)
        ctl.observe(env.agent_observation(seat))
        if opponent is not None:
            opponent.observe(env.agent_observation(1 - seat))
        diag["steps"] += 1
    obs = env.agent_observation(seat)
    ctl.observe(obs)
    return env, ctl, opponent, obs, diag


def play_from(env, ctl, opponent, spec, later):
    """Play a branch to the end. `later` maps future decision step -> program."""
    seat, actions = spec["seat"], spec["actions"]
    fallbacks = 0
    while not env.done:
        index = env.step_index
        obs = env.agent_observation(seat)
        if index in later:
            ctl.observe(obs)
            try:
                ctl = ctl.publish(obs, ctl.propose(obs, later[index]))
            except ValueError:
                fallbacks += 1
                ctl = ctl.publish(obs, ctl.propose(obs, STOP))
        own = ctl(obs, env.configuration)
        if actions is not None:
            joint = copy.deepcopy(actions[index])
            joint[seat] = own
        else:
            joint = [copy.deepcopy(PASS_ACTION), copy.deepcopy(PASS_ACTION)]
            joint[seat] = own
            if opponent is not None:
                joint[1 - seat] = opponent.act(env.agent_observation(1 - seat), env.configuration)
        env.step(joint)
        ctl.observe(env.agent_observation(seat))
        if opponent is not None:
            opponent.observe(env.agent_observation(1 - seat))
    farms = env.agent_observation(seat)["farms"]
    return float(farms[seat]["money"]), float(farms[1 - seat]["money"]), fallbacks


# --------------------------------------------------------------- candidates

def _start(row):
    return row["command"].get("start", 0)


def _end_of(command):
    return LAST_STEP if command.get("until") == "GAME_END" else command.get("until", LAST_STEP)


def _end(row):
    return row.get("stage_until", _end_of(row["command"]))


def _free_tiles(observation, seat, order_stages, step):
    """Tiles a new stage starting at `step` can claim without a keyed conflict.

    A stage installed at start=step and until=GAME_END overlaps any existing
    stage whose end >= step, so those tiles are refused unless REPLACE is used.
    """
    tiles = observation["farms"][seat]["tiles"]
    busy = {key for key, rows in order_stages.items() if any(_end(r) >= step for r in rows)}
    open_tiles, locked = [], []
    for y in range(10):
        for x in range(10):
            if f"{x},{y}" in busy:
                continue
            cell = tiles[y][x]
            if cell is None:
                open_tiles.append((x, y))
            elif cell == "LOCKED":
                locked.append((x, y))
    return open_tiles, locked


def _rule_start(rule_stages, key, step):
    """Start of the stage a new rule command must key onto, or `step` if free.

    Rule stages are keyed by start: a command whose start differs from an
    existing overlapping stage is refused (CONFLICTING_TILE_ORDER), so an
    increment to a live HANDS/LAND/SALE must reuse that stage's own start.
    """
    rows = rule_stages.get(key, [])
    for row in sorted(rows, key=_start):
        if _end(row) >= step:
            return _start(row), row["command"]
    return step, None


def _standing_products(order_stages, step):
    products = set()
    for rows in order_stages.values():
        for row in rows:
            command = row["command"]
            if _end(row) < step:
                continue
            if command["op"] == "SET_CROP":
                products.add(command["crop"])
            elif command["op"] == "SET_ANIMAL":
                products.add(ANIMAL_PRODUCT[command["animal"]])
    return products


def gen_candidates(rng, ctl, observation, seat, step, count):
    """Coordinated economic edits (search4.joint_mutate vocabulary) as EDIT programs.

    Only the new/changed commands are emitted; the standing plan carries the
    rest. Windows are opened at `step` for new production and keyed onto the
    live stage's start for rule increments.
    """
    from farm_neat.training.teachers import search3 as s3
    orders, rules = ctl.executor.order_stages, ctl.executor.rule_stages
    open_tiles, locked = _free_tiles(observation, seat, orders, step)
    produced = _standing_products(orders, step)
    shed = observation.get("private", {}).get("shed", {})
    sale_keys = [k for k in rules if k.startswith("SALE:")]
    out, seen = [], set()

    def take(tiles, n):
        picks = rng.sample(tiles, min(n, len(tiles)))
        return picks

    def new_sale(product, price=None, reserve=None, quantity=None):
        start, _ = _rule_start(rules, "SALE:" + product, step)
        command = s3.gen_sale(rng)
        command["product"] = product
        command["start"] = start
        command["until"] = "GAME_END"
        if price is not None:
            command["price_above"] = price
        if reserve is not None:
            command["reserve"] = reserve
        if quantity is not None:
            command["quantity"] = quantity
        return command

    def animal_sale():
        tiles = open_tiles or locked
        if not tiles:
            return None
        command = s3.gen_animal(rng)
        x, y = take(tiles, 1)[0]
        command["scope"] = {"kind": "TILE", "x": x, "y": y}
        command["start"], command["until"] = step, "GAME_END"
        commands = [command]
        product = ANIMAL_PRODUCT[command["animal"]]
        if "SALE:" + product not in rules:
            commands.append(new_sale(product))
        if not open_tiles:
            start, live = _rule_start(rules, "LAND", step)
            commands.append({"op": "LAND", "target": min(3, (live or {}).get("target", 0) + 1),
                             "start": start, "until": "GAME_END", "priority": 1})
        return "animal+sale", commands

    def crop_sale():
        if not open_tiles:
            return None
        command = s3.gen_crop(rng)
        commands = []
        for x, y in take(open_tiles, rng.randint(2, 5)):
            cell = copy.deepcopy(command)
            cell["scope"] = {"kind": "TILE", "x": x, "y": y}
            cell["start"], cell["until"] = step, "GAME_END"
            commands.append(cell)
        if "SALE:" + command["crop"] not in rules:
            commands.append(new_sale(command["crop"]))
        return "crop_cluster+sale", commands

    def hands():
        start, live = _rule_start(rules, "HANDS", step)
        target = min(14, (live or {}).get("target", 2) + rng.randint(1, 4))
        return "hands+", [{"op": "HANDS", "target": target, "start": start,
                           "until": "GAME_END", "priority": 0}]

    def land():
        start, live = _rule_start(rules, "LAND", step)
        target = (live or {}).get("target", 0) + 1
        if target > 3:
            return None
        return "land+", [{"op": "LAND", "target": target, "start": start,
                          "until": "GAME_END", "priority": 1}]

    def sale_retune():
        if not sale_keys:
            return None
        product = rng.choice(sale_keys).split(":", 1)[1]
        return "sale_retune", [new_sale(product)]

    def liquidate():
        held = [(int(shed.get(p, 0)), p) for p in PRODUCTS if shed.get(p, 0)]
        product = max(held)[1] if held else (rng.choice(sorted(produced)) if produced else rng.choice(PRODUCTS))
        return "liquidate", [new_sale(product, price=0.0, reserve=0, quantity=1000)]

    def missing_sale():
        gaps = sorted(p for p in produced if "SALE:" + p not in rules)
        if not gaps:
            return None
        return "missing_sale", [new_sale(rng.choice(gaps))]

    def animal_hands():
        base = animal_sale()
        if base is None:
            return None
        _, commands = base
        return "animal+hands", commands + hands()[1]

    pool = [animal_sale, crop_sale, hands, land, sale_retune, liquidate, missing_sale, animal_hands]
    weights = [3, 3, 2, 2, 2, 1, 2, 2]
    for _ in range(count * 12):
        if len(out) >= count:
            break
        maker = rng.choices(pool, weights=weights)[0]
        made = maker()
        if made is None:
            continue
        kind, commands = made
        program = [*copy.deepcopy(commands), {"op": "STOP"}]
        key = json.dumps(program, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append({"kind": kind, "program": program})
    return out


# ------------------------------------------------------------ state evaluation

def evaluate_state(job):
    spec, step, count, rng_seed = job
    started = time.time()
    try:
        env, ctl, opponent, observation, diag = reconstruct(spec, step)
    except Exception as exc:                              # noqa: BLE001
        return {**{k: spec[k] for k in ("label", "program_id", "seed", "seat", "opponent")},
                "source": spec["kind"], "step": step, "error": repr(exc)[:300]}
    snapshot = ctl.snapshot()
    later = {s: p for s, p in spec["programs"].items() if s > step}
    rng = random.Random(rng_seed)
    proposals = [{"kind": "KEEP", "program": copy.deepcopy(STOP)}]
    if spec["kind"] == "campaign" and step in spec["programs"]:
        proposals.append({"kind": "RECORDED", "program": copy.deepcopy(spec["programs"][step])})
    proposals += gen_candidates(rng, ctl, observation, spec["seat"], step, count)

    results = []
    for entry in proposals:
        branch = dict(entry)
        try:
            fork = copy.deepcopy(ctl)
            proposal = fork.propose(observation, entry["program"])
            branch["refused"] = [{"reason": r["reason"].split(":")[0], "detail": r["reason"],
                                  "command": r["command"]} for r in proposal["rejected"]]
            fork = fork.publish(observation, proposal)
            own, other, fallbacks = play_from(clone_env(env), fork, copy.deepcopy(opponent), spec, later)
            branch.update({"cash": own, "opponent_cash": other, "later_publication_fallbacks": fallbacks})
        except Exception as exc:                          # noqa: BLE001
            branch.update({"cash": None, "error": repr(exc)[:200], "refused": branch.get("refused", [])})
        results.append(branch)

    keep = results[0]["cash"]
    for branch in results:
        branch["delta"] = None if (branch["cash"] is None or keep is None) else branch["cash"] - keep
    scored = [b for b in results[1:] if b["delta"] is not None]
    best = max(scored, key=lambda b: b["delta"]) if scored else None
    return {"schema": "farm-plan-repair-state-v1", "comparison": "PAIRED_SEED_SINGLE_FUTURE",
            "source": spec["kind"], "label": spec["label"], "program_id": spec["program_id"],
            "seed": spec["seed"], "seat": spec["seat"], "opponent": spec["opponent"], "step": step,
            "keep_cash": keep, "recorded_cash": spec["recorded_cash"],
            "reconstruction": diag, "candidates": results,
            "best": None if best is None else {"candidate_index": results.index(best), "kind": best["kind"], "delta": best["delta"],
                                               "cash": best["cash"], "opponent_cash": best["opponent_cash"],
                                               "program": best["program"],
                                               "refused": best["refused"]},
            "observation": observation, "snapshot": snapshot,
            "configuration": dict(env.configuration), "later_programs": later,
            "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "seconds": round(time.time() - started, 2)}


# ------------------------------------------------------------------- reporting

def summarize(states, threshold):
    good = [s for s in states if "error" not in s and s.get("best")]
    by_step, by_opponent, kinds = {}, {}, {}
    for state in good:
        by_step.setdefault(state["step"], []).append(state["best"]["delta"])
        by_opponent.setdefault(state["opponent"], []).append(state["best"]["delta"])
        if state["best"]["delta"] >= threshold:
            kinds[state["best"]["kind"]] = kinds.get(state["best"]["kind"], 0) + 1

    def block(groups):
        rows = {}
        for key, values in sorted(groups.items(), key=lambda kv: str(kv[0])):
            rows[str(key)] = {"n": len(values), "hit_fraction": sum(v >= threshold for v in values) / len(values),
                              "mean_best_delta": statistics.fmean(values),
                              "median_best_delta": statistics.median(values),
                              "max_best_delta": max(values)}
        return rows

    deltas = [s["best"]["delta"] for s in good]
    refusals = {}
    for state in good:
        for branch in state["candidates"]:
            for item in branch.get("refused", []):
                refusals[item["reason"]] = refusals.get(item["reason"], 0) + 1
    recorded = [b["delta"] for s in good for b in s["candidates"] if b["kind"] == "RECORDED" and b["delta"] is not None]
    return {"schema": "farm-plan-repair-summary-v1", "comparison": "PAIRED_SEED_SINGLE_FUTURE",
            "threshold": threshold, "states": len(states), "evaluated": len(good),
            "errored": sum(1 for s in states if "error" in s),
            "hit_fraction": (sum(d >= threshold for d in deltas) / len(deltas)) if deltas else 0.0,
            "mean_best_delta": statistics.fmean(deltas) if deltas else 0.0,
            "median_best_delta": statistics.median(deltas) if deltas else 0.0,
            "by_step": block(by_step), "by_opponent": block(by_opponent),
            "winning_kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
            "refused_commands_by_reason": refusals,
            "recorded_edit_delta_mean": statistics.fmean(recorded) if recorded else None,
            "seconds_per_state": statistics.fmean([s["seconds"] for s in good]) if good else 0.0}


def print_summary(tag, summary):
    print(f"\n=== {tag} === states {summary['evaluated']}/{summary['states']} "
          f"(errored {summary['errored']}) threshold ${summary['threshold']:,}")
    print(f"  some edit beats KEEP by >= threshold: {summary['hit_fraction']*100:.1f}%   "
          f"mean best Delta ${summary['mean_best_delta']:,.0f}   median ${summary['median_best_delta']:,.0f}   "
          f"{summary['seconds_per_state']:.1f} s/state")
    for name, key in (("by step", "by_step"), ("by opponent", "by_opponent")):
        print(f"  {name}:")
        print(f"    {'group':>14} {'n':>4} {'hit%':>7} {'mean':>12} {'median':>12} {'max':>12}")
        for group, row in summary[key].items():
            print(f"    {group:>14} {row['n']:>4} {row['hit_fraction']*100:>6.1f}% "
                  f"{row['mean_best_delta']:>12,.0f} {row['median_best_delta']:>12,.0f} {row['max_best_delta']:>12,.0f}")
    print(f"  winning edit kinds: {summary['winning_kinds']}")
    print(f"  refused commands by reason: {summary['refused_commands_by_reason']}")
    if summary["recorded_edit_delta_mean"] is not None:
        print(f"  student's own recorded edit vs KEEP: mean Delta ${summary['recorded_edit_delta_mean']:,.0f}")


def _outcome(branch):
    """Never join cash across branches or fabricate terminal labels on failure."""
    if branch.get("cash") is None:
        return None
    if branch.get("error"):
        raise ValueError("failed branch cannot carry terminal cash")
    values = [branch.get("cash"), branch.get("opponent_cash")]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("finite nonnegative own and opponent cash from one branch required")
    own, other = values
    return {"own_cash": own, "opponent_cash": other, "cash_margin": own-other,
            "wdl_score": 1. if own > other else .5 if own == other else 0.}


def _data_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def effect_record(state):
    """A complete all-candidate, actor-context record, independent of winner selection.

    A world group deliberately ignores seat, opponent, plan and phase: all such
    rows must stay in one data split. No hidden seed is an actor input. Old
    failures without identity remain explicitly ungroupable, not invented data.
    Caller registration must fix world splits BEFORE labels; this assigns none.
    """
    identity = {k: state.get(k) for k in ("source", "program_id", "seed", "seat", "opponent", "step", "label")}
    identified = type(identity["seed"]) is int and type(identity["seat"]) is int and identity["seat"] in (0, 1)
    failed = "error" in state
    actor_input = None if failed else {k: copy.deepcopy(state[k]) for k in ("observation", "snapshot", "configuration")}
    row = {"schema": "farm-plan-edit-effects-v1", "status": "RECONSTRUCTION_FAILED" if failed else "RECORDED",
           "identity": identity, "world_group": _data_hash({"world_seed": identity["seed"]}) if identified else None,
           "state_id": _data_hash({"identity": identity, "actor_input": actor_input}) if identified else None,
           "actor_input": actor_input, "independent_future_samples": 1 if not failed else 0,
           "comparison": "PAIRED_SEED_SINGLE_FUTURE_NOT_EXPECTATION",
           "opponent_continuation": ("RECORDED_OPEN_LOOP" if state.get("source") == "campaign" else
               "RESPONSIVE_ROUTER" if state.get("opponent") == "router" else
               "PASS" if state.get("opponent") == "pass" else "UNKNOWN"),
           "later_programs": copy.deepcopy(state.get("later_programs")),
           "later_programs_available": "later_programs" in state,
           "producer_sha256": state.get("producer_sha256"),
           "reconstruction": copy.deepcopy(state.get("reconstruction")),
           "candidates": [], "paired_terminal_candidates": 0}
    if failed:
        row["error"] = state["error"]
        return row
    candidates = state["candidates"]
    if not candidates or candidates[0].get("kind") != "KEEP" or candidates[0].get("program") != STOP:
        raise ValueError("first candidate must be explicit KEEP")
    keep = _outcome(candidates[0])
    for index, branch in enumerate(candidates):
        outcome = _outcome(branch)
        delta = {k: outcome[k]-keep[k] for k in keep} if outcome is not None and keep is not None else None
        if branch.get("delta") != (delta["own_cash"] if delta is not None else None):
            raise ValueError("saved own-cash delta disagrees with same-branch terminal labels")
        saved = {k: copy.deepcopy(branch[k]) for k in
                 ("kind", "program", "refused", "error", "later_publication_fallbacks") if k in branch}
        saved.update(candidate_index=index, program_sha256=_data_hash(branch["program"]),
                     status="TERMINAL" if outcome is not None else "FAILED",
                     outcome=outcome, delta_vs_keep=delta,
                     within_64_command_budget=len(branch["program"]) <= 64,
                     same_fallback_count_as_keep=(branch["later_publication_fallbacks"] ==
                         candidates[0]["later_publication_fallbacks"]) if outcome is not None and keep is not None
                         and "later_publication_fallbacks" in branch
                         and "later_publication_fallbacks" in candidates[0] else None)
        row["candidates"].append(saved)
        row["paired_terminal_candidates"] += int(delta is not None)
    return row


def _selected_branch(state):
    """Resolve a saved best summary back to exactly one real candidate."""
    best = state["best"]
    candidates = state["candidates"][1:]
    if "candidate_index" in best:
        index = best["candidate_index"]
        if type(index) is not int or not 1 <= index < len(state["candidates"]):
            raise ValueError("selected branch index is invalid")
        candidates = [state["candidates"][index]]
    matches = [b for b in candidates if all(b.get(k) == best.get(k)
               for k in ("kind", "program", "cash", "delta", "refused"))]
    if len(matches) != 1:
        raise ValueError("selected branch must join exactly one candidate")
    branch = matches[0]
    if _outcome(branch) is None:
        raise ValueError("selected branch requires a terminal outcome")
    if "opponent_cash" in best and best["opponent_cash"] != branch["opponent_cash"]:
        raise ValueError("selected branch opponent cash disagrees")
    return branch


def write_outputs(tag, states, threshold):
    """Validate before writing; never overwrite an earlier result or failed attempt."""
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError("output tag must be a plain filename stem")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("finite nonnegative repair threshold required")
    states_path = os.path.join(OUTDIR, tag + ".states.jsonl")
    repairs_path = os.path.join(OUTDIR, tag + ".repairs.jsonl.gz")
    effects_path = os.path.join(OUTDIR, tag + ".effects.jsonl.gz")
    summary_path = os.path.join(OUTDIR, tag + ".summary.json")
    for path in (states_path, repairs_path, effects_path, summary_path):
        if os.path.lexists(path):
            raise FileExistsError("preserve previous repair attempt: " + path)
    effects = [effect_record(state) for state in states]
    selected = []
    for state in states:
        best = state.get("best")
        if not best or best["delta"] < threshold:
            continue
        branch = _selected_branch(state)
        selected.append({"program_id": state["program_id"], "seed": state["seed"], "seat": state["seat"],
            "opponent": state["opponent"], "cash": [branch["cash"], branch["opponent_cash"]],
            "selection_objective": "OWN_CASH_NOT_WDL", "configuration": state["configuration"],
            "records": [{"step": state["step"], "observation": state["observation"],
                         "snapshot": state["snapshot"], "program": branch["program"]}]})
    summary = summarize(states, threshold)
    summary.update(repair_records=len(selected), states_file=states_path, repairs_file=repairs_path,
                   effects_file=effects_path, effect_states=len(effects),
                   selection_objective="OWN_CASH_NOT_WDL")
    os.makedirs(OUTDIR, exist_ok=True)
    with open(states_path, "x") as handle:
        for state in states:
            slim = {k: v for k, v in state.items() if k not in ("observation", "snapshot", "configuration")}
            handle.write(json.dumps(slim, allow_nan=False) + "\n")
    for path, rows in ((repairs_path, selected), (effects_path, effects)):
        with gzip.open(path, "xt") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
    with open(summary_path, "x") as handle:
        json.dump(summary, handle, indent=1, allow_nan=False)
    print_summary(tag, summary)
    print(f"  repair corpus: {len(selected)} records -> {repairs_path}")
    print(f"  all-candidate effect states: {len(effects)} -> {effects_path}")
    return summary



def audit_records(states, repairs):
    """Check legacy selected rows against retained branches; never guess missing data.

    This verifies the same-branch export join, not engine execution or historical
    source authenticity. Bulk corpora need this audit before reuse; missing
    candidate receipts are explicitly UNVERIFIABLE, not clean data.
    """
    def key(row, step, program):
        return _data_hash({"identity": {k: row[k] for k in
                           ("program_id", "seed", "seat", "opponent")},
                           "step": step, "program": program})

    index = {}
    source_errors = []
    for line, state in enumerate(states, 1):
        if "error" in state or not state.get("best"):
            continue
        try:
            selected = _selected_branch(state)
            group = key(state, state["step"], selected["program"])
            index.setdefault(group, []).append((state, selected))
        except (KeyError, TypeError, ValueError) as exc:
            source_errors.append({"state_line": line, "error": str(exc)[:250]})
    results = []
    for line, record in enumerate(repairs, 1):
        row = {"repair_line": line}
        try:
            if len(record["records"]) != 1:
                raise ValueError("selected repair must contain exactly one decision")
            decision = record["records"][0]
            group = key(record, decision["step"], decision["program"])
            matches = index.get(group, [])
            if len(matches) != 1:
                raise ValueError("selected state/branch join is missing or ambiguous")
            state, selected = matches[0]
            observed = record["cash"]
            if not isinstance(observed, list) or len(observed) != 2:
                raise ValueError("own-relative cash pair required")
            actual = _outcome({"cash": observed[0], "opponent_cash": observed[1]})
            if actual is None:
                raise ValueError("selected repair has no terminal cash")
            expected = _outcome(selected)
            correct = [expected["own_cash"], expected["opponent_cash"]]
            row.update(join_key=group, status="MATCH" if observed == correct else "MISMATCH",
                       saved_cash=observed, branch_cash=correct,
                       saved_wdl=actual["wdl_score"], branch_wdl=expected["wdl_score"],
                       wdl_changed=actual["wdl_score"] != expected["wdl_score"])
            if observed != correct:
                row["matches_legacy_cross_branch_bug"] = (observed[0] == correct[0] and
                    observed[1] == state["candidates"][0].get("opponent_cash") and observed[1] != correct[1])
        except (KeyError, TypeError, ValueError) as exc:
            row.update(status="UNVERIFIABLE", error=str(exc)[:250])
        results.append(row)
    counts = {status: sum(row["status"] == status for row in results)
              for status in ("MATCH", "MISMATCH", "UNVERIFIABLE")}
    return {"schema": "farm-repair-branch-audit-v1", "scope": "EXPORT_JOIN_ONLY_NOT_ENGINE_VERIFICATION",
            "counts": counts, "repair_records": len(results), "source_join_errors": source_errors,
            "wdl_label_changes": sum(row.get("wdl_changed", False) for row in results),
            "all_export_joins_valid": bool(results) and counts["MATCH"] == len(results) and not source_errors,
            "rows": results, "games_executed": 0, "data_modified": False}


def audit_files(states_path, repairs_path):
    """Read literal input bytes once so reported hashes bind exactly what was audited."""
    paths = [Path(states_path), Path(repairs_path)]
    payloads = [path.read_bytes() for path in paths]
    rows = []
    for path, payload in zip(paths, payloads):
        text = (gzip.decompress(payload) if path.suffix == ".gz" else payload).decode("utf-8")
        rows.append([json.loads(line) for line in text.splitlines() if line.strip()])
    result = audit_records(*rows)
    result["sources"] = {str(path): hashlib.sha256(payload).hexdigest()
                         for path, payload in zip(paths, payloads)}
    result["producer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def run(tag, specs, steps, count, workers, threshold):
    jobs = [(spec, step, count, hash((spec["seed"], spec["seat"], step)) & 0xFFFFFFFF)
            for spec in specs for step in steps]
    started = time.time()
    states = []
    with Pool(workers) as pool:
        for done, state in enumerate(pool.imap_unordered(evaluate_state, jobs, chunksize=1), 1):
            states.append(state)
            if done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} states  ({time.time()-started:.0f}s)", flush=True)
    states.sort(key=lambda s: (s["label"], s["step"]))
    print(f"  wall {time.time()-started:.0f}s for {len(jobs)} states on {workers} workers")
    return write_outputs(tag, states, threshold)


# --------------------------------------------------------------------- selftest

def _selftest_corpus(games=5):
    print("== corpus reconstruction (observation + controller snapshot exact match) ==")
    specs = corpus_specs(programs=games, seeds=1)
    obs_hits = snap_hits = total = 0
    for spec in specs:
        for step in (144, 288, 432, 576):
            _, ctl, _, observation, _ = reconstruct(spec, step)
            want = spec["recorded"][step]
            total += 1
            same_obs = json.loads(json.dumps(observation)) == want["observation"]
            got = json.loads(json.dumps(ctl.snapshot()))
            same_snap = got == want["snapshot"]
            obs_hits += same_obs
            snap_hits += same_snap
            if not (same_obs and same_snap):
                fields = [k for k in set(got) | set(want["snapshot"])
                          if got.get(k) != want["snapshot"].get(k)]
                print(f"   MISMATCH {spec['label']} step {step} obs={same_obs} snap={same_snap} fields={fields}")
        print(f"   {spec['label']}: ok through step 576")
    print(f"   observations {obs_hits}/{total} exact; controller snapshots {snap_hits}/{total} exact")
    return obs_hits, snap_hits, total


def _selftest_campaign(producer_dir, worlds=3):
    print("== campaign reconstruction (recorded joint actions -> RESULT.json cash) ==")
    ok = 0
    specs = campaign_specs(producer_dir, worlds=worlds)
    for spec in specs:
        env, ctl, _, _, diag = reconstruct(spec, len(spec["actions"]))
        farms = env.agent_observation(0)["farms"]
        cash = [float(farms[0]["money"]), float(farms[1]["money"])]
        match = cash == list(spec["recorded_cash"])
        ok += match
        print(f"   {spec['label']} seat {spec['seat']} seed {spec['seed']}: replay cash {cash} "
              f"vs RESULT {spec['recorded_cash']} -> {'MATCH' if match else 'DIFFER'}; "
              f"re-executed controller actions differing from the tape: {diag['action_mismatch']}/{diag['steps']}")
    return ok, len(specs)


def _selftest_clone():
    print("== environment fork determinism ==")
    specs = corpus_specs(programs=1, seeds=1)
    spec = specs[0]
    env, ctl, opponent, _, _ = reconstruct(spec, 288)
    a = play_from(clone_env(env), copy.deepcopy(ctl), copy.deepcopy(opponent), spec, {432: STOP, 576: STOP})
    b = play_from(clone_env(env), copy.deepcopy(ctl), copy.deepcopy(opponent), spec, {432: STOP, 576: STOP})
    c = play_from(env, ctl, opponent, spec, {432: STOP, 576: STOP})
    print(f"   fork1 {a[:2]} fork2 {b[:2]} original {c[:2]} -> {'IDENTICAL' if a == b == c else 'DIVERGED'}")
    print(f"   recorded terminal cash for the same game: {spec['recorded_cash']}")
    return a == b == c


def _selftest_semantics(step=144):
    """Why the generator is refuse-aware: what the executor refuses at a live state."""
    print("== edit refusal semantics at a live state ==")
    spec = corpus_specs(programs=1, seeds=1)[0]
    _, ctl, _, observation, _ = reconstruct(spec, step)
    rules, orders = ctl.executor.rule_stages, ctl.executor.order_stages

    def probe(name, program):
        try:
            proposal = copy.deepcopy(ctl).propose(observation, program)
            reasons = [r["reason"].split(";")[0] for r in proposal["rejected"]]
            print(f"   {name}: {reasons or 'ACCEPTED'}")
        except Exception as exc:                          # noqa: BLE001
            print(f"   {name}: RAISED {repr(exc)[:110]}")

    live = sorted((k, _start(r)) for k, rows in rules.items() for r in rows if _end(r) >= step)
    print(f"   live rule stages (key, start): {live}")
    for key, start in live:
        op = key.split(":")[0]
        if op == "HANDS":
            probe(f"HANDS start={step} (live stage starts at {start})",
                  [{"op": "HANDS", "target": 9, "start": step, "until": "GAME_END", "priority": 0}, {"op": "STOP"}])
            probe(f"HANDS start={start} (keyed onto the live stage)",
                  [{"op": "HANDS", "target": 9, "start": start, "until": "GAME_END", "priority": 0}, {"op": "STOP"}])
    tile = sorted(k for k, rows in orders.items() if any(_end(r) >= step for r in rows))
    if tile:
        x, y = (int(v) for v in tile[0].split(","))
        probe(f"SET_CROP on busy tile {tile[0]} start={step}",
              [{"op": "SET_CROP", "scope": {"kind": "TILE", "x": x, "y": y}, "crop": "WHEAT", "cycles": None,
                "harvest_threshold": 1, "watering": "DAILY", "fertilizing": "OFF",
                "start": step, "until": "GAME_END", "priority": 4}, {"op": "STOP"}])
    probe("past-only window (until < step)",
          [{"op": "HANDS", "target": 9, "start": 0, "until": step - 44, "priority": 0}, {"op": "STOP"}])
    free, locked = _free_tiles(observation, spec["seat"], orders, step)
    print(f"   tiles a start={step} stage can claim: {len(free)} unlocked, {len(locked)} still LOCKED")


def main(argv):
    mode = argv[1] if len(argv) > 1 else "selftest"
    args = {}
    for token in argv[2:]:
        key, _, value = token.partition("=")
        args[key.lstrip("-")] = value
    if mode == "audit":
        result = audit_files(args["states"], args["repairs"])
        with open(args["out"], "x") as handle:
            json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({k: result[k] for k in ("counts", "wdl_label_changes", "all_export_joins_valid")}))
        raise SystemExit(0 if result["all_export_joins_valid"] else 2)
    threshold = float(args.get("threshold", 2000))
    count = int(args.get("candidates", 8))
    workers = int(args.get("workers", 6))
    producer = args.get("producer", os.path.dirname(_SRC) + "/work/runs/full-economy-20260909/"
                                    "post-bc-selfplay-02/rounds/round-000/producers/p00")
    if mode == "selftest":
        _selftest_corpus(int(args.get("games", 5)))
        _selftest_campaign(producer, int(args.get("worlds", 3)))
        _selftest_clone()
        _selftest_semantics()
    elif mode == "corpus":
        steps = [int(s) for s in args.get("steps", "144,288,432,576").split(",")]
        specs = corpus_specs(programs=int(args.get("programs", 10)), seeds=int(args.get("seeds", 4)))
        print(f"corpus states: {len(specs)} games x {len(steps)} steps, {count} candidates each")
        run(args.get("tag", "corpus-v1"), specs, steps, count, workers, threshold)
    elif mode == "campaign":
        steps = [int(s) for s in args.get("steps", "144,288").split(",")]
        specs = campaign_specs(producer, int(args.get("worlds", 8)))
        print(f"campaign states: {len(specs)} worlds x {len(steps)} steps, {count} candidates each")
        run(args.get("tag", "campaign-r000-p00"), specs, steps, count, workers, threshold)
    elif mode == "semantics":
        _selftest_semantics(int(args.get("step", 144)))
    else:
        raise SystemExit("modes: selftest | semantics | corpus | campaign")


if __name__ == "__main__":
    main(sys.argv)
