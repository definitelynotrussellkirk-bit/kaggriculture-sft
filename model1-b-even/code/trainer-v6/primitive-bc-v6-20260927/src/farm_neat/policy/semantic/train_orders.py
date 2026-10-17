"""Bounded self-play for the small command interface; reuses exact engine/PPO.

Not a canonical20k continuation or a new submission. Every fit follows a second
fresh construction of all its games. Compact action/decision/state records stay
readable; no historical raw dependency is added to optimizer continuation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
from farm_neat.training.schedule_model import bound, canonical, file_sha
from farm_neat.training.schedule import default_settings
from farm_neat.policy.semantic import policy_v1 as binding
from farm_neat.economy.orders_v1 import SemanticOrders, digest


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def fixed_opponent():
    orders = binding.initial_orders()
    orders.farm_tile((4, 3), cycles=3)
    orders.farm_tile((3, 3), cycles=3)
    orders.keep_animal((3, 4))
    return orders


def play(job):
    policies, models = [], []
    for model in job["models"]:
        if model == "FIXED_COMMANDS":
            policies.append(None); models.append("FIXED_COMMANDS")
        else:
            policies.append(binding.Policy(*binding.load_model(model)))
            models.append(file_sha(model))
    env = ExactKaggricultureEnvironment(seed=job["seed"])
    controllers = [binding.initial_orders() if policy else fixed_opponent() for policy in policies]
    rngs = [np.random.default_rng(job["rng_seed"] + s) for s in range(2)]
    episodes = [{"episode_id": f'{job["id"]}-seat{s}', "seed": job["seed"], "seat": s,
        "behavior_model_sha256": models[s],
        "rollout_iteration": policies[s].manifest["rollout_iteration"] + 1 if policies[s] else 0,
        "calls": []} for s in range(2)]
    record = {"schema": "farm-semantic-orders-physical-game-v1", "job": copy.deepcopy(job),
              "model_sha256": models, "decisions": [[], []], "turns": []}
    started = time.monotonic()
    while not env.done:
        joint = []
        for seat in range(2):
            obs = env.agent_observation(seat)
            controller, policy = controllers[seat], policies[seat]
            if controller.pending:
                controller.observe(obs)
            if obs["hour"] == 0 and policy is not None:
                rows, mask = binding.pack(obs, controller)
                _, value, _, probs, logp = policy.distribution(rows, mask)
                index = (int(rngs[seat].choice(len(probs), p=probs)) if job["sample"]
                         else int(np.argmax(probs)))
                call = {"candidate_contexts": rows, "mask": mask, "temperature": policy.manifest["temperature"],
                        "action_index": index, "old_log_probability": float(logp[index]), "old_value": value}
                episodes[seat]["calls"].append(call)
                record["decisions"][seat].append({"step": obs["step"], "index": index,
                    "command": binding.ACTIONS[index], "observation_sha256": digest(obs),
                    "candidate_sha256": digest({"rows": rows.tolist(), "mask": mask.tolist()}),
                    "log_probability": float(logp[index]), "value": value,
                    "selection_mode": "SAMPLED" if job["sample"] else "GREEDY"})
                binding.apply_command(controller, index)
            joint.append(controller.act(obs, env.configuration))
        env.step(joint)
        successor = [env.agent_observation(s) for s in range(2)]
        record["turns"].append({"joint_action": joint, "successor_sha256": digest(successor)})
    cash = [float(env.agent_observation(s)["farms"][s]["money"]) for s in range(2)]
    for seat in range(2):
        controllers[seat].observe(env.agent_observation(seat))
        episodes[seat]["terminal_reward"] = int(np.sign(cash[seat] - cash[1-seat]))
    record.update(terminal_cash=cash, terminal_steps=env.step_index,
                  final_orders=[c.snapshot() for c in controllers])
    if env.step_index != 719 or any(len(ep["calls"]) != 30 for ep, policy in zip(episodes, policies) if policy):
        raise ValueError("full physical game and30 natural calls required")
    return {"record": record, "episodes": episodes, "elapsed_seconds": time.monotonic()-started}


def pool_play(jobs, workers):
    # Separate pool invocations for producer/reference, not flags on supplied data.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(play, jobs))


def verify_pair(produced, reconstructed):
    if produced["record"] != reconstructed["record"]:
        raise ValueError("fresh complete-game reconstruction mismatch; no fitting")
    return digest(produced["record"])


def episode_json(episode):
    result = {**episode, "calls": []}
    for call in episode["calls"]:
        result["calls"].append({**call, "candidate_contexts": call["candidate_contexts"].tolist(),
                                "mask": call["mask"].tolist()})
    return result


def evaluate(model, initial, output, *, workers, seeds):
    jobs = []
    for opponent_name, opponent in (("initial_nn", str(initial)), ("fixed_commands", "FIXED_COMMANDS")):
        for seed in seeds:
            for seat in (0, 1):
                models = [opponent, opponent]; models[seat] = str(model)
                jobs.append({"id": f"eval-{opponent_name}-{seed}-{seat}", "seed": seed,
                             "models": models, "rng_seed": 1, "sample": False, "seat": seat,
                             "opponent": opponent_name})
    p = pool_play(jobs, workers); r = pool_play(jobs, workers)
    conditions = []
    for job, produced, reference in zip(jobs, p, r):
        identity = verify_pair(produced, reference)
        s = job["seat"]; cash = produced["record"]["terminal_cash"]
        conditions.append({"seed": job["seed"], "seat": s, "opponent": job["opponent"],
            "cash": cash[s], "opponent_cash": cash[1-s], "wdl": int(np.sign(cash[s]-cash[1-s])),
            "record_sha256": identity})
    value = bound({"schema": "farm-semantic-orders-development-panel-v1", "model_sha256": file_sha(model),
        "conditions": conditions, "physical_games": len(p), "fresh_reference_games": len(r),
        "score": sum((c["wdl"]+1)/2 for c in conditions)/len(conditions),
        "wins": sum(c["wdl"] == 1 for c in conditions), "draws": sum(c["wdl"] == 0 for c in conditions),
        "losses": sum(c["wdl"] == -1 for c in conditions),
        "scope": "REPEATED_DEVELOPMENT_SMALL_COMMAND_CURRICULUM_NOT_LADDER_OR_HELDOUT"})
    write(output, value)
    return value


def run(args):
    root = Path(args.output).resolve()
    code = {**binding.sources(), "src/farm_neat/policy/semantic/train_orders.py": file_sha(__file__)}
    plan = bound({"schema": "farm-semantic-orders-selfplay-plan-v1", "source_pins": code,
        "games": args.games, "batch_size": args.batch_size, "workers": args.workers,
        "train_seed_base": args.seed_base, "evaluation_seeds": list(range(args.eval_seed_base, args.eval_seed_base+args.eval_seeds)),
        "evaluate_every": args.evaluate_every, "action_schema": binding.ACTION_SCHEMA,
        "settings": {**default_settings(100000), "epochs": 1, "minibatch_size": 256, "warmup_steps": 32,
                     "entropy_coefficient": .01, "seed": 908083001},
        "initialization_seed": 908083002, "comparison": "FROZEN_INITIAL_NN_AND_FIXED_COMMANDS_BOTH_SEATS",
        "game_objective": "TERMINAL_BANK_CASH_WIN_DRAW_LOSS", "canonical_parent_unchanged": True})
    if root.exists():
        if not args.resume or json.loads((root/"PLAN.json").read_text()) != plan:
            raise ValueError("new directory or exact source/plan-matching explicit resume required")
    else:
        write(root/"PLAN.json", plan)
        binding.initialize(root/"initial.npz", plan["initialization_seed"])
    initial = root/"initial.npz"
    model, optimizer, done, rounds = initial, None, 0, 0
    for path in sorted(root.glob("round-*/COMPLETED.json")):
        receipt = json.loads(path.read_text())
        if receipt != bound(receipt) or receipt["parent_model_sha256"] != file_sha(model):
            raise ValueError("semantic continuation chain mismatch")
        model, optimizer = path.parent/"model.npz", path.parent/"optimizer.pt"
        if file_sha(model) != receipt["model_sha256"] or file_sha(optimizer) != receipt["optimizer_sha256"]:
            raise ValueError("semantic continuation state changed")
        done += receipt["physical_games"]; rounds += 1
    if not (root/"evaluation-initial.json").exists():
        result = evaluate(initial, initial, root/"evaluation-initial.json", workers=args.workers,
                          seeds=plan["evaluation_seeds"])
        print(canonical({"phase": "INITIAL_EVALUATION", "score": result["score"]}), flush=True)
    started = time.monotonic(); invocation_start = done
    while done < args.games:
        directory = root/f"round-{rounds:05d}"
        if directory.exists():
            raise ValueError("partial round preserved: inspect it before any retry")
        count = min(args.batch_size, args.games - done)
        jobs = [{"id": f"train-{done+i:07d}", "seed": args.seed_base+done+i,
            "models": [str(model), str(model)], "sample": True,
            "rng_seed": 908100000+2*(done+i)} for i in range(count)]
        write(directory/"JOBS.json", jobs)
        phase_start = time.monotonic()
        produced = pool_play(jobs, args.workers)
        producer_seconds = time.monotonic()-phase_start
        phase_start = time.monotonic()
        references = pool_play(jobs, args.workers)
        reference_seconds = time.monotonic()-phase_start
        verified, episodes = [], []
        for job, p, r in zip(jobs, produced, references):
            identity = verify_pair(p, r)
            write(directory/"games"/(job["id"]+".json"), p["record"])
            verified.append({"game_id": job["id"], "producer_sha256": identity, "fresh_reference_sha256": identity})
            episodes.extend(episode_json(ep) for ep in p["episodes"])
        write(directory/"WITNESSES.json", verified)
        del references, produced
        before = file_sha(model)
        phase_start = time.monotonic()
        result = binding.fit(model, episodes, directory/"model.npz", optimizer_path=optimizer,
            output_optimizer=directory/"optimizer.pt", settings=plan["settings"])
        fit_seconds = time.monotonic()-phase_start
        write(directory/"FIT.json", result)
        numerical = result["numerical_fit"]
        receipt = bound({"schema": "farm-semantic-orders-round-v1", "round": rounds,
            "physical_games": count, "fresh_reference_games": count,
            "optimizer_steps": numerical["optimizer_steps"], "model_sha256": file_sha(directory/"model.npz"),
            "optimizer_sha256": file_sha(directory/"optimizer.pt"), "parent_model_sha256": before,
            "producer_seconds": producer_seconds, "reference_seconds": reference_seconds,
            "fit_seconds": fit_seconds, "plan_sha256": plan["content_sha256"],
            "fit_sha256": file_sha(directory/"FIT.json"), "witnesses_sha256": file_sha(directory/"WITNESSES.json")})
        write(directory/"COMPLETED.json", receipt)
        model, optimizer = directory/"model.npz", directory/"optimizer.pt"
        done += count; rounds += 1
        print(canonical({"phase": "COMMITTED", "physical_games": done,
            "global_optimizer_steps": numerical["global_optimizer_steps_after"],
            "new_games_per_second_including_fit_and_intermediate_evaluation": (done-invocation_start)/(time.monotonic()-started),
            "changed_choices_on_batch": numerical["changed_greedy_choices_on_rollout"],
            "entropy": numerical["final_entropy"], "value_mse": numerical["final_value_mse"],
            "model_sha256": receipt["model_sha256"]}), flush=True)
        if done % args.evaluate_every == 0 or done == args.games:
            evaluation = evaluate(model, initial, root/f"evaluation-{done:07d}.json", workers=args.workers,
                                  seeds=plan["evaluation_seeds"])
            print(canonical({"phase": "DEVELOPMENT_EVALUATION", "physical_training_games": done,
                             "score": evaluation["score"], "wins": evaluation["wins"],
                             "draws": evaluation["draws"], "losses": evaluation["losses"]}), flush=True)
    report = bound({"schema": "farm-semantic-orders-run-result-v1", "status": "BUDGET_COMPLETED",
        "physical_training_games": done, "fresh_training_reference_games": done,
        "new_games_this_invocation": done-invocation_start, "elapsed_this_invocation_seconds": time.monotonic()-started,
        "initial_model_sha256": file_sha(initial), "model": str(model.relative_to(root)),
        "model_sha256": file_sha(model), "optimizer": str(optimizer.relative_to(root)),
        "global_optimizer_steps": binding.load_model(model)[0]["optimizer_steps"],
        "canonical_model_changed": False, "submission": None,
        "strength_scope": "SMALL_COMMAND_CURRICULUM_REPEATED_DEV_NOT_FULL_STRATEGY_COVERAGE"})
    if not (root/"RESULT.json").exists():
        write(root/"RESULT.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=908083000)
    parser.add_argument("--eval-seed-base", type=int, default=908090000)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--evaluate-every", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (not 1 <= args.games <= 1024 or not 1 <= args.batch_size <= 64 or not 1 <= args.workers <= 4
            or not 1 <= args.eval_seeds <= 32 or args.evaluate_every < 1
            or args.seed_base < 0 or args.eval_seed_base < 0
            or set(range(args.seed_base, args.seed_base+args.games)) &
               set(range(args.eval_seed_base, args.eval_seed_base+args.eval_seeds))
            or any(941000000 <= s < 942000000 or s == 5601011
                   for s in (*range(args.seed_base, args.seed_base+args.games),
                             *range(args.eval_seed_base, args.eval_seed_base+args.eval_seeds)))):
        parser.error("bounded positive counts and disjoint nonprotected TRAIN/DEV ranges required")
    print(canonical(run(args)), flush=True)


if __name__ == "__main__":
    main()
