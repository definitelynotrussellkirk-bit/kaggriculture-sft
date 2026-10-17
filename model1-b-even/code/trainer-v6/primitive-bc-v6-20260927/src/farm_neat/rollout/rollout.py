"""Full-economy producer capture, fresh engine reconstruction and PPO admission.

Engine reconstruction executes the recorded joint economic programs in a NEW
exact environment. It is not a responsive counterfactual opponent experiment.
Admission additionally loads the exact frozen model and reproduces its sampled
factor paths, masks, logps and critic values on the original two-seat batch shape.
No old29-row episode, incomplete game or caller-supplied 'verified' flag suffices.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
from farm_neat.economy import action_space as actions
from farm_neat.policy import observation as features
from farm_neat.economy.orders import EconomicOrders

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "farm-full-economic-rollout-v1"
PLAN_SCHEMA, GAME_SCHEMA = SCHEMA + "-plan", SCHEMA + "-game"
EPISODE_SCHEMA, ADMISSION_SCHEMA = SCHEMA + "-episode", SCHEMA + "-admission"
STEPS = tuple(range(0, 719, 24))
ATOL = 2e-5
digest = actions.digest


def policy_module():
    from farm_neat.policy import network as full_economic_policy
    return full_economic_policy


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources():
    names = ("src/farm_neat/rollout/rollout.py", "src/farm_neat/economy/orders.py",
        "src/farm_neat/economy/orders_v1.py", "src/farm_neat/game/exact_engine.py",
        "src/farm_neat/game/kaggriculture_engine_reference.py", "src/farm_neat/game/kaggriculture.json",
        "src/farm_neat/economy/heuristic/policy.py", "src/farm_neat/economy/heuristic/state.py", "src/farm_neat/economy/heuristic/constants.py", "src/farm_neat/economy/heuristic/econ.py",
        "src/farm_neat/economy/heuristic/market.py", "src/farm_neat/economy/heuristic/valuation.py", "src/farm_neat/economy/heuristic/animal_dp.py",
        "src/farm_neat/economy/heuristic/search.py", "src/farm_neat/game/macro.py")
    return {**policy_module().sources(), **{n: file_sha(ROOT/n) for n in names}}


def attest(pins):
    if not isinstance(pins, dict) or any(pins.get(k) != v for k, v in sources().items()):
        raise ValueError("required full-economy source closure changed or missing")
    for name, expected in pins.items():
        relative = Path(name)
        path = ROOT/relative
        if (relative.is_absolute() or ".." in relative.parts or str(relative) != name
                or any(p.is_symlink() for p in (path, *path.parents)) or file_sha(path) != expected):
            raise ValueError("declared source path or bytes changed")


def bound(value):
    result = {k: copy.deepcopy(v) for k, v in value.items() if k != "content_sha256"}
    return {**result, "content_sha256": digest(result)}


def _equal(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(label + " mismatch")


def _sealed(record):
    if not isinstance(record, dict) or record != bound(record):
        raise ValueError("content-bound full-economy receipt required")


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("explicit SHA256 required")
    return value


def _finite(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("finite behavior output required")
    return float(value)


def make_plan(manifest, model_sha256, *, seed, actor_rng_seeds, command_budget,
              game_id, environment, source_pins=None):
    """Register before collection; no checkpoint loading/model allocation here."""
    policy = policy_module()
    _sealed(manifest)
    _equal(manifest["schema"], policy.SCHEMA, "model schema")
    _equal(manifest["architecture"], json.loads(json.dumps(policy.architecture())), "intended full model")
    _equal(manifest["source_pins"], policy.sources(), "model sources")
    actions._integer(seed, label="physical seed")
    if (not isinstance(actor_rng_seeds, list) or len(actor_rng_seeds) != 2
            or any(type(v) is not int or v < 0 for v in actor_rng_seeds)
            or len(set(actor_rng_seeds)) != 2 or not isinstance(game_id, str) or not game_id):
        raise ValueError("unique two-seat sampler streams and game identity required")
    actions._integer(command_budget, 2, actions.MAX_COMMANDS, "command budget")
    pins = sources() if source_pins is None else source_pins
    attest(pins)
    if (environment.get("dtype") != "FLOAT32" or environment.get("tf32") is not False
            or environment.get("autocast") is not False):
        raise ValueError("explicit FP32 no-TF32/no-autocast producer profile required")
    return bound({"schema": PLAN_SCHEMA, "mode": "FRESH_SAMPLED_FULL_ECONOMIC_SELFPLAY",
        "game_id": game_id, "seed": seed, "actor_rng_seeds": actor_rng_seeds,
        "command_budget": command_budget, "temperature": 1.0,
        "model_sha256": _sha(model_sha256), "model_manifest_sha256": manifest["content_sha256"],
        "model_optimizer_age": manifest["optimizer_steps"], "rollout_iteration": manifest["rollout_iteration"]+1,
        "architecture_sha256": manifest["architecture"]["content_sha256"], "numerics": policy.NUMERICS,
        "source_pins": pins, "source_sha256": digest(pins), "environment": environment,
        "decision_steps": list(STEPS), "physical_steps": 719,
        "neural_batch": "BOTH_SEATS_TOGETHER_DYNAMIC_PADDING_B2",
        "terminal_training_admission": False})


def _validate_plan(plan):
    _sealed(plan)
    policy = policy_module()
    _equal(plan["schema"], PLAN_SCHEMA, "new full-economy plan")
    _equal(plan["mode"], "FRESH_SAMPLED_FULL_ECONOMIC_SELFPLAY", "sampled selfplay mode")
    _equal(plan["numerics"], policy.NUMERICS, "full-economy numerics")
    _equal(plan["architecture_sha256"], policy.architecture()["content_sha256"], "intended full architecture")
    _equal(plan["source_sha256"], digest(plan["source_pins"]), "rollout source identity")
    _equal(plan["decision_steps"], list(STEPS), "all30 decision steps")
    _equal(plan["physical_steps"], 719, "full season")
    _equal(plan["temperature"], 1.0, "sampled temperature")
    _equal(plan["neural_batch"], "BOTH_SEATS_TOGETHER_DYNAMIC_PADDING_B2", "numerical batch shape")
    _equal(plan["terminal_training_admission"], False, "producer is not admission")
    _sha(plan["model_sha256"]); _sha(plan["model_manifest_sha256"])
    actions._integer(plan["seed"], label="physical seed")
    actions._integer(plan["model_optimizer_age"], label="model age")
    actions._integer(plan["rollout_iteration"], 1, actions.MAX_INTEGER, "rollout age")
    actions._integer(plan["command_budget"], 2, actions.MAX_COMMANDS, "command budget")
    rng = plan["actor_rng_seeds"]
    if not isinstance(rng, list) or len(rng) != 2 or len(set(rng)) != 2 or any(type(v) is not int or v < 0 for v in rng):
        raise ValueError("separate exact actor RNG streams required")
    attest(plan["source_pins"])


def _trace(record, plan, *, step=None):
    policy = policy_module()
    as_of_step = policy.trace_as_of(record)
    if plan.get("full_plan"):
        actions._integer(step, 0, 718, "actual decision step")
        if as_of_step is None or as_of_step != step:
            raise ValueError("sampling as_of_step must match the actual decision step")
    elif as_of_step is not None:
        raise ValueError("remaining-window support requires an explicit full-plan execution")
    for key, expected in (("architecture_sha256", plan["architecture_sha256"]),
            ("action_schema_sha256", actions.ACTION_SCHEMA_SHA256), ("temperature", plan["temperature"]),
            ("max_commands", plan["command_budget"]), ("training_admission", False)):
        _equal(record[key], expected, "behavior " + key)
    program = actions.validate_program(record["program"])
    if (len(program) > plan["command_budget"] or len(record["commands"]) != len(program)
            or len(record["sampling_masks"]) != len(program)):
        raise ValueError("factor/program/budget cardinality mismatch")
    count = 0
    for slot, (command, encoded, masks) in enumerate(zip(program, record["commands"], record["sampling_masks"], strict=True)):
        _equal(actions.decode_command(encoded), command, "decoded actual program")
        if len(masks) != len(encoded["factors"]):
            raise ValueError("one syntax mask per conditional factor required")
        for factor, mask in zip(encoded["factors"], masks, strict=True):
            expected = policy.sampling_mask(factor["path"], factor["choices"],
                stop_only=slot == plan["command_budget"]-1 and factor["path"] == "op", as_of_step=as_of_step)
            if not isinstance(mask, list) or any(type(v) is not bool for v in mask) or mask != expected or not mask[factor["index"]]:
                raise ValueError("actual conditional syntax/budget mask mismatch")
            count += 1
    old = record["selected_logps"]
    if (not isinstance(old, list) or len(old) != count or any(_finite(v) > 1e-6 for v in old)
            or not math.isclose(math.fsum(old), _finite(record["joint_logp"]), rel_tol=0, abs_tol=1e-8)):
        raise ValueError("stored joint logp must sum all active conditional logps")
    return record


def pack_wire(packed):
    features.validate(packed)
    return {k: v.tolist() if isinstance(v, np.ndarray) else list(v) if k == "token_ids" else copy.deepcopy(v)
            for k, v in packed.items()}


def unpack(wire):
    packed = copy.deepcopy(wire)
    for key, dtype in (("rows", np.float32), ("knownness", np.bool_), ("token_mask", np.bool_)):
        raw = np.asarray(packed[key])
        if (key != "rows" and raw.dtype != np.bool_) or (key == "rows" and not np.array_equal(raw, raw.astype(np.float32))):
            raise ValueError("exact FP32 features and boolean masks required")
        packed[key] = raw.astype(dtype)
    packed["token_ids"] = tuple(packed["token_ids"])
    return features.validate(packed)


def capture_decision(plan, step, seat, packed, behavior, value, orders_snapshot):
    """Called on the actual PRE-edit pack and critic, before applying commands."""
    if type(step) is not int or step not in STEPS or type(seat) is not int or seat not in (0, 1):
        raise ValueError("natural daily step and seat required")
    _trace(behavior, plan)
    return bound({"schema": SCHEMA+"-decision", "step": step, "seat": seat,
        "input": pack_wire(packed), "orders_sha256": digest(orders_snapshot),
        "behavior": copy.deepcopy(behavior), "old_log_probability": behavior["joint_logp"],
        "old_value": _finite(value)})


def observation_sha(observation):
    return digest({"observation": observation,
                   "inventory_key_order": [list(i) for i in observation["private"]["inventories"]]})


def capture_turn(step, observations, programs, decisions, actions_taken, successor_observations,
                 controllers, rejected_edits):
    return {"schema": SCHEMA+"-turn", "step": step, "observations": copy.deepcopy(observations),
        "inventory_key_order": [[list(i) for i in obs["private"]["inventories"]] for obs in observations],
        "programs": copy.deepcopy(programs), "decisions": copy.deepcopy(decisions),
        "actions": copy.deepcopy(actions_taken), "rejected_edits": copy.deepcopy(rejected_edits),
        "successor_observation_sha256": [observation_sha(obs) for obs in successor_observations],
        "issued_orders_sha256": [digest(c.snapshot()) for c in controllers]}


def finish(plan, env, controllers, *, turns_sha256):
    if not env.done or env.step_index != 719 or [r.status for r in env.state] != ["DONE", "DONE"]:
        raise ValueError("no terminal labels from censored or failed games")
    terminal = [env.agent_observation(s) for s in (0, 1)]
    for seat in (0, 1):
        controllers[seat].observe(terminal[seat])
    cash = [terminal[0]["farms"][s]["money"] for s in (0, 1)]
    if any(_finite(v) < 0 for v in cash):
        raise ValueError("invalid terminal cash")
    return bound({"schema": GAME_SCHEMA, "status": "COMPLETE_PENDING_INDEPENDENT_ADMISSION",
        "plan_sha256": plan["content_sha256"], "model_sha256": plan["model_sha256"],
        "source_pins": plan["source_pins"], "turns_sha256": _sha(turns_sha256),
        "physical_games": 1, "physical_steps": 719, "model_decisions": 60, "optimizer_updates": 0,
        "cash": cash, "terminal_observation_sha256": [observation_sha(o) for o in terminal],
        "terminal_orders_sha256": [digest(c.snapshot()) for c in controllers],
        "terminal_training_admission": False, "responsive_reference_games": 0})


def _restore_observation(obs, orders):
    result = copy.deepcopy(obs)
    invs = result["private"]["inventories"]
    if len(invs) != len(orders):
        raise ValueError("inventory order cardinality")
    for i, order in enumerate(orders):
        if len(order) != len(set(order)) or set(order) != set(invs[i]):
            raise ValueError("inventory insertion-order mismatch")
        invs[i] = {k: invs[i][k] for k in order}
    return result


def _reconstruct(plan, result, rows):
    _validate_plan(plan); _sealed(result)
    if plan["schema"] != PLAN_SCHEMA or result["schema"] != GAME_SCHEMA:
        raise ValueError("new full-economy plan/game schema required")
    _equal(result["plan_sha256"], plan["content_sha256"], "game plan")
    _equal(result["status"], "COMPLETE_PENDING_INDEPENDENT_ADMISSION", "complete producer")
    _equal(plan["decision_steps"], list(STEPS), "all30 decision steps")
    _equal(plan["physical_steps"], 719, "full season")
    attest(plan["source_pins"])
    env = ExactKaggricultureEnvironment(seed=plan["seed"])
    controllers = [EconomicOrders(), EconomicOrders()]
    episodes = [{"schema": EPISODE_SCHEMA, "episode_id": plan["game_id"]+f"-seat{s}",
        "seed": plan["seed"], "seat": s, "behavior_model_sha256": plan["model_sha256"],
        "model_manifest_sha256": plan["model_manifest_sha256"], "numerics": plan["numerics"],
        "rollout_iteration": plan["rollout_iteration"], "source_sha256": plan["source_sha256"],
        "calls": []} for s in (0, 1)]
    count = 0
    for step, row in enumerate(rows):
        if step >= 719 or env.done:
            raise ValueError("extra physical suffix after terminal")
        _equal(row["step"], step, "consecutive physical step")
        _equal(row["schema"], SCHEMA+"-turn", "producer turn schema")
        obs = [env.agent_observation(s) for s in (0, 1)]
        _equal(len(row["observations"]), 2, "both actor views")
        _equal(len(row["inventory_key_order"]), 2, "both inventory orders")
        for s in (0, 1):
            recorded = _restore_observation(row["observations"][s], row["inventory_key_order"][s])
            _equal(observation_sha(recorded), observation_sha(obs[s]), "fresh actor observation")
            controllers[s].observe(obs[s])
        rejected = []
        if step in STEPS:
            if not isinstance(row["decisions"], list) or len(row["decisions"]) != 2 or len(row["programs"]) != 2:
                raise ValueError("both complete actual neural decisions required")
            for s, call in enumerate(row["decisions"]):
                _sealed(call)
                _equal(call["schema"], SCHEMA+"-decision", "decision schema")
                _equal([call["step"],call["seat"]], [step,s], "decision identity")
                packed = features.pack(obs[s], controllers[s], env.configuration)
                _equal(unpack(call["input"])["content_sha256"], packed["content_sha256"], "fresh actor-safe full input")
                _equal(call["orders_sha256"], digest(controllers[s].snapshot()), "pre-edit commitments")
                _trace(call["behavior"], plan)
                _equal(row["programs"][s], call["behavior"], "actual sampled program")
                _equal(call["old_log_probability"], call["behavior"]["joint_logp"], "emitted joint logp")
                _finite(call["old_value"])
                episodes[s]["calls"].append(copy.deepcopy(call))
                for command in call["behavior"]["program"]:
                    try:
                        controllers[s].apply(command)
                    except ValueError as exc:
                        rejected.append({"step":step,"seat":s,"command":command,"reason":str(exc)})
        elif row["decisions"] is not None or row["programs"] is not None:
            raise ValueError("unregistered extra neural decision")
        _equal(row["rejected_edits"], rejected, "actual rejected edits")
        joint = [controllers[s](obs[s], env.configuration) for s in (0, 1)]
        _equal(row["actions"], joint, "lowered physical actions")
        _equal(row["issued_orders_sha256"], [digest(c.snapshot()) for c in controllers], "issued commitments")
        env.step(joint)
        _equal(row["successor_observation_sha256"], [observation_sha(env.agent_observation(s)) for s in (0,1)], "fresh successors")
        count += 1
    if count != 719 or any([c["step"] for c in e["calls"]] != list(STEPS) for e in episodes):
        raise ValueError("all719 turns and30 decisions per seat required")
    expected = finish(plan, env, controllers, turns_sha256=result["turns_sha256"])
    _equal(result, expected, "complete terminal/model/source/trajectory result")
    for s, episode in enumerate(episodes):
        delta = result["cash"][s]-result["cash"][1-s]
        episode["terminal_reward"] = int(delta > 0)-int(delta < 0)
    attest(plan["source_pins"])
    return {"episodes": episodes, "engine_witness": bound({"schema":SCHEMA+"-engine-witness",
        "plan_sha256":plan["content_sha256"], "producer_result_sha256":result["content_sha256"],
        "turns_sha256":result["turns_sha256"], "training_batch_sha256":digest({"episodes":episodes}),
        "reconstructed_physical_games":1, "responsive_reference_games":0,
        "physical_steps":719, "decision_calls":60, "terminal_cash":result["cash"],
        "method":"FRESH_ENGINE_RECORDED_PROGRAMS_EXACT_EXECUTOR_INPUTS_ACTIONS_SUCCESSORS",
        "terminal_training_admission":False})}


def witness(folder, *, expected_plan_file_sha256):
    """Read-only CPU reconstruction, never loads a neural checkpoint."""
    folder = Path(folder)
    plan_bytes = (folder/"PLAN.json").read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != _sha(expected_plan_file_sha256):
        raise ValueError("externally pinned producer plan file mismatch")
    plan = json.loads(plan_bytes)
    result_bytes = (folder/"RESULT.json").read_bytes()
    result_sha = hashlib.sha256(result_bytes).hexdigest()
    result = json.loads(result_bytes)
    before = file_sha(folder/"turns.jsonl.gz")
    _equal(before, result["turns_sha256"], "exact producer payload")
    with gzip.open(folder/"turns.jsonl.gz", "rt") as stream:
        reconstructed = _reconstruct(plan, result, (json.loads(line) for line in stream))
    _equal(file_sha(folder/"PLAN.json"), expected_plan_file_sha256, "unchanged producer plan")
    _equal(file_sha(folder/"RESULT.json"), result_sha, "unchanged producer result")
    _equal(file_sha(folder/"turns.jsonl.gz"), before, "unchanged producer payload")
    return {"plan":plan, **reconstructed}


def _verify_graph(graph, plan, episodes):
    import torch
    device = next(graph.parameters()).device
    generators = [torch.Generator(device=device).manual_seed(v) for v in plan["actor_rng_seeds"]]
    graph.eval()
    max_logp, max_value = 0.0, 0.0
    with torch.no_grad():
        for ordinal, step in enumerate(STEPS):
            calls = [e["calls"][ordinal] for e in episodes]
            packed = features.collate([unpack(c["input"]) for c in calls])
            tensor = [torch.as_tensor(packed[k],device=device) for k in ("rows","token_mask","knownness")]
            context, values, _ = graph.encode(*tensor)
            behavior = [c["behavior"] for c in calls]
            logp, _ = graph.replay(context, behavior)
            for seat, call in enumerate(calls):
                sampled = graph.sample(context[seat:seat+1], generator=generators[seat],
                    max_commands=plan["command_budget"], temperature=plan["temperature"])[0][0]
                for key in ("schema","architecture_sha256","action_schema_sha256","temperature",
                            "max_commands","program","commands","sampling_masks","training_admission"):
                    _equal(sampled[key], behavior[seat][key], "fresh exact sampler "+key)
                errors = [abs(_finite(a)-_finite(b)) for a,b in zip(sampled["selected_logps"],behavior[seat]["selected_logps"],strict=True)]
                logp_error = max([abs(float(logp[seat])-call["old_log_probability"]),
                    abs(sampled["joint_logp"]-call["old_log_probability"]),*errors])
                value_error = abs(_finite(float(values[seat]))-call["old_value"])
                if not math.isfinite(logp_error) or max(logp_error, value_error) > ATOL:
                    raise ValueError("actual frozen model logp/value numerical replay mismatch")
                max_logp,max_value = max(max_logp,logp_error),max(max_value,value_error)
    return bound({"schema":SCHEMA+"-numerical-witness", "plan_sha256":plan["content_sha256"],
        "model_sha256":plan["model_sha256"], "training_batch_sha256":digest({"episodes":episodes}),
        "calls":2*len(STEPS),"numeric_atol":ATOL,"max_logp_error":max_logp,"max_value_error":max_value,
        "method":"LOADED_EXACT_MODEL_B2_GRAPH_REPLAY_PLUS_EXACT_SAMPLED_FACTOR_PATHS"})


def admit(folder, model_path, *, expected_plan_file_sha256, expected_model_sha256, device="cuda"):
    """Execute BOTH verification stages; no supplied verified booleans/receipts.

    Root invokes this bounded expensive stage. It allocates the intended model;
    it does not optimize, generate new economic programs for other worlds, or
    modify the producer's observations/logps/rewards to make admission pass.
    """
    import torch
    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError("actual CUDA behavior verification required; no silent CPU fallback")
    checked = witness(folder, expected_plan_file_sha256=expected_plan_file_sha256)
    plan, episodes = checked["plan"], checked["episodes"]
    _equal(plan["model_sha256"], _sha(expected_model_sha256), "externally pinned model")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, manifest = policy_module().load_model(model_path, device=device, expected_file_sha256=expected_model_sha256)
    _equal(manifest["content_sha256"], plan["model_manifest_sha256"], "behavior manifest")
    _equal(manifest["architecture"]["content_sha256"], plan["architecture_sha256"], "behavior architecture")
    _equal(manifest["optimizer_steps"], plan["model_optimizer_age"], "behavior optimizer age")
    _equal(manifest["rollout_iteration"]+1, plan["rollout_iteration"], "behavior rollout age")
    current = {"python":__import__("sys").version,"numpy":np.__version__,"torch":torch.__version__,
               "device":torch.cuda.get_device_name(),"dtype":"FLOAT32","tf32":False,"autocast":False}
    _equal(current, plan["environment"], "same numerical environment")
    numeric = _verify_graph(graph, plan, episodes)
    _equal(file_sha(model_path), expected_model_sha256, "frozen behavior model")
    attest(plan["source_pins"])
    return {"episodes":episodes,"admission":bound({"schema":ADMISSION_SCHEMA,
        "model_sha256":plan["model_sha256"],"model_manifest_sha256":plan["model_manifest_sha256"],
        "source_pins":plan["source_pins"],"source_sha256":plan["source_sha256"],"numerics":plan["numerics"],
        "training_batch_sha256":digest({"episodes":episodes}),"episode_ids":[e["episode_id"] for e in episodes],
        "physical_games":1,"engine_reconstruction_games":1,"responsive_reference_games":0,
        "engine_witness":checked["engine_witness"],"numerical_witness":numeric,
        "terminal_training_admission":True,"optimizer_updates":0})}
