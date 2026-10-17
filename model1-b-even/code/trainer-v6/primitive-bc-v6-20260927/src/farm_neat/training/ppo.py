"""Fresh full-economic programs -> maintained terminal PPO, with real BC Adam.

This is the full-program graph binding, not a second environment or reward model.
Every fit executes a spawned fresh-engine witness and exact parent neural replay;
caller-supplied admission booleans are not an API. The owner still owns exclusive
seed allocation/consumption and storage. No old29-row or teacher prefix is an RL row.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from farm_neat.training.schedule import learning_rate_at, validate_settings
from farm_neat.policy import network as policy
from farm_neat.rollout import rollout as rollout
from farm_neat.training import moe_lab as lab

SCHEMA = "farm-full-economic-terminal-ppo-v1"
OPTIMIZER_SCHEMA = SCHEMA + "-optimizer"
EXECUTION = {"epochs": 1, "minibatch_size": 4,
    "forward_shape": "TWO_ORIGINAL_B2_SEAT_PAIRS_PER_MINIBATCH",
    "reward": "TERMINAL_WDL_PLUS1_ZERO_MINUS1_NO_SHAPING",
    "advantages": "MC_TERMINAL_RETURN_MINUS_OLD_VALUE",
    "kl": "SAMPLED_PROGRAM_K3_MEAN_EXPM1_LOG_RATIO_MINUS_LOG_RATIO_NOT_EXACT_KL",
    "entropy": "SUM_OF_ACTUAL_CONDITIONAL_ENTROPIES_AT_RETAINED_BEHAVIOR_PREFIXES"}
digest, bound, file_sha = rollout.digest, rollout.bound, rollout.file_sha


def sources():
    # Changing this pin list changes the source identity of every future checkpoint; intended.
    return {**rollout.sources(), **lab.bc_sources(),
            "src/farm_neat/training/ppo.py": file_sha(Path(__file__)),
            "tests/contracts/test_ppo.py": file_sha(ROOT/"tests/contracts/test_ppo.py")}


def _attest(expected):
    if expected != sources():
        raise ValueError("full-program PPO execution source changed")
    rollout.attest(expected)


def _fresh_witness(folder, expected_plan_file_sha256):
    """A fresh interpreter/environment, not imported/rehashed witness metadata."""
    completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
        "witness", "--producer", str(folder), "--plan-sha256", expected_plan_file_sha256],
        cwd=ROOT, capture_output=True, text=True, timeout=120, check=True)
    return json.loads(completed.stdout)


def validate_parent(saved, result, plan, manifest, *, model_sha, source, profile, parameter_names):
    """Explicit BC-v2 -> PPO objective transition; no optimizer conversion."""
    validate_settings(saved["settings"])
    age = saved["updates"]
    if (type(age) is not int or age < 1 or age != manifest["optimizer_steps"]
            or age != result["global_optimizer_updates"] or saved["model_sha256"] != model_sha
            or result["after_model_sha256"] != model_sha
            or result["manifest"] != manifest["content_sha256"]
            or saved["environment"] != profile or plan["environment"] != profile
            or saved["parameter_names"] != parameter_names
            or saved["optimizer_config"] != lab.OPTIMIZER_CONFIG
            or saved["settings"] != plan["schedule"]
            or age != plan["start_update"]+result["optimizer_updates"]):
        raise ValueError("exact parent model/Adam/age/RNG profile/parameter order required")
    if not saved["optimizer"].get("param_groups"):
        raise ValueError("actual persisted Adam parameter groups required")
    for group in saved["optimizer"]["param_groups"]:
        if (group["lr"] != learning_rate_at(age, saved["settings"])
                or list(group["betas"]) != lab.OPTIMIZER_CONFIG["betas"]
                or group["eps"] != lab.OPTIMIZER_CONFIG["eps"] or group["weight_decay"] != 0.0):
            raise ValueError("parent8bit Adam learning rate/moments configuration mismatch")
    if saved.get("schema") == lab.BC_OPTIMIZER_SCHEMA:
        if (result["status"] != "BC_UPDATES_COMPLETED_NOT_RL_STRENGTH"
                or saved["source_pins"] != lab.bc_sources()
                or result["source_pins"] != saved["source_pins"]
                or plan["source_pins"] != saved["source_pins"]
                or saved["plan_sha256"] != result["plan_sha256"]
                or manifest.get("lineage", {}).get("plan_sha256") != saved["plan_sha256"]
                or manifest["lineage"].get("bc_protocol") != lab.BC_PROTOCOL
                or manifest["lineage"].get("selection_sha256") != digest(saved["selection"])
                or result.get("bc_protocol") != lab.BC_PROTOCOL or plan.get("bc_protocol") != lab.BC_PROTOCOL
                or result["optimizer_updates"] != plan["requested_updates"]
                or saved["selection"] != plan["selection"] or saved["selection"] != result["selection"]
                or saved["teacher"] != plan["teacher"] or saved["teacher"] != result["teacher"]
                or saved["next_example_index"] != result["next_example_index"]
                or saved["next_example_index"] != plan["start_example_index"] + result["optimizer_updates"]*plan["batch_size"]
                or manifest["rollout_iteration"] != 0):
            raise ValueError("source-bound completed BC-v2 anchor required for objective transition")
        return {"kind": "EXPLICIT_BC_V2_TO_FULL_PROGRAM_TERMINAL_PPO",
            "bc_model_sha256": model_sha, "bc_optimizer_age": age,
            "bc_next_example_index": saved["next_example_index"], "bc_selection": saved["selection"],
            "bc_teacher": saved["teacher"], "old_source_pins": saved["source_pins"],
            "new_source_pins": source, "optimizer_conversion": "NONE_ADAM8BIT_AND_RNG_UNCHANGED"}, []
    if (saved.get("schema") != OPTIMIZER_SCHEMA or saved.get("source_pins") != source
            or result.get("source_pins") != source or plan.get("source_pins") != source
            or saved.get("execution") != EXECUTION or plan.get("execution") != EXECUTION
            or saved.get("last_rollout_iteration") != manifest["rollout_iteration"]
            or result.get("status") != "FRESH_FULL_PROGRAM_PPO_COMMITTED_NOT_STRENGTH"
            or manifest.get("lineage", {}).get("ppo_protocol") != SCHEMA
            or saved.get("plan_sha256") != result.get("plan_sha256")
            or manifest["lineage"].get("plan_sha256") != saved.get("plan_sha256")):
        raise ValueError("exact full-program PPO successor source/objective/rollout required")
    return saved["bc_origin"], saved["recent_episode_ids"]


def admitted_pairs(reconstructed, manifest, model_sha, prior_ids):
    """Validate actual freshly reconstructed objects before numerical admission."""
    pairs, ids, seeds = [], [], []
    for checked in reconstructed:
        plan, episodes, witness = checked["plan"], checked["episodes"], checked["engine_witness"]
        rollout._sealed(plan); rollout._sealed(witness)
        rollout.attest(plan["source_pins"])
        if (plan["model_sha256"] != model_sha or plan["model_manifest_sha256"] != manifest["content_sha256"]
                or plan["model_optimizer_age"] != manifest["optimizer_steps"]
                or plan["rollout_iteration"] != manifest["rollout_iteration"]+1
                or plan["neural_batch"] != "BOTH_SEATS_TOGETHER_DYNAMIC_PADDING_B2"
                or plan["numerics"] != policy.NUMERICS or plan["temperature"] != 1.0
                or witness["plan_sha256"] != plan["content_sha256"]
                or witness["training_batch_sha256"] != digest({"episodes": episodes})
                or witness["reconstructed_physical_games"] != 1 or witness["physical_steps"] != 719
                or witness["responsive_reference_games"] != 0 or len(episodes) != 2):
            raise ValueError("same-parent full-season original B2 witness required")
        seeds.append(plan["seed"])
        cash = witness["terminal_cash"]
        if not isinstance(cash, list) or len(cash) != 2 or any(rollout._finite(v) < 0 for v in cash):
            raise ValueError("actual reconstructed terminal cash pair required")
        for seat, episode in enumerate(episodes):
            reward = episode["terminal_reward"]
            delta = cash[seat]-cash[1-seat]
            actual_reward = int(delta > 0)-int(delta < 0)
            if (episode["schema"] != rollout.EPISODE_SCHEMA or episode["seat"] != seat
                    or episode["episode_id"] != plan["game_id"]+f"-seat{seat}"
                    or episode["seed"] != plan["seed"] or episode["behavior_model_sha256"] != model_sha
                    or episode["model_manifest_sha256"] != manifest["content_sha256"]
                    or episode["rollout_iteration"] != plan["rollout_iteration"]
                    or episode["source_sha256"] != plan["source_sha256"]
                    or episode["numerics"] != policy.NUMERICS
                    or type(reward) is not int or reward != actual_reward
                    or [call["step"] for call in episode["calls"]] != list(rollout.STEPS)):
                raise ValueError("new full-program30-call WDL episode required; no BC/old29 rows")
            ids.append(episode["episode_id"])
            for call in episode["calls"]:
                rollout._sealed(call)
                if call["seat"] != seat or call["old_log_probability"] != call["behavior"]["joint_logp"]:
                    raise ValueError("actual seat and complete-program logp required")
                rollout._trace(call["behavior"], plan)
                rollout.unpack(call["input"])
                rollout._finite(call["old_value"])
        if episodes[0]["terminal_reward"] != -episodes[1]["terminal_reward"]:
            raise ValueError("whole-game zero-sum terminal WDL required")
        for ordinal in range(30):
            pairs.append({"calls": [e["calls"][ordinal] for e in episodes],
                          "returns": [e["terminal_reward"] for e in episodes]})
    if (not pairs or len(ids) != len(set(ids)) or len(seeds) != len(set(seeds))
            or set(ids) & set(prior_ids)):
        raise ValueError("unique fresh physical worlds and unconsumed episode identities required")
    return pairs, ids


def replay_pairs(graph, pairs):
    import torch
    logps, values, entropies = [], [], []
    for pair in pairs:
        calls = pair["calls"]
        context, value, _ = graph.encode(*lab.tensor_inputs(graph, [rollout.unpack(c["input"]) for c in calls]))
        logp, entropy = graph.replay(context, [c["behavior"] for c in calls])
        logps.append(logp); values.append(value); entropies.append(entropy)
    return torch.cat(logps), torch.cat(values), torch.cat(entropies)


def ppo_terms(logp, value, entropy, old_logp, returns, advantages, settings):
    """Maintained clipped selected-action PPO and MC critic; full joint action."""
    import torch
    ratio = (logp-old_logp).exp()
    surrogate = torch.minimum(ratio*advantages,
        ratio.clamp(1-settings["clip_ratio"], 1+settings["clip_ratio"])*advantages)
    actor, critic, ent = -surrogate.mean(), (value-returns).square().mean(), entropy.mean()
    loss = actor+settings["value_coefficient"]*critic-settings["entropy_coefficient"]*ent
    return loss, actor, critic, ent, ((ratio-1).abs()>settings["clip_ratio"]).double().mean()


def metrics(logp, value, entropy, old_logp, returns):
    import torch
    delta = logp-old_logp
    tensors = [torch.expm1(delta)-delta, -delta, entropy, (value-returns).square()]
    if any(not bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError("nonfinite full-program PPO diagnostics")
    variance = returns.var(unbiased=False)
    return {"sampled_program_kl_k3": float(tensors[0].mean()),
        "sampled_old_minus_new_logp": float(tensors[1].mean()),
        "conditional_path_entropy": float(tensors[2].mean()), "value_mse": float(tensors[3].mean()),
        "value_explained_variance": float(1-(returns-value).var(unbiased=False)/variance) if variance > 1e-12 else None}


def _cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def optimize(graph, optimizer, pairs, settings, global_step, *, max_updates=4):
    """One bounded epoch; a failed invocation restores all in-memory fit state."""
    import torch
    validate_settings(settings)
    if type(max_updates) is not int or not 1 <= max_updates <= 16:
        raise ValueError("explicit1..16 update allocation required")
    device = next(graph.parameters()).device
    saved_model, saved_adam = _cpu_copy(graph.state_dict()), _cpu_copy(optimizer.state_dict())
    saved_cpu = torch.get_rng_state().clone()
    saved_cuda = [x.clone() for x in torch.cuda.get_rng_state_all()] if device.type == "cuda" else []
    start, history = global_step, []
    try:
        old_logp = torch.tensor([c["old_log_probability"] for p in pairs for c in p["calls"]], dtype=torch.float64, device=device)
        old_value = torch.tensor([c["old_value"] for p in pairs for c in p["calls"]], dtype=torch.float64, device=device)
        returns = torch.tensor([v for p in pairs for v in p["returns"]], dtype=torch.float64, device=device)
        with torch.no_grad():
            logp, value, entropy = replay_pairs(graph, pairs)
            logp_error, value_error = float((logp-old_logp).abs().max()), float((value-old_value).abs().max())
            if max(logp_error, value_error) > rollout.ATOL:
                raise ValueError("unchanged2e-5 parent full-program behavior parity failed before PPO")
            current = initial = metrics(logp, value, entropy, old_logp, returns)
        advantages = returns-old_value
        raw_std = float(advantages.std(unbiased=False))
        if settings["normalize_advantages"] and raw_std > 1e-8:
            advantages = (advantages-advantages.mean())/raw_std
        advantages = advantages.detach()  # Constant/all-draw batches remain present.
        order = torch.randperm(len(pairs), device="cpu").tolist()
        stop = "ONE_EPOCH_COMPLETE"
        for offset in range(0, len(order), 2):
            if current["sampled_program_kl_k3"] > settings["target_kl"]:
                stop = "SAMPLED_PROGRAM_KL_EARLY_STOP"; break
            if len(history) >= max_updates or global_step >= settings["planned_optimizer_steps"]:
                stop = "EXPLICIT_UPDATE_ALLOCATION_COMPLETE"; break
            chosen = order[offset:offset+2]
            idx = torch.tensor([2*i+s for i in chosen for s in (0, 1)], device=device)
            logp, value, entropy = replay_pairs(graph, [pairs[i] for i in chosen])
            loss, actor, critic, ent, clipped = ppo_terms(logp, value, entropy, old_logp[idx], returns[idx], advantages[idx], settings)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite PPO loss; no accepted child")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(graph.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            lr = learning_rate_at(global_step+1, settings)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            global_step += 1
            if any(not bool(torch.isfinite(p).all()) for p in graph.parameters()):
                raise ValueError("nonfinite updated weights; invocation restored")
            with torch.no_grad():
                current = metrics(*replay_pairs(graph, pairs), old_logp, returns)
            receipt = {"optimizer_step": global_step, "minibatch_calls": len(idx), "learning_rate": lr,
                "loss": float(loss.detach()), "actor_loss": float(actor.detach()), "value_loss": float(critic.detach()),
                "conditional_path_entropy_before": float(ent.detach()), "gradient_norm": float(norm),
                "clip_fraction": float(clipped.detach()), "post_update": current}
            history.append(receipt)
            print(lab.canonical(receipt), flush=True)
        if not history:
            raise ValueError("no optimizer update; no completed PPO fit")
        changed = [name for name, t in graph.state_dict().items() if not torch.equal(t.detach().cpu(), saved_model[name])]
        if not changed:
            raise ValueError("unchanged tensors are not a completed PPO fit")
        if current["sampled_program_kl_k3"] > settings["target_kl"]:
            stop = "SAMPLED_PROGRAM_KL_EXCEEDED_AFTER_UPDATE_NOT_HARD_BOUND"
        return {"optimizer_steps": len(history), "global_optimizer_steps_before": start,
            "global_optimizer_steps_after": global_step, "history": history, "stop_reason": stop,
            "initial": initial, "final": current, "raw_advantage_std": raw_std,
            "old_logp_max_error": logp_error, "old_value_max_error": value_error,
            "numeric_atol": rollout.ATOL, "changed_tensors": changed,
            "kl_is_sampled_not_exact": True, "post_update_kl_is_not_a_hard_bound": True,
            "all_draw_filtering": False}
    except BaseException:
        graph.load_state_dict(saved_model, strict=True)
        optimizer.load_state_dict(saved_adam)
        torch.set_rng_state(saved_cpu)
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved_cuda)
        raise


def fit(model_path, output, *, optimizer_dir, producers, expected_model_sha256,
        expected_optimizer_sha256, max_updates=4, device="cuda"):
    """Owner-only runtime API. producers=[{directory,plan_file_sha256}, ...]."""
    import bitsandbytes as bnb
    import torch
    if device != "cuda" or not sys.dont_write_bytecode or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("explicit CUDA -B :4096:8 full-program numerical profile required before witnesses")
    profile = {**lab.device_profile(), "bitsandbytes": str(bnb.__version__)}
    if not isinstance(producers, list) or not 1 <= len(producers) <= 8:
        raise ValueError("one..eight explicitly pinned fresh producer games required")
    source = sources(); _attest(source)
    model_path, optimizer_dir = Path(model_path), Path(optimizer_dir)
    if (optimizer_dir/"FAILURE.json").exists():
        raise ValueError("failed or partially committed optimizer directory is not a parent")
    result = json.loads((optimizer_dir/"RESULT.json").read_text())
    parent_plan = json.loads((optimizer_dir/"PLAN.json").read_text())
    if (file_sha(model_path) != expected_model_sha256
            or file_sha(optimizer_dir/"optimizer.pt") != expected_optimizer_sha256
            or result["optimizer_sha256"] != expected_optimizer_sha256
            or result["plan_sha256"] != file_sha(optimizer_dir/"PLAN.json")):
        raise ValueError("externally pinned completed parent model/optimizer/plan required")
    graph, manifest = policy.load_model(model_path, device=device, expected_file_sha256=expected_model_sha256)
    if manifest["architecture"] != json.loads(json.dumps(policy.architecture())) or graph.config != policy.DEFAULT:
        raise ValueError("intended full60M16-layer policy required; no tiny fallback")
    saved = torch.load(optimizer_dir/"optimizer.pt", map_location="cpu", weights_only=True)
    names = [name for name, _ in graph.named_parameters()]
    origin, prior_ids = validate_parent(saved, result, parent_plan, manifest, model_sha=expected_model_sha256,
        source=source, profile=profile, parameter_names=names)
    if saved["plan_sha256"] != file_sha(optimizer_dir/"PLAN.json"):
        raise ValueError("parent optimizer/PLAN file join mismatch")
    optimizer = bnb.optim.AdamW8bit(graph.parameters(), lr=learning_rate_at(saved["updates"], saved["settings"]),
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, min_8bit_size=4096, block_wise=True)
    lab.restore_bc_optimizer(optimizer, saved)
    graph.eval()  # No dropout; the actual producer uses the same mode.
    checked = [_fresh_witness(p["directory"], p["plan_file_sha256"]) for p in producers]
    pairs, episode_ids = admitted_pairs(checked, manifest, expected_model_sha256, prior_ids)
    numerical = []
    for game in checked:
        expected_profile = {k: v for k, v in profile.items() if k != "bitsandbytes"}
        if game["plan"]["environment"] != expected_profile:
            raise ValueError("producer/fitter numerical profile mismatch")
        numerical.append(rollout._verify_graph(graph, game["plan"], game["episodes"]))
    episodes = [episode for game in checked for episode in game["episodes"]]
    admission = bound({"schema": SCHEMA+"-executed-admission", "source_pins": source,
        "model_sha256": expected_model_sha256, "model_manifest_sha256": manifest["content_sha256"],
        "training_batch_sha256": digest({"episodes": episodes}), "episode_ids": episode_ids,
        "physical_producer_games": len(checked), "fresh_engine_reconstruction_games": len(checked),
        "responsive_reference_games": 0, "engine_witnesses": [g["engine_witness"] for g in checked],
        "numerical_witnesses": numerical, "caller_verified_boolean_accepted": False})
    output = lab.output_directory(output)
    plan = {"schema": SCHEMA+"-plan", "source_pins": source, "environment": profile,
        "execution": EXECUTION, "schedule": saved["settings"], "max_updates": max_updates,
        "parent_model_sha256": expected_model_sha256, "parent_optimizer_sha256": expected_optimizer_sha256,
        "start_update": saved["updates"], "parent_manifest_sha256": manifest["content_sha256"],
        "producer_files": producers, "admission_sha256": admission["content_sha256"], "bc_origin": origin}
    lab.write(output/"PLAN.json", plan); lab.write(output/"ADMISSION.json", admission)
    plan_sha = file_sha(output/"PLAN.json")
    started = time.monotonic()
    rollback_model = _cpu_copy(graph.state_dict())
    try:
        _attest(source)
        receipt = optimize(graph, optimizer, pairs, saved["settings"], saved["updates"], max_updates=max_updates)
        _attest(source)
        if (file_sha(model_path) != expected_model_sha256
                or file_sha(optimizer_dir/"optimizer.pt") != expected_optimizer_sha256):
            raise ValueError("parent checkpoint changed during fresh fit")
        age = receipt["global_optimizer_steps_after"]
        child = policy.save_model(output/"STAGED-model.pt", graph, optimizer_steps=age,
            rollout_iteration=manifest["rollout_iteration"]+1, initialization_seed=manifest["initialization_seed"],
            lineage={"kind": "FRESH_FULL_PROGRAM_TERMINAL_PPO", "ppo_protocol": SCHEMA,
                "parent_file_sha256": expected_model_sha256, "plan_sha256": plan_sha,
                "admission_sha256": admission["content_sha256"]})
        torch.save({"schema": OPTIMIZER_SCHEMA, "source_pins": source, "execution": EXECUTION,
            "environment": profile, "updates": age, "settings": saved["settings"],
            "optimizer": optimizer.state_dict(), "optimizer_config": lab.OPTIMIZER_CONFIG,
            "parameter_names": names, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "model_sha256": file_sha(output/"STAGED-model.pt"), "plan_sha256": plan_sha, "bc_origin": origin,
            "last_rollout_iteration": manifest["rollout_iteration"]+1,
            "recent_episode_ids": (prior_ids+episode_ids)[-4096:]}, output/"STAGED-optimizer.pt")
        _attest(source)
        for old, new in (("STAGED-model.pt", "model.pt"), ("STAGED-model.pt.json", "model.pt.json"),
                         ("STAGED-optimizer.pt", "optimizer.pt")):
            (output/old).rename(output/new)
        final = {"schema": SCHEMA+"-result", "status": "FRESH_FULL_PROGRAM_PPO_COMMITTED_NOT_STRENGTH",
            "plan_sha256": plan_sha, "source_pins": source, "optimizer_updates": receipt["optimizer_steps"],
            "global_optimizer_updates": age, "before_model_sha256": expected_model_sha256,
            "after_model_sha256": file_sha(output/"model.pt"), "optimizer_sha256": file_sha(output/"optimizer.pt"),
            "manifest": child["content_sha256"], "admission_sha256": admission["content_sha256"],
            "physical_games": len(checked), "engine_reconstruction_games": len(checked),
            "responsive_reference_games": 0, "natural_calls": len(pairs)*2,
            "selfplay_terminal_outcomes_not_strength": [e["terminal_reward"] for e in episodes],
            "elapsed_fit_seconds": time.monotonic()-started, "fit": receipt, "population_strength_established": False}
        lab.write(output/"STAGED-RESULT.json", final)
        (output/"STAGED-RESULT.json").rename(output/"RESULT.json")
        return final
    except BaseException as exc:
        graph.load_state_dict(rollback_model, strict=True)
        lab.restore_bc_optimizer(optimizer, saved)
        # Parent files are immutable. No RESULT means any partially written files
        # are failed staging, not an accepted optimizer update or continuation.
        lab.write(output/"FAILURE.json", {"schema": SCHEMA+"-failure", "error": type(exc).__name__+": "+str(exc),
            "committed_optimizer_updates": 0, "parent_model_sha256": expected_model_sha256,
            "parent_optimizer_sha256": expected_optimizer_sha256, "no_result_no_admitted_child": True,
            "parameters_adam_rng_restored_to_parent": True,
            "global_optimizer_age_restored": saved["updates"]})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("witness")
    p.add_argument("--producer", required=True); p.add_argument("--plan-sha256", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--model", required=True); p.add_argument("--model-sha256", required=True)
    p.add_argument("--optimizer-dir", required=True); p.add_argument("--optimizer-sha256", required=True)
    p.add_argument("--producer", action="append", required=True)
    p.add_argument("--plan-sha256", action="append", required=True)
    p.add_argument("--output", required=True); p.add_argument("--max-updates", type=int, default=4)
    args = parser.parse_args()
    if args.command == "witness":
        result = rollout.witness(args.producer, expected_plan_file_sha256=args.plan_sha256)
    else:
        if len(args.producer) != len(args.plan_sha256):
            raise ValueError("one external plan SHA per producer required")
        result = fit(args.model, args.output, optimizer_dir=args.optimizer_dir,
            producers=[{"directory": folder, "plan_file_sha256": sha} for folder, sha in zip(args.producer, args.plan_sha256, strict=True)],
            expected_model_sha256=args.model_sha256, expected_optimizer_sha256=args.optimizer_sha256,
            max_updates=args.max_updates)
    print(lab.canonical(result), flush=True)


if __name__ == "__main__":
    main()
