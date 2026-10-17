"""Source-versioned B32 continuation of the maintained full-economic PPO.

One original daily cohort (16 worlds, 32 seats) is a minibatch. This changes
batching, not the model, WDL objective, schedule, optimizer or admissibility of
old data. The fitter spawns a fresh physical witness, reconstructs the exact
cohort neural sampling, then uses unchanged PPO terms and diagnostics. Block
checkpointing limits gradient memory; typed-factor replay groups equivalent
operations without shortening a program. Neither helper supplies labels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT / "src"))

from farm_neat.policy import network as policy
from farm_neat.training import ppo as legacy
from farm_neat.rollout import batched_rollout as rollout
from farm_neat.rollout import batched_replay as heads
from farm_neat.rollout import checkpointed_encode as encoder
from farm_neat.training import moe_lab as lab
from farm_neat.training.schedule import learning_rate_at, validate_settings

SCHEMA = "farm-full-economic-day-cohort-ppo-v1"
OPTIMIZER_SCHEMA = SCHEMA + "-optimizer"
EXECUTION = {**legacy.EXECUTION, "minibatch_size": 32,
    "forward_shape": "ONE_ORIGINAL_16_WORLD_B32_DAY_COHORT_PER_MINIBATCH",
    "encoder": encoder.SCHEMA, "factor_replay": heads.SCHEMA,
    "post_update_metrics": "ALL_30_ORIGINAL_COHORTS_AFTER_EACH_UPDATE",
    "sampling": rollout.BATCH_MODE}
file_sha, digest, bound = rollout.file_sha, rollout.digest, rollout.bound


def sources():
    # Changing this pin list changes the source identity of every future checkpoint; intended.
    paths = ("src/farm_neat/training/batched_training.py", "tests/contracts/test_batched_training.py")
    return {**legacy.sources(), **rollout.sources(), **heads.identity()["source_pins"],
        **encoder.identity()["source_pins"], **{p: file_sha(ROOT/p) for p in paths}}


def attest(pins):
    if pins != sources():
        raise ValueError("day-cohort PPO source closure changed")
    legacy._attest(legacy.sources())
    rollout.attest(pins)


def validate_parent(saved, result, plan, manifest, *, model_sha, source, profile, parameter_names):
    """Validate the old identity as old; never relabel its schema/source pins."""
    args = dict(model_sha=model_sha, profile=profile, parameter_names=parameter_names)
    if saved.get("schema") == legacy.OPTIMIZER_SCHEMA:
        origin, ids = legacy.validate_parent(saved, result, plan, manifest, source=legacy.sources(), **args)
        transition = {"kind": "EXPLICIT_B2_TO_B32_EXECUTION_VERSION",
            "parent_model_sha256": model_sha, "global_optimizer_age": saved["updates"],
            "old_source_pins": saved["source_pins"], "old_execution": legacy.EXECUTION,
            "new_source_pins": source, "new_execution": EXECUTION,
            "optimizer_conversion": "NONE_ADAM_MOMENTS_RNG_AGE_AND_SCHEDULE_PRESERVED"}
        return origin, ids, transition
    # Reuse every maintained successor guard with explicitly NEW constants;
    # no mutation of a shared module, saved receipt or old validator occurs.
    check = FunctionType(legacy.validate_parent.__code__, {**legacy.validate_parent.__globals__,
        "SCHEMA": SCHEMA, "OPTIMIZER_SCHEMA": OPTIMIZER_SCHEMA, "EXECUTION": EXECUTION})
    origin, ids = check(saved, result, plan, manifest, source=source, **args)
    transition = saved.get("batch_origin")
    if (not isinstance(transition, dict) or transition != plan.get("batch_origin")
            or transition.get("kind") != "EXPLICIT_B2_TO_B32_EXECUTION_VERSION"
            or transition.get("new_source_pins") != source
            or transition.get("new_execution") != EXECUTION):
        raise ValueError("source-bound original batching transition required")
    return origin, ids, transition


def fresh_witness(folder, expected_plan_file_sha256):
    result = subprocess.run([sys.executable, "-B", "-m", "farm_neat.rollout.batched_rollout",
        "witness", "--producer", str(folder), "--plan-sha256", expected_plan_file_sha256,
        "--include-episodes"], cwd=ROOT, capture_output=True, text=True, timeout=600, check=True)
    return json.loads(result.stdout)


def admitted_groups(checked, manifest, model_sha, prior_ids):
    plan, episodes, witness = checked["plan"], checked["episodes"], checked["engine_witness"]
    rollout.validate_plan(plan); rollout._sealed(witness)
    if (plan["model_sha256"] != model_sha or plan["model_manifest_sha256"] != manifest["content_sha256"]
            or plan["model_optimizer_age"] != manifest["optimizer_steps"]
            or plan["rollout_iteration"] != manifest["rollout_iteration"]+1
            or plan["worlds"] != 16 or plan["lanes"] != 32
            or witness["schema"] != rollout.SCHEMA+"-engine-witness"
            or witness["plan_sha256"] != plan["content_sha256"]
            or witness["training_batch_sha256"] != digest({"episodes": episodes})
            or witness["reconstructed_physical_games"] != 16 or witness["physical_steps"] != 16*719
            or witness["responsive_reference_games"] != 0 or len(witness["terminal_cash"]) != 16):
        raise ValueError("same-parent complete fresh16-world B32 witness required")
    groups = rollout.cohort_groups(plan, episodes)
    if witness["cohorts_sha256"] != digest([g["batch"] for g in groups]):
        raise ValueError("freshly reconstructed original cohort padding required")
    ids = [e["episode_id"] for e in episodes]
    if len(set(ids)) != 32 or set(ids) & set(prior_ids):
        raise ValueError("unique unconsumed episode identities required")
    for w, cash in enumerate(witness["terminal_cash"]):
        if not isinstance(cash, list) or len(cash) != 2 or any(legacy.rollout._finite(c) < 0 for c in cash):
            raise ValueError("actual reconstructed terminal cash pair required")
        delta = cash[0]-cash[1]
        reward = int(delta > 0)-int(delta < 0)
        if [episodes[2*w+s]["terminal_reward"] for s in (0,1)] != [reward, -reward]:
            raise ValueError("terminal WDL must match fresh physical cash, not claimed labels")
    return groups, ids


def replay_groups(graph, groups):
    import torch
    values = []
    enc_identity, head_identity = encoder.identity(), heads.identity()
    for group in groups:
        calls = group["calls"]
        tensors = lab.tensor_inputs(graph, [legacy.rollout.unpack(c["input"]) for c in calls])
        if list(tensors[0].shape) != group["batch"]["arrays"]["rows"]["shape"]:
            raise ValueError("training may not change original cohort padding")
        context, value, _ = encoder.encode(graph, *tensors, expected_identity=enc_identity)
        logp, entropy = heads.replay(graph, context, [c["behavior"] for c in calls],
            expected_identity=head_identity)
        values.append((logp, value, entropy))
    return tuple(torch.cat([v[i] for v in values]) for i in range(3))


def optimize(graph, optimizer, groups, settings, global_step, *, max_updates=8):
    """One shuffled daily-cohort epoch; any failure restores tensors/Adam/RNG."""
    import torch
    validate_settings(settings)
    if type(max_updates) is not int or not 1 <= max_updates <= 16:
        raise ValueError("explicit1..16 optimizer update allocation required")
    sizes = [len(g["calls"]) for g in groups]
    if not sizes or len(set(sizes)) != 1 or sizes[0] < 2 or any(len(g["returns"]) != sizes[0] for g in groups):
        raise ValueError("equal complete original day cohorts required")
    lanes = sizes[0]
    device = next(graph.parameters()).device
    model_before, adam_before = legacy._cpu_copy(graph.state_dict()), legacy._cpu_copy(optimizer.state_dict())
    cpu_before = torch.get_rng_state().clone()
    cuda_before = [x.clone() for x in torch.cuda.get_rng_state_all()] if device.type == "cuda" else []
    start, history = global_step, []
    try:
        old_logp = torch.tensor([c["old_log_probability"] for g in groups for c in g["calls"]], dtype=torch.float64, device=device)
        old_value = torch.tensor([c["old_value"] for g in groups for c in g["calls"]], dtype=torch.float64, device=device)
        returns = torch.tensor([r for g in groups for r in g["returns"]], dtype=torch.float64, device=device)
        with torch.no_grad():
            logp, value, entropy = replay_groups(graph, groups)
            logp_error, value_error = float((logp-old_logp).abs().max()), float((value-old_value).abs().max())
            if max(logp_error, value_error) > rollout.ATOL:
                raise ValueError("unchanged2e-5 parent behavior parity failed before PPO")
            initial = current = legacy.metrics(logp, value, entropy, old_logp, returns)
        advantages = returns-old_value
        raw_std = float(advantages.std(unbiased=False))
        if settings["normalize_advantages"] and raw_std > 1e-8:
            advantages = (advantages-advantages.mean())/raw_std
        advantages = advantages.detach()
        order = torch.randperm(len(groups), device="cpu").tolist()
        stop = "ONE_EPOCH_COMPLETE"
        for ordinal in order:
            if current["sampled_program_kl_k3"] > settings["target_kl"]:
                stop = "SAMPLED_PROGRAM_KL_EARLY_STOP"; break
            if len(history) >= max_updates or global_step >= settings["planned_optimizer_steps"]:
                stop = "EXPLICIT_UPDATE_ALLOCATION_COMPLETE"; break
            tick = time.monotonic()
            selected = slice(ordinal*lanes, (ordinal+1)*lanes)
            optimizer.zero_grad(set_to_none=True)
            logp, value, entropy = replay_groups(graph, [groups[ordinal]])
            loss, actor, critic, ent, clipped = legacy.ppo_terms(logp, value, entropy,
                old_logp[selected], returns[selected], advantages[selected], settings)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite PPO loss; no child")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(graph.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            lr = learning_rate_at(global_step+1, settings)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            global_step += 1
            if any(not bool(torch.isfinite(p).all()) for p in graph.parameters()):
                raise ValueError("nonfinite updated weights; invocation restored")
            update_seconds = time.monotonic()-tick
            tick = time.monotonic()
            with torch.no_grad():
                current = legacy.metrics(*replay_groups(graph, groups), old_logp, returns)
            receipt = {"optimizer_step": global_step, "minibatch_calls": lanes, "cohort_ordinal": ordinal,
                "learning_rate": lr, "loss": float(loss.detach()), "actor_loss": float(actor.detach()),
                "value_loss": float(critic.detach()), "conditional_path_entropy_before": float(ent.detach()),
                "gradient_norm": float(norm), "clip_fraction": float(clipped.detach()), "post_update": current,
                "update_seconds": update_seconds, "full_batch_metric_seconds": time.monotonic()-tick}
            history.append(receipt); print(lab.canonical(receipt), flush=True)
        if not history:
            raise ValueError("no optimizer update; no completed fit")
        changed = [k for k,t in graph.state_dict().items() if not torch.equal(t.detach().cpu(), model_before[k])]
        if not changed:
            raise ValueError("unchanged tensors are not a completed fit")
        if current["sampled_program_kl_k3"] > settings["target_kl"]:
            stop = "SAMPLED_PROGRAM_KL_EXCEEDED_AFTER_UPDATE_NOT_HARD_BOUND"
        return {"optimizer_steps": len(history), "global_optimizer_steps_before": start,
            "global_optimizer_steps_after": global_step, "history": history, "stop_reason": stop,
            "initial": initial, "final": current, "raw_advantage_std": raw_std,
            "old_logp_max_error": logp_error, "old_value_max_error": value_error, "numeric_atol": rollout.ATOL,
            "changed_tensors": changed, "kl_is_sampled_not_exact": True,
            "post_update_kl_is_not_a_hard_bound": True, "all_draw_filtering": False}
    except BaseException:
        graph.load_state_dict(model_before, strict=True); optimizer.load_state_dict(adam_before)
        torch.set_rng_state(cpu_before)
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda_before)
        raise


def fit(model_path, output, *, optimizer_dir, producer, expected_plan_file_sha256,
        expected_model_sha256, expected_optimizer_sha256, max_updates=8, device="cuda"):
    if device != "cuda" or not sys.dont_write_bytecode or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("explicit CUDA -B :4096:8 continuation required")
    if type(max_updates) is not int or not 1 <= max_updates <= 16:
        raise ValueError("explicit1..16 update allocation required before witnesses")
    target = Path(output).absolute()
    if target.exists() or target != target.resolve() or not target.is_relative_to(ROOT/"work/runs"):
        raise ValueError("exclusive new fit output below work/runs required")
    import bitsandbytes as bnb
    import torch
    profile = {**lab.device_profile(), "bitsandbytes": str(bnb.__version__)}
    source = sources(); attest(source)
    model_path, optimizer_dir = Path(model_path), Path(optimizer_dir)
    if (optimizer_dir/"FAILURE.json").exists():
        raise ValueError("failed or partially committed parent refused")
    result = json.loads((optimizer_dir/"RESULT.json").read_text())
    parent_plan = json.loads((optimizer_dir/"PLAN.json").read_text())
    if (file_sha(model_path) != expected_model_sha256
            or file_sha(optimizer_dir/"optimizer.pt") != expected_optimizer_sha256
            or result["optimizer_sha256"] != expected_optimizer_sha256
            or result["plan_sha256"] != file_sha(optimizer_dir/"PLAN.json")):
        raise ValueError("externally pinned completed parent model/Adam/PLAN required")
    graph, manifest = policy.load_model(model_path, device=device, expected_file_sha256=expected_model_sha256)
    if graph.config != policy.DEFAULT:
        raise ValueError("intended60M16layer full-economic model required")
    saved = torch.load(optimizer_dir/"optimizer.pt", map_location="cpu", weights_only=True)
    names = [name for name,_ in graph.named_parameters()]
    origin, prior_ids, transition = validate_parent(saved, result, parent_plan, manifest,
        model_sha=expected_model_sha256, source=source, profile=profile, parameter_names=names)
    if saved["plan_sha256"] != file_sha(optimizer_dir/"PLAN.json"):
        raise ValueError("parent optimizer/PLAN file join mismatch")
    optimizer = bnb.optim.AdamW8bit(graph.parameters(), lr=learning_rate_at(saved["updates"],saved["settings"]),
        betas=(.9,.999), eps=1e-8, weight_decay=0., min_8bit_size=4096, block_wise=True)
    lab.restore_bc_optimizer(optimizer, saved); graph.eval()
    started = time.monotonic()
    print(lab.canonical({"phase":"fresh_physical_witness", "worlds":16}), flush=True)
    checked = fresh_witness(producer, expected_plan_file_sha256)
    groups, episode_ids = admitted_groups(checked, manifest, expected_model_sha256, prior_ids)
    if checked["plan"]["environment"] != {k:v for k,v in profile.items() if k != "bitsandbytes"}:
        raise ValueError("producer/fitter exact numerical profile mismatch")
    print(lab.canonical({"phase":"fresh_neural_witness", "calls":960}), flush=True)
    numerical = rollout.verify_graph(graph, checked["plan"], checked["episodes"])
    admission = bound({"schema": SCHEMA+"-executed-admission", "source_pins":source,
        "model_sha256":expected_model_sha256, "model_manifest_sha256":manifest["content_sha256"],
        "training_batch_sha256":digest({"episodes":checked["episodes"]}), "episode_ids":episode_ids,
        "physical_producer_games":16, "fresh_engine_reconstruction_games":16,
        "responsive_reference_games":0, "engine_witness":checked["engine_witness"],
        "numerical_witness":numerical, "caller_verified_boolean_accepted":False})
    admission_seconds = time.monotonic()-started
    output = lab.output_directory(output)
    plan = {"schema":SCHEMA+"-plan", "source_pins":source,"environment":profile,
        "execution":EXECUTION,"schedule":saved["settings"],"max_updates":max_updates,
        "parent_model_sha256":expected_model_sha256,"parent_optimizer_sha256":expected_optimizer_sha256,
        "start_update":saved["updates"],"parent_manifest_sha256":manifest["content_sha256"],
        "producer":{"directory":str(producer),"plan_file_sha256":expected_plan_file_sha256},
        "admission_sha256":admission["content_sha256"],"bc_origin":origin,"batch_origin":transition}
    lab.write(output/"PLAN.json",plan); lab.write(output/"ADMISSION.json",admission)
    plan_sha = file_sha(output/"PLAN.json")
    rollback_model = legacy._cpu_copy(graph.state_dict())
    try:
        attest(source)
        receipt = optimize(graph,optimizer,groups,saved["settings"],saved["updates"],max_updates=max_updates)
        attest(source)
        if file_sha(model_path) != expected_model_sha256 or file_sha(optimizer_dir/"optimizer.pt") != expected_optimizer_sha256:
            raise ValueError("immutable parent changed during fit")
        age = receipt["global_optimizer_steps_after"]
        child = policy.save_model(output/"STAGED-model.pt",graph,optimizer_steps=age,
            rollout_iteration=manifest["rollout_iteration"]+1,initialization_seed=manifest["initialization_seed"],
            lineage={"kind":"FRESH_FULL_PROGRAM_DAY_COHORT_PPO","ppo_protocol":SCHEMA,
                "parent_file_sha256":expected_model_sha256,"plan_sha256":plan_sha,
                "admission_sha256":admission["content_sha256"]})
        torch.save({"schema":OPTIMIZER_SCHEMA,"source_pins":source,"execution":EXECUTION,
            "environment":profile,"updates":age,"settings":saved["settings"],
            "optimizer":optimizer.state_dict(),"optimizer_config":lab.OPTIMIZER_CONFIG,"parameter_names":names,
            "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
            "model_sha256":file_sha(output/"STAGED-model.pt"),"plan_sha256":plan_sha,"bc_origin":origin,
            "batch_origin":transition,"last_rollout_iteration":manifest["rollout_iteration"]+1,
            "recent_episode_ids":(prior_ids+episode_ids)[-4096:]},output/"STAGED-optimizer.pt")
        attest(source)
        for old,new in (("STAGED-model.pt","model.pt"),("STAGED-model.pt.json","model.pt.json"),
                        ("STAGED-optimizer.pt","optimizer.pt")):
            (output/old).rename(output/new)
        final = {"schema":SCHEMA+"-result","status":"FRESH_FULL_PROGRAM_PPO_COMMITTED_NOT_STRENGTH",
            "plan_sha256":plan_sha,"source_pins":source,"optimizer_updates":receipt["optimizer_steps"],
            "global_optimizer_updates":age,"before_model_sha256":expected_model_sha256,
            "after_model_sha256":file_sha(output/"model.pt"),"optimizer_sha256":file_sha(output/"optimizer.pt"),
            "manifest":child["content_sha256"],"admission_sha256":admission["content_sha256"],
            "physical_games":16,"engine_reconstruction_games":16,"responsive_reference_games":0,
            "natural_calls":960,"selfplay_terminal_outcomes_not_strength":[e["terminal_reward"] for e in checked["episodes"]],
            "admission_seconds":admission_seconds,"elapsed_fit_seconds":time.monotonic()-started,
            "fit":receipt,"population_strength_established":False}
        lab.write(output/"STAGED-RESULT.json",final); (output/"STAGED-RESULT.json").rename(output/"RESULT.json")
        return final
    except BaseException as exc:
        graph.load_state_dict(rollback_model,strict=True); lab.restore_bc_optimizer(optimizer,saved)
        lab.write(output/"FAILURE.json",{"schema":SCHEMA+"-failure","error":type(exc).__name__+": "+str(exc),
            "committed_optimizer_updates":0,"parent_model_sha256":expected_model_sha256,
            "parent_optimizer_sha256":expected_optimizer_sha256,"global_optimizer_age_restored":saved["updates"],
            "parameters_adam_rng_restored_to_parent":True,"no_result_no_admitted_child":True})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model","output","optimizer-dir","producer","plan-sha256","model-sha256","optimizer-sha256"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--max-updates",type=int,default=8)
    p.add_argument("--execute",action="store_true")
    a = p.parse_args()
    if not a.execute:
        print(lab.canonical({"status":"NOT_EXECUTED","execution":EXECUTION,"optimizer_updates":0})); return
    result = fit(a.model,a.output,optimizer_dir=a.optimizer_dir,producer=a.producer,
        expected_plan_file_sha256=a.plan_sha256,expected_model_sha256=a.model_sha256,
        expected_optimizer_sha256=a.optimizer_sha256,max_updates=a.max_updates)
    print(lab.canonical(result),flush=True)


if __name__ == "__main__":
    main()
