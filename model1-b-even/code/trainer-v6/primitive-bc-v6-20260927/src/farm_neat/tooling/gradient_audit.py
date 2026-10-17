"""Execute independent physical and actual-model checks for one current MoE game.

This is a thin CLI over full_economic_rollout.admit, not another collector.
Both expected hashes come from the caller's frozen source/checkpoint record.
No supplied verified flag is trusted. Default modes execute no optimizer update.
The opt-in gradient audits reconstruct real games and replay their sampled
programs. The old fast route separates PPO losses; the grouped route measures
current GRPO, entropy, plan-critic and future gradients. Neither route fits.
--recorded-grouped-gradient instead joins saved inputs to an exact retained
admission receipt and evaluates derivatives without physical reconstruction.
With --opponent-update-probe it executes two disposable optimizer steps from
the same saved parent, measures their effects, and publishes no checkpoint.
--closed-block joins a completed fast full-plan run's raw files and retained
admission receipts. It does not rerun neural witnesses or certify strength.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from farm_neat.training.moe_lab import canonical, device_profile, output_directory, write
from farm_neat.rollout.rollout import admit, file_sha


def gradient_components(graph, forward, returns, old_values, old_logp, settings):
    """Read-only loss-component backprop; no optimizer or surrogate engine gradient.

    ``forward`` must replay the same frozen sampled factors, masks and prefixes.
    This is a cohort diagnostic, not the larger training batch's noise scale.
    """
    import torch
    from farm_neat.training.ppo import ppo_terms
    advantages = returns-old_values
    std = advantages.std(unbiased=False)
    if settings["normalize_advantages"] and float(std) > 1e-8:
        advantages = (advantages-advantages.mean())/std
    versions = {n: p._version for n, p in graph.named_parameters()}
    result = {}
    try:
        for component in ("actor", "weighted_critic", "weighted_entropy"):
            graph.zero_grad(set_to_none=True)
            logp, values, entropy = forward()
            terms = ppo_terms(logp, values, entropy, old_logp, returns, advantages, settings)
            loss = {"actor": terms[1], "weighted_critic": settings["value_coefficient"]*terms[2],
                    "weighted_entropy": -settings["entropy_coefficient"]*terms[3]}[component]
            loss.backward()
            sums, active, absent = {}, [], []
            for name, parameter in graph.named_parameters():
                if parameter.grad is None:
                    absent.append(name); continue
                grad = parameter.grad.detach()
                if not bool(torch.isfinite(grad).all()):
                    raise ValueError("nonfinite "+component+" gradient: "+name)
                key = ".".join(name.split(".")[:2]) if name.startswith("blocks.") else name.split(".")[0]
                sq = float(grad.double().square().sum())
                sums[key] = sums.get(key, 0.0)+sq
                if sq > 0: active.append(name)
            result[component] = {"loss": float(loss.detach()),
                "l2_norm": sum(sums.values())**0.5,
                "group_l2_norms": {k: v**0.5 for k, v in sums.items()},
                "nonzero_parameter_names": active, "disconnected_parameter_names": absent}
        if versions != {n: p._version for n, p in graph.named_parameters()}:
            raise ValueError("gradient-only audit mutated weights")
        return result
    finally:
        graph.zero_grad(set_to_none=True)


def fast_gradient_audit(args, output):
    """One original fast cohort, freshly reconstructed; no new outcome labels."""
    import os
    import torch
    from farm_neat.training import campaign as campaign
    from farm_neat.policy import network as policy, full_economic_fast_profile as fast
    game, model = Path(args.game).resolve(), Path(args.model).resolve()
    if (not sys.dont_write_bytecode or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or file_sha(game/"PLAN.json") != args.expect_plan_sha256):
        raise ValueError("-B, explicit CUDA profile and exact cohort plan hash required")
    plan = campaign.io.read(game/"PLAN.json")
    campaign.validate_cohort_plan(plan)
    if not plan["opening_only"] or plan["model_sha256"] != args.expect_model_sha256:
        raise ValueError("same-parent opening cohort required for this diagnostic")
    if campaign.environment_profile() != plan["environment"]:
        raise ValueError("exact fast collection numerical profile required")
    graph, manifest = policy.load_model(model, device="cuda", expected_file_sha256=args.expect_model_sha256)
    if manifest["content_sha256"] != plan["model_manifest_sha256"]:
        raise ValueError("same recorded model manifest required")
    graph.eval()
    episodes = []
    for i in range(len(plan["seeds"])):
        episodes.extend(campaign.witness_world(game, i, plan["content_sha256"])["episodes"])
    groups = campaign.cohort_batches(plan, episodes)
    if len(groups) != 1:
        raise ValueError("one opening cohort required")
    group = groups[0]
    tensors = tuple(torch.as_tensor(group["arrays"][k], device="cuda")
                    for k in ("rows", "token_mask", "knownness"))
    identity = fast.identity()
    encoder = fast.FastEncoder(graph, expected_identity=identity)
    records = [c["behavior"] for c in group["calls"]]
    def forward():
        context, value, _ = encoder(*tensors)
        logp, entropy = fast.replay_padded(graph, context, records, expected_identity=identity)
        return logp, value, entropy
    old_logp = torch.tensor([c["old_log_probability"] for c in group["calls"]], dtype=torch.float64, device="cuda")
    old_values = torch.tensor([c["old_value"] for c in group["calls"]], dtype=torch.float64, device="cuda")
    returns = torch.tensor(group["returns"], dtype=torch.float64, device="cuda")
    with torch.no_grad():
        lp, value, _ = forward()
        parity = {"logp_max_abs": float((lp-old_logp).abs().max()), "value_max_abs": float((value-old_values).abs().max())}
    if (parity["logp_max_abs"] > fast.TOLERANCES["within_profile_logp_atol"] or
            parity["value_max_abs"] > fast.TOLERANCES["within_profile_value_atol"]):
        raise ValueError("actual behavior replay parity failed")
    settings = campaign.io.read(model.parent/"PLAN.json")["schedule"]
    components = gradient_components(graph, forward, returns, old_values, old_logp, settings)
    actor = components["actor"]["group_l2_norms"]
    expected = ("factor_query", "option_embedding", "path_embedding", "prefix", "input", "knownness_input",
                *(f"blocks.{i}" for i in range(graph.config.layers)))
    connected = all(actor.get(k, 0) > 0 for k in expected) and actor.get("value_head", 0) == 0
    if file_sha(model) != args.expect_model_sha256:
        raise ValueError("model file changed during read-only audit")
    result = {"schema": "farm-full-economic-gradient-audit-v1", "status": "PASS" if connected else "DISCONNECTED_OR_ZERO_ACTOR_GROUP",
        "source_sha256": file_sha(__file__), "model_sha256": args.expect_model_sha256,
        "plan_file_sha256": args.expect_plan_sha256, "cohort": str(game.relative_to(ROOT)),
        "physical_games_reconstructed": len(plan["seeds"]), "decisions": len(records),
        "fresh_games_collected": 0, "optimizer_updates": 0, "weights_changed": False,
        "source_pins": campaign.sources(), "parity": parity, "components": components,
        "scope": "ONE_COHORT_NOT_FULL_ACCUMULATED_TRAINING_GRADIENT_OR_NOISE_SCALE",
        "engine_differentiated": False, "strength_claim": False}
    write(output/"RESULT.json", result)
    return result


def grouped_gradient_components(learner, groups, *, strata=None):
    """All decision times in ONE collection cohort, not a full training round.

    Use the actual GRPO/auxiliary losses and equal-decision weights. Measure
    before clipping, separately on policy parameters and auxiliary heads.
    Policy-gradient alignment is relative to the actor term on this same cohort;
    it does not measure Adam updates, full-round noise scale, or strength.
    Optional strata partition only actor loss, retaining the total-decision denominator.
    """
    import math
    import torch
    from farm_neat.policy.plan_learning import (
        group_advantages, margin_group_advantages, grpo_terms)
    if not groups or not learner.learning:
        raise ValueError("nonempty witnessed grouped-learning cohorts required")
    named = list(learner.graph.named_parameters()) + [
        ("auxiliary." + n, p) for n, p in learner.auxiliary.named_parameters()]
    versions = {n: p._version for n, p in named}
    device = named[0][1].device
    prepared, diagnostics = [], []
    for group in groups:
        size = group["group_size"]
        if size != learner.learning["group_size"] or len(group["calls"]) % size:
            raise ValueError("complete registered GRPO groups required")
        returns = torch.tensor(group["returns"], dtype=torch.float64, device=device)
        old = torch.tensor([c["old_log_probability"] for c in group["calls"]],
                           dtype=torch.float64, device=device)
        if returns.shape != old.shape:
            raise ValueError("every recorded decision needs its witnessed outcome")
        wdl = returns.reshape(-1, size)
        if "reward" in learner.learning:
            cash = torch.tensor(group["terminal_cash"], dtype=torch.float64,
                                device=device).reshape(*wdl.shape, 2)
            advantage, diagnostic = margin_group_advantages(wdl, cash, learner.learning["reward"])
        else:
            advantage, diagnostic = group_advantages(wdl)
        prepared.append((group, returns, old, advantage.flatten()))
        diagnostics.append({"decision_step": group["batch"]["step"], **diagnostic})
    count = sum(len(ret) for _, ret, _, _ in prepared)
    if not count:
        raise ValueError("nonempty decision cohort required")
    opponent_mode = strata is not None
    counts = {}
    if opponent_mode:
        if len(strata) != len(groups):
            raise ValueError("one opponent label vector per original cohort required")
        for group, labels in zip(groups, strata):
            size = group["group_size"]
            if (len(labels) != len(group["calls"]) or any(x not in ("PASS", "SELF", "SHOP_ROUTER") for x in labels)
                    or any(len(set(labels[i:i+size])) != 1 for i in range(0, len(labels), size))):
                raise ValueError("complete GRPO groups need one supported opponent identity")
            for label in labels: counts[label] = counts.get(label, 0) + 1
        components = ("actor", *("opponent:" + x for x in sorted(counts)))
    else:
        components = ("actor", "weighted_entropy", "weighted_plan_critic", "weighted_future", "combined")
    result, actor_gradients, partition_gradients = {}, {}, {}
    try:
        for component in components:
            for _, parameter in named:
                parameter.grad = None
            loss_sum = 0.
            for ordinal, (group, returns, old, advantage) in enumerate(prepared):
                output = learner.outputs_stacked([group], with_plan=True)
                if component.startswith("opponent:"):
                    selected = torch.tensor([x == component.split(":", 1)[1] for x in strata[ordinal]], device=device)
                    advantage = advantage * selected
                pg = grpo_terms(output[0], old, advantage, output[2],
                    clip_ratio=learner.settings["clip_ratio"],
                    entropy_coefficient=learner.settings["entropy_coefficient"])
                if opponent_mode:
                    loss = pg["actor_loss"]
                else:
                    critic, future = learner.auxiliary_terms(output[3], output[4], [group], returns)
                    losses = {"actor": pg["actor_loss"],
                        "weighted_entropy": -learner.settings["entropy_coefficient"] * pg["entropy"],
                        "weighted_plan_critic": learner.learning["critic_coefficient"] * critic["loss"],
                        "weighted_future": learner.learning["future_coefficient"] * future}
                    loss = sum(losses.values()) if component == "combined" else losses[component]
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite " + component + " loss")
                weight = len(returns) / count
                (loss * weight).backward()
                loss_sum += float(loss.detach()) * weight
            sums, active, absent = {}, [], []
            policy_sq = auxiliary_sq = actor_dot = 0.
            for name, parameter in named:
                if parameter.grad is None:
                    absent.append(name)
                    continue
                grad = parameter.grad.detach()
                if not bool(torch.isfinite(grad).all()):
                    raise ValueError("nonfinite " + component + " gradient: " + name)
                sq = float(grad.double().square().sum())
                key = ".".join(name.split(".")[:2]) if name.startswith(("blocks.", "auxiliary.")) else name.split(".")[0]
                sums[key] = sums.get(key, 0.) + sq
                if name.startswith("auxiliary."): auxiliary_sq += sq
                else:
                    policy_sq += sq
                    if component == "actor":
                        actor_gradients[name] = grad.to(device="cpu", copy=True)
                        actor_dot += sq
                    elif name in actor_gradients:
                        cpu_grad = grad.to(device="cpu", copy=True)
                        actor_dot += float(torch.dot(actor_gradients[name].double().flatten(),
                                                     cpu_grad.double().flatten()))
                        if opponent_mode:
                            if name in partition_gradients: partition_gradients[name].add_(cpu_grad)
                            else: partition_gradients[name] = cpu_grad
                if sq: active.append(name)
            actor_sq = policy_sq if component == "actor" else result["actor"]["policy_l2_norm"] ** 2
            denominator = math.sqrt(actor_sq * policy_sq)
            result[component] = {"loss": loss_sum, "l2_norm": math.sqrt(policy_sq + auxiliary_sq),
                "policy_l2_norm": math.sqrt(policy_sq), "auxiliary_head_l2_norm": math.sqrt(auxiliary_sq),
                "policy_dot_actor": actor_dot,
                "policy_cosine_actor": max(-1., min(1., actor_dot / denominator)) if denominator else None,
                "group_l2_norms": {k: math.sqrt(v) for k, v in sums.items()},
                "nonzero_parameter_names": active, "disconnected_parameter_names": absent}
        if versions != {n: p._version for n, p in named}:
            raise ValueError("gradient-only audit mutated parameters")
        actor_norm = result["actor"]["policy_l2_norm"]
        if opponent_mode:
            residual = math.sqrt(sum(float((partition_gradients.get(n, torch.zeros_like(g)).double() - g.double()).square().sum())
                                     for n, g in actor_gradients.items()))
            for label, n in counts.items():
                result["opponent:" + label].update(decisions=n, fraction_of_original_cohort=n/count,
                    conditional_mean_policy_l2_norm=result["opponent:" + label]["policy_l2_norm"]*count/n)
            return {"components": result, "decision_diagnostics": diagnostics, "decisions": count,
                "opponent_decisions": counts, "actor_signal_nonzero": actor_norm > 0,
                "auxiliary_to_actor_policy_norm": {},
                "partition_residual_l2_norm": residual,
                "partition_residual_relative_to_actor": residual/actor_norm if actor_norm else None,
                "scope": "RECORDED_COHORT_ACTOR_GRADIENT_BY_OPPONENT; ORIGINAL_INPUT_PADDING_AND_TOTAL_DECISION_DENOMINATOR; NO_AUXILIARY_LOSS_OR_ADAM_EFFECT; NOT_FULL_ROUND_CAUSALITY"}
        return {"components": result, "decision_diagnostics": diagnostics, "decisions": count,
            "auxiliary_to_actor_policy_norm": {k: result[k]["policy_l2_norm"] / actor_norm if actor_norm else None
                for k in ("weighted_plan_critic", "weighted_future")},
            "actor_signal_nonzero": actor_norm > 0,
            "scope": "ONE_COLLECTION_COHORT_ALL_DECISION_TIMES; ORIGINAL_PADDING; POLICY_GRADIENT_ALIGNMENT_NOT_ADAM_UPDATES_OR_FULL_ROUND_NOISE_SCALE"}
    finally:
        for _, parameter in named:
            parameter.grad = None


from farm_neat.training.campaign import weighted_actor_sum


def paired_optimizer_effects(factory, groups, strata, *, router_weight, progress=None):
    """Two cold parent loads, one ephemeral combined-loss step each; no publish.

    The caller supplies the maintained Learner/AdamW8bit factory and immutable
    admitted inputs. Small CPU fixtures may supply an ordinary AdamW factory.
    This is one cohort, not a campaign update or a playing-strength comparison.
    """
    import collections
    import gc
    import math
    import torch
    from farm_neat.training.moe_lab import _state_digest
    from farm_neat.training.schedule import learning_rate_at
    from farm_neat.policy.plan_learning import group_advantages, margin_group_advantages
    if type(router_weight) not in (int, float) or not math.isfinite(router_weight) or router_weight <= 0:
        raise ValueError("finite positive router weight required")
    if not groups or len(groups) != len(strata):
        raise ValueError("one label vector per nonempty original cohort required")
    counts = collections.Counter()
    for group, labels in zip(groups, strata, strict=True):
        size = group["group_size"]
        if (type(size) is not int or size < 2 or not labels or len(labels) % size
                or len(labels) != len(group["calls"]) or len(labels) != len(group["returns"])
                or any(x not in ("PASS", "SELF", "SHOP_ROUTER") for x in labels)
                or any(len(set(labels[i:i+size])) != 1 for i in range(0, len(labels), size))):
            raise ValueError("complete same-opponent GRPO groups required")
        counts.update(labels)
    if not counts["SHOP_ROUTER"] or sum(counts.values()) == counts["SHOP_ROUTER"]:
        raise ValueError("Router and at least one other opponent are required")
    total = sum(counts.values())
    arms, baseline_gradient, baseline_delta = [], None, None
    emit = progress if progress is not None else lambda event: None
    rng_cpu = torch.get_rng_state().clone()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

    def vector(named, *, gradients=False):
        return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().cpu().flatten()
                          if gradients else p.detach().cpu().flatten() for _, p in named])

    def compare(value, reference):
        v, r = value.double(), reference.double()
        norm, ref_norm = float(v.norm()), float(r.norm())
        return {"l2_norm": norm, "reference_l2_norm": ref_norm,
                "difference_l2_norm": float((v-r).norm()),
                "cosine": max(-1., min(1., float(torch.dot(v, r))/(norm*ref_norm))) if norm and ref_norm else None}

    try:
        for weight in (1., float(router_weight)):
            emit({"event": "ARM_STARTED", "arm": len(arms), "router_weight": weight})
            learner = factory()
            named = list(learner.graph.named_parameters()) + [
                ("auxiliary."+n, p) for n, p in learner.auxiliary.named_parameters()]
            policy_numel = sum(p.numel() for p in learner.graph.parameters())
            device = named[0][1].device
            start_state = {"model": learner.graph.state_dict(), "optimizer": learner.optimizer.state_dict(),
                "auxiliary": learner.auxiliary.state_dict(), "aux_optimizer": learner.aux_optimizer.state_dict(),
                "age": learner.age, "settings": learner.settings, "learning": learner.learning,
                "parameter_names": [n for n, _ in named], "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
            start_hash = _state_digest(start_state)
            if arms and start_hash != arms[0]["starting_state_sha256"]:
                raise ValueError("both arms must load identical model/Adam/auxiliary/RNG/age/settings")
            del start_state
            before = vector(named)
            prepared = []
            for group, labels in zip(groups, strata, strict=True):
                if group["group_size"] != learner.learning["group_size"]:
                    raise ValueError("registered group size required")
                returns = torch.tensor(group["returns"], dtype=torch.float64, device=device)
                old = torch.tensor([c["old_log_probability"] for c in group["calls"]], dtype=torch.float64, device=device)
                wdl = returns.reshape(-1, group["group_size"])
                if "reward" in learner.learning:
                    cash = torch.tensor(group["terminal_cash"], dtype=torch.float64, device=device).reshape(*wdl.shape, 2)
                    adv, _ = margin_group_advantages(wdl, cash, learner.learning["reward"])
                else:
                    adv, _ = group_advantages(wdl)
                prepared.append((group, labels, returns, old, adv.flatten()))
            labels = {"old_logp": torch.cat([p[3] for p in prepared]),
                      "old_value": torch.tensor([c["old_value"] for g in groups for c in g["calls"]], dtype=torch.float64, device=device),
                      "returns": torch.cat([p[2] for p in prepared])}
            parity = learner.measure(groups, labels, parity=True)
            from farm_neat.training import campaign
            if (parity["old_logp_max_error"] > campaign.fast.TOLERANCES["within_profile_logp_atol"]
                    or parity["old_value_max_error"] > campaign.fast.TOLERANCES["within_profile_value_atol"]):
                raise ValueError("original-padding behavior parity failed before disposable update")
            for _, p in named: p.grad = None
            denominator = total + (weight-1.)*counts["SHOP_ROUTER"]
            losses = collections.Counter()
            for group, labels, returns, old, adv in prepared:
                output = learner.outputs_stacked([group], with_plan=True)
                weights = torch.tensor([weight if x == "SHOP_ROUTER" else 1. for x in labels], dtype=torch.float64, device=device)
                actor = weighted_actor_sum(output[0], old, adv, output[2], weights, denominator, learner.settings)
                critic, future = learner.auxiliary_terms(output[3], output[4], [group], returns)
                fraction = len(returns)/total
                terms = {"actor": actor, "entropy": -learner.settings["entropy_coefficient"]*output[2].mean()*fraction,
                    "critic": learner.learning["critic_coefficient"]*critic["loss"]*fraction,
                    "future": learner.learning["future_coefficient"]*future*fraction}
                loss = sum(terms.values())
                if not bool(torch.isfinite(loss)): raise ValueError("nonfinite probe loss")
                loss.backward()
                for key, value in terms.items(): losses[key] += float(value.detach())
            grad = vector(named, gradients=True)
            if not bool(torch.isfinite(grad).all()): raise ValueError("nonfinite probe gradient")
            if baseline_gradient is None: baseline_gradient = grad.clone()
            preclip = compare(grad, baseline_gradient)
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], learner.settings["gradient_clip"], error_if_nonfinite=True)
            clipped_norm = float(vector(named, gradients=True).double().norm())
            lr = learning_rate_at(learner.age+1, learner.settings)
            for role, optimizer in (("policy", learner.optimizer), ("auxiliary", learner.aux_optimizer)):
                for param_group in optimizer.param_groups: param_group["lr"] = lr
                emit({"event": "EPHEMERAL_STEP_STARTED", "arm": len(arms), "optimizer": role})
                optimizer.step()
                emit({"event": "EPHEMERAL_STEP_COMPLETED", "arm": len(arms), "optimizer": role})
            after = vector(named)
            if not bool(torch.isfinite(after).all()): raise ValueError("nonfinite disposable update")
            delta = after-before
            if baseline_delta is None: baseline_delta = delta.clone()
            policy_delta = compare(delta[:policy_numel], baseline_delta[:policy_numel])
            aux_delta = compare(delta[policy_numel:], baseline_delta[policy_numel:])
            likelihoods = collections.defaultdict(list)
            with torch.no_grad():
                for group, labels, returns, old, adv in prepared:
                    logp = learner.outputs_stacked([group], with_plan=False)[0]
                    diff = logp-old
                    ratio = diff.exp()
                    surrogate = torch.minimum(ratio*adv, ratio.clamp(1-learner.settings["clip_ratio"], 1+learner.settings["clip_ratio"])*adv)
                    values = torch.stack([diff, ratio-1-diff, surrogate], dim=1)
                    if not bool(torch.isfinite(values).all()): raise ValueError("nonfinite post-update likelihood")
                    for label, row in zip(labels, values.cpu().tolist(), strict=True): likelihoods[label].append(row)
            arms.append({"router_weight": weight, "starting_state_sha256": start_hash,
                "starting_age": learner.age, "ephemeral_age_after": learner.age+1,
                "optimizer_types": [type(o).__module__+"."+type(o).__name__ for o in (learner.optimizer, learner.aux_optimizer)],
                "decision_denominator": total, "weighted_actor_denominator": denominator,
                "losses": dict(losses), "parity": parity, "combined_preclip_gradient_vs_weight1": preclip,
                "clip_grad_norm_return": float(norm), "combined_postclip_l2_norm": clipped_norm,
                "learning_rate": lr, "policy_update_vs_weight1": policy_delta, "auxiliary_update_vs_weight1": aux_delta,
                "opponent_likelihood_after": {k: {"decisions": len(v),
                    "mean_new_minus_old_logp": sum(r[0] for r in v)/len(v),
                    "sampled_kl_k3": sum(r[1] for r in v)/len(v),
                    "mean_clipped_advantage_surrogate": sum(r[2] for r in v)/len(v)} for k,v in likelihoods.items()}})
            emit({"event": "ARM_COMPLETED", "arm": len(arms)-1, "router_weight": weight})
            # No publication or tensor serialization: discard this arm completely.
            del learner, named, before, after, delta, grad, prepared, output, terms, loss, actor, critic, future, logp, optimizer, param_group, p, value
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
        return {"arms": arms, "decisions": total, "opponent_decisions": dict(counts),
                "ephemeral_policy_optimizer_steps": 2, "ephemeral_auxiliary_optimizer_steps": 2,
                "committed_optimizer_updates": 0, "checkpoint_published": False,
                "scope": "TWO_DISPOSABLE_SAME_PARENT_COMBINED_LOSS_STEPS_ON_ONE_RETAINED_COHORT; NOT_FULL_ROUND_OR_STRENGTH"}
    finally:
        torch.set_rng_state(rng_cpu)
        if rng_cuda: torch.cuda.set_rng_state_all(rng_cuda)


def recorded_cohort(game, expected_plan_sha256, admission_path, expected_admission_sha256):
    """Recover the exact previously admitted learner inputs; execute no game.

    The caller pins the admission produced by the original physical witness.
    This read joins its episode digest, including future labels and input hashes,
    to saved turns and sidecars. It is not a new independent environment check.
    """
    import copy
    import math
    import numpy as np
    from farm_neat.training import campaign as c

    def require(condition, message):
        if not condition:
            raise ValueError("recorded cohort: " + message)

    def pinned(path, expected):
        require(path.is_file() and not path.is_symlink() and file_sha(path) == expected,
                "missing or changed " + str(path))

    game = c.io.path(ROOT, game)
    admission_path = c.io.path(ROOT, admission_path)
    pinned(game / "PLAN.json", expected_plan_sha256)
    pinned(admission_path, expected_admission_sha256)
    plan, admission = c.io.read(game / "PLAN.json"), c.io.read(admission_path)
    c.validate_cohort_plan(plan)
    c._sealed(admission)
    require(plan.get("plan_learning") and plan.get("full_plan"), "grouped full-plan cohort required")
    require(admission["schema"] == c.SCHEMA + "-executed-admission"
            and admission["model_sha256"] == plan["model_sha256"]
            and admission["source_pins"] == plan["source_pins"], "admission model or source mismatch")
    c.attest(plan["source_pins"])
    witnesses = {}
    for witness in admission["engine_witnesses"]:
        c._sealed(witness)
        key = witness["producer_result_sha256"]
        require(key not in witnesses, "duplicate retained witness")
        witnesses[key] = witness
    collection = c.io.read(game / "RESULT.json")
    c._sealed(collection)
    require(collection["plan_sha256"] == plan["content_sha256"]
            and collection["model_sha256"] == plan["model_sha256"]
            and collection["physical_games"] == len(plan["seeds"])
            and len(collection["world_results"]) == len(plan["seeds"]), "producer identity or count mismatch")
    pinned(game / "COHORTS.json", collection["cohorts_file_sha256"])
    cohorts = c.io.read(game / "COHORTS.json")
    c._sealed(cohorts)
    require(cohorts["plan_sha256"] == plan["content_sha256"], "recorded cohort plan mismatch")
    episodes, seen = [], set()
    for index, member in enumerate(collection["world_results"]):
        require(member["world_index"] == index and plan["lane_order"][index][0] == index,
                "one declared learner lane per world required")
        world = game / f"world-{index:03d}"
        for name, key in (("RESULT.json", "result_file_sha256"),
                          ("turns.jsonl.gz", "turns_sha256"), (c.INPUT_SIDECAR, "inputs_sha256")):
            pinned(world / name, member[key])
        result = c.io.read(world / "RESULT.json")
        c._sealed(result)
        registration = c.world_plan(plan, index)
        witness = witnesses.get(result["content_sha256"])
        require(witness is not None and witness["content_sha256"] not in seen,
                "missing or reused retained engine witness")
        require(witness["plan_sha256"] == plan["content_sha256"]
                and witness["world_plan_sha256"] == result["plan_sha256"] == registration["content_sha256"]
                and witness["turns_sha256"] == result["turns_sha256"] == member["turns_sha256"]
                and witness["terminal_cash"] == result["cash"]
                and witness["physical_steps"] == 719 and witness["reconstructed_physical_games"] == 1,
                "retained world identity, terminal or trajectory mismatch")
        require(len(result["cash"]) == 2 and all(type(x) in (int, float) and math.isfinite(x) and x >= 0
                                                for x in result["cash"]), "invalid witnessed cash")
        seat = plan["lane_order"][index][1]
        delta = result["cash"][seat] - result["cash"][1-seat]
        episode = {"schema": c.EPISODE_SCHEMA,
            "episode_id": registration["game_id"] + f"-seat{seat}",
            "seed": registration["seed"], "seat": seat,
            "behavior_model_sha256": plan["model_sha256"],
            "model_manifest_sha256": plan["model_manifest_sha256"], "numerics": plan["numerics"],
            "rollout_iteration": plan["rollout_iteration"], "source_sha256": plan["source_sha256"],
            "calls": [], "terminal_reward": int(delta > 0) - int(delta < 0),
            "future": copy.deepcopy(result["future"]), "group_id": registration["group_id"],
            "group_size": plan["plan_learning"]["group_size"],
            "opponent_sha256": c.digest(registration["opponent"])}
        if "reward" in plan["plan_learning"]:
            episode["terminal_cash"] = [result["cash"][seat], result["cash"][1-seat]]
        turns = 0
        with gzip.open(world / "turns.jsonl.gz", "rt") as stream:
            for line in stream:
                row = json.loads(line)
                require(row["step"] == turns and turns < 719, "incomplete or repeated physical suffix")
                turns += 1
                if row["decisions"] is not None:
                    call = row["decisions"][seat]
                    require(isinstance(call, dict) and call["step"] == row["step"] and call["seat"] == seat,
                            "wrong recorded learner decision")
                    c._sealed(call)
                    episode["calls"].append(call)
        require(turns == 719 and [x["step"] for x in episode["calls"]] == list(c.decision_steps(plan)),
                "complete registered game and decision clock required")
        require(c.digest({"episodes": [episode]}) == witness["training_batch_sha256"],
                "recorded inputs, behavior or labels differ from retained admission")
        with np.load(world / c.INPUT_SIDECAR, allow_pickle=False) as arrays:
            for call in episode["calls"]:
                call["input"] = c.recorded_pack(call["input"], arrays)
        episodes.append(episode)
        seen.add(witness["content_sha256"])
    groups = c.cohort_batches(plan, episodes)
    require([g["batch"] for g in groups] == cohorts["batches"], "original padding or input identity differs")
    peer = [w for w in admission.get("neural_opponent_witnesses", [])
            if w["cohort_plan_sha256"] == plan["content_sha256"]]
    expected_peer_calls = len(c.opponent_lanes(plan)) * len(c.decision_steps(plan))
    require(len(peer) == int(expected_peer_calls > 0), "complete retained neural opponent admission required")
    for witness in peer:
        c._sealed(witness)
        require(witness["model_sha256"] == plan["model_sha256"] and witness["opponent_programs_and_factors_exact"]
                and witness["decision_calls"] == expected_peer_calls
                and 0 <= witness["max_logp_error"] <= c.fast.TOLERANCES["within_profile_logp_atol"]
                and 0 <= witness["max_value_error"] <= c.fast.TOLERANCES["within_profile_value_atol"],
                "retained neural opponent check failed")
    pinned(game / "PLAN.json", expected_plan_sha256)
    pinned(admission_path, expected_admission_sha256)
    c.attest(plan["source_pins"])
    return plan, groups, {"admission_file_sha256": expected_admission_sha256,
        "admission_content_sha256": admission["content_sha256"],
        "retained_engine_witness_sha256": sorted(seen), "retained_peer_checks": peer,
        "previously_admitted_physical_games": len(episodes), "physical_games_reconstructed": 0,
        "scope": "EXACT_RETAINED_ADMISSION_INPUT_AND_LABEL_JOIN; NO_NEW_ENVIRONMENT_OR_PEER_RESAMPLING"}


def grouped_gradient_audit(args, output):
    """Fresh engine + peer/model reconstruction, then zero-update derivatives."""
    import os
    import torch
    from farm_neat.training import campaign as campaign
    game, model = Path(args.game).resolve(), Path(args.model).resolve()
    if not sys.dont_write_bytecode or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("-B and explicit CUDA numerical profile required")
    if file_sha(game / "PLAN.json") != args.expect_plan_sha256:
        raise ValueError("exact collection plan hash required")
    plan = campaign.io.read(game / "PLAN.json")
    campaign.validate_cohort_plan(plan)
    if not plan.get("plan_learning") or plan["model_sha256"] != args.expect_model_sha256:
        raise ValueError("same-parent grouped cohort required")
    if campaign.environment_profile() != plan["environment"]:
        raise ValueError("exact fast collection numerical profile required")
    expected = campaign.io.checkpoint(ROOT, model.parent)
    if (model.name != "model.pt" or expected["model_sha256"] != args.expect_model_sha256
            or expected["optimizer_sha256"] != args.expect_optimizer_sha256
            or expected["manifest_sha256"] != plan["model_manifest_sha256"]):
        raise ValueError("exact same-parent model, auxiliary state and Adam required")
    episodes, peers, witnesses, retained = [], [], [], None
    if args.recorded_grouped_gradient:
        plan, groups, retained = recorded_cohort(game, args.expect_plan_sha256,
                                                args.admission, args.expect_admission_sha256)
    else:
        for i in range(len(plan["seeds"])):
            checked = campaign.witness_world(game, i, plan["content_sha256"])
            episodes.extend(checked["episodes"])
            peers.extend(checked.get("opponent_episodes", []))
            witnesses.append(checked["engine_witness"]["content_sha256"])
        groups = campaign.cohort_batches(plan, episodes)
    def factory():
        return campaign.Learner(model.parent, expected, stack=1, checkpoint_blocks=True,
            opening_only=plan["opening_only"], plan_learning=plan["plan_learning"],
            shop_revision_every=plan.get("shop_revision_every", 0), full_plan=plan.get("full_plan", False))
    if args.opponent_update_probe:
        labels = []
        for i in range(len(plan["seeds"])):
            opponent = campaign.opponent_for(plan, i)
            kind = opponent["kind"]
            if kind == campaign.NEURAL_PEER: label = "SELF"
            elif kind == "SHOP_ROUTER": label = "SHOP_ROUTER"
            elif kind == "FROZEN_ECONOMIC_PROGRAM" and opponent["program"] == [{"op": "STOP"}]: label = "PASS"
            else: raise ValueError("unsupported opponent stratum")
            labels.append(label)
        def progress(event):
            with (output / "attempts.jsonl").open("a") as stream:
                stream.write(canonical(event)+"\n"); stream.flush(); os.fsync(stream.fileno())
        report = paired_optimizer_effects(factory, groups, [list(labels) for _ in groups],
            router_weight=args.probe_router_weight, progress=progress)
        if campaign.io.checkpoint(ROOT, model.parent) != expected:
            raise ValueError("checkpoint changed during disposable optimizer probe")
        campaign.attest(plan["source_pins"])
        result = {"schema": "farm-full-economic-opponent-optimizer-probe-v1", "status": "PASS",
            "source_sha256": file_sha(__file__), "source_pins": campaign.sources(),
            "checkpoint": expected, "plan_file_sha256": args.expect_plan_sha256,
            "cohort": str(game.relative_to(ROOT)), "retained_admission": retained,
            "physical_games_reconstructed": 0, "fresh_games_collected": 0,
            "optimizer_updates": 2, "in_memory_weights_changed": True,
            "checkpoint_files_changed": False, "strength_claim": False, **report}
        write(output / "RESULT.json", result)
        return result
    learner = factory()
    learner.load_behavior(model.parent, args.expect_model_sha256)
    peer_check = learner.verify_opponents(plan, campaign.cohort_opponent_batches(plan, peers)) if peers else None
    # Reuse the maintained original-padding likelihood gate; do not loosen it.
    labels = {
        "old_logp": torch.tensor([c["old_log_probability"] for g in groups for c in g["calls"]], dtype=torch.float64, device="cuda"),
        "old_value": torch.tensor([c["old_value"] for g in groups for c in g["calls"]], dtype=torch.float64, device="cuda"),
        "returns": torch.tensor([r for g in groups for r in g["returns"]], dtype=torch.float64, device="cuda")}
    parity = learner.measure(groups, labels, parity=True)
    if (parity["old_logp_max_error"] > campaign.fast.TOLERANCES["within_profile_logp_atol"]
            or parity["old_value_max_error"] > campaign.fast.TOLERANCES["within_profile_value_atol"]):
        raise ValueError("actual behavior replay parity failed")
    strata = None
    if args.opponent_components:
        labels = []
        for i in range(len(plan["seeds"])):
            opponent = campaign.opponent_for(plan, i)
            kind = opponent["kind"]
            if kind == campaign.NEURAL_PEER: label = "SELF"
            elif kind == "SHOP_ROUTER": label = "SHOP_ROUTER"
            elif kind == "FROZEN_ECONOMIC_PROGRAM" and opponent["program"] == [{"op": "STOP"}]: label = "PASS"
            else: raise ValueError("unsupported opponent stratum")
            labels.append(label)
        strata = [list(labels) for _ in groups]
    report = grouped_gradient_components(learner, groups, strata=strata)
    if campaign.io.checkpoint(ROOT, model.parent) != expected:
        raise ValueError("checkpoint changed during gradient audit")
    campaign.attest(plan["source_pins"])
    result = {"schema": "farm-full-economic-gradient-audit-v3", "status": "PASS",
        "source_sha256": file_sha(__file__), "checkpoint": expected,
        "plan_file_sha256": args.expect_plan_sha256, "cohort": str(game.relative_to(ROOT)),
        "physical_games_reconstructed": 0 if retained else len(plan["seeds"]),
        "engine_witness_sha256": witnesses, "retained_admission": retained,
        "peer_check": peer_check, "source_pins": campaign.sources(), "parity": parity,
        **report, "fresh_games_collected": 0, "optimizer_updates": 0,
        "weights_changed": False, "engine_differentiated": False, "strength_claim": False,
        "zero_actor_signal_is_not_automatically_a_bug": True}
    write(output / "RESULT.json", result)
    return result


def closed_block_audit(run, expected_plan_sha256, expected_result_sha256):
    """Stream every raw turn/learner pack and join an already admitted closed run.

    Explicit PLAN/RESULT pins anchor the read; current sources must still match.
    Intermediate retired model tensors are unnecessary. The final model/Adam
    files must exist and match, but actual cold optimizer/RNG loading and paired
    evaluation remain separate checks. No inferred terminal labels or fitting.
    Reconciled multi-run blocks retain their separate composition audit.
    """
    import collections
    import math
    import numpy as np
    from farm_neat.training import campaign as c

    def require(condition, message):
        if not condition:
            raise ValueError("closed block: " + message)

    def pinned(path, digest):
        require(path.is_file() and not path.is_symlink() and file_sha(path) == digest,
                "missing or changed " + str(path))

    run = Path(run)
    pinned(run / "PLAN.json", expected_plan_sha256)
    pinned(run / "RESULT.json", expected_result_sha256)
    plan, result = c.io.read(run / "PLAN.json"), c.io.read(run / "RESULT.json")
    c._sealed(plan)
    c.attest(plan["source_pins"])
    require(plan.get("full_plan") and plan.get("plan_learning") and not plan["overlap"],
            "registered non-overlapped grouped full-plan run required")
    rounds, per_round = plan["rounds"], plan["games_per_round"]
    require(type(rounds) is int and rounds > 0 and type(per_round) is int and per_round > 0,
            "positive round allocation required")
    games = rounds * per_round
    require(games == plan["physical_games"] == len(plan["world_seeds"]) == len(plan["actor_rng_seeds"]),
            "complete declared allocation required")
    require(result.get("status") == "FRESH_FAST_PROFILE_GRPO_AUX_ROUNDS_CLOSED_NOT_STRENGTH"
            and result["physical_games"] == games and result["uncommitted_collected_games"] == 0,
            "run is not complete; partial/failed runs are not a closed block")
    require(result["owner_plan_file_sha256"] == expected_plan_sha256
            and result["optimizer_updates"] == rounds * plan["updates_per_round"], "result allocation mismatch")
    require(plan["updates_per_round"] == 1, "this audit supports one fully accumulated update per round")
    counts, ops, products, attention = (collections.Counter() for _ in range(4))
    allocations, samplers, best, learning, summaries = [], [], [], [], []
    witnesses = peer_calls = 0
    last_model, last_adam = (plan["parent"][k] for k in ("model_sha256", "optimizer_sha256"))
    start_age = plan["parent"]["optimizer_age"]
    for ordinal in range(rounds):
        rd = run / "rounds" / f"round-{ordinal:03d}"
        summary, fit = c.io.read(rd / "RESULT.json"), c.io.read(rd / "fit/RESULT.json")
        summaries.append(summary)
        fp, admission = c.io.read(rd / "fit/PLAN.json"), c.io.read(rd / "fit/ADMISSION.json")
        c._sealed(admission)
        require(fp["owner_plan_file_sha256"] == expected_plan_sha256
                and fit["plan_sha256"] == file_sha(rd / "fit/PLAN.json"), "fit plan mismatch")
        require(fit["admission_sha256"] == admission["content_sha256"] == fp["admission_sha256"], "admission mismatch")
        require(fit["source_pins"] == admission["source_pins"] == plan["source_pins"], "source mismatch")
        require(fp["parent_model_sha256"] == fit["before_model_sha256"] == last_model
                and fp["parent_optimizer_sha256"] == last_adam, "broken model/Adam lineage")
        require(fit["global_optimizer_updates"] == summary["age_after"] == start_age + ordinal + 1
                and fit["optimizer_updates"] == 1 and fit["natural_calls"] == per_round * len(plan["decision_steps"]),
                "incomplete update or decision coverage")
        require(fit["after_model_sha256"] != last_model, "unchanged model is not an update")
        parity = fit["fit"]["parity"]
        require(parity["coverage_fraction"] == 1.0
                and parity["old_logp_max_error"] <= c.fast.TOLERANCES["within_profile_logp_atol"]
                and parity["old_value_max_error"] <= c.fast.TOLERANCES["within_profile_value_atol"], "failed likelihood gate")
        round_peers = 0
        for receipt in admission.get("neural_opponent_witnesses", []):
            c._sealed(receipt)
            require(receipt["opponent_programs_and_factors_exact"]
                    and receipt["max_logp_error"] == receipt["max_value_error"] == 0, "peer resampling mismatch")
            round_peers += receipt["decision_calls"]
        require(round_peers == summary["frozen_opponent_model_decisions"], "missing peer witness coverage")
        peer_calls += round_peers
        require(len(admission["engine_witnesses"]) == per_round, "missing engine witnesses")
        witness_map = {}
        for receipt in admission["engine_witnesses"]:
            c._sealed(receipt)
            key = receipt["producer_result_sha256"]
            require(key not in witness_map, "duplicated engine witness")
            witness_map[key] = receipt
        seen_witnesses = set()
        for producer in sorted((rd / "producers").iterdir()):
            require(producer.is_dir() and not producer.is_symlink(), "invalid producer directory")
            cp, cr = c.io.read(producer / "PLAN.json"), c.io.read(producer / "RESULT.json")
            c.validate_cohort_plan(cp)
            c._sealed(cr)
            require(cp["model_sha256"] == cr["model_sha256"] == last_model, "wrong behavior model")
            require(len(cr["world_results"]) == len(cp["seeds"]), "missing producer worlds")
            allocations.extend(cp["seeds"])
            samplers.extend(cp["actor_rng_seeds"])
            for i, member in enumerate(cr["world_results"]):
                world = producer / f"world-{i:03d}"
                wr = c.io.read(world / "RESULT.json")
                c._sealed(wr)
                pinned(world / "RESULT.json", member["result_file_sha256"])
                pinned(world / "turns.jsonl.gz", member["turns_sha256"])
                pinned(world / c.INPUT_SIDECAR, member["inputs_sha256"])
                require(wr["plan_sha256"] == c.world_plan(cp, i)["content_sha256"], "wrong world identity")
                witness = witness_map.get(wr["content_sha256"])
                require(witness is not None and wr["content_sha256"] not in seen_witnesses, "missing/duplicate world witness")
                require(witness["turns_sha256"] == wr["turns_sha256"] == member["turns_sha256"]
                        and witness["terminal_cash"] == wr["cash"], "raw/outcome witness mismatch")
                seen_witnesses.add(wr["content_sha256"])
                seat = cp["lane_order"][i][1]
                cash, margin = wr["cash"][seat], wr["cash"][seat] - wr["cash"][1-seat]
                counts["physical_games"] += 1
                counts["wins" if margin > 0 else "losses" if margin < 0 else "draws"] += 1
                counts["cash_above_start_games"] += cash > 3000
                active = plants = harvests = turns = 0
                calls = []
                with np.load(world / c.INPUT_SIDECAR, allow_pickle=False) as arrays, gzip.open(world / "turns.jsonl.gz", "rt") as stream:
                    for line in stream:
                        row = json.loads(line)
                        require(row["step"] == turns and turns < 719, "missing/repeated physical turn")
                        turns += 1
                        action = row["actions"][seat]
                        units = [action["farmer"], *action["hands"]]
                        active += bool(action["market"]) or any(x[0] != "PASS" for x in units)
                        plants += any(x[0] == "PLANT" for x in units)
                        harvests += any(x[0] == "HARVEST" for x in units)
                        call = row["decisions"][seat] if row["decisions"] is not None else None
                        if call is None:
                            continue
                        c._sealed(call)
                        require(call["seat"] == seat and call["step"] == row["step"], "wrong learner call identity")
                        calls.append(row["step"])
                        counts["learner_decisions"] += 1
                        packed = c.recorded_pack(call["input"], arrays)
                        mask = packed["token_mask"]
                        markers = (packed["rows"][:, c.features.KINDS.index("ARGUMENT")] == 1)
                        markers &= packed["knownness"][:, -1] & (packed["rows"][:, -1] > 0) & mask
                        attention["attention_rows"] += int(markers.sum())
                        attention["learner_decisions_with_attention"] += bool(markers.any())
                        attention["total_valid_rows"] += int(mask.sum())
                        program = call["behavior"]["program"]
                        counts["stop_only_calls"] += program == [{"op": "STOP"}]
                        for command in program:
                            ops[command["op"]] += 1
                            if command["op"] == "STOP":
                                continue
                            actual = command["replacement"] if command["op"] == "REPLACE" else command
                            counts["commands_excluding_stop"] += 1
                            counts["expired_deadline_commands"] += type(actual.get("until")) is int and actual["until"] < row["step"]
                            if actual["op"] in ("SET_CROP", "SET_ANIMAL"):
                                products[actual.get("crop", actual.get("animal"))] += 1
                require(turns == 719 and calls == plan["decision_steps"], "incomplete physical game or learner calls")
                counts["idle_games"] += active == 0
                counts["plant_attempt_games"] += plants > 0
                counts["harvest_attempt_games"] += harvests > 0
                best.append({"path": str(world), "seat": seat, "cash": cash, "margin": margin, "active_turns": active})
        require(len(seen_witnesses) == per_round, "unmatched engine witnesses")
        witnesses += len(seen_witnesses)
        fd = fit["fit"]
        learning.append({"age": summary["age_after"], "changed_model_tensors": fd["changed_tensors"],
            "changed_auxiliary_tensors": fd["changed_auxiliary_tensors"], "initial": fd["initial"], "final": fd["final"],
            "history": fd["history"], "group_diagnostics": [d for d in fd["group_diagnostics"] if d.get("decision_step") == 0]})
        last_model, last_adam = fit["after_model_sha256"], fit["optimizer_sha256"]
    require(allocations == plan["world_seeds"] and samplers == plan["actor_rng_seeds"], "seed/sampler allocation mismatch")
    require(len(set(allocations)) == plan["distinct_initial_worlds"] and len(set(samplers)) == games, "wrong independent counts")
    require(counts["physical_games"] == witnesses == games
            and counts["learner_decisions"] == games * len(plan["decision_steps"]), "wrong aggregate coverage")
    child = result["child"]
    require(result["rounds_completed"] == rounds and result["rounds"] == summaries,
            "changed or missing round summaries")
    require(result["model_decisions"] == counts["learner_decisions"]
            and result.get("frozen_opponent_model_decisions", 0) == peer_calls,
            "aggregate decision accounting mismatch")
    require(child["model_sha256"] == last_model and child["optimizer_sha256"] == last_adam
            and child["optimizer_age"] == start_age + rounds, "wrong final checkpoint")
    observed_child = c.io.checkpoint(ROOT, child["directory"])
    require(all(observed_child[k] == v for k, v in child.items()), "changed final checkpoint files")
    seconds = result["elapsed_run_seconds"]
    reported_rate = result["games_per_second_end_to_end_including_learner_startup"]
    require(math.isfinite(seconds) and seconds > 0 and math.isfinite(reported_rate)
            and reported_rate > 0, "invalid throughput accounting")
    # The retained producer reads monotonic() twice when creating these fields.
    # Report the discrepancy; never rewrite either receipt or pretend bit equality.
    recording_gap = games / reported_rate - seconds
    require(-1e-9 <= recording_gap <= .001, "invalid throughput accounting: clock gap exceeds 1ms")
    pinned(run / "PLAN.json", expected_plan_sha256)
    pinned(run / "RESULT.json", expected_result_sha256)
    c.attest(plan["source_pins"])
    return {"schema": "farm-full-economic-closed-block-audit-v1", "status": "CLOSED_SOURCE_RECEIPT_RAW_INPUT_AUDIT_NOT_STRENGTH",
        "auditor_source_sha256": file_sha(__file__), "counts": dict(counts), "command_ops": dict(ops),
        "production_family_commands": dict(products), "attention_inputs": dict(attention),
        "distinct_initial_worlds": len(set(allocations)), "model_decisions": counts["learner_decisions"],
        "optimizer_updates": rounds, "independent_engine_reconstructions_joined": witnesses,
        "frozen_neural_peer_calls": peer_calls, "all_peer_resamples_exact": True,
        "all_original_world_sampler_allocations_match": True, "source_pins": plan["source_pins"],
        "child": child, "learning": learning, "end_to_end_seconds": seconds,
        "end_to_end_games_per_second": games / seconds,
        "reported_games_per_second": reported_rate, "throughput_recording_gap_seconds": recording_gap,
        "most_cash_examples": sorted(best, key=lambda x: (-x["cash"], x["path"]))[:5],
        "scope": "All raw turns/learner inputs joined to retained fresh-engine and neural admission receipts; no new reconstruction, fit, cold optimizer load or strength claim. Issued PLANT/HARVEST are attempts, not verified fills. Post-block time excluded. Partial multi-run compositions need a separate explicit audit.",
        "original_plan_file_sha256": expected_plan_sha256, "result_file_sha256": expected_result_sha256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    parser.add_argument("--model")
    parser.add_argument("--expect-plan-sha256", required=True)
    parser.add_argument("--expect-model-sha256")
    parser.add_argument("--expect-result-sha256", help="required frozen completed RESULT hash for --closed-block")
    parser.add_argument("--expect-optimizer-sha256", help="required for exact grouped auxiliary continuation")
    parser.add_argument("--admission", help="retained ADMISSION.json for the no-game grouped route")
    parser.add_argument("--expect-admission-sha256", help="external hash of the retained physical admission")
    parser.add_argument("--opponent-components", action="store_true", help="partition the recorded actor gradient by opponent with the original full-cohort denominator; no auxiliary backward")
    parser.add_argument("--opponent-update-probe", action="store_true", help="two disposable same-parent combined-loss Adam steps; no checkpoint publication")
    parser.add_argument("--probe-router-weight", type=float, help="explicit candidate Router actor weight versus1 for the disposable update probe")
    parser.add_argument("--output", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--closed-block", action="store_true",
        help="read-only completed full-plan receipt/raw/input audit; --game is RUN, --output is a new JSON file under RUN/checks")
    modes.add_argument("--fast-gradient", action="store_true", help="old opening-only PPO component audit; no fit")
    modes.add_argument("--grouped-gradient", action="store_true", help="current GRPO/entropy/plan-critic/future gradients, all cohort decision times; no fit")
    modes.add_argument("--recorded-grouped-gradient", action="store_true",
        help="same grouped derivative probe from exact retained admission and actor inputs; no gameplay")
    parser.add_argument("--export-episodes", action="store_true",
                        help="optional compressed derived tensors; normally reconstruct from raw game")
    args = parser.parse_args()
    if args.opponent_components and not args.recorded_grouped_gradient:
        parser.error("--opponent-components requires --recorded-grouped-gradient")
    if args.opponent_update_probe:
        import math
        if (not args.recorded_grouped_gradient or args.opponent_components
                or args.probe_router_weight is None or not math.isfinite(args.probe_router_weight)
                or args.probe_router_weight <= 0):
            parser.error("--opponent-update-probe requires recorded admission and an explicit positive --probe-router-weight; excludes --opponent-components")
    elif args.probe_router_weight is not None:
        parser.error("--probe-router-weight requires --opponent-update-probe")
    if args.recorded_grouped_gradient:
        if not args.admission or not args.expect_admission_sha256 or args.export_episodes:
            parser.error("--recorded-grouped-gradient requires both admission arguments and excludes episode export")
    elif args.admission or args.expect_admission_sha256:
        parser.error("admission arguments require --recorded-grouped-gradient")
    if args.closed_block:
        if not args.expect_result_sha256 or args.export_episodes or args.model or args.expect_model_sha256 or args.expect_optimizer_sha256:
            parser.error("--closed-block requires --expect-result-sha256 and excludes model/episode/gradient options")
        run, target = Path(args.game).absolute(), Path(args.output).absolute()
        if (run != run.resolve() or not run.is_relative_to(ROOT / "work/runs")
                or target != target.resolve() or not target.is_relative_to(run / "checks")
                or target.suffix != ".json" or target.exists()):
            parser.error("--closed-block needs an actual work/runs run and a new JSON file under its checks directory")
        result = closed_block_audit(run, args.expect_plan_sha256, args.expect_result_sha256)
        write(target, result)
        print(canonical({k: result[k] for k in ("status", "counts", "child", "end_to_end_seconds",
            "end_to_end_games_per_second", "attention_inputs", "scope")}), flush=True)
        return
    if not args.model or not args.expect_model_sha256 or args.expect_result_sha256:
        parser.error("game/gradient checks require --model and --expect-model-sha256; result hash is only for --closed-block")
    if (args.grouped_gradient or args.recorded_grouped_gradient) and not args.expect_optimizer_sha256:
        parser.error("grouped gradients require --expect-optimizer-sha256")
    output = output_directory(args.output)
    write(output / "REQUEST.json", {**vars(args), "environment": device_profile(),
        "command_source_sha256": file_sha(__file__), "optimizer_updates": 0,
        "maximum_ephemeral_policy_steps": 2 if args.opponent_update_probe else 0,
        "maximum_committed_updates": 0})
    started = time.monotonic()
    if args.grouped_gradient or args.recorded_grouped_gradient:
        result = grouped_gradient_audit(args, output)
        print(canonical({k: result[k] for k in ("status", "decisions", "physical_games_reconstructed",
            "actor_signal_nonzero", "auxiliary_to_actor_policy_norm", "scope", "optimizer_updates",
            "committed_optimizer_updates", "ephemeral_policy_optimizer_steps", "arms") if k in result}), flush=True)
        return
    if args.fast_gradient:
        result = fast_gradient_audit(args, output)
        print(canonical({k: v for k, v in result.items() if k not in ("components", "source_pins")}), flush=True)
        if result["status"] != "PASS": raise SystemExit(1)
        return
    checked = admit(args.game, args.model, expected_plan_file_sha256=args.expect_plan_sha256,
                    expected_model_sha256=args.expect_model_sha256)
    episodes_sha = None
    if args.export_episodes:
        with gzip.open(output / "EPISODES.json.gz", "xt") as stream:
            json.dump(checked["episodes"], stream, separators=(",", ":"), allow_nan=False)
        episodes_sha = file_sha(output / "EPISODES.json.gz")
    write(output / "ADMISSION.json", checked["admission"])
    result = {"schema": "farm-full-economic-game-check-v1", "status": "PASS",
        "admission_file_sha256": file_sha(output / "ADMISSION.json"),
        "episodes_file_sha256": episodes_sha,
        "episodes_exported": args.export_episodes,
        "reconstruction_input": str(Path(args.game)),
        "seconds": time.monotonic() - started, "optimizer_updates": 0,
        "strength_claim": False, "admission": checked["admission"]}
    write(output / "RESULT.json", result)
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
