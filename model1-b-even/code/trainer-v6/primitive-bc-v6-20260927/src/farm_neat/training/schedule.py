"""Fresh-trajectory PPO with a persistent optimizer and campaign-wide schedule.

The caller supplies independently reconstructed full episodes. This module
checks behavior identities and numerical replay; it is not an engine witness.
No historical outcome rows or unchosen action labels are admitted implicitly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from farm_neat.training.schedule_model import (ACTION_IDS, NORMALIZATION, TRAINABLE,
    NUMERICS_CANONICAL, NUMERICS_LEGACY, numerical_contract,
    bound, canonical, file_sha, load_model, save_model, tensor_sha)
from farm_neat.training.schedule_policy import ActorCriticPolicy, validate_inputs


def default_settings(planned_optimizer_steps=1024):
    return {"planned_optimizer_steps": planned_optimizer_steps, "learning_rate": 3e-4,
        "warmup_fraction": 0.1, "warmup_steps": 1000,
        "initial_lr_fraction": 0.1, "final_lr_fraction": 0.1,
        "epochs": 4, "minibatch_size": 256, "clip_ratio": 0.2,
        "value_coefficient": 0.5, "entropy_coefficient": 0.01,
        "gradient_clip": 0.5, "target_kl": 0.02, "weight_decay": 0.0,
        "seed": 9062026, "normalize_advantages": True}


def validate_settings(settings):
    if set(settings) != set(default_settings()):
        raise ValueError("closed declared PPO settings required")
    for key in ("planned_optimizer_steps", "epochs", "minibatch_size", "warmup_steps"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError("positive integer " + key + " required")
    if type(settings["seed"]) is not int or settings["seed"] < 0:
        raise ValueError("nonnegative training RNG seed required")
    if type(settings["normalize_advantages"]) is not bool:
        raise ValueError("explicit advantage normalization required")
    for key, value in settings.items():
        if key in ("seed", "normalize_advantages", "planned_optimizer_steps", "epochs", "minibatch_size", "warmup_steps"):
            continue
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative " + key + " required")
    if (settings["learning_rate"] <= 0 or settings["gradient_clip"] <= 0
            or settings["target_kl"] <= 0 or not 0 < settings["clip_ratio"] < 1
            or not 0 < settings["warmup_fraction"] < 1
            or not 0 < settings["initial_lr_fraction"] <= 1
            or not 0 < settings["final_lr_fraction"] <= 1):
        raise ValueError("invalid PPO trust region or schedule")


def learning_rate_at(step, settings):
    """One-based global optimizer step; warmup never restarts on a new rollout."""
    budget = settings["planned_optimizer_steps"]
    if type(step) is not int or not 1 <= step <= budget:
        raise ValueError("optimizer step outside declared campaign budget")
    warmup = max(1, min(settings["warmup_steps"], math.ceil(budget * settings["warmup_fraction"])))
    if step <= warmup:
        progress = (step - 1) / max(1, warmup - 1)
        scale = settings["initial_lr_fraction"] + (1 - settings["initial_lr_fraction"]) * progress
    else:
        progress = (step - warmup) / max(1, budget - warmup)
        scale = settings["final_lr_fraction"] + (1 - settings["final_lr_fraction"]) * (1 + math.cos(math.pi * progress)) / 2
    return settings["learning_rate"] * scale


def torch_graph(weights, numerics=NUMERICS_LEGACY):
    import torch
    from torch import nn
    numerical_contract({"numerics": numerics})

    class Graph(nn.Module):
        def __init__(self):
            super().__init__()
            for key in NORMALIZATION:
                self.register_buffer(key, torch.tensor(weights[key].copy()))
            self.public_linear = nn.Linear(166, 32)
            self.candidate_linear = nn.Linear(168, 64)
            self.actor_head = nn.Linear(64, 1)
            self.value_head = nn.Linear(64, 1)
            self.load_state_dict({key: torch.tensor(value.copy()) for key, value in weights.items()})

        def forward(self, rows, enabled):
            if numerics == NUMERICS_CANONICAL:
                # Casts are differentiable: gradients return to the original
                # float32 Parameters and existing float32 AdamW moment tensors.
                # No rounding/straight-through surrogate replaces this graph.
                rows = rows.to(torch.float64)
                def linear(layer, values):
                    return nn.functional.linear(values, layer.weight.to(torch.float64),
                                                layer.bias.to(torch.float64))
                public = torch.tanh(linear(self.public_linear,
                    (rows[..., :166] - self.public_mean.to(torch.float64))
                    / self.public_scale.to(torch.float64)))
                tail = ((rows[..., 166:] - self.tail_mean.to(torch.float64))
                        / self.tail_scale.to(torch.float64))
                features = torch.tanh(linear(self.candidate_linear, torch.cat((public, tail), dim=-1)))
                scores = linear(self.actor_head, features).squeeze(-1)
                pooled = (features * enabled[..., None]).sum(dim=1) / enabled.sum(dim=1, keepdim=True)
                return scores, linear(self.value_head, pooled).squeeze(-1)
            public = torch.tanh(self.public_linear((rows[..., :166] - self.public_mean) / self.public_scale))
            tail = (rows[..., 166:] - self.tail_mean) / self.tail_scale
            features = torch.tanh(self.candidate_linear(torch.cat((public, tail), dim=-1)))
            scores = self.actor_head(features).squeeze(-1)
            pooled = (features * enabled[..., None]).sum(dim=1) / enabled.sum(dim=1, keepdim=True)
            return scores, self.value_head(pooled).squeeze(-1)

    return Graph()


def _batch(policy, model_sha, episodes):
    if not episodes:
        raise ValueError("fresh full episodes required")
    ids, rows, masks, positions, log_probs, values, returns, sample_weights, old_probabilities = [], [], [], [], [], [], [], [], []
    game_outcomes = {-1: 0, 0: 0, 1: 0}
    for episode in episodes:
        identity = episode["episode_id"]
        if type(identity) is not str or not identity or identity in ids:
            raise ValueError("unique episode identities required")
        ids.append(identity)
        if episode["behavior_model_sha256"] != model_sha:
            raise ValueError("stale or foreign behavior policy cannot enter on-policy PPO")
        if (type(episode.get("rollout_iteration")) is not int
                or episode["rollout_iteration"] != policy.manifest.get("rollout_iteration", 0) + 1):
            raise ValueError("next monotonic rollout iteration required")
        reward = episode["terminal_reward"]
        if type(reward) not in (int, float) or reward not in (-1, 0, 1):
            raise ValueError("physical terminal WDL reward required")
        if type(episode["seed"]) is not int or type(episode["seat"]) is not int or episode["seat"] not in (0, 1):
            raise ValueError("episode environment identity required")
        if len(episode["calls"]) != 30:
            raise ValueError("all 30 natural economic calls required")
        game_outcomes[int(reward)] += 1
        for call in episode["calls"]:
            x, enabled = validate_inputs(call["candidate_contexts"], call["mask"])
            if call["temperature"] != policy.manifest["temperature"]:
                raise ValueError("behavior temperature drift")
            selected = call["action_index"]
            if type(selected) is not int or selected not in ACTION_IDS or not enabled[ACTION_IDS.index(selected)]:
                raise ValueError("actual selected legal action required")
            pos = ACTION_IDS.index(selected)
            old_logp, old_value = call["old_log_probability"], call["old_value"]
            if not np.isfinite(old_logp) or not np.isfinite(old_value) or old_logp > 0:
                raise ValueError("finite behavior log probability and value required")
            _, replay_value, _, probabilities, replay_log_probs = policy.distribution(x, call["mask"])
            if (abs(old_logp - replay_log_probs[pos]) > 2e-5
                    or abs(old_value - replay_value) > 2e-5):
                raise ValueError("saved behavior probability/value does not replay under exact model")
            rows.append(x); masks.append(enabled); positions.append(pos)
            log_probs.append(old_logp); values.append(old_value); returns.append(reward)
            old_probabilities.append(probabilities)
            sample_weights.append(1 / (len(episodes) * len(episode["calls"])))
    return {"x": np.asarray(rows), "masks": np.asarray(masks),
        "positions": np.asarray(positions), "old_log_probs": np.asarray(log_probs),
        "old_values": np.asarray(values), "returns": np.asarray(returns),
        "weights": np.asarray(sample_weights), "old_probabilities": np.asarray(old_probabilities),
        "episode_ids": ids, "outcomes": {str(k): v for k, v in game_outcomes.items()}}


def train_ppo(model_path, episodes, output_model, optimizer_path=None, settings=None, *, output_optimizer=None):
    """Update on this exact model's fresh full trajectories, then save both states."""
    import torch
    from torch.distributions import Categorical

    settings = default_settings() if settings is None else dict(settings)
    validate_settings(settings)
    output_model = Path(output_model)
    output_optimizer = Path(output_optimizer) if output_optimizer is not None else output_model.with_suffix(".optimizer.pt")
    if output_model.exists() or output_optimizer.exists() or output_model.resolve() == output_optimizer.resolve():
        raise ValueError("new distinct model and optimizer outputs required")
    manifest, initial = load_model(model_path)
    numerics = numerical_contract(manifest)
    model_sha = file_sha(model_path)
    policy = ActorCriticPolicy(manifest, initial)
    batch = _batch(policy, model_sha, episodes)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(settings["seed"])
    graph = torch_graph(initial, numerics)
    optimizer = torch.optim.AdamW(graph.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    rng = np.random.default_rng(settings["seed"])
    global_step, prior_episode_ids = 0, []
    if optimizer_path is not None:
        checkpoint = torch.load(optimizer_path, map_location="cpu", weights_only=True)
        if (checkpoint.get("schema") != "farm-economic-ppo-optimizer-v2"
                or checkpoint.get("model_sha256") != model_sha or checkpoint.get("settings") != settings):
            raise ValueError("optimizer/model/settings identity mismatch")
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]
        rng.bit_generator.state = json.loads(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if checkpoint["last_rollout_iteration"] != manifest.get("rollout_iteration", 0):
            raise ValueError("optimizer/model rollout iteration mismatch")
        prior_episode_ids = checkpoint["recent_episode_ids"]
    if global_step != manifest["optimizer_steps"]:
        raise ValueError("optimizer state required for exact trained-checkpoint continuation")
    if set(prior_episode_ids) & set(batch["episode_ids"]):
        raise ValueError("previously consumed episodes cannot be relabeled fresh")
    if global_step >= settings["planned_optimizer_steps"]:
        raise ValueError("declared campaign optimizer budget exhausted")
    x = torch.tensor(batch["x"], dtype=torch.float32)
    enabled = torch.tensor(batch["masks"], dtype=torch.bool)
    actions = torch.tensor(batch["positions"], dtype=torch.long)
    statistics_dtype = torch.float64 if numerics == NUMERICS_CANONICAL else torch.float32
    old_logp = torch.tensor(batch["old_log_probs"], dtype=statistics_dtype)
    old_value = torch.tensor(batch["old_values"], dtype=statistics_dtype)
    returns = torch.tensor(batch["returns"], dtype=statistics_dtype)
    weights = torch.tensor(batch["weights"], dtype=statistics_dtype)
    old_probs = torch.tensor(batch["old_probabilities"], dtype=statistics_dtype)
    temperature = manifest["temperature"]

    def distribution(indices):
        scores, value = graph(x[indices], enabled[indices])
        logits = (scores / temperature).masked_fill(~enabled[indices], -torch.inf)
        if numerics == NUMERICS_CANONICAL:
            # Remove the common offset before normalization, matching NumPy.
            logits = logits - logits.max(dim=1, keepdim=True).values
        return Categorical(logits=logits), value

    all_indices = torch.arange(len(x))
    with torch.no_grad():
        dist, value = distribution(all_indices)
        max_logp_error = float((dist.log_prob(actions) - old_logp).abs().max())
        max_value_error = float((value - old_value).abs().max())
        if max_logp_error > 2e-5 or max_value_error > 2e-5:
            raise ValueError("NumPy/PyTorch behavior logprob/value parity failed before PPO")
    advantages = returns - old_value
    raw_advantage_std = float(advantages.std(unbiased=False))
    if settings["normalize_advantages"] and raw_advantage_std > 1e-8:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False)
    # Constant advantages are retained; they are honestly reported and do not
    # become a fabricated contrast or automatic claim of actor improvement.
    advantages = advantages.detach()
    history, start_step, stop_reason = [], global_step, "EPOCH_BUDGET_COMPLETE"

    def exact_kl(dist):
        safe_old = old_probs.clamp_min(1e-30).log()
        safe_new = dist.logits.masked_fill(~enabled, 0)
        return float((weights * (old_probs * (safe_old - safe_new)).sum(dim=1)).sum())

    for epoch in range(settings["epochs"]):
        order = rng.permutation(len(x))
        for offset in range(0, len(x), settings["minibatch_size"]):
            with torch.no_grad():
                whole_dist, _ = distribution(all_indices)
                kl_before = max(0.0, exact_kl(whole_dist))
            if kl_before > settings["target_kl"]:
                stop_reason = "TARGET_KL_EARLY_STOP"
                break
            if global_step >= settings["planned_optimizer_steps"]:
                stop_reason = "CAMPAIGN_OPTIMIZER_BUDGET_COMPLETE"
                break
            idx = torch.tensor(order[offset:offset + settings["minibatch_size"]], dtype=torch.long)
            dist, value = distribution(idx)
            ratios = torch.exp(dist.log_prob(actions[idx]) - old_logp[idx])
            surrogate = torch.minimum(ratios * advantages[idx],
                ratios.clamp(1-settings["clip_ratio"], 1+settings["clip_ratio"]) * advantages[idx])
            w = weights[idx] / weights[idx].sum()
            actor_loss = -(surrogate * w).sum()
            value_loss = ((value - returns[idx]).square() * w).sum()
            entropy = (dist.entropy() * w).sum()
            loss = actor_loss + settings["value_coefficient"] * value_loss - settings["entropy_coefficient"] * entropy
            if not torch.isfinite(loss):
                raise ValueError("nonfinite PPO loss; no child saved")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(graph.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            lr = learning_rate_at(global_step + 1, settings)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            global_step += 1
            if any(not torch.isfinite(param).all() for param in graph.parameters()):
                raise ValueError("nonfinite PPO weights; no child saved")
            history.append({"optimizer_step": global_step, "epoch": epoch + 1,
                "minibatch_rows": len(idx), "learning_rate": lr, "loss": float(loss.detach()),
                "actor_loss": float(actor_loss.detach()), "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()), "gradient_norm": float(grad_norm),
                "exact_kl_before_update": kl_before,
                "clip_fraction": float(((ratios - 1).abs() > settings["clip_ratio"]).float().mean())})
        if stop_reason != "EPOCH_BUDGET_COMPLETE":
            break
    if not history:
        raise ValueError("no PPO optimizer updates performed")
    trained = {key: value.detach().numpy().copy() for key, value in graph.state_dict().items()}
    if any(not np.array_equal(trained[k], initial[k]) for k in NORMALIZATION):
        raise ValueError("frozen normalization changed")
    changed_tensors = [k for k in TRAINABLE if not np.array_equal(trained[k], initial[k])]
    if not changed_tensors:
        raise ValueError("unchanged weights are not a completed fit")
    with torch.no_grad():
        final_dist, final_value = distribution(all_indices)
        final_kl = max(0.0, exact_kl(final_dist))
        final_entropy = float((final_dist.entropy() * weights).sum())
        value_mse = float(((final_value - returns).square() * weights).sum())
        changed_choices = int((old_probs.argmax(dim=1) != final_dist.probs.argmax(dim=1)).sum())
    if final_kl > settings["target_kl"] and stop_reason == "EPOCH_BUDGET_COMPLETE":
        stop_reason = "TARGET_KL_EXCEEDED_AT_FINAL_UPDATE"
    # Exercise the exported NumPy graph before saving a child. The next rollout
    # must use the distribution PPO just optimized, including legal masks.
    exported = ActorCriticPolicy(manifest, trained)
    final_logp_error, final_value_error, final_choice_mismatches = 0.0, 0.0, 0
    torch_final_logits = final_dist.logits.detach().numpy()
    torch_final_values = final_value.detach().numpy()
    for index, (row_pack, legal) in enumerate(zip(batch["x"], batch["masks"])):
        mask = np.zeros(29, dtype=bool)
        mask[list(ACTION_IDS)] = legal
        _, value, _, probs, log_probs = exported.distribution(row_pack, mask)
        final_logp_error = max(final_logp_error,
            float(np.max(np.abs(log_probs[legal] - torch_final_logits[index, legal]))))
        final_value_error = max(final_value_error, abs(value - float(torch_final_values[index])))
        final_choice_mismatches += int(probs.argmax() != torch_final_logits[index].argmax())
    if final_logp_error > 2e-5 or final_value_error > 2e-5 or final_choice_mismatches:
        raise ValueError("trained NumPy/PyTorch policy distribution or greedy-choice parity failed; no child saved")
    receipt = bound({"schema": "farm-economic-ppo-fit-v2", "settings": settings,
        "behavior_model_sha256": model_sha, "parent_tensor_sha256": tensor_sha(initial),
        "child_tensor_sha256": tensor_sha(trained), "optimizer_steps": len(history),
        "global_optimizer_steps_before": start_step, "global_optimizer_steps_after": global_step,
        "rollout_iteration": manifest.get("rollout_iteration", 0) + 1,
        "episodes": len(episodes), "natural_calls": len(x),
        "independent_environment_seeds": len({episode["seed"] for episode in episodes}),
        "terminal_outcomes": batch["outcomes"], "episode_ids": batch["episode_ids"],
        "training_batch_sha256": bound({"episodes": episodes})["content_sha256"],
        "trainable_parameters": sum(trained[k].size for k in TRAINABLE),
        "changed_tensors": changed_tensors, "raw_advantage_std": raw_advantage_std,
        "old_behavior_logp_torch_numpy_max_error": max_logp_error,
        "old_behavior_value_torch_numpy_max_error": max_value_error,
        "trained_logp_torch_numpy_max_error": final_logp_error,
        "trained_value_torch_numpy_max_error": final_value_error,
        "trained_greedy_choice_mismatches": final_choice_mismatches,
        "final_exact_kl": final_kl, "final_entropy": final_entropy, "final_value_mse": value_mse,
        "kl_target_exceeded": final_kl > settings["target_kl"],
        "maximum_measured_rollout_kl": max(final_kl, *(entry["exact_kl_before_update"] for entry in history)),
        "kl_stop_is_not_a_hard_post_update_bound": True,
        "changed_greedy_choices_on_rollout": changed_choices, "stop_reason": stop_reason,
        "history": history, "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "loss": "PPO_CLIPPED_SELECTED_ACTION_ADVANTAGE_PLUS_MC_CRITIC_MINUS_ENTROPY",
        "data_scope": "FRESH_BEHAVIOR_FULL_EPISODES_CALLER_ENGINE_RECONSTRUCTED",
        "rollout_outcomes_are_not_policy_evaluation": True, "promotion_authority": False})
    save_model(output_model, trained, {**manifest,
        "stage": "TRAINED_ON_POLICY_PPO_ACTOR_CRITIC", "optimizer_steps": global_step,
        "rollout_iteration": manifest.get("rollout_iteration", 0) + 1,
        "parent_model_sha256": model_sha, "training_receipt": receipt})
    output_optimizer.parent.mkdir(parents=True, exist_ok=True)
    with output_optimizer.open("xb") as stream:
        torch.save({"schema": "farm-economic-ppo-optimizer-v2", "model_sha256": file_sha(output_model),
            "global_step": global_step, "settings": settings, "optimizer": optimizer.state_dict(),
            "numpy_rng_state": canonical(rng.bit_generator.state), "torch_rng_state": torch.get_rng_state(),
            # Older receipts already carry an obsolete model hash and rollout
            # ordinal. Keep bounded recent IDs instead of serializing a growing
            # multi-million-episode history at every update. The runner retains
            # its append-only game/exposure ledger and supplies fresh witnesses.
            "last_rollout_iteration": manifest.get("rollout_iteration", 0) + 1,
            "recent_episode_ids": batch["episode_ids"]}, stream)
    return bound({**receipt, "output_model_sha256": file_sha(output_model),
        "output_optimizer_sha256": file_sha(output_optimizer)})
