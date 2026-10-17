"""Opt-in indexed PPO numerical core; owned by the full-economic maintainer.

Purpose: consume many original day cohorts without retaining their actor inputs
or rescanning the entire rollout after every microbatch. This is NOT a collector,
physical/neural admission authority, checkpoint writer, or production CLI. The
owner must validate the actual parent/Adam/RNG and execute fresh reconstruction
and sampling verification before calling this core. IndexedReader does not mint
that admission. Never accept imported/rehashed witness booleans as authentication.

The maintained replay, terminal MC objective, settings and LR schedule are reused.
This NEW execution accumulates sample-weighted microbatch gradients, reduces full
metrics on CPU, and checks sampled-program K3 only at registered interim subsets
and full invocation boundaries. Neither an interim subset nor final K3 is exact
KL/a hard post-step bound. No entropy, reward, action, or tolerance changes.

An owner may persist the returned cursor WITH the exact updated model, optimizer
and RNG, externally pin its content hash, then finish this ONE shuffled epoch.
That is continuation of already admitted data, not fresh-game credit. A cursor
cannot extend the registered allocation/schedule or reopen a KL-stopped epoch.
All exceptions restore invocation-entry tensors, Adam, gradients and RNG. No
files are written. A source-versioned production binding is still required.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

from farm_neat.training import batched_training as base

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "farm-full-economic-indexed-ppo-v1"
EXECUTION = {
    "objective": "UNCHANGED_TERMINAL_MC_FULL_PROGRAM_PPO",
    "advantage_scope": "GLOBAL_COMPLETE_ROLLOUT",
    "gradient_weighting": "CALL_COUNT_OVER_ACTUAL_ACCUMULATION_GROUP",
    "shuffle": "ONE_CPU_TORCH_RANDPERM_SAVED_CURSOR_NO_REPLACEMENT",
    "initial": "FULL_NUMERICAL_LOGP_VALUE_GATE_2E-5",
    "metrics": "CPU_REDUCTION_FULL_INITIAL_FINAL_REGISTERED_INTERIM_SUBSET",
    "replay": "MAINTAINED_ORIGINAL_B32_REPLAY_NO_NUMERICAL_BACKEND_SUBSTITUTION",
    "training_admission": False,
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _bound(value):
    result = {k: v for k, v in value.items() if k != "content_sha256"}
    return {**result, "content_sha256": _digest(result)}


def _sealed(value):
    if not isinstance(value, dict) or value != _bound(value):
        raise ValueError("exact content-bound object required")


def identity():
    """Pins reused implementations/tests without model loading or CUDA execution."""
    # Changing this pin list changes the source identity of every future checkpoint; intended.
    paths = ("src/farm_neat/training/streaming_ppo.py",
             "tests/contracts/test_streaming_ppo.py",
             "src/farm_neat/rollout/buffer.py",
             "tests/contracts/test_buffer.py")
    return _bound({"schema": SCHEMA, "execution": EXECUTION,
        "source_pins": {**base.sources(), **{p: base.file_sha(ROOT/p) for p in paths}}})


def make_protocol(*, accumulation_cohorts, optimizer_updates, metric_every_updates,
                  metric_cohorts):
    """Register BEFORE outcomes; update allocation is total, not per invocation."""
    for value in (accumulation_cohorts, optimizer_updates, metric_every_updates):
        if type(value) is not int or value < 1:
            raise ValueError("positive integer accumulation/allocation/cadence required")
    if (not isinstance(metric_cohorts, (list, tuple)) or not metric_cohorts
            or any(type(i) is not int or i < 0 for i in metric_cohorts)
            or len(set(metric_cohorts)) != len(metric_cohorts)):
        raise ValueError("explicit unique interim cohort ordinals required")
    return _bound({"schema": SCHEMA+"-protocol", "accumulation_cohorts": accumulation_cohorts,
        "optimizer_updates": optimizer_updates, "metric_every_updates": metric_every_updates,
        "metric_cohorts": list(metric_cohorts), "epochs": 1})


def state_digest(value):
    """Exact tensor/storage hash without torch.save, casts, or optimizer mutation."""
    import torch
    h = hashlib.sha256()
    def feed(item):
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided:
                raise ValueError("dense optimizer/model tensors required")
            h.update(f"tensor:{item.dtype}:{tuple(item.shape)}:".encode())
            raw = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            h.update(memoryview(raw))
        elif isinstance(item, dict):
            h.update(b"dict[")
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                feed(key); feed(item[key])
            h.update(b"]")
        elif isinstance(item, (tuple, list)):
            h.update((type(item).__name__+"[").encode())
            for child in item: feed(child)
            h.update(b"]")
        elif item is None or type(item) in (str, int, float, bool):
            h.update((type(item).__name__+":"+json.dumps(item, allow_nan=False)+";").encode())
        else:
            raise ValueError("unsupported persistent optimizer state value")
    feed(value)
    return h.hexdigest()


def _rng(device):
    import torch
    return {"cpu": torch.get_rng_state().clone(), "cuda":
        [s.clone() for s in torch.cuda.get_rng_state_all()] if device.type == "cuda" else []}


def _restore_rng(state, device):
    import torch
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda": torch.cuda.set_rng_state_all(state["cuda"])


def _state(graph, optimizer, device):
    return {"model": state_digest(graph.state_dict()),
            "optimizer": state_digest(optimizer.state_dict()), "rng": state_digest(_rng(device))}


def _vectors(buffer):
    import torch
    data = buffer.vectors()
    if data.get("identity") != buffer.identity or data.get("groups") != buffer.group_count:
        raise ValueError("exact indexed buffer vector identity required")
    sizes = data.get("group_sizes", [data.get("lanes")]*buffer.group_count)
    if (len(sizes) != buffer.group_count or not sizes
            or any(type(n) is not int or n < 1 for n in sizes)):
        raise ValueError("positive exact cohort sizes required")
    count = sum(sizes)
    ids = data.get("sample_ids", ())
    if (count != buffer.sample_count or len(ids) != count
            or any(not isinstance(x, (tuple, list)) or len(x) != 2
                   or not isinstance(x[0], str) or not x[0] or type(x[1]) is not int for x in ids)
            or len(set(map(tuple, ids))) != count):
        raise ValueError("unique complete episode/step sample identities required")
    vectors = {}
    for key in ("old_logp", "old_value", "returns"):
        vectors[key] = torch.tensor(data[key], dtype=torch.float64, device="cpu")
        if vectors[key].shape != (count,) or not bool(torch.isfinite(vectors[key]).all()):
            raise ValueError("finite compact complete rollout vectors required")
    if bool((vectors["old_logp"] > 0).any()) or not bool(
            ((vectors["returns"] == -1) | (vectors["returns"] == 0) | (vectors["returns"] == 1)).all()):
        raise ValueError("nonpositive behavior logp and terminal WDL required")
    offsets = [0]
    for n in sizes: offsets.append(offsets[-1]+n)
    vector_sha = state_digest({**vectors, "ids": list(map(tuple, ids)), "sizes": sizes})
    return vectors, offsets, vector_sha


def _load(buffer, ordinal, vectors, offsets):
    import torch
    group = buffer.group(ordinal)
    begin, end = offsets[ordinal:ordinal+2]
    calls = group["calls"]
    if len(calls) != end-begin or len(group["returns"]) != end-begin:
        raise ValueError("indexed group size drift")
    actual = {"old_logp": [c["old_log_probability"] for c in calls],
              "old_value": [c["old_value"] for c in calls], "returns": group["returns"]}
    for name, values in actual.items():
        if not torch.equal(torch.tensor(values, dtype=torch.float64), vectors[name][begin:end]):
            raise ValueError("indexed group targets differ from global vector")
    return group


def _outputs(graph, buffer, ordinal, vectors, offsets):
    import torch
    group = _load(buffer, ordinal, vectors, offsets)
    values = base.replay_groups(graph, [group])
    size = offsets[ordinal+1]-offsets[ordinal]
    if (len(values) != 3 or any(v.shape != (size,) or not bool(torch.isfinite(v).all()) for v in values)):
        raise ValueError("finite exact per-call replay outputs required")
    return values


def _measure(graph, buffer, ordinals, vectors, offsets, *, parity=False):
    import torch
    tick = time.monotonic()
    outputs, old, returns = [[], [], []], [], []
    error_logp = error_value = 0.0
    with torch.no_grad():
        for ordinal in ordinals:
            values = _outputs(graph, buffer, ordinal, vectors, offsets)
            selected = slice(offsets[ordinal], offsets[ordinal+1])
            for j, value in enumerate(values): outputs[j].append(value.detach().cpu())
            old.append(vectors["old_logp"][selected]); returns.append(vectors["returns"][selected])
            if parity:
                error_logp = max(error_logp, float((outputs[0][-1]-old[-1]).abs().max()))
                error_value = max(error_value, float((outputs[1][-1]-vectors["old_value"][selected]).abs().max()))
    if parity and max(error_logp, error_value) > base.rollout.ATOL:
        raise ValueError("unchanged2e-5 FULL initial behavior parity failed before PPO")
    result = base.legacy.metrics(*(torch.cat(x) for x in outputs), torch.cat(old), torch.cat(returns))
    return {**result, "scope": "FULL_ROLLOUT" if len(ordinals) == buffer.group_count else "REGISTERED_COHORT_SUBSET",
        "cohort_ordinals": list(ordinals), "calls": sum(len(x) for x in old),
        "seconds": time.monotonic()-tick, "old_logp_max_error": error_logp if parity else None,
        "old_value_max_error": error_value if parity else None}


def optimize(graph, optimizer, buffer, settings, global_step, *, protocol,
             expected_identity, max_updates, cursor=None, expected_cursor_sha256=None):
    """Run bounded numerical updates; caller owns admission and atomic publication.

    buffer follows full_economic_rollout_buffer.IndexedReader. Original full B32
    replay is used. max_updates limits this invocation; protocol.optimizer_updates
    fixes the lifetime allocation of the one-epoch cursor. No automatic schedule
    extension or reshuffle. A changed numerical/backend binding needs a separately
    versioned owner adapter, not an injected unlabelled replay callback.
    """
    import torch
    if expected_identity != identity(): raise ValueError("streaming execution/source identity drift")
    base.validate_settings(settings); _sealed(protocol)
    wanted = make_protocol(**{k: protocol[k] for k in
        ("accumulation_cohorts", "optimizer_updates", "metric_every_updates", "metric_cohorts")})
    if protocol != wanted: raise ValueError("registered streaming protocol required")
    if type(max_updates) is not int or max_updates < 1:
        raise ValueError("positive bounded invocation allocation required")
    if type(global_step) is not int or not 0 <= global_step < settings["planned_optimizer_steps"]:
        raise ValueError("remaining inherited global schedule required; no extension")
    if any(i >= buffer.group_count for i in protocol["metric_cohorts"]):
        raise ValueError("interim metrics outside registered buffer")
    behavior_age = buffer.identity.get("behavior", {}).get("model_optimizer_age")
    if (type(behavior_age) is not int or behavior_age < 0
            or (cursor is None and behavior_age != global_step)):
        raise ValueError("registered behavior age required; changed child needs its exact cursor")
    parameters = list(graph.parameters())
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    if [id(p) for p in owned] != [id(p) for p in parameters]:
        raise ValueError("exact ordered graph parameters in existing optimizer required")
    if global_step and any(group["lr"] != base.learning_rate_at(global_step, settings)
                           for group in optimizer.param_groups):
        raise ValueError("saved global-age learning rate required; no reset")
    if graph.training: raise ValueError("maintained eval-mode policy required")
    device = parameters[0].device
    if any(p.device != device for p in parameters): raise ValueError("one graph device required")
    before = {"model": base.legacy._cpu_copy(graph.state_dict()),
              "adam": base.legacy._cpu_copy(optimizer.state_dict()), "rng": _rng(device),
              "grads": [None if p.grad is None else p.grad.detach().clone() for p in parameters]}
    start, history = global_step, []
    try:
        if buffer.verify() != buffer.identity: raise ValueError("indexed source closure drift")
        vectors, offsets, vector_sha = _vectors(buffer)
        binding = {"execution_sha256": expected_identity["content_sha256"],
            "protocol_sha256": protocol["content_sha256"], "settings_sha256": _digest(settings),
            "buffer_sha256": _digest(buffer.identity), "vectors_sha256": vector_sha}
        if cursor is None:
            if expected_cursor_sha256 is not None: raise ValueError("missing externally pinned cursor")
            first = _measure(graph, buffer, range(buffer.group_count), vectors, offsets, parity=True)
            order = torch.randperm(buffer.group_count, device="cpu").tolist()
            position, origin, committed = 0, global_step, 0
            initial_admission = {"scope": "NUMERICAL_ONLY_NOT_PHYSICAL_OR_SAMPLER_ADMISSION",
                "model_sha256": state_digest(before["model"]), "metrics": first}
        else:
            _sealed(cursor)
            if expected_cursor_sha256 != cursor["content_sha256"]:
                raise ValueError("external exact cursor hash required for continuation")
            if (cursor.get("schema") != SCHEMA+"-cursor" or cursor.get("binding") != binding
                    or cursor.get("closed") is not False or cursor.get("global_step") != global_step
                    or cursor.get("state") != _state(graph, optimizer, device)):
                raise ValueError("exact open cursor model/Adam/RNG/age/source required")
            order, position = cursor["order"], cursor["position"]
            origin, committed = cursor["origin_global_step"], cursor["committed_updates"]
            if (not isinstance(order, list) or any(type(i) is not int for i in order)
                    or sorted(order) != list(range(buffer.group_count))
                    or type(position) is not int or not 0 <= position < len(order)
                    or type(origin) is not int or type(committed) is not int
                    or committed < 1 or origin != behavior_age or origin+committed != global_step
                    or position != min(committed*protocol["accumulation_cohorts"], len(order))
                    or committed >= protocol["optimizer_updates"]):
                raise ValueError("one-epoch deterministic coverage/age cursor required")
            initial_admission = cursor["initial_numerical_admission"]
            # Changed child is NOT falsely rechecked against behavior-parent logps.
            first = _measure(graph, buffer, range(buffer.group_count), vectors, offsets)
        if first["sampled_program_kl_k3"] > settings["target_kl"]:
            raise ValueError("full invocation-boundary sampled KL already exceeds target; no update")
        advantages = vectors["returns"]-vectors["old_value"]
        raw_std, raw_mean = float(advantages.std(unbiased=False)), float(advantages.mean())
        if settings["normalize_advantages"] and raw_std > 1e-8:
            advantages = (advantages-raw_mean)/raw_std
        stop = "INVOCATION_UPDATE_LIMIT"
        while position < len(order):
            if committed >= protocol["optimizer_updates"]: stop = "REGISTERED_ALLOCATION_COMPLETE"; break
            if global_step >= settings["planned_optimizer_steps"]: stop = "INHERITED_SCHEDULE_COMPLETE"; break
            if len(history) >= max_updates: break
            chosen = order[position:position+protocol["accumulation_cohorts"]]
            count = sum(offsets[i+1]-offsets[i] for i in chosen)
            tick, totals = time.monotonic(), [0.0]*5
            optimizer.zero_grad(set_to_none=True)
            for ordinal in chosen:
                selected = slice(offsets[ordinal], offsets[ordinal+1])
                weight = (selected.stop-selected.start)/count
                logp, value, entropy = _outputs(graph, buffer, ordinal, vectors, offsets)
                terms = base.legacy.ppo_terms(logp, value, entropy,
                    vectors["old_logp"][selected].to(device), vectors["returns"][selected].to(device),
                    advantages[selected].to(device), settings)
                if not bool(torch.isfinite(terms[0])): raise ValueError("nonfinite PPO loss")
                (terms[0]*weight).backward()
                totals = [a+float(b.detach())*weight for a,b in zip(totals, terms)]
                del logp, value, entropy, terms
            norm = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip"], error_if_nonfinite=True)
            lr = base.learning_rate_at(global_step+1, settings)
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.step(); global_step += 1; committed += 1; position += len(chosen)
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                raise ValueError("nonfinite updated weights; invocation restored")
            row = {"optimizer_step": global_step, "cohort_ordinals": chosen,
                "microbatches": len(chosen), "effective_batch_calls": count, "learning_rate": lr,
                "loss": totals[0], "actor_loss": totals[1], "value_loss": totals[2],
                "conditional_path_entropy_before": totals[3], "clip_fraction": totals[4],
                "gradient_norm": float(norm), "update_seconds": time.monotonic()-tick,
                "post_update": None}
            history.append(row)
            if committed % protocol["metric_every_updates"] == 0:
                row["post_update"] = _measure(graph, buffer, protocol["metric_cohorts"], vectors, offsets)
                if row["post_update"]["sampled_program_kl_k3"] > settings["target_kl"]:
                    stop = "REGISTERED_SUBSET_SAMPLED_KL_STOP_NOT_FULL_BOUND"; break
        if not history: raise ValueError("no optimizer update; no completed numerical fit")
        final = _measure(graph, buffer, range(buffer.group_count), vectors, offsets)
        if final["sampled_program_kl_k3"] > settings["target_kl"]:
            stop = "FULL_FINAL_SAMPLED_KL_EXCEEDED_AFTER_UPDATE_NOT_HARD_BOUND"
        elif position == len(order): stop = "ONE_EPOCH_COMPLETE"
        elif committed >= protocol["optimizer_updates"]: stop = "REGISTERED_ALLOCATION_COMPLETE"
        elif global_step >= settings["planned_optimizer_steps"]: stop = "INHERITED_SCHEDULE_COMPLETE"
        changed = [k for k,v in graph.state_dict().items() if not torch.equal(v.detach().cpu(), before["model"][k])]
        if not changed: raise ValueError("unchanged tensors are not a completed fit")
        if buffer.verify() != buffer.identity or identity() != expected_identity:
            raise ValueError("final indexed/source identity drift; invocation restored")
        next_cursor = _bound({"schema": SCHEMA+"-cursor", "binding": binding,
            "origin_global_step": origin, "global_step": global_step, "committed_updates": committed,
            "order": order, "position": position, "closed": stop != "INVOCATION_UPDATE_LIMIT",
            "initial_numerical_admission": initial_admission, "state": _state(graph, optimizer, device)})
        return {"schema": SCHEMA, "execution": copy.deepcopy(EXECUTION), "protocol": protocol,
            "optimizer_steps": len(history), "global_optimizer_steps_before": start,
            "global_optimizer_steps_after": global_step, "history": history,
            "initial": first, "final": final, "raw_advantage_std": raw_std, "raw_advantage_mean": raw_mean,
            "changed_tensors": changed, "stop_reason": stop, "cursor": next_cursor,
            "coverage": {"total_cohorts": buffer.group_count, "total_calls": buffer.sample_count,
                "unique_cohorts_used": position, "unique_calls_used": sum(offsets[i+1]-offsets[i] for i in order[:position]),
                "call_fraction": sum(offsets[i+1]-offsets[i] for i in order[:position])/buffer.sample_count},
            "numeric_atol": base.rollout.ATOL, "kl_is_sampled_not_exact": True,
            "post_update_kl_is_not_a_hard_bound": True, "training_admission": False,
            "fresh_games_by_optimizer": 0, "all_draw_filtering": False}
    except BaseException:
        graph.load_state_dict(before["model"], strict=True); optimizer.load_state_dict(before["adam"])
        for parameter, grad in zip(parameters, before["grads"]): parameter.grad = grad
        _restore_rng(before["rng"], device)
        raise
