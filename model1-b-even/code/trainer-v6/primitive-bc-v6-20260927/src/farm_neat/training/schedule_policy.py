"""Unbounded learned candidate preferences and a state baseline, NumPy only."""
from __future__ import annotations

import numpy as np

from farm_neat.training.schedule_model import (ACTION_IDS, NUMERICS_CANONICAL, bound,
    load_model, numerical_contract)


def validate_inputs(rows, selection_mask):
    rows, mask = np.asarray(rows, dtype=np.float32), np.asarray(selection_mask)
    if (rows.shape != (7, 302) or not np.isfinite(rows).all()
            or mask.shape != (29,) or mask.dtype != np.bool_ or not mask[0]
            or any(mask[i] for i in range(29) if i not in ACTION_IDS)):
        raise ValueError("finite ordered seven-candidate pack and exact legal mask required")
    for row, index in zip(rows, ACTION_IDS):
        if (not np.array_equal(row[229:258], np.eye(29, dtype=np.float32)[index])
                or row[258] != 1 or row[259] != float(mask[index])):
            raise ValueError("candidate identity or mask encoding drift")
    if not np.array_equal(rows[:, :229], np.broadcast_to(rows[0, :229], (7, 229))):
        raise ValueError("candidate pack must share a single actor-visible planning state")
    return rows, mask[list(ACTION_IDS)]


def hidden(weights, rows):
    public = np.tanh(((rows[..., :166] - weights["public_mean"]) / weights["public_scale"])
        @ weights["public_linear.weight"].T + weights["public_linear.bias"])
    tail = (rows[..., 166:] - weights["tail_mean"]) / weights["tail_scale"]
    return np.tanh(np.concatenate((public, tail), axis=-1)
        @ weights["candidate_linear.weight"].T + weights["candidate_linear.bias"])


class ActorCriticPolicy:
    def __init__(self, manifest, weights):
        self.manifest, self.weights = manifest, weights
        self.numerics = numerical_contract(manifest)
        # Checkpoint/optimizer tensors stay float32. The versioned forward graph
        # promotes BEFORE normalization/reductions, not merely at softmax. Old
        # models keep their exact original arithmetic and selection receipts.
        self._forward_weights = ({key: value.astype(np.float64) for key, value in weights.items()}
                                 if self.numerics == NUMERICS_CANONICAL else weights)
        self.last_diagnostics = None

    @classmethod
    def load(cls, path):
        return cls(*load_model(path))

    def evaluate(self, rows, mask):
        rows, enabled = validate_inputs(rows, mask)
        if self.numerics == NUMERICS_CANONICAL:
            rows = rows.astype(np.float64)
        weights = self._forward_weights
        features = hidden(weights, rows)
        scores = (features @ weights["actor_head.weight"].T + weights["actor_head.bias"])[:, 0]
        # Averaging the legal candidate set makes this a state baseline. It never
        # conditions on the subsequently sampled action or terminal outcome.
        pooled = features[enabled].mean(axis=0)
        value = float((pooled @ weights["value_head.weight"].T + weights["value_head.bias"])[0])
        if not np.isfinite(scores).all() or not np.isfinite(value):
            raise ValueError("nonfinite actor-critic output")
        return scores.astype(np.float64), value, enabled

    def distribution(self, rows, mask):
        scores, value, enabled = self.evaluate(rows, mask)
        logits = scores / self.manifest["temperature"]
        shifted = logits - logits[enabled].max()
        normalizer = np.log(np.exp(shifted[enabled]).sum())
        log_probs = np.where(enabled, shifted - normalizer, -np.inf)
        probs = np.where(enabled, np.exp(log_probs), 0.0)
        return scores, value, enabled, probs, log_probs

    def select(self, rows, selection_mask, *, rng=None, temperature=1.0):
        # Historical runtime passes temperature=1.0. The v2 checkpoint owns the
        # declared distribution temperature so wrappers cannot change behavior.
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("positive finite runtime temperature required")
        self.last_diagnostics = None
        scores, value, enabled, probabilities, log_probs = self.distribution(rows, selection_mask)
        if rng is not None and (not isinstance(rng, np.random.Generator)
                or type(rng.bit_generator) is not np.random.PCG64):
            raise ValueError("replayable PCG64 sampling required")
        position = (int(np.argmax(np.where(enabled, scores, -np.inf))) if rng is None
                    else int(rng.choice(len(ACTION_IDS), p=probabilities)))
        behavior = probabilities if rng is not None else np.eye(len(ACTION_IDS))[position]
        entropy = -float(np.sum(probabilities[enabled] * log_probs[enabled]))
        self.last_diagnostics = bound({
            "schema": "farm-economic-actor-critic-selection-v2",
            "model_content_sha256": self.manifest["content_sha256"],
            "model_stage": self.manifest["stage"], "action_index": ACTION_IDS[position],
            "selection_mode": "SAMPLED" if rng is not None else "GREEDY",
            "selection_probabilities": behavior.tolist(),
            "sampling_probabilities": probabilities.tolist(),
            "behavior_log_probability": float(log_probs[position]) if rng is not None else 0.0,
            "state_value": value, "actor_scores": scores.tolist(),
            "temperature": self.manifest["temperature"], "entropy": entropy,
            "normalized_entropy": entropy / np.log(enabled.sum()) if enabled.sum() > 1 else 0.0,
            "no_edit_score": float(scores[0]), "network_values_calibrated": False,
            "score_rule": "TRAINABLE_SCALAR_PREFERENCE",
            "sampling_rule": "MASKED_CATEGORICAL_LOGITS_FIXED_MODEL_TEMPERATURE",
        })
        return self.last_diagnostics
