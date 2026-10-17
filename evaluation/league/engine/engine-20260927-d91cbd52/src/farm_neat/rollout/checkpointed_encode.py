"""Opt-in per-block recomputation for the existing full-economic encoder.

Owner: learning/throughput maintainer. Status: isolated memory experiment.
The exact original Graph.encode performs input checks, knownness/masking,
pooling, critic and nested routing auxiliaries. A temporary non-Module view
wraps only its existing blocks with non-reentrant activation checkpointing.
No parameter keys/values, training mode, optimizer, saved traces or old sources
change. This grants no training admission or GPU memory/speed guarantee; an
owner benchmark must measure the recomputation cost and actual allocation.
identity() is standard-library-only for freezing a plan without Torch/GPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "farm-full-economic-checkpointed-encode-v1"
# Re-pinned to the migrated encoder bytes, as batched_replay.PREIMAGES already was; the farm
# values named bytes that no longer exist in either workspace, so identity() could never pass.
PREIMAGES = {
    "src/farm_neat/policy/network.py": "ec805f612e6deb6451701a6bfd6e18b0459dad94b535a6f132c9cb3c9e1d6a91",
    "src/farm_neat/policy/qwen_hybrid.py": "cc85d803c1c5a9ea4a569111a615ff43d260feb15a82dcfd1f89068150c5403d",
}


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity():
    """Freeze this plus the caller's full model/encode/runtime source closure."""
    for name, expected in PREIMAGES.items():
        if _sha(ROOT/name) != expected:
            raise ValueError("unsupported original encoder source: " + name)
    # Changing this pin list changes the source identity of every future checkpoint; intended.
    names = ("src/farm_neat/rollout/checkpointed_encode.py", "tests/unit/test_checkpointed_encode.py")
    payload = {"schema": SCHEMA, "source_pins": {**PREIMAGES, **{n: _sha(ROOT/n) for n in names}},
        "encoder": "ORIGINAL_GRAPH_ENCODE_WITH_TEMPORARY_CALLABLE_BLOCK_VIEWS",
        "checkpoint": {"per_block": True, "use_reentrant": False, "preserve_rng_state": True},
        "no_grad": "DIRECT_ORIGINAL_BLOCK_CALLS",
        "parameter_or_optimizer_change": False, "training_admission": False,
        "cpu_test_scope": "ORIGINAL_OUTPUT_AUX_INPUT_AND_PARAMETER_GRADIENT_COMPARISON",
        "gpu_memory_or_speed_guarantee": False}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {**payload, "content_sha256": digest}


class _BlockView:
    def __init__(self, block, checkpoint):
        self._block, self._checkpoint = block, checkpoint

    def __call__(self, residual, token_mask):
        return self._checkpoint(self._block, residual, token_mask,
                                use_reentrant=False, preserve_rng_state=True)


class _EncoderView:
    def __init__(self, graph, checkpoint):
        self._graph = graph
        self.blocks = tuple(_BlockView(block, checkpoint) for block in graph.blocks)

    def __getattr__(self, name):
        return getattr(self._graph, name)


def encode(graph, rows, token_mask, knownness, *, expected_identity, diagnostics=None):
    """Same three outputs as Graph.encode; use backward normally afterward.

    Expected identity is identity()'s complete mapping. No modules are installed
    on graph and no flags are changed. Non-reentrant checkpointing supports the
    original nested tensor/dictionary auxiliaries and torch.autograd.grad.
    Without autograd enabled, delegate directly with no checkpoint overhead.
    """
    if expected_identity != identity():
        raise ValueError("externally frozen checkpointed-encoder identity required")
    import torch
    from torch.utils.checkpoint import checkpoint
    from farm_neat.policy import network as policy
    if type(graph) is not policy.Graph or "encode" in graph.__dict__:
        raise ValueError("unmodified original Graph encoder required")
    if diagnostics is not None and type(diagnostics) is not dict:
        raise ValueError("optional diagnostics must be a dict")
    active = torch.is_grad_enabled()
    view = _EncoderView(graph, checkpoint) if active else graph
    result = policy.Graph.encode(view, rows, token_mask, knownness)
    if diagnostics is not None:
        diagnostics.update({"schema": SCHEMA, "execution_sha256": expected_identity["content_sha256"],
            "checkpointing_active": active, "checkpointed_blocks": len(graph.blocks) if active else 0,
            "training_admission": False})
    return result
