#!/usr/bin/env python3
"""Attention-head regrouping primitives for B-even.

Requires PyTorch.  The key safety property is that reordering old heads is
function preserving when Q/K/V rows and output-projection columns are moved
together.  Changing the number of heads is *not* function preserving; group
similar old QK score operators first, then measure candidate agreement.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch


def _attention_kind(module) -> str:
    if hasattr(module, "qkv") and hasattr(module, "out"):
        return "self"
    if hasattr(module, "q") and hasattr(module, "kv"):
        return "cross"
    raise TypeError(f"unsupported attention module: {type(module).__name__}")


def _validate_shape(module) -> Tuple[str, int, int]:
    kind = _attention_kind(module)
    heads = int(module.h)
    dh = int(module.dh)
    if heads <= 0 or dh <= 0:
        raise ValueError("attention h and dh must be positive")
    inner = heads * dh
    if kind == "self":
        if module.qkv.out_features != 3 * inner:
            raise ValueError("qkv output width does not equal 3*h*dh")
        if module.out.in_features != inner:
            raise ValueError("output input width does not equal h*dh")
    else:
        if module.q.out_features != inner or module.kv.out_features != 2 * inner:
            raise ValueError("cross-attention projections do not match h*dh")
        if getattr(module, "out", None) is not None and module.out.in_features != inner:
            raise ValueError("cross-attention output projection does not match h*dh")
    return kind, heads, dh


def qk_operator_vectors(module) -> torch.Tensor:
    """Return one flattened, scaled QK bilinear operator per current head.

    For a head, attention logits before softmax are x_q @ (Wq.T @ Wk) @ x_k.T
    divided by sqrt(dh). Cosine similarity between these operators is a
    calibration-free approximation to attention-map similarity.
    """
    kind, heads, dh = _validate_shape(module)
    with torch.no_grad():
        if kind == "self":
            d = heads * dh
            w = module.qkv.weight.detach().float()
            q = w[:d].reshape(heads, dh, w.shape[1])
            k = w[d:2 * d].reshape(heads, dh, w.shape[1])
        else:
            q_w = module.q.weight.detach().float()
            kv_w = module.kv.weight.detach().float()
            q = q_w.reshape(heads, dh, q_w.shape[1])
            k = kv_w[:heads * dh].reshape(heads, dh, kv_w.shape[1])
        op = torch.einsum("hdi,hdj->hij", q, k) / math.sqrt(float(dh))
        return op.flatten(1)


def qk_similarity(module, eps: float = 1e-12) -> torch.Tensor:
    """Cosine-similarity matrix of per-head QK score operators."""
    vec = qk_operator_vectors(module)
    norm = vec.norm(dim=1, keepdim=True).clamp_min(eps)
    unit = vec / norm
    sim = unit @ unit.T
    sim.fill_diagonal_(1.0)
    return sim


def _group_score(sim: torch.Tensor, group: Sequence[int]) -> float:
    if len(group) < 2:
        return 0.0
    values = [float(sim[a, b]) for i, a in enumerate(group) for b in group[i + 1:]]
    return sum(values) / len(values)


def balanced_similarity_groups(similarity: torch.Tensor, group_size: int) -> List[List[int]]:
    """Greedily partition heads into equal-size, high-similarity groups.

    Deterministic tie-breaking uses the lexicographically smallest head ids.
    The objective is local and intentionally simple; the emitted metadata lets
    a later agent compare it against adjacent grouping on a real cache.
    """
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square matrix")
    heads = int(similarity.shape[0])
    if group_size <= 0 or heads % group_size:
        raise ValueError("group_size must be positive and divide head count")
    if not torch.isfinite(similarity).all():
        raise ValueError("similarity contains nonfinite values")
    if group_size == 1:
        return [[i] for i in range(heads)]

    remaining = set(range(heads))
    groups: List[List[int]] = []
    while remaining:
        candidates = []
        rem = sorted(remaining)
        for ai, a in enumerate(rem):
            for b in rem[ai + 1:]:
                candidates.append((float(similarity[a, b]), -a, -b, a, b))
        _, _, _, a, b = max(candidates)
        group = [a, b]
        remaining.remove(a)
        remaining.remove(b)
        while len(group) < group_size:
            best = max(
                remaining,
                key=lambda x: (
                    sum(float(similarity[x, y]) for y in group) / len(group),
                    -x,
                ),
            )
            group.append(best)
            remaining.remove(best)
        groups.append(sorted(group))
    groups.sort(key=lambda g: (min(g), g))
    return groups


def groups_to_permutation(groups: Iterable[Iterable[int]]) -> List[int]:
    return [int(head) for group in groups for head in group]


def grouping_objective(similarity: torch.Tensor, groups: Sequence[Sequence[int]]) -> float:
    scores = [_group_score(similarity, group) for group in groups if len(group) > 1]
    return sum(scores) / len(scores) if scores else 1.0


def _validate_permutation(permutation: Sequence[int], heads: int) -> List[int]:
    perm = [int(x) for x in permutation]
    if len(perm) != heads or sorted(perm) != list(range(heads)):
        raise ValueError(f"permutation must contain each head 0..{heads - 1} exactly once")
    return perm


def apply_head_permutation_(module, permutation: Sequence[int]) -> None:
    """Reorder heads in place without changing the old-head function."""
    kind, heads, dh = _validate_shape(module)
    perm = _validate_permutation(permutation, heads)
    with torch.no_grad():
        if kind == "self":
            weight = module.qkv.weight.detach().reshape(3, heads, dh, -1)
            module.qkv.weight.copy_(weight[:, perm].reshape_as(module.qkv.weight))
            if module.qkv.bias is not None:
                bias = module.qkv.bias.detach().reshape(3, heads, dh)
                module.qkv.bias.copy_(bias[:, perm].reshape_as(module.qkv.bias))
        else:
            q = module.q.weight.detach().reshape(heads, dh, -1)
            module.q.weight.copy_(q[perm].reshape_as(module.q.weight))
            if module.q.bias is not None:
                qb = module.q.bias.detach().reshape(heads, dh)
                module.q.bias.copy_(qb[perm].reshape_as(module.q.bias))
            kv = module.kv.weight.detach().reshape(2, heads, dh, -1)
            module.kv.weight.copy_(kv[:, perm].reshape_as(module.kv.weight))
            if module.kv.bias is not None:
                kvb = module.kv.bias.detach().reshape(2, heads, dh)
                module.kv.bias.copy_(kvb[:, perm].reshape_as(module.kv.bias))
        out = getattr(module, "out", None)
        if out is not None:
            ow = out.weight.detach().reshape(out.out_features, heads, dh)
            out.weight.copy_(ow[:, perm].reshape_as(out.weight))


def scale_qk_(module, factor: float) -> None:
    """Scale every Q and K channel in place, including biases if present."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("q/k scale must be finite and positive")
    kind, heads, dh = _validate_shape(module)
    inner = heads * dh
    with torch.no_grad():
        if kind == "self":
            w = module.qkv.weight
            w[:2 * inner].mul_(factor)
            if module.qkv.bias is not None:
                module.qkv.bias[:2 * inner].mul_(factor)
        else:
            module.q.weight.mul_(factor)
            module.kv.weight[:inner].mul_(factor)
            if module.q.bias is not None:
                module.q.bias.mul_(factor)
            if module.kv.bias is not None:
                module.kv.bias[:inner].mul_(factor)


def trunk_attention_modules(net) -> List[Tuple[str, object]]:
    """The B trunk modules controlled by ModelConfig.trunk_heads."""
    found: List[Tuple[str, object]] = []
    cross = getattr(getattr(net, "cross", None), "attn", None)
    if cross is not None:
        found.append(("cross.attn", cross))
    for index, block in enumerate(getattr(net, "blocks", ())):
        attn = getattr(block, "attn", None)
        if attn is not None:
            found.append((f"blocks.{index}.attn", attn))
    if not found:
        raise ValueError("no B trunk attention modules found (expected cross.attn and blocks.*.attn)")
    return found
