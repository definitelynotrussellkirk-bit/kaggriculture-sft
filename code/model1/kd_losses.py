"""Finite-support distillation and LOCAL exact class-mass utilities.

No rule here claims equivalence of autoregressive suffixes or game trajectories.
"""
from __future__ import annotations
import math
import torch

def _check(z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if z.ndim < 2 or valid.shape != z.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError('Expected logits [..., K] and boolean valid [...]')
    z = z.float()
    active = valid.unsqueeze(-1)
    if ((torch.isnan(z) | torch.isposinf(z)) & active).any():
        raise ValueError('Active logits contain NaN or positive infinity')
    if (valid & ~torch.isfinite(z).any(-1)).any():
        raise ValueError('An active decision has empty finite support')
    return torch.where(active, z, torch.zeros_like(z))

def masked_kl(student: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor,
              temperature: float = 1.0) -> torch.Tensor:
    if student.shape != teacher.shape or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Matching shapes and positive finite temperature are required')
    s, t = _check(student, valid), _check(teacher.detach(), valid)
    support = torch.isfinite(t)
    if (valid.unsqueeze(-1) & support & ~torch.isfinite(s)).any():
        raise ValueError('Student excludes a class with positive teacher support')
    lp, lq = (t / temperature).log_softmax(-1), (s / temperature).log_softmax(-1)
    # Never form -inf - -inf or 0 * inf, including in the backward graph.
    lp_safe = torch.where(support, lp, 0.0)
    lq_safe = torch.where(support, lq, 0.0)
    k = (lp.exp() * (lp_safe - lq_safe)).sum(-1) * temperature**2
    return torch.where(valid, k, 0.0)

def set_nll(logits: torch.Tensor, target: torch.Tensor, allowed: torch.Tensor,
            valid: torch.Tensor) -> torch.Tensor:
    z = _check(logits, valid)
    if allowed.shape != z.shape or allowed.dtype != torch.bool or target.shape != valid.shape:
        raise ValueError('Invalid target/allowed shapes or allowed dtype')
    if target.dtype != torch.long:
        raise ValueError('Targets must be int64')
    if (valid & ((target < 0) | (target >= z.shape[-1]))).any():
        raise ValueError('Active target outside vocabulary')
    if (valid & ~allowed.gather(-1, target.clamp(0, z.shape[-1]-1).unsqueeze(-1)).squeeze(-1)).any():
        raise ValueError('Active target is not a member of its equivalence class')
    if (valid & ~(allowed & torch.isfinite(z)).any(-1)).any():
        raise ValueError('Active target class has no finite support')
    ok = allowed | ~valid.unsqueeze(-1)
    nll = torch.logsumexp(z, -1) - torch.logsumexp(z.masked_fill(~ok, float('-inf')), -1)
    return torch.where(valid, nll, 0.0)

def class_log_probs(logits: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    """Log mass at canonical representative indices; -inf at other indices.

    Caller supplies a COMPLETE state-conditioned partition, not only one
    recorded target's allowed set. Classes include every vocabulary item.
    """
    z = _check(logits, torch.ones(logits.shape[:-1], dtype=torch.bool, device=logits.device))
    if class_ids.shape != z.shape or class_ids.dtype != torch.long:
        raise ValueError('class_ids must be int64 with the logits shape')
    if ((class_ids < 0) | (class_ids >= z.shape[-1])).any():
        raise ValueError('Class representative outside vocabulary')
    if not torch.equal(class_ids.gather(-1, class_ids), class_ids):
        raise ValueError('Representatives must point to themselves')
    # Scatter log-sum-exp without a [K,K] intermediate or softmax underflow.
    maxima = torch.full_like(z, float('-inf')).scatter_reduce(-1, class_ids, z, reduce='amax', include_self=True)
    m = maxima.gather(-1, class_ids)
    finite = torch.isfinite(z)
    difference = torch.where(finite, z, 0.0) - torch.where(finite, m, 0.0)
    terms = torch.where(finite, difference.exp(), 0.0)
    totals = torch.zeros_like(z).scatter_add(-1, class_ids, terms)
    nonempty = totals > 0
    mass = torch.where(nonempty, maxima, 0.0) + totals.clamp_min(torch.finfo(z.dtype).tiny).log()
    return torch.where(nonempty, mass - torch.logsumexp(z, -1, keepdim=True), float('-inf'))

def class_argmax(logits: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    return class_log_probs(logits, class_ids).argmax(-1)

def wire_classes(codec, operation: str, reference: int) -> list[int]:
    if not isinstance(reference, int) or reference < 0:
        raise ValueError('reference must be a nonnegative integer')
    groups = {}
    result = []
    for q in range(codec.QTY_BIN_COUNT):
        wire = codec.decode_qty(operation, q, reference)
        groups.setdefault(wire, q)
        result.append(groups[wire])
    return result
