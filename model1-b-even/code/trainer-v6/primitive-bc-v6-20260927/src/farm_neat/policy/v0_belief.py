"""Permutation-equivariant, fixture-first belief mass and conditional WDL heads.

``V0BeliefHead(context_dim, hypothesis_dim)(actor_features, hypothesis_features,
    hypothesis_ids=..., kinds=...)`` scores each public hypothetical model with
ONE shared MLP. There are no slot embeddings, position features or K-dependent
parameters. Hashes only associate rows; neither their bytes nor sorted positions
enter the network. UNKNOWN is an optional row, not an extra forced reserve. The
record's ``k`` counts ALL rows, including UNKNOWN (at most one, at most 64 total).

The caller owns the public-only encoder and predictive-model definitions. Tensors
must contain actor-time public/own-memory features, NOT actual opponent identity,
hidden state, realized outcomes or future randomness. This API accepts no record
or trainer dictionary. It cannot certify where arbitrary caller tensors came from.
Equal descriptors receive equal scores up to floating-point scorer tolerance;
different IDs do not certify distinctness.
Use v0_belief_records' model/profile validation when constructing a record.

``belief_masses`` also consumes fixture/public-likelihood logits directly. It
exposes the raw scores, raw prior, transformed initial prior, shifted log evidence,
posterior and constrained output, with differentiable float64 probabilities AND
log probabilities. A neural score is an UNCALIBRATED relative evidence estimate,
not an observed likelihood. Evidence-producer/source/through-step provenance and
real calibration remain the caller's responsibility. No likelihood is fitted here.

Defaults: uniform prior, zero floor, no UNKNOWN. The optional prior transform is
``power_log_odds_cap_v1``: subtract max log(raw_prior), multiply by strength, clamp
below -log(odds_cap) if supplied, then softmax. The cap applies ONLY at initialization;
subsequent evidence can reverse or exceed it. This is a declared experimental
transform, not an adopted popularity prior. The frozen record validates the initial
vector/cap but does not specify a unique prior algorithm.

Output-only floors: ``none``; the exact record's ``two_hypothesis_clamp``; or
``uniform_mix`` = p_min + (1-K*p_min)*posterior. The latter exports the record's
``lower_bound_only`` enum, while retaining its exact algorithm here. It is NOT a
Euclidean/KL projection. A positive floor may distort even overwhelming evidence;
constrained probabilities are never called calibrated and never fed into a later
posterior. No missing conditional law is inferred or renormalized away.

The positive-support wire refuses underflow or nonfinite probabilities rather than
adding epsilon, clipping logits or pruning hypotheses. Arithmetic promotes logits
to float64 before shifts/normalization. A common evidence offset is removed BEFORE
adding the prior, so huge common offsets do not erase its odds. Logits and prior
probabilities must be finite; raw priors must already sum to one within the wire's
1e-12 tolerance (use float64 for serialization-quality inputs). Probability tensors
use the wire's exponentiate-then-normalize operation order. Stable log probabilities
remain in score space to avoid reciprocal-subnormal gradients; exp(log_mass) need
not be bitwise equal to rounded subnormal mass. Final wire validation is mandatory.

Permutation/batch comparisons use scorer-dtype tolerance (FP64 1e-12, FP32 1e-6
for small fixtures); bitwise equal rescores or serialized hashes across different
row/batch layouts are not promised. Identity-only relabeling never changes inputs.

Supports one unpadded catalog, optionally batched over contexts with that same row
association: context [D]/[B,D], hypotheses [K,F]/[B,K,F]. There is no padding mask or
cross-example broadcast of hypotheses. ``record_fields`` explicitly detaches ONE
prediction for a new record; join its hypothesis rows by hash, NOT by list position.
It supplies numerical fields only, never inventing model or evidence provenance,
conditional WDL laws, distinctness proofs, or a frozen actor snapshot. Final sealing
and validation remain ``v0_belief_records.make_record``. Existing records must not be
rewritten with hindsight. No V0 action, compiler, actor architecture, trainer,
continuation, objective or regret-admission behavior is changed by this module.

Conditional-value extension (opt in via ``V0BeliefHead(..., candidate_dim=F)`` or
instantiate ``V0BeliefConditionalValueHead``): a separate shared nonlinear head
outputs LOSS/DRAW/WIN laws per hypothesis/candidate, indexed by both canonical IDs.
The mass head's return type and default parameters are unchanged. The public
``encode_conditional_candidates`` adapter reuses regret-comparator geometry and
features. The TRAINER-ONLY ``conditional_targets_from_tensor`` bridge consumes
existing verified outcome tensors; labels/masks/support metadata never enter the
actor forward pass. Missing predictions/targets stay NaN internally and predictions
export as None; ``observed_rows`` refuses broadcasting or an empty implicit loss.
No mixture-consistency loss, selector or production integration is implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

import torch
from torch import Tensor, nn

from farm_neat.training.v0_belief_records import MAX_CANDIDATES, MAX_HYPOTHESES

SCHEMA = 'farm-v0-belief-head-v1'
_HASH = re.compile(r'[0-9a-f]{64}\Z')


def _number(value: float, name: str, *, low: float = 0.) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < low:
        raise ValueError(f'{name} must be a finite number >= {low}')
    return float(value)


def _tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f'{name} must be a floating tensor, never a record/dictionary')
    if not value.is_floating_point():
        raise TypeError(f'{name} must be a floating tensor')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')


def _catalog(hypothesis_ids: Sequence[str], kinds: Sequence[str] | None, k: int):
    if isinstance(hypothesis_ids, (str, bytes)):
        raise TypeError('hypothesis_ids must be a sequence of model hashes')
    keys = tuple(hypothesis_ids)
    if not 1 <= k <= MAX_HYPOTHESES or len(keys) != k:
        raise ValueError(f'one to {MAX_HYPOTHESES} hypothesis rows/IDs must match')
    if any(type(key) is not str or not _HASH.fullmatch(key) for key in keys):
        raise ValueError('hypothesis_ids must be lowercase model SHA256 hashes')
    if len(set(keys)) != k:
        raise ValueError('duplicate hypothesis_id; slots do not create new models')
    tags = ('MODEL',) * k if kinds is None else tuple(kinds)
    if len(tags) != k or any(tag not in ('MODEL', 'UNKNOWN') for tag in tags):
        raise ValueError('kinds must match rows and be MODEL or UNKNOWN')
    if tags.count('UNKNOWN') > 1:
        raise ValueError('at most one UNKNOWN residual expert is supported')
    return keys, tags


def _positive_probabilities(value: Tensor, name: str) -> None:
    if (not bool(torch.isfinite(value).all()) or not bool((value > 0).all())
            or not bool((value <= 1).all())):
        raise ValueError(f'{name} must be finite, positive and representable; no epsilon repair')
    if not bool(((value.sum(-1) - 1.).abs() <= 1e-12).all()):
        raise ValueError(f'{name} must sum to one within 1e-12')


@dataclass(frozen=True)
class BeliefHeadOutput:
    """Differentiable tensors; not an immutable actor-time record or a calibration certificate."""
    hypothesis_ids: tuple[str, ...]
    kinds: tuple[str, ...]
    unconstrained_logits: Tensor
    raw_prior: Tensor
    initial_mass: Tensor
    log_evidence: Tensor
    posterior_log_mass: Tensor
    posterior_mass: Tensor
    constrained_log_mass: Tensor
    constrained_mass: Tensor
    prior_transform: str
    prior_strength: float
    prior_odds_cap: float | None
    prior_source_sha256: str | None
    floor_transform: str
    p_min: float

    @property
    def calibration_status(self) -> str:
        return 'UNMEASURED'

    @property
    def unknown_posterior_mass(self) -> Tensor | None:
        return (self.posterior_mass[..., self.kinds.index('UNKNOWN')]
                if 'UNKNOWN' in self.kinds else None)

    @property
    def unknown_constrained_mass(self) -> Tensor | None:
        return (self.constrained_mass[..., self.kinds.index('UNKNOWN')]
                if 'UNKNOWN' in self.kinds else None)

    def record_fields(self, *, batch_index: int | None = None,
                      mass_used: str = 'posterior_mass') -> dict:
        """Detach exact float64 numerical fields for ONE new record, not a sealed record.

        Merge rows by hypothesis_id into a public model/evidence template, retaining
        nullable conditionals/probes. Then call the existing make_record validator.
        A batch requires an explicit index; this boundary alone does not pin context.
        """
        if mass_used not in ('posterior_mass', 'constrained_mass'):
            raise ValueError('mass_used must explicitly name posterior_mass or constrained_mass')
        if self.posterior_mass.ndim == 2:
            if type(batch_index) is not int or not 0 <= batch_index < self.posterior_mass.shape[0]:
                raise ValueError('a valid explicit batch_index is required for batched output')
        elif batch_index is not None:
            raise ValueError('batch_index must be None for unbatched output')
        names = ('raw_prior', 'initial_mass', 'log_evidence', 'posterior_mass', 'constrained_mass')
        vectors = {}
        for name in names:
            value = getattr(self, name)
            if value.ndim == 2:
                value = value[batch_index]
            vectors[name] = value.detach().cpu().tolist()
        return dict(k=len(self.hypothesis_ids), p_min=self.p_min,
                    floor_transform=('lower_bound_only' if self.floor_transform == 'uniform_mix'
                                     else self.floor_transform), mass_used=mass_used,
                    prior=dict(mode=('uniform' if self.prior_transform == 'uniform_v1' else 'preloaded'),
                               strength=self.prior_strength, odds_cap=self.prior_odds_cap,
                               source_sha256=self.prior_source_sha256),
                    hypotheses=[dict(hypothesis_id=key, kind=self.kinds[i],
                                     **{name: values[i] for name, values in vectors.items()})
                                for i, key in enumerate(self.hypothesis_ids)])


def belief_masses(logits: Tensor, *, hypothesis_ids: Sequence[str],
                  kinds: Sequence[str] | None = None, raw_prior: Tensor | None = None,
                  prior_strength: float = 1., prior_odds_cap: float | None = None,
                  prior_source_sha256: str | None = None, p_min: float = 0.,
                  floor_transform: str = 'none') -> BeliefHeadOutput:
    """Normalize [K] or [B,K] relative public-evidence scores without detaching gradients.

    Preloaded raw_prior may be [K] (shared) or match logits; it must be positive,
    normalized and associated in exactly the supplied hypothesis order. A uniform
    prior has no source or cap options and exports strength=0, odds_cap=1.
    """
    _tensor(logits, 'logits')
    if logits.ndim not in (1, 2) or (logits.ndim == 2 and logits.shape[0] == 0):
        raise ValueError('logits must be nonempty [K] or [B,K]')
    k = logits.shape[-1]
    keys, tags = _catalog(hypothesis_ids, kinds, k)
    floor = _number(p_min, 'p_min')
    num, den = floor.as_integer_ratio()
    if num*k > den:
        raise ValueError('infeasible floor: exact K*p_min exceeds one')
    if floor_transform not in ('none', 'two_hypothesis_clamp', 'uniform_mix'):
        raise ValueError('unknown floor_transform')
    if floor_transform == 'none' and floor != 0:
        raise ValueError('none floor_transform requires p_min=0')
    if floor_transform == 'two_hypothesis_clamp' and k != 2:
        raise ValueError('two_hypothesis_clamp requires exactly two rows')
    strength = _number(prior_strength, 'prior_strength')
    x = logits.to(dtype=torch.float64)
    if raw_prior is None:
        if prior_source_sha256 is not None or prior_odds_cap is not None or strength != 1.:
            raise ValueError('uniform prior accepts no preloaded-prior options')
        prior = torch.full_like(x, 1./k)
        initial = prior
        strength, cap, mode = 0., 1., 'uniform_v1'
    else:
        if (type(prior_source_sha256) is not str
                or not _HASH.fullmatch(prior_source_sha256) or strength <= 0):
            raise ValueError('preloaded prior needs source_sha256 and positive prior_strength')
        cap = None if prior_odds_cap is None else _number(prior_odds_cap, 'prior_odds_cap', low=1.)
        _tensor(raw_prior, 'raw_prior')
        if raw_prior.device != logits.device:
            raise ValueError('raw_prior and logits must share device')
        if raw_prior.shape != logits.shape and raw_prior.shape != (k,):
            raise ValueError('raw_prior must be [K] or match logits shape')
        prior = raw_prior.to(dtype=torch.float64).expand_as(x)
        _positive_probabilities(prior, 'raw_prior')
        log_prior = prior.log()
        weighted = (log_prior - log_prior.amax(-1, keepdim=True)) * strength
        if not bool(torch.isfinite(weighted).all()):
            raise ValueError('weighted log prior must be finite and representable')
        if cap is not None:
            weighted = weighted.clamp_min(-math.log(cap))
        initial = weighted.softmax(-1)
        mode = 'power_log_odds_cap_v1'
    _positive_probabilities(initial, 'initial_mass')
    # Crucial order: do not add log(initial) to a potentially enormous offset.
    evidence = x - x.amax(-1, keepdim=True)
    if not bool(torch.isfinite(evidence).all()):
        raise ValueError('relative log evidence must be finite and representable')
    weights = initial.log() + evidence
    log_posterior = weights.log_softmax(-1)
    # Match the wire's exponentiate-then-normalize order. exp(log_softmax)
    # can round a positive subnormal tail one ULP away from the record oracle.
    unnormalized = (weights - weights.amax(-1, keepdim=True)).exp()
    posterior = unnormalized / unnormalized.sum(-1, keepdim=True)
    _positive_probabilities(posterior, 'posterior_mass')
    if floor_transform == 'none':
        constrained, log_constrained = posterior, log_posterior
    elif floor_transform == 'two_hypothesis_clamp':
        # Never compute the other coordinate as 1-first: that loses tiny live tails.
        constrained = posterior.clamp(min=floor, max=1.-floor)
        log_constrained = log_posterior.clamp(
            min=(math.log(floor) if floor else -math.inf), max=math.log1p(-floor))
    else:
        remaining = (den-k*num)/den  # exact feasibility; no rounded K*p_min loophole
        constrained = floor + remaining*posterior
        # Stay in score space: log(subnormal mass) backpropagates 1/p == inf.
        log_constrained = (torch.logaddexp(
            torch.full_like(log_posterior, math.log(floor)),
            log_posterior + (math.log(remaining) if remaining else -math.inf))
            if floor else log_posterior)
    _positive_probabilities(constrained, 'constrained_mass')
    if not bool((constrained >= floor).all()):
        raise ValueError('constrained_mass violates p_min')
    return BeliefHeadOutput(keys, tags, logits, prior, initial, evidence, log_posterior,
                            posterior, log_constrained, constrained, mode, strength, cap,
                            prior_source_sha256, floor_transform, floor)


class V0BeliefHead(nn.Module):
    """A shared pointwise scorer; the production V0 trunk/encoder is caller-owned.

    known/unknown kind is the only categorical feature. Permuting descriptor rows,
    kinds and optional priors permutes output rows; changing hash IDs alone does
    not change numbers. Tensor batches share one catalog. No stateful evidence
    updater, choice selector or training loop is owned. Optional ``candidate_dim``
    adds a shared ``conditional_value`` subhead, called separately from mass scoring.
    Without it, old parameter names, RNG consumption and forward behavior are unchanged.
    """
    def __init__(self, context_dim: int, hypothesis_dim: int, *, hidden_dim: int = 64,
                 candidate_dim: int | None = None):
        super().__init__()
        if any(type(n) is not int or n <= 0 for n in (context_dim, hypothesis_dim, hidden_dim)):
            raise ValueError('feature and hidden dimensions must be positive integers')
        self.context_dim, self.hypothesis_dim = context_dim, hypothesis_dim
        self.scorer = nn.Sequential(nn.Linear(context_dim + hypothesis_dim + 1, hidden_dim),
                                    nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))
        self.conditional_value = (None if candidate_dim is None else
            V0BeliefConditionalValueHead(context_dim, hypothesis_dim, candidate_dim, hidden_dim=hidden_dim))

    def forward(self, actor_features: Tensor, hypothesis_features: Tensor, *,
                hypothesis_ids: Sequence[str], kinds: Sequence[str] | None = None,
                raw_prior: Tensor | None = None, prior_strength: float = 1.,
                prior_odds_cap: float | None = None, prior_source_sha256: str | None = None,
                p_min: float = 0., floor_transform: str = 'none') -> BeliefHeadOutput:
        _tensor(actor_features, 'actor_features')
        _tensor(hypothesis_features, 'hypothesis_features')
        if (actor_features.ndim not in (1, 2)
                or hypothesis_features.ndim != actor_features.ndim+1
                or actor_features.shape[-1] != self.context_dim
                or hypothesis_features.shape[-1] != self.hypothesis_dim
                or actor_features.shape[:-1] != hypothesis_features.shape[:-2]
                or (actor_features.ndim == 2 and actor_features.shape[0] == 0)):
            raise ValueError('expected context [D]/[B,D] and hypotheses [K,F]/[B,K,F]; no padding')
        parameter = self.scorer[0].weight
        if any(t.device != parameter.device or t.dtype != parameter.dtype
               for t in (actor_features, hypothesis_features)):
            raise ValueError('features and head parameters must share dtype/device')
        k = hypothesis_features.shape[-2]
        keys, tags = _catalog(hypothesis_ids, kinds, k)
        context = actor_features.unsqueeze(-2).expand(*hypothesis_features.shape[:-1], self.context_dim)
        unknown = hypothesis_features.new_tensor([tag == 'UNKNOWN' for tag in tags])
        unknown = unknown.reshape(*([1] * (hypothesis_features.ndim-2)), k, 1)
        unknown = unknown.expand(*hypothesis_features.shape[:-1], 1)
        logits = self.scorer(torch.cat((context, hypothesis_features, unknown), dim=-1)).squeeze(-1)
        return belief_masses(logits, hypothesis_ids=keys, kinds=tags, raw_prior=raw_prior,
                             prior_strength=prior_strength, prior_odds_cap=prior_odds_cap,
                             prior_source_sha256=prior_source_sha256,
                             p_min=p_min, floor_transform=floor_transform)


CONDITIONAL_SCHEMA = 'farm-v0-belief-conditional-value-v1'
WDL_ORDER = ('LOSS', 'DRAW', 'WIN')


def _candidate_catalog(candidate_ids: Sequence[str], count: int) -> tuple[str, ...]:
    if isinstance(candidate_ids, (str, bytes)):
        raise TypeError('candidate_ids must be a sequence of canonical candidate hashes')
    keys = tuple(candidate_ids)
    if not 1 <= count <= MAX_CANDIDATES or len(keys) != count:
        raise ValueError(f'one to {MAX_CANDIDATES} candidate rows/IDs must match')
    if any(type(key) is not str or not _HASH.fullmatch(key) for key in keys):
        raise ValueError('candidate_ids must be lowercase canonical candidate SHA256 hashes')
    if len(set(keys)) != count:
        raise ValueError('duplicate candidate_id; aliases do not create new candidates')
    return keys


def _wdl_shape(value: Tensor) -> None:
    if (value.ndim not in (3, 4) or value.shape[-1] != 3
            or any(n == 0 for n in value.shape)):
        raise ValueError('expected nonempty [K,C,3] or [B,K,C,3]; no implicit batch broadcast')


def _boolean_mask(mask: Tensor, shape: torch.Size, device: torch.device, name: str) -> None:
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
        raise TypeError(f'{name} must be a boolean tensor')
    if mask.shape != shape or mask.device != device:
        raise ValueError(f'{name} must exactly match shape/device; no broadcasting')


@dataclass(frozen=True)
class ConditionalValueTargets:
    """TRAINER ONLY, in LOSS/DRAW/WIN order. NaN triplets denote unobserved cells.

    An explicit target mask is distinct from a prediction's support mask. No label
    is inferred from a hypothesis's mass, kind, hash or row position. Rows may be
    reordered; observed_rows performs the identity join. Batches share both catalogs
    and must match prediction batches exactly. This low-level carrier does not
    authenticate labels; ``conditional_targets_from_tensor`` revalidates receipts.
    Targets never enter the actor head. No loss/weight/admission policy is owned.
    """
    hypothesis_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    probabilities: Tensor
    observed_mask: Tensor

    def __post_init__(self) -> None:
        p = self.probabilities
        if not isinstance(p, Tensor) or not p.is_floating_point():
            raise TypeError('target probabilities must be a floating tensor')
        _wdl_shape(p)
        keys, _ = _catalog(self.hypothesis_ids, None, p.shape[-3])
        candidates = _candidate_catalog(self.candidate_ids, p.shape[-2])
        object.__setattr__(self, 'hypothesis_ids', keys)
        object.__setattr__(self, 'candidate_ids', candidates)
        _boolean_mask(self.observed_mask, p.shape[:-1], p.device, 'observed_mask')
        if p.requires_grad:
            raise ValueError('evidence targets must not require gradients')
        observed = p[self.observed_mask].to(torch.float64)
        if (not bool(torch.isfinite(observed).all()) or not bool((observed >= 0).all())
                or not bool((observed <= 1).all())
                or not bool(((observed.sum(-1)-1.).abs() <= 1e-12).all())):
            raise ValueError('observed targets must be finite nonnegative WDL laws summing to one')
        if not bool(torch.isnan(p[~self.observed_mask]).all()):
            raise ValueError('unobserved targets must be NaN triplets, never fabricated zeros/laws')


@dataclass(frozen=True)
class ConditionalValueOutput:
    """Differentiable, uncalibrated conditional laws, axes [..., hypothesis, candidate, WDL].

    IDs associate rows only, never features. ``prediction_mask`` means a conditional
    model is available; it does NOT mean observed training evidence exists. Missing
    predictions are NaN internally / None on the wire, not a uniform law. Known
    hypotheses default to available; UNKNOWN requires explicit predictive support.
    A caller setting that support must supply a residual predictor, not label truth.
    This object does not aggregate masses or select actions, and cannot admit teachers.
    """
    hypothesis_ids: tuple[str, ...]
    kinds: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    logits: Tensor
    log_probabilities: Tensor
    probabilities: Tensor
    prediction_mask: Tensor

    @property
    def wdl_order(self) -> tuple[str, ...]:
        return WDL_ORDER

    @property
    def calibration_status(self) -> str:
        return 'UNMEASURED'

    def pair(self, hypothesis_id: str, candidate_id: str) -> Tensor:
        """Look up one [3] or [B,3] law by canonical identities, not numeric slots."""
        return self.probabilities[..., self.hypothesis_ids.index(hypothesis_id),
                                  self.candidate_ids.index(candidate_id), :]

    def record_fields(self, *, batch_index: int | None = None) -> dict:
        """Detach fields for ONE NEW actor-time record; merge by both identities.

        Use v0_belief_records.make_record after joining with the mass/model/probe
        template. Does not create a sealed record or rewrite past predictions.
        """
        p, mask = self.probabilities, self.prediction_mask
        if p.ndim == 4:
            if type(batch_index) is not int or not 0 <= batch_index < p.shape[0]:
                raise ValueError('a valid explicit batch_index is required for batched output')
            p, mask = p[batch_index], mask[batch_index]
        elif batch_index is not None:
            raise ValueError('batch_index must be None for unbatched output')
        values, available = p.detach().cpu().tolist(), mask.detach().cpu().tolist()
        return dict(hypotheses=[dict(hypothesis_id=h, conditional_wdl={
            c: values[i][j] if available[i][j] else None
            for j, c in enumerate(self.candidate_ids)}) for i, h in enumerate(self.hypothesis_ids)])

    def observed_rows(self, targets: ConditionalValueTargets) -> tuple[Tensor, Tensor]:
        """TRAINER ONLY: aligned [N,3] log predictions/labels for supported observations.

        Select BEFORE computing a loss: NaN*0 is not masking. Reject no observations
        rather than return a spurious zero loss. The loss owner decides an explicit
        skip, weighting, split independence, objective and training eligibility.
        No target or mask is fed back into forward(). No broadcasting is permitted.
        """
        if not isinstance(targets, ConditionalValueTargets):
            raise TypeError('targets must be ConditionalValueTargets, never an actor record')
        targets.__post_init__()  # Tensor contents can change despite the frozen carrier.
        if (set(targets.hypothesis_ids) != set(self.hypothesis_ids)
                or set(targets.candidate_ids) != set(self.candidate_ids)):
            raise ValueError('prediction and target identities must match exactly')
        p = targets.probabilities
        if p.shape != self.probabilities.shape or p.device != self.probabilities.device:
            raise ValueError('target and prediction batch shape/device must match; no broadcasting')
        hi = [targets.hypothesis_ids.index(h) for h in self.hypothesis_ids]
        ci = [targets.candidate_ids.index(c) for c in self.candidate_ids]
        aligned = p[..., hi, :, :][..., ci, :]
        mask = targets.observed_mask[..., hi, :][..., ci] & self.prediction_mask
        if not bool(mask.any()):
            raise ValueError('no observed supported conditional targets; explicitly skip this batch')
        return self.log_probabilities[mask], aligned[mask].to(self.log_probabilities.dtype)


def conditional_wdl(logits: Tensor, *, hypothesis_ids: Sequence[str], candidate_ids: Sequence[str],
                    kinds: Sequence[str] | None = None,
                    prediction_mask: Tensor | None = None) -> ConditionalValueOutput:
    """Normalize [K,C,3]/[B,K,C,3] logits in float64; no mass-weighting or scalarization.

    Outcome probability zero is legal (unlike live belief mass). Finite stable log
    probabilities remain available for observed rows even when exp underflows.
    ``prediction_mask`` must exactly match [...,K,C], not a broadcastable label mask.
    """
    _tensor(logits, 'conditional logits')
    _wdl_shape(logits)
    keys, tags = _catalog(hypothesis_ids, kinds, logits.shape[-3])
    candidates = _candidate_catalog(candidate_ids, logits.shape[-2])
    if prediction_mask is None:
        shape = ([1] if logits.ndim == 4 else []) + [len(keys), 1]
        prediction_mask = torch.tensor([tag != 'UNKNOWN' for tag in tags],
            dtype=torch.bool, device=logits.device).reshape(shape).expand(logits.shape[:-1])
    _boolean_mask(prediction_mask, logits.shape[:-1], logits.device, 'prediction_mask')
    x = logits.to(torch.float64)
    log_p = x.log_softmax(-1)
    if not bool(torch.isfinite(log_p).all()):
        raise ValueError('relative WDL logits must be finite and representable')
    p = log_p.exp()
    return ConditionalValueOutput(keys, tags, candidates, logits,
        log_p.masked_fill(~prediction_mask.unsqueeze(-1), math.nan),
        p.masked_fill(~prediction_mask.unsqueeze(-1), math.nan), prediction_mask)


class V0BeliefConditionalValueHead(nn.Module):
    """Shared nonlinear public-context/hypothesis/candidate WDL predictor.

    Takes context [D]/[B,D], hypotheses [K,H]/[B,K,H], candidates [C,F]/[B,C,F].
    Batches share ID catalogs; no padding, slot embeddings or cross-batch broadcast.
    Parameters do not depend on K/C. Nonlinearity allows opposite candidate rankings
    under different hypotheses even when belief masses are equal. Kind is the only
    categorical input; SHA bytes and true opponent classes never enter the network.

    Caller owns actor-safe encodings; arbitrary tensors cannot certify provenance.
    Use encode_conditional_candidates for the existing comparator's public features.
    Work is O(B*K*C*hidden_dim*(D+H+F)); K<=64, C<=32. This optional head is not
    attached to the production scorer/trainer and is not a value/admission oracle.
    """
    def __init__(self, context_dim: int, hypothesis_dim: int, candidate_dim: int, *, hidden_dim: int = 64):
        super().__init__()
        if any(type(n) is not int or n <= 0 for n in (context_dim, hypothesis_dim, candidate_dim, hidden_dim)):
            raise ValueError('feature and hidden dimensions must be positive integers')
        self.context_dim, self.hypothesis_dim, self.candidate_dim = context_dim, hypothesis_dim, candidate_dim
        self.scorer = nn.Sequential(nn.Linear(context_dim+hypothesis_dim+candidate_dim+1, hidden_dim),
                                    nn.Tanh(), nn.Linear(hidden_dim, 3))

    def forward(self, actor_features: Tensor, hypothesis_features: Tensor, candidate_features: Tensor, *,
                hypothesis_ids: Sequence[str], candidate_ids: Sequence[str],
                kinds: Sequence[str] | None = None,
                prediction_mask: Tensor | None = None) -> ConditionalValueOutput:
        for value, name in ((actor_features, 'actor_features'), (hypothesis_features, 'hypothesis_features'),
                            (candidate_features, 'candidate_features')):
            _tensor(value, name)
        x, h, c = actor_features, hypothesis_features, candidate_features
        if (x.ndim not in (1, 2) or h.ndim != x.ndim+1 or c.ndim != x.ndim+1
                or x.shape[-1] != self.context_dim or h.shape[-1] != self.hypothesis_dim
                or c.shape[-1] != self.candidate_dim or x.shape[:-1] != h.shape[:-2]
                or x.shape[:-1] != c.shape[:-2] or (x.ndim == 2 and x.shape[0] == 0)):
            raise ValueError('context/hypothesis/candidate shapes must match exactly; no batch broadcast/padding')
        param = self.scorer[0].weight
        if any(t.dtype != param.dtype or t.device != param.device for t in (x, h, c)):
            raise ValueError('features and head parameters must share dtype/device')
        k, count = h.shape[-2], c.shape[-2]
        keys, tags = _catalog(hypothesis_ids, kinds, k)
        candidates = _candidate_catalog(candidate_ids, count)
        shape = (*x.shape[:-1], k, count)
        context = x.unsqueeze(-2).unsqueeze(-2).expand(*shape, self.context_dim)
        hypotheses = h.unsqueeze(-2).expand(*shape, self.hypothesis_dim)
        proposals = c.unsqueeze(-3).expand(*shape, self.candidate_dim)
        unknown = h.new_tensor([tag == 'UNKNOWN' for tag in tags]).reshape(
            *([1] * (x.ndim-1)), k, 1, 1).expand(*shape, 1)
        logits = self.scorer(torch.cat((context, hypotheses, proposals, unknown), dim=-1))
        return conditional_wdl(logits, hypothesis_ids=keys, candidate_ids=candidates,
                               kinds=tags, prediction_mask=prediction_mask)


def encode_conditional_candidates(actor_context: dict, candidates: list[dict]) -> dict:
    """Actor-only bridge to the EXISTING 128-D regret-comparator representation.

    Accept only full canonical rows from v0_belief_records.make_candidate. Rebuild
    their geometry on their own prefixes, validate both identity and continuation,
    then encode each candidate against a fixed public STOP/KEEP reference. Do not
    use a selected winner or an evaluator-provided reference. The separate public
    context features supplied to the head retain the absolute state information.

    Returns input-order ``candidate_ids`` and plain ``features`` (caller chooses
    tensor dtype/device), plus public context/reference/schema identities. No
    hypothesis inference, outcome/evaluator record, mass, label, private provenance
    or callback enters this path. Closed actor-context validation is inherited;
    the current belief-record owner restricts this adapter to opening contexts.
    Feature hashing is approximate, not a state/equivalence or sufficiency proof.
    """
    from farm_neat.economy.v0 import vocabulary as V
    from farm_neat.policy import v0_regret_comparator as comparator
    from farm_neat.training import v0_belief_records as records, v0_regret_records as R

    if type(candidates) is not list or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(f'candidates must contain one to {MAX_CANDIDATES} canonical rows')
    state, prefix, options = records._context(actor_context)
    menus = {}
    reference = records._candidate(actor_context, [V.STOP_ID], state, prefix, options, menus)
    checked = []
    for row in candidates:
        if type(row) is not dict or 'raw_actions' not in row:
            raise ValueError('candidate must be a complete canonical candidate row')
        expected = records._candidate(actor_context, row['raw_actions'], state, prefix, options, menus)
        if not R._equal(row, expected):
            raise ValueError('candidate identity/geometry must match its own prefix; extra fields are refused')
        checked.append(expected)
    keys = _candidate_catalog([c['candidate_sha256'] for c in checked], len(checked))
    encoded = comparator.encode_pairs([actor_context]*len(checked),
        [dict(learner=reference, challenger=c) for c in checked])
    return dict(feature_schema='farm-v0-belief-conditional-features-v1',
                comparator_feature_schema=comparator.FEATURE_SCHEMA,
                actor_context_sha256=R.actor_context_key(actor_context),
                reference_candidate_sha256=reference['candidate_sha256'],
                candidate_ids=keys, features=encoded['features'])


def conditional_targets_from_tensor(document: dict, *, expected_actor_sha256: str,
                                    hypothesis_ids: Sequence[str], candidate_ids: Sequence[str],
                                    device: str | torch.device = 'cpu') -> dict:
    """TRAINER ONLY: revalidate the existing outcome-tensor receipts, then align labels.

    Requires the independently saved pre-evaluation actor digest. Returns a bundle
    with ``targets`` (float64 ConditionalValueTargets) and private ``support`` keyed
    by candidate then hypothesis, preserving complete declared-family/group IDs,
    counts and missing-policy diagnostics from aggregate_tensor. Also retains the
    source tensor hash, actor hash, split and non-admission/zero-teacher-weight flags.
    The consumer must not pass this bundle into forward or rewrite frozen actor
    predictions with its labels. Source/conditioning and family-independence claims
    remain the existing evidence owner's declarations, NOT calibration certificates.

    Exact same ID sets are required, but their order is arbitrary. Absent/censored/
    unverified hypothesis evidence remains NaN/False. No downweighting, smoothing,
    mixture completion, proper-loss policy, selector or actor admission is added.
    This is a bounded offline bridge, NOT a per-token/per-step replay hot path.
    """
    from farm_neat.training import v0_belief_outcomes as outcomes

    if type(expected_actor_sha256) is not str or not _HASH.fullmatch(expected_actor_sha256):
        raise ValueError('expected_actor_sha256 must be an independently saved actor SHA256')
    # aggregate_tensor itself validates and rebuilds all source-bound derived cells.
    summary = outcomes.aggregate_tensor(document)
    if document['actor_payload_sha256'] != expected_actor_sha256:
        raise ValueError('outcome tensor does not match the expected actor snapshot')
    keys, _ = _catalog(hypothesis_ids, None, len(document['axes']['hypothesis_id']))
    candidates = _candidate_catalog(candidate_ids, len(document['axes']['candidate_sha256']))
    if (set(keys) != set(document['axes']['hypothesis_id'])
            or set(candidates) != set(document['axes']['candidate_sha256'])):
        raise ValueError('target identities must exactly match the authenticated outcome tensor axes')
    laws = torch.full((len(keys), len(candidates), 3), math.nan, dtype=torch.float64, device=device)
    mask = torch.zeros(laws.shape[:-1], dtype=torch.bool, device=device)
    support = {c: summary['candidates'][c]['per_hypothesis'] for c in candidates}
    for i, h in enumerate(keys):
        for j, c in enumerate(candidates):
            law = support[c][h]['wdl']
            if law is not None:
                laws[i, j] = laws.new_tensor(law)
                mask[i, j] = True
    return dict(targets=ConditionalValueTargets(keys, candidates, laws, mask),
                source_tensor_sha256=summary['tensor_sha256'],
                actor_payload_sha256=document['actor_payload_sha256'], split=document['split'],
                support=support, scope=summary['scope'], independence=summary['independence'],
                actor_training_admissible=False, teacher_weight=0.)
