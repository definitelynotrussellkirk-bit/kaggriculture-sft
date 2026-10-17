"""Trainer-only proper-score harness; no adopted auxiliary objective or actor wiring.

``belief_loss_terms`` reports THREE UNWEIGHTED losses: belief-mass log/Brier,
per-hypothesis categorical LOSS/DRAW/WIN proper loss over the accepted target carrier,
and unconditional mixture log/Brier. No combined loss, coefficient, selector,
regret gate, teacher admission or production update is enabled. Optional independent
aggregate inputs add a separately reported, UNWEIGHTED mixture-consistency penalty;
without them the existing three-term interface remains unchanged. Equality is a
coherence check, NOT a proper score against truth or a calibration certificate.

For a detached soft law q and forecast p, log = -sum(q log p); Brier is the
EXPECTED categorical score, ||p-q||^2 + 1-||q||^2, SUM over classes (not divided
by K). Thus scores match enumeration of hard outcomes, including irreducible
uncertainty. Both are minimized at q. A mixture score alone cannot identify the
mass or conditional laws: compensating errors can leave sum_h p_h P(Y|h,c)
unchanged. Per-hypothesis CE keeps the accepted mean-over-observed-pairs reduction;
it is NOT weighted by predicted mass, which could hide errors by starving a row.

Soft mass labels need no realized opponent class: a separate, frozen public
model can form q(h|o) proportional to q0(h)*L(public history through actor time|h).
Use the accepted belief_masses/public-update producer, retaining its source,
through-step, catalog and split evidence. Unknown/unidentifiable classes retain
soft mass; no MAP hardening, floor, hidden true-family lookup or hindsight actor
rewrite is performed here. A teacher posterior is a pseudo-target, NOT proof of
calibration. Detaching the current student's own prediction does not create an
independent target. Callers must separate target construction/training/evaluation
sources, bind saved actor identities, and keep monitor/final labels out of fitting.

Standalone minimum_mass_loss and decision_credit_loss are opt-in experiments.
They are NOT called by belief_loss_terms: the former keeps raw versus constrained
mass separate; the latter replaces, rather than duplicates, the existing admitted
class-path correction. Neither is a new competitive reward or a production default.

All targets are detached; exact shapes and identity joins forbid broadcasting,
class dropping or renormalizing observed subsets. Missing LABEL masks differ from
ACTOR-TIME conditional availability. A positive-mass missing conditional makes
that candidate's mixture unavailable. Excluded forecasts are counted explicitly;
compare losses only on matched coverage. No forecast/label is imputed as a draw.

The optional offline mixture-target bridge REUSES v0_belief_outcomes' validation,
deduplication and COMMON COMPLETE DECLARED source families. Its frozen-belief-
weighted target is a conditional simulation reference, not an independently
observed marginal or a calibrated true-family posterior. Family counts are
provenance, never loss weights or certified independent n. Replays/cache aliases
add neither precision nor weight. Arbitrary tensor inputs cannot prove source
independence, public-only feature provenance, chronology or corpus split hygiene.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor

from farm_neat.policy.v0_belief import (
    BeliefHeadOutput, ConditionalValueOutput, ConditionalValueTargets,
)

if TYPE_CHECKING:
    from farm_neat.training.v0_belief_credit import DecisionCreditRecord

SCHEMA = 'farm-v0-belief-proper-loss-v1'
_HASH = re.compile(r'[0-9a-f]{64}\Z')


def _floating(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f'{name} must be a floating tensor, never a record/dictionary')


def _mask(value: Tensor, shape: torch.Size, device: torch.device) -> None:
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise TypeError('mask must be a boolean tensor')
    if value.shape != shape:
        raise ValueError('mask shape must match exactly; no broadcasting')
    if value.device != device:
        raise ValueError('mask and predictions must share device')


def _target_shape(target: Tensor, reference: Tensor, mask: Tensor) -> None:
    _floating(target, 'target')
    if target.shape != reference.shape:
        raise ValueError('target shape must match predictions exactly; no broadcasting')
    if target.device != reference.device:
        raise ValueError('targets and predictions must share device')
    _mask(mask, target.shape[:-1], reference.device)


def _target_laws(target: Tensor, mask: Tensor) -> Tensor:
    # Select OUT unobserved NaN placeholders before validation or arithmetic.
    selected = target.detach()[mask].double()
    tolerance = max(1e-12, 8 * torch.finfo(target.dtype).eps)
    if (not bool(torch.isfinite(selected).all())
            or not bool(((selected >= 0) & (selected <= 1)).all())
            or not bool(((selected.sum(-1) - 1).abs() <= tolerance).all())):
        raise ValueError('observed targets must be nonnegative normalized categorical laws')
    return selected


def _log_laws(log_probabilities: Tensor, mask: Tensor) -> Tensor:
    """Validate/select supported normalized log laws without normalizing or detaching."""
    lp = log_probabilities[mask].double()
    tolerance = max(1e-12, 8 * torch.finfo(log_probabilities.dtype).eps)
    if (not bool((torch.isfinite(lp) | torch.isneginf(lp)).all())
            or not bool((lp <= tolerance).all())
            or not bool((lp.logsumexp(-1).abs() <= tolerance).all())):
        raise ValueError('observed log probabilities must describe normalized categorical laws')
    return lp


def _order(supplied: Sequence[str], reference: Sequence[str], device: torch.device) -> Tensor:
    if isinstance(supplied, (str, bytes)):
        raise TypeError('identities must be a sequence, not a string')
    keys = tuple(supplied)
    if (any(type(x) is not str for x in keys) or len(keys) != len(reference)
            or len(set(keys)) != len(keys) or set(keys) != set(reference)):
        raise ValueError('target/prediction identities must match exactly without duplicates')
    return torch.tensor([keys.index(key) for key in reference], dtype=torch.long, device=device)


def _mass(output: BeliefHeadOutput, mass_used: str) -> tuple[Tensor, Tensor]:
    if not isinstance(output, BeliefHeadOutput):
        raise TypeError('belief must be an accepted BeliefHeadOutput')
    if mass_used not in ('posterior_mass', 'constrained_mass'):
        raise ValueError('mass_used must name posterior_mass or constrained_mass explicitly')
    return ((output.posterior_log_mass, output.posterior_mass) if mass_used == 'posterior_mass'
            else (output.constrained_log_mass, output.constrained_mass))


def _number(value: Tensor) -> float | str:
    v = float(value.detach().cpu())
    return ('+inf' if v > 0 else '-inf') if math.isinf(v) else v


@dataclass(frozen=True)
class ProperLoss:
    """Mean of observed rows, not an independent-sample count or confidence bound.

    per_row is zero outside observed_mask solely for safe reduction. The mask and
    counts, not that placeholder, identify evidence. An empty loss is connected
    exact zero and status NO_EVIDENCE/NO_FORECAST, never a successful perfect fit.
    """
    loss: Tensor
    per_row: Tensor
    observed_mask: Tensor
    observed_rows: int
    total_rows: int
    requested_rows: int
    unavailable_rows: int
    score: str
    status: str

    def report(self) -> dict:
        return dict(loss=_number(self.loss), score=self.score, status=self.status,
                    observed_rows=self.observed_rows, total_rows=self.total_rows,
                    requested_rows=self.requested_rows, unavailable_rows=self.unavailable_rows,
                    reduction='MEAN_OBSERVED_ROWS_NOT_INDEPENDENT_SAMPLE_N')


def categorical_proper_loss(log_probabilities: Tensor, target_probabilities: Tensor,
                            observed_mask: Tensor, *, score: str = 'log') -> ProperLoss:
    """Expected proper score of normalized log probabilities [...,K], K >= 1.

    No clipping/epsilon or normalization repair. Exact zero probability (-inf log)
    yields +inf log loss for positive target mass; 0*log(0) contributes zero.
    Training callers should supply stable log_softmax/head logs, not log of a
    rounded probability: this loss cannot repair an upstream NaN Jacobian.
    Both forecasts and targets may have NaN placeholders on UNOBSERVED rows.
    """
    if score not in ('log', 'brier'):
        raise ValueError('score must be log or brier')
    _floating(log_probabilities, 'log_probabilities')
    if log_probabilities.ndim < 1 or log_probabilities.shape[-1] == 0:
        raise ValueError('probability shape must have a nonempty categorical axis')
    _target_shape(target_probabilities, log_probabilities, observed_mask)
    mask = observed_mask.detach().clone()
    q = _target_laws(target_probabilities, mask)
    lp = _log_laws(log_probabilities, mask)
    n = int(mask.sum())
    zero = log_probabilities.reshape(-1)[:0].double().sum()
    if score == 'log':
        # where BEFORE multiplication avoids 0 * -inf and keeps exact +inf limits.
        values = -(q * torch.where(q > 0, lp, torch.zeros_like(lp))).sum(-1)
    else:
        p = lp.exp()
        values = ((p - q).square() + q * (1 - q)).sum(-1)
    per_row = torch.zeros(mask.shape, dtype=torch.float64, device=lp.device) + zero
    per_row = per_row.masked_scatter(mask, values)
    return ProperLoss(values.mean() if n else zero, per_row, mask, n, mask.numel(), n, 0,
                      score, 'OBSERVED' if n else 'NO_EVIDENCE')


def belief_mass_loss(output: BeliefHeadOutput, target_mass: Tensor, observed_mask: Tensor, *,
                     hypothesis_ids: Sequence[str], mass_used: str = 'posterior_mass',
                     score: str = 'log') -> ProperLoss:
    """Score the full named mass vector; mask contexts, NEVER individual classes.

    Targets use supplied hypothesis_ids, and may be soft even if the hidden actual
    class is unknown. Missing catalog hypotheses refuse rather than renormalize.
    posterior_mass is the default; constrained_mass scoring is explicitly labeled.
    """
    lp, _ = _mass(output, mass_used)
    _target_shape(target_mass, lp, observed_mask)
    order = _order(hypothesis_ids, output.hypothesis_ids, lp.device)
    return categorical_proper_loss(lp, target_mass.detach().index_select(-1, order),
                                   observed_mask, score=score)


def conditional_wdl_loss(output: ConditionalValueOutput, targets: ConditionalValueTargets, *,
                         score: str = 'log') -> ProperLoss:
    """Score identity-aligned conditional laws on observed, predicted pairs only.

    ``targets.observed_mask`` is evaluator/trainer evidence. The output's
    ``prediction_mask`` is actor-time model support. They remain separate until
    this loss owner intersects them; unavailable requested rows are counted and
    never renormalized, imputed, or treated as successful zero loss.
    """
    if not isinstance(output, ConditionalValueOutput):
        raise TypeError('output must be an accepted ConditionalValueOutput')
    if not isinstance(targets, ConditionalValueTargets):
        raise TypeError('targets must be the accepted ConditionalValueTargets carrier')
    targets.__post_init__()  # Recheck mutable tensor contents.
    lp = output.log_probabilities
    _target_shape(targets.probabilities, lp, targets.observed_mask)
    _mask(output.prediction_mask, lp.shape[:-1], lp.device)
    hi = _order(targets.hypothesis_ids, output.hypothesis_ids, lp.device)
    ci = _order(targets.candidate_ids, output.candidate_ids, lp.device)
    target = targets.probabilities.index_select(-3, hi).index_select(-2, ci)
    requested = targets.observed_mask.index_select(-2, hi).index_select(-1, ci)
    valid = requested & output.prediction_mask
    result = categorical_proper_loss(lp, target, valid, score=score)
    unavailable = int((requested & ~output.prediction_mask).sum())
    status = ('PARTIAL_FORECAST' if result.observed_rows else 'NO_FORECAST') if unavailable else result.status
    return replace(result, requested_rows=int(requested.sum()),
                   unavailable_rows=unavailable, status=status)


@dataclass(frozen=True)
class MixtureWDL:
    """Differentiable complete marginals [...,C,3]; missing forecasts remain NaN."""
    candidate_ids: tuple[str, ...]
    log_probabilities: Tensor
    probabilities: Tensor
    complete_mask: Tensor
    missing_mass: Tensor
    mass_used: str


def mixture_wdl(belief: BeliefHeadOutput, conditional: ConditionalValueOutput, *,
                mass_used: str = 'posterior_mass') -> MixtureWDL:
    """Join hashes, then logsumexp_h(log mass_h + log P(WDL|h,c)).

    The conditional output's ``prediction_mask`` is the sole actor-time
    availability authority; callers cannot supply a second mask that can drift.
    It is not a target-observation mask. Same catalogs/batch/device; no broadcast.
    The accepted mass head has positive support including UNKNOWN.
    No known-subset renormalization, mean-outcome approximation or tail clipping.
    """
    mlp, masses = _mass(belief, mass_used)
    if not isinstance(conditional, ConditionalValueOutput):
        raise TypeError('conditional must be an accepted ConditionalValueOutput')
    lp = conditional.log_probabilities
    if mlp.shape[:-1] != lp.shape[:-3] or mlp.device != lp.device:
        raise ValueError('belief and conditional batch shape/device must match; no broadcasting')
    available_mask = conditional.prediction_mask
    _mask(available_mask, lp.shape[:-1], lp.device)
    hi = _order(belief.hypothesis_ids, conditional.hypothesis_ids, lp.device)
    kinds = dict(zip(belief.hypothesis_ids, belief.kinds))
    if tuple(kinds[h] for h in conditional.hypothesis_ids) != conditional.kinds:
        raise ValueError('hypothesis kinds must agree across both predictions')
    mlp, masses = mlp.index_select(-1, hi), masses.index_select(-1, hi)
    complete = available_mask.all(-2)
    missing = (masses.unsqueeze(-1) * (~available_mask)).sum(-2)
    joint = lp + mlp[..., :, None, None]
    # Select complete candidate rows BEFORE logsumexp: no all--inf reduction or
    # exp(NaN) backward graph on unavailable rows, including an all-missing batch.
    selected = joint.movedim(-3, -2)[complete].logsumexp(-2)
    zero = joint.reshape(-1)[:0].sum()
    shape = (*complete.shape, 3)
    logp = torch.full(shape, float('nan'), dtype=joint.dtype, device=joint.device) + zero
    probabilities = torch.full_like(logp, float('nan')) + zero
    logp[complete] = selected
    probabilities[complete] = selected.exp()
    return MixtureWDL(conditional.candidate_ids, logp, probabilities,
                      complete.detach().clone(), missing, mass_used)


def mixture_wdl_loss(output: MixtureWDL, target_wdl: Tensor, observed_mask: Tensor, *,
                     candidate_ids: Sequence[str], score: str = 'log') -> ProperLoss:
    """Score complete mixture forecasts against separately supplied marginal laws.

    Report requested-but-unavailable forecasts separately. These coverage changes
    cannot be treated as an improvement against controls scored on different rows.
    Validate EVERY observed target even when its corresponding forecast is missing.
    """
    if not isinstance(output, MixtureWDL):
        raise TypeError('output must be MixtureWDL')
    lp = output.log_probabilities
    _target_shape(target_wdl, lp, observed_mask)
    order = _order(candidate_ids, output.candidate_ids, lp.device)
    target = target_wdl.detach().index_select(-2, order)
    requested = observed_mask.index_select(-1, order)
    _target_laws(target, requested)
    valid = requested & output.complete_mask
    result = categorical_proper_loss(lp, target, valid, score=score)
    unavailable = int((requested & ~output.complete_mask).sum())
    status = ('PARTIAL_FORECAST' if result.observed_rows else 'NO_FORECAST') if unavailable else result.status
    return replace(result, requested_rows=int(requested.sum()), unavailable_rows=unavailable, status=status)


@dataclass(frozen=True)
class MixtureConsistency:
    """A label-free equality diagnostic/soft penalty, NEVER evidence of calibration.

    Axes follow ``projection.candidate_ids`` in LOSS/DRAW/WIN order. Residuals are
    aggregate minus weighted conditional probabilities. Missing rows remain NaN;
    per_candidate uses zero ONLY for reduction. Always inspect compared_mask and
    status. ``projection`` is the existing mixture_wdl output: a one-way hard
    replacement of the aggregate holding components fixed, not a joint nearest
    projection of all heads. It has no gradient path to the independent aggregate.
    """
    projection: MixtureWDL
    loss: Tensor
    residual: Tensor
    per_candidate: Tensor
    aggregate_prediction_mask: Tensor
    compared_mask: Tensor
    consistent_mask: Tensor
    unknown_mass: Tensor | None
    unknown_missing_mass: Tensor | None
    atol: float
    status: str

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return self.projection.candidate_ids

    def report(self) -> dict:
        n = int(self.compared_mask.sum())
        return dict(schema='farm-v0-belief-mixture-consistency-v1',
                    loss=_number(self.loss), status=self.status,
                    penalty='SQUARED_L2_PROBABILITY_RESIDUAL',
                    reduction='MEAN_COMPLETE_CANDIDATES_SUM_WDL_CLASSES',
                    residual_sign='AGGREGATE_MINUS_WEIGHTED_CONDITIONAL',
                    wdl_order=['LOSS', 'DRAW', 'WIN'], atol=self.atol,
                    max_absolute_residual=(_number(self.residual[self.compared_mask].abs().max())
                                           if n else None),
                    total_candidates=self.compared_mask.numel(), compared_candidates=n,
                    inconsistent_candidates=int((self.compared_mask & ~self.consistent_mask).sum()),
                    missing_conditionals=int((~self.projection.complete_mask).sum()),
                    missing_aggregates=int((~self.aggregate_prediction_mask).sum()),
                    mass_used=self.projection.mass_used,
                    unknown_represented=self.unknown_mass is not None,
                    catalog_scope='DECLARED_CATALOG_ONLY', calibration_status='UNMEASURED',
                    projection='REPLACE_AGGREGATE_ONLY_HOLD_COMPONENTS_FIXED',
                    weights_applied=False)

    def require_consistent(self) -> None:
        """Refuse inconsistent OR incomplete batches; no missing-mass tolerance.

        The scalar check tolerance never changes loss, masks, masses or predictions.
        Inspect residual/compared_mask by candidate ID for the full failure witness.
        """
        if self.status != 'CONSISTENT':
            raise ValueError(f'mixture equality not certified: {self.report()}')


def mixture_consistency_loss(belief: BeliefHeadOutput, conditional: ConditionalValueOutput,
                             aggregate_log_probabilities: Tensor,
                             aggregate_prediction_mask: Tensor, *,
                             candidate_ids: Sequence[str], mass_used: str = 'posterior_mass',
                             atol: float = 1e-8) -> MixtureConsistency:
    """Compare independent aggregate P(Y|c) to sum_h mass[h]*P(Y|h,c).

    The aggregate is a FORECAST, not a detached target. Both it and the component
    forecasts get gradients from mean_c sum_y residual[c,y]^2. No coefficient,
    automatic detach, label, production loss or hard projection is applied. Supply
    stable normalized log probabilities [...,C,3], not logits. Hash joins reorder
    candidates; all batch axes/devices must match exactly. Available aggregate
    laws are validated even when their matching mixture is unavailable.

    mixture_wdl remains the single aggregation owner. It requires identical named
    hypothesis catalogs and preserves every positive mass, including UNKNOWN.
    A missing conditional makes the mixture unavailable, however tiny its mass.
    No missing/excluded hypothesis is renormalized away. UNKNOWN needs an explicit
    residual predictor; absent UNKNOWN is not evidence that external mass is zero.
    Equality is conditional on the DECLARED catalog being the intended full model;
    tensors cannot certify exhaustive support in the real opponent population.

    ``result.require_consistent()`` is a strict whole-batch check. For a HARD
    parameterization, consume ``result.projection`` (or call mixture_wdl directly)
    instead of the aggregate. That fixes agreement by construction, not correctness:
    wrong components can worsen a previously calibrated aggregate. Compare proper
    scores on matched coverage using independent held-out target laws.
    """
    if type(atol) not in (int, float) or not math.isfinite(atol) or atol < 0:
        raise ValueError('atol must be a finite nonnegative number')
    projection = mixture_wdl(belief, conditional, mass_used=mass_used)
    lp = aggregate_log_probabilities
    _floating(lp, 'aggregate_log_probabilities')
    if lp.shape != projection.log_probabilities.shape or lp.device != projection.log_probabilities.device:
        raise ValueError('aggregate and mixture shape/device must match exactly; no broadcasting')
    _mask(aggregate_prediction_mask, lp.shape[:-1], lp.device)
    ci = _order(candidate_ids, projection.candidate_ids, lp.device)
    lp = lp.index_select(-2, ci)
    available = aggregate_prediction_mask.index_select(-1, ci).detach().clone()
    _log_laws(lp, available)
    _log_laws(projection.log_probabilities, projection.complete_mask)
    compared = available & projection.complete_mask
    # Select BEFORE exp/subtraction. NaN placeholders on missing rows never enter
    # the loss or its backward graph. Empty sums retain connected zero gradients.
    difference = lp[compared].double().exp() - projection.probabilities[compared]
    values = difference.square().sum(-1)
    zero = lp.reshape(-1)[:0].double().sum() + projection.log_probabilities.reshape(-1)[:0].sum()
    residual = torch.full_like(projection.probabilities, math.nan) + zero
    residual[compared] = difference
    per_candidate = torch.zeros_like(projection.missing_mass) + zero
    per_candidate[compared] = values
    consistent = torch.zeros_like(compared)
    consistent[compared] = difference.detach().abs().amax(-1) <= atol
    n = int(compared.sum())
    status = ('NO_FORECAST' if not n else 'INCONSISTENT' if bool((compared & ~consistent).any())
              else 'PARTIAL_FORECAST' if n != compared.numel() else 'CONSISTENT')
    unknown, unknown_missing = None, None
    if 'UNKNOWN' in belief.kinds:
        _, mass = _mass(belief, mass_used)
        i = belief.kinds.index('UNKNOWN')
        unknown = mass[..., i]
        hi = conditional.hypothesis_ids.index(belief.hypothesis_ids[i])
        unknown_missing = unknown.unsqueeze(-1) * (~conditional.prediction_mask[..., hi, :])
    return MixtureConsistency(projection, values.mean() if n else zero, residual, per_candidate,
                              available, compared, consistent, unknown, unknown_missing,
                              float(atol), status)


@dataclass(frozen=True)
class BeliefLossTerms:
    """Separate diagnostic/fit terms; the caller, not this module, owns weighting."""
    belief_mass: ProperLoss
    per_hypothesis_outcome: ProperLoss
    mixture_outcome: ProperLoss
    mass_used: str
    consistency: MixtureConsistency | None = None

    def report(self) -> dict:
        c = self.per_hypothesis_outcome
        return dict(schema=SCHEMA, weights_applied=False, mass_used=self.mass_used, terms=dict(
            belief_mass=self.belief_mass.report(),
            per_hypothesis_outcome=dict(loss=_number(c.loss), score='log', status=c.status,
                observed_pairs=c.observed_rows, total_pairs=c.total_rows,
                requested_pairs=c.requested_rows, unavailable_pairs=c.unavailable_rows,
                reduction='MEAN_OBSERVED_SUPPORTED_CONDITIONAL_PAIRS'),
            mixture_outcome=self.mixture_outcome.report(),
            consistency=(self.consistency.report() if self.consistency is not None
                         else dict(loss=None, status='NOT_ENABLED'))))


def belief_loss_terms(belief: BeliefHeadOutput, conditional: ConditionalValueOutput, *,
                      target_mass: Tensor, mass_observed_mask: Tensor,
                      conditional_targets: ConditionalValueTargets,
                      target_mixture_wdl: Tensor, mixture_observed_mask: Tensor,
                      hypothesis_ids: Sequence[str], candidate_ids: Sequence[str],
                      mass_used: str = 'posterior_mass',
                      mass_score: str = 'log', mixture_score: str = 'log',
                      aggregate_log_probabilities: Tensor | None = None,
                      aggregate_prediction_mask: Tensor | None = None,
                      aggregate_candidate_ids: Sequence[str] | None = None,
                      consistency_atol: float = 1e-8) -> BeliefLossTerms:
    """Score three sources separately; do not synthesize a target from predictions.

    Mass/mixture targets share the supplied target ID order. Conditional targets
    use the accepted identity-bearing carrier and may come from
    ``conditional_targets_from_tensor``; the marginal adapter below uses
    common complete families rather than mixing unmatched per-cell means. These
    are trainer-only tensors, never arguments to a public actor forward method.

    Gradients: mass term -> belief; conditional CE -> conditional head; mixture
    term -> BOTH. An optional shared trunk/isolation choice remains caller-owned.
    No implicit detach of predicted mixture weights and no teacher-private labels
    used as actor features. All-empty terms retain their own no-evidence status.
    Supplying ALL three aggregate_* arguments opts into a separate consistency term
    with gradients to aggregate, mass and conditional predictions. No sum or weight
    is adopted, and all three original losses and default reporting are unchanged.
    """
    supplied = (aggregate_log_probabilities is not None, aggregate_prediction_mask is not None,
                aggregate_candidate_ids is not None)
    if any(supplied) and not all(supplied):
        raise ValueError('aggregate log probabilities, prediction mask and candidate IDs are all required')
    hs, cs = tuple(hypothesis_ids), tuple(candidate_ids)
    mass = belief_mass_loss(belief, target_mass, mass_observed_mask,
                           hypothesis_ids=hs, mass_used=mass_used, score=mass_score)
    per_h = conditional_wdl_loss(conditional, conditional_targets)
    mixed = mixture_wdl(belief, conditional, mass_used=mass_used)
    marginal = mixture_wdl_loss(mixed, target_mixture_wdl, mixture_observed_mask,
                                candidate_ids=cs, score=mixture_score)
    consistency = (mixture_consistency_loss(belief, conditional, aggregate_log_probabilities,
        aggregate_prediction_mask, candidate_ids=aggregate_candidate_ids, mass_used=mass_used,
        atol=consistency_atol) if all(supplied) else None)
    return BeliefLossTerms(mass, per_h, marginal, mass_used, consistency)


@dataclass(frozen=True)
class MixtureWDLTargets:
    """Offline reference, not calibrated ground truth or permission to fit a split."""
    candidate_ids: tuple[str, ...]
    wdl: Tensor
    observed_mask: Tensor
    declared_family_counts: Tensor
    actor_payload_sha256: str
    tensor_sha256: str
    source_split: str | None
    target_semantics: str = 'FROZEN_BELIEF_WEIGHTED_COMMON_FAMILY_REFERENCE'


def mixture_targets_from_outcomes(document: dict, *, expected_actor_sha256: str,
                                  candidate_ids: Sequence[str], dtype: torch.dtype = torch.float64,
                                  device: torch.device | str | None = None) -> MixtureWDLTargets:
    """Reuse validated matched-source aggregation; no new grouping or imputation.

    Equal weight over COMPLETE DECLARED families, equal groups within family,
    exact saved actor mass within group. The current student's mass NEVER builds
    its own label. The source controller bindings do not reveal the actual hidden
    family. Unlike per-hypothesis targets, every live hypothesis must have coverage
    on the SAME complete families for a marginal reference to be available.
    All missing/censored/unverified outcomes stay NaN+false; count is descriptive.
    The accepted adapter verifies receipt/cache/replay joins; IDs cannot certify
    statistical independence. Keep this offline document away from actor inputs.
    """
    from farm_neat.training import v0_belief_outcomes as O

    if type(expected_actor_sha256) is not str or not _HASH.fullmatch(expected_actor_sha256):
        raise ValueError('a separately saved actor payload SHA256 is required')
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError('target dtype must be floating')
    summary = O.aggregate_tensor(document)  # Full source/receipt/derived-field validation.
    if document['actor_payload_sha256'] != expected_actor_sha256:
        raise ValueError('outcome tensor actor payload differs from the saved actor identity')
    if isinstance(candidate_ids, (str, bytes)):
        raise TypeError('candidate identities must be a sequence')
    cs = tuple(candidate_ids)
    _order(cs, document['axes']['candidate_sha256'], torch.device('cpu'))
    laws = torch.full((len(cs), 3), float('nan'), dtype=dtype, device=device)
    mask = torch.zeros(len(cs), dtype=torch.bool, device=device)
    counts = torch.zeros(len(cs), dtype=torch.long, device=device)
    for i, c in enumerate(cs):
        stats = summary['candidates'][c]['mixture']
        if stats['wdl'] is not None:
            laws[i] = torch.tensor(stats['wdl'], dtype=dtype, device=device)
            mask[i] = True
            counts[i] = stats['independent_group_n']  # Historical name; DECLARED families only.
    return MixtureWDLTargets(cs, laws, mask, counts, expected_actor_sha256,
                             summary['tensor_sha256'], document['split'])


@dataclass(frozen=True)
class PredictionDiversityLoss:
    """Unweighted finite-panel diagnostic; not calibration or a slot-count target.

    Zero placeholders outside observed_mask are excluded from ALL reductions.
    effective_k is exp(mean_probe I(H; outcome | probe)), not exp(H(mass)),
    the number of IDs, or a global identification of opponent types. This
    one-outcome information-K is at most the outcome count; it is not joint
    information across probes or the number of distinct full profiles. Missing
    protected probes remain unavailable, not evidence of equivalence.
    """
    loss: Tensor
    per_probe: Tensor
    information_per_probe: Tensor
    observed_mask: Tensor
    live_hypotheses: int
    observed_probes: int
    total_probes: int
    method: str
    status: str

    @property
    def unavailable_probes(self) -> int:
        return self.total_probes - self.observed_probes

    @property
    def effective_k(self) -> Tensor | None:
        return (self.information_per_probe[self.observed_mask].mean().exp()
                if self.observed_probes else None)

    def report(self) -> dict:
        k = self.effective_k
        return dict(schema='farm-v0-prediction-diversity-loss-v1', method=self.method,
            loss=_number(self.loss), status=self.status,
            effective_k=_number(k) if k is not None else None,
            live_hypotheses=self.live_hypotheses, observed_probes=self.observed_probes,
            total_probes=self.total_probes, unavailable_probes=self.unavailable_probes,
            coverage_semantics='COMMON_COMPLETE_POSITIVE_MASS_PROBES',
            reduction=('MEAN_COMMON_PROBES_FIXED_MASS_ORDERED_PAIR_SUM'
                       if self.method in ('pairwise_repulsion', 'evidence_coverage')
                       else 'MEAN_COMMON_PROBES'),
            weights_applied=False, production_enabled=False,
            mass_detached=True, reference_detached=True,
            provenance_or_independence_verified=False)


def _prediction_entropy(probabilities: Tensor) -> Tensor:
    # Select BEFORE log: exact zero entries have zero contribution and no NaN
    # Jacobian. Supply stable log_softmax logs upstream, as for proper scores.
    safe = torch.where(probabilities > 0, probabilities, torch.ones_like(probabilities))
    return -(probabilities * safe.log()).sum(-1)


def _prediction_pair_js(probabilities: Tensor) -> Tensor:
    """Per-probe JS in nats for [probe, hypothesis, outcome] laws."""
    entropy = _prediction_entropy(probabilities)
    middle = (probabilities.unsqueeze(-2) + probabilities.unsqueeze(-3)) / 2
    return (_prediction_entropy(middle)
            - (entropy.unsqueeze(-1) + entropy.unsqueeze(-2)) / 2).clamp_min(0)


def _prediction_axis(ids: Sequence[str], size: int, name: str) -> tuple[str, ...]:
    if isinstance(ids, (str, bytes)):
        raise TypeError(f'{name} must be a sequence of identities, not a string')
    keys = tuple(ids)
    if (len(keys) != size or any(type(x) is not str or not x.strip() for x in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError(f'{name} must name every axis entry once with a nonempty identity')
    return keys


def prediction_diversity_loss(log_probabilities: Tensor, mass: Tensor,
                             prediction_mask: Tensor, *, hypothesis_ids: Sequence[str],
                             probe_ids: Sequence[str], method: str = 'none',
                             temperature: float = 0.1,
                             reference_probabilities: Tensor | None = None,
                             reference_prediction_mask: Tensor | None = None,
                             reference_hypothesis_ids: Sequence[str] | None = None,
                             reference_probe_ids: Sequence[str] | None = None,
                             separation_margin: float = 0.0) -> PredictionDiversityLoss:
    """Opt-in observable-prediction penalties; NO call from belief_loss_terms.

    Inputs: one panel of normalized log laws [H, P, O], fixed normalized mass
    [H], and actor-time forecast availability [H, P]. Probe identities must
    bind the SAME public context/prefix/horizon/intervention and outcome axis
    for each hypothesis. Do not pool contexts or use arbitrary embeddings.
    Caller owns public provenance, legal probes and independent reference
    construction. This tensor API cannot establish any of those premises.

    Methods, reported separately BEFORE a caller chooses a coefficient:
      none: connected exact zero (default).
      pairwise_repulsion: sum_{i!=j} m_i*m_j*exp(-JS(p_i,p_j)/temperature).
      set_information: -I(H;outcome|probe), using ALL active components jointly.
      evidence_coverage: sum_{i!=j} m_i*m_j*relu(JS(q_i,q_j)-margin-JS(p_i,p_j))^2.
    Pair weights are NOT renormalized: dominant mass reduces rare-pair pressure.
    Pairwise penalties depend on slot decomposition; set information and its
    effective-K statistic are invariant to splitting exact duplicates' mass.
    Units differ (similarity, nats, squared nats); coefficients are not comparable
    across methods. Temperature/margin are experimental, not calibrated defaults.

    Unconditional repulsion can invent false distinctions. Coverage only pushes
    separations supported by an independently constructed, DETACHED reference q;
    a true single-mode reference gives zero pressure. It does not constrain excess
    separation or fix a wrong reference. Independent conditional proper scores
    remain necessary; mixture fit alone cannot identify component predictions.
    Mass is detached to prevent reducing the loss by starving/reshuffling slots.
    Reference IDs are joined exactly and masks keep missing labels separate.
    All positive-mass hypotheses must have coverage on the SAME probe, with no
    observed-subset mass renormalization. Compare methods on matched coverage.

    Exact duplicates are stationary under these smooth permutation-invariant
    losses. A higher duplicate loss is NOT a symmetry-breaking mechanism: this
    function adds no ID-based jitter or latent-identity supervision. Runtime is
    O(P*H^2*O), diagnostic only, not profiled production training throughput.
    """
    methods = ('none', 'pairwise_repulsion', 'evidence_coverage', 'set_information')
    if method not in methods:
        raise ValueError(f'method must be one of {methods}')
    for value, name in ((temperature, 'temperature'), (separation_margin, 'separation_margin')):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
    if temperature <= 0 or separation_margin < 0:
        raise ValueError('temperature must be positive and separation_margin nonnegative')
    _floating(log_probabilities, 'log_probabilities')
    if log_probabilities.ndim != 3 or any(size == 0 for size in log_probabilities.shape):
        raise ValueError('log_probabilities must have nonempty [hypothesis, probe, outcome] axes')
    h, probes, _ = log_probabilities.shape
    hs = _prediction_axis(hypothesis_ids, h, 'hypothesis_ids')
    ps = _prediction_axis(probe_ids, probes, 'probe_ids')
    _mask(prediction_mask, log_probabilities.shape[:-1], log_probabilities.device)
    _floating(mass, 'mass')
    if mass.shape != (h,) or mass.device != log_probabilities.device:
        raise ValueError('mass must match the hypothesis axis and prediction device exactly')
    # A scalar boolean selection adds a row axis; restore the declared [H] shape.
    weights = _target_laws(mass, torch.tensor(True, device=mass.device)).reshape(h)
    live = weights > 0
    weights = weights[live]  # No renormalization; zeros are the ONLY dropped mass.
    active = int(live.sum())
    _log_laws(log_probabilities[live], prediction_mask[live])
    observed = prediction_mask[live].all(0).detach().clone()

    supplied = (reference_probabilities is not None, reference_prediction_mask is not None,
                reference_hypothesis_ids is not None, reference_probe_ids is not None)
    reference = None
    if method == 'evidence_coverage':
        if not all(supplied):
            raise ValueError('coverage requires reference laws, mask, hypothesis IDs and probe IDs')
        _target_shape(reference_probabilities, log_probabilities, reference_prediction_mask)
        rh = _order(reference_hypothesis_ids, hs, mass.device)
        rp = _order(reference_probe_ids, ps, mass.device)
        reference = reference_probabilities.detach().index_select(0, rh).index_select(1, rp)
        ref_mask = reference_prediction_mask.detach().index_select(0, rh).index_select(1, rp)
        _target_laws(reference[live], ref_mask[live])
        observed &= ref_mask[live].all(0)
    elif any(supplied):
        raise ValueError('reference inputs are only consumed by evidence_coverage')

    count = int(observed.sum())
    zero = log_probabilities.reshape(-1)[:0].double().sum()
    per_probe = torch.zeros(probes, dtype=torch.float64, device=mass.device) + zero
    information = torch.zeros_like(per_probe) + zero
    if count:
        p = log_probabilities[live][:, observed].double().exp().permute(1, 0, 2)
        mixture = (weights[None, :, None] * p).sum(1)
        info = (_prediction_entropy(mixture)
                - (weights[None, :] * _prediction_entropy(p)).sum(1)).clamp_min(0)
        information = information.masked_scatter(observed, info)
        values = torch.zeros_like(info) + zero
        if active > 1 and method != 'none':
            if method == 'set_information':
                values = -info
            else:
                js = _prediction_pair_js(p)
                pair_weights = weights[:, None] * weights[None, :]
                pair_weights = pair_weights * (~torch.eye(active, dtype=torch.bool, device=mass.device))
                if method == 'pairwise_repulsion':
                    penalties = (-js / temperature).exp()
                else:
                    q = reference[live][:, observed].double().permute(1, 0, 2)
                    supported = (_prediction_pair_js(q) - separation_margin).clamp_min(0)
                    penalties = (supported - js).clamp_min(0).square()
                values = (pair_weights * penalties).sum((-2, -1))
        per_probe = per_probe.masked_scatter(observed, values)
    status = 'NO_COMMON_EVIDENCE' if not count else ('NO_PAIRS' if active < 2 else 'OBSERVED')
    return PredictionDiversityLoss(per_probe[observed].mean() if count else zero,
        per_probe, information, observed, active, count, probes, method, status)


# This is a conservative numerical conditioning guard, NOT a calibrated choice of
# hypothesis coverage. K*Y close to one removes the evidence-dependent part of the
# diagnostic uniform mixture. Exact feasibility is checked before float division.
MINIMUM_MASS_MIN_SLACK = 1e-6


@dataclass(frozen=True)
class MinimumMassLoss:
    """Raw soft-floor pressure and a SEPARATE, output-only constrained view.

    The constrained view is not a target, a calibration score, or feedback into
    the public posterior. It is produced by the existing mass-head owner. The
    soft penalty does not guarantee the floor: only the diagnostic view does.
    """
    loss: Tensor
    per_context: Tensor
    raw_mass: Tensor
    constrained_mass: Tensor
    hypothesis_ids: tuple[str, ...]
    k: int
    y: float
    method: str
    evidence_slack: float

    def report(self) -> dict:
        return dict(schema='farm-v0-belief-minimum-mass-loss-v1', loss=_number(self.loss),
                    method=self.method, k=self.k, y=self.y, evidence_slack=self.evidence_slack,
                    minimum_evidence_slack=MINIMUM_MASS_MIN_SLACK,
                    reduction='MEAN_CONTEXTS_SUM_HYPOTHESIS_SQUARED_SHORTFALL',
                    raw_mass=self.raw_mass.detach().cpu().tolist(),
                    constrained_mass=self.constrained_mass.detach().cpu().tolist(),
                    hypothesis_ids=list(self.hypothesis_ids),
                    constrained_transform='none' if self.method == 'none' else 'uniform_mix',
                    constraint_is_diagnostic_only=True, weights_applied=False,
                    production_enabled=False)


def minimum_mass_loss(output: BeliefHeadOutput, *, k: int, y: float = 0.,
                      method: str = 'none') -> MinimumMassLoss:
    """Opt-in mean_b sum_h relu(Y - RAW posterior[b,h])**2; default exact zero.

    K means the ENTIRE named catalog, not its top-K subset, effective-K, or proof
    of distinctness. Reject K mismatches, exact K*Y >= 1 and evidence slack below
    1e-6. The slack rule is an explicit conservative conditioning limit, not a
    universal statistical stability threshold. A caller must select Y on separate
    evidence; a feasible floor can still badly distort dominant evidence.

    No in-place changes, prior modification, forecast scoring, or actor wiring.
    The returned constrained_mass is an output-only uniform mixture from the
    existing belief_masses implementation, never substituted into proper scores.
    A pre-existing head floor is ignored: both pressure and view start at RAW mass.
    """
    from farm_neat.policy.v0_belief import belief_masses

    raw_log, raw = _mass(output, 'posterior_mass')
    if type(k) is not int or k < 1 or k != len(output.hypothesis_ids):
        raise ValueError('k must be the positive integer size of the entire named catalog')
    if type(y) not in (int, float) or not math.isfinite(y) or y < 0:
        raise ValueError('y must be a finite nonnegative number, never bool')
    y = float(y)
    num, den = y.as_integer_ratio()
    remainder = den - k*num
    if remainder <= 0:
        raise ValueError('infeasible or degenerate minimum mass: exact K*Y must be < 1')
    slack = remainder / den
    if slack < MINIMUM_MASS_MIN_SLACK:
        raise ValueError('unstable minimum mass: evidence slack must be at least 1e-6')
    if method not in ('none', 'soft_floor'):
        raise ValueError('minimum-mass method must be none or soft_floor')
    if method == 'none' and y != 0:
        raise ValueError('none requires y=0; do not hide a floor behind a disabled method')
    # Recheck the public carrier's mutable tensors, not just its dataclass type.
    _floating(raw, 'posterior_mass')
    if raw.shape != raw_log.shape or raw.shape[-1] != k or raw.device != raw_log.device:
        raise ValueError('posterior mass/log mass shape, device and catalog must agree')
    mask = torch.ones(raw.shape[:-1], dtype=torch.bool, device=raw.device)
    _target_laws(raw, mask)
    logs = _log_laws(raw_log, mask)
    if not torch.allclose(logs.exp(), raw[mask].double(), rtol=1e-10, atol=1e-14):
        raise ValueError('posterior mass and log mass disagree')
    zero = raw.reshape(-1)[:0].double().sum()
    if method == 'none':
        values = torch.zeros_like(raw[..., 0], dtype=torch.float64) + zero
        constrained = raw
    else:
        values = (y - raw.double()).clamp_min(0).square().sum(-1)
        constrained = belief_masses(raw_log, hypothesis_ids=output.hypothesis_ids,
            kinds=output.kinds, p_min=y, floor_transform='uniform_mix').constrained_mass
    return MinimumMassLoss(values.mean() if values.numel() else zero, values, raw,
                           constrained, output.hypothesis_ids, k, y, method, slack)


@dataclass(frozen=True)
class DecisionCreditLoss:
    """ONE replacement correction term, not an extra copy of the base BC loss.

    base_weights are the unchanged admission-only control; weights are the chosen
    detached attenuation. The loss still uses the original class-path teacher
    objective and divides by registered opportunities, not admitted rows. Report
    a separately computed base correction and (loss - base) when composing an
    additive ablation; NEVER add base + loss and double-count correction.
    """
    loss: Tensor
    base_weights: Tensor
    weights: Tensor
    confidence: Tensor
    distribution_support: Tensor
    credit_eligible: Tensor
    weighting: str
    mode: str
    opportunity_count: int
    confidence_source_sha256: str
    credit_sha256s: tuple[str, ...]

    def report(self) -> dict:
        return dict(schema='farm-v0-belief-credit-loss-v1', loss=_number(self.loss),
                    objective='TERMINAL_WDL', weighting=self.weighting, mode=self.mode,
                    base_weights=self.base_weights.detach().cpu().tolist(),
                    weights=self.weights.detach().cpu().tolist(),
                    confidence=self.confidence.detach().cpu().tolist(),
                    distribution_support=self.distribution_support.detach().cpu().tolist(),
                    credit_eligible=self.credit_eligible.detach().cpu().tolist(),
                    confidence_source_sha256=self.confidence_source_sha256,
                    confidence_is_calibrated=False, credit_sha256s=list(self.credit_sha256s),
                    opportunity_count=self.opportunity_count,
                    reduction='SUM_WEIGHTED_CLASS_PATH_TERMS_OVER_REGISTERED_OPPORTUNITIES',
                    usage='REPLACE_BASE_CORRECTION_NOT_ADD_TO_IT',
                    teacher_scope='OPPONENT_CONDITIONED',
                    weight_inputs_detached=True, variance_or_tail_penalty=False,
                    scalar_coefficient_applied=False, production_enabled=False)


def decision_credit_loss(current_winner_logp: Tensor, current_learner_logp: Tensor,
                         reference_winner_logp: Tensor, reference_learner_logp: Tensor, *,
                         records: Sequence['DecisionCreditRecord'],
                         expected_actor_sha256s: Sequence[str],
                         candidate_pairs: Sequence[tuple[str, str]],
                         confidence: Tensor, confidence_source_sha256: str,
                         weighting: str = 'none', mode: str = 'gated_bc', beta: float = .1,
                         opportunity_count: int) -> DecisionCreditLoss:
    """Offline trainer-only credit seam, reusing the existing teacher_loss owner.

    Each typed immutable DecisionCreditRecord was already rebuilt/validated by its
    owning constructor. Join its independently saved actor digest and ordered
    (learner, challenger) candidate digests to the path rows. Callers still own
    exact prefix class-path scoring, frozen references, chronology and global B.
    Do not pass records, opponent identities, future outcomes, or confidence to
    actor forward methods. This boundary performs Python record work, not a
    production GPU hot-path kernel.

    none: EXACT original admission-only correction (adverse/no-op control).
    binary_admitted: forward original admission weight only if the credit owner
      marked it eligible and frozen confidence > 0. No magnitude reweighting.
    confidence: binary_admitted weight times externally frozen confidence [0,1].
    distribution: confidence weight times the frozen mass of hypotheses with
      positive ex-ante expected terminal-WDL gain. This is a bounded attenuation
      experiment, NOT an unbiased policy gradient or a new risk/wealth utility.

    All enabled modes refuse teaching from unresolved, negative ex-ante, unbound,
    held-out or unadmitted records. They never invent precision from entropy,
    effective-n, variance, tails or one realized lucky result. The external
    confidence/source pin is explicit but cannot certify calibration or source
    honesty. Zero confidence masks out active path arithmetic in all enabled
    modes. The admission-only `none` deliberately ignores credit and confidence;
    it exists to preserve the old recipe, not to claim those new safeguards.

    A record's teacher scope stays opponent-conditioned even for distribution
    weighting. Positive hypothesis support is not a transfer certificate.
    Neither the fixed-N admission weight nor a target law is increased or changed.
    """
    from farm_neat.training.v0_belief_credit import DecisionCreditRecord
    from farm_neat.training.v0_regret_loss import teacher_loss

    if weighting not in ('none', 'binary_admitted', 'confidence', 'distribution'):
        raise ValueError('unknown decision-credit weighting')
    _floating(current_winner_logp, 'current_winner_logp')
    if current_winner_logp.ndim != 1:
        raise ValueError('class-path likelihoods must be one-dimensional opportunity rows')
    n = current_winner_logp.numel()
    collections = (records, expected_actor_sha256s, candidate_pairs)
    if any(isinstance(v, (str, bytes)) or not isinstance(v, Sequence) or len(v) != n
           for v in collections):
        raise ValueError('one credit, saved actor digest and candidate pair per opportunity required')
    _floating(confidence, 'confidence')
    if (confidence.shape != current_winner_logp.shape
            or confidence.device != current_winner_logp.device
            or confidence.dtype != current_winner_logp.dtype):
        raise ValueError('confidence must match path shape/device/dtype exactly; no broadcasting')
    frozen_confidence = confidence.detach().clone()
    if not bool((torch.isfinite(frozen_confidence)
                 & (frozen_confidence >= 0) & (frozen_confidence <= 1)).all()):
        raise ValueError('frozen confidence must be finite in [0,1]')
    if type(confidence_source_sha256) is not str or not _HASH.fullmatch(confidence_source_sha256):
        raise ValueError('a frozen confidence source SHA256 is required')
    base_values, eligible_values, supports, hashes = [], [], [], []
    for record, anchor, pair in zip(records, expected_actor_sha256s, candidate_pairs):
        if not isinstance(record, DecisionCreditRecord):
            raise TypeError('records must be validated immutable DecisionCreditRecord values')
        d = record.to_dict()
        if (type(anchor) is not str or not _HASH.fullmatch(anchor)
                or d['actor_payload_sha256'] != anchor):
            raise ValueError('credit does not match separately saved actor identity')
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or tuple(pair) != (d['learner_candidate_sha256'], d['challenger_candidate_sha256'])
                or pair[0] == pair[1]):
            raise ValueError('credit/path learner and challenger identities must match exactly')
        base = float(d['admission']['teacher_weight'])
        eligible = (d['status'] == 'RESOLVED' and d['actor_training_admissible']
                    and d['teacher_weight'] > 0 and d['expected_wdl_delta'] is not None
                    and d['expected_wdl_delta'] > 0)
        # Do not import realized wealth, paired variance/tails, concentration or
        # outcome labels into this factor. The record owner already owns admission.
        support = math.fsum(v['mass'] for v in d['forecast']['per_hypothesis'].values()
                            if v['expected_wdl_delta'] is not None and v['expected_wdl_delta'] > 0)
        base_values.append(base)
        eligible_values.append(eligible)
        supports.append(min(1., max(0., support)))  # only remove sum-roundoff
        hashes.append(d['content_sha256'])
    base_weights = current_winner_logp.new_tensor(base_values)
    distribution_support = current_winner_logp.new_tensor(supports)
    eligible = torch.tensor(eligible_values, dtype=torch.bool, device=current_winner_logp.device)
    if weighting == 'none':
        weights = base_weights.clone()
    else:
        weights = base_weights * (eligible & (frozen_confidence > 0))
        if weighting in ('confidence', 'distribution'):
            weights = weights * frozen_confidence
        if weighting == 'distribution':
            weights = weights * distribution_support
    # teacher_loss is the sole owner of BC/pairwise arithmetic, references, exact
    # empty-zero handling and registered-opportunity normalization. Supplying
    # 2*w uses its original bounded-weight parameterization, without relaxing it.
    loss = teacher_loss(current_winner_logp, current_learner_logp,
        reference_winner_logp, reference_learner_logp, weights > 0, 2*weights,
        mode=mode, beta=beta, opportunity_count=opportunity_count)
    return DecisionCreditLoss(loss, base_weights, weights, frozen_confidence,
        distribution_support, eligible, weighting, mode, opportunity_count,
        confidence_source_sha256, tuple(hashes))
