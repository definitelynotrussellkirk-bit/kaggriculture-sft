"""Separate, frozen-gate V0 teacher objectives; never an RL policy baseline.

``gated_bc`` is the simple control. ``gated_pair`` is reference-centered
softplus preference teaching, not a KL constraint or an unbiased policy gradient.
Both divide by registered PAIR OPPORTUNITIES, including unadmitted rows. For
microbatches pass the same global opportunity_count and SUM the returned losses;
or weight each local mean by local_count / global_count. Never average tokens.

Only admitted positive-weight entries participate in arithmetic. An empty slice
sum anchors exact zero to both actor inputs, even when excluded entries are NaN.
The caller must also avoid constructing undefined upstream model operations on
excluded paths: a loss cannot repair a NaN Jacobian already built by its caller.
Zero teacher gradients do not disable optimizer momentum, weight decay or RL.

Reference, gate and lower-bound inputs are always detached. The caller owns the
frozen parent identity and the admission verdict; policy likelihood preference
is NOT evidence of terminal-WDL superiority. Never invent a behavior logp for a
search winner. This module neither executes games nor loads/saves checkpoints.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

MODES = ('gated_bc', 'gated_pair')


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all()), 'finite active ' + name + ' required')


def _selection(current_winner_logp, current_learner_logp, reference_winner_logp,
               reference_learner_logp, admitted, lower_bound, opportunity_count):
    """Validate shape first; inspect numerical values only after frozen selection."""
    xs = (current_winner_logp, current_learner_logp, reference_winner_logp,
          reference_learner_logp, lower_bound)
    _require(all(isinstance(x, torch.Tensor) and x.ndim == 1 and x.is_floating_point()
                 for x in xs), 'aligned one-dimensional floating tensors required')
    first = xs[0]
    _require(all(x.shape == first.shape and x.device == first.device and x.dtype == first.dtype
                 for x in xs), 'aligned shape, dtype and device required')
    _require(isinstance(admitted, torch.Tensor) and admitted.shape == first.shape
             and admitted.device == first.device
             and (admitted.dtype == torch.bool or admitted.is_floating_point()),
             'aligned boolean or exact binary floating admitted gate required')
    gate = admitted.detach()
    _require(bool(((gate == 0) | (gate == 1)).all()), 'binary admitted gate required')
    _require(type(opportunity_count) is int and 0 <= opportunity_count <= 2**63 - 1
             and opportunity_count >= first.numel(),
             'opportunity_count must be an integer >= supplied opportunities (zero only for empty batches)')
    indices = torch.nonzero(gate == 1, as_tuple=True)[0]
    bounds = lower_bound.detach()[indices]
    _finite(bounds, 'lower_bound')
    # Select before clamping/weighting or touching any actor/reference logp.
    positive = bounds > 0
    weights = bounds[positive].clamp(max=2) / 2
    # A positive subnormal bound can round to zero when scaled; exclude that
    # numerical zero before reading path logps, just like an exact zero bound.
    nonzero = weights > 0
    return indices[positive][nonzero], weights[nonzero]


def _actor_zero(winner: torch.Tensor, learner: torch.Tensor) -> torch.Tensor:
    # Unlike (x * 0).sum(), this never reads excluded NaN/Inf values.
    return winner[:0].sum() + learner[:0].sum()


def teacher_loss(current_winner_logp: torch.Tensor, current_learner_logp: torch.Tensor,
                 reference_winner_logp: torch.Tensor, reference_learner_logp: torch.Tensor,
                 admitted: torch.Tensor, lower_bound: torch.Tensor, *, mode: str,
                 beta: float, opportunity_count: int) -> torch.Tensor:
    """Return a scalar sum / B; weights = stop_gradient(clip(lower_bound, 0, 2)/2).

    Inputs are aligned vectors of complete class-path log likelihoods (not token
    means). B may exceed vector length when it is the accumulation-wide count.
    Unadmitted/zero-weight values may be nonfinite; active required values fail
    closed. BC does not evaluate unused learner or reference values. No gradient
    ever reaches reference, gate or lower-bound tensors. beta is an explicit,
    finite positive Python scalar even for the BC control where it is unused.
    """
    _require(isinstance(mode, str) and mode in MODES, 'mode must be gated_bc or gated_pair')
    _require(type(beta) in (int, float) and math.isfinite(beta) and beta > 0,
             'finite positive beta required')
    indices, weights = _selection(current_winner_logp, current_learner_logp,
        reference_winner_logp, reference_learner_logp, admitted, lower_bound, opportunity_count)
    zero = _actor_zero(current_winner_logp, current_learner_logp)
    if indices.numel() == 0:
        return zero
    winner = current_winner_logp[indices]
    _finite(winner, 'winner logp')
    if mode == 'gated_bc':
        terms = -winner
    else:
        learner = current_learner_logp[indices]
        ref_winner = reference_winner_logp.detach()[indices]
        ref_learner = reference_learner_logp.detach()[indices]
        for value, name in ((learner, 'learner logp'), (ref_winner, 'reference winner logp'),
                            (ref_learner, 'reference learner logp')):
            _finite(value, name)
        contrast = beta * ((winner - learner) - (ref_winner - ref_learner))
        _finite(contrast, 'reference-centered contrast')
        terms = F.softplus(-contrast)
    result = zero + (weights * terms).sum() / opportunity_count
    _finite(result, 'teacher loss')
    return result


@torch.no_grad()
def teacher_diagnostics(current_winner_logp: torch.Tensor, current_learner_logp: torch.Tensor,
                        reference_winner_logp: torch.Tensor, reference_learner_logp: torch.Tensor,
                        admitted: torch.Tensor, lower_bound: torch.Tensor, *,
                        opportunity_count: int) -> dict:
    """Detached active-pair means in nats, including absolute NLL and log ratio.

    These are NOT token-normalized or final-Plan marginals. Counts and weight_sum
    make the empty case explicit (means reported as zero, not invented samples).
    Diagnostics require both current and reference paths, even for a BC run;
    teacher_loss itself deliberately does not depend on those unused BC values.
    """
    indices, weights = _selection(current_winner_logp, current_learner_logp,
        reference_winner_logp, reference_learner_logp, admitted, lower_bound, opportunity_count)
    xs = [x[indices] for x in (current_winner_logp, current_learner_logp,
                              reference_winner_logp, reference_learner_logp)]
    for x in xs:
        _finite(x, 'diagnostic logp')
    c, p, cr, pr = xs
    values = dict(winner_nll=-c, learner_nll=-p, log_ratio=c-p,
                  reference_centered_log_ratio=(c-p)-(cr-pr))
    for value in values.values():
        _finite(value, 'diagnostic contrast')
    return {'opportunity_count': opportunity_count,
            'supplied_opportunities': current_winner_logp.numel(),
            'admitted_count': int((admitted.detach() == 1).sum()),
            'positive_weight_count': indices.numel(), 'weight_sum': float(weights.sum()),
            **{name: float(value.mean()) if value.numel() else 0.
               for name, value in values.items()}}


def class_path_logp(model, path: dict) -> torch.Tensor:
    """Recompute one source-bound regret-record path with the production scorer.

    ``path = {'pair': document, 'side': 'challenger' | 'learner'}``, where document
    is the existing farm-v0-regret-pair-v1 JSON record. validate_pair owns source,
    actor allowlist, continuation and geometry checks; this adapter does not
    introduce another trace schema or relax admission. Scoring is possible for
    valid UNADMITTED records (e.g. diagnostics); ONLY teacher_loss applies gates.

    Both paths start at actor_context.prefix, then advance from freshly resolved
    successors, NEVER from stored teacher prefixes. All selected path menus are
    scored in one ragged score_batch, using only public/own State and HOLD memory.
    This reuses BC.path_nll (sum of alias-class logsumexp terms, sampled STOP
    included; forced cap STOP = 0). It is a class-PATH likelihood at temperature
    one, NOT a final-Plan marginal, nor the direct draw's raw behavior likelihood.

    Caller owns eval/no_grad for the frozen reference and its identity. This
    function changes neither model mode nor RNG. The conservative v1 adapter
    validates the complete pair before rebuilding the chosen path; it deliberately
    caches no model tensors or cross-call geometry. Runtime profiling is deferred.
    """
    from farm_neat.policy import v0_menu_fast
    from farm_neat.training import v0_bc as BC
    from farm_neat.training import v0_regret_records as R

    _require(type(path) is dict and set(path) == {'pair', 'side'},
             'path requires exactly pair and side')
    _require(type(path['side']) is str and path['side'] in ('challenger', 'learner'),
             'side must be challenger or learner')
    document = R.validate_pair(path['pair'])
    state, prefix, options = R._actor(document['actor_context'])
    menus, targets = [], []
    for action in document[path['side']]['raw_actions']:
        menu = v0_menu_fast.build_menu(state, prefix, projection_kwargs=options, backend='fast')
        index = menu.legal_ids.index(action)
        cls = next(c for c in menu.classes if index in c.indices)
        menus.append(menu)
        targets.append(cls.indices)
        prefix = cls.successor
    pack = model.score_batch([state] * len(menus), [m.prefix for m in menus],
        [m.legal_ids for m in menus], [m.phi for m in menus],
        memories=[{'hold_releases': options['hold_releases']} for _ in menus])
    return -BC.path_nll(list(pack.logits), targets, valid=list(pack.valid))


def main(argv: list[str] | None = None) -> int:
    """CPU scalar-logp diagnostics; no model allocation, checkpoint or optimizer.

    python3 -B -m farm_neat.training.v0_regret_loss diagnose --input LOGPS.json

    JSON keys match teacher_loss except mode (both arms run). admitted is a
    boolean list; other vectors are numbers, with null allowed for excluded
    entries. beta and opportunity_count are explicit. Output is atomic at the
    process level: invalid inputs print no partial success. This is a runnable
    consumer for maintainer command registration, not a production trainer.
    """
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from farm_neat.training import v0_regret_records as R

    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    diagnostic = commands.add_parser('diagnose', help='evaluate both teacher arms on CPU logp vectors')
    diagnostic.add_argument('--input', type=Path, required=True)
    args = parser.parse_args(argv)
    names = ('current_winner_logp', 'current_learner_logp', 'reference_winner_logp',
             'reference_learner_logp')
    try:
        document = R._loads(args.input.read_text(encoding='utf-8'))
        _require(type(document) is dict and set(document) == set(names) | {
            'admitted', 'lower_bound', 'beta', 'opportunity_count'}, 'exact diagnostic input keys required')
        _require(type(document['admitted']) is list and all(type(v) is bool for v in document['admitted']),
                 'diagnostic admitted must be a boolean list')
        def vector(name):
            values = document[name]
            _require(type(values) is list and all(v is None or type(v) in (int, float) for v in values),
                     'diagnostic ' + name + ' must be a number/null list')
            return torch.tensor([float('nan') if v is None else v for v in values],
                                dtype=torch.float64, requires_grad=True)
        vectors = [vector(name) for name in names]
        gate = torch.tensor(document['admitted'], dtype=torch.bool)
        bound = vector('lower_bound')
        count, beta = document['opportunity_count'], document['beta']
        diagnostics = teacher_diagnostics(*vectors, gate, bound, opportunity_count=count)
        arms = {}
        for mode in MODES:
            for value in [*vectors, bound]:
                value.grad = None
            loss = teacher_loss(*vectors, gate, bound, mode=mode, beta=beta, opportunity_count=count)
            loss.backward()
            arms[mode] = {'loss': float(loss.detach()),
                          'winner_gradient': vectors[0].grad.tolist(),
                          'learner_gradient': vectors[1].grad.tolist(),
                          'evidence_received_gradients': any(x.grad is not None for x in [*vectors[2:], bound])}
        result = {'schema': 'farm-v0-regret-loss-diagnostic-v1', 'diagnostic_only': True,
                  'input_sha256': R.canonical_sha256(document),
                  'module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'arms': arms, 'diagnostics': diagnostics, 'games': 0, 'optimizer_updates': 0}
        output = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
    except (ValueError, TypeError, OSError, OverflowError) as exc:
        parser.exit(2, f'regret loss diagnostic refused: {exc}\n')
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
