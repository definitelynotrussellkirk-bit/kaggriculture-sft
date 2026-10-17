"""Finite-menu terminal sequence-Q, separate from the direct-policy trainer.

The production trunk is the existing 16-layer V0Scorer. A small batched GRU
encodes selected native successor-class actions and their actual-prefix phi.
A reduced CPU graph is an explicitly diagnostic connectivity/overfit test.

Objectives: terminal WDL cross entropy, terminal scalar fitted regression, and
terminal regression plus a finite-menu log-mean-exp conservatism penalty. The
last is an ablation inspired by CQL, NOT bootstrapped CQL or its lower-bound
claim. No hidden-state sufficiency, calibration or policy-strength guarantee.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import platform
import random
import time
import tomllib
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.policy import v0_scorer as B
from farm_neat.policy.qwen_hybrid import Config
from farm_neat.training import v0_sequence_q_data as D
from farm_neat.training import v0_regret_records as R
from farm_neat.training.teachers import v0_regret_proposals as P

OBJECTIVES = ('wdl', 'fitted', 'conservative')
DIAGNOSTIC_CONFIG = dict(width=16, layers=4, heads=2, experts=2, expert_hidden=8, conv_size=2)


@dataclass(frozen=True)
class PreparedMenu:
    """Actor-only CPU feature pack; no labels, source IDs or evaluator fields."""
    candidate_ids: tuple[str, ...]
    state: Any
    prefix: B.Prefix
    memory: dict
    actions: tuple[tuple[int, ...], ...]
    phi: tuple[tuple[tuple[float, ...], ...], ...]
    cursors: tuple[tuple[tuple[float, ...], ...], ...]
    menu_sha256: str


def prepare_menu(menu: dict) -> PreparedMenu:
    D.validate_menu(menu)
    context = menu['actor_context']; state, initial, options = R._actor(context)
    cache = {}; actions, features, cursors = [], [], []
    for c in menu['candidates']:
        prefix = initial; aa, ff, cc = [], [], []
        for raw in c['raw_actions']:
            current = P._menu(state, prefix, options, cache)
            index = current.legal_ids.index(raw)
            cls = next(x for x in current.classes if index in x.indices)
            # Aliases share the canonical successor. Never encode sampled raw history.
            canonical = min(current.legal_ids[i] for i in cls.indices)
            canonical_index = current.legal_ids.index(canonical)
            aa.append(canonical)
            ff.append(tuple(float(x) for x in current.phi[canonical_index]))
            cc.append(((prefix.last_key + 1) / V.N_KEYS, prefix.edit_count / 16.,
                       float(raw == V.STOP_ID), float(prefix.edit_count == 16)))
            prefix = cls.successor
        D.need(prefix.terminated, 'COMPLETE_SEQUENCE_REQUIRED')
        actions.append(tuple(aa)); features.append(tuple(ff)); cursors.append(tuple(cc))
    return PreparedMenu(tuple(c['candidate_id'] for c in menu['candidates']), state, initial,
                        {'hold_releases': options['hold_releases']}, tuple(actions), tuple(features),
                        tuple(cursors), menu['content_sha256'])


@dataclass
class QOutput:
    q: torch.Tensor
    probabilities: torch.Tensor | None
    logits: torch.Tensor | None
    groups: tuple[tuple[int, int], ...]
    candidate_ids: tuple[str, ...]


def validate_config(config: Config, *, diagnostic: bool) -> None:
    """Bound allocations before constructing a trunk from checkpoint metadata."""
    values = asdict(config)
    epsilon = values.pop('rms_epsilon')
    D.need(type(epsilon) is float and math.isfinite(epsilon) and 0 < epsilon <= .01, 'CHECKPOINT_CONFIG_EPSILON')
    limits = dict(width=144, layers=16, heads=8, experts=64, top_k=2,
                  expert_hidden=128, branches=4, conv_size=4)
    D.need(set(values) == set(limits) and all(type(v) is int and 1 <= v <= limits[k]
           for k, v in values.items()), 'CHECKPOINT_CONFIG_BOUNDS')
    D.need(config.width % config.heads == 0 and config.top_k <= config.experts, 'CHECKPOINT_CONFIG_SHAPE')
    D.need(type(diagnostic) is bool and (diagnostic or config == Config()), 'PRODUCTION_CONFIG_REQUIRED')


class V0SequenceQ(nn.Module):
    def __init__(self, config: Config | None = None, *, diagnostic: bool = False,
                 objective: str = 'wdl'):
        super().__init__()
        D.need(objective in OBJECTIVES, 'OBJECTIVE')
        self.objective, self.diagnostic = objective, diagnostic
        config = Config() if config is None else config
        validate_config(config, diagnostic=diagnostic)
        self.scorer = B.V0Scorer(config, diagnostic=diagnostic)
        width = self.scorer.config.width
        # These scalar policy-logit layers are not part of sequence-Q. Do not
        # claim that unused parameters learn; they remain frozen in the lineage.
        for layer in (self.scorer.context_query, self.scorer.score_head):
            layer.requires_grad_(False)
        self.cursor_input = nn.Linear(4, width, bias=False)
        self.sequence = nn.GRU(width, width, batch_first=True)
        self.context = nn.Linear(width, width)
        self.readout = nn.Linear(2 * width, width)
        self.head = nn.Linear(width, 3 if objective == 'wdl' else 1)

    def forward(self, menus: list[PreparedMenu]) -> QOutput:
        D.need(type(menus) is list and 0 < len(menus) <= 256
               and all(isinstance(x, PreparedMenu) for x in menus), 'PREPARED_MENU_BATCH_REQUIRED')
        device = self.head.weight.device
        D.need(self.head.weight.dtype == torch.float32 and not torch.is_autocast_enabled(device.type), 'FP32_REQUIRED')
        D.need(device.type != 'cuda' or not torch.backends.cuda.matmul.allow_tf32, 'TF32_DISABLED_REQUIRED')
        context, _ = self.scorer._encode([m.state for m in menus], [m.prefix for m in menus],
                                         [m.memory for m in menus])
        aa, ff, cc, owner, ids, lengths, groups = [], [], [], [], [], [], []
        for i, menu in enumerate(menus):
            start = len(ids)
            for key, a, f, c in zip(menu.candidate_ids, menu.actions, menu.phi, menu.cursors):
                aa.append(a); ff.append(f); cc.append(c); owner.append(i); ids.append(key); lengths.append(len(a))
            groups.append((start, len(ids)))
        n, length = len(ids), max(lengths)
        # Pack on CPU once, then one transfer per tensor. No Python-per-token GPU loop.
        action = torch.zeros(n, length, dtype=torch.long)
        phi = torch.zeros(n, length, len(B.PHI_NAMES), dtype=torch.float32)
        cursor = torch.zeros(n, length, 4, dtype=torch.float32)
        for i, (a, f, c) in enumerate(zip(aa, ff, cc)):
            action[i, :len(a)] = torch.tensor(a)
            phi[i, :len(a)] = torch.tensor(f)
            cursor[i, :len(a)] = torch.tensor(c)
        action, phi, cursor = action.to(device), phi.to(device), cursor.to(device)
        D.need(bool(torch.isfinite(phi).all()), 'NONFINITE_PHI')
        transformed = phi.sign() * phi.abs().log1p() / 10.
        tokens = (self.scorer.action_embedding(self.scorer.action_fields[action]).sum(-2)
                  + self.scorer.phi_input(transformed) + self.cursor_input(cursor))
        hidden, _ = self.sequence(tokens)
        last = hidden[torch.arange(n, device=device), torch.tensor(lengths, device=device) - 1]
        state = self.context(context)[torch.tensor(owner, device=device)]
        logits = self.head(torch.tanh(self.readout(torch.cat((state, last), dim=1))))
        D.need(bool(torch.isfinite(logits).all()), 'NONFINITE_Q')
        probabilities = logits.softmax(-1) if self.objective == 'wdl' else None
        q = probabilities[:, 2] - probabilities[:, 0] if probabilities is not None else logits[:, 0].tanh()
        return QOutput(q, probabilities, logits if probabilities is not None else None, tuple(groups), tuple(ids))


def terminal_loss(output: QOutput, labels: torch.Tensor, *, objective: str,
                  known: torch.Tensor | None = None, alpha: float = .1,
                  group_weights: list[float] | None = None) -> torch.Tensor:
    D.need(objective in OBJECTIVES and type(alpha) in (float, int)
           and math.isfinite(alpha) and alpha >= 0, 'LOSS_CONFIG')
    q = output.q
    labels = labels.to(q.device)
    known = torch.ones_like(q, dtype=torch.bool) if known is None else known.to(q.device)
    D.need(labels.shape == q.shape and known.shape == q.shape and known.dtype == torch.bool, 'LABEL_ALIGNMENT')
    D.need(bool(known.any()), 'NO_LABELS')
    D.need(labels.dtype != torch.bool and bool(torch.isfinite(labels[known]).all())
           and bool(((labels[known] == -1) | (labels[known] == 0) | (labels[known] == 1)).all()), 'TERMINAL_WDL_REQUIRED')
    weights = [1.] * len(output.groups) if group_weights is None else group_weights
    D.need(len(weights) == len(output.groups) and all(math.isfinite(x) and x > 0 for x in weights), 'GROUP_WEIGHTS')
    terms, used_weights = [], []
    for (lo, hi), weight in zip(output.groups, weights):
        selected = known[lo:hi]
        if not bool(selected.any()):
            continue
        target = labels[lo:hi][selected]
        if objective == 'wdl':
            D.need(output.logits is not None, 'WDL_HEAD_REQUIRED')
            loss = F.cross_entropy(output.logits[lo:hi][selected], target.long() + 1)
        else:
            D.need(output.probabilities is None, 'SCALAR_HEAD_REQUIRED')
            loss = F.mse_loss(q[lo:hi][selected], target.float())
            if objective == 'conservative':
                # Equal-score menu has zero penalty. There is no TD target or
                # claim that this regularizer fixes missing action coverage.
                penalty = torch.logsumexp(q[lo:hi], 0) - math.log(hi - lo) - q[lo:hi][selected].mean()
                loss = loss + alpha * penalty
        terms.append(loss * weight); used_weights.append(weight)
    return torch.stack(terms).sum() / sum(used_weights)


def support_profile(rows: list[dict], packs: list[PreparedMenu] | None = None) -> dict:
    packs = [prepare_menu(r['menu']) for r in rows] if packs is None else packs
    support = defaultdict(set); sequences = defaultdict(set)
    for row, pack in zip(rows, packs):
        world = row['provenance']['world_id']
        for key, actions in zip(pack.candidate_ids, pack.actions):
            if row['labels'][key] is None:
                continue
            sequences[D.digest(list(actions))].add(world)
            for action in set(actions):
                support[str(action)].add(world)
    obs = [r['menu']['actor_context']['observation'] for r in rows]
    return D.seal(dict(schema=D.SCHEMA + '-support', action_world_counts={k: len(v) for k, v in sorted(support.items())},
        sequence_world_counts={k: len(v) for k, v in sorted(sequences.items())},
        training_world_ids=sorted({r['provenance']['world_id'] for r in rows}),
        training_family_ids=sorted({r['provenance']['family_id'] for r in rows}),
        envelope={k: [min(o[k] for o in obs), max(o[k] for o in obs)] for k in ('step', 'cash')},
        scope='HEURISTIC_MARGINAL_SUPPORT_NOT_IDENTIFIABILITY_OR_CALIBRATION'))


def production_runtime() -> None:
    """Do not publish production weights from the portable CPU-review runtime."""
    with (D.ROOT / 'pyproject.toml').open('rb') as stream:
        pinned = tomllib.load(stream)['tool']['farm-neat']['runtime']
    import numpy
    D.need(bool(os.environ.get('SLURM_JOB_ID')) and torch.cuda.is_available(), 'PRODUCTION_REQUIRES_SLURM_CUDA')
    D.need(platform.python_version() == pinned['python'] and torch.__version__ == pinned['torch']
           and numpy.__version__ == pinned['numpy'], 'PRODUCTION_RUNTIME_MISMATCH')


def fit(rows: list[dict], *, diagnostic: bool = False, objective: str = 'wdl', steps: int = 200,
        learning_rate: float = .003, seed: int = 0, batch_menus: int = 8, alpha: float = .1,
        min_contrasting_worlds: int = 32, evidence_root: str | Path | None = None):
    """Bounded fresh fit, not resume or policy deployment. World-balanced menu loss."""
    D.need(type(steps) is int and 1 <= steps <= 100000 and type(batch_menus) is int and 1 <= batch_menus <= 256, 'FIT_BUDGET')
    D.need(type(learning_rate) in (int, float) and math.isfinite(learning_rate) and 0 < learning_rate <= .1, 'LEARNING_RATE')
    D.need(type(seed) is int and 0 <= seed < 2**63, 'SEED')
    if not diagnostic:
        production_runtime()
    stats = D.training_gate(rows, diagnostic=diagnostic, min_contrasting_worlds=min_contrasting_worlds,
                            evidence_root=evidence_root)
    # A completely censored menu is retained in the corpus, but not an optimizer batch.
    rows = [r for r in rows if any(v is not None for v in r['labels'].values())]
    began = time.monotonic(); packs = [prepare_menu(r['menu']) for r in rows]
    profile = support_profile(rows, packs)
    feature_seconds = time.monotonic() - began
    torch.manual_seed(seed)
    if not diagnostic:
        torch.backends.cuda.matmul.allow_tf32 = False
    model = V0SequenceQ(Config(**DIAGNOSTIC_CONFIG) if diagnostic else Config(),
                       diagnostic=diagnostic, objective=objective).to('cpu' if diagnostic else 'cuda')
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    rng = random.Random(seed); counts = Counter(r['provenance']['world_id'] for r in rows)
    weights = [1. / counts[r['provenance']['world_id']] for r in rows]

    def batch(indices):
        output = model([packs[i] for i in indices])
        targets = [rows[i]['labels'][k] for i in indices for k in packs[i].candidate_ids]
        labels = torch.tensor([0 if z is None else z for z in targets], device=output.q.device)
        known = torch.tensor([z is not None for z in targets], device=output.q.device)
        return output, labels, known, [weights[i] for i in indices]

    def measure():
        losses, correct, total = [], 0, 0
        model.eval()
        with torch.no_grad():
            for start in range(0, len(rows), batch_menus):
                idx = list(range(start, min(start + batch_menus, len(rows))))
                out, y, known, w = batch(idx)
                loss = terminal_loss(out, y, objective=objective, known=known, alpha=alpha, group_weights=w)
                losses.append((loss.item(), sum(w)))
                pred = out.probabilities.argmax(-1) - 1 if out.probabilities is not None else out.q.round()
                correct += int((pred[known] == y[known]).sum()); total += int(known.sum())
        model.train()
        return sum(x * w for x, w in losses) / sum(w for _, w in losses), correct / total

    initial, _ = measure(); losses = []; began = time.monotonic()
    for _ in range(steps):
        indices = rng.sample(range(len(rows)), min(batch_menus, len(rows)))
        output, labels, known, w = batch(indices)
        loss = terminal_loss(output, labels, objective=objective, known=known, alpha=alpha, group_weights=w)
        D.need(bool(torch.isfinite(loss)), 'NONFINITE_LOSS')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); losses.append(float(loss.detach().cpu()))
    if not diagnostic:
        torch.cuda.synchronize()
    elapsed = time.monotonic() - began
    final, accuracy = measure(); model.eval()
    report = D.seal(dict(schema=D.SCHEMA + '-fit', scope='DIAGNOSTIC_CPU' if diagnostic else 'PRODUCTION_NOT_STRENGTH',
        objective=objective, optimizer_steps=steps, learning_rate=learning_rate, seed=seed, alpha=alpha,
        initial_loss=initial, final_loss=final, training_accuracy=accuracy,
        feature_preparation_seconds=feature_seconds, training_seconds=elapsed,
        native_games_run=0, gpu_hours=0. if diagnostic else elapsed / 3600., corpus=stats,
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trunk_layers=model.scorer.config.layers, last_gradient_norm=float(norm.detach().cpu()),
        loss_history=losses, validation_or_playing_strength_measured=False))
    return model, report, profile


def save_checkpoint(path: str | Path, model: V0SequenceQ, *, support: dict, training: dict) -> dict:
    path = D.A.safe_path(path)
    metadata = dict(schema=D.SCHEMA + '-checkpoint', config=asdict(model.scorer.config),
        diagnostic=model.diagnostic, objective=model.objective, source_pins=D.source_pins(),
        support=support, training=training)
    D.wire(metadata)
    state = model.state_dict()
    for value in state.values():
        if isinstance(value, torch.Tensor):
            D.need(bool(torch.isfinite(value).all()), 'NONFINITE_CHECKPOINT')
    with path.open('xb') as stream:
        torch.save({'metadata': metadata, 'state_dict': state}, stream)
        stream.flush(); os.fsync(stream.fileno())
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def load_checkpoint(path: str | Path, *, expected_sha256: str, allow_diagnostic: bool = False):
    path = D.A.safe_path(path)
    D.need(path.is_file() and path.stat().st_size <= 1024**3, 'CHECKPOINT_SIZE_OR_PATH')
    D.need(hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256, 'CHECKPOINT_SHA256_MISMATCH')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    D.need(type(payload) is dict and set(payload) == {'metadata', 'state_dict'}, 'CHECKPOINT_FIELDS')
    m = payload['metadata']
    R._fields(m, ('schema', 'config', 'diagnostic', 'objective', 'source_pins', 'support', 'training'), '$.checkpoint')
    D.need(m['schema'] == D.SCHEMA + '-checkpoint' and m['source_pins'] == D.source_pins(), 'CHECKPOINT_SOURCE_DRIFT')
    D.need(type(m['diagnostic']) is bool and (allow_diagnostic or not m['diagnostic']), 'DIAGNOSTIC_CHECKPOINT_REFUSED')
    model = V0SequenceQ(Config(**m['config']), diagnostic=m['diagnostic'], objective=m['objective'])
    expected = model.state_dict(); received = payload['state_dict']
    D.need(set(expected) == set(received), 'CHECKPOINT_PARAMETER_KEYS')
    for key, dest in expected.items():
        value = received[key]
        if isinstance(dest, torch.Tensor):
            D.need(isinstance(value, torch.Tensor) and value.shape == dest.shape and value.dtype == dest.dtype
                   and bool(torch.isfinite(value).all()), 'CHECKPOINT_PARAMETER_SHAPE_DTYPE_OR_FINITE')
        else:
            D.need(value == dest, 'CHECKPOINT_NATIVE_IDENTITY')
    model.load_state_dict(received, strict=True); model.eval()
    return model, m


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--objective', choices=OBJECTIVES, default='wdl'); parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--steps', type=int, default=200); parser.add_argument('--learning-rate', type=float, default=.003)
    parser.add_argument('--seed', type=int, default=0); parser.add_argument('--batch-menus', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=.1); parser.add_argument('--min-contrasting-worlds', type=int, default=32)
    parser.add_argument('--evidence-root', default=str(D.ROOT))
    args = parser.parse_args(argv)
    try:
        corpus = D.read_json(args.corpus)
        D.check(corpus, D.SCHEMA + '-corpus', ('rows', 'stats', 'source_pins'))
        D.need(corpus['source_pins'] == D.source_pins(), 'CORPUS_SOURCE_DRIFT')
        D.need(corpus['stats'] == D.corpus_stats(corpus['rows']), 'CORPUS_STATS_DRIFT')
        out = D.A.safe_path(args.output); D.need(not out.exists(), 'NEW_OUTPUT_REQUIRED')
        model, report, support = fit(corpus['rows'], diagnostic=args.diagnostic, objective=args.objective,
            steps=args.steps, learning_rate=args.learning_rate, seed=args.seed, batch_menus=args.batch_menus,
            alpha=args.alpha, min_contrasting_worlds=args.min_contrasting_worlds, evidence_root=args.evidence_root)
        out.mkdir(parents=True, exist_ok=False)
        checkpoint = save_checkpoint(out / 'model.pt', model, support=support,
                                     training=dict(report=report, corpus_sha256=corpus['content_sha256']))
        D.write_new(out / 'FIT.json', report)
        receipt = D.seal(dict(schema=D.SCHEMA + '-training-receipt', checkpoint=checkpoint, report=report))
        D.write_new(out / 'CHECKPOINT.json', receipt)
        print(D.wire(receipt)); return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(D.wire(dict(schema=D.SCHEMA, status='REFUSED', reason=str(exc)))); return 2


if __name__ == '__main__':
    raise SystemExit(main())
