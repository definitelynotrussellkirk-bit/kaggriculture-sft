"""Primitive-BC v3 trainer: stored-state streaming over per-step primitive actions.

A pool of open perspectives (one seat of one episode, streamed from step 0) is kept, each
with its next ``chunk``-step chunk and its detached carried state.  Each update samples
``batch`` pool slots, runs every slot's next chunk from its stored state (teacher forcing,
truncated BPTT inside the chunk), writes the detached state back and replaces finished
perspectives with fresh ones.  ``--lockstep``: the pool is exactly the batch and every slot
advances every update (state never stale).  carry_scope='day' resets the carried state at
hour 0 (done here and in agent.py; idempotent if model.py does it too).

Loss: summed cross-entropy per example step over present units (token, quantity class,
digits), and the order sequence incl. NOOP/STOP (token, quantity class, digits), times the
step's loss_mask, averaged over loss-bearing example steps.
Optimisation (standard): AdamW betas (0.9, 0.95), weight decay 0.1 on nn.Linear weight
matrices only, linear warmup then cosine to ``min_lr_frac`` of the peak, grad-norm clip 1.0,
bf16 autocast on CUDA only (fp32 master weights), fp32 on CPU.

Held-out metrics (perspectives streamed from step 0 with the same carry): per-head accuracy,
exact unit/order/whole step, non-PASS unit accuracy, change-point unit accuracy (label differs
from prev_unit_tok), per-team (meta team_id) unit/step accuracy, free-running greedy decode
(exact step, grammar validity, act()-vs-forward() carried-state difference), field statistics.
Collapse abort (think field/fsq) at collapse_check_update: > collapse_frac of field positions
constant (std < collapse_std over >= collapse_steps held-out steps) or FSQ entropy < 0.5 bits.

Synthetic data (--tiny): real engine observations through agent.InputBuilder (= data.step_inputs),
a fresh cue v_t in {0..V-1} one-hot in unused payload columns of every TILE row (default;
--synthetic-cue market: every MARKET and STOCK row; features: every feature token), never the
GLOBAL row unless --synthetic-cue-global. Default labels ('react'): units follow v_t; order
slot 0 = choice(v_{t-lag}) for t >= lag, then STOP (acc_lag). The lagged cue is visible only
through the carried state (think='off' has none); no earlier decision of the step reveals it.
Under 'react' the step's own unit labels also encode v_t, so the FIFO can carry the cue through
its action write; 'pass' leaves only the trunk-summary write (units all PASS).
'lag': units follow v_{t-lag} too; then only the farmer (the step's first decision) is a clean
probe (acc_lag_first), because teacher forcing shows each later decision the previous label.

v3.1 (ARCH-PBC-v3.md owner decisions, slot 103; each a TrainConfig flag, CLI --name value):
  wins_only / exclude_static_tapes (default on, train and held-out): the data rules of data.py
    (data.filter_reason); a missing EPISODE-TAPE-LABELS is refused unless --allow-no-tape-labels.
  cache (default "" = live data.iter_perspectives workers): --cache DIR reads a step cache
    (pbc-step-cache/build_cache.py CachedPerspectives: refuses other source pins, applies the
    same data rules to the cached metas); deterministic, so resume restores the exact pool.
  curriculum (default on for training, never for held-out): at each pool refill a candidate is
    accepted with probability data.Curriculum.weight(meta, f), f = completed updates / updates;
    after curriculum_max_rejects rejections in a row, Curriculum.pick chooses among them. The
    Curriculum is fitted on the split's population (header metas / cached metas); RUN-<n>.json
    has it and the expected distribution, metrics.jsonl the realised one per log window.
  round_robin (default on): a fixed permutation of the pool slots, drawn once from the seed,
    advanced `batch` slots per update: every re-used state is exactly pool/batch updates old
    (logged: state_age_*); off = random `batch` of `pool` per update (no bound).
  feature_dropout / feature_dropout_all (0.15 / 0.05, training only): per chunk (same mask for
    all its steps), a pure function of (seed, perspective id, chunk index): all feature groups
    (feature_tokens.SEGMENTS + the theorem tile block) with p_all, else each with p; dropped
    columns take the model's learned MISSING values with knownness 0 (model.py feat_missing);
    never the raw pack rows, never in evaluation or at inference.
  deferred_metrics (default on): metric tensors stay on the device, one transfer per update
    (Meter.materialize); losses, gradients and reports identical to the immediate path.
  copy_dropout (0.5, training only; stock-blind v3 diagnosis 2026-09-23, fix 2): per chunk (same
    mask for all its steps), a pure function of (seed, perspective id, chunk index), the
    perspective loses every teacher-action input with p: prev_unit_tok -1 (previous-unit tokens
    and the decoder's prev_own), teacher token team 0 / decile unknown, and no action term in
    the FIFO write (model.py batch['copy_drop']); never in evaluation or at inference.
  Configs saved before these flags load with TrainConfig.LEGACY (rules, curriculum, round
  robin and dropout off), so an old run resumes unchanged.

Commands:
  fit    train; --tiny = tiny model (model.SIZES['tiny']) + synthetic data on CPU;
         --synthetic-lag N (the carried-memory gate), --synthetic-lag-ramp K (curriculum: training
         perspective k uses lag min(N, 1 + k // K)); --lockstep; --resume.
  gate   synthetic-lag gate for several think modes (subprocesses) -> one JSON report.
  eval   held-out metrics of a checkpoint.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import glob as _glob
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CKPT_SCHEMA = "primitive-bc-train-checkpoint-v3"
RUN_SCHEMA = "primitive-bc-run-v3"
THINK_MODES = ("field", "fsq", "carry", "off")
LABEL_KEYS = ("unit_tok", "unit_qty", "unit_digits", "order_tok", "order_qty", "order_digits")
HEADS = {  # head -> (candidate output keys, label key)
    "unit_tok": (("unit_logits", "unit_tok_logits"), "unit_tok"),
    "unit_qty": (("unit_qty_logits",), "unit_qty"),
    "unit_digits": (("unit_digit_logits", "unit_digits_logits"), "unit_digits"),
    "order_tok": (("order_logits", "order_tok_logits"), "order_tok"),
    "order_qty": (("order_qty_logits",), "order_qty"),
    "order_digits": (("order_digit_logits", "order_digits_logits"), "order_digits"),
}


def _model():
    import model  # primitive-bc/model.py (lazy: tests inject stubs)
    return model


def _data():
    import data
    return data


def _codec():
    import codec
    return codec


@dataclass
class TrainConfig:
    chunk: int = 24                 # W: steps per chunk (truncated BPTT span)
    pool: int = 256                 # open perspectives
    batch: int = 16                 # perspectives advanced per update
    lockstep: bool = False          # pool == batch, every slot advances every update
    updates: int = 20000
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    qty_weight: float = 1.0
    digit_weight: float = 1.0
    bf16: bool = True               # autocast bf16 on CUDA only; CPU always fp32
    seed: int = 0
    ckpt_every: int = 500
    keep_ckpts: int = 3
    eval_every: int = 1000
    eval_perspectives: int = 8
    eval_steps: int = 256           # per held-out perspective, from step 0 (0 = whole episode)
    eval_free_steps: int = 48       # free-running decode on the first steps of each held-out perspective
    log_every: int = 20
    collapse_check_update: int = 500
    collapse_std: float = 0.01
    collapse_frac: float = 0.5
    collapse_steps: int = 256
    fsq_entropy_min: float = 0.5
    # data
    archives: list = dc_field(default_factory=list)
    split: str = "train"
    eval_split: str = "val"
    strength_top: float = 0.6
    seats: str = "both"
    workers: int = 4
    max_episodes: Optional[int] = None   # passed to data.iter_perspectives only if it accepts it
    # synthetic data (engine observations through the real input path, synthetic labels)
    synthetic: bool = False
    synthetic_lag: int = -1         # >=0: order slot 0 = cue shown `lag` steps earlier (-1: no order label)
    synthetic_values: int = 4       # cue alphabet (chance = 1 / values)
    synthetic_units: str = "react"  # react: unit labels follow the current cue; pass: every unit PASS;
                                    # lag: unit labels follow the lagged cue (PASS before step lag) - only
                                    # the farmer is a clean memory probe there (acc_lag_first): teacher
                                    # forcing shows each later decision the previous unit's label
    synthetic_cue: str = "tiles"    # cue one-hot in: tiles = every TILE row | market = every MARKET and
                                    # STOCK row | features = every feature token
    synthetic_cue_global: bool = False  # also in the GLOBAL row (the pre-2026-09-23 gate; the old FIFO
                                        # wrote the GLOBAL token, so that gate passed without game memory)
    synthetic_lag_ramp: int = 0     # >0: training perspective k uses lag min(synthetic_lag, 1 + k // ramp)
                                    # (curriculum, deterministic per perspective); held-out: synthetic_lag
    synthetic_len: int = 96         # steps per synthetic perspective
    synthetic_worlds: int = 2       # engine worlds in the observation bank (x2 seats)
    synthetic_bank_steps: int = 48  # engine steps per world in the bank
    # v3.1 (ARCH-PBC-v3.md owner decisions / slot 103); configs saved before a flag load with LEGACY
    wins_only: bool = True          # BC targets only from seats whose reward beats the rival's (train + eval)
    exclude_static_tapes: bool = True   # no perspective from a seat labelled static in EPISODE-TAPE-LABELS
    tape_labels: str = ""           # EPISODE-TAPE-LABELS path ("" = data.TAPE_LABELS)
    allow_no_tape_labels: bool = False  # run without the labels file (else a missing file is refused)
    cache: str = ""                 # step-cache directory (pbc-step-cache/build_cache.py); "" = live workers
    curriculum: bool = True         # refill-time sampling by data.Curriculum.weight(meta, f) (training only)
    curriculum_max_rejects: int = 64    # consecutive rejections before the fallback pick among them
    curriculum_min_final: int = 512     # data.Curriculum.fit (its default): widen the final set below this many
    round_robin: bool = True        # fixed-permutation slot rotation: exact state age pool/batch updates
    feature_dropout: float = 0.15   # per chunk and feature group: learned MISSING values, knownness 0
    feature_dropout_all: float = 0.05   # per chunk: every feature group at once
    deferred_metrics: bool = True   # metrics stay on device; one host transfer per update
    copy_dropout: float = 0.5       # per chunk: drop every teacher-action input (prev tokens, teacher
                                    # token, FIFO action write) of the perspective

    LEGACY = {"wins_only": False, "exclude_static_tapes": False, "curriculum": False, "round_robin": False,
              "feature_dropout": 0.0, "feature_dropout_all": 0.0, "synthetic_cue_global": True,
              "copy_dropout": 0.0}

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        for k, v in cls.LEGACY.items():
            kw.setdefault(k, v)
        return cls(**kw)


# ----------------------------------------------------------------------------- numerics
def autocast(device, enabled=True):
    """bf16 autocast on CUDA only; CPU training is plain fp32."""
    if enabled and torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def lr_at(update, cfg):
    """Linear warmup over cfg.warmup updates, then cosine from lr to min_lr_frac * lr."""
    if cfg.warmup and update < cfg.warmup:
        return cfg.lr * (update + 1) / cfg.warmup
    f = min(1.0, (update - cfg.warmup) / max(1, cfg.updates - cfg.warmup))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * f)))


def param_groups(net, weight_decay):
    """Weight decay on nn.Linear weight matrices only; none on biases, norms, embeddings and
    free parameters (latent/slot/init/position).  PolicyNet.param_groups() when it has one."""
    custom = getattr(net, "param_groups", None)
    if callable(custom):
        return custom(weight_decay)
    decay_ids = {id(m.weight) for m in net.modules() if isinstance(m, nn.Linear)}
    decay, no_decay, seen = [], [], set()
    for p in net.parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        (decay if id(p) in decay_ids else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def make_optimizer(net, cfg):
    return torch.optim.AdamW(param_groups(net, cfg.weight_decay), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


# ----------------------------------------------------------------------------- carried-state trees
def tree_map(fn, *trees):
    """Map over tensors of equally shaped trees (dict / list / tuple / dataclass / None)."""
    t0 = trees[0]
    if t0 is None:
        return None
    if torch.is_tensor(t0):
        return fn(*trees)
    if isinstance(t0, dict):
        return {k: tree_map(fn, *(t[k] for t in trees)) for k in t0}
    if isinstance(t0, (list, tuple)):
        items = [tree_map(fn, *xs) for xs in zip(*trees)]
        return type(t0)(*items) if hasattr(t0, "_fields") else type(t0)(items)
    if dataclasses.is_dataclass(t0) and not isinstance(t0, type):
        return dataclasses.replace(t0, **{f.name: tree_map(fn, *(getattr(t, f.name) for t in trees))
                                          for f in dataclasses.fields(t0) if f.init})
    return t0


def tree_cat(states):
    return tree_map(lambda *xs: torch.cat(xs, 0), *states)


def tree_rows(state, n):
    """Split a batch-first state into n single-row states (detached, fp32, CPU): one device ->
    host copy per tensor, then per-row clones."""
    host = tree_map(lambda x: x.detach().float().cpu() if x.is_floating_point() else x.detach().cpu(), state)
    return [tree_map(lambda x, i=i: x[i:i + 1].clone(), host) for i in range(n)]


def tree_select(state, idx):
    return tree_map(lambda x: x.index_select(0, idx.to(x.device)), state)


def tree_put(state, idx, part):
    return tree_map(lambda full, new: full.index_copy(0, idx.to(full.device), new.to(full.dtype)), state, part)


def tree_where(mask, a, b):
    """Row-wise select: a where mask[b] else b."""
    def pick(x, y):
        m = mask.to(x.device).view(-1, *([1] * (x.dim() - 1)))
        return torch.where(m, x.to(y.dtype), y)
    return tree_map(pick, a, b)


def tree_to(state, device):
    return tree_map(lambda x: x.to(device), state)


def initial_state(net, n, device):
    return tree_to(net.initial_state(n), device)


def carry_scope(net):
    return getattr(getattr(net, "config", None), "carry_scope", "episode")


def dawn_reset(net, state, batch, device):
    """carry_scope='day': rows whose step is hour 0 restart from the initial state."""
    if state is None or carry_scope(net) != "day":
        return state
    hour = batch.get("hour")
    if hour is None:
        return state
    dawn = hour == 0
    if not bool(dawn.any()):
        return state
    return tree_where(dawn, initial_state(net, int(dawn.shape[0]), device), state)


# ----------------------------------------------------------------------------- collate
def _pad_value(arr):
    if arr.dtype == np.bool_:
        return False
    if np.issubdtype(arr.dtype, np.integer):
        return -1
    return 0


def _stack(values):
    arrs = [np.asarray(v) for v in values]
    if arrs[0].ndim == 0:
        return torch.from_numpy(np.stack(arrs))
    shape = tuple(max(a.shape[d] for a in arrs) for d in range(arrs[0].ndim))
    if all(a.shape == shape for a in arrs):
        return torch.from_numpy(np.ascontiguousarray(np.stack(arrs)))
    out = np.full((len(arrs), *shape), _pad_value(arrs[0]), dtype=arrs[0].dtype)
    for i, a in enumerate(arrs):
        out[(i, *tuple(slice(0, s) for s in a.shape))] = a
    return torch.from_numpy(out)


def collate_tree(values):
    v0 = values[0]
    if isinstance(v0, dict):
        return {k: collate_tree([v[k] for v in values]) for k in v0}
    if isinstance(v0, (str, bytes)) or v0 is None:
        return list(values)
    if isinstance(v0, (bool, int, float, np.generic, np.ndarray, list, tuple)) or torch.is_tensor(v0):
        try:
            return _stack([v.numpy() if torch.is_tensor(v) else v for v in values])
        except (ValueError, TypeError):
            return list(values)
    return list(values)


def collate_steps(net, examples, device="cpu"):
    """Per-step example dicts (data.py layout) -> one batch [B, ...]: model.collate (the model
    builder's own padding/typing), else a generic padded stack for stub models."""
    fn = getattr(_model(), "collate", None)
    if fn is not None:
        return fn(list(examples), n_order_slots=getattr(net, "O", None),
                  digit_slots=getattr(net, "D", None), device=device)
    batch = _to_device(collate_tree(list(examples)), device)
    if "rows" in batch and batch["rows"].dtype == torch.float16:
        batch["rows"] = batch["rows"].float()
    meta = batch.get("meta") or {}
    for k in ("team_id", "strength_decile", "hour", "step"):
        if isinstance(meta, dict) and torch.is_tensor(meta.get(k)):
            batch.setdefault(k, meta[k])
    return batch


def meta_field(batch, key):
    """[B] tensor of a meta field: top level, dict of arrays, or list of dicts; None if absent."""
    if torch.is_tensor(batch.get(key)):
        return batch[key]
    meta = batch.get("meta")
    if isinstance(meta, dict) and key in meta:
        return torch.as_tensor(np.asarray(meta[key]))
    if isinstance(meta, (list, tuple)) and meta and all(isinstance(m, dict) and key in m for m in meta):
        return torch.as_tensor([m[key] for m in meta])
    return None


def _to_device(tree, device):
    if torch.is_tensor(tree):
        return tree.to(device, non_blocking=True)
    if isinstance(tree, dict):
        return {k: _to_device(v, device) for k, v in tree.items()}
    return tree


# ----------------------------------------------------------------------------- losses / metrics
def _logits(out, head):
    for key in HEADS[head][0]:
        if key in out and out[key] is not None:
            return out[key]
    return None


def _ce(logits, labels, valid):
    """Summed CE per example over the valid label positions; logits [B, ..., V]."""
    B = logits.shape[0]
    flat = logits.float().reshape(-1, logits.shape[-1])
    y = torch.where(valid, labels, torch.zeros_like(labels)).reshape(-1)
    ce = F.cross_entropy(flat, y, reduction="none").view(valid.shape)
    return (ce * valid).reshape(B, -1).sum(1)


def _detach(v):
    return v.detach() if torch.is_tensor(v) else v


class Meter:
    """Metric sums. defer=False reads every tensor contribution to the host at once (a sync
    per call). defer=True queues the detached tensors on their device and materialize() moves
    them to the host in one transfer per device, then applies the same sequential float64
    sums in call order, so reports are identical to the immediate path."""

    def __init__(self, defer=False):
        self.pairs, self.sums = {}, {}
        self.defer = defer
        self._queue = []

    def add_pair(self, key, hit, n):
        if self.defer:
            self._queue.append((key, _detach(hit), _detach(n)))
            return
        a, b = self.pairs.get(key, (0, 0))
        self.pairs[key] = (a + int(hit), b + int(n))

    def add(self, key, value):
        if self.defer:
            self._queue.append((key, _detach(value)))
            return
        self.sums[key] = self.sums.get(key, 0.0) + float(value)

    def materialize(self):
        """Apply queued contributions (one device -> host transfer per device)."""
        queue, self._queue = self._queue, []
        if not queue:
            return self
        by_dev = {}
        for item in queue:
            for v in item[1:]:
                if torch.is_tensor(v):
                    by_dev.setdefault(v.device, []).append(v)
        host = {}
        for dev, vals in by_dev.items():
            flat = torch.cat([v.reshape(1).to(torch.float64) for v in vals]).cpu().tolist()
            host.update(zip(map(id, vals), flat))
        get = lambda v: host[id(v)] if torch.is_tensor(v) else v  # noqa: E731
        for item in queue:
            if len(item) == 3:
                a, b = self.pairs.get(item[0], (0, 0))
                self.pairs[item[0]] = (a + int(get(item[1])), b + int(get(item[2])))
            else:
                self.sums[item[0]] = self.sums.get(item[0], 0.0) + float(get(item[1]))
        return self

    def merge(self, other):
        other.materialize()
        for k, (a, b) in other.pairs.items():
            self.add_pair(k, a, b)
        for k, v in other.sums.items():
            self.add(k, v)

    def report(self):
        self.materialize()
        out = {f"acc_{k}": (a / b if b else None) for k, (a, b) in self.pairs.items()}
        out.update({f"n_{k}": b for k, (a, b) in self.pairs.items()})
        out.update(self.sums)
        return out


def batch_labels(batch):
    lab = batch.get("labels") or {}
    return {k: (lab[k] if k in lab else batch.get(k)) for k in LABEL_KEYS}


def masked_labels(net, out, batch):
    """Loss/metric targets with model.compute_loss's masks: units only where present; a quantity
    only for a quantity token; digits only under QTY_EXACT; the STOP-only last order slot and
    padding -1."""
    lab = batch_labels(batch)
    c = getattr(net, "config", None)
    codec = _net_codec(net)
    exact = getattr(c, "qty_exact", None)
    exact = int(getattr(codec, "QTY_EXACT", 75)) if exact is None else exact
    uhas = getattr(net, "unit_qty_mask", None)
    ohas = getattr(net, "order_qty_mask", None)
    if uhas is None:
        uhas = torch.tensor([bool(v) for v in codec.UNIT_HAS_QTY])
    if ohas is None:
        ohas = torch.tensor([bool(v) for v in codec.ORDER_HAS_QTY])
    device = next(v for v in out.values() if torch.is_tensor(v)).device
    t = {k: (v.to(device).long() if v is not None else None) for k, v in lab.items()}
    if t["unit_tok"] is not None:
        ut = t["unit_tok"]
        if torch.is_tensor(out.get("unit_present")):
            ut = torch.where(out["unit_present"].to(device), ut, -1)
        t["unit_tok"] = ut
        if t["unit_qty"] is not None:
            t["unit_qty"] = torch.where((ut >= 0) & uhas.to(device)[ut.clamp(min=0)], t["unit_qty"], -1)
            if t["unit_digits"] is not None:
                t["unit_digits"] = torch.where((t["unit_qty"] == exact)[..., None], t["unit_digits"], -1)
    if t["order_tok"] is not None:
        ot = t["order_tok"].clone()
        O = getattr(net, "O", ot.shape[1])
        if ot.shape[1] >= O:
            ot[:, O - 1] = -1                          # STOP-only slot: no information
        t["order_tok"] = ot
        if t["order_qty"] is not None:
            t["order_qty"] = torch.where((ot >= 0) & ohas.to(device)[ot.clamp(min=0)], t["order_qty"], -1)
            if t["order_digits"] is not None:
                t["order_digits"] = torch.where((t["order_qty"] == exact)[..., None], t["order_digits"], -1)
    return t


def loss_mask_of(batch, B, device):
    lm = batch.get("loss_mask")
    if lm is None:
        return torch.ones(B, device=device)
    lm = lm.to(device).float()
    while lm.dim() > 1:
        lm = lm.amin(-1)
    return lm


def step_loss(net, out, batch, *, qty_weight=1.0, digit_weight=1.0, meter=None, pass_index=None, per_team=False):
    """(per-example weighted loss [B] (loss_mask applied), loss_mask sum, meter).

    Same targets and normalisation as model.compute_loss (sum of every head's CE / sum of
    loss_mask); qty_weight / digit_weight scale those heads (1.0 = compute_loss).  Missing
    heads contribute nothing and count their labels under n_missing_<head>."""
    meter = meter if meter is not None else Meter()
    labels = masked_labels(net, out, batch)
    first = next(v for v in out.values() if torch.is_tensor(v))
    B, device = first.shape[0], first.device
    lm = loss_mask_of(batch, B, device)
    per = torch.zeros(B, device=device)
    ok = {}
    for head, (_, lab_key) in HEADS.items():
        y = labels.get(lab_key)
        if y is None:
            continue
        logits = _logits(out, head)
        valid = y >= 0
        if logits is None:
            meter.add(f"n_missing_{head}", valid.sum())
            continue
        ce = _ce(logits, y, valid)
        weight = qty_weight if head.endswith("qty") else digit_weight if head.endswith("digits") else 1.0
        per = per + weight * ce
        with torch.no_grad():
            hit = (logits.argmax(-1) == y) & valid
            ok[head] = (hit, valid)
            w = lm.view(-1, *([1] * (valid.dim() - 1))) > 0
            meter.add_pair(head, (hit & w).sum(), (valid & w).sum())
            meter.add(f"loss_{head}", (ce * lm).sum())
    with torch.no_grad():
        _exact_counts(ok, lm > 0, meter, labels, pass_index, batch, per_team)
    # deferred: the loss-bearing count stays a device tensor (no host read); else a float
    n = lm.sum().detach() if meter.defer else float(lm.sum())
    meter.add("loss_bearing_steps", n)
    return per * lm, n, meter


def _item_ok(ok, tok, qty, dig, shape, device):
    """Per-slot exactness: token right and (quantity/digit labels absent or right)."""
    if tok not in ok:
        return None, None
    hit, valid = ok[tok]
    good = hit.clone()
    for key in (qty, dig):
        if key in ok:
            h, v = ok[key]
            if h.dim() > good.dim():                 # digits [B, n, S]
                h, v = (h | ~v).all(-1), v.any(-1)
                good = good & h
            else:
                good = good & (h | ~v)
    return good, valid


def _exact_counts(ok, use, meter, labels, pass_index, batch, per_team=False):
    unit_good, unit_valid = _item_ok(ok, "unit_tok", "unit_qty", "unit_digits", None, None)
    order_good, order_valid = _item_ok(ok, "order_tok", "order_qty", "order_digits", None, None)
    step_ok = use.clone()
    w = use[:, None]
    if unit_good is not None:
        meter.add_pair("unit_exact", (unit_good & unit_valid & w).sum(), (unit_valid & w).sum())
        step_ok &= (unit_good | ~unit_valid).all(1)
        if pass_index is not None:
            nonpass = unit_valid & (labels["unit_tok"][:, :unit_valid.shape[1]] != pass_index)
            meter.add_pair("unit_tok_nonpass", (ok["unit_tok"][0] & nonpass & w).sum(), (nonpass & w).sum())
        prev = meta_field(batch, "prev_unit_tok")
        if prev is not None:
            n = min(prev.shape[1], unit_valid.shape[1])
            change = unit_valid[:, :n] & (labels["unit_tok"][:, :n] != prev[:, :n].long())
            meter.add_pair("unit_tok_change", (ok["unit_tok"][0][:, :n] & change & w).sum(), (change & w).sum())
    if order_good is not None:
        meter.add_pair("order_exact", (order_good & order_valid & w).sum(), (order_valid & w).sum())
        step_ok &= (order_good | ~order_valid).all(1)
    meter.add_pair("step_exact", step_ok.sum(), use.sum())
    team = meta_field(batch, "team_id") if per_team else None
    if team is not None and unit_good is not None:
        unit_hit = ok["unit_tok"][0]
        for tid in torch.unique(team.to(use.device)[use]).tolist():
            rows = use & (team.to(use.device) == tid)
            meter.add_pair(f"team{int(tid)}_unit_tok", (unit_hit & unit_valid & rows[:, None]).sum(),
                           (unit_valid & rows[:, None]).sum())
            meter.add_pair(f"team{int(tid)}_step_exact", (step_ok & rows).sum(), rows.sum())
    return step_ok


# ----------------------------------------------------------------------------- field statistics
class FieldStats:
    """Per field position/channel mean and std over held-out steps (collapse check), and the
    per-position FSQ level entropy when the model exposes codes (or the field is quantised)."""

    def __init__(self, fsq_levels=None):
        self.n, self.s1, self.s2, self.levels = 0, None, None, fsq_levels
        self.hist = None

    @staticmethod
    def field_tensor(out):
        f = out.get("field")
        if f is None and ("cells" in out or "slots" in out):
            f = {k: out[k] for k in ("cells", "slots") if out.get(k) is not None}
        parts = []

        def grab(x):
            parts.append(x.detach().float().reshape(x.shape[0], -1, x.shape[-1]))
            return x
        tree_map(grab, f)
        if not parts:
            return None
        width = max(p.shape[-1] for p in parts)
        parts = [torch.nn.functional.pad(p, (0, width - p.shape[-1])) for p in parts]
        return torch.cat(parts, 1)

    def add(self, out, mask=None):
        f = self.field_tensor(out)
        if f is None:
            return
        if mask is not None:
            f = f[mask.to(f.device)]
        if not len(f):
            return
        f64 = f.double().cpu()
        if self.s1 is None:
            self.s1, self.s2 = torch.zeros(f64.shape[1:], dtype=torch.float64), torch.zeros(f64.shape[1:], dtype=torch.float64)
        self.s1 += f64.sum(0)
        self.s2 += (f64 * f64).sum(0)
        self.n += len(f64)
        codes = out.get("fsq_codes")
        if codes is None and self.levels:
            half = (self.levels - 1) / 2
            codes = torch.round((f.clamp(-1, 1) + 1) * half).long()
        elif codes is not None and mask is not None:
            codes = codes[mask.to(codes.device)]
        if codes is not None and self.levels:
            c = codes.detach().long().cpu().reshape(len(codes), -1).clamp(0, self.levels - 1)
            onehot = torch.nn.functional.one_hot(c, self.levels).sum(0).double()
            self.hist = onehot if self.hist is None else self.hist + onehot

    def report(self, std_threshold=0.01):
        if not self.n:
            return {"field_steps": 0}
        mean = self.s1 / self.n
        std = (self.s2 / self.n - mean * mean).clamp_min(0).sqrt()
        const_elem = std < std_threshold
        const_pos = const_elem.all(-1)
        rep = {"field_steps": self.n, "field_positions": int(std.shape[0]), "field_channels": int(std.shape[-1]),
               "field_const_position_frac": float(const_pos.float().mean()),
               "field_const_element_frac": float(const_elem.float().mean()),
               "field_std_mean": float(std.mean()), "field_abs_mean": float(mean.abs().mean())}
        if self.hist is not None:
            p = self.hist / self.hist.sum(-1, keepdim=True).clamp_min(1)
            ent = -(p * torch.log2(p.clamp_min(1e-12))).sum(-1)
            rep.update(fsq_entropy_bits_mean=float(ent.mean()), fsq_levels=self.levels,
                       fsq_entropy_bits_min=float(ent.min()))
        return rep


# ----------------------------------------------------------------------------- one chunk
def run_chunk(net, chunks, state, *, device, bf16=True, qty_weight=1.0, digit_weight=1.0, meter=None,
              field_stats=None, free_run=0, pass_index=None, lag_meter=None, feat_drop=None, copy_drop=None):
    """Teacher-forced pass over one chunk per perspective, carrying ``state`` step to step.

    chunks: list (B) of lists of per-step example dicts; the shorter ones stop early.
    feat_drop: training feature dropout for the B perspectives ({"cols": [B,K,F], "tile": [B]}
    bool, from feature_drop_batch), the same for every step of the chunk; None = no dropout.
    copy_drop: training copy dropout [B] bool (copy_drop_mask per perspective chunk), the same for
    every step of the chunk (model batch['copy_drop']); None = no dropout.
    Returns (mean loss over loss-bearing example steps, meter, new state [B], step count)."""
    B = len(chunks)
    T = max(len(c) for c in chunks)
    meter = meter if meter is not None else Meter()
    total, n_loss = None, 0.0
    drop = tree_to(feat_drop, device) if feat_drop is not None else None
    cdrop = torch.as_tensor(copy_drop).bool().to(device) if copy_drop is not None else None
    for t in range(T):
        live = [b for b in range(B) if t < len(chunks[b])]
        batch = collate_steps(net, [chunks[b][t] for b in live], device)
        full = len(live) == B
        idx = torch.tensor(live, dtype=torch.long)
        if drop is not None:
            batch["feat_drop"] = drop if full else tree_select(drop, idx)
        if cdrop is not None:
            batch["copy_drop"] = cdrop if full else tree_select(cdrop, idx)
        st = state if full or state is None else tree_select(state, idx)
        st = dawn_reset(net, st, batch, device)
        with autocast(device, bf16):
            out, new = net(batch, st)
        per, n, _ = step_loss(net, out, batch, qty_weight=qty_weight, digit_weight=digit_weight, meter=meter,
                              pass_index=pass_index, per_team=lag_meter is not None)
        total = per.sum() if total is None else total + per.sum()
        n_loss = n_loss + n
        if field_stats is not None:
            field_stats.add(out)
        if lag_meter is not None:
            _lag_counts(net, out, batch, lag_meter)
        if free_run and t < free_run:
            _free_run(net, batch, st, new, meter)
        if new is not None:
            state = new if full or state is None else tree_put(state, idx, new)
    meter.add("loss_sum", total.detach())
    # same value either way: an exact integer count, as a python float or a float32 tensor
    loss = total / (n_loss.clamp_min(1.0) if torch.is_tensor(n_loss) else max(1.0, n_loss))
    return loss, meter, state, T


def _lag_counts(net, out, batch, meter):
    """Synthetic gate: order slot 0 token on steps whose label depends on the lagged cue."""
    lag = meta_field(batch, "lag_label")
    logits = _logits(out, "order_tok")
    if lag is None or logits is None:
        return
    y = batch_labels(batch)["order_tok"][:, 0].long().to(logits.device)
    use = lag.bool().to(logits.device) & (y >= 0)
    hit = (logits[:, 0].argmax(-1) == y) & use
    meter.add_pair("lag", hit.sum(), use.sum())
    units = meta_field(batch, "lag_units")
    ulog = _logits(out, "unit_tok")
    if units is not None and ulog is not None and bool(units.any()):
        yu = masked_labels(net, out, batch)["unit_tok"]
        ok = (yu >= 0) & units.bool().to(yu.device)[:, None]
        hit = (ulog.argmax(-1) == yu) & ok
        meter.add_pair("lag_units", hit.sum(), ok.sum())
        # the farmer is the step's first decision: no teacher-forced label of this step precedes it
        # (later units and the orders see the previous decision's label, which reveals the cue)
        meter.add_pair("lag_first", hit[:, 0].sum(), ok[:, 0].sum())


def _free_run(net, batch, state_in, state_tf, meter):
    """Greedy free-running decode from the same incoming state: exactness vs labels, grammar
    validity, and parity of the carried state with the teacher-forced pass."""
    act = getattr(net, "act", None)
    if act is None:
        return
    try:
        # the carried state writes the step's action: commit the teacher's labels, as forward() does
        kw = {"commit_labels": batch_labels(batch)} if "commit_labels" in inspect.signature(act).parameters else {}
        labels, new = act(batch, state_in, **kw)
    except Exception as error:  # noqa: BLE001 - count and skip
        meter.add("free_run_errors", 1)
        return
    B = int(labels["unit_tok"].shape[0])
    rows = [net.codec_labels(labels, i) if hasattr(net, "codec_labels") else
            {k: v[i] for k, v in labels.items() if torch.is_tensor(v)} for i in range(B)]
    if state_tf is not None and new is not None:
        diffs = []
        tree_map(lambda a, b: diffs.append(float((a.float() - b.float()).abs().max())) or a, state_tf, new)
        meter.add("free_state_maxdiff_sum", max(diffs) if diffs else 0.0)
        meter.add("free_state_checks", 1)
    codec = _net_codec(net)
    ref = batch_labels(batch)
    lm = loss_mask_of(batch, B, labels["unit_tok"].device)
    for i, lab in enumerate(rows):
        if not bool(lm[i] > 0):
            continue
        exact = all(_same_label(lab.get(k), ref[k][i]) for k in LABEL_KEYS if ref.get(k) is not None)
        meter.add_pair("free_step_exact", int(exact), 1)
        meter.add_pair("free_valid", int(label_grammar_ok(lab, codec)), 1)


def _net_codec(net):
    c = getattr(net, "codec", None)
    return c if c is not None else _codec()


def _as_list(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return np.asarray(x).tolist()


def _same_label(pred, ref):
    """Equal after dropping -1 padding at the tail of each row."""
    a, b = _as_list(pred), _as_list(ref)
    if a is None:
        return False

    def trim(v):
        if isinstance(v, list) and v and isinstance(v[0], list):
            rows = [trim(r) for r in v]
            while rows and not rows[-1]:
                rows.pop()
            return rows
        v = list(v) if isinstance(v, list) else [v]
        while v and v[-1] == -1:
            v.pop()
        return v
    return trim(a) == trim(b)


def label_grammar_ok(lab, codec):
    """Structural validity of one step's labels: vocab ranges, quantity iff the token takes
    one, digits END-terminated with no leading zero when QTY_EXACT, orders end at STOP."""
    try:
        exact_cls = getattr(codec, "QTY_EXACT", None)
        n_q = getattr(codec, "QTY_CLASSES", 76)
        end = getattr(codec, "DIGIT_END", 10)

        def item_ok(tok, qty, digs, vocab, has_qty):
            if not 0 <= tok < len(vocab):
                return False
            if has_qty[tok]:
                if not 0 <= qty < n_q:
                    return False
                if qty == exact_cls:
                    d = [x for x in (digs or []) if x != -1]
                    if end not in d:
                        return False
                    d = d[:d.index(end)]
                    if not d or len(d) > getattr(codec, "MAX_DIGITS", 5) or (len(d) > 1 and d[0] == 0):
                        return False
                    if int("".join(map(str, d))) < 1:
                        return False
            elif qty not in (-1, None):
                return False
            return True
        ut, uq = _as_list(lab["unit_tok"]), _as_list(lab.get("unit_qty"))
        ud = _as_list(lab.get("unit_digits")) or [None] * len(ut)
        for i, tok in enumerate(ut):
            if tok == -1:
                continue
            if not item_ok(tok, uq[i] if uq else -1, ud[i] if i < len(ud) else None, codec.UNIT_VOCAB, codec.UNIT_HAS_QTY):
                return False
        ot, oq = _as_list(lab["order_tok"]), _as_list(lab.get("order_qty"))
        od = _as_list(lab.get("order_digits")) or [None] * len(ot)
        toks = [t for t in ot if t != -1]
        stop = codec.ORDER_VOCAB.index("STOP")
        if not toks or toks[-1] != stop or stop in toks[:-1] or len(toks) > getattr(codec, "MAX_ORDERS", 10) + 1:
            return False
        for i, tok in enumerate(toks[:-1]):
            if not item_ok(tok, oq[i] if oq else -1, od[i] if i < len(od) else None, codec.ORDER_VOCAB, codec.ORDER_HAS_QTY):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------------- feature dropout
_FEATURE_GROUPS = None


def feature_groups():
    """Training feature-dropout groups: one per feature_tokens.SEGMENTS entry (its token slots
    x its payload columns; the identity one-hot columns are never dropped), then the theorem
    tile residual block. -> (names, cols bool [G-1, K, F]); the raw pack rows are no group."""
    global _FEATURE_GROUPS
    if _FEATURE_GROUPS is None:
        import feature_tokens as TK
        names, cols = [], []
        for slots, block, _, off in TK.SEGMENTS:
            m = np.zeros((TK.K, TK.WIDTH), np.bool_)
            m[list(slots), off:off + len(TK.FT.BLOCKS[block][1])] = True
            names.append(f"{block}@{TK.SLOTS[slots[0]][0]}")
            cols.append(m)
        names.append("theorem.tile")
        _FEATURE_GROUPS = (tuple(names), np.stack(cols))
    return _FEATURE_GROUPS


def feature_drop_groups(cfg, pid, chunk_index, n_groups):
    """Group mask [G] of one perspective chunk, a pure function of (seed, pid, chunk index):
    all groups with p = feature_dropout_all, else each group independently with p =
    feature_dropout.  Same mask for every step of the chunk; exact under resume."""
    rng = np.random.default_rng([int(cfg.seed), 0xFD, int(pid), int(chunk_index)])
    every = rng.random() < cfg.feature_dropout_all
    each = rng.random(n_groups) < cfg.feature_dropout
    return np.ones(n_groups, np.bool_) if every else each


def feature_drop_batch(groups):
    """[B, G] group masks -> model batch['feat_drop'] {"cols": [B, K, F], "tile": [B]} (CPU)."""
    _, cols = feature_groups()
    g = np.asarray(groups, np.bool_)
    per_col = (g[:, :-1, None, None] & cols[None]).any(1)
    return {"cols": torch.from_numpy(per_col), "tile": torch.from_numpy(g[:, -1].copy())}


def feature_dropout_on(cfg, net):
    return bool(getattr(net.config, "features", False)) and (cfg.feature_dropout > 0 or cfg.feature_dropout_all > 0)


# ----------------------------------------------------------------------------- copy dropout
def copy_drop_mask(cfg, pid, chunk_index):
    """Copy dropout of one perspective chunk: True with p = copy_dropout, a pure function of
    (seed, pid, chunk index) (exact under resume); the same for every step of the chunk."""
    p = float(getattr(cfg, "copy_dropout", 0.0) or 0.0)
    if p <= 0:
        return False
    return bool(np.random.default_rng([int(cfg.seed), 0xCD, int(pid), int(chunk_index)]).random() < p)


# ----------------------------------------------------------------------------- perspective sources
class SyntheticSource:
    """Deterministic synthetic perspectives: pid -> the same chunks every time (exact resume)."""
    exact_resume = True

    def __init__(self, cfg, codec, *, seed_offset=0, bank=None, ramp=True):
        self.cfg, self.codec = cfg, codec
        self.seed = cfg.seed * 1_000_003 + seed_offset
        self.bank = bank
        self.next_pid = 0
        self.ramp = cfg.synthetic_lag_ramp if ramp else 0
        self.lag = cfg.synthetic_lag           # lag of the newest perspective (logged)

    def lag_of(self, pid):
        if self.ramp and self.cfg.synthetic_lag > 0:
            return min(self.cfg.synthetic_lag, 1 + pid // self.ramp)
        return self.cfg.synthetic_lag

    def _bank(self):
        if self.bank is None:
            self.bank = observation_bank(worlds=self.cfg.synthetic_worlds, steps=self.cfg.synthetic_bank_steps,
                                         seed=self.cfg.seed)
        return self.bank

    def perspective(self, pid):
        self.lag = self.lag_of(pid)
        return SyntheticPerspective(pid, self.seed + pid, self._bank(), self.codec, self.cfg, lag=self.lag)

    def next(self):
        pid = self.next_pid
        self.next_pid += 1
        return pid, (0, pid), self.perspective(pid)

    def position(self):
        return {"next_pid": self.next_pid}

    def restore(self, pos, wanted):
        """wanted: pid -> stream position; returns pid -> perspective."""
        self.next_pid = pos["next_pid"]
        return {pid: self.perspective(pid) for pid in wanted}


class RealSource:
    """data.iter_perspectives, reopened with a new shuffle seed per epoch.  Perspectives are
    numbered in stream order; with workers=0 the order is deterministic, so a resume can
    rebuild the exact pool by replaying the stream (``exact_resume``)."""

    def __init__(self, cfg, *, split, seed_salt=0):
        self.cfg, self.split, self.salt = cfg, split, seed_salt
        self.exact_resume = cfg.workers == 0
        self.epoch, self.index, self.next_pid = 0, 0, 0
        self.stats = {}
        self._it = None

    def _seed(self, epoch):
        return self.cfg.seed * 1_000_003 + epoch * 7919 + self.salt

    def _open(self, epoch):
        D = _data()
        fn = D.iter_perspectives
        kw = dict(split=self.split, strength_top=self.cfg.strength_top, seats=_seats(self.cfg.seats),
                  workers=self.cfg.workers, shuffle_seed=self._seed(epoch))
        params = inspect.signature(fn).parameters
        if self.cfg.max_episodes is not None and "max_episodes" in params:
            kw["max_episodes"] = self.cfg.max_episodes
        if "stats" in params:
            kw["stats"] = self.stats
        kw.update(data_rule_kwargs(self.cfg, params))
        return iter(fn(self.cfg.archives or None, **kw))

    def population(self):
        """Header-only metas of the split that pass the same data rules (Curriculum.fit)."""
        D = _data()
        tapes = tape_labels_of(self.cfg)
        zips = [p for p in (self.cfg.archives or D.default_archives()) if not Path(p).is_dir()]
        if not zips:                       # pretraining batches only: undated, never in the final set
            return []
        metas, _ = D.header_metas(zips, split=self.split, workers=max(1, self.cfg.workers), tape_labels=tapes)
        return [m for m in metas if rule_reason(self.cfg, m) is None]

    def next(self):
        if self._it is None:
            self._it = self._open(self.epoch)
        for _ in range(2):
            p = next(self._it, None)
            if p is not None:
                out = (self.next_pid, (self.epoch, self.index), p)
                self.next_pid += 1
                self.index += 1
                return out
            if self.index == 0:
                break
            self.epoch, self.index = self.epoch + 1, 0
            self._it = self._open(self.epoch)
        raise RuntimeError(f"data.iter_perspectives({self.split}) yielded no perspectives; stats={self.stats}")

    def position(self):
        return {"epoch": self.epoch, "index": self.index, "next_pid": self.next_pid}

    def close(self):
        """Stop the underlying stream (its worker processes exit)."""
        it, self._it = self._it, None
        close = getattr(it, "close", None)
        if close is not None:
            close()

    def restore(self, pos, wanted):
        """wanted: pid -> (epoch, index).  Replays the stream; returns pid -> Perspective."""
        by_pos = {tuple(v): pid for pid, v in wanted.items()}
        found = {}
        last = (pos["epoch"], pos["index"])
        for epoch in range(min([e for e, _ in by_pos] + [last[0]]), last[0] + 1):
            it = self._open(epoch)
            stop = last[1] if epoch == last[0] else None
            i = 0
            while stop is None or i < stop:
                p = next(it, None)
                if p is None:
                    break
                if (epoch, i) in by_pos:
                    found[by_pos[(epoch, i)]] = p
                i += 1
            if epoch == last[0]:
                self._it = it
        self.epoch, self.index, self.next_pid = last[0], last[1], pos["next_pid"]
        return found


def _seats(seats):
    return int(seats) if str(seats) in ("0", "1") else seats


# ----------------------------------------------------------------------------- data rules / step cache
_TAPES = {}


def tape_labels_of(cfg):
    """data.TapeLabels for cfg (one load per process and path); None when the file is absent
    and allowed (allow_no_tape_labels) or not needed. A missing file with exclude_static_tapes
    on and allow_no_tape_labels off raises (data.load_tape_labels)."""
    key = (cfg.tape_labels or None, cfg.exclude_static_tapes and not cfg.allow_no_tape_labels)
    if key not in _TAPES:
        _TAPES[key] = _data().load_tape_labels(key[0], required=key[1])
    return _TAPES[key]


def data_rule_kwargs(cfg, params):
    """data.iter_perspectives keyword arguments of the v3.1 data rules; a rule that is on but
    not supported by data.py is refused (never silently trained without it)."""
    want = {"wins_only": cfg.wins_only, "exclude_static_tapes": cfg.exclude_static_tapes,
            "allow_no_tape_labels": cfg.allow_no_tape_labels}
    kw = {}
    for name, value in want.items():
        if name in params:
            kw[name] = value
        elif value:
            raise RuntimeError(f"data.iter_perspectives has no {name}= (required by the train config)")
    if "tape_labels" in params:
        kw["tape_labels"] = tape_labels_of(cfg)
    return kw


_STRENGTH_THR = {}


def rule_reason(cfg, meta):
    """data.filter_reason with this config's rules (wins_only, static tapes, strength_top)."""
    D = _data()
    top = cfg.strength_top
    if top not in _STRENGTH_THR:
        _STRENGTH_THR[top] = D.load_strength(None).threshold(top) if top is not None else -math.inf
    source = meta.get("source", "kaggle")
    thr = _STRENGTH_THR[top] if source == "kaggle" else -math.inf
    return D.filter_reason(meta, wins_only=cfg.wins_only, exclude_static_tapes=cfg.exclude_static_tapes,
                           strength_threshold=thr)


_BUILD_CACHE = None


def build_cache_module():
    """pbc-step-cache/build_cache.py (CachedPerspectives: manifest, source-pin check, data rules),
    imported as ``build_cache`` (its directory appended to sys.path, so spawned workers find it)."""
    global _BUILD_CACHE
    if _BUILD_CACHE is None:
        d = str(HERE.parent / "pbc-step-cache")
        if d not in sys.path:
            sys.path.append(d)
        import build_cache
        _BUILD_CACHE = build_cache
    return _BUILD_CACHE


class _LazyPerspective:
    """A cached perspective whose blobs are read on first use (a curriculum-rejected candidate
    is never read)."""
    __slots__ = ("meta", "_cp", "_i", "_p")

    def __init__(self, meta, cp, i):
        self.meta, self._cp, self._i, self._p = meta, cp, i, None

    def _get(self):
        if self._p is None:
            self._p = self._cp.load(self._i)
        return self._p

    def chunks(self, W=24, start=0):
        return self._get().chunks(W, start)

    def __len__(self):
        return len(self._get())


class CacheSource:
    """Perspectives of one split of a step cache (pbc-step-cache/build_cache.py
    CachedPerspectives) instead of live workers. CachedPerspectives refuses a cache whose
    source pins differ and applies the same data rules as the live stream to every cached meta
    (wins_only, static tapes re-read from the current labels, strength_top). Epoch e visits
    the kept perspectives in a random.Random(seed, epoch) permutation, so the stream is
    deterministic and a resume restores the exact pool (positions (epoch, index)). Blobs are
    read when a slot first takes a chunk."""
    exact_resume = True

    def __init__(self, cfg, directory, *, split, seed_salt=0):
        BC = build_cache_module()
        self.cfg, self.split, self.salt = cfg, split, seed_salt
        self.cp = BC.CachedPerspectives(directory, split, wins_only=cfg.wins_only,
                                        exclude_static_tapes=cfg.exclude_static_tapes,
                                        strength_top=cfg.strength_top, tape_labels=cfg.tape_labels or None,
                                        allow_no_tape_labels=cfg.allow_no_tape_labels)
        if not len(self.cp):
            raise RuntimeError(f"step cache {directory} split {split}: no perspective passes the data rules; "
                               f"{self.cp.stats}")
        self.stats = self.cp.stats
        self.metas = self.cp.metas
        self.epoch, self.index, self.next_pid = 0, 0, 0
        self._order = None

    def _perm(self, epoch):
        order = list(range(len(self.metas)))
        random.Random(self.cfg.seed * 1_000_003 + epoch * 7919 + self.salt).shuffle(order)
        return order

    def next(self):
        if self._order is None:
            self._order = self._perm(self.epoch)
        if self.index >= len(self._order):
            self.epoch, self.index = self.epoch + 1, 0
            self._order = self._perm(self.epoch)
        i = self._order[self.index]
        out = (self.next_pid, (self.epoch, self.index), _LazyPerspective(self.metas[i], self.cp, i))
        self.next_pid += 1
        self.index += 1
        return out

    def population(self):
        return list(self.metas)

    def position(self):
        return {"epoch": self.epoch, "index": self.index, "next_pid": self.next_pid}

    def restore(self, pos, wanted):
        """wanted: pid -> (epoch, index); returns pid -> perspective."""
        perms = {}
        found = {}
        for pid, (epoch, index) in wanted.items():
            if epoch not in perms:
                perms[epoch] = self._perm(epoch)
            i = perms[epoch][index]
            found[pid] = _LazyPerspective(self.metas[i], self.cp, i)
        self.epoch, self.index, self.next_pid = pos["epoch"], pos["index"], pos["next_pid"]
        self._order = self._perm(self.epoch)
        return found

    def close(self):
        pass


class CurriculumSource:
    """Training curriculum at pool refill (owner 2026-09-23): every perspective the inner
    source yields is accepted with probability data.Curriculum.weight(meta, f) in [0, 1],
    f = training progress set by the pool before each refill (uniform until 0.5, tilt toward
    high strength deciles and recent days, final set only from 0.8; Curriculum.fit widens the
    final set when the population holds too few). After max_rejects consecutive rejections,
    Curriculum.pick chooses among the rejected candidates (weights, then full tilt, then
    uniform). Deterministic given the seed: its random.Random is saved with the pool. The
    realised distribution (accepted by strength_decile and by day) is counted per log window
    and in total."""

    def __init__(self, inner, curriculum, *, seed, max_rejects=64):
        self.inner, self.curriculum = inner, curriculum
        self.rng = random.Random(seed * 1_000_003 + 0xC0)
        self.max_rejects = max(1, int(max_rejects))
        self.f = 0.0
        self.exact_resume = inner.exact_resume
        self.total = {"candidates": 0, "accepted": 0, "fallbacks": 0, "decile": {}, "day": {}}
        self.window = _curriculum_counts()

    # delegated attributes (synthetic bank, source lag, data stats)
    bank = property(lambda self: getattr(self.inner, "bank", None))
    lag = property(lambda self: getattr(self.inner, "lag", None))
    stats = property(lambda self: getattr(self.inner, "stats", None))

    def set_progress(self, f):
        self.f = min(1.0, max(0.0, float(f)))

    def _count(self, meta, fallback):
        for c in (self.total, self.window):
            c["accepted"] += 1
            c["fallbacks"] += int(fallback)
            for key, value in (("decile", meta.get("strength_decile")), ("day", meta.get("day"))):
                c[key][str(value)] = c[key].get(str(value), 0) + 1

    def next(self):
        rejected = []
        while True:
            opened = self.inner.next()
            meta = opened[2].meta
            w = float(self.curriculum.weight(meta, self.f))
            self.total["candidates"] += 1
            self.window["candidates"] += 1
            if w >= 1.0 or self.rng.random() < w:
                self._count(meta, False)
                return opened
            rejected.append(opened)
            if len(rejected) >= self.max_rejects:
                k = self.curriculum.pick([o[2].meta for o in rejected], self.f, self.rng)[0]
                self._count(rejected[k][2].meta, True)
                return rejected[k]

    def drain(self):
        """Counts since the last drain (log window), with the progress f."""
        out, self.window = self.window, _curriculum_counts()
        out["f"] = self.f
        return out

    def position(self):
        return {"inner": self.inner.position(), "rng": self.rng.getstate(), "f": self.f,
                "total": json.loads(json.dumps(self.total))}

    def restore(self, pos, wanted):
        found = self.inner.restore(pos["inner"], wanted)
        self.rng.setstate(_tuplify(pos["rng"]))
        self.f = pos.get("f", 0.0)
        self.total = pos.get("total", self.total)
        return found

    def close(self):
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()


def _curriculum_counts():
    return {"candidates": 0, "accepted": 0, "fallbacks": 0, "decile": {}, "day": {}}


def make_source(cfg, *, split, train=True, seed_salt=0, bank=None, codec=None, log=print):
    """Perspective source: synthetic (--tiny), step cache (--cache DIR) or live workers
    (data.iter_perspectives, the default). Training sources get the curriculum (cfg.curriculum;
    not for synthetic data, whose perspectives carry no meta). -> (source, curriculum info)."""
    if cfg.synthetic:
        return SyntheticSource(cfg, codec, seed_offset=seed_salt, bank=bank, ramp=train), None
    src = (CacheSource(cfg, cfg.cache, split=split, seed_salt=seed_salt) if cfg.cache
           else RealSource(cfg, split=split, seed_salt=seed_salt))
    if not (train and cfg.curriculum):
        return src, None
    t0 = time.time()
    metas = src.population()
    cur = _data().Curriculum.fit(metas, min_final=cfg.curriculum_min_final)
    info = {"population": len(metas), "final_set": sum(bool(cur.in_final(m)) for m in metas),
            "curriculum": cur.describe(), "seconds": round(time.time() - t0, 1),
            "expected_by_f": {str(f): _expected(cur, metas, f) for f in (0.0, 0.5, 0.65, 0.8, 1.0)}}
    log(json.dumps({"event": "curriculum", **{k: v for k, v in info.items() if k != "expected_by_f"}}))
    return CurriculumSource(src, cur, seed=cfg.seed, max_rejects=cfg.curriculum_max_rejects), info


def _expected(cur, metas, f):
    """Expected accepted share by strength decile and by day at progress f (population)."""
    by_dec, by_day = {}, {}
    total = 0.0
    for m in metas:
        w = cur.weight(m, f)
        total += w
        by_dec[str(m.get("strength_decile"))] = by_dec.get(str(m.get("strength_decile")), 0.0) + w
        by_day[str(m.get("day"))] = by_day.get(str(m.get("day")), 0.0) + w
    if not total:
        return {"mass": 0.0}
    return {"mass": total / max(1, len(metas)), "decile": {k: v / total for k, v in sorted(by_dec.items())},
            "day": {k: v / total for k, v in sorted(by_day.items())}}


# ----------------------------------------------------------------------------- pool
class Slot:
    __slots__ = ("perspective", "chunks", "pending", "state", "chunk_index", "last_update", "pid", "pos")

    def __init__(self, opened, W, state):
        self.pid, self.pos, self.perspective = opened
        self.chunks = iter(self.perspective.chunks(W))
        self.pending = next(self.chunks, None)
        self.state, self.chunk_index, self.last_update = state, 0, -1

    def take(self):
        chunk, self.pending = self.pending, None
        self.chunk_index += 1
        self.pending = next(self.chunks, None)
        return chunk

    def skip(self, n):
        for _ in range(n):
            self.take()

    @property
    def done(self):
        return self.pending is None


class Pool:
    """Open perspectives with their next chunk and detached carried state (one row each).

    Slot schedule: --lockstep (every slot, every update); round_robin (default): a fixed
    permutation of the slot indices, drawn once from the seed and never reshuffled, is walked
    cyclically ``batch`` slots at a time, so with pool divisible by batch every slot is advanced
    exactly once every pool/batch updates (state age = pool/batch; slot 103: random sampling
    gave no bound); otherwise the reuse interval is floor or ceil of pool/batch; else a random
    ``batch`` of slots per update (pool_rng). A finished perspective is replaced in its slot;
    the curriculum source (if any) is told the training progress before each refill."""

    def __init__(self, source, net, cfg, rng, device):
        self.source, self.net, self.cfg, self.rng, self.device = source, net, cfg, rng, device
        self.size = cfg.batch if cfg.lockstep else cfg.pool
        self.slots = []
        self.finished = 0
        self.fresh_state = None
        self.round_robin = bool(getattr(cfg, "round_robin", False)) and not cfg.lockstep
        self.order, self.cursor = None, 0
        self.progress_update = 0
        if self.round_robin:
            self.order = list(range(self.size))
            random.Random(cfg.seed * 1_000_003 + 0x5107).shuffle(self.order)

    @property
    def age_bound(self):
        """Largest state age of a re-used slot: ceil(pool / batch) updates for round_robin (every
        age equals pool/batch when batch divides pool), 1 for lockstep, None (no bound) when random."""
        if self.cfg.lockstep:
            return 1
        return -(-self.size // min(self.cfg.batch, self.size)) if self.round_robin else None

    def _fresh(self):
        if self.fresh_state is None:
            init = self.net.initial_state(1)
            self.fresh_state = ("none",) if init is None else ("state", tree_rows(init, 1)[0])
        return self.fresh_state[1] if self.fresh_state[0] == "state" else None

    def _progress(self, completed_updates):
        set_progress = getattr(self.source, "set_progress", None)
        if set_progress is not None:
            set_progress(completed_updates / max(1, self.cfg.updates))

    def _new_slot(self):
        while True:
            slot = Slot(self.source.next(), self.cfg.chunk, self._fresh())
            if not slot.done:
                return slot

    def fill(self):
        if len(self.slots) < self.size:
            self._progress(self.progress_update)
        while len(self.slots) < self.size:
            self.slots.append(self._new_slot())

    def sample(self):
        self.fill()
        if self.cfg.lockstep:
            return list(range(len(self.slots)))
        if self.round_robin:
            n = min(self.cfg.batch, self.size)
            out = [self.order[(self.cursor + k) % self.size] for k in range(n)]
            self.cursor = (self.cursor + n) % self.size
            return sorted(out)
        return sorted(self.rng.sample(range(len(self.slots)), min(self.cfg.batch, len(self.slots))))

    def ages(self, idx, update):
        """State age (updates since the slot's state was written) of each re-used slot in idx."""
        return [update - self.slots[i].last_update for i in idx if self.slots[i].last_update >= 0]

    def batch_state(self, idx):
        states = [self.slots[i].state for i in idx]
        return None if states[0] is None else tree_to(tree_cat(states), self.device)

    def write_back(self, idx, state, update):
        rows = tree_rows(state, len(idx)) if state is not None else [None] * len(idx)
        self._progress(update + 1)
        for i, st in zip(idx, rows):
            slot = self.slots[i]
            slot.state, slot.last_update = st, update
            if slot.done:
                self.finished += 1
                self.slots[i] = self._new_slot()

    def state_dict(self):
        return {"slots": [{"pid": s.pid, "stream_pos": list(s.pos),
                           "chunk_index": s.chunk_index, "last_update": s.last_update, "state": s.state}
                          for s in self.slots],
                "source": self.source.position(), "finished": self.finished,
                "round_robin": {"order": self.order, "cursor": self.cursor} if self.round_robin else None}

    def load_state_dict(self, blob):
        wanted = {s["pid"]: tuple(s["stream_pos"]) for s in blob["slots"]}
        found = self.source.restore(blob["source"], wanted)
        self.slots = []
        for s in blob["slots"]:
            slot = Slot((s["pid"], tuple(s["stream_pos"]), found[s["pid"]]), self.cfg.chunk, s["state"])
            slot.skip(s["chunk_index"])
            slot.last_update = s["last_update"]
            self.slots.append(slot)
        self.finished = blob["finished"]
        rr = blob.get("round_robin")
        if self.round_robin and rr:
            self.order, self.cursor = list(rr["order"]), int(rr["cursor"])


# ----------------------------------------------------------------------------- synthetic data
_BANK_CACHE = {}


def observation_bank(*, worlds=2, steps=48, seed=0):
    """Real engine observations (random grammar actions for both seats) through the agent's
    input path (canonical observation, pack, FeatureState, feature tokens): per seat, a list
    of example dicts without labels.  Cached per process."""
    key = (worlds, steps, seed)
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]
    import agent as A
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    bank = []
    for w in range(worlds):
        env = ExactKaggricultureEnvironment(seed=seed * 1000 + 17 + w)
        stubs = [A.StubPolicy(seed * 31 + 2 * w + s, max_units=13) for s in range(2)]
        seats = [A.InputBuilder(env.configuration) for _ in range(2)]
        per_seat = [[], []]
        for _ in range(steps):
            actions = []
            for s in range(2):
                clean = A.actor_view(env.agent_observation(s))
                canon, inputs = seats[s].inputs(clean)
                action, _ = stubs[s].act(canon)
                action = A.sanitize_action(action)
                seats[s].record(canon, action)
                per_seat[s].append(inputs)
                actions.append(action)
            env.step(actions)
        bank.extend(per_seat)
    _BANK_CACHE[key] = bank
    return bank


class SyntheticPerspective:
    """A learnable toy perspective over real observations.

    Each step t shows a fresh cue v_t in {0..V-1} as a one-hot, at step t only, in unused
    payload columns (60..60+V, known) of every TILE row (synthetic_cue='tiles'), every MARKET
    and STOCK row ('market'), or the last V columns of every feature token ('features'); the
    GLOBAL row carries it only with synthetic_cue_global.  Labels: unit u -> unit choice (v + u)
    with v = v_{t-lag} ('lag'; PASS for t < lag) or v_t ('react'), or PASS ('pass'); a quantity
    token gets class v, or the QTY_EXACT spelling of 100 + v on odd units; order slot 0 -> order
    choice v_{t-lag} (t >= lag, meta.lag_label), then STOP; t < lag: STOP only."""

    CUE_COL = 60
    GLOBAL_COL, TILE_COL, MARKET_COL, STOCK_COL = 0, 2, 4, 5    # observation.KINDS indices

    @classmethod
    def cue_rows(cls, rows, cfg):
        """[N] bool: the pack rows that show the cue (payload columns 60.. are unused by these kinds)."""
        at = np.zeros(len(rows), np.bool_)
        if cfg.synthetic_cue == "tiles":
            at |= rows[:, cls.TILE_COL] > 0.5
        elif cfg.synthetic_cue == "market":
            at |= (rows[:, cls.MARKET_COL] > 0.5) | (rows[:, cls.STOCK_COL] > 0.5)
        elif cfg.synthetic_cue != "features":
            raise ValueError(f"synthetic_cue {cfg.synthetic_cue!r}: tiles | market | features")
        if cfg.synthetic_cue_global:
            at |= rows[:, cls.GLOBAL_COL] > 0.5
        return at

    def __init__(self, pid, seed, bank, codec, cfg, lag=None):
        self.pid, self.seed, self.bank, self.codec, self.cfg = pid, seed, bank, codec, cfg
        self.lag = cfg.synthetic_lag if lag is None else lag
        rng = np.random.default_rng(seed)
        self.track = bank[int(rng.integers(len(bank)))]
        self.offset = int(rng.integers(len(self.track)))
        self.cues = rng.integers(0, cfg.synthetic_values, size=cfg.synthetic_len)
        self.team = int(rng.integers(1, 4))
        c = codec
        plain = [i for i, t in enumerate(c.UNIT_VOCAB) if not c.UNIT_HAS_QTY[i]]
        qty = [i for i, t in enumerate(c.UNIT_VOCAB) if c.UNIT_HAS_QTY[i]]
        self.unit_choices = (plain[:3] + qty[:1]) if qty else plain[:4]
        stop = c.ORDER_VOCAB.index("STOP")
        noop = c.ORDER_VOCAB.index("NOOP") if "NOOP" in c.ORDER_VOCAB else None
        plain_o = [i for i, t in enumerate(c.ORDER_VOCAB) if i not in (stop, noop) and not c.ORDER_HAS_QTY[i]]
        qty_o = [i for i, t in enumerate(c.ORDER_VOCAB) if c.ORDER_HAS_QTY[i]]
        self.order_choices = (qty_o[:max(1, cfg.synthetic_values // 2)] + plain_o)[:cfg.synthetic_values]
        if len(self.order_choices) < cfg.synthetic_values:
            raise ValueError("order vocabulary smaller than the synthetic cue alphabet")
        self.stop = stop

    def chunks(self, W=24):
        for lo in range(0, self.cfg.synthetic_len, W):
            yield [self.example(t) for t in range(lo, min(lo + W, self.cfg.synthetic_len))]

    def _qty(self, has, value, exact_digits_of=None):
        c = self.codec
        S = getattr(c, "DIGIT_SLOTS", getattr(c, "MAX_DIGITS", 5) + 1)
        digits = [-1] * S
        if not has:
            return -1, digits
        if exact_digits_of is not None and hasattr(c, "QTY_EXACT"):
            return int(c.QTY_EXACT), list(c.spell_digits(exact_digits_of))
        return int(value), digits

    def labels(self, t, n_units):
        c, V, lag = self.codec, self.cfg.synthetic_values, self.lag
        U = int(c.MAX_UNITS)
        S = getattr(c, "DIGIT_SLOTS", getattr(c, "MAX_DIGITS", 5) + 1)
        O = int(getattr(c, "N_ORDER_SLOTS", c.MAX_ORDERS + 1))
        v = int(self.cues[t])
        ut, uq, ud = [-1] * U, [-1] * U, [[-1] * S for _ in range(U)]
        mode = self.cfg.synthetic_units
        if mode == "lag":
            v = int(self.cues[t - lag]) if lag >= 0 and t >= lag else None
        for u in range(min(n_units, U)):
            if mode == "pass" or v is None:
                tok = int(getattr(c, "PASS", 0))
            else:
                tok = self.unit_choices[(v + u) % len(self.unit_choices)]
            ut[u] = tok
            uq[u], ud[u] = self._qty(c.UNIT_HAS_QTY[tok], v or 0, 100 + (v or 0) if u % 2 else None)
        ot, oq, od = [-1] * O, [-1] * O, [[-1] * S for _ in range(O)]
        k = 0
        if lag >= 0 and t >= lag:
            w = int(self.cues[t - lag])
            tok = self.order_choices[w]
            ot[0] = tok
            oq[0], od[0] = self._qty(c.ORDER_HAS_QTY[tok], w, 200 + w if w % 2 else None)
            k = 1
        ot[k] = self.stop
        return {"unit_tok": ut, "unit_qty": uq, "unit_digits": ud, "order_tok": ot, "order_qty": oq,
                "order_digits": od, "exact": True}

    def _n_units(self, t):
        base = self.track[(self.offset + t) % len(self.track)]
        return int((np.asarray(base["unit_pos"])[:, 0] >= 0).sum())

    def example(self, t):
        base = self.track[(self.offset + t) % len(self.track)]
        ex = {k: v for k, v in base.items() if k not in ("meta",)}
        rows, known = np.array(base["rows"], copy=True), np.array(base["knownness"], copy=True)
        at = self.cue_rows(rows, self.cfg)
        V = self.cfg.synthetic_values
        rows[at, self.CUE_COL:self.CUE_COL + V] = 0
        rows[at, self.CUE_COL + int(self.cues[t])] = 1
        known[at, self.CUE_COL:self.CUE_COL + V] = True
        ex["rows"], ex["knownness"] = rows, known
        if self.cfg.synthetic_cue == "features":           # every feature token, last payload columns
            feat = dict(base["feat"])
            fv, fk = np.array(feat["values"], copy=True), np.array(feat["known"], copy=True)
            col = fv.shape[1] - V
            fv[:, col:] = 0
            fv[:, col + int(self.cues[t])] = 1
            fk[:, col:] = True
            feat["values"], feat["known"] = fv, fk
            ex["feat"] = feat
        n_units = self._n_units(t)
        ex["labels"] = self.labels(t, n_units)
        U = int(self.codec.MAX_UNITS)
        prev = np.full(U, -1, np.int64)            # data.prev_unit_tokens rule
        if t > 0:
            n = 1 if t % 24 == 0 else n_units
            prev[:n] = self.labels(t - 1, self._n_units(t - 1))["unit_tok"][:n]
        ex["prev_unit_tok"] = prev
        ex["loss_mask"] = True
        lag = self.lag
        ex["meta"] = {"episode_id": int(self.pid), "seat": 0, "step": t, "hour": t % 24, "day": t // 24,
                      "team_id": self.team, "strength_decile": 9, "lag": lag,
                      "lag_label": bool(lag >= 0 and t >= lag),
                      "lag_units": bool(self.cfg.synthetic_units == "lag" and lag >= 0 and t >= lag)}
        return ex


# ----------------------------------------------------------------------------- checkpoints
def _write_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(out_dir, net, opt, cfg, *, update, pool, pool_rng, metrics, run):
    out_dir = Path(out_dir)
    blob = {"schema": CKPT_SCHEMA, "config": net.config.to_dict(), "train_config": dataclasses.asdict(cfg),
            "model": net.state_dict(), "optimizer": opt.state_dict(), "update": update,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "pool_rng": pool_rng.getstate(), "pool": pool.state_dict(), "lr": lr_at(update, cfg),
            "metrics": metrics, "run": run}
    path = out_dir / f"ckpt-{update:07d}.pt"
    _write_atomic(blob, path)
    (out_dir / "LATEST.tmp").write_text(path.name + "\n")
    os.replace(out_dir / "LATEST.tmp", out_dir / "LATEST")
    for p in sorted(out_dir.glob("ckpt-*.pt"))[:-max(1, cfg.keep_ckpts)]:
        p.unlink()
    return path


def load_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        path = path / (path / "LATEST").read_text().strip()
    blob = torch.load(str(path), map_location="cpu", weights_only=True)
    if blob.get("schema") != CKPT_SCHEMA:
        raise ValueError(f"{path}: not a {CKPT_SCHEMA} checkpoint")
    return blob, path


# ----------------------------------------------------------------------------- build
def model_config_fields():
    return {f.name for f in dataclasses.fields(_model().ModelConfig)}


def model_config(**kw):
    """ModelConfig with the given overrides (unknown names are refused)."""
    names = model_config_fields()
    unknown = sorted(set(kw) - names)
    if unknown:
        raise ValueError(f"ModelConfig has no field(s) {unknown}")
    return _model().ModelConfig(**kw)


TINY = dict(trunk_latents=8, trunk_width=64, trunk_layers=2, trunk_hidden=128, tok_width=64, field_dc=16,
            n_slots=8, dec_width=64, dec_layers=1, heads=4, trunk_heads=4, dec_heads=4)


def tiny_model_config(**kw):
    """model.SIZES['tiny'] (the model builder's tiny shape) when present, else TINY."""
    names = model_config_fields()
    tiny = getattr(_model(), "SIZES", {}).get("tiny") or TINY
    base = {k: v for k, v in tiny.items() if k in names}
    base.update(kw)
    return model_config(**base)


def build(model_cfg, device, codec=None):
    M = _model()
    params = inspect.signature(M.PolicyNet).parameters
    net = M.PolicyNet(model_cfg, codec=codec) if codec is not None and "codec" in params else M.PolicyNet(model_cfg)
    return net.to(device)


def parameter_count(net):
    return sum(p.numel() for p in net.parameters())


def pass_index_of(net):
    p = getattr(getattr(net, "config", None), "pass_index", None)
    if p is not None:
        return int(p)
    c = _net_codec(net)
    try:
        return int(getattr(c, "PASS", list(c.UNIT_VOCAB).index("PASS")))
    except (ValueError, AttributeError):
        return None


# ----------------------------------------------------------------------------- held-out
def held_out(net, cfg, *, source=None, perspectives=None, steps=None, bank=None):
    """Fixed held-out perspectives: lists of their first ``steps`` example dicts, from step 0."""
    n = perspectives or cfg.eval_perspectives
    steps = cfg.eval_steps if steps is None else steps
    if source is None:          # the data rules apply; the curriculum does not (training only)
        if cfg.synthetic:
            source = SyntheticSource(cfg, _net_codec(net), seed_offset=99_991, bank=bank, ramp=False)
        else:
            ecfg = dataclasses.replace(cfg, seed=cfg.seed + 17)
            source, _ = make_source(ecfg, split=cfg.eval_split, train=False)
    out = []
    try:
        for _ in range(n):
            _, _, p = source.next()
            exs = []
            for chunk in p.chunks(cfg.chunk):
                exs.extend(chunk)
                if steps and len(exs) >= steps:
                    break
            out.append(exs[:steps] if steps else exs)
    finally:
        if hasattr(source, "close"):
            source.close()
    return out


def evaluate(net, persps, cfg, device, *, free_run=None, fsq_levels=None):
    """Held-out metrics: perspectives streamed from step 0 with the carried state, in chunks."""
    was = net.training
    net.eval()
    meter, lag = Meter(), Meter()
    think = getattr(net.config, "think", None)
    levels = fsq_levels or (getattr(net.config, "fsq_levels", 8) if think == "fsq" else None)
    stats = FieldStats(levels)
    free = cfg.eval_free_steps if free_run is None else free_run
    state = initial_state(net, len(persps), device)
    total, n = 0.0, 0.0
    t0 = time.time()
    T = max(len(p) for p in persps)
    with torch.no_grad():
        for lo in range(0, T, cfg.chunk):
            chunks = [p[lo:lo + cfg.chunk] for p in persps]
            live = [i for i, c in enumerate(chunks) if c]
            idx = torch.tensor(live, dtype=torch.long)
            st = state if state is None or len(live) == len(persps) else tree_select(state, idx)
            loss, m, st, _ = run_chunk(net, [chunks[i] for i in live], st, device=device, bf16=cfg.bf16,
                                       qty_weight=cfg.qty_weight, digit_weight=cfg.digit_weight, meter=meter,
                                       field_stats=stats, free_run=max(0, free - lo), pass_index=pass_index_of(net),
                                       lag_meter=lag)
            if st is not None:
                state = st if len(live) == len(persps) else tree_put(state, idx, st)
    rep = meter.report()
    rep.update(lag.report())
    k = rep.get("loss_bearing_steps", 0) or 0
    rep["loss_per_step"] = meter.sums.get("loss_sum", 0.0) / max(1.0, k)
    checks = meter.sums.get("free_state_checks")
    if checks:
        rep["free_state_maxdiff_mean"] = meter.sums["free_state_maxdiff_sum"] / checks
    rep.update(stats.report(cfg.collapse_std))
    rep["eval_perspectives"], rep["eval_seconds"] = len(persps), round(time.time() - t0, 2)
    rep["per_team"] = {key[4:-9]: {"unit_tok": rep.get(f"acc_{key}"), "n_units": rep.get(f"n_{key}"),
                                   "step_exact": rep.get(f"acc_team{key[4:-9]}_step_exact"),
                                   "n_steps": rep.get(f"n_team{key[4:-9]}_step_exact")}
                       for key in list(meter.pairs) if key.startswith("team") and key.endswith("_unit_tok")}
    for key in [k2 for k2 in rep if k2.startswith(("acc_team", "n_team"))]:
        del rep[key]
    if was:
        net.train()
    return rep


def collapse_verdict(rep, cfg, think):
    """Abort reasons at the collapse check, or [] (field/fsq only)."""
    reasons = []
    if think not in ("field", "fsq") or not rep.get("field_steps"):
        return reasons
    if rep["field_steps"] < cfg.collapse_steps:
        reasons.append(f"only {rep['field_steps']} held-out field steps < {cfg.collapse_steps} (check incomplete)")
        return reasons
    if rep["field_const_position_frac"] > cfg.collapse_frac:
        reasons.append(f"{rep['field_const_position_frac']:.3f} of field positions constant (std < {cfg.collapse_std})")
    if think == "fsq" and rep.get("fsq_entropy_bits_mean") is not None and rep["fsq_entropy_bits_mean"] < cfg.fsq_entropy_min:
        reasons.append(f"FSQ entropy {rep['fsq_entropy_bits_mean']:.3f} bits < {cfg.fsq_entropy_min}")
    return reasons


# ----------------------------------------------------------------------------- fit
def fit(cfg: TrainConfig, model_cfg=None, out_dir=".", *, device="cpu", resume=False, codec=None, log=print,
        stop_after=None, held=None, source=None, bank=None):
    """Train to cfg.updates (or stop after ``stop_after`` updates of this call, with a checkpoint).

    Resume restores model, optimizer, RNGs, update count and (when the source is deterministic:
    synthetic, the step cache, or real data with workers=0) the exact pool; otherwise the pool
    restarts from fresh perspectives and RUN-<n>.json records it. An explicit ``source`` is used
    as given (no curriculum wrapper)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    pool_rng = random.Random(cfg.seed * 7 + 3)
    update, metrics, resumes, pool_restored = 0, {}, 0, None
    blob = None
    if resume and (out_dir / "LATEST").exists():
        blob, path = load_checkpoint(out_dir)
        model_cfg = _model().ModelConfig.from_dict(blob["config"])
        resumes = int(blob["run"].get("resumes", 0)) + 1
    else:
        torch.manual_seed(cfg.seed)
    net = build(model_cfg, device, codec)
    opt = make_optimizer(net, cfg)
    dropout = feature_dropout_on(cfg, net)
    if dropout and not hasattr(net, "feat_missing"):
        raise ValueError("feature dropout needs a model with feat_missing=True (MISSING vectors); "
                         "pass --feature-dropout 0 --feature-dropout-all 0 for this model")
    n_groups = len(feature_groups()[0]) if dropout else 0
    copy_dropout = float(getattr(cfg, "copy_dropout", 0.0) or 0.0) > 0
    cur_info = None
    if source is None:
        source, cur_info = make_source(cfg, split=cfg.split, bank=bank, codec=_net_codec(net), log=log)
    pool = Pool(source, net, cfg, pool_rng, device)
    if blob is not None:
        net.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        torch.set_rng_state(blob["torch_rng"])
        if torch.cuda.is_available() and blob.get("cuda_rng"):
            torch.cuda.set_rng_state_all(blob["cuda_rng"])
        pool_rng.setstate(_tuplify(blob["pool_rng"]))
        update, metrics = blob["update"], blob.get("metrics", {})
        if source.exact_resume:
            pool.load_state_dict(blob["pool"])
            pool_restored = "exact"
        else:
            pool_restored = "fresh: stream order is not deterministic (workers > 0); pool restarted from new perspectives"
            src_pos = blob["pool"]["source"]
            src_pos = src_pos.get("inner", src_pos)          # curriculum wrapper
            inner = getattr(source, "inner", source)
            inner.epoch = src_pos.get("epoch", 0) + 1
            inner.next_pid = src_pos.get("next_pid", 0)
        log(json.dumps({"event": "resume", "checkpoint": str(path), "update": update, "pool": pool_restored}))
    pool.progress_update = update
    net.train()
    think = getattr(net.config, "think", None)
    run = {"schema": RUN_SCHEMA, "model_config": net.config.to_dict(), "train_config": dataclasses.asdict(cfg),
           "parameters": parameter_count(net), "device": str(device), "torch": str(torch.__version__),
           "numerics": "bf16 autocast (CUDA) / fp32 master weights" if (device.type == "cuda" and cfg.bf16) else "fp32",
           "optimizer": {"kind": "AdamW", "betas": [cfg.beta1, cfg.beta2], "weight_decay": cfg.weight_decay,
                         "decay_params": sum(p.numel() for p in opt.param_groups[0]["params"]),
                         "no_decay_params": sum(p.numel() for g in opt.param_groups[1:] for p in g["params"]),
                         "schedule": f"linear warmup {cfg.warmup} -> cosine to {cfg.min_lr_frac} x lr at {cfg.updates}",
                         "grad_clip": cfg.grad_clip},
           "resumes": resumes, "started_update": update, "pool_on_resume": pool_restored,
           "pool_mode": ("lockstep" if cfg.lockstep else
                         f"stored-state pool {cfg.pool}, {'round-robin' if pool.round_robin else 'random'} {cfg.batch}"),
           "state_age_bound": pool.age_bound,
           "state_age_exact": bool(pool.round_robin and pool.size % cfg.batch == 0) or cfg.lockstep,
           "data_source": ("synthetic" if cfg.synthetic else f"step cache {cfg.cache}" if cfg.cache
                           else "live data.iter_perspectives"),
           "data_rules": {"wins_only": cfg.wins_only, "exclude_static_tapes": cfg.exclude_static_tapes,
                          "allow_no_tape_labels": cfg.allow_no_tape_labels, "strength_top": cfg.strength_top},
           "data_stats": _data_stats(source),
           "curriculum": cur_info,
           "feature_dropout": ({"groups": list(feature_groups()[0]), "p_group": cfg.feature_dropout,
                                "p_all": cfg.feature_dropout_all, "unit": "chunk"} if dropout else None),
           "copy_dropout": ({"p": cfg.copy_dropout, "unit": "chunk",
                             "channels": ["prev_unit_tok -> -1 (prev-unit tokens, decoder prev_own)",
                                          "teacher token -> team 0 / decile unknown", "FIFO action write -> 0"]}
                            if copy_dropout else None),
           "deferred_metrics": cfg.deferred_metrics,
           "load_average": list(os.getloadavg())}
    (out_dir / f"RUN-{resumes}.json").write_text(json.dumps(run, indent=1, default=str) + "\n")
    if held is None and (cfg.eval_every or cfg.collapse_check_update):
        held = held_out(net, cfg, bank=getattr(source, "bank", None))
    pass_index = pass_index_of(net)
    mlog = (out_dir / "metrics.jsonl").open("a")
    meter, t0, steps_done, ages = Meter(), time.time(), 0, []
    age_max_run = 0
    first_update = update
    aborted = None
    try:
        while update < cfg.updates:
            if stop_after is not None and update - first_update >= stop_after:
                break
            pool.progress_update = update
            idx = pool.sample()
            used = pool.ages(idx, update)
            ages.extend(used)
            age_max_run = max([age_max_run, *used])
            drop = None
            m = Meter(defer=cfg.deferred_metrics)
            if dropout:
                groups = np.stack([feature_drop_groups(cfg, pool.slots[i].pid, pool.slots[i].chunk_index, n_groups)
                                   for i in idx])
                drop = feature_drop_batch(groups)
                m.add("feat_drop_groups", int(groups.sum()))
                m.add("feat_drop_all", int(groups.all(1).sum()))
                m.add("feat_drop_chunks", len(idx))
            cdrop = None
            if copy_dropout:
                cdrop = torch.tensor([copy_drop_mask(cfg, pool.slots[i].pid, pool.slots[i].chunk_index) for i in idx],
                                     dtype=torch.bool)
                m.add("copy_drop_dropped", int(cdrop.sum()))
                m.add("copy_drop_chunks", len(idx))
            chunks = [pool.slots[i].take() for i in idx]
            state = pool.batch_state(idx)
            for g in opt.param_groups:
                g["lr"] = lr_at(update, cfg)
            loss, m, new_state, _ = run_chunk(net, chunks, state, device=device, bf16=cfg.bf16,
                                              qty_weight=cfg.qty_weight, digit_weight=cfg.digit_weight,
                                              pass_index=pass_index, meter=m, feat_drop=drop, copy_drop=cdrop)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            update += 1
            pool.write_back(idx, new_state, update - 1)
            m.add("loss", loss.detach())
            m.add("grad_norm", gnorm.detach())
            m.add("updates_in_log", 1)
            m.materialize()                      # the update's one device -> host transfer of metrics
            steps_done += int(m.sums.get("loss_bearing_steps", 0))
            meter.merge(m)
            if update % cfg.log_every == 0 or update == cfg.updates:
                r = meter.report()
                k = r.pop("updates_in_log")
                r["loss"], r["grad_norm"] = r["loss"] / k, r["grad_norm"] / k
                dt = time.time() - t0
                bound = pool.age_bound
                r.update(update=update, lr=lr_at(update - 1, cfg), steps_per_s=steps_done / max(dt, 1e-9),
                         train_lag=getattr(source, "lag", None),
                         seconds=round(dt, 2), perspectives_finished=pool.finished,
                         state_age_mean=(sum(ages) / len(ages)) if ages else 0.0, state_age_max=max(ages, default=0),
                         state_age_max_run=age_max_run, state_age_bound=bound, state_age_uses=len(ages),
                         state_age_over_bound=(sum(a > bound for a in ages) if bound else None),
                         staleness_mean=(sum(ages) / len(ages) - 1) if ages else 0.0,
                         staleness_max=max(ages, default=1) - 1)
                chunks_n = r.pop("feat_drop_chunks", 0)
                if chunks_n:
                    r["feat_drop_group_rate"] = r.pop("feat_drop_groups") / (chunks_n * n_groups)
                    r["feat_drop_all_rate"] = r.pop("feat_drop_all") / chunks_n
                copy_n = r.pop("copy_drop_chunks", 0)
                if copy_n:
                    r["copy_drop_rate"] = r.pop("copy_drop_dropped") / copy_n
                if hasattr(source, "drain"):
                    r["curriculum"] = source.drain()
                stats = _data_stats(source)
                if stats and not cfg.cache:
                    r["data_stats"] = stats
                if device.type == "cuda":
                    r["cuda_max_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
                mlog.write(json.dumps(r) + "\n")
                mlog.flush()
                log(json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in r.items()
                                if not k2.startswith(("n_", "loss_")) and k2 not in ("curriculum", "data_stats")}))
                metrics = r
                meter, t0, steps_done, ages = Meter(), time.time(), 0, []
            do_eval = cfg.eval_every and update % cfg.eval_every == 0
            do_collapse = cfg.collapse_check_update and update == cfg.collapse_check_update and think in ("field", "fsq")
            if (do_eval or do_collapse) and held:
                ev = evaluate(net, held, cfg, device)
                ev["update"] = update
                mlog.write(json.dumps({"eval": ev}) + "\n")
                mlog.flush()
                log(json.dumps({"eval": {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in ev.items()
                                         if k2.startswith(("acc_", "loss_per", "field_const", "fsq_entropy_bits_mean", "update"))}}))
                metrics = {**metrics, "eval": ev}
                if do_collapse:
                    reasons = collapse_verdict(ev, cfg, think)
                    if reasons and not reasons[0].endswith("(check incomplete)"):
                        aborted = {"update": update, "reasons": reasons, "eval": ev}
                        (out_dir / "ABORT.json").write_text(json.dumps(aborted, indent=1) + "\n")
                        log(json.dumps({"event": "collapse_abort", **{k2: v for k2, v in aborted.items() if k2 != "eval"}}))
            last = (update == cfg.updates or aborted is not None
                    or (stop_after is not None and update - first_update >= stop_after))
            if update % cfg.ckpt_every == 0 or last:
                run["resumes"] = resumes
                save_checkpoint(out_dir, net, opt, cfg, update=update, pool=pool, pool_rng=pool_rng,
                                metrics=metrics, run=run)
            if aborted is not None:
                break
    finally:
        mlog.close()
        if hasattr(source, "close"):
            source.close()
    metrics = {**metrics, "aborted": aborted, "state_age_max_run": age_max_run}
    if hasattr(source, "total"):
        metrics["curriculum_total"] = source.total
    return net, metrics


def _data_stats(source):
    """Data-rule counts of a real source (live: data.iter_perspectives stats; cache:
    CachedPerspectives.stats); None for synthetic data."""
    stats = getattr(source, "stats", None)
    if not isinstance(stats, dict):
        return None
    keep = ("filters", "data_rules", "perspectives", "steps", "literal_episodes")
    return json.loads(json.dumps({k: stats[k] for k in keep if k in stats}, default=str))


def _tuplify(x):
    if isinstance(x, list):
        return tuple(_tuplify(v) for v in x)
    if isinstance(x, tuple):
        return tuple(_tuplify(v) for v in x)
    return x


# ----------------------------------------------------------------------------- gate
def gate(modes, out, *, lag=8, updates=600, jobs=4, extra=(), workdir=None):
    """Synthetic-lag gate: one `fit --tiny --synthetic-lag` subprocess per think mode, then the
    held-out lag accuracy of each (pass: >= 0.9 for field/fsq/carry; off is the no-memory control)."""
    workdir = Path(workdir or Path(out).with_suffix(""))
    workdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    procs, t0 = {}, time.time()
    pending = list(modes)
    results = {}
    while pending or procs:
        while pending and len(procs) < jobs:
            mode = pending.pop(0)
            d = workdir / mode
            cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "fit", "--tiny", "--think", mode,
                   "--synthetic-lag", str(lag), "--updates", str(updates), "--out", str(d), "--device", "cpu",
                   "--threads", "1", *extra]
            d.mkdir(parents=True, exist_ok=True)
            procs[mode] = (subprocess.Popen(cmd, stdout=open(d / "stdout.log", "w"), stderr=subprocess.STDOUT, env=env),
                           time.time())
        time.sleep(2)
        for mode, (p, started) in list(procs.items()):
            if p.poll() is None:
                continue
            del procs[mode]
            d = workdir / mode
            evals = [json.loads(l)["eval"] for l in (d / "metrics.jsonl").read_text().splitlines()
                     if l.startswith('{"eval"')] if (d / "metrics.jsonl").exists() else []
            last = evals[-1] if evals else {}
            results[mode] = {"returncode": p.returncode, "seconds": round(time.time() - started, 1),
                             "lag_acc": last.get("acc_lag"), "n_lag": last.get("n_lag"),
                             "lag_units_acc": last.get("acc_lag_units"), "n_lag_units": last.get("n_lag_units"),
                             "lag_first_acc": last.get("acc_lag_first"), "n_lag_first": last.get("n_lag_first"),
                             "lag_first_by_eval": [(e.get("update"), e.get("acc_lag_first")) for e in evals],
                             "unit_tok_acc": last.get("acc_unit_tok"), "step_exact": last.get("acc_step_exact"),
                             "field_const_position_frac": last.get("field_const_position_frac"),
                             "fsq_entropy_bits_mean": last.get("fsq_entropy_bits_mean"),
                             "lag_acc_by_eval": [(e.get("update"), e.get("acc_lag")) for e in evals],
                             "aborted": (d / "ABORT.json").exists()}
    for mode, r in results.items():
        # gate metric: lag labels no earlier teacher-forced decision of the same step reveals:
        # the farmer under --synthetic-units lag, else order slot 0
        r["gate_metric"] = "lag_first_acc" if r.get("lag_first_acc") is not None else "lag_acc"
        r["pass"] = (r[r["gate_metric"]] or 0) >= 0.9 if mode != "off" else None
    report = {"schema": "primitive-bc-synthetic-lag-gate-v3", "lag": lag, "updates": updates, "modes": results,
              "gate_pass": all(r["pass"] for m, r in results.items() if m != "off"),
              "seconds": round(time.time() - t0, 1), "load_average_end": os.getloadavg(), "extra_args": list(extra)}
    Path(out).write_text(json.dumps(report, indent=1) + "\n")
    return report


# ----------------------------------------------------------------------------- CLI
MODEL_ARGS = ("think", "carry_scope", "trunk_latents", "trunk_width", "trunk_layers", "trunk_hidden",
              "tok_width", "field_dc", "n_slots", "teacher_token", "stock_path")


def _parse_bool(s):
    return str(s).lower() in ("1", "true", "yes", "on")


def _model_cfg_from_args(a):
    over = {}
    for k in MODEL_ARGS:
        v = getattr(a, k, None)
        if v is not None:
            over[k] = v
    for item in a.model_set or []:
        k, _, v = item.partition("=")
        try:
            over[k] = json.loads(v)
        except ValueError:
            over[k] = v
    return tiny_model_config(**over) if a.tiny else model_config(**over)


def _add_train_args(f):
    for fld in dataclasses.fields(TrainConfig):
        if fld.name == "archives":
            continue
        name = "--" + fld.name.replace("_", "-")
        if fld.type in ("bool", bool):
            f.add_argument(name, type=_parse_bool, default=None, nargs="?", const=True)
        elif fld.name == "max_episodes":
            f.add_argument(name, type=int, default=None)
        else:
            typ = {"int": int, "float": float, "str": str}.get(str(fld.type), str)
            f.add_argument(name, type=typ, default=None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--out", required=True)
    f.add_argument("--resume", action="store_true")
    f.add_argument("--stop-after", type=int, default=None, help="stop (with a checkpoint) after this many updates")
    f.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    f.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    f.add_argument("--tiny", action="store_true", help="tiny model + synthetic data (CPU gate)")
    f.add_argument("--archives", nargs="*", default=[])
    f.add_argument("--archive-glob", default=None)
    f.add_argument("--think", choices=THINK_MODES, default=None)
    f.add_argument("--carry-scope", choices=("episode", "day"), default=None)
    for k in ("trunk_latents", "trunk_width", "trunk_layers", "trunk_hidden", "tok_width", "field_dc", "n_slots"):
        f.add_argument("--" + k.replace("_", "-"), type=int, default=None)
    f.add_argument("--teacher-token", type=_parse_bool, default=None)
    f.add_argument("--stock-path", type=_parse_bool, default=None,
                   help="decoder queries + stock_mlp(own stock + cash) (ModelConfig.stock_path; default on)")
    f.add_argument("--model-set", nargs="*", default=None, help="extra ModelConfig overrides key=value (JSON values)")
    _add_train_args(f)
    g = sub.add_parser("gate")
    g.add_argument("--out", required=True)
    g.add_argument("--modes", default="field,fsq,carry,off")
    g.add_argument("--lag", type=int, default=8)
    g.add_argument("--updates", type=int, default=600)
    g.add_argument("--jobs", type=int, default=4)
    g.add_argument("extra", nargs="*", help="extra fit arguments after --")
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out")
    e.add_argument("--device", default="cpu")
    e.add_argument("--split", default=None)
    e.add_argument("--perspectives", type=int, default=None)
    e.add_argument("--steps", type=int, default=None)
    e.add_argument("--workers", type=int, default=None)
    e.add_argument("--cache", default=None, help="held-out perspectives from this step cache")
    a = ap.parse_args(argv)

    if a.cmd == "fit":
        if a.threads:
            torch.set_num_threads(a.threads)
        resuming = a.resume and (Path(a.out) / "LATEST").exists()
        cfg = TrainConfig.from_dict(load_checkpoint(a.out)[0]["train_config"]) if resuming else TrainConfig()
        overrides = {fld.name: getattr(a, fld.name) for fld in dataclasses.fields(TrainConfig)
                     if getattr(a, fld.name, None) is not None and fld.name != "archives"}
        if a.tiny and not resuming:
            tiny = dict(synthetic=True, chunk=24, pool=32, batch=16, updates=600, warmup=50, lr=2e-3, eval_every=200,
                        ckpt_every=200, log_every=25, eval_perspectives=16, eval_steps=48, eval_free_steps=0,
                        collapse_check_update=500, collapse_steps=256, synthetic_len=96, workers=0)
            cfg = dataclasses.replace(cfg, **{k: v for k, v in tiny.items() if k not in overrides})
        cfg = dataclasses.replace(cfg, **overrides)
        archives = list(a.archives)
        if a.archive_glob:
            archives += sorted(_glob.glob(a.archive_glob))
        if archives or not resuming:
            cfg.archives = archives
        model_cfg = None if resuming else _model_cfg_from_args(a)
        _, metrics = fit(cfg, model_cfg, a.out, device=a.device, resume=a.resume, stop_after=a.stop_after)
        return 3 if metrics.get("aborted") else 0
    if a.cmd == "gate":
        rep = gate(a.modes.split(","), a.out, lag=a.lag, updates=a.updates, jobs=a.jobs, extra=tuple(a.extra))
        print(json.dumps(rep, indent=1))
        return 0 if rep["gate_pass"] else 2
    blob, path = load_checkpoint(a.checkpoint)
    net = build(_model().ModelConfig.from_dict(blob["config"]), a.device)
    net.load_state_dict(blob["model"])
    cfg = TrainConfig.from_dict(blob["train_config"])
    if a.split:
        cfg.eval_split = a.split
    if a.workers is not None:
        cfg.workers = a.workers
    if a.cache is not None:
        cfg.cache = a.cache
    held = held_out(net, cfg, perspectives=a.perspectives, steps=a.steps)
    rep = evaluate(net, held, cfg, torch.device(a.device))
    rep.update(checkpoint=str(path), update=blob["update"], split=cfg.eval_split)
    text = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
