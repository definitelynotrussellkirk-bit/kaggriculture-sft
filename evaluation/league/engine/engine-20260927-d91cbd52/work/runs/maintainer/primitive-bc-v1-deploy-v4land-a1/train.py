"""Primitive-BC trainer: data.py windows -> PolicyNet with the carried thinking field.

Each update takes a batch of contiguous per-perspective windows (data.iter_windows:
up to ``burn_in`` no-loss steps, then ``window`` scored steps).  Steps run in time
order; every step's field codes are carried into the next step exactly as act() does
at inference.  Burn-in steps run without gradient (encoder + writer only); scored
steps keep the whole graph, so losses at step t+k backpropagate through the carried
codes to step t (truncated BPTT over the window, no detach inside it).

Commands:
  fit    stream -> AdamW (bf16 autocast only on CUDA), periodic atomic checkpoints,
         --resume, held-out metrics; --tiny = synthetic data + tiny model on CPU.
  eval   held-out metrics of a checkpoint (teacher-forced and free-running greedy).
  bench  parameter count and single-thread CPU latency for the model sizes.
  probe  gradient norm reaching codes written at a window's first step from later
         losses, and activation bytes saved for backward per window.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import model as M  # noqa: E402

CKPT_SCHEMA = "primitive-bc-train-checkpoint-v1"
STEP_KEYS = ("rows", "token_mask", "knownness", "unit_pos", "unit_tok", "unit_qty", "order_tok", "order_qty")


@dataclass
class TrainConfig:
    window: int = 24
    burn_in: int = 24
    stride: int = 0                 # 0 -> window
    batch_windows: int = 16
    updates: int = 20000
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup: int = 500
    weight_decay: float = 0.05
    beta2: float = 0.95
    grad_clip: float = 1.0
    qty_weight: float = 1.0
    tau_start: float = 2.0          # Gumbel-softmax temperature (backward relaxation)
    tau_end: float = 0.5
    tau_anneal_updates: int = 0     # 0 -> updates
    tau_schedule: str = "exp"       # exp | linear
    usage_reg: float = 0.0          # per-position code-usage regulariser; 0 unless collapse is observed
    grad_checkpoint: bool = False
    bf16: bool = True               # autocast bf16 on CUDA only; CPU always fp32
    seed: int = 0
    shuffle_buffer: int = 256
    ckpt_every: int = 500
    keep_ckpts: int = 3
    eval_every: int = 1000
    eval_windows: int = 128
    log_every: int = 20
    # data
    archives: list = dc_field(default_factory=list)
    split: str = "train"
    eval_split: str = "val"
    rank_max: Optional[int] = None
    seats: str = "both"
    workers: int = 4
    max_episodes: Optional[int] = None
    data_resume: str = "reseed"     # reseed | skip (skip replays the stream; exact only with workers=0)
    # synthetic (tiny) data
    synthetic: bool = False
    synthetic_lag: int = 0          # >0: order labels depend on the global value `lag` steps earlier
    synthetic_hands: int = 3
    synthetic_len: int = 24

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# ----------------------------------------------------------------------------- numerics
def autocast(device, enabled=True):
    """bf16 autocast on CUDA only; CPU training is plain fp32."""
    if enabled and torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def tau_at(update, cfg):
    total = cfg.tau_anneal_updates or cfg.updates
    f = min(1.0, update / max(1, total))
    if cfg.tau_schedule == "linear":
        return cfg.tau_start + (cfg.tau_end - cfg.tau_start) * f
    return cfg.tau_start * (cfg.tau_end / cfg.tau_start) ** f


def lr_at(update, cfg):
    if update < cfg.warmup:
        return cfg.lr * (update + 1) / cfg.warmup
    f = min(1.0, (update - cfg.warmup) / max(1, cfg.updates - cfg.warmup))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * f)))


# ----------------------------------------------------------------------------- batches
def collate_windows(windows, max_units, n_order_slots):
    """Shape-generic copy of data.collate_windows (used for synthetic data / stub codecs)."""
    burn = max(w["n_burn"] for w in windows)
    width = max(w["n_scored"] for w in windows)
    T, B = burn + width, len(windows)
    n = max(len(ex["rows"]) for w in windows for ex in w["steps"])
    out = {"rows": torch.zeros(B, T, n, M.IN_WIDTH), "knownness": torch.zeros(B, T, n, M.IN_WIDTH, dtype=torch.bool),
           "token_mask": torch.zeros(B, T, n, dtype=torch.bool),
           "unit_pos": torch.full((B, T, max_units, 2), -1, dtype=torch.long),
           "unit_tok": torch.full((B, T, max_units), -1, dtype=torch.long),
           "unit_qty": torch.full((B, T, max_units), -1, dtype=torch.long),
           "order_tok": torch.full((B, T, n_order_slots), -1, dtype=torch.long),
           "order_qty": torch.full((B, T, n_order_slots), -1, dtype=torch.long),
           "exact": torch.zeros(B, T, dtype=torch.bool), "step_valid": torch.zeros(B, T, dtype=torch.bool),
           "loss_mask": torch.zeros(B, T, dtype=torch.bool), "meta": [w.get("meta", {}) for w in windows]}
    for i, w in enumerate(windows):
        for j, ex in enumerate(w["steps"]):
            col = burn - w["n_burn"] + j
            k = len(ex["rows"])
            out["rows"][i, col, :k] = torch.as_tensor(np.asarray(ex["rows"], np.float32))
            out["knownness"][i, col, :k] = torch.as_tensor(np.asarray(ex["knownness"]))
            out["token_mask"][i, col, :k] = torch.as_tensor(np.asarray(ex["token_mask"]))
            out["unit_pos"][i, col] = torch.as_tensor(np.asarray(ex["unit_pos"]))
            out["unit_tok"][i, col] = torch.as_tensor(ex["unit_tok"])
            out["unit_qty"][i, col] = torch.as_tensor(ex["unit_qty"])
            out["order_tok"][i, col, :len(ex["order_tok"])] = torch.as_tensor(ex["order_tok"])
            out["order_qty"][i, col, :len(ex["order_qty"])] = torch.as_tensor(ex["order_qty"])
            out["exact"][i, col] = bool(ex.get("exact", True))
            out["step_valid"][i, col] = True
            out["loss_mask"][i, col] = col >= burn
    return out


def make_windows(examples, *, window, burn_in, stride=None):
    stride = stride or window
    for start in range(0, len(examples), stride):
        lo = max(0, start - burn_in)
        part = examples[lo:start + window]
        yield {"steps": part, "n_burn": start - lo, "n_scored": len(part) - (start - lo),
               "start_step": start, "from_episode_start": lo == 0, "meta": dict(examples[0]["meta"])}


# ----------------------------------------------------------------------------- synthetic data
def synthetic_episode(seed, *, length, max_units, n_unit_tokens, n_order_tokens, unit_has_qty,
                      order_has_qty, stop_index, hands=3, lag=0, n_qty=75):
    """A learnable toy perspective with the packed-row layout.

    unit label  = f(tile kind under the unit)     (read through the cell codes)
    unit qty    = (x + y) % 7 for quantity tokens (unit position)
    orders      = n = v orders, token g(k, v), qty k (read through the global slots)
    with v the GLOBAL value at step t - lag (lag > 0 needs the carried field)."""
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 4, size=length)
    plain = [t for t in range(n_unit_tokens) if not unit_has_qty[t]]
    qty = [t for t in range(n_unit_tokens) if unit_has_qty[t]]
    unit_choices = plain[:3] + qty[:3]
    order_choices = [t for t in range(n_order_tokens) if t != stop_index][:6]
    examples = []
    n_h = int(rng.integers(0, min(hands, max_units - 1) + 1))
    for t in range(length):
        kinds = rng.integers(0, 6, size=(2, 10, 10))
        units = rng.integers(0, 10, size=(1 + n_h, 2))
        rows, known = [], []

        def add(kind, role=None, xy=None, tile=None, payload=None):
            r = np.zeros(M.IN_WIDTH, np.float32)
            k = np.zeros(M.IN_WIDTH, bool)
            r[kind] = 1
            k[:12] = True
            if role is not None:
                r[12 + role] = 1
                k[12:14] = True
            if xy is not None:
                r[14], r[15] = xy[0] / 9, xy[1] / 9
                k[14:16] = True
            if tile is not None:
                r[29 + tile] = 1
                k[29:35] = True
            if payload is not None:
                r[35] = payload
                k[35] = True
            rows.append(r)
            known.append(k)

        add(0, payload=values[t] / 3)
        for role in (0, 1):
            add(1, role=role)
            for y in range(10):
                for x in range(10):
                    add(M.TILE_KIND_COL, role=role, xy=(x, y), tile=int(kinds[role, y, x]))
            for u in (units if role == 0 else units[:1]):
                add(3, role=role, xy=(int(u[0]), int(u[1])))
        v = int(values[t - lag]) if t >= lag else 0
        unit_tok, unit_qty = [-1] * max_units, [-1] * max_units
        for i, (x, y) in enumerate(units[:max_units]):
            tok = unit_choices[int(kinds[0, y, x]) % len(unit_choices)]
            unit_tok[i] = tok
            unit_qty[i] = (int(x) + int(y)) % 7 if unit_has_qty[tok] else -1
        order_tok, order_qty = [], []
        for k in range(v):
            tok = order_choices[(k + v) % len(order_choices)]
            order_tok.append(tok)
            order_qty.append(k % n_qty if order_has_qty[tok] else -1)
        order_tok.append(stop_index)
        order_qty.append(-1)
        pos = np.full((max_units, 2), -1, np.int64)
        pos[:len(units)] = units[:max_units]
        examples.append({"rows": np.stack(rows).astype(np.float16), "knownness": np.stack(known),
                         "token_mask": np.ones(len(rows), bool), "unit_pos": pos,
                         "unit_tok": unit_tok, "unit_qty": unit_qty, "order_tok": order_tok,
                         "order_qty": order_qty, "exact": True,
                         "meta": {"episode_id": int(seed), "seat": 0, "step": t}})
    return examples


class SyntheticStream:
    """Deterministic, index-addressable window stream (exact resume by window count)."""

    def __init__(self, net_config, cfg, seed_offset=0):
        self.c, self.cfg, self.seed = net_config, cfg, cfg.seed * 1_000_003 + seed_offset
        self.per_episode = math.ceil(cfg.synthetic_len / (cfg.stride or cfg.window))
        self._cache = (None, None)

    def window(self, index):
        episode = index // self.per_episode
        if self._cache[0] != episode:
            c, cfg = self.c, self.cfg
            ex = synthetic_episode(self.seed + episode, length=cfg.synthetic_len, max_units=c.max_units,
                                   n_unit_tokens=c.n_unit_tokens, n_order_tokens=c.n_order_tokens,
                                   unit_has_qty=c.unit_has_qty, order_has_qty=c.order_has_qty,
                                   stop_index=c.stop_index, hands=cfg.synthetic_hands,
                                   lag=cfg.synthetic_lag, n_qty=c.n_qty)
            self._cache = (episode, list(make_windows(ex, window=cfg.window, burn_in=cfg.burn_in,
                                                      stride=cfg.stride or cfg.window)))
        return self._cache[1][index % self.per_episode]

    def batch(self, first, count):
        return [self.window(first + i) for i in range(count)]


class RealStream:
    """data.iter_windows through a seeded shuffle buffer; restarts with a new seed per epoch."""

    def __init__(self, cfg, *, split, epoch=0, resumes=0, skip=0, workers=None):
        import data
        self.data, self.cfg, self.split = data, cfg, split
        self.epoch, self.resumes = epoch, resumes
        self.workers = cfg.workers if workers is None else workers
        self.rng = random.Random(self._seed())
        self.buffer, self.stats = [], {}
        self.consumed = 0
        self._it = self._open()
        for _ in range(skip):
            self.next()

    def _seed(self):
        return self.cfg.seed * 1_000_003 + self.epoch * 7919 + self.resumes * 104_729

    def _open(self):
        cfg = self.cfg
        seats = int(cfg.seats) if str(cfg.seats) in ("0", "1") else cfg.seats
        return iter(self.data.iter_windows(cfg.archives or None, split=self.split, rank_max=cfg.rank_max,
                                           seats=seats, workers=self.workers, window=cfg.window,
                                           burn_in=cfg.burn_in, stride=cfg.stride or None,
                                           shuffle_seed=self._seed(), max_episodes=cfg.max_episodes,
                                           stats=self.stats))

    def _pull(self):
        while True:
            try:
                return next(self._it)
            except StopIteration:
                if not self.buffer and self.consumed == 0:
                    errors = self.stats.get("episode_errors", [])
                    raise RuntimeError(f"data stream ({self.split}) yielded no windows; "
                                       f"{len(errors)} episode errors, first: {errors[:1]}")
                self.epoch += 1
                self.rng = random.Random(self._seed())
                self._it = self._open()
                if not self.buffer:
                    continue
                return None

    def next(self):
        while len(self.buffer) < max(1, self.cfg.shuffle_buffer):
            w = self._pull()
            if w is None:
                break
            self.buffer.append(w)
        i = self.rng.randrange(len(self.buffer))
        self.buffer[i], self.buffer[-1] = self.buffer[-1], self.buffer[i]
        self.consumed += 1
        return self.buffer.pop()

    def batch(self, count):
        return [self.next() for _ in range(count)]


def collate(windows, net):
    if windows and len(windows[0]["steps"][0]["unit_tok"]) == net.U:
        try:
            import data
            if data.codec.MAX_UNITS == net.U and data.N_ORDER_SLOTS == net.O:
                return data.collate_windows(windows)
        except ImportError:
            pass
    return collate_windows(windows, net.U, net.O)


# ----------------------------------------------------------------------------- losses / metrics
COUNT_KEYS = ("unit_tok", "unit_qty", "order_tok", "order_qty", "unit_exact", "order_exact", "step_exact")


def step_loss(out, lab, net, qty_weight=1.0):
    """Per-example summed CE over present units and the order sequence incl. STOP."""
    ut, uq, ot, oq = lab
    ignore = -100
    present = ut >= 0
    has_uq = present & net.unit_qty_mask[ut.clamp(min=0)] & (uq >= 0)
    ovalid = ot >= 0
    has_oq = ovalid & net.order_qty_mask[ot.clamp(min=0)] & (oq >= 0)
    ce = lambda logits, y, m: F.cross_entropy(logits.float().transpose(1, 2), torch.where(m, y, ignore),
                                              ignore_index=ignore, reduction="none").sum(1)
    parts = {"unit_tok": ce(out["unit_logits"], ut, present), "unit_qty": ce(out["unit_qty_logits"], uq, has_uq),
             "order_tok": ce(out["order_logits"], ot, ovalid), "order_qty": ce(out["order_qty_logits"], oq, has_oq)}
    per_example = parts["unit_tok"] + parts["order_tok"] + qty_weight * (parts["unit_qty"] + parts["order_qty"])
    with torch.no_grad():
        pu, pq = out["unit_logits"].argmax(-1), out["unit_qty_logits"].argmax(-1)
        po, poq = out["order_logits"].argmax(-1), out["order_qty_logits"].argmax(-1)
        counts = _exact_counts(pu, pq, po, poq, ut, uq, ot, oq, present, has_uq, ovalid, has_oq)
        counts["label_qty_on_plain"] = (int(((uq >= 0) & present & ~net.unit_qty_mask[ut.clamp(min=0)]).sum())
                                        + int(((oq >= 0) & ovalid & ~net.order_qty_mask[ot.clamp(min=0)]).sum()))
        counts.update({f"loss_{k}": float(v.sum()) for k, v in parts.items()})
    return per_example, counts


def _exact_counts(pu, pq, po, poq, ut, uq, ot, oq, present, has_uq, ovalid, has_oq):
    tok_u, qty_u = (pu == ut) & present, (pq == uq) & has_uq
    tok_o, qty_o = (po == ot) & ovalid, (poq == oq) & has_oq
    unit_ok = tok_u & (qty_u | ~has_uq)
    order_ok = tok_o & (qty_o | ~has_oq)
    step_ok = (unit_ok | ~present).all(1) & (order_ok | ~ovalid).all(1)
    return {"unit_tok": (int(tok_u.sum()), int(present.sum())), "unit_qty": (int(qty_u.sum()), int(has_uq.sum())),
            "order_tok": (int(tok_o.sum()), int(ovalid.sum())), "order_qty": (int(qty_o.sum()), int(has_oq.sum())),
            "unit_exact": (int(unit_ok.sum()), int(present.sum())),
            "order_exact": (int(order_ok.sum()), int(ovalid.sum())),
            "step_exact": (int(step_ok.sum()), int(step_ok.numel()))}


def free_counts(pred, lab):
    """Exact-match counts of a free-running decode against the labels."""
    ut, uq, ot, oq = lab
    present, ovalid = ut >= 0, ot >= 0
    has_uq, has_oq = uq >= 0, oq >= 0
    unit_ok = (pred["unit_tok"] == ut) & (pred["unit_qty"] == uq) & present
    n = ot.shape[1]
    po, poq = pred["order_tok"][:, :n], pred["order_qty"][:, :n]
    order_ok = (po == ot) & (poq == oq) & ovalid
    step_ok = (unit_ok | ~present).all(1) & ((po == ot) & (poq == oq)).all(1)
    return {"free_unit_exact": (int(unit_ok.sum()), int(present.sum())),
            "free_order_exact": (int(order_ok.sum()), int(ovalid.sum())),
            "free_step_exact": (int(step_ok.sum()), int(step_ok.numel()))}


class Meter:
    def __init__(self):
        self.pairs, self.sums = {}, {}

    def add(self, counts):
        for k, v in counts.items():
            if isinstance(v, tuple):
                a, b = self.pairs.get(k, (0, 0))
                self.pairs[k] = (a + v[0], b + v[1])
            else:
                self.sums[k] = self.sums.get(k, 0) + v

    def report(self):
        out = {f"acc_{k}": (a / b if b else None) for k, (a, b) in self.pairs.items()}
        out.update({f"n_{k}": b for k, (a, b) in self.pairs.items()})
        out.update(self.sums)
        return out


class CodeStats:
    """Argmax-code histogram per field position -> usage / per-cell entropy across examples
    (collapse check: an entropy of 0 means that position writes the same code for every input;
    argmax is used so Gumbel sampling noise is not counted as information)."""

    def __init__(self, S, C, K):
        self.counts = torch.zeros(S, C, K, dtype=torch.float64)
        self.soft_entropy, self.n = 0.0, 0

    def add(self, codes, logits):
        top = torch.nn.functional.one_hot(logits.detach().argmax(-1), logits.shape[-1])
        self.counts += top.sum(0).double().cpu()
        p = torch.softmax(logits.detach().float(), -1)
        self.soft_entropy += float(-(p * torch.log2(p.clamp_min(1e-12))).sum(-1).mean()) * len(logits)
        self.n += len(logits)

    def report(self):
        c = self.counts
        tot = c.sum(-1, keepdim=True).clamp_min(1)
        p = c / tot
        ent = -(p * torch.log2(p.clamp_min(1e-12))).sum(-1)          # [S, C] bits
        used = (c.sum((0, 1)) > 0)
        cells, slots = ent[:M.N_CELLS], ent[M.N_CELLS:]
        return {"codes_distinct_used": int(used.sum()), "codes_K": int(c.shape[-1]),
                "code_pairs_used": int((c.sum(0) > 0).sum()), "code_pairs_total": int(c.shape[1] * c.shape[2]),
                "cell_entropy_bits_mean": float(cells.mean()), "slot_entropy_bits_mean": float(slots.mean()) if len(slots) else None,
                "cell_entropy_bits_min": float(cells.min()), "code_softmax_entropy_bits": self.soft_entropy / max(1, self.n),
                "code_samples": self.n}


# ----------------------------------------------------------------------------- window pass
def _to(batch, t, idx, device):
    return {k: batch[k][idx, t].to(device, non_blocking=True) for k in STEP_KEYS}


def run_window_batch(net, batch, *, device, tau=1.0, code_mode="gumbel", generator=None, bf16=True,
                     grad_checkpoint=False, qty_weight=1.0, usage_reg=0.0, free_run=False,
                     code_stats=None, keep_first_codes=False):
    """One forward over a [B, T] window batch.  Returns (mean loss over scored example-steps,
    count dict, per-step losses list, codes written at the first scored step)."""
    B, T = batch["step_valid"].shape
    bottleneck = net.config.think == "bottleneck"
    field = net.none_field(B, device) if bottleneck else None
    meter = Meter()
    total, n_scored, usage_terms, step_losses, first_codes = 0.0, 0, [], [], None
    for t in range(T):
        valid = batch["step_valid"][:, t]
        if not bool(valid.any()):
            continue
        idx = valid.nonzero().squeeze(1)
        scored = bool(batch["loss_mask"][idx, t].any())
        x = _to(batch, t, idx.to(batch["step_valid"].device), device)
        didx = idx.to(device)
        f_in = M.FieldState(field.codes.index_select(0, didx), field.fresh.index_select(0, didx)) if bottleneck else None
        noise = None
        if bottleneck and code_mode != "argmax":
            noise = torch.rand((len(idx), net.S, net.config.C, net.config.K), generator=generator, device=device)
        if not scored:
            with torch.no_grad(), autocast(device, bf16):
                ctx = net.encode(x["rows"], x["token_mask"], x["knownness"], f_in, tau=tau,
                                 code_mode=code_mode, code_noise=noise)
            if bottleneck:
                field = M.FieldState(field.codes.index_copy(0, didx, ctx.field.codes.float()),
                                     field.fresh.index_fill(0, didx, False))
            continue
        up = torch.stack((x["unit_tok"], x["unit_qty"]), -1)
        op = torch.stack((x["order_tok"], x["order_qty"]), -1)
        args = (x["rows"], x["token_mask"], x["knownness"], x["unit_pos"], up, op)
        pred = None
        with autocast(device, bf16):
            if grad_checkpoint and torch.is_grad_enabled():
                def fn(*a, _noise=noise):
                    field_in = M.FieldState(a[6], a[7]) if bottleneck else None
                    o = net(*a[:6], field=field_in, tau=tau, code_mode=code_mode, code_noise=_noise)
                    f_out = o.pop("field")
                    return {**o, "codes": f_out.codes} if bottleneck else o
                extra = (f_in.codes, f_in.fresh) if bottleneck else (None, None)
                out = dict(torch.utils.checkpoint.checkpoint(fn, *args, *extra, use_reentrant=False))
                if bottleneck:
                    out["field"] = M.FieldState(out.pop("codes"), torch.zeros(len(idx), dtype=torch.bool, device=device))
            else:
                ctx = net.encode(*args[:3], f_in, tau=tau, code_mode=code_mode, code_noise=noise)
                out = net.decode_teacher(ctx, x["unit_pos"], up, op)
                out.update(value=ctx.value, field=ctx.field, code_logits=ctx.code_logits)
                if free_run:
                    with torch.no_grad():
                        pred = net.decode_free(ctx, x["unit_pos"], temperature=0.0)
        lab = (x["unit_tok"], x["unit_qty"], x["order_tok"], x["order_qty"])
        per_example, counts = step_loss(out, lab, net, qty_weight)
        meter.add(counts)
        loss_t = per_example.sum()
        step_losses.append(loss_t)
        total = total + loss_t
        n_scored += len(idx)
        if bottleneck:
            codes = out["field"].codes
            if keep_first_codes and first_codes is None:
                codes.retain_grad()
                first_codes = codes
            if code_stats is not None:
                code_stats.add(codes, out["code_logits"])
            if usage_reg > 0 and len(idx) > 1:
                # per-position usage across the batch: log K - H(mean_b p[b, s, c]); 0 when every
                # field position spreads its codes over examples, log K when it writes one code
                p = torch.softmax(out["code_logits"].float(), -1).mean(0)            # [S, C, K]
                usage_terms.append(math.log(net.config.K) + (p * torch.log(p.clamp_min(1e-12))).sum(-1).mean())
            field = M.FieldState(field.codes.index_copy(0, didx, codes.float()), field.fresh.index_fill(0, didx, False))
        if pred is not None:
            meter.add(free_counts(pred, lab))
    loss = total / max(1, n_scored)
    if usage_terms:
        reg = torch.stack(usage_terms).mean()
        meter.add({"usage_reg_term": reg.item()})
        loss = loss + usage_reg * reg
    meter.add({"scored_example_steps": n_scored})
    return loss, meter, step_losses, first_codes


# ----------------------------------------------------------------------------- checkpoints
def _write_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(out_dir, net, opt, cfg, *, update, windows_consumed, epoch, resumes, generator, metrics):
    out_dir = Path(out_dir)
    blob = {"schema": CKPT_SCHEMA, "config": net.config.to_dict(), "train_config": dataclasses.asdict(cfg),
            "model": net.state_dict(), "optimizer": opt.state_dict(), "update": update,
            "windows_consumed": windows_consumed, "epoch": epoch, "resumes": resumes,
            "generator_state": generator.get_state(), "torch_rng": torch.get_rng_state(),
            "tau": tau_at(update, cfg), "lr": lr_at(update, cfg), "metrics": metrics}
    path = out_dir / f"ckpt-{update:07d}.pt"
    _write_atomic(blob, path)
    (out_dir / "LATEST.tmp").write_text(path.name + "\n")
    os.replace(out_dir / "LATEST.tmp", out_dir / "LATEST")
    old = sorted(out_dir.glob("ckpt-*.pt"))[:-max(1, cfg.keep_ckpts)]
    for p in old:
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


# ----------------------------------------------------------------------------- fit / eval
def tiny_model_config(**kw):
    base = dict(width=32, layers=2, heads=2, ff_mult=2, dec_layers=1, G=4, K=8, C=2)
    base.update(kw)
    return M.ModelConfig(**base)


def build(model_cfg, device, codec=None):
    net = M.PolicyNet(model_cfg, codec=codec).to(device)
    return net


def make_optimizer(net, cfg):
    decay, no_decay = [], []
    for n, p in net.named_parameters():
        (decay if p.ndim >= 2 and "emb" not in n and not n.startswith(("query", "slot_in")) else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=cfg.lr, betas=(0.9, cfg.beta2))


def evaluate(net, windows_or_batches, cfg, device, *, seed=12345, free_run=True, code_mode=None):
    """Teacher-forced + free-running metrics over held-out windows (field carried as at inference)."""
    net.eval()
    meter, stats = Meter(), (CodeStats(net.S, net.config.C, net.config.K) if net.S else None)
    gen = torch.Generator(device=device).manual_seed(seed)
    mode = code_mode or net.config.infer_code_mode
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in windows_or_batches:
            loss, m, _, _ = run_window_batch(net, batch, device=device, tau=1.0, code_mode=mode, generator=gen,
                                             bf16=cfg.bf16, qty_weight=cfg.qty_weight, free_run=free_run,
                                             code_stats=stats)
            k = m.sums.get("scored_example_steps", 0)
            loss_sum += loss.item() * k
            n += k
            for key, v in m.pairs.items():
                meter.add({key: v})
            meter.add({k2: v for k2, v in m.sums.items()})
    report = meter.report()
    report["loss_per_step"] = loss_sum / max(1, n)
    report["code_mode"] = mode
    if stats is not None:
        report["codes"] = stats.report()
    net.train()
    return report


def fit(cfg: TrainConfig, model_cfg: M.ModelConfig, out_dir, *, device="cpu", resume=False, codec=None,
        log=print, eval_batches=None, stop_after=None):
    """Train to cfg.updates (or stop after ``stop_after`` updates of this call, with a checkpoint).
    On resume the model config and all optimizer/RNG/stream-position state come from the
    checkpoint; ``cfg`` is used as given (the CLI rebuilds it from the stored train_config)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    update, consumed, epoch, resumes, metrics = 0, 0, 0, 0, {}
    if resume and (out_dir / "LATEST").exists():
        blob, path = load_checkpoint(out_dir)
        model_cfg = M.ModelConfig.from_dict(blob["config"])
        net = build(model_cfg, device, codec)
        net.load_state_dict(blob["model"])
        opt = make_optimizer(net, cfg)
        opt.load_state_dict(blob["optimizer"])
        gen = torch.Generator(device=device)
        gen.set_state(blob["generator_state"])
        torch.set_rng_state(blob["torch_rng"])
        update, consumed, epoch = blob["update"], blob["windows_consumed"], blob["epoch"]
        resumes = blob["resumes"] + 1
        log(json.dumps({"event": "resume", "checkpoint": str(path), "update": update, "windows_consumed": consumed}))
    else:
        torch.manual_seed(cfg.seed)
        net = build(model_cfg, device, codec)
        opt = make_optimizer(net, cfg)
        gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    net.train()
    run = {"schema": "primitive-bc-run-v1", "model_config": net.config.to_dict(), "train_config": dataclasses.asdict(cfg),
           "parameters": net.parameter_count(), "device": str(device), "torch": torch.__version__,
           "numerics": "bf16 autocast (CUDA) / fp32 master weights" if (device.type == "cuda" and cfg.bf16) else "fp32",
           "tau_schedule": {"start": cfg.tau_start, "end": cfg.tau_end, "kind": cfg.tau_schedule,
                            "anneal_updates": cfg.tau_anneal_updates or cfg.updates, "straight_through_hard_forward": True},
           "resumes": resumes, "started_update": update}
    (out_dir / f"RUN-{resumes}.json").write_text(json.dumps(run, indent=1, default=str) + "\n")
    if cfg.synthetic:
        stream = SyntheticStream(net.config, cfg)
    else:
        skip = consumed if cfg.data_resume == "skip" else 0
        stream = RealStream(cfg, split=cfg.split, epoch=epoch, resumes=0 if cfg.data_resume == "skip" else resumes,
                            skip=skip)
    if eval_batches is None and cfg.eval_every:
        eval_batches = held_out_batches(net, cfg)
    mlog = (out_dir / "metrics.jsonl").open("a")
    meter, stats, t0, window_steps = Meter(), None, time.time(), 0
    first_update = update
    while update < cfg.updates:
        if stop_after is not None and update - first_update >= stop_after:
            break
        if cfg.synthetic:
            windows = stream.batch(consumed, cfg.batch_windows)
        else:
            windows = stream.batch(cfg.batch_windows)
        consumed += len(windows)
        batch = collate(windows, net)
        tau = tau_at(update, cfg)
        for g in opt.param_groups:
            g["lr"] = lr_at(update, cfg)
        if stats is None and net.S:
            stats = CodeStats(net.S, net.config.C, net.config.K)
        loss, m, _, _ = run_window_batch(net, batch, device=device, tau=tau, generator=gen, bf16=cfg.bf16,
                                         grad_checkpoint=cfg.grad_checkpoint, qty_weight=cfg.qty_weight,
                                         usage_reg=cfg.usage_reg, code_stats=stats)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        update += 1
        meter.add({k: v for k, v in m.pairs.items()})
        meter.add({"loss": loss.item(), "updates_in_log": 1, "grad_norm": gnorm.item()})
        window_steps += m.sums.get("scored_example_steps", 0)
        if update % cfg.log_every == 0 or update == cfg.updates:
            r = meter.report()
            k = r.pop("updates_in_log")
            r["loss"], r["grad_norm"] = r["loss"] / k, r["grad_norm"] / k
            dt = time.time() - t0
            r.update(update=update, tau=tau, lr=lr_at(update - 1, cfg), windows_consumed=consumed, epoch=getattr(stream, "epoch", 0),
                     scored_steps_per_s=window_steps / max(dt, 1e-9), seconds=dt)
            if stats is not None:
                r["codes"] = stats.report()
            if device.type == "cuda":
                r["cuda_max_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            mlog.write(json.dumps(r) + "\n")
            mlog.flush()
            log(json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in r.items() if k2 != "codes"}))
            metrics = r
            meter, stats, t0, window_steps = Meter(), None, time.time(), 0
        if cfg.eval_every and update % cfg.eval_every == 0 and eval_batches:
            ev = evaluate(net, eval_batches, cfg, device)
            ev["update"] = update
            mlog.write(json.dumps({"eval": ev}) + "\n")
            mlog.flush()
            log(json.dumps({"eval": {k2: v for k2, v in ev.items() if k2 != "codes"}}))
            metrics = {**metrics, "eval": ev}
        last = update == cfg.updates or (stop_after is not None and update - first_update >= stop_after)
        if update % cfg.ckpt_every == 0 or last:
            save_checkpoint(out_dir, net, opt, cfg, update=update, windows_consumed=consumed,
                            epoch=getattr(stream, "epoch", 0), resumes=resumes, generator=gen, metrics=metrics)
    mlog.close()
    return net, metrics


def held_out_batches(net, cfg, *, windows=None, seed_offset=99_991):
    """Fixed held-out window batches: synthetic (different seed) or data.py eval split."""
    count = windows or cfg.eval_windows
    if cfg.synthetic:
        s = SyntheticStream(net.config, cfg, seed_offset=seed_offset)
        ws = s.batch(0, count)
    else:
        ecfg = dataclasses.replace(cfg, shuffle_buffer=1, seed=cfg.seed + 17)
        s = RealStream(ecfg, split=cfg.eval_split, workers=cfg.workers)
        ws = []
        while len(ws) < count:   # every 6th window of each perspective: 5 game phases incl. step 0
            w = s.next()
            if (w["start_step"] // cfg.window) % 6 == 0:
                ws.append(w)
    return [collate(ws[i:i + cfg.batch_windows], net) for i in range(0, len(ws), cfg.batch_windows)]


# ----------------------------------------------------------------------------- probe / bench
def saved_activation_bytes(fn):
    """Bytes of distinct non-parameter storages saved for backward while fn() runs."""
    seen = {}

    def pack(t):
        try:
            s = t.untyped_storage()
            seen[s.data_ptr()] = s.nbytes()
        except Exception:
            pass
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        result = fn()
    return seen, result


def carry_probe(net, batch, device="cpu", lags=(1, 2, 4, 8, 16, 23), seed=0, tau=1.0):
    """||d(later losses)/d(codes written at the first scored step)|| through the carried field."""
    if net.config.think != "bottleneck":
        raise ValueError("carry probe needs think='bottleneck'")
    gen = torch.Generator(device=device).manual_seed(seed)
    params = {p.untyped_storage().data_ptr() for p in net.parameters()}
    seen, (loss, m, step_losses, codes0) = saved_activation_bytes(
        lambda: run_window_batch(net, batch, device=device, tau=tau, generator=gen, bf16=False,
                                 keep_first_codes=True))
    act_bytes = sum(v for k, v in seen.items() if k not in params)
    report = {"scored_steps": len(step_losses), "windows": int(batch["step_valid"].shape[0]),
              "saved_activation_bytes": act_bytes,
              "saved_activation_mb_per_window": act_bytes / 2**20 / batch["step_valid"].shape[0],
              "tokens_max": int(batch["rows"].shape[2])}
    grads = {}
    for lag in lags:
        if lag < len(step_losses):
            g, = torch.autograd.grad(step_losses[lag], codes0, retain_graph=True)
            grads[str(lag)] = float(g.norm())
    later = torch.stack(step_losses[1:]).sum()
    g_all, = torch.autograd.grad(later, codes0, retain_graph=True)
    g_own, = torch.autograd.grad(step_losses[0], codes0, retain_graph=True)
    report.update(grad_norm_first_codes_from_later_losses=float(g_all.norm()),
                  grad_norm_first_codes_from_own_step=float(g_own.norm()),
                  grad_norm_first_codes_by_lag=grads)
    return report


def bench(sizes=("1m", "4m", "12m"), *, n_tokens=250, repeats=30, codec=None, think=("bottleneck",)):
    out = {"n_tokens_packed": n_tokens, "threads": 1, "sizes": {}}
    for name in sizes:
        for th in think:
            cfg = M.ModelConfig(**M.SIZES[name], think=th)
            net = M.PolicyNet(cfg, codec=codec).eval()
            key = name if th == "bottleneck" else f"{name}-{th}"
            out["sizes"][key] = {"parameters": net.parameter_count(), "shape": M.SIZES[name], "think": th,
                                 "latency": {str(u): M.latency(net, n_tokens=n_tokens, n_units=u, repeats=repeats)
                                             for u in (1, 2, 13)}}
    return out


def act_latency(net, archive, *, seat=0, stride=6, threads=1, member_index=0):
    """Single-thread wall time of PolicyNet.act (features/pack + model + codec decode) on the
    observations of one real replay, carrying the field step to step."""
    import zipfile
    from farm_neat.training import bc
    with zipfile.ZipFile(archive) as z:
        name = sorted(n for n in z.namelist() if n.endswith(".json"))[member_index]
        replay = json.loads(z.read(name))
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        net = net.eval()
        state, times, units = net.initial_state(), [], []
        gen = torch.Generator().manual_seed(0)
        for t in range(0, 719, stride):
            obs = bc.public_observation(replay, t, seat)
            if t:
                state.last_step = t - 1          # strided sample: keep the field as if consecutive
            t0 = time.perf_counter()
            with torch.inference_mode():
                _, state = net.act(obs, generator=gen, temperature=1.0, state=state)
            times.append(time.perf_counter() - t0)
            units.append(1 + len(obs["farms"][seat]["hands"]))
        times_sorted = sorted(times)
        return {"archive": str(archive), "member": name, "seat": seat, "steps": len(times), "threads": threads,
                "median_ms": 1000 * times_sorted[len(times) // 2], "p90_ms": 1000 * times_sorted[int(0.9 * (len(times) - 1))],
                "max_ms": 1000 * times_sorted[-1], "units_max": max(units)}
    finally:
        torch.set_num_threads(old)


# ----------------------------------------------------------------------------- CLI
def _model_cfg_from_args(a):
    if a.tiny:
        return tiny_model_config(think=a.think)
    shape = dict(M.SIZES[a.size])
    for k in ("width", "layers", "heads", "dec_layers", "G", "K", "C"):
        v = getattr(a, k.lower() if k not in ("G", "K", "C") else "field_" + k.lower())
        if v:
            shape[k] = v
    return M.ModelConfig(**shape, think=a.think)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--out", required=True)
    f.add_argument("--resume", action="store_true")
    f.add_argument("--stop-after", type=int, default=None, help="stop (with a checkpoint) after this many updates")
    f.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    f.add_argument("--tiny", action="store_true", help="synthetic data + tiny model (CPU tests)")
    f.add_argument("--size", default="4m", choices=sorted(M.SIZES))
    f.add_argument("--think", default="bottleneck", choices=("bottleneck", "off"))
    for k in ("width", "layers", "heads", "dec_layers", "field_g", "field_k", "field_c"):
        f.add_argument("--" + k.replace("_", "-"), type=int, default=0)
    f.add_argument("--archives", nargs="*", default=[])
    f.add_argument("--archive-glob", default=None)
    for fld in dataclasses.fields(TrainConfig):
        if fld.name in ("archives", "synthetic"):
            continue
        name = "--" + fld.name.replace("_", "-")
        if fld.type in ("bool", bool):
            f.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        elif fld.name == "rank_max" or fld.name == "max_episodes":
            f.add_argument(name, type=int, default=None)
        else:
            typ = {"int": int, "float": float, "str": str}.get(str(fld.type), str)
            f.add_argument(name, type=typ, default=None)
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out")
    e.add_argument("--device", default="cpu")
    e.add_argument("--split", default=None)
    e.add_argument("--windows", type=int, default=256)
    e.add_argument("--workers", type=int, default=None)
    e.add_argument("--code-mode", default=None, choices=("sample", "argmax"))
    b = sub.add_parser("bench")
    b.add_argument("--out")
    b.add_argument("--repeats", type=int, default=30)
    b.add_argument("--tokens", type=int, default=250)
    b.add_argument("--sizes", nargs="+", default=["1m", "4m", "12m"])
    b.add_argument("--off", action="store_true", help="also time think='off'")
    b.add_argument("--replay-archive", default=None, help="also time act() on one real replay (default size)")
    p = sub.add_parser("probe")
    p.add_argument("--out")
    p.add_argument("--size", default="4m")
    p.add_argument("--windows", type=int, default=1)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--burn-in", type=int, default=24)
    p.add_argument("--archives", nargs="*", default=[])
    p.add_argument("--max-episodes", type=int, default=1)
    a = ap.parse_args(argv)

    if a.cmd == "fit":
        resuming = a.resume and (Path(a.out) / "LATEST").exists()
        cfg = TrainConfig.from_dict(load_checkpoint(a.out)[0]["train_config"]) if resuming else TrainConfig()
        overrides = {fld.name: getattr(a, fld.name) for fld in dataclasses.fields(TrainConfig)
                     if hasattr(a, fld.name) and getattr(a, fld.name) is not None and fld.name not in ("archives", "synthetic")}
        cfg = dataclasses.replace(cfg, **overrides)
        archives = list(a.archives)
        if a.archive_glob:
            import glob
            archives += sorted(glob.glob(a.archive_glob))
        if archives or not resuming:
            cfg.archives = archives
        if a.tiny and not resuming:
            cfg = dataclasses.replace(cfg, synthetic=True, **{k: v for k, v in dict(
                window=4, burn_in=2, batch_windows=8, updates=200, warmup=10, lr=3e-3, eval_every=0,
                ckpt_every=50, log_every=20, synthetic_len=12, tau_start=1.0, tau_end=0.5).items()
                if overrides.get(k) is None})
        fit(cfg, _model_cfg_from_args(a), a.out, device=a.device, resume=a.resume, stop_after=a.stop_after)
    elif a.cmd == "eval":
        blob, path = load_checkpoint(a.checkpoint)
        net = build(M.ModelConfig.from_dict(blob["config"]), a.device)
        net.load_state_dict(blob["model"])
        cfg = TrainConfig.from_dict(blob["train_config"])
        if a.split:
            cfg.eval_split = a.split
        if a.workers is not None:
            cfg.workers = a.workers
        batches = held_out_batches(net, cfg, windows=a.windows)
        rep = evaluate(net, batches, cfg, torch.device(a.device), code_mode=a.code_mode)
        rep.update(checkpoint=str(path), update=blob["update"], split=cfg.eval_split, windows=a.windows)
        text = json.dumps(rep, indent=1)
        if a.out:
            Path(a.out).write_text(text + "\n")
        print(text)
    elif a.cmd == "bench":
        rep = bench(tuple(a.sizes), n_tokens=a.tokens, repeats=a.repeats,
                    think=("bottleneck", "off") if a.off else ("bottleneck",))
        if a.replay_archive:
            torch.manual_seed(0)
            rep["act_real_replay"] = {name: act_latency(M.PolicyNet(M.ModelConfig(**M.SIZES[name])), a.replay_archive)
                                      for name in a.sizes}
        rep["load_average"] = os.getloadavg()
        text = json.dumps(rep, indent=1)
        if a.out:
            Path(a.out).write_text(text + "\n")
        print(text)
    elif a.cmd == "probe":
        torch.manual_seed(0)
        net = M.PolicyNet(M.ModelConfig(**M.SIZES[a.size])).train()
        cfg = TrainConfig(window=a.window, burn_in=a.burn_in, archives=a.archives, workers=0,
                          max_episodes=a.max_episodes, shuffle_buffer=1)
        if a.archives:
            s = RealStream(cfg, split="train", workers=0)
            ws = [w for w in (s.next() for _ in range(40)) if w["n_burn"] == a.burn_in and w["n_scored"] == a.window][:a.windows]
            source = "data.iter_windows"
        else:
            cfg.synthetic, cfg.synthetic_len, cfg.synthetic_hands = True, a.window + a.burn_in, 12
            ws = [w for w in SyntheticStream(net.config, cfg).batch(0, 4) if w["n_burn"] == a.burn_in][:a.windows]
            source = "synthetic"
        batch = collate(ws, net)
        rep = carry_probe(net, batch, lags=(1, 2, 4, 8, 16, a.window - 1))
        rep.update(source=source, size=a.size, parameters=net.parameter_count(), window=a.window, burn_in=a.burn_in,
                   note="bytes = distinct non-parameter storages saved for backward over the scored steps "
                        "(burn-in runs without grad); fp32 CPU; bf16 autocast roughly halves matmul activations")
        text = json.dumps(rep, indent=1)
        if a.out:
            Path(a.out).write_text(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
