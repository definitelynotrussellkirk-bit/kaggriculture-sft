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
STEP_KEYS = ("rows", "token_mask", "knownness", "unit_pos", "unit_tok", "unit_qty", "order_tok", "order_qty",
             "extra", "unit_extra", "reach_eta", "unit_prefix", "order_prefix", "order_cash",
             "v6_token", "v6_resource_eta", "v6_resource_route", "v6_job_deadline", "v6_job_deadline_present",
             "prev_unit", "prev_order", "next_unit_tok", "opp_unit_tok",   # v6 keys exist only with TrainConfig.v6_data
             "next_unit_qty", "next_order_tok", "next_order_qty", "opp_unit_qty", "opp_order_tok", "opp_order_qty",
             "x8_token", "x8_unit", "lattice", "unit_loss_keep",                                # v8 keys only with TrainConfig.v8_data
             # optional quantity-equivalence masks (equiv_build.py side-cars, merged by EquivChunks): label-side only
             "unit_qty_set", "order_qty_set", "next_unit_qty_set", "next_order_qty_set", "opp_unit_qty_set", "opp_order_qty_set")
EQUIV_SCHEMA = "engine-effect-quantity-v1"   # PRO-EQUIV-0928 RETURN-1 container; quantity sets only, token sets singleton
STEP_KEYS += ("unit_infeasible", "order_infeasible", "order_loss_keep", "order_token_weight")


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
    muon_momentum: float = 0.95     # explicit, serialized; match_rms_adamw uses the AdamW LR scale
    muon_ns_steps: int = 5
    recur_train_min: int = 1         # per-update uniform [min, model.recur_train_max]; resumable without shape edits
    weight_decay: float = 0.05
    beta2: float = 0.95
    grad_clip: float = 1.0
    mask_unit_noops: bool = False  # require independently cached unit-noop-mask-v1 labels
    qty_weight: float = 1.0
    supervision_dir: str = ""      # actor-only loss sidecar; disabled unless an objective flag is set
    supervision_manifest_sha256: str = ""
    mask_order_noops: bool = False  # certified certain-failure order labels only
    infeasible_w: float = 0.0       # per-example mean forbidden unit-token mass
    order_infeasible_w: float = 0.0 # per-example mean forbidden order-token mass
    commitment_weighting: bool = False   # opt-in receipt-use order-token CE reweighting; never an actor input
    commitment_dir: str = ""              # per-seat cache-bound sidecars (training split only)
    commitment_manifest_sha256: str = ""
    tau_start: float = 2.0          # Gumbel-softmax temperature (backward relaxation)
    tau_end: float = 0.5
    tau_anneal_updates: int = 0     # 0 -> updates
    tau_schedule: str = "exp"       # exp | linear
    usage_reg: float = 0.0          # per-position code-usage regulariser; 0 unless collapse is observed
    grad_checkpoint: bool = False
    pack_steps: int = 0          # > 0: stateless models run the scored (window, step) pairs as [n, 1] chunks of this many rows
                                 # (same loss and gradient as the per-step loop, far fewer sequential forwards; owner 2026-09-28)
    compile_blocks: bool = False    # CUDA: torch.compile each encoder block in place (dynamic shapes; state_dict keys unchanged)
    compile_model: bool = False     # CUDA: torch.compile the whole encode + decode_teacher (one graph once CHECK_INPUTS is off)
    step_cache: str = ""            # step_cache.py directory: packed chunks come from cached per-step features (needs pack_steps)
    cache_pool_seats: int = 4       # seats mixed per loader worker when drawing chunks from the step cache
    lr_schedule: str = "cosine"     # cosine over `updates` | wsd: constant after warmup, linear decay over the last decay_updates
    decay_updates: int = 0          # wsd only; 0 = no decay yet (a chunked run sets it in its final chunk)
    ema_decay: float = 0.0          # > 0: keep an fp32 EMA of the weights in the checkpoint ("ema"; export with ema_export.py)
    max_seconds: float = 0.0        # > 0: this fit() call checkpoints and stops after that much wall time (time-boxed chunks)
    optimizer: str = "adamw"        # adamw | muon: torch.optim.Muon (match_rms_adamw, same lr) on the 2-D weights of encoder/decoder
                                    # layers, AdamW (the groups below) on everything else
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
    wins_only: bool = True          # data.select_seats; True = the 2026-09-23 wins-only patch (fit-675)
    recent_days: int = 0            # >0: only the last N archive days of `archives`
    min_decile: int = 0             # >0: only seats with team-day strength decile >= N (lineage A STRENGTH.json)
    beat_archives_file: str = ""    # file listing beat archives (data._tasks / data.beat_seats); "" = off
    beat_ids_file: str = ""         # file: one episode id per line (winner seat trained, no filters); "" = off
    beat_lookup_file: str = ""      # file listing extra archives searched only for beat_ids episodes
    force_teams_file: str = ""      # file: one team name per line; their winning seats skip rank/decile filters
    priority_file: str = ""         # TSV archive,member,day,eid,seat,weight[,reason]: (episode, seat) repeated weight times
    priority_only: bool = False     # train on the priority rows alone (no archive tasks)
    min_final_quadrants: int = 0    # keep only seats owning at least this many quadrants at the end
    worth_min: float = 0.0          # >0: data.WORTH_FILE rows with W >= worth_min, sampled in proportion to W (team cap 10%)
    grow_lr_mult: float = 10.0      # lr multiplier of the zero-initialised grow_* parameters
    grow_prefixes: str = ""         # comma-separated parameter-name prefixes (grown layers) that join the grow_* group
    v6_data: bool = False           # v6: windows carry the previous two own turns and the aux labels (data.add_v6)
    feature_schema: str = "legacy"  # must equal ModelConfig.feature_schema; recorded in checkpoints
    aux_reduction: str = "sum"      # sum (legacy) | example_lag_mean (fixed lag denominator, chunk-invariant)
    aux_lag_decay: float = 1.0       # gamma**lag_index; own offsets 1/2, opponent offsets 0/1/2
    aux_order_qty_scale: float = 1.0 # independent multiplier within each order head's configured weight
    aux_scale: float = 1.0           # one common multiplier on every aux weight (calibration / per-chunk schedule)
    # PRO-EQUIV-0928 (RETURN-1 + RETURN-2): opt-in, actor-only QUANTITY aliases. Off = canonical CE, bit-identical. Can be switched
    # on at a chunk boundary (resume with the flag); every loaded seat then needs both seats' side-cars in equiv_dir (fail closed).
    equiv_dir: str = ""             # directory of <eid>-<seat>.equiv.npz from equiv_build.py
    equiv_loss: bool = False        # main unit/order quantity heads: set NLL over the certified alias set (training only)
    aux_equiv_loss: bool = False    # aux quantity heads (own t+1/t+2, opponent t/t+1/t+2): same, from the target turn's side-car
    equiv_prefix_aug: bool = False  # training: feed a random certified alias as the teacher-forced quantity prefix (needs equiv_loss)
    aux_next_w: float = 0.01        # v6: weight per own t+1 / t+2 head (aux <= ~0.2 of BC loss; pbc-v6-design/DESIGN.md)
    aux_next_qty_w: float = 0.0     # v6_aux_turn: own t+1/t+2 worker quantities (CE per labelled worker)
    aux_opp_qty_w: float = 0.0      # v6_aux_turn: opponent t/t+1/t+2 worker quantities
    aux_next_order_w: float = 0.0   # v6_aux_turn: own t+1/t+2 order sequences (token + quantity CE per slot)
    aux_opp_order_w: float = 0.0    # v6_aux_turn: opponent t/t+1/t+2 order sequences
    aux_opp_w: float = 0.006        # v6: weight per opponent t / t+1 / t+2 head
    new_prefixes: str = ""          # parameter-name prefixes of freshly grafted tensors: own group, no decay, lr x new_lr_mult
    new_lr_mult: float = 1.0        #   (grow LR x10 from the first update HURT: start-of-phase loss spike, WHAT-HELPS.md L39)
    new_warmup: int = 0             #   so the multiplier ramps linearly over new_warmup updates counted from new_from
    new_from: int = 0
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
    if getattr(cfg, "lr_schedule", "cosine") == "wsd":
        d = int(getattr(cfg, "decay_updates", 0) or 0)
        if d <= 0 or update < cfg.updates - d:
            return cfg.lr
        # `updates` is an exclusive horizon: the last executed index is updates - 1.
        progress = 1.0 if d == 1 else min(1.0, (update - (cfg.updates - d)) / (d - 1))
        return cfg.lr * (1 - (1 - cfg.min_lr_frac) * progress)
    f = min(1.0, (update - cfg.warmup) / max(1, cfg.updates - cfg.warmup))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * f)))


def recur_loops_at(update, model_cfg, cfg):
    """One deterministic draw per optimizer update, identical across resume and accumulation."""
    hi = int(getattr(model_cfg, "recur_train_max", 0) or 0)
    lo = int(getattr(cfg, "recur_train_min", 1))
    if hi <= 0:
        if lo != 1:
            raise ValueError("recur_train_min requires a positive model recur_train_max")
        return int(model_cfg.recur_loops)
    if not 1 <= lo <= hi:
        raise ValueError("recur_train_min must be in [1, recur_train_max]")
    return random.Random(f"recur:{cfg.seed}:{update}").randint(lo, hi)


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
    flat = [ex for w in windows for ex in w["steps"]]
    if any("unit_loss_keep" in ex for ex in flat):
        from noop_mask import SCHEMA
        if any(ex.get("unit_noop_mask_schema") != SCHEMA or "unit_loss_keep" not in ex for ex in flat):
            raise ValueError("mixed or stale unit no-op mask examples")
        out["unit_loss_keep"] = torch.zeros(B, T, max_units, dtype=torch.bool)
        out["unit_noop_mask_schema"] = SCHEMA
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
            if "unit_loss_keep" in out:
                out["unit_loss_keep"][i, col] = torch.as_tensor(ex["unit_loss_keep"], dtype=torch.bool)
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
        self._epoch_yielded = 0
        cfg = self.cfg
        seats = int(cfg.seats) if str(cfg.seats) in ("0", "1") else cfg.seats
        return iter(self.data.iter_windows(cfg.archives or None, split=self.split, rank_max=cfg.rank_max,
                                           seats=seats, workers=self.workers, window=cfg.window,
                                           burn_in=cfg.burn_in, stride=cfg.stride or None,
                                           shuffle_seed=self._seed(), max_episodes=cfg.max_episodes,
                                           stats=self.stats, wins_only=cfg.wins_only,
                                           recent_days=cfg.recent_days, min_decile=cfg.min_decile,
                                           worth_min=getattr(cfg, "worth_min", 0.0),
                                           priority_only=getattr(cfg, "priority_only", False),
                                           min_final_quadrants=getattr(cfg, "min_final_quadrants", 0),
                                           v6=getattr(cfg, "v6_data", False),
                                           feature_schema=getattr(cfg, "feature_schema", "legacy"), **self._beat()))

    def _beat(self):
        out = {}
        for key, field, conv in (("beat_archives", "beat_archives_file", str), ("beat_ids", "beat_ids_file", int),
                                 ("beat_lookup", "beat_lookup_file", str), ("force_teams", "force_teams_file", str)):
            f = getattr(self.cfg, field, "")
            if f:
                with open(f) as fh:
                    out[key] = [conv(ln.strip()) for ln in fh if ln.strip()]
        pf = getattr(self.cfg, "priority_file", "")
        if pf:
            with open(pf) as fh:
                rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
            out["priority_rows"] = [(r[0], r[1], r[2], int(r[3]), int(r[4]), int(r[5])) for r in rows[1:]]
        return out

    def _pull(self):
        while True:
            try:
                window = next(self._it)
                self._epoch_yielded += 1
                return window
            except StopIteration:
                if self._epoch_yielded == 0:
                    errors = self.stats.get("episode_errors", [])
                    raise RuntimeError(f"data stream ({self.split}) yielded no windows in epoch {self.epoch}; "
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


_QTY_COUNT = {}


def _qty_count(n_qty, device):
    """Report-only: count of every codec quantity bin (0..73 = 1..64, 66, ..., 100); +inf for the reference-relative bin 74."""
    key = (int(n_qty), str(device))
    if key not in _QTY_COUNT:
        try:
            import codec
            bins = list(codec.QTY_BINS)
        except ImportError:
            bins = []
        vals = [float(num) if kind == "n" else math.inf for kind, num, _ in bins] if len(bins) == n_qty else [math.inf] * n_qty
        _QTY_COUNT[key] = torch.tensor(vals, device=device)
    return _QTY_COUNT[key]


def set_nll(logits, target, allowed, valid):
    """-log sum_{a in allowed} softmax(logits)[a] where valid, else 0 (PRO-EQUIV-0928 RETURN-2 kernel). Padding rows allow every
    class, so neither logsumexp sees an all -inf row (no NaN gradients). No host sync; membership is checked by the loader."""
    z = logits.float()
    ok = allowed | ~valid[..., None]
    nll = z.logsumexp(-1) - z.masked_fill(~ok, float("-inf")).logsumexp(-1)
    return torch.where(valid, nll, torch.zeros_like(nll))


def _set_counts(h, logits, allowed, valid, nll):
    """Pooled set metrics (RETURN-1/2): set NLL sum, argmax inside the set, rows with >1 member, summed set size."""
    size = allowed.sum(-1)
    hit = allowed.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1) & valid
    return {f"loss_set_{h}": nll.detach().sum(), f"set_hit_{h}": (hit.sum(), valid.sum()),
            f"set_multi_{h}": ((size > 1) & valid).sum(), f"set_size_{h}": (size * valid).sum()}


def sample_alias(target, allowed):
    """Training-only (RETURN-1 equiv_prefix_aug): a uniformly random certified alias bin per labelled row; -1 stays -1."""
    chosen = torch.rand(allowed.shape, device=allowed.device).masked_fill(~allowed, -1.0).argmax(-1)
    return torch.where(target >= 0, chosen, target)


def order_weight_scale(chunks, *, mask_order_noops=False, stop_index=None):
    """One order-head normalizer for every packed chunk in an optimizer update."""
    n_labels, sum_weights = 0, 0.0
    for chunk in chunks:
        target, weight = chunk["order_tok"], chunk["order_token_weight"]
        scored = target >= 0
        if mask_order_noops:
            from loss_supervision import checked_keep
            scored = checked_keep(target, chunk["order_loss_keep"], stop_index=stop_index)
        if (target.shape != weight.shape or not bool(torch.isfinite(weight).all())
                or not bool((weight >= 0).all()) or not bool((weight[scored] > 0).all())):
            raise ValueError("order_token_weight must align with scored labels and be finite nonnegative")
        n_labels += int(scored.sum())
        sum_weights += float(weight[scored].sum())
    if not n_labels or not sum_weights:
        raise ValueError("weighted order update has no scored labels")
    return n_labels / sum_weights


def _grad_clip(cfg):
    """Owner 2026-09-28: PBC_GRAD_CLIP overrides cfg.grad_clip without changing the run identity (unset = cfg)."""
    v = os.environ.get("PBC_GRAD_CLIP", "")
    return float(v) if v else cfg.grad_clip


COS_EVERY = int(os.environ.get("PBC_COS_EVERY", "0") or 0)   # owner: weight cosine to adjacent / original every N updates (0 = off)


class _WeightSketch:
    """Fixed random 1% sample (>= 1024 elements) of every parameter; saved every COS_EVERY updates under out_dir/weight_sketch/
    so cosines to the previous sketch, the run's origin (PBC_COS_ORIGIN checkpoint) and any later "final" can be computed."""

    def __init__(self, net, out_dir, device):
        g = torch.Generator().manual_seed(20260928)
        self.idx = {}
        for n, p in sorted(net.named_parameters()):
            k = min(p.numel(), max(1024, p.numel() // 100))
            self.idx[n] = torch.randint(0, p.numel(), (k,), generator=g).to(device)
        self.dir = Path(out_dir) / "weight_sketch"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prev, self.origin = None, None
        o = os.environ.get("PBC_COS_ORIGIN", "")
        if o:
            sd = torch.load(o, map_location="cpu", weights_only=False)["model"]
            self.origin = {n: sd[n].float().flatten()[self.idx[n].cpu()] for n in self.idx if n in sd}
            torch.save(self.origin, self.dir / "origin.pt")

    def take(self, net):
        with torch.no_grad():
            return {n: p.detach().float().flatten()[self.idx[n]].cpu() for n, p in net.named_parameters() if n in self.idx}

    @staticmethod
    def cos(a, b):
        groups = {}
        for n, x in a.items():
            if n not in b:
                continue
            y = b[n]
            parts = n.split(".")
            layer = ".".join(parts[:2]) if len(parts) > 2 and parts[1].isdigit() else parts[0]
            opt = "muon" if (x.ndim == 1 and n.startswith(("blocks.", "dec.")) and n.endswith(".weight") and not n.startswith("grow_")) else "other"
            kind = ("grow" if n.startswith("grow_") else "aux" if n.startswith("aux_") else "encoder" if n.startswith("blocks.")
                    else "decoder" if n.startswith("dec.") else "other")
            for key in ("all", "kind:" + kind, "layer:" + layer):
                gsum = groups.setdefault(key, [0.0, 0.0, 0.0])
                gsum[0] += float((x * y).sum()); gsum[1] += float((x * x).sum()); gsum[2] += float((y * y).sum())
        return {k: round(v[0] / max((v[1] * v[2]) ** 0.5, 1e-30), 6) for k, v in groups.items()}

    def step(self, net, update):
        cur = self.take(net)
        torch.save(cur, self.dir / f"u{update:07d}.pt")
        rec = {"update": update}
        if self.prev is not None:
            rec["adjacent"] = self.cos(cur, self.prev)
        if self.origin is not None:
            rec["origin"] = self.cos(cur, self.origin)
        self.prev = cur
        with open(self.dir.parent / "cos_stats.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


UPDATE_STATS = int(os.environ.get("PBC_UPDATE_STATS", "0") or 0)   # owner: log |dW|/|W| per layer every N updates (0 = off)


def _write_update_stats(out_dir, update, named, before, gpre):
    """One line in out_dir/update_stats.jsonl: per layer and per group, |dW| / |W| of this update and the pre-clip grad norm."""
    agg = {}
    def add(key, dw2, w2, g2):
        a = agg.setdefault(key, [0.0, 0.0, 0.0])
        a[0] += dw2; a[1] += w2; a[2] += g2
    with torch.no_grad():
        for n, p in named:
            w0 = before[n]
            dw2 = float((p.detach().float() - w0).pow(2).sum()); w2 = float(w0.pow(2).sum()); g2 = gpre.get(n, 0.0) ** 2
            parts = n.split(".")
            layer = ".".join(parts[:2]) if len(parts) > 2 and parts[1].isdigit() else parts[0]
            opt = "muon" if (p.ndim == 2 and n.startswith(("blocks.", "dec.")) and n.endswith(".weight") and not n.startswith("grow_")) else "adamw"
            kind = ("grow" if n.startswith("grow_") else "aux" if n.startswith("aux_") else "encoder" if n.startswith("blocks.")
                    else "decoder" if n.startswith("dec.") else "other")
            for key in ("layer:" + layer, "opt:" + opt, "kind:" + kind, "all"):
                add(key, dw2, w2, g2)
    out = {k: {"rel_update": (v[0] ** 0.5) / max(v[1] ** 0.5, 1e-12), "w_norm": v[1] ** 0.5, "grad_norm": v[2] ** 0.5}
           for k, v in agg.items()}
    layers = sorted(((k[6:], v["rel_update"]) for k, v in out.items() if k.startswith("layer:")), key=lambda kv: -kv[1])
    rec = {"update": update, "groups": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in out.items() if not k.startswith("layer:")},
           "layers": {k[6:]: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in out.items() if k.startswith("layer:")},
           "top_moving_layers": [(k, round(v, 6)) for k, v in layers[:10]]}
    with open(Path(out_dir) / "update_stats.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


TURN_NORM = os.environ.get("PBC_TURN_NORM", "") == "1"   # per-turn mean objective (logged loss_* sums stay canonical)
TURN_W_MOVE = float(os.environ.get("PBC_TURN_W_MOVE", "1") or 1)    # owner: worker movement has many equally valid choices
TURN_W_ORDER = float(os.environ.get("PBC_TURN_W_ORDER", "1") or 1)  # owner: market timing and quantities matter more
TURN_ORDER_DEN = float(os.environ.get("PBC_TURN_ORDER_DEN", "2.35") or 2.35)   # mean order slots incl. STOP per turn (measured)


def step_loss(out, lab, net, qty_weight=1.0, *, unit_loss_keep=None, order_loss_keep=None,
              sets=None, use_sets=False, order_token_weight=None, order_weight_scale=None):
    """Per-example summed CE over present units and the order sequence incl. STOP. sets: optional {"unit_qty"|"order_qty":
    [B, X, 75] bool} certified quantity-alias masks. With use_sets the two quantity terms of the returned objective are set NLL;
    the loss_* sums always stay canonical CE, and loss_set_* / set_hit_* are logged whenever masks are present."""
    ut, uq, ot, oq = lab
    ignore = -100
    present = ut >= 0
    score_units = present
    if unit_loss_keep is not None:
        if unit_loss_keep.dtype != torch.bool or unit_loss_keep.shape != ut.shape:
            raise ValueError("unit_loss_keep must be bool with the same shape as unit_tok")
        score_units = present & unit_loss_keep
    has_uq = present & net.unit_qty_mask[ut.clamp(min=0)] & (uq >= 0)
    ovalid = ot >= 0
    score_orders = ovalid
    if order_loss_keep is not None:
        from loss_supervision import checked_keep
        score_orders = checked_keep(ot, order_loss_keep, stop_index=net.config.stop_index)
    has_oq = ovalid & net.order_qty_mask[ot.clamp(min=0)] & (oq >= 0)
    ce = lambda logits, y, m: F.cross_entropy(logits.float().transpose(1, 2), torch.where(m, y, ignore),
                                              ignore_index=ignore, reduction="none").sum(1)
    parts = {"unit_tok": ce(out["unit_logits"], ut, score_units), "unit_qty": ce(out["unit_qty_logits"], uq, has_uq & score_units),
             "order_tok": ce(out["order_logits"], ot, score_orders),
             "order_qty": ce(out["order_qty_logits"], oq, has_oq & score_orders)}
    obj, set_counts = dict(parts), {}
    if order_token_weight is not None:
        if (order_token_weight.shape != ot.shape or not bool(torch.isfinite(order_token_weight).all())
                or not bool((order_token_weight >= 0).all()) or not bool((order_token_weight[score_orders] > 0).all())):
            raise ValueError("order_token_weight must align with scored labels and be finite nonnegative")
        if order_weight_scale is None or not math.isfinite(order_weight_scale) or order_weight_scale <= 0:
            raise ValueError("order_weight_scale must be finite positive when order weights are enabled")
        per_token = F.cross_entropy(out["order_logits"].float().transpose(1, 2),
                                    torch.where(score_orders, ot, ignore), ignore_index=ignore, reduction="none")
        obj["order_tok"] = (per_token * order_token_weight).sum(1) * order_weight_scale
        set_counts["loss_weighted_order_tok"] = obj["order_tok"].detach().sum()
    for h, logits, y, m in (("unit_qty", out["unit_qty_logits"], uq, has_uq & score_units),
                            ("order_qty", out["order_qty_logits"], oq, has_oq & score_orders)):
        allowed = (sets or {}).get(h)
        if allowed is None:
            if use_sets:
                raise ValueError(f"equiv_loss needs {h}_set masks for every scored step (equiv_dir side-cars)")
            continue
        nll = set_nll(logits, y, allowed, m)
        if use_sets:
            obj[h] = nll.sum(1)
        with torch.no_grad():
            set_counts.update(_set_counts(h, logits, allowed, m, nll))
    if TURN_NORM and net.training:   # owner 2026-09-28 (eval stays canonical): every turn counts equally for workers
        n_u = score_units.sum(1).clamp(min=1).to(obj["unit_tok"].dtype)
        n_o = TURN_ORDER_DEN   # constant (review: a per-turn order count puts up to 2x weight on STOP; 48.9% of turns have no orders)
        unit_tok = obj["unit_tok"]
        if TURN_W_MOVE != 1.0:   # NORTH/SOUTH/EAST/WEST labels (codec.UNIT_VOCAB ids 1-4) weighted TURN_W_MOVE
            lab_ce = F.cross_entropy(out["unit_logits"].float().transpose(1, 2), torch.where(score_units, ut, ignore),
                                     ignore_index=ignore, reduction="none")
            unit_tok = (lab_ce * torch.where((ut >= 1) & (ut <= 4), TURN_W_MOVE, 1.0)).sum(1)
        per_example = ((unit_tok + qty_weight * obj["unit_qty"]) / n_u
                       + TURN_W_ORDER * (obj["order_tok"] + qty_weight * obj["order_qty"]) / n_o)
    else:
        per_example = obj["unit_tok"] + obj["order_tok"] + qty_weight * (obj["unit_qty"] + obj["order_qty"])
    with torch.no_grad():
        pu, pq = out["unit_logits"].argmax(-1), out["unit_qty_logits"].argmax(-1)
        po, poq = out["order_logits"].argmax(-1), out["order_qty_logits"].argmax(-1)
        counts = _exact_counts(pu, pq, po, poq, ut, uq, ot, oq, present, has_uq, ovalid, has_oq)
        counts.update(set_counts)
        counts["label_qty_on_plain"] = (((uq >= 0) & present & ~net.unit_qty_mask[ut.clamp(min=0)]).sum()
                                        + ((oq >= 0) & ovalid & ~net.order_qty_mask[ot.clamp(min=0)]).sum())
        if unit_loss_keep is not None:
            counts.update(masked_unit_commands=(present & ~score_units).sum(),
                          masked_unit_quantities=(has_uq & ~score_units).sum(),
                          scored_unit_tok=(((pu == ut) & score_units).sum(), score_units.sum()),
                          scored_unit_qty=(((pq == uq) & has_uq & score_units).sum(), (has_uq & score_units).sum()))
        if order_loss_keep is not None:
            counts.update(masked_order_commands=(ovalid & ~score_orders).sum(),
                          masked_order_quantities=(has_oq & ~score_orders).sum(),
                          scored_order_tok=(((po == ot) & score_orders).sum(), score_orders.sum()),
                          scored_order_qty=(((poq == oq) & has_oq & score_orders).sum(),
                                            (has_oq & score_orders).sum()),
                          loss_ref_order_tok=ce(out["order_logits"], ot, ovalid).sum(),
                          loss_ref_order_qty=ce(out["order_qty_logits"], oq, has_oq).sum())
        # Original exact-match denominators stay comparable across ablations.
        counts.update({f"loss_{k}": v.sum() for k, v in parts.items()})
    return per_example, counts


def _exact_counts(pu, pq, po, poq, ut, uq, ot, oq, present, has_uq, ovalid, has_oq):
    tok_u, qty_u = (pu == ut) & present, (pq == uq) & has_uq
    tok_o, qty_o = (po == ot) & ovalid, (poq == oq) & has_oq
    unit_ok = tok_u & (qty_u | ~has_uq)
    order_ok = tok_o & (qty_o | ~has_oq)
    step_ok = (unit_ok | ~present).all(1) & (order_ok | ~ovalid).all(1)
    return {"unit_tok": (tok_u.sum(), present.sum()), "unit_qty": (qty_u.sum(), has_uq.sum()),
            "order_tok": (tok_o.sum(), ovalid.sum()), "order_qty": (qty_o.sum(), has_oq.sum()),
            "unit_exact": (unit_ok.sum(), present.sum()),
            "order_exact": (order_ok.sum(), ovalid.sum()),
            "step_exact": (step_ok.sum(), step_ok.numel())}


AUX_W_KEYS = ("aux_next_w", "aux_opp_w", "aux_next_qty_w", "aux_opp_qty_w", "aux_next_order_w", "aux_opp_order_w")


def aux_weights(cfg):
    scale = float(getattr(cfg, "aux_scale", 1.0))
    return {**{k: scale * float(getattr(cfg, k, 0.0) or 0.0) for k in AUX_W_KEYS},
            "_reduction": getattr(cfg, "aux_reduction", "sum"),
            "_lag_decay": float(getattr(cfg, "aux_lag_decay", 1.0)),
            "_order_qty_scale": float(getattr(cfg, "aux_order_qty_scale", 1.0)),
            "_aux_equiv": bool(getattr(cfg, "aux_equiv_loss", False))}


_LAG_ALPHA = {}


def _lag_alpha(n, lag_decay, like):
    """(alpha, alpha / alpha.sum()) for n lags on like's device/dtype, built once: new_tensor from a Python list is a
    blocking host->device copy (a full stream sync on CUDA), which ran 8 times per chunk."""
    key = (n, float(lag_decay), like.dtype, like.device)
    got = _LAG_ALPHA.get(key)
    if got is None:
        alpha = like.new_tensor([lag_decay ** i for i in range(n)])
        got = _LAG_ALPHA[key] = (alpha, alpha / alpha.sum())
    return got


def _aux_ce(logits, y, w, name, counts, *, reduction="sum", lag_decay=1.0, allowed=None):
    """Masked CE [B,L,X,V] -> [B]. Legacy sum is unchanged by default.

    allowed [B,L,X,V] bool: certified quantity-alias masks (aux_equiv_loss); the objective uses set NLL, loss_<head><lag> stays
    canonical CE and loss_set_<head><lag> logs the set NLL.

    example_lag_mean divides each example/lag by its own valid-item count,
    then mixes lags with fixed normalized gamma**index weights. Empty lags
    contribute zero, never increase another lag's coefficient. No batch-wide
    denominator: changing pack_steps or gradient accumulation preserves loss.
    This does NOT inverse-weight rare label presence across examples.
    """
    if reduction not in ("sum", "example_lag_mean"):
        raise ValueError("aux reduction must be sum|example_lag_mean")
    if not math.isfinite(lag_decay) or not 0 < lag_decay <= 1:
        raise ValueError("aux lag_decay must be in (0,1]")
    if not math.isfinite(w) or w < 0:
        raise ValueError("aux weights must be finite and nonnegative")
    if logits.ndim != 4 or tuple(y.shape) != tuple(logits.shape[:-1]):
        raise ValueError("aux logits [B,L,X,V] and labels [B,L,X] must align")
    logits = logits.float()
    m = y >= 0
    ce = F.cross_entropy(logits.permute(0, 3, 1, 2), torch.where(m, y, -100), ignore_index=-100, reduction="none")
    obj = ce if allowed is None else set_nll(logits, y, allowed, m)
    raw = obj.sum(-1)
    alpha, alpha_norm = _lag_alpha(y.shape[1], lag_decay, logits)
    if reduction == "example_lag_mean":
        terms = raw / m.sum(-1).clamp_min(1) * alpha_norm
    else:
        terms = raw * alpha
    # Preserve the original reduction order, too, for the default path.
    per = w * obj.sum((1, 2)) if reduction == "sum" and lag_decay == 1 else w * terms.sum(-1)
    with torch.no_grad():
        # per-lag statistics as [L] vectors (a fixed handful of kernels per head instead of seven per lag); the per-lag
        # entries below are views of them
        hit = (logits.argmax(-1) == y) & m
        L, X = ce.shape[1], ce.shape[2]
        loss_l, hit_l, m_l = ce.detach().sum((0, 2)), hit.sum((0, 2)), m.sum((0, 2))
        active_l = m.any(-1).sum(0)
        weighted_l = (w * terms.detach()).sum(0)
        for lag in range(L):
            key = f"{name}{lag}"
            counts[f"loss_{key}"] = loss_l[lag]
            if allowed is not None:
                counts[f"loss_set_{key}"] = obj[:, lag].sum().detach()
            counts[key] = (hit_l[lag], m_l[lag])
            counts[f"aux_capacity_{key}"] = ce.shape[0] * X
            counts[f"aux_active_{key}"] = (active_l[lag], m.shape[0])
            counts[f"loss_weighted_{key}"] = weighted_l[lag]
        counts["loss_aux_weighted"] = counts.get("loss_aux_weighted", 0.0) + per.sum().detach()
    return per


def aux_metric_sums(meter):
    """Keep pooled loss/support numerators through fit's logging aggregation."""
    return {k: v for k, v in meter.sums.items()
            if k.startswith(("loss_aux", "loss_weighted_aux", "aux_capacity_", "loss_set_", "set_multi_", "set_size_"))}


def aux_loss(out, x, w, net=None):
    """v6 training-only heads: weighted CE summed per example over labelled (>= 0) workers, order slots and lags, plus
    per-head loss sums and accuracies for the log. w = aux_weights(cfg). Returns (per_example [B] or None, counts)."""
    per, counts = None, {}
    def ce(logits, y, weight, name, set_key=None):
        allowed = None
        if set_key is not None and w.get("_aux_equiv", False):   # quantity heads only; token sets are singleton (RETURN-1/2)
            allowed = x.get(set_key)
            if allowed is None:
                raise ValueError(f"aux_equiv_loss needs {set_key} (equiv_dir side-cars of both seats)")
        return _aux_ce(logits, y, weight, name, counts, reduction=w.get("_reduction", "sum"),
                       lag_decay=w.get("_lag_decay", 1.0), allowed=allowed)
    def add(term):
        nonlocal per
        per = term if per is None else per + term
    for key, lab_key, wk, name in (("aux_next_logits", "next_unit_tok", "aux_next_w", "aux_next"),
                                   ("aux_opp_logits", "opp_unit_tok", "aux_opp_w", "aux_opp")):
        if key in out and x.get(lab_key) is not None:
            add(ce(out[key].transpose(1, 2), x[lab_key].long(), w.get(wk, 0.0), name))   # [B, L, U, V]
    if net is None or "aux_mem" not in out:
        return per, counts
    for which, tok_key, qty_key, wk, name in (("next_unit", "next_unit_tok", "next_unit_qty", "aux_next_qty_w", "aux_nextq"),
                                              ("opp_unit", "opp_unit_tok", "opp_unit_qty", "aux_opp_qty_w", "aux_oppq")):
        if (w.get(wk, 0.0) or w.get("_metrics_all", False)) and x.get(qty_key) is not None and (which == "next_unit" or out.get("aux_opp_h") is not None):
            tok = x[tok_key].long()
            add(ce(net.aux_qty_logits(out, which, tok), x[qty_key].long(), w.get(wk, 0.0), name, qty_key + "_set"))
    for which, sl, tok_key, qty_key, wk, name in (("next_order", slice(0, 2), "next_order_tok", "next_order_qty",
                                                   "aux_next_order_w", "aux_nexto"),
                                                  ("opp_order", slice(2, 5), "opp_order_tok", "opp_order_qty",
                                                   "aux_opp_order_w", "aux_oppo")):
        if (w.get(wk, 0.0) or w.get("_metrics_all", False)) and x.get(tok_key) is not None:
            tok = x[tok_key].long()
            logits, hs = net.aux_order_decode(out, which, tok, x[qty_key].long())
            out.setdefault("aux_order_hs", {})[which] = hs
            add(ce(logits, tok, w.get(wk, 0.0), name))
            add(ce(net.aux_qty_logits(out, which, tok), x[qty_key].long(), w.get(wk, 0.0) * w.get("_order_qty_scale", 1.0),
                   name + "q", qty_key + "_set"))
    return per, counts


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
    return {"free_unit_exact": (unit_ok.sum(), present.sum()),
            "free_order_exact": (order_ok.sum(), ovalid.sum()),
            "free_step_exact": (step_ok.sum(), step_ok.numel())}


class Meter:
    def __init__(self):
        self.pairs, self.sums = {}, {}

    def add(self, counts):
        # A key's first value is stored as is (0 + v would launch one copy kernel per tensor); bool tensors still promote.
        first = lambda v: v if torch.is_tensor(v) and v.dtype != torch.bool else 0 + v
        for k, v in counts.items():
            if isinstance(v, tuple):
                if k in self.pairs:
                    a, b = self.pairs[k]
                    self.pairs[k] = (a + v[0], b + v[1])
                else:
                    self.pairs[k] = (first(v[0]), first(v[1]))
            else:
                self.sums[k] = self.sums[k] + v if k in self.sums else first(v)

    def materialize(self):
        """Device-tensor counts (kept on the GPU so a step costs no host sync) -> Python numbers with one sync."""
        refs, ts = [], []
        for k, v in self.sums.items():
            if torch.is_tensor(v):
                refs.append((k, None)); ts.append(v)
        for k, ab in self.pairs.items():
            for j, x in enumerate(ab):
                if torch.is_tensor(x):
                    refs.append((k, j)); ts.append(x)
        if ts:
            dev = ts[0].device
            vals = torch.stack([t.detach().reshape(()).to(dev, torch.float64) for t in ts]).tolist()
            for (k, j), t, v in zip(refs, ts, vals):
                v = v if t.is_floating_point() else int(round(v))
                if j is None:
                    self.sums[k] = v
                else:
                    ab = list(self.pairs[k]); ab[j] = v; self.pairs[k] = tuple(ab)
        return self

    def report(self):
        self.materialize()
        out = {f"acc_{k}": (a / b if b else None) for k, (a, b) in self.pairs.items()}
        out.update({f"n_{k}": b for k, (a, b) in self.pairs.items()})
        out.update(self.sums)
        if "acc_recur_loops_mean" in out:
            out["recur_loops_mean"] = out.pop("acc_recur_loops_mean")
            out["recur_updates"] = out.pop("n_recur_loops_mean")
        for key, (_, n) in self.pairs.items():
            if key.startswith("aux_") and f"loss_{key}" in self.sums:
                out[f"ce_{key}"] = self.sums[f"loss_{key}"] / n if n else None
                capacity = self.sums.get(f"aux_capacity_{key}", 0)
                out[f"coverage_{key}"] = n / capacity if capacity else None
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
    # a pageable host->device copy synchronizes the stream; pinned copies stay asynchronous
    pin = torch.device(device).type == "cuda"
    # nonzero() returns sorted indices.  When every example is active (always true for
    # a pack_batch chunk), a view avoids a second advanced-indexing copy of each array.
    full = len(idx) == batch["step_valid"].shape[0]
    out = {}
    for k in STEP_KEYS:
        if k not in batch:
            continue
        x = batch[k][:, t] if full else batch[k][idx, t]
        if pin and not x.is_cuda:
            x = x.pin_memory()
        out[k] = x.to(device, non_blocking=True)
    return out


def pack_batch(batch, chunk):
    """The scored (window, step) pairs of a [B, T] window batch as [n, 1] batches of <= chunk rows. Valid only for stateless
    models (think != bottleneck): every step's inputs are complete on their own (history arrays are per-step inputs)."""
    B, T = batch["step_valid"].shape
    b_idx, t_idx = (batch["step_valid"] & batch["loss_mask"]).nonzero(as_tuple=True)
    out = []
    for s in range(0, len(b_idx), chunk):
        bi, ti = b_idx[s:s + chunk], t_idx[s:s + chunk]
        nb = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] == B and v.shape[1] == T:
                nb[k] = v[bi, ti].unsqueeze(1)
            elif torch.is_tensor(v) and v.dim() == 1 and v.shape[0] == B:
                nb[k] = torch.zeros(len(bi), dtype=v.dtype) if k == "first_valid" else v[bi]
            else:
                nb[k] = v
        out.append(nb)
    return out


def run_window_batch(net, batch, *, device, tau=1.0, code_mode="gumbel", generator=None, bf16=True,
                     grad_checkpoint=False, qty_weight=1.0, usage_reg=0.0, free_run=False,
                     code_stats=None, keep_first_codes=False, aux_w=None, mask_unit_noops=False, use_sets=False,
                     prefix_aug=False, aux_detail_on=False, return_parts=False,
                     mask_order_noops=False, infeasible_w=0.0, order_infeasible_w=0.0,
                     commitment_weighting=False, order_weight_scale=None):
    """One forward over a [B, T] window batch.  Returns (mean loss over scored example-steps,
    count dict, per-step losses list, codes written at the first scored step).  aux_w = aux_weights(cfg): weights of the
    v6 training-only heads; they join the loss only while net.training (eval losses stay comparable)."""
    aux_w = aux_w or {}
    if net.training and any(aux_w.get(k, 0.0) for k in AUX_W_KEYS[2:]) and not getattr(net.config, "v6_aux", False):
        raise ValueError("aux quantity/order weights require ModelConfig.v6_aux and v6_aux_turn")
    if mask_unit_noops:
        from noop_mask import SCHEMA
        if batch.get("unit_noop_mask_schema") != SCHEMA or "unit_loss_keep" not in batch:
            raise ValueError("mask_unit_noops requires unit-noop-mask-v1 unit_loss_keep; rebuild or augment the step cache")
    if net.training and (mask_order_noops or infeasible_w or order_infeasible_w):
        from loss_supervision import validate_batch
        validate_batch(batch, mask_orders=mask_order_noops, unit_weight=infeasible_w,
                       order_weight=order_infeasible_w)
    if net.training and commitment_weighting and "order_token_weight" not in batch:
        raise ValueError("commitment weighting requires order_token_weight sidecar labels")
    B, T = batch["step_valid"].shape
    if net.training and getattr(net.config, "v6_aux", False):
        need = {"aux_next_w": ("next_unit_tok",), "aux_opp_w": ("opp_unit_tok",),
                "aux_next_qty_w": ("next_unit_tok", "next_unit_qty"), "aux_opp_qty_w": ("opp_unit_tok", "opp_unit_qty"),
                "aux_next_order_w": ("next_order_tok", "next_order_qty"), "aux_opp_order_w": ("opp_order_tok", "opp_order_qty")}
        for wk, keys in need.items():
            for key in keys:
                if aux_w.get(wk, 0.0) != 0 and batch.get(key) is None:
                    raise ValueError(f"enabled v6 aux loss {wk} requires {key}; rebuild the step cache with the aux-turn labels")
        if any(aux_w.get(k, 0.0) for k in AUX_W_KEYS[2:]) and not getattr(net.config, "v6_aux_turn", False):
            raise ValueError("aux quantity/order weights need ModelConfig.v6_aux_turn")
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
        didx = idx.to(device) if bottleneck else None   # a blocking host->device copy (stream sync): bottleneck only
        f_in = M.FieldState(field.codes.index_select(0, didx), field.fresh.index_select(0, didx)) if bottleneck else None
        noise = None
        if bottleneck and code_mode != "argmax":
            noise = torch.rand((len(idx), net.S, net.config.C, net.config.K), generator=generator, device=device)
        grow = {"extra": x.get("extra"), "unit_extra": x.get("unit_extra"), "reach_eta": x.get("reach_eta"),
                "unit_prefix": x.get("unit_prefix"), "order_prefix": x.get("order_prefix"),
                "order_cash": x.get("order_cash"),
                "v6_token": x.get("v6_token"), "v6_resource_eta": x.get("v6_resource_eta"),
                "v6_resource_route": x.get("v6_resource_route"), "v6_job_deadline": x.get("v6_job_deadline"),
                "v6_job_deadline_present": x.get("v6_job_deadline_present"),
                "prev_unit": x.get("prev_unit"), "prev_order": x.get("prev_order"),
                "x8_token": x.get("x8_token"), "x8_unit": x.get("x8_unit"), "lattice": x.get("lattice")}
        if not scored:
            with torch.no_grad(), autocast(device, bf16):
                ctx = net.encode(x["rows"], x["token_mask"], x["knownness"], f_in, tau=tau,
                                 code_mode=code_mode, code_noise=noise, **grow)
            if bottleneck:
                field = M.FieldState(field.codes.index_copy(0, didx, ctx.field.codes.float()),
                                     field.fresh.index_fill(0, didx, False))
            continue
        uq_in, oq_in = x["unit_qty"], x["order_qty"]
        if prefix_aug and net.training and "unit_qty_set" in x:   # certified alias as the teacher-forced quantity prefix
            uq_in, oq_in = sample_alias(uq_in, x["unit_qty_set"]), sample_alias(oq_in, x["order_qty_set"])
        up = torch.stack((x["unit_tok"], uq_in), -1)
        op = torch.stack((x["order_tok"], oq_in), -1)
        args = (x["rows"], x["token_mask"], x["knownness"], x["unit_pos"], up, op)
        pred = None
        with autocast(device, bf16):
            if grad_checkpoint and torch.is_grad_enabled():
                def fn(*a, _noise=noise):
                    field_in = M.FieldState(a[6], a[7]) if bottleneck else None
                    o = net(*a[:6], field=field_in, tau=tau, code_mode=code_mode, code_noise=_noise,
                            extra=a[8], unit_extra=a[9], reach_eta=a[10], unit_prefix_state=a[11],
                            order_prefix_state=a[12], prev_unit=a[13], prev_order=a[14],
                            order_cash_state=a[15], v6_token=a[16], v6_resource_eta=a[17],
                            v6_resource_route=a[18], v6_job_deadline=a[19], v6_job_deadline_present=a[20],
                            x8_token=a[21], x8_unit=a[22], lattice=a[23])
                    f_out = o.pop("field")
                    return {**o, "codes": f_out.codes} if bottleneck else o
                extra = (f_in.codes, f_in.fresh) if bottleneck else (None, None)
                out = dict(torch.utils.checkpoint.checkpoint(fn, *args, *extra, grow["extra"], grow["unit_extra"],
                                                             grow["reach_eta"], grow["unit_prefix"], grow["order_prefix"],
                                                             grow["prev_unit"], grow["prev_order"], grow["order_cash"],
                                                             grow["v6_token"], grow["v6_resource_eta"],
                                                             grow["v6_resource_route"], grow["v6_job_deadline"],
                                                             grow["v6_job_deadline_present"], grow["x8_token"],
                                                             grow["x8_unit"], grow["lattice"],
                                                             use_reentrant=False))
                if bottleneck:
                    out["field"] = M.FieldState(out.pop("codes"), torch.zeros(len(idx), dtype=torch.bool, device=device))
            else:
                ctx = net.encode(*args[:3], f_in, tau=tau, code_mode=code_mode, code_noise=noise, **grow)
                out = net.decode_teacher(ctx, x["unit_pos"], up, op)
                out.update(value=ctx.value, field=ctx.field, code_logits=ctx.code_logits)
                if free_run:
                    with torch.no_grad():
                        pred = net.decode_free(ctx, x["unit_pos"], temperature=0.0)
        lab = (x["unit_tok"], x["unit_qty"], x["order_tok"], x["order_qty"])
        per_example, counts = step_loss(out, lab, net, qty_weight,
                                        unit_loss_keep=x["unit_loss_keep"] if mask_unit_noops else None,
                                        order_loss_keep=x["order_loss_keep"] if mask_order_noops and net.training else None,
                                        sets={h: x[f"{h}_set"] for h in ("unit_qty", "order_qty") if f"{h}_set" in x},
                                        use_sets=use_sets and net.training,
                                        order_token_weight=x["order_token_weight"] if commitment_weighting and net.training else None,
                                        order_weight_scale=order_weight_scale if commitment_weighting and net.training else None)
        bc_objective = per_example.sum() if return_parts else None
        if net.training and (infeasible_w or order_infeasible_w):
            from loss_supervision import infeasible_loss
            extra_loss, extra_counts = infeasible_loss(
                out, lab, x, unit_weight=infeasible_w, order_weight=order_infeasible_w,
                unit_keep=x["unit_loss_keep"] if mask_unit_noops else None,
                order_keep=x["order_loss_keep"] if mask_order_noops else None)
            per_example = per_example + extra_loss
            counts.update(extra_counts)
            if return_parts:
                counts["_supervision_obj"] = extra_loss.sum()
        with autocast(device, bf16):   # the aux order decoder otherwise runs fp32 (and fails if mem is bf16, dec_width != width)
            aux_example, aux_counts = aux_loss(out, x, aux_w, net)
            if aux_detail_on:
                with torch.no_grad():
                    aux_counts.update(aux_detail(out, x, net))
        counts.update(aux_counts)
        if return_parts:   # grad_probe: BC and aux objectives separately (sums over this call's scored examples)
            counts["_bc_obj"] = bc_objective
            counts["_aux_obj"] = aux_example.sum() if aux_example is not None else per_example.sum() * 0
        if aux_example is not None and net.training:
            per_example = per_example + aux_example
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


def save_checkpoint(out_dir, net, opt, cfg, *, update, windows_consumed, epoch, resumes, generator, metrics, ema=None):
    out_dir = Path(out_dir)
    blob = {"ema": ema} if ema is not None else {}
    blob |= {"schema": CKPT_SCHEMA, "config": net.config.to_dict(), "train_config": dataclasses.asdict(cfg),
            "model": net.state_dict(), "optimizer": opt.state_dict(), "update": update,
            "windows_consumed": windows_consumed, "epoch": epoch, "resumes": resumes,
            "generator_state": generator.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if next(net.parameters()).is_cuda and torch.cuda.is_initialized() else [],
            "optimizer_param_names": optimizer_param_names(net, opt),
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
    import std_init
    net = std_init.apply(M.PolicyNet(model_cfg, codec=codec)).zero_growth().to(device)
    return net


def make_optimizer(net, cfg):
    """Groups: decay, no-decay (the pre-growth layout, so old optimizer states load), then grow_* parameters
    (zero-initialised growth, no weight decay, lr x cfg.grow_lr_mult) when present."""
    decay, no_decay, grow, new = [], [], [], []
    grown = tuple(x for x in str(getattr(cfg, "grow_prefixes", "") or "").split(",") if x)
    fresh = tuple(x for x in str(getattr(cfg, "new_prefixes", "") or "").split(",") if x)
    for n, p in net.named_parameters():
        if fresh and n.startswith(fresh):
            new.append(p)
            continue
        if n.startswith("grow_") or (grown and n.startswith(grown)):
            grow.append(p)
            continue
        (decay if p.ndim >= 2 and "emb" not in n and not n.startswith(("query", "slot_in")) else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay, "lr_mult": 1.0},
              {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0}]
    if grow:
        groups.append({"params": grow, "weight_decay": 0.0, "lr_mult": cfg.grow_lr_mult})
    if new:
        groups.append({"params": new, "weight_decay": 0.0, "lr_mult": cfg.new_lr_mult, "new": True})
    if getattr(cfg, "optimizer", "adamw") == "muon":
        is_mat = lambda n, p: p.ndim == 2 and n.startswith(("blocks.", "dec.")) and n.endswith(".weight") and not n.startswith("grow_")
        mats = [p for n, p in net.named_parameters() if is_mat(n, p)]
        ids = {id(p) for p in mats}
        for g in groups:
            g["params"] = [p for p in g["params"] if id(p) not in ids]
        if not 0.0 <= cfg.muon_momentum < 1.0 or cfg.muon_ns_steps < 1:
            raise ValueError("Muon requires momentum in [0, 1) and positive ns_steps")
        muon = torch.optim.Muon([{"params": mats, "lr_mult": 1.0}], lr=cfg.lr, weight_decay=cfg.weight_decay,
                                momentum=cfg.muon_momentum, nesterov=True, ns_steps=cfg.muon_ns_steps,
                                adjust_lr_fn="match_rms_adamw")
        return MultiOptimizer([muon, torch.optim.AdamW([g for g in groups if g["params"]], lr=cfg.lr, betas=(0.9, cfg.beta2))])
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, cfg.beta2))


class MultiOptimizer:
    """Several optimizers stepped together (Muon for matrices + AdamW for the rest); param_groups is their concatenation so
    the per-group LR schedule applies unchanged."""

    def __init__(self, opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return {"multi": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        if "multi" not in sd or len(sd["multi"]) != len(self.opts):
            raise ValueError("incompatible multi-optimizer state count/type")
        for o, s in zip(self.opts, sd["multi"]):
            o.load_state_dict(s)


def optimizer_param_names(net, opt):
    names = {id(p): n for n, p in net.named_parameters()}
    return [[names[id(p)] for p in g["params"]] for g in opt.param_groups]


def restore_optimizer(net, opt, cfg, blob):
    """Restore moments, then apply requested group settings; never silently repartition moments."""
    old = blob.get("train_config", {})
    if old.get("optimizer", "adamw") != cfg.optimizer:
        raise ValueError("optimizer kind changed; create an explicit weights-only new run")
    if old.get("new_prefixes", "") != cfg.new_prefixes:
        raise ValueError("optimizer parameter partition changed (new_prefixes)")
    names = optimizer_param_names(net, opt)
    if "optimizer_param_names" in blob and names != blob["optimizer_param_names"]:
        raise ValueError("optimizer parameter groups/order changed; do not reuse these moments")
    # Parameter IDs/state stay loaded. Hyperparameters describe this run, not the old checkpoint.
    requested = [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]
    opt.load_state_dict(blob["optimizer"])
    if len(requested) != len(opt.param_groups):
        raise ValueError("optimizer group count changed")
    for group, settings in zip(opt.param_groups, requested):
        for key in tuple(group):
            if key != "params" and key not in settings:
                del group[key]
        group.update(settings)


def group_lr(update, cfg, group):
    """Scheduled LR of one parameter group; the fresh-graft group ramps its multiplier over cfg.new_warmup updates."""
    mult = group.get("lr_mult", 1.0)
    if group.get("new") and getattr(cfg, "new_warmup", 0):
        mult = 1.0 + (mult - 1.0) * min(1.0, max(0, update - cfg.new_from + 1) / cfg.new_warmup)
    return lr_at(update, cfg) * mult


# ----------------------------------------------------------------------------- quantity-equivalence side-cars (PRO-EQUIV-0928)
EQUIV_IDENTITY = ("episode_id", "seat", "engine_sha256", "codec_sha256", "builder_sha256", "labels_sha256", "configuration")
_EQUIV_HASHES = {}


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def equiv_expected_hashes():
    """Identity the side-cars must carry, computed from THIS code tree (never from the side-car header): engine, codec, the
    builder, and data.py (the label producer). A changed file invalidates every side-car (fail closed)."""
    if not _EQUIV_HASHES:
        import importlib.util
        engine = importlib.util.find_spec("farm_neat.game.kaggriculture_engine_reference").origin
        _EQUIV_HASHES.update(engine_sha256=_sha256_file(engine), codec_sha256=_sha256_file(HERE / "codec.py"),
                             builder_sha256=_sha256_file(HERE / "equiv_build.py"), labels_sha256=_sha256_file(HERE / "data.py"))
    return dict(_EQUIV_HASHES)


def load_equiv(path, episode_id, seat):
    """Load + validate one equiv_build.py side-car (RETURN-1 NPZ container, allow_pickle=False): identity against this code
    tree, per-array SHA-256, packed-bit padding, token sets singleton, quantity sets non-empty exactly on labelled rows and
    containing the label. Returns {"steps", "index", "labels": {key: [n, X]}, "exact", "sets": {unit_qty/order_qty: [n, X, 75]}}."""
    import hashlib
    exp = {**equiv_expected_hashes(), "episode_id": str(episode_id), "seat": int(seat)}
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(d["metadata"].tobytes())
        arr = {k: d[k].copy() for k in d.files if k != "metadata"}
    if meta.get("schema") != EQUIV_SCHEMA or meta.get("version") != 1 or meta.get("bitorder") != "little":
        raise ValueError(f"{path}: unsupported side-car schema")
    bad = [k for k in EQUIV_IDENTITY if k != "configuration" and meta.get(k) != exp[k]]
    if bad:
        raise ValueError(f"{path}: stale side-car identity {bad}; rebuild with equiv_build.py from this code tree")
    if set(meta.get("digests", {})) != set(arr) or any(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                                                        != meta["digests"][k] for k, a in arr.items()):
        raise ValueError(f"{path}: array checksum mismatch")
    n, out = int(meta["n"]), {"labels": {}, "sets": {}}
    for k, (X, V) in {"unit_tok": (25, 44), "unit_qty": (25, 75), "order_tok": (11, 23), "order_qty": (11, 75)}.items():
        y, bits = arr[k].astype(np.int64), arr[k + "_bits"]
        if y.shape != (n, X) or bits.shape != (n, X, (V + 7) // 8):
            raise ValueError(f"{path}: bad shape for {k}")
        full = np.unpackbits(bits, axis=-1, bitorder="little").astype(bool)
        if full[..., V:].any():
            raise ValueError(f"{path}: nonzero padding bits in {k}")
        m, act, size = full[..., :V], y >= 0, full[..., :V].sum(-1)
        if (size[~act] != 0).any() or (size[act] == 0).any() or not np.take_along_axis(m, y.clip(0)[..., None], -1)[..., 0][act].all():
            raise ValueError(f"{path}: {k} masks must be empty off-label and contain the label")
        if k.endswith("_tok") and (size[act] != 1).any():
            raise ValueError(f"{path}: token alternatives are not certified by {EQUIV_SCHEMA}")
        out["labels"][k] = y
        if k.endswith("_qty"):
            out["sets"][k] = m
    steps = arr["steps"].astype(np.int64)
    if steps.shape != (n,) or (np.diff(steps) <= 0).any():
        raise ValueError(f"{path}: bad step table")
    out.update(steps=steps, exact=arr["exact"].astype(bool), index=np.full(720, -1, np.int64))
    out["index"][steps] = np.arange(n)
    return out


def equiv_merge(arrs, stp_path, equiv_dir, aux=False):
    """Add unit_qty_set / order_qty_set (and with aux the four next_* / opp_* quantity sets) to one decoded cached seat, joined
    by (episode, seat, actual step). The side-car's labels must equal the cached labels at every step (own seat) and every
    active aux target (own t+1/t+2 from this seat's side-car, opponent t/t+1/t+2 from the other seat's): any mismatch or a
    missing source row for an active label raises. Inactive labels get empty masks. Token masks are not merged (singleton)."""
    eid, seat = Path(stp_path).stem.split("-")
    own = load_equiv(Path(equiv_dir) / f"{eid}-{seat}.equiv.npz", eid, int(seat))
    steps = np.asarray(arrs["step"]).astype(np.int64)
    rows = own["index"][steps]
    if (rows < 0).any():
        raise ValueError(f"{stp_path}: side-car lacks cached steps")
    for k in ("unit_tok", "unit_qty", "order_tok", "order_qty"):
        if not np.array_equal(np.asarray(arrs[k]).astype(np.int64), own["labels"][k][rows]):
            raise ValueError(f"{stp_path}: side-car labels differ from the cached {k}")
    if "exact" in arrs and not np.array_equal(np.asarray(arrs["exact"]).astype(bool), own["exact"][rows]):
        raise ValueError(f"{stp_path}: side-car exact flags differ from the cache")
    arrs["unit_qty_set"], arrs["order_qty_set"] = own["sets"]["unit_qty"][rows], own["sets"]["order_qty"][rows]
    if aux:
        opp = load_equiv(Path(equiv_dir) / f"{eid}-{1 - int(seat)}.equiv.npz", eid, 1 - int(seat))
        for pre, lags, src in (("next_", (1, 2), own), ("opp_", (0, 1, 2), opp)):
            for k in ("unit_qty", "order_qty"):
                y = np.asarray(arrs[pre + k]).astype(np.int64)                       # [n, L, X]
                m = np.zeros(y.shape + (75,), bool)
                for li, lag in enumerate(lags):
                    act = y[:, li] >= 0
                    src_t = steps + lag
                    srow = np.where(src_t <= 719, src["index"][np.minimum(src_t, 719)], -1)
                    need = act.any(-1)
                    if (need & (srow < 0)).any():
                        raise ValueError(f"{stp_path}: no side-car row for an active {pre}{k} target")
                    r = np.where(need, srow, 0)
                    truth = src["labels"][k][r]
                    if not np.array_equal(np.where(act, y[:, li], -1), np.where(act, truth, -1)):
                        raise ValueError(f"{stp_path}: {pre}{k} labels differ from the target turn's side-car")
                    m[:, li] = src["sets"][k][r] & act[..., None]
                arrs[pre + k + "_set"] = m
    return arrs


class EquivChunks:
    """step_cache.StepChunks with the side-car merged when a seat is loaded (lives here, not in step_cache.py: that file is
    hashed into every cache identity). Same sampling and collation as StepChunks."""

    def __new__(cls, cache_dir, equiv_dir, chunk, seed, pool_seats=4, aux=False):
        import step_cache

        class _Chunks(step_cache.StepChunks):
            def __iter__(self):
                wi = torch.utils.data.get_worker_info()
                wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
                rng = random.Random(self.seed * 9_999_991 + wid)
                seats = self._seats(wid, nw)

                def load():
                    path = next(seats)
                    arrs, dt = step_cache.decode(path.read_bytes())
                    arrs = equiv_merge(dict(arrs), path, equiv_dir, aux=aux)
                    idx = list(range(arrs["step"].shape[0]))
                    if not idx:
                        raise ValueError("empty cached seat has zero steps")
                    rng.shuffle(idx)
                    return [arrs, idx], dt

                live, dtypes = [], None
                while len(live) < self.pool_seats:
                    e, dtypes = load()
                    live.append(e)
                while True:
                    take = []
                    for _ in range(self.chunk):
                        j = rng.randrange(len(live))
                        arrs, idx = live[j]
                        take.append((arrs, idx.pop()))
                        if not idx:
                            live[j], dtypes = load()
                    yield step_cache.collate_steps(take, dtypes, self.noop_schema)

        return _Chunks(cache_dir, chunk, seed, pool_seats=pool_seats)


def aux_detail(out, x, net):
    """Report-only aux metrics (no gradient): top-3; order sequences per lag (exact sequence, STOP slot, bag-of-orders true
    positives / predicted / labelled with the argmax prefix up to its first STOP, per order token too); copy-the-current-
    turn (own heads) and STOP-only (order heads) baselines; decoded-count closeness of argmax quantities on count bins
    (bin 74 is reference-relative and only counts when exact); 10-bin top-1 calibration sums per token head."""
    c = {}
    stop, V = int(net.config.stop_index), int(net.config.n_order_tokens)
    cnt = _qty_count(int(net.config.n_qty), x["unit_tok"].device)

    def calib(name, logits, y):
        m = y >= 0
        p = logits.float().softmax(-1)
        conf, pred = p.max(-1)
        top3 = (logits.topk(3, -1).indices == y[..., None]).any(-1) & m
        for lag in range(y.shape[1]):
            c[f"{name}{lag}_top3"] = (top3[:, lag].sum(), m[:, lag].sum())
        b = (conf * 10).long().clamp(max=9)
        for i in range(10):
            sel = m & (b == i)
            c[f"cal_{name}_{i}"] = (((pred == y) & sel).sum(), sel.sum())
            c[f"calconf_{name}_{i}"] = (conf * sel).sum()

    def near(name, pred, y):
        m = y >= 0
        vp, vy = cnt[pred], cnt[y.clamp(min=0)]
        fin = torch.isfinite(vp) & torch.isfinite(vy)
        ok = ((((vp - vy).abs() <= 1) & fin) | (pred == y)) & m
        c[f"{name}_near1"] = (ok.sum(), m.sum())
        both = m & fin
        c[f"{name}_log2ratio"] = (torch.log2(vp.where(both, 1.0)) - torch.log2(vy.where(both, 1.0))).abs().sum()
        c[f"{name}_log2n"] = both.sum()

    near("main_unit_qty", out["unit_qty_logits"].argmax(-1), torch.where(x["unit_qty"] >= 0, x["unit_qty"], -1).long())
    near("main_order_qty", out["order_qty_logits"].argmax(-1), torch.where(x["order_qty"] >= 0, x["order_qty"], -1).long())
    for key, lab_key, name in (("aux_next_logits", "next_unit_tok", "aux_next"), ("aux_opp_logits", "opp_unit_tok", "aux_opp")):
        if key in out and x.get(lab_key) is not None:
            calib(name, out[key].transpose(1, 2), x[lab_key].long())
    if x.get("next_unit_tok") is not None:
        y, cur = x["next_unit_tok"].long(), x["unit_tok"].long()
        for lag in range(y.shape[1]):
            m = y[:, lag] >= 0
            c[f"base_copy_aux_next{lag}"] = (((y[:, lag] == cur) & m).sum(), m.sum())
    if "aux_order_hs" not in out:   # set by aux_loss (the causal aux order decoder, teacher-forced on the target prefix)
        return c
    for which, sl, tok_key, qty_key, name in (("next_order", slice(0, 2), "next_order_tok", "next_order_qty", "aux_nexto"),
                                              ("opp_order", slice(2, 5), "opp_order_tok", "opp_order_qty", "aux_oppo")):
        if x.get(tok_key) is None or which not in out["aux_order_hs"]:
            continue
        lg, y = net.aux_order_head(out["aux_order_hs"][which]).float(), x[tok_key].long()   # teacher-forced [B, L, O, V]
        calib(name, lg, y)
        pred, O = lg.argmax(-1), y.shape[-1]
        pos = torch.arange(O, device=y.device)
        lab = (y >= 0).any(-1)
        tlen = torch.where(y == stop, pos, O).min(-1).values
        plen = torch.where(pred == stop, pos, O).min(-1).values
        seq_ok = ((pred == y) | (y < 0)).all(-1) & lab

        def bag(tok, n):
            keep = (pos < n[..., None]) & (tok >= 0) & (tok != stop)
            return torch.zeros(*tok.shape[:2], V, device=tok.device).scatter_add_(-1, tok.clamp(min=0), keep.float())
        bt, bp = bag(y, tlen), bag(pred, plen)
        lf = lab[..., None].float()
        tp = torch.minimum(bt, bp) * lf
        for lag in range(y.shape[1]):
            c[f"{name}{lag}_seq"] = (seq_ok[:, lag].sum(), lab[:, lag].sum())
            c[f"{name}{lag}_len"] = (((plen == tlen) & lab)[:, lag].sum(), lab[:, lag].sum())
            c[f"{name}{lag}_bagP"] = (tp[:, lag].sum(), (bp * lf)[:, lag].sum())
            c[f"{name}{lag}_bagR"] = (tp[:, lag].sum(), (bt * lf)[:, lag].sum())
            c[f"base_empty_{name}{lag}"] = (((tlen == 0) & lab)[:, lag].sum(), lab[:, lag].sum())
        tpv, btv, bpv = tp.sum((0, 1)), (bt * lf).sum((0, 1)), (bp * lf).sum((0, 1))
        for v in range(V):
            c[f"{name}_tokR_{v}"], c[f"{name}_tokP_{v}"] = (tpv[v], btv[v]), (tpv[v], bpv[v])
        if name == "aux_nexto":
            base = ((x["order_tok"].long()[:, None] == y) | (y < 0)).all(-1) & lab
            for lag in range(y.shape[1]):
                c[f"base_copy_{name}{lag}_seq"] = (base[:, lag].sum(), lab[:, lag].sum())
        if x.get(qty_key) is not None:
            q = x[qty_key].long()
            pq = net.aux_qty_logits(out, which, y).argmax(-1)
            for lag in range(q.shape[1]):
                near(f"{name}q{lag}", pq[:, lag], q[:, lag])
    return c


def aux_summary(rep, cfg, n):
    """Per-step weighted aux contribution by head under the checkpoint's weights, share of canonical BC on the same steps, and
    derived ratios from an evaluate_aux(detail=True) report (rep = its meter report)."""
    bc = sum(rep.get(f"loss_{h}", 0.0) for h in ("unit_tok", "unit_qty", "order_tok", "order_qty")) / n
    heads = {}
    for k, v in rep.items():
        if k.startswith("loss_weighted_aux_"):
            head = k[len("loss_weighted_"):].rstrip("0123456789")
            heads[head] = heads.get(head, 0.0) + v / n
    out = {"bc_ce_per_step": bc, "aux_weighted_per_step": heads, "aux_to_bc": sum(heads.values()) / bc if bc else None}
    for k in list(rep):
        if k.endswith("_log2ratio"):
            m = rep.get(k[:-len("_log2ratio")] + "_log2n", 0)
            out[k[:-len("_log2ratio")] + "_mean_abs_log2_ratio"] = rep[k] / m if m else None
    for name in {k[4:].rsplit("_", 1)[0] for k in rep if k.startswith("n_cal_")}:
        tot = sum(rep.get(f"n_cal_{name}_{i}", 0) or 0 for i in range(10))
        if tot:
            out[f"ece_{name}"] = sum(abs((rep.get(f"acc_cal_{name}_{i}") or 0.0) * (rep.get(f"n_cal_{name}_{i}") or 0)
                                         - rep.get(f"calconf_{name}_{i}", 0.0)) for i in range(10)) / tot
    return out


# ----------------------------------------------------------------------------- quantity-equivalence side-cars (PRO-EQUIV-0928)
EQUIV_IDENTITY = ("episode_id", "seat", "engine_sha256", "codec_sha256", "builder_sha256", "labels_sha256", "configuration")
_EQUIV_HASHES = {}


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def equiv_expected_hashes():
    """Identity the side-cars must carry, computed from THIS code tree (never from the side-car header): engine, codec, the
    builder, and data.py (the label producer). A changed file invalidates every side-car (fail closed)."""
    if not _EQUIV_HASHES:
        import importlib.util
        engine = importlib.util.find_spec("farm_neat.game.kaggriculture_engine_reference").origin
        _EQUIV_HASHES.update(engine_sha256=_sha256_file(engine), codec_sha256=_sha256_file(HERE / "codec.py"),
                             builder_sha256=_sha256_file(HERE / "equiv_build.py"), labels_sha256=_sha256_file(HERE / "data.py"))
    return dict(_EQUIV_HASHES)


def load_equiv(path, episode_id, seat):
    """Load + validate one equiv_build.py side-car (RETURN-1 NPZ container, allow_pickle=False): identity against this code
    tree, per-array SHA-256, packed-bit padding, token sets singleton, quantity sets non-empty exactly on labelled rows and
    containing the label. Returns {"steps", "index", "labels": {key: [n, X]}, "exact", "sets": {unit_qty/order_qty: [n, X, 75]}}."""
    import hashlib
    exp = {**equiv_expected_hashes(), "episode_id": str(episode_id), "seat": int(seat)}
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(d["metadata"].tobytes())
        arr = {k: d[k].copy() for k in d.files if k != "metadata"}
    if meta.get("schema") != EQUIV_SCHEMA or meta.get("version") != 1 or meta.get("bitorder") != "little":
        raise ValueError(f"{path}: unsupported side-car schema")
    bad = [k for k in EQUIV_IDENTITY if k != "configuration" and meta.get(k) != exp[k]]
    if bad:
        raise ValueError(f"{path}: stale side-car identity {bad}; rebuild with equiv_build.py from this code tree")
    if set(meta.get("digests", {})) != set(arr) or any(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                                                        != meta["digests"][k] for k, a in arr.items()):
        raise ValueError(f"{path}: array checksum mismatch")
    n, out = int(meta["n"]), {"labels": {}, "sets": {}}
    for k, (X, V) in {"unit_tok": (25, 44), "unit_qty": (25, 75), "order_tok": (11, 23), "order_qty": (11, 75)}.items():
        y, bits = arr[k].astype(np.int64), arr[k + "_bits"]
        if y.shape != (n, X) or bits.shape != (n, X, (V + 7) // 8):
            raise ValueError(f"{path}: bad shape for {k}")
        full = np.unpackbits(bits, axis=-1, bitorder="little").astype(bool)
        if full[..., V:].any():
            raise ValueError(f"{path}: nonzero padding bits in {k}")
        m, act, size = full[..., :V], y >= 0, full[..., :V].sum(-1)
        if (size[~act] != 0).any() or (size[act] == 0).any() or not np.take_along_axis(m, y.clip(0)[..., None], -1)[..., 0][act].all():
            raise ValueError(f"{path}: {k} masks must be empty off-label and contain the label")
        if k.endswith("_tok") and (size[act] != 1).any():
            raise ValueError(f"{path}: token alternatives are not certified by {EQUIV_SCHEMA}")
        out["labels"][k] = y
        if k.endswith("_qty"):
            out["sets"][k] = m
    steps = arr["steps"].astype(np.int64)
    if steps.shape != (n,) or (np.diff(steps) <= 0).any():
        raise ValueError(f"{path}: bad step table")
    out.update(steps=steps, exact=arr["exact"].astype(bool), index=np.full(720, -1, np.int64))
    out["index"][steps] = np.arange(n)
    return out


def equiv_merge(arrs, stp_path, equiv_dir, aux=False):
    """Add unit_qty_set / order_qty_set (and with aux the four next_* / opp_* quantity sets) to one decoded cached seat, joined
    by (episode, seat, actual step). The side-car's labels must equal the cached labels at every step (own seat) and every
    active aux target (own t+1/t+2 from this seat's side-car, opponent t/t+1/t+2 from the other seat's): any mismatch or a
    missing source row for an active label raises. Inactive labels get empty masks. Token masks are not merged (singleton)."""
    eid, seat = Path(stp_path).stem.split("-")
    own = load_equiv(Path(equiv_dir) / f"{eid}-{seat}.equiv.npz", eid, int(seat))
    steps = np.asarray(arrs["step"]).astype(np.int64)
    rows = own["index"][steps]
    if (rows < 0).any():
        raise ValueError(f"{stp_path}: side-car lacks cached steps")
    for k in ("unit_tok", "unit_qty", "order_tok", "order_qty"):
        if not np.array_equal(np.asarray(arrs[k]).astype(np.int64), own["labels"][k][rows]):
            raise ValueError(f"{stp_path}: side-car labels differ from the cached {k}")
    if "exact" in arrs and not np.array_equal(np.asarray(arrs["exact"]).astype(bool), own["exact"][rows]):
        raise ValueError(f"{stp_path}: side-car exact flags differ from the cache")
    arrs["unit_qty_set"], arrs["order_qty_set"] = own["sets"]["unit_qty"][rows], own["sets"]["order_qty"][rows]
    if aux:
        opp = load_equiv(Path(equiv_dir) / f"{eid}-{1 - int(seat)}.equiv.npz", eid, 1 - int(seat))
        for pre, lags, src in (("next_", (1, 2), own), ("opp_", (0, 1, 2), opp)):
            for k in ("unit_qty", "order_qty"):
                y = np.asarray(arrs[pre + k]).astype(np.int64)                       # [n, L, X]
                m = np.zeros(y.shape + (75,), bool)
                for li, lag in enumerate(lags):
                    act = y[:, li] >= 0
                    src_t = steps + lag
                    srow = np.where(src_t <= 719, src["index"][np.minimum(src_t, 719)], -1)
                    need = act.any(-1)
                    if (need & (srow < 0)).any():
                        raise ValueError(f"{stp_path}: no side-car row for an active {pre}{k} target")
                    r = np.where(need, srow, 0)
                    truth = src["labels"][k][r]
                    if not np.array_equal(np.where(act, y[:, li], -1), np.where(act, truth, -1)):
                        raise ValueError(f"{stp_path}: {pre}{k} labels differ from the target turn's side-car")
                    m[:, li] = src["sets"][k][r] & act[..., None]
                arrs[pre + k + "_set"] = m
    return arrs


class EquivChunks:
    """step_cache.StepChunks with the side-car merged when a seat is loaded (lives here, not in step_cache.py: that file is
    hashed into every cache identity). Same sampling and collation as StepChunks."""

    def __new__(cls, cache_dir, equiv_dir, chunk, seed, pool_seats=4, aux=False):
        import step_cache

        class _Chunks(step_cache.StepChunks):
            def __iter__(self):
                wi = torch.utils.data.get_worker_info()
                wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
                rng = random.Random(self.seed * 9_999_991 + wid)
                seats = self._seats(wid, nw)

                def load():
                    path = next(seats)
                    arrs, dt = step_cache.decode(path.read_bytes())
                    arrs = equiv_merge(dict(arrs), path, equiv_dir, aux=aux)
                    idx = list(range(arrs["step"].shape[0]))
                    if not idx:
                        raise ValueError("empty cached seat has zero steps")
                    rng.shuffle(idx)
                    return [arrs, idx], dt

                live, dtypes = [], None
                while len(live) < self.pool_seats:
                    e, dtypes = load()
                    live.append(e)
                while True:
                    take = []
                    for _ in range(self.chunk):
                        j = rng.randrange(len(live))
                        arrs, idx = live[j]
                        take.append((arrs, idx.pop()))
                        if not idx:
                            live[j], dtypes = load()
                    yield step_cache.collate_steps(take, dtypes, self.noop_schema)

        return _Chunks(cache_dir, chunk, seed, pool_seats=pool_seats)


def cache_batches(cache_dir, chunk, device, max_steps=0, equiv_dir="", equiv_aux=False):
    """Every cached step of a step_cache.py directory in a fixed order (seat files sorted, steps in order) as device chunks;
    with equiv_dir the certified quantity-alias masks are merged (equiv_merge, fail closed)."""
    import step_cache
    meta = step_cache.validate_metadata(Path(cache_dir) / "CACHE.json")
    take, n, dt = [], 0, None
    for f in sorted(Path(cache_dir).glob("*.stp")):
        arrs, dt = step_cache.decode(f.read_bytes())
        if equiv_dir:
            arrs = equiv_merge(dict(arrs), f, equiv_dir, aux=equiv_aux)
        for i in range(arrs["step"].shape[0]):
            take.append((arrs, i))
            n += 1
            if len(take) == chunk or (max_steps and n >= max_steps):
                yield step_cache.to_device(step_cache.collate_steps(take, dt, meta.get("noop_mask_schema")), device)
                take = []
            if max_steps and n >= max_steps:
                return
    if take:
        yield step_cache.to_device(step_cache.collate_steps(take, dt, meta.get("noop_mask_schema")), device)


def grad_probe(net, batch, cfg, device):
    """Shared-parameter gradient norms of canonical BC and of the configured weighted aux objective on one batch (dropout
    off, heads on, no optimizer step, .grad untouched): norm ratio and cosine over parameters reached by both."""
    saved = {k: (hasattr(net, k), getattr(net, k, None)) for k in ("_aux_eval", "recur_T")}
    was = net.training
    net.eval()
    net._aux_eval, net.recur_T = True, None
    try:
        _, m, _, _ = run_window_batch(net, batch, device=device, code_mode="argmax", bf16=cfg.bf16, qty_weight=1.0,
                                      aux_w=aux_weights(cfg), return_parts=True)
        bc, aux = m.sums.pop("_bc_obj"), m.sums.pop("_aux_obj")
        params = [p for p in net.parameters() if p.requires_grad]
        gb = torch.autograd.grad(bc, params, retain_graph=True, allow_unused=True)
        ga = torch.autograd.grad(aux, params, allow_unused=True) if aux.requires_grad else [None] * len(params)
    finally:
        net.train(was)
        for key, (exists, value) in saved.items():
            if exists:
                setattr(net, key, value)
            else:
                delattr(net, key)
    pairs = [(b.float().flatten(), a.float().flatten()) for b, a in zip(gb, ga) if b is not None and a is not None]
    if not pairs:
        return {"shared_params": 0}
    b, a = torch.cat([q[0] for q in pairs]), torch.cat([q[1] for q in pairs])
    return {"shared_params": int(b.numel()), "aux_to_bc_grad_norm": float(a.norm() / b.norm().clamp_min(1e-12)),
            "cosine": float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12)), "bc_obj": float(bc.detach()),
            "aux_obj": float(aux.detach())}


def evaluate_aux(net, batches, cfg, device, detail=False):
    """Deterministic auxiliary diagnostics, separate from canonical BC VAL.

    Requires v6 labels, keeps dropout/feature-dropout OFF, never adds aux to
    the returned BC loss. All built heads are measured even at zero weight.
    Training flags and diagnostic/recurrent attributes are restored on error.
    """
    if not getattr(net.config, "v6_aux", False) or net.config.think != "off":
        raise ValueError("evaluate_aux requires a stateless v6_aux model")
    flags = [(module, module.training) for module in net.modules()]
    saved = {k: (hasattr(net, k), getattr(net, k, None)) for k in ("_aux_eval", "recur_T")}
    net.eval()
    net._aux_eval, net.recur_T = True, None
    meter, loss_sum, n = Meter(), 0.0, 0
    weights = {**aux_weights(cfg), "_metrics_all": True}
    needed = ("next_unit_tok", "opp_unit_tok")
    if getattr(net.config, "v6_aux_turn", False):
        needed += ("next_unit_qty", "opp_unit_qty", "next_order_tok", "next_order_qty", "opp_order_tok", "opp_order_qty")
    try:
        with torch.no_grad():
            for batch in batches:
                for key in needed:
                    if batch.get(key) is None:
                        raise ValueError(f"auxiliary VAL requires {key}")
                loss, m, _, _ = run_window_batch(net, batch, device=device, code_mode="argmax", bf16=cfg.bf16,
                                                 qty_weight=1.0, free_run=False, aux_w=weights, aux_detail_on=detail)
                m.materialize()
                count = m.sums.get("scored_example_steps", 0)
                n += count
                loss_sum += float(loss) * count
                meter.add(m.pairs)
                meter.add(m.sums)
        if not n:
            raise ValueError("auxiliary VAL contains no scored examples")
        return {**meter.report(), "loss_per_step": loss_sum / n, "eval_aux_in_loss": False,
                "eval_aux_schema": "aux-turn-diagnostics-v1", "aux_dropout_enabled": False,
                "aux_own_worker_conditioning": "teacher-forced current own worker prefix"}
    finally:
        for module, training in flags:
            module.training = training
        for key, (exists, value) in saved.items():
            if exists:
                setattr(net, key, value)
            else:
                delattr(net, key)


def evaluate(net, windows_or_batches, cfg, device, *, seed=12345, free_run=True, code_mode=None, recur_loops=None):
    """Canonical, unweighted BC metrics on held-out windows; aux heads excluded.

    All four CE heads have coefficient 1 regardless of checkpoint train settings.
    Preserve the original sum-per-scored-step reduction (including order STOP).
    Training weights/masks must never be threaded into this evaluator implicitly.
    """
    was_training, was_T = net.training, getattr(net, "recur_T", None)   # PRO-BH-DATALOADER-0927 #7: restore on exit
    if recur_loops is not None and (isinstance(recur_loops, bool) or not isinstance(recur_loops, int) or recur_loops < 1):
        raise ValueError("recur_loops must be a positive integer or None")
    if recur_loops is not None and not net.config.recur_core:
        raise ValueError("explicit recur_loops requires a recurrent model")
    net.eval()
    net.recur_T = recur_loops   # None preserves the configured deployment depth
    try:
        meter, stats = Meter(), (CodeStats(net.S, net.config.C, net.config.K) if net.S else None)
        gen = torch.Generator(device=device).manual_seed(seed)
        mode = code_mode or net.config.infer_code_mode
        loss_sum, n = 0.0, 0
        with torch.no_grad():
            for batch in windows_or_batches:
                loss, m, _, _ = run_window_batch(net, batch, device=device, tau=1.0, code_mode=mode, generator=gen,
                                                 bf16=cfg.bf16, qty_weight=1.0, free_run=free_run,
                                                 code_stats=stats)
                k = m.sums.get("scored_example_steps", 0)
                loss_sum += loss.item() * k
                n += k
                for key, v in m.pairs.items():
                    meter.add({key: v})
                meter.add({k2: v for k2, v in m.sums.items()})
        if n == 0:
            raise ValueError("evaluation contains no scored example steps")
        report = meter.report()
        report["loss_per_step"] = loss_sum / n
        report["eval_loss_schema"] = "primitive-bc-canonical-ce-v1"
        report["eval_qty_weight"] = 1.0
        report["eval_aux_in_loss"] = False
        report["eval_recur_loops"] = int(recur_loops or net.config.recur_loops)
        heads = ("unit_tok", "unit_qty", "order_tok", "order_qty")
        report["bc_loss_per_head"] = {h: meter.sums.get(f"loss_{h}", 0.0) / n for h in heads}
        report["labels_per_head"] = {h: meter.pairs.get(h, (0, 0))[1] for h in heads}
        report["ce_per_label"] = {
            h: (meter.sums.get(f"loss_{h}", 0.0) / report["labels_per_head"][h]
                if report["labels_per_head"][h] else None) for h in heads}
        if any(f"loss_set_{h}" in meter.sums for h in ("unit_qty", "order_qty")):   # side-car diagnostics, never in the loss
            report["set_nll_per_step"] = {h: meter.sums.get(f"loss_set_{h}", 0.0) / n for h in ("unit_qty", "order_qty")}
            report["set_hit"] = {h: (meter.pairs[f"set_hit_{h}"][0] / meter.pairs[f"set_hit_{h}"][1]
                                     if meter.pairs.get(f"set_hit_{h}", (0, 0))[1] else None) for h in ("unit_qty", "order_qty")}
            report["set_multi_labels"] = {h: meter.sums.get(f"set_multi_{h}", 0) for h in ("unit_qty", "order_qty")}
        report["code_mode"] = mode
        if stats is not None:
            report["codes"] = stats.report()
        return report
    finally:
        net.train(was_training)
        net.recur_T = was_T


def fit(cfg: TrainConfig, model_cfg: M.ModelConfig, out_dir, *, device="cpu", resume=False, codec=None,
        log=print, eval_batches=None, stop_after=None):
    """Train to cfg.updates (or stop after ``stop_after`` updates of this call, with a checkpoint).
    On resume the model config and all optimizer/RNG/stream-position state come from the
    checkpoint; ``cfg`` is used as given (the CLI rebuilds it from the stored train_config)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    if cfg.step_cache:
        total_steps = int(cfg.batch_windows) * int(cfg.window)
        if cfg.pack_steps <= 0 or total_steps <= 0 or total_steps % int(cfg.pack_steps):
            raise ValueError("cache batch_windows * window must be positive and divisible by pack_steps")
    def validate_cache(mc):
        if cfg.step_cache and not cfg.synthetic:
            import step_cache
            step_cache.validate_for_model(cfg.step_cache, mc, cfg)
    # data parallel over nodes (torchrun/srun env): every rank resumes the same checkpoint, draws its own data, and the
    # summed gradients are averaged once per update before clipping, so all ranks take identical optimizer steps
    world, rank = int(os.environ.get("WORLD_SIZE", "1") or 1), int(os.environ.get("RANK", "0") or 0)
    if world > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        if cfg.step_cache == "":
            raise ValueError("multi-rank training needs step_cache (the replay stream is not sharded)")
    update, consumed, epoch, resumes, metrics = 0, 0, 0, 0, {}
    if resume and (out_dir / "LATEST").exists():
        blob, path = load_checkpoint(out_dir)
        model_cfg = M.ModelConfig.from_dict(blob["config"])
        validate_cache(model_cfg)
        net = build(model_cfg, device, codec)
        net.load_state_dict(blob["model"])
        opt = make_optimizer(net, cfg)
        restore_optimizer(net, opt, cfg, blob)
        gen = torch.Generator(device=device)
        gen.set_state(blob["generator_state"])
        torch.set_rng_state(blob["torch_rng"])
        if device.type == "cuda":
            cuda_rng = blob.get("cuda_rng", [])
            if cuda_rng:
                if len(cuda_rng) != torch.cuda.device_count():
                    raise ValueError("CUDA RNG device count changed; exact RNG resume is unavailable")
                torch.cuda.set_rng_state_all(cuda_rng)
            else:
                # Legacy and CPU-built scratch checkpoints have no CUDA default-generator states.
                torch.cuda.manual_seed_all(cfg.seed + int(blob["update"]))
                log(json.dumps({"event": "cuda_rng_seeded", "exact_rng_resume": False,
                                "reason": "legacy or CPU checkpoint lacks CUDA RNG states"}))
        update, consumed, epoch = blob["update"], blob["windows_consumed"], blob["epoch"]
        resumes = blob["resumes"] + 1
        log(json.dumps({"event": "resume", "checkpoint": str(path), "update": update, "windows_consumed": consumed}))
    else:
        validate_cache(model_cfg)
        torch.manual_seed(cfg.seed)
        net = build(model_cfg, device, codec)
        opt = make_optimizer(net, cfg)
        gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    use_equiv, aux_equiv = bool(getattr(cfg, "equiv_loss", False)), bool(getattr(cfg, "aux_equiv_loss", False))
    prefix_aug = bool(getattr(cfg, "equiv_prefix_aug", False))
    use_supervision = bool(cfg.mask_order_noops or cfg.infeasible_w or cfg.order_infeasible_w)
    if cfg.commitment_weighting:
        from commitment_sidecar import read_manifest as read_commitment_manifest
        if not cfg.step_cache or not cfg.commitment_dir:
            raise ValueError("commitment weighting requires step_cache and commitment_dir")
        if world != 1 or use_supervision or use_equiv or aux_equiv or prefix_aug:
            raise ValueError("commitment weighting currently requires one rank and a separate sidecar arm")
        read_commitment_manifest(cfg.commitment_dir, cfg.commitment_manifest_sha256)
    if use_supervision:
        from supervision_sidecar import read_manifest
        if not cfg.step_cache or not cfg.supervision_dir:
            raise ValueError("loss supervision requires step_cache and supervision_dir")
        if use_equiv or aux_equiv or prefix_aug:
            raise ValueError("loss supervision and quantity equivalence need separate measured arms")
        read_manifest(cfg.supervision_dir, cfg.supervision_manifest_sha256)
    if (use_equiv or aux_equiv) and not (getattr(cfg, "equiv_dir", "") and getattr(cfg, "step_cache", "")):
        raise ValueError("equiv_loss / aux_equiv_loss need equiv_dir side-cars and a step_cache")
    if prefix_aug and not use_equiv:
        raise ValueError("equiv_prefix_aug needs equiv_loss")
    ema_state = None
    if float(getattr(cfg, "ema_decay", 0.0) or 0.0) > 0:
        sd = net.state_dict()
        ema_keys = [k for k, v in sd.items() if v.dtype.is_floating_point]
        ema_live = [sd[k] for k in ema_keys]                      # detached views: follow the in-place optimizer updates
        prev = blob.get("ema") if resume and (out_dir / "LATEST").exists() else None
        ema_vals = [(prev[k] if prev and k in prev else sd[k]).detach().float().clone().to(sd[k].device) for k in ema_keys]
        ema_state = (ema_keys, ema_live, ema_vals)
    if getattr(cfg, "feature_schema", "legacy") != net.config.feature_schema:
        raise ValueError("training feature_schema must match the checkpoint/model feature_schema")
    if (getattr(cfg, "compile_blocks", False) or getattr(cfg, "compile_model", False)) and device.type == "cuda":
        from torch import _dynamo
        _dynamo.config.recompile_limit = max(64, _dynamo.config.recompile_limit)
        if getattr(cfg, "compile_model", False):   # instance attributes: state_dict and checkpoints are unchanged
            net.encode = torch.compile(net.encode, dynamic=True)
            net.decode_teacher = torch.compile(net.decode_teacher, dynamic=True)
        else:
            for blk in net.blocks:
                blk.compile(dynamic=True)
    net.train()
    run = {"schema": "primitive-bc-run-v1", "model_config": net.config.to_dict(), "train_config": dataclasses.asdict(cfg),
           "parameters": net.parameter_count(), "device": str(device), "torch": torch.__version__,
           "numerics": "bf16 autocast (CUDA) / fp32 master weights" if (device.type == "cuda" and cfg.bf16) else "fp32",
           "tau_schedule": {"start": cfg.tau_start, "end": cfg.tau_end, "kind": cfg.tau_schedule,
                            "anneal_updates": cfg.tau_anneal_updates or cfg.updates, "straight_through_hard_forward": True},
           "resumes": resumes, "started_update": update}
    if rank == 0:
        run["world_size"] = world
        (out_dir / f"RUN-{resumes}.json").write_text(json.dumps(run, indent=1, default=str) + "\n")
    cache_it = None
    if cfg.synthetic:
        stream = SyntheticStream(net.config, cfg)
    elif getattr(cfg, "step_cache", ""):
        if int(getattr(cfg, "pack_steps", 0) or 0) <= 0 or net.config.think == "bottleneck":
            raise ValueError("step_cache needs pack_steps > 0 and a stateless model")
        import step_cache
        stream = None
        load_seed = cfg.seed * 1_000_003 + resumes * 104_729 + update + rank * 7_777_777
        if cfg.commitment_weighting:
            from commitment_sidecar import CommitmentChunks
            ds = CommitmentChunks(cfg.step_cache, cfg.commitment_dir, cfg.commitment_manifest_sha256,
                                  int(cfg.pack_steps), load_seed, pool_seats=int(cfg.cache_pool_seats),
                                  rows=os.environ.get("PBC_CACHE_ROWS") or None)
            cache_it = iter(torch.utils.data.DataLoader(
                ds, batch_size=None, num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0,
                prefetch_factor=4 if cfg.workers > 0 else None,
                multiprocessing_context="fork" if cfg.workers > 0 else None))
        elif use_supervision:
            from supervision_sidecar import SupervisionChunks
            ds = SupervisionChunks(cfg.step_cache, cfg.supervision_dir, cfg.supervision_manifest_sha256,
                                   int(cfg.pack_steps), load_seed, pool_seats=int(cfg.cache_pool_seats))
            cache_it = iter(torch.utils.data.DataLoader(
                ds, batch_size=None, num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0,
                prefetch_factor=4 if cfg.workers > 0 else None,
                multiprocessing_context="fork" if cfg.workers > 0 else None))
        elif use_equiv or aux_equiv:   # side-car masks only while a flag is on: no extra loader work otherwise
            ds = EquivChunks(cfg.step_cache, cfg.equiv_dir, int(cfg.pack_steps), load_seed,
                             pool_seats=int(getattr(cfg, "cache_pool_seats", 4)), aux=aux_equiv)
            cache_it = iter(torch.utils.data.DataLoader(
                ds, batch_size=None, num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0,
                prefetch_factor=4 if cfg.workers > 0 else None, multiprocessing_context="fork" if cfg.workers > 0 else None))
        else:
            # an update takes all its chunks at once while the GPU is idle: queue at least two updates' worth
            per_update = -(-cfg.batch_windows * cfg.window // int(cfg.pack_steps))
            ds = step_cache.StepChunks(cfg.step_cache, int(cfg.pack_steps), load_seed,
                                       pool_seats=int(getattr(cfg, "cache_pool_seats", 4)),
                                       rows=os.environ.get("PBC_CACHE_ROWS") or None)   # anneal chunks: a narrower roster
            cache_it = iter(torch.utils.data.DataLoader(
                ds, batch_size=None, num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0,
                prefetch_factor=max(4, -(-2 * per_update // cfg.workers)) if cfg.workers > 0 else None,
                multiprocessing_context="fork" if cfg.workers > 0 else None))
    else:
        skip = consumed if cfg.data_resume == "skip" else 0
        # skip is the LIFETIME consumed-window count, so replay from the original
        # epoch/seed, not from the already-prefetched checkpoint epoch.
        stream = RealStream(cfg, split=cfg.split, epoch=0 if cfg.data_resume == "skip" else epoch,
                            resumes=0 if cfg.data_resume == "skip" else resumes, skip=skip)
    if eval_batches is None and cfg.eval_every:
        eval_batches = held_out_batches(net, cfg)
    mlog = (out_dir / "metrics.jsonl").open("a") if rank == 0 else open(os.devnull, "w")
    meter, stats, t0, window_steps = Meter(), None, time.time(), 0
    first_update = update
    _wsk = [None]   # weight sketches (PBC_COS_EVERY), created lazily on the first due update
    prof_at, prof = int(os.environ.get("PBC_PROFILE_AT", "0") or 0), None   # opt-in: torch.profiler over 5 updates after prof_at
    fit_t0 = time.time()
    nonfinite = 0
    try:
        while update < cfg.updates:
            if stop_after is not None and update - first_update >= stop_after:
                break
            M.CHECK_INPUTS = update < first_update + 20   # validate inputs on each run's first updates only (host syncs)
            if prof_at and rank == 0:
                try:
                    if prof is None and update == first_update + prof_at:
                        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]
                                                      + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []))
                        prof.__enter__()
                    elif prof is not None and update == first_update + prof_at + 5:
                        prof.__exit__(None, None, None)
                        ka = prof.key_averages()
                        key = "device_time_total" if device.type == "cuda" else "cpu_time_total"
                        (out_dir / f"UPDATE-PROFILE-u{update}.txt").write_text(
                            f"5 updates {update - 5}..{update - 1}\n" + ka.table(sort_by=key, row_limit=80) + "\n"
                            + ka.table(sort_by="cpu_time_total", row_limit=40))
                        prof, prof_at = None, 0
                except Exception as e:   # profiling must never stop training
                    print(f"profile failed: {e!r}", file=sys.stderr, flush=True)
                    prof, prof_at = None, 0
            tm0 = time.time()
            if cache_it is not None:   # the same number of scored steps as batch_windows full windows
                cached = [next(cache_it) for _ in range(-(-cfg.batch_windows * cfg.window // int(cfg.pack_steps)))]
                consumed += cfg.batch_windows
                tm1 = time.time()
                batch = None
            else:
                if cfg.synthetic:
                    windows = stream.batch(consumed, cfg.batch_windows)
                else:
                    windows = stream.batch(cfg.batch_windows)
                consumed += len(windows)
                tm1 = time.time()
                batch = collate(windows, net)
            tm2 = time.time()
            tau = tau_at(update, cfg)
            for g in opt.param_groups:
                g["lr"] = group_lr(update, cfg, g)
            if int(getattr(net.config, "recur_train_max", 0) or 0) > 0:
                net.recur_T = recur_loops_at(update, net.config, cfg)
                # Retain legacy sum for old readers; expose an unambiguous count/mean pair too.
                meter.add({"recur_T": net.recur_T, "recur_loops_mean": (net.recur_T, 1)})
            if stats is None and net.S:
                stats = CodeStats(net.S, net.config.C, net.config.K)
            kw = dict(device=device, tau=tau, generator=gen, bf16=cfg.bf16, grad_checkpoint=cfg.grad_checkpoint,
                      qty_weight=cfg.qty_weight, usage_reg=cfg.usage_reg, code_stats=stats,
                      mask_unit_noops=cfg.mask_unit_noops, mask_order_noops=cfg.mask_order_noops,
                      infeasible_w=cfg.infeasible_w, order_infeasible_w=cfg.order_infeasible_w,
                      aux_w=aux_weights(cfg), use_sets=use_equiv, prefix_aug=prefix_aug,
                      commitment_weighting=cfg.commitment_weighting)
            opt.zero_grad(set_to_none=True)
            if int(getattr(cfg, "pack_steps", 0) or 0) > 0 and net.config.think != "bottleneck":
                if cache_it is not None:
                    n_all = sum(int(c["step_valid"].shape[0]) for c in cached)
                    if cfg.commitment_weighting:
                        kw["order_weight_scale"] = order_weight_scale(cached)
                    chunks = (step_cache.to_device(c, device) for c in cached)
                else:
                    chunks = pack_batch(batch, int(cfg.pack_steps))
                    n_all = int((batch["step_valid"] & batch["loss_mask"]).sum())
                m, loss_sum = Meter(), 0.0
                for cb in chunks:
                    with (torch.profiler.record_function("pbc_forward") if prof is not None else contextlib.nullcontext()):
                        lc, mc, _, _ = run_window_batch(net, cb, **kw)
                    nc = mc.sums.get("scored_example_steps", 0)
                    with (torch.profiler.record_function("pbc_backward") if prof is not None else contextlib.nullcontext()):
                        (lc * (nc / max(1, n_all))).backward()      # mean over all scored steps, accumulated chunk by chunk
                    loss_sum = loss_sum + lc.detach().float() * nc
                    m.add(mc.pairs)
                    m.add(mc.sums)
                loss = loss_sum / max(1, n_all)
            else:
                loss, m, _, _ = run_window_batch(net, batch, **kw)
                loss.backward()
            if world > 1:
                _allreduce_grads(net, world)
            # growth: the base network is clipped exactly as before; grow_* parameters are clipped on their own
            fresh_names = tuple(x for x in str(getattr(cfg, "new_prefixes", "") or "").split(",") if x)
            is_new = (lambda n: bool(fresh_names) and n.startswith(fresh_names))
            base = [p for n, p in net.named_parameters() if not n.startswith("grow_") and not is_new(n)]
            grown = [p for n, p in net.named_parameters() if n.startswith("grow_") and not is_new(n)]
            fresh = [p for n, p in net.named_parameters() if is_new(n)]
            _us = bool(UPDATE_STATS) and rank == 0 and update % UPDATE_STATS == 0
            if _us:
                with torch.no_grad():
                    _named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
                    _gpre = {n: float(p.grad.detach().float().norm()) for n, p in _named if p.grad is not None}
                    _before = {n: p.detach().float().clone() for n, p in _named}
            gnorm = torch.nn.utils.clip_grad_norm_(base, _grad_clip(cfg))
            gg = torch.nn.utils.clip_grad_norm_(grown, _grad_clip(cfg)) if grown else None
            gf = torch.nn.utils.clip_grad_norm_(fresh, _grad_clip(cfg)) if fresh else None
            if not bool(torch.isfinite(torch.stack([g.float() for g in (gnorm, gg, gf) if g is not None])).all()):
                nonfinite += 1   # skip this update (no step, no EMA); three in a row is a real divergence
                meter.add({"nonfinite_skipped": 1})
                print(json.dumps({"event": "nonfinite_grad_skipped", "update": update, "consecutive": nonfinite}), flush=True)
                if nonfinite >= 3:
                    raise RuntimeError(f"3 consecutive nonfinite gradients at update {update}")
                opt.zero_grad(set_to_none=True)
                continue
            nonfinite = 0
            if grown:
                meter.add({"grow_grad_norm": gg.item(),
                           "grow_weight_norm": float(torch.sqrt(sum((p.detach().float() ** 2).sum() for p in grown)))})
            if fresh:
                meter.add({"new_grad_norm": gf.item()})
            tm3 = time.time()
            opt.step()
            if COS_EVERY and rank == 0 and update % COS_EVERY == 0:
                try:
                    if _wsk[0] is None:
                        _wsk[0] = _WeightSketch(net, out_dir, device)
                    _wsk[0].step(net, update)
                except Exception as exc:   # sketches must never stop training
                    print(json.dumps({"event": "weight_sketch_failed", "error": repr(exc)}), flush=True)
            if _us:
                try:
                    _write_update_stats(out_dir, update, _named, _before, _gpre)
                except Exception as exc:   # stats must never stop training
                    print(json.dumps({"event": "update_stats_failed", "error": repr(exc)}), flush=True)
                del _before, _gpre, _named
            if ema_state is not None:
                with torch.no_grad():
                    d = min(float(cfg.ema_decay), (1.0 + update) / (10.0 + update))   # early EMA must not keep the random init
                    torch._foreach_lerp_(ema_state[2], [v.float() for v in ema_state[1]], 1.0 - d)
            update += 1
            meter.add({k: v for k, v in m.pairs.items()})
            meter.add(aux_metric_sums(m))
            # loss components (PRO-PROJECT-0927 #5): pooled per scored step at log time; aux weighted as in the objective
            comp = {"sum_scored": m.sums.get("scored_example_steps", 0),
                    "sum_bc": m.sums.get("loss_unit_tok", 0.0)
                    + m.sums.get("loss_weighted_order_tok" if cfg.commitment_weighting else "loss_order_tok", 0.0)
                    + cfg.qty_weight * sum(m.sums.get(k, 0.0) for k in ("loss_unit_qty", "loss_order_qty")),
                    "sum_aux": m.sums.get("loss_aux_weighted", 0.0)}
            if use_supervision:
                comp["sum_supervision"] = m.sums.get("loss_supervision", 0.0)
                if cfg.mask_order_noops:
                    comp["sum_bc_reference"] = (
                        sum(m.sums.get(k, 0.0) for k in ("loss_unit_tok", "loss_ref_order_tok"))
                        + cfg.qty_weight * sum(m.sums.get(k, 0.0) for k in
                                               ("loss_unit_qty", "loss_ref_order_qty")))
                meter.add({k: v for k, v in m.sums.items() if k.startswith(
                    ("infeasible_", "masked_order_", "loss_ref_order", "loss_supervision_"))})
                meter.add({k: v for k, v in m.pairs.items() if k.startswith(
                    ("infeasible_", "scored_order_"))})
            meter.add(comp)
            meter.add({"loss": loss.item(), "updates_in_log": 1, "grad_norm": gnorm.item()})
            tm4 = time.time()   # phase seconds summed over the log interval (compute = CPU time to queue fwd/bwd; opt includes the GPU wait)
            meter.add({"t_data": tm1 - tm0, "t_collate": tm2 - tm1, "t_compute": tm3 - tm2, "t_opt_sync": tm4 - tm3})
            window_steps += m.sums.get("scored_example_steps", 0)
            if update % cfg.log_every == 0 or update == cfg.updates:
                r = meter.report()
                k = r.pop("updates_in_log")
                r["loss"], r["grad_norm"] = r["loss"] / k, r["grad_norm"] / k
                ns = max(1, r.pop("sum_scored", 0))
                sbc, saux = r.pop("sum_bc", 0.0), r.pop("sum_aux", 0.0)
                r.update(bc_loss_per_step=sbc / ns, aux_loss_per_step=saux / ns, aux_to_bc_ratio=saux / max(sbc, 1e-9))
                if use_supervision:
                    ss = r.pop("sum_supervision", 0.0)
                    r.update(supervision_loss_per_step=ss / ns,
                             supervision_to_bc_ratio=ss / max(sbc, 1e-9))
                    if cfg.mask_order_noops:
                        r["bc_reference_loss_per_step"] = r.pop("sum_bc_reference", 0.0) / ns
                for h in ("unit_qty", "order_qty"):   # set NLL is reported beside the selected CE objective
                    for k in (f"loss_set_{h}", f"set_multi_{h}", f"set_size_{h}"):
                        if k in r:
                            r[k + "_per_step"] = r.pop(k) / ns
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
            timed_out = bool(getattr(cfg, "max_seconds", 0.0)) and time.time() - fit_t0 >= cfg.max_seconds
            last = update == cfg.updates or (stop_after is not None and update - first_update >= stop_after) or timed_out
            if update % cfg.ckpt_every == 0 or last:
                if world > 1:
                    _check_replicas(net, update, log)
                if rank == 0:
                    save_checkpoint(out_dir, net, opt, cfg, update=update, windows_consumed=consumed * world,
                                    epoch=getattr(stream, "epoch", 0), resumes=resumes, generator=gen, metrics=metrics,
                                    ema=dict(zip(ema_state[0], ema_state[2])) if ema_state is not None else None)
            if timed_out:
                log(json.dumps({"event": "max_seconds", "update": update, "seconds": round(time.time() - fit_t0)}))
                break
    finally:
        mlog.close()
    return net, metrics


def _allreduce_grads(net, world):
    """Average gradients over ranks in one flat fp32 all-reduce; a parameter no rank produced a gradient for stays None
    (as on one GPU, AdamW then skips it)."""
    import torch.distributed as dist
    params = [p for p in net.parameters() if p.requires_grad]
    dev = params[0].device
    has = torch.tensor([p.grad is not None for p in params], dtype=torch.float32, device=dev)
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in params])
    dist.all_reduce(has)
    dist.all_reduce(flat)
    flat.div_(world)
    o = 0
    for p, h in zip(params, has.tolist()):
        n = p.numel()
        if h > 0:
            g = flat[o:o + n].view_as(p).to(p.dtype)
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)
        o += n


def _check_replicas(net, update, log):
    """Ranks must hold identical weights; compare a float64 checksum and stop loudly if they drifted."""
    import torch.distributed as dist
    c = torch.stack([sum(p.detach().double().sum() for p in net.parameters()),
                     sum((p.detach().double() ** 2).sum() for p in net.parameters())])
    lo, hi = c.clone(), c.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if not torch.equal(lo, hi):
        raise RuntimeError(f"replica weights differ at update {update}: {lo.tolist()} vs {hi.tolist()}")
    log(json.dumps({"event": "replicas_equal", "update": update}))


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
    # v5s (2026-09-27): from-scratch runs can switch on the growth-era inputs directly and pick the block MLP
    feats = {f for f in (getattr(a, "features", "") or "").split(",") if f}
    extra = {}
    if feats:
        import data
        import flip
        import prefix
        widths = {"extra": ("extra_width", flip.LIN), "unit_extra": ("unit_extra_width", flip.UNIT_WIDTH),
                  "calc": ("calc_width", data.CALC_WIDTH), "prefix": None, "reach_bias": ("reach_bias", True),
                  "carrier_read": ("carrier_read", True), "reach_job": ("reach_job", True),
                  "v6prev": ("v6_prev", True), "v6aux": ("v6_aux", True)}
        unknown = feats - set(widths)
        if unknown:
            raise SystemExit(f"unknown --features {sorted(unknown)}")
        for f in feats:
            if f == "prefix":
                extra.update(prefix_unit_width=prefix.F_UP, prefix_order_width=prefix.F_OP)
            else:
                extra[widths[f][0]] = widths[f][1]
    return M.ModelConfig(**shape, think=a.think, ffn=getattr(a, "ffn", "gelu") or "gelu",
                         feature_schema=getattr(a, "feature_schema", None) or "legacy", **extra)


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
    f.add_argument("--ffn", default="gelu", choices=("gelu", "swiglu"))
    f.add_argument("--features", default="", help="comma list: extra,unit_extra,calc,prefix,reach_bias,carrier_read,reach_job")
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
    ae = sub.add_parser("auxeval", help="deterministic VAL report of the training-only aux heads on a held-out step cache")
    ae.add_argument("--checkpoint", required=True)
    ae.add_argument("--step-cache", required=True, help="step_cache.py directory of held-out seats built with --v6")
    ae.add_argument("--equiv-dir", default="", help="equiv_build.py side-cars: adds set NLL / set-hit diagnostics")
    ae.add_argument("--chunk", type=int, default=256)
    ae.add_argument("--max-steps", type=int, default=0)
    ae.add_argument("--grad-probe", action="store_true", help="BC vs aux shared-gradient norm ratio / cosine on the 1st chunk")
    ae.add_argument("--device", default="cpu")
    ae.add_argument("--out")
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
    elif a.cmd == "auxeval":
        blob, path = load_checkpoint(a.checkpoint)
        net = build(M.ModelConfig.from_dict(blob["config"]), a.device)
        net.load_state_dict(blob["model"])
        cfg = TrainConfig.from_dict(blob["train_config"])
        dev = torch.device(a.device)
        batches = lambda n=a.max_steps: cache_batches(a.step_cache, a.chunk, dev, n, a.equiv_dir, bool(a.equiv_dir))
        rep = evaluate_aux(net, batches(), cfg, dev, detail=True)
        rep["summary"] = aux_summary(rep, cfg, rep["scored_example_steps"])
        if a.grad_probe:
            rep["grad_probe"] = grad_probe(net, next(batches(a.chunk)), cfg, dev)
        rep.update(checkpoint=str(path), update=blob["update"], step_cache=a.step_cache)
        text = json.dumps(rep, indent=1, default=float)
        if a.out:
            Path(a.out).write_text(text + "\n")
        print(json.dumps({k: rep[k] for k in ("summary", "grad_probe") if k in rep}, default=float))
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
