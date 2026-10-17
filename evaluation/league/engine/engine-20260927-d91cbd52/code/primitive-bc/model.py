"""Primitive-BC policy v3 (ARCH-PBC-v3): pack + feature tokens -> latent trunk -> thinking field
-> sequential per-step primitive action (farmer -> hands -> market orders -> STOP).

One step (``PolicyNet.forward(batch, state)``, teacher forcing; ``act`` decodes greedily):

1. Input tokens (width tok_width): each observation.pack row (Linear(values masked by
   knownness) + Linear(knownness)); the 22 feature_tokens tokens (same form); the theorem
   tile rows as a residual on the 200 TILE tokens (Linear, located by each row's own
   role/x/y columns); the actor's previous own unit tokens (prev_unit_tok, one token per unit
   slot with a previous action); an optional teacher token (team embedding, dropped to
   'unknown' = 0 with p=team_dropout while training, + team-day strength decile). Each group
   adds its token-type embedding. Value columns (pack [35,80), feature tokens [20,128), theorem
   tile rows) pass through sign(v) log1p(c|v|)/log1p(c), c = value_log (identity columns stay
   linear); inp/known_inp/feat_inp/feat_known/tile_inp (and carrier_mlp) run in fp32 outside
   autocast (input_fp32). pos_embed: learned role + x + y embeddings on every TILE and CARRIER
   token, + an ordinal embedding on CARRIER tokens.
2. Trunk: L learned latents + Linear(carried state), RMS-normalised (the trunk input), then
   one cross-attention block (latents <- tokens, + SwiGLU) and trunk_layers pre-norm
   self-attention + dense SwiGLU blocks on the latents, final RMSNorm.
3. Thinking field (think='field'): each of the 200 cells = tanh(Linear([its current tile
   token embedding incl. theorem residual, cross-attention read of the latents])) width
   field_dc; global slots = tanh(Linear(latents) + carried FIFO) (n_slots x field_dc). Cells
   are recomputed each step. 'fsq': the same with FSQ (fsq_dims x fsq_levels, straight-through
   rounding) instead of tanh, and the FSQ slots are the carried state. 'carry': no bottleneck
   (cells width dec_width, slots = Linear(latents) + Linear(FIFO)). 'off': as 'carry' without
   any carried state. field/carry carried state = a causal FIFO of n_slots entries, one per
   step, written by ``commit`` after the step's action is fixed: Linear(attention pool of the
   final trunk latents) + Linear(summary of the step's own submitted action, units + market);
   it feeds the next step's latent init (Linear), the slots, and every decoder query
   (history_query). carry_scope='episode' resets the carried state at step 0, 'day' at every
   hour 0.
4. Decoders (width dec_width, dec_layers pre-norm layers: causal self-attention over the
   decisions of this step with a K/V cache, cross-attention into the field memory
   = Linear(field) + farm/row/col or slot position embeddings, SwiGLU). Decision queries:
   decision-slot identity + the previous decision's embedding (token + quantity class +
   digits) + for units: x/y embeddings, the unit's previous own token, its own + 4
   neighbour field cells and (unit_carrier) carrier_mlp of its own CARRIER row's value columns
   (batch['carrier_row'] from unit_index.carrier_rows in ``collate``; 0 without a row). Stock path
   (stock_path, 2026-09-23 stock-blind diagnosis fix 1): every decision query (units and orders,
   teacher-forced and free-running, where the history readout is added) also gets
   stock_mlp(econ), econ = signed_log of the actor's own STOCK quantities in a fixed order
   (shed[item] for the 12 observation.ITEMS, then seeds[item] on the same item index; only the 5
   crops occur) and own FARM cash, gathered from the pack rows by kind/role/item/seed columns
   (``econ_values``); fp32 outside autocast, last layer zero-initialised (an untrained model's
   outputs are unchanged). Before it, stock and cash reached the decoders only through the
   32-latent trunk cross-attention. Order: farmer, hands in observation order, orders (the last of
   n_order_slots slots is STOP-only; NOOP is its own token), each quantity head (QTY_CLASSES
   = 75 bins + QTY_EXACT) after its token, the digit head (DIGIT_VOCAB, END-terminated)
   after QTY_EXACT. No executor.

Initialisation (``init_module``, applied to every module at construction): Linear weights
truncated normal with realised std 0.02 (the underlying normal has std 0.02 / TRUNC_STD =
0.02 / 0.8796 and is cut at +-2 of its own std), biases 0; Embedding N(0, 0.02); norms weight 1 (bias 0);
residual-branch output projections (attention out-proj and SwiGLU down-proj of the trunk
and decoder stacks) std 0.02/sqrt(2 x blocks in that stack); latents / bos N(0, 0.02);
output heads (unit, order, quantity, digit) zero, so initial logits are uniform; the learned
MISSING vectors (feature dropout) N(0, 0.02).
``param_groups`` puts weight decay on nn.Linear weights only.

Feature dropout (training only; train.py draws the masks): ``batch['feat_drop']`` =
{"cols": [B, K, F] bool, "tile": [B] bool}. In train mode, the dropped feature-token columns
take the learned ``feat_missing`` [K, F] values with knownness 0, and a dropped theorem tile
block takes the learned ``tile_missing`` [TW] on every tile. Ignored in eval mode (``act``,
evaluation, the agent), so it never reaches inference. The raw pack rows are never dropped.

Copy (teacher-action channel) dropout (training only; train.py copy_dropout draws one mask per
perspective chunk): ``batch['copy_drop']`` [B] bool. In train mode a True row loses every input
that carries the teacher's own earlier actions: prev_unit_tok -> -1 (the previous-unit tokens
are masked out and the decoder's prev_own takes its 'no previous token' entry), the teacher
token -> team 0 / decile unknown, and ``commit`` writes no action term into the FIFO (the
trunk-latent term only). Ignored in eval mode (``act``, evaluation, the agent).

Labels (codec.encode_step, collated by ``collate``): unit_tok/unit_qty [B,U], unit_digits
[B,U,D], order_tok/order_qty [B,O], order_digits [B,O,D]; -1 = absent. D = MAX_DIGITS + 1.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SCHEMA = "primitive-bc-policy-v3"
IN_WIDTH = 80            # observation.WIDTH
TILE_KIND_COL = 2        # observation.KINDS.index("TILE")
ROLE_COL = 12            # 12 = own farm, 13 = opponent farm
XY_COL = 14              # x/9, y/9
CARRIER_KIND_COL = 3     # observation.KINDS.index("CARRIER")
ORD_COL = 16             # ordinal/100
PACK_PAYLOAD = 35        # observation.PAYLOAD: pack value columns [35, 80); identity columns before
FEAT_PAYLOAD = 20        # feature_tokens.PAYLOAD: feature-token value columns [20, WIDTH)
FARM_KIND_COL = 1        # observation.KINDS.index("FARM"); value column PACK_PAYLOAD = cash / 10000
STOCK_KIND_COL = 5       # observation.KINDS.index("STOCK"); FIELDS: quantity / 100, shed, seed
ITEM_COL = 17            # item one-hot [17, 29) over observation.ITEMS (9 products + 3 animals)
N_ITEMS = 12
STOCK_SEED_COL = PACK_PAYLOAD + 2   # STOCK 'seed' flag (1 = seed storage, 0 = shed)
ECON_WIDTH = 2 * N_ITEMS + 1        # shed[item], seeds[item], own cash
BOARD = 10
N_CELLS = 2 * BOARD * BOARD
INIT_STD = 0.02
TRUNC_A = 2.0            # Linear init: normal truncated at +-TRUNC_A of its own std
# std of a unit normal truncated to [-a, a]: sqrt(1 - 2 a phi(a) / (2 Phi(a) - 1)) = 0.87963 at a = 2
TRUNC_STD = math.sqrt(1 - 2 * TRUNC_A * math.exp(-TRUNC_A ** 2 / 2) / math.sqrt(2 * math.pi)
                      / math.erf(TRUNC_A / math.sqrt(2)))
THINK_MODES = ("field", "fsq", "carry", "off")
CARRY_SCOPES = ("episode", "day")
TOKEN_TYPES = ("pack", "feature", "prev_unit", "teacher")
T_PACK, T_FEAT, T_PREV, T_TEACHER = range(len(TOKEN_TYPES))


def _now():
    return time.perf_counter()


def _codec():
    import codec  # primitive-bc/codec.py (lazy: tests may inject a stub module)
    return codec


# ----------------------------------------------------------------------------- config
@dataclass
class ModelConfig:
    # vocabulary / interface (None -> filled from codec.py by resolve())
    n_unit_tokens: Optional[int] = None
    n_order_tokens: Optional[int] = None
    max_units: Optional[int] = None
    n_order_slots: Optional[int] = None     # MAX_ORDERS + 1 (last slot STOP only)
    n_qty: Optional[int] = None             # quantity classes incl. QTY_EXACT
    qty_exact: Optional[int] = None
    n_digit: Optional[int] = None           # len(DIGIT_VOCAB) incl. END
    digit_slots: Optional[int] = None       # MAX_DIGITS + 1
    digit_end: Optional[int] = None
    stop_index: Optional[int] = None
    noop_index: Optional[int] = None
    pass_index: Optional[int] = None
    unit_has_qty: Optional[list] = None
    order_has_qty: Optional[list] = None
    # input tokens
    tok_width: int = 256
    features: bool = True                   # feature_tokens input (22 tokens + tile residual)
    feat_missing: bool = True               # learned MISSING vectors for training feature dropout
                                            # (absent from configs saved before it: False)
    feat_tokens: int = 22                   # feature_tokens.K
    feat_width: int = 128                   # feature_tokens.WIDTH
    tile_width: int = 31                    # feature_tokens.TILE_WIDTH
    prev_unit_tokens: bool = True
    teacher_token: bool = True
    n_teams: int = 1024                     # team vocabulary size, 0 = unknown
    n_strength: int = 10                    # strength deciles (index n_strength = unknown)
    team_dropout: float = 0.1
    # input numerics / identity (gap review 2026-09-23 #1/#4; configs saved before them: off)
    value_log: float = 100.0                # value columns -> sign(v) log1p(c|v|) / log1p(c), c = value_log
                                            # (pack [35,80), feature tokens [20,128), theorem tile rows,
                                            # carrier MLP input); 0 = linear (legacy)
    input_fp32: bool = True                 # input projections + carrier MLP in fp32 outside autocast
    pos_embed: bool = True                  # learned role/x/y (+ ordinal for CARRIER) on TILE and CARRIER tokens
    unit_carrier: bool = True               # unit query + MLP(own CARRIER row values) via batch['carrier_row']
    stock_path: bool = True                 # every decision query + stock_mlp(own STOCK quantities + own cash)
                                            # (configs saved before it: off)
    # trunk
    trunk_latents: int = 32
    trunk_width: int = 512
    trunk_layers: int = 6
    trunk_hidden: int = 1536
    trunk_heads: int = 8
    # thinking field
    think: str = "field"
    field_dc: int = 48
    n_slots: Optional[int] = None           # None -> trunk_latents
    fsq_dims: int = 8
    fsq_levels: int = 8
    carry_scope: str = "episode"
    # decoder
    dec_width: int = 256
    dec_layers: int = 2
    dec_heads: int = 4
    dec_hidden: int = 768
    local: bool = True                      # unit query reads its own + 4 neighbour field cells
    schema: str = SCHEMA

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        kw.setdefault("feat_missing", False)          # configs saved before the MISSING vectors
        for k, v in LEGACY_INPUTS.items():            # configs saved before the input fixes
            kw.setdefault(k, v)
        kw.setdefault("stock_path", False)            # configs saved before the stock path (stock-blind v3)
        return cls(**kw)

    def resolve(self, codec=None):
        """Fill the vocabulary fields from codec.py; returns a new, validated config."""
        c = dataclasses.replace(self)
        vocab = ("n_unit_tokens", "n_order_tokens", "max_units", "n_order_slots", "n_qty", "qty_exact",
                 "n_digit", "digit_slots", "digit_end", "stop_index", "noop_index", "pass_index",
                 "unit_has_qty", "order_has_qty")
        if any(getattr(c, k) is None for k in vocab):
            m = codec or _codec()
            bins = int(getattr(m, "QTY_BIN_COUNT", 75))
            digits = tuple(getattr(m, "DIGIT_VOCAB", tuple("0123456789") + ("END",)))
            fill = {"n_unit_tokens": len(m.UNIT_VOCAB), "n_order_tokens": len(m.ORDER_VOCAB),
                    "max_units": int(m.MAX_UNITS),
                    "n_order_slots": int(getattr(m, "N_ORDER_SLOTS", int(m.MAX_ORDERS) + 1)),
                    "n_qty": int(getattr(m, "QTY_CLASSES", bins + 1)),
                    "qty_exact": int(getattr(m, "QTY_EXACT", bins)),
                    "n_digit": len(digits),
                    "digit_slots": int(getattr(m, "DIGIT_SLOTS", int(getattr(m, "MAX_DIGITS", 5)) + 1)),
                    "digit_end": int(getattr(m, "DIGIT_END", digits.index("END"))),
                    "stop_index": list(m.ORDER_VOCAB).index("STOP"),
                    "noop_index": list(m.ORDER_VOCAB).index("NOOP") if "NOOP" in m.ORDER_VOCAB else -1,
                    "pass_index": list(m.UNIT_VOCAB).index("PASS"),
                    "unit_has_qty": [bool(v) for v in m.UNIT_HAS_QTY],
                    "order_has_qty": [bool(v) for v in m.ORDER_HAS_QTY]}
            for k in vocab:
                if getattr(c, k) is None:
                    setattr(c, k, fill[k])
        c.unit_has_qty = [bool(v) for v in c.unit_has_qty]
        c.order_has_qty = [bool(v) for v in c.order_has_qty]
        if c.n_slots is None:
            c.n_slots = c.trunk_latents
        if len(c.unit_has_qty) != c.n_unit_tokens or len(c.order_has_qty) != c.n_order_tokens:
            raise ValueError("quantity masks must match the vocabularies")
        if c.think not in THINK_MODES:
            raise ValueError("think must be one of " + ", ".join(THINK_MODES))
        if c.carry_scope not in CARRY_SCOPES:
            raise ValueError("carry_scope must be one of " + ", ".join(CARRY_SCOPES))
        for w, h in ((c.trunk_width, c.trunk_heads), (c.dec_width, c.dec_heads)):
            if w % h:
                raise ValueError("width must be divisible by heads")
        if not 0 <= c.qty_exact < c.n_qty or not 0 <= c.digit_end < c.n_digit:
            raise ValueError("QTY_EXACT / END out of range")
        if c.schema != SCHEMA:
            raise ValueError(f"config schema {c.schema!r} is not {SCHEMA} (v1 checkpoints: primitive-bc-v1-code)")
        return c


LEGACY_INPUTS = {"value_log": 0.0, "input_fp32": False, "pos_embed": False, "unit_carrier": False}


def default_config():
    return ModelConfig()


def signed_log(v, c):
    """sign(v) log1p(c |v|) / log1p(c) (c > 0; identity for c = 0): 1 unit of a /100 column
    -> 0.150, 1.0 -> 1.0, 12 (cash 120k) -> 1.537, 120 -> 2.035."""
    if not c:
        return v
    return torch.sign(v) * torch.log1p(c * v.abs()) / math.log1p(c)


def fp32_inputs(device, enabled=True):
    """Context for the input projections: autocast off (fp32) when enabled."""
    import contextlib
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=torch.device(device).type, enabled=False)


SIZES = {  # name -> trunk/decoder shape overrides (vocabulary from codec)
    "tiny": dict(tok_width=32, trunk_latents=8, trunk_width=64, trunk_layers=2, trunk_hidden=128,
                 trunk_heads=2, field_dc=16, dec_width=32, dec_layers=2, dec_heads=2, dec_hidden=64, n_teams=16),
    "l32": dict(trunk_latents=32),
    "l64": dict(trunk_latents=64),
}


# ----------------------------------------------------------------------------- layers
class SwiGLU(nn.Module):
    """Dense SwiGLU feed-forward: w2(silu(x W1) * (x W3)); W1/W3 fused in one Linear."""

    def __init__(self, d, hidden, depth_key):
        super().__init__()
        self.w13 = nn.Linear(d, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
        self.w2.residual = depth_key

    def forward(self, x):
        a, b = self.w13(x).chunk(2, -1)
        return self.w2(F.silu(a) * b)


class SelfAttention(nn.Module):
    def __init__(self, d, heads, depth_key):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.out.residual = depth_key

    def forward(self, x, mask=None, cache=None):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat((cache["k"], k), 2), torch.cat((cache["v"], v), 2)
            cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(B, T, d))


class CrossAttention(nn.Module):
    """Queries width dq attend to a memory of width dm through `inner` channels; the output
    projection (inner -> dq) is omitted when out=False (the caller projects)."""

    def __init__(self, dq, dm, heads, inner=None, out=True, depth_key=None):
        super().__init__()
        inner = inner or dq
        self.h, self.dh, self.inner = heads, inner // heads, inner
        self.q = nn.Linear(dq, inner, bias=False)
        self.kv = nn.Linear(dm, 2 * inner, bias=False)
        self.out = nn.Linear(inner, dq, bias=False) if out else None
        if out:
            self.out.residual = depth_key

    def memory(self, mem):
        B, S, _ = mem.shape
        k, v = self.kv(mem).view(B, S, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x, kv, mask=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=mask).transpose(1, 2).reshape(B, T, self.inner)
        return y if self.out is None else self.out(y)


class CrossBlock(nn.Module):
    """Latents <- tokens (pre-norm cross-attention), then a SwiGLU block."""

    def __init__(self, W, tw, heads, hidden):
        super().__init__()
        self.n_q, self.attn = nn.RMSNorm(W), CrossAttention(W, tw, heads, depth_key="trunk")
        self.n_ff, self.ff = nn.RMSNorm(W), SwiGLU(W, hidden, "trunk")

    def forward(self, x, tokens, mask):
        x = x + self.attn(self.n_q(x), self.attn.memory(tokens), mask)
        return x + self.ff(self.n_ff(x))


class LatentBlock(nn.Module):
    def __init__(self, W, heads, hidden):
        super().__init__()
        self.n1, self.attn = nn.RMSNorm(W), SelfAttention(W, heads, "trunk")
        self.n2, self.ff = nn.RMSNorm(W), SwiGLU(W, hidden, "trunk")

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ff(self.n2(x))


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, hidden):
        super().__init__()
        self.n1, self.attn = nn.RMSNorm(d), SelfAttention(d, heads, "dec")
        self.n2, self.xattn = nn.RMSNorm(d), CrossAttention(d, d, heads, depth_key="dec")
        self.n3, self.ff = nn.RMSNorm(d), SwiGLU(d, hidden, "dec")

    def memory(self, mem):
        return self.xattn.memory(mem)

    def forward(self, x, self_mask, mem_kv, cache=None):
        x = x + self.attn(self.n1(x), self_mask, cache)
        x = x + self.xattn(self.n2(x), mem_kv)
        return x + self.ff(self.n3(x))


class QtyHead(nn.Module):
    """Quantity class after a token: h + token conditioning -> SwiGLU residual -> logits."""

    def __init__(self, d, hidden, n_tok, n_qty):
        super().__init__()
        self.cond = nn.Embedding(n_tok, d)
        self.norm, self.ff, self.out_norm = nn.RMSNorm(d), SwiGLU(d, hidden, "dec"), nn.RMSNorm(d)
        self.head = nn.Linear(d, n_qty)
        self.head.zero_init = True

    def forward(self, h, tok):
        x = h + self.cond(tok)
        x = x + self.ff(self.norm(x))
        return self.head(self.out_norm(x)), x


class DigitHead(nn.Module):
    """Canonical decimal digits of a QTY_EXACT count, autoregressive over digit_slots positions:
    input_j = h_q + pos_j + sum_{i<j} emb(i, d_i) (position-specific digit embeddings)."""

    def __init__(self, d, hidden, slots, n_digit, end):
        super().__init__()
        self.slots, self.n, self.end = slots, n_digit, end
        self.pos = nn.Embedding(slots, d)
        self.emb = nn.Embedding(slots * n_digit, d)
        self.norm, self.ff, self.out_norm = nn.RMSNorm(d), SwiGLU(d, hidden, "dec"), nn.RMSNorm(d)
        self.head = nn.Linear(d, n_digit)
        self.head.zero_init = True
        self.register_buffer("offset", torch.arange(slots) * n_digit, persistent=False)

    def embed(self, digits):
        """[..., D] digit labels (-1 = none) -> [..., D, d] (zero where -1)."""
        e = self.emb(self.offset + digits.clamp(min=0))
        return e * (digits >= 0).unsqueeze(-1).to(e.dtype)

    def summary(self, digits):
        return self.embed(digits).sum(-2)

    def logits(self, x):
        x = x + self.ff(self.norm(x))
        return self.head(self.out_norm(x))

    def teacher(self, hq, digits):
        e = self.embed(digits)
        prefix = e.cumsum(-2) - e
        return self.logits(hq.unsqueeze(-2) + self.pos.weight + prefix)

    def decode(self, hq, need, choose, force_full=False):
        """-> (digits [B, D] long, summary [B, d]); rows with need False stay -1 / 0."""
        B = hq.shape[0]
        digits = torch.full((B, self.slots), -1, dtype=torch.long, device=hq.device)
        prefix = torch.zeros_like(hq)
        done = ~need
        for j in range(self.slots):
            dj = choose(self.logits(hq + self.pos.weight[j] + prefix))
            if force_full:      # latency worst case: every digit slot is decoded
                dj = torch.full_like(dj, self.end if j == self.slots - 1 else 1)
            digits[:, j] = torch.where(done, -1, dj)
            prefix = prefix + self.emb.weight[j * self.n + dj] * (~done).unsqueeze(-1).to(prefix.dtype)
            done = done | (dj == self.end)
            if bool(done.all()):
                break
        return digits, prefix


def fsq(z, levels):
    """Finite scalar quantisation: tanh(z) rounded to `levels` evenly spaced values in [-1, 1]
    with a straight-through gradient. -> (values, level indices long)."""
    t = torch.tanh(z.float())
    idx = torch.round((t + 1) * ((levels - 1) / 2))
    q = idx * (2 / (levels - 1)) - 1
    return (t + (q - t).detach()).to(z.dtype), idx.long()


def fsq_entropy_bits(codes, levels):
    """Mean over field positions and FSQ dims of the entropy (bits) of the level distribution,
    pooled over every leading axis. codes [..., P, dims] long."""
    flat = codes.reshape(-1, codes.shape[-2] * codes.shape[-1])
    counts = F.one_hot(flat, levels).sum(0).float()          # [P*dims, levels]
    p = counts / counts.sum(-1, keepdim=True)
    h = -(p * torch.log2(p.clamp_min(1e-12))).sum(-1)
    return float(h.mean())


def constant_fraction(fields, tol=0.01):
    """Fraction of field positions whose every channel has std < tol across the leading axes
    (e.g. a stack of out['field'] over 256 held-out steps). fields [..., P, C]."""
    flat = fields.reshape(-1, fields.shape[-2], fields.shape[-1]).float()
    return float((flat.std(0).amax(-1) < tol).float().mean())


def _neighbours():
    """[100, 5] own-farm cell ids of (self, N, S, E, W); 100 = off-board pad."""
    table = torch.full((BOARD * BOARD, 5), BOARD * BOARD, dtype=torch.long)
    for y in range(BOARD):
        for x in range(BOARD):
            for j, (dx, dy) in enumerate(((0, 0), (0, -1), (0, 1), (1, 0), (-1, 0))):
                nx, ny = x + dx, y + dy
                if 0 <= nx < BOARD and 0 <= ny < BOARD:
                    table[y * BOARD + x, j] = ny * BOARD + nx
    return table


def trunc_normal_(weight, std):
    """Truncated normal with realised std ``std``: the underlying normal has std / TRUNC_STD
    and is cut at +-TRUNC_A of its own std (so |w| <= TRUNC_A * std / TRUNC_STD)."""
    s = std / TRUNC_STD
    return nn.init.trunc_normal_(weight, std=s, a=-TRUNC_A * s, b=TRUNC_A * s)


def init_module(m, depths):
    """The one initialisation, applied to every module (``PolicyNet.apply``)."""
    if isinstance(m, nn.Linear):
        if getattr(m, "zero_init", False):
            nn.init.zeros_(m.weight)
        else:
            key = getattr(m, "residual", None)
            std = INIT_STD / math.sqrt(2 * depths[key]) if key else INIT_STD
            trunc_normal_(m.weight, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=INIT_STD)
    elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, PolicyNet):
        for p in (m.latent, m.bos, getattr(m, "feat_missing", None), getattr(m, "tile_missing", None)):
            if p is not None:
                nn.init.normal_(p, std=INIT_STD)


@dataclass
class Context:
    mem_kv: list                 # per decoder layer (k, v) of the field memory
    cells: torch.Tensor          # [B, 100, d] own-farm field memory (local reads)
    unit_feats: torch.Tensor     # [B, U, d] unit identity/position/previous-token/local features
    present: torch.Tensor        # [B, U] bool
    field: torch.Tensor          # [B, 200 + S, width] the thinking field (diagnostics)
    codes: Optional[torch.Tensor]  # [B, 200 + S, fsq_dims] FSQ level indices (think='fsq')
    history: Optional[torch.Tensor] = None  # [B, d] direct causal readout of carried slots
    state_in: Optional[torch.Tensor] = None     # incoming carried state after resets (commit input)
    write_src: Optional[torch.Tensor] = None    # [B, L, W] final trunk latents (field/carry FIFO write)
    next_state: Optional[torch.Tensor] = None   # fsq/off: the new state (no action dependence)
    econ: Optional[torch.Tensor] = None         # [B, d] stock_mlp(own stock + cash), added to every decision query
    copy_drop: Optional[torch.Tensor] = None    # [B] bool: training copy dropout (no FIFO action write)


def _get(batch, key):
    """Top-level tensor, else from batch['meta'] (dict of arrays or list of dicts); None if absent."""
    if key in batch and batch[key] is not None:
        return batch[key]
    meta = batch.get("meta")
    if isinstance(meta, dict) and key in meta:
        return torch.as_tensor(np.asarray(meta[key]))
    if isinstance(meta, (list, tuple)) and meta and all(isinstance(m, dict) and key in m for m in meta):
        return torch.as_tensor([int(m[key]) for m in meta])
    return None


# ----------------------------------------------------------------------------- policy
class PolicyNet(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None, codec=None):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        c = self.config = (config or ModelConfig()).resolve(codec)
        self._codec_module = codec
        tw, W, d = c.tok_width, c.trunk_width, c.dec_width
        L, S = c.trunk_latents, c.n_slots
        self.U, self.O, self.D = c.max_units, c.n_order_slots, c.digit_slots
        self.bottleneck = c.think in ("field", "fsq")
        self.S = S if self.bottleneck else L            # global slots in the field memory
        self.field_width = {"field": c.field_dc, "fsq": c.fsq_dims}.get(c.think, d)
        # input tokens
        self.inp = nn.Linear(IN_WIDTH, tw)
        self.known_inp = nn.Linear(IN_WIDTH, tw, bias=False)
        self.tok_type = nn.Embedding(len(TOKEN_TYPES), tw)
        if c.features:
            self.feat_inp = nn.Linear(c.feat_width, tw)
            self.feat_known = nn.Linear(c.feat_width, tw, bias=False)
            self.tile_inp = nn.Linear(c.tile_width, tw)
            if c.feat_missing:          # training feature dropout: learned MISSING values (knownness 0)
                self.feat_missing = nn.Parameter(torch.empty(c.feat_tokens, c.feat_width))
                self.tile_missing = nn.Parameter(torch.empty(c.tile_width))
        if c.prev_unit_tokens:
            self.prev_tok = nn.Embedding(c.n_unit_tokens, tw)
            self.prev_slot = nn.Embedding(self.U, tw)
        if c.teacher_token:
            self.team = nn.Embedding(c.n_teams, tw)
            self.strength = nn.Embedding(c.n_strength + 1, tw)
        if c.pos_embed:                 # TILE + CARRIER tokens: role, x, y; CARRIER also ordinal (U = >= U)
            self.pos_role = nn.Embedding(2, tw)
            self.pos_x, self.pos_y = nn.Embedding(BOARD, tw), nn.Embedding(BOARD, tw)
            self.pos_ord = nn.Embedding(self.U + 1, tw)
        self.tok_norm = nn.RMSNorm(tw)
        # trunk
        self.latent = nn.Parameter(torch.empty(L, W))
        if c.think in ("field", "fsq"):
            self.carry_in = nn.Linear(self.field_width, W)
            if S != L:
                self.slot_mix = nn.Linear(L, S, bias=False)
                self.carry_mix = nn.Linear(S, L, bias=False)
        elif c.think == "carry":
            self.carry_in = nn.Linear(W, W)
        if c.think in ("field", "carry"):
            # Causal FIFO register, one entry per step (a learned recurrent slot update attenuated
            # sparse cues before they reached the next decision). commit() writes the entry after
            # the step's action is fixed: Linear(attention-pooled final trunk latents) +
            # Linear(pooled embedding of the step's own submitted action, units + market).
            wd = self.state_shape[1]
            self.write_pool = nn.Linear(W, 1)
            self.history_write = nn.Linear(W, wd, bias=False)
            self.act_norm = nn.RMSNorm(d)
            self.act_ff = SwiGLU(d, c.dec_hidden, None)
            self.act_out_norm = nn.RMSNorm(d)
            self.action_write = nn.Linear(d, wd, bias=False)
            self.history_read = (nn.Identity() if c.think == "field"
                                 else nn.Linear(W, self.field_width, bias=False))
            self.history_query = nn.Linear(self.S * self.state_shape[1], d, bias=False)
        self.in_norm = nn.RMSNorm(W)
        self.cross = CrossBlock(W, tw, c.trunk_heads, c.trunk_hidden)
        self.blocks = nn.ModuleList(LatentBlock(W, c.trunk_heads, c.trunk_hidden) for _ in range(c.trunk_layers))
        self.trunk_norm = nn.RMSNorm(W)
        # thinking field
        self.cell_qnorm = nn.RMSNorm(tw)
        self.cell_read = CrossAttention(tw, W, c.dec_heads, inner=d, out=False)
        self.cell_norm = nn.RMSNorm(tw + d)
        self.cell_out = nn.Linear(tw + d, self.field_width)
        self.slot_out = nn.Linear(W, self.field_width)
        self.mem_in = nn.Linear(self.field_width, d) if self.bottleneck else nn.Identity()
        self.farm_emb, self.row_emb, self.col_emb = nn.Embedding(2, d), nn.Embedding(BOARD, d), nn.Embedding(BOARD, d)
        self.slot_pos = nn.Embedding(self.S, d)
        self.mem_norm = nn.RMSNorm(d)
        # decoder
        U, O = self.U, self.O
        self.dec_slot = nn.Embedding(U + O, d)
        self.unit_row, self.unit_col = nn.Embedding(BOARD, d), nn.Embedding(BOARD, d)
        self.prev_own = nn.Embedding(c.n_unit_tokens + 1, d)          # 0 = no previous token
        if c.local:
            self.local_proj = nn.Linear(5 * d, d)
        if c.unit_carrier:              # the unit's own CARRIER row value columns (carried, key rank, flags)
            self.carrier_mlp = nn.Sequential(nn.Linear(IN_WIDTH - PACK_PAYLOAD, d), nn.SiLU(), nn.Linear(d, d))
        self.bos = nn.Parameter(torch.empty(d))
        self.unit_tok_emb = nn.Embedding(c.n_unit_tokens, d)
        self.order_tok_emb = nn.Embedding(c.n_order_tokens, d)
        self.qty_emb = nn.Embedding(c.n_qty + 1, d)                    # index n_qty = no quantity
        self.dec = nn.ModuleList(DecoderLayer(d, c.dec_heads, c.dec_hidden) for _ in range(c.dec_layers))
        self.dec_norm = nn.RMSNorm(d)
        self.unit_head = nn.Linear(d, c.n_unit_tokens)
        self.order_head = nn.Linear(d, c.n_order_tokens)
        self.unit_head.zero_init = self.order_head.zero_init = True
        self.unit_qty = QtyHead(d, c.dec_hidden, c.n_unit_tokens, c.n_qty)
        self.order_qty = QtyHead(d, c.dec_hidden, c.n_order_tokens, c.n_qty)
        self.digits = DigitHead(d, c.dec_hidden, self.D, c.n_digit, c.digit_end)
        # constant buffers
        cell = torch.arange(N_CELLS)
        self.register_buffer("cell_role", cell // 100, persistent=False)
        self.register_buffer("cell_y", (cell % 100) // BOARD, persistent=False)
        self.register_buffer("cell_x", cell % BOARD, persistent=False)
        self.register_buffer("unit_qty_mask", torch.tensor(c.unit_has_qty, dtype=torch.bool), persistent=False)
        self.register_buffer("order_qty_mask", torch.tensor(c.order_has_qty, dtype=torch.bool), persistent=False)
        self.register_buffer("neigh", _neighbours(), persistent=False)
        self.register_buffer("causal", torch.ones(U + O, U + O, dtype=torch.bool).tril(), persistent=False)
        forbid = torch.zeros(O, c.n_order_tokens, dtype=torch.bool)
        forbid[O - 1] = True
        forbid[O - 1, c.stop_index] = False
        self.register_buffer("order_forbid", forbid, persistent=False)
        self._trace = None
        depths = {"trunk": c.trunk_layers + 1, "dec": c.dec_layers}
        self.apply(partial(init_module, depths=depths))
        if c.stock_path:
            # built after the shared init and from a forked RNG: every other parameter and the RNG
            # stream after construction are those of the same config with stock_path off; the last
            # layer is zero, so an untrained model's outputs are unchanged too
            with torch.random.fork_rng(devices=[]):         # nn.Linear's own reset draws too
                self.stock_mlp = nn.Sequential(nn.Linear(ECON_WIDTH, d), nn.SiLU(), nn.Linear(d, d))
                self.stock_mlp[2].zero_init = True
                self.stock_mlp.apply(partial(init_module, depths=depths))

    # ------------------------------------------------------------------ helpers
    @property
    def codec(self):
        return self._codec_module or _codec()

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def param_groups(self, weight_decay=0.1):
        """AdamW groups: decay on nn.Linear weights only (no biases, norms, embeddings, latents)."""
        decay, rest = [], []
        for mod in self.modules():
            for name, p in mod.named_parameters(recurse=False):
                (decay if isinstance(mod, nn.Linear) and name == "weight" else rest).append(p)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}]

    @property
    def state_shape(self):
        """(rows, width) of the carried state per example."""
        c = self.config
        if c.think == "carry":
            return c.trunk_latents, c.trunk_width
        return c.n_slots, (c.fsq_dims if c.think == "fsq" else c.field_dc)   # 'off': unused zeros

    def initial_state(self, batch_size, device=None, dtype=torch.float32):
        """Carried state at an episode (or day) start: zeros [B, rows, width]."""
        return torch.zeros(batch_size, *self.state_shape, device=device, dtype=dtype)

    def _tiles(self, rows, token_mask):
        """Locate the 200 TILE tokens from their own columns: (is_tile, cell_of_token, tile_index)."""
        B, N, _ = rows.shape
        is_tile = (rows[..., TILE_KIND_COL] > 0.5) & token_mask
        role = (rows[..., ROLE_COL + 1] > 0.5).long()
        x = torch.round(rows[..., XY_COL] * (BOARD - 1)).long().clamp(0, BOARD - 1)
        y = torch.round(rows[..., XY_COL + 1] * (BOARD - 1)).long().clamp(0, BOARD - 1)
        cell = torch.where(is_tile, role * 100 + y * BOARD + x, torch.full_like(x, N_CELLS))
        index = torch.full((B, N_CELLS + 1), -1, dtype=torch.long, device=rows.device)
        index.scatter_(1, cell, torch.arange(N, device=rows.device).expand(B, N).contiguous())
        index = index[:, :N_CELLS]
        if bool((index < 0).any()) or bool((is_tile.sum(1) != N_CELLS).any()):
            raise ValueError("exactly one TILE token per cell of both 10x10 farms required")
        return is_tile, cell.clamp(max=N_CELLS - 1), index

    def _reset(self, batch, B, device):
        """[B] bool: carried state restarts (step 0; hour 0 when carry_scope='day'; batch['reset'])."""
        reset = torch.zeros(B, dtype=torch.bool, device=device)
        explicit = batch.get("reset")
        if explicit is not None:
            reset |= torch.as_tensor(explicit, device=device).bool()
        step = _get(batch, "step")
        if step is not None:
            reset |= step.to(device) <= 0
        if self.config.carry_scope == "day":
            hour = _get(batch, "hour")
            if hour is None and step is not None:
                hour = step % 24
            if hour is not None:
                reset |= hour.to(device) == 0
        return reset

    # ------------------------------------------------------------------ encoder
    def _tokens(self, batch, rows, token_mask, known):
        c = self.config
        B, N, _ = rows.shape
        types = self.tok_type.weight
        vals = rows.float().masked_fill(~known, 0)
        if c.value_log:             # value columns only; kind/role/x/y/ordinal/item/tile-kind stay linear
            vals = torch.cat((vals[..., :PACK_PAYLOAD], signed_log(vals[..., PACK_PAYLOAD:], c.value_log)), -1)
        with fp32_inputs(rows.device, c.input_fp32):
            x = self.inp(vals) + self.known_inp(known.to(vals.dtype))
        is_tile, cell_of, tile_index = self._tiles(rows, token_mask)
        if c.pos_embed:
            x = x + self._pos_embed(rows, token_mask)
        groups, masks = [], [token_mask]
        if c.features:
            feat = batch["feat"]
            fk = feat["known"].to(rows.device).bool()
            fv = feat["values"].to(rows.device).float().masked_fill(~fk, 0)
            tile_in = feat["tile"].to(rows.device).float()
            if c.value_log:         # before dropout: the MISSING vectors live in the transformed space
                fv = torch.cat((fv[..., :FEAT_PAYLOAD], signed_log(fv[..., FEAT_PAYLOAD:], c.value_log)), -1)
                tile_in = signed_log(tile_in, c.value_log)
            drop = batch.get("feat_drop") if self.training else None
            if drop is not None:
                fv, fk, tile_in = self._feature_dropout(drop, fv, fk, tile_in, is_tile)
            with fp32_inputs(rows.device, c.input_fp32):
                groups.append(self.feat_inp(fv) + self.feat_known(fk.to(fv.dtype)) + types[T_FEAT])
                tile = self.tile_inp(tile_in)
            masks.append(torch.ones(fv.shape[:2], dtype=torch.bool, device=rows.device))
            if tile.shape[1] == N_CELLS:          # cell-indexed [B, 200, TW]
                tile = tile.gather(1, cell_of[..., None].expand(B, N, tile.shape[-1])) * is_tile[..., None].to(tile.dtype)
            elif tile.shape[1] != N:              # feature_tokens.align_tiles layout [B, N, TW]
                raise ValueError("feat['tile'] must be [B, 200, TW] or aligned [B, N, TW]")
            x = x + tile
        tile_tok = x.gather(1, tile_index[..., None].expand(B, N_CELLS, x.shape[-1]))
        if c.prev_unit_tokens:
            prev = _get(batch, "prev_unit_tok")
            if prev is None:
                prev = torch.full((B, self.U), -1, dtype=torch.long, device=rows.device)
            prev = prev.to(rows.device).long()
            groups.append(self.prev_tok(prev.clamp(min=0)) + self.prev_slot.weight + types[T_PREV])
            masks.append(prev >= 0)
        if c.teacher_token:
            team = _get(batch, "team_id")
            team = torch.zeros(B, dtype=torch.long) if team is None else team.long()
            team = team.to(rows.device).clamp(0, c.n_teams - 1)
            if self.training and c.team_dropout > 0:
                team = torch.where(torch.rand(B, device=rows.device) < c.team_dropout, 0, team)
            dec = _get(batch, "strength_decile")
            dec = torch.full((B,), -1, dtype=torch.long) if dec is None else dec.long()
            dec = dec.to(rows.device)
            dec = torch.where((dec >= 0) & (dec < c.n_strength), dec, c.n_strength)
            groups.append((self.team(team) + self.strength(dec) + types[T_TEACHER])[:, None])
            masks.append(torch.ones(B, 1, dtype=torch.bool, device=rows.device))
        tokens = torch.cat([x + types[T_PACK], *groups], 1)
        mask = torch.cat(masks, 1)
        return tokens, mask, tile_tok

    def _pos_embed(self, rows, token_mask):
        """[B, N, tok_width]: learned role + x + y embeddings on every TILE and CARRIER token (both
        farms), + an ordinal embedding on CARRIER tokens (ordinal >= max_units share the last
        entry), read from each row's own kind/role/x/y/ordinal columns; 0 on other tokens."""
        tile = (rows[..., TILE_KIND_COL] > 0.5) & token_mask
        car = (rows[..., CARRIER_KIND_COL] > 0.5) & token_mask
        role = (rows[..., ROLE_COL + 1] > 0.5).long()
        x = torch.round(rows[..., XY_COL].float() * (BOARD - 1)).long().clamp(0, BOARD - 1)
        y = torch.round(rows[..., XY_COL + 1].float() * (BOARD - 1)).long().clamp(0, BOARD - 1)
        o = torch.round(rows[..., ORD_COL].float() * 100).long().clamp(0, self.U)
        e = (self.pos_role(role) + self.pos_x(x) + self.pos_y(y)) * (tile | car)[..., None].float()
        return e + self.pos_ord(o) * car[..., None].float()

    def _feature_dropout(self, drop, fv, fk, tile_in, is_tile):
        """Training feature dropout: dropped columns take the learned MISSING values with
        knownness 0; a dropped tile block takes the learned tile MISSING vector on every tile."""
        if not hasattr(self, "feat_missing"):
            raise ValueError("feature dropout needs a model built with feat_missing=True")
        cols = drop["cols"].to(fv.device).bool()
        fv = torch.where(cols, self.feat_missing.to(fv.dtype), fv)
        fk = fk & ~cols
        tdrop = drop["tile"].to(fv.device).bool().view(-1, 1, 1)
        if tile_in.shape[1] != N_CELLS:          # aligned [B, N, TW] layout: tiles only
            tdrop = tdrop & is_tile[..., None]
        tile_in = torch.where(tdrop, self.tile_missing.to(tile_in.dtype), tile_in)
        return fv, fk, tile_in

    def _trace_add(self, x):
        if self._trace is not None:
            self._trace.append(float(x.detach().float().std()))

    def encode(self, batch, state=None):
        """Inputs + carried state -> (Context, new_state)."""
        c = self.config
        batch, copy_drop = self._copy_dropout(batch)
        rows = batch["rows"].float()
        token_mask = batch["token_mask"].bool()
        known = batch["knownness"].bool() & token_mask[..., None]
        B, device = rows.shape[0], rows.device
        tokens, tmask, tile_tok = self._tokens(batch, rows, token_mask, known)
        tokens = self.tok_norm(tokens)
        # trunk
        if state is None:
            state = self.initial_state(B, device, rows.dtype)
        reset = self._reset(batch, B, device)
        if bool(reset.any()):
            state = torch.where(reset[:, None, None], torch.zeros((), dtype=state.dtype, device=device), state)
        lat = self.latent.expand(B, -1, -1)
        if c.think != "off":
            carried = self.carry_in(state.to(lat.dtype))
            if hasattr(self, "carry_mix"):
                carried = self.carry_mix(carried.transpose(1, 2)).transpose(1, 2)
            lat = lat + carried
        lat = self.in_norm(lat)
        self._trace_add(lat)
        attn = None if bool(tmask.all()) else tmask[:, None, None, :]
        lat = self.cross(lat, tokens, attn)
        self._trace_add(lat)
        for block in self.blocks:
            lat = block(lat)
            self._trace_add(lat)
        lat = self.trunk_norm(lat)
        # thinking field
        read = self.cell_read(self.cell_qnorm(tile_tok), self.cell_read.memory(lat))
        cells = self.cell_out(self.cell_norm(torch.cat((tile_tok, read), -1)))
        slot_src = self.slot_mix(lat.transpose(1, 2)).transpose(1, 2) if hasattr(self, "slot_mix") else lat
        slots = self.slot_out(slot_src)
        if c.think in ("field", "carry"):
            slots = slots + self.history_read(state.to(slots.dtype))
        codes = None
        if c.think == "field":
            cells, slots = torch.tanh(cells), torch.tanh(slots)
        elif c.think == "fsq":
            (cells, cc), (slots, sc) = fsq(cells, c.fsq_levels), fsq(slots, c.fsq_levels)
            codes = torch.cat((cc, sc), 1)
        field = torch.cat((cells, slots), 1)
        pos = torch.cat((self.farm_emb(self.cell_role) + self.row_emb(self.cell_y) + self.col_emb(self.cell_x),
                         self.slot_pos.weight), 0)
        mem = self.mem_norm(self.mem_in(field) + pos)
        present = batch["unit_pos"][..., 0].to(device) >= 0
        present[:, 0] = True
        ctx = Context([layer.memory(mem) for layer in self.dec], mem[:, :BOARD * BOARD], None, present,
                      field, codes)
        ctx.state_in = state
        if c.think in ("field", "carry"):
            ctx.history = self.history_query(state.flatten(1).to(mem.dtype))
            ctx.write_src = lat
        else:
            ctx.next_state = {"fsq": slots}.get(c.think, state)
        ctx.unit_feats = self._unit_feats(ctx, batch)
        ctx.copy_drop = copy_drop
        if c.stock_path:
            with fp32_inputs(device, c.input_fp32):
                ctx.econ = self.stock_mlp(self.econ_values(rows, token_mask, known))
        return ctx

    def econ_values(self, rows, token_mask, known):
        """[B, ECON_WIDTH] fp32, fixed order: shed quantity of each observation.ITEMS item, seed
        quantity on the same item index (only the 5 crops occur), own cash; signed_log of the pack
        value (quantity / 100, cash / 10000). Read from the actor's own (role column 12) STOCK
        rows by item one-hot and seed flag and from its own FARM row; 0 where absent or unknown."""
        rows = rows.float()
        own = (rows[..., ROLE_COL] > 0.5) & token_mask.bool()
        value = rows[..., PACK_PAYLOAD].masked_fill(~known[..., PACK_PAYLOAD].bool(), 0)
        stock = own & (rows[..., STOCK_KIND_COL] > 0.5)
        item = ((rows[..., ITEM_COL:ITEM_COL + N_ITEMS] > 0.5) & stock[..., None]).to(value.dtype)
        seed = (rows[..., STOCK_SEED_COL] > 0.5).to(value.dtype)
        shed = (item * (value * (1 - seed))[..., None]).sum(1)
        seeds = (item * (value * seed)[..., None]).sum(1)
        farm = (own & (rows[..., FARM_KIND_COL] > 0.5)).to(value.dtype)
        cash = (farm * value).sum(1, keepdim=True)
        return signed_log(torch.cat((shed, seeds, cash), -1), self.config.value_log)

    def _copy_dropout(self, batch):
        """Training copy dropout: rows with batch['copy_drop'] True lose the teacher-action inputs
        (prev_unit_tok -1, team 0, strength decile unknown; commit writes no action term). Train
        mode only. -> (batch view with those inputs replaced, [B] bool mask or None)."""
        drop = batch.get("copy_drop") if self.training else None
        if drop is None:
            return batch, None
        device = batch["rows"].device
        drop = torch.as_tensor(drop).to(device).bool().view(-1)
        out = dict(batch)
        prev = _get(batch, "prev_unit_tok")
        if prev is not None:
            prev = prev.to(device).long()
            out["prev_unit_tok"] = torch.where(drop[:, None], torch.full_like(prev, -1), prev)
        team = _get(batch, "team_id")
        if team is not None:
            team = team.to(device).long()
            out["team_id"] = torch.where(drop, torch.zeros_like(team), team)
        dec = _get(batch, "strength_decile")
        if dec is not None:
            dec = dec.to(device).long()
            out["strength_decile"] = torch.where(drop, torch.full_like(dec, -1), dec)
        return out, drop

    def commit(self, ctx, labels):
        """Carried state for the next step, once this step's action is fixed (``labels``: the
        action submitted at this step in the codec label layout, the teacher's in training and
        the decoded / submitted one at play). field/carry: push Linear(attention pool of the
        final trunk latents) + Linear(action summary) onto the FIFO (newest first, the oldest
        entry drops); fsq: the FSQ slots; off: unchanged. Causal: the entry is read from the
        next step on."""
        if ctx.write_src is None:
            return ctx.next_state
        lat = ctx.write_src
        attn = torch.softmax(self.write_pool(lat).float(), 1).to(lat.dtype)          # [B, L, 1]
        write = self.history_write((attn * lat).sum(1))
        act = self.action_write(self._action_summary(ctx, labels)).to(write.dtype)
        if ctx.copy_drop is not None:            # training copy dropout: no action term for these rows
            act = act * (~ctx.copy_drop).to(act.dtype)[:, None]
        write = write + act
        state = ctx.state_in
        return torch.cat((write.unsqueeze(1), state[:, :-1].to(write.dtype)), 1)

    def _action_summary(self, ctx, labels):
        """[B, d]: RMSNorm(sum over the step's decisions of SwiGLU(RMSNorm(decision embedding +
        decision-slot embedding))); units only where present, orders up to STOP (tok >= 0)."""
        present = ctx.present
        B, device = present.shape[0], present.device
        U, O, D = self.U, self.O, self.D

        def get(key, shape):
            v = labels.get(key) if labels is not None else None
            if v is None:
                return torch.full(shape, -1, dtype=torch.long, device=device)
            return torch.as_tensor(v).to(device).long().reshape(shape)
        ut, uq, ud = get("unit_tok", (B, U)), get("unit_qty", (B, U)), get("unit_digits", (B, U, D))
        ot, oq, od = get("order_tok", (B, O)), get("order_qty", (B, O)), get("order_digits", (B, O, D))
        ut = torch.where(present, ut, -1)
        emb = torch.cat((self._decision_emb(ut, uq, ud, True), self._decision_emb(ot, oq, od, False)), 1)
        emb = emb + self.dec_slot.weight.to(emb.dtype)
        valid = torch.cat((ut >= 0, ot >= 0), 1)
        h = self.act_ff(self.act_norm(emb)) * valid[..., None].to(emb.dtype)
        return self.act_out_norm(h.sum(1))

    def _unit_feats(self, ctx, batch):
        """[B, U, d]: x/y embeddings + previous own token + own and 4 neighbour field cells
        (+ carrier_mlp of the unit's own CARRIER row when unit_carrier)."""
        pos = batch["unit_pos"].to(ctx.cells.device).long()
        B, U = pos.shape[:2]
        x, y = pos[..., 0].clamp(0, BOARD - 1), pos[..., 1].clamp(0, BOARD - 1)
        prev = _get(batch, "prev_unit_tok")
        prev = torch.full((B, U), -1, dtype=torch.long) if prev is None else prev.long()
        out = self.unit_row(y) + self.unit_col(x) + self.prev_own(prev.to(pos.device).clamp(min=-1) + 1)
        if self.config.local:
            cells = torch.cat((ctx.cells, ctx.cells.new_zeros(B, 1, ctx.cells.shape[-1])), 1)
            ids = self.neigh[(y * BOARD + x).reshape(-1)].view(B, U * 5)
            local = cells.gather(1, ids[..., None].expand(B, U * 5, cells.shape[-1])).view(B, U, -1)
            out = out + self.local_proj(local)
        if self.config.unit_carrier:
            out = out + self._carrier_feats(batch, B, U, pos.device).to(out.dtype)
        return out

    def _carrier_feats(self, batch, B, U, device):
        """[B, U, d]: carrier_mlp(signed-log value columns of each unit's own CARRIER row) (fp32,
        outside autocast); 0 where batch['carrier_row'] is -1. carrier_row comes from collate
        (unit_index.carrier_rows); a batch without it gets the same function here."""
        c = self.config
        rows = batch["rows"].to(device).float()
        tm = batch["token_mask"].to(device).bool()
        known = batch["knownness"].to(device).bool() & tm[..., None]
        cr = batch.get("carrier_row")
        if cr is None:
            import unit_index
            cr = torch.as_tensor(unit_index.carrier_rows(rows.detach().cpu().numpy(), tm.cpu().numpy(),
                                                         batch["unit_pos"].detach().cpu().numpy()))
        cr = cr.to(device).long()
        vals = rows[..., PACK_PAYLOAD:].masked_fill(~known[..., PACK_PAYLOAD:], 0)
        own = signed_log(vals.gather(1, cr.clamp(min=0)[..., None].expand(B, U, vals.shape[-1])), c.value_log)
        with fp32_inputs(device, c.input_fp32):
            e = self.carrier_mlp(own)
        return e * (cr >= 0)[..., None].to(e.dtype)

    # ------------------------------------------------------------------ decoder pieces
    def _decision_emb(self, tok, qty, digits, unit):
        """Embedding of a chosen decision: token + quantity class (or 'none') + digit summary."""
        table, has = (self.unit_tok_emb, self.unit_qty_mask) if unit else (self.order_tok_emb, self.order_qty_mask)
        t = tok.clamp(min=0)
        q = torch.where(has[t] & (qty >= 0), qty, torch.full_like(qty, self.config.n_qty))
        out = table(t) + self.qty_emb(q)
        if digits is not None:
            exact = (q == self.config.qty_exact).unsqueeze(-1).to(out.dtype)
            out = out + self.digits.summary(digits) * exact
        return out

    @staticmethod
    def _labels(batch):
        lab = batch.get("labels", batch)
        return {k: lab.get(k) for k in ("unit_tok", "unit_qty", "unit_digits", "order_tok", "order_qty", "order_digits")}

    def decode_teacher(self, ctx, batch):
        U, O, D = self.U, self.O, self.D
        lab = self._labels(batch)
        device = ctx.present.device
        ut, uq, ot, oq = (lab[k].to(device).long() for k in ("unit_tok", "unit_qty", "order_tok", "order_qty"))
        B = ut.shape[0]
        ud = lab["unit_digits"]
        od = lab["order_digits"]
        ud = torch.full((B, U, D), -1, dtype=torch.long, device=device) if ud is None else ud.to(device).long()
        od = torch.full((B, O, D), -1, dtype=torch.long, device=device) if od is None else od.to(device).long()
        present = ctx.present
        dec = torch.cat((self._decision_emb(ut, uq, ud, True), self._decision_emb(ot, oq, od, False)), 1)
        d = dec.shape[-1]
        prev = (torch.arange(U + O, device=device) - 1).expand(B, -1).clone()
        prev[:, U] = present.sum(1) - 1                          # order 0 follows the last present unit
        prev_emb = dec.gather(1, prev.clamp(min=0)[..., None].expand(-1, -1, d))
        prev_emb = torch.cat((self.bos.to(dec.dtype).expand(B, 1, d), prev_emb[:, 1:]), 1)
        x = self.dec_slot.weight[None] + prev_emb
        if ctx.history is not None:
            x = x + ctx.history[:, None]
        if ctx.econ is not None:
            x = x + ctx.econ[:, None].to(x.dtype)
        x = torch.cat((x[:, :U] + ctx.unit_feats, x[:, U:]), 1)
        valid = torch.cat((present, present.new_ones(B, O)), 1)
        mask = self.causal[None, None] & valid[:, None, None, :]
        for layer, kv in zip(self.dec, ctx.mem_kv):
            x = layer(x, mask, kv)
        h = self.dec_norm(x)
        hu, ho = h[:, :U], h[:, U:]
        uql, uhq = self.unit_qty(hu, ut.clamp(min=0))
        oql, ohq = self.order_qty(ho, ot.clamp(min=0))
        return {"unit_logits": self.unit_head(hu),
                "unit_qty_logits": uql,
                "unit_digit_logits": self.digits.teacher(uhq, ud),
                "order_logits": self.order_head(ho).masked_fill(self.order_forbid, float("-inf")),
                "order_qty_logits": oql,
                "order_digit_logits": self.digits.teacher(ohq, od),
                "unit_present": present}

    def forward(self, batch, state=None):
        """Teacher-forced step -> (outputs, new_state). outputs: unit_logits [B,U,Vu],
        unit_qty_logits [B,U,Q], unit_digit_logits [B,U,D,Nd], order_logits [B,O,Vo] (the last
        slot STOP only), order_qty_logits [B,O,Q], order_digit_logits [B,O,D,Nd], unit_present,
        field [B, 200+S, width], fsq_codes (think='fsq') . new_state keeps its graph (detach it
        at chunk boundaries); field/carry write the teacher's labels of this step (commit)."""
        ctx = self.encode(batch, state)
        out = self.decode_teacher(ctx, batch)
        new_state = self.commit(ctx, self._labels(batch))
        out["field"] = ctx.field
        out["fsq_codes"] = ctx.codes
        return out, new_state

    # ------------------------------------------------------------------ free-running decode (cached)
    @staticmethod
    def _choose(logits, temperature, generator):
        if not temperature:
            return logits.argmax(-1)
        p = torch.softmax(logits.float() / temperature, -1)
        return torch.multinomial(p, 1, generator=generator).squeeze(-1)

    def _dec_step(self, x, caches, mem_kv, valid):
        mask = None
        if x.shape[0] > 1:          # B == 1: every decoded decision is valid (units are a prefix)
            v = torch.stack(valid, 1)
            mask = None if bool(v.all()) else v[:, None, None, :]
        x = x[:, None]
        for layer, kv, cache in zip(self.dec, mem_kv, caches):
            x = layer(x, mask, kv, cache)
        return self.dec_norm(x[:, 0])

    def _decide(self, h, head, qty_head, has_mask, choose, forbid=None, force_tok=None, force_exact=False,
                unit=True):
        """One decision from its hidden state: token -> quantity class -> digits.
        -> (tok, qty, digits, decision embedding)."""
        B = h.shape[0]
        logits = head(h)
        if forbid is not None:
            logits = logits.masked_fill(forbid, float("-inf"))
        tok = choose(logits) if force_tok is None else torch.full((B,), force_tok, dtype=torch.long, device=h.device)
        has = has_mask[tok]
        qty = torch.full_like(tok, -1)
        digits = None
        if bool(has.any()):
            qlog, hq = qty_head(h, tok)
            q = choose(qlog)
            if force_exact:
                q = torch.full_like(q, self.config.qty_exact)
            qty = torch.where(has, q, -1)
            need = qty == self.config.qty_exact
            if bool(need.any()):
                digits, _ = self.digits.decode(hq, need, choose, force_full=force_exact)
        return tok, qty, digits, self._decision_emb(tok, qty, digits, unit)

    def decode_free(self, ctx, *, temperature=0.0, generator=None, deadline=None,
                    force_orders=False, force_exact=False):
        """Sequential decode farmer -> hands -> orders -> STOP with a K/V cache (one query token
        per decision). deadline: time.perf_counter() value; once passed, every undecided unit is
        PASS and the market gets STOP. force_orders / force_exact are latency worst cases (all
        order slots decoded; every decision a quantity token with QTY_EXACT and all digits)."""
        c = self.config
        U, O, D = self.U, self.O, self.D
        present = ctx.present
        B, device = present.shape[0], present.device
        n_units = int(present.sum(1).max())
        caches = [dict() for _ in self.dec]
        choose = partial(self._choose, temperature=temperature, generator=generator)
        last = self.bos.expand(B, -1)
        valid = []
        unit_tok = torch.full((B, U), -1, dtype=torch.long, device=device)
        unit_qty = torch.full_like(unit_tok, -1)
        unit_digits = torch.full((B, U, D), -1, dtype=torch.long, device=device)
        order_tok = torch.full((B, O), -1, dtype=torch.long, device=device)
        order_qty = torch.full_like(order_tok, -1)
        order_digits = torch.full((B, O, D), -1, dtype=torch.long, device=device)
        unit_force = next(i for i, v in enumerate(c.unit_has_qty) if v) if force_exact else None
        order_force = next(i for i, v in enumerate(c.order_has_qty) if v) if force_exact else None
        cut = False
        for p in range(n_units):
            here = present[:, p]
            if deadline is not None and not cut and _now() > deadline:
                cut = True
            if cut:
                unit_tok[:, p] = torch.where(here, c.pass_index, -1)
                continue
            valid.append(here)
            query = self.dec_slot.weight[p] + last + ctx.unit_feats[:, p]
            if ctx.history is not None:
                query = query + ctx.history
            if ctx.econ is not None:
                query = query + ctx.econ.to(query.dtype)
            h = self._dec_step(query, caches, ctx.mem_kv, valid)
            tok, qty, digits, emb = self._decide(h, self.unit_head, self.unit_qty, self.unit_qty_mask, choose,
                                                 force_tok=unit_force, force_exact=force_exact, unit=True)
            unit_tok[:, p] = torch.where(here, tok, -1)
            unit_qty[:, p] = torch.where(here, qty, -1)
            if digits is not None:
                unit_digits[:, p] = torch.where(here[:, None], digits, -1)
            last = torch.where(here[:, None], emb, last)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        stop = c.stop_index
        for k in range(O):
            if deadline is not None and not cut and _now() > deadline:
                cut = True
            if cut:
                order_tok[:, k] = torch.where(done, -1, stop)
                break
            valid.append(torch.ones(B, dtype=torch.bool, device=device))
            query = self.dec_slot.weight[U + k] + last
            if ctx.history is not None:
                query = query + ctx.history
            if ctx.econ is not None:
                query = query + ctx.econ.to(query.dtype)
            h = self._dec_step(query, caches, ctx.mem_kv, valid)
            force = order_force if (force_exact and k < O - 1) else None
            tok, qty, digits, emb = self._decide(h, self.order_head, self.order_qty, self.order_qty_mask, choose,
                                                 forbid=self.order_forbid[k], force_tok=force,
                                                 force_exact=force_exact, unit=False)
            order_tok[:, k] = torch.where(done, -1, tok)
            order_qty[:, k] = torch.where(done, -1, qty)
            if digits is not None:
                order_digits[:, k] = torch.where(done[:, None], -1, digits)
            done = done | (tok == stop)
            last = emb
            if bool(done.all()) and not force_orders:
                break
        return {"unit_tok": unit_tok, "unit_qty": unit_qty, "unit_digits": unit_digits,
                "order_tok": order_tok, "order_qty": order_qty, "order_digits": order_digits,
                "deadline_hit": torch.full((B,), cut, dtype=torch.bool, device=device)}

    @torch.no_grad()
    def act(self, inputs, state=None, *, deadline=None, temperature=0.0, generator=None,
            force_orders=False, force_exact=False, commit_labels=None):
        """Free-running step -> (labels, new_state). inputs = collate([example]) (no labels
        needed); labels = decode_free output (tensors, codec label layout); pass the returned
        state to the next step of the same seat; state=None starts fresh. The new state writes
        the decoded labels, or ``commit_labels`` (same layout) when the submitted action is not
        the decoded one (teacher-forced agent mode, parity checks)."""
        ctx = self.encode(inputs, state)
        labels = self.decode_free(ctx, temperature=temperature, generator=generator, deadline=deadline,
                                  force_orders=force_orders, force_exact=force_exact)
        return labels, self.commit(ctx, labels if commit_labels is None else commit_labels)

    def codec_labels(self, labels, index=0):
        """One row of act() labels -> python lists for codec.decode_step (orders up to STOP)."""
        out = {k: labels[k][index].tolist() for k in ("unit_tok", "unit_qty", "unit_digits")}
        ot = labels["order_tok"][index].tolist()
        n = next((i + 1 for i, t in enumerate(ot) if t == self.config.stop_index), len(ot))
        out["order_tok"] = ot[:n]
        out["order_qty"] = labels["order_qty"][index].tolist()[:n]
        out["order_digits"] = labels["order_digits"][index].tolist()[:n]
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path, *, dtype=torch.float32, extra=None):
        blob = {"schema": SCHEMA, "config": self.config.to_dict(),
                "model": {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in self.state_dict().items()},
                "extra": extra or {}}
        torch.save(blob, str(path))

    @classmethod
    def load(cls, path, map_location="cpu", codec=None):
        blob = torch.load(str(path), map_location=map_location, weights_only=True)
        config = blob.get("config") or blob.get("model_config")
        if (config or {}).get("schema") != SCHEMA:
            raise ValueError(f"{path}: not a {SCHEMA} checkpoint (v1: primitive-bc-v1-code)")
        net = cls(ModelConfig.from_dict(config), codec=codec)
        net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in blob["model"].items()})
        return net.eval()

    from_checkpoint = load


def load_policy(path):
    """Training checkpoint (train.py: {'config', 'model', ...}) or PolicyNet.save file -> eval PolicyNet."""
    return PolicyNet.load(path)


# ----------------------------------------------------------------------------- batching / loss
LABEL_KEYS = ("unit_tok", "unit_qty", "unit_digits", "order_tok", "order_qty", "order_digits")
META_KEYS = ("step", "hour", "team_id", "strength_decile")   # the meta fields the model reads


def _pad(values, shape, fill=-1):
    out = np.full(shape, fill, np.int64)
    a = np.asarray(values if values is not None else [], np.int64)
    if a.size:
        out[tuple(slice(0, s) for s in a.shape)] = a
    return out


def collate(examples, *, n_order_slots=None, digit_slots=None, device=None):
    """Per-step example dicts (data.Perspective.chunks layout: rows [N,80], token_mask,
    knownness, unit_pos, feat (feature_tokens.tokens), prev_unit_tok, labels, loss_mask, meta)
    -> batch dict of tensors for PolicyNet.forward/act. Rows are padded to the longest roster."""
    if n_order_slots is None or digit_slots is None:
        m = _codec()
        n_order_slots = n_order_slots or int(getattr(m, "N_ORDER_SLOTS", m.MAX_ORDERS + 1))
        digit_slots = digit_slots or int(getattr(m, "DIGIT_SLOTS", getattr(m, "MAX_DIGITS", 5) + 1))
    B = len(examples)
    n = max(len(ex["rows"]) for ex in examples)
    width = np.asarray(examples[0]["rows"]).shape[-1]
    rows = np.zeros((B, n, width), np.float32)
    known = np.zeros((B, n, width), np.bool_)
    mask = np.zeros((B, n), np.bool_)
    for i, ex in enumerate(examples):
        k = len(ex["rows"])
        rows[i, :k], known[i, :k], mask[i, :k] = ex["rows"], ex["knownness"], ex["token_mask"]
    U = np.asarray(examples[0]["unit_pos"]).shape[0]
    t = partial(torch.as_tensor, device=device)
    unit_pos = np.stack([np.asarray(ex["unit_pos"], np.int64) for ex in examples])
    import unit_index           # the one carrier-row function of the train (collate_steps) and play (agent) paths
    out = {"rows": t(rows), "knownness": t(known), "token_mask": t(mask), "unit_pos": t(unit_pos),
           "carrier_row": t(unit_index.carrier_rows(rows, mask, unit_pos))}
    if all("feat" in ex and ex["feat"] is not None for ex in examples):
        f = [ex["feat"] for ex in examples]
        out["feat"] = {"values": t(np.stack([np.asarray(x["values"], np.float32) for x in f])),
                       "known": t(np.stack([np.asarray(x["known"], np.bool_) for x in f])),
                       "tile": t(np.stack([np.asarray(x["tile"], np.float32) for x in f]))}
    out["prev_unit_tok"] = t(np.stack([_pad(ex.get("prev_unit_tok"), (U,)) for ex in examples]))
    metas = [ex.get("meta") or {} for ex in examples]
    for key in META_KEYS:
        try:
            out[key] = t(np.asarray([int(m[key]) for m in metas], np.int64))
        except (KeyError, TypeError, ValueError):
            pass
    if "step" in out and "hour" not in out:
        out["hour"] = out["step"] % 24
    out["meta"] = metas
    labs = [ex.get("labels", ex) for ex in examples]
    if all("unit_tok" in lab for lab in labs):
        out["labels"] = {
            "unit_tok": t(np.stack([_pad(lab["unit_tok"], (U,)) for lab in labs])),
            "unit_qty": t(np.stack([_pad(lab["unit_qty"], (U,)) for lab in labs])),
            "unit_digits": t(np.stack([_pad(lab.get("unit_digits"), (U, digit_slots)) for lab in labs])),
            "order_tok": t(np.stack([_pad(lab["order_tok"], (n_order_slots,)) for lab in labs])),
            "order_qty": t(np.stack([_pad(lab["order_qty"], (n_order_slots,)) for lab in labs])),
            "order_digits": t(np.stack([_pad(lab.get("order_digits"), (n_order_slots, digit_slots)) for lab in labs]))}
    if all("loss_mask" in ex for ex in examples):
        out["loss_mask"] = t(np.asarray([float(np.asarray(ex["loss_mask"], np.float32).min()) for ex in examples],
                                        np.float32))
    return out


def _ce(logits, target, weight):
    """Sum of weighted CE over positions with target >= 0, and the weighted count."""
    valid = target >= 0
    if not bool(valid.any()):
        z = logits.sum() * 0
        return z.float(), z.float()
    w = weight.expand_as(target)[valid].float()
    ce = F.cross_entropy(logits[valid].float(), target[valid], reduction="none")
    return (ce * w).sum(), w.sum()


def compute_loss(net, out, batch):
    """Per-head CE (sum, count, mean) over present units, unit quantities, unit digits, orders
    incl. STOP/NOOP (the STOP-only last slot excluded), order quantities and order digits,
    weighted by batch['loss_mask'] [B] (default 1). 'loss' = joint NLL of the whole step action
    per weighted step (sum over every head / sum of loss_mask)."""
    c = net.config
    lab = PolicyNet._labels(batch)
    device = out["unit_logits"].device
    B = out["unit_logits"].shape[0]
    lm = batch.get("loss_mask")
    lm = torch.ones(B, device=device) if lm is None else lm.to(device).float()
    ut, uq, ud, ot, oq, od = (lab[k].to(device).long() if lab[k] is not None else None for k in LABEL_KEYS)
    ut = torch.where(out["unit_present"], ut, -1)
    uq = torch.where((ut >= 0) & net.unit_qty_mask[ut.clamp(min=0)], uq, -1)
    ot = ot.clone()
    ot[:, net.O - 1] = -1                               # STOP-only slot: no information
    oq = torch.where((ot >= 0) & net.order_qty_mask[ot.clamp(min=0)], oq, -1)
    res = {}
    w2 = lm[:, None]
    w3 = lm[:, None, None]
    parts = {"unit": (out["unit_logits"], ut, w2), "unit_qty": (out["unit_qty_logits"], uq, w2),
             "order": (out["order_logits"], ot, w2), "order_qty": (out["order_qty_logits"], oq, w2)}
    if ud is not None:
        parts["unit_digit"] = (out["unit_digit_logits"], torch.where((uq == c.qty_exact)[..., None], ud, -1), w3)
    if od is not None:
        parts["order_digit"] = (out["order_digit_logits"], torch.where((oq == c.qty_exact)[..., None], od, -1), w3)
    total = None
    for name, (logits, target, w) in parts.items():
        s, n = _ce(logits, target, w)
        res[name] = {"sum": s, "count": n, "mean": s / n.clamp(min=1)}
        total = s if total is None else total + s
    res["steps"] = lm.sum()
    res["loss"] = total / res["steps"].clamp(min=1)
    return res


def lr_multiplier(update, warmup, total, floor=0.1):
    """Linear warmup to 1 over `warmup` updates, then cosine decay to `floor` at `total`."""
    if update < warmup:
        return (update + 1) / max(1, warmup)
    frac = min(1.0, (update - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * frac))


def make_optimizer(net, lr, *, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8):
    return torch.optim.AdamW(net.param_groups(weight_decay), lr=lr, betas=betas, eps=eps)


# ----------------------------------------------------------------------------- size / latency
def size_report(config=None, codec=None):
    net = PolicyNet(config, codec=codec)
    total = net.parameter_count()
    groups = {"inputs": ("inp", "known_inp", "tok_type", "feat_", "tile_inp", "prev_tok", "prev_slot", "team",
                         "strength", "pos_", "tok_norm"),
              "trunk": ("latent", "carry_in", "carry_mix", "slot_mix", "in_norm", "cross", "blocks", "trunk_norm"),
              "field": ("cell_", "slot_out", "mem_in", "farm_emb", "row_emb", "col_emb", "slot_pos", "mem_norm")}
    out = {"parameters": total}
    for g, prefixes in groups.items():
        out[g + "_parameters"] = sum(p.numel() for n, p in net.named_parameters() if n.startswith(prefixes))
    out["decoder_parameters"] = total - sum(out[g + "_parameters"] for g in groups)
    out["trunk_share"] = round(out["trunk_parameters"] / total, 4)
    out["config"] = {k: v for k, v in net.config.to_dict().items() if k not in ("unit_has_qty", "order_has_qty")}
    return out


def force_units(inputs, units):
    """Copy of a collated B=1 input with `units` present units (latency worst case): missing
    unit positions are filled with distinct cells of the own farm."""
    out = dict(inputs)
    pos = inputs["unit_pos"].clone()
    have = int((pos[0, :, 0] >= 0).sum())
    for i in range(have, min(units, pos.shape[1])):
        pos[0, i, 0], pos[0, i, 1] = i % BOARD, (i // BOARD) % BOARD
    pos[0, units:] = -1
    out["unit_pos"] = pos
    if "carrier_row" in inputs:              # forced units have no carrier row (the MLP still runs on every slot)
        cr = inputs["carrier_row"].clone()
        cr[0, units:] = -1
        out["carrier_row"] = cr
    return out


@torch.no_grad()
def latency(net, observations, *, units=16, orders=11, threads=1, warmup=3, force_exact=False, carry=True):
    """Per-step ms of net.act on real collated B=1 inputs (list), `units` present units forced,
    all `orders` order slots decoded (orders=11 -> force_orders), state carried step to step.
    -> {n, p50, p95, max, mean} (+ settings)."""
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        net = net.eval()
        state = None
        times = []
        seq = [force_units(o, units) for o in observations]
        for i, inp in enumerate(list(seq[:warmup]) + seq):
            t0 = time.perf_counter()
            labels, new = net.act(inp, state if carry else None, force_orders=orders >= net.O, force_exact=force_exact)
            dt = time.perf_counter() - t0
            state = new
            if i >= warmup:
                times.append(dt * 1e3)
        a = np.asarray(times)
        return {"n": int(a.size), "units": units, "orders": orders, "threads": threads, "force_exact": force_exact,
                "p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
                "max": round(float(a.max()), 2), "mean": round(float(a.mean()), 2)}
    finally:
        torch.set_num_threads(old)


# ----------------------------------------------------------------------------- real observations (bench)
def canonical_observation(actor_obs):
    """data.canonical_observation when present, else the same rule locally (inventories by key)."""
    try:
        import data
        fn = getattr(data, "canonical_observation", None)
    except Exception:  # noqa: BLE001 - data.py may be mid-edit by another build
        fn = None
    if fn is not None:
        return fn(actor_obs)
    import copy
    out = copy.deepcopy(actor_obs)
    inv = out.get("private", {}).get("inventories")
    if isinstance(inv, dict):
        out["private"]["inventories"] = {k: inv[k] for k in sorted(inv)}
    return out


def real_examples(replay, seat, steps, configuration=None):
    """Real per-step model inputs for one seat of a replay (agent path: canonical observation ->
    observation.pack -> FeatureState -> feature_tokens), with the observation/features time
    (ms) of each step. -> (examples, prep_ms, observations)."""
    import feature_tokens as TK
    import features as FT
    import codec
    from farm_neat.policy import observation as OBS
    from farm_neat.training import bc
    cfg = configuration or replay["configuration"]
    pack_cfg = {k: cfg[k] for k in OBS.CONFIG if k in cfg}
    if "marketParams" in cfg:
        pack_cfg["marketParams"] = {k: dict(v) for k, v in cfg["marketParams"].items()}
    fs = FT.FeatureState(cfg)
    FT.warm(cfg)
    examples, prep, raw = [], [], []
    want = set(steps)
    for t in range(max(want) + 1):
        obs = bc.public_observation(replay, t, seat)
        t0 = time.perf_counter()
        obs = canonical_observation(obs)
        packed = OBS.pack(obs, None, pack_cfg)
        feats = TK.tokens(fs.update(obs).compute())
        pos = np.full((codec.MAX_UNITS, 2), -1, np.int64)
        up = codec.unit_positions(obs)[:codec.MAX_UNITS]
        pos[:len(up)] = np.asarray(up, np.int64)
        ex = {"rows": packed["rows"].astype(np.float16), "token_mask": packed["token_mask"],
              "knownness": packed["knownness"], "unit_pos": pos, "feat": feats,
              "meta": {"step": t, "hour": t % 24, "day": t // 24}}
        dt = (time.perf_counter() - t0) * 1e3
        action = replay["steps"][t + 1][seat].get("action")
        if action is not None:
            fs.record_action(action)
        if t in want:
            examples.append(ex)
            prep.append(dt)
            raw.append(obs)
    return examples, prep, raw


def _stats(ms):
    a = np.asarray(ms, np.float64)
    return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "max": round(float(a.max()), 2), "mean": round(float(a.mean()), 2)}


@torch.no_grad()
def bench_step(net, replay, seat, *, stride=1, last=718, units=16, orders=11, force_exact=False):
    """Whole agent step on one thread, every `stride`-th step of one seat of a real replay:
    prep (canonical observation, observation.pack, FeatureState update/compute, feature_tokens,
    collate), model (act with carried state, `units` present units forced, all order slots when
    orders >= n_order_slots), post (codec labels, codec.decode_step, record_action). Features
    run on every step (history is causal); untimed steps only update them. The observation
    list is built first and frozen out of the collector; gc is disabled inside each step."""
    import gc
    import codec
    import feature_tokens as TK
    import features as FT
    from farm_neat.policy import observation as OBS
    from farm_neat.training import bc
    cfg = replay["configuration"]
    pack_cfg = {k: cfg[k] for k in OBS.CONFIG if k in cfg}
    if "marketParams" in cfg:
        pack_cfg["marketParams"] = {k: dict(v) for k, v in cfg["marketParams"].items()}
    observations = [bc.public_observation(replay, t, seat) for t in range(last + 1)]
    actions = [replay["steps"][t + 1][seat].get("action") for t in range(last + 1)]
    FT.warm(cfg)
    fs = FT.FeatureState(cfg)
    net = net.eval()
    gc.collect()
    gc.freeze()
    parts = {"prep": [], "model": [], "post": [], "whole": []}
    state = None
    hands = []
    try:
        for t in range(last + 1):
            timed = t % stride == 0
            gc.disable()
            t0 = time.perf_counter()
            obs = canonical_observation(observations[t])
            feats = fs.update(obs).compute()
            if timed:
                packed = OBS.pack(obs, None, pack_cfg)
                pos = np.full((codec.MAX_UNITS, 2), -1, np.int64)
                up = codec.unit_positions(obs)[:codec.MAX_UNITS]
                pos[:len(up)] = np.asarray(up, np.int64)
                ex = {"rows": packed["rows"].astype(np.float16), "token_mask": packed["token_mask"],
                      "knownness": packed["knownness"], "unit_pos": pos, "feat": TK.tokens(feats),
                      "meta": {"step": t, "hour": t % 24, "day": t // 24}}
                inputs = force_units(collate([ex]), units)
                t1 = time.perf_counter()
                labels, state = net.act(inputs, state, force_orders=orders >= net.O, force_exact=force_exact)
                t2 = time.perf_counter()
                codec.decode_step(net.codec_labels(labels), obs)
                hands.append(len(up) - 1)
            if actions[t] is not None:
                fs.record_action(actions[t])
            t3 = time.perf_counter()
            gc.enable()
            if timed:
                for k, v in (("prep", t1 - t0), ("model", t2 - t1), ("post", t3 - t2), ("whole", t3 - t0)):
                    parts[k].append(v * 1e3)
    finally:
        gc.enable()
        gc.unfreeze()
    out = {k: _stats(v) for k, v in parts.items()}
    out.update(units=units, orders=orders, force_exact=force_exact, seat=seat, stride=stride,
               real_hands_max=max(hands), threads=torch.get_num_threads())
    return out


def _replay(path=None):
    import glob
    import zipfile
    paths = [path] if path else sorted(glob.glob("/mnt/steam/farm-data/kaggle/kaggriculture-episodes-2026-09-2*.zip")
                                       + glob.glob("/mnt/steam/farm-data/kaggle/kaggriculture-episodes-2026-09-2*/*.zip"))
    with zipfile.ZipFile(paths[-1]) as z:
        name = next(n for n in sorted(z.namelist()) if n.endswith(".json"))
        return json.loads(z.read(name)), f"{paths[-1]}::{name}"


def _cpu_info():
    info = {"loadavg": open("/proc/loadavg").read().split()[:3]}
    try:
        info["cpu_max"] = Path("/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip().split("::")[-1]
                               + "/cpu.max").read_text().strip()
    except OSError:
        info["cpu_max"] = None
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("size")
    s.add_argument("--latents", type=int, nargs="*", default=[32, 64])
    b = sub.add_parser("bench", help="whole-step latency on a real replay, random weights, one thread")
    b.add_argument("--replay", default=None, help="archive zip (first episode) or episode JSON")
    b.add_argument("--seat", type=int, default=0)
    b.add_argument("--stride", type=int, default=2)
    b.add_argument("--last", type=int, default=718)
    b.add_argument("--latents", type=int, nargs="*", default=[32, 64])
    b.add_argument("--int8", action="store_true", help="also torch dynamic int8 (nn.Linear)")
    b.add_argument("--exact", action="store_true", help="also the force_exact worst case")
    b.add_argument("--think", default="field")
    b.add_argument("--out", default=None)
    c = sub.add_parser("cold", help="fresh-process cold load: import torch + model, load, first act")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--replay", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    if a.cmd == "size":
        for L in a.latents:
            r = size_report(ModelConfig(trunk_latents=L))
            r.pop("config")
            print(json.dumps({"trunk_latents": L, **r}), flush=True)
        return
    if a.replay and a.replay.endswith(".json"):
        replay, src = json.loads(Path(a.replay).read_text()), a.replay
    else:
        replay, src = _replay(a.replay)
    if a.cmd == "cold":
        t0 = time.perf_counter()
        net = load_policy(a.checkpoint)
        t1 = time.perf_counter()
        r = bench_step(net, replay, 0, stride=1, last=0)
        print(json.dumps({"load_s": round(t1 - t0, 3), "first_step_ms": r["whole"]["max"], **_cpu_info()}))
        return
    import copy
    import warnings
    results = []
    print(json.dumps({"source": src, "seat": a.seat, "stride": a.stride, **_cpu_info()}), flush=True)
    for L in a.latents:
        torch.manual_seed(0)
        t0 = time.perf_counter()
        net = PolicyNet(ModelConfig(trunk_latents=L, think=a.think)).eval()
        build_s = time.perf_counter() - t0
        for head in (net.unit_head, net.order_head, net.unit_qty.head, net.order_qty.head, net.digits.head):
            nn.init.normal_(head.weight, std=0.5)      # random decisions (zero heads always pick index 0)
        variants = [("fp32", net)]
        if a.int8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(net), {nn.Linear}, dtype=torch.qint8)
            variants.append(("int8dyn", q))
        for name, m in variants:
            for exact in ((False, True) if a.exact else (False,)):
                r = bench_step(m, replay, a.seat, stride=a.stride, last=a.last, force_exact=exact)
                r.update(trunk_latents=L, think=a.think, variant=name, params=net.parameter_count(),
                         build_s=round(build_s, 2), **_cpu_info())
                results.append(r)
                print(json.dumps(r), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"source": src, "results": results}, indent=1))


if __name__ == "__main__":
    main()
