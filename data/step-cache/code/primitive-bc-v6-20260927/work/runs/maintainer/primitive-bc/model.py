"""Primitive-BC policy: packed actor observation -> thinking field -> per-step joint action.

Computation (think='bottleneck', the default):

1. Encoder: pre-LN transformer over the packed rows of observation.pack (masked
   rows + knownness), plus G global thought-slot tokens.  Every TILE token (one per
   cell of both 10x10 farms, located from its own kind/role/x/y columns) and every
   slot token receives the embedding of the code written there at the previous step
   (a learned 'none' vector at episode start).
2. Writer: C codes from a K-code vocabulary per cell (200) and per global slot (G),
   straight-through Gumbel-softmax while training, sampled/argmax at inference.
   No labels or auxiliary targets touch the codes.
3. Decoder (the only reader of the codes): memory = code embeddings + fixed farm/row/
   column (cells) or slot (globals) embeddings.  The encoder hidden state never
   reaches the decoder or the value head.  Decisions are sequential: farmer, each
   hand in observation order, then market orders ending with STOP.  Each decision
   query = slot identity (+ unit x/y and the codes of the unit's own and 4
   neighbouring cells) + the embedding of the previous decision (token + quantity);
   causal self-attention sees all earlier decisions; cross-attention reads the field.
   Quantity bins are predicted after the token, conditioned on it.
4. The codes are carried to the next step; gradients flow through them (and across
   steps inside a training window) by the straight-through estimator.

think='off' is the ablation: the decoder cross-attends to the encoder states
directly and nothing is carried.

Labels follow codec.encode_step: unit_tok/unit_qty [MAX_UNITS] (-1 absent / no qty),
order_tok/order_qty [<=MAX_ORDERS+1] ending with STOP (padded here with -1).
"""
from __future__ import annotations

import dataclasses
from os import environ as _os_env
import math
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

SCHEMA = "primitive-bc-policy-v1"
IN_WIDTH = 80            # observation.WIDTH
TILE_KIND_COL = 2        # observation.KINDS.index("TILE")
ROLE_COL = 12            # 12 = own farm, 13 = opponent farm
XY_COL = 14              # x/9, y/9
CARRIER_KIND = 3         # observation.KINDS.index("CARRIER")
# Fixed-identity state entities that anchor global slots (GLOBAL, FARM, MARKET, STOCK, SHOP,
# CONTROL); CONSTANT rows are static mechanics and are not anchored.
ANCHOR_KINDS = (0, 1, 4, 5, 6, 10)
BOARD = 10
N_CELLS = 2 * BOARD * BOARD
# Input validity checks below need a GPU->host sync each; the trainer turns them off after its first updates (CUDA only).
CHECK_INPUTS = True
CODE_MODES = ("gumbel", "sample", "argmax")
# reachability buckets: ETA = Manhattan + 1 in 1..19 -> 9 buckets; slack = deadline - ETA -> 0 none, 1 late, 2.. early
ETA_BUCKET = (0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8)
N_ETA = 9
SLACK_BUCKET = (1,) + (2, 3, 4, 4, 5, 5, 5, 5) + (6,) * 8 + (7,) * 16 + (8,) * 9   # index = clamp(slack, -1, 40) + 1
N_SLACK = 9


def _codec():
    import codec  # primitive-bc/codec.py (lazy: tests may inject a stub module)
    return codec


@dataclass
class ModelConfig:
    # vocabulary / interface (None -> filled from codec.py by resolve())
    n_unit_tokens: Optional[int] = None
    n_order_tokens: Optional[int] = None
    max_units: Optional[int] = None
    max_orders: Optional[int] = None
    n_qty: Optional[int] = None
    stop_index: Optional[int] = None
    unit_has_qty: Optional[list] = None
    order_has_qty: Optional[list] = None
    # encoder
    width: int = 192
    layers: int = 6
    heads: int = 6
    ff_mult: int = 4
    ffn: str = "gelu"          # gelu MLP (w -> ff_mult*w -> w) | swiglu (w -> 2h -> h -> w, h = ff_mult*w*2/3 rounded to 32)
    swiglu_branch: int = 0     # v5w (2026-09-27): hidden width of a parallel SwiGLU branch beside each block MLP (0 = none);
                               # grown inert: its output projection sg2 starts at zero, so the grown net equals its parent
    # decoder
    dec_width: int = 0          # 0 -> width
    dec_layers: int = 2
    dec_heads: int = 0          # 0 -> heads
    # thinking field
    think: str = "bottleneck"   # or "off"
    G: int = 64
    K: int = 64
    C: int = 4
    local: bool = True          # unit query reads own + 4 neighbour cells of the field
    global_read: bool = True    # every decision query also reads all G slot codes directly
    global_dim: int = 32        # per-slot width of that direct read
    slot_anchor: str = "entities"  # entities: slot g is written from the g-th state entity token,
                                   # then own carriers, then free tokens; free: all G free tokens
    infer_code_mode: str = "sample"  # code choice at inference: sample | argmax
    # growth (2026-09-26, flip.py): zero-initialised inputs; 0 / False reproduces the pre-growth network exactly
    extra_width: int = 0        # flip.LIN linear columns of the per-token extra array
    unit_extra_width: int = 0   # flip.UNIT_WIDTH per-unit summary columns read by each unit query
    calc_width: int = 0         # v4: data.CALC_WIDTH calc_features columns after the kind/deadline pair (MARKET/GLOBAL tokens)
    reach_bias: bool = False    # decoder unit-query x own-TILE attention bias from (ETA, job kind, slack)
    carrier_read: bool = False  # each unit query adds a zero-initialised projection of its own encoded CARRIER
    reach_job: bool = False     # v2: second decoder bias from flip.reach_matrix (wheat-aware FEED ETA, unreachable bucket)
    noop_unit_mask: bool = False  # inference-only; no new weights; enable only after paired evaluation
    prefix_unit_width: int = 0  # v3: prefix.F_UP same-step state rows added to each unit query (teacher / sampled prefix)
    prefix_order_width: int = 0 # v3: prefix.F_OP order-stock interval rows added to each order query
    prefix_cash_width: int = 0  # v6: separately ablatable, inert-at-graft market-prefix cash ladder
    input_schema: str = "pbc-job-input-v5-legacy"  # actor-input contract; action schema below is unchanged
    feature_schema: str = "legacy"  # v7 opts into corrected finite-horizon mechanics; old checkpoints remain legacy
    job_eta_v6: bool = False   # pbc-job-input-v6.0 finite multi-job ETA/mask residuals, off by default
    v6_prev: bool = False       # v6 (2026-09-27): the previous two own turns, in order, as input (pbc-v6-design/DESIGN.md)
    v6_aux: bool = False        # v6: training-only heads for own t+1/t+2 and opponent t/t+1/t+2 worker commands
    v6_aux_turn: bool = False   # training-only heads (with v6_aux): worker quantities and whole order sequences, own t+1/t+2 and opponent t/t+1/t+2
    dropout: float = 0.0        # residual-branch dropout in every encoder/decoder layer (training only)
    feature_dropout: float = 0.0  # per example, drop each auxiliary input group's whole contribution (inverted scaling)
    x8_token_width: int = 0     # v8 feature families (pbc-features-v8): per-token columns, one zero-init Linear into the stream
    x8_unit_width: int = 0      # v8: per-unit columns read by each unit query
    lattice_channels: int = 0   # v8: owner's space-time lattice (2 farms x 10 x 10 x C) through a small CNN into TILE tokens
    lattice_width: int = 64
    recur_core: str = ""        # "a:b": encoder blocks a..b-1 are a shared core run recur_loops times, the prelude output
                                # re-injected each loop (Parcae-style; owner 2026-09-28); loops after the first enter through
                                # tanh(grow_recur_gate), zero at graft, so the grafted net equals its parent exactly
    recur_loops: int = 1        # core loops at inference (1 = the plain encoder)
    recur_train_max: int = 0    # > 1: each training update samples the loop count uniformly from 1..recur_train_max
    recur_mode: str = "gate"    # "gate": x = x + tanh(g) * (core(x + e) - x) (inert graft); "norm": x = core(LN(x) + e), the
                                # re-injected input keeps a fixed scale at any depth (from-scratch screens; PRO-RECUR-0928 #2)
    n_kinds: int = 12           # flip.N_KINDS
    schema: str = SCHEMA

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def resolve(self, codec=None):
        """Fill the vocabulary fields from codec.py; returns a new config."""
        c = dataclasses.replace(self)
        if None in (c.n_unit_tokens, c.n_order_tokens, c.max_units, c.max_orders, c.n_qty,
                    c.stop_index, c.unit_has_qty, c.order_has_qty):
            codec = codec or _codec()
            c.n_unit_tokens = c.n_unit_tokens or len(codec.UNIT_VOCAB)
            c.n_order_tokens = c.n_order_tokens or len(codec.ORDER_VOCAB)
            c.max_units = c.max_units or int(codec.MAX_UNITS)
            c.max_orders = c.max_orders or int(codec.MAX_ORDERS)
            c.n_qty = c.n_qty or int(getattr(codec, "QTY_BIN_COUNT", 75))
            if c.stop_index is None:
                c.stop_index = int(list(codec.ORDER_VOCAB).index("STOP"))
            if c.unit_has_qty is None:
                c.unit_has_qty = [bool(v) for v in codec.UNIT_HAS_QTY]
            if c.order_has_qty is None:
                c.order_has_qty = [bool(v) for v in codec.ORDER_HAS_QTY]
        c.unit_has_qty = [bool(v) for v in c.unit_has_qty]
        c.order_has_qty = [bool(v) for v in c.order_has_qty]
        if len(c.unit_has_qty) != c.n_unit_tokens or len(c.order_has_qty) != c.n_order_tokens:
            raise ValueError("quantity masks must match the vocabularies")
        if c.width <= 0 or c.heads <= 0 or c.width % c.heads:
            raise ValueError("encoder width must be positive and divisible by positive heads")
        dec_width, dec_heads = c.dec_width or c.width, c.dec_heads or c.heads
        if dec_width <= 0 or dec_heads <= 0 or dec_width % dec_heads:
            raise ValueError("decoder width must be positive and divisible by positive dec_heads")
        if getattr(c, "ffn", "gelu") not in ("gelu", "swiglu"):
            raise ValueError("ffn must be gelu|swiglu")
        if c.think not in ("bottleneck", "off") or c.infer_code_mode not in ("sample", "argmax"):
            raise ValueError("think must be bottleneck|off; infer_code_mode sample|argmax")
        if c.think == "bottleneck" and (getattr(c, "v6_prev", False) or getattr(c, "v6_aux", False)):
            raise ValueError("v6 history / aux heads need think='off' (the bottleneck encode returns before they are built)")
        if c.slot_anchor not in ("entities", "free"):
            raise ValueError("slot_anchor must be entities|free")
        if c.input_schema not in ("pbc-job-input-v5-legacy", "pbc-job-input-v6.0"):
            raise ValueError("unknown actor input schema")
        if c.feature_schema not in ("legacy", "v7"):
            raise ValueError("feature_schema must be legacy|v7")
        if not 0.0 <= float(c.feature_dropout) < 1.0:
            raise ValueError("feature_dropout must be finite and in [0, 1)")
        if c.job_eta_v6 != (c.input_schema == "pbc-job-input-v6.0"):
            raise ValueError("job_eta_v6 must match input_schema")
        return c


def default_config():
    return ModelConfig()


SIZES = {  # name -> encoder/decoder shape (vocabulary from codec)
    "1m": dict(width=96, layers=4, heads=4, dec_layers=2),
    "4m": dict(width=192, layers=6, heads=6, dec_layers=2),
    "12m": dict(width=288, layers=10, heads=8, dec_layers=2),
}


# ----------------------------------------------------------------------------- field
@dataclass
class FieldState:
    """Carried thinking field for a batch: one-hot codes and an episode-start flag."""
    codes: torch.Tensor   # [B, S, C, K] float one-hot (straight-through while training)
    fresh: torch.Tensor   # [B] bool: True -> 'none' code (nothing written yet)

    @staticmethod
    def none(batch, S, C, K, device=None, dtype=torch.float32):
        return FieldState(torch.zeros(batch, S, C, K, device=device, dtype=dtype),
                          torch.ones(batch, dtype=torch.bool, device=device))

    def update(self, codes, active):
        """Write new codes for active examples; inactive ones keep their field."""
        a = active.to(codes.device)
        return FieldState(torch.where(a[:, None, None, None], codes, self.codes.to(codes.dtype)),
                          self.fresh & ~a)

    def detach(self):
        return FieldState(self.codes.detach(), self.fresh)

    def indices(self):
        return self.codes.argmax(-1)


@dataclass
class ThinkState:
    """Per (episode, seat) inference state returned by act(). hist (v6): this model's own last two decisions as
    (step, unit_tok[U], unit_qty[U], order_tok, order_qty) canonical codec labels of the returned wire action,
    before any external play-time rule edits it."""
    field: Optional[FieldState] = None
    last_step: int = -1
    hist: tuple = ()


def v6_history_arrays(hist, step, max_units, n_orders):
    """prev_unit [1, 2, U, 2] and prev_order [1, 2, O, 2] (codec token, quantity; -1 = none) for decision `step` from the
    recorded (step, unit_tok, unit_qty, order_tok, order_qty) entries: the same rule as data.add_v6 (lag 1 then 2, the
    farmer across days, a hand only within its day, nothing for a step that has no entry)."""
    pu = torch.full((1, 2, max_units, 2), -1, dtype=torch.long)
    po = torch.full((1, 2, n_orders, 2), -1, dtype=torch.long)
    by_step = {int(e[0]): e for e in hist}
    for li, lag in enumerate((1, 2)):
        e = by_step.get(step - lag)
        if e is None:
            continue
        s, ut, uq, ot, oq = e
        for k in range(min(max_units, len(ut))):
            if k == 0 or s // 24 == step // 24:
                pu[0, li, k, 0], pu[0, li, k, 1] = int(ut[k]), int(uq[k])
        n = min(n_orders, len(ot))
        po[0, li, :n, 0] = torch.as_tensor([int(x) for x in ot[:n]])
        po[0, li, :n, 1] = torch.as_tensor([int(x) for x in oq[:n]])
    return pu, po


# ----------------------------------------------------------------------------- layers
def _hist_p(net):
    """Owner 2026-09-28 (copycat / causal-confusion guard): own-history memory and previous-turn embeddings get their own,
    higher group-dropout rate PBC_HIST_DROPOUT (unset = the normal feature_dropout, bit-identical)."""
    v = _os_env.get("PBC_HIST_DROPOUT", "")
    return float(v) if v else None


def _sdpa(q, k, v, mask):
    dh = q.shape[-1]
    if q.is_cuda and dh % 8:
        # CUDA fused kernels need head dim % 8 == 0; else the math kernel materializes B*H*N*N (5-7x slower, 6x memory at dh 36).
        # Zero-padding q/k/v keeps q.k and the kept output columns exact; the scale stays 1/sqrt(dh).
        p = 8 - dh % 8
        q, k, v = (F.pad(t, (0, p)) for t in (q, k, v))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=dh ** -0.5)[..., :dh]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def _ffn_layers(w, ff_mult, ffn):
    """(ff1, ff2) of the block MLP. swiglu: ff1 emits [gate, value] (2h), ff2 maps h back; h = ff_mult*w*2/3 rounded to 32
    keeps the parameter count of the gelu MLP."""
    if ffn == "swiglu":
        h = max(32, int(round(ff_mult * w * 2 / 3 / 32)) * 32)
        return nn.Linear(w, 2 * h), nn.Linear(h, w)
    return nn.Linear(w, ff_mult * w), nn.Linear(ff_mult * w, w)


def _ffn(ff1, ff2, x, ffn):
    if ffn == "swiglu":
        g, v = ff1(x).chunk(2, dim=-1)
        return ff2(F.silu(g) * v)
    return ff2(F.gelu(ff1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, w, heads, ff_mult, ffn="gelu", sg=0, dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.h, self.dh, self.ffn = heads, w // heads, ffn
        self.ln1, self.qkv, self.proj = nn.LayerNorm(w), nn.Linear(w, 3 * w), nn.Linear(w, w)
        self.ln2 = nn.LayerNorm(w)
        self.ff1, self.ff2 = _ffn_layers(w, ff_mult, ffn)
        self.sg = sg
        if sg:
            self.sg1, self.sg2 = nn.Linear(w, 2 * sg), nn.Linear(sg, w)

    def forward(self, x, mask):
        B, N, w = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        x = x + self.drop(self.proj(_sdpa(q, k, v, mask).transpose(1, 2).reshape(B, N, w)))
        y = self.ln2(x)
        out = x + self.drop(_ffn(self.ff1, self.ff2, y, self.ffn))
        return out + self.drop(_ffn(self.sg1, self.sg2, y, "swiglu")) if self.sg else out


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, ff_mult, ffn="gelu", sg=0, dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.h, self.dh, self.ffn = heads, d // heads, ffn
        self.ln1, self.qkv, self.o1 = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ln2, self.q2, self.kv2, self.o2 = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.ln3 = nn.LayerNorm(d)
        self.ff1, self.ff2 = _ffn_layers(d, ff_mult, ffn)
        self.sg = sg
        if sg:
            self.sg1, self.sg2 = nn.Linear(d, 2 * sg), nn.Linear(sg, d)

    def memory(self, mem):
        B, S, d = mem.shape
        k, v = self.kv2(mem).view(B, S, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x, self_mask, mem_kv, mem_mask, cache=None, mem_bias=None):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat((cache["k"], k), 2), torch.cat((cache["v"], v), 2)
            cache["k"], cache["v"] = k, v
        x = x + self.drop(self.o1(_sdpa(q, k, v, self_mask).transpose(1, 2).reshape(B, T, d)))
        q2 = self.q2(self.ln2(x)).view(B, T, self.h, self.dh).transpose(1, 2)
        cross = mem_mask
        if mem_bias is not None:      # [B, h, T, S] additive logits; masked memory stays excluded
            cross = mem_bias.to(q2.dtype)
            if mem_mask is not None:
                cross = cross.masked_fill(~mem_mask, float("-inf"))
        x = x + self.drop(self.o2(_sdpa(q2, mem_kv[0], mem_kv[1], cross).transpose(1, 2).reshape(B, T, d)))
        y = self.ln3(x)
        out = x + self.drop(_ffn(self.ff1, self.ff2, y, self.ffn))
        return out + self.drop(_ffn(self.sg1, self.sg2, y, "swiglu")) if self.sg else out

    def forward_lags(self, x, lags, self_mask, mem_kv, mem_mask):
        """forward() for x [B*lags, T, d] whose `lags` consecutive rows share one example's memory: mem_kv = memory(mem)
        of mem [B, S, d] and mem_mask [B, 1, 1, S] (or None), NOT expanded per lag. Self-attention stays per row; the
        cross-attention folds the lags into the query axis ([B, lags*T] queries), so each query sees exactly the keys,
        values and mask of forward() on the expanded memory. Every dropout acts on a tensor of forward()'s shape, so the
        RNG stream is consumed identically."""
        BL, T, d = x.shape
        B = BL // lags
        q, k, v = self.qkv(self.ln1(x)).view(BL, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        x = x + self.drop(self.o1(_sdpa(q, k, v, self_mask).transpose(1, 2).reshape(BL, T, d)))
        q2 = self.q2(self.ln2(x)).view(B, lags * T, self.h, self.dh).transpose(1, 2)
        x = x + self.drop(self.o2(_sdpa(q2, mem_kv[0], mem_kv[1], mem_mask).transpose(1, 2).reshape(BL, T, d)))
        y = self.ln3(x)
        out = x + self.drop(_ffn(self.ff1, self.ff2, y, self.ffn))
        return out + self.drop(_ffn(self.sg1, self.sg2, y, "swiglu")) if self.sg else out


def carrier_indices(rows, token_mask, knownness, max_units):
    """[B, max_units] token index of each own CARRIER by its known ordinal (observation column 16 = ordinal/100;
    exact in float16 for the 25 unit slots), -1 when absent (from capacity-use/growth.py)."""
    own = token_mask.bool() & (rows[..., CARRIER_KIND] > .5) & (rows[..., ROLE_COL] > .5)
    if rows.is_cuda and not CHECK_INPUTS:   # sync-free: same result on inputs that pass the checks below
        B, N = own.shape
        col = torch.where(own, (rows[..., 16].float() * 100).round().long().clamp(0, max_units - 1), max_units)
        result = torch.full((B, max_units + 1), -1, dtype=torch.long, device=rows.device)
        result.scatter_(1, col, torch.arange(N, device=rows.device).expand(B, N).contiguous())
        return result[:, :max_units]
    required = knownness[..., CARRIER_KIND] & knownness[..., ROLE_COL] & knownness[..., 16]
    if bool((own & ~required).any()):
        raise ValueError("own carrier kind, role and ordinal must be known")
    b, r = own.nonzero(as_tuple=True)
    value = rows[b, r, 16].float() * 100
    ids = value.round().long()
    if bool(((value - ids).abs() > .02).any()) or bool(((ids < 0) | (ids >= max_units)).any()):
        raise ValueError("carrier ordinal out of range or not integer")
    keys = b * max_units + ids
    if len(keys.unique()) != len(keys):
        raise ValueError("duplicate own carrier ordinal")
    result = torch.full((rows.shape[0], max_units), -1, dtype=torch.long, device=rows.device)
    result[b, ids] = r
    return result


def opp_carrier_indices(rows, token_mask, knownness, max_units):
    """[B, max_units] token index of each opponent CARRIER by its ordinal, -1 when absent or unknown (v6 training heads)."""
    opp = token_mask.bool() & (rows[..., CARRIER_KIND] > .5) & (rows[..., ROLE_COL + 1] > .5) & knownness[..., 16].bool()
    if rows.is_cuda and not CHECK_INPUTS:
        B, N = opp.shape
        ids = (rows[..., 16].float() * 100).round().long()
        col = torch.where(opp & (ids >= 0) & (ids < max_units), ids.clamp(0, max_units - 1), max_units)
        result = torch.full((B, max_units + 1), -1, dtype=torch.long, device=rows.device)
        result.scatter_(1, col, torch.arange(N, device=rows.device).expand(B, N).contiguous())
        return result[:, :max_units]
    b, r = opp.nonzero(as_tuple=True)
    ids = (rows[b, r, 16].float() * 100).round().long()
    ok = (ids >= 0) & (ids < max_units)
    result = torch.full((rows.shape[0], max_units), -1, dtype=torch.long, device=rows.device)
    result[b[ok], ids[ok]] = r[ok]
    return result


def _mlp(d, out):
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, out))


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


@dataclass
class Context:
    mem: torch.Tensor                 # [B, S, d] decoder memory
    mem_mask: Optional[torch.Tensor]  # [B, 1, 1, S] bool or None
    cells: torch.Tensor               # [B, 100, d] own-farm cell memory (local reads)
    slots: Optional[torch.Tensor]     # [B, G, d] global-slot memory (direct global read)
    value: torch.Tensor               # [B]
    field: Optional[FieldState]       # codes written this step (bottleneck)
    code_logits: Optional[torch.Tensor]  # [B, S, C, K] float32
    unit_extra: Optional[torch.Tensor] = None   # [B, U, unit_extra_width] (growth)
    carriers: Optional[torch.Tensor] = None     # [B, U, d] each unit's own encoded CARRIER, 0 if absent (growth)
    reach_job: Optional[torch.Tensor] = None    # [B, U, S] bucket of the unit's resource-aware ETA to each own TILE
    unit_prefix: Optional[torch.Tensor] = None  # [B, U, F_UP] teacher-forced same-step prefix state (v3)
    order_prefix: Optional[torch.Tensor] = None # [B, O, F_OP] teacher-forced order-stock intervals (v3)
    order_cash: Optional[torch.Tensor] = None   # [B, O, F_CASH] actor-visible cash ladder (v6)
    job_v6: Optional[tuple] = None               # (job bits, resource ETA/mask, deadlines/mask), own farm only
    reach: Optional[tuple] = None               # (own_tile [B,S] bool, x, y, kind, deadline [B,S] long) (growth)
    hist: Optional[tuple] = None                # v6: (tokens [B, H, d], valid [B, H]) previous two turns in order
    prev_unit_emb: Optional[torch.Tensor] = None   # v6: [B, U, d] each worker's own previous commands
    opp_carriers: Optional[torch.Tensor] = None    # v6: [B, U, d] opponent CARRIER encodings (training-only heads)
    x8_unit: Optional[torch.Tensor] = None         # v8: [B, U, x8_unit_width]


# ----------------------------------------------------------------------------- policy
class PolicyNet(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None, codec=None):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        c = self.config = (config or ModelConfig()).resolve(codec)
        self._codec_module = codec
        w = c.width
        d = self.d = c.dec_width or w
        dh = c.dec_heads or c.heads
        self.U, self.O = c.max_units, c.max_orders + 1
        self.S = N_CELLS + c.G if c.think == "bottleneck" else None
        # encoder
        self.inp = nn.Linear(IN_WIDTH, w)
        self.known_inp = nn.Linear(IN_WIDTH, w, bias=False)
        self.blocks = nn.ModuleList(EncoderBlock(w, c.heads, c.ff_mult, getattr(c, "ffn", "gelu"), getattr(c, "swiglu_branch", 0), getattr(c, "dropout", 0.0) or 0.0) for _ in range(c.layers))
        self.enc_norm = nn.LayerNorm(w)
        self._recur, self.recur_T = None, None
        if getattr(c, "recur_core", ""):
            a, b = (int(v) for v in c.recur_core.split(":"))
            if not (0 <= a < b <= c.layers) or c.think != "off" or int(getattr(c, "recur_loops", 1)) < 1:
                raise ValueError("recur_core needs 0 <= a < b <= layers, think='off' and recur_loops >= 1")
            self._recur = (a, b)
            if getattr(c, "recur_mode", "gate") not in ("gate", "norm"):
                raise ValueError("recur_mode must be gate or norm")
            if getattr(c, "recur_mode", "gate") == "norm":
                self.recur_norm = nn.LayerNorm(w)
            else:
                self.grow_recur_gate = nn.Parameter(torch.zeros(w))
        if c.think == "bottleneck":
            self.slot_in = nn.Parameter(torch.randn(c.G, w) * 0.02)
            self.carry_emb = nn.Parameter(torch.randn(c.C * c.K, w) * (0.5 / math.sqrt(c.C)))
            self.none_cell = nn.Parameter(torch.zeros(w))
            self.none_slot = nn.Parameter(torch.zeros(w))
            self.write = nn.Linear(w, c.C * c.K)
            self.code_emb = nn.Parameter(torch.randn(c.C * c.K, d) / math.sqrt(c.C))   # sum of C codes ~ unit scale
            self.farm_emb, self.row_emb, self.col_emb = nn.Embedding(2, d), nn.Embedding(BOARD, d), nn.Embedding(BOARD, d)
            self.slot_emb = nn.Embedding(c.G, d)
            for e in (self.farm_emb, self.row_emb, self.col_emb, self.slot_emb):
                nn.init.normal_(e.weight, std=0.3)
            self.mem_norm = nn.LayerNorm(d)
            cell = torch.arange(N_CELLS)
            self.register_buffer("cell_role", cell // 100, persistent=False)
            self.register_buffer("cell_y", (cell % 100) // BOARD, persistent=False)
            self.register_buffer("cell_x", cell % BOARD, persistent=False)
        else:
            self.mem_proj = nn.Linear(w, d) if d != w else nn.Identity()
        # decoder
        self.query = nn.Embedding(self.U + self.O, d)
        self.unit_row, self.unit_col = nn.Embedding(BOARD, d), nn.Embedding(BOARD, d)
        if c.local:
            self.local_proj = nn.Linear(5 * d, d)
        if c.global_read:
            self.global_down = nn.Linear(d, c.global_dim)
            self.global_proj = nn.Linear(c.G * c.global_dim, d)
        self.bos = nn.Parameter(torch.zeros(d))
        self.unit_tok_emb = nn.Embedding(c.n_unit_tokens, d)
        self.order_tok_emb = nn.Embedding(c.n_order_tokens, d)
        self.qty_emb = nn.Embedding(c.n_qty + 1, d)      # index n_qty = no quantity
        self.dec = nn.ModuleList(DecoderLayer(d, dh, c.ff_mult, getattr(c, "ffn", "gelu"), getattr(c, "swiglu_branch", 0), getattr(c, "dropout", 0.0) or 0.0) for _ in range(c.dec_layers))
        self.dec_norm = nn.LayerNorm(d)
        self.unit_head = nn.Linear(d, c.n_unit_tokens)
        self.order_head = nn.Linear(d, c.n_order_tokens)
        self.unit_qty_cond = nn.Embedding(c.n_unit_tokens, d)
        self.order_qty_cond = nn.Embedding(c.n_order_tokens, d)
        self.unit_qty_head = _mlp(d, c.n_qty)
        self.order_qty_head = _mlp(d, c.n_qty)
        self.value_head = _mlp(d, 1)
        self.register_buffer("unit_qty_mask", torch.tensor(c.unit_has_qty, dtype=torch.bool), persistent=False)
        self.register_buffer("order_qty_mask", torch.tensor(c.order_has_qty, dtype=torch.bool), persistent=False)
        self.register_buffer("neigh", _neighbours(), persistent=False)
        self.register_buffer("anchor_kinds", torch.tensor(ANCHOR_KINDS), persistent=False)
        causal = torch.ones(self.U + self.O, self.U + self.O, dtype=torch.bool).tril()
        self.register_buffer("causal", causal, persistent=False)
        last = torch.zeros(c.n_order_tokens, dtype=torch.bool)
        last[c.stop_index] = True
        self.register_buffer("stop_only", last, persistent=False)
        # growth: every grow_* parameter starts at zero (zero_growth), so outputs equal the pre-growth network
        if c.extra_width:
            self.grow_extra = nn.Linear(c.extra_width, w)
        if c.unit_extra_width:
            self.grow_unit = nn.Linear(c.unit_extra_width, d)
        if c.calc_width:
            self.grow_calc = nn.Linear(c.calc_width, w)
        if c.carrier_read:
            self.grow_carrier = nn.Linear(d, d)
        if c.reach_bias:
            self.grow_reach_eta = nn.Parameter(torch.zeros(c.dec_layers, N_ETA, dh))
            self.grow_reach_kind = nn.Parameter(torch.zeros(c.dec_layers, c.n_kinds, dh))
            self.grow_reach_slack = nn.Parameter(torch.zeros(c.dec_layers, N_SLACK, dh))
            self.register_buffer("eta_bucket", torch.tensor(ETA_BUCKET), persistent=False)
            self.register_buffer("slack_bucket", torch.tensor(SLACK_BUCKET), persistent=False)
        if c.reach_job:
            self.grow_reach_job = nn.Parameter(torch.zeros(c.dec_layers, N_ETA + 1, dh))   # last bucket: unreachable
        if getattr(c, "x8_token_width", 0):
            self.grow_x8_token = nn.Linear(c.x8_token_width, w)
        if getattr(c, "x8_unit_width", 0):
            self.grow_x8_unit = nn.Linear(c.x8_unit_width, d)
        if getattr(c, "lattice_channels", 0):
            import v8_lattice_net          # not a grow_ name: zero_growth() would zero both convolutions (no gradient path)
            self.lattice_net = v8_lattice_net.LatticeCNN(c.lattice_channels, c.lattice_width, out_dim=w)
        if c.prefix_unit_width:
            self.grow_prefix_unit = nn.Linear(c.prefix_unit_width, d)
        if c.prefix_order_width:
            self.grow_prefix_order = nn.Linear(c.prefix_order_width, d)
        if c.prefix_cash_width:
            self.grow_prefix_cash = nn.Linear(c.prefix_cash_width, d)
        if c.job_eta_v6:
            self.grow_v6_token_in = nn.Linear(8, 32)
            self.grow_v6_token_out = nn.Linear(32, w)
            self.grow_v6_job_in = nn.Linear(14, 32)  # five edge facts + nine job-kind bits
            self.grow_v6_job_out = nn.Linear(32, c.dec_layers * dh)
        if getattr(c, "v6_prev", False):
            self.v6_lag = nn.Embedding(2, d)
            self.v6_uslot = nn.Embedding(self.U, d)
            self.v6_oslot = nn.Embedding(self.O, d)
            self.v6_hq, self.v6_hk, self.v6_hv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
            self.grow_v6_hist = nn.Linear(d, d)
            self.grow_v6_prev_unit = nn.Linear(2 * d, d)   # [t-1 | t-2] concatenated, so the two turns keep their order
        if getattr(c, "v6_aux", False):
            self.aux_next = nn.Linear(d, 2 * c.n_unit_tokens)
            self.aux_opp = nn.Linear(d, 3 * c.n_unit_tokens)
        if getattr(c, "v6_aux", False) and getattr(c, "v6_aux_turn", False):
            self.aux_lag = nn.Embedding(5, d)                    # own t+1, t+2, opponent t, t+1, t+2
            self.aux_unit_tok = nn.Embedding(c.n_unit_tokens, d)   # quantity heads read the labelled command
            self.aux_unit_qty = _mlp(d, c.n_qty)
            self.aux_post = nn.Linear(d, d)       # own worker's state AFTER its current (teacher) command, for its t+1/t+2
            # future/opponent order programs: a causal decoder over (token, quantity) slots, teacher-forced on the target
            # prefix, cross-attending to the encoder memory plus the current turn's decoded program
            self.aux_order_bos = nn.Parameter(torch.zeros(d))
            self.aux_order_slot = nn.Embedding(self.O, d)
            self.aux_order_dec = nn.ModuleList([DecoderLayer(d, dh, c.ff_mult, getattr(c, "ffn", "gelu"), 0,
                                                             getattr(c, "dropout", 0.0) or 0.0)])
            self.aux_order_norm = nn.LayerNorm(d)
            self.aux_order_head = nn.Linear(d, c.n_order_tokens)
            self.aux_order_tok = nn.Embedding(c.n_order_tokens, d)
            self.aux_order_qty = _mlp(d, c.n_qty)
        self.zero_growth()

    def zero_growth(self):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n.startswith("grow_") and n not in (
                    "grow_v6_token_in.weight", "grow_v6_token_in.bias",
                    "grow_v6_job_in.weight", "grow_v6_job_in.bias"):
                    p.zero_()
        return self

    def growth_parameter_names(self):
        return [n for n, _ in self.named_parameters() if n.startswith("grow_")]

    # ------------------------------------------------------------------ helpers
    @property
    def codec(self):
        return self._codec_module or _codec()

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def initial_state(self):
        return ThinkState()

    new_state = initial_state

    def none_field(self, batch, device=None):
        c = self.config
        return FieldState.none(batch, self.S, c.C, c.K, device=device)

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
        if (CHECK_INPUTS or not rows.is_cuda) and (bool((index < 0).any()) or bool((is_tile.sum(1) != N_CELLS).any())):
            raise ValueError("exactly one TILE token per cell of both 10x10 farms required")
        return is_tile, cell.clamp(max=N_CELLS - 1), index

    def _slot_sources(self, rows, token_mask):
        """[B, G] token index writing each global slot: the g-th fixed state entity (pack order),
        then own carriers (farmer, hands), else the slot's own free token at N + g."""
        B, N, _ = rows.shape
        G = self.config.G
        ar = torch.arange(G, device=rows.device)
        src = (N + ar).expand(B, G).clone()
        if self.config.slot_anchor == "free":
            return src
        kind = rows[..., :12].argmax(-1)
        fixed = token_mask & (rows[..., :12].amax(-1) > 0.5) & torch.isin(kind, self.anchor_kinds)
        count = fixed.sum(1)
        if rows.is_cuda and not CHECK_INPUTS:   # sync-free (no graph break): same src on inputs that pass the check below
            own_carrier = token_mask & (kind == CARRIER_KIND) & (rows[..., :12].amax(-1) > 0.5) & (rows[..., ROLE_COL] > 0.5)
            tok = torch.arange(N, device=rows.device).expand(B, N)
            src = torch.cat((src, src.new_zeros(B, 1)), 1)
            for mask, offset in ((fixed, 0), (own_carrier, count[:, None])):
                slot = offset + mask.long().cumsum(1) - 1
                src.scatter_(1, torch.where(mask & (slot < G), slot, G), tok)   # column G collects the non-writes
            return src[:, :G]
        if bool((count != count[0]).any()):
            raise ValueError("every observation must carry the same fixed state entities")
        F_ = int(count[0])
        own_carrier = token_mask & (kind == CARRIER_KIND) & (rows[..., :12].amax(-1) > 0.5) & (rows[..., ROLE_COL] > 0.5)
        tok = torch.arange(N, device=rows.device).expand(B, N)
        for mask, offset in ((fixed, 0), (own_carrier, F_)):
            slot = offset + mask.long().cumsum(1) - 1
            b, t = (mask & (slot < G)).nonzero(as_tuple=True)
            src[b, slot[b, t]] = tok[b, t]
        return src

    def _codes(self, logits, tau, mode, noise, generator):
        if mode == "argmax":
            return F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
        if noise is None:
            noise = torch.rand(logits.shape, generator=generator, device=logits.device)
        g = -torch.log(-torch.log(noise.to(logits.dtype).clamp(1e-10, 1 - 1e-7)))
        hard = F.one_hot((logits + g).argmax(-1), logits.shape[-1]).to(logits.dtype)
        if mode == "sample" or not torch.is_grad_enabled():
            return hard
        soft = torch.softmax((logits + g) / tau, -1)
        return hard - soft.detach() + soft

    # ------------------------------------------------------------------ encoder + writer
    def encode(self, rows, token_mask, knownness, field=None, *, tau=1.0, code_mode="gumbel",
               code_noise=None, generator=None, extra=None, unit_extra=None, reach_eta=None, unit_prefix=None,
               order_prefix=None, order_cash=None, prev_unit=None, prev_order=None,
               v6_token=None, v6_resource_eta=None, v6_resource_route=None,
               v6_job_deadline=None, v6_job_deadline_present=None, x8_token=None, x8_unit=None, lattice=None):
        c = self.config
        for name, value, width in (("x8_token", x8_token, c.x8_token_width),
                                   ("x8_unit", x8_unit, c.x8_unit_width)):
            if width and value is None:
                raise ValueError(f"enabled {name} requires an explicit input tensor")
            if width and value.shape[-1] != width:
                raise ValueError(f"{name} width does not match the model configuration")
        if c.lattice_channels and lattice is None:
            raise ValueError("enabled lattice_channels requires an explicit lattice input")
        if c.v6_prev and (prev_unit is None or prev_order is None):
            raise ValueError("v6 history requires prev_unit and prev_order; use explicit -1 arrays at episode start")
        rows = rows.float()
        token_mask = token_mask.bool()
        known = knownness.bool() & token_mask[..., None]
        B, N, _ = rows.shape
        # Configuration is a contract, not a best-effort feature request. This also
        # prevents singleton batches/widths broadcasting into a different schema.
        for name, value, width, shape in (
                ("x8_token", x8_token, c.x8_token_width, (B, N, c.x8_token_width)),
                ("x8_unit", x8_unit, c.x8_unit_width, (B, c.max_units, c.x8_unit_width)),
                ("lattice", lattice, c.lattice_channels, (B, 2, BOARD, BOARD, c.lattice_channels))):
            if width and (value is None or tuple(value.shape) != shape):
                actual = None if value is None else tuple(value.shape)
                raise ValueError(f"{name} requires shape {shape}, got {actual}; use explicit zeros for ablation")
        x = self.inp(rows.masked_fill(~known, 0)) + self.known_inp(known.to(rows.dtype))
        if c.job_eta_v6:
            if any(z is None for z in (v6_token, v6_resource_eta, v6_resource_route,
                                       v6_job_deadline, v6_job_deadline_present, extra)):
                raise ValueError("pbc-job-input-v6.0 requires all job tensors and legacy extra")
            x = x + self._fdrop(self.grow_v6_token_out(torch.nn.functional.gelu(
                self.grow_v6_token_in(v6_token.float()))))
        reach = None
        if extra is not None and c.extra_width:
            ex = extra.float().masked_fill(~token_mask[..., None], 0)
            x = x + self._fdrop(self.grow_extra(ex[..., :c.extra_width]))
            if c.calc_width:
                x = x + self._fdrop(self.grow_calc(ex[..., c.extra_width + 2:c.extra_width + 2 + c.calc_width]))
            if c.reach_bias:
                own_tile = token_mask & (rows[..., TILE_KIND_COL] > 0.5) & (rows[..., ROLE_COL] > 0.5)
                tx = torch.round(rows[..., XY_COL] * (BOARD - 1)).long()
                ty = torch.round(rows[..., XY_COL + 1] * (BOARD - 1)).long()
                kind = torch.round(ex[..., c.extra_width]).long().clamp(0, c.n_kinds - 1)
                deadline = torch.round(ex[..., c.extra_width + 1]).long()
                reach = (own_tile, tx, ty, kind * own_tile, deadline * own_tile)
        reach_job = None
        if c.reach_job and reach is not None and reach_eta is not None:
            own_tile, tx, ty = reach[:3]
            cell = (ty * BOARD + tx).clamp(0, BOARD * BOARD - 1)                             # [B, S]
            eta = reach_eta.long().gather(2, cell[:, None, :].expand(B, reach_eta.shape[1], N))   # [B, U, S]
            b = self.eta_bucket[eta.clamp(0, len(ETA_BUCKET) - 1)]
            reach_job = torch.where(eta >= 999, torch.full_like(b, N_ETA), b)
        is_tile, cell_of, tile_index = self._tiles(rows, token_mask)
        if getattr(c, "x8_token_width", 0) and x8_token is not None:
            x = x + self._fdrop(self.grow_x8_token(x8_token.float().masked_fill(~token_mask[..., None], 0)))
        if getattr(c, "lattice_channels", 0) and lattice is not None:
            import v8_lattice_net
            x = x + self._fdrop(v8_lattice_net.scatter_to_tokens(self.lattice_net(lattice.float()), cell_of, is_tile).to(x.dtype))
        job_v6 = None
        if c.job_eta_v6:
            own_idx = tile_index[:, :100]
            bits = extra.float().gather(1, own_idx[..., None].expand(B, 100, extra.shape[-1]))
            bits = bits[..., 96:105] > 0.5
            job_v6 = (bits, v6_resource_eta, v6_resource_route,
                      v6_job_deadline, v6_job_deadline_present)
        mask = token_mask
        if c.think == "bottleneck":
            if code_mode not in CODE_MODES:
                raise ValueError("code_mode must be one of " + ", ".join(CODE_MODES))
            if field is None:
                field = self.none_field(B, rows.device)
            carried = (field.codes.reshape(B, self.S, c.C * c.K).to(x.dtype) @ self.carry_emb)
            fresh = field.fresh.to(rows.device)[:, None, None]
            cells_in = torch.where(fresh, self.none_cell.to(x.dtype), carried[:, :N_CELLS])
            slots_in = torch.where(fresh, self.none_slot.to(x.dtype), carried[:, N_CELLS:])
            add = cells_in.gather(1, cell_of[..., None].expand(B, N, x.shape[-1]))
            x = x + add * is_tile[..., None].to(x.dtype)
            # global slot g is written from src[:, g]: an anchored state-entity / own-carrier token
            # (which receives the slot identity + carried code there) or the slot's own free token
            src = self._slot_sources(rows, token_mask)
            x = torch.cat((x, x.new_zeros(B, c.G, x.shape[-1])), 1)
            x = x.scatter_add(1, src[..., None].expand(B, c.G, x.shape[-1]), self.slot_in.to(x.dtype)[None] + slots_in)
            mask = torch.cat((token_mask, src >= N), 1)
        attn = mask[:, None, None, :] if (rows.is_cuda and not CHECK_INPUTS) or not bool(mask.all()) else None
        if self._recur is None:
            for block in self.blocks:
                x = block(x, attn)
        else:
            a, b = self._recur
            loops = int(self.recur_T or getattr(c, "recur_loops", 1) or 1)
            for block in self.blocks[:a]:
                x = block(x, attn)
            e = x
            for block in self.blocks[a:b]:
                x = block(x, attn)
            if getattr(c, "recur_mode", "gate") == "norm":
                for _ in range(loops - 1):
                    y = self.recur_norm(x) + e
                    for block in self.blocks[a:b]:
                        y = block(y, attn)
                    x = y
            else:
                g = torch.tanh(self.grow_recur_gate).to(x.dtype)
                for _ in range(loops - 1):
                    y = x + e
                    for block in self.blocks[a:b]:
                        y = block(y, attn)
                    x = x + g * (y - x)
            for block in self.blocks[b:]:
                x = block(x, attn)
        h = self.enc_norm(x)
        tiles = h.gather(1, tile_index[..., None].expand(B, N_CELLS, h.shape[-1]))
        if c.think == "bottleneck":
            slots = h.gather(1, src[..., None].expand(B, c.G, h.shape[-1]))
            logits = self.write(torch.cat((tiles, slots), 1)).float().view(B, self.S, c.C, c.K)
            codes = self._codes(logits, tau, code_mode, code_noise, generator)
            pos = torch.cat((self.farm_emb(self.cell_role) + self.row_emb(self.cell_y) + self.col_emb(self.cell_x),
                             self.slot_emb.weight), 0)
            mem = self.mem_norm(codes.reshape(B, self.S, c.C * c.K).to(self.code_emb.dtype) @ self.code_emb + pos)
            value = self.value_head(mem.mean(1)).squeeze(-1)
            new = FieldState(codes, torch.zeros(B, dtype=torch.bool, device=rows.device))
            return Context(mem, None, mem[:, :100], mem[:, N_CELLS:], value, new, logits,
                           unit_extra=unit_extra, unit_prefix=unit_prefix, order_prefix=order_prefix,
                           order_cash=order_cash, job_v6=job_v6)
        mem = self.mem_proj(h)
        carriers = None
        if c.carrier_read:
            idx = carrier_indices(rows, token_mask, known, self.U)
            carriers = mem.gather(1, idx.clamp(min=0)[..., None].expand(B, self.U, mem.shape[-1]))
            carriers = carriers * (idx >= 0)[..., None].to(carriers.dtype)
        m = token_mask[..., None].to(mem.dtype)
        value = self.value_head((mem * m).sum(1) / m.sum(1)).squeeze(-1)
        slots = None
        if c.global_read:   # ablation: the same entity/carrier positions, read from encoder states
            src = self._slot_sources(rows, token_mask)
            got = h.gather(1, src.clamp(max=N - 1)[..., None].expand(B, c.G, h.shape[-1]))
            slots = self.mem_proj(got * (src < N)[..., None].to(got.dtype))
        hist = prev_emb = opp = None
        if getattr(c, "v6_prev", False) and prev_unit is not None and prev_order is not None:
            hist, prev_emb = self._history(prev_unit, prev_order)
        if getattr(c, "v6_aux", False) and (self.training or getattr(self, "_aux_eval", False)):
            oi = opp_carrier_indices(rows, token_mask, known, self.U)
            opp = mem.gather(1, oi.clamp(min=0)[..., None].expand(B, self.U, mem.shape[-1]))
            opp = opp * (oi >= 0)[..., None].to(opp.dtype)
        return Context(mem, attn, self.mem_proj(tiles[:, :100]), slots, value, None, None, unit_extra=unit_extra,
                       reach=reach, carriers=carriers, reach_job=reach_job, unit_prefix=unit_prefix,
                       order_prefix=order_prefix, order_cash=order_cash, job_v6=job_v6, hist=hist,
                       prev_unit_emb=prev_emb, opp_carriers=opp, x8_unit=x8_unit)

    # ------------------------------------------------------------------ decoder pieces
    def _history(self, prev_unit, prev_order):
        """v6: the previous two own turns as tokens in order. prev_unit [B, 2, U, 2] and prev_order [B, 2, O, 2] hold
        (codec token, quantity), -1 = none. Returns (tokens [B, 2U+2O, d], valid [B, 2U+2O]) and each worker's own
        previous commands with the lags concatenated in order [B, U, 2d]."""
        B = prev_unit.shape[0]
        ut, uq = prev_unit[..., 0].long(), prev_unit[..., 1].long()
        ot, oq = prev_order[..., 0].long(), prev_order[..., 1].long()
        lag = self.v6_lag.weight[:, None, :]                                              # [2, 1, d]
        ue = self._decision_emb(ut, uq, True) + lag[None] + self.v6_uslot.weight[None, None]   # [B, 2, U, d]
        oe = self._decision_emb(ot, oq, False) + lag[None] + self.v6_oslot.weight[None, None]  # [B, 2, O, d]
        uv, ov = ut >= 0, ot >= 0
        own = (ue * uv[..., None].to(ue.dtype)).permute(0, 2, 1, 3).reshape(B, ue.shape[2], 2 * ue.shape[-1])
        tokens = torch.cat((ue.reshape(B, -1, ue.shape[-1]), oe.reshape(B, -1, oe.shape[-1])), 1)
        valid = torch.cat((uv.reshape(B, -1), ov.reshape(B, -1)), 1)
        return (tokens, valid), own

    def _fdrop(self, t, p=None):
        """Feature dropout: in training, zero this input group's contribution for a random p of the examples (x 1/(1-p))."""
        p = float(getattr(self.config, "feature_dropout", 0.0) or 0.0) if p is None else p
        if not self.training or p <= 0:
            return t
        keep = torch.rand((t.shape[0],) + (1,) * (t.dim() - 1), device=t.device) >= p
        return t * keep.to(t.dtype) / (1.0 - p)

    def _hist_memory(self, ctx):
        """Project invariant history once per greedy turn; never persist an autograd cache on Context."""
        tokens, valid = ctx.hist
        B = tokens.shape[0]
        k = torch.cat((tokens.new_zeros(B, 1, tokens.shape[-1]), self.v6_hk(tokens)), 1)
        v = torch.cat((tokens.new_zeros(B, 1, tokens.shape[-1]), self.v6_hv(tokens)), 1)
        ok = torch.cat((valid.new_ones(B, 1), valid), 1)
        return k, v, ok

    def _hist_ctx(self, x, ctx, memory=None):
        """v6: queries read previous turns with a zero null key. Teacher forcing projects once too."""
        k, v, ok = self._hist_memory(ctx) if memory is None else memory
        a = (self.v6_hq(x) @ k.transpose(1, 2)) / math.sqrt(k.shape[-1])
        a = a.masked_fill(~ok[:, None, :], float("-inf")).softmax(-1)
        return self._fdrop(self.grow_v6_hist(a @ v), _hist_p(self))   # own-history memory (copycat guard)

    def _unit_features(self, ctx, unit_pos):
        """[B, U, d]: unit identity comes from the query; x/y and the local field cells here."""
        pos = unit_pos.long()
        x, y = pos[..., 0].clamp(0, BOARD - 1), pos[..., 1].clamp(0, BOARD - 1)
        out = self.unit_row(y) + self.unit_col(x)
        if self.config.local:
            B, U = x.shape
            cells = torch.cat((ctx.cells, ctx.cells.new_zeros(B, 1, ctx.cells.shape[-1])), 1)
            ids = self.neigh[(y * BOARD + x).reshape(-1)].view(B, U * 5)
            local = cells.gather(1, ids[..., None].expand(B, U * 5, cells.shape[-1])).view(B, U, -1)
            out = out + self.local_proj(local)
        if ctx.unit_extra is not None and self.config.unit_extra_width:
            out = out + self._fdrop(self.grow_unit(ctx.unit_extra.float()[:, :out.shape[1]]))
        if ctx.carriers is not None:
            out = out + self.grow_carrier(ctx.carriers[:, :out.shape[1]])
        if ctx.prev_unit_emb is not None:
            out = out + self._fdrop(self.grow_v6_prev_unit(ctx.prev_unit_emb[:, :out.shape[1]]), _hist_p(self))   # previous turns
        if ctx.x8_unit is not None and getattr(self.config, "x8_unit_width", 0):
            out = out + self._fdrop(self.grow_x8_unit(ctx.x8_unit.float()[:, :out.shape[1]]))
        return out

    def _reach_bias(self, ctx, unit_pos):
        """Per decoder layer [B, h, U, S] additive cross-attention logits for unit queries over own TILE tokens,
        from ETA = Manhattan + 1, the tile's job kind and the slack to its deadline (flip.py); None when off."""
        if ctx.reach is None or not self.config.reach_bias:
            return None
        own_tile, tx, ty, kind, deadline = ctx.reach
        pos = unit_pos.long()
        ux, uy = pos[..., 0], pos[..., 1]
        eta = (ux[:, :, None] - tx[:, None, :]).abs() + (uy[:, :, None] - ty[:, None, :]).abs() + 1   # [B, U, S]
        eta_b = self.eta_bucket[eta.clamp(0, len(ETA_BUCKET) - 1)]
        dl = deadline[:, None, :]
        slack_b = torch.where(dl > 0, self.slack_bucket[(dl - eta).clamp(-1, 40) + 1], torch.zeros_like(eta_b))
        valid = (own_tile[:, None, :] & (ux >= 0)[:, :, None]).unsqueeze(-1)
        job_bias = self._v6_job_bias(ctx, unit_pos) if self.config.job_eta_v6 else None
        # One lookup for all decoder layers: (eta, slack[, job]) bucket triples index a summed [combos, L*h] table. Advanced
        # indexing table[idx] had a sorted index_put backward that serialized ~1e6 duplicate indices (85% of GPU time);
        # one embedding backward sorts the keys once instead of once per table and layer.
        L = self.config.dec_layers
        U = eta_b.shape[1]
        table = self.grow_reach_eta.permute(1, 0, 2)[:, None] + self.grow_reach_slack.permute(1, 0, 2)[None, :]  # [E, S, L, h]
        key = eta_b * N_SLACK + slack_b
        if ctx.reach_job is not None:
            table = table[:, :, None] + self.grow_reach_job.permute(1, 0, 2)[None, None]                         # [E, S, J, L, h]
            key = key * (N_ETA + 1) + ctx.reach_job[:, :U]
        h = table.shape[-1]
        b_all = F.embedding(key, table.reshape(-1, L * h)).view(*key.shape, L, h)                               # [B, U, S, L, h]
        b_all = b_all + F.embedding(kind, self.grow_reach_kind.permute(1, 0, 2).reshape(-1, L * h)).view(*kind.shape, L, h)[:, None]
        out = []
        for layer in range(L):
            b = b_all[..., layer, :]
            if job_bias is not None:
                b = b + job_bias[layer]
            out.append((b * valid).permute(0, 3, 1, 2))
        return out

    def _v6_job_bias(self, ctx, unit_pos):
        """Finite per-job relation residual; absent jobs and padded units contribute exactly zero."""
        if ctx.job_v6 is None or ctx.reach is None:
            return None
        bits, resource_eta, resource_route, deadline, deadline_present = ctx.job_v6
        B, U = unit_pos.shape[:2]
        xy = torch.arange(100, device=unit_pos.device)
        x, y = xy % BOARD, xy // BOARD
        direct = ((unit_pos[..., 0, None] - x).abs() + (unit_pos[..., 1, None] - y).abs() + 1)
        present_worker = (unit_pos[..., 0] >= 0)[:, :, None, None]
        eta = direct[..., None].expand(B, U, 100, 9).clone()
        route = torch.ones_like(eta, dtype=torch.bool)
        eta[..., 2] = resource_eta[:, :U, ..., 0]
        eta[..., 6] = resource_eta[:, :U, ..., 1]
        route[..., 2] = resource_route[:, :U, ..., 0]
        route[..., 6] = resource_route[:, :U, ..., 1]
        present = bits[:, None] & present_worker
        route = route & present
        eta = torch.where(route, eta, 0)
        dl_present = deadline_present[:, None] & present
        slack = torch.where(route & dl_present,
                            (deadline[:, None] - eta).float().div(24).clamp(-2, 2), 0)
        facts = torch.stack((present.float(), route.float(),
                             torch.where(route, eta.float().div(24).clamp(0, 2), 0), slack,
                             (route & dl_present & (eta <= deadline[:, None])).float()), -1)
        names = torch.nn.functional.one_hot(torch.arange(9, device=unit_pos.device), 9).to(facts.dtype)
        names = names.view(1, 1, 1, 9, 9).expand(B, U, 100, 9, 9)
        edge = torch.cat((facts, names), -1)
        gain = self.grow_v6_job_out(torch.nn.functional.gelu(self.grow_v6_job_in(edge)))
        gain = (gain * present[..., None]).sum(3)
        h = self.config.dec_heads or self.config.heads
        gain = gain.view(B, U, 100, self.config.dec_layers, h)
        _, tx, ty = ctx.reach[:3]
        cell = (ty * BOARD + tx).clamp(0, 99)
        full = gain.gather(2, cell[:, None, :, None, None].expand(B, U, cell.shape[1],
                                                                 self.config.dec_layers, h))
        return [full[..., layer, :] for layer in range(self.config.dec_layers)]

    def _global_feature(self, ctx):
        """[B, d] direct read of all G slot memories (codes + slot embeddings), or 0."""
        if not self.config.global_read:
            return 0
        return self.global_proj(self.global_down(ctx.slots).flatten(1))

    def _decision_emb(self, tok, qty, unit):
        """Embedding of a chosen decision; -1 qty (or non-quantity token) -> 'no quantity'."""
        table, has = (self.unit_tok_emb, self.unit_qty_mask) if unit else (self.order_tok_emb, self.order_qty_mask)
        t = tok.clamp(min=0)
        q = torch.where(has[t] & (qty >= 0), qty, torch.full_like(qty, self.config.n_qty))
        return table(t) + self.qty_emb(q)

    # ------------------------------------------------------------------ teacher forcing
    def decode_teacher(self, ctx, unit_pos, unit_prefix, order_prefix):
        U, O, B = self.U, self.O, unit_pos.shape[0]
        ut, uq = unit_prefix[..., 0].long(), unit_prefix[..., 1].long()
        ot, oq = order_prefix[..., 0].long(), order_prefix[..., 1].long()
        present = unit_pos[..., 0] >= 0
        present[:, 0] = True
        dec = torch.cat((self._decision_emb(ut, uq, True), self._decision_emb(ot, oq, False)), 1)
        prev = (torch.arange(U + O, device=dec.device) - 1).expand(B, -1).clone()
        prev[:, U] = present.sum(1) - 1                          # order 0 follows the last present unit
        prev_emb = dec.gather(1, prev.clamp(min=0)[..., None].expand(-1, -1, dec.shape[-1]))
        prev_emb[:, 0] = self.bos
        x = self.query.weight[None] + prev_emb
        gf = self._global_feature(ctx)
        x = x + (gf[:, None] if torch.is_tensor(gf) else 0)
        xu, xo = x[:, :U] + self._unit_features(ctx, unit_pos), x[:, U:]
        if ctx.unit_prefix is not None and self.config.prefix_unit_width:
            xu = xu + self._fdrop(self.grow_prefix_unit(ctx.unit_prefix.float()[:, :U]))
        if ctx.order_prefix is not None and self.config.prefix_order_width:
            xo = xo + self._fdrop(self.grow_prefix_order(ctx.order_prefix.float()[:, :O]))
        if ctx.order_cash is not None and self.config.prefix_cash_width:
            xo = xo + self._fdrop(self.grow_prefix_cash(ctx.order_cash.float()[:, :O]))
        x = torch.cat((xu, xo), 1)
        if ctx.hist is not None:
            x = x + self._hist_ctx(x, ctx)
        valid = torch.cat((present, present.new_ones(B, O)), 1)
        self_mask = self.causal[None, None] & valid[:, None, None, :]
        mem_kv = [layer.memory(ctx.mem) for layer in self.dec]
        biases = self._reach_bias(ctx, unit_pos)
        for i, (layer, kv) in enumerate(zip(self.dec, mem_kv)):
            bias = None
            if biases is not None:
                bu = biases[i]
                bias = torch.cat((bu, bu.new_zeros(B, bu.shape[1], O, bu.shape[3])), 2)
            x = layer(x, self_mask, kv, ctx.mem_mask, mem_bias=bias)
        h = self.dec_norm(x)
        hu, ho = h[:, :U], h[:, U:]
        order_logits = self.order_head(ho)
        order_logits[:, O - 1] = order_logits[:, O - 1].masked_fill(~self.stop_only, float("-inf"))
        out = {"unit_logits": self.unit_head(hu),
               "unit_qty_logits": self.unit_qty_head(hu + self.unit_qty_cond(ut.clamp(min=0))),
               "order_logits": order_logits,
               "order_qty_logits": self.order_qty_head(ho + self.order_qty_cond(ot.clamp(min=0))),
               "unit_present": present}
        if getattr(self.config, "v6_aux", False) and (self.training or getattr(self, "_aux_eval", False)):
            # Explicit diagnostic opt-in; normal eval/deploy still does no auxiliary work.
            V = self.config.n_unit_tokens
            turn = getattr(self.config, "v6_aux_turn", False)
            h_own = hu + self.aux_post(dec[:, :U]) if turn else hu      # post-decision state of each own worker
            out["aux_next_logits"] = self.aux_next(h_own).view(B, U, 2, V)
            if ctx.opp_carriers is not None:
                out["aux_opp_logits"] = self.aux_opp(ctx.opp_carriers).view(B, U, 3, V)
            if turn:
                out["aux_unit_h"], out["aux_opp_h"] = h_own, ctx.opp_carriers
                mem = torch.cat((ctx.mem, h.to(ctx.mem.dtype)), 1)                 # encoder memory + current program
                prog_ok = torch.cat((present, present.new_ones(B, O)), 1)
                mm = ctx.mem_mask if ctx.mem_mask is not None else prog_ok.new_ones(B, 1, 1, ctx.mem.shape[1])
                out["aux_mem"] = mem
                out["aux_mem_mask"] = torch.cat((mm, prog_ok[:, None, None, :]), -1)
        return out

    def aux_order_decode(self, out, which, tok, qty):
        """Training-only causal decoder for future/opponent order programs, teacher-forced on the target prefix.
        tok/qty [B, L, O] target (token, quantity), -1 = none; which "next_order" (lags own t+1, t+2) or "opp_order"
        (opponent t, t+1, t+2). Returns (token logits [B, L, O, Vo], slot states [B, L, O, d])."""
        mem, mm = out["aux_mem"], out["aux_mem_mask"]
        B, L, O = tok.shape
        d = mem.shape[-1]
        lag0 = 0 if which == "next_order" else 2
        prev = self._decision_emb(tok.clamp(min=0), qty, False)                  # [B, L, O, d]
        prev = torch.where((tok >= 0)[..., None], prev, torch.zeros_like(prev))
        x = torch.cat((self.aux_order_bos.expand(B, L, 1, d), prev[:, :, :-1]), 2)
        lags = self.aux_lag(torch.arange(lag0, lag0 + L, device=tok.device))       # [L, d]
        x = (x + lags[None, :, None] + self.aux_order_slot.weight[None, None, :O]).reshape(B * L, O, d).to(mem.dtype)
        causal = torch.ones(O, O, dtype=torch.bool, device=tok.device).tril()[None, None]
        # The memory K/V are projected once per example and shared by every lag and by both calls (own / opponent):
        # identical values to projecting B*L expanded copies, at 1/(total lags) of the projection work and memory.
        kv = out.get("aux_mem_kv")
        if kv is None:
            kv = out["aux_mem_kv"] = [layer.memory(mem) for layer in self.aux_order_dec]
        for layer, lkv in zip(self.aux_order_dec, kv):
            x = layer.forward_lags(x, L, causal, lkv, mm)
        hs = self.aux_order_norm(x).reshape(B, L, O, d)
        return self.aux_order_head(hs), hs

    def aux_qty_logits(self, out, which, tok):
        """Training-only quantity logits given the labelled command/order token tok [B, L, X] (-1 = none, clamped):
        which in ("next_unit", "opp_unit", "next_order", "opp_order") -> [B, L, X, n_qty]."""
        t = tok.clamp(min=0)
        if which.endswith("_unit"):
            h = out["aux_unit_h"] if which == "next_unit" else out["aux_opp_h"]
            lags = torch.arange(2, device=t.device) if which == "next_unit" else torch.arange(2, 5, device=t.device)
            h = h[:, None] + self.aux_lag(lags)[None, :, None] + self.aux_unit_tok(t)
            return self.aux_unit_qty(h)
        return self.aux_order_qty(out["aux_order_hs"][which] + self.aux_order_tok(t))

    def forward(self, rows, token_mask, knownness, unit_pos, unit_prefix, order_prefix, *,
                field=None, tau=1.0, code_mode="gumbel", code_noise=None, generator=None, extra=None,
                unit_extra=None, reach_eta=None, unit_prefix_state=None, order_prefix_state=None, prev_unit=None,
                prev_order=None, order_cash_state=None, v6_token=None, v6_resource_eta=None,
                v6_resource_route=None, v6_job_deadline=None, v6_job_deadline_present=None, x8_token=None, x8_unit=None,
                lattice=None):
        """Teacher-forced step.  unit_prefix [B,U,2] / order_prefix [B,O,2] hold (token, qty)
        labels (-1 = absent / none), or pass (tok, qty) tuples.  Returns a dict with
        unit_logits [B,U,Vu], unit_qty_logits [B,U,Q], order_logits [B,O,Vo],
        order_qty_logits [B,O,Q], value [B], field (FieldState or None), code_logits."""
        unit_prefix = torch.stack(unit_prefix, -1) if isinstance(unit_prefix, (tuple, list)) else unit_prefix
        order_prefix = torch.stack(order_prefix, -1) if isinstance(order_prefix, (tuple, list)) else order_prefix
        ctx = self.encode(rows, token_mask, knownness, field, tau=tau, code_mode=code_mode,
                          code_noise=code_noise, generator=generator, extra=extra, unit_extra=unit_extra,
                          reach_eta=reach_eta, unit_prefix=unit_prefix_state, order_prefix=order_prefix_state,
                          order_cash=order_cash_state, prev_unit=prev_unit, prev_order=prev_order,
                          v6_token=v6_token, v6_resource_eta=v6_resource_eta,
                          v6_resource_route=v6_resource_route, v6_job_deadline=v6_job_deadline,
                          v6_job_deadline_present=v6_job_deadline_present, x8_token=x8_token, x8_unit=x8_unit,
                          lattice=lattice)
        out = self.decode_teacher(ctx, unit_pos, unit_prefix, order_prefix)
        out.update(value=ctx.value, field=ctx.field, code_logits=ctx.code_logits)
        return out

    # ------------------------------------------------------------------ free-running decode (cached)
    @staticmethod
    def _choose(logits, temperature, generator):
        if not temperature:
            return logits.argmax(-1)
        p = torch.softmax(logits.float() / temperature, -1)
        return torch.multinomial(p, 1, generator=generator).squeeze(-1)

    def _dec_step(self, x, caches, mem_kv, mem_mask, valid, biases=None, p=None):
        v = torch.stack(valid, 1)
        mask = None if bool(v.all()) else v[:, None, None, :]
        x = x[:, None]
        for i, (layer, kv, cache) in enumerate(zip(self.dec, mem_kv, caches)):
            bias = None if biases is None or p is None else biases[i][:, :, p:p + 1]
            x = layer(x, mask, kv, mem_mask, cache, mem_bias=bias)
        return self.dec_norm(x[:, 0])

    def decode_free(self, ctx, unit_pos, *, temperature=0.0, generator=None, full_orders=False, prefix_fn=None):
        """Sequential decode farmer -> hands -> orders -> STOP; the field memory is projected
        once per layer and the decoder keeps a K/V cache, so each decision is one query token."""
        if self.config.noop_unit_mask and prefix_fn is None:
            raise ValueError("noop_unit_mask requires an actor-observation prefix callback")
        U, O = self.U, self.O
        B = unit_pos.shape[0]
        present = unit_pos[..., 0] >= 0
        present[:, 0] = True
        n_units = int(present.sum(1).max())
        mem_kv = [layer.memory(ctx.mem) for layer in self.dec]
        hist_memory = self._hist_memory(ctx) if ctx.hist is not None else None
        caches = [dict() for _ in self.dec]
        # These optimizations target one-thread CPU play, not compiled training/GPU execution.
        cpu_eval = not self.training and ctx.mem.device.type == "cpu"
        active_pos = unit_pos[:, :n_units] if cpu_eval else unit_pos
        feats = self._unit_features(ctx, active_pos)
        biases = self._reach_bias(ctx, active_pos)
        gf = self._global_feature(ctx)
        last = self.bos.expand(B, -1)
        valid = []
        unit_tok = torch.full((B, U), -1, dtype=torch.long, device=unit_pos.device)
        unit_qty = torch.full_like(unit_tok, -1)
        for p in range(n_units):
            valid.append(present[:, p])
            q = self.query.weight[p] + gf + last + feats[:, p]
            if prefix_fn is not None and self.config.prefix_unit_width:
                q = q + self.grow_prefix_unit(prefix_fn("unit", p).to(q.dtype))
            if ctx.hist is not None:
                q = q + self._hist_ctx(q[:, None], ctx, hist_memory)[:, 0]
            h = self._dec_step(q, caches, mem_kv, ctx.mem_mask, valid, biases, p)
            logits = self.unit_head(h)
            if self.config.noop_unit_mask:
                allowed = prefix_fn("unit_mask", p).to(device=logits.device, dtype=torch.bool)
                if allowed.shape != logits.shape:
                    raise ValueError("unit_mask must match [batch, unit vocabulary]; act() supports batch 1")
                allowed = allowed.clone()
                allowed[:, self.codec.PASS] = True
                logits = logits.masked_fill(~allowed, float("-inf"))
            tok = self._choose(logits, temperature, generator)
            needs_qty = self.unit_qty_mask[tok]
            if cpu_eval and temperature == 0.0 and not bool(needs_qty.any()):
                qty = torch.full_like(tok, -1)
            else:
                # Preserve all RNG draws in stochastic decoding, including discarded quantities.
                qty = self._choose(self.unit_qty_head(h + self.unit_qty_cond(tok)), temperature, generator)
                qty = torch.where(needs_qty, qty, torch.full_like(qty, -1))
            here = present[:, p]
            unit_tok[:, p] = torch.where(here, tok, -1)
            unit_qty[:, p] = torch.where(here, qty, -1)
            if prefix_fn is not None:
                prefix_fn("apply_unit", p, tok, qty)
            last = torch.where(here[:, None], self._decision_emb(tok, qty, True), last)
        order_tok = torch.full((B, O), -1, dtype=torch.long, device=unit_pos.device)
        order_qty = torch.full_like(order_tok, -1)
        done = torch.zeros(B, dtype=torch.bool, device=unit_pos.device)
        stop = self.config.stop_index
        for k in range(O):
            valid.append(torch.ones(B, dtype=torch.bool, device=unit_pos.device))
            q = self.query.weight[U + k] + gf + last
            if prefix_fn is not None and self.config.prefix_order_width:
                q = q + self.grow_prefix_order(prefix_fn("order", k).to(q.dtype))
            if prefix_fn is not None and self.config.prefix_cash_width:
                q = q + self.grow_prefix_cash(prefix_fn("cash", k).to(q.dtype))
            if ctx.hist is not None:
                q = q + self._hist_ctx(q[:, None], ctx, hist_memory)[:, 0]
            h = self._dec_step(q, caches, mem_kv, ctx.mem_mask, valid)
            logits = self.order_head(h)
            if k == O - 1:
                logits = logits.masked_fill(~self.stop_only, float("-inf"))
            tok = self._choose(logits, temperature, generator)
            needs_qty = self.order_qty_mask[tok]
            if cpu_eval and temperature == 0.0 and not bool(needs_qty.any()):
                qty = torch.full_like(tok, -1)
            else:
                # Preserve all RNG draws in stochastic decoding, including discarded quantities.
                qty = self._choose(self.order_qty_head(h + self.order_qty_cond(tok)), temperature, generator)
                qty = torch.where(needs_qty, qty, torch.full_like(qty, -1))
            order_tok[:, k] = torch.where(done, -1, tok)
            order_qty[:, k] = torch.where(done, -1, qty)
            if prefix_fn is not None and not bool(done.all()):
                prefix_fn("apply_order", k, tok, qty)
            done = done | (tok == stop)
            last = self._decision_emb(tok, qty, False)
            if bool(done.all()) and not full_orders:
                break
        return {"unit_tok": unit_tok, "unit_qty": unit_qty, "order_tok": order_tok, "order_qty": order_qty}

    # ------------------------------------------------------------------ single-observation play
    def observation_tensors(self, clean, configuration=None):
        """Model inputs for one actor view through data.features (float16-rounded rows, exactly
        as in training); a local mirror of it is used when data.py is not shipped."""
        try:
            import data as _data
            feats = _data.features(clean, configuration or {}, feature_schema=self.config.feature_schema)
        except ImportError as exc:
            # A grown checkpoint must not silently lose trained feature inputs.
            c = self.config
            if (c.extra_width or c.unit_extra_width or c.calc_width
                    or c.reach_bias or c.reach_job or c.job_eta_v6 or c.feature_schema != "legacy"):
                raise RuntimeError("required actor-feature extractor unavailable") from exc
            import numpy as np
            from farm_neat.policy import observation as features
            cfg = {k: v for k, v in (configuration or {}).items() if k in features.CONFIG or k == "marketParams"}
            packed = features.pack(clean, None, cfg)
            farm = clean["farms"][clean["player"]]
            units = [farm["farmer"], *farm["hands"]][:self.U]
            pos = np.full((self.U, 2), -1, np.int64)
            pos[:len(units)] = np.asarray(units, np.int64)
            feats = {"rows": packed["rows"].astype(np.float16), "token_mask": packed["token_mask"],
                     "knownness": packed["knownness"], "unit_pos": pos}
        rows = torch.from_numpy(feats["rows"].astype("float32"))[None]
        token_mask = torch.from_numpy(feats["token_mask"])[None]
        known = torch.from_numpy(feats["knownness"])[None]
        pos = torch.full((1, self.U, 2), -1, dtype=torch.long)
        up = torch.from_numpy(feats["unit_pos"])[:self.U]
        pos[0, :len(up)] = up
        self._last_extra = None
        self._last_job_eta = None
        if self.config.extra_width or self.config.unit_extra_width:
            # PRO-FEATURES-R2-0927 #1: a successful but stale extractor must not silently drop trained inputs either
            need = ("extra", "unit_extra") + (("reach_eta",) if self.config.reach_job else ())
            missing = [k for k in need if k not in feats]
            if missing:
                raise KeyError(f"grown model inputs missing from data.features: {missing}")
            self._last_extra = (torch.from_numpy(feats["extra"].astype("float32"))[None],
                                torch.from_numpy(feats["unit_extra"].astype("float32"))[None],
                                torch.from_numpy(feats["reach_eta"].astype("int64"))[None] if "reach_eta" in feats else None)
        if self.config.job_eta_v6:
            names = ("v6_token", "v6_resource_eta", "v6_resource_route", "v6_job_deadline",
                     "v6_job_deadline_present")
            if any(k not in feats for k in names):
                raise ValueError("pbc-job-input-v6.0 extractor is unavailable")
            self._last_job_eta = {k: torch.from_numpy(feats[k].astype("float32" if k == "v6_token" else
                                                             "int64" if k in ("v6_resource_eta", "v6_job_deadline") else "bool"))[None]
                                  for k in names}
        return rows, token_mask, known, pos

    @torch.no_grad()
    def act(self, observation, generator=None, temperature=1.0, state=None, configuration=None,
            code_mode=None):
        """One decision -> (action dict, ThinkState).  ``observation`` is the raw or actor-
        whitelisted observation; the field carried in ``state`` resets at step 0 or when
        the step does not advance (a new episode)."""
        from farm_neat.training.bc import actor_observation
        clean = actor_observation(dict(observation))
        if "step" not in clean:
            clean["step"] = int(clean["day"]) * 24 + int(clean["hour"])
        step = int(clean["step"])
        rows, token_mask, known, pos = self.observation_tensors(clean, configuration)
        field = None
        if isinstance(state, ThinkState) and state.field is not None and 0 < step and step > state.last_step:
            field = state.field
        extra, unit_extra, reach_eta = getattr(self, "_last_extra", None) or (None, None, None)
        hist, prev_unit, prev_order = (), None, None
        if getattr(self.config, "v6_prev", False):
            if isinstance(state, ThinkState) and 0 < step and step > state.last_step:
                hist = tuple(state.hist)
            prev_unit, prev_order = v6_history_arrays(hist, step, self.U, self.O)
        ctx = self.encode(rows, token_mask, known, field, code_mode=code_mode or self.config.infer_code_mode,
                          generator=generator, extra=extra, unit_extra=unit_extra, reach_eta=reach_eta,
                          prev_unit=prev_unit, prev_order=prev_order, **(self._last_job_eta or {}))
        labels = self.decode_free(ctx, pos, temperature=temperature, generator=generator,
                                  prefix_fn=self._prefix_fn(clean, configuration))
        lab = {k: [int(v) for v in labels[k][0].tolist()] for k in ("unit_tok", "unit_qty")}
        ot, oq = labels["order_tok"][0].tolist(), labels["order_qty"][0].tolist()
        n = next((i + 1 for i, t in enumerate(ot) if t == self.config.stop_index), len(ot))
        lab["order_tok"], lab["order_qty"] = [int(t) for t in ot[:n]], [int(q) for q in oq[:n]]
        action = self.codec.decode_step(lab, clean)
        if getattr(self.config, "v6_prev", False):
            # Replay history is encode_step(wire, obs), not the decoder's latent labels:
            # decode_step strips trailing NOOPs and multiple quantity bins can name the same count.
            try:   # the same encoder the training history uses (data.add_v6 <- episode_examples <- encode_step_lenient)
                import data as _data
                saved = _data.encode_step_lenient(action, clean)
            except ImportError:
                saved = self.codec.encode_step(action, clean)
            hist = (hist + ((step, saved["unit_tok"], saved["unit_qty"],
                            saved["order_tok"], saved["order_qty"]),))[-2:]
        return action, ThinkState(ctx.field, step, hist)

    def _prefix_fn(self, clean, configuration=None):
        """Free-running same-step prefix state for act() (batch 1): the executor follows the sampled decisions exactly
        as data.episode_examples follows the teacher's (both decode tokens with codec before executing them)."""
        c = self.config
        if not (c.noop_unit_mask or c.prefix_unit_width or c.prefix_order_width or c.prefix_cash_width):
            return None
        if c.noop_unit_mask:
            private = clean.get("private")
            if not isinstance(private, dict) or any(k not in private for k in ("shed", "seeds", "inventories")):
                raise ValueError("noop_unit_mask requires complete actor private inventories, shed and seeds")
            if int((configuration or {}).get("turnsPerDay", 24)) != 24:
                raise ValueError("noop_unit_mask: current PrefixExecutor requires turnsPerDay=24")
        import prefix as _prefix
        pe = _prefix.PrefixExecutor(clean, configuration=configuration,
                                    cash_enabled=bool(c.prefix_cash_width))
        n_units = len(pe.positions)
        state = {"units_done": 0}

        def fn(kind, k, tok=None, qty=None):
            if kind == "unit_mask":
                from noop_mask import feasible_unit_tokens
                return torch.tensor([feasible_unit_tokens(pe, self.codec, k)], dtype=torch.bool)
            if kind == "unit":
                return torch.from_numpy(pe.unit_features(k).astype("float16").astype("float32"))[None]
            if kind == "apply_unit":
                t = int(tok[0])
                pe.apply_unit(k, self.codec.decode_unit(t, int(qty[0]), clean, k) if t >= 0 else ["PASS"])
                state["units_done"] = k + 1
                return None
            while state["units_done"] < n_units:                 # units beyond the decoder's slots still act (PASS)
                pe.apply_unit(state["units_done"], ["PASS"])
                state["units_done"] += 1
            if kind == "order":
                return torch.from_numpy(pe.order_features(k).astype("float16").astype("float32"))[None]
            if kind == "cash":
                return torch.from_numpy(pe.cash_features(k, self.config.max_orders).astype("float16").astype("float32"))[None]
            t = int(tok[0])
            if t != self.config.stop_index:
                pe.apply_order(self.codec.decode_order(t, int(qty[0]), clean))
            return None
        return fn

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
        net = cls(ModelConfig.from_dict(config), codec=codec)
        net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in blob["model"].items()})
        return net.eval()

    from_checkpoint = load


def load_policy(path):
    """Training checkpoint (train.py) or PolicyNet.save file -> PolicyNet in eval mode."""
    return PolicyNet.load(path)


# ----------------------------------------------------------------------------- latency / size
def synthetic_inputs(n_tokens=250, n_units=1, max_units=16, batch=1, seed=0, hands_opp=0):
    """Packed-shape inputs: GLOBAL, own FARM/200-cell tiles/carriers, filler tokens."""
    g = torch.Generator().manual_seed(seed)
    base = 1 + 1 + 100 + n_units + 1 + 100 + 1 + hands_opp
    n = max(n_tokens, base)
    rows = torch.zeros(batch, n, IN_WIDTH)
    known = torch.zeros(batch, n, IN_WIDTH, dtype=torch.bool)
    rows[:, :, 35:60] = torch.rand(batch, n, 25, generator=g)
    known[:, :, :60] = True
    rows[:, 0, 0] = 1
    i = 1
    for role in (0, 1):
        rows[:, i, 1] = 1
        rows[:, i, 12 + role] = 1
        i += 1
        for y in range(BOARD):
            for x in range(BOARD):
                rows[:, i, TILE_KIND_COL] = 1
                rows[:, i, 12 + role] = 1
                rows[:, i, XY_COL], rows[:, i, XY_COL + 1] = x / 9, y / 9
                rows[:, i, 29 + int(torch.randint(6, (1,), generator=g))] = 1
                i += 1
        for _ in range(n_units if role == 0 else 1 + hands_opp):
            rows[:, i, 3] = 1
            rows[:, i, 12 + role] = 1
            i += 1
    for j in range(i, n):
        rows[:, j, 5 + j % 5] = 1
    token_mask = torch.ones(batch, n, dtype=torch.bool)
    pos = torch.full((batch, max_units, 2), -1, dtype=torch.long)
    pos[:, :n_units] = torch.randint(BOARD, (batch, n_units, 2), generator=g)
    return rows, token_mask, known, pos


@torch.no_grad()
def latency(net, *, n_tokens=250, n_units=1, repeats=30, warmup=5, threads=1, full_orders=True):
    """Median/p90 ms of encode + greedy sequential decode for one observation on CPU."""
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        net = net.eval()
        rows, mask, known, pos = synthetic_inputs(n_tokens, n_units, net.U)
        field = net.none_field(1) if net.config.think == "bottleneck" else None
        if field is not None:
            field.fresh[:] = False
        times, enc_times = [], []
        for i in range(warmup + repeats):
            t0 = time.perf_counter()
            ctx = net.encode(rows, mask, known, field, code_mode="argmax")
            t1 = time.perf_counter()
            net.decode_free(ctx, pos, temperature=0.0, full_orders=full_orders)
            t2 = time.perf_counter()
            if i >= warmup:
                times.append(t2 - t0)
                enc_times.append(t1 - t0)
        times.sort()
        return {"n_tokens": int(rows.shape[1]), "units": n_units, "threads": threads,
                "full_orders": full_orders, "repeats": repeats,
                "median_ms": 1000 * statistics.median(times),
                "p90_ms": 1000 * times[int(0.9 * (len(times) - 1))],
                "encode_median_ms": 1000 * statistics.median(enc_times)}
    finally:
        torch.set_num_threads(old)


def size_report(config=None, codec=None):
    net = PolicyNet(config, codec=codec)
    enc = sum(p.numel() for n, p in net.named_parameters() if n.startswith(("inp", "known_inp", "blocks", "enc_norm")))
    return {"parameters": net.parameter_count(), "encoder_parameters": enc,
            "config": {k: v for k, v in net.config.to_dict().items() if k not in ("unit_has_qty", "order_has_qty")}}
