"""N flow policy network (schema nflow-policy-v1): typed resource-flow graph + fast/slow/worker memory + sequential
primitive decoder over the existing codec grammar. Fresh random initialisation only (no loader for foreign weights).

Per physical step t (batch rows are independent graphs; memory links steps of one seat):
  1. nodes: 296 fixed entity slots (nf_format layout; typed input projections per node type) + FAST + SLOW memory
     nodes + 8 calendar landmarks (now, next shop tick, dusk, +2d, +4d, +8d, +12d, terminal) + pad -> N_NODES = 320.
  2. relational blocks 1..R-1 (stateless, run for all B*L steps in parallel): pre-RMSNorm, attention with H heads split
     into relational heads (additive learned bias per (relation type, head), relation types from the static typed
     graph: tile adjacency, mirror tile, farm membership, carrier-tile, custody, market-market, shop-market,
     resource-resource, calendar forward/backward sweep, calendar-resource, hub, memory) and global heads (no bias,
     flash-eligible), then SwiGLU.
  3. memory (sequential over the L steps of a chunk, carried across chunks): fast GRU (128) from a pooled graph summary
     and the actor's own previous action; slow GRU (256) updated at dawn from the previous day's mean summary; one
     shared worker GRU (64) per own worker slot from its carrier node + its previous command, reset at dawn for hands
     (hands are dismissed at dusk), never for the farmer.
  4. relational block R reads memory: FAST/SLOW nodes <- memory projections, own carrier nodes += worker state.
  5. decoder (teacher forcing in training, greedy sequential with K/V cache in play): 25 unit positions (farmer, hands
     in engine order; absent slots decode an ABSENT symbol so train/serve prefixes match) then <= 11 order positions
     (STOP forced at the last); each query adds the exact same-step prefix features (prefix.PrefixExecutor rows),
     the worker memory, the carrier node, unit ETA/extra features; cross-attention to the node states with a learned
     per-head bias from the unit's real ETA to each own tile.
Aux heads (training targets only): WDL (3-class sign of final cash margin), own next-step unit tokens, opponent
current unit tokens.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SCHEMA = "nflow-policy-v1"
TYPES = ("GLOBAL", "FARM", "TILE", "CARRIER", "STOCK", "MARKET", "SHOP", "CONSTANT", "CONTROL")
TYPE_RANGES = {"GLOBAL": (0, 1), "FARM": (1, 3), "TILE": (3, 203), "CARRIER": (203, 253), "STOCK": (253, 270),
               "MARKET": (270, 279), "SHOP": (279, 287), "CONSTANT": (287, 295), "CONTROL": (295, 296)}
N_ENT = 296
FAST_NODE, SLOW_NODE = 296, 297
CAL0, N_CAL = 298, 8
N_NODES = 320
U, O = 25, 11
OWN_CARRIER = 203
OPP_CARRIER = 228
OWN_TILE = 3
REL_NAMES = ("other", "self", "tile_adj", "tile_mirror", "farm_member", "carrier_tile", "carrier_carrier", "custody",
             "market_market", "shop_market", "resource", "hub", "memory", "cal_later", "cal_earlier", "cal_resource",
             "pad")
ETA_BUCKETS = 12


def eta_bucket(eta):
    """reach_eta (0..998 steps, 999 = unreachable) -> 0..11."""
    e = eta.long()
    b = torch.where(e <= 8, e, 8 + (e > 12).long() + (e > 24).long())
    return torch.where(e >= 999, torch.full_like(b, ETA_BUCKETS - 1), b.clamp(max=ETA_BUCKETS - 2))


@dataclass
class NFlowConfig:
    width: int = 256
    blocks: int = 4
    heads: int = 8
    rel_heads: int = 4
    ffn_hidden: int = 0            # 0 -> round_up(8/3 * width, 64)
    fast: int = 128
    slow: int = 256
    worker: int = 64
    dec_layers: int = 2
    dropout: float = 0.0
    type_widths: dict = field(default_factory=dict)   # from COLUMNS.json "widths"
    columns_sha256: str = ""
    # grammar (codec.py)
    n_unit_tokens: int = 44
    n_order_tokens: int = 23
    n_qty: int = 75
    max_units: int = 25
    max_orders: int = 10
    stop_index: int = 0
    unit_has_qty: list = field(default_factory=list)
    order_has_qty: list = field(default_factory=list)
    # serving-harness interface flags read by agent.py (features it must supply)
    x8_token_width: int = 354
    x8_unit_width: int = 41
    lattice_channels: int = 0
    v6_prev: bool = True
    feature_schema: str = "legacy"
    schema: str = SCHEMA

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @property
    def hidden(self):
        return self.ffn_hidden or int(math.ceil(8 / 3 * self.width / 64) * 64)


def resolve_config(cfg: NFlowConfig, codec=None, columns=None):
    c = dataclasses.replace(cfg)
    if codec is not None:
        c.n_unit_tokens, c.n_order_tokens = len(codec.UNIT_VOCAB), len(codec.ORDER_VOCAB)
        c.n_qty, c.max_units, c.max_orders = int(codec.QTY_BIN_COUNT), int(codec.MAX_UNITS), int(codec.MAX_ORDERS)
        c.stop_index = int(codec.STOP)
        c.unit_has_qty = [bool(v) for v in codec.UNIT_HAS_QTY]
        c.order_has_qty = [bool(v) for v in codec.ORDER_HAS_QTY]
    if columns is not None:
        c.type_widths = dict(columns["widths"])
        import nf_format
        c.columns_sha256 = nf_format.columns_hash(columns)
    if c.max_units != U or c.max_orders + 1 != O:
        raise ValueError("N flow is laid out for 25 unit and 11 order slots")
    if c.width % c.heads or (c.width // c.heads) % 8 or not 0 < c.rel_heads <= c.heads:
        raise ValueError("width/heads must give a head dim that is a multiple of 8; 0 < rel_heads <= heads")
    return c


# ----------------------------------------------------------------------------- static relation graph
def relation_table():
    """[N_NODES, N_NODES] long relation type of (query i, key j)."""
    R = torch.zeros(N_NODES, N_NODES, dtype=torch.long)
    idx = {n: i for i, n in enumerate(REL_NAMES)}
    kind = torch.full((N_NODES,), -1, dtype=torch.long)
    for ti, name in enumerate(TYPES):
        a, b = TYPE_RANGES[name]
        kind[a:b] = ti
    T = {n: i for i, n in enumerate(TYPES)}
    role = torch.full((N_NODES,), -1, dtype=torch.long)
    role[1], role[2] = 0, 1
    role[3:103], role[103:203] = 0, 1
    role[203:228], role[228:253] = 0, 1
    role[253:270] = 0                     # own shed / seeds
    is_tile = kind == T["TILE"]
    is_carrier = kind == T["CARRIER"]
    res = (kind == T["STOCK"]) | (kind == T["MARKET"]) | (kind == T["CONSTANT"]) | (kind == T["SHOP"])
    same_role = (role[:, None] == role[None, :]) & (role[:, None] >= 0)
    farm = kind == T["FARM"]
    stock = kind == T["STOCK"]
    mk = kind == T["MARKET"]
    shop = kind == T["SHOP"]
    anyn = torch.zeros(N_NODES, dtype=torch.bool)
    anyn[:CAL0 + N_CAL] = True
    mem = torch.zeros(N_NODES, dtype=torch.bool)
    mem[FAST_NODE] = mem[SLOW_NODE] = True
    cal = torch.zeros(N_NODES, dtype=torch.bool)
    cal[CAL0:CAL0 + N_CAL] = True
    pad = torch.zeros(N_NODES, dtype=torch.bool)
    pad[CAL0 + N_CAL:] = True
    # general relations first; specific ones below override them
    hub = (kind == T["GLOBAL"]) | (kind == T["CONTROL"]) | farm
    R[(hub[:, None] & anyn[None, :]) | (anyn[:, None] & hub[None, :])] = idx["hub"]
    R[(mem[:, None] & anyn[None, :]) | (anyn[:, None] & mem[None, :])] = idx["memory"]
    R[res[:, None] & res[None, :]] = idx["resource"]
    R[mk[:, None] & mk[None, :]] = idx["market_market"]
    R[(shop[:, None] & mk[None, :]) | (mk[:, None] & shop[None, :])] = idx["shop_market"]
    cal_res = res | is_tile | farm
    R[(cal[:, None] & cal_res[None, :]) | (cal_res[:, None] & cal[None, :])] = idx["cal_resource"]
    t = torch.arange(200)
    tr, ty, tx = t // 100, (t % 100) // 10, t % 10
    adj = (tr[:, None] == tr[None, :]) & ((ty[:, None] - ty[None, :]).abs() + (tx[:, None] - tx[None, :]).abs() == 1)
    mir = (tr[:, None] != tr[None, :]) & (ty[:, None] == ty[None, :]) & (tx[:, None] == tx[None, :])
    R[3:203, 3:203] = torch.where(adj, idx["tile_adj"], torch.where(mir, idx["tile_mirror"], 0))
    member = (farm[:, None] & (is_tile | is_carrier)[None, :]) | ((is_tile | is_carrier)[:, None] & farm[None, :])
    R[member & same_role] = idx["farm_member"]
    R[(is_carrier[:, None] & is_tile[None, :] | is_tile[:, None] & is_carrier[None, :]) & same_role] = idx["carrier_tile"]
    R[is_carrier[:, None] & is_carrier[None, :] & same_role] = idx["carrier_carrier"]
    own_car = is_carrier & (role == 0)
    R[(own_car[:, None] & stock[None, :]) | (stock[:, None] & own_car[None, :])] = idx["custody"]
    ci = torch.arange(N_CAL)
    R[CAL0:CAL0 + N_CAL, CAL0:CAL0 + N_CAL] = torch.where(ci[None, :] > ci[:, None], idx["cal_later"],
                                                          torch.where(ci[None, :] < ci[:, None], idx["cal_earlier"], 0))
    R[:, pad] = idx["pad"]
    R[pad, :] = idx["pad"]
    R[torch.arange(N_NODES), torch.arange(N_NODES)] = idx["self"]
    return R


# ----------------------------------------------------------------------------- layers
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w12 = nn.Linear(d, 2 * h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)


class RelBlock(nn.Module):
    """Pre-norm block: rel heads (additive relation bias) + global heads (no bias) + SwiGLU."""

    def __init__(self, d, heads, rel_heads, hidden, n_rel):
        super().__init__()
        self.h, self.hr, self.dh = heads, rel_heads, d // heads
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ffn = SwiGLU(d, hidden)
        self.rel_bias = nn.Parameter(torch.zeros(n_rel, rel_heads))

    def forward(self, x, R1h, pad_bias):
        """R1h: [N*N, n_rel] one-hot relation table (a matmul, not an index gather: the gather's backward serialises
        ~1e5 duplicate indices on GPU)."""
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        hr = self.hr
        rel = (R1h @ self.rel_bias.to(R1h.dtype)).view(N, N, hr).permute(2, 0, 1)
        bias = (rel + pad_bias).to(q.dtype)[None]                                  # [1, hr, N, N]
        a_rel = F.scaled_dot_product_attention(q[:, :hr], k[:, :hr], v[:, :hr], attn_mask=bias.expand(B, -1, -1, -1))
        if hr < self.h:
            a_glob = F.scaled_dot_product_attention(q[:, hr:], k[:, hr:], v[:, hr:])
            a = torch.cat((a_rel, a_glob), 1)
        else:
            a = a_rel
        x = x + self.o(a.transpose(1, 2).reshape(B, N, d))
        return x + self.ffn(self.n2(x))


class DecLayer(nn.Module):
    """Causal self-attention over decision positions + cross-attention to node states (+ optional bias) + SwiGLU."""

    def __init__(self, d, heads, hidden):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.n1, self.n2, self.n3 = RMSNorm(d), RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o1 = nn.Linear(d, d, bias=False)
        self.q2 = nn.Linear(d, d, bias=False)
        self.kv2 = nn.Linear(d, 2 * d, bias=False)
        self.o2 = nn.Linear(d, d, bias=False)
        self.ffn = SwiGLU(d, hidden)

    def memory(self, mem):
        B, N, d = mem.shape
        k, v = self.kv2(mem).view(B, N, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x, mem_kv, cross_bias=None, cache=None):
        B, P, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, P, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if cache is not None:
            if "k" in cache:
                k = torch.cat((cache["k"], k), 2)
                v = torch.cat((cache["v"], v), 2)
            cache["k"], cache["v"] = k, v
            a = F.scaled_dot_product_attention(q, k, v)          # one new position attends to the whole prefix
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o1(a.transpose(1, 2).reshape(B, P, d))
        q2 = self.q2(self.n2(x)).view(B, P, self.h, self.dh).transpose(1, 2)
        mk, mv = mem_kv
        a2 = F.scaled_dot_product_attention(q2, mk, mv, attn_mask=None if cross_bias is None else cross_bias.to(q2.dtype))
        x = x + self.o2(a2.transpose(1, 2).reshape(B, P, d))
        return x + self.ffn(self.n3(x))


# ----------------------------------------------------------------------------- network
class NFlowNet(nn.Module):
    def __init__(self, config: NFlowConfig):
        super().__init__()
        c = self.config = config
        d = c.width
        if not c.type_widths or not c.unit_has_qty:
            raise ValueError("resolve_config(config, codec, columns) first")
        self.inp = nn.ModuleDict({t: nn.Linear(int(c.type_widths[t]), d) for t in TYPES})
        self.slot_emb = nn.Embedding(N_NODES, d)
        self.absent_emb = nn.Parameter(torch.zeros(d))
        self.pos_x, self.pos_y = nn.Embedding(11, d), nn.Embedding(11, d)       # index 10 = absent
        self.cal_in = nn.Linear(6, d)
        self.in_norm = RMSNorm(d)
        self.register_buffer("R", relation_table(), persistent=False)
        self.register_buffer("R1h", F.one_hot(self.R, len(REL_NAMES)).float().view(N_NODES * N_NODES, -1),
                             persistent=False)
        pad = torch.zeros(N_NODES, N_NODES)
        pad[:, CAL0 + N_CAL:] = -1e4
        pad[torch.arange(N_NODES), torch.arange(N_NODES)] = 0.0
        self.register_buffer("pad_bias", pad, persistent=False)
        self.blocks = nn.ModuleList(RelBlock(d, c.heads, c.rel_heads, c.hidden, len(REL_NAMES)) for _ in range(c.blocks))
        # memory
        self.tok_emb_u = nn.Embedding(c.n_unit_tokens + 2, d)      # + ABSENT(-1 -> n) + BOS
        self.tok_emb_o = nn.Embedding(c.n_order_tokens + 2, d)
        self.qty_emb = nn.Embedding(c.n_qty + 1, d)                # n_qty = none
        self.lag_emb = nn.Embedding(2, d)
        self.pool = nn.Linear(4 * d, d)
        self.fast_cell = nn.GRUCell(2 * d, c.fast)
        self.slow_cell = nn.GRUCell(d, c.slow)
        self.worker_cell = nn.GRUCell(2 * d, c.worker)
        self.fast_out, self.slow_out = nn.Linear(c.fast, d), nn.Linear(c.slow, d)
        self.worker_out = nn.Linear(c.worker, d)
        # decoder
        self.qpos = nn.Embedding(U + O, d)
        self.unit_in = nn.Linear(27 + 12 + 41, d)
        self.order_in = nn.Linear(22 + 82, d)
        self.worker_q = nn.Linear(c.worker, d)
        self.carrier_q = nn.Linear(d, d, bias=False)
        self.prev_q = nn.Linear(d, d, bias=False)
        self.dec = nn.ModuleList(DecLayer(d, c.heads, c.hidden) for _ in range(c.dec_layers))
        self.dec_norm = RMSNorm(d)
        self.reach_bias = nn.Parameter(torch.zeros(c.dec_layers, ETA_BUCKETS + 1, c.heads))
        self.unit_head = nn.Linear(d, c.n_unit_tokens)
        self.unit_qty_head = nn.Linear(d, c.n_qty)
        self.order_head = nn.Linear(d, c.n_order_tokens)
        self.order_qty_head = nn.Linear(d, c.n_qty)
        self.qty_cond_u = nn.Embedding(c.n_unit_tokens, d)
        self.qty_cond_o = nn.Embedding(c.n_order_tokens, d)
        # aux heads (training targets only)
        self.wdl_head = nn.Sequential(RMSNorm(d + c.fast + c.slow), nn.Linear(d + c.fast + c.slow, d), nn.GELU(),
                                      nn.Linear(d, 3))
        self.next_unit_head = nn.Linear(d, c.n_unit_tokens)
        self.opp_unit_head = nn.Linear(d, c.n_unit_tokens)
        self.register_buffer("unit_qty_mask", torch.tensor(c.unit_has_qty, dtype=torch.bool), persistent=False)
        self.register_buffer("order_qty_mask", torch.tensor(c.order_has_qty, dtype=torch.bool), persistent=False)
        stop_only = torch.zeros(c.n_order_tokens, dtype=torch.bool)
        stop_only[c.stop_index] = True
        self.register_buffer("stop_only", stop_only, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        for b in self.blocks:
            nn.init.normal_(b.o.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))
            nn.init.normal_(b.ffn.w3.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ graph
    def calendar(self, step):
        """[B, N_CAL, 6] landmark features from the physical step (0..718)."""
        s = step.long()
        off = torch.stack((torch.zeros_like(s), 4 - s % 4, 24 - s % 24, torch.full_like(s, 48), torch.full_like(s, 96),
                           torch.full_like(s, 192), torch.full_like(s, 288), 718 - s), -1).float()
        tgt = s[:, None].float() + off
        beyond = (tgt > 718).float()
        tgt = tgt.clamp(max=718)
        return torch.stack((off / 720, tgt / 720, beyond, torch.log1p(off) / 7, torch.div(tgt, 24, rounding_mode="floor") / 30,
                            (718 - tgt) / 720), -1)

    def embed(self, x):
        """x: dict of [B, ...] step inputs -> [B, N_NODES, d] initial node states."""
        B = x["node_present"].shape[0]
        d = self.config.width
        parts = [self.inp[t](x["node_" + t].float()) for t in TYPES]
        ent = torch.cat(parts, 1)                                           # [B, 296, d]
        present = x["node_present"].bool()
        ent = torch.where(present[..., None], ent, self.absent_emb.to(ent.dtype))
        cp = x["carrier_pos"].long()
        cx = torch.where(cp[..., 0] >= 0, cp[..., 0], 10).clamp(0, 10)
        cy = torch.where(cp[..., 1] >= 0, cp[..., 1], 10).clamp(0, 10)
        car = self.pos_x(cx) + self.pos_y(cy)
        a, b = TYPE_RANGES["CARRIER"]
        ent = torch.cat((ent[:, :a], ent[:, a:b] + car, ent[:, b:]), 1)
        cal = self.cal_in(self.calendar(x["step"]))
        rest = torch.zeros(B, N_NODES - N_ENT, d, device=ent.device, dtype=ent.dtype)
        rest = torch.cat((rest[:, :CAL0 - N_ENT], cal.to(ent.dtype), rest[:, CAL0 - N_ENT + N_CAL:]), 1)
        h = torch.cat((ent, rest), 1) + self.slot_emb.weight[None]
        return self.in_norm(h)

    def encode_pre(self, x):
        """Stateless relational blocks 1..R-1 -> [B, N_NODES, d]."""
        h = self.embed(x)
        for blk in self.blocks[:-1]:
            h = blk(h, self.R1h, self.pad_bias)
        return h

    def summary(self, h):
        own_tiles = h[:, 3:103].mean(1)
        market = h[:, 270:279].mean(1)
        return self.pool(torch.cat((h[:, 0], h[:, 1], own_tiles, market), -1))

    def prev_action_emb(self, x):
        """Actor's own lag-1/lag-2 decisions: ([B, d] summary, [B, U, d] per worker)."""
        pu, po = x["prev_unit"].long(), x["prev_order"].long()             # [B,2,U,2], [B,2,O,2]
        nu, no = self.config.n_unit_tokens, self.config.n_order_tokens
        ut = torch.where(pu[..., 0] >= 0, pu[..., 0], nu)
        uq = torch.where(pu[..., 1] >= 0, pu[..., 1], self.config.n_qty)
        ue = self.tok_emb_u(ut) + self.qty_emb(uq) + self.lag_emb.weight[None, :, None]
        ue = ue * (pu[..., 0] >= 0)[..., None]
        ot = torch.where(po[..., 0] >= 0, po[..., 0], no)
        oq = torch.where(po[..., 1] >= 0, po[..., 1], self.config.n_qty)
        oe = (self.tok_emb_o(ot) + self.qty_emb(oq)) * (po[..., 0] >= 0)[..., None]
        per_worker = ue.sum(1)                                              # [B, U, d]
        total = ue.sum((1, 2)) / 8 + oe.sum((1, 2)) / 8
        return total, per_worker

    # ------------------------------------------------------------------ memory
    def initial_memory(self, B, device):
        c = self.config
        z = lambda n: torch.zeros(B, n, device=device)
        return {"fast": z(c.fast), "slow": z(c.slow), "worker": torch.zeros(B, U, c.worker, device=device),
                "day_sum": z(c.width), "day_n": torch.zeros(B, device=device), "day": torch.full((B,), -1, device=device)}

    def memory_step(self, mem, s, act, h_car, act_w, step, present_units):
        """One step of the three recurrent memories. s [B,d] graph summary, act [B,d] own previous action,
        h_car [B,U,d] own carrier nodes, act_w [B,U,d] per-worker previous command, step [B], present_units [B,U]."""
        day = torch.div(step, 24, rounding_mode="floor").float()
        new_day = (mem["day"] >= 0) & (day > mem["day"])
        day_mean = mem["day_sum"] / mem["day_n"].clamp(min=1)[:, None]
        slow_upd = self.slow_cell(day_mean, mem["slow"])
        slow = torch.where(new_day[:, None], slow_upd, mem["slow"])
        day_sum = torch.where(new_day[:, None], torch.zeros_like(mem["day_sum"]), mem["day_sum"]) + s.float()
        day_n = torch.where(new_day, torch.zeros_like(mem["day_n"]), mem["day_n"]) + 1
        fast = self.fast_cell(torch.cat((s, act), -1).float(), mem["fast"])
        w = mem["worker"]
        keep = torch.ones_like(present_units, dtype=torch.bool)
        keep[:, 1:] = ~new_day[:, None]
        w = torch.where((keep & present_units)[..., None], w, torch.zeros_like(w))
        B = w.shape[0]
        w_new = self.worker_cell(torch.cat((h_car, act_w), -1).float().reshape(B * U, -1), w.reshape(B * U, -1)).view(B, U, -1)
        w = torch.where(present_units[..., None], w_new, torch.zeros_like(w_new))
        return {"fast": fast, "slow": slow, "worker": w, "day_sum": day_sum, "day_n": day_n, "day": day}

    def encode_post(self, h, fast, slow, worker):
        """Relational block R with memory: [B, N, d]."""
        d = h.shape[-1]
        dt = h.dtype
        mem_nodes = torch.stack((self.fast_out(fast.to(dt)), self.slow_out(slow.to(dt))), 1)
        wo = self.worker_out(worker.to(dt))
        h = torch.cat((h[:, :OWN_CARRIER], h[:, OWN_CARRIER:OWN_CARRIER + U] + wo, h[:, OWN_CARRIER + U:FAST_NODE],
                       h[:, FAST_NODE:SLOW_NODE + 1] + mem_nodes, h[:, SLOW_NODE + 1:]), 1)
        return self.blocks[-1](h, self.R1h, self.pad_bias)

    # ------------------------------------------------------------------ decoder
    def dec_context(self, h, x, worker, act_w):
        """Per-step decoder context: memory K/V per layer, unit query features [B,U,d], order features [B,O,d],
        cross-attention bias per layer for unit queries [B,H,U,N]."""
        mem_kv = [layer.memory(h) for layer in self.dec]
        dt = h.dtype
        uf = torch.cat((x["unit_extra"].float(), x["x8_unit"].float()), -1)
        up = x["unit_pos"].long()
        ux = torch.where(up[..., 0] >= 0, up[..., 0], 10).clamp(0, 10)
        uy = torch.where(up[..., 1] >= 0, up[..., 1], 10).clamp(0, 10)
        unit_q = (self.qpos.weight[:U][None] + self.pos_x(ux) + self.pos_y(uy) + self.worker_q(worker.to(dt))
                  + self.carrier_q(h[:, OWN_CARRIER:OWN_CARRIER + U]) + self.prev_q(act_w.to(dt)))
        eta = eta_bucket(x["reach_eta"])                                     # [B, U, 100]
        eta = torch.where((up[..., 0] >= 0)[..., None], eta, torch.full_like(eta, ETA_BUCKETS))
        L_, E_, H_ = self.reach_bias.shape
        oh = F.one_hot(eta, E_).to(dt)                                       # one-hot matmul, not an index gather
        allb = (oh @ self.reach_bias.permute(1, 0, 2).reshape(E_, L_ * H_).to(dt)).view(*eta.shape, L_, H_)
        biases = [F.pad(allb[..., li, :].permute(0, 3, 1, 2), (OWN_TILE, N_NODES - OWN_TILE - 100))
                  for li in range(L_)]                                       # [B, H, U, N]
        return {"mem_kv": mem_kv, "unit_q": unit_q, "unit_f": uf, "biases": biases}

    def decision_emb(self, tok, qty, unit):
        """Embedding of decision labels (-1 token = ABSENT / none)."""
        c = self.config
        if unit:
            t = torch.where(tok >= 0, tok, c.n_unit_tokens)
            has = self.unit_qty_mask[tok.clamp(min=0)] & (tok >= 0)
            e = self.tok_emb_u(t)
        else:
            t = torch.where(tok >= 0, tok, c.n_order_tokens)
            has = self.order_qty_mask[tok.clamp(min=0)] & (tok >= 0)
            e = self.tok_emb_o(t)
        q = torch.where(has & (qty >= 0), qty, c.n_qty)
        return e + self.qty_emb(q)

    def queries_teacher(self, ctx, x):
        """All 36 decoder inputs under teacher forcing (labels shifted right)."""
        c = self.config
        dt = ctx["unit_q"].dtype
        ut, uq = x["unit_tok"].long(), x["unit_qty"].long()
        ot, oq = x["order_tok"].long(), x["order_qty"].long()
        prev_u = self.decision_emb(ut, uq, True)                            # [B, U, d]
        prev_o = self.decision_emb(ot, oq, False)                           # [B, O, d]
        bos = self.tok_emb_u.weight[c.n_unit_tokens + 1].expand(ut.shape[0], 1, -1)
        prev = torch.cat((bos, prev_u, prev_o[:, :-1]), 1).to(dt)           # [B, U+O, d]
        uq_in = ctx["unit_q"] + self.unit_in(torch.cat((x["unit_prefix"].float(), ctx["unit_f"]), -1)).to(dt)
        oq_in = self.qpos.weight[U:][None].to(dt) + self.order_in(torch.cat((x["order_prefix"].float(),
                                                                             x["order_cash"].float()), -1)).to(dt)
        return torch.cat((uq_in, oq_in), 1) + prev

    def run_decoder(self, q, ctx, caches=None, pos=None):
        """q [B, P, d] decoder inputs at positions pos..pos+P-1 (teacher: pos=0, P=36)."""
        x = q
        P = q.shape[1]
        p0 = 0 if pos is None else pos
        for li, layer in enumerate(self.dec):
            b = ctx["biases"][li]                                            # [B, H, U, N]
            if p0 < U:
                hi = min(U, p0 + P)
                bias = b[:, :, p0:hi]
                if p0 + P > U:
                    bias = torch.cat((bias, bias.new_zeros(bias.shape[0], bias.shape[1], p0 + P - U, N_NODES)), 2)
            else:
                bias = None
            x = layer(x, ctx["mem_kv"][li], bias, None if caches is None else caches[li])
        return self.dec_norm(x)

    def heads_teacher(self, hdec, x):
        ut, ot = x["unit_tok"].long(), x["order_tok"].long()
        hu, ho = hdec[:, :U], hdec[:, U:]
        unit_logits = self.unit_head(hu)
        unit_qty_logits = self.unit_qty_head(hu + self.qty_cond_u(ut.clamp(min=0)))
        order_logits = self.order_head(ho)
        order_logits = torch.cat((order_logits[:, :-1],
                                  order_logits[:, -1:].masked_fill(~self.stop_only, float("-inf"))), 1)
        order_qty_logits = self.order_qty_head(ho + self.qty_cond_o(ot.clamp(min=0)))
        return {"unit_logits": unit_logits, "unit_qty_logits": unit_qty_logits,
                "order_logits": order_logits, "order_qty_logits": order_qty_logits}

    def aux_heads(self, h, fast, slow):
        dt = h.dtype
        wdl = self.wdl_head(torch.cat((h[:, 0], fast.to(dt), slow.to(dt)), -1))
        return {"wdl_logits": wdl, "next_unit_logits": self.next_unit_head(h[:, OWN_CARRIER:OWN_CARRIER + U]),
                "opp_unit_logits": self.opp_unit_head(h[:, OPP_CARRIER:OPP_CARRIER + U])}

    # ------------------------------------------------------------------ training forward over a chunk
    def forward_chunk(self, x, mem, pre_fn=None, post_fn=None):
        """x: dict of [B, L, ...] tensors; mem: memory dict for the B streams at the chunk start (reset rows already
        zeroed by the caller). Returns (outputs dict of [B, L, ...], new memory)."""
        B, L = x["step"].shape
        flat = {k: v.reshape(B * L, *v.shape[2:]) for k, v in x.items() if v.dim() >= 2 and tuple(v.shape[:2]) == (B, L)}
        pre = (pre_fn or self.encode_pre)(flat)                             # [B*L, N, d]
        d = pre.shape[-1]
        pre_bl = pre.view(B, L, N_NODES, d)
        s = self.summary(pre).view(B, L, d)
        act, act_w = self.prev_action_emb(flat)
        act, act_w = act.view(B, L, d), act_w.view(B, L, U, d)
        present = (x["unit_pos"][..., 0] >= 0)
        present[..., 0] = True
        fasts, slows, workers = [], [], []
        for t in range(L):
            mem = self.memory_step(mem, s[:, t], act[:, t], pre_bl[:, t, OWN_CARRIER:OWN_CARRIER + U], act_w[:, t],
                                   x["step"][:, t], present[:, t])
            fasts.append(mem["fast"]); slows.append(mem["slow"]); workers.append(mem["worker"])
        fast = torch.stack(fasts, 1).reshape(B * L, -1)
        slow = torch.stack(slows, 1).reshape(B * L, -1)
        worker = torch.stack(workers, 1).reshape(B * L, U, -1)
        out = (post_fn or self.post_teacher)(pre, flat, fast, slow, worker, act_w.reshape(B * L, U, d))
        return {k: v.view(B, L, *v.shape[1:]) for k, v in out.items()}, mem

    def post_teacher(self, pre, flat, fast, slow, worker, act_w):
        h = self.encode_post(pre, fast, slow, worker)
        ctx = self.dec_context(h, flat, worker, act_w)
        hdec = self.run_decoder(self.queries_teacher(ctx, flat), ctx)
        return {**self.heads_teacher(hdec, flat), **self.aux_heads(h, fast, slow)}
