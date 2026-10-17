"""Untrained Qwen3.8-inspired economic-token CPU reference, not the live actor.

Primary equations: https://arxiv.org/html/2608.30320v1, §2.1.1 (1–11)
and §2.2 (30–34); model card:
https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/README.md.

Default: 16 blocks, global attention at 1-based 4/8/12/16, otherwise
shared-weight forward/reverse Gated DeltaNet; four-branch Gated Residual around
EACH mixer and MoE. Tokens are the existing fixed-order 29 economic candidates,
NOT time/text. Recurrence and convolution start empty on every call. Illegal
tokens have zero features, skip recurrence updates and never enter MoE losses.
These candidate rows are NOT a complete world representation. This version
uses action legality as token eligibility. Dedicated state/entity tokens are
a separate NEXT schema: token-validity and action-legality masks must differ,
only 29 action tokens emit logits, and the critic pool/unknowns must be declared.

Explicit adaptations: bidirectional GDN, noncausal gated full attention with
fixed-candidate-index RoPE, scaled heads/experts, top2 normalized routing plus a
sigmoid-gated shared expert, mean-branch readout and actor/critic heads. No QSA,
n-gram/text/vision embeddings, MTP, pretrained weights, Muon or trainer. The
Python loops are a correctness reference, NOT a fused-kernel/throughput claim.
Unique active parameters PER TOKEN are not FLOPs or the union touched by all
29 tokens. No old model/Adam/RNG/schema can be relabelled as this new lineage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from farm_neat.training.schedule_model import bound, file_sha
from farm_neat.policy.semantic import joint_policy_v3 as joint

SCHEMA = "farm-semantic-qwen-hybrid-reference-v1"
NUMERICS = "TORCH_FP32_SEMANTIC_QWEN_HYBRID_CPU_REFERENCE_V1"
WIDTH, TOKENS = 80, 29
ACTIONS, ACTION_IDS, ACTION_SCHEMA = joint.ACTIONS, joint.ACTION_IDS, joint.ACTION_SCHEMA
REPRESENTATION_SCHEMA = joint.REPRESENTATION_SCHEMA
initial_orders, pack, apply_command, validate_inputs = (
    joint.initial_orders, joint.pack, joint.apply_command, joint.validate_inputs)


@dataclass(frozen=True)
class Config:
    width: int = 144
    layers: int = 16
    heads: int = 4
    experts: int = 64
    top_k: int = 2
    expert_hidden: int = 128
    branches: int = 4
    conv_size: int = 4
    rms_epsilon: float = 1e-6

    def __post_init__(self):
        counts = (self.width, self.layers, self.heads, self.experts, self.top_k,
                  self.expert_hidden, self.branches, self.conv_size)
        if (any(type(v) is not int or v < 1 for v in counts)
                or self.width % 8 or self.width % self.heads
                or (self.width // self.heads) % 2
                or self.layers % 4 or self.top_k != 2 or self.experts < self.top_k
                or self.branches != 4 or self.conv_size > TOKENS
                or type(self.rms_epsilon) is not float
                or not math.isfinite(self.rms_epsilon) or self.rms_epsilon <= 0):
            raise ValueError("four-branch/every-fourth-global, even-head top2 reference config required")


DEFAULT = Config()


def sources():
    return {**joint.sources(), "src/farm_neat/policy/qwen_hybrid.py": file_sha(__file__)}


def architecture(config=DEFAULT):
    return bound({"schema": SCHEMA, "configuration": asdict(config),
        "candidate_tokens": TOKENS, "candidate_width": WIDTH,
        "action_schema_sha256": ACTION_SCHEMA["content_sha256"],
        "representation_schema_sha256": REPRESENTATION_SCHEMA["content_sha256"],
        "global_attention_layers_1based": list(range(4, config.layers+1, 4)),
        "candidate_order": "EXACT_EXISTING_ACTION_IDS; NOT_ENVIRONMENT_TIME",
        "token_mask_contract": "CURRENT_ACTION_LEGALITY_IS_TOKEN_ELIGIBILITY",
        "dedicated_state_tokens": False, "complete_world_representation": False,
        "next_state_token_schema": "SEPARATE_VALIDITY_AND_LEGALITY; ACTOR_29_ACTION_TOKENS; DECLARED_CRITIC_POOL_AND_UNKNOWNS",
        "gdn": "BIDIRECTIONAL_SHARED_WEIGHTS_ZERO_STATE_EACH_CALL",
        "global_attention": "NONCAUSAL_LEGAL_KEYS_ONLY_FULL_HEAD_ROPE",
        "residual": "PAPER_EQ30_34_SEPARATE_MIXER_AND_MOE_FOUR_BRANCH_GR",
        "routing": "LEARNED_SOFTMAX_STABLE_TOP2_RENORMALIZED_NO_TOKEN_DROPPING",
        "shared_experts_per_layer": 1, "shared_expert_gate": "LEARNED_SIGMOID",
        "router_auxiliary": "RETURNED_SEPARATELY; NOT_AUTOMATIC_PPO_CREDIT",
        "readout": "MEAN_BRANCHES_ZERO_CENTERED_RMS; MEAN_LEGAL_TOKENS_CRITIC",
        "numerics": NUMERICS, "status": "REFERENCE_GRAPH_ONLY_NOT_TRAINED",
        "old_optimizer_compatible": False, "gpu_throughput_measured": False,
        "omitted": ["QSA", "NGRAM", "MTP", "TEXT_VISION", "PRETRAINED_WEIGHTS", "MUON"]})


class ZeroCenteredRMSNorm(nn.Module):
    """RMS is along the last dimension; gains may be branch/head-specific."""
    def __init__(self, shape, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(shape))
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True)+self.epsilon) * (1+self.weight)


class GatedResidual(nn.Module):
    """Equations30–34; this replaces, rather than precedes, block pre-norm."""
    def __init__(self, config):
        super().__init__()
        self.branches, self.width = config.branches, config.width
        n = self.branches * self.width
        self.norm = ZeroCenteredRMSNorm((self.branches, self.width), config.rms_epsilon)
        self.read_down = nn.Linear(n, self.width//8, bias=False)
        self.read_up = nn.Linear(self.width//8, n, bias=False)
        self.write_gate = nn.Linear(n, self.branches, bias=False)

    def read(self, residual):
        normalized = self.norm(residual)
        flat = normalized.flatten(-2)
        gate = torch.sigmoid(self.read_up(F.silu(self.read_down(flat)/self.branches)))
        x = (gate.reshape_as(residual)*normalized).mean(-2)
        write = 2*torch.sigmoid(self.write_gate(flat)/self.branches)
        return x, write

    @staticmethod
    def write(residual, value, gate, enabled):
        result = residual + gate[..., None]*value[..., None, :]
        return result.masked_fill(~enabled[..., None, None], 0)


def gated_delta_recurrence(q, k, v, alpha, beta, enabled, *, reverse=False):
    """Literal equations1–4, no supplied/carried environment state.

    Inputs [batch,tokens,heads,key/value_dim], scalar gates [batch,tokens,heads].
    Disabled positions are true identity transitions, including no decay.
    Float64 is accepted here solely for independent numerical-reference tests.
    """
    if (q.ndim != 4 or q.shape[1] < 1 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]
            or alpha.shape != q.shape[:3] or beta.shape != alpha.shape
            or enabled.shape != q.shape[:2] or enabled.dtype != torch.bool
            or any(t.device != q.device or t.dtype != q.dtype for t in (k, v, alpha, beta))
            or enabled.device != q.device or q.dtype not in (torch.float32, torch.float64)
            or type(reverse) is not bool):
        raise ValueError("paired GDN head streams/gates and bool mask required")
    batch, length, heads, key_dim = q.shape
    state = q.new_zeros(batch, heads, key_dim, v.shape[-1])
    output = [None]*length
    for index in (range(length-1, -1, -1) if reverse else range(length)):
        valid = enabled[:, index, None, None]
        query = q[:, index].masked_fill(~valid, 0)
        key = k[:, index].masked_fill(~valid, 0)
        value = v[:, index].masked_fill(~valid, 0)
        decay = torch.where(valid.squeeze(-1), alpha[:, index], 1)
        strength = torch.where(valid.squeeze(-1), beta[:, index], 0)
        decayed = decay[..., None, None]*state
        error = value-(decayed*key[..., :, None]).sum(-2)
        updated = decayed + strength[..., None, None]*key[..., :, None]*error[..., None, :]
        state = torch.where(valid[..., None], updated, state)
        output[index] = (state*query[..., :, None]).sum(-2).masked_fill(~valid, 0)
    return torch.stack(output, dim=1), state


class BidirectionalGDN(nn.Module):
    def __init__(self, config):
        super().__init__()
        w, h = config.width, config.heads
        self.width, self.heads, self.head_dim = w, h, w//h
        self.conv_size, self.epsilon = config.conv_size, config.rms_epsilon
        self.qkv = nn.Linear(w, 3*w, bias=False)
        self.conv = nn.Conv1d(3*w, 3*w, self.conv_size, groups=3*w, bias=False)
        self.beta = nn.Linear(w, h, bias=False)
        self.alpha = nn.Linear(w, h, bias=True)
        self.A = nn.Parameter(torch.zeros(h))
        self.norm = ZeroCenteredRMSNorm((h, w//h), config.rms_epsilon)
        self.output_gate = nn.Linear(w, w, bias=False)
        self.output = nn.Linear(w, w, bias=False)

    def forward(self, x, enabled, *, return_states=False):
        x = x.masked_fill(~enabled[..., None], 0)
        batch, length, _ = x.shape
        projected = self.qkv(x).transpose(1, 2)
        alpha = torch.exp(-self.A.exp()*F.softplus(self.alpha(x)))
        beta = torch.sigmoid(self.beta(x))
        results, states = [], []
        for reverse in (False, True):
            stream = projected.flip(-1) if reverse else projected
            convolved = F.conv1d(stream, self.conv.weight, padding=self.conv_size-1,
                                 groups=3*self.width)[..., :length]
            if reverse:
                convolved = convolved.flip(-1)
            q, k, v = F.silu(convolved.transpose(1, 2)).chunk(3, dim=-1)
            q = F.normalize(q.reshape(batch, length, self.heads, self.head_dim), dim=-1, eps=self.epsilon)
            k = F.normalize(k.reshape_as(q), dim=-1, eps=self.epsilon)
            v = v.reshape_as(q)
            y, state = gated_delta_recurrence(q, k, v, alpha, beta, enabled, reverse=reverse)
            results.append(y); states.append(state)
        y = self.norm((results[0]+results[1])*0.5).flatten(-2)
        y = self.output(torch.sigmoid(self.output_gate(x))*y).masked_fill(~enabled[..., None], 0)
        return (y, tuple(states)) if return_states else y


class GlobalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.head_dim = config.heads, config.width//config.heads
        self.qkv = nn.Linear(config.width, 3*config.width, bias=False)
        self.q_norm = ZeroCenteredRMSNorm((self.heads, self.head_dim), config.rms_epsilon)
        self.k_norm = ZeroCenteredRMSNorm((self.heads, self.head_dim), config.rms_epsilon)
        self.output_gate = nn.Linear(config.width, config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)

    def _rope(self, x):
        # Fixed semantic-slot positions, not positions in an environment history.
        index = torch.arange(self.head_dim//2, device=x.device, dtype=x.dtype)
        inverse = 10000.**(-2*index/self.head_dim)
        angle = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]*inverse
        cosine, sine = angle.cos()[None, :, None, :], angle.sin()[None, :, None, :]
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even*cosine-odd*sine, even*sine+odd*cosine), dim=-1).flatten(-2)

    def forward(self, x, enabled):
        x = x.masked_fill(~enabled[..., None], 0)
        batch, length, width = x.shape
        q, k, v = [z.reshape(batch, length, self.heads, self.head_dim) for z in self.qkv(x).chunk(3, -1)]
        q, k = self._rope(self.q_norm(q)), self._rope(self.k_norm(k))
        logits = torch.matmul(q.transpose(1, 2), k.permute(0, 2, 3, 1))/math.sqrt(self.head_dim)
        probabilities = logits.masked_fill(~enabled[:, None, None, :], -torch.inf).softmax(-1)
        y = torch.matmul(probabilities, v.transpose(1, 2)).transpose(1, 2).reshape(batch, length, width)
        return self.output(torch.sigmoid(self.output_gate(x))*y).masked_fill(~enabled[..., None], 0)


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate_up = nn.Linear(width, 2*hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate)*up)


class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k, self.expert_count = config.top_k, config.experts
        self.router = nn.Linear(config.width, config.experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(config.width, config.expert_hidden) for _ in range(config.experts)])
        self.shared = SwiGLU(config.width, config.expert_hidden)
        self.shared_gate = nn.Linear(config.width, 1, bias=False)

    def forward(self, x, enabled):
        flat = x.reshape(-1, x.shape[-1])
        valid_positions = enabled.flatten().nonzero().flatten()
        tokens = flat[valid_positions]
        if len(tokens) == 0:
            raise ValueError("at least one enabled token required")
        logits = self.router(tokens)
        probabilities = logits.softmax(-1)
        # Stable ties have a specified slot order; assignment is learned, not family-coded.
        selected = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :self.top_k]
        weights = probabilities.gather(1, selected)
        weights = weights/weights.sum(-1, keepdim=True)
        combined = torch.zeros_like(tokens)
        for expert_index, expert in enumerate(self.experts):
            token_index, route_index = (selected == expert_index).nonzero(as_tuple=True)
            if len(token_index):
                value = expert(tokens[token_index])*weights[token_index, route_index, None]
                combined = combined.index_add(0, token_index, value)
        combined = combined + torch.sigmoid(self.shared_gate(tokens))*self.shared(tokens)
        output = torch.zeros_like(flat).index_copy(0, valid_positions, combined).reshape_as(x)
        counts = torch.bincount(selected.flatten(), minlength=self.expert_count)
        fraction = counts.to(x.dtype)/(len(tokens)*self.top_k)
        load_balance = self.expert_count*(probabilities.mean(0)*fraction.detach()).sum()
        z_loss = logits.logsumexp(-1).square().mean()
        indices = torch.full((len(flat), self.top_k), -1, device=x.device, dtype=torch.long)
        indices = indices.index_copy(0, valid_positions, selected).reshape(*enabled.shape, self.top_k)
        route_weights = x.new_zeros(len(flat), self.top_k).index_copy(0, valid_positions, weights)
        return output, {"indices": indices, "weights": route_weights.reshape(*enabled.shape, self.top_k),
            "expert_counts": counts, "valid_tokens": len(tokens),
            "load_balance_loss": load_balance, "router_z_loss": z_loss}


class Block(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.kind = "GLOBAL" if (index+1) % 4 == 0 else "BIDIRECTIONAL_GDN"
        self.mixer_gr = GatedResidual(config)
        self.mixer = GlobalAttention(config) if self.kind == "GLOBAL" else BidirectionalGDN(config)
        self.moe_gr = GatedResidual(config)
        self.moe = SparseMoE(config)

    def forward(self, residual, enabled):
        x, write = self.mixer_gr.read(residual)
        residual = self.mixer_gr.write(residual, self.mixer(x, enabled), write, enabled)
        x, write = self.moe_gr.read(residual)
        y, routing = self.moe(x, enabled)
        return self.moe_gr.write(residual, y, write, enabled), routing


class Graph(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input = nn.Linear(WIDTH, config.width, bias=False)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.layers)])
        self.final_norm = ZeroCenteredRMSNorm(config.width, config.rms_epsilon)
        self.actor_head = nn.Linear(config.width, 1)
        self.value_head = nn.Linear(config.width, 1)

    def forward_with_aux(self, rows, enabled):
        if (rows.ndim != 3 or rows.shape[0] < 1 or rows.shape[1:] != (TOKENS, WIDTH)
                or rows.dtype != torch.float32 or rows.device.type != "cpu"
                or enabled.dtype != torch.bool or enabled.shape != rows.shape[:2]
                or enabled.device != rows.device or not bool(enabled[:, 0].all())
                or not bool(torch.isfinite(rows).all())
                or next(self.parameters()).device != rows.device
                or torch.is_autocast_enabled("cpu")):
            raise ValueError("CPU FP32 finite29x80, legal NO_EDIT and no autocast required")
        x = self.input(rows.masked_fill(~enabled[..., None], 0))
        residual = x[..., None, :].expand(*x.shape[:2], self.config.branches, self.config.width)
        routes = []
        for block in self.blocks:
            residual, routing = block(residual, enabled)
            routes.append(routing)
        h = self.final_norm(residual.mean(-2)).masked_fill(~enabled[..., None], 0)
        scores = self.actor_head(h).squeeze(-1).masked_fill(~enabled, 0)
        pooled = h.sum(1)/enabled.sum(1, keepdim=True)
        values = self.value_head(pooled).squeeze(-1)
        auxiliary = {"layers": tuple(routes),
            "load_balance_loss": torch.stack([r["load_balance_loss"] for r in routes]).mean(),
            "router_z_loss": torch.stack([r["router_z_loss"] for r in routes]).mean()}
        return scores, values, auxiliary

    def forward(self, rows, enabled):
        scores, values, _ = self.forward_with_aux(rows, enabled)
        return scores, values


def torch_graph(weights=None, numerics=NUMERICS, *, device="cpu", config=DEFAULT):
    """Same functional actor/critic seam, explicitly NEW CPU/reference numerics.

    `device='meta'` constructs full-size shapes without allocating any weights;
    it is accounting only and cannot run data. No checkpoint or optimizer I/O.
    """
    if numerics != NUMERICS or str(device) not in ("cpu", "meta") or not isinstance(config, Config):
        raise ValueError("explicit new CPU/reference lineage; no legacy/GPU fallback")
    with torch.device(device):
        graph = Graph(config).float()
    if weights is not None:
        if str(device) == "meta":
            raise ValueError("meta is shape accounting, not checkpoint loading")
        expected = graph.state_dict()
        if set(weights) != set(expected):
            raise ValueError("exact new hybrid parameter keys required")
        tensors = {name: torch.as_tensor(value) for name, value in weights.items()}
        if any(t.shape != expected[n].shape or t.dtype != torch.float32
               or t.device.type != "cpu" or not bool(torch.isfinite(t).all()) for n, t in tensors.items()):
            raise ValueError("exact finite new CPU FP32 tensor shapes required")
        graph.load_state_dict(tensors, strict=True)
    return graph


def parameter_report(graph):
    """Actual registered tensors, including every always-on parameter."""
    total = sum(p.numel() for p in graph.parameters())
    routed = sum(p.numel() for n, p in graph.named_parameters() if ".moe.experts." in n)
    per_expert = 3*graph.config.width*graph.config.expert_hidden
    if routed != graph.config.layers*graph.config.experts*per_expert:
        raise ValueError("routed expert accounting changed")
    always = total-routed  # Includes shared expert, router, GR, all heads/norms.
    return {"total_trainable_parameters": total, "always_on_including_shared": always,
        "routed_parameters_total": routed, "routed_expert_parameters": per_expert,
        "active_unique_parameters_per_valid_token": always+graph.config.layers*graph.config.top_k*per_expert,
        "not_whole_candidate_set_active_union": True, "not_flops_or_runtime": True,
        "gdn_shared_weight_directions": 2, "parameter_tensors": len(tuple(graph.parameters()))}


def masked_distribution(scores, values, mask, temperature=1.0):
    if (type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0
            or scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] != TOKENS or scores.dtype != torch.float32
            or scores.device.type != "cpu" or mask.shape != scores.shape or mask.dtype != torch.bool
            or mask.device != scores.device or values.device != scores.device
            or values.dtype != torch.float32 or values.shape != (scores.shape[0],)
            or not bool(mask[:, 0].all()) or not bool(torch.isfinite(scores).all())
            or not bool(torch.isfinite(values).all())):
        raise ValueError("new CPU actor/critic and legal nonempty29 categorical required")
    logits = (scores/temperature).masked_fill(~mask, -torch.inf)
    if not bool(torch.isfinite(logits[mask]).all()):
        raise ValueError("temperature-scaled legal logits must remain finite")
    logp = logits.log_softmax(-1)
    return scores, values, mask, logp.exp(), logp
