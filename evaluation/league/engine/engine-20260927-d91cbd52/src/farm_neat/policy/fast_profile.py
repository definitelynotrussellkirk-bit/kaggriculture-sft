"""Explicit NEW execution profile for the full-economic MoE: fused batched kernels.

Owner: full-economic runtime/throughput maintainer. Status: NEW numerical
profile ``TORCH_FP32_FULL_ECONOMIC_MOE_FAST_V1``; not the historical
``TORCH_FP32_FULL_ECONOMIC_MOE_V1`` reference and not bit-identical to it.

The v1 kernels change arithmetic only. The explicit v2 profile additionally
supports remaining-window traces: only deadlines before the actual turn are
masked, including nested replacements. All other economic support and the v1
sampler default remain unchanged. No weights, prior, Adam or schedule reset.

* Gated DeltaNet runs as a chunked WY-form parallel scan (Yang et al. 2024)
  in IEEE FP32 GEMMs with FP64 decay ratios, instead of a per-token Python
  loop. Same equations 1-4; reassociated reductions. Autograd is native, so
  training needs no per-block recomputation.
* SparseMoE routes with one stable sort per layer and one host count read,
  computing each expert on its exact token set; outputs are combined in a
  fixed order (deterministic). No token dropping; same stable top-2 argsort,
  renormalized weights, shared sigmoid-gated expert and auxiliaries.
* The typed conditional-factor sampler runs all lanes in lockstep, grouped
  exactly like ``full_economic_batched_replay`` (path/typed choices/temperature
  /budget/as-of mask). Each lane keeps a private ``numpy.random.PCG64`` stream and
  inverse-CDF sampling; the v2 trace explicitly records its turn. This is a NEW
  sampler stream, not the historical per-lane ``torch.Generator``.

Preregistered tolerances live in ``TOLERANCES``: cross-profile canaries versus
the frozen reference (reported, never a gate for reference receipts) and the
within-profile collect/replay parity gate used by the fast fitter. Dependencies
are bound by versioned hashes and read through private views; no live module is
monkeypatched. Legacy source identities remain separate.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import FunctionType

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "farm-full-economic-fast-profile-v2"
NUMERICS = "TORCH_FP32_FULL_ECONOMIC_MOE_FAST_V1"
LEGACY_PREIMAGES = {
    "src/farm_neat/policy/network.py": "ec805f612e6deb6451701a6bfd6e18b0459dad94b535a6f132c9cb3c9e1d6a91",
    "src/farm_neat/policy/qwen_hybrid.py": "cc85d803c1c5a9ea4a569111a615ff43d260feb15a82dcfd1f89068150c5403d",
    "src/farm_neat/economy/action_space.py": "2df38e7072ab338e8d5008b0ed5515754061627c1cda3250cfe629267ca6d3ee",
    "src/farm_neat/rollout/batched_replay.py": "6c201ed69dcfbdf68118c374c86184e05153aafc9abebe3bfba50b91f7495dca",
}
PREIMAGES = {**LEGACY_PREIMAGES,
    "src/farm_neat/policy/network.py": "ec805f612e6deb6451701a6bfd6e18b0459dad94b535a6f132c9cb3c9e1d6a91",
    "src/farm_neat/rollout/batched_replay.py": "6c201ed69dcfbdf68118c374c86184e05153aafc9abebe3bfba50b91f7495dca"}
TOLERANCES = {
    # Same-profile: collection (no-grad) versus fitter replay (grad) at the SAME
    # cohort padding. This is the admission gate for fast-profile data.
    "within_profile_logp_atol": 2e-5, "within_profile_value_atol": 2e-5,
    # Cross-profile canary: fast versus frozen reference on identical inputs.
    # Reported for every fit; refuses only above the coarse bound.
    "cross_profile_logp_atol": 5e-4, "cross_profile_value_atol": 5e-4,
    # Padding-bucket invariance (fast profile only): same lane, longer padding.
    "padding_bucket_atol": 5e-5,
}
CHUNK = 64


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity():
    """Standard-library identity of this profile and its pinned dependencies."""
    for name, expected in PREIMAGES.items():
        if _sha(ROOT/name) != expected:
            raise ValueError("unsupported original dependency for the fast profile: " + name)
    names = ("src/farm_neat/policy/fast_profile.py", "tests/unit/test_fast_profile.py")
    payload = {"schema": SCHEMA, "numerics": NUMERICS,
        "source_pins": {**PREIMAGES, **{n: _sha(ROOT/n) for n in names}},
        "recurrence": "CHUNKED_WY_PARALLEL_SCAN_FP32_GEMM_FP64_DECAY_RATIOS_ZERO_STATE_PER_CALL",
        "stacking": "COHORTS_CONCATENATED_ALONG_BATCH_PADDED_TO_LONGEST_BUCKET; OPTIONAL_PER_BLOCK_RECOMPUTATION",
        "moe": "STABLE_SORT_ONE_HOST_COUNT_EXACT_EXPERT_TOKEN_SETS_FIXED_ORDER_COMBINE",
        "sampler": "LOCKSTEP_PADDED_POSITIONS_DEVICE_CACHED_SPECS_PER_LANE_PCG64_INVERSE_CDF; CUDA_GRAPH_STATIC_KMAX_VARIANT",
        "support": "LEGACY_V1_OR_EXPLICIT_REMAINING_WINDOW_V2; WITNESS_BINDS_TIME",
        "tolerances": TOLERANCES, "chunk": CHUNK,
        "historical_bit_exact_claim": False, "parameter_or_optimizer_change": False,
        "training_admission": False}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {**payload, "content_sha256": digest}


# --------------------------------------------------------------------------
# Chunked gated delta recurrence (same equations as the reference loop)
# --------------------------------------------------------------------------
def chunked_gated_delta_recurrence(q, k, v, alpha, beta, enabled, *, reverse=False, chunk=CHUNK):
    """Inputs [B,T,H,dk]/[B,T,H,dv], gates [B,T,H], mask [B,T]; returns (out [B,T,H,dv], state [B,H,dk,dv]).

    Disabled tokens are exact identity transitions (no decay, no update, zero
    output), as in the reference. Within a chunk the recurrence is solved as a
    lower-triangular system; chunks are carried sequentially.
    """
    import torch
    if (q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]
            or alpha.shape != q.shape[:3] or beta.shape != alpha.shape or enabled.shape != q.shape[:2]
            or enabled.dtype != torch.bool or type(reverse) is not bool
            or q.dtype not in (torch.float32, torch.float64)
            or any(t.dtype != q.dtype or t.device != q.device for t in (k, v, alpha, beta))):
        raise ValueError("paired GDN head streams/gates and bool mask required")
    if reverse:
        q, k, v, alpha, beta, enabled = (x.flip(1) for x in (q, k, v, alpha, beta, enabled))
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    valid = enabled[..., None]                                      # [B,T,1]
    q = q.masked_fill(~valid[..., None], 0).transpose(1, 2)         # [B,H,T,dk]
    k = k.masked_fill(~valid[..., None], 0).transpose(1, 2)
    v = v.masked_fill(~valid[..., None], 0).transpose(1, 2)         # [B,H,T,dv]
    beta = beta.masked_fill(~valid, 0).transpose(1, 2)              # [B,H,T]
    # log decay in FP64; disabled tokens keep the state (alpha == 1 -> log 0).
    # exact FP32 zeros are clamped so log/1/alpha never produce inf*0 in backward
    log_alpha = alpha.double().masked_fill(~valid, 1.0).clamp(min=1e-300).log().transpose(1, 2)  # [B,H,T]
    pad = (-T) % chunk
    if pad:
        z = lambda x, fill=0: torch.nn.functional.pad(x, (0, 0, 0, pad) if x.ndim == 4 else (0, pad), value=fill)
        q, k, v, beta, log_alpha = z(q), z(k), z(v), z(beta), z(log_alpha)
    n = (T + pad)//chunk
    C = chunk
    q = q.reshape(B, H, n, C, dk); k = k.reshape(B, H, n, C, dk); v = v.reshape(B, H, n, C, dv)
    beta = beta.reshape(B, H, n, C); log_alpha = log_alpha.reshape(B, H, n, C)
    g = log_alpha.cumsum(-1)                                        # [B,H,n,C] cumulative within chunk
    # ratios gamma_t/gamma_i for i<=t (strictly lower for L; full lower incl. diag for Q-K)
    diff = g[..., :, None] - g[..., None, :]                        # [B,H,n,C,C] log(gamma_t/gamma_i)
    tril = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    strict = tril & ~torch.eye(C, dtype=torch.bool, device=q.device)
    ratio = diff.masked_fill(~tril, -math.inf).exp().to(q.dtype)    # zero above diagonal
    gamma = g.exp().to(q.dtype)                                     # [B,H,n,C]
    KK = torch.matmul(k, k.transpose(-1, -2))                       # [B,H,n,C,C]
    L = (ratio * KK).masked_fill(~strict, 0)                        # strictly lower
    A = torch.eye(C, dtype=q.dtype, device=q.device) + beta[..., :, None] * L
    QK = torch.matmul(q, k.transpose(-1, -2)) * ratio               # [B,H,n,C,C] lower incl. diag
    state = q.new_zeros(B, H, dk, dv)
    outputs = []
    for c in range(n):
        kc, vc, qc = k[:, :, c], v[:, :, c], q[:, :, c]
        rhs = beta[:, :, c, :, None] * (vc - gamma[:, :, c, :, None] * torch.matmul(kc, state))   # [B,H,C,dv]
        u = torch.linalg.solve_triangular(A[:, :, c], rhs, upper=False, unitriangular=True)      # [B,H,C,dv]
        out = gamma[:, :, c, :, None] * torch.matmul(qc, state) + torch.matmul(QK[:, :, c], u)
        outputs.append(out)
        carry = ratio[:, :, c, -1, :, None]                         # gamma_C/gamma_i, FP64-derived
        state = gamma[:, :, c, -1, None, None] * state + torch.matmul((kc * carry).transpose(-1, -2), u)
    out = torch.cat(outputs, dim=2)[:, :, :T].transpose(1, 2)       # [B,T,H,dv]
    out = out.masked_fill(~valid[..., None], 0)
    if reverse:
        out = out.flip(1)
    return out, state


# --------------------------------------------------------------------------
# Fast MoE dispatch (same routing semantics and auxiliaries as SparseMoE)
# --------------------------------------------------------------------------
def moe_forward(moe, x, enabled):
    import torch
    flat = x.reshape(-1, x.shape[-1])
    valid_positions = enabled.flatten().nonzero().flatten()
    tokens = flat[valid_positions]
    T = tokens.shape[0]
    if T == 0:
        raise ValueError("at least one enabled token required")
    E, top_k = moe.expert_count, moe.top_k
    logits = moe.router(tokens)
    probabilities = logits.softmax(-1)
    selected = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :top_k]
    weights = probabilities.gather(1, selected)
    weights = weights/weights.sum(-1, keepdim=True)
    flat_e = selected.reshape(-1)                                    # [T*k], pair p = t*k + r
    order = torch.argsort(flat_e, stable=True)
    counts = torch.bincount(flat_e, minlength=E)
    host_counts = counts.tolist()                                    # one host synchronization per layer
    sorted_tokens = tokens[order // top_k]                           # [T*k, W]
    pieces, start = [], 0
    for e, count in enumerate(host_counts):
        if count:
            pieces.append(moe.experts[e](sorted_tokens[start:start+count]))
        start += count
    y_sorted = torch.cat(pieces, 0) if pieces else tokens.new_zeros(0, tokens.shape[-1])
    y = torch.empty_like(y_sorted).index_copy(0, order, y_sorted)   # back to pair order
    y = (y.reshape(T, top_k, -1) * weights[..., None]).sum(1)        # fixed combine order
    combined = y + torch.sigmoid(moe.shared_gate(tokens))*moe.shared(tokens)
    output = torch.zeros_like(flat).index_copy(0, valid_positions, combined).reshape_as(x)
    fraction = counts.to(x.dtype)/(T*top_k)
    load_balance = E*(probabilities.mean(0)*fraction.detach()).sum()
    z_loss = logits.logsumexp(-1).square().mean()
    indices = torch.full((len(flat), top_k), -1, device=x.device, dtype=torch.long)
    indices = indices.index_copy(0, valid_positions, selected).reshape(*enabled.shape, top_k)
    route_weights = x.new_zeros(len(flat), top_k).index_copy(0, valid_positions, weights)
    return output, {"indices": indices, "weights": route_weights.reshape(*enabled.shape, top_k),
        "expert_counts": counts, "valid_tokens": T,
        "load_balance_loss": load_balance, "router_z_loss": z_loss}


# --------------------------------------------------------------------------
# Private views binding the fast kernels into the original Graph.encode
# --------------------------------------------------------------------------
class _MoEView:
    def __init__(self, moe):
        self._moe = moe
    def __getattr__(self, name):
        return getattr(self._moe, name)
    def __call__(self, x, enabled):
        return moe_forward(self._moe, x, enabled)


class _GDNView:
    def __init__(self, mixer, forward):
        self._mixer, self._forward = mixer, forward
    def __getattr__(self, name):
        return getattr(self._mixer, name)
    def __call__(self, x, enabled, *, return_states=False):
        return self._forward(self, x, enabled, return_states=return_states)


class _BlockView:
    def __init__(self, block, mixer, forward):
        self._block, self.mixer, self.moe, self._forward = block, mixer, _MoEView(block.moe), forward
    def __getattr__(self, name):
        return getattr(self._block, name)
    def __call__(self, residual, enabled):
        return self._forward(self, residual, enabled)


class _CheckpointedBlock:
    """Per-block recomputation under autograd so stacked cohorts fit in VRAM; no-grad calls pass through."""
    def __init__(self, view):
        self._view = view
    def __getattr__(self, name):
        return getattr(self._view, name)
    def __call__(self, residual, enabled):
        import torch
        if not torch.is_grad_enabled():
            return self._view(residual, enabled)
        from torch.utils.checkpoint import checkpoint
        return checkpoint(self._view, residual, enabled, use_reentrant=False, preserve_rng_state=False)


class FastEncoder:
    """Bind once per graph; call like Graph.encode. Reads live parameters; no copies.

    ``checkpoint_blocks`` recomputes each block during backward (same arithmetic,
    same outputs and gradients; activation memory drops from O(blocks) to O(1) blocks).
    """
    def __init__(self, graph, *, expected_identity, checkpoint_blocks=False):
        if expected_identity != identity():
            raise ValueError("externally frozen fast-profile identity required")
        if type(checkpoint_blocks) is not bool:
            raise ValueError("explicit checkpoint_blocks flag required")
        from farm_neat.policy import network as policy
        hybrid = policy.hybrid
        if type(graph) is not policy.Graph or "encode" in graph.__dict__:
            raise ValueError("unmodified original full-economic Graph required")
        original = hybrid.BidirectionalGDN.forward
        forward = FunctionType(original.__code__, {**original.__globals__, "gated_delta_recurrence": chunked_gated_delta_recurrence},
                               original.__name__, original.__defaults__, original.__closure__)
        forward.__kwdefaults__ = original.__kwdefaults__
        views = []
        for index, block in enumerate(graph.blocks):
            expected = "GLOBAL" if (index+1) % 4 == 0 else "BIDIRECTIONAL_GDN"
            if type(block) is not hybrid.Block or block.kind != expected or type(block.moe) is not hybrid.SparseMoE:
                raise ValueError("original block/mixer instances required")
            mixer = _GDNView(block.mixer, forward) if expected == "BIDIRECTIONAL_GDN" else block.mixer
            view = _BlockView(block, mixer, hybrid.Block.forward)
            views.append(_CheckpointedBlock(view) if checkpoint_blocks else view)
        self._graph, self.blocks, self._policy = graph, tuple(views), policy
        self.checkpoint_blocks = checkpoint_blocks

    def __getattr__(self, name):
        return getattr(self._graph, name)

    def __call__(self, rows, token_mask, knownness):
        return self._policy.Graph.encode(self, rows, token_mask, knownness)


def pad_bucket(length, multiple=32):
    """Fast-profile collection pads cohorts to a bucket so shapes repeat across steps."""
    return max(multiple, ((length + multiple - 1)//multiple)*multiple)


def stack_arrays(groups):
    """Concatenate cohort arrays along the batch axis, padding tokens to the longest bucket.

    Returns (rows, token_mask, knownness) numpy arrays and the per-cohort row slices.
    Padding is inert for the fast encoder (masked tokens are identity transitions,
    excluded from routing and pooling), which the CPU tests and CUDA canaries check.
    """
    width = groups[0]["rows"].shape[-1]
    longest = max(g["rows"].shape[1] for g in groups)
    total = sum(g["rows"].shape[0] for g in groups)
    rows = np.zeros((total, longest, width), np.float32)
    known = np.zeros((total, longest, width), np.bool_)
    mask = np.zeros((total, longest), np.bool_)
    slices, start = [], 0
    for g in groups:
        b, n = g["rows"].shape[:2]
        rows[start:start+b, :n] = g["rows"]; known[start:start+b, :n] = g["knownness"]; mask[start:start+b, :n] = g["token_mask"]
        slices.append(slice(start, start+b)); start += b
    return rows, mask, known, slices


# --------------------------------------------------------------------------
# Lockstep grouped sampler with private per-lane PCG64 streams
# --------------------------------------------------------------------------
class LaneStream:
    """One private sampler stream per lane; ``draws`` counts consumed uniforms."""
    def __init__(self, seed):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("explicit nonnegative lane seed required")
        self.seed, self.draws = seed, 0
        self._bits = np.random.PCG64(seed)

    def uniform(self):
        raw = int(self._bits.random_raw())
        self.draws += 1
        return (raw >> 11) * (1.0/(1 << 53))

    def skip(self, count):
        if type(count) is not int or count < 0:
            raise ValueError("nonnegative skip count required")
        self._bits.advance(count); self.draws += count


def _walk(prefix="", production_only=False, fixed_scope=None):
    """Generator twin of actions._traverse: yields (path, choices), receives the chosen value."""
    from farm_neat.economy import action_space as A
    def categorical(name, choices):
        if fixed_scope is not None and name.startswith("scope."):
            choices = (fixed_scope[name.split(".", 1)[1]],)
        return (yield (prefix + name, tuple(choices)))
    def integer(name, maximum=A.MAX_INTEGER):
        value = 0
        for digit in range(7, -1, -1):
            remainder = 16**digit
            high = min(15, maximum // remainder - value * 16)
            value = value * 16 + (yield from categorical(f"{name}.hex{digit}", range(high + 1)))
        return value
    def scope(name="scope."):
        kind = yield from categorical(name + "kind", ("TILE", "REGION"))
        if kind == "TILE":
            x = yield from categorical(name + "x", range(10)); y = yield from categorical(name + "y", range(10))
            return {"kind": kind, "x": x, "y": y}
        region = yield from categorical(name + "region", A.REGIONS)
        count = yield from categorical(name + "count", range(1, (100 if region == "ALL" else 25) + 1))
        return {"kind": kind, "region": region, "count": count}
    def window():
        start = yield from categorical("start", range(A.LAST_STEP + 1))
        until = yield from categorical("until", ("GAME_END", *range(start, A.LAST_STEP + 1)))
        priority = yield from categorical("priority", range(256))
        return {"start": start, "until": until, "priority": priority}
    op = yield from categorical("op", ("SET_CROP", "SET_ANIMAL") if production_only else A.OPS)
    c = {"op": op}
    if op in ("SET_CROP", "SET_ANIMAL", "CLEAR", "REPLACE"):
        c["scope"] = yield from scope()
    if op == "REPLACE":
        c["replacement"] = yield from _walk(prefix + "replacement.", True, c["scope"])
    elif op == "SET_CROP":
        c["crop"] = yield from categorical("crop", A.CROPS)
        has_cycles = yield from categorical("cycles.present", (False, True))
        c["cycles"] = ((yield from integer("cycles.minus_one", A.MAX_INTEGER - 1)) + 1) if has_cycles else None
        c["harvest_threshold"] = (yield from integer("harvest_threshold.minus_one", A.MAX_INTEGER - 1)) + 1
        c["watering"] = yield from categorical("watering", A.WATERING)
        c["fertilizing"] = yield from categorical("fertilizing", A.FERTILIZING)
        c.update((yield from window()))
    elif op == "SET_ANIMAL":
        c["animal"] = yield from categorical("animal", A.ANIMALS)
        c["feed"] = yield from categorical("feed", A.FEED)
        c["care"] = yield from categorical("care", A.CARE)
        c["harvest_threshold"] = (yield from integer("harvest_threshold.minus_one", A.MAX_INTEGER - 1)) + 1
        c["manure"] = yield from categorical("manure", A.MANURE)
        c.update((yield from window()))
    elif op == "CLEAR":
        c["priority"] = yield from categorical("priority", range(256))
    elif op == "CANCEL":
        kind = yield from categorical("target.kind", A.CANCEL_KINDS)
        c["target"] = {"kind": kind}
        if kind == "PRODUCTION":
            c["target"]["scope"] = yield from scope("target.scope.")
        elif kind == "SALE":
            c["target"]["product"] = yield from categorical("target.product", A.PRODUCTS)
        elif kind in ("STOCK", "BUY"):
            stock_kind = yield from categorical("target.stock_kind", A.STOCK_KINDS)
            c["target"]["stock_kind"] = stock_kind
            c["target"]["item"] = yield from categorical("target.item", {"SEED": A.CROPS, "ANIMAL": A.ANIMALS, "PRODUCT": A.BUYABLE_PRODUCTS}[stock_kind])
    elif op in ("LAND", "HANDS"):
        c["target"] = (yield from categorical("target", range(4))) if op == "LAND" else (yield from integer("target"))
        c.update((yield from window()))
    elif op == "SALE":
        c["product"] = yield from categorical("product", A.PRODUCTS)
        c["quantity"] = yield from integer("quantity")
        c["reserve"] = yield from integer("reserve")
        c["price_above"] = (yield from integer("price_above.cents")) / 100
        c.update((yield from window()))
    elif op in ("STOCK", "BUY"):
        c["kind"] = yield from categorical("kind", A.STOCK_KINDS)
        c["item"] = yield from categorical("item", {"SEED": A.CROPS, "ANIMAL": A.ANIMALS, "PRODUCT": A.BUYABLE_PRODUCTS}[c["kind"]])
        name = "target" if op == "STOCK" else "quantity"
        c[name] = yield from integer(name)
        c["price_below"] = (yield from integer("price_below.cents")) / 100
        c.update((yield from window()))
    elif op == "CASH_RESERVE":
        c["amount"] = (yield from integer("amount.cents")) / 100
    return c


class _Lane:
    """Drives one lane's program through the generator walk, slot by slot."""
    def __init__(self, hidden, stream, max_commands, as_of_step=None):
        self.hidden, self.stream, self.max_commands = hidden, stream, max_commands
        self.as_of_step = as_of_step
        self.commands, self.masks, self.selected_logps = [], [], []
        self.total = None; self.entropy = None
        self.slot, self.done = 0, False
        self._start_slot()

    def _start_slot(self):
        self.walk = _walk()
        self.factors, self.factor_masks = [], []
        self.pending = next(self.walk)          # (path, choices)

    def key(self):
        path, choices = self.pending
        return path, choices, (self.slot == self.max_commands-1 and path == "op"), self.as_of_step

    def advance_choice(self, index, logp_value, mask_list, hidden):
        from farm_neat.economy import action_space as A
        path, choices = self.pending
        self.factors.append({"path": path, "choices": list(choices), "index": index})
        self.factor_masks.append(mask_list)
        self.selected_logps.append(logp_value)
        self.hidden = hidden
        try:
            self.pending = self.walk.send(choices[index])
        except StopIteration as stop:
            command = A.validate_command(stop.value)
            self.commands.append({"schema": A.FACTOR_SCHEMA, "action_schema_sha256": A.ACTION_SCHEMA_SHA256, "factors": self.factors})
            self.masks.append(self.factor_masks)
            self.slot += 1
            if command["op"] == "STOP" or self.slot >= self.max_commands:
                self.done = True
                if command["op"] != "STOP":
                    raise ValueError("last slot must be STOP-only")
            else:
                self._start_slot()


_SPEC_CACHE = {}


def _spec(policy, path, choices, stop_only, as_of_step=None):
    """Cache includes as_of_step: mixed-day cohorts must never share a wrong mask."""
    key = (path, tuple(policy._key(c) for c in choices), stop_only, as_of_step)
    entry = _SPEC_CACHE.get(key)
    if entry is None:
        if stop_only and (path != "op" or "STOP" not in choices):
            raise ValueError("budget mask only applies to outer op")
        base, depth = policy._path(path)
        mask = policy.sampling_mask(path, choices, stop_only=stop_only, as_of_step=as_of_step)
        entry = {"ids": [policy.OPTION_IDS[k] for k in key[1]], "prior": policy._prior(path, choices), "mask": mask,
                 "path_id": policy.PATH_IDS[base], "depth": depth, "last_legal": max(i for i, m in enumerate(mask) if m)}
        _SPEC_CACHE[key] = entry
    return entry


_DEVICE_SPECS = {}


def _device_spec(policy, path, choices, stop_only, device, as_of_step=None):
    entry = _spec(policy, path, choices, stop_only, as_of_step)
    key = (id(entry), str(device))
    cached = _DEVICE_SPECS.get(key)
    if cached is None:
        import torch
        cached = {"ids": torch.as_tensor(entry["ids"], device=device),
                  "prior": torch.as_tensor(np.asarray(entry["prior"], np.float32), device=device),
                  "mask": torch.as_tensor(np.asarray(entry["mask"], np.bool_), device=device),
                  "k": len(entry["ids"]), "path_id": entry["path_id"], "depth": entry["depth"], "last_legal": entry["last_legal"]}
        _DEVICE_SPECS[key] = cached
    return cached


def factor_position(graph, hidden, specs, temperatures, *, extra_rows=None):
    """One batched typed-factor step for every row.

    Returns (logp [n,Kmax], entropy [n], ids [n,Kmax], path_vector [n,W], mask [n,Kmax], extra [n] or None).
    Same per-row arithmetic as Graph.factor / batched replay: shared option
    embeddings, path+replacement vectors, tanh query, /sqrt(width), option bias,
    explicit prior, temperature, budget mask, log-softmax. Padded slots are
    masked to -inf and never selected. Per-spec tensors are cached on the device;
    one small host->device copy carries path ids, depths and optional extras.
    ``temperatures`` may be a list or an already resident [n] float32 tensor.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from farm_neat.policy import network as policy
    device = hidden.device
    entries = [_device_spec(policy, spec[0], spec[1], spec[2], device, spec[3] if len(spec) == 4 else None)
               for spec in specs]
    if all(e is entries[0] for e in entries):
        ids_t = entries[0]["ids"][None].expand(len(entries), -1); prior_t = entries[0]["prior"][None].expand(len(entries), -1)
        mask_t = entries[0]["mask"][None].expand(len(entries), -1)
    else:
        ids_t = pad_sequence([e["ids"] for e in entries], batch_first=True, padding_value=0)
        prior_t = pad_sequence([e["prior"] for e in entries], batch_first=True, padding_value=0.0)
        mask_t = pad_sequence([e["mask"] for e in entries], batch_first=True, padding_value=False)
    columns = [[e["path_id"] for e in entries], [e["depth"] for e in entries]]
    if extra_rows is not None:
        columns.append(list(extra_rows))
    meta = torch.as_tensor(np.asarray(columns, np.int64), device=device)
    temps = temperatures if isinstance(temperatures, torch.Tensor) else torch.as_tensor(np.asarray(temperatures, np.float32), device=device)
    path_vector = graph.path_embedding.weight[meta[0]] + graph.replacement_embedding.weight[meta[1]]
    option_vectors = graph.option_embedding(ids_t)
    query = graph.factor_query(torch.tanh(hidden + path_vector))
    logits = (option_vectors * query[:, None, :]).sum(-1)/math.sqrt(graph.config.width) + graph.option_bias(ids_t).squeeze(-1)
    logits = (logits + prior_t) / temps[:, None]
    logp = logits.masked_fill(~mask_t, -torch.inf).log_softmax(-1)
    safe = logp.masked_fill(~mask_t, 0.0)            # masked slots contribute exp(0)*0 = 0 and no NaN gradient
    entropy = -(safe.exp()*safe).sum(-1)
    return logp, entropy, ids_t, path_vector, mask_t, (meta[2] if extra_rows is not None else None)


def replay_padded(graph, context, records, *, expected_identity, return_factor_logps=False):
    """Differentiable (joint_logp[B], path_entropy[B]) in FP64 for exact recorded factor paths.

    Validation and grouping keys reuse the pinned batched replay's ``_paths``;
    every position runs as one padded batch over all live rows. The opt-in third
    return is selected factor logps[B, positions], zero outside each trace.
    It exposes the same replay for supervised reductions; joint likelihood and
    entropy retain their original meanings and arithmetic.
    """
    import torch
    from farm_neat.policy import network as policy
    from farm_neat.rollout import batched_replay as batch
    if expected_identity != identity():
        raise ValueError("externally frozen fast-profile identity required")
    if (context.ndim != 2 or len(context) < 1 or context.shape[1] != graph.config.width or len(records) != len(context)
            or context.dtype != torch.float32 or not bool(torch.isfinite(context).all())):
        raise ValueError("finite FP32 context and one typed record per context required")
    sequences = batch._paths(policy, graph, records)
    hidden = context
    # FP64 per-row sums accumulate one term per position in program order (same order as the scalar
    # reference); one index_add per position instead of a Python loop over rows.
    totals = context.new_zeros(len(context), dtype=torch.float64) + context.sum(-1, dtype=torch.float64)*0
    entropies = totals.clone()
    support_ok = []
    selected_positions = [] if return_factor_logps else None
    for position in range(max(map(len, sequences))):
        rows = [r for r, seq in enumerate(sequences) if position < len(seq)]
        specs, temps, selected = [], [], []
        for r in rows:
            (path, _, temperature, stop_only, as_of_step), choices, index = sequences[r][position]
            specs.append((path, choices, stop_only, as_of_step)); temps.append(temperature); selected.append(index)
        index_t = torch.as_tensor(selected, device=context.device)
        full = len(rows) == len(hidden)
        rows_t = None if full else torch.as_tensor(rows, device=context.device)
        h = hidden if full else hidden[rows_t]
        logp, entropy, ids_t, path_vector, mask_t, _ = factor_position(graph, h, specs, temps)
        support_ok.append(mask_t.gather(1, index_t[:, None]).all())
        selected_logp = logp.gather(1, index_t[:, None]).squeeze(1)
        if selected_positions is not None:
            selected_positions.append(selected_logp.double() if full else
                totals.new_zeros(len(context)).index_copy(0, rows_t, selected_logp.double()))
        advanced = graph.prefix(graph.option_embedding(ids_t.gather(1, index_t[:, None]).squeeze(1)) + path_vector, h)
        if full:
            totals = totals + selected_logp.double(); entropies = entropies + entropy.double(); hidden = advanced
        else:
            totals = totals.index_add(0, rows_t, selected_logp.double()); entropies = entropies.index_add(0, rows_t, entropy.double())
            hidden = hidden.index_copy(0, rows_t, advanced)
    if not bool(torch.stack(support_ok).all()):       # one host read for the whole replay
        raise ValueError("selected command lacks declared support")
    result = totals, entropies
    if not all(bool(torch.isfinite(v).all()) for v in result):
        raise ValueError("nonfinite padded replay output")
    return (*result, torch.stack(selected_positions, dim=1)) if return_factor_logps else result


def sample_batched(graph, context, streams, *, max_commands=8, temperature=1.0, as_of_step=None):
    """Lockstep sampled programs for every context row; returns (records, joint_logp, entropy).

    Records are exactly ``Graph.sample`` trace records (same schema, factors,
    masks, selected logps and FP64 joint sum), drawn from private lane streams.
    Every factor position is one padded batch over all live lanes.
    """
    import torch
    from farm_neat.policy import network as policy
    from farm_neat.economy import action_space as A
    if context.ndim != 2 or context.shape[1] != graph.config.width or len(streams) != len(context):
        raise ValueError("batched context and one private stream per lane required")
    A._integer(max_commands, 2, A.MAX_COMMANDS, "command budget")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive finite temperature required")
    device = context.device
    header = policy.trace_header(as_of_step)
    lanes = [_Lane(h, s, max_commands, as_of_step) for h, s in zip(context.unbind(0), streams, strict=True)]
    totals = context.new_zeros(len(context), dtype=torch.float64) + context.sum(-1, dtype=torch.float64)*0
    entropies = totals.clone()
    temps_all = torch.full((len(lanes),), float(temperature), dtype=torch.float32, device=device)
    while True:
        live = [i for i, lane in enumerate(lanes) if not lane.done]
        if not live:
            break
        specs = [lanes[i].key() for i in live]
        h = torch.stack([lanes[i].hidden for i in live])
        logp, entropy, ids_t, path_vector, mask_t, last_legal = factor_position(graph, h, specs, temps_all[:len(live)],
            extra_rows=[_spec(policy, *spec)["last_legal"] for spec in specs])
        uniforms = torch.as_tensor(np.asarray([lanes[i].stream.uniform() for i in live], np.float64), device=device)
        cdf = logp.exp().double().cumsum(-1)
        index = (cdf < uniforms[:, None]).sum(-1).clamp_(max=mask_t.shape[1]-1)
        # cumulative rounding at the tail can land on a masked/padded slot: snap to the last legal option.
        index = torch.where(mask_t.gather(1, index[:, None]).squeeze(1), index, last_legal)
        selected_logp = logp.gather(1, index[:, None]).squeeze(1)
        advanced = graph.prefix(graph.option_embedding(ids_t.gather(1, index[:, None]).squeeze(1)) + path_vector, h)
        host = torch.stack([index.double(), selected_logp.detach().double()]).tolist()   # one host read per position
        live_t = torch.as_tensor(live, device=device)
        totals = totals.index_add(0, live_t, selected_logp.double()); entropies = entropies.index_add(0, live_t, entropy.double())
        for slot, i in enumerate(live):
            lanes[i].advance_choice(int(host[0][slot]), float(host[1][slot]), _spec(policy, *specs[slot])["mask"], advanced[slot])
    arch = policy.architecture(graph.config)["content_sha256"]
    records = []
    joint = totals.detach().tolist()
    for lane, total in zip(lanes, joint, strict=True):
        program = A.validate_program([A.decode_command(c) for c in lane.commands])
        records.append({**header, "architecture_sha256": arch,
            "action_schema_sha256": A.ACTION_SCHEMA_SHA256, "temperature": temperature,
            "max_commands": max_commands, "program": program, "commands": lane.commands,
            "sampling_masks": lane.masks, "selected_logps": lane.selected_logps,
            "joint_logp": float(total), "training_admission": False})
    return records, totals, entropies


# --------------------------------------------------------------------------
# CUDA-graph lockstep sampler: one captured factor position per lane count
# --------------------------------------------------------------------------
def _kmax():
    from farm_neat.economy import action_space as A
    return A.LAST_STEP + 2


KMAX = None  # resolved at first use (720): "until" = GAME_END plus 719 steps


class GraphedSampler:
    """Same records/logps as sample_batched (bit-identical on the CUDA canaries); ~3x fewer host round trips.

    Requires CUDA. Captured once per (graph, lane count, temperature); replayed per factor position
    with static [B, KMAX] buffers. Dead lanes are masked to one dummy option and never contribute.
    """
    def __init__(self, graph, lanes, temperature=1.0):
        import torch
        global KMAX
        if KMAX is None: KMAX = _kmax()
        if graph.option_embedding.weight.device.type != "cuda": raise ValueError("graphed sampler requires CUDA")
        self.graph, self.B, self.temperature = graph, lanes, float(temperature)
        dev = graph.option_embedding.weight.device; W = graph.config.width
        self.dev = dev
        self.hidden = torch.zeros(lanes, W, device=dev)
        self.ids = torch.zeros(lanes, KMAX, dtype=torch.int64, device=dev)
        self.prior = torch.zeros(lanes, KMAX, device=dev)
        self.mask = torch.zeros(lanes, KMAX, dtype=torch.bool, device=dev)
        self.meta = torch.zeros(3, lanes, dtype=torch.int64, device=dev)      # path_id, depth, last_legal
        self.uniforms = torch.zeros(lanes, dtype=torch.float64, device=dev)
        self.temps = torch.full((lanes,), self.temperature, device=dev)
        self.out = torch.zeros(3, lanes, dtype=torch.float64, device=dev)     # index, selected logp, entropy
        # pinned host staging
        self.h_ids = torch.zeros(lanes, KMAX, dtype=torch.int64).pin_memory()
        self.h_prior = torch.zeros(lanes, KMAX).pin_memory()
        self.h_mask = torch.zeros(lanes, KMAX, dtype=torch.bool).pin_memory()
        self.h_meta = torch.zeros(3, lanes, dtype=torch.int64).pin_memory()
        self.h_uniforms = torch.zeros(lanes, dtype=torch.float64).pin_memory()
        self._np = {"ids": self.h_ids.numpy(), "prior": self.h_prior.numpy(), "mask": self.h_mask.numpy(),
                    "meta": self.h_meta.numpy(), "uniforms": self.h_uniforms.numpy()}
        self.stream = torch.cuda.Stream()
        self._capture()

    def _body(self):
        import torch
        g = self.graph
        path_vector = g.path_embedding.weight[self.meta[0]] + g.replacement_embedding.weight[self.meta[1]]
        option_vectors = g.option_embedding(self.ids)
        query = g.factor_query(torch.tanh(self.hidden + path_vector))
        logits = (option_vectors * query[:, None, :]).sum(-1)/math.sqrt(g.config.width) + g.option_bias(self.ids).squeeze(-1)
        logits = (logits + self.prior) / self.temps[:, None]
        logp = logits.masked_fill(~self.mask, -torch.inf).log_softmax(-1)
        safe = logp.masked_fill(~self.mask, 0.0)
        entropy = -(safe.exp()*safe).sum(-1)
        cdf = logp.exp().double().cumsum(-1)
        index = (cdf < self.uniforms[:, None]).sum(-1).clamp_(max=KMAX-1)
        index = torch.where(self.mask.gather(1, index[:, None]).squeeze(1), index, self.meta[2])
        selected = logp.gather(1, index[:, None]).squeeze(1)
        new_hidden = g.prefix(g.option_embedding(self.ids.gather(1, index[:, None]).squeeze(1)) + path_vector, self.hidden)
        self.hidden.copy_(new_hidden)
        self.out[0].copy_(index.double()); self.out[1].copy_(selected.double()); self.out[2].copy_(entropy.double())

    def _capture(self):
        import torch
        with torch.no_grad():
            self.mask[:, 0] = True
            s = self.stream; s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): self._body()
            torch.cuda.current_stream().wait_stream(s)
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph, stream=s):
                self._body()
            torch.cuda.synchronize()

    def sample(self, context, streams, *, max_commands=8, as_of_step=None):
        """Same records as sample_batched; context rows == lanes."""
        import torch
        from farm_neat.policy import network as policy
        from farm_neat.economy import action_space as A
        if context.shape[0] != self.B or len(streams) != self.B: raise ValueError("captured lane count required")
        A._integer(max_commands, 2, A.MAX_COMMANDS, "command budget")
        header = policy.trace_header(as_of_step)
        with torch.no_grad():
            self.hidden.copy_(context)
            lanes = [_Lane(None, s, max_commands, as_of_step) for s in streams]
            totals = np.zeros(self.B, np.float64); entropies = np.zeros(self.B, np.float64)
            ids_np, prior_np, mask_np, meta_np, uni_np = (self._np[k] for k in ("ids", "prior", "mask", "meta", "uniforms"))
            while True:
                live = [i for i, lane in enumerate(lanes) if not lane.done]
                if not live: break
                mask_np[:] = False; mask_np[:, 0] = True; ids_np[:] = 0; prior_np[:] = 0.0; meta_np[:] = 0; uni_np[:] = 0.5
                specs = {}
                for i in live:
                    e = _spec(policy, *lanes[i].key()); specs[i] = e
                    k = len(e["ids"]); ids_np[i, :k] = e["ids"]; prior_np[i, :k] = e["prior"]; mask_np[i, :k] = e["mask"]
                    if k < KMAX: mask_np[i, k:] = False
                    meta_np[0, i] = e["path_id"]; meta_np[1, i] = e["depth"]; meta_np[2, i] = e["last_legal"]
                    uni_np[i] = lanes[i].stream.uniform()
                self.ids.copy_(self.h_ids, non_blocking=True); self.prior.copy_(self.h_prior, non_blocking=True)
                self.mask.copy_(self.h_mask, non_blocking=True); self.meta.copy_(self.h_meta, non_blocking=True)
                self.uniforms.copy_(self.h_uniforms, non_blocking=True)
                self.cuda_graph.replay()
                host = self.out.cpu().numpy()          # one synchronizing read per position
                for i in live:
                    index, logp_value, ent = int(host[0, i]), float(host[1, i]), float(host[2, i])
                    totals[i] += logp_value; entropies[i] += ent
                    lanes[i].advance_choice(index, logp_value, specs[i]["mask"], None)
        arch = policy.architecture(self.graph.config)["content_sha256"]
        records = []
        for lane, total in zip(lanes, totals):
            program = A.validate_program([A.decode_command(c) for c in lane.commands])
            records.append({**header, "architecture_sha256": arch, "action_schema_sha256": A.ACTION_SCHEMA_SHA256,
                "temperature": self.temperature, "max_commands": max_commands, "program": program, "commands": lane.commands,
                "sampling_masks": lane.masks, "selected_logps": lane.selected_logps, "joint_logp": float(math.fsum(lane.selected_logps)),
                "training_admission": False})
        return records, torch.as_tensor(totals, device=self.dev), torch.as_tensor(entropies, device=self.dev)
