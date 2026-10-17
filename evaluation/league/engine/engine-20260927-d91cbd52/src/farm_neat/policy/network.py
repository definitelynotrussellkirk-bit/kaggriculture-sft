"""New full-economics16-layer MoE actor/critic with real conditional logps.

Reuses the maintained Qwen-inspired mixer/residual/MoE modules, not its old29
candidate wrapper or checkpoint semantics. Entity validity and action syntax
are different masks. This is executable FP32 CPU/CUDA reference code, not a
fused-kernel speed claim; GDN still traverses entity tokens in Python.

No old actor/optimizer migration is implied. New economic features, action
factors and initialization require a new explicit training lineage. Router
auxiliaries are returned separately, never silently added to terminal reward.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from farm_neat.economy import action_space as actions
from farm_neat.policy import observation as observation
from farm_neat.policy import qwen_hybrid as hybrid

SCHEMA = "farm-full-economic-moe-policy-v1"
TRACE_SCHEMA = "farm-full-economic-moe-behavior-v1"
REMAINING_TRACE_SCHEMA = "farm-full-economic-moe-behavior-remaining-window-v2"
LEGACY_SOURCE_SHA256 = "26610f6456afec4d0ca953b65ec928881b2c397beada78fa56a91b9340ec34f5"
NUMERICS = "TORCH_FP32_FULL_ECONOMIC_MOE_V1"
DEFAULT = hybrid.Config()
WIDTH = actions.WIDTH

# Typed, collision-free option embeddings. All big quantities use digit factors.
_STRINGS = tuple(dict.fromkeys((*actions.OPS, "TILE", "REGION", *actions.REGIONS,
    *actions.CROPS, *actions.ANIMALS, *actions.PRODUCTS, *actions.WATERING,
    *actions.FERTILIZING, *actions.FEED, *actions.CARE, *actions.MANURE,
    *actions.STOCK_KINDS, *actions.CANCEL_KINDS, "GAME_END")))
OPTIONS = tuple(("int", i) for i in range(719)) + (("bool", False), ("bool", True)) + tuple(("str", s) for s in _STRINGS)
OPTION_IDS = {value: i for i, value in enumerate(OPTIONS)}
_BASE_PATHS = ("op", "scope.kind", "scope.x", "scope.y", "scope.region", "scope.count",
    "crop", "cycles.present", "watering", "fertilizing", "animal", "feed", "care", "manure",
    "start", "until", "priority", "target", "product", "kind", "item", "target.kind",
    "target.product", "target.stock_kind", "target.item", "target.scope.kind", "target.scope.x",
    "target.scope.y", "target.scope.region", "target.scope.count")
_NUMERIC_PATHS = ("cycles.minus_one", "harvest_threshold.minus_one", "target", "quantity", "reserve",
                  "price_above.cents", "price_below.cents", "amount.cents")
PATHS = _BASE_PATHS + tuple(f"{path}.hex{digit}" for path in _NUMERIC_PATHS for digit in range(7, -1, -1))
PATH_IDS = {name: i for i, name in enumerate(PATHS)}
INITIAL_PRIOR = {
    "name": "EXPLICIT_SMALL_NUMBER_FULL_SUPPORT_V1", "nonpreferred_high_hex_logit": -14.0,
    "low_hex_distance_logit": -0.5, "integer_cent_targets": {"cycles.minus_one": 2,
        "harvest_threshold.minus_one": 0, "target": 2, "quantity": 3, "reserve": 0,
        "price_above.cents": 0, "price_below.cents": 100000, "amount.cents": 0},
    "until": "GAME_END", "start": 0, "scope_count": 1,
    "service": {"watering": "DAILY", "fertilizing": "OFF", "feed": "DAILY",
                "care": "DAILY", "manure": "WHEN_AVAILABLE"},
    "crop_animal_economic_ranking": "NONE_ALL_FAMILIES_HAVE_POSITIVE_SUPPORT",
}


def _key(value):
    return ("bool" if type(value) is bool else "int" if type(value) is int else "str", value)


def _path(path):
    replacement = path.startswith("replacement.")
    base = path[len("replacement."):] if replacement else path
    if base not in PATH_IDS:
        raise ValueError(f"unregistered factor path: {path}")
    return base, int(replacement)


def trace_header(as_of_step=None):
    """Explicit opt-in support version; legacy trace bytes remain unchanged."""
    if as_of_step is None:
        return {"schema": TRACE_SCHEMA}
    actions._integer(as_of_step, 0, 718, "sampling as_of_step")
    return {"schema": REMAINING_TRACE_SCHEMA, "as_of_step": as_of_step}


def trace_as_of(record):
    fields = ("schema", "architecture_sha256", "action_schema_sha256", "temperature", "max_commands",
              "program", "commands", "sampling_masks", "selected_logps", "joint_logp", "training_admission")
    if isinstance(record, dict) and record.get("schema") == REMAINING_TRACE_SCHEMA:
        actions._exact(record, (*fields, "as_of_step"), "remaining-window behavior trace")
        return actions._integer(record["as_of_step"], 0, 718, "sampling as_of_step")
    actions._exact(record, fields, "behavior trace")
    if record["schema"] != TRACE_SCHEMA:
        raise ValueError("explicit versioned behavior trace required")
    return None


def sampling_mask(path, choices, *, stop_only=False, as_of_step=None):
    """Only expired deadlines are removed; past starts/ongoing duties stay legal.

    No feasibility, profitability, family, reserve or logistics filtering.
    The independent world witness binds as_of_step to its actual decision turn.
    """
    trace_header(as_of_step)
    if stop_only:
        if path != "op" or "STOP" not in choices:
            raise ValueError("budget mask only applies to outer op")
        return [c == "STOP" for c in choices]
    if as_of_step is not None and _path(path)[0] == "until":
        return [c == "GAME_END" or (type(c) is int and c >= as_of_step) for c in choices]
    return [True]*len(choices)


def _prior(path, choices):
    base, _ = _path(path)
    if ".hex" in base:
        name, digit = base.rsplit(".hex", 1)
        target = INITIAL_PRIOR["integer_cent_targets"][name]
        preferred = (target >> (4 * int(digit))) & 15
        if int(digit) > 0:
            return [0.0 if v == preferred else -14.0 for v in choices]
        return [-0.5 * abs(v-preferred) for v in choices]
    if base in ("scope.count", "target.scope.count"):
        return [-min(14.0, 0.5 * (v-1)) for v in choices]
    if base == "start":
        return [0.0 if v == 0 else -8.0 for v in choices]
    if base == "until":
        return [0.0 if v == "GAME_END" else -8.0 for v in choices]
    if base == "priority":
        return [-min(14.0, 0.5*v) for v in choices]
    if base in INITIAL_PRIOR["service"]:
        return [1.0 if v == INITIAL_PRIOR["service"][base] else 0.0 for v in choices]
    return [0.0] * len(choices)


def architecture(config=DEFAULT):
    payload = {"schema": SCHEMA, "numerics": NUMERICS, "config": asdict(config),
        "input_contract_sha256": actions.INPUT_SCHEMA_SHA256,
        "representation_schema_sha256": observation.REPRESENTATION_SCHEMA_SHA256,
        "action_schema_sha256": actions.ACTION_SCHEMA_SHA256,
        "token_order": "EXACT_TYPED_ENTITY_ENCODER_ORDER_NOT_ACTION_LEGALITY",
        "token_count": "DYNAMIC_NO_29_CANDIDATE_LIMIT", "knownness_projection": True,
        "trunk": "REUSED_QWEN_HYBRID_BLOCKS_ZERO_GDN_STATE_EACH_CALL",
        "conditional_policy": "SHARED_OPTION_EMBEDDING_QUERY_AND_GRUCELL_PREFIX",
        "critic": "MEAN_VALID_ENTITY_TOKENS", "paths": PATHS, "options": OPTIONS,
        "initial_prior": INITIAL_PRIOR, "joint_logp_accumulator": "FLOAT64_SUM_OF_FP32_CONDITIONAL_LOGPS",
        "old_checkpoint_or_optimizer_compatible": False}
    return {**payload, "content_sha256": actions.digest(payload)}


def sources():
    root = Path(__file__).resolve().parents[3]
    names = ("src/farm_neat/policy/network.py", "src/farm_neat/economy/action_space.py",
             "src/farm_neat/policy/qwen_hybrid.py", "src/farm_neat/game/constants.py")
    return {**hybrid.sources(), **observation.sources(),
            **{name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}}


class Graph(nn.Module):
    """Default is the full16-layer model. Tiny configs are numerical tests only."""
    def __init__(self, config=DEFAULT):
        super().__init__()
        if not isinstance(config, hybrid.Config):
            raise ValueError("explicit hybrid configuration required")
        self.config = config
        width = config.width
        self.input = nn.Linear(WIDTH, width, bias=False)
        self.knownness_input = nn.Linear(WIDTH, width, bias=False)
        self.blocks = nn.ModuleList([hybrid.Block(config, i) for i in range(config.layers)])
        self.final_norm = hybrid.ZeroCenteredRMSNorm(width, config.rms_epsilon)
        self.value_head = nn.Linear(width, 1)
        self.option_embedding = nn.Embedding(len(OPTIONS), width)
        self.path_embedding = nn.Embedding(len(PATHS), width)
        self.replacement_embedding = nn.Embedding(2, width)
        self.factor_query = nn.Linear(width, width)
        self.prefix = nn.GRUCell(width, width)
        self.option_bias = nn.Embedding(len(OPTIONS), 1)
        nn.init.normal_(self.option_embedding.weight, std=0.03)
        nn.init.normal_(self.path_embedding.weight, std=0.03)
        nn.init.normal_(self.replacement_embedding.weight, std=0.03)
        nn.init.normal_(self.factor_query.weight, std=0.01)
        nn.init.zeros_(self.factor_query.bias)
        nn.init.zeros_(self.option_bias.weight)

    def encode(self, rows, token_mask, knownness):
        if (rows.ndim != 3 or rows.shape[0] < 1 or rows.shape[1] < 1 or rows.shape[2] != WIDTH
                or rows.dtype != torch.float32 or token_mask.shape != rows.shape[:2]
                or knownness.shape != rows.shape or token_mask.dtype != torch.bool
                or knownness.dtype != torch.bool or rows.device != token_mask.device
                or rows.device != knownness.device or next(self.parameters()).device != rows.device
                or not bool(token_mask[:, 0].all()) or not bool(torch.isfinite(rows).all())
                or torch.is_autocast_enabled(rows.device.type)):
            raise ValueError("FP32 BxNx80, separate entity validity/knownness and GLOBAL first required")
        if rows.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("new FP32 execution profile requires TF32 disabled")
        visible = rows.masked_fill(~knownness | ~token_mask[..., None], 0)
        x = self.input(visible) + self.knownness_input((knownness & token_mask[..., None]).to(rows.dtype))
        residual = x[..., None, :].expand(*x.shape[:2], self.config.branches, self.config.width)
        routes = []
        for block in self.blocks:
            residual, routing = block(residual, token_mask)
            routes.append(routing)
        h = self.final_norm(residual.mean(-2)).masked_fill(~token_mask[..., None], 0)
        pooled = h.sum(1)/token_mask.sum(1, keepdim=True)
        values = self.value_head(pooled).squeeze(-1)
        return pooled, values, {"layers": tuple(routes),
            "load_balance_loss": torch.stack([r["load_balance_loss"] for r in routes]).mean(),
            "router_z_loss": torch.stack([r["router_z_loss"] for r in routes]).mean()}

    def factor(self, hidden, path, choices, temperature=1.0, *, stop_only=False, as_of_step=None):
        if (hidden.shape != (self.config.width,) or not bool(torch.isfinite(hidden).all())
                or type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("finite prefix and positive temperature required")
        base, depth = _path(path)
        try:
            ids = torch.tensor([OPTION_IDS[_key(c)] for c in choices], device=hidden.device)
        except KeyError as exc:
            raise ValueError("unregistered option value") from exc
        path_vector = self.path_embedding.weight[PATH_IDS[base]] + self.replacement_embedding.weight[depth]
        option_vectors = self.option_embedding(ids)
        query = self.factor_query(torch.tanh(hidden + path_vector))
        logits = (option_vectors * query).sum(-1)/math.sqrt(self.config.width) + self.option_bias(ids).squeeze(-1)
        logits = (logits + logits.new_tensor(_prior(path, choices))) / temperature
        mask = torch.tensor(sampling_mask(path, choices, stop_only=stop_only, as_of_step=as_of_step),
                            device=hidden.device, dtype=torch.bool)
        logp = logits.masked_fill(~mask, -torch.inf).log_softmax(-1)
        entropy = -(logp[mask].exp()*logp[mask]).sum()
        return logp, entropy, ids, path_vector, mask

    def advance(self, hidden, option_id, path_vector):
        return self.prefix(self.option_embedding.weight[option_id] + path_vector, hidden)

    def sample(self, context, *, generator, max_commands=8, temperature=1.0, as_of_step=None):
        """Actual sampled conditional program. Return tensors retain gradients.

        Sampling generator must be explicitly supplied on the context device.
        Last-slot STOP-only is in the actual distribution and trace, not a
        posthoc discarded suffix or an old-logp mismatch. Other commands remain
        supported in every earlier slot, irrespective of economic forecasts.
        """
        if context.ndim != 2 or context.shape[1] != self.config.width or not isinstance(generator, torch.Generator):
            raise ValueError("batched context and explicit torch generator required")
        actions._integer(max_commands, 2, actions.MAX_COMMANDS, "command budget")
        header = trace_header(as_of_step)
        records, totals, entropies = [], [], []
        for initial in context:
            hidden = initial
            commands, masks, selected_logps = [], [], []
            total, entropy_total = initial.sum(dtype=torch.float64)*0, initial.sum(dtype=torch.float64)*0
            for slot in range(max_commands):
                factors, factor_masks = [], []
                def read(path, choices):
                    nonlocal hidden, total, entropy_total
                    logp, entropy, ids, path_vector, mask = self.factor(hidden, path, choices, temperature,
                        stop_only=(slot == max_commands-1 and path == "op"), as_of_step=as_of_step)
                    index = int(torch.multinomial(logp.exp(), 1, generator=generator).item())
                    factors.append({"path": path, "choices": list(choices), "index": index})
                    factor_masks.append(mask.detach().cpu().tolist())
                    selected_logps.append(float(logp[index].detach()))
                    total, entropy_total = total+logp[index], entropy_total+entropy
                    hidden = self.advance(hidden, ids[index], path_vector)
                    return choices[index]
                command = actions.validate_command(actions._traverse(read))
                commands.append({"schema": actions.FACTOR_SCHEMA,
                    "action_schema_sha256": actions.ACTION_SCHEMA_SHA256, "factors": factors})
                masks.append(factor_masks)
                if command["op"] == "STOP":
                    break
            program = actions.validate_program([actions.decode_command(c) for c in commands])
            records.append({**header, "architecture_sha256": architecture(self.config)["content_sha256"],
                "action_schema_sha256": actions.ACTION_SCHEMA_SHA256, "temperature": temperature,
                "max_commands": max_commands, "program": program, "commands": commands,
                "sampling_masks": masks, "selected_logps": selected_logps,
                "joint_logp": float(total.detach()), "training_admission": False})
            totals.append(total); entropies.append(entropy_total)
        return records, torch.stack(totals), torch.stack(entropies)

    def replay(self, context, records):
        """Differentiable fresh-policy logp for the EXACT sampled factor path.

        Saved behavior logps are retained evidence, not recomputed/relabelled.
        Full fresh-world witness/admission and PPO optimizer remain owner-side.
        """
        if context.ndim != 2 or context.shape[1] != self.config.width or len(records) != len(context):
            raise ValueError("one typed record per context required")
        totals, entropies = [], []
        arch = architecture(self.config)["content_sha256"]
        for initial, record in zip(context, records, strict=True):
            as_of_step = trace_as_of(record)
            if (record["architecture_sha256"] != arch
                    or record["action_schema_sha256"] != actions.ACTION_SCHEMA_SHA256
                    or record["training_admission"] is not False):
                raise ValueError("new source/architecture/action trace required, not old29 admission")
            budget = actions._integer(record["max_commands"], 2, actions.MAX_COMMANDS, "command budget")
            program = actions.validate_program(record["program"])
            if len(program) > budget or len(record["commands"]) != len(program) or len(record["sampling_masks"]) != len(program):
                raise ValueError("program/trace/budget length mismatch")
            hidden, total, entropy_total = initial, initial.sum(dtype=torch.float64)*0, initial.sum(dtype=torch.float64)*0
            sampled_count = 0
            for slot, (encoded, command, saved_masks) in enumerate(zip(record["commands"], program, record["sampling_masks"], strict=True)):
                if actions.decode_command(encoded) != command or len(saved_masks) != len(encoded["factors"]):
                    raise ValueError("command factors or mask cardinality mismatch")
                for factor, saved_mask in zip(encoded["factors"], saved_masks, strict=True):
                    logp, entropy, ids, path_vector, mask = self.factor(hidden, factor["path"], tuple(factor["choices"]),
                        record["temperature"], stop_only=(slot == budget-1 and factor["path"] == "op"), as_of_step=as_of_step)
                    if saved_mask != mask.detach().cpu().tolist():
                        raise ValueError("actual sampling budget/syntax mask mismatch")
                    index = factor["index"]
                    if not bool(mask[index]):
                        raise ValueError("selected command lacks declared support")
                    total, entropy_total = total+logp[index], entropy_total+entropy
                    hidden = self.advance(hidden, ids[index], path_vector)
                    sampled_count += 1
            old = record["selected_logps"]
            if (not isinstance(old, list) or len(old) != sampled_count
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v > 1e-6 for v in old)
                    or type(record["joint_logp"]) not in (int, float) or not math.isfinite(record["joint_logp"])
                    or not math.isclose(math.fsum(old), record["joint_logp"], rel_tol=1e-6, abs_tol=2e-4)):
                raise ValueError("finite consistent old joint behavior logp required")
            totals.append(total); entropies.append(entropy_total)
        return torch.stack(totals), torch.stack(entropies)

    def replay_prefix(self, context, records):
        """BC logps over observed conditional factors ONLY, not guessed plans.

        Unobserved cycles, thresholds, reserves and other suffix factors carry
        no BC loss. This is a supervised policy seam, not PPO return admission.
        Multiple observed actions from one world still belong to that world.
        """
        if context.ndim != 2 or context.shape[1] != self.config.width or len(records) != len(context):
            raise ValueError("one prefix per context required")
        totals, entropies = [], []
        for initial, record in zip(context, records, strict=True):
            prefix = actions.validate_prefix(record)
            hidden = initial
            total = initial.sum(dtype=torch.float64)*0
            entropy_total = total
            for factor in prefix["factors"]:
                logp, entropy, ids, path_vector, _ = self.factor(hidden, factor["path"], tuple(factor["choices"]))
                index = factor["index"]
                total, entropy_total = total+logp[index], entropy_total+entropy
                hidden = self.advance(hidden, ids[index], path_vector)
            totals.append(total); entropies.append(entropy_total)
        return torch.stack(totals), torch.stack(entropies)


def torch_graph(*, config=DEFAULT, device="cpu"):
    if str(device) not in ("cpu", "cuda", "meta"):
        raise ValueError("explicit CPU/CUDA/meta device required")
    if str(device) == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no silent CPU fallback")
    with torch.device(device):
        return Graph(config).float()


def parameter_report(graph):
    report = hybrid.parameter_report(graph)
    report.update({"schema": SCHEMA, "layers": graph.config.layers,
                   "entity_tokens": "DYNAMIC", "full_economic_factor_sampling": True,
                   "parameter_accounting_is_training_evidence": False})
    return report


def _tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def save_model(path, graph, *, optimizer_steps, rollout_iteration, initialization_seed,
               lineage=None):
    """Exact NEW tensor checkpoint + small manifest, no optimizer compatibility claim.

    The caller owns the optimizer and accepted update receipt. This API records
    supplied age, not proof those updates happened. Paths must be new; partial
    failed exports remain inspectable instead of overwriting an earlier actor.
    """
    path = Path(path)
    sidecar = Path(str(path)+".json")
    if path.exists() or sidecar.exists():
        raise ValueError("new model/manifest output required")
    for value in (optimizer_steps, rollout_iteration, initialization_seed):
        actions._integer(value, label="model age/seed")
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in graph.state_dict().items()}
    if any(tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
        raise ValueError("finite FP32 new model tensors required")
    tensors = {name: {"shape": list(t.shape), "sha256": _tensor_sha(t)} for name, t in state.items()}
    manifest = {"schema": SCHEMA, "numerics": NUMERICS, "architecture": architecture(graph.config),
        "parameter_report": parameter_report(graph), "tensors": tensors,
        "optimizer_steps": optimizer_steps, "rollout_iteration": rollout_iteration,
        "initialization_seed": initialization_seed, "source_pins": sources(),
        "lineage": copy.deepcopy(lineage), "old_optimizer_compatible": False}
    manifest["content_sha256"] = actions.digest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    sidecar.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)+"\n")
    return manifest


def check_model_sources(manifest, *, allow_remaining_window_transition=False):
    """One explicit source transition, not arbitrary stale-model acceptance.

    Tensor architecture/prior/encoder are unchanged. Only this exact old policy
    source may differ; every other model dependency must still match. Old trace
    probabilities retain v1 behavior. Callers record current execution separately
    from the unchanged parent manifest and join Adam to that original manifest.
    """
    current = sources()
    old = manifest["source_pins"]
    if old == current:
        return None
    if (allow_remaining_window_transition is not True or
            old != {**current, "src/farm_neat/policy/network.py": LEGACY_SOURCE_SHA256}):
        raise ValueError("model architecture/representation/action/source mismatch")
    return {"kind": "EXACT_POLICY_V1_TO_OPT_IN_REMAINING_WINDOW_V2",
            "old_source_pins": old, "new_source_pins": current,
            "tensors_optimizer_prior_encoder": "UNCHANGED; EXPLICIT_SAMPLING_SUPPORT_VERSION"}


def read_manifest(path, *, allow_remaining_window_transition=False):
    manifest = json.loads(Path(str(path)+".json").read_text())
    content = dict(manifest)
    claimed = content.pop("content_sha256", None)
    if claimed != actions.digest(content) or manifest.get("schema") != SCHEMA or manifest.get("numerics") != NUMERICS:
        raise ValueError("exact new model manifest identity required")
    config = hybrid.Config(**manifest["architecture"]["config"])
    if manifest["architecture"] != json.loads(json.dumps(architecture(config))):
        raise ValueError("model architecture/representation/action/source mismatch")
    check_model_sources(manifest, allow_remaining_window_transition=allow_remaining_window_transition)
    for key in ("optimizer_steps", "rollout_iteration", "initialization_seed"):
        actions._integer(manifest[key], label=key)
    return manifest


def load_model(path, *, device="cpu", expected_file_sha256=None, allow_remaining_window_transition=False):
    path = Path(path)
    if expected_file_sha256 is not None:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected_file_sha256:
            raise ValueError("model file SHA mismatch")
    manifest = read_manifest(path, allow_remaining_window_transition=allow_remaining_window_transition)
    config = hybrid.Config(**manifest["architecture"]["config"])
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = manifest["tensors"]
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("model tensor keys mismatch")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or list(tensor.shape) != expected[name]["shape"] or not bool(torch.isfinite(tensor).all())
                or _tensor_sha(tensor) != expected[name]["sha256"]):
            raise ValueError("model tensor identity mismatch")
    graph = torch_graph(config=config, device=device)
    if parameter_report(graph) != manifest["parameter_report"]:
        raise ValueError("model actual parameter accounting mismatch")
    graph.load_state_dict(state, strict=True)
    return graph, manifest


def initialize(path, *, seed, config=DEFAULT, device="cpu"):
    """Explicit new MoE genesis; never rebind the closed restricted actor/Adam."""
    actions._integer(seed, label="genesis seed")
    devices = [torch.cuda.current_device()] if str(device) == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        graph = torch_graph(config=config, device=device)
    return save_model(path, graph, optimizer_steps=0, rollout_iteration=0,
                      initialization_seed=seed, lineage={"kind": "NEW_FULL_ECONOMIC_MOE_GENESIS"})
