"""Opt-in conditional-head batching; existing weights, grammar and trace order.

Owner: learning/throughput maintainer. Status: isolated numerical experiment.
Groups independent programs at each dependent factor position, then by exact
path/typed choices/temperature/budget/as-of mask. No parameters or saved traces
change. V1 stays unmasked by time; explicit v2 traces exclude expired deadlines.
FP32 batched GEMMs may round differently; CPU equivalence tolerances below are
tests, NOT a waiver of any historical replay/admission guard. Conditional entropy
is along the supplied paths, not exhaustive joint-program entropy or exact KL.
This exports no checkpoint and grants no training admission. A future producer/
witness/fitter adoption must bind its numerical execution profile and fresh data.
Standard-library-only identity() can freeze a benchmark without importing Torch.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "farm-full-economic-batched-factor-replay-v2"
LEGACY_PREIMAGES = {
    "src/farm_neat/policy/network.py": "ec805f612e6deb6451701a6bfd6e18b0459dad94b535a6f132c9cb3c9e1d6a91",
    "src/farm_neat/economy/action_space.py": "2df38e7072ab338e8d5008b0ed5515754061627c1cda3250cfe629267ca6d3ee",
    "src/farm_neat/policy/observation.py": "eafb4addbb2e70d4200020ee59f18d3f44a5795513b419b82e9d1c565c4aa214",
    "src/farm_neat/policy/qwen_hybrid.py": "cc85d803c1c5a9ea4a569111a615ff43d260feb15a82dcfd1f89068150c5403d",
    "src/farm_neat/game/constants.py": "b96aec0bcf86858e0cbd032d56a35f6a64c27b1870adb28b19fec36a4bc6376b",
}
PREIMAGES = {**LEGACY_PREIMAGES, "src/farm_neat/policy/network.py": "ec805f612e6deb6451701a6bfd6e18b0459dad94b535a6f132c9cb3c9e1d6a91"}
CPU_EQUIVALENCE = {"logp_atol": 2e-5, "entropy_atol": 2e-5,
                   "gradient_atol": 2e-5, "gradient_rtol": 2e-4}


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity():
    """Bind before outcomes; context encoder/model identity remains caller-owned."""
    for name, expected in PREIMAGES.items():
        if _sha(ROOT/name) != expected:
            raise ValueError("unsupported original replay source: " + name)
    names = ("src/farm_neat/rollout/batched_replay.py", "tests/unit/test_batched_replay.py")
    payload = {"schema": SCHEMA, "source_pins": {**PREIMAGES, **{n: _sha(ROOT/n) for n in names}},
        "mechanism": "DEPENDENT_FACTOR_POSITION_GROUPED_BY_PATH_TYPED_CHOICES_TEMPERATURE_STOP_ONLY_AS_OF_STEP",
        "support": "LEGACY_V1_OR_EXPLICIT_REMAINING_WINDOW_V2; WITNESS_BINDS_TIME",
        "arithmetic": "FP32_BATCHED_HEAD_AND_GRUCELL_FP64_ORDERED_PER_PROGRAM_SUM",
        "cpu_test_tolerances": CPU_EQUIVALENCE, "historical_bit_exact_claim": False,
        "encoder_or_optimizer_changed": False, "training_admission": False}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {**payload, "content_sha256": digest}


def _paths(policy, graph, records):
    """CPU syntax/old-evidence checks mirror the pinned Graph.replay contract."""
    actions = policy.actions
    arch = policy.architecture(graph.config)["content_sha256"]
    sequences = []
    for record in records:
        as_of_step = policy.trace_as_of(record)
        if (record["architecture_sha256"] != arch
                or record["action_schema_sha256"] != actions.ACTION_SCHEMA_SHA256
                or record["training_admission"] is not False):
            raise ValueError("new source/architecture/action trace required, not old29 admission")
        temperature = record["temperature"]
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("finite positive temperature required")
        budget = actions._integer(record["max_commands"], 2, actions.MAX_COMMANDS, "command budget")
        program = actions.validate_program(record["program"])
        if len(program) > budget or len(record["commands"]) != len(program) or len(record["sampling_masks"]) != len(program):
            raise ValueError("program/trace/budget length mismatch")
        sequence = []
        for slot, (encoded, command, saved_masks) in enumerate(zip(record["commands"], program, record["sampling_masks"], strict=True)):
            if actions.decode_command(encoded) != command or len(saved_masks) != len(encoded["factors"]):
                raise ValueError("command factors or mask cardinality mismatch")
            for factor, saved_mask in zip(encoded["factors"], saved_masks, strict=True):
                path, choices, index = factor["path"], tuple(factor["choices"]), factor["index"]
                stop_only = slot == budget-1 and path == "op"
                mask = policy.sampling_mask(path, choices, stop_only=stop_only, as_of_step=as_of_step)
                if saved_mask != mask:
                    raise ValueError("actual sampling budget/syntax mask mismatch")
                if not mask[index]:
                    raise ValueError("selected command lacks declared support")
                # Typed options distinguish bool from integers, even in grouping.
                key = (path, tuple(policy._key(c) for c in choices), temperature, stop_only, as_of_step)
                sequence.append((key, choices, index))
        old = record["selected_logps"]
        if (not isinstance(old, list) or len(old) != len(sequence)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v > 1e-6 for v in old)
                or type(record["joint_logp"]) not in (int, float) or not math.isfinite(record["joint_logp"])
                or not math.isclose(math.fsum(old), record["joint_logp"], rel_tol=1e-6, abs_tol=2e-4)):
            raise ValueError("finite consistent old joint behavior logp required")
        sequences.append(sequence)
    return sequences


def replay(graph, context, records, *, expected_identity, diagnostics=None):
    """Return differentiable (joint_logp[B], path_entropy[B]), both float64.

    Pass identity()'s full frozen mapping explicitly. Ordinary original typed
    traces are read without mutation; stored behavior numbers are validated but
    never replaced. No sampling occurs and no optimizer or RNG state is touched.
    Optional diagnostics describes actual grouping, not games/s or admission.
    """
    if expected_identity != identity():
        raise ValueError("externally frozen batched-replay execution identity required")
    import torch
    from farm_neat.policy import network as policy
    if (type(graph) is not policy.Graph or any(name in graph.__dict__ for name in ("factor", "advance", "replay"))):
        raise ValueError("unmodified existing Graph head required")
    if (context.ndim != 2 or len(context) < 1 or context.shape[1] != graph.config.width
            or len(records) != len(context) or context.dtype != torch.float32
            or context.device != graph.option_embedding.weight.device
            or not bool(torch.isfinite(context).all())
            or torch.is_autocast_enabled(context.device.type)
            or any(p.dtype != torch.float32 or p.device != context.device for p in graph.parameters())):
        raise ValueError("finite FP32 context, same-device weights and one typed record per context required")
    if context.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("explicit FP32 replay requires TF32 disabled")
    if diagnostics is not None and type(diagnostics) is not dict:
        raise ValueError("optional diagnostics must be a dict")
    sequences = _paths(policy, graph, records)
    hidden = list(context.unbind(0))
    totals = [h.sum(dtype=torch.float64)*0 for h in hidden]
    entropies = list(totals)
    groups_total, max_group, total_factors = 0, 0, sum(map(len, sequences))
    for position in range(max(map(len, sequences))):
        groups = {}
        for row, sequence in enumerate(sequences):
            if position < len(sequence):
                key, choices, index = sequence[position]
                groups.setdefault(key, []).append((row, choices, index))
        for key, members in groups.items():
            path, typed_choices, temperature, stop_only, as_of_step = key
            rows, choices, selected = zip(*members, strict=True)
            choices = choices[0]
            h = torch.stack([hidden[row] for row in rows])
            if not bool(torch.isfinite(h).all()):
                raise ValueError("finite prefix required")
            base, depth = policy._path(path)
            ids = torch.tensor([policy.OPTION_IDS[c] for c in typed_choices], device=context.device)
            selected = torch.tensor(selected, device=context.device)
            path_vector = graph.path_embedding.weight[policy.PATH_IDS[base]] + graph.replacement_embedding.weight[depth]
            option_vectors = graph.option_embedding(ids)
            query = graph.factor_query(torch.tanh(h + path_vector))
            logits = (option_vectors[None, :, :] * query[:, None, :]).sum(-1)/math.sqrt(graph.config.width)
            logits = logits + graph.option_bias(ids).squeeze(-1)
            logits = (logits + logits.new_tensor(policy._prior(path, choices))) / temperature
            mask = torch.tensor(policy.sampling_mask(path, choices, stop_only=stop_only, as_of_step=as_of_step),
                                device=context.device, dtype=torch.bool)
            logp = logits.masked_fill(~mask, -torch.inf).log_softmax(-1)
            legal_logp = logp[:, mask]
            entropy = -(legal_logp.exp()*legal_logp).sum(-1)
            selected_logp = logp.gather(1, selected[:, None]).squeeze(1)
            advanced = graph.prefix(graph.option_embedding(ids[selected]) + path_vector, h)
            for slot, row in enumerate(rows):
                # Sequential float64 accumulation follows this program's order.
                totals[row] = totals[row] + selected_logp[slot]
                entropies[row] = entropies[row] + entropy[slot]
                hidden[row] = advanced[slot]
            groups_total += 1
            max_group = max(max_group, len(rows))
    result = torch.stack(totals), torch.stack(entropies)
    if not all(bool(torch.isfinite(value).all()) for value in result):
        raise ValueError("nonfinite batched replay output")
    if diagnostics is not None:
        diagnostics.update({"schema": SCHEMA, "execution_sha256": expected_identity["content_sha256"],
            "programs": len(records), "dependent_positions": max(map(len, sequences)),
            "factor_instances": total_factors, "head_and_gru_batches": groups_total,
            "largest_head_batch": max_group, "training_admission": False})
    return result
