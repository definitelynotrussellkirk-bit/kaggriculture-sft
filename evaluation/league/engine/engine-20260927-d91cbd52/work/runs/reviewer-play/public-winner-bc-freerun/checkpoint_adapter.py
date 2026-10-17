"""Explicit frozen full-economic checkpoint -> live observed-BC actor adapter.

No teacher extraction/replay, prefix labels, full-plan feature conversion, V0 head,
optimizer, economic fallback, or native environment construction lives here.
Torch imports and Graph construction/loading are deliberately lazy.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

BINDING_SCHEMA = "public-winner-bc-checkpoint-binding-v1"
MODEL_SCHEMA = "farm-full-economic-moe-policy-v1"
NUMERICS = "TORCH_FP32_FULL_ECONOMIC_MOE_V1"
INPUT_ADAPTER = "LIVE_ACTOR_OBSERVATION_PACK_NONE_MEMORY_OBSERVED_BC_V1"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def bound(value: dict) -> dict:
    body = {k: copy.deepcopy(v) for k, v in value.items() if k != "content_sha256"}
    return {**body, "content_sha256": digest(body)}


def check_bound(value: dict, label: str) -> None:
    if not isinstance(value, dict) or value != bound(value):
        raise ValueError(f"{label} content digest mismatch")


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=bad_constant)


def write_new(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"explicit SHA-256 required: {label}")


def artifact_path(root: Path, relative: str) -> Path:
    """No path redirection, traversal, external model lookup or basename fallback."""
    root = Path(root).resolve()
    rel = Path(relative)
    if not relative or rel.is_absolute() or ".." in rel.parts or str(rel) != relative:
        raise ValueError("checkpoint must have an exact repository-relative path")
    path = root / rel
    if path.resolve() != path or not path.is_relative_to(root):
        raise ValueError("checkpoint symlink/path substitution refused")
    return path


def validate_binding(binding: dict) -> dict:
    fields = {"schema", "name", "kind", "model_path", "model_sha256", "manifest_file_sha256",
              "manifest_content_sha256", "architecture_sha256", "allow_remaining_window_transition",
              "content_sha256"}
    check_bound(binding, "checkpoint binding")
    if (set(binding) != fields or binding["schema"] != BINDING_SCHEMA
            or binding["kind"] not in ("FROZEN_FULL_ECONOMIC", "INJECTED_FIXTURE_ONLY")
            or not isinstance(binding["name"], str) or not binding["name"].strip()
            or type(binding["allow_remaining_window_transition"]) is not bool):
        raise ValueError("closed explicit full-economic checkpoint binding required")
    if not isinstance(binding["model_path"], str):
        raise ValueError("checkpoint model_path must be a string")
    artifact_path(Path("/"), binding["model_path"])
    for key in ("model_sha256", "manifest_file_sha256", "manifest_content_sha256", "architecture_sha256"):
        _sha(binding[key], key)
    return binding


def _manifest_contract(manifest: dict, *, fixture: bool) -> None:
    check_bound(manifest, "model manifest")
    if manifest.get("schema") != MODEL_SCHEMA or manifest.get("numerics") != NUMERICS:
        raise ValueError("full-economic Graph manifest required; V0/prefix-only checkpoint substitution refused")
    arch = manifest.get("architecture")
    check_bound(arch, "architecture")
    report = manifest.get("parameter_report", {})
    if not fixture and (report.get("total_trainable_parameters", 0) < 30_000_000 or report.get("layers", 0) < 14):
        raise ValueError("production full-economic >=30M/>=14-layer checkpoint required; tiny fixtures are not native policies")
    if not isinstance(manifest.get("tensors"), dict) or not manifest["tensors"]:
        raise ValueError("full frozen tensor manifest required, not just a prefix head")


def pin_checkpoint(root: Path, model: Path, *, name: str, fixture: bool = False,
                   allow_remaining_window_transition: bool = False) -> dict:
    """Read-only binding. Actual load still verifies every tensor and source pin.

    The fixture flag is Python-test-only; the native CLI never exposes it.
    Missing weights are not searched for or replaced with another checkpoint.
    """
    root = Path(root).resolve()
    model = Path(model)
    model = model if model.is_absolute() else root / model
    relative = str(model.relative_to(root))
    model = artifact_path(root, relative)
    sidecar = artifact_path(root, relative + ".json")
    manifest = read_json(sidecar)
    _manifest_contract(manifest, fixture=fixture)
    return validate_binding(bound({"schema": BINDING_SCHEMA, "name": name,
        "kind": "INJECTED_FIXTURE_ONLY" if fixture else "FROZEN_FULL_ECONOMIC",
        "model_path": relative, "model_sha256": sha256_file(model),
        "manifest_file_sha256": sha256_file(sidecar),
        "manifest_content_sha256": manifest["content_sha256"],
        "architecture_sha256": manifest["architecture"]["content_sha256"],
        "allow_remaining_window_transition": allow_remaining_window_transition}))


def verify_artifacts(binding: dict, root: Path, *, fixture: bool = False) -> tuple[Path, dict]:
    validate_binding(binding)
    if not fixture and binding["kind"] != "FROZEN_FULL_ECONOMIC":
        raise ValueError("injected fixture cannot be substituted for a native checkpoint")
    model = artifact_path(root, binding["model_path"])
    sidecar = artifact_path(root, binding["model_path"] + ".json")
    if not model.is_file() or not sidecar.is_file():
        raise ValueError(f"missing required checkpoint files: {model} and {sidecar}; no fallback")
    if sha256_file(model) != binding["model_sha256"] or sha256_file(sidecar) != binding["manifest_file_sha256"]:
        raise ValueError(f"checkpoint/manifest SHA mismatch: {binding['name']}; substitution refused")
    manifest = read_json(sidecar)
    _manifest_contract(manifest, fixture=fixture)
    if (manifest["content_sha256"] != binding["manifest_content_sha256"]
            or manifest["architecture"]["content_sha256"] != binding["architecture_sha256"]):
        raise ValueError("checkpoint manifest/architecture identity mismatch")
    return model, manifest


@dataclass(frozen=True)
class LoadedCheckpoint:
    binding: dict
    graph: Any
    manifest: dict
    loader_kind: str


def require_device(actual, requested: str) -> None:
    """cuda is a device type; its realized current device is e.g. cuda:0."""
    if requested not in ("cpu", "cuda") or getattr(actual, "type", None) != requested:
        raise ValueError("loaded Graph device differs from the explicit inference profile")


def load_frozen(binding: dict, *, root: Path, device: str = "cpu",
                loader: Callable | None = None, fixture: bool = False) -> LoadedCheckpoint:
    """Injection seam: loader(path, device=..., expected_file_sha256=...,
    allow_remaining_window_transition=...) -> (network.Graph, exact_manifest).

    Real execution uses network.load_model, not a guessed checkpoint conversion.
    Tests can inject a tiny in-memory Graph without owning training artifacts.
    The returned manifest MUST equal the pinned on-disk manifest, not just a name.
    """
    from farm_neat.policy import network as policy
    if device not in ("cpu", "cuda"):
        raise ValueError("explicit CPU or CUDA device required")
    model, expected = verify_artifacts(binding, root, fixture=fixture)
    injected = loader is not None
    if injected and not fixture:
        raise ValueError("injected loaders are diagnostic only; native execution requires network.load_model")
    loader = policy.load_model if loader is None else loader
    graph, manifest = loader(model, device=device, expected_file_sha256=binding["model_sha256"],
        allow_remaining_window_transition=binding["allow_remaining_window_transition"])
    if manifest != expected:
        raise ValueError("loaded manifest substitution refused")
    if (not isinstance(graph, policy.Graph) or not callable(getattr(graph, "encode", None))
            or not callable(getattr(graph, "sample", None))):
        raise ValueError("missing free-running interface: full-economic Graph.encode and Graph.sample required; replay_prefix alone is not a decoder")
    if policy.architecture(graph.config)["content_sha256"] != binding["architecture_sha256"]:
        raise ValueError("loaded Graph architecture substitution refused")
    require_device(next(graph.parameters()).device, device)
    graph.eval()
    graph.requires_grad_(False)
    # Detect file changes while the loader was running, without rebinding anything.
    verify_artifacts(binding, root, fixture=fixture)
    return LoadedCheckpoint(copy.deepcopy(binding), graph, copy.deepcopy(manifest),
                            "INJECTED_FIXTURE" if injected else "NETWORK_LOAD_MODEL")


def actor_view(observation: dict, seat: int) -> dict:
    from farm_neat.training.bc import actor_observation
    if type(seat) is not int or seat not in (0, 1) or observation.get("player") != seat:
        raise ValueError("actor observation player/seat mismatch")
    clean = actor_observation(observation)
    if len(clean["farms"]) != 2:
        raise ValueError("exactly two public farms and one own-private view required")
    return clean


def public_configuration(configuration: dict | None) -> dict:
    from farm_neat.policy.observation import CONFIG
    cfg = {} if configuration is None else configuration
    # This is exactly the model's declared public configuration feature vocabulary.
    # In particular the environment's reconstruction seed is not an actor feature.
    public = {key: copy.deepcopy(cfg.get(key, default)) for key, default in CONFIG.items()}
    # pack._world also consumes public curve overrides when the observation does
    # not include market.params. Preserve these, never silently default the curve.
    public["marketParams"] = copy.deepcopy(cfg.get("marketParams", {}))
    return public


def observation_identity(observation: dict) -> dict:
    order = [list(inv) for inv in observation["private"]["inventories"]]
    return {"sha256": digest({"observation": observation, "inventory_key_order": order}),
            "inventory_key_order": order}


def repack_decision(decision: dict) -> dict:
    """Reproduce exact actor tensors after JSON key sorting, without teacher actions."""
    from farm_neat.policy.observation import pack
    obs = copy.deepcopy(decision["observation"])
    inventories = obs["private"]["inventories"]
    orders = decision["inventory_key_order"]
    if len(orders) != len(inventories):
        raise ValueError("inventory order cardinality mismatch")
    for index, keys in enumerate(orders):
        if len(set(keys)) != len(keys) or set(keys) != set(inventories[index]):
            raise ValueError("inventory key-order identity mismatch")
        inventories[index] = {k: inventories[index][k] for k in keys}
    packed = pack(obs, None, decision["configuration"])
    if packed["content_sha256"] != decision["input_sha256"]:
        raise ValueError("reconstructed actor input hash mismatch")
    return packed


class LiveBCActor:
    """Fresh per-seat generator and dawn-gated canonical economic controller.

    Observed-prefix BC packed no latent commitments. Keeping orders=None matches
    that input interface; the executor still retains its actual mechanical memory.
    All full-program suffixes come from Graph.sample, including STOP. They are not
    labels inferred from successful replay actions and not economic defaults here.
    """
    def __init__(self, loaded: LoadedCheckpoint, *, seat: int, sampler_seed: int,
                 controls: dict, emit: Callable[[dict], None]):
        import torch
        from farm_neat.economy.orders import FullSeasonPlan
        if type(seat) is not int or seat not in (0, 1):
            raise ValueError("explicit actor seat required")
        if type(sampler_seed) is not int or not 0 <= sampler_seed < 2**63:
            raise ValueError("explicit bounded actor sampler seed required")
        self.graph, self.binding = loaded.graph, copy.deepcopy(loaded.binding)
        if self.graph.training or any(p.requires_grad for p in self.graph.parameters()):
            raise ValueError("read-only eval Graph required")
        self.seat, self.controls, self.emit = seat, copy.deepcopy(controls), emit
        self.device = next(self.graph.parameters()).device
        self.generator = torch.Generator(device=self.device).manual_seed(sampler_seed)
        self.controller = FullSeasonPlan(f"public-bc-freerun-seat-{seat}",
            publication_steps=self.controls["publication_steps"])
        self.last_step = -1

    def __call__(self, observation: dict, configuration: dict | None = None) -> dict:
        import torch
        from farm_neat.policy import observation as codec
        clean = actor_view(observation, self.seat)
        cfg = public_configuration(configuration)
        step = clean.get("step")
        if type(step) is not int or step != self.last_step + 1 or not 0 <= step < 719:
            raise ValueError("live actor requires each actual consecutive step, beginning at opening")
        self.controller.observe(clean)
        if step in self.controls["publication_steps"]:
            before = self.controller.snapshot()
            packed = codec.pack(clean, None, cfg)
            batch = codec.collate([packed])
            rng_before = hashlib.sha256(self.generator.get_state().cpu().numpy().tobytes()).hexdigest()
            with torch.no_grad():
                context, values, _ = self.graph.encode(*(torch.as_tensor(batch[key], device=self.device)
                    for key in ("rows", "token_mask", "knownness")))
                records, _, _ = self.graph.sample(context, generator=self.generator,
                    max_commands=self.controls["command_budget"], temperature=self.controls["temperature"],
                    as_of_step=step)
            if len(records) != 1:
                raise ValueError("exactly one sampled behavior record per live actor required")
            behavior = records[0]
            # The full native record is preserved before an executor can refuse it.
            self.emit({"type": "DECISION", "step": step, "seat": self.seat,
                "model_binding_sha256": self.binding["content_sha256"], "input_adapter": INPUT_ADAPTER,
                "observation": copy.deepcopy(clean), "configuration": cfg,
                "inventory_key_order": observation_identity(clean)["inventory_key_order"],
                "input_sha256": packed["content_sha256"], "token_ids": list(packed["token_ids"]),
                "snapshot_before": before, "behavior": copy.deepcopy(behavior),
                "value_prediction": float(values[0]), "sampler_rng_before_sha256": rng_before,
                "sampler_rng_after_sha256": hashlib.sha256(self.generator.get_state().cpu().numpy().tobytes()).hexdigest()})
            proposal = self.controller.propose(clean, behavior["program"])
            self.controller = self.controller.publish(clean, proposal)
            self.emit({"type": "PUBLICATION", "step": step, "seat": self.seat,
                       "proposal_sha256": proposal["content_sha256"], "refused_commands": proposal["rejected"]})
        action = self.controller(clean, cfg)
        self.last_step = step
        return action
