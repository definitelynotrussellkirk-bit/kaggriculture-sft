"""Primitive-BC actor: raw Kaggle observation -> actor view -> PolicyNet.act -> joint action.

``make_agent(model_path)`` returns a callable ``agent(observation, configuration)``:

* CPU only; the policy loads lazily on the first call (``load_ms`` is part of that call).
* The observation is reduced with ``farm_neat.training.bc.actor_observation`` (the same
  whitelist the replay pipeline uses) and the configuration with ``public_configuration``
  (the observation module's CONFIG keys plus ``marketParams``, as the frozen BC actor did).
* The thinking-field state is kept per (episode, seat): a new episode starts at step 0 or
  whenever the step does not advance; the state returned by ``act`` is fed back next step.
* Every returned action passes ``sanitize_action`` (plain str/int JSON, <= 10 orders).
  Any exception anywhere in the step returns PASS and counts a fallback.
* One compact stderr line per step when ``log`` is true:
  ``[pbc] step=<t> seat=<s> ms=<wall ms> fb=<0|1>`` (+ ``load_ms`` on the loading call,
  + ``err=<type>`` on a fallback).

Model specs accepted by ``load_policy``: ``stub[:seed]`` (torch-free random grammar policy
with the act() signature, for harness tests), ``random[:seed]`` (randomly initialised
``model.PolicyNet``), a ``primitive-bc-weights-v1`` file written by ``build_archive.py``,
or a training checkpoint (``model`` helper if it exposes one, else a torch dict holding a
config and a state dict).
"""
from __future__ import annotations

import copy
import inspect
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}
MAX_ORDERS = 10
WEIGHTS_FORMAT = "primitive-bc-weights-v1"
STATE_PARAMS = ("state", "field", "field_state", "think_state", "carry", "memory")
LOG_PREFIX = "[pbc]"


def pass_action():
    return copy.deepcopy(PASS_ACTION)


# ----------------------------------------------------------------------------- inputs
def public_configuration(configuration):
    """Declared public configuration vocabulary (seed and extras excluded)."""
    from farm_neat.policy.observation import CONFIG
    cfg = {} if configuration is None else configuration
    public = {key: copy.deepcopy(cfg.get(key, default)) for key, default in CONFIG.items()}
    public["marketParams"] = copy.deepcopy(cfg.get("marketParams", {}) or {})
    return public


def actor_view(observation):
    from farm_neat.training.bc import actor_observation
    return actor_observation(_plain(observation))


def _plain(value):
    """Kaggle Struct / AttrDict trees -> plain dict/list trees (no attribute magic)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# ----------------------------------------------------------------------------- output
def _int(value, what):
    if isinstance(value, bool):
        raise TypeError(f"{what}: bool is not a quantity")
    if isinstance(value, int):
        return int(value)
    # numpy / torch integer scalars
    if hasattr(value, "item") and hasattr(value, "dtype"):
        item = value.item()
        if isinstance(item, int) and not isinstance(item, bool):
            return int(item)
    raise TypeError(f"{what}: integer quantity required, got {type(value).__name__}")


QTY_ORDER_OPS = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL")


def _atom(value, what):
    return str(value) if isinstance(value, str) else _int(value, what)


def _unit(action):
    """[op] / [op, item] / [op, item, int]: the engine indexes item as a key and int()s the count."""
    if not isinstance(action, (list, tuple)) or not action or not isinstance(action[0], str):
        raise TypeError(f"unit action must be a nonempty [op, ...] list: {action!r}")
    if len(action) > 3:
        raise TypeError(f"unit action too long: {action!r}")
    out = [str(action[0])]
    if len(action) >= 2:
        if not isinstance(action[1], str):
            raise TypeError(f"unit argument must be a string: {action!r}")
        out.append(str(action[1]))
    if len(action) >= 3:
        out.append(_int(action[2], "unit quantity"))
    return out


def _order(order):
    """Engine market grammar; any other op (e.g. the codec's NOOP) is kept as an ignored slot.

    The engine skips an order it cannot parse, but the slot still counts toward the
    10-order cap and the per-index lockstep with the rival, so it is not dropped here.
    """
    if not isinstance(order, (list, tuple)) or not order or not isinstance(order[0], str):
        raise TypeError(f"market order must be a nonempty [op, ...] list: {order!r}")
    op = str(order[0])
    if op in ("HIRE", "BUY_LAND"):
        return [op]
    if op in QTY_ORDER_OPS:
        if len(order) != 3 or not isinstance(order[1], str):
            raise TypeError(f"market order must be [op, item, quantity]: {order!r}")
        return [op, str(order[1]), _int(order[2], "order quantity")]
    return [op, *(_atom(v, "order field") for v in order[1:])]


def sanitize_action(action):
    """Plain JSON action with only engine keys; raises on structural invalidity."""
    if not isinstance(action, dict):
        raise TypeError(f"action must be a dict, got {type(action).__name__}")
    hands = action.get("hands", [])
    market = action.get("market", [])
    if not isinstance(hands, (list, tuple)) or not isinstance(market, (list, tuple)):
        raise TypeError("hands and market must be lists")
    if len(market) > MAX_ORDERS:
        raise ValueError(f"{len(market)} market orders > {MAX_ORDERS}")
    return {"farmer": _unit(action.get("farmer", ["PASS"])),
            "hands": [_unit(a) for a in hands],
            "market": [_order(o) for o in market]}


# ----------------------------------------------------------------------------- stub
UNIT_OPS = ("PASS", "NORTH", "SOUTH", "EAST", "WEST", "PLANT", "WATER", "HARVEST",
            "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED",
            "COLLECT_FERTILIZER", "CARE", "DROP", "PICKUP", "PLACE")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")


class StubPolicy:
    """Torch-free random grammar policy with the PolicyNet.act() shape (harness tests only).

    ``act(obs, generator=None, temperature=1.0, state=None) -> (action, state)``; the
    state is a tiny carried dict so the per-episode/seat field plumbing is exercised.
    """

    kind = "stub"

    def __init__(self, seed=0, max_units=13):
        self.seed = int(seed)
        self.max_units = int(max_units)
        self.rng = random.Random(self.seed)

    def act(self, observation, generator=None, temperature=1.0, state=None):
        rng = generator if isinstance(generator, random.Random) else self.rng
        farm = observation["farms"][observation["player"]]
        n_units = min(self.max_units, 1 + len(farm.get("hands", [])))
        units = [self._unit(rng) for _ in range(n_units)]
        market = [self._order(rng) for _ in range(rng.randrange(0, 4))]
        carried = {"steps_seen": (state or {}).get("steps_seen", 0) + 1,
                   "last_step": observation.get("step")}
        return {"farmer": units[0], "hands": units[1:], "market": market}, carried

    @staticmethod
    def _unit(rng):
        op = rng.choice(UNIT_OPS)
        if op == "PLANT":
            return [op, rng.choice(CROPS)]
        if op in ("PICKUP", "PLACE"):
            return [op, rng.choice(PRODUCTS + ANIMALS), rng.randint(1, 5)]
        return [op]

    @staticmethod
    def _order(rng):
        op = rng.choice(("SELL", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "HIRE", "BUY_LAND"))
        if op in ("HIRE", "BUY_LAND"):
            return [op]
        item = {"SELL": PRODUCTS, "BUY_SEED": CROPS, "BUY_ANIMAL": ANIMALS,
                "BUY_PRODUCT": ("WHEAT", "FERTILIZER")}[op]
        return [op, rng.choice(item), rng.randint(1, 5)]

    def export(self):
        return {"format": WEIGHTS_FORMAT, "kind": "stub", "seed": self.seed, "max_units": self.max_units}


# ----------------------------------------------------------------------------- loading
def _model_module():
    import model  # primitive-bc/model.py
    return model


def default_config(model):
    for name in ("default_config", "DefaultConfig"):
        fn = getattr(model, name, None)
        if callable(fn):
            return fn()
    for name in ("PolicyConfig", "ModelConfig", "Config"):
        cls = getattr(model, name, None)
        if isinstance(cls, type):
            return cls()
    for name in ("DEFAULT_CONFIG", "CONFIG"):
        if isinstance(getattr(model, name, None), dict):
            return copy.deepcopy(getattr(model, name))
    return None


def _config_to_plain(config):
    import dataclasses
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config), type(config).__name__
    if isinstance(config, dict):
        return copy.deepcopy(config), "dict"
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return to_dict(), type(config).__name__
    raise TypeError(f"cannot serialise PolicyNet config of type {type(config).__name__}")


def _config_from_plain(model, plain, kind):
    if kind in (None, "dict"):
        return copy.deepcopy(plain)
    cls = getattr(model, kind, None)
    if cls is None:
        raise ValueError(f"model.py has no config class {kind}")
    from_dict = getattr(cls, "from_dict", None)
    return from_dict(plain) if callable(from_dict) else cls(**plain)


def net_config(net):
    for name in ("config", "cfg"):
        if hasattr(net, name):
            return getattr(net, name)
    raise AttributeError("PolicyNet exposes no .config/.cfg; cannot export it")


def random_policy(seed=0, config=None):
    import torch
    model = _model_module()
    torch.manual_seed(int(seed))
    config = default_config(model) if config is None else config
    net = model.PolicyNet(config) if config is not None else model.PolicyNet()
    return net.eval()


def _dequantize(entry, torch):
    if isinstance(entry, dict) and entry.get("q") is not None:
        q, scale = entry["q"], entry["scale"]
        return (q.float() * scale.float().view(-1, *([1] * (q.dim() - 1)))).reshape(entry["shape"])
    return entry.float() if entry.is_floating_point() else entry


def read_weights_file(path):
    """The ``primitive-bc-weights-v1`` blob at ``path``, or None when it is not one."""
    import torch
    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - not a weights_only-loadable file: a training checkpoint
        return None
    return blob if isinstance(blob, dict) and blob.get("format") == WEIGHTS_FORMAT else None


def policy_from_weights(blob):
    """``primitive-bc-weights-v1`` (fp32 / fp16 / int8 storage) -> policy in fp32 eval mode."""
    import torch
    if blob.get("kind") == "stub":
        return StubPolicy(blob["seed"], blob.get("max_units", 13))
    model = _model_module()
    config = _config_from_plain(model, blob["config"], blob.get("config_kind"))
    net = model.PolicyNet(config)
    net.load_state_dict({name: _dequantize(entry, torch) for name, entry in blob["tensors"].items()},
                        strict=True)
    return net.eval()


def load_checkpoint(path):
    """Training checkpoint -> policy: model.py's own loader first, then a generic torch dict."""
    import torch
    model = _model_module()
    loaders = [getattr(model, n, None) for n in ("load_policy", "load_checkpoint")]
    loaders += [getattr(model.PolicyNet, n, None) for n in ("load", "from_checkpoint")]
    first_error = None
    for fn in (f for f in loaders if callable(f)):
        try:
            net = fn(str(path))
            net = net[0] if isinstance(net, tuple) else net
            return net.eval()
        except Exception as error:  # noqa: BLE001 - try the next loader, report the first failure
            first_error = first_error or error
    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        config = next((blob[k] for k in ("config", "model_config", "cfg") if k in blob), None)
        state = next((blob[k] for k in ("model", "state_dict", "model_state", "weights") if k in blob), None)
        if state is None:
            raise ValueError(f"{path}: no state dict under model/state_dict/model_state/weights")
        if isinstance(config, dict):
            from_dict = getattr(getattr(model, "ModelConfig", None), "from_dict", None)
            config = from_dict(config) if callable(from_dict) else config
        net = model.PolicyNet(config) if config is not None else model.PolicyNet()
        net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in state.items()},
                            strict=True)
        return net.eval()
    except Exception:
        if first_error is not None:
            raise first_error
        raise


def is_stub_spec(spec):
    return str(spec).partition(":")[0] == "stub"


def load_policy(spec, *, seed=0):
    """Resolve a model spec to a policy object exposing ``act``."""
    spec = str(spec)
    head, _, tail = spec.partition(":")
    if head == "stub":
        return StubPolicy(int(tail) if tail else seed)
    if head == "random":
        return random_policy(int(tail) if tail else seed)
    path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(spec)
    blob = read_weights_file(path)
    return policy_from_weights(blob) if blob is not None else load_checkpoint(path)


def export_weights(policy, *, storage="fp16", int8_min_numel=4096):
    """Policy -> ``primitive-bc-weights-v1`` blob (dict for torch.save, weights_only-safe)."""
    if isinstance(policy, StubPolicy):
        return policy.export()
    import torch
    if storage not in ("fp32", "fp16", "int8"):
        raise ValueError("storage must be fp32, fp16 or int8")
    plain, kind = _config_to_plain(net_config(policy))
    # Training-only heads (ModelConfig.v6_aux / v6_aux_turn) never ship: drop their tensors and switch the flags off in the
    # exported config so policy_from_weights rebuilds the network without them (B w432 with heads = 100.0 MB fp16 tar.gz).
    if isinstance(plain, dict) and plain.get("v6_aux"):
        plain["v6_aux"], plain["v6_aux_turn"] = False, False
    tensors = {}
    for name, value in policy.state_dict().items():
        if name.startswith("aux_"):
            continue
        value = value.detach().cpu()
        if not value.is_floating_point():
            tensors[name] = value.clone()
        elif storage == "int8" and value.dim() >= 2 and value.numel() >= int8_min_numel:
            flat = value.float().reshape(value.shape[0], -1)
            scale = flat.abs().amax(dim=1).clamp_min(1e-12) / 127.0
            q = torch.round(flat / scale[:, None]).clamp(-127, 127).to(torch.int8)
            tensors[name] = {"q": q.reshape(value.shape).contiguous(), "scale": scale.float(),
                             "shape": list(value.shape)}
        elif storage in ("fp16", "int8"):
            tensors[name] = value.half().clone()
        else:
            tensors[name] = value.float().clone()
    return {"format": WEIGHTS_FORMAT, "kind": "policynet", "storage": storage,
            "config": plain, "config_kind": kind, "tensors": tensors}


# ----------------------------------------------------------------------------- agent
class Agent:
    """Callable Kaggle agent. Never raises out of ``__call__``; PASS on any exception."""

    def __init__(self, model_path, *, seed=0, temperature=1.0, threads=1, log=True,
                 stream=None, policy=None):
        self.model_path = str(model_path)
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.threads = int(threads)
        self.log = bool(log)
        self.stream = stream
        self.policy = policy
        self.generator = None
        self._act_params = None
        self._states = {}           # seat -> {"last_step": int, "state": object}
        self.fallbacks = 0
        self.errors = []
        self.calls = 0
        self.ms = []
        self.load_ms = None
        self.last_fallback = False
        self.last_error = None
        self.load_error = None

    # -- loading
    def _ensure_loaded(self):
        if self.policy is not None and self._act_params is not None:
            return
        if self.load_error is not None:
            # One load attempt per process: a failed load must not re-pay its cost every step.
            raise RuntimeError(f"policy load failed earlier: {self.load_error}")
        begun = time.perf_counter()
        try:
            self._load()
        except Exception as error:
            self.load_error = f"{type(error).__name__}: {error}"
            self.load_ms = (time.perf_counter() - begun) * 1000.0
            raise
        self.load_ms = (time.perf_counter() - begun) * 1000.0

    def _load(self):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        if self.policy is None:
            if not is_stub_spec(self.model_path):
                import torch
                torch.set_num_threads(max(1, self.threads))
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass
            self.policy = load_policy(self.model_path, seed=self.seed)
        if isinstance(self.policy, StubPolicy):
            self.generator = random.Random(self.seed)
        else:
            import torch
            self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        act = getattr(self.policy, "act")
        try:
            self._act_params = tuple(inspect.signature(act).parameters)
        except (TypeError, ValueError):
            self._act_params = ()

    def _initial_state(self):
        for name in ("initial_state", "init_state", "new_state", "reset_state"):
            fn = getattr(self.policy, name, None)
            if callable(fn):
                return fn()
        return None

    def _state_for(self, seat, step):
        slot = self._states.get(seat)
        if slot is None or step == 0 or step <= slot["last_step"]:
            reset = getattr(self.policy, "reset", None)
            if slot is not None and callable(reset):
                reset()
            slot = self._states[seat] = {"last_step": -1, "state": self._initial_state()}
        return slot

    def _act(self, clean, cfg, state):
        params = self._act_params or ()
        kwargs = {}
        if "generator" in params:
            kwargs["generator"] = self.generator
        if "temperature" in params:
            kwargs["temperature"] = self.temperature
        if "configuration" in params:
            kwargs["configuration"] = cfg
        state_param = next((p for p in STATE_PARAMS if p in params), None)
        if state_param is not None:
            kwargs[state_param] = state
        if not isinstance(self.policy, StubPolicy):
            import torch
            with torch.inference_mode():
                out = self.policy.act(clean, **kwargs)
        else:
            out = self.policy.act(clean, **kwargs)
        if isinstance(out, dict):
            return out, state
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], dict):
            return out[0], (out[1] if len(out) > 1 else state)
        action = getattr(out, "action", None)
        if isinstance(action, dict):
            return action, getattr(out, "state", state)
        raise TypeError(f"act() returned {type(out).__name__}, expected action dict or (action, state)")

    # -- call
    def step(self, observation, configuration=None):
        """One decision; raises on any error (``__call__`` wraps it with the fallback)."""
        self._ensure_loaded()
        clean = actor_view(observation)
        if clean.get("step") is None and "day" in clean and "hour" in clean:
            clean["step"] = int(clean["day"]) * 24 + int(clean["hour"])
        cfg = public_configuration(configuration)
        step, seat = int(clean["step"]), int(clean["player"])
        slot = self._state_for(seat, step)
        action, new_state = self._act(clean, cfg, slot["state"])
        action = sanitize_action(action)
        slot["state"], slot["last_step"] = new_state, step
        return action

    def __call__(self, observation, configuration=None):
        begun = time.perf_counter()
        loading = self.load_ms is None
        self.calls += 1
        self.last_fallback, self.last_error = False, None
        try:
            action = self.step(observation, configuration)
        except Exception as error:  # noqa: BLE001 - PASS fallback is the contract
            action = pass_action()
            self.fallbacks += 1
            self.last_fallback = True
            self.last_error = f"{type(error).__name__}: {error}"
            if len(self.errors) < 20:
                self.errors.append(self.last_error[:500])
        ms = (time.perf_counter() - begun) * 1000.0
        self.ms.append(ms)
        if self.log:
            self._log(observation, ms, loading)
        return action

    def _log(self, observation, ms, loading):
        try:
            step = observation.get("step") if hasattr(observation, "get") else None
            seat = observation.get("player") if hasattr(observation, "get") else None
            extra = f" load_ms={self.load_ms:.0f}" if loading and self.load_ms is not None else ""
            if self.last_fallback:
                extra += f" err={self.last_error.split(':', 1)[0]}"
            print(f"{LOG_PREFIX} step={step} seat={seat} ms={ms:.1f} fb={int(self.last_fallback)}{extra}",
                  file=self.stream or sys.stderr, flush=True)
        except Exception:  # noqa: BLE001 - logging never breaks a step
            pass


def make_agent(model_path, *, seed=0, temperature=1.0, threads=1, log=True, stream=None):
    """``model_path`` -> callable(observation, configuration) -> action (lazy CPU load)."""
    return Agent(model_path, seed=seed, temperature=temperature, threads=threads,
                 log=log, stream=stream)
