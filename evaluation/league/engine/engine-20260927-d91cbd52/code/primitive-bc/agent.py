"""Primitive-BC v3 actor: raw Kaggle observation -> canonical actor view -> inputs -> PolicyNet.act.

``make_agent(model_path)`` returns a callable ``agent(observation, configuration)``.  Per step
and seat, exactly the training input path (data.py builds the same example dicts):

  bc.actor_observation (+ step) -> data.canonical_observation -> data.features (pack rows,
  knownness, unit positions) + features.FeatureState.update/compute -> feature_tokens.tokens,
  prev_unit_tok (own previous unit tokens; -1 at step 0 and for hands at hour 0), teacher meta
  -> PolicyNet.encode -> decode_free (deadline=) -> codec.decode_step -> sanitize_action
  -> history action -> PolicyNet.commit(history labels) + FeatureState.record_action(history action).

EVAL copy of the v3.1-fix replica (2026-09-23, adversarial findings LP-1 and skew-4). The
checkpoints of this lineage train from pbc-step-cache-v31, built by this tree's data.py
(build_perspective: fs.record_action(labels.history), prev_unit_tok and the FIFO commit from
codec.encode_step of that label). The replica agent recorded the submitted action instead.
This copy records what the cache recorded; model.py is unchanged.

* History action (``history='audit'``, the Agent default; port of the live v3.2 tree's rule):
  data.label_episode's per-step rule with the v3.1 label rules of this tree (codec.canonical,
  then data._strip: tail strip after the last filling order, SELL that empties the item ->
  SELL_ALL; None -> codec.canonical(None); unrepresentable -> recorded as given, no labels and
  no previous unit tokens), with ``one_seat_audit`` (data.audit_step on the own farm, own
  private state and public market as observed; the rival PASSes with an empty market queue) in
  place of the two-seat engine audit. The history action goes to FeatureState.record_action; its
  codec labels become prev_unit_tok and are what the carried state commits (all -1 when not
  encodable, as the cache stores). The next observation is checked against the audit
  (``history_fill_mismatch`` / ``history_cash_mismatch``: the rival's concurrent orders changed
  the fills, which the actor cannot see). ``history='submitted'``: the replica behaviour.
* Teacher token: the ``team_id`` / ``strength_decile`` args, else (team 0, decile UNKNOWN), the
  signature copy dropout trains (model._copy_dropout: team 0, decile -1 -> index n_strength). The
  replica default (0, 9) never occurs as training metadata. The choice is logged once per load
  (always, independent of ``log``) and reported by ``summary()``.
* Source check: for a train.py checkpoint, the sha256 of this tree's step-cache PINNED files is
  compared with the MANIFEST pins of the step cache its train_config names; the result is logged
  and reported (``cache_pins``). A mismatch means the history rule / inputs here are not the
  ones the checkpoint was trained on.

* Carried state per seat; reset on a new episode (step 0, or a step that does not advance),
  and at hour 0 when the model's carry_scope is 'day'.
* Deadline guard: per-step budget ``budget_s`` (0.8 s) from the call's start, halved when the
  observation's remainingOverageTime bank is below ``low_bank_s`` (10 s); model.act gets the
  absolute time.perf_counter() deadline (remaining hands PASS, market STOP once it passes).
* gc.freeze() after the load; cyclic GC disabled during each turn; torch threads 1.
* Loading: ``background=False`` loads on the first call (blocking, v1 behaviour; that call's
  step budget starts when the load ends, so step 0 is decided by the model, not cut by the load);
  ``background=True`` starts a loader thread at construction and returns PASS until the
  policy is ready (feature history is still kept from step 0 once the light modules load).
* ``agent(observation, configuration, teacher_action=a)``: teacher-forced mode for the
  train/play parity test; the model still runs, ``a`` (a raw submitted action or an already
  canonical label action) goes through the same history rule and is returned unchanged.
* Any exception -> PASS and ``fallbacks += 1``; a failed load is not retried.

Model specs accepted by ``load_policy``: ``stub[:seed]`` (torch-free random grammar policy for
harness tests; bypasses the input path), ``random[:seed]`` (randomly initialised
``model.PolicyNet``), a ``primitive-bc-weights-v3`` (or v1) file written by build_archive.py,
or a train.py checkpoint.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}
MAX_ORDERS = 10
WEIGHTS_FORMAT = "primitive-bc-weights-v3"
WEIGHTS_FORMATS = ("primitive-bc-weights-v3", "primitive-bc-weights-v1")
LOG_PREFIX = "[pbc]"
NOTICE_PREFIX = "[pbc-eval]"
TURNS_PER_DAY = 24
DEFAULT_BUDGET_S = 0.8
LOW_BANK_S = 10.0
HISTORY_MODES = ("audit", "submitted")
TEAM_UNKNOWN = 0                # model: team vocabulary index 0 = unknown
STRENGTH_UNKNOWN = -1           # model: decile outside [0, n_strength) -> index n_strength (unknown)
# the step-cache source pins (pbc-step-cache/build_cache.py PINNED)
PINNED = ("data.py", "codec.py", "features.py", "feature_tokens.py", "theorem_features.py",
          "history_features.py", "strength.py", "STRENGTH.json")


def pass_action():
    return copy.deepcopy(PASS_ACTION)


# ----------------------------------------------------------------------------- observation
def public_configuration(configuration):
    """Declared public configuration vocabulary (seed and extras excluded)."""
    from farm_neat.policy.observation import CONFIG
    cfg = {} if configuration is None else configuration
    public = {key: copy.deepcopy(cfg.get(key, default)) for key, default in CONFIG.items()}
    public["marketParams"] = copy.deepcopy(cfg.get("marketParams", {}) or {})
    return public


def _plain(value):
    """Kaggle Struct / AttrDict trees -> plain dict/list trees (no attribute magic)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def actor_view(observation):
    """bc.actor_observation whitelist with an explicit ``step``."""
    from farm_neat.training.bc import actor_observation
    clean = actor_observation(_plain(observation))
    if clean.get("step") is None and "day" in clean and "hour" in clean:
        clean["step"] = int(clean["day"]) * TURNS_PER_DAY + int(clean["hour"])
    return clean


def canonical_observation(clean):
    """data.canonical_observation: THE shared train/play observation function."""
    import data
    return data.canonical_observation(clean)


def clean_observation(observation):
    return canonical_observation(actor_view(observation))


# ----------------------------------------------------------------------------- history action
_ENGINE = None


def engine_module():
    """The bundled exact interpreter (farm_neat.game.exact_engine.load_engine, SHA-pinned)."""
    global _ENGINE
    if _ENGINE is None:
        from farm_neat.game.exact_engine import load_engine
        _ENGINE = load_engine()
    return _ENGINE


def one_seat_audit(view, action, configuration=None, engine=None):
    """data.audit_step for one seat from its own actor view: own farm, own private state and the
    public market as observed, the rival PASSing with an empty market queue (its farm is only
    read; its private state is empty). The unit phase is exact (units never touch the rival);
    order fills equal the two-seat audit unless the rival's concurrent orders change the market
    or our cash path. -> (own audit dict, own simulated post-market cash)."""
    import data
    from farm_neat.game.exact_engine import AttrDict
    E = engine if engine is not None else engine_module()
    seat = int(view["player"])
    cfg = AttrDict({**data.DEFAULT_CONFIGURATION, **dict(configuration or {})})
    empty = {"shed": {}, "seeds": {}, "inventories": [{}]}
    state = [AttrDict(observation=AttrDict(farms=view["farms"], market=view["market"], step=int(view["step"]),
                                           private=view["private"] if p == seat else empty)) for p in (0, 1)]
    env = AttrDict(engine=E, configuration=cfg, state=state)
    actions = [PASS_ACTION, PASS_ACTION]
    actions[seat] = action if isinstance(action, dict) else {}
    audits, cash = data.audit_step(env, actions)
    return audits[seat], cash[seat]


def history_action(view, canon, action, configuration=None, *, engine=None, stats=None, info=None):
    """The action the v3.1 step cache records for ``action`` (data.label_episode's per-step rule,
    one seat, this tree's v3.1 label rules): None -> codec.canonical(None) (the engine runs it as
    PASS); unrepresentable -> as given; else data._strip(codec.canonical(action), action,
    one_seat_audit(view, action)), or the canonical action when _strip refuses (codec/engine
    disagreement). view: the step's actor view (engine inventory order); canon: its canonical
    observation. -> (history action, own simulated post-market cash or None, encodable: False
    when the action is unrepresentable). ``info`` (a dict) receives the audit."""
    import codec
    import data
    if action is None:
        return codec.canonical(None, canon), None, True
    try:
        lit = codec.canonical(action, canon)
    except codec.Unrepresentable:
        return action, None, False
    audit, cash = one_seat_audit(view, action, configuration, engine)
    if info is not None:
        info["audit"] = audit
    label = data._strip(lit, action, audit, Counter() if stats is None else stats)
    return (lit if label is None else label), cash, True


def fill_prediction(view, action, audit, turns_per_day=TURNS_PER_DAY):
    """Own state the next observation shows if the one-seat audit's order fills are the engine's
    (prices aside): the shed stock of every item a SELL / BUY_PRODUCT / BUY_ANIMAL order touched and
    the hand count (HIRE), both unless the step ends a day (carried goods are deposited, hands
    leave); the unlocked quadrants (BUY_LAND); the seeds of every BUY_SEED crop no unit PLANTs."""
    seat = int(view["player"])
    farm, private = view["farms"][seat], view.get("private") or {}
    action = action if isinstance(action, dict) else {}
    market = action.get("market", [])
    market = market if isinstance(market, list) else []
    units = [action.get("farmer", ["PASS"]), *(action.get("hands") or [])]
    planted = {a[1] for a in units if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and isinstance(a[1], str)}
    shed, seeds, land, hire = {}, {}, 0, 0
    for o, (label, _status, _requested, filled, post) in zip(market, audit["orders"]):
        item = o[1] if isinstance(o, list) and len(o) > 1 and isinstance(o[1], str) else None
        if label in ("SELL", "BUY_PRODUCT", "BUY_ANIMAL") and post is not None and item is not None:
            shed[item] = int(post)
        elif label == "BUY_SEED" and item is not None and item not in planted:
            seeds[item] = seeds.get(item, int((private.get("seeds") or {}).get(item, 0))) + int(filled)
        elif label == "BUY_LAND":
            land += int(filled)
        elif label == "HIRE":
            hire += int(filled)
    pred = {"quadrants": len(farm.get("unlocked_quadrants", [])) + land, "seeds": seeds}
    if int(view["step"]) % turns_per_day != turns_per_day - 1:
        pred.update(shed=shed, hands=len(farm.get("hands", [])) + hire)
    return pred


def fill_mismatch(pred, obs):
    """True when the next observation contradicts fill_prediction (the rival's concurrent orders
    changed our fills, so the recorded history may differ from what training records)."""
    seat = int(obs["player"])
    farm, private = obs["farms"][seat], obs.get("private") or {}
    if len(farm.get("unlocked_quadrants", [])) != pred["quadrants"]:
        return True
    if "hands" in pred and len(farm.get("hands", [])) != pred["hands"]:
        return True
    shed, seeds = private.get("shed") or {}, private.get("seeds") or {}
    return (any(int(shed.get(k, 0)) != v for k, v in pred.get("shed", {}).items())
            or any(int(seeds.get(k, 0)) != v for k, v in pred.get("seeds", {}).items()))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_pins():
    """sha256 of this tree's step-cache PINNED files."""
    return {name: sha256_file(HERE / name) for name in PINNED if (HERE / name).exists()}


def checkpoint_cache_pins(path):
    """Compare this tree's PINNED sources with the MANIFEST pins of the step cache a train.py
    checkpoint's train_config names. -> dict (cache, schema, match, mismatch list) or None when
    the file is not a checkpoint naming a readable cache. Memory-mapped: tensors are not read."""
    import torch
    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except Exception:  # noqa: BLE001 - e.g. a weights blob loaded elsewhere
        return None
    if not isinstance(blob, dict) or not isinstance(blob.get("train_config"), dict):
        return None
    cache = blob["train_config"].get("cache")
    if not cache:
        return {"cache": None, "schema": None, "match": None, "mismatch": [], "note": "no step cache (live workers)"}
    try:
        m = json.loads((Path(cache) / "MANIFEST.json").read_text())
    except (OSError, ValueError) as error:
        return {"cache": cache, "schema": None, "match": None, "mismatch": [], "note": f"manifest unreadable: {error}"}
    pins, here = m.get("pins") or {}, tree_pins()
    bad = sorted(k for k in set(pins) | set(here) if pins.get(k) != here.get(k))
    return {"cache": cache, "schema": m.get("schema"), "match": not bad, "mismatch": bad}


def _labels_or_empty(enc):
    """codec labels of the history action, or empty labels (collate pads them to all -1: what the
    step cache stores, and training commits, for a step without labels)."""
    if enc is not None:
        return enc
    return {"unit_tok": [], "unit_qty": [], "unit_digits": None, "order_tok": [], "order_qty": [],
            "order_digits": None}


# ----------------------------------------------------------------------------- inputs
class InputBuilder:
    """One seat of one episode: per-step model inputs through data.step_inputs (the input
    function data.py builds training examples with), plus the teacher meta.

    ``inputs(obs)`` once per step in order, then ``record(obs, action[, unit_tok])`` with the
    action submitted for that step: its history action (``history``) goes to
    FeatureState.record_action and its unit tokens become the next prev_unit_tok.
    ``history(canon, action)`` alone gives (history action, codec labels) without recording.
    history='submitted' (this class's default) records the action as given; Agent builds its
    seats with history='audit' (the step cache's rule)."""

    def __init__(self, configuration=None, *, team_id=0, strength_decile=9, history="submitted", counters=None):
        import codec
        import features
        if history not in HISTORY_MODES:
            raise ValueError(f"history must be one of {HISTORY_MODES}")
        self.codec = codec
        self.configuration = public_configuration(configuration) if configuration is not None else None
        self.fs = features.FeatureState(self.configuration)
        self.team_id, self.strength_decile = int(team_id), int(strength_decile)
        self.history_mode = history
        self.counters = Counter() if counters is None else counters
        self.last_step = None
        self.last_unit_tok = None
        self.last_history = None
        self.view = None            # this step's actor view (engine inventory order): the audit's input
        self.pred_cash = None       # the audit's own post-market cash of the recorded step
        self.pred_fill = None       # fill_prediction of the recorded step

    def inputs(self, obs):
        """-> (canonical observation, example dict without labels)."""
        import data
        step = int(obs["step"])
        follows = self.last_step is not None and step == self.last_step + 1
        if follows and self.pred_cash is not None:
            try:
                money = float(obs["farms"][int(obs["player"])]["money"])
                self.counters["history_cash_checks"] += 1
                self.counters["history_cash_mismatch"] += money != self.pred_cash
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        if follows and self.pred_fill is not None:
            try:
                bad = fill_mismatch(self.pred_fill, obs)
                self.counters["history_fill_checks"] += 1
                self.counters["history_fill_mismatch"] += bad
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        self.pred_cash = self.pred_fill = None
        prev = self.last_unit_tok if follows else None
        self.view = obs
        canon, ex = data.step_inputs(obs, self.configuration or {}, self.fs, prev)
        meta = dict(ex.get("meta") or {})
        meta.update(step=step, hour=step % TURNS_PER_DAY, game_day=step // TURNS_PER_DAY,
                    seat=int(canon["player"]), team_id=self.team_id, strength_decile=self.strength_decile)
        ex["meta"] = meta
        return canon, ex

    def history(self, canon, action):
        """-> (history action, its codec labels or None when not encodable) of ``action``."""
        hist, encodable, audited = action, True, False
        if self.history_mode == "audit":
            view = self.view if self.view is not None else canon
            info = {}
            try:
                hist, self.pred_cash, encodable = history_action(view, canon, action, self.configuration, info=info)
                audited = True
                self.pred_fill = fill_prediction(view, action, info["audit"]) if "audit" in info else None
            except Exception:  # noqa: BLE001 - e.g. no engine: record as submitted, counted
                self.counters["history_errors"] += 1
                self.pred_cash = self.pred_fill = None
        enc = None
        if encodable:
            try:       # the audited history is canonical already (the cache encodes it as given)
                enc = (self.codec.encode_step(None, canon, canonical_action=hist) if audited
                       else self.codec.encode_step(action, canon))
            except Exception:  # noqa: BLE001 - not encodable: no labels (cache: no prev tokens either)
                enc = None
        return hist, enc

    def record(self, canon, action, unit_tok=None, *, history=None):
        """Record the step's submitted ``action`` (or a precomputed ``history`` pair) -> its labels."""
        hist, enc = history if history is not None else self.history(canon, action)
        self.fs.record_action(hist)
        tok = unit_tok if (self.history_mode == "submitted" and unit_tok is not None) else \
            (enc["unit_tok"] if enc is not None else None)
        self.last_unit_tok = None if tok is None else \
            [int(t) for t in (tok.tolist() if hasattr(tok, "tolist") else tok)]
        self.last_step = int(canon["step"])
        self.last_history = hist
        self.counters["history_recorded"] += 1
        return enc


# ----------------------------------------------------------------------------- output
def _int(value, what):
    if isinstance(value, bool):
        raise TypeError(f"{what}: bool is not a quantity")
    if isinstance(value, int):
        return int(value)
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
    """Engine market grammar; any other op (e.g. the codec's NOOP) is kept as an ignored slot
    (the engine skips it, but the slot still counts toward the cap and the lockstep index)."""
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
    """Torch-free random grammar policy (harness tests only): ``act(obs, generator=None,
    temperature=1.0, state=None) -> (action, state)`` on the actor view; no model inputs."""

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
        carried = {"steps_seen": (state or {}).get("steps_seen", 0) + 1, "last_step": observation.get("step")}
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
    fn = getattr(model, "default_config", None)
    if callable(fn):
        return fn()
    return model.ModelConfig()


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
        cls = getattr(model, "ModelConfig", None)
        from_dict = getattr(cls, "from_dict", None)
        return from_dict(plain) if callable(from_dict) else copy.deepcopy(plain)
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
    net = model.PolicyNet(default_config(model) if config is None else config)
    return net.eval()


def _dequantize(entry, torch):
    if isinstance(entry, dict) and entry.get("q") is not None:
        q, scale = entry["q"], entry["scale"]
        return (q.float() * scale.float().view(-1, *([1] * (q.dim() - 1)))).reshape(entry["shape"])
    return entry.float() if entry.is_floating_point() else entry


def read_weights_file(path):
    """The weights blob at ``path``, or None when it is not one (e.g. a training checkpoint)."""
    import torch
    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - not weights_only-loadable: a training checkpoint
        return None
    return blob if isinstance(blob, dict) and blob.get("format") in WEIGHTS_FORMATS else None


def policy_from_weights(blob):
    """weights blob (fp32 / fp16 / int8 storage) -> policy in fp32 eval mode."""
    import torch
    if blob.get("kind") == "stub":
        return StubPolicy(blob["seed"], blob.get("max_units", 13))
    model = _model_module()
    config = _config_from_plain(model, blob["config"], blob.get("config_kind"))
    net = model.PolicyNet(config)
    net.load_state_dict({name: _dequantize(entry, torch) for name, entry in blob["tensors"].items()}, strict=True)
    if blob.get("play_teacher"):
        net.play_teacher = dict(blob["play_teacher"])
    return net.eval()


def load_checkpoint(path):
    """Training checkpoint -> policy: model.py's own loader first, then {config, model} dicts."""
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
        blob = torch.load(str(path), map_location="cpu", weights_only=True)
        config = next((blob[k] for k in ("config", "model_config", "cfg") if k in blob), None)
        state = next((blob[k] for k in ("model", "state_dict", "model_state", "weights") if k in blob), None)
        if state is None:
            raise ValueError(f"{path}: no state dict under model/state_dict/model_state/weights")
        if isinstance(config, dict):
            config = model.ModelConfig.from_dict(config)
        net = model.PolicyNet(config) if config is not None else model.PolicyNet()
        net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}, strict=True)
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
    """Policy -> weights blob (dict for torch.save, weights_only-safe)."""
    if isinstance(policy, StubPolicy):
        return policy.export()
    import torch
    if storage not in ("fp32", "fp16", "int8"):
        raise ValueError("storage must be fp32, fp16 or int8")
    plain, kind = _config_to_plain(net_config(policy))
    tensors = {}
    for name, value in policy.state_dict().items():
        value = value.detach().cpu()
        if not value.is_floating_point():
            tensors[name] = value.clone()
        elif storage == "int8" and value.dim() >= 2 and value.numel() >= int8_min_numel:
            flat = value.float().reshape(value.shape[0], -1)
            scale = flat.abs().amax(dim=1).clamp_min(1e-12) / 127.0
            q = torch.round(flat / scale[:, None]).clamp(-127, 127).to(torch.int8)
            tensors[name] = {"q": q.reshape(value.shape).contiguous(), "scale": scale.float(), "shape": list(value.shape)}
        elif storage in ("fp16", "int8"):
            tensors[name] = value.half().clone()
        else:
            tensors[name] = value.float().clone()
    blob = {"format": WEIGHTS_FORMAT, "kind": "policynet", "storage": storage,
            "config": plain, "config_kind": kind, "tensors": tensors}
    if getattr(policy, "play_teacher", None):
        blob["play_teacher"] = dict(policy.play_teacher)
    return blob


# ----------------------------------------------------------------------------- agent
class SeatSlot:
    __slots__ = ("builder", "state", "last_step", "stub_state")

    def __init__(self):
        self.builder, self.state, self.last_step, self.stub_state = None, None, -1, None


class Agent:
    """Callable Kaggle agent. Never raises out of ``__call__``; PASS on any exception."""

    def __init__(self, model_path, *, seed=0, temperature=0.0, threads=1, log=True, stream=None, policy=None,
                 background=False, budget_s=DEFAULT_BUDGET_S, low_bank_s=LOW_BANK_S, team_id=None,
                 strength_decile=None, history="audit"):
        if history not in HISTORY_MODES:
            raise ValueError(f"history must be one of {HISTORY_MODES}")
        self.model_path = str(model_path)
        self.seed, self.temperature, self.threads = int(seed), float(temperature), int(threads)
        self.log, self.stream = bool(log), stream
        self.policy = policy
        self.budget_s, self.low_bank_s = float(budget_s), float(low_bank_s)
        self.team_id, self.strength_decile = team_id, strength_decile
        self.teacher_source = "explicit" if team_id is not None and strength_decile is not None else None
        self.history = history
        self.history_counts = Counter()      # InputBuilder counters of every seat and episode
        self.cache_pins = None
        self.generator = None
        self._act_kwargs = None
        self._decode_kwargs = {}
        self._split_act = False
        self._seats = {}
        self.fallbacks, self.errors, self.calls, self.ms = 0, [], 0, []
        self.load_ms, self.load_error = None, None
        self.last_fallback, self.last_error = False, None
        self.deadline_hits, self.pass_while_loading, self.model_steps = 0, 0, 0
        self.first_model_step = None
        self.last = None
        self._ready = threading.Event()
        self._light = threading.Event()
        self._thread = None
        if policy is not None:
            self._finish_load(time.perf_counter())
        elif background:
            self._thread = threading.Thread(target=self._background_load, name="pbc-load", daemon=True)
            self._thread.start()

    # -- loading
    def _background_load(self):
        begun = time.perf_counter()
        try:
            self._load()
        except BaseException as error:  # noqa: BLE001
            self.load_error = f"{type(error).__name__}: {error}"
            self.load_ms = (time.perf_counter() - begun) * 1000.0
            self._light.set()
            self._ready.set()

    def _load(self):
        begun = time.perf_counter()
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        if not is_stub_spec(self.model_path):
            import codec  # noqa: F401 - light modules first: PASS-while-loading keeps features causal
            import data  # noqa: F401
            import feature_tokens  # noqa: F401
            import features
            warm = getattr(features, "warm", None)
            if callable(warm):
                warm()
            if self.history == "audit":
                try:
                    engine_module()          # the history audit's interpreter before the first step
                except Exception:  # noqa: BLE001 - history falls back per step (history_errors)
                    pass
            self._light.set()
            import torch
            torch.set_num_threads(max(1, self.threads))
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        self.policy = load_policy(self.model_path, seed=self.seed)
        if not is_stub_spec(self.model_path) and Path(self.model_path).is_file():
            try:
                self.cache_pins = checkpoint_cache_pins(self.model_path)
            except Exception as error:  # noqa: BLE001 - diagnostics only
                self.cache_pins = {"match": None, "note": f"{type(error).__name__}: {error}"}
            if self.cache_pins is not None:
                self._notice(f"cache pins match={self.cache_pins.get('match')} cache={self.cache_pins.get('cache')} "
                             f"mismatch={self.cache_pins.get('mismatch')}"
                             + (f" note={self.cache_pins['note']}" if self.cache_pins.get("note") else ""))
        self._finish_load(begun)

    def _notice(self, text):
        """One-off load-time notice (teacher token, source check): always printed to stderr, independent
        of ``log`` / ``stream`` (those carry the per-step lines)."""
        try:
            print(f"{NOTICE_PREFIX} {text}", file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001 - logging never breaks a load
            pass

    def _finish_load(self, begun):
        if isinstance(self.policy, StubPolicy):
            self.generator = random.Random(self.seed)
            self._act_kwargs = {}
        else:
            import torch
            self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
            try:
                params = inspect.signature(self.policy.act).parameters
            except (TypeError, ValueError):
                params = {}
            kw = {}
            if "generator" in params:
                kw["generator"] = self.generator
            if "temperature" in params:
                kw["temperature"] = self.temperature
            self._act_kwargs = kw
            self._act_commit = "commit_labels" in params     # carried state writes the submitted action
            # free play: encode -> decode_free -> commit(history labels), the steps PolicyNet.act runs
            self._split_act = all(callable(getattr(self.policy, n, None)) for n in ("encode", "decode_free", "commit"))
            if self._split_act:
                try:
                    dparams = inspect.signature(self.policy.decode_free).parameters
                except (TypeError, ValueError):
                    dparams = {}
                self._decode_kwargs = {k: kw[k] for k in ("generator", "temperature") if k in kw and k in dparams}
            # teacher token: explicit args, else (team 0, decile UNKNOWN), the copy-dropout signature;
            # a checkpoint's play_teacher is not used (v3.1-lineage checkpoints carry none)
            if self.team_id is None or self.strength_decile is None:
                self.teacher_source = "default (team unknown, decile unknown: copy-dropout signature)"
                if getattr(self.policy, "play_teacher", None):
                    self.teacher_source += f"; checkpoint play_teacher {dict(self.policy.play_teacher)} ignored"
                self.team_id = TEAM_UNKNOWN if self.team_id is None else self.team_id
                self.strength_decile = STRENGTH_UNKNOWN if self.strength_decile is None else self.strength_decile
            n_strength = getattr(getattr(self.policy, "config", None), "n_strength", 10)
            dec = int(self.strength_decile)
            dec_txt = f"{dec}" if 0 <= dec < n_strength else f"UNKNOWN({dec} -> index {n_strength})"
            self._notice(f"teacher token team_id={int(self.team_id)} strength_decile={dec_txt} "
                         f"source={self.teacher_source} history={self.history}")
        gc.collect()
        gc.freeze()
        self.load_ms = (time.perf_counter() - begun) * 1000.0
        self._light.set()
        self._ready.set()

    def _ensure_loaded(self, deadline):
        if self._ready.is_set():
            if self.load_error is not None:
                raise RuntimeError(f"policy load failed earlier: {self.load_error}")
            return True
        if self._thread is None:                    # synchronous lazy load, attempted once
            begun = time.perf_counter()
            try:
                self._load()
            except Exception as error:
                self.load_error = f"{type(error).__name__}: {error}"
                self.load_ms = (time.perf_counter() - begun) * 1000.0
                self._ready.set()
                raise
            return True
        self._ready.wait(timeout=max(0.0, deadline - time.perf_counter() - 0.05))
        if self._ready.is_set() and self.load_error is not None:
            raise RuntimeError(f"policy load failed earlier: {self.load_error}")
        return self._ready.is_set()

    @property
    def loaded(self):
        return self._ready.is_set() and self.load_error is None

    # -- per seat
    def _slot(self, seat, step):
        slot = self._seats.get(seat)
        if slot is None or step == 0 or step <= slot.last_step:
            slot = self._seats[seat] = SeatSlot()
        return slot

    def _builder(self, slot, configuration):
        if slot.builder is None:
            slot.builder = InputBuilder(configuration, history=self.history, counters=self.history_counts)
        slot.builder.team_id = TEAM_UNKNOWN if self.team_id is None else int(self.team_id)
        slot.builder.strength_decile = STRENGTH_UNKNOWN if self.strength_decile is None else int(self.strength_decile)
        return slot.builder

    def budget(self, observation):
        bank = observation.get("remainingOverageTime") if hasattr(observation, "get") else None
        try:
            low = bank is not None and float(bank) < self.low_bank_s
        except (TypeError, ValueError):
            low = False
        return self.budget_s / 2 if low else self.budget_s

    # -- call
    def step(self, observation, configuration=None, *, begun=None, teacher_action=None):
        """One decision; raises on any error (``__call__`` wraps it with the fallback)."""
        begun = time.perf_counter() if begun is None else begun
        deadline = begun + self.budget(observation)
        stub = is_stub_spec(self.model_path) or isinstance(self.policy, StubPolicy)
        sync_load = not self._ready.is_set() and self._thread is None     # this call loads the policy
        loaded = self._ensure_loaded(deadline)
        if sync_load and loaded:
            # the step budget starts after a synchronous load (it took 2.6-3.8 s against the 0.8 s
            # budget and cut step 0 to PASS/STOP, which was then committed to the carried state)
            deadline = time.perf_counter() + self.budget(observation)
        clean = actor_view(observation)
        step, seat = int(clean["step"]), int(clean["player"])
        slot = self._slot(seat, step)
        if loaded and isinstance(self.policy, StubPolicy):
            action, slot.stub_state = self.policy.act(clean, generator=self.generator, state=slot.stub_state)
            action = sanitize_action(action)
            slot.last_step = step
            return sanitize_action(teacher_action) if teacher_action is not None else action
        if not loaded:
            action = sanitize_action(teacher_action) if teacher_action is not None else pass_action()
            if self._light.is_set():
                b = self._builder(slot, configuration)
                canon, _ = b.inputs(clean)
                b.record(canon, action)
            self.pass_while_loading += 1
            slot.last_step = step
            return action
        b = self._builder(slot, configuration)
        clean, inputs = b.inputs(clean)
        try:
            return self._model_step(slot, b, clean, inputs, step, seat, deadline, teacher_action)
        except Exception:
            b.record(clean, pass_action())            # the engine gets PASS; keep update/record paired
            slot.last_step = step
            raise

    def _model_step(self, slot, b, clean, inputs, step, seat, deadline, teacher_action):
        import torch
        state = slot.state
        day_scope = getattr(getattr(self.policy, "config", None), "carry_scope", "episode") == "day"
        if (state is None or (day_scope and step % TURNS_PER_DAY == 0)) and hasattr(self.policy, "initial_state"):
            state = self.policy.initial_state(1)
        batch = collate_inputs(self.policy, inputs)
        kw = self._act_kwargs
        audit = b.history_mode == "audit"
        teacher = None
        hist = None
        commit = None
        if teacher_action is not None:
            # teacher-forced: the history rule reads the action as submitted (as data.py reads the replay's
            # wire action); a malformed one (e.g. an empty order) is returned unsanitized
            try:
                teacher = sanitize_action(teacher_action)
            except (TypeError, ValueError):
                if not audit:
                    raise                     # replica behaviour of history='submitted'
                teacher = copy.deepcopy(teacher_action)
            hist = b.history(clean, copy.deepcopy(teacher_action) if audit else teacher)
            if getattr(self, "_act_commit", False):
                # the carried state writes the teacher's history labels, as training commits the cache labels
                commit = self._commit_labels(inputs, hist[1])
                kw = dict(kw, commit_labels=commit)
        # free play: decode, then commit the history labels of the submitted action (as training commits
        # the label action); a wrapped act (parity capture) or history='submitted' uses act()
        split = (teacher is None and self._split_act and audit and "act" not in getattr(self.policy, "__dict__", {}))
        ctx = None
        with torch.inference_mode():
            if split:
                ctx = self.policy.encode(batch, state)
                out = self.policy.decode_free(ctx, deadline=deadline, **self._decode_kwargs)
            else:
                out, new_state = self.policy.act(batch, state, deadline=deadline, **kw)
        hit = out.get("deadline_hit") if isinstance(out, dict) else None
        if (bool(hit.any()) if hasattr(hit, "any") else bool(hit)) or time.perf_counter() > deadline:
            self.deadline_hits += 1
        to_lists = getattr(self.policy, "codec_labels", None)
        labels = to_lists(out, 0) if callable(to_lists) else _plain_labels(out)
        own = sanitize_action(b.codec.decode_step(labels, clean))
        if teacher is not None:
            action = teacher
        else:
            action = own
            hist = b.history(clean, own)
        if split:
            commit = self._commit_labels(inputs, hist[1])
            with torch.inference_mode():
                new_state = self.policy.commit(ctx, commit)
        # history='submitted' free play keeps the replica's decoded previous unit tokens; audit mode and
        # teacher-forced steps use the history labels' unit tokens
        b.record(clean, action, labels.get("unit_tok") if teacher is None else None, history=hist)
        slot.state, slot.last_step = new_state, step
        self.model_steps += 1
        if self.first_model_step is None:
            self.first_model_step = step
        self.last = {"inputs": inputs, "labels": labels, "state": new_state, "own_action": own, "step": step,
                     "seat": seat, "history": hist[0], "history_labels": hist[1], "commit_labels": commit}
        return action

    def _commit_labels(self, inputs, enc):
        """B=1 label tensors (model.collate) of the history labels; all -1 when not encodable."""
        return collate_inputs(self.policy, dict(inputs, labels=_labels_or_empty(enc)))["labels"]

    def __call__(self, observation, configuration=None, teacher_action=None):
        begun = time.perf_counter()
        was_enabled = gc.isenabled()
        gc.disable()
        loading = self.load_ms is None
        self.calls += 1
        self.last_fallback, self.last_error = False, None
        try:
            action = self.step(observation, configuration, begun=begun, teacher_action=teacher_action)
        except Exception as error:  # noqa: BLE001 - PASS fallback is the contract
            action = pass_action()
            self.fallbacks += 1
            self.last_fallback = True
            self.last_error = f"{type(error).__name__}: {error}"
            if len(self.errors) < 20:
                self.errors.append(self.last_error[:500])
        finally:
            if was_enabled:
                gc.enable()
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

    def summary(self):
        h = self.history_counts
        return {"calls": self.calls, "fallbacks": self.fallbacks, "deadline_hits": self.deadline_hits,
                "pass_while_loading": self.pass_while_loading, "model_steps": self.model_steps,
                "first_model_step": self.first_model_step, "load_ms": self.load_ms, "load_error": self.load_error,
                "history": self.history, "label_rules": "v3.1 (this tree's data._strip)",
                "teacher_token": {"team_id": self.team_id, "strength_decile": self.strength_decile,
                                  "source": self.teacher_source},
                "cache_pins": self.cache_pins,
                "history_recorded": h["history_recorded"], "history_errors": h["history_errors"],
                "history_cash_checks": h["history_cash_checks"], "history_cash_mismatch": h["history_cash_mismatch"],
                "history_fill_checks": h["history_fill_checks"], "history_fill_mismatch": h["history_fill_mismatch"]}


def collate_inputs(policy, example):
    """One example dict -> the B=1 batch PolicyNet.act takes (model.collate)."""
    model = sys.modules.get(type(policy).__module__)
    fn = getattr(model, "collate", None) if model is not None else None
    fn = fn or getattr(_model_module(), "collate")
    return fn([example], n_order_slots=getattr(policy, "O", None), digit_slots=getattr(policy, "D", None))


def _plain_labels(labels):
    """model.act labels (tensors / arrays / lists, optional batch dim 1) -> plain lists."""
    out = {}
    for k, v in dict(labels).items():
        if hasattr(v, "detach"):
            v = v.detach().cpu()
        if hasattr(v, "tolist"):
            v = v.tolist()
        out[k] = v
    ut = out.get("unit_tok")
    if isinstance(ut, list) and ut and isinstance(ut[0], list) and len(ut) == 1:
        out = {k: (v[0] if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list) else v) for k, v in out.items()}
    return out


def make_agent(model_path, *, seed=0, temperature=0.0, threads=1, log=True, stream=None, background=False,
               budget_s=DEFAULT_BUDGET_S, low_bank_s=LOW_BANK_S, team_id=None, strength_decile=None,
               history="audit"):
    """``model_path`` -> callable(observation, configuration) -> action (lazy CPU load)."""
    return Agent(model_path, seed=seed, temperature=temperature, threads=threads, log=log, stream=stream,
                 background=background, budget_s=budget_s, low_bank_s=low_bank_s, team_id=team_id,
                 strength_decile=strength_decile, history=history)
