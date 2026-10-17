"""Small experimental NN for the semantic-command curriculum.

Fresh schema/initialization, not relabelled 302-value rows or submitted weights.
Uses the maintained PPO function's unchanged bytecode with explicit model/feature
bindings in a private globals dictionary. No second policy-gradient algorithm.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import types

import numpy as np

from farm_neat.training import schedule
from farm_neat.training.schedule_model import NUMERICS_CANONICAL, bound, canonical, file_sha, tensor_sha
from farm_neat.economy.orders_v1 import SemanticOrders

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "farm-semantic-command-actor-v1"
SITES = ((4, 3), (3, 4), (3, 3), (2, 4))
ACTIONS = [{"kind": "NO_EDIT"}]
for tile in SITES:
    ACTIONS.extend({"kind": "WHEAT", "tile": list(tile), "cycles": n} for n in (1, 3))
    ACTIONS.append({"kind": "COW", "tile": list(tile)})
for item, thresholds in (("MILK", (80, 120, 160, 200, 240, 320)),
                         ("FERTILIZER", (0, 80, 120)), ("WHEAT", (0, 25, 45))):
    ACTIONS.extend({"kind": "SELL", "item": item, "threshold": threshold} for threshold in thresholds)
ACTIONS.extend({"kind": "HANDS", "target": target} for target in (0, 1, 2, 4))
ACTION_IDS = tuple(range(len(ACTIONS)))
assert len(ACTIONS) == 29
ACTION_SCHEMA = bound({"schema": "farm-semantic-command-vocabulary-v1", "actions": ACTIONS,
    "scope": "FOUR_NW_TILES_WHEAT_CYCLES_COWS_SALE_THRESHOLDS_DAILY_HANDS",
    "disjoint_from_historical_29_index_meanings": True})
WIDTH = 64
SHAPES = {"shared.weight": (32, WIDTH), "shared.bias": (32,),
          "actor_head.weight": (1, 32), "actor_head.bias": (1,),
          "value_head.weight": (1, 32), "value_head.bias": (1,)}
TRAINABLE = tuple(SHAPES)
SOURCE_PATHS = ("src/farm_neat/economy/orders_v1.py", "src/farm_neat/policy/semantic/policy_v1.py",
    "src/farm_neat/training/schedule.py", "src/farm_neat/training/schedule_model.py",
    "src/farm_neat/game/exact_engine.py", "src/farm_neat/economy/heuristic/policy.py", "src/farm_neat/economy/heuristic/state.py",
    "src/farm_neat/economy/heuristic/constants.py", "src/farm_neat/economy/heuristic/econ.py", "src/farm_neat/game/constants.py")


def sources():
    return {name: file_sha(ROOT / name) for name in SOURCE_PATHS}


def initial_orders():
    orders = SemanticOrders(hands=2)
    for item in ("WHEAT", "MILK", "FERTILIZER"):
        orders.sell_when(item, price_above=0)
    return orders


def pack(observation, orders):
    player = observation["player"]
    own, opp = observation["farms"][player], observation["farms"][1-player]
    private, prices = observation["private"], observation["market"]["prices"]
    def counts(farm, kind):
        return sum(isinstance(t, dict) and (t.get("crop") == kind or t.get("animal") == kind)
                   for row in farm["tiles"] for t in row)
    common = [observation["day"]/29, own["money"]/10000, opp["money"]/10000,
              len(own["hands"])/4, orders.hands/4, counts(own, "WHEAT")/4,
              counts(own, "COW")/4, counts(opp, "WHEAT")/25, counts(opp, "COW")/25]
    for item in ("WHEAT", "MILK", "FERTILIZER"):
        common.extend([prices.get(item, 0)/200, private["shed"].get(item, 0)/100,
                       sum(inv.get(item, 0) for inv in private["inventories"])/100,
                       orders.sale_rules.get(item, {"price_above": 0})["price_above"]/320])
    shops = observation["town"].get("unlocked_shops", [])
    common.extend(shops.count(s)/8 for s in ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
        "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"))
    common.extend([private["seeds"].get("WHEAT", 0)/8,
                   sum(o.get("completed", 0) for o in orders.orders.values())/12])
    rows, mask = [], []
    for index, command in enumerate(ACTIONS):
        kind = command["kind"]
        extras = [float(kind == k) for k in ("NO_EDIT", "WHEAT", "COW", "SELL", "HANDS")]
        extras.extend(float(command.get("item") == k) for k in ("WHEAT", "MILK", "FERTILIZER"))
        extras.extend([command.get("cycles", 0)/3, command.get("threshold", 0)/320,
                       command.get("target", 0)/4])
        allowed = True
        tile_features = [0.] * 13
        if "tile" in command:
            x, y = command["tile"]
            tile = own["tiles"][y][x]
            current = orders.orders.get(f"{x},{y}")
            tile_dict = tile if isinstance(tile, dict) else {}
            tile_features = [x/9, y/9, float(tile is None), float(tile_dict.get("kind") == "WEED"),
                float(tile_dict.get("crop") == "WHEAT"), float(tile_dict.get("animal") == "COW"),
                tile_dict.get("yield_units", 0)/6,
                (observation["day"] - tile_dict.get("planted_day", observation["day"]))/30,
                float(tile_dict.get("watered_today", False)), float(tile_dict.get("fed_today", False)),
                (current or {}).get("cycles", 0)/3, (current or {}).get("completed", 0)/3,
                float(current is not None)]
            allowed = tile != "LOCKED"
            if current:
                # Small curriculum: one stable use per site, no implicit disposal.
                if kind == "COW":
                    allowed = False
                else:
                    allowed = (allowed and current["kind"] == "CROP"
                               and current["cycles"] != command["cycles"]
                               and current["completed"] < command["cycles"])
            elif isinstance(tile, dict) and tile.get("kind") != "WEED":
                allowed = False
        elif kind == "SELL":
            allowed = orders.sale_rules.get(command["item"], {}).get("price_above") != command["threshold"]
        elif kind == "HANDS":
            allowed = orders.hands != command["target"]
        features = [*common, *extras, *tile_features]
        rows.append([*features, *([0.] * (WIDTH - 1 - len(features))), index/28])
        mask.append(bool(allowed))
    return np.asarray(rows, np.float32), np.asarray(mask, np.bool_)


def apply_command(orders, index):
    command = ACTIONS[index]
    if command["kind"] == "WHEAT":
        orders.farm_tile(command["tile"], cycles=command["cycles"])
    elif command["kind"] == "COW":
        orders.keep_animal(command["tile"])
    elif command["kind"] == "SELL":
        orders.sell_when(command["item"], price_above=command["threshold"])
    elif command["kind"] == "HANDS":
        orders.hands = command["target"]


def validate_inputs(rows, mask):
    rows, mask = np.asarray(rows, np.float32), np.asarray(mask)
    if (rows.shape != (29, WIDTH) or not np.isfinite(rows).all() or mask.shape != (29,)
            or mask.dtype != np.bool_ or not mask[0]
            or not np.array_equal(rows[:, -1], np.asarray(ACTION_IDS, np.float32)/28)):
        raise ValueError("exact semantic29 x64 rows and mask required")
    return rows, mask


def save_model(path, weights, metadata):
    if set(weights) != set(SHAPES) or any(weights[k].shape != s or weights[k].dtype != np.float32
          or not np.isfinite(weights[k]).all() for k, s in SHAPES.items()):
        raise ValueError("exact finite semantic actor tensors required")
    manifest = bound({**metadata, "schema": SCHEMA, "action_schema_sha256": ACTION_SCHEMA["content_sha256"],
        "source_pins": sources(), "numerics": NUMERICS_CANONICAL,
        "tensor_sha256": tensor_sha(weights), "candidate_width": WIDTH})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **weights,
            __manifest_json__=np.frombuffer(canonical(manifest).encode(), dtype=np.uint8))
    return manifest


def load_model(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 1000000:
        raise ValueError("small regular semantic checkpoint required")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {*SHAPES, "__manifest_json__"}:
            raise ValueError("exact semantic checkpoint members required")
        manifest = json.loads(archive["__manifest_json__"].tobytes())
        weights = {k: archive[k].copy() for k in SHAPES}
    if (manifest != bound(manifest) or manifest["schema"] != SCHEMA
            or manifest["source_pins"] != sources()
            or manifest["action_schema_sha256"] != ACTION_SCHEMA["content_sha256"]
            or manifest["tensor_sha256"] != tensor_sha(weights)):
        raise ValueError("semantic checkpoint source/meaning/tensor mismatch")
    return manifest, weights


def initialize(path, seed=908080001):
    rng = np.random.default_rng(seed)
    weights = {k: (rng.normal(0, .03, shape).astype(np.float32) if k.endswith("weight")
                   else np.zeros(shape, np.float32)) for k, shape in SHAPES.items()}
    weights["value_head.weight"].fill(0)
    return save_model(path, weights, {"optimizer_steps": 0, "rollout_iteration": 0,
        "temperature": 1., "initialization": "FROM_SCRATCH_SMALL_SEMANTIC_CURRICULUM",
        "initialization_seed": seed, "canonical_submission": False})


class Policy:
    def __init__(self, manifest, weights):
        self.manifest, self.weights = manifest, weights
        self.w = {k: v.astype(np.float64) for k, v in weights.items()}

    def distribution(self, rows, mask):
        rows, mask = validate_inputs(rows, mask)
        w = self.w
        h = np.tanh(rows.astype(np.float64) @ w["shared.weight"].T + w["shared.bias"])
        scores = (h @ w["actor_head.weight"].T + w["actor_head.bias"])[:, 0]
        value = float((h[mask].mean(0) @ w["value_head.weight"].T + w["value_head.bias"])[0])
        logits = scores/self.manifest["temperature"]
        shifted = logits - logits[mask].max()
        logp = np.where(mask, shifted - np.log(np.exp(shifted[mask]).sum()), -np.inf)
        return scores, value, mask, np.where(mask, np.exp(logp), 0), logp


def torch_graph(weights, numerics=NUMERICS_CANONICAL):
    import torch
    from torch import nn
    class Graph(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(WIDTH, 32)
            self.actor_head = nn.Linear(32, 1)
            self.value_head = nn.Linear(32, 1)
            self.load_state_dict({k: torch.tensor(v) for k, v in weights.items()})
        def forward(self, rows, enabled):
            def linear(layer, x):
                return nn.functional.linear(x, layer.weight.double(), layer.bias.double())
            h = torch.tanh(linear(self.shared, rows.double()))
            pooled = (h * enabled[..., None]).sum(1)/enabled.sum(1, keepdim=True)
            return linear(self.actor_head, h).squeeze(-1), linear(self.value_head, pooled).squeeze(-1)
    return Graph()


def fit(model_path, episodes, output_model, *, optimizer_path=None, output_optimizer=None, settings=None):
    """Unchanged maintained PPO bytecode, explicitly separate semantic bindings.

    The caller must freshly reconstruct the complete games before calling. As
    with the original fitter, numerical likelihood validation isn't a witness.
    """
    scope = {**training.__dict__, "ACTION_IDS": ACTION_IDS, "NORMALIZATION": (),
        "TRAINABLE": TRAINABLE, "load_model": load_model, "save_model": save_model,
        "ActorCriticPolicy": Policy, "validate_inputs": validate_inputs, "torch_graph": torch_graph}
    scope["_batch"] = types.FunctionType(training._batch.__code__, scope)
    fitted = types.FunctionType(training.train_ppo.__code__, scope,
                               argdefs=training.train_ppo.__defaults__)
    fitted.__kwdefaults__ = training.train_ppo.__kwdefaults__
    receipt = fitted(model_path, episodes, output_model, optimizer_path=optimizer_path,
                     output_optimizer=output_optimizer, settings=settings)
    return bound({"schema": "farm-semantic-command-ppo-fit-v1", "source_pins": sources(),
        "action_schema_sha256": ACTION_SCHEMA["content_sha256"], "numerical_fit": receipt,
        "canonical_model_changed": False, "strength_established": False})
