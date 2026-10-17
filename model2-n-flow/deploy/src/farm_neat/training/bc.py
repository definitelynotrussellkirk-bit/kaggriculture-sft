"""Observed primitive-command prefixes, not reconstructed economic intentions.

The teacher's attempted order is imitated, NOT its unknown fill or latent plan.
Only BUY(kind,item), SALE(product,quantity), and SET_CROP(tile,crop)
prefixes are labeled. No cycles, thresholds, reserves, STOP or future inputs are
invented. These examples alone are not an engine witness or training admission.
BUY quantity is NOT a label: the persistent executor retains acquired credit,
so a one-unit primitive cannot reveal the rule's cumulative purchase target.
"""
from __future__ import annotations

import copy
import hashlib
import math
from collections import Counter
from pathlib import Path

from farm_neat.economy import action_space as actions
from farm_neat.policy import observation as features

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "farm-full-economic-observed-bc-step-v1"
EXAMPLE_SCHEMA = "farm-full-economic-observed-bc-example-v1"
PUBLIC_SCHEMA = "farm-full-economic-public-winner-prefixes-v1"
PROVENANCE_FIELDS = {"source_name", "source_uri", "source_sha256", "license", "episode_id", "player"}
digest = actions.digest


def sources():
    pins = features.sources()
    for name in ("src/farm_neat/training/bc.py", "src/farm_neat/game/kaggriculture_engine_reference.py"):
        pins[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return pins


_LOADED_SOURCES = sources()


def _seal(record):
    record["content_sha256"] = digest(record)
    return record


def _check_seal(record):
    if not isinstance(record, dict) or record.get("content_sha256") != digest(
            {k: v for k, v in record.items() if k != "content_sha256"}):
        raise ValueError("BC content hash mismatch")


def _provenance(value, player):
    if not isinstance(value, dict) or set(value) != PROVENANCE_FIELDS:
        raise ValueError("explicit source/license/episode/player provenance required")
    if type(value["player"]) is not int or value["player"] != player:
        raise ValueError("provenance actor mismatch")
    for name in PROVENANCE_FIELDS - {"player"}:
        if not isinstance(value[name], str) or not value[name] or len(value[name]) > 2048:
            raise ValueError("nonempty bounded provenance strings required")
    sha = value["source_sha256"]
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("exact teacher source SHA256 required")
    return copy.deepcopy(value)


def actor_observation(observation):
    """Whitelist actor-visible fields; never retain another seat's private data."""
    if not isinstance(observation, dict):
        raise ValueError("actor observation required")
    out = {k: copy.deepcopy(observation[k]) for k in ("player", "step", "day", "hour") if k in observation}
    out["farms"] = []
    for farm in observation.get("farms", []):
        clean = {k: copy.deepcopy(farm[k]) for k in
                 ("money", "hires_today", "unlocked_quadrants", "farmer", "hands") if k in farm}
        clean["tiles"] = [[{k: copy.deepcopy(t[k]) for k in
                           ("kind", "crop", "animal", *(n for n, _ in features.TILE_FIELDS)) if k in t}
                           if isinstance(t, dict) else copy.deepcopy(t) for t in row]
                          for row in farm.get("tiles", [])]
        out["farms"].append(clean)
    own = observation.get("private", {})
    out["private"] = {k: copy.deepcopy(own[k]) for k in ("shed", "seeds", "inventories") if k in own}
    market = observation.get("market", {})
    out["market"] = {k: copy.deepcopy(market[k]) for k in ("prices", "inventory") if k in market}
    if "params" in market:
        out["market"]["params"] = {item: {k: copy.deepcopy(p[k]) for k in
                ("base", "I0", "T", "below_target", "above_target", "below_func", "above_func") if k in p}
                for item, p in market["params"].items() if item in actions.PRODUCTS}
    out["town"] = {k: copy.deepcopy(observation.get("town", {})[k]) for k in
                   ("unlocked_shops",) if k in observation.get("town", {})}
    return out


def _observation_identity(observation):
    order = [list(inv) for inv in observation["private"]["inventories"]]
    return digest({"observation": observation, "inventory_key_order": order}), order


def _prefix(values):
    factors, consumed = [], set()
    class EndObserved(Exception):
        pass
    def read(path, choices):
        if path not in values:
            raise EndObserved
        chosen = values[path]
        index = next((i for i, choice in enumerate(choices)
                      if type(choice) is type(chosen) and choice == chosen), None)
        if index is None:
            raise ValueError("observed value outside conditional grammar")
        consumed.add(path)
        factors.append({"path": path, "choices": list(choices), "index": index})
        return chosen
    try:
        actions._traverse(read)
    except EndObserved:
        pass
    if consumed != set(values):
        raise ValueError("known fields are not a contiguous grammar prefix")
    return actions.validate_prefix({"schema": actions.PREFIX_SCHEMA,
        "action_schema_sha256": actions.ACTION_SCHEMA_SHA256, "factors": factors})


def _quantity(value):
    if type(value) is not int or not 1 <= value <= actions.MAX_INTEGER:
        raise ValueError("only positive typed integer attempted quantities are labeled")
    return {f"quantity.hex{digit}": (value >> (4*digit)) & 15 for digit in range(7, -1, -1)}


def _values(wire, phase, index, observation):
    if not isinstance(wire, list) or not wire or not isinstance(wire[0], str):
        raise ValueError("MALFORMED_PRIMITIVE")
    op = wire[0]
    if phase == "UNIT":
        farm = observation["farms"][observation["player"]]
        positions = [farm["farmer"], *farm["hands"]]
        if index >= len(positions):
            raise ValueError("NO_EXISTING_UNIT")
        if op != "PLANT":
            raise ValueError("UNOBSERVED_ECONOMIC_INTENT")
        if len(wire) < 2 or wire[1] not in actions.CROPS:
            raise ValueError("MALFORMED_PLANT")
        x, y = positions[index]
        return {"op": "SET_CROP", "scope.kind": "TILE", "scope.x": x, "scope.y": y, "crop": wire[1]}, 1
    if op not in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL"):
        raise ValueError("UNOBSERVED_ECONOMIC_INTENT")
    if len(wire) < 3:
        raise ValueError("MALFORMED_MARKET_ORDER")
    amount = _quantity(wire[2])
    if op == "SELL":
        if wire[1] not in actions.PRODUCTS:
            raise ValueError("UNKNOWN_PRODUCT")
        return {"op": "SALE", "product": wire[1], **amount}, wire[2]
    kind = {"BUY_SEED": "SEED", "BUY_ANIMAL": "ANIMAL", "BUY_PRODUCT": "PRODUCT"}[op]
    vocabulary = {"SEED": actions.CROPS, "ANIMAL": actions.ANIMALS, "PRODUCT": actions.BUYABLE_PRODUCTS}[kind]
    if wire[1] not in vocabulary:
        raise ValueError("UNKNOWN_BUY_ITEM")
    return {"op": "BUY", "kind": kind, "item": wire[1]}, wire[2]


def _diagnostic(wire, phase, index, before, after, plant_demand):
    result = {"filled_quantity": None, "known_noop": None,
              "scope": "ATTEMPTED_COMMAND_NOT_FILL", "reasons": []}
    if after is None:
        return result
    result["scope"] = "AFTER_ONLY_BOUNDARY_DIAGNOSTIC_NOT_TRAINING_INPUT"
    seat = before["player"]
    result["own_cash_delta"] = after["farms"][seat]["money"] - before["farms"][seat]["money"]
    if phase == "UNIT" and wire[0] == "PLANT":
        farm = before["farms"][seat]
        x, y = [farm["farmer"], *farm["hands"]][index]
        if plant_demand[wire[1]] > before["private"]["seeds"].get(wire[1], 0):
            result["known_noop"] = True
            result["reasons"].append("ATOMIC_PLANT_SEED_SHORTFALL")
        if farm["tiles"][y][x] == "LOCKED":
            result["known_noop"] = True
            result["reasons"].append("LOCKED_DURING_UNIT_PHASE")
        tile = after["farms"][seat]["tiles"][y][x]
        result["successor_site_matches_crop"] = isinstance(tile, dict) and tile.get("crop") == wire[1]
        # A matching tile could have been planted by another unit. Absence may
        # be same-night death. Neither proves this unit's successful planting.
    return result


def inspect_step(observation, action, *, provenance, successor=None, configuration=None):
    """One observation + ordered labels/omissions. No teacher plan is available.

    Returned examples are independent prefixes, not one reconstructed program;
    original unit/market indices and the whole submitted queue remain available.
    JSON writers may sort keys: inventory_key_order restores this visible detail.
    """
    obs = actor_observation(observation)
    cfg = {k: copy.deepcopy(v) for k, v in (configuration or {}).items() if k in features.CONFIG}
    if configuration and "marketParams" in configuration:
        cfg["marketParams"] = {item: {k: copy.deepcopy(v) for k, v in p.items()
            if k in features.MARKET_PARAMS[item]} for item, p in configuration["marketParams"].items()
            if item in actions.PRODUCTS}
    packed = features.pack(obs, None, cfg)
    if obs["step"] > actions.LAST_STEP:
        raise ValueError("no action labels after last actionable step")
    provenance = _provenance(provenance, obs["player"])
    if not isinstance(action, dict):
        raise ValueError("submitted action object required")
    wire = {k: copy.deepcopy(action.get(k, default)) for k, default in
            (("farmer", ["PASS"]), ("hands", []), ("market", []))}
    if not isinstance(wire["hands"], list) or not isinstance(wire["market"], list):
        raise ValueError("typed unit actions and ordered market queue required")
    after = None
    if successor is not None:
        after = actor_observation(successor)
        features.pack(after, None, cfg)
        if after["player"] != obs["player"] or after["step"] != obs["step"]+1:
            raise ValueError("successor must be next physical step for this actor")
    obs_sha, inventory_order = _observation_identity(obs)
    action_sha = digest(wire)
    examples, omitted = [], []
    units = [wire["farmer"], *wire["hands"]]
    demand = Counter(a[1] for a in units if isinstance(a, list) and len(a) >= 2
                     and a[0] == "PLANT" and isinstance(a[1], str))
    limit = cfg.get("maxMarketOrdersPerTurn", features.CONFIG["maxMarketOrdersPerTurn"])
    for phase, queue in (("UNIT", units), ("MARKET", wire["market"])):
        for index, primitive in enumerate(queue):
            locator = {"phase": phase, "index": index}
            try:
                if phase == "MARKET" and index >= limit:
                    raise ValueError("OUTSIDE_ENGINE_QUEUE_PREFIX")
                values, quantity = _values(primitive, phase, index, obs)
                prefix = _prefix(values)
            except ValueError as exc:
                omitted.append({"locator": locator, "reason": str(exc)})
                continue
            examples.append(_seal({"schema": EXAMPLE_SCHEMA, "source_pins": dict(_LOADED_SOURCES),
                "source_sha256": digest(_LOADED_SOURCES), "provenance": provenance,
                "step": obs["step"], "observation_sha256": obs_sha,
                "input_sha256": packed["content_sha256"], "action_sha256": action_sha,
                "locator": locator, "attempted_primitive": copy.deepcopy(primitive),
                "attempted_quantity": quantity, "prefix": prefix,
                "diagnostic": _diagnostic(primitive, phase, index, obs, after, demand),
                "training_admission": False, "latent_intent_reconstruction": False}))
    return _seal({"schema": SCHEMA, "source_pins": dict(_LOADED_SOURCES),
        "source_sha256": digest(_LOADED_SOURCES), "provenance": provenance,
        "observation": obs, "inventory_key_order": inventory_order, "configuration": cfg,
        "observation_sha256": obs_sha, "input_sha256": packed["content_sha256"],
        "action": wire, "action_sha256": action_sha,
        "successor_observation_sha256": None if after is None else _observation_identity(after)[0],
        "examples": examples, "omitted": omitted, "training_admission": False,
        "label_scope": "OBSERVED_PRIMITIVE_PREFIXES_ONLY_NOT_FULL_PROGRAM_OR_SUCCESS"})


def extract(observation, action, *, provenance, successor=None, configuration=None):
    """Convenience labels-only API; the caller retains the same-step input once."""
    return inspect_step(observation, action, provenance=provenance,
                        successor=successor, configuration=configuration)["examples"]


def from_replay_step(replay, t, player, *, provenance, configuration=None):
    """Kaggle pairing: steps[t].observation -> steps[t+1].action, same seat.

    No opponent private log, recorded world injection, inference or engine run.
    The surrounding collector must authenticate its replay/source file separately.
    """
    if type(t) is not int or t < 0 or type(player) is not int or player not in (0, 1):
        raise ValueError("explicit step index and actor seat required")
    steps = replay.get("steps")
    if not isinstance(steps, list) or t+1 >= len(steps):
        raise ValueError("both replay boundaries required")
    before, after = steps[t][player], steps[t+1][player]
    obs, successor = before["observation"], after["observation"]
    if obs.get("player") != player or obs.get("step") != t:
        raise ValueError("replay actor/index clock mismatch; no silent rebasing")
    return inspect_step(obs, after["action"], provenance=provenance, successor=successor,
                        configuration=configuration if configuration is not None else replay.get("configuration"))


def public_observation(replay, t, player):
    """Normalize Kaggle's shared clock, never borrow the other seat's private data.

    Boundary t precedes the action stored at t+1. Only seat0 carries shared step
    in the public daily JSON; require it and independently check each local clock.
    The returned whitelist contains no replay seed, outcome, future or agent name.
    """
    if type(t) is not int or not 0 <= t <= 719 or type(player) is not int or player not in (0, 1):
        raise ValueError("explicit physical boundary and seat required")
    try:
        boundary = replay["steps"][t]
        shared = boundary[0]["observation"]["step"]
        raw = boundary[player]["observation"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("public replay boundary/clock missing") from error
    if (len(boundary) != 2 or type(shared) is not int or shared != t
            or type(raw.get("player")) is not int or raw["player"] != player
            or ("step" in raw and (type(raw["step"]) is not int or raw["step"] != t))
            or type(raw.get("day")) is not int or type(raw.get("hour")) is not int
            or (raw["day"], raw["hour"]) != divmod(t, 24)):
        raise ValueError("public replay actor/shared clock mismatch")
    out = actor_observation(raw)
    out["step"] = t
    return out


def public_replay_header(replay):
    """Require a complete normal game; cash chooses winner, never replay position.

    This header alone is NOT a witness. verify_public_replay additionally checks
    all actor observations against recorded joint actions in the exact engine.
    """
    if (replay.get("name") != "kaggriculture" or replay.get("version") != "0.1.0"
            or replay.get("module_version") != "1.32.7" or replay.get("schema_version") != 1
            or not isinstance(replay.get("steps"), list) or len(replay["steps"]) != 720
            or replay.get("statuses") != ["DONE", "DONE"]):
        raise ValueError("complete supported public Kaggriculture replay required")
    cfg, info = replay.get("configuration", {}), replay.get("info", {})
    if any(cfg.get(k) != v for k, v in features.CONFIG.items()):
        raise ValueError("public replay mechanics configuration mismatch")
    if cfg.get("startingMoney") != 3000 or cfg.get("marketParams") != {}:
        raise ValueError("default starting cash and market mechanics required")
    if type(info.get("EpisodeId")) is not int or type(info.get("seed")) is not int:
        raise ValueError("explicit public episode identity and reconstruction seed required")
    cash = replay.get("rewards")
    if (not isinstance(cash, list) or len(cash) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in cash)):
        raise ValueError("finite terminal cash required")
    for t, boundary in enumerate(replay["steps"]):
        if not isinstance(boundary, list) or len(boundary) != 2:
            raise ValueError("exactly two actors per public boundary required")
        for seat, row in enumerate(boundary):
            if row.get("status") != ("DONE" if t == 719 else "ACTIVE"):
                raise ValueError("censored/error/inactive public game refused")
            public_observation(replay, t, seat)
            if t and not isinstance(row.get("action"), dict):
                raise ValueError("submitted joint actions required")
    for seat in (0, 1):
        final = public_observation(replay, 719, seat)
        if ([f["money"] for f in final["farms"]] != cash
                or replay["steps"][-1][seat].get("reward") != cash[seat]):
            raise ValueError("terminal public cash/reward disagreement")
    return {"schema": PUBLIC_SCHEMA, "episode_id": info["EpisodeId"], "seed": info["seed"],
            "physical_turns": 719, "terminal_cash": list(cash),
            "winner_seat": None if cash[0] == cash[1] else int(cash[1] > cash[0]),
            "split": "OPENED_PUBLIC_TRAINING_CANDIDATE_NOT_HELDOUT",
            "latent_plan_labels": False, "terminal_rl_admission": False}


def verify_public_replay(replay):
    """Replay the recorded JOINT actions, not a responsive counterfactual.

    Fresh engine state must reproduce both actor-only views at all720 boundaries.
    JSON int/float representations may differ; numeric values must compare exactly,
    with no numerical tolerance. No neural sampling, fitting or privileged inputs.
    """
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment, assert_bundled_engine_identity
    header = public_replay_header(replay)
    env = ExactKaggricultureEnvironment(seed=header["seed"], configuration=replay["configuration"])
    for t in range(720):
        for seat in (0, 1):
            expected = public_observation(replay, t, seat)
            actual = actor_observation(env.agent_observation(seat))
            if actual != expected:
                raise ValueError(f"public exact-engine observation mismatch at boundary {t}, seat {seat}")
        if t < 719:
            env.step([replay["steps"][t+1][seat]["action"] for seat in (0, 1)])
    if not env.done or env.step_index != 719:
        raise ValueError("complete exact public reconstruction required")
    return {**header, "engine": assert_bundled_engine_identity(),
            "checked_actor_boundaries": 1440, "recorded_joint_actions_checked": 719,
            "witness_kind": "RECORDED_JOINT_ACTION_RECONSTRUCTION_NOT_RESPONSIVE_COUNTERFACTUAL"}


def public_winner_reports(replay, *, provenance):
    """Yield existing SFT step reports; no learned full-plan suffix is invented.

    Caller authenticates raw file and runs verify_public_replay before admitting
    the export. Terminal cash selects the episode perspective, not individual
    actions or actor inputs. Report diagnostics can exclude provably ineffective
    primitives; winning alone does not make every command a useful target.
    """
    header = public_replay_header(replay)
    seat = header["winner_seat"]
    if seat is None:
        return
    _provenance(provenance, seat)
    if provenance["episode_id"] != str(header["episode_id"]):
        raise ValueError("public teacher episode provenance mismatch")
    for t in range(719):
        yield inspect_step(public_observation(replay, t, seat), replay["steps"][t+1][seat]["action"],
            provenance=provenance, successor=public_observation(replay, t+1, seat),
            configuration=replay["configuration"])


def validate_example(example):
    _check_seal(example)
    if (example.get("schema") != EXAMPLE_SCHEMA or example.get("source_pins") != _LOADED_SOURCES
            or example.get("source_sha256") != digest(_LOADED_SOURCES)
            or example.get("training_admission") is not False
            or example.get("latent_intent_reconstruction") is not False):
        raise ValueError("new source-bound partial BC example required")
    _provenance(example["provenance"], example["provenance"]["player"])
    prefix = actions.validate_prefix(example["prefix"])
    values = {f["path"]: f["choices"][f["index"]] for f in prefix["factors"]}
    wire, locator = example["attempted_primitive"], example["locator"]
    if (not isinstance(locator, dict) or set(locator) != {"phase", "index"}
            or type(locator["index"]) is not int or locator["index"] < 0
            or locator["phase"] not in ("UNIT", "MARKET")):
        raise ValueError("typed primitive locator required")
    if locator["phase"] == "MARKET":
        expected, quantity = _values(wire, "MARKET", locator["index"], None)
    else:
        if (not isinstance(wire, list) or len(wire) < 2 or wire[0] != "PLANT"
                or wire[1] not in actions.CROPS):
            raise ValueError("only observed PLANT receives a unit prefix")
        expected = {"op":"SET_CROP", "scope.kind":"TILE", "scope.x":values.get("scope.x"),
                    "scope.y":values.get("scope.y"), "crop":wire[1]}
        quantity = 1
    if values != expected or prefix != _prefix(expected) or example["attempted_quantity"] != quantity:
        raise ValueError("observed prefix includes unknown intent or mismatches primitive")
    return example


def pack_step(report):
    """Recreate only BEFORE inputs after JSON key sorting, checking every join."""
    _check_seal(report)
    if report.get("schema") != SCHEMA or report.get("source_pins") != _LOADED_SOURCES:
        raise ValueError("new source-bound BC step required")
    obs = copy.deepcopy(report["observation"])
    inventories, orders = obs["private"]["inventories"], report["inventory_key_order"]
    if len(inventories) != len(orders):
        raise ValueError("carrier inventory order length mismatch")
    for i, keys in enumerate(orders):
        if len(keys) != len(set(keys)) or set(keys) != set(inventories[i]):
            raise ValueError("carrier inventory key order mismatch")
        inventories[i] = {k: inventories[i][k] for k in keys}
    rebuilt = inspect_step(obs, report["action"], provenance=report["provenance"],
                           configuration=report["configuration"])
    for key in ("observation_sha256", "input_sha256", "action_sha256", "omitted"):
        if report[key] != rebuilt[key]:
            raise ValueError("same-step BC input/action mismatch")
    if len(report["examples"]) != len(rebuilt["examples"]):
        raise ValueError("omitted or extra BC examples")
    for actual, expected in zip(report["examples"], rebuilt["examples"]):
        validate_example(actual)
        if {k: v for k, v in actual.items() if k not in ("diagnostic", "content_sha256")} != {
                k: v for k, v in expected.items() if k not in ("diagnostic", "content_sha256")}:
            raise ValueError("primitive/prefix/order mismatch")
    return features.pack(obs, None, report["configuration"])


def bc_loss(graph, context, examples):
    """Differentiable observed-factor mean NLL; no critic/unknown-suffix loss.

    Caller encodes pack_step(report) and repeats its context for its examples in
    recorded order. This function has no optimizer, dataset admission or I/O.
    """
    if not isinstance(examples, (list, tuple)) or not examples or len(context) != len(examples):
        raise ValueError("one encoded context per nonempty observed example required")
    prefixes = [validate_example(e)["prefix"] for e in examples]
    factor_count = sum(len(p["factors"]) for p in prefixes)
    logp, entropy = graph.replay_prefix(context, prefixes)
    return {"loss": -logp.sum()/factor_count, "logp": logp, "entropy": entropy,
            "observed_factors": factor_count, "examples": len(examples),
            "unobserved_factor_loss": 0, "critic_loss": 0}
