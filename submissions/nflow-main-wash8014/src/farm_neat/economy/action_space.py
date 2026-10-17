"""Full economic command and factored neural I/O contract; no economic oracle.

This NEW vocabulary exposes every engine crop, animal and market family. It is
not the old29 categorical, cannot admit its rows/checkpoints, and does not itself
execute games or train. The existing14x512/FF2048 attention trunk can be reused;
its old29-shape checks, candidate encoder and scalar categorical cannot.

Commands compose in order until STOP. Every economic parameter is explicit.
Locked sites, low cash, poor estimated profit and missing supplies are NOT syntax
masks: they are pending executor prerequisites/shortfalls. Monetary policy values
have explicit cent resolution; native prices/rewards are never rounded here.
Quantities are nonnegative signed32-bit integers, not the old4/12 population caps.
Numeric heads emit base16 digits, never a Cartesian list of economic plans.
"""
from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math

from farm_neat.game.constants import ANIMALS as ENGINE_ANIMALS
from farm_neat.game.constants import BUYABLE_PRODUCTS, CROPS as ENGINE_CROPS
from farm_neat.game.constants import PRODUCTS as ENGINE_PRODUCTS

SCHEMA = "farm-full-economic-commands-v1"
INPUT_SCHEMA = "farm-full-economic-entity-input-v1"
FACTOR_SCHEMA = "farm-full-economic-autoregressive-factors-v1"
PREFIX_SCHEMA = "farm-full-economic-observed-factor-prefix-v1"
CROPS = tuple(ENGINE_CROPS)
ANIMALS = tuple(ENGINE_ANIMALS)
PRODUCTS = tuple(ENGINE_PRODUCTS)
ITEMS = PRODUCTS + ANIMALS
REGIONS = ("NW", "NE", "SW", "SE", "ALL")
MAX_INTEGER = 2**31 - 1
LAST_STEP = 718
MAX_COMMANDS = 256
WIDTH = 80
OPS = ("STOP", "SET_CROP", "SET_ANIMAL", "CANCEL", "CLEAR", "REPLACE",
       "LAND", "HANDS", "SALE", "STOCK", "BUY", "CASH_RESERVE")
WATERING = ("OFF", "DAILY", "WHEN_DRY")
FERTILIZING = ("OFF", "DAILY", "WHEN_EXPIRED")
FEED = ("OFF", "DAILY", "WHEN_UNFED")
CARE = ("OFF", "DAILY")
MANURE = ("OFF", "WHEN_AVAILABLE")
STOCK_KINDS = ("SEED", "ANIMAL", "PRODUCT")
CANCEL_KINDS = ("PRODUCTION", "SALE", "STOCK", "BUY", "LAND", "HANDS", "CASH_RESERVE")
COMMAND_FIELDS = {
    "STOP": ("op",),
    "SET_CROP": ("op", "scope", "crop", "cycles", "until", "start", "harvest_threshold", "watering", "fertilizing", "priority"),
    "SET_ANIMAL": ("op", "scope", "animal", "until", "start", "feed", "care", "harvest_threshold", "manure", "priority"),
    "CANCEL": ("op", "target"), "CLEAR": ("op", "scope", "priority"),
    "REPLACE": ("op", "scope", "replacement"),
    "LAND": ("op", "target", "start", "until", "priority"),
    "HANDS": ("op", "target", "start", "until", "priority"),
    "SALE": ("op", "product", "quantity", "price_above", "reserve", "start", "until", "priority"),
    "STOCK": ("op", "kind", "item", "target", "price_below", "start", "until", "priority"),
    "BUY": ("op", "kind", "item", "quantity", "price_below", "start", "until", "priority"),
    "CASH_RESERVE": ("op", "amount"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _exact(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{label}: exact fields required: {sorted(fields)}")


def _integer(value, minimum=0, maximum=MAX_INTEGER, label="integer"):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label}: integer in [{minimum},{maximum}] required")
    return value


def _enum(value, choices, label):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label}: expected one of {choices}")
    return value


def money_cents(value):
    """Strict policy wire precision; reject rather than silently round a rule."""
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError("money: finite nonnegative dollars with cent precision required")
    try:
        cents = Decimal(str(value)) * 100
        if cents != cents.to_integral_value() or not 0 <= cents <= MAX_INTEGER:
            raise ValueError("money: cent precision/range")
        return int(cents)
    except InvalidOperation as exc:
        raise ValueError("money: invalid decimal") from exc


def validate_scope(scope):
    if not isinstance(scope, dict):
        raise ValueError("scope must be an object")
    kind = _enum(scope.get("kind"), ("TILE", "REGION"), "scope kind")
    if kind == "TILE":
        _exact(scope, ("kind", "x", "y"), "tile scope")
        _integer(scope["x"], 0, 9, "x")
        _integer(scope["y"], 0, 9, "y")
    else:
        _exact(scope, ("kind", "region", "count"), "region scope")
        region = _enum(scope["region"], REGIONS, "region")
        _integer(scope["count"], 1, 100 if region == "ALL" else 25, "target site count")
    return copy.deepcopy(scope)


def scope_sites(scope):
    """Candidate coordinates only; never allocation, legality or profitability."""
    scope = validate_scope(scope)
    if scope["kind"] == "TILE":
        return [(scope["x"], scope["y"])]
    region = scope["region"]
    return [(x, y) for y in range(10) for x in range(10)
            if region == "ALL" or ("N" if y < 5 else "S") + ("W" if x < 5 else "E") == region]


def _window(command):
    start = _integer(command["start"], 0, LAST_STEP, "start")
    until = command["until"]
    if until != "GAME_END":
        _integer(until, start, LAST_STEP, "until")
    _integer(command["priority"], 0, 255, "priority")


def validate_command(command):
    """Return an independent canonical command, without consulting world state.

    SET is idempotent desired state, not an incremental purchase. Crop cycles are
    completed plant lifecycles since that commitment began. Multiple harvests of
    an ongoing tomato/strawberry plant are ONE lifecycle; its natural exhaustion
    completes that cycle only if it yielded a successful harvest. Harvest events
    and harvested units are separately reported. Null cycles means until only.
    CANCEL retires intent; CLEAR explicitly requests clearing; REPLACE retires old
    intent and installs a new one, never teleports/disposes an obstructing animal.
    SALE quantity is a per-eligible-turn ceiling, STOCK target desired on-hand
    total, BUY quantity a one-shot total whose fulfilled count must be retained.
    Every market unit is still priced/checked in native queue order. Lower numeric
    priority dispatches first; stable ties preserve program installation order.
    """
    if not isinstance(command, dict):
        raise ValueError("command must be an object")
    op = _enum(command.get("op"), OPS, "command op")
    _exact(command, COMMAND_FIELDS[op], op)
    if "scope" in command:
        validate_scope(command["scope"])
    if "start" in command:
        _window(command)
    elif "priority" in command:
        _integer(command["priority"], 0, 255, "priority")
    if op == "SET_CROP":
        _enum(command["crop"], CROPS, "crop")
        if command["cycles"] is not None:
            _integer(command["cycles"], 1, MAX_INTEGER, "cycles")
        _enum(command["watering"], WATERING, "watering")
        _enum(command["fertilizing"], FERTILIZING, "fertilizing")
        _integer(command["harvest_threshold"], 1, MAX_INTEGER, "harvest threshold")
    elif op == "SET_ANIMAL":
        _enum(command["animal"], ANIMALS, "animal")
        _enum(command["feed"], FEED, "feed")
        _enum(command["care"], CARE, "care")
        _enum(command["manure"], MANURE, "manure")
        _integer(command["harvest_threshold"], 1, MAX_INTEGER, "harvest threshold")
    elif op == "REPLACE":
        replacement = command["replacement"]
        if not isinstance(replacement, dict) or replacement.get("op") not in ("SET_CROP", "SET_ANIMAL"):
            raise ValueError("replacement must be an explicit production command")
        validate_command(replacement)
        if replacement["scope"] != command["scope"]:
            raise ValueError("replacement scope must equal the retired scope")
    elif op == "CANCEL":
        target = command["target"]
        if not isinstance(target, dict):
            raise ValueError("cancel target must be a discriminated object")
        kind = _enum(target.get("kind"), CANCEL_KINDS, "cancel kind")
        fields = {"PRODUCTION": ("kind", "scope"), "SALE": ("kind", "product"),
                  "STOCK": ("kind", "stock_kind", "item"), "BUY": ("kind", "stock_kind", "item")}.get(kind, ("kind",))
        _exact(target, fields, "cancel target")
        if kind == "PRODUCTION":
            validate_scope(target["scope"])
        elif kind == "SALE":
            _enum(target["product"], PRODUCTS, "cancel product")
        elif kind in ("STOCK", "BUY"):
            stock_kind = _enum(target["stock_kind"], STOCK_KINDS, "cancel stock kind")
            _enum(target["item"], {"SEED": CROPS, "ANIMAL": ANIMALS, "PRODUCT": BUYABLE_PRODUCTS}[stock_kind], "cancel stock item")
    elif op in ("LAND", "HANDS"):
        _integer(command["target"], 0, 3 if op == "LAND" else MAX_INTEGER, "target")
    elif op == "SALE":
        _enum(command["product"], PRODUCTS, "sale product")
        _integer(command["quantity"], label="quantity")
        _integer(command["reserve"], label="reserve")
        money_cents(command["price_above"])
    elif op in ("STOCK", "BUY"):
        kind = _enum(command["kind"], STOCK_KINDS, "stock kind")
        _enum(command["item"], {"SEED": CROPS, "ANIMAL": ANIMALS, "PRODUCT": BUYABLE_PRODUCTS}[kind], "stock item")
        _integer(command["target" if op == "STOCK" else "quantity"], label="stock quantity")
        money_cents(command["price_below"])
    elif op == "CASH_RESERVE":
        money_cents(command["amount"])
    return copy.deepcopy(command)


def validate_program(commands):
    if not isinstance(commands, list) or not 1 <= len(commands) <= MAX_COMMANDS:
        raise ValueError(f"program requires1..{MAX_COMMANDS} ordered commands including STOP")
    out = [validate_command(c) for c in commands]
    if out[-1]["op"] != "STOP" or any(c["op"] == "STOP" for c in out[:-1]):
        raise ValueError("one terminal STOP required; no ignored suffix")
    return out


def _traverse(read, prefix="", production_only=False, fixed_scope=None):
    """The SAME conditional field tree drives encoding and neural decoding."""
    def categorical(name, choices):
        if fixed_scope is not None and name.startswith("scope."):
            choices = (fixed_scope[name.split(".", 1)[1]],)
        return read(prefix + name, tuple(choices))

    def integer(name, maximum=MAX_INTEGER):
        # Prefix masks only enforce the declared integer range, not good economics.
        value = 0
        for digit in range(7, -1, -1):
            remainder = 16**digit
            high = min(15, maximum // remainder - value * 16)
            value = value * 16 + categorical(f"{name}.hex{digit}", range(high + 1))
        return value

    def scope(name="scope."):
        kind = categorical(name + "kind", ("TILE", "REGION"))
        if kind == "TILE":
            return {"kind": kind, "x": categorical(name + "x", range(10)), "y": categorical(name + "y", range(10))}
        region = categorical(name + "region", REGIONS)
        return {"kind": kind, "region": region,
                "count": categorical(name + "count", range(1, (100 if region == "ALL" else 25) + 1))}

    def window():
        start = categorical("start", range(LAST_STEP + 1))
        return {"start": start, "until": categorical("until", ("GAME_END", *range(start, LAST_STEP + 1))),
                "priority": categorical("priority", range(256))}

    op = categorical("op", ("SET_CROP", "SET_ANIMAL") if production_only else OPS)
    c = {"op": op}
    if op in ("SET_CROP", "SET_ANIMAL", "CLEAR", "REPLACE"):
        c["scope"] = scope()
    if op == "REPLACE":
        c["replacement"] = _traverse(read, prefix + "replacement.", True, c["scope"])
    elif op == "SET_CROP":
        c["crop"] = categorical("crop", CROPS)
        has_cycles = categorical("cycles.present", (False, True))
        c["cycles"] = integer("cycles.minus_one", MAX_INTEGER - 1) + 1 if has_cycles else None
        c["harvest_threshold"] = integer("harvest_threshold.minus_one", MAX_INTEGER - 1) + 1
        c["watering"] = categorical("watering", WATERING)
        c["fertilizing"] = categorical("fertilizing", FERTILIZING)
        c.update(window())
    elif op == "SET_ANIMAL":
        c["animal"] = categorical("animal", ANIMALS)
        c["feed"] = categorical("feed", FEED)
        c["care"] = categorical("care", CARE)
        c["harvest_threshold"] = integer("harvest_threshold.minus_one", MAX_INTEGER - 1) + 1
        c["manure"] = categorical("manure", MANURE)
        c.update(window())
    elif op == "CLEAR":
        c["priority"] = categorical("priority", range(256))
    elif op == "CANCEL":
        kind = categorical("target.kind", CANCEL_KINDS)
        c["target"] = {"kind": kind}
        if kind == "PRODUCTION":
            c["target"]["scope"] = scope("target.scope.")
        elif kind == "SALE":
            c["target"]["product"] = categorical("target.product", PRODUCTS)
        elif kind in ("STOCK", "BUY"):
            stock_kind = categorical("target.stock_kind", STOCK_KINDS)
            c["target"]["stock_kind"] = stock_kind
            c["target"]["item"] = categorical("target.item", {"SEED": CROPS, "ANIMAL": ANIMALS, "PRODUCT": BUYABLE_PRODUCTS}[stock_kind])
    elif op in ("LAND", "HANDS"):
        c["target"] = categorical("target", range(4)) if op == "LAND" else integer("target")
        c.update(window())
    elif op == "SALE":
        c["product"] = categorical("product", PRODUCTS)
        c["quantity"] = integer("quantity")
        c["reserve"] = integer("reserve")
        c["price_above"] = integer("price_above.cents") / 100
        c.update(window())
    elif op in ("STOCK", "BUY"):
        c["kind"] = categorical("kind", STOCK_KINDS)
        c["item"] = categorical("item", {"SEED": CROPS, "ANIMAL": ANIMALS, "PRODUCT": BUYABLE_PRODUCTS}[c["kind"]])
        name = "target" if op == "STOCK" else "quantity"
        c[name] = integer(name)
        c["price_below"] = integer("price_below.cents") / 100
        c.update(window())
    elif op == "CASH_RESERVE":
        c["amount"] = integer("amount.cents") / 100
    return c


def _field(command, path):
    value = command
    bits = path.split(".")
    for name in bits:
        if name == "present":
            value = value is not None
        elif name == "minus_one":
            value -= 1
        elif name == "cents":
            value = money_cents(value)
        elif name.startswith("hex"):
            value = (value >> (4 * int(name[3:]))) & 15
        else:
            value = value[name]
    return value


def _same_choices(actual, expected):
    # JSON bool1/0 must not silently become a different numeric option embedding.
    return (isinstance(actual, list) and len(actual) == len(expected)
            and all(type(a) is type(b) and a == b for a, b in zip(actual, expected, strict=True)))


def encode_command(command):
    command = validate_command(command)
    factors = []
    def read(path, choices):
        value = _field(command, path)
        if value not in choices:
            raise ValueError(f"factor value outside domain: {path}")
        factors.append({"path": path, "choices": list(choices), "index": choices.index(value)})
        return value
    decoded = _traverse(read)
    if decoded != command:
        raise ValueError("command encoding round-trip mismatch")
    return {"schema": FACTOR_SCHEMA, "action_schema_sha256": ACTION_SCHEMA_SHA256, "factors": factors}


def decode_command(record):
    _exact(record, ("schema", "action_schema_sha256", "factors"), "factor record")
    if record["schema"] != FACTOR_SCHEMA or record["action_schema_sha256"] != ACTION_SCHEMA_SHA256:
        raise ValueError("new explicit action/factor schema required; old29 rows refused")
    if not isinstance(record["factors"], list):
        raise ValueError("factor list required")
    cursor = 0
    def read(path, choices):
        nonlocal cursor
        if cursor >= len(record["factors"]):
            raise ValueError("missing factor")
        factor = record["factors"][cursor]
        cursor += 1
        _exact(factor, ("path", "choices", "index"), "factor")
        if factor["path"] != path or not _same_choices(factor["choices"], choices):
            raise ValueError("factor path/domain drift")
        return choices[_integer(factor["index"], 0, len(choices) - 1, "factor index")]
    result = validate_command(_traverse(read))
    if cursor != len(record["factors"]):
        raise ValueError("unconsumed factor suffix")
    return result


def joint_log_probability(record, probabilities):
    """Diagnostic reference for the required differentiable PPO sum of logps.

    One *natural policy call* owns every chosen command/parameter/STOP factor.
    The training implementation must sum the corresponding differentiable logps;
    saving only the family/op probability would be a different, incorrect policy.
    This function is not rollout authentication or training admission.
    """
    decode_command(record)
    if not isinstance(probabilities, list) or len(probabilities) != len(record["factors"]):
        raise ValueError("one actual conditional probability vector per factor required")
    total = 0.0
    for factor, probs in zip(record["factors"], probabilities, strict=True):
        if (not isinstance(probs, (list, tuple)) or len(probs) != len(factor["choices"])
                or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in probs)
                or not math.isclose(math.fsum(probs), 1.0, abs_tol=1e-7, rel_tol=0)):
            raise ValueError("finite normalized conditional distribution required")
        chosen = probs[factor["index"]]
        if chosen <= 0:
            raise ValueError("selected factor must have positive behavior support")
        total += math.log(chosen)
    return total


def validate_prefix(record):
    """Validate a contiguous observed factor prefix, without inventing a suffix.

    This syntax check does not prove the observation/action came from a strong
    player or disclose the unobserved economic intent behind a primitive action.
    Acquisition/sale quantities are attempted orders, not assumed physical fills.
    """
    _exact(record, ("schema", "action_schema_sha256", "factors"), "factor prefix")
    if record["schema"] != PREFIX_SCHEMA or record["action_schema_sha256"] != ACTION_SCHEMA_SHA256:
        raise ValueError("new observed-prefix schema required")
    factors = record["factors"]
    if not isinstance(factors, list) or not factors:
        raise ValueError("nonempty observed factor prefix required")
    cursor = 0
    class PrefixEnded(Exception):
        pass
    def read(path, choices):
        nonlocal cursor
        if cursor == len(factors):
            raise PrefixEnded
        factor = factors[cursor]
        cursor += 1
        _exact(factor, ("path", "choices", "index"), "prefix factor")
        if factor["path"] != path or not _same_choices(factor["choices"], choices):
            raise ValueError("observed prefix cannot skip or invent conditional fields")
        return choices[_integer(factor["index"], 0, len(choices)-1, "prefix factor index")]
    try:
        validate_command(_traverse(read))
    except PrefixEnded:
        pass
    if cursor != len(factors):
        raise ValueError("prefix has ignored suffix")
    return copy.deepcopy(record)


ACTION_SCHEMA = {"schema": SCHEMA, "ops": OPS, "crops": CROPS, "animals": ANIMALS,
    "products": PRODUCTS, "buyable_products": BUYABLE_PRODUCTS, "regions": REGIONS,
    "max_integer": MAX_INTEGER, "last_step": LAST_STEP, "max_commands": MAX_COMMANDS,
    "command_fields": COMMAND_FIELDS, "cancel_kinds": CANCEL_KINDS,
    "money_resolution": "CENTS", "cycles": "COMPLETED_PLANT_LIFECYCLES_NOT_HARVEST_EVENTS",
    "scope": "ALL_ECONOMIC_FAMILIES_NO_PROFITABILITY_MASK",
    "rules": {"watering": WATERING, "fertilizing": FERTILIZING, "feed": FEED, "care": CARE, "manure": MANURE}}
ACTION_SCHEMA_SHA256 = digest(ACTION_SCHEMA)
INPUT_CONTRACT = {"schema": INPUT_SCHEMA, "width": WIDTH,
    "encoder": "src/farm_neat/policy/observation.py",
    "required_arrays": ("rows_FP32_BxNx80", "knownness_BOOL_BxNx80", "token_mask_BOOL_BxN"),
    "padding": "EXPLICIT_FALSE_MASK_DISTINCT_FROM_ACTION_LEGALITY",
    "trunk": {"blocks": 14, "dim": 512, "heads": 8, "ff": 2048},
    "heads": "CONDITIONAL_OP_ASSET_SCOPE_RULE_AND_NUMERIC_DIGITS_PLUS_GLOBAL_VALUE",
    "policy_logp": "SUM_ALL_SAMPLED_CONDITIONAL_FACTORS_INCLUDING_STOP_PER_NATURAL_CALL",
    "legacy_checkpoint_compatibility": False, "execution_or_training_implemented_here": False}
INPUT_SCHEMA_SHA256 = digest(INPUT_CONTRACT)
