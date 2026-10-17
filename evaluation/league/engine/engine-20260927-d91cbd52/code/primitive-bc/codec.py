"""Per-step joint-action codec for primitive behavior cloning.

One label set per actor step: a unit token (+ quantity class, + digits when the
class is QTY_EXACT) for the farmer and every active hand in observation order,
then an ordered market queue of at most MAX_ORDERS order tokens (+ quantity
classes / digits) terminated by STOP.

Vocabularies are generated from the engine's accepted action grammar
(src/farm_neat/game/kaggriculture_engine_reference.py: _apply_unit_action,
_parse_order and the lockstep quote in _process_market), not from replays.
The tables below are self-contained so the codec ships without the engine;
test_codec.py asserts they equal the grammar extracted from the engine source.

canonical(action, obs) is a semantics-preserving normalization of a submitted
wire action under the fixed competition configuration (shedCapacity=100,
maxMarketOrdersPerTurn=10):
  * malformed / unknown / non-positive unit actions -> ["PASS"] (engine no-op);
  * extra args dropped from argument-free ops; PLANT keeps only its crop;
  * PICKUP/PLACE count defaults to 1, int() applied, clamped to SHED_CAPACITY
    (every PICKUP/PLACE moves at most the shed capacity, so larger counts are
    engine-identical);
  * hands padded with PASS / truncated to the observed hand count (a hand
    action without a unit is a no-op, except that a PLANT in it still counts
    toward the engine's atomic per-crop seed demand: the blocking those extra
    entries add is applied to the real units' PLANTs using the visible seeds);
  * market truncated to MAX_ORDERS, orders the engine drops (unparseable,
    unknown item, qty <= 0) become ["NOOP"] (they still occupy their lockstep
    index against the opponent); trailing NOOPs are removed (no effect);
  * SELL / BUY_PRODUCT / BUY_ANIMAL quantities clamped to SHED_CAPACITY (each
    unit needs shed stock / shed room, so larger counts are engine-identical);
    BUY_SEED clamped to MAX_QTY = 99,999: _process_market's per-index lockstep
    loop breaks at its 100,000th iteration, so no order slot can commit more
    than 99,999 units and every larger count is engine-identical.

Quantities (v3): QTY_CLASSES = 76 classes per quantity head, the 75 QTY_BINS
plus QTY_EXACT (= 75). encode prefers a reproducing discrete bin, then a
reproducing fraction bin, else QTY_EXACT with the canonical shortest decimal
spelling of the count in a digit sub-sequence (DIGIT_VOCAB '0'..'9', 'END';
no leading zeros, exactly one spelling per integer, END-terminated, -1 padded
to MAX_DIGITS + 1 positions). Every canonical action with at most MAX_UNITS
units is encoded exactly; decode of any class/digit sequence yields an
engine-accepted count in 1..cap.
"""
from __future__ import annotations

import ast
import math
import operator
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE_SOURCE = HERE.parents[3] / "src/farm_neat/game/kaggriculture_engine_reference.py"

SHED_CAPACITY = 100
MAX_ORDERS = 10
N_ORDER_SLOTS = MAX_ORDERS + 1          # order queue positions incl. the STOP slot
QTY_BIN_COUNT = 75
QTY_EXACT = QTY_BIN_COUNT               # escape class: the count is spelled in digits
QTY_CLASSES = QTY_BIN_COUNT + 1         # 76 classes per quantity head
# Engine bound (engine_order_unit_bound() reads it from the engine source): the market
# loop guard `idx_esc >= 100_000` stops an order slot after 99,999 committed units.
MAX_QTY = 99_999
DIGIT_VOCAB = tuple(str(d) for d in range(10)) + ("END",)
DIGIT_END = DIGIT_VOCAB.index("END")    # 10
MAX_DIGITS = len(str(MAX_QTY))          # 5
DIGIT_SLOTS = MAX_DIGITS + 1            # digits + END
SELL_ALL = SHED_CAPACITY                # canonical SELL count when the audited fill empties the item
BOARD = 10

# Frozen copy of engine_grammar() (test_codec asserts equality with the engine
# source): accepted unit ops, order ops, lockstep item tables and price tables.
GRAMMAR = {
    "unit_ops": ["NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "PICKUP", "PLACE", "PLANT",
                 "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED",
                 "COLLECT_FERTILIZER", "CARE"],
    "order_ops": ["HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"],
    "lockstep_items": {
        "SELL": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"],
        "BUY_PRODUCT": ["WHEAT", "FERTILIZER"],
        "BUY_SEED": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"],
        "BUY_ANIMAL": ["GOOSE", "COW", "SHEEP"]},
    "crops": {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80},
    "animals": {"GOOSE": 300, "COW": 400, "SHEEP": 500},
    "products": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"],
    "moves": ["NORTH", "SOUTH", "EAST", "WEST"],
}
ITEM_UNIT_OPS = ("PICKUP", "PLACE")


def vocab_from_grammar(g):
    """UNIT_VOCAB / ORDER_VOCAB generated from an engine grammar table.

    Units: PASS, the other argument-free ops in engine order, PLANT:<crop>,
    PICKUP:<shed item>, PLACE:<shed item> (shed items = products + animals,
    the only items a shed or a unit inventory can hold). Orders: STOP, NOOP,
    the atomic orders, then <op>:<item> for every lockstep-quoted item."""
    argful = ("PLANT",) + ITEM_UNIT_OPS
    plain = ["PASS"] + [op for op in g["unit_ops"] if op not in argful and op != "PASS"]
    shed = list(g["products"]) + list(g["animals"])
    unit = (plain + [f"PLANT:{c}" for c in g["crops"]]
            + [f"{op}:{i}" for op in ITEM_UNIT_OPS for i in shed])
    atomic = [op for op in g["order_ops"] if op not in g["lockstep_items"]]
    order = (["STOP", "NOOP"] + atomic + [f"{op}:{i}" for op in g["order_ops"]
                                          if op in g["lockstep_items"] for i in g["lockstep_items"][op]])
    return unit, order


CROPS = dict(GRAMMAR["crops"])        # seed price
ANIMALS = dict(GRAMMAR["animals"])    # animal price
PRODUCTS = tuple(GRAMMAR["products"])
MOVES = tuple(GRAMMAR["moves"])
SHED_ITEMS = PRODUCTS + tuple(ANIMALS)
MARKET_QTY_OPS = tuple(op for op in GRAMMAR["order_ops"] if op in GRAMMAR["lockstep_items"])
MARKET_ITEMS = {op: tuple(GRAMMAR["lockstep_items"][op]) for op in MARKET_QTY_OPS}
UNIT_VOCAB, ORDER_VOCAB = vocab_from_grammar(GRAMMAR)
PLAIN_UNIT_OPS = tuple(t for t in UNIT_VOCAB if ":" not in t)
UNIT_INDEX = {t: i for i, t in enumerate(UNIT_VOCAB)}
ORDER_INDEX = {t: i for i, t in enumerate(ORDER_VOCAB)}
STOP = ORDER_INDEX["STOP"]
NOOP = ORDER_INDEX["NOOP"]
PASS = UNIT_INDEX["PASS"]
UNIT_HAS_QTY = [t.split(":")[0] in ITEM_UNIT_OPS for t in UNIT_VOCAB]
ORDER_HAS_QTY = [t.split(":")[0] in MARKET_QTY_OPS for t in ORDER_VOCAB]
QTY_CAP = {"PICKUP": SHED_CAPACITY, "PLACE": SHED_CAPACITY, "SELL": SHED_CAPACITY,
           "BUY_PRODUCT": SHED_CAPACITY, "BUY_ANIMAL": SHED_CAPACITY, "BUY_SEED": MAX_QTY}

# Farmer + hands. Largest active-hand count over all 13,061 episodes of the 20
# daily archives is 16; the engine's practical bound at the largest own cash
# observed (195,808) is 24 hands (n hires in one day cost fib(n+1)-1). Hands
# beyond MAX_UNITS-1 decode to PASS (a hand without an entry is an engine
# no-op) and encode as non-representable (HANDS_OVER_MAX_UNITS).
MAX_UNITS = 25

# 75 shared quantity bins: ("n", v, 1) = the count v; ("p", num, den) = the
# fraction num/den of the context reference, decoded max(1, round_half_up).
# Chosen by `python codec.py measure`: greedy exact-coverage over discrete
# counts and fractions on 320 episodes (16 per day x 20 days); VOCAB.json
# records the search table and the held-out coverage.
QTY_BINS = (tuple(("n", v, 1) for v in range(1, 65))
            + tuple(("n", v, 1) for v in (66, 68, 72, 78, 80, 85, 89, 90, 99, 100))
            + (("p", 1, 1),))
assert len(QTY_BINS) == QTY_BIN_COUNT


class Unrepresentable(ValueError):
    """The wire action would crash the engine or its effect depends on state."""


def _hashable(value):
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _cap(op, n):
    cap = QTY_CAP[op]
    return n if cap is None else min(n, cap)


# ---------------------------------------------------------------- canonical
def canonical_unit(a):
    if not isinstance(a, list) or not a:
        return ["PASS"]
    op = a[0]
    if not _hashable(op):
        raise Unrepresentable("UNHASHABLE_UNIT_OP")
    if op in PLAIN_UNIT_OPS:
        return [op]
    if op == "PLANT":
        if len(a) < 2:
            return ["PASS"]
        if not _hashable(a[1]):
            raise Unrepresentable("UNHASHABLE_PLANT_CROP")
        return ["PLANT", a[1]] if a[1] in CROPS else ["PASS"]
    if op in ITEM_UNIT_OPS:
        if len(a) < 2:
            return ["PASS"]
        item = a[1]
        if not _hashable(item):
            raise Unrepresentable("UNHASHABLE_ITEM")
        if item not in SHED_ITEMS:
            return ["PASS"]
        if len(a) >= 3:
            try:
                n = int(a[2])
            except (TypeError, ValueError, OverflowError):
                raise Unrepresentable("UNPARSEABLE_UNIT_COUNT")
        else:
            n = 1
        if n <= 0:
            if op == "PLACE" and item in ANIMALS:
                raise Unrepresentable("PLACE_ANIMAL_NONPOSITIVE_COUNT")
            return ["PASS"]
        return [op, item, _cap(op, n)]
    return ["PASS"]


def canonical_order(o):
    if not isinstance(o, list) or not o:
        return ["NOOP"]
    op = o[0]
    if op == "HIRE" or op == "BUY_LAND":
        return [op]
    if isinstance(op, str) and op in MARKET_QTY_OPS:
        if len(o) < 3:
            return ["NOOP"]
        try:
            n = int(o[2])
        except (TypeError, ValueError):
            return ["NOOP"]
        except OverflowError:
            raise Unrepresentable("MARKET_QTY_OVERFLOW")
        if n <= 0:
            return ["NOOP"]
        item = o[1]
        if not _hashable(item):
            raise Unrepresentable("UNHASHABLE_MARKET_ITEM")
        if item not in MARKET_ITEMS[op]:
            return ["NOOP"]
        return [op, item, _cap(op, n)]
    return ["NOOP"]


def _blocked(seeds, units):
    """Engine atomic PLANT rule: crops whose PLANT demand exceeds the seed stock."""
    demand = {}
    for a in units:
        if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT":
            if not _hashable(a[1]):
                raise Unrepresentable("UNHASHABLE_PLANT_CROP")
            demand[a[1]] = demand.get(a[1], 0) + 1
    return {crop for crop, k in demand.items() if k > seeds.get(crop, 0)}


def hand_count(actor_obs):
    return len(actor_obs["farms"][actor_obs["player"]]["hands"])


def canonical(action, actor_obs=None, reasons=None):
    """Engine-equivalent normal form. With actor_obs, hands match the observed count."""
    if not isinstance(action, dict):
        action = {}
    farmer = canonical_unit(action.get("farmer", ["PASS"]))
    hands = action.get("hands", [])
    hands = hands if isinstance(hands, list) else []
    raw_farmer = action.get("farmer", ["PASS"])
    if actor_obs is not None:
        n = hand_count(actor_obs)
        newly_blocked = set()
        if len(hands) > n:
            # A hand entry without a unit is a no-op, but its PLANT still counts
            # toward the engine's atomic per-crop seed demand. Apply exactly the
            # blocking that the extra entries add (seeds are actor-visible).
            seeds = actor_obs["private"]["seeds"]
            newly_blocked = _blocked(seeds, [raw_farmer, *hands]) - _blocked(seeds, [raw_farmer, *hands[:n]])
        hands = hands[:n] + [["PASS"]] * max(0, n - len(hands))
        if newly_blocked:
            farmer = ["PASS"] if farmer[0] == "PLANT" and farmer[1] in newly_blocked else farmer
            hands = [["PASS"] if isinstance(h, list) and len(h) >= 2 and h[0] == "PLANT"
                     and h[1] in newly_blocked else h for h in hands]
    hands = [canonical_unit(h) for h in hands]
    market = action.get("market", [])
    market = list(market) if isinstance(market, list) else []
    orders = [canonical_order(o) for o in market[:MAX_ORDERS]]
    while orders and orders[-1] == ["NOOP"]:
        orders.pop()
    return {"farmer": farmer, "hands": hands, "market": orders}


# ---------------------------------------------------------------- references
def unit_positions(actor_obs):
    farm = actor_obs["farms"][actor_obs["player"]]
    return [list(farm["farmer"]), *[list(h) for h in farm["hands"]]]


def reference(token, actor_obs, unit_index=0):
    """Context amount for percentage bins, from the actor-visible step-t view only."""
    op, item = token.split(":")
    seat = actor_obs["player"]
    private = actor_obs["private"]
    if op in ("SELL", "PICKUP"):
        return int(private["shed"].get(item, 0))
    if op == "PLACE":
        inventories = private.get("inventories", [])
        return int(inventories[unit_index].get(item, 0)) if unit_index < len(inventories) else 0
    cash = actor_obs["farms"][seat]["money"]
    if op == "BUY_SEED":
        price = CROPS[item]
    elif op == "BUY_ANIMAL":
        price = ANIMALS[item]
    else:
        price = max(1, int(actor_obs["market"]["prices"][item]))
    return max(0, int(math.floor(cash / price)))


def bin_value(b, ref):
    kind, num, den = QTY_BINS[b]
    if kind == "n":
        return num
    return max(1, (2 * num * ref + den) // (2 * den))


def _index(value):
    """int from Python/numpy/torch integer scalars; floats and bools are rejected."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a label index")
    try:
        return operator.index(value)
    except TypeError:
        if hasattr(value, "item") and not isinstance(value.item(), float):
            return operator.index(value.item())
        raise ValueError("integer label index required")


def spell_digits(q):
    """Canonical digit labels of a non-negative integer: shortest decimal (no leading
    zeros, '0' for 0), then END, -1 padded to DIGIT_SLOTS. One spelling per integer."""
    q = _index(q)
    if not 0 <= q <= MAX_QTY:
        raise ValueError("digit spelling needs 0 <= q <= MAX_QTY")
    out = [int(c) for c in str(q)] + [DIGIT_END]
    return out + [-1] * (DIGIT_SLOTS - len(out))


def parse_digits(seq):
    """Integer read from a digit label/prediction sequence: digits up to the first END
    (or -1, or MAX_DIGITS digits); empty -> 0. Total on any sequence of class ids."""
    value = n = 0
    for d in (seq if seq is not None else ())[:DIGIT_SLOTS]:
        d = _index(d)
        if d < 0 or d == DIGIT_END or n == MAX_DIGITS:
            break
        if d > DIGIT_END:
            raise ValueError("digit class out of range")
        value, n = value * 10 + d, n + 1
    return value


def decode_qty(op, b, ref, digits=None):
    """Count for quantity class b: a bin decoded against ref, or (QTY_EXACT) the digit
    sequence; always an engine-accepted count in 1..cap."""
    b = _index(b)
    if not 0 <= b < QTY_CLASSES:
        raise ValueError("quantity class out of range")
    if b == QTY_EXACT:
        return _cap(op, max(1, parse_digits(digits)))
    return _cap(op, bin_value(b, ref))


def encode_qty(op, q, ref, escape=True):
    """(class, exact). Prefer a discrete bin; else the percentage whose unrounded
    value is closest; else (escape) QTY_EXACT, spelled by spell_digits(q); without
    escape (bin-coverage measurement), the closest decoded bin (discrete first on ties)."""
    best_pct, best_err = None, None
    for b, (kind, num, den) in enumerate(QTY_BINS):
        if kind == "n" and _cap(op, num) == q and num == q:
            return b, True
    for b, (kind, num, den) in enumerate(QTY_BINS):
        if kind == "p" and decode_qty(op, b, ref) == q:
            err = abs(num * ref - q * den) / den
            if best_err is None or err < best_err:
                best_pct, best_err = b, err
    if best_pct is not None:
        return best_pct, True
    # Discrete bins above the cap decode to the cap; an exact cap hit is exact.
    for b, (kind, num, den) in enumerate(QTY_BINS):
        if kind == "n" and _cap(op, num) == q:
            return b, True
    if escape and 1 <= q <= (QTY_CAP[op] or MAX_QTY):
        return QTY_EXACT, True
    ranked = sorted(range(QTY_BIN_COUNT), key=lambda b: (abs(decode_qty(op, b, ref) - q),
                                                         QTY_BINS[b][0] != "n", b))
    return ranked[0], False


# ---------------------------------------------------------------- tokens
def unit_token(canon_unit):
    op = canon_unit[0]
    if op in PLAIN_UNIT_OPS:
        return op
    return f"{op}:{canon_unit[1]}"


def order_token(canon_order):
    op = canon_order[0]
    if op in ("NOOP", "HIRE", "BUY_LAND"):
        return op
    return f"{op}:{canon_order[1]}"


NO_DIGITS = (-1,) * DIGIT_SLOTS


def encode_step(action, actor_obs, reasons=None, details=None, *, canonical_action=None):
    """Labels for one step. exact is False when decode(encode(a)) != canonical(a).

    unit_tok/unit_qty: [MAX_UNITS] (-1 = no unit); order_tok/order_qty: the queue then
    STOP (length 1..N_ORDER_SLOTS); unit_digits [MAX_UNITS][DIGIT_SLOTS] and order_digits
    [N_ORDER_SLOTS][DIGIT_SLOTS]: spell_digits(q) where the quantity class is QTY_EXACT,
    else -1 rows. canonical_action: an already canonical action (e.g. data.canonical_labels
    output) encoded as given instead of canonical(action, actor_obs).
    reasons (list) receives a code per non-representable part; details (list)
    receives {token, q, ref, decoded} for every inexact quantity."""
    local = [] if reasons is None else reasons
    canon = canonical(action, actor_obs, local) if canonical_action is None else canonical_action
    units = [canon["farmer"], *canon["hands"]]
    if len(units) > MAX_UNITS:
        local.append("HANDS_OVER_MAX_UNITS")
    unit_tok, unit_qty = [-1] * MAX_UNITS, [-1] * MAX_UNITS
    unit_digits = [list(NO_DIGITS) for _ in range(MAX_UNITS)]
    for i, u in enumerate(units[:MAX_UNITS]):
        tok = unit_token(u)
        unit_tok[i] = UNIT_INDEX[tok]
        if UNIT_HAS_QTY[unit_tok[i]]:
            ref = reference(tok, actor_obs, i)
            b, ok = encode_qty(u[0], u[2], ref)
            unit_qty[i] = b
            if b == QTY_EXACT:
                unit_digits[i] = spell_digits(u[2])
            if not ok:
                local.append(f"QTY_INEXACT:{tok}")
                if details is not None:
                    details.append({"token": tok, "unit": i, "q": u[2], "ref": ref,
                                    "decoded": decode_qty(u[0], b, ref)})
    order_tok, order_qty = [], []
    order_digits = [list(NO_DIGITS) for _ in range(N_ORDER_SLOTS)]
    for o in canon["market"][:MAX_ORDERS]:
        tok = order_token(o)
        order_tok.append(ORDER_INDEX[tok])
        if ORDER_HAS_QTY[order_tok[-1]]:
            ref = reference(tok, actor_obs)
            b, ok = encode_qty(o[0], o[2], ref)
            order_qty.append(b)
            if b == QTY_EXACT:
                order_digits[len(order_tok) - 1] = spell_digits(o[2])
            if not ok:
                local.append(f"QTY_INEXACT:{tok}")
                if details is not None:
                    details.append({"token": tok, "order": len(order_tok) - 1, "q": o[2], "ref": ref,
                                    "decoded": decode_qty(o[0], b, ref)})
        else:
            order_qty.append(-1)
    order_tok.append(STOP)
    order_qty.append(-1)
    return {"unit_tok": unit_tok, "unit_qty": unit_qty, "unit_digits": unit_digits,
            "order_tok": order_tok, "order_qty": order_qty, "order_digits": order_digits,
            "exact": not local}


def decode_unit(tok, qty, actor_obs, unit_index, digits=None):
    tok = _index(tok)
    if not 0 <= tok < len(UNIT_VOCAB):
        raise ValueError("unit token out of range")
    name = UNIT_VOCAB[tok]
    if not UNIT_HAS_QTY[tok]:
        return name.split(":") if ":" in name else [name]
    op, item = name.split(":")
    return [op, item, decode_qty(op, qty, reference(name, actor_obs, unit_index), digits)]


def decode_order(tok, qty, actor_obs, digits=None):
    tok = _index(tok)
    if not 0 <= tok < len(ORDER_VOCAB) or tok == STOP:
        raise ValueError("order token out of range")
    name = ORDER_VOCAB[tok]
    if not ORDER_HAS_QTY[tok]:
        return [name]
    op, item = name.split(":")
    return [op, item, decode_qty(op, qty, reference(name, actor_obs), digits)]


def _row(rows, i):
    return rows[i] if rows is not None and i < len(rows) else None


def decode_step(labels, actor_obs):
    """Wire action for the observed units: farmer, every observed hand, queue to STOP.
    unit_digits / order_digits (optional) are read where the quantity class is QTY_EXACT."""
    n_units = 1 + hand_count(actor_obs)
    unit_digits, order_digits = labels.get("unit_digits"), labels.get("order_digits")
    units = []
    for i in range(n_units):
        tok = _index(labels["unit_tok"][i]) if i < min(MAX_UNITS, len(labels["unit_tok"])) else -1
        if tok == -1:
            units.append(["PASS"])
        else:
            units.append(decode_unit(tok, labels["unit_qty"][i], actor_obs, i, _row(unit_digits, i)))
    market = []
    for k, (tok, qty) in enumerate(zip(labels["order_tok"], labels["order_qty"])):
        tok = _index(tok)
        if tok == STOP or len(market) == MAX_ORDERS:
            break
        market.append(decode_order(tok, qty, actor_obs, _row(order_digits, k)))
    while market and market[-1] == ["NOOP"]:
        market.pop()
    return {"farmer": units[0], "hands": units[1:], "market": market}


# ---------------------------------------------------------------- grammar
def engine_order_unit_bound(path=ENGINE_SOURCE):
    """Largest number of units one order slot can commit, read from the engine source:
    _process_market's lockstep loop increments idx_esc before every iteration and breaks
    on `idx_esc >= LIMIT`, so at most LIMIT - 1 iterations (one unit per seat each) run."""
    tree = ast.parse(Path(path).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_process_market")
    limits = [node.comparators[0].value for node in ast.walk(fn)
              if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
              and node.left.id == "idx_esc" and len(node.ops) == 1 and isinstance(node.ops[0], ast.GtE)
              and isinstance(node.comparators[0], ast.Constant)]
    increments = [node for node in ast.walk(fn) if isinstance(node, ast.AugAssign)
                  and isinstance(node.target, ast.Name) and node.target.id == "idx_esc"
                  and isinstance(node.op, ast.Add) and isinstance(node.value, ast.Constant)
                  and node.value.value == 1]
    if len(limits) != 1 or len(increments) != 1:
        raise ValueError("engine market loop guard not found")
    return limits[0] - 1


def engine_grammar(path=ENGINE_SOURCE):
    """Accepted unit ops, order ops and argument tables read from the engine source."""
    tree = ast.parse(Path(path).read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass

    def compared_literals(fn, name):
        out = []
        for node in ast.walk(functions[fn]):
            if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                    and node.left.id == name and len(node.ops) == 1):
                right = node.comparators[0]
                if isinstance(node.ops[0], ast.Eq) and isinstance(right, ast.Constant):
                    out.append(right.value)
                elif isinstance(node.ops[0], ast.In) and isinstance(right, (ast.Tuple, ast.List)):
                    out.extend(e.value for e in right.elts if isinstance(e, ast.Constant))
                elif isinstance(node.ops[0], ast.In) and isinstance(right, ast.Name):
                    out.extend(constants[right.id])
        return list(dict.fromkeys(out))

    def items_of(node):
        if isinstance(node, (ast.Tuple, ast.List)):
            return [e.value for e in node.elts]
        value = constants[node.id]
        return list(value)

    lockstep_items = {}
    for node in ast.walk(functions["_process_market"]):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And) and len(node.values) == 2:
            a, b = node.values
            if (isinstance(a, ast.Compare) and isinstance(a.left, ast.Name) and a.left.id == "op"
                    and isinstance(b, ast.Compare) and isinstance(b.left, ast.Name) and b.left.id == "item"
                    and isinstance(b.ops[0], ast.In)):
                lockstep_items[a.comparators[0].value] = items_of(b.comparators[0])
    unit_ops = compared_literals("_apply_unit_action", "op")
    order_ops = compared_literals("_parse_order", "op")
    quote_ops = compared_literals("_process_market", "op")
    return {"unit_ops": unit_ops, "order_ops": order_ops, "lockstep_ops": quote_ops,
            "lockstep_items": lockstep_items,
            "crops": {k: v["seed"] for k, v in constants["CROPS"].items()},
            "animals": {k: v["cost"] for k, v in constants["ANIMALS"].items()},
            "products": list(constants["PRODUCTS"]),
            "moves": list(constants["FARMER_MOVES"])}


def vocab_summary():
    return {"UNIT_VOCAB": list(UNIT_VOCAB), "ORDER_VOCAB": list(ORDER_VOCAB),
            "MAX_UNITS": MAX_UNITS, "MAX_ORDERS": MAX_ORDERS, "QTY_BIN_COUNT": QTY_BIN_COUNT,
            "QTY_EXACT": QTY_EXACT, "QTY_CLASSES": QTY_CLASSES, "DIGIT_VOCAB": list(DIGIT_VOCAB),
            "MAX_QTY": MAX_QTY, "MAX_DIGITS": MAX_DIGITS, "DIGIT_SLOTS": DIGIT_SLOTS,
            "QTY_BINS": [list(b) for b in QTY_BINS],
            "UNIT_HAS_QTY": UNIT_HAS_QTY, "ORDER_HAS_QTY": ORDER_HAS_QTY,
            "QTY_CAP": QTY_CAP, "SHED_CAPACITY": SHED_CAPACITY,
            "pct_decode": "max(1, (2*num*ref + den) // (2*den))  (round half up, exact integers)",
            "references": {"SELL": "own shed stock of the product at t",
                           "PICKUP": "own shed stock of the item at t",
                           "PLACE": "amount of the item carried by the acting unit at t",
                           "BUY_SEED": "floor(own cash / engine seed price)",
                           "BUY_ANIMAL": "floor(own cash / engine animal price)",
                           "BUY_PRODUCT": "floor(own cash / max(1, visible market price))"}}


# ================================================================ measurement CLI
# Everything below runs only from the command line / tests; the codec above has
# no dependency on farm_neat, the engine or numpy.

def _form(a):
    if not isinstance(a, list):
        return "NONLIST:" + type(a).__name__
    if not a:
        return "EMPTY"
    head = a[0] if isinstance(a[0], str) else "T:" + type(a[0]).__name__
    return "|".join([head] + [type(x).__name__ for x in a[1:]])


def clone_env(env):
    """Independent copy of an ExactKaggricultureEnvironment (the engine module is shared)."""
    import copy
    twin = object.__new__(type(env))
    twin.__dict__.update({k: v if k == "engine" else copy.deepcopy(v) for k, v in env.__dict__.items()})
    return twin


def _state_wire(env):
    from farm_neat.game.exact_engine import _copy_wire
    obs0 = env.state[0].observation
    return _copy_wire({"farms": obs0.farms, "market": obs0.market, "town": obs0.town,
                       "private": [env.state[p].observation.private for p in (0, 1)],
                       "rewards": [env.state[p].reward for p in (0, 1)],
                       "status": [env.state[p].status for p in (0, 1)]})


def measure_episode(task, roundtrip=False, semantic=False, list_limit=50):
    """Scan one replay (both seats, t = 0..718): forms, hands, queue lengths,
    quantity rows (token, is_order, q, ref, t, seat), optional codec round trip
    through bc.public_observation and optional exact-engine semantic check."""
    import copy
    import json
    import zipfile
    from collections import Counter
    import numpy as np
    archive, member, day, eid = task
    with zipfile.ZipFile(archive) as z:
        replay = json.loads(z.read(member))
    steps = replay["steps"]
    res = {"episode_id": eid, "day": day, "teams": replay.get("info", {}).get("TeamNames"),
           "complete": len(steps) == 720 and replay.get("statuses") == ["DONE", "DONE"]}
    if not res["complete"]:
        return res, np.zeros((0, 6), np.int32)
    c = {k: Counter() for k in ("unit_forms", "order_forms", "unit_tokens", "order_tokens",
                                "farmer_tokens", "hand_tokens",
                                "unrepresentable", "clamped", "hands_hist", "queue_len",
                                "raw_queue_len", "order_qty_type", "unit_qty_type")}
    n = {k: 0 for k in ("steps", "extra_hand_steps", "extra_hand_plant_steps", "missing_hand_steps",
                        "raw_queue_over_max", "mid_noop_orders", "trailing_noop_orders",
                        "action_not_dict")}
    max_hands = max_action_hands = max_cash = 0
    rows = []
    for t in range(719):
        for s in (0, 1):
            obs = steps[t][s]["observation"]
            action = steps[t + 1][s].get("action")
            n["steps"] += 1
            if not isinstance(action, dict):
                n["action_not_dict"] += 1
                action = {}
            nh = len(obs["farms"][s]["hands"])
            max_hands = max(max_hands, nh)
            max_cash = max(max_cash, obs["farms"][s]["money"])
            c["hands_hist"][nh] += 1
            hands = action.get("hands", [])
            hands = hands if isinstance(hands, list) else []
            max_action_hands = max(max_action_hands, len(hands))
            if len(hands) > nh:
                n["extra_hand_steps"] += 1
                if any(isinstance(h, list) and len(h) >= 2 and h[0] == "PLANT" for h in hands[nh:]):
                    n["extra_hand_plant_steps"] += 1
            elif len(hands) < nh:
                n["missing_hand_steps"] += 1
            units = [action.get("farmer", ["PASS"]), *hands[:nh], *[["PASS"]] * max(0, nh - len(hands))]
            market = action.get("market", [])
            market = market if isinstance(market, list) else []
            c["raw_queue_len"][len(market)] += 1
            if len(market) > MAX_ORDERS:
                n["raw_queue_over_max"] += 1
            view = dict(obs, player=s)
            for i, u in enumerate(units):
                c["unit_forms"][_form(u)] += 1
                try:
                    cu = canonical_unit(u)
                except Unrepresentable as error:
                    c["unrepresentable"][str(error)] += 1
                    continue
                tok = unit_token(cu)
                c["unit_tokens"][tok] += 1
                c["farmer_tokens" if i == 0 else "hand_tokens"][tok] += 1
                if cu[0] in ITEM_UNIT_OPS:
                    c["unit_qty_type"][type(u[2]).__name__ if len(u) >= 3 else "absent"] += 1
                    if len(u) >= 3 and int(u[2]) != cu[2]:
                        c["clamped"][cu[0]] += 1
                    rows.append((UNIT_INDEX[tok], 0, cu[2], reference(tok, view, i), t, s))
            canon = []
            for o in market[:MAX_ORDERS]:
                c["order_forms"][_form(o)] += 1
                try:
                    canon.append((o, canonical_order(o)))
                except Unrepresentable as error:
                    c["unrepresentable"][str(error)] += 1
            k = len(canon)
            while k and canon[k - 1][1] == ["NOOP"]:
                k -= 1
            n["trailing_noop_orders"] += len(canon) - k
            c["queue_len"][k] += 1
            for o, co in canon[:k]:
                tok = order_token(co)
                c["order_tokens"][tok] += 1
                if co == ["NOOP"]:
                    n["mid_noop_orders"] += 1
                if co[0] in MARKET_QTY_OPS:
                    c["order_qty_type"][type(o[2]).__name__] += 1
                    if int(o[2]) != co[2]:
                        c["clamped"][co[0]] += 1
                    rows.append((ORDER_INDEX[tok], 1, co[2], reference(tok, view), t, s))
    res.update({k: dict(v) for k, v in c.items()})
    res.update(n, max_obs_hands=max_hands, max_action_hands=max_action_hands, max_cash=max_cash)
    if roundtrip or semantic:
        from farm_neat.training import bc
        rt = {"steps": 0, "exact_steps": 0, "exact_but_mismatch": 0, "reasons": Counter(), "cases": []}
        decoded = {}
        for t in range(719):
            for s in (0, 1):
                view = bc.public_observation(replay, t, s)
                action = steps[t + 1][s].get("action")
                reasons, details = [], []
                try:
                    labels = encode_step(action, view, reasons, details)
                    out = decode_step(labels, view)
                    canon = canonical(action, view)
                except Unrepresentable as error:
                    reasons.append(str(error))
                    labels, out, canon = {"exact": False}, None, None
                rt["steps"] += 1
                if labels["exact"]:
                    rt["exact_steps"] += 1
                    if out != canon:
                        rt["exact_but_mismatch"] += 1
                else:
                    for r in dict.fromkeys(reasons):
                        rt["reasons"][r] += 1
                    if list_limit is None or len(rt["cases"]) < list_limit:
                        rt["cases"].append({"episode_id": eid, "step": t, "seat": s,
                                            "reasons": reasons, "details": details})
                decoded[t, s] = (labels["exact"], out if out is not None else action)
        rt["reasons"] = dict(rt["reasons"])
        res["roundtrip"] = rt
    if semantic:
        from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
        env = ExactKaggricultureEnvironment(seed=replay["info"]["seed"], configuration=replay["configuration"])
        sem = {"steps": 0, "equal": 0, "exact_steps": 0, "exact_equal": 0, "diff_cases": []}
        for t in range(719):
            raw = [steps[t + 1][s]["action"] for s in (0, 1)]
            dec = [decoded[t, s][1] for s in (0, 1)]
            both_exact = decoded[t, 0][0] and decoded[t, 1][0]
            twin = clone_env(env)
            env.step(raw)
            twin.step(dec)
            same = _state_wire(env) == _state_wire(twin)
            sem["steps"] += 1
            sem["equal"] += same
            sem["exact_steps"] += both_exact
            sem["exact_equal"] += both_exact and same
            if not same and (list_limit is None or len(sem["diff_cases"]) < list_limit):
                sem["diff_cases"].append({"episode_id": eid, "step": t, "both_exact": both_exact})
        res["semantic"] = sem
    return res, np.asarray(rows, np.int32).reshape(-1, 6)


def set_reference(obs, token, unit_index, ref):
    """Copy of obs whose reference(token, unit_index) equals ref."""
    import copy
    op, item = token.split(":")
    obs = copy.deepcopy(obs)
    seat = obs["player"]
    if op in ("SELL", "PICKUP"):
        obs["private"]["shed"][item] = ref
    elif op == "PLACE":
        obs["private"]["inventories"][unit_index] = {item: ref}
    else:
        price = (CROPS[item] if op == "BUY_SEED" else ANIMALS[item] if op == "BUY_ANIMAL"
                 else max(1, int(obs["market"]["prices"][item])))
        obs["farms"][seat]["money"] = float(ref * price)
    if reference(token, obs, unit_index) != ref:
        raise ValueError("reference construction failed")
    return obs


FORWARD_REFS = (0, 1, 2, 3, 5, 7, 10, 13, 20, 33, 50, 64, 99, 100)
FORWARD_SEED_REFS = (150, 300, 1000, 3000)


def forward_coverage(obs, refs=FORWARD_REFS, seed_refs=FORWARD_SEED_REFS):
    """Every token at every unit slot of obs; for quantity tokens every q in
    1..cap (1..max(100, ref) for BUY_SEED) at each reference. Returns
    ({token: [exact, checked]}, [(token, action) where exact but decode != canonical],
    {op: {ref: [inexact q values]}})."""
    n_units = 1 + hand_count(obs)
    report, mismatches, gaps = {}, [], {}

    def check(token, action, o):
        labels = encode_step(action, o)
        out = decode_step(labels, o)
        if labels["exact"] and out != canonical(action, o):
            mismatches.append((token, action))
        report.setdefault(token, [0, 0])
        report[token][0] += labels["exact"]
        report[token][1] += 1
        return labels["exact"]

    for token in UNIT_VOCAB:
        op = token.split(":")[0]
        for i in range(n_units):
            if not UNIT_HAS_QTY[UNIT_INDEX[token]]:
                units = [["PASS"]] * n_units
                units[i] = token.split(":")
                check(token, {"farmer": units[0], "hands": units[1:], "market": []}, obs)
                continue
            item = token.split(":")[1]
            for ref in refs:
                o = set_reference(obs, token, i, ref)
                for q in range(1, QTY_CAP[op] + 1):
                    units = [["PASS"]] * n_units
                    units[i] = [op, item, q]
                    if not check(token, {"farmer": units[0], "hands": units[1:], "market": []}, o) and i == 0:
                        gaps.setdefault(op, {}).setdefault(str(ref), set()).add(q)
    for token in ORDER_VOCAB[1:]:
        op = token.split(":")[0]
        if not ORDER_HAS_QTY[ORDER_INDEX[token]]:
            wire = [["PASS"], ["HIRE"]] if token == "NOOP" else [token.split(":")]  # NOOP survives mid-queue
            check(token, {"farmer": ["PASS"], "hands": [], "market": wire}, obs)
            continue
        item = token.split(":")[1]
        for ref in tuple(refs) + (tuple(seed_refs) if op == "BUY_SEED" else ()):
            o = set_reference(obs, token, 0, ref)
            top = max(100, ref) if op == "BUY_SEED" else QTY_CAP[op]
            for q in range(1, top + 1):
                if not check(token, {"farmer": ["PASS"], "hands": [], "market": [[op, item, q]]}, o):
                    gaps.setdefault(op, {}).setdefault(str(ref), set()).add(q)
    gaps = {op: {ref: sorted(v) for ref, v in by.items()} for op, by in gaps.items()}
    return report, mismatches, gaps


def engine_accepts(engine, grammar, obs, action):
    """Raise ValueError unless every unit op/argument and every non-NOOP order of a
    decoded action is one the engine parses (unit op grammar, _parse_order and the
    lockstep item tables); NOOP must be dropped by _parse_order."""
    shed = set(engine._new_private()["shed"])
    if len(action["hands"]) != hand_count(obs) or len(action["market"]) > MAX_ORDERS:
        raise ValueError("unit count or queue length")
    for u in [action["farmer"], *action["hands"]]:
        ok = u[0] in grammar["unit_ops"]
        if u[0] == "PLANT":
            ok = ok and len(u) == 2 and u[1] in engine.CROPS
        elif u[0] in ITEM_UNIT_OPS:
            ok = ok and len(u) == 3 and u[1] in shed and type(u[2]) is int and u[2] >= 1
        else:
            ok = ok and len(u) == 1
        if not ok:
            raise ValueError(f"unit action not accepted: {u}")
    for o in action["market"]:
        parsed = engine._parse_order(o)
        if o == ["NOOP"]:
            if parsed is not None:
                raise ValueError("NOOP must be dropped by the engine parser")
            continue
        if parsed is None or (o[0] in grammar["lockstep_items"] and not (
                o[1] in grammar["lockstep_items"][o[0]] and type(o[2]) is int and o[2] >= 1)):
            raise ValueError(f"order not accepted: {o}")


def backward_check(env, seat=0):
    """Decode every (unit token, bin) and (order token, bin) on env's state for
    seat, check engine acceptance, and step a clone of env with it."""
    engine, grammar = env.engine, engine_grammar()
    obs = _view(env, seat)
    n_units = 1 + hand_count(obs)
    pass_action = {"farmer": ["PASS"], "hands": [], "market": []}
    units = [(t, b) for t in range(len(UNIT_VOCAB)) for b in range(QTY_CLASSES) if UNIT_HAS_QTY[t] or b == 0]
    orders = [(t, b) for t in range(len(ORDER_VOCAB)) if t != STOP
              for b in range(QTY_CLASSES) if ORDER_HAS_QTY[t] or b == 0]
    # QTY_EXACT combos get a digit sequence; cycling through these covers 0, leading
    # zeros, the engine bound, over-long and END-less spellings.
    spellings = ([DIGIT_END], [0, DIGIT_END], [0, 0, 7, DIGIT_END], [4, 2, DIGIT_END],
                 [9, 9, 9, 9, 9, DIGIT_END], [9, 9, 9, 9, 9, 9], [1, 0, 0, 0, 0, 0], [3, -1])
    exact_digits = [spellings[k % len(spellings)] for k in range(max(len(units), len(orders)))]
    steps = 0
    for start in range(0, len(units), n_units):
        part = units[start:start + n_units]
        labels = {"unit_tok": [t for t, _ in part] + [-1] * (MAX_UNITS - len(part)),
                  "unit_qty": [b for _, b in part] + [-1] * (MAX_UNITS - len(part)),
                  "unit_digits": exact_digits[start:start + len(part)],
                  "order_tok": [STOP], "order_qty": [-1]}
        action = decode_step(labels, obs)
        engine_accepts(engine, grammar, obs, action)
        clone_env(env).step([action, pass_action] if seat == 0 else [pass_action, action])
        steps += 1
    for start in range(0, len(orders), MAX_ORDERS):
        part = orders[start:start + MAX_ORDERS]
        labels = {"unit_tok": [PASS] + [-1] * (MAX_UNITS - 1), "unit_qty": [-1] * MAX_UNITS,
                  "order_tok": [t for t, _ in part] + [STOP], "order_qty": [b for _, b in part] + [-1],
                  "order_digits": exact_digits[start:start + len(part)] + [None]}
        action = decode_step(labels, obs)
        engine_accepts(engine, grammar, obs, action)
        clone_env(env).step([action, pass_action] if seat == 0 else [pass_action, action])
        steps += 1
    return {"unit_combos": len(units), "order_combos": len(orders), "engine_steps": steps,
            "units_per_step": n_units, "accepted": len(units) + len(orders)}


def _view(env, seat):
    from farm_neat.training import bc
    return bc.actor_observation(env.agent_observation(seat))


def mid_game_env(replay, min_hands=3, start=300):
    """Exact engine at the first boundary >= start where seat 0 has >= min_hands
    hands and shed stock, reached by the recorded joint actions."""
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    from farm_neat.training import bc
    steps = replay["steps"]
    target = next(t for t in range(start, 719)
                  if len(steps[t][0]["observation"]["farms"][0]["hands"]) >= min_hands
                  and sum(steps[t][0]["observation"]["private"]["shed"].values()) > 0)
    env = ExactKaggricultureEnvironment(seed=replay["info"]["seed"], configuration=replay["configuration"])
    for t in range(target):
        env.step([steps[t + 1][s]["action"] for s in (0, 1)])
    if _view(env, 0) != bc.public_observation(replay, target, 0):
        raise ValueError("engine reconstruction differs from the replay")
    return env, target


def _run_pool(fn, tasks, workers):
    import multiprocessing as mp
    if workers <= 1:
        return [fn(t) for t in tasks]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(fn, tasks, chunksize=1)


def sample_tasks(per_day, offset=0):
    """Deterministic per-day sample: members sorted by sha256(member name)."""
    import hashlib
    import data
    by_day = {}
    for task in data.list_members(split="all"):
        by_day.setdefault(task[2], []).append(task)
    out = []
    for day in sorted(by_day):
        tasks = sorted(by_day[day], key=lambda t: hashlib.sha256(t[1].encode()).hexdigest())
        out += tasks[offset:] if per_day is None else tasks[offset:offset + per_day]
    return out


def _candidate_fractions(dens=(100,), extra_dens=tuple(range(2, 13))):
    from fractions import Fraction
    fr = set()
    for den in dens:
        fr.update(Fraction(k, den) for k in range(1, den + 1))
    for den in extra_dens:
        fr.update(Fraction(k, den) for k in range(1, den + 1))
    return sorted(fr)


def _decode_many(kind, num, den, ref, cap):
    import numpy as np
    if kind == "n":
        v = np.full(ref.shape, num, np.int64)
    else:
        v = np.maximum(1, (2 * num * ref.astype(np.int64) + den) // (2 * den))
    return np.where(cap > 0, np.minimum(v, np.where(cap > 0, cap, 1)), v)


def search_bins(qrows, n_bins=QTY_BIN_COUNT):
    """Greedy exact-coverage bin choice on (token, is_order, q, ref) rows.

    Strategies: (a) discrete 1..K plus the greedy best n_bins-K fractions, for
    every K; (b) fully greedy mixture of any discrete value and fraction.
    Returns the best bins (ties: contiguous strategy) and the whole search table."""
    import numpy as np
    tok_cap = np.array([QTY_CAP[t.split(":")[0]] or 0 if UNIT_HAS_QTY[i] else 0
                        for i, t in enumerate(UNIT_VOCAB)])
    ord_cap = np.array([QTY_CAP[t.split(":")[0]] or 0 if ORDER_HAS_QTY[i] else 0
                        for i, t in enumerate(ORDER_VOCAB)])
    is_order = qrows[:, 1] == 1
    cap = np.zeros(len(qrows), np.int64)
    cap[is_order] = ord_cap[qrows[is_order, 0]]
    cap[~is_order] = tok_cap[qrows[~is_order, 0]]
    key = np.stack([qrows[:, 2], qrows[:, 3], cap], 1)
    uniq, weight = np.unique(key, axis=0, return_counts=True)
    q, ref, cp = uniq[:, 0].astype(np.int64), uniq[:, 1].astype(np.int64), uniq[:, 2].astype(np.int64)
    total = weight.sum()
    fracs = _candidate_fractions()
    disc_values = sorted(set(int(v) for v in q))
    cands = [("n", v, 1) for v in disc_values] + [("p", f.numerator, f.denominator) for f in fracs]
    cover = np.stack([_decode_many(k, a, b, ref, cp) == q for k, a, b in cands]).astype(np.float32)
    w = weight.astype(np.float32)
    is_pct = np.array([k == "p" for k, _, _ in cands])

    def greedy(chosen, allowed, budget):
        covered = cover[chosen].max(0) if chosen else np.zeros(len(q), np.float32)
        chosen = list(chosen)
        for _ in range(budget):
            gain = cover @ (w * (1 - covered))
            gain[~allowed] = -1
            gain[chosen] = -1
            best = int(gain.argmax())
            if gain[best] <= 0:
                break
            chosen.append(best)
            covered = np.maximum(covered, cover[best])
        return chosen, float((covered * w).sum())

    table = []
    best = None
    index = {c: i for i, c in enumerate(cands)}
    for k in range(0, n_bins + 1):
        disc = [index[("n", v, 1)] for v in range(1, k + 1) if ("n", v, 1) in index]
        chosen, cov = greedy(disc, is_pct, n_bins - k)
        # pad with unused discrete values after the run so every strategy has n_bins
        table.append({"strategy": "contiguous", "K": k, "exact": cov / total})
        if best is None or cov > best[1]:
            best = (chosen, cov, "contiguous", k)
    chosen, cov = greedy([], np.ones(len(cands), bool), n_bins)
    table.append({"strategy": "greedy_mixture", "K": sum(1 for i in chosen if not is_pct[i]),
                  "exact": cov / total})
    if cov > best[1]:
        best = (chosen, cov, "greedy_mixture", None)
    chosen = best[0]
    # Fill to exactly n_bins with the next discrete counts not yet chosen (zero gain).
    have = {cands[i] for i in chosen}
    v = 1
    bins = [cands[i] for i in chosen]
    while len(bins) < n_bins:
        if ("n", v, 1) not in have:
            bins.append(("n", v, 1))
        v += 1
    bins = sorted(bins, key=lambda b: (b[0] != "n", b[1] / b[2]))
    return {"bins": bins, "strategy": best[2], "K": best[3], "fit_exact": best[1] / total,
            "fit_instances": int(total), "table": table,
            "candidates": {"discrete": len(disc_values), "fractions": len(fracs)}}


def qty_coverage(qrows, bins):
    """Exact share per quantity-bearing token (and per op) with the given bins."""
    import numpy as np
    from collections import defaultdict
    out = defaultdict(lambda: [0, 0])
    global QTY_BINS
    saved = QTY_BINS
    QTY_BINS = tuple(tuple(b) for b in bins)
    try:
        cache = {}
        for tok, is_order, q, ref in qrows[:, :4].tolist():
            name = ORDER_VOCAB[tok] if is_order else UNIT_VOCAB[tok]
            k = (name, q, ref)
            if k not in cache:
                cache[k] = encode_qty(name.split(":")[0], q, ref, escape=False)[1]
            ok = cache[k]
            for key in (name, name.split(":")[0], "ALL"):
                out[key][0] += ok
                out[key][1] += 1
    finally:
        QTY_BINS = saved
    return {k: {"exact": v[0], "instances": v[1], "share": v[0] / v[1]} for k, v in sorted(out.items())}


def _merge(results):
    from collections import Counter
    agg = {}
    for res in results:
        for key, value in res.items():
            if isinstance(value, dict) and key not in ("roundtrip", "semantic"):
                agg.setdefault(key, Counter()).update({str(k): v for k, v in value.items()})
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and key != "episode_id":
                if key.startswith("max_"):
                    agg[key] = max(agg.get(key, 0), value)
                else:
                    agg[key] = agg.get(key, 0) + value
    return {k: (dict(sorted(v.items(), key=lambda kv: -kv[1])) if isinstance(v, Counter) else v)
            for k, v in agg.items()}


def _quantity_distribution(qrows, top=25):
    import numpy as np
    from collections import Counter
    out = {}
    for is_order, vocab in ((0, UNIT_VOCAB), (1, ORDER_VOCAB)):
        part = qrows[qrows[:, 1] == is_order]
        for op in sorted({vocab[i].split(":")[0] for i in set(part[:, 0].tolist())}):
            ids = [i for i, t in enumerate(vocab) if t.split(":")[0] == op]
            sel = part[np.isin(part[:, 0], ids)]
            qs, refs = sel[:, 2], sel[:, 3]
            out[op] = {"instances": int(len(sel)), "q_max": int(qs.max()),
                       "q_equals_ref_share": float((qs == refs).mean()),
                       "q_top": {str(k): v for k, v in Counter(qs.tolist()).most_common(top)},
                       "q_quantiles": {str(p): float(np.quantile(qs, p)) for p in (0.5, 0.9, 0.99, 1.0)},
                       "ref_quantiles": {str(p): float(np.quantile(refs, p)) for p in (0.5, 0.9, 0.99, 1.0)}}
    return out


def cmd_measure(args):
    import json
    import time
    import numpy as np
    start = time.time()
    fit_tasks = sample_tasks(args.fit_per_day)
    all_tasks = sample_tasks(None) if args.all else fit_tasks + sample_tasks(args.eval_per_day, args.fit_per_day)
    fit_ids = {t[3] for t in fit_tasks}
    results = _run_pool(measure_episode, all_tasks, args.workers)
    stats, rows = [r for r, _ in results], [q for _, q in results]
    fit_rows = np.concatenate([q for (r, q) in results if r["episode_id"] in fit_ids])
    eval_rows = np.concatenate([q for (r, q) in results if r["episode_id"] not in fit_ids])
    search = search_bins(fit_rows)
    fit_stats = [r for r in stats if r["episode_id"] in fit_ids]
    report = {
        "schema": "primitive-bc-vocab-v1",
        "vocab": vocab_summary(),
        "engine_grammar": engine_grammar(),
        "chosen_bins": [list(b) for b in search["bins"]],
        "bin_search": {k: v for k, v in search.items() if k != "bins"},
        "fit_sample": {"episodes": len(fit_tasks), "per_day": args.fit_per_day,
                       "days": sorted({t[2] for t in fit_tasks}),
                       "selection": "per day, members sorted by sha256(member name), first fit_per_day; sealed excluded",
                       "stats": _merge(fit_stats), "quantities": _quantity_distribution(fit_rows),
                       "exact_coverage_chosen_bins": qty_coverage(fit_rows, search["bins"])},
        "eval_sample": {"episodes": len(all_tasks) - len(fit_tasks),
                        "exact_coverage_chosen_bins": qty_coverage(eval_rows, search["bins"])
                        if len(eval_rows) else None},
        "all_scanned": {"episodes": len(all_tasks), "complete": sum(r["complete"] for r in stats),
                        "stats": _merge(stats)},
        "seconds": round(time.time() - start, 1)}
    merged = report["all_scanned"]["stats"]
    report["max_units"] = {
        "MAX_UNITS": MAX_UNITS,
        "max_obs_hands_all_scanned": merged["max_obs_hands"],
        "max_action_hand_entries_all_scanned": merged["max_action_hands"],
        "max_own_cash_all_scanned": merged["max_cash"],
        "engine_practical_max_hands": practical_max_hands(merged["max_cash"]),
        "rule": ("hires cost fib(k) (1,1,2,3,5,...) for the k-th hire of a day and hands reset at day end, "
                 "so n hands in one day cost fib(n+1)-1; engine_practical_max_hands is the largest n "
                 "affordable with the largest own cash observed at any step"),
        "beyond": ("encode_step marks HANDS_OVER_MAX_UNITS (exact=False); decode_step emits PASS for "
                   "hands past MAX_UNITS-1 (a hand without an entry is an engine no-op)")}
    report["label_stats"] = label_stats(merged)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"bins": report["chosen_bins"], "strategy": search["strategy"], "K": search["K"],
                      "fit_exact": search["fit_exact"],
                      "eval_exact": report["eval_sample"]["exact_coverage_chosen_bins"]["ALL"]
                      if len(eval_rows) else None,
                      "max_obs_hands": report["all_scanned"]["stats"].get("max_obs_hands")}, indent=1))


def practical_max_hands(cash):
    n, spent, a, b = 0, 0, 1, 1
    while spent + a <= cash:
        spent += a
        n += 1
        a, b = b, a + b
    return n


def label_stats(merged):
    farmer, hands = merged.get("farmer_tokens", {}), merged.get("hand_tokens", {})
    qlen = {int(k): v for k, v in merged.get("queue_len", {}).items()}
    steps = sum(qlen.values())
    return {"steps": steps,
            "farmer_pass_share": farmer.get("PASS", 0) / max(1, sum(farmer.values())),
            "hand_pass_share": hands.get("PASS", 0) / max(1, sum(hands.values())),
            "hand_actions": sum(hands.values()),
            "orders_per_step_mean": sum(k * v for k, v in qlen.items()) / max(1, steps),
            "empty_queue_share": qlen.get(0, 0) / max(1, steps),
            "full_queue_share": qlen.get(MAX_ORDERS, 0) / max(1, steps),
            "queue_len_hist": dict(sorted(qlen.items())),
            "farmer_top": dict(sorted(farmer.items(), key=lambda kv: -kv[1])[:12]),
            "hand_top": dict(sorted(hands.items(), key=lambda kv: -kv[1])[:12]),
            "order_top": dict(sorted(merged.get("order_tokens", {}).items(), key=lambda kv: -kv[1])[:12])}


def _roundtrip_task(task):
    return measure_episode(task, roundtrip=True, semantic=True, list_limit=None)[0]


def cmd_coverage(args):
    import json
    import time
    import zipfile
    from collections import Counter
    start = time.time()
    tasks = sample_tasks(args.per_day)
    vocab = json.loads(Path(args.vocab).read_text())
    # forward/backward on a real mid-game state of the first sampled episode with >= 3 hands
    for task in tasks[10:] + tasks[:10]:
        with zipfile.ZipFile(task[0]) as z:
            replay = json.loads(z.read(task[1]))
        try:
            env, boundary = mid_game_env(replay)
            break
        except StopIteration:
            continue
    obs = _view(env, 0)
    fwd, mismatches, gaps = forward_coverage(obs)
    per_op = {}
    for token, (ok, total) in fwd.items():
        op = token.split(":")[0]
        per_op.setdefault(op, [0, 0])
        per_op[op][0] += ok
        per_op[op][1] += total
    backward = backward_check(env)
    # market expressiveness
    empty = encode_step({"market": []}, obs)
    full_wire = [["HIRE"], ["SELL", "MILK", 3], ["BUY_SEED", "WHEAT", 5], ["PASS"], ["BUY_LAND"],
                 ["SELL", "WHEAT", 1], ["BUY_PRODUCT", "FERTILIZER", 2], ["BUY_ANIMAL", "COW", 1],
                 ["HIRE"], ["SELL", "EGG", 2]]
    full = encode_step({"market": full_wire}, obs)
    # replay round trip + exact-engine semantics
    results = _run_pool(_roundtrip_task, tasks, args.workers)
    rt = {"episodes": len(results), "steps": 0, "exact_steps": 0, "exact_but_mismatch": 0,
          "reasons": Counter(), "cases_by_reason": {}}
    sem = {"steps": 0, "equal": 0, "exact_steps": 0, "exact_equal": 0, "diff_cases": []}
    for res in results:
        r = res["roundtrip"]
        for k in ("steps", "exact_steps", "exact_but_mismatch"):
            rt[k] += r[k]
        rt["reasons"].update(r["reasons"])
        for case in r["cases"]:
            for reason in dict.fromkeys(case["reasons"]):
                code = reason.split(":")[0]
                rt["cases_by_reason"].setdefault(code, []).append(case)
        for k in ("steps", "equal", "exact_steps", "exact_equal"):
            sem[k] += res["semantic"][k]
        sem["diff_cases"] += res["semantic"]["diff_cases"]
    rt["exact_share"] = rt["exact_steps"] / max(1, rt["steps"])
    rt["reasons"] = dict(rt["reasons"].most_common())
    report = {
        "schema": "primitive-bc-coverage-v1",
        "vocab_bins": [list(b) for b in QTY_BINS], "MAX_UNITS": MAX_UNITS,
        "forward": {"state": {"episode_id": task[3], "boundary": boundary, "seat": 0,
                              "units": 1 + hand_count(obs)},
                    "refs": list(FORWARD_REFS), "buy_seed_extra_refs": list(FORWARD_SEED_REFS),
                    "q_range": "1..100 (shed-capped ops), 1..max(100, ref) for BUY_SEED",
                    "exact_but_decode_differs": len(mismatches),
                    "per_op": {op: {"exact": v[0], "checked": v[1], "share": v[0] / v[1]}
                               for op, v in sorted(per_op.items())},
                    "per_token": {t: {"exact": v[0], "checked": v[1]} for t, v in fwd.items()},
                    "inexact_q_farmer_or_order_by_ref": gaps},
        "backward": backward,
        "max_units": vocab.get("max_units"),
        "market": {"empty_queue_labels": empty["order_tok"], "empty_decodes_to": decode_step(empty, obs)["market"],
                   "full_queue_tokens": len(full["order_tok"]), "full_queue_exact": full["exact"],
                   "full_decodes_to_len": len(decode_step(full, obs)["market"]),
                   "raw_queue_over_10_truncated": "engine keeps q[:10]; canonical keeps the first 10"},
        "replay_roundtrip": rt,
        "engine_semantics": {**sem, "method": ("replay each sampled episode in the exact engine; at every step a clone "
                             "is stepped with decode(encode(a)) for both seats (canonical(a) where "
                             "encode is inexact) and its full state (farms, privates, market, town, "
                             "rewards) compared with the recorded-action step")},
        "sample": {"episodes": len(tasks), "per_day": args.per_day,
                   "selection": "per day, members sorted by sha256(member name); sealed excluded"},
        "seconds": round(time.time() - start, 1)}
    Path(args.out).write_text(json.dumps(report, indent=1, default=list) + "\n")
    print(json.dumps({"forward_per_op": report["forward"]["per_op"], "backward": backward,
                      "replay_exact_share": rt["exact_share"], "steps": rt["steps"],
                      "reasons": rt["reasons"], "semantic": {k: sem[k] for k in ("steps", "equal", "exact_steps", "exact_equal")}},
                     indent=1))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="primitive-BC codec measurements")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="scan replays, choose quantity bins, write VOCAB.json")
    m.add_argument("--fit-per-day", type=int, default=16)
    m.add_argument("--eval-per-day", type=int, default=8)
    m.add_argument("--all", action="store_true", help="scan every episode of every day")
    m.add_argument("--workers", type=int, default=8)
    m.add_argument("--out", default=str(HERE / "VOCAB.json"))
    c = sub.add_parser("coverage", help="forward/backward/bounds/replay round trip -> COVERAGE.json")
    c.add_argument("--per-day", type=int, default=16)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--vocab", default=str(HERE / "VOCAB.json"))
    c.add_argument("--out", default=str(HERE / "COVERAGE.json"))
    args = ap.parse_args(argv)
    if args.workers > 8:
        raise SystemExit("at most 8 worker processes")
    {"measure": cmd_measure, "coverage": cmd_coverage}[args.cmd](args)


if __name__ == "__main__":
    import sys
    sys.path[:0] = [str(HERE.parents[3] / "src"), str(HERE)]
    main()
