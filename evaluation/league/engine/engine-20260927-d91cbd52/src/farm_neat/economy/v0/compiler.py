"""Plan -> executor stages. The compiler owns tiles, workers, purchases, reserves and timing.

Conventions (frozen with the vocabulary): explicit TILE scopes ordered by shed distance
within NW, NE, SW, SE; animals first, then crops by birth day, each quadrant open from its
purchase step; when that order would leave an instance unplaced (a site kept for a later
animal), birth order instead (K-V0-LAND-RELEASE-LIFT-9F4D7ABF); a cohort instance keeps its
tile for the whole game (stable identity across epochs); HANDS as one stage per run of equal
daily hand counts from the action bound; LAND as one stage per quadrant at its projected
purchase step; SALE stages per product from the market overlay; CARRY as a BUY stage at the
demand step and a SALE stage the step after. Same-turn BUY-then-SELL words (FRONTRUN,
PIGGYBACK) have no executor rule and are never emitted.
"""
from __future__ import annotations

from farm_neat.economy.action_space import validate_program
from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import Projection, State, frontier, hold_release

SHEDS = ((4, 4), (5, 4), (4, 5), (5, 5))
QUADS = {"NW": (range(0, 5), range(0, 5)), "NE": (range(5, 10), range(0, 5)),
         "SW": (range(0, 5), range(5, 10)), "SE": (range(5, 10), range(5, 10))}
COMMAND_BUDGET = 256
GAME_END = "GAME_END"


def coords(quadrants: int) -> list[tuple[int, int]]:
    """Tiles of the first `quadrants` quadrants in the engine's land order, nearest shed first."""
    out = []
    for q in V.QUADRANTS[:quadrants]:
        xs, ys = QUADS[q]
        pts = [(x, y) for y in ys for x in xs]
        pts.sort(key=lambda p: (min(abs(p[0] - sx) + abs(p[1] - sy) for sx, sy in SHEDS), p[1], p[0]))
        out += pts
    return out


def quadrant_index(tile: tuple[int, int]) -> int:
    x, y = tile
    return V.QUADRANTS.index(("N" if y < 5 else "S") + ("W" if x < 5 else "E"))


def window(module: int, regime: int) -> tuple[int, int | str]:
    """Executor window of one instance: birth dawn to the last occupied day, or the game end."""
    row = V.tape(module, regime)
    birth = V.MODULES[module]["birth"]
    last = max((d for d, o in enumerate(row["occupancy"]) if o), default=birth)
    return 24 * birth, GAME_END if last >= V.SEASON_DAYS - 1 else 24 * last + 23


def _end(until) -> int:
    return V.LAST_STEP if until == GAME_END else until


def allocate_sites(plan: Plan, land_steps: dict[int, int], previous: dict | None = None, *, owned: int = 1,
                   now: int = 0) -> dict[tuple[int, int], tuple[int, int]]:
    """The compiler's allocation. Animals first (they are served every day, so they take the
    tiles nearest the shed) whenever that places every instance; otherwise birth order, which
    leaves an instance unplaced only when no site is open at its birth."""
    first = allocate(plan, land_steps, previous, owned=owned, now=now)
    if not unplaced(first):
        return first
    return allocate_by_birth(plan, land_steps, previous, owned=owned, now=now)


def allocate_by_birth(plan: Plan, land_steps: dict[int, int], previous: dict | None = None, *, owned: int = 1,
                      now: int = 0) -> dict[tuple[int, int], tuple[int, int]]:
    """The compiler's allocation: unborn instances in birth order (animals first on a tie).

    With started instances pinned and every unborn window starting later, a birth-ordered
    greedy leaves an instance unplaced only when every site open at its birth is occupied,
    so no plan the projection counts as feasible loses an instance to visit order."""
    order = sorted(plan.instances(), key=lambda key: V.MODULES[key[0]]["birth"])
    return _allocate(plan, land_steps, previous, owned=owned, now=now, order=order)


def allocate(plan: Plan, land_steps: dict[int, int], previous: dict | None = None, *, owned: int = 1,
             now: int = 0) -> dict[tuple[int, int], tuple[int, int]]:
    """Animal-first visit order. It can reserve a site for a later animal and leave an earlier
    crop unplaced (25 of 26); allocate_sites falls back to birth order then."""
    return _allocate(plan, land_steps, previous, owned=owned, now=now, order=plan.instances())


def _allocate(plan: Plan, land_steps: dict[int, int], previous: dict | None, *, owned: int, now: int,
              order) -> dict[tuple[int, int], tuple[int, int]]:
    """Tile per cohort instance. Instances whose window has started keep their tile (a
    physical asset); unborn instances are placed afresh, in `order`, on the nearest free tile
    of a quadrant owned by their birth (free: every other occupant's window is disjoint)."""
    previous = dict(previous or {})
    windows = {}
    for module, index in plan.instances():
        windows[(module, index)] = window(module, plan.regime(module))
    allocation = {k: v for k, v in previous.items() if k in windows and windows[k][0] <= now}
    available_from = {j: 0 for j in range(max(1, owned))}
    for j, step in land_steps.items():
        available_from.setdefault(j, step)
    for j in range(1, 4):
        available_from.setdefault(j, None)
    tiles = coords(4)
    unplaced = []
    for key in order:
        if key in allocation:
            continue
        start, until = windows[key]
        chosen = None
        for tile in tiles:
            q = quadrant_index(tile)
            if available_from.get(q) is None or available_from[q] > start:
                continue
            busy = False
            for other, other_tile in allocation.items():
                if other_tile != tile:
                    continue
                o_start, o_until = windows[other]
                if not (_end(o_until) < start or _end(until) < o_start):
                    busy = True
                    break
            if not busy:
                chosen = tile
                break
        if chosen is None:
            unplaced.append(key)  # no owned site by its birth: the instance waits (a blocker, not a crash)
            continue
        allocation[key] = chosen
    allocation_unplaced[id(allocation)] = unplaced
    return allocation


allocation_unplaced: dict[int, list] = {}


def unplaced(allocation: dict) -> list:
    """Instances the last `allocate` call could not place (empty when every instance has a tile)."""
    return list(allocation_unplaced.get(id(allocation), []))


def production_command(module: int, regime: int, tile: tuple[int, int], *, replace: bool = False) -> dict:
    """The stage of one instance; REPLACE when another instance's asset may still stand on the tile."""
    m = V.MODULES[module]
    start, until = window(module, regime)
    row = V.tape(module, regime)
    scope = {"kind": "TILE", "x": tile[0], "y": tile[1]}
    if m["animal"]:
        c = {"op": "SET_ANIMAL", "scope": scope, "animal": m["producer"], "feed": "DAILY", "care": "DAILY",
             "harvest_threshold": 1, "manure": "WHEN_AVAILABLE", "start": start, "until": until, "priority": 3}
    else:
        name = V.REGIMES[regime]
        threshold = max(row["primary"]) if not name.endswith("FIRST_ELIGIBLE") and m["default_regime"] == "held_cap" else 1
        c = {"op": "SET_CROP", "scope": scope, "crop": m["producer"], "cycles": 1 if name.startswith("ONE_CYCLE") else None,
             "until": until, "start": start,
             "harvest_threshold": max(1, threshold), "watering": "DAILY",
             "fertilizing": "WHEN_EXPIRED" if m["policy"] == "watered_fertilized" else "OFF", "priority": 4}
    return {"op": "REPLACE", "scope": scope, "replacement": c} if replace else c


def hands_commands(proj: Projection, *, mode: str = "daily") -> list[dict]:
    today = proj.state.day
    counts = [(V.HAND_CAP if h is None else h) for h in proj.hands]
    if mode == "constant":
        peak = max(counts[today:], default=0)
        return [{"op": "HANDS", "target": peak, "start": 24 * today, "until": GAME_END, "priority": 0}]
    out, d = [], today
    while d < V.SEASON_DAYS:
        e = d
        while e + 1 < V.SEASON_DAYS and counts[e + 1] == counts[d]:
            e += 1
        out.append({"op": "HANDS", "target": counts[d], "start": 24 * d,
                    "until": GAME_END if e == V.SEASON_DAYS - 1 else 24 * e + 23, "priority": 0})
        d = e + 1
    return out


def land_commands(proj: Projection) -> list[dict]:
    # Backdated starts apply to the remaining interval in FullSeasonPlan. Do not
    # rewrite a bound authorization date merely because execution was delayed.
    steps = sorted((step, j) for j, step in proj.land_steps.items())
    out = []
    for i, (step, j) in enumerate(steps):
        until = GAME_END if i + 1 == len(steps) else max(step, steps[i + 1][0] - 1)
        out.append({"op": "LAND", "target": j, "start": step, "until": until, "priority": 2})
    return out


def sale_commands(state: State, proj: Projection, product: int, epoch_end: int) -> list[dict]:
    """Stages for one product's overlay from this epoch until the next (or the horizon)."""
    name = V.PRODUCTS[product]
    primitive, horizon, rank = proj.plan.overlay(product)
    kind = V.PRIMITIVES[primitive]
    base = {"op": "SALE", "product": name, "price_above": 0.0, "reserve": 0, "priority": 0}
    if kind == "CLEAR":
        return [{**base, "quantity": 1000, "start": 0 if state.step == 0 else state.step, "until": GAME_END}]
    if kind == "HOLD":
        # A kept HOLD releases at the event bound when it was set, not at a horizon re-resolved
        # each dawn, which would move a NEXT_DAWN sale a day later at every publication.
        release = hold_release(state, proj.plan, getattr(proj, "hold_releases", None), product)
        return [{**base, "quantity": 1000, "start": state.step if release is None else release, "until": GAME_END}]
    h = state.horizon_step(product, horizon)
    if h is None:
        return [{**base, "quantity": 1000, "start": state.step, "until": GAME_END}]
    points = frontier(state, proj, product, primitive, horizon)
    if rank >= len(points):
        return [{**base, "quantity": 1000, "start": state.step, "until": GAME_END}]
    q = points[rank]["q"] or 0
    if kind == "SELL":
        out = [{**base, "quantity": q, "start": h, "until": h}] if q else []
        later = state.next_demand(name, h) or min(h + 1, V.LAST_STEP)
        if later > h:
            out.append({**base, "quantity": 1000, "start": later, "until": GAME_END})
        return out
    if kind == "CARRY" and q:
        return [{"op": "BUY", "kind": "PRODUCT", "item": name, "quantity": q, "price_below": 1000000.0,
                 "start": h, "until": h, "priority": 1},
                {**base, "quantity": 1000, "start": h + 1, "until": GAME_END}]
    return [{**base, "quantity": 1000, "start": state.step, "until": GAME_END}]


def compile_epoch(state: State, proj: Projection, allocation: dict, *, hands_mode: str = "daily",
                  fresh: bool = True, dropped: list | None = None, replace_tiles: set | None = None) -> list[dict]:
    """The commands published at one epoch.

    A later epoch first retires every rule stage (HANDS, LAND, SALE, BUY) and the production
    stages of dropped tiles, then re-emits the plan's stages; production stages of kept
    instances are re-applied identically, which the executor treats as no change.
    """
    epoch_end = min(V.LAST_STEP, 24 * (state.day + 1) - 1)
    commands = []
    if not fresh:
        commands += [{"op": "CANCEL", "target": {"kind": kind}} for kind in ("HANDS", "LAND")]
        commands += [{"op": "CANCEL", "target": {"kind": "SALE", "product": p}} for p in V.PRODUCTS]
        commands += [{"op": "CANCEL", "target": {"kind": "BUY", "stock_kind": "PRODUCT", "item": p}} for p in ("WHEAT", "FERTILIZER")]
    for tile in dropped or []:
        commands.append({"op": "CANCEL", "target": {"kind": "PRODUCTION", "scope": {"kind": "TILE", "x": tile[0], "y": tile[1]}}})
    commands += hands_commands(proj, mode=hands_mode) + land_commands(proj)
    for key in proj.plan.instances():
        module, _ = key
        if key not in allocation:
            continue
        tile = allocation[key]
        commands.append(production_command(module, proj.plan.regime(module), tile,
                                           replace=tile in (replace_tiles or set()) and not V.MODULES[module]["animal"]))
    for product in range(len(V.PRODUCTS)):
        commands += sale_commands(state, proj, product, epoch_end)
    commands = [c for c in commands if "until" not in c or _end(c["until"]) >= state.step]
    commands.append({"op": "STOP"})
    if len(commands) > COMMAND_BUDGET:
        raise ValueError(f"compiled program has {len(commands)} commands, budget {COMMAND_BUDGET}")
    return validate_program(commands)


def compile_opening(plan: Plan, observation: dict, *, hands_mode: str = "daily", **kw) -> tuple[list[dict], Projection, dict]:
    """The whole-season program of a plan at the opening observation."""
    state = State(observation)
    proj = Projection(state, plan, **kw)
    allocation = allocate_sites(plan, proj.land_steps, owned=state.quadrants)
    return compile_epoch(state, proj, allocation, hands_mode=hands_mode), proj, allocation


def livestock_service_commands(control, plan: Plan, observation: dict, allocation: dict,
                               started: dict, *, game_id: str) -> list[dict]:
    """Lower one explicitly bound service choice; never select a mode from economics.

    This independently usable compiler slice is NOT wired into compile_epoch or
    the old V0 action/Plan/continuation schemas. A caller must retain the separate
    control and publish the commands at a dawn through FullSeasonPlan, whose
    immutable elapsed prefix prevents retroactive service/credit changes.
    Existing stage start, harvest threshold, manure intent and priority remain
    exactly as compiled. Reduced feed is NOT preserving retirement: OFF animals
    can escape. WHEN_UNFED and DAILY currently schedule identical executor jobs.
    The requested preserving zero-service retirement fails closed, never emits
    CANCEL/CLEAR/SALE/HARVEST, and cannot be silently replaced by maintenance feed.
    """
    from farm_neat.economy.v0.plan import BornLivestockControl, LIVESTOCK_RETIRE, LIVESTOCK_SERVICE_MODES

    if not isinstance(control, BornLivestockControl):
        raise ValueError("LIVESTOCK_CONTROL_TYPED_VALUE_REQUIRED")
    control.validate_context(plan, observation, allocation, started, game_id=game_id)
    if observation["step"] % 24:
        raise ValueError("LIVESTOCK_CONTROL_PUBLICATION_DAWN")
    if control.mode == LIVESTOCK_RETIRE:
        raise ValueError("LIVESTOCK_RETIRE_PRESERVATION_UNSUPPORTED")
    feed, care = next((feed, care) for mode, feed, care in LIVESTOCK_SERVICE_MODES if mode == control.mode)
    commands = []
    for _index, x, y, _placed_day in control.instances:
        command = production_command(control.module, plan.regime(control.module), (x, y))
        commands.append({**command, "feed": feed, "care": care})
    return validate_program(commands + [{"op": "STOP"}])[:-1]
