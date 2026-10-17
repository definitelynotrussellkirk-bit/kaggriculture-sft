"""Space-time lattice input (pbc-v6-design/DESIGN.md item 7): both 10x10 farms as a grid, per-cell channels at t, t-1, t-2.

compute_lattice(actor_obs, configuration, history=()) -> np.ndarray [2, 10, 10, WIDTH] float16
  farm axis: 0 = own farm (obs["farms"][player]), 1 = opponent farm (obs["farms"][1 - player]);
  then y (row, 0..9), x (column, 0..9), as in the observation's tiles[y][x];
  last axis: time folded into channels, slot-major: index = slot * C + c, slot 0 = t, 1 = t-1, 2 = t-2,
  c = SLOT_CHANNELS index (C = len(SLOT_CHANNELS)); CHANNELS names every index as "<slot>/<channel>".
  history = previous actor observations of the same seat, most recent first (at most 2 are used). An entry fills slot k only
  when its step is exactly t - k and its player is the same seat; otherwise that slot stays all zeros (its 'present' = 0).

Grid -> TILE token (the rule of model.py PolicyNet._tiles and flip.token_features):
  lattice[f, y, x] describes the TILE token whose id is "tile/{f}/{x},{y}" (observation.pack; role f: 0 own, 1 opponent),
  i.e. model cell = f * 100 + y * 10 + x, located by _tiles from the row's ROLE_COL + 1 (13) flag and the x/9, y/9 columns
  14/15. Hence lattice.reshape(2 * 100, WIDTH)[cell] is that token's cell row; tile_cells(token_ids) gives the cell of
  every packed token (-1 for non-TILE tokens), and lattice_net.scatter_to_tokens applies the same map to the CNN output.

Visibility: only the actor observation and the seat's own past observations. Carried units exist only for own workers
(private inventories); the opponent farm's carried channels are always 0 (unknown, never estimated). Opponent seeds are
unknown, so every empty opponent tile gets job PLANT_EMPTY, exactly as flip.farm_view(..., seeds_known=False).

Job kind uses flip.py's priority code numbering. V8 corrects terminal-day priorities:
no fatal water/feed or future-only care without a remaining dusk; immediate one-shot
water gains, harvests and fertilizer collection remain possible. Deliberately not
identical to legacy flip on day 29. No forecast loop is used.

Training: episode_lattices(replay, seat) computes each step's frame once and stacks t-1, t-2 from the same replay. Serving:
the agent keeps its last two actor observations (reset at step 0 / a new episode) and passes them as history; or it keeps
the last two frames and calls stack_frames directly (identical output).
"""
from __future__ import annotations

import math

import numpy as np

from farm_neat.economy.heuristic.constants import ANIMALS, CROPS

BOARD = 10
DAY = 24
LAST_STEP = 719
SLOTS = ("t", "t-1", "t-2")
LAGS = (0, 1, 2)
TILE_KINDS = ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE")
ITEMS = tuple(CROPS) + tuple(ANIMALS)            # WHEAT CARROT TOMATO STRAWBERRY MELON GOOSE COW SHEEP
# flip.py job kind codes (same numbering; NONE = 0 has no channel)
NONE, WATER_TONIGHT, FEED_TONIGHT, HARVEST_DECAYING, WATER_TODAY, FEED_TODAY, HARVEST, COLLECT, FERTILIZER, \
    PLANT_EMPTY, WEED, CARE = range(12)
JOB_NAMES = ("WATER_TONIGHT", "FEED_TONIGHT", "HARVEST_DECAYING", "WATER_TODAY", "FEED_TODAY", "HARVEST", "COLLECT",
             "FERTILIZER", "PLANT_EMPTY", "WEED", "CARE")
SHED = ((4, 4), (5, 4), (4, 5), (5, 5))           # shed-access tiles (x, y)

SLOT_CHANNELS = (
    *("kind:" + k for k in TILE_KINDS),           # 0-5   tile kind one-hot (EMPTY = None tile, LOCKED = "LOCKED")
    *("item:" + i for i in ITEMS),                # 6-13  crop (PLANT) or animal (COOP/PASTURE) one-hot
    "watered_today",                              # 14    plant watered_today
    "fed_today",                                  # 15    animal fed_today
    "cared_today",                                # 16    animal cared_today
    "days_unwatered",                             # 17    plant consecutive_unwatered / 2 (2 = becomes WEED at dusk)
    "days_unfed",                                 # 18    animal consecutive_unfed / 2 (2 = escapes at dusk)
    "units_held",                                 # 19    yield_units / 6 (max_yield / max_held <= 6)
    "fertilizer_available",                       # 20    animal fertilizer_available
    "workers",                                    # 21    log1p(workers standing here) / log1p(25) (own on farm 0, rival on 1)
    "farmer",                                     # 22    1 if that farm's farmer stands here
    "carried_units",                              # 23    own farm only: log1p(sum of inventories of workers here) / log1p(100)
    "carried_wheat",                              # 24    own farm only: log1p(WHEAT carried by workers here) / log1p(100)
    *("job:" + j for j in JOB_NAMES),             # 25-35 flip job kind one-hot (NONE = all zero)
    "shed_access",                                # 36    static: one of the four shed-access tiles
    "shed_dist",                                  # 37    static: Manhattan distance to the nearest shed-access tile / 9
    "present",                                    # 38    1 on every cell when this time slot has an observation
    "same_day",                                   # 39    1 when this slot's day == day(t) (no dusk respawn in between)
)
C = len(SLOT_CHANNELS)
CHANNELS = tuple(f"{s}/{c}" for s in SLOTS for c in SLOT_CHANNELS)
WIDTH = len(CHANNELS)
COL = {c: i for i, c in enumerate(SLOT_CHANNELS)}
_KIND = {k: COL["kind:" + k] for k in TILE_KINDS}
_ITEM = {i: COL["item:" + i] for i in ITEMS}
_JOB0 = COL["job:WATER_TONIGHT"] - 1               # channel of job code k (k >= 1) = _JOB0 + k
_LOG25 = math.log1p(25)
_LOG100 = math.log1p(100)

_STATIC = np.zeros((BOARD, BOARD, C), np.float32)
for _y in range(BOARD):
    for _x in range(BOARD):
        _STATIC[_y, _x, COL["shed_access"]] = float((_x, _y) in SHED)
        _STATIC[_y, _x, COL["shed_dist"]] = min(abs(_x - sx) + abs(_y - sy) for sx, sy in SHED) / 9.0
        _STATIC[_y, _x, COL["present"]] = 1.0


def obs_step(obs):
    return int(obs["step"]) if "step" in obs else int(obs["day"]) * DAY + int(obs["hour"])


def job_kind(tile, t, own, has_seeds):
    """V8 terminal-aware priority (legacy code numbers; rival seeds remain unknown)."""
    if t >= LAST_STEP:
        return NONE  # observation 719 is terminal; no action remains
    has_dusk = (t // DAY + 1) * DAY - 1 < LAST_STEP
    if tile is None:
        return PLANT_EMPTY if (has_seeds or not own) else NONE
    if not isinstance(tile, dict):
        return NONE                                                     # "LOCKED"
    kind = tile.get("kind")
    if kind == "PLANT":
        cd = CROPS[tile["crop"]]
        w, c, y = bool(tile["watered_today"]), tile["consecutive_unwatered"], tile["yield_units"]
        if has_dusk and not w and c >= 1:
            return WATER_TONIGHT
        left = DAY - t % DAY
        ready = max(0, (tile["planted_day"] + cd["first_yield_day"]) * DAY - t) == 0 and y > 0
        if ready:
            mls = tile["max_lifespan_step"]
            if mls >= 0:
                s = max(t, mls)
                s += (s - mls) % 2                                      # first decay tick at or after t
                if s <= min(t + left - 1, LAST_STEP - 1):              # flip: decay ETA s + 1 - t <= left
                    return HARVEST_DECAYING
        if has_dusk and not w and c == 0:
            return WATER_TODAY
        if ready:
            return HARVEST
        if not cd["ongoing"] and not w:
            ws, age = (cd["max_yield_day"] + 1) // 2, t // DAY - tile["planted_day"]
            if ws <= age <= cd["max_yield_day"]:
                return WATER_TODAY
        return NONE
    if "animal" in tile:
        f, c = bool(tile["fed_today"]), tile["consecutive_unfed"]
        if has_dusk and not f and c >= 1:
            return FEED_TONIGHT
        if has_dusk and not f and c == 0:
            return FEED_TODAY
        if tile["yield_units"] > 0:
            return COLLECT
        if tile.get("fertilizer_available"):
            return FERTILIZER
        if has_dusk and not tile.get("cared_today"):
            return CARE
        return NONE
    if kind == "WEED":
        return WEED
    return NONE


def _farm_frame(out, farm, t, own, has_seeds, inventories):
    """Fill out [10, 10, C] (float32, pre-filled with the static channels) for one farm."""
    for y, line in enumerate(farm["tiles"]):
        for x, tile in enumerate(line):
            cell = out[y, x]
            if tile is None:
                cell[_KIND["EMPTY"]] = 1.0
            elif not isinstance(tile, dict):
                cell[_KIND["LOCKED"]] = 1.0
                continue
            else:
                kind = tile.get("kind")
                if kind in _KIND:
                    cell[_KIND[kind]] = 1.0
                if kind == "PLANT":
                    cell[_ITEM[tile["crop"]]] = 1.0
                    cell[COL["watered_today"]] = float(bool(tile.get("watered_today")))
                    cell[COL["days_unwatered"]] = min(max(tile.get("consecutive_unwatered") or 0, 0), 2) / 2.0
                    cell[COL["units_held"]] = min(max(tile.get("yield_units") or 0, 0), 6) / 6.0
                elif tile.get("animal") in _ITEM:
                    cell[_ITEM[tile["animal"]]] = 1.0
                    cell[COL["fed_today"]] = float(bool(tile.get("fed_today")))
                    cell[COL["cared_today"]] = float(bool(tile.get("cared_today")))
                    cell[COL["days_unfed"]] = min(max(tile.get("consecutive_unfed") or 0, 0), 2) / 2.0
                    cell[COL["units_held"]] = min(max(tile.get("yield_units") or 0, 0), 6) / 6.0
                    cell[COL["fertilizer_available"]] = float(bool(tile.get("fertilizer_available")))
            k = job_kind(tile, t, own, has_seeds)
            if k:
                cell[_JOB0 + k] = 1.0
    units = [farm["farmer"], *farm["hands"]]
    count = {}
    carried, wheat = {}, {}
    for i, (x, y) in enumerate(units):
        count[(x, y)] = count.get((x, y), 0) + 1
        if own and i < len(inventories) and isinstance(inventories[i], dict):
            inv = inventories[i]
            carried[(x, y)] = carried.get((x, y), 0) + sum(max(0, int(v)) for v in inv.values())
            wheat[(x, y)] = wheat.get((x, y), 0) + max(0, int(inv.get("WHEAT", 0)))
    for (x, y), n in count.items():
        out[y, x, COL["workers"]] = min(1.0, math.log1p(n) / _LOG25)
    fx, fy = units[0]
    out[fy, fx, COL["farmer"]] = 1.0
    for (x, y), n in carried.items():
        out[y, x, COL["carried_units"]] = min(2.0, math.log1p(n) / _LOG100)
    for (x, y), n in wheat.items():
        out[y, x, COL["carried_wheat"]] = min(2.0, math.log1p(n) / _LOG100)


def compute_frame(actor_obs, configuration=None):
    """One time slot [2, 10, 10, C] float16 ('present' = 1; 'same_day' = 0 here, set by stack_frames)."""
    t = obs_step(actor_obs)
    seat = int(actor_obs["player"])
    private = actor_obs.get("private") or {}
    has_seeds = any(v > 0 for v in (private.get("seeds") or {}).values())
    inventories = private.get("inventories") or []
    frame = np.empty((2, BOARD, BOARD, C), np.float32)
    frame[:] = _STATIC
    farms = actor_obs["farms"]
    _farm_frame(frame[0], farms[seat], t, True, has_seeds, inventories)
    _farm_frame(frame[1], farms[1 - seat], t, False, True, ())
    return frame.astype(np.float16)


_SAME_DAY = COL["same_day"]


def stack_frames(t, frames):
    """frames[k] = (step, frame) of slot k (lag k) or None -> [2, 10, 10, WIDTH] float16; a slot whose step is not t - k
    stays all zeros."""
    out = np.zeros((2, BOARD, BOARD, WIDTH), np.float16)
    for k, item in enumerate(frames[:len(LAGS)]):
        if item is None or item[1] is None or int(item[0]) != t - k:
            continue
        out[..., k * C:(k + 1) * C] = item[1]
        out[..., k * C + _SAME_DAY] = float((t - k) // DAY == t // DAY)
    return out


def compute_lattice(actor_obs, configuration=None, history=()):
    """[2, 10, 10, WIDTH] float16 lattice of the current actor observation and up to two previous own observations."""
    t = obs_step(actor_obs)
    seat = int(actor_obs["player"])
    slots = [(t, compute_frame(actor_obs, configuration)), None, None]
    for h in tuple(history or ())[:2]:
        if not isinstance(h, dict) or int(h.get("player", -1)) != seat:
            continue
        k = t - obs_step(h)
        if k in (1, 2) and slots[k] is None:
            slots[k] = (t - k, compute_frame(h, configuration))
    return stack_frames(t, slots)


def episode_lattices(replay, seat, steps=None):
    """Training path: yield (t, lattice) for boundaries t (default 0..718) of one perspective; each frame is computed once
    and t-1, t-2 come from the same replay (bc.public_observation, the actor-visible view)."""
    from farm_neat.training import bc
    cache = {}
    for t in (range(LAST_STEP) if steps is None else steps):
        for s in (t - 2, t - 1, t):
            if s >= 0 and s not in cache:
                cache[s] = compute_frame(bc.public_observation(replay, s, seat))
        for s in [s for s in cache if s < t - 2]:
            del cache[s]
        yield t, stack_frames(t, [(t - k, cache.get(t - k)) if t - k >= 0 else None for k in LAGS])


def tile_cells(token_ids):
    """[N] int64: model cell (farm * 100 + y * 10 + x) of each packed TILE token id "tile/{farm}/{x},{y}", -1 elsewhere."""
    out = np.full(len(token_ids), -1, np.int64)
    for n, tid in enumerate(token_ids):
        if tid.startswith("tile/"):
            role, xy = tid[5:].split("/")
            x, y = (int(v) for v in xy.split(","))
            out[n] = int(role) * 100 + y * BOARD + x
    return out
