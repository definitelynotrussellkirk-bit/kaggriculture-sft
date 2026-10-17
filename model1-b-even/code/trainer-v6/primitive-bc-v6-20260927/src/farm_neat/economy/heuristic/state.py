"""Observation wrapper.

The raw observation is nested dicts with two traps worth isolating here:
  * tiles are indexed tiles[y][x] while positions are [x, y];
  * `farms` is public (both players), `private` is ours only.
"""

from .constants import BOARD_SIZE, shed_access_tiles


class View:
    """Read-only helper over one turn's observation."""

    # Engine 1.32.7 permits movement across every in-bounds ground cell, including locked
    # quadrants.  The route planner uses this explicit invariant to reuse exact rectangular-board
    # shortest paths; synthetic/custom views without the marker retain the generic live BFS.
    all_board_cells_walkable = True

    def __init__(self, obs):
        self.obs = obs
        self.player = obs.get("player", 0)
        self.step = obs.get("step", 0)
        self.day = obs.get("day", 0)
        self.hour = obs.get("hour", 0)
        self.farms = obs.get("farms", []) or []
        self.farm = self.farms[self.player] if self.player < len(self.farms) else {}
        self.opponent = self.farms[1 - self.player] if len(self.farms) > 1 else {}
        self.private = obs.get("private", {}) or {}
        self.shed = self.private.get("shed", {}) or {}
        self.seeds = self.private.get("seeds", {}) or {}
        self.inventories = self.private.get("inventories", [{}]) or [{}]
        self.market = obs.get("market", {}) or {}
        self.prices = self.market.get("prices", {}) or {}
        self.market_inventory = self.market.get("inventory", {}) or {}
        self.town = obs.get("town", {}) or {}
        self.tiles = self.farm.get("tiles", [])
        self.board = len(self.tiles) or BOARD_SIZE
        self._tile_groups_cache = None

    # -- units ---------------------------------------------------------------
    @property
    def money(self):
        return self.farm.get("money", 0)

    @property
    def n_units(self):
        return 1 + len(self.farm.get("hands", []))

    def unit_pos(self, idx):
        if idx == 0:
            return tuple(self.farm.get("farmer", (0, 0)))
        hands = self.farm.get("hands", [])
        return tuple(hands[idx - 1]) if idx - 1 < len(hands) else None

    def unit_inv(self, idx):
        return self.inventories[idx] if idx < len(self.inventories) else {}

    # -- tiles ---------------------------------------------------------------
    def tile(self, x, y):
        if 0 <= y < len(self.tiles) and 0 <= x < len(self.tiles[y]):
            return self.tiles[y][x]
        return "LOCKED"

    def walkable(self, x, y):
        # Since kaggle-environments 1.32.3, locked ground is passable. It still rejects
        # tile-mutating actions; only movement and virtual-shed operations work there.
        return 0 <= x < self.board and 0 <= y < self.board

    def iter_tiles(self):
        for y, row in enumerate(self.tiles):
            for x, t in enumerate(row):
                yield x, y, t

    def _tile_groups(self):
        """Classify the read-only turn snapshot once for repeated planner queries."""

        if self._tile_groups_cache is None:
            empty = []
            weeds = []
            plants = []
            animals = []
            empty_structures = []
            for x, y, tile in self.iter_tiles():
                if tile is None:
                    empty.append((x, y))
                    continue
                if not isinstance(tile, dict):
                    continue
                kind = tile.get("kind")
                if kind == "WEED":
                    weeds.append((x, y))
                elif kind == "PLANT":
                    plants.append((x, y, tile))
                if "animal" in tile:
                    animals.append((x, y, tile))
                elif kind in ("COOP", "PASTURE"):
                    empty_structures.append((x, y, tile))
            self._tile_groups_cache = (
                empty,
                weeds,
                plants,
                animals,
                empty_structures,
            )
        return self._tile_groups_cache

    def empty_tiles(self):
        return list(self._tile_groups()[0])

    def weeds(self):
        return list(self._tile_groups()[1])

    def plants(self):
        return list(self._tile_groups()[2])

    def animals(self):
        return list(self._tile_groups()[3])

    def empty_structures(self):
        return list(self._tile_groups()[4])

    def shed_used(self):
        return sum(self.shed.values())


def step_toward(src, dst, view):
    """One movement op from src toward dst, staying within the board."""
    sx, sy = src
    dx, dy = dst[0] - sx, dst[1] - sy
    options = []
    if abs(dx) >= abs(dy):
        if dx:
            options.append(("EAST", (sx + 1, sy)) if dx > 0 else ("WEST", (sx - 1, sy)))
        if dy:
            options.append(("SOUTH", (sx, sy + 1)) if dy > 0 else ("NORTH", (sx, sy - 1)))
    else:
        if dy:
            options.append(("SOUTH", (sx, sy + 1)) if dy > 0 else ("NORTH", (sx, sy - 1)))
        if dx:
            options.append(("EAST", (sx + 1, sy)) if dx > 0 else ("WEST", (sx - 1, sy)))
    for op, (nx, ny) in options:
        if view.walkable(nx, ny):
            return op
    return None


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_shed_tile(pos, view):
    reachable = [t for t in shed_access_tiles(view.board) if view.walkable(*t)]
    if not reachable:
        return None
    return min(reachable, key=lambda t: manhattan(pos, t))
