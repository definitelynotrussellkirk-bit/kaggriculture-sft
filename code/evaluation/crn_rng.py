"""Opt-in common-random-numbers (CRN) dusk for the pinned Kaggriculture engine.

The pinned engine seeds ONE generator per dusk, ``random.Random((seed * 1_000_003) ^ day)``, draws a weed
uniform for every empty tile of farm 0 then farm 1 in scan order, and then draws the shop unlock from the
same generator. Two action streams on one seed therefore get different shops as soon as their empty-tile
counts differ, and a tile's weed outcome depends on how many empty tiles precede it.

Configuration key ``crnRng: true`` selects the CRN variant. Only the two random draws change:

* weed: tile (x, y) of farm ``farm`` (seat index) at dusk ``day`` becomes WEED iff it is empty and
  ``weed_uniform(seed, day, farm, x, y) < weedSpawnChance``;
* shop: when the unchanged schedule unlocks a shop at dusk ``day``, it is
  ``sorted(SHOPS)[shop_index(seed, day, len(SHOPS))]``.

Each value comes from BLAKE2b-64 of the ASCII key ``kaggriculture-crn-v1|weed|{seed}|{day}|{farm}|{x}|{y}``
or ``kaggriculture-crn-v1|shop|{seed}|{day}`` (standard library; stable across processes, platforms and
Python versions; never the salted built-in ``hash``). Weed: top 53 bits / 2**53. Shop: value mod n (exact for
the 8 shops). Everything else in the dusk -- plant/animal refresh, shed drop, farmer/hand reset, the unlock
schedule and its 8-instance cap -- is the reference code.

Who reads the key: ``environment()`` / ``set_crn()`` / ``CrnKaggricultureEnvironment`` here. The pinned files
(src/farm_neat/game/kaggriculture_engine_reference.py, exact_engine.py) are not modified -- their SHA-256 is bound
by source closures, and this module sits outside src/farm_neat/game so the V0 identities that glob that folder are
unchanged -- so a plain ``ExactKaggricultureEnvironment`` IGNORES ``crnRng``; build CRN games through this module.
The CRN engine is a private, freshly loaded copy of the pinned engine with ``_end_of_day`` dispatching on the
flag; with the flag absent it runs the reference code. Actors see the key in the configuration (our deploy
agents whitelist configuration keys). Games played with the flag are a rules variant with the same per-tile
weed and per-unlock shop marginals; they are not records of the original engine.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import operator
import sys
import types
from pathlib import Path

CONFIG_KEY = "crnRng"
KEY_VERSION = "kaggriculture-crn-v1"
_TWO_POW_53 = 9007199254740992.0


# ------------------------------------------------------------------------------------------ keyed draws
def enabled(configuration) -> bool:
    """True iff the configuration declares ``crnRng: true`` (bool or 0/1 only; anything else is refused)."""
    if configuration is None:
        return False
    if isinstance(configuration, dict):
        value = configuration.get(CONFIG_KEY, False)
    else:
        value = getattr(configuration, CONFIG_KEY, False)
    if value is None or value is False:
        return False
    if value is True:
        return True
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(f"configuration {CONFIG_KEY} must be true or false, got {value!r}")


def with_flag(configuration=None, on: bool = True) -> dict:
    """Copy of ``configuration`` with ``crnRng: true`` when ``on``; without the key otherwise."""
    out = {k: v for k, v in dict(configuration or {}).items() if k != CONFIG_KEY}
    if on:
        out[CONFIG_KEY] = True
    return out


def _u64(*parts) -> int:
    key = "|".join([KEY_VERSION] + [str(p) for p in parts]).encode("ascii")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def weed_uniform(seed, day, farm, x, y) -> float:
    """Uniform [0, 1) for one tile at one dusk; depends on nothing else."""
    seed, day, farm, x, y = (operator.index(v) for v in (seed, day, farm, x, y))
    return (_u64("weed", seed, day, farm, x, y) >> 11) / _TWO_POW_53


def shop_index(seed, day, n) -> int:
    """Index into sorted(SHOPS) for the unlock at dusk ``day``; depends on nothing else."""
    seed, day, n = (operator.index(v) for v in (seed, day, n))
    if n <= 0:
        raise ValueError("no shops to draw from")
    return _u64("shop", seed, day) % n


def shop_sequence(seed, shops, *, interval=3, max_instances=8, days=30):
    """Every shop the CRN variant unlocks for ``seed`` on the engine's schedule, as (dusk day, shop)."""
    names = sorted(shops)
    out = []
    for day in range(days):
        if (day + 1) % max(1, int(interval)) == 0 and len(out) < max_instances:
            out.append((day, names[shop_index(seed, day, len(names))]))
    return out


class TileKeyedWeeds:
    """Stands in for the dusk generator passed to ``_spawn_weeds`` under CRN: one uniform per tile key."""

    __slots__ = ("seed", "day", "farm")

    def __init__(self, seed, day, farm):
        self.seed, self.day, self.farm = seed, day, farm

    def tile_uniform(self, x, y):
        return weed_uniform(self.seed, self.day, self.farm, x, y)

    def random(self):  # a sequential draw would reintroduce order dependence
        raise TypeError("CRN weed draws are keyed by tile; use tile_uniform(x, y)")


# ------------------------------------------------------------------------------------------ engine level
# The three functions below are re-bound into an engine namespace by install(); their free names (get, SHOPS,
# _daily_refresh_plants, ...) resolve in that namespace at call time, so hooks installed later on the same
# namespace (e.g. evaluation instruments) still see every call.


def _crn_spawn_weeds(farm, board_size, weed_chance, rng):
    tile_uniform = getattr(rng, "tile_uniform", None)
    if tile_uniform is None:  # the reference generator: reference behaviour, draw for draw
        return _reference_spawn_weeds(farm, board_size, weed_chance, rng)  # noqa: F821
    for y in range(board_size):
        for x in range(board_size):
            if farm["tiles"][y][x] is None and tile_uniform(x, y) < weed_chance:
                farm["tiles"][y][x] = {"kind": "WEED"}


def _crn_end_of_day(state, env, day):
    # The reference _end_of_day with the two draws keyed; statement order is unchanged.
    obs0 = state[0].observation
    cfg = env.configuration
    board_size = int(get(cfg, "boardSize", 10))  # noqa: F821
    turns_per_day = max(1, int(get(cfg, "turnsPerDay", 24)))  # noqa: F821
    weed_chance = float(get(cfg, "weedSpawnChance", 0.005))  # noqa: F821
    shed_cap = int(get(cfg, "shedCapacity", 100))  # noqa: F821
    shop_interval = max(1, int(get(cfg, "townShopUnlockInterval", 3)))  # noqa: F821

    seed = env.info.get("seed", 0)

    for player_id, farm in enumerate(obs0.farms):
        private = state[player_id].observation.private
        _daily_refresh_plants(farm, day, turns_per_day)  # noqa: F821
        _daily_refresh_animals(farm, day)  # noqa: F821
        _spawn_weeds(farm, board_size, weed_chance, _crn_TileKeyedWeeds(seed, day, player_id))  # noqa: F821
        _drop_inventories_to_shed(private, shed_cap)  # noqa: F821
        farm["farmer"] = list(_default_spawn(board_size))  # noqa: F821
        farm["hands"] = []
        farm["hires_today"] = 0
        private["inventories"] = [{}]

    next_day = day + 1
    town = obs0.town
    if next_day > 0 and next_day % shop_interval == 0:
        if len(town["unlocked_shops"]) < MAX_SHOP_INSTANCES:  # noqa: F821
            names = sorted(SHOPS)  # noqa: F821
            town["unlocked_shops"].append(names[_crn_shop_index(seed, day, len(names))])  # noqa: F821


def _crn_dispatch_end_of_day(state, env, day):
    if _crn_enabled(env.configuration):  # noqa: F821
        return _crn_end_of_day(state, env, day)  # noqa: F821
    return _reference_end_of_day(state, env, day)  # noqa: F821


def _rebind(function, namespace):
    bound = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__,
                               function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__doc__ = function.__doc__
    return bound


def install(namespace: dict) -> dict:
    """Make an engine namespace CRN-capable in place (idempotent). Only ever call it on a private copy."""
    if namespace.get("_crn_installed"):
        return namespace
    for name in ("_end_of_day", "_spawn_weeds", "get", "SHOPS", "MAX_SHOP_INSTANCES", "_default_spawn",
                 "_daily_refresh_plants", "_daily_refresh_animals", "_drop_inventories_to_shed"):
        if name not in namespace:
            raise RuntimeError(f"engine namespace has no {name}; not the Kaggriculture interpreter")
    namespace["_reference_end_of_day"] = namespace["_end_of_day"]
    namespace["_reference_spawn_weeds"] = namespace["_spawn_weeds"]
    namespace["_crn_enabled"] = enabled
    namespace["_crn_shop_index"] = shop_index
    namespace["_crn_TileKeyedWeeds"] = TileKeyedWeeds
    namespace["_crn_end_of_day"] = _rebind(_crn_end_of_day, namespace)
    namespace["_spawn_weeds"] = _rebind(_crn_spawn_weeds, namespace)
    namespace["_end_of_day"] = _rebind(_crn_dispatch_end_of_day, namespace)
    namespace["_crn_installed"] = True
    namespace["_crn_key_version"] = KEY_VERSION
    return namespace


def private_copy(engine_module, name=None):
    """A module sharing the engine's values whose functions are re-bound to its own namespace."""
    name = name or f"{engine_module.__name__}_private"
    module = types.ModuleType(name)
    namespace = module.__dict__
    namespace.update(engine_module.__dict__)
    namespace["__name__"] = name
    source = engine_module.__dict__
    for key, value in list(namespace.items()):
        if isinstance(value, types.FunctionType) and value.__globals__ is source:
            namespace[key] = _rebind(value, namespace)
    return module


def make_crn_engine(engine_module, name=None):
    """Private CRN-capable copy of a loaded engine module; ``engine_module`` itself is not modified."""
    module = private_copy(engine_module, name or f"{engine_module.__name__}_crn")
    install(module.__dict__)
    return module


def is_crn_capable(engine_module) -> bool:
    return bool(getattr(engine_module, "_crn_installed", False))


# ------------------------------------------------------------------------------------------ environment level
def _exact_engine():
    """farm_neat.game.exact_engine in the repository; the sibling exact_engine.py in a flat copy (analysts' kit)."""
    if __package__:
        return importlib.import_module("farm_neat.game.exact_engine")
    here = Path(__file__).resolve().parent
    if not (here / "exact_engine.py").is_file():
        raise RuntimeError(f"flat crn_rng copy needs exact_engine.py next to it in {here}")
    module = sys.modules.get("exact_engine")
    if module is not None and Path(module.__file__).resolve().parent == here:
        return module
    spec = importlib.util.spec_from_file_location("exact_engine", here / "exact_engine.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["exact_engine"] = module
    spec.loader.exec_module(module)
    return module


_CRN_ENGINES = {}
_CRN_CLASSES = {}


def load_crn_engine(ee=None):
    """The CRN-capable engine: a fresh load of the pinned engine file (hash-checked) with install() applied.

    Fresh rather than copied, so hooks another tool may have set on load_engine()'s shared module never leak in.
    """
    ee = ee or _exact_engine()
    key = id(ee)
    if key not in _CRN_ENGINES:
        ee.assert_bundled_engine_identity()
        created_package, created_utils = ee._install_resolve_episode_seed_stub()
        spec = importlib.util.spec_from_file_location("farm_default_plan_bundled_kaggriculture_crn",
                                                      ee.ENGINE_PATH)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        finally:
            if created_utils:
                sys.modules.pop("kaggle_environments.utils", None)
            if created_package:
                sys.modules.pop("kaggle_environments", None)
        install(module.__dict__)
        _CRN_ENGINES[key] = module
    return _CRN_ENGINES[key]


def engine_for(configuration, ee=None):
    """load_engine() unless ``configuration`` declares ``crnRng: true`` (then load_crn_engine())."""
    ee = ee or _exact_engine()
    return load_crn_engine(ee) if enabled(configuration) else ee.load_engine()


def environment_class(ee=None):
    """CrnKaggricultureEnvironment: the plain wrapper whose engine is the CRN-capable one."""
    ee = ee or _exact_engine()
    cls = _CRN_CLASSES.get(id(ee))
    if cls is None:
        base = ee.ExactKaggricultureEnvironment

        class CrnKaggricultureEnvironment(base):
            """ExactKaggricultureEnvironment with ``crnRng: true`` (crn_rng.py). Refuses a swapped-in engine."""

            def __init__(self, *, seed, episode_steps=720, configuration=None):
                super().__init__(seed=seed, episode_steps=episode_steps, configuration=with_flag(configuration))
                self.engine = load_crn_engine(ee)

            def step(self, joint_action):
                if not is_crn_capable(self.engine) or not enabled(self.configuration):
                    raise RuntimeError("CRN environment without crnRng or with a non-CRN engine (engine swapped or "
                                       "cloned); use crn_rng.engine_for(env.configuration)")
                return super().step(joint_action)

        cls = _CRN_CLASSES[id(ee)] = CrnKaggricultureEnvironment
    return cls


def environment(*, seed, episode_steps=720, configuration=None):
    """The environment ``configuration`` asks for. Without ``crnRng: true`` this is exactly
    ``ExactKaggricultureEnvironment(seed=..., episode_steps=..., configuration=...)`` (a false-valued key is
    dropped, so the configuration actors see is unchanged); with it, CrnKaggricultureEnvironment."""
    ee = _exact_engine()
    if not enabled(configuration):
        cfg = None if configuration is None else with_flag(configuration, False)
        return ee.ExactKaggricultureEnvironment(seed=seed, episode_steps=episode_steps, configuration=cfg)
    return environment_class(ee)(seed=seed, episode_steps=episode_steps, configuration=configuration)


def set_crn(env, on: bool = True):
    """Switch an existing environment's later dusks to (or back from) CRN, e.g. at a rollout branch point.

    Draws at dusks already played are not changed. Returns ``env``.
    """
    ee = _exact_engine()
    crn_cls = environment_class(ee)
    if on:
        env.configuration[CONFIG_KEY] = True
        env.engine = load_crn_engine(ee)
        if not isinstance(env, crn_cls):
            env.__class__ = crn_cls
    else:
        env.configuration.pop(CONFIG_KEY, None)
        env.engine = ee.load_engine()
        if type(env) is crn_cls:
            env.__class__ = ee.ExactKaggricultureEnvironment
    return env


def is_crn_environment(env) -> bool:
    return enabled(getattr(env, "configuration", None)) and is_crn_capable(getattr(env, "engine", None))


def main(argv=None):
    """python3 -B src/farm_neat/rollout/crn_rng.py shops SEED [SEED ...] | weeds SEED DAY FARM [CHANCE]"""
    import argparse
    import json
    ap = argparse.ArgumentParser(description="CRN dusk draws for a world seed (no game is played)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shops", help="the shop each unlock dusk draws under crnRng")
    s.add_argument("seeds", type=int, nargs="+")
    w = sub.add_parser("weeds", help="tiles whose keyed weed draw is below CHANCE at one dusk")
    w.add_argument("seed", type=int)
    w.add_argument("day", type=int)
    w.add_argument("farm", type=int, choices=(0, 1))
    w.add_argument("chance", type=float, nargs="?", default=0.005)
    a = ap.parse_args(argv)
    if a.cmd == "shops":
        shops = _exact_engine().load_engine().SHOPS
        for seed in a.seeds:
            print(json.dumps({"seed": seed, "shops": shop_sequence(seed, shops)}))
    else:
        tiles = [[x, y] for y in range(10) for x in range(10) if weed_uniform(a.seed, a.day, a.farm, x, y) < a.chance]
        print(json.dumps({"seed": a.seed, "day": a.day, "farm": a.farm, "chance": a.chance, "weed_if_empty": tiles}))
    return 0


if __name__ == "__main__":
    if not __package__:
        _src = Path(__file__).resolve().parents[2]
        if (_src / "farm_neat" / "game" / "exact_engine.py").is_file() and str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
            __package__ = "farm_neat.rollout"  # noqa: A001 - run as a script from the repository
    sys.exit(main())
