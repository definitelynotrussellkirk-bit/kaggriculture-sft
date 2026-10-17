"""flip.py against the reference engine's own functions, from real replay states.

For every plant/animal tile of both farms at sampled boundaries of real replays, the engine's _decay_plants /
_daily_refresh_plants / _daily_refresh_animals are run forward in the interpreter's order under two scenarios:
no further action (death, decay) and kept watered/fed with no care (next production). Movement: a worker needs
d + 1 decisions to act on a tile at Manhattan distance d (engine _apply_unit_action).
    ../../../venv/python -B test_flip.py [N_REPLAYS]
"""
import copy
import glob
import json
import random
import sys
import types
import zipfile

sys.path.insert(0, ".")
if "kaggle_environments" not in sys.modules:           # the engine imports it only for _initialize
    ke, ku = types.ModuleType("kaggle_environments"), types.ModuleType("kaggle_environments.utils")
    ku.resolve_episode_seed = lambda env: 0
    sys.modules["kaggle_environments"], sys.modules["kaggle_environments.utils"] = ke, ku

import numpy as np

import data
import flip
from farm_neat.game import kaggriculture_engine_reference as ref

ARCH = sorted(glob.glob("/mnt/ai/scratch/farmNEAT/primitive-bc-data/kaggriculture-episodes-2026-09-2*.zip"))


def simulate(farm, t, scenario, horizon=72):
    """Per cell: first visible change of kind (WEED / escape), first decay tick, first production (step, units)."""
    farm = copy.deepcopy(farm)
    events = {}
    prev = {(y, x): copy.deepcopy(tile) for y, line in enumerate(farm["tiles"]) for x, tile in enumerate(line)}
    for s in range(t, min(t + horizon, 719)):
        if scenario == "kept":
            for line in farm["tiles"]:
                for tile in line:
                    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                        tile["watered_today"] = True
                    if isinstance(tile, dict) and "animal" in tile:
                        tile["fed_today"] = True
        ref._decay_plants(farm, s)
        if (s + 1) % 24 == 0:
            ref._daily_refresh_plants(farm, s // 24, 24)
            ref._daily_refresh_animals(farm, s // 24)
        for (y, x), old in prev.items():
            new = farm["tiles"][y][x]
            ev = events.setdefault((y, x), {})
            if not isinstance(old, dict):
                continue
            if "gone" not in ev and (not isinstance(new, dict) or new.get("kind") != old.get("kind")
                                     or new.get("animal") != old.get("animal")):
                ev["gone"] = s + 1 - t
            if isinstance(new, dict) and new.get("kind") == old.get("kind") and new.get("animal") == old.get("animal"):
                dy = new.get("yield_units", 0) - old.get("yield_units", 0)
                if dy < 0 and "decay" not in ev:
                    ev["decay"] = s + 1 - t
                if dy > 0 and "grow" not in ev and (s + 1) % 24 == 0:
                    ev["grow"] = (s + 1 - t, dy)
            prev[(y, x)] = copy.deepcopy(new)
    return events


def check_state(obs):
    t = obs["step"]
    seat = obs["player"]
    bad, n = [], 0
    for role, farm in enumerate((obs["farms"][seat], obs["farms"][1 - seat])):
        none_ev = simulate(farm, t, "none")
        kept_ev = simulate(farm, t, "kept")
        for y, line in enumerate(farm["tiles"]):
            for x, tile in enumerate(line):
                if not isinstance(tile, dict) or not (tile.get("kind") == "PLANT" or "animal" in tile):
                    continue
                n += 1
                plant = tile.get("kind") == "PLANT"
                fc = flip.plant_forecast(tile, t) if plant else flip.animal_forecast(tile, t)
                ev, kv = none_ev[(y, x)], kept_ev[(y, x)]
                if ev.get("gone") != fc["gone"]:
                    bad.append(("gone", role, x, y, t, fc["gone"], ev.get("gone"), tile))
                if plant and fc["decay"] is not None and fc["decay"] != fc["gone"] and ev.get("decay") != fc["decay"]:
                    bad.append(("decay", role, x, y, t, fc["decay"], ev.get("decay"), tile))
                if not plant and fc["death"] <= 72 and t + fc["death"] <= 719 and ev.get("gone") != fc["death"]:
                    bad.append(("escape", role, x, y, t, fc["death"], ev.get("gone"), tile))
                if fc["next_eta"] is not None and fc["next_eta"] <= 72:
                    got = kv.get("grow")
                    if fc["next_units"] > 0 and got != (fc["next_eta"], fc["next_units"]):
                        bad.append(("grow", role, x, y, t, (fc["next_eta"], fc["next_units"]), got, tile))
    return n, bad


def check_moves(trials=200, seed=0):
    rng = random.Random(seed)
    bad = 0
    for _ in range(trials):
        farm = ref._new_farm(10, 3000)
        for y in range(10):
            for x in range(10):
                farm["tiles"][y][x] = None
        wx, wy, cx, cy = (rng.randrange(10) for _ in range(4))
        farm["farmer"] = [wx, wy]
        farm["tiles"][cy][cx] = ref._new_plant("WHEAT", 0, 24)
        private = ref._new_private()
        decisions = 0
        while not farm["tiles"][cy][cx]["watered_today"] and decisions < 30:
            fx, fy = farm["farmer"]
            op = "EAST" if fx < cx else "WEST" if fx > cx else "SOUTH" if fy < cy else "NORTH" if fy > cy else "WATER"
            ref._apply_unit_action(farm, private, 0, [op], 10, 0, 24)
            decisions += 1
        bad += decisions != abs(wx - cx) + abs(wy - cy) + 1
    return trials, bad


def check_alignment(obs, cfg):
    packed = data.features_mod.pack(obs, None, data.pack_config(cfg))
    extra, unit_extra, reach_eta = flip.token_features(obs, packed["token_ids"])
    rows = packed["rows"]
    bad = 0
    for i, tid in enumerate(packed["token_ids"]):
        if tid.startswith("tile/"):
            x, y = (int(v) for v in tid[7:].split(","))
            bad += round(rows[i, 14] * 9) != x or round(rows[i, 15] * 9) != y
            bad += int(extra[i, flip.T0 + 20:flip.T0 + 32].astype(np.float32).argmax()) != int(extra[i, flip.KIND_COL])
    own = [i for i, tid in enumerate(packed["token_ids"]) if tid.startswith("carrier/0/")]
    for k, i in enumerate(own[:25]):
        bad += not np.array_equal(extra[i, flip.C0:flip.C0 + flip.UNIT_WIDTH], unit_extra[k])
    return bad


def check_market(obs, cfg):
    """flip.price_path against the engine's _town_consume + market_price with no trades (visible after each tick)."""
    params = flip._market_params(obs, cfg)
    market = {"inventory": dict(obs["market"]["inventory"]), "prices": dict(obs["market"]["prices"]),
              "params": params}
    town = {"unlocked_shops": list((obs.get("town") or {}).get("unlocked_shops") or [])}
    state = [types.SimpleNamespace(observation=types.SimpleNamespace(market=market, town=town))]
    env = types.SimpleNamespace(configuration={"townShopSellInterval": 4, "townCenterSellInterval": 24})
    t = obs["step"]
    path, _ = flip.price_path(obs, cfg)
    got = {item: [] for item in path}
    for s in range(t, 719):
        ref._town_consume(env, state, s)
        if s % 4 == 0:
            for item in got:
                got[item].append((s + 1 - t, market["prices"][item]))
        if len(got["WHEAT"]) >= flip.HORIZON_TICKS:
            break
    return sum(got[item] != path[item] for item in path)


def check_witnesses(obs, cfg):
    """The architecture review's float16 collisions must be distinguishable in the rounded extra rows."""
    ids = data.features_mod.pack(obs, None, data.pack_config(cfg))["token_ids"]
    seat = obs["player"]

    def rows(mutate):
        o = copy.deepcopy(obs)
        mutate(o)
        return flip.token_features(o, ids, 25, cfg)[0]
    farm0, mkt = ids.index("farm/0"), ids.index("market/CARROT")
    bad = 0

    def cash(v, rival=100000.0):
        def f(o):
            o["farms"][seat]["money"], o["farms"][1 - seat]["money"] = v, rival
        return f
    bad += np.array_equal(rows(cash(99999.0))[farm0], rows(cash(100001.0))[farm0])

    def hire(v):
        def f(o):
            o["farms"][seat]["money"], o["farms"][seat]["hires_today"] = v, 17
        return f
    bad += np.array_equal(rows(hire(2583.0))[farm0], rows(hire(2584.0))[farm0])

    def carrot(inv):
        def f(o):
            o["market"]["inventory"]["CARROT"], o["market"]["prices"]["CARROT"] = inv, 221
            o["town"] = {"unlocked_shops": ["PET_CAFE"]}
        return f
    a, b = rows(carrot(9246))[mkt], rows(carrot(9247))[mkt]
    params = flip._market_params(obs, cfg)
    distinct = flip.price_at("CARROT", 9244, params) != flip.price_at("CARROT", 9245, params)
    bad += distinct and np.array_equal(a, b)
    return bad, bool(distinct)


def main(n_replays=6, seed=0):
    rng = random.Random(seed)
    members = []
    for a in ARCH[-3:]:
        z = zipfile.ZipFile(a)
        members += [(a, m) for m in z.namelist() if m.endswith(".json")][:40]
    rng.shuffle(members)
    tiles = states = 0
    bads, align_bad, market_bad, witness_bad, witness_live = [], 0, 0, 0, 0
    for a, m in members[:n_replays]:
        replay = json.loads(zipfile.ZipFile(a).read(m))
        for t in (5, 47, 130, 250, 263, 400, 551, 600, 671, 700):
            for seat in (0, 1):
                obs = data.bc.public_observation(replay, t, seat)
                n, bad = check_state(obs)
                tiles += n
                states += 1
                bads += bad
                align_bad += check_alignment(obs, replay["configuration"])
                market_bad += check_market(obs, replay["configuration"])
                wb, live = check_witnesses(obs, replay["configuration"])
                witness_bad, witness_live = witness_bad + wb, witness_live + live
    trials, move_bad = check_moves()
    print(json.dumps({"states": states, "tiles_checked": tiles, "forecast_mismatches": len(bads),
                      "alignment_mismatches": align_bad, "market_path_mismatches": market_bad,
                      "witness_collisions": witness_bad, "states_with_live_carrot_witness": witness_live,
                      "move_trials": trials, "move_mismatches": move_bad}))
    for b in bads[:10]:
        print("MISMATCH", b)
    return 1 if bads or align_bad or move_bad or market_bad or witness_bad else 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 6))
