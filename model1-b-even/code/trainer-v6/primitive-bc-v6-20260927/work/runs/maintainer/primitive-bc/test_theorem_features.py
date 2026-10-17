"""Engine-checked exactness of theorem_features on real replayed states.

The recorded public episode 111424871 is replayed through the exact bundled engine
(src/farm_neat/game/exact_engine.py); its observations match the recording byte-for-byte.
At every step each feature is compared with what the engine then does: real day-end refreshes,
decay, town consumption and market fills (instrumented engine functions), plus counterfactual
calls of the engine's own functions (_process_market, _apply_unit_action, _daily_refresh_*,
_decay_plants, _do_hire, _do_buy_land) on copies of the real state. ``python -B
test_theorem_features.py`` prints the per-feature report as JSON.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import theorem_features as TF  # noqa: E402
from farm_neat.evaluation import unit_economics  # noqa: E402
from farm_neat.game.exact_engine import AttrDict, ExactKaggricultureEnvironment, load_engine  # noqa: E402
from farm_neat.training.bc import actor_observation  # noqa: E402

DAILY = ROOT / "work/datasets/public-episodes/daily"
EPISODE = DAILY / "111424871.json"          # the assigned episode
COVERAGE_EPISODE = DAILY / "111419545.json"  # adds GOOSE/COOP/EGG states, absent from 111424871
EPISODES = (EPISODE, COVERAGE_EPISODE)
LAST = 718
LASTDAY = 29
SIM_HOURS = (0, 11, 23)
T = {n: i for i, n in enumerate(TF.TILE_NAMES)}
PN = {n: i for i, n in enumerate(TF.PRODUCT_NAMES)}
GN = {n: i for i, n in enumerate(TF.GLOBAL_NAMES)}
SN = {n: i for i, n in enumerate(TF.STRUCTURE_NAMES)}
PRODUCTS = TF.PRODUCTS
# theorem K-MARKET-DEEP-FLOOR-LOCK-9A7C2E4D (F_p), WHEAT and EGG unreachable
FLOOR_THEOREM = {"STRAWBERRY": 10062, "WOOL": 10059, "MILK": 10076, "MELON": 10158,
                 "FERTILIZER": 10493, "TOMATO": 10529, "CARROT": 10842, "WHEAT": None, "EGG": None}


# ----------------------------------------------------------------------------- tally
class Tally:
    def __init__(self):
        self.rows = {}

    def check(self, feature, method, ok, detail=None):
        row = self.rows.setdefault(feature, {"checked": 0, "mismatch": 0, "methods": {}, "first": None})
        row["checked"] += 1
        per = row["methods"].setdefault(method, [0, 0])
        per[0] += 1
        per[1] += not ok
        if not ok:
            row["mismatch"] += 1
            if row["first"] is None:
                row["first"] = detail

    def eq(self, feature, method, predicted, actual, detail=None):
        ok = float(np.float32(predicted)) == float(np.float32(actual))
        self.check(feature, method, ok, None if ok else {"pred": float(predicted), "actual": float(actual), **(detail or {})})
        if float(actual) != 0:
            self.rows[feature]["nonzero"] = self.rows[feature].get("nonzero", 0) + 1

    def report(self):
        return {k: {"checked": v["checked"], "mismatch": v["mismatch"], "nonzero_actual": v.get("nonzero", 0),
                    "methods": {m: {"checked": c, "mismatch": x} for m, (c, x) in sorted(v["methods"].items())},
                    **({"first_mismatch": v["first"]} if v["first"] else {})}
                for k, v in sorted(self.rows.items())}


# ----------------------------------------------------------------------------- engine helpers
def _mini(tile):
    return {"tiles": [[tile]], "farmer": [0, 0], "hands": [], "money": 0.0,
            "unlocked_quadrants": ["NW"], "hires_today": 0}


def _act(E, farm, private, op, day):
    E._apply_unit_action(farm, private, 0, [op], 1, day, 24)


def _alive(tile):
    return isinstance(tile, dict) and (tile.get("kind") == "PLANT" or "animal" in tile)


def _refresh(E, farm, day):
    tile = farm["tiles"][0][0]
    if tile.get("kind") == "PLANT":
        E._daily_refresh_plants(farm, day, 24)
    else:
        E._daily_refresh_animals(farm, day)


def _product_of(E, tile):
    return tile["crop"] if tile.get("kind") == "PLANT" else E.ANIMALS[tile["animal"]]["product"]


def _cf_market(E, market, moneys, privates, queues):
    """Run the engine market phase on copies. Returns (cash deltas, market after)."""
    market = {"inventory": dict(market["inventory"]), "prices": dict(market["prices"])}
    farms = [{"money": float(m), "hires_today": 0, "hands": [], "unlocked_quadrants": ["NW"]} for m in moneys]
    state = [AttrDict(observation=AttrDict(farms=farms, market=market, private=privates[0]),
                      action={"market": queues[0]}),
             AttrDict(observation=AttrDict(private=privates[1]), action={"market": queues[1]})]
    E._process_market(state, SimpleNamespace(configuration={}))
    return [farms[i]["money"] - moneys[i] for i in range(2)], market


def _private(shed=None, seeds=None):
    base = {item: 0 for item in PRODUCTS + ["GOOSE", "COW", "SHEEP"]}
    base.update(shed or {})
    return {"shed": base, "seeds": dict(seeds or {}), "inventories": [{}]}


_ANIMAL_SIM = {}


def _animal_sim(name, days):
    key = (name, days)
    if key not in _ANIMAL_SIM:
        _ANIMAL_SIM[key] = unit_economics.simulate(name, "fed_cared", days)
    return _ANIMAL_SIM[key]


def _service_sim(E, tile, d):
    """Daily service (WATER / FEED+CARE) with the declared harvest policy; engine arithmetic."""
    t = copy.deepcopy(tile)
    farm = _mini(t)
    private = {"inventories": [{"WHEAT": 10 ** 6}], "seeds": {}, "shed": {}}
    product = _product_of(E, t)
    is_plant = t.get("kind") == "PLANT"
    one_shot = is_plant and not E.CROPS[t["crop"]]["ongoing"]
    units, events = 0, []
    if one_shot:
        c = E.CROPS[t["crop"]]
        age = d - t["planted_day"]
        harvest_day = d if age > c["max_yield_day"] else min(t["planted_day"] + c["max_yield_day"], LASTDAY)
    for day in range(d, LASTDAY + 1):
        cur = farm["tiles"][0][0]
        if not _alive(cur):
            break
        if is_plant:
            if not cur["watered_today"]:
                _act(E, farm, private, "WATER", day)
        else:
            if not cur["fed_today"]:
                _act(E, farm, private, "FEED", day)
            if not cur["cared_today"]:
                _act(E, farm, private, "CARE", day)
        if not one_shot or day == harvest_day:
            before = private["inventories"][0].get(product, 0)
            _act(E, farm, private, "HARVEST", day)
            units += private["inventories"][0].get(product, 0) - before
            if one_shot:
                break
        if day < LASTDAY:
            y0 = farm["tiles"][0][0].get("yield_units", 0)
            _refresh(E, farm, day)
            cur = farm["tiles"][0][0]
            if _alive(cur) and cur.get("yield_units", 0) > y0:
                events.append((day + 1, cur["yield_units"] - y0))
    return units, events


def _no_harvest_until(E, tile, d, event_day):
    """Serviced, never harvested, up to and including the refresh that makes event_day's yield.
    Returns (yield after that refresh, engine pending_care_bonus just before it)."""
    t = copy.deepcopy(tile)
    farm = _mini(t)
    private = {"inventories": [{"WHEAT": 10 ** 6}], "seeds": {}, "shed": {}}
    is_plant = t.get("kind") == "PLANT"
    for day in range(d, event_day):
        cur = farm["tiles"][0][0]
        if is_plant:
            if not cur["watered_today"]:
                _act(E, farm, private, "WATER", day)
        else:
            if not cur["fed_today"]:
                _act(E, farm, private, "FEED", day)
            if not cur["cared_today"]:
                _act(E, farm, private, "CARE", day)
        pending = farm["tiles"][0][0].get("pending_care_bonus", 0)
        _refresh(E, farm, day)
    return farm["tiles"][0][0]["yield_units"], pending


# ----------------------------------------------------------------------------- instrumentation
class Recorder:
    def __init__(self, E, env, cfg, tally):
        self.E, self.env, self.cfg, self.tally = E, env, cfg, tally
        self.town, self.market, self.decay, self.refresh_steps = {}, {}, {}, []
        self.saved = {}

    def install(self):
        E = self.E
        for name in ("_town_consume", "_process_market", "_decay_plants", "_end_of_day"):
            self.saved[name] = getattr(E, name)
        orig_town, orig_market = self.saved["_town_consume"], self.saved["_process_market"]
        orig_decay, orig_eod = self.saved["_decay_plants"], self.saved["_end_of_day"]

        def town(env, state, step):
            inv = state[0].observation.market["inventory"]
            before = dict(inv)
            orig_town(env, state, step)
            self.town[step] = tuple(before[p] - inv[p] for p in PRODUCTS)

        def market(state, env):
            if state is not self.env.state:          # counterfactual call on a copy
                return orig_market(state, env)
            inv = state[0].observation.market["inventory"]
            before = dict(inv)
            orig_market(state, env)
            self.market[state[0].observation.step] = tuple(inv[p] - before[p] for p in PRODUCTS)

        def decay(farm, step):
            farms = self.env.state[0].observation.farms
            fi = next((i for i, f in enumerate(farms) if f is farm), None)
            if fi is None:                             # counterfactual call on a copy
                return orig_decay(farm, step)
            pre = {}
            for y, line in enumerate(farm["tiles"]):
                for x, t in enumerate(line):
                    if isinstance(t, dict) and t.get("kind") == "PLANT":
                        pre[(x, y)] = (t["crop"], t["planted_day"], t["yield_units"])
            orig_decay(farm, step)
            out = {}
            for (x, y), (crop, pd, yu) in pre.items():
                t = farm["tiles"][y][x]
                same = isinstance(t, dict) and t.get("kind") == "PLANT" and t["yield_units"] == yu
                out[(x, y)] = (crop, pd, not same)
            self.decay[(step, fi)] = out

        def end_of_day(state, env, day):
            step = state[0].observation.step
            self.refresh_steps.append(step)
            obs0 = state[0].observation
            pre_obs = [actor_observation({"player": p, "step": step, "day": step // 24, "hour": step % 24,
                                          "farms": copy.deepcopy(obs0.farms), "market": copy.deepcopy(obs0.market),
                                          "town": copy.deepcopy(obs0.town),
                                          "private": copy.deepcopy(state[p].observation.private)})
                       for p in range(2)]
            feats = [TF.compute(o, self.cfg) for o in pre_obs]
            pre_farms = copy.deepcopy(obs0.farms)
            pre_sheds = [sum(state[p].observation.private["shed"].values()) for p in range(2)]
            pre_carried = [sum(sum(inv.values()) for inv in state[p].observation.private["inventories"])
                           for p in range(2)]
            orig_eod(state, env, day)
            self._check_refresh(pre_farms, obs0.farms, feats, pre_sheds, pre_carried, state)

        E._town_consume, E._process_market, E._decay_plants, E._end_of_day = town, market, decay, end_of_day

    def uninstall(self):
        for name, fn in self.saved.items():
            setattr(self.E, name, fn)

    def _check_refresh(self, pre_farms, post_farms, feats, pre_sheds, pre_carried, state):
        tally, m = self.tally, "engine day-end refresh"
        for p in range(2):
            discarded = pre_carried[p] - (sum(state[p].observation.private["shed"].values()) - pre_sheds[p])
            tally.eq("global.dayend_overflow", m, feats[p]["global"][GN["dayend_overflow"]], discarded)
            block = feats[p]["tile_own"]
            for y in range(10):
                for x in range(10):
                    pre = pre_farms[p]["tiles"][y][x]
                    if not _alive(pre):
                        continue
                    post = post_farms[p]["tiles"][y][x]
                    row = block[y * 10 + x]
                    survived = _alive(post)
                    tally.eq("tile.dies_tonight_if_unserviced", m, row[T["dies_tonight_if_unserviced"]], not survived)
                    gain = post["yield_units"] - pre["yield_units"] if survived else 0
                    tally.eq("tile.gain_tonight_current", m, row[T["gain_tonight_current"]], gain)
                    if "animal" in pre:
                        tally.eq("tile.pending_after_tonight_current", m, row[T["pending_after_tonight_current"]],
                                 post.get("pending_care_bonus", 0) if survived else 0)
                        tally.eq("tile.fertilizer_available_tomorrow", m, row[T["fertilizer_available_tomorrow"]],
                                 bool(survived and post["fertilizer_available"]))


# ----------------------------------------------------------------------------- per-observation checks
def _check_tiles(E, tally, s, farm, block, sim):
    d = s // 24
    for y in range(10):
        for x in range(10):
            tile = farm["tiles"][y][x]
            row = block[y * 10 + x]
            kind = None if tile is None else ("LOCKED" if tile == "LOCKED" else tile.get("kind"))
            tally.eq("tile.owned", "engine tile", row[T["owned"]], tile != "LOCKED")
            tally.eq("tile.plantable", "engine tile", row[T["plantable"]], tile is None)
            tally.eq("tile.weed", "engine tile", row[T["weed"]], kind == "WEED")
            tally.eq("tile.structure_free", "engine tile", row[T["structure_free"]],
                     kind in ("COOP", "PASTURE") and "animal" not in tile)
            # distance: every move is legal in bounds (locked traversable); verify by walking
            units = [farm["farmer"], *farm["hands"]]
            if s % 24 == 0 and y == x:
                best = None
                for ux, uy in units:
                    walker = {"tiles": [[None] * 10 for _ in range(10)], "farmer": [ux, uy], "hands": [],
                              "money": 0.0, "unlocked_quadrants": ["NW"], "hires_today": 0}
                    n = 0
                    while tuple(walker["farmer"]) != (x, y):
                        fx, fy = walker["farmer"]
                        op = "EAST" if fx < x else "WEST" if fx > x else "SOUTH" if fy < y else "NORTH"
                        E._apply_unit_action(walker, _private(), 0, [op], 10, d, 24)
                        n += 1
                    best = n if best is None else min(best, n)
                tally.eq("tile.unit_distance", "engine moves along a Manhattan path", row[T["unit_distance"]], best)
            if not _alive(tile):
                for col in ("age_days", "stored_yield", "season_units_remaining", "harvestable_now"):
                    tally.eq("tile." + col, "engine tile", row[T[col]], 0)
                continue
            is_plant = kind == "PLANT"
            product = _product_of(E, tile)
            yu = tile["yield_units"]
            cap = E.CROPS[tile["crop"]]["max_yield"] if is_plant else E.ANIMALS[tile["animal"]]["max_held"]
            tally.eq("tile.age_days", "engine field", row[T["age_days"]],
                     d - (tile["planted_day"] if is_plant else tile["placed_day"]))
            tally.eq("tile.stored_yield", "engine field", row[T["stored_yield"]], yu)
            tally.eq("tile.yield_room", "engine cap", row[T["yield_room"]], cap - yu)
            tally.eq("tile.service_needed_today", "engine field", row[T["service_needed_today"]],
                     not (tile["watered_today"] if is_plant else tile["fed_today"]))
            if not is_plant:
                tally.eq("tile.care_needed_today", "engine field", row[T["care_needed_today"]], not tile["cared_today"])
                tally.eq("tile.pending_care_bonus", "engine field", row[T["pending_care_bonus"]], tile["pending_care_bonus"])
                tally.eq("tile.fertilizer_available_now", "engine field", row[T["fertilizer_available_now"]],
                         tile["fertilizer_available"])
            else:
                fud = tile["fertilized_until_day"]
                tally.eq("tile.fert_active_today", "engine field", row[T["fert_active_today"]], fud >= d)
                tally.eq("tile.fert_days_left", "engine field", row[T["fert_days_left"]], max(0, fud - d + 1))
            # counterfactual actions on a copy
            mini, priv = _mini(copy.deepcopy(tile)), _private()
            _act(E, mini, priv, "HARVEST", d)
            got = priv["inventories"][0].get(product, 0)
            tally.eq("tile.harvestable_now", "engine HARVEST on copy", row[T["harvestable_now"]], got)
            if is_plant:
                mini = _mini(copy.deepcopy(tile))
                _act(E, mini, _private(), "WATER", d)
                gain = mini["tiles"][0][0]["yield_units"] - yu if not tile["watered_today"] else 0
                tally.eq("tile.water_gain_now", "engine WATER on copy", row[T["water_gain_now"]], gain)
            # counterfactual refresh tonight: current flags, serviced, zero-yield (event detector)
            refresh = d < LASTDAY
            cur = _mini(copy.deepcopy(tile))
            _refresh(E, cur, d)
            ct = cur["tiles"][0][0]
            survives = _alive(ct)
            tally.eq("tile.dies_tonight_if_unserviced", "engine refresh on copy", row[T["dies_tonight_if_unserviced"]],
                     refresh and not survives)
            tally.eq("tile.gain_tonight_current", "engine refresh on copy", row[T["gain_tonight_current"]],
                     (ct["yield_units"] - yu if survives else 0) if refresh else 0)
            if not is_plant:
                tally.eq("tile.pending_after_tonight_current", "engine refresh on copy",
                         row[T["pending_after_tonight_current"]],
                         (ct.get("pending_care_bonus", 0) if survives else 0) if refresh else tile["pending_care_bonus"])
                tally.eq("tile.fertilizer_available_tomorrow", "engine refresh on copy",
                         row[T["fertilizer_available_tomorrow"]], refresh and survives and ct["fertilizer_available"])
            srv_tile = copy.deepcopy(tile)
            srv_tile["watered_today" if is_plant else "fed_today"] = True
            srv = _mini(srv_tile)
            _refresh(E, srv, d)
            tally.eq("tile.gain_tonight_serviced", "engine refresh on serviced copy", row[T["gain_tonight_serviced"]],
                     (srv["tiles"][0][0]["yield_units"] - yu) if refresh else 0)
            zero_tile = copy.deepcopy(srv_tile)
            zero_tile["yield_units"] = 0
            zero = _mini(zero_tile)
            _refresh(E, zero, d)
            tally.eq("tile.event_tonight", "engine refresh on zero-yield serviced copy", row[T["event_tonight"]],
                     refresh and zero["tiles"][0][0]["yield_units"] > 0)
            # unserviced death clock: refresh repeatedly with no service
            dead = _mini(copy.deepcopy(tile))
            nights = 0
            while _alive(dead["tiles"][0][0]) and nights < 6:
                _refresh(E, dead, d + nights)
                nights += 1
            tally.eq("tile.day_ends_to_death_unserviced", "engine refreshes on copy, no service",
                     row[T["day_ends_to_death_unserviced"]], nights)
            # decay-death clock on a copy (no further actions)
            if is_plant and tile["max_lifespan_step"] >= 0 and row[T["decay_scheduled"]]:
                dec = _mini(copy.deepcopy(tile))
                t = s
                while dec["tiles"][0][0].get("kind") == "PLANT" and t < s + 400:
                    E._decay_plants(dec, t)
                    t += 1
                tally.eq("tile.steps_to_decay_death", "engine _decay_plants on copy", row[T["steps_to_decay_death"]], t - 1 - s)
            if sim:
                units, events = _service_sim(E, tile, d)
                tally.eq("tile.season_units_remaining", "engine service simulation", row[T["season_units_remaining"]], units)
                tally.eq("tile.season_value_at_quote", "engine service simulation x quote",
                         row[T["season_value_at_quote"]], units * sim[product])
                tally.eq("tile.events_remaining", "engine service simulation", row[T["events_remaining"]], len(events))
                tally.eq("tile.next_event_days", "engine service simulation", row[T["next_event_days"]],
                         events[0][0] - d if events else 0)
                if events:
                    post, pending = _no_harvest_until(E, tile, d, events[0][0])
                    # uncapped event gain: plants from the harvested run (held 0, gain <= 2 < cap);
                    # animals 1 + the engine's pending bonus right before the event refresh (fed)
                    g = events[0][1] if is_plant else 1 + pending
                    tally.eq("tile.overflow_next_event_serviced", "engine service simulation without harvest",
                             row[T["overflow_next_event_serviced"]], yu + g - post)


def replay_report(episode=EPISODE, sim_hours=SIM_HOURS):
    E = load_engine()
    data = json.loads(Path(episode).read_text())
    steps, cfg = data["steps"], data["configuration"]
    env = ExactKaggricultureEnvironment(seed=data["info"]["seed"])
    tally = Tally()
    rec = Recorder(E, env, cfg, tally)
    TF.warm(cfg)
    history = []            # per step: dict of what later-window checks need
    timings = []
    fidelity = 0
    identity_mls = {}       # (farm, x, y, crop, pd) -> predicted mls from an earlier obs
    rec.install()
    try:
        for s in range(0, LAST + 1):
            raw = [env.agent_observation(p) for p in range(2)]
            if json.loads(json.dumps({k: raw[0][k] for k in ("farms", "market", "town")})) != \
                    {k: steps[s][0]["observation"][k] for k in ("farms", "market", "town")}:
                fidelity += 1
            actor = [actor_observation(r) for r in raw]
            feats = []
            for p in range(2):
                t0 = time.perf_counter()
                feats.append(TF.compute(actor[p], cfg))
                timings.append(time.perf_counter() - t0)
            _check_step(E, tally, s, raw, feats, env, identity_mls, sim_hours)
            history.append({"shops": list(raw[0]["town"]["unlocked_shops"]),
                            "prices": dict(raw[0]["market"]["prices"]),
                            "inventory": dict(raw[0]["market"]["inventory"]),
                            "hands": [len(raw[p]["farms"][p]["hands"]) for p in range(2)],
                            "hires": [raw[p]["farms"][p]["hires_today"] for p in range(2)],
                            "feats": [{"global": f["global"], "product": f["product"]} for f in feats]})
            env.step([steps[s + 1][p]["action"] for p in range(2)])
            for fi in range(2):
                _check_decay(tally, s, fi, feats[fi]["tile_own"], raw[fi]["farms"][fi], rec.decay[(s, fi)])
        final = env.agent_observation(0)
        history.append({"shops": list(final["town"]["unlocked_shops"]), "prices": dict(final["market"]["prices"]),
                        "inventory": dict(final["market"]["inventory"])})
    finally:
        rec.uninstall()
    _check_windows(tally, history, rec)
    t = np.asarray(timings) * 1e6
    return {"episode": Path(episode).name, "seed": data["info"]["seed"], "replayed_steps": LAST + 1,
            "replay_mismatched_observations": fidelity, "compute_calls": len(t),
            "cpu_us_per_call": {"mean": round(float(t.mean()), 1), "median": round(float(np.median(t)), 1),
                                "p99": round(float(np.percentile(t, 99)), 1), "max": round(float(t.max()), 1)},
            "features": tally.report()}


def _check_step(E, tally, s, raw, feats, env, identity_mls, sim_hours):
    d = s // 24
    market = raw[0]["market"]
    params = TF.MARKET_PARAMS
    f0, f1 = feats
    # ---- cross-seat symmetry: public quantities seen from both seats are identical
    for c in TF.TILE_NAMES:
        for fi in range(2):
            own = feats[fi]["tile_own"][:, T[c]]
            other = feats[1 - fi]["tile_rival"][:, T[c]]
            tally.check("tile." + c, "own view == rival's view of the same farm", bool((own == other).all()))
    market_cols = [c for c in TF.PRODUCT_NAMES if not c.startswith(("own_", "rival_", "shed_", "carried_",
                                                                       "buy_affordable", "quote_after_shed"))]
    for c in market_cols:
        tally.check("product." + c, "equal from both seats", bool((f0["product"][:, PN[c]] == f1["product"][:, PN[c]]).all()))
    for c in TF.PRODUCT_NAMES:
        if c.startswith("own_"):
            r = "rival_" + c[4:]
            tally.check("product." + c, "own(seat) == rival(other seat)",
                        bool((f0["product"][:, PN[c]] == f1["product"][:, PN[r]]).all()))
            tally.check("product." + r, "own(seat) == rival(other seat)",
                        bool((f1["product"][:, PN[c]] == f0["product"][:, PN[r]]).all()))
    # ---- global fields read from the engine state
    for p in range(2):
        g = feats[p]["global"]
        me, rv = raw[p]["farms"][p], raw[p]["farms"][1 - p]
        priv = raw[p]["private"]
        tally.eq("global.steps_remaining", "engine calendar", g[GN["steps_remaining"]], LAST - s + 1)
        tally.eq("global.days_remaining", "engine calendar", g[GN["days_remaining"]], LASTDAY - d + 1)
        tally.eq("global.own_cash", "engine field", g[GN["own_cash"]], me["money"])
        tally.eq("global.rival_cash", "engine field", g[GN["rival_cash"]], rv["money"])
        tally.eq("global.cash_diff", "engine field", g[GN["cash_diff"]], me["money"] - rv["money"])
        tally.eq("global.own_hands", "engine field", g[GN["own_hands"]], len(me["hands"]))
        tally.eq("global.rival_hands", "engine field", g[GN["rival_hands"]], len(rv["hands"]))
        tally.eq("global.shops_visible", "engine field", g[GN["shops_visible"]], len(raw[p]["town"]["unlocked_shops"]))
        shed_used = sum(priv["shed"].values())
        tally.eq("global.shed_used", "engine field", g[GN["shed_used"]], shed_used)
        tally.eq("global.carried_total", "engine field", g[GN["carried_total"]],
                 sum(sum(i.values()) for i in priv["inventories"]))
        tally.eq("global.own_seed_total", "engine field", g[GN["own_seed_total"]], sum(priv["seeds"].values()))
        tally.eq("global.own_fertilizer_held", "engine field", g[GN["own_fertilizer_held"]],
                 priv["shed"]["FERTILIZER"] + sum(i.get("FERTILIZER", 0) for i in priv["inventories"]))
        # wages and hires: engine _hire_cost / _do_hire on copies
        for who, farm in (("own", me), ("rival", rv)):
            h = farm["hires_today"]
            tally.eq(f"global.{who}_wages_today", "engine _hire_cost", g[GN[f"{who}_wages_today"]],
                     sum(E._hire_cost(i) for i in range(h)))
            rich = {"money": 1e12, "hires_today": h, "hands": list(farm["hands"]), "farmer": farm["farmer"]}
            E._do_hire(rich, {"inventories": [{}]}, 10)
            tally.eq(f"global.{who}_next_hire_cost", "engine _do_hire on copy", g[GN[f"{who}_next_hire_cost"]], 1e12 - rich["money"])
            land = {"money": 1e12, "unlocked_quadrants": list(farm["unlocked_quadrants"]),
                    "tiles": [["LOCKED"] * 10 for _ in range(10)]}
            E._do_buy_land(land, 10)
            tally.eq(f"global.{who}_next_land_price", "engine _do_buy_land on copy",
                     g[GN[f"{who}_next_land_price"]], 1e12 - land["money"])
            tally.eq(f"global.{who}_land_count", "engine field", g[GN[f"{who}_land_count"]], len(farm["unlocked_quadrants"]))
        real = {"money": me["money"], "hires_today": me["hires_today"], "hands": list(me["hands"]), "farmer": me["farmer"]}
        n = 0
        while n < 60:
            before = real["hires_today"]
            E._do_hire(real, {"inventories": [{}]}, 10)
            if real["hires_today"] == before:
                break
            n += 1
        tally.eq("global.own_hires_affordable", "engine _do_hire repeated on copy", g[GN["own_hires_affordable"]], n)
        land = {"money": me["money"], "unlocked_quadrants": list(me["unlocked_quadrants"]),
                "tiles": [["LOCKED"] * 10 for _ in range(10)]}
        E._do_buy_land(land, 10)
        tally.eq("global.own_land_affordable", "engine _do_buy_land on copy", g[GN["own_land_affordable"]],
                 land["money"] < me["money"])
        # fertilizer collection on copies of each animal tile
        for who, farm in (("own", me), ("rival", rv)):
            got = 0
            for line in farm["tiles"]:
                for t in line:
                    if isinstance(t, dict) and "animal" in t:
                        mini, pv = _mini(copy.deepcopy(t)), _private()
                        _act(E, mini, pv, "COLLECT_FERTILIZER", d)
                        got += pv["inventories"][0].get("FERTILIZER", 0)
            tally.eq(f"global.{who}_fertilizer_collectable", "engine COLLECT_FERTILIZER on copies",
                     g[GN[f"{who}_fertilizer_collectable"]], got)
        # liquidation: SELL every shed product in one queue, rival queue empty
        orders = [["SELL", q, priv["shed"][q]] for q in PRODUCTS if priv["shed"].get(q, 0) > 0]
        cash, _ = _cf_market(E, market, [me["money"], 0], [_private(priv["shed"]), _private()], [orders, []])
        tally.eq("global.own_shed_liquidation", "engine _process_market on copy", g[GN["own_shed_liquidation"]], cash[0])
        tally.eq("global.liquidation_lead", "engine _process_market on copy", g[GN["liquidation_lead"]],
                 me["money"] + cash[0] - rv["money"])
        tally.eq("global.shed_room", "engine capacity", g[GN["shed_room"]], 100 - shed_used)
        held = sum((priv["shed"].get(q, 0) + sum(i.get(q, 0) for i in priv["inventories"])) * market["prices"][q]
                   for q in PRODUCTS)
        tally.eq("global.own_held_value_at_quote", "held units x engine quote", g[GN["own_held_value_at_quote"]], held)
        # product/seat specific market counterfactuals
        pr = feats[p]["product"]
        for i, q in enumerate(PRODUCTS):
            sq = priv["shed"].get(q, 0)
            tally.eq("product.shed_qty", "engine field", pr[i, PN["shed_qty"]], sq)
            tally.eq("product.carried_qty", "engine field", pr[i, PN["carried_qty"]],
                     sum(inv.get(q, 0) for inv in priv["inventories"]))
            if sq:
                cash, after = _cf_market(E, market, [0, 0], [_private({q: sq}), _private()], [[["SELL", q, sq]], []])
                tally.eq("product.shed_receipts", "engine _process_market on copy", pr[i, PN["shed_receipts"]], cash[0])
                tally.eq("product.quote_after_shed_sale", "engine _process_market on copy",
                         pr[i, PN["quote_after_shed_sale"]], E.market_price(q, after["inventory"][q]))
            if q in TF.BUYABLE:
                cash, after = _cf_market(E, market, [me["money"], 0], [_private(priv["shed"]), _private()],
                                         [[["BUY_PRODUCT", q, 1000]], []])
                tally.eq("product.buy_affordable_units", "engine _process_market on copy",
                         pr[i, PN["buy_affordable_units"]], market["inventory"][q] - after["inventory"][q])
        # producers: fills of BUY_SEED / BUY_ANIMAL
        st = feats[p]["structure"]
        for r, name in enumerate(TF.PRODUCERS):
            op = "BUY_SEED" if name in E.CROPS else "BUY_ANIMAL"
            pv = _private(priv["shed"], priv["seeds"])
            cash, _ = _cf_market(E, market, [me["money"], 0], [pv, _private()], [[[op, name, 99_000]], []])
            cost = E.CROPS[name]["seed"] if name in E.CROPS else E.ANIMALS[name]["cost"]
            tally.eq("structure.buyable_now", "engine _process_market on copy", st[r, SN["buyable_now"]], -cash[0] / cost)
            tally.eq("structure.start_cost", "engine price table", st[r, SN["start_cost"]], cost)
            stock = priv["seeds"].get(name, 0) if name in E.CROPS else priv["shed"].get(name, 0) + sum(
                inv.get(name, 0) for inv in priv["inventories"])
            tally.eq("structure.own_start_stock", "engine field", st[r, SN["own_start_stock"]], stock)
            days = LASTDAY - d + 1
            pol = "watered" if name in E.CROPS else "fed_cared"
            ref = unit_economics.formula(name, pol, days)
            tally.eq("structure.season_units_watered", "unit_economics.formula (theorem module)",
                     st[r, SN["season_units_watered"]], ref["units"])
            tally.eq("structure.season_cycles", "unit_economics.formula (theorem module)", st[r, SN["season_cycles"]],
                     ref.get("cycles", ref.get("events")))
            tally.eq("structure.first_output_days", "unit_economics.formula (theorem module)",
                     st[r, SN["first_output_days"]], ref.get("cycle_days", ref["first_output_day"])
                     if name in E.CROPS and not E.CROPS[name]["ongoing"] else ref["first_output_day"])
            quote = market["prices"][TF.PRODUCT_OF[name]]
            value = ref["units"] * quote
            tally.eq("structure.season_value_at_quote", "formula units x engine quote", st[r, SN["season_value_at_quote"]], value)
            if name in E.CROPS:
                in_cost = cost * ref["cycles"]
            else:
                wheat, _ = _cf_market(E, market, [1e12, 0], [_private(), _private()], [[["BUY_PRODUCT", "WHEAT", 1]], []])
                in_cost = cost - days * wheat[0]
                sim_ref = _animal_sim(name, days)
                tally.eq("structure.wheat_needed", "unit_economics.simulate (engine one-tile season)",
                         st[r, SN["wheat_needed"]], sim_ref["wheat_fed"])
                tally.eq("structure.fertilizer_output", "unit_economics.simulate (engine one-tile season)",
                         st[r, SN["fertilizer_output"]], sim_ref["fertilizer_collected"])
            tally.eq("structure.season_input_cost", "formula cycles x price / engine wheat buy quote",
                     st[r, SN["season_input_cost"]], in_cost)
            tally.eq("structure.season_net_at_quote", "value - input cost", st[r, SN["season_net_at_quote"]], value - in_cost)
            if name in E.CROPS:
                tally.eq("structure.season_units_fertilized", "unit_economics.formula (theorem module)",
                         st[r, SN["season_units_fertilized"]], unit_economics.formula(name, "watered_fertilized", days)["units"])
    # ---- market quotes/receipts/costs: once per step (market is shared)
    pr = f0["product"]
    for i, q in enumerate(PRODUCTS):
        inv = market["inventory"][q]
        tally.eq("product.sell_quote", "engine price in observation", pr[i, PN["sell_quote"]], market["prices"][q])
        tally.eq("product.inventory_offset", "engine field", pr[i, PN["inventory_offset"]], inv - 10000)
        for k in TF.K_SET:
            cash, after = _cf_market(E, market, [0, 0], [_private({q: k}), _private()], [[["SELL", q, k]], []])
            tally.eq(f"product.sell_receipts_{k}", "engine _process_market on copy", pr[i, PN[f"sell_receipts_{k}"]], cash[0])
            tally.eq(f"product.sell_quote_after_{k}", "engine _process_market on copy",
                     pr[i, PN[f"sell_quote_after_{k}"]], E.market_price(q, after["inventory"][q]))
            if q in TF.BUYABLE:
                cash, _ = _cf_market(E, market, [1e12, 0], [_private(), _private()], [[["BUY_PRODUCT", q, k]], []])
                tally.eq(f"product.buy_cost_{k}", "engine _process_market on copy", pr[i, PN[f"buy_cost_{k}"]], -cash[0])
            else:
                tally.eq(f"product.buy_cost_{k}", "not buyable", pr[i, PN[f"buy_cost_{k}"]], 0)
        if q in TF.BUYABLE:
            cash, _ = _cf_market(E, market, [1e12, 0], [_private(), _private()], [[["BUY_PRODUCT", q, 1]], []])
            tally.eq("product.buy_quote", "engine _process_market on copy", pr[i, PN["buy_quote"]], -cash[0])
        tally.eq("product.buyable", "engine BUY_PRODUCT item set", pr[i, PN["buyable"]], q in ("WHEAT", "FERTILIZER"))
        cash, _ = _cf_market(E, market, [0, 0], [_private({q: 10}), _private({q: 10})],
                             [[["SELL", q, 10]], [["SELL", q, 10]]])
        tally.eq("product.mirror_receipts_10", "engine _process_market both seats on copy",
                 pr[i, PN["mirror_receipts_10"]], cash[0])
        fl = FLOOR_THEOREM[q]
        tally.eq("product.floor_reachable", "theorem F_p + engine market_price", pr[i, PN["floor_reachable"]], fl is not None)
        if fl is not None:
            ok = E.market_price(q, fl) == 1 and E.market_price(q, fl - 1) > 1
            tally.check("product.floor_distance", "theorem F_p + engine market_price",
                        ok and float(pr[i, PN["floor_distance"]]) == fl - inv)
    # ---- tiles (both farms, from each owner's view; rival views checked by symmetry above)
    sim = None
    if s % 24 in sim_hours:
        sim = {q: market["prices"][q] for q in PRODUCTS}
    for fi in range(2):
        _check_tiles(E, tally, s, raw[fi]["farms"][fi], feats[fi]["tile_own"], sim)
        # aggregates equal the tile columns
        block, g, pr, st = feats[fi]["tile_own"], feats[fi]["global"], feats[fi]["product"], feats[fi]["structure"]
        farm = raw[fi]["farms"][fi]
        tally.eq("global.own_unserviced_today", "sum of tile column", g[GN["own_unserviced_today"]],
                 block[:, T["service_needed_today"]].sum())
        tally.eq("global.own_dying_tonight", "sum of tile column", g[GN["own_dying_tonight"]],
                 block[:, T["dies_tonight_if_unserviced"]].sum())
        tally.eq("global.own_harvestable_units", "sum of tile column", g[GN["own_harvestable_units"]],
                 block[:, T["harvestable_now"]].sum())
        tally.eq("global.own_uncared_animals", "sum of tile column", g[GN["own_uncared_animals"]],
                 block[:, T["care_needed_today"]].sum())
        for i, q in enumerate(PRODUCTS):
            idx = [y * 10 + x for y in range(10) for x in range(10)
                   if _alive(farm["tiles"][y][x]) and _product_of(E, farm["tiles"][y][x]) == q]
            if q != "FERTILIZER":
                tally.eq("product.own_field_stock", "sum of engine yield_units", pr[i, PN["own_field_stock"]],
                         sum(farm["tiles"][j // 10][j % 10]["yield_units"] for j in idx))
            tally.eq("product.own_producers", "engine tile count", pr[i, PN["own_producers"]], len(idx))
            tally.eq("product.own_output_tonight", "sum of tile column", pr[i, PN["own_output_tonight"]],
                     block[idx, T["gain_tonight_current"]].sum() if idx else 0)
            tally.eq("product.own_season_units", "sum of tile column", pr[i, PN["own_season_units"]],
                     block[idx, T["season_units_remaining"]].sum() if idx else 0)
        for r, name in enumerate(TF.PRODUCERS):
            idx = [y * 10 + x for y in range(10) for x in range(10)
                   if _alive(farm["tiles"][y][x]) and farm["tiles"][y][x].get("crop", farm["tiles"][y][x].get("animal")) == name]
            tally.eq("structure.own_count", "engine tile count", st[r, SN["own_count"]], len(idx))
            tally.eq("structure.own_stored_total", "sum of engine yield_units", st[r, SN["own_stored_total"]],
                     sum(farm["tiles"][j // 10][j % 10]["yield_units"] for j in idx))
            tally.eq("structure.own_season_units", "sum of tile column", st[r, SN["own_season_units"]],
                     block[idx, T["season_units_remaining"]].sum() if idx else 0)
            tally.eq("structure.rival_count", "own(seat) == rival(other seat)", feats[1 - fi]["structure"][r, SN["rival_count"]], len(idx))
            tally.eq("structure.rival_stored_total", "own(seat) == rival(other seat)",
                     feats[1 - fi]["structure"][r, SN["rival_stored_total"]], st[r, SN["own_stored_total"]])
            tally.eq("structure.rival_season_units", "own(seat) == rival(other seat)",
                     feats[1 - fi]["structure"][r, SN["rival_season_units"]], st[r, SN["own_season_units"]])
        for c in ("unserviced_today", "dying_tonight", "harvestable_units", "uncared_animals", "fertilizer_collectable"):
            tally.eq("global.rival_" + c, "own(seat) == rival(other seat)", feats[1 - fi]["global"][GN["rival_" + c]],
                     g[GN["own_" + c]])
        tally.eq("global.rival_next_hire_cost", "own(seat) == rival(other seat)",
                 feats[1 - fi]["global"][GN["rival_next_hire_cost"]], g[GN["own_next_hire_cost"]])
        # ongoing decay-start prediction, compared when the engine later sets max_lifespan_step
        for y in range(10):
            for x in range(10):
                t = farm["tiles"][y][x]
                if isinstance(t, dict) and t.get("kind") == "PLANT" and E.CROPS[t["crop"]]["ongoing"]:
                    key = (fi, x, y, t["crop"], t["planted_day"])
                    row = block[y * 10 + x]
                    if t["max_lifespan_step"] < 0:
                        pred = s + row[T["steps_to_next_decay"]] if row[T["decay_scheduled"]] else None
                        identity_mls.setdefault(key, pred)
                    elif key in identity_mls:
                        pred, mls = identity_mls.pop(key), t["max_lifespan_step"]
                        m = "engine sets max_lifespan_step at the 4th event (ongoing)"
                        tally.check("tile.decay_scheduled", m, (pred is not None) == (mls <= LAST), {"mls": mls})
                        if pred is not None:
                            tally.eq("tile.steps_to_next_decay", m, pred, mls)


def _check_decay(tally, s, fi, block, farm_at_s, decayed):
    """Predicted decay at step s vs the engine's _decay_plants at step s (same plant identity)."""
    for (x, y), (crop, pd, happened) in decayed.items():
        t = farm_at_s["tiles"][y][x]
        if not (isinstance(t, dict) and t.get("kind") == "PLANT" and t["crop"] == crop and t["planted_day"] == pd):
            continue
        row = block[y * 10 + x]
        predicted = bool(row[T["decay_scheduled"]]) and row[T["steps_to_next_decay"]] == 0
        tally.check("tile.steps_to_next_decay", "engine _decay_plants during replay", predicted == happened,
                    {"step": s, "farm": fi, "x": x, "y": y})
        tally.check("tile.decay_scheduled", "engine _decay_plants during replay", (not happened) or bool(row[T["decay_scheduled"]]))


def _ticks(a, b, n):
    return b // n - (a - 1) // n if b >= a else 0


def _check_windows(tally, history, rec):
    """Checks that need what the engine did after the observation."""
    refresh = sorted(rec.refresh_steps)
    shops_final = history[-1]["shops"]
    unlock_step = []
    for j in range(len(shops_final)):
        unlock_step.append(next(t for t, h in enumerate(history) if len(h["shops"]) > j))
    mult = TF._SHOP_MULT
    for s in range(0, LAST + 1):
        h = history[s]
        for p in range(2):
            g, pr = h["feats"][p]["global"], h["feats"][p]["product"]
            # day-end refresh timing
            nxt = next((r for r in refresh if r >= s), None)
            if g[GN["refresh_tonight"]]:
                ok = nxt == s + g[GN["day_actions_remaining"]] - 1
            else:
                ok = nxt is None and g[GN["day_actions_remaining"]] == LAST - s + 1
            tally.check("global.refresh_tonight", "engine _end_of_day calls", ok, {"s": s})
            tally.check("global.day_actions_remaining", "engine _end_of_day calls", ok, {"s": s})
            # shops
            future = [u for u in unlock_step if u > s]
            tally.eq("global.shop_unlocks_remaining", "later observations", g[GN["shop_unlocks_remaining"]], len(future))
            tally.eq("global.steps_to_next_shop_unlock", "later observations", g[GN["steps_to_next_shop_unlock"]],
                     future[0] - s if future else 0)
            # hands hired at s act until the refresh
            if s < LAST and history[s + 1]["hires"][p] > h["hires"][p]:
                n_after = history[s + 1]["hands"][p]
                t = s + 1
                while t <= LAST and history[t]["hands"][p] >= n_after:
                    t += 1
                tally.eq("global.hire_useful_actions", "hands present in later observations",
                         g[GN["hire_useful_actions"]], t - (s + 1))
            # one-step town removal from tick flags and visible rates
            rem = rec.town[s]
            for i, q in enumerate(PRODUCTS):
                pred = pr[i, PN["visible_rate"]] * g[GN["shop_tick_now"]] + (q != "FERTILIZER") * g[GN["center_tick_now"]]
                tally.eq("product.visible_rate", "engine _town_consume at this step", pred, rem[i])
                tally.check("global.shop_tick_now", "engine _town_consume at this step", pred == rem[i])
                tally.check("global.center_tick_now", "engine _town_consume at this step", pred == rem[i])
            # next ticks: MELON is consumed only by the town centre; shop ticks seen when shops are visible
            melon = PRODUCTS.index("MELON")
            nc = next((t for t in range(s, LAST + 1) if rec.town[t][melon] > 0), None)
            if nc is not None:
                tally.eq("global.steps_to_center_tick", "engine _town_consume later", g[GN["steps_to_center_tick"]], nc - s)
            if h["shops"]:
                ns = next((t for t in range(s, LAST + 1)
                           if sum(rec.town[t]) - (8 if t % 24 == 0 else 0) > 0), None)
                if ns is not None:
                    tally.eq("global.steps_to_shop_tick", "engine _town_consume later", g[GN["steps_to_shop_tick"]], ns - s)
            # window demand: visible + realised new shops == engine removal
            w_end = min(s + 23, LAST)
            new_shops = [(shops_final[j], unlock_step[j]) for j in range(len(shops_final)) if unlock_step[j] > s]
            for i, q in enumerate(PRODUCTS):
                act24 = sum(rec.town[t][i] for t in range(s, w_end + 1))
                act_rest = sum(rec.town[t][i] for t in range(s, LAST + 1))
                rev24 = sum(mult[sh].get(q, 0) * _ticks(u, w_end, 4) for sh, u in new_shops)
                rev_rest = sum(mult[sh].get(q, 0) * _ticks(u, LAST, 4) for sh, u in new_shops)
                tally.eq("product.demand24_visible", "engine _town_consume over the window (+ realised new shops)",
                         pr[i, PN["demand24_visible"]] + rev24, act24)
                tally.eq("product.demand_rest_visible", "engine _town_consume to the end (+ realised new shops)",
                         pr[i, PN["demand_rest_visible"]] + rev_rest, act_rest)
                tally.check("product.demand_rest_max_new", "upper bound on engine removal to the end",
                            act_rest <= pr[i, PN["demand_rest_visible"]] + pr[i, PN["demand_rest_max_new"]])
                ticks24 = sum(_ticks(u, w_end, 4) for _, u in new_shops)
                ticks_rest = sum(_ticks(u, LAST, 4) for _, u in new_shops)
                tally.eq("product.demand24_expected_new", "prior rate x realised unlock ticks",
                         pr[i, PN["demand24_expected_new"]], TF._EXPECTED_RATE[i] * ticks24)
                tally.eq("product.demand_rest_expected_new", "prior rate x realised unlock ticks",
                         pr[i, PN["demand_rest_expected_new"]], TF._EXPECTED_RATE[i] * ticks_rest)
                # passive quote in 24 steps: exact when no trade on q and no unlock in the window
                if s + 24 <= LAST + 1 and not rev24 and all(rec.market[t][i] == 0 for t in range(s, s + 24)) \
                        and not any(u <= s + 23 for _, u in new_shops):
                    tally.eq("product.quote_after_demand24", "engine price 24 steps later (no trades, no unlock)",
                             pr[i, PN["quote_after_demand24"]], history[s + 24]["prices"][q])
                if pr[i, PN["floor_locked"]]:
                    later = [history[t]["prices"][q] for t in range(s, len(history))]
                    tally.check("product.floor_locked", "every later engine quote is $1", all(v == 1 for v in later))
                else:
                    tally.check("product.floor_locked", "not locked (no claim)", True)


# ----------------------------------------------------------------------------- tests
_REPORTS = {}


def _report(episode=EPISODE):
    if episode not in _REPORTS:
        _REPORTS[episode] = replay_report(episode)
    return _REPORTS[episode]


def test_constants_match_engine():
    E = load_engine()
    assert TF.CROPS == E.CROPS and TF.ANIMALS == E.ANIMALS and TF.MARKET_PARAMS == E.MARKET_PARAMS
    assert TF.SHOPS == E.SHOPS and list(TF.LAND_PRICES) == E.LAND_PRICES and TF.HINGE_GAIN == E.HINGE_GAIN
    assert TF.MAX_SHOP_INSTANCES == E.MAX_SHOP_INSTANCES and list(TF.PRODUCTS) == E.PRODUCTS
    assert all(TF._fib(n) == E._fib(n) for n in range(40))


def test_price_and_curve_tables_match_engine():
    E = load_engine()
    curves = TF._curves(TF.MARKET_PARAMS)
    for i, q in enumerate(TF.PRODUCTS):
        for inv in list(range(9000, 11200)) + [0, 5000, 7000, 7951, 7952, 12048, 12049, 15000, 30000]:
            assert curves[i].quote(inv) == E.market_price(q, inv), (q, inv)
        for a, b in ((9950, 10050), (7900, 8000), (12000, 12100), (10000, 10000)):
            assert curves[i].total(a, b) == sum(E.market_price(q, v) for v in range(a, b))


def test_floor_inventories_match_theorem():
    for i, q in enumerate(TF.PRODUCTS):
        assert TF._curves(TF.MARKET_PARAMS)[i].floor == FLOOR_THEOREM[q], q


def test_season_closed_forms_match_unit_economics():
    for name in TF.PRODUCERS:
        for days in range(1, 31):
            if name in TF.CROPS:
                for pol, fert in (("watered", False), ("watered_fertilized", True)):
                    ref = unit_economics.formula(name, pol, days)
                    units, cycles, _ = TF.season_formula(name, fert, days)
                    assert (units, cycles) == (ref["units"], ref["cycles"]), (name, pol, days)
            else:
                ref = unit_economics.formula(name, "fed_cared", days)
                units, events, first = TF.season_formula(name, False, days)
                assert (units, events, first) == (ref["units"], ref["events"], ref["first_output_day"])


def test_spec_shapes_and_scales():
    for block, names in TF.NAMES.items():
        assert len(TF.SCALES[block]) == len(names) and (TF.SCALES[block] > 0).all()
    assert len(set(TF.GLOBAL_NAMES)) == len(TF.GLOBAL_NAMES)
    assert all(len(r) == 4 and r[2] for rows in TF.SPEC.values() for r in rows)


@pytest.mark.parametrize("episode", EPISODES, ids=[e.stem for e in EPISODES])
def test_replay_matches_recording_and_every_feature_is_exact(episode):
    report = _report(episode)
    assert report["replay_mismatched_observations"] == 0
    bad = {k: v for k, v in report["features"].items() if v["mismatch"]}
    assert not bad, json.dumps(bad, indent=1)[:4000]


def test_every_feature_column_is_engine_checked():
    checked = {k for k, v in _report(EPISODE)["features"].items() if v["checked"]}
    expected = ({"global." + n for n in TF.GLOBAL_NAMES} | {"product." + n for n in TF.PRODUCT_NAMES}
                | {"tile." + n for n in TF.TILE_NAMES} | {"structure." + n for n in TF.STRUCTURE_NAMES})
    missing = expected - checked
    assert not missing, sorted(missing)


def test_features_depend_only_on_the_actor_view():
    data = json.loads(EPISODE.read_text())
    obs = copy.deepcopy(data["steps"][300][0]["observation"])
    obs["step"] = 300
    base = TF.compute(actor_observation(obs), data["configuration"])
    noisy = copy.deepcopy(obs)
    noisy["remainingOverageTime"] = 1.0
    noisy["seed"] = 123
    other = TF.compute(actor_observation(noisy), data["configuration"])
    for k in base:
        assert np.array_equal(base[k], other[k])


def test_cpu_budget():
    report = _report()
    assert report["cpu_us_per_call"]["mean"] < 5000, report["cpu_us_per_call"]


if __name__ == "__main__":
    print(json.dumps([replay_report(e) for e in EPISODES], indent=1))
