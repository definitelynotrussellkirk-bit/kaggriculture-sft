"""Tests for history_features: exactness against engine instrumentation, day boundaries,
episode start/gaps, causality, actor-visibility, previous action and CPU cost.

Run (pinned interpreter, CPU only):
  cd /home/russ/Desktop/farmNEAT && CUDA_VISIBLE_DEVICES= \
  PYTHONPATH=src:work/runs/maintainer/primitive-bc:work/cache/pytest-path \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  work/venv/python -B -m pytest -q work/runs/maintainer/primitive-bc/test_history_features.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LEAGUE_DIR = ROOT / "work/runs/reviewer-play/opponent-league-adapters"
for _p in (ROOT / "src", HERE, LEAGUE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import history_features as H  # noqa: E402
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment, load_engine  # noqa: E402

PRODUCTS = H.PRODUCTS


# ----------------------------------------------------------------------------- game harness
def _agents(kind):
    engine = load_engine()
    starter = lambda o, c: engine.starter_agent(o)  # noqa: E731
    if kind == "pass":
        return [lambda o, c: {"farmer": ["PASS"], "hands": [], "market": []}] * 2, "pass"
    try:
        from league import League
        league = League()
        if kind == "mixed":   # asymmetric: league router versus the engine's carrot starter
            return [league.new_actor("SHOP_ROUTER", seat=0, sampler_seed=3), starter], "league+engine"
        return [league.new_actor("SHOP_ROUTER", seat=0, sampler_seed=1),
                league.new_actor("K0006", seat=1, sampler_seed=2)], "league"
    except Exception:  # league assets absent: engine's own scripted agents
        return [starter, lambda o, c: engine.random_agent(o)], "engine"


def play(kind, seed=7):
    """Full game with engine instrumentation of market, town consumption and commits."""
    engine = load_engine()
    agents, source = _agents(kind)
    env = ExactKaggricultureEnvironment(seed=seed)
    truth = {"cons": [], "market_delta": [], "supply": [], "holdings": []}
    originals = {n: getattr(engine, n) for n in ("_process_market", "_town_consume", "_commit_unit")}
    current = {}

    def process_market(state, env_):
        farms = state[0].observation.farms
        current["farms"] = farms
        current["supply"] = np.zeros((2, len(PRODUCTS)))
        before = dict(state[0].observation.market["inventory"])
        originals["_process_market"](state, env_)
        after = state[0].observation.market["inventory"]
        truth["market_delta"].append(np.array([after[p] - before[p] for p in PRODUCTS], float))

    def town_consume(env_, state, step):
        before = dict(state[0].observation.market["inventory"])
        originals["_town_consume"](env_, state, step)
        after = state[0].observation.market["inventory"]
        truth["cons"].append(np.array([before[p] - after[p] for p in PRODUCTS], float))

    def commit_unit(op, item, price, farm, private, market, shed_capacity=100):
        before = market["inventory"].get(item) if item in PRODUCTS else None
        ok = originals["_commit_unit"](op, item, price, farm, private, market, shed_capacity)
        if before is not None:
            player = 0 if farm is current["farms"][0] else 1
            current["supply"][player, PRODUCTS.index(item)] += market["inventory"][item] - before
        return ok

    engine._process_market, engine._town_consume, engine._commit_unit = process_market, town_consume, commit_unit
    observations, actions = ([], []), ([], [])
    try:
        while not env.done:
            obs = [env.agent_observation(p) for p in range(2)]
            acts = [agents[p](copy.deepcopy(obs[p]), env.configuration) for p in range(2)]
            for p in range(2):
                observations[p].append(obs[p])
                actions[p].append(copy.deepcopy(acts[p]))
            truth["holdings"].append(np.array([[
                env.state[q].observation.private["shed"].get(item, 0)
                + sum(inv.get(item, 0) for inv in env.state[q].observation.private["inventories"])
                for item in PRODUCTS] for q in range(2)], float))
            env.step(acts)
            truth["supply"].append(current["supply"].copy())
    finally:
        for name, fn in originals.items():
            setattr(engine, name, fn)
    return {"obs": observations, "actions": actions, "truth": truth, "cfg": env.configuration,
            "source": source}


@pytest.fixture(scope="module")
def game():
    return play("trade")


@pytest.fixture(scope="module")
def quiet():
    return play("pass")


@pytest.fixture(scope="module")
def mixed():
    return play("mixed", seed=11)


def run_tracker(g, seat, record=True):
    tracker = H.HistoryFeatures()
    outs = []
    for t, obs in enumerate(g["obs"][seat]):
        outs.append(tracker.update(obs, g["cfg"]))
        if record:
            tracker.record_action(g["actions"][seat][t])
    return outs


def field(out, block, row, name):
    rows, fields, values, known = out.blocks[block]
    r, f = rows.index(row), fields.index(name)
    scale = dict(H.BLOCKS[block][1])[name]
    # round away float32 storage noise (integer-valued fields compare exactly)
    return round(float(values[r, f]) * scale, 4), bool(known[r, f])


# ----------------------------------------------------------------------------- schema
def test_schema_matches_observation_items_and_token_width():
    from farm_neat.policy import observation
    assert H.ITEMS == observation.ITEMS
    for rows, fields in H.BLOCKS.values():
        assert len(fields) <= observation.WIDTH - observation.PAYLOAD
        assert len({n for n, _ in fields}) == len(fields)
    assert len(set(H.flat_names())) == len(H.flat_names())


def test_values_finite_and_unknown_is_zero(game):
    for seat in (0, 1):
        for out in run_tracker(game, seat):
            names, values, known = out.flat()
            assert names == H.flat_names()
            assert values.dtype == np.float32 and np.isfinite(values).all()
            assert not np.any(values[~known])


# ----------------------------------------------------------------------------- exactness
def test_town_consumption_and_trade_flow_exact_against_engine(game):
    assert game["source"] in ("league", "engine")
    cons, delta, supply = (np.array(game["truth"][k]) for k in ("cons", "market_delta", "supply"))
    for seat in (0, 1):
        outs = run_tracker(game, seat)
        for t in range(1, len(outs)):
            for i, p in enumerate(PRODUCTS):
                c, kc = field(outs[t], "history_item_lag", p, "lag1.town_consumed")
                f, kf = field(outs[t], "history_item_lag", p, "lag1.trade_flow")
                assert kc and kf
                assert c == pytest.approx(cons[t - 1, i], abs=1e-3)
                assert f == pytest.approx(delta[t - 1, i], abs=1e-3)
                assert f == pytest.approx(supply[t - 1, :, i].sum(), abs=1e-3)
    # at least one real trade happened, so the test is not vacuous
    assert np.abs(delta).sum() > 0 and cons.sum() > 0


def test_pass_game_has_zero_trade_flow_and_zero_rival_stock(quiet):
    outs = run_tracker(quiet, 0)
    for out in outs[1:]:
        for p in PRODUCTS:
            assert field(out, "history_item_lag", p, "lag1.trade_flow")[0] == pytest.approx(0, abs=1e-3)
            assert field(out, "history_item_now", p, "rival.stock_est") == (0.0, True)


def own_and_rival_trade_accuracy(g):
    """Exact-match fraction of lag-1 own/rival trade estimates versus engine truth, over all
    (step, product) cells and over cells where that player's true supply change is nonzero."""
    supply = np.array(g["truth"]["supply"])
    rows = {"own": [], "rival": [], "own_nonzero": [], "rival_nonzero": []}
    for seat in (0, 1):
        outs = run_tracker(g, seat)
        for t in range(1, len(outs)):
            for i, p in enumerate(PRODUCTS):
                for who, player in (("own", seat), ("rival", 1 - seat)):
                    est = field(outs[t], "history_item_lag", p, f"lag1.{who}.trade_est")[0]
                    truth = supply[t - 1, player, i]
                    rows[who].append(abs(est - truth) < 1e-3)
                    if truth != 0:
                        rows[who + "_nonzero"].append(abs(est - truth) < 1e-3)
    return {k: (round(float(np.mean(v)), 4), len(v)) for k, v in rows.items()}


def rival_stock_error(g):
    """Seat s tracker's public rival.stock_est versus the rival's true shed+carried holdings."""
    holdings = np.array(g["truth"]["holdings"])      # [t, seat, product]
    errors = []
    for seat in (0, 1):
        outs = run_tracker(g, seat)
        for t, out in enumerate(outs):
            for i, p in enumerate(PRODUCTS):
                est, known = field(out, "history_item_now", p, "rival.stock_est")
                assert known
                errors.append(est - holdings[t, 1 - seat, i])
    errors = np.abs(np.array(errors))
    truth = np.abs(np.concatenate([holdings[:, 1, :].reshape(-1), holdings[:, 0, :].reshape(-1)]))
    return {"cells": int(errors.size), "exact_fraction": round(float(np.mean(errors < 1e-3)), 4),
            "mean_abs": round(float(errors.mean()), 3), "max_abs": float(errors.max()),
            "zero_baseline_exact": round(float(np.mean(truth < 1e-3)), 4),
            "zero_baseline_mean_abs": round(float(truth.mean()), 3)}


@pytest.mark.parametrize("which", ["game", "mixed"])
def test_trade_decomposition_and_rival_stock_estimates(which, request, capsys):
    g = request.getfixturevalue(which)
    acc = own_and_rival_trade_accuracy(g)
    stock = rival_stock_error(g)
    with capsys.disabled():
        print(f"\n[history] {which} source={g['source']} lag1 trade_est exact (fraction, cells) {acc}"
              f"\n[history] {which} rival.stock_est vs rival true shed+carried {stock}")
    # Estimates, not exact.  Guards: well below the measured values and strictly better than
    # predicting zero rival stock.
    assert acc["own"][0] >= 0.99 and acc["rival"][0] >= 0.99
    assert stock["mean_abs"] < stock["zero_baseline_mean_abs"]
    assert stock["exact_fraction"] >= 0.75


# ----------------------------------------------------------------------------- deltas / days
def _raw(obs):
    seat = obs["player"]
    own, rival = obs["farms"][seat], obs["farms"][1 - seat]
    carried = sum(sum(inv.values()) for inv in obs["private"]["inventories"])
    return {"d.own_cash": own["money"], "d.rival_cash": rival["money"],
            "d.own_units": 1 + len(own["hands"]), "d.rival_units": 1 + len(rival["hands"]),
            "d.own_carried_total": carried,
            "d.own_shed_total": sum(obs["private"]["shed"].values())}


def test_lag_deltas_equal_literal_differences_across_day_boundaries(game):
    obs_seq = game["obs"][0]
    outs = run_tracker(game, 0)
    boundaries = 0
    for t, out in enumerate(outs):
        for lag in H.LAGS:
            row = f"lag{lag}"
            exact = field(out, "history_global_lag", row, "lag.exact")[0]
            age = field(out, "history_global_lag", row, "lag.age")[0] * lag
            then = max(0, t - lag)
            assert age == pytest.approx(t - then, abs=0.01)
            assert exact == (t >= lag)
            day_ends = field(out, "history_global_lag", row, "lag.day_ends")[0]
            assert day_ends == obs_seq[t]["day"] - obs_seq[then]["day"]
            for name, value in _raw(obs_seq[t]).items():
                got, known = field(out, "history_global_lag", row, name)
                assert known == (t > 0)
                if t > 0:
                    assert got == pytest.approx(value - _raw(obs_seq[then])[name], abs=1e-3)
        if t > 0 and obs_seq[t]["hour"] == 0:
            boundaries += 1
            # day-end reset is visible in the literal lag-1 deltas
            assert field(out, "history_global_lag", "lag1", "lag.day_ends")[0] == 1
            assert field(out, "history_global_now", "now", "phase.hour_cos")[0] == 1
            assert obs_seq[t]["farms"][0]["hands"] == []
            assert all(not inv for inv in obs_seq[t]["private"]["inventories"])
        if obs_seq[t]["hour"] == 23 and t < 718:
            assert field(out, "history_global_now", "now", "phase.day_end_follows")[0] == 1
    assert boundaries == 29


def test_phase_encodings(game):
    outs = run_tracker(game, 1)
    for t, out in enumerate(outs):
        assert field(out, "history_global_now", "now", f"phase.tick4_{t % 4}")[0] == 1
        assert field(out, "history_global_now", "now", f"phase.day_mod3_{(t // 24) % 3}")[0] == 1
        s, _ = field(out, "history_global_now", "now", "phase.hour_sin")
        c, _ = field(out, "history_global_now", "now", "phase.hour_cos")
        assert s == pytest.approx(np.sin(2 * np.pi * (t % 24) / 24), abs=1e-3)
        assert c == pytest.approx(np.cos(2 * np.pi * (t % 24) / 24), abs=1e-3)
    assert field(outs[718], "history_global_now", "now", "phase.final_action")[0] == 1


def test_return_trip_stranded_and_feed_cover_match_direct_recount(game):
    access = ((4, 4), (5, 4), (4, 5), (5, 5))
    stranded_seen = loaded_seen = 0
    for seat in (0, 1):
        outs = run_tracker(game, seat)
        for t, (obs, out) in enumerate(zip(game["obs"][seat], outs)):
            farm = obs["farms"][seat]
            units = [farm["farmer"], *farm["hands"]]
            total = longest = stranded = 0
            for (x, y), inv in zip(units, obs["private"]["inventories"]):
                load = sum(inv.values())
                if load:
                    need = min(abs(x - a) + abs(y - b) for a, b in access) + 1
                    total, longest = total + need, max(longest, need)
                    if t >= 696 and need > 718 - t + 1:      # day 29 has no day-end deposit
                        stranded += load
            g = lambda n: field(out, "history_global_now", "now", n)  # noqa: E731
            assert g("own.loaded_return_steps")[0] == pytest.approx(total, abs=1e-3)
            assert g("own.max_return_steps")[0] == pytest.approx(longest, abs=1e-3)
            assert g("own.stranded_carried")[0] == pytest.approx(stranded, abs=1e-3)
            loaded_seen += total > 0
            stranded_seen += stranded > 0
            animals = sum(isinstance(c, dict) and "animal" in c for row in farm["tiles"] for c in row)
            wheat = obs["private"]["shed"].get("WHEAT", 0) + sum(
                inv.get("WHEAT", 0) for inv in obs["private"]["inventories"])
            cover, known = g("own.feed_cover_days")
            assert known == (animals > 0)
            if animals:
                assert cover == pytest.approx(wheat / animals, abs=1e-3)
    assert loaded_seen > 0


# ----------------------------------------------------------------------------- start / reset
def test_episode_start_has_no_history_and_new_episode_resets(game):
    tracker = H.HistoryFeatures()
    first = tracker.update(game["obs"][0][0], game["cfg"])
    assert field(first, "history_global_now", "now", "hist.from_start")[0] == 1
    assert field(first, "history_prev_action", "prev", "prev.known")[0] == 0
    for lag in H.LAGS:
        assert not field(first, "history_global_lag", f"lag{lag}", "d.own_cash")[1]
    for t in range(1, 60):
        tracker.record_action(game["actions"][0][t - 1])
        tracker.update(game["obs"][0][t], game["cfg"])
    again = tracker.update(game["obs"][0][0], game["cfg"])      # new episode, same seat
    fresh = H.HistoryFeatures().update(game["obs"][0][0], game["cfg"])
    _, v1, k1 = again.flat()
    _, v2, k2 = fresh.flat()
    assert np.array_equal(v1, v2) and np.array_equal(k1, k2)


def test_step_gap_resets_and_marks_estimates_unknown(game):
    tracker = H.HistoryFeatures()
    for t in range(0, 30):
        tracker.update(game["obs"][0][t], game["cfg"])
    out = tracker.update(game["obs"][0][40], game["cfg"])
    assert field(out, "history_global_now", "now", "hist.reset_gap")[0] == 1
    assert field(out, "history_global_now", "now", "hist.from_start")[0] == 0
    assert not field(out, "history_item_now", "MILK", "rival.stock_est")[1]
    assert not field(out, "history_global_lag", "lag1", "d.own_cash")[1]
    nxt = tracker.update(game["obs"][0][41], game["cfg"])
    assert field(nxt, "history_global_lag", "lag1", "d.own_cash")[1]
    assert not field(nxt, "history_item_now", "MILK", "rival.stock_est")[1]


def test_repeated_step_update_is_idempotent(game):
    full = run_tracker(game, 0)
    tracker = H.HistoryFeatures()
    for t in range(0, 80):
        obs = game["obs"][0][t]
        first = tracker.update(obs, game["cfg"])
        again = tracker.update(obs, game["cfg"])           # e.g. a retried agent call
        assert np.array_equal(first.flat()[1], again.flat()[1])
        assert np.array_equal(again.flat()[1], full[t].flat()[1])
        tracker.record_action(game["actions"][0][t])


def test_one_tracker_per_seat_and_seat_router(game):
    tracker = H.HistoryFeatures()
    tracker.update(game["obs"][0][0], game["cfg"])
    with pytest.raises(ValueError):
        tracker.update(game["obs"][1][1], game["cfg"])
    router = H.SeatHistories()
    for t in range(5):
        for seat in (0, 1):
            router.update(game["obs"][seat][t], game["cfg"])
            router.record_action(seat, game["actions"][seat][t])
    direct = run_tracker(game, 1)[4]
    routed = router._by_seat[1]._features()
    assert np.array_equal(direct.flat()[1], routed.flat()[1])


# ----------------------------------------------------------------------------- information
def test_actor_whitelist_and_extra_keys_do_not_change_features(game):
    from farm_neat.training.bc import actor_observation
    tracker_a, tracker_b = H.HistoryFeatures(), H.HistoryFeatures()
    for t in range(0, 120):
        obs = game["obs"][0][t]
        noisy = copy.deepcopy(dict(obs))
        noisy["seed"] = 123456 + t
        noisy["future_shops"] = ["BAKERY"] * 8
        noisy["rival_private"] = {"shed": {"MILK": 99}}
        noisy["farms"] = [dict(f, secret=t) for f in noisy["farms"]]
        a = tracker_a.update(actor_observation(obs), game["cfg"])
        b = tracker_b.update(noisy, game["cfg"])
        tracker_a.record_action(game["actions"][0][t])
        tracker_b.record_action(game["actions"][0][t])
        assert np.array_equal(a.flat()[1], b.flat()[1])
        assert np.array_equal(a.flat()[2], b.flat()[2])


def test_features_are_causal_prefix_invariant(game):
    full = run_tracker(game, 0)
    for cut in (1, 23, 24, 25, 200):
        short = H.episode_features(game["obs"][0][:cut], game["actions"][0][:cut], game["cfg"])
        assert np.array_equal(short[-1].flat()[1], full[cut - 1].flat()[1])
    # the rival's actions never enter; only observations and own actions do
    import inspect
    source = inspect.getsource(H)
    for forbidden in ("\"seed\"", "rival_action", "env.info", "next_obs"):
        assert forbidden not in source


# ----------------------------------------------------------------------------- previous action
def test_previous_action_counts_and_unknown_without_record(game):
    tracker = H.HistoryFeatures()
    tracker.update(game["obs"][0][0], game["cfg"])
    action = {"farmer": ["NORTH"], "hands": [["PLANT", "WHEAT"]],
              "market": [["SELL", "MILK", 3], ["BUY_SEED", "CARROT", 2], ["HIRE"], ["BUY_LAND"],
                         ["BUY_PRODUCT", "WHEAT", 5], ["BOGUS"]]}
    tracker.record_action(action)
    out = tracker.update(game["obs"][0][1], game["cfg"])
    g = lambda name: field(out, "history_prev_action", "prev", name)[0]  # noqa: E731
    assert g("prev.known") == 1 and g("prev.op.NORTH") == 1
    assert g("prev.units_commanded") == 1                  # one unit existed at step 0
    assert g("prev.op.PLANT") == 0                         # hand command beyond units ignored
    assert g("prev.market_orders") == 5 and g("prev.hire_orders") == 1
    assert g("prev.land_orders") == 1 and g("prev.malformed") == 1
    assert field(out, "history_item_now", "MILK", "prev.sell") == (3.0, True)
    assert field(out, "history_item_now", "CARROT", "prev.buy_seed") == (2.0, True)
    assert field(out, "history_item_now", "WHEAT", "prev.buy_product") == (5.0, True)
    assert not field(out, "history_item_now", "MILK", "prev.buy_seed")[1]
    nxt = tracker.update(game["obs"][0][2], game["cfg"])    # no record for step 1
    assert field(nxt, "history_prev_action", "prev", "prev.known") == (0.0, True)
    assert not field(nxt, "history_item_now", "MILK", "prev.sell")[1]


def test_quantities_clamped_like_engine_and_codec_tokens_when_available(game):
    tracker = H.HistoryFeatures()
    obs0 = game["obs"][0][0]
    tracker.update(obs0, game["cfg"])
    action = {"farmer": ["PICKUP", "MILK", 5000], "hands": [],
              "market": [["SELL", "MILK", 1000], ["BUY_SEED", "WHEAT", 700]]}
    tracker.record_action(action)
    out = tracker.update(game["obs"][0][1], game["cfg"])
    assert field(out, "history_item_now", "MILK", "prev.sell")[0] == 100
    assert field(out, "history_item_now", "MILK", "prev.pickup")[0] == 100
    assert field(out, "history_item_now", "WHEAT", "prev.buy_seed")[0] == 700
    codec = H._codec()
    if codec is None:
        assert out.prev_tokens is None
    else:
        assert out.prev_tokens == codec.encode_step(action, obs0)
    first = H.HistoryFeatures().update(obs0, game["cfg"])
    assert first.prev_tokens is None


# ----------------------------------------------------------------------------- cost
def test_cpu_cost_per_step(game, capsys):
    stats = H.benchmark(game["obs"][0], game["actions"][0], game["cfg"], repeat=2)
    with capsys.disabled():
        print(f"\n[history] cost {stats}")
    assert stats["mean_ms"] < 1.5
