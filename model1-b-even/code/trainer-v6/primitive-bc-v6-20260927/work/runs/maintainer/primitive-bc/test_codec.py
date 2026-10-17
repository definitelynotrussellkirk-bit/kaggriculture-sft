"""Codec contract tests: generated vocabulary, forward/backward one-to-one,
rounding, unit/queue bounds, engine-equivalence of canonical() and replay round trip."""
from __future__ import annotations

import copy
import json
import random
import zipfile
from pathlib import Path

import pytest

import codec
from farm_neat.game.exact_engine import load_engine
from farm_neat.training import bc

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def engine():
    return load_engine()


@pytest.fixture(scope="module")
def replay_task():
    return codec.sample_tasks(1)[10]


@pytest.fixture(scope="module")
def replay(replay_task):
    with zipfile.ZipFile(replay_task[0]) as z:
        return json.loads(z.read(replay_task[1]))


@pytest.fixture(scope="module")
def mid_env(replay):
    """Exact engine at a real mid-game boundary with >= 3 hands and shed stock (seat 0)."""
    return codec.mid_game_env(replay)[0]


def view(env, seat=0):
    return bc.actor_observation(env.agent_observation(seat))


def state(env):
    return codec._state_wire(env)


def step_copy(env, actions):
    twin = codec.clone_env(env)
    twin.step(actions)
    return twin


PASS = {"farmer": ["PASS"], "hands": [], "market": []}


# ---------------------------------------------------------------- vocabulary
def test_grammar_tables_equal_engine_source_and_module(engine):
    g = codec.engine_grammar()
    g.pop("lockstep_ops")
    assert g == codec.GRAMMAR
    assert codec.CROPS == {k: v["seed"] for k, v in engine.CROPS.items()}
    assert codec.ANIMALS == {k: v["cost"] for k, v in engine.ANIMALS.items()}
    assert list(codec.PRODUCTS) == engine.PRODUCTS
    assert list(codec.MOVES) == list(engine.FARMER_MOVES)
    assert list(codec.SHED_ITEMS) == list(engine._new_private()["shed"])


def test_vocab_is_generated_from_engine_grammar():
    unit, order = codec.vocab_from_grammar(codec.engine_grammar())
    assert unit == codec.UNIT_VOCAB and order == codec.ORDER_VOCAB
    assert len(set(unit)) == len(unit) and len(set(order)) == len(order)
    assert codec.ORDER_VOCAB[codec.STOP] == "STOP" and codec.UNIT_VOCAB[codec.PASS] == "PASS"


def test_vocab_json_matches_codec():
    vocab = json.loads((HERE / "VOCAB.json").read_text())
    assert vocab["vocab"]["UNIT_VOCAB"] == codec.UNIT_VOCAB
    assert vocab["vocab"]["ORDER_VOCAB"] == codec.ORDER_VOCAB
    assert [tuple(b) for b in vocab["vocab"]["QTY_BINS"]] == list(codec.QTY_BINS)
    assert [tuple(b) for b in vocab["chosen_bins"]] == list(codec.QTY_BINS)
    assert vocab["vocab"]["MAX_UNITS"] == codec.MAX_UNITS


# ---------------------------------------------------------------- quantity bins
def test_percentage_rounding_is_exact_round_half_up():
    saved = codec.QTY_BINS
    try:
        codec.QTY_BINS = (("p", 1, 2), ("p", 1, 3), ("p", 2, 3), ("p", 1, 1)) + saved[4:]
        assert codec.bin_value(0, 3) == 2        # 1.5 -> 2 (half up)
        assert codec.bin_value(0, 5) == 3        # 2.5 -> 3
        assert codec.bin_value(0, 0) == 1        # never 0
        assert codec.bin_value(0, 1) == 1        # 0.5 -> 1
        assert codec.bin_value(1, 5) == 2        # 1.667 -> 2
        assert codec.bin_value(1, 4) == 1        # 1.333 -> 1
        assert codec.bin_value(2, 4) == 3        # 2.667 -> 3
        assert codec.bin_value(3, 37) == 37
        assert codec.decode_qty("SELL", 3, 250) == 100      # clamped to the shed capacity
        assert codec.decode_qty("BUY_SEED", 3, 250) == 250  # seeds are not clamped
    finally:
        codec.QTY_BINS = saved
    assert len(codec.QTY_BINS) == codec.QTY_BIN_COUNT == 75
    assert len(set(codec.QTY_BINS)) == 75


def test_encode_prefers_discrete_bin():
    for b, (kind, num, den) in enumerate(codec.QTY_BINS):
        if kind == "n" and num <= 100:
            assert codec.encode_qty("SELL", num, num) == (b, True)


# ---------------------------------------------------------------- forward
def test_forward_every_engine_form_round_trips(mid_env):
    obs = view(mid_env)
    report, mismatches, gaps = codec.forward_coverage(obs, refs=(0, 3, 37, 100), seed_refs=(400,))
    assert mismatches == []
    for token, (ok, total) in report.items():
        has_qty = (codec.UNIT_HAS_QTY[codec.UNIT_INDEX[token]] if token in codec.UNIT_INDEX
                   else codec.ORDER_HAS_QTY[codec.ORDER_INDEX[token]])
        assert ok == total if not has_qty else ok > 0, token
    discrete = sorted(v for kind, v, _ in codec.QTY_BINS if kind == "n")
    for q in discrete:
        if q <= 100:
            labels = codec.encode_step({"market": [["SELL", "MILK", q]]}, obs)
            assert labels["exact"], q


def test_argument_variants_normalize(mid_env):
    obs = view(mid_env)
    n = codec.hand_count(obs)
    variants = [
        ({"farmer": ["WATER", 3, "x"]}, ["WATER"]),
        ({"farmer": ["PLANT", "MELON", 2]}, ["PLANT", "MELON"]),
        ({"farmer": ["PICKUP", "WHEAT"]}, ["PICKUP", "WHEAT", 1]),
        ({"farmer": ["PICKUP", "WHEAT", 250]}, ["PICKUP", "WHEAT", 100]),
        ({"farmer": ["PLACE", "MILK", 2.9]}, ["PLACE", "MILK", 2]),
        ({"farmer": ["PLACE", "MILK", "3"]}, ["PLACE", "MILK", 3]),
        ({"farmer": ["PLACE"]}, ["PASS"]),
        ({"farmer": ["PICKUP", "SEEDS", 3]}, ["PASS"]),
        ({"farmer": ["PICKUP", "WHEAT", 0]}, ["PASS"]),
        ({"farmer": ["water"]}, ["PASS"]),
        ({"farmer": []}, ["PASS"]),
        ({"farmer": "PASS"}, ["PASS"]),
        ({}, ["PASS"]),
    ]
    for action, farmer in variants:
        assert codec.canonical(action, obs)["farmer"] == farmer, action
        assert len(codec.canonical(action, obs)["hands"]) == n
    market = [["SELL", "MILK", 999], [], ["PASS"], ["HIRE", None, 3], ["SELL", "COW", 1],
              ["BUY_PRODUCT", "MILK", 1], ["BUY_SEED", "WHEAT", "abc"], ["BUY_SEED", "WHEAT", 0],
              ["BUY_LAND", "x"], ["BUY_SEED", "WHEAT", 12.7], ["SELL", "EGG"], ["PASS"]]
    assert codec.canonical({"market": market}, obs)["market"] == [
        ["SELL", "MILK", 100], ["NOOP"], ["NOOP"], ["HIRE"], ["NOOP"], ["NOOP"], ["NOOP"],
        ["NOOP"], ["BUY_LAND"], ["BUY_SEED", "WHEAT", 12]]


# ---------------------------------------------------------------- backward
def test_backward_every_token_and_bin_is_engine_accepted(mid_env):
    before = state(mid_env)
    report = codec.backward_check(mid_env)
    assert report["accepted"] == report["unit_combos"] + report["order_combos"]
    assert report["unit_combos"] == sum(codec.QTY_BIN_COUNT if q else 1 for q in codec.UNIT_HAS_QTY)
    assert state(mid_env) == before


def test_random_model_outputs_decode_to_accepted_actions(engine, mid_env):
    rng = random.Random(0)
    obs = view(mid_env)
    grammar = codec.engine_grammar()
    for _ in range(200):
        k = rng.randrange(0, codec.MAX_ORDERS + 1)
        labels = {"unit_tok": [rng.randrange(len(codec.UNIT_VOCAB)) for _ in range(codec.MAX_UNITS)],
                  "unit_qty": [rng.randrange(codec.QTY_BIN_COUNT) for _ in range(codec.MAX_UNITS)],
                  "order_tok": [rng.randrange(1, len(codec.ORDER_VOCAB)) for _ in range(k)] + [codec.STOP],
                  "order_qty": [rng.randrange(codec.QTY_BIN_COUNT) for _ in range(k + 1)]}
        action = codec.decode_step(labels, obs)
        codec.engine_accepts(engine, grammar, obs, action)
        step_copy(mid_env, [action, action])


# ---------------------------------------------------------------- bounds
def test_max_units_covers_measured_and_engine_bound(mid_env):
    vocab = json.loads((HERE / "VOCAB.json").read_text())
    assert codec.MAX_UNITS >= 1 + vocab["all_scanned"]["stats"]["max_obs_hands"]
    assert codec.MAX_UNITS >= 1 + vocab["max_units"]["engine_practical_max_hands"]
    obs = view(mid_env)
    base = obs["farms"][0]["hands"][0]
    for hands, exact in ((codec.MAX_UNITS - 1, True), (codec.MAX_UNITS + 2, False)):
        o = copy.deepcopy(obs)
        o["farms"][0]["hands"] = [list(base)] * hands
        o["private"]["inventories"] = [{}] * (hands + 1)
        action = {"farmer": ["WATER"], "hands": [["DIG"]] * hands, "market": []}
        reasons = []
        labels = codec.encode_step(action, o, reasons)
        assert labels["exact"] is exact
        out = codec.decode_step(labels, o)
        assert len(out["hands"]) == hands
        if exact:
            assert out == codec.canonical(action, o)
        else:
            assert reasons == ["HANDS_OVER_MAX_UNITS"]
            assert out["hands"][codec.MAX_UNITS - 1:] == [["PASS"]] * (hands - codec.MAX_UNITS + 1)


def test_market_empty_and_full_queue(mid_env):
    obs = view(mid_env)
    labels = codec.encode_step({"farmer": ["PASS"], "hands": [], "market": []}, obs)
    assert labels["order_tok"] == [codec.STOP] and labels["order_qty"] == [-1]
    assert codec.decode_step(labels, obs)["market"] == []
    full = [["HIRE"], ["SELL", "MILK", 3], ["BUY_SEED", "WHEAT", 5], ["PASS"], ["BUY_LAND"],
            ["SELL", "WHEAT", 1], ["BUY_PRODUCT", "FERTILIZER", 2], ["BUY_ANIMAL", "COW", 1],
            ["HIRE"], ["SELL", "EGG", 2], ["SELL", "WOOL", 9], ["HIRE"]]
    labels = codec.encode_step({"market": full}, obs)
    assert len(labels["order_tok"]) == codec.MAX_ORDERS + 1 and labels["order_tok"][-1] == codec.STOP
    out = codec.decode_step(labels, obs)
    assert len(out["market"]) == codec.MAX_ORDERS
    assert out == codec.canonical({"market": full}, obs) and labels["exact"]


# ---------------------------------------------------------------- engine equivalence
def _random_unit(rng):
    kind = rng.random()
    if kind < 0.1:
        return rng.choice([[], None, "WATER", ["water"], [7], ["PLACE"], ["PICKUP"], ["PLANT"],
                           ["PLANT", "ROSE"], ["FLY"], ["PICKUP", "SEEDS", 2]])
    token = rng.choice(codec.UNIT_VOCAB).split(":")
    if token[0] in codec.ITEM_UNIT_OPS:
        count = rng.choice([None, 1, 2, 5, 17, 99, 100, 101, 400, 2.5, "3", 0, -2])
        return token if count is None else token + [count]
    if rng.random() < 0.1:
        return token + [rng.choice([1, "x", None])]
    return token


def _random_order(rng):
    kind = rng.random()
    if kind < 0.1:
        return rng.choice([[], ["PASS"], None, "HIRE", ["HIRE", None, 2], ["BUY_LAND", "x"],
                           ["SELL", "COW", 3], ["BUY_PRODUCT", "MILK", 2], ["BUY_SEED", "ROSE", 1],
                           ["SELL", "MILK"], ["SELL", "MILK", "abc"], ["SELL", "MILK", 0]])
    token = rng.choice(codec.ORDER_VOCAB[2:]).split(":")
    if token[0] in codec.MARKET_QTY_OPS:
        return token + [rng.choice([1, 2, 3, 10, 50, 100, 101, 1000, 7.9, "4"])]
    return token


def test_canonical_and_decoded_actions_are_engine_equivalent(mid_env):
    """Random (incl. malformed) wire actions: raw, canonical and decode(encode) give identical states."""
    rng = random.Random(1)
    obs = [view(mid_env, s) for s in (0, 1)]
    for _ in range(150):
        raw = []
        for s in (0, 1):
            n = codec.hand_count(obs[s]) + rng.choice([-1, 0, 0, 1, 2])
            raw.append({"farmer": _random_unit(rng), "hands": [_random_unit(rng) for _ in range(max(0, n))],
                        "market": [_random_order(rng) for _ in range(rng.randrange(0, 13))]})
        try:
            canon = [codec.canonical(raw[s], obs[s]) for s in (0, 1)]
        except codec.Unrepresentable:
            continue
        expected = state(step_copy(mid_env, raw))
        assert state(step_copy(mid_env, canon)) == expected
        labels = [codec.encode_step(raw[s], obs[s]) for s in (0, 1)]
        if all(l["exact"] for l in labels):
            decoded = [codec.decode_step(labels[s], obs[s]) for s in (0, 1)]
            assert decoded == canon
            assert state(step_copy(mid_env, decoded)) == expected


def test_extra_hand_plant_demand_is_applied(mid_env):
    obs = view(mid_env)
    n = codec.hand_count(obs)
    o = copy.deepcopy(obs)
    o["private"]["seeds"]["MELON"] = 1
    action = {"farmer": ["PLANT", "MELON"], "hands": [["PASS"]] * n + [["PLANT", "MELON"]]}
    assert codec.canonical(action, o)["farmer"] == ["PASS"]
    o["private"]["seeds"]["MELON"] = 2
    assert codec.canonical(action, o)["farmer"] == ["PLANT", "MELON"]


# ---------------------------------------------------------------- replays
def test_one_replay_round_trip_and_engine_semantics(replay_task):
    res, rows = codec.measure_episode(replay_task, roundtrip=True, semantic=True)
    rt, sem = res["roundtrip"], res["semantic"]
    assert rt["steps"] == 1438 and rt["exact_but_mismatch"] == 0
    assert sem["steps"] == 719 and sem["exact_equal"] == sem["exact_steps"]
    assert rows.shape[1] == 6
