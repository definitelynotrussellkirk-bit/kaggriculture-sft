"""features.py: exact composition of theorem_features + history_features, causality, actor view.

Run (pinned interpreter, CPU only):
  cd /home/russ/Desktop/farmNEAT && CUDA_VISIBLE_DEVICES= \
  PYTHONPATH=src:work/runs/maintainer/primitive-bc:work/cache/pytest-path \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  work/venv/python -B -m pytest -q -p no:cacheprovider work/runs/maintainer/primitive-bc/test_features.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import features as FT  # noqa: E402
import history_features as HF  # noqa: E402
import theorem_features as TF  # noqa: E402
from farm_neat.training import bc  # noqa: E402

DAILY = ROOT / "work/datasets/public-episodes/daily"
EPISODES = ("111424871", "111419545")
LAST = 718


@lru_cache(maxsize=None)
def replay(eid):
    return json.loads((DAILY / f"{eid}.json").read_text())


@lru_cache(maxsize=None)
def stream(eid, seat, steps=LAST + 1):
    data = replay(eid)
    obs = tuple(bc.public_observation(data, t, seat) for t in range(steps))
    acts = tuple(data["steps"][t + 1][seat]["action"] for t in range(steps))
    return obs, acts, data["configuration"]


def _same(a, b):
    assert set(a) == set(b)
    for name in FT.BLOCKS:
        assert a[name].dtype == np.float32 and a[name].shape == FT.SHAPES[name], name
        assert np.array_equal(a[name], b[name]), name
        assert np.array_equal(a["known"][name], b["known"][name]), name
    assert a["step"] == b["step"]


def test_schema_is_versioned_and_complete():
    digest = hashlib.sha256(json.dumps(FT.SCHEMA_PAYLOAD, sort_keys=True).encode()).hexdigest()
    assert FT.FEATURE_SCHEMA == "primitive-bc-features-v1" and FT.FEATURE_SCHEMA_SHA256 == digest
    assert len(FT.flat_names()) == len(set(FT.flat_names())) == FT.N_VALUES
    assert FT.SHAPES["theorem.tile"] == (200, len(TF.TILE_NAMES))
    assert FT.SHAPES["history.item_lag"] == (len(HF.ITEMS), len(HF.ITEM_HIST))
    obs, _, cfg = stream(EPISODES[0], 0, 3)
    f = FT.FeatureState(cfg).step(obs[0])
    values, known = FT.flat(f)
    assert values.shape == known.shape == (FT.N_VALUES,)
    assert np.all(values[~known] == 0)


@pytest.mark.parametrize("eid", EPISODES)
@pytest.mark.parametrize("seat", (0, 1))
def test_every_block_equals_its_source_module(eid, seat):
    """features = TF.compute/SCALES (tile own then rival) and the history blocks, bit for bit,
    on every step of a full recorded episode."""
    obs, acts, cfg = stream(eid, seat)
    state = FT.FeatureState(cfg)
    ref = HF.HistoryFeatures()
    for t, o in enumerate(obs):
        f = state.step(o)
        raw = TF.compute(o, cfg)
        assert np.array_equal(f["theorem.global"], raw["global"] / TF.SCALES["global"])
        assert np.array_equal(f["theorem.product"], raw["product"] / TF.SCALES["product"])
        assert np.array_equal(f["theorem.structure"], raw["structure"] / TF.SCALES["structure"])
        assert np.array_equal(f["theorem.tile"][:100], raw["tile_own"] / TF.SCALES["tile_own"])
        assert np.array_equal(f["theorem.tile"][100:], raw["tile_rival"] / TF.SCALES["tile_rival"])
        h = ref.update(o, cfg)
        for name, src in FT._HIST.items():
            assert np.array_equal(f[name], h.blocks[src][2]), (t, name)
            assert np.array_equal(f["known"][name], h.blocks[src][3]), (t, name)
        for name in FT.THEOREM_BLOCKS:
            assert f["known"][name].all() and np.isfinite(f[name]).all()
        assert f["step"] == t
        state.record_action(acts[t])
        ref.record_action(acts[t])


def test_tile_rows_follow_observation_grid():
    """row role*100 + y*10 + x: own weeds/plants land on the tile the observation shows."""
    obs, _, cfg = stream(EPISODES[0], 1, 400)
    o = obs[399]
    f = FT.FeatureState(cfg).step(o)
    col = FT.BLOCKS["theorem.tile"][1]
    owned, plant = col.index("owned"), col.index("age_days")
    for role, farm in enumerate((o["farms"][1], o["farms"][0])):
        for y, line in enumerate(farm["tiles"]):
            for x, tile in enumerate(line):
                row = f["theorem.tile"][role * 100 + y * 10 + x]
                assert row[owned] == (tile != "LOCKED")
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    assert np.isclose(row[plant] * TF.SCALES["tile_own"][plant], 399 // 24 - tile["planted_day"])


def test_actor_view_only():
    """Hidden fields (seed, future shops, rival private stock) change nothing, whether they are
    stripped by bc.actor_observation or passed straight into the actor view."""
    data = replay(EPISODES[0])
    cfg = data["configuration"]
    base, stripped, direct = FT.FeatureState(cfg), FT.FeatureState(cfg), FT.FeatureState(cfg)
    for t in range(60):
        raw = copy.deepcopy(data["steps"][t][0]["observation"])
        raw["step"] = t
        noisy = copy.deepcopy(raw)
        noisy.update(seed=987, remainingOverageTime=3.0, rival_private={"shed": {"MILK": 50}})
        noisy["town"]["future_shops"] = ["DAIRY"] * 5
        noisy["farms"][1]["shed"] = {"WHEAT": 99}
        dirty = bc.actor_observation(raw)
        dirty.update(seed=987, rival_private={"shed": {"MILK": 50}})
        dirty["town"]["future_shops"] = ["DAIRY"] * 5
        dirty["farms"][1]["shed"] = {"WHEAT": 99}
        a = base.step(bc.actor_observation(raw))
        _same(a, stripped.step(bc.actor_observation(noisy)))
        _same(a, direct.step(dirty))
        action = data["steps"][t + 1][0]["action"]
        for s in (base, stripped, direct):
            s.record_action(action)


def test_episode_helper_seat_dispatch_and_prefix_causality():
    obs0, acts0, cfg = stream(EPISODES[0], 0, 200)
    obs1, acts1, _ = stream(EPISODES[0], 1, 200)
    full = FT.episode_features(obs0, acts0, cfg)
    short = FT.episode_features(obs0[:120], acts0[:120], cfg)
    for a, b in zip(full[:120], short):
        _same(a, b)
    seats = FT.SeatFeatures(cfg)
    ref1 = FT.episode_features(obs1, acts1, cfg)
    for t in range(200):   # interleaved like one callable playing both seats
        _same(seats.step(obs0[t]), full[t])
        seats.record_action(0, acts0[t])
        _same(seats.step(obs1[t]), ref1[t])
        seats.record_action(1, acts1[t])


def test_new_episode_resets_and_compute_is_cached():
    obs, acts, cfg = stream(EPISODES[0], 0, 100)
    fresh = FT.episode_features(obs[:10], acts[:10], cfg)
    state = FT.FeatureState(cfg)
    for t in range(100):
        state.step(obs[t])
        state.record_action(acts[t])
    for t in range(10):   # a second episode on the same object starts clean at step 0
        f = state.step(obs[t])
        assert state.compute() is f
        _same(f, fresh[t])
        state.record_action(acts[t])
    with pytest.raises(ValueError):
        FT.FeatureState(cfg).compute()
