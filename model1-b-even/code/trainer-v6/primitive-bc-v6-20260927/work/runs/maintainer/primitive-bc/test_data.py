"""data.py contract tests: sealed exclusion, split, header read, example
layout/labels, worker parity, windows and collate."""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

import codec
import data
from farm_neat.policy import observation
from farm_neat.training import bc


@pytest.fixture(scope="module")
def archive():
    return data.default_archives()[12]


@pytest.fixture(scope="module")
def members(archive):
    return data.list_members([archive], split="all")


@pytest.fixture(scope="module")
def leaderboard():
    return data.load_leaderboard()


def test_twenty_daily_archives_and_leaderboard(leaderboard):
    archives = data.default_archives()
    assert len(archives) == 20
    assert [data.archive_day(a) for a in archives] == [f"2026-09-{d:02d}" for d in range(3, 23)]
    assert len(leaderboard) > 1000 and min(leaderboard.values()) == 1


def test_sealed_ids_are_excluded_by_name_without_opening(monkeypatch):
    sealed = data.sealed_ids()
    assert len(sealed) == 16
    fake = next(iter(sealed))
    opened = []
    real_open = zipfile.ZipFile.open

    def guard(self, name, *a, **k):
        opened.append(name)
        return real_open(self, name, *a, **k)

    monkeypatch.setattr(zipfile.ZipFile, "open", guard)
    monkeypatch.setattr(data, "sealed_ids", lambda path=None: frozenset({fake}))
    tasks = data.list_members(data.default_archives()[:1], split="all")
    assert fake not in {t[3] for t in tasks}
    assert opened == []
    real = data.list_members(data.default_archives()[-3:], split="all", sealed=sealed)
    assert not ({t[3] for t in real} & sealed)


def test_split_is_deterministic_disjoint_and_covering(members):
    ids = [t[3] for t in members]
    parts = {s: {i for i in ids if data.in_split(i, s)} for s in ("train", "val", "test")}
    assert parts["train"] | parts["val"] | parts["test"] == set(ids)
    assert not (parts["train"] & parts["val"]) and not (parts["val"] & parts["test"])
    assert parts["val"] | parts["test"] == {i for i in ids if data.in_split(i, "heldout")}
    assert 0.8 < len(parts["train"]) / len(ids) < 0.97
    assert data.split_bucket(ids[0]) == data.split_bucket(ids[0])


def test_header_read_matches_full_parse(members):
    archive, member, day, eid = members[0]
    header = data.read_header(archive, member)
    with zipfile.ZipFile(archive) as z:
        full = json.loads(z.read(member))
    for key in ("info", "rewards", "statuses", "configuration"):
        assert header[key] == full[key]
    assert header["info"]["EpisodeId"] == eid


def test_examples_layout_and_labels(members, leaderboard):
    task = members[0]
    ex = list(data._episode_task(task, rank_max=None, seats=1, leaderboard=leaderboard))
    flat = [e for chunk in ex for e in chunk]
    assert len(flat) == data.STEPS and [e["meta"]["step"] for e in flat] == list(range(719))
    with zipfile.ZipFile(task[0]) as z:
        replay = json.loads(z.read(task[1]))
    for t in (0, 1, 250, 718):
        e = flat[t]
        obs = bc.public_observation(replay, t, 1)
        packed = observation.pack(obs, None, data.pack_config(replay["configuration"]))
        assert e["rows"].dtype == np.float16 and e["rows"].shape == packed["rows"].shape
        assert np.array_equal(e["rows"], packed["rows"].astype(np.float16))
        assert np.array_equal(e["knownness"], packed["knownness"]) and e["token_mask"].all()
        assert e["unit_pos"].shape == (codec.MAX_UNITS, 2)
        units = codec.unit_positions(obs)
        assert e["unit_pos"][:len(units)].tolist() == units and (e["unit_pos"][len(units):] == -1).all()
        labels = codec.encode_step(replay["steps"][t + 1][1]["action"], obs)
        assert {k: e[k] for k in labels} == labels
        assert e["meta"]["seat"] == 1 and e["meta"]["episode_id"] == task[3]
        assert e["meta"]["team"] == replay["info"]["TeamNames"][1]
        assert e["meta"]["rank"] == leaderboard.get(replay["info"]["TeamNames"][1])


def test_rank_filter_selects_seats(members):
    task = members[1]
    teams = data.read_header(task[0], task[1])["info"]["TeamNames"]
    lb = {teams[0]: 3, teams[1]: 500}
    got = [e["meta"]["seat"] for chunk in data._episode_task(task, rank_max=10, seats="both", leaderboard=lb)
           for e in chunk]
    assert set(got) == {0} and len(got) == data.STEPS
    assert list(data._episode_task(task, rank_max=2, seats="both", leaderboard=lb)) == []


def test_workers_match_in_process(archive, leaderboard):
    kw = dict(split="all", rank_max=None, seats=0, max_episodes=2, leaderboard=leaderboard)
    serial = list(data.iter_examples([archive], workers=0, **kw))
    parallel = list(data.iter_examples([archive], workers=2, **kw))
    key = lambda e: (e["meta"]["episode_id"], e["meta"]["step"])
    assert sorted(map(key, serial)) == sorted(map(key, parallel)) and len(serial) == 2 * data.STEPS
    by = {key(e): e for e in parallel}
    for e in serial[::97]:
        other = by[key(e)]
        assert np.array_equal(e["rows"], other["rows"]) and e["unit_tok"] == other["unit_tok"]


def test_windows_and_collate(members, leaderboard):
    import torch
    examples = [e for chunk in data._episode_task(members[0], rank_max=None, seats=0, leaderboard=leaderboard)
                for e in chunk]
    windows = list(data.make_windows(examples, window=24, burn_in=24))
    assert len(windows) == 30
    assert windows[0]["n_burn"] == 0 and windows[0]["from_episode_start"]
    assert windows[1]["n_burn"] == 24 and windows[1]["from_episode_start"]
    assert windows[2]["n_burn"] == 24 and not windows[2]["from_episode_start"]
    assert windows[-1]["n_scored"] == 719 - 29 * 24
    for w in windows:
        steps = [e["meta"]["step"] for e in w["steps"]]
        assert steps == list(range(w["start_step"] - w["n_burn"], w["start_step"] + w["n_scored"]))
    short = list(data.make_windows(examples, window=24, burn_in=10))
    assert short[1]["n_burn"] == 10 and not short[1]["from_episode_start"]

    batch = data.collate(examples[:5])
    assert batch["rows"].dtype == torch.float32 and batch["rows"].shape[0] == 5
    assert batch["order_tok"].shape == (5, codec.MAX_ORDERS + 1)
    assert batch["unit_tok"].shape == (5, codec.MAX_UNITS)
    n = batch["token_mask"].sum(1)
    assert (n == torch.tensor([len(e["rows"]) for e in examples[:5]])).all()

    wb = data.collate_windows([windows[0], windows[5], windows[-1]])
    t_len = 24 + 24
    assert wb["rows"].shape[:2] == (3, t_len)
    assert wb["first_valid"].tolist() == [24, 0, 0]
    assert wb["loss_mask"][:, :24].sum() == 0
    assert wb["loss_mask"].sum(1).tolist() == [24, 24, 719 - 29 * 24]
    assert (wb["step_valid"] | ~wb["loss_mask"]).all()
    assert wb["step"][1, 0].item() == windows[5]["start_step"] - 24
    assert wb["step"][0, 24].item() == 0
    col = 24 + 3
    ex = windows[5]["steps"][24 + 3]
    assert wb["unit_tok"][1, col].tolist() == ex["unit_tok"]
    k = len(ex["rows"])
    assert torch.equal(wb["rows"][1, col, :k], torch.from_numpy(ex["rows"].astype(np.float32)))


def test_iter_windows_streams_whole_perspectives(archive, leaderboard):
    windows = list(data.iter_windows([archive], split="all", seats=1, max_episodes=1, workers=0,
                                     leaderboard=leaderboard))
    assert len(windows) == 30 and {w["meta"]["seat"] for w in windows} == {1}
    assert sum(w["n_scored"] for w in windows) == data.STEPS


def test_abandoned_worker_stream_exits_promptly(archive):
    """Regression: undelivered tasks must not block interpreter exit."""
    import os
    import subprocess
    import sys
    code = ("import data; it = data.iter_examples([%r], split='all', workers=2, leaderboard={}); "
            "next(it); it.close()" % archive)
    r = subprocess.run([sys.executable, "-B", "-c", code], timeout=120, env=dict(os.environ),
                       cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True)
    assert r.returncode == 0, r.stderr.decode()[-2000:]
