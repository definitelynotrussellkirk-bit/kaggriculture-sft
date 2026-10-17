"""CRN league games (games.crn_rng -> league_game.py --crn-rng) without playing a full game: the worker passes the
flag only when configured, build_replay is unchanged without it and CRN-exact with it, and the export marks CRN
episodes.   python3 -m pytest -q tests/test_crn.py     (needs the farmNEAT src with farm_neat/rollout/crn_rng.py:
LEAGUE_ENGINE_SRC, default /home/russ/Desktop/farmNEAT/src)"""
import copy
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import league_core as core  # noqa: E402
import league_export  # noqa: E402
import league_game  # noqa: E402
import league_worker  # noqa: E402

SRC = os.environ.get("LEAGUE_ENGINE_SRC", "/home/russ/Desktop/farmNEAT/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
ee = pytest.importorskip("farm_neat.game.exact_engine")
crn_rng = pytest.importorskip("farm_neat.rollout.crn_rng")
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
TEMPLATE = json.loads((HERE / "replay-template.json").read_text())


def _worker(tmp_path, crn):
    cfg = core.load_config("/nonexistent")
    feed = tmp_path / "thermal.json"
    feed.write_text(json.dumps({"epoch": 1e12, "inlet_c": 30.0}))
    cfg["service"].update({"db": str(tmp_path / "db.sqlite3"), "data_dir": str(tmp_path / "data")})
    cfg["games"].update({"tmpdir": str(tmp_path / "tmp"), "steps": 719, "seed_base": 5000})
    if crn is not None:
        cfg["games"]["crn_rng"] = crn
    cfg["thermal"].update({"file": str(feed), "check_every_s": 0})
    w = league_worker.Worker(cfg, log=lambda m: None)
    for sid, name in ((1, "fwd1-46236"), (2, "b")):
        w.con.execute("INSERT INTO submissions(id, name, status, source, archive_path, archive_sha256, archive_bytes, "
                      "created_utc, updated_utc) VALUES (?,?, 'active', 'test', '/x', 'h', 1, 'u', 'u')", (sid, name))
    w.con.commit()
    return w


@pytest.mark.parametrize("crn,flag", [(None, False), (False, False), (True, True)])
def test_worker_passes_the_flag_only_when_configured(tmp_path, monkeypatch, crn, flag):
    w = _worker(tmp_path, crn)
    w.startup()
    assert w.create_match()
    seen = {}

    class FakeProc:
        pid = 1

        def poll(self):
            return None

    def fake_popen(cmd, **kw):
        seen["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(league_worker.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(w, "archive_ok", lambda sub: (True, None))
    ep = w.con.execute("SELECT * FROM episodes WHERE state='queued' ORDER BY id LIMIT 1").fetchone()
    assert w.launch(ep)
    assert ("--crn-rng" in seen["cmd"]) is flag
    runtime = json.loads(w.con.execute("SELECT runtime FROM episodes WHERE id=?", (ep["id"],)).fetchone()[0])
    assert runtime["crn_rng"] is flag


def _actions(n=150):
    """Joint actions that change the farm (seeds bought and planted) so a dusk sees different empty tiles."""
    acts = []
    for t in range(n):
        seat0 = {"farmer": ["PLANT", "WHEAT"] if t % 3 else ["EAST"], "hands": [],
                 "market": [["BUY_SEED", "WHEAT", 2]] if t < 40 else []}
        acts.append([seat0, copy.deepcopy(PASS)])
    return acts


def _base_game_module():
    import importlib.util
    base = HERE.parent / "base" / "league_game.py"
    if not base.is_file():
        pytest.skip("unpatched league_game.py not beside this copy")
    spec = importlib.util.spec_from_file_location("league_game_base", base)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_replay_without_the_flag_is_unchanged():
    base = _base_game_module()
    acts = _actions()
    kw = dict(world_seed=32000123, actions=acts, names=["SELF:a", "SELF:b"], episode_id=7100000000000123)
    old = base.build_replay(ee, TEMPLATE, **kw)
    new = league_game.build_replay(ee, TEMPLATE, **kw)
    assert json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True)
    assert "crnRng" not in new[0]["configuration"]


def test_build_replay_with_the_flag_is_a_crn_replay():
    acts = _actions()
    seed = 32000123
    replay, cash, _ = league_game.build_replay(ee, TEMPLATE, world_seed=seed, actions=acts, names=["SELF:a", "SELF:b"],
                                               episode_id=7100000000000123, crn=True)
    assert replay["configuration"]["crnRng"] is True
    cfg = {k: v for k, v in replay["configuration"].items() if k not in ("seed", "actTimeout", "runTimeout")}
    env = crn_rng.environment(seed=seed, configuration=cfg)
    plain = ee.ExactKaggricultureEnvironment(seed=seed)
    first_plain_diff = None
    for t, joint in enumerate(acts):
        env.step(copy.deepcopy(joint))
        plain.step(copy.deepcopy(joint))
        for p in (0, 1):
            assert ee._copy_wire(dict(env.state[p].observation)) == replay["steps"][t + 1][p]["observation"]
        if first_plain_diff is None and plain.state[0].observation.town != env.state[0].observation.town:
            first_plain_diff = t
    assert cash == [float(f["money"]) for f in env.state[0].observation.farms]
    shops = replay["steps"][-1][0]["observation"]["town"]["unlocked_shops"]
    keyed = [s for d, s in crn_rng.shop_sequence(seed, ee.load_engine().SHOPS) if d * 24 + 23 < len(acts)]
    assert shops == keyed
    assert first_plain_diff is not None                     # the flag changed the game


def test_export_marks_crn_episodes(tmp_path):
    cfg = core.load_config("/nonexistent")
    cfg["service"]["db"] = str(tmp_path / "db.sqlite3")
    cfg["export"] = {"enabled": True, "dir": str(tmp_path / "extra" / "league")}
    con = core.connect(cfg["service"]["db"])
    for i, name in enumerate(("fwd1-46236", "b"), 1):
        con.execute("INSERT INTO submissions(id, name, status, source, archive_path, archive_sha256, archive_bytes, "
                    "created_utc, updated_utc) VALUES (?,?, 'active', 't', '/x', 'h', 1, 'u', 'u')", (i, name))
    for eid, crn in ((1, True), (2, False)):
        rp = tmp_path / f"r{eid}.json.gz"
        with gzip.open(rp, "wt") as f:
            json.dump({"info": {"EpisodeId": core.LEAGUE_EPISODE_BASE + eid}, "steps": []}, f)
        con.execute("INSERT INTO episodes(id, world_seed, seat0, seat1, state, created_utc, finished_utc, cash0, cash1, "
                    "score0, rated, replay_path, source, runtime) VALUES (?, 7, 1, 2, 'done', 'u', "
                    "'2026-09-27T10:05:00Z', 5, 3, 1.0, 1, ?, 'league', ?)", (eid, str(rp), json.dumps({"crn_rng": crn})))
    con.commit()
    assert league_export.export(cfg)["packed"] == 2
    rows = {r["league_episode"]: r for r in map(json.loads, open(tmp_path / "extra" / "league" / "MANIFEST.jsonl"))}
    assert rows[1].get("crn_rng") is True and "crn_rng" not in rows[2]


def test_runner_refuses_a_snapshot_without_the_module(tmp_path):
    """--crn-rng on an old snapshot fails loudly (runner-error.txt, exit 3), it never plays the original rules."""
    snap = tmp_path / "snap"
    (snap / "work/runs/maintainer/pbc-ladder").mkdir(parents=True)
    (snap / "src/farm_neat/game").mkdir(parents=True)
    (snap / "src/farm_neat/__init__.py").write_text("")
    (snap / "src/farm_neat/game/__init__.py").write_text("")
    game_dir = Path(ee.__file__).parent
    for name in ("exact_engine.py", "kaggriculture_engine_reference.py", "kaggriculture.json"):
        (snap / "src/farm_neat/game" / name).write_bytes((game_dir / name).read_bytes())
    (snap / "work/runs/maintainer/pbc-ladder/ladder.py").write_text(
        "import sys\nfrom pathlib import Path\nROOT = Path(__file__).resolve().parents[4]\nHOSTED_DEFAULTS = {}\n"
        "LADDER_SHA256 = 'x'\n\ndef _setup(code):\n    sys.path.insert(0, str(ROOT / 'src'))\n\n"
        "def play(spec):\n    raise AssertionError('must not play')\n")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-B", str(HERE / "league_game.py"), "--engine", str(snap), "--code", str(snap),
                        "--seat0", "a", "--seat1", "b", "--labels", "a", "b", "--world-seed", "1", "--episode-id", "1",
                        "--template", str(HERE / "replay-template.json"), "--out", str(out), "--crn-rng"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ""})
    assert r.returncode == 3 and "crn_rng" in (out / "runner-error.txt").read_text()
