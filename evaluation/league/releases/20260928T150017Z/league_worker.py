#!/usr/bin/env python3
"""League worker: validation queue, Kaggle-like matchmaking, match launching, result ingestion, rating job and the
thermal guard, in one loop (one process; each game is a child process group). Standard library only.

  LEAGUE_CONFIG=/etc/league/league.toml python3 league_worker.py            (systemd: league-worker.service)
  python3 league_worker.py --once                                           (one scheduling pass, for tests)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import league_core as core  # noqa: E402


def mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return 0.0


def tail(path, n=3000):
    try:
        return Path(path).read_text(errors="replace")[-n:]
    except OSError:
        return ""


def validate_game_record(rec, ep, submissions):
    """Validate the runner/worker boundary before any rating writes.

    This validates the emitted record, not the archive's policy strength. Agent
    fallbacks and replay mismatches still follow the existing unrated policy.
    """
    if not isinstance(rec, dict) or rec.get("schema") != "league-episode-v1":
        raise ValueError("missing or unsupported game record schema")
    expected = {"episode_id": core.LEAGUE_EPISODE_BASE + ep["id"], "world_seed": ep["world_seed"],
                "labels": [submissions[s]["name"] for s in (ep["seat0"], ep["seat1"])],
                "archives": [submissions[s]["archive_path"] for s in (ep["seat0"], ep["seat1"])]}
    for key, value in expected.items():
        if rec.get(key) != value:
            raise ValueError(f"{key} does not match the scheduled episode")
    game = rec.get("game")
    if not isinstance(game, dict):
        raise ValueError("game must be an object")

    def pair(value, name):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{name} must contain exactly two seats")
        return value

    def finite(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        return float(value)

    cash = [finite(x, "cash") for x in pair(game.get("cash"), "cash")]
    status = pair(game.get("status"), "status")
    if not all(isinstance(x, str) and x for x in status):
        raise ValueError("status must contain two nonempty strings")
    scores = [finite(x, "score") for x in pair(game.get("score"), "score")]
    score0 = core.episode_score0(*cash, *status)
    if scores != [score0, 1.0 - score0]:
        raise ValueError("score disagrees with cash/status")
    fb = pair(game.get("fallbacks"), "fallbacks")
    if any(type(x) is not int or x < 0 for x in fb):
        raise ValueError("fallbacks must be nonnegative integers")
    stats = pair(game.get("seat_stats", [{}, {}]), "seat_stats")
    for stat in stats:
        if not isinstance(stat, dict) or type(stat.get("hard_errors", 0)) is not int or stat.get("hard_errors", 0) < 0:
            raise ValueError("seat_stats must contain objects with nonnegative integer hard_errors")
    if type(game.get("steps")) is not int or game["steps"] < 0:
        raise ValueError("steps must be a nonnegative integer")
    if game.get("seconds") is not None:
        finite(game["seconds"], "seconds")
    for x in pair(game.get("ms_p50", [None, None]), "ms_p50"):
        if x is not None:
            finite(x, "ms_p50")
    if type(rec.get("replay_verified")) is not bool:
        raise ValueError("replay_verified must be a boolean")
    if rec["replay_verified"]:
        replay_cash = [finite(x, "replay_cash") for x in pair(rec.get("replay_cash"), "replay_cash")]
        if replay_cash != cash:
            raise ValueError("replay_verified contradicts replay_cash")
    if not isinstance(rec.get("runtime", {}), dict):
        raise ValueError("runtime must be an object")


# ----------------------------------------------------------------------------- desktop cold-verify receipts
RECEIPT_MAX_BYTES = 8 << 20
VERIFY_WORLD_SEED = 20260923      # build_archive.py verify defaults; the league's verify command passes no seeds
VERIFY_SAMPLER_SEED = 7100        # + our seat
TIME_BANK_MS = 60000.0
ACT_MS = 1000.0


def _is_int(value, expected):
    return type(value) is int and value == expected


def _obj(value):
    return value if isinstance(value, dict) else {}


def latency_rows(report):
    """Per-seat step latency (ms after the first request) of a build_archive.py verify report."""
    rows = []
    for game in report.get("games", []):
        cold = _obj(game.get("cold"))
        after = _obj(game.get("latency_ms_after_first") or cold.get("child_ms_after_first"))
        rows.append({"seat": game.get("our_seat", game.get("seat")), "p50": after.get("p50"), "p95": after.get("p95"),
                     "max": after.get("max"), "first_ms": _obj(cold.get("child_ms")).get("max")})
    return rows


def receipt_game_problems(game, seat, cfg):
    """Why one receipt game is not the league's cold game for ``seat`` ([] = it is)."""
    steps = int(cfg["games"]["steps"])
    out = []

    def need(ok, why):
        if not ok:
            out.append(why)

    need(game.get("done") is True and game.get("censored") is not True, "game not done or censored")
    need(_is_int(game.get("steps"), steps), f"steps {game.get('steps')!r} != {steps}")
    need(game.get("statuses") == ["DONE", "DONE"], f"statuses {game.get('statuses')!r}")
    need(_is_int(game.get("fallbacks"), 0), f"fallbacks {game.get('fallbacks')!r}")
    need(game.get("fallback_errors") in (None, []), "fallback errors recorded")
    need(_is_int(game.get("world_seed"), VERIFY_WORLD_SEED), f"world_seed {game.get('world_seed')!r}")
    need(_is_int(game.get("sampler_seed"), VERIFY_SAMPLER_SEED + seat), f"sampler_seed {game.get('sampler_seed')!r}")
    need(_is_int(_obj(game.get("latency_ms")).get("n"), steps),
         f"{_obj(game.get('latency_ms')).get('n')!r} timed decisions != {steps}")
    cold = game.get("cold")
    if not isinstance(cold, dict):
        return out + ["no cold-host record"]
    need(cold.get("complete") is True, "cold complete is not true")
    need(cold.get("engine_complete") is True, "engine_complete is not true")
    need(_is_int(cold.get("hard_errors"), 0), f"hard_errors {cold.get('hard_errors')!r}")
    need(_is_int(cold.get("log_lines"), steps) and cold.get("log_steps_contiguous") is True,
         f"step log {cold.get('log_lines')!r} lines, contiguous={cold.get('log_steps_contiguous')!r}")
    need(_is_int(cold.get("log_fallback_lines"), 0), f"log_fallback_lines {cold.get('log_fallback_lines')!r}")
    need(_is_int(_obj(cold.get("child_ms")).get("n"), steps), "cold host did not time every decision")
    need(_is_int(cold.get("threads"), 1), f"threads {cold.get('threads')!r}")
    need(_obj(cold.get("hello")).get("ready") is True, "cold host not ready")
    closed = _obj(cold.get("closed"))
    need(closed.get("closed") is True and closed.get("our_modules_outside_archive") == []
         and cold.get("close_error") is None, "closed-archive import check failed")
    tb = cold.get("timing_budget")
    if not isinstance(tb, dict):
        out.append("no timing_budget (verifier without the 60 s bank gate)")
    else:
        used = tb.get("used_bank_ms")
        need(tb.get("valid") is True and tb.get("within_bank") is True and _is_int(tb.get("calls"), steps)
             and tb.get("initial_bank_ms") == TIME_BANK_MS and tb.get("act_ms") == ACT_MS
             and not isinstance(used, bool) and isinstance(used, (int, float)) and math.isfinite(used)
             and 0.0 <= used <= TIME_BANK_MS and tb.get("first_exhausted_step") is None,
             f"not within the 60 s bank (within_bank={tb.get('within_bank')!r}, used_bank_ms={used!r})")
    # The torch/numpy sites the cold host imported must be the league's runtime (games.python's environment).
    runtime = str(Path(cfg["games"]["python"]).parents[1]).rstrip("/") + "/"
    sites = cold.get("pythonpath")
    sites = sites[1:] if isinstance(sites, list) else []
    need(bool(sites) and all(isinstance(s, str) and s.startswith(runtime) for s in sites),
         f"runtime sites {sites!r} are not under {runtime}")
    return out


def receipt_problems(rec, sub, manifest, cfg):
    """Every reason a desktop cold-verify receipt does not stand in for the league's cold verify of ``sub``.
    -> (problems, verifier_sha256, where the verifier identity came from); no problems = accept."""
    vc = cfg["validation"]
    if not isinstance(rec, dict):
        return ["receipt is not a JSON object"], None, None
    out = []
    if rec.get("schema") not in list(vc.get("receipt_schemas") or []):
        out.append(f"schema {rec.get('schema')!r} not in validation.receipt_schemas")
    # The v8 receipt does not name its verifier; deploy_checkpoint.py builds and verifies with the same
    # build_archive.py, whose sha256 the archive's MANIFEST records. A receipt that names its verifier wins.
    verifier, source = rec.get("verifier_sha256"), "receipt verifier_sha256"
    if verifier is None:
        verifier, source = _obj(manifest).get("builder_sha256"), "archive MANIFEST builder_sha256"
    if not isinstance(verifier, str) or verifier not in list(vc.get("receipt_verifiers") or []):
        out.append(f"verifier {verifier!r} ({source}) not in validation.receipt_verifiers")
    if rec.get("archive_sha256") != sub["archive_sha256"]:
        out.append(f"receipt archive_sha256 {str(rec.get('archive_sha256'))[:16]} != registered "
                   f"{sub['archive_sha256'][:16]}")
    if not _is_int(rec.get("archive_bytes"), sub["archive_bytes"]):
        out.append(f"receipt archive_bytes {rec.get('archive_bytes')!r} != registered {sub['archive_bytes']}")
    if rec.get("complete") is not True:
        out.append(f"complete is {rec.get('complete')!r}")
    if not _is_int(rec.get("fallbacks"), 0):
        out.append(f"fallbacks {rec.get('fallbacks')!r}")
    if not _is_int(rec.get("threads"), 1):
        out.append(f"threads {rec.get('threads')!r}")
    games = rec.get("games")
    if not isinstance(games, list):
        return out + ["no games list"], verifier, source
    want = {(op.strip(), int(s)) for op in str(vc["opponents"]).split(",") for s in str(vc["seats"]).split(",")}
    seen = set()
    for i, game in enumerate(games):
        if not isinstance(game, dict):
            out.append(f"games[{i}] is not an object")
            continue
        key = (game.get("opponent"), game.get("our_seat"))
        label = f"seat {key[1]!r} vs {key[0]!r}"
        if not isinstance(key[0], str) or type(key[1]) is not int or key not in want:
            out.append(f"{label}: not a league validation game")
        elif key in seen:
            out.append(f"{label}: duplicate game")
        else:
            seen.add(key)
            out += [f"{label}: {p}" for p in receipt_game_problems(game, key[1], cfg)]
    out += [f"seat {s} vs {op!r}: missing" for op, s in sorted(want - seen)]
    return out, verifier, source


class Worker:
    def __init__(self, cfg, *, log=print):
        self.cfg = cfg
        self.log = log
        self.con = core.connect(cfg["service"]["db"])
        self.guard = core.ThermalGuard(cfg)
        self.procs = {}              # episode id -> dict(proc, dir, started, log)
        self.validation = None       # dict(sub_id, kind, proc, dir, started, load1)
        self.sha_cache = {}
        self.dirty = True
        self.last_rating = 0.0
        self.last_thermal = 0.0
        self.thermal_paused = False
        self.thermal_pause_started = None
        self.rng = core.rng_for()
        self.stopping = False
        self.data = Path(cfg["service"]["data_dir"])

    # ------------------------------------------------------------------ lifecycle
    def startup(self):
        g = self.cfg["games"]
        # no game of this worker runs yet: extraction dirs left by killed cold hosts can go
        import shutil
        for d in Path(g["tmpdir"]).glob("pbc-cold-*"):
            shutil.rmtree(d, ignore_errors=True)
        for r in self.con.execute("SELECT id, attempt FROM episodes WHERE state='running'").fetchall():
            if r["attempt"] >= int(g["max_attempts"]):
                self.con.execute("UPDATE episodes SET state='error', error=?, finished_utc=? WHERE id=?",
                                 ("worker restarted during the game; attempts exhausted", core.now_utc(), r["id"]))
            else:
                self.con.execute("UPDATE episodes SET state='queued', attempt=attempt+1, started_utc=NULL WHERE id=?",
                                 (r["id"],))
        for r in self.con.execute("SELECT id FROM submissions WHERE status='validating'").fetchall():
            core.set_status(self.con, r["id"], "pending", "worker restarted during validation")
        if core.meta_get(self.con, "next_seed") is None:
            top = self.con.execute("SELECT MAX(world_seed) AS m FROM matches").fetchone()["m"]
            core.meta_set(self.con, "next_seed", max(int(g["seed_base"]), (top or 0) + 1))
        core.event(self.con, "worker_start", version=core.SERVICE_VERSION, config=self.cfg.get("_path"))
        self.con.commit()

    def stop(self, *_args):
        self.stopping = True

    def shutdown(self):
        for info in list(self.procs.values()) + ([self.validation] if self.validation else []):
            try:
                os.killpg(info["proc"].pid, signal.SIGCONT)
                os.killpg(info["proc"].pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.time() + 10
        for info in list(self.procs.values()) + ([self.validation] if self.validation else []):
            try:
                info["proc"].wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(info["proc"].pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for eid in list(self.procs):
            ep = self.con.execute("SELECT * FROM episodes WHERE id=?", (eid,)).fetchone()
            self.retry_episode(ep, "worker stopped during the game")
        if self.validation:
            core.set_status(self.con, self.validation["sub_id"], "pending", "worker stopped during validation") \
                if self.validation["kind"] == "submission" else None
        core.event(self.con, "worker_stop")
        self.con.commit()

    # ------------------------------------------------------------------ helpers
    def archive_ok(self, sub):
        path = sub["archive_path"]
        try:
            st = os.stat(path)
        except OSError as error:
            return False, f"archive unreadable: {error}"
        key = (path, st.st_size, st.st_mtime_ns)
        sha = self.sha_cache.get(key)
        if sha is None:
            sha = core.sha256_file(path)
            self.sha_cache[key] = sha
        if sha != sub["archive_sha256"]:
            return False, f"archive changed on disk (sha256 {sha[:12]} != registered {sub['archive_sha256'][:12]})"
        return True, ""

    def slots_free(self):
        return int(self.cfg["games"]["max_concurrent_games"]) - len(self.procs) - (1 if self.validation else 0)

    def env(self):
        g = self.cfg["games"]
        Path(g["tmpdir"]).mkdir(parents=True, exist_ok=True)
        return {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/var/lib/league", "LANG": "C.UTF-8",
                "TMPDIR": g["tmpdir"], "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1"}

    # ------------------------------------------------------------------ matchmaking
    def create_match(self):
        playable = [dict(r) for r in self.con.execute(
            "SELECT id, name, status FROM submissions WHERE status IN ('provisional','active') ORDER BY id")]
        if len(playable) < 2:
            return False
        board = {r["submission_id"]: dict(r) for r in self.con.execute("SELECT * FROM ratings")}
        pair_counts = {}
        for r in self.con.execute("SELECT seat0, seat1, COUNT(*) AS n FROM episodes WHERE state='done' AND rated=1 "
                                  "GROUP BY seat0, seat1"):
            key = (min(r["seat0"], r["seat1"]), max(r["seat0"], r["seat1"]))
            pair_counts[key] = pair_counts.get(key, 0) + r["n"]
        inflight = {}
        for r in self.con.execute("SELECT seat0, seat1 FROM episodes WHERE state IN ('queued','running')"):
            for s in (r["seat0"], r["seat1"]):
                inflight[s] = inflight.get(s, 0) + 1
        choice = core.choose_match(playable, board, pair_counts, inflight, self.rng, self.cfg)
        if choice is None:
            return False
        a, b, reason = choice
        seed = int(core.meta_get(self.con, "next_seed"))
        while self.con.execute("SELECT 1 FROM matches WHERE world_seed=?", (seed,)).fetchone():
            seed += 1
        core.meta_set(self.con, "next_seed", seed + 1)
        cur = self.con.execute("INSERT INTO matches(world_seed, a, b, created_utc, reason) VALUES (?,?,?,?,?)",
                               (seed, a, b, core.now_utc(), reason))
        mid = cur.lastrowid
        for s0, s1 in ((a, b), (b, a)):
            self.con.execute("INSERT INTO episodes(match_id, world_seed, seat0, seat1, state, created_utc) "
                             "VALUES (?,?,?,?, 'queued', ?)", (mid, seed, s0, s1, core.now_utc()))
        self.con.commit()
        return True

    # ------------------------------------------------------------------ games
    def launch(self, ep):
        g = self.cfg["games"]
        subs = {r["id"]: r for r in self.con.execute("SELECT * FROM submissions WHERE id IN (?, ?)",
                                                      (ep["seat0"], ep["seat1"]))}
        for sid in (ep["seat0"], ep["seat1"]):
            sub = subs.get(sid)
            if sub is None or sub["status"] not in core.PLAYABLE:
                self.con.execute("UPDATE episodes SET state='cancelled', finished_utc=?, error=? WHERE id=?",
                                 (core.now_utc(), f"entry {sid} is not playable", ep["id"]))
                self.con.commit()
                return False
            ok, why = self.archive_ok(sub)
            if not ok:
                core.set_status(self.con, sid, "failed", why)
                core.event(self.con, "archive_changed", submission=sid, detail=why)
                self.con.execute("UPDATE episodes SET state='cancelled', finished_utc=?, error=? WHERE id=?",
                                 (core.now_utc(), why, ep["id"]))
                self.dirty = True
                self.con.commit()
                return False
        day = time.strftime("%Y-%m-%d", time.gmtime())
        out = self.data / "episodes" / day / str(ep["id"])
        if out.exists():
            out = out.with_name(f"{ep['id']}-a{ep['attempt']}")
        out.mkdir(parents=True, exist_ok=True)
        s0, s1 = subs[ep["seat0"]], subs[ep["seat1"]]
        engine = g["engine"]
        cmd = ["nice", "-n", str(int(g["nice"])), g["python"], "-B", g["runner"], "--engine", engine,
               "--code", f"{engine}/code/primitive-bc", "--seat0", s0["archive_path"], "--seat1", s1["archive_path"],
               "--labels", s0["name"], s1["name"], "--world-seed", str(ep["world_seed"]),
               "--sampler-seed", str(int(g["sampler_seed"])), "--steps", str(int(g["steps"])),
               "--episode-id", str(core.LEAGUE_EPISODE_BASE + ep["id"]), "--template", g["template"], "--out", str(out)]
        if g.get("hosted_rules"):
            cmd.append("--hosted")
        if g.get("crn_rng"):
            cmd.append("--crn-rng")  # opt-in CRN dusk: shops/weeds keyed by seed, shared by both seat orders
        log = open(out / "game.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=g["tmpdir"], env=self.env(),
                                start_new_session=True)
        log.close()
        load1 = os.getloadavg()[0]
        runtime = {"python": g["python"], "engine": engine, "runner": g["runner"], "hosted_rules": bool(g.get("hosted_rules")),
                   "crn_rng": bool(g.get("crn_rng"))}
        self.con.execute("UPDATE episodes SET state='running', started_utc=?, episode_dir=?, load1=?, running_at_start=?,"
                         " runtime=?, thermal_paused=0 WHERE id=?",
                         (core.now_utc(), str(out), round(load1, 2), len(self.procs), json.dumps(runtime), ep["id"]))
        self.con.commit()
        self.procs[ep["id"]] = {"proc": proc, "dir": out, "started": time.time()}
        return True

    def fill_slots(self):
        g = self.cfg["games"]
        guard = 0
        while self.slots_free() > 0 and guard < 50:
            guard += 1
            if mem_available_gb() < float(g["min_free_mem_gb"]):
                return
            ep = self.con.execute("SELECT * FROM episodes WHERE state='queued' ORDER BY id LIMIT 1").fetchone()
            if ep is None:
                if not self.create_match():
                    return
                continue
            self.launch(ep)

    def retry_episode(self, ep, error):
        """One retry-budget rule for runner failures and graceful worker shutdowns."""
        if ep["attempt"] < int(self.cfg["games"]["max_attempts"]):
            self.con.execute("UPDATE episodes SET state='queued', attempt=attempt+1, started_utc=NULL, finished_utc=NULL,"
                             " rated=0, error=? WHERE id=?", (error[-2000:], ep["id"]))
            self.log(f"episode {ep['id']} runner failed (attempt {ep['attempt']}), requeued: {error[-300:]}")
        else:
            self.con.execute("UPDATE episodes SET state='error', finished_utc=?, rated=0, error=? WHERE id=?",
                             (core.now_utc(), error[-2000:], ep["id"]))
            self.log(f"episode {ep['id']} failed after {ep['attempt']} attempts")

    def ingest(self, eid, info, rc):
        out = info["dir"]
        rec_path = out / "game.json"
        ep = self.con.execute("SELECT * FROM episodes WHERE id=?", (eid,)).fetchone()
        g = self.cfg["games"]
        if rc == 0 and rec_path.is_file():
            try:
                rec = json.loads(rec_path.read_text())
                subs = {r["id"]: r for r in self.con.execute("SELECT * FROM submissions WHERE id IN (?, ?)",
                                                            (ep["seat0"], ep["seat1"]))}
                validate_game_record(rec, ep, subs)
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
                self.retry_episode(ep, f"invalid game.json: {type(error).__name__}: {error}")
                self.con.commit()
                self.dirty = True
                return
            game = rec["game"]
            cash = [float(x) for x in game["cash"]]
            status = list(game.get("status") or ["DONE", "DONE"])
            fb = [int(x or 0) for x in (game.get("fallbacks") or [0, 0])]
            stats = game.get("seat_stats") or [{}, {}]
            hard = [int((s or {}).get("hard_errors") or 0) for s in stats]
            score0 = core.episode_score0(cash[0], cash[1], *status)
            reasons = []
            if any(x != "DONE" for x in status):
                reasons.append(f"status {status}")
            if any(fb):
                reasons.append(f"fallbacks {fb}")
            if any(hard) and not g.get("hosted_rules"):
                reasons.append(f"hard errors {hard}")
            if game.get("steps") != int(g["steps"]) and all(x == "DONE" for x in status):
                reasons.append(f"steps {game.get('steps')}")
            if not rec.get("replay_verified"):
                reasons.append("replay rebuild cash differs")
            if g.get("hosted_rules") and ep["thermal_paused"]:
                reasons.append("thermal pause under hosted time rules")
            rated = 0 if reasons else 1
            ms = game.get("ms_p50") or [None, None]
            self.con.execute(
                "UPDATE episodes SET state='done', finished_utc=?, cash0=?, cash1=?, score0=?, rated=?, unrated_reason=?,"
                " fallbacks0=?, fallbacks1=?, status0=?, status1=?, steps=?, seconds=?, ms_p50_0=?, ms_p50_1=?,"
                " replay_path=?, replay_verified=?, runtime=?, error=NULL WHERE id=?",
                (core.now_utc(), cash[0], cash[1], score0, rated, "; ".join(reasons) or None, fb[0], fb[1], status[0],
                 status[1], game.get("steps"), game.get("seconds"), ms[0], ms[1], str(out / "replay.json.gz"),
                 1 if rec.get("replay_verified") else 0,
                 json.dumps({**json.loads(ep["runtime"] or "{}"), **(rec.get("runtime") or {})}), eid))
            self.log(f"episode {eid} seed {ep['world_seed']} {ep['seat0']}v{ep['seat1']}: cash {cash} "
                     f"rated={rated} {game.get('seconds')}s")
        else:
            err = tail(out / "runner-error.txt", 1500) or tail(out / "game.log", 1500) or f"runner exit {rc}"
            self.retry_episode(ep, err)
        self.con.commit()
        self.dirty = True

    def reap(self):
        g = self.cfg["games"]
        for eid, info in list(self.procs.items()):
            rc = info["proc"].poll()
            if rc is None:
                if not self.thermal_paused and time.time() - info["started"] > float(g["game_timeout_s"]):
                    try:
                        os.killpg(info["proc"].pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    info["proc"].wait()
                    (info["dir"] / "runner-error.txt").write_text(f"killed after {g['game_timeout_s']} s wall\n")
                    rc = -9
                else:
                    continue
            self.ingest(eid, info, rc)
            del self.procs[eid]  # retain the completed process if a transient DB write failed
        if self.validation:
            rc = self.validation["proc"].poll()
            if rc is not None:
                self.finish_validation(self.validation, rc)
                self.validation = None
            elif time.time() - self.validation["started"] > float(self.cfg["validation"]["timeout_s"]) \
                    and not self.thermal_paused:
                try:
                    os.killpg(self.validation["proc"].pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    # ------------------------------------------------------------------ validation
    def next_validation(self):
        row = self.con.execute("SELECT * FROM submissions WHERE status='pending' ORDER BY id LIMIT 1").fetchone()
        if row is not None:
            return row, "submission"
        if self.cfg["validation"].get("reverify_imports"):
            for r in self.con.execute("SELECT * FROM submissions WHERE source LIKE 'import:%' AND status IN "
                                      "('provisional','active') ORDER BY id"):
                v = json.loads(r["validation"] or "{}")
                if v.get("method") == "imported" and not v.get("reverify"):
                    return r, "reverify"
        return None, None

    def maybe_validate(self):
        # An accepted receipt settles a submission without holding the slot, so the next one can go in the same tick.
        for _ in range(8):
            if self.validation or not self.cfg["validation"].get("enabled", True) or self.slots_free() <= 0:
                return
            if self.start_validation() != "receipt":
                return

    def check_receipt(self, sub, manifest, out):
        """validation.accept_receipts: the desktop cold-verify receipt (deploy_checkpoint.py's VERIFY.json) next to a
        NAS-registered archive. -> (receipt info for the validation record, validation result when accepted else
        None). The exact bytes judged are hashed, and kept as receipt.json in the validation folder when accepted."""
        path = Path(sub["archive_path"]).parent / str(self.cfg["validation"].get("receipt_name") or "VERIFY.json")
        info = {"path": str(path), "accepted": False}
        if sub["source"] != "path":
            info["problems"] = [f"no receipt for a {sub['source']!r} submission (only archive_path registrations)"]
            return info, None
        try:
            with open(path, "rb") as handle:
                data = handle.read(RECEIPT_MAX_BYTES + 1)
        except OSError as error:
            info["problems"] = [f"no readable receipt: {type(error).__name__}: {error}"[:300]]
            return info, None
        info.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        if len(data) > RECEIPT_MAX_BYTES:
            info["problems"] = [f"receipt over {RECEIPT_MAX_BYTES} bytes"]
            return info, None
        try:
            rec = json.loads(data)
        except ValueError as error:
            info["problems"] = [f"receipt is not JSON: {error}"[:300]]
            return info, None
        try:
            problems, verifier, source = receipt_problems(rec, sub, manifest, self.cfg)
            latency = latency_rows(rec) if not problems else None
        except Exception as error:  # noqa: BLE001 - a malformed receipt falls back to the cold verify, never passes
            problems, verifier, source = [f"receipt check error: {type(error).__name__}: {error}"[:300]], None, None
        info.update(schema=rec.get("schema") if isinstance(rec, dict) else None, verifier_sha256=verifier,
                    verifier_source=source)
        if problems:
            info["problems"] = problems[:20]
            return info, None
        copy = out / "receipt.json"
        try:
            with open(copy, "xb") as handle:
                handle.write(data)
        except OSError as error:
            info["problems"] = [f"could not keep a copy of the receipt: {type(error).__name__}: {error}"[:300]]
            return info, None
        info.update(accepted=True, copy=str(copy), receipt_archive=rec.get("archive"))
        res = {"method": "desktop-receipt", "passed": True, "utc": core.now_utc(), "dir": str(out),
               "complete": True, "fallbacks": 0, "archive_sha256": rec["archive_sha256"],
               "latency_ms": latency, "latency_measured_on": "desktop (receipt)", "receipt": info}
        return info, res

    def start_validation(self):
        """Validate the next queued entry. -> "receipt" when an accepted receipt settled it, else None."""
        sub, kind = self.next_validation()
        if sub is None:
            return None
        vc = self.cfg["validation"]
        receipts = kind == "submission" and bool(vc.get("accept_receipts"))
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out = self.data / "validation" / f"{sub['id']}-{sub['name']}" / stamp
        out.mkdir(parents=True, exist_ok=True)
        if kind == "submission":
            core.set_status(self.con, sub["id"], "validating",
                            "checking the desktop cold-verify receipt" if receipts else "cold verify started")
            self.con.commit()
        try:
            sha = core.sha256_file(sub["archive_path"])
            if sha != sub["archive_sha256"]:
                raise ValueError(f"archive sha256 {sha} != registered {sub['archive_sha256']}")
            manifest = core.inspect_archive(sub["archive_path"], int(self.cfg["service"]["max_upload_mb"]) << 20)
        except (OSError, ValueError) as error:
            self._validation_result(sub, kind, {"method": "cold-verify", "passed": False, "utc": core.now_utc(),
                                                "error": f"{type(error).__name__}: {error}"})
            return None
        receipt = None
        if receipts:
            receipt, res = self.check_receipt(sub, manifest, out)
            if res is not None:
                res["manifest"] = manifest
                try:
                    prior = json.loads(sub["validation"] or "{}")
                except ValueError:
                    prior = {}
                res["attempts"] = int(prior.get("attempts", 0) or 0) + 1 if isinstance(prior, dict) else 1
                core.event(self.con, "receipt_accepted", submission=sub["id"], name=sub["name"],
                           receipt=receipt["path"], sha256=receipt["sha256"])
                self._validation_result(sub, kind, res, passed_reason=(
                    f"desktop cold-verify receipt {receipt['path']} (sha256 {receipt['sha256']}) accepted: both seats "
                    f"complete, 0 fallbacks, within the 60 s bank"))
                return "receipt"
            core.event(self.con, "receipt_not_used", submission=sub["id"], name=sub["name"], receipt=receipt["path"],
                       sha256=receipt.get("sha256"), problems=receipt["problems"][:5])
            self.con.execute("UPDATE submissions SET status_reason=?, updated_utc=? WHERE id=? AND status='validating'",
                             (f"cold verify started (desktop receipt not used: {receipt['problems'][0]})"[:500],
                              core.now_utc(), sub["id"]))
            self.con.commit()
            self.log(f"receipt for {sub['name']} not used: {'; '.join(receipt['problems'][:3])}")
        g = self.cfg["games"]
        verifier = f"{g['engine']}/work/runs/maintainer/primitive-bc-v1-deploy-v4land-a1/build_archive.py"
        cmd = ["nice", "-n", str(int(g["nice"])), g["python"], "-B", verifier, "verify", "--archive", sub["archive_path"],
               "--opponents", vc["opponents"], "--seats", vc["seats"], "--out", str(out / "verify.json")]
        log = open(out / "verify.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=g["tmpdir"], env=self.env(),
                                start_new_session=True)
        log.close()
        self.validation = {"sub_id": sub["id"], "kind": kind, "proc": proc, "dir": out, "started": time.time(),
                           "load1": round(os.getloadavg()[0], 2), "running": len(self.procs), "manifest": manifest,
                           "verifier": verifier, "receipt": receipt}
        self.log(f"validation of {sub['name']} ({kind}) started")
        return None

    def finish_validation(self, v, rc):
        sub = self.con.execute("SELECT * FROM submissions WHERE id=?", (v["sub_id"],)).fetchone()
        res = {"method": "cold-verify", "utc": core.now_utc(), "dir": str(v["dir"]), "exit": rc,
               "runtime": self.cfg["games"]["python"], "verifier": v["verifier"],
               "seconds": round(time.time() - v["started"], 1),
               "machine_load": {"load1_start": v["load1"], "load1_end": round(os.getloadavg()[0], 2),
                                "games_running_at_start": v["running"], "games_running_at_end": len(self.procs)}}
        path = v["dir"] / "verify.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text())
                lat = latency_rows(data)
                res.update({"complete": bool(data.get("complete")), "fallbacks": data.get("fallbacks"),
                            "latency_ms": lat, "archive_sha256": data.get("archive_sha256")})
                res["passed"] = (rc == 0 and data.get("complete") is True and data.get("fallbacks") == 0
                                 and data.get("archive_sha256") == sub["archive_sha256"])
                if not res["passed"]:
                    res["error"] = (f"cold verify failed: exit={rc}, complete={data.get('complete')}, "
                                    f"fallbacks={data.get('fallbacks')}, "
                                    f"archive_sha256_match={data.get('archive_sha256') == sub['archive_sha256']}")
            except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
                res.update({"passed": False, "error": f"invalid verify.json: {type(error).__name__}: {error}"[:800]})
        else:
            res.update({"passed": False, "error": f"no verify.json (exit {rc}): {tail(v['dir'] / 'verify.log', 800)}"})
        res["manifest"] = v.get("manifest")
        res["thermal_paused"] = bool(v.get("paused"))
        if v.get("receipt"):
            res["receipt"] = v["receipt"]         # why the desktop receipt did not stand in for this verify
        prior = json.loads(sub["validation"] or "{}") if v["kind"] == "submission" else {}
        attempts = int(prior.get("attempts", 0)) + 1 if isinstance(prior, dict) else 1
        res["attempts"] = attempts
        if v["kind"] == "submission" and not res.get("passed") and v.get("paused") and attempts < 3:
            self.con.execute("UPDATE submissions SET validation=?, updated_utc=? WHERE id=?",
                             (json.dumps(res), core.now_utc(), sub["id"]))
            core.set_status(self.con, sub["id"], "pending", "validation interrupted by a thermal pause; retrying")
            self.con.commit()
            self.log(f"validation of {sub['name']} was thermally paused and did not pass; re-queued")
            return
        self._validation_result(sub, v["kind"], res)

    def _validation_result(self, sub, kind, res, passed_reason="cold verify complete, 0 fallbacks"):
        if kind == "submission":
            self.con.execute("UPDATE submissions SET validation=?, validated_utc=?, manifest=COALESCE(?, manifest), "
                             "updated_utc=? WHERE id=?",
                             (json.dumps(res), core.now_utc(), json.dumps(res.get("manifest")) if res.get("manifest")
                              else None, core.now_utc(), sub["id"]))
            if res.get("passed"):
                core.set_status(self.con, sub["id"], "provisional", passed_reason[:500])
            else:
                core.set_status(self.con, sub["id"], "failed", res.get("error", "validation failed")[:500])
        else:
            v = json.loads(sub["validation"] or "{}")
            v["reverify"] = res
            self.con.execute("UPDATE submissions SET validation=?, updated_utc=? WHERE id=?",
                             (json.dumps(v), core.now_utc(), sub["id"]))
            if not res.get("passed"):
                core.event(self.con, "reverify_failed", submission=sub["id"], name=sub["name"], error=res.get("error"))
        self.con.commit()
        self.dirty = True
        self.log(f"validation of {sub['name']} ({kind}): passed={res.get('passed')} {res.get('error', '')}")

    # ------------------------------------------------------------------ thermal / ratings / heartbeat
    def signal_all(self, sig):
        for info in list(self.procs.values()) + ([self.validation] if self.validation else []):
            try:
                os.killpg(info["proc"].pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

    def thermal_tick(self):
        if time.time() - self.last_thermal < float(self.cfg["thermal"]["check_every_s"]):
            return
        self.last_thermal = time.time()
        st = self.guard.check()
        if st["paused"] and not self.thermal_paused:
            self.signal_all(signal.SIGSTOP)
            self.thermal_paused = True
            self.thermal_pause_started = time.time()
            if self.validation:
                self.validation["paused"] = True     # its per-request deadlines may expire across the pause
            if self.procs:
                self.con.execute(f"UPDATE episodes SET thermal_paused=1 WHERE id IN ({','.join('?' * len(self.procs))})",
                                 list(self.procs))
            core.event(self.con, "thermal_pause", **{k: st.get(k) for k in ("rule", "value_c", "pause_at_c")})
            self.log(f"thermal pause: {st['rule']} {st.get('value_c')} C")
            self.con.commit()
        elif not st["paused"] and self.thermal_paused:
            self.signal_all(signal.SIGCONT)
            self.thermal_paused = False
            # Exclude only the pause duration, not the active time already consumed.
            duration = max(0.0, time.time() - self.thermal_pause_started) if self.thermal_pause_started is not None else 0.0
            for info in self.procs.values():
                info["started"] += duration
            self.thermal_pause_started = None
            core.event(self.con, "thermal_resume", **{k: st.get(k) for k in ("rule", "value_c", "resume_below_c")})
            self.log(f"thermal resume: {st['rule']} {st.get('value_c')} C")
            self.con.commit()

    def maybe_rate(self, force=False):
        if not (self.dirty or force):
            return
        if not force and time.time() - self.last_rating < float(self.cfg["ratings"]["min_interval_s"]):
            return
        began = time.time()
        rows, changed = core.recompute(self.con, self.cfg)
        for name, status in changed:
            core.event(self.con, "status_change", name=name, status=status)
            self.log(f"{name} -> {status}")
        core.meta_set(self.con, "rating_seconds", round(time.time() - began, 3))
        self.con.commit()
        self.dirty = False
        self.last_rating = time.time()

    def heartbeat(self):
        g = self.cfg["games"]
        admin = core.meta_get(self.con, "admin_paused", False)
        queued = self.con.execute("SELECT COUNT(*) AS n FROM episodes WHERE state='queued'").fetchone()["n"]
        core.meta_set(self.con, "worker", {
            "utc": core.now_utc(), "epoch": time.time(), "pid": os.getpid(), "running_games": len(self.procs),
            "running_episode_ids": sorted(self.procs), "max_concurrent_games": int(g["max_concurrent_games"]),
            "validating": ({"submission": self.validation["sub_id"], "kind": self.validation["kind"]}
                           if self.validation else None),
            "queued_episodes": queued, "thermal": self.guard.state, "thermal_paused": self.thermal_paused,
            "admin_paused": bool(admin), "games_enabled": bool(g.get("enabled", True)),
            "mem_available_gb": round(mem_available_gb(), 1), "load": [round(x, 2) for x in os.getloadavg()],
            "engine": g["engine"], "version": core.SERVICE_VERSION})
        self.con.commit()

    def tick(self):
        self.reap()
        self.thermal_tick()
        admin = core.meta_get(self.con, "admin_paused", False)
        if not (self.thermal_paused or admin) and self.cfg["games"].get("enabled", True):
            self.maybe_validate()
            self.fill_slots()
        self.maybe_rate()
        self.heartbeat()

    def run(self):
        self.startup()
        self.maybe_rate(force=True)
        while not self.stopping:
            try:
                self.tick()
            except Exception as error:  # noqa: BLE001 - keep scheduling; the error is logged and recorded
                self.log(f"tick error: {type(error).__name__}: {error}")
                try:
                    self.con.rollback()
                    core.event(self.con, "worker_error", error=f"{type(error).__name__}: {error}"[:500])
                    self.con.commit()
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(2)
        self.shutdown()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    cfg = core.load_config(a.config)
    w = Worker(cfg, log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    signal.signal(signal.SIGTERM, w.stop)
    signal.signal(signal.SIGINT, w.stop)
    if a.once:
        w.startup()
        w.tick()
        return 0
    w.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
