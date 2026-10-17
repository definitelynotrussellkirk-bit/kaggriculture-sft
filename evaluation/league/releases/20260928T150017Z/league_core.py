"""Shared core of the farmNEAT league service: configuration, SQLite state, archive checks, ratings, matchmaking and the
thermal rule. Standard library only (the API and worker run on the container's system python3).

Contract: CONTRACT.md in this directory (the version is CONTRACT_VERSION below)."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import tarfile
import time
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - python < 3.11
    tomllib = None

CONTRACT_VERSION = "1.0.0"
SERVICE_VERSION = "1.0.0"
LEAGUE_EPISODE_BASE = 7_100_000_000_000_000        # replay EpisodeId = base + episode id (catalog: >= 10**12 is local)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
STATUSES = ("pending", "validating", "provisional", "active", "retired", "failed")
PLAYABLE = ("provisional", "active")
ELO = 400.0 / math.log(10.0)

DEFAULTS = {
    "service": {"listen": "0.0.0.0", "port": 80, "db": "/var/lib/league/league.sqlite3",
                "data_dir": "/mnt/ai/scratch/farmNEAT/league", "token_file": "/etc/league/api-token",
                "public_url": "http://league.lan", "path_prefixes": ["/mnt/ai/"], "max_upload_mb": 100},
    "games": {"enabled": True, "max_concurrent_games": 6, "nice": 10,
              "python": "/mnt/ai/scratch/farmNEAT/runtime-hosted-py311/venv/bin/python",
              "engine": "/mnt/ai/scratch/farmNEAT/league/engine/engine-20260927-b31876a9",
              "runner": "/opt/league/current/league_game.py", "template": "/opt/league/current/replay-template.json",
              "tmpdir": "/var/lib/league/tmp", "steps": 719, "sampler_seed": 7100, "hosted_rules": False,
              "game_timeout_s": 3600, "seed_base": 31000000, "min_free_mem_gb": 4.0, "max_attempts": 3,
              "new_entry_share": 0.6},
    "validation": {"enabled": True, "timeout_s": 3600, "reverify_imports": True, "opponents": "PASS",
                   "seats": "0,1",
                   # Desktop cold-verify receipt in place of the league's own cold verify (CONTRACT.md section 2).
                   # Off by default: turning it on is an explicit deployment step.
                   "accept_receipts": False, "receipt_name": "VERIFY.json",
                   "receipt_schemas": ["primitive-bc-cold-archive-verify-v2"],
                   # sha256 of accepted verifier builds: primitive-bc-v1-deploy-v8/build_archive.py (2026-09-27 21:44),
                   # whose verify() is the league verifier's (v4land-a1) plus the 60 s time-bank gate.
                   "receipt_verifiers": ["c56789e636824866f5a24f705fbe6cdda2c18f1e32d9c155f44ceea930a394ed"]},
    "ratings": {"anchor": "fwd1-46236", "anchor_rating": 2880.0, "provisional_games": 20, "max_active": 16,
                "auto_retire": True, "prior_draws": 1.0, "min_interval_s": 30},
    "thermal": {"enabled": True, "file": "/mnt/host-thermal/thermal.json", "max_age_s": 180,
                "inlet_pause_c": 34.0, "inlet_resume_c": 32.0, "cpu_pause_c": 80.0, "cpu_resume_c": 72.0,
                "fail_closed": True, "check_every_s": 15},
    "retention": {"replay_days": 30, "db_snapshot_hours": 168},
    "export": {"enabled": True, "dir": "/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/league"},
    "kaggle": {"replay_download_enabled": False},
}


# ----------------------------------------------------------------------------- config / time
def load_config(path=None):
    cfg = copy.deepcopy(DEFAULTS)
    path = path or os.environ.get("LEAGUE_CONFIG", "/etc/league/league.toml")
    if path and Path(path).is_file():
        if tomllib is None:
            raise RuntimeError("tomllib unavailable")
        with open(path, "rb") as handle:
            user = tomllib.load(handle)
        for section, values in user.items():
            if isinstance(values, dict):
                cfg.setdefault(section, {}).update(values)
            else:
                cfg[section] = values
    cfg["_path"] = str(path)
    return cfg


def now_utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text):
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- database
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS submissions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL,
  status_reason TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL,
  archive_path TEXT NOT NULL,
  archive_sha256 TEXT NOT NULL,
  archive_bytes INTEGER NOT NULL,
  manifest TEXT,
  validation TEXT,
  created_utc TEXT NOT NULL,
  updated_utc TEXT NOT NULL,
  validated_utc TEXT
);
CREATE TABLE IF NOT EXISTS status_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, submission_id INTEGER NOT NULL, utc TEXT NOT NULL,
  old TEXT, new TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '', actor TEXT NOT NULL DEFAULT 'system');
CREATE TABLE IF NOT EXISTS matches (
  id INTEGER PRIMARY KEY AUTOINCREMENT, world_seed INTEGER UNIQUE NOT NULL, a INTEGER NOT NULL, b INTEGER NOT NULL,
  created_utc TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT 'league');
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id INTEGER,
  world_seed INTEGER NOT NULL,
  seat0 INTEGER NOT NULL,
  seat1 INTEGER NOT NULL,
  state TEXT NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 1,
  created_utc TEXT NOT NULL,
  started_utc TEXT,
  finished_utc TEXT,
  cash0 REAL, cash1 REAL, score0 REAL,
  rated INTEGER NOT NULL DEFAULT 0,
  unrated_reason TEXT,
  fallbacks0 INTEGER, fallbacks1 INTEGER,
  status0 TEXT, status1 TEXT,
  steps INTEGER, seconds REAL, ms_p50_0 REAL, ms_p50_1 REAL,
  episode_dir TEXT, replay_path TEXT, replay_verified INTEGER,
  thermal_paused INTEGER NOT NULL DEFAULT 0,
  load1 REAL, running_at_start INTEGER,
  runtime TEXT,
  source TEXT NOT NULL DEFAULT 'league',
  error TEXT
);
CREATE INDEX IF NOT EXISTS episodes_state ON episodes(state);
CREATE INDEX IF NOT EXISTS episodes_seat0 ON episodes(seat0);
CREATE INDEX IF NOT EXISTS episodes_seat1 ON episodes(seat1);
CREATE INDEX IF NOT EXISTS episodes_finished ON episodes(finished_utc);
CREATE TABLE IF NOT EXISTS ratings (
  submission_id INTEGER PRIMARY KEY, rating REAL, sigma REAL, lo REAL, hi REAL, games INTEGER, wins INTEGER,
  draws INTEGER, losses INTEGER, win_share REAL, mean_margin REAL, rank INTEGER, computed_utc TEXT);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, utc TEXT NOT NULL, kind TEXT NOT NULL,
  detail TEXT NOT NULL DEFAULT '');
"""


def connect(path, *, readonly=False, init=True):
    """SQLite connection (WAL). ``init`` creates the schema (once per process is enough: the API's per-request
    connections pass init=False so a read never takes a write lock)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30, check_same_thread=False)
    else:
        con = sqlite3.connect(path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    if not readonly and init:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.executescript(SCHEMA)
        con.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '1')")
        con.commit()
    return con


def meta_get(con, key, default=None):
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def meta_set(con, key, value):
    con.execute("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)))


def event(con, kind, **detail):
    con.execute("INSERT INTO events(utc, kind, detail) VALUES (?, ?, ?)", (now_utc(), kind, json.dumps(detail)))


def set_status(con, sub_id, new, reason="", actor="system", *, expected=None):
    """Compare-and-set a status; stale worker snapshots must not undo an admin change."""
    row = con.execute("SELECT status FROM submissions WHERE id=?", (sub_id,)).fetchone()
    old = row["status"] if row else None
    if old is None or old == new or (expected is not None and old != expected):
        return False
    changed = con.execute("UPDATE submissions SET status=?, status_reason=?, updated_utc=? WHERE id=? AND status=?",
                          (new, reason, now_utc(), sub_id, old))
    if changed.rowcount != 1:
        return False
    con.execute("INSERT INTO status_log(submission_id, utc, old, new, reason, actor) VALUES (?, ?, ?, ?, ?, ?)",
                (sub_id, now_utc(), old, new, reason, actor))
    return True


def row_dict(row, json_fields=("manifest", "validation", "runtime")):
    if row is None:
        return None
    out = dict(row)
    for k in json_fields:
        if k in out and isinstance(out[k], str):
            try:
                out[k] = json.loads(out[k])
            except ValueError:
                pass
    return out


def find_submission(con, key):
    """Submission by numeric id or by name."""
    key = str(key)
    if key.isdigit():
        row = con.execute("SELECT * FROM submissions WHERE id=?", (int(key),)).fetchone()
        if row:
            return row
    return con.execute("SELECT * FROM submissions WHERE name=?", (key,)).fetchone()


# ----------------------------------------------------------------------------- archives
def inspect_archive(path, max_bytes=100 * 1024 * 1024):
    """Kaggle archive checks that need no execution: size, gzip tar, regular members with safe relative names, main.py
    present; MANIFEST.json summary when present. -> dict; raises ValueError on a bad archive."""
    p = Path(path)
    size = p.stat().st_size
    if size > max_bytes:
        raise ValueError(f"archive is {size} bytes, over the {max_bytes} byte limit")
    names = []
    manifest = None
    try:
        with tarfile.open(p, "r:gz") as tar:
            for m in tar.getmembers():
                if m.isdir():
                    continue
                if not m.isfile():
                    raise ValueError(f"non-regular member {m.name}")
                if m.name.startswith("/") or ".." in Path(m.name).parts:
                    raise ValueError(f"unsafe member name {m.name}")
                names.append(m.name)
            if "MANIFEST.json" in names:
                manifest = json.loads(tar.extractfile("MANIFEST.json").read())
    except (tarfile.TarError, OSError, EOFError) as error:
        raise ValueError(f"not a readable tar.gz: {error}") from None
    if "main.py" not in names:
        raise ValueError("main.py missing at the archive root")
    summary = {"members": len(names), "bytes": size}
    if manifest is not None:
        summary.update({k: manifest.get(k) for k in ("schema", "declared", "weights_storage", "model_spec",
                                                      "model_sha256", "builder_sha256") if k in manifest})
    else:
        summary["schema"] = None
    return summary


# ----------------------------------------------------------------------------- ratings
def episode_score0(cash0, cash1, status0="DONE", status1="DONE"):
    ok0, ok1 = status0 == "DONE", status1 == "DONE"
    if ok0 and ok1:
        return 1.0 if cash0 > cash1 else 0.0 if cash0 < cash1 else 0.5
    return 1.0 if ok0 else 0.0 if ok1 else 0.5


def fit_bt(ids, results, anchor, *, prior_draws=1.0, iters=5000, tol=1e-10):
    """Bradley-Terry MAP (MM iterations, Hunter 2004) with ``prior_draws`` virtual draws of every entry against the
    anchor (the prototype league's prior). results: [(seat0_id, seat1_id, seat0_score)]. -> {id: theta (natural log,
    anchor 0)}, pair table {(i, j): [score_i, n]} used by fisher_sigma."""
    idx = {n: i for i, n in enumerate(ids)}
    pairs = {}
    for a, b, s in results:
        if a in idx and b in idx and a != b:
            i, j = (idx[a], idx[b]) if idx[a] < idx[b] else (idx[b], idx[a])
            s_i = s if idx[a] == i else 1.0 - s
            cell = pairs.setdefault((i, j), [0.0, 0.0])
            cell[0] += s_i
            cell[1] += 1.0
    anc = idx.get(anchor)
    if anc is not None and prior_draws > 0:
        for i in range(len(ids)):
            if i != anc:
                key = (i, anc) if i < anc else (anc, i)
                cell = pairs.setdefault(key, [0.0, 0.0])
                cell[0] += 0.5 * prior_draws
                cell[1] += prior_draws
    p = [1.0] * len(ids)
    for _ in range(iters):
        wins = [0.0] * len(ids)
        denom = [0.0] * len(ids)
        for (i, j), (s, n) in pairs.items():
            wins[i] += s
            wins[j] += n - s
            d = n / (p[i] + p[j])
            denom[i] += d
            denom[j] += d
        new = [max(1e-12, wins[i] / denom[i]) if denom[i] > 0 else p[i] for i in range(len(ids))]
        if anc is not None:
            base = new[anc]
            new = [x / base for x in new]
        delta = max((abs(math.log(new[i]) - math.log(p[i])) for i in range(len(ids))), default=0.0)
        p = new
        if delta < tol:
            break
    base = p[anc] if anc is not None else math.exp(sum(math.log(x) for x in p) / max(1, len(p)))
    return {n: math.log(p[idx[n]] / base) for n in ids}, pairs


def _invert(m):
    n = len(m)
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(a[r][c]))
        if abs(a[piv][c]) < 1e-15:
            raise ValueError("singular information matrix")
        a[c], a[piv] = a[piv], a[c]
        f = a[c][c]
        a[c] = [x / f for x in a[c]]
        for r in range(n):
            if r != c and a[r][c] != 0.0:
                g = a[r][c]
                a[r] = [x - g * y for x, y in zip(a[r], a[c])]
    return [row[n:] for row in a]


def fisher_sigma(ids, theta, pairs, anchor):
    """Standard errors (natural-log scale) from the observed Fisher information of the BT likelihood (including the
    prior's virtual games), with the anchor fixed at 0. -> {id: sigma}; the anchor's sigma is 0."""
    idx = {n: i for i, n in enumerate(ids)}
    anc = idx.get(anchor)
    free = [i for i in range(len(ids)) if i != anc]
    pos = {i: k for k, i in enumerate(free)}
    h = [[0.0] * len(free) for _ in free]
    th = [theta[n] for n in ids]
    for (i, j), (_s, n) in pairs.items():
        pij = 1.0 / (1.0 + math.exp(th[j] - th[i]))
        w = n * pij * (1.0 - pij)
        for x, y, sign in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
            if x in pos and y in pos:
                h[pos[x]][pos[y]] += sign * w
    out = {ids[anc]: 0.0} if anc is not None else {}
    if not free:
        return out
    try:
        inv = _invert(h)
        for i in free:
            out[ids[i]] = math.sqrt(max(0.0, inv[pos[i]][pos[i]]))
    except ValueError:
        for i in free:
            out[ids[i]] = float("inf")
    return out


def compute_board(entries, games, cfg):
    """entries: [{id, name, status}], games: [(seat0_id, seat1_id, cash0, cash1, score0)] of rated episodes.
    -> list of rows sorted by rating (all entries that have any game or are playable), each with rating, sigma,
    lo/hi (95%), games, wins, draws, losses, win_share, mean_margin, rank (1-based among non-failed)."""
    rc = cfg["ratings"]
    by_name = {e["name"]: e for e in entries}
    anchor = by_name.get(rc["anchor"])
    anchor_id = anchor["id"] if anchor else None
    ids = [e["id"] for e in entries if e["status"] in ("provisional", "active", "retired")]   # not pending/validating/failed
    # Use exactly the same real observations for the likelihood and establishment.
    # Excluding an unplayable identity from the fit must also exclude its games
    # from the other entry's activation count. Retired identities remain included.
    fit_ids = set(ids)
    games = [g for g in games if g[0] in fit_ids and g[1] in fit_ids and g[0] != g[1]
             and g[4] in (0.0, 0.5, 1.0)
             and all(isinstance(c, (int, float)) and math.isfinite(c) for c in g[2:4])]
    results = [(a, b, s) for a, b, _c0, _c1, s in games]
    theta, pairs = fit_bt(ids, results, anchor_id, prior_draws=float(rc["prior_draws"]))
    sigma = fisher_sigma(ids, theta, pairs, anchor_id)
    stats = {i: {"games": 0, "wins": 0, "draws": 0, "losses": 0, "margin": 0.0} for i in ids}
    for a, b, c0, c1, s in games:
        for me, own, other, score in ((a, c0, c1, s), (b, c1, c0, 1.0 - s)):
            st = stats.get(me)
            if st is None:
                continue
            st["games"] += 1
            st["wins" if score == 1.0 else "losses" if score == 0.0 else "draws"] += 1
            st["margin"] += own - other
    base = float(rc["anchor_rating"])
    rows = []
    for e in entries:
        if e["id"] not in theta:
            continue
        st = stats[e["id"]]
        r = base + ELO * theta[e["id"]]
        s = ELO * sigma.get(e["id"], float("inf"))
        rows.append({"id": e["id"], "name": e["name"], "status": e["status"], "rating": round(r, 1),
                     "sigma": round(s, 1) if math.isfinite(s) else None,
                     "lo": round(r - 1.96 * s, 1) if math.isfinite(s) else None,
                     "hi": round(r + 1.96 * s, 1) if math.isfinite(s) else None,
                     "games": st["games"], "wins": st["wins"], "draws": st["draws"], "losses": st["losses"],
                     "win_share": round((st["wins"] + 0.5 * st["draws"]) / st["games"], 4) if st["games"] else None,
                     "mean_margin": round(st["margin"] / st["games"], 1) if st["games"] else None})
    # Display rounding is not a tie in fitted strength (and must not choose the leader).
    rows.sort(key=lambda x: (-theta[x["id"]], x["id"]))
    for k, row in enumerate(rows, 1):
        row["rank"] = k
    return rows


def leader(rows, min_games=20):
    """The league leader: the top-rated playable entry, when it is established (status active, >= provisional_games
    rated games); None while the top entry is still provisional."""
    top = next((r for r in rows if r["status"] in PLAYABLE), None)
    return top if top is not None and top["status"] == "active" and top.get("games", 0) >= min_games else None


def store_board(con, rows):
    stamp = now_utc()
    con.execute("DELETE FROM ratings")
    con.executemany("INSERT INTO ratings(submission_id, rating, sigma, lo, hi, games, wins, draws, losses, win_share, "
                    "mean_margin, rank, computed_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(r["id"], r["rating"], r["sigma"], r["lo"], r["hi"], r["games"], r["wins"], r["draws"],
                      r["losses"], r["win_share"], r["mean_margin"], r["rank"], stamp) for r in rows])
    meta_set(con, "board_computed_utc", stamp)


def rated_games(con):
    return [(r["seat0"], r["seat1"], r["cash0"], r["cash1"], r["score0"]) for r in con.execute(
        "SELECT seat0, seat1, cash0, cash1, score0 FROM episodes WHERE state='done' AND rated=1 ORDER BY id")]


def recompute(con, cfg):
    """Refit ratings over every rated episode, store them, apply provisional->active and the retire rule."""
    entries = [dict(r) for r in con.execute("SELECT id, name, status FROM submissions ORDER BY id")]
    rows = compute_board(entries, rated_games(con), cfg)
    rc = cfg["ratings"]
    changed = []
    for row in rows:
        # Repair already-promoted identities too; do not reactivate retired ones.
        if row["status"] == "active" and row["games"] < int(rc["provisional_games"]):
            if set_status(con, row["id"], "provisional", f"only {row['games']} games in current fit"):
                row["status"] = "provisional"
                changed.append((row["name"], "provisional"))
        if row["status"] == "provisional" and row["games"] >= int(rc["provisional_games"]):
            if set_status(con, row["id"], "active", f"{row['games']} rated games", expected="provisional"):
                row["status"] = "active"
                changed.append((row["name"], "active"))
        # A fit can overlap an API retirement. Refresh even when the CAS was rejected.
        row["status"] = con.execute("SELECT status FROM submissions WHERE id=?", (row["id"],)).fetchone()["status"]
    if rc.get("auto_retire"):
        est = [r for r in rows if r["status"] == "active"]
        for row in est[int(rc["max_active"]):]:
            if row["name"] != rc["anchor"]:
                if set_status(con, row["id"], "retired", f"outside the best {rc['max_active']} active entries", expected="active"):
                    row["status"] = "retired"
                    changed.append((row["name"], "retired"))
    store_board(con, rows)
    con.commit()
    return rows, changed


# ----------------------------------------------------------------------------- matchmaking
def win_prob(ra, rb):
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def choose_match(playable, board, pair_counts, inflight, rng, cfg):
    """Kaggle-like pairing. playable: [{id, name, status}]; board: {id: {rating, sigma, games}}; pair_counts:
    {(min_id, max_id): rated games}; inflight: {id: queued+running episodes}. -> (a_id, b_id, reason) or None.

    * A new entry (fewer than provisional_games rated games) is picked with probability new_entry_share while one
      exists, fewest games (+ in-flight) first, so new submissions get many games early.
    * Otherwise A is drawn with weight (sigma/100)^2 + 0.25 (uncertain ratings play more).
    * The opponent is drawn with weight 4 p (1 - p) (close ratings; p = BT win probability) x (1 + sigma_B/200)
      / (1 + games already played by the pair / 8), so pairings favour uncertain and close ratings and spread over
      opponents.
    * Both uncertainty weights are divided by (1 + queued/running episodes of the entry)^2, so matches created back to
      back (e.g. after a restart) do not pile onto one entry."""
    if len(playable) < 2:
        return None
    rc = cfg["ratings"]
    need = int(rc["provisional_games"])
    default = {"rating": float(rc["anchor_rating"]), "sigma": 350.0, "games": 0}

    def info(e):
        b = board.get(e["id"]) or default
        sig = b.get("sigma")
        return (b.get("rating") if b.get("rating") is not None else default["rating"],
                sig if sig is not None else 350.0, b.get("games") or 0)

    def busy(e):                     # queued + running episodes: spread concurrent matches over entries
        return (1.0 + inflight.get(e["id"], 0)) ** 2

    new = [e for e in playable if info(e)[2] < need]
    if new and rng.random() < float(cfg["games"].get("new_entry_share", 0.6)):
        a = min(new, key=lambda e: (info(e)[2] + inflight.get(e["id"], 0), rng.random()))
        reason = "new-entry"
    else:
        weights = [((info(e)[1] / 100.0) ** 2 + 0.25) / busy(e) for e in playable]
        a = rng.choices(playable, weights)[0]
        reason = "uncertainty"
    ra = info(a)[0]
    others = [e for e in playable if e["id"] != a["id"]]
    ws = []
    for e in others:
        rb, sb, _g = info(e)
        p = win_prob(ra, rb)
        key = (min(a["id"], e["id"]), max(a["id"], e["id"]))
        w = 4.0 * p * (1.0 - p) * (1.0 + sb / 200.0) / (1.0 + pair_counts.get(key, 0) / 8.0) / busy(e)
        ws.append(max(w, 1e-4))
    b = rng.choices(others, ws)[0]
    return a["id"], b["id"], reason


# ----------------------------------------------------------------------------- thermal
class ThermalGuard:
    """Pause rule with hysteresis. Primary: the R730xd inlet temperature from the host's IPMI feed (pause at
    >= inlet_pause_c, resume below inlet_resume_c). Fallback when the feed is missing or older than max_age_s: the
    host CPU package temperature (coretemp, visible in the privileged CT; pause at >= cpu_pause_c, resume below
    cpu_resume_c). Neither readable: pause when fail_closed."""

    def __init__(self, cfg, hwmon_root="/sys/class/hwmon"):
        self.cfg = cfg["thermal"]
        self.hwmon_root = hwmon_root
        self.paused = False
        self.state = {}

    def read_inlet(self):
        rec = None
        try:
            rec = json.loads(Path(self.cfg["file"]).read_text())
            if not isinstance(rec, dict):
                return None, None
            epoch = float(rec.get("epoch") or 0)
            inlet = float(rec["inlet_c"])
            age = time.time() - epoch
            if not all(math.isfinite(x) for x in (epoch, inlet, age)) or not 0 <= age <= float(self.cfg["max_age_s"]):
                return None, rec
            return inlet, rec
        except (OSError, ValueError, TypeError, KeyError):
            return None, rec if isinstance(rec, dict) else None

    def read_cpu(self):
        temps = []
        root = Path(self.hwmon_root)
        try:
            for h in root.iterdir():
                try:
                    if (h / "name").read_text().strip() != "coretemp":
                        continue
                    for label in h.glob("temp*_label"):
                        if label.read_text().startswith("Package"):
                            temps.append(int((h / label.name.replace("_label", "_input")).read_text()) / 1000.0)
                except (OSError, ValueError):
                    continue
        except (OSError, ValueError):
            return None
        return max(temps) if temps else None

    def check(self):
        if not self.cfg.get("enabled", True):
            self.paused = False
            self.state = {"rule": "disabled", "paused": False, "utc": now_utc()}
            return self.state
        inlet, rec = self.read_inlet()
        if inlet is not None:
            rule, value = "inlet", inlet
            pause_at, resume_below = float(self.cfg["inlet_pause_c"]), float(self.cfg["inlet_resume_c"])
        else:
            value = self.read_cpu()
            rule = "cpu-package" if value is not None else "unreadable"
            pause_at, resume_below = float(self.cfg["cpu_pause_c"]), float(self.cfg["cpu_resume_c"])
        if value is None:
            self.paused = bool(self.cfg.get("fail_closed", True))
        elif self.paused:
            self.paused = value >= resume_below
        else:
            self.paused = value >= pause_at
        self.state = {"rule": rule, "value_c": value, "pause_at_c": pause_at, "resume_below_c": resume_below,
                      "paused": self.paused, "utc": now_utc(),
                      "feed": ({k: rec.get(k) for k in ("utc", "inlet_c", "exhaust_c", "cpu_c", "fans_rpm")}
                               if rec else None)}
        return self.state


def rng_for(seed=None):
    return random.Random(seed if seed is not None else time.time_ns())
