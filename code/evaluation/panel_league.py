#!/usr/bin/env python3
"""Panel league (owner 09-29 16:52-17:16): paired CRN panels are the primary board.
Records (append-only, /mnt/ai/scratch/farmNEAT/league/records/): games.jsonl (one immutable game record per (seed, seat0, seat1);
corrections are new records with "supersedes" + reason), panels.jsonl (dispatch/stop records), best_log.jsonl (BEST changes).
Everything else (worlds, pair stats, Elo/BT, BEST) is recomputed on demand from games.jsonl by ONE reducer.
Hard rule: a game counts only if crn_rng is true on engine-20260927-d91cbd52 (fixed-seed shops), replay verified, DONE/DONE,
0 fallbacks, 0 hard errors.
  python3 panel_league.py ingest        backfill/ingest every panel dir + league DB CRN games into games.jsonl
  python3 panel_league.py board         BOARD.md + PANELS.md (pair lines, BEST, T, R, BT ranks)
  python3 panel_league.py plan [--cap N] dispatch missing T/R panel games on WORLDS-T16 + screening, via SLURM
  python3 panel_league.py loop           ingest + board + plan every 5 min; hourly export to records/export/<UTC date>/
Pair line (same everywhere): p 95% [LB, UB] (world-cluster bootstrap, 2,000 draws; for < 8 worlds the lower of that and the
Jeffreys-Beta world interval) | p = (W + T/2)/n | Wilson games | P(p>.5) Beta | W-L-T | worlds complete | world wins/non-tied,
sign p | mean world margin [t95] median | per-seat-order margins | decided (A>B / B>A / undecided) | early-stop flag."""
import argparse, fcntl, glob, hashlib, json, math, os, re, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel_guard as PG                                    # ASTRA-PRO-PANEL-LEAGUE-STATS-0930: KL/Chernoff bounded-mean bound

R_DIR = Path("/mnt/ai/scratch/farmNEAT/league/records")
GAMES = R_DIR / "games.jsonl"
PANELS = R_DIR / "panels.jsonl"
BEST_LOG = R_DIR / "best_log.jsonl"
STATE = R_DIR / "dispatch-state.json"
SHA_CACHE = R_DIR / "sha-cache.json"
ENGINE = "engine-20260927-d91cbd52"
FARM_SBATCH = "/mnt/ai/scratch/farmNEAT/league/extfarm/farm_game.sbatch"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "root@192.168.88.2"]
REJECTED = Path("/mnt/ai/scratch/farmNEAT/league/records/REJECTED.jsonl")
REDUCER_VERSION = "panel-reducer-v3 (ASTRA guard merge: KL decision bound, oriented recompute, qualified/discovery split, provenance gate; was v2 (PRO-PANEL-LEAGUE-STATS-0930 hotfix + Hoeffding decision bound + pop-weighted)"
PANEL_DIRS = ["/mnt/ai/scratch/farmNEAT/league/extfarm/finalpanel/*", "/mnt/ai/scratch/farmNEAT/league/extfarm/matrix/*",
              "/mnt/ai/scratch/farmNEAT/league/extfarm/bo33/*", "/mnt/ai/scratch/farmNEAT/league/extworker/waves/*",
              "/mnt/ai/scratch/farmNEAT/colab-eval-0929/*", "/mnt/ai/scratch/farmNEAT/league/records/waves/*",
              "/mnt/ai/scratch/farmNEAT/league/records/finalists/*", "/mnt/ai/scratch/farmNEAT/league/records/land64/*",
              "/mnt/ai/scratch/farmNEAT/league/records/oa3screen/*",
              "/mnt/ai/scratch/farmNEAT/league/records/dagger64/*",
              "/mnt/ai/scratch/farmNEAT/league/records/nscreen/*",
              "/mnt/ai/scratch/farmNEAT/league/records/combo3066/*",
              "/mnt/ai/scratch/farmNEAT/league/records/surgery/*",
              "/mnt/ai/scratch/farmNEAT/league/records/regular/*",
              "/mnt/ai/scratch/farmNEAT/league/records/gpu/*",
              "/mnt/ai/scratch/farmNEAT/league/records/gaterun/*",
              "/mnt/ai/scratch/farmNEAT/league/records/ladder/run/*",
              "/mnt/ai/scratch/farmNEAT/league/records/nflow/run/*",
              "/mnt/ai/scratch/farmNEAT/league/records/salvage/run/*",
              "/mnt/ai/scratch/farmNEAT/league/records/colab/*",
              "/mnt/ai/scratch/farmNEAT/league/records/t4wrun/*",
              "/mnt/ai/scratch/farmNEAT/league/records/slurmlane/*",
              "/mnt/ai/scratch/farmNEAT/league/records/colabgpu/*",
              "/mnt/ai/scratch/farmNEAT/league/records/gpu-sp/*"]      # Colab #3 x86 CPU lane (coordinator 04:4x): whole worlds per batch
MIXED = [("/mnt/ai/scratch/farmNEAT/colab-eval-0929/ALL.tsv", "/mnt/ai/scratch/farmNEAT/colab-eval-0929/games", "import:codex-1864")]
EXCLUDED = ("base1-c-",)   # owner 21:33 "cut c entirely": registered, never scheduled or considered
R_NAMES = ["pub-ice-fire-wr1k-lw", "pub-shepherds-ledger-wr1k-lw", "pub-godsmode-shops-lw", "tape-mmpq-115130538-lw",
           "tape-decem-114962244-lw", "tape-boey-115025308-lw", "tape-victor-tufa-115017808-lw", "tape-marketshock-115158982-lw",
           "tape-marketshock-115161684-lw", "v4b-g2-48236-land-a1fx-carrydrain", "fwd1land-46236-rs0929"]
TAPES = {n for n in R_NAMES if n.startswith(("pub-", "tape-"))}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def worlds16():
    seeds = json.loads((R_DIR / "WORLDS-T16.json").read_text()).get("seeds")
    if (not isinstance(seeds, list) or len(seeds) != 16 or any(type(x) is not int for x in seeds) or len(set(seeds)) != 16):
        raise SystemExit("WORLDS-T16.json: exactly 16 distinct integer seeds required (fail closed)")
    return list(seeds)


# ------------------------------------------------------------------ league DB
def league_db():
    q = r"""
import sqlite3, json
c = sqlite3.connect('file:/var/lib/league/league.sqlite3?mode=ro', uri=True); c.row_factory = sqlite3.Row
subs = [dict(r) for r in c.execute("select id, name, status, archive_path, archive_sha256, archive_bytes from submissions")]
eps = [dict(r) for r in c.execute("select id, world_seed, seat0, seat1, state, cash0, cash1, rated, unrated_reason, fallbacks0, fallbacks1, status0, status1, steps, started_utc, finished_utc, replay_path, replay_verified, runtime, source from episodes where state='done'")]
print(json.dumps({"subs": subs, "eps": eps}))
"""
    out = subprocess.run(SSH + ["pct exec 118 -- python3 -"], input=q, capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-300:])
    return json.loads(out.stdout)


def sha_of(path, cache):
    try:
        st = os.stat(path)
    except OSError:
        return None, None
    key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
    if key not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        cache[key] = h.hexdigest()
    return cache[key], st.st_size


# ------------------------------------------------------------------ ingest
def existing_keys():
    keys = set()
    if GAMES.exists():
        with GAMES.open() as fh:
            for line in fh:
                try:
                    keys.add(json.loads(line)["key"])
                except (ValueError, KeyError):
                    pass
    return keys


def clean_of(g, rec):
    return (bool(rec.get("replay_verified")) and list(g.get("status") or []) == ["DONE", "DONE"] and not any(g.get("fallbacks") or [])
            and hard_errors_zero(hard_of(g)))


def hard_of(g):
    st = g.get("seat_stats")
    if not isinstance(st, list) or len(st) != 2:
        return None
    h = [x.get("hard_errors") if isinstance(x, dict) and "hard_errors" in x else None for x in st]   # PRO VERIFY-FIX-GATES #1: missing = ineligible
    return h if all(type(v) is int and v >= 0 for v in h) else None


def hard_errors_zero(counts):
    """THE hard-error definition (identical in r3-*/tools/panel_io.py, used by pairlab validate_native and release_plan receipts):
    exactly two per-seat counts from game.seat_stats[i].hard_errors, each an explicit int (not bool) equal to 0."""
    return isinstance(counts, list) and len(counts) == 2 and all(type(x) is int and x == 0 for x in counts)


def forfeit_seat(r):
    """RUNTIME-PARITY P0: a seat whose status is not DONE (TIMEOUT, ERROR, ...) forfeits: it loses (league_core.episode_score0); both
    non-DONE = draw. Returns 0/1 for a single forfeiting seat, 'both', or None for a normal DONE/DONE game."""
    s = r.get("status")
    if not isinstance(s, list) or len(s) != 2 or s == ["DONE", "DONE"]:
        return None
    bad = [i for i in (0, 1) if s[i] != "DONE"]
    return bad[0] if len(bad) == 1 else "both"


def record_issues(r):
    """Final read-boundary provenance gate (ASTRA F3): every counting game must attest all of these. A forfeit (one seat not DONE, e.g.
    TIMEOUT) is admitted and scored as a loss for that seat, never dropped (RUNTIME-PARITY P0); the non-forfeiting seat must still be clean."""
    bad = ["receipt_integrity_issues"] if r.get("integrity_issues") else []
    fs = forfeit_seat(r)
    if fs is not None:
        if r.get("crn_rng") is not True: bad.append("crn_not_true")
        if PG.normalize_engine(r.get("engine")) != ENGINE: bad.append("engine_mismatch")
        if type(r.get("world_seed")) is not int: bad.append("invalid_seed")
        if not isinstance(r.get("seat0"), str) or not isinstance(r.get("seat1"), str) or r.get("seat0") == r.get("seat1"): bad.append("invalid_players")
        sh = r.get("archive_sha256")
        if not isinstance(sh, list) or len(sh) != 2 or any(not isinstance(x, str) or not re.fullmatch("[0-9a-f]{64}", x) for x in sh):
            bad.append("archive_identity_missing")
        if fs in (0, 1):
            ok_ = 1 - fs; fb, he = r.get("fallbacks"), r.get("hard_errors")
            if not isinstance(fb, list) or len(fb) != 2 or type(fb[ok_]) is not int or fb[ok_] != 0: bad.append("forfeit_opponent_fallbacks_nonzero")
            if not isinstance(he, list) or len(he) != 2 or type(he[ok_]) is not int or he[ok_] != 0: bad.append("forfeit_opponent_hard_errors")
        return bad
    if r.get("clean") is not True: bad.append("clean_not_true")
    if r.get("crn_rng") is not True: bad.append("crn_not_true")
    if PG.normalize_engine(r.get("engine")) != ENGINE: bad.append("engine_mismatch")
    if r.get("replay_verified") is not True: bad.append("replay_unverified")
    if r.get("status") != ["DONE", "DONE"]: bad.append("status_not_done_done")
    fb = r.get("fallbacks")
    if not isinstance(fb, list) or len(fb) != 2 or any(type(x) is not int or x != 0 for x in fb): bad.append("fallbacks_missing_or_nonzero")
    if not hard_errors_zero(r.get("hard_errors")): bad.append("hard_errors_unattested_or_nonzero")
    c = r.get("cash")
    if not isinstance(c, list) or len(c) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in c):
        bad.append("cash_missing_or_nonfinite")
    if type(r.get("actions_recorded")) is not int or r["actions_recorded"] != 719: bad.append("actions_not_719_or_unattested")
    if type(r.get("world_seed")) is not int: bad.append("invalid_seed")
    if not isinstance(r.get("seat0"), str) or not isinstance(r.get("seat1"), str) or r.get("seat0") == r.get("seat1"): bad.append("invalid_players")
    sh = r.get("archive_sha256")
    if not isinstance(sh, list) or len(sh) != 2 or any(not isinstance(x, str) or not re.fullmatch("[0-9a-f]{64}", x) for x in sh):
        bad.append("archive_identity_missing")
    return bad


def backfill_provenance():
    """One-time append-only correction: attest hard errors / engine on pre-v3 records (superseding records with a reason)."""
    latest = {}
    for line in GAMES.open():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        latest[r.get("supersedes") or r["key"]] = r
    fixes = []
    for k, r in latest.items():
        if isinstance(r.get("hard_errors"), list):
            continue
        h = None; src = None
        prov = str(r.get("provenance") or "")
        if prov.startswith("league episode"):
            if r.get("clean"):
                h, src = [0, 0], "league rated=1 (worker rejects hard errors)"
        else:
            try:
                g = json.loads(Path(prov, "game.json").read_text())["game"]; h = hard_of(g); src = "game.json seat_stats (backfill)"
            except (OSError, ValueError, KeyError):
                h = None
        if h is None:
            continue
        fixes.append(dict(r, supersedes=k, hard_errors=h, hard_errors_source=src, correction_utc=now(),
                          correction_reason="v3 provenance gate: hard_errors attested from the source receipt"))
    with GAMES.open("a") as f:
        for r in fixes:
            f.write(json.dumps(r) + "\n")
    print("backfill: superseding records", len(fixes))


def receipt_check(gd, rec, archives, pair=None):
    """LOCKED-SPECS R3 P0 #1: bind the bytes actually played. With RUN-RECEIPT.json (farm_game.sbatch since 00:10 09-30): pre == post archive
    digests and the game.json digest must match; its digests become the record identity. Without one (older games): each archive's mtime
    must predate the game start, else the path hash at ingest cannot stand for the played bytes. Returns (issues, source, receipt)."""
    gd = Path(gd); gj = gd / "game.json"; rcp = gd / "RUN-RECEIPT.json"
    if rcp.is_file():
        try:
            rc = json.loads(rcp.read_text())
        except (OSError, ValueError):
            return ["run_receipt_unreadable"], "run-receipt", {}
        if not isinstance(rc, dict):
            return ["run_receipt_unreadable"], "run-receipt", {}
        iss = []
        if PG.normalize_engine(rc.get("engine")) != ENGINE:
            iss.append("run_receipt_engine_mismatch_or_missing")
        if rc.get("archives") != list(archives) or rec.get("archives") != list(archives):
            iss.append("run_receipt_archive_paths_mismatch")
        if pair is not None:
            try:
                if (rec.get("labels") != [pair[0], pair[2]] or
                        type(rec.get("world_seed")) is not int or rec["world_seed"] != int(pair[4]) or
                        type(rec.get("episode_id")) is not int or rec["episode_id"] != int(pair[5])):
                    iss.append("game_identity_differs_from_pairs_row")
            except (ValueError, TypeError, IndexError, KeyError):
                iss.append("malformed_pairs_row_identity")
        if utc_ts(rc.get("started_utc")) is None or utc_ts(rc.get("finished_utc")) is None:
            iss.append("run_receipt_timestamps_missing_or_invalid")
        elif utc_ts(rc["finished_utc"]) < utc_ts(rc["started_utc"]):
            iss.append("run_receipt_finish_before_start")
        pre, post = rc.get("archive_sha256_pre"), rc.get("archive_sha256_post")
        if not isinstance(pre, list) or not isinstance(post, list) or len(pre) != 2 or len(post) != 2:
            iss.append("run_receipt_archive_cardinality_not_two")                     # PRO VERIFY-FIX-GATES #2
        elif pre != post or not all(isinstance(x, str) and re.fullmatch("[0-9a-f]{64}", x) for x in pre):
            iss.append("archive_changed_during_game_or_bad_digest")
        if rc.get("game_json_sha256") != hashlib.sha256(gj.read_bytes()).hexdigest():
            iss.append("game_json_changed_after_receipt")
        return iss, "run-receipt", rc
    iss = []
    start = gj.stat().st_mtime - float(rec.get("wall_seconds") or 0)
    for ap in archives:
        try:
            if os.stat(ap).st_mtime > start:
                iss.append("archive_modified_after_game_start")
        except OSError:
            iss.append("archive_missing_at_ingest")
    return iss, "path-at-ingest", {}


def ingest(pats=None):
    """Serialized (P0 11:3x): one ingest at a time across the panel loop and the dedicated ingest unit. pats: only these globs (fast lane)."""
    with open(R_DIR / ".ingest.lock", "w") as lk_:
        fcntl.flock(lk_, fcntl.LOCK_EX)
        return _ingest(pats)


def _ingest(pats=None):
    db = league_db()
    by_sha = {s["archive_sha256"]: s["name"] for s in db["subs"] if s.get("archive_sha256")}
    cache = json.loads(SHA_CACHE.read_text()) if SHA_CACHE.exists() else {}
    keys = existing_keys()
    done_prov = set()                                   # P0 11:3x: skip game dirs already ingested clean (no NFS re-read of every old game)
    if GAMES.exists():
        with GAMES.open() as fh:
            for line in fh:
                try:
                    r_ = json.loads(line)
                    if r_.get("clean") and r_.get("provenance"):
                        done_prov.add(r_["provenance"])
                except ValueError:
                    pass
    def ingest_fingerprint(r):
        # Different provenance/corrections must survive until gate eligibility and dedup.
        return json.dumps({k: v for k, v in r.items() if k != "ingested_utc"}, sort_keys=True)
    seen_records = set()
    if GAMES.exists():
        with GAMES.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if isinstance(r, dict):
                        seen_records.add(ingest_fingerprint(r))
                except ValueError:
                    continue
    new = []

    def name_for(label, arch):
        sha, nbytes = sha_of(arch, cache)
        name = by_sha.get(sha) or Path(arch).name.replace(".tar.gz", "").replace(".tgz", "")
        return name, sha, nbytes

    def add_farm(pairs_file, outdir, source, row_filter=None):
        lines = [l.split("\t") for l in Path(pairs_file).read_text().splitlines()[1:] if l.strip()]
        for i, p in enumerate(lines):
            if len(p) < 6:
                continue
            gd = Path(outdir, str(i))
            if str(gd) in done_prov:
                continue
            gj = gd / "game.json"
            if not gj.is_file():
                continue
            try:
                rec = json.loads(gj.read_text()); g = rec["game"]
            except (OSError, ValueError, KeyError):
                continue
            n0, sha0, b0 = name_for(p[0], p[1]); n1, sha1, b1 = name_for(p[2], p[3])
            integ, src_, rc_ = receipt_check(gd, rec, (p[1], p[3]), pair=p)
            if src_ == "run-receipt" and not any(x.startswith("run_receipt_archive_cardinality") for x in integ):
                pre = rc_.get("archive_sha256_pre") or [None, None]; ab_ = rc_.get("archive_bytes") or [b0, b1]
                ab_ = ab_ if isinstance(ab_, list) and len(ab_) == 2 else [b0, b1]
                n0, sha0, b0 = by_sha.get(pre[0]) or n0, pre[0], ab_[0]
                n1, sha1, b1 = by_sha.get(pre[1]) or n1, pre[1], ab_[1]
            seed = int(rec.get("world_seed") or p[4])
            key = f"{seed}|{n0}|{n1}"
            crn = bool(rec.get("crn_rng")) and bool((rec.get("runtime") or {}).get("crn_rng", True))
            rt = rec.get("runtime") or {}
            row = {"key": key, "world_seed": seed, "seat0": n0, "seat1": n1, "archive_sha256": [sha0, sha1],
                        "archive_bytes": [b0, b1], "engine": ENGINE + " (farm_game.sbatch)", "runner_sha256": rec.get("ladder_sha256"),
                        "hard_errors": hard_of(g), "hard_errors_source": "game.json seat_stats",
                        "actions_recorded": rec.get("actions_recorded"), "runner_kind": rec.get("runner") or "native",
                        "crn_rng": crn, "host": rt.get("host"), "python": rt.get("python"), "torch": rt.get("torch"),
                        "finished_utc": rc_.get("finished_utc") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(gj.stat().st_mtime)),
                        "cash": g.get("cash"), "status": g.get("status"), "fallbacks": g.get("fallbacks"),
                        "replay_verified": bool(rec.get("replay_verified")), "clean": clean_of(g, rec) and crn and not integ,
                        "archive_sha_source": src_, "integrity_issues": integ, "runner_tree": {k: rc_.get(k) for k in ("runner_sha256", "engine_tree_sha256")},
                        "started_utc": rc_.get("started_utc") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(gj.stat().st_mtime - float(rec.get("wall_seconds") or 0))),
                        "replay_path": str(gd / "replay.json.gz"), "source": source, "provenance": str(gd), "ingested_utc": now()}
            fp = ingest_fingerprint(row)
            if fp not in seen_records:
                new.append(row)
                seen_records.add(fp)
            keys.add(key)

    for pat in (pats or PANEL_DIRS):
        for d in sorted(glob.glob(pat)):
            pf = Path(d, "pairs.tsv")
            if pf.is_file():
                src = ("import:finalpanel" if "finalpanel" in d else "import:codex-panel" if "colab-eval" in d else
                       "import:extworker" if "extworker" in d else "panel-league" if "/records/waves/" in d else
                       f"finalist-{Path(d).name}" if "/records/finalists/" in d else "dagger64" if "/records/dagger64/" in d else "n-screen" if "/records/nscreen/" in d else "regular" if "/records/regular/" in d else "gpu-served" if "/records/gpu/" in d else "gate-run" if "/records/gaterun/" in d else "ladder" if "/records/ladder/" in d else "nflow-panel" if "/records/nflow/" in d else "colab-cpu" if "/records/colab/" in d else "t4w-spark" if "/records/t4wrun/" in d else "slurm-lane" if "/records/slurmlane/" in d else "gpu-served" if "/records/colabgpu/" in d else "sp-rand" if "sp-rand" in Path(d).name else "sp-best" if "sp-best" in Path(d).name else "sp-pool" if "sp-pool" in Path(d).name else "sp-nfbe" if ("/records/gpu-sp/" in d or "/slurmlane/sp-nfbe" in d or "sp-nfbe" in Path(d).name) else "import:farm")
                add_farm(pf, d, src)
    for pf, od, src in ([] if pats else MIXED):
        if Path(pf).is_file():
            add_farm(pf, od, src)
    names = {s["id"]: s["name"] for s in db["subs"]}
    shas = {s["id"]: (s["archive_sha256"], s["archive_bytes"]) for s in db["subs"]}
    for e in db["eps"]:
        if str(e.get("source") or "").startswith("import:"):
            continue                                          # prototype/extworker imports: taken from their game dirs above
        rt = json.loads(e["runtime"] or "{}")
        n0, n1 = names.get(e["seat0"]), names.get(e["seat1"])
        key = f"{e['world_seed']}|{n0}|{n1}"
        if n0 is None or n1 is None or key in keys:
            continue
        crn = bool(rt.get("crn_rng")) and ENGINE in str(rt.get("engine", ""))
        new.append({"key": key, "world_seed": e["world_seed"], "seat0": n0, "seat1": n1,
                    "archive_sha256": [shas[e["seat0"]][0], shas[e["seat1"]][0]], "archive_bytes": [shas[e["seat0"]][1], shas[e["seat1"]][1]],
                    "engine": rt.get("engine"), "runner": rt.get("runner"), "crn_rng": crn,
                    "hard_errors": [0, 0] if e["rated"] else None, "hard_errors_source": "league rated=1 (worker rejects hard errors)",
                    "host": (rt.get("external") or {}).get("host") or "ct118-league-worker", "python": rt.get("python"),
                    "started_utc": e.get("started_utc"), "finished_utc": e.get("finished_utc"), "cash": [e["cash0"], e["cash1"]],
                    "status": [e["status0"], e["status1"]], "fallbacks": [e["fallbacks0"], e["fallbacks1"]],
                    "replay_verified": bool(e.get("replay_verified")), "clean": bool(e["rated"]) and crn,
                    "replay_path": e.get("replay_path"), "source": e.get("source") or "league", "provenance": f"league episode {e['id']}",
                    "ingested_utc": now()})
        keys.add(key)
    SHA_CACHE.write_text(json.dumps(cache))
    with GAMES.open("a") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    print(f"ingest: +{len(new)} games ({sum(r['clean'] for r in new)} clean)")
    return db


# ------------------------------------------------------------------ reducer (one for board, PANELS.md, export)
ACTIONS = R_DIR / "ACTIONS.json"
CONFLICTS = R_DIR / "CONFLICTS.jsonl"


def execution_identity(r):
    """Keep the simulation hash AND the receipt entrypoint/engine tree; neither replaces the other."""
    tree = r.get("runner_tree") if isinstance(r.get("runner_tree"), dict) else {}
    kind = "gpu-served" if (r.get("runner_kind") not in (None, "native") or r.get("source") == "gpu-served") else "native"
    return (kind, r.get("runner_sha256") or tree.get("runner_sha256"), PG.normalize_engine(r.get("engine")),
            tree.get("runner_sha256"), tree.get("engine_tree_sha256"))


def outcome_id(r):
    """What a duplicate must agree on to be the same game: seats, per-seat archive SHAs, world, final cash."""
    return json.dumps([r.get("seat0"), r.get("seat1"), r.get("archive_sha256"), r.get("world_seed"), r.get("cash"), r.get("actions_recorded"),
                       r.get("hard_errors"), r.get("fallbacks"), r.get("status"), r.get("replay_verified"), r.get("crn_rng"), execution_identity(r)], sort_keys=True)


def dedup(records):
    """ASTRA LOCKED-SPECS #3: order-independent. Per key: an explicit 'supersedes' record replaces the key it names (two conflicting
    supersessions = conflict); otherwise identical outcomes collapse to one (the canonically smallest record) and conflicting outcomes
    drop the key and are reported. Returns (kept records, conflicts)."""
    base, sup = defaultdict(list), defaultdict(list)
    for r in records:
        (sup[r["supersedes"]] if r.get("supersedes") else base[r["key"]]).append(r)
    kept, conflicts = [], []
    for k in sorted(set(base) | set(sup)):
        group = sup[k] or base[k]
        outs = sorted({outcome_id(r) for r in group})
        if len(outs) > 1:
            conflicts.append({"key": k, "n": len(group), "outcomes": outs, "sources": sorted({str(r.get("source")) for r in group})})
            continue
        kept.append(min(group, key=lambda r: json.dumps(r, sort_keys=True)))
    return kept, conflicts


def load_gate_games():
    """Unreduced receipts for frozen gates: eligibility MUST precede deduplication."""
    raw = []
    if GAMES.exists():
        with GAMES.open() as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        raw.append(row)
                except ValueError:
                    continue
    try:
        acts = json.loads(ACTIONS.read_text())
    except (OSError, ValueError):
        acts = {}
    for r in raw:
        if "actions_recorded" not in r and r.get("key") in acts:
            r["actions_recorded"] = acts[r["key"]]["actions"]
    return raw


def load_games():
    raw = load_gate_games()
    ok, rejected, conflicts = admit(raw)
    load_games.conflicts = conflicts; load_games.rejected = rejected
    try:
        CONFLICTS.write_text("".join(json.dumps(c) + "\n" for c in conflicts))
    except OSError:
        pass
    return ok


def admit(raw):
    """Read boundary + dedup, order-independent. Records failing record_issues never compete in dedup, so a clean rerun supersedes an
    unclean/failed record of the same key; conflicts are only between admissible records. Returns (clean games, rejected, conflicts)."""
    good = [r for r in raw if not record_issues(r)]
    kept, conflicts = dedup(good)
    taken = {r["key"] for r in good}
    rejected, seen = [], set()
    for r in sorted((r for r in raw if record_issues(r)), key=lambda r: json.dumps(r, sort_keys=True)):
        if r.get("key") in seen:
            continue
        seen.add(r.get("key"))
        rejected.append({"key": r.get("key"), "source": r.get("source"), "issues": record_issues(r),
                         "superseded_by_clean_rerun": r.get("key") in taken})
    return kept, rejected, conflicts


def pair_table(games):
    """(A, B) sorted -> {seed: {order(bool A is seat0): margin A-B}}. Order-independent (ASTRA LOCKED-SPECS #3): a cell with two different
    margins, or a world whose two orders carry different archive SHAs for the same name, is dropped and listed in pair_table.conflicts."""
    cells = defaultdict(set)
    for r in games:
        a, b = r["seat0"], r["seat1"]
        if a == b:
            continue
        A, B = sorted((a, b))
        cs = r.get("cash") if isinstance(r.get("cash"), list) and len(r.get("cash")) == 2 else [0.0, 0.0]
        c0, c1 = [float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else 0.0 for x in cs]
        m0 = c0 - c1
        fs = forfeit_seat(r)
        if fs == 0:
            m0 = min(m0, -1.0)                                     # seat 0 forfeited (TIMEOUT/ERROR): a loss, never dropped
        elif fs == 1:
            m0 = max(m0, 1.0)
        elif fs == "both":
            m0 = 0.0
        m = m0 if a == A else -m0
        sh = r.get("archive_sha256") or [None, None]
        ids = (sh[0], sh[1]) if a == A else (sh[1], sh[0])
        cells[(A, B, int(r["world_seed"]), a == A)].add((m, ids))
    P = defaultdict(lambda: defaultdict(dict)); conf = []
    for (A, B, s, o), v in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        if len(v) > 1:
            conf.append({"pair": [A, B], "seed": s, "A_seat0": o, "values": sorted(map(str, v))}); continue
        P[(A, B)][s][o] = next(iter(v))
    for (A, B), W in P.items():
        for s in list(W):
            ids = {x[1] for x in W[s].values()}
            if len(ids) > 1:
                conf.append({"pair": [A, B], "seed": s, "mixed_identities": sorted(map(str, ids))}); del W[s]; continue
            W[s] = {o: x[0] for o, x in W[s].items()}
    pair_table.conflicts = conf
    return P


def pts(m):
    return 1.0 if m > 0 else 0.5 if m == 0 else 0.0


def reduce_pair(worlds, rng=None, draws=2000):
    rng = rng or np.random.default_rng(12345)
    W = [(sum(pts(m) for m in w.values()), len(w), w) for w in worlds.values()]
    n = sum(k for _, k, _ in W); wins = sum(1 for _, _, w in W for m in w.values() if m > 0)
    ties = sum(1 for _, _, w in W for m in w.values() if m == 0); losses = n - wins - ties
    p = (wins + 0.5 * ties) / n if n else float("nan")
    pw = np.array([s for s, _, _ in W]); kw = np.array([k for _, k, _ in W])
    if len(W) >= 2:
        idx = rng.integers(0, len(W), size=(draws, len(W)))
        boot = pw[idx].sum(1) / kw[idx].sum(1)
        lb, ub = np.percentile(boot, [2.5, 97.5])
    else:
        lb, ub = 0.0, 1.0
    wp = pw / np.maximum(kw, 1)                                   # world-level outcome in [0, 1]
    beta = rng.beta(0.5 + wp.sum(), 0.5 + len(W) - wp.sum(), 20000)
    blb, bub = np.percentile(beta, [2.5, 97.5])
    small = len(W) < 8
    if small or len(set(np.round(wp, 6))) <= 1:          # few worlds or a degenerate (all-identical) bootstrap: Jeffreys floor
        lb, ub = min(lb, blb), max(ub, bub)
    boot = [float(lb), float(ub)]                         # DESCRIPTIVE only (PRO-PANEL-LEAGUE-STATS F2: not calibrated)
    cvals = [sum(pts(m) for m in w.values()) / 2 for _, _, w in W if len(w) == 2]   # world-averaged points, complete worlds
    if cvals:
        eps = math.sqrt(math.log(1 / 0.025) / (2 * len(cvals)))
        hoeff = [max(0.0, float(np.mean(cvals)) - eps), min(1.0, float(np.mean(cvals)) + eps)]
        lb, ub = PG.kl_interval(cvals, 0.05)             # KL/Chernoff >= Hoeffding (Pinsker): the tighter valid bound decides
    else:
        hoeff = [0.0, 1.0]; lb, ub = 0.0, 1.0
    z = 1.96
    wil = ((p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n),
           (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)) if n else (0, 1)
    comp = [w for _, _, w in W if len(w) == 2]
    wm = [(w[True] + w[False]) / 2 for w in comp]
    nt = [x for x in wm if x != 0]; kpos = sum(1 for x in nt if x > 0)
    sign = sum(math.comb(len(nt), j) for j in range(kpos, len(nt) + 1)) / 2 ** len(nt) if nt else float("nan")
    from scipy.stats import t as _t_dist
    tcrit = float(_t_dist.ppf(0.975, len(wm) - 1)) if len(wm) >= 2 else float("nan")   # exact two-sided 95% Student t (MARGIN_CI)
    mean = float(np.mean(wm)) if wm else float("nan")
    h = tcrit * float(np.std(wm, ddof=1)) / math.sqrt(len(wm)) if len(wm) >= 2 else float("nan")
    seat0 = [w[True] for _, _, w in W if True in w]; seat1 = [w[False] for _, _, w in W if False in w]
    decided = "A>B" if lb > 0.5 else "B>A" if ub < 0.5 else "undecided"
    return {"lb": float(lb), "ub": float(ub), "p": p, "wilson": [float(wil[0]), float(wil[1])], "p_gt_half": float((beta > 0.5).mean()),
            "W": wins, "L": losses, "T": ties, "n": n, "worlds": len(W), "worlds_complete": len(comp), "world_wins": kpos,
            "worlds_nontied": len(nt), "sign_p": sign, "margin": mean, "margin_ci": [mean - h, mean + h],
            "margin_median": float(np.median(wm)) if wm else float("nan"), "margin_A_seat0": float(np.mean(seat0)) if seat0 else None,
            "margin_A_seat1": float(np.mean(seat1)) if seat1 else None, "decided": decided, "small": small, "boot": boot,
            "cp": clopper_pearson(kpos, len(nt)), "world_mean": float(wp.mean()) if len(W) else float("nan"), "hoeff": hoeff,
            **sweep_stats(W)}


MARGIN_CI = {"margin_ci_tail": "two-sided", "margin_ci_level": 0.95,
             "margin_ci_critical_value": "Student t quantile 0.975 with df = complete worlds - 1 (scipy.stats.t.ppf)",
             "margin_unit": "final-cash margin (variant minus parent) in $ per seat-game, averaged over the two seat orders within a world, then over worlds",
             "noninferiority_margin_per_world": -1500}
MARGIN_CI_CUTOFF = "2026-09-30T05:20:00Z"     # manifests frozen at/after this must carry MARGIN_CI verbatim (ASTRA CHECK-GATE-MATH, 01:15 EDT)
GATE_RULE = ("Declared decision rule for every binary A-vs-B gate (LAND64, surgery, freshen, combo pick-confirm; coordinator 22:10 09-29, "
             "locked-specs review): paired-world sign test on sweeps (a sweep = both seat orders won outright, a sweep loss = both lost, "
             "any other complete world = split = tie). Only complete, pre-declared banks decide. Any individually tied game -> HOLD (the sweep "
             "endpoint is not reinterpreted). ADOPT a change when the variant's one-sided sweep-sign p <= .05 AND its game-point share is not "
             "losing (>= .5). NON-INFERIORITY (freshen, surgery+adapt): the parent must NOT beat the variant at one-sided sweep-sign p <= .10, "
             "the variant's mean-margin 95% lower bound must be > -$1,500/world, and at least 32 complete worlds x 2 seats.")


def gate_verdict(st, kind="superiority", need_worlds=None):
    """Apply GATE_RULE to a variant-oriented reduce_pair row of a complete bank (need_worlds = the bank size)."""
    sw, sl = st["sweeps"], st["sweep_losses"]; n = sw + sl
    p_parent = sum(math.comb(n, j) for j in range(sl, n + 1)) / 2 ** n if n else 1.0
    if st.get("tie_games", 0):                                        # PRO VERIFY-FIX-GATES #4: an observed tie is terminal even while incomplete
        return "HOLD", f"{st['tie_games']} individually tied games in {st.get('tie_worlds')} worlds: sweep endpoint undefined, not reinterpreted", p_parent
    if need_worlds is not None and st["worlds_complete"] < need_worlds:
        return "PENDING", f"{st['worlds_complete']}/{need_worlds} complete worlds (full horizon required)", p_parent
    if kind == "noninferiority":
        mc = st.get("margin_ci") or [float("nan"), float("nan")]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [st.get("margin", float("nan")), *mc]):
            return "HOLD", "margin evidence missing or non-finite: non-inferiority cannot be shown (fails closed)", p_parent
        ok = st["worlds_complete"] >= 32 and p_parent > 0.10 and st["margin_ci"][0] > -1500
        c1, c2, c3 = st["worlds_complete"] >= 32, p_parent > 0.10, st["margin_ci"][0] > -1500
        pf = lambda c: "pass" if c else "FAIL"
        why = (f"worlds {st['worlds_complete']} (need >= 32: {pf(c1)}), parent-beats-variant sweep-sign p {p_parent:.3f} (need > .10: {pf(c2)}), "
               f"margin two-sided t95 LB {st['margin_ci'][0]:+,.0f}/world (need > -1,500: {pf(c3)})")
        return ("NON-INFERIOR" if ok else "NOT SHOWN NON-INFERIOR"), why, p_parent
    ok = st["sweep_sign_p"] <= 0.05 and st["p"] >= 0.5
    return (("ADOPT" if ok else "DO NOT ADOPT"),
            f"sweeps {sw} vs sweep losses {sl}, one-sided sweep-sign p {st['sweep_sign_p']:.4f} (need <= .05: {'pass' if st['sweep_sign_p'] <= 0.05 else 'FAIL'}), "
            f"game-point share {st['p']:.3f} (need >= .5: {'pass' if st['p'] >= 0.5 else 'FAIL'})", p_parent)


def utc_ts(s):
    """Strict ISO-8601 UTC ('YYYY-MM-DDTHH:MM:SSZ' or with +00:00) -> datetime, else None."""
    from datetime import datetime, timezone
    if not isinstance(s, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(Z|\+00:00)", s):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def manifest_issues(M):
    """Mandatory frozen-gate invariants: two distinct named identities with full SHA-256, a fixed bank of distinct int worlds,
    a UTC deadline, a declared kind; both seats and the full horizon are enforced by exact_gate itself."""
    bad = []
    try:
        for k in ("challenger", "incumbent"):
            x = M[k]
            if not isinstance(x.get("name"), str) or not x["name"]: bad.append(f"{k} name")
            if not isinstance(x.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", x["sha256"]): bad.append(f"{k} full sha256")
            if x.get("name", "").startswith(EXCLUDED): bad.append(f"{k} is cut lineage C")
        if M["challenger"].get("name") == M["incumbent"].get("name") or M["challenger"].get("sha256") == M["incumbent"].get("sha256"):
            bad.append("identities not distinct")
        w = M["worlds"]
        if not isinstance(w, list) or not w or any(type(s) is not int for s in w) or len(set(w)) != len(w): bad.append("fixed bank of distinct int worlds")
        if not isinstance(M.get("deadline_utc"), str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", M["deadline_utc"]): bad.append("deadline_utc")
        if M.get("kind") not in ("superiority", "noninferiority"): bad.append("kind")
        if M.get("schema") != "frozen-gate/v1": bad.append("schema")
        body = {k: v for k, v in M.items() if k != "manifest_sha256"}
        if M.get("manifest_sha256") != hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(): bad.append("manifest_sha256 (edited after freeze)")
        if M.get("one_look") is not True: bad.append("one_look")
        if "rule" in M and M["rule"] != GATE_RULE: bad.append("rule differs from the rule this code applies")
        exposure_ok = type(M.get("games_seen_at_freeze")) is int and M.get("games_seen_at_freeze") == 0     # PRO VERIFY-FIX-GATES #3: both, not either
        if "prior_looks" in M:
            exposure_ok = exposure_ok and type(M.get("prior_looks")) is int and M.get("prior_looks") == 0
        if not exposure_ok and not M.get("post_hoc"): bad.append("exposure: games seen at freeze or prior looks")
        if utc_ts(M.get("deadline_utc")) is None: bad.append("deadline_utc not strict ISO UTC")
        if utc_ts(M.get("frozen_at_utc")) is None: bad.append("frozen_at_utc (registration time) missing")
        elif utc_ts(M["frozen_at_utc"]) >= utc_ts(MARGIN_CI_CUTOFF) and any(M.get(k) != v for k, v in MARGIN_CI.items()):
            bad.append("margin CI tail / critical value / unit not frozen as data (MARGIN_CI)")
    except (KeyError, TypeError, AttributeError) as e:
        bad.append(f"malformed: {e}")
    return bad


GPU_PARITY = "/mnt/ai/scratch/farmNEAT/league/gpuserve/parity/spark1/PARITY.jsonl"
GPU_PARITY_AUDIT = "/mnt/ai/scratch/farmNEAT/league/gpuserve/audit/v2-parity-spark/PARITY.jsonl"
SERVED_MODEL_PY = {"60958b937dfa6a44": "B", "b60e3228cb6fc68b": "O"}
PARITY_LABEL_FAMILY = {"B31kG": "B", "OA1u65": "O", "OA1u79": "O"}
_mfam = {}


def model_family(n):
    """Architecture family of a deployed archive by the sha256 of its model.py (the code the Spark daemon serves), else None."""
    if n not in _mfam:
        f = Path("/mnt/ai/scratch/farmNEAT/pbc-deploy") / n / f"{n}.tar.gz"
        try:
            names = subprocess.run(["tar", "-tzf", str(f)], capture_output=True, text=True, timeout=300).stdout.split()
            m = next((x for x in names if x.endswith("/model.py") or x == "model.py"), None)
            h = hashlib.sha256(subprocess.run(["tar", "-xzf", str(f), "-O", m], capture_output=True, timeout=300).stdout).hexdigest() if m else ""
        except (OSError, subprocess.TimeoutExpired):
            h = ""
        _mfam[n] = next((fam for pre, fam in SERVED_MODEL_PY.items() if h.startswith(pre)), None)
    return _mfam[n]


def parity_coverage(paths=(GPU_PARITY, GPU_PARITY_AUDIT)):
    """RUNTIME-PARITY P1: which families the parity evidence actually covers (0 mismatches, identical cash, >= 719 steps, recognised
    labels, >= 8 games, no bad row). A success printed for another experiment covers nothing here."""
    fams, games, bad = set(), 0, 0
    for p_ in paths:
        try:
            rows = [json.loads(l) for l in Path(p_).read_text().splitlines() if l.strip()]
        except (OSError, ValueError):
            continue
        for r_ in rows:
            if "labels" not in r_:
                continue
            games += 1
            if r_.get("mismatches") != 0 or r_.get("same_cash") is not True or min(r_.get("steps") or [0]) < 719:
                bad += 1; continue
            labs = r_["labels"][0] if r_["labels"] and isinstance(r_["labels"][0], list) else r_["labels"]
            for lab in labs:
                if lab in PARITY_LABEL_FAMILY:
                    fams.add(PARITY_LABEL_FAMILY[lab])
    return (fams if games >= 8 and bad == 0 else set()), games, bad


FROZEN_FIELDS = {"schema", "kind", "challenger", "incumbent", "worlds", "deadline_utc", "rule", "games_seen_at_freeze", "one_look",
                 "frozen_at_utc", "manifest_sha256", "gpu_parity_evidence"} | set(MARGIN_CI)


def freeze_gate(path, challenger, incumbent, worlds, kind, deadline_utc, **extra):
    """Write a frozen gate manifest once (never overwritten): identities are the sha256/bytes of the deployed archives right now. Every
    freeze field (time, SHAs, worlds, rule, exposure, margin-CI spec) is set here, never taken from the caller (EASY-CHECK-09 (1)); an
    invalid manifest is refused and nothing is written (EASY-CHECK-09 (3))."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    forged = sorted(set(extra) & FROZEN_FIELDS)
    if forged:
        raise ValueError(f"freeze fields cannot be supplied by the caller: {forged}")
    sys.path.insert(0, str(R_DIR / "gates")); from verify_strict import verify_strict          # coordinator 07:1x: HUB-DONE certifies nothing
    for n in (challenger, incumbent):
        if not n.startswith(("tape-", "pub-")):
            ok_, why_ = verify_strict(n)
            if not ok_:
                raise ValueError(f"refused to freeze: {n} fails the strict VERIFY check ({why_})")
    ident = {}
    for n in (challenger, incumbent):
        f = Path("/mnt/ai/scratch/farmNEAT/pbc-deploy") / n / f"{n}.tar.gz"
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        ident[n] = {"name": n, "sha256": h.hexdigest(), "bytes": f.stat().st_size, "path": str(f)}
    seen = sum(1 for r in load_gate_games() if {r["seat0"], r["seat1"]} == {challenger, incumbent} and r["world_seed"] in set(worlds))
    if extra.get("allow_gpu_served"):                              # coordinator 23:12: only at freeze, before any game, citing the parity evidence
        if seen or not Path(GPU_PARITY).is_file():
            raise ValueError("allow_gpu_served needs a gate frozen before any game and the spark1 PARITY.jsonl evidence")
        fams, ng, nb = parity_coverage()
        need = {model_family(challenger), model_family(incumbent)}
        if None in need or not need <= fams:
            raise ValueError(f"allow_gpu_served: parity evidence covers {sorted(fams)} ({ng} games, {nb} bad) but the gate needs {sorted(map(str, need))}")
        extra["gpu_parity_families"] = sorted(fams)
        extra["gpu_parity_evidence"] = f"{GPU_PARITY} sha256 {hashlib.sha256(Path(GPU_PARITY).read_bytes()).hexdigest()}"   # computed here
    par_ev = extra.pop("gpu_parity_evidence", None)
    M = {**extra, **MARGIN_CI, "schema": "frozen-gate/v1", "kind": kind, "challenger": ident[challenger], "incumbent": ident[incumbent],
         "worlds": list(worlds), "deadline_utc": deadline_utc, "rule": GATE_RULE, "games_seen_at_freeze": seen, "one_look": True,
         "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if par_ev:
        M["gpu_parity_evidence"] = par_ev
    M["manifest_sha256"] = hashlib.sha256(json.dumps(M, sort_keys=True).encode()).hexdigest()
    bad = manifest_issues(M)
    if bad:
        raise ValueError(f"refusing to freeze an invalid manifest (nothing written): {bad}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(M, indent=1)); tmp.replace(path); os.chmod(path, 0o444)
    return M


def runtime_of(r):
    """Execution runtime of one game: gpu-served (Spark GB10 daemon), aarch64 (Spark CPU, system python 3.12 / torch 2.10) or x86 (venv
    python 3.11 on cpu-hub / laptop / desktop / 4090b); None if unknown."""
    if r.get("runner_kind") not in (None, "native") or r.get("source") == "gpu-served":
        return "gpu-served"
    h = str(r.get("host") or "")
    py = str(r.get("python") or "")
    if h.startswith("spark") or py.startswith("3.12"):
        return "aarch64"
    if h or py.startswith("3.11"):
        return "x86"
    return None


def started_of(r):
    """Game start time: the record's started_utc, else the game dir's game.json mtime minus wall_seconds (farm records)."""
    s = utc_ts(r.get("started_utc"))
    if s is not None:
        return s
    rp = str(r.get("replay_path") or "")
    gj = Path(rp).parent / "game.json" if rp and not rp.startswith("zip:") else None
    try:
        from datetime import datetime, timezone
        g = json.loads(gj.read_text())
        return datetime.fromtimestamp(gj.stat().st_mtime - float(g.get("wall_seconds") or 0), timezone.utc)
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def validate_gate_runtime(sel, M, rt):
    """Fail closed on unknown/mixed paired runtimes and unavailable frozen GPU evidence.

    Only the HASH-BOUND primary parity file counts. The legacy extra path is not
    hash-bound, so it cannot supply coverage. This does not certify a new GPU
    daemon/configuration; an exact executed-runner contract is still required.
    """
    if any(None in kinds or len(kinds) != 1 for kinds in rt.values()):
        return "unknown or mixed runtime within a predeclared world (do not shrink the bank)"
    for r in sel:
        h = r.get("runner_sha256") or (r.get("runner_tree") or {}).get("runner_sha256")
        if not isinstance(h, str) or not re.fullmatch("[0-9a-f]{64}", h):
            return "counted game lacks explicit runner identity"
    if not any(runtime_of(r) == "gpu-served" for r in sel):
        return None
    ev = re.fullmatch(r"(.+) sha256 ([0-9a-f]{64})", str(M.get("gpu_parity_evidence", "")))
    if not ev:
        return "GPU parity receipt is not hash-bound"
    try:
        raw = Path(ev.group(1)).read_bytes()
    except OSError:
        return "frozen GPU parity receipt unavailable (not an inference of parity)"
    if hashlib.sha256(raw).hexdigest() != ev.group(2):
        return "frozen GPU parity receipt bytes changed"
    names = [M[k]["name"] for k in ("challenger", "incumbent")]
    if any(n.startswith(("n1-", "nflow", "n-flow", "n2-")) for n in names):
        return "N configuration not covered by the declared B/O parity scope"
    fams, _, bad = parity_coverage(paths=(ev.group(1),))
    need = {model_family(n) for n in names}
    if bad or None in need or not need <= fams:
        return "hash-bound primary parity receipt does not cover both model families"
    # zip-ops 04:2x: a GPU-permission string is never evidence. Every counted gpu-served game must carry its own execution receipt
    # (game.json gpu_serve: all seats served, none declined) whose served-code identity (server_sha256) was itself parity-tested:
    # identities are read from the game.json of the parity rows in the hash-bound file, plus the manifest's extra parity file only if it
    # was last modified before the gate froze.
    def parity_servers(path):
        """(server_sha256, family) pairs actually parity-tested: family from each row's labels (B31kG -> B, OA1u65/u79 -> O)."""
        out = set(); base = Path("/mnt/ai/scratch/farmNEAT/league/gpuserve")
        try:
            rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        except (OSError, ValueError):
            return out
        for r_ in rows:
            if "test" not in r_ or r_.get("mismatches") != 0 or r_.get("same_cash") is not True:
                continue
            try:
                s_ = (json.loads((base / r_["test"] / "game.json").read_text()).get("gpu_serve") or {}).get("server_sha256")
            except (OSError, ValueError):
                s_ = None
            if s_:
                labs = r_["labels"][0] if r_.get("labels") and isinstance(r_["labels"][0], list) else (r_.get("labels") or [])
                for lab in labs:
                    if lab in PARITY_LABEL_FAMILY:
                        out.add((s_, PARITY_LABEL_FAMILY[lab]))
        return out

    def gpu_receipt(r):
        if r.get("gpu_serve"):
            return r["gpu_serve"]
        rp = str(r.get("replay_path") or "")
        if not rp or rp.startswith("zip:"):
            return None
        try:
            return json.loads((Path(rp).parent / "game.json").read_text()).get("gpu_serve")
        except (OSError, ValueError):
            return None

    covered = parity_servers(ev.group(1))
    ex = M.get("gpu_parity_extra")
    try:
        if ex and Path(ex).stat().st_mtime <= utc_ts(M["frozen_at_utc"]).timestamp():
            covered |= parity_servers(ex)
    except (OSError, AttributeError, KeyError):
        pass
    for r in sel:
        if runtime_of(r) != "gpu-served":
            continue
        rc = gpu_receipt(r)
        if not rc:
            return "gpu-served game lacks its execution receipt (a permission string is not parity evidence)"
        if rc.get("declined") or not all(rc.get("served_seats") or [False]):
            return "gpu-served game has a declined or unserved seat"
        for seat_name in (r.get("seat0"), r.get("seat1")):           # PRO CHECK-0300-GATES #2: server AND model family must be parity-tested
            if (rc.get("server_sha256"), model_family(seat_name)) not in covered:
                return (f"gpu-served game ran served code {str(rc.get('server_sha256'))[:12]} for family {model_family(seat_name)} "
                        "not covered by parity-tested (server, family) pairs")
    return None

    base = Path("/mnt/ai/scratch/farmNEAT/league/gpuserve")
    for r_ in rows:
        if "test" not in r_ or r_.get("mismatches") != 0 or r_.get("same_cash") is not True:
            continue
        try:
            s_ = (json.loads((base / r_["test"] / "game.json").read_text()).get("gpu_serve") or {}).get("server_sha256")
        except (OSError, ValueError):
            s_ = None
        if s_:
            out.add(s_)
    return out


EXPIRE_GRACE_MIN = 30


def gate_expired(M, now=None):
    from datetime import datetime, timedelta, timezone
    d = utc_ts(M.get("deadline_utc"))
    return d is not None and (now or datetime.now(timezone.utc)) > d + timedelta(minutes=EXPIRE_GRACE_MIN)


def exact_gate(games, M):
    """ASTRA LOCKED-SPECS #2/#17: a frozen manifest (records/gates/*.json) decides only from clean records whose per-seat archive SHAs
    equal the manifest identities, on the manifest worlds, both orders, finished at or before the deadline."""
    bad = manifest_issues(M)
    if bad:
        return "INVALID MANIFEST", {"manifest_issues": bad, "complete": 0, "wrong_sha": 0, "late": 0, "conflicts": 0, "rejected": 0}, None
    a, b = M["challenger"], M["incumbent"]; sha = {a["name"]: a["sha256"], b["name"]: b["sha256"]}
    Wd = set(M["worlds"]); sel = []; info = {"wrong_sha": 0, "late": 0, "rejected": 0, "complete": 0, "conflicts": 0}
    for r in list(games):                                          # EASY-CHECK-09 (2): eligibility first, dedup only among eligible records
        if {r.get("seat0"), r.get("seat1")} != set(sha) or r.get("world_seed") not in Wd:
            continue
        if record_issues(r):                                       # mandatory runtime contract: 719, DONE/DONE, zero fallbacks/hard errors
            info["rejected"] += 1; continue
        st_ = started_of(r)
        if not M.get("post_hoc") and (st_ is None or st_ < utc_ts(M["frozen_at_utc"])):
            info["pre_freeze"] = info.get("pre_freeze", 0) + 1; continue       # a game that began before the gate was registered never counts
        if (r.get("runner_kind") not in (None, "native") or r.get("source") == "gpu-served") and not (
                M.get("allow_gpu_served") is True and M.get("games_seen_at_freeze") == 0 and "PARITY.jsonl" in str(M.get("gpu_parity_evidence", ""))):
            info["rejected"] += 1; continue                        # frozen gates stay CPU-native unless the manifest says parity is exact
        if r.get("archive_sha256") != [sha[r["seat0"]], sha[r["seat1"]]]:
            info["wrong_sha"] += 1; continue
        fin = utc_ts(r.get("finished_utc"))
        if fin is None or fin > utc_ts(M["deadline_utc"]):
            info["late"] += 1; continue
        sel.append(r)
    sel, cf_ = dedup(sel); info["dedup_conflicts"] = len(cf_)
    ids_ = {execution_identity(r) for r in sel}
    info["runner_identities"] = len(ids_)
    if len({i[0] for i in ids_}) != len(ids_):                      # one execution identity per lane (native; gpu-served only if allowed)
        return "HOLD", dict(info, why=f"{len(ids_)} runner/engine identities among counted records"), None
    if M.get("runner_sha256") and sel and any(i[1] != M["runner_sha256"] for i in ids_ if i[0] == "native"):
        return "HOLD", dict(info, why="runner identity differs from the manifest"), None
    Pe = pair_table(sel); info["conflicts"] = len(pair_table.conflicts)
    rt_ = defaultdict(set)                                          # GATE-MATH (coordinator 02:00): runtime per game, per world
    for r in sel:
        rt_[r["world_seed"]].add(runtime_of(r))
    runtime_issue = validate_gate_runtime(sel, M, rt_)
    if runtime_issue:
        return "HOLD", dict(info, why=runtime_issue), None
    A_, B_ = sorted(sha); w = Pe.get((A_, B_), {})
    ws = {s: w[s] for s in M["worlds"] if len(w.get(s, {})) == 2}
    info["complete"] = len(ws)
    if len(ws) < len(M["worlds"]) and gate_expired(M):             # PRO CHECK-LADDER-R4: incomplete after deadline + ingest grace -> terminal
        return "EXPIRED", dict(info, why=f"{len(ws)}/{len(M['worlds'])} worlds complete at deadline {M['deadline_utc']} (+{EXPIRE_GRACE_MIN} min ingest grace)"), None
    if not ws:
        return "PENDING", info, None
    if M["kind"] == "noninferiority" and any(forfeit_seat(r) is not None for r in sel):
        # Keep forfeits in the outcome record; never turn their sign encoding or
        # partial cash into a terminal-$ confidence interval. No post-hoc deletion.
        return "HOLD", dict(info, why="forfeit outcome has no declared terminal-cash estimand for non-inferiority",
                            forfeit_games=sum(forfeit_seat(r) is not None for r in sel)), None
    st = reduce_pair(ws if a["name"] == A_ else orient(ws))
    v, why, _ = gate_verdict(st, M.get("kind", "superiority"), need_worlds=len(M["worlds"]))
    info["why"] = why
    lab = {s: (next(iter(rt_[s])) if len(rt_[s]) == 1 else "mixed") for s in ws}
    info["world_runtimes"] = {k: sum(1 for x in lab.values() if x == k) for k in sorted(set(lab.values()), key=str)}
    same = {s: w for s, w in ws.items() if lab[s] != "mixed"}
    if same and len(same) < len(ws):                              # descriptive: worlds whose two seats ran on one runtime
        s2 = reduce_pair(same if a["name"] == A_ else orient(same))
        v2, why2, _ = gate_verdict(s2, M.get("kind", "superiority"))
        info["same_runtime_only"] = (f"{len(same)}/{len(ws)} same-runtime worlds (descriptive): {v2} | W-D-L {s2['W']}-{s2['T']}-{s2['L']}, "
                                     f"sweeps/splits/losses {s2['sweeps']}/{s2['splits']}/{s2['sweep_losses']}, points {s2['p']:.3f}, "
                                     f"margin {s2['margin']:+,.0f} [{s2['margin_ci'][0]:+,.0f}, {s2['margin_ci'][1]:+,.0f}] | {why2}")
    if M.get("post_hoc"):                                          # rule declared after the result was visible: never a decision
        return "POST-HOC (not a decision)", info, st
    return {"ADOPT": "PROMOTE", "DO NOT ADOPT": "NOT PROMOTED"}.get(v, v), info, st


def sweep_stats(W):
    """EZ-B31-CONFIRM: per complete world, sweep = both seat orders won, sweep loss = both lost, split = anything else (ties too).
    Exact one-sided sign test on competitive (sweep) worlds; splits are ties."""
    sw = sum(1 for s_, k, _ in W if k == 2 and s_ == 2.0)
    sl = sum(1 for s_, k, _ in W if k == 2 and s_ == 0.0)
    sp = sum(1 for _, k, w in W if k == 2 and all(m != 0 for m in w.values())        # PRO VERIFY-FIX-GATES #5: a tied world is not a split
             and any(m > 0 for m in w.values()) and any(m < 0 for m in w.values()))
    tg = sum(1 for _, k, w in W if k == 2 for m in w.values() if m == 0)          # individual tied games (world points 0.5)
    tw = sum(1 for _, k, w in W if k == 2 and any(m == 0 for m in w.values()))
    n = sw + sl
    p1 = sum(math.comb(n, j) for j in range(sw, n + 1)) / 2 ** n if n else float("nan")
    return {"sweeps": sw, "splits": sp, "sweep_losses": sl, "sweep_sign_p": p1, "tie_games": tg, "tie_worlds": tw}


def _binom_cdf(k, n, q):
    """P(X <= k), X ~ Bin(n, q), in log space (EASY-CHECK-09: the direct comb * q**j form overflows near n = 1,100)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if q <= 0.0:
        return 1.0
    if q >= 1.0:
        return 0.0
    lq, l1q, lnf = math.log(q), math.log1p(-q), math.lgamma(n + 1)
    terms = [lnf - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lq + (n - j) * l1q for j in range(k + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(x - m) for x in terms))


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided (alpha/2 per side) interval for the world-win rate among non-tied worlds."""
    if n == 0:
        return [0.0, 1.0]
    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid): hi = mid
            else: lo = mid
        return (lo + hi) / 2
    lo = 0.0 if k == 0 else solve(lambda q: 1 - _binom_cdf(k - 1, n, q) >= alpha / 2)
    hi = 1.0 if k == n else solve(lambda q: _binom_cdf(k, n, q) <= alpha / 2)
    return [lo, hi]


def orient(worlds):
    """A-view worlds -> B-view worlds (B's margins; B's seat0 is A's seat1). Every directional field is then recomputed."""
    return {seed: {(not k): -m for k, m in w.items()} for seed, w in worlds.items()}


def flip_pair_stats(s):
    """Orient an A-vs-B reducer row to B-vs-A, including diagnostics."""
    world_wins = s["worlds_nontied"] - s["world_wins"]
    sign_p = (
        sum(math.comb(s["worlds_nontied"], j) for j in range(world_wins, s["worlds_nontied"] + 1))
        / 2 ** s["worlds_nontied"]
        if s["worlds_nontied"]
        else float("nan")
    )
    return dict(
        s,
        lb=1 - s["ub"],
        ub=1 - s["lb"],
        p=1 - s["p"],
        wilson=[1 - s["wilson"][1], 1 - s["wilson"][0]],
        boot=[1 - s["boot"][1], 1 - s["boot"][0]] if s.get("boot") else None,
        cp=[1 - s["cp"][1], 1 - s["cp"][0]] if s.get("cp") else None,
        world_mean=1 - s["world_mean"] if s.get("world_mean") == s.get("world_mean") else s.get("world_mean"),
        p_gt_half=1 - s["p_gt_half"],
        W=s["L"],
        L=s["W"],
        margin=-s["margin"],
        margin_ci=[-s["margin_ci"][1], -s["margin_ci"][0]],
        margin_median=-s["margin_median"],
        margin_A_seat0=None if s["margin_A_seat1"] is None else -s["margin_A_seat1"],
        margin_A_seat1=None if s["margin_A_seat0"] is None else -s["margin_A_seat0"],
        decided={"A>B": "B>A", "B>A": "A>B"}.get(s["decided"], s["decided"]),
        world_wins=world_wins,
        sign_p=sign_p,
    )


def fmt(A, B, s, flag=""):
    mci = s["margin_ci"]
    bo = s.get("boot") or [float("nan")] * 2; cp = s.get("cp") or [float("nan")] * 2; ho = s.get("hoeff") or [float("nan")] * 2
    return (f"{A} vs {B}: p 95% KL (world-avg points, complete worlds) [{s['lb']:.2f}, {s['ub']:.2f}]"
            f" | Hoeffding [{ho[0]:.2f}, {ho[1]:.2f}] | p {s['p']:.2f}"
            f" | boot (descriptive) [{bo[0]:.2f}, {bo[1]:.2f}] | world-win CP [{cp[0]:.2f}, {cp[1]:.2f}]"
            f" | P(p>.5) {s['p_gt_half']:.2f} | {s['W']}-{s['L']}-{s['T']} | worlds {s['worlds_complete']}/{s['worlds']}"
            f" | sweeps/splits/sweep-losses {s.get('sweeps', 0)}/{s.get('splits', 0)}/{s.get('sweep_losses', 0)} sweep-sign p {s.get('sweep_sign_p', float('nan')):.3f}"
            f" | cash-positive worlds {s['world_wins']}/{s['worlds_nontied']} sign p {s['sign_p']:.3f}"
            f" | margin {s['margin']:+,.0f} [{mci[0]:+,.0f}, {mci[1]:+,.0f}] med {s['margin_median']:+,.0f}"
            f" | seat0 {s['margin_A_seat0'] if s['margin_A_seat0'] is None else round(s['margin_A_seat0'])}"
            f" seat1 {s['margin_A_seat1'] if s['margin_A_seat1'] is None else round(s['margin_A_seat1'])}"
            f" | {s['decided']}{' | small-sample' if s['small'] else ''}{flag}")


def bt(P):
    names = sorted({n for k in P for n in k})
    idx = {n: i for i, n in enumerate(names)}
    w = np.zeros((len(names), len(names)))
    for (A, B), worlds in P.items():
        for wd in worlds.values():
            pa = sum(pts(m) for m in wd.values()) / len(wd)
            w[idx[A], idx[B]] += pa; w[idx[B], idx[A]] += 1 - pa
    p = np.ones(len(names))
    for _ in range(500):
        num = w.sum(1) + 0.5
        den = np.array([1.0 / (p[i] + 1.0) + sum((w[i, j] + w[j, i]) / (p[i] + p[j]) for j in range(len(names)) if w[i, j] + w[j, i] > 0)
                        for i in range(len(names))])
        p = num / den; p /= np.exp(np.log(p).mean())
    r = 400 * np.log10(p)
    var = []
    for i in range(len(names)):
        info = p[i] / (p[i] + 1) ** 2 + sum((w[i, j] + w[j, i]) * p[i] * p[j] / (p[i] + p[j]) ** 2 for j in range(len(names)) if j != i)
        var.append((400 / math.log(10)) ** 2 / info)
    return {n: (float(r[idx[n]]), float(math.sqrt(var[idx[n]]))) for n in names}


def implied(ra, rb):
    d = ra[0] - rb[0]; s = math.sqrt(ra[1] ** 2 + rb[1] ** 2)
    f = lambda x: 1 / (1 + 10 ** (-x / 400))
    return f(d - 1.96 * s), f(d), f(d + 1.96 * s)


def best_analysis(P, stats, ratings, pool):
    """Compute provisional T and BEST without mixing incomparable evidence.

    A completed WORLDS-T16 panel is reduced from exactly those 16 reciprocal
    worlds. Until that panel is complete, every edge uses the same BT-implied
    provisional lower bound; a thin empirical edge cannot abruptly replace it.
    Cycles or non-convergence suppress BEST rather than selecting an arbitrary
    iteration.
    """
    R = list(R_NAMES)                                         # immutable required list; unavailable refs block BEST (qualified())
    cand = [n for n in pool if n not in R]
    W16 = tuple(worlds16())
    panel_cache = {}

    def panel_stat(e, o):
        A, B = sorted((e, o))
        key = (A, B)
        if key not in panel_cache:
            worlds = P.get(key, {})
            if all(set(worlds.get(seed, {})) == {False, True} for seed in W16):
                panel_cache[key] = reduce_pair({seed: worlds[seed] for seed in W16})
            else:
                panel_cache[key] = None
        return panel_cache[key]

    def lb(e, o):
        s = panel_stat(e, o)
        if s is not None:
            A, _ = sorted((e, o))
            return (s["lb"] if e == A else 1 - s["ub"]), False
        if e in ratings and o in ratings:
            return implied(ratings[e], ratings[o])[0], True
        return 0.0, True

    def complete(e, o):
        return panel_stat(e, o) is not None

    def score_against(n, opponents):
        values = [lb(n, o)[0] for o in opponents if o != n]
        return float(np.mean(values)) if values else 0.0

    # A comparable deterministic start: everyone is scored against the same
    # candidate/reference universe, not a self-selected set of played edges.
    universe = set(cand) | set(R)
    S = {n: score_against(n, universe) for n in cand}
    T = sorted(cand, key=lambda n: (-S[n], n))[:8]
    hist = [tuple(T)]
    seen = {frozenset(T): 0}
    fixed = False
    cycled = False
    for _ in range(100):
        opponents = set(T) | set(R)
        S = {n: score_against(n, opponents) for n in cand}
        T2 = sorted(cand, key=lambda n: (-S[n], n))[:8]
        if set(T2) == set(T):
            T = T2
            fixed = True
            break
        key = frozenset(T2)
        if key in seen:
            T = T2
            hist.append(tuple(T2))
            cycled = True
            break
        T = T2
        seen[key] = len(hist)
        hist.append(tuple(T))

    S = {n: score_against(n, set(T) | set(R)) for n in cand}   # recompute S on the returned roster (owner 17:53)
    cycle_sets = [list(h) for h in hist[seen.get(frozenset(T), 0):]] if cycled else ([list(h) for h in hist] if not fixed else [])
    rows = []
    for n in T:
        opps = (set(T) | set(R)) - {n}
        lbs = [lb(n, o) for o in opps]
        missing = sorted(o for o in opps if not complete(n, o))
        rows.append({"name": n, "S": float(np.mean([x for x, _ in lbs])) if lbs else 0.0,
                     "minLB": float(min(x for x, _ in lbs)) if lbs else 0.0,
                     "model_implied": sum(1 for _, m in lbs if m), "missing": missing,
                     "provenance": {"direct-t16": sum(1 for _, m in lbs if not m), "model-provisional": sum(1 for _, m in lbs if m)}})
    eligible = [r for r in rows if fixed and not r["missing"]]
    best = max(eligible, key=lambda r: (r["S"], r["minLB"])) if eligible else None
    min_lbs = {}
    final_opponents = set(T) | set(R)
    for n in cand:
        values = [lb(n, o)[0] for o in final_opponents if o != n]
        min_lbs[n] = float(min(values)) if values else 0.0
    return {"T": rows, "R": R, "best": best, "cycled": cycled, "fixed": fixed,
            "S": {k: float(v) for k, v in S.items()}, "minLB": min_lbs,
            "fixed_point_history": hist, "cycle_sets": cycle_sets}


def qualified(T, R, Pq, pool, reg_sha):
    """S_qualified per T member from the exact-world (W16) exact-artifact-SHA panel table only. Missing/unavailable edges
    contribute LB 0 (never imputed); panels_complete counts complete 16x2 panels of the 18 required."""
    W16 = worlds16()
    rows = []
    for n in T:
        opps = [o for o in list(T) + list(R) if o != n]
        lbs, miss, unav = [], [], []
        for o in opps:
            if o not in pool or o not in reg_sha:
                unav.append(o); lbs.append(0.0); continue
            A, B = sorted((n, o)); w = Pq.get((A, B), {})
            if all(set(w.get(x, {})) == {False, True} for x in W16):
                ws = {x: w[x] for x in W16}
                st = reduce_pair(ws if n == A else orient(ws)); lbs.append(st["lb"])
            else:
                miss.append(o); lbs.append(0.0)
        rows.append({"name": n, "S_qualified": float(np.mean(lbs)) if lbs else 0.0, "panels_complete": len(opps) - len(miss) - len(unav),
                     "required": len(opps), "missing": miss, "unavailable": unav, "minLB_qualified": float(min(lbs)) if lbs else 0.0})
    return rows


def board():
    games = load_games()
    P = pair_table(games)
    stats = {k: reduce_pair(v) for k, v in P.items()}
    ratings = bt(P) if P else {}
    try:
        db = league_db(); pool = sorted(s["name"] for s in db["subs"] if s["status"] in ("active", "provisional"))
    except Exception:  # noqa: BLE001
        pool = sorted(ratings)
    pool = [n for n in pool if (n in ratings or n in R_NAMES) and not n.startswith(EXCLUDED)]
    ba = best_analysis(P, stats, ratings, pool)            # DISCOVERY view (all clean history + provisional model); picks T
    W16 = worlds16()
    reg_sha = {s_["name"]: s_["archive_sha256"] for s_ in db["subs"]} if "db" in dir() else {}
    games_q = [r for r in games if int(r["world_seed"]) in set(W16)
               and r["archive_sha256"] == [reg_sha.get(r["seat0"]), reg_sha.get(r["seat1"])]]
    Pq = pair_table(games_q)
    qrows = qualified([r["name"] for r in ba["T"]], ba["R"], Pq, set(pool), reg_sha)
    qb = [r for r in qrows if ba.get("fixed") and not r["missing"] and not r["unavailable"]]
    ba["best"] = ({"name": max(qb, key=lambda r: (r["S_qualified"], r["minLB_qualified"], r["name"]))["name"]} if qb else None)
    if ba["best"]:
        qr = next(r for r in qrows if r["name"] == ba["best"]["name"])
        ba["best"].update(S=qr["S_qualified"], minLB=qr["minLB_qualified"])
    ba["qualified"] = qrows; ba["Pq"] = Pq
    rej = getattr(load_games, "rejected", [])
    REJECTED.write_text("".join(json.dumps(x) + "\n" for x in rej))
    DS = decision_set()
    def pl(a_, b_, seeds, label):
        A_, B_ = sorted((a_, b_)); w = P.get((A_, B_), {})
        ws = {x: w[x] for x in seeds if x in w}
        if not ws:
            return f"  - {a_} vs {b_} [{label}]: 0/{len(seeds)} worlds"
        st = reduce_pair(ws if a_ == A_ else orient(ws))
        ho = st.get("hoeff") or [float("nan")] * 2
        return (f"  - {a_} vs {b_} [{label}]: world-score p 95% KL [{st['lb']:.2f}, {st['ub']:.2f}] (Hoeffding [{ho[0]:.2f}, {ho[1]:.2f}]),"
                f" p {st['p']:.2f} | games W-D-L {st['W']}-{st['T']}-{st['L']} | sweeps/splits/sweep-losses {st['sweeps']}/{st['splits']}/{st['sweep_losses']}"
                f" (sweep-sign p {st['sweep_sign_p']:.3f}) | worlds {st['worlds_complete']}/{len(seeds)}"
                f" | cash margin/world {st['margin']:+,.0f} [{st['margin_ci'][0]:+,.0f}, {st['margin_ci'][1]:+,.0f}] (cash, not a win bound) | "
                + (f"{st['decided']} (complete bank)" if st['worlds_complete'] == len(seeds) and "descriptive" not in label else
                   "descriptive only" if "descriptive" in label else f"discovery only ({st['worlds_complete']}/{len(seeds)} worlds; no decision)"))
    F96 = fresh96()
    short = lambda n: (n or "none").replace("base1-", "").replace("-a1-ema", "").replace("oldarch", "oa").replace("-cd-ema", "")
    def cx(a_, b_):
        A_, B_ = sorted((a_, b_)); w = P.get((A_, B_), {})
        ws = {x: w[x] for x in W16 if x in w}
        if not ws:
            fp = {x: w[x] for x in range(32200001, 32200017) if x in w}
            if fp:
                st = reduce_pair(fp if a_ == A_ else orient(fp))
                return f"missing (FP16, not T16: {st['p']:.2f} {st['worlds_complete']}/16)"
            return "**missing**"
        st = reduce_pair(ws if a_ == A_ else orient(ws))
        return f"[{st['lb']:.2f},{st['ub']:.2f}] {st['p']:.2f} {st['worlds_complete']}/16 {st['decided']}"
    m12 = DS["m12"]
    live = [(a_, sl, n) for a_, sl, n in m12 if n]
    npairs = len(live) * (len(live) - 1) // 2
    def done(a_, b_):
        A_, B_ = sorted((a_, b_)); w = P.get((A_, B_), {})
        return all(len(w.get(x, {})) == 2 for x in W16)
    ndone = sum(1 for i_, (_, _, a_) in enumerate(live) for (_, _, b_) in live[i_ + 1:] if done(a_, b_))
    DB = ["# Decision set: MATRIX (owner 21:33 C cut; rows B, O = OA1, OA3, N x {parent = super-confident backup, control = current best, experiment})", "",
          f"Upload shortlist = parent (overall) **{DS['old_guaranteed']}** + best control overall (from the MATRIX + fresh48/96 finalist pairs;"
          " discovery results never feed BEST or the shortlist).", "",
          f"Pair lists: {DS.get('finalist_pairs_note', '')} Upload-pair plan on record: reliable **{(DS.get('upload_pair_plan') or {}).get('reliable')}**, "
          f"spike **{(DS.get('upload_pair_plan') or {}).get('spike')}** ({(DS.get('upload_pair_plan') or {}).get('source', '')}).", "",
          f"T16 pairs complete: {ndone}/{npairs} among the {len(live)} filled cells ({len(m12) * (len(m12) - 1) // 2} when all {len(m12)} exist). Cells: row vs column, "
          "p 95% KL [LB,UB], p, worlds complete/16, decided. FP16/history results are shown only as 'missing (FP16 ...)' and never count.", "",
          "| row \\ col | " + " | ".join(f"{a_}/{ROLE[sl]}: {short(n)}" for a_, sl, n in m12) + " |", "|---|" + "---|" * len(m12)]
    for a_, sl, r in m12:
        DB.append(f"| {a_}/{ROLE[sl]}: {short(r)} | " + " | ".join("—" if (not r or not c or c == r) else cx(r, c) for _, _, c in m12) + " |")
    if DS.get("labels"):
        DB += ["", "Labels:"] + [f"- **{k}**: {v}" for k, v in DS["labels"].items()]
    DB += [""] + [f"- {a_} parent (super-confident backup): {v.get('old_guaranteed')} [{v.get('old_guaranteed_status')}]; note: "
                  f"{v.get('old_guaranteed_source', '-')}" for a_, v in DS["architectures"].items()]
    # owner 18:39: final pair = one "most solid" + one "most likely to spike" (DESCRIPTIVE columns, not decision rules)
    SPIKE_REFS = ["tape-mmpq-115130538-lw", "pub-shepherds-ledger-wr1k-lw", "pub-ice-fire-wr1k-lw"]
    def t16_complete(a_, b_):
        A_, B_ = sorted((a_, b_)); w = Pq.get((A_, B_), {})
        if not all(set(w.get(x, {})) == {False, True} for x in W16):
            return None
        ws = {x: w[x] for x in W16}
        return reduce_pair(ws if a_ == A_ else orient(ws))
    names_q = sorted({n for k in Pq for n in k})
    DB += ["", "Solid / spike (DESCRIPTIVE, not decision rules; complete exact-SHA T16 panels only): solid = min KL LB over opponents; "
           "spike = max KL UB over opponents; win share (game points) vs tape-mmpq / shepherds-ledger / PUBice.", "",
           "| cell | entry | complete T16 panels | solid (min LB, opponent) | spike (max UB, opponent) | vs tape-mmpq | vs shepherds | vs PUBice |",
           "|---|---|---|---|---|---|---|---|"]
    for a_, sl, n in m12:
        if not n:
            continue
        pan = [(o, t16_complete(n, o)) for o in names_q if o != n]
        pan = [(o, st) for o, st in pan if st]
        so = min(pan, key=lambda t: t[1]["lb"]) if pan else None
        sp = max(pan, key=lambda t: t[1]["ub"]) if pan else None
        refs = []
        for rname in SPIKE_REFS:
            st = t16_complete(n, rname)
            refs.append(f"{st['p']:.2f} ({st['W']}-{st['T']}-{st['L']})" if st else "no complete panel")
        DB.append(f"| {a_}/{ROLE[sl]} | {short(n)} | {len(pan)} | "
                  + (f"{so[1]['lb']:.2f} vs {short(so[0])}" if so else "-") + " | "
                  + (f"{sp[1]['ub']:.2f} vs {short(sp[0])}" if sp else "-") + " | " + " | ".join(refs) + " |")
    ch = DS["architectures"]["B"].get("challenger")
    if ch:
        DB += ["", f"**B challenger** {ch} ({DS['architectures']['B'].get('challenger_note', '')})"]
        for a_, b_ in DS.get("t16_requests", []):
            DB.append(pl(a_, b_, W16, "T16"))
        for gf in sorted(glob.glob(str(R_DIR / "gates" / "FRESH48*.json"))):
            M = json.loads(Path(gf).read_text())
            v_, info_, st_ = exact_gate(load_gate_games(), M)
            DB.append(f"  - {Path(gf).name}: exact-SHA frozen gate {M['challenger']['name']} ({M['challenger']['sha256'][:12]}) vs "
                      f"{M['incumbent']['name']} ({M['incumbent']['sha256'][:12]}), {len(M['worlds'])} worlds x 2 seats, deadline {M['deadline_utc']}: "
                      f"**{v_}** | complete {info_['complete']}/{len(M['worlds'])}, wrong-SHA {info_['wrong_sha']}, late {info_['late']}, "
                      f"conflicts {info_['conflicts']}" + (f" | {info_['why']}" if info_.get("why") else "")
                      + (f" | W-D-L {st_['W']}-{st_['T']}-{st_['L']}, margin {st_['margin']:+,.0f}" if st_ else "")
                      + (f" | runtimes per world {info_.get('world_runtimes')}" if info_.get("world_runtimes") else "")
                      + (f"\n    - {info_['same_runtime_only']}" if info_.get("same_runtime_only") else ""))
    rel_, sps_ = spike_names(DS)
    if sps_:
        rows_ = []
        banks_ = complete_banks()
        for c_ in sps_:
            A_, B_ = sorted((c_, rel_)); w = P.get((A_, B_), {})
            best = None
            for bn, seeds_ in banks_.items():
                ws = {sd: w[sd] for sd in seeds_ if sd in w}
                if ws and (best is None or len(ws) > len(best[1])):
                    best = (bn, ws)
            if best:
                st = reduce_pair(best[1] if c_ == A_ else orient(best[1]))
                rows_.append((st["p"], f"| {c_} | {best[0]} {st['worlds_complete']}/{len(banks_[best[0]])} worlds | {st['p']:.2f} | "
                                       f"[{st['lb']:.2f}, {st['ub']:.2f}] | {st['W']}-{st['T']}-{st['L']} |"))
            else:
                rows_.append((-1, f"| {c_} | no games yet | - | - | - |"))
        DB += ["", f"**Spike slot: candidates ranked by paired score vs the reliable entry {rel_}** (adversarial merge: not a .60 bar; "
               "the reliable entry still needs its own fresh confirmation)", "",
               "| candidate | bank (worlds) | vs reliable p | KL 95% | W-D-L |", "|---|---|---|---|---|"] + [r for _, r in sorted(rows_, key=lambda t: -t[0])]
    LG = DS.get("land_gates") or []
    if LG:
        try:
            L64 = json.loads((R_DIR / "banks" / "LAND64.json").read_text())["seeds"]
            sub = json.loads((R_DIR / "land64" / "state.json").read_text())["submitted"] if (R_DIR / "land64" / "state.json").exists() else {}
        except (OSError, ValueError, KeyError):
            L64, sub = [], {}
        DB += ["", "**3Q-vs-4Q land gate (LAND64: 64 fresh worlds x 2 seats; land-off variant vs unmodified parent)**", "", GATE_RULE, ""]
        for g in LG:
            s_ = sub.get(g["name"])
            DB.append(pl(s_["a"], s_["b"], L64, "LAND64") if s_ else f"  - {g['name']}: waiting for the land-off archive to be registered")
            mf_ = R_DIR / "gates" / f"LAND64-{g['name']}.json"
            if s_ and mf_.exists():
                M_ = json.loads(mf_.read_text()); v_, info_, st_ = exact_gate(load_gate_games(), M_)
                try:
                    lf_ = json.loads((R_DIR / "land64" / g["name"] / "LAND-FIRE.json").read_text())["rule_on_seat"]
                    fire_ = (f"; rule-on seat: land predicate held in {lf_['predicate_held_games']}/{lf_['games']} games, BUY_LAND ordered at the earliest "
                             f"predicate step in {lf_.get('buy_land_at_earliest_pred_step', lf_.get('buy_land_on_first_pred_step'))}/{lf_['games']}")
                except (OSError, ValueError, KeyError):
                    fire_ = "; land-rule diagnostic not yet run"
                DB.append(f"    - frozen gate {mf_.name} ({M_['challenger']['sha256'][:12]} vs {M_['incumbent']['sha256'][:12]}, deadline {M_['deadline_utc']}"
                          + (", POST-HOC: rule declared after this result was visible" if M_.get("post_hoc") else "") + f"): **{v_}** | "
                          f"complete {info_['complete']}/{len(M_['worlds'])}, wrong-SHA {info_['wrong_sha']}, late {info_['late']}, rejected {info_['rejected']}"
                          + (f" | {info_['why']}" if info_.get("why") else "") + fire_
                          + (f" | runtimes per world {info_.get('world_runtimes')}" if info_.get("world_runtimes") else "")
                          + (f"\n      - {info_['same_runtime_only']}" if info_.get("same_runtime_only") else ""))
    def common_opp(x_, y_, o_):
        """Paired x-minus-y world-mean cash margin vs a common opponent o on worlds where both pairs are complete (both seats)."""
        def wm(a__):
            A__, B__ = sorted((a__, o_)); w__ = P.get((A__, B__), {})
            return {s: sum(v.values()) / 2 * (1 if a__ == A__ else -1) for s, v in w__.items() if len(v) == 2}
        X_, Y_ = wm(x_), wm(y_); S_ = sorted(set(X_) & set(Y_))
        d_ = np.array([X_[s] - Y_[s] for s in S_])
        if len(d_) < 3:
            return f"  - {short(x_)} minus {short(y_)} vs {short(o_)}: {len(d_)} common complete worlds (too few)"
        m_ = d_.mean(); h_ = 1.96 * d_.std(ddof=1) / math.sqrt(len(d_))
        loo = [(d_.sum() - v) / (len(d_) - 1) for v in d_]
        return (f"  - {short(x_)} minus {short(y_)} vs {short(o_)} (league recomputation): {m_:+,.0f}/world [{m_ - h_:+,.0f}, {m_ + h_:+,.0f}] "
                f"(normal 95%), leave-one-world-out {min(loo):+,.0f} to {max(loo):+,.0f}, {len(d_)} common complete worlds (all banks, discovery)")
    DB += ["", "**Diagnostics (not decisions)**",
           "  - DOT check (owner's DOT assistant, 22:40): vs OA79, paired B31 - B25 = +$1,727/world (t 95% CI +$499 to +$2,956; "
           "leave-one-world-out +$1,468 to +$1,990). Source: review/inbox/HEY CLAUDE README.txt; quoted, not recomputed.",
           common_opp("base1-b-u31047-v8g-a1-ema", "base1-b-u25000-v8g-a1-ema", "oldarch1-v4bg2-u79000-cd-ema")]
    try:
        DB += [""] + (R_DIR / "regular" / "REGULAR.md").read_text().splitlines()
    except OSError:
        pass
    TR = DS.get("tape_refs") or {}
    if TR:
        DB += ["", "**Open-loop tape references (NOT policies: recorded seats replayed without reacting; first 8 T16 worlds x 2 seats)**"]
        for c_ in TR.get("candidates", []):
            for t_ in TR.get("tapes", []):
                DB.append(pl(c_, t_, W16[:8], "tape ref"))
    DB += ["", "**Finalist confirmation (fresh48 = first 48 of FRESH96; then fresh96)**"]
    for a_, b_ in DS["finalist_pairs"]:
        DB.append(pl(a_, b_, F96[:48], "fresh48")); DB.append(pl(a_, b_, F96, "fresh96 (descriptive only, post-decision; not a second look)"))
    DB.append("")
    L = DB + [f"# Panel league board ({now()}; {len(games)} clean CRN games ({len(rej)} rejected by the provenance gate -> REJECTED.jsonl),"
         f" {len(P)} pairs; qualified table {len(games_q)} exact-W16 exact-SHA games; reducer {REDUCER_VERSION})", ""]
    b = ba["best"]
    if b:
        L.append(f"**BEST:** {b['name']} S={b['S']:.3f} minLB={b['minLB']:.3f}")
    elif not ba.get("fixed"):
        L.append("**BEST:** none (T re-pick did not reach a fixed point; sets in the cycle/cap: "
                 + " -> ".join("{" + ", ".join(c) + "}" for c in ba.get("cycle_sets", [])) + ")")
    else:
        L.append("**BEST:** none yet (no T member has completed best-of-32 panels vs every other T member and every R member)")
    L += ["", f"T (top 8 by S_discovery, re-picked after every result; fixed point{', CYCLED' if ba['cycled'] else ''}); "
          f"R (immutable, required) = {', '.join(ba['R'])}", "",
          "S_qualified = mean KL LB95 over the 18 required T/R edges from the exact-W16, exact-archive-SHA panel table only "
          "(missing or unavailable edges count 0, never imputed). S_discovery = the scheduler's view (history + BT-provisional).", "",
          "| entry | S_qualified | panels_complete | S_discovery (provisional) | model-provisional terms | missing | unavailable refs |",
          "|---|---|---|---|---|---|---|"]
    for r, q in zip(ba["T"], qrows):
        L.append(f"| {r['name']} | {q['S_qualified']:.3f} | {q['panels_complete']}/{q['required']} | {r['S']:.3f} | {r['model_implied']} | "
                 f"{len(q['missing'])} | {', '.join(q['unavailable']) or '-'} |")
    bank_pairs = defaultdict(lambda: [0, 0])
    bank_seeds = {"T16": set(W16), "FP16": set(range(32200001, 32200017))}
    for (A_, B_), worlds in P.items():
        present = {bank(sd) for sd in worlds}
        for bk in present:
            bank_pairs[bk][0] += 1
            need = bank_seeds.get(bk)
            if need and all(len(worlds.get(x, {})) == 2 for x in need):
                bank_pairs[bk][1] += 1
    L += ["", "## Banks (T16 = qualification; FP16 (32200001-16) and fresh16 = confirmation; history = discovery only)", "",
          "| bank | pairs with games | pairs complete (16 worlds x 2 seats) | games |", "|---|---|---|---|"]
    gcount = defaultdict(int)
    for r in games:
        gcount[bank(r["world_seed"])] += 1
    for bk in ("T16", "FP16", "fresh16", "history"):
        L.append(f"| {bk} | {bank_pairs[bk][0]} | {bank_pairs[bk][1] if bk in bank_seeds else '-'} | {gcount[bk]} |")
    L += ["", "## Elo/BT on world outcomes (ties 1/2; 1 virtual draw vs the mean), 95% interval", "", "| entry | rating | 95% |", "|---|---|---|"]
    for n, (r, s) in sorted(ratings.items(), key=lambda kv: -(kv[1][0] - 1.96 * kv[1][1]))[:60]:
        L.append(f"| {n} | {r:.0f} | [{r - 1.96 * s:.0f}, {r + 1.96 * s:.0f}] |")
    L += ["", "## Pairs (sorted by lower bound of p for the better side)", ""]
    lines = []
    for (A, B), s in stats.items():
        if s["p"] >= 0.5:
            lines.append((s["lb"], fmt(A, B, s)))
        else:
            t = reduce_pair(orient(P[(A, B)]))
            lines.append((t["lb"], fmt(B, A, t)))
    for _, l in sorted(lines, key=lambda x: -x[0]):
        L.append("- " + l)
    (R_DIR / "BOARD.md").write_text("\n".join(L) + "\n")
    # PANELS.md: every pair with a T16 panel (both orders on the common worlds)
    PL = [f"# Panels on WORLDS-T16 ({now()}); same reducer; rare worlds {W16[-2:]} included (and excluded in the second line)", ""]
    for (A, B), worlds in sorted(P.items()):
        w16 = {s: worlds[s] for s in W16 if s in worlds}
        if not w16:
            continue
        s = reduce_pair(w16)
        pw_ = popw_p(w16)
        PL.append("- " + fmt(A, B, s, f" | T16 {sum(1 for w in w16.values() if len(w) == 2)}/16"
                             f" | pop-weighted p (balanced 14) {'n/a' if pw_ is None else f'{pw_:.2f}'}"))
        w14 = {k: v for k, v in w16.items() if k not in W16[-2:]}
        if w14 and len(w14) != len(w16):
            s = reduce_pair(w14); PL.append("  - without rare: " + fmt(A, B, s))
    names_all = {n for k in P for n in k} | set(pool)
    for title, kinds in (("Parent panels (child vs parent; child vs S leader)", ("parent", "vs-S-leader")),
                         ("Arm vs control (tu8 arms vs tu8c / c2, same worlds)", ("arm-vs-control",)),
                         ("Keeper / matched-LR / requested panels", ("vs-OA1-keeper", "matched-lr", "request"))):
        PL += ["", f"## {title}", ""]
        for child, par, kind in parent_edges(names_all):
            if kind.split(":")[0] not in kinds:
                continue
            A, B = sorted((child, par))
            w16 = {x: P.get((A, B), {})[x] for x in W16 if x in P.get((A, B), {})}
            if not w16:
                PL.append(f"- {child} vs {par} ({kind}): scheduled, no games yet"); continue
            st = reduce_pair(w16 if child == A else orient(w16))
            PL.append("- " + fmt(child, par, st, f" | {kind} | T16 {sum(1 for w in w16.values() if len(w) == 2)}/16"))
    (R_DIR / "PANELS.md").write_text("\n".join(PL) + "\n")
    matrix16(stats, ba, P)
    prev = None
    if BEST_LOG.exists():
        for line in BEST_LOG.open():
            prev = json.loads(line).get("best")
    cur = b["name"] if b else None
    if cur != prev:
        with BEST_LOG.open("a") as f:
            f.write(json.dumps({"utc": now(), "best": cur, "S": b["S"] if b else None, "T": [r["name"] for r in ba["T"]]}) + "\n")
    print(L[2] if len(L) > 2 else "", "| T:", [r["name"] for r in ba["T"]])
    return P, stats, ratings, ba, pool


OA1_PARENT = "oldarch1-v4bg2-u65000-cd-ema"
S_LEADER_REF = "base1-b-u25000-v8g-a1-ema"
EXPLICIT_PARENTS = {"base1-b-soup75-u25k-u27k-v8g-a1-ema": "base1-b-u25000-v8g-a1-ema", "base1-b-soup50-u25k-u27k-v8g-a1-ema": "base1-b-u25000-v8g-a1-ema"}   # GPU operator 09-29: EMA soup trial parent
ARM_CONTROL = {"decw": "c2", "tet": "c2", "boeyc": "c2"}      # gpu-ops 18:20: leak-fixed roster arms -> c2; old-roster arms -> tu8c
REQUESTS = Path("/mnt/ai/scratch/farmNEAT/league/records/REQUESTS.tsv")   # the only way to ask for a panel (a<TAB>b<TAB>reason)
CT118_QUEUE = Path("/mnt/ai/scratch/farmNEAT/league/records/ct118-queue.jsonl")
CHUNK9_END = {"b": 31047, "c": 27912}


def chunk_ends(run):
    out = []
    for f in glob.glob(f"/mnt/ai/scratch/farmNEAT/base-v1/runs/{run}/chunks/chunk*-ema-ckpt-*.pt"):
        m = re.search(r"chunk(\d+)-ema-ckpt-(\d+)\.pt$", f)
        if m:
            out.append(int(m.group(2)))
    return sorted(out)


def parent_edges(names):
    """Parent-panel rule (coordinator 09-29 17:30): (child, parent, kind). OA1 tu8 touch-up children -> OA1 u65000; B/C checkpoints
    newer than chunk 9 -> greedy archive of the preceding chunk end; every child also vs B u25000 greedy (S leader);
    tu8 arms (boey, tu8e, tv) -> tu8c control at the same (else nearest) update."""
    names = set(names)
    edges = []
    ctrl = {"tu8c": {}, "c2": {}}
    for n in names:
        m = re.match(r"oldarch1-tu8-(\w+?)-u(\d+)-", n)
        if m and m.group(1) in ctrl:
            ctrl[m.group(1)][int(m.group(2))] = n
    for n in sorted(names):
        m = re.match(r"oldarch1-tu8-(\w+?)-u(\d+)-", n)
        if m:
            edges.append((n, OA1_PARENT, "parent"))
            fam = ARM_CONTROL.get(m.group(1), "tu8c")
            cm = ctrl[fam]
            if m.group(1) not in ("tu8c", "c2") and cm:
                u = int(m.group(2)); c = cm.get(u) or cm[min(cm, key=lambda k: abs(k - u))]
                edges.append((n, c, "arm-vs-control"))
        else:
            m = re.match(r"base1-([bc])-u(\d+)-", n)
            if not m or int(m.group(2)) <= CHUNK9_END[m.group(1)]:
                continue
            run, u = m.group(1), int(m.group(2))
            prev = [e for e in chunk_ends(run) if e < u]
            if not prev:
                continue
            par = next((x for x in (f"base1-{run}-u{prev[-1]}-v8g-a1-ema", f"base1-{run}-u{prev[-1]}-v8-a1-ema") if x in names), None)
            if par and par != n:
                edges.append((n, par, "parent"))
        if n != S_LEADER_REF and n in names and S_LEADER_REF in names and (n.startswith("oldarch1-tu8-") or edges and edges[-1][0] == n):
            edges.append((n, S_LEADER_REF, "vs-S-leader"))
    for child, par in EXPLICIT_PARENTS.items():   # names outside the regex families (e.g. EMA soups)
        if child in names and par in names and (child, par, "parent") not in edges:
            edges.append((child, par, "parent"))
    for n in sorted(names):
        m = re.match(r"base1-b-u(\d+)-v8g-a1-ema$", n)            # B chunk-end greedy after chunk 9 (B10, B11...) also vs OA1 u65000
        if m and int(m.group(1)) > CHUNK9_END["b"] and int(m.group(1)) in chunk_ends("b") and OA1_PARENT in names:
            edges.append((n, OA1_PARENT, "vs-OA1-keeper"))
        m = re.match(r"base1-b-lr1e5-u31047-u(\d+)-v8g", n)         # B31@1e-5 branch vs the main run at the same update (matched LR)
        if m and f"base1-b-u{m.group(1)}-v8g-a1-ema" in names:
            edges.append((n, f"base1-b-u{m.group(1)}-v8g-a1-ema", "matched-lr"))
    if REQUESTS.exists():
        for line in REQUESTS.read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 2 and not line.startswith("#") and f[0] in names and f[1] in names and f[0] != f[1]:
                edges.append((f[0], f[1], "request:" + (f[2] if len(f) > 2 else "")))
    return edges


def popw_p(w16):
    """World-average score weighted by the population share of each balanced medoid's cluster (WORLDS-T16.json); rare worlds out."""
    J = json.loads((R_DIR / "WORLDS-T16.json").read_text())
    share = {int(k): v for k, v in J["cluster_population_share"].items()}
    num = den = 0.0
    for w in J["worlds"]:
        if w.get("tag") != "balanced" or int(w["seed"]) not in w16:
            continue
        g = w16[int(w["seed"])]
        x = sum(pts(m) for m in g.values()) / len(g)
        num += share[int(w["cluster"])] * x; den += share[int(w["cluster"])]
    return num / den if den else None


def fresh96():
    try:
        return json.loads((R_DIR / "banks" / "FRESH96.json").read_text())["seeds"]
    except (OSError, ValueError, KeyError):
        return []


def decision_set():
    D = json.loads((R_DIR / "DECISION-SET.json").read_text())
    try:
        G = json.loads((R_DIR / "GUARANTEED.json").read_text())
    except (OSError, ValueError):
        G = {}
    for arch_, a in D["architectures"].items():             # owner 18:59: old guaranteed = backup (previous champion); no survival override
        a["old_guaranteed_status"] = ("control = parent (best = backup)" if a.get("old_guaranteed") and a.get("old_guaranteed") == a.get("current_best")
                                      else "parent = previous champion")
    e = [u for u in chunk_ends("b") if u > CHUNK9_END["b"]]
    D["m12"] = []                                                # (arch, slot, name) in MATRIX-12 order
    for arch_, a in D["architectures"].items():
        if a.get("experiment") == "auto:B10":
            a["experiment"] = f"base1-b-u{e[0]}-v8g-a1-ema" if e else None
        if a.get("experiment") == "auto:newest" or a.get("current_best") == "auto:screen-best":
            ns = lineage_archives(a.get("name_regex"))
            if a.get("experiment") == "auto:newest":
                a["experiment"] = ns[-1] if ns else None
            if a.get("current_best") == "auto:screen-best":
                try:
                    a["current_best"] = json.loads((R_DIR / "nscreen" / f"{arch_}-BEST.json").read_text())["best"]
                except (OSError, ValueError, KeyError):
                    a["current_best"] = None
        for slot in ("old_guaranteed", "current_best", "experiment"):
            v = a.get(slot)
            if slot == "old_guaranteed" and v and v == a.get("current_best"):
                v = None                                             # best = backup: shown once (in the current-best cell)
            D["m12"].append((arch_, slot, v))
    D["bests"] = [a["current_best"] for a in D["architectures"].values() if a.get("current_best")]
    return D


def lineage_archives(rx):
    """Deployed archives whose name matches rx, oldest first (by the -u<N> update number, else archive mtime)."""
    if not rx:
        return []
    out = []
    for d in glob.glob("/mnt/ai/scratch/farmNEAT/pbc-deploy/*/"):
        n = Path(d).name
        if re.search(rx, n) and Path(d, f"{n}.tar.gz").exists() and Path(d, "VERIFY.json").exists():
            m = re.search(r"-u(\d+)", n)
            out.append((int(m.group(1)) if m else 0, Path(d, f"{n}.tar.gz").stat().st_mtime, n))
    return [n for *_, n in sorted(out)]


LINEAGE_PREFIX = {"B": "base1-b-", "C": "base1-c-", "OA1": "oldarch1-"}


def complete_banks():
    """Named world sets on which a pair's panel counts as complete (all seeds, both seat orders)."""
    banks = {"T16": list(worlds16()), "FP16": list(range(32200001, 32200017))}
    try:
        banks["fresh16"] = json.loads((R_DIR / "finalists" / "fresh16" / "MANIFEST.json").read_text())["seeds"]
    except (OSError, ValueError, KeyError):
        pass
    f96 = fresh96()
    if f96:
        banks["fresh48"] = f96[:48]; banks["fresh96"] = f96
    for f in glob.glob("/mnt/ai/scratch/farmNEAT/league/extfarm/finalpanel/FRESH-WORLDS-set*.txt"):
        banks[Path(f).stem.lower()] = [int(x) for x in Path(f).read_text().split() if x.strip().isdigit()]
    try:
        banks["land64"] = json.loads((R_DIR / "banks" / "LAND64.json").read_text())["seeds"]
    except (OSError, ValueError, KeyError):
        pass
    for d in ("r3-oa79-oa65", "r3-c22-oa79"):
        try:
            rows = [l.split("\t") for l in (R_DIR / d / "panel" / "pairs.tsv").read_text().splitlines()[1:] if l.strip()]
            banks[d] = sorted({int(r[4]) for r in rows if len(r) > 4 and r[4].isdigit()})
        except (OSError, ValueError):
            pass
    return banks


def guaranteed_analysis(P):
    """Owner 18:58: old guaranteed = the entry most often challenged by its own lineage without being beaten. A challenge = a complete
    panel (largest complete bank for that pair); survived = challenger's KL LB <= .5 (close-loss if challenger p > .5); decisive =
    challenger's LB > .5. Old guaranteed = most survived with >= 3 and 0 decisive; else the configured pick, marked provisional."""
    banks = complete_banks()
    stats = {}
    for arch_, pre in LINEAGE_PREFIX.items():
        names = sorted({n for k in P for n in k if n.startswith(pre)})
        for x in names:
            sv = cl = dec = 0; det = []
            for y in names:
                if y == x:
                    continue
                A_, B_ = sorted((x, y)); w = P.get((A_, B_), {})
                best = None
                for bn, seeds in banks.items():
                    if seeds and all(set(w.get(sd, {})) == {False, True} for sd in seeds) and (best is None or len(seeds) > len(best[1])):
                        best = (bn, seeds)
                if not best:
                    continue
                ws = {sd: w[sd] for sd in best[1]}
                st = reduce_pair(ws if y == A_ else orient(ws))           # challenger y's view
                if st["lb"] > 0.5:
                    dec += 1; det.append(f"LOST to {y} ({best[0]} LB {st['lb']:.2f})")
                else:
                    sv += 1
                    if st["p"] > 0.5:
                        cl += 1
            if sv or dec:
                stats[x] = {"arch": arch_, "survived": sv, "close_loss": cl, "decisive": dec, "detail": det}
    picks = {}
    for arch_ in LINEAGE_PREFIX:
        c = [(n, v) for n, v in stats.items() if v["arch"] == arch_ and v["survived"] >= 3 and v["decisive"] == 0]
        picks[arch_] = max(c, key=lambda t: (t[1]["survived"], t[0]))[0] if c else None
    return stats, picks


def spike_names(D):
    sp = D.get("spike") or {}
    rel = D["architectures"]["B"]["current_best"] if sp.get("reliable") == "B:current_best" else sp.get("reliable")
    out = []
    for c in sp.get("candidates", []) + sp.get("extra", []):
        if c == "oa3:entry":                                      # only OA3 screen checkpoints enter the spike slot (gpu-ops 20:25)
            e_ = (D.get("oa3") or {}).get("entry") or ""
            m_ = re.match(r"oa3-e6d2-288-v8-u(\d+)-ema$", e_)
            try:
                keys_ = (json.loads((R_DIR / "banks" / "OA3.json").read_text()).get("continuation") or {}).keys()
            except (OSError, ValueError):
                keys_ = []
            if m_ and m_.group(1) not in keys_ and e_ != (D.get("oa3") or {}).get("final_entry"):
                continue
        n = ((D.get("oa3") or {}).get("entry") if c == "oa3:entry" else
             D["architectures"][c.split(":")[0]].get("experiment") if c.endswith(":experiment") else c)
        if n and n != rel:
            out.append(n)
    return rel, out


def bank(seed):
    seed = int(seed)
    if seed in set(worlds16()):
        return "T16"
    f96 = fresh96()
    if seed in f96:
        return "fresh48" if f96.index(seed) < 48 else "fresh96"
    if 32200001 <= seed <= 32200016:
        return "FP16"
    try:
        if seed in set(json.loads((R_DIR / "finalists" / "fresh16" / "MANIFEST.json").read_text())["seeds"]):
            return "fresh16"
    except (OSError, ValueError, KeyError):
        pass
    try:
        if seed in set(json.loads((R_DIR / "banks" / "DAGGER64.json").read_text())["seeds"]):
            return "DAGGER64"
    except (OSError, ValueError, KeyError):
        pass
    try:
        cb = json.loads((R_DIR / "banks" / "COMBO.json").read_text())
        if seed in set(cb["screen16"]) | set(cb["confirm32"]):
            return "COMBO"
    except (OSError, ValueError, KeyError):
        pass
    try:
        if seed in set(json.loads((R_DIR / "banks" / "SURGERY.json").read_text())["seeds"]):
            return "SURGERY"
    except (OSError, ValueError, KeyError):
        pass
    return "history"


ROLE = {"old_guaranteed": "parent", "current_best": "control", "experiment": "experiment"}


def lineage(n):
    return ("OA1" if n.startswith("oldarch1-") else "OA2" if n.startswith("oldarch2-") else "B" if n.startswith("base1-b-") else
            "C" if n.startswith("base1-c-") else "N" if n.startswith("n1-") else "tape" if n.startswith(("pub-", "tape-")) else
            "ref" if n.startswith(("v4b-", "fwd1", "lineageB-")) else "other")


_CELL = {}


def cell(stats, a, b, P=None, Pq=None):
    """Row a vs column b, every field recomputed from a-oriented worlds. Source 'T16' = complete exact-W16 exact-SHA panel;
    'history' = discovery data (observed, not qualified). Model-implied values are never shown as cells."""
    A, B = sorted((a, b))
    key = (a, b)
    if key in _CELL:
        return _CELL[key]
    W16 = worlds16()
    src, worlds = None, None
    if Pq is not None and all(set(Pq.get((A, B), {}).get(x, {})) == {False, True} for x in W16):
        src, worlds = "T16", {x: Pq[(A, B)][x] for x in W16}
    elif P is not None and P.get((A, B)):
        src, worlds = "history", P[(A, B)]
    if worlds is None:
        _CELL[key] = None; return None
    s = reduce_pair(worlds if a == A else orient(worlds))
    tag = {"A>B": "row>col", "B>A": "col>row"}.get(s["decided"], "undecided")
    _CELL[key] = {"lb": round(s["lb"], 3), "ub": round(s["ub"], 3), "p": round(s["p"], 3), "worlds": s["worlds"],
                  "worlds_complete": s["worlds_complete"], "decided": tag, "thin": s["worlds"] < 4, "source": src}
    return _CELL[key]


def matrix16(stats, ba, P=None):
    _CELL.clear()
    Pq = ba.get("Pq")
    _cell = cell
    cell_ = lambda st, a, b: _cell(st, a, b, P, Pq)
    ranked = sorted(ba["S"], key=lambda n: (-ba["S"][n], -ba["minLB"].get(n, 0)))[:16]
    cols = ranked + ba["R"]
    everyone = sorted({n for k in stats for n in k})
    outside = [n for n in everyone if n not in ranked]
    def worst_outside(a):
        cs = [(c, cell_(stats, a, c)) for c in outside if c != a]
        cs = [(c, x) for c, x in cs if x]
        return min(cs, key=lambda t: (t[1]["lb"], t[1]["p"])) if cs else (None, None)
    def strongest_outsider(b):
        cs = [(c, cell_(stats, c, b)) for c in outside if c != b]
        cs = [(c, x) for c, x in cs if x]
        return max(cs, key=lambda t: (t[1]["lb"], t[1]["p"])) if cs else (None, None)
    J = {"utc": now(), "reducer": REDUCER_VERSION, "rubric": "S = mean LB95 of p over T u R (T = top 8 fixed point)",
         "rows": [{"rank": i + 1, "name": n, "S": round(ba["S"][n], 3), "minLB": round(ba["minLB"].get(n, 0), 3), "lineage": lineage(n),
                   "cells": {c: cell_(stats, n, c) for c in cols if c != n},
                   "worst_outside": dict(zip(("name", "cell"), worst_outside(n)))} for i, n in enumerate(ranked)], "columns": cols,
         "reference_columns": ba["R"],
         "strongest_outsider": {c: dict(zip(("name", "cell"), strongest_outsider(c))) for c in cols}}
    (R_DIR / "MATRIX-16.json").write_text(json.dumps(J, indent=1))
    short = lambda n: n.replace("base1-", "").replace("-a1-ema", "").replace("oldarch", "oa").replace("-cd-ema", "").replace("-lw", "")
    L = [f"# MATRIX-16 ({J['utc']}; reducer {REDUCER_VERSION}; rubric {J['rubric']})", "",
         "Cell = row vs column: p 95% KL [LB, UB] (world-averaged points, complete worlds), p, n worlds (complete), decided, source (T16 = qualified exact panel; history = discovery only). **missing** = no games; never model-imputed.", "",
         "| # | entry | S | minLB | lin | " + " | ".join(short(c) for c in cols) + " | worst outside |", "|" + "---|" * (6 + len(cols))]
    fc = lambda x: (f"[{x['lb']:.2f},{x['ub']:.2f}] {x['p']:.2f} n{x['worlds']}({x['worlds_complete']}) {x['decided']}"
                    f"{' thin' if x['thin'] else ''} {x['source']}")
    for r in J["rows"]:
        cs = []
        for c in cols:
            if c == r["name"]:
                cs.append("—"); continue
            x = r["cells"].get(c)
            cs.append("**missing**" if x is None else fc(x))
        wo = r["worst_outside"]
        cs.append("none" if not wo.get("cell") else f"{short(wo['name'])} {fc(wo['cell'])}")
        L.append(f"| {r['rank']} | {short(r['name'])} | {r['S']:.3f} | {r['minLB']:.3f} | {r['lineage']} | " + " | ".join(cs) + " |")
    so = J["strongest_outsider"]
    L.append("| — | strongest outsider (beats column) | | | | " + " | ".join(
        "none" if not so[c].get("cell") else f"{short(so[c]['name'])} {fc(so[c]['cell'])}" for c in cols) + " | |")
    (R_DIR / "MATRIX-16.md").write_text("\n".join(L) + "\n")


def screen_tally(margins, candidate_is_A):
    """Strict W/L/T tally from A-minus-B margins; zero is always a tie."""
    wins = sum(1 for margin in margins if (margin > 0 if candidate_is_A else margin < 0))
    losses = sum(1 for margin in margins if (margin < 0 if candidate_is_A else margin > 0))
    ties = sum(1 for margin in margins if margin == 0)
    return wins, losses, ties


def dispatch_top_names(analysis):
    """Return a fixed top set, or None when dispatch would chase a cycle."""
    if not analysis.get("fixed"):
        return None
    return [row["name"] for row in analysis["T"]]


# ------------------------------------------------------------------ matchmaking + SLURM dispatch
GPU_INFLIGHT = R_DIR / "GPU-INFLIGHT.json"
GPU_TTL = 3 * 3600


def gpu_reserved():
    """Units handed to the gpu-served lane (records/gpu/<batch>/<i>): reserved until game.json exists or GPU_TTL passes."""
    try:
        g = json.loads(GPU_INFLIGHT.read_text())
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in g.items() if not Path(v["dir"], "game.json").exists() and time.time() - v["t"] < GPU_TTL}


def sbatch_job(j):
    """Job id of a completed 'sbatch --parsable' call, or None unless rc == 0 and the id is numeric."""
    jid = (j.stdout or "").strip().split(";")[0]
    return jid if j.returncode == 0 and re.fullmatch(r"\d+", jid) else None


def plan(cap, gpu=None):
    P, stats, ratings, ba, pool = board()
    T = dispatch_top_names(ba)
    if T is None:
        print("plan: T is not a unique fixed point; freeze/resolve T before dispatch")
        return
    db = league_db()
    arch = {s["name"]: s["archive_path"] for s in db["subs"] if s["status"] in ("active", "provisional")}
    st = json.loads(STATE.read_text()) if STATE.exists() else {"inflight": {}, "next_eid": 950000000, "wave": 0}
    live = set(subprocess.run(["squeue", "-h", "-n", "panel-league", "-o", "%F"], capture_output=True, text=True).stdout.split())
    for k, v in list(st["inflight"].items()):
        if Path(v["dir"], "game.json").exists() or (v["job"] not in live and time.time() - v["t"] > 600):
            st["inflight"].pop(k)
    reg_sha = {s_["name"]: s_["archive_sha256"] for s_ in db["subs"]}
    have = {(int(r["world_seed"]), r["seat0"], r["seat1"]) for r in load_games()
            if r["archive_sha256"] == [reg_sha.get(r["seat0"]), reg_sha.get(r["seat1"])]}    # an old artifact SHA does not count
    todo = []
    W16 = worlds16()
    R = ba["R"]
    # owner 18:24 priority: [2558 finalist contrast: own SLURM job, submitted first] > P1 current-best cross matrix (T16: every
    # lane's current best vs every other + old guaranteed) > P2 every experiment vs every lane current best + old guaranteed (T16) >
    # P3 finalist fresh confirmation (FRESH96; first 48 = fresh48) > P4 attribution, requests, FP16 completion > P5 T/R, parent, matrix
    # > P6 screening, old-roster arms
    D = decision_set()
    F96 = fresh96()
    bests = set(D["bests"])
    names12 = [n for _, _, n in D["m12"] if n]
    for i, a_ in enumerate(names12):
        for b_ in names12[i + 1:]:
            if a_ in arch and b_ in arch and a_ != b_:
                pr = 1 if (a_ in bests and b_ in bests) else 2          # MATRIX-12: current-best pairs first, then the rest
                for sd in W16:
                    for x, y in ((a_, b_), (b_, a_)):
                        todo.append((pr, sd, x, y))
    for a_, b_ in D.get("t16_requests", []):                     # e.g. B challenger vs other architectures' current bests + old guaranteed
        if a_ in arch and b_ in arch:
            for sd in W16:
                for x, y in ((a_, b_), (b_, a_)):
                    todo.append((2, sd, x, y))
    OB = json.loads((R_DIR / "banks" / "OA3.json").read_text())
    rel_b = D["architectures"]["B"]["current_best"]
    fin = (D.get("oa3") or {}).get("final_entry")
    for n in sorted(arch):                                        # OA3 continuation screens (gpu-ops 20:05), top priority
        m_ = re.match(r"oa3-e6d2-288-v8-u(\d+)-ema$", n)
        if not m_:
            continue
        key = "final" if n == fin else m_.group(1)
        for bank_, opps in (OB.get("continuation") or {}).get(key, []):
            for o_ in opps:
                on = "oldarch1-v4bg2-u65000-cd-ema" if o_ == "OA1:u65000" else rel_b if o_ == "B:reliable" else o_
                if on in arch:
                    for sd in OB[bank_]:
                        for x, y in ((n, on), (on, n)):
                            todo.append((0.5, sd, x, y))
    for i, sd in enumerate(F96[:48]):
        for a_, b_ in D.get("fresh48_pairs", []):
            if a_ in arch and b_ in arch and [a_, b_] not in [x[:2] for x in D.get("fresh48_submitted", [])]:
                for x, y in ((a_, b_), (b_, a_)):
                    todo.append((3 + i / 1000.0, sd, x, y))
    for i, sd in enumerate(F96):
        for a_, b_ in D["finalist_pairs"]:
            if a_ in arch and b_ in arch:
                for x, y in ((a_, b_), (b_, a_)):
                    todo.append((3 + i / 1000.0, sd, x, y))
    rel_, sps_ = spike_names(D)
    for c_ in sps_:                                             # spike slot: every candidate vs the reliable entry, T16 (gate level)
        if c_ in arch and rel_ in arch:
            for sd in W16:
                for x, y in ((c_, rel_), (rel_, c_)):
                    todo.append((2, sd, x, y))
    TR = D.get("tape_refs") or {}
    for c_ in TR.get("candidates", []):                           # open-loop tape references, after the fresh finalist matches
        for t_ in TR.get("tapes", []):
            if c_ in arch and t_ in arch:
                for sd in W16[:8]:
                    for x, y in ((c_, t_), (t_, c_)):
                        todo.append((3.5, sd, x, y))
    for a_, b_ in D.get("attribution", []):
        if a_ in arch and b_ in arch:
            for sd in W16:
                for x, y in ((a_, b_), (b_, a_)):
                    todo.append((4, sd, x, y))
    FP = list(range(32200001, 32200017))
    for (A_, B_), worlds in P.items():                              # finish FP16 panels that are 1-2 games short
        have_fp = sum(len(worlds.get(sd, {})) for sd in FP)
        if 28 <= have_fp < 32 and A_ in arch and B_ in arch:
            for sd in FP:
                w = worlds.get(sd, {})
                if True not in w: todo.append((4, sd, A_, B_))
                if False not in w: todo.append((4, sd, B_, A_))
    old_roster = re.compile(r"oldarch1-tu8-(lr3|pro|tv|boey)-")
    for child, par, kind in parent_edges(arch):
        if child in arch and par in arch:
            pr = 0 if kind.startswith("request:") else 6 if old_roster.match(child) else 5   # REQUESTS.tsv header: priority 0
            for sd in W16:
                for x, y in ((child, par), (par, child)):
                    todo.append((pr, sd, x, y))
    # 1) missing T/R matchups (full best of 32 on the common worlds; no early stop)
    for i, a in enumerate(T):
        for o in T[i + 1:] + R:
            if a in arch and o in arch:
                for s in W16:
                    for x, y in ((a, o), (o, a)):
                        todo.append((5, s, x, y))
    # 2) MATRIX-16 rows 9-16 vs T (full best of 32 on the common worlds): fills the matrix's missing cells next
    ranked = sorted(ba["S"], key=lambda n: -ba["S"][n])[:16]
    for a in ranked[8:]:
        for o in T:
            if a in arch and o in arch:
                for s in W16:
                    for x, y in ((a, o), (o, a)):
                        todo.append((5, s, x, y))
    # 3) screening: fixed 8 games (4 reciprocal worlds); no optional early stop
    lead = T[0] if T else None
    for n in pool:
        if n in T or n in R or n in ranked or n not in arch:
            continue
        for o in [lead, "pub-ice-fire-wr1k-lw"]:
            if o and o in arch and o != n:
                A, B = sorted((n, o))
                w = P.get((A, B), {})
                games = [m for s in W16[:4] for m in w.get(s, {}).values()]
                screen_tally(games, candidate_is_A=(n == A))   # diagnostic only; fixed eight games are always completed
                for s in W16[:4]:
                    for x, y in ((n, o), (o, n)):
                        todo.append((6, s, x, y))
    gres = gpu_reserved()
    todo = [t for t in todo if (t[1], t[2], t[3]) not in have and f"{t[1]}|{t[2]}|{t[3]}" not in st["inflight"] and f"{t[1]}|{t[2]}|{t[3]}" not in gres]
    todo = [t for t in todo if not (t[2].startswith(EXCLUDED) or t[3].startswith(EXCLUDED))]
    todo.sort(key=lambda t: t[0])
    _seen = set()
    todo = [t for t in todo if (t[1], t[2], t[3]) not in _seen and not _seen.add((t[1], t[2], t[3]))]
    if gpu:                                                       # pull N planner units (both seat orders kept together) for the gpu lane
        n_, batch_ = gpu
        d_ = R_DIR / "gpu" / batch_
        if d_.exists():
            print("gpu-batch: directory exists", d_); return
        units, seen_ = [], set()
        for pr, sd, x, y in todo:
            k_ = (sd,) + tuple(sorted((x, y)))
            if k_ in seen_:
                continue
            seen_.add(k_); units += [t_ for t_ in todo if t_[1] == sd and {t_[2], t_[3]} == {x, y}]
            if len(units) >= n_:
                break
        d_.mkdir(parents=True)
        lines = ["a_name\ta_archive\tb_name\tb_archive\tworld_seed\tepisode_id"]
        with open(GPU_INFLIGHT.with_suffix(".lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            g_ = gpu_reserved()
            for i_, (pr, sd, x, y) in enumerate(units):
                lines.append(f"{x}\t{arch[x]}\t{y}\t{arch[y]}\t{sd}\t{int(time.time()) * 1000 + i_}")
                g_[f"{sd}|{x}|{y}"] = {"dir": str(d_ / str(i_)), "batch": batch_, "prio": pr, "t": time.time()}
            tmpp_ = d_ / ".pairs.tsv.tmp"; tmpp_.write_text("\n".join(lines) + "\n"); tmpp_.replace(d_ / "pairs.tsv")   # atomic: claimable once it exists
            tmp_ = GPU_INFLIGHT.with_suffix(".tmp"); tmp_.write_text(json.dumps(g_)); tmp_.replace(GPU_INFLIGHT)
        print(f"gpu-batch: {len(units)} games -> {d_ / 'pairs.tsv'} (priorities {sorted({u[0] for u in units})})")
        return
    room = cap - len(st["inflight"])
    played = {n for k in P for n in k}
    explore = [t for t in todo if t[0] == 6 and (t[2] not in played or t[3] not in played)]
    k = min(len(explore), max(2, max(0, room) // 10))           # exploration share for candidates with no games (ASTRA F7)
    rest = [t for t in todo if t not in explore[:k]]
    batch = (explore[:k] + rest)[:max(0, room)]
    if not batch:
        write_ct118_queue(todo)
        STATE.write_text(json.dumps(st)); print("plan: nothing to dispatch", len(st["inflight"]), "in flight"); return
    write_ct118_queue([t for t in todo if t not in batch])
    st["wave"] += 1
    # owner 18:40 "better use of slurm": per-node-class arrays with --mem = measured peak game RSS + 20% (x86 1162-1317 MB -> 1400;
    # Spark aarch64 2213 MB -> 2700) and --time 45 min for backfill; gpu-4090b (skynet never) at most 10 panel games while its
    # trainer holds 4 CPUs (2 spare for its loader), unless records/HOLD-4090B; no x86 array while the R3 confirmation is pending.
    def n4090b():
        out = subprocess.run(["squeue", "-h", "-r", "-w", "gpu-4090b", "-o", "%j"], capture_output=True, text=True).stdout.split()
        return sum(1 for x in out if x in ("panel-league", "r3-oa79-oa65", "league-dispatch"))
    classes = [("spark", ["-p", "spark", "--mem", "2700"], 20)]
    if not (R_DIR / "R3-RUNNING").exists():
        classes.append(("x86", ["-p", "cpu", "--mem", "1400"], 31))
    if not (R_DIR / "HOLD-4090B").exists():
        room4 = max(0, 10 - n4090b())
        if room4:
            classes.append(("g4090b", ["-p", "gpu", "-w", "gpu-4090b", "--mem", "1400"], room4))
    tot = sum(c[2] for c in classes)
    parts = {c[0]: [] for c in classes}
    acc = {c[0]: 0.0 for c in classes}
    for item in batch:                                            # proportional round-robin, priority order kept inside each class
        for c in classes:
            acc[c[0]] += c[2] / tot
        k = max(acc, key=acc.get); acc[k] -= 1.0
        if k == "g4090b" and len(parts[k]) >= classes[-1][2]:
            k = "spark"
        parts[k].append(item)
    jobs = []
    for name, args, _cap in classes:
        items = parts[name]
        if not items:
            continue
        d = R_DIR / "waves" / f"p{st['wave']:04d}-{name}"
        d.mkdir(parents=True, exist_ok=True)
        lines = ["a_name\ta_archive\tb_name\tb_archive\tworld_seed\tepisode_id"]
        for _, s_, x, y in items:
            st["next_eid"] += 1
            lines.append(f"{x}\t{arch[x]}\t{y}\t{arch[y]}\t{s_}\t{st['next_eid']}")
        (d / "pairs.tsv").write_text("\n".join(lines) + "\n")
        j = subprocess.run(["sbatch", "--parsable", *args, "-c", "1", "-t", "00:45:00", f"--array=0-{len(items) - 1}",
                            "-J", "panel-league", "-o", str(d / "slurm-%A_%a.out"), FARM_SBATCH, str(d / "pairs.tsv"), str(d)],
                           capture_output=True, text=True)
        job = sbatch_job(j)
        if job is None:
            print(f"plan: sbatch FAILED for {name} ({j.returncode}): {(j.stderr or '')[-200:]}; nothing marked in flight"); continue
        jobs.append(f"{name}:{job}:{len(items)}")
        for i_, (pr, s_, x, y) in enumerate(items):
            st["inflight"][f"{s_}|{x}|{y}"] = {"dir": str(d / str(i_)), "job": job, "t": time.time(), "prio": pr}
    job = ",".join(jobs)
    j = type("J", (), {"stderr": ""})()
    with PANELS.open("a") as f:
        f.write(json.dumps({"utc": now(), "wave": st["wave"], "job": job, "games": len(batch), "prio_counts":
                            {str(p): sum(1 for b in batch if int(b[0]) == p) for p in (0, 1, 2, 3, 4, 5, 6)}, "worlds": "WORLDS-T16",
                            "stop_rule": "parent/T/R: none (best of 32); screening: fixed 8", "reducer": REDUCER_VERSION}) + "\n")
    STATE.write_text(json.dumps(st))
    print(f"plan: wave {st['wave']} {len(batch)} games job {job} {j.stderr.strip()[:200]}")


def write_ct118_queue(rest, units=40):
    """Panel units (one world seed, both seat orders) for the league.lan worker: the next units the SLURM batch did not take."""
    seen, out = set(), []
    for pr, sd, x, y in rest:
        k = (sd,) + tuple(sorted((x, y)))
        if k in seen:
            continue
        seen.add(k); out.append({"prio": pr, "world_seed": sd, "a": x, "b": y, "bank": bank(sd)})
        if len(out) >= units:
            break
    tmp = CT118_QUEUE.with_suffix(".tmp"); tmp.write_text("".join(json.dumps(u) + "\n" for u in out)); os.replace(tmp, CT118_QUEUE)


def export():
    day = time.strftime("%Y-%m-%d", time.gmtime())
    d = R_DIR / "export" / day
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M", time.gmtime())
    for src in (GAMES, PANELS, BEST_LOG, R_DIR / "BOARD.md", R_DIR / "PANELS.md", R_DIR / "WORLDS-T16.json"):
        if src.exists():
            (d / f"{stamp}-{src.name}").write_bytes(src.read_bytes())
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(d.iterdir()) if p.name != "SHA256SUMS"]
    (d / "SHA256SUMS").write_text("\n".join(sums) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest", "board", "plan", "loop", "export", "gpu-batch"])
    ap.add_argument("--n", type=int, default=64); ap.add_argument("--batch")
    ap.add_argument("--cap", type=int, default=60)
    a = ap.parse_args()
    if a.cmd == "ingest":
        ingest()
    elif a.cmd == "board":
        board()
    elif a.cmd == "plan":
        plan(a.cap)
    elif a.cmd == "export":
        export()
    elif a.cmd == "gpu-batch":
        plan(a.cap, gpu=(a.n, a.batch or time.strftime("b%m%d-%H%M%S")))
    else:
        last_export = 0
        while not (R_DIR / "STOP").exists():
            try:
                ingest()
                bf = subprocess.run([sys.executable, str(R_DIR / "actions_backfill.py")], capture_output=True, text=True, timeout=600)
                if bf.returncode != 0:
                    print("actions backfill failed:", bf.stderr[-300:], flush=True)
                plan(a.cap)
                if time.time() - last_export > 3600:
                    export(); last_export = time.time()
            except Exception as e:  # noqa: BLE001
                print("loop error:", type(e).__name__, str(e)[:400], flush=True)
            sys.stdout.flush()
            time.sleep(300)


if __name__ == "__main__":
    main()
