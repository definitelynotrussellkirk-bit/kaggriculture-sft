"""Round-robin ladder: primitive-BC checkpoints against each other and against scripted league actors.

Every pair plays both seats on the same fixed worlds (paired comparison). Players:
  <checkpoint path>      played in-process by the agent module of --code (e.g. the frozen v1 build);
  archive:<tar.gz>       a Kaggle archive, its own main.py in a cold process (temperature, seed and weight storage
                         are the ones its MANIFEST declares, i.e. exactly as shipped);
  ship:<checkpoint>      built into a Kaggle archive by --code's build_archive.py (its defaults, or the declared
                         settings of --ship-like), then played as that archive;
  tape:<tape id>         a mined static tape's exact submitted source actions (TAPE-PANEL.json.gz, tape_panel.py);
  league names           PASS, SHOP_ROUTER, K0006, ...
Scores: win = 1, draw = 0.5 per game; Bradley-Terry ratings over all games.
  nice -n 19 work/venv/python -B work/runs/maintainer/pbc-ladder/ladder.py --code work/runs/maintainer/primitive-bc-v1-code \
    --players ckptA.pt ckptB.pt PASS SHOP_ROUTER K0006 --worlds 2 --jobs 6 --out DIR

2026-09-24 fixes (adversarial rounds 2-3). An invocation with only the old flags plays the same specs as before
(worlds base+w, sampler seeds 7100+10w, --temperature, outcome by cash); the only default change is the preload:
* R2E-F3 preload: a checkpoint player's weights are loaded before step 0 (``--no-preload``: lazy load inside step 0,
  the old behaviour). The job-731 replica agent set its 0.8 s deadline before a synchronous load, so step 0 was cut
  to PASS/STOP whenever the load took longer; the outcome then depended on NFS timing. Decision-neutral for v1 code
  (no deadline) and for v3.3+ agents (they restart the budget after a synchronous load). load_ms and deadline_hits
  per seat are in games.jsonl (seat_stats); SUMMARY.json counts games with deadline hits.
* R2E-F1 as shipped: archive: players record their MANIFEST (declared temperature, seed, budget, background load,
  weight storage; checked against main.py's settings). ``ship:`` / ``--as-shipped`` build the archive that would
  ship. A checkpoint player whose --temperature differs from --code's builder default is flagged
  (player_meta[..].temperature_vs_ship = MISMATCH) and warned on stderr.
* R2E-F7 hosted: ``--hosted`` passes remainingOverageTime, charges max(0, factor*seconds - actTimeout) per call
  to a 60 s bank (``--bank``, ``--act-timeout``, ``--hosted-factor``), kills an archive call that runs past
  (actTimeout + bank)/factor + 5 s, and ends the game on the first TIMEOUT (bank < 0 or killed) or ERROR (the
  archive raised; an in-process actor raised); the failing seat loses (both: draw). Without --hosted a hard error
  is still counted (seat_stats.hard_errors) and the game continues on cash as before.
* R2E-F6 audit: SUMMARY.json records the code directory and the sha256 of every source module the games imported
  (plus the code directory's .json data files), world and sampler seeds, temperature, preload/hosted settings and
  each player's sha256; each games.jsonl line has its sampler seed, temperature, statuses and a sources digest.
* R2E-F8: Bradley-Terry is the MAP estimate with one virtual draw per player against a fixed anchor rated 0
  (MM algorithm, no clip); the old +-8 clip set PASS equal to archive/current although PASS won their pair 3-1.
* R2E-F2 support: ``--world-seeds`` (explicit, e.g. held-out worlds), ``--focus P`` (only pairs with P); the
  powered upload decision is gate.py.
* R3-OL-2: ``tape:`` players (see above).
Repairs after the 2026-09-24 verification (same CLI; games of the old flags unchanged):
* cold-host transport: hosted archive calls speak the JSON-lines protocol on the raw pipe descriptors (cold_call),
  not through ColdArchiveAgent's private _read/_send, whose signatures and pipe modes differ between builders (the
  v1-deploy rewrite made every hosted archive game an ERROR at step 0);
* a builder that requires --temperature (live primitive-bc) makes the shipped temperature UNKNOWN: checkpoint
  players are flagged temperature_vs_ship = UNKNOWN with a warning, and ship: players are refused unless
  --ship-like or --ship-args states a temperature (a stated one also feeds the MISMATCH check);
* the ship: cache reuses an archive only while every bundled source member still matches its tree
  (stale_members); a stale archive is moved aside and rebuilt; player_meta[..].ship_cache records the check.
Second repair (verification 2, 2026-09-24; same CLI, old keys unchanged, new keys added):
* the archive's own ``[pbc] step=`` stderr log is parsed by the ladder (ladder_step_log: fb, loading, dh, cut,
  load_ms), not only by --code's parse_step_log, whose v1 generation has no ``loading`` field: a v3 archive's
  PASS-while-loading steps were None (read as 0) under --code primitive-bc-v1-deploy. Cold seats get
  seat_stats.cold, .calls and .step_log; seat_flags() marks a cold seat whose loading was not measured
  ``unmeasured``; archive player_meta records host_builder_sha256 / host_matches_builder (MANIFEST builder_sha256
  vs --code/build_archive.py);
* R2E-F6 hashes are the bytes the import read (an import hook installed at the start of play), not the file at
  game end; games.jsonl sources_changed lists modules or --code .json files edited during the game, and
  SUMMARY.code.sources_conflicts lists paths whose imported bytes differed between games.
"""
import argparse
import ast
import copy
import gzip
import hashlib
import inspect
import itertools
import json
import math
import multiprocessing as mp
import os
import re
import select
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
LEAGUE_DIR = ROOT / "work/runs/reviewer-play/opponent-league-adapters"
TAPE_PANEL = HERE / "TAPE-PANEL.json.gz"
SCHEMA = "pbc-ladder-v2"
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}
START_CASH = 3000.0
HOSTED_DEFAULTS = {"bank_s": 60.0, "act_timeout_s": 1.0, "factor": 1.0, "slack_s": 5.0}


def _setup(code):
    for p in (ROOT / "src", Path(code).resolve(), LEAGUE_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def is_archive(name):
    return name.startswith("archive:")


def is_ship(name):
    return name.startswith("ship:")


def is_tape(name):
    return name.startswith("tape:")


def is_ckpt(name):
    return (not is_archive(name) and not is_ship(name) and not is_tape(name)
            and (name.endswith(".pt") or os.path.sep in name))


def is_learned(name):
    return is_ckpt(name) or is_archive(name) or is_ship(name)


def label(name):
    if is_archive(name):
        return "archive/" + Path(name[len("archive:"):]).name.replace(".tar.gz", "")
    if is_ship(name):
        p = Path(name[len("ship:"):])
        return f"ship/{p.parent.name}/{p.stem}"
    if is_tape(name):
        return "tape/" + name[len("tape:"):]
    if not is_ckpt(name):
        return name
    p = Path(name)
    return f"{p.parent.name}/{p.stem}"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


LADDER_SHA256 = sha256_file(__file__)          # this file as imported (not as found at the end of a run)


def _generic_extract(archive_path, destination):
    """Any Kaggle tar.gz: regular members with safe relative names only."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isdir():
                continue
            if not m.isfile() or m.name.startswith("/") or ".." in Path(m.name).parts:
                raise ValueError(f"unsafe archive member {m.name}")
            target = destination / m.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(m).read())
    return {}


# ----------------------------------------------------------------------------- tapes (R3-OL-2)
_TAPES = {}


def load_tape(tape_id, panel=None):
    panel = str(panel or TAPE_PANEL)
    if panel not in _TAPES:
        with gzip.open(panel, "rt") as handle:
            data = json.load(handle)
        _TAPES[panel] = {t["id"]: t for t in data["tapes"]}
    tapes = _TAPES[panel]
    if tape_id not in tapes:
        raise KeyError(f"tape {tape_id} is not in {panel} ({len(tapes)} tapes)")
    return tapes[tape_id]


class TapeActor:
    """A mined static tape as an opponent: the exact submitted source actions, indexed by the observation's step.
    A null source action or a step past the tape plays PASS and is counted (none_steps / beyond_end)."""

    def __init__(self, tape, *, seat):
        self.tape_id = tape["id"]
        self.actions = tape["actions"]
        self.seat = int(seat)
        self.calls = self.none_steps = self.beyond_end = 0
        self.fallbacks = 0

    def __call__(self, observation, configuration=None):
        self.calls += 1
        step = observation.get("step") if hasattr(observation, "get") else None
        if type(step) is not int or not 0 <= step < len(self.actions):
            self.beyond_end += 1
            return copy.deepcopy(PASS_ACTION)
        action = self.actions[step]
        if action is None:
            self.none_steps += 1
            return copy.deepcopy(PASS_ACTION)
        return copy.deepcopy(action)


# ----------------------------------------------------------------------------- actors
def _actor(name, seat, sampler_seed, league, temperature, opts=None):
    opts = opts or {}
    if is_archive(name):
        import build_archive
        build_archive.extract = _generic_extract
        return build_archive.ColdArchiveAgent(name[len("archive:"):], threads=1, timeout=120.0)
    if is_tape(name):
        return TapeActor(load_tape(name[len("tape:"):], opts.get("tape_panel")), seat=seat)
    if is_ckpt(name):
        import agent as agent_module
        kw = {}
        if opts.get("budget_s") is not None and "budget_s" in inspect.signature(agent_module.make_agent).parameters:
            kw["budget_s"] = float(opts["budget_s"])
        teacher = (opts.get("team_id"), opts.get("strength_decile"))
        if any(value is not None for value in teacher):
            params = inspect.signature(agent_module.make_agent).parameters
            if not {"team_id", "strength_decile"} <= set(params) and not any(
                    p.kind == p.VAR_KEYWORD for p in params.values()):
                raise ValueError("checkpoint agent cannot accept an explicit teacher token")
            kw.update(team_id=int(teacher[0]), strength_decile=int(teacher[1]))
        return agent_module.make_agent(name, seed=sampler_seed, temperature=temperature, threads=1, log=False, **kw)
    return league.new_actor(name, seat=seat, sampler_seed=sampler_seed)


def preload(actor):
    """R2E-F3: finish a checkpoint agent's weight load before step 0. -> {"load_ms", "error"}.
    Works with the v1 agent (_ensure_loaded()), the v3.1 agent (_ensure_loaded(deadline)) and the v3.3+ agent
    (_ensure_loaded(deadline, wait_s); a background loader thread is awaited). A failed load is left to the agent's
    own contract (it plays PASS and counts fallbacks)."""
    begun = time.perf_counter()
    out = {"load_ms": None, "error": None}
    try:
        thread, ready = getattr(actor, "_thread", None), getattr(actor, "_ready", None)
        ensure = getattr(actor, "_ensure_loaded", None)
        if thread is not None and ready is not None:
            ready.wait()
        elif callable(ensure):
            params = [p for p in inspect.signature(ensure).parameters.values()
                      if p.default is inspect.Parameter.empty]
            ensure() if not params else ensure(time.perf_counter() + 1e9)
        else:
            return None
    except Exception as error:  # noqa: BLE001 - the agent's own PASS fallback applies from step 0
        out["error"] = f"{type(error).__name__}: {error}"[:300]
    out["load_ms"] = round((time.perf_counter() - begun) * 1000.0, 1)
    return out


class StepTimeout(RuntimeError):
    pass


def _is_cold(actor):
    return hasattr(actor, "proc") and hasattr(actor, "child_ms") and hasattr(actor, "hard_errors")


# The cold-host transport is the ladder's own (repair 2026-09-24): build_archive.ColdArchiveAgent's private I/O changed
# under the ladder (v1-deploy 56719f050221: binary non-blocking pipes, _send/_read(deadline); older builders: text
# pipes, _read()), and a ladder that called cold._read() raised TypeError at step 0 of every hosted game. Only the
# JSON-lines protocol is shared by every builder, so the ladder speaks it on the raw pipe descriptors itself: it
# depends on ``proc`` (a Popen with stdin/stdout pipes) and the counters (child_ms, hard_errors, errors, fallbacks,
# optional counts), not on any builder's read/write methods or pipe mode.
_REPLY_KEYS = ("ms", "error", "fallback", "action")


def _pipe_io(fd, deadline, write=None):
    """One select-guarded os.read (write is None) or a complete os.write of ``write`` before ``deadline``
    (time.monotonic). The descriptor is non-blocking for the call and restored afterwards."""
    blocking = os.get_blocking(fd)
    os.set_blocking(fd, False)
    try:
        view = memoryview(write) if write is not None else None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StepTimeout("no reply before the wall limit")
            r, w, _ = select.select([] if view is not None else [fd], [fd] if view is not None else [], [], remaining)
            if not r and not w:
                raise StepTimeout("no reply before the wall limit")
            try:
                if view is None:
                    return os.read(fd, 1 << 16)
                view = view[os.write(fd, view):]
            except BlockingIOError:
                continue
            if not view:
                return None
    finally:
        os.set_blocking(fd, blocking)


def _cold_exited(cold):
    try:
        cold._stderr.flush()
        tail = Path(cold.stderr_path).read_text(errors="replace")[-3000:]
    except Exception:  # noqa: BLE001
        tail = ""
    return RuntimeError(f"cold host exited (code {cold.proc.poll()}): {tail}")


def cold_request(cold, request, deadline):
    """Send one JSON request to a cold host and return its JSON reply (a dict), before ``deadline``."""
    try:
        _pipe_io(cold.proc.stdin.fileno(), deadline, write=(json.dumps(request) + "\n").encode("utf-8"))
    except (BrokenPipeError, ConnectionResetError):
        raise _cold_exited(cold) from None
    buf = bytearray(getattr(cold, "_read_buffer", b"") or b"")      # bytes a builder's own reader left behind
    if hasattr(cold, "_read_buffer"):
        cold._read_buffer = bytearray()
    buf += getattr(cold, "_ladder_buffer", b"")
    while b"\n" not in buf:
        chunk = _pipe_io(cold.proc.stdout.fileno(), deadline)
        if not chunk:
            raise _cold_exited(cold)
        buf += chunk
    line, _, rest = bytes(buf).partition(b"\n")
    cold._ladder_buffer = bytearray(rest)
    reply = json.loads(line)
    if not isinstance(reply, dict):
        raise RuntimeError(f"cold host protocol reply is not an object: {line[:200]!r}")
    if isinstance(reply.get("counts"), dict) and hasattr(cold, "counts"):
        cold.counts = reply["counts"]                                  # live builder: in-archive agent counters
    return reply


def _kill(cold):
    try:
        cold.proc.kill()
        cold.proc.wait(timeout=5.0)
    except Exception:  # noqa: BLE001
        pass


def cold_call(cold, observation, configuration, timeout_s):
    """One Kaggle agent call on a ColdArchiveAgent with a wall limit (R2E-F7), for every builder's ColdArchiveAgent.
    -> (action, hard_error or None). The child is killed and StepTimeout raised when no reply arrives within
    timeout_s; a host that exits raises RuntimeError (the caller scores ERROR)."""
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    try:
        reply = cold_request(cold, {"observation": observation, "configuration": dict(configuration or {})},
                             deadline)
    except StepTimeout as error:
        _kill(cold)
        raise StepTimeout(f"no reply within {timeout_s:.1f} s") from error
    except Exception:
        _kill(cold)          # a half-read reply cannot be paired with the next request
        raise
    missing = [k for k in _REPLY_KEYS if k not in reply]
    if missing:
        _kill(cold)
        raise RuntimeError(f"cold host reply lacks {missing}: {str(reply)[:200]}")
    cold.child_ms.append(reply["ms"])
    if reply["error"] is not None:
        cold.hard_errors += 1
        cold.errors.append("HARD " + str(reply["error"])[:300])
        return copy.deepcopy(PASS_ACTION), reply["error"]
    if reply["fallback"]:
        cold.fallbacks += 1
        if len(cold.errors) < 20:
            cold.errors.append(str(reply.get("fallback_error"))[:300])
    return reply["action"], None


def _pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)], 1)


_IMPORT_HASHES = {}      # resolved .py path -> sha256[:16] of the bytes its import read (R2E-F6 repair)
_HASH_ROOTS = []


def _hash_roots(code):
    return [Path(code).resolve(), (ROOT / "src").resolve(), LEAGUE_DIR.resolve(), HERE]


def _under(p, roots):
    return any(r == p.parent or r in p.parents for r in roots)


def _install_import_hasher(code):
    """Record, for every module under the hash roots imported from now on, the sha256 of the source bytes the
    import compiled (FileLoader.get_data), so a file edited after its import is recorded as imported (verification
    2, defect 3). A module loaded from a valid bytecode cache records its source as found when the cache was read
    (the cache was validated against that source's mtime and size). Modules already imported are hashed now."""
    for r in _hash_roots(code):
        if r not in _HASH_ROOTS:
            _HASH_ROOTS.append(r)
    import importlib._bootstrap_external as bootstrap_external
    loader = bootstrap_external.FileLoader
    if not getattr(loader.get_data, "_ladder_import_hasher", False):
        original = loader.get_data

        def get_data(self, path):
            data = original(self, path)
            try:
                source = getattr(self, "path", None)
                if isinstance(source, str) and source.endswith(".py"):
                    p = Path(source).resolve()
                    if _under(p, _HASH_ROOTS):
                        if os.fspath(path) == source:          # the source bytes this module is compiled from
                            _IMPORT_HASHES[str(p)] = hashlib.sha256(data).hexdigest()[:16]
                        elif str(p) not in _IMPORT_HASHES:     # bytecode cache read first (overwritten if stale)
                            _IMPORT_HASHES[str(p)] = sha256_file(p)[:16]
            except Exception:  # noqa: BLE001 - auditing must never break an import
                pass
            return data

        get_data._ladder_import_hasher = True
        loader.get_data = get_data
    me = Path(__file__).resolve()
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if not f or not str(f).endswith(".py"):
            continue
        p = Path(f).resolve()
        if str(p) not in _IMPORT_HASHES and _under(p, _HASH_ROOTS) and p.is_file():
            _IMPORT_HASHES[str(p)] = LADDER_SHA256[:16] if p == me else sha256_file(p)[:16]


def _data_hashes(code):
    """--code's top-level .json data files (read by agents at construction/load)."""
    return {str(p): sha256_file(p)[:16] for p in sorted(Path(code).resolve().glob("*.json"))}


def _source_hashes(code, data_at_start=None):
    """-> ({path: sha256[:16]}, [changed paths]). Every imported .py under --code, src/, the league adapters and
    this directory, as its import read it (_install_import_hasher; a module the hook never saw is hashed now), plus
    --code's top-level .json data files as found before the actors were built (``data_at_start``). ``changed``
    lists the paths whose file differs now (edited during the game) (R2E-F6, verification 2 defect 3)."""
    roots = _hash_roots(code)
    out, changed = {}, []
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if not f or not str(f).endswith(".py"):
            continue
        p = Path(f).resolve()
        if _under(p, roots) and p.is_file():
            now = sha256_file(p)[:16]
            out[str(p)] = _IMPORT_HASHES.get(str(p), now)
            if out[str(p)] != now:
                changed.append(str(p))
    now_data = _data_hashes(code)
    start = now_data if data_at_start is None else data_at_start
    for path in sorted(set(start) | set(now_data)):
        out[path] = start.get(path, now_data.get(path))
        if start.get(path) != now_data.get(path):
            changed.append(path)
    return out, sorted(set(changed))


# ----------------------------------------------------------------------------- step log and seat flags
def ladder_step_log(text):
    """The archive's own ``[pbc] step=`` stderr lines, parsed by the ladder (verification 2, defect 1): every builder
    generation writes ``step= ms= fb=`` and, where it has them, ``dh= cut=1 loading=1 load_ms= lerr=``; --code's
    parse_step_log knows only its own generation's fields (the v1 one has no ``loading``)."""
    lines = [l for l in text.splitlines() if l.startswith("[pbc] step=")]
    fields = [dict(kv.split("=", 1) for kv in l[6:].split() if "=" in kv) for l in lines]

    def first_float(key):
        for f in fields:
            if key in f:
                try:
                    return float(f[key])
                except ValueError:
                    return None
        return None

    return {"parser": "ladder", "lines": len(lines),
            "fallback_lines": sum(f.get("fb") == "1" for f in fields),
            "loading_lines": sum(f.get("loading") == "1" for f in fields),
            "deadline_hit_lines": sum(f.get("dh") == "1" for f in fields),
            "cut_lines": sum(f.get("cut") == "1" for f in fields),
            "load_ms": first_float("load_ms"), "wait_ms": first_float("wait_ms"),
            "load_error": next((f["lerr"] for f in fields if "lerr" in f), None)}


# actor attributes copied into seat_stats when present (cut_steps / model_steps: ColdArchiveAgent and the live
# agent since the serving fixes of 2026-09-24, R3-HA-6; integration pass)
SEAT_COUNTERS = ("fallbacks", "hard_errors", "deadline_hits", "pass_while_loading", "load_ms", "load_error",
                 "none_steps", "beyond_end", "cut_steps", "model_steps")


def _is_archive_label(name):
    return str(name).startswith(("archive/", "ship/"))


def seat_flags(g, i):
    """Runtime-failure flags of seat i of one games.jsonl record (shared by failure_counts and gate.py):
    timeout, error, hard_error, fallback, pass_while_loading, deadline_hit, and ``unmeasured``: a cold archive
    seat whose PASS-while-loading count is unknown (no in-archive counter, no step-log ``loading`` field that
    covers every call, or no seat_stats at all), which must not read as 0 (verification 2, defect 1)."""
    stats = g.get("seat_stats")
    st = (stats or [{}, {}])[i] or {}
    status = (g.get("status") or [st.get("status") or "DONE"] * 2)[i]
    step_log = st.get("step_log") if isinstance(st.get("step_log"), dict) else {}
    fb = (g.get("fallbacks") or [None, None])[i]
    flags = {
        "timeout": status == "TIMEOUT",
        "error": status == "ERROR",
        "hard_error": (st.get("hard_errors") or 0) > 0,
        "fallback": ((fb or 0) > 0 or (st.get("log_fallback_lines") or 0) > 0
                     or (step_log.get("fallback_lines") or 0) > 0),
        "pass_while_loading": ((st.get("pass_while_loading") or 0) > 0 or (st.get("log_loading_lines") or 0) > 0
                               or (step_log.get("loading_lines") or 0) > 0),
        "deadline_hit": (st.get("deadline_hits") or 0) > 0 or (step_log.get("deadline_hit_lines") or 0) > 0,
    }
    cold = bool(st.get("cold")) or "log_lines" in st or (stats is None and _is_archive_label(g["seats"][i]))
    calls = st.get("calls")
    log_covers = step_log.get("lines") is not None and (calls is None or step_log["lines"] >= calls)
    measured = (st.get("pass_while_loading") is not None or log_covers
                or (not step_log and st.get("log_loading_lines") is not None
                    and (calls is None or (st.get("log_lines") or 0) >= calls)))
    flags["unmeasured"] = cold and not measured
    return flags


def _digest(mapping):
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()[:16]


def seat0_score(g):
    """Seat 0's game score: the recorded score (hosted failures) if present, else cash (win 1, draw 0.5)."""
    if g.get("score") is not None:
        return float(g["score"][0])
    return 1.0 if g["cash"][0] > g["cash"][1] else 0.5 if g["cash"][0] == g["cash"][1] else 0.0


def play(spec):
    _setup(spec["code"])
    _install_import_hasher(spec["code"])
    data_at_start = _data_hashes(spec["code"])
    import torch
    torch.set_num_threads(1)
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    from league import League
    league = League()
    names = spec["seats"]
    labels = spec.get("labels") or [label(n) for n in names]
    opts = spec.get("opts") or {}
    hosted = spec.get("hosted")
    actors = [_actor(names[s], s, spec["sampler_seed"] + s, league, spec["temperature"], opts) for s in (0, 1)]
    loads = [preload(actors[s]) if spec.get("preload") and is_ckpt(names[s]) else None for s in (0, 1)]
    env = ExactKaggricultureEnvironment(seed=spec["world_seed"])
    t0 = time.perf_counter()
    ms = [[], []]
    status = ["DONE", "DONE"]
    errors = [[], []]
    bank = [float(hosted["bank_s"]) if hosted else None] * 2
    overage = [0.0, 0.0]
    failed_step = None
    while not env.done and env.step_index < spec["steps"]:
        acts = []
        for s in (0, 1):
            obs = env.agent_observation(s)
            if not hosted:
                b = time.perf_counter()
                acts.append(actors[s](obs, env.configuration))
                ms[s].append((time.perf_counter() - b) * 1000)
                continue
            obs["remainingOverageTime"] = round(max(0.0, bank[s]), 3)
            fail, action = None, None
            b = time.perf_counter()
            try:
                if _is_cold(actors[s]):
                    limit = (hosted["act_timeout_s"] + max(0.0, bank[s])) / hosted["factor"] + hosted["slack_s"]
                    action, hard = cold_call(actors[s], obs, env.configuration, limit)
                    if hard is not None:
                        fail = ("ERROR", hard)
                else:
                    action = actors[s](obs, env.configuration)
            except StepTimeout as error:
                fail = ("TIMEOUT", str(error))
            except Exception as error:  # noqa: BLE001 - an agent exception is a hard failure on Kaggle
                fail = ("ERROR", f"{type(error).__name__}: {error}")
            dt = time.perf_counter() - b
            ms[s].append(dt * 1000)
            over = max(0.0, hosted["factor"] * dt - hosted["act_timeout_s"])
            overage[s] += over
            bank[s] -= over
            if fail is None and bank[s] < 0:
                fail = ("TIMEOUT", "overage bank exhausted")
            if fail is not None:
                status[s] = fail[0]
                errors[s].append(str(fail[1])[:300])
            acts.append(action)
        if hosted and any(x != "DONE" for x in status):
            failed_step = env.step_index
            break
        env.step(acts)
    farms = env.state[0].observation.farms
    cash = [float(farms[s]["money"]) for s in (0, 1)]
    ok = [x == "DONE" for x in status]
    if all(ok):
        score = [1.0, 0.0] if cash[0] > cash[1] else [0.5, 0.5] if cash[0] == cash[1] else [0.0, 1.0]
    else:
        score = [1.0, 0.0] if ok[0] else [0.0, 1.0] if ok[1] else [0.5, 0.5]
    stats = []
    for s, a in enumerate(actors):
        st = {"status": status[s]}
        stderr = None
        close = getattr(a, "close", None)
        if close:
            try:
                stderr = close()
            except Exception as error:  # noqa: BLE001
                st["close_error"] = f"{type(error).__name__}: {error}"[:300]
        for key in SEAT_COUNTERS:
            value = getattr(a, key, None)
            if value is not None:
                st[key] = round(value, 1) if isinstance(value, float) else value
        if loads[s] is not None:
            st["preload"] = loads[s]
        if _is_cold(a):
            st["cold"] = True
            st["calls"] = len(ms[s])
            st["step_log"] = ladder_step_log(stderr) if isinstance(stderr, str) else None
        if _is_cold(a) and isinstance(stderr, str):
            parse = getattr(sys.modules.get("build_archive"), "parse_step_log", None)
            if callable(parse):
                log = parse(stderr)
                st.update({"log_lines": log.get("lines"), "log_fallback_lines": log.get("fallback_lines"),
                           "log_loading_lines": log.get("loading_lines"), "log_load_ms": log.get("load_ms")})
        st.update({"ms_first": round(ms[s][0], 1) if ms[s] else None, "ms_p99": _pct(ms[s], 0.99),
                   "ms_max": round(max(ms[s]), 1) if ms[s] else None})
        if hosted:
            st.update({"bank_left_s": round(bank[s], 3), "overage_s": round(overage[s], 3)})
        if errors[s]:
            st["errors"] = errors[s][:5]
        stats.append(st)
    sources, changed = _source_hashes(spec["code"], data_at_start)
    return {"seats": labels, "world_seed": spec["world_seed"], "cash": cash,
            "steps": env.step_index, "fallbacks": [st.get("fallbacks") for st in stats],
            "seconds": round(time.perf_counter() - t0, 1),
            "ms_p50": [sorted(m)[len(m) // 2] if m else None for m in ms],
            "score": score, "status": status, "failed_step": failed_step, "sampler_seed": spec["sampler_seed"],
            "temperature": spec["temperature"], "hosted": hosted, "seat_stats": stats,
            "sources": sources, "sources_changed": changed}


# ----------------------------------------------------------------------------- ratings / summary
def bradley_terry(games, players, iters=5000, prior_draws=1.0, tol=1e-10):
    """MAP Bradley-Terry ratings (natural-log scale) relative to a fixed anchor rated 0: every player gets
    ``prior_draws`` virtual draws against the anchor (R2E-F8), so all-win / all-loss records stay finite and no
    clip decides the output. MM iterations (Hunter 2004); a draw counts half a win for each side."""
    players = list(dict.fromkeys(players))
    wins = {p: 0.5 * prior_draws for p in players}
    pairs = {}
    for g in games:
        a, b = g["seats"]
        if a not in wins or b not in wins or a == b:
            continue
        sa = seat0_score(g)
        wins[a] += sa
        wins[b] += 1.0 - sa
        key = (a, b) if a < b else (b, a)
        pairs[key] = pairs.get(key, 0) + 1
    gamma = {p: 1.0 for p in players}
    for _ in range(iters):
        new = {}
        for p in players:
            den = prior_draws / (gamma[p] + 1.0)
            for (x, y), n in pairs.items():
                if p == x:
                    den += n / (gamma[x] + gamma[y])
                elif p == y:
                    den += n / (gamma[x] + gamma[y])
            new[p] = wins[p] / den
        delta = max(abs(math.log(new[p]) - math.log(gamma[p])) for p in players) if players else 0.0
        gamma = new
        if delta < tol:
            break
    return {p: math.log(gamma[p]) for p in players}


def summarize(games, players):
    table = {}
    for g in games:
        a, b = g["seats"]
        s0 = seat0_score(g)
        for x, y, i in ((a, b, 0), (b, a, 1)):
            t = table.setdefault(x, {}).setdefault(y, {"W": 0, "D": 0, "L": 0, "margin": 0.0, "n": 0})
            own, other = g["cash"][i], g["cash"][1 - i]
            score = s0 if i == 0 else 1.0 - s0
            t["W" if score == 1.0 else "D" if score == 0.5 else "L"] += 1
            t["margin"] += own - other
            t["n"] += 1
    for x in table.values():
        for t in x.values():
            t["score"] = (t["W"] + 0.5 * t["D"]) / t["n"]
            t["mean_margin"] = t["margin"] / t["n"]
    return {"pairs": table, "bradley_terry": bradley_terry(games, [label(p) for p in players]),
            "bradley_terry_method": "MAP, 1 virtual draw per player vs a 0-rated anchor, no clip (R2E-F8)"}


def failure_counts(games):
    """Per player label: games with a hosted failure, a hard error, a fallback or a deadline hit (R2E-F3/F7)."""
    out = {}
    for g in games:
        for i, who in enumerate(g["seats"]):
            c = out.setdefault(who, {"games": 0, "timeout": 0, "error": 0, "hard_error": 0, "fallback": 0,
                                     "deadline_hit": 0, "pass_while_loading": 0, "unmeasured": 0})
            c["games"] += 1
            for key, flag in seat_flags(g, i).items():
                c[key] += int(flag)
    return out


# ----------------------------------------------------------------------------- player metadata / shipping
def archive_meta(path):
    """MANIFEST.json and main.py settings of a Kaggle archive (R2E-F1)."""
    meta = {"kind": "archive", "path": str(path), "sha256": sha256_file(path), "bytes": Path(path).stat().st_size}
    with tarfile.open(path, "r:gz") as tar:
        names = set(tar.getnames())
        manifest = json.loads(tar.extractfile("MANIFEST.json").read()) if "MANIFEST.json" in names else None
        main_src = tar.extractfile("main.py").read().decode() if "main.py" in names else None
    if manifest is not None:
        meta.update({"schema": manifest.get("schema"), "declared": manifest.get("declared"),
                     "weights_storage": manifest.get("weights_storage"), "model_spec": manifest.get("model_spec"),
                     "model_sha256": manifest.get("model_sha256"), "builder_sha256": manifest.get("builder_sha256")})
    settings = None
    if main_src:
        m = re.search(r"_PBC_SETTINGS\s*=\s*(\{.*?\})\s*\n", main_src, re.S)
        if m:
            try:
                settings = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                settings = None
    meta["main_settings"] = settings
    declared = (meta.get("declared") or {})
    if settings is not None and "temperature" in declared:
        meta["manifest_matches_main"] = (float(settings.get("temperature")) == float(declared["temperature"])
                                         and int(settings.get("seed", -1)) == int(declared.get("seed", -2)))
    return meta


def host_check(meta, code):
    """The cold host (--code's build_archive.ColdArchiveAgent / parse_step_log) vs the builder that made the archive
    (MANIFEST builder_sha256). -> {"host_builder_sha256", "host_matches_builder": True/False/None (no MANIFEST)}."""
    builder = Path(code) / "build_archive.py"
    host = sha256_file(builder) if builder.is_file() else None
    made_by = (meta or {}).get("builder_sha256")
    return {"host_builder_sha256": host,
            "host_matches_builder": None if not made_by or not host else made_by == host}


_UNSET = object()


def _cli_flag_spec(builder_source, flag):
    """{"required": bool, "default": value or _UNSET} of ``add_argument(flag, ...)`` in a builder's source (the
    ``build`` sub-command's flags; the first call naming ``flag``)."""
    try:
        tree = ast.parse(builder_source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == flag):
            spec = {"required": False, "default": _UNSET}
            for kw in node.keywords:
                if kw.arg == "required" and isinstance(kw.value, ast.Constant):
                    spec["required"] = bool(kw.value.value)
                if kw.arg == "default":
                    try:
                        spec["default"] = ast.literal_eval(kw.value)
                    except ValueError:
                        pass
            return spec
    return None


def builder_defaults(code):
    """What a checkpoint ships as unless told otherwise: --code's ``build_archive.py build`` CLI defaults (the path
    ship: uses), else build()'s defaults. ``temperature`` is None and ``temperature_required`` True when the CLI
    demands --temperature (live builder since 2026-09-24 01:24): the shipped temperature is then UNKNOWN until a
    build command states it (repair of the R2E-F1 guard)."""
    _setup(code)
    import importlib.util
    out = {}
    try:                                   # by path: a build_archive of another --code may already be imported
        spec = importlib.util.spec_from_file_location("_ladder_builder_defaults", Path(code) / "build_archive.py")
        ba = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ba)
        params = inspect.signature(ba.build).parameters
        out = {k: params[k].default for k in ("temperature", "storage", "seed", "threads", "budget_s", "background")
               if k in params}
    except Exception:  # noqa: BLE001
        out = {}
    try:
        source = (Path(code) / "build_archive.py").read_text()
    except OSError:
        source = ""
    for key, flag in (("temperature", "--temperature"), ("storage", "--storage"), ("seed", "--seed"),
                      ("threads", "--threads")):
        spec = _cli_flag_spec(source, flag)
        if spec is None:
            continue
        if spec["required"]:
            out[key] = None
            out[f"{key}_required"] = True
        elif spec["default"] is not _UNSET:
            out[key] = spec["default"]
    if source and out.get("temperature") is None:
        out["temperature_required"] = True
    return out


def _arg_value(args, flag):
    """The last value given to ``flag`` in a build argument list ("--flag V" or "--flag=V"), else None."""
    value = None
    args = list(args)
    for i, x in enumerate(args):
        if x == flag and i + 1 < len(args):
            value = args[i + 1]
        elif x.startswith(flag + "="):
            value = x[len(flag) + 1:]
    return value


def ship_temperature(defaults, args):
    """(temperature a ship: build of these arguments declares, its source). (None, "UNKNOWN ...") when the builder
    has no default and the arguments name none."""
    given = _arg_value(args or [], "--temperature")
    if given is not None:
        return float(given), "ship arguments"
    if defaults.get("temperature_required"):
        return None, "UNKNOWN: the builder has no default temperature (--temperature is required)"
    if defaults.get("temperature") is None:
        return None, "UNKNOWN: no build_archive.py default temperature found"
    return float(defaults["temperature"]), "builder default"


def ship_args_from(ship_like, code):
    """build_archive.py build arguments reproducing the declared settings of an archive / MANIFEST.json."""
    p = Path(ship_like)
    if p.name.endswith(".tar.gz"):
        with tarfile.open(p, "r:gz") as tar:
            manifest = json.loads(tar.extractfile("MANIFEST.json").read())
    else:
        manifest = json.loads(p.read_text())
    declared = manifest.get("declared") or {}
    builder = (Path(code) / "build_archive.py").read_text()
    args = []
    if "temperature" in declared:
        args += ["--temperature", repr(float(declared["temperature"]))]
    if "seed" in declared:
        args += ["--seed", str(int(declared["seed"]))]
    if "threads" in declared:
        args += ["--threads", str(int(declared["threads"]))]
    if manifest.get("weights_storage") in ("fp32", "fp16", "int8"):
        args += ["--storage", manifest["weights_storage"]]
    if "budget_s" in declared:
        if '"--budget"' not in builder:
            raise ValueError(f"{ship_like} declares budget_s but {code}/build_archive.py has no --budget")
        args += ["--budget", repr(float(declared["budget_s"]))]
    if declared.get("background_load") is False:
        if '"--sync-load"' not in builder:
            raise ValueError(f"{ship_like} declares a synchronous load but the builder has no --sync-load")
        args += ["--sync-load"]
    # live builder (serving fixes 2026-09-24, R3-HA-3/-4/-5): the training replica it served, pins, int8, first wait
    tree = manifest.get("code_tree") or {}

    def need(flag, what):
        if f'"{flag}"' not in builder:
            raise ValueError(f"{ship_like} declares {what} but {code}/build_archive.py has no {flag}")

    if tree.get("root"):
        need("--code-tree", f"code tree {tree['root']}")
        args += ["--code-tree", str(tree["root"])]
        sums = tree.get("sums")
        if sums and Path(sums).resolve() != (Path(tree["root"]) / "SHA256SUMS").resolve():
            need("--code-tree-sums", f"code tree sums {sums}")
            args += ["--code-tree-sums", str(sums)]
    else:
        unpinned = (manifest.get("source_pins") or {}).get("unpinned_allowed")
        if unpinned:
            need("--allow-unpinned", f"an unpinned build ({unpinned})")
            args += ["--allow-unpinned", str(unpinned)]
    if declared.get("int8_allowed"):
        need("--allow-int8", f"int8 storage ({declared['int8_allowed']})")
        args += ["--allow-int8", str(declared["int8_allowed"])]
    for key, flag in (("first_wait_max_s", "--first-wait-max"), ("first_wait_reserve_s", "--first-wait-reserve")):
        if declared.get(key) is not None:
            need(flag, f"{key} {declared[key]}")
            args += [flag, repr(float(declared[key]))]
    return args


_ARCHIVE_NON_SOURCE = ("MANIFEST.json", "main.py")


def stale_members(archive, code, args=()):
    """Bundled source files of a built archive that no longer match the tree they were copied from (defect 4 of the
    2026-09-24 verification). Member names are relative to the builder's repository root (build_archive.py's
    ``HERE.parents[3]``, or ``--code-tree ROOT``). -> {"stale": [...], "checked": n, "unmapped": [...]}."""
    tree = _arg_value(args, "--code-tree")
    parents = Path(code).resolve().parents
    base = Path(tree) if tree else (parents[3] if len(parents) > 3 else None)
    out = {"stale": [], "checked": 0, "unmapped": []}
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile() or m.name in _ARCHIVE_NON_SOURCE or not m.name.endswith((".py", ".json")):
                continue
            src = base / m.name if base is not None else None
            if src is None or not src.is_file():
                out["unmapped"].append(m.name)
                continue
            out["checked"] += 1
            if hashlib.sha256(tar.extractfile(m).read()).hexdigest() != sha256_file(src):
                out["stale"].append(m.name)
    return out


def build_ship(model, code, cache, args, report=None):
    """Build ``model`` into a Kaggle archive with --code's build_archive.py. -> archive path. Cached by model,
    builder, agent and arguments; a cached archive is reused only while every bundled source file still matches
    its tree (stale_members), otherwise it is moved aside (``.stale-<time>``) and rebuilt. ``report`` (a dict)
    receives {"cache": "hit"/"built"/"rebuilt-stale", "stale": [...], "checked": n, "unmapped": [...]}."""
    code = Path(code).resolve()
    model = Path(model).resolve()
    key = _digest({"model": sha256_file(model), "builder": sha256_file(code / "build_archive.py"),
                   "agent": sha256_file(code / "agent.py"), "args": list(args)})[:12]
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{model.parent.name}-{model.stem}-{key}.tar.gz"
    report = report if report is not None else {}
    state = "built"
    if out.exists():
        check = stale_members(out, code, args)
        report.update(check)
        if not check["stale"]:
            report["cache"] = "hit"
            return out
        stamp = time.strftime("%Y%m%dT%H%M%S")
        for suffix in ("", ".build.json", ".log"):
            old = Path(str(out) + suffix)
            if old.exists():
                old.rename(Path(str(out) + suffix + f".stale-{stamp}"))
        state = "rebuilt-stale"
    stale = Path(str(out) + ".build.json")          # a receipt without its archive: the builder refuses to overwrite
    if stale.exists():
        stale.unlink()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    cmd = [sys.executable, "-B", str(code / "build_archive.py"), "build", "--model", str(model), "--out", str(out),
           "--no-verify", *args]
    done = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    Path(str(out) + ".log").write_text(" ".join(shlex.quote(c) for c in cmd) + "\n" + done.stdout + done.stderr)
    if done.returncode != 0 or not out.exists():
        raise RuntimeError(f"ship build failed ({done.returncode}) for {model}: {(done.stdout + done.stderr)[-2000:]}")
    report.update(stale_members(out, code, args))
    report["cache"] = state if not report["stale"] else "built-while-sources-changed"
    return out


def player_meta(p, a, defaults, seat_name):
    if is_archive(p) or is_ship(p):
        meta = archive_meta(seat_name[p][len("archive:"):])
        meta.update(host_check(meta, a.code))
        if is_ship(p):
            model = p[len("ship:"):]
            meta.update({"kind": "ship", "model": str(Path(model).resolve()),
                         "ship_args": a.ship_args_resolved, "ship_settings_from": a.ship_like or "builder defaults",
                         "ship_cache": (getattr(a, "ship_cache_reports", None) or {}).get(p)})
        return meta
    if is_tape(p):
        t = load_tape(p[len("tape:"):], a.tape_panel)
        return {"kind": "tape", **{k: t.get(k) for k in ("support", "wins", "decided", "win_rate", "first_day",
                                                          "last_day", "groups", "source", "actions_sha256")}}
    if is_ckpt(p):
        ship_t, ship_src = ship_temperature(defaults, getattr(a, "ship_args_resolved", None))
        meta = {"kind": "ckpt", "path": str(Path(p).resolve()), "sha256": sha256_file(p),
                "temperature": a.temperature, "budget_s": a.budget, "ship_temperature": ship_t,
                "ship_temperature_source": ship_src, "ship_storage": defaults.get("storage")}
        if ship_t is None:
            meta["temperature_vs_ship"] = "UNKNOWN"
            print(f"WARNING {label(p)} plays at T={a.temperature}; the temperature it ships at is UNKNOWN: "
                  f"{a.code}/build_archive.py has no default (--temperature is required). Ladder what ships with "
                  f"ship: / --as-shipped and --ship-like ARCHIVE or --ship-args '--temperature T' (R2E-F1)",
                  file=sys.stderr, flush=True)
        else:
            meta["temperature_vs_ship"] = "match" if float(ship_t) == float(a.temperature) else "MISMATCH"
            if meta["temperature_vs_ship"] == "MISMATCH":
                print(f"WARNING {label(p)} plays at T={a.temperature}, {a.code}/build_archive.py ships T={ship_t} "
                      f"({ship_src}; {defaults.get('storage')} weights); use ship: / --as-shipped to ladder what "
                      f"ships (R2E-F1)", file=sys.stderr, flush=True)
        return meta
    return {"kind": "league"}


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="directory holding agent.py for checkpoints")
    ap.add_argument("--players", nargs="+", required=True)
    ap.add_argument("--worlds", type=int, default=2)
    ap.add_argument("--world-seed-base", type=int, default=20260923)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--team-id", type=int, help="checkpoint players only: explicit team vocabulary ID")
    ap.add_argument("--strength-decile", type=int, help="checkpoint players only: teacher strength 0..9")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=719)
    ap.add_argument("--skip-scripted-pairs", action="store_true", help="do not play scripted vs scripted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--world-seeds", type=int, nargs="+", help="explicit world seeds (replace --worlds/--world-seed-base)")
    ap.add_argument("--focus", nargs="+", help="only pairs that include one of these players")
    ap.add_argument("--no-preload", action="store_true", help="checkpoint weights load inside step 0 (old behaviour)")
    ap.add_argument("--budget", type=float, help="checkpoint players' per-step budget_s (agents that take it)")
    ap.add_argument("--as-shipped", action="store_true", help="play every checkpoint player as ship:<checkpoint>")
    ap.add_argument("--ship-like", help="archive or MANIFEST.json whose declared settings ship: builds reproduce")
    ap.add_argument("--ship-args", default="", help="extra build_archive.py build arguments for ship: players")
    ap.add_argument("--ship-cache", help="directory for ship: archives (default OUT/ship)")
    ap.add_argument("--hosted", action="store_true", help="bank, per-step wall limit, hard errors end the game")
    ap.add_argument("--hosted-factor", type=float, default=HOSTED_DEFAULTS["factor"])
    ap.add_argument("--bank", type=float, default=HOSTED_DEFAULTS["bank_s"])
    ap.add_argument("--act-timeout", type=float, default=HOSTED_DEFAULTS["act_timeout_s"])
    ap.add_argument("--tape-panel", default=str(TAPE_PANEL))
    a = ap.parse_args(argv)
    if (a.team_id is None) != (a.strength_decile is None):
        ap.error("--team-id and --strength-decile must be supplied together")
    if a.team_id is not None and not (0 <= a.team_id < 1024 and 0 <= a.strength_decile <= 9):
        ap.error("teacher token requires team ID 0..1023 and strength decile 0..9")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    players = [("ship:" + p) if a.as_shipped and is_ckpt(p) else p for p in a.players]
    focus = None if not a.focus else {("ship:" + p) if a.as_shipped and is_ckpt(p) else p for p in a.focus}
    defaults = builder_defaults(a.code)
    a.ship_args_resolved = (ship_args_from(a.ship_like, a.code) if a.ship_like else []) + shlex.split(a.ship_args)
    if any(is_ship(p) for p in players) and ship_temperature(defaults, a.ship_args_resolved)[0] is None:
        ap.error(f"ship: players: {a.code}/build_archive.py has no default temperature (--temperature is "
                 f"required), so the shipped temperature is unknown; pass --ship-like ARCHIVE (its declared settings) "
                 f"or --ship-args '--temperature T'")
    seat_name = {}
    if any(is_ship(p) for p in players) and not a.ship_like:
        print(f"WARNING ship: players use {a.code}/build_archive.py defaults (temperature "
              f"{ship_temperature(defaults, a.ship_args_resolved)[0]}, {defaults.get('storage')} weights) plus "
              f"--ship-args {a.ship_args_resolved}; a deployed archive may declare other settings (the v1 20000 "
              f"archive is fp32): pass --ship-like <that archive> to reproduce it (R2E-F1)", file=sys.stderr, flush=True)
    a.ship_cache_reports = {}
    for p in players:
        if is_ship(p):
            rep = a.ship_cache_reports.setdefault(p, {})
            seat_name[p] = "archive:" + str(build_ship(p[len("ship:"):], a.code, a.ship_cache or out / "ship",
                                                        a.ship_args_resolved, report=rep))
        else:
            seat_name[p] = p
    meta = {label(p): player_meta(p, a, defaults, seat_name) for p in players}
    hosted = ({"bank_s": a.bank, "act_timeout_s": a.act_timeout, "factor": a.hosted_factor,
               "slack_s": HOSTED_DEFAULTS["slack_s"]} if a.hosted else None)
    worlds = list(a.world_seeds) if a.world_seeds else [a.world_seed_base + w for w in range(a.worlds)]
    opts = {"budget_s": a.budget, "tape_panel": a.tape_panel,
            "team_id": a.team_id, "strength_decile": a.strength_decile}
    specs = []
    for x, y in itertools.combinations(players, 2):
        if a.skip_scripted_pairs and not is_learned(x) and not is_learned(y):
            continue
        if focus is not None and x not in focus and y not in focus:
            continue
        for w, world in enumerate(worlds):
            for seats in ((x, y), (y, x)):
                specs.append({"code": a.code, "seats": [seat_name[s] for s in seats], "labels": [label(s) for s in seats],
                              "world_seed": world, "sampler_seed": 7100 + 10 * w, "temperature": a.temperature,
                              "steps": a.steps, "preload": not a.no_preload, "hosted": hosted, "opts": opts})
    games = []
    sources, conflicts, changed = {}, {}, set()
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.jobs) as pool, open(out / "games.jsonl", "w") as f:
        for g in pool.imap_unordered(play, specs):
            src = g.pop("sources", None) or {}
            for path, sha in src.items():
                if path in sources and sources[path] != sha:
                    conflicts.setdefault(path, [sources[path]])
                    if sha not in conflicts[path]:
                        conflicts[path].append(sha)
                sources.setdefault(path, sha)
            changed.update(g.get("sources_changed") or [])
            g["sources_digest"] = _digest(src)
            games.append(g)
            f.write(json.dumps(g) + "\n")
            f.flush()
            bad = "" if all(x == "DONE" for x in g["status"]) else f" status={g['status']}@{g['failed_step']}"
            print(f"{len(games)}/{len(specs)} {g['seats']} {g['cash']} {g['seconds']}s{bad}", flush=True)
    summary = summarize(games, players)
    summary.update({"players": [label(p) for p in players], "worlds": len(worlds), "games": len(games),
                    "temperature": a.temperature, "seconds": round(time.time() - t0, 1)})
    summary.update({"schema": SCHEMA, "world_seeds": worlds, "sampler_seeds": [7100 + 10 * w for w in range(len(worlds))],
                    "steps": a.steps, "preload": not a.no_preload, "budget_s": a.budget, "hosted": hosted,
                    "player_labels": {raw: label(p) for raw, p in zip(a.players, players)}, "player_meta": meta,
                    "failures": failure_counts(games),
                    "code": {"dir": str(Path(a.code).resolve()), "sources": dict(sorted(sources.items())),
                             "sources_conflicts": conflicts, "sources_changed_during_games": sorted(changed),
                             "builder_defaults": {k: v for k, v in defaults.items() if isinstance(v, (int, float, str, bool))}},
                    "ladder_sha256": LADDER_SHA256})
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["bradley_terry"], indent=1))


if __name__ == "__main__":
    main()
