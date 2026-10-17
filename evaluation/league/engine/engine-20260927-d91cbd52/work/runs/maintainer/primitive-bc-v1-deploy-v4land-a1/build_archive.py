"""Kaggle submission archive for the primitive-BC agent, plus its cold-process verifier.

``build``  model spec -> deterministic tar.gz (< 100 MB) holding
    main.py          exposes ``agent(observation, configuration)``; loads the policy lazily on
                     the first call, logs one stderr line per step
                     (``[pbc] step=<t> ms=<wall ms> fb=<0|1>``), PASS on any exception;
    policy.pt        ``primitive-bc-weights-v1`` weights (fp32 / fp16 / int8 storage, declared);
    source files     exactly the repository files a traced subprocess opened or imported while
                     loading the policy and playing ``--trace-steps`` steps in both seats, at
                     their repository-relative paths (so ``parents[3]`` source pins resolve);
    MANIFEST.json    member sha256/bytes, model identity, declared threads/seed/temperature.
``verify`` extracts the archive into an empty directory and plays complete 719-step games
    with the agent in a FRESH process (``python -S -B``, cwd = extraction, PYTHONPATH =
    extraction + the torch/numpy site directories only, main.py executed like Kaggle does:
    compiled and exec'd without ``__file__``, last callable taken as the agent).  The engine
    and the rival run in this process; observations/actions cross a JSON pipe.
``trace``  (internal) dependency tracer used by ``build``.

Nothing is uploaded.  ``ColdArchiveAgent`` is also what ``play.py --archive`` uses.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import selectors
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HERE_REL = HERE.relative_to(ROOT).as_posix()
SCHEMA = "primitive-bc-kaggle-archive-v1"
MANIFEST = "MANIFEST.json"
WEIGHTS = "policy.pt"
MAX_BYTES = 100_000_000
STEPS = 719


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _interpreter_env(threads, pythonpath):
    keep = {k: v for k, v in os.environ.items()
            if k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER")}
    return {**keep, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1", "PYTHONPATH": os.pathsep.join(str(p) for p in pythonpath),
            "PBC_THREADS": str(threads),
            **{k: str(threads) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def runtime_sites():
    """Site directories holding torch and numpy (and their pure-Python dependencies)."""
    import numpy
    import torch
    sites = []
    for module in (torch, numpy):
        site = str(Path(module.__file__).resolve().parents[1])
        if site not in sites:
            sites.append(site)
    return sites


# ----------------------------------------------------------------------------- main.py
MAIN_TEMPLATE = '''"""Primitive-BC Kaggle agent: lazy CPU policy load, PASS on any exception, one stderr line per step."""
import os as _os
import sys as _sys
import time as _time
from pathlib import Path as _Path

_THREADS = max(1, int(_os.environ.get("PBC_THREADS", "{threads}")))
_os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_key] = str(_THREADS)
_ROOT = _Path("/kaggle_simulations/agent")
if not (_ROOT / "{manifest}").is_file():
    _ROOT = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
for _p in (_ROOT / "{here_rel}", _ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
_PBC_SETTINGS = {{"seed": {seed}, "temperature": {temperature}, "weights": "{weights}"}}
_PBC_STATUS = {{"calls": 0, "fallbacks": 0, "last_fallback": False, "last_error": None,
               "load_ms": None, "import_error": None}}
_PBC_HOLDER = {{}}


def agent(observation, configuration=None):
    begun = _time.perf_counter()
    status = _PBC_STATUS
    status["calls"] += 1
    fallback, error, loading = False, None, "agent" not in _PBC_HOLDER
    try:
        if loading:
            if status["import_error"] is not None:
                raise RuntimeError(status["import_error"])
            try:
                import importlib.util as _util
                for _name in ("model", "codec"):
                    _mod = _sys.modules.get(_name)
                    if _mod is not None and not str(getattr(_mod, "__file__", "")).startswith(str(_ROOT)):
                        del _sys.modules[_name]
                _spec = _util.spec_from_file_location("pbc_agent", str(_ROOT / "{here_rel}" / "agent.py"))
                _module = _util.module_from_spec(_spec)
                _sys.modules["pbc_agent"] = _module
                _spec.loader.exec_module(_module)
                _PBC_HOLDER["agent"] = _module.make_agent(
                    str(_ROOT / _PBC_SETTINGS["weights"]), seed=_PBC_SETTINGS["seed"],
                    temperature=_PBC_SETTINGS["temperature"], threads=_THREADS, log=False)
            except Exception as exc:
                status["import_error"] = type(exc).__name__ + ": " + str(exc)
                raise
        actor = _PBC_HOLDER["agent"]
        action = actor(observation, configuration)
        fallback, error = actor.last_fallback, actor.last_error
        if loading:
            status["load_ms"] = actor.load_ms
    except Exception as exc:
        action = {{"farmer": ["PASS"], "hands": [], "market": []}}
        fallback, error = True, type(exc).__name__ + ": " + str(exc)
    status["last_fallback"], status["last_error"] = fallback, error
    status["fallbacks"] += int(bool(fallback))
    try:
        ms = (_time.perf_counter() - begun) * 1000.0
        step = observation.get("step") if hasattr(observation, "get") else None
        extra = ""
        if loading and status["load_ms"] is not None:
            extra += " load_ms=%.0f" % status["load_ms"]
        if fallback:
            extra += " err=" + str(error).split(":", 1)[0][:40]
        print("[pbc] step=%s ms=%.1f fb=%d%s" % (step, ms, int(bool(fallback)), extra),
              file=_sys.stderr, flush=True)
    except Exception:
        pass
    return action
'''


def main_source(*, threads, seed, temperature):
    text = MAIN_TEMPLATE.format(threads=int(threads), manifest=MANIFEST, here_rel=HERE_REL,
                                seed=int(seed), temperature=float(temperature), weights=WEIGHTS)
    compile(text, "main.py", "exec")
    return text.encode()


# ----------------------------------------------------------------------------- trace
def fixture_observations(policy, steps, *, world_seed=20260923):
    """(builder process) ``steps`` x 2 seats of real observations with the policy in both seats."""
    import agent as agent_module
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    actor = agent_module.Agent("preloaded", policy=policy, log=False)
    env = ExactKaggricultureEnvironment(seed=world_seed)
    rows = []
    for _ in range(steps):
        observations = [json.loads(json.dumps(env.agent_observation(p))) for p in range(2)]
        rows.append({"observations": observations, "configuration": dict(env.configuration)})
        env.step([actor(o, env.configuration) for o in observations])
    if actor.fallbacks:
        raise RuntimeError(f"fixture agent fell back {actor.fallbacks}x: {actor.errors[:3]}")
    return rows


def trace(model, fixtures, out):
    """(subprocess, engine never imported) repository files the agent opens or imports.

    The audit hook is installed before ``agent`` is imported, so every repository module
    in ``sys.modules`` at exit (besides this script) and every repository file opened
    (data such as vocabularies, source-pin reads) belongs to the agent.
    """
    opened = set()

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            opened.add(os.fsdecode(args[0]))

    sys.addaudithook(hook)
    import agent as agent_module
    actor = agent_module.make_agent(model, seed=0, threads=1, log=False)
    rows = json.loads(Path(fixtures).read_text())
    for row in rows:
        for observation in row["observations"]:
            actor(observation, row["configuration"])
    if actor.fallbacks:
        raise RuntimeError(f"traced agent fell back {actor.fallbacks}x: {actor.errors[:3]}")
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and name != "__main__":
            opened.add(path)
    this = Path(__file__).resolve()
    model_path = Path(model).resolve() if Path(model).is_file() else None
    files = set()
    for raw in opened:
        path = Path(raw)
        path = (path if path.is_absolute() else Path.cwd() / path).resolve()
        if ("__pycache__" in path.parts or path.suffix == ".pyc" or not path.is_file()
                or path in (model_path, this, Path(fixtures).resolve())):
            continue
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            continue
        if rel.startswith("src/") or rel.startswith(HERE_REL + "/"):
            files.add(rel)
    Path(out).write_text(json.dumps(sorted(files), indent=1))
    return sorted(files)


def traced_sources(model, rows):
    with tempfile.TemporaryDirectory(prefix="pbc-trace-") as tmp:
        out, fixtures = Path(tmp) / "deps.json", Path(tmp) / "fixtures.json"
        fixtures.write_text(json.dumps(rows))
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(HERE)])}
        done = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "trace",
                               "--model", str(model), "--fixtures", str(fixtures), "--out", str(out)],
                              cwd=tmp, env=env, capture_output=True, text=True, timeout=1800)
        if done.returncode or not out.is_file():
            raise RuntimeError(f"dependency trace failed ({done.returncode}): {done.stderr[-3000:]}")
        return json.loads(out.read_text())


# ----------------------------------------------------------------------------- tar
def checked_name(name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or any(part in ("", ".", "..") for part in p.parts)
            or "\\" in name or name != p.as_posix()):
        raise ValueError(f"unsafe archive member name: {name!r}")
    return name


def write_tar(path, members):
    """Deterministic gzip tar of regular files (sorted names, mtime 0, mode 0644)."""
    with open(path, "xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
            for name, value in sorted(members.items()):
                data = value.read_bytes() if isinstance(value, Path) else value
                info = tarfile.TarInfo(checked_name(name))
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(data))


def extract(archive_path, destination):
    """Regular members only, safe names, every member matching MANIFEST.json exactly."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("empty extraction directory required")
    if Path(archive_path).stat().st_size > MAX_BYTES:
        raise ValueError("archive over 100 MB")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError("duplicate archive members")
        for m in members:
            checked_name(m.name)
            if not m.isfile():
                raise ValueError(f"non-regular member {m.name}")
        manifest = json.loads(archive.extractfile(MANIFEST).read())
        if set(manifest["members"]) | {MANIFEST} != set(names):
            raise ValueError("archive inventory differs from manifest")
        for m in members:
            data = archive.extractfile(m).read()
            if m.name != MANIFEST:
                expected = manifest["members"][m.name]
                if expected != {"sha256": sha256_bytes(data), "bytes": len(data)}:
                    raise ValueError(f"member identity mismatch: {m.name}")
            target = destination / m.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return manifest


# ----------------------------------------------------------------------------- build
def _git_state(paths):
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        lines = subprocess.run(["git", "status", "--porcelain", "--ignored", "--", *paths], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.split("\n")
        state = {"head": head, "modified_or_untracked": [], "ignored_not_versioned": []}
        for line in (l for l in lines if l.strip()):
            state["ignored_not_versioned" if line.startswith("!!") else "modified_or_untracked"].append(line[3:])
        # Member sha256 values in MANIFEST.json are the identity; git only says what is versioned.
        return state
    except (OSError, subprocess.CalledProcessError):
        return {"head": None}


def build(model, out, *, storage="fp16", threads=1, seed=0, temperature=1.0, trace_steps=24,
          max_bytes=MAX_BYTES):
    import torch
    import agent as agent_module
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    policy = agent_module.load_policy(model, seed=seed)
    blob = agent_module.export_weights(policy, storage=storage)
    with tempfile.TemporaryDirectory(prefix="pbc-build-") as tmp:
        weights = Path(tmp) / WEIGHTS
        torch.save(blob, weights)
        sources = traced_sources(weights, fixture_observations(policy, trace_steps))
        if f"{HERE_REL}/agent.py" not in sources:
            raise RuntimeError("trace did not record agent.py")
        # Read every source once: the manifest hashes exactly the bytes written to the tar.
        members = {name: (ROOT / name).read_bytes() for name in sources}
        members[WEIGHTS] = weights.read_bytes()
        members["main.py"] = main_source(threads=threads, seed=seed, temperature=temperature)
        inventory = {name: {"sha256": sha256_bytes(v), "bytes": len(v)} for name, v in members.items()}
        params = None if isinstance(policy, agent_module.StubPolicy) else \
            int(sum(p.numel() for p in policy.parameters()))
        manifest = {
            "schema": SCHEMA, "entrypoint": "main.py:agent",
            "model_spec": str(model),
            "model_sha256": sha256_file(model) if Path(model).is_file() else None,
            "policy_kind": "stub" if isinstance(policy, agent_module.StubPolicy) else "PolicyNet",
            "parameters": params, "weights_storage": blob.get("storage", "none"),
            "declared": {"threads": int(threads), "seed": int(seed), "temperature": float(temperature),
                         "device": "cpu", "fallback": "PASS on any exception"},
            "sources_traced_steps": int(trace_steps),
            "git": _git_state(sources),
            "builder_sha256": sha256_file(Path(__file__)),
            "members": inventory,
        }
        members[MANIFEST] = json.dumps(manifest, indent=1, sort_keys=True).encode()
        partial = out.with_name(out.name + ".partial")
        if partial.exists():
            partial.unlink()
        out.parent.mkdir(parents=True, exist_ok=True)
        write_tar(partial, members)
        size = partial.stat().st_size
        if size > max_bytes:
            partial.unlink()
            raise ValueError(f"archive {size} bytes > {max_bytes}")
        partial.rename(out)
    return {"archive": str(out), "archive_sha256": sha256_file(out), "archive_bytes": out.stat().st_size,
            "members": len(members), "uncompressed_bytes": sum(v["bytes"] for v in inventory.values()),
            "weights_bytes": inventory[WEIGHTS]["bytes"], "manifest": manifest}


# ----------------------------------------------------------------------------- cold agent
HOST_SOURCE = r'''
import json, os, sys, time
proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)                      # anything the agent prints goes to stderr, not the protocol
root = sys.argv[1]
os.chdir(root)


class Struct(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def structify(value):
    if isinstance(value, dict):
        return Struct({k: structify(v) for k, v in value.items()})
    if isinstance(value, list):
        return [structify(v) for v in value]
    return value


begun = time.perf_counter()
source = open(os.path.join(root, "main.py")).read()
namespace = {}
exec(compile(source, os.path.join(root, "main.py"), "exec"), namespace)   # Kaggle-style, no __file__
agent = [v for v in namespace.values() if callable(v)][-1]
status = namespace.get("_PBC_STATUS", {})
proto.write(json.dumps({"ready": True, "import_ms": (time.perf_counter() - begun) * 1000.0,
                        "agent_name": getattr(agent, "__name__", None), "sys_path": sys.path}) + "\n")
for line in sys.stdin:
    request = json.loads(line)
    if request.get("close"):
        break
    t0 = time.perf_counter()
    try:
        action = agent(structify(request["observation"]), structify(request["configuration"]))
        reply = {"action": action, "error": None}
        text = json.dumps(reply)
    except BaseException as error:           # an agent exception is a hard failure on Kaggle
        text = json.dumps({"action": None, "error": type(error).__name__ + ": " + str(error)})
    reply = json.loads(text)
    reply["ms"] = (time.perf_counter() - t0) * 1000.0
    reply["fallback"] = bool(status.get("last_fallback"))
    reply["fallback_error"] = status.get("last_error")
    proto.write(json.dumps(reply) + "\n")
loaded = {n: getattr(m, "__file__", None) for n, m in list(sys.modules.items())}
outside = sorted(n for n, f in loaded.items() if f and ("farm_neat" in n or n in ("agent", "model", "codec", "pbc_agent"))
                 and not str(f).startswith(root))
proto.write(json.dumps({"closed": True, "our_modules_outside_archive": outside,
                        "archive_modules": sorted(n for n, f in loaded.items() if f and str(f).startswith(root))}) + "\n")
'''


class ColdArchiveAgent:
    """Cold child with finite startup/request/shutdown deadlines (POSIX pipes)."""

    def __init__(self, archive, *, threads=1, workdir=None, timeout=60.0):
        self.archive = str(archive)
        self.threads = int(threads)
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("cold timeout must be finite and strictly positive")
        self.proc = None
        self._stderr = None
        self._tmp = None
        self._closed = False
        self._read_buffer = bytearray()
        self.stderr_text = ""
        self.close_error = None
        self.closed_report = None
        self.fallbacks = 0
        self.hard_errors = 0
        self.errors = []
        self.child_ms = []
        try:
            self._tmp = tempfile.TemporaryDirectory(prefix="pbc-cold-", dir=workdir)
            base = Path(self._tmp.name)
            self.root = base / "agent"
            self.manifest = extract(self.archive, self.root)
            host_dir = base / "host"
            host_dir.mkdir()
            (host_dir / "pbc_host.py").write_text(HOST_SOURCE)
            self.pythonpath = [str(self.root), *runtime_sites()]
            self.stderr_path = base / "stderr.log"
            self._stderr = open(self.stderr_path, "w")
            self.proc = subprocess.Popen(
                [sys.executable, "-S", "-B", str(host_dir / "pbc_host.py"), str(self.root)],
                cwd=str(self.root), env=_interpreter_env(self.threads, self.pythonpath),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr, bufsize=0)
            # A blocking readline after select still hangs on a partial JSON line.
            # Use raw nonblocking bytes consistently, with our own line buffer.
            os.set_blocking(self.proc.stdin.fileno(), False)
            os.set_blocking(self.proc.stdout.fileno(), False)
            self.hello = self._read(time.monotonic() + self.timeout)
            if self.hello.get("ready") is not True:
                raise RuntimeError(f"cold host failed to start: {self.hello}")
        except BaseException:
            self._dispose()
            raise

    def _wait(self, stream, event, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"cold host protocol deadline exceeded ({self.timeout:g}s)")
        with selectors.DefaultSelector() as selector:
            selector.register(stream, event)
            if not selector.select(remaining):
                raise TimeoutError(f"cold host protocol deadline exceeded ({self.timeout:g}s)")

    def _send(self, request, deadline):
        data = memoryview((json.dumps(request) + "\n").encode("utf-8"))
        while data:
            self._wait(self.proc.stdin, selectors.EVENT_WRITE, deadline)
            try:
                written = os.write(self.proc.stdin.fileno(), data)
            except BlockingIOError:
                continue
            if written == 0:
                raise RuntimeError("cold host input pipe closed")
            data = data[written:]

    def _read(self, deadline):
        while b"\n" not in self._read_buffer:
            self._wait(self.proc.stdout, selectors.EVENT_READ, deadline)
            try:
                chunk = os.read(self.proc.stdout.fileno(), 65536)
            except BlockingIOError:
                continue
            if not chunk:
                self._stderr.flush()
                tail = self.stderr_path.read_text(errors="replace")[-3000:]
                raise RuntimeError(f"cold host exited (code {self.proc.poll()}): {tail}")
            self._read_buffer.extend(chunk)
        line, _, rest = self._read_buffer.partition(b"\n")
        self._read_buffer = bytearray(rest)
        reply = json.loads(line)
        if not isinstance(reply, dict):
            raise RuntimeError("cold host protocol reply must be an object")
        return reply

    def __call__(self, observation, configuration=None):
        if self._closed:
            raise RuntimeError("cold host is closed")
        try:
            deadline = time.monotonic() + self.timeout
            self._send({"observation": observation, "configuration": dict(configuration or {})}, deadline)
            reply = self._read(deadline)
            self.child_ms.append(reply["ms"])
            if reply["error"] is not None:
                self.hard_errors += 1
                self.errors.append("HARD " + reply["error"][:300])
                return {"farmer": ["PASS"], "hands": [], "market": []}
            if reply["fallback"]:
                self.fallbacks += 1
                if len(self.errors) < 20:
                    self.errors.append(str(reply["fallback_error"])[:300])
            return reply["action"]
        except BaseException:
            # A timed-out reply cannot be paired safely with a subsequent action.
            # Fail the evaluation, reap this child, and never fabricate a terminal result.
            self._dispose()
            raise

    def _dispose(self):
        if self._closed:
            return self.stderr_text
        self._closed = True
        try:
            if self.proc is not None:
                if self.proc.poll() is None:
                    self.proc.kill()
                self.proc.wait(timeout=1.0)
        finally:
            if self.proc is not None:
                for pipe in (self.proc.stdin, self.proc.stdout):
                    if pipe is not None:
                        pipe.close()
            if self._stderr is not None:
                self._stderr.close()
                self.stderr_text = self.stderr_path.read_text(errors="replace")
            if self._tmp is not None:
                self._tmp.cleanup()
        return self.stderr_text

    def close(self):
        if self._closed:
            return self.stderr_text
        try:
            if self.proc.poll() is None:
                deadline = time.monotonic() + self.timeout
                self._send({"close": True}, deadline)
                self.closed_report = self._read(deadline)
                if self.closed_report.get("closed") is not True:
                    raise RuntimeError("cold host did not acknowledge shutdown")
                self.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except Exception as error:  # preserve the primary game/protocol error
            self.close_error = type(error).__name__ + ": " + str(error)
        finally:
            self._dispose()
        return self.stderr_text


def parse_step_log(text):
    lines = [l for l in text.splitlines() if l.startswith("[pbc] step=")]
    fields = [dict(kv.split("=", 1) for kv in l[6:].split() if "=" in kv) for l in lines]
    return {"lines": len(lines), "steps": [int(f["step"]) for f in fields if f.get("step", "").isdigit()],
            "fallback_lines": sum(f.get("fb") == "1" for f in fields),
            "ms": [float(f["ms"]) for f in fields if "ms" in f],
            "load_ms": next((float(f["load_ms"]) for f in fields if "load_ms" in f), None),
            "other_stderr_lines": len([l for l in text.splitlines() if l.strip() and not l.startswith("[pbc] step=")])}


def verify(archive, *, opponents=("PASS",), seats=(0,), threads=1, world_seed=20260923, sampler_seed=7100,
           workdir=None, timeout=60.0):
    """Complete cold games; v2 certification requires zero fallback/hard failures.

    Engine completion alone is diagnostic, not verification of policy execution.
    """
    opponents, seats = tuple(opponents), tuple(seats)
    if not opponents or not seats:
        raise ValueError("archive verification needs nonempty opponents and seats")
    if any(not isinstance(op, str) or not op.strip() for op in opponents):
        raise ValueError("opponents must be nonempty names")
    if any(type(seat) is not int or seat not in (0, 1) for seat in seats):
        raise ValueError("verification seats must be 0 or 1")
    sys.path[:0] = [str(HERE)]
    import play
    games = []
    for opponent in opponents:
        for seat in seats:
            cold = ColdArchiveAgent(archive, threads=threads, workdir=workdir, timeout=timeout)
            try:
                game = play.play_game(cold, opponent, our_seat=seat, world_seed=world_seed,
                                      sampler_seed=sampler_seed + seat)
            finally:
                stderr = cold.close()
            log = parse_step_log(stderr)
            game["cold"] = {
                "hello": cold.hello, "closed": cold.closed_report, "hard_errors": cold.hard_errors,
                "close_error": cold.close_error,
                "pythonpath": cold.pythonpath, "threads": threads,
                "child_ms": play.latency_summary(cold.child_ms),
                "child_ms_after_first": play.latency_summary(cold.child_ms[1:]),
                "log_lines": log["lines"], "log_steps_contiguous": log["steps"] == list(range(STEPS)),
                "log_fallback_lines": log["fallback_lines"], "log_load_ms": log["load_ms"],
                "other_stderr_lines": log["other_stderr_lines"],
                "stderr_tail": stderr[-1500:] if (log["fallback_lines"] or cold.hard_errors) else "",
            }
            closed = cold.closed_report or {}
            game["cold"]["engine_complete"] = bool(
                game["done"] and game["steps"] == STEPS and game["statuses"] == ["DONE", "DONE"])
            game["cold"]["complete"] = bool(
                game["cold"]["engine_complete"] and cold.hard_errors == 0
                and cold.fallbacks == 0 and log["fallback_lines"] == 0
                and log["lines"] == STEPS and game["cold"]["log_steps_contiguous"]
                and closed.get("closed") is True and cold.close_error is None
                and closed.get("our_modules_outside_archive") == [])
            games.append(game)
    return {"schema": "primitive-bc-cold-archive-verify-v2", "archive": str(archive),
            "archive_sha256": sha256_file(archive), "archive_bytes": Path(archive).stat().st_size,
            "threads": threads, "games": games,
            "complete": bool(games) and all(g["cold"]["complete"] for g in games),
            "fallbacks": sum(g["fallbacks"] or 0 for g in games)}


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the archive (and verify it cold unless --no-verify)")
    b.add_argument("--model", required=True, help="checkpoint / weights path, 'random[:seed]' or 'stub[:seed]'")
    b.add_argument("--out", required=True)
    b.add_argument("--storage", choices=("fp32", "fp16", "int8"), default="fp16")
    b.add_argument("--threads", type=int, default=1, help="declared torch threads in main.py")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--temperature", type=float, default=1.0)
    b.add_argument("--trace-steps", type=int, default=24)
    b.add_argument("--no-verify", action="store_true")
    b.add_argument("--receipt", help="JSON receipt path (default <out>.build.json)")
    v = sub.add_parser("verify", help="cold complete games from an existing archive")
    v.add_argument("--archive", required=True)
    v.add_argument("--opponents", default="PASS")
    v.add_argument("--seats", default="0")
    v.add_argument("--threads", type=int, default=1)
    v.add_argument("--timeout", type=float, default=60.0, help="per-request cold-host deadline in seconds")
    v.add_argument("--out", required=True)
    t = sub.add_parser("trace", help=argparse.SUPPRESS)
    t.add_argument("--model", required=True)
    t.add_argument("--fixtures", required=True)
    t.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if args.cmd == "trace":
        trace(args.model, args.fixtures, args.out)
        return 0
    for p in (ROOT / "src", HERE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if args.cmd == "build":
        receipt_path = Path(args.receipt or args.out + ".build.json")
        if receipt_path.exists():
            raise SystemExit(f"refusing to overwrite {receipt_path}")
        begun = time.perf_counter()
        result = build(args.model, args.out, storage=args.storage, threads=args.threads, seed=args.seed,
                       temperature=args.temperature, trace_steps=args.trace_steps)
        result["build_seconds"] = round(time.perf_counter() - begun, 2)
        print(json.dumps({k: result[k] for k in ("archive", "archive_sha256", "archive_bytes", "members",
                                                  "uncompressed_bytes", "weights_bytes")}), flush=True)
        if not args.no_verify:
            result["verify"] = verify(args.out, threads=args.threads)
            print(json.dumps({"cold_complete": result["verify"]["complete"],
                              "fallbacks": result["verify"]["fallbacks"]}), flush=True)
        receipt_path.write_text(json.dumps(result, indent=1, sort_keys=True))
        return 0 if result.get("verify", {"complete": True})["complete"] else 2
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    result = verify(args.archive, opponents=tuple(args.opponents.split(",")),
                    seats=tuple(int(s) for s in args.seats.split(",")), threads=args.threads, timeout=args.timeout)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    for g in result["games"]:
        c = g["cold"]
        print(f"{g['opponent']} seat{g['our_seat']} {g['outcome']} steps={g['steps']} own={g['own_cash']:.0f} "
              f"rival={g['rival_cash']:.0f} fallbacks={g['fallbacks']} hard={c['hard_errors']} "
              f"log_lines={c['log_lines']} child_p50={c['child_ms'].get('p50')} p95={c['child_ms'].get('p95')} "
              f"max={c['child_ms'].get('max')} load_ms={c['log_load_ms']} complete={c['complete']}", flush=True)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
