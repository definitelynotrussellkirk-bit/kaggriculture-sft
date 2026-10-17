"""Owner-only, bounded repetition of the maintained full-economic learning loop.

Purpose: fresh child self-play -> executed original witness/PPO -> next child.
Owner: learning maintainer. Status: orchestration, not another trainer or admission.
The source-bound collector/fitter run unchanged in fresh logged subprocesses. This
driver never loads tensors, changes schedules, deletes data, or uploads models.
Use RUN.py storage run around run; children stay in that guard's process group.
Only the owner may assert exclusive, previously unexposed world-seed ownership.
--seconds is a between-round budget, not an interruption deadline. Each stage has
its own --stage-seconds limit; the outer storage guard is the whole-job deadline.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "farm-full-economic-sustain-v1"
PPO_STATUS = "FRESH_FULL_PROGRAM_PPO_COMMITTED_NOT_STRENGTH"
GRPO_STATUS = "FRESH_GROUPED_PROGRAM_GRPO_AUX_COMMITTED_NOT_STRENGTH"
BC_STATUS = "BC_UPDATES_COMPLETED_NOT_RL_STRENGTH"
EFFECT_STATUS = "SUPERVISED_PLAN_EFFECT_UPDATES_COMPLETED_NOT_RL_STRENGTH"
TENSORS = ("model.pt", "optimizer.pt")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key: " + key)
        value[key] = item
    return value


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def path(root, value, *, output=False):
    root = Path(root).resolve()
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else root/candidate
    if candidate != candidate.resolve() or not candidate.is_relative_to(root):
        raise ValueError("explicit non-symlink path inside this workspace required")
    if output and not candidate.is_relative_to(root/"work/runs"):
        raise ValueError("generated output must be under work/runs")
    return candidate


def relative(root, value):
    return path(root, value).relative_to(Path(root).resolve()).as_posix()


def sources(root=ROOT):
    # Changing this pin list changes the source identity of every future checkpoint; intended.
    # Importing the declarations does not initialize CUDA or load a checkpoint.
    if str(Path(root) / "src") not in sys.path:
        sys.path.insert(0, str(Path(root) / "src"))
    from farm_neat.training import ppo as fitter
    from farm_neat.training import moe_lab as collector
    return {**fitter.sources(), **collector.sources(),
            "src/farm_neat/training/sustain.py": sha(Path(root)/"src/farm_neat/training/sustain.py"),
            "tests/contracts/test_sustain.py": sha(Path(root)/"tests/contracts/test_sustain.py")}


def attest(root, pins):
    if not isinstance(pins, dict) or not pins:
        raise ValueError("nonempty execution source closure required")
    for name, expected in pins.items():
        if sha(path(root, name)) != expected:
            raise ValueError("frozen source changed: " + name)


def checkpoint(root, folder, *, tensors=True):
    """Outer file joins only; the real fitter still validates Adam/RNG/numerics."""
    folder = path(root, folder)
    for name in ("RESULT.json", "PLAN.json", "model.pt.json", *TENSORS):
        path(root, folder/name)
    if (folder/"FAILURE.json").exists():
        raise ValueError("failed checkpoint cannot be a parent")
    result, plan, manifest = (read(folder/name) for name in ("RESULT.json", "PLAN.json", "model.pt.json"))
    age = result["global_optimizer_updates"]
    if (result["status"] not in (PPO_STATUS, GRPO_STATUS, BC_STATUS, EFFECT_STATUS)
            or type(age) is not int or age < 1 or manifest["optimizer_steps"] != age
            or age != plan["start_update"] + result["optimizer_updates"]
            or result["optimizer_updates"] < 1
            or result["plan_sha256"] != sha(folder/"PLAN.json")
            or result["manifest"] != manifest["content_sha256"]
            or digest({k: v for k, v in manifest.items() if k != "content_sha256"}) != manifest["content_sha256"]
            or manifest["lineage"]["plan_sha256"] != result["plan_sha256"]):
        raise ValueError("completed model/plan/manifest/age join mismatch")
    budget = plan["schedule"]["planned_optimizer_steps"]
    if type(budget) is not int or not age <= budget:
        raise ValueError("original schedule exhausted or malformed")
    if tensors and any(sha(folder/name) != result[key] for name, key in
                       (("model.pt", "after_model_sha256"), ("optimizer.pt", "optimizer_sha256"))):
        raise ValueError("actual model/Adam file hash mismatch")
    return {"directory": relative(root, folder), "model_sha256": result["after_model_sha256"],
            "optimizer_sha256": result["optimizer_sha256"], "manifest_sha256": result["manifest"],
            "plan_file_sha256": sha(folder/"PLAN.json"), "result_file_sha256": sha(folder/"RESULT.json"),
            "optimizer_age": age, "rollout_iteration": manifest["rollout_iteration"],
            "schedule": plan["schedule"]}


def prepare(root, output, *, parent, model_sha256, optimizer_sha256, owner,
            seed_start, seed_count, exclusive_seeds=False, max_updates=4,
            command_budget=8, run_byte_limit=850_000_000, stage_seconds=900):
    root, output = Path(root).resolve(), path(root, output, output=True)
    if output.exists():
        raise ValueError("prepare requires a new output; never overwrite a seed allocation")
    if not isinstance(owner, str) or not owner.strip() or not exclusive_seeds:
        raise ValueError("explicit owner assertion of exclusive fresh TRAIN seeds required")
    for name, value, lower, upper in (("seed_start", seed_start, 0, 2**32-2),
            ("seed_count", seed_count, 1, 1_000_000), ("max_updates", max_updates, 1, 16),
            ("command_budget", command_budget, 2, 8), ("run_byte_limit", run_byte_limit, 1, 2_000_000_000),
            ("stage_seconds", stage_seconds, 1, 3600)):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError("invalid explicit " + name)
    if seed_start + seed_count - 1 > 2**32-3:
        raise ValueError("world plus the existing two sampler seeds must fit the numeric contract")
    anchor = checkpoint(root, parent)
    if anchor["model_sha256"] != model_sha256 or anchor["optimizer_sha256"] != optimizer_sha256:
        raise ValueError("externally pinned parent required")
    # A one-game epoch has15 minibatches; a16 allocation does not add an epoch.
    upper_updates = seed_count * min(max_updates, 15)
    if anchor["optimizer_age"] + upper_updates > anchor["schedule"]["planned_optimizer_steps"]:
        raise ValueError("requested worst-case updates exceed the original schedule; explicit transition required")
    pins = sources(root); attest(root, pins)
    plan = {"schema": SCHEMA+"-plan", "owner": owner, "source_pins": pins,
        "parent": anchor, "seed_start": seed_start, "seed_count": seed_count,
        "seed_ownership": "OWNER_ASSERTED_EXCLUSIVE_FRESH_TRAIN_NOT_CROSS_WORKSPACE_AUTHENTICATION",
        "max_updates_per_round": max_updates, "command_budget": command_budget,
        "games_per_round": 1, "stage_seconds": stage_seconds, "run_byte_limit": run_byte_limit,
        "retention": "NO_AUTOMATIC_DELETION_OWNER_RETIREMENT_AFTER_VERIFIED_COMMIT_ONLY",
        "schedule_extension": False, "population_strength_claim": False}
    output.mkdir(parents=True)
    write(output/"PLAN.json", plan)
    return status(root, output)


def status(root, output):
    output = path(root, output, output=True)
    plan = read(output/"PLAN.json")
    if plan["schema"] != SCHEMA+"-plan":
        raise ValueError("typed sustained plan required")
    attest(root, plan["source_pins"])
    current, committed, updates, previous = plan["parent"], [], 0, None
    for ordinal in range(plan["seed_count"]):
        folder = output/f"round-{ordinal:06d}"
        if not (folder/"COMPLETED.json").exists():
            break
        row = read(folder/"COMPLETED.json")
        if (row["schema"] != SCHEMA+"-round" or row["ordinal"] != ordinal
                or row["seed"] != plan["seed_start"]+ordinal or row["parent"] != current
                or row["plan_file_sha256"] != sha(output/"PLAN.json")
                or row["child"]["optimizer_age"] != current["optimizer_age"]+row["optimizer_updates"]
                or not 1 <= row["optimizer_updates"] <= plan["max_updates_per_round"]
                or row["child"]["schedule"] != plan["parent"]["schedule"]):
            raise ValueError("committed child/parent/cursor/schedule chain mismatch")
        previous, current = current, row["child"]
        committed.append(row)
        updates += row["optimizer_updates"]
    # Older raw/checkpoint tensors are NOT a resume dependency. No deletion here.
    for tip in (current, previous):
        if tip is not None and checkpoint(root, tip["directory"]) != tip:
            raise ValueError("current/rollback checkpoint changed")
    pending = output/f"round-{len(committed):06d}"
    expected_rounds = {f"round-{i:06d}" for i in range(len(committed)+1)}
    if any(p.name not in expected_rounds for p in output.glob("round-*")):
        raise ValueError("noncontiguous or unexpected round directory")
    if any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("no symlink outputs in an owner campaign")
    logical = sum(p.stat().st_size for p in output.rglob("*") if p.is_file() and not p.is_symlink())
    return {"schema": SCHEMA+"-status", "plan_file_sha256": sha(output/"PLAN.json"),
        "owner": plan["owner"], "current": current, "rollback": previous,
        "committed_fresh_games": len(committed), "actual_optimizer_updates": updates,
        "next_seed": plan["seed_start"]+len(committed),
        "next_seed_exclusive_end": plan["seed_start"]+plan["seed_count"],
        "pending_round": relative(root, pending) if pending.exists() else None,
        "retained_logical_bytes": logical, "live_process_claim": False,
        "responsive_reference_games": 0, "population_strength_claim": False}


def gpu_sample():
    """Observed utilization, not a promise that the device is continuously full."""
    try:
        row = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2, check=True).stdout.splitlines()[0]
        utilization, memory = (float(x.strip()) for x in row.split(","))
        if not all(math.isfinite(x) for x in (utilization, memory)):
            raise ValueError("nonfinite telemetry")
        return {"gpu_utilization_percent": utilization, "gpu_memory_MiB": memory}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {"telemetry_unavailable": True}


def _descendants(pid):
    """Linux owned process tree, including children spawned by worker threads."""
    result, seen, pending = [], {pid}, [pid]
    while pending:
        parent = pending.pop()
        try:
            tasks = list((Path("/proc")/str(parent)/"task").iterdir())
        except FileNotFoundError:
            continue
        for task in tasks:
            try:
                children = (task/"children").read_text().split()
            except FileNotFoundError:
                continue
            for child in map(int, children):
                if child not in seen:
                    seen.add(child); result.append(child); pending.append(child)
    return result


def _signal_descendants(number):
    for pid in reversed(_descendants(os.getpid())):
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            try:
                signal.pidfd_send_signal(fd, number)
            except ProcessLookupError:
                pass
        finally:
            os.close(fd)


def _reap_children():
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                return
        except ChildProcessError:
            return


@contextmanager
def _stage_tree():
    """TERM/KILL/reap owned descendants without escaping the storage group.

    Linux subreaping keeps an orphaned witness attached to THIS stage owner even
    when its immediate fitter exits. Refuse unrelated pre-existing children. This
    is lifecycle supervision for the frozen helpers, not a malicious-code sandbox.
    """
    if sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise ValueError("Linux pidfd/subreaper supervision required before launching a stage")
    if _descendants(os.getpid()):
        raise ValueError("stage owner must have no unrelated existing child process")
    libc = ctypes.CDLL(None, use_errno=True)
    old = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(old), 0, 0, 0) or libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "cannot establish stage child subreaper")
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    def cancel(number, frame):
        raise KeyboardInterrupt("stage cancelled by signal " + str(number))
    try:
        for number in handlers:
            signal.signal(number, cancel)
        yield
    finally:
        for number in handlers:
            signal.signal(number, signal.SIG_IGN)
        try:
            for number, grace in ((signal.SIGTERM, .25), (signal.SIGKILL, 1.0)):
                _signal_descendants(number)
                deadline = time.monotonic()+grace
                while True:
                    _reap_children()
                    if not _descendants(os.getpid()):
                        break
                    if time.monotonic() >= deadline:
                        break
                    if number == signal.SIGKILL:
                        _signal_descendants(number)  # Include newly adopted orphans.
                    time.sleep(.01)
            if _descendants(os.getpid()):
                raise RuntimeError("owned stage descendants survived cleanup; outer guard must stop")
        finally:
            libc.prctl(36, old.value, 0, 0, 0)
            for number, handler in handlers.items():
                signal.signal(number, handler)


def execute(command, folder, *, root, timeout, marker=None):
    """Durable logs; inherited process group stays inside the outer storage guard."""
    folder = Path(folder)
    started, samples, marker_after = time.monotonic(), [], None
    write(folder.with_suffix(".STARTED.json"), {"command": command, "timeout_seconds": timeout,
          "started_unix": time.time(), "same_storage_process_group": True})
    with _stage_tree(), folder.with_suffix(".log").open("xb") as log:
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=False)
        try:
            while True:
                elapsed = time.monotonic()-started
                if marker and marker_after is None and Path(marker).exists():
                    marker_after = elapsed
                samples.append({"seconds": elapsed, **gpu_sample()})
                print(json.dumps({"stage": folder.name, "seconds": elapsed,
                    "fit_plan_visible_after_seconds": marker_after, "pid": child.pid}), flush=True)
                try:
                    code = child.wait(timeout=min(15, max(.001, timeout-elapsed)))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic()-started >= timeout:
                        raise TimeoutError("stage timeout; original inputs retained")
        except BaseException as exc:
            write(folder.with_suffix(".FAILURE.json"), {"error": type(exc).__name__+": "+str(exc),
                "elapsed_seconds": time.monotonic()-started, "inputs_retained": True,
                "descendant_cleanup": "STAGE_SUBREAPER_FINALLY_TERM_KILL_AND_REAP"})
            raise
    result = {"command": command, "exit_code": code, "elapsed_seconds": time.monotonic()-started,
        "fit_plan_visible_after_seconds": marker_after, "samples": samples,
        "log_file_sha256": sha(folder.with_suffix(".log")),
        "owned_descendants_reaped": True}
    write(folder.with_suffix(".FINISHED.json"), result)
    if code:
        raise RuntimeError("original stage failed; inspect " + str(folder.with_suffix(".log")))
    return result


def run(root, output, *, owner, plan_sha256, rounds=2, seconds=1800, retry_fit=False, executor=execute):
    root, output = Path(root).resolve(), path(root, output, output=True)
    if not sys.dont_write_bytecode or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("run requires -B and original CUBLAS :4096:8 profile")
    if type(rounds) is not int or not 1 <= rounds <= 1024 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("explicit bounded rounds and wall time required")
    plan = read(output/"PLAN.json")
    if owner != plan["owner"] or sha(output/"PLAN.json") != plan_sha256:
        raise ValueError("externally pinned plan and prepared exclusive owner required")
    fd = os.open(output/"OWNER.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    started, reason, new_games = time.monotonic(), "INVOCATION_ROUND_BOUNDARY", 0
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for _ in range(rounds):
            state = status(root, output)
            ordinal, parent = state["committed_fresh_games"], state["current"]
            if ordinal == plan["seed_count"]:
                reason = "DECLARED_FRESH_GAME_ALLOCATION_COMPLETE"; break
            if time.monotonic()-started >= seconds:
                reason = "INVOCATION_TIME_BOUNDARY"; break
            parent_dir = path(root, parent["directory"])
            next_bytes = sum((parent_dir/name).stat().st_size for name in TENSORS)+50_000_000
            if state["retained_logical_bytes"]+next_bytes > plan["run_byte_limit"]:
                reason = "RUN_STORAGE_BOUNDARY_OWNER_RETIREMENT_REQUIRED"; break
            folder = output/f"round-{ordinal:06d}"
            intent = {"schema": SCHEMA+"-intent", "ordinal": ordinal, "seed": state["next_seed"],
                "parent": parent, "plan_file_sha256": state["plan_file_sha256"]}
            if folder.exists():
                if read(folder/"INTENT.json") != intent:
                    raise ValueError("pending round parent/seed/source changed")
            else:
                folder.mkdir(); write(folder/"INTENT.json", intent)
            producer = folder/f"game-{state['next_seed']}"
            if producer.exists() and not (producer/"RESULT.json").exists():
                raise ValueError("incomplete exposed world retained; no automatic reroll or seed skip")
            if not producer.exists():
                if (folder/"collect.STARTED.json").exists():
                    raise ValueError("uncertain collection exposure retained; no automatic reroll")
                command = [sys.executable, "-B", "src/farm_neat/training/moe_lab.py", "selfplay",
                    "--model", str(parent_dir/"model.pt"), "--seed", str(state["next_seed"]),
                    "--command-budget", str(plan["command_budget"]), "--output", str(producer)]
                executor(command, folder/"collect", root=root, timeout=plan["stage_seconds"])
                new_games += 1
            game, game_result = read(producer/"PLAN.json"), read(producer/"RESULT.json")
            if (game["seed"] != state["next_seed"] or game["model_sha256"] != parent["model_sha256"]
                    or game["game_id"] != producer.name or game["model_optimizer_age"] != parent["optimizer_age"]
                    or game["rollout_iteration"] != parent["rollout_iteration"]+1):
                raise ValueError("original producer is not the current policy/fresh intended world")
            attempts = sorted(folder.glob("fit-[0-9][0-9][0-9]"))
            completed = [p for p in attempts if (p/"RESULT.json").exists() and not (p/"FAILURE.json").exists()]
            if len(completed) > 1:
                raise ValueError("multiple uncommitted fit children; explicit owner resolution required")
            if completed:
                fit_dir = completed[0]
            else:
                started_fits = list(folder.glob("fit-[0-9][0-9][0-9].STARTED.json"))
                if (attempts or started_fits) and not retry_fit:
                    raise ValueError("uncommitted fit retained; explicit --retry-fit reuses SAME producer only")
                number = 1+max([int(p.name[4:7]) for p in attempts+started_fits] or [0])
                fit_dir = folder/f"fit-{number:03d}"
                command = [sys.executable, "-B", "-m", "farm_neat.training.ppo", "fit",
                    "--model", str(parent_dir/"model.pt"), "--model-sha256", parent["model_sha256"],
                    "--optimizer-dir", str(parent_dir), "--optimizer-sha256", parent["optimizer_sha256"],
                    "--producer", str(producer), "--plan-sha256", sha(producer/"PLAN.json"),
                    "--max-updates", str(plan["max_updates_per_round"]), "--output", str(fit_dir)]
                executor(command, fit_dir, root=root, timeout=plan["stage_seconds"], marker=fit_dir/"PLAN.json")
            child = checkpoint(root, fit_dir)
            fitted = read(fit_dir/"RESULT.json"); fit_plan = read(fit_dir/"PLAN.json")
            if (fitted["status"] != PPO_STATUS or fitted["before_model_sha256"] != parent["model_sha256"]
                    or fit_plan["parent_optimizer_sha256"] != parent["optimizer_sha256"]
                    or fit_plan["producer_files"] != [{"directory": str(producer), "plan_file_sha256": sha(producer/"PLAN.json")}]
                    or fitted["physical_games"] != 1 or fitted["engine_reconstruction_games"] != 1
                    or fitted["responsive_reference_games"] != 0 or fitted["natural_calls"] != 60
                    or child["rollout_iteration"] != parent["rollout_iteration"]+1
                    or child["schedule"] != parent["schedule"]):
                raise ValueError("original executed witness/fit/child chain mismatch")
            attest(root, plan["source_pins"])
            diagnostic = read(producer/"DIAGNOSTICS.json")
            stage_times = {}
            for label, stage in (("collection", folder/"collect"), ("fitting", fit_dir)):
                receipt = stage.with_suffix(".FINISHED.json")
                stage_times[label] = read(receipt) if receipt.exists() else {
                    "measurement_unavailable": "completed child recovered without finished supervisor receipt"}
            row = {**intent, "schema": SCHEMA+"-round", "child": child,
                "optimizer_updates": fitted["optimizer_updates"],
                "producer_plan_file_sha256": sha(producer/"PLAN.json"),
                "producer_result_file_sha256": sha(producer/"RESULT.json"),
                "admission_file_sha256": sha(fit_dir/"ADMISSION.json"),
                "selfplay_result_not_strength": game_result, "commands_and_lint": diagnostic,
                "fit_diagnostics": fitted["fit"], "stage_measurements": stage_times, "physical_games": 1,
                "engine_reconstruction_games": 1, "responsive_reference_games": 0}
            write(folder/"STAGED-COMPLETED.json", row)
            (folder/"STAGED-COMPLETED.json").rename(folder/"COMPLETED.json")
            print(json.dumps({"committed_fresh_games": ordinal+1, "global_optimizer_age": child["optimizer_age"],
                "same_child_next_round": child["model_sha256"], "strength_claim": False}), flush=True)
        elapsed = time.monotonic()-started
        return {**status(root, output), "invocation_stop": reason, "invocation_seconds": elapsed,
            "newly_completed_physical_games_this_invocation": new_games,
            "fresh_physical_games_per_invocation_second": new_games/max(elapsed, 1e-12),
            "new_model_decisions_this_invocation": 60*new_games,
            "utilization_scope": "timestamped observed nvidia-smi samples in each stage; not maximum-utilization claim"}
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("output", "parent", "model-sha256", "optimizer-sha256", "owner"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--seed-start", required=True, type=int); p.add_argument("--seed-count", required=True, type=int)
    p.add_argument("--exclusive-seeds", action="store_true")
    p.add_argument("--max-updates", type=int, default=4); p.add_argument("--command-budget", type=int, default=8)
    p.add_argument("--run-byte-limit", type=int, default=850_000_000)
    p.add_argument("--stage-seconds", type=int, default=900, help="per-stage hard timeout plus bounded descendant cleanup")
    p = sub.add_parser("status"); p.add_argument("--output", required=True)
    p = sub.add_parser("run"); p.add_argument("--output", required=True); p.add_argument("--owner", required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--seconds", type=float, default=1800, help="checked BETWEEN rounds; use storage run for hard whole-job timeout")
    p.add_argument("--retry-fit", action="store_true")
    args = vars(parser.parse_args()); command = args.pop("command")
    print(json.dumps(globals()[command](ROOT, **args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
