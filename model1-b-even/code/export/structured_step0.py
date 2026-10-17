#!/usr/bin/env python3
"""Fail-closed, SHA-bound step-0 archive probe for Kaggriculture agents.

Drop-in caller interface:

    python -B step0_wire.py CANDIDATE.tar.gz --replay FIXTURE.json.gz --out STEP0.json

The parent validates the replay, safely extracts the exact archive, launches one
fresh Python process per seat, accounts for initialization plus free action time and observed overage, validates
the returned wire action, and rejects an all-PASS fallback.  A private ``--child``
mode is used so ``SystemExit(0)``, ``os._exit(0)``, module-cache state, or a hung
agent cannot accidentally certify the archive.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import signal
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any
import warnings

DEFAULT_REPLAY = '/mnt/ai/scratch/farmNEAT/tools-0929/step0-replay.json.gz'
DEFAULT_AGENT_TIMEOUT = 60.0

CROPS = frozenset({"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"})
ANIMALS = frozenset({"GOOSE", "COW", "SHEEP"})
PRODUCTS = frozenset({
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
})
ITEMS = PRODUCTS | ANIMALS
NOARG_UNIT_OPS = frozenset({
    "NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "WATER",
    "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE",
    "FEED", "CARE", "COLLECT_FERTILIZER",
})
NOARG_MARKET_OPS = frozenset({"NOOP", "HIRE", "BUY_LAND"})


class ProbeError(RuntimeError):
    """A contract failure that must make the probe exit non-zero."""


def _load_replay(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            replay = json.load(handle)
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as error:
        raise ProbeError(f"cannot read replay {path}: {type(error).__name__}: {error}") from error

    if not isinstance(replay, dict):
        raise ProbeError("replay root must be an object")
    config = replay.get("configuration")
    steps = replay.get("steps")
    if not isinstance(config, dict):
        raise ProbeError("replay.configuration must be an object")
    if not isinstance(steps, list) or not steps:
        raise ProbeError("replay.steps must contain a first boundary")
    first = steps[0]
    if not isinstance(first, list) or len(first) < 2:
        raise ProbeError("replay.steps[0] must contain both seats")

    observations: list[dict[str, Any]] = []
    for seat in (0, 1):
        record = first[seat]
        if not isinstance(record, dict) or not isinstance(record.get("observation"), dict):
            raise ProbeError(f"replay.steps[0][{seat}].observation must be an object")
        observation = record["observation"]
        if type(observation.get("step")) is not int or observation["step"] != 0:
            raise ProbeError(
                f"seat {seat} first observation is step {observation.get('step')!r}, not step 0"
            )
        if type(observation.get("player")) is not int or observation["player"] != seat:
            raise ProbeError(
                f"seat {seat} observation.player is {observation.get('player')!r}"
            )
        observations.append(observation)
    return config, observations


def _positive_timeout(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbeError(f"{label} must be a positive finite number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ProbeError(f"{label} must be a positive finite number")
    return timeout


def _configured_timeouts(
    config: dict[str, Any],
    act_override: float | None,
    agent_override: float | None,
) -> tuple[float, float]:
    act_value: Any = act_override if act_override is not None else config.get("actTimeout")
    agent_value: Any = (
        agent_override
        if agent_override is not None
        else config.get("agentTimeout", DEFAULT_AGENT_TIMEOUT)
    )
    return (
        _positive_timeout(act_value, "configuration.actTimeout"),
        _positive_timeout(agent_value, "configuration.agentTimeout"),
    )


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract only ordinary in-tree files/directories; reject ambiguous archives."""
    try:
        with tarfile.open(archive, "r:*") as bundle:
            members = bundle.getmembers()
            if not members:
                raise ProbeError("archive is empty")
            seen: set[str] = set()
            root = destination.resolve()
            for member in members:
                raw = member.name
                if "\\" in raw:
                    raise ProbeError(f"archive member uses backslash path: {raw!r}")
                name = PurePosixPath(raw)
                if name.is_absolute() or ".." in name.parts or not name.parts:
                    raise ProbeError(f"archive member escapes extraction root: {raw!r}")
                normalized = name.as_posix()
                if normalized in seen:
                    raise ProbeError(f"duplicate archive member: {normalized!r}")
                seen.add(normalized)
                if not (member.isdir() or member.isfile()):
                    raise ProbeError(f"links/devices are not allowed in archive: {raw!r}")
                target = (destination / normalized).resolve()
                try:
                    target.relative_to(root)
                except ValueError as error:
                    raise ProbeError(f"archive member escapes extraction root: {raw!r}") from error
            # We validated every member above.  Omit the newer ``filter=`` API
            # so the same file runs on the pinned hosted Python 3.11 runtime.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                bundle.extractall(destination, members=members)
    except ProbeError:
        raise
    except (OSError, EOFError, tarfile.TarError) as error:
        raise ProbeError(f"cannot extract archive: {type(error).__name__}: {error}") from error

    main = destination / "main.py"
    if not main.is_file():
        raise ProbeError("archive must contain one regular root main.py")


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _validate_unit(command: Any, label: str) -> None:
    if not isinstance(command, list) or not command or not isinstance(command[0], str):
        raise ProbeError(f"{label} must be a non-empty command list")
    op = command[0]
    if op in NOARG_UNIT_OPS:
        if len(command) != 1:
            raise ProbeError(f"{label} operation {op} takes no arguments")
        return
    if op == "PLANT":
        if len(command) != 2 or command[1] not in CROPS:
            raise ProbeError(f"{label} PLANT requires one known crop")
        return
    if op in {"PICKUP", "PLACE"}:
        if len(command) not in {2, 3} or command[1] not in ITEMS:
            raise ProbeError(f"{label} {op} requires a known item and optional quantity")
        if len(command) == 3 and not _positive_int(command[2]):
            raise ProbeError(f"{label} {op} quantity must be a positive integer")
        return
    raise ProbeError(f"{label} has unknown unit operation {op!r}")


def _validate_market(order: Any, index: int) -> None:
    label = f"market[{index}]"
    if not isinstance(order, list) or not order or not isinstance(order[0], str):
        raise ProbeError(f"{label} must be a non-empty command list")
    op = order[0]
    if op in NOARG_MARKET_OPS:
        if len(order) != 1:
            raise ProbeError(f"{label} operation {op} takes no arguments")
        return
    domains = {
        "BUY_SEED": CROPS,
        "BUY_ANIMAL": ANIMALS,
        "BUY_PRODUCT": PRODUCTS,
        "SELL": PRODUCTS,
    }
    domain = domains.get(op)
    if domain is None:
        raise ProbeError(f"{label} has unknown market operation {op!r}")
    if len(order) != 3 or order[1] not in domain or not _nonnegative_int(order[2]):
        raise ProbeError(
            f"{label} {op} requires a known item and non-negative integer quantity"
        )


def _hand_count(observation: dict[str, Any]) -> int:
    player = observation.get("player")
    farms = observation.get("farms")
    if type(player) is not int or not isinstance(farms, list) or not (0 <= player < len(farms)):
        raise ProbeError("observation lacks a usable player/farms shape")
    farm = farms[player]
    if not isinstance(farm, dict) or not isinstance(farm.get("hands"), list):
        raise ProbeError("own farm.hands must be a list")
    return len(farm["hands"])


def validate_action(action: Any, observation: dict[str, Any], config: dict[str, Any]) -> None:
    if not isinstance(action, dict):
        raise ProbeError(f"action must be an object, got {type(action).__name__}")
    expected_keys = {"farmer", "hands", "market"}
    if set(action) != expected_keys:
        raise ProbeError(
            f"action keys must be exactly {sorted(expected_keys)}, got {sorted(map(str, action))}"
        )

    farmer = action["farmer"]
    hands = action["hands"]
    market = action["market"]
    if not isinstance(hands, list):
        raise ProbeError("action.hands must be a list")
    expected_hands = _hand_count(observation)
    if len(hands) != expected_hands:
        raise ProbeError(f"action has {len(hands)} hand commands; observation has {expected_hands}")
    if not isinstance(market, list):
        raise ProbeError("action.market must be a list")

    cap_value = config.get("maxMarketOrdersPerTurn", 10)
    if isinstance(cap_value, bool) or not isinstance(cap_value, int) or cap_value < 0:
        raise ProbeError("configuration.maxMarketOrdersPerTurn must be a non-negative integer")
    if len(market) > cap_value:
        raise ProbeError(f"action has {len(market)} market orders; cap is {cap_value}")

    _validate_unit(farmer, "farmer")
    for index, command in enumerate(hands):
        _validate_unit(command, f"hands[{index}]")
    for index, order in enumerate(market):
        _validate_market(order, index)

    effective_market = [
        order
        for order in market
        if order[0] != "NOOP" and (len(order) == 1 or order[2] > 0)
    ]
    if farmer[0] == "PASS" and all(command[0] == "PASS" for command in hands) and not effective_market:
        raise ProbeError("whole-turn PASS fallback is forbidden by the step-0 probe contract")

    # Prove strict JSON-wire serializability (not merely Python object shape).
    try:
        json.dumps(action, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ProbeError(f"action is not strict JSON: {error}") from error


class _PhaseTimeout(TimeoutError):
    pass


def _run_timed(label: str, timeout: float, function: Any) -> Any:
    def handler(signum: int, frame: Any) -> None:
        del signum, frame
        raise _PhaseTimeout(f"{label} exceeded {timeout:.6f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    started = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        result = function()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
    elapsed = time.monotonic() - started
    if elapsed >= timeout:
        # A candidate can accidentally catch BaseException/TimeoutError.  Wall time
        # remains authoritative even if it swallowed our signal exception.
        raise ProbeError(f"{label} returned after {elapsed:.6f}s; limit is {timeout:.6f}s")
    return result


def _import_and_call(
    root: Path,
    observation: dict[str, Any],
    config: dict[str, Any],
    act_timeout: float,
    agent_timeout: float,
) -> Any:
    main_path = root / "main.py"
    spec = importlib.util.spec_from_file_location("step0_candidate_main", main_path)
    if spec is None or spec.loader is None:
        raise ProbeError("cannot construct import spec for archive main.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    bank = observation.get("remainingOverageTime")
    has_bank = bank is not None
    if has_bank and (type(bank) not in (int, float) or not math.isfinite(bank) or bank < 0):
        raise ProbeError("remainingOverageTime must be a finite non-negative number")
    if has_bank and bank <= 0:
        raise ProbeError("no initial overage bank remains for cold archive import")
    init_limit = min(agent_timeout, bank) if has_bank else agent_timeout
    start = time.monotonic()
    _run_timed("archive initialization", init_limit, lambda: spec.loader.exec_module(module))
    imported = time.monotonic() - start
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise ProbeError("archive main.py has no callable agent")
    # Lazy weight loading is part of agent(), not module import. Both phases
    # consume the SAME observed bank; action gets the free per-call allowance.
    limit = act_timeout + bank - imported if has_bank else act_timeout
    if limit <= 0:
        raise ProbeError("archive initialization exhausted the observed bank")
    start = time.monotonic()
    action = _run_timed("agent()", limit, lambda: agent(observation, config))
    action_seconds = time.monotonic() - start
    status = getattr(module, "_PBC_STATUS", None)
    if status is not None:
        if not isinstance(status, dict) or status.get("last_fallback") is not False or status.get("fallbacks") != 0:
            raise ProbeError("candidate reports a PBC fallback or malformed status")
    used = imported + max(0.0, action_seconds - act_timeout)
    if has_bank and used > bank:
        raise ProbeError("cold step exceeded its shared import/action time bank")
    return {"action": action, "pbc_status": status,
            "timing": {"model": "hosted-bank" if has_bank else "act-only-no-bank",
                       "import_s": imported, "action_s": action_seconds,
                       "act_allowance_s": act_timeout, "initial_bank_s": bank,
                       "used_bank_s": used,
                       "remaining_bank_s": bank - used if has_bank else None}}


def _write_result(path: Path, seat: int, action: Any) -> None:
    payload = {"seat": seat, **action}
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _child(
    root: Path,
    replay_path: Path,
    seat: int,
    result_path: Path,
    act_timeout: float,
    agent_timeout: float,
) -> int:
    config, observations = _load_replay(replay_path)
    if seat not in (0, 1):
        raise ProbeError(f"invalid seat {seat}")
    action = _import_and_call(
        root, observations[seat], config, act_timeout, agent_timeout
    )
    _write_result(result_path, seat, action)
    return 0


def _run_seat(
    script: Path,
    root: Path,
    replay_path: Path,
    seat: int,
    result_path: Path,
    act_timeout: float,
    agent_timeout: float,
) -> Any:
    command = [
        sys.executable,
        "-B",
        str(script),
        "--child",
        str(root),
        str(replay_path),
        str(seat),
        str(result_path),
        repr(act_timeout),
        repr(agent_timeout),
    ]
    _, observations = _load_replay(replay_path)
    bank = observations[seat].get("remainingOverageTime")
    deadline = (float(bank) + act_timeout + 2.0 if type(bank) in (int, float)
                and math.isfinite(bank) and bank > 0 else agent_timeout + act_timeout + 2.0)
    timed_out = False
    with tempfile.TemporaryFile() as child_stdout, tempfile.TemporaryFile() as child_stderr:
        process = subprocess.Popen(
            command,
            stdout=child_stdout,
            stderr=child_stderr,
            start_new_session=True,
            cwd=root,
            env={**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        try:
            process.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        finally:
            # A candidate may spawn a descendant that inherits output descriptors.
            # Reap the entire private process group even when the seat leader exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        def read_log(handle: Any, limit: int = 65536) -> str:
            handle.seek(0)
            payload = handle.read(limit + 1)
            suffix = b"\n...[truncated]" if len(payload) > limit else b""
            return (payload[:limit] + suffix).decode("utf-8", errors="replace")

        stdout = read_log(child_stdout)
        stderr = read_log(child_stderr)

    if stdout:
        print(f"[seat {seat} child stdout]", file=sys.stderr)
        print(stdout, end="" if stdout.endswith("\n") else "\n", file=sys.stderr)
    if stderr:
        print(f"[seat {seat} child stderr]", file=sys.stderr)
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
    if timed_out:
        raise ProbeError(
            f"seat {seat} exceeded initialization+action process deadline "
            f"({agent_timeout:.3f}s + {act_timeout:.3f}s + 2s overhead)"
        )
    if process.returncode != 0:
        raise ProbeError(f"seat {seat} child exited {process.returncode}")
    if not result_path.is_file():
        raise ProbeError(
            f"seat {seat} child exited 0 without a result (SystemExit/os._exit/import abort)"
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProbeError(f"seat {seat} result is unreadable: {error}") from error
    if not isinstance(payload, dict) or payload.get("seat") != seat or "action" not in payload:
        raise ProbeError(f"seat {seat} result envelope is malformed")
    return payload


def _parent(
    archive: Path,
    replay_path: Path,
    act_timeout_override: float | None,
    agent_timeout_override: float | None,
    receipt_path: Path | None = None,
) -> int:
    config, observations = _load_replay(replay_path)
    act_timeout, agent_timeout = _configured_timeouts(
        config, act_timeout_override, agent_timeout_override
    )
    if not archive.is_file():
        raise ProbeError(f"archive does not exist: {archive}")

    archive, replay_path = archive.resolve(), replay_path.resolve()
    script = Path(__file__).resolve()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    archive_sha, replay_sha = digest(archive), digest(replay_path)
    receipts = []
    with tempfile.TemporaryDirectory(prefix="step0-wire-") as td:
        root = Path(td) / "archive"
        root.mkdir()
        _safe_extract(archive, root)
        for seat in (0, 1):
            result_path = Path(td) / f"seat-{seat}.json"
            receipt = _run_seat(
                script, root, replay_path, seat, result_path, act_timeout, agent_timeout
            )
            action = receipt["action"]
            validate_action(action, observations[seat], config)
            receipts.append(receipt)
            encoded = json.dumps(action, separators=(",", ":"), allow_nan=False)
            print(seat, encoded[:400])
    if digest(archive) != archive_sha or digest(replay_path) != replay_sha:
        raise ProbeError("archive/replay bytes changed during certification")
    certificate = {"schema": "plain-v9-step0-v1", "complete": True,
                   "archive_sha256": archive_sha, "replay_sha256": replay_sha,
                   "probe_sha256": digest(script), "seats": receipts}
    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
        tmp.write_text(json.dumps(certificate, indent=2, allow_nan=False) + "\n")
        os.replace(tmp, receipt_path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path)
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path(os.environ.get("STEP0_REPLAY", DEFAULT_REPLAY)),
        help="step-0 replay fixture (default: STEP0_REPLAY or production path)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="override free per-action allowance (observed overage is additional)",
    )
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=None,
        help=(
            "override configuration.agentTimeout for archive initialization "
            f"(default if absent: {DEFAULT_AGENT_TIMEOUT:g}s)"
        ),
    )
    parser.add_argument("--out", type=Path, help="write a SHA-bound JSON receipt on success")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("child_args", nargs="*", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.child:
            # With argparse's positional layout, --child consumes the remaining
            # values through archive + child_args.  Reassemble them explicitly.
            values = ([] if args.archive is None else [str(args.archive)]) + list(args.child_args)
            if len(values) != 6:
                raise ProbeError(
                    "internal child invocation requires "
                    "ROOT REPLAY SEAT RESULT ACT_TIMEOUT AGENT_TIMEOUT"
                )
            root, replay, seat_text, result, act_text, agent_text = values
            return _child(
                Path(root),
                Path(replay),
                int(seat_text),
                Path(result),
                float(act_text),
                float(agent_text),
            )
        if args.archive is None or args.child_args:
            raise ProbeError(
                "usage: step0_wire.py ARCHIVE "
                "[--replay PATH] [--timeout SEC] [--agent-timeout SEC]"
            )
        return _parent(args.archive, args.replay, args.timeout, args.agent_timeout, args.out)
    except BaseException as error:
        # Parent must convert even SystemExit(0) raised by our own helper into a
        # non-zero contract result.  KeyboardInterrupt remains visible as 130.
        if isinstance(error, KeyboardInterrupt):
            print("step-0 probe interrupted", file=sys.stderr)
            return 130
        if isinstance(error, SystemExit):
            code = error.code
            print(f"step-0 probe aborted by SystemExit({code!r})", file=sys.stderr)
            return 1
        print(f"step-0 probe FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
