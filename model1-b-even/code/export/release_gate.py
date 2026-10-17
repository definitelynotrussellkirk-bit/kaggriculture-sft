#!/usr/bin/env python3
"""Fail-closed offline gate for a plain-v9 archive and its verification receipts.

This tool deliberately distinguishes two questions:
  * are the bytes/receipts presently self-consistent? (``check``), and
  * is the builder source itself fail-closed? (``audit-builder``).

The legacy STEP0.txt can be parsed with ``--allow-legacy-step0``, but it is not
cryptographically bound to the archive. Strict mode rejects it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import shlex
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from typing import Any


PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_name(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and ".." not in p.parts and "" not in p.parts


def json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def action_shape(action: Any) -> bool:
    if not isinstance(action, dict) or set(action) != {"farmer", "hands", "market"}:
        return False
    farmer, hands, market = action["farmer"], action["hands"], action["market"]
    if not isinstance(farmer, list) or not farmer or not isinstance(farmer[0], str):
        return False
    if not isinstance(hands, list) or not isinstance(market, list) or len(market) > 10:
        return False
    if any(not isinstance(row, list) or not row or not isinstance(row[0], str) for row in hands):
        return False
    if any(not isinstance(row, list) or not row or not isinstance(row[0], str) for row in market):
        return False
    return True


def read_archive(path: Path, *, fileobj=None) -> tuple[dict[str, Any], dict[str, str]]:
    seen: set[str] = set()
    payloads: dict[str, bytes] = {}
    with tarfile.open(path if fileobj is None else None, "r:*", fileobj=fileobj) as archive:
        for member in archive.getmembers():
            require(safe_member_name(member.name), f"unsafe archive member: {member.name!r}")
            require(member.name not in seen, f"duplicate archive member: {member.name}")
            seen.add(member.name)
            require(member.isfile(), f"non-regular archive member is forbidden: {member.name}")
            handle = archive.extractfile(member)
            require(handle is not None, f"cannot read archive member: {member.name}")
            payloads[member.name] = handle.read()
    require("MANIFEST.json" in payloads, "archive lacks MANIFEST.json")
    manifest = json.loads(payloads["MANIFEST.json"])
    require(isinstance(manifest, dict), "MANIFEST.json is not an object")
    declared = manifest.get("members")
    require(isinstance(declared, dict), "manifest.members is not an object")
    actual_names = set(payloads) - {"MANIFEST.json"}
    require(set(declared) == actual_names,
            f"manifest inventory mismatch: missing={sorted(actual_names-set(declared))}, extra={sorted(set(declared)-actual_names)}")
    hashes: dict[str, str] = {}
    for name in sorted(actual_names):
        details = declared[name]
        require(isinstance(details, dict), f"manifest entry is not an object: {name}")
        observed = hashlib.sha256(payloads[name]).hexdigest()
        hashes[name] = observed
        require(details.get("sha256") == observed, f"manifest hash mismatch: {name}")
        require(details.get("bytes") == len(payloads[name]), f"manifest byte count mismatch: {name}")
    require("main.py" in actual_names and "policy.pt" in actual_names, "archive lacks main.py or policy.pt")
    return manifest, hashes


def _zero(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value)) and float(value) == 0.0


def _nonnegative_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value)) and float(value) >= 0.0


def _sha256_text(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def check_step0_status(status: Any, seat: Any) -> None:
    require(isinstance(status, dict), f"structured STEP0 seat {seat} lacks pbc_status")
    require(type(status.get("calls")) is int and status.get("calls") == 1,
            f"structured STEP0 seat {seat} is not exactly one fresh-process call")
    require(_zero(status.get("fallbacks")) and status.get("last_fallback") is False,
            f"structured STEP0 seat {seat} reports a fallback")
    require(status.get("last_error") is None and status.get("import_error") is None,
            f"structured STEP0 seat {seat} reports an actor/import error")
    require(_nonnegative_number(status.get("load_ms")),
            f"structured STEP0 seat {seat} has invalid load_ms")


def check_step0_timing(timing: Any, seat: Any) -> dict[str, float]:
    require(isinstance(timing, dict) and timing.get("model") == "hosted-bank",
            f"structured STEP0 seat {seat} lacks hosted-bank timing")
    keys = ("import_s", "action_s", "act_allowance_s", "initial_bank_s",
            "used_bank_s", "remaining_bank_s")
    require(all(_nonnegative_number(timing.get(key)) for key in keys),
            f"structured STEP0 seat {seat} has invalid timing values")
    values = {key: float(timing[key]) for key in keys}
    require(values["act_allowance_s"] > 0.0 and values["initial_bank_s"] > 0.0,
            f"structured STEP0 seat {seat} has a nonpositive timing allowance/bank")
    # ASTRA-2 G1-BYTES #2 (coordinator 08:1x): the hosted clock is fixed; a receipt may not declare a looser one.
    require(values["act_allowance_s"] == 1.0 and values["initial_bank_s"] == 60.0,
            f"structured STEP0 seat {seat} does not use the required 1s/60s clock")
    expected_used = max(0.0, values["import_s"] + values["action_s"] - values["act_allowance_s"])
    require(math.isclose(values["used_bank_s"], expected_used, rel_tol=1e-9, abs_tol=1e-6),
            f"structured STEP0 seat {seat} used-bank arithmetic is inconsistent")
    require(values["used_bank_s"] <= values["initial_bank_s"] + 1e-9,
            f"structured STEP0 seat {seat} exhausts the hosted timing bank")
    expected_remaining = values["initial_bank_s"] - values["used_bank_s"]
    require(math.isclose(values["remaining_bank_s"], expected_remaining, rel_tol=1e-9, abs_tol=1e-6),
            f"structured STEP0 seat {seat} remaining-bank arithmetic is inconsistent")
    return values


def expected_decode(value: str) -> dict[str, Any]:
    """Normalize the builder's documented ``greedy | JSON`` decode argument."""
    if value == "greedy":
        return {"mode": "greedy"}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise GateError(f"expected decode is neither 'greedy' nor JSON: {error}") from error
    require(isinstance(parsed, dict), "expected decode JSON must be an object")
    return parsed


def check_declared_runtime(manifest: dict[str, Any], expected_decode_text: str) -> dict[str, Any]:
    """Fail if generated runtime settings differ from the requested release contract."""
    require(manifest.get("entrypoint") == "main.py:agent", "unexpected archive entrypoint")
    declared = manifest.get("declared")
    require(isinstance(declared, dict), "manifest.declared is not an object")
    requested = expected_decode(expected_decode_text)
    require(declared.get("decode") == requested,
            f"archive decode differs from requested decode: {declared.get('decode')!r} != {requested!r}")
    require(declared.get("device") == "cpu", "archive device is not cpu")
    require(declared.get("fallback") == "PASS on any exception", "archive fallback contract drifted")
    require(type(declared.get("seed")) is int and declared.get("seed") == 0, "archive seed is not integer 0")
    require(type(declared.get("threads")) is int and declared.get("threads") == 1,
            "archive thread count is not integer 1")
    temperature = declared.get("temperature")
    require(type(temperature) in (int, float) and math.isfinite(float(temperature))
            and float(temperature) == 0.0, "archive temperature is not finite zero")
    return {"entrypoint": manifest["entrypoint"], "decode": requested, "device": "cpu",
            "seed": 0, "temperature": 0.0, "threads": 1,
            "fallback": "PASS on any exception"}


def check_source_pins(manifest: dict[str, Any], pins: dict[str, Any]) -> int:
    require(isinstance(pins, dict), "source pins must be a JSON object")
    require(pins.get("builder_sha256") == manifest.get("builder_sha256"),
            "archive builder hash differs from reviewed source pin")
    expected = pins.get("members")
    declared = manifest.get("members")
    require(isinstance(expected, dict) and isinstance(declared, dict),
            "source pins and manifest members must be objects")
    source_names = set(declared) - {"policy.pt"}
    require(set(expected) == source_names,
            f"source pin inventory mismatch: missing={sorted(source_names-set(expected))}, extra={sorted(set(expected)-source_names)}")
    for name in sorted(source_names):
        require(isinstance(expected[name], str), f"source pin for {name} is not a SHA string")
        details = declared[name]
        require(isinstance(details, dict) and details.get("sha256") == expected[name],
                f"archive source differs from reviewed pin: {name}")
    return len(source_names)


def check_verify_timing(timing: Any, index: int) -> dict[str, float]:
    require(isinstance(timing, dict), f"VERIFY.games[{index}] lacks timing_budget")
    require(timing.get("valid") is True and timing.get("within_bank") is True
            and timing.get("first_exhausted_step") is None,
            f"VERIFY.games[{index}] violates the hosted timing bank")
    require(type(timing.get("calls")) is int and timing.get("calls") == 719,
            f"VERIFY.games[{index}] timing budget does not cover all 719 calls")
    keys = ("act_ms", "import_ms", "initial_bank_ms", "remaining_bank_ms", "used_bank_ms")
    require(all(_nonnegative_number(timing.get(key)) for key in keys),
            f"VERIFY.games[{index}] has invalid timing values")
    values = {key: float(timing[key]) for key in keys}
    require(values["act_ms"] > 0.0 and values["initial_bank_ms"] > 0.0,
            f"VERIFY.games[{index}] has a nonpositive allowance/bank")
    require(values["act_ms"] == 1000.0 and values["initial_bank_ms"] == 60000.0,
            f"VERIFY.games[{index}] does not use the required 1s/60s clock")
    require(values["used_bank_ms"] <= values["initial_bank_ms"] + 1e-6,
            f"VERIFY.games[{index}] used bank exceeds its initial bank")
    expected_remaining = values["initial_bank_ms"] - values["used_bank_ms"]
    require(math.isclose(values["remaining_bank_ms"], expected_remaining,
                         rel_tol=1e-9, abs_tol=1e-6),
            f"VERIFY.games[{index}] timing-bank arithmetic is inconsistent")
    return values


def check_verify(receipt: dict[str, Any], archive_sha: str) -> dict[str, Any]:
    require(receipt.get("archive_sha256") == archive_sha, "VERIFY archive SHA differs from current archive")
    require(receipt.get("complete") is True, "VERIFY.complete is not true")
    require(_zero(receipt.get("fallbacks")), "VERIFY.fallbacks is not zero")
    games = receipt.get("games")
    require(isinstance(games, list) and len(games) >= 2, "VERIFY must contain at least two cold games")
    pairs: dict[tuple[Any, Any], set[int]] = {}
    for index, game in enumerate(games):
        require(isinstance(game, dict), f"VERIFY.games[{index}] is not an object")
        require(game.get("done") is True and game.get("steps") == 719,
                f"VERIFY.games[{index}] is not a complete 719-step game")
        require(game.get("statuses") == ["DONE", "DONE"], f"VERIFY.games[{index}] statuses are not both DONE")
        require(_zero(game.get("fallbacks")), f"VERIFY.games[{index}] has fallbacks")
        cold = game.get("cold")
        require(isinstance(cold, dict) and cold.get("complete") is True and cold.get("engine_complete") is True,
                f"VERIFY.games[{index}] cold process incomplete")
        require(_zero(cold.get("hard_errors")) and _zero(cold.get("log_fallback_lines")),
                f"VERIFY.games[{index}] cold process has hard/fallback errors")
        require(cold.get("log_steps_contiguous") is True and cold.get("log_lines") == 719,
                f"VERIFY.games[{index}] actor log is incomplete")
        timing = cold.get("timing_budget")
        check_verify_timing(timing, index)
        require(_nonnegative_number(game.get("first_step_ms")),
                f"VERIFY.games[{index}] has invalid first_step_ms")
        closed = cold.get("closed")
        require(isinstance(closed, dict) and closed.get("closed") is True
                and closed.get("our_modules_outside_archive") == [] and cold.get("close_error") is None,
                f"VERIFY.games[{index}] did not close archive/module provenance")
        seat = game.get("our_seat")
        require(type(seat) is int and seat in (0, 1), f"VERIFY.games[{index}] has invalid seat")
        require(isinstance(game.get("opponent"), str) and bool(game["opponent"].strip()),
                f"VERIFY.games[{index}] lacks opponent identity")
        require(type(game.get("world_seed")) is int and game["world_seed"] >= 0,
                f"VERIFY.games[{index}] lacks an integer world identity")
        key = (game["opponent"], game["world_seed"])
        require(seat not in pairs.setdefault(key, set()), f"duplicate seat for pair {key}")
        pairs[key].add(seat)
    require(bool(pairs) and all(seats == {0, 1} for seats in pairs.values()),
            "VERIFY contains an incomplete same-world both-seat pair")
    return {
        "games": len(games),
        "pairs": sum(seats == {0, 1} for seats in pairs.values()),
        "max_used_bank_ms": max(float(g["cold"]["timing_budget"]["used_bank_ms"]) for g in games),
        "max_first_step_ms": max(float(g["first_step_ms"]) for g in games),
        "fallbacks": 0,
    }


def parse_step0(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        require(value.get("schema") == "plain-v9-step0-v1",
                "structured STEP0 has an unexpected schema")
        require(value.get("complete") is True, "structured STEP0.complete is not true")
        for name in ("archive_sha256", "replay_sha256", "probe_sha256"):
            require(_sha256_text(value.get(name)), f"structured STEP0 has invalid {name}")
        seats = value.get("seats")
        require(isinstance(seats, list) and len(seats) == 2
                and all(isinstance(row, dict) for row in seats)
                and {row.get("seat") for row in seats} == {0, 1},
                "structured STEP0 must contain exactly one row for seats 0 and 1")
        max_used_bank_s = 0.0
        for row in seats:
            seat = row.get("seat")
            action = row.get("action")
            require(action_shape(action), f"structured STEP0 seat {seat} has invalid action")
            require(action != PASS_ACTION,
                    f"structured STEP0 seat {seat} is a whole-turn PASS that can mask fallback")
            check_step0_status(row.get("pbc_status"), seat)
            timing = check_step0_timing(row.get("timing"), seat)
            max_used_bank_s = max(max_used_bank_s, timing["used_bank_s"])
        return {"structured": True, "bound_sha": value.get("archive_sha256"),
                "probe_sha256": value.get("probe_sha256"),
                "replay_sha256": value.get("replay_sha256"),
                "seats": seats, "max_used_bank_s": max_used_bank_s}

    seats: dict[int, dict[str, Any]] = {}
    for line in text.splitlines():
        match = re.match(r"^([01])\s+(\{.*\})\s*$", line)
        if not match:
            continue
        seat = int(match.group(1))
        require(seat not in seats, f"legacy STEP0 duplicates seat {seat}")
        action = json.loads(match.group(2))
        require(action_shape(action), f"legacy STEP0 seat {seat} action has invalid wire shape")
        require(action != PASS_ACTION,
                f"legacy STEP0 seat {seat} is a whole-turn PASS that can mask fallback")
        seats[seat] = action
    require(set(seats) == {0, 1}, "legacy STEP0 must contain exactly one action for seats 0 and 1")
    require("fb=1" not in text, "legacy STEP0 reports a fallback")
    return {"structured": False, "bound_sha": None,
            "probe_sha256": None, "replay_sha256": None,
            "seats": [{"seat": seat, "action": seats[seat]} for seat in (0, 1)]}


def check_step0_provenance(step0: dict[str, Any], probe_sha: str, replay_sha: str) -> None:
    """Bind a structured receipt to the exact probe code and replay fixture consumed now."""
    require(step0.get("structured") is True,
            "legacy STEP0 has no probe/replay identity to verify")
    require(_sha256_text(probe_sha) and _sha256_text(replay_sha),
            "expected step-0 probe/replay identities are invalid")
    require(step0.get("probe_sha256") == probe_sha,
            "structured STEP0 probe SHA differs from the current probe file")
    require(step0.get("replay_sha256") == replay_sha,
            "structured STEP0 replay SHA differs from the current replay fixture")


def check_hub(hub: dict[str, Any], archive_sha: str) -> None:
    require(hub.get("archive_sha256") == archive_sha, "HUB-DONE archive SHA differs from current archive")
    require(hub.get("sha_match") is True, "HUB-DONE.sha_match is not true")
    require(hub.get("complete") is True, "HUB-DONE.complete is not true")
    require(_zero(hub.get("fallbacks")), "HUB-DONE.fallbacks is not zero")


def audit_builder_text(text: str) -> list[str]:
    failures: list[str] = []
    if "RS_OVERRIDE" in text:
        failures.append("mutable RS_OVERRIDE source")
    if re.search(r"step0_wire\.py.*\|\|\s*true", text):
        failures.append("step-0 probe is fail-open")
    # A comment mentioning the filename is not an invocation. This remains a
    # conservative text audit, NOT proof of shell control flow or source identity.
    has_final_gate = False
    for line in text.replace("\\\n", " ").splitlines():
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        if not words:
            continue
        interpreter = words[0] == "$HOSTED" or Path(words[0]).name in ("python", "python3", "python3.11", "python3.12")
        invoked_gate = "$GATE" in words or any(Path(word).name == "release_gate.py" for word in words)
        if interpreter and invoked_gate and "check" in words and not ("||" in words or "&" in words):
            has_final_gate = True
    if '"sha_match"' in text and not has_final_gate:
        failures.append("HUB-DONE records sha_match without asserting it")
    if ('v.get("complete")' in text or "v.get('complete')" in text) and not has_final_gate:
        failures.append("HUB-DONE records complete/fallbacks without asserting them")
    if not has_final_gate:
        failures.append("no final fail-closed release gate")
    return failures


def _hash_handle(handle) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(value) -> tuple:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def check(args: argparse.Namespace) -> dict[str, Any]:
    # One no-follow FD supplies BOTH the digest and the tar contents. A pathname
    # swap cannot silently select a second object between those operations.
    archive = Path(args.archive).absolute()
    fd = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        require(stat.S_ISREG(before.st_mode), "archive is not a regular file")
        archive_sha = _hash_handle(handle)
        handle.seek(0)
        result = _check_opened(args, archive, handle, archive_sha, before.st_size)
        require(_hash_handle(handle) == archive_sha, "archive bytes changed during gate")
        require(_stat_identity(os.fstat(handle.fileno())) == _stat_identity(before),
                "archive metadata changed during gate")
        require(_stat_identity(archive.lstat()) == _stat_identity(before),
                "archive pathname changed identity during gate")
        return result


def _check_opened(args, archive, handle, archive_sha, archive_bytes) -> dict[str, Any]:
    manifest, hashes = read_archive(archive, fileobj=handle)
    declared_contract = None
    if args.expected_decode is not None:
        declared_contract = check_declared_runtime(manifest, args.expected_decode)
    source_pin_count = None
    if args.source_pins:
        source_pin_count = check_source_pins(manifest, json_file(Path(args.source_pins)))
    verify = json_file(Path(args.verify))
    verify_summary = check_verify(verify, archive_sha)
    step0 = parse_step0(Path(args.step0))
    bound = step0["bound_sha"] == archive_sha
    probe_verified = replay_verified = False
    if not args.allow_legacy_step0:
        require(step0["structured"] and bound,
                "STEP0 is legacy/unbound; rerun a structured probe that records the archive SHA")
        require(args.step0_probe and args.step0_replay,
                "strict STEP0 verification requires --step0-probe and --step0-replay")
    if args.step0_probe or args.step0_replay:
        require(args.step0_probe and args.step0_replay,
                "provide both --step0-probe and --step0-replay")
        probe_sha = sha256_file(Path(args.step0_probe))
        replay_sha = sha256_file(Path(args.step0_replay))
        check_step0_provenance(step0, probe_sha, replay_sha)
        probe_verified = replay_verified = True
    hub = json_file(Path(args.hub))
    check_hub(hub, archive_sha)
    builder_failures: list[str] = []
    if args.builder:
        builder_failures = audit_builder_text(Path(args.builder).read_text(encoding="utf-8"))
        require(not builder_failures, "; ".join(builder_failures))
    return {
        "schema": "plain-v9-release-gate-v1",
        "complete": True,
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "manifest_members": len(hashes),
        "source_pins_verified": source_pin_count,
        "entrypoint": manifest.get("entrypoint"),
        "parameters": manifest.get("parameters"),
        "declared": manifest.get("declared"),
        "declared_contract_verified": declared_contract,
        "verify": verify_summary,
        "step0_structured": step0["structured"],
        "step0_bound_to_archive": bound,
        "step0_probe_verified": probe_verified,
        "step0_replay_verified": replay_verified,
        "legacy_step0_allowed": bool(args.allow_legacy_step0),
        "hub_verified": True,
        "builder_verified": bool(args.builder),
        "builder_audit_scope": "heuristic_text_only",
        "warnings": ([] if bound else ["STEP0 actions are valid but the legacy text receipt is not bound to archive bytes"]),
        "upload_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-builder")
    audit.add_argument("builder")
    gate = sub.add_parser("check")
    gate.add_argument("--archive", required=True)
    gate.add_argument("--verify", required=True)
    gate.add_argument("--step0", required=True)
    gate.add_argument("--step0-probe", help="probe source whose SHA must match the structured receipt")
    gate.add_argument("--step0-replay", help="replay fixture whose SHA must match the structured receipt")
    gate.add_argument("--hub", required=True)
    gate.add_argument("--builder")
    gate.add_argument("--source-pins")
    gate.add_argument("--expected-decode",
                      help="required decoder contract: literal greedy or one JSON object")
    gate.add_argument("--allow-legacy-step0", action="store_true")
    gate.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        if args.command == "audit-builder":
            failures = audit_builder_text(Path(args.builder).read_text(encoding="utf-8"))
            result = {"schema": "plain-v9-builder-audit-v1", "complete": not failures,
                      "builder": os.fspath(Path(args.builder).resolve()), "failures": failures}
            code = 0 if not failures else 1
        else:
            result = check(args)
            code = 0
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if getattr(args, "out", None):
            Path(args.out).write_text(text, encoding="utf-8")
        print(text, end="")
        return code
    except (GateError, OSError, ValueError, TypeError, KeyError, tarfile.TarError) as error:
        result = {"schema": "plain-v9-release-gate-v1", "complete": False,
                  "error": f"{type(error).__name__}: {error}", "upload_authorized": False}
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if getattr(args, "out", None):
            Path(args.out).write_text(text, encoding="utf-8")
        print(text, end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
