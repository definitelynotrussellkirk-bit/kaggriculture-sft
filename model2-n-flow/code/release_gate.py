#!/usr/bin/env python3
"""Fail-closed final gate for one N-flow release directory.

Requires the full link receipt, cold/HUB receipt, P2 check, and parity check to
agree. Writes RELEASE-DONE.json atomically only after every condition passes.
Standard library only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path, issues: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        issues.append(f"{path.name}: unreadable JSON: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(value, dict):
        issues.append(f"{path.name}: root is not an object")
        return {}
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_release(out: Path, name: str) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    links_path = out / f"LINKS-{name}.json"
    links = load_json(links_path, issues)
    hub = load_json(out / "HUB-DONE.json", issues)
    p2 = load_json(out / "P2-CHECK.json", issues)

    if links.get("name") != name:
        issues.append(f"LINKS name {links.get('name')!r} != {name!r}")
    if links.get("all_pass") is not True:
        issues.append(f"LINKS all_pass is {links.get('all_pass')!r}, not true")
    link_rows = links.get("links")
    if not isinstance(link_rows, dict) or not link_rows:
        issues.append("LINKS links is missing or empty")
        link_rows = {}
    for key, row in sorted(link_rows.items()):
        if not isinstance(row, dict) or row.get("pass") is not True:
            detail = row.get("detail") if isinstance(row, dict) else row
            issues.append(f"link {key} failed: {json.dumps(detail, sort_keys=True)}")

    expected_hub = {
        "name": name,
        "complete": True,
        "fallbacks": 0,
        "sha_match": True,
    }
    for key, expected in expected_hub.items():
        if hub.get(key) != expected:
            issues.append(f"HUB-DONE {key}={hub.get(key)!r}, expected {expected!r}")
    if p2.get("pass") is not True:
        issues.append(f"P2-CHECK pass is {p2.get('pass')!r}, not true")
    if p2.get("effective_binds_checkpoint") is not True:
        issues.append(
            f"P2-CHECK effective_binds_checkpoint is {p2.get('effective_binds_checkpoint')!r}, not true"
        )

    export_row = link_rows.get("export") if isinstance(link_rows, dict) else None
    export_detail = (
        export_row.get("detail")
        if isinstance(export_row, dict) and isinstance(export_row.get("detail"), dict)
        else {}
    )
    checkpoint_hashes = {
        "HUB-DONE": hub.get("checkpoint_sha256"),
        "P2-CHECK": p2.get("checkpoint_sha256"),
        "LINKS export": export_detail.get("checkpoint_sha256"),
    }
    if (any(not isinstance(value, str) or not value for value in checkpoint_hashes.values())
            or len(set(checkpoint_hashes.values())) != 1):
        issues.append(f"checkpoint sha256 mismatch: {json.dumps(checkpoint_hashes, sort_keys=True)}")

    archive_path = out / f"{name}.tar.gz"
    try:
        actual_archive_sha = sha256(archive_path)
        actual_archive_bytes = archive_path.stat().st_size
    except Exception as exc:
        issues.append(f"archive unreadable: {type(exc).__name__}: {exc}")
        actual_archive_sha = None
        actual_archive_bytes = None
    archive_row = link_rows.get("archive") if isinstance(link_rows, dict) else None
    archive_detail = archive_row.get("detail") if isinstance(archive_row, dict) and isinstance(archive_row.get("detail"), dict) else {}
    expected_archive_sha = {
        "HUB-DONE": hub.get("archive_sha256"),
        "LINKS": archive_detail.get("archive_sha256"),
    }
    for source, expected in expected_archive_sha.items():
        if expected != actual_archive_sha:
            issues.append(f"archive sha256 mismatch: actual={actual_archive_sha!r}, {source}={expected!r}")
    if archive_detail.get("archive_bytes") != actual_archive_bytes:
        issues.append(
            f"archive size mismatch: actual={actual_archive_bytes!r}, LINKS={archive_detail.get('archive_bytes')!r}"
        )

    parity_path = out / "PARITY.txt"
    try:
        parity_lines = [line.strip() for line in parity_path.read_text().splitlines() if line.strip()]
    except Exception as exc:
        issues.append(f"PARITY.txt unreadable: {type(exc).__name__}: {exc}")
        parity_lines = []
    if not parity_lines or parity_lines[-1] != "PASS":
        issues.append(f"PARITY final line is {parity_lines[-1] if parity_lines else None!r}, not 'PASS'")

    detail = {
        "name": name,
        "all_pass": not issues,
        "links_sha256": sha256(links_path) if links_path.is_file() else None,
        "archive_sha256": actual_archive_sha,
        "archive_bytes": actual_archive_bytes,
        "checkpoint": hub.get("checkpoint"),
        "checkpoint_sha256": hub.get("checkpoint_sha256"),
        "issues": issues,
    }
    return issues, detail


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: release_gate.py OUT_DIR NAME", file=sys.stderr)
        return 64
    out, name = Path(argv[1]), argv[2]
    issues, detail = check_release(out, name)
    if issues:
        (out / "RELEASE-DONE.json").unlink(missing_ok=True)
        print(json.dumps(detail, indent=1, sort_keys=True))
        return 2
    detail["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    destination = out / "RELEASE-DONE.json"
    temporary = out / "RELEASE-DONE.json.tmp"
    temporary.write_text(json.dumps(detail, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(detail, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
