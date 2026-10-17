#!/usr/bin/env python3
"""Atomically pin a checkpoint by basename without ever reusing different bytes.

Usage: pin_checkpoint.py SOURCE KEEP_DIR
Prints the stable destination path on success. Standard library only.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pin(source: Path, keep: Path) -> Path:
    if not source.is_file():
        raise RuntimeError(f"source checkpoint is not a file: {source}")
    keep.mkdir(parents=True, exist_ok=True)
    destination = keep / source.name
    source_before = sha256(source)

    if destination.exists():
        destination_sha = sha256(destination)
        if destination_sha != source_before:
            raise RuntimeError(
                "checkpoint basename collision: "
                f"source={source} sha256={source_before} "
                f"stable={destination} sha256={destination_sha}"
            )
        return destination

    fd, temporary_name = tempfile.mkstemp(prefix=f".{source.name}.", suffix=".tmp", dir=keep)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
            shutil.copyfileobj(inp, out, length=1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        copied_sha = sha256(temporary)
        source_after = sha256(source)
        if copied_sha != source_before or source_after != source_before:
            raise RuntimeError(
                f"source checkpoint changed while pinning: before={source_before} "
                f"after={source_after} copied={copied_sha}"
            )
        try:
            os.link(temporary, destination)  # atomic no-clobber publication on KEEP's filesystem
        except FileExistsError:
            destination_sha = sha256(destination)
            if destination_sha != source_before:
                raise RuntimeError(
                    "checkpoint basename collision during concurrent pin: "
                    f"source={source} sha256={source_before} "
                    f"stable={destination} sha256={destination_sha}"
                )
        if sha256(destination) != source_before:
            raise RuntimeError(f"pinned checkpoint hash mismatch: {destination}")
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: pin_checkpoint.py SOURCE KEEP_DIR", file=sys.stderr)
        return 64
    try:
        destination = pin(Path(argv[1]), Path(argv[2]))
    except Exception as exc:  # concise shell-facing failure
        print(f"pin checkpoint failed: {exc}", file=sys.stderr)
        return 2
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
