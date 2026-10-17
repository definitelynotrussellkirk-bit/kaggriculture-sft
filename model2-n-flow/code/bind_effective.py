#!/usr/bin/env python3
"""Bind an EFFECTIVE/SPEC receipt to the exact checkpoint released.

Usage: bind_effective.py SOURCE_EFFECTIVE.json CHECKPOINT.pt OUT.json

The source receipt is copied, its own byte hash is retained, and exact release
checkpoint path/hash fields are added. A source receipt that already claims a
different checkpoint is rejected rather than overwritten. Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: bind_effective.py SOURCE_EFFECTIVE.json CHECKPOINT.pt OUT.json", file=sys.stderr)
        return 64
    source, checkpoint, out = map(Path, argv[1:])
    try:
        source_before = sha256(source)
        value: Any = json.loads(source.read_text())
        if not isinstance(value, dict) or not value or not value.get("cmd"):
            raise RuntimeError("source EFFECTIVE receipt is not a non-empty object with cmd")
        checkpoint_before = sha256(checkpoint)

        for prefix in ("final", "release"):
            claimed_path = value.get(f"{prefix}_checkpoint")
            claimed_sha = value.get(f"{prefix}_checkpoint_sha256")
            if claimed_path is None and claimed_sha is None:
                continue
            if not isinstance(claimed_path, str) or not isinstance(claimed_sha, str):
                raise RuntimeError(f"source receipt has incomplete {prefix}_checkpoint identity")
            if Path(claimed_path).name != checkpoint.name or claimed_sha != checkpoint_before:
                raise RuntimeError(
                    f"source receipt {prefix}_checkpoint contradicts release bytes: "
                    f"claimed={claimed_path} sha256={claimed_sha}, "
                    f"release={checkpoint} sha256={checkpoint_before}"
                )

        result = dict(value)
        result.update(
            source_effective=str(source),
            source_effective_sha256=source_before,
            release_checkpoint=str(checkpoint),
            release_checkpoint_sha256=checkpoint_before,
        )
        source_after = sha256(source)
        checkpoint_after = sha256(checkpoint)
        if source_after != source_before or checkpoint_after != checkpoint_before:
            raise RuntimeError("source receipt or checkpoint changed while binding")

        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_name(out.name + f".tmp-{os.getpid()}")
        try:
            temporary.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
            os.replace(temporary, out)
        finally:
            temporary.unlink(missing_ok=True)
    except Exception as exc:
        print(f"bind effective failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "effective": str(out),
        "source_effective_sha256": source_before,
        "release_checkpoint": str(checkpoint),
        "release_checkpoint_sha256": checkpoint_before,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
