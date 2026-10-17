"""Compare exported policy tensors/configs, ignoring torch serialization bytes."""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import torch


def load(path):
    with tarfile.open(path, "r:gz") as archive:
        member = archive.extractfile("policy.pt")
        if member is None:
            raise ValueError(f"{path}: no policy.pt")
        return torch.load(io.BytesIO(member.read()), map_location="cpu", weights_only=True)


def main():
    a, b = map(Path, sys.argv[1:3])
    x, y = load(a), load(b)
    if x.keys() != y.keys() or x["config"] != y["config"] or x["tensors"].keys() != y["tensors"].keys():
        raise AssertionError("config or tensor inventory differs")
    different = [k for k in x["tensors"] if not torch.equal(x["tensors"][k], y["tensors"][k])]
    report = {"archive_a": str(a), "archive_b": str(b),
              "tensors": len(x["tensors"]), "different_tensors": different}
    print(json.dumps(report))
    if different:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
