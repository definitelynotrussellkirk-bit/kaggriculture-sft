"""Small serialization helpers, shared by orchestration, not actor semantics."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def bound(value):
    body = dict(value)
    body.pop("content_sha256", None)
    return {**body, "content_sha256": digest(body)}


def equal(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError("mismatch: " + label)


def file_sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular file required: " + str(path))
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path, *, binding=True):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def bad(value):
        raise ValueError("nonfinite JSON value")
    path = Path(path)
    file_sha(path)
    value = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=bad)
    if binding:
        equal(value, bound(value), "JSON content binding " + path.name)
    return value


def write_new(path, value):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlinked output")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = bound(value)
    with path.open("x") as stream:
        stream.write(canonical(value) + "\n")
    return value


def tree_hashes(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in evidence tree")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_sha(path)
    return result


def numerical_environment():
    import numpy as np
    return {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "thread_settings": {name: os.environ.get(name) for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}


FLAGS = {"promotion_authority": False, "submission_authority": False,
         "population_probability_calibrated": False, "fresh_heldout": False}
