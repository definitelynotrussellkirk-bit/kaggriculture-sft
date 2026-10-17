"""Frozen opponent specs for the fast macro-policy loop: registry snapshots or public tapes.

A spec is either a registry snapshot id (``v3-herd-solver``) or ``tape:<path>[#<seat>]``:

* ``tape:<main.py>`` — a public Kaggle tape kernel (a fixed action trace with a Kaggle
  ``agent(obs, config)`` entry), loaded as a module from its own directory.
* ``tape:<replay.json>#<seat>`` — the per-step actions one seat took in a recorded episode,
  replayed blind by step index exactly as the public tapes do.

Tapes are fixed schedules, so they are ideal frozen opponents: the network can learn to exploit
them (the ladder's top is tapes) and their content identity is just the file's SHA-256.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def is_tape_spec(spec) -> bool:
    return isinstance(spec, str) and spec.startswith("tape:")


def _parse_tape_spec(spec):
    body = spec[len("tape:"):]
    seat = None
    if "#" in body:
        body, seat_text = body.rsplit("#", 1)
        seat = int(seat_text)
    # Repository-relative specs are the portable form: an absolute /home/... path pins the pool
    # to one machine and makes a review bundle unresolvable elsewhere.  Absolute specs still work.
    path = Path(body).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent.parent / path)
    path = path.resolve()
    if not path.is_file():
        capsule = _capsule_for(path, seat)
        if capsule is not None:
            return capsule, seat
        raise FileNotFoundError(f"tape opponent file does not exist: {path}")
    return path, seat


#: Where the compact capsules live when the 1,190 MB replay corpus is not shipped.
CAPSULE_DIR = Path(__file__).resolve().parent.parent / "replays/capsules"


def _capsule_for(path: Path, seat):
    """The capsule standing in for a replay that is not present.

    A review packet carries 8.8 MB of capsules instead of the full corpus, so every hardcoded
    `tape:replays/.../episode-*.json#N` in the tests and the smoke script has to keep working
    from a clean extraction -- that is the whole point of execution closure, and resolving it
    only in `frontier_panel` left ten tests still failing on an absent file.
    """
    if seat is None or not CAPSULE_DIR.is_dir():
        return None
    candidate = CAPSULE_DIR / f"{path.stem}.seat{int(seat)}.capsule.json"
    return candidate if candidate.is_file() else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def opponent_identity(spec) -> str:
    """Content identity for the frozen-opponent guards (registry identity or tape file SHA)."""
    if is_tape_spec(spec):
        path, seat = _parse_tape_spec(spec)
        return {
            "schema": "kaggriculture-tape-opponent-identity-v1",
            "kind": "tape",
            "path": str(path),
            "seat": seat,
            "sha256": _file_sha256(path),
        }
    raise NotImplementedError('the neural opponent registry was not migrated')

    return registry.snapshot_identity(spec)


def _replay_trace(path: Path, seat: int):
    replay = json.loads(path.read_text())
    # A CAPSULE is the same schedule with the observations stripped: 1.2 GB of panel replays
    # become 8.8 MB, which is the difference between a review packet that can run the panel it
    # documents and one that cannot. Both 2026-09-01 reviews reported the missing corpus.
    # `scripts/replay_capsules.py` builds them; the seat is baked in, so `#seat` must agree.
    if isinstance(replay, dict) and replay.get("schema") == 1 and "actions" in replay:
        if int(replay.get("seat", seat)) != int(seat):
            raise ValueError(f"{path} is seat {replay.get('seat')}, not the requested {seat}")
        return [a if isinstance(a, dict) and a else PASS_ACTION for a in replay["actions"]]
    steps = replay["steps"]
    if seat not in (0, 1):
        raise ValueError("replay tape seat must be 0 or 1")
    # Kaggle stores, at index t, the action that produced observation t (i.e. taken at t-1),
    # so the schedule for step t is the action recorded at index t+1.
    actions = []
    for index in range(1, len(steps)):
        action = steps[index][seat].get("action")
        actions.append(action if isinstance(action, dict) and action else PASS_ACTION)
    if not actions:
        raise ValueError(f"replay {path} has no actions for seat {seat}")
    return actions


def _tape_module_agent(path: Path):
    directory = str(path.parent)
    name = f"tape_{_file_sha256(path)[:16]}"
    if name in sys.modules:
        return sys.modules[name].agent
    added = directory not in sys.path
    if added:
        sys.path.insert(0, directory)
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        if added:
            sys.path.remove(directory)
    if not callable(getattr(module, "agent", None)):
        raise ValueError(f"tape kernel {path} exposes no callable agent(obs, config)")
    return module.agent


def build_opponent(spec):
    """Instantiate a frozen opponent callable ``agent(obs, config)`` from a spec."""
    if not is_tape_spec(spec):
        raise NotImplementedError('the neural opponent registry was not migrated')

        return registry.build(spec)
    path, seat = _parse_tape_spec(spec)
    if path.suffix == ".py":
        kernel = _tape_module_agent(path)

        def agent(obs, config=None):
            return kernel(obs, config)

        return agent
    if seat is None:
        raise ValueError("replay tapes need an explicit seat: tape:<replay.json>#<seat>")
    # One replay implementation, in train.tape: a recorded opponent and a plan our own policy
    # commits to are the same object, and two readers of one file is how they came to disagree
    # about which index holds the action for step t.
    raise NotImplementedError('replay-tape opponents were not migrated')

    return Tape.from_replay(path, seat).agent()


__all__ = ["PASS_ACTION", "build_opponent", "is_tape_spec", "opponent_identity"]
