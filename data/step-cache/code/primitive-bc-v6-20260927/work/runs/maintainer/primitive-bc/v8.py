"""v8 feature families (owner 2026-09-28 "build all that shit"; contract /mnt/ai/scratch/farmNEAT/pbc-features-v8/CONTRACT.md).
Each family v8_<name>.py returns per-token [N, W] and per-unit [25, V] float16 columns from the actor observation and the seat's
previous two observations; this module concatenates them (FAMILIES order is part of the input schema) and adds the lattice."""
import hashlib
import importlib
from pathlib import Path

import numpy as np

FAMILIES = ("dusk", "cohorts", "tileclock", "flags", "gaps")   # ordered, required input schema; never silently skip
LATTICE = True
LONG = True    # v8_gaps.compute_long: per-seat running state (GapsState), appended after the families' token columns
_mods = None


def available():
    global _mods
    if _mods is None:
        # Assign only after every import succeeds, including transitive dependencies.
        # Otherwise a missing family silently changes the learned column meanings.
        modules = [importlib.import_module(f"v8_{f}") for f in FAMILIES]
        _mods = modules
    return _mods


def widths():
    """(token width, unit width, lattice channels) of the current family set."""
    ms = available()
    channels = importlib.import_module("v8_lattice").WIDTH if LATTICE else 0
    long_w = importlib.import_module("v8_gaps").WIDTH_LONG if LONG else 0
    return sum(m.WIDTH_TOKEN for m in ms) + long_w, sum(m.WIDTH_UNIT for m in ms), channels


def names():
    long_cols = [f"gaps_long:{c}" for c in importlib.import_module("v8_gaps").LONG_TOKEN_COLS] if LONG else []
    return [f"{m.__name__[3:]}:{c}" for m in available() for c in m.TOKEN_COLS] + long_cols, \
           [f"{m.__name__[3:]}:{c}" for m in available() for c in m.UNIT_COLS]


def compute(actor_obs, configuration, token_ids, history, long_state=None):
    # Every family must see the same seat-local, causal history. A bad serving
    # caller must not expose another seat's private inventories through dusk.
    step, seat = int(actor_obs["step"]), int(actor_obs["player"])
    hist = []
    for lag, h in enumerate(tuple(history or ())[:2], 1):
        try:
            valid = (isinstance(h, dict) and step >= lag and int(h["player"]) == seat
                     and int(h["step"]) == step - lag)
        except (KeyError, TypeError, ValueError):
            valid = False
        hist.append(h if valid else None)
    hist = tuple(hist)
    toks, units = [np.zeros((len(token_ids), 0), np.float16)], [np.zeros((25, 0), np.float16)]
    for m in available():
        r = m.compute(actor_obs, configuration, token_ids, history=hist)
        toks.append(_checked_half(r["token"], (len(token_ids), m.WIDTH_TOKEN), m.__name__ + ":token"))
        units.append(_checked_half(r["unit"], (25, m.WIDTH_UNIT), m.__name__ + ":unit"))
    if LONG:
        # the running columns need ONE per-seat state that saw every earlier step in order (data.episode_examples,
        # tape_examples and the deploy agent carry it); a missing state would silently shift the column schema
        if long_state is None:
            raise ValueError("v8 long columns need the seat's v8_gaps.GapsState (features(..., v8_long=state))")
        import v8_gaps
        r = v8_gaps.compute_long(actor_obs, configuration, token_ids, long_state)
        toks.append(_checked_half(r["token"], (len(token_ids), v8_gaps.WIDTH_LONG), "v8_gaps:long"))
    out = {"x8_token": np.concatenate(toks, 1), "x8_unit": np.concatenate(units, 1)}
    if LATTICE:
        import v8_lattice
        out["lattice"] = _checked_half(v8_lattice.compute_lattice(actor_obs, configuration, hist),
                                       (2, 10, 10, v8_lattice.WIDTH), "v8_lattice")
    return out


def _checked_half(value, shape, name):
    """Reject corrupt schemas and overflow rather than silently poisoning training."""
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind != "f":
        raise ValueError(f"{name}: expected floating {shape}, got {value.dtype} {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: non-finite feature")
    # Existing float16 families need no second range scan or allocation.
    if value.dtype != np.float16 and value.size and np.max(np.abs(value)) > np.finfo(np.float16).max:
        raise ValueError(f"{name}: feature overflows float16")
    return value.astype(np.float16, copy=False)


def schema():
    """Content-bound descriptor for cache/build/serve handoffs. Never called per step.

    Column names and order alone cannot detect a same-width semantic correction.
    Hash the family code and the directly shared valuation/mechanics providers too.
    The cache separately fingerprints its packing pipeline and numeric versions.
    """
    def source(module):
        return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    modules = available()
    deps = {}
    for m in modules:
        for attr in ("TF", "CALC", "flip"):
            dep = getattr(m, attr, None)
            if dep is not None:
                deps[dep.__name__] = source(dep)
    lattice = importlib.import_module("v8_lattice") if LATTICE else None
    return {"revision": 1, "widths": list(widths()),
            "families": [{"name": m.__name__, "source_sha256": source(m),
                           "token_columns": list(m.TOKEN_COLS), "unit_columns": list(m.UNIT_COLS)}
                          for m in modules],
            "lattice": None if lattice is None else {"source_sha256": source(lattice),
                                                       "channels": list(lattice.CHANNELS)},
            "long": None if not LONG else {"name": "v8_gaps:long",
                                           "token_columns": list(importlib.import_module("v8_gaps").LONG_TOKEN_COLS)},
            "dependencies": dict(sorted(deps.items()))}
