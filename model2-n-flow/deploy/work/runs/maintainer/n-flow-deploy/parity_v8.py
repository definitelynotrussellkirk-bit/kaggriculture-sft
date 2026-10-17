"""Compare deployed causal v8 inputs with replay training examples on three full seats."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (ROOT / "src", HERE):
    sys.path.insert(0, str(p))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import agent  # noqa: E402
import data  # noqa: E402
import v8  # noqa: E402

REPLAYS = (
    (HERE / "fixtures/1528.json.gz", 1528, 1),
    (HERE / "fixtures/1529.json.gz", 1529, 0),
    (HERE / "fixtures/1530.json.gz", 1530, 1),
)
KEYS = ("rows", "token_mask", "knownness", "unit_pos", "extra", "unit_extra", "reach_eta",
        "v6_token", "v6_resource_eta", "v6_resource_route", "v6_job_deadline",
        "v6_job_deadline_present", "x8_token", "x8_unit", "lattice")


class CapturePolicy:
    def __init__(self):
        tw, uw, lc = v8.widths()
        self.config = SimpleNamespace(x8_token_width=tw, x8_unit_width=uw, lattice_channels=lc)
        self.last_features = None

    def act(self, observation, *, configuration=None, v8_history=None, v8_long=None):
        self.last_features = data.features(observation, configuration, v8_history=v8_history,
                                           v8_long=v8_long)
        return {"farmer": ["PASS"], "hands": [], "market": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "PARITY-3-FROZEN.json")
    args = parser.parse_args()
    results = []
    for path, episode, seat in REPLAYS:
        raw = gzip.decompress(path.read_bytes())
        replay = json.loads(raw)
        policy = CapturePolicy()
        actor = agent.Agent("preloaded", policy=policy, log=False)
        checked = 0
        for t, expected in enumerate(data.episode_examples(replay, seat, meta_base={}, v8=True)):
            actor(replay["steps"][t][seat]["observation"], replay["configuration"])
            if actor.fallbacks:
                raise AssertionError(f"episode {episode} t{t}: fallback {actor.errors[-1]}")
            got = policy.last_features
            for key in KEYS:
                if not np.array_equal(got[key], expected[key]):
                    raise AssertionError(f"episode {episode} seat{seat} t{t} {key}: "
                                         f"{got[key].shape} != {expected[key].shape}")
            checked += 1
        if checked != 719:
            raise AssertionError(f"episode {episode}: {checked} decisions")
        results.append({"episode_id": episode, "seat": seat, "replay_sha256": hashlib.sha256(raw).hexdigest(),
                        "decisions": checked, "keys": len(KEYS), "fallbacks": actor.fallbacks})
        print(json.dumps(results[-1]), flush=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps({"schema": "v8-deploy-input-parity-v1", "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
