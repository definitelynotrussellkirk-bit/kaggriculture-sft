"""tape_examples (compact tape re-simulated in the exact engine) must equal episode_examples (official replay) at every step
and seat, for every array: rows, masks, positions, v3 extra/unit_extra/reach_eta, labels and prefix arrays.

  PYTHONPATH=<this dir>:<replica src>:<runtime site> python3 -B test_tape_v3.py TAPE.json.gz REPLAY.json.gz
"""
import gzip
import json
import sys

import numpy as np

import data as D


def main():
    tape_path, replay_path = sys.argv[1], sys.argv[2]
    replay = json.load(gzip.open(replay_path))
    D.check_replay(replay)
    for seat in (0, 1):
        a = list(D.episode_examples(replay, seat, meta_base={}))
        b = D.tape_examples(tape_path, seat, meta_base={})
        assert len(a) == len(b) == D.STEPS, (len(a), len(b))
        bad, keys = 0, set()
        for t, (x, y) in enumerate(zip(a, b)):
            assert set(x) == set(y), (set(x) ^ set(y))
            for k in x:
                if k == "meta":
                    continue
                if not np.array_equal(np.asarray(x[k]), np.asarray(y[k])):
                    bad += 1
                    keys.add(k)
                    if bad <= 5:
                        print("DIFF seat", seat, "step", t, k)
        print(json.dumps({"seat": seat, "steps": len(a), "keys": sorted(k for k in a[0] if k != "meta"), "diffs": bad,
                          "diff_keys": sorted(keys)}))
        assert bad == 0
    print("TAPE == REPLAY PASS")


if __name__ == "__main__":
    main()
