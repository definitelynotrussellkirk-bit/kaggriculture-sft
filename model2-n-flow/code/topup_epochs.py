#!/usr/bin/env python3
"""Epochs of a top-up over its roster: steps trained (updates x 60 streams x 24) / steps in the roster's seat files.
  python3 topup_epochs.py ROSTER UPDATES [CACHE_DIR]   -> one JSON line"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import nf_format as NF

CACHE = sys.argv[3] if len(sys.argv) > 3 else "/content/cache"


def n(r):
    try:
        return int(NF.read_stp(f"{CACHE}/{r['eid']}-{r['seat']}.stp")["step"].shape[0])
    except Exception:  # noqa: BLE001
        return 0


if __name__ == "__main__":
    rows = NF.read_roster(sys.argv[1])
    upd = int(sys.argv[2])
    with ProcessPoolExecutor(16) as ex:
        per = list(ex.map(n, rows, chunksize=8))
    trained = upd * 60 * 24
    print(json.dumps({"updates": upd, "streams_per_update": 60, "chunk": 24, "steps_trained": trained,
                      "roster_seats": len(rows), "seats_unreadable": sum(1 for x in per if not x),
                      "roster_steps": sum(per), "epochs": round(trained / max(1, sum(per)), 3)}))
