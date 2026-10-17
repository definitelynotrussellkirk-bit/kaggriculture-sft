"""PRO CHECK-BUCKETS: log every seat a step-cache worker draws, in draw order: <PBC_DRAW_LOG>/draws-w<wid>-p<pid>.log, one
'<unix time>\t<eid>-<seat>.stp' line per seat. Wraps StepChunks._seats (step_cache.py itself is hashed into the cache identity,
so it stays byte-identical); imported by train.py only when PBC_DRAW_LOG is set. Draw order and data are unchanged."""
import os
import time

import step_cache

_orig_seats = step_cache.StepChunks._seats


def _logged_seats(self, wid, nw):
    d = os.environ.get("PBC_DRAW_LOG")
    if not d:
        yield from _orig_seats(self, wid, nw)
        return
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"draws-w{wid:02d}-p{os.getpid()}.log"), "a", buffering=1) as f:
        for s in _orig_seats(self, wid, nw):
            f.write(f"{time.time():.1f}\t{s.name}\n")
            yield s


step_cache.StepChunks._seats = _logged_seats
