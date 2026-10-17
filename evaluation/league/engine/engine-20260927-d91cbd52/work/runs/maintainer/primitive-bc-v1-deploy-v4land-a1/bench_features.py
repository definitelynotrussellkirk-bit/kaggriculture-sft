"""Per-step CPU cost of the full feature path on one thread, real replay steps, both seats.

Timed per seat-step: FeatureState.update (history) + compute (theorem + scaling) +
feature_tokens.tokens + record_action (history action parse + codec labels). Observations are
built beforehand (bc.public_observation, untimed); steps before the timed window are run untimed
so the history state is the real one. Wall (perf_counter) and thread CPU (thread_time) are both
reported; observation.pack is timed on the same steps as a reference for the existing input path.
The pre-built episode (2 x 719 deep-copied observations) is moved out of the collector with
gc.freeze() before timing, so full collections scan what a live agent holds, not the harness.
Steps during which the cyclic collector ran (gc.callbacks) are counted, and the statistics are
also given without them, so collector pauses can be told apart from feature work.

  cd /home/russ/Desktop/farmNEAT && CUDA_VISIBLE_DEVICES= \
  PYTHONPATH=src:work/runs/maintainer/primitive-bc:work/cache/pytest-path \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  work/venv/python -B work/runs/maintainer/primitive-bc/bench_features.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import feature_tokens as TK  # noqa: E402
import features as FT  # noqa: E402
from farm_neat.policy import observation as OBS  # noqa: E402
from farm_neat.training import bc  # noqa: E402

DAILY = ROOT / "work/datasets/public-episodes/daily"
PARTS = ("update", "compute", "tokens", "record_action")
_GC = {"runs": 0, "ns": 0, "t": 0}


def _gc_callback(phase, info):
    if phase == "start":
        _GC["t"] = time.perf_counter_ns()
    else:
        _GC["runs"] += 1
        _GC["ns"] += time.perf_counter_ns() - _GC["t"]


def _stats(ns):
    ms = np.asarray(ns, np.float64) / 1e6
    return {"n": int(ms.size), "p50": round(float(np.percentile(ms, 50)), 3),
            "p95": round(float(np.percentile(ms, 95)), 3), "p99": round(float(np.percentile(ms, 99)), 3),
            "max": round(float(ms.max()), 3), "mean": round(float(ms.mean()), 3),
            "over_5ms": int((ms > 5).sum())}


def bench_episode(eid, window, repeat=1):
    data = json.loads((DAILY / f"{eid}.json").read_text())
    cfg = data["configuration"]
    pack_cfg = {k: cfg[k] for k in OBS.CONFIG}
    obs = [[bc.public_observation(data, t, s) for t in range(719)] for s in (0, 1)]
    acts = [[data["steps"][t + 1][s]["action"] for t in range(719)] for s in (0, 1)]
    gc.collect()
    gc.freeze()
    lo, hi = window
    wall = {p: [] for p in (*PARTS, "total", "pack")}
    cpu = {p: [] for p in ("total", "pack")}
    gc_ns = []
    gc.callbacks.append(_gc_callback)
    for _ in range(repeat):
        states = [FT.FeatureState(cfg), FT.FeatureState(cfg)]
        for t in range(hi):
            for s in (0, 1):
                st, o = states[s], obs[s][t]
                if t < lo:
                    st.update(o)
                    TK.tokens(st.compute())
                    st.record_action(acts[s][t])
                    continue
                g0 = _GC["ns"]
                c0 = time.thread_time_ns()
                w0 = time.perf_counter_ns()
                st.update(o)
                w1 = time.perf_counter_ns()
                f = st.compute()
                w2 = time.perf_counter_ns()
                TK.tokens(f)
                w3 = time.perf_counter_ns()
                st.record_action(acts[s][t])
                w4 = time.perf_counter_ns()
                c1 = time.thread_time_ns()
                gc_ns.append(_GC["ns"] - g0)
                OBS.pack(o, None, pack_cfg)
                w5 = time.perf_counter_ns()
                c2 = time.thread_time_ns()
                for p, v in zip(PARTS, (w1 - w0, w2 - w1, w3 - w2, w4 - w3)):
                    wall[p].append(v)
                wall["total"].append(w4 - w0)
                wall["pack"].append(w5 - w4)
                cpu["total"].append(c1 - c0)
                cpu["pack"].append(c2 - c1)
    gc.callbacks.remove(_gc_callback)
    gc.unfreeze()
    gc_ns = np.asarray(gc_ns)
    total = np.asarray(wall["total"])
    return {"episode": eid, "steps": [lo, hi - 1], "seats": [0, 1], "repeat": repeat,
            "features_wall_ms": _stats(total), "features_thread_cpu_ms": _stats(cpu["total"]),
            "gc": {"steps_with_collection": int((gc_ns > 0).sum()),
                   "over_5ms_steps_with_collection": int(((total > 5e6) & (gc_ns > 0)).sum()),
                   "collection_ms_max": round(float(gc_ns.max()) / 1e6, 3),
                   "features_wall_ms_steps_without_collection": _stats(total[gc_ns == 0])},
            "parts_wall_ms": {p: _stats(wall[p]) for p in PARTS},
            "reference_pack_wall_ms": _stats(wall["pack"]), "reference_pack_thread_cpu_ms": _stats(cpu["pack"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", nargs="+", default=["111424871", "111419545"])
    ap.add_argument("--windows", nargs="+", default=["419:719", "0:300"],
                    help="timed step windows lo:hi (hi exclusive); 300 steps each by default")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default=None, help="also write the JSON report here")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    FT.warm()   # exact price tables, built once per process
    warm_ms = (time.perf_counter() - t0) * 1e3
    report = {"schema": FT.FEATURE_SCHEMA, "feature_schema_sha256": FT.FEATURE_SCHEMA_SHA256,
              "token_schema": TK.TOKEN_SCHEMA, "tokens": [TK.K, TK.WIDTH], "tile": [TK.N_CELLS, TK.TILE_WIDTH],
              "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "warm_ms_once_per_process": round(warm_ms, 1), "loadavg_start": os.getloadavg(),
              "runs": []}
    for eid in args.episodes:
        for w in args.windows:
            lo, hi = (int(v) for v in w.split(":"))
            report["runs"].append(bench_episode(eid, (lo, hi), args.repeat))
    report["loadavg_end"] = os.getloadavg()
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return report


if __name__ == "__main__":
    main()
