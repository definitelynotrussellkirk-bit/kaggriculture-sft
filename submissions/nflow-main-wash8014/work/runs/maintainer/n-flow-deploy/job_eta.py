"""Additive actor-visible, finite nine-job route facts (pbc-job-input-v6.0).

The old flip columns and reach matrix are deliberately untouched. A zero ETA in
the resource arrays is meaningful only together with its separate route mask.
"""
from __future__ import annotations

import numpy as np

import flip

N_JOBS = 9
N_RESOURCES = 2
TOKEN_WIDTH = 8
JOB_WATER, JOB_HARVEST, JOB_FEED, JOB_COLLECT, JOB_COLLECT_FERT, JOB_CARE, JOB_FERTILIZE, JOB_PLANT, JOB_DIG = range(N_JOBS)
_PRIMARY = {
    flip.WATER_TONIGHT: JOB_WATER, flip.WATER_TODAY: JOB_WATER,
    flip.HARVEST_DECAYING: JOB_HARVEST, flip.HARVEST: JOB_HARVEST,
    flip.FEED_TONIGHT: JOB_FEED, flip.FEED_TODAY: JOB_FEED,
    flip.COLLECT: JOB_COLLECT, flip.FERTILIZER: JOB_COLLECT_FERT,
    flip.CARE: JOB_CARE, flip.PLANT_EMPTY: JOB_PLANT, flip.WEED: JOB_DIG,
}


def extract(obs, token_ids, max_units=25):
    """Return start-of-decision own-farm job tensors and token summaries.

    Rival/private-hidden routes are intentionally not inferred. Future purchases
    and same-step earlier worker commands are not included in this snapshot.
    """
    t, seat = int(obs["step"]), int(obs["player"])
    farm = obs["farms"][seat]
    private = obs.get("private") or {}
    shed = private.get("shed") or {}
    inventories = private.get("inventories") or []
    workers = flip._workers(farm)[:max_units]
    n = len(workers)
    cells = np.asarray([(x, y) for y in range(10) for x in range(10)], np.int64)
    direct = np.zeros((max_units, 100), np.uint8)
    direct[:n] = (flip._dist(workers, cells) + 1).astype(np.uint8)
    resource_eta = np.zeros((max_units, 100, N_RESOURCES), np.uint8)
    resource_route = np.zeros((max_units, 100, N_RESOURCES), bool)
    via = (flip._dist(workers, np.asarray(flip.SHED, np.int64))[:, :, None]
           + flip._dist(np.asarray(flip.SHED, np.int64), cells)[None]).min(1) + 2
    for r, product in enumerate(("WHEAT", "FERTILIZER")):
        carried = np.asarray([bool((inventories[k] or {}).get(product, 0)) if k < len(inventories) else False
                              for k in range(n)], bool)
        shed_has = bool(shed.get(product, 0))
        route = carried | shed_has
        resource_route[:n, :, r] = route[:, None]
        resource_eta[:n, :, r] = np.where(route[:, None],
                                         np.where(carried[:, None], direct[:n], via), 0).astype(np.uint8)

    job_present = np.zeros((100, N_JOBS), bool)
    job_deadline = np.zeros((100, N_JOBS), np.uint16)
    job_deadline_present = np.zeros((100, N_JOBS), bool)
    primary = np.full(100, -1, np.int8)
    seeds = private.get("seeds") or {}
    has_seeds = any(v > 0 for v in seeds.values())
    jobs = flip._jobs(farm, t, True, has_seeds)
    for cell, (x, y) in enumerate(cells):
        tile = farm["tiles"][y][x]
        kind, _, fc = jobs[(int(y), int(x))]
        job_present[cell] = np.asarray(flip.job_bits(tile, t, True, has_seeds, fc)[:N_JOBS], bool)
        primary[cell] = _PRIMARY.get(kind, -1)
        if fc is None:
            continue
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if job_present[cell, JOB_WATER] and t + fc["death"] <= flip.LAST_STEP:
                job_deadline[cell, JOB_WATER] = fc["death"]
                job_deadline_present[cell, JOB_WATER] = True
            if job_present[cell, JOB_HARVEST]:
                decay = fc["decay"]
                cutoff = flip.LAST_STEP - t
                # No-action dehydration may destroy a ripe crop before yield decay.
                gone = fc.get("gone")
                job_deadline[cell, JOB_HARVEST] = min(
                    cutoff, decay if decay is not None else cutoff,
                    gone if gone is not None else cutoff)
                job_deadline_present[cell, JOB_HARVEST] = True
        elif isinstance(tile, dict) and "animal" in tile:
            if job_present[cell, JOB_FEED] and t + fc["death"] <= flip.LAST_STEP:
                job_deadline[cell, JOB_FEED] = fc["death"]
                job_deadline_present[cell, JOB_FEED] = True

    token = np.zeros((len(token_ids), TOKEN_WIDTH), np.float32)
    min_eta = np.zeros((100, N_JOBS), np.uint8)
    route_exists = np.zeros((100, N_JOBS), bool)
    for j in range(N_JOBS):
        if j in (JOB_FEED, JOB_FERTILIZE):
            r = 0 if j == JOB_FEED else 1
            valid = resource_route[:n, :, r]
            eta = resource_eta[:n, :, r]
        else:
            valid = np.ones((n, 100), bool)
            eta = direct[:n]
        valid = valid & job_present[None, :, j]
        route_exists[:, j] = valid.any(0)
        min_eta[:, j] = np.where(valid, eta, 255).min(0) if n else 255
        min_eta[~route_exists[:, j], j] = 0
    for i, tid in enumerate(token_ids):
        if tid.startswith("tile/0/"):
            x, y = (int(v) for v in tid[7:].split(","))
            cell = y * 10 + x
            j = int(primary[cell])
            if j < 0 or not job_present[cell, j]:
                continue
            present, route = True, bool(route_exists[cell, j])
            deadline = bool(job_deadline_present[cell, j])
            eta = int(min_eta[cell, j]) if route else 0
            dl = int(job_deadline[cell, j]) if deadline else 0
            token[i, :6] = (present, route, deadline, min(2.0, eta / 24) if route else 0,
                            max(-2.0, min(2.0, (dl - eta) / 24)) if route and deadline else 0,
                            float(route and deadline and eta <= dl))
        elif tid == "farm/0":
            tonight = job_present & job_deadline_present & (job_deadline <= flip.DAY - t % flip.DAY)
            feasible = tonight & route_exists
            token[i, 6] = float(min_eta[feasible].sum()) / 200
            token[i, 7] = float((tonight & ~route_exists).sum()) / 100
    return {"token": token.astype(np.float16), "resource_eta": resource_eta,
            "resource_route": resource_route, "job_deadline": job_deadline,
            "job_deadline_present": job_deadline_present,
            "job_present": job_present, "direct_eta": direct}
