"""prefix.PrefixExecutor (incremental, replay on first over-subscription) against a from-scratch recomputation of every
prefix with the engine's own unit function (the Astra reference definition), on real replay steps and on atomic-PLANT
witnesses.   ../../../venv/python -B test_prefix.py [N_REPLAYS]"""
import copy, json, random, sys, zipfile, glob
sys.path.insert(0, ".")
import numpy as np
import data  # noqa: E402  (puts ../../../../src on sys.path)
import prefix as P
from collections import Counter


def reference_state(obs, cmds):
    seat = int(obs["player"])
    farm = copy.deepcopy(obs["farms"][seat])
    pv = obs.get("private") or {}
    private = {k: copy.deepcopy(pv.get(k, d)) for k, d in (("shed", {}), ("seeds", {}), ("inventories", [{}]))}
    req = Counter(a[1] for a in cmds if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT")
    blocked = {c for c, n in req.items() if n > private["seeds"].get(c, 0)}
    for i, a in enumerate(cmds):
        if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] in blocked:
            a = ["PASS"]
        P.E._apply_unit_action(farm, private, i, a, 10, int(obs["step"]) // 24, 24, 100)
    return farm, private


def check(obs, units):
    pe = P.PrefixExecutor(obs)
    bad = 0
    for k, a in enumerate(units):
        rf, rp = reference_state(obs, units[:k])
        bad += (pe.farm != rf) or (pe.private != rp)
        pe.apply_unit(k, a)
    rf, rp = reference_state(obs, units)
    bad += (pe.farm != rf) or (pe.private != rp)
    return bad


def witnesses():
    """One wheat seed: PLANT, WATER, PLANT blocks both PLANTs; the WATER then acts on the restored tile."""
    z = zipfile.ZipFile(sorted(glob.glob("/mnt/ai/scratch/farmNEAT/primitive-bc-data/kaggriculture-episodes-2026-09-2*.zip"))[-1])
    rep = json.loads(z.read([m for m in z.namelist() if m.endswith(".json")][0]))
    obs = data.bc.public_observation(rep, 300, 0)
    farm = obs["farms"][0]
    empty = [(x, y) for y in range(10) for x in range(10) if farm["tiles"][y][x] is None]
    if len(empty) < 2 or len(farm["hands"]) < 2:
        return 0, 0
    (x, y) = empty[0]
    obs = copy.deepcopy(obs)
    obs["farms"][0]["farmer"], obs["farms"][0]["hands"][0], obs["farms"][0]["hands"][1] = [x, y], [x, y], [x, y]
    obs["private"]["seeds"] = {"WHEAT": 1}
    units = [["PLANT", "WHEAT"], ["WATER"], ["PLANT", "WHEAT"]] + [["PASS"]] * (len(farm["hands"]) - 2)
    pe = P.PrefixExecutor(obs)
    pe.apply_unit(0, units[0])
    planted = isinstance(pe.farm["tiles"][y][x], dict)
    pe.apply_unit(1, units[1]); pe.apply_unit(2, units[2])
    restored = pe.farm["tiles"][y][x] is None and pe.private["seeds"].get("WHEAT") == 1
    f = pe.unit_features(3) if len(units) > 3 else pe.unit_features(2)
    return int(planted and restored and f[18] == 1.0), 1


def main(n=6):
    arch = sorted(glob.glob("/mnt/ai/scratch/farmNEAT/primitive-bc-data/kaggriculture-episodes-2026-09-2*.zip"))[-2:]
    rng = random.Random(0)
    members = [(a, m) for a in arch for m in zipfile.ZipFile(a).namelist() if m.endswith(".json")][:60]
    rng.shuffle(members)
    states = bad = 0
    for a, m in members[:n]:
        rep = json.loads(zipfile.ZipFile(a).read(m))
        for t in range(3, 719, 37):
            for s in (0, 1):
                obs = data.bc.public_observation(rep, t, s)
                act = rep["steps"][t + 1][s].get("action") or {}
                units = [act.get("farmer", ["PASS"]), *(act.get("hands") or [])]
                units = units[:1 + len(obs["farms"][s]["hands"])]
                bad += check(obs, units)
                states += 1
    ok, n_w = witnesses()
    print(json.dumps({"states": states, "prefix_mismatches": bad, "atomic_plant_witness": f"{ok}/{n_w}"}))
    return 1 if bad or ok != n_w else 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 6))
