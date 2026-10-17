"""Evolution v3: seeded with the hand-written $93k program; generators biased to scale (hands, land, crops)."""
import copy, os, sys
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the src/ directory, so farm_neat imports work when run as a script
sys.path.insert(0, _SRC)
from farm_neat.training.teachers import search as s1, search2 as s2
from farm_neat.evaluation.eval_vs import play as play_vs

def gen_animal(rng):
    c = s1.gen_animal(rng); c["feed"] = "DAILY"; c["care"] = "DAILY"; c["manure"] = "WHEAT" if False else "WHEN_AVAILABLE"
    c["harvest_threshold"] = 1; c["priority"] = rng.choice([2, 3, 3]); c["start"] = rng.choice([0, 0, 0, rng.randint(0, 120)]); c["until"] = "GAME_END"
    if rng.random() < 0.7: c["scope"] = {"kind": "TILE", "x": rng.randint(0, 9), "y": rng.randint(0, 9)}
    return c
def gen_crop(rng):
    c = s1.gen_crop(rng); c["watering"] = "DAILY"; c["priority"] = rng.choice([4, 4, 5, 1]); c["harvest_threshold"] = 1
    c["cycles"] = rng.choice([None, None, rng.randint(2, 30)]); c["start"] = rng.choice([0, 0, rng.randint(0, 200)]); c["until"] = "GAME_END"
    c["crop"] = rng.choice(["WHEAT", "WHEAT", "STRAWBERRY", "MELON", "CARROT", "TOMATO"])
    if rng.random() < 0.6: c["scope"] = {"kind": "TILE", "x": rng.randint(0, 9), "y": rng.randint(0, 9)}
    return c
def gen_hands(rng): return {"op": "HANDS", "target": rng.randint(4, 12), "start": rng.choice([0, 0, rng.randint(0, 96)]), "until": "GAME_END", "priority": rng.choice([0, 1])}
def gen_land(rng): return {"op": "LAND", "target": rng.randint(1, 3), "start": rng.choice([0, 0, rng.randint(0, 240)]), "until": "GAME_END", "priority": rng.choice([0, 1, 2])}
def gen_sale(rng):
    c = s2.gen_sale_timed(rng); c["priority"] = 0; c["start"] = 0; c["until"] = "GAME_END"; return c
GENS = [gen_animal, gen_animal, gen_animal, gen_crop, gen_crop, gen_sale, gen_sale, s2.gen_stock_timed, gen_hands, gen_land, s1.gen_reserve]
MUT = {"SET_ANIMAL": gen_animal, "SET_CROP": gen_crop, "SALE": gen_sale, "BUY": s2.gen_buy_timed, "STOCK": s2.gen_stock_timed, "HANDS": gen_hands, "LAND": gen_land, "CASH_RESERVE": s1.gen_reserve}
def mutate(rng, prog):
    p = copy.deepcopy(prog); r = rng.random()
    if r < 0.30 and len(p) < 48: p.insert(rng.randint(0, len(p)), rng.choice(GENS)(rng))
    elif r < 0.42 and len(p) > 1: p.pop(rng.randrange(len(p)))
    elif r < 0.50 and len(p) > 1:
        i, j = rng.sample(range(len(p)), 2); p[i], p[j] = p[j], p[i]
    else:
        i = rng.randrange(len(p)); c = p[i]; fresh = MUT[c["op"]](rng)
        k = rng.choice([k for k in c if k != "op"]); c[k] = fresh[k]
        if isinstance(c.get("until"), int) and c["until"] < c.get("start", 0): c["until"] = "GAME_END"
    return p
def evaluate(pool, programs, seeds):
    jobs = [(p, sd, sd % 2, "router" if i < 2 else "pass") for p in programs for i, sd in enumerate(seeds)]
    res = pool.map(play_vs, jobs, chunksize=1); out = []; detail = []
    for i in range(len(programs)):
        vals = res[i * len(seeds):(i + 1) * len(seeds)]; out.append(sum(v[0] for v in vals) / len(vals)); detail.append(vals)
    return out, detail
