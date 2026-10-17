"""Evolution v2: fitness in the real 2-player condition (2 seeds vs ShopRouter + 1 seed vs PASS), market-timing motifs."""
import os, sys
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the src/ directory, so farm_neat imports work when run as a script
sys.path.insert(0, _SRC)
from farm_neat.game.constants import PRODUCTS, BUYABLE_PRODUCTS
from farm_neat.training.teachers import search as s1
from farm_neat.evaluation.eval_vs import play as play_vs
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}

def gen_sale_timed(rng):
    p = rng.choice(list(PRODUCTS)); b = BASE[p]
    return {"op": "SALE", "product": p, "quantity": rng.choice([100, 1000, rng.randint(1, 30)]),
            "price_above": round(b * rng.choice([0, 0.8, 1.0, 1.1, 1.25, 1.5, 2.0]), 2), "reserve": rng.choice([0, 0, rng.randint(1, 30)]), **s1.window(rng)}

def gen_stock_timed(rng):
    p = rng.choice(list(BUYABLE_PRODUCTS)); b = BASE[p]
    return {"op": "STOCK", "kind": "PRODUCT", "item": p, "target": rng.randint(5, 100),
            "price_below": round(b * rng.choice([0.5, 0.7, 0.85, 1.0, 1.2]), 2), **s1.window(rng)}

def gen_buy_timed(rng):
    p = rng.choice(list(BUYABLE_PRODUCTS)); b = BASE[p]
    return {"op": "BUY", "kind": "PRODUCT", "item": p, "quantity": rng.randint(5, 100),
            "price_below": round(b * rng.choice([0.5, 0.7, 0.85, 1.0, 1.2]), 2), **s1.window(rng, late=True)}

s1.GENERATORS[:] = [s1.gen_animal, s1.gen_animal, s1.gen_animal, s1.gen_crop, s1.gen_sale, gen_sale_timed, gen_sale_timed,
                    gen_stock_timed, gen_buy_timed, s1.gen_hands, s1.gen_land, s1.gen_land, s1.gen_reserve]
MUT = {"SET_ANIMAL": s1.gen_animal, "SET_CROP": s1.gen_crop, "SALE": gen_sale_timed, "BUY": gen_buy_timed, "STOCK": gen_stock_timed,
       "HANDS": s1.gen_hands, "LAND": s1.gen_land, "CASH_RESERVE": s1.gen_reserve}

def mutate(rng, prog):
    p = copy.deepcopy(prog); r = rng.random()
    if r < 0.25 and len(p) < 24: p.insert(rng.randint(0, len(p)), rng.choice(s1.GENERATORS)(rng))
    elif r < 0.40 and len(p) > 1: p.pop(rng.randrange(len(p)))
    elif r < 0.50 and len(p) > 1:
        i, j = rng.sample(range(len(p)), 2); p[i], p[j] = p[j], p[i]
    else:
        i = rng.randrange(len(p)); c = p[i]; fresh = MUT[c["op"]](rng)
        k = rng.choice([k for k in c if k != "op"]); c[k] = fresh[k]
        if isinstance(c.get("until"), int) and c["until"] < c.get("start", 0): c["until"] = "GAME_END"
    return p

def evaluate(pool, programs, seeds):
    jobs = []
    for p in programs:
        for i, sd in enumerate(seeds): jobs.append((p, sd, sd % 2, "router" if i < 2 else "pass"))
    res = pool.map(play_vs, jobs, chunksize=1)
    out = []; detail = []
    for i in range(len(programs)):
        vals = res[i * len(seeds):(i + 1) * len(seeds)]
        own = [v[0] for v in vals]; out.append(sum(own) / len(own)); detail.append(vals)
    return out, detail
