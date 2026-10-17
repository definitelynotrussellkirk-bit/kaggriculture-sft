"""Evolutionary search over full-economic plan programs, scored by terminal cash vs PASS.

Purpose: find profitable programs in the existing plan language (support for BC/GRPO),
and measure the ceiling of the language through the existing executor. No neural net.
"""
import copy, os, sys
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the src/ directory, so farm_neat imports work when run as a script
sys.path.insert(0, _SRC)
from farm_neat.game.constants import ANIMALS, CROPS, PRODUCTS, BUYABLE_PRODUCTS
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment, PASS_ACTION

ANIMAL_PRODUCT = {a: ANIMALS[a]["product"] for a in ANIMALS}
REGIONS = ("NW", "NE", "SW", "SE", "ALL")

def window(rng, late=False):
    start = rng.choice([0, 0, 0, 0, rng.randint(0, 96), rng.randint(0, 400)]) if late else rng.choice([0, 0, 0, rng.randint(0, 96)])
    until = rng.choice(["GAME_END", "GAME_END", "GAME_END", rng.randint(start + 24, 718)])
    return {"start": start, "until": until, "priority": rng.randint(0, 7)}

def scope(rng):
    if rng.random() < 0.5:
        return {"kind": "TILE", "x": rng.randint(0, 9), "y": rng.randint(0, 9)}
    r = rng.choice(REGIONS)
    return {"kind": "REGION", "region": r, "count": rng.randint(1, 25 if r != "ALL" else 40)}

def gen_animal(rng):
    a = rng.choice(list(ANIMALS))
    return {"op": "SET_ANIMAL", "scope": scope(rng), "animal": a, "feed": rng.choice(["DAILY", "DAILY", "WHEN_UNFED"]),
            "care": rng.choice(["DAILY", "DAILY", "OFF"]), "harvest_threshold": rng.randint(1, 8),
            "manure": rng.choice(["WHEN_AVAILABLE", "OFF"]), **window(rng)}

def gen_crop(rng):
    c = rng.choice(list(CROPS))
    return {"op": "SET_CROP", "scope": scope(rng), "crop": c, "cycles": rng.choice([None, None, rng.randint(1, 30)]),
            "harvest_threshold": rng.randint(1, 8), "watering": rng.choice(["DAILY", "DAILY", "WHEN_DRY"]),
            "fertilizing": rng.choice(["OFF", "OFF", "WHEN_EXPIRED", "DAILY"]), **window(rng)}

def gen_sale(rng):
    return {"op": "SALE", "product": rng.choice(list(PRODUCTS)), "quantity": rng.choice([100, 100, 1000, rng.randint(1, 50)]),
            "price_above": rng.choice([0, 0, 0, round(rng.uniform(0, 60), 2)]), "reserve": rng.choice([0, 0, rng.randint(0, 20)]), **window(rng)}

def gen_buy(rng):
    kind = rng.choice(["PRODUCT", "PRODUCT", "SEED", "ANIMAL"])
    item = rng.choice({"SEED": list(CROPS), "ANIMAL": list(ANIMALS), "PRODUCT": list(BUYABLE_PRODUCTS)}[kind])
    return {"op": "BUY", "kind": kind, "item": item, "quantity": rng.randint(1, 60),
            "price_below": round(rng.uniform(0.5, 60), 2), **window(rng)}

def gen_stock(rng):
    kind = rng.choice(["PRODUCT", "SEED"])
    item = rng.choice({"SEED": list(CROPS), "PRODUCT": list(BUYABLE_PRODUCTS)}[kind])
    return {"op": "STOCK", "kind": kind, "item": item, "target": rng.randint(1, 60),
            "price_below": round(rng.uniform(0.5, 60), 2), **window(rng)}

def gen_hands(rng):
    return {"op": "HANDS", "target": rng.randint(1, 6), **window(rng)}

def gen_land(rng):
    return {"op": "LAND", "target": rng.randint(0, 3), **window(rng)}

def gen_reserve(rng):
    return {"op": "CASH_RESERVE", "amount": rng.choice([0, 100, 300, 1000])}

GENERATORS = [gen_animal, gen_animal, gen_crop, gen_sale, gen_sale, gen_buy, gen_stock, gen_hands, gen_land, gen_reserve]

def random_program(rng, n=None):
    n = n or rng.randint(1, 6)
    return [rng.choice(GENERATORS)(rng) for _ in range(n)]

def mutate(rng, prog):
    p = copy.deepcopy(prog)
    r = rng.random()
    if r < 0.25 and len(p) < 30:
        p.insert(rng.randint(0, len(p)), rng.choice(GENERATORS)(rng))
    elif r < 0.40 and len(p) > 1:
        p.pop(rng.randrange(len(p)))
    elif r < 0.55 and len(p) > 1:
        i, j = rng.sample(range(len(p)), 2); p[i], p[j] = p[j], p[i]
    else:
        i = rng.randrange(len(p)); c = p[i]
        fresh = {"SET_ANIMAL": gen_animal, "SET_CROP": gen_crop, "SALE": gen_sale, "BUY": gen_buy, "STOCK": gen_stock,
                 "HANDS": gen_hands, "LAND": gen_land, "CASH_RESERVE": gen_reserve}[c["op"]](rng)
        keys = [k for k in c if k != "op"]
        k = rng.choice(keys); c[k] = fresh[k]
        if k == "start" and c.get("until") != "GAME_END" and isinstance(c["until"], int) and c["until"] < c["start"]:
            c["until"] = "GAME_END"
    return p

def crossover(rng, a, b):
    if len(a) < 2 or len(b) < 2: return copy.deepcopy(a)
    i = rng.randint(1, len(a) - 1); j = rng.randint(1, len(b) - 1)
    return copy.deepcopy(a[:i] + b[j:])[:30]

def play(args):
    prog, seed, seat = args
    from farm_neat.economy.orders import FullSeasonPlan
    try:
        env = ExactKaggricultureEnvironment(seed=seed)
        ctl = FullSeasonPlan(f"evo-{seed}-{seat}")
        obs = env.agent_observation(seat); ctl.observe(obs)
        proposal = ctl.propose(obs, [*prog, {"op": "STOP"}])
        ctl = ctl.publish(obs, proposal)
        while not env.done:
            a = ctl(env.agent_observation(seat), env.configuration)
            joint = [copy.deepcopy(PASS_ACTION), copy.deepcopy(PASS_ACTION)]; joint[seat] = a
            env.step(joint); ctl.observe(env.agent_observation(seat))
        return float(env.agent_observation(seat)["farms"][seat]["money"])
    except Exception as e:
        return -1.0

def evaluate(pool, programs, seeds):
    jobs = [(p, s, s % 2) for p in programs for s in seeds]
    res = pool.map(play, jobs, chunksize=2)
    out = []
    for i in range(len(programs)):
        vals = res[i * len(seeds):(i + 1) * len(seeds)]
        out.append(sum(vals) / len(vals))
    return out
