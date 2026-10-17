"""Evolution v4: search3 generators + joint economic mutations + fixed held-out panel ranking of the hall.
Seeds: cleaned teachers (work/datasets/corpus/programs-v2-clean.json) and the previous hall."""
import copy, json, math, os, random, sys, time
from multiprocessing import Pool
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the src/ directory, so farm_neat imports work when run as a script
sys.path.insert(0, _SRC)
from farm_neat.game.constants import ANIMALS
from farm_neat.training.teachers import DATASETS, output_dir, search as s1, search2 as s2, search3 as s3
from farm_neat.evaluation.eval_vs import play as play_vs
OUT = output_dir("work/datasets/search")  # generated halls/panels/log; never the source tree
PRODUCT = {a: ANIMALS[a]["product"] for a in ANIMALS}
PANEL = [4300000 + i for i in range(6)]

def free_tile(rng, prog):
    used = {(c["scope"]["x"], c["scope"]["y"]) for c in prog if c.get("scope", {}).get("kind") == "TILE"}
    for _ in range(50):
        t = (rng.randint(0, 9), rng.randint(0, 9))
        if t not in used: return {"kind": "TILE", "x": t[0], "y": t[1]}
    return {"kind": "TILE", "x": rng.randint(0, 9), "y": rng.randint(0, 9)}

def joint_mutate(rng, prog):
    """Economic, coordinated edits: production + disposal, staffing for a cluster, crop cluster + sale, land."""
    p = copy.deepcopy(prog); r = rng.random()
    if r < 0.30:   # add a serviced animal and make sure its product has a SALE rule
        a = s3.gen_animal(rng); a["scope"] = free_tile(rng, p); p.append(a)
        if not any(c["op"] == "SALE" and c["product"] == PRODUCT[a["animal"]] for c in p):
            s = s3.gen_sale(rng); s["product"] = PRODUCT[a["animal"]]; p.append(s)
    elif r < 0.55:  # add a crop cluster (2-5 tiles, same crop, same settings) and its SALE
        c = s3.gen_crop(rng); n = rng.randint(2, 5)
        for _ in range(n):
            d = copy.deepcopy(c); d["scope"] = free_tile(rng, p); p.append(d)
        if not any(x["op"] == "SALE" and x["product"] == c["crop"] for x in p):
            s = s3.gen_sale(rng); s["product"] = c["crop"]; p.append(s)
    elif r < 0.75:  # scale staffing with production: bump HANDS by +1..+4 (or add HANDS), keep it early
        hs = [c for c in p if c["op"] == "HANDS"]
        if hs: hs[0]["target"] = min(14, hs[0]["target"] + rng.randint(1, 4))
        else: p.insert(0, {"op": "HANDS", "target": rng.randint(6, 11), "start": 0, "until": "GAME_END", "priority": 0})
    elif r < 0.88:  # land expansion (more tiles for the cluster)
        ls = [c for c in p if c["op"] == "LAND"]
        if ls: ls[0]["target"] = min(3, ls[0]["target"] + 1)
        else: p.insert(0, {"op": "LAND", "target": rng.randint(1, 3), "start": 0, "until": "GAME_END", "priority": 1})
    else:           # retune a sale threshold or reserve (market policy), keep the product
        ss = [c for c in p if c["op"] == "SALE"]
        if ss:
            s = rng.choice(ss); fresh = s3.gen_sale(rng); s["price_above"] = fresh["price_above"]; s["reserve"] = fresh["reserve"]
    return p[:48]

def mutate(rng, prog):
    return joint_mutate(rng, prog) if rng.random() < 0.5 else s3.mutate(rng, prog)

def panel_row(program, jobs, results):
    """Keep every scheduled identity; an incomplete panel cannot rank a teacher."""
    if (len(jobs) != len(results) or len(jobs) != 6
            or sorted(j[3] for j in jobs) != ['pass']*3+['router']*3):
        raise ValueError("one result per scheduled panel game required")
    rows = []
    for (_, seed, seat, opponent), value in zip(jobs, results, strict=True):
        row = {"seed": seed, "seat": seat, "opponent": opponent}
        if (isinstance(value, (list, tuple)) and len(value) == 2
                and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in value)):
            row.update(own_cash=value[0], opponent_cash=value[1])
        else:
            row["error"] = repr(value)
        rows.append(row)
    if len({(r['seed'], r['seat'], r['opponent']) for r in rows}) != len(rows):
        raise ValueError("duplicate panel identity")
    complete = all("error" not in row for row in rows)
    def mean(opponent, field):
        values = [r[field] for r in rows if r['opponent'] == opponent and 'error' not in r]
        return sum(values)/len(values) if values else None
    return {"schema": "farm-teacher-fixed-panel-v2", "program": program,
            "complete": complete, "eligible_for_selection": complete,
            "requested_games": len(jobs), "completed_games": sum('error' not in r for r in rows),
            "per_game": rows,
            "panel_fitness": sum(r['own_cash'] for r in rows)/len(rows) if complete else None,
            "vs_router_own": mean('router', 'own_cash'),
            "vs_router_opp": mean('router', 'opponent_cash'),
            "vs_pass_own": mean('pass', 'own_cash')}

def main():
    pop_size = int(os.environ.get("POP", 96)); gens = int(os.environ.get("GENS", 400)); workers = int(os.environ.get("WORKERS", 22))
    rng = random.Random(20260914); log = open(os.path.join(OUT, "gen4.log"), "a")
    clean = [e["clean"] for e in json.load(open(DATASETS / "corpus/programs-v2-clean.json"))]
    prior = DATASETS / "search/hall3.json"   # the previous generation's hall, if it was kept
    h3 = [h["program"] for h in json.load(open(prior))[:12]] if prior.exists() else []
    seeds_pop = clean[:12] + h3
    pop = list(copy.deepcopy(seeds_pop))
    while len(pop) < pop_size: pop.append(mutate(rng, rng.choice(seeds_pop)))
    hall = []; panel = {}
    with Pool(workers) as pool:
        for g in range(gens):
            t = time.time(); seeds = [3300000 + g * 10 + i for i in range(3)]
            fit, detail = s3.evaluate(pool, pop, seeds)
            order = sorted(range(len(pop)), key=lambda i: -fit[i]); b = order[0]
            for i in order[:3]: hall.append({"gen": g, "fitness": fit[i], "detail": detail[i], "program": pop[i], "seeds": seeds})
            hall = sorted(hall, key=lambda h: -h["fitness"])[:80]
            json.dump(hall, open(os.path.join(OUT, "hall4.json"), "w"), indent=1)
            line = f"gen {g} best {fit[b]:.0f} (vsRouter {detail[b][0][0]:.0f},{detail[b][1][0]:.0f} vsPass {detail[b][2][0]:.0f}; router got {detail[b][0][1]:.0f}) top10 {sum(fit[i] for i in order[:10])/10:.0f} mean {sum(fit)/len(fit):.0f} len {len(pop[b])} secs {time.time()-t:.1f}"
            print(line, flush=True); log.write(line + "\n"); log.flush()
            if g % 10 == 9:   # held-out panel: rank the hall on fixed seeds (2 router + 4 pass per program is too costly; use 3 router + 3 pass)
                cand = []; keys = set()
                for h in hall:
                    k = json.dumps(h["program"], sort_keys=True)
                    if k not in keys: keys.add(k); cand.append(h["program"])
                    if len(cand) >= 16: break
                jobs = [(p, sd, sd % 2, "router" if i < 3 else "pass") for p in cand for i, sd in enumerate(PANEL)]
                res = pool.map(play_vs, jobs, chunksize=1)
                rows = []
                for i, p in enumerate(cand):
                    rows.append(panel_row(p, jobs[i*6:(i+1)*6], res[i*6:(i+1)*6]))
                rows.sort(key=lambda x: (not x['complete'], -(x['panel_fitness'] or 0)))
                with open(os.path.join(OUT, "panel4.json"), "w") as stream:
                    json.dump(rows, stream, indent=1, allow_nan=False)
                pl = f"panel@gen{g}: incomplete {sum(not r['complete'] for r in rows)}; " + " | ".join(f"{r['panel_fitness']:.0f} (R {r['vs_router_own']:.0f}/{r['vs_router_opp']:.0f}, P {r['vs_pass_own']:.0f})" for r in rows if r['complete'])
                print(pl, flush=True); log.write(pl + "\n"); log.flush()
            elites = [pop[i] for i in order[:pop_size // 8]]; children = list(copy.deepcopy(elites))
            while len(children) < pop_size:
                r = rng.random()
                if r < 0.2: children.append(s1.crossover(rng, rng.choice(elites), rng.choice(pop))[:48])
                else: children.append(mutate(rng, rng.choice(elites)))
            pop = children
if __name__ == "__main__":
    main()
