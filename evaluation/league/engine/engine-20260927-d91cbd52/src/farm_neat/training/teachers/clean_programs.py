"""Teacher cleaning: drop commands that leave the scheduled-executor state unchanged or are refused,
then verify by full-executor replay (terminal cash vs PASS on 3 seeds must match the raw program)."""
import copy, json, math, os, sys
from multiprocessing import Pool
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the src/ directory, so farm_neat imports work when run as a script
sys.path.insert(0, _SRC)
from farm_neat.economy.orders import ScheduledEconomicOrders
from farm_neat.economy.action_space import validate_command
from farm_neat.evaluation.eval_vs import play

def clean(program):
    ctl = ScheduledEconomicOrders(); kept = []; events = []
    for c in program:
        if c.get("op") == "STOP": continue
        before = copy.deepcopy(ctl.snapshot())
        try:
            validate_command(c); ctl.apply(copy.deepcopy(c))
        except Exception as exc:
            events.append(("refused", str(exc)[:80])); continue
        if ctl.snapshot() == before: events.append(("no_state_change", c["op"])); continue
        kept.append(c); events.append(("kept", c["op"]))
    check = ScheduledEconomicOrders()
    for c in kept: check.apply(copy.deepcopy(c))
    assert check.snapshot() == ctl.snapshot(), "cleaning changed the scheduled executor snapshot"
    return kept, events


def assess_replays(results, seeds):
    """Only complete two-cash engine returns can attest cleaning equivalence.

    eval_vs.play returns a THIRD error field on failure. Comparing element zero
    alone would otherwise make two failed replays (-1 == -1) look verified.
    """
    if len(results) != 2*len(seeds) or not seeds:
        raise ValueError("one raw/clean replay pair per nonempty seed panel required")
    raw_cash, clean_cash, errors = [], [], []
    for i, seed in enumerate(seeds):
        for name, row, destination in (("raw", results[2*i], raw_cash),
                                        ("clean", results[2*i+1], clean_cash)):
            valid = (isinstance(row, (list, tuple)) and len(row) == 2
                     and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in row))
            if valid:
                destination.append(row[0])
            else:
                destination.append(None)
                errors.append({"seed": seed, "variant": name, "result": row})
    complete = not errors
    return {"raw_cash": raw_cash, "clean_cash": clean_cash,
            "replay_panel": [{"seed": seed, "seat": 0, "opponent": "pass"} for seed in seeds],
            "replay_complete": complete, "replay_errors": errors,
            "replay_identical": complete and raw_cash == clean_cash,
            "max_rel_diff": (max(abs(a-b)/max(1.0,abs(a))
                                 for a,b in zip(raw_cash,clean_cash)) if complete else None)}


def main(inputs, out_path, seeds=(4200000, 4200001, 4200002), workers=8):
    progs = []
    for path in inputs:
        with open(path, encoding="utf-8") as stream:
            d = json.load(stream)
        if isinstance(d, list) and d and isinstance(d[0], dict) and "op" in d[0]: d = [{"program": d}]
        items = d if isinstance(d, list) else [d]
        for it in items[:24]:
            p = it["program"] if isinstance(it, dict) and "program" in it else it
            progs.append({"source": os.path.basename(path), "raw": p})
    seen = set(); uniq = []
    for e in progs:
        k = json.dumps(e["raw"], sort_keys=True)
        if k not in seen: seen.add(k); uniq.append(e)
    for e in uniq:
        e["clean"], ev = clean(e["raw"])
        e["dropped"] = len([x for x in e["raw"] if x.get("op") != "STOP"]) - len(e["clean"])
        e["refused"] = sum(1 for t, _ in ev if t == "refused"); e["noop"] = sum(1 for t, _ in ev if t == "no_state_change")
    jobs = []
    for i, e in enumerate(uniq):
        for s in seeds: jobs.append((e["raw"], s, 0, "pass")); jobs.append((e["clean"], s, 0, "pass"))
    with Pool(workers) as pool: res = pool.map(play, jobs, chunksize=1)
    k = 0; out = []
    for e in uniq:
        e.update(assess_replays(res[k:k+2*len(seeds)], seeds)); k += 2*len(seeds)
        out.append(e)
        print(f"{e['source'][:18]:18s} raw {len([x for x in e['raw'] if x.get('op')!='STOP']):2d} -> clean {len(e['clean']):2d} (noop {e['noop']}, refused {e['refused']}) cash raw {e['raw_cash']} clean {e['clean_cash']} complete={e['replay_complete']} identical={e['replay_identical']}")
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(out, stream, indent=1)
    print("verified identical:", sum(e["replay_identical"] for e in out), "/", len(out),
          "incomplete:", sum(not e["replay_complete"] for e in out))

if __name__ == "__main__":
    main(sys.argv[2:], sys.argv[1])
