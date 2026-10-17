"""Gather candidate teacher programs from search outputs, dedup, rank, write {program, source, score} list.
Usage: select_teachers.py OUT.json  (reads work/datasets/search and work/datasets/corpus)
Then: clean_programs.py CLEAN.json OUT.json ; cleaned_to_hall.py CLEAN.json HALL.json
"""
import json, os, sys, collections
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the src/ directory, so farm_neat imports work when run as a script
sys.path.insert(0, _SRC)
from farm_neat.training.teachers import DATASETS as DATA


def key(p): return json.dumps([c for c in p if c.get("op") != "STOP"], sort_keys=True)


def candidates():
    cands = []
    for e in json.load(open(DATA/"search/panel4.json")):
        if e.get("eligible_for_selection") is not True:
            continue  # Legacy/partial panels have no verified completeness claim.
        cands.append({"program": e["program"], "source": "panel4", "score": e["panel_fitness"]})
    for e in json.load(open(DATA/"search/hall4.json")):
        cands.append({"program": e["program"], "source": f"hall4-gen{e.get('gen')}", "score": e["fitness"] * 0.8})  # 3-seed fitness discounted vs 6-seed panel
    for e in json.load(open(DATA/"corpus/programs-v2-clean.json")):
        cands.append({"program": e["clean"], "source": "v2clean:" + e["source"], "score": sum(e["clean_cash"]) / len(e["clean_cash"])})
    cands.append({"program": json.load(open(DATA/"search/best_program.json")), "source": "best_program", "score": 93679})
    return cands


def main(out_path):
    cands = candidates()
    seen = {}; out = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        k = key(c["program"])
        if k in seen: continue
        seen[k] = True; out.append(c)
    json.dump(out, open(out_path, "w"), indent=1)
    srcs = collections.Counter(c["source"].split(":")[0].split("-")[0] for c in out)
    print(f"{len(cands)} candidates -> {len(out)} unique; sources {dict(srcs)}; score range {out[-1]['score']:.0f}..{out[0]['score']:.0f}")
    for c in out[:30]: print(f"{c['score']:8.0f} {c['source'][:28]:28s} len {len(c['program'])}")
    return out


if __name__ == "__main__":   # importing this tool must never write a file
    main(sys.argv[1])
