"""Convert clean_programs.py output to the {program} list record_corpus_v2 reads; keep only complete, exactly cash-equivalent replay-verified programs.

  cleaned_to_hall.py CLEAN.json HALL.json
"""
import json, math, sys


def exact_replay_verified(entry):
    """Fail closed on approximate, missing, nonfinite, or failed replay evidence."""
    if entry.get("replay_identical") is not True or entry.get("replay_complete", True) is not True:
        return False
    if entry.get("replay_errors") or entry.get("max_rel_diff") != 0.0:
        return False
    raw, clean = entry.get("raw_cash"), entry.get("clean_cash")
    return (isinstance(raw, list) and isinstance(clean, list) and bool(raw)
            and len(raw) == len(clean)
            and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in raw+clean)
            and raw == clean)


def main(clean_path, hall_path):
    with open(clean_path, encoding="utf-8") as stream:
        d = json.load(stream)
    out = [{"program": e["clean"], "source": e["source"], "clean_cash": e["clean_cash"]} for e in d if exact_replay_verified(e)]
    if not out:
        raise ValueError("no complete exactly replay-verified teacher; output not written")
    out.sort(key=lambda e: -sum(e["clean_cash"]) / len(e["clean_cash"]))
    with open(hall_path, "w", encoding="utf-8") as stream:
        json.dump(out, stream, indent=1)
    print(f"{len(out)}/{len(d)} verified programs written; mean clean cash top: {[round(sum(e['clean_cash'])/len(e['clean_cash'])) for e in out[:10]]}")


if __name__ == "__main__":   # importing this tool must never write a file
    main(sys.argv[1], sys.argv[2])
