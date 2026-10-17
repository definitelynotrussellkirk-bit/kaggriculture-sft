"""Scale-out driver for repair_search.py: corpus v2 states + multi-producer campaign states.

repair_search.main() hard-codes one corpus path and one producer directory; this only
chooses the state sets and calls repair_search.run() unchanged (same candidate generator,
same paired-seed Delta, same output writer).

  corpus    <programs> x <seeds> games of corpus/games-v2.jsonl.gz at steps 144,288,432
            (step 576 is dropped: corpus-v1 measured a 0/40 hit rate there)
  campaign  worlds pooled over several producers of a fast-campaign round

Usage:
  python3 repair_at_scale.py corpus   tag=corpus-v2 programs=51 seeds=4 workers=24
  python3 repair_at_scale.py campaign tag=campaign-r004 producers=DIR1,DIR2 worlds=20 workers=24
"""
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the src/ directory, so farm_neat imports work when run as a script
sys.path.insert(0, _SRC)

from farm_neat.training.teachers import DATASETS
from farm_neat.training.teachers import repair_search as rs

CORPUS_V2 = str(DATASETS / "corpus/games-v2.jsonl.gz")


def main(argv):
    mode = argv[1]
    args = {}
    for token in argv[2:]:
        key, _, value = token.partition("=")
        args[key.lstrip("-")] = value
    workers = int(args.get("workers", 24))
    count = int(args.get("candidates", 8))
    threshold = float(args.get("threshold", 2000))
    if mode == "corpus":
        steps = [int(s) for s in args.get("steps", "144,288,432").split(",")]
        specs = rs.corpus_specs(path=args.get("corpus", CORPUS_V2),
                                programs=int(args.get("programs", 51)),
                                seeds=int(args.get("seeds", 4)))
        tag = args.get("tag", "corpus-v2")
    elif mode == "campaign":
        steps = [int(s) for s in args.get("steps", "144,288").split(",")]
        producers = args["producers"].split(",")
        specs = []
        for producer in producers:
            specs.extend(rs.campaign_specs(producer, int(args.get("worlds", 20))))
        tag = args.get("tag", "campaign-scale")
    else:
        raise SystemExit("modes: corpus | campaign")
    print(f"{mode}: {len(specs)} games x {len(steps)} steps = {len(specs)*len(steps)} states, "
          f"{count} candidates, {workers} workers", flush=True)
    rs.run(tag, specs, steps, count, workers, threshold)


if __name__ == "__main__":
    main(sys.argv)
