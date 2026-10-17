#!/usr/bin/env python3
"""Review TRAINING-DATA #1 witness: a winning roster row loaded from the WRONG seat file must be refused from the file's
own content, not accepted from the roster's stamps.
  witness: row 111017926-1 (act, W) with its path pointed at 113911274-0.stp (a wdl-loser seat of another game)
  control: the same row with its own file loads; the loser row with its own file loads
  swap:    the loser row pointed at its own game's winning seat (113911274-1) is refused (seat field)
Uses the live roster (fingerprints from catalog SEAT_DAYS)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nf_data  # noqa: E402
import nf_format as NF  # noqa: E402

R = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ai/scratch/farmNEAT/n-flow/roster/rows-nflow-v3.fp-test.tsv"
C = "/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11"
rows = {(r["eid"], r["seat"]): r for r in NF.read_roster(R)}
cols = NF.load_columns()
win, lose = rows[(111017926, 1)], rows[(113911274, 0)]
assert win["fp"] and lose["fp"], "roster rows must carry catalog fingerprints"
fails = []


def loads(row, path):
    r = dict(row, path=path, _src="stp")
    try:
        nf_data.load_seat(r, None, cols)
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


ok, why = loads(win, f"{C}/113911274-0.stp")
print("witness winner<-other game's loser file:", "REFUSED" if not ok else "ACCEPTED", why[:120])
if ok:
    fails.append("witness accepted")
ok, why = loads(lose, f"{C}/113911274-1.stp")
print("loser row<-own game's winner file:", "REFUSED" if not ok else "ACCEPTED", why[:120])
if ok:
    fails.append("seat swap accepted")
for row, p in ((win, f"{C}/111017926-1.stp"), (lose, f"{C}/113911274-0.stp")):
    ok, why = loads(row, p)
    print(f"control {row['eid']}-{row['seat']} own file:", "LOADED" if ok else f"REFUSED {why[:120]}")
    if not ok:
        fails.append(f"control {row['eid']}-{row['seat']} refused")
print("PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
