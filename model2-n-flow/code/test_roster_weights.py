#!/usr/bin/env python3
"""Roster WDL weights and row kinds.

  python3 test_roster_weights.py [ROSTER]
1. wdl_w is finite and > 0 for every row, mean 1 over the roster, and equals 1/(1+|margin|/M) up to one global scale
   (M = median |margin|): close games weigh more, landslides still count.
2. act rows are all catalog winners with act_loss 1; wdl-loser and own-loss rows have act_loss 0 (no action loss);
   own-loss rows are losses (wdl 0).
3. the loader passes each row's wdl_w and act_loss unchanged (checked on a small real batch incl. an own-loss row), and
   train.masks gives zero action-mask cells on every own-loss / wdl-loser stream.
"""
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nf_format as NF  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ai/scratch/farmNEAT/n-flow/roster/rows-nflow.1ad4d8296e84.tsv"
rows = NF.read_roster(path)
ws = [r["wdl_w"] for r in rows]
assert all(math.isfinite(w) and w > 0 for w in ws), "non-finite or non-positive wdl_w"
mean = sum(ws) / len(ws)
assert abs(mean - 1) < 1e-3, mean
m = [abs(r["final_cash"] - r["opp_cash"]) for r in rows]
M = statistics.median(m)
ratio = [w * (1 + mm / M) for w, mm in zip(ws, m)]
assert max(ratio) / min(ratio) < 1 + 1e-4, "weights do not follow 1/(1+|margin|/M)"
kinds = {}
for r in rows:
    kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    if r["kind"] == "act":
        assert r["act_loss"] == 1 and r["wdl"] == 2
    else:
        assert r["act_loss"] == 0, r
    if r["kind"] == "own-loss":
        assert r["wdl"] == 0, r
ess = sum(ws) ** 2 / sum(w * w for w in ws)
print(f"rows {len(rows)} kinds {kinds}; wdl_w min {min(ws):.4f} max {max(ws):.4f} mean {mean:.6f} ESS {ess:.1f}; M={M}")

import torch  # noqa: E402
import nf_data  # noqa: E402
import train as T  # noqa: E402
cols = NF.load_columns()
own = [r for r in rows if r["kind"] == "own-loss"][:2]
los = [r for r in rows if r["kind"] == "wdl-loser"][:2]
act = [r for r in rows if r["kind"] == "act"][:2]
pick = own + los + act
ds = nf_data.Streams(pick, None, cols, streams=len(pick), chunk=4, seed=1, wdl_frac=0.5, first_random=False)
x = nf_data.gather(iter(ds), 1)
by = {(r["eid"], r["seat"]): r for r in pick}
for i in range(len(pick)):
    r = by[(int(x["eid"][i]), int(x["seat"][i]))]
    assert abs(float(x["wdl_w"][i]) - r["wdl_w"]) < 1e-6 and bool(x["act_loss"][i]) == bool(r["act_loss"])
net = T.build_net(cols, HERE / "pbc/work/runs/maintainer")
mk = T.masks({k: v for k, v in x.items() if k not in ("eid", "seat")}, net)
wd = ~x["act_loss"].bool()
for k in ("act", "score_u", "has_uq", "score_o", "has_oq"):
    assert not bool(mk[k][wd].any()), k
print(f"batch kinds {[by[(int(e), int(s))]['kind'] for e, s in zip(x['eid'], x['seat'])]}: action-mask cells on "
      f"non-act rows 0; wdl_w and act_loss passed through")
print("PASS")
