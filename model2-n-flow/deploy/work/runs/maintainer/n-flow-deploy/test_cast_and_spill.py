#!/usr/bin/env python3
"""Audit 09-30 low-severity items: (1) step_inputs clips (and counts) out-of-range integer inputs instead of wrapping;
(2) nf_format.slot_index counts carriers with ordinal >= 25 (no node slot) instead of dropping them silently.
In-range inputs are unchanged."""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "src"))
sys.path.insert(0, str(HERE))
import model as MD  # noqa: E402
import nf_format as NF  # noqa: E402

fails = []
N = 272
feats = {"rows": np.zeros((N, 80), np.float16), "knownness": np.zeros((N, 80), bool), "token_mask": np.zeros(N, bool),
         "extra": np.zeros((N, 152), np.float16), "v6_token": np.zeros((N, 8), np.float16), "x8_token": np.zeros((N, 354), np.float16),
         "unit_pos": np.full((25, 2), -1, np.int64), "unit_extra": np.zeros((25, 12), np.float32),
         "x8_unit": np.zeros((25, 41), np.float32), "reach_eta": np.zeros((25, 100), np.int64),
         "v6_job_deadline": np.zeros((100, 9), np.int64), "v6_job_deadline_present": np.zeros((100, 9), bool)}
feats["rows"][0, 0] = 1; feats["token_mask"][0] = True           # a GLOBAL token
pu, po = torch.full((1, 2, 25, 2), -1), torch.full((1, 2, 11, 2), -1)
x = MD.step_inputs(feats, pu, po, 5)
if MD.CAST_CLIPS:
    fails.append(f"in-range inputs clipped: {MD.CAST_CLIPS}")
feats["v6_job_deadline"][3, 2] = 200
feats["reach_eta"][0, 0] = 40000
x = MD.step_inputs(feats, pu, po, 5)
if int(x["v6_job_deadline"][0, 3, 2]) != 127 or int(x["reach_eta"][0, 0, 0]) != 32767:
    fails.append("out-of-range values were not clipped to the dtype limit")
if MD.CAST_CLIPS.get("v6_job_deadline") != 1 or MD.CAST_CLIPS.get("reach_eta") != 1:
    fails.append(f"clip counts wrong: {MD.CAST_CLIPS}")
rows = np.zeros((1, 3, 80), np.float32); mask = np.ones((1, 3), bool)
for j, o in enumerate((0, 24, 26)):
    rows[0, j, 3] = 1; rows[0, j, 12] = 1; rows[0, j, 16] = o / 100       # own CARRIER ordinals 0, 24, 26
NF.DROPS.clear()
sl = NF.slot_index(rows, mask)
if sl[0].tolist() != [203, 227, -1] or NF.DROPS.get("carrier_ordinal_ge_25") != 1:
    fails.append(f"spill not counted: slots {sl[0].tolist()} drops {NF.DROPS}")
print({"cast_clips": MD.CAST_CLIPS, "drops": NF.DROPS, "fails": fails})
print("PASS" if not fails else "FAIL")
sys.exit(1 if fails else 0)
