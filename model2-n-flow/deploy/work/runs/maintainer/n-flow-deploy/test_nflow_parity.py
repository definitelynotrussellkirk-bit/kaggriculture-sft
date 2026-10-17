#!/usr/bin/env python3
"""Serve/train parity for N flow: the play path (model.PolicyNet.act: data.features at play time + nf_format encoder +
step-by-step memory + sequential K/V-cached decoder + PrefixExecutor prefix rows) gives the same logits as the
training path (nf_model.forward_chunk on the stored shard, teacher forcing) for every decision of the first T steps of
a real replay seat, when the play path is fed the recorded decisions (force=labels). Then greedy play is replayed
twice from the same observations and must give identical actions (determinism), and its first decision at each step
must equal the argmax of the training-path logits.

  python3 test_nflow_parity.py [--steps 24] [--row 0] [--ckpt CKPT_or_weights]
Default model: fresh random init (seed 0). fp32 CPU both paths.
"""
import argparse
import json
import math
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (str(ROOT / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import model as MD  # noqa: E402
import nf_format as NF  # noqa: E402
import nf_model as M  # noqa: E402

ROSTER = "/mnt/ai/scratch/farmNEAT/n-flow/roster/rows-nflow.1ad4d8296e84.tsv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)   # crosses the first dawn (step 24)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tol", type=float, default=2e-3)
    a = ap.parse_args()
    if not math.isfinite(a.tol) or a.tol < 0:
        ap.error("--tol must be finite and nonnegative")
    if not 1 <= a.steps <= 719:
        ap.error("--steps must be in 1..719")
    torch.manual_seed(0)
    net = MD.PolicyNet.load(a.ckpt) if a.ckpt else MD.PolicyNet().eval()
    net.eval()
    from farm_neat.training.bc import public_observation
    from v8_gaps import GapsState
    cols = MD.columns()
    rows = [r for r in NF.read_roster(ROSTER) if r["act_loss"]]
    row = rows[a.row]
    with zipfile.ZipFile(row["archive"]) as z:
        rep = json.loads(z.read(row["member"]))
    seat = row["seat"]
    sh = NF.compact_seat(NF.read_stp(row["path"]), cols, row)
    T = a.steps
    x = {k: torch.from_numpy(np.ascontiguousarray(v[:T]))[None] for k, v in sh.items() if v.ndim >= 1}
    with torch.no_grad():
        out, _ = net.forward_chunk(x, net.initial_memory(1, "cpu"))
    cfg = {k: v for k, v in rep["configuration"].items()}
    import agent as AG
    cfg = AG.public_configuration(cfg)
    worst, n_cmp = 0.0, 0
    fails = []
    runs = {}
    for mode in ("force", "greedy1", "greedy2"):
        state, hist, gl, actions = None, [], GapsState(), []
        for t in range(T):
            obs = public_observation(rep, t, seat)
            clean = AG.actor_view(obs)
            force = None
            if mode == "force":
                force = {k: [int(v) for v in sh[k][t]] for k in ("unit_tok", "unit_qty", "order_tok", "order_qty")}
            tr = {}
            action, state = net.act(clean, state=state, configuration=cfg, v8_history=list(hist), v8_long=gl,
                                    force=force, trace=tr)
            hist = [clean, *hist[:1]]
            actions.append(action)
            if mode == "greedy1" and t == 0:
                step0_greedy = int(next(lg for kd, kk, lg in tr["logits"] if kd == "unit_tok" and kk == 0).argmax())
            if mode == "force":
                for kind, k, logits in tr["logits"]:
                    ref = out[{"unit_tok": "unit_logits", "unit_qty": "unit_qty_logits", "order_tok": "order_logits",
                               "order_qty": "order_qty_logits"}[kind]][0, t, k]
                    actual = logits[0].float()
                    fin = torch.isfinite(ref)
                    n_cmp += 1
                    if (actual.shape != ref.shape or not bool(fin.any())
                            or bool(torch.isnan(actual).any()) or bool(torch.isnan(ref).any())
                            or bool(torch.isposinf(actual).any()) or bool(torch.isposinf(ref).any())
                            or not torch.equal(torch.isfinite(actual), fin)):
                        fails.append((t, kind, k, "nonfinite logits or mask mismatch"))
                        continue
                    d = (actual[fin] - ref.float()[fin]).abs().max().item()
                    if not math.isfinite(d):
                        fails.append((t, kind, k, "nonfinite logit difference"))
                        continue
                    worst = max(worst, d)
                    if d > a.tol:
                        fails.append((t, kind, k, d))
        runs[mode] = actions
    greedy1, greedy2 = runs["greedy1"], runs["greedy2"]
    if greedy1 != greedy2:
        fails.append(("greedy play not deterministic",))
    # first decision of each greedy step vs training-path argmax (same observation, same memory: history differs after
    # step 0 because greedy actions differ from the recorded ones, so only step 0 is comparable exactly)
    a0 = int(out["unit_logits"][0, 0, 0].argmax())
    g0 = step0_greedy
    first_match = a0 == g0
    if not first_match:
        fails.append(("step-0 farmer greedy token differs from training argmax", a0, g0))
    print(json.dumps({"eid": row["eid"], "seat": seat, "steps": T, "decisions_compared": n_cmp,
                      "max_abs_logit_diff": worst, "tol": a.tol, "greedy_deterministic": greedy1 == greedy2,
                      "step0_farmer_argmax": [a0, int(g0)], "fails": fails[:5]}))
    print("PASS" if not fails else f"FAIL {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
