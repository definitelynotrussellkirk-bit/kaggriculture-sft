#!/usr/bin/env python3
"""No action (imitation) loss is ever computed on a WDL-only (losing-seat) row.

  python3 test_wdl_only.py [--roster R] [--shards DIR]

Real roster rows through the real loader and train.chunk_loss on a small fresh CPU model:
  1. every action mask (act, score_u, has_uq, score_o, has_oq) is all-False on rows with act_loss = 0;
  2. the imitation loss and its gradient do not change when every action label of the WDL-only rows is scrambled;
  3. the WDL loss does see those rows (its mask covers them), and act rows are all catalog winners (wdl = 2);
  4. every WDL-only roster row is a non-winner and every act row a winner.
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nf_data  # noqa: E402
import nf_format as NF  # noqa: E402
import train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", default="/mnt/ai/scratch/farmNEAT/n-flow/roster/rows-nflow.1ad4d8296e84.tsv")
    ap.add_argument("--shards", default="")
    a = ap.parse_args()
    cols = NF.load_columns()
    rows = NF.read_roster(a.roster)
    bad_roster = [r for r in rows if (r["act_loss"] and r["wdl"] != 2) or (not r["act_loss"] and r["wdl"] == 2)]
    assert not bad_roster, f"roster rows with act/wdl mismatch: {bad_roster[:3]}"
    act = [r for r in rows if r["act_loss"]][:3]
    wdl = [r for r in rows if not r["act_loss"]][:3]
    ds = nf_data.Streams(act + wdl, a.shards or None, cols, streams=6, chunk=6, seed=3, wdl_frac=0.5, first_random=False)
    x = nf_data.gather(iter(ds), 1)
    x.pop("eid"); x.pop("seat")
    assert bool(x["act_loss"].any()) and bool((~x["act_loss"]).any()), "batch needs both row kinds"
    torch.manual_seed(0)
    net = T.build_net(cols, NF.PRIMITIVE_BC)
    net.train()

    def run(xx):
        net.zero_grad(set_to_none=True)
        mem = T.mem_reset(net.initial_memory(xx["step"].shape[0], "cpu"), xx["reset"])
        out, _ = net.forward_chunk(xx, mem)
        _, st = T.chunk_loss(out, xx, net)
        m = T.masks(xx, net)
        imit = st["imit"]
        # imitation-only gradient
        net.zero_grad(set_to_none=True)
        mem = T.mem_reset(net.initial_memory(xx["step"].shape[0], "cpu"), xx["reset"])
        out, _ = net.forward_chunk(xx, mem)
        loss, st2 = T.chunk_loss(out, xx, net, w_wdl=0.0, w_next=0.0, w_opp=0.0)
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])
        return m, float(imit), g

    m, imit, g = run(x)
    wd = ~x["act_loss"].bool()
    for k in ("act", "score_u", "has_uq", "score_o", "has_oq"):
        assert not bool(m[k][wd].any()), f"action mask {k} is set on a WDL-only row"
    assert bool(m["valid"][wd].any()), "WDL loss must see WDL-only rows"
    assert bool((x["wdl"][x["act_loss"].bool()] == 2).all()), "act rows must be winners"
    y = {k: v.clone() for k, v in x.items()}
    gen = torch.Generator().manual_seed(1)
    for k, hi in (("unit_tok", 44), ("unit_qty", 75), ("order_tok", 23), ("order_qty", 75)):
        r = torch.randint(0, hi, y[k].shape, generator=gen).to(y[k].dtype)
        y[k][wd] = r[wd]
    y["unit_loss_keep"][wd] = True
    m2, imit2, g2 = run(y)
    for k in ("act", "score_u", "has_uq", "score_o", "has_oq"):
        assert not bool(m2[k][wd].any()), f"action mask {k} set on a WDL-only row after label scramble"
    # teacher-forced prefixes of WDL-only rows change their own outputs but never enter the imitation loss
    assert abs(imit - imit2) < 1e-5, (imit, imit2)
    assert torch.allclose(g, g2, atol=1e-6), "imitation gradient depends on WDL-only action labels"
    print(f"rows: act={int(x['act_loss'].sum())} wdl_only={int(wd.sum())}; action-mask cells on WDL-only rows: 0; "
          f"imitation loss {imit:.6f} == {imit2:.6f} after scrambling WDL-only labels; gradient identical")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
