#!/usr/bin/env python3
"""GatherAhead (throughput only) yields exactly gather()'s batches in the same order.
  python3 test_gather_ahead.py [ROSTER] [N_BATCHES]      (CPU; real loader, 2 workers x 2 streams, chunk 24)"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import nf_data  # noqa: E402
import nf_format as NF  # noqa: E402

ROSTER = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ai/scratch/farmNEAT/n-flow/t4b/rows-t4b.tsv"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
STP = "/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11"
COLS = NF.load_columns(os.path.join(os.path.dirname(os.path.abspath(__file__)), "COLUMNS.json"))


def batches(ahead):
    dl = nf_data.loader(ROSTER, "/mnt/ai/scratch/farmNEAT/n-flow/shards-v1", COLS, workers=2, streams=2, chunk=24, seed=7,
                        stp_dir=STP)
    it = iter(dl)
    if ahead:
        g = nf_data.GatherAhead(it, 2, depth=3)
        return [g.next() for _ in range(N)]
    return [nf_data.gather(it, 2) for _ in range(N)]


a, b = batches(False), batches(True)
for i, (x, y) in enumerate(zip(a, b)):
    assert set(x) == set(y), (i, set(x) ^ set(y))
    for k in x:
        assert torch.equal(x[k], y[k]), (i, k)
print(f"PASS: {N} batches identical ({len(a[0])} keys, batch rows {a[0]['step'].shape[0] if 'step' in a[0] else '?'})")
