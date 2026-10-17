#!/usr/bin/env python3
"""Seed checkpoint for the top-4 top-up: the P2 candidate's checkpoint with raw weights := its EMA (the deployed
policy), EMA := the same EMA, optimizer/RNG/args kept. The top-up trajectory therefore starts at exactly the policy that
nflow-u16000 ships.   python3 make_t4_seed.py SRC_CKPT OUT_CKPT"""
import hashlib
import io
import sys

import torch

src, out = sys.argv[1:3]
raw = open(src, "rb").read()
b = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
ema = b["ema"]
model = {}
for k, v in b["model"].items():
    e = ema[k]
    model[k] = e.to(v.dtype).clone() if v.is_floating_point() else v
b["model"] = model
b["ema"] = {k: v.clone() for k, v in ema.items()}
b.setdefault("extra", {})["t4_seed"] = {"source": src, "source_sha256": hashlib.sha256(raw).hexdigest(),
                                         "raw_from": "ema", "ema_from": "ema"}
# This is a new fork: its direct parent supersedes inherited ancestry.
# Keep both keys consistent because train.py prefers extra.base over extra.t4_seed.
b["extra"]["base"] = dict(b["extra"]["t4_seed"])
torch.save(b, out + ".tmp")
import os
os.replace(out + ".tmp", out)
print(out, "update", b["update"])
