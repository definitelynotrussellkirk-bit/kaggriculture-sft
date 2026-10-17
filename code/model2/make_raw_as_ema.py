#!/usr/bin/env python3
"""Raw-weights release checkpoint (ASTRA NFLOW-IMPROVE E7): EMA := the checkpoint's raw weights, so the unchanged
release chain (which exports the EMA) ships the raw policy.   python3 make_raw_as_ema.py SRC_CKPT OUT_CKPT"""
import hashlib, os, sys
import torch
src, out = sys.argv[1:3]
b = torch.load(src, map_location="cpu", weights_only=False)
b["ema"] = {k: (v.detach().clone() if v.is_floating_point() else v.clone()) for k, v in b["model"].items()}
b.setdefault("extra", {})["raw_export"] = {"source": src, "source_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
                                           "ema_from": "model (raw)"}
torch.save(b, out + ".tmp"); os.replace(out + ".tmp", out)
print(out, "update", b["update"])
