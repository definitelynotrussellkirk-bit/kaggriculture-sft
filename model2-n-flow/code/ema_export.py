#!/usr/bin/env python3
"""N flow training checkpoint -> weights file holding the EMA (or raw) weights: {"schema", "config", "model"}.

  python3 ema_export.py CKPT OUT [--raw]
"""
import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--raw", action="store_true", help="export the trained weights instead of the EMA")
    a = ap.parse_args()
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if blob.get("schema") != "nflow-policy-v1":
        sys.exit(f"not an N flow checkpoint: {blob.get('schema')!r}")
    src = blob["model"] if a.raw else blob["ema"]
    model = {k: (src[k].float() if src[k].is_floating_point() else src[k]) for k in blob["model"]}
    bad = [k for k, v in model.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
    if bad:     # review VERIFY-FIX: never export nonfinite weights
        sys.exit(f"nonfinite {'raw' if a.raw else 'EMA'} weights in {a.ckpt}: {bad[:5]}")
    torch.save({"schema": blob["schema"], "config": blob["config"], "model": model,
                "extra": {"source": a.ckpt, "update": blob["update"], "weights": "raw" if a.raw else "ema"}}, a.out)
    print(f"{a.out}: update {blob['update']} ({'raw' if a.raw else 'ema'})")


if __name__ == "__main__":
    main()
