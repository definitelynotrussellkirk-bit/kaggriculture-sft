#!/usr/bin/env python3
"""Checkpoint whose "model" is the EMA weights of a training checkpoint (TrainConfig.ema_decay > 0), for eval/deploy tools.
  python ema_export.py CKPT_OR_RUN_DIR OUT.pt"""
import sys
from pathlib import Path

import torch

import train as T


def main(src, out):
    blob, path = T.load_checkpoint(src)
    if "ema" not in blob:
        raise SystemExit(f"{path}: no EMA weights (trained without ema_decay)")
    model = dict(blob["model"])
    for k, v in blob["ema"].items():
        model[k] = v.to(model[k].dtype)
    blob = {k: v for k, v in blob.items() if k not in ("ema", "optimizer")} | {"model": model, "ema_of": str(path)}
    torch.save(blob, out)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
