"""Held-out loss of several checkpoints on ONE fixed set of windows (owner 2026-09-27: pilots judged by best "loss" and best
"play"). The windows come from the val split of a fixed train config (default: pilot A's, i.e. the v5s usability rows), drawn
once with a fixed seed and reused for every checkpoint, so the numbers are comparable.

  work/venv/python -B eval_fixed.py OUT.json --data-from CKPT_OR_DIR --windows 256 CKPT_OR_DIR [CKPT_OR_DIR ...]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

import model as M  # noqa: E402
import train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--data-from", required=True)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(4)
    blob, _ = T.load_checkpoint(a.data_from)
    cfg = T.TrainConfig.from_dict(blob["train_config"])
    cfg.workers = a.workers
    cfg.bf16 = False
    net0 = T.build(M.ModelConfig.from_dict(blob["config"]), "cpu")
    batches = list(T.held_out_batches(net0, cfg, windows=a.windows))
    out = {"data_from": str(a.data_from), "windows": a.windows, "split": cfg.eval_split, "results": []}
    for c in a.ckpts:
        b, path = T.load_checkpoint(c)
        net = T.build(M.ModelConfig.from_dict(b["config"]), "cpu")
        net.load_state_dict(b["model"])
        rep = T.evaluate(net, batches, cfg, torch.device("cpu"), free_run=False)
        row = {"checkpoint": str(path), "update": b["update"], "loss_per_step": rep["loss_per_step"],
               **{k: v for k, v in rep.items() if k.startswith("acc_")}}
        out["results"].append(row)
        print(json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
