"""fp32 vs int8-stored weights for one checkpoint (owner 2026-09-27: "use int8 then"; "bigger model ... downcast to int8
should be fine, right?"). Rounds every weight exactly as the archive does (agent.export_weights: int8 per output row for tensors
with >= 4096 values, fp16 otherwise, expanded back to fp32 at load), then plays both copies through the deploy agent on the
same recorded observations and reports how often the submitted action differs, per step and per unit command.

  python int8_check.py CKPT.pt REPLAY.json[.gz] [REPLAY ...] --steps 60 --out OUT.json
"""
import argparse
import gzip
import json
import sys
import tempfile
from pathlib import Path

DEPLOY = Path("/home/russ/Desktop/farmNEAT/work/runs/maintainer/primitive-bc-v1-deploy-v5w-a1")
sys.path[:0] = [str(Path("/home/russ/Desktop/farmNEAT/src")), str(DEPLOY)]
import torch  # noqa: E402

import agent as AG  # noqa: E402


def roundtrip(blob, min_numel=4096):
    out = {}
    for name, value in blob["model"].items():
        if not torch.is_floating_point(value):
            out[name] = value
        elif value.dim() >= 2 and value.numel() >= min_numel:
            flat = value.reshape(value.shape[0], -1).float()
            scale = flat.abs().amax(dim=1).clamp_min(1e-12) / 127.0
            q = torch.round(flat / scale[:, None]).clamp(-127, 127)
            out[name] = (q * scale[:, None]).reshape(value.shape).to(value.dtype)
        else:
            out[name] = value.half().to(value.dtype)
    return {**blob, "model": out}


def load(path):
    return json.load(gzip.open(path, "rt")) if str(path).endswith(".gz") else json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("replays", nargs="+")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    tmp = Path(tempfile.mkdtemp())
    q_path = tmp / "ckpt-int8-roundtrip.pt"
    torch.save(roundtrip(blob), q_path)
    fp = AG.make_agent(a.ckpt, seed=0, temperature=0.0, threads=1, log=False)
    q8 = AG.make_agent(str(q_path), seed=0, temperature=0.0, threads=1, log=False)
    steps = same = units = same_units = orders = same_orders = 0
    for rp in a.replays:
        rep = load(rp)
        cfg = rep["configuration"]
        for seat in (0, 1):
            for t in range(0, 719, max(1, 719 // a.steps)):
                obs = rep["steps"][t][seat]["observation"]
                x, y = fp(obs, cfg), q8(obs, cfg)
                steps += 1
                same += x == y
                ux = [x.get("farmer")] + list(x.get("hands") or [])
                uy = [y.get("farmer")] + list(y.get("hands") or [])
                units += len(ux)
                same_units += sum(1 for i, c in enumerate(ux) if i < len(uy) and uy[i] == c)
                ox, oy = list(x.get("market") or []), list(y.get("market") or [])
                orders += max(len(ox), len(oy))
                same_orders += sum(1 for i, o in enumerate(ox) if i < len(oy) and oy[i] == o)
    res = {"checkpoint": a.ckpt, "update": blob.get("update"), "steps": steps, "identical_steps": same,
           "identical_step_share": round(same / max(1, steps), 4), "unit_commands": units,
           "identical_unit_share": round(same_units / max(1, units), 4), "orders": orders,
           "identical_order_share": round(same_orders / max(1, orders), 4)}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
