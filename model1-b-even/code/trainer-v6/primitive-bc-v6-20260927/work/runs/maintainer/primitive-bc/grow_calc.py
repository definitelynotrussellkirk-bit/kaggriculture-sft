"""Grow a v3 lineage-B checkpoint (any layer count) with the v4 calc_features inputs, inert at growth (owner 2026-09-26 goal:
"try to improve representations"). ModelConfig.calc_width = data.CALC_WIDTH; the new grow_calc Linear reads the calc block that
data.features appends after the kind/deadline pair (product rows on MARKET tokens, the global row on the GLOBAL token) and starts
at zero, so every output equals the parent's. Existing tensors and Adam state are carried by name; grow_calc joins the grow_* group.

  python3 grow_calc.py --parent CKPT.pt --out DIR [--verify REPLAY.json.gz]
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

import data as D  # noqa: E402
import model as M  # noqa: E402
import train as T  # noqa: E402
from grow_layers import verify  # noqa: E402


def grow(parent_path):
    blob = torch.load(str(parent_path), map_location="cpu", weights_only=True)
    pc = M.ModelConfig.from_dict(blob["config"])
    if not pc.extra_width:
        raise SystemExit("parent has no v3 extra inputs")
    if getattr(pc, "calc_width", 0):
        raise SystemExit("parent already reads calc columns")
    parent = M.PolicyNet(pc)
    parent.load_state_dict(blob["model"])
    gc = dataclasses.replace(pc, calc_width=D.CALC_WIDTH)
    cfg = T.TrainConfig.from_dict(blob["train_config"])
    grown = T.build(gc, "cpu")
    res = grown.load_state_dict(blob["model"], strict=False)
    if set(res.missing_keys) != {"grow_calc.weight", "grow_calc.bias"} or res.unexpected_keys:
        raise SystemExit(f"unexpected load: missing {res.missing_keys[:6]} unexpected {res.unexpected_keys[:6]}")
    assert float(grown.grow_calc.weight.abs().sum()) == 0.0 and float(grown.grow_calc.bias.abs().sum()) == 0.0
    popt = T.make_optimizer(parent, cfg)
    popt.load_state_dict(blob["optimizer"])
    pstate = {n: popt.state[p] for n, p in parent.named_parameters() if p in popt.state}
    gopt = T.make_optimizer(grown, cfg)
    gn = {id(p): n for n, p in grown.named_parameters()}
    for g in gopt.param_groups:
        for p in g["params"]:
            st = pstate.get(gn[id(p)])
            if st:
                gopt.state[p] = {k: v.clone() for k, v in st.items()}
    pp = dict(parent.named_parameters())
    for n, p in grown.named_parameters():
        if n in pp:
            assert torch.equal(p, pp[n]), n
    out = {**blob, "config": grown.config.to_dict(), "model": grown.state_dict(), "optimizer": gopt.state_dict(),
           "train_config": dataclasses.asdict(cfg)}
    info = {"update": blob["update"], "calc_width": D.CALC_WIDTH,
            "parent_parameters": sum(p.numel() for p in parent.parameters()),
            "grown_parameters": sum(p.numel() for p in grown.parameters()),
            "groups": [len(g["params"]) for g in gopt.param_groups], "lr_mult": [g["lr_mult"] for g in gopt.param_groups],
            "adam_state_carried": sum(1 for p in gopt.state if gopt.state[p])}
    return out, info, parent, grown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", default=None)
    a = ap.parse_args()
    out, info, parent, grown = grow(a.parent)
    if a.verify:
        v = verify(parent, grown, a.verify)
        info["verify"] = v
        if abs(v["loss_parent"] - v["loss_grown"]) > 1e-5 or v["act_same"] != v["act_n"]:
            raise SystemExit(f"growth is not the identity: {v}")
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=False)
    path = d / f"ckpt-{out['update']:07d}.pt"
    T._write_atomic(out, path)
    (d / "LATEST").write_text(path.name + "\n")
    (d / "GROWN-FROM.json").write_text(json.dumps({"parent": str(a.parent), **info}, indent=1) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
