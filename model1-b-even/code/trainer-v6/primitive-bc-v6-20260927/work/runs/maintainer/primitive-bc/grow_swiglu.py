"""Grow a checkpoint with a parallel SwiGLU branch in every encoder block and decoder layer, inert at growth (owner 2026-09-27:
"prefer warmstarting of course if you can do surgery to get our new architectures working"). Each block gets sg1 (w -> 2h,
random) and sg2 (h -> w, zero), so every output equals the parent's until training moves sg2. Existing tensors and their Adam
state are carried by name; the new tensors join the grow_* group (cfg.grow_prefixes: no weight decay, LR x grow_lr_mult).

  python3 grow_swiglu.py --parent CKPT.pt --out DIR [--hidden 384] --verify ARCHIVE.zip:MEMBER|REPLAY.json.gz
"""
import argparse
import dataclasses
import gzip
import json
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

import model as M  # noqa: E402
import train as T  # noqa: E402
from grow_layers import verify  # noqa: E402


def grow(parent_path, hidden):
    blob = torch.load(str(parent_path), map_location="cpu", weights_only=True)
    pc = M.ModelConfig.from_dict(blob["config"])
    if getattr(pc, "swiglu_branch", 0):
        raise SystemExit("parent already has a SwiGLU branch")
    parent = M.PolicyNet(pc)
    parent.load_state_dict(blob["model"])
    gc = dataclasses.replace(pc, swiglu_branch=int(hidden))
    prefixes = [f"blocks.{i}.sg" for i in range(pc.layers)] + [f"dec.{i}.sg" for i in range(pc.dec_layers)]
    pcfg = T.TrainConfig.from_dict(blob["train_config"])
    old = str(pcfg.grow_prefixes or "")
    cfg = dataclasses.replace(pcfg, grow_prefixes=",".join([x for x in old.split(",") if x] + prefixes))
    grown = T.build(gc, "cpu")
    res = grown.load_state_dict(blob["model"], strict=False)
    new = {n for n in grown.state_dict() if n.startswith(tuple(prefixes))}
    if set(res.missing_keys) != new or res.unexpected_keys:
        raise SystemExit(f"unexpected load: missing {sorted(set(res.missing_keys) - new)[:5]} unexpected {res.unexpected_keys[:5]}")
    with torch.no_grad():
        for blk in list(grown.blocks) + list(grown.dec):
            blk.sg2.weight.zero_()
            blk.sg2.bias.zero_()
    popt = T.make_optimizer(parent, pcfg)
    popt.load_state_dict(blob["optimizer"])
    pstate = {n: popt.state[p] for n, p in parent.named_parameters() if p in popt.state}
    gopt = T.make_optimizer(grown, cfg)
    gn = {id(p): n for n, p in grown.named_parameters()}
    for g in gopt.param_groups:
        for p in g["params"]:
            st = pstate.get(gn[id(p)])
            if st:
                gopt.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    pp = dict(parent.named_parameters())
    for n, p in grown.named_parameters():
        if n in pp:
            assert torch.equal(p, pp[n]), n
    out = {**blob, "config": grown.config.to_dict(), "model": grown.state_dict(), "optimizer": gopt.state_dict(),
           "train_config": dataclasses.asdict(cfg)}
    info = {"update": blob["update"], "swiglu_hidden": int(hidden),
            "parent_parameters": sum(p.numel() for p in parent.parameters()),
            "grown_parameters": sum(p.numel() for p in grown.parameters()), "new_tensors": len(new),
            "groups": [len(g["params"]) for g in gopt.param_groups], "lr_mult": [g["lr_mult"] for g in gopt.param_groups],
            "adam_state_carried": sum(1 for p in gopt.state if gopt.state[p])}
    return out, info, parent, grown


def replay_file(spec, tmp):
    if ":" in spec and spec.split(":", 1)[0].endswith(".zip"):
        z, member = spec.split(":", 1)
        path = Path(tmp) / "replay.json.gz"
        with gzip.open(path, "wb") as f:
            f.write(zipfile.ZipFile(z).read(member))
        return str(path)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--verify", required=True, help="official replay: ARCHIVE.zip:MEMBER or REPLAY.json.gz")
    a = ap.parse_args()
    out, info, parent, grown = grow(a.parent, a.hidden)
    with tempfile.TemporaryDirectory() as tmp:
        v = verify(parent, grown, replay_file(a.verify, tmp))
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
