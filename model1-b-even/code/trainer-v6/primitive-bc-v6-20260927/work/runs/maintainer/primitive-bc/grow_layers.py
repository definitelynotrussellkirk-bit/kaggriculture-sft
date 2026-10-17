"""Grow a v3 lineage-B checkpoint by k encoder blocks and m decoder layers, inert at growth (owner 2026-09-26: "find ways to
increase features, give more layers"; goal "try to improve representations"). New blocks are appended with zero residual output
projections (encoder proj/ff2, decoder o1/o2/ff2), so every output equals the parent's. Existing tensors and their Adam state are
carried by name; the new tensors join the grow_* group (cfg.grow_prefixes: no weight decay, LR x grow_lr_mult).

  python3 grow_layers.py --parent CKPT.pt --out DIR --enc 2 --dec 2 [--verify REPLAY.json.gz]
"""
import argparse
import dataclasses
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

import model as M  # noqa: E402
import train as T  # noqa: E402

ZERO_ENC, ZERO_DEC = ("proj", "ff2"), ("o1", "o2", "ff2")


def grow(parent_path, enc=2, dec=2):
    blob = torch.load(str(parent_path), map_location="cpu", weights_only=True)
    pc = M.ModelConfig.from_dict(blob["config"])
    parent = M.PolicyNet(pc)
    parent.load_state_dict(blob["model"])
    gc = dataclasses.replace(pc, layers=pc.layers + enc, dec_layers=pc.dec_layers + dec)
    prefixes = [f"blocks.{i}." for i in range(pc.layers, gc.layers)] + [f"dec.{i}." for i in range(pc.dec_layers, gc.dec_layers)]
    pcfg = T.TrainConfig.from_dict(blob["train_config"])
    old = str(pcfg.grow_prefixes or "")
    cfg = dataclasses.replace(pcfg, grow_prefixes=",".join([x for x in old.split(",") if x] + prefixes))
    grown = T.build(gc, "cpu")
    state, resized = dict(blob["model"]), {}
    for n, t in grown.state_dict().items():      # per-decoder-layer tables (v3 reach bias): old rows kept, new layers' rows zero
        if n in state and state[n].shape != t.shape:
            o = state[n]
            if not (o.dim() == t.dim() and o.shape[1:] == t.shape[1:] and (o.shape[0], t.shape[0]) in
                    ((pc.dec_layers, gc.dec_layers), (pc.layers, gc.layers))):
                raise SystemExit(f"cannot grow {n}: {tuple(o.shape)} -> {tuple(t.shape)}")
            z = torch.zeros_like(t)
            z[:o.shape[0]] = o
            state[n], resized[n] = z, o.shape[0]
    res = grown.load_state_dict(state, strict=False)
    new = {n for n in grown.state_dict() if n.startswith(tuple(prefixes))}
    if set(res.missing_keys) != new or res.unexpected_keys:
        raise SystemExit(f"unexpected load: missing {sorted(set(res.missing_keys) - new)[:5]} unexpected {res.unexpected_keys[:5]}")
    with torch.no_grad():
        for i in range(pc.layers, gc.layers):
            for leaf in ZERO_ENC:
                getattr(grown.blocks[i], leaf).weight.zero_()
                getattr(grown.blocks[i], leaf).bias.zero_()
        for i in range(pc.dec_layers, gc.dec_layers):
            for leaf in ZERO_DEC:
                getattr(grown.dec[i], leaf).weight.zero_()
                getattr(grown.dec[i], leaf).bias.zero_()
    popt = T.make_optimizer(parent, pcfg)
    popt.load_state_dict(blob["optimizer"])
    pstate = {n: popt.state[p] for n, p in parent.named_parameters() if p in popt.state}
    gopt = T.make_optimizer(grown, cfg)
    gn = {id(p): n for n, p in grown.named_parameters()}
    for g in gopt.param_groups:
        for p in g["params"]:
            st = pstate.get(gn[id(p)])
            if st:
                carried = {}
                for k, v in st.items():
                    if gn[id(p)] in resized and torch.is_tensor(v) and v.dim() == p.dim() and v.shape != p.shape:
                        z = torch.zeros_like(p, dtype=v.dtype)
                        z[:v.shape[0]] = v
                        v = z
                    carried[k] = v.clone()
                gopt.state[p] = carried
    pp = dict(parent.named_parameters())
    for n, p in grown.named_parameters():
        if n in pp:
            ref = pp[n]
            assert torch.equal(p[:ref.shape[0]] if n in resized else p, ref), n
    out = {**blob, "config": grown.config.to_dict(), "model": grown.state_dict(), "optimizer": gopt.state_dict(),
           "train_config": dataclasses.asdict(cfg)}
    info = {"update": blob["update"], "layers": [pc.layers, gc.layers], "dec_layers": [pc.dec_layers, gc.dec_layers],
            "parent_parameters": sum(p.numel() for p in parent.parameters()),
            "grown_parameters": sum(p.numel() for p in grown.parameters()), "new_tensors": len(new),
            "groups": [len(g["params"]) for g in gopt.param_groups], "lr_mult": [g["lr_mult"] for g in gopt.param_groups],
            "adam_state_carried": sum(1 for p in gopt.state if gopt.state[p]), "resized": sorted(resized)}
    return out, info, parent, grown


def verify(parent, grown, replay_path, steps=48):
    """Teacher-forced loss on real windows and greedy act() on real observations must match the parent's."""
    import data as D
    replay = json.load(gzip.open(replay_path))
    exs = list(D.episode_examples(replay, 0, meta_base={"episode_id": 0, "seat": 0}, steps=range(steps)))
    batch = D.collate_windows(list(D.make_windows(exs, window=24, burn_in=0)))
    cfg = T.TrainConfig(bf16=False)
    losses = []
    for net in (parent, grown):
        net.eval()
        with torch.no_grad():
            loss, _, _, _ = T.run_window_batch(net, batch, device=torch.device("cpu"), tau=1.0, code_mode="sample",
                                               generator=torch.Generator().manual_seed(0), bf16=False,
                                               qty_weight=cfg.qty_weight, free_run=False)
        losses.append(float(loss))
    from farm_neat.training.bc import public_observation
    same = 0
    for t in range(0, steps, 2):
        obs = public_observation(replay, t, 0)
        a1, _ = parent.act(obs, temperature=0.0, configuration=replay["configuration"])
        a2, _ = grown.act(obs, temperature=0.0, configuration=replay["configuration"])
        same += json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)
    return {"loss_parent": losses[0], "loss_grown": losses[1], "act_same": same, "act_n": len(range(0, steps, 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--enc", type=int, default=2)
    ap.add_argument("--dec", type=int, default=2)
    ap.add_argument("--verify", default=None, help="official replay (json.gz) for the identity check")
    a = ap.parse_args()
    out, info, parent, grown = grow(a.parent, a.enc, a.dec)
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
