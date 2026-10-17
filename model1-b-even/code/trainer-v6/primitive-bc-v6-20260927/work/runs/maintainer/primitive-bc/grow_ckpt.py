"""Grow a lineage-B training checkpoint with the zero-initialised flip/reachability inputs (flip.py).

Every old tensor, the optimizer moments, step counters and RNG states are kept; the grow_* parameters start at
zero in a third optimizer group (no weight decay, lr x grow_lr_mult). --verify runs the old and the grown network on
real replay boundaries (grown with the flip inputs present) and requires identical greedy decisions and near-equal
teacher-forced logits.

  ../../../venv/python -B grow_ckpt.py SRC.pt OUT_DIR [--grow-lr-mult 10] [--verify-steps 96]
"""
import argparse
import dataclasses
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import flip  # noqa: E402
import model as M  # noqa: E402
import prefix  # noqa: E402
import train as T  # noqa: E402


def grow(blob, grow_lr_mult):
    """Ungrown or v1-grown checkpoint -> v2 growth (flip.LIN extra columns, reach_job bias). Old tensors are kept; wider
    grow_extra columns and all new grow_* parameters start at zero; Adam moments are kept and zero-padded."""
    old_cfg = M.ModelConfig.from_dict(blob["config"])
    if old_cfg.think != "off":
        raise ValueError("reachability bias is defined for think='off' only")
    cfg = dataclasses.replace(old_cfg, extra_width=flip.LIN, unit_extra_width=flip.UNIT_WIDTH, reach_bias=True,
                              n_kinds=flip.N_KINDS, carrier_read=True, reach_job=True, prefix_unit_width=prefix.F_UP,
                              prefix_order_width=prefix.F_OP)
    net = M.PolicyNet(cfg)
    target = net.state_dict()
    state, padded = {}, {}
    for k, v in blob["model"].items():
        if k in target and tuple(target[k].shape) != tuple(v.shape):
            if v.ndim != 2 or v.shape[0] != target[k].shape[0] or v.shape[1] > target[k].shape[1]:
                raise ValueError(f"cannot widen {k}: {tuple(v.shape)} -> {tuple(target[k].shape)}")
            w = torch.zeros_like(target[k])
            w[:, :v.shape[1]] = v
            state[k], padded[k] = w, v.shape[1]
        else:
            state[k] = v
    missing, unexpected = net.load_state_dict(state, strict=False)
    new_names = set(net.growth_parameter_names()) - set(blob["model"])
    if unexpected or set(missing) != new_names:
        raise ValueError(f"state mismatch: missing {missing} unexpected {unexpected}")
    with torch.no_grad():
        for n, p in net.named_parameters():
            if n in new_names:
                p.zero_()
    tcfg = T.TrainConfig.from_dict({**blob["train_config"], "grow_lr_mult": grow_lr_mult})
    opt = T.make_optimizer(net, tcfg)
    old, new = blob["optimizer"], opt.state_dict()
    n_old = sum(len(g["params"]) for g in old["param_groups"])
    base = [len(g["params"]) for g in old["param_groups"][:2]]
    if base != [len(g["params"]) for g in new["param_groups"][:2]]:
        raise ValueError("old optimizer base groups do not match the pre-growth layout")
    names = [n for n, _ in net.named_parameters()]
    grow_names = [n for n in names if n.startswith("grow_")]
    state_new = {}
    for i, st in old["state"].items():
        i = int(i)
        if i < sum(base):
            state_new[i] = st
            continue
        name = grow_names[i - sum(base)]                   # v1 grow params keep their registration order
        st = dict(st)
        if name in padded:
            for key in ("exp_avg", "exp_avg_sq"):
                t = torch.zeros(target[name].shape, dtype=st[key].dtype)
                t[:, :padded[name]] = st[key]
                st[key] = t
        state_new[i] = st
    new["state"] = state_new
    for g_new, g_old in zip(new["param_groups"][:2], old["param_groups"]):
        g_new.update({k: v for k, v in g_old.items() if k != "params"})
        g_new["lr_mult"] = g_old.get("lr_mult", 1.0)
    tail = new["param_groups"][2]
    src = old["param_groups"][2] if len(old["param_groups"]) > 2 else old["param_groups"][1]
    tail.update({k: v for k, v in src.items() if k != "params"})
    tail.update(weight_decay=0.0, lr_mult=grow_lr_mult)
    opt.load_state_dict(new)
    out = dict(blob)
    out.update(config=net.config.to_dict(), train_config=dataclasses.asdict(tcfg), model=net.state_dict(),
               optimizer=opt.state_dict())
    old_net = M.PolicyNet(old_cfg)
    old_net.load_state_dict(blob["model"])
    return out, net, old_net


def real_batch(n_steps, seed=0):
    arch = sorted(Path("/mnt/ai/scratch/farmNEAT/primitive-bc-data").glob("kaggriculture-episodes-2026-09-2*.zip"))[-1]
    z = zipfile.ZipFile(arch)
    members = [m for m in z.namelist() if m.endswith(".json")][:3]
    examples = []
    for m in members:
        replay = json.loads(z.read(m))
        for seat in (0, 1):
            steps = range(seed + seat, 719, max(1, 719 * 6 // n_steps))
            examples += list(data.episode_examples(replay, seat, meta_base={}, steps=steps))
    return data.collate(examples[:n_steps])


def old_inputs(old_net, b):
    """The growth inputs an older network reads: v1 used flip columns [0, extra_width) plus kind/deadline right after."""
    c = old_net.config
    out = {}
    if c.extra_width:
        out["extra"] = torch.cat((b["extra"][..., :c.extra_width], b["extra"][..., flip.LIN:flip.LIN + 2]), -1)
        out["unit_extra"] = b["unit_extra"]
    if c.reach_job:
        out["reach_eta"] = b["reach_eta"]
    return out


@torch.no_grad()
def verify(old_net, net, blob, n_steps):
    old_net.eval(), net.eval()
    b = real_batch(n_steps)
    up = torch.stack((b["unit_tok"], b["unit_qty"]), -1)
    op = torch.stack((b["order_tok"], b["order_qty"]), -1)
    base = old_net(b["rows"], b["token_mask"], b["knownness"], b["unit_pos"], up, op, **old_inputs(old_net, b))
    grown = net(b["rows"], b["token_mask"], b["knownness"], b["unit_pos"], up, op,
                extra=b["extra"], unit_extra=b["unit_extra"], reach_eta=b["reach_eta"],
                unit_prefix_state=b["unit_prefix"], order_prefix_state=b["order_prefix"])
    diff = max(float((base[k] - grown[k]).abs().max()) for k in ("unit_logits", "unit_qty_logits", "order_logits",
                                                                  "order_qty_logits", "value")
               if torch.isfinite(base[k]).all())
    ctx_old = old_net.encode(b["rows"], b["token_mask"], b["knownness"], **old_inputs(old_net, b))
    ctx_new = net.encode(b["rows"], b["token_mask"], b["knownness"], extra=b["extra"], unit_extra=b["unit_extra"],
                         reach_eta=b["reach_eta"], unit_prefix=b["unit_prefix"], order_prefix=b["order_prefix"])
    d_old = old_net.decode_free(ctx_old, b["unit_pos"], temperature=0.0, full_orders=True)
    d_new = net.decode_free(ctx_new, b["unit_pos"], temperature=0.0, full_orders=True)
    same = all(torch.equal(d_old[k], d_new[k]) for k in d_old)
    return {"steps": int(b["rows"].shape[0]), "max_abs_logit_diff": diff, "greedy_identical": same,
            "grow_parameters": sum(p.numel() for n, p in net.named_parameters() if n.startswith("grow_")),
            "total_parameters": net.parameter_count()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--grow-lr-mult", type=float, default=10.0)
    ap.add_argument("--verify-steps", type=int, default=96)
    a = ap.parse_args(argv)
    src = Path(a.src)
    blob = torch.load(str(src), map_location="cpu", weights_only=True)
    out, net, old_net = grow(blob, a.grow_lr_mult)
    report = verify(old_net, net, blob, a.verify_steps)
    report.update(source=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                  grow_lr_mult=a.grow_lr_mult, update=blob["update"])
    print(json.dumps(report))
    if not report["greedy_identical"] or report["max_abs_logit_diff"] > 1e-3:
        raise SystemExit("growth is not output-preserving; nothing written")
    out["grown_from"] = report
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"ckpt-{blob['update']:07d}.pt"
    T._write_atomic(out, path)
    (d / "LATEST").write_text(path.name + "\n")
    (d / "GROWN-FROM.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
