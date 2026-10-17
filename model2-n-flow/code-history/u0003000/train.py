#!/usr/bin/env python3
"""N flow BC trainer (fresh random init; resumable).

  python3 train.py --roster ROSTER.tsv --out OUT [--shards DIR] [--max-seconds S] [--updates N] [--compile] ...

Objective: per-step primitive-action cross-entropy on WINNING seats only (act_loss rows) + auxiliary heads:
WDL (3-class sign of final cash margin; every row incl. WDL-only loser seats), own next-step unit tokens (act rows),
opponent current unit tokens (every row). Optimizer: torch.optim.Muon (match_rms_adamw) on 2-D weights of the
relational blocks and decoder layers, AdamW on everything else (B trainer's split and hyperparameters). bf16 autocast.
Checkpoints <out>/ckpt-XXXXXXX.pt every --ckpt-every updates and at the end, <out>/LATEST names the newest; a restart
resumes model, EMA, optimizer, schedule position, RNG and data seed from LATEST. Data streams restart on resume
(first seat of each stream is warm-masked, see nf_data).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nf_format as NF  # noqa: E402
import nf_model as M  # noqa: E402
import nf_data  # noqa: E402

U, O = M.U, M.O


# ----------------------------------------------------------------------------- loss
def masks(x, net):
    """Scored-label masks. Action (imitation) masks are False on every WDL-only row, warm chunk and padded step."""
    valid = x["valid"].bool() & ~x["warm"].bool()[:, None]
    act = valid & x["act_loss"].bool()[:, None]
    ut, uq, ot, oq = (x[k].long() for k in ("unit_tok", "unit_qty", "order_tok", "order_qty"))
    score_u = (ut >= 0) & x["unit_loss_keep"].bool() & act[..., None]
    has_uq = score_u & net.unit_qty_mask[ut.clamp(min=0)] & (uq >= 0)
    score_o = (ot >= 0) & act[..., None]
    has_oq = score_o & net.order_qty_mask[ot.clamp(min=0)] & (oq >= 0)
    return {"valid": valid, "act": act, "score_u": score_u, "has_uq": has_uq, "score_o": score_o, "has_oq": has_oq}


def _ce(logits, y, m):
    return F.cross_entropy(logits.float().flatten(0, -2), torch.where(m, y, -100).flatten(), ignore_index=-100,
                           reduction="none").view(m.shape)


def chunk_loss(out, x, net, w_wdl=0.2, w_next=0.1, w_opp=0.1):
    mk = masks(x, net)
    ut, uq, ot, oq = (x[k].long() for k in ("unit_tok", "unit_qty", "order_tok", "order_qty"))
    ce_ut = _ce(out["unit_logits"], ut, mk["score_u"])
    ce_uq = _ce(out["unit_qty_logits"], uq, mk["has_uq"])
    ce_ot = _ce(out["order_logits"], ot, mk["score_o"])
    ce_oq = _ce(out["order_qty_logits"], oq, mk["has_oq"])
    n_u = mk["score_u"].sum(-1).clamp(min=1)
    per_step = (ce_ut.sum(-1) + ce_uq.sum(-1)) / n_u + (ce_ot.sum(-1) + ce_oq.sum(-1)) / 2
    per_step = per_step * mk["act"]
    n_act = mk["act"].sum()
    imit = per_step.sum() / n_act.clamp(min=1)
    valid = mk["valid"]
    wdl_y = x["wdl"].long()[:, None].expand_as(valid)
    ce_w = _ce(out["wdl_logits"], wdl_y, valid)
    row_w = x["wdl_w"].float()[:, None] if "wdl_w" in x else torch.ones_like(ce_w)
    wdl = (ce_w * row_w).sum() / valid.sum().clamp(min=1)       # closeness weights have mean 1 over the roster
    nu = x["next_unit_tok"][:, :, 0].long()
    m_next = (nu >= 0) & mk["act"][..., None]
    ce_n = _ce(out["next_unit_logits"], nu, m_next)
    nxt = ce_n.sum() / m_next.sum().clamp(min=1)
    op = x["opp_unit_tok"][:, :, 0].long()
    m_opp = (op >= 0) & valid[..., None]
    ce_p = _ce(out["opp_unit_logits"], op, m_opp)
    opp = ce_p.sum() / m_opp.sum().clamp(min=1)
    total = imit + w_wdl * wdl + w_next * nxt + w_opp * opp
    with torch.no_grad():
        pu, pq = out["unit_logits"].argmax(-1), out["unit_qty_logits"].argmax(-1)
        po, poq = out["order_logits"].argmax(-1), out["order_qty_logits"].argmax(-1)
        pw = out["wdl_logits"].argmax(-1)
        unit_ok = (((pu == ut) | ~mk["score_u"]) & ((pq == uq) | ~mk["has_uq"])).all(-1)
        order_ok = (((po == ot) | ~mk["score_o"]) & ((poq == oq) | ~mk["has_oq"])).all(-1)
        stats = {
            "loss": total.detach(), "imit": imit.detach(), "wdl": wdl.detach(), "next": nxt.detach(), "opp": opp.detach(),
            "n_act_steps": n_act, "n_valid_steps": valid.sum(),
            "acc_unit_tok": (((pu == ut) & mk["score_u"]).sum(), mk["score_u"].sum()),
            "acc_unit_qty": (((pq == uq) & mk["has_uq"]).sum(), mk["has_uq"].sum()),
            "acc_order_tok": (((po == ot) & mk["score_o"]).sum(), mk["score_o"].sum()),
            "acc_order_qty": (((poq == oq) & mk["has_oq"]).sum(), mk["has_oq"].sum()),
            "acc_step_exact": ((unit_ok & order_ok & mk["act"]).sum(), mk["act"].sum()),
            "acc_wdl": (((pw == wdl_y) & valid).sum(), valid.sum()),
            "acc_wdl_wdlonly_rows": (((pw == wdl_y) & valid & ~x["act_loss"].bool()[:, None]).sum(),
                                     (valid & ~x["act_loss"].bool()[:, None]).sum()),
            "acc_opp_unit": (((out["opp_unit_logits"].argmax(-1) == op) & m_opp).sum(), m_opp.sum()),
        }
    return total, stats


# ----------------------------------------------------------------------------- optimizer (copied from B's train.py)
class MultiOptimizer:
    """Several optimizers stepped together (Muon for matrices + AdamW for the rest)."""

    def __init__(self, opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return {"multi": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        if "multi" not in sd or len(sd["multi"]) != len(self.opts):
            raise ValueError("incompatible multi-optimizer state")
        for o, s in zip(self.opts, sd["multi"]):
            o.load_state_dict(s)


def is_muon_matrix(n, p):
    return p.ndim == 2 and n.startswith(("blocks.", "dec.")) and n.endswith(".weight")


def make_optimizer(net, a):
    mats = [p for n, p in net.named_parameters() if is_muon_matrix(n, p)]
    ids = {id(p) for p in mats}
    decay, no_decay = [], []
    for n, p in net.named_parameters():
        if id(p) in ids:
            continue
        embedding_like = any(t in n for t in ("emb", "qpos", "pos_x", "pos_y", "qty_cond", "rel_bias", "reach_bias"))
        (decay if p.ndim >= 2 and not embedding_like else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": a.weight_decay, "lr_mult": 1.0},
              {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0}]
    for g in groups:
        g["lr_mult"] = getattr(a, "adamw_lr_mult", 1.0)
    adamw = torch.optim.AdamW([g for g in groups if g["params"]], lr=a.lr, betas=(0.9, a.beta2),
                              fused=torch.cuda.is_available())
    if a.optimizer == "adamw":
        for p in mats:
            groups[0]["params"].append(p)
        return torch.optim.AdamW([g for g in groups if g["params"]], lr=a.lr, betas=(0.9, a.beta2),
                                 fused=torch.cuda.is_available())
    muon = torch.optim.Muon([{"params": mats, "lr_mult": 1.0}], lr=a.lr, weight_decay=a.weight_decay,
                            momentum=a.muon_momentum, nesterov=True, ns_steps=a.muon_ns_steps,
                            adjust_lr_fn="match_rms_adamw")
    return MultiOptimizer([muon, adamw])


def lr_at(u, a):
    if u < a.warmup:
        return a.lr * (u + 1) / a.warmup
    if getattr(a, "decay_ts", None):
        now = time.time()
        if now < a.decay_ts:
            return a.lr
        p = min(1.0, (now - a.decay_ts) / max(1.0, a.stop_ts - 120 - a.decay_ts))
        return a.lr * (1 - (1 - a.min_lr_frac) * p)
    if a.schedule == "wsd":
        start = a.updates - a.decay_updates
        if a.decay_updates and u >= start:
            return a.lr * max(a.min_lr_frac, 1 - (1 - a.min_lr_frac) * (u - start) / a.decay_updates)
        return a.lr
    p = min(1.0, (u - a.warmup) / max(1, a.updates - a.warmup))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


# ----------------------------------------------------------------------------- model construction
def build_net(cols, codec_dir, a=None):
    sys.path.insert(0, str(codec_dir))
    import codec
    cfg = M.NFlowConfig()
    if a is not None:
        cfg = dataclasses.replace(cfg, width=a.width, blocks=a.blocks, heads=a.heads, rel_heads=a.rel_heads,
                                  fast=a.fast, slow=a.slow, worker=a.worker, dec_layers=a.dec_layers)
    cfg = M.resolve_config(cfg, codec, cols)
    return M.NFlowNet(cfg)


def to_dev(x, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in x.items()}


def mem_reset(mem, reset):
    r = reset.to(mem["fast"].device)
    out = {}
    for k, v in mem.items():
        if k == "day":
            out[k] = torch.where(r, torch.full_like(v, -1), v)
        else:
            shape = (-1,) + (1,) * (v.dim() - 1)
            out[k] = torch.where(r.view(shape), torch.zeros_like(v), v)
    return out


def save_ckpt(path, net, ema, opt, u, a, extra):
    blob = {"schema": M.SCHEMA, "config": net.config.to_dict(), "model": net.state_dict(),
            "ema": ema, "optimizer": opt.state_dict(), "update": u, "args": vars(a),
            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "numpy": np.random.get_state(), "python": random.getstate()},
            "extra": extra}
    tmp = str(path) + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", default="")
    ap.add_argument("--stp-dir", default="", help="fallback .stp dir for rows without a shard (compacted on the fly)")
    ap.add_argument("--bucket-probs", default="", help="e.g. A=.35,B=.20,C=.20,D=.15,E=.05: act rows are drawn by "
                    "their roster 'bucket' column (bucket by probability, seat uniform within it) instead of by weight")
    ap.add_argument("--columns", default=str(HERE / "COLUMNS.json"))
    ap.add_argument("--codec-dir", default=str(HERE / "pbc/work/runs/maintainer"),
                    help="dir holding codec.py (a byte copy of the B deploy codec; nested so its parents[3] exists)")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--stop-utc", default="", help="ISO UTC time; stop (with a checkpoint) 120 s before it")
    ap.add_argument("--decay-start-utc", default="", help="ISO UTC time: from here the LR decays linearly to "
                    "min_lr_frac at --stop-utc (wall-clock WSD for time-boxed runs; overrides --schedule after warmup)")
    ap.add_argument("--updates", type=int, default=20000, help="schedule length (total updates)")
    ap.add_argument("--stop-at", type=int, default=0, help="stop after this update (0 = --updates)")
    ap.add_argument("--streams", type=int, default=8, help="streams per loader worker")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--wdl-frac", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--schedule", default="cosine", choices=("cosine", "wsd"))
    ap.add_argument("--decay-updates", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=10.0, help="B trainer value (coordinator 02:2x); Muon is scale-invariant")
    ap.add_argument("--adamw-lr-mult", type=float, default=1.0, help="LR multiplier of the AdamW (non-Muon) groups")
    ap.add_argument("--optimizer", default="muon", choices=("muon", "adamw"))
    ap.add_argument("--muon-momentum", type=float, default=0.95)
    ap.add_argument("--muon-ns-steps", type=int, default=5)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--compile", default="", choices=("", "default", "max-autotune-no-cudagraphs"))
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--ckpt-minutes", type=float, default=0.0, help="also checkpoint when this much wall time passed")
    ap.add_argument("--keep-ckpts", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--profile-at", type=int, default=0, help="profile this update (CUDA kernel table to stdout)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--rel-heads", type=int, default=4)
    ap.add_argument("--fast", type=int, default=128)
    ap.add_argument("--slow", type=int, default=256)
    ap.add_argument("--worker", type=int, default=64)
    ap.add_argument("--dec-layers", type=int, default=2)
    ap.add_argument("--w-wdl", type=float, default=0.2)
    ap.add_argument("--w-next", type=float, default=0.1)
    ap.add_argument("--w-opp", type=float, default=0.1)
    a = ap.parse_args(argv)
    a.stop_ts = a.decay_ts = None
    if a.decay_start_utc and not a.stop_utc:
        raise SystemExit("--decay-start-utc needs --stop-utc")
    if a.stop_utc:
        import datetime as _dt
        _iso = lambda v: _dt.datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        a.stop_ts = _iso(a.stop_utc)
        a.decay_ts = _iso(a.decay_start_utc) if a.decay_start_utc else None
        left = (_dt.datetime.fromisoformat(a.stop_utc.replace("Z", "+00:00"))
                - _dt.datetime.now(_dt.timezone.utc)).total_seconds() - 120
        if left <= 0:
            raise SystemExit("stop time already passed")
        a.max_seconds = min(a.max_seconds, left) if a.max_seconds else left
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cols = NF.load_columns(a.columns)

    latest = out / "LATEST"
    blob = None
    if latest.exists():
        blob = torch.load(out / latest.read_text().strip(), map_location="cpu", weights_only=False)
        if blob["args"].get("seed") != a.seed:
            print("resume: keeping the checkpoint's seed", blob["args"].get("seed"), flush=True)
            a.seed = blob["args"]["seed"]
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    net = build_net(cols, a.codec_dir, a).to(dev)
    if blob is not None:
        if blob["config"] != net.config.to_dict():
            raise ValueError("checkpoint config differs from this run's model config")
        net.load_state_dict(blob["model"])
    opt = make_optimizer(net, a)
    if blob is not None:
        opt.load_state_dict(blob["optimizer"])
    ema = blob["ema"] if blob is not None else {k: v.detach().float().clone() for k, v in net.state_dict().items()}
    ema = {k: v.to(dev) for k, v in ema.items()}
    u0 = int(blob["update"]) if blob is not None else 0
    if blob is not None:
        torch.set_rng_state(blob["rng"]["torch"])
        if torch.cuda.is_available() and blob["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(blob["rng"]["cuda"])
        np.random.set_state(blob["rng"]["numpy"])
        random.setstate(blob["rng"]["python"])
    del blob
    n_params = net.parameter_count()
    run_info = {"schema": M.SCHEMA, "args": vars(a), "params": n_params, "config": net.config.to_dict(),
                "roster_sha256": __import__("hashlib").sha256(Path(a.roster).read_bytes()).hexdigest(),
                "columns_sha256": NF.columns_hash(cols), "start_update": u0, "device": str(dev),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "fresh_init": u0 == 0, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "code_sha256": {f: __import__("hashlib").sha256((HERE / f).read_bytes()).hexdigest()
                                for f in ("train.py", "nf_model.py", "nf_data.py", "nf_format.py", "COLUMNS.json")
                                if (HERE / f).exists()}}
    (out / f"RUN-{u0:07d}.json").write_text(json.dumps(run_info, indent=1) + "\n")
    print(json.dumps({k: run_info[k] for k in ("params", "start_update", "gpu", "roster_sha256")}), flush=True)

    pre_fn, post_fn = net.encode_pre, net.post_teacher
    profile_window = None
    if a.compile:
        cpre = torch.compile(net.encode_pre, mode=a.compile, dynamic=False)
        cpost = torch.compile(net.post_teacher, mode=a.compile, dynamic=False)
        state = {"ok": None}

        def _guard(cf, ef):
            def f(*args):
                if state["ok"] is False:
                    return ef(*args)
                try:
                    return cf(*args)
                except Exception as exc:  # noqa: BLE001 - e.g. no C toolchain for Triton launchers on this node
                    if state["ok"]:
                        raise
                    state["ok"] = False
                    print(f"COMPILE-FAILED, continuing eager: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
                    return ef(*args)
            return f
        pre_fn, post_fn = _guard(cpre, net.encode_pre), _guard(cpost, net.post_teacher)
        profile_window = state

    data_seed = a.seed * 1000 + u0
    dl = nf_data.loader(a.roster, a.shards or None, cols, workers=a.workers, streams=a.streams, chunk=a.chunk,
                        seed=data_seed, wdl_frac=a.wdl_frac, stp_dir=a.stp_dir,
                        bucket_probs={k: float(v) for k, v in (x.split("=") for x in a.bucket_probs.split(",") if x)})
    it = iter(dl)
    nw = max(1, a.workers)
    B = nw * a.streams
    mem = net.initial_memory(B, dev)
    t_start = time.time()
    stop_at = a.stop_at or a.updates
    u = u0
    acc = {}
    clip_hits = torch.zeros((), device=dev)
    gns_g2 = torch.zeros((), device=dev)
    gns_s = torch.zeros((), device=dev)
    t_log, n_log, wait_s = time.time(), 0, 0.0
    last_ckpt = time.time()
    skipped = 0
    log = open(out / "train.jsonl", "a")
    ema_keys = [k for k, v in net.state_dict().items() if v.is_floating_point()]

    def checkpoint(uu):
        name = f"ckpt-{uu:07d}.pt"
        bad = [k for k, v in list(net.state_dict().items()) + list(ema.items())
               if v.is_floating_point() and not bool(torch.isfinite(v).all())]
        if bad:      # never overwrite the last good checkpoint with nonfinite weights
            raise SystemExit(f"nonfinite weights at update {uu}: {bad[:5]}; not checkpointed")
        save_ckpt(out / name, net, ema, opt, uu, a, {"streams": B, "chunk": a.chunk, "params": n_params})
        (out / "LATEST.tmp").write_text(name + "\n")
        os.replace(out / "LATEST.tmp", out / "LATEST")
        ck = sorted(out.glob("ckpt-*.pt"))
        for old in ck[:-a.keep_ckpts]:
            if old.name != name:
                old.unlink()
        print(f"checkpoint {name}", flush=True)

    while u < stop_at:
        if a.max_seconds and time.time() - t_start > a.max_seconds:
            break
        prof = None
        if ((a.profile_at and u == a.profile_at) or (profile_window is not None and profile_window.get("ok") is False
                                                      and u == u0 + 25)):
            acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if dev.type == "cuda" else [])
            prof = torch.profiler.profile(activities=acts)
            prof.__enter__()
        lr = lr_at(u, a)
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
        opt.zero_grad(set_to_none=True)
        acc_steps_this = torch.zeros((), device=dev)
        for mi in range(a.accum):
            tw = time.time()
            x = nf_data.gather(it, nw)
            wait_s += time.time() - tw
            meta = {k: x.pop(k) for k in ("eid", "seat")}
            x = to_dev(x, dev)
            mem = mem_reset(mem, x["reset"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                outp, mem = net.forward_chunk(x, mem, pre_fn, post_fn)
                loss, st = chunk_loss(outp, x, net, a.w_wdl, a.w_next, a.w_opp)
            acc_steps_this = acc_steps_this + st["n_valid_steps"]      # scored (valid, non-warm) steps
            (loss / a.accum).backward()
            if a.accum >= 2 and mi == 0:          # |g_small|^2 from the first micro-batch (gradient noise scale)
                gs = [p.grad for p in net.parameters() if p.grad is not None]
                n_small = torch.stack(torch._foreach_norm(gs)).pow(2).sum() * a.accum ** 2
            mem = {k: v.detach() for k, v in mem.items()}
            for k, v in st.items():
                if isinstance(v, tuple):
                    p = acc.get(k, (0, 0))
                    acc[k] = (p[0] + v[0], p[1] + v[1])
                else:
                    acc[k] = acc.get(k, 0) + v
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), a.grad_clip)
        # review CHECK-NFLOW P0: never step on a nonfinite gradient or a batch with no scored step (one sync/update)
        if not bool(torch.isfinite(gn) & (acc_steps_this > 0)):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            mem = {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in mem.items()}
            print(json.dumps({"skip_update": u + 1, "grad_norm_finite": bool(torch.isfinite(gn)),
                              "scored_steps": int(acc_steps_this)}), flush=True)
            u += 1
            continue
        clip_hits += (gn > a.grad_clip).float()
        if a.accum >= 2:   # McCandlish et al. two-batch estimator, B_small = 1 micro-batch, B_big = accum
            n_big = gn.float() ** 2
            gns_g2 += (a.accum * n_big - n_small) / (a.accum - 1)
            gns_s += (n_small - n_big) / (1 - 1 / a.accum)
        opt.step()
        with torch.no_grad():
            d = a.ema if u >= a.warmup else min(a.ema, (1 + u) / (10 + u))
            sd = net.state_dict()
            torch._foreach_lerp_([ema[k] for k in ema_keys], [sd[k].detach().float() for k in ema_keys], 1 - d)
        if prof is not None:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            key = "cuda_time_total" if dev.type == "cuda" else "cpu_time_total"
            print(prof.key_averages().table(sort_by=key, row_limit=30), flush=True)
        u += 1
        n_log += 1
        if u % a.log_every == 0 or u == stop_at:
            dt = time.time() - t_log
            rec = {"update": u, "lr": lr, "skipped_updates": skipped, "s_per_update": dt / n_log, "data_wait_s": wait_s / n_log,
                   "steps_per_s": n_log * a.accum * B * a.chunk / dt, "clip_rate": float(clip_hits) / n_log,
                   "grad_norm": float(gn), "elapsed_s": time.time() - t_start,
                   "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0}
            for k, v in acc.items():
                if isinstance(v, tuple):
                    rec[k] = float(v[0]) / max(1.0, float(v[1]))
                elif k.startswith("n_"):
                    rec[k] = float(v)
                else:
                    rec[k] = float(v) / (n_log * a.accum)
            if a.accum >= 2:
                g2, sv = float(gns_g2) / n_log, float(gns_s) / n_log
                rec["gns_B_simple_steps"] = (sv / g2 * B * a.chunk) if g2 > 0 else None   # in graph-steps
                rec["grad_snr"] = (g2 / (sv / a.accum)) if sv > 0 else None
                gns_g2.zero_(); gns_s.zero_()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            acc, n_log, wait_s, t_log = {}, 0, 0.0, time.time()
            clip_hits.zero_()
        if u % a.log_every == 0 and (out / "STOP").exists():      # graceful stop request: checkpoint and exit
            print(json.dumps({"stop_file": True, "update": u}), flush=True)
            (out / "STOP").rename(out / f"STOP.done-{u:07d}")
            break
        if u % a.ckpt_every == 0 or (a.ckpt_minutes and time.time() - last_ckpt >= 60 * a.ckpt_minutes):
            checkpoint(u)
            last_ckpt = time.time()
    if not (out / f"ckpt-{u:07d}.pt").exists():
        checkpoint(u)
    print(json.dumps({"done": u, "elapsed_s": time.time() - t_start}), flush=True)
    log.close()
    shutdown = getattr(it, "_shutdown_workers", None)
    if callable(shutdown):      # stop loader workers before exiting (os._exit skips multiprocessing's cleanup)
        shutdown()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    os._exit(0)


if __name__ == "__main__":
    main()
