"""Graft the v6 inputs and training-only heads (pbc-v6-design/DESIGN.md) onto a trained checkpoint, inert at graft: the
previous-two-turns history enters through zero-initialised gates (grow_v6_*), the aux heads only add training loss. Existing
tensors and their Adam state are carried by name; every new tensor (v6_*, grow_v6_*, aux_*) joins the fresh-graft group
(no weight decay, lr multiplier ramped over --new-warmup updates). The child's teacher-forced loss on a recorded game must
equal the parent's exactly, else nothing is written.

  python grow_v6.py --parent CKPT.pt --out DIR [--no-v6] [--cash] [--no-aux] [--new-lr-mult 10] [--new-warmup 200]
                    [--aux-next-w 0.01] [--aux-opp-w 0.006]
--cash also grafts Codex's market-prefix cash ladder (Task 71: grow_prefix_cash, zero-init); --no-v6 grafts only the cash block.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

import model as M  # noqa: E402
import train as T  # noqa: E402

V6_NEW = ("v6_", "grow_v6_hist", "grow_v6_prev_unit", "aux_")
CASH_NEW = ("grow_prefix_cash",)
JOB_NEW = ("grow_v6_token_", "grow_v6_job_")   # Codex Task 72 (pbc-job-input-v6.0): *_in random, *_out zero at build
RECUR_NEW = ("grow_recur_",)                    # shared encoder core loop gate (owner 2026-09-28, Parcae-style)


def grow(blob, *, aux, new_lr_mult, new_warmup, aux_next_w, aux_opp_w, v6=True, cash=False, job_eta=False,
         recur="", recur_loops=1, recur_train_max=0):
    import prefix
    pc = M.ModelConfig.from_dict(blob["config"])
    if v6 and getattr(pc, "v6_prev", False):
        raise SystemExit("parent already has v6 inputs")
    if cash and getattr(pc, "prefix_cash_width", 0):
        raise SystemExit("parent already has the cash ladder")
    if job_eta and getattr(pc, "job_eta_v6", False):
        raise SystemExit("parent already has the job-ETA block")
    if recur and getattr(pc, "recur_core", ""):
        raise SystemExit("parent already has a recurrent core")
    if not (v6 or cash or job_eta or recur):
        raise SystemExit("nothing to graft")
    parent = M.PolicyNet(pc)
    parent.load_state_dict(blob["model"])
    NEW = (V6_NEW if v6 else ()) + (CASH_NEW if cash else ()) + (JOB_NEW if job_eta else ()) + (RECUR_NEW if recur else ())
    gc = dataclasses.replace(pc, v6_prev=bool(v6 or pc.v6_prev), v6_aux=bool(aux) if v6 else pc.v6_aux,
                             prefix_cash_width=prefix.F_CASH if cash else getattr(pc, "prefix_cash_width", 0),
                             **({"job_eta_v6": True, "input_schema": "pbc-job-input-v6.0"} if job_eta else {}),
                             **({"recur_core": recur, "recur_loops": int(recur_loops), "recur_train_max": int(recur_train_max)}
                                if recur else {}))
    pcfg = T.TrainConfig.from_dict(blob["train_config"])
    # This config has one fresh-group schedule. A second graft cannot preserve a
    # previous fresh group's independent warmup/clipping by overwriting its prefixes.
    if pcfg.new_prefixes:
        raise SystemExit("parent has an existing fresh-graft group; compose features from the common base "
                         "or explicitly migrate optimizer groups before a second graft")
    cfg = dataclasses.replace(pcfg, v6_data=bool(v6 or pcfg.v6_data),
                              aux_next_w=(aux_next_w if aux else 0.0) if v6 else pcfg.aux_next_w,
                              aux_opp_w=(aux_opp_w if aux else 0.0) if v6 else pcfg.aux_opp_w,
                              new_prefixes=",".join(NEW), new_lr_mult=new_lr_mult, new_warmup=new_warmup,
                              new_from=int(blob["update"]))
    torch.manual_seed(int(blob["update"]))
    grown = T.build(gc, "cpu")
    res = grown.load_state_dict(blob["model"], strict=False)
    new = {n for n in grown.state_dict() if n.startswith(NEW)}
    if set(res.missing_keys) != new or res.unexpected_keys:
        raise SystemExit(f"unexpected load: missing {sorted(set(res.missing_keys) - new)[:5]} unexpected {res.unexpected_keys[:5]}")
    # new tensors keep their build-time init (PolicyNet.zero_growth: every grow_* output projection zero; the job-ETA *_in
    # projections random so gradients reach them); parent tensors were just loaded, so nothing is re-zeroed here
    popt = T.make_optimizer(parent, pcfg)
    popt.load_state_dict(blob["optimizer"])
    pstate = {n: popt.state[p] for n, p in parent.named_parameters() if p in popt.state}
    gopt = T.make_optimizer(grown, cfg)
    names = {id(p): n for n, p in grown.named_parameters()}
    for g in gopt.param_groups:
        for p in g["params"]:
            st = pstate.get(names[id(p)])
            if st:
                gopt.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    pp = dict(parent.named_parameters())
    for n, p in grown.named_parameters():
        if n in pp:
            assert torch.equal(p, pp[n]), n
    out = {**blob, "config": grown.config.to_dict(), "model": grown.state_dict(), "optimizer": gopt.state_dict(),
           "train_config": dataclasses.asdict(cfg)}
    info = {"update": blob["update"], "v6": bool(v6), "cash": bool(cash), "job_eta": bool(job_eta), "aux": bool(v6 and aux),
            "recur": {"core": recur, "loops": int(recur_loops), "train_max": int(recur_train_max)} if recur else None,
            "new_tensors": len(new),
            "parent_parameters": parent.parameter_count(), "grown_parameters": grown.parameter_count(),
            "groups": [len(g["params"]) for g in gopt.param_groups], "lr_mult": [g["lr_mult"] for g in gopt.param_groups],
            "new_warmup": new_warmup, "adam_state_carried": sum(1 for p in gopt.state if gopt.state[p])}
    return out, info, parent, grown


def verify(parent, grown):
    """Teacher-forced loss of parent and child on the test_v6 batch (a recorded game with v6 arrays): must be identical."""
    import test_v6
    batch = test_v6._batch()
    with torch.no_grad():
        parent.eval()
        grown.eval()
        lp, mp, _, _ = T.run_window_batch(parent, batch, device="cpu", bf16=False)
        lg, mg, _, _ = T.run_window_batch(grown, batch, device="cpu", bf16=False, aux_w=(0.05, 0.03))
    same = float(lp) == float(lg) and all(mp.pairs[k] == mg.pairs[k] for k in ("unit_tok", "order_tok", "unit_qty", "order_qty"))
    if getattr(grown, "_recur", None) is not None:   # every loop count the child may run must also equal the parent
        for loops in range(1, max(int(grown.config.recur_train_max or 0), int(grown.config.recur_loops), 4) + 1):
            grown.recur_T = loops
            with torch.no_grad():
                lr_, mr_, _, _ = T.run_window_batch(grown, batch, device="cpu", bf16=False, aux_w=(0.05, 0.03))
            same = same and float(lr_) == float(lp) and all(mp.pairs[k] == mr_.pairs[k]
                                                             for k in ("unit_tok", "order_tok", "unit_qty", "order_qty"))
        grown.recur_T = None
    return {"parent_loss": float(lp), "grown_loss": float(lg), "identical": same,
            "history_tokens": int((batch["prev_unit"][..., 0] >= 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--no-v6", action="store_true")
    ap.add_argument("--cash", action="store_true")
    ap.add_argument("--job-eta", action="store_true")
    ap.add_argument("--new-lr-mult", type=float, default=10.0)
    ap.add_argument("--new-warmup", type=int, default=200)
    ap.add_argument("--aux-next-w", type=float, default=0.01)
    ap.add_argument("--aux-opp-w", type=float, default=0.006)
    ap.add_argument("--recur", default="", help="a:b = encoder blocks a..b-1 become a shared core looped at inference")
    ap.add_argument("--recur-loops", type=int, default=2, help="core loops at inference/evaluation")
    ap.add_argument("--recur-train-max", type=int, default=4, help="training samples 1..N loops per update (0/1 = fixed)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    blob = torch.load(a.parent, map_location="cpu", weights_only=False)
    out, info, parent, grown = grow(blob, aux=not a.no_aux, new_lr_mult=a.new_lr_mult, new_warmup=a.new_warmup,
                                    aux_next_w=a.aux_next_w, aux_opp_w=a.aux_opp_w, v6=not a.no_v6, cash=a.cash,
                                    job_eta=a.job_eta, recur=a.recur, recur_loops=a.recur_loops,
                                    recur_train_max=a.recur_train_max)
    info["verify"] = verify(parent, grown)
    if not info["verify"]["identical"]:
        print(json.dumps(info, indent=1))
        raise SystemExit("grafted network is not identical to the parent; nothing written")
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    name = f"ckpt-{int(blob['update']):07d}.pt"
    torch.save(out, d / (name + ".tmp"))
    (d / (name + ".tmp")).rename(d / name)
    (d / "LATEST").write_text(name + "\n")
    info["parent"] = str(a.parent)
    (d / "GROW-V6.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
