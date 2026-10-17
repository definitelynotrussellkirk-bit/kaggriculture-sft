#!/usr/bin/env python3
"""Update-0 seed for a FROM-SCRATCH practice screen (owner 2026-09-28: "just get a good base model", "24 hours of training is enough
to do a new from scratch"): random init exactly as fit() does (torch.manual_seed(cfg.seed)), cosine LR schedule over --updates,
written into a practice pack's seeds/<name>/ so run_arm.sh / arm-multi-local.sbatch resume it on CUDA. The auxiliary CUDA generator state is copied from a trusted CUDA-saved checkpoint (a CPU generator state cannot be loaded
into a CUDA generator). Actual model initialization uses --seed. CPU RNG follows that initialization; on first CUDA resume the
trainer explicitly seeds the CUDA default RNG and records that this is not an exact CUDA-state resume.
Recurrent core (--recur a:b): its gate starts at --gate-init (tanh(2.0) = 0.96, i.e. loops ON from the first update), unlike the
inert graft (gate 0).

  python make_practice_scratch.py PACK/seeds/NAME --updates 600 --warmup 50 [--width 288 --layers 12 --ffn gelu]
         [--recur 4:8 --recur-train-max 4 --recur-loops 2 --gate-init 2.0] [--lr 3e-4] [--batch-windows 64]
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

import model as M
import train as T

RNG_FROM = "/mnt/ai/scratch/farmNEAT/pbc-colab-v6/seeds/control/ckpt-0048456.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--updates", type=int, required=True)
    ap.add_argument("--warmup", type=int, required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0, help="actual initialization and training/data seed")
    ap.add_argument("--grow-lr-mult", type=float, default=1.0, help="scratch default; not the graft multiplier")
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--batch-windows", type=int, default=64)
    ap.add_argument("--width", type=int, default=288)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", default="gelu", choices=("gelu", "swiglu"))
    ap.add_argument("--recur", default="")
    ap.add_argument("--recur-train-max", type=int, default=4)
    ap.add_argument("--recur-loops", type=int, default=2)
    ap.add_argument("--gate-init", type=float, default=2.0)
    ap.add_argument("--recur-mode", default="gate", choices=("gate", "norm"))
    ap.add_argument("--rng-from", default=RNG_FROM)
    ap.add_argument("--dec-layers", type=int, default=4)
    ap.add_argument("--dec-heads", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--feature-dropout", type=float, default=0.0)
    ap.add_argument("--v6", action="store_true", help="all built v6 inputs: own last two turns, cash ladder, job ETA, aux heads")
    ap.add_argument("--optimizer", default="adamw", choices=("adamw", "muon"))
    ap.add_argument("--v8", action="store_true", help="v8 families + lattice CNN (widths from v8.widths())")
    ap.add_argument("--aux-turn", action="store_true",
                    help="with --v6: training-only heads for worker quantities and order sequences (own t+1/t+2, opponent t..t+2)")
    ap.add_argument("--aux-turn-w", default="0.045,0.035,0.015,0.0123,0.03,0.026",
                    help="aux_next_w,aux_opp_w,aux_next_qty_w,aux_opp_qty_w,aux_next_order_w,aux_opp_order_w "
                         "(example_lag_mean reduction, lag decay 0.5, order-qty scale 0.3: PRO-AUX-TURN-0928 R2 table)")
    a = ap.parse_args()
    shape = argparse.Namespace(tiny=False, size="12m", width=a.width, layers=a.layers, heads=a.heads, dec_layers=a.dec_layers, field_g=0,
                               field_k=0, field_c=0, think="off", ffn=a.ffn,
                               features="extra,unit_extra,calc,prefix,reach_bias,carrier_read,reach_job")
    mc = T._model_cfg_from_args(shape)
    mc = dataclasses.replace(mc, dec_layers=a.dec_layers, dec_heads=a.dec_heads, dropout=a.dropout, feature_dropout=a.feature_dropout)
    if a.v8:
        import v8
        tw, uw, lc = v8.widths()
        mc = dataclasses.replace(mc, x8_token_width=tw, x8_unit_width=uw, lattice_channels=lc)
    if a.v6:
        import prefix
        mc = dataclasses.replace(mc, v6_prev=True, v6_aux=True, prefix_cash_width=prefix.F_CASH, job_eta_v6=True,
                                 input_schema="pbc-job-input-v6.0")
    if a.aux_turn:
        if not a.v6:
            raise SystemExit("--aux-turn needs --v6")
        mc = dataclasses.replace(mc, v6_aux_turn=True)
    if a.recur:
        mc = dataclasses.replace(mc, recur_core=a.recur, recur_loops=a.recur_loops, recur_train_max=a.recur_train_max,
                                 recur_mode=a.recur_mode)
    cfg = dataclasses.replace(T.TrainConfig(), updates=a.updates, warmup=a.warmup, lr=a.lr, min_lr_frac=a.min_lr_frac,
                              batch_windows=a.batch_windows, window=24, priority_file="rows.tsv", priority_only=True,
                              min_final_quadrants=0, ckpt_every=50, keep_ckpts=100, eval_every=0, bf16=True, grad_checkpoint=True,
                              log_every=10, seed=a.seed, grow_lr_mult=a.grow_lr_mult,
                              workers=14, archives=[], v6_data=a.v6, optimizer=a.optimizer)
    if a.aux_turn:
        wn_, wo_, wq, wo, wn, wp = (float(v) for v in a.aux_turn_w.split(","))
        cfg = dataclasses.replace(cfg, aux_next_w=wn_, aux_opp_w=wo_, aux_next_qty_w=wq, aux_opp_qty_w=wo,
                                  aux_next_order_w=wn, aux_opp_order_w=wp, aux_reduction="example_lag_mean",
                                  aux_lag_decay=0.5, aux_order_qty_scale=0.3)
    torch.manual_seed(cfg.seed)
    net = T.build(mc, "cpu")
    if a.recur and a.recur_mode == "gate":
        with torch.no_grad():
            net.grow_recur_gate.fill_(a.gate_init)     # build() zeroes grow_*; a scratch recurrence starts with its loops on
    opt = T.make_optimizer(net, cfg)
    ref = torch.load(a.rng_from, map_location="cpu", weights_only=True)

    class _Gen:
        def get_state(self):
            return ref["generator_state"]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Keep the CPU RNG following this seed's initialization, not an unrelated parent's state.
    path = T.save_checkpoint(out, net, opt, cfg, update=0, windows_consumed=0, epoch=0, resumes=0, generator=_Gen(), metrics={})
    (out / "LATEST").write_text(Path(path).name + "\n")
    info = {"seed": str(path), "init_seed": a.seed, "grow_lr_mult": a.grow_lr_mult,
            "params": net.parameter_count(), "width": a.width, "layers": a.layers, "ffn": a.ffn,
            "recur": a.recur or None, "recur_mode": a.recur_mode if a.recur else None,
            "gate_init": a.gate_init if a.recur and a.recur_mode == "gate" else None, "lr": a.lr, "warmup": a.warmup,
            "updates": a.updates, "batch_windows": a.batch_windows, "dec_layers": a.dec_layers, "dropout": a.dropout,
            "feature_dropout": a.feature_dropout, "v6": a.v6, "optimizer": a.optimizer, "v8": a.v8,
            "aux_turn": a.aux_turn, "aux_turn_w": a.aux_turn_w if a.aux_turn else None}
    (out / "SCRATCH.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
