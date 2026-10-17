"""Update-0 seed checkpoint for a from-scratch fit that launch-phase-local.sbatch can resume on CUDA (owner 2026-09-27: "try to
train both ... from scratch", GELU vs SwiGLU, "fix the data they train over, and time the run"). Random init with
torch.manual_seed(cfg.seed) exactly as fit() does; the CUDA data-order generator state and CPU RNG are copied from a CUDA-saved
checkpoint (a CPU generator state cannot be loaded into a CUDA generator).

  python make_scratch_seed.py OUT_DIR --ffn gelu|swiglu --rng-from CKPT.pt --updates 250 --warmup 25 --priority-file TSV --archives ...
"""
import argparse
import dataclasses
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
    ap.add_argument("--ffn", required=True, choices=("gelu", "swiglu"))
    ap.add_argument("--rng-from", required=True)
    ap.add_argument("--updates", type=int, required=True)
    ap.add_argument("--warmup", type=int, required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--priority-file", required=True)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--width", type=int, default=288)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--archives", nargs="+", required=True)
    a = ap.parse_args()
    shape = argparse.Namespace(tiny=False, size="12m", width=a.width, layers=a.layers, heads=8, dec_layers=4, field_g=0, field_k=0, field_c=0,
                               think="off", ffn=a.ffn, features="extra,unit_extra,calc,prefix,reach_bias,carrier_read,reach_job")
    mc = T._model_cfg_from_args(shape)
    cfg = dataclasses.replace(T.TrainConfig(), updates=a.updates, warmup=a.warmup, lr=a.lr, min_lr_frac=a.min_lr_frac,
                              batch_windows=64, window=24, priority_file=a.priority_file, priority_only=True,
                              min_final_quadrants=0, ckpt_every=a.ckpt_every, keep_ckpts=100, eval_every=0, bf16=True,
                              grad_checkpoint=True, log_every=10, seed=0, workers=14, archives=list(a.archives))
    torch.manual_seed(cfg.seed)
    net = T.build(mc, "cpu")
    opt = T.make_optimizer(net, cfg)
    ref = torch.load(a.rng_from, map_location="cpu", weights_only=False)

    class _Gen:
        def get_state(self):
            return ref["generator_state"]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_rng_state(ref["torch_rng"])
    path = T.save_checkpoint(out, net, opt, cfg, update=0, windows_consumed=0, epoch=0, resumes=0, generator=_Gen(),
                             metrics={})
    print(path, mc.ffn, net.parameter_count())
    return 0


if __name__ == "__main__":
    sys.exit(main())
