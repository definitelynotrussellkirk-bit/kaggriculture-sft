"""CUDA checks for N flow (run by smoke.sbatch before training):
1. attention backends: global heads and the decoder's causal self-attention run flash SDPA (bf16, head dim 32, no
   mask; raises if flash is not selectable); biased relational heads run the memory-efficient kernel.
2. torch.compile vs eager: on one fixed real batch (2 streams x 24 steps from the roster), the compiled encoder and
   post-memory encoder+decoder give the eager logits within bf16 tolerance, and the same loss.
"""
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def backends():
    q = torch.randn(64, 4, 320, 32, device="cuda", dtype=torch.bfloat16)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        F.scaled_dot_product_attention(q, q, q)
        F.scaled_dot_product_attention(q[:, :, :36], q[:, :, :36], q[:, :, :36], is_causal=True)
    bias = torch.zeros(1, 4, 320, 320, device="cuda", dtype=torch.bfloat16).expand(64, -1, -1, -1)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        F.scaled_dot_product_attention(q, q, q, attn_mask=bias)
    print("PASS attention: flash (global heads, decoder causal self-attn); mem-efficient (biased relational heads)",
          flush=True)


def compile_equal():
    import nf_data
    import nf_format as NF
    import train as T
    cols = NF.load_columns()
    rows = [r for r in NF.read_roster(os.environ.get("NFLOW_ROSTER",
            "/mnt/ai/scratch/farmNEAT/n-flow/roster/rows-nflow.1ad4d8296e84.tsv")) if r["act_loss"]][:4]
    x = nf_data.gather(iter(nf_data.Streams(rows, None, cols, streams=2, chunk=24, seed=5, first_random=False)), 1)
    x.pop("eid"), x.pop("seat")
    x = {k: v.cuda() for k, v in x.items()}
    torch.manual_seed(0)
    net = T.build_net(cols, NF.PRIMITIVE_BC).cuda().eval()
    outs = {}
    for name, pre, post in (("eager", net.encode_pre, net.post_teacher),
                            ("compiled", torch.compile(net.encode_pre, dynamic=False),
                             torch.compile(net.post_teacher, dynamic=False))):
        t = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = net.forward_chunk(x, net.initial_memory(2, "cuda"), pre, post)
            loss, _ = T.chunk_loss(out, x, net)
        torch.cuda.synchronize()
        outs[name] = (out, float(loss), time.time() - t)
    worst = 0.0
    for k in ("unit_logits", "unit_qty_logits", "order_logits", "order_qty_logits", "wdl_logits"):
        a, b = outs["eager"][0][k].float(), outs["compiled"][0][k].float()
        fin = torch.isfinite(a)
        worst = max(worst, (a[fin] - b[fin]).abs().max().item())
    le, lc = outs["eager"][1], outs["compiled"][1]
    ok = worst < 0.15 and abs(le - lc) < 0.02 * max(1.0, abs(le))
    print(f"{'PASS' if ok else 'FAIL'} compile vs eager: max |logit diff| {worst:.4f} (bf16), loss {le:.5f} vs {lc:.5f}, "
          f"first-call s eager {outs['eager'][2]:.1f} compiled {outs['compiled'][2]:.1f}", flush=True)
    return ok


def main():
    if not torch.cuda.is_available():
        print("FAIL no CUDA (this check must run on a GPU node)")
        return 2
    backends()
    return 0 if compile_equal() else 1


if __name__ == "__main__":
    sys.exit(main())
