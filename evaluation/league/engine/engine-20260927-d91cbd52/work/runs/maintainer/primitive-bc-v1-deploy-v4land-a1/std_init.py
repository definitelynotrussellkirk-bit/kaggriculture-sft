"""Standard transformer initialization for the frozen v1 PolicyNet (GPT-2 style)."""
import math

import torch
from torch import nn

RESIDUAL_OUT = ("proj", "ff2", "o1", "o2")
HEADS = ("unit_head", "order_head", "unit_qty_head", "order_qty_head", "value_head")


def apply(net):
    c = net.config
    enc_std = 0.02 / math.sqrt(2 * max(1, c.layers))
    dec_std = 0.02 / math.sqrt(2 * max(1, c.dec_layers))
    for name, m in net.named_modules():
        if isinstance(m, nn.Linear):
            leaf = name.rsplit(".", 1)[-1]
            std = 0.02
            if leaf in RESIDUAL_OUT:
                std = enc_std if name.startswith("blocks.") else dec_std
            nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    for name, p in net.named_parameters(recurse=False):
        nn.init.normal_(p, std=0.02) if p.ndim >= 2 else nn.init.zeros_(p)
    for head in HEADS:
        mod = getattr(net, head, None)
        if mod is None:
            continue
        last = mod[-1] if isinstance(mod, nn.Sequential) else mod
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)
    return net
