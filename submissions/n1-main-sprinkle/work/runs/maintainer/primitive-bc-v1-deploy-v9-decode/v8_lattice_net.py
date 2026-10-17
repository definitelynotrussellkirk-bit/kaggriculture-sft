"""Space-time lattice CNN (pbc-v6-design/DESIGN.md item 7) and its scatter onto the policy's TILE tokens.

LatticeCNN(in_ch, width=64, *, out_dim): input [B, 2, 10, 10, in_ch] (lattice.compute_lattice, farm 0 own, 1 opponent;
channels last). Each farm is run separately through the same two 3x3 convolutions (padding 1, GELU), with a learned farm
embedding added after the first convolution; a per-cell Linear(width, out_dim), initialised to ZERO, gives
[B, 2, 10, 10, out_dim]. At initialisation the output is exactly zero, so adding it to the TILE tokens leaves a grafted
policy unchanged.

scatter_to_tokens(cell_out, cell_of, is_tile) -> [B, N, out_dim]: the value of cell (farm, y, x) on the token whose
model.py PolicyNet._tiles cell is farm * 100 + y * 10 + x, zero on every non-TILE token. Intended use inside
PolicyNet.encode, after `is_tile, cell_of, tile_index = self._tiles(rows, token_mask)` and before the blocks:

    x = x + lattice_net.scatter_to_tokens(self.lattice_cnn(lattice), cell_of, is_tile)

Integration note: do NOT name the module attribute with the "grow_" prefix. PolicyNet.zero_growth() zeroes every
parameter whose name starts with "grow_", which would zero both convolutions and the farm embedding as well; with the
zero output layer that leaves no gradient path (the branch would stay dead). Only `out` must start at zero, and
LatticeCNN does that itself (zero_output()).
"""
from __future__ import annotations

import statistics
import time

import torch
import torch.nn.functional as F
from torch import nn

BOARD = 10
N_CELLS = 2 * BOARD * BOARD


class LatticeCNN(nn.Module):
    def __init__(self, in_ch, width=64, *, out_dim):
        super().__init__()
        self.in_ch, self.width, self.out_dim = int(in_ch), int(width), int(out_dim)
        self.conv1 = nn.Conv2d(self.in_ch, self.width, 3, padding=1)
        self.conv2 = nn.Conv2d(self.width, self.width, 3, padding=1)
        self.farm_emb = nn.Embedding(2, self.width)
        nn.init.normal_(self.farm_emb.weight, std=0.02)
        self.out = nn.Linear(self.width, self.out_dim)
        self.zero_output()

    def zero_output(self):
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()
        return self

    def forward(self, lattice):
        """lattice [B, 2, 10, 10, in_ch] (any float dtype) -> [B, 2, 10, 10, out_dim]."""
        B, farms, H, W, Cin = lattice.shape
        if farms != 2 or H != BOARD or W != BOARD or Cin != self.in_ch:
            raise ValueError(f"lattice must be [B, 2, {BOARD}, {BOARD}, {self.in_ch}], got {tuple(lattice.shape)}")
        dtype = self.conv1.weight.dtype
        x = lattice.to(dtype).permute(0, 1, 4, 2, 3).reshape(B * 2, Cin, H, W)      # farms share the conv weights
        farm = self.farm_emb.weight.to(dtype).repeat(B, 1)[:, :, None, None]         # row b * 2 + f -> farm f
        h = F.gelu(self.conv1(x) + farm)
        h = F.gelu(self.conv2(h))
        h = h.reshape(B, 2, self.width, H, W).permute(0, 1, 3, 4, 2)
        return self.out(h)


def scatter_to_tokens(cell_out, cell_of, is_tile):
    """cell_out [B, 2, 10, 10, D]; cell_of [B, N] long and is_tile [B, N] bool from PolicyNet._tiles -> [B, N, D]."""
    B, D = cell_out.shape[0], cell_out.shape[-1]
    N = cell_of.shape[1]
    flat = cell_out.reshape(B, N_CELLS, D)                                          # index farm * 100 + y * 10 + x
    add = flat.gather(1, cell_of.long().clamp(0, N_CELLS - 1)[..., None].expand(B, N, D))
    return add * is_tile[..., None].to(add.dtype)


def cpu_latency(in_ch, width=64, out_dim=192, *, batch=1, threads=1, repeats=200, warmup=20, n_tokens=250):
    """Median / p90 ms of LatticeCNN forward + scatter on CPU at the given batch and thread count (no grad)."""
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        net = LatticeCNN(in_ch, width, out_dim=out_dim).eval()
        with torch.no_grad():
            nn.init.normal_(net.out.weight, std=0.02)                              # timing is value-independent
        g = torch.Generator().manual_seed(0)
        lat = torch.rand(batch, 2, BOARD, BOARD, in_ch, generator=g).to(torch.float16)
        cell_of = torch.arange(n_tokens).remainder(N_CELLS).expand(batch, n_tokens)
        is_tile = cell_of < N_CELLS
        times = []
        with torch.no_grad():
            for i in range(warmup + repeats):
                t0 = time.perf_counter()
                scatter_to_tokens(net(lat.float()), cell_of, is_tile)
                if i >= warmup:
                    times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        return {"median_ms": statistics.median(times), "p90_ms": times[int(0.9 * len(times))],
                "params": sum(p.numel() for p in net.parameters()), "batch": batch, "threads": threads}
    finally:
        torch.set_num_threads(old)
