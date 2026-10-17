"""LLM-paper style architecture diagrams for the two final models (data flows bottom to top). Structure follows the code:
B-even: primitive-bc model.py (EncoderBlock, DecoderLayer, PolicyNet.encode / query building); N flow: n-flow/code/nf_model.py.
Run with /usr/bin/python3 -s -B kagglewriteup/figures/final/arch_figs.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent
INK, INK2, MUTED, GRID = "#16202c", "#52514e", "#8a8984", "#d9d7d0"
GREEN, ORANGE = "#1baf7a", "#eb6834"
C = dict(state=("#f4f6fa", "#6f6e69"), feat=("#fff4d6", "#c99a2e"), embed=("#eef1f5", "#8a8984"), norm=("#f2f2ef", "#a5a39c"),
         attn=("#dde9f9", "#2a78d6"), ffn=("#ece3f7", "#8e5cc4"), mem=("#fff0e0", ORANGE), head=("#fdeee7", ORANGE),
         aux=("#ffffff", "#a5a39c"), out=("#e8f6ef", GREEN))


def setup(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / w)
    ax.axis("off")
    return fig, ax, 100 * h / w


def box(ax, x, y, w, h, text, kind, fs=10, bold=False, ls="-", lw=1.4, ha="center"):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6", fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=3))
    tx = x + w / 2 if ha == "center" else x + 1.2
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, weight="bold" if bold else "normal", color=INK, zorder=4,
            linespacing=1.35)
    return (x, y, w, h)


def frame(ax, x, y, w, h, label, rep=""):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="#fafafa", ec=GRID, lw=1.6, zorder=1))
    ax.text(x + 1, y + 1, label, ha="left", va="bottom", fontsize=10.5, weight="bold", color=INK2, zorder=2)
    if rep:
        ax.text(x + w - 1, y + 1, rep, ha="right", va="bottom", fontsize=16, weight="bold", color=INK, zorder=2)


def arr(ax, pts, color=INK2, lw=1.5, ls="-"):
    """Polyline through pts with an arrowhead on the last segment."""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=2))


def plus(ax, x, y):
    ax.add_patch(plt.Circle((x, y), 1.0, fc="white", ec=INK2, lw=1.3, zorder=4))
    ax.text(x, y - 0.05, "+", ha="center", va="center", fontsize=11, weight="bold", zorder=5)


def up_stack(ax, x, y0, w, layers, h=3.6):
    """Draw layers bottom->top joined by arrows; '+' is a residual add. Returns (y of top edge, dict name->box)."""
    y, prev, boxes = y0, None, {}
    for text, kind in layers:
        if text == "+":
            cy = y + 1.6
            if prev is not None:
                arr(ax, [prev, (x + w / 2, cy - 1.0)])
            plus(ax, x + w / 2, cy)
            prev, y = (x + w / 2, cy + 1.0), cy + 2.4
        else:
            b = box(ax, x, y + 0.6, w, h, text, kind, fs=9.6)
            if prev is not None:
                arr(ax, [prev, (x + w / 2, y + 0.6)])
            boxes[text] = b
            prev, y = (x + w / 2, y + 0.6 + h), y + 0.6 + h + 0.8
    return prev, boxes


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Model 1: B-even -----------------------------------------
def arch_b():
    fig, ax, H = setup(16, 14.5)
    ax.text(1, H - 1, "Model 1: B-even (54.6M)", fontsize=21, weight="bold", va="top")
    ax.text(1, H - 5, "Encoder-decoder transformer, width 432, pre-LayerNorm, SwiGLU. Data flows bottom to top; one action is decoded one token at a time.",
            fontsize=11.5, color=INK2, va="top")
    # lane 1: game state and its two products
    gs = box(ax, 1, 2, 23, 6.5, "GAME STATE\nwhat our seat sees", "state", fs=11.5, bold=True)
    tok = box(ax, 1, 14, 23, 9, "one token per entity\ntiles, workers, stock,\nmarket, shops\n80 features + known flags", "embed", fs=9.6)
    fe = box(ax, 1, 27, 23, 16, "preprocess to features\njob timing, worker ETA\nreachability, cash effects\ncrop / timing / resource gaps\nspatial history of the last\n2 observations (CNN)", "feat", fs=9.6)
    arr(ax, [(8, 8.5), (8, 14)])
    arr(ax, [(3, 8.5), (3, 27)])
    # lane 2: encoder
    emb = box(ax, 29, 14, 18, 5, "linear embed", "embed", fs=10)
    arr(ax, [(24, 16.5), (29, 16.5)])
    plus(ax, 38, 23)
    arr(ax, [(38, 19), (38, 22)])
    arr(ax, [(24, 35), (26.5, 35), (26.5, 23), (37, 23)])
    frame(ax, 27.5, 25.5, 21, 36, "encoder block", "× 10")
    top_enc, _ = up_stack(ax, 29, 29, 18, [("LayerNorm", "norm"), ("self-attention\n18 heads x 24", "attn"), ("+", None),
                                            ("LayerNorm", "norm"), ("SwiGLU FFN (x4)", "ffn"), ("+", None)])
    arr(ax, [(38, 24), (38, 29.6)])
    mem = box(ax, 29, 64, 18, 5, "LayerNorm → entity memory", "norm", fs=9.4)
    arr(ax, [top_enc, (38, 64)])
    # lane 3: decoder input + decoder
    din = box(ax, 52, 2, 25, 17, "decoder input (summed)\n\nslot query: 25 worker + 11 order slots\nprefix: workers so far, orders so far,\n    cash left after earlier orders\ncontext: worker's tile, carrier, global\nhistory: own last 2 turns (attention)",
              "feat", fs=9.2, ha="left")
    arr(ax, [(24, 5.25), (52, 5.25)])
    frame(ax, 51, 21.5, 27, 42, "decoder layer", "× 8")
    top_dec, b = up_stack(ax, 54, 25, 21, [("LayerNorm", "norm"), ("causal self-attention\n12 heads x 36, KV cache", "attn"), ("+", None),
                                            ("LayerNorm", "norm"), ("cross-attention to memory\n+ reach bias: ETA, job, slack", "attn"),
                                            ("+", None), ("LayerNorm", "norm"), ("SwiGLU FFN (x4)", "ffn"), ("+", None)])
    arr(ax, [(64.5, 19), (64.5, 25.6)])
    ca = b["cross-attention to memory\n+ reach bias: ETA, job, slack"]
    arr(ax, [(47, 66.5), (50, 66.5), (50, ca[1] + ca[3] / 2), (54, ca[1] + ca[3] / 2)], color="#2a78d6")
    # heads and output
    hd = box(ax, 54, 66.5, 21, 7.5, "command head (44) + qty (75)\norder head (23) + qty (75)", "head", fs=9.4)
    arr(ax, [top_dec, (64.5, 66.5)])
    outb = box(ax, 54, 77, 21, 5.5, "workers → orders → STOP", "out", fs=10.5, bold=True)
    arr(ax, [(64.5, 74), (64.5, 77)])
    arr(ax, [(75, 79.75), (79.5, 79.75), (79.5, 10.5), (77, 10.5)], color=GREEN, ls="--")
    ax.text(80.3, 45, "next token\n(autoregressive)", fontsize=9.5, color=GREEN, va="center")
    aux = box(ax, 83, 52, 16, 14, "training only\n\naux order decoder\nnext-turn prediction\nopponent's actions\nvalue", "aux", fs=9.2, ls="--")
    arr(ax, [(64.5, 65.2), (80.5, 65.2), (80.5, 59), (83, 59)], color=MUTED, ls="--")
    save(fig, "figG-arch-b")


# ---------------------------------------------------------------- Model 2: N flow ------------------------------------------
def arch_n():
    fig, ax, H = setup(16, 13.2)
    ax.text(1, H - 1, "Model 2: N flow (7.25M)", fontsize=21, weight="bold", va="top")
    ax.text(1, H - 5, "A typed graph of 320 nodes, three recurrent memories and a 2-layer decoder. Width 256, random init. Data flows bottom to top.",
            fontsize=11.5, color=INK2, va="top")
    gs = box(ax, 1, 2, 22, 6.5, "GAME STATE\nwhat our seat sees", "state", fs=11.5, bold=True)
    ent = box(ax, 1, 13, 22, 12, "296 entity slots\ntyped projection per kind:\ntiles 200, carriers 50,\nstock 17, market 9, shops 8,\nconstants 8, farms 2, global,\ncontrol", "embed", fs=9.2)
    cal = box(ax, 1, 28, 22, 8, "8 calendar landmarks\nnow, next shop tick, dusk,\n+2d, +4d, +8d, +12d, end", "embed", fs=9.2)
    fe = box(ax, 1, 39, 22, 10, "preprocess to features\nprefix, worker ETA,\ncrop / timing / resource gaps,\nown previous action", "feat", fs=9.2)
    arr(ax, [(8, 8.5), (8, 13)])
    arr(ax, [(4, 8.5), (4, 28)])
    arr(ax, [(2, 8.5), (2, 39)])
    # lane 2: graph
    nodes = box(ax, 27, 13, 19, 6, "320 nodes\n296 + FAST + SLOW + 8 + pad", "embed", fs=9.2)
    arr(ax, [(23, 16), (27, 16)])
    arr(ax, [(23, 32), (25, 32), (25, 18), (27, 18)])
    frame(ax, 26, 21.5, 21, 34, "relational block", "× 3")
    top_g, _ = up_stack(ax, 27, 25, 19, [("RMSNorm", "norm"), ("attention: 4 relation heads\n(17 relation-type biases)\n+ 4 global heads", "attn"),
                                          ("+", None), ("RMSNorm", "norm"), ("SwiGLU FFN", "ffn"), ("+", None)], h=4.0)
    arr(ax, [(36.5, 19), (36.5, 25.6)])
    b4 = box(ax, 27, 58, 19, 7.5, "relational block 4\nreads memory into\nFAST / SLOW / carrier nodes", "attn", fs=9.2)
    arr(ax, [top_g, (36.5, 58)])
    # lane 3: memory
    frame(ax, 50, 21.5, 22, 34, "memory, carried across steps")
    m1 = box(ax, 51.5, 45, 19, 7, "fast GRU 128\nevery step: graph summary\n+ own previous action", "mem", fs=9)
    m2 = box(ax, 51.5, 35, 19, 7, "slow GRU 256\nat dawn: yesterday's\nmean summary", "mem", fs=9)
    m3 = box(ax, 51.5, 25, 19, 7, "worker GRU 64, per worker\nhands reset at dawn,\nthe farmer never", "mem", fs=9)
    arr(ax, [(46, 40), (51.5, 48.5)], color=ORANGE)
    ax.text(47, 43.5, "summary", fontsize=8.8, color=ORANGE, rotation=58)
    arr(ax, [(61, 55.5), (61, 61.75), (46, 61.75)], color=ORANGE)
    ax.text(61.8, 58.3, "all three memories", fontsize=8.8, color=ORANGE)
    arr(ax, [(23, 44), (24.5, 44), (24.5, 69.5), (71.2, 69.5), (71.2, 48.5), (70.5, 48.5)], color=MUTED, ls="--")
    ax.text(48, 70.4, "own previous action", fontsize=8.8, color=MUTED, ha="center")
    # lane 4: decoder
    din = box(ax, 76, 2, 23, 12, "decoder input (summed)\nquery: 25 worker + ≤11 order\nslots, prefix features,\nworker memory, carrier node", "feat", fs=9, ha="left")
    arr(ax, [(23, 5.25), (76, 5.25)])
    frame(ax, 75, 17, 25, 33, "decoder layer", "× 2")
    top_d, b = up_stack(ax, 77, 20, 21, [("causal self-attention", "attn"), ("+", None), ("cross-attention to nodes\n+ per-head ETA bias", "attn"),
                                          ("+", None), ("SwiGLU FFN", "ffn"), ("+", None)], h=4.0)
    arr(ax, [(87.5, 14), (87.5, 20.6)])
    ca = b["cross-attention to nodes\n+ per-head ETA bias"]
    arr(ax, [(46, 64), (74, 64), (74, ca[1] + ca[3] / 2), (77, ca[1] + ca[3] / 2)], color="#2a78d6")
    hd = box(ax, 76, 53, 13.5, 7, "command (44)\norder (23)\n+ quantity (75)", "head", fs=9)
    arr(ax, [top_d, (top_d[0], 51.2), (82.75, 51.2), (82.75, 53)])
    aux = box(ax, 90.5, 53, 9, 7, "training\nonly: WDL,\nnext step,\nopponent", "aux", fs=8.2, ls="--")
    arr(ax, [(87.5, 51.2), (95, 51.2), (95, 53)], color=MUTED, ls="--")
    outb = box(ax, 76, 64, 23, 5.5, "workers → orders → STOP", "out", fs=10.5, bold=True)
    arr(ax, [(82.75, 60), (82.75, 64)])
    ax.text(87.5, 70.4, "one token at a time", ha="center", fontsize=9.5, color=GREEN)
    save(fig, "figH-arch-nflow")


if __name__ == "__main__":
    arch_b()
    arch_n()
