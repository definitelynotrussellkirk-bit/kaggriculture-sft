"""Figures for the final write-up (30 Sep 2026). Numbers are typed in with their source; fig7 reads b-vs-p1-spread.json.

Run: /usr/bin/python3 -s -B kagglewriteup/figures/final/make_final_figures.py [name ...]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA, PURPLE, GOLD, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8e5cc4", "#c99a2e", "#c0392b"
INK, INK2, MUTED, GRID = "#16202c", "#52514e", "#8a8984", "#e6e5e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": MUTED, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("wrote", name)


def rbox(ax, x, y, w, h, fc, ec, lw=1.4, r=0.015):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(ax, a, b, color=INK2, lw=1.8, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# 2 ---------------------------------------------------------------------------------------------------------------
def decision():
    fig, ax = canvas(12, 4.6)
    ax.text(0.01, 0.97, "One decision = one sentence", fontsize=19, weight="bold", va="top")
    ax.text(0.01, 0.87, "719 decisions per game. Each token is conditioned on the state and the tokens already written.",
            fontsize=12.5, color=INK2, va="top")
    rbox(ax, 0.01, 0.17, 0.2, 0.57, "#f4f6fa", MUTED)
    ax.text(0.025, 0.7, "What our seat sees", fontsize=13.5, weight="bold", va="top")
    for i, s in enumerate(["both 10x10 farms", "workers, stock, cash", "prices, shops, clock", "visible opponent state"]):
        ax.text(0.025, 0.6 - i * 0.075, s, fontsize=11.5, color=INK2, va="top")
    ax.text(0.025, 0.24, "hidden: their inventory,\nthe seed, the future", fontsize=11, color=MUTED, va="center")
    rbox(ax, 0.25, 0.17, 0.14, 0.57, "#eef4fc", BLUE)
    ax.text(0.265, 0.7, "Encoder", fontsize=13.5, weight="bold", va="top")
    for i, s in enumerate(["token per entity", "+ job timing,", "reachability,", "cash effects"]):
        ax.text(0.265, 0.6 - i * 0.075, s, fontsize=11.5, color=INK2, va="top")
    arrow(ax, (0.212, 0.48), (0.248, 0.48))
    arrow(ax, (0.392, 0.48), (0.428, 0.48))
    toks = [("FEED", "farmer"), ("COLLECT\nFERT", "hand 1"), ("COLLECT\nFERT", "hand 2"), ("EAST", "hand 3"),
            ("SOUTH", "hand 4"), ("CARE", "hand 5"), ("...", ""), ("CARE", "hand 14"), ("STOP", "market")]
    x0, w, gap = 0.43, 0.052, 0.011
    ax.text(x0, 0.74, "Decoder: workers, then market orders", fontsize=13.5, weight="bold", va="top")
    for i, (t, lab) in enumerate(toks):
        x = x0 + i * (w + gap)
        if lab:
            mk = lab == "market"
            rbox(ax, x, 0.4, w, 0.17, "#fdeee7" if mk else "#e8f6ef", ORANGE if mk else AQUA, lw=1.4, r=0.008)
        ax.text(x + w / 2, 0.485, t, ha="center", va="center", fontsize=10 if "\n" in t else 11, weight="bold", family="DejaVu Sans Mono")
        ax.text(x + w / 2, 0.37, lab, ha="center", va="top", fontsize=10.5, color=MUTED)
        if i:
            arrow(ax, (x - gap - w * 0.3, 0.59), (x + w * 0.3, 0.59), color=MUTED, lw=1, rad=-0.6)
    ax.text(x0 + 8 * (w + gap) + w / 2, 0.29, "(no orders\nthis step)", ha="center", va="top", fontsize=9.5, color=ORANGE)
    ax.text(0.01, 0.05, "Real step, day 10, 08:00:  \\$21 cash, 14 hands, 11 wheat in the shed, farmer holding 3 wheat + 1 fertilizer, wheat at \\$35.",
            fontsize=11.5, color=INK2, va="center")
    save(fig, "fig2-decision")


# 3 ---------------------------------------------------------------------------------------------------------------
def models():
    fig, ax = canvas(13, 5.2)
    ax.text(0.01, 0.97, "Final agents", fontsize=19, weight="bold", va="top")
    rows = [0.58, 0.44, 0.30, 0.16]
    hgt = 0.1

    def stack(x0, labels, fills):
        for (s, y, fc) in zip(labels, rows, fills):
            rbox(ax, x0, y, 0.34, hgt, fc, MUTED, lw=1, r=0.01)
            ax.text(x0 + 0.17, y + hgt / 2, s, ha="center", va="center", fontsize=11.2)
        for y in rows[:-1]:
            arrow(ax, (x0 + 0.17, y - 0.002), (x0 + 0.17, y - 0.038))

    rbox(ax, 0.01, 0.04, 0.47, 0.8, "#f3f7fd", BLUE, lw=2, r=0.02)
    ax.text(0.03, 0.8, "B-even: 54.6M transformer", fontsize=15, weight="bold", color=BLUE, va="top")
    ax.text(0.03, 0.735, "warm start from B (1k updates BC + distillation), then SFT", fontsize=11, color=INK2, va="top")
    stack(0.075, ["entity tokens + features of 2 previous obs.", "10 encoder blocks, width 432, 18 x 24 heads",
                  "8 decoder blocks, 12 x 36 heads", "workers → orders → STOP"],
          ["#ffffff", "#dde9f9", "#dde9f9", "#fdeee7"])
    ax.text(0.245, 0.075, "no recurrent state", fontsize=11, color=MUTED, ha="center")

    rbox(ax, 0.52, 0.04, 0.47, 0.8, "#f1faf6", AQUA, lw=2, r=0.02)
    ax.text(0.54, 0.8, "N flow: 7.25M graph + memory", fontsize=15, weight="bold", color="#11825a", va="top")
    ax.text(0.54, 0.735, "random init; first submission 4.7 h after init", fontsize=11, color=INK2, va="top")
    stack(0.54, ["320 slots: entities + memory + calendar", "4 relational blocks, width 256",
                 "2 causal decoder blocks", "workers → orders → STOP"],
          ["#ffffff", "#dcf3e9", "#dcf3e9", "#fdeee7"])
    rbox(ax, 0.9, 0.3, 0.08, 0.38, "#fff7e0", GOLD, lw=1.3, r=0.01)
    ax.text(0.94, 0.655, "memory", ha="center", va="top", fontsize=11, weight="bold", color=GOLD)
    for i, s in enumerate(["fast GRU\n128, per turn", "slow GRU\n256, at dawn", "per-worker\nGRU 64"]):
        ax.text(0.94, 0.57 - i * 0.1, s, ha="center", va="center", fontsize=9.8)
    arrow(ax, (0.9, 0.63), (0.882, 0.63), color=GOLD, lw=1.5)
    arrow(ax, (0.882, 0.49), (0.9, 0.49), color=GOLD, lw=1.5, ls="--")
    ax.text(0.765, 0.075, "state carried all game; hands reset at dawn", fontsize=11, color=MUTED, ha="center")
    save(fig, "fig3-models")


# 4 ---------------------------------------------------------------------------------------------------------------
def data():
    fig = plt.figure(figsize=(14, 5.0))
    fig.text(0.01, 0.99, "Data: winners only, weighted to fresh and strong", ha="left", fontsize=19, weight="bold", va="top")
    ax1 = fig.add_axes([0.25, 0.12, 0.33, 0.68])
    ax2 = fig.add_axes([0.64, 0.12, 0.35, 0.68])
    # one hourly generation (30 Sep 11:51 UTC); seats and effective draw shares from its receipt
    b = [("top-4 beaters + top-12 wins", 4308, 34.95), ("other wins, 29-30 Sep", 2101, 39.95),
         ("top 10%/day, 24-28 Sep", 1157, 14.98), ("top 10%/day, 3-23 Sep", 4759, 4.99), ("winners of games we lost", 158, 5.13)]
    tot = sum(x[1] for x in b)
    for i, (lab, n, sh) in enumerate(b):
        ax1.barh(i - 0.19, 100 * n / tot, height=0.36, color="#cfcdc6")
        ax1.barh(i + 0.19, sh, height=0.36, color=ORANGE if "lost" in lab else BLUE)
        ax1.text(sh + 0.8, i + 0.19, f"{sh:.0f}%", va="center", fontsize=11)
    ax1.set_yticks(range(len(b)))
    ax1.set_yticklabels([x[0] for x in b], fontsize=11.5)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 48)
    ax1.set_xlabel("% of seats (grey) vs % of training draws (colour)")
    ax1.set_title("Hourly mixture, 30 Sep 11:51 UTC\n12,483 cached winner seats", loc="left", fontsize=13, weight="bold")
    ax1.grid(axis="y", visible=False)
    p = [("#1 wins", 71.8, 105, AQUA), ("beat #1 (x4)", 8.2, 3, "#0d6b4a"), ("#2 wins", 14.1, 122, PURPLE), ("#3 wins", 6.0, 174, GOLD)]
    left = 0
    for lab, sh, n, c in p:
        ax2.barh(0, sh, left=left, color=c, height=0.55)
        if sh > 5:
            ax2.text(left + sh / 2, 0, f"{sh:.0f}%", ha="center", va="center", color="white", fontsize=12.5, weight="bold")
        left += sh
    ax2.text(50, -0.38, "Share of training draws after weighting", ha="center", fontsize=11, color=INK2)
    for i, (lab, sh, n, c) in enumerate(p):
        ax2.text(0, -0.6 - i * 0.3, "\u25A0", color=c, fontsize=15, va="center")
        ax2.text(4, -0.6 - i * 0.3, f"{lab}: {n} seats", color=INK, fontsize=11.5, va="center")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-1.7, 0.6)
    ax2.axis("off")
    ax2.set_title("Late pool: 404 source seats, 30 Sep", loc="left", fontsize=13, weight="bold")
    save(fig, "fig4-data")


# 5 ---------------------------------------------------------------------------------------------------------------
def recipe():
    fig, ax = canvas(13, 3.5)
    ax.text(0.01, 0.97, "Recipe", fontsize=19, weight="bold", va="top")
    steps = [("replays", "top teams,\nhourly", "#f4f6fa", MUTED), ("winner seats", "never a\nlosing action", "#f4f6fa", MUTED),
             ("SFT", "hourly mixture,\nhours", "#eef4fc", BLUE), ("wash", "15 min, low LR,\ntop-3 wins, RAW", "#e8f6ef", AQUA),
             ("paired games", "vs live or\nprevious agent", "#fdeee7", ORANGE), ("ship", "exact bytes,\nsha-checked", "#fff7e0", GOLD)]
    w, gap, y, h = 0.127, 0.04, 0.3, 0.44
    for i, (t, sub, fc, ec) in enumerate(steps):
        x = 0.01 + i * (w + gap)
        rbox(ax, x, y, w, h, fc, ec, lw=1.8, r=0.02)
        ax.text(x + w / 2, y + h * 0.7, t, ha="center", va="center", fontsize=14.5, weight="bold")
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=11.5, color=INK2)
        if i:
            arrow(ax, (x - gap + 0.006, y + h / 2), (x - 0.006, y + h / 2), lw=2)
    xs = lambda i: 0.01 + i * (w + gap)  # noqa: E731
    ax.text(xs(5) - gap / 2, y + h / 2 + 0.07, "won", ha="center", fontsize=11, color=AQUA, weight="bold")
    arrow(ax, (xs(4) + w / 2, y - 0.01), (xs(2) + w / 2, y - 0.01), color=ORANGE, lw=1.6, ls="--", rad=-0.25)
    ax.text(xs(3) + w / 2, 0.0, "lost or tied: keep the live one, train more", ha="center", fontsize=11.5, color=ORANGE)
    save(fig, "fig5-recipe")


# 6 ---------------------------------------------------------------------------------------------------------------
def surprise_vs_games():
    # N flow checkpoints on the last-day pools (lr 3e-4; pool changed every ~20 min, 404-seat 80/14/6 pool from u57900).
    # Surprise: perplexity on 3 unseen #1-team wins (SURPRISE-3; scorer outputs sw-A/B/C, surp-live2 per_game).
    ppl_pool = [(56574, 3.92), (56952, 6.83), (57204, 9.28), (57519, 12.95), (57708, 11.59), (58023, 9.84)]
    ppl_general, ppl_wash = 2.64, 2.35
    # Games vs the general model (u56259): ledger R-VALIDATE-NF (final), 32 worlds x 2 seats.
    games = [(56511, .891, .791, .990), (56952, .828, .710, .946), (57204, .828, .730, .926), (57582, .734, .597, .872),
             (57708, .719, .590, .848), (58275, .844, .737, .951)]
    fig = plt.figure(figsize=(15, 4.8))
    fig.text(0.01, 1.0, "Surprise up 4-5x. Games stayed well above even. (N flow, last-day pools)", fontsize=19, weight="bold", va="top")
    a1 = fig.add_axes([0.05, 0.12, 0.27, 0.7])
    a2 = fig.add_axes([0.39, 0.12, 0.27, 0.7])
    a3 = fig.add_axes([0.77, 0.12, 0.22, 0.7])
    a1.plot([u - 56259 for u, _ in ppl_pool], [v for _, v in ppl_pool], "-o", color=ORANGE, lw=2.4)
    a1.axhline(ppl_general, color=INK2, ls="--", lw=1.2)
    a1.text(1800, ppl_general + 0.35, "before the pool 2.64", fontsize=10.5, color=INK2, ha="right")
    a1.axhline(ppl_wash, color=BLUE, ls=":", lw=1.6)
    a1.text(1800, ppl_wash - 0.95, "shipped 15-min wash 2.35", fontsize=10.5, color=BLUE, ha="right")
    a1.set_ylim(1, 14)
    a1.set_xlim(0, 2100)
    a1.set_title("surprise on 3 unseen #1 wins", loc="left", weight="bold", fontsize=13)
    a1.set_ylabel("perplexity")
    a1.set_xlabel("updates on the pool")
    gx = [u - 56259 for u, *_ in games]
    a2.errorbar(gx, [g[1] for g in games], yerr=[[g[1] - g[2] for g in games], [g[3] - g[1] for g in games]],
                fmt="o-", color=PURPLE, lw=2.2, capsize=4)
    a2.axhline(0.5, color=INK2, ls="--", lw=1.2)
    a2.text(2050, 0.52, "even", fontsize=10.5, color=INK2, ha="right")
    a2.set_ylim(0.4, 1.02)
    a2.set_xlim(0, 2100)
    a2.set_title("share of games vs the model before the pool", loc="left", weight="bold", fontsize=13)
    a2.set_xlabel("updates on the pool (32 worlds x 2 seats, 95% CI)")
    bars = [("pool ckpt\n(+945 upd.)", 31, ORANGE), ("shipped\nwash", 28, BLUE)]
    for i, (lab, wv, c) in enumerate(bars):
        a3.bar(i, wv, color=c, width=0.6)
        a3.text(i, wv + 0.5, f"{wv}-{32 - wv}", ha="center", fontsize=12, weight="bold")
    a3.set_xticks([0, 1])
    a3.set_xticklabels([b[0] for b in bars], fontsize=11)
    a3.set_ylim(0, 35)
    a3.set_title("vs live N flow, same 16 worlds", loc="left", weight="bold", fontsize=13)
    fig.text(0.01, -0.06, "Shipped the wash on surprise; games say the pool checkpoint was at least as good. Pool LR 3e-4, wash 3e-5.",
             fontsize=12.5, color=INK, weight="bold")
    a3.set_ylabel("games won of 32")
    a3.grid(axis="x", visible=False)
    save(fig, "fig6-surprise-vs-games")


# 7 ---------------------------------------------------------------------------------------------------------------
def paired():
    sp = json.load(open(OUT / "b-vs-p1-spread.json"))
    fig = plt.figure(figsize=(14, 4.6))
    fig.text(0.01, 1.0, "Deciding: paired games", fontsize=19, weight="bold", va="top")
    ax = fig.add_axes([0.0, 0.02, 0.3, 0.8])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (y, a, b) in enumerate([(0.6, "new", "old"), (0.28, "old", "new")]):
        rbox(ax, 0.05, y, 0.9, 0.25, "#f4f6fa", MUTED, lw=1.2, r=0.03)
        ax.text(0.5, y + 0.19, f"same world seed, game {i + 1}", ha="center", fontsize=11, color=INK2)
        ax.text(0.28, y + 0.07, f"seat 0: {a}", ha="center", fontsize=13, weight="bold", color=BLUE if a == "new" else INK2)
        ax.text(0.72, y + 0.07, f"seat 1: {b}", ha="center", fontsize=13, weight="bold", color=BLUE if b == "new" else INK2)
    ax.text(0.5, 0.14, "same shops, weeds, market rules", ha="center", fontsize=11.5, color=INK2)
    ax.text(0.5, 0.04, "CI clustered by world", ha="center", fontsize=11.5, color=INK2)
    ax2 = fig.add_axes([0.5, 0.14, 0.44, 0.68])
    rows = [("+40 updates", 0.594, 0.461, 0.727), ("+120 updates", 0.922, 0.841, 1.003), ("+200 updates", 0.906, 0.835, 0.978)]
    for i, (lab, sh, lo, hi) in enumerate(rows):
        ax2.barh(i, sh, color=BLUE, height=0.6)
        ax2.plot([lo, min(hi, 1.0)], [i, i], color=INK, lw=1.4)
        ax2.text(1.02, i, f"{sh:.2f}", va="center", fontsize=12)
    ax2.axvline(0.5, color=INK2, ls="--", lw=1.2)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([r[0] for r in rows], fontsize=11.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel("share of games won, 32 worlds x 2 seats; line = 95% CI")
    ax2.set_title("B-even on the last-day pool vs itself before the pool", loc="left", fontsize=13, weight="bold")
    ax2.grid(axis="y", visible=False)
    save(fig, "fig7-paired")


# 8 ---------------------------------------------------------------------------------------------------------------
def failed():
    # WHAT-HELPS.md 'Pre-upload freshens (09-30)': B31 + 666 updates on the last 2 days, tier mix fell back to 0/40/60.
    fig = plt.figure(figsize=(11, 3.4))
    fig.text(0.01, 1.0, "Failed case: fresh-only fine-tune", fontsize=19, weight="bold", va="top")
    a1 = fig.add_axes([0.05, 0.12, 0.38, 0.62])
    a2 = fig.add_axes([0.58, 0.12, 0.38, 0.62])
    tiers = ["tier 1 (older days)", "tier 2", "tier 3 (newest)"]
    a1.barh([0, 1, 2], [0, 40, 60], color=[RED, BLUE, BLUE], height=0.55)
    a1.text(1, 0, "0%  (silently empty)", va="center", color=RED, fontsize=12, weight="bold")
    for i, v in [(1, 40), (2, 60)]:
        a1.text(v + 1, i, f"{v}%", va="center", fontsize=12)
    a1.set_yticks([0, 1, 2])
    a1.set_yticklabels(tiers, fontsize=12)
    a1.invert_yaxis()
    a1.set_xlim(0, 70)
    a1.set_title("tier mix it actually trained on", loc="left", fontsize=13, weight="bold")
    a1.grid(axis="y", visible=False)
    a2.bar([0, 1], [26, 38], color=[RED, "#cfcdc6"], width=0.55)
    for i, v in enumerate([26, 38]):
        a2.text(i, v + 0.8, str(v), ha="center", fontsize=13, weight="bold")
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["B31 + 666 updates", "B31 (parent)"], fontsize=12)
    a2.set_ylim(0, 45)
    a2.set_title("games won, 32 worlds x 2 seats", loc="left", fontsize=13, weight="bold")
    a2.text(0.5, 42, "-\\$802 per world", ha="center", fontsize=12, color=RED)
    fig.text(0.05, -0.1, "Fix: an empty tier now refuses to build.", fontsize=12.5, color=INK, weight="bold")
    a2.grid(axis="x", visible=False)
    save(fig, "fig8-failed")


# A/B: resources ----------------------------------------------------------------------------------------------------
def resources():
    # SLURM accounting 1 Sep - 1 Oct, farmNEAT jobs only, allocated GPU-hours (sacct ElapsedRaw of jobs holding a GPU).
    machines = [("4090 (gpu-4090b)", 193, 21), ("4090 (desktop)", 189, 20), ("DGX Spark", 19, 32), ("3090", 7, 0), ("laptop 4060", 0, 1)]
    # Logged training seconds of the two final lineages (metrics.jsonl / train.log; SLURM elapsed for B-even 4090 jobs).
    fig = plt.figure(figsize=(14, 4.6))
    fig.text(0.01, 1.0, "Resources", fontsize=19, weight="bold", va="top")
    a1 = fig.add_axes([0.13, 0.14, 0.33, 0.66])
    a2 = fig.add_axes([0.66, 0.14, 0.32, 0.66])
    for i, (m, tr, ev) in enumerate(machines):
        a1.barh(i, tr, color=BLUE, height=0.6)
        a1.barh(i, ev, left=tr, color="#cfcdc6", height=0.6)
        a1.text(tr + ev + 4, i, f"{tr + ev:.0f} h", va="center", fontsize=11)
    a1.set_yticks(range(len(machines)))
    a1.set_yticklabels([m[0] for m in machines], fontsize=11.5)
    a1.invert_yaxis()
    a1.set_xlim(0, 260)
    a1.set_xlabel("GPU-hours allocated, 1-30 Sep (blue: training, grey: eval games)")
    a1.set_title("Local GPUs: ~7.9 days of both 4090s training", loc="left", fontsize=13, weight="bold")
    a1.grid(axis="y", visible=False)
    segs = [("Model 1\nB-even", [(29.54, BLUE, "B from scratch, 4090"), (9.21, "#6aa6e8", "B-even, 4090"), (2.11, GOLD, "G4*")]),
            ("Model 2\nN flow", [(11.34, AQUA, "random init, G4"), (0.23, "#11825a", "")])]
    for i, (lab, parts) in enumerate(segs):
        left = 0
        for v, c, t in parts:
            a2.barh(i, v, left=left, color=c, height=0.55)
            left += v
        a2.text(left + 0.8, i, f"{left:.1f} h" + ("+" if i == 0 else ""), va="center", fontsize=11.5, weight="bold")
    a2.set_yticks([0, 1])
    a2.set_yticklabels([s[0] for s in segs], fontsize=11.5)
    a2.invert_yaxis()
    a2.set_xlim(0, 52)
    a2.set_xlabel("logged training hours behind each final model")
    a2.set_title("Final models: logged portions", loc="left", fontsize=13, weight="bold")
    a2.grid(axis="y", visible=False)
    leg = [(BLUE, "B from scratch, one 4090 (29.5 h)"), ("#6aa6e8", "B-even, one 4090 (9.2 h)"),
           (GOLD, "B-even, G4: logged part only (2.1 h)"), (AQUA, "N flow from random init, one G4 (11.3 h)"),
           ("#11825a", "15-min wash")]
    for j, (c, t) in enumerate(leg):
        fig.text(0.66, -0.02 - j * 0.05, "\u25A0 ", color=c, fontsize=13, va="center")
        fig.text(0.675, -0.02 - j * 0.05, t, color=INK2, fontsize=10.5, va="center")
    fig.text(0.13, -0.08, "G4 = Colab RTX PRO 6000 96 GB, 48 vCPU. Colab hours are not in SLURM.", fontsize=9.8, color=MUTED)
    save(fig, "figB-resources")


# C: games -> SFT ---------------------------------------------------------------------------------------------------
def games_to_sft():
    fig, ax = canvas(14, 3.9)
    ax.text(0.01, 0.97, "Translating games to SFT", fontsize=19, weight="bold", va="top")
    steps = [("replay", "one game,\n2 seats x 719 steps", "#f4f6fa", MUTED),
             ("winner's seat", "the loser's actions\nare never targets", "#f4f6fa", MUTED),
             ("one step", "state it saw ->\nactions it sent", "#eef4fc", BLUE),
             ("token targets", "workers, orders,\nSTOP", "#fdeee7", ORANGE),
             ("windows", "24 consecutive\nsteps", "#e8f6ef", AQUA),
             ("update", "128 windows (4090)\n1,024 (G4)", "#fff7e0", GOLD)]
    w, gap, y, h = 0.135, 0.03, 0.28, 0.46
    for i, (t, sub, fc, ec) in enumerate(steps):
        x = 0.01 + i * (w + gap)
        rbox(ax, x, y, w, h, fc, ec, lw=1.8, r=0.02)
        ax.text(x + w / 2, y + h * 0.72, t, ha="center", va="center", fontsize=14, weight="bold")
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=11, color=INK2)
        if i:
            arrow(ax, (x - gap + 0.005, y + h / 2), (x - 0.005, y + h / 2), lw=2)
    ax.text(0.01, 0.06, "Parity check: 4 hosted games re-encoded, 2,876 steps, 0 differences across 16 encoder inputs.",
            fontsize=12, color=INK, va="center")
    save(fig, "figC-games-to-sft")


# D: features -------------------------------------------------------------------------------------------------------
def features():
    # Feature-drop at fixed weights, chunk-9 EMA checkpoints, 59 seats / 7,021 teacher-forced steps, 95% seat-based CI.
    rows = [("crop / timing / resource-gap features", 0.2782, 0.0382, 0.4666, 0.0426),
            ("spatial history (2 previous obs.)", 0.2546, 0.0273, 0.2365, 0.0317),
            ("cash ladder (cash after earlier orders)", 0.0219, 0.0073, 0.0183, 0.0059),
            ("order prefix", 0.0078, 0.0032, 0.0078, 0.0033),
            ("worker prefix", -0.0399, 0.0204, -0.0274, 0.0176)]
    fig = plt.figure(figsize=(14, 4.4))
    fig.text(0.01, 1.0, "Features: what breaks when you take one away", fontsize=19, weight="bold", va="top")
    fig.text(0.01, 0.9, "Earlier B/C chunk-9 EMA checkpoints; C was the recurrence-in-depth experiment.", fontsize=12, color=INK2, va="top")
    ax = fig.add_axes([0.3, 0.14, 0.62, 0.68])
    for i, (lab, b, be, c, ce) in enumerate(rows):
        ax.barh(i - 0.17, b, xerr=be, color=BLUE, height=0.32, capsize=3, error_kw=dict(lw=1, ecolor=INK))
        ax.barh(i + 0.17, c, xerr=ce, color=ORANGE, height=0.32, capsize=3, error_kw=dict(lw=1, ecolor=INK))
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlabel("loss increase, bits per step, when the group is zeroed (fixed weights; 59 seats, 7,021 steps; 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.text(0.36, 3.6, "\u25A0 B (54.6M lineage)", color=BLUE, fontsize=12, weight="bold")
    ax.text(0.36, 4.1, "\u25A0 C (looped sibling)", color=ORANGE, fontsize=12, weight="bold")
    ax.text(0.01, 4.0, "removing it helped", fontsize=10, color=MUTED, ha="left", va="center")
    save(fig, "figD-features")


FIGS = {"fig2-decision": decision, "fig3-models": models, "fig4-data": data, "fig5-recipe": recipe,
        "fig6-surprise-vs-games": surprise_vs_games, "fig7-paired": paired, "fig8-failed": failed,
        "figB-resources": resources, "figC-games-to-sft": games_to_sft, "figD-features": features}


# I: feature table -------------------------------------------------------------------------------------------------------
def feature_table():
    # Sources: B-even checkpoint config + primitive-bc model.py/data.py; N flow nf_model.py/nf_format.py (node columns
    # = rows 80 | extra 152 | v6_token 8 | x8_token 354 | knownness 80; per-worker unit_extra + x8_unit; prefix; order_cash).
    BOTH, M1, M2, NONE = ("both", "#8e5cc4"), ("Model 1 only", BLUE), ("Model 2 only", AQUA), ("neither (tried)", "#b9b7b0")
    rows = [
        ("entity tokens", "one token per tile, worker, stock item, market, shop, constant, farm, clock; 80 values + known flags", BOTH),
        ("no-action forecasts", "what each tile and animal does if nobody acts: weeds, decay, escapes, yields; duty counts", BOTH),
        ("market calculations", "projected end inventory, clearing price of the next unit, value of one more producer", BOTH),
        ("job routes and ETA", "nine-job route facts, resource ETAs, job deadlines", BOTH),
        ("crop / timing / resource gaps", "354 values per token and 41 per worker", BOTH),
        ("worker extras", "12 values per worker", BOTH),
        ("reach bias", "decoder attention bias from each worker's ETA to each own tile", BOTH),
        ("same-step prefix", "own state after the earlier workers' commands this step; orders so far", BOTH),
        ("cash ladder", "cash left after the earlier orders this step", BOTH),
        ("own last 2 turns", "previous commands and orders", BOTH),
        ("spatial history lattice", "both farms as 10x10 grids at t, t-1, t-2 (120 channels), through a CNN", M1),
        ("typed relation graph", "17 relation types: adjacency, mirror tile, custody, market, shop, resource, calendar ...", M2),
        ("calendar landmarks", "now, next shop tick, dusk, +2d, +4d, +8d, +12d, end of game", M2),
        ("game memory", "fast, slow and per-worker GRU state carried through the game", M2),
        ("theorem features", "7,564 values per step derived from the macro theorems", NONE),
        ("long-lag history", "clock and item history 1, 4, 24 and 72 steps back", NONE),
        ("intent tokens", "an unsupervised label on every tile before acting (chain of thought)", NONE),
    ]
    order = {M1[0]: 0, M2[0]: 1, NONE[0]: 2, BOTH[0]: 3}
    rows = sorted(rows, key=lambda r: order[r[2][0]])
    n = len(rows)
    fig, ax = canvas(14, 1.3 + 0.42 * n)
    H = 1.3 + 0.42 * n
    yy = lambda i: 1 - (1.15 + 0.42 * (i + 0.5)) / H  # noqa: E731
    ax.text(0.01, 1 - 0.25 / H, "Features", fontsize=19, weight="bold", va="top")
    for j, (lab, c) in enumerate([M1, M2, NONE, BOTH]):
        x = 0.42 + j * 0.145
        ax.add_patch(FancyBboxPatch((x, 1 - 0.78 / H), 0.016, 0.22 / H, boxstyle="round,pad=0,rounding_size=0.004", fc=c, ec="none"))
        ax.text(x + 0.022, 1 - 0.67 / H, lab, fontsize=11, va="center", color=INK)
    for i, (name, desc, (lab, c)) in enumerate(rows):
        y = yy(i)
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.005, y - 0.21 / H), 0.99, 0.42 / H, fc="#f6f7f9", ec="none"))
        ax.add_patch(FancyBboxPatch((0.012, y - 0.13 / H), 0.012, 0.26 / H, boxstyle="round,pad=0,rounding_size=0.003", fc=c, ec="none"))
        ax.text(0.034, y, name, fontsize=11.5, weight="bold", va="center", color=INK if lab != NONE[0] else MUTED)
        ax.text(0.27, y, desc, fontsize=10.8, va="center", color=INK2 if lab != NONE[0] else MUTED)
    save(fig, "figI-feature-table")


FIGS["figI-feature-table"] = feature_table


if __name__ == "__main__":
    for n in (sys.argv[1:] or FIGS):
        FIGS[n]()
