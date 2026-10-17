"""Five figure options for each final model (Model 1 = B-even, Model 2 = N flow), stats shown mostly in the picture.

Run from the repo root: /usr/bin/python3 -s -B kagglewriteup/figures/final/model_options.py
Inputs (all in this folder): model-params.json (checkpoint parameter groups), model-series.json (training logs),
b-vs-p1-spread.json (16 identical games per B-even checkpoint vs P1). Other numbers are typed in with their source.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent / "options"
OUT.mkdir(exist_ok=True)
SRC = Path(__file__).resolve().parent
BLUE, BLUE2, BLUE3, ORANGE, AQUA, AQUA2, PURPLE, GOLD, RED = ("#2a78d6", "#6aa6e8", "#a9c9f0", "#eb6834", "#1baf7a",
                                                              "#7fd3b0", "#8e5cc4", "#c99a2e", "#c0392b")
INK, INK2, MUTED, GRID, PAPER = "#16202c", "#52514e", "#8a8984", "#e6e5e0", "#f6f7f9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": MUTED,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})
P = json.load(open(SRC / "model-params.json"))
S = json.load(open(SRC / "model-series.json"))
SP = json.load(open(SRC / "b-vs-p1-spread.json"))

M1 = dict(name="B-even", color=BLUE, params=54_639_683, updates=46_484, windows=6_834_048,
          hours=[("B from scratch, one 4090", 29.5, BLUE), ("B-even, one 4090", 9.2, BLUE2), ("B-even, G4 (logged part)", 2.1, GOLD)])
M2 = dict(name="N flow", color=AQUA, params=7_252_244, updates=56_559, hours=[("random init, one G4", 11.3, AQUA), ("15-min wash", 0.25, "#11825a")])


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fc, ec, lw=1.3, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(ax, a, b, color=INK2, lw=1.6, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def tile(ax, x, y, w, h, big, small, color):
    rbox(ax, x, y, w, h, PAPER, GRID, lw=1)
    ax.add_patch(Rectangle((x, y + 0.012), 0.006, h - 0.024, fc=color, ec="none"))
    ax.text(x + 0.022, y + h * 0.62, big, fontsize=21, weight="bold", va="center", color=INK)
    ax.text(x + 0.022, y + h * 0.25, small, fontsize=10.5, va="center", color=INK2)


def stack(ax, x, y0, w, rows, color_body, h=0.1, gap=0.035):
    """rows top->bottom: (label, fill). Returns y of each row."""
    ys = []
    for i, (lab, fc) in enumerate(rows):
        y = y0 - i * (h + gap)
        rbox(ax, x, y, w, h, fc, MUTED, lw=1)
        ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=11)
        if i:
            arrow(ax, (x + w / 2, y + h + gap - 0.003), (x + w / 2, y + h + 0.003))
        ys.append(y)
    return ys


# ------------------------------------------------------------------ A: spec sheet ----------------------------------
def spec(m, rows, tiles, title, sub):
    fig, ax = canvas(13.5, 4.3)
    ax.text(0.01, 0.97, title, fontsize=19, weight="bold", va="top")
    ax.text(0.01, 0.86, sub, fontsize=12, color=INK2, va="top")
    stack(ax, 0.02, 0.6, 0.4, rows, m["color"], h=0.115, gap=0.045)
    tw, th = 0.165, 0.26
    for i, (big, small, c) in enumerate(tiles):
        x = 0.47 + (i % 3) * (tw + 0.012)
        y = 0.42 - (i // 3) * (th + 0.05)
        tile(ax, x, y, tw, th, big, small, c)
    return fig


def m1_a():
    windows = M1["windows"]
    fig = spec(M1, [("entity tokens + history of 2 previous observations", "#ffffff"),
                    ("encoder: 10 blocks, width 432, 18 heads x 24", "#dde9f9"),
                    ("decoder: 8 blocks, 12 heads x 36", "#dde9f9"),
                    ("25 worker slots → ≤10 orders → STOP", "#fdeee7")],
               [("54.6M", "parameters", BLUE), ("46,484", "updates (31,047 as B)", BLUE),
                (f"{windows * 24 / 1e6:.0f}M", "sampled decision steps", BLUE),
                ("40.9 h+", "logged training", GOLD), ("45-19", "u5000 vs B31\n32 worlds x 2 seats", AQUA)],
               "Model 1: B-even", "A 54.6M transformer, warm-started from B, then SFT on winners only")
    save(fig, "m1-a-spec-sheet")


def m2_a():
    fig = spec(M2, [("320 graph nodes: 296 entity slots, memory, calendar", "#ffffff"),
                    ("4 relational blocks, width 256, 17 relation types", "#dcf3e9"),
                    ("2 causal decoder blocks", "#dcf3e9"),
                    ("25 worker slots → ≤10 orders → STOP", "#fdeee7")],
               [("7.25M", "parameters", AQUA), ("56,559", "updates from random init", AQUA),
                ("11.3 h", "logged training, one G4", AQUA), ("4.7 h", "init to first submission", GOLD),
                ("28-4", "vs the previous N flow", BLUE), ("320", "graph nodes per step", BLUE)],
               "Model 2: N flow", "A 7.25M graph + memory model, trained from random weights on the last day")
    save(fig, "m2-a-spec-sheet")


# ------------------------------------------------------------------ B: parameter map (waffle) -----------------------
def waffle(m, key, unit, order, colors, title, note):
    g = P[key]["groups"]
    cells = []
    for name in order:
        n = round(g[name] / unit)
        cells += [name] * n
    cols = 30
    rows = (len(cells) + cols - 1) // cols
    fig = plt.figure(figsize=(13.5, 5.8))
    fig.text(0.01, 0.99, title, fontsize=19, weight="bold", va="top")
    fig.text(0.01, 0.925, note, fontsize=12, color=INK2, va="top")
    ax = fig.add_axes([0.01, 0.04, 0.5, 0.82])
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.axis("off")
    ax.set_aspect("equal")
    for i, c in enumerate(cells):
        ax.add_patch(Rectangle((i % cols + 0.08, i // cols + 0.08), 0.84, 0.84, fc=colors[c], ec="none"))
    y = 0.8
    for name in order:
        v = g[name]
        fig.text(0.56, y, "■", color=colors[name], fontsize=20, va="center")
        fig.text(0.59, y, name, fontsize=13, va="center", weight="bold")
        fig.text(0.84, y, f"{v / 1e6:.2f}M", fontsize=13, va="center", ha="right")
        fig.text(0.9, y, f"{100 * v / P[key]['total']:.1f}%", fontsize=12.5, va="center", ha="right", color=INK2)
        y -= 0.095
    return fig


def m1_b():
    order = ["encoder blocks", "decoder blocks", "auxiliary heads", "state inputs", "history (previous obs.)", "output heads", "prefix features"]
    colors = dict(zip(order, [BLUE, "#1b4f99", PURPLE, GOLD, ORANGE, AQUA, RED]))
    fig = waffle(M1, "beven", 100_000, order, colors, "Model 1: where the 54.6M parameters sit",
                 "One square = 100k parameters. Encoder and decoder are nearly equal; everything else is 15%.")
    save(fig, "m1-b-parameter-map")


def m2_b():
    order = ["relational blocks", "decoder blocks", "memory (GRUs)", "state inputs", "auxiliary heads", "output heads"]
    colors = dict(zip(order, ["#11825a", AQUA, GOLD, BLUE, PURPLE, ORANGE]))
    fig = waffle(M2, "nflow", 10_000, order, colors, "Model 2: where the 7.25M parameters sit",
                 "One square = 10k parameters. One seventh of the model is memory.")
    save(fig, "m2-b-parameter-map")


# ------------------------------------------------------------------ C: training timeline ----------------------------
def m1_c():
    fig = plt.figure(figsize=(13.5, 5.6))
    fig.text(0.01, 0.99, "Model 1: 46,484 updates, three phases", fontsize=19, weight="bold", va="top")
    fig.text(0.01, 0.925, "Training loss per phase (log scale; the data changes between phases, so levels are not comparable across them).",
             fontsize=12, color=INK2, va="top")
    ax = fig.add_axes([0.06, 0.13, 0.9, 0.66])
    for key, c, lab in [("b_scratch", BLUE, "B from scratch, one 4090, 29.5 h"), ("beven_4090", BLUE2, "B-even, one 4090, 9.2 h"),
                        ("beven_g4_last", GOLD, "B-even, G4 (last segment), 2.1 h")]:
        rows = [r for r in S[key] if r[0] <= 46484]
        ax.plot([r[0] for r in rows], [r[1] for r in rows], color=c, lw=1.3, label=lab)
    ax.set_yscale("log")
    ax.set_xlim(0, 48000)
    ax.set_ylim(0.3, 70)
    ax.axvspan(42010, 45930, color=GRID, alpha=0.6, lw=0)
    ax.text(43970, 4, "G4,\nnot\nlogged", ha="center", fontsize=10, color=MUTED)
    marks = [(6200, "recipe change"), (31047, "heads regrouped\n36x12 → 18x24"), (35000, "u5000 screen\n45-19 vs B31"),
             (46484, "last-day pool,\nthen shipped")]
    for x, t in marks:
        ax.axvline(x, color=INK2, lw=1, ls=":")
        ax.text(x - 300, 0.36, t, fontsize=10, ha="right", va="bottom", color=INK)
    ax.set_xlabel("updates")
    ax.set_ylabel("training loss")
    ax.legend(loc="upper center", fontsize=10.5)
    save(fig, "m1-c-timeline")


def m2_c():
    rows = S["nflow"]
    t, prev, hrs = 0.0, 0, []
    for r in rows:
        t += (r[0] - prev) * (r[3] or 0)
        prev = r[0]
        hrs.append(t / 3600)
    fig = plt.figure(figsize=(13.5, 5.6))
    fig.text(0.01, 0.99, "Model 2: random weights to shipped in one day", fontsize=19, weight="bold", va="top")
    fig.text(0.01, 0.925, "Training loss and exact-step accuracy against logged hours on one G4. The data mix changed several times, so the curves jump.", fontsize=12, color=INK2, va="top")
    ax = fig.add_axes([0.06, 0.13, 0.86, 0.66])
    gen = [(h, r) for h, r in zip(hrs, rows) if r[0] <= 56259]
    pool = [(h, r) for h, r in zip(hrs, rows) if r[0] > 56259]
    ax.plot([h for h, _ in gen], [r[1] for _, r in gen], color=AQUA, lw=1.2, label="training loss")
    ax.plot([h for h, _ in pool], [r[1] for _, r in pool], color=MUTED, lw=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("training loss")
    ax.set_xlabel("logged training hours")
    a2 = ax.twinx()
    a2.plot([h for h, _ in gen], [r[2] for _, r in gen], color=PURPLE, lw=1.2, alpha=0.8, label="exact-step accuracy")
    a2.set_ylim(0, 1)
    a2.set_ylabel("exact-step accuracy (training batch)", color=PURPLE)
    a2.grid(False)
    a2.spines["right"].set_visible(True)
    h_end = gen[-1][0]
    ax.axvspan(h_end, hrs[-1], color=GRID, alpha=0.7, lw=0)
    ax.text((h_end + hrs[-1]) / 2, 0.065, "2 h on a tiny pool:\nnot shipped", ha="center", fontsize=10, color=MUTED)
    for x, lab in [(2.72, "16,000 updates:\nfirst submission"), (h_end, "56,259 updates\n+ 15-min wash:\nshipped")]:
        ax.axvline(x, color=INK2, lw=1, ls=":")
        ax.text(x + 0.12, 3.5, lab, fontsize=10.5, va="top")
    ax.legend(loc="lower left", fontsize=10.5)
    a2.legend(loc="lower center", fontsize=10.5)
    save(fig, "m2-c-timeline")


# ------------------------------------------------------------------ D: evidence dashboard ---------------------------
def m1_d():
    fig = plt.figure(figsize=(10, 4.8))
    fig.text(0.01, 1.0, "Model 1: the evidence", fontsize=19, weight="bold", va="top")
    a1 = fig.add_axes([0.05, 0.12, 0.38, 0.66])
    a1.bar([0, 1], [45, 19], color=[BLUE, "#cfcdc6"], width=0.6)
    for i, v in enumerate([45, 19]):
        a1.text(i, v + 1, str(v), ha="center", fontsize=14, weight="bold")
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["18 x 24 heads", "36 x 12 parent"])
    a1.set_ylim(0, 55)
    a1.set_title("games won, 32 worlds x 2 seats", loc="left", fontsize=12.5, weight="bold")
    a1.text(0.5, 51, "+$1,463 per world", ha="center", fontsize=11.5, color=BLUE)
    a1.grid(axis="x", visible=False)
    a3 = fig.add_axes([0.58, 0.12, 0.4, 0.66])
    sv = [("before", 27.88), ("+120", 25.20), ("+200", 24.42), ("+320", 25.34)]
    a3.bar(range(4), [v for _, v in sv], color=[BLUE3, BLUE, BLUE, BLUE], width=0.6)
    for i, (_, v) in enumerate(sv):
        a3.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=11)
    a3.set_xticks(range(4))
    a3.set_xticklabels([s for s, _ in sv])
    a3.set_ylim(20, 30)
    a3.set_title("perplexity, 4 unseen #1 wins", loc="left", fontsize=12.5, weight="bold")
    a3.set_xlabel("lower = predicts #1 better")
    a3.grid(axis="x", visible=False)
    save(fig, "m1-d-evidence")


def m2_d():
    fig = plt.figure(figsize=(10, 4.8))
    fig.text(0.01, 1.0, "Model 2: the evidence", fontsize=19, weight="bold", va="top")
    a1 = fig.add_axes([0.05, 0.12, 0.33, 0.66])
    a1.bar([0, 1], [28, 4], color=[AQUA, "#cfcdc6"], width=0.6)
    for i, v in enumerate([28, 4]):
        a1.text(i, v + 0.6, str(v), ha="center", fontsize=14, weight="bold")
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["shipped N flow", "previous N flow"])
    a1.set_ylim(0, 34)
    a1.set_title("games won, 16 worlds x 2 seats", loc="left", fontsize=12.5, weight="bold")
    a1.grid(axis="x", visible=False)
    a3 = fig.add_axes([0.62, 0.12, 0.34, 0.66])
    rows = [("vs older\n18.9M model", 60, 4), ("vs P1\n(54.6M B)", 35, 29)]
    for i, (lab, w, l) in enumerate(rows):
        a3.barh(i, w, color=AQUA, height=0.5)
        a3.barh(i, l, left=w, color="#cfcdc6", height=0.5)
        a3.text(w / 2, i, str(w), ha="center", va="center", color="white", weight="bold", fontsize=12)
        a3.text(w + l / 2, i, str(l), ha="center", va="center", color=INK, fontsize=11)
    a3.set_yticks([0, 1])
    a3.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    a3.invert_yaxis()
    a3.set_xlim(0, 64)
    a3.set_title("first N flow (16k updates + top-up)", loc="left", fontsize=12.5, weight="bold")
    a3.set_xlabel("games, 32 worlds x 2 seats")
    a3.grid(axis="y", visible=False)
    save(fig, "m2-d-evidence")


# ------------------------------------------------------------------ E: signature mechanism ---------------------------
def m1_e():
    fig, ax = canvas(13.5, 5.2)
    ax.text(0.01, 0.97, "Model 1: same width, half the heads", fontsize=19, weight="bold", va="top")
    ax.text(0.01, 0.89, "Width 432 split into attention heads. B-even regrouped B's 36 narrow heads into 18 wider ones.",
            fontsize=12, color=INK2, va="top")
    for row, (n, w, c, lab) in enumerate([(36, 12, BLUE3, "B: 36 heads x 12"), (18, 24, BLUE, "B-even: 18 heads x 24")]):
        y = 0.6 - row * 0.26
        x0, W = 0.2, 0.55
        for i in range(n):
            ax.add_patch(Rectangle((x0 + i * W / n, y), W / n * 0.9, 0.14, fc=c, ec="none"))
        ax.text(x0 - 0.015, y + 0.07, lab, ha="right", va="center", fontsize=12.5, weight="bold")
    arrow(ax, (0.475, 0.585), (0.475, 0.5), lw=2)
    ax.text(0.49, 0.545, "1,000 updates of BC + distillation from B, then SFT", fontsize=11, va="center", color=INK2)
    rbox(ax, 0.8, 0.3, 0.18, 0.44, PAPER, GRID)
    ax.text(0.89, 0.66, "45-19", ha="center", fontsize=26, weight="bold", color=BLUE)
    ax.text(0.89, 0.57, "u5000 vs B31", ha="center", fontsize=10.5, color=INK2)
    ax.text(0.89, 0.47, "+$1,463", ha="center", fontsize=20, weight="bold", color=INK)
    ax.text(0.89, 0.39, "per world [+599, +2,326]", ha="center", fontsize=10, color=INK2)
    ax.text(0.89, 0.33, "32 worlds x 2 seats", ha="center", fontsize=10, color=MUTED)
    ax.text(0.2, 0.12, "Same 54,639,683 parameters either way.", fontsize=11.5, color=INK2)
    save(fig, "m1-e-signature")


def m2_e():
    fig, ax = canvas(13.5, 5.6)
    ax.text(0.01, 0.97, "Model 2: a graph that remembers", fontsize=19, weight="bold", va="top")
    ax.text(0.01, 0.89, "320 nodes linked by 17 relation types; three memories carry state through the game.", fontsize=12, color=INK2, va="top")
    # left: node budget
    nodes = [("entity slots", 296, BLUE), ("memory", 2, GOLD), ("calendar", 8, PURPLE), ("pad", 14, "#d8d6cf")]
    x0, y0, cols = 0.02, 0.8, 20
    i = 0
    for name, n, c in nodes:
        for _ in range(n):
            ax.add_patch(Rectangle((x0 + (i % cols) * 0.0145, y0 - (i // cols) * 0.04), 0.011, 0.03, fc=c, ec="none"))
            i += 1
    for j, (name, n, c) in enumerate(nodes):
        ax.text(0.02 + (j % 2) * 0.16, 0.075 - (j // 2) * 0.05, f"■ {name} {n}", color=c if c != "#d8d6cf" else MUTED, fontsize=10.5, weight="bold")
    ax.text(0.02, 0.16, "relations include tile adjacency, mirror tile, farm,\ncarrier-tile, custody, market, shop, resource, calendar, hub", fontsize=9.6, color=INK2, va="center")
    # right: memory timeline over 3 days
    X0, X1 = 0.47, 0.98
    days = 3
    turns = 24 * days
    xt = lambda k: X0 + (X1 - X0) * k / turns  # noqa: E731
    lanes = [("fast GRU 128", 0.7, AQUA), ("slow GRU 256", 0.55, GOLD), ("farmer", 0.4, BLUE), ("hand", 0.28, BLUE2)]
    for lab, y, c in lanes:
        ax.text(X0 - 0.01, y, lab, ha="right", va="center", fontsize=11, weight="bold", color=c)
        ax.plot([X0, X1], [y, y], color=GRID, lw=1)
    for k in range(turns):
        ax.plot([xt(k), xt(k)], [0.68, 0.72], color=AQUA, lw=1)
    for d in range(days):
        ax.scatter([xt(24 * d)], [0.55], s=70, color=GOLD, zorder=3)
        ax.axvline(xt(24 * d), ymin=0.2, ymax=0.78, color=MUTED, lw=0.8, ls=":")
        ax.text(xt(24 * d) + 0.004, 0.79, f"dawn, day {d + 1}", fontsize=9.5, color=MUTED)
        ax.plot([xt(24 * d + 1), xt(24 * d + 20)], [0.28, 0.28], color=BLUE2, lw=4, solid_capstyle="butt")
    ax.plot([X0, X1], [0.4, 0.4], color=BLUE, lw=4)
    ax.text(X0, 0.17, "fast: every turn  |  slow: once per day, from the day's summary\nhands reset at dawn; the farmer never does", fontsize=10.5, color=INK2, va="top")
    save(fig, "m2-e-signature")


if __name__ == "__main__":
    for f in (m1_a, m1_b, m1_c, m1_d, m1_e, m2_a, m2_b, m2_c, m2_d, m2_e):
        f()
