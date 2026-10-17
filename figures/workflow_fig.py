"""Idea / fallback / best / experiment workflow diagram. Run with /usr/bin/python3 -s -B."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2, MUTED, GRID = "#16202c", "#52514e", "#8a8984", "#e6e5e0"
GREY, BLUE, ORANGE, AQUA = "#cfcdc6", "#2a78d6", "#eb6834", "#1baf7a"
fig = plt.figure(figsize=(14, 6.2))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.6, ls="-", r=0.015):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, linestyle=ls))


def arrow(a, b, color=INK2, rad=0.0, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


ax.text(0.01, 0.97, "The workflow: tree search over ideas", fontsize=19, weight="bold", va="top", color=INK)
ax.text(0.01, 0.9, "Keep 2-3 ideas alive. Each has a fallback, a current best and a current experiment.", fontsize=12.5, color=INK2, va="top")
ax.text(0.01, 0.03, "loses \u2192 try a new crazy experiment   |   beats the best \u2192 it becomes the best   |   experimented enough \u2192 the best becomes the fallback", fontsize=11.5, color=INK)
w, h = 0.155, 0.12
slots = [("current experiment", "#fdeee7", ORANGE, "--", 0.6), ("current best", "#e8f1fc", BLUE, "-", 0.42), ("fallback", "#f1f0ec", MUTED, "-", 0.24)]
for i, (name, sub) in enumerate([("Idea 1", "micro vs macro"), ("Idea 2", "architecture"), ("Idea 3", "data selection")]):
    x = 0.03 + i * 0.185
    ax.text(x + w / 2, 0.8, name, ha="center", fontsize=14, weight="bold", color=INK)
    ax.text(x + w / 2, 0.755, sub, ha="center", fontsize=11, color=INK2)
    for lab, fc, ec, ls, y in slots:
        box(x, y, w, h, fc, ec, ls=ls)
        ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=11.5, color=INK)
    if i == 0:
        arrow((x + w * 0.5, 0.6), (x + w * 0.5, 0.54), color=BLUE)
        ax.text(x + w * 0.53, 0.57, "beats best", fontsize=10, color=BLUE, va="center")
        arrow((x + w * 0.85, 0.42), (x + w * 0.85, 0.36), color=MUTED)
        ax.text(x + w * 0.88, 0.39, "experimented\nenough", fontsize=9.5, color=MUTED, va="center")
        arrow((x + w, 0.69), (x + w, 0.64), color=ORANGE, rad=-2.2)
# this competition's tree (right panel)
X = 0.6
box(X, 0.08, 0.39, 0.78, "#fafbfd", GRID, lw=1.2, r=0.02)
ax.text(X + 0.015, 0.84, "This competition's tree", fontsize=13.5, weight="bold", va="top", color=INK)
RED = "#c0392b"


def node(x, y, label, state, wd=0.1):
    fc, ec, col = {"alive": ("#e8f6ef", AQUA, INK), "dead": ("#f6f5f2", GREY, MUTED), "root": ("#ffffff", INK2, INK)}[state]
    ax.add_patch(FancyBboxPatch((x - wd / 2, y - 0.03), wd, 0.06, boxstyle="round,pad=0,rounding_size=0.01", fc=fc, ec=ec, lw=1.4, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=9.6, color=col, zorder=4)
    if state == "dead":
        ax.text(x + wd / 2, y + 0.03, "\u2715", ha="center", va="center", fontsize=11, color=RED, weight="bold", zorder=5)


def edge(a, b, dead=False, lw=1.4):
    ax.plot([a[0], b[0]], [a[1] - 0.03, b[1] + 0.03], color=GREY if dead else INK2, lw=lw, ls="--" if dead else "-", zorder=2)


root = (X + 0.195, 0.74)
node(*root, "how to decide", "root", 0.12)
macro, micro = (X + 0.085, 0.6), (X + 0.29, 0.6)
edge(root, macro, dead=True, lw=4)
edge(root, micro)
node(*macro, "macro + theorems", "dead", 0.13)
ax.text(macro[0], macro[1] - 0.05, "most of my time", ha="center", va="top", fontsize=9, color=MUTED)
node(*micro, "micro, one token\nat a time", "alive", 0.13)
kids = [("chain of\nthought", "dead"), ("recurrence\nin depth", "dead"), ("depth over\nwidth", "dead"), ("massive\nheads", "alive"), ("graph +\nmemory", "alive")]
for j, (lab, st) in enumerate(kids):
    k = (X + 0.04 + j * 0.078, 0.4)
    edge(micro, k, dead=(st == "dead"))
    node(*k, lab, st, 0.068)
HX, GX = X + 0.04 + 3 * 0.078, X + 0.04 + 4 * 0.078
node(HX, 0.25, "18 x 24\nheads", "alive", 0.068)
edge((HX, 0.4), (HX, 0.25))
ax.text(HX - 0.042, 0.25, "Model 1", ha="right", va="center", fontsize=10.5, weight="bold", color=BLUE)
ax.text(GX, 0.335, "Model 2", ha="center", va="top", fontsize=10.5, weight="bold", color=AQUA)
ax.text(X + 0.38, 0.17, "Model 2 is a small leaf: 7.25M, built on the last day.\nIt didn't beat B, but it beat my other alternatives.", ha="right", va="top", fontsize=9.3, color=INK2)
fig.savefig("figF-workflow.png", dpi=150, bbox_inches="tight", pad_inches=0.2, facecolor="white")
fig.savefig("figF-workflow.svg", bbox_inches="tight", pad_inches=0.2, facecolor="white")
print("wrote figF-workflow")
