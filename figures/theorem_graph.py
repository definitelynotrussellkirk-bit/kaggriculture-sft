"""All 311 THEOREMS.md claims as a layered link graph, with 12 seeded-random readable samples. Run from the repo root
with /usr/bin/python3 -s -B kagglewriteup/figures/final/theorem_graph.py. Sources: review/THEOREMS.md (layers, forms,
titles) and project/registries/search-ledger/*.json (depends_on + relations)."""
import glob, json, random, re, textwrap
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FORM = {"STRUCTURE": "#1baf7a", "REPRESENTATION": "#2a78d6", "BOUNDARY": "#eb6834", "MEASUREMENT": "#8e5cc4"}
INK, INK2, MUTED = "#16202c", "#52514e", "#8a8984"

lvl, th = None, []
for l in open(ROOT / "review/THEOREMS.md"):
    if l.startswith("## "):
        lvl = l[3:].strip()
    m = re.match(r"- \*\*(.+?)\*\* (?:\[(\w+)\] )?(.*)", l)
    if m:
        th.append(dict(key=m.group(1), layer=lvl, form=m.group(2) or "", title=m.group(3).split(" — the model need not learn")[0].strip()))
keys = {t["key"] for t in th}
recs = {f.stem: json.load(open(f)) for f in (ROOT / "project/registries/search-ledger").glob("*.json")}
edges = set()
for d in recs.values():
    k = d.get("key")
    if k not in keys:
        continue
    for t in (d.get("depends_on") or []) + [r.get("target") for r in (d.get("relations") or [])]:
        tk = recs.get(t, {}).get("key")
        if tk in keys and tk != k:
            edges.add(tuple(sorted((k, tk))))
layers = []
for t in th:
    if t["layer"] not in layers:
        layers.append(t["layer"])
deg = Counter()
nbr = defaultdict(set)
for a, b in edges:
    deg[a] += 1; deg[b] += 1; nbr[a].add(b); nbr[b].add(a)
col = {t["key"]: layers.index(t["layer"]) for t in th}
bycol = defaultdict(list)
for t in sorted(th, key=lambda t: -deg[t["key"]]):
    bycol[col[t["key"]]].append(t["key"])
ypos = {}
for c, ks in bycol.items():
    for i, k in enumerate(ks):
        ypos[k] = (i + 0.5) / len(ks)
for _ in range(30):  # barycenter ordering to shorten links
    for c, ks in bycol.items():
        ks.sort(key=lambda k: np.mean([ypos[n] for n in nbr[k]]) if nbr[k] else ypos[k])
        for i, k in enumerate(ks):
            ypos[k] = (i + 0.5) / len(ks)
rng = random.Random(11)
ok = lambda t: len(t["title"]) <= 95 and not re.search(r"[0-9a-f]{8}|q[0-9]|SELL[0-9]|H8|PASS gate|^\[", t["title"]) \
    and not t["title"].lower().startswith(("consensus", "boundary:", "representation:", "structure:"))
picks = []
for L, n in zip(layers, [1, 2, 2, 2, 2, 2, 1]):
    picks += rng.sample([t for t in th if t["layer"] == L and ok(t)], n)
pk = {t["key"]: i + 1 for i, t in enumerate(picks)}

fig = plt.figure(figsize=(16, 9))
fig.text(0.01, 0.985, "Abandoned: the macro approach, as theorems", fontsize=20, weight="bold", va="top", color=INK)
fig.text(0.01, 0.94, f"{len(th)} claims in {len(layers)} layers; {len(edges):,} linked pairs (depends on, corroborates, narrows, supersedes, refutes). Numbered: random readable samples.",
         fontsize=12, color=INK2, va="top")
ax = fig.add_axes([0.01, 0.04, 0.5, 0.82]); ax.axis("off"); ax.set_xlim(-0.6, len(layers) - 0.4); ax.set_ylim(-0.04, 1.08)
X = lambda k: col[k] + 0.18 * np.sin(7.3 * ypos[k] * len(bycol[col[k]]))  # small jitter so dense columns read as clouds
for a, b in edges:
    hot = a in pk or b in pk
    ax.plot([X(a), X(b)], [ypos[a], ypos[b]], color=INK if hot else "#9aa3ae", alpha=0.4 if hot else 0.09, lw=0.8 if hot else 0.5, zorder=1)
for t in th:
    k = t["key"]
    ax.scatter(X(k), ypos[k], s=8 + 5 * deg[k], color=FORM.get(t["form"], "#b9b7b0"), edgecolor="white", lw=0.4, zorder=2)
for k, i in pk.items():
    ax.scatter(X(k), ypos[k], s=200, facecolor="white", edgecolor=INK, lw=1.6, zorder=3)
    ax.text(X(k), ypos[k], str(i), ha="center", va="center", fontsize=8.5, weight="bold", zorder=4)
short = ["Outcome", "Market", "Production", "Capacity\n& cash", "Custody\n& logistics", "Learning", "Unplaced"]
for c, L in enumerate(layers):
    ax.text(c, 1.05, short[c] if c < len(short) else L, ha="center", va="center", fontsize=10.5, color=INK2, weight="bold")
    ax.text(c, -0.03, f"{len(bycol[c])}", ha="center", va="center", fontsize=10, color=MUTED)
for j, (f, c) in enumerate(FORM.items()):
    fig.text(0.02 + j * 0.115, 0.012, "● " + f.lower(), color=c, fontsize=10.5, weight="bold")
y = 0.86
for i, t in enumerate(picks, 1):
    head = f"{i}. {short[layers.index(t['layer'])].replace(chr(10), ' ')} · {t['form'].lower() or 'claim'} · {deg[t['key']]} links"
    fig.text(0.53, y, head, fontsize=10, color=FORM.get(t["form"], INK2), weight="bold", va="top")
    body = textwrap.fill(t["title"], 78)
    fig.text(0.53, y - 0.024, body, fontsize=10.8, color=INK, va="top")
    y -= 0.024 + 0.022 * (body.count("\n") + 1) + 0.022
fig.savefig(OUT / "figA-theorems.png", dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.2)
fig.savefig(OUT / "figA-theorems.svg", facecolor="white", bbox_inches="tight", pad_inches=0.2)
print("wrote figA-theorems", len(th), len(edges), [t["key"] for t in picks])
