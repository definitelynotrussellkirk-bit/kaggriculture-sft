"""Generate review/FRONTIER.md: the unconsumed theorem outputs, grouped by layer, with
the output interface each one exposes and the compositions the maintainer proposes.

The ledger records stay the authority; this page is derived from them and from the
`[[frontier]]` entries of project/review.toml. Never edit the page by hand.

    python3 -B src/farm_neat/tooling/frontier.py            # rewrite review/FRONTIER.md
    python3 -B src/farm_neat/tooling/frontier.py --stats    # print the graph statistics only
"""
from __future__ import annotations

import json
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEDGER = "project/registries/search-ledger"
SPEC = "project/review.toml"
PAGE = "review/FRONTIER.md"


def load_records(root: Path) -> dict[str, dict]:
    folder = root / LEDGER
    out = {}
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        out[path.stem] = json.loads(path.read_text())
    return out


def graph(records: dict[str, dict]) -> dict:
    """Live claims (not superseded), their claim-to-claim edges, consumers, leaves and depth."""
    claims = {h: r for h, r in records.items() if r.get("type") == "claim"}
    superseded: set[str] = set()
    for r in records.values():
        for rel in r.get("relations") or []:
            if rel.get("kind") == "supersedes":
                superseded.add(rel["target"])
        s = r.get("supersedes")
        if isinstance(s, str):
            superseded.add(s)
        elif isinstance(s, list):
            superseded.update(s)
    live = {h: r for h, r in claims.items() if h not in superseded}
    consumers: dict[str, list[str]] = defaultdict(list)
    edges = 0
    for h, r in live.items():
        for p in r.get("depends_on") or []:
            if p in live:
                consumers[p].append(h)
                edges += 1
    followups: dict[str, list[str]] = defaultdict(list)
    for h, r in records.items():
        if r.get("type") == "followup":
            for c in r.get("claims") or []:
                followups[c].append(str(r.get("key")))
    depth: dict[str, int] = {}

    def d(h: str, stack=()) -> int:
        if h in depth:
            return depth[h]
        if h in stack:
            return 1
        parents = [p for p in live[h].get("depends_on") or [] if p in live]
        depth[h] = 1 + max((d(p, stack + (h,)) for p in parents), default=0)
        return depth[h]

    for h in live:
        d(h)
    leaves = [h for h in live if not consumers.get(h)]
    with_parents = sum(1 for h, r in live.items() if any(p in live for p in r.get("depends_on") or []))
    with_relations = sum(1 for r in live.values() if r.get("relations"))
    return {"records": len(records), "claims": len(claims), "live_claims": len(live),
            "claims_with_parents": with_parents, "edges": edges,
            "consumed": sum(1 for h in live if consumers.get(h)), "leaves": len(leaves),
            "with_relations": with_relations, "max_depth": max(depth.values(), default=0),
            "_live": live, "_leaves": leaves, "_consumers": consumers, "_followups": followups,
            "_depth": depth}


def interface(rec: dict) -> str:
    """The output interface convention: the first `reuse` line names the sufficient statistic
    the next theorem consumes and what remains."""
    reuse = rec.get("reuse") or []
    return (reuse[0] if reuse else "(no reuse line: interface not stated)").strip()


def page(root: Path) -> str:
    records = load_records(root)
    spec = tomllib.loads((root / SPEC).read_text())
    g = graph(records)
    live, leaves = g["_live"], g["_leaves"]
    key2layer: dict[str, str] = {}
    for layer in spec.get("layers", []):
        for k in layer.get("claims", []):
            key2layer[k] = layer["name"]
    lines = ["# Frontier: unconsumed theorem outputs", "",
             "Generated from the immutable ledger and `[[frontier]]` in project/review.toml by "
             "`frontier.py`; never edit by hand. A leaf is a live claim no later claim depends on. "
             "Its interface line is the first `reuse` entry of its record: the statistic it hands "
             "to the next theorem and the variables it leaves. Prefer a composition that consumes "
             "two leaves over a new primitive; state which distinctions it deletes and which "
             "state variables it adds.", "",
             "## Graph", "",
             f"- live claims {g['live_claims']} (of {g['claims']} admitted); {g['claims_with_parents']} depend on "
             f"an earlier claim through {g['edges']} edges; {g['consumed']} have been consumed by a later claim; "
             f"{g['leaves']} leaves; longest chain {g['max_depth']}; {g['with_relations']} carry an explicit "
             f"narrows/corroborates/refutes/supersedes relation.", ""]
    proposals = spec.get("frontier", [])
    if proposals:
        lines += ["## Proposed compositions", "",
                  "A.output + B.output -> C, as the maintainer currently sees them; a reviewer may take one "
                  "or propose a better one. Each names the leaves it would consume.", ""]
        for p in proposals:
            lines.append(f"- **{p['name']}**: {' + '.join(p['consumes'])} -> {p['target']}. {p.get('note', '')}".rstrip())
        lines.append("")
    retired = spec.get("retired_distinctions", [])
    if retired:
        lines += ["## Retired distinctions", "",
                  "Distinctions a downstream theorem has subsumed; their proofs stay in the ledger but they are "
                  "no longer research choices. Start from the theorem named, not the raw object.", ""]
        for r in retired:
            lines.append(f"- {r['distinction']} -> **{r['by']}**: {r.get('note', '')}".rstrip())
        lines.append("")
    by_layer: dict[str, list[str]] = defaultdict(list)
    for h in leaves:
        by_layer[key2layer.get(live[h]["key"], "Unplaced")].append(h)
    order = [L["name"] for L in spec.get("layers", [])] + ["Unplaced"]
    lines += ["## Leaves by layer", ""]
    for name in order:
        hs = by_layer.get(name)
        if not hs:
            continue
        lines += [f"### {name} ({len(hs)})", ""]
        for h in sorted(hs, key=lambda x: live[x]["key"]):
            rec = live[h]
            fu = g["_followups"].get(h)
            tail = f" Follow-ups: {', '.join(fu)}." if fu else ""
            lines.append(f"- **{rec['key']}** (depth {g['_depth'][h]}, `{h[:12]}`): {interface(rec)}{tail}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str]) -> int:
    root = ROOT
    if "--stats" in argv:
        g = graph(load_records(root))
        print(json.dumps({k: v for k, v in g.items() if not k.startswith("_")}, indent=1))
        return 0
    text = page(root)
    (root / PAGE).write_text(text)
    print(json.dumps({"file": PAGE, "bytes": len(text.encode()), "commands_executed": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
