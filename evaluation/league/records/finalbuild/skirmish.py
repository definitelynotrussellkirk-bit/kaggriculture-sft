#!/usr/bin/env python3
"""Owner 19:0x "skirmishes of 3, best of 3, like 5 times": for every running VS-P2 (N flow vs P2) and VS-P1-NB (B vs P1) panel, once 15 games
are finished: the first 15 finished games in completion order -> 5 best-of-3 skirmishes (points 1/.5/0 per game; > 1.5 = skirmish won,
< 1.5 lost, = 1.5 drawn); R-SKIRMISH-<candidate> with the tally, raw game W-L-D and share of all finished games. Same game filter as the
panel reader (frozen manifest identity and window, record_issues, forfeit = loss for that seat). One line per candidate at 15 games."""
import json, sys, time
from pathlib import Path
R = Path("/mnt/ai/scratch/farmNEAT/league/records"); sys.path[:0] = [str(R), str(R / "gates"), str(R / "ladder"), str(R / "colab")]; sys.argv = sys.argv[:1]
import panel_league as P
FB = R / "finalbuild"; LEDGER = Path("/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md"); DONE = FB / "SKIRMISH-DONE.json"


def games(man, G):
    M = json.loads((R / "gates" / man).read_text()); a, b = M["challenger"], M["incumbent"]; W = set(M["worlds"])
    sha = {a["name"]: a["sha256"], b["name"]: b["sha256"]}; fz = P.utc_ts(M["frozen_at_utc"]); out = []
    for r in G:
        if {r.get("seat0"), r.get("seat1")} != set(sha) or r.get("world_seed") not in W:
            continue
        st_, fin = P.started_of(r), P.utc_ts(r.get("finished_utc"))
        if P.record_issues(r) or r.get("archive_sha256") != [sha[r["seat0"]], sha[r["seat1"]]] or st_ is None or st_ < fz or fin is None:
            continue
        me = 0 if r["seat0"] == a["name"] else 1; m = float(r["cash"][me]) - float(r["cash"][1 - me]); fs = P.forfeit_seat(r)
        pts = (1.0 if m > 0 else .5 if m == 0 else 0.0) if fs is None else .5 if fs == "both" else (0.0 if fs == me else 1.0)
        out.append((fin.timestamp(), pts))
    return a["name"], b["name"], sorted(out)


def main():
    done = json.loads(DONE.read_text()) if DONE.exists() else {}
    while True:
        try:
            mans = []
            for sf in ("VSP2-STATE.json", "VSP1NB-STATE.json"):
                if (FB / sf).exists():
                    mans += [c["manifest"] for c in json.loads((FB / sf).read_text())["cands"].values()]
            todo = [m for m in mans if m not in done]
            if todo:
                G = P.load_gate_games()
                for man in todo:
                    a, b, gs = games(man, G)
                    if len(gs) < 15:
                        continue
                    sk = [sum(p for _, p in gs[i:i + 3]) for i in range(0, 15, 3)]
                    w, l, d = sum(s > 1.5 for s in sk), sum(s < 1.5 for s in sk), sum(s == 1.5 for s in sk)
                    gw, gl, gd = sum(p == 1 for _, p in gs), sum(p == 0 for _, p in gs), sum(p == .5 for _, p in gs)
                    line = (f"R-SKIRMISH-{a} ({time.strftime('%H:%M')} 09-30): vs {b}: best-of-3 skirmishes (first 15 finished games, completion order) {w}-{l}"
                            + (f"-{d}" if d else "") + f" [{' '.join(f'{s:g}' for s in sk)}]; raw games {gw}-{gl}" + (f"-{gd}" if gd else "")
                            + f" of {len(gs)} finished, share {sum(p for _, p in gs) / len(gs):.3f} (still moving; full world-clustered reads continue)")
                    with LEDGER.open("a") as fh:
                        fh.write(line + "\n")
                    done[man] = line; DONE.write_text(json.dumps(done, indent=1))
        except Exception as e:  # noqa: BLE001
            print(time.strftime("%H:%M"), "error", type(e).__name__, str(e)[:200], flush=True)
        time.sleep(20)


if __name__ == "__main__":
    main()
