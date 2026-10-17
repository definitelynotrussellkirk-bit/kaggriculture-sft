#!/usr/bin/env python3
"""Coordinator 19:0x: the GPU-spool B vs P1 panels never ran (no daemon claimed them). CPU-native instead, VS-P2 bank worlds 1-8 x 2 seats per
candidate vs P1 b-colab-u38760-d30-v9gsafe (95e5b4e5), all in parallel: n1-u16200 -> Colab #2 nflow-g4 lane; n1-main-1730 -> spark;
n1-main-sprinkle (when built + strictly verified) -> Colab #2 5 worlds + laptop 3 worlds. R-VS-P1-NB whenever a candidate's finished-game
count grows: raw game W-L(-D), share of finished games, world-clustered share on complete worlds; still moving until 16 games."""
import json, os, subprocess, sys, time
from pathlib import Path
R = Path("/mnt/ai/scratch/farmNEAT/league/records"); sys.path[:0] = [str(R), str(R / "gates"), str(R / "ladder"), str(R / "colab"), str(R / "finalbuild")]; sys.argv = sys.argv[:1]
import panel_league as P, gate_freezer as GF, ladder as LD, tplan as T
from verify_strict import verify_strict
import skirmish as SK
FB = R / "finalbuild"; ST = FB / "VSP1CPU-STATE.json"; LEDGER = Path("/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md")
LC = Path("/mnt/ai/scratch/farmNEAT/selfplay-s/colab-lane"); P1 = "b-colab-u38760-d30-v9gsafe"
PLAN = {"n1-u16200": [("g4", 0, 8), ("gpu", 8, 16)], "n1-main-1730": [("spark", 0, 8), ("gpu", 8, 16)], "n1-main-sprinkle": [("g4", 0, 5), ("laptop", 5, 8), ("gpu", 8, 16)]}
# owner 19:1x "use all other GPUS": worlds 9-16 of each candidate on the GPU spool (records/gpu/val-vsp1g-<n>, PRIORITY 0; disjoint from the CPU worlds 1-8)


def rows(n, ws, e0):
    L = ["a_name\ta_archive\tb_name\tb_archive\tworld_seed\tepisode_id"]; e = e0
    for w in ws:
        for x, y in ((n, P1), (P1, n)):
            e += 1; L.append(f"{x}\t{GF.tgz(x)}\t{y}\t{GF.tgz(y)}\t{w}\t{e}")
    return L


def launch(n, W, e0):
    for lane, i0, i1 in PLAN[n]:
        L = rows(n, W[i0:i1], e0 + i0 * 10)
        if lane == "gpu":
            gd = R / "gpu" / f"val-vsp1g-{n}"
            if not (gd / "pairs.tsv").exists():
                gd.mkdir(parents=True, exist_ok=True); (gd / "PRIORITY").write_text("0\n"); t = gd / ".pairs.tsv.tmp"; t.write_text("\n".join(L) + "\n"); t.replace(gd / "pairs.tsv")
            continue
        if lane == "g4":
            b = f"t-vsp1cpu-{n}"; bd = R / "colab" / b
            if (bd / "pairs.tsv").exists():
                continue
            bd.mkdir(parents=True, exist_ok=True); (bd / "PRIORITY").write_text("0\n"); t = bd / ".pairs.tsv.tmp"; t.write_text("\n".join(L) + "\n"); t.replace(bd / "pairs.tsv")
            with (LC / "DISPATCHED.txt").open("a") as fh:
                fh.write(b + "\n")
            with (LC / "ASSIGN.tsv").open("a") as fh:
                fh.write(f"{b}\tnflow-g4\t{len(L) - 1}\t{P.now()}\n")
            subprocess.Popen(["bash", str(LC / "p0batch3.sh"), b, b.upper(), "64"], env={**os.environ, "VMQ_HOST": "nflow-g4"},
                             stdout=open(LC / f"p0-{b}.out", "w"), stderr=subprocess.STDOUT, cwd=str(LC), start_new_session=True)
        else:
            args = {"spark": ["-p", "spark", "--mem", "2700"], "laptop": ["-p", "cpu", "-w", "laptop-4060", "--mem", "1500"]}[lane]
            od = R / "slurmlane" / f"val-vsp1cpu-{n}__{lane}"
            if (od / "JOB").exists():
                continue
            od.mkdir(parents=True, exist_ok=True); (od / "pairs.tsv").write_text("\n".join(L) + "\n")
            j = subprocess.run(["sbatch", "--parsable", *args, "-c", "1", "-t", "00:50:00", f"--array=0-{len(L) - 2}", "-J", "val-vsp1cpu",
                                "-o", str(od / "slurm-%A_%a.out"), str(P.FARM_SBATCH), str(od / "pairs.tsv"), str(od)], capture_output=True, text=True)
            jid = P.sbatch_job(j)
            if not jid:
                raise RuntimeError(f"sbatch failed {od.name}: {(j.stderr or '')[:200]}")
            (od / "JOB").write_text(jid + "\n")


def main():
    W = json.loads((R / "banks" / "VS-P2.json").read_text())["seeds"]
    st = json.loads(ST.read_text()) if ST.exists() else {"cands": {}, "rule": __doc__}
    while True:
        try:
            for i, n in enumerate(PLAN):
                if n in st["cands"] or not (GF.DEP / n).exists() or not verify_strict(n)[0]:
                    continue
                tag = f"VS-P1NB-{n}-vs-{P1}"; mf = R / "gates" / f"{tag}.json"
                M = json.loads(mf.read_text()) if mf.exists() else P.freeze_gate(
                    mf, n, P1, W, "superiority", "2026-09-30T23:55:00Z", bank=tag, allow_gpu_served=False, cpu_only_reason="CPU-native lanes",
                    source="owner 18:5x B slot vs P1", descriptive="VS-P1 B panel", binding="outer archive sha256 only", size={"worlds": 16, "games": 32})
                launch(n, W, 998900000000 + i * 1000)
                st["cands"][n] = {"sha": LD.GF_sha(n), "manifest": mf.name, "n": 0}; ST.write_text(json.dumps(st, indent=1))
                with LEDGER.open("a") as fh:
                    fh.write(f"R-VS-P1-NB-START ({time.strftime('%H:%M')} 09-30): {n} ({LD.GF_sha(n)[:8]}) vs P1 {P1} (95e5b4e5) CPU-native on VS-P2 worlds 1-8 x 2 seats "
                             f"(16 games): " + ", ".join(f"{'Colab #2 nflow-g4' if l == 'g4' else l} worlds {a + 1}-{b}" for l, a, b in PLAN[n]) + f"; manifest {M['manifest_sha256'][:16]}\n")
            if st["cands"]:
                G = P.load_gate_games()
                for n, c in st["cands"].items():
                    _, _, gs = SK.games(c["manifest"], G)
                    if not gs or len(gs) == c["n"]:
                        continue
                    c["n"] = len(gs); ST.write_text(json.dumps(st, indent=1))
                    gw, gl, gd = sum(p == 1 for _, p in gs), sum(p == 0 for _, p in gs), sum(p == .5 for _, p in gs)
                    T.STATS.pop(c["manifest"], None); ws, _, _ = T.world_scores(c["manifest"], G, allow_gpu=True)
                    sc = [(v[0][0] + v[1][0]) / 2 for v in ws.values() if len(v) == 2 and len({x[2] for x in v.values()}) == 1]
                    wc = ""
                    if len(sc) >= 2:
                        s, ci = T.ci(sc); wc = f"; world-clustered {s:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] on {len(sc)} complete worlds"
                    with LEDGER.open("a") as fh:
                        fh.write(f"R-VS-P1-NB ({time.strftime('%H:%M')} 09-30): {n} ({c['sha'][:8]}) vs P1: games {gw}-{gl}" + (f"-{gd}" if gd else "")
                                 + f" of {len(gs)} finished (of 32), share {sum(p for _, p in gs) / len(gs):.3f}{wc} ({'final' if len(gs) >= 32 else 'still moving'})\n")
        except Exception as e:  # noqa: BLE001
            print(time.strftime("%H:%M"), "error", type(e).__name__, str(e)[:200], flush=True)
        time.sleep(20)


if __name__ == "__main__":
    main()
