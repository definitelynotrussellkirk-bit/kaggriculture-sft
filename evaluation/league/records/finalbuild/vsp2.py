#!/usr/bin/env python3
"""Owner 18:5x "make sure whatever we submit is better than" P2 nflow-u16000-t4 (6033c855, hosted 2776.1): each N flow candidate vs P2 on
ONE shared fresh 16-world bank (banks/VS-P2.json, both seats), CPU-native, all lanes: spark 6 / gpu-skynet 4 / laptop 3 / gpu-4090b 3 worlds.
Candidates: nflow-main-wash8014, nflow-main-1730, nflow-u56070-t1w now; nflow-main-washnew when R-WASH-READY-nflow-main-washnew carries the
deployed sha. R-VS-P2 per candidate whenever its complete-world count grows (still moving until 16): share, 95% world-clustered CI."""
import json, subprocess, sys, time
from pathlib import Path
R = Path("/mnt/ai/scratch/farmNEAT/league/records"); sys.path[:0] = [str(R), str(R / "gates"), str(R / "ladder"), str(R / "colab")]; sys.argv = sys.argv[:1]
import panel_league as P, gate_freezer as GF, ladder as LD, tplan as T
from verify_strict import verify_strict
FB = R / "finalbuild"; ST = FB / "VSP2-STATE.json"; LEDGER = Path("/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md")
P2 = "nflow-u16000-t4"; CANDS = ["nflow-main-wash8014", "nflow-main-1730", "nflow-u56070-t1w", "nflow-main-washnew"]
LANES = (("spark", ["-p", "spark", "--mem", "2700"], str(P.FARM_SBATCH), 20), ("skynet", ["-p", "gpu", "-w", "gpu-skynet", "--mem", "1500"], str(R / "skynet_game.sbatch"), 8),
         ("laptop", ["-p", "cpu", "-w", "laptop-4060", "--mem", "1500"], str(P.FARM_SBATCH), 6), ("4090b", ["-p", "gpu", "-w", "gpu-4090b", "--mem", "2500"], str(P.FARM_SBATCH), 8))


def ready(n):
    if not (GF.DEP / n).exists() or not verify_strict(n)[0]:
        return False
    if n != "nflow-main-washnew":
        return True
    sha = LD.GF_sha(n)
    return any(l.startswith(f"R-WASH-READY-{n} ") and f"sha256 {sha}" in l for l in LEDGER.read_text().splitlines())


def launch_g4(n, W, e0):
    """Owner 19:0x: Colab #2 (nflow-g4, 48 vCPU, G 24) as a CPU-native lane: the whole panel as colab/t-vsp2-<n>, run by p0batch3.sh."""
    Lc = Path("/mnt/ai/scratch/farmNEAT/selfplay-s/colab-lane"); b = f"t-vsp2-{n}"; bd = R / "colab" / b
    if (bd / "pairs.tsv").exists():
        return
    L = ["a_name\ta_archive\tb_name\tb_archive\tworld_seed\tepisode_id"]; e = e0
    for w in W:
        for x, y in ((n, P2), (P2, n)):
            e += 1; L.append(f"{x}\t{GF.tgz(x)}\t{y}\t{GF.tgz(y)}\t{w}\t{e}")
    bd.mkdir(parents=True, exist_ok=True); (bd / "PRIORITY").write_text("0\n"); t = bd / ".pairs.tsv.tmp"; t.write_text("\n".join(L) + "\n"); t.replace(bd / "pairs.tsv")
    with (Lc / "DISPATCHED.txt").open("a") as fh:
        fh.write(b + "\n")
    with (Lc / "ASSIGN.tsv").open("a") as fh:
        fh.write(f"{b}\tnflow-g4\t{len(L) - 1}\t{P.now()}\n")
    subprocess.Popen(["bash", str(Lc / "p0batch3.sh"), b, b.upper().replace("_", "-"), "64"], env={**__import__("os").environ, "VMQ_HOST": "nflow-g4"},
                     stdout=open(Lc / f"p0-{b}.out", "w"), stderr=subprocess.STDOUT, cwd=str(Lc), start_new_session=True)


def launch(n, W, e0):
    if n == "nflow-main-washnew":
        return launch_g4(n, W, e0)
    for (lane, args, script, thr), ws, off in zip(LANES, (W[:6], W[6:10], W[10:13], W[13:16]), (0, 100, 200, 300)):
        od = R / "slurmlane" / f"val-vsp2-{n}__{lane}"
        if (od / "JOB").exists():
            continue
        L = ["a_name\ta_archive\tb_name\tb_archive\tworld_seed\tepisode_id"]; e = e0 + off
        for w in ws:
            for x, y in ((n, P2), (P2, n)):
                e += 1; L.append(f"{x}\t{GF.tgz(x)}\t{y}\t{GF.tgz(y)}\t{w}\t{e}")
        od.mkdir(parents=True, exist_ok=True); (od / "pairs.tsv").write_text("\n".join(L) + "\n")
        j = subprocess.run(["sbatch", "--parsable", *args, "-c", "1", "-t", "00:50:00", f"--array=0-{len(L) - 2}%{thr}", "-J", "val-vsp2",
                            "-o", str(od / "slurm-%A_%a.out"), script, str(od / "pairs.tsv"), str(od)], capture_output=True, text=True)
        jid = P.sbatch_job(j)
        if not jid:
            raise RuntimeError(f"sbatch failed {od.name}: {(j.stderr or '')[:200]}")
        (od / "JOB").write_text(jid + "\n")


def main():
    f = R / "banks" / "VS-P2.json"
    if not f.exists():
        f.write_text(json.dumps({"bank": "VS-P2", "seeds": LD.fresh(16), "created_utc": P.now(), "rule": __doc__.split("\n")[0]}, indent=1))
    W = json.loads(f.read_text())["seeds"]
    st = json.loads(ST.read_text()) if ST.exists() else {"cands": {}, "rule": __doc__}
    while True:
        try:
            for i, n in enumerate(CANDS):
                if n in st["cands"] or not ready(n):
                    continue
                tag = f"VS-P2-{n}-vs-{P2}"; mf = R / "gates" / f"{tag}.json"
                M = json.loads(mf.read_text()) if mf.exists() else P.freeze_gate(
                    mf, n, P2, W, "superiority", "2026-09-30T23:55:00Z", bank=tag, allow_gpu_served=False, cpu_only_reason="CPU-native SLURM lanes",
                    source="owner 18:5x vs P2", descriptive="VS-P2 panel", binding="outer archive sha256 only", size={"worlds": 16, "games": 32})
                launch(n, W, 998700000000 + i * 1000)
                st["cands"][n] = {"sha": LD.GF_sha(n), "manifest": mf.name, "n": 0}; ST.write_text(json.dumps(st, indent=1))
                with LEDGER.open("a") as fh:
                    fh.write(f"R-VS-P2-START ({time.strftime('%H:%M')} 09-30): {n} ({LD.GF_sha(n)[:8]}) vs P2 {P2} ({LD.GF_sha(P2)[:8]}) on bank VS-P2 (16 worlds x 2 seats; "
                             f"manifest {M['manifest_sha256'][:16]}); " + ("Colab #2 nflow-g4 (all 16 worlds)" if n == "nflow-main-washnew" else "spark 6 / gpu-skynet 4 / laptop 3 / gpu-4090b 3 worlds") + "\n")
            if st["cands"]:
                G = P.load_gate_games()
                for n, c in st["cands"].items():
                    T.STATS.pop(c["manifest"], None); ws, _, _ = T.world_scores(c["manifest"], G)
                    sc = [(v[0][0] + v[1][0]) / 2 for v in ws.values() if len(v) == 2 and len({x[2] for x in v.values()}) == 1]
                    if len(sc) < 2 or len(sc) == c["n"]:
                        continue
                    c["n"] = len(sc); s, ci = T.ci(sc); ST.write_text(json.dumps(st, indent=1))
                    stc = {k: T.STATS.get(c["manifest"], {}).get(k, 0) for k in ("forfeits", "runtime_rejected", "conflicts")}
                    with LEDGER.open("a") as fh:
                        fh.write(f"R-VS-P2 ({time.strftime('%H:%M')} 09-30): {n} ({c['sha'][:8]}) vs P2 {P2}: share {s:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] on {len(sc)} worlds x 2 seats "
                                 f"({'final' if len(sc) == 16 else 'still moving'}); structural {stc}\n")
        except Exception as e:  # noqa: BLE001
            print(time.strftime("%H:%M"), "error", type(e).__name__, str(e)[:200], flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
