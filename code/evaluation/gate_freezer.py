#!/usr/bin/env python3
"""u-0300 gates (owner 00:5x / coordinator 01:00 09-30). Each row of gates/REQUESTS.tsv (archive, parent, rank, note; gpu-ops appends) is
frozen the moment both archives are deployed and verified (tar.gz + VERIFY.json), before any game: 32 new worlds, kind non-inferiority
vs the parent (superiority read is descriptive), deadline = freeze + 3 h capped at 23:30Z. GPU-served play is allowed at freeze only for
architecture families covered by the Spark parity evidence (base1-b / b-colab B and oldarch1 O); anything else (N B-even, N flow, ...) is
CPU-native only. GPU-allowed gates split worlds between one GPU batch (records/gpu/0gate-<rank>-<archive>, PRIORITY first, pairs.tsv
written atomically) and one CPU array (records/gaterun/<archive>, job 'u0300-gate'); every pairs file is written from the manifest's own
worlds. A request is marked done only after the CPU array has a job id; a failed sbatch is retried next pass against the same manifest."""
import json, random, subprocess, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
R = Path("/mnt/ai/scratch/farmNEAT/league/records"); sys.path.insert(0, str(R))
import panel_league as P
G = R / "gates"; ST = G / "freezer-state.json"; DEP = Path("/mnt/ai/scratch/farmNEAT/pbc-deploy")
GPU_WORLDS = 20                     # of 32: measured ~150/h GPU vs ~90/h CPU (B-vs-B), 00:55 09-30
SPARK_SHARE = 2 / 3                 # of the CPU worlds: ~14 Spark slots vs ~7 x86 slots
PARITY_EXTRA = "/mnt/ai/scratch/farmNEAT/league/gpuserve/audit/v2-parity-spark/PARITY.jsonl"
GPU_PARITY_FAMILIES = ("base1-b-", "b-colab-", "oldarch1-")   # Spark parity: B31kG (base1-b v8g) and OA1 u65000/u79000 (red-team 01:30)
DEADLINE_CAP = "2026-09-30T23:30:00Z"
sys.path.insert(0, str(Path(__file__).resolve().parent)); from verify_strict import verify_strict   # coordinator 07:1x
ready = lambda n: (DEP / n / f"{n}.tar.gz").is_file() and (DEP / n / "VERIFY.json").is_file() and verify_strict(n)[0]
tgz = lambda n: f"{DEP}/{n}/{n}.tar.gz"


def validate_dispatch_manifest(M, a, b, kind=None):
    """Validate a reload against the request and outer bytes; no writes or scheduler calls."""
    bad = P.manifest_issues(M)
    if bad:
        raise ValueError(f"invalid dispatch manifest: {bad}")
    if M["challenger"]["name"] != a or M["incumbent"]["name"] != b:
        raise ValueError("dispatch request identities differ from frozen manifest")
    if kind is not None and M["kind"] != kind:
        raise ValueError("dispatch request kind differs from frozen manifest")
    if M.get("post_hoc"):
        raise ValueError("post-hoc manifest is not a prospective dispatch")
    import hashlib
    for role in ("challenger", "incumbent"):
        x = M[role]
        try:
            with open(tgz(x["name"]), "rb") as fh:
                h = hashlib.file_digest(fh, "sha256").hexdigest()
        except OSError as e:
            raise ValueError(f"dispatch archive unavailable: {x['name']}") from e
        if h != x["sha256"]:
            raise ValueError(f"dispatch archive bytes differ from frozen manifest: {x['name']}")


def deadline_for(now):
    """freeze + 3 h, capped at 23:30Z (red-team P1-1: a fixed 18:00Z deadline can never be met by late requests)."""
    cap = datetime.fromisoformat(DEADLINE_CAP.replace("Z", "+00:00"))
    return min(now + timedelta(hours=3), cap).strftime("%Y-%m-%dT%H:%M:%SZ")


SERVED_MODEL_PY = {"60958b937dfa6a44": "B", "b60e3228cb6fc68b": "O"}   # model.py sha256 prefixes of the parity-tested B31kG / OA1 archives
NO_GPU_NAMES = ("n1-", "nflow", "n-flow", "n2-")                        # N: identical B model.py but regrouped heads; no N parity yet (red-team)
_msha = {}


def model_family(n):
    """Architecture family by the sha256 of the archive's model.py (the code the Spark daemon serves), else None."""
    if n not in _msha:
        f = DEP / n / f"{n}.tar.gz"
        try:
            names = subprocess.run(["tar", "-tzf", str(f)], capture_output=True, text=True, timeout=300).stdout.split()
            m = next((x for x in names if x.endswith("/model.py") or x == "model.py"), None)
            h = __import__("hashlib").sha256(subprocess.run(["tar", "-xzf", str(f), "-O", m], capture_output=True, timeout=300).stdout).hexdigest() if m else ""
        except (OSError, subprocess.TimeoutExpired):
            h = ""
        _msha[n] = next((fam for pre, fam in SERVED_MODEL_PY.items() if h.startswith(pre)), None)
    return _msha[n]


def gpu_ok(a, par):
    """GPU-served only when both archives run a parity-tested model.py family (B or O) and neither is an N line."""
    if a.startswith(NO_GPU_NAMES) or par.startswith(NO_GPU_NAMES):
        return False
    fa = model_family(a) if (DEP / a).exists() else ("B" if a.startswith(("base1-b-", "b-colab-")) else "O" if a.startswith("oldarch1-") else None)
    fb = model_family(par) if (DEP / par).exists() else ("B" if par.startswith(("base1-b-", "b-colab-")) else "O" if par.startswith("oldarch1-") else None)
    return fa is not None and fb is not None


def used_seeds():
    k = set()
    for l in (R / "games.jsonl").open():
        try:
            k.add(int(json.loads(l)["world_seed"]))
        except (ValueError, KeyError, TypeError):
            pass
    for f in (R / "banks").glob("*.json"):
        d = json.loads(f.read_text())
        for v in d.values():
            if isinstance(v, list):
                k |= {x for x in v if type(x) is int}
    for f in G.glob("*.json"):
        k |= set(json.loads(f.read_text()).get("worlds") or [])
    for pf in list(R.glob("*/*/pairs.tsv")) + list(R.glob("*/pairs.tsv")) + list(R.glob("*/*/*/pairs.tsv")):
        for l in pf.read_text().splitlines()[1:]:
            try:
                k.add(int(l.split("\t")[4]))
            except (IndexError, ValueError):
                pass
    k |= set(json.loads((R / "WORLDS-T16.json").read_text())["seeds"]) | set(range(32200001, 32200017))
    for d in ("r3-oa79-oa65", "r3-c22-oa79"):
        k |= set(json.loads((R / d / "exposed-worlds.json").read_text())["worlds"])
    return k


def write_pairs(d, a, par, worlds, eid0, prio=None):
    d.mkdir(parents=True, exist_ok=True); L = ["a_name\ta_archive\tb_name\tb_archive\tworld_seed\tepisode_id"]; eid = eid0
    for sd in worlds:
        for x, y in ((a, par), (par, a)):
            eid += 1; L.append(f"{x}\t{tgz(x)}\t{y}\t{tgz(y)}\t{sd}\t{eid}")
    if prio is not None:
        (d / "PRIORITY").write_text(f"{prio}\n")             # before pairs.tsv: a dir is claimable once pairs.tsv exists
    tmp = d / ".pairs.tsv.tmp"; tmp.write_text("\n".join(L) + "\n"); tmp.replace(d / "pairs.tsv")
    return len(L) - 1


def gate_one(a, par, rank, note, st, *, gates=G, rec=R, freeze=P.freeze_gate, run=subprocess.run, seeds=None, now=None):
    """Freeze (or reload) the manifest, write pairs from M['worlds'], submit the CPU array; returns the state entry, or None to retry."""
    now = now or datetime.now(timezone.utc)
    mf = gates / f"U0300-{a}.json"
    if mf.exists():
        M = json.loads(mf.read_text())                       # retry: never redraw worlds for an existing manifest
    else:
        g_ok = gpu_ok(a, par); size = None
        if seeds is None:
            sys.path.insert(0, str(R / "ladder")); import ladder as LD_    # owner 04:1x/04:2x: 64 worlds where CPU allows before the use time, else >= 32
            n_w, size = LD_.size_for("direct", g_ok)
            known = used_seeds(); rng = random.Random(int(time.time())); seeds = []
            while len(seeds) < n_w:
                x = rng.randrange(40_000_000, 2_000_000_000)
                if x not in known and x not in seeds:
                    seeds.append(x)
        M = freeze(mf, a, par, seeds, "noninferiority", deadline_for(now), bank=f"U0300-{a}", allow_gpu_served=g_ok,
                   **({"gpu_parity_extra": PARITY_EXTRA} if g_ok else {"cpu_only_reason": "no Spark parity evidence for this architecture family"}),
                   rank=rank, descriptive="superiority read shown, not decisive", note=note,
                   source="owner 00:5x / coordinator 01:00 09-30 (u-0300 snapshot gates)", **({"size": size} if size else {}))
    validate_dispatch_manifest(M, a, par, "noninferiority")
    if M.get("games_seen_at_freeze") != 0:
        return {"error": "games seen at freeze", "manifest_sha256": M.get("manifest_sha256")}
    W = list(M["worlds"]); eid0 = 993500000 + 1000 * (len(st["done"]) + 1)
    ng, gd = 0, None
    if M.get("allow_gpu_served"):
        gd = rec / "gpu" / f"0gate-{rank}-{a}"[:200]
        if not (gd / "pairs.tsv").exists():
            ng = write_pairs(gd, a, par, W[:round(len(W) * GPU_WORLDS / 32)], eid0, prio=rank)
        else:
            ng = sum(1 for l in (gd / "pairs.tsv").read_text().splitlines()[1:] if l.strip())
        cpu_worlds = W[round(len(W) * GPU_WORLDS / 32):]
    else:
        cpu_worlds = W
    # GATE-MATH (coordinator 02:00): both seats of a world on ONE runtime -> CPU worlds split into a Spark-pinned (aarch64) array and an
    # x86-pinned array; each world wholly in one of them (and GPU worlds wholly on the GPU daemon)
    ks = round(len(cpu_worlds) * SPARK_SHARE)
    parts = [("spark", cpu_worlds[:ks], rec / "gaterun" / a), ("cpu", cpu_worlds[ks:], rec / "gaterun" / f"{a}__x86")]
    jids, nc = [], 0
    for part, ws_, cd in parts:
        if not ws_:
            continue
        prev = (cd / "JOB").read_text().strip() if (cd / "JOB").exists() else None
        if prev:
            jids.append(prev); nc += 2 * len(ws_); continue
        n_ = write_pairs(cd, a, par, ws_, eid0 + (500 if part == "spark" else 700))
        j = run(["sbatch", "--parsable", "-p", part, "-c", "1", "--mem", ("2700" if part == "spark" else "1500"), "-t", "00:45:00", f"--array=0-{n_ - 1}",
                 "-J", "u0300-gate", "-o", str(cd / "slurm-%A_%a.out"), P.FARM_SBATCH, str(cd / "pairs.tsv"), str(cd)], capture_output=True, text=True)
        jid_ = P.sbatch_job(j)
        if jid_ is None:
            return None                                      # not done: retried next pass against the same manifest
        (cd / "JOB").write_text(jid_ + "\n"); jids.append(jid_); nc += n_
    jid = ",".join(jids); cd = parts[0][2]
    return {"parent": par, "rank": rank, "manifest_sha256": M["manifest_sha256"], "gpu_batch": str(gd) if gd else None, "gpu_games": ng,
            "cpu_dir": str(cd), "cpu_games": nc, "cpu_job": jid, "frozen_at_utc": M["frozen_at_utc"], "deadline_utc": M["deadline_utc"],
            "allow_gpu_served": bool(M.get("allow_gpu_served"))}


def main():
    st = json.loads(ST.read_text()) if ST.exists() else {"done": {}}
    while True:
        try:
            rows = [l.split("\t") for l in (G / "REQUESTS.tsv").read_text().splitlines()[1:] if l.strip() and not l.startswith("#")]
            for f in rows:
                a, par = f[0].strip(), f[1].strip(); rank = int(f[2]) if len(f) > 2 and f[2].strip().isdigit() else 9
                if a in st["done"] or not (ready(a) and ready(par)) or a.startswith(P.EXCLUDED) or par.startswith(P.EXCLUDED):
                    continue
                e = gate_one(a, par, rank, f[3].strip() if len(f) > 3 else "", st)
                if e is None:
                    print(time.strftime("%H:%M"), "sbatch failed for", a, "- retry next pass (manifest kept)", flush=True); continue
                st["done"][a] = e
                tmp = ST.with_suffix(".tmp"); tmp.write_text(json.dumps(st, indent=1)); tmp.replace(ST)
                print(time.strftime("%H:%M"), "gate", a, "vs", par, json.dumps({k: e.get(k) for k in ("rank", "manifest_sha256", "gpu_games", "cpu_games",
                                                                                                    "cpu_job", "deadline_utc", "allow_gpu_served")}), flush=True)
        except Exception as e:  # noqa: BLE001
            print(time.strftime("%H:%M"), "error:", type(e).__name__, str(e)[:300], flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
