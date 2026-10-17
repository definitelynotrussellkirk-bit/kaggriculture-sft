#!/usr/bin/env python3
"""Coordinator NEW PLAN + owner 'do 128 and see what happens' (10:1x). Decision pairs T4, T1, T3 run to 128 fresh worlds x 2 seats and are read
ONCE: world-clustered 95% CI of point share (world score = mean of its two seats' points: 0/.5/1) and of margin. .50 excluded -> verdict,
else TIE (incumbent stays). Each pair = its existing frozen 32-world DECL manifest (games unread) + a 96-world extension manifest frozen now,
pooled as ONE declared 128-world bank (declaration in TPLAN.json, before any read). T2 = the three ARMS B rows vs P1 (64 games each,
selection by best mean margin) -> T3 = that B row vs n1-u10000-h08. Lanes for extension worlds, whole worlds per runtime: 3 Colab batches of
24 worlds (dispatched whole to the least-loaded Colab VM), SLURM spark 8, laptop 4, skynet 6 (skynet_game.sbatch), gpu-4090b 6 (after 6595).
usage: tplan.py freeze T4|T1|T3 [B_ROW]  |  tplan.py report"""
import fcntl, hashlib, json, math, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
R = Path("/mnt/ai/scratch/farmNEAT/league/records"); sys.path[:0] = [str(R), str(R / "gates"), str(R / "ladder")]
argv = sys.argv[1:]; sys.argv = sys.argv[:1]
import panel_league as P, gate_freezer as GF, ladder as LD
from verify_strict import verify_strict
C = R / "colab"; TP = C / "TPLAN.json"; P1 = "b-colab-u38760-d30-v9gsafe"; N1 = "n1-u10000-h08"; NT = "nflow-u24000-t4"
TP_OWN = ("tournaments", "overfit_O")          # written only by tournament.py


def save(D, own=None):
    """Locked merge write: tplan keeps the on-disk tournament.py keys; tournament.py (own=keys) replaces only its own keys on the on-disk copy."""
    with open(C / ".TPLAN.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = json.loads(TP.read_text())
        if own is None:
            for k in TP_OWN:
                if k in cur:
                    D[k] = cur[k]
            out = D
        else:
            out = cur
            for k in own:
                if k in D:
                    out[k] = D[k]
        t = TP.with_suffix(".tmp"); t.write_text(json.dumps(out, indent=1)); t.replace(TP)
    return out
BASE = {"T4": ("DECL-nbeven-h2h-nt4b-n1-u10000-h08-vs-nflow-u24000-t4", N1, NT), "T1": ("DECL-nbeven-confirm-p1-n1-u10000-h08-vs-b-colab-u38760-d30-v9gsafe", N1, P1)}
SL = (("spark", ["-p", "spark", "--mem", "2700"], 8, str(P.FARM_SBATCH)), ("laptop", ["-p", "cpu", "-w", "laptop-4060", "--mem", "1500"], 4, str(P.FARM_SBATCH)),
      ("skynet", ["-p", "gpu", "-w", "gpu-skynet", "--mem", "1500"], 6, str(R / "skynet_game.sbatch")),
      ("4090b", None, 6, None))   # gpu-4090b: gpu-scout's direct lane via records/colab batches (10:4x)


def freeze(t, b_row=None):
    D = json.loads(TP.read_text()) if TP.exists() else {"declared_utc": P.now(), "rule": __doc__.split("\n")[1:4], "pairs": {}}
    if t in D["pairs"]:
        return D["pairs"][t]
    if t == "T3":
        base_mf = {"b-colab-u51000-h09": "DECL-e1-nbeven-vs-bmain-n1-u10000-h08-vs-b-colab-u51000-h09"}.get(b_row)
        a, b = N1, b_row; nx = 96 if base_mf else 128
    else:
        base_mf, a, b = BASE[t]; nx = 96
    for n in (a, b):
        if not verify_strict(n)[0]:
            raise SystemExit(f"{n} fails strict VERIFY")
    tag = f"{t}X-{a}-vs-{b}"; mf = R / "gates" / f"{tag}.json"
    M = json.loads(mf.read_text()) if mf.exists() else P.freeze_gate(
        mf, a, b, LD.fresh(nx), "superiority", "2026-09-30T23:30:00Z", bank=tag, allow_gpu_served=False, cpu_only_reason="CPU-native lanes",
        source=f"coordinator NEW PLAN {t}: 128-world single read (owner 'do 128 and see what happens')", descriptive=f"{t} extension ({nx} worlds)",
        binding="outer archive sha256 only", size={"worlds": nx, "games": 2 * nx})
    W = list(M["worlds"]); eid = 999950000 + 10000 * ["T4", "T1", "T3"].index(t); k = 0
    for c in range(3):
        ws = W[k:k + 24] if nx == 96 else W[k:k + 32]; k += len(ws); bd = C / f"t-{t.lower()}x-{c}"
        if not (bd / "pairs.tsv").exists():
            bd.mkdir(parents=True, exist_ok=True); (bd / "PRIORITY").write_text({"T4": "0", "T1": "1", "T3": "1"}[t] + "\n"); (bd / "MANIFEST").write_text(str(mf) + "\n")
            GF.write_pairs(bd, a, b, ws, eid + 1000 * c)
    for ln, args, nw, script in SL:
        ws = W[k:k + nw]; k += nw; od = R / "slurmlane" / f"t-{t.lower()}x__{ln}"
        if ws and not (od / "JOB").exists():
            n_ = GF.write_pairs(od, a, b, ws, eid + 5000 + 2 * (k - nw))
            j = subprocess.run(["sbatch", "--parsable", *args, "-c", "1", "-t", "00:45:00", f"--array=0-{n_ - 1}" + ("%8" if ln == "skynet" else ""), "-J", "skynet-game" if ln == "skynet" else "slurm-lane",
                                "-o", str(od / "slurm-%A_%a.out"), script, str(od / "pairs.tsv"), str(od)], capture_output=True, text=True)
            jid = P.sbatch_job(j)
            if jid:
                (od / "JOB").write_text(jid + "\n")
    assert k == len(W), (k, len(W))
    D["pairs"][t] = {"a": a, "b": b, "manifests": ([f"{base_mf}.json"] if base_mf else []) + [mf.name], "worlds": 128,
                     "incumbent": {"T4": "nflow-u24000-t4 (on TIE goes first at 16:00)", "T1": "P1 stays", "T3": "n1-u10000-h08 stays"}[t],
                     "declared_utc": P.now()}
    save(D); return D["pairs"][t]


def freeze_fresh(t, a, b, incumbent, prio=1, nx=128):
    """A fresh decision pair (final test rules): 128 fresh worlds x 2 queued at once + 8 declared spare seeds; never pooled with a screen."""
    D = json.loads(TP.read_text())
    if t in D["pairs"]:
        return D["pairs"][t]
    for n in (a, b):
        if not verify_strict(n)[0]:
            raise SystemExit(f"{n} fails strict VERIFY")
    tag = f"{t}-{a}-vs-{b}"; mf = R / "gates" / f"{tag}.json"
    M = json.loads(mf.read_text()) if mf.exists() else P.freeze_gate(
        mf, a, b, LD.fresh(nx), "superiority", "2026-09-30T23:30:00Z", bank=tag, allow_gpu_served=False, cpu_only_reason="CPU-native lanes",
        source=f"final test rules {t}: 128 fresh worlds, one read on world-clustered share", descriptive=t, binding="outer archive sha256 only",
        size={"worlds": nx, "games": 2 * nx})
    sp = R / "banks" / f"{tag}-SPARES.json"
    if not sp.exists():
        sp.write_text(json.dumps({"bank": f"{tag}-SPARES", "seeds": LD.fresh(8), "rule": "declared spares for worlds lost to mixed runtimes"}, indent=1))
    W = list(M["worlds"]); eid = 999960000 + 10000 * (int(__import__('hashlib').sha256(t.encode()).hexdigest()[:8], 16) % 50); k = 0
    ncol = 96 if nx == 128 else nx - nx // 4
    for c in range(4):
        ws = W[k:k + ncol // 4]; k += len(ws); bd = C / f"t-{t.lower()}-{c}"
        if not (bd / "pairs.tsv").exists():
            bd.mkdir(parents=True, exist_ok=True); (bd / "PRIORITY").write_text(f"{prio}\n"); (bd / "MANIFEST").write_text(str(mf) + "\n")
            GF.write_pairs(bd, a, b, ws, eid + 1000 * c)
    rest = len(W) - k; sh = [10, 6, 8, 8] if rest == 32 else [rest - 3 * (rest // 4)] + [rest // 4] * 3
    for (ln, args, _, script), nw in zip(SL, sh):
        ws = W[k:k + nw]; k += nw; od = R / "slurmlane" / f"t-{t.lower()}__{ln}"
        if ws and ln == "4090b":
            bd4 = C / f"t-{t.lower()}-g4090"
            if not (bd4 / "pairs.tsv").exists():
                bd4.mkdir(parents=True, exist_ok=True); (bd4 / "PRIORITY").write_text(f"{prio}\n"); (bd4 / "MANIFEST").write_text(str(mf) + "\n")
                (bd4 / "LANE").write_text("gpu-4090b direct lane (gpu-scout)\n"); GF.write_pairs(bd4, a, b, ws, eid + 6000 + 2 * (k - nw))
            continue
        if ws and not (od / "JOB").exists():
            n_ = GF.write_pairs(od, a, b, ws, eid + 6000 + 2 * (k - nw))
            j = subprocess.run(["sbatch", "--parsable", *args, "-c", "1", "-t", "00:45:00", f"--array=0-{n_ - 1}" + ("%8" if ln == "skynet" else ""), "-J", "skynet-game" if ln == "skynet" else "slurm-lane",
                                "-o", str(od / "slurm-%A_%a.out"), script, str(od / "pairs.tsv"), str(od)], capture_output=True, text=True)
            jid = P.sbatch_job(j)
            if jid:
                (od / "JOB").write_text(jid + "\n")
    assert k == len(W)
    D["pairs"][t] = {"a": a, "b": b, "manifests": [mf.name], "worlds": nx, "incumbent": incumbent, "spares": str(sp.name), "declared_utc": P.now()}
    save(D); return D["pairs"][t]


def extend_t6(G):
    """Owner 11:5x: T6 pairs (48 fresh worlds x 2) extend in 32-world batches until the 95% world-clustered share CI excludes .50, or 128 worlds.
    A look happens only when every declared world so far is complete; each look is recorded in TPLAN.json (t6_looks)."""
    D = json.loads(TP.read_text())
    for t, p in list(D["pairs"].items()):
        if not t.startswith("T6-") or t in D.get("no_extend", []) or p.get("stopped"):
            continue
        ws = {}
        for man in p["manifests"]:
            o, _, _ = world_scores(man, G); ws.update(o)
        order = declared_order(p, D, t)
        comp = [w for w in order if len(ws.get(w, {})) == 2 and len({x[2] for x in ws[w].values()}) == 1]
        looks = D.setdefault("t6_looks", {}).setdefault(t, [])
        if len(comp) < len(order) or (looks and looks[-1]["n"] == len(order)):
            continue
        def _look():
            sh = [(ws[w][0][0] + ws[w][1][0]) / 2 for w in order]; mg = [(ws[w][0][1] + ws[w][1][1]) / 2 for w in order]
            s_, sci = ci(sh); m_, mci = ci(mg)
            dec = "PASS" if sci[0] > .5 else "FAIL" if sci[1] < .5 else ("UNDECIDED at 128 (TIE)" if len(order) >= 128 else "EXTEND +32")
            return {"n": len(order), "share": s_, "ci95": list(sci), "margin": m_, "margin_ci95": list(mci), "decision": dec, "worlds": order}
        sl = tseal(f"{t}-look-{len(order)}", _look)      # R2: the look is sealed; the extension below is bound to this sealed decision
        s_, sci, m_, mci, dec = sl["share"], sl["ci95"], sl["margin"], sl["margin_ci95"], sl["decision"]
        looks.append({"n": len(order), "share": s_, "ci95": list(sci), "margin": m_, "margin_ci95": list(mci), "decision": dec, "utc": P.now(),
                      "seal": f"seals/{t}-look-{len(order)}.json", "seal_sha256": sl["seal_sha256"]})
        save(D)
        with (Path("/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md")).open("a") as fh:
            fh.write(f"R-T6 ({time.strftime('%H:%M')} 09-30): {t} {p['a']} vs {p['b']} look at {len(order)} worlds: share {s_:.3f} [{sci[0]:.3f}, {sci[1]:.3f}], margin {m_:+,.0f} [{mci[0]:+,.0f}, {mci[1]:+,.0f}] -> {dec}\n")
        if dec == "EXTEND +32":
            k = len([m for m in p["manifests"]]); add = min(32, 128 - len(order))
            freeze_fresh(f"{t}E{k}", p["a"], p["b"], "extension", 0, add)
            D = json.loads(TP.read_text()); ext = D["pairs"].pop(f"{t}E{k}")
            D["pairs"][t]["manifests"] += ext["manifests"]; D["pairs"][t]["worlds"] = len(order) + add; D["pairs"][t].setdefault("spares", ext.get("spares"))
            save(D)


SEALS = C / "seals"


def tseal(name, compute):
    """R2: a look/read is computed ONCE and written to seals/<name>.json with a sha256; later rounds read the seal (an edited seal is refused)."""
    SEALS.mkdir(exist_ok=True); f = SEALS / f"{name}.json"
    if f.exists():
        d = json.loads(f.read_text()); body = {k: v for k, v in d.items() if k != "seal_sha256"}
        if d.get("seal_sha256") != hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest():
            raise RuntimeError(f"seal {f.name} was edited")
        return d
    r = compute()
    if r is None:
        return None
    r = dict(r, sealed_utc=P.now()); r["seal_sha256"] = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
    t = f.with_suffix(".tmp"); t.write_text(json.dumps(r, indent=1)); t.replace(f); return r


UNCLEAN = {}
STATS = {}
GPUROWS = {}


def declared_order(p, D, t):
    """Declared worlds in order (substitution manifests excluded; substitutions applied)."""
    subs = {int(k): v for k, v in ((D.get("subs") or {}).get(t) or {}).items()}
    return [subs.get(w, w) for man in p["manifests"] if man not in p.get("sub_manifests", []) for w in json.loads((R / "gates" / man).read_text())["worlds"]]


def boundaries(p, t):
    """Fallback read boundaries: T6 = cumulative declared batch sizes below the total (48/80/112); others 32/64/96."""
    if t.startswith("T6"):
        c = 0; out = []
        for man in p["manifests"]:
            if man in p.get("sub_manifests", []):
                continue
            c += len(json.loads((R / "gates" / man).read_text())["worlds"]); out.append(c)
        return sorted([x for x in out if x < p.get("worlds", 128)], reverse=True)
    return [96, 64, 32]


def repair(G):
    """Coordinator 12:2x: a declared world that is mixed-runtime, or has a seat with only an unclean game, is re-run on the same runtime family
    (the drain rule; x86 -> a Colab/4090b batch, aarch64 -> a spark array), at most 2 attempts. If it cannot finish, the first unused declared
    spare is substituted, declared in TPLAN.json BEFORE any look at a boundary that contains it; substitutions are never changed after a look."""
    D = json.loads(TP.read_text()); now = time.time(); changed = False
    for t, p in D["pairs"].items():
        if p.get("stopped"):
            continue
        mans = p["manifests"]; ws = {}; unclean = set()
        for man in mans:
            o, _, _ = world_scores(man, G); ws.update(o); unclean |= UNCLEAN.get(man, set())
        order = declared_order(p, D, t); looks = (D.get("t6_looks") or {}).get(t) or []; looked = max((l["n"] for l in looks), default=0)
        rp = D.setdefault("repairs", {}).setdefault(t, {})
        dt = (D.get("fallback") or {}).get("decision_times_utc", {}).get(t.split("-S")[0] if t.startswith("T6") else t)
        dts = datetime.fromisoformat(dt.replace("Z", "+00:00")).timestamp() if dt else None
        for pos, w in enumerate(order):
            v = ws.get(w, {}); mixed = len(v) == 2 and len({x[2] for x in v.values()}) > 1
            miss_bad = [s_ for s_ in (0, 1) if s_ not in v and (w, s_) in unclean]
            if not mixed and not miss_bad:
                continue
            r_ = rp.setdefault(str(w), {"attempts": 0, "batches": []})
            pending = any(not all((Path(b) / str(i) / "game.json").exists() for i in range(len((Path(b) / "pairs.tsv").read_text().splitlines()) - 1))
                          for b in r_["batches"] if (Path(b) / "pairs.tsv").exists())
            late = dts is not None and now >= dts - 1800
            if (r_["attempts"] >= 2 and not pending) or late:
                if pos >= looked and str(w) not in ((D.get("subs") or {}).get(t) or {}):      # substitute only before a look covers it
                    spares = json.loads((R / "banks" / p["spares"]).read_text())["seeds"] if p.get("spares") else []
                    used = set(((D.get("subs") or {}).get(t) or {}).values())
                    sp = next((x for x in spares if x not in used), None)
                    if sp is not None:
                        D.setdefault("subs", {}).setdefault(t, {})[str(w)] = sp
                        D.setdefault("subs_declared_utc", {}).setdefault(t, {})[str(w)] = P.now()
                        save(D)                     # declared before any look / dispatch
                        man = freeze_sub(t, p, sp); p.setdefault("sub_manifests", []).append(man); p["manifests"].append(man); changed = True
                continue
            if pending or r_["attempts"] >= 2:
                continue
            if mixed:
                fam = min(((x[2], s_) for s_, x in v.items()), key=lambda z: z[1])[0]   # keep seat 0's family, re-run the other seat
                seats = [s_ for s_, x in v.items() if x[2] != fam]
            else:
                fam = next(iter(v.values()))[2] if v else "x86"; seats = miss_bad
            b = dispatch_rows(t, p, w, seats, fam, r_["attempts"] + 1); r_["attempts"] += 1; r_["batches"].append(b); r_["family"] = fam; changed = True
    if changed:
        save(D)


def freeze_sub(t, p, sp):
    tag = f"{t}-SUB-{sp}-{p['a']}-vs-{p['b']}"; mf = R / "gates" / f"{tag}.json"
    M = json.loads(mf.read_text()) if mf.exists() else P.freeze_gate(
        mf, p["a"], p["b"], [sp], "superiority", "2026-09-30T23:30:00Z", bank=tag, allow_gpu_served=False, cpu_only_reason="CPU-native lanes",
        source=f"{t} declared spare substitution (coordinator 12:2x)", descriptive=f"{t} spare world", binding="outer archive sha256 only",
        size={"worlds": 1, "games": 2})
    bd = C / f"t-{t.lower()}-sub-{sp}"
    if not (bd / "pairs.tsv").exists():
        bd.mkdir(parents=True, exist_ok=True); (bd / "PRIORITY").write_text("0\n"); (bd / "MANIFEST").write_text(str(mf) + "\n")
        GF.write_pairs(bd, p["a"], p["b"], [sp], 998000000 + int(hashlib.sha256(tag.encode()).hexdigest()[:6], 16) % 900000 * 10)
    return mf.name


def dispatch_rows(t, p, w, seats, fam, k):
    """Re-run the given seats of world w on runtime family fam (x86: Colab/4090b batch; aarch64: spark array)."""
    base = 997000000 + int(hashlib.sha256(f"{t}|{w}|{k}".encode()).hexdigest()[:6], 16) % 900000 * 10
    rows = ["a_name\ta_archive\tb_name\tb_archive\tworld_seed\tepisode_id"]
    for s_ in seats:
        x, y = (p["a"], p["b"]) if s_ == 0 else (p["b"], p["a"])
        rows.append(f"{x}\t{GF.tgz(x)}\t{y}\t{GF.tgz(y)}\t{w}\t{base + s_ + 1}")
    if fam == "aarch64":
        od = R / "slurmlane" / f"t-{t.lower()}-fix-{w}-{k}__spark"; od.mkdir(parents=True, exist_ok=True); (od / "pairs.tsv").write_text("\n".join(rows) + "\n")
        j = subprocess.run(["sbatch", "--parsable", "-p", "spark", "--mem", "2700", "-c", "1", "-t", "00:45:00", f"--array=0-{len(rows) - 2}", "-J", "slurm-lane",
                            "-o", str(od / "slurm-%A_%a.out"), str(P.FARM_SBATCH), str(od / "pairs.tsv"), str(od)], capture_output=True, text=True)
        jid = P.sbatch_job(j)
        if jid:
            (od / "JOB").write_text(jid + "\n")
        return str(od)
    bd = C / f"t-{t.lower()}-fix-{w}-{k}"; bd.mkdir(parents=True, exist_ok=True); (bd / "PRIORITY").write_text("0\n")
    (bd / "MANIFEST").write_text(str(R / "gates" / p["manifests"][0]) + "\n")
    t_ = bd / ".pairs.tsv.tmp"; t_.write_text("\n".join(rows) + "\n"); t_.replace(bd / "pairs.tsv")
    return str(bd)


TSELF_RULE = ("owner 12:2x, declared before any game: at every 32-world boundary where T1 (n1-u10000-h08 vs P1) and T-SELF (b-colab-u51000-h09 vs P1) "
              "are both complete on the same first k T1 worlds, report B-vs-P1 share, n1-vs-P1 share, and the paired per-world difference (n1 - B) mean "
              "with a 95% t-CI. LB > 0 -> N B-even beats P1 more: stop every remaining B-vs-N-B-even test (ARMS B rows, T3/T-B1/T-B2 remnants, B t4w "
              "screens). UB < 0 -> B beats P1 more: report only. Otherwise TIE: continue to 128.")


def freeze_tself():
    D = json.loads(TP.read_text())
    if "T-SELF" in D["pairs"]:
        return D["pairs"]["T-SELF"]
    a, b = "b-colab-u51000-h09", "b-colab-u38760-d30-v9gsafe"
    if not (verify_strict(a)[0] and LD.GF_sha(a).startswith("e1f3387a") and verify_strict(b)[0] and LD.GF_sha(b).startswith("95e5b4e5")):
        raise SystemExit("strict check failed")
    W = [w for man in D["pairs"]["T1"]["manifests"] for w in json.loads((R / "gates" / man).read_text())["worlds"]]
    assert len(W) == 128 and len(set(W)) == 128
    tag = f"T-SELF-{a}-vs-{b}"; mf = R / "gates" / f"{tag}.json"
    M = json.loads(mf.read_text()) if mf.exists() else P.freeze_gate(
        mf, a, b, W, "superiority", "2026-09-30T23:30:00Z", bank=tag, allow_gpu_served=False, cpu_only_reason="CPU-native lanes",
        source="owner 12:2x 'do the self match': B main vs P1 on EXACTLY T1's declared worlds in T1's order (paired with T1)",
        descriptive="T-SELF (paired with T1 per world)", owner_rule=TSELF_RULE, binding="outer archive sha256 only", size={"worlds": 128, "games": 256})
    eid = 999970000; k = 0
    for c in range(4):
        ws = W[k:k + 24]; k += 24; bd = C / f"t-tself-{c}"
        if not (bd / "pairs.tsv").exists():
            bd.mkdir(parents=True, exist_ok=True); (bd / "PRIORITY").write_text("2\n"); (bd / "MANIFEST").write_text(str(mf) + "\n")
            GF.write_pairs(bd, a, b, ws, eid + 1000 * c)
    for (ln, args, _, script), nw in zip(SL, [10, 6, 8, 8]):
        ws = W[k:k + nw]; k += nw
        if ln == "4090b":
            bd4 = C / "t-tself-g4090"
            if not (bd4 / "pairs.tsv").exists():
                bd4.mkdir(parents=True, exist_ok=True); (bd4 / "PRIORITY").write_text("2\n"); (bd4 / "MANIFEST").write_text(str(mf) + "\n")
                (bd4 / "LANE").write_text("gpu-4090b direct lane (gpu-scout)\n"); GF.write_pairs(bd4, a, b, ws, eid + 6000 + 2 * (k - nw))
            continue
        od = R / "slurmlane" / f"t-tself__{ln}"
        if ws and not (od / "JOB").exists():
            n_ = GF.write_pairs(od, a, b, ws, eid + 6000 + 2 * (k - nw))
            j = subprocess.run(["sbatch", "--parsable", *args, "-c", "1", "-t", "00:45:00", f"--array=0-{n_ - 1}" + ("%8" if ln == "skynet" else ""),
                                "-J", "skynet-game" if ln == "skynet" else "slurm-lane", "-o", str(od / "slurm-%A_%a.out"), script, str(od / "pairs.tsv"), str(od)],
                               capture_output=True, text=True)
            jid = P.sbatch_job(j)
            if jid:
                (od / "JOB").write_text(jid + "\n")
    assert k == 128
    D = json.loads(TP.read_text())
    D["pairs"]["T-SELF"] = {"a": a, "b": b, "manifests": [mf.name], "worlds": 128, "incumbent": "report only unless N B-even beats P1 more",
                            "paired_with": "T1", "rule": TSELF_RULE, "declared_utc": P.now()}
    save(D); return D["pairs"]["T-SELF"]


BPANEL_RULE = ("owner 12:2x (replaces the T-SELF stop rule; nothing stops early): each B-PANEL candidate vs P1 on EXACTLY T1's 128 declared worlds in T1's "
               "order; read at each 32-world boundary where T1 and all candidates are complete on the same first k worlds: each candidate's share vs P1 "
               "(95% world-clustered CI) and every pairwise paired per-world difference (mean, 95% t-CI) among the 4; 'still moving' until 128.")
BPANEL = [("T1", "n1-u10000-h08"), ("T-SELF", "b-colab-u51000-h09"), ("BP-T4W", "b-colab-u51000-t4w"), ("BP-N11", "n1-u11000-h09")]


def freeze_bpanel_member(key, a, pre):
    D = json.loads(TP.read_text())
    if key in D["pairs"]:
        return D["pairs"][key]
    b = "b-colab-u38760-d30-v9gsafe"
    if not (verify_strict(a)[0] and LD.GF_sha(a).startswith(pre) and verify_strict(b)[0] and LD.GF_sha(b).startswith("95e5b4e5")):
        raise SystemExit(f"strict check failed for {a}")
    W = [w for man in D["pairs"]["T1"]["manifests"] for w in json.loads((R / "gates" / man).read_text())["worlds"]]
    tag = f"BPANEL-{a}-vs-{b}"; mf = R / "gates" / f"{tag}.json"
    M = json.loads(mf.read_text()) if mf.exists() else P.freeze_gate(
        mf, a, b, W, "superiority", "2026-09-30T23:30:00Z", bank=tag, allow_gpu_served=False, cpu_only_reason="CPU-native lanes",
        source="owner 12:2x B-PANEL: vs P1 on EXACTLY T1's declared worlds in T1's order", descriptive="B-PANEL member (paired with T1 per world)",
        owner_rule=BPANEL_RULE, binding="outer archive sha256 only", size={"worlds": 128, "games": 256})
    D = json.loads(TP.read_text())
    D["pairs"][key] = {"a": a, "b": b, "manifests": [mf.name], "worlds": 128, "incumbent": "report only", "paired_with": "T1", "rule": BPANEL_RULE, "declared_utc": P.now()}
    save(D); return D["pairs"][key]


def bpanel_looks(G):
    D = json.loads(TP.read_text())
    mem = [("T1", "n1-u10000-h08")] + sorted(((k, D["pairs"][k]["a"]) for k in D["pairs"] if k.startswith("BP-N") and not D["pairs"][k].get("stopped")),
                                               key=lambda x: x[0])      # owner 12:3x: B-PANEL = N B-even checkpoint ladder
    if len(mem) < 2:
        return
    order = declared_order(D["pairs"]["T1"], D, "T1"); ws = {}; wc = {}
    for k, _ in mem:
        ws[k] = {}; wc[k] = {}
        for man in D["pairs"][k]["manifests"]:
            o, _, _ = world_scores(man, G, allow_gpu=True); ws[k].update(o)
            o2, _, _ = world_scores(man, G); wc[k].update(o2)
    ok = lambda k, w: len(ws[k].get(w, {})) == 2 and len({x[2] for x in ws[k][w].values()}) == 1
    okc = lambda k, w: len(wc[k].get(w, {})) == 2 and len({x[2] for x in wc[k][w].values()}) == 1
    looks = D.setdefault("bpanel_looks", []); memkey = "|".join(n for _, n in mem); done_k = {l["k"] for l in looks if "|".join(l.get("members", [])) == memkey}
    for kb in (32, 64, 96, 128):
        if kb in done_k or not all(ok(k, w) for k, _ in mem for w in order[:kb]):
            continue
        cpu_only = {n: (lambda m_: {"share": m_[0], "ci95": list(m_[1]), "n": sum(1 for w in order[:kb] if okc(k, w))})(
            ci([(wc[k][w][0][0] + wc[k][w][1][0]) / 2 for w in order[:kb] if okc(k, w)])) for k, n in mem}
        sh = {k: [(ws[k][w][0][0] + ws[k][w][1][0]) / 2 for w in order[:kb]] for k, _ in mem}
        each = {n: (lambda m_: {"share": m_[0], "ci95": list(m_[1])})(ci(sh[k])) for k, n in mem}
        pw = {}
        for kj, nj in mem[1:]:                            # each member minus n1-u10000-h08, paired per world
            dm, dci = ci([x - y for x, y in zip(sh[kj], sh["T1"])]); pw[f"{nj} - n1-u10000-h08"] = {"mean": dm, "ci95": list(dci)}
        fams = {n: sorted({x[2] for w in order[:kb] for x in ws[k][w].values()}) for k, n in mem}
        looks.append({"k": kb, "members": [n for _, n in mem], "each_vs_p1_all_family": each, "each_vs_p1_cpu_only": cpu_only, "pairwise_all_family": pw,
                      "families": fams, "status": "final" if kb == 128 else "still moving", "utc": P.now()})
        save(D)
        with Path("/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md").open("a") as fh:
            fh.write(f"R-BPANEL ({time.strftime('%H:%M')} 09-30): boundary {kb} ({'final' if kb == 128 else 'still moving'}), all-family (decision) share vs P1: "
                     + "; ".join(f"{n} {v['share']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]" for n, v in each.items()) + " | CPU-only: "
                     + "; ".join(f"{n} {v['share']:.3f} (n {v['n']})" for n, v in cpu_only.items()) + " | pairwise (all-family) "
                     + "; ".join(f"{k_} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k_, v in pw.items()) + "\n")


def tself_looks(G):
    D = json.loads(TP.read_text())
    if "T-SELF" not in D["pairs"]:
        return
    order = declared_order(D["pairs"]["T1"], D, "T1"); w1, w2 = {}, {}
    for man in D["pairs"]["T1"]["manifests"]:
        o, _, _ = world_scores(man, G); w1.update(o)
    for man in D["pairs"]["T-SELF"]["manifests"]:
        o, _, _ = world_scores(man, G); w2.update(o)
    ok = lambda ws, w: len(ws.get(w, {})) == 2 and len({x[2] for x in ws[w].values()}) == 1
    looks = D.setdefault("tself_looks", [])
    done_k = {l["k"] for l in looks}
    for k in (32, 64, 96, 128):
        if k in done_k or not all(ok(w1, w) and ok(w2, w) for w in order[:k]):
            continue
        n1 = [(w1[w][0][0] + w1[w][1][0]) / 2 for w in order[:k]]; bm = [(w2[w][0][0] + w2[w][1][0]) / 2 for w in order[:k]]
        d = [x - y for x, y in zip(n1, bm)]; dm, dci = ci(d); s1, s1ci = ci(n1); s2, s2ci = ci(bm)
        dec = ("N B-EVEN BEATS P1 MORE: stop the remaining B-vs-N-B-even tests" if dci[0] > 0 else "B BEATS P1 MORE (report only)" if dci[1] < 0 else "TIE: continue")
        looks.append({"k": k, "n1_vs_p1": s1, "b_vs_p1": s2, "diff_mean": dm, "diff_ci95": list(dci), "decision": dec, "utc": P.now()})
        save(D)
        with Path("/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md").open("a") as fh:
            fh.write(f"R-TSELF ({time.strftime('%H:%M')} 09-30): look at {k} paired T1 worlds: n1-u10000-h08 vs P1 {s1:.3f}, b-colab-u51000-h09 vs P1 {s2:.3f}, "
                     f"paired diff (n1 - B) {dm:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}] -> {dec}\n")
        if dci[0] > 0 or dci[1] < 0:
            break


def world_scores(man, G, allow_gpu=False):
    """Eligible records of one frozen manifest (exact_gate rules: clean, exact shas, manifest worlds, started after freeze, finished by deadline)."""
    M = json.loads((R / "gates" / man).read_text()); a, b = M["challenger"], M["incumbent"]; sha = {a["name"]: a["sha256"], b["name"]: b["sha256"]}
    W = set(M["worlds"]); fz = P.utc_ts(M["frozen_at_utc"]); dl = P.utc_ts(M["deadline_utc"]); cand = {}; rej = 0; bad = set()
    for r in G:
        if {r.get("seat0"), r.get("seat1")} != set(sha) or r.get("world_seed") not in W:
            continue
        st_, fin = P.started_of(r), P.utc_ts(r.get("finished_utc")); me = 0 if r["seat0"] == a["name"] else 1
        if P.record_issues(r) or r.get("archive_sha256") != [sha[r["seat0"]], sha[r["seat1"]]] or st_ is None or st_ < fz or fin is None or fin > dl:
            rej += 1
            if P.record_issues(r):
                bad.add((r["world_seed"], me))
            continue
        rt = P.runtime_of(r)
        if rt == "gpu-served" and allow_gpu:                # B-PANEL reader (12:28): gpu is its own runtime family
            rt = "gpu"
            try:                                             # per-row receipt: was the P1 (incumbent) seat GPU-served or CPU-played?
                gs = json.loads((Path(r["provenance"]) / "game.json").read_text()).get("gpu_serve") or {}
                p1s = 0 if r["seat0"] == b["name"] else 1; srv = (gs.get("served_seats") or [None, None])[p1s]
                GPUROWS[(man, r["world_seed"], r["seat0"])] = (srv, gs.get("device"), r.get("provenance"))
            except (OSError, ValueError, KeyError, TypeError):
                GPUROWS[(man, r["world_seed"], r["seat0"])] = (None, None, r.get("provenance"))
        if rt not in ("x86", "aarch64", "gpu"):                  # Codex 12:15 #3: unknown or GPU runtime -> rejected (world stays incomplete -> drain re-run)
            rej += 1; STATS.setdefault(man, {}).setdefault("runtime_rejected", 0); STATS[man]["runtime_rejected"] += 1; continue
        m = float(r["cash"][me]) - float(r["cash"][1 - me]); fs = P.forfeit_seat(r)
        if fs is None:
            pts = 1.0 if m > 0 else .5 if m == 0 else 0.0
        else:                                            # Codex 12:15 #1: forfeit = loss for that seat; both forfeit = draw; never by cash
            STATS.setdefault(man, {}).setdefault("forfeits", 0); STATS[man]["forfeits"] += 1
            if fs == "both":
                pts, m = .5, 0.0
            else:
                lost_me = (fs == 0 and r["seat0"] == a["name"]) or (fs == 1 and r["seat1"] == a["name"])
                pts = 0.0 if lost_me else 1.0; m = -abs(m) if lost_me else abs(m)
        cand.setdefault(r["world_seed"], {}).setdefault(me, []).append((fin.timestamp(), pts, m, rt))
    out = {}
    STATS.setdefault(man, {})["conflicts"] = sum(1 for seats in cand.values() for v in seats.values() if len({(x[1], x[3]) for x in v}) > 1 and
                                                 len({x[1] for x in v if x[3] == v[0][3]}) > 1)   # Codex 12:15 #2: same family, different outcomes
    STATS[man]["outside_manifest"] = sum(1 for r in G if {r.get("seat0"), r.get("seat1")} == set(sha) and r.get("world_seed") not in W)
    for w, seats in cand.items():                       # drain rule: both seats on one runtime family; earliest-finished clean game per seat
        if len(seats) == 2:
            common = {x[3] for x in seats[0]} & {x[3] for x in seats[1]}
            if common:
                fam = min((x for s_ in (0, 1) for x in seats[s_] if x[3] in common))[3]
                out[w] = {s_: min(x for x in seats[s_] if x[3] == fam)[1:] for s_ in (0, 1)}
                continue
        out[w] = {s_: min(v)[1:] for s_, v in seats.items()}
    if not allow_gpu:
        UNCLEAN[man] = {k for k in bad if len(out.get(k[0], {})) < 2 and k[1] not in out.get(k[0], {})}
    return out, rej, len(W)


def ci(xs):
    n = len(xs); mu = sum(xs) / n if n else float("nan")
    if n < 2:
        return mu, (float("nan"), float("nan"))
    from scipy.stats import t
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)); h = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n); return mu, (mu - h, mu + h)


def named(f):
    try:
        return {l.split()[0]: l.split()[1] for l in (C / f).read_text().splitlines() if len(l.split()) >= 2 and not l.startswith("#")}
    except OSError:
        return {}


def auto(G):
    """T3 after the ARMS B selection screens complete; T-H when norules verifies (land-off fallback from 11:30 EDT); T-N and the anneal read
    when nflow names them in TN-CHALLENGER.txt / ANNEAL.txt ('name sha' lines; for ANNEAL: 'decayed name sha' and 'flat name sha')."""
    D = json.loads(TP.read_text())
    if "T3" not in D["pairs"]:
        rows = []
        for mfn in ("DECL-arms-bmain-b-colab-u51000-h09-vs-b-colab-u38760-d30-v9gsafe.json", "DECL-arms-btop-u49000-b-colab-u49000-t4w-vs-b-colab-u38760-d30-v9gsafe.json",
                    "DECL-arms-btop-u51000-b-colab-u51000-t4w-vs-b-colab-u38760-d30-v9gsafe.json"):
            f = R / "gates" / mfn
            if not f.exists():
                continue
            M = json.loads(f.read_text()); v, i, s_ = P.exact_gate(G, M)
            rows.append((M["challenger"]["name"], s_ if s_ and s_["worlds_complete"] == len(M["worlds"]) else None))
        if rows and all(r[1] for r in rows):
            best = max(rows, key=lambda r: (r[1]["p"], r[1]["margin"]))
            D.setdefault("T2_selection", {"rows": {n: {"share": s_["p"], "margin": s_["margin"], "W-D-L": f"{s_['W']}-{s_['T']}-{s_['L']}"} for n, s_ in rows},
                                          "pick": best[0], "rule": "highest point share vs P1 (tiebreak margin); selection only", "utc": P.now()})
            save(D)
            freeze_fresh("T3", N1, best[0], "n1-u10000-h08 stays (TIE -> n1)", 1)
    D = json.loads(TP.read_text())
    if "T-H" not in D["pairs"]:
        from datetime import datetime, timezone
        if verify_strict("nflow-u24000-t4-norules")[0]:
            freeze_fresh("T-H", "nflow-u24000-t4-norules", NT, "CI wholly below .5 -> keep rules; else ship norules (owner train-play preference)", 1)
        elif datetime.now(timezone.utc) >= datetime(2026, 9, 30, 15, 30, tzinfo=timezone.utc) and verify_strict("nflow-u24000-t4-landoff")[0]:
            freeze_fresh("T-H", "nflow-u24000-t4-landoff", NT, "norules not verified by 11:30: land-off is T-H; CI wholly below .5 -> keep rules", 1)
    D = json.loads(TP.read_text())
    tn = named("TN-CHALLENGER.txt")
    if "T-N" not in D["pairs"] and tn:
        n, sha = next(iter(tn.items()))
        if len(sha) == 64 and verify_strict(n, sha)[0]:
            freeze_fresh("T-N", n, NT, "nflow-u24000-t4 stays slot 1 (CI above .5 -> challenger becomes slot 1)", 1)
    an = {l.split()[0]: (l.split()[1], l.split()[2]) for l in ((C / "ANNEAL.txt").read_text().splitlines() if (C / "ANNEAL.txt").exists() else []) if len(l.split()) >= 3}
    D = json.loads(TP.read_text())
    if "ANNEAL" not in D["pairs"] and "decayed" in an and "flat" in an:
        (dn, ds), (fn, fs) = an["decayed"], an["flat"]
        if verify_strict(dn, ds)[0] and verify_strict(fn, fs)[0]:
            freeze_fresh("ANNEAL", dn, fn, "informational: CI above .5 -> decay", 2, 64)


def report():
    while True:
        try:
            G0 = P.load_gate_games(); repair(G0); extend_t6(G0); bpanel_looks(G0)
            if not (C / "PAUSE-AUTO").exists():
                auto(G0)
            D = json.loads(TP.read_text()); G = P.load_gate_games(); L = ["# T-plan decision pairs (128 worlds, one read)", "",
                "Read ONCE at the pair's declared world count (128; ANNEAL 64): world-clustered 95% CI (Student t) of point share (world score = mean of both seats: 0/.5/1) and margin. "
                ".50 excluded -> verdict; else TIE (incumbent stays). Rows below 128 worlds show only progress (no share/margin until complete).", "",
                "| pair | A vs B | complete worlds | mixed-runtime worlds | rejected | share (A) [95% CI] | margin $/world [95% CI] | verdict |", "|---|---|---|---|---|---|---|---|"]
            for t, p in D["pairs"].items():
                ws = {}; rej = 0
                for man in p["manifests"]:
                    o, r_, _ = world_scores(man, G); ws.update(o); rej += r_
                comp = {w: v for w, v in ws.items() if len(v) == 2}; mixed = sum(1 for v in comp.values() if len({x[2] for x in v.values()}) > 1)
                NW = p.get("worlds", 128)
                from datetime import datetime, timezone
                dt = (D.get("fallback") or {}).get("decision_times_utc", {}).get("T6" if t.startswith("T6") else t)
                late = dt is not None and datetime.now(timezone.utc) >= datetime.fromisoformat(dt.replace("Z", "+00:00"))
                order = declared_order(p, D, t)
                comp = {w: v for w, v in comp.items() if w in set(order) and len({x[2] for x in v.values()}) == 1}; mixed = 0
                if late and len(comp) < NW:
                    kb = next((k for k in boundaries(p, t) if all(w in comp for w in order[:k])), 0)   # largest fully complete boundary
                    prefix = order[:kb]
                    if prefix:
                        from scipy.stats import t as _t
                        sh = [(comp[w][0][0] + comp[w][1][0]) / 2 for w in prefix]; mg = [(comp[w][0][1] + comp[w][1][1]) / 2 for w in prefix]
                        n_ = len(sh); mu = sum(sh) / n_; sd = math.sqrt(sum((x - mu) ** 2 for x in sh) / (n_ - 1)); h = float(_t.ppf(0.995, n_ - 1)) * sd / math.sqrt(n_)
                        m, mci = ci(mg)
                        verdict = (f"{p['a']} BETTER" if mu - h > .5 else f"{p['b']} BETTER" if mu + h < .5 else f"TIE ({p['incumbent']})")
                        L.append(f"| {t} | {p['a']} vs {p['b']} | FALLBACK {kb}/{NW} (first {kb} declared worlds, 99% CI) | {mixed} | {rej} | {mu:.3f} [{mu - h:.3f}, {mu + h:.3f}] | "
                                 f"{m:+,.0f} [{mci[0]:+,.0f}, {mci[1]:+,.0f}] | {verdict} |")
                        continue
                if len(order) == NW and all(w in comp for w in order):      # Codex 12:15 #4: exact declared union
                    comp = {w: comp[w] for w in order}
                    _c = dict(comp)
                    def _read():
                        sh_ = [(v[0][0] + v[1][0]) / 2 for v in _c.values()]; mg_ = [(v[0][1] + v[1][1]) / 2 for v in _c.values()]
                        a_, b_ = ci(sh_), ci(mg_); return {"n": len(sh_), "share": a_[0], "ci95": list(a_[1]), "margin": b_[0], "margin_ci95": list(b_[1])}
                    rd = tseal(f"{t}-read-{NW}", _read)          # R2: the declared read is sealed once
                    s, sci, m, mci = rd["share"], rd["ci95"], rd["margin"], rd["margin_ci95"]      # reported from the seal, never recomputed
                    verdict = (f"{p['a']} BETTER" if sci[0] > .5 else f"{p['b']} BETTER" if sci[1] < .5 else f"TIE ({p['incumbent']})")
                    L.append(f"| {t} | {p['a']} vs {p['b']} | {len(comp)}/{NW} | {mixed} | {rej} | {s:.3f} [{sci[0]:.3f}, {sci[1]:.3f}] | {m:+,.0f} [{mci[0]:+,.0f}, {mci[1]:+,.0f}] | {verdict} |")
                else:
                    L.append(f"| {t} | {p['a']} vs {p['b']} | {len(comp)}/{NW} | {mixed} | {rej} | (not read) | (not read) | {'HOLD: mixed runtime' if mixed else 'running'} |")
            L += ["", "| pair | forfeits (admitted, scored as losses) | same-seat conflicts (earliest clean row used) | runtime-rejected | games outside the declared worlds (ignored) |", "|---|---|---|---|---|"]
            for t, p in D["pairs"].items():
                st_ = [STATS.get(m_, {}) for m_ in p["manifests"]]
                un = {w for m_ in p["manifests"] for w in json.loads((R / "gates" / m_).read_text())["worlds"]}; nm = {p["a"], p["b"]}
                outside = sum(1 for r in G if {r.get("seat0"), r.get("seat1")} == nm and r.get("world_seed") not in un)   # same two archives, other worlds (ignored)
                L.append(f"| {t} | {sum(x.get('forfeits', 0) for x in st_)} | {sum(x.get('conflicts', 0) for x in st_)} | {sum(x.get('runtime_rejected', 0) for x in st_)} | "
                         f"{outside} |")
            if GPUROWS:
                (C / "BPANEL-GPU-ROWS.tsv").write_text("manifest\tworld\tseat0\tp1_seat_gpu_served\tdevice\tprovenance\n" + "".join(
                    f"{m_}\t{w}\t{s0}\t{v[0]}\t{v[1]}\t{v[2]}\n" for (m_, w, s0), v in sorted(GPUROWS.items(), key=str)))
            opr = D.get("owner_directed_partial_read")
            if opr:
                L += ["", f"## {opr['label']} (recorded {opr['recorded_utc']})", "", opr["source"], ""] + [f"- {k}: {v}" for k, v in opr["reads"].items()] + \
                     ["", "Decisions: " + "; ".join(f"{k} = {v}" for k, v in opr["decisions"].items()), "", opr["note"]]
            (C / "TPLAN.md").write_text("\n".join(L + ["", f"Updated {P.now()}"]) + "\n")
        except Exception as e:  # noqa: BLE001
            print(time.strftime("%H:%M"), "error", type(e).__name__, str(e)[:200], flush=True)
        time.sleep(120)


if __name__ == "__main__":
    if argv and argv[0] == "freeze":
        print(freeze(argv[1], argv[2] if len(argv) > 2 else None))
    elif argv and argv[0] == "bpanel":
        print(freeze_bpanel_member(argv[1], argv[2], argv[3]))
    elif argv and argv[0] == "tself":
        print(freeze_tself())
    elif argv and argv[0] == "fresh":                     # fresh T A B INCUMBENT [PRIO] [NWORLDS]
        print(freeze_fresh(argv[1], argv[2], argv[3], argv[4], int(argv[5]) if len(argv) > 5 else 1, int(argv[6]) if len(argv) > 6 else 128))
    else:
        report()
