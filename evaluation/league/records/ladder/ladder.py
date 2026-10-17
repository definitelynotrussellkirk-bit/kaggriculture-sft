#!/usr/bin/env python3
"""Fail-forward ladder, gate side, R3 rules (LADDER.md, adopted 02:03 EDT 09-30 from EASY-CHECK-LADDER-R2 PROPOSED-R3 + two local choices).

Arms (rule 8): B, O, N-beven, N-flow; each has its own saved X / current X' SHAs in DECISION-SET.json 'ladder'.
Candidate round for X'' (ladder/REQUESTS.tsv 'candidate' or 'confirm'):
  screen  16 fresh worlds vs X'  - selection only, never promotes (rule 4)
  confirm 32 fresh worlds vs X'  - superiority at the arm's sequential alpha_i                 -> branch winner
  direct  32 fresh worlds vs X   - superiority at alpha_i (skipped when X' == X: the confirm IS the direct gate)   (rule 3)
  audit   32 fresh one-use worlds vs each reference component - bounded regression: one-sided t p(mean <= -1,500) <= alpha_i (rule 5, 6)
  commit  compare-and-swap on the ticket (arm_id, control_sha, saved_sha, reference_version); mismatch = STALE -> rerun on new worlds (rule 2)
Stall promotion X' -> X (2 counted failed confirms after the line budget, or 'promote-saved'): direct X' vs live X + audit, CAS; else HOLD (rule 1).
Sequential alpha per arm: alpha_i = .10 / (i (i + 1)), i = the arm's statistical attempt index (confirm/direct/audit component) (rule 4).
One statistical attempt per (candidate_sha, opponent_sha, reference_version) (rule 4). FLAGGED lines (failure before budget) may keep
producing candidates but their winners never become X/X' (rule 7). ROLLBACK after budget. Frozen gates keep their verdicts.
"""
import hashlib, json, math, random, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
R = Path("/mnt/ai/scratch/farmNEAT/league/records"); sys.path[:0] = [str(R), str(R / "gates")]
import panel_league as P
import gate_freezer as GF
H = R / "ladder"; ST = H / "state.json"; DS = R / "DECISION-SET.json"
LEDGER = Path("/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md")
KILL = Path("/mnt/ai/scratch/farmNEAT/league/records/ladder/COMMITS-DISABLED")   # kill switch (ASTRA CHECK-0300-GATES, coordinator 04:1x)
ALPHA_TOTAL = 0.10
ALPHA_AUDIT = 0.05                  # coordinator 02:4x: reference-audit components (bounded regression) fixed, not charged to the arm budget
REF_NAMES = ["base1-b-u25000-v8g-a1-ema", "oldarch1-v4bg2-u79000-cd-ema", "tape-dsm-115407073-lw", "tape-boey-115400455-lw", "tape-decem-115345668-lw"]
ARMS = ("B", "O", "N-beven", "N-flow")
GPU_PRIO = 20
SCREEN_WORLDS = 32                  # owner 04:2x: screens 32 worlds x 2 seats = 64 games
GATE_SIZES = (64, 48, 32)           # owner 04:1x/04:2x: confirms, direct gates, reference components 64 where CPU allows before use, else >= 32
USE_TIMES_UTC = ("2026-09-30T14:30:00Z", "2026-09-30T18:30:00Z", "2026-09-30T22:30:00Z")   # G1, G2, K uploads (10:30 / 14:30 / 18:30 EDT)
CPU_RATE, GPU_RATE = 87.0, 129.0    # measured games/h (CPU pool, GPU-served pool), 09-30 early
GATE_JOBS = ("ladder-top", "ladder-n", "ladder-gate", "u0300-gate", "land-gate", "b31f-fresh48f", "salvage-ab", "nflow-panel")
BR_MARGIN = -1500.0


def ref_ready(n):
    """References: deployed tar.gz is enough for the static tapes (no VERIFY.json; played in the league before); models need VERIFY."""
    return GF.ready(n) or (n.startswith("tape-") and (GF.DEP / n / f"{n}.tar.gz").is_file())


def IDENTITY_REQUIRED():
    try:
        return json.loads(DS.read_text())["ladder"].get("_identity_required", [])
    except (OSError, ValueError, KeyError):
        return ["__unreadable__"]


def alpha_i(i):
    return ALPHA_TOTAL / (i * (i + 1))


def arm_of(name):
    if name.startswith(("nflow", "n-flow", "n2-")):
        return "N-flow"
    if name.startswith("n1-"):
        return "N-beven"
    if name.startswith(("oldarch1-", "obroad-", "ogp-", "osvc-")):
        return "O"
    if name.startswith(("base1-b-", "b-colab-", "b31fx-")):
        return "B"
    return None


def line_of(L, name):
    return max((p for p in ((L.get("_budget") or {}).get("lines") or {}) if name.startswith(p)), key=len, default=name)


def decide_sup(s, a):
    """Superiority at alpha a: complete, no individually tied game, sweep-sign one-sided p <= a and point share >= .5. Rule R3.2 (coordinator
    04:3x, declared before any affected result was read): if the sweep-sign verdict and the two-sided t95 margin interval disagree
    (sweep pass with the interval wholly below 0, or sweep fail with the interval wholly above 0) the outcome is HOLD."""
    if s.get("tie_games"):
        return "HOLD", "tied games: sweep endpoint undefined"
    ok = s["sweep_sign_p"] <= a and s["p"] >= 0.5
    mc = s.get("margin_ci") or [float("nan")] * 2
    if all(isinstance(x, (int, float)) and math.isfinite(x) for x in mc) and ((ok and mc[1] < 0) or (not ok and mc[0] > 0)):
        return "HOLD", (f"R3.2: sweep-sign p {s['sweep_sign_p']:.4f} ({'pass' if ok else 'fail'} at alpha_i {a:.4f}) disagrees with the "
                        f"two-sided t95 margin interval [{mc[0]:+,.0f}, {mc[1]:+,.0f}] -> HOLD")
    return ("PASS" if ok else "FAIL"), (f"sweeps {s['sweeps']}/{s['splits']}/{s['sweep_losses']}, sweep-sign p {s['sweep_sign_p']:.4f} (need <= alpha_i {a:.4f}: "
                                        f"{'pass' if s['sweep_sign_p'] <= a else 'FAIL'}), points {s['p']:.3f} (need >= .5: {'pass' if s['p'] >= 0.5 else 'FAIL'})")


def br_pvalue(s):
    """One-sided t p-value for H0: mean world margin <= -1,500 (bounded regression), from the frozen two-sided t95 interval."""
    from scipy.stats import t as T
    n = s["worlds_complete"]
    if n < 2:
        return 1.0
    half = (s["margin_ci"][1] - s["margin_ci"][0]) / 2
    se = half / T.ppf(0.975, n - 1) if half > 0 else 1e-9
    return float(1 - T.cdf((s["margin"] - BR_MARGIN) / se, n - 1))


def decide_screen(s):
    """Single screen rule (DECISION-SET.ladder._rule_version R3.2, PRO CHECK-0300-GATES #1): POSITIVE = complete, no tied game, sweeps >
    sweep losses and point share >= .5; R3.2: a positive screen whose t95 margin interval lies wholly below 0, or a negative one whose
    interval lies wholly above 0, is HOLD. Screens select only; they never promote."""
    if s.get("tie_games"):
        return "HOLD", "tied games"
    pos = s["sweeps"] > s["sweep_losses"] and s["p"] >= 0.5
    mc = s.get("margin_ci") or [float("nan")] * 2
    if all(isinstance(x, (int, float)) and math.isfinite(x) for x in mc) and ((pos and mc[1] < 0) or (not pos and mc[0] > 0)):
        return "HOLD", f"R3.2: screen {'positive' if pos else 'negative'} but t95 margin interval [{mc[0]:+,.0f}, {mc[1]:+,.0f}] disagrees"
    return ("POSITIVE" if pos else "NEGATIVE"), f"sweeps {s['sweeps']}/{s['splits']}/{s['sweep_losses']}, points {s['p']:.3f}"


def decide_br(s, a):
    if s.get("tie_games"):
        return "HOLD", "tied games"
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [s.get("margin", float("nan")), *(s.get("margin_ci") or [float("nan")] * 2)]):
        return "HOLD", "margin evidence missing or non-finite (fails closed)"
    pv = br_pvalue(s); one = pv <= a; t95 = s["margin_ci"][0] > BR_MARGIN
    txt = (f"bounded regression: mean {s['margin']:+,.0f}/world, one-sided p(mean <= -1,500) {pv:.4f} (need <= alpha_i {a:.4f}), "
           f"two-sided t95 LB {s['margin_ci'][0]:+,.0f} (need > -1,500)")
    if one != t95:                                                 # ASTRA CHECK-0300-GATES: the two frozen readings disagree -> no rule choice
        return "HOLD", txt + " -> readings disagree: HOLD until a declared rule version resolves the boundary"
    return ("PASS" if one else "FAIL"), txt


def tier_rev(A):
    """Live revision of an arm's tiers: (saved sha, control sha, reference_version)."""
    if not isinstance(A, dict) or not A.get("control"):
        return None
    return (A["saved"]["sha256"], A["control"]["sha256"], A.get("reference_version"))


def write_ladder(L, loaded, ds=None):
    """ASTRA CHECK-0300-GATES P1 (atomic CAS): under an exclusive flock, re-read DECISION-SET, and for each arm write our state only if the
    live tier revision still equals the one we loaded this round (else keep the live arm and log a conflict); non-arm keys (_budget,
    _references, _auto, ...) are never overwritten by stale local copies. Write temp + rename while holding the lock."""
    import fcntl
    ds = Path(ds or DS)
    with open(ds.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        D2 = json.loads(ds.read_text()); live = D2.setdefault("ladder", {}); conflicts = []
        for a in ARMS:
            if a not in L:
                continue
            if tier_rev(live.get(a)) != loaded.get(a):
                conflicts.append(a); merge_acct(live.setdefault(a, {}), L[a]); continue      # red-team 04:3x #4: keep the round's accounting
            new_ = L[a]; merge_acct(new_, live.get(a) or {}); live[a] = new_
        tmp = ds.with_suffix(".tmp"); tmp.write_text(json.dumps(D2, indent=1)); tmp.replace(ds)
    if conflicts:
        ledger(f"write-back conflict: live tiers of {conflicts} changed since this round loaded them; kept the live tiers and merged this round's tickets, tuples and alpha into them")
    return conflicts


def cas(ticket, arm):
    """Rule 2: the live arm must still equal the ticket."""
    live = (arm["control"]["sha256"], arm["saved"]["sha256"], arm.get("reference_version"))
    return "OK" if (ticket.get("control_sha"), ticket.get("saved_sha"), ticket.get("reference_version")) == live else "STALE"


_psha = {}


_psha2 = {}


def payload_sha(n):
    """Policy identity = sha256 over (path, sha256) of every archive member except MANIFEST.json (provenance only; main.py just checks it
    exists): weights AND harness. Same weights + same harness under two names share it (obroad-u82000-0300 and
    oldarch1-tu8-obroad-u82000-cd-ema); a q4off build and its plain twin do not (agent.py differs). Falls back to the tar sha."""
    if n not in _psha2:
        import tarfile
        try:
            h = hashlib.sha256()
            with tarfile.open(GF.DEP / n / f"{n}.tar.gz", "r:gz") as tf:
                mem = sorted((m for m in tf.getmembers() if m.isfile() and m.name.lstrip("./") != "MANIFEST.json"), key=lambda m: m.name)
                if not mem:
                    raise ValueError("no members")
                for m in mem:
                    d = hashlib.sha256(); fh = tf.extractfile(m)
                    for blk in iter(lambda: fh.read(1 << 20), b""):
                        d.update(blk)
                    h.update(f"{m.name.lstrip('./')}\0{d.hexdigest()}\n".encode())
            _psha2[n] = h.hexdigest()
        except (OSError, tarfile.TarError, ValueError):
            _psha2[n] = GF_sha(n)
    return _psha2[n]


def payload_sha_v1(n):
    """Pre-v2 key (largest member only); used only to recognise tuples recorded before the harness-identity migration."""
    if n not in _psha:
        f = GF.DEP / n / f"{n}.tar.gz"
        try:
            lst = subprocess.run(["tar", "-tvzf", str(f)], capture_output=True, text=True, timeout=300).stdout.splitlines()
            big = max((l.split() for l in lst if l.strip()), key=lambda x: int(x[2]))[-1]
            _psha[n] = hashlib.sha256(subprocess.run(["tar", "-xzf", str(f), "-O", big], capture_output=True, timeout=300).stdout).hexdigest()
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
            _psha[n] = GF_sha(n)
    return _psha[n]


def tuple_key(cand_sha, opp_sha, ref_ver):
    return f"{cand_sha}|{opp_sha}|{ref_ver}"


def gate_backlog():
    """(CPU gate-class tasks queued or running, GPU-served gate rows not yet played); None when squeue cannot be read (fail closed)."""
    q = subprocess.run(["squeue", "-h", "-r", "-t", "PD,R", "-o", "%j"], capture_output=True, text=True)
    if q.returncode != 0:
        return None
    cpu = sum(1 for j in q.stdout.split() if j in GATE_JOBS); gpu = 0
    for d in (R / "gpu").iterdir():
        try:
            if (d / "STOP").exists() or int((d / "PRIORITY").read_text().strip()) > GPU_PRIO:
                continue
            rows = sum(1 for l in (d / "pairs.tsv").read_text().splitlines()[1:] if l.strip())
        except (OSError, ValueError):
            continue
        gpu += max(0, rows - len(list(d.glob("*/game.json"))))
    return cpu, gpu


def size_for(role, gpu_ok, now=None, backlog="live"):
    """Worlds for a new gate. Screens: SCREEN_WORLDS. Others: the use time the gate makes at 32 worlds fixes the target; take the largest
    of 64/48/32 that still finishes before that use time (never push a gate past a use it would make); 32 when nothing fits or the queue
    cannot be read."""
    if role == "screen":
        return SCREEN_WORLDS, {"worlds": SCREEN_WORLDS, "games": 2 * SCREEN_WORLDS, "rule": "owner 04:2x: screens 32 worlds x 2 seats"}
    now = now or datetime.now(timezone.utc); b = gate_backlog() if backlog == "live" else backlog
    info = {"rule": "owner 04:1x/04:2x: 64 worlds where CPU allows before the use time, otherwise >= 32", "backlog_cpu_gpu": b,
            "rates_cpu_gpu_per_h": [CPU_RATE, GPU_RATE]}
    if b is None:
        return 32, dict(info, worlds=32, games=64, why="gate backlog unreadable: minimum size")
    def eta(n):
        g = round(n * GF.GPU_WORLDS / 32) if gpu_ok else 0
        return max((b[0] + 2 * (n - g)) / CPU_RATE, (b[1] + 2 * g) / GPU_RATE if g else 0.0)
    uses = [datetime.fromisoformat(u.replace("Z", "+00:00")) for u in USE_TIMES_UTC]
    t32 = now.timestamp() + eta(32) * 3600
    use = next((u for u in uses if u.timestamp() >= t32), None)
    if use is None:
        return 32, dict(info, worlds=32, games=64, eta_h={32: round(eta(32), 2)}, why="no use time reachable even at 32 worlds")
    n = next(n for n in GATE_SIZES if now.timestamp() + eta(n) * 3600 <= use.timestamp())
    return n, dict(info, worlds=n, games=2 * n, use_by_utc=use.strftime("%Y-%m-%dT%H:%M:%SZ"), eta_h={k: round(eta(k), 2) for k in GATE_SIZES})


WINDOWS = (("G1", "2026-09-30T12:30:00Z", "2026-09-30T14:30:00Z"), ("G2", "2026-09-30T16:30:00Z", "2026-09-30T18:30:00Z"),
           ("K", "2026-09-30T20:30:00Z", "2026-09-30T22:30:00Z"))      # coordinator 05:1x (PRO GPU-USES): (window, freeze-by, card) UTC
COLAB_RATE = 400.0                  # games/h of the Colab #3 x86 lane when it is playing (39 slots, 93-290 s/game)


def _ts(x):
    return datetime.fromisoformat(x.replace("Z", "+00:00"))


def colab_live(now=None):
    t = (now or datetime.now(timezone.utc)).timestamp() - 1800
    return any(g.stat().st_mtime > t for g in (R / "colab").glob("*/*/game.json"))


def book(L, st, arm, cand, key, now=None, backlog="live", colab=None):
    """One frozen candidate per arm per window; the candidate's full chain (screen, confirm, direct when X' != X, every declared reference
    component) is booked against the window's card time at freeze. Returns (window, text) or (None, text)."""
    now = now or datetime.now(timezone.utc)
    w = next((x for x in WINDOWS if now <= _ts(x[1])), None)
    if w is None:
        return None, "no decision window left (K freeze-by 16:30 EDT passed)"
    other = sorted((k for k, c in st.get("cands", {}).items() if k != key and c.get("arm") == arm and c.get("window") == w[0]),
                   key=lambda k: st["cands"][k].get("stage") in ("rejected", "hold", "done"))
    if other:
        return None, f"window {w[0]} already has {arm} candidate {other[0]} (one per arm per window); waits for the next window"
    A = L[arm]; refs, _ = declared_reference_components(L, arm, cand); g_ok = GF.gpu_ok(cand, A["control"]["name"])
    n_c = GATE_SIZES[0]; comps = 1 + (1 if A["control"]["name"] != A["saved"]["name"] else 0) + len(refs)
    worlds = SCREEN_WORLDS + n_c * comps
    b = gate_backlog() if backlog == "live" else backlog
    if b is None:
        return None, "gate backlog unreadable: cannot book the chain (fails closed)"
    cpu_rate = CPU_RATE + (COLAB_RATE if (colab_live(now) if colab is None else colab) else 0.0)
    g = round(worlds * GF.GPU_WORLDS / 32) if g_ok else 0
    eta = max((b[0] + 2 * (worlds - g)) / cpu_rate, (b[1] + 2 * g) / GPU_RATE if g else 0.0)
    card = _ts(w[2]); fin = now.timestamp() + eta * 3600
    txt = (f"{w[0]}: chain {2 * worlds} games (screen {SCREEN_WORLDS}w + {comps} x {n_c}w), backlog CPU {b[0]} / GPU {b[1]}, "
           f"ETA {eta:.1f} h -> {datetime.fromtimestamp(fin, timezone.utc):%H:%MZ} vs card {w[2][11:16]}Z")
    return (w[0], "booked " + txt) if fin <= card.timestamp() else (None, "cannot finish by its card, not started: " + txt)


ACCT = ("tuples", "attempt_gates", "audit_gates", "attempts", "audits", "alpha_spent", "flagged_lines", "flagged_candidates",
        "flagged_payloads", "rollbacks", "index_collisions", "acct_version", "tuples_pre_v2")


def acct_charge(A, ticket, tk):
    """Idempotent accounting of one ticket: its tuple and its attempt (charged, alpha_i) or audit index are recorded exactly once; a
    re-freeze that supersedes a void keeps the same index. alpha_spent is derived from the recorded attempt indices."""
    if tk and tk not in A.setdefault("tuples", []):
        A["tuples"].append(tk)
    if ticket is None:
        return
    key, idx = ("attempt_gates", ticket["attempt_index"]) if ticket.get("charged") else ("audit_gates", ticket.get("audit_index"))
    if idx is None:
        return
    g = A.setdefault(key, {}); k = str(idx)
    if k not in g or g[k] == "pre-v2" or ticket.get("supersedes_void"):
        g[k] = ticket["gate_id"]
    elif g[k] != ticket["gate_id"]:
        c_ = {"index": k, "kind": key, "gates": [g[k], ticket["gate_id"]]}
        if c_ not in A.setdefault("index_collisions", []):
            A["index_collisions"].append(c_)
    if key == "attempt_gates":
        A["attempts"] = max(A.get("attempts") or 0, idx); A["alpha_spent"] = round(sum(alpha_i(int(x)) for x in g), 12)
    else:
        A["audits"] = max(A.get("audits") or 0, idx)


def merge_acct(dst, src):
    """Union of the accounting fields of src into dst (tiers untouched)."""
    pre = (src.get("acct_version") or 0) < (dst.get("acct_version") or 0)     # a pre-migration copy: its old-form keys live in tuples_pre_v2
    for k in ("tuples", "flagged_lines", "flagged_candidates", "flagged_payloads", "rollbacks", "index_collisions"):
        if pre and k in ("tuples", "flagged_payloads"):
            continue
        for x in src.get(k) or []:
            if x not in dst.setdefault(k, []):
                dst[k].append(x)
    for k in ("attempt_gates", "audit_gates"):
        for i, gid in (src.get(k) or {}).items():
            d = dst.setdefault(k, {})
            if i not in d or d[i] == "pre-v2":
                d[i] = gid
            elif gid not in (d[i], "pre-v2"):
                c_ = {"index": i, "kind": k, "gates": [d[i], gid]}
                if c_ not in dst.setdefault("index_collisions", []):
                    dst["index_collisions"].append(c_)
    for k in ("attempts", "audits", "acct_version"):
        if src.get(k) is not None:
            dst[k] = max(dst.get(k) or 0, src[k])
    if dst.get("attempt_gates"):
        dst["alpha_spent"] = round(sum(alpha_i(int(x)) for x in dst["attempt_gates"]), 12)
    elif src.get("alpha_spent") is not None:
        dst["alpha_spent"] = max(dst.get("alpha_spent") or 0.0, src["alpha_spent"])
    if src.get("tuples_pre_v2") and not dst.get("tuples_pre_v2"):
        dst["tuples_pre_v2"] = src["tuples_pre_v2"]


def stage_key(M):
    """v2 tuple key of a frozen ladder manifest (candidate, opponent, reference_version by harness identity)."""
    t = M.get("ticket") or {}
    return tuple_key(payload_sha(M["challenger"]["name"]), payload_sha(M["incumbent"]["name"]), t.get("reference_version"))


def migrate_acct(L, st, log=None):
    """One-time (acct_version 2): rebuild each arm's tuples as v2 harness-identity keys from the frozen ticketed manifests the ladder
    launched; old tar-sha and weights-only keys are kept in tuples_pre_v2 and any old key no manifest explains stays in tuples. Attempt
    indices 1..attempts become attempt_gates (gate ids from manifests, else 'pre-v2'); flagged payloads are recomputed."""
    log = log or ledger; out = {}
    for a in ARMS:
        A = L.get(a)
        if not isinstance(A, dict) or not A.get("control") or A.get("acct_version") == 2:
            continue
        old = list(A.get("tuples") or []); new = []; explained = set(); ag = {str(i): "pre-v2" for i in range(1, (A.get("attempts") or 0) + 1)}; dg = {}
        for tag, x in st.get("stages", {}).items():
            try:
                M = json.loads(Path(x["manifest"]).read_text())
            except (OSError, ValueError):
                continue
            t = M.get("ticket")
            if not t or t.get("arm_id") != a:
                continue
            c_, o_ = M["challenger"]["name"], M["incumbent"]["name"]; rv = t.get("reference_version")
            for f in (GF_sha, payload_sha_v1):
                explained.add(tuple_key(f(c_), f(o_), rv))
            k = stage_key(M)
            if k not in new:
                new.append(k)
            if t.get("charged") and t.get("role") != "ref" and t.get("attempt_index"):
                ag[str(t["attempt_index"])] = t["gate_id"]
            elif t.get("role") == "ref" and t.get("audit_index"):
                dg[str(t["audit_index"])] = t["gate_id"]
        unexplained = [k for k in old if k not in explained]
        A["tuples_pre_v2"] = old; A["tuples"] = new + unexplained
        spent0 = A.get("alpha_spent"); A["attempt_gates"] = ag
        if dg:
            A["audit_gates"] = dg; A["audits"] = max(A.get("audits") or 0, max(int(i) for i in dg))
        A["alpha_spent"] = round(sum(alpha_i(int(i)) for i in ag), 12)
        A["flagged_payloads"] = [payload_sha(n) for n in A.get("flagged_candidates") or []]
        A["acct_version"] = 2
        out[a] = {"tuples_old": len(old), "tuples_v2": len(new), "unexplained_kept": len(unexplained), "alpha_spent_before": spent0,
                  "alpha_spent_after": A["alpha_spent"]}
        log(f"{a} accounting migrated to v2 harness-identity tuple keys: {out[a]}")
    return out


def reconcile(L, st):
    """Round start: every v2 ticket the ladder reserved (pending launches) or launched (frozen manifests) is in its arm's accounting."""
    for c in st.get("cands", {}).values():
        for k, pd in list(c.items()):
            if k.startswith("pending_") and isinstance(pd, dict):
                for p_ in ([pd] if "tag" in pd else list(pd.values())):
                    t = p_.get("ticket")
                    if t and t.get("arm_id") in L:
                        acct_charge(L[t["arm_id"]], t, t.get("tuple_key"))
    for x in st.get("stages", {}).values():
        try:
            t = json.loads(Path(x["manifest"]).read_text()).get("ticket")
        except (OSError, ValueError):
            continue
        if t and t.get("acct") == 2 and t.get("arm_id") in L:
            acct_charge(L[t["arm_id"]], t, t.get("tuple_key"))


def stage_for_tuple(st, tk, role, A=None):
    """Tag of a live frozen ladder gate that already holds this tuple (so a later chain reuses it instead of re-running it). With A,
    only a gate whose ticket still passes CAS against the live arm is reusable (a STALE one is a spent generation)."""
    for tag, x in st.get("stages", {}).items():
        try:
            M = json.loads(Path(x["manifest"]).read_text())
        except (OSError, ValueError):
            continue
        t = M.get("ticket") or {}
        if t.get("role") == role and (t.get("tuple_key") or stage_key(M)) == tk and (A is None or cas(t, A) == "OK"):
            return tag
    return None


def generation_key(tk, A):
    """PRO CHECK-LADDER-R4: after a tier change (STALE) the same (candidate, opponent, reference_version) gets one fresh generation per
    tier revision, instead of colliding with the spent attempt forever."""
    return f"{tk}|gen:{A['control']['sha256'][:16]}.{A['saved']['sha256'][:16]}"


def resolve_tuple(A, st, tk, role):
    """('new', key) to launch, ('reuse', tag) for a current frozen gate of this tuple, ('attempted', None) when spent."""
    tuples = A.setdefault("tuples", [])
    if tk not in tuples:
        return "new", tk
    old = stage_for_tuple(st, tk, role, A)
    if old:
        return "reuse", old
    gk = generation_key(tk, A)
    if gk not in tuples:
        return "new", gk
    old = stage_for_tuple(st, gk, role, A)
    return ("reuse", old) if old else ("attempted", None)


def save_state(st):
    tmp = ST.with_suffix(".tmp"); tmp.write_text(json.dumps(st, indent=1)); tmp.replace(ST)


def ledger(line):
    with open(LEDGER, "a") as f:
        f.write(f"R-LADDER3 ({time.strftime('%H:%M')} {time.strftime('%m-%d')}): {line}\n")


def fmt(s):
    return (f"W-D-L {s['W']}-{s['T']}-{s['L']}, sweeps/splits/losses {s['sweeps']}/{s['splits']}/{s['sweep_losses']}, points {s['p']:.3f}, "
            f"margin {s['margin']:+,.0f} [{s['margin_ci'][0]:+,.0f}, {s['margin_ci'][1]:+,.0f}]") if s else "-"


def budget(L, cand, now=None):
    B = L.get("_budget") or {}; now = now or datetime.now(timezone.utc)
    pre = max((p for p in (B.get("lines") or {}) if cand.startswith(p)), key=len, default=None)
    if pre is None:
        return True, "line budget: no line config (treated as reached)"
    c = {**(B.get("default") or {}), **B["lines"][pre]}
    t0 = datetime.fromisoformat(c["start_utc"].replace("Z", "+00:00")); hrs = (now - t0).total_seconds() / 3600
    m = re.search(r"-u(\d+)", cand); upd = int(m.group(1)) - int(c.get("start_update", 0)) if m else None
    if c.get("until_utc"):
        ok = now >= datetime.fromisoformat(c["until_utc"].replace("Z", "+00:00"))
        return ok, f"line {pre}: no stop before {c['until_utc']} ({'reached' if ok else 'not reached'})"
    if c.get("steps_per_update") and upd is not None:
        steps = upd * c["steps_per_update"]; ok = hrs >= c.get("min_hours", 4) and steps >= c["min_steps"]
        return ok, f"line {pre}: {hrs:.1f} h and {steps:,} steps vs min {c.get('min_hours', 4)} h AND {c['min_steps']:,} steps ({'reached' if ok else 'not reached'})"
    ok = hrs >= c.get("min_hours", 4) and (upd is None or upd >= c.get("min_updates", 8000))
    return ok, (f"line {pre}: {hrs:.1f} h and {upd if upd is not None else '?'} updates vs min {c.get('min_hours', 4)} h AND {c.get('min_updates', 8000)} "
                f"updates ({'reached' if ok else 'not reached'})")


def fresh(n):
    known = GF.used_seeds(); rng = random.Random(time.time_ns()); s = []
    while len(s) < n:
        x = rng.randrange(40_000_000, 2_000_000_000)
        if x not in known and x not in s:
            s.append(x)
    return s


def launch(tag, a, b, n_worlds, kind, st, ticket=None, prio=GPU_PRIO, size=None):
    mf = R / "gates" / f"LADDER-{tag}.json"
    if not mf.exists():                                   # a new freeze never reuses a voided/cancelled predecessor's JOB or pairs
        for d_ in (H / "run" / tag, H / "run" / f"{tag}__x86", R / "gpu" / f"1ladder-{tag}"[:200]):
            if d_.exists():
                d_.rename(d_.with_name(d_.name + time.strftime(".stale-%H%M%S")))
    if mf.exists():
        M = json.loads(mf.read_text())
    else:
        g_ok = GF.gpu_ok(a, b)
        M = P.freeze_gate(mf, a, b, fresh(n_worlds), kind, GF.deadline_for(datetime.now(timezone.utc)), bank=f"LADDER-{tag}",
                          allow_gpu_served=g_ok, **({"gpu_parity_extra": GF.PARITY_EXTRA} if g_ok else {"cpu_only_reason": "no Spark parity for this family"}),
                          source="LADDER.md R3 (02:03 EDT 09-30)", ladder_stage=tag, **({"ticket": ticket} if ticket else {}),
                          **({"size": size} if size else {}))
    GF.validate_dispatch_manifest(M, a, b, kind)
    if M.get("games_seen_at_freeze") != 0:
        return None
    W = list(M["worlds"]); eid0 = 994500000 + 1000 * (len(st["stages"]) + 1); d = H / "run" / tag; cw = W
    if M.get("allow_gpu_served"):
        k = round(len(W) * GF.GPU_WORLDS / 32); gd = R / "gpu" / f"1ladder-{tag}"[:200]
        if not (gd / "pairs.tsv").exists():
            GF.write_pairs(gd, a, b, W[:k], eid0, prio=prio)
        cw = W[k:]
    ks = round(len(cw) * GF.SPARK_SHARE); jids = []
    for part, ws_, dd in (("spark", cw[:ks], d), ("cpu", cw[ks:], H / "run" / f"{tag}__x86")):
        if not ws_:
            continue
        if (dd / "JOB").exists():
            jids.append((dd / "JOB").read_text().strip()); continue
        n = GF.write_pairs(dd, a, b, ws_, eid0 + (500 if part == "spark" else 700))
        pin_ = ["-w", "laptop-4060"] if (prio == 0 and part == "cpu") else []    # top priority: skip slow cpu-hub for B games
        j = subprocess.run(["sbatch", "--parsable", "-p", part, *pin_, "-c", "1", "--mem", ("2700" if part == "spark" else "1500"), "-t", "00:45:00", f"--array=0-{n - 1}",
                            "-J", "ladder-top" if prio == 0 else ("ladder-n" if tag.startswith(("N-beven-", "N-flow-", "N-S-", "N-C-")) else "ladder-gate"),
                            "-o", str(dd / "slurm-%A_%a.out"), P.FARM_SBATCH, str(dd / "pairs.tsv"), str(dd)], capture_output=True, text=True)
        jid_ = P.sbatch_job(j)
        if jid_ is None:
            return None
        (dd / "JOB").write_text(jid_ + "\n"); jids.append(jid_)
    st["stages"][tag] = {"manifest": str(mf), "job": ",".join(jids), "a": a, "b": b, "kind": kind, "utc": P.now()}
    return str(mf)


def gate_result(tag, st, G):
    M = json.loads(Path(st["stages"][tag]["manifest"]).read_text()); v, info, s = P.exact_gate(G, M)
    return M, v, info, s


def new_ticket(L, arm, role, cand, opp):
    """Charged tickets (confirm / direct-vs-saved) take the arm's next sequential alpha_i; reference-audit components ('ref') are
    bounded-regression guards at fixed ALPHA_AUDIT and are not charged (coordinator 02:4x)."""
    A = L[arm]
    if role == "ref":
        A["audits"] = (A.get("audits") or 0) + 1
        return {"arm_id": arm, "gate_id": f"{arm}-{role}-{cand}-vs-{opp}-r{A['audits']}", "candidate_sha": GF_sha(cand), "control_sha": A["control"]["sha256"],
                "saved_sha": A["saved"]["sha256"], "reference_version": A.get("reference_version"), "attempt_index": None, "audit_index": A["audits"],
                "alpha_i": ALPHA_AUDIT, "charged": False, "role": role, "opponent": opp, "opponent_sha": GF_sha(opp)}
    i = (A.get("attempts") or 0) + 1
    t = {"arm_id": arm, "gate_id": f"{arm}-{role}-{cand}-vs-{opp}-a{i}", "candidate_sha": GF_sha(cand), "control_sha": A["control"]["sha256"],
         "saved_sha": A["saved"]["sha256"], "reference_version": A.get("reference_version"), "attempt_index": i, "alpha_i": alpha_i(i), "charged": True,
         "role": role, "opponent": opp, "opponent_sha": GF_sha(opp)}
    if A.get("acct_version") == 2:
        acct_charge(A, t, None)
    else:
        A["attempts"] = i; A.setdefault("alpha_spent", 0.0); A["alpha_spent"] = round(A["alpha_spent"] + t["alpha_i"], 12)
    return t


def effective_alpha(M, G=None):
    """Only an intact, scope-bound prospective supersession can alter alpha.
    Hash integrity cannot establish that nobody read results: that still needs
    an independent exposure/authorization receipt (not supplied in this pack).
    """
    sp = Path(str(M.get("_path", ""))[:-5] + ".supersession.json") if M.get("_path") else None
    frozen = M["ticket"]["alpha_i"]
    if not sp or not sp.exists():
        return frozen
    try:
        s = json.loads(sp.read_text())
    except (OSError, ValueError):
        return frozen
    body = {k:v for k,v in s.items() if k != "supersession_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    when = P.utc_ts(s.get("declared_utc"))
    alpha = (s.get("change") or {}).get("alpha_i")
    if (s.get("schema") != "gate-supersession/v1" or s.get("supersession_sha256") != digest
            or s.get("manifest_sha256") != M["manifest_sha256"]
            or s.get("manifest") != Path(M["_path"]).name
            or s.get("results_read_before_declaration") is not False or s.get("worlds_unchanged") is not True
            or M["ticket"].get("role") != "ref" or when is None
            or not P.utc_ts(M["frozen_at_utc"]) <= when <= P.utc_ts(M["deadline_utc"])
            or (s.get("from") or {}).get("alpha_i") != M["ticket"].get("alpha_i")
            or (s.get("from") or {}).get("attempt_index") != M["ticket"].get("attempt_index")
            or type(alpha) not in (float, int) or not math.isfinite(alpha) or alpha != ALPHA_AUDIT
            or (s.get("change") or {}).get("charged_to_arm_budget") is not False):
        return frozen                                   # PRO CHECK-LADDER-R4: an unbound sidecar never changes the frozen threshold
    if G is None:
        return frozen                                   # exposure cannot be verified without the game records
    try:
        names = {M["challenger"]["name"], M["incumbent"]["name"]}; W = set(M["worlds"])
    except (KeyError, TypeError):
        return frozen
    seen = [r for r in G if {r.get("seat0"), r.get("seat1")} == names and r.get("world_seed") in W
            and (P.utc_ts(r.get("finished_utc")) is None or P.utc_ts(r.get("finished_utc")) <= when)]
    return frozen if seen else alpha                    # record-based pre-look check: no game of this gate had finished at declaration


def declared_reference_components(L, arm, subject):
    """Availability cannot redefine a versioned reference hypothesis."""
    A = L[arm]
    names = ((L.get("_references") or {}).get(A.get("reference_version")) or {}).get("names")
    if not isinstance(names, dict) or not names:
        return [], "missing versioned reference identity registry"
    refs = []
    for name, sha in names.items():
        if name in (A["saved"]["name"], subject):
            continue  # These roles have their separate direct/self handling.
        if not ref_ready(name):
            return [], f"declared reference unavailable: {name}; preserve the full panel"
        if GF_sha(name) != sha:
            return [], f"declared reference bytes changed: {name}; cannot substitute a new hypothesis"
        refs.append(("ref", name))
    return refs, None


def identity_receipt_ok(tag, st):
    """PRO CHECK-0300-GATES #4: a candidate whose identity is a claim (e.g. q4off = O79 weights with one harness change) needs a receipt bound
    to its confirm manifest: self-checksum, manifest sha, and archive / ema.pt hashes that still match the deployed files."""
    mf = Path(st["stages"][tag]["manifest"]); rp = mf.with_name(mf.name[:-5] + ".identity.json")
    if not rp.exists():
        return False, "identity receipt missing"
    rc = json.loads(rp.read_text()); body = {k: v for k, v in rc.items() if k != "receipt_sha256"}
    if rc.get("receipt_sha256") != hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest():
        return False, "identity receipt checksum mismatch"
    if rc.get("manifest_sha256") != json.loads(mf.read_text())["manifest_sha256"]:
        return False, "identity receipt bound to another manifest"
    for side in ("candidate", "reference"):
        n = rc[side]["name"]
        if GF_sha(n) != rc[side]["archive_sha256"]:
            return False, f"{side} archive bytes changed since the receipt"
        h = hashlib.sha256()
        with (GF.DEP / n / "ema.pt").open("rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        if h.hexdigest() != rc[side]["ema_sha256"]:
            return False, f"{side} ema.pt changed since the receipt"
    if rc.get("same_ema") is not True:
        return False, "identity receipt does not show identical weights"
    return True, "identity receipted"


def promotion_chain_tickets(c, st, games):
    """Carry the confirmation/direct receipts to the final CAS, not just audit receipts."""
    confirm = c.get("tag_confirm")
    if not confirm:
        return [], "missing confirmation provenance"
    tickets = []
    for tag in [confirm, c.get("tag_direct")]:
        if not tag:
            continue
        M, v, info, s = gate_result(tag, st, games)
        ticket = M.get("ticket")
        if not ticket:
            return [], "legacy confirmation/direct gate lacks a CAS ticket; explicit migration required"
        if s is None or v in ("PENDING", "HOLD", "INVALID MANIFEST", "POST-HOC (not a decision)"):
            return [], "confirmation/direct gate is not decision eligible"
        if s.get("worlds_complete") != len(M["worlds"]):
            return [], "incomplete confirmation/direct gate"
        if ticket.get("candidate_sha") != M["challenger"]["sha256"] or ticket.get("opponent_sha") != M["incumbent"]["sha256"]:
            return [], "confirmation/direct ticket differs from frozen identities"
        if decide_sup(s, ticket["alpha_i"])[0] != "PASS":
            return [], "confirmation/direct gate did not pass its frozen alpha"
        tickets.append(ticket)
    if tickets[0]["control_sha"] != tickets[0]["saved_sha"] and not c.get("tag_direct"):
        return [], "missing direct-versus-saved gate"
    need_ = (globals().get("IDENTITY_REQUIRED") or (lambda: []))()
    if M["challenger"]["name"] in need_:
        ok_, why_ = identity_receipt_ok(confirm, st)
        if not ok_:
            return [], why_
    return tickets, None


_sha = {}


def GF_sha(n):
    if n not in _sha:
        import hashlib
        f = GF.DEP / n / f"{n}.tar.gz"; h = hashlib.sha256()
        with f.open("rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        _sha[n] = h.hexdigest()
    return _sha[n]


def main():
    st = json.loads(ST.read_text()) if ST.exists() else {"stages": {}, "cands": {}, "seen": []}
    while True:
        try:
            D = json.loads(DS.read_text()); L = D["ladder"]; before = json.dumps(L, sort_keys=True)
            loaded = {a: tier_rev(L.get(a)) for a in ARMS}
            migrate_acct(L, st); reconcile(L, st)
            for c_ in st["cands"].values():                   # chains frozen before the 05:1x window rule belong to G1
                if not c_.get("window") and any(k.startswith(("tag_", "pending_", "tags_")) for k in c_):
                    c_["window"] = "G1"; c_["booking"] = "frozen before the 05:1x window rule"
            rows = [l.split("\t") for l in (H / "REQUESTS.tsv").read_text().splitlines()[1:] if l.strip() and not l.startswith("#")]
            for f in rows:
                key = "\t".join(x.strip() for x in f[:4])
                if key in st["seen"] or len(f) < 2:
                    continue
                kind = f[0].strip(); cand = f[2].strip() if len(f) > 2 else ""; arm = arm_of(cand) if cand else {"N": "N-beven"}.get(f[1].strip(), f[1].strip())
                vs_ = f[3].strip() if len(f) > 3 and f[3].strip() else None
                if arm not in L or not isinstance(L[arm], dict) or not L[arm].get("control"):
                    if key not in st.setdefault("waiting", []):     # PRO CHECK-LADDER-R4: stays pending, retried every round
                        st["waiting"].append(key); ledger(f"request {kind} {cand} waiting: arm {arm} has no saved/current tiers yet")
                    continue
                st["seen"].append(key)
                if key in st.get("waiting", []):
                    st["waiting"].remove(key)
                if kind in ("candidate", "confirm") and cand and not cand.startswith(P.EXCLUDED):
                    k_ = cand if (cand not in st["cands"] or kind == "candidate") else f"{cand}@confirm"   # prospective confirm alongside a running screen
                    st["cands"].setdefault(k_, {"arm": arm, "stage": "screen" if kind == "candidate" else "confirm", "vs": vs_, "vs_fixed": bool(vs_),
                                                **({"archive": cand} if k_ != cand else {})})
                elif kind == "promote-saved":
                    st["cands"][f"promote:{arm}:{L[arm]['control']['name']}"] = {"arm": arm, "stage": "stall", "archive": L[arm]["control"]["name"]}
            for rule in L.get("_auto", []):                                   # auto candidates by deploy-name regex (coordinator 02:30: d30)
                for dd in GF.DEP.iterdir():
                    if re.fullmatch(rule["regex"], dd.name) and GF.ready(dd.name) and dd.name not in st["cands"]:
                        st["cands"][dd.name] = {"arm": arm_of(dd.name), "stage": rule.get("stage", "screen"), "vs": None, "vs_fixed": False,
                                                "prio": rule.get("prio", GPU_PRIO), "note": rule.get("note", "")}
                        ledger(f"auto candidate {dd.name} ({rule.get('note', '')}), priority {rule.get('prio', GPU_PRIO)}")
            G = None
            for key_, c in list(st["cands"].items()):
                arm = c.get("arm") or {"N": "N-beven"}.get(c.get("lineage"), c.get("lineage")); c["arm"] = arm; A = L[arm]
                cand = c.get("archive", key_)
                if c["stage"] in ("done", "rejected", "hold"):
                    continue
                if cand == A["control"]["name"] and c["stage"] in ("screen", "confirm"):
                    c["stage"] = "rejected"; ledger(f"{arm}: {cand} is the current X' (no self-gate)"); continue
                if not c.get("vs_fixed"):
                    c["vs"] = A["control"]["name"]
                opp = {"screen": c.get("vs"), "confirm": c.get("vs"), "direct": A["saved"]["name"], "stall": A["saved"]["name"]}.get(c["stage"])
                if c["stage"] in ("screen", "confirm", "direct"):
                    if not (GF.ready(cand) and GF.ready(opp)):
                        continue
                    tag = c.get("tag_" + c["stage"])
                    if not tag:
                        legacy = f"{'N' if arm == 'N-beven' else arm}-{c['stage'][0].upper()}-{cand}" + (f"-vs-{c['vs']}" if c.get("vs_fixed") else "")
                        if legacy in st["stages"] and c["stage"] in ("screen", "confirm"):
                            tag = legacy; c["legacy_" + c["stage"]] = True          # frozen pre-R3 gate: keeps its verdict
                        else:
                            pk = "pending_" + c["stage"]; pd = c.get(pk)
                            if not pd and not c.get("window"):
                                w_, btxt_ = book(L, st, arm, cand, key_)
                                if w_ is None:
                                    if c.get("book_note") != btxt_:
                                        c["book_note"] = btxt_; ledger(f"{arm} {cand}: {btxt_}")
                                    continue
                                c["window"] = w_; c["booking"] = btxt_; ledger(f"{arm} {cand}: {btxt_}")
                            if not pd:
                                ticket = None
                                if c["stage"] != "screen":
                                    kind_, tk = resolve_tuple(A, st, tuple_key(payload_sha(cand), payload_sha(opp), A.get("reference_version")), c["stage"])
                                    if kind_ == "reuse":
                                        c["tag_" + c["stage"]] = tk; ledger(f"{arm} {c['stage']} {cand} vs {opp}: reuses its current frozen gate {tk}"); continue
                                    if kind_ == "attempted":
                                        c["stage"] = "rejected"; ledger(f"{arm}: {cand} vs {opp} already had its one statistical attempt ({A.get('reference_version')})"); continue
                                    ticket = new_ticket(L, arm, c["stage"], cand, opp); ticket.update(tuple_key=tk, acct=2); acct_charge(A, ticket, tk)
                                n_, size_ = size_for(c["stage"], GF.gpu_ok(cand, opp))
                                tag = f"{arm}-{c['stage'][0].upper()}-{cand}-vs-{opp}" + (f"-a{ticket['attempt_index']}" if ticket else "")
                                pd = c[pk] = {"tag": tag, "ticket": ticket, "worlds": n_, "size": size_, "fails": 0}
                                save_state(st)                                   # red-team 04:3x #1: the tag is saved before sbatch; retries reuse it
                            if not launch(pd["tag"], cand, opp, pd["worlds"], "superiority", st, pd["ticket"], prio=c.get("prio", GPU_PRIO), size=pd["size"]):
                                pd["fails"] += 1
                                if pd["fails"] == 1:
                                    ledger(f"{arm} launch of {pd['tag']} failed (sbatch/freeze); ticket kept, retrying the same tag every round")
                                continue
                            tag = pd["tag"]; c.pop(pk, None)
                            print(time.strftime("%H:%M"), "frozen+submitted", tag, pd["worlds"], "worlds", flush=True)
                        c["tag_" + c["stage"]] = tag
                        continue
                    G = G if G is not None else P.load_gate_games()
                    M, v, info, s = gate_result(tag, st, G)
                    mopp = M["incumbent"]["name"]                               # PRO CHECK-LADDER-R4: the opponent is the frozen one
                    if mopp != opp:
                        if c["stage"] in ("confirm", "direct"):
                            ledger(f"{arm} {c['stage']} {tag} was frozen vs {mopp}, live opponent is now {opp}: STALE, reruns vs {opp} on fresh worlds")
                            c.pop("tag_" + c["stage"], None); continue
                        opp = mopp
                    if v == "EXPIRED":
                        ledger(f"{arm} {c['stage']} EXPIRED {cand} vs {opp}: {info.get('why')} -> HOLD (attempt spent, no verdict) [{tag}]")
                        c["stage"] = "hold"; continue
                    if v in ("PENDING",) or s is None:
                        continue
                    rtl = f"; runtimes {info.get('world_runtimes')}" + (f"; {info['same_runtime_only']}" if info.get("same_runtime_only") else "")
                    tk_ = M.get("ticket") or {"alpha_i": 0.05, "attempt_index": 1, "legacy": True}
                    if c["stage"] == "screen":
                        sv_, swhy_ = decide_screen(s)
                        if sv_ == "HOLD":
                            ledger(f"{arm} screen HOLD {cand} vs {opp} ({swhy_}; {fmt(s)}) [{tag}]"); c["stage"] = "hold"; continue
                        if sv_ == "POSITIVE":
                            c["stage"] = "confirm"; ledger(f"{arm} screen POSITIVE (selection only) {cand} vs X' {opp} ({fmt(s)}); confirm next on fresh worlds (size by use time){rtl} [{tag}]")
                        else:
                            fail_branch(L, arm, cand, c, f"screen NEGATIVE {cand} vs X' {opp} ({fmt(s)}){rtl} [{tag}]")
                        continue
                    dv, why = decide_sup(s, tk_["alpha_i"])
                    if c["stage"] == "confirm":
                        if dv == "PASS":
                            c["branch_winner"] = {"vs": opp, "gate": tag}
                            if c.get("vs_fixed") and opp != A["control"]["name"]:
                                c["stage"] = "hold"; ledger(f"{arm} confirm PASS {cand} vs {opp} ({why}; {fmt(s)}) but {opp} is not X' {A['control']['name']}: branch result only{rtl} [{tag}]"); continue
                            c["stage"] = "direct" if A["control"]["name"] != A["saved"]["name"] else "audit"
                            ledger(f"{arm} confirm PASS -> branch winner {cand} vs X' {opp} ({why}; {fmt(s)}); next: {c['stage']}{rtl} [{tag}]")
                        else:
                            fail_branch(L, arm, cand, c, f"confirm {dv} {cand} vs X' {opp} ({why}; {fmt(s)}){rtl} [{tag}]", counted_confirm=True)
                    else:
                        if dv == "PASS":
                            c["stage"] = "audit"; ledger(f"{arm} direct PASS {cand} vs saved X {opp} ({why}; {fmt(s)}); next: reference audit{rtl} [{tag}]")
                        else:
                            c["stage"] = "hold"; ledger(f"{arm} direct {dv} {cand} vs saved X {opp} ({why}; {fmt(s)}): stays an experiment, X' unchanged{rtl} [{tag}]")
                elif c["stage"] in ("audit", "stall"):
                    subject = cand if c["stage"] == "audit" else A["control"]["name"]
                    if c["stage"] == "stall" and A["control"]["name"] == A["saved"]["name"]:
                        c["stage"] = "rejected"; ledger(f"{arm} stall promotion: X' == X, nothing to promote"); continue
                    refs, reference_issue = declared_reference_components(L, arm, subject)
                    if reference_issue:
                        ledger(f"{arm} {c['stage']} HOLD: {reference_issue}"); continue
                    comps = ([("direct", A["saved"]["name"])] if c["stage"] == "stall" else []) + refs
                    tags = c.setdefault("tags_" + c["stage"], {})
                    pend_ = c.setdefault("pending_" + c["stage"], {})
                    for role, opp_ in comps:
                        k_ = f"{role}:{opp_}"
                        if k_ in tags:
                            continue
                        pd = pend_.get(k_)
                        if not pd:
                            kind_, tk = resolve_tuple(A, st, tuple_key(payload_sha(subject), payload_sha(opp_), A.get("reference_version")), role)
                            if kind_ != "new":                             # red-team 04:3x #3 / R4: reuse a current frozen gate of this tuple
                                tags[k_] = tk if kind_ == "reuse" else "ALREADY-ATTEMPTED"
                                if kind_ == "reuse":
                                    ledger(f"{arm} {c['stage']} {subject}: {k_} reuses its current frozen gate {tk} (one attempt per tuple)")
                                continue
                            ticket = new_ticket(L, arm, role, subject, opp_); ticket.update(tuple_key=tk, acct=2); acct_charge(A, ticket, tk)
                            n_, size_ = size_for(role, GF.gpu_ok(subject, opp_))
                            tg = f"{arm}-{'P' if role == 'direct' else 'R'}-{subject}-vs-{opp_}-" + (f"a{ticket['attempt_index']}" if ticket.get("charged") else f"r{ticket['audit_index']}")
                            pd = pend_[k_] = {"tag": tg, "ticket": ticket, "worlds": n_, "size": size_, "fails": 0}
                            save_state(st)
                        if launch(pd["tag"], subject, opp_, pd["worlds"], "superiority" if role == "direct" else "noninferiority", st, pd["ticket"], size=pd["size"]):
                            tags[k_] = pd["tag"]; pend_.pop(k_, None)
                        else:
                            pd["fails"] += 1
                            if pd["fails"] == 1:
                                ledger(f"{arm} launch of {pd['tag']} failed (sbatch/freeze); ticket kept, retrying the same tag every round")
                    if not all(k_ in tags for k_ in (f"{r}:{o}" for r, o in comps)):
                        continue
                    if any(t_ == "ALREADY-ATTEMPTED" for t_ in tags.values()):
                        ledger(f"{arm} {c['stage']} for {subject}: a component already had its one statistical attempt; HOLD"); c["stage"] = "hold"; continue
                    G = G if G is not None else P.load_gate_games()
                    res = {}
                    for k_, tg in tags.items():
                        M, v, info, s = gate_result(tg, st, G)
                        if v == "EXPIRED":
                            res[k_] = ("EXPIRED", info.get("why"), M, s); continue
                        if v == "PENDING" or s is None:
                            break
                        M["_path"] = st["stages"][tg]["manifest"]; a_ = effective_alpha(M, G)
                        res[k_] = (decide_sup(s, a_) if k_.startswith("direct") else decide_br(s, a_)) + (M, s)
                    else:
                        ok = all(r_[0] == "PASS" for r_ in res.values())
                        detail = "; ".join(f"{k_}: {r_[0]} ({r_[1]})" for k_, r_ in res.items())
                        if not ok:
                            ledger(f"{arm} {'reference audit' if c['stage'] == 'audit' else 'X->X stall promotion'} for {subject}: HOLD; {detail}"); c["stage"] = "hold"; continue
                        tickets = [r_[2]["ticket"] for r_ in res.values()]
                        if c["stage"] == "audit":
                            chain, chain_issue = promotion_chain_tickets(c, st, G)
                            if chain_issue:
                                c["stage"] = "hold"; ledger(f"{arm} promotion HOLD: {chain_issue}"); continue
                            tickets = chain + tickets
                        if c["stage"] == "audit":
                            status, text = commit_xprime(L, arm, subject, tickets, list(tags.values()) + [c.get("tag_confirm"), c.get("tag_direct")])
                            if status == "STALE":
                                if any(cas(tk_, A) == "STALE" for tk_ in chain):   # the confirm/direct receipt itself is stale: rerun the FULL chain
                                    for k__ in ("tag_confirm", "tag_direct", "tags_audit"):
                                        c.pop(k__, None)
                                    c["stage"] = "confirm"; c["vs_fixed"] = False
                                    text += "; confirm/direct ticket is STALE (descriptive only) -> full chain reruns on fresh worlds with current tickets"
                                else:
                                    c["tags_" + c["stage"]] = {}
                            else:
                                c["stage"] = "done" if status == "COMMIT" else "hold"
                            ledger(f"{arm} {text}; {detail}")
                            continue
                        if KILL.exists():
                            ledger(f"{arm} X->X commit for {subject} blocked: automatic tier commits DISABLED ({KILL.name})"); c["stage"] = "hold"; continue
                        if any(cas(tk_, A) == "STALE" for tk_ in tickets):
                            c["tags_" + c["stage"]] = {}; ledger(f"{arm} X->X commit STALE for {subject}: live tiers changed since the ticket; rerun on new worlds")
                            continue
                        else:
                            old = A["saved"]["name"]
                            A["saved"] = dict(A["control"], promoted_to_saved=P.now()); c["stage"] = "done"
                            ledger(f"{arm} COMMIT X := {A['saved']['name']} (was {old}); CAS ok; {detail}")
            for k_, v_ in _pending_stall:
                st["cands"].setdefault(k_, v_)
            _pending_stall.clear()
            if json.dumps(L, sort_keys=True) != before:
                write_ladder(L, loaded)
            save_state(st)
        except Exception as e:  # noqa: BLE001
            print(time.strftime("%H:%M"), "error:", type(e).__name__, str(e)[:300], flush=True)
        time.sleep(60)


def commit_xprime(L, arm, subject, tickets, via):
    """Commit X' := subject after its full chain passed. CAS on every ticket (STALE -> rerun on new worlds). Coordinator 02:25 (owner:
    'keep training until it passes'): FLAG marks the failed checkpoint, not the line; only a checkpoint that itself failed can never
    become X/X'. A later checkpoint of a flagged line commits, and its pass clears the line's flag."""
    A = L[arm]
    if not tickets or any(tk_.get("arm_id") != arm or tk_.get("candidate_sha") != GF_sha(subject) for tk_ in tickets):
        return "HOLD", "empty evidence or ticket belongs to a different arm/candidate"
    kill_ = globals().get("KILL")                                  # ASTRA CHECK-0300-GATES: automatic tier commits blocked
    if kill_ is not None and kill_.exists():
        return "HOLD", f"automatic tier commits DISABLED ({kill_.name}): {subject} would be considered for X'; no tier change"
    if any(cas(tk_, A) == "STALE" for tk_ in tickets):
        return "STALE", f"commit STALE for {subject}: live tiers changed since the ticket; rerun on new worlds"
    if subject in A.get("flagged_candidates", []) or (A.get("flagged_payloads") and globals().get("payload_sha", GF_sha)(subject) in A["flagged_payloads"]):
        return "HOLD", f"{subject} is a FLAGGED (failed) checkpoint: it can never become X/X' (R3 rule 7)"
    line = line_of(L, subject); old = A["control"]["name"]
    A["control"] = {"name": subject, "sha256": GF_sha(subject), "since": P.now(), "via": via}
    A["fails"] = 0
    cleared = line in A.get("flagged_lines", [])
    if cleared:
        A["flagged_lines"].remove(line)
    return "COMMIT", f"COMMIT X' := {subject} (was {old}); CAS ok" + (f"; line {line} flag cleared by this pass" if cleared else "")


def fail_branch(L, arm, cand, c, text, counted_confirm=False):
    """Before the line budget: FLAG (the line may continue, its winners never become X/X'). After: FAIL + ROLLBACK; a failed confirm counts
    toward the stall promotion (2 consecutive counted failures -> X'->X gate)."""
    A = L[arm]; ok_b, btxt = budget(L, cand); line = line_of(L, cand); c["stage"] = "rejected"
    if not ok_b:
        A.setdefault("flagged_lines", []); A.setdefault("flagged_candidates", [])
        if line not in A["flagged_lines"]:
            A["flagged_lines"].append(line)
        if cand not in A["flagged_candidates"]:
            A["flagged_candidates"].append(cand)
        ps_ = payload_sha(cand)
        if ps_ not in A.setdefault("flagged_payloads", []):
            A["flagged_payloads"].append(ps_)
        ledger(f"{arm} {text} -> FLAG ({btxt}); checkpoint {cand} can never become X/X' or parent a branch; line {line} continues and a later "
               f"checkpoint that passes the full chain is promotable (clears the flag)")
        return
    ledger(f"{arm} {text} -> FAIL ({btxt})")
    A.setdefault("rollbacks", []).append({"line": line, "candidate": cand, "reset_to": A["control"]["name"], "utc": P.now()})
    ledger(f"{arm} ROLLBACK: line {line} discarded after its budget; next branch forks from canonical X' {A['control']['name']}")
    if counted_confirm:
        A["fails"] = A.get("fails", 0) + 1
        if A["fails"] >= 2 and A["control"]["name"] != A["saved"]["name"]:
            ledger(f"{arm} 2 counted failed confirms: opening the X'->X stall promotion gate for {A['control']['name']}")
            c2 = {"arm": arm, "stage": "stall", "archive": A["control"]["name"]}
            _pending_stall.append((f"promote:{arm}:{A['control']['name']}", c2))


_pending_stall = []

if __name__ == "__main__":
    main()
