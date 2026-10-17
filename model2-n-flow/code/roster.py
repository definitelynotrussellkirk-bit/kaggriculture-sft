#!/usr/bin/env python3
"""N flow roster from a wins-only broad roster (gpu-ops build_broad.py output) + catalog SEATS.parquet.

  python3 roster.py ROWS_BROAD.tsv OUT.tsv [--cache DIR] [--seats SEATS.parquet]

Row kinds (column `kind`), in this order (old rows keep their positions when rows are appended):
  act        every roster row; must be a catalog result-W seat (refused otherwise); tier from ":broad:tierN";
             weight = the roster weight (B's winner/opponent-strength/recency weight x tier factor). Action + aux loss.
  wdl-loser  the opposite (losing/drawing) seat of the same episode when its .stp is cached. WDL/aux only, never
             action loss (owner 09-30). Sampling weight = the game's roster weight.
  own-loss   every seat our teams (OUR_TEAMS) lost (catalog result L, day >= 09-24, valid config, cached .stp, not
             already a row), after the DATA-PLAN-0930 exclusions (leak ids, MMPQ-VAL, sealed, trainer heldout split,
             practice VAL, decode-h2h holdout seeds, blocked sources/subs). WDL only, label L. Sampling weight = the
             mean weight of the wdl-loser rows. tier = 1 (09-24..27) or 2 (09-28..).
WDL label = sign(final_cash - opp_cash) from the catalog. WDL loss weight (owner 09-30, "a lot of close games, a few
landslides"): w = 1 / (1 + |margin| / M), M = median |margin| over all rows, renormalised to mean 1 over all rows.
Writes OUT.tsv and OUT.receipt.json (row counts and mass per tier and kind, margin distribution, ESS, refusals).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq

CACHE = "/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11"
CAT = "/mnt/ai/scratch/farmNEAT/primitive-bc-data/catalog"
SEATS = f"{CAT}/SEATS.parquet"
PBC = "/home/russ/Desktop/farmNEAT/work/runs/maintainer/primitive-bc-v1-deploy-v9-decode"
OUR_TEAMS = ("有辣条有权", "Russell Kirk")
OWN_LOSS_DAYS = ("2026-09-29", "2026-09-30")          # owner 09-30 02:2x: our losses from these days only
OWN_CACHE = "/mnt/ai/scratch/farmNEAT/n-flow/stp-own-loss"
COLS = ("path", "eid", "seat", "weight", "tier", "wdl", "act_loss", "final_cash", "opp_cash", "day", "archive", "member",
        "kind", "wdl_w", "bucket", "fp")
LEAK = "/mnt/ai/scratch/farmNEAT/touchup-top8-0929/LEAK-HOLDOUT-IDS.txt"
MMPQ_VAL = "/mnt/ai/scratch/farmNEAT/mmpq-val/VAL-GAMES.tsv"
PRACTICE_VAL = "/mnt/ai/scratch/farmNEAT/practice-v1/val/VAL.tsv"
HOLDOUT = "/mnt/ai/scratch/farmNEAT/decode-h2h/HOLDOUT-EPISODES.txt"
BLOCKED_SOURCES = ("decode-h2h", "decode-exp-0929")
BLOCKED_SUBS = {56680736}


EXCL_LATEST = "/mnt/ai/scratch/farmNEAT/broad/exclusions/EXCLUSIONS-LATEST"
TOP4_VAL = "/mnt/ai/scratch/farmNEAT/corpus-final/special/TOP4-BEATERS-LATEST/VAL.tsv"
LINEAGE = "N"      # which own-win lineage view this roster consumes
TOP4_VAL_EIDS = "/mnt/ai/scratch/farmNEAT/corpus-final/special/TOP4-BEATERS-VAL.eids"   # PRO BUCKETS: registry, unioned
RECIPE = "/mnt/ai/scratch/farmNEAT/broad/BUCKET-RECIPE.json"


def usable_cache(path):
    """PRO BUCKETS: a cache path is consumable only when it is a nonempty regular file."""
    return os.path.isfile(path) and os.path.getsize(path) > 0


def source_text(c):
    """PRO BUCKETS: archive + sources as one string; a string-valued sources field is not split into characters."""
    src = c.get("sources")
    items = [str(v) for v in src] if isinstance(src, (list, tuple, set)) else ([] if src is None else [str(src)])
    return " ".join([str(c.get("archive"))] + items)


def recompute_effective_shares(identity, counts):
    """Producer-equivalent effective shares (a port of broad/build_buckets.py: dynamic 'cap_max_vs_A' buckets, then the
    one-pass small-bucket cap) from the hash-bound identity_json and the exact signed row counts."""
    raw = identity.get("shares") if isinstance(identity, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise SystemExit("REFUSED: identity_json has no recipe shares")
    recipe = {}
    for name, value in raw.items():
        try:
            x = float(value)
        except (TypeError, ValueError):
            raise SystemExit(f"REFUSED: identity share for {name} is not numeric")
        if not math.isfinite(x) or x < 0:
            raise SystemExit(f"REFUSED: identity share for {name} is invalid")
        recipe[name] = x
    dynr = identity.get("dynamic_shares") or {}
    cap = identity.get("small_bucket_cap") or {}
    live = {k: v for k, v in recipe.items() if counts.get(k, 0) > 0}
    tot = sum(v for k, v in live.items() if k not in dynr)
    if not live or not math.isfinite(tot) or tot <= 0:
        raise SystemExit("REFUSED: identity shares have no live positive mass")
    base = {k: v / tot for k, v in live.items()}
    for k, rule in dynr.items():
        if rule == "lineage_only":
            base.pop(k, None); live.pop(k, None)
    dyn = set()
    for k, rule in dynr.items():
        if rule == "lineage_only":
            continue
        if rule != "cap_max_vs_A":
            raise SystemExit(f"REFUSED: unknown dynamic share rule {k}: {rule}")
        if k in live and "A" in live and counts.get("A", 0) > 0:
            mult = float(cap.get("multiplier", 4))
            others = {j: v for j, v in base.items() if j != k}
            ot = sum(others.values())
            kk = mult * (others["A"] / ot) * counts[k] / counts["A"]
            x = kk / (1 + kk)
            base = {j: v / ot * (1 - x) for j, v in others.items()}
            base[k] = x
            dyn.add(k)
    eff = dict(base)
    if cap and "A" in base and counts.get("A", 0) > 0:
        mult = float(cap.get("multiplier", 4))
        if not math.isfinite(mult) or mult <= 0:
            raise SystemExit("REFUSED: invalid small_bucket_cap multiplier")
        rate_a, excess = base["A"] / counts["A"], 0.0
        for k in live:
            if k == "A" or k in dyn:
                continue
            lim = mult * rate_a * counts[k]
            if base[k] > lim:
                excess += base[k] - lim
                eff[k] = lim
        eff["A"] = base["A"] + excess
    return {k: round(v, 6) for k, v in eff.items()}


def read_bucket_keys(path, bucket):
    """Parse one signed bucket and return exact admitted seat keys."""
    try:
        with open(path, newline="") as src:
            reader = csv.DictReader(src, delimiter="\t")
            fields = reader.fieldnames
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise SystemExit(f"REFUSED: cannot parse signed bucket {bucket}: {exc}")
    if not fields or len(fields) != len(set(fields)):
        raise SystemExit(f"REFUSED: signed bucket {bucket} has no unique header")
    eid_key = "eid" if "eid" in fields else ("episode_id" if "episode_id" in fields else None)
    if eid_key is None or "seat" not in fields:
        raise SystemExit(f"REFUSED: signed bucket {bucket} lacks eid/episode_id or seat")
    keys = []
    for lineno, row in enumerate(rows, 2):
        try:
            eid = int((row.get(eid_key) or "").strip())
            seat = int((row.get("seat") or "").strip())
        except ValueError:
            raise SystemExit(f"REFUSED: malformed eid/seat in signed bucket {bucket} at {path}:{lineno}")
        if seat not in (0, 1):
            raise SystemExit(f"REFUSED: invalid seat {seat} in signed bucket {bucket} at {path}:{lineno}")
        keys.append((eid, seat))
    return keys


def load_bucket_generation(bucket_dir):
    """Validate and return the signed bucket order/shares/files for direct N-flow consumption."""
    bd = os.path.realpath(bucket_dir)
    rp = os.path.join(bd, "RECEIPT.json")
    ep = os.path.join(bd, "EFFECTIVE-SHARES.json")
    if not os.path.isfile(rp) or not os.path.isfile(ep):
        raise SystemExit(f"REFUSED: bucket generation lacks RECEIPT.json or EFFECTIVE-SHARES.json: {bd}")
    with open(rp) as f:
        receipt = json.load(f)
    with open(ep) as f:
        effective = json.load(f)
    expected = receipt.get("effective_shares")
    observed = effective.get("effective_shares")
    if not isinstance(expected, dict) or not expected:
        raise SystemExit("REFUSED: receipt has no effective_shares")
    if not isinstance(observed, dict) or set(observed) != set(expected):
        raise SystemExit("REFUSED: EFFECTIVE-SHARES keys differ from receipt")
    shares = {}
    for name, value in expected.items():
        try:
            x = float(value); y = float(observed[name])
        except (TypeError, ValueError):
            raise SystemExit(f"REFUSED: nonnumeric effective share for {name}")
        if not math.isfinite(x) or x <= 0 or not math.isfinite(y) or abs(x - y) > 1e-12:
            raise SystemExit(f"REFUSED: EFFECTIVE-SHARES value differs from receipt for {name}: {y} != {x}")
        shares[name] = x
    if not math.isfinite(sum(shares.values())) or sum(shares.values()) <= 0:
        raise SystemExit("REFUSED: invalid effective-share total")
    files = effective.get("files")
    if not isinstance(files, list) or not files or len(files) != len(set(files)):
        raise SystemExit("REFUSED: invalid or duplicate EFFECTIVE-SHARES files list")
    paths = {}
    body = bytearray()
    for name in files:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".tsv"):
            raise SystemExit(f"REFUSED: unsafe bucket filename {name!r}")
        path = os.path.join(bd, name)
        if not os.path.isfile(path):
            raise SystemExit(f"REFUSED: listed bucket file missing or not regular: {path}")
        body.extend(Path(path).read_bytes())
        paths[Path(name).stem] = path
    for name in shares:
        if paths.get(name) != os.path.join(bd, f"{name}.tsv"):
            raise SystemExit(f"REFUSED: signed file for bucket {name} is not exactly {name}.tsv")
    ident = receipt.get("identity_json")
    if not isinstance(ident, str):
        raise SystemExit("REFUSED: receipt has no identity_json")
    try:
        identity = json.loads(ident)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"REFUSED: identity_json is invalid JSON: {exc}")
    identity_shares = identity.get("shares") if isinstance(identity, dict) else None
    if not isinstance(identity_shares, dict) or not identity_shares:
        raise SystemExit("REFUSED: identity_json has no ordered recipe shares")
    hashed = list(identity_shares)                   # every hash-bound bucket file (incl. lineage-only W_*)
    lin_only = {k for k, r in (identity.get("dynamic_shares") or {}).items() if r == "lineage_only"}
    bucket_order = [k for k in hashed if k not in lin_only]     # neutral buckets, recipe order -> tiers
    if set(bucket_order) != set(shares):
        raise SystemExit("REFUSED: sidecar bucket keys differ from hash-bound identity_json")
    expected_files = [f"{name}.tsv" for name in hashed]
    # the producer (build_buckets.py) hashes files in its sidecar list order (e.g. A, L, BC, D, E); the byte hash below
    # binds that order, tiers come from identity_json order, so only the file SET must equal the hash-bound buckets
    if sorted(files) != sorted(expected_files) or set(paths) != set(hashed):
        raise SystemExit(f"REFUSED: signed files must be exactly the hash-bound buckets: {expected_files} vs {files}")
    counts, all_keys = {}, {}
    receipt_buckets = receipt.get("buckets")
    if not isinstance(receipt_buckets, dict):
        raise SystemExit("REFUSED: receipt has no bucket accounting")
    for name in hashed:
        keys = read_bucket_keys(paths[name], name)
        counts[name] = len(keys)
        expected_count = (receipt_buckets.get(name) or {}).get("cached")
        try:
            expected_count = int(expected_count)
        except (TypeError, ValueError):
            raise SystemExit(f"REFUSED: receipt has no valid cached count for bucket {name}")
        if expected_count != len(keys):
            raise SystemExit(f"REFUSED: bucket {name} rows={len(keys)} but receipt cached={expected_count}")
        for key in keys:
            if key in all_keys:
                raise SystemExit(f"REFUSED: duplicate signed seat {key[0]}/{key[1]} in buckets {all_keys[key]} and {name}")
            all_keys[key] = name
    recomputed = recompute_effective_shares(identity, counts)
    if set(recomputed) != set(shares) or any(abs(recomputed[name] - shares[name]) > 2e-6 for name in recomputed):
        raise SystemExit(f"REFUSED: sidecar effective shares differ from hash-bound recipe/count recomputation: {shares} != {recomputed}")
    digest = hashlib.sha256(ident.encode() + b"\n" + bytes(body)).hexdigest()
    if digest != receipt.get("sha256"):
        raise SystemExit(f"REFUSED: generation byte hash mismatch: {digest[:12]} != {str(receipt.get('sha256'))[:12]}")
    if os.path.basename(bd) != digest[:12] or effective.get("generation") != digest[:12]:
        raise SystemExit("REFUSED: generation directory/EFFECTIVE-SHARES identity mismatch")
    recipe_sha = (receipt.get("recipe") or {}).get("sha256")
    if recipe_sha and effective.get("recipe_sha256") != recipe_sha:
        raise SystemExit("REFUSED: EFFECTIVE-SHARES recipe hash differs from receipt")
    # data-ops 09:1x: N's own-win bucket (W_N) enters through lineage_views["N"] only; W_B / W_O are other lineages'
    owc = identity.get("own_win_buckets") or {}
    wb = (owc.get("lineages") or {}).get(LINEAGE)
    view = (effective.get("lineage_views") or {}).get(LINEAGE)
    if wb and counts.get(wb, 0) > 0 and counts.get("A", 0) > 0:
        if not isinstance(view, dict) or view.get("bucket") != wb:
            raise SystemExit(f"REFUSED: {wb} has {counts[wb]} cached seats but EFFECTIVE-SHARES has no lineage view {LINEAGE}")
        mult = float((identity.get("small_bucket_cap") or {}).get("multiplier", 4))
        kk = mult * recomputed["A"] * counts[wb] / counts["A"]
        x = min(float(owc.get("max_share", .05)), kk / (1 + kk))
        want = {k: round(v * (1 - x), 6) for k, v in recomputed.items()}
        want[wb] = round(x, 6)
        got = view.get("effective_shares") or {}
        if set(got) != set(want) or any(abs(float(got[k]) - want[k]) > 5e-6 for k in want):
            raise SystemExit(f"REFUSED: lineage view {LINEAGE} shares {got} != recomputed {want}")
        return bd, bucket_order + [wb], {k: float(got[k]) for k in want}, paths, digest
    if view:
        raise SystemExit(f"REFUSED: lineage view {LINEAGE} present but its W bucket has no cached seats")
    return bd, bucket_order, shares, paths, digest


def exclusions():
    """Exclusions rechecked at roster time (review BUCKETS #2 / ASTRA-5 #6), fail closed if a source is missing/empty:
    returns ({family: eids excluded for both seats}, {(eid, seat) excluded}, trainer data module). Families: leak,
    MMPQ-VAL, practice VAL, decode-holdout seeds, sealed, TOP4-BEATERS VAL, plus every family of the shared exclusion
    union broad/exclusions/EXCLUSIONS-LATEST (hash_val_split, quarantine_sub, top4_val, ...)."""
    sys.path.insert(0, PBC)
    sys.path.insert(0, str(Path(PBC).parents[3] / "src"))
    import data as trainer_data
    leak = {int(x) for x in open(LEAK).read().split()} | {115279125}
    mval = {int(l.split("\t")[0]) for l in open(MMPQ_VAL).read().splitlines()[1:] if l.strip()}
    prac = {int(l.split("\t")[3]) for l in open(PRACTICE_VAL).read().splitlines()[1:] if l.strip()}
    seeds = {int(l.split("\t")[0]) for l in open(HOLDOUT).read().splitlines() if l.strip() and not l.startswith("#")}
    top4 = set()
    for p in (TOP4_VAL, TOP4_VAL_EIDS):       # VAL.tsv (eid column) and the one-eid-per-line registry
        if not os.path.exists(p):
            continue
        ls = [l for l in open(p).read().splitlines() if l.strip()]
        if ls and "\t" in ls[0]:
            h = ls[0].split("\t")
            top4 |= {int(l.split("\t")[h.index("eid")]) for l in ls[1:]}
        else:
            top4 |= {int(x) for x in ls}
    ep = pq.read_table(f"{CAT}/EPISODES.parquet", columns=["episode_id", "seed"]).to_pydict()
    hold = {int(e) for e, s in zip(ep["episode_id"], ep["seed"]) if s is not None and int(s) in seeds}
    sealed = set(int(x) for x in trainer_data.sealed_ids())
    empty = [k for k, v in (("leak", leak), ("mmpq_val", mval), ("practice_val", prac), ("holdout_seeds", seeds),
                            ("top4_val", top4), ("sealed", sealed)) if not v]
    if empty:
        raise SystemExit(f"exclusion list empty (fail closed): {', '.join(empty)}")
    fam = {"leak": leak, "mmpq_val": mval, "practice_val": prac, "holdout_seed": hold, "sealed": sealed, "top4_val": top4}
    seat_ex = set()
    lines = open(os.path.realpath(EXCL_LATEST)).read().splitlines()
    if len(lines) < 2 or lines[0].split("\t")[:3] != ["eid", "seat", "family"]:
        raise SystemExit(f"{EXCL_LATEST}: missing or malformed")
    for l in lines[1:]:
        e, st, f = l.split("\t")[:3]
        if st == "*":
            fam.setdefault("union:" + f, set()).add(int(e))
        else:
            seat_ex.add((int(e), int(st)))
    return fam, seat_ex, trainer_data


def admission(e, s, c, ex, seat_ex, trainer_data):
    """One admission check for EVERY row kind (review BUCKETS #3): None if the (episode, seat) may enter, else why."""
    hit = [k for k, v in ex.items() if e in v]
    if (e, s) in seat_ex:
        hit.append("union:seat")
    if not trainer_data.in_split(e, "train"):
        hit.append("heldout_split")
    if hit:
        return "excluded: " + ",".join(hit)
    if c is None:
        return "not in catalog"
    if any(b in source_text(c) for b in BLOCKED_SOURCES):
        return "excluded: blocked source"
    if c["submission_id"] in BLOCKED_SUBS:
        return "excluded: blocked sub"
    if not c["config_ok"]:
        return "config not ok"
    if c.get("static"):
        return "static tape"
    if c["team"] in OUR_TEAMS and c["day"] not in OWN_LOSS_DAYS:
        return "own seat outside 09-29/09-30"
    if c["final_cash"] is None or c["opp_cash"] is None:
        return "no final cash"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("out")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--seats", default=SEATS)
    ap.add_argument("--buckets", default="", help="bucket dir with A..E.tsv; act rows carry their bucket, tier = 1..5")
    a = ap.parse_args()
    bucket_names, bucket_probs = list("ABCDE"), None
    if a.buckets:            # owner 09-30: bucket seat lists (gpu-ops build_buckets.py); act rows sampled by bucket
        # review PRO BUCKETS-AND-ROSTERS #2: bucket order (-> tier) and probabilities come from the hash-bound receipt
        # (identity_json recipe order, receipt effective_shares == sidecar == recomputation), refusing any mismatch
        bd, bucket_names, bucket_probs, _paths, gen_digest = load_bucket_generation(a.buckets)
        if os.path.exists(RECIPE):    # and the generation must be built under the current bucket policy
            want = hashlib.sha256(Path(RECIPE).read_bytes()).hexdigest()
            got = json.load(open(os.path.join(a.buckets, "EFFECTIVE-SHARES.json"))).get("recipe_sha256")
            if got != want:
                sys.exit(f"bucket generation recipe {got} != current BUCKET-RECIPE.json {want}: refusing")
        head, body = None, []
        for b in bucket_names:
            f = os.path.join(a.buckets, f"{b}.tsv")
            if not os.path.exists(f):
                continue
            ls = [l.rstrip("\n").split("\t") for l in open(f) if l.strip()]
            head = head or ls[0]
            if ls[0] != head:
                sys.exit(f"bucket {b} header differs")
            ix0 = {k: i for i, k in enumerate(ls[0])}
            for r in ls[1:]:
                r = list(r)
                r[ix0["reason"]] = r[ix0["reason"]] + f":broad:tier{bucket_names.index(b) + 1}"
                body.append(r + [b])
        head = head + ["bucket"]
        a.rows = os.path.join(a.buckets, "A.tsv")
    else:
        lines = [l.rstrip("\n").split("\t") for l in open(a.rows) if l.strip()]
        head, body = lines[0], lines[1:]
    ix = {k: i for i, k in enumerate(head)}
    t = pq.read_table(a.seats, columns=["episode_id", "seat", "final_cash", "opp_cash", "result", "team", "day", "archive",
                                        "member", "config_ok", "submission_id", "sources", "static"]).to_pylist()
    cat = {(r["episode_id"], r["seat"]): r for r in t}
    ex, seat_ex, trainer_data = exclusions()
    out, refused, seen = [], [], set()
    for r in body:
        eid, seat, w = int(r[ix["eid"]]), int(r[ix["seat"]]), float(r[ix["weight"]])
        why = admission(eid, seat, cat.get((eid, seat)), ex, seat_ex, trainer_data)
        if why is None and any(bs in r[ix["archive"]] for bs in BLOCKED_SOURCES):
            why = "excluded: blocked source"
        if why:
            refused.append({"eid": eid, "seat": seat, "why": why})
            continue
        reason = r[ix["reason"]]
        tier = int(reason.rsplit(":tier", 1)[1]) if ":broad:tier" in reason else None
        c = cat.get((eid, seat))
        if tier is None or c is None or c["result"] != "W" or c["final_cash"] is None or not c["final_cash"] > c["opp_cash"]:
            refused.append({"eid": eid, "seat": seat, "why": "no tier" if tier is None else "not a catalog W seat"})
            continue
        if (eid, seat) in seen:
            refused.append({"eid": eid, "seat": seat, "why": "duplicate"})
            continue
        p = f"{a.cache}/{eid}-{seat}.stp"
        if not usable_cache(p):
            refused.append({"eid": eid, "seat": seat, "why": "no .stp"})
            continue
        seen.add((eid, seat))
        bucket = r[ix["bucket"]] if "bucket" in ix else ""
        out.append([p, eid, seat, w, tier, 2, 1, c["final_cash"], c["opp_cash"], r[ix["day"]], r[ix["archive"]],
                    r[ix["member"]], "act", bucket])
        o = 1 - seat
        po = f"{a.cache}/{eid}-{o}.stp"
        co = cat.get((eid, o))
        why_o = admission(eid, o, co, ex, seat_ex, trainer_data)      # the opposite seat passes the same admission
        if why_o:
            refused.append({"eid": eid, "seat": o, "why": "wdl-loser dropped: " + why_o})
            continue
        if usable_cache(po):
            wdl = 2 if co["final_cash"] > co["opp_cash"] else 1 if co["final_cash"] == co["opp_cash"] else 0
            seen.add((eid, o))
            out.append([po, eid, o, w, tier, wdl, 0, co["final_cash"], co["opp_cash"], r[ix["day"]], r[ix["archive"]],
                        r[ix["member"]], "wdl-loser", ""])
    # own losses
    loser_w = [row[3] for row in out if row[12] == "wdl-loser"]
    own_w = round(statistics.mean(loser_w), 3) if loser_w else 1.0
    own_seen, own_drop, unbuilt = 0, {}, []
    for c in t:
        if c["team"] not in OUR_TEAMS or c["result"] != "L":
            continue
        own_seen += 1
        e, s = int(c["episode_id"]), int(c["seat"])
        why = None
        if (e, s) in seen:
            why = "already a row"
        elif c["day"] not in OWN_LOSS_DAYS:
            why = "day not 09-29/09-30"
        else:
            why = admission(e, s, c, ex, seat_ex, trainer_data)
        if why is None and not c["final_cash"] < c["opp_cash"]:
            why = "cash not a loss"
        stp = next((p for p in (f"{a.cache}/{e}-{s}.stp", f"{OWN_CACHE}/{e}-{s}.stp") if usable_cache(p)), None)
        if why is None and stp is None:
            why = "no v11 .stp (listed for build)"
            unbuilt.append((c["archive"], c["member"], c["day"], e, s, 1, "nflow:own-loss"))
        if why:
            own_drop[why] = own_drop.get(why, 0) + 1
            continue
        seen.add((e, s))
        out.append([stp, e, s, own_w, 1 if c["day"] <= "2026-09-27" else 2, 0, 0, c["final_cash"],
                    c["opp_cash"], c["day"], c["archive"], c["member"], "own-loss", ""])
    # closeness weights for the WDL head
    margins = [abs(row[7] - row[8]) for row in out]
    M = statistics.median(margins)
    raw = [1.0 / (1.0 + m / M) for m in margins]
    mean = sum(raw) / len(raw)
    for row, w in zip(out, raw):
        row.insert(13, round(w / mean, 6))
    ws = [row[13] for row in out]
    ess = sum(ws) ** 2 / sum(w * w for w in ws)
    # catalog fingerprint per row (review TRAINING-DATA #1): SEAT_DAYS cash of the seat and its opponent on FP_DAYS; the
    # loader checks the seat file's own content against it
    sys.path.insert(0, str(Path(__file__).resolve().parent))     # this dir's nf_format (not a staged copy)
    for m in [m for m in sys.modules if m == "nf_format"]:
        del sys.modules[m]
    import nf_format as NF
    sd = pq.read_table(f"{CAT}/SEAT_DAYS.parquet", columns=["episode_id", "seat", "game_day", "cash"],
                       filters=[("episode_id", "in", sorted({row[1] for row in out})),
                                ("game_day", "in", list(NF.FP_DAYS))]).to_pydict()
    sdc = {(e, s_, d): c for e, s_, d, c in zip(sd["episode_id"], sd["seat"], sd["game_day"], sd["cash"])}
    n_fp = 0
    for row in out:
        fp = {d: [sdc.get((row[1], row[2], d)), sdc.get((row[1], 1 - row[2], d))] for d in NF.FP_DAYS}
        fp = {d: v for d, v in fp.items() if v[0] is not None}
        n_fp += bool(fp)
        row.append(json.dumps(fp, separators=(",", ":")) if fp else "")
    q = statistics.quantiles(margins, n=4)
    groups = {}
    for row in out:
        k = f"{row[12]}:tier{row[4]}"
        d = groups.setdefault(k, {"rows": 0, "mass": 0.0})
        d["rows"] += 1
        d["mass"] += row[3]
    act_mass = sum(v["mass"] for k, v in groups.items() if k.startswith("act:"))
    for k, v in groups.items():
        if k.startswith("act:"):
            v["share"] = round(v["mass"] / act_mass, 4)
    for need in (() if a.buckets else ("act:tier1", "act:tier2", "act:tier3")):
        if not groups.get(need, {}).get("rows"):
            sys.exit(f"empty tier {need}: refusing")
    if a.buckets:
        receipt_buckets = {b: sum(1 for r in out if r[12] == "act" and r[14] == b) for b in bucket_names}
    with open(a.out + ".tmp", "w") as f:
        f.write("\t".join(COLS) + "\n")
        for row in out:
            f.write("\t".join(str(x) for x in row) + "\n")
    os.replace(a.out + ".tmp", a.out)
    consumed = ([os.path.join(a.buckets, f"{b}.tsv") for b in bucket_names]
                + [os.path.join(a.buckets, f) for f in ("EFFECTIVE-SHARES.json", "RECEIPT.json")]) if a.buckets else [a.rows]
    receipt = {"source_rows": a.rows, "source_sha256": hashlib.sha256(Path(a.rows).read_bytes()).hexdigest(),
               # review TRAINING-DATA #6: every consumed bucket file (not only A.tsv) plus the generation's receipts
               "source_files_sha256": {os.path.basename(f): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                                       for f in consumed if os.path.exists(f)},
               "out": a.out, "out_sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(), "catalog": a.seats,
               "groups": groups, "kinds": {k: sum(1 for r in out if r[12] == k) for k in ("act", "wdl-loser", "own-loss")},
               "refused": len(refused), "refused_sample": refused[:20],
               "refused_by_reason": dict(__import__("collections").Counter(x["why"] for x in refused)),
               "own_loss": {"teams": OUR_TEAMS, "catalog_L_seats": own_seen, "kept": sum(1 for r in out if r[12] == "own-loss"),
                            "dropped": own_drop, "sampling_weight": own_w, "days": OWN_LOSS_DAYS},
               "wdl_labels": {k: sum(1 for r in out if r[5] == v) for k, v in (("L", 0), ("D", 1), ("W", 2))},
               "buckets": receipt_buckets if a.buckets else None,
               "fingerprinted_rows": n_fp,
               "bucket_probs_arg": ",".join(f"{k}={v:.4f}" for k, v in bucket_probs.items()) if bucket_probs else None,
               "margin_abs": {"median_M": M, "q1": q[0], "q2": q[1], "q3": q[2], "min": min(margins), "max": max(margins),
                              "share_lt_2000": round(sum(m < 2000 for m in margins) / len(margins), 4),
                              "share_gt_20000": round(sum(m > 20000 for m in margins) / len(margins), 4)},
               "wdl_weight": {"rule": "w = 1/(1+|margin|/M), M = median |margin| of all rows, mean-normalised to 1",
                              "min": min(ws), "max": max(ws), "mean": round(sum(ws) / len(ws), 6), "ess": round(ess, 1),
                              "rows": len(ws)}}
    with open(a.out + ".own-loss-unbuilt.tsv", "w") as f:     # step_cache.py build rows (archive member day eid seat weight reason)
        f.write("archive\tmember\tday\teid\tseat\tweight\treason\n")
        for row in unbuilt:
            f.write("\t".join(str(x) for x in row) + "\n")
    receipt["own_loss"]["unbuilt_listed"] = len(unbuilt)
    Path(a.out + ".receipt.json").write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("kinds", "own_loss", "margin_abs", "wdl_weight", "wdl_labels")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
