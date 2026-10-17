#!/usr/bin/env python3
"""roster.load_bucket_generation on a producer-shaped fixture (broad/build_buckets.py, 09-30 09:1x): neutral buckets
A/BC/D/E/L (L dynamic cap_max_vs_A), lineage-only W_B/W_N/W_O, lineage view N. Checks tier order, the N view shares,
and refusals for tampered views, a missing view, sidecar key order games and a changed file.
  python3 test_generation_view.py
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roster as R  # noqa: E402

HEAD = "archive\tmember\tday\teid\tseat\tweight\treason\n"
COUNTS = {"A": 40, "L": 3, "BC": 20, "D": 10, "E": 30, "W_B": 2, "W_N": 2, "W_O": 1}
IDENT = {"schema": "bucket-recipe/v4", "version": None,
         "shares": {"A": .35, "BC": .4, "D": .15, "E": .05, "L": 0.0, "W_B": 0.0, "W_N": 0.0, "W_O": 0.0},
         "merge": {}, "small_bucket_cap": {"multiplier": 4}, "beat_us_days": [], "beat_us_bucket": "L",
         "dynamic_shares": {"L": "cap_max_vs_A", "W_B": "lineage_only", "W_N": "lineage_only", "W_O": "lineage_only"},
         "reserved": {}, "sampling": "x", "a_top": None,
         "own_win_buckets": {"lineages": {"B": "W_B", "N": "W_N", "O": "W_O"}, "max_share": 0.05}}


def build(root, counts=COUNTS, tamper=None):
    ident = json.dumps(IDENT)
    files = list(counts)          # producer list order (A, L, BC, ...), not recipe order
    eid = 1000
    body = b""
    for b in files:
        txt = HEAD
        for _ in range(counts[b]):
            eid += 1
            txt += f"a.zip\t{eid}.json\t2026-09-30\t{eid}\t0\t1\tbucket{b}\n"
        body += txt.encode()
    digest = hashlib.sha256(ident.encode() + b"\n" + body).hexdigest()
    d = os.path.join(root, digest[:12])
    os.makedirs(d)
    eid = 1000
    for b in files:
        with open(f"{d}/{b}.tsv", "w") as f:
            f.write(HEAD)
            for _ in range(counts[b]):
                eid += 1
                f.write(f"a.zip\t{eid}.json\t2026-09-30\t{eid}\t0\t1\tbucket{b}\n")
    eff = R.recompute_effective_shares(IDENT, counts)
    views = {}
    for lin, wb in IDENT["own_win_buckets"]["lineages"].items():
        if counts.get(wb):
            kk = 4 * eff["A"] * counts[wb] / counts["A"]
            x = min(.05, kk / (1 + kk))
            v = {k: round(val * (1 - x), 6) for k, val in eff.items()}
            v[wb] = round(x, 6)
            views[lin] = {"bucket": wb, "n": counts[wb], "share": round(x, 6), "effective_shares": v}
    effective = {"schema": "effective-shares/v1", "generation": digest[:12], "recipe_sha256": "r" * 64,
                 "effective_shares": eff, "lineage_views": views, "files": [f"{b}.tsv" for b in files],
                 "cached_seats": dict(counts)}
    receipt = {"sha256": digest, "identity_json": ident, "effective_shares": eff, "recipe": {"sha256": "r" * 64},
               "buckets": {b: {"cached": n} for b, n in counts.items()}}
    if tamper:
        tamper(d, effective, receipt)
    json.dump(effective, open(f"{d}/EFFECTIVE-SHARES.json", "w"))
    json.dump(receipt, open(f"{d}/RECEIPT.json", "w"))
    return d


def refuses(fn, what):
    try:
        fn()
    except SystemExit as exc:
        print(f"  refused ({what}): {str(exc)[:110]}")
        return
    raise AssertionError(f"accepted: {what}")


def main():
    root = tempfile.mkdtemp()
    try:
        bd, order, shares, _p, _d = R.load_bucket_generation(build(root))
        assert order == ["A", "BC", "D", "E", "L", "W_N"], order
        assert abs(sum(shares.values()) - 1) < 1e-5, shares
        assert 0 < shares["W_N"] <= .05 and "W_B" not in shares and "W_O" not in shares, shares
        print("view N accepted:", order, shares)
        c0 = dict(COUNTS, W_N=0)
        bd, order, shares, _p, _d = R.load_bucket_generation(build(os.path.join(root, "z"), c0))
        assert order == ["A", "BC", "D", "E", "L"] and "W_N" not in shares
        print("no W_N seats: neutral shares", shares)

        def bump(d, e, r):
            e["lineage_views"]["N"]["effective_shares"]["W_N"] += .01
        refuses(lambda: R.load_bucket_generation(build(os.path.join(root, "t1"), tamper=bump)), "view share edited")

        def drop(d, e, r):
            e["lineage_views"].pop("N")
        refuses(lambda: R.load_bucket_generation(build(os.path.join(root, "t2"), tamper=drop)), "W_N cached, no view")

        def reorder(d, e, r):
            e["effective_shares"] = dict(reversed(list(e["effective_shares"].items())))
            r["effective_shares"] = dict(e["effective_shares"])
        bd, order, *_ = R.load_bucket_generation(build(os.path.join(root, "t3"), tamper=reorder))
        assert order[:5] == ["A", "BC", "D", "E", "L"], order
        print("sidecar key reorder: tiers unchanged", order)

        def edit_file(d, e, r):
            with open(f"{d}/D.tsv", "a") as f:
                f.write("a.zip\t9.json\t2026-09-30\t9\t0\t1\tbucketD\n")
            r["buckets"]["D"]["cached"] += 1
        refuses(lambda: R.load_bucket_generation(build(os.path.join(root, "t4"), tamper=edit_file)), "bucket file changed")
    finally:
        shutil.rmtree(root)
    print("PASS")


if __name__ == "__main__":
    main()
