#!/usr/bin/env python3
"""Realised bucket shares of the N flow loader (review PRO BUCKETS r2).

  python3 test_bucket_draw.py                       synthetic roster, adversarial row weights
  python3 test_bucket_draw.py ROSTER PROBS [N]      live roster, e.g. A=0.3495,BC=0.3995,D=0.1498,E=0.0499,L=0.0513

Draws go through the real Streams._select/_draw. Every act bucket's realised share must be within 2 points of its
normalised probability; a multi-bucket roster without --bucket-probs, or with a bucket missing from it, must refuse.
"""
import collections
import os
import random
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nf_data  # noqa: E402
import nf_format as NF  # noqa: E402

TOL = 0.02


def stream(rows, probs):
    s = nf_data.Streams.__new__(nf_data.Streams)
    s.all_rows, s.shard_dir, s.stp_dir, s.refresh_s = rows, "", "", 0.0
    s.bucket_probs, s.wdl_frac = dict(probs), 0.1
    s._select()
    return s


def realised(s, n, seed=1):
    rng, c = random.Random(seed), collections.Counter()
    for _ in range(n):
        r = s._draw(rng)
        if r["act_loss"]:
            c[r["bucket"]] += 1
    tot = sum(c.values())
    return {b: c[b] / tot for b in c}


def check(s, probs, n):
    got = realised(s, n)
    tot = sum(probs[b] for b in s.groups)
    bad = []
    for b in sorted(s.groups):
        want = probs[b] / tot
        print(f"  {b:3s} rows {len(s.groups[b]):6d}  target {100 * want:5.1f}%  realised {100 * got.get(b, 0):5.1f}%")
        if abs(got.get(b, 0) - want) > TOL:
            bad.append(b)
    assert not bad, f"buckets off by > {100 * TOL:.0f} points: {bad}"


def synthetic():
    d = tempfile.mkdtemp()
    rows = []
    # bucket sizes and row weights chosen so the row-weight path would be badly wrong (E: many rows, heavy weight)
    for b, n, w in (("A", 300, 1.0), ("BC", 80, 0.5), ("D", 40, 0.5), ("E", 900, 3.0), ("L", 10, 0.2)):
        for i in range(n):
            eid = 100000 + len(rows)
            p = f"{d}/{eid}-0.stp"
            open(p, "wb").write(b"x")
            rows.append({"path": p, "eid": eid, "seat": 0, "weight": w, "tier": 1, "wdl": 2, "act_loss": 1,
                         "final_cash": 1.0, "opp_cash": 0.0, "bucket": b, "kind": "act", "wdl_w": 1.0, "fp": None})
    for i in range(50):
        eid = 900000 + i
        p = f"{d}/{eid}-1.stp"
        open(p, "wb").write(b"x")
        rows.append({"path": p, "eid": eid, "seat": 1, "weight": 1.0, "tier": 1, "wdl": 0, "act_loss": 0,
                     "final_cash": 0.0, "opp_cash": 1.0, "bucket": "", "kind": "wdl-loser", "wdl_w": 1.0, "fp": None})
    probs = {"A": 0.3495, "BC": 0.3995, "D": 0.1498, "E": 0.0499, "L": 0.0513}
    print("synthetic, 20000 draws:")
    check(stream(rows, probs), probs, 20000)
    for bad_probs, why in (({}, "no --bucket-probs"), ({"A": .5, "BC": .5, "D": .1, "E": .1}, "L missing")):
        try:
            stream(rows, bad_probs)
        except RuntimeError as exc:
            print(f"  refused ({why}): {exc}")
        else:
            raise AssertionError(f"multi-bucket roster accepted with {why}")
    one = [dict(r, bucket="T4") for r in rows]          # single-bucket t4 roster keeps the row-weight path
    stream(one, {})
    print("  single-bucket roster without probs: accepted (row weights)")


def live(roster, probs_arg, n):
    probs = {k: float(v) for k, v in (x.split("=") for x in probs_arg.split(",") if x)}
    rows = NF.read_roster(roster)
    for r in rows:
        r["path"] = "/dev/null"        # presence is not what is tested here
    print(f"{roster}, {n} draws:")
    check(stream(rows, probs), probs, n)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        live(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5000)
    else:
        synthetic()
    print("PASS")
