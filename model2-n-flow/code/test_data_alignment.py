#!/usr/bin/env python3
"""Shard rows decode back to the original replay actions, and shard inputs equal a fresh actor-feature recompute.

  python3 test_data_alignment.py [--roster R] [--shards DIR] [--seats 6] [--steps 40] [--feature-steps 30]

For sampled roster seats (every tier, incl. a WDL-only row) the test reads the raw public replay (archive/member):
  labels  : for sampled boundaries t, shard (unit_tok, unit_qty, order_tok, order_qty) == data.encode_step_lenient(
            replay.steps[t+1][seat].action, bc.public_observation(replay, t, seat)) (the primitive-bc-v31 pairing
            observation[t] -> action[t+1]); codec.decode_step(shard labels, obs) == codec.canonical(original action)
            whenever the shard marks the step exact.
  features: for steps 0..F-1 of the first seat, data.episode_examples(v8) + data.add_v6 recomputed from the replay,
            compacted by nf_format, equal the shard's node/unit/prefix/history arrays bit for bit.
Shards come from --shards when given, else each .stp is compacted and round-tripped through write_seat/read_seat.
"""
import argparse
import json
import os
import random
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nf_format as NF  # noqa: E402

PBC = Path(os.environ.get("NFLOW_PBC_DIR", str(NF.PRIMITIVE_BC)))
sys.path.insert(0, str(PBC))


def replay_of(row):
    with zipfile.ZipFile(row["archive"]) as z:
        return json.loads(z.read(row["member"]))


def shard_of(row, shards, cols, tmp):
    if shards:
        return NF.read_seat(Path(shards) / NF.shard_name(row), cols)
    d = NF.compact_seat(NF.read_stp(row["path"]), cols, row)
    p = Path(tmp) / NF.shard_name(row)
    NF.write_seat(p, d, cols)
    return NF.read_seat(p, cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", default="/mnt/ai/scratch/farmNEAT/n-flow/roster/rows-nflow.1ad4d8296e84.tsv")
    ap.add_argument("--shards", default="")
    ap.add_argument("--seats", type=int, default=6)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--feature-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import codec
    import data
    from farm_neat.training import bc
    cols = NF.load_columns()
    rows = NF.read_roster(a.roster)
    rng = random.Random(a.seed)
    pick = []
    for t in (1, 2, 3):
        pool = [r for r in rows if r["tier"] == t and r["act_loss"]]
        pick += rng.sample(pool, max(1, (a.seats - 1) // 3))
    pick.append(rng.choice([r for r in rows if not r["act_loss"]]))
    fails, checked, exact_checked = [], 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        for si, row in enumerate(pick):
            rep = replay_of(row)
            sh = shard_of(row, a.shards, cols, tmp)
            seat = row["seat"]
            steps = list(sh["step"].astype(int))
            if steps != list(range(719)):
                fails.append((row["eid"], seat, "shard steps are not 0..718"))
                continue
            # seat scalars
            if int(sh["act_loss"]) != row["act_loss"] or int(sh["wdl"]) != row["wdl"]:
                fails.append((row["eid"], seat, "seat scalars differ from roster"))
            final = rep["rewards"] if "rewards" in rep else None
            if final is not None and None not in final:
                want = 2 if final[seat] > final[1 - seat] else 1 if final[seat] == final[1 - seat] else 0
                if want != int(sh["wdl"]):
                    fails.append((row["eid"], seat, f"wdl {int(sh['wdl'])} != replay rewards {final}"))
            for t in sorted(rng.sample(range(719), min(a.steps, 719))):
                obs = bc.public_observation(rep, t, seat)
                action = rep["steps"][t + 1][seat].get("action")
                lab = data.encode_step_lenient(action, obs)
                got = {k: sh[k][t] for k in ("unit_tok", "unit_qty", "order_tok", "order_qty")}
                want_u = np.full(25, -1)
                want_u[:len(lab["unit_tok"])] = lab["unit_tok"]
                want_uq = np.full(25, -1)
                want_uq[:len(lab["unit_qty"])] = lab["unit_qty"]
                want_o = np.full(11, -1)
                want_o[:len(lab["order_tok"])] = lab["order_tok"]
                want_oq = np.full(11, -1)
                want_oq[:len(lab["order_qty"])] = lab["order_qty"]
                for k, w in (("unit_tok", want_u), ("unit_qty", want_uq), ("order_tok", want_o), ("order_qty", want_oq)):
                    if not np.array_equal(got[k].astype(int), w):
                        fails.append((row["eid"], seat, t, k, got[k].tolist(), w.tolist()))
                if bool(sh["exact"][t]) != bool(lab.get("exact", True)):
                    fails.append((row["eid"], seat, t, "exact flag"))
                n_o = int((got["order_tok"] >= 0).sum())
                n_u = int((got["unit_tok"] >= 0).sum())
                dec = codec.decode_step({"unit_tok": [int(v) for v in got["unit_tok"][:n_u]],
                                         "unit_qty": [int(v) for v in got["unit_qty"][:n_u]],
                                         "order_tok": [int(v) for v in got["order_tok"][:n_o]],
                                         "order_qty": [int(v) for v in got["order_qty"][:n_o]]}, obs)
                if bool(sh["exact"][t]):
                    exact_checked += 1
                    if codec.canonical(dec, obs) != codec.canonical(action, obs):
                        fails.append((row["eid"], seat, t, "decoded shard action != canonical replay action"))
                checked += 1
            if si == 0 and a.feature_steps:
                F_ = a.feature_steps
                meta = {"episode_id": row["eid"], "seat": seat}
                exs = list(data.episode_examples(rep, seat, meta_base=meta, steps=list(range(F_)), v8=True))
                data.add_v6(exs, data.opponent_unit_tokens(rep, seat, list(range(F_))))
                keys = set(exs[0])
                arrs = {k: np.stack([np.asarray(e[k]) for e in exs]) for k in keys
                        if k not in ("meta", "unit_noop_mask_schema") and not isinstance(exs[0][k], (dict, str))
                        and len({np.asarray(e[k]).shape for e in exs}) == 1}
                ntok = max(e["rows"].shape[0] for e in exs)
                for k in ("rows", "knownness", "token_mask", "extra", "v6_token", "x8_token"):
                    arrs[k] = np.stack([np.pad(np.asarray(e[k]), [(0, ntok - np.asarray(e[k]).shape[0])] +
                                               [(0, 0)] * (np.asarray(e[k]).ndim - 1)) for e in exs])
                for k in ("unit_tok", "unit_qty", "order_tok", "order_qty"):
                    W = 25 if k.startswith("unit") else 11
                    arrs[k] = np.stack([np.pad(np.asarray(e[k]), (0, W - len(e[k])), constant_values=-1) for e in exs])
                arrs["exact"] = np.array([bool(e.get("exact", True)) for e in exs])
                arrs["step"] = np.arange(F_)
                fresh = {**NF.compact_tokens(arrs, cols), **NF.compact_units(arrs), **NF.compact_labels(arrs)}
                for k, v in fresh.items():
                    n = F_ - 2 if k.startswith(("next_", "opp_")) else F_   # t+1/t+2 targets need steps past the window
                    if not np.array_equal(v[:n], sh[k][:n]):
                        bad = np.argwhere(v[:n] != sh[k][:n])[:3].tolist()
                        fails.append((row["eid"], seat, "features", k, bad))
                checked += 1
            print(json.dumps({"eid": row["eid"], "seat": seat, "tier": row["tier"], "act_loss": row["act_loss"],
                              "fails_so_far": len(fails)}), flush=True)
    print(json.dumps({"seats": len(pick), "label_steps_checked": checked, "exact_decode_checked": exact_checked,
                      "fails": fails[:10]}))
    print("PASS" if not fails else f"FAIL {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
