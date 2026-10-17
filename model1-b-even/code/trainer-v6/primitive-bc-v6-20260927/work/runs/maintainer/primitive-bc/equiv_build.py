#!/usr/bin/env python3
"""Quantity-equivalence side-cars for the step cache (PRO-EQUIV-0928; standalone, not imported by any cache-hashed file).

Classes (RETURN-1 `equiv_labels.build_step`, the policy both RETURN-1 and RETURN-2 admit): for every labelled quantity at the
seat's own actor observation and DECODED teacher prefix, the bins with
  * the identical decoded wire count (codec aliases, e.g. bin 6 and bin 74 at reference 7), for every quantity token;
  * the same immediate own-unit effect for PICKUP / PLACE (shed-adjacent clip; quantity-free placement only on an UNOCCUPIED
    matching structure), executed through the engine's own unit-phase function on the evolving prefix;
  * the same SELL fill when own shed stock is provable (not after a same-product BUY_PRODUCT earlier in the queue).
BUY_* quantities keep identical-wire aliases only (fills depend on the hidden rival queue). exact=False labels, and every label
after a multi-item overflowing DROP (saved inventory order may be lost), keep identical-wire aliases only. Token sets are
singleton: no queue permutations, upkeep/route suffixes, hand permutations, no-op relabels (rejected by both returns).

Labels are recomputed with the cache's own functions (bc.public_observation + data.encode_step_lenient), for BOTH seats of
every episode in ROWS (the opponent seat supplies the opponent aux targets). Output: <eid>-<seat>.equiv.npz in the RETURN-1
NPZ container (allow_pickle=False; int16 labels; little-endian packed masks; SHA-256 per array; identity = episode, seat,
engine / codec / builder / data.py hashes, configuration). train.load_equiv recomputes the identity from its own tree.

  build:  python equiv_build.py build ROWS.tsv OUT_DIR [--procs 4] [--shard i/n]
  check:  python equiv_build.py check CACHE_DIR OUT_DIR [--aux]   (every cached seat joins; exit 1 on any missing/mismatch)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA, VERSION = "engine-effect-quantity-v1", 1
SLOTS = {"unit_tok": 25, "unit_qty": 25, "order_tok": 11, "order_qty": 11}
VOCABS = {"unit_tok": 44, "unit_qty": 75, "order_tok": 23, "order_qty": 75}
FIXED = {"boardSize": 10, "turnsPerDay": 24, "shedCapacity": 100, "maxMarketOrdersPerTurn": 10}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity_hashes():
    engine = importlib.util.find_spec("farm_neat.game.kaggriculture_engine_reference").origin
    return {"engine_sha256": _sha(engine), "codec_sha256": _sha(HERE / "codec.py"),
            "builder_sha256": _sha(Path(__file__).resolve()), "labels_sha256": _sha(HERE / "data.py")}


def dense(lab):
    import codec
    out = {}
    for k, n in SLOTS.items():
        a = np.full(n, -1, np.int16)
        v = list(lab[k])[:n]
        a[:len(v)] = v
        out[k] = a
    for p, has in (("unit", codec.UNIT_HAS_QTY), ("order", codec.ORDER_HAS_QTY)):
        tok, qty = out[p + "_tok"], out[p + "_qty"]
        if np.any((qty >= 0) != ((tok >= 0) & np.asarray(has)[tok.clip(0)])):
            raise ValueError(f"{p} quantity presence mismatch")
    return out


def build_step(obs, lab, E):
    """Masks for one actor step: RETURN-1 equiv_labels.build_step (conservative actor-only quantity classes), verbatim logic."""
    import codec
    if obs.get("player") not in (0, 1) or not 0 <= int(obs["step"]) <= 718:
        raise ValueError("invalid actor seat/step")
    y = dense(lab)
    masks = {k + "_set": np.eye(VOCABS[k], dtype=bool)[y[k].clip(0)] & (y[k] >= 0)[:, None] for k in SLOTS}
    exact = bool(lab.get("exact", True))
    wire = codec.decode_step(y, obs)
    farm, private = copy.deepcopy(obs["farms"][obs["player"]]), copy.deepcopy(obs["private"])
    commands = [wire["farmer"], *wire["hands"]]
    demand = {}
    for a in commands:
        if a[0] == "PLANT":
            demand[a[1]] = demand.get(a[1], 0) + 1
    blocked = {c for c, n in demand.items() if n > private["seeds"].get(c, 0)}
    values = {}

    def decoded(name, k=0):
        key = (name, k if name.startswith("PLACE:") else -1)
        if key not in values:
            op, ref = name.split(":")[0], codec.reference(name, obs, k)
            values[key] = np.asarray([codec.decode_qty(op, b, ref) for b in range(codec.QTY_BIN_COUNT)], np.int64)
        return values[key]

    order_ambiguous = False
    for k, a in enumerate(commands):
        b = int(y["unit_qty"][k])
        if b >= 0:
            name = codec.UNIT_VOCAB[int(y["unit_tok"][k])]
            op, item = name.split(":")
            v = decoded(name, k)
            mask = v == v[b]
            if exact and not order_ambiguous:
                pos = E._farmer_position(farm, k)
                inv = private["inventories"][k] if k < len(private["inventories"]) else {}
                adj = pos is not None and E._is_shed_adjacent(pos, 10)
                tile = farm["tiles"][pos[1]][pos[0]] if pos is not None else None
                if op == "PICKUP":
                    m = private["shed"].get(item, 0) if adj else 0
                elif (item in E.ANIMALS and isinstance(tile, dict)
                      and tile.get("kind") == E.ANIMALS[item]["structure"] and "animal" not in tile):
                    m = 0      # only an UNOCCUPIED matching structure ignores the quantity
                elif adj:
                    m = min(inv.get(item, 0), max(0, 100 - sum(private["shed"].values())))
                else:
                    m = 0
                mask = np.minimum(v, m) == min(int(v[b]), m)
            masks["unit_qty_set"][k] = mask
        if a[0] == "DROP":
            pos = E._farmer_position(farm, k)
            inv = private["inventories"][k] if k < len(private["inventories"]) else {}
            amounts = [n for n in inv.values() if n > 0]
            room = max(0, 100 - sum(private["shed"].values()))
            if pos is not None and E._is_shed_adjacent(pos, 10) and len(amounts) > 1 and 0 < room < sum(amounts):
                order_ambiguous = True
        applied = ["PASS"] if a[0] == "PLANT" and a[1] in blocked else a
        E._apply_unit_action(farm, private, k, applied, 10, int(obs["step"]) // 24, 24, 100)
    stock, unknown = dict(private["shed"]), set()
    for j, tok in enumerate(y["order_tok"]):
        if tok < 0 or tok == codec.STOP:
            break
        name, b = codec.ORDER_VOCAB[int(tok)], int(y["order_qty"][j])
        if b < 0:
            continue
        op, item = name.split(":")
        v = decoded(name)
        q = int(v[b])
        mask = v == q
        if exact and not order_ambiguous and op == "SELL" and item not in unknown:
            m = max(0, stock.get(item, 0))
            mask = np.minimum(v, m) == min(q, m)
            stock[item] = m - min(q, m)
        if op == "BUY_PRODUCT":
            unknown.add(item)
        masks["order_qty_set"][j] = mask
    return y, masks


def save(path, meta, rows):
    labels = {k: np.stack([r[0][k] for r in rows]) for k in SLOTS}
    arr = {"steps": np.asarray([r[2] for r in rows], np.int16), "exact": np.asarray([r[3] for r in rows], bool)}
    for k in SLOTS:
        m = np.stack([r[1][k + "_set"] for r in rows])
        y = labels[k]
        act, size = y >= 0, m.sum(-1)
        if (size[~act] != 0).any() or (size[act] == 0).any() or not np.take_along_axis(m, y.clip(0)[..., None], -1)[..., 0][act].all():
            raise ValueError(f"{path}: invalid {k} masks")
        arr[k] = y
        arr[k + "_bits"] = np.packbits(m, axis=-1, bitorder="little")
    meta = {**meta, "schema": SCHEMA, "version": VERSION, "bitorder": "little", "n": len(rows),
            "digests": {k: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for k, a in arr.items()}}
    arr["metadata"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), np.uint8)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arr)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def build_episode(row, out):
    import data
    import prefix  # noqa: F401  (installs the kaggle_environments stub the engine imports)
    from farm_neat.game import kaggriculture_engine_reference as E
    from farm_neat.training import bc
    archive, member, eid = row[0], row[1], int(row[3])
    todo = [s for s in (0, 1) if not (out / f"{eid}-{s}.equiv.npz").exists()]
    if not todo:
        return {"eid": eid, "status": "exists"}
    t0 = time.time()
    with zipfile.ZipFile(archive) as z:
        replay = json.loads(z.read(member))
    data.check_replay(replay)
    cfg = replay.get("configuration", {})
    if any(cfg.get(k, v) != v for k, v in FIXED.items()):
        raise ValueError("unsupported mechanics configuration")
    ident = identity_hashes()
    stats = {}
    for seat in todo:
        rows = []
        for t in range(data.STEPS):
            obs = bc.public_observation(replay, t, seat)
            lab = data.encode_step_lenient(replay["steps"][t + 1][seat].get("action"), obs)
            y, masks = build_step(obs, lab, E)
            rows.append((y, masks, t, bool(lab.get("exact", True))))
        meta = {"episode_id": str(eid), "seat": seat, **ident, "configuration": {k: cfg.get(k) for k in FIXED}}
        save(out / f"{eid}-{seat}.equiv.npz", meta, rows)
        stats[seat] = {"unit_multi": int(sum((r[1]["unit_qty_set"].sum(-1) > 1).sum() for r in rows)),
                       "order_multi": int(sum((r[1]["order_qty_set"].sum(-1) > 1).sum() for r in rows))}
    return {"eid": eid, "status": "ok", "s": round(time.time() - t0, 2), "multi": stats}


def _build_one(args):
    row, out = args
    try:
        return build_episode(row, Path(out))
    except Exception as e:  # noqa: BLE001 - reported per episode, never silently dropped
        return {"eid": row[3], "status": f"error: {e!r}"[:300]}


def cmd_build(a):
    import multiprocessing as mp
    rows = [l.rstrip("\n").split("\t") for l in open(a.rows) if l.strip()][1:]
    i, n = (int(x) for x in a.shard.split("/"))
    seen, todo = set(), []
    for j, r in enumerate(rows):
        if j % n == i and r[3] not in seen:     # one task per episode: both seats
            seen.add(r[3])
            todo.append((r, str(a.out)))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    log = open(Path(a.out) / f"EQUIV-BUILD-{i}of{n}.jsonl", "a")
    bad = 0
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for rec in pool.imap_unordered(_build_one, todo, chunksize=1):
            bad += not str(rec["status"]).startswith(("ok", "exists"))
            log.write(json.dumps(rec) + "\n")
            log.flush()
    print(json.dumps({"episodes": len(todo), "errors": bad}))
    return 1 if bad else 0


def cmd_check(a):
    """Every cached seat must join (own side-car; with --aux also the opponent's), exactly as the trainer will join it."""
    sys.path.insert(0, str(HERE))
    import step_cache
    import train
    step_cache.validate_metadata(Path(a.cache) / "CACHE.json")
    ok = bad = 0
    for f in sorted(Path(a.cache).glob("*.stp")):
        try:
            arrs, _ = step_cache.decode(f.read_bytes(), widen=True)
            train.equiv_merge(dict(arrs), f, a.out, aux=a.aux)
            ok += 1
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(json.dumps({"seat": f.stem, "error": repr(e)[:300]}), flush=True)
    print(json.dumps({"joined": ok, "failed": bad}))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("rows")
    b.add_argument("out")
    b.add_argument("--procs", type=int, default=4)
    b.add_argument("--shard", default="0/1")
    c = sub.add_parser("check")
    c.add_argument("cache")
    c.add_argument("out")
    c.add_argument("--aux", action="store_true")
    a = ap.parse_args()
    return cmd_build(a) if a.cmd == "build" else cmd_check(a)


if __name__ == "__main__":
    sys.exit(main())
