#!/usr/bin/env python3
"""N flow compact per-seat shard format (schema nflow-shard-v1) and its reference encoder.

One file per (episode, seat): <eid>-<seat>.nfz = numpy .npz (zip deflate, allow_pickle=False), self-describing:
every array has a leading step axis S (<= 719, boundary t = step[i]); seat scalars are 0-d arrays; "schema_json"
holds this module's SCHEMA plus the column schema hash.

The encoder is a pure projection of the primitive-BC trainer's per-step arrays (step_cache .stp == data.features +
codec.encode_step labels + data.add_v6 + prefix.teacher_arrays; pairing observation[t] -> action[t+1] exactly as
primitive-bc-v31-code/data.py), so training (from .stp) and serving (from data.features at play time) call the same
`compact_tokens` / `compact_units` functions:

  graph nodes: the observation.pack tokens mapped to FIXED slots (N_SLOTS = 296):
      0 GLOBAL | 1-2 FARM own,opp | 3-202 TILE role*100+y*10+x | 203-252 CARRIER role*25+ordinal |
      253-264 STOCK shed item | 265-269 STOCK seed crop | 270-278 MARKET item | 279-286 SHOP ordinal |
      287-294 CONSTANT (5 crops, 3 animals) | 295 CONTROL
  node features: per node TYPE a fixed column subset of src = [rows 80 | extra 152 | v6_token 8 | x8_token 354 |
      knownness 80] (674), chosen by `derive` (columns that vary within the type on a seat sample) and frozen in
      COLUMNS.json (hash recorded in every shard). Dropped columns are dropped identically at play time.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

SCHEMA = "nflow-shard-v1"
HERE = Path(__file__).resolve().parent
COLUMNS_PATH = HERE / "COLUMNS.json"
PRIMITIVE_BC = Path(os.environ.get("NFLOW_PBC_DIR",
                                   "/home/russ/Desktop/farmNEAT/work/runs/maintainer/primitive-bc-v1-deploy-v9-decode"))

# ------------------------------------------------------------------ fixed node layout
KINDS = ("GLOBAL", "FARM", "TILE", "CARRIER", "MARKET", "STOCK", "SHOP", "COMMAND", "ARGUMENT", "CREDIT", "CONTROL", "CONSTANT")
TYPES = ("GLOBAL", "FARM", "TILE", "CARRIER", "STOCK", "MARKET", "SHOP", "CONSTANT", "CONTROL")
TYPE_RANGES = {"GLOBAL": (0, 1), "FARM": (1, 3), "TILE": (3, 203), "CARRIER": (203, 253), "STOCK": (253, 270),
               "MARKET": (270, 279), "SHOP": (279, 287), "CONSTANT": (287, 295), "CONTROL": (295, 296)}
N_SLOTS = 296
MAX_UNITS = 25
N_ORDER_SLOTS = 11
SRC_WIDTH = 80 + 152 + 8 + 354 + 80
SRC_KEYS = (("rows", 80), ("extra", 152), ("v6_token", 8), ("x8_token", 354), ("knownness", 80))
ITEM_COL = 17          # rows[17:29] item one-hot (observation.ITEMS: 9 products then GOOSE, COW, SHEEP)
ROLE_COL = 12
XY_COL = 14
ORD_COL = 16

UNIT_KEYS = {  # key -> (stored dtype, trailing shape) ; all per step
    "unit_pos": (np.int8, (MAX_UNITS, 2)), "unit_extra": (np.float16, (MAX_UNITS, 12)),
    "x8_unit": (np.float16, (MAX_UNITS, 41)), "reach_eta": (np.int16, (MAX_UNITS, 100)),
    "unit_prefix": (np.float16, (MAX_UNITS, 27)), "order_prefix": (np.float16, (N_ORDER_SLOTS, 22)),
    "order_cash": (np.float16, (N_ORDER_SLOTS, 82)), "v6_job_deadline": (np.int8, (100, 9)),
    "v6_job_deadline_present": (np.bool_, (100, 9)),
    "prev_unit": (np.int8, (2, MAX_UNITS, 2)), "prev_order": (np.int8, (2, N_ORDER_SLOTS, 2)),
}
LABEL_KEYS = {
    "unit_tok": (np.int8, (MAX_UNITS,)), "unit_qty": (np.int8, (MAX_UNITS,)),
    "order_tok": (np.int8, (N_ORDER_SLOTS,)), "order_qty": (np.int8, (N_ORDER_SLOTS,)),
    "unit_loss_keep": (np.bool_, (MAX_UNITS,)), "exact": (np.bool_, ()), "step": (np.int16, ()),
    "next_unit_tok": (np.int8, (2, MAX_UNITS)), "opp_unit_tok": (np.int8, (3, MAX_UNITS)),
    "opp_order_tok": (np.int8, (3, N_ORDER_SLOTS)),
}
SEAT_KEYS = ("eid", "seat", "wdl", "final_cash", "opp_cash", "act_loss", "tier", "weight")
WDL = {"L": 0, "D": 1, "W": 2}


def type_of_slot():
    t = np.zeros(N_SLOTS, np.int64)
    for i, name in enumerate(TYPES):
        a, b = TYPE_RANGES[name]
        t[a:b] = i
    return t


def src_matrix(arrs):
    """[S, N, 674] float32 source columns from trainer arrays (keys rows/extra/v6_token/x8_token/knownness)."""
    parts = []
    for k, w in SRC_KEYS:
        a = np.asarray(arrs[k])
        if a.shape[-1] != w:
            raise ValueError(f"{k} width {a.shape[-1]} != {w}")
        parts.append(a.astype(np.float32))
    return np.concatenate(parts, -1)


def slot_index(rows, token_mask):
    """[S, N] fixed slot of every token (-1 = padding / unmapped). rows [S, N, 80] (any float dtype)."""
    r = np.asarray(rows)[..., :ITEM_COL + 12 + 9].astype(np.float32)   # kind, role, xy, ordinal, item, STOCK payload
    m = np.asarray(token_mask, bool) & (r[..., :12].max(-1) > .5)
    kind = r[..., :12].argmax(-1)
    role = (r[..., ROLE_COL + 1] > .5).astype(np.int64)
    x = np.clip(np.rint(r[..., XY_COL] * 9), 0, 9).astype(np.int64)
    y = np.clip(np.rint(r[..., XY_COL + 1] * 9), 0, 9).astype(np.int64)
    ordv = np.rint(r[..., ORD_COL] * 100).astype(np.int64)
    item = r[..., ITEM_COL:ITEM_COL + 12].argmax(-1)
    has_item = r[..., ITEM_COL:ITEM_COL + 12].max(-1) > .5
    out = np.full(kind.shape, -1, np.int64)
    K = {k: i for i, k in enumerate(KINDS)}
    out = np.where(m & (kind == K["GLOBAL"]), 0, out)
    out = np.where(m & (kind == K["FARM"]), 1 + role, out)
    out = np.where(m & (kind == K["TILE"]), 3 + role * 100 + y * 10 + x, out)
    out = np.where(m & (kind == K["CARRIER"]) & (ordv >= 0) & (ordv < MAX_UNITS), 203 + role * 25 + ordv, out)
    is_seed = r[..., 35 + 2] > .5          # STOCK payload (quantity, shed, seed)
    out = np.where(m & (kind == K["STOCK"]) & has_item & ~is_seed, 253 + item, out)
    out = np.where(m & (kind == K["STOCK"]) & has_item & is_seed & (item < 5), 265 + item, out)
    out = np.where(m & (kind == K["MARKET"]) & has_item & (item < 9), 270 + item, out)
    out = np.where(m & (kind == K["SHOP"]) & (ordv >= 0) & (ordv < 8), 279 + ordv, out)
    const_j = np.where(item < 5, item, item - 9 + 5)
    out = np.where(m & (kind == K["CONSTANT"]) & has_item & ((item < 5) | (item >= 9)), 287 + const_j, out)
    out = np.where(m & (kind == K["CONTROL"]), 295, out)
    return out


def load_columns(path=COLUMNS_PATH):
    blob = json.loads(Path(path).read_text())
    if blob.get("schema") != SCHEMA:
        raise ValueError("column schema mismatch")
    return blob


def columns_hash(cols):
    return hashlib.sha256(json.dumps(cols["types"], sort_keys=True).encode()).hexdigest()


def compact_tokens(arrs, cols):
    """Trainer token arrays [S, N, ...] -> {"node_<TYPE>": [S, n_type, c_type] float16, "node_present": [S, 296] bool,
    "carrier_pos": [S, 50, 2] int8}. Raises if two tokens claim one slot (layout contract broken)."""
    rows = np.asarray(arrs["rows"])
    S, N = rows.shape[:2]
    slot = slot_index(rows, arrs["token_mask"])             # [S, N]
    s_i, n_i = np.nonzero(slot >= 0)
    sl = slot[s_i, n_i]
    flat = s_i * N_SLOTS + sl
    if len(np.unique(flat)) != len(flat):
        raise ValueError("two tokens map to one node slot")
    present = np.zeros((S, N_SLOTS), bool)
    present[s_i, sl] = True
    perm = np.zeros((S, N_SLOTS), np.int64)                 # slot -> token index (absent slots zeroed below)
    perm[s_i, sl] = n_i
    offs, o = {}, 0
    for k, w in SRC_KEYS:
        offs[k] = (o, w)
        o += w
    srcs = {k: np.asarray(arrs[k]) for k in offs}
    out = {"node_present": present}
    for name in TYPES:
        a, b = TYPE_RANGES[name]
        c = np.asarray(cols["types"][name], np.int64)
        idx = perm[:, a:b, None]
        parts = []
        for k, (off, w) in offs.items():
            pos = c[(c >= off) & (c < off + w)] - off
            if not len(pos):
                continue
            src = srcs[k]
            if (b - a) * w > src.shape[1] * len(pos):      # few columns of many rows: select columns first
                g = np.take_along_axis(src[..., pos], idx, 1)
            else:
                g = np.take_along_axis(src, idx, 1)[..., pos]
            parts.append(g.astype(np.float16))
        block = np.concatenate(parts, -1)
        block[~present[:, a:b]] = 0
        out["node_" + name] = block
        if name == "CARRIER":
            xy = np.rint(np.take_along_axis(srcs["rows"][..., XY_COL:XY_COL + 2], idx, 1).astype(np.float32) * 9)
            out["carrier_pos"] = np.where(present[:, a:b, None], np.clip(xy, 0, 9), -1).astype(np.int8)
    return out


def compact_units(arrs):
    out = {}
    for k, (dt, shape) in {**UNIT_KEYS}.items():
        a = np.asarray(arrs[k])
        if a.shape[1:] != shape:
            raise ValueError(f"{k} shape {a.shape[1:]} != {shape}")
        if dt is np.float16:
            out[k] = a.astype(np.float16)
        elif dt is np.bool_:
            out[k] = a.astype(bool)
        else:
            info = np.iinfo(dt)
            if a.size and (a.min() < info.min or a.max() > info.max):
                raise ValueError(f"{k} out of {dt} range")
            out[k] = a.astype(dt)
    return out


def compact_labels(arrs):
    out = {}
    for k, (dt, shape) in LABEL_KEYS.items():
        a = np.asarray(arrs[k])
        if a.shape[1:] != shape:
            raise ValueError(f"{k} shape {a.shape[1:]} != {shape}")
        out[k] = a.astype(dt)
    return out


def compact_seat(arrs, cols, meta):
    """All per-step trainer arrays of one seat -> shard dict (without schema_json)."""
    valid = np.asarray(arrs["step_valid"], bool) & np.asarray(arrs["loss_mask"], bool)
    order = np.argsort(np.asarray(arrs["step"]))
    keep = order[valid[order]]
    sub = {k: np.asarray(v)[keep] for k, v in arrs.items()}
    out = {**compact_tokens(sub, cols), **compact_units(sub), **compact_labels(sub)}
    for k in SEAT_KEYS:
        out[k] = np.asarray(meta[k])
    return out


def content_hash(d):
    """sha256 over the decoded arrays (sorted keys; name, dtype, shape, raw bytes): identical for identical content
    even when two zlib builds compress differently."""
    h = hashlib.sha256()
    for k in sorted(d):
        a = np.ascontiguousarray(d[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def write_seat(path, shard, cols):
    """Deterministic .npz: members in sorted order, fixed zip timestamps, deflate level 6 (same bytes for the same
    content and zlib build; content_hash identifies content across zlib builds)."""
    path = Path(path)
    blob = dict(shard)
    blob["schema_json"] = np.asarray(json.dumps({"schema": SCHEMA, "columns_sha256": columns_hash(cols)}))
    tmp = path.with_suffix(f".tmp{os.getpid()}.npz")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for k in sorted(blob):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(blob[k]), allow_pickle=False)
            info = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    os.replace(tmp, path)


def read_seat(path, cols=None):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    info = json.loads(str(d.pop("schema_json")))
    if info.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {info.get('schema')!r}")
    if cols is not None and info.get("columns_sha256") != columns_hash(cols):
        raise ValueError(f"{path}: column schema differs from COLUMNS.json")
    return d


# ------------------------------------------------------------------ .stp access (trusted local cache files only)
def read_stp(path):
    sys.path.insert(0, str(PRIMITIVE_BC))
    import pickle
    import zlib
    d = pickle.loads(Path(path).read_bytes())
    if d.get("v") != 2:
        raise ValueError(f"unsupported step cache version {d.get('v')}")
    out = {}
    for k, (orig, stored, shape, z) in d["keys"].items():
        out[k] = np.frombuffer(zlib.decompress(z), dtype=stored).reshape(shape)
    return out


# ------------------------------------------------------------------ roster
def read_roster(path):
    """N flow roster TSV (roster.py): path eid seat weight tier wdl act_loss final_cash opp_cash."""
    rows = []
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            if line.strip():
                rows.append(dict(zip(head, line.rstrip("\n").split("\t"))))
    for r in rows:
        for k in ("eid", "seat", "tier", "wdl", "act_loss"):
            r[k] = int(r[k])
        for k in ("weight", "final_cash", "opp_cash"):
            r[k] = float(r[k])
        r["wdl_w"] = float(r.get("wdl_w", 1.0))
        r.setdefault("kind", "act" if r["act_loss"] else "wdl-loser")
        if r["act_loss"] and (r["kind"] != "act" or r["wdl"] != 2):
            raise ValueError(f"roster row {r['eid']}-{r['seat']}: action rows must be kind act and winners")
    return rows


def shard_name(r):
    return f"{r['eid']}-{r['seat']}.nfz"


# ------------------------------------------------------------------ CLI
def cmd_derive(a):
    rows = read_roster(a.roster)
    rng = np.random.default_rng(20260930)
    by_tier = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r)
    pick = []
    for t, rs in sorted(by_tier.items()):
        idx = rng.choice(len(rs), size=min(a.per_tier, len(rs)), replace=False)
        pick += [rs[i] for i in idx]
    lo = {name: None for name in TYPES}
    hi = {name: None for name in TYPES}
    seen = []
    for r in pick:
        arrs = read_stp(r["path"])
        src = src_matrix(arrs)
        slot = slot_index(arrs["rows"], arrs["token_mask"])
        for name in TYPES:
            a0, b0 = TYPE_RANGES[name]
            sel = (slot >= a0) & (slot < b0)
            if not sel.any():
                continue
            v = src[sel]
            mn, mx = v.min(0), v.max(0)
            lo[name] = mn if lo[name] is None else np.minimum(lo[name], mn)
            hi[name] = mx if hi[name] is None else np.maximum(hi[name], mx)
        seen.append(f"{r['eid']}-{r['seat']}")
    types = {}
    for name in TYPES:
        if lo[name] is None:
            types[name] = [0]
            continue
        vary = np.nonzero(hi[name] > lo[name])[0].tolist()
        types[name] = vary or [0]
    blob = {"schema": SCHEMA, "src_keys": [list(x) for x in SRC_KEYS], "src_width": SRC_WIDTH,
            "types": types, "widths": {k: len(v) for k, v in types.items()}, "sample_seats": seen,
            "rule": "columns whose value varies across the type's tokens in the sample (min < max)",
            "roster": str(a.roster)}
    Path(a.out).write_text(json.dumps(blob, indent=1) + "\n")
    print(json.dumps(blob["widths"]))


def cmd_build(a):
    cols = load_columns(a.columns)
    rows = read_roster(a.roster)
    i, n = (int(x) for x in a.shard.split("/"))
    todo = [r for j, r in enumerate(rows) if j % n == i]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / f"BUILD-{i}of{n}.jsonl", "a")
    t0 = time.time()
    for j, r in enumerate(todo):
        dest = out / shard_name(r)
        if dest.exists():
            continue
        if a.only_missing_stp and not os.path.exists(f"{a.stp_dir}/{r['eid']}-{r['seat']}.stp" if a.stp_dir else r["path"]):
            continue
        t = time.time()
        src = f"{a.stp_dir}/{r['eid']}-{r['seat']}.stp" if a.stp_dir else r["path"]
        try:
            shard = compact_seat(read_stp(src), cols, r)
            write_seat(dest, shard, cols)
            status = "ok"
        except Exception as e:  # reported, never silently dropped
            status = f"error: {e!r}"[:300]
        log.write(json.dumps({"eid": r["eid"], "seat": r["seat"], "status": status, "s": round(time.time() - t, 2)}) + "\n")
        log.flush()
        if (j + 1) % 25 == 0:
            print(f"{j + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


def cmd_manifest(a):
    cols = load_columns(a.columns)
    rows = read_roster(a.roster)
    out = Path(a.dir)
    files, missing = [], []
    for r in rows:
        p = out / shard_name(r)
        if not p.exists():
            missing.append(shard_name(r))
            continue
        files.append({"file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                      "content_sha256": content_hash(read_seat(p, cols)),
                      "tier": r["tier"], "act_loss": r["act_loss"], "kind": r["kind"], "wdl": r["wdl"], "wdl_w": r["wdl_w"]})
    tiers = {}
    for f in files:
        k = f"{f['kind']}:tier{f['tier']}"
        tiers[k] = tiers.get(k, 0) + 1
    man = {"schema": SCHEMA, "columns_sha256": columns_hash(cols),
           "roster": str(a.roster), "roster_sha256": hashlib.sha256(Path(a.roster).read_bytes()).hexdigest(),
           "n_files": len(files), "bytes": sum(f["bytes"] for f in files), "tier_counts": tiers,
           "act_rows_wins": sum(1 for f in files if f["act_loss"]), "wdl_only_rows": sum(1 for f in files if not f["act_loss"]),
           "kinds": {k: sum(1 for f in files if f["kind"] == k) for k in ("act", "wdl-loser", "own-loss")},
           "note": "act = winning seats with action/imitation loss; wdl-loser = losing seat of the same games, WDL/aux only; "
                   "own-loss = seats our teams lost, WDL/aux only, label L; wdl_w = WDL loss closeness weight (roster)",
           "missing": missing, "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("n_files", "bytes", "tier_counts", "act_rows_wins", "wdl_only_rows", "kinds")}),
          "missing", len(missing))


def cmd_compare(a):
    """Two shard dirs: content_sha256 (and file bytes) of every common file, or of --names."""
    cols = load_columns(a.columns)
    names = a.names.split(",") if a.names else sorted(set(os.listdir(a.dir_a)) & set(os.listdir(a.dir_b)))
    names = [n for n in names if n.endswith(".nfz")][:a.limit]
    res = {"compared": 0, "content_equal": 0, "bytes_equal": 0, "differ": []}
    for n in names:
        pa, pb = Path(a.dir_a) / n, Path(a.dir_b) / n
        ca, cb = content_hash(read_seat(pa, cols)), content_hash(read_seat(pb, cols))
        res["compared"] += 1
        res["content_equal"] += ca == cb
        res["bytes_equal"] += pa.read_bytes() == pb.read_bytes()
        if ca != cb:
            res["differ"].append(n)
    print(json.dumps(res))
    return 0 if res["compared"] and res["content_equal"] == res["compared"] else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("derive"); p.add_argument("roster"); p.add_argument("out"); p.add_argument("--per-tier", type=int, default=16)
    p = sub.add_parser("build"); p.add_argument("roster"); p.add_argument("out")
    p.add_argument("--columns", default=str(COLUMNS_PATH)); p.add_argument("--shard", default="0/1")
    p.add_argument("--stp-dir", default="", help="read <eid>-<seat>.stp from here instead of the roster path")
    p.add_argument("--only-missing-stp", action="store_true", help="skip rows whose .stp is absent (no error line)")
    p = sub.add_parser("manifest"); p.add_argument("roster"); p.add_argument("dir"); p.add_argument("--columns", default=str(COLUMNS_PATH))
    p = sub.add_parser("compare"); p.add_argument("dir_a"); p.add_argument("dir_b"); p.add_argument("--names", default="")
    p.add_argument("--limit", type=int, default=20); p.add_argument("--columns", default=str(COLUMNS_PATH))
    a = ap.parse_args()
    sys.exit({"derive": cmd_derive, "build": cmd_build, "manifest": cmd_manifest, "compare": cmd_compare}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
