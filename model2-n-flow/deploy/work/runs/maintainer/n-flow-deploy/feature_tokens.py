"""features.FeatureState.compute() -> K extra input tokens [K, F] + per-cell tile additions.

Layout (mirrors observation.pack: identity one-hots first, payload after, shared projection):

  K = 22 tokens, F = 128 columns, type ids TYPES = GLOBAL(0) LAG(1) PRODUCT(2) PRODUCER(3)
    slot 0          GLOBAL    theorem.global(46) | history.global_now(27) | history.prev_action(24)
    slots 1..4      LAG       history.global_lag row for lag 1, 4, 24, 72 (22)
    slots 5..13     PRODUCT   PRODUCTS order: theorem.product(43) | history.item_now(18) | history.item_lag(36)
    slots 14..21    PRODUCER  PRODUCERS order (5 crops, GOOSE, COW, SHEEP): theorem.structure(18) |
                              animals only: history.item_now(18) | history.item_lag(36) of that animal
                              (a crop's item history is already in its PRODUCT token; unknown here)
  columns [0,4) type one-hot, [4,16) item one-hot over observation.ITEMS, [16,20) lag one-hot,
  [20, 20+payload) payload, rest padding (0, known False).
  tile: [200, 31] theorem tile columns, row = role*100 + y*10 + x (role 0 = observing seat), which is
  PolicyNet's own cell index (model._tiles: cell = role*100 + y*BOARD + x).

How PolicyNet consumes it (model.py is owned by the other build, not edited here):
  __init__: self.feat = feature_tokens.FeatureTokenInput(width)
  encode(..., feat=None): the three edits in POLICYNET_PATCH (anchored on lines of the current
  PolicyNet.encode; inert when feat is None):
    1. after `is_tile, cell_of, tile_index = self._tiles(...)`: project the tokens and add the
       tile projection to each TILE token through its own cell index (cell_of);
    2. before `attn = ...`: append the K feature tokens LAST (after the G slot tokens, so the slot
       sources `src` and tile indices keep pointing at the same positions) and extend the mask;
    3. think='off': pool the value over the extended mask.
  feat = {"values": [B,K,F] float, "known": [B,K,F] bool, "tile": [B,200,TW] float} (collate()).
test_feature_tokens.py applies POLICYNET_PATCH to the source of the current PolicyNet.encode and
trains/decodes through it; it fails when an anchor line no longer exists.

Values are divided by fixed scales (never clipped); float16 storage is safe (see bench/tests).
"""
from __future__ import annotations

import numpy as np

import features as FT
from farm_neat.economy.heuristic.constants import ANIMALS, PRODUCTS

TOKEN_SCHEMA = "primitive-bc-feature-tokens-v1"
TYPES = ("GLOBAL", "LAG", "PRODUCT", "PRODUCER")
ITEMS = tuple(PRODUCTS) + tuple(ANIMALS)       # == observation.ITEMS
LAGS = (1, 4, 24, 72)                            # == history_features.LAGS
PRODUCERS = FT.TF.PRODUCERS
ID_TYPE, ID_ITEM, ID_LAG = 0, len(TYPES), len(TYPES) + len(ITEMS)
PAYLOAD = ID_LAG + len(LAGS)                     # 20
WIDTH = 128
TILE_WIDTH = len(FT.BLOCKS["theorem.tile"][1])   # 31
N_CELLS = 200

SLOTS = (("GLOBAL", None), *(("LAG", i) for i in range(len(LAGS))),
         *(("PRODUCT", i) for i in range(len(PRODUCTS))), *(("PRODUCER", i) for i in range(len(PRODUCERS))))
K = len(SLOTS)                                   # 22
TYPE_ID = np.asarray([TYPES.index(t) for t, _ in SLOTS], np.int64)
S_GLOBAL, S_LAG, S_PRODUCT, S_PRODUCER = 0, 1, 1 + len(LAGS), 1 + len(LAGS) + len(PRODUCTS)
_ANIMAL_SLOTS = [S_PRODUCER + PRODUCERS.index(a) for a in ANIMALS]
_ANIMAL_ITEMS = [ITEMS.index(a) for a in ANIMALS]


def _width(block):
    return len(FT.BLOCKS[block][1])


# segments: (slots, block, source rows (None = the 1-D global block), payload offset)
def _segments():
    segs, off = [], PAYLOAD
    for block in ("theorem.global", "history.global_now", "history.prev_action"):
        segs.append(((S_GLOBAL,), block, None if block == "theorem.global" else (0,), off))
        off += _width(block)
    segs.append((tuple(range(S_LAG, S_LAG + len(LAGS))), "history.global_lag", tuple(range(len(LAGS))), PAYLOAD))
    off = PAYLOAD
    for block in ("theorem.product", "history.item_now", "history.item_lag"):
        segs.append((tuple(range(S_PRODUCT, S_PRODUCT + len(PRODUCTS))), block, tuple(range(len(PRODUCTS))), off))
        off += _width(block)
    segs.append((tuple(range(S_PRODUCER, S_PRODUCER + len(PRODUCERS))), "theorem.structure",
                 tuple(range(len(PRODUCERS))), PAYLOAD))
    off = PAYLOAD + _width("theorem.structure")
    for block in ("history.item_now", "history.item_lag"):
        segs.append((tuple(_ANIMAL_SLOTS), block, tuple(_ANIMAL_ITEMS), off))
        off += _width(block)
    return tuple(segs)


SEGMENTS = _segments()
assert max(off + _width(b) for _, b, _, off in SEGMENTS) <= WIDTH

# identity template + names
_ID_VALUES = np.zeros((K, WIDTH), np.float32)
_ID_KNOWN = np.zeros((K, WIDTH), np.bool_)
_ID_KNOWN[:, :PAYLOAD] = True
COLUMN_NAMES = [[""] * WIDTH for _ in range(K)]
for _k, (_t, _r) in enumerate(SLOTS):
    _ID_VALUES[_k, ID_TYPE + TYPES.index(_t)] = 1
    if _t == "LAG":
        _ID_VALUES[_k, ID_LAG + _r] = 1
    elif _t == "PRODUCT":
        _ID_VALUES[_k, ID_ITEM + ITEMS.index(PRODUCTS[_r])] = 1
    elif _t == "PRODUCER":
        _ID_VALUES[_k, ID_ITEM + ITEMS.index(PRODUCERS[_r])] = 1
    for _j, _t2 in enumerate(TYPES):
        COLUMN_NAMES[_k][ID_TYPE + _j] = "id.type:" + _t2
    for _j, _it in enumerate(ITEMS):
        COLUMN_NAMES[_k][ID_ITEM + _j] = "id.item:" + _it
    for _j, _lag in enumerate(LAGS):
        COLUMN_NAMES[_k][ID_LAG + _j] = f"id.lag:{_lag}"
for _slots, _block, _rows, _off in SEGMENTS:
    for _k in _slots:
        for _j, _c in enumerate(FT.BLOCKS[_block][1]):
            COLUMN_NAMES[_k][_off + _j] = f"{_block}/{_c}"
COLUMN_NAMES = tuple(tuple(r) for r in COLUMN_NAMES)
SLOT_NAMES = tuple(t if r is None else f"{t}:{LAGS[r] if t == 'LAG' else (PRODUCTS if t == 'PRODUCT' else PRODUCERS)[r]}"
                   for t, r in SLOTS)

# ablation groups: name -> (block, column predicate); dropping zeroes values AND knownness
GROUPS = {**{b: (b, lambda c: True) for b in FT.BLOCKS},
          "history.item_now.prev": ("history.item_now", lambda c: c.startswith("prev."))}
PREV_ACTION_GROUPS = ("history.prev_action", "history.item_now.prev")   # previous own action (copy risk)
_DROP_CACHE = {}


def _drop_mask(drop):
    key = tuple(sorted(set(drop)))
    hit = _DROP_CACHE.get(key)
    if hit is None:
        keep = np.ones((K, WIDTH), np.bool_)
        keep_tile = True
        for g in key:
            block, pred = GROUPS[g]
            if block == "theorem.tile":
                keep_tile = False
                continue
            cols = [j for j, c in enumerate(FT.BLOCKS[block][1]) if pred(c)]
            for slots, b, _, off in SEGMENTS:
                if b == block:
                    for k in slots:
                        keep[k, [off + j for j in cols]] = False
        hit = _DROP_CACHE[key] = (keep, keep_tile)
    return hit


def tokens(features, drop=(), dtype=np.float32):
    """features.FeatureState.compute() dict -> {"values" [K,F], "known" bool [K,F],
    "type_id" int64 [K], "tile" [200, TILE_WIDTH]}. ``drop``: GROUPS names to ablate."""
    values = _ID_VALUES.copy()
    known = _ID_KNOWN.copy()
    kn = features["known"]
    for slots, block, rows, off in SEGMENTS:
        src = features[block]
        ks = kn[block]
        w = src.shape[-1]
        s0, s1 = slots[0], slots[-1] + 1
        if rows is None:
            values[s0, off:off + w] = src
            known[s0, off:off + w] = ks
        elif len(slots) == s1 - s0 and rows == tuple(range(rows[0], rows[0] + len(rows))):
            values[s0:s1, off:off + w] = src[rows[0]:rows[0] + len(rows)]
            known[s0:s1, off:off + w] = ks[rows[0]:rows[0] + len(rows)]
        else:
            values[list(slots), off:off + w] = src[list(rows)]
            known[list(slots), off:off + w] = ks[list(rows)]
    tile = features["theorem.tile"]
    if drop:
        keep, keep_tile = _drop_mask(drop)
        values *= keep
        known &= keep
        if not keep_tile:
            tile = np.zeros_like(tile)
    return {"values": values.astype(dtype, copy=False), "known": known, "type_id": TYPE_ID,
            "tile": np.asarray(tile, dtype)}


def collate(batch):
    """List of tokens() dicts -> stacked arrays [B, K, F] / [B, 200, TILE_WIDTH]."""
    return {"values": np.stack([b["values"] for b in batch]), "known": np.stack([b["known"] for b in batch]),
            "type_id": np.broadcast_to(TYPE_ID, (len(batch), K)).copy(),
            "tile": np.stack([b["tile"] for b in batch])}


# ----------------------------------------------------------------------------- pack alignment
def pack_tile_rows(token_ids):
    """observation.pack token_ids -> int64 [200]: pack row of each cell role*100+y*10+x."""
    out = np.full(N_CELLS, -1, np.int64)
    for i, ident in enumerate(token_ids):
        if ident.startswith("tile/"):
            _, role, xy = ident.split("/")
            x, y = (int(v) for v in xy.split(","))
            out[int(role) * 100 + y * 10 + x] = i
    if (out < 0).any():
        raise ValueError("pack must hold one TILE token per cell of both farms")
    return out


def align_tiles(tile, token_ids):
    """[200, TILE_WIDTH] -> [N, TILE_WIDTH] aligned with pack rows (zeros on non-tile tokens),
    for a consumer that adds a per-row projection instead of gathering by cell."""
    rows = pack_tile_rows(token_ids)
    out = np.zeros((len(token_ids), tile.shape[-1]), tile.dtype)
    out[rows] = tile
    return out


# ----------------------------------------------------------------------------- PolicyNet wiring
# (anchor, mode, code): mode "after"/"before" inserts `code` (indented like the anchor line) next
# to the unique line of PolicyNet.encode whose stripped text equals `anchor`; "replace" replaces
# the unique occurrence of `anchor` text.
POLICYNET_PATCH = (
    ("field=None, *,", "replace", "field=None, *, feat=None,"),
    ("is_tile, cell_of, tile_index = self._tiles(rows, token_mask)", "after",
     "if feat is not None:\n"
     "    ftok, tile_add = self.feat(feat[\"values\"], feat[\"known\"], feat[\"tile\"])\n"
     "    x = x + tile_add.to(x.dtype).gather(1, cell_of[..., None].expand(B, N, x.shape[-1])) * is_tile[..., None].to(x.dtype)"),
    ("attn = None if bool(mask.all()) else mask[:, None, None, :]", "before",
     "if feat is not None:\n"
     "    x = torch.cat((x, ftok.to(x.dtype)), 1)\n"
     "    mask = torch.cat((mask, mask.new_ones(B, ftok.shape[1])), 1)"),
    ("m = token_mask[..., None].to(mem.dtype)", "replace", "m = mask[..., None].to(mem.dtype)"),
)


def patch_encode_source(source):
    """Apply POLICYNET_PATCH to the (dedented) source text of PolicyNet.encode."""
    for anchor, mode, code in POLICYNET_PATCH:
        if mode == "replace":
            if source.count(anchor) != 1:
                raise ValueError("PolicyNet.encode anchor not unique/absent: " + anchor)
            source = source.replace(anchor, code)
            continue
        lines = source.split("\n")
        hits = [i for i, line in enumerate(lines) if line.strip() == anchor]
        if len(hits) != 1:
            raise ValueError("PolicyNet.encode anchor line not unique/absent: " + anchor)
        i = hits[0]
        indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        block = [indent + c for c in code.split("\n")]
        lines =lines[:i + 1] + block + lines[i + 1:] if mode == "after" else lines[:i] + block + lines[i:]
        source = "\n".join(lines)
    return source


# ----------------------------------------------------------------------------- torch reference
def FeatureTokenInput(width):
    """nn.Module: (values [B,K,F], known [B,K,F], tile [B,200,TW]) -> (tokens [B,K,w], tile_add [B,200,w]).
    Same form as PolicyNet's pack input (Linear(values masked by known) + Linear(known)); the tile
    projection starts at zero so the per-cell addition is initially a no-op."""
    import torch
    from torch import nn

    class _FeatureTokenInput(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.inp = nn.Linear(WIDTH, w)
            self.known_inp = nn.Linear(WIDTH, w, bias=False)
            self.tile = nn.Linear(TILE_WIDTH, w)
            nn.init.zeros_(self.tile.weight)
            nn.init.zeros_(self.tile.bias)

        def forward(self, values, known, tile):
            known = known.bool()
            v = values.float().masked_fill(~known, 0)
            return self.inp(v) + self.known_inp(known.float()), self.tile(tile.float())

    return _FeatureTokenInput(width)
