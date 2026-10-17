"""Per-unit index into the observation.pack rows: which row is each unit slot's own CARRIER row.

Unit slot i (codec.unit_positions: farmer, then hands in observation order; data.features
``unit_pos``) is the own carrier with ordinal i: observation.pack writes one CARRIER row per own
carrier with role column 12 set, ``ordinal/100`` in column 16 and ``x/9, y/9`` in columns 14-15
(observation.py ``_Rows.add``). ``carrier_rows`` finds that row from the pack columns alone and
checks its position against ``unit_pos``; an absent unit, a missing row, a position mismatch or two
own carrier rows with the same ordinal give -1.

Computed from the arrays every per-step example dict already holds (rows, token_mask, unit_pos),
so the step cache and data.py are unchanged. ``model.collate`` calls it once per batch; train.py
(``collate_steps``) and agent.py (``collate_inputs``) both collate through ``model.collate``, so
the train and play inputs are the same function of the same arrays. numpy only (no torch).
"""
from __future__ import annotations

import numpy as np

# observation.py layout (checked against observation.KINDS / PAYLOAD / WIDTH in test_unit_index.py)
WIDTH = 80
CARRIER_KIND_COL = 3            # observation.KINDS.index("CARRIER")
OWN_ROLE_COL = 12               # role one-hot: 12 = own farm, 13 = opponent
X_COL, Y_COL = 14, 15           # x/9, y/9
ORD_COL = 16                    # ordinal/100
XY_SCALE, ORD_SCALE = 9, 100
PAYLOAD = 35                    # observation.PAYLOAD: value columns start here
CARRIER_FIELDS = 26             # len(observation.FIELDS["CARRIER"]): carried x12, key_rank x12, farmer, shed_access


def carrier_rows(rows, token_mask, unit_pos):
    """rows [..., N, 80] (float16/32 pack rows), token_mask [..., N] bool, unit_pos [..., U, 2]
    int (-1 = absent) -> int64 [..., U]: the pack row index of each unit slot's own CARRIER row,
    -1 where the slot has no unit or no single own carrier row with that ordinal at that position."""
    rows = np.asarray(rows)
    mask = np.asarray(token_mask, np.bool_)
    pos = np.asarray(unit_pos, np.int64)
    if rows.shape[-1] != WIDTH or rows.shape[:-1] != mask.shape or pos.shape[-1] != 2 \
            or pos.shape[:-2] != rows.shape[:-2]:
        raise ValueError("rows [..., N, 80], token_mask [..., N] and unit_pos [..., U, 2] required")
    lead, N, U = rows.shape[:-2], rows.shape[-2], pos.shape[-2]
    r = rows.reshape(-1, N, WIDTH)
    m = mask.reshape(-1, N)
    p = pos.reshape(-1, U, 2)
    B = r.shape[0]
    out = np.full((B, U), -1, np.int64)
    b, n = np.nonzero((r[..., CARRIER_KIND_COL] > 0.5) & (r[..., OWN_ROLE_COL] > 0.5) & m)
    if b.size:
        cols = r[b, n][:, [ORD_COL, X_COL, Y_COL]].astype(np.float32)
        o = np.rint(cols[:, 0] * ORD_SCALE).astype(np.int64)
        x = np.rint(cols[:, 1] * XY_SCALE).astype(np.int64)
        y = np.rint(cols[:, 2] * XY_SCALE).astype(np.int64)
        ok = (o >= 0) & (o < U)
        b, n, o, x, y = b[ok], n[ok], o[ok], x[ok], y[ok]
        count = np.zeros((B, U), np.int64)
        np.add.at(count, (b, o), 1)
        hit = (p[b, o, 0] == x) & (p[b, o, 1] == y) & (count[b, o] == 1)
        out[b[hit], o[hit]] = n[hit]
    return out.reshape(*lead, U)


def attach(example):
    """Adds example['carrier_row'] (int64 [U]) to a per-step example dict; returns it."""
    example["carrier_row"] = carrier_rows(example["rows"], example["token_mask"], example["unit_pos"])
    return example


def coverage(carrier_row, unit_pos):
    """-> dict of counts: present unit slots, resolved among them, and absent slots with a row."""
    cr = np.asarray(carrier_row)
    present = np.asarray(unit_pos)[..., 0] >= 0
    return {"present": int(present.sum()), "resolved": int(((cr >= 0) & present).sum()),
            "absent_with_row": int(((cr >= 0) & ~present).sum())}
