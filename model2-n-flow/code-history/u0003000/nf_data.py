"""N flow training stream loader: B seat streams walk complete games in chunks of L steps (truncated BPTT with the
recurrent memory carried across chunks by the trainer).

Each DataLoader worker owns `streams` streams; the trainer concatenates one item per worker (items arrive in worker
round-robin order) into a [B = workers * streams, L] batch and keeps a memory bank per global stream index.
A stream draws a seat (act rows with probability 1 - wdl_frac, WDL-only rows otherwise; within each kind in
proportion to the roster weight, so the declared tier mass shares hold), walks it from step 0 to the end, then draws
another. `reset[s]` marks a chunk that starts a new seat (memory must restart from zero = true game start).
Only the FIRST seat of every stream starts at a random chunk (to decorrelate streams); its memory cannot be
reconstructed, so all its chunks carry warm=True and the trainer masks their losses.
Source: compact shards (<eid>-<seat>.nfz, nf_format) or, when no shard dir is given, the v11 .stp files compacted
on the fly by the same reference encoder.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch

import nf_format as NF

STEP_KEYS = None   # filled on first seat: every per-step array key


def load_seat(row, shard_dir, cols):
    """A seat's compact arrays: its .nfz shard when the row has one, else its .stp compacted by the same encoder."""
    if shard_dir and row.get("_src", "nfz") == "nfz":
        return NF.check_seat(NF.read_seat(Path(shard_dir) / NF.shard_name(row), cols), row)
    return NF.check_seat(NF.compact_seat(NF.read_stp(row["path"]), cols, row), row)


def step_keys(d):
    return [k for k, v in d.items() if v.ndim >= 1 and k not in NF.SEAT_KEYS]


class Streams(torch.utils.data.IterableDataset):
    def __init__(self, roster, shard_dir, cols, *, streams, chunk, seed, wdl_frac=0.1, first_random=True,
                 stp_dir="", refresh_s=600.0, bucket_probs=None):
        self.all_rows = NF.read_roster(roster) if isinstance(roster, (str, os.PathLike)) else list(roster)
        self.shard_dir, self.cols, self.stp_dir = shard_dir, cols, stp_dir
        self.streams, self.chunk, self.seed = int(streams), int(chunk), int(seed)
        self.wdl_frac, self.first_random, self.refresh_s = float(wdl_frac), bool(first_random), float(refresh_s)
        self.bucket_probs = dict(bucket_probs or {})     # owner 09-30: pick a bucket, then a seat uniformly within it
        t0 = time.time()
        while True:
            try:
                self._select(verbose=True)
                break
            except ValueError as exc:     # staging still in progress: wait for the first action rows to land
                if not stp_dir or time.time() - t0 > 7200:
                    raise
                print(f"nf_data: waiting for data ({exc})", flush=True)
                time.sleep(30)

    def _select(self, verbose=False):
        """Rows with a .nfz shard use it; with stp_dir (or no shard dir) the others fall back to their .stp when it
        exists; rows with neither wait (re-listed every refresh_s: shards that land later join the pool)."""
        rows, n_nfz, n_stp = [], 0, 0
        have = set(os.listdir(self.shard_dir)) if self.shard_dir else set()
        for r in self.all_rows:
            r = dict(r)
            if self.stp_dir:
                r["path"] = f"{self.stp_dir}/{r['eid']}-{r['seat']}.stp"
            if self.shard_dir and NF.shard_name(r) in have:
                r["_src"] = "nfz"; n_nfz += 1
            elif (not self.shard_dir or self.stp_dir) and os.path.exists(r["path"]):
                r["_src"] = "stp"; n_stp += 1
            else:
                continue
            rows.append(r)
        self.rows = rows
        self.act = [r for r in rows if r["act_loss"]]
        self.wdl = [r for r in rows if not r["act_loss"]]
        self._listed = time.time()
        if verbose or len(rows) != getattr(self, "_n_rows", len(rows)):
            print(f"nf_data: {len(rows)} of {len(self.all_rows)} roster rows usable ({n_nfz} .nfz, {n_stp} .stp); "
                  f"act {len(self.act)} wdl-only {len(self.wdl)}", flush=True)
        self._n_rows = len(rows)
        self.act = [r for r in self.rows if r["act_loss"]]
        self.wdl = [r for r in self.rows if not r["act_loss"]]
        if not self.act:
            raise ValueError("roster has no action rows")
        if any(r["wdl"] != 2 for r in self.act):
            raise ValueError("action rows must all be winning seats")

    def _load(self, rng):
        """Draw a seat and load it; a seat that fails to load is reported and replaced (never kills the worker)."""
        for _ in range(20):
            row = self._draw(rng)
            try:
                return row, load_seat(row, self.shard_dir, self.cols)
            except Exception as exc:  # noqa: BLE001
                print(f"nf_data: skip {row['eid']}-{row['seat']} ({row.get('_src')}): {type(exc).__name__}: {exc}"[:300],
                      flush=True)
        raise RuntimeError("20 consecutive seat loads failed")

    def _draw(self, rng):
        if (self.shard_dir or self.stp_dir) and self.refresh_s and time.time() - self._listed > self.refresh_s:
            try:
                self._select()
            except ValueError:
                self._listed = time.time()
        if self.wdl and rng.random() < self.wdl_frac:
            return rng.choices(self.wdl, weights=[r["weight"] for r in self.wdl], k=1)[0]
        if self.bucket_probs:
            groups = {}
            for r in self.act:
                groups.setdefault(r.get("bucket", ""), []).append(r)
            names = [b for b in self.bucket_probs if groups.get(b)]      # empty buckets: probability renormalised away
            if names:
                b = rng.choices(names, weights=[self.bucket_probs[b] for b in names], k=1)[0]
                return rng.choice(groups[b])
        pool = self.act
        w = [r["weight"] for r in pool]
        return rng.choices(pool, weights=w, k=1)[0]

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = random.Random(self.seed * 1_000_003 + wid * 7919 + 17)
        S, L = self.streams, self.chunk
        state = []
        for s in range(S):
            row, d = self._load(rng)
            n = d["step"].shape[0]
            c0 = rng.randrange(max(1, (n + L - 1) // L)) if self.first_random else 0
            state.append({"row": row, "d": d, "c": c0, "warm": self.first_random and c0 > 0, "reset": True})
        keys = step_keys(state[0]["d"])
        seq = 0
        while True:
            out = {k: [] for k in keys}
            valid, reset, warm, act, wdl, eid, seat, wdl_w = [], [], [], [], [], [], [], []
            for st in state:
                d, c = st["d"], st["c"]
                n = d["step"].shape[0]
                a, b = c * L, min(n, (c + 1) * L)
                for k in keys:
                    v = d[k][a:b]
                    if b - a < L:
                        pad = np.zeros((L - (b - a),) + v.shape[1:], v.dtype)
                        if k in ("unit_tok", "unit_qty", "order_tok", "order_qty", "next_unit_tok", "opp_unit_tok",
                                 "opp_order_tok", "prev_unit", "prev_order", "unit_pos", "carrier_pos"):
                            pad -= 1
                        if k == "step":
                            pad = np.arange(b, b + L - (b - a), dtype=v.dtype).clip(max=718)
                        v = np.concatenate((v, pad), 0)
                    out[k].append(v)
                vm = np.zeros(L, bool)
                vm[:b - a] = True
                valid.append(vm)
                reset.append(st["reset"])
                warm.append(st["warm"])
                act.append(bool(st["row"]["act_loss"]))
                wdl.append(int(st["row"]["wdl"]))
                wdl_w.append(float(st["row"].get("wdl_w", 1.0)))
                eid.append(int(st["row"]["eid"]))
                seat.append(int(st["row"]["seat"]))
                st["reset"] = False
                st["c"] += 1
                if st["c"] * L >= n:          # seat finished: next chunk starts a new seat from step 0
                    row, d_new = self._load(rng)
                    st.update(row=row, d=d_new, c=0, warm=False, reset=True)
            batch = {k: torch.from_numpy(np.stack(v)) for k, v in out.items()}
            batch.update(valid=torch.from_numpy(np.stack(valid)), reset=torch.tensor(reset), warm=torch.tensor(warm),
                         act_loss=torch.tensor(act), wdl=torch.tensor(wdl, dtype=torch.long),
                         wdl_w=torch.tensor(wdl_w, dtype=torch.float32),
                         eid=torch.tensor(eid, dtype=torch.long), seat=torch.tensor(seat, dtype=torch.long),
                         worker=torch.tensor(wid), seq=torch.tensor(seq))
            seq += 1
            yield batch


def loader(roster, shard_dir, cols, *, workers, streams, chunk, seed, wdl_frac=0.1, first_random=True, stp_dir="",
           bucket_probs=None):
    ds = Streams(roster, shard_dir, cols, streams=streams, chunk=chunk, seed=seed, wdl_frac=wdl_frac,
                 first_random=first_random, stp_dir=stp_dir, bucket_probs=bucket_probs)
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=torch.cuda.is_available(),
                                       persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
                                       multiprocessing_context="fork" if workers > 0 else None)


def gather(it, nw):
    """Concatenate one item per worker (round-robin order) into a full [B, L] batch."""
    items = [next(it) for _ in range(max(1, nw))]
    ws = [int(b["worker"]) for b in items]
    if nw > 1 and ws != list(range(nw)):
        raise RuntimeError(f"loader items out of worker order: {ws}")
    out = {}
    for k in items[0]:
        if k in ("worker", "seq"):
            continue
        out[k] = torch.cat([b[k] for b in items], 0)
    return out
