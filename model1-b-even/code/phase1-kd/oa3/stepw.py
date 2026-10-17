"""Module-level StepChunks subclass carrying per-step step_w (picklable for spawn DataLoader workers; gpu-ops 09-29)."""
import random

import numpy as np
import torch
import step_cache


def load_table(path):
    table = {}
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(head, line.rstrip("\n").split("\t")))
            e = table.setdefault((int(r["episode_id"]), int(r["seat"])), np.ones(720, dtype=np.float32))
            e[int(r["first_step"]):int(r["last_step"]) + 1] *= float(r["step_multiplier"])
    return table


class StepWChunks(step_cache.StepChunks):
    def __init__(self, cache_dir, chunk, seed, pool_seats=4, rows=None, table_path=None):
        super().__init__(cache_dir, chunk, seed, pool_seats=pool_seats, rows=rows)
        self.table = load_table(table_path)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = random.Random(self.seed * 9_999_991 + wid)
        seats = self._seats(wid, nw)
        table = self.table

        def load():
            path = next(seats)
            arrs, dt = step_cache.decode(path.read_bytes())
            idx = list(range(arrs["step"].shape[0]))
            if not idx:
                raise ValueError("empty cached seat has zero steps")
            eid, seat = (int(v) for v in path.stem.split("-"))
            steps = np.asarray(arrs["step"]).reshape(len(idx)).astype(np.int64)
            e = table.get((eid, seat))
            arrs = dict(arrs)
            arrs["step_w"] = e[steps] if e is not None else np.ones(len(idx), dtype=np.float32)
            rng.shuffle(idx)
            return [arrs, idx], dt

        live, dtypes = [], None
        while len(live) < self.pool_seats:
            e, dtypes = load()
            live.append(e)
        while True:
            take = []
            for _ in range(self.chunk):
                j = rng.randrange(len(live))
                arrs, idx = live[j]
                take.append((arrs, idx.pop()))
                if not idx:
                    live[j], dtypes = load()
            yield step_cache.collate_steps(take, dtypes, self.noop_schema)
