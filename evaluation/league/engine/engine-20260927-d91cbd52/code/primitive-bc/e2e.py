"""Primitive-BC v3 end-to-end checks: agent-path latency gate, cold load, CPU smoke fits.

Commands (pinned interpreter, CPU only):

  make-weights  --out W.pt [--storage fp32|fp16] [--head-std 0.5] [--set key=value ...]
      Default-config PolicyNet (standard init, torch seed 0) exported as agent weights; the
      output heads are redrawn N(0, head_std) so greedy decisions vary (zero heads always pick
      index 0).  Random weights: latency only.
  latency       --weights W.pt --variant fp32|int8 [--episodes 2] [--seats 0,1] [--units 16]
                [--orders 11] [--exact] --out L.json
      Whole agent step through agent.Agent on every step of complete public episodes: raw
      replay observation -> actor view -> canonical observation -> pack -> FeatureState ->
      feature tokens -> collate -> PolicyNet.act (units forced to `units` present, all order
      slots decoded when orders >= 11; --exact: every decision a quantity token with QTY_EXACT
      and all digit slots) -> codec.decode_step -> sanitize -> record_action.  The agent runs
      teacher-forced with the replay's canonical label action (so the feature history follows
      the real game); the model's own decision is still decoded every step.  int8 = torch
      dynamic quantisation of nn.Linear (qint8) after the load.  Timing = agent.Agent.ms (wall)
      and the calling thread's CPU time per step (time.thread_time; less sensitive to host load).
  cold          --weights W.pt --variant fp32|int8 [--runs 3] --out C.json
      Fresh processes (the caller supplies the CPU quota): interpreter start -> import agent ->
      Agent(weights) -> first call (synchronous load + first step).  Wall time per process is
      measured by this parent; the child reports import / load / first-step times.
  smoke-data    --days D1 D2 --episodes 40 --heldout-episodes 4 --workers 8 --out CACHE
      data.iter_perspectives over the two days' archives (train split, strength_top 0.6, both
      seats, seeded shuffle, first `episodes` tasks) -> CACHE/train.pkl; held-out perspectives
      (split 'heldout' of the same days, same filter) -> CACHE/heldout.pkl; ids, counts and the
      training label counts for the most-frequent baseline -> CACHE/SMOKE-DATA.json.
  smoke-fit     --cache CACHE --think field|off|carry|fsq --updates N --threads 4 --out RUN
      train.fit on the cached perspectives (fixed order: CachedSource, same seed for every arm)
      at a reduced model size, held-out = every step of every cached held-out perspective,
      streamed from step 0 with the carried state (train.evaluate).
  smoke-report  --cache CACHE --runs RUN... --out R.json
      Final held-out metrics of each run next to the most-frequent baseline (per slot mode of
      each head over the training labels, scored with the same masks by train.step_loss) and a
      copy-previous-unit-token baseline.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import gc
import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

SMALL = dict(tok_width=128, trunk_latents=16, trunk_width=256, trunk_layers=3, trunk_hidden=768, trunk_heads=4,
             field_dc=32, dec_width=128, dec_layers=2, dec_heads=4, dec_hidden=384)
STEPS = 719


def _torch():
    import torch
    return torch


def pct(values):
    a = np.asarray(values, np.float64)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "max": round(float(a.max()), 2), "mean": round(float(a.mean()), 2)}


def cpu_info():
    info = {"loadavg": list(os.getloadavg())}
    try:
        cg = open("/proc/self/cgroup").read().strip().split("::")[-1]
        info["cpu_max"] = Path("/sys/fs/cgroup" + cg + "/cpu.max").read_text().strip()
    except OSError:
        info["cpu_max"] = None
    return info


def latest_archives(n=1):
    import data
    return data.default_archives()[-n:]


# ----------------------------------------------------------------------------- weights
def make_weights(out, *, storage="fp32", seed=0, head_std=0.5, overrides=None):
    torch = _torch()
    import agent as A
    import model as M
    torch.manual_seed(seed)
    net = M.PolicyNet(M.ModelConfig(**(overrides or {}))).eval()
    if head_std:
        for head in (net.unit_head, net.order_head, net.unit_qty.head, net.order_qty.head, net.digits.head):
            torch.nn.init.normal_(head.weight, std=head_std)
    torch.save(A.export_weights(net, storage=storage), str(out))
    return {"weights": str(out), "storage": storage, "bytes": Path(out).stat().st_size,
            "parameters": net.parameter_count(), "head_std": head_std, "overrides": overrides or {},
            "size": M.size_report(net.config)}


def int8_dynamic(policy):
    torch = _torch()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.ao.quantization.quantize_dynamic(policy, {torch.nn.Linear}, dtype=torch.qint8)


def force_decode(policy, *, units=16, orders=11, exact=False):
    """Instance-level act wrapper: `units` present units, every order slot decoded (orders >=
    n_order_slots), optionally every decision QTY_EXACT with all digit slots."""
    import model as M
    orig = type(policy).act.__get__(policy)
    full = orders >= policy.O

    @functools.wraps(orig)
    def act(inputs, state=None, **kw):
        return orig(M.force_units(inputs, units) if units else inputs, state, force_orders=full, force_exact=exact, **kw)
    policy.act = act
    return policy


# ----------------------------------------------------------------------------- latency (agent path)
def latency(weights, *, variant="fp32", archives=None, episodes=2, seats=(0, 1), units=16, orders=11, exact=False,
            steps=STEPS, log=print):
    torch = _torch()
    import agent as A
    import data as D
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    archives = archives or latest_archives()
    members = D.list_members(archives, split="all")[:episodes]
    t0 = time.perf_counter()
    policy = A.load_policy(str(weights))
    if variant == "int8":
        policy = int8_dynamic(policy)
    load_s = time.perf_counter() - t0
    force_decode(policy, units=units, orders=orders, exact=exact)
    games = []
    all_ms, all_cpu = [], []
    for archive, member, day, eid in members:
        replay = D.read_replay(archive, member)
        labels = D.canonical_labels(replay)
        cfg = replay["configuration"]
        for seat in seats:
            actor = A.Agent("preloaded", policy=policy, log=False, threads=1)     # gc.freeze() here
            hands, cpu = [], []
            for t in range(steps):
                obs = replay["steps"][t][seat]["observation"]
                c0 = time.thread_time()
                actor(obs, cfg, teacher_action=labels.history[seat][t])
                cpu.append((time.thread_time() - c0) * 1e3)
                hands.append(len(obs["farms"][seat].get("hands", [])))
            ms = list(actor.ms)
            all_ms += ms
            all_cpu += cpu
            g = {"episode": eid, "day": day, "seat": seat, "steps": len(ms), **pct(ms), "first_ms": round(ms[0], 2),
                 "cpu_ms": pct(cpu),
                 "over_150ms": int(sum(m > 150 for m in ms)), "fallbacks": actor.fallbacks,
                 "deadline_hits": actor.deadline_hits, "errors": actor.errors[:3], "real_hands_max": max(hands),
                 "loadavg": list(os.getloadavg())}
            games.append(g)
            log(json.dumps(g))
            del actor
            gc.unfreeze()
            gc.collect()
    total = pct(all_ms)
    return {"schema": "primitive-bc-e2e-latency-v1", "weights": str(weights), "variant": variant,
            "units": units, "orders": orders, "force_exact": exact, "path": "agent.Agent teacher-forced (canonical labels)",
            "load_s": round(load_s, 3), "games": games, "all": total, "all_cpu_ms": pct(all_cpu),
            "over_150ms": int(sum(m > 150 for m in all_ms)), "verdict_max_le_150ms": total.get("max", 1e9) <= 150,
            "torch_threads": torch.get_num_threads(), **cpu_info()}


COLD_CHILD = r"""
import json, os, sys, time
t0 = time.perf_counter()
sys.path[:0] = [{src!r}, {here!r}]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import agent as A
import data as D
t1 = time.perf_counter()
member = D.list_members(D.default_archives()[-1:], split="all")[0]
replay = D.read_replay(member[0], member[1])
obs = replay["steps"][0][0]["observation"]
t2 = time.perf_counter()
actor = A.Agent({weights!r}, background=False, log=False, threads=1)
if {variant!r} == "int8":
    import torch
    t3 = time.perf_counter()
    actor.policy = A.load_policy({weights!r})
    import warnings
    warnings.simplefilter("ignore")
    actor.policy = torch.ao.quantization.quantize_dynamic(actor.policy, {{torch.nn.Linear}}, dtype=torch.qint8)
    actor._finish_load(t3)
    t4 = time.perf_counter()
a = actor(obs, replay["configuration"])
t5 = time.perf_counter()
print(json.dumps({{"import_s": round(t1 - t0, 3), "replay_read_s": round(t2 - t1, 3),
                  "load_ms": actor.load_ms, "first_call_ms": round(actor.ms[0], 1),
                  "fallbacks": actor.fallbacks, "errors": actor.errors,
                  "in_process_s": round(t5 - t0 - (t2 - t1), 3)}}))
"""


def cold(weights, *, variant="fp32", runs=3, python=None, log=print):
    """Fresh-process cold start; wall_s = process launch to exit (incl. interpreter start and the
    replay read, which a Kaggle run does not have: in_process_s excludes the replay read)."""
    python = python or sys.executable
    code = COLD_CHILD.format(src=str(ROOT / "src"), here=str(HERE), weights=str(weights), variant=variant)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    out = []
    for i in range(runs):
        t0 = time.perf_counter()
        p = subprocess.run([python, "-B", "-c", code], capture_output=True, text=True, env=env)
        wall = time.perf_counter() - t0
        try:
            row = json.loads(p.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            row = {"stderr": p.stderr[-2000:]}
        row.update(run=i, wall_s=round(wall, 3), returncode=p.returncode, loadavg=list(os.getloadavg()))
        out.append(row)
        log(json.dumps(row))
    walls = [r["wall_s"] for r in out]
    return {"schema": "primitive-bc-e2e-cold-v1", "weights": str(weights), "variant": variant,
            "bytes": Path(weights).stat().st_size, "runs": out, "wall_s_max": max(walls),
            "verdict_le_5s": max(walls) <= 5.0, **cpu_info()}


# ----------------------------------------------------------------------------- smoke data
class CachedSource:
    """train.fit source over a fixed list of perspectives: epoch e visits them in a
    random.Random(seed + e) permutation; deterministic, so resume is exact."""

    exact_resume = True

    def __init__(self, persps, seed=0):
        self.persps, self.seed = list(persps), int(seed)
        self.epoch, self.index, self.next_pid = 0, 0, 0
        self._order = None

    def _perm(self, epoch):
        order = list(range(len(self.persps)))
        random.Random(self.seed * 1_000_003 + epoch).shuffle(order)
        return order

    def next(self):
        if self._order is None:
            self._order = self._perm(self.epoch)
        if self.index >= len(self._order):
            self.epoch, self.index = self.epoch + 1, 0
            self._order = self._perm(self.epoch)
        p = self.persps[self._order[self.index]]
        out = (self.next_pid, (self.epoch, self.index), p)
        self.next_pid += 1
        self.index += 1
        return out

    def position(self):
        return {"epoch": self.epoch, "index": self.index, "next_pid": self.next_pid}

    def restore(self, pos, wanted):
        found = {pid: self.persps[self._perm(e)[i]] for pid, (e, i) in wanted.items()}
        self.epoch, self.index, self.next_pid = pos["epoch"], pos["index"], pos["next_pid"]
        self._order = self._perm(self.epoch)
        return found

    def close(self):
        pass


def day_archives(days, archives=None):
    import data as D
    paths = archives or D.default_archives()
    out = [p for p in paths if D.archive_day(p) in set(days)]
    if len(out) != len(set(days)):
        raise FileNotFoundError(f"archives for {days} not all found among {len(paths)}")
    return out


def majority_counts(persps):
    """Per head, per slot label counts over loss-bearing training steps, with train.py's masks
    (units present; quantity only for quantity tokens; digits only under QTY_EXACT; the last,
    STOP-only order slot excluded)."""
    import codec as C
    uhas = np.asarray(C.UNIT_HAS_QTY, bool)
    ohas = np.asarray(C.ORDER_HAS_QTY, bool)
    U, O, S = C.MAX_UNITS, C.N_ORDER_SLOTS, C.DIGIT_SLOTS
    counts = {"unit": np.zeros((U, len(C.UNIT_VOCAB)), np.int64),
              "unit_qty": np.zeros((U, C.QTY_CLASSES), np.int64),
              "unit_digit": np.zeros((S, len(C.DIGIT_VOCAB)), np.int64),
              "order": np.zeros((O, len(C.ORDER_VOCAB)), np.int64),
              "order_qty": np.zeros((O, C.QTY_CLASSES), np.int64),
              "order_digit": np.zeros((S, len(C.DIGIT_VOCAB)), np.int64)}
    steps = 0
    for p in persps:
        for t in range(len(p)):
            ex = p.example(t)
            if not ex["loss_mask"]:
                continue
            steps += 1
            lab = ex["labels"]
            present = ex["unit_pos"][:, 0] >= 0
            present[0] = True
            for side, has, n_slots in (("unit", uhas, U), ("order", ohas, O - 1)):
                tok = lab[f"{side}_tok"][:n_slots]
                qty = lab[f"{side}_qty"][:n_slots]
                dig = lab[f"{side}_digits"][:n_slots]
                ok = tok >= 0
                if side == "unit":
                    ok &= present[:n_slots]
                for k in np.nonzero(ok)[0]:
                    counts[side][k, tok[k]] += 1
                    if has[tok[k]] and qty[k] >= 0:
                        counts[f"{side}_qty"][k, qty[k]] += 1
                        if qty[k] == C.QTY_EXACT:
                            for j, d in enumerate(dig[k]):
                                if d >= 0:
                                    counts[f"{side}_digit"][j, d] += 1
    return {"steps": steps, "counts": {k: v.tolist() for k, v in counts.items()}}


def smoke_data(days, *, episodes=40, heldout_episodes=4, workers=8, strength_top=0.6, seed=0, out, log=print):
    import data as D
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    archives = day_archives(days)
    rep = {"schema": "primitive-bc-e2e-smoke-data-v1", "days": list(days), "archives": archives,
           "strength_top": strength_top, "seats": "both", "shuffle_seed": seed}
    for name, split, n in (("train", "train", episodes), ("heldout", "heldout", heldout_episodes)):
        stats = {}
        t0 = time.time()
        persps = list(D.iter_perspectives(archives, split=split, strength_top=strength_top, seats="both",
                                          workers=workers, shuffle_seed=seed, max_episodes=n, stats=stats))
        persps.sort(key=lambda p: (int(p.meta["episode_id"]), int(p.meta["seat"])))   # completion order varies
        with open(out / f"{name}.pkl", "wb") as f:
            pickle.dump(persps, f, protocol=pickle.HIGHEST_PROTOCOL)
        steps = sum(len(p) for p in persps)
        rep[name] = {"split": split, "episode_tasks": n, "episodes": len({p.meta["episode_id"] for p in persps}),
                     "perspectives": len(persps), "steps": steps, "seconds": round(time.time() - t0, 1),
                     "steps_per_s": round(steps / max(1e-9, time.time() - t0), 1),
                     "literal_episodes": stats.get("literal_episodes"), "errors": stats.get("errors"),
                     "ids": [[int(p.meta["episode_id"]), int(p.meta["seat"]), p.meta.get("team")] for p in persps]}
        log(json.dumps({k: v for k, v in rep[name].items() if k != "ids"}))
        if name == "train":
            rep["majority"] = majority_counts(persps)
    rep["loadavg_end"] = list(os.getloadavg())
    (out / "SMOKE-DATA.json").write_text(json.dumps(rep, indent=1) + "\n")
    return rep


def load_cache(cache, name):
    with open(Path(cache) / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------- smoke fit
def smoke_fit(cache, out, *, think="field", updates=300, threads=4, seed=0, batch=16, pool=32, chunk=24, lr=1e-3,
              warmup=None, eval_free_steps=24, size=None, log=print):
    torch = _torch()
    import train as T
    torch.set_num_threads(threads)
    train_p = load_cache(cache, "train")
    held_p = load_cache(cache, "heldout")
    held = [p.examples() for p in held_p]
    over = {**SMALL, **(size or {}), "think": think, "n_teams": 1024}
    model_cfg = T.model_config(**over)
    cfg = T.TrainConfig(chunk=chunk, pool=pool, batch=batch, updates=updates, lr=lr,
                        warmup=warmup if warmup is not None else max(1, updates // 10), seed=seed,
                        ckpt_every=updates, keep_ckpts=1, eval_every=updates, eval_perspectives=len(held),
                        eval_steps=0, eval_free_steps=eval_free_steps, log_every=max(1, updates // 20),
                        collapse_check_update=500 if updates >= 500 else 0, workers=0, archives=[])
    t0 = time.time()
    net, metrics = T.fit(cfg, model_cfg, out, device="cpu", source=CachedSource(train_p, seed), held=held, log=log)
    rep = {"schema": "primitive-bc-e2e-smoke-fit-v1", "think": think, "updates": updates, "threads": threads,
           "seconds": round(time.time() - t0, 1), "parameters": T.parameter_count(net),
           "model_config": {k: v for k, v in dataclasses.asdict(net.config).items()
                            if k not in ("unit_has_qty", "order_has_qty")},
           "train_perspectives": len(train_p), "heldout_perspectives": len(held),
           "heldout_steps": sum(len(h) for h in held), "eval": metrics.get("eval"),
           "aborted": metrics.get("aborted"), **cpu_info()}
    Path(out, "SMOKE-FIT.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return rep


# ----------------------------------------------------------------------------- baseline + report
def _mode_logits(counts):
    torch = _torch()
    c = np.asarray(counts, np.float64)
    logits = np.full(c.shape, -30.0, np.float32)
    logits[np.arange(c.shape[0]), c.argmax(1)] = 0.0
    return torch.as_tensor(logits)


def baseline_eval(held_p, majority, *, net=None, chunk=64):
    """Most-frequent (per slot) and copy-previous-unit baselines on held-out steps, scored by
    train.step_loss with the same masks as the model's metrics."""
    torch = _torch()
    import model as M
    import train as T
    net = net or M.PolicyNet(M.ModelConfig(**M.SIZES["tiny"]))
    cnt = majority["counts"]
    U, O, D = net.U, net.O, net.D
    unit = _mode_logits(cnt["unit"])                            # [U, Vu]
    unit_qty = _mode_logits(cnt["unit_qty"])                    # [U, Q]
    order = _mode_logits(cnt["order"])                          # [O, Vo]
    order_qty = _mode_logits(cnt["order_qty"])
    udig = _mode_logits(cnt["unit_digit"])                      # [D, Nd]
    odig = _mode_logits(cnt["order_digit"])
    meter = T.Meter()
    copy = T.Meter()
    pass_index = T.pass_index_of(net)
    for p in held_p:
        exs = p.examples()
        for lo in range(0, len(exs), chunk):
            batch = T.collate_steps(net, exs[lo:lo + chunk], "cpu")
            B = batch["unit_pos"].shape[0]
            present = batch["unit_pos"][..., 0] >= 0
            present[:, 0] = True
            out = {"unit_logits": unit.expand(B, U, -1), "unit_qty_logits": unit_qty.expand(B, U, -1),
                   "unit_digit_logits": udig.expand(B, U, D, -1),
                   "order_logits": order.expand(B, O, -1), "order_qty_logits": order_qty.expand(B, O, -1),
                   "order_digit_logits": odig.expand(B, O, D, -1), "unit_present": present}
            T.step_loss(net, out, batch, meter=meter, pass_index=pass_index)
            prev = batch["prev_unit_tok"].long()
            guess = torch.where(prev >= 0, prev, pass_index)
            ulog = torch.full((B, U, unit.shape[-1]), -30.0)
            ulog.scatter_(2, guess[..., None], 0.0)
            T.step_loss(net, {"unit_logits": ulog, "unit_present": present}, batch, meter=copy, pass_index=pass_index)
    rep = meter.report()
    crep = copy.report()
    return {"majority": {k: v for k, v in rep.items() if k.startswith(("acc_", "n_"))},
            "copy_prev_unit": {k: crep[k] for k in ("acc_unit_tok", "n_unit_tok", "acc_unit_tok_nonpass",
                                                    "acc_unit_tok_change") if k in crep}}


REPORT_KEYS = ("acc_unit_tok", "acc_unit_qty", "acc_unit_digits", "acc_order_tok", "acc_order_qty",
               "acc_order_digits", "acc_unit_exact", "acc_order_exact", "acc_step_exact", "acc_unit_tok_nonpass",
               "acc_unit_tok_change", "acc_free_step_exact", "acc_free_valid", "loss_per_step",
               "field_const_position_frac", "fsq_entropy_bits_mean", "n_unit_tok", "n_order_tok", "n_step_exact",
               "n_unit_tok_nonpass", "n_unit_tok_change", "n_free_step_exact", "n_unit_digits", "n_order_digits")


def smoke_report(cache, runs, out):
    data_rep = json.loads((Path(cache) / "SMOKE-DATA.json").read_text())
    held_p = load_cache(cache, "heldout")
    base = baseline_eval(held_p, data_rep["majority"])
    rows = {}
    for run in runs:
        fit = json.loads((Path(run) / "SMOKE-FIT.json").read_text())
        ev = fit.get("eval") or {}
        rows[fit["think"]] = {"updates": fit["updates"], "seconds": fit["seconds"], "parameters": fit["parameters"],
                              "aborted": fit.get("aborted"), **{k: ev.get(k) for k in REPORT_KEYS}}
    rep = {"schema": "primitive-bc-e2e-smoke-report-v1", "data": {k: data_rep[k] for k in ("days", "strength_top")},
           "train": {k: data_rep["train"][k] for k in ("episodes", "perspectives", "steps")},
           "heldout": {k: data_rep["heldout"][k] for k in ("episodes", "perspectives", "steps")},
           "runs": rows, "baseline": base,
           "note": "CPU-scale fits of a reduced model; they do not establish GPU-scale results."}
    Path(out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return rep


# ----------------------------------------------------------------------------- CLI
def _kv(items):
    out = {}
    for item in items or []:
        k, _, v = item.partition("=")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("make-weights")
    w.add_argument("--out", required=True)
    w.add_argument("--storage", choices=("fp32", "fp16"), default="fp32")
    w.add_argument("--head-std", type=float, default=0.5)
    w.add_argument("--set", nargs="*", default=None, help="ModelConfig overrides key=value")
    lt = sub.add_parser("latency")
    lt.add_argument("--weights", required=True)
    lt.add_argument("--variant", choices=("fp32", "int8"), default="fp32")
    lt.add_argument("--archives", nargs="*", default=None)
    lt.add_argument("--episodes", type=int, default=2)
    lt.add_argument("--seats", default="0,1")
    lt.add_argument("--units", type=int, default=16)
    lt.add_argument("--orders", type=int, default=11)
    lt.add_argument("--exact", action="store_true")
    lt.add_argument("--steps", type=int, default=STEPS)
    lt.add_argument("--out", required=True)
    c = sub.add_parser("cold")
    c.add_argument("--weights", required=True)
    c.add_argument("--variant", choices=("fp32", "int8"), default="fp32")
    c.add_argument("--runs", type=int, default=3)
    c.add_argument("--out", required=True)
    sd = sub.add_parser("smoke-data")
    sd.add_argument("--days", nargs="+", required=True)
    sd.add_argument("--episodes", type=int, default=40)
    sd.add_argument("--heldout-episodes", type=int, default=4)
    sd.add_argument("--workers", type=int, default=8)
    sd.add_argument("--strength-top", type=float, default=0.6)
    sd.add_argument("--seed", type=int, default=0)
    sd.add_argument("--out", required=True)
    sf = sub.add_parser("smoke-fit")
    sf.add_argument("--cache", required=True)
    sf.add_argument("--think", default="field")
    sf.add_argument("--updates", type=int, default=300)
    sf.add_argument("--threads", type=int, default=4)
    sf.add_argument("--seed", type=int, default=0)
    sf.add_argument("--batch", type=int, default=16)
    sf.add_argument("--pool", type=int, default=32)
    sf.add_argument("--lr", type=float, default=1e-3)
    sf.add_argument("--set", nargs="*", default=None, help="ModelConfig overrides key=value (on top of SMALL)")
    sf.add_argument("--out", required=True)
    sr = sub.add_parser("smoke-report")
    sr.add_argument("--cache", required=True)
    sr.add_argument("--runs", nargs="+", required=True)
    sr.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "make-weights":
        rep = make_weights(a.out, storage=a.storage, head_std=a.head_std, overrides=_kv(a.set))
    elif a.cmd == "latency":
        rep = latency(a.weights, variant=a.variant, archives=a.archives, episodes=a.episodes,
                      seats=tuple(int(s) for s in a.seats.split(",")), units=a.units, orders=a.orders,
                      exact=a.exact, steps=a.steps)
    elif a.cmd == "cold":
        rep = cold(a.weights, variant=a.variant, runs=a.runs)
    elif a.cmd == "smoke-data":
        rep = smoke_data(a.days, episodes=a.episodes, heldout_episodes=a.heldout_episodes, workers=a.workers,
                         strength_top=a.strength_top, seed=a.seed, out=a.out)
    elif a.cmd == "smoke-fit":
        rep = smoke_fit(a.cache, a.out, think=a.think, updates=a.updates, threads=a.threads, seed=a.seed,
                        batch=a.batch, pool=a.pool, lr=a.lr, size=_kv(a.set))
    else:
        rep = smoke_report(a.cache, a.runs, a.out)
    if getattr(a, "out", None) and a.cmd not in ("make-weights", "smoke-data", "smoke-fit"):
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps(rep if a.cmd != "smoke-report" else {k: rep[k] for k in ("runs", "baseline")}, indent=1,
                     default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
