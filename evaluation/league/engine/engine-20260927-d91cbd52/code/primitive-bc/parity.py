"""Train/play parity of primitive-BC v3 over complete public episodes.

For each chosen episode and seat, two paths run in lockstep, step by step from step 0:

  training path  data.iter_perspectives(...) -> Perspective.chunks(W) -> per step
                 train.collate_steps -> train.dawn_reset -> PolicyNet.forward(batch, state)
                 (exactly train.run_chunk's per-step body; the carried state crosses chunk
                 boundaries; each chunk's final state is also checked against run_chunk itself)
  agent path     agent.Agent (teacher-forced mode) fed the raw replay observation of that seat
                 (as the Kaggle runtime delivers it; seat 1's replay observation has no 'step')
                 and the canonical label action data.py recorded for that step
                 (data.canonical_labels(replay).history); the model still runs (PolicyNet.act).

Compared at every step: the per-step example arrays (pack rows, knownness, token mask, unit
positions, tile rows, previous unit tokens, feature_tokens values/known/type_id/tile, meta
step/hour/team_id/strength_decile), every tensor of the collated batch the model receives, the
carried state entering the model, and the carried state it returns.  Result: max abs difference
per tensor over all steps, the number of steps with a nonzero difference, and the first one.

``--teacher literal`` feeds codec.canonical(submitted action) without the engine audit (no tail
strip, no SELL relabel) instead (diagnostic: data.py records the audited canonical action, so the
history features and previous unit tokens are expected to diverge there).

CLI:  parity.py run --episodes 2 --out PARITY.json [--config default|tiny] [--think field]
"""
from __future__ import annotations

import argparse
import functools
import json
import os
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

import torch  # noqa: E402

import agent as A  # noqa: E402
import data as D  # noqa: E402
import model as M  # noqa: E402
import train as T  # noqa: E402
from farm_neat.training import bc  # noqa: E402

EXAMPLE_KEYS = ("rows", "knownness", "token_mask", "unit_pos", "tile_row", "prev_unit_tok")
FEAT_KEYS = ("values", "known", "type_id", "tile")
META_KEYS = ("step", "hour", "team_id", "strength_decile")
SKIP_BATCH = ("labels", "loss_mask", "meta")


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _flat(tree, prefix=""):
    """{dotted key: tensor/array} of a nested batch dict (non-array leaves dropped)."""
    out = {}
    for k, v in tree.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif torch.is_tensor(v) or isinstance(v, np.ndarray):
            out[key] = v
    return out


class Diffs:
    """Per tensor: max abs difference, steps compared, steps with a difference, first one."""

    def __init__(self):
        self.rows = {}

    def add(self, key, a, b, where):
        r = self.rows.setdefault(key, {"max_abs_diff": 0.0, "steps": 0, "steps_differ": 0, "first": None,
                                       "dtype_mismatch": 0, "shape_mismatch": 0})
        r["steps"] += 1
        a, b = _np(a), _np(b)
        if a.shape != b.shape:
            r["shape_mismatch"] += 1
            r["steps_differ"] += 1
            r["max_abs_diff"] = float("inf")
            r["first"] = r["first"] or {**where, "shapes": [list(a.shape), list(b.shape)]}
            return
        if a.dtype != b.dtype:
            r["dtype_mismatch"] += 1
        d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
        if d > 0:
            r["steps_differ"] += 1
            r["first"] = r["first"] or {**where, "diff": d}
        r["max_abs_diff"] = max(r["max_abs_diff"], d)

    def missing(self, key, where, side):
        r = self.rows.setdefault(key, {"max_abs_diff": float("inf"), "steps": 0, "steps_differ": 0, "first": None,
                                       "dtype_mismatch": 0, "shape_mismatch": 0})
        r["steps"] += 1
        r["steps_differ"] += 1
        r["max_abs_diff"] = float("inf")
        r["first"] = r["first"] or {**where, "only_in": side}

    def report(self):
        return {k: self.rows[k] for k in sorted(self.rows)}


class ActCapture:
    """Wraps net.act (instance attribute): records the batch, incoming and returned state."""

    def __init__(self, net):
        self.net, self.last = net, None
        orig = type(net).act.__get__(net)

        @functools.wraps(orig)
        def act(inputs, state=None, **kw):
            labels, new = orig(inputs, state, **kw)
            self.last = (inputs, state, new)
            return labels, new
        net.act = act

    def close(self):
        self.net.__dict__.pop("act", None)


def compare_example(diffs, got, ref, where):
    for k in EXAMPLE_KEYS:
        diffs.add(f"example.{k}", got[k], ref[k], where)
    for k in FEAT_KEYS:
        diffs.add(f"example.feat.{k}", got["feat"][k], ref["feat"][k], where)
    for k in META_KEYS:
        diffs.add(f"example.meta.{k}", np.asarray(int(got["meta"][k])), np.asarray(int(ref["meta"][k])), where)


def compare_batch(diffs, got, ref, where):
    g = _flat({k: v for k, v in got.items() if k not in SKIP_BATCH})
    r = _flat({k: v for k, v in ref.items() if k not in SKIP_BATCH})
    for k in sorted(set(g) | set(r)):
        if k not in g:
            diffs.missing(f"batch.{k}", where, "training")
        elif k not in r:
            diffs.missing(f"batch.{k}", where, "agent")
        else:
            diffs.add(f"batch.{k}", g[k], r[k], where)


def build_net(config="default", think="field", carry_scope="episode", seed=0):
    torch.manual_seed(seed)
    over = dict(think=think, carry_scope=carry_scope)
    cfg = M.ModelConfig(**{**(M.SIZES["tiny"] if config == "tiny" else {}), **over})
    if config == "tiny":
        cfg.n_teams = 1024                     # real team ids (STRENGTH.json vocab) are not clamped
    return M.PolicyNet(cfg).eval()


def run(*, archives=None, episodes=2, split="all", workers=0, W=24, config="default", think="field",
        carry_scope="episode", teacher="canonical", steps=None, check_run_chunk=1, seed=0, log=print):
    """-> report dict (see module doc)."""
    torch.set_num_threads(1)
    archives = list(archives or [D.default_archives()[-1]])
    net = build_net(config, think, carry_scope, seed)
    capture = ActCapture(net)
    diffs = Diffs()
    t0 = time.time()
    persps = D.iter_perspectives(archives, split=split, strength_top=None, seats="both", workers=workers,
                                 shuffle_seed=None, max_episodes=episodes, wins_only=False,
                                 exclude_static_tapes=False)       # mechanics check: every seat
    replays, labels_cache = {}, {}
    run_chunk_checks = {"chunks": 0, "max_abs_diff": 0.0}
    done = []
    agent_summaries = []
    try:
        for n_p, p in enumerate(persps):
            meta = p.meta
            eid, seat = int(meta["episode_id"]), int(meta["seat"])
            if eid not in replays:
                member = next(m for m in D.list_members(archives, split="all") if m[3] == eid)
                replays[eid] = D.read_replay(member[0], member[1])
                labels_cache[eid] = D.canonical_labels(replays[eid])
            replay, labels = replays[eid], labels_cache[eid]
            actor = A.Agent("preloaded", policy=net, log=False, team_id=int(meta["team_id"]),
                            strength_decile=int(meta["strength_decile"]))
            state = None
            t = 0
            for chunk in p.chunks(W):
                chunk_start_state = state
                processed = 0
                for ex in chunk:
                    if steps is not None and t >= steps:
                        break
                    where = {"episode": eid, "seat": seat, "step": t}
                    if int(ex["meta"]["step"]) != t:
                        raise AssertionError(f"perspective step order: {ex['meta']['step']} != {t}")
                    # training path (train.run_chunk's per-step body)
                    with torch.no_grad():
                        batch_tr = T.collate_steps(net, [ex], "cpu")
                        st_tr = T.dawn_reset(net, state, batch_tr, "cpu")
                        _, new_tr = net(batch_tr, st_tr)
                    # agent path
                    obs = replay["steps"][t][seat]["observation"]
                    if teacher == "canonical":
                        act = labels.history[seat][t]
                    else:                    # literal: codec.canonical of the submitted action (no audit)
                        raw = replay["steps"][t + 1][seat].get("action")
                        view = D.canonical_observation(bc.public_observation(replay, t, seat))
                        act = D.codec.canonical(raw, view)
                    capture.last = None
                    actor(obs, replay["configuration"], teacher_action=act)
                    if actor.last_fallback or capture.last is None:
                        raise AssertionError(f"agent fallback at {where}: {actor.last_error}")
                    batch_ag, st_ag, new_ag = capture.last
                    compare_example(diffs, actor.last["inputs"], ex, where)
                    compare_batch(diffs, batch_ag, batch_tr, where)
                    st_tr_in = net.initial_state(1) if st_tr is None else st_tr
                    diffs.add("state.in", st_ag, st_tr_in, where)
                    diffs.add("state.out", new_ag, new_tr, where)
                    state = new_tr
                    t += 1
                    processed += 1
                if n_p < check_run_chunk and processed == len(chunk):
                    with torch.no_grad():
                        _, _, st_rc, _ = T.run_chunk(net, [chunk], chunk_start_state, device="cpu", bf16=False)
                    d = float((st_rc - state).abs().max())
                    run_chunk_checks["chunks"] += 1
                    run_chunk_checks["max_abs_diff"] = max(run_chunk_checks["max_abs_diff"], d)
                if steps is not None and t >= steps:
                    break
            agent_summaries.append({"episode": eid, "seat": seat, **actor.summary()})
            done.append({"episode": eid, "seat": seat, "steps": t, "team_id": int(meta["team_id"]),
                         "strength_decile": int(meta["strength_decile"]),
                         "literal_labels": bool(meta["literal_labels"])})
            log(json.dumps({"perspective": done[-1], "seconds": round(time.time() - t0, 1)}))
    finally:
        capture.close()
        close = getattr(persps, "close", None)
        if close is not None:
            close()
    rep = diffs.report()
    worst = {k: v["max_abs_diff"] for k, v in rep.items()}
    inputs_exact = all(v == 0 for k, v in worst.items() if not k.startswith("state."))
    state_max = max((v for k, v in worst.items() if k.startswith("state.")), default=0.0)
    return {"schema": "primitive-bc-parity-v3", "archives": archives, "split": split, "workers": workers,
            "chunk": W, "config": config, "think": think, "carry_scope": carry_scope, "teacher": teacher,
            "parameters": net.parameter_count(), "perspectives": done,
            "steps_compared": sum(d["steps"] for d in done), "inputs_bit_equal": inputs_exact,
            "state_max_abs_diff": state_max, "pass": inputs_exact and state_max <= 1e-5,
            "run_chunk_final_state": run_chunk_checks, "agent": agent_summaries, "diffs": rep,
            "seconds": round(time.time() - t0, 1), "load_average": list(os.getloadavg()),
            "torch_threads": torch.get_num_threads()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--archives", nargs="*", default=None, help="daily zips (default: the latest local day)")
    r.add_argument("--episodes", type=int, default=2)
    r.add_argument("--split", default="all")
    r.add_argument("--workers", type=int, default=0)
    r.add_argument("--chunk", type=int, default=24)
    r.add_argument("--config", choices=("default", "tiny"), default="default")
    r.add_argument("--think", choices=M.THINK_MODES, default="field")
    r.add_argument("--carry-scope", choices=M.CARRY_SCOPES, default="episode")
    r.add_argument("--teacher", choices=("canonical", "literal"), default="canonical")
    r.add_argument("--steps", type=int, default=None, help="per perspective (default: the whole episode)")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = run(archives=a.archives, episodes=a.episodes, split=a.split, workers=a.workers, W=a.chunk,
              config=a.config, think=a.think, carry_scope=a.carry_scope, teacher=a.teacher, steps=a.steps)
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    summary = {k: rep[k] for k in ("config", "think", "carry_scope", "teacher", "steps_compared", "inputs_bit_equal",
                                   "state_max_abs_diff", "pass", "run_chunk_final_state", "seconds")}
    summary["nonzero"] = {k: v["max_abs_diff"] for k, v in rep["diffs"].items() if v["max_abs_diff"]}
    print(json.dumps(summary, indent=1))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
