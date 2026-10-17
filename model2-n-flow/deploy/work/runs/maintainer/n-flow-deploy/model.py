"""N flow policy for the primitive-BC Kaggle harness (agent.py imports ``model.PolicyNet``).

The network is nf_model.NFlowNet (fresh-init lineage, schema nflow-policy-v1); this module adds the single-observation
play path: actor features through data.features (the same function the step cache was built with) and the
nf_format reference encoder, the recurrent memory carried in the per-seat state, and greedy sequential decoding with
the exact same-step prefix executor (prefix.PrefixExecutor) and a per-layer K/V cache.

``act(..., force=labels)`` replays given decisions instead of choosing (serve/train parity tests only).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import nf_format as NF
import nf_model as M
from nf_model import NFlowConfig, U, O

HERE = Path(__file__).resolve().parent
ModelConfig = NFlowConfig
SCHEMA = M.SCHEMA
_COLS = None


def columns():
    global _COLS
    if _COLS is None:
        _COLS = NF.load_columns(HERE / "COLUMNS.json")
    return _COLS


def default_config():
    import codec
    return M.resolve_config(NFlowConfig(), codec, columns())


@dataclass
class ThinkState:
    """Per (episode, seat) play state. hist: own last two submitted decisions (step, unit_tok, unit_qty, order_tok,
    order_qty) as codec labels (the harness rewrites the newest entry when a play-time rule edits the action).
    mem: the N flow recurrent memory after this step (fast, slow, worker, day accumulators)."""
    field: Optional[object] = None
    last_step: int = -1
    hist: tuple = ()
    mem: Optional[dict] = None


def v6_history_arrays(hist, step, max_units, n_orders):
    """prev_unit [1, 2, U, 2], prev_order [1, 2, O, 2] from recorded entries (data.add_v6 rule: lag 1 then 2; the
    farmer across days, a hand only within its day). Copied from the B deploy model.py."""
    pu = torch.full((1, 2, max_units, 2), -1, dtype=torch.long)
    po = torch.full((1, 2, n_orders, 2), -1, dtype=torch.long)
    by_step = {int(e[0]): e for e in hist}
    for li, lag in enumerate((1, 2)):
        e = by_step.get(step - lag)
        if e is None:
            continue
        s, ut, uq, ot, oq = e
        for k in range(min(max_units, len(ut))):
            if k == 0 or s // 24 == step // 24:
                pu[0, li, k, 0], pu[0, li, k, 1] = int(ut[k]), int(uq[k])
        n = min(n_orders, len(ot))
        po[0, li, :n, 0] = torch.as_tensor([int(x) for x in ot[:n]])
        po[0, li, :n, 1] = torch.as_tensor([int(x) for x in oq[:n]])
    return pu, po


def step_inputs(feats, prev_unit, prev_order, step):
    """data.features output for ONE observation -> batch-1 model inputs (float16 rounding exactly as the shards)."""
    one = {k: np.asarray(feats[k])[None] for k in ("rows", "knownness", "token_mask", "extra", "v6_token", "x8_token")}
    tok = NF.compact_tokens(one, columns())
    x = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in tok.items()}
    for k, dt in (("unit_pos", np.int8), ("unit_extra", np.float16), ("x8_unit", np.float16), ("reach_eta", np.int16),
                  ("v6_job_deadline", np.int8), ("v6_job_deadline_present", np.bool_)):
        a = np.asarray(feats[k])
        if k == "unit_pos":
            a = a[:U]
        x[k] = torch.from_numpy(np.ascontiguousarray(a.astype(dt)))[None]
    x["prev_unit"] = prev_unit.to(torch.int8)
    x["prev_order"] = prev_order.to(torch.int8)
    x["step"] = torch.tensor([int(step)], dtype=torch.int16)
    return x


class PolicyNet(M.NFlowNet):
    def __init__(self, config: Optional[NFlowConfig] = None, codec=None):
        # review ASTRA R2 #8 / CHECK-NFLOW-DEPLOY #3: every config (default or from a checkpoint) must match the shipped
        # feature / codec / columns contract
        import codec as _codec
        c = config if config is not None else default_config()
        expected = M.resolve_config(c, _codec, columns())
        if c.schema != SCHEMA:
            raise ValueError("N flow config schema mismatch")
        if c.columns_sha256 != expected.columns_sha256 or c.type_widths != expected.type_widths:
            raise ValueError("N flow columns contract differs from COLUMNS.json")
        for name in ("n_unit_tokens", "n_order_tokens", "n_qty", "max_units", "max_orders",
                     "stop_index", "unit_has_qty", "order_has_qty"):
            if getattr(c, name) != getattr(expected, name):
                raise ValueError(f"N flow codec contract mismatch: {name}")
        for name, value in (("x8_token_width", 354), ("x8_unit_width", 41), ("lattice_channels", 0),
                            ("v6_prev", True), ("feature_schema", "legacy")):
            if getattr(c, name) != value:
                raise ValueError(f"N flow feature contract mismatch: {name}")
        super().__init__(c)

    @property
    def codec(self):
        import codec
        return codec

    def initial_state(self):
        return ThinkState()

    def save(self, path, *, dtype=torch.float32, extra=None):
        blob = {"schema": SCHEMA, "config": self.config.to_dict(),
                "model": {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in self.state_dict().items()},
                "extra": extra or {}}
        torch.save(blob, str(path))

    @classmethod
    def load(cls, path, map_location="cpu", codec=None):
        """Training checkpoint (n-flow train.py: model or ema weights) or PolicyNet.save file."""
        blob = torch.load(str(path), map_location=map_location, weights_only=False)
        if blob.get("schema") != SCHEMA:
            raise ValueError(f"not an N flow checkpoint: {blob.get('schema')!r}")
        net = cls(NFlowConfig.from_dict(blob["config"]))
        state = blob["model"]
        net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in state.items()})
        return net.eval()

    from_checkpoint = load

    # ------------------------------------------------------------------ play
    def _features(self, clean, configuration, v8_history, v8_long):
        import data as _data
        if v8_history is None or v8_long is None:
            raise ValueError("N flow needs the v8 history and GapsState (harness supplies them)")
        return _data.features(clean, configuration or {}, feature_schema="legacy", v8_history=v8_history, v8_long=v8_long)

    @torch.no_grad()
    def act(self, observation, generator=None, temperature=0.0, state=None, configuration=None,
            v8_history=None, v8_long=None, force=None, trace=None):
        """One decision -> (action dict, ThinkState). Greedy (argmax) decoding; temperature is ignored."""
        from farm_neat.training.bc import actor_observation
        clean = actor_observation(dict(observation))
        if "step" not in clean:
            clean["step"] = int(clean["day"]) * 24 + int(clean["hour"])
        step = int(clean["step"])
        feats = self._features(clean, configuration, v8_history, v8_long)
        contiguous = isinstance(state, ThinkState) and 0 < step and step == state.last_step + 1
        fresh = not (contiguous and state.mem is not None)
        # CHECK-NFLOW-DEPLOY #2: a failed decoder can invalidate learned memory without erasing the submitted PASS
        # from the two-step wire-action history
        hist = tuple(state.hist) if contiguous else ()
        pu, po = v6_history_arrays(hist, step, U, O)
        x = step_inputs(feats, pu, po, step)
        mem = self.initial_memory(1, "cpu") if fresh else state.mem
        labels, mem = self.decide(x, mem, clean, configuration, force=force, trace=trace)
        lab = {k: [int(v) for v in labels[k]] for k in ("unit_tok", "unit_qty")}
        n = next((i + 1 for i, t in enumerate(labels["order_tok"]) if t == self.config.stop_index), len(labels["order_tok"]))
        lab["order_tok"], lab["order_qty"] = [int(t) for t in labels["order_tok"][:n]], [int(q) for q in labels["order_qty"][:n]]
        n_units = len([clean["farms"][clean["player"]]["farmer"], *clean["farms"][clean["player"]]["hands"]])
        lab["unit_tok"], lab["unit_qty"] = lab["unit_tok"][:min(n_units, U)], lab["unit_qty"][:min(n_units, U)]
        action = self.codec.decode_step(lab, clean)
        try:
            import data as _data
            saved = _data.encode_step_lenient(action, clean)
        except ImportError:
            saved = self.codec.encode_step(action, clean)
        hist = (hist + ((step, saved["unit_tok"], saved["unit_qty"], saved["order_tok"], saved["order_qty"]),))[-2:]
        return action, ThinkState(None, step, hist, mem)

    def decide(self, x, mem, clean, configuration, force=None, trace=None):
        """Encode one step, advance memory, greedy-decode 25 unit + <= 11 order positions."""
        import prefix as _prefix
        c = self.config
        pre = self.encode_pre(x)
        s = self.summary(pre)
        act, act_w = self.prev_action_emb(x)
        present = x["unit_pos"][..., 0] >= 0
        present[:, 0] = True
        mem = self.memory_step(mem, s, act, pre[:, M.OWN_CARRIER:M.OWN_CARRIER + U], act_w, x["step"].long(), present)
        h = self.encode_post(pre, mem["fast"], mem["slow"], mem["worker"])
        ctx = self.dec_context(h, x, mem["worker"], act_w)
        if trace is not None:
            trace["aux"] = self.aux_heads(h, mem["fast"], mem["slow"])
        pe = _prefix.PrefixExecutor(clean, configuration=configuration, cash_enabled=True)
        n_units = len(pe.positions)
        caches = [dict() for _ in self.dec]
        last = self.tok_emb_u.weight[c.n_unit_tokens + 1][None]
        unit_tok, unit_qty = [-1] * U, [-1] * U
        logs = []
        for k in range(U):
            up = pe.unit_features(k) if k < n_units else np.zeros(27, np.float32)
            up = torch.from_numpy(up.astype(np.float16).astype(np.float32))[None]
            q = ctx["unit_q"][:, k] + self.unit_in(torch.cat((up, ctx["unit_f"][:, k].float()), -1)) + last
            hq = self.run_decoder(q[:, None], ctx, caches, k)[:, 0]
            if k < n_units:
                logits = self.unit_head(hq)
                tok = int(logits.argmax(-1)) if force is None else int(force["unit_tok"][k])
                qty = -1
                if c.unit_has_qty[tok]:
                    ql = self.unit_qty_head(hq + self.qty_cond_u(torch.tensor([tok])))
                    qty = int(ql.argmax(-1)) if force is None else int(force["unit_qty"][k])
                    logs.append(("unit_qty", k, ql))
                logs.append(("unit_tok", k, logits))
                pe.apply_unit(k, self.codec.decode_unit(tok, qty, clean, k))
            else:
                tok, qty = -1, -1
            unit_tok[k], unit_qty[k] = tok, qty
            last = self.decision_emb(torch.tensor([tok]), torch.tensor([qty]), True)
        for k in range(U, n_units):          # workers beyond the 25 decoder slots still act: PASS (as B)
            pe.apply_unit(k, ["PASS"])
        order_tok, order_qty = [], []
        for j in range(O):
            op = torch.from_numpy(pe.order_features(j).astype(np.float16).astype(np.float32))[None]
            cash = torch.from_numpy(pe.cash_features(j, c.max_orders).astype(np.float16).astype(np.float32))[None]
            q = self.qpos.weight[U + j][None] + self.order_in(torch.cat((op, cash), -1)) + last
            hq = self.run_decoder(q[:, None], ctx, caches, U + j)[:, 0]
            logits = self.order_head(hq)
            if j == O - 1:
                logits = logits.masked_fill(~self.stop_only, float("-inf"))
            tok = int(logits.argmax(-1)) if force is None else int(force["order_tok"][j])
            qty = -1
            if c.order_has_qty[tok]:
                ql = self.order_qty_head(hq + self.qty_cond_o(torch.tensor([tok])))
                qty = int(ql.argmax(-1)) if force is None else int(force["order_qty"][j])
                logs.append(("order_qty", j, ql))
            logs.append(("order_tok", j, logits))
            order_tok.append(tok)
            order_qty.append(qty)
            if tok == c.stop_index:
                break
            pe.apply_order(self.codec.decode_order(tok, qty, clean))
            last = self.decision_emb(torch.tensor([tok]), torch.tensor([qty]), False)
        if trace is not None:
            trace["logits"] = logs
        return {"unit_tok": unit_tok, "unit_qty": unit_qty, "order_tok": order_tok, "order_qty": order_qty}, mem


def load_policy(path):
    return PolicyNet.load(path)


def random_policy(seed=0):
    torch.manual_seed(int(seed))
    return PolicyNet().eval()
