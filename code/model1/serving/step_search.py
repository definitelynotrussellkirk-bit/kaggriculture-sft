"""Sequence-level decoding of one step for PolicyNet (v9-decode).

Token-level greedy (``decode_free`` at temperature 0) picks the argmax token at every decoder position; that is not the
most probable whole step.  This module scores a complete step by the model's joint log-probability at temperature 1:
every present worker token, every order token up to and including STOP, and the quantity of every token that carries
one (all under the same masks as ``decode_free``).

Modes (``normalize``):
  greedy  the unchanged ``PolicyNet.act`` path (default; this module is not called).
  bestk   row 0 = token-level greedy, rows 1..k = complete steps sampled at ``temperature`` from a per-step seed that is
          a hash of the actor observation (play is a deterministic function of the observation and carried state);
          the row with the highest joint log-prob is played (ties -> lowest row, so greedy wins ties).
  beam    beam search over the step's token sequence, ``width`` beams, (token, quantity) expansions scored jointly.
``first_n``: apply at the seat's steps < first_n only (None = every step); other steps use the greedy path.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

MODES = ("greedy", "bestk", "beam")
TIE = 1e-6      # joint log-prob differences below this are batch-numerics noise (rows of one batch differ ~1e-7)


def normalize(settings):
    """None / "greedy" / dict -> validated plain dict (JSON-serialisable, recorded in the archive MANIFEST)."""
    if settings is None or settings == "greedy":
        return {"mode": "greedy"}
    if isinstance(settings, str):
        settings = json.loads(settings)
    if not isinstance(settings, dict):
        raise TypeError("decode settings must be None, 'greedy' or a dict")
    mode = settings.get("mode", "greedy")
    if mode not in MODES:
        raise ValueError(f"decode mode must be one of {MODES}")
    if mode == "greedy":
        return {"mode": "greedy"}
    first_n = settings.get("first_n")
    if first_n is not None and (isinstance(first_n, bool) or not isinstance(first_n, int) or first_n < 1):
        raise ValueError("first_n must be a positive int or None")
    out = {"mode": mode, "first_n": first_n}
    if mode == "bestk":
        k = settings.get("k", 32)
        t = float(settings.get("temperature", 1.0))
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 256:
            raise ValueError("k must be an int in [1, 256]")
        if not t > 0:
            raise ValueError("bestk temperature must be > 0")
        out.update(k=k, temperature=t, salt=int(settings.get("salt", 0)))
    else:
        w = settings.get("width", 4)
        if isinstance(w, bool) or not isinstance(w, int) or not 1 <= w <= 32:
            raise ValueError("width must be an int in [1, 32]")
        out.update(width=w)
    return out


def applies(settings, step):
    if not settings or settings.get("mode", "greedy") == "greedy":
        return False
    n = settings.get("first_n")
    return n is None or int(step) < int(n)


def step_seed(clean, salt=0):
    text = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=repr)
    digest = hashlib.sha256(f"{int(salt)}|{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


# ----------------------------------------------------------------------------- prefix state (clone-able)
class Prefix:
    """PolicyNet._prefix_fn as an object that can be cloned for beam search; same features and applications."""

    def __init__(self, net, clean, configuration):
        import prefix as _prefix
        c = net.config
        self.net, self.clean = net, clean
        self.pe = _prefix.PrefixExecutor(clean, configuration=configuration, cash_enabled=bool(c.prefix_cash_width))
        self.n_units = len(self.pe.positions)
        self.units_done = 0

    def clone(self):
        new = object.__new__(Prefix)
        new.net, new.clean, new.n_units, new.units_done = self.net, self.clean, self.n_units, self.units_done
        pe = self.pe
        memo = {id(pe.init_farm): pe.init_farm, id(pe.init_private): pe.init_private,
                id(pe.init_market): pe.init_market, id(pe.configuration): pe.configuration,
                id(pe.seeds0): pe.seeds0, id(pe.positions): pe.positions}   # read-only after construction
        new.pe = copy.deepcopy(pe, memo)
        return new

    def _f(self, arr):
        import torch
        return torch.from_numpy(arr.astype("float16").astype("float32"))[None]

    def unit(self, k):
        return self._f(self.pe.unit_features(k))

    def apply_unit(self, k, t, q):
        self.pe.apply_unit(k, self.net.codec.decode_unit(t, q, self.clean, k) if t >= 0 else ["PASS"])
        self.units_done = k + 1

    def _flush(self):
        while self.units_done < self.n_units:              # units beyond the decoder's slots still act (PASS)
            self.pe.apply_unit(self.units_done, ["PASS"])
            self.units_done += 1

    def order(self, k):
        self._flush()
        return self._f(self.pe.order_features(k))

    def cash(self, k):
        self._flush()
        return self._f(self.pe.cash_features(k, self.net.config.max_orders))

    def apply_order(self, k, t, q):
        self._flush()
        if t != self.net.config.stop_index:
            self.pe.apply_order(self.net.codec.decode_order(t, q, self.clean))


def _uses_prefix(net):
    c = net.config
    if c.noop_unit_mask:
        raise ValueError("step_search does not implement noop_unit_mask")
    return bool(c.prefix_unit_width or c.prefix_order_width or c.prefix_cash_width)


# ----------------------------------------------------------------------------- shared decoder pieces
class _Dec:
    """Per-observation decoder inputs shared by every row (all rows read the same batch-1 encoding)."""

    def __init__(self, net, ctx, unit_pos):
        import torch
        self.net, self.ctx = net, ctx
        present = unit_pos[..., 0] >= 0
        present[:, 0] = True
        self.present = present
        self.n_units = int(present.sum(1).max())
        self.mem_kv = [layer.memory(ctx.mem) for layer in net.dec]
        self.hist_memory = net._hist_memory(ctx) if ctx.hist is not None else None
        cpu_eval = not net.training and ctx.mem.device.type == "cpu"
        active_pos = unit_pos[:, :self.n_units] if cpu_eval else unit_pos
        self.feats = net._unit_features(ctx, active_pos)
        self.biases = net._reach_bias(ctx, active_pos)
        self.gf = net._global_feature(ctx)
        self.torch = torch

    def _kv(self, B):
        if B == 1:
            return self.mem_kv
        return [(k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)) for k, v in self.mem_kv]

    def unit_step(self, p, last, caches, valid, prefixes):
        net, torch = self.net, self.torch
        B = last.shape[0]
        q = net.query.weight[p] + self.gf + last + self.feats[:, p]
        if prefixes is not None and net.config.prefix_unit_width:
            q = q + net.grow_prefix_unit(torch.cat([pf.unit(p) for pf in prefixes]).to(q.dtype))
        if self.ctx.hist is not None:
            q = q + net._hist_ctx(q[:, None], self.ctx, self.hist_memory)[:, 0]
        h = net._dec_step(q, caches, self._kv(B), self.ctx.mem_mask, valid, self.biases, p)
        return h, net.unit_head(h)

    def order_step(self, k, last, caches, valid, prefixes):
        net, torch = self.net, self.torch
        B = last.shape[0]
        q = net.query.weight[net.U + k] + self.gf + last
        if prefixes is not None and net.config.prefix_order_width:
            q = q + net.grow_prefix_order(torch.cat([pf.order(k) for pf in prefixes]).to(q.dtype))
        if prefixes is not None and net.config.prefix_cash_width:
            q = q + net.grow_prefix_cash(torch.cat([pf.cash(k) for pf in prefixes]).to(q.dtype))
        if self.ctx.hist is not None:
            q = q + net._hist_ctx(q[:, None], self.ctx, self.hist_memory)[:, 0]
        h = net._dec_step(q, caches, self._kv(B), self.ctx.mem_mask, valid)
        logits = net.order_head(h)
        if k == net.O - 1:
            logits = logits.masked_fill(~net.stop_only, float("-inf"))
        return h, logits


def _labels(net, ut, uq, ot, oq):
    """Row labels in PolicyNet.act's format: unit lists of length U (-1 = absent), orders cut after the first STOP."""
    U = net.U
    unit_tok = list(ut) + [-1] * (U - len(ut))
    unit_qty = list(uq) + [-1] * (U - len(uq))
    stop = net.config.stop_index
    n = next((i + 1 for i, t in enumerate(ot) if t == stop), len(ot))
    return {"unit_tok": unit_tok, "unit_qty": unit_qty, "order_tok": list(ot[:n]), "order_qty": list(oq[:n])}


# ----------------------------------------------------------------------------- batched sampling (best-of-K)
def sample_rows(net, ctx, unit_pos, clean, configuration, rows, generator, temperature, greedy_row=True):
    """``rows`` complete steps decoded as one batch; row 0 is token-level greedy when ``greedy_row``.
    Returns (labels per row, joint log-prob per row [float])."""
    import torch
    import torch.nn.functional as F
    dec = _Dec(net, ctx, unit_pos)
    prefixes = [Prefix(net, clean, configuration) for _ in range(rows)] if _uses_prefix(net) else None
    B = rows
    caches = [dict() for _ in net.dec]
    last = net.bos.expand(B, -1)
    lp = torch.zeros(B, dtype=torch.float64)
    ar = torch.arange(B)
    valid = []

    def choose(logits):
        p = torch.softmax(logits.float() / temperature, -1)
        tok = torch.multinomial(p, 1, generator=generator).squeeze(-1)
        if greedy_row:
            tok[0] = logits[0].argmax(-1)
        return tok

    ut, uq = [[] for _ in range(B)], [[] for _ in range(B)]
    for p in range(dec.n_units):
        valid.append(dec.present[:, p])
        h, logits = dec.unit_step(p, last, caches, valid, prefixes)
        tok = choose(logits)
        needs = net.unit_qty_mask[tok]
        qlog = net.unit_qty_head(h + net.unit_qty_cond(tok))
        qty = torch.where(needs, choose(qlog), torch.full_like(tok, -1))
        lsm, qlsm = F.log_softmax(logits.float(), -1), F.log_softmax(qlog.float(), -1)
        lp += (lsm[ar, tok] + torch.where(needs, qlsm[ar, qty.clamp(min=0)], torch.zeros(B))).double()
        here = dec.present[:, p]
        for b in range(B):
            t, q = int(tok[b]), int(qty[b])
            ut[b].append(t if bool(here[0]) else -1)
            uq[b].append(q if bool(here[0]) else -1)
            if prefixes is not None:
                prefixes[b].apply_unit(p, t, q)
        last = torch.where(here[:, None], net._decision_emb(tok, qty, True), last)
    ot, oq = [[] for _ in range(B)], [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool)
    stop = net.config.stop_index
    for k in range(net.O):
        valid.append(torch.ones(1, dtype=torch.bool))
        h, logits = dec.order_step(k, last, caches, valid, prefixes)
        tok = choose(logits)
        needs = net.order_qty_mask[tok]
        qlog = net.order_qty_head(h + net.order_qty_cond(tok))
        qty = torch.where(needs, choose(qlog), torch.full_like(tok, -1))
        lsm, qlsm = F.log_softmax(logits.float(), -1), F.log_softmax(qlog.float(), -1)
        step_lp = lsm[ar, tok] + torch.where(needs, qlsm[ar, qty.clamp(min=0)], torch.zeros(B))
        lp += torch.where(done, torch.zeros(B), step_lp).double()
        for b in range(B):
            if bool(done[b]):
                continue
            t, q = int(tok[b]), int(qty[b])
            ot[b].append(t)
            oq[b].append(q)
            if prefixes is not None:
                prefixes[b].apply_order(k, t, q)
        done = done | (tok == stop)
        last = net._decision_emb(tok, qty, False)
        if bool(done.all()):
            break
    return [_labels(net, ut[b], uq[b], ot[b], oq[b]) for b in range(B)], [float(x) for x in lp]


# ----------------------------------------------------------------------------- beam search
def beam_search(net, ctx, unit_pos, clean, configuration, width):
    """Beam search over (token, quantity) decisions. Returns (best labels, best joint log-prob, finished count)."""
    import torch
    import torch.nn.functional as F
    dec = _Dec(net, ctx, unit_pos)
    prefixes = [Prefix(net, clean, configuration)] if _uses_prefix(net) else None
    W = int(width)
    caches = [dict() for _ in net.dec]
    last = net.bos.expand(1, -1)
    scores = [0.0]
    seqs = [([], [], [], [])]
    valid = []

    def expand(h, logits, qty_mask, qty_head, qty_cond, allow_stop=False):
        """Top candidates (score, parent, tok, qty) over all beams."""
        B = logits.shape[0]
        lsm = F.log_softmax(logits.float(), -1)
        m = min(W + (1 if allow_stop else 0), lsm.shape[-1])
        top = lsm.topk(m, -1)
        toks = top.indices.reshape(-1)
        par = torch.arange(B).repeat_interleave(m)
        need = qty_mask[toks]
        cands = []
        if bool(need.any()):
            idx = need.nonzero().squeeze(-1)
            hq = h[par[idx]] + qty_cond(toks[idx])
            qlsm = F.log_softmax(qty_head(hq).float(), -1)
            qtop = qlsm.topk(min(W, qlsm.shape[-1]), -1)
            qv, qi = qtop.values.tolist(), qtop.indices.tolist()
            for j, i in enumerate(idx.tolist()):
                b, t = int(par[i]), int(toks[i])
                base = scores[b] + float(top.values.reshape(-1)[i])
                for v, q in zip(qv[j], qi[j]):
                    cands.append((base + v, b, t, int(q)))
        tv = top.values.reshape(-1).tolist()
        for i in (~need).nonzero().squeeze(-1).tolist():
            b, t = int(par[i]), int(toks[i])
            cands.append((scores[b] + tv[i], b, t, -1))
        cands.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))
        return cands

    def reorder(chosen):
        nonlocal last, prefixes
        idx = torch.tensor([c[1] for c in chosen], dtype=torch.long)
        for cache in caches:
            if "k" in cache:
                cache["k"], cache["v"] = cache["k"][idx], cache["v"][idx]
        if prefixes is not None:
            used, new = set(), []
            for c in chosen:
                new.append(prefixes[c[1]] if c[1] not in used else prefixes[c[1]].clone())
                used.add(c[1])
            # clones were taken before any application below
            prefixes = new
        return idx

    for p in range(dec.n_units):
        valid.append(dec.present[:, p])
        h, logits = dec.unit_step(p, last, caches, valid, prefixes)
        chosen = expand(h, logits, net.unit_qty_mask, net.unit_qty_head, net.unit_qty_cond)[:W]
        idx = reorder(chosen)
        here = bool(dec.present[0, p])
        tok = torch.tensor([c[2] for c in chosen], dtype=torch.long)
        qty = torch.tensor([c[3] for c in chosen], dtype=torch.long)
        new_seqs = []
        for i, (s, b, t, q) in enumerate(chosen):
            u_t, u_q, _, _ = seqs[b]
            new_seqs.append((u_t + [t if here else -1], u_q + [q if here else -1], [], []))
            if prefixes is not None:
                prefixes[i].apply_unit(p, t, q)
        seqs, scores = new_seqs, [c[0] for c in chosen]
        prev = last if last.shape[0] == 1 else last[idx]
        last = torch.where(dec.present[:, p][:, None], net._decision_emb(tok, qty, True), prev.expand(len(chosen), -1))
    stop = net.config.stop_index
    finished = []
    for k in range(net.O):
        valid.append(torch.ones(1, dtype=torch.bool))
        h, logits = dec.order_step(k, last, caches, valid, prefixes)
        cands = expand(h, logits, net.order_qty_mask, net.order_qty_head, net.order_qty_cond, allow_stop=True)
        active = []
        for c in cands:
            if c[2] == stop:
                u_t, u_q, o_t, o_q = seqs[c[1]]
                finished.append((c[0], (u_t, u_q, o_t + [stop], o_q + [-1])))
            elif len(active) < W:
                active.append(c)
            if len(active) >= W and len(finished) >= 1 and c[0] < max(f[0] for f in finished):
                break
        best_done = max(f[0] for f in finished) if finished else float("-inf")
        active = [c for c in active if c[0] > best_done]   # a longer step only loses log-prob
        if not active:
            break
        idx = reorder(active)
        tok = torch.tensor([c[2] for c in active], dtype=torch.long)
        qty = torch.tensor([c[3] for c in active], dtype=torch.long)
        new_seqs = []
        for i, (s, b, t, q) in enumerate(active):
            u_t, u_q, o_t, o_q = seqs[b]
            new_seqs.append((u_t, u_q, o_t + [t], o_q + [q]))
            if prefixes is not None:
                prefixes[i].apply_order(k, t, q)
        seqs, scores = new_seqs, [c[0] for c in active]
        last = net._decision_emb(tok, qty, False)
    best = max(finished, key=lambda f: f[0])
    return _labels(net, *best[1]), best[0], len(finished)


# ----------------------------------------------------------------------------- act
def act(net, observation, settings, generator=None, temperature=0.0, state=None, configuration=None,
        code_mode=None, v8_history=None, v8_long=None, info=None):
    """PolicyNet.act with the step chosen by ``settings`` (bestk / beam) instead of decode_free."""
    import torch
    import model as _model
    from farm_neat.training.bc import actor_observation
    with torch.no_grad():
        begun = time.perf_counter()
        clean = actor_observation(dict(observation))
        if "step" not in clean:
            clean["step"] = int(clean["day"]) * 24 + int(clean["hour"])
        step = int(clean["step"])
        rows, token_mask, known, pos = net.observation_tensors(clean, configuration, v8_history=v8_history,
                                                               v8_long=v8_long)
        field = None
        if isinstance(state, _model.ThinkState) and state.field is not None and 0 < step and step > state.last_step:
            field = state.field
        extra, unit_extra, reach_eta = getattr(net, "_last_extra", None) or (None, None, None)
        hist, prev_unit, prev_order = (), None, None
        if getattr(net.config, "v6_prev", False):
            if isinstance(state, _model.ThinkState) and 0 < step and step > state.last_step:
                hist = tuple(state.hist)
            prev_unit, prev_order = _model.v6_history_arrays(hist, step, net.U, net.O)
        ctx = net.encode(rows, token_mask, known, field, code_mode=code_mode or net.config.infer_code_mode,
                         generator=generator, extra=extra, unit_extra=unit_extra, reach_eta=reach_eta,
                         prev_unit=prev_unit, prev_order=prev_order,
                         **(getattr(net, "_last_job_eta", None) or {}), **(getattr(net, "_last_v8", None) or {}))
        encoded = time.perf_counter()
        record = {"step": step, "mode": settings["mode"]}
        if settings["mode"] == "bestk":
            g = torch.Generator(device="cpu").manual_seed(step_seed(clean, settings.get("salt", 0)))
            labels, lps = sample_rows(net, ctx, pos, clean, configuration, 1 + settings["k"], g,
                                      settings["temperature"], greedy_row=True)
            best = max(range(len(lps)), key=lambda i: (lps[i], -i))
            if labels[best] == labels[0] or lps[best] - lps[0] < TIE:
                best = 0                     # same step, or a batch-numerics tie: greedy wins ties
            lab = labels[best]
            record.update(greedy_lp=lps[0], chosen_lp=lps[best], chosen_row=best, changed=best != 0,
                          distinct=len({json.dumps(l, sort_keys=True) for l in labels}))
        elif settings["mode"] == "beam":
            lab, lp, n_fin = beam_search(net, ctx, pos, clean, configuration, settings["width"])
            record.update(chosen_lp=lp, finished=n_fin)
        else:
            raise ValueError("step_search.act called in greedy mode")
        action = net.codec.decode_step(lab, clean)
        if getattr(net.config, "v6_prev", False):
            try:
                import data as _data
                saved = _data.encode_step_lenient(action, clean)
            except ImportError:
                saved = net.codec.encode_step(action, clean)
            hist = (hist + ((step, saved["unit_tok"], saved["unit_qty"],
                             saved["order_tok"], saved["order_qty"]),))[-2:]
        record.update(encode_ms=(encoded - begun) * 1000.0, search_ms=(time.perf_counter() - encoded) * 1000.0)
        if isinstance(info, list):
            info.append(record)
        return action, _model.ThinkState(ctx.field, step, hist)
