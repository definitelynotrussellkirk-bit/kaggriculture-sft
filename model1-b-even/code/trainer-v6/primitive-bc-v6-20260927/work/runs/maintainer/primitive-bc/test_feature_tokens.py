"""feature_tokens.py: token layout, pack/model tile alignment, ablation groups, fp16 storage,
and the documented PolicyNet wiring run on the current model.py.

Run (pinned interpreter, CPU only):
  cd /home/russ/Desktop/farmNEAT && CUDA_VISIBLE_DEVICES= \
  PYTHONPATH=src:work/runs/maintainer/primitive-bc:work/cache/pytest-path \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  work/venv/python -B -m pytest -q -p no:cacheprovider work/runs/maintainer/primitive-bc/test_feature_tokens.py
"""
from __future__ import annotations

import inspect
import json
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import feature_tokens as TK  # noqa: E402
import features as FT  # noqa: E402
from farm_neat.policy import observation as OBS  # noqa: E402
from farm_neat.training import bc  # noqa: E402

DAILY = ROOT / "work/datasets/public-episodes/daily"
EPISODES = ("111424871", "111419545")


@lru_cache(maxsize=None)
def replay(eid):
    return json.loads((DAILY / f"{eid}.json").read_text())


@lru_cache(maxsize=None)
def run(eid, seat):
    """(actor views, features) for every step 0..718 of one seat."""
    data = replay(eid)
    cfg = data["configuration"]
    state = FT.FeatureState(cfg)
    obs, feats = [], []
    for t in range(719):
        o = bc.public_observation(data, t, seat)
        obs.append(o)
        feats.append(state.step(o))
        state.record_action(data["steps"][t + 1][seat]["action"])
    return tuple(obs), tuple(feats), cfg


def _coded_features():
    """Synthetic features whose every value is a unique positive code (block, row, col)."""
    f, code = {"known": {}}, 1.0
    where = {}
    for name, shape in FT.SHAPES.items():
        n = int(np.prod(shape))
        f[name] = (np.arange(code, code + n, dtype=np.float64).reshape(shape)).astype(np.float32)
        f["known"][name] = np.ones(shape, np.bool_)
        rows, cols, _ = FT.BLOCKS[name]
        for i in range(n):
            r, c = divmod(i, len(cols))
            where[code + i] = (name, rows[r], cols[c])
        code += n
    assert code < 2 ** 24   # exact in float32
    return f, where


def test_layout_constants_match_sources():
    assert TK.ITEMS == OBS.ITEMS and TK.LAGS == FT.HF.LAGS and TK.PRODUCERS == FT.TF.PRODUCERS
    assert (TK.K, TK.WIDTH, TK.PAYLOAD, TK.TILE_WIDTH) == (22, 128, 20, 31)
    assert [TK.TYPES[i] for i in TK.TYPE_ID] == [t for t, _ in TK.SLOTS]
    assert len(TK.COLUMN_NAMES) == TK.K and all(len(r) == TK.WIDTH for r in TK.COLUMN_NAMES)


def test_every_feature_value_lands_in_exactly_one_token_cell():
    f, where = _coded_features()
    tok = TK.tokens(f)
    values, known = tok["values"], tok["known"]
    payload = values[:, TK.PAYLOAD:]
    seen = {}
    for k, row in enumerate(payload):
        for j, v in enumerate(row):
            if v:
                assert v not in seen, (TK.SLOT_NAMES[k], j)
                seen[float(v)] = (k, TK.PAYLOAD + j)
    expected = {c for c, (b, _, _) in where.items() if b != "theorem.tile"}
    assert set(seen) == expected       # nothing lost, nothing duplicated
    for c, (k, j) in seen.items():
        block, row, col = where[c]
        assert TK.COLUMN_NAMES[k][j] == f"{block}/{col}"
        slot_type, slot_row = TK.SLOTS[k]
        if block.startswith("history.item"):
            ident = values[k, TK.ID_ITEM:TK.ID_ITEM + len(TK.ITEMS)]
            assert ident[TK.ITEMS.index(row)] == 1 and ident.sum() == 1
        if block == "history.global_lag":
            assert slot_type == "LAG" and row == f"lag{TK.LAGS[slot_row]}"
        if block in ("theorem.product",):
            assert slot_type == "PRODUCT" and TK.PRODUCTS[slot_row] == row
        if block in ("theorem.structure",):
            assert slot_type == "PRODUCER" and TK.PRODUCERS[slot_row] == row
    # history item rows: products only in PRODUCT tokens, animals only in PRODUCER tokens
    for c, (k, _) in seen.items():
        block, row, _ = where[c]
        if block.startswith("history.item"):
            assert TK.SLOTS[k][0] == ("PRODUCT" if row in TK.PRODUCTS else "PRODUCER")
    # identity columns, padding and knownness
    for k, (t, r) in enumerate(TK.SLOTS):
        ident = values[k, :TK.PAYLOAD]
        assert ident[TK.ID_TYPE + TK.TYPES.index(t)] == 1 and ident[:len(TK.TYPES)].sum() == 1
        assert known[k, :TK.PAYLOAD].all()
        used = np.asarray([bool(n) for n in TK.COLUMN_NAMES[k][TK.PAYLOAD:]])
        assert np.all(values[k, TK.PAYLOAD:][~used] == 0) and not known[k, TK.PAYLOAD:][~used].any()
        assert known[k, TK.PAYLOAD:][used].all()
    assert np.array_equal(tok["tile"], f["theorem.tile"])


@pytest.mark.parametrize("eid", EPISODES)
def test_real_replay_tokens(eid):
    """Every step, both seats: values and knownness copied exactly, finite, fp16-safe."""
    worst = 0.0
    for seat in (0, 1):
        _, feats, _ = run(eid, seat)
        for f in feats:
            tok = TK.tokens(f)
            v, k = tok["values"], tok["known"]
            assert np.isfinite(v).all() and np.all(v[~k] == 0)
            for slots, block, rows, off in TK.SEGMENTS:
                w = FT.SHAPES[block][-1]
                src = f[block][None] if rows is None else f[block][list(rows)]
                ks = f["known"][block][None] if rows is None else f["known"][block][list(rows)]
                assert np.array_equal(v[list(slots), off:off + w], src)
                assert np.array_equal(k[list(slots), off:off + w], ks)
            h = v.astype(np.float16)
            assert np.isfinite(h).all() and np.isfinite(tok["tile"].astype(np.float16)).all()
            big = np.abs(v) > 1e-3
            worst = max(worst, float((np.abs(h.astype(np.float32) - v)[big] / np.abs(v[big])).max()))
    assert worst < 1e-3, worst


@pytest.mark.parametrize("seat", (0, 1))
def test_tiles_align_with_pack_tokens_and_model_cells(seat):
    obs, feats, cfg = run(EPISODES[1], seat)
    for t in (0, 250, 600):
        packed = OBS.pack(obs[t], None, {k: cfg[k] for k in OBS.CONFIG})
        rows, ids = packed["rows"], packed["token_ids"]
        index = TK.pack_tile_rows(ids)
        for cell, i in enumerate(index):
            role, rest = divmod(cell, 100)
            y, x = divmod(rest, 10)
            r = rows[i]
            # PolicyNet._tiles reads exactly these columns to place a token in its cell
            assert r[OBS.KINDS.index("TILE")] == 1 and r[12 + role] == 1
            assert round(r[14] * 9) == x and round(r[15] * 9) == y
            assert ids[i] == f"tile/{role}/{x},{y}"
        aligned = TK.align_tiles(TK.tokens(feats[t])["tile"], ids)
        assert np.array_equal(aligned[index], feats[t]["theorem.tile"])
        other = np.setdiff1d(np.arange(len(ids)), index)
        assert not aligned[other].any()
        # own farm tile under the farmer has unit_distance 0
        col = FT.BLOCKS["theorem.tile"][1].index("unit_distance")
        fx, fy = obs[t]["farms"][seat]["farmer"]
        assert feats[t]["theorem.tile"][fy * 10 + fx, col] == 0


def test_drop_groups_ablate_values_and_knownness_only():
    _, feats, _ = run(EPISODES[0], 0)
    f = feats[500]
    full = TK.tokens(f)
    cut = TK.tokens(f, drop=TK.PREV_ACTION_GROUPS)
    names = np.asarray(TK.COLUMN_NAMES)
    dropped = np.char.startswith(names, "history.prev_action/") | np.char.startswith(names, "history.item_now/prev.")
    assert dropped.sum() == 24 + 7 * 12
    assert not cut["values"][dropped].any() and not cut["known"][dropped].any()
    assert np.array_equal(cut["values"][~dropped], full["values"][~dropped])
    assert np.array_equal(cut["known"][~dropped], full["known"][~dropped])
    assert full["values"][dropped].any()
    no_tile = TK.tokens(f, drop=("theorem.tile",))
    assert not no_tile["tile"].any() and np.array_equal(no_tile["values"], full["values"])
    with pytest.raises(KeyError):
        TK.tokens(f, drop=("no.such.group",))


def _feature_policy_class(M, torch):
    """PolicyNet subclass whose encode is the CURRENT model.PolicyNet.encode source with
    feature_tokens.POLICYNET_PATCH applied (fails if an anchor line is gone)."""
    source = textwrap.dedent(inspect.getsource(M.PolicyNet.encode))
    namespace = dict(vars(M))
    exec(compile(TK.patch_encode_source(source), "<patched PolicyNet.encode>", "exec"), namespace)

    class FeaturePolicy(M.PolicyNet):
        def __init__(self, config):
            super().__init__(config)
            self.feat = TK.FeatureTokenInput(self.config.width)

    FeaturePolicy.encode = namespace["encode"]
    return FeaturePolicy


def test_documented_policynet_wiring_runs_on_current_model():
    """POLICYNET_PATCH applied to the current model.PolicyNet.encode: inert without features;
    with features forward, backward and free decode run in both think modes; gradients reach
    every feature parameter; the zero-initialised tile projection adds nothing at init."""
    torch = pytest.importorskip("torch")
    M = pytest.importorskip("model")
    data_mod = pytest.importorskip("data")
    import torch.nn.functional as F
    FeaturePolicy = _feature_policy_class(M, torch)

    data = replay(EPISODES[1])
    cfg = data["configuration"]
    examples, toks = [], []
    for seat, t in ((0, 300), (1, 450)):
        obs, feats, _ = run(EPISODES[1], seat)
        ex = data_mod.features(obs[t], cfg)
        ex.update(data_mod.codec.encode_step(data["steps"][t + 1][seat]["action"], obs[t]))
        examples.append(ex)
        toks.append(TK.tokens(feats[t], dtype=np.float16))    # stored like data.py rows
    n = max(len(e["rows"]) for e in examples)

    def pad(a, fill=0):
        out = np.full((n,) + a.shape[1:], fill, a.dtype)
        out[:len(a)] = a
        return out
    rows = torch.from_numpy(np.stack([pad(e["rows"]) for e in examples]).astype(np.float32))
    mask = torch.from_numpy(np.stack([pad(e["token_mask"], False) for e in examples]))
    known = torch.from_numpy(np.stack([pad(e["knownness"], False) for e in examples]))
    batch = TK.collate(toks)
    feat = {k: torch.from_numpy(v.astype(np.float32) if v.dtype == np.float16 else v) for k, v in batch.items()}
    for think in ("bottleneck", "off"):
        torch.manual_seed(0)
        net = FeaturePolicy(M.ModelConfig(width=32, layers=2, heads=2, ff_mult=2, dec_layers=1, G=4, K=8, C=2,
                                          think=think))
        torch.manual_seed(0)
        plain = M.PolicyNet(net.config)
        plain.load_state_dict({k: v for k, v in net.state_dict().items() if not k.startswith("feat.")})
        with torch.no_grad():   # the patch is inert when no features are passed
            a = plain.encode(rows, mask, known, None, code_mode="argmax")
            b = net.encode(rows, mask, known, None, code_mode="argmax")
            assert torch.equal(a.mem, b.mem) and torch.equal(a.value, b.value)
        U = net.U
        pos = torch.full((2, U, 2), -1, dtype=torch.long)
        ut = torch.full((2, U), -1, dtype=torch.long)
        uq = torch.full((2, U), -1, dtype=torch.long)
        for b, e in enumerate(examples):
            up = torch.from_numpy(e["unit_pos"])[:U]
            pos[b, :len(up)] = up
            k = min(U, len(e["unit_tok"]))
            ut[b, :k] = torch.tensor(e["unit_tok"][:k])
            uq[b, :k] = torch.tensor(e["unit_qty"][:k])
        O = net.O
        ot = torch.full((2, O), -1, dtype=torch.long)
        oq = torch.full((2, O), -1, dtype=torch.long)
        for b, e in enumerate(examples):
            m = min(O, len(e["order_tok"]))
            ot[b, :m] = torch.tensor(e["order_tok"][:m])
            oq[b, :m] = torch.tensor(e["order_qty"][:m])
        ctx = net.encode(rows, mask, known, None, feat=feat)
        out = net.decode_teacher(ctx, pos, torch.stack((ut, uq), -1), torch.stack((ot, oq), -1))
        present = ut >= 0
        loss = F.cross_entropy(out["unit_logits"][present], ut[present]) + ctx.value.sum() * 0
        ov = ot >= 0
        loss = loss + F.cross_entropy(out["order_logits"][ov], ot[ov])
        loss.backward()
        grads = {name: p.grad for name, p in net.feat.named_parameters()}
        assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
        assert grads["inp.weight"].abs().sum() > 0 and grads["tile.weight"].abs().sum() > 0
        # zero-initialised tile projection: tile additions are exactly zero before training
        _, tile_add = net.feat(feat["values"], feat["known"], feat["tile"])
        assert not tile_add.any()
        with torch.no_grad():
            free = net.decode_free(net.encode(rows, mask, known, None, feat=feat, code_mode="argmax"), pos)
        assert free["unit_tok"].shape == (2, U)
