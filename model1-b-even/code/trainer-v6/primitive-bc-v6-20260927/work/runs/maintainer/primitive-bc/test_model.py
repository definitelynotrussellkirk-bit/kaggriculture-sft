"""model.py checks on a small stub vocabulary (the real codec is used only where noted)."""
from __future__ import annotations

import glob
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import model as M  # noqa: E402


def stub_codec():
    c = types.ModuleType("stub_codec")
    c.UNIT_VOCAB = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP:A", "PLACE:A"]
    c.ORDER_VOCAB = ["STOP", "HIRE", "SELL:A", "BUY_SEED:B", "BUY_LAND"]
    c.MAX_UNITS, c.MAX_ORDERS, c.QTY_BIN_COUNT = 5, 4, 75
    c.UNIT_HAS_QTY = [":" in t for t in c.UNIT_VOCAB]
    c.ORDER_HAS_QTY = [t.startswith(("SELL", "BUY_SEED")) for t in c.ORDER_VOCAB]

    def decode_step(labels, obs):
        return {"farmer": [c.UNIT_VOCAB[labels["unit_tok"][0]]], "hands": [], "market": []}
    c.decode_step = decode_step
    return c


CODEC = stub_codec()


def tiny(**kw):
    base = dict(width=32, layers=2, heads=2, ff_mult=2, dec_layers=2, G=4, K=8, C=2)
    base.update(kw)
    torch.manual_seed(0)
    return M.PolicyNet(M.ModelConfig(**base), codec=CODEC)


def inputs(batch=3, units=(1, 3, 5), seed=0):
    rows, mask, known, _ = M.synthetic_inputs(230, 1, CODEC.MAX_UNITS, batch=batch, seed=seed)
    pos = torch.full((batch, CODEC.MAX_UNITS, 2), -1, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    for b, u in enumerate(units[:batch]):
        pos[b, :u] = torch.randint(10, (u, 2), generator=g)
    return rows, mask, known, pos


def labels(pos, seed=0):
    g = torch.Generator().manual_seed(seed)
    B, U = pos.shape[:2]
    O = CODEC.MAX_ORDERS + 1
    ut = torch.randint(len(CODEC.UNIT_VOCAB), (B, U), generator=g)
    ut[pos[..., 0] < 0] = -1
    has = torch.tensor(CODEC.UNIT_HAS_QTY)[ut.clamp(min=0)] & (ut >= 0)
    uq = torch.where(has, torch.randint(75, (B, U), generator=g), -1)
    ot = torch.full((B, O), -1, dtype=torch.long)
    oq = torch.full((B, O), -1, dtype=torch.long)
    for b in range(B):
        n = b % O
        ot[b, :n] = torch.randint(1, len(CODEC.ORDER_VOCAB), (n,), generator=g)
        ot[b, n] = 0
        ohas = torch.tensor(CODEC.ORDER_HAS_QTY)[ot[b, :n]]
        oq[b, :n] = torch.where(ohas, torch.randint(75, (n,), generator=g), -1)
    return torch.stack((ut, uq), -1), torch.stack((ot, oq), -1)


@pytest.mark.parametrize("think", ["bottleneck", "off"])
def test_forward_shapes(think):
    net = tiny(think=think)
    rows, mask, known, pos = inputs()
    up, op = labels(pos)
    out = net(rows.half(), mask, known, pos, up, op)
    B = rows.shape[0]
    assert out["unit_logits"].shape == (B, 5, 7)
    assert out["unit_qty_logits"].shape == (B, 5, 75)
    assert out["order_logits"].shape == (B, 5, 5)
    assert out["order_qty_logits"].shape == (B, 5, 75)
    assert out["value"].shape == (B,)
    # the last order slot can only be STOP
    assert torch.isinf(out["order_logits"][:, -1, 1:]).all() and torch.isfinite(out["order_logits"][:, -1, 0]).all()
    if think == "bottleneck":
        assert out["field"].codes.shape == (B, 200 + 4, 2, 8)
        assert torch.equal(out["field"].codes.sum(-1), torch.ones(B, 204, 2))
    else:
        assert out["field"] is None


def test_tile_locator_rejects_missing_cells():
    net = tiny()
    rows, mask, known, pos = inputs()
    rows[0, 5, M.TILE_KIND_COL] = 0          # one own tile loses its TILE kind
    with pytest.raises(ValueError):
        net.encode(rows, mask, known)


def test_tile_locator_is_order_free():
    """Cells are located from the rows' own role/x/y columns, not from token order
    (free slots; anchored slots follow the pack order of the state entities by design)."""
    net = tiny(slot_anchor="free").eval()
    rows, mask, known, pos = inputs(batch=1)
    perm = torch.cat((torch.tensor([0]), 1 + torch.randperm(rows.shape[1] - 1, generator=torch.Generator().manual_seed(3))))
    a = net.encode(rows, mask, known, code_mode="argmax")
    b = net.encode(rows[:, perm], mask[:, perm], known[:, perm], code_mode="argmax")
    assert torch.allclose(a.code_logits, b.code_logits, atol=1e-4)


@pytest.mark.parametrize("think", ["bottleneck", "off"])
def test_cached_decode_matches_teacher_forcing(think):
    """Greedy cached sequential decode == argmax of the teacher-forced pass on its own output."""
    net = tiny(think=think).eval()
    rows, mask, known, pos = inputs(batch=4, units=(1, 2, 5, 3))
    with torch.no_grad():
        ctx = net.encode(rows, mask, known, code_mode="argmax")
        pred = net.decode_free(ctx, pos, temperature=0.0, full_orders=True)
        up = torch.stack((pred["unit_tok"], pred["unit_qty"]), -1)
        op = torch.stack((pred["order_tok"], pred["order_qty"]), -1)
        tf = net.decode_teacher(ctx, pos, up, op)
    present = pos[..., 0] >= 0
    assert torch.equal(tf["unit_logits"].argmax(-1)[present], pred["unit_tok"][present])
    has_q = pred["unit_qty"] >= 0
    assert torch.equal(tf["unit_qty_logits"].argmax(-1)[has_q], pred["unit_qty"][has_q])
    ovalid = pred["order_tok"] >= 0
    assert torch.equal(tf["order_logits"].argmax(-1)[ovalid], pred["order_tok"][ovalid])
    oq = pred["order_qty"] >= 0
    assert torch.equal(tf["order_qty_logits"].argmax(-1)[oq], pred["order_qty"][oq])
    # every row stops, and nothing after STOP
    for b in range(4):
        toks = pred["order_tok"][b].tolist()
        assert 0 in toks and all(t == -1 for t in toks[toks.index(0) + 1:])
    assert (pred["unit_tok"][~present] == -1).all()


def test_decoder_reads_encoder_only_through_codes():
    """With non-differentiable (argmax) codes, action losses give the encoder no gradient;
    with straight-through Gumbel codes they do.  think='off' reaches it directly."""
    rows, mask, known, pos = inputs()
    up, op = labels(pos)

    def enc_grad(net, mode):
        net.zero_grad()
        out = net(rows, mask, known, pos, up, op, code_mode=mode)
        (out["unit_logits"].sum() + out["order_logits"].masked_fill(torch.isinf(out["order_logits"]), 0).sum()
         + out["unit_qty_logits"].sum() + out["value"].sum()).backward()
        g = [p.grad for p in net.blocks.parameters()]
        return sum(0.0 if x is None else float(x.abs().sum()) for x in g)

    assert enc_grad(tiny(), "argmax") == 0.0
    assert enc_grad(tiny(), "gumbel") > 0.0
    assert enc_grad(tiny(think="off"), "argmax") > 0.0


def test_field_is_carried():
    net = tiny().eval()
    rows, mask, known, pos = inputs()
    up, op = labels(pos)
    with torch.no_grad():
        a = net(rows, mask, known, pos, up, op, code_mode="argmax")
        f = a["field"]
        b = net(rows, mask, known, pos, up, op, field=f, code_mode="argmax")
        other = M.FieldState(torch.roll(f.codes, 1, dims=-1), f.fresh)
        c = net(rows, mask, known, pos, up, op, field=other, code_mode="argmax")
    assert not torch.allclose(a["code_logits"], b["code_logits"])     # 'none' vs carried codes
    assert not torch.allclose(b["code_logits"], c["code_logits"])     # carried content matters
    upd = M.FieldState.none(3, net.S, 2, 8).update(f.codes, torch.tensor([True, False, True]))
    assert upd.fresh.tolist() == [False, True, False]
    assert torch.equal(upd.codes[1], torch.zeros_like(upd.codes[1]))


def test_gumbel_forward_is_hard_and_seeded():
    net = tiny()
    rows, mask, known, pos = inputs()
    noise = torch.rand(3, net.S, 2, 8, generator=torch.Generator().manual_seed(1))
    a = net.encode(rows, mask, known, code_mode="gumbel", code_noise=noise, tau=2.0)
    b = net.encode(rows, mask, known, code_mode="gumbel", code_noise=noise, tau=0.3)
    assert torch.equal(a.field.codes.detach(), b.field.codes.detach())   # forward = hard sample, tau-free
    assert set(a.field.codes.detach().unique().tolist()) <= {0.0, 1.0}
    s = net.encode(rows, mask, known, code_mode="sample", code_noise=noise)
    assert torch.equal(s.field.codes, a.field.codes.detach())            # inference sample == training forward


def test_slot_anchoring():
    """Slot g is written from the g-th fixed state entity, then own carriers, then free tokens."""
    net = tiny(G=8)
    rows, mask, known, pos = inputs(batch=2)
    src = net._slot_sources(rows, mask)
    kinds = rows[..., :12].argmax(-1)
    for b in range(2):
        fixed = [t for t in range(rows.shape[1]) if int(kinds[b, t]) in M.ANCHOR_KINDS]
        carriers = [t for t in range(rows.shape[1]) if int(kinds[b, t]) == M.CARRIER_KIND and rows[b, t, 12] > 0.5]
        expect = (fixed + carriers)[:8]
        expect += [rows.shape[1] + g for g in range(len(expect), 8)]
        assert src[b].tolist() == expect
    free = tiny(G=8, slot_anchor="free")._slot_sources(rows, mask)
    assert free[0].tolist() == [rows.shape[1] + g for g in range(8)]
    bad = mask.clone()
    bad[1, 0] = False                                   # unequal fixed-entity counts in one batch
    with pytest.raises(ValueError):
        net._slot_sources(rows, bad)


def test_save_load_roundtrip(tmp_path):
    net = tiny().eval()
    path = tmp_path / "w.pt"
    net.save(path)
    blob = torch.load(path, weights_only=True)
    assert blob["config"]["n_unit_tokens"] == 7
    back = M.PolicyNet.load(path, codec=CODEC)
    for (k, v), (k2, v2) in zip(net.state_dict().items(), back.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)


def test_latency_helper_runs():
    r = M.latency(tiny(), n_tokens=240, n_units=3, repeats=2, warmup=1)
    assert r["median_ms"] > 0 and r["units"] == 3


# ----------------------------------------------------------------------------- real codec
def _real_replay():
    paths = sorted(glob.glob("/mnt/steam/farm-data/kaggle/kaggriculture-episodes-2026-09-2*/*.zip"))
    if not paths:
        pytest.skip("no replay archive on this machine")
    with zipfile.ZipFile(paths[-1]) as z:
        name = next(n for n in z.namelist() if n.endswith(".json"))
        return json.loads(z.read(name))


def test_act_real_codec_and_field_state():
    pytest.importorskip("codec")
    import codec
    from farm_neat.training import bc
    replay = _real_replay()
    torch.manual_seed(0)
    net = M.PolicyNet(M.ModelConfig(width=64, layers=2, heads=2, G=8, K=16, C=2)).eval()
    assert net.U == codec.MAX_UNITS and net.config.n_unit_tokens == len(codec.UNIT_VOCAB)
    gen = torch.Generator().manual_seed(0)
    state = net.initial_state()
    for t in (0, 1, 2):
        obs = bc.public_observation(replay, t, 1)
        action, state = net.act(obs, generator=gen, temperature=1.0, state=state)
        assert set(action) == {"farmer", "hands", "market"}
        assert len(action["hands"]) == len(obs["farms"][1]["hands"]) and len(action["market"]) <= codec.MAX_ORDERS
        assert state.last_step == t and state.field is not None
    # a new episode (step 0) must not reuse the carried field
    obs0 = bc.public_observation(replay, 0, 1)
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    a1, _ = net.act(obs0, generator=g1, temperature=0.0, state=state, code_mode="argmax")
    a2, _ = net.act(obs0, generator=g2, temperature=0.0, state=None, code_mode="argmax")
    assert a1 == a2
    # the observation without a 'step' key (seat 1 in public replays) is accepted
    raw = dict(replay["steps"][5][1]["observation"])
    raw.pop("step", None)
    action, st = net.act(raw, state=state)
    assert st.last_step == 5
