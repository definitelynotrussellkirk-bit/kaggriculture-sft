"""train.py checks: synthetic tiny fit, exact resume, CPU numerics, carried-field gradients."""
from __future__ import annotations

import contextlib
import math
import dataclasses
import glob
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import model as M  # noqa: E402
import train as T  # noqa: E402
from test_model import CODEC  # noqa: E402

CPU = torch.device("cpu")


def tiny_cfg(**kw):
    base = dict(synthetic=True, window=4, burn_in=2, batch_windows=8, updates=6, warmup=2, lr=3e-3, eval_every=0,
                ckpt_every=1000, log_every=1000, synthetic_len=12, tau_start=1.0, tau_end=0.5)
    base.update(kw)
    return T.TrainConfig(**base)


def quiet(_):
    pass


def test_tiny_fit_drives_accuracy_up(tmp_path):
    cfg = tiny_cfg(updates=300, warmup=20)
    mc = T.tiny_model_config()
    torch.manual_seed(cfg.seed)
    before_net = T.build(mc, CPU, CODEC)
    held = T.held_out_batches(before_net, cfg, windows=32)
    before = T.evaluate(before_net, held, cfg, CPU)
    net, _ = T.fit(cfg, mc, tmp_path, codec=CODEC, log=quiet)
    after = T.evaluate(net, held, cfg, CPU)
    for key in ("acc_unit_tok", "acc_order_tok", "acc_unit_exact", "acc_step_exact", "acc_free_step_exact"):
        assert after[key] > before[key], (key, before[key], after[key])
    assert after["acc_unit_tok"] > before["acc_unit_tok"] + 0.3
    assert after["acc_order_tok"] > before["acc_order_tok"] + 0.3
    assert after["codes"]["codes_distinct_used"] > 1
    assert (tmp_path / "LATEST").exists() and (tmp_path / "metrics.jsonl").exists()


def _states_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_exact_checkpoint_resume(tmp_path):
    cfg = tiny_cfg(updates=6, ckpt_every=3)
    mc = T.tiny_model_config()
    full, _ = T.fit(cfg, mc, tmp_path / "full", codec=CODEC, log=quiet)
    T.fit(cfg, mc, tmp_path / "split", codec=CODEC, log=quiet, stop_after=3)
    blob, _ = T.load_checkpoint(tmp_path / "split")
    assert blob["update"] == 3 and blob["windows_consumed"] == 3 * cfg.batch_windows
    resumed, _ = T.fit(cfg, mc, tmp_path / "split", codec=CODEC, log=quiet, resume=True)
    assert _states_equal(full.state_dict(), resumed.state_dict())
    a, _ = T.load_checkpoint(tmp_path / "full")
    b, _ = T.load_checkpoint(tmp_path / "split")
    assert a["update"] == b["update"] == 6
    assert torch.equal(a["generator_state"], b["generator_state"])
    for sa, sb in zip(a["optimizer"]["state"].values(), b["optimizer"]["state"].values()):
        assert all(torch.equal(sa[k], sb[k]) for k in sa)
    # the checkpoint loads as an inference policy
    pol = M.PolicyNet.load(tmp_path / "split" / (tmp_path / "split" / "LATEST").read_text().strip(), codec=CODEC)
    assert _states_equal(pol.state_dict(), resumed.state_dict())


def test_bf16_autocast_only_under_cuda():
    assert isinstance(T.autocast("cpu", True), contextlib.nullcontext)
    net = T.build(T.tiny_model_config(), CPU, CODEC)
    seen = []
    net.inp.register_forward_hook(lambda m, i, o: seen.append((torch.is_autocast_enabled("cpu"), o.dtype)))
    cfg = tiny_cfg(bf16=True)
    batch = T.collate(T.SyntheticStream(net.config, cfg).batch(0, 2), net)
    loss, *_ = T.run_window_batch(net, batch, device=CPU, bf16=True, generator=torch.Generator().manual_seed(0))
    assert seen and all(not on and dt == torch.float32 for on, dt in seen)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    if torch.cuda.is_available():  # pragma: no cover - CPU-only test machine
        ctx = T.autocast("cuda", True)
        assert isinstance(ctx, torch.autocast)


def test_gradient_reaches_first_codes_through_the_carry():
    torch.manual_seed(0)
    net = T.build(T.tiny_model_config(), CPU, CODEC).train()
    cfg = tiny_cfg(window=8, burn_in=4, synthetic_len=24)
    ws = [w for w in T.SyntheticStream(net.config, cfg).batch(0, 3) if w["n_burn"] == 4]
    batch = T.collate(ws, net)
    rep = T.carry_probe(net, batch, lags=(1, 3, 7))
    assert rep["scored_steps"] == 8
    assert rep["grad_norm_first_codes_from_later_losses"] > 0
    assert all(v > 0 for v in rep["grad_norm_first_codes_by_lag"].values()), rep
    assert rep["saved_activation_bytes"] > 0


def test_burn_in_has_no_loss_and_no_gradient():
    torch.manual_seed(0)
    net = T.build(T.tiny_model_config(), CPU, CODEC).train()
    cfg = tiny_cfg(window=2, burn_in=3, synthetic_len=12)
    ws = [w for w in T.SyntheticStream(net.config, cfg).batch(0, 6) if w["n_burn"] == 3][:2]
    batch = T.collate(ws, net)
    loss, meter, step_losses, _ = T.run_window_batch(net, batch, device=CPU, generator=torch.Generator().manual_seed(0))
    assert len(step_losses) == 2 and meter.sums["scored_example_steps"] == 2 * len(ws)


def test_step_loss_ignores_absent_and_forced_stop():
    net = T.build(T.tiny_model_config(), CPU, CODEC)
    B, U, O = 2, net.U, net.O
    out = {"unit_logits": torch.zeros(B, U, 7), "unit_qty_logits": torch.zeros(B, U, 75),
           "order_logits": torch.zeros(B, O, 5), "order_qty_logits": torch.zeros(B, O, 75)}
    out["order_logits"][:, -1, 1:] = float("-inf")
    ut = torch.full((B, U), -1)
    ut[:, 0] = 5                                           # PICKUP:A with a quantity
    uq = torch.full((B, U), -1)
    uq[:, 0] = 3
    ot = torch.full((B, O), -1)
    ot[0, 0] = 0                                           # empty queue
    ot[1, :] = torch.tensor([1, 2, 3, 4, 0])               # full queue, STOP in the forced slot
    oq = torch.full((B, O), -1)
    per, counts = T.step_loss(out, (ut, uq, ot, oq), net)
    assert torch.isfinite(per).all()
    assert counts["unit_tok"][1] == 2 and counts["unit_qty"][1] == 2 and counts["order_tok"][1] == 6


def test_real_data_window_batch():
    """data.iter_windows -> data.collate_windows -> one training pass with the real codec."""
    data = pytest.importorskip("data")
    if not glob.glob(data.ARCHIVE_GLOB) or not data.LEADERBOARD.exists():
        pytest.skip("replay archives / leaderboard not on this machine")
    archive = sorted(glob.glob(data.ARCHIVE_GLOB))[-1:]
    it = data.iter_windows(archive, split="train", seats=0, workers=0, window=4, burn_in=3, max_episodes=1)
    ws = [next(it) for _ in range(3)]
    torch.manual_seed(0)
    net = T.build(M.ModelConfig(width=32, layers=1, heads=2, dec_layers=1, G=4, K=8, C=2), CPU)
    batch = T.collate(ws, net)
    assert batch["rows"].shape[:2] == (3, 7) and batch["unit_tok"].shape[-1] == net.U
    loss, meter, steps, codes0 = T.run_window_batch(net, batch, device=CPU, generator=torch.Generator().manual_seed(0),
                                                    keep_first_codes=True)
    assert torch.isfinite(loss) and len(steps) == 4
    loss.backward()
    assert codes0.grad is not None and float(codes0.grad.norm()) > 0


def test_grad_checkpoint_matches_plain_backward():
    torch.manual_seed(0)
    net = T.build(T.tiny_model_config(), CPU, CODEC).train()
    cfg = tiny_cfg(window=3, burn_in=2, synthetic_len=12)
    batch = T.collate(T.SyntheticStream(net.config, cfg).batch(0, 4), net)
    grads = []
    for ckpt in (False, True):
        net.zero_grad()
        loss, *_ = T.run_window_batch(net, batch, device=CPU, generator=torch.Generator().manual_seed(3),
                                      grad_checkpoint=ckpt)
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    assert all(torch.allclose(grads[0][n], grads[1][n], atol=1e-6) for n in grads[0])


def test_usage_regulariser_is_reported():
    torch.manual_seed(0)
    net = T.build(T.tiny_model_config(), CPU, CODEC).train()
    batch = T.collate(T.SyntheticStream(net.config, tiny_cfg()).batch(0, 4), net)
    loss0, m0, *_ = T.run_window_batch(net, batch, device=CPU, generator=torch.Generator().manual_seed(1))
    loss1, m1, *_ = T.run_window_batch(net, batch, device=CPU, generator=torch.Generator().manual_seed(1), usage_reg=0.5)
    assert "usage_reg_term" not in m0.sums and 0 <= m1.sums["usage_reg_term"] <= math.log(8) + 1e-6
    assert loss1.item() >= loss0.item()
