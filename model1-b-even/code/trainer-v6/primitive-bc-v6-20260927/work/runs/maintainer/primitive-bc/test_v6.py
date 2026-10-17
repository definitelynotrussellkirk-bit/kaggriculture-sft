"""v6 (pbc-v6-design/DESIGN.md): a parent grafted with v6_prev + v6_aux gives the parent's exact loss in eval mode, the aux
heads add a finite loss only in training mode, gradients reach the zero-initialised gates, and the gradient-checkpoint path
matches the plain path.  Needs the recorded replay and parent checkpoint named below (skips when absent).

  python -m unittest test_v6 -v
"""
import dataclasses
import json
import os
import sys
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

import data  # noqa: E402
import model as M  # noqa: E402
import train as T  # noqa: E402

PARENT = Path(os.environ.get("V6_PARENT", "/mnt/ai/scratch/farmNEAT/primitive-bc-v1-v4-20260926/work/runs/maintainer/"
                                          "primitive-bc/results/v4b-g2/ckpt-0048236.pt"))
PRIO = HERE / "PRIORITY-V6-NO4Q.tsv"


def _batch():
    row = [ln.rstrip("\n").split("\t") for ln in open(PRIO)][1]
    archive, member, day, eid, seat = row[0], row[1], row[2], int(row[3]), int(row[4])
    replay = json.loads(zipfile.ZipFile(archive).read(member))
    meta = {"episode_id": eid, "seat": seat, "team": None, "rank": None, "opponent_rank": None, "day": day}
    ex = list(data.episode_examples(replay, seat, meta_base=meta, steps=range(100, 112)))
    data.add_v6(ex, data.opponent_unit_tokens(replay, seat, [e["meta"]["step"] for e in ex]))
    wins = list(data.make_windows(ex, window=4, burn_in=2))
    return data.collate_windows(wins[:2])


@unittest.skipUnless(PARENT.exists() and PRIO.exists(), "parent checkpoint or priority rows not available")
class V6Graft(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        blob = torch.load(str(PARENT), map_location="cpu", weights_only=True)
        pc = M.ModelConfig.from_dict(blob["config"])
        cls.parent = M.PolicyNet(pc)
        cls.parent.load_state_dict(blob["model"])
        cls.child = M.PolicyNet(dataclasses.replace(pc, v6_prev=True, v6_aux=True))
        res = cls.child.load_state_dict(blob["model"], strict=False)
        cls.missing, cls.unexpected = res.missing_keys, res.unexpected_keys
        cls.batch = _batch()

    def run_(self, net, train, **kw):
        net.train(train)
        return T.run_window_batch(net, self.batch, device="cpu", bf16=False, aux_w=(0.05, 0.03), **kw)

    def test_only_v6_tensors_are_new(self):
        new = [k for k in self.missing if not k.startswith(("v6_", "grow_v6_", "aux_"))]
        self.assertEqual(new, [])
        self.assertEqual(self.unexpected, [])

    def test_batch_carries_v6(self):
        b = self.batch
        self.assertEqual(tuple(b["prev_unit"].shape[2:]), (2, self.parent.U, 2))
        self.assertEqual(tuple(b["prev_order"].shape[2:]), (2, self.parent.O, 2))
        self.assertTrue(bool((b["prev_unit"][..., 0] >= 0).any()))
        self.assertTrue(bool((b["next_unit_tok"] >= 0).any()))
        self.assertTrue(bool((b["opp_unit_tok"] >= 0).any()))

    def test_inert_at_graft(self):
        with torch.no_grad():
            lp, mp, _, _ = self.run_(self.parent, False)
            lc, mc, _, _ = self.run_(self.child, False)
        self.assertEqual(float(lp), float(lc))
        for k in ("unit_tok", "order_tok", "unit_qty", "order_qty"):
            self.assertEqual(mp.pairs[k], mc.pairs[k])

    def test_aux_in_training_only(self):
        with torch.no_grad():
            le, me, _, _ = self.run_(self.child, False)
            self.child.train(True)
            torch.manual_seed(1)
            lt, mt, _, _ = self.run_(self.child, True)
        self.assertIn("loss_aux_opp0", mt.sums)
        self.assertNotIn("loss_aux_opp0", me.sums)
        aux = 0.05 * (mt.sums["loss_aux_next0"] + mt.sums["loss_aux_next1"]) + 0.03 * sum(
            mt.sums[f"loss_aux_opp{i}"] for i in range(3))
        n = mt.sums["scored_example_steps"]
        self.assertTrue(torch.isfinite(lt))
        self.assertAlmostEqual(float(lt), float(le) + aux / n, delta=1e-3 * abs(float(lt)))

    def test_gradients_reach_gates_and_heads(self):
        self.child.zero_grad(set_to_none=True)
        loss, _, _, _ = self.run_(self.child, True)
        loss.backward()
        g = {n: p.grad for n, p in self.child.named_parameters()}
        for name in ("grow_v6_hist.weight", "grow_v6_prev_unit.weight", "aux_next.weight", "aux_opp.weight"):
            self.assertIsNotNone(g[name], name)
            self.assertGreater(float(g[name].abs().sum()), 0.0, name)
        self.assertEqual(tuple(self.child.grow_v6_prev_unit.weight.shape), (self.child.d, 2 * self.child.d))

    def test_checkpoint_path_matches(self):
        torch.manual_seed(2)
        with torch.no_grad():
            a, _, _, _ = self.run_(self.child, False)
            b, _, _, _ = self.run_(self.child, False, grad_checkpoint=True)
        self.assertAlmostEqual(float(a), float(b), places=4)


@unittest.skipUnless(PRIO.exists(), "priority rows not available")
class V6PlayHistoryParity(unittest.TestCase):
    """The play-side builder (model.v6_history_arrays over the agent's own decisions) equals data.add_v6 when fed the
    recorded decisions, on every step of a recorded game (day boundaries included)."""

    def test_every_step(self):
        row = [ln.rstrip("\n").split("\t") for ln in open(PRIO)][1]
        archive, member, day, eid, seat = row[0], row[1], row[2], int(row[3]), int(row[4])
        replay = json.loads(zipfile.ZipFile(archive).read(member))
        meta = {"episode_id": eid, "seat": seat, "team": None, "rank": None, "opponent_rank": None, "day": day}
        ex = data.add_v6(list(data.episode_examples(replay, seat, meta_base=meta)))
        U, O = data.codec.MAX_UNITS, data.N_ORDER_SLOTS
        hist, checked = (), 0
        for e in ex:
            t = e["meta"]["step"]
            pu, po = M.v6_history_arrays(hist, t, U, O)
            self.assertTrue(bool((pu[0].numpy() == e["prev_unit"]).all()), t)
            self.assertTrue(bool((po[0].numpy() == e["prev_order"]).all()), t)
            hist = (hist + ((t, list(e["unit_tok"]), list(e["unit_qty"]), list(e["order_tok"]), list(e["order_qty"])),))[-2:]
            checked += 1
        self.assertEqual(checked, len(ex))
        self.assertGreater(checked, 700)


@unittest.skipUnless(PARENT.exists() and PRIO.exists(), "parent checkpoint or priority rows not available")
class V6PlayInert(unittest.TestCase):
    """act() of the grafted child equals the parent's action on consecutive recorded observations, and keeps its last two
    decisions."""

    def test_act(self):
        torch.set_num_threads(1)
        blob = torch.load(str(PARENT), map_location="cpu", weights_only=True)
        pc = M.ModelConfig.from_dict(blob["config"])
        parent = M.PolicyNet(pc)
        parent.load_state_dict(blob["model"])
        child = M.PolicyNet(dataclasses.replace(pc, v6_prev=True, v6_aux=True))
        child.load_state_dict(blob["model"], strict=False)
        parent.eval()
        child.eval()
        row = [ln.rstrip("\n").split("\t") for ln in open(PRIO)][1]
        replay = json.loads(zipfile.ZipFile(row[0]).read(row[1]))
        seat, cfg = int(row[4]), replay["configuration"]
        sp = sc = None
        for t in range(20, 29):
            obs = replay["steps"][t][seat]["observation"]
            ap, sp = parent.act(obs, temperature=0.0, state=sp, configuration=cfg)
            ac, sc = child.act(obs, temperature=0.0, state=sc, configuration=cfg)
            self.assertEqual(ap, ac, t)
        self.assertEqual([h[0] for h in sc.hist], [27, 28])


if __name__ == "__main__":
    unittest.main()
