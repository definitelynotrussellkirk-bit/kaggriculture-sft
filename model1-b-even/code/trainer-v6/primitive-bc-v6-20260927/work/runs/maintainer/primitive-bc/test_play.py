"""play.py: engine no-op audit labels, cash fidelity against the real engine, game records."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import agent  # noqa: E402
import play  # noqa: E402
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment  # noqa: E402


class AuditTests(unittest.TestCase):
    def test_step0_labels(self):
        env = ExactKaggricultureEnvironment(seed=1)
        ours = {"farmer": ["NORTH"],
                "hands": [["WATER"]],                                # no hand exists yet
                "market": [["SELL", "MILK", 1], ["NOOP"], ["HIRE"], ["BUY_SEED", "WHEAT", 2],
                           ["BUY_LAND"], ["BUY_ANIMAL", "COW", 100]]}
        rival = {"farmer": ["WATER"], "hands": [], "market": []}   # empty tile: ignored
        audits, cash = play.audit_step(env, [ours, rival])
        self.assertEqual(audits[0]["units"], [("NORTH", "effective"), ("WATER", "no_such_unit")])
        self.assertEqual(audits[1]["units"], [("WATER", "ignored")])
        self.assertEqual([o[:2] for o in audits[0]["orders"]],
                         [("SELL", "ignored"), ("NOOP", "unparsed"), ("HIRE", "effective"),
                          ("BUY_SEED", "effective"), ("BUY_LAND", "effective"), ("BUY_ANIMAL", "partial")])
        env.step([ours, rival])
        real = [float(env.state[0].observation.farms[p]["money"]) for p in range(2)]
        self.assertEqual(cash, real)

    def test_plant_seed_block(self):
        env = ExactKaggricultureEnvironment(seed=1)
        audits, _ = play.audit_step(env, [{"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []},
                                          play_pass()])
        self.assertEqual(audits[0]["units"], [("PLANT", "plant_seed_block")])

    def test_percentile(self):
        self.assertEqual(play.percentile([3, 1, 2], 0.5), 2)
        self.assertAlmostEqual(play.percentile([0, 10], 0.95), 9.5)
        self.assertIsNone(play.percentile([], 0.5))


def play_pass():
    return {"farmer": ["PASS"], "hands": [], "market": []}


class GameTests(unittest.TestCase):
    def test_short_game_vs_k0006_fidelity(self):
        actor = agent.make_agent("stub:2", log=False)
        game = play.play_game(actor, "K0006", our_seat=1, world_seed=77, sampler_seed=5, steps=96)
        self.assertEqual(game["steps"], 96)
        self.assertEqual(game["audit_cash_mismatch_steps"], 0)
        self.assertEqual(game["fallbacks"], 0)
        self.assertEqual(len(game["step_ms"]), 96)
        audit = game["noop_audit_ours"]
        self.assertEqual(audit["steps"], 96)
        self.assertGreater(audit["unit_noops"], 0)
        self.assertEqual(game["noop_audit_rival"]["steps"], 96)
        summary = play.summarize([game])
        self.assertEqual(summary["cells"]["K0006/seat1"]["games"], 1)
        self.assertEqual(summary["latency_ms_after_first_pooled"]["n"], 95)


if __name__ == "__main__":
    unittest.main()
