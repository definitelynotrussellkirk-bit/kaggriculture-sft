"""agent.py: sanitizer, PASS fallback, per-seat field carry, lazy load, weights round trip."""
from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import agent  # noqa: E402
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment  # noqa: E402


def _model_available():
    try:
        import model  # noqa: F401
        import codec  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


class Recorder:
    """Policy double: records the state it is handed, returns (PASS, state + 1)."""

    def __init__(self):
        self.seen = []

    def act(self, observation, generator=None, temperature=1.0, state=None, configuration=None):
        self.seen.append((observation["player"], observation["step"], state, sorted(configuration)))
        return agent.pass_action(), (state or 0) + 1


class Exploding:
    def act(self, observation, generator=None, temperature=1.0, state=None):
        raise RuntimeError("boom")


class SanitizeTests(unittest.TestCase):
    def test_numpy_ints_and_tuples_become_plain_json(self):
        import numpy as np
        out = agent.sanitize_action({"farmer": ("PICKUP", "WOOL", np.int64(3)), "hands": [("PASS",)],
                                     "market": [["SELL", "MILK", np.int32(2)], ("HIRE",)], "extra": 1})
        self.assertEqual(out, {"farmer": ["PICKUP", "WOOL", 3], "hands": [["PASS"]],
                               "market": [["SELL", "MILK", 2], ["HIRE"]]})
        self.assertIs(type(out["farmer"][2]), int)

    def test_engine_ignored_order_slot_is_kept(self):
        # The engine skips an unparsable order but it still occupies its lockstep index.
        out = agent.sanitize_action({"farmer": ["PASS"], "hands": [], "market": [["NOOP"], ["BUY_LAND"]]})
        self.assertEqual(out["market"], [["NOOP"], ["BUY_LAND"]])

    def test_crashing_shapes_are_rejected(self):
        bad = [
            {"farmer": ["PLANT", ["WHEAT"]], "hands": [], "market": []},     # unhashable crop
            {"farmer": ["PICKUP", "WOOL", "x"], "hands": [], "market": []},  # engine int() crash
            {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 1.5]]},
            {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]] * 11},
            {"farmer": ["PASS"], "hands": "PASS", "market": []},
            ["PASS"],
        ]
        for action in bad:
            with self.assertRaises((TypeError, ValueError), msg=repr(action)):
                agent.sanitize_action(action)


class AgentTests(unittest.TestCase):
    def test_pass_fallback_and_log_line(self):
        stream = io.StringIO()
        actor = agent.Agent("preloaded", policy=Exploding(), log=True, stream=stream)
        env = ExactKaggricultureEnvironment(seed=3)
        action = actor(env.agent_observation(0), env.configuration)
        self.assertEqual(action, agent.PASS_ACTION)
        self.assertEqual((actor.fallbacks, actor.last_fallback), (1, True))
        self.assertIn("RuntimeError: boom", actor.errors[0])
        line = stream.getvalue().strip()
        self.assertTrue(line.startswith("[pbc] step=0 seat=0 ms="), line)
        self.assertIn("fb=1", line)
        self.assertEqual(len(stream.getvalue().splitlines()), 1)

    def test_failed_load_is_attempted_once(self):
        actor = agent.make_agent("/nonexistent/policy.pt", log=False)
        env = ExactKaggricultureEnvironment(seed=3)
        for _ in range(3):
            self.assertEqual(actor(env.agent_observation(0), env.configuration), agent.PASS_ACTION)
        self.assertEqual(actor.fallbacks, 3)
        self.assertIn("FileNotFoundError", actor.load_error)
        self.assertIn("policy load failed earlier", actor.errors[-1])

    def test_state_is_carried_per_seat_and_reset_per_episode(self):
        policy = Recorder()
        actor = agent.Agent("preloaded", policy=policy, log=False)
        for episode in range(2):
            env = ExactKaggricultureEnvironment(seed=10 + episode)
            for _ in range(3):
                env.step([actor(env.agent_observation(p), env.configuration) for p in range(2)])
        states = [(seat, step, state) for seat, step, state, _ in policy.seen]
        expected = []
        for _episode in range(2):
            for step in range(3):
                expected += [(0, step, None if step == 0 else step), (1, step, None if step == 0 else step)]
        self.assertEqual(states, expected)
        self.assertEqual(policy.seen[0][3], sorted(["episodeSteps", "boardSize", "turnsPerDay", "shedCapacity",
                                                     "maxMarketOrdersPerTurn", "farmHandCostMult",
                                                     "weedSpawnChance", "townShopUnlockInterval",
                                                     "townShopSellInterval", "townCenterSellInterval",
                                                     "marketParams"]))
        self.assertEqual(actor.fallbacks, 0)

    def test_stub_plays_valid_actions(self):
        actor = agent.make_agent("stub:4", log=False)
        env = ExactKaggricultureEnvironment(seed=4)
        for _ in range(48):
            env.step([actor(env.agent_observation(p), env.configuration) for p in range(2)])
        self.assertEqual(actor.fallbacks, 0)
        self.assertEqual(actor.calls, 96)

    def test_lazy_load(self):
        actor = agent.make_agent("stub:1", log=False)
        self.assertIsNone(actor.policy)
        self.assertIsNone(actor.load_ms)


@unittest.skipUnless(_model_available(), "model.py / codec.py not importable")
class PolicyNetTests(unittest.TestCase):
    def _tiny(self):
        import torch
        import model
        torch.manual_seed(0)
        config = model.default_config()
        for name, value in (("width", 32), ("layers", 1), ("heads", 2), ("dec_layers", 1), ("G", 8), ("K", 8)):
            if hasattr(config, name):
                setattr(config, name, value)
        return model.PolicyNet(config).eval()

    def test_weights_round_trip_and_act(self):
        import torch
        net = self._tiny()
        env = ExactKaggricultureEnvironment(seed=6)
        with tempfile.TemporaryDirectory() as tmp:
            for storage, tol in (("fp32", 0.0), ("fp16", 1e-3), ("int8", 5e-2)):
                path = Path(tmp) / f"{storage}.pt"
                torch.save(agent.export_weights(net, storage=storage, int8_min_numel=64), path)
                loaded = agent.load_policy(str(path))
                for (name, a), b in zip(net.state_dict().items(), loaded.state_dict().values()):
                    if a.is_floating_point():
                        self.assertLessEqual(float((a - b).abs().max()) if a.numel() else 0.0,
                                             tol * max(1.0, float(a.abs().max())), f"{storage} {name}")
                    else:
                        self.assertTrue(torch.equal(a, b), name)
                actor = agent.Agent("preloaded", policy=loaded, log=False)
                for _ in range(3):
                    env_actions = [actor(env.agent_observation(p), env.configuration) for p in range(2)]
                    env.step(env_actions)
                self.assertEqual(actor.fallbacks, 0, actor.errors)


if __name__ == "__main__":
    unittest.main()
