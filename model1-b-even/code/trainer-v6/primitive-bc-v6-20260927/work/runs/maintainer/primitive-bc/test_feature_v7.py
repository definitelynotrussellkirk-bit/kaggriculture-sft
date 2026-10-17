"""Opt-in feature-v7 boundary checks; legacy checkpoint inputs stay unchanged."""

from __future__ import annotations

import copy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import data
import model
import train
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
from farm_neat.training.bc import actor_observation


class FeatureV7BoundaryTests(unittest.TestCase):
    def test_legacy_import_does_not_load_v7_extractors(self):
        script = ("import data,sys; "
                  "assert not any(x in sys.modules for x in "
                  "('flip_v7','job_eta_v7','calc_features_v7','theorem_features_v7'))")
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                cwd=__file__.rsplit("/", 1)[0], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_legacy_is_default_and_v7_is_explicit(self):
        self.assertEqual(model.ModelConfig().feature_schema, "legacy")
        self.assertEqual(model.ModelConfig(feature_schema="v7").resolve().feature_schema, "v7")
        with self.assertRaises(ValueError):
            model.ModelConfig(feature_schema="unknown").resolve()

    def test_legacy_features_are_exactly_the_default(self):
        env = ExactKaggricultureEnvironment(seed=7)
        obs = actor_observation(copy.deepcopy(env.agent_observation(0)))
        default = data.features(copy.deepcopy(obs), env.configuration)
        declared = data.features(copy.deepcopy(obs), env.configuration, feature_schema="legacy")
        self.assertEqual(default.keys(), declared.keys())
        for key in default:
            self.assertTrue(np.array_equal(default[key], declared[key]), key)

    def test_v7_canonical_view_does_not_mutate_prefix_observation(self):
        env = ExactKaggricultureEnvironment(seed=7)
        obs = actor_observation(copy.deepcopy(env.agent_observation(0)))
        obs["private"]["inventories"][0] = {"WHEAT": 1, "MILK": 1}
        data.features(obs, env.configuration, feature_schema="v7")
        self.assertEqual(list(obs["private"]["inventories"][0]), ["WHEAT", "MILK"])

    def test_model_reads_the_selected_v7_features(self):
        env = ExactKaggricultureEnvironment(seed=7)
        obs = actor_observation(copy.deepcopy(env.agent_observation(0)))
        expected = data.features(copy.deepcopy(obs), env.configuration, feature_schema="v7")
        net = model.PolicyNet(model.ModelConfig(width=32, layers=1, heads=4, dec_layers=1,
                                                G=2, K=4, C=1, extra_width=data.flip.LIN,
                                                feature_schema="v7"))
        rows, mask, known, pos = net.observation_tensors(obs, env.configuration)
        self.assertTrue(np.array_equal(rows[0].numpy(), expected["rows"].astype("float32")))
        self.assertTrue(np.array_equal(mask[0].numpy(), expected["token_mask"]))
        self.assertTrue(np.array_equal(known[0].numpy(), expected["knownness"]))
        self.assertTrue(np.array_equal(pos[0].numpy(), expected["unit_pos"]))
        self.assertTrue(np.array_equal(net._last_extra[0][0].numpy(), expected["extra"].astype("float32")))

    def test_v7_cannot_silently_fall_back_when_extractor_is_missing(self):
        env = ExactKaggricultureEnvironment(seed=7)
        obs = actor_observation(copy.deepcopy(env.agent_observation(0)))
        net = model.PolicyNet(model.ModelConfig(width=32, layers=1, heads=4, dec_layers=1,
                                                G=2, K=4, C=1, feature_schema="v7"))
        with patch.dict(sys.modules, {"data": None}):
            with self.assertRaisesRegex(RuntimeError, "required actor-feature extractor"):
                net.observation_tensors(obs, env.configuration)

    def test_training_rejects_mismatched_feature_schema(self):
        cfg = train.TrainConfig(synthetic=True, updates=0, eval_every=0, workers=0,
                                feature_schema="legacy")
        net_cfg = model.ModelConfig(width=32, layers=1, heads=4, dec_layers=1,
                                    G=2, K=4, C=1, feature_schema="v7")
        with tempfile.TemporaryDirectory(prefix="pbc-v7-guard-") as out:
            with self.assertRaisesRegex(ValueError, "feature_schema"):
                train.fit(cfg, net_cfg, out, device="cpu", stop_after=0)


if __name__ == "__main__":
    unittest.main()
