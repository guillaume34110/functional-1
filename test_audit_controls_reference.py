import unittest
from unittest.mock import patch

import numpy as np

from audit_controls_reference import (ReplayMismatch, conditional_controls,
                                      reference_branch, replay_prefix)


class FakeGame:
    def __init__(self):
        self.closed = False
        self.ticks = []

    def set_emulation_speed(self, speed):
        self.speed = speed

    def tick(self, frames, render):
        self.ticks.append((frames, render))

    def stop(self, save):
        self.closed = not save


class ReferenceControlsTest(unittest.TestCase):
    def replay_fixture(self, wrong=False):
        seed = 123
        rng = np.random.default_rng(seed)
        boot, warmup = int(rng.integers(240, 1000)), int(rng.integers(24, 90))
        metadata = {"seed": seed, "profile": "heuristic", "boot_frames": boot, "warmup_steps": warmup}
        zero = np.zeros((72, 80), dtype=np.uint8)
        one = np.ones_like(zero)
        expected = np.stack((zero, one))
        calls = []
        def mock_step(game, action):
            calls.append(action)
            if len(calls) <= warmup + 2:
                return zero.copy()
            return (zero if wrong else one).copy()
        class Controller:
            def __init__(self, supplied_seed, profile):
                self.seed, self.profile = supplied_seed, profile
            def choose(self, frame):
                return "noop"
        return expected, metadata, mock_step, Controller, calls

    def test_exact_prefix_reconstruction_and_mismatch_refusal(self):
        frames, metadata, mock_step, controller, calls = self.replay_fixture()
        game = FakeGame()
        evidence = replay_prefix(game, frames, np.array([1]), metadata, 1,
                                 mock_step, lambda game: {"level": 1}, controller)
        self.assertEqual(evidence["verified_frames"], 2)
        self.assertEqual(calls[0:2], ["a", "noop"])
        self.assertEqual(calls[-1], "left")
        self.assertEqual(game.speed, 0)
        self.assertEqual(game.ticks, [(metadata["boot_frames"], True)])
        frames, metadata, mock_step, controller, _ = self.replay_fixture(wrong=True)
        with self.assertRaises(ReplayMismatch) as caught:
            replay_prefix(FakeGame(), frames, np.array([1]), metadata, 1,
                          mock_step, lambda game: {"level": 1}, controller)
        self.assertEqual(caught.exception.evidence["frame"], 1)
        self.assertEqual(caught.exception.evidence["different_pixels"], 72 * 80)

    def test_failed_replay_always_closes_reference_emulator_without_save(self):
        game = FakeGame()
        image = np.zeros((72, 80), dtype=np.uint8)
        with patch("audit_controls_reference.replay_prefix", side_effect=ReplayMismatch(0, image, image + 1)):
            with self.assertRaises(ReplayMismatch):
                reference_branch(lambda: game, None, None, None, 32, "a", 8)
        self.assertTrue(game.closed)

    def test_reference_nonpass_cannot_become_neural_failure(self):
        reference = {"a": {"status": "fail"}, "b": {"status": "inconclusive"},
                     "horizontal": {"status": "pass"}, "down": {"status": "pass"}}
        neural = {name: {"status": "fail"} for name in reference}
        neural["down"]["status"] = "inconclusive"
        result = conditional_controls(reference, neural)
        self.assertEqual(result["a"]["status"], "inconclusive")
        self.assertEqual(result["b"]["status"], "inconclusive")
        self.assertEqual(result["horizontal"]["status"], "fail")
        self.assertEqual(result["down"]["status"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
