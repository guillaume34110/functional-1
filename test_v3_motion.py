from pathlib import Path
import unittest

import numpy as np
import torch

from functional_two import FunctionalTwoSession
from model import ACTIONS
from play import RELEASE


class V3MotionTest(unittest.TestCase):
    def test_witnessed_rotations_and_lateral_taps(self):
        torch.set_num_threads(1)
        model = FunctionalTwoSession(RELEASE.with_name("functional-2"),
                                     RELEASE.with_name("functional-3") / "motion.pt").model
        path = Path(__file__).parent / "checks" / "fixtures" / "v3-motion.npz"
        with np.load(path, allow_pickle=False) as data, torch.inference_mode():
            frames = torch.from_numpy(data["frames"][:, :, None])
            actions = torch.from_numpy(data["actions"]).long()
            state = None
            for t in range(actions.shape[1]):
                state = model.advance_observed_frame(frames[:, t], actions[:, t], state)
            prediction, _ = model.predict_frame(frames[:, -1], torch.from_numpy(data["buttons"]).long(), state)
            for index, case in enumerate(data["cases"]):
                with self.subTest(case=case, seed=int(data["seeds"][index]), anchor=int(data["anchors"][index])):
                    np.testing.assert_array_equal(prediction[index, 0, :, 8:48].numpy(), data["expected"][index])

    def test_terminal_state_cannot_start_the_line_clear_animation(self):
        torch.set_num_threads(1)
        game = FunctionalTwoSession(RELEASE.with_name("functional-2"))
        game.step("start")
        game.step("noop")
        model = game.model
        model.animation_start_logit = lambda tokens, details: tokens.new_ones(len(tokens)).float()
        with torch.inference_mode():
            memory = game.state.clone()
            memory[:, model.state_channels, 0, 0] = -1
            frame, _ = model.predict_frame(game.frame, torch.tensor([ACTIONS.index("noop")]), memory)
        self.assertEqual(frame.shape, game.frame.shape)
