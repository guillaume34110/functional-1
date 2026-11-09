import unittest

import numpy as np
import torch

from probe_conditional_pose import (ConditionalPoseView, batch_joint, cache_bundle, cache_loss,
                                    conditional_joint, future_logits, residual_head)
from scene_model import ObjectSceneDynamics
from test_scene_model import assets
from test_intervention_training import fixture
from token_model import decode_tokens


class ConditionalPoseProbeTest(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_action_filter_selection_and_only_selected_filters_receive_gradients(self):
        residual = residual_head(19, action_conditioned=True)
        self.assertEqual(sum(parameter.numel() for parameter in residual.parameters()), 2584)
        hidden = torch.ones(2, 16, 18, 10)
        with torch.no_grad():
            residual.weight[4] = 1
            residual.bias[5] = 2
        result = residual(hidden, torch.tensor([4, 5]))
        self.assertTrue(torch.all(result[0] == 16))
        self.assertTrue(torch.all(result[1] == 2))
        result.sum().backward()
        for action in range(8):
            self.assertEqual(bool(residual.weight.grad[action].abs().sum() > 0), action in (4, 5))
            self.assertEqual(bool(residual.bias.grad[action].abs().sum() > 0), action in (4, 5))

    def test_delayed_a_b_effect_cache_uses_actual_noop_not_nominal_branch(self):
        base, bundle = fixture(); base.eval()
        # Delay visual effect until transition1; A/B impulses were at transition0.
        for arm in (4, 5):
            bundle["tokens"][arm, 1] = bundle["tokens"][arm, 0]
            for key, values in bundle["labels"].items(): values[arm, 1] = values[arm, 0]
        cache = cache_bundle(base, residual_head(2, True), bundle, None)
        np.testing.assert_array_equal(cache["effect_step"][4:], [1, 1])
        np.testing.assert_array_equal(cache["button"][4:], [4, 5])
        np.testing.assert_array_equal(cache["actual_action"][4:], [0, 0])
        residual = residual_head(2, True)
        loss = cache_loss(residual, cache, base.valid_poses, np.array([4, 5]))
        loss.backward()
        self.assertGreater(float(residual.weight.grad[0].abs().sum()), 0.)
        self.assertTrue(torch.all(residual.weight.grad[4:6] == 0))

    def test_action_conditioned_zero_parity_all_entry_points_and_actual_runtime_action(self):
        base = ObjectSceneDynamics(*assets()).eval()
        residual = residual_head(2, True); view = ConditionalPoseView(base, residual)
        tokens, actions = torch.zeros(1, 18, 20, dtype=torch.long), torch.tensor([5])
        original = {key: value.clone() for key, value in base.state_dict().items()}
        expected, state = base(tokens, actions)
        actual, next_state = view(tokens, actions)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(next_state, state, rtol=0, atol=0)
        self.assertTrue(torch.equal(base.predict_tokens(tokens, actions)[0], view.predict_tokens(tokens, actions)[0]))
        pixels = decode_tokens(tokens, base.codebook)
        self.assertTrue(torch.equal(base.predict_frame(pixels, actions)[0], view.predict_frame(pixels, actions)[0]))
        with torch.no_grad(): residual.bias[5, 1] = 3
        changed = view.scene_details(tokens, actions)
        unchanged = view.scene_details(tokens, torch.tensor([0]))
        self.assertFalse(torch.equal(changed["next"]["joint"], base.scene_details(tokens, actions)["next"]["joint"]))
        self.assertTrue(torch.equal(unchanged["next"]["joint"], base.scene_details(tokens, torch.tensor([0]))["next"]["joint"]))
        for key, value in base.state_dict().items(): self.assertTrue(torch.equal(value, original[key]))

    def test_zero_all_api_codec_state_geometry_and_core_exact(self):
        base = ObjectSceneDynamics(*assets()).eval()
        residual = residual_head(len(base.object_shapes)); view = ConditionalPoseView(base, residual)
        self.assertEqual(sum(p.numel() for p in residual_head(19).parameters()), 323)
        original = {key: value.clone() for key, value in base.state_dict().items()}
        tokens, actions = torch.zeros(1, 18, 20, dtype=torch.long), torch.tensor([4])
        details = base.scene_details(tokens, actions)
        hidden, pose, shape = future_logits(base, details)
        joint = conditional_joint(hidden, pose, shape, residual, base.valid_poses)
        self.assertTrue(torch.equal(joint, details["next"]["joint"]))
        self.assertTrue(torch.isneginf(joint[0][~base.valid_poses]).all())
        for key in ("state", "mode", "write", "gate"):
            torch.testing.assert_close(view.scene_details(tokens, actions)[key], details[key], rtol=0, atol=0)
        expected, state = base(tokens, actions)
        actual, actual_state = view(tokens, actions)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_state, state, rtol=0, atol=0)
        self.assertTrue(torch.equal(base.predict_tokens(tokens, actions)[0], view.predict_tokens(tokens, actions)[0]))
        pixels = decode_tokens(tokens, base.codebook)
        self.assertTrue(torch.equal(base.predict_frame(pixels, actions)[0], view.predict_frame(pixels, actions)[0]))
        for key, value in base.state_dict().items(): self.assertTrue(torch.equal(value, original[key]))

    def test_cache_first_effect_per_arm_and_uniform_weighted6_objective(self):
        base, bundle = fixture(); base.eval()
        residual = residual_head(2)
        cache = cache_bundle(base, residual, bundle, None)
        self.assertEqual(cache["hidden"].shape, (6, 16, 18, 10))
        self.assertEqual(cache["pose"].shape, (6, 180))
        self.assertEqual(cache["shape"].shape, (6, 2))
        self.assertEqual(str(cache["hidden"].dtype), "float32")
        np.testing.assert_array_equal(cache["button"], np.arange(6))
        self.assertAlmostEqual(float(cache["weight"].sum()), 1., places=6)
        indices = np.arange(6)
        actual = cache_loss(residual, cache, base.valid_poses, indices)
        expected = -batch_joint(residual, cache, base.valid_poses, indices).flatten(1).gather(1, torch.from_numpy(cache["target"])[:, None]).mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in residual.parameters()))
        self.assertGreater(float(residual.weight.grad.abs().sum()), 0.)
        cache["weight"][:] = 0
        residual.zero_grad(set_to_none=True)
        zero = cache_loss(residual, cache, base.valid_poses, indices)
        self.assertEqual(float(zero), 0.)
        zero.backward()
        self.assertTrue(torch.all(residual.weight.grad == 0))


if __name__ == "__main__": unittest.main()
