import unittest
from unittest.mock import patch

import torch

from audit_mode_prior import ModePrior, mode_trace, prior_offset
from scene_model import ObjectSceneDynamics
from test_scene_model import assets
from token_model import decode_tokens


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = torch.nn.Parameter(torch.tensor([[1., 2., 3.]]))
        self.details = {"mode": self.mode, "state": torch.ones(1, 20, 18, 20),
                        "write": torch.ones(1, 2, 18, 20), "gate": torch.zeros(1, 1, 18, 20),
                        "current": {"joint": torch.zeros(1, 2, 180)}, "next": {"joint": torch.ones(1, 2, 180)}}

    def scene_details(self, tokens, actions, state=None):
        return self.details


class ModePriorTest(unittest.TestCase):
    @torch.inference_mode()
    def test_all_prediction_entry_points_use_offset_and_exact_same_recurrent_state(self):
        torch.set_num_threads(1)
        core = ObjectSceneDynamics(*assets()).eval()
        weights = {key: value.clone() for key, value in core.state_dict().items()}
        tokens, actions = torch.zeros(1, 18, 20, dtype=torch.long), torch.tensor([0])
        baseline = core.scene_details(tokens, actions)
        offset = torch.tensor([0., .5, 1.])
        view = ModePrior(core, offset).eval()
        expected = baseline["mode"] - offset
        frame = decode_tokens(tokens, core.codebook)
        with patch.object(core, "decode_scene", wraps=core.decode_scene) as decoder:
            _, forward_state = view(tokens, actions)
            _, tokens_state = view.predict_tokens(tokens, actions)
            _, frame_state = view.predict_frame(frame, actions)
            self.assertEqual(decoder.call_count, 3)
            for call in decoder.call_args_list:
                torch.testing.assert_close(call.args[1]["mode"], expected)
        for state in (forward_state, tokens_state, frame_state):
            torch.testing.assert_close(state, baseline["state"], rtol=0, atol=0)
        torch.testing.assert_close(view.advance(tokens, actions), baseline["state"], rtol=0, atol=0)
        for key, value in core.state_dict().items(): torch.testing.assert_close(value, weights[key], rtol=0, atol=0)

    def test_fixed_formula_zero_and_half_strength(self):
        weights = {"parent": [1., 3., 5.], "interventions": [1., 5., 3.]}
        torch.testing.assert_close(prior_offset(weights, 0), torch.zeros(3, dtype=torch.float64))
        expected = torch.tensor([0., 4., 4.], dtype=torch.float64)
        expected[1:] = expected[1:].log()
        torch.testing.assert_close(prior_offset(weights, 1), expected)
        torch.testing.assert_close(prior_offset(weights, .5), expected / 2)
        with self.assertRaises(ValueError): prior_offset(weights, .25)

    def test_wrapper_changes_only_mode_preserves_core_and_gradients(self):
        core = Core()
        weights = {name: value.detach().clone() for name, value in core.state_dict().items()}
        view = ModePrior(core, torch.tensor([0., .5, 1.])).eval()
        original = core.scene_details(None, None)
        result = view.scene_details(None, None)
        self.assertIsNot(result, original)
        torch.testing.assert_close(result["mode"], original["mode"] - torch.tensor([0., .5, 1.]))
        for key in ("state", "write", "gate", "current", "next"):
            self.assertIs(result[key], original[key])
        result["mode"].sum().backward()
        torch.testing.assert_close(core.mode.grad, torch.ones_like(core.mode))
        for key, value in core.state_dict().items(): torch.testing.assert_close(value, weights[key])
        zero = ModePrior(core, torch.zeros(3)).scene_details(None, None)
        torch.testing.assert_close(zero["mode"], original["mode"], rtol=0, atol=0)
        self.assertIs(zero["state"], original["state"])

    def test_rewrite_run_counts_consecutive_events_only(self):
        result = mode_trace([{"mode": mode} for mode in (2, 2, 0, 2, 2, 2, 1)])
        self.assertEqual(result["longest_rewrite_run"], 3)
        self.assertEqual(result["mode_counts"], {"copy": 1, "compose": 1, "rewrite": 5})


if __name__ == "__main__":
    unittest.main()
