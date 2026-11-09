import unittest
import copy

import torch

from probe_identity_prior import (IdentityGate, IdentityView, cache_bundle, current_priors,
                                  head_inputs, reconstruct_joint, target_weights, verify_seed_prefixes)
from intervention_training import BUTTONS
from scene_model import ObjectSceneDynamics
from test_scene_model import assets
from test_intervention_training import fixture
from token_model import decode_tokens


class IdentityProbeTest(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_seed_proof_requires_all_buttons_and_five_distinct_pairs(self):
        seeds = [{"file": f"validation/{index}.npz", "frame": 32,
                  "reference_verified_frames": dict.fromkeys(BUTTONS, 33)} for index in range(5)]
        verify_seed_prefixes(seeds)
        for proof in ({}, {key: 33 for key in BUTTONS[:-1]}, dict.fromkeys(BUTTONS, 32)):
            bad = copy.deepcopy(seeds); bad[0]["reference_verified_frames"] = proof
            with self.assertRaises(ValueError): verify_seed_prefixes(bad)
        duplicate = copy.deepcopy(seeds); duplicate[-1] = copy.deepcopy(duplicate[0])
        with self.assertRaises(ValueError): verify_seed_prefixes(duplicate)

    def test_zero_gate_all_entry_points_and_compact_joint_reconstruction_are_exact(self):
        model = ObjectSceneDynamics(*assets()).eval()
        gate = IdentityGate()
        self.assertEqual(sum(p.numel() for p in gate.parameters()), 75)
        tokens = torch.zeros(1, 18, 20, dtype=torch.long)
        action = torch.tensor([4])
        original = {key: value.clone() for key, value in model.state_dict().items()}
        details = model.scene_details(tokens, action)
        pose, shape, features, priors = head_inputs(model, details, action)
        rebuilt = reconstruct_joint(pose, shape, features, priors, gate, model.valid_poses)
        self.assertTrue(torch.equal(rebuilt, details["next"]["joint"]))
        self.assertTrue(torch.isfinite(priors).all())
        self.assertFalse(priors.requires_grad)
        self.assertTrue(torch.isneginf(rebuilt[0][~model.valid_poses]).all())
        view = IdentityView(model, gate)
        for key in ("state", "mode", "write", "gate"):
            torch.testing.assert_close(view.scene_details(tokens, action)[key], details[key], rtol=0, atol=0)
        base_logits, base_state = model(tokens, action)
        logits, state = view(tokens, action)
        torch.testing.assert_close(logits, base_logits, rtol=0, atol=0)
        torch.testing.assert_close(state, base_state, rtol=0, atol=0)
        expected, _ = model.predict_tokens(tokens, action)
        actual, _ = view.predict_tokens(tokens, action)
        self.assertTrue(torch.equal(actual, expected))
        image = decode_tokens(tokens, model.codebook)
        expected, _ = model.predict_frame(image, action)
        actual, _ = view.predict_frame(image, action)
        self.assertTrue(torch.equal(actual, expected))
        for key, value in model.state_dict().items(): self.assertTrue(torch.equal(value, original[key]))

    def test_only_gate_receives_gradient_and_invalid_geometry_remains_masked(self):
        model = ObjectSceneDynamics(*assets()).eval()
        gate = IdentityGate()
        tokens = torch.randint(0, 3, (1, 18, 20))
        details = IdentityView(model, gate).scene_details(tokens, torch.tensor([1]))
        loss = -details["next"]["joint"][0, 0, 23]
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in gate.parameters()))
        self.assertGreater(float(gate.linear.weight.grad.abs().sum()), 0.)
        self.assertTrue(torch.isneginf(details["next"]["joint"][0][~model.valid_poses]).all())

    def test_cache_covers_all_steps_compact_float32_with_exact_target_weighting(self):
        model, bundle = fixture()
        model.eval(); zero = IdentityGate()
        cache, qualified = cache_bundle(model, zero, bundle, None)
        self.assertEqual(qualified, 6)
        self.assertEqual(cache["pose"].shape, (48, 180))
        self.assertEqual(cache["shape"].shape, (48, 2))
        self.assertEqual(cache["features"].shape, (48, 24))
        self.assertEqual(cache["priors"].shape, (48, 30))
        self.assertEqual(str(cache["pose"].dtype), "float32")
        self.assertAlmostEqual(float(cache["weight"].sum()), 1.1, places=5)
        # Time-major cache: all six first effects at t0; remaining teacher targets still retained.
        self.assertTrue((cache["weight"][:6] > 1 / 6).all())
        self.assertTrue((cache["weight"][6:] > 0).all())
        with torch.no_grad():
            frames = torch.from_numpy(bundle["tokens"]).long()
            labels = {key: torch.from_numpy(value).long() for key, value in bundle["labels"].items()}
            labels["evidence"][:, 2] = 0
            targets, weights, _ = target_weights(model, frames, labels)
            self.assertTrue(torch.all(weights[:, 1] == 0))
            self.assertTrue(torch.all(targets[:, 1] == 0))


if __name__ == "__main__": unittest.main()
