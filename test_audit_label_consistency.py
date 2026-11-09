import unittest

import numpy as np
import torch

from audit_label_consistency import analyze
from audit_label_consistency import add_delta_counts, delta_evidence, delta_rates
from audit_label_consistency import input_history, latency_evidence, paired_latency
from collections import Counter
from scene_model import ObjectSceneDynamics
from test_scene_model import assets


class LabelConsistencyTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = ObjectSceneDynamics(*assets())
        self.old = np.array([[2, 2], [2, 3], [3, 2], [3, 3]])

    def test_actual_history_parent_and_branch_anchor_predecessor(self):
        parent = np.array([4, 1, 2, 0, 3, 1, 1, 2, 0, 5], dtype=np.uint8)
        self.assertEqual(input_history(parent, 0), [])
        self.assertEqual(input_history(parent, 9), [1, 2, 0, 3, 1, 1, 2, 0])
        branch = np.array([3, 3, 3, 3], dtype=np.uint8)
        self.assertEqual(input_history(parent, 0, 8, branch), parent[:8].tolist())
        self.assertEqual(input_history(parent, 1, 8, branch)[-2:], [2, 3])
        self.assertEqual(input_history(parent, 2, 8, branch)[-3:], [2, 3, 3])
        with self.assertRaises(ValueError): input_history(parent, -1)
        with self.assertRaises(ValueError): input_history(parent, 0, 0, branch)

    def test_horizontal_lag_direction_and_pair_evidence(self):
        delta = {"category": "continuous_visual", "delta_origin_yx": [0, 1], "shape_equal": True,
                 "old_supported": False, "next_supported": False}
        one = latency_evidence(delta, 0, [0, 2])
        self.assertEqual(one["last_horizontal_lag"], 1)
        self.assertTrue(one["same_sense"])
        nine = latency_evidence(delta, 3, [2] + [0] * 8)
        self.assertIsNone(nine["last_horizontal_lag"])
        self.assertIsNone(nine["same_sense"])
        opposite = latency_evidence(delta, 0, [1])
        self.assertFalse(opposite["same_sense"])
        self.assertIsNone(latency_evidence(delta, 1, [2]))
        self.assertEqual(paired_latency(one, {**one, "history8": [3, 2]}), "equal_signed_dx")
        self.assertEqual(paired_latency(one, {**one, "signed_dx": -1}), "mismatched_signed_dx")
        self.assertEqual(paired_latency(one, None), "unprovable")

    def test_delta_local_bounds_rotation_bbox_and_masked_cases(self):
        old = np.array([[6, 3], [6, 4], [6, 5], [7, 4]])
        def board(cells):
            value = np.zeros((18, 10), dtype=np.int64)
            value[cells[:, 0], cells[:, 1]] = 1
            return value
        for delta, expected_local in (([0, 0], True), ([3, 3], True), ([-3, -3], True), ([4, 0], False)):
            new = old + delta
            result = delta_evidence(board(old), board(new), old, new, 1, 1, 0, True)
            self.assertEqual(result["category"], "continuous_visual")
            self.assertEqual(result["delta_origin_yx"], delta)
            self.assertEqual(result["local3"], expected_local)
            self.assertEqual(result["all_translation_candidates_invalid"], False)
        rotated = np.array([[5, 4], [6, 3], [6, 4], [7, 4]])
        result = delta_evidence(board(old), board(rotated), old, rotated, 1, 1, 0, True)
        self.assertEqual(result["delta_origin_yx"], [-1, 0])
        self.assertFalse(result["shape_equal"])
        self.assertEqual(result["delta_centre_quarter_cells_yx"], [-1, -1])
        new = old + [5, 0]
        deposited = board(old); deposited[new[:, 0], new[:, 1]] = 1
        self.assertEqual(delta_evidence(board(old), deposited, old, new, 1, 1, 0, True)["category"], "deposit")
        self.assertEqual(delta_evidence(board(old), board(new), old, new, 1, 1, 0, False), {"category": "unknown"})

    def test_delta_coverage_has_conditional_and_total_denominators(self):
        counts = Counter()
        old = np.array([[6, 3], [6, 4], [6, 5], [7, 4]])
        value = np.zeros((18, 10), dtype=np.int64); value[old[:, 0], old[:, 1]] = 1
        result = delta_evidence(value, value, old, old, 1, 1, 0, True)
        add_delta_counts(counts, result, 4)
        add_delta_counts(counts, {"category": "unknown"}, 4)
        rates = delta_rates(counts)["action4"]
        self.assertEqual(rates["local3_given_continuous"], 1.)
        self.assertEqual(rates["local3_of_all_transitions"], .5)
        self.assertEqual(rates["hist49"][3][3], 1)

    def pair(self, new, deposit=False, support=False):
        current = torch.zeros(18, 20, dtype=torch.uint8)
        current[self.old[:, 0], self.old[:, 1] + 2] = 1
        if support: current[4, 4] = 2
        target = current.clone()
        if not deposit: target[self.old[:, 0], self.old[:, 1] + 2] = 0
        target[new[:, 0], new[:, 1] + 2] = 1
        labels = {"cells": np.stack((self.old, new)), "shape_id": np.array([0, 0]),
                  "evidence": np.ones(2, dtype=np.uint8), "track": np.array([0, 0])}
        return current, target, labels

    def test_correct_copy_deposit_and_rewrite(self):
        current, target, labels = self.pair(self.old + [1, 0])
        result, categories = analyze(self.model, current, target, labels)
        self.assertEqual(result["mode"], 0)
        self.assertTrue(result["copy_feasible"])
        self.assertNotIn("copy_target_not_renderable", categories)
        current, target, labels = self.pair(self.old + [6, 0], deposit=True, support=True)
        result, categories = analyze(self.model, current, target, labels)
        self.assertEqual(result["mode"], 1)
        self.assertTrue(result["old_supported"])
        self.assertTrue(result["compose_feasible"])
        current[16, 2:8] = 2
        result, _ = analyze(self.model, current, target, labels)
        self.assertEqual(result["mode"], 2)
        self.assertFalse(result["copy_feasible"])
        self.assertFalse(result["compose_feasible"])

    def test_texture_error_and_support_warning_are_descriptive(self):
        current, target, labels = self.pair(self.old + [6, 0], deposit=True)
        result, categories = analyze(self.model, current, target, labels)
        self.assertFalse(result["old_supported"])
        self.assertIn("compose_without_visible_support", categories)
        labels["shape_id"][1] = 1
        result, categories = analyze(self.model, current, target, labels)
        self.assertEqual(result["label_texture_errors"], 1)
        self.assertIn("label_texture_mismatch", categories)

    def test_static_label_and_other_motion_is_ambiguous_not_internal_truth(self):
        current, target, labels = self.pair(self.old)
        moving0 = np.array([[8, 5], [8, 6], [9, 5], [9, 6]])
        moving1 = moving0 + [1, 0]
        current[moving0[:, 0], moving0[:, 1] + 2] = 2
        target[moving1[:, 0], moving1[:, 1] + 2] = 2
        # Synthetic codebook fixtures aren't game textures, so only the track warning is asserted.
        result, categories = analyze(self.model, current, target, labels)
        self.assertTrue(result["same_track_background_changed"])
        self.assertIn("same_track_background_changed", categories)


if __name__ == "__main__": unittest.main()
