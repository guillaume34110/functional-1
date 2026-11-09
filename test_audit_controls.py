import unittest

import numpy as np

from audit_controls import compare_controls, direction_oracle, select_seed, transition_oracle
from test_object_annotations import screen


def board(cells, fixed=()):
    result = np.zeros((18, 10), dtype=np.int64)
    for y, x in cells:
        result[y, x] = 1
    for y, x in fixed:
        result[y, x] = 2
    return result


class ControlOracleTest(unittest.TestCase):
    def setUp(self):
        self.cells = np.array([[2, 3], [2, 4], [2, 5], [3, 4]])

    def test_gt_calibrated_horizontal_lag1_is_unknown_candidate_only(self):
        for current in ("noop", "down"):
            self.assertEqual(direction_oracle(self.cells, self.cells + [0, 1], current, 0, "right")[0], "possible-delayed-horizontal")
            self.assertEqual(direction_oracle(self.cells, self.cells - [0, 1], current, 0, "left")[0], "possible-delayed-horizontal")
            for previous, delta in ((None, [0, 1]), ("down", [0, 1]), ("left", [0, 1]), ("right", [0, 2])):
                self.assertEqual(direction_oracle(self.cells, self.cells + delta, current, 0, previous)[0], "uncommanded-horizontal-motion")
        rotated = np.array([[2, 4], [3, 3], [3, 4], [4, 4]])
        self.assertEqual(direction_oracle(self.cells, rotated, "down", 0, "right")[0], "uncommanded-orientation-change")

    def test_seed_requires_observed_release_before_branch(self):
        frames = np.stack([screen(self.cells)] * 35)
        labels = {"shape_id": np.zeros(35, dtype=np.int64), "evidence": np.ones(35, dtype=np.uint8),
                  "cells": np.stack([self.cells] * 35), "texture": np.full(35, 6)}
        actions = np.zeros(34, dtype=np.uint8)
        self.assertEqual(select_seed(frames, labels, actions), 32)
        actions[31] = 4
        self.assertEqual(select_seed(frames, labels, actions), 33)
        actions[:] = 5
        self.assertIsNone(select_seed(frames, labels, actions))

    def test_direction_and_single_rotation_press_constraints(self):
        rotated = np.array([[2, 4], [3, 3], [3, 4], [4, 4]])
        for button in ("noop", "down", "left", "right"):
            self.assertEqual(direction_oracle(self.cells, rotated, button, 0)[0], "uncommanded-orientation-change")
        self.assertEqual(direction_oracle(self.cells, self.cells + [1, 1], "noop", 0)[0], "uncommanded-horizontal-motion")
        self.assertEqual(direction_oracle(self.cells, self.cells + [1, -1], "down", 0)[0], "uncommanded-horizontal-motion")
        self.assertEqual(direction_oracle(self.cells, self.cells + [0, 1], "left", 0)[0], "horizontal-motion-opposes-command")
        self.assertEqual(direction_oracle(self.cells, self.cells - [0, 1], "right", 0)[0], "horizontal-motion-opposes-command")
        self.assertIsNone(direction_oracle(self.cells, self.cells + [1, 0], "down", 0)[0])
        reason, changes = direction_oracle(self.cells, rotated, "a", 0)
        self.assertIsNone(reason); self.assertEqual(changes, 1)
        reason, changes = direction_oracle(rotated, self.cells, "a", changes)
        self.assertEqual(reason, "multiple-orientation-changes-after-one-press")
        self.assertEqual(changes, 2)

    def test_collision_and_unexplained_background_change_are_violations(self):
        new = self.cells + [1, 0]
        before = board(self.cells, [(4, 4)])
        after = board(new, [(4, 4)])
        after[new[:, 0], new[:, 1]] = 1  # Neural overlay hides the existing cell.
        result = transition_oracle(before, after, self.cells, new, 0)
        self.assertEqual(result["status"], "violation")
        self.assertTrue(result["collision"])
        before = board(self.cells, [(16, 8)])
        result = transition_oracle(before, board(new), self.cells, new, 0)
        self.assertEqual(result["reason"], "unexplained-background-mutation")

    def test_deposit_requires_visible_support(self):
        new = self.cells + [6, 0]
        before = board(self.cells)
        after = board(np.concatenate((self.cells, new)))
        self.assertEqual(transition_oracle(before, after, self.cells, new, 0)["reason"], "unsupported-deposit")
        before = board(self.cells, [(4, 4)])
        after = board(np.concatenate((self.cells, new)), [(4, 4)])
        self.assertEqual(transition_oracle(before, after, self.cells, new, 0)["status"], "deposit")

    def test_motion_rotation_and_ambiguous_endpoints(self):
        new = self.cells + [1, 0]
        self.assertEqual(transition_oracle(board(self.cells), board(new), self.cells, new, 0)["status"], "motion")
        rotated = np.array([[2, 4], [3, 3], [3, 4], [4, 4]])
        branches = {name: {"continuous": True, "endpoint": cells.tolist()} for name, cells in
                    (("noop", new), ("left", new - [0, 1]), ("right", new + [0, 1]),
                     ("down", new + [1, 0]), ("a", rotated), ("b", rotated))}
        result = compare_controls(branches, self.cells)
        self.assertTrue(all(value["status"] == "pass" for value in result.values()))
        branches["b"]["endpoint"] = self.cells.tolist()
        self.assertEqual(compare_controls(branches, self.cells)["b"]["status"], "fail")
        branches["left"]["continuous"] = False
        branches["a"]["endpoint"] = None
        result = compare_controls(branches, self.cells)
        self.assertEqual(result["horizontal"]["status"], "inconclusive")
        self.assertEqual(result["a"]["status"], "inconclusive")
        missing = transition_oracle(board(self.cells), board(new), self.cells, None, 0)
        self.assertEqual(missing["status"], "unknown")
        flash = transition_oracle(board(self.cells), board(new), self.cells, new, 0, invalid=True)
        self.assertEqual(flash["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
