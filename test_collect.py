import unittest

import numpy as np

from collect import DIGITS, ScreenController, best_placement, normalize, occupancy, read_number, split_assignments
from collect_counters import native_digits


class CollectorTest(unittest.TestCase):
    def test_native_screen_counters(self):
        frame = np.full((144, 160), 3, dtype=np.uint8)
        for i, digit in enumerate([1, 4, 8, 9, 0]):
            frame[96:104, 112 + i * 8:120 + i * 8] = DIGITS[digit]
        self.assertEqual(read_number(frame, 96, 112, 5), 14890)
        self.assertEqual(native_digits(frame)[5:10].tolist(), [1, 4, 8, 9, 0])
        frame[96:104, 112:120] = 3
        self.assertEqual(read_number(frame, 96, 112, 5), 4890)
        frame[100, 130] = 3 - frame[100, 130]
        self.assertIsNone(read_number(frame, 96, 112, 5))
        self.assertEqual(native_digits(frame)[7], -1)

    def test_geometry_and_placement_clears_line(self):
        board = np.zeros((18, 10), dtype=bool)
        board[-1, :6] = True
        shape = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
        rotations, column = best_placement(board, shape, np.random.default_rng(1))
        self.assertEqual((rotations, column), (0, 6))
        frame = np.full((72, 80), 3, dtype=np.uint8)
        frame[68:72, 8:32] = 0
        np.testing.assert_array_equal(occupancy(frame), board)

    def test_splits_before_windows(self):
        assignments = split_assignments(64, 77)
        self.assertEqual((assignments.count("train"), assignments.count("validation"), assignments.count("test")), (52, 6, 6))
        self.assertEqual(assignments, split_assignments(64, 77))
        with self.assertRaises(ValueError):
            split_assignments(3, 1)

    def test_controller_does_not_pause_competent_play(self):
        frame = np.full((72, 80), 3, dtype=np.uint8)
        frame[:4, 20:32] = 0
        frame[4:8, 24:28] = 0
        controller = ScreenController(3, "heuristic")
        self.assertIn(controller.choose(frame), ("noop", "a", "left", "right", "down"))
        self.assertEqual(normalize([[2, 3], [2, 4]]).tolist(), [[0, 0], [0, 1]])


if __name__ == "__main__":
    unittest.main()
