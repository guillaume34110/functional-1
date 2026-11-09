import unittest

import numpy as np

from object_annotations import PATTERNS, annotate_sequence, classify_board, shape_key


def screen(cells, texture=6):
    frame = np.full((72, 80), 3, dtype=np.uint8)
    for y, x in cells:
        frame[y * 4:y * 4 + 4, 8 + x * 4:12 + x * 4] = PATTERNS[texture]
    return frame


class ObjectAnnotationTest(unittest.TestCase):
    def test_exact_texture_and_complete_motion_then_stationary(self):
        self.assertEqual(len(np.unique(PATTERNS.reshape(7, 16), axis=0)), 7)
        np.testing.assert_array_equal(PATTERNS[6], [[2, 2, 2, 0], [2, 1, 3, 0], [2, 3, 3, 0], [0, 0, 0, 0]])
        old = np.array([[2, 3], [2, 4], [2, 5], [3, 4]])
        new = old + [1, 0]
        frames = np.stack((screen(old), screen(new), screen(new)))
        labels, stats = annotate_sequence(frames)
        np.testing.assert_array_equal(labels["texture"], [6, 6, 6])
        np.testing.assert_array_equal(labels["evidence"], [1, 1, 2])
        np.testing.assert_array_equal(labels["cells"][1], new)
        self.assertEqual(stats["motion_pairs"], 1)
        self.assertEqual(stats["groups"], 1)
        self.assertEqual(shape_key(6, old), shape_key(6, new))

    def test_stationary_without_motion_proof_is_unlabeled(self):
        cells = [[2, 3], [2, 4], [2, 5], [3, 4]]
        labels, stats = annotate_sequence(np.stack((screen(cells), screen(cells))))
        self.assertTrue(np.all(labels["texture"] == -1))
        self.assertEqual(stats["labeled_frames"], 0)

    def test_fusion_and_flash_are_not_fabricated_objects(self):
        old = np.array([[2, 3], [2, 4], [2, 5], [3, 4]])
        new = old + [1, 0]
        merged = screen(np.concatenate((new, [[5, 4]])))
        flash = screen(new); flash[40:44, 8:48] = 1
        labels, stats = annotate_sequence(np.stack((screen(old), screen(new), merged, flash)))
        self.assertEqual(int(labels["texture"][2]), -1)
        self.assertEqual(int(labels["evidence"][3]), 3)
        self.assertEqual(stats["invalid_pattern_frames"], 1)
        self.assertTrue(np.any(classify_board(flash) == -2))

    def test_other_board_changes_prevent_false_motion_proof(self):
        old = np.array([[2, 3], [2, 4], [2, 5], [3, 4]])
        new = screen(old + [1, 0]); new[68:72, 8:12] = PATTERNS[0]
        labels, stats = annotate_sequence(np.stack((screen(old), new)))
        self.assertEqual(stats["labeled_frames"], 0)

    def test_immediate_post_flash_updates_remain_uncertain(self):
        old = np.array([[2, 3], [2, 4], [2, 5], [3, 4]])
        flash = screen(old); flash[40:44, 8:48] = 1
        new = screen(old + [1, 0])
        labels, _ = annotate_sequence(np.stack((flash, screen(old), new, new)))
        self.assertTrue(np.all(labels["texture"] == -1))


if __name__ == "__main__":
    unittest.main()
