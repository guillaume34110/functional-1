import unittest

import numpy as np

from object_annotations import shape_key
from object_annotations_v2 import extend_labels
from test_object_annotations import screen


def seed(frames, poses):
    count = len(frames)
    labels = {"cells": np.full((count, 4, 2), -1, dtype=np.int16),
              "texture": np.full(count, -1, dtype=np.int8),
              "track": np.full(count, -1, dtype=np.int32),
              "shape_id": np.full(count, -1, dtype=np.int32),
              "evidence": np.zeros(count, dtype=np.uint8)}
    dictionary = {}
    for index, cells in poses.items():
        key = shape_key(6, cells)
        dictionary.setdefault(key, len(dictionary))
        labels["cells"][index], labels["texture"][index] = cells, 6
        labels["shape_id"][index] = dictionary[key]
        labels["track"][index], labels["evidence"][index] = index, 1
    return labels, dictionary


class PropagationTest(unittest.TestCase):
    def setUp(self):
        self.old = np.array([[2, 3], [2, 4], [2, 5], [3, 4]])

    def test_spawn_backfill_and_deposit_pair_known(self):
        landing = self.old + [5, 0]
        spawn = self.old
        frames = np.stack((screen(landing), screen(np.concatenate((landing, spawn))),
                           screen(np.concatenate((landing, spawn)))))
        original, dictionary = seed(frames, {0: landing, 2: spawn})
        before = frames.copy()
        labels, stats = extend_labels(frames, original, dictionary)
        np.testing.assert_array_equal(labels["cells"][1], spawn)
        self.assertTrue(np.all(labels["shape_id"][:2] >= 0))
        self.assertEqual(stats["backfilled"], 1)
        # Future evidence modifies offline labels only; images and source labels stay intact.
        np.testing.assert_array_equal(frames, before)
        self.assertEqual(original["shape_id"][1], -1)

    def test_landing_attached_to_same_texture_stack(self):
        new = self.old + [1, 0]
        stack = [[5, 4]]
        frames = np.stack((screen(np.concatenate((self.old, stack))),
                           screen(np.concatenate((new, stack)))))
        original, dictionary = seed(frames, {0: self.old})
        labels, stats = extend_labels(frames, original, dictionary)
        np.testing.assert_array_equal(labels["cells"][1], new)
        self.assertEqual(stats["continued"], 1)

    def test_conflicting_proof_is_not_overwritten_or_crossed(self):
        other = self.old + [7, 0]
        image = screen(np.concatenate((self.old, other)))
        frames = np.stack((image, image, image))
        original, dictionary = seed(frames, {0: self.old, 1: other})
        labels, stats = extend_labels(frames, original, dictionary)
        np.testing.assert_array_equal(labels["cells"][0], self.old)
        np.testing.assert_array_equal(labels["cells"][1], other)
        np.testing.assert_array_equal(labels["cells"][2], other)
        self.assertGreater(stats["conflicts"], 0)

    def test_flash_and_unknown_shape_are_barriers(self):
        flash = screen(self.old); flash[40:44, 8:48] = 1
        frames = np.stack((screen(self.old), flash, screen(self.old)))
        original, dictionary = seed(frames, {0: self.old})
        labels, _ = extend_labels(frames, original, dictionary)
        self.assertTrue(np.all(labels["shape_id"][1:] == -1))
        other_shape = np.array([[3, 3], [3, 4], [3, 5], [3, 6]])
        frames = np.stack((screen(self.old), screen(other_shape)))
        original, dictionary = seed(frames, {0: self.old})
        labels, stats = extend_labels(frames, original, dictionary)
        self.assertEqual(labels["shape_id"][1], -1)
        self.assertEqual(stats["unseen_rejected"], 1)


if __name__ == "__main__":
    unittest.main()
