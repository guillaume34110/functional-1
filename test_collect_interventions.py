import unittest
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from audit_controls_reference import ReplayMismatch
from collect_interventions import branch_labels, choose_anchors, choose_contact_anchors, contact_candidates, contact_stratum, fork_frames, inherit_base, witnessed_forks
from model import ACTIONS
from object_annotations import shape_key
from test_object_annotations import screen


class FakeGame:
    def __init__(self, fail=False):
        self.fail = fail
        self.closed = False
        self.released = []
        self.frame = np.zeros((72, 80), dtype=np.uint8)

    def set_emulation_speed(self, speed):
        self.speed = speed

    def load_state(self, buffer):
        self.snapshot = buffer.read()
        if self.fail:
            raise RuntimeError("unsupported snapshot")

    def button_release(self, action):
        self.released.append(action)

    def stop(self, save):
        self.closed = not save


def fake_step(game, action):
    game.frame = np.full((72, 80), ACTIONS.index(action) % 4, dtype=np.uint8)
    return game.frame.copy()


class InterventionCollectionTest(unittest.TestCase):
    def test_v3_selection_uses_observed_motion_and_previous_down(self):
        old = np.array([[7, 4], [7, 5], [8, 4], [8, 5]])
        new = old + [1, 0]
        pile = np.array([[12, 4]])
        before = screen(np.concatenate((old, pile)))
        after = screen(np.concatenate((new, pile)))
        frames = np.stack([before] * 32 + [after] * 9)
        actions = np.zeros(40, dtype=np.uint8)
        actions[30:32] = ACTIONS.index("down")
        labels = {"shape_id": np.zeros(41, dtype=np.int64), "evidence": np.ones(41, dtype=np.uint8),
                  "texture": np.full(41, 6), "cells": np.stack([old] * 32 + [new] * 9)}
        self.assertEqual(contact_candidates(frames, actions, labels), {32: "pile/gap2/down2"})
        actions[32:] = ACTIONS.index("down")
        self.assertEqual(contact_candidates(frames, actions, labels), {32: "pile/gap2/down2"})
        board = np.full((18, 10), -1)
        self.assertEqual(contact_stratum(board, [(17, 4)], []), "floor/gap0/down0")
        strata = {t: "pile/gap2/down2" for t in range(32, 52)} | {60: "floor/gap1/down0"}
        selected = choose_contact_anchors(strata, {36, 46}, count=8)
        self.assertEqual(len(selected), 8)
        self.assertIn(60, selected)
        self.assertFalse({36, 46} & set(selected))

    def test_extension_preserves_old_quantiles_and_npz_bytes(self):
        labels = {"shape_id": np.zeros(300, dtype=np.int64), "evidence": np.ones(300, dtype=np.uint8)}
        old = choose_anchors(labels, 299)
        added = choose_anchors(labels, 299, 24, old)
        self.assertEqual(len(added), 24)
        self.assertFalse(set(old) & set(added))
        self.assertEqual(len(set(old) | set(added)), 32)
        with tempfile.TemporaryDirectory() as temporary:
            directory, output = Path(temporary) / "v1", Path(temporary) / "v2"
            directory.mkdir(); output.mkdir()
            data = b"immutable-old-npz"
            (directory / "old.npz").write_bytes(data)
            record = {"file": "old.npz", "parent_file": "parent", "parent_sha256": "source", "split": "train",
                      "anchor": old[0], "witness_verified": True, "sha256": hashlib.sha256(data).hexdigest()}
            contract = {"parent_manifest_sha256": "parent-sha", "annotations_sha256": "ann-sha", "dictionary": [], "buttons": [], "steps": 8}
            base = {**contract, "version": 1, "status": "complete", "test_used": False, "episodes": [record], "witness_frames": 9}
            (directory / "manifest.json").write_text(json.dumps(base))
            report = {**contract, "episodes": [], "compressed_bytes": 0}
            inherit_base(directory, output, report, {"parent": {"sha256": "source", "split": "train"}})
            self.assertEqual((output / "old.npz").read_bytes(), data)
            self.assertEqual(report["episodes"][0], record)
            self.assertEqual(report["inherited_anchors"], 1)
            self.assertEqual(report["witness_frames"], 9)

    def test_fork_restores_pending_release_and_closes_on_success_or_failure(self):
        game = FakeGame()
        frames = fork_frames(lambda: game, b"ephemeral", "a", game.frame.copy(), ["b", "noop"],
                             fake_step, lambda game: game.frame)
        self.assertEqual(game.released, ["a"])
        self.assertEqual(game.snapshot, b"ephemeral")
        self.assertTrue(game.closed)
        self.assertEqual(frames.shape, (3, 72, 80))
        self.assertEqual(int(frames[1, 0, 0]), ACTIONS.index("b") % 4)
        self.assertEqual(int(frames[2, 0, 0]), 0)
        broken = FakeGame(fail=True)
        with self.assertRaises(RuntimeError):
            fork_frames(lambda: broken, b"ephemeral", "noop", broken.frame.copy(), [], fake_step, lambda game: game.frame)
        self.assertTrue(broken.closed)

    def test_witness_precedes_branches_and_images_actions_align(self):
        calls = []
        initial = np.zeros((72, 80), dtype=np.uint8)
        def fork(factory, snapshot, previous, frame, commands):
            calls.append(commands)
            return np.stack([frame.copy()] + [np.full_like(frame, ACTIONS.index(command) % 4) for command in commands])
        actual = np.array([0, 1, 2, 3, 4, 5, 0, 0], dtype=np.uint8)
        expected = fork(None, None, None, initial, [ACTIONS[int(action)] for action in actual])
        calls.clear()
        frames, actions = witnessed_forks(None, b"ephemeral", "a", initial, actual, expected, fork)
        self.assertEqual(len(calls), 7)
        self.assertEqual(frames.shape, (6, 9, 72, 80))
        self.assertEqual(actions.shape, (6, 8))
        np.testing.assert_array_equal(frames[:, 1:, 0, 0], actions % 4)
        np.testing.assert_array_equal(actions[4], [4, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(actions[5], [5, 0, 0, 0, 0, 0, 0, 0])
        calls.clear(); bad = expected.copy(); bad[2] = 3
        with self.assertRaises(ReplayMismatch):
            witnessed_forks(None, b"ephemeral", "a", initial, actual, bad, fork)
        self.assertEqual(len(calls), 1)  # No intervention branches after a failed witness.

    def test_anchor_quantiles_use_proven_frames_and_preserve_parent_split(self):
        labels = {"shape_id": np.zeros(120, dtype=np.int64), "evidence": np.ones(120, dtype=np.uint8)}
        labels["evidence"][50:60] = 3
        anchors = choose_anchors(labels, 119)
        self.assertEqual(len(anchors), 8)
        self.assertEqual((anchors[0], anchors[-1]), (32, 111))
        self.assertTrue(all(32 <= index <= 111 and not 50 <= index < 60 for index in anchors))
        # Selection is per parent, never pooled across train/validation or shared windows.
        labels["shape_id"][:] = -1
        self.assertEqual(choose_anchors(labels, 119), [])

    def test_labels_contain_only_visual_targets_and_frozen_shape_ids(self):
        old = np.array([[2, 3], [2, 4], [2, 5], [3, 4]])
        new = old + [1, 0]
        frames = np.stack((screen(old), screen(new), screen(new)))
        seed = {"cells": old, "texture": 6, "shape_id": 0, "track": 0, "evidence": 1}
        dictionary = {shape_key(6, old): 0}
        labels, unknown = branch_labels(frames, seed, dictionary)
        self.assertEqual(unknown, 0)
        self.assertEqual(set(labels), {"cells", "texture", "shape_id", "track", "evidence", "propagation"})
        np.testing.assert_array_equal(labels["shape_id"], [0, 0, 0])
        self.assertFalse(any("state" in key or "memory" in key for key in labels))


if __name__ == "__main__":
    unittest.main()
