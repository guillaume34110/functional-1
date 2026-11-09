import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import shutil

import numpy as np
import torch

from intervention_training import (BUTTONS, branch_objective, branch_validation,
                                   finish_validation, first_effect_joint_loss, load_interventions,
                                   temporal_weights, tensors, windowanchors)
from intervention_training import verify_extension
from intervention_training import uniform_next_joint_loss, uniform_next_weights
from model import ACTIONS
from object_annotations import PATTERNS
from scene_model import ObjectSceneDynamics
from token_model import decode_tokens


def fixture():
    shapes = torch.tensor([[[0, 0], [0, 1], [0, 2], [1, 1]], [[0, 1], [1, 0], [1, 1], [2, 1]]])
    book = torch.stack((torch.full((4, 4), 3, dtype=torch.uint8), torch.from_numpy(PATTERNS[6].copy())))
    model = ObjectSceneDynamics(book, shapes, torch.tensor([1, 1]))
    tokens = np.zeros((6, 9, 18, 20), dtype=np.uint8)
    ids = np.zeros((6, 9), dtype=np.int32); ids[4:, 1:] = 1
    cells = np.empty((6, 9, 4, 2), dtype=np.int16)
    for arm in range(6):
        for t in range(9):
            pose = shapes[int(ids[arm, t])].numpy() + [2, 3]
            cells[arm, t] = pose
            tokens[arm, t, pose[:, 0], pose[:, 1] + 2] = 1
    actions = np.array([[ACTIONS.index(button) if button not in ("a", "b") or t == 0 else 0
                         for t in range(8)] for button in BUTTONS], dtype=np.uint8)
    labels = {"cells": cells, "shape_id": ids, "evidence": np.ones((6, 9), dtype=np.uint8),
              "texture": np.full((6, 9), 6, dtype=np.int8), "track": np.zeros((6, 9), dtype=np.int32)}
    return model, {"tokens": tokens, "actions": actions, "labels": labels}


def disk_fixture(directory):
    model, bundle = fixture()
    context = {"rom": {"sha256": "romhash"}}
    parent = {"context": context, "episodes": [{"file": "train/parent.npz", "split": "train", "sha256": "parenthash"}]}
    parent_tokens = np.repeat(bundle["tokens"][0, :1], 49, axis=0)
    episodes = {"train": [(parent_tokens, np.zeros(48, dtype=np.uint8))], "validation": []}
    annotations = {"train": [{key: np.repeat(values[0, :1], 49, axis=0) for key, values in bundle["labels"].items()}], "validation": []}
    metadata = {"source": "capture-intervention", "parent_file": "train/parent.npz", "parent_sha256": "parenthash",
                "split": "train", "anchor": 32, "previous_action": "noop", "context": context,
                "rom_sha256": "romhash", "witness_verified": True, "labels_target_only": True, "snapshot_persisted": False}
    path = directory / "anchor.npz"
    frames = decode_tokens(torch.from_numpy(bundle["tokens"].reshape(-1, 18, 20)).long(), model.codebook)[:, 0].numpy().reshape(6, 9, 72, 80)
    np.savez_compressed(path, frames=frames, actions=bundle["actions"], **bundle["labels"], metadata=json.dumps(metadata))
    record = {**metadata, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"status": "complete", "test_used": False, "version": 1, "parent_manifest_sha256": "parent",
                "annotations_sha256": "annotations", "buttons": list(BUTTONS), "steps": 8, "witness_frames": 9,
                "labels_target_only": True, "dictionary": [{"id": i, "texture": 6, "cells": shape.tolist()}
                                                           for i, shape in enumerate(model.object_shapes)], "episodes": [record]}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return model, parent, episodes, annotations, manifest


def disk_extension_fixture(base, target, original):
    shutil.copyfile(base / "anchor.npz", target / "anchor.npz")
    with np.load(base / "anchor.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(arrays["metadata"])); metadata["anchor"] = 33
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(target / "extra.npz", **arrays)
    record = {**original["episodes"][0], "anchor": 33, "file": "extra.npz",
              "sha256": hashlib.sha256((target / "extra.npz").read_bytes()).hexdigest()}
    extended = copy.deepcopy(original)
    extended["episodes"].append(record)
    extended.update(version=2, base_directory=base.name, base_manifest_sha256=hashlib.sha256((base / "manifest.json").read_bytes()).hexdigest(),
                    inherited_anchors=1, new_anchors=1, inherited_witness_frames=9, new_witness_frames=9,
                    witness_frames=18, anchors_per_parent_max=32)
    (target / "manifest.json").write_text(json.dumps(extended))
    return extended


class InterventionTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_uniform_next_balances_valid_arms_and_masks_unknown_before_nll(self):
        labels = {"shape_id": torch.full((6, 9), -1, dtype=torch.long), "evidence": torch.zeros(6, 9, dtype=torch.long),
                  "cells": torch.full((6, 9, 4, 2), -1, dtype=torch.long)}
        labels["shape_id"][0, 1] = 0; labels["evidence"][0, 1] = 1; labels["cells"][0, 1] = 0
        labels["shape_id"][1, 1:] = 0; labels["evidence"][1, 1:] = 2; labels["cells"][1, 1:] = 0
        weights = uniform_next_weights(labels)
        torch.testing.assert_close(weights.sum(1), torch.tensor([.5, .5, 0., 0., 0., 0.]))
        self.assertEqual(float(weights[0, 0]), .5)
        self.assertEqual(float(weights[1, 4]), 1 / 16)
        logits = torch.zeros(6, 2, 180, requires_grad=True)
        joint = logits.flatten(1).log_softmax(1).reshape_as(logits)
        loss = sum(uniform_next_joint_loss(joint, labels, t, weights) for t in range(8))
        loss.backward()
        torch.testing.assert_close(logits.grad[0], logits.grad[1])
        self.assertTrue(torch.all(logits.grad[2:] == 0))
        labels["shape_id"][:] = -1; labels["evidence"][:] = 0
        weights = uniform_next_weights(labels)
        fresh = torch.zeros(6, 2, 180, requires_grad=True)
        masked = fresh.flatten(1).log_softmax(1).reshape_as(fresh).masked_fill(torch.arange(180)[None, None] == 179, -torch.inf)
        zero = uniform_next_joint_loss(masked, labels, 4, weights)
        self.assertEqual(float(zero), 0.)
        zero.backward(); self.assertTrue(torch.all(fresh.grad == 0))

    def test_uniform_next_scene_late_step_equal_first_and_legacy_default_preserved(self):
        real, bundle = fixture()
        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.register_buffer("codebook", real.codebook)
                self.object_tiles, self.blank_id = real.object_tiles, real.blank_id
                self.values = torch.nn.Parameter(torch.ones(8)); self.calls = 0; self.inputs = []
            def scene_details(self, tokens, actions, state):
                value = self.values[self.calls]; self.calls += 1
                self.inputs.append(tokens.clone())
                return {"state": state, "write": value, "next": {"joint": torch.zeros(6, 2, 180).flatten(1).log_softmax(1).reshape(6, 2, 180)}}
        model = Fake()
        def loss_fn(model, current, target, details, labels, t, valid, matches): return details["write"].square()
        loss = branch_objective(model, bundle, None, loss_fn, supervision="uniform-next")
        loss.backward()
        torch.testing.assert_close(model.values.grad, torch.full((8,), .25))
        for t, tokens in enumerate(model.inputs): np.testing.assert_array_equal(tokens.numpy(), bundle["tokens"][:, t])
        model.calls = 0
        default = branch_objective(model, bundle, None, loss_fn)
        model.calls = 0
        explicit = branch_objective(model, bundle, None, loss_fn, supervision="first-effect")
        torch.testing.assert_close(default, explicit, rtol=0, atol=0)
        with self.assertRaises(ValueError): branch_objective(model, bundle, None, loss_fn, autonomous=True, supervision="uniform-next")

    def test_autonomous_feedback_masks_current_labels_and_keeps_recurrent_graph(self):
        class Feedback(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.register_buffer("codebook", torch.zeros(1, 4, 4, dtype=torch.uint8))
                self.weight = torch.nn.Parameter(torch.tensor(.2)); self.calls = []
            def scene_details(self, tokens, actions, state):
                state = state + self.weight
                self.calls.append((tokens.clone(), state, state.detach().clone()))
                return {"state": state, "write": state.mean()}
            def decode_scene(self, current, details, hard):
                return torch.ones_like(current)
        model = Feedback()
        bundle = {"tokens": np.zeros((6, 9, 18, 20), dtype=np.uint8),
                  "actions": np.zeros((6, 8), dtype=np.uint8), "labels": {}}
        parent = torch.zeros(1, 20, 18, 20, requires_grad=True)
        matches_seen = []
        def loss_fn(model, current, target, details, labels, t, valid, matches):
            matches_seen.append((t, bool(matches[0])))
            self.assertTrue(torch.all(target == 0))
            return details["write"].square()
        teacher = branch_objective(model, bundle, parent, loss_fn)
        teacher_value = float(teacher.detach())
        (.5 * teacher).backward()
        teacher_first = model.calls[0][0].clone(); model.calls.clear(); matches_seen.clear()
        autonomous = branch_objective(model, bundle, parent, loss_fn, autonomous=True)
        self.assertAlmostEqual(float(autonomous.detach()), teacher_value, places=6)
        (.5 * autonomous).backward()
        self.assertTrue(torch.equal(model.calls[0][0], teacher_first))
        self.assertTrue(torch.all(model.calls[1][0] == 1))
        self.assertTrue(all(matched == (t == 0) for t, matched in matches_seen))
        self.assertTrue(all(state.grad_fn is not None for tokens, state, saved in model.calls))
        self.assertAlmostEqual(float(model.calls[-1][2].mean()), 1.6, places=5)
        self.assertIsNone(parent.grad)
        self.assertTrue(torch.all(parent == 0))
        self.assertTrue(torch.isfinite(model.weight.grad))
        weights = temporal_weights(torch.from_numpy(bundle["tokens"]))[0]
        expected_gradient = sum(float(weights[t]) * 2 * .2 * (t + 1) ** 2 for t in range(8))
        self.assertAlmostEqual(float(model.weight.grad), expected_gradient, places=4)

    def test_v2_superset_preserves_inherited_records_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, target = Path(temporary) / "v1", Path(temporary) / "v2"
            base.mkdir(); target.mkdir()
            model, parent, episodes, annotations, original = disk_fixture(base)
            extended = disk_extension_fixture(base, target, original)
            proof = verify_extension(target)
            self.assertTrue(proof["inherited_npz_sha256_preserved"])
            tables, digest = load_interventions(target, model, parent, episodes, annotations, "parent", "annotations")
            self.assertEqual(set(tables["train"]), {(0, 32), (0, 33)})
            self.assertEqual(digest, proof["to_sha256"])
            for key in ("annotations_sha256", "parent_manifest_sha256", "dictionary"):
                bad = copy.deepcopy(extended); bad[key] = "changed"
                (target / "manifest.json").write_text(json.dumps(bad))
                with self.assertRaises(ValueError): verify_extension(target)
            bad = copy.deepcopy(extended); bad["episodes"][0]["anchor"] = 33
            (target / "manifest.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError): verify_extension(target)
            (target / "manifest.json").write_text(json.dumps(extended))
            (target / "anchor.npz").write_bytes(b"corrupt")
            with self.assertRaises(ValueError): verify_extension(target)

    def test_v3_loads_strict_train_only_superset_of_v2(self):
        with tempfile.TemporaryDirectory() as temporary:
            v1, v2, v3 = (Path(temporary) / f"v{version}" for version in (1, 2, 3))
            for directory in (v1, v2, v3):
                directory.mkdir()
            model, parent, episodes, annotations, original = disk_fixture(v1)
            base = disk_extension_fixture(v1, v2, original)
            for record in base["episodes"]:
                (v3 / record["file"]).write_bytes((v2 / record["file"]).read_bytes())
            with np.load(v2 / "extra.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            metadata = json.loads(str(arrays["metadata"]))
            metadata["anchor"] = 34
            arrays["metadata"] = np.array(json.dumps(metadata))
            np.savez_compressed(v3 / "contact.npz", **arrays)
            record = {**base["episodes"][1], "anchor": 34, "file": "contact.npz",
                      "sha256": hashlib.sha256((v3 / "contact.npz").read_bytes()).hexdigest()}
            extended = copy.deepcopy(base)
            extended["episodes"].append(record)
            extended.update(version=3, base_directory=v2.name, anchors_per_parent_max=40,
                            base_manifest_sha256=hashlib.sha256((v2 / "manifest.json").read_bytes()).hexdigest(),
                            inherited_anchors=2, new_anchors=1, inherited_witness_frames=18,
                            new_witness_frames=9, witness_frames=27)
            path = v3 / "manifest.json"
            path.write_text(json.dumps(extended))
            proof = verify_extension(v3, v2)
            tables, digest = load_interventions(v3, model, parent, episodes, annotations, "parent", "annotations")
            self.assertEqual(set(tables["train"]), {(0, 32), (0, 33), (0, 34)})
            self.assertEqual(digest, proof["to_sha256"])
            record["split"] = "validation"
            path.write_text(json.dumps(extended))
            with self.assertRaisesRegex(ValueError, "only add witnessed TRAIN"):
                verify_extension(v3, v2)
            record["split"] = "train"
            path.write_text(json.dumps(extended))
            (v3 / "extra.npz").write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "inherited NPZ"):
                verify_extension(v3, v2)

    def test_extension_rejects_parent_path_empty_extension_and_overfull_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, target = Path(temporary) / "v1", Path(temporary) / "v2"
            base.mkdir(); target.mkdir()
            _, _, _, _, original = disk_fixture(base)
            extended = disk_extension_fixture(base, target, original)
            for name in (".", ".."):
                bad = copy.deepcopy(extended); bad["base_directory"] = name
                (target / "manifest.json").write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "sibling"): verify_extension(target)
            (base / "nested").mkdir()
            (target.parent / "link").symlink_to(base / "nested", target_is_directory=True)
            bad = copy.deepcopy(extended); bad["base_directory"] = "link"
            (target / "manifest.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "sibling"): verify_extension(target)
            bad = copy.deepcopy(extended); bad["episodes"] = bad["episodes"][:1]
            (target / "manifest.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "strict superset"): verify_extension(target)
            bad = copy.deepcopy(extended)
            bad["episodes"] += [{**bad["episodes"][1], "file": f"new-{i}.npz", "anchor": 34 + i} for i in range(31)]
            (target / "manifest.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "at most32"): verify_extension(target)

    def test_joint_loss_penalizes_wrong_pose_even_with_identical_shape_probabilities(self):
        model, bundle = fixture()
        frames, _, labels, _ = tensors(model, bundle, None)
        first = temporal_weights(frames).argmax(1)
        def distribution(correct_pose_probability):
            probabilities = torch.zeros(6, 2, 180)
            probabilities[:, :, 23] = .5 * correct_pose_probability
            probabilities[:, :, 24] = .5 * (1 - correct_pose_probability)
            return probabilities.log()
        good, wrong_pose = distribution(.8), distribution(.2)
        torch.testing.assert_close(good.logsumexp(-1), wrong_pose.logsumexp(-1))
        good_loss, count = first_effect_joint_loss(model, frames, labels, {"next": {"joint": good}}, 0, first)
        bad_loss, bad_count = first_effect_joint_loss(model, frames, labels, {"next": {"joint": wrong_pose}}, 0, first)
        self.assertEqual((count, bad_count), (6, 6))
        self.assertGreater(float(bad_loss), float(good_loss))
        self.assertAlmostEqual(float(good_loss / count), -np.log(.4), places=6)

    def test_first_effect_joint_all_six_arms_qualify_and_unknown_is_masked(self):
        model, bundle = fixture()
        frames, _, labels, _ = tensors(model, bundle, None)
        first = temporal_weights(frames).argmax(1)
        logits = torch.zeros(6, 2, 180, requires_grad=True)
        joint = logits.flatten(1).log_softmax(1).reshape_as(logits)
        loss, count = first_effect_joint_loss(model, frames, labels, {"next": {"joint": joint}}, 0, first)
        self.assertEqual(count, 6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[0, 0].sum()), 0)
        self.assertLess(float(logits.grad[4, 1].sum()), 0)
        self.assertLess(float(logits.grad[5, 1].sum()), 0)
        self.assertLess(float(logits.grad[4, 1, 23]), 0)
        self.assertGreater(float(logits.grad[4, 1, 24]), 0)
        for arm in range(6):
            target_shape = int(labels["shape_id"][arm, 1])
            self.assertLess(float(logits.grad[arm, target_shape, 23]), 0)
        self.assertAlmostEqual(float(logits.grad[0, 0, 23]), float(logits.grad[1, 0, 23]), places=6)
        partially_masked = {key: value.clone() for key, value in labels.items()}
        partially_masked["evidence"][1, 1] = 0
        _, partial_count = first_effect_joint_loss(model, frames, partially_masked, {"next": {"joint": joint}}, 0, first)
        self.assertEqual(partial_count, 5)
        # Blocked rotation is still a qualified target pair, at fallback transition0.
        blocked_frames = frames[0:1].expand(6, -1, -1, -1).clone()
        blocked_labels = {key: value[0:1].expand(6, *value.shape[1:]).clone() for key, value in labels.items()}
        _, count = first_effect_joint_loss(model, blocked_frames, blocked_labels, {"next": {"joint": joint}}, 0,
                                           temporal_weights(blocked_frames).argmax(1))
        self.assertEqual(count, 6)
        masked = {key: value.clone() for key, value in labels.items()}
        masked["evidence"][:] = 0
        fresh = torch.zeros(6, 2, 180, requires_grad=True)
        fresh_joint = fresh.flatten(1).log_softmax(1).reshape_as(fresh)
        fresh_joint = fresh_joint.masked_fill(torch.arange(180)[None, None] == 179, -torch.inf)
        zero, count = first_effect_joint_loss(model, frames, masked, {"next": {"joint": fresh_joint}}, 0, first)
        self.assertEqual(count, 0)
        self.assertEqual(float(zero), 0.)
        zero.backward()
        self.assertTrue(torch.all(fresh.grad == 0))
        _, late_count = first_effect_joint_loss(model, frames, labels, {"next": {"joint": joint}}, 1, first)
        self.assertEqual(late_count, 0)

    def test_spawn_or_changed_background_does_not_qualify_joint_auxiliary(self):
        model, bundle = fixture()
        frames, _, labels, _ = tensors(model, bundle, None)
        frames[:, 1:, 17, 2] = 1
        logits = torch.zeros(6, 2, 180).flatten(1).log_softmax(1).reshape(6, 2, 180)
        _, count = first_effect_joint_loss(model, frames, labels, {"next": {"joint": logits}}, 0,
                                           temporal_weights(frames).argmax(1))
        self.assertEqual(count, 0)

    def test_temporal_weights_focus_delayed_effect_and_fallback_for_blockage(self):
        tokens = torch.zeros(6, 9, 18, 20, dtype=torch.long)
        tokens[4, 3:, 2, 5] = 1  # First divergent target is frame3, hence transition index2.
        tokens[1, 1:, 0, 0] = 1  # HUD differences must never choose an effect time.
        weights = temporal_weights(tokens)
        torch.testing.assert_close(weights.sum(1), torch.ones(6))
        self.assertEqual(float(weights[4, 2]), .5)
        self.assertEqual(float(weights[4, 0]), float(torch.tensor(.5 / 7)))
        for arm in (0, 1, 2, 3, 5):
            self.assertEqual(float(weights[arm, 0]), .5)
        self.assertFalse(weights.requires_grad)

    def test_gradient_favors_first_effect_without_extra_neural_forwards(self):
        real, bundle = fixture()
        bundle["tokens"][:] = 0
        bundle["tokens"][4, 3:, 2, 5] = 1
        class TimeModel(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.register_buffer("codebook", real.codebook)
                self.object_tiles, self.blank_id = real.object_tiles, real.blank_id
                self.values = torch.nn.Parameter(torch.ones(8)); self.calls = 0
            def scene_details(self, tokens, actions, state):
                value = self.values[self.calls]
                self.calls += 1
                return {"state": state, "nested": {"value": value.expand(6, 1)}}
        model = TimeModel()
        parent = torch.zeros(1, 20, 18, 20, requires_grad=True)
        seen = []
        def loss_fn(model, current, target, details, labels, t, valid, matches):
            seen.append((current.shape[0], details["nested"]["value"].shape[0], labels["cells"].shape[0]))
            # Isolate the A arm; labels identify target-only loss selection, never NN inputs.
            return details["nested"]["value"].square().mean() * (target[0, 2, 5] == 1).float()
        loss = branch_objective(model, bundle, parent, loss_fn)
        loss.backward()
        self.assertEqual(model.calls, 8)
        self.assertEqual(len(seen), 48)
        self.assertTrue(all(shapes == (1, 1, 1) for shapes in seen))
        self.assertGreater(float(model.values.grad[2]), 2 * (1 / 8) / 6)
        self.assertGreater(float(model.values.grad[2]), float(model.values.grad[3]))
        self.assertAlmostEqual(float(model.values.grad[2] / model.values.grad[3]), 7., places=5)
        self.assertIsNone(parent.grad)
        self.assertTrue(torch.all(parent == 0))

    def test_loader_provenance_split_alignment_and_vocabulary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model, parent, episodes, annotations, manifest = disk_fixture(directory)
            def load():
                return load_interventions(directory, model, parent, episodes, annotations, "parent", "annotations")
            tables, digest = load()
            self.assertEqual(len(digest), 64)
            self.assertEqual(set(tables["train"]), {(0, 32)})
            self.assertFalse(tables["validation"])
            self.assertEqual(tables["train"][(0, 32)]["tokens"].dtype, np.uint8)
            for key, value in (("parent_manifest_sha256", "wrong"), ("annotations_sha256", "wrong"),
                               ("status", "partial"), ("witness_frames", 0)):
                bad = copy.deepcopy(manifest); bad[key] = value
                (directory / "manifest.json").write_text(json.dumps(bad))
                with self.assertRaises(ValueError): load()
            bad = copy.deepcopy(manifest); bad["episodes"][0]["split"] = "validation"
            (directory / "manifest.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError): load()
            (directory / "manifest.json").write_text(json.dumps(manifest))
            episodes["train"][0][0][32] = 0
            with self.assertRaisesRegex(ValueError, "anchor image"): load()
            model, parent, episodes, annotations, manifest = disk_fixture(directory)
            path = directory / "anchor.npz"
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key].copy() for key in archive.files}
            data["frames"][0, 1, :4, :4] = 2
            np.savez_compressed(path, **data)
            manifest["episodes"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unseen visual tiles"): load()

    def test_window_anchor_alignment_excludes_padding_and_endpoint(self):
        a, b, padding, endpoint = {}, {}, {}, {}
        table = {(2, 10): a, (2, 12): b, (2, 13): padding, (2, 14): endpoint}
        found = windowanchors([(2, 13)], torch.tensor([[True, True, True, False]]), table)
        self.assertEqual(set(found), {0, 2})
        self.assertIs(found[0][0][1], a)
        self.assertIs(found[2][0][1], b)

    def test_objective_observes_only_pixels_buttons_and_preserves_parent_context(self):
        real, bundle = fixture()
        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.register_buffer("codebook", real.codebook)
                self.object_tiles, self.blank_id = real.object_tiles, real.blank_id
                self.weight = torch.nn.Parameter(torch.tensor(2.)); self.calls = []
            def scene_details(self, tokens, actions, state):
                self.calls.append((tokens.clone(), actions.clone(), state.detach().clone()))
                joint = torch.zeros(6, 2, 180).flatten(1).log_softmax(1).reshape(6, 2, 180)
                return {"state": state + 1, "write": self.weight, "next": {"joint": joint}}
        model = Fake()
        parent = torch.zeros(1, 20, 18, 20, requires_grad=True)
        def loss_fn(model, current, target, details, labels, t, valid, matches):
            self.assertTrue(bool(valid.all() and matches.all()))
            return details["write"].square()
        loss = branch_objective(model, bundle, parent, loss_fn)
        self.assertIsNone(model.weight.grad)
        loss.backward()
        self.assertIsNone(parent.grad)
        self.assertTrue(torch.all(parent == 0))
        self.assertAlmostEqual(float(model.weight.grad), 4., places=5)
        for t, (tokens, actions, state) in enumerate(model.calls):
            np.testing.assert_array_equal(tokens.numpy(), bundle["tokens"][:, t])
            np.testing.assert_array_equal(actions.numpy(), bundle["actions"][:, t])
            self.assertEqual(state.shape[0], 6)

    def test_button_ignoring_network_cannot_pass_causal_rotation(self):
        real, bundle = fixture()
        class IgnoreButtons:
            codebook = real.codebook
            object_tiles = real.object_tiles
            blank_id = real.blank_id
            def scene_details(self, tokens, actions, state):
                joint = torch.full((6, 2, 180), -torch.inf)
                joint[:, 0, 23] = 0  # Always keeps the initial orientation at row2,col3.
                return {"state": state, "next": {"joint": joint, "visible": torch.ones(6, 1)}}
            def decode_scene(self, current, details, hard):
                return current
        counts = branch_validation(IgnoreButtons(), bundle, None)
        for button in ("a", "b"):
            self.assertEqual(counts[f"{button}/effect_total"], 8)
            self.assertEqual(counts[f"{button}/effect_correct"], 0)
        result = finish_validation(counts)
        self.assertEqual(result["rates"]["a/rotation_effect_accuracy"], 0)
        self.assertIsNone(result["rates"]["a/rotation_noeffect_accuracy"])


if __name__ == "__main__":
    unittest.main()
