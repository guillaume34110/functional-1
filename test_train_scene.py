import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from functools import partial

import numpy as np
import torch

from scene_model import (ObjectSceneDynamics, ExtendedSceneDynamics, RelativeSceneDynamics, FootprintSceneDynamics,
                         ActionInvariantCurrentSceneDynamics, action_invariant_current_scene_weights,
                         LearnedOverlapSceneDynamics, learned_overlap_scene_weights,
                         SpawnPreviewSceneDynamics, spawn_preview_scene_weights,
                         PoseRerankSceneDynamics, pose_rerank_scene_weights,
                         expand_scene_weights, expand_scene_state, footprint_scene_weights, relative_scene_weights,
                         scene_from_weights, upgrade_scene_weights)
from token_model import ObservedHistoryDynamics, decode_tokens, load_tile_model
from train_scene import (append_scene_optimizer, composition_coverage, composition_targets, critical_event_counts, critical_event_rates, label_mask, load_action_interventions, load_scene_data, rollout_horizon, scene_loss, score,
                         exposure_key, exposure_summary, intervention_directory, mode_class_weights, optimizer_step_with_exposure,
                         append_memory_optimizer, branch_passes, next_label_coverage, resume_exposure, scale_optimizer_lr, selection_baseline, training_input, training_mode_balance, validate_branch_supervision, visual_absence, weighted_mode_loss, window_labels)
from train_scene import (copy_overlap_loss, descent_focal_loss, descent_ranking_loss, overlap_ranking_loss, recovery_mask,
                         relative_aux_loss, relative_targets, relative_supervision_coverage)
from train_scene import learned_overlap_loss, learned_overlap_targets, spawn_preview_targets
from train_scene import scene_optimizer as actual_scene_optimizer
from intervention_training import verify_extension
from intervention_training import BUTTONS, branch_objective, windowanchors
from train_stateful import StreamWindows
from run import ROOT


def assets():
    book = torch.stack((torch.full((4, 4), 3, dtype=torch.uint8), torch.zeros((4, 4), dtype=torch.uint8),
                        torch.eye(4, dtype=torch.uint8), torch.full((4, 4), 2, dtype=torch.uint8)))
    shapes = torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 0], [1, 0], [2, 0], [2, 1]]])
    return book, shapes, torch.tensor([1, 2])


class CopyOverlapTest(unittest.TestCase):
    def test_only_proven_clear_target_with_occupied_background_contributes(self):
        model = ObjectSceneDynamics(*assets())
        frames = torch.zeros(3, 18, 20, dtype=torch.long)
        frames[:, 0, 2:4] = 1
        current = torch.zeros_like(frames, dtype=torch.bool)
        current[:, 0, 2] = True
        future = torch.zeros_like(current)
        future[1, 0, 3] = True
        logits = torch.zeros(3, 2, 180, requires_grad=True)
        joint = logits.masked_fill(~model.valid_poses, -torch.inf).flatten(1).log_softmax(1).reshape_as(logits)
        loss = copy_overlap_loss(model, frames, joint, current, future, torch.tensor([True, True, False]))
        self.assertGreater(loss.item(), 0.)
        loss.backward()
        self.assertGreater(logits.grad[0, 0, 0].item(), 0.)
        self.assertLess(logits.grad[0, 1, 0].item(), 0.)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[1:].count_nonzero().item(), 0)


class RecoveryMaskTest(unittest.TestCase):
    def test_requires_bounded_board_error_and_correct_pose_after_observed_input(self):
        observed = torch.zeros(5, 18, 20, dtype=torch.long)
        generated = observed.clone()
        generated[:4, 3, 4] = 1
        generated[3, 4, 2:12] = 1
        generated[4, 1, 15] = 1
        cells = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        labels = {"shape_id": torch.zeros(5, 1, dtype=torch.long),
                  "cells": cells.expand(5, 1, 4, 2),
                  "evidence": torch.ones(5, 1, dtype=torch.long)}
        mask = recovery_mask(generated, observed, torch.tensor([0, 1, 0, 0, 0]), labels, 0,
                             torch.tensor([True, True, False, True, True]), torch.ones(5, dtype=torch.bool))
        self.assertEqual(mask.tolist(), [True, False, False, False, False])


class OverlapRankingTest(unittest.TestCase):
    def test_only_strongest_valid_clash_competes_with_free_target(self):
        model = ObjectSceneDynamics(*assets())
        frame = torch.zeros(3, 18, 20, dtype=torch.long)
        frame[:2, 0, 2] = 1
        cells = torch.tensor([[[0, 1], [0, 2], [1, 1], [1, 2]],
                              [[0, 0], [0, 1], [1, 0], [1, 1]],
                              [[0, 1], [0, 2], [1, 1], [1, 2]]])
        raw = torch.zeros(3, 2, 180, requires_grad=True)
        with torch.no_grad():
            raw[0, 0, 0] = 1.
            raw[0, 1, 0] = .2
        joint = raw.masked_fill(~model.valid_poses, -torch.inf)
        loss = overlap_ranking_loss(model, frame, joint, torch.zeros_like(frame, dtype=torch.bool), cells,
                                    torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.bool))
        self.assertEqual(loss.item(), 1.5)
        loss.backward()
        self.assertEqual(raw.grad[0, 0, :2].tolist(), [1., -1.])
        self.assertEqual(raw.grad.count_nonzero().item(), 2)


class DescentFocalTest(unittest.TestCase):
    def test_uncertain_targets_are_excluded_and_low_probability_has_more_weight(self):
        logits = torch.zeros(3, 1, 180, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 0] = 9
        log_joint = logits.log_softmax(-1)
        cells = torch.zeros(3, 4, 2, dtype=torch.long)
        shapes = torch.tensor([0, 0, -1])
        known = torch.tensor([True, True, False])
        loss = descent_focal_loss(log_joint, cells, shapes, known, known)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(abs(logits.grad[1, 0, 0]), abs(logits.grad[0, 0, 0]))
        self.assertEqual(logits.grad[2].count_nonzero().item(), 0)
        self.assertEqual(descent_focal_loss(log_joint, cells, shapes, ~known & known, known).item(), 0.)


class DescentRankingTest(unittest.TestCase):
    def test_selects_strongest_valid_wrong_shape_or_column(self):
        model = ObjectSceneDynamics(*assets())
        cells = model.object_shapes[0][None, None] + torch.tensor([[[[2, 2]], [[3, 2]]]])
        shape_ids = torch.zeros(1, 2, dtype=torch.long)
        joint = torch.zeros(1, 2, 180, requires_grad=True)
        with torch.no_grad():
            joint[0, 0, 42] = 9.  # Same shape and column, so not a distractor.
            joint[0, 0, 179] = 9.  # Outside the geometrically valid origins.
            joint[0, 0, 33] = .6
            joint[0, 1, 32] = 1.
        loss = descent_ranking_loss(model, joint, cells, shape_ids, torch.tensor([True]))
        self.assertEqual(loss.item(), 1.5)
        loss.backward()
        self.assertEqual(joint.grad[0, 0, 32].item(), -1.)
        self.assertEqual(joint.grad[0, 1, 32].item(), 1.)
        self.assertEqual(joint.grad.count_nonzero().item(), 2)

    def test_scene_loss_gates_descent_ranking_by_parent_action_and_proof(self):
        model = ObjectSceneDynamics(*assets())
        current = torch.zeros(6, 18, 20, dtype=torch.long)
        target = current.clone()
        shape_ids = torch.zeros(6, 2, dtype=torch.long)
        shape_ids[4, 1] = 1
        origins = torch.tensor([[[2, 2], [3, 2]]] * 6)
        origins[3, 1, 1] += 1
        cells = model.object_shapes[shape_ids] + origins[:, :, None]
        for i in range(6):
            for t, board in enumerate((current, target)):
                rows, columns = cells[i, t].unbind(1)
                board[i, rows, columns + 2] = model.object_tiles[shape_ids[i, t]]
        target[2, 12, 7] = 1  # A background edit makes this a non-copy transition.
        evidence = torch.ones(6, 2, dtype=torch.long)
        evidence[5, 1] = 0
        labels = {"cells": cells, "shape_id": shape_ids, "evidence": evidence}
        actions = torch.tensor([3, 0, 3, 3, 3, 3])
        valid = torch.ones(6, dtype=torch.bool)
        details = model.scene_details(current, actions)
        raw = torch.zeros_like(details["next"]["joint"], requires_grad=True)
        with torch.no_grad():
            raw[:, 1, 32] = 1
        details["next"]["joint"] = raw.masked_fill(~model.valid_poses, -torch.inf).flatten(1).log_softmax(1).reshape_as(raw)
        base = scene_loss(model, current, target, details, labels, 0, valid, valid, actions=actions)
        ranked = scene_loss(model, current, target, details, labels, 0, valid, valid,
                            actions=actions, descent_ranking_weight=1.)
        torch.testing.assert_close(ranked - base, torch.tensor(1.5))
        gradient = torch.autograd.grad(ranked - base, raw)[0]
        self.assertGreater(gradient[0].abs().sum().item(), 0.)
        torch.testing.assert_close(gradient[1:], torch.zeros_like(gradient[1:]))
        without_commands = scene_loss(model, current, target, details, labels, 0, valid, valid,
                                      descent_ranking_weight=1.)
        torch.testing.assert_close(without_commands, base)


def scene_optimizer(model):
    return actual_scene_optimizer(model)


class SceneTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_spawn_preview_requires_exact_supported_addition(self):
        model = SpawnPreviewSceneDynamics(*assets())
        current = torch.zeros(7, 18, 20, dtype=torch.long)
        mask = torch.zeros_like(current, dtype=torch.bool)
        mask[:, 16:, 4:6] = True
        mask[2] = False
        mask[2, 15:17, 4:6] = True
        current[mask] = 1
        target = current.clone()
        target[:, 1:3, 6:8] = 1
        target[3, 16, 4] = 0
        target[4, 1, 6] = 2
        target[5, 1, 8] = 1
        target[6, 1:3, 6:8] = 0
        target[6, 3:5, 6:8] = 1
        positive = torch.tensor([True, False, True, True, True, True, True])
        rows, wanted = spawn_preview_targets(model, current, target, mask, positive, torch.zeros(7, dtype=torch.long))
        torch.testing.assert_close(rows, torch.tensor([0]))
        torch.testing.assert_close(wanted, torch.tensor([14]))

    def test_spawn_preview_migration_preserves_three_optimizer_groups(self):
        baseline = LearnedOverlapSceneDynamics(*assets())
        optimizer = scene_optimizer(baseline)
        sum(parameter.square().sum() for parameter in baseline.parameters()).backward()
        optimizer.step()
        before = optimizer.state_dict()
        model = spawn_preview_scene_weights(baseline.state_dict())
        restored = scene_optimizer(model)
        restored.load_state_dict(append_memory_optimizer(before, model))
        actual = restored.state_dict()
        self.assertEqual(actual["param_groups"][:3], before["param_groups"])
        self.assertEqual(len(actual["param_groups"][3]["params"]), 4)
        self.assertEqual(actual["state"].keys(), before["state"].keys())
        for key, state in before["state"].items():
            torch.testing.assert_close(actual["state"][key], state, rtol=0, atol=0)

    def test_learned_overlap_supervision_removes_current_object_and_balances_classes(self):
        model = LearnedOverlapSceneDynamics(*assets())
        current = torch.zeros(3, 18, 20, dtype=torch.long)
        current[:, 2:4, 4:6] = 1
        current[0, 6, 8] = 2
        current[2, 6, 8] = 2
        current_mask = torch.zeros_like(current, dtype=torch.bool)
        current_mask[:, 2:4, 4:6] = True
        targets = learned_overlap_targets(model, current, current_mask)
        self.assertFalse(targets[0, 0, 22])
        self.assertTrue(targets[0, 0, 66])
        self.assertFalse(targets[1].any())
        logits = torch.zeros(3, *model.valid_poses.shape, requires_grad=True)
        loss = learned_overlap_loss(model, logits, current, current_mask, torch.tensor([True, True, False]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        valid = model.valid_poses
        positive = targets[0] & valid
        negative = ~targets[0] & valid
        torch.testing.assert_close(logits.grad[0][positive].abs().sum(), logits.grad[0][negative].abs().sum())
        self.assertTrue(torch.all(logits.grad[1][valid] > 0))
        self.assertTrue(torch.all(logits.grad[:, ~valid] == 0))
        self.assertTrue(torch.all(logits.grad[2] == 0))

    def test_learned_overlap_migration_retains_existing_optimizer_state(self):
        baseline = ObjectSceneDynamics(*assets())
        optimizer = scene_optimizer(baseline)
        sum(parameter.square().sum() for parameter in baseline.parameters()).backward()
        optimizer.step()
        before = optimizer.state_dict()
        model = learned_overlap_scene_weights(baseline.state_dict())
        restored = scene_optimizer(model)
        restored.load_state_dict(append_memory_optimizer(before, model))
        actual = restored.state_dict()
        self.assertEqual(actual["param_groups"][:2], before["param_groups"])
        self.assertEqual(len(actual["param_groups"][2]["params"]), 4)
        for identifier, moment in before["state"].items():
            for key, value in moment.items():
                torch.testing.assert_close(actual["state"][identifier][key], value, rtol=0, atol=0)

    def test_action_invariant_current_preserves_parameters_rng_and_optimizer_groups(self):
        baseline = ObjectSceneDynamics(*assets())
        optimizer = scene_optimizer(baseline)
        sum(parameter.square().sum() for parameter in baseline.parameters()).backward()
        optimizer.step()
        before = optimizer.state_dict()
        rng = torch.get_rng_state()
        model = action_invariant_current_scene_weights(baseline.state_dict())
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(list(dict(model.named_parameters())), list(dict(baseline.named_parameters())))
        restored = scene_optimizer(model)
        restored.load_state_dict(before)
        actual = restored.state_dict()
        self.assertEqual(len(actual["param_groups"]), 2)
        self.assertEqual(actual["param_groups"], before["param_groups"])
        for identifier, moment in before["state"].items():
            for key, value in moment.items():
                torch.testing.assert_close(actual["state"][identifier][key], value, rtol=0, atol=0)

    def test_local_pose_migration_preserves_optimizer_moments_and_cursor(self):
        baseline = ObjectSceneDynamics(*assets())
        optimizer = scene_optimizer(baseline)
        sum(parameter.square().sum() for parameter in baseline.parameters()).backward()
        optimizer.step()
        before = optimizer.state_dict()
        checkpoint = {"model": baseline.state_dict(), "stream_cursors": [[0, 32]]}
        model = footprint_scene_weights(checkpoint["model"])
        restored = scene_optimizer(model)
        restored.load_state_dict(append_memory_optimizer(before, model))
        actual = restored.state_dict()
        self.assertEqual(actual["param_groups"][:2], before["param_groups"])
        self.assertEqual(len(actual["param_groups"][2]["params"]), 2)
        for identifier, moment in before["state"].items():
            for key, value in moment.items():
                torch.testing.assert_close(actual["state"][identifier][key], value, rtol=0, atol=0)
        self.assertEqual(checkpoint["stream_cursors"], [[0, 32]])
        with self.assertRaises(ValueError):
            footprint_scene_weights(model.state_dict())

    def test_pose_rerank_migration_adds_only_new_parameters_to_optimizer(self):
        baseline = ObjectSceneDynamics(*assets())
        optimizer = scene_optimizer(baseline)
        sum(parameter.square().sum() for parameter in baseline.parameters()).backward()
        optimizer.step()
        before = optimizer.state_dict()
        model = pose_rerank_scene_weights(baseline.state_dict())
        restored = scene_optimizer(model)
        restored.load_state_dict(append_memory_optimizer(before, model))
        actual = restored.state_dict()
        self.assertEqual(actual["param_groups"][:2], before["param_groups"])
        self.assertEqual(len(actual["param_groups"][2]["params"]), 3)
        for identifier, moment in before["state"].items():
            for key, value in moment.items():
                torch.testing.assert_close(actual["state"][identifier][key], value, rtol=0, atol=0)

    def test_relative_optimizer_migration_and_raw_full_ui_state20(self):
        from play import NeuralSession
        old = ObjectSceneDynamics(*assets()); optimizer = scene_optimizer(old)
        sum(p.square().sum() for p in old.parameters()).backward(); optimizer.step()
        original = copy.deepcopy(optimizer.state_dict())
        model = relative_scene_weights(old.state_dict()).eval()
        upgraded = append_memory_optimizer(original, model)
        self.assertEqual(upgraded["param_groups"][:2], original["param_groups"])
        self.assertEqual(len(upgraded["param_groups"][2]["params"]), 4)
        resumed = scene_optimizer(model); resumed.load_state_dict(upgraded)
        self.assertEqual(len(resumed.param_groups), 3)
        with self.assertRaises(ValueError): append_memory_optimizer(upgraded, model)
        pixels = decode_tokens(torch.zeros(1, 18, 20, dtype=torch.long), model.codebook)
        captures = np.repeat(pixels[:, 0].numpy(), 3, axis=0)
        with tempfile.TemporaryDirectory() as directory:
            for wrapped in (False, True):
                path = Path(directory) / f"relative-{wrapped}.pt"
                torch.save({"model": model.state_dict()} if wrapped else model.state_dict(), path)
                loaded = load_tile_model(path)
                self.assertIsInstance(loaded, RelativeSceneDynamics)
                with torch.inference_mode():
                    expected, state = model.predict_frame(pixels, torch.tensor([1]))
                    actual, loaded_state = loaded.predict_frame(pixels, torch.tensor([1]))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(loaded_state, state, rtol=0, atol=0)
                with patch("play.read_episode", return_value=(captures, np.zeros(2, dtype=np.uint8), {"source": "capture"})):
                    session = NeuralSession(path, Path("unused.npz")); initial = session.reset()["frame"]
                    self.assertEqual(session.state.shape, (1, 20, 18, 20))
                    session.step("left"); self.assertEqual(session.reset()["frame"], initial)

    def test_relative_auxiliary_masks_changed_background_and_requires_positive_objects(self):
        model = RelativeSceneDynamics(*assets())
        current = torch.zeros(5, 18, 20, dtype=torch.long)
        old = torch.tensor([[2, 2], [2, 3], [3, 2], [3, 3]])
        cells = old.expand(5, 2, 4, 2).clone()
        cells[:, 1] += torch.tensor([1, 0])
        cells[2, 1] += torch.tensor([4, 0])
        labels = {"shape_id": torch.zeros(5, 2, dtype=torch.long), "evidence": torch.ones(5, 2, dtype=torch.long), "cells": cells}
        labels["shape_id"][3, 1] = 1
        labels["evidence"][4, 0] = 0
        target = current.clone()
        for sample in range(5):
            current[sample, old[:, 0], old[:, 1] + 2] = 1
            c = cells[sample, 1]; target[sample, c[:, 0], c[:, 1] + 2] = model.object_tiles[labels["shape_id"][sample, 1]]
        target[1, 15, 10] = 2  # Changed background alone is UNKNOWN, never gate-negative.
        valid = torch.ones(5, dtype=torch.bool)
        cp, cm = label_mask(labels, 0, valid); np_, nm = label_mask(labels, 1, valid)
        positive, negative, delta = relative_targets(model, current, target, labels, 0, cp, np_, cm, nm)
        torch.testing.assert_close(positive, torch.tensor([True, False, False, False, False]))
        torch.testing.assert_close(negative, torch.tensor([False, False, True, True, False]))
        details = model.scene_details(current, torch.tensor([0, 1, 2, 4, 5]))
        details["transport"]["delta_logits"].retain_grad(); details["transport"]["gate_logits"].retain_grad()
        loss = relative_aux_loss(model, current, target, details, labels, 0, cp, np_, cm, nm)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        delta_grad, gate_grad = details["transport"]["delta_logits"].grad, details["transport"]["gate_logits"].grad
        self.assertTrue(torch.all(delta_grad[1:] == 0)); self.assertLess(float(delta_grad[0, delta[0]]), 0.)
        self.assertTrue(torch.all(gate_grad[[1, 4]] == 0))
        self.assertLess(float(gate_grad[0]), 0.); self.assertTrue(torch.all(gate_grad[[2, 3]] > 0))
        model.zero_grad(set_to_none=True)
        unknown_details = model.scene_details(current, torch.tensor([0, 1, 2, 4, 5]))
        absent = torch.zeros(5, dtype=torch.bool)
        zero = relative_aux_loss(model, current, target, unknown_details, labels, 0, absent, absent, cm, nm)
        self.assertEqual(float(zero), 0.)
        zero.backward()
        self.assertTrue(torch.all(model.motion_delta.weight.grad == 0))
        self.assertTrue(torch.all(model.motion_gate.weight.grad == 0))

    def test_uniform_supervision_rejects_mixed_and_coverage_is_train_table_only(self):
        validate_branch_supervision("teacher", "uniform-next")
        validate_branch_supervision("mixed", "first-effect")
        with self.assertRaises(ValueError): validate_branch_supervision("mixed", "uniform-next")
        bundle = {"labels": {"shape_id": np.zeros((6, 9), dtype=np.int32), "evidence": np.ones((6, 9), dtype=np.uint8)}}
        bundle["labels"]["evidence"][0, 1:8] = 0
        coverage = next_label_coverage({(0, 32): bundle})
        self.assertEqual(coverage["valid_next_per_arm"], [1, 8, 8, 8, 8, 8])
        self.assertEqual(coverage["valid_count_histogram_per_arm"][0][1], 1)
        self.assertEqual(coverage["total_next_per_arm"], [8] * 6)

    def test_memory_optimizer_migration_preserves_two_groups_and_resumes_three(self):
        old = ObjectSceneDynamics(*assets()); optimizer = scene_optimizer(old)
        sum(p.square().sum() for p in old.parameters()).backward(); optimizer.step()
        original = copy.deepcopy(optimizer.state_dict())
        model = expand_scene_weights(old.state_dict())
        upgraded = append_memory_optimizer(original, model)
        self.assertEqual(len(upgraded["param_groups"]), 3)
        self.assertEqual(upgraded["param_groups"][:2], original["param_groups"])
        self.assertEqual(len(upgraded["param_groups"][2]["params"]), 6)
        actual = scene_optimizer(model); actual.load_state_dict(upgraded)
        self.assertEqual(actual.param_groups[2]["lr"], 3e-3)
        rates = scale_optimizer_lr(actual, .1)
        for actual_lr, expected_lr in zip(rates["after"], [3e-5, 3e-4, 3e-4], strict=True):
            self.assertAlmostEqual(actual_lr, expected_lr)
        for identifier, values in original["state"].items():
            for key, value in values.items(): torch.testing.assert_close(actual.state_dict()["state"][identifier][key], value)
        with self.assertRaises(ValueError): append_memory_optimizer(upgraded, model)
        resumed = scene_optimizer(scene_from_weights(model.state_dict())); resumed.load_state_dict(actual.state_dict())
        self.assertEqual(len(resumed.param_groups), 3)
        for key, value in original["state"].items():
            for name, item in value.items(): torch.testing.assert_close(resumed.state_dict()["state"][key][name], item)

    def test_lr_scaling_preserves_moments_and_default_resume_does_not_scale_twice(self):
        parameters = [torch.nn.Parameter(torch.tensor(1.)), torch.nn.Parameter(torch.tensor(2.))]
        optimizer = torch.optim.AdamW([{"params": [parameters[0]], "lr": 3e-4}, {"params": [parameters[1]], "lr": 3e-3}])
        sum(parameter.square() for parameter in parameters).backward(); optimizer.step()
        before = copy.deepcopy(optimizer.state_dict())
        rates = scale_optimizer_lr(optimizer, .1)
        self.assertEqual(rates["before"], [3e-4, 3e-3])
        for actual, expected in zip(rates["after"], [3e-5, 3e-4], strict=True): self.assertAlmostEqual(actual, expected)
        for identifier, values in before["state"].items():
            for key, value in values.items(): torch.testing.assert_close(optimizer.state_dict()["state"][identifier][key], value)
        resumed = torch.optim.AdamW([{"params": [parameters[0]]}, {"params": [parameters[1]]}])
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        again = scale_optimizer_lr(resumed, 1.)
        self.assertEqual(again["before"], rates["after"])
        self.assertEqual(again["after"], rates["after"])
        for scale in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError): scale_optimizer_lr(resumed, scale)
        self.assertEqual([group["lr"] for group in resumed.param_groups], rates["after"])

    def test_branch_input_modes_have_one_bundle_weight_and_no_randomness(self):
        rng = torch.get_rng_state().clone()
        self.assertEqual(branch_passes("teacher"), ((False, 1.),))
        self.assertEqual(branch_passes("mixed"), ((False, .5), (True, .5)))
        self.assertEqual(sum(weight for _, weight in branch_passes("mixed")), 1.)
        with self.assertRaises(ValueError): branch_passes("other")
        torch.testing.assert_close(torch.get_rng_state(), rng)

    def test_exposure_commits_only_after_success_and_counts_duplicate_streams(self):
        table = {(2, 10): {}, (2, 12): {}}
        exposure = resume_exposure({"step": 100}, table, "addon")
        self.assertEqual(exposure["tracking_since_step"], 101)
        self.assertEqual(exposure["keys"], [])
        valid = torch.tensor([[True, True, True, False], [True, True, True, False]])
        cursors = [(2, 13), (2, 13)]
        keys = [exposure_key(cursors, valid, slot, t) for slot, t in ((0, 0), (1, 0), (0, 2))]
        self.assertEqual(keys, [(2, 10), (2, 10), (2, 12)])
        with self.assertRaises(ValueError): exposure_key(cursors, valid, 0, 3)
        class Failing:
            def step(self): raise RuntimeError("optimizer failed")
        with self.assertRaises(RuntimeError): optimizer_step_with_exposure(Failing(), exposure, 101, keys, table)
        self.assertEqual((exposure["bundles"], exposure["keys"], exposure["last_completed_step"]), (0, [], 100))
        class Success:
            calls = 0
            def step(self): self.calls += 1
        optimizer = Success()
        updated = optimizer_step_with_exposure(optimizer, exposure, 101, keys, table)
        self.assertEqual(optimizer.calls, 1)
        self.assertEqual(updated["bundles"], 3)
        self.assertEqual(updated["keys"], [[2, 10], [2, 12]])
        summary = exposure_summary(updated, table)
        self.assertNotIn("keys", summary)
        self.assertEqual((summary["unique_anchors"], summary["total_train_anchors"], summary["applied_bundles"]), (2, 2, 3))
        selected = exposure_summary(updated, table)
        later = optimizer_step_with_exposure(optimizer, updated, 102, [(2, 10)], table)
        self.assertEqual(selected["applied_bundles"], 3)
        self.assertEqual(exposure_summary(later, table)["applied_bundles"], 4)
        with self.assertRaises(ValueError): optimizer_step_with_exposure(optimizer, later, 104, [], table)
        with self.assertRaises(ValueError): optimizer_step_with_exposure(optimizer, later, 103, [(9, 9)], table)
        self.assertEqual(optimizer.calls, 2)

    def test_exposure_resume_validates_schema_hash_subset_and_migration_reset(self):
        table = {(0, 32): {}, (0, 40): {}}
        ledger = {"version": 1, "manifest_sha256": "addon", "tracking_since_step": 91, "last_completed_step": 100,
                  "bundles": 3, "keys": [[0, 32]], "initialization": "legacy-checkpoint-no-history"}
        resumed = resume_exposure({"step": 100, "training_exposure": ledger}, table, "addon")
        self.assertEqual(resumed, ledger)
        self.assertIsNot(resumed, ledger)
        for field, value in (("keys", [[0, 999]]), ("keys", [[0, 32], [0, 32]]), ("keys", [[0, True]]),
                             ("bundles", -1), ("bundles", 0), ("last_completed_step", 99), ("tracking_since_step", 102),
                             ("version", True), ("keys", [])):
            malformed = copy.deepcopy(ledger); malformed[field] = value
            with self.assertRaises(ValueError): resume_exposure({"step": 100, "training_exposure": malformed}, table, "addon")
        with self.assertRaises(ValueError): resume_exposure({"step": 100, "training_exposure": ledger}, table, "new-addon")
        reset = resume_exposure({"step": 100, "training_exposure": ledger}, table, "new-addon",
                                {"from_sha256": "addon", "to_sha256": "new-addon"})
        self.assertEqual(reset["initialization"], "verified-corpus-migration-reset")
        self.assertEqual((reset["tracking_since_step"], reset["bundles"], reset["keys"]), (101, 0, []))

    def test_explicit_v2_migration_and_auto_relative_reload_preserve_resume_state(self):
        from test_intervention_training import disk_extension_fixture, disk_fixture
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "data").mkdir()
            base, target = root / "data/v1", root / "data/v2"
            base.mkdir(); target.mkdir()
            model, parent, episodes, annotations, original = disk_fixture(base)
            (root / "data/manifest.json").write_text(json.dumps(parent))
            base_sha = hashlib.sha256((base / "manifest.json").read_bytes()).hexdigest()
            disk_extension_fixture(base, target, original)
            checkpoint = {"config": {"protocol": "object-scene-v5-action-interventions", "interventions_sha256": base_sha,
                                      "interventions_path": "data/v1", "manifest_sha256": "parent", "annotations_sha256": "annotations"},
                          "model": copy.deepcopy(model.state_dict()), "optimizer": {"moments": 123}, "stream_cursors": [[0, 32]]}
            rng = torch.get_rng_state().clone(); weights = copy.deepcopy(model.state_dict())
            with patch("train_scene.ROOT", root):
                with self.assertRaisesRegex(ValueError, "lineage mismatch"):
                    load_action_interventions(checkpoint, model, episodes, annotations, "parent", "annotations", Path("data/v2"))
                tables, target_sha = load_action_interventions(checkpoint, model, episodes, annotations, "parent", "annotations",
                                                               Path("data/v2"), Path("data/v1"))
                self.assertEqual(set(tables["train"]), {(0, 32), (0, 33)})
                migrated = copy.deepcopy(checkpoint)
                migrated["config"].update(interventions_sha256=target_sha, interventions_path="data/v2")
                _, loaded_sha = load_action_interventions(migrated, model, episodes, annotations, "parent", "annotations")
                self.assertEqual(loaded_sha, target_sha)
                with self.assertRaises(ValueError): selection_baseline(migrated, checkpoint, model)
                baseline = selection_baseline(migrated, checkpoint, model, verify_extension(target))
                self.assertIsNot(baseline, model)
                with self.assertRaises(ValueError): intervention_directory(checkpoint, Path("../outside"))
            torch.testing.assert_close(torch.get_rng_state(), rng)
            self.assertEqual(checkpoint["optimizer"], {"moments": 123})
            self.assertEqual(checkpoint["stream_cursors"], [[0, 32]])
            for key, value in weights.items(): torch.testing.assert_close(model.state_dict()[key], value)

    def test_dense_profile_selects_train_windows_with_most_anchors(self):
        episodes = [(np.zeros((201, 18, 20), dtype=np.uint8), np.zeros(200, dtype=np.uint8)) for _ in range(5)]
        table = {(episode, anchor): {} for episode in range(5) for anchor in (32, 70, 71, 72, 150)}
        windows = dense_train_windows(episodes, table)
        self.assertEqual(len(windows), 4)
        self.assertTrue(all(count == 3 and start == 70 for count, episode, start in windows))
        self.assertEqual(len({episode for count, episode, start in windows}), 4)

    def test_global_mode_weighting_balances_rare_classes_without_batch_renormalization(self):
        weights = mode_class_weights({"copy": 90, "compose": 5, "rewrite": 5})
        self.assertAlmostEqual(weights[0], 100 / 270)
        self.assertAlmostEqual(weights[1], 100 / 15)
        with self.assertRaises(ValueError):
            mode_class_weights({"copy": 90, "compose": 0, "rewrite": 5})
        logits = torch.zeros(100, 3, requires_grad=True)
        targets = torch.tensor([0] * 90 + [1] * 5 + [2] * 5)
        loss = weighted_mode_loss(logits, targets, torch.ones(100, dtype=torch.bool), weights)
        loss.backward()
        contributions = [logits.grad[targets == kind].abs().sum() for kind in range(3)]
        torch.testing.assert_close(contributions[0], contributions[1])
        torch.testing.assert_close(contributions[1], contributions[2])
        rare_only = weighted_mode_loss(torch.zeros(1, 3), torch.tensor([1]), torch.tensor([True]), weights)
        self.assertAlmostEqual(float(rare_only), np.log(3) * weights[1], places=5)
        invalid = torch.tensor([[float("inf"), 0., 0.]], requires_grad=True)
        zero = weighted_mode_loss(invalid, torch.tensor([0]), torch.tensor([False]), weights)
        self.assertEqual(float(zero), 0.)
        zero.backward()
        self.assertTrue(torch.all(invalid.grad == 0))

    def test_mode_frequency_coverage_reads_only_train_and_separates_corpora(self):
        parent_episodes, parent_labels = object(), object()
        bundle = {"tokens": np.zeros((6, 9, 18, 20), dtype=np.uint8), "actions": np.zeros((6, 8), dtype=np.uint8),
                  "labels": {"shape_id": np.zeros((6, 9), dtype=np.int32)}}
        calls = []
        def coverage(model, episodes, annotations):
            calls.append((episodes, annotations))
            return {"copy": 90, "compose": 5, "rewrite": 5, "masked": 0} if episodes is parent_episodes else {"copy": 6, "compose": 3, "rewrite": 3, "masked": 0}
        with patch("train_scene.composition_coverage", side_effect=coverage):
            counts, weights = training_mode_balance(None, {"train": parent_episodes}, {"train": parent_labels}, {"train": {(0, 32): bundle}})
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][0], parent_episodes)
        self.assertIs(calls[0][1], parent_labels)
        self.assertEqual(len(calls[1][0]), 6)
        self.assertEqual(counts["parent"]["copy"], 90)
        self.assertEqual(counts["interventions"]["copy"], 6)
        self.assertNotEqual(weights["parent"], weights["interventions"])

    def test_v5_selection_and_curriculum_require_actual_button_effects(self):
        teacher = {"addition_accuracy": .99, "erasure_accuracy": .99, "event_fpr": 0.,
                   "critical_counts": {"mode_confusion": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "motion": 1, "still": 1},
                   "mode_recall": dict.fromkeys(("copy", "compose", "rewrite"), .99),
                   "next_joint_accuracy": dict.fromkeys(("motion", "still"), .99)}
        counts, rates = {}, {}
        for button in BUTTONS:
            counts[f"{button}/joint_total"] = 1; counts[f"{button}/board_pixels"] = 1
            rates[f"{button}/joint_accuracy"] = .99; rates[f"{button}/board_pixel_error"] = 0.
        for button in ("a", "b"):
            for kind in ("effect", "noeffect"):
                counts[f"{button}/{kind}_total"] = 1
                rates[f"{button}/rotation_{kind}_rendered_accuracy"] = .99
        teacher["interventions"] = {"counts": counts, "rates": rates}
        good = score(teacher, {"tile_error": .05})
        ignored = copy.deepcopy(teacher)
        for button in ("a", "b"):
            ignored["interventions"]["rates"][f"{button}/rotation_effect_rendered_accuracy"] = 0.
        self.assertGreater(score(ignored, {"tile_error": 0.}), good)
        self.assertEqual(rollout_horizon(1, good, good + 1, teacher), 2)
        self.assertEqual(rollout_horizon(1, good, good + 1, ignored), 1)
        missing = copy.deepcopy(teacher); missing["interventions"]["counts"]["b/effect_total"] = 0
        missing["interventions"]["rates"]["b/rotation_effect_rendered_accuracy"] = None
        self.assertEqual(score(missing, {"tile_error": 0.}), float("inf"))
        self.assertEqual(rollout_horizon(1, good, good + 1, missing), 1)

    def test_sequential_branch_gradients_accumulate_without_parent_state_gradient(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.register_buffer("codebook", torch.zeros(1, 4, 4, dtype=torch.uint8))
                self.weight = torch.nn.Parameter(torch.tensor(2.))
            def scene_details(self, tokens, actions, state):
                return {"state": state + 1, "write": self.weight}
            def decode_scene(self, current, details, hard):
                return torch.ones_like(current)
        model = Tiny()
        context = model.weight.reshape(1, 1, 1, 1); context.retain_grad()
        original = context.detach().clone()
        parent_loss = 3 * context.mean().square()
        bundle = {"tokens": np.zeros((6, 9, 18, 20), dtype=np.uint8),
                  "actions": np.zeros((6, 8), dtype=np.uint8), "labels": {}}
        def objective(model, current, target, details, labels, t, valid, matches):
            return details["write"].square()
        for autonomous in (False, True):
            loss = branch_objective(model, bundle, context, objective, autonomous=autonomous) / 2
            loss.backward()
        self.assertIsNone(context.grad)
        torch.testing.assert_close(context.detach(), original)
        torch.testing.assert_close(model.weight.grad, torch.tensor(4.), rtol=1e-6, atol=1e-6)
        parent_loss.backward()
        torch.testing.assert_close(model.weight.grad, torch.tensor(16.), rtol=1e-6, atol=1e-6)
        table = {(0, 32): bundle}
        ledger = resume_exposure({"step": 100}, table, "addon")
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        ledger = optimizer_step_with_exposure(optimizer, ledger, 101, [(0, 32)], table)
        self.assertEqual(ledger["bundles"], 1)
        self.assertEqual(ledger["keys"], [[0, 32]])

    def test_v4_v5_selection_migration_and_addon_provenance(self):
        model = ObjectSceneDynamics(*assets())
        checkpoint = {"config": {"protocol": "object-scene-v4-critical-events", "manifest_sha256": "parent", "annotations_sha256": "anno"},
                      "model": copy.deepcopy(model.state_dict())}
        v5 = copy.deepcopy(checkpoint)
        v5["config"].update(protocol="object-scene-v5-action-interventions", interventions_sha256="addon")
        selection_baseline(v5, checkpoint, model)
        selection_baseline(checkpoint, v5, model)
        other = copy.deepcopy(v5); other["config"]["interventions_sha256"] = "other"
        with self.assertRaises(ValueError): selection_baseline(v5, other, model)
        del other["config"]["interventions_sha256"]
        with self.assertRaises(ValueError): selection_baseline(checkpoint, other, model)

    def test_independent_selection_preserves_resume_weights_rng_and_state(self):
        model = ObjectSceneDynamics(*assets())
        config = {"protocol": "object-scene-v4-critical-events", "manifest_sha256": "dataset", "annotations_sha256": "v2"}
        resume = {"config": config, "model": copy.deepcopy(model.state_dict()), "step": 18864,
                  "optimizer": {"unchanged": 123}, "stream_cursors": [[0, 32]]}
        selection = copy.deepcopy(resume); selection["step"] = 18000
        selection["model"]["readout.bias"].add_(2)
        rng = torch.get_rng_state().clone()
        baseline = selection_baseline(resume, selection, model)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        torch.testing.assert_close(baseline.readout.bias, selection["model"]["readout.bias"])
        torch.testing.assert_close(model.readout.bias, resume["model"]["readout.bias"])
        with torch.no_grad():
            baseline.readout.bias.add_(3)
        torch.testing.assert_close(model.readout.bias, resume["model"]["readout.bias"])
        self.assertEqual(resume["step"], 18864)
        self.assertEqual(resume["optimizer"], {"unchanged": 123})
        self.assertEqual(resume["stream_cursors"], [[0, 32]])

    def test_selection_rejects_wrong_lineage_vocabulary_and_architecture(self):
        model = ObjectSceneDynamics(*assets())
        resume = {"config": {"protocol": "object-scene-v4-critical-events", "manifest_sha256": "dataset", "annotations_sha256": "v2"},
                  "model": copy.deepcopy(model.state_dict())}
        for key in ("manifest_sha256", "annotations_sha256", "protocol"):
            selection = copy.deepcopy(resume); selection["config"][key] = "other"
            with self.assertRaises(ValueError):
                selection_baseline(resume, selection, model)
        for key in ("codebook", "object_shapes", "object_tiles"):
            selection = copy.deepcopy(resume); selection["model"][key].flatten()[0] += 1
            with self.assertRaises(ValueError):
                selection_baseline(resume, selection, model)
        selection = copy.deepcopy(resume); del selection["model"]["readout.bias"]
        with self.assertRaises(RuntimeError):
            selection_baseline(resume, selection, model)

    def test_anchor_phase_covers_both_transition_parities(self):
        truth = torch.zeros(1, 18, 20, dtype=torch.long)
        generated = torch.ones_like(truth)
        for t in (1, 2):
            matches = [bool(training_input(truth, generated, t + step, 2)[1][0]) for step in (1, 2)]
            self.assertEqual(sum(matches), 1)

    def test_composition_targets_include_deposit_and_required_rewrite(self):
        model = ObjectSceneDynamics(*assets())
        current = torch.zeros(4, 18, 20, dtype=torch.long)
        current_mask = torch.zeros_like(current, dtype=torch.bool)
        current_mask[:, 2:4, 4:6] = True
        current[current_mask] = 1
        target = current.clone()
        next_mask = torch.zeros_like(current_mask)
        target[0, 2:4, 4:6] = 0
        next_mask[0, 3:5, 4:6] = True
        target[0][next_mask[0]] = 1
        next_mask[1, :2, 8:10] = True
        target[1][next_mask[1]] = 2
        current[2, 16, 2:8] = 1
        target[2, 16, 2:8] = 0
        target[3, :2, 8:10] = 2
        mode, known = composition_targets(model, current, target, current_mask, next_mask,
                                          torch.ones(4, dtype=torch.bool), torch.tensor([True, True, False, False]))
        torch.testing.assert_close(mode[:3], torch.tensor([0, 1, 2]))
        torch.testing.assert_close(known, torch.tensor([True, True, True, False]))
        _, excluded = composition_targets(model, current, target, current_mask, next_mask,
                                           torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool))
        self.assertFalse(bool(excluded.any()))

    def test_rollout_schedule_preserves_observed_training_until_all_events_ready(self):
        ready = {"addition_accuracy": .95, "erasure_accuracy": .95,
                 "mode_recall": dict.fromkeys(("copy", "compose", "rewrite"), .95),
                 "next_joint_accuracy": dict.fromkeys(("motion", "still"), .95)}
        for old, expected in ((1, 1), (2, 2), (4, 4), (8, 8)):
            self.assertEqual(rollout_horizon(old), expected)
        missing = copy.deepcopy(ready); missing["mode_recall"]["compose"] = None
        self.assertEqual(rollout_horizon(1, .9, 1., missing), 1)
        weak = copy.deepcopy(ready); weak["next_joint_accuracy"]["motion"] = .949
        self.assertEqual(rollout_horizon(1, .9, 1., weak), 1)
        self.assertEqual(rollout_horizon(1, .9, 1., ready), 2)
        self.assertEqual(rollout_horizon(2, .8, .9, ready), 4)
        self.assertEqual(rollout_horizon(4, .7, .8, ready), 8)
        self.assertEqual(rollout_horizon(8, .6, .7, ready), 8)
        self.assertEqual(rollout_horizon(1, 1., 1., ready), 1)
        self.assertEqual(rollout_horizon(1, 1.1, 1., ready), 1)
        with self.assertRaises(ValueError):
            rollout_horizon(3)

    def test_selection_penalizes_missing_deposits_and_missing_classes_are_invalid(self):
        good = {"addition_accuracy": .9, "erasure_accuracy": .9, "event_fpr": .01,
                "critical_counts": {"mode_confusion": [[9, 1, 0], [1, 9, 0], [0, 1, 9]], "motion": 10, "still": 10},
                "mode_recall": dict.fromkeys(("copy", "compose", "rewrite"), .9),
                "next_joint_accuracy": dict.fromkeys(("motion", "still"), .9)}
        static = copy.deepcopy(good)
        static["mode_recall"]["compose"] = 0
        static["mode_recall"]["rewrite"] = 0
        self.assertGreater(score(static, {"tile_error": .01}), score(good, {"tile_error": .1}))
        for category, key in (("mode_recall", "compose"), ("next_joint_accuracy", "motion")):
            missing = copy.deepcopy(good); missing[category][key] = None
            self.assertEqual(score(missing, {"tile_error": .01}), float("inf"))
        self.assertEqual(score({"addition_accuracy": 1}, {"tile_error": 0}), float("inf"))
        missing_counts = copy.deepcopy(good); missing_counts["critical_counts"]["motion"] = 0
        self.assertEqual(score(missing_counts, {"tile_error": 0}), float("inf"))

    def test_critical_events_detect_motion_deposit_rewrite_and_still(self):
        model = ObjectSceneDynamics(*assets())
        current = torch.zeros(4, 18, 20, dtype=torch.long)
        current[:, 2:4, 4:6] = 1
        old = torch.tensor([[2, 2], [2, 3], [3, 2], [3, 3]])
        poses = torch.stack((old + torch.tensor([1, 0]), old + torch.tensor([-2, 4]), old, old))
        target = current.clone(); target[0, 2:4, 4:6] = 0
        for sample in (0, 1):
            target[sample, poses[sample, :, 0], poses[sample, :, 1] + 2] = 1
        current[2, 16, 2:8] = 1  # Visible canvas removal beyond one overlay requires rewrite.
        labels = {"shape_id": torch.zeros(4, 2, dtype=torch.long),
                  "evidence": torch.ones(4, 2, dtype=torch.long),
                  "cells": torch.stack((old.expand(4, -1, -1), poses), 1)}
        joint = torch.full((4, 2, 180), -torch.inf)
        for sample in range(4):
            origin = poses[sample].amin(0)
            joint[sample, 0, origin[0] * 10 + origin[1]] = 0
        modes = torch.tensor([[9., 0, 0], [0., 9, 0], [0., 0, 9], [9., 0, 0]])
        counts = critical_event_counts(model, current, target, {"mode": modes, "next": {"joint": joint}}, labels)
        self.assertEqual(counts["mode_confusion"], [[2, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual((counts["motion"], counts["still"]), (2, 2))
        rates = critical_event_rates(counts)
        self.assertEqual(rates["next_joint_accuracy"], {"motion": 1., "still": 1.})
        self.assertEqual(rates["mode_recall"], {"copy": 1., "compose": 1., "rewrite": 1.})
        counts["mode_confusion"][1] = [0, 0, 0]
        self.assertIsNone(critical_event_rates(counts)["mode_recall"]["compose"])

    def test_generated_tokens_are_reinjected_and_object_labels_remain_causal(self):
        truth = torch.zeros(2, 18, 20, dtype=torch.int64)
        generated = truth.clone(); generated[0] = 1
        first, matches = training_input(truth, None, 0, 2)
        self.assertIs(first, truth)
        self.assertTrue(torch.all(matches))
        own_input, matches = training_input(truth, generated, 1, 2)
        self.assertIs(own_input, generated)
        torch.testing.assert_close(matches, torch.tensor([False, True]))
        labels = {"shape_id": torch.zeros(2, 2, dtype=torch.int64),
                  "evidence": torch.ones(2, 2, dtype=torch.int64),
                  "cells": torch.tensor([[[[2, 2], [2, 3], [3, 2], [3, 3]]] * 2] * 2)}
        positive, _ = label_mask(labels, 1, torch.tensor([True, True]) & matches)
        torch.testing.assert_close(positive, torch.tensor([False, True]))
        anchored, anchored_matches = training_input(truth, generated, 2, 2)
        self.assertIs(anchored, truth)
        self.assertTrue(torch.all(anchored_matches))
        future_target = torch.full_like(truth, 3)
        future_labels = torch.full_like(labels["shape_id"], -1)
        self.assertFalse(torch.equal(own_input, future_target))
        self.assertTrue(torch.all(future_labels == -1))
        # The selector has no target/annotation argument; changing future labels cannot feed the input.
        repeated, repeated_matches = training_input(truth, generated, 1, 2)
        torch.testing.assert_close(repeated, own_input)
        torch.testing.assert_close(repeated_matches, matches)

    def test_labels_align_with_contiguous_inputs_targets_and_padding(self):
        observations = np.zeros((81, 18, 20), dtype=np.uint8)
        actions = np.arange(80, dtype=np.uint8) % 8
        annotations = [{"cells": np.repeat(np.arange(81, dtype=np.int16)[:, None, None], 8, axis=1).reshape(81, 4, 2),
                        "shape_id": np.arange(81, dtype=np.int32), "track": np.arange(81, dtype=np.int32),
                        "evidence": np.ones(81, dtype=np.uint8)}]
        streams = StreamWindows([(observations, actions)], 1, random.Random(3), length=8)
        first = streams.next()
        labels = window_labels(streams.cursors, first[2], annotations)
        torch.testing.assert_close(labels["shape_id"][0], torch.arange(32, 41))
        second = streams.next()
        next_labels = window_labels(streams.cursors, second[2], annotations)
        torch.testing.assert_close(labels["cells"][:, -1], next_labels["cells"][:, 0])
        short = StreamWindows([(observations[:36], actions[:35])], 1, random.Random(3), length=8)
        tail = short.next()
        padded = window_labels(short.cursors, tail[2], annotations)
        torch.testing.assert_close(padded["shape_id"][0, :4], torch.arange(32, 36))
        self.assertTrue(torch.all(padded["shape_id"][0, 4:] == -1))
        self.assertTrue(torch.all(padded["evidence"][0, 4:] == 0))

    def test_optimizer_upgrade_preserves_core_moments_and_adds_fresh_heads(self):
        book, shapes, tiles = assets()
        old = ObservedHistoryDynamics(book)
        old_optimizer = torch.optim.AdamW(old.parameters(), lr=3e-4, weight_decay=1e-5)
        for i, parameter in enumerate(old.parameters()):
            parameter.grad = torch.full_like(parameter, (i % 3 + 1) * 0.01)
        old_optimizer.step()
        original = copy.deepcopy(old_optimizer.state_dict())
        model = ObjectSceneDynamics(book, shapes, tiles)
        model.load_state_dict(upgrade_scene_weights(old.state_dict(), shapes, tiles))
        migrated = append_scene_optimizer(original, model)
        optimizer = scene_optimizer(model)
        optimizer.load_state_dict(migrated)
        parameters = list(model.parameters())
        for number, identifier in enumerate(original["param_groups"][0]["params"]):
            for key, value in original["state"][identifier].items():
                torch.testing.assert_close(optimizer.state[parameters[number]][key], value)
        self.assertEqual(optimizer.param_groups[0]["lr"], 3e-4)
        self.assertEqual(optimizer.param_groups[1]["lr"], 3e-3)
        self.assertEqual(len(original["param_groups"]), 1)
        self.assertTrue(all(parameter not in optimizer.state for parameter in parameters[model.base_parameter_count:]))
        invalid = copy.deepcopy(original); invalid["param_groups"].append(copy.deepcopy(invalid["param_groups"][0]))
        with self.assertRaises(ValueError):
            append_scene_optimizer(invalid, model)

    def test_visual_absence_requires_no_representable_texture_on_board(self):
        tokens = torch.full((3, 18, 20), 3, dtype=torch.int64)
        tokens[0, 5, 5] = 1
        tokens[1, 0, 0] = 1  # The HUD must not prevent a board absence label.
        evidence = torch.tensor([3, 3, 0])
        actual = visual_absence(tokens, evidence, torch.tensor([True, True, True]), assets()[2])
        torch.testing.assert_close(actual, torch.tensor([False, True, False]))

    def test_loss_masks_impossible_targets_and_phase_uncertainty(self):
        torch.manual_seed(43)
        model = ObjectSceneDynamics(*assets())
        current = torch.zeros(4, 18, 20, dtype=torch.int64)
        target = current.clone()
        current[0, 2:4, 4:6] = 1; target[0, 3:5, 4:6] = 1
        current[1, 6, 6] = target[1, 6, 6] = 1
        current[3] = target[3] = 3
        labels = {"cells": torch.full((4, 2, 4, 2), -1, dtype=torch.int64),
                  "shape_id": torch.full((4, 2), -1, dtype=torch.int64),
                  "evidence": torch.zeros(4, 2, dtype=torch.int64), "track": torch.full((4, 2), -1)}
        labels["cells"][0, 0] = model.object_shapes[0] + torch.tensor([2, 2])
        labels["cells"][0, 1] = model.object_shapes[0] + torch.tensor([3, 2])
        labels["shape_id"][0] = 0; labels["evidence"][0] = 1
        # Masked geometric targets have -inf probability; NLL must not evaluate them.
        labels["cells"][1:3] = model.object_shapes[0] + torch.tensor([17, 9])
        labels["shape_id"][1:3] = 0; labels["evidence"][1:3] = 3
        labels["evidence"][3] = 3
        valid, matches = torch.tensor([True, True, False, True]), torch.tensor([True, False, True, True])
        positive, mask = label_mask(labels, 0, valid & matches)
        torch.testing.assert_close(positive, torch.tensor([True, False, False, False]))
        self.assertEqual(int(mask[0].sum()), 4)
        details = model.scene_details(current, torch.tensor([1, 2, 3, 0]))
        loss = scene_loss(model, current, target, details, labels, 0, valid, matches)
        self.assertTrue(torch.isfinite(loss))
        ignored = {name: value.clone() for name, value in labels.items()}
        ignored["cells"][1:3] = -1; ignored["shape_id"][1:3] = -1
        torch.testing.assert_close(loss, scene_loss(model, current, target, details, ignored, 0, valid, matches))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_raw_and_full_scene_loader_and_keyboard_session(self):
        from play import NeuralSession
        model = ObjectSceneDynamics(*assets()).eval()
        tokens = torch.arange(360).reshape(1, 18, 20) % 4
        pixels = decode_tokens(tokens, model.codebook)
        context = json.loads((ROOT / "context.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            np.savez_compressed(directory / "seed.npz", frames=np.repeat(pixels[:, 0].numpy(), 3, axis=0),
                                actions=np.zeros(2, dtype=np.uint8), metadata=json.dumps({"context": context, "source": "capture"}))
            for wrapped in (False, True):
                path = directory / f"scene-{wrapped}.pt"
                torch.save({"model": model.state_dict()} if wrapped else model.state_dict(), path)
                loaded = load_tile_model(path)
                self.assertIs(type(loaded), ObjectSceneDynamics)
                expected, _ = model.predict_frame(pixels, torch.tensor([1]))
                actual, _ = loaded.predict_frame(pixels, torch.tensor([1]))
                torch.testing.assert_close(actual, expected)
                session = NeuralSession(path, directory / "seed.npz")
                self.assertIs(type(session.model), ObjectSceneDynamics)
                initial = session.reset()["frame"]
                self.assertEqual(session.step("left")["steps"], 1)
                self.assertEqual(session.reset()["frame"], initial)

    def test_extended_raw_full_load_and_keyboard_session_reset36(self):
        from play import NeuralSession
        model = expand_scene_weights(ObjectSceneDynamics(*assets()).state_dict()).eval()
        pixels = decode_tokens(torch.zeros(1, 18, 20, dtype=torch.long), model.codebook)
        captures = np.repeat(pixels[:, 0].numpy(), 3, axis=0)
        with tempfile.TemporaryDirectory() as directory:
            for wrapped in (False, True):
                path = Path(directory) / f"extended-{wrapped}.pt"
                torch.save({"model": model.state_dict()} if wrapped else model.state_dict(), path)
                loaded = load_tile_model(path)
                self.assertIsInstance(loaded, ExtendedSceneDynamics)
                expected, state = model.predict_frame(pixels, torch.tensor([1]))
                actual, actual_state = loaded.predict_frame(pixels, torch.tensor([1]))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_state, state, rtol=0, atol=0)
                with patch("play.read_episode", return_value=(captures, np.zeros(2, dtype=np.uint8), {"source": "capture"})):
                    session = NeuralSession(path, Path("unused-seed.npz"))
                    self.assertEqual(session.state.shape, (1, 36, 18, 20))
                    first = session.reset()["frame"]
                    session.step("left")
                    self.assertEqual(session.state.shape[1], 36)
                    self.assertEqual(session.reset()["frame"], first)


def dense_train_windows(episodes, table, batch=4):
    windows = []
    for episode, (_, actions) in enumerate(episodes):
        if len(actions) < 64:
            continue
        anchors = sorted(anchor for index, anchor in table if index == episode)
        starts = {max(32, min(anchor, len(actions) - 32)) for anchor in anchors}
        if starts:
            start = max(starts, key=lambda s: (sum(s <= anchor < s + 32 for anchor in anchors), -s))
            windows.append((sum(start <= anchor < start + 32 for anchor in anchors), episode, start))
    chosen = sorted(windows, key=lambda item: (-item[0], item[1], item[2]))[:batch]
    if len(chosen) != batch:
        raise ValueError("dense self-check needs four train episodes with interventions")
    return chosen


def real_self_check(checkpoint_path, output, threads, intervention_path=None, migrate_from=None, lr_scale=1., branch_inputs="teacher", expand_memory=False, branch_supervision="uniform-next", relative_motion=False, local_pose=False, pose_rerank=False, action_invariant_current=False, learned_overlap=False, descent_ranking_weight=0., spawn_preview=False):
    """One unchanged final-objective update on the preserved real stream; discard weights."""
    if output.exists():
        raise FileExistsError(output)
    validate_branch_supervision(branch_inputs, branch_supervision)
    if sum((relative_motion, expand_memory, local_pose, pose_rerank, action_invariant_current, learned_overlap, spawn_preview)) > 1:
        raise ValueError("choose at most one scene architecture migration")
    torch.set_num_threads(threads); torch.manual_seed(1042029)
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw, episodes, annotations, shapes, tiles, digest, annotation_digest = load_scene_data(checkpoint)
    if checkpoint["config"]["batch"] != 4:
        raise ValueError("self-check requires the preserved batch of four streams")
    upgrading = "_object_scene" not in checkpoint["model"]
    weights = upgrade_scene_weights(checkpoint["model"], shapes, tiles) if upgrading else checkpoint["model"]
    if not torch.equal(weights["object_shapes"], shapes) or not torch.equal(weights["object_tiles"], tiles):
        raise ValueError("changed object vocabulary on resume")
    adding_parameters = expand_memory or relative_motion or local_pose or pose_rerank or learned_overlap or spawn_preview
    migrating_family = adding_parameters or action_invariant_current
    if migrating_family and upgrading:
        raise ValueError("migration requires legacy scene checkpoint")
    if spawn_preview:
        model = spawn_preview_scene_weights(weights)
    elif learned_overlap:
        model = learned_overlap_scene_weights(weights)
    elif action_invariant_current:
        model = action_invariant_current_scene_weights(weights)
    elif pose_rerank:
        model = pose_rerank_scene_weights(weights)
    elif local_pose:
        model = footprint_scene_weights(weights)
    elif relative_motion:
        model = relative_scene_weights(weights)
    elif expand_memory:
        model = expand_scene_weights(weights)
    else:
        model = scene_from_weights(weights)
    addon_directory = intervention_directory(checkpoint, intervention_path)
    interventions, addon_digest = load_action_interventions(checkpoint, model, episodes, annotations, digest, annotation_digest,
                                                           addon_directory, migrate_from)
    migration = verify_extension(addon_directory, intervention_directory(checkpoint, migrate_from)) if migrate_from else None
    coverage = {split: composition_coverage(model, episodes[split], annotations[split]) for split in ("train", "validation")}
    if any(min(counts[kind] for kind in ("copy", "compose", "rewrite")) < 1 for counts in coverage.values()):
        raise ValueError(f"missing composition supervision: {coverage}")
    mode_train_coverage, mode_weights = training_mode_balance(model, episodes, annotations, interventions)
    relative_coverage = relative_supervision_coverage(model, episodes, annotations, interventions) if isinstance(model, RelativeSceneDynamics) else None
    parent_loss_fn = partial(scene_loss, mode_weights=mode_weights["parent"], descent_ranking_weight=descent_ranking_weight)
    branch_loss_fn = partial(scene_loss, mode_weights=mode_weights["interventions"])
    optimizer = scene_optimizer(model)
    optimizer_weights = append_scene_optimizer(checkpoint["optimizer"], model) if upgrading else checkpoint["optimizer"]
    if adding_parameters: optimizer_weights = append_memory_optimizer(optimizer_weights, model)
    optimizer.load_state_dict(optimizer_weights)
    learning_rates = scale_optimizer_lr(optimizer, lr_scale)
    identifiers = [identifier for group in checkpoint["optimizer"]["param_groups"] for identifier in group["params"]]
    for number, identifier in enumerate(identifiers):
        for key, value in checkpoint["optimizer"]["state"].get(identifier, {}).items():
            torch.testing.assert_close(optimizer.state[list(model.parameters())[number]][key], value)
    rng = random.Random(); rng.setstate(checkpoint["python_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    streams = StreamWindows(episodes["train"], 4, rng)
    streams.cursors = copy.deepcopy(checkpoint["stream_cursors"])
    state = (expand_scene_state(checkpoint["stream_state"]) if expand_memory else checkpoint["stream_state"]).detach().clone()
    if tuple(state.shape) != (4, model.state_channels, 18, 20): raise ValueError("invalid preserved scene state")
    frames, actions, valid, reset, prefix, prefix_actions = streams.next()
    labels = window_labels(streams.cursors, valid, annotations["train"])
    anchors = windowanchors(streams.cursors, valid, interventions["train"])
    anchor_count = sum(len(bundles) for bundles in anchors.values())
    setup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    chosen = reset.nonzero(as_tuple=True)[0]
    if len(chosen):
        warm = None
        with torch.no_grad():
            for t in range(32):
                _, warm = model(prefix[chosen, t], prefix_actions[chosen, t], warm)
        state[chosen] = warm
    dense_windows = []
    dense_context_seconds = 0.
    if json.loads((addon_directory / "manifest.json").read_bytes())["version"] == 2:
        context_started = time.perf_counter()
        dense_windows = dense_train_windows(episodes["train"], interventions["train"])
        frame_batch, action_batch, contexts, cursors = [], [], [], []
        with torch.no_grad():
            for count, episode, start in dense_windows:
                codes, commands = episodes["train"][episode]
                context = None
                for t in range(start):
                    context = model.advance(torch.from_numpy(codes[t].copy())[None].long(), torch.tensor([int(commands[t])]), context)
                contexts.append(context)
                frame_batch.append(codes[start:start + 33]); action_batch.append(commands[start:start + 32])
                cursors.append((episode, start + 32))
        frames = torch.from_numpy(np.stack(frame_batch)).long()
        actions = torch.from_numpy(np.stack(action_batch)).long()
        valid = torch.ones(4, 32, dtype=torch.bool)
        state = torch.cat(contexts).detach()
        labels = window_labels(cursors, valid, annotations["train"])
        anchors = windowanchors(cursors, valid, interventions["train"])
        anchor_count = sum(len(bundles) for bundles in anchors.values())
        dense_context_seconds = time.perf_counter() - context_started
        setup_seconds += dense_context_seconds
        started += dense_context_seconds
    optimizer.zero_grad(set_to_none=True)
    total, generated = torch.zeros(()), None
    branch_value = 0.
    fallback = anchor_count == 0
    actual_bundles = anchor_count
    if fallback:
        if not interventions["train"]:
            raise ValueError("real self-check requires at least one train intervention")
        (episode, anchor), bundle = next(iter(sorted(interventions["train"].items())))
        causal_context = None
        codes, commands = episodes["train"][episode]
        with torch.no_grad():
            for t in range(anchor):
                details = model.scene_details(torch.from_numpy(codes[t].copy())[None].long(), torch.tensor([int(commands[t])]), causal_context)
                causal_context = details["state"]
        for autonomous, weight in branch_passes(branch_inputs):
            objective = weight * branch_objective(model, bundle, causal_context, branch_loss_fn, autonomous=autonomous, supervision=branch_supervision)
            if not torch.isfinite(objective):
                raise RuntimeError("real intervention loss is not finite")
            branch_value += float(objective.detach()); objective.backward(); del objective
        actual_bundles = 1
    horizon = rollout_horizon(checkpoint.get("horizon", 1))
    for t in range(32):
        for slot, bundle in anchors.get(t, []):
            for autonomous, weight in branch_passes(branch_inputs):
                objective = weight * branch_objective(model, bundle, state[slot:slot + 1], branch_loss_fn,
                                                 autonomous=autonomous, supervision=branch_supervision) / anchor_count
                if not torch.isfinite(objective):
                    raise RuntimeError("real intervention loss is not finite")
                branch_value += float(objective.detach()); objective.backward(); del objective
        current, current_matches = training_input(frames[:, t], generated, t + checkpoint["step"] + 1, horizon)
        details = model.scene_details(current, actions[:, t], state)
        total = total + parent_loss_fn(model, current, frames[:, t + 1], details, labels, t, valid[:, t],
                                   current_matches, actions=actions[:, t])
        state = details["state"]
        generated = model.decode_scene(current, details, hard=True).detach()
    loss = total / 32
    if not torch.isfinite(loss):
        raise RuntimeError("real objective loss is not finite")
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True); optimizer.step()
    migrated_model = isinstance(model, (ExtendedSceneDynamics, RelativeSceneDynamics, FootprintSceneDynamics,
                                       PoseRerankSceneDynamics, ActionInvariantCurrentSceneDynamics, LearnedOverlapSceneDynamics))
    if migrated_model:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "extended-selfcheck.pt"
            torch.save({"model": model.state_dict()}, path)
            reloaded = load_tile_model(path)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, reloaded.state_dict()[name], rtol=0, atol=0)
            # The training instance was train(), whereas load_tile_model returns eval().
            # MHA chooses a different execution path in those modes, even with dropout0.
            model.eval(); reloaded.eval()
            replay_tokens = frames[:, 0].contiguous()
            replay_actions = actions[:, 0].contiguous()
            replay_state = state.detach().contiguous()
            with torch.inference_mode():
                before = model.scene_details(replay_tokens, replay_actions, replay_state)
                after = reloaded.scene_details(replay_tokens, replay_actions, replay_state)
            torch.testing.assert_close(before["state"], after["state"], rtol=0, atol=0)
            torch.testing.assert_close(before["next"]["joint"], after["next"]["joint"], rtol=0, atol=0)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"status": "real-objective-single-update-passed-weights-discarded", "loss": float(loss.detach()) + branch_value,
              "parent_loss": float(loss.detach()), "intervention_loss": branch_value,
              "interventions_sha256": addon_digest, "intervention_bundles": actual_bundles,
              "interventions_path": str(addon_directory.relative_to(ROOT.resolve())), "interventions_migration": migration,
              "dense_train_windows": dense_windows, "profile_context": "observed-full-prefix" if dense_windows else "preserved-stream",
              "dense_context_setup_seconds": dense_context_seconds,
              "intervention_context_replayed_if_window_empty": fallback, "labels_never_neural_inputs": True,
              "intervention_objective": branch_inputs, "branch_passes": list(branch_passes(branch_inputs)),
              "branch_supervision": branch_supervision, "next_label_train_coverage": next_label_coverage(interventions["train"]),
              "lr_before": learning_rates["before"], "lr_after": learning_rates["after"], "lr_scale": lr_scale,
              "descent_ranking_weight": descent_ranking_weight,
              "batch": 4, "unroll": 32, "horizon": horizon, "threads": threads,
              "model_family": type(model).__name__, "state_channels": model.state_channels, "memory_expanded": expand_memory,
              "relative_motion": relative_motion, "initial_prediction_parity_expected": not relative_motion,
              "relative_auxiliary_train_coverage": relative_coverage,
              "extended_save_reload_exact": migrated_model,
              "extended_reload_comparison_mode": "both-eval-inference-identical-contiguous-inputs; state_dict-bit-exact" if migrated_model else None,
              "valid_transitions": int(valid.sum()), "core_optimizer_preserved": True,
              "manifest_sha256": digest, "annotations_sha256": annotation_digest, "mode_supervision": coverage,
              "mode_training_coverage": mode_train_coverage, "mode_class_weights": mode_weights,
              "setup_seconds": setup_seconds, "step_seconds": time.perf_counter() - started,
              "peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        parser = argparse.ArgumentParser(description="Profile one real scene-training objective update, with no checkpoint export")
        parser.add_argument("--self-check", action="store_true")
        parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/observed-last-checkpoint.pt")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--threads", type=int, default=2)
        parser.add_argument("--interventions", type=Path)
        parser.add_argument("--migrate-interventions-from", type=Path)
        parser.add_argument("--lr-scale", type=float, default=1.)
        parser.add_argument("--branch-inputs", choices=("teacher", "mixed"), default="teacher")
        parser.add_argument("--branch-supervision", choices=("first-effect", "uniform-next"), default="uniform-next")
        parser.add_argument("--expand-memory", action="store_true")
        parser.add_argument("--relative-motion", action="store_true")
        parser.add_argument("--local-pose", action="store_true")
        parser.add_argument("--pose-rerank", action="store_true")
        parser.add_argument("--action-invariant-current", action="store_true")
        parser.add_argument("--learned-overlap", action="store_true")
        parser.add_argument("--spawn-preview", action="store_true")
        parser.add_argument("--descent-ranking-weight", type=float, default=0.)
        args = parser.parse_args()
        if args.threads < 1:
            parser.error("positive thread count required")
        real_self_check(args.checkpoint, args.output, args.threads, args.interventions, args.migrate_interventions_from,
                        args.lr_scale, args.branch_inputs, args.expand_memory, args.branch_supervision, args.relative_motion,
                        args.local_pose, args.pose_rerank, args.action_invariant_current, args.learned_overlap,
                        args.descent_ranking_weight, args.spawn_preview)
    else:
        unittest.main()
