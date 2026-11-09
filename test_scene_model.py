import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scene_model import (ObjectSceneDynamics, ExtendedSceneDynamics, RelativeSceneDynamics, FootprintSceneDynamics,
                         ActionInvariantCurrentSceneDynamics, action_invariant_current_scene_weights,
                         LearnedOverlapSceneDynamics, learned_overlap_scene_weights,
                         SpawnPreviewSceneDynamics, spawn_preview_scene_weights,
                         MotionClockSceneDynamics, motion_clock_scene_weights,
                         ConsistencySceneDynamics, FactorizedPreviewRenderSceneDynamics, consistency_scene_weights,
                         ModeCorrectionSceneDynamics, mode_correction_scene_weights,
                         SpatialModeSceneDynamics, spatial_mode_scene_weights,
                         RowTransportSceneDynamics, row_transport_scene_weights,
                         ContactModeSceneDynamics, contact_mode_scene_weights,
                         JointEventSceneDynamics, joint_event_scene_weights,
                         TimedEventSceneDynamics, timed_event_scene_weights,
                         CanvasOverlapSceneDynamics, canvas_overlap_scene_weights,
                         TrackedCanvasSceneDynamics, tracked_canvas_scene_weights,
                         ContinuationSceneDynamics, continuation_scene_weights,
                         RelativeProposalSceneDynamics, relative_proposal_scene_weights,
                         OrientedProposalSceneDynamics, oriented_proposal_scene_weights,
                         RewriteEvidenceSceneDynamics, rewrite_evidence_scene_weights,
                         FeedbackCarrySceneDynamics, feedback_carry_scene_weights,
                         ReplayEventSceneDynamics, replay_event_scene_weights,
                         ClearRoutingSceneDynamics, clear_routing_scene_weights,
                         ClearAnimationSceneDynamics, clear_animation_scene_weights,
                         ClearDurationSceneDynamics, clear_duration_scene_weights,
                         TerminalSceneDynamics, terminal_scene_weights,
                         TerminalContactSceneDynamics, terminal_contact_scene_weights,
                         ClearOnsetSceneDynamics, clear_onset_scene_weights,
                         ComposeContactSceneDynamics, compose_contact_scene_weights,
                         RewriteModeSceneDynamics, rewrite_mode_scene_weights,
                         CurrentReadoutSceneDynamics, current_readout_scene_weights,
                         RowPreserveSceneDynamics, row_preserve_scene_weights,
                         PoseRerankSceneDynamics, pose_rerank_scene_weights,
                         expand_scene_state, expand_scene_weights, footprint_scene_weights, relative_scene_weights,
                         scene_from_weights, upgrade_scene_weights)
from token_model import ObservedHistoryDynamics, decode_tokens


def assets():
    book = torch.stack((torch.full((4, 4), 3, dtype=torch.uint8),
                        torch.zeros((4, 4), dtype=torch.uint8), torch.ones((4, 4), dtype=torch.uint8)))
    shapes = torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 0], [1, 0], [2, 0], [2, 1]]])
    return book, shapes, torch.tensor([1, 2])


def chosen(model, kind, row, column, visible=True):
    joint = torch.full((1, len(model.object_shapes), 180), -torch.inf)
    joint[0, kind, row * 10 + column] = 0
    return {"joint": joint, "visible": torch.tensor([[20.0 if visible else -20.0]])}


def synthetic_details(model, mode=0):
    writes = torch.full((1, len(model.codebook), 18, 20), -20.0)
    writes[:, 2] = 20
    gate = torch.full((1, 1, 18, 20), -20.0)
    gate[:, :, 0, 0] = 20
    modes = torch.full((1, 3), -20.0); modes[:, mode] = 20
    return {"write": writes, "gate": gate, "mode": modes,
            "current": chosen(model, 0, 2, 2), "next": chosen(model, 1, 7, 5)}


class SceneModelTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_rewrite_evidence_only_changes_the_rewrite_score(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = OrientedProposalSceneDynamics(*assets(), patterns)
        model = scene_from_weights(rewrite_evidence_scene_weights(base.state_dict()).state_dict())
        self.assertIsInstance(model, RewriteEvidenceSceneDynamics)
        frame = torch.full((1, 18, 20), base.blank_id, dtype=torch.long)
        previous = frame.clone()
        previous[:, -1, 2:12] = base.object_tiles[0]
        action = torch.tensor([0])
        state = base.advance_observed(previous, action)
        before = base.scene_details(frame, action, state)
        after = model.scene_details(frame, action, state)
        torch.testing.assert_close(model.rewrite_features(after)[0, :3], torch.tensor([0., 0., 1.]))
        torch.testing.assert_close(after["event_scores"][:, :2], before["event_scores"][:, :2], rtol=0, atol=0)
        valid = model.valid_poses.flatten()
        self.assertTrue((after["event_scores"][:, 2, valid] < before["event_scores"][:, 2, valid]).all())

    @torch.no_grad()
    def test_feedback_carry_commits_one_render_and_only_adjusts_changed_boards(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = RewriteEvidenceSceneDynamics(*assets(), patterns)
        model = feedback_carry_scene_weights(base.state_dict())
        model.carry_agreement.weight.fill_(-2)
        model = scene_from_weights(model.state_dict())
        self.assertIsInstance(model, FeedbackCarrySceneDynamics)
        frame = torch.full((1, 18, 20), base.blank_id, dtype=torch.long)
        action = torch.tensor([0])
        with patch.object(model, "decode_scene", wraps=model.decode_scene) as decode:
            prediction, state = model.predict_tokens(frame, action)
            decode.assert_called_once()
        previous = base.advance_observed(frame, action)
        torch.testing.assert_close(state[:, :-1], previous[:, :-1])
        self.assertEqual(state[:, -1, 0, 0].item(), previous[:, -1, 0, 0].item())
        self.assertEqual(state[:, -1, 0, 1].item(), -2)
        self.assertTrue(torch.equal(state[:, -1, :, 2:12], prediction[:, :, 2:12]))
        torch.testing.assert_close(model.advance_observed(frame, action), state)
        features = torch.tensor([[0., 0., 1.]])
        changed = prediction.clone()
        changed[:, 0, 0] = (prediction[:, 0, 0] + 1) % len(model.codebook)
        torch.testing.assert_close(model.carry_logit(changed, state, features), base.pose_carry(features))
        changed[:, 0, 2] = (prediction[:, 0, 2] + 1) % len(model.codebook)
        self.assertLess(model.carry_logit(changed, state, features).item(), base.pose_carry(features).item())
        self.assertEqual(model.agreement_features(prediction, previous, features)[0, 1].item(), 1)

    @torch.no_grad()
    def test_replay_events_preserve_initial_render_and_exact_grouped_scores(self):
        from train_event_replay import compress_scores

        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = FeedbackCarrySceneDynamics(*assets(), patterns).eval()
        model = scene_from_weights(replay_event_scene_weights(base.state_dict()).state_dict()).eval()
        self.assertIsInstance(model, ReplayEventSceneDynamics)
        frame = torch.full((2, 18, 20), base.blank_id, dtype=torch.long)
        action = torch.tensor([0, 4])
        before, state = base.predict_tokens(frame, action)
        after, carried = model.predict_tokens(frame, action)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        torch.testing.assert_close(carried, state, rtol=0, atol=0)
        details = base.scene_details(frame, action)
        current = details["current"]["joint"].flatten(1).argmax(1)
        scores = details["event_scores"]
        model.replay_calibration[-1].bias.copy_(torch.linspace(-1, 1, 19))
        offsets = model.replay_offsets(*model.replay_features(details, action, current))
        mode = torch.tensor([0, 2])
        pose = scores[torch.arange(len(mode)), mode].argmax(1)
        masses, peaks, bins, gold = compress_scores(model, scores, current, mode, pose)
        bins_for_poses = model.replay_pose_bins(current)
        copy = scores[:, 0] + offsets.gather(1, bins_for_poses)
        full = torch.cat((copy[:, None], scores[:, 1:] + offsets[:, -2:, None]), dim=1)
        torch.testing.assert_close((masses + offsets).logsumexp(1), full.flatten(1).logsumexp(1))
        torch.testing.assert_close((peaks + offsets).amax(1), full.flatten(1).amax(1))
        torch.testing.assert_close(gold + offsets[torch.arange(2), bins], full[torch.arange(2), mode, pose])

    @torch.no_grad()
    def test_clear_route_separates_render_gate_and_pose_and_respects_visibility(self):
        from train_clear_routing import route_matches, route_log_prob

        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = ReplayEventSceneDynamics(*assets(), patterns).eval()
        model = scene_from_weights(clear_routing_scene_weights(base.state_dict()).state_dict()).eval()
        self.assertIsInstance(model, ClearRoutingSceneDynamics)
        frame = torch.full((2, 18, 20), base.blank_id, dtype=torch.long)
        frame[:, -1, 2:12] = base.object_tiles[0]
        action = torch.tensor([0, 4])
        before, state = base.predict_tokens(frame, action)
        after, carried = model.predict_tokens(frame, action)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        torch.testing.assert_close(carried, state, rtol=0, atol=0)

        base_details = base.scene_details(frame, action)
        model.clear_routing.bias[3] = 20
        details = model.scene_details(frame, action)
        torch.testing.assert_close(details["event_scores"], base_details["event_scores"], rtol=0, atol=0)
        torch.testing.assert_close(details["next"]["joint"], base_details["next"]["joint"], rtol=0, atol=0)
        torch.testing.assert_close(details["row_transport_gate"], base_details["row_transport_gate"], rtol=0, atol=0)
        self.assertTrue((details["render_transport_gate"] > 0).all())
        _, rewrite = model._hard_rewrite_candidates(frame, details, details["gate"] - model.gate_logit_offset)
        self.assertTrue(torch.equal(rewrite[:, :, 2:12], model.transported_rows(frame, details)))
        details["next"]["visible"].fill_(-20)
        matches, preferred = route_matches(model, frame, details, frame[:, :, 2:12], frame)
        self.assertTrue(matches[:, 1].all())
        self.assertTrue(torch.equal(matches, preferred))
        log_prob = route_log_prob(details["mode"], details["render_transport_gate"])
        torch.testing.assert_close(log_prob.logsumexp(1), torch.zeros(2), atol=1e-6, rtol=0)

    @torch.no_grad()
    def test_clear_animation_latches_rows_until_learned_spawn(self):
        book, shapes, tiles = assets()
        book = torch.cat((book, torch.full((1, 4, 4), 2, dtype=torch.uint8)))
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = ClearRoutingSceneDynamics(book, shapes, tiles, patterns).eval()
        model = scene_from_weights(clear_animation_scene_weights(base.state_dict(), 3).state_dict()).eval()
        self.assertIsInstance(model, ClearAnimationSceneDynamics)
        frame = torch.full((1, 18, 20), base.blank_id, dtype=torch.long)
        frame[:, -1, 2:12] = base.object_tiles[0]
        command = torch.tensor([0])
        expected, state = base.predict_tokens(frame, command)
        initial, extended = model.predict_tokens(frame, command)
        torch.testing.assert_close(initial, expected, rtol=0, atol=0)
        torch.testing.assert_close(extended[:, :-1], state, rtol=0, atol=0)
        self.assertEqual(extended[0, -1, 0, 0].item(), 0)

        model.animation_onset[-1].bias.fill_(20)
        model.animation_end[-1].weight.zero_()
        model.animation_end[-1].bias.fill_(-20)
        model.animation_rows[-1].weight.zero_()
        model.animation_rows[-1].bias.fill_(-20)
        model.animation_rows[-1].bias[10] = 20
        flash, phase = model.predict_tokens(frame, command)
        self.assertTrue((flash[:, :, 2:12] == 3).all())
        self.assertTrue(torch.equal(flash[:, :, 12:], frame[:, :, 12:]))
        self.assertTrue(torch.equal(phase[:, -1, :, 2:12], frame[:, :, 2:12]))
        self.assertEqual(phase[0, -2, 0, 0].item(), -1)
        self.assertEqual(phase[0, -1, 0, 0].item(), 1)
        model.animation_onset[-1].bias.fill_(-20)
        model.animation_end[-1].bias.fill_(20)
        model.animation_rows[-1].bias[10] = -20
        model.animation_rows[-1].bias[4] = 20
        details = model.scene_details(flash, command, phase)
        background = model._hard_background(flash, details, details["gate"] - model.gate_logit_offset)
        self.assertTrue(torch.equal(background[:, :, 2:12], frame[:, :, 2:12]))
        self.assertTrue((details["next"]["visible"] > 0).all())
        _, resolved = model.predict_tokens(flash, command, phase)
        self.assertEqual(resolved[0, -1, 0, 0].item(), 0)
        self.assertGreaterEqual(resolved[0, -2, 0, 0].item(), 0)

    @torch.no_grad()
    def test_clear_duration_uses_fixed_source_and_a_terminal_render_phase(self):
        book, shapes, tiles = assets()
        book = torch.cat((book, torch.full((1, 4, 4), 2, dtype=torch.uint8)))
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = ClearAnimationSceneDynamics(book, shapes, tiles, patterns, 3)
        model = scene_from_weights(clear_duration_scene_weights(base.state_dict(), 3).state_dict())
        self.assertIsInstance(model, ClearDurationSceneDynamics)
        source = torch.full((2, 18, 20), base.blank_id, dtype=torch.long)
        source[:, -1, 2:12] = base.object_tiles[0]
        age = torch.tensor([2, 19])
        phase, _ = model.animation_features(source, age, torch.tensor([0, 4]))
        duration = model.clear_duration(phase)
        torch.testing.assert_close(duration[0], duration[1])
        finish, _ = model.animation_end_logits(phase, age)
        torch.testing.assert_close(finish[1] - finish[0], torch.tensor(34.))
        terminal = model.animation_output_phase(phase, torch.tensor([True, True]))
        torch.testing.assert_close(terminal[0, :13], terminal[1, :13])

    @torch.no_grad()
    def test_terminal_latches_observed_rows_and_holds_learned_panel(self):
        book, shapes, tiles = assets()
        book = torch.cat((book, torch.full((1, 4, 4), 2, dtype=torch.uint8)))
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = ClearDurationSceneDynamics(book, shapes, tiles, patterns, 2)
        frame = torch.full((1, 18, 20), base.blank_id, dtype=torch.long)
        frame[:, -1, 2:12] = base.object_tiles[0]
        panel = torch.full((18, 20), base.blank_id, dtype=torch.long)
        panel[:, 14:17] = 1
        model = scene_from_weights(terminal_scene_weights(base.state_dict(), 3, panel[None]).state_dict())
        self.assertIsInstance(model, TerminalSceneDynamics)
        action = torch.tensor([0])
        expected, state = base.predict_tokens(frame, action)
        predicted, initial = model.predict_tokens(frame, action)
        torch.testing.assert_close(expected, predicted, rtol=0, atol=0)
        torch.testing.assert_close(state, initial, rtol=0, atol=0)
        calibrated = scene_from_weights(terminal_contact_scene_weights(model.state_dict()).state_dict())
        self.assertIsInstance(calibrated, TerminalContactSceneDynamics)
        detail = model.scene_details(frame, action)
        detail["contact_visible"].fill_(True)
        detail["contact_neighbors"][:, 7] = 0
        self.assertEqual(calibrated.terminal_contact_features(frame, detail)[0, 0].item(), 1)
        calibrated.terminal_contact_gate.weight[0, 0] = -3
        torch.testing.assert_close(calibrated.terminal_start_logit(frame, detail),
                                   model.terminal_start_logit(frame, detail) - 3)
        detail["contact_neighbors"][:, 7] = 1
        torch.testing.assert_close(calibrated.terminal_start_logit(frame, detail),
                                   model.terminal_start_logit(frame, detail))
        clear = scene_from_weights(clear_onset_scene_weights(calibrated.state_dict()).state_dict())
        self.assertIsInstance(clear, ClearOnsetSceneDynamics)
        provisional = frame.clone()
        provisional[:, 6, 2:12] = base.object_tiles[0]
        detail = calibrated.scene_details(provisional, action)
        detail["current"] = chosen(calibrated, 0, 6, 2)
        detail["contact_visible"].fill_(True)
        detail["contact_neighbors"][:, 7] = 0
        torch.testing.assert_close(clear.animation_start_logit(provisional, detail),
                                   calibrated.animation_start_logit(provisional, detail))
        clear.animation_contact_gate.weight[0, 0] = -3
        self.assertLess(clear.animation_start_logit(provisional, detail).item(),
                        calibrated.animation_start_logit(provisional, detail).item())
        detail["contact_neighbors"][:, 7] = 1
        torch.testing.assert_close(clear.animation_start_logit(provisional, detail),
                                   calibrated.animation_start_logit(provisional, detail))
        contact = scene_from_weights(compose_contact_scene_weights(clear.state_dict()).state_dict())
        self.assertIsInstance(contact, ComposeContactSceneDynamics)
        scores = torch.arange(18.).reshape(3, 3, 2)
        inputs = {"contact_visible": torch.tensor([True, True, False]), "contact_neighbors": torch.zeros(3, 9)}
        inputs["contact_neighbors"][1, 7] = 1
        with patch.object(ClearOnsetSceneDynamics, "_refine_events", return_value=scores):
            torch.testing.assert_close(contact._refine_events(scores, inputs, None, None), scores)
            contact.compose_contact_gate.weight.fill_(-3)
            expected = scores.clone()
            expected[0, 1] -= 3
            torch.testing.assert_close(contact._refine_events(scores, inputs, None, None), expected)
        model.terminal_onset[-1].bias.fill_(20)
        model.terminal_wipe[-1].weight.zero_()
        model.terminal_wipe[-1].bias.fill_(20)
        model.terminal_duration.weight.zero_()
        model.terminal_duration.bias.fill_(torch.tensor(.05).logit().item())
        wipe, latched = model.predict_tokens(frame, action)
        self.assertTrue((wipe[:, :, 2:12] == 3).all())
        self.assertEqual(latched[0, -1, 0, 0].item(), -1)
        self.assertEqual(latched[0, -2, 0, 0].item(), -1)
        self.assertTrue(torch.equal(latched[0, -1, :, 2:12], frame[0, :, 2:12]))
        model.terminal_onset[-1].bias.fill_(-20)
        later, latched = model.predict_tokens(wipe, action, latched)
        shown, latched = model.predict_tokens(later, action, latched)
        self.assertTrue(torch.equal(shown[0], panel))
        held, _ = model.predict_tokens(shown, torch.tensor([3]), latched)
        self.assertTrue(torch.equal(held[0], panel))

    def test_oriented_proposal_can_rotate_without_changing_style(self):
        book, shapes, _ = assets()
        shapes = torch.cat((shapes, torch.tensor([[[0, 0], [0, 1], [0, 2], [1, 0]]])))
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = RelativeProposalSceneDynamics(book, shapes, torch.tensor([1, 2, 1]), patterns)
        weights = oriented_proposal_scene_weights(base.state_dict()).state_dict()
        weights.pop("oriented_command_gate.weight")
        weights.pop("motion_agreement_gate.weight")
        model = scene_from_weights(weights)
        self.assertIsInstance(model, OrientedProposalSceneDynamics)
        torch.testing.assert_close(model.oriented_command_gate.weight, torch.zeros_like(model.oriented_command_gate.weight))
        torch.testing.assert_close(model.motion_agreement_gate.weight, torch.zeros_like(model.motion_agreement_gate.weight))
        candidates = torch.zeros(3, 3, 540)
        candidates[0, 0, 22] = 1
        candidates[1, 0, 382] = 1
        candidates[2, 0, 122] = 1
        self.assertEqual(model.motion_agreement(candidates, torch.full((3,), 22)).tolist(), [0, 1, 2])
        with torch.no_grad():
            for parameter in model.oriented_proposal.parameters():
                parameter.zero_()
            model.oriented_proposal[-1].bias[:3] = torch.tensor([0., 20., 8.])
        translation = torch.zeros(1, 49)
        translation[0, 24] = 8
        local = model.local_proposal(torch.zeros(1, 540), translation, torch.zeros(1, 37), torch.tensor([22]))
        self.assertEqual(int(local.argmax()), 382)
        self.assertTrue(torch.isneginf(local[:, 180:360]).all())
        torch.testing.assert_close(local.logsumexp(1), torch.zeros(1))
        (-local[0, 382]).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.oriented_proposal.parameters()))

    def test_relative_proposal_can_recover_a_low_absolute_pose(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = ContinuationSceneDynamics(*assets(), patterns)
        weights = relative_proposal_scene_weights(base.state_dict()).state_dict()
        weights.pop("proposal_gate_calibration")
        model = scene_from_weights(weights)
        self.assertIsInstance(model, RelativeProposalSceneDynamics)
        torch.testing.assert_close(model.proposal_gate_calibration, torch.zeros(2))
        with torch.no_grad():
            for parameter in model.relative_proposal.parameters():
                parameter.zero_()
            model.relative_proposal[-1].bias[24] = 8
            model.relative_proposal[-1].bias[-1] = 5
        scores = torch.full((1, 3, len(model.event_poses)), -70.)
        scores[0, 0, 235] = 0
        scores = scores.masked_fill(~model.valid_poses.flatten(), -torch.inf)
        with torch.no_grad():
            model.proposal_gate_calibration.copy_(torch.tensor([0., 2.]))
            model.orientation_transition.weight[3] = -2
        delta = model.proposal_gate_delta(scores, torch.tensor([22]), torch.tensor([3]), torch.tensor([0]))
        torch.testing.assert_close(delta, torch.tensor([[4.]]))
        updated = model.blend_copy(scores, torch.zeros(1, 37), torch.tensor([22]), torch.tensor([True]))
        torch.testing.assert_close(updated.logsumexp(2), scores.logsumexp(2))
        torch.testing.assert_close(updated[:, 1:], scores[:, 1:])
        self.assertEqual(int(updated[0, 0].argmax()), 22)
        (-updated[0, 0, 22]).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.relative_proposal.parameters()))

    def test_continuation_offsets_respect_previous_button_and_contact(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = TrackedCanvasSceneDynamics(*assets(), patterns)
        model = scene_from_weights(continuation_scene_weights(base.state_dict()).state_dict())
        self.assertIsInstance(model, ContinuationSceneDynamics)
        with torch.no_grad():
            model.orientation_transition.weight[3] = -2
            model.orientation_transition.weight[4 * 8 + 3] = 1
        scores = model.continuation_offsets(torch.tensor([3, 3]), torch.tensor([0, 4]), torch.tensor([-5., 5.]))
        self.assertEqual(scores[:, 1].tolist(), [-2., 1.])
        self.assertLess(scores[0, 2].item(), scores[1, 2].item())
        torch.testing.assert_close(scores[:, [0, 3]], torch.zeros(2, 2))

    def test_carry_state_preserves_rendered_pose_and_requires_matching_pixels(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = CanvasOverlapSceneDynamics(*assets(), patterns)
        model = scene_from_weights(tracked_canvas_scene_weights(base.state_dict()).state_dict())
        self.assertIsInstance(model, TrackedCanvasSceneDynamics)
        tokens = torch.zeros(1, 18, 20, dtype=torch.long)
        action = torch.tensor([0])
        prior = model.advance(tokens, action, base.advance(tokens, action))
        self.assertEqual(prior.shape[1], 21)
        self.assertEqual(prior[0, -1, 0, 0].item(), -1)
        prior[:, -1] = 22
        with torch.no_grad():
            model.pose_carry.weight.zero_()
            model.pose_carry.bias.fill_(8)
        wrong = chosen(model, 1, 5, 5)
        with patch.object(model, "_current_details", return_value=(wrong, torch.zeros(1, 16))):
            empty, _ = model._read_current(tokens, torch.zeros(1, 16, 18, 10), prior)
            self.assertEqual(int(empty["joint"].argmax()), 235)
        tokens[:, 2:4, 4:6] = 1
        with patch.object(model, "_current_details", return_value=(chosen(model, 1, 5, 5), torch.zeros(1, 16))):
            details = model.scene_details(tokens, action, prior)
            self.assertEqual(int(details["current"]["joint"].argmax()), 22)
        future = details["next"]
        expected = int(future["joint"].argmax()) if future["visible"].item() > 0 else -1
        self.assertEqual(int(details["state"][0, -1, 0, 0]), expected)

    def test_overlap_uses_each_events_canvas(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = TimedEventSceneDynamics(*assets(), patterns)
        model = scene_from_weights(canvas_overlap_scene_weights(base.state_dict()).state_dict())
        self.assertIsInstance(model, CanvasOverlapSceneDynamics)
        state = torch.zeros(1, 20, 18, 20)
        state[:, 16, 2:4, 4:6] = 1
        state[:, 16, 4, 4] = 2
        details = {"state": state, "current": chosen(model, 0, 2, 2), "next": chosen(model, 0, 3, 2),
                   "contact_visible": torch.ones(1, dtype=torch.bool), "contact_logit": torch.zeros(1)}
        scores = torch.zeros(1, 3, len(model.event_poses))
        updated = model._refine_events(scores, details, torch.tensor([0]), torch.tensor([22]))
        counts = model.canvas_overlap(details)
        self.assertEqual(counts[0, :, 22].tolist(), [0, 4])
        self.assertEqual(counts[0, :, 32].tolist(), [1, 3])
        torch.testing.assert_close(updated[:, 2], scores[:, 2])
        self.assertLess(updated[0, 0, 32], updated[0, 0, 22])

    def test_drop_clock_preserves_pair_mass_and_other_actions(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = JointEventSceneDynamics(*assets(), patterns)
        fitted = timed_event_scene_weights(base.state_dict())
        model = scene_from_weights(fitted.state_dict())
        self.assertIsInstance(model, TimedEventSceneDynamics)
        with torch.no_grad():
            model.drop_clock.weight.zero_()
            model.drop_clock.bias.copy_(torch.tensor([2., 1., 0.]))
        scores = torch.zeros(3, 3, len(model.event_poses))
        scores[:, 0, 22] = 5
        state = torch.zeros(3, 20, 18, 20)
        state[:, 17] = 16
        state[:, 18] = 17
        details = {"state": state, "contact_visible": torch.ones(3, dtype=torch.bool),
                   "contact_logit": torch.tensor([-20., -20., 20.])}
        adjusted = model._refine_events(scores, details, torch.tensor([3, 0, 3]), torch.tensor([22, 22, 22]))
        torch.testing.assert_close(adjusted.logsumexp(2), scores.logsumexp(2))
        torch.testing.assert_close(adjusted[1:], scores[1:])
        self.assertGreater(adjusted[0, 0, 32].item(), adjusted[0, 0, 22].item())
        self.assertGreater(model.drop_log_odds(torch.tensor([0]), torch.tensor([16]), torch.tensor([17])).item(),
                           model.drop_log_odds(torch.tensor([0]), torch.tensor([1]), torch.tensor([17])).item())

    def test_joint_event_initial_parity_and_coupled_decision(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = ContactModeSceneDynamics(*assets(), patterns)
        fitted = joint_event_scene_weights(base.state_dict())
        model = scene_from_weights(fitted.state_dict())
        self.assertIsInstance(model, JointEventSceneDynamics)
        tokens = torch.zeros(1, 18, 20, dtype=torch.long)
        action = torch.tensor([3])
        before, after = base.scene_details(tokens, action), model.scene_details(tokens, action)
        for hard in (False, True):
            torch.testing.assert_close(base.decode_scene(tokens, before, hard=hard),
                                       model.decode_scene(tokens, after, hard=hard))
        current = torch.tensor([22])
        future = torch.full((1, len(model.event_poses)), -40.)
        future[0, 22] = 0
        future[0, 32] = -.5
        mode = torch.tensor([[0., -1., -2.]])
        with torch.no_grad():
            model.event_row.weight[18, 1] = 3
        scores = model.event_scores(mode, future, current, torch.zeros(1), torch.ones(1, dtype=torch.bool))
        self.assertEqual(divmod(int(scores.flatten().argmax()), future.size(1)), (1, 32))

    def test_contact_mode_conditions_only_deposit_when_piece_matches_screen(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        base = RowTransportSceneDynamics(*assets(), patterns)
        fitted = contact_mode_scene_weights(base.state_dict())
        with torch.no_grad():
            fitted.contact_head.weight.zero_()
            fitted.contact_head.weight[0, 7] = 16
            fitted.contact_head.bias.fill_(-8)
        model = scene_from_weights(fitted.state_dict())
        self.assertIsInstance(model, ContactModeSceneDynamics)
        scene = {"state": torch.zeros(1, 20, 18, 20), "current": chosen(model, 0, 2, 2)}
        tokens = torch.zeros(1, 18, 20, dtype=torch.long)
        pooled = torch.zeros(1, 48)
        action = torch.tensor([3])
        expected = base._scene_mode(tokens, action, pooled, scene)["mode"]
        torch.testing.assert_close(model._scene_mode(tokens, action, pooled, scene)["mode"], expected)
        tokens[:, 2:4, 4:6] = 1
        floating = model._scene_mode(tokens, action, pooled, scene)["mode"] - expected
        self.assertLess(floating[0, 1].item(), -8)
        torch.testing.assert_close(floating[:, [0, 2]], torch.zeros(1, 2))
        tokens[:, 4, 4] = 2
        supported = model._scene_mode(tokens, action, pooled, scene)["mode"] - expected
        self.assertGreater(supported[0, 1].item(), -.001)

    def test_row_preserve_selects_after_transport_and_before_overlay(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        baseline = RowTransportSceneDynamics(*assets(), patterns)
        model = row_preserve_scene_weights(baseline.state_dict())
        tokens = torch.zeros(1, 18, 20, dtype=torch.long)
        tokens[:, 2, 2:12] = 1
        tokens[:, 16, 2:12] = 2
        actions = torch.tensor([4])
        expected = baseline.scene_details(tokens, actions)
        actual = model.scene_details(tokens, actions)
        self.assertEqual(actual["row_preserve_features"].shape, (1, 18, 18))
        self.assertTrue((actual["row_preserve_logits"] < 0).all())
        torch.testing.assert_close(actual["state"], expected["state"], rtol=0, atol=0)
        torch.testing.assert_close(model.decode_scene(tokens, actual, hard=True),
                                   baseline.decode_scene(tokens, expected, hard=True), rtol=0, atol=0)
        logits = model.row_preserve_head(actual["row_preserve_features"])
        torch.testing.assert_close(logits, actual["row_preserve_logits"])
        F.softplus(-logits).mean().backward()
        self.assertTrue(torch.isfinite(model.row_preserve_head.readout.weight.grad).all())
        self.assertGreater(model.row_preserve_head.readout.bias.grad.abs().item(), 0)

        details = synthetic_details(model, mode=2)
        details["preview_render_logits"] = torch.tensor([[-40., 0., 0.]])
        details["row_transport_gate"] = torch.tensor([40.])
        rows = torch.full((1, 18, 19), -40.)
        rows[:, :, 18] = 40.
        rows[:, 8, 18] = -40.
        rows[:, 8, 16] = 40.
        details["row_transport_logits"] = rows
        details["row_preserve_logits"] = torch.full((1, 18), -40.)
        details["row_preserve_logits"][:, 2] = 40.
        background, rewritten = model.row_preserve_candidates(tokens, details)
        torch.testing.assert_close(background[0, 2, 2:12], torch.tensor([1, 1, 0, 0, 1, 1, 1, 1, 1, 1]))
        torch.testing.assert_close(rewritten[0, 8, 2:12], torch.full((10,), 2))
        desired = rewritten[0, :, 2:12].clone()
        desired[2] = background[0, 2, 2:12]
        desired[7:10, 5] = 2
        desired[9, 6] = 2
        for hard in (True, False):
            rendered = model.decode_scene(tokens, details, hard)
            original = baseline.decode_scene(tokens, details, hard)
            selected = rendered if hard else rendered.argmax(1)
            torch.testing.assert_close(selected[0, :, 2:12], desired)
            torch.testing.assert_close(rendered[..., :2], original[..., :2], rtol=0, atol=0)
            torch.testing.assert_close(rendered[..., 12:], original[..., 12:], rtol=0, atol=0)
        for payload in (model.state_dict(), {"model": model.state_dict()}):
            restored = scene_from_weights(payload)
            self.assertIsInstance(restored, RowPreserveSceneDynamics)
            torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            row_preserve_scene_weights(model.state_dict())

    def test_current_readout_checks_appearance_without_hard_mask(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        baseline = RowTransportSceneDynamics(*assets(), patterns)
        model = current_readout_scene_weights(baseline.state_dict())
        tokens = torch.zeros(2, 18, 20, dtype=torch.long)
        tokens[:, 3:5, 6:8] = 1
        actions = torch.tensor([0, 4])
        expected = baseline.scene_details(tokens, actions)
        actual = model.scene_details(tokens, actions)
        for key in ("current", "next"):
            torch.testing.assert_close(actual[key]["joint"], expected[key]["joint"], rtol=0, atol=0)
        torch.testing.assert_close(model.decode_scene(tokens, actual, hard=True),
                                   baseline.decode_scene(tokens, expected, hard=True), rtol=0, atol=0)
        lookup = model.current_readout_lookup()
        features = model.current_readout_features(tokens[:, :, 2:12], lookup)
        torch.testing.assert_close(features[:, 0, 34], torch.ones(2, 3))
        torch.testing.assert_close(features[:, 0, 35, 0], torch.full((2,), 0.5))
        (-actual["current"]["joint"][:, 0, 34].mean()).backward()
        gradient = model.current_readout_head[-1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        with torch.no_grad():
            model.current_readout_head[0].weight.zero_()
            model.current_readout_head[0].bias.zero_()
            model.current_readout_head[0].weight[0, 0] = 1.
            model.current_readout_head[-1].weight[0, 0] = 2.
        changed = model.scene_details(tokens, actions)["current"]["joint"]
        self.assertTrue(torch.isfinite(changed[:, model.valid_poses]).all())
        self.assertTrue(torch.isneginf(changed[:, ~model.valid_poses]).all())
        self.assertTrue((changed[:, 0, 34] > expected["current"]["joint"][:, 0, 34]).all())
        for payload in (model.state_dict(), {"model": model.state_dict()}):
            restored = scene_from_weights(payload)
            self.assertIsInstance(restored, CurrentReadoutSceneDynamics)
            torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            current_readout_scene_weights(model.state_dict())

    def test_rewrite_mode_reads_own_proposal_and_preserves_base_at_init(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        baseline = RowTransportSceneDynamics(*assets(), patterns)
        model = rewrite_mode_scene_weights(baseline.state_dict())
        tokens = torch.zeros(2, 18, 20, dtype=torch.long)
        actions = torch.tensor([0, 4])
        expected = baseline.scene_details(tokens, actions)
        actual = model.scene_details(tokens, actions)
        for key in ("mode", "state", "row_transport_logits", "row_transport_gate", "preview_render_logits"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(actual["rewrite_mode_base_logits"], expected["mode"], rtol=0, atol=0)
        torch.testing.assert_close(actual["next"]["joint"], expected["next"]["joint"], rtol=0, atol=0)
        for hard in (True, False):
            torch.testing.assert_close(model.decode_scene(tokens, actual, hard),
                                       baseline.decode_scene(tokens, expected, hard), rtol=0, atol=0)
        self.assertEqual(actual["rewrite_mode_features"].shape, (2, 10))
        self.assertTrue(torch.isfinite(actual["rewrite_mode_features"]).all())
        F.cross_entropy(actual["mode"], torch.tensor([0, 2])).backward()
        gradient = model.rewrite_mode_head.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)

        proposal = synthetic_details(model, mode=2)
        proposal["gate"].fill_(-40.)
        proposal["gate"][:, :, 12, 2:12] = 40.
        summary = model.rewrite_mode_features(tokens[:1], proposal)
        torch.testing.assert_close(summary[0, :4], torch.full((4,), 1 / 18))
        torch.testing.assert_close(summary[0, 6], torch.tensor(1.))
        with torch.no_grad():
            model.rewrite_mode_head.bias.copy_(torch.tensor([0., -1., 3.]))
        updated = model.scene_details(tokens, actions)
        torch.testing.assert_close(updated["mode"], expected["mode"] + model.rewrite_mode_head.bias)
        for payload in (model.state_dict(), {"model": model.state_dict()}):
            restored = scene_from_weights(payload)
            self.assertIsInstance(restored, RewriteModeSceneDynamics)
            torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            rewrite_mode_scene_weights(model.state_dict())

    def test_row_transport_rewrites_background_before_foreground(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        baseline = ModeCorrectionSceneDynamics(*assets(), patterns)
        model = row_transport_scene_weights(baseline.state_dict())
        tokens = torch.zeros(1, 18, 20, dtype=torch.long)
        tokens[:, 15, 2:12] = 1
        tokens[:, 16, 2:12] = 2
        actions = torch.tensor([4])
        expected = baseline.scene_details(tokens, actions)
        actual = model.scene_details(tokens, actions)
        self.assertEqual(actual["row_transport_logits"].shape, (1, 18, 19))
        self.assertTrue((actual["row_transport_gate"] < 0).all())
        torch.testing.assert_close(actual["state"], expected["state"], rtol=0, atol=0)
        torch.testing.assert_close(model.decode_scene(tokens, actual, hard=True),
                                   baseline.decode_scene(tokens, expected, hard=True), rtol=0, atol=0)
        features = model.row_transport_features(tokens, actual)
        self.assertEqual(features.shape, (1, 18, 17))
        self.assertEqual(features[0, 15, 16].item(), 1.)
        rows, gate = model.row_transport_head(features)
        torch.testing.assert_close(rows, actual["row_transport_logits"])
        (F.cross_entropy(rows.flatten(0, 1), torch.arange(18)) + F.softplus(-gate).mean()).backward()
        for parameter in model.row_transport_head.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())

        details = synthetic_details(model, mode=2)
        details["preview_render_logits"] = torch.tensor([[-40., 0., 0.]])
        details["row_transport_gate"] = torch.tensor([40.])
        chosen_rows = torch.full((1, 18, 19), -40.)
        chosen_rows[:, :, 18] = 40.
        chosen_rows[:, 8, 18] = -40.
        chosen_rows[:, 8, 16] = 40.
        details["row_transport_logits"] = chosen_rows
        desired = torch.zeros(18, 10, dtype=torch.long)
        desired[8] = 2
        desired[7:10, 5] = 2
        desired[9, 6] = 2
        for hard in (True, False):
            transported = model.decode_scene(tokens, details, hard)
            rendered = transported if hard else transported.argmax(1)
            torch.testing.assert_close(rendered[0, :, 2:12], desired)
            original = baseline.decode_scene(tokens, details, hard)
            torch.testing.assert_close(transported[..., :2], original[..., :2], rtol=0, atol=0)
            torch.testing.assert_close(transported[..., 12:], original[..., 12:], rtol=0, atol=0)
        details["mode"] = torch.tensor([[40., -40., -40.]])
        torch.testing.assert_close(model.decode_scene(tokens, details, hard=True),
                                   baseline.decode_scene(tokens, details, hard=True), rtol=0, atol=0)
        for payload in (model.state_dict(), {"model": model.state_dict()}):
            restored = scene_from_weights(payload)
            self.assertIsInstance(restored, RowTransportSceneDynamics)
            torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            row_transport_scene_weights(model.state_dict())

    def test_spatial_mode_uses_board_rows_and_preserves_base_at_init(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        baseline = ModeCorrectionSceneDynamics(*assets(), patterns)
        with torch.no_grad():
            baseline.mode_correction_head.bias.copy_(torch.tensor([0.4, 0.2, -1.]))
            baseline.consistency_head.bias.copy_(torch.tensor([2., 3.]))
        model = spatial_mode_scene_weights(baseline.state_dict())
        tokens = torch.zeros(2, 18, 20, dtype=torch.long)
        tokens[:, :, 13:19] = 1
        tokens[0, 10, 2:5] = 1
        tokens[1, 12, 2:12] = 2
        actions = torch.tensor([0, 4])
        prior = baseline.advance(tokens, actions)
        expected = baseline.scene_details(tokens, actions, prior)
        actual = model.scene_details(tokens, actions, prior)
        for key in ("mode", "state", "preview_render_logits", "consistency_joint_offset"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(actual["next"]["joint"], expected["next"]["joint"], rtol=0, atol=0)
        torch.testing.assert_close(model.decode_scene(tokens, actual, hard=True),
                                   baseline.decode_scene(tokens, expected, hard=True), rtol=0, atol=0)
        features = actual["spatial_mode_features"]
        self.assertEqual(features.shape, (2, 17, 18))
        occupancy = torch.zeros(2, 18)
        occupancy[0, 10] = 0.3
        occupancy[1, 12] = 1.
        torch.testing.assert_close(features[:, 16], occupancy)
        torch.testing.assert_close(features[:, :16], actual["state"][:, :16, :, 2:12].mean(3))
        torch.testing.assert_close(actual["spatial_mode_base_logits"], expected["mode"], rtol=0, atol=0)
        F.cross_entropy(actual["mode"], torch.tensor([0, 2])).backward()
        gradient = model.spatial_mode_head.readout.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        with torch.no_grad():
            model.spatial_mode_head.readout.bias.copy_(torch.tensor([5., 0., -4.]))
        updated = model.scene_details(tokens, actions, prior)
        torch.testing.assert_close(updated["mode"], expected["mode"] + model.spatial_mode_head(features))
        self.assertFalse(torch.allclose(updated["consistency_joint_offset"], expected["consistency_joint_offset"]))
        for payload in (model.state_dict(), {"model": model.state_dict()}):
            restored = scene_from_weights(payload)
            self.assertIsInstance(restored, SpatialModeSceneDynamics)
            torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            spatial_mode_scene_weights(model.state_dict())

    def test_mode_correction_precedes_mode_consumers(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        baseline = ConsistencySceneDynamics(*assets(), patterns)
        with torch.no_grad():
            baseline.mode_head.bias.zero_()
            baseline.overlap_head[-1].bias.fill_(1.)
            baseline.spawn_preview_head[-1].bias.copy_(torch.tensor([-0.3, 0.2]))
            baseline.consistency_head.bias.copy_(torch.tensor([2., 3.]))
            baseline.preview_render_head[0].weight[0, -3] = 1.
            baseline.preview_render_head[-1].weight[0, 0] = 1.
        model = mode_correction_scene_weights(baseline.state_dict())
        tokens = torch.randint(0, 3, (2, 18, 20))
        actions = torch.tensor([0, 4])
        prior = baseline.advance(tokens, actions)
        expected = baseline.scene_details(tokens, actions, prior)
        actual = model.scene_details(tokens, actions, prior)
        self.assertEqual(actual["mode_correction_features"].shape, (2, 58))
        torch.testing.assert_close(actual["mode_base_logits"], expected["mode"], rtol=0, atol=0)
        torch.testing.assert_close(actual["next"]["joint"], expected["next"]["joint"], rtol=0, atol=0)
        for key in ("mode", "state", "preview_joint_offset", "consistency_joint_offset", "preview_render_logits"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(model.decode_scene(tokens, actual, hard=True),
                                   baseline.decode_scene(tokens, expected, hard=True), rtol=0, atol=0)
        F.cross_entropy(actual["mode"], torch.tensor([0, 1])).backward()
        self.assertTrue(torch.isfinite(model.mode_correction_head.weight.grad).all())
        self.assertGreater(model.mode_correction_head.bias.grad.abs().sum().item(), 0)
        with torch.no_grad():
            model.mode_correction_head.bias.copy_(torch.tensor([4., -3., 0.]))
        updated = model.scene_details(tokens, actions, prior)
        torch.testing.assert_close(updated["mode"], expected["mode"] + model.mode_correction_head.bias)
        torch.testing.assert_close(updated["mode_base_logits"], expected["mode"])
        for key in ("preview_joint_offset", "consistency_joint_offset", "preview_render_logits"):
            self.assertFalse(torch.allclose(updated[key], expected[key]), key)
        self.assertFalse(torch.allclose(updated["next"]["joint"], expected["next"]["joint"]))
        for payload in (model.state_dict(), {"model": model.state_dict()}):
            restored = scene_from_weights(payload)
            self.assertIsInstance(restored, ModeCorrectionSceneDynamics)
            torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            mode_correction_scene_weights(model.state_dict())

    def test_consistency_scene_parity_groups_and_roundtrip(self):
        patterns = torch.zeros(2, 6, 6, dtype=torch.long)
        patterns[0, :2, :2] = 1
        patterns[1, :2, :2] = 2
        baseline = FactorizedPreviewRenderSceneDynamics(*assets(), patterns)
        with torch.no_grad():
            baseline.preview_render_head[-1].bias[0] = 0.75
            baseline.overlap_scale.fill_(2.5)
        model = consistency_scene_weights(baseline.state_dict())
        tokens = torch.randint(0, 3, (2, 18, 20))
        actions = torch.tensor([0, 4])
        prior = baseline.advance(tokens, actions)
        expected = baseline.scene_details(tokens, actions, prior)
        actual = model.scene_details(tokens, actions, prior)
        for key in ("state", "preview_render_logits", "preview_joint_offset", "overlap_logits"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(actual["next"]["joint"], expected["next"]["joint"], rtol=0, atol=0)
        torch.testing.assert_close(model.decode_scene(tokens, actual, hard=True),
                                   baseline.decode_scene(tokens, expected, hard=True), rtol=0, atol=0)
        self.assertEqual(model.consistency_features(actual["state"], actions).shape, (2, 26))
        group = model.consistency_group(chosen(model, 0, 2, 2)["joint"])
        self.assertEqual(group[0, 0, 32].item(), 0)
        self.assertEqual(group[0, 1, 32].item(), 1)
        self.assertEqual(group[0, 0, 33].item(), 2)
        self.assertEqual(group[0, 1, 33].item(), 3)
        (-actual["next"]["joint"][:, 0, 22].mean()).backward()
        self.assertTrue(torch.isfinite(model.consistency_head.bias.grad).all())
        self.assertGreater(model.consistency_head.bias.grad.abs().sum().item(), 0)
        with torch.no_grad():
            model.consistency_head.bias.copy_(torch.tensor([2., 3.]))
        updated = model.scene_details(tokens, actions, prior)
        groups = model.consistency_group(updated["current"]["joint"])
        changes = torch.tensor([0., 2., 3., 5.])[groups]
        copy = updated["mode"].softmax(1)[:, 0, None, None]
        torch.testing.assert_close(updated["consistency_joint_offset"], changes * model.valid_poses * copy)
        for payload in (model.state_dict(), {"model": model.state_dict()}):
            restored = scene_from_weights(payload)
            self.assertIsInstance(restored, ConsistencySceneDynamics)
            torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            consistency_scene_weights(model.state_dict())
        incomplete = dict(model.state_dict())
        del incomplete["_factorized_preview_render_scene"]
        with self.assertRaises(ValueError):
            scene_from_weights(incomplete)

    def test_motion_clock_migration_features_and_parity(self):
        baseline = SpawnPreviewSceneDynamics(*assets())
        baseline.overlap_scale.fill_(2.5)
        with torch.no_grad():
            baseline.spawn_preview_head[-1].bias.copy_(torch.tensor([-0.3, 0.2]))
        model = motion_clock_scene_weights(baseline.state_dict())
        tokens = torch.randint(0, 3, (2, 18, 20))
        actions = torch.tensor([1, 4])
        prefix = baseline.advance(tokens, actions)
        expected = baseline.scene_details(tokens, actions, prefix)
        actual = model.scene_details(tokens, actions, prefix)
        for key in ("state", "overlap_logits", "preview_joint_offset"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(actual["next"]["joint"], expected["next"]["joint"], rtol=0, atol=0)
        self.assertTrue(torch.equal(model.decode_scene(tokens, actual, hard=True),
                                    baseline.decode_scene(tokens, expected, hard=True)))
        state = actual["state"].detach().clone()
        state[:, 17] = 0
        state[:, 17, :, 2:12] = 70
        state[:, 17, 3, 3] = 10
        state[:, 18] = 33
        features = model.motion_clock_features(state, actions)
        self.assertEqual(features.shape, (2, 26))
        torch.testing.assert_close(features[:, 16:18], torch.tensor([[10., 33.], [10., 33.]]) / 255)
        torch.testing.assert_close(features[:, 18:], model.action_embedding(actions))
        current = actual["current"]["joint"].flatten(1).argmax(1, keepdim=True)
        loss = -actual["next"]["joint"].flatten(1).gather(1, current).mean()
        loss.backward()
        gradient = model.motion_clock_head.bias.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        with torch.no_grad():
            model.motion_clock_head.bias.fill_(2)
        updated = model.scene_details(tokens, actions, prefix)
        offset = updated["motion_joint_offset"]
        torch.testing.assert_close(offset.flatten(1).gather(1, current), torch.zeros(2, 1))
        self.assertTrue((offset[:, model.valid_poses] > 0).any())
        self.assertTrue((offset[:, ~model.valid_poses] == 0).all())
        for payload in (model.state_dict(), {"model": model.state_dict()}):
            restored = scene_from_weights(payload)
            self.assertIsInstance(restored, MotionClockSceneDynamics)
            torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            motion_clock_scene_weights(model.state_dict())
        incomplete = dict(model.state_dict())
        del incomplete["_spawn_preview_scene"]
        with self.assertRaises(ValueError):
            scene_from_weights(incomplete)

    def test_spawn_preview_parity_gradient_and_crop(self):
        baseline = LearnedOverlapSceneDynamics(*assets())
        baseline.overlap_scale.fill_(2.5)
        with torch.no_grad():
            baseline.overlap_head[-1].weight.fill_(0.1)
        model = spawn_preview_scene_weights(baseline.state_dict())
        tokens = torch.randint(0, 3, (2, 18, 20))
        actions = torch.tensor([1, 4])
        expected = baseline.scene_details(tokens, actions)
        actual = model.scene_details(tokens, actions)
        for key in ("state", "overlap_logits"):
            self.assertTrue(torch.equal(actual[key], expected[key]))
        self.assertTrue(torch.equal(actual["next"]["joint"], expected["next"]["joint"]))
        self.assertTrue(torch.equal(model.decode_scene(tokens, actual, hard=True),
                                    baseline.decode_scene(tokens, expected, hard=True)))
        self.assertEqual(actual["preview_joint_offset"].shape, (2, 2, 1))
        (-actual["next"]["joint"][:, 0, 22].mean()).backward()
        gradient = model.spawn_preview_head[-1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        with torch.no_grad():
            model.spawn_preview_head[-1].weight.copy_(gradient)
        outside = (tokens + 1) % 3
        outside[:, :6, 13:19] = tokens[:, :6, 13:19]
        self.assertTrue(torch.equal(model._preview_shape_scores(tokens), model._preview_shape_scores(outside)))
        for payload in (model.state_dict(), {"model": model.state_dict()}):
            restored = scene_from_weights(payload)
            self.assertIsInstance(restored, SpawnPreviewSceneDynamics)
            self.assertEqual(restored.overlap_scale.item(), 2.5)
            torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
        for invalid in (ObjectSceneDynamics(*assets()), model):
            with self.assertRaises(ValueError):
                spawn_preview_scene_weights(invalid.state_dict())

    def test_learned_overlap_zero_init_preserves_joint_and_learns_from_next_nll(self):
        baseline = ObjectSceneDynamics(*assets())
        model = learned_overlap_scene_weights(baseline.state_dict())
        tokens = torch.randint(0, 3, (2, 18, 20))
        actions = torch.tensor([1, 4])
        expected = baseline.scene_details(tokens, actions)
        actual = model.scene_details(tokens, actions)
        self.assertTrue(torch.equal(actual["next"]["joint"], expected["next"]["joint"]))
        self.assertTrue(torch.equal(actual["state"], expected["state"]))
        self.assertEqual(actual["overlap_logits"].shape, (2, 2, 180))
        self.assertTrue(torch.equal(model.decode_scene(tokens, actual, hard=True),
                                    baseline.decode_scene(tokens, expected, hard=True)))
        (-actual["next"]["joint"][:, 0, 22].mean()).backward()
        gradient = model.overlap_head[-1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        for payload in (model.state_dict(), {"model": model.state_dict()}):
            self.assertIsInstance(scene_from_weights(payload), LearnedOverlapSceneDynamics)

    def test_learned_overlap_changes_relative_score_without_a_hard_mask(self):
        model = LearnedOverlapSceneDynamics(*assets())
        tokens = torch.zeros(1, 18, 20, dtype=torch.long)
        actions = torch.zeros(1, dtype=torch.long)
        cell_logits = torch.zeros(1, 1, 18, 10)
        cell_logits[:, :, 2, 2] = 4
        with torch.no_grad():
            model.mode_head.bias.copy_(torch.tensor([8., 0., 0.]))
            before = model.scene_details(tokens, actions)["next"]["joint"]
            with patch.object(model.overlap_head, "forward", return_value=cell_logits):
                details = model.scene_details(tokens, actions)
            after = details["next"]["joint"]
        self.assertEqual(details["overlap_logits"][0, 0, 22].item(), 4)
        self.assertEqual(details["overlap_logits"][0, 0, 66].item(), 0)
        self.assertLess((after[0, 0, 22] - after[0, 0, 66]).item(),
                        (before[0, 0, 22] - before[0, 0, 66]).item())
        self.assertTrue(torch.isfinite(after[0, 0, 22]))

    @torch.inference_mode()
    def test_learned_overlap_scale_defaults_to_one_and_roundtrips(self):
        model = LearnedOverlapSceneDynamics(*assets()).eval()
        model.overlap_head[-1].weight.fill_(0.25)
        tokens = torch.randint(0, 3, (2, 18, 20))
        actions = torch.tensor([1, 4])
        historical = dict(model.state_dict())
        del historical["overlap_scale"]
        expected = model.scene_details(tokens, actions)
        for payload in (historical, {"model": historical}):
            restored = scene_from_weights(payload).eval()
            self.assertEqual(restored.overlap_scale.item(), 1)
            actual = restored.scene_details(tokens, actions)
            self.assertTrue(torch.equal(actual["next"]["joint"], expected["next"]["joint"]))
            self.assertTrue(torch.equal(actual["overlap_logits"], expected["overlap_logits"]))
        self.assertNotIn("overlap_scale", historical)
        model.overlap_scale.fill_(2.5)
        calibrated = scene_from_weights({"model": model.state_dict()}).eval()
        self.assertEqual(calibrated.overlap_scale.item(), 2.5)
        calibrated_details = calibrated.scene_details(tokens, actions)
        self.assertTrue(torch.equal(calibrated_details["next"]["joint"], model.scene_details(tokens, actions)["next"]["joint"]))
        self.assertTrue(torch.equal(calibrated_details["overlap_logits"], expected["overlap_logits"]))
        for invalid in (torch.tensor(-1.), torch.tensor(float("nan")), torch.tensor(float("inf")),
                        torch.tensor([1.]), torch.tensor(1e100, dtype=torch.float64)):
            with self.assertRaisesRegex(ValueError, "overlap_scale"):
                scene_from_weights({**historical, "overlap_scale": invalid})
        incomplete = dict(historical)
        del incomplete["overlap_head.0.weight"]
        with self.assertRaises(RuntimeError):
            scene_from_weights(incomplete)

    @torch.inference_mode()
    def test_current_observation_is_action_invariant_and_transition_matches_legacy(self):
        baseline = ObjectSceneDynamics(*assets()).eval()
        model = action_invariant_current_scene_weights(baseline.state_dict()).eval()
        tokens = torch.randint(0, 3, (2, 18, 20))
        prior = baseline.advance(tokens, torch.tensor([2, 4]))
        noop = torch.zeros(2, dtype=torch.long)
        expected_observation = baseline.scene_details(tokens, noop, prior)
        for actions in (noop, torch.tensor([1, 4]), torch.tensor([3, 7]), torch.tensor([6, 2])):
            actual = model.scene_details(tokens, actions, prior)
            expected_transition = baseline.scene_details(tokens, actions, prior)
            for key in ("joint", "visible"):
                self.assertTrue(torch.equal(actual["current"][key], expected_observation["current"][key]))
                self.assertTrue(torch.equal(actual["next"][key], expected_transition["next"][key]))
            for key in ("state", "write", "gate"):
                self.assertTrue(torch.equal(actual[key], expected_transition[key]))
            if torch.equal(actions, noop):
                self.assertTrue(torch.equal(actual["mode"], expected_observation["mode"]))
                self.assertTrue(torch.equal(model.decode_scene(tokens, actual, hard=True),
                                            baseline.decode_scene(tokens, expected_observation, hard=True)))
        self.assertEqual(prior.shape, (2, 20, 18, 20))
        self.assertEqual(set(model.state_dict()) - set(baseline.state_dict()), {"_action_invariant_current_scene"})
        for payload in (model.state_dict(), {"model": model.state_dict()}):
            loaded = scene_from_weights(payload).eval()
            self.assertIsInstance(loaded, ActionInvariantCurrentSceneDynamics)
            actions = torch.tensor([1, 4])
            actual = loaded.scene_details(tokens, actions, prior)
            expected = model.scene_details(tokens, actions, prior)
            self.assertTrue(torch.equal(actual["current"]["joint"], expected["current"]["joint"]))
            self.assertTrue(torch.equal(actual["state"], expected["state"]))
        with self.assertRaises(ValueError):
            action_invariant_current_scene_weights(model.state_dict())

    def test_pose_rerank_zero_init_preserves_joint_state_and_hard_render(self):
        baseline = ObjectSceneDynamics(*assets())
        model = pose_rerank_scene_weights(baseline.state_dict())
        tokens = torch.randint(0, 3, (2, 18, 20))
        actions = torch.tensor([1, 4])
        expected = baseline.scene_details(tokens, actions)
        actual = model.scene_details(tokens, actions)
        self.assertTrue(torch.equal(expected["next"]["joint"], actual["next"]["joint"]))
        self.assertTrue(torch.equal(expected["state"], actual["state"]))
        self.assertEqual(actual["state"].shape, (2, 20, 18, 20))
        self.assertTrue(torch.equal(baseline.decode_scene(tokens, expected, hard=True),
                                    model.decode_scene(tokens, actual, hard=True)))
        self.assertIsInstance(scene_from_weights({"model": model.state_dict()}), PoseRerankSceneDynamics)

    def test_pose_rerank_only_redistributes_top3_mass_and_gradients(self):
        model = PoseRerankSceneDynamics(*assets())
        hidden = torch.zeros(1, 16, 18, 10)
        joint = torch.zeros(1, 2, 180)
        joint[0, 0, 22:25] = torch.tensor([3., 2., 1.])
        current = chosen(model, 0, 2, 2)["joint"]
        with torch.no_grad():
            model.pose_reranker[0].weight.zero_()
            model.pose_reranker[0].bias.zero_()
            model.pose_reranker[0].weight[0, 17] = 1
        valid = model.valid_poses.flatten()
        adjusted = model._rerank_joint(joint, hidden, current).flatten(1)
        log_joint = adjusted.masked_fill(~valid, -torch.inf).log_softmax(1)
        inside = torch.autograd.grad(-log_joint[0, 24], model.pose_reranker[-1].weight, retain_graph=True)[0]
        outside = torch.autograd.grad(-log_joint[0, 33], model.pose_reranker[-1].weight)[0]
        self.assertTrue(torch.isfinite(inside).all())
        self.assertGreater(inside.abs().sum().item(), 0)
        torch.testing.assert_close(outside, torch.zeros_like(outside), rtol=0, atol=1e-7)
        with torch.no_grad():
            model.pose_reranker[-1].weight[0, 0] = 2
        corrected = model._rerank_joint(joint, hidden, current).flatten(1)
        outside_group = torch.ones(360, dtype=torch.bool)
        outside_group[22:25] = False
        self.assertTrue(torch.equal(corrected[:, outside_group], joint.flatten(1)[:, outside_group]))
        torch.testing.assert_close(corrected[:, 22:25].logsumexp(1), joint[:, 0, 22:25].logsumexp(1))
        self.assertFalse(torch.equal(corrected[:, 22:25], joint[:, 0, 22:25]))
    def test_local_pose_reads_four_cells_and_preserves_zero_init_parity(self):
        baseline = ObjectSceneDynamics(*assets())
        model = footprint_scene_weights(baseline.state_dict())
        tokens = torch.randint(0, 3, (2, 18, 20))
        actions = torch.tensor([1, 4])
        expected = baseline.scene_details(tokens, actions)
        actual = model.scene_details(tokens, actions)
        self.assertTrue(torch.equal(expected["next"]["joint"], actual["next"]["joint"]))
        self.assertTrue(torch.equal(expected["state"], actual["state"]))
        self.assertEqual(actual["state"].shape, (2, 20, 18, 20))
        (-actual["next"]["joint"][:, 0, 22].mean()).backward()
        self.assertTrue(torch.isfinite(model.footprint_score.weight.grad).all())
        self.assertGreater(model.footprint_score.weight.grad.abs().sum().item(), 0)
        self.assertIsInstance(scene_from_weights({"model": model.state_dict()}), FootprintSceneDynamics)
        with torch.no_grad():
            model.footprint_score.weight[0, 0, 0, 0] = 1
        hidden = torch.zeros(2, 16, 18, 10)
        hidden[:, 0, 2, 2] = 8
        scores = model._next_joint_offset(hidden)
        self.assertEqual(scores.shape, (2, 2, 180))
        self.assertEqual(scores[0, 0, 22].item(), 2)
        self.assertEqual(scores[0, 0, 44].item(), 0)
        self.assertEqual(scores[0, 1, 22].item(), 0)

    def test_relative_transport_shifts_masks_normalization_and_empty_fallback(self):
        model = RelativeSceneDynamics(*assets())
        for dy, dx in ((0, 0), (3, 3), (-3, -3)):
            joint = chosen(model, 0, 6, 4)["joint"]
            delta = torch.zeros(1, 49); delta[:, (dy + 3) * 7 + dx + 3] = 1
            transported, mass, fallback = model.transport_joint(joint, delta)
            self.assertFalse(bool(fallback[0]))
            self.assertEqual(float(mass[0]), 1.)
            self.assertEqual(int(transported.flatten(1).argmax()), (6 + dy) * 10 + 4 + dx)
            torch.testing.assert_close(transported.flatten(1).logsumexp(1), torch.zeros(1))
            self.assertTrue(torch.isneginf(transported[0][~model.valid_poses]).all())
        joint = chosen(model, 1, 0, 0)["joint"]
        delta = torch.zeros(1, 49); delta[:, 0] = 1
        transported, mass, fallback = model.transport_joint(joint, delta)
        self.assertTrue(bool(fallback[0])); self.assertEqual(float(mass[0]), 0.)
        torch.testing.assert_close(transported.flatten(1).logsumexp(1), torch.zeros(1))
        tokens, actions = torch.zeros(1, 18, 20, dtype=torch.long), torch.tensor([0])
        absolute = ObjectSceneDynamics.scene_details(model, tokens, actions)["next"]["joint"]
        with patch.object(model, "transport_joint", return_value=(joint, torch.zeros(1), torch.ones(1, dtype=torch.bool))):
            fallback_details = model.scene_details(tokens, actions)
        torch.testing.assert_close(fallback_details["next"]["joint"], absolute, rtol=0, atol=0)
        self.assertTrue(bool(fallback_details["transport"]["numerical_fallback"][0]))

    def test_relative_full_gradient_legacy_weights_state_and_parameter_count(self):
        old = ObjectSceneDynamics(*assets())
        model = relative_scene_weights(old.state_dict())
        self.assertEqual(sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in old.parameters()), 2050)
        for name, value in old.state_dict().items(): torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        tokens = torch.randint(0, 3, (2, 18, 20)); actions = torch.tensor([1, 4])
        details = model.scene_details(tokens, actions)
        details["current"]["joint"].retain_grad()
        # Recompute through the public forward to test all renderer and auxiliary parameters.
        logits, state = model(tokens, actions)
        self.assertEqual(state.shape, (2, 20, 18, 20))
        self.assertTrue(torch.isfinite(logits).all())
        loss = F.nll_loss(logits, (tokens + 1) % 3)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.motion_delta.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.motion_gate.weight.grad.abs().sum()), 0.)
        torch.testing.assert_close(details["transport"]["alpha"], torch.full((2, 1), torch.sigmoid(torch.tensor(-4.)).item()))
        with self.assertRaises(ValueError): relative_scene_weights(model.state_dict())
        with self.assertRaises(ValueError): relative_scene_weights(expand_scene_weights(old.state_dict()).state_dict())
        mixed = dict(model.state_dict()); mixed["_extended_scene"] = torch.tensor(True)
        with self.assertRaises(ValueError): scene_from_weights(mixed)
        ids = torch.arange(172, dtype=torch.long)
        digits = ((ids[:, None] // (4 ** torch.arange(16, dtype=torch.long))[None]) % 4).to(torch.uint8)
        book = torch.cat((torch.full((1, 4, 4), 3, dtype=torch.uint8), digits.reshape(172, 4, 4)))
        production = RelativeSceneDynamics(book, old.object_shapes[:1].repeat(19, 1, 1), torch.ones(19, dtype=torch.long))
        self.assertEqual(sum(p.numel() for p in production.parameters()), 47085)

    def test_extended_memory_zero_projection_multistep_parity_and_metadata(self):
        old = ObjectSceneDynamics(*assets()).eval()
        model = expand_scene_weights(old.state_dict()).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in old.parameters()), 24512)
        pattern_ids = torch.arange(172, dtype=torch.long)
        digits = ((pattern_ids[:, None] // (4 ** torch.arange(16, dtype=torch.long))[None]) % 4).to(torch.uint8)
        production_book = torch.cat((torch.full((1, 4, 4), 3, dtype=torch.uint8), digits.reshape(172, 4, 4)))
        production_shapes = old.object_shapes[:1].repeat(19, 1, 1)
        production = ExtendedSceneDynamics(production_book, production_shapes, torch.ones(19, dtype=torch.long))
        self.assertEqual(sum(parameter.numel() for parameter in production.parameters()), 69547)
        names = [name for name, _ in old.named_parameters()]
        self.assertEqual([name for name, _ in model.named_parameters()][:len(names)], names)
        tokens = torch.randint(0, 3, (2, 18, 20)); state_old = None; state = None
        for action in (torch.tensor([1, 4]), torch.tensor([2, 0]), torch.tensor([3, 5])):
            before = old.scene_details(tokens, action, state_old)
            after = model.scene_details(tokens, action, state)
            state_old, state = before["state"], after["state"]
            self.assertEqual(state.shape, (2, 36, 18, 20))
            torch.testing.assert_close(state[:, :20], state_old, rtol=0, atol=0)
            for key in ("write", "gate", "mode"):
                torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
            for key in ("current", "next"):
                torch.testing.assert_close(after[key]["joint"], before[key]["joint"], rtol=0, atol=0)
            self.assertTrue(torch.equal(model.decode_scene(tokens, after, hard=True), old.decode_scene(tokens, before, hard=True)))
        padded = expand_scene_state(state_old)
        self.assertTrue(torch.equal(padded[:, :20], state_old)); self.assertTrue(torch.all(padded[:, 20:] == 0))
        with self.assertRaises(ValueError): expand_scene_state(state)
        with self.assertRaises(ValueError): model.advance(tokens, torch.tensor([0, 0]), state_old)
        with self.assertRaises(ValueError): expand_scene_weights(model.state_dict())
        self.assertIsInstance(scene_from_weights({"model": model.state_dict()}), ExtendedSceneDynamics)

    def test_extended_projection_then_gru_gradients(self):
        model = expand_scene_weights(ObjectSceneDynamics(*assets()).state_dict())
        tokens = torch.randint(0, 3, (2, 18, 20)); actions = torch.tensor([4, 2])
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        def objective():
            state = model.advance(tokens, actions)
            return state[:, :16].square().mean()
        objective().backward()
        self.assertGreater(float(model.extra_projection.weight.grad.abs().sum()), 0.)
        self.assertTrue(torch.all(model.extra_gates.weight.grad == 0))
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        objective().backward()
        for module in (model.extra_gates, model.extra_candidate, model.extra_projection):
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(float(module.weight.grad.abs().sum()), 0.)

    def test_full_object_composition_and_canvas_modes(self):
        model = ObjectSceneDynamics(*assets())
        tokens = torch.zeros(1, 18, 20, dtype=torch.int64)
        tokens[:, 2:4, 4:6] = 1
        tokens[:, 15, 10] = 2
        details = synthetic_details(model)
        prediction = model.decode_scene(tokens, details, hard=True)
        self.assertTrue(torch.all(prediction[:, 2:4, 4:6] == 0))
        self.assertEqual(int((prediction[:, :, 2:12] == 2).sum()), 5)
        self.assertEqual(int(prediction[0, 0, 0]), 2)  # HUD always uses the legacy raw rewrite.
        retained = model.decode_scene(tokens, synthetic_details(model, mode=1), hard=True)
        self.assertTrue(torch.all(retained[:, 2:4, 4:6] == 1))
        rewrite = synthetic_details(model, mode=2)
        rewrite["gate"][:] = 20
        rewritten = model.decode_scene(tokens, rewrite, hard=True)
        self.assertTrue(torch.all(rewritten[:, :, 2:12] == 2))
        # Geometry stamps over arbitrary existing content; there is no collision rule.
        details["next"] = chosen(model, 0, 14, 8)
        overlapped = model.decode_scene(tokens, details, hard=True)
        self.assertEqual(int(overlapped[0, 15, 10]), 1)

    def test_normalized_finite_soft_scene_and_all_parameter_gradients(self):
        torch.manual_seed(19)
        model = ObjectSceneDynamics(*assets())
        tokens = torch.randint(0, 3, (2, 18, 20))
        details = model.scene_details(tokens, torch.tensor([1, 3]))
        for name in ("current", "next"):
            torch.testing.assert_close(details[name]["joint"].flatten(1).logsumexp(1), torch.zeros(2), atol=1e-6, rtol=0)
        logits = model.decode_scene(tokens, details, hard=False)
        self.assertEqual(logits.shape, (2, 3, 18, 20))
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(logits.logsumexp(1), torch.zeros(2, 18, 20), atol=1e-6, rtol=0)
        F.nll_loss(logits, (tokens + 1) % 3).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(details["state"].shape, (2, 20, 18, 20))

    def test_geometry_masks_exclude_out_of_bounds_and_are_not_serialized(self):
        model = ObjectSceneDynamics(*assets())
        self.assertFalse(bool(model.valid_poses[0, 179]))
        self.assertFalse(bool(model.valid_poses[1, 160]))
        self.assertTrue(torch.all(model.shape_masks.sum((2, 3))[model.valid_poses] == 4))
        self.assertTrue(torch.all(model.shape_masks.sum((2, 3))[~model.valid_poses] == 0))
        details = model.scene_details(torch.zeros(1, 18, 20, dtype=torch.int64), torch.tensor([0]))
        self.assertTrue(torch.isneginf(details["next"]["joint"][0][~model.valid_poses]).all())
        self.assertNotIn("shape_masks", model.state_dict())
        self.assertNotIn("valid_poses", model.state_dict())
        self.assertIn("object_shapes", model.state_dict())

    def test_upgrade_preserves_parameter_prefix_and_initial_hard_predictions(self):
        torch.manual_seed(73)
        book, shapes, tiles = assets()
        old = ObservedHistoryDynamics(book).eval()
        model = ObjectSceneDynamics(book, shapes, tiles).eval()
        model.load_state_dict(upgrade_scene_weights(old.state_dict(), shapes, tiles))
        old_names = [name for name, _ in old.named_parameters()]
        self.assertEqual(model.base_parameter_count, len(old_names))
        self.assertEqual([name for name, _ in model.named_parameters()][:len(old_names)], old_names)
        tokens = torch.randint(0, 3, (2, 18, 20))
        old_state = state = None
        for action in (torch.tensor([1, 2]), torch.tensor([3, 2])):
            logits, old_state = old(tokens, action, old_state)
            predicted, state = model.predict_tokens(tokens, action, state)
            torch.testing.assert_close(predicted, logits.argmax(1))
            torch.testing.assert_close(state, old_state)
        frame = decode_tokens(tokens, book)
        expected, _ = old.predict_frame(frame, torch.tensor([1, 2]))
        actual, _ = model.predict_frame(frame, torch.tensor([1, 2]))
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
