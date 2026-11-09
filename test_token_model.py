import tempfile
import math
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from token_model import (CalibratedEventDynamics, EventTileDynamics, TileDynamics, ObservedHistoryDynamics,
                         build_codebook, decode_tokens, encode_frames, load_tile_model,
                         expand_history_kernel, upgrade_history_weights)


def patterns():
    return torch.tensor([[[0, 1, 2, 3]] * 4, [[3] * 4] * 4,
                         [[3, 0, 3, 3], [0, 1, 0, 3], [3, 0, 3, 3], [3, 3, 3, 3]]], dtype=torch.uint8)


class PatternDynamicsTest(unittest.TestCase):
    def test_codec_exact_spatial_alignment_and_unsorted_vocabulary(self):
        book = patterns()
        tokens = torch.arange(2 * 18 * 20).reshape(2, 18, 20) % len(book)
        frames = decode_tokens(tokens, book)
        self.assertEqual(frames.shape, (2, 1, 72, 80))
        np.testing.assert_array_equal(frames[0, 0, :4, :4], book[0])
        np.testing.assert_array_equal(frames[0, 0, :4, 4:8], book[1])
        np.testing.assert_array_equal(frames[0, 0, 4:8, :4], book[2])
        encoded, unknown = encode_frames(frames[:, 0].numpy(), book.numpy())
        self.assertEqual(unknown, 0)
        np.testing.assert_array_equal(encoded, tokens.numpy())
        learned_book = build_codebook(iter([frames[:1, 0].numpy(), frames[1:, 0].numpy()]))
        rebuilt, unknown = encode_frames(frames[:, 0].numpy(), learned_book.numpy())
        self.assertEqual(unknown, 0)
        torch.testing.assert_close(decode_tokens(torch.from_numpy(rebuilt), learned_book), frames)

    def test_unknown_fallback_counted_and_vocabulary_unchanged(self):
        book = np.stack((np.zeros((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=np.uint8)))
        frames = np.full((2, 72, 80), 2, dtype=np.uint8)
        tokens, unknown = encode_frames(frames, book)
        self.assertEqual(unknown, 720)
        self.assertTrue(np.all(tokens == 0))
        self.assertEqual(len(book), 2)
        with self.assertRaises(ValueError):
            build_codebook(iter([]))
        with self.assertRaises(ValueError):
            encode_frames(frames.astype(np.float32), book)

    def test_small_recurrent_action_dependent_trainable_model(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        model = TileDynamics(patterns())
        self.assertLess(sum(p.numel() for p in model.parameters()), 50000)
        tokens = torch.randint(0, 3, (2, 18, 20))
        actions = torch.tensor([1, 3])
        logits, state = model(tokens, actions)
        self.assertEqual(logits.shape, (2, 3, 18, 20))
        self.assertEqual(state.shape, (2, 16, 18, 20))
        torch.testing.assert_close(logits.argmax(1), tokens)
        changed_actions, _ = model(tokens, torch.tensor([2, 4]))
        self.assertFalse(torch.equal(logits, changed_actions))
        continued, _ = model(tokens, actions, state)
        self.assertFalse(torch.equal(logits, continued))
        torch.testing.assert_close(logits, model(tokens, actions)[0])
        F.cross_entropy(continued, (tokens + 1) % 3).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_pixel_bridge_and_persistent_codebook(self):
        torch.set_num_threads(1)
        model = TileDynamics(patterns()).eval()
        tokens = torch.arange(360).reshape(1, 18, 20) % 3
        frame = decode_tokens(tokens, model.codebook)
        predicted, state = model.predict_frame(frame, torch.tensor([0]))
        torch.testing.assert_close(predicted, frame)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patterns.pt"
            torch.save(model.state_dict(), path)
            loaded = load_tile_model(path)
            other, other_state = loaded.predict_frame(frame, torch.tensor([0]))
            torch.testing.assert_close(other, predicted)
            torch.testing.assert_close(other_state, state)
        with self.assertRaises(ValueError):
            model.predict_frame(frame.float(), torch.tensor([0]))
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 18, 20)), torch.tensor([0]))

    def test_event_model_changes_copy_gate_and_has_finite_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        model = EventTileDynamics(patterns())
        tokens = torch.randint(0, 3, (2, 18, 20))
        actions = torch.tensor([1, 4])
        write, change, state = model.forward_details(tokens, actions)
        self.assertEqual(write.shape, (2, 3, 18, 20))
        self.assertEqual(change.shape, (2, 1, 18, 20))
        self.assertEqual(state.shape, (2, 16, 18, 20))
        logits = model.combine_logits(tokens, write, change)
        torch.testing.assert_close(logits.argmax(1), tokens)
        torch.testing.assert_close(logits.logsumexp(1), torch.zeros_like(tokens, dtype=torch.float32), atol=1e-6, rtol=0)
        torch.testing.assert_close(logits, model(tokens, actions)[0])
        F.nll_loss(logits, (tokens + 1) % 3).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertNotIn("copy_strength", model.state_dict())
        self.assertTrue(torch.equal(model.change_head.bias, torch.tensor([-2.0])))

    def test_event_mixture_extremes_and_checkpoint_versions(self):
        torch.set_num_threads(1)
        tokens = torch.zeros((1, 18, 20), dtype=torch.int64)
        write = torch.zeros((1, 3, 18, 20))
        write[:, 2] = 10
        for strength, expected in ((-1000.0, 0), (1000.0, 2)):
            change = torch.full((1, 1, 18, 20), strength)
            mixture = EventTileDynamics.combine_logits(tokens, write, change)
            self.assertTrue(torch.isfinite(mixture).all())
            self.assertTrue(torch.all(mixture.argmax(1) == expected))
        with tempfile.TemporaryDirectory() as directory:
            for model_class in (TileDynamics, EventTileDynamics, CalibratedEventDynamics, ObservedHistoryDynamics):
                model = model_class(patterns()).eval()
                path = Path(directory) / f"{model_class.__name__}.pt"
                torch.save(model.state_dict(), path)
                loaded = load_tile_model(path)
                self.assertIs(type(loaded), model_class)
                frame = decode_tokens(tokens, model.codebook)
                expected, expected_state = model.predict_frame(frame, torch.tensor([0]))
                actual, actual_state = loaded.predict_frame(frame, torch.tensor([0]))
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual_state, expected_state)

    def test_calibrated_gate_keeps_raw_training_logits_and_parameter_count(self):
        torch.set_num_threads(1)
        torch.manual_seed(103)
        base = EventTileDynamics(patterns())
        model = CalibratedEventDynamics(patterns())
        mismatch = model.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(mismatch.missing_keys, ["gate_logit_offset"])
        self.assertEqual(mismatch.unexpected_keys, [])
        self.assertEqual(sum(p.numel() for p in base.parameters()), sum(p.numel() for p in model.parameters()))
        torch.testing.assert_close(model.gate_logit_offset, torch.tensor(math.log(16)))
        self.assertFalse(model.gate_logit_offset.requires_grad)
        tokens = torch.randint(0, 3, (2, 18, 20))
        actions = torch.tensor([1, 2])
        write, raw_change, state = model.forward_details(tokens, actions)
        base_write, base_change, _ = base.forward_details(tokens, actions)
        torch.testing.assert_close(write, base_write)
        torch.testing.assert_close(raw_change, base_change)
        expected = model.combine_logits(tokens, write, raw_change - math.log(16))
        actual, actual_state = model(tokens, actions)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_state, state)
        torch.testing.assert_close(actual.argmax(1), tokens)
        torch.testing.assert_close(torch.sigmoid(torch.tensor(math.log(16)) - model.gate_logit_offset), torch.tensor(0.5))
        F.nll_loss(actual, (tokens + 1) % 3).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_calibrated_checkpoint_and_keyboard_session(self):
        from play import NeuralSession
        from run import ROOT
        torch.set_num_threads(1)
        model = CalibratedEventDynamics(patterns()).eval()
        tokens = torch.arange(360).reshape(1, 18, 20) % 3
        frame = decode_tokens(tokens, model.codebook)
        context = json.loads((ROOT / "context.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            np.savez_compressed(directory / "seed.npz", frames=np.repeat(frame[:, 0].numpy(), 3, axis=0),
                                actions=np.zeros(2, dtype=np.uint8), metadata=json.dumps({"context": context, "source": "capture"}))
            for wrapped in (False, True):
                path = directory / f"calibrated-{wrapped}.pt"
                torch.save({"model": model.state_dict()} if wrapped else model.state_dict(), path)
                loaded = load_tile_model(path)
                self.assertIs(type(loaded), CalibratedEventDynamics)
                torch.testing.assert_close(loaded.gate_logit_offset, model.gate_logit_offset)
                predicted, expected_state = model.predict_frame(frame, torch.tensor([0]))
                actual, actual_state = loaded.predict_frame(frame, torch.tensor([0]))
                torch.testing.assert_close(actual, predicted)
                torch.testing.assert_close(actual_state, expected_state)
                session = NeuralSession(path, directory / "seed.npz")
                self.assertIs(type(session.model), CalibratedEventDynamics)
                self.assertEqual(session.step("left")["steps"], 1)

    def test_visible_history_ages_and_held_actions(self):
        torch.set_num_threads(1)
        model = ObservedHistoryDynamics(patterns())
        tokens = torch.zeros((2, 18, 20), dtype=torch.int64)
        actions = torch.tensor([1, 2])
        _, state = model(tokens, actions)
        self.assertEqual(state.shape, (2, 20, 18, 20))
        torch.testing.assert_close(state[:, 16], tokens.float())
        self.assertTrue(torch.all(state[:, 17] == 0))
        self.assertTrue(torch.all(state[:, 18] == 1))
        self.assertTrue(torch.all(state[0, 19] == 1))
        self.assertTrue(torch.all(state[1, 19] == 2))
        state.retain_grad()
        next_tokens = tokens.clone(); next_tokens[0, 2, 3] = 2
        logits, next_state = model(next_tokens, torch.tensor([1, 3]), state)
        self.assertEqual(float(next_state[0, 17, 2, 3]), 0)
        self.assertEqual(float(next_state[0, 17, 0, 0]), 1)
        self.assertTrue(torch.all(next_state[0, 18] == 2))
        self.assertTrue(torch.all(next_state[1, 18] == 1))
        F.nll_loss(logits, (next_tokens + 1) % 3).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(torch.all(state.grad[:, 16:] == 0))
        saturated = next_state.detach().clone()
        saturated[:, 17:19] = 255
        _, saturated = model(next_tokens, torch.tensor([1, 3]), saturated)
        self.assertTrue(torch.all(saturated[:, 17:19] == 255))
        with self.assertRaises(ValueError):
            model(tokens, actions, torch.zeros(2, 16, 18, 20))

    def test_history_upgrade_preserves_legacy_function_and_parameter_order(self):
        torch.set_num_threads(1)
        torch.manual_seed(211)
        legacy = CalibratedEventDynamics(patterns()).eval()
        model = ObservedHistoryDynamics(patterns()).eval()
        old = legacy.state_dict()
        upgraded = upgrade_history_weights(old)
        model.load_state_dict(upgraded)
        old_names = [name for name, _ in legacy.named_parameters()]
        new_names = [name for name, _ in model.named_parameters()]
        self.assertEqual(new_names[:len(old_names)], old_names)
        self.assertEqual(new_names[len(old_names):], ["tile_age_embedding.weight", "held_duration_embedding.weight"])
        self.assertTrue(torch.all(upgraded["gates.weight"][:, 40:64] == 0))
        torch.testing.assert_close(upgraded["gates.weight"][:, 64:80], old["gates.weight"][:, 40:56])
        torch.testing.assert_close(expand_history_kernel(upgraded["gates.weight"]), upgraded["gates.weight"])
        self.assertEqual(old["gates.weight"].shape[1], 56)
        tokens = torch.randint(0, 3, (2, 18, 20))
        old_state = new_state = None
        for actions in (torch.tensor([1, 2]), torch.tensor([1, 3])):
            expected, old_state = legacy(tokens, actions, old_state)
            actual, new_state = model(tokens, actions, new_state)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(new_state[:, :16], old_state, atol=1e-6, rtol=1e-5)
        self.assertEqual(model.state_channels, 20)
        self.assertEqual(legacy.state_channels, 16)

    def test_history_keyboard_session_load_and_reset(self):
        from play import NeuralSession
        from run import ROOT
        torch.set_num_threads(1)
        model = ObservedHistoryDynamics(patterns()).eval()
        tokens = torch.arange(360).reshape(1, 18, 20) % 3
        frame = decode_tokens(tokens, model.codebook)
        context = json.loads((ROOT / "context.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "history.pt"
            torch.save({"model": model.state_dict()}, path)
            np.savez_compressed(directory / "seed.npz", frames=np.repeat(frame[:, 0].numpy(), 3, axis=0),
                                actions=np.zeros(2, dtype=np.uint8), metadata=json.dumps({"context": context, "source": "capture"}))
            loaded = load_tile_model(path)
            self.assertIs(type(loaded), ObservedHistoryDynamics)
            session = NeuralSession(path, directory / "seed.npz")
            self.assertIs(type(session.model), ObservedHistoryDynamics)
            self.assertEqual(session.state.shape[1], 20)
            before = session.reset()["frame"]
            session.step("left")
            self.assertTrue(torch.all(session.state[:, 18] == 1))
            self.assertEqual(session.reset()["frame"], before)


if __name__ == "__main__":
    unittest.main()
