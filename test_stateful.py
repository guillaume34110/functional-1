import random
import unittest
import numpy as np
import torch
from train_stateful import StreamWindows, calibrated_loss, checkpoint_baseline, observed_stream_state, expand_history_optimizer


class StatefulTrainingTest(unittest.TestCase):
    def test_observed_context_comes_from_recorded_history_only(self):
        frames = np.ones((41, 18, 20), dtype=np.uint8)
        frames[39, 0, 0] = 2
        actions = np.ones(40, dtype=np.uint8); actions[-4:] = 3
        hidden = torch.randn(1, 16, 18, 20)
        state = observed_stream_state(hidden, [(0, 40)], [(frames, actions)])
        torch.testing.assert_close(state[:, :16], hidden)
        self.assertEqual(state.shape, (1, 20, 18, 20))
        self.assertEqual(state[0, 16, 0, 0], 2)
        self.assertEqual(state[0, 17, 0, 0], 0)
        self.assertEqual(state[0, 17, 1, 1], 39)
        self.assertEqual(state[0, 18, 0, 0], 4)
        self.assertEqual(state[0, 19, 0, 0], 3)

    def test_optimizer_migration_preserves_moments_and_can_step(self):
        from token_model import CalibratedEventDynamics, ObservedHistoryDynamics, upgrade_history_weights
        torch.set_num_threads(1)
        book = torch.stack((torch.zeros(4, 4, dtype=torch.uint8), torch.full((4, 4), 3, dtype=torch.uint8)))
        old = CalibratedEventDynamics(book)
        optimizer = torch.optim.AdamW(old.parameters(), lr=3e-4)
        tokens = torch.zeros(1, 18, 20, dtype=torch.long)
        actions = torch.tensor([1])
        old(tokens, actions)[0].square().mean().backward(); optimizer.step()
        old_optimizer = optimizer.state_dict()
        model = ObservedHistoryDynamics(book)
        model.load_state_dict(upgrade_history_weights(old.state_dict()))
        upgraded = expand_history_optimizer(old_optimizer, model)
        names = [n for n, _ in model.named_parameters()]
        identifier = old_optimizer["param_groups"][0]["params"][names.index("gates.weight")]
        before = old_optimizer["state"][identifier]["exp_avg"]
        after = upgraded["state"][identifier]["exp_avg"]
        self.assertEqual(before.shape[1], 56)
        torch.testing.assert_close(after[:, :40], before[:, :40])
        torch.testing.assert_close(after[:, 64:80], before[:, 40:56])
        self.assertEqual(float(after[:, 40:64].abs().sum()), 0)
        resumed = torch.optim.AdamW(model.parameters(), lr=3e-4)
        resumed.load_state_dict(upgraded)
        model(tokens, actions)[0].square().mean().backward(); resumed.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_resume_keeps_existing_selection_baseline(self):
        checkpoint = {"config": {"protocol": "stateful-contiguous-calibrated"},
                      "teacher": {"addition_accuracy": 0.9, "erasure_accuracy": 0.9, "gate_fpr": 0.01},
                      "validation": {"tile_error": 0.1}}
        self.assertAlmostEqual(checkpoint_baseline(checkpoint), 0.4)
        self.assertEqual(checkpoint_baseline({"config": {"protocol": "older"}}), float("inf"))
    def test_contiguous_history_and_terminal_padding(self):
        frames = np.stack([np.full((18, 20), t, dtype=np.int64) for t in range(81)])
        actions = np.arange(80, dtype=np.uint8) % 8
        sampler = StreamWindows([(frames, actions)], 1, random.Random(4), length=8)
        first = sampler.next()
        second = sampler.next()
        self.assertTrue(first[3][0]); self.assertFalse(second[3][0])
        torch.testing.assert_close(first[0][:, -1], second[0][:, 0])
        self.assertEqual(int(second[0][0, 0, 0, 0]), 40)
        self.assertEqual(first[4].shape, (1, 32, 18, 20))
        short = StreamWindows([(frames[:36], actions[:35])], 1, random.Random(4), length=8)
        window = short.next()
        self.assertEqual(int(window[2].sum()), 3)
        self.assertTrue(torch.all(window[0][0, 3:] == 35))
        self.assertTrue(short.next()[3][0])

    def test_loss_ignores_padded_stream_and_has_valid_gradients(self):
        book = torch.stack((torch.zeros(4, 4, dtype=torch.uint8), torch.full((4, 4), 3, dtype=torch.uint8)))
        current = torch.ones(2, 18, 20, dtype=torch.long)
        target = current.clone(); target[0, 2, 2] = 0; target[1] = 0
        write = torch.zeros(2, 2, 18, 20, requires_grad=True)
        change = torch.zeros(2, 1, 18, 20, requires_grad=True)
        loss = calibrated_loss(write, change, target, current, book, torch.tensor([True, False]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(change.grad[0, 0, 2, 2], 0)
        self.assertEqual(float(change.grad[1].abs().sum()), 0)
        self.assertEqual(float(write.grad[1].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
