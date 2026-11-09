import random
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from train import sample_batch, transition_loss, rollout_metrics
from model import NeuralTetris
from run import ROOT


class TrainingTest(unittest.TestCase):
    def test_event_loss_teaches_creation_and_generated_error_correction(self):
        from train_tokens import event_loss
        book = torch.stack((torch.zeros(4, 4, dtype=torch.uint8), torch.full((4, 4), 3, dtype=torch.uint8)))
        current = torch.ones((1, 18, 20), dtype=torch.long)
        target = current.clone()
        target[0, 0, 0] = 0
        current[0, 1, 1] = 0  # Erroneous generated block must be removed even outside a real-motion mask.
        write = torch.zeros(1, 2, 18, 20, requires_grad=True)
        change = torch.zeros(1, 1, 18, 20, requires_grad=True)
        loss = event_loss(write, change, target, current, book)
        loss.backward()
        self.assertLess(write.grad[0, 0, 0, 0], 0)
        self.assertLess(write.grad[0, 1, 1, 1], 0)
        self.assertLess(change.grad[0, 0, 0, 0], 0)
        self.assertLess(change.grad[0, 0, 1, 1], 0)
        self.assertGreater(change.grad[0, 0, 2, 2], 0)
    def test_alignment_and_change_weighting(self):
        frames = np.stack([np.full((72, 80), t % 4, dtype=np.uint8) for t in range(81)])
        actions = np.array([t % 8 for t in range(80)], dtype=np.uint8)
        batch_frames, batch_actions = sample_batch([(frames, actions)], random.Random(5), 3, 32)
        self.assertEqual(batch_frames.shape, (3, 33, 72, 80))
        np.testing.assert_array_equal(batch_frames[:, :-1, 0, 0].numpy(), batch_actions.numpy() % 4)
        logits = torch.zeros(1, 4, 72, 80, requires_grad=True)
        target = torch.zeros(1, 72, 80, dtype=torch.long)
        target[:, 0, 0] = 1
        transition_loss(logits, target, torch.zeros_like(target)).backward()
        self.assertGreater(logits.grad[0, 1, 0, 0].abs(), 10 * logits.grad[0, 1, 1, 1].abs())

    def test_rollout_produces_real_comparison(self):
        torch.set_num_threads(1)
        frames = np.zeros((81, 72, 80), dtype=np.uint8)
        frames[40:, 20:24, 20:24] = 3
        actions = np.zeros(80, dtype=np.uint8)
        metrics, examples = rollout_metrics(NeuralTetris(), [(frames, actions)], horizon=40)
        self.assertGreater(metrics["64"]["changed_pixels"], 0)
        self.assertEqual(examples[0][0].shape, (41, 72, 80))
        np.testing.assert_array_equal(examples[0][1], frames[32:73])

    def test_keyboard_session_uses_only_model(self):
        from play import NeuralSession
        torch.set_num_threads(1)
        context = json.loads((ROOT / "context.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            torch.save(NeuralTetris().state_dict(), directory / "model.pt")
            np.savez_compressed(directory / "seed.npz", frames=np.zeros((3, 72, 80), dtype=np.uint8),
                                actions=np.zeros(2, dtype=np.uint8), metadata=json.dumps({"source": "capture", "context": context}))
            session = NeuralSession(directory / "model.pt", directory / "seed.npz")
            initial = session.reset()["frame"]
            self.assertEqual(session.step("left")["steps"], 1)
            self.assertEqual(session.reset()["frame"], initial)
            with self.assertRaises(ValueError):
                session.step("invalid")
            import threading
            import urllib.request
            from http.server import HTTPServer
            from play import handler_for
            server = HTTPServer(("127.0.0.1", 0), handler_for(session))
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/step",
                    data=json.dumps({"action": "left"}).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=5) as response:
                    self.assertEqual(json.load(response)["steps"], 1)
            finally:
                server.shutdown(); server.server_close(); thread.join(5)
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
