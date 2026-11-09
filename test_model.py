import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from model import ACTIONS, NeuralTetris
from run import ROOT, capture, generate, main, read_episode


class DynamicsTest(unittest.TestCase):
    def test_size_memory_actions_and_gradient(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = NeuralTetris()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 9776)
        frame = torch.randint(0, 4, (2, 1, 72, 80), dtype=torch.uint8)
        action = torch.tensor([0, 2])
        logits, state = model(frame, action)
        self.assertEqual(logits.shape, (2, 4, 72, 80))
        self.assertEqual(state.shape, (2, 8, 18, 20))
        resumed, _ = model(frame, action, state)
        self.assertFalse(torch.equal(logits, resumed))
        self.assertFalse(torch.equal(logits, model(frame, torch.tensor([1, 3]))[0]))
        torch.testing.assert_close(logits, model(frame, action)[0])
        target = torch.randint(0, 4, (2, 72, 80))
        torch.nn.functional.cross_entropy(resumed, target).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_reject_wrong_frame(self):
        for frame in (torch.zeros(1, 1, 72, 80), torch.full((1, 1, 72, 80), 255, dtype=torch.uint8)):
            with self.assertRaises(ValueError):
                NeuralTetris()(frame, torch.zeros(1, dtype=torch.int64))

    def test_record_then_generate_without_emulator(self):
        context = json.loads((ROOT / "context.json").read_text())
        actions = ["noop"] * 30 + ["a", "noop", "start", "left", "right", "down", "b", "up"]
        frames = capture(actions, context)
        self.assertEqual(frames.shape, (len(actions) + 1, 72, 80))
        self.assertEqual(frames.dtype, np.uint8)
        self.assertGreater(np.unique(frames).size, 1)
        self.assertLessEqual(frames.max(), 3)
        self.assertFalse(np.array_equal(frames[0], frames[-1]))
        model = NeuralTetris().eval()
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            np.savez_compressed(temp / "seed.npz", frames=frames, actions=np.array([ACTIONS.index(a) for a in actions], dtype=np.uint8),
                                metadata=json.dumps({"context": context, "source": "capture"}))
            seed_frames, seed_actions, _ = read_episode(temp / "seed.npz")
            torch.save(model.state_dict(), temp / "weights.pt")
            (temp / "buttons.json").write_text(json.dumps(["left", "a", "noop"]))
            argv = ["run.py", "generate", "--actions", str(temp / "buttons.json"), "--checkpoint", str(temp / "weights.pt"),
                    "--seed", str(temp / "seed.npz"), "--output", str(temp / "predicted.npz")]
            with patch("sys.argv", argv), patch("run.capture", side_effect=AssertionError("generation must not use the ROM")):
                main()
            predicted, labels, _ = read_episode(temp / "predicted.npz")
            self.assertEqual(predicted.shape, (4, 72, 80))
            np.testing.assert_array_equal(labels, [1, 4, 0])
            np.testing.assert_array_equal(predicted[0], frames[-1])
            np.testing.assert_array_equal(predicted, generate(model, seed_frames, seed_actions, ["left", "a", "noop"]))
            with patch("sys.argv", argv), self.assertRaises(FileExistsError):
                main()
            argv[argv.index("--seed") + 1] = str(temp / "predicted.npz")
            argv[argv.index("--output") + 1] = str(temp / "invalid.npz")
            with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "reference capture"):
                main()


if __name__ == "__main__":
    unittest.main()
