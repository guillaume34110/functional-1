"""Compare sustained wall input against the native reference."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from checks.reference_game import record
from object_annotations import classify_board
from play import NeuralSession, RELEASE


def activity(frames, title):
    matches = np.flatnonzero((frames[:, :len(title)] == title).all((1, 2)))
    terminal = int(matches[0]) if len(matches) else None
    game = frames[:terminal]
    board = game[:, :, 8:48]
    change = np.flatnonzero((board[1:] != board[:-1]).any((1, 2))) + 1
    runs = np.diff(np.r_[0, change, len(board)])
    settled = np.array([int((classify_board(frame) >= 0).sum()) for frame in game])
    return {"return_to_title": terminal, "visible_deposits": int((np.diff(settled) == 4).sum()),
            "longest_static_frames": int(runs.max()), "stationary_from": int(np.r_[0, change][runs.argmax()]),
            "board_changes": len(change)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    torch.set_num_threads(1)
    seed = RELEASE / "first-piece.npz"
    game = NeuralSession(args.checkpoint, seed)
    result = {}
    for action in ("left", "right"):
        game.reset()
        initial = game.frame[0, 0].numpy().copy()
        title, native, commands, _ = record(initial, (action,) * 800, 800)
        frames = [initial]
        for button in commands:
            game.step(button)
            frames.append(game.frame[0, 0].numpy().copy())
        frames = np.stack(frames)
        reference = native[:, 1::2, 1::2]
        signature = title[1:118:2, 1::2]
        result[action] = {"reference": activity(reference, signature), "candidate": activity(frames, signature)}
        np.savez_compressed(args.output / f"{action}.npz", reference=reference, predicted=frames)
    (args.output / "held-wall.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
