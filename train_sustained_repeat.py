"""Fit directional repeat after the first lock in uninterrupted ROM input."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_spawn import spawn_evidence
from object_annotations import candidates, classify_board, shape_key
from run import ROOT
from scene_model import scene_from_weights
from train_native_render import capture
from train_wall_rotation import digest


def repeats(frames, direction):
    boards = [classify_board(frame) for frame in frames]
    lock = None
    for i, (before, after) in enumerate(zip(boards[:-1], boards[1:], strict=True)):
        if (before >= 0).sum() == 4 and (after >= 0).sum() == 8:
            visible = candidates(before)
            if len(visible) == 1 and spawn_evidence(frames[i], frames[i + 1], np.asarray(visible[0][1])):
                lock = i + 1
                break
    if lock is None:
        raise ValueError("held direction did not reach a visible first lock")
    background = boards[lock - 1]
    for i in range(lock, min(lock + 48, len(boards) - 1)):
        observed = boards[i:i+2]
        if any(np.any(board[background >= 0] != background[background >= 0]) for board in observed):
            break
        old, new = [candidates(np.where(background >= 0, -1, board)) for board in observed]
        if len(old) != 1 or len(new) != 1 or shape_key(*old[0]) != shape_key(*new[0]):
            break
        _, cells = old[0]
        shifted = np.asarray(cells) + [0, direction]
        if (shifted[:, 1] < 0).any() or (shifted[:, 1] >= 10).any() or (background[tuple(shifted.T)] >= 0).any():
            continue
        delta = np.min(new[0][1], axis=0) - np.min(cells, axis=0)
        if delta[0] in (0, 1) and delta[1] in (0, direction):
            yield i - lock, int(delta[1] != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = json.loads((ROOT / "context.json").read_text())
    rom = ROOT / context["rom"]["file"]
    if digest(rom) != context["rom"]["sha256"]:
        raise ValueError("reference ROM checksum mismatch")
    args.output.mkdir()
    examples = {"train": [], "validation": []}
    for split, boots in (("train", range(5101, 5101 + 10 * 53, 53)),
                         ("validation", range(6401, 6401 + 5 * 67, 67))):
        for boot in boots:
            for button, direction in (("left", -1), ("right", 1)):
                native = capture(rom, boot, ("a",) + ("noop",) * 4 + (button,) * 400)
                low = native[:, 1::2, 1::2]
                examples[split].extend(repeats(low, direction))
    train = np.asarray(examples["train"])
    valid = np.asarray(examples["validation"])
    if min(len(train), len(valid)) == 0:
        raise ValueError("missing observed held-direction repeats")
    trials = []
    for period in range(2, 9):
        offsets = train[:, 0] % period
        motion = np.array([np.bincount(train[offsets == i, 1], minlength=2).argmax() for i in range(period)])
        trials.append((int((motion[offsets] != train[:, 1]).sum()), period, motion))
    errors, period, motion = min(trials, key=lambda x: x[:2])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {**checkpoint["model"], "sustained_repeat": torch.tensor(motion, dtype=torch.int8)}
    scene_from_weights(state)
    report = {"source_sha256": digest(args.checkpoint), "rom_sha256": context["rom"]["sha256"],
              "training_examples": len(train), "train_errors": errors, "validation_examples": len(valid),
              "validation_errors": int((motion[valid[:, 0] % period] != valid[:, 1]).sum()),
              "repeat_pattern": motion.tolist()}
    torch.save({**checkpoint, "model": state, "sustained_repeat_fit": report}, args.output / "model.pt")
    report["checkpoint_sha256"] = digest(args.output / "model.pt")
    (args.output / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
