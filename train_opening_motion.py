"""Fit the horizontal repeat readout from witnessed first-piece motion."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from model import ACTIONS
from object_annotations import candidates, classify_board, shape_key
from scene_model import scene_from_weights
from train_wall_rotation import digest


def transitions(frames, actions):
    previous, held = None, 0
    for t, action in enumerate(actions):
        held = held + 1 if action == previous else 1
        previous = action
        command = ACTIONS[action]
        if command not in ("left", "right"):
            continue
        a, b = [classify_board(frame) for frame in frames[t:t+2]]
        before, after = candidates(a), candidates(b)
        if (a >= 0).sum() != 4 or (b >= 0).sum() != 4 or len(before) != 1 or len(after) != 1:
            continue
        color, cells = before[0]
        next_color, next_cells = after[0]
        if shape_key(color, cells) != shape_key(next_color, next_cells):
            continue
        old, new = np.asarray(cells), np.asarray(next_cells)
        direction = -1 if command == "left" else 1
        if (old[:, 1] + direction < 0).any() or (old[:, 1] + direction >= 10).any():
            continue
        dy, dx = new.min(0) - old.min(0)
        if dy in (0, 1) and dx in (0, direction):
            yield held, int(dx != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    if manifest["rom_sha256"] != json.loads(Path("context.json").read_text())["rom"]["sha256"]:
        raise ValueError("opening corpus refers to a different ROM")
    votes, validation = defaultdict(Counter), []
    for row in manifest["records"]:
        if row["scenario"] not in ("left_wall", "right_wall"):
            continue
        file = args.corpus / row["file"]
        if digest(file) != row["sha256"]:
            raise ValueError("opening capture digest mismatch")
        with np.load(file, allow_pickle=False) as capture:
            pairs = list(transitions(capture["frames"][:, 1::2, 1::2], capture["actions"]))
        if row["split"] == "train":
            for held, shift in pairs:
                votes[held][shift] += 1
        elif row["split"] == "validation":
            validation.extend(pairs)
    motion = torch.full((256,), -1, dtype=torch.int8)
    for held, targets in votes.items():
        if len(targets) == 1:
            motion[held] = next(iter(targets))
    covered = [(h, v) for h, v in validation if motion[h] >= 0]
    if not covered or (motion == 1).sum() == 0:
        raise ValueError("opening corpus has no usable horizontal repeat supervision")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {**checkpoint["model"], "lateral_motion": motion}
    scene_from_weights(state)
    report = {"source_sha256": digest(args.checkpoint), "train_examples": sum(sum(v.values()) for v in votes.values()),
              "validation_examples": len(validation), "validation_covered": len(covered),
              "validation_exact": sum(int(motion[h]) == v for h, v in covered),
              "repeat_steps": torch.nonzero(motion == 1).flatten().tolist()}
    args.output.mkdir()
    torch.save({**checkpoint, "model": state, "opening_fit": report}, args.output / "model.pt")
    report["checkpoint_sha256"] = digest(args.output / "model.pt")
    (args.output / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
