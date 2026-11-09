"""Fit the visible idle transition clock on independent ROM captures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_controls import erase, supported
from model import ACTIONS
from object_annotations import candidates, classify_board
from scene_model import scene_from_weights


def observations(frames, actions):
    """Keep the first piece, where the active cells are independently identifiable."""
    age = 0
    entered = False
    for t, action in enumerate(actions):
        board = classify_board(frames[t])
        poses = candidates(board)
        if len(poses) != 1 or (board >= 0).sum() != 4:
            if entered:
                break
            continue
        if int(action) != ACTIONS.index("noop"):
            raise ValueError("idle witness contains an active command")
        entered = True
        color, cells = poses[0]
        cells = np.asarray(cells)
        after = classify_board(frames[t + 1])
        if np.array_equal(after, board):
            yield age, False
            age += 1
            continue
        shifted = cells + [1, 0]
        if shifted[:, 0].max() < 18 and (after >= 0).sum() == 4:
            pose = candidates(after)
            if len(pose) == 1 and pose[0][0] == color and np.array_equal(np.asarray(pose[0][1]), shifted):
                yield age, True
                age = 0
                continue
        if ((after >= 0).sum() > 4 and np.all(after[tuple(cells.T)] == color)
                and supported(erase(board, cells, -1), cells, -1)):
            yield age, True
        break


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    context = json.loads(Path("context.json").read_text())
    if manifest["rom_sha256"] != context["rom"]["sha256"]:
        raise ValueError("idle corpus refers to a different ROM")
    samples = {"train": [], "validation": []}
    for row in manifest["records"]:
        if row["scenario"] != "gravity":
            continue
        file = args.corpus / row["file"]
        if digest(file) != row["sha256"]:
            raise ValueError("idle capture digest mismatch")
        with np.load(file, allow_pickle=False) as capture:
            samples[row["split"]].extend(observations(capture["frames"][:, 1::2, 1::2], capture["actions"]))
    train = np.asarray(samples["train"])
    valid = np.asarray(samples["validation"])
    if min(len(train), len(valid)) == 0 or len(np.unique(train[:, 1])) != 2:
        raise ValueError("incomplete idle supervision")
    offsets = np.arange(256)
    errors = ((train[:, :1] >= offsets) != train[:, 1:]).sum(0)
    threshold = int(errors.argmin())
    learned = torch.arange(256, dtype=torch.float32) - threshold + 0.5
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {**checkpoint["model"], "gravity_transition": learned}
    scene_from_weights(state)
    report = {"source_sha256": digest(args.checkpoint), "threshold_age": threshold,
              "train_examples": len(train), "train_errors": int(errors[threshold]),
              "validation_examples": len(valid),
              "validation_errors": int(((valid[:, 0] >= threshold) != valid[:, 1]).sum())}
    args.output.mkdir()
    torch.save({**checkpoint, "model": state, "gravity_fit": report}, args.output / "model.pt")
    report["checkpoint_sha256"] = digest(args.output / "model.pt")
    (args.output / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
