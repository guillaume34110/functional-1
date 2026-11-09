"""Learn where pieces appear after observed idle-gravity deposits."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_spawn import spawn_evidence
from object_annotations import PATTERNS, candidates, classify_board, shape_key
from scene_model import scene_from_weights
from train_wall_rotation import digest


def spawns(frames, kinds):
    for before, after in zip(frames[:-1], frames[1:], strict=True):
        board = classify_board(before)
        if (board >= 0).sum() != 4:
            continue
        poses = candidates(board)
        if len(poses) != 1:
            continue
        event = spawn_evidence(before, after, np.asarray(poses[0][1]))
        if event is None or event["preview"] != event["texture"]:
            continue
        cells = np.asarray(event["cells"])
        key = shape_key(event["texture"], sorted(event["cells"]))
        if key in kinds:
            y, x = map(int, cells.min(0))
            yield kinds[key], y * 10 + x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint)
    kinds = {}
    for i, (shape, token) in enumerate(zip(base.object_shapes, base.object_tiles, strict=True)):
        textures = np.flatnonzero((PATTERNS == base.codebook[token].numpy()).all((1, 2)))
        if len(textures) == 1:
            kinds[shape_key(int(textures[0]), sorted(shape.tolist()))] = i
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    if manifest["rom_sha256"] != json.loads(Path("context.json").read_text())["rom"]["sha256"]:
        raise ValueError("spawn corpus refers to a different ROM")
    targets = {"train": [], "validation": []}
    for row in manifest["records"]:
        if row["scenario"] != "gravity":
            continue
        file = args.corpus / row["file"]
        if digest(file) != row["sha256"]:
            raise ValueError("spawn capture digest mismatch")
        with np.load(file, allow_pickle=False) as capture:
            targets[row["split"]].extend(spawns(capture["frames"][:, 1::2, 1::2], kinds))
    allowed = torch.zeros_like(base.valid_poses)
    for kind, pose in targets["train"]:
        allowed[kind, pose] = True
    if not targets["validation"] or set(base.object_tiles[allowed.any(1)].tolist()) != set(base.object_tiles.tolist()):
        raise ValueError("spawn corpus does not cover all piece textures")
    state = {**checkpoint["model"], "gravity_spawn_poses": allowed}
    scene_from_weights(state)
    report = {"source_sha256": digest(args.checkpoint), "training_examples": len(targets["train"]),
              "validation_examples": len(targets["validation"]),
              "validation_exact": sum(bool(allowed[k, p]) for k, p in targets["validation"]),
              "observed_spawn_poses": allowed.nonzero().tolist()}
    args.output.mkdir()
    torch.save({**checkpoint, "model": state, "gravity_spawn_fit": report}, args.output / "model.pt")
    report["checkpoint_sha256"] = digest(args.output / "model.pt")
    (args.output / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
