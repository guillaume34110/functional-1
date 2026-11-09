"""Offline visual object evidence only; labels must never become runtime inputs.

Block art: Dave VanEe, CC0, Adjustris v1.1 gfx/Global_Blocks.asm.
https://github.com/tbsp/Adjustris/blob/v1.1/gfx/Global_Blocks.asm
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from run import read_episode

BLOCK_BYTES = bytes.fromhex(
    "ffffff81ff81e799e799ff81ff81ffff ffff8181bd81a599a599bd818181ffff "
    "ffffff81c3bddbbddbbdc3bdff81ffff ffff8181bd81bd99bd99bd818181ffff "
    "ffffff81c3bdc3bdc3bdc3bdff81ffff ffff8181bd81bd81bd81bd818181ffff "
    "ffffff81c381c399c399c381ff81ffff")


def texture_patterns():
    data = np.frombuffer(BLOCK_BYTES, dtype=np.uint8).reshape(7, 8, 2)
    bits = np.arange(7, -1, -1)
    native = 3 - (((data[:, :, 0, None] >> bits) & 1) + 2 * ((data[:, :, 1, None] >> bits) & 1))
    patterns = np.stack([np.asarray(Image.fromarray(tile.astype(np.uint8)).resize((4, 4), Image.Resampling.NEAREST))
                         for tile in native])
    assert np.array_equal(patterns, native[:, 1::2, 1::2])
    assert len(np.unique(patterns.reshape(7, 16), axis=0)) == 7
    return patterns


PATTERNS = texture_patterns()


def classify_board(frame):
    if frame.dtype != np.uint8 or frame.shape != (72, 80) or np.any(frame > 3):
        raise ValueError("expected palette uint8 frame [72,80]")
    patches = frame[:, 8:48].reshape(18, 4, 10, 4).transpose(0, 2, 1, 3)
    matches = np.all(patches[:, :, None] == PATTERNS[None, None], axis=(3, 4))
    grid = np.where(matches.any(2), matches.argmax(2), -2).astype(np.int8)
    grid[np.all(patches == 3, axis=(2, 3))] = -1
    return grid


def candidates(grid):
    result = []
    for texture in range(7):
        remaining = set(map(tuple, np.argwhere(grid == texture)))
        while remaining:
            stack = [remaining.pop()]
            component = set(stack)
            while stack:
                y, x = stack.pop()
                for cell in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if cell in remaining:
                        remaining.remove(cell); component.add(cell); stack.append(cell)
            if len(component) == 4:
                result.append((texture, tuple(sorted(component))))
    return result


def without(grid, cells):
    result = grid.copy()
    for cell in cells:
        result[cell] = -1
    return result


def annotate_sequence(frames):
    """Find unique complete-object motion; stable carry requires unchanged whole board.

    Evidence: 0 unknown, 1 motion-pair proof, 2 stable carry, 3 invalid patterns,
    4 multiple possible moving objects. A pair may label its previous frame:
    this is supervision using a future observation, never a causal model input.
    """
    if frames.ndim != 3 or len(frames) < 1:
        raise ValueError("expected nonempty frame sequence")
    labels = {"cells": np.full((len(frames), 4, 2), -1, dtype=np.int16),
              "texture": np.full(len(frames), -1, dtype=np.int8),
              "track": np.full(len(frames), -1, dtype=np.int32),
              "evidence": np.zeros(len(frames), dtype=np.uint8)}
    previous, old_candidates, groups, motions = None, [], 0, 0
    previous_was_post_invalid = False
    for t, frame in enumerate(frames):
        grid = classify_board(frame)
        current_candidates = candidates(grid) if np.all(grid >= -1) else []
        if np.any(grid < -1):
            labels["evidence"][t] = 3
        elif previous is not None and np.all(previous >= -1) and not previous_was_post_invalid:
            pairs = []
            changed = np.any(grid != previous)
            if changed and np.count_nonzero(grid != previous) <= 8:
                for old_texture, old_cells in old_candidates:
                    for texture, cells in current_candidates:
                        if texture == old_texture and cells != old_cells and np.array_equal(without(previous, old_cells), without(grid, cells)):
                            pairs.append((texture, old_cells, cells))
            if len(pairs) == 1:
                texture, old_cells, cells = pairs[0]
                same_track = labels["texture"][t - 1] == texture and np.array_equal(labels["cells"][t - 1], old_cells)
                group = int(labels["track"][t - 1]) if same_track else groups
                groups += int(not same_track)
                for index, pose in ((t - 1, old_cells), (t, cells)):
                    labels["cells"][index], labels["texture"][index] = pose, texture
                    labels["track"][index], labels["evidence"][index] = group, 1
                motions += 1
            elif len(pairs) > 1:
                labels["evidence"][t] = 4
            elif not changed and labels["texture"][t - 1] >= 0:
                pose = tuple(map(tuple, labels["cells"][t - 1]))
                if (int(labels["texture"][t - 1]), pose) in current_candidates:
                    for name in ("cells", "texture", "track"):
                        labels[name][t] = labels[name][t - 1]
                    labels["evidence"][t] = 2
        previous_was_post_invalid = previous is not None and np.any(previous < -1)
        previous, old_candidates = grid, current_candidates
    valid = labels["texture"] >= 0
    return labels, {"frames": len(frames), "labeled_frames": int(valid.sum()), "motion_pairs": motions,
                    "groups": len(np.unique(labels["track"][valid])),
                    "invalid_pattern_frames": int((labels["evidence"] == 3).sum()),
                    "ambiguous_motion_frames": int((labels["evidence"] == 4).sum())}


def shape_key(texture, cells):
    cells = np.asarray(cells)
    return (int(texture), *tuple((cells - cells.min(axis=0)).ravel().tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = (args.data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "complete":
        raise ValueError("dataset must be complete")
    args.output.mkdir(parents=True, exist_ok=False)
    dictionary, report = {}, {"test_used": False, "dictionary_from": "train-only", "episodes": [],
                              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    for split in ("train", "validation"):
        for i, entry in enumerate(manifest["episodes"]):
            if entry["split"] != split:
                continue
            source = (args.data / entry["file"]).resolve()
            if not source.is_relative_to(args.data.resolve()) or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("episode path or digest mismatch")
            frames, _, metadata = read_episode(source)
            if metadata.get("source") != "capture":
                raise ValueError("annotations require reference captures")
            labels, stats = annotate_sequence(frames)
            ids, unknown = np.full(len(frames), -1, dtype=np.int32), set()
            for t in np.flatnonzero(labels["texture"] >= 0):
                key = shape_key(labels["texture"][t], labels["cells"][t])
                if split == "train" and key not in dictionary:
                    dictionary[key] = len(dictionary)
                ids[t] = dictionary.get(key, -1)
                if ids[t] < 0:
                    unknown.add(key)
            stats.update({"file": entry["file"], "split": split,
                          "unknown_shape_frames": int(((ids < 0) & (labels["texture"] >= 0)).sum()), "unknown_shapes": len(unknown)})
            folder = args.output / split
            folder.mkdir(exist_ok=True)
            label_file = folder / f"episode-{i:03d}.npz"
            stats["labels_file"] = str(label_file.relative_to(args.output))
            np.savez_compressed(label_file, **labels, shape_id=ids,
                                metadata=json.dumps({"source_sha256": entry["sha256"], "future_pair_is_target_only": True}))
            report["episodes"].append(stats)
    report["dictionary"] = [{"id": value, "texture": key[0], "cells": np.array(key[1:]).reshape(4, 2).tolist()}
                            for key, value in dictionary.items()]
    report["shapes_per_texture"] = {str(k): sum(key[0] == k for key in dictionary) for k in range(7)}
    report["coverage"] = {split: {"frames": sum(e["frames"] for e in report["episodes"] if e["split"] == split),
                                  "labeled_frames": sum(e["labeled_frames"] for e in report["episodes"] if e["split"] == split)}
                          for split in ("train", "validation")}
    (args.output / "object-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(args.output / "object-audit.json"), "coverage": report["coverage"],
                      "shapes_per_texture": report["shapes_per_texture"]}), flush=True)


if __name__ == "__main__":
    main()
