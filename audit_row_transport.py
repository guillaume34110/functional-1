"""Measure how much of proved training rewrites can copy complete visible rows."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from scene_model import scene_from_weights
from train_scene import composition_targets, label_mask, visual_absence
from train_spawn_preview import sha, training_captures


def compare_rows(source, target, blank):
    equal = (target[:, None] == source[None]).all(2)
    blank_target = (target == blank).all(1)
    copyable = blank_target | equal.any(1)
    # A monotone transport may skip source rows and insert blank rows.
    source_index = -1
    ordered = True
    for i in np.flatnonzero(~blank_target):
        available = np.flatnonzero(equal[i] & (np.arange(len(source)) > source_index))
        if not len(available):
            ordered = False
            break
        source_index = available[0]
    return int(copyable.sum()), bool(ordered)


def compacted_background(source, blank):
    full = (source != blank).all(1)
    count = int(full.sum())
    if not 1 <= count <= 4:
        return None
    return np.concatenate((np.full((count, source.shape[1]), blank, dtype=source.dtype), source[~full]))


def classify_clear(source, target, blank):
    shifted = compacted_background(source, blank)
    return int((source != blank).all(1).sum()) if shifted is not None and np.array_equal(shifted, target) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = scene_from_weights(checkpoint["model"])
    counts = Counter()
    events = []
    board_vocab = np.r_[model.blank_id, model.object_tiles.numpy()]
    for name, frames, actions, labels in training_captures(checkpoint["config"], model.codebook.numpy()):
        before, after = frames[:-1], frames[1:]
        valid = torch.ones(len(actions), dtype=torch.bool)
        old, old_mask = label_mask({k: v[:-1, None] for k, v in labels.items()}, 0, valid)
        new, new_mask = label_mask({k: v[1:, None] for k, v in labels.items()}, 0, valid)
        old_known = old | visual_absence(before, labels["evidence"][:-1], valid, model.object_tiles)
        new_known = new | visual_absence(after, labels["evidence"][1:], valid, model.object_tiles)
        modes, known = composition_targets(model, before, after, old_mask, new_mask, old_known, new_known)
        for t in torch.nonzero(known & (modes == 2)).flatten().tolist():
            counts["known_rewrites"] += 1
            if not bool(old_known[t] and new_known[t]):
                counts["unproven_foreground"] += 1
                continue
            source = before[t, :, 2:12].numpy()
            erased = before[t].masked_fill(old_mask[t], model.blank_id)[:, 2:12].numpy()
            target = after[t].masked_fill(new_mask[t], model.blank_id)[:, 2:12].numpy()
            if not np.isin(source, board_vocab).all() or not np.isin(target, board_vocab).all():
                counts["non_piece_palette"] += 1
                continue
            counts["ordinary_known_boards"] += 1
            rows, ordered = compare_rows(source, target, model.blank_id)
            rows_erased, ordered_erased = compare_rows(erased, target, model.blank_id)
            counts["target_rows"] += len(target)
            counts["rows_copyable_full"] += rows
            counts["rows_copyable_erased"] += rows_erased
            counts["exact_unordered_full"] += rows == len(target)
            counts["exact_ordered_full"] += ordered
            counts["exact_ordered_erased"] += ordered_erased
            cleared = classify_clear(source, target, model.blank_id)
            if cleared:
                counts[f"exact_full_row_removal/{cleared}"] += 1
            events.append({"file": name, "step": t, "rows_full": rows, "rows_erased": rows_erased,
                           "ordered_full": ordered, "ordered_erased": ordered_erased,
                           "exact_full_row_removal": cleared})
    report = {"fit_used": False, "split": "train", "test_used": False, "model_forward_used": False,
              "checkpoint_sha256": sha(args.checkpoint), "manifest_sha256": checkpoint["config"]["manifest_sha256"],
              "counts": counts, "events": events}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
