"""Find strictly proved clear events in parent train captures, offline only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_row_transport import compacted_background
from run import ROOT
from scene_model import scene_from_weights
from train_spawn_preview import observed_captures, sha


MAX_DELAY = 24


def foreground_background(board, labels, shapes, tiles, blank, step):
    shape = int(labels["shape_id"][step])
    if int(labels["evidence"][step]) not in (1, 2) or not 0 <= shape < len(shapes):
        return None
    cells = labels["cells"][step]
    if (np.any(cells < 0) or np.any(cells >= np.array(board.shape))
            or not np.array_equal(np.unique(cells - cells.min(0), axis=0), np.unique(shapes[shape], axis=0))
            or not np.all(board[cells[:, 0], cells[:, 1]] == tiles[shape])):
        return None
    background = board.copy()
    background[cells[:, 0], cells[:, 1]] = blank
    return background


def board_hash(board):
    return hashlib.sha256(board.astype("<i8", copy=False).tobytes()).hexdigest()


def exact_foreground_pose(board, background, shapes, tiles, blank):
    """Prove a background with a unique full pose or a compatible clipped piece."""
    changed = board != background
    cells = np.argwhere(changed)
    if not len(cells):
        return -1
    if len(cells) > 4 or (background[changed] != blank).any():
        return None
    colors = board[changed]
    if not np.all(colors == colors[0]):
        return None
    y, x = cells.min(0)
    motif = cells - (y, x)
    if len(cells) == 4:
        kinds = [int(kind) for kind in np.flatnonzero(tiles == colors[0])
                 if np.array_equal(np.unique(shapes[kind], axis=0), motif)]
        return kinds[0] * 180 + int(y) * 10 + int(x) if len(kinds) == 1 else None
    if y != 0:
        return None
    for kind in np.flatnonzero(tiles == colors[0]):
        shape = shapes[kind]
        for lift in range(1, int(shape[:, 0].max()) + 1):
            visible = shape[shape[:, 0] >= lift]
            if len(visible) != len(cells):
                continue
            column = x - visible[:, 1].min()
            if column < 0 or column + shape[:, 1].max() >= board.shape[1]:
                continue
            placed = visible + (-lift, column)
            if np.array_equal(np.unique(placed, axis=0), cells):
                return -2
    return None


def clear_events(boards, labels, shapes, tiles, blank, max_delay=MAX_DELAY, visible_overlay=False):
    ordinary = np.isin(boards, np.r_[blank, tiles]).all((1, 2))
    full = (boards != blank).all(2)
    full_count = full.sum(1)
    counts = Counter(frames=len(boards), ordinary_full_frames=int((ordinary & (full_count > 0)).sum()))
    events = []
    t = 0
    while t < len(boards):
        if not ordinary[t] or not 1 <= full_count[t] <= 4:
            t += 1
            continue
        start = t
        source = boards[t]
        while t + 1 < len(boards) and np.array_equal(boards[t + 1], source):
            t += 1
        lock = t
        counts["distinct_lock_candidates"] += 1
        rows = np.flatnonzero(full[lock])
        compacted = compacted_background(source, blank)
        flashed, first_match, first_unconfirmed = [], None, None
        matched = []
        rejected = None
        for step in range(lock + 1, min(lock + max_delay + 2, len(boards))):
            board = boards[step]
            if np.array_equal(source, board):
                first_match = None
                t = step  # A flashing repetition is part of the same visible lock.
                continue
            if ordinary[step] and full_count[step]:
                rejected = "different_full_board_before_confirmation"
                break
            match = None
            if ordinary[step]:
                if np.array_equal(compacted, board):
                    match = ("raw", board)
                else:
                    background = foreground_background(board, labels, shapes, tiles, blank, step)
                    if background is not None and np.array_equal(compacted, background):
                        match = ("masked_proved_foreground", background)
                    elif visible_overlay:
                        pose = exact_foreground_pose(board, compacted, shapes, tiles, blank)
                        if pose is not None and pose >= 0:
                            match = ("masked_exact_foreground", compacted)
                        elif pose == -2 and first_match is not None:
                            match = ("masked_clipped_foreground", compacted)
            if match is None:
                first_match = None
                changed_rows = np.flatnonzero(np.any(board != source, axis=1))
                if len(changed_rows) and np.isin(changed_rows, rows).all():
                    flashed.append(step)
                continue
            if first_match is None:
                if step - lock <= max_delay:
                    first_match = step
                    first_unconfirmed = step
                    matched = [match]
                continue
            if not any(index < first_match for index in flashed):
                rejected = "exact_pair_without_visible_flash"
                break
            matched.append(match)
            events.append({"first_lock_frame": start, "last_consecutive_lock_frame": lock,
                           "flash_frames": [index for index in flashed if index < first_match],
                           "first_outcome_frame": first_match, "confirmation_frame": step,
                           "delay_from_lock_frames": first_match - lock, "delay_from_first_lock_frames": first_match - start,
                           "rows_removed": rows.tolist(), "row_count": len(rows),
                           "outcome_proof": [method for method, _ in matched],
                           "source_board_sha256": board_hash(source),
                           "outcome_board_sha256": [board_hash(boards[index]) for index in (first_match, step)],
                           "proved_background_sha256": board_hash(matched[0][1])})
            counts["proved_events"] += 1
            counts[f"proved_rows/{len(rows)}"] += 1
            t = step
            break
        else:
            rejected = ("exact_outcome_without_next_confirmation" if first_unconfirmed is not None
                        else "no_exact_outcome_within_window")
        if rejected is not None:
            counts[f"inconclusive/{rejected}"] += 1
        t += 1
    return counts, events


def self_check():
    source = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1]])
    flash = source.copy()
    flash[2] = 9
    after = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])
    boards = np.stack((source, source, flash, source, flash, after, after))
    labels = {"shape_id": np.full(len(boards), -1), "evidence": np.zeros(len(boards))}
    counts, events = clear_events(boards, labels, np.empty((0, 4, 2)), np.array([1]), 0)
    assert counts["proved_events"] == 1 and len(events) == 1
    assert events[0]["rows_removed"] == [2] and events[0]["confirmation_frame"] == 6
    assert events[0]["delay_from_lock_frames"] == 4 and events[0]["outcome_proof"] == ["raw", "raw"]
    counts, events = clear_events(boards[:-1], labels, np.empty((0, 4, 2)), np.array([1]), 0)
    assert not events and counts["inconclusive/exact_outcome_without_next_confirmation"] == 1
    with_piece = boards.copy()
    with_piece[-1, :2, :2] = 2
    square = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels["shape_id"][-1] = 1
    labels["evidence"][-1] = 1
    labels["cells"] = np.broadcast_to(square, (len(boards), 4, 2))
    _, events = clear_events(with_piece, labels, np.stack((square, square)), np.array([1, 2]), 0)
    assert events[0]["outcome_proof"] == ["raw", "masked_proved_foreground"]
    labels["evidence"][-1] = 0
    _, events = clear_events(with_piece, labels, np.stack((square, square)), np.array([1, 2]), 0,
                             visible_overlay=True)
    assert events[0]["outcome_proof"] == ["raw", "masked_exact_foreground"]
    clipped = after.copy()
    clipped[0, :2] = 2
    assert exact_foreground_pose(clipped, after, square[None], np.array([2]), 0) == -2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--max-delay", type=int, choices=(MAX_DELAY, 48), default=MAX_DELAY)
    parser.add_argument("--prove-visible-overlay", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    self_check()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = scene_from_weights(checkpoint["model"])
    shapes, tiles = model.object_shapes.numpy(), model.object_tiles.numpy()
    manifest = json.loads((ROOT / "data/manifest.json").read_bytes())
    recorded = {entry["file"]: entry["events"] for entry in manifest["episodes"] if entry["split"] == args.split}
    counts, proof = Counter(), []
    for name, frames, actions, labels in observed_captures(checkpoint["config"], model.codebook.numpy(), args.split):
        boards = frames[:, :, 2:12].numpy()
        results, events = clear_events(boards, {key: values.numpy() for key, values in labels.items()},
                                       shapes, tiles, model.blank_id, max_delay=args.max_delay,
                                       visible_overlay=args.prove_visible_overlay)
        counts.update(results)
        references = recorded[name]
        counts["manifest_screen_lines"] += references["lines_cleared"]
        counts["manifest_screen_events"] += len(references["line_clear_steps"])
        ordinary_full = (np.isin(boards, np.r_[model.blank_id, tiles]).all((1, 2))
                         & (boards != model.blank_id).all(2).any(1))
        for step in references["line_clear_steps"]:
            center = step + 1  # Collector records the transition index, not the new frame index.
            lower, upper = max(0, center - args.max_delay), min(len(boards), center + args.max_delay + 1)
            counts["manifest_events_with_visible_full_board_within_window"] += int(ordinary_full[lower:upper].any())
            counts["manifest_events_with_proved_outcome_within_window"] += int(any(
                lower <= event["first_outcome_frame"] < upper for event in events))
        for event in events:
            begin, end = event["last_consecutive_lock_frame"], event["confirmation_frame"]
            proof.append({"file": name, **event, "actions_until_confirmation": actions[begin:end].tolist()})
    report = {"split": args.split, "test_used": False, "validation_used": args.split == "validation", "fit_used": False,
              "model_forward_used": False, "max_delay_frames": args.max_delay, "synthetic_self_check": True,
              "visible_overlay_proof": args.prove_visible_overlay,
              "checkpoint_sha256": sha(args.checkpoint), "script_sha256": sha(Path(__file__)),
              "manifest_sha256": checkpoint["config"]["manifest_sha256"],
              "annotations_sha256": checkpoint["config"]["annotations_sha256"],
              "manifest_screen_event_window_frames": args.max_delay,
              "counts": counts, "events": proof,
              "limits": ["Requires an ordinary full-row board, a visible change confined to the removed rows, and two consecutive exact outcomes.",
                         "An ordinary full board can be absent from sampled images before an actual ROM clear. Absence of proved events does not imply zero ROM clears.",
                         "A different full board during the proof window makes attribution inconclusive. No partial or best-fit match is accepted.",
                         "Manifest screen counters measure coverage only and do not provide geometric proof or one-to-one event matches."]}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
