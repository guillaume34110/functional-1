"""Record complete Functional 2 rollouts with fixed, reproducible control profiles."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from audit_spawn import preview_texture, spawn_evidence
from collect import ScreenController
from functional_two import FunctionalTwoSession
from object_annotations import candidates, classify_board
from train_counters import values


CASES = (
    ("topout", 86), ("heuristic", 86), ("heuristic", 87), ("exploration", 88),
    ("sustained", 91), ("sustained", 92), ("sustained", 93), ("sustained", 94),
)


def pose_pairs(before, after):
    def foregrounds(board):
        for style, cells in candidates(board):
            cells = np.asarray(cells)
            pile = board.copy()
            pile[tuple(cells.T)] = -1
            yield style, cells, pile

    return [(old, new) for style, old, pile in foregrounds(before)
            for new_style, new, new_pile in foregrounds(after)
            if style == new_style and np.array_equal(pile, new_pile)]


def transition_metrics(frames, actions):
    counts = Counter()
    anomalies = []
    previews = [preview_texture(frame) for frame in frames]
    boards = [classify_board(frame) for frame in frames]
    for i, (before, after, action) in enumerate(zip(boards, boards[1:], actions)):
        preview, next_preview = previews[i:i + 2]
        changes = before != after
        if preview is not None and next_preview is not None and preview != next_preview:
            counts["preview_changes"] += 1
            if not changes.any():
                counts["preview_changes_on_static_board"] += 1
        if (before < -1).any() or (after < -1).any() or not changes.any():
            continue
        if changes.sum() == 4 and (after[changes] == -1).all() and not (before >= 0).all(1).any():
            if any(changes[tuple(np.asarray(cells).T)].all() for _, cells in candidates(before)):
                counts["piece_disappeared"] += 1
                anomalies.append({"step": i + 1, "action": str(action), "piece_disappeared": True})
        if changes.sum() == 4 and ((before == -1) & (after >= 0) & changes).sum() == 4:
            proof = next((event for _, cells in candidates(before)
                          if (event := spawn_evidence(frames[i], frames[i + 1], np.asarray(cells)))), None)
            if proof is not None:
                counts["proved_spawns"] += 1
                counts["preview_matches_spawn"] += proof["texture"] == preview
        if (before >= 0).sum() != (after >= 0).sum():
            continue
        pairs = pose_pairs(before, after)
        if len(pairs) != 1:
            continue
        old, new = pairs[0]
        old_origin, new_origin = old.min(0), new.min(0)
        rotates = not np.array_equal(old - old_origin, new - new_origin)
        counts["proved_rotations" if rotates else "proved_translations"] += 1
        if preview is not None and next_preview is not None and preview != next_preview:
            counts["preview_changes_on_motion"] += 1
        if not rotates and action not in ("a", "b", "up", "start"):
            dy, dx = new_origin - old_origin
            previous = actions[i - 1] if i else None
            max_drop = 2 if action == "down" or previous == "down" else 1
            min_shift = -int("left" in (action, previous))
            max_shift = int("right" in (action, previous))
            if previous not in ("a", "b") and (dy < 0 or dy > max_drop or not min_shift <= dx <= max_shift):
                counts["translations_outside_reference"] += 1
                anomalies.append({"step": i + 1, "action": str(action), "shift": [int(dy), int(dx)]})
        if rotates and action not in ("a", "b") and (i == 0 or actions[i - 1] not in ("a", "b")):
            counts["rotation_without_recent_button"] += 1
            anomalies.append({"step": i + 1, "action": str(action), "rotation": True})
    return {**counts, "preview_textures": sorted({p for p in previews if p is not None}),
            "motion_witnesses": anomalies}


def play(game, profile, seed, limit, *, finish=0):
    game.reset()
    game.step("start")
    controller = ScreenController(seed, profile)
    frames = [game.frame[0, 0].numpy().copy()]
    actions = []
    for index in range(limit + finish):
        # Exercise the terminal screen even when the regular policy is still playing.
        if index >= limit or (profile == "sustained" and index < 160):
            action = "down"
        else:
            action = controller.choose(frames[-1])
        result = game.step(action)
        frames.append(game.frame[0, 0].numpy().copy())
        actions.append(action)
        if result["menu"]:
            break

    frames = np.stack(frames)
    board = frames[:, :, 8:48]
    changes = np.flatnonzero((board[1:] != board[:-1]).any(axis=(1, 2))) + 1
    starts = np.r_[0, changes]
    runs = np.diff(np.r_[starts, len(board)])
    longest = int(runs.argmax())
    with torch.inference_mode():
        displayed = values(game.model.counter.read(torch.from_numpy(frames[:, None]))).numpy()
    result = {
        "steps": len(actions),
        "finishing_steps": max(0, len(actions) - limit),
        "returned_to_title": result["menu"],
        "longest_static_board": int(runs[longest]),
        "static_start": int(starts[longest]),
        "displayed_lines": int(displayed[:, 1].max()),
        "transitions": transition_metrics(frames, actions),
    }
    return result, frames, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--release", type=Path, default=Path("releases/functional-2"))
    cases = {f"{profile}-{seed}": (profile, seed) for profile, seed in CASES}
    parser.add_argument("--case", action="append", choices=cases, help="replay only the selected cases")
    args = parser.parse_args()
    torch.set_num_threads(1)
    game = FunctionalTwoSession(args.release)
    args.output.mkdir(parents=True)
    report = {}
    for name in args.case or cases:
        profile, seed = cases[name]
        metrics, frames, actions = play(game, profile, seed, 1600, finish=900)
        np.savez_compressed(args.output / f"{name}.npz", frames=frames, actions=actions)
        report[name] = metrics
        print(json.dumps({name: metrics}), flush=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
