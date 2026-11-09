"""Replay a verified top-out capture from the first visible piece."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_controls import supported
from audit_long import recorded_path, screen_progress, summarize
from collect import ScreenController, counters, occupancy
from model import ACTIONS
from object_annotations import candidates, classify_board
from run import ROOT, read_episode, step
from token_model import encode_frames, load_tile_model
from train_spawn_preview import sha


def replay_reference(entry, tail_steps=0):
    from pyboy import PyBoy

    capture = ROOT / "data" / entry["file"]
    if sha(capture) != entry["sha256"]:
        raise ValueError("top-out capture digest mismatch")
    saved, commands, metadata = read_episode(capture)
    down = ACTIONS.index("down")
    if not metadata["terminal_annotated"] or (metadata["profile"] == "topout" and not np.all(commands == down)):
        raise ValueError("expected a complete terminal witness")
    warm = metadata["warmup_steps"]
    controller = ScreenController(metadata["seed"], metadata["profile"])
    rom = ROOT / metadata["context"]["rom"]["file"]
    if sha(rom) != metadata["context"]["rom"]["sha256"]:
        raise ValueError("top-out reference ROM digest mismatch")
    game = PyBoy(str(rom), window="null", sound_emulated=False,
                 color_palette=(0xFFFFFF, 0xAAAAAA, 0x555555, 0x000000))
    try:
        game.set_emulation_speed(0)
        game.tick(metadata["boot_frames"], True)
        step(game, "a")
        for _ in range(10):
            frame = step(game, "noop")
            if counters(game)["level"] == 1:
                break
        else:
            raise ValueError("reference failed to start")
        frames = [frame]
        valid_level = [True]
        warm_commands = []
        for _ in range(warm):
            action = controller.choose(frames[-1])
            warm_commands.append(ACTIONS.index(action))
            frames.append(step(game, action))
            valid_level.append(counters(game)["level"] is not None)
        for command in commands:
            frames.append(step(game, ACTIONS[int(command)]))
            valid_level.append(counters(game)["level"] is not None)
        if not np.array_equal(np.stack(frames[warm:]), saved) or any(valid_level[-8:]):
            raise ValueError("fresh reference differs from the preserved terminal witness")
        tail = [step(game, "noop") for _ in range(tail_steps)]
    finally:
        game.stop(save=False)
    actions = np.concatenate((np.asarray(warm_commands, dtype=np.uint8), commands))
    return np.stack(frames), actions, valid_level, tail


def start_piece(frames, warm):
    for index, frame in enumerate(frames[:warm+1]):
        grid = classify_board(frame)
        shapes = candidates(grid)
        if np.all(grid >= -1) and np.count_nonzero(grid >= 0) == 4 and len(shapes) == 1:
            return index, np.asarray(shapes[0][1])
    raise ValueError("reference never shows a unique first piece on an empty well")


@torch.inference_mode()
def audit(model, reference, actions, valid_level, warm, continue_to=0):
    start, cells = start_piece(reference, warm)
    tokens, missing = encode_frames(reference[:start+1], model.codebook.numpy())
    if missing:
        raise ValueError("first playable prefix contains tiles outside the frozen vocabulary")
    tokens = torch.from_numpy(tokens).long()
    state = None
    for index in range(start):
        state = model.advance_observed(tokens[index:index+1], torch.tensor([int(actions[index])]), state)
    previous = ACTIONS[int(actions[start - 1])] if start else None
    commands = actions[start:]
    if continue_to > len(commands):
        commands = np.pad(commands, (0, continue_to - len(commands)), constant_values=ACTIONS.index("down"))
    first_terminal = None

    def observe(frame, old, _, event):
        nonlocal first_terminal
        if event.get("terminal_phase") and first_terminal is None:
            background = frame[0, :, 2:12].numpy().copy()
            if old is not None:
                y, x = np.asarray(old).T
                background[y, x] = model.blank_id
            first_terminal = {"foreground": None if old is None else old.tolist(),
                              "supported": None if old is None else supported(background, old, model.blank_id),
                              "pile_top_four_occupied": (background[:4] != model.blank_id).sum(1).tolist()}
        return True

    events, images = recorded_path(model, tokens[start:start+1], state, cells, commands, previous, observer=observe)
    pictures = np.stack(images)
    game_over = next(i for i in range(start, len(valid_level)-7) if not any(valid_level[i:i+8]))
    hud_signature = {frame[:, 48:].tobytes() for frame in reference[-8:]}
    level_signature = reference[start, 60:64, 68:76]
    observed = summarize(events)
    tail = occupancy(pictures[-1])
    last_cells = events[-1]["cells"]
    if last_cells is not None:
        y, x = np.asarray(last_cells).T
        tail[y, x] = False
    return {"start_frame": start, "reference_terminal_frame": game_over,
            "commands": len(commands), "reference_command_count": len(actions) - start, "autonomous": True,
            "reference_board_exact_steps": int((pictures[1:len(actions)-start+1, :, 8:48] ==
                                                 reference[start+1:, :, 8:48]).all((1, 2)).sum()),
            "reference_terminal_hud_matches": sum(frame[:, 48:].tobytes() in hud_signature for frame in pictures),
            "reference_terminal_full_frame_matches": sum(
                any(np.array_equal(frame, target) for target in reference[-8:]) for frame in pictures),
            "first_exact_terminal_frame": next((i + start for i, frame in enumerate(pictures)
                                                if np.array_equal(frame, reference[game_over])), None),
            "first_exact_terminal_menu_frame": next((i + start for i, frame in enumerate(pictures)
                                                     if np.array_equal(frame, reference[game_over+2])), None),
            "settled_terminal_menu_exact": bool(np.array_equal(pictures[-1], reference[-1])),
            "reference_settled_menu_sha256": hashlib.sha256(reference[-1].tobytes()).hexdigest(),
            "terminal_phase_step": next((e["step"] for e in events if e.get("terminal_phase")), None),
            "first_terminal_context": first_terminal,
            "original_level_visible_last_eight": int((pictures[-8:, 60:64, 68:76] == level_signature).all((1, 2)).sum()),
            "neural_visible_deposits": observed["counts"].get("deposit", 0),
            "first_unknown": observed["first_unknown"], "first_violation": observed["first_violation"],
            "screen_progress": screen_progress(images),
            "last_visible_object": last_cells, "last_background_occupied_by_row": tail.sum(1).tolist(),
            "visible_deposit_steps": [index for index, event in enumerate(events) if event["status"] == "deposit"],
            "maximum_occupied_tiles": max(int(occupancy(frame).sum()) for frame in pictures)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed-output", type=Path, help="save the verified first playable prefix")
    parser.add_argument("--continue-to", type=int, default=0, help="extend the neural-only held-down trace to this many commands")
    args = parser.parse_args()
    for path in (args.output, args.seed_output):
        if path is not None and path.exists():
            raise FileExistsError(path)
    if not 0 <= args.continue_to <= 2048:
        parser.error("--continue-to must be between 0 and 2048")
    torch.set_num_threads(1)
    manifest_path = ROOT / "data/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    entry, = [row for row in manifest["episodes"] if row["split"] == "validation" and row["profile"] == "topout"]
    reference, actions, valid_level, _ = replay_reference(entry)
    warm = len(actions) - entry["steps"]
    model = load_tile_model(args.checkpoint)
    report = {"checkpoint_sha256": sha(args.checkpoint), "manifest_sha256": sha(manifest_path),
              "source": entry["file"], "source_sha256": entry["sha256"], "test_used": False,
              "reference_actions": "down", **audit(model, reference, actions, valid_level, warm, args.continue_to)}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    if args.seed_output:
        start = report["start_frame"]
        metadata = {"context": manifest["context"], "source": "capture", "source_sha256": entry["sha256"],
                    "reference_start_frame": start}
        with args.seed_output.open("xb") as handle:
            np.savez_compressed(handle, frames=reference[:start+1], actions=actions[:start],
                                metadata=json.dumps(metadata, allow_nan=False))


if __name__ == "__main__":
    main()
