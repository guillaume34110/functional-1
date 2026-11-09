"""Offline pixel-driven benchmark for the reference ROM and the neural game."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyboy import PyBoy
import torch

from collect import ScreenController, counters
from object_annotations import classify_board
from play import NeuralSession, RELEASE
from run import ROOT, step
from train_counters import values


def image(frame):
    return Image.fromarray(frame.astype(np.uint8) * 85)


def native(game):
    return np.asarray(game.screen.image.convert("L"), dtype=np.uint8) // 85


def record(seed, actions, steps):
    context = json.loads((ROOT / "context.json").read_text())
    rom = ROOT / context["rom"]["file"]
    if hashlib.sha256(rom.read_bytes()).hexdigest() != context["rom"]["sha256"]:
        raise ValueError("reference ROM checksum mismatch")
    game = PyBoy(str(rom), window="null", sound_emulated=False,
                 color_palette=(0xFFFFFF, 0xAAAAAA, 0x555555, 0))
    try:
        game.set_emulation_speed(0)
        game.tick(368, True)
        menu = native(game)
        frame = step(game, "a")
        for _ in range(10):
            if np.array_equal(frame, seed):
                break
            frame = step(game, "noop")
        else:
            raise ValueError("reference did not reach the verified Functional 1 seed")
        frames = [native(game)]
        controller = ScreenController(86, "heuristic")
        buttons = []
        for t in range(steps):
            action = actions[t] if t < len(actions) else controller.choose(frame)
            frame = step(game, action)
            buttons.append(action)
            frames.append(native(game))
        return menu, np.stack(frames), buttons, counters(game)
    finally:
        game.stop(save=False)


def replay(buttons, adaptive=False, checkpoint=RELEASE / "model.pt"):
    game = NeuralSession(checkpoint, RELEASE / "first-piece.npz")
    controller = ScreenController(86, "heuristic")
    frames = [game.frame[0, 0].numpy().copy()]
    choices = []
    for action in buttons:
        if adaptive:
            action = controller.choose(frames[-1])
        game.step(action)
        choices.append(action)
        frames.append(game.frame[0, 0].numpy().copy())
    pictures = np.stack(frames)
    with torch.inference_mode():
        hud = values(game.model.counter.read(torch.from_numpy(pictures[:, None]))).numpy()
    return pictures, choices, int(hud[:, 1].max())


def compare(ref, predicted, buttons):
    small = ref[:, 1::2, 1::2]
    enlarged = np.repeat(np.repeat(small, 2, axis=1), 2, axis=2)
    board_equal = (small[:, :, 8:48] == predicted[:, :, 8:48]).all(axis=(1, 2))

    def static_runs(frames):
        board = frames[:, :, 8:48]
        changes = np.flatnonzero((board[1:] != board[:-1]).any(axis=(1, 2))) + 1
        return {"first_change_step": int(changes[0]) if len(changes) else None,
                "longest_static_frames": int(np.diff(np.r_[0, changes, len(board)]).max())}

    return {
        "steps": len(buttons),
        "first_board_divergence": next((i for i, same in enumerate(board_equal) if not same), None),
        "board_exact_frames": int(board_equal.sum()),
        "native_pixels_lost_at_half_resolution": float((ref != enlarged).mean()),
        "native_hud_pixels_lost_at_half_resolution": float((ref[:, :, 96:] != enlarged[:, :, 96:]).mean()),
        "reference_board_activity": static_runs(small),
        "neural_board_activity": static_runs(predicted),
        "predicted_board_invalid_pattern_frames": sum(bool((classify_board(f) < -1).any()) for f in predicted),
    }


def sheet(path, menu, reference, predicted, buttons):
    times = sorted({0, 2, 4, 8, 32, 64, 128, len(buttons)})
    board = Image.new("RGB", (840, 340 * (1 + len(times))), "#111121")
    draw = ImageDraw.Draw(board)
    draw.text((18, 8), "Reference title screen at native resolution", fill="white")
    board.paste(image(menu).resize((320, 288), Image.Resampling.NEAREST), (18, 28))
    for row, t in enumerate(times, 1):
        y = row * 340
        draw.text((18, y + 8), f"ROM native 160x144 | step {t}", fill="white")
        draw.text((426, y + 8), f"Functional 1 80x72 | {buttons[t-1] if t else 'seed'}", fill="white")
        board.paste(image(reference[t]).resize((320, 288), Image.Resampling.NEAREST), (18, y + 28))
        board.paste(image(predicted[t]).resize((320, 288), Image.Resampling.NEAREST), (426, y + 28))
    board.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=RELEASE / "model.pt")
    args = parser.parse_args()
    args.output.mkdir()
    torch.set_num_threads(1)
    initial = NeuralSession(RELEASE / "model.pt", RELEASE / "first-piece.npz").frame[0, 0].numpy()
    cases = {
        "gravity": ("noop",) * 256,
        "walls": ("left",) * 24 + ("a", "noop", "b", "noop") * 4
                 + ("right",) * 24 + ("b", "noop", "a", "noop") * 4,
        "game": (),
    }
    result = {}
    for name, prefix in cases.items():
        menu, ref, commands, hud = record(initial, prefix, 1024 if name == "game" else 256)
        frames, _, lines = replay(commands, checkpoint=args.checkpoint)
        result[name] = {**compare(ref, frames, commands), "reference_final_hud": hud, "neural_max_lines": lines}
        np.savez_compressed(args.output / f"{name}.npz", reference=ref, predicted=frames, actions=commands)
        sheet(args.output / f"{name}.png", menu, ref, frames, commands)
        if name == "game":
            adaptive, adaptive_buttons, adaptive_lines = replay(commands, adaptive=True, checkpoint=args.checkpoint)
            np.savez_compressed(args.output / "adaptive.npz", predicted=adaptive, actions=adaptive_buttons)
            result[name]["adaptive_neural_max_lines"] = adaptive_lines
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
