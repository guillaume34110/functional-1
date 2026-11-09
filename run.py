"""Capture ROM transitions or generate screens using only a neural model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model import ACTIONS, NeuralTetris

ROOT = Path(__file__).resolve().parent


def observe(game):
    from PIL import Image
    image = game.screen.image.convert("L").resize((80, 72), Image.Resampling.NEAREST)
    return (np.array(image, dtype=np.uint8) // 85).astype(np.uint8)


def step(game, action):
    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action}")
    if action != "noop":
        game.button_press(action)
    try:
        if not game.tick(4, True):
            raise RuntimeError("emulator stopped")
    finally:
        if action != "noop":
            game.button_release(action)
    return observe(game)


def capture(actions, context):
    from pyboy import PyBoy
    rom = ROOT / context["rom"]["file"]
    if hashlib.sha256(rom.read_bytes()).hexdigest() != context["rom"]["sha256"]:
        raise ValueError("ROM hash mismatch")
    game = PyBoy(str(rom), window="null", sound_emulated=False,
                 color_palette=(0xFFFFFF, 0xAAAAAA, 0x555555, 0x000000))
    try:
        game.set_emulation_speed(0)
        game.tick(1, True)
        frames = [observe(game)]
        for action in actions:
            frames.append(step(game, action))
        return np.stack(frames)
    finally:
        game.stop(save=False)


def read_episode(path):
    with np.load(path, allow_pickle=False) as episode:
        frames, actions = episode["frames"], episode["actions"]
        metadata = json.loads(str(episode["metadata"]))
    if frames.dtype != np.uint8 or frames.shape != (len(actions) + 1, 72, 80) or np.any(frames > 3):
        raise ValueError("invalid episode frames")
    if actions.dtype != np.uint8 or actions.ndim != 1 or np.any(actions >= len(ACTIONS)):
        raise ValueError("invalid episode actions")
    if metadata["context"] != json.loads((ROOT / "context.json").read_text()):
        raise ValueError("incompatible episode context")
    return frames, actions, metadata


def generate(model, seed_frames, seed_actions, actions):
    """Warm memory on a real prefix, then feed predictions back without the ROM."""
    state = None
    with torch.inference_mode():
        for frame, action in zip(seed_frames[:-1], seed_actions):
            _, state = model(torch.from_numpy(frame.copy())[None, None], torch.tensor([int(action)]), state)
        frame = torch.from_numpy(seed_frames[-1].copy())[None, None]
        frames = [seed_frames[-1].copy()]
        for action in actions:
            logits, state = model(frame, torch.tensor([ACTIONS.index(action)]), state)
            frame = logits.argmax(dim=1, keepdim=True).to(torch.uint8)
            frames.append(frame[0, 0].numpy().copy())
    return np.stack(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "generate"))
    parser.add_argument("--actions", type=Path, required=True, help="JSON buttons supplied by the user")
    parser.add_argument("--checkpoint", type=Path, help="trained model state_dict for generate")
    parser.add_argument("--seed", type=Path, help="real episode prefix NPZ for generate")
    parser.add_argument("--output", type=Path, required=True, help="new .npz; never overwritten")
    args = parser.parse_args()
    context = json.loads((ROOT / "context.json").read_text())
    if tuple(context["actions"]) != ACTIONS or context["frames_per_action"] != 4:
        raise ValueError("context does not match runtime")
    actions = json.loads(args.actions.read_text())
    if not isinstance(actions, list) or not 1 <= len(actions) <= 9000 or any(a not in ACTIONS for a in actions):
        raise ValueError("require a JSON list of 1..9000 known button names")
    if args.output.suffix != ".npz":
        raise ValueError("output must be .npz")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "capture" and (args.checkpoint or args.seed):
        parser.error("capture does not use weights or seed")
    torch.set_num_threads(1)
    if args.mode == "generate":
        if not args.checkpoint or not args.seed:
            parser.error("generate requires --checkpoint and --seed")
        seed_frames, seed_actions, seed_metadata = read_episode(args.seed)
        if seed_metadata.get("source") != "capture":
            raise ValueError("seed must be a reference capture, not generated frames")
        model = NeuralTetris().eval()
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        frames = generate(model, seed_frames, seed_actions, actions)
    else:
        frames = capture(actions, context)
    metadata = {"context": context, "source": args.mode, "terminal_annotated": False,
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
                "seed_sha256": hashlib.sha256(args.seed.read_bytes()).hexdigest() if args.seed else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        np.savez_compressed(output, frames=frames, actions=np.array([ACTIONS.index(a) for a in actions], dtype=np.uint8),
                            metadata=json.dumps(metadata))
    print(json.dumps({"steps": len(actions), "output": str(args.output), "source": args.mode}))


if __name__ == "__main__":
    main()
