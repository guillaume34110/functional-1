"""Recover exact HUD glyphs from native-resolution, frame-verified ROM replays."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_controls_reference import replay_prefix
from collect import COUNTER_FIELDS, DIGITS
from run import ROOT, read_episode, step
from train_spawn_preview import sha


GLYPHS = np.concatenate((DIGITS, np.full((1, 8, 8), 3, dtype=np.uint8)))
SLOTS = tuple((y, x + 8 * i) for _, y, x, length in COUNTER_FIELDS for i in range(length))


def native_digits(image):
    tiles = np.stack([image[y:y+8, x:x+8] for y, x in SLOTS])
    match = (tiles[:, None] == GLYPHS).all((2, 3))
    return np.where(match.any(1), match.argmax(1), -1).astype(np.int8)


def replay_digits(game, frames, actions, metadata):
    observed = []

    def record(game, action):
        frame = step(game, action)
        native = np.asarray(game.screen.image.convert("L"), dtype=np.uint8) // 85
        observed.append(native_digits(native))
        return frame

    proof = replay_prefix(game, frames, actions, metadata, len(actions), step_fn=record)
    assert proof["verified_frames"] == len(frames)
    return np.stack(observed[-len(frames):])


def read_digit_captures(path, manifest_path, split):
    manifest = json.loads(manifest_path.read_bytes())
    rows = [{"file": row["file"], "sha256": row["sha256"], "frames": row["steps"] + 1}
            for row in manifest["episodes"] if row["split"] == split]
    with np.load(path, allow_pickle=False) as archive:
        digits = archive["digits"]
        metadata = json.loads(str(archive["metadata"]))
    if (split not in ("train", "validation") or metadata["split"] != split or metadata["test_used"]
            or metadata["manifest_sha256"] != sha(manifest_path) or metadata["entries"] != rows
            or metadata["rom_sha256"] != manifest["context"]["rom"]["sha256"] or not rows
            or len({row["file"] for row in rows}) != len(rows)
            or digits.dtype != np.int8 or digits.shape != (sum(row["frames"] for row in rows), len(SLOTS))
            or np.any((digits < -1) | (digits >= len(GLYPHS)))):
        raise ValueError("invalid native counter capture lineage or glyphs")
    offsets = np.cumsum([row["frames"] for row in rows])[:-1]
    return dict(zip((row["file"] for row in rows), np.split(digits, offsets), strict=True))


def main():
    from pyboy import PyBoy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = ROOT / "data/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    context = json.loads((ROOT / "context.json").read_bytes())
    rom = ROOT / context["rom"]["file"]
    if manifest["status"] != "complete" or manifest["context"] != context or sha(rom) != context["rom"]["sha256"]:
        raise ValueError("incompatible capture manifest or ROM")
    captures, entries = [], []
    for row in manifest["episodes"]:
        if row["split"] != args.split:
            continue
        path = (manifest_path.parent / row["file"]).resolve()
        if not path.is_relative_to(manifest_path.parent) or sha(path) != row["sha256"]:
            raise ValueError("capture path or digest mismatch")
        frames, actions, metadata = read_episode(path)
        if metadata["source"] != "capture" or metadata["split"] != args.split or len(actions) != row["steps"]:
            raise ValueError("capture length or split mismatch")
        game = PyBoy(str(rom), window="null", sound_emulated=False,
                     color_palette=(0xFFFFFF, 0xAAAAAA, 0x555555, 0x000000))
        try:
            captures.append(replay_digits(game, frames, actions, metadata))
        finally:
            game.stop(save=False)
        entries.append({"file": row["file"], "sha256": row["sha256"], "frames": len(frames)})
    if not captures:
        raise ValueError("no captures for split")
    report = {"split": args.split, "manifest_sha256": sha(manifest_path), "test_used": False,
              "rom_sha256": sha(rom), "entries": entries}
    with args.output.open("xb") as out:
        np.savez_compressed(out, digits=np.concatenate(captures), metadata=json.dumps(report))


if __name__ == "__main__":
    main()
