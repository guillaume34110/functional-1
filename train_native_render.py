"""Collect disjoint boot trajectories and fit a 160×144 tile readout."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from pyboy import PyBoy

from model import ACTIONS
from native_render import NativeRenderer, patches, tile_keys
from run import ROOT, step


SCHEDULES = {
    "gravity": ("a",) + ("noop",) * 325,
    "left_wall": ("a",) + ("noop",) * 6 + ("left",) * 18 + ("a", "noop", "b", "noop") * 12,
    "right_wall": ("a",) + ("noop",) * 6 + ("right",) * 18 + ("b", "noop", "a", "noop") * 12,
    "pause": ("a",) + ("noop",) * 12 + ("start",) + ("noop",) * 12 + ("start", "down") + ("down",) * 48,
    "menu_probe": ("right", "noop", "a") + ("noop",) * 12
                  + ("up", "noop", "down", "noop", "right", "noop", "left", "noop") * 2
                  + ("b", "noop", "start") + ("noop",) * 12,
}


def capture(rom, boot, commands):
    game = PyBoy(str(rom), window="null", sound_emulated=False,
                 color_palette=(0xFFFFFF, 0xAAAAAA, 0x555555, 0))
    try:
        game.set_emulation_speed(0)
        game.tick(boot, True)
        frames = [np.asarray(game.screen.image.convert("L"), dtype=np.uint8) // 85]
        for action in commands:
            step(game, action)
            frames.append(np.asarray(game.screen.image.convert("L"), dtype=np.uint8) // 85)
        return np.stack(frames)
    finally:
        game.stop(save=False)


def fit_patch_votes(votes, native):
    observed = patches(native, 8)
    keys = tile_keys(observed[..., 1::2, 1::2]).reshape(-1)
    raw = np.ascontiguousarray(observed).reshape(-1, 64)
    patterns, first, counts = np.unique(raw.view("V64"), return_index=True, return_counts=True)
    for i, count in zip(first, counts, strict=True):
        votes[int(keys[i])][raw[i].tobytes()] += int(count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    context = json.loads((ROOT / "context.json").read_text())
    rom = ROOT / context["rom"]["file"]
    digest = hashlib.sha256(rom.read_bytes()).hexdigest()
    if digest != context["rom"]["sha256"]:
        raise ValueError("reference ROM checksum mismatch")
    votes = defaultdict(Counter)
    records = []
    for split, boots in (("train", range(1201, 1201 + 18 * 47, 47)),
                         ("validation", range(4001, 4001 + 6 * 59, 59))):
        for boot in boots:
            for name, commands in SCHEDULES.items():
                frames = capture(rom, boot, commands)
                path = args.output / f"{split}-{boot}-{name}.npz"
                np.savez_compressed(path, frames=frames, actions=np.array([ACTIONS.index(c) for c in commands], dtype=np.uint8))
                records.append({"file": path.name, "split": split, "boot": boot, "scenario": name,
                                "frames": len(frames), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                if split == "train":
                    fit_patch_votes(votes, frames)
    keys = np.array(sorted(votes), dtype=np.uint32)
    tiles = np.stack([np.frombuffer(votes[int(key)].most_common(1)[0][0], dtype=np.uint8).reshape(8, 8) for key in keys])
    weights = args.output / "native-render.npz"
    np.savez_compressed(weights, keys=keys, tiles=tiles)
    renderer = NativeRenderer(weights)
    totals = Counter()
    for record in records:
        if record["split"] != "validation":
            continue
        with np.load(args.output / record["file"], allow_pickle=False) as data:
            frames = data["frames"]
        for target in frames:
            low = target[1::2, 1::2]
            prediction = renderer(low)
            baseline = np.repeat(np.repeat(low, 2, axis=0), 2, axis=1)
            assert np.array_equal(prediction[1::2, 1::2], low)
            totals.update(frames=1, pixels=target.size, hud_pixels=target[:, 96:].size,
                          baseline=int((target != baseline).sum()), learned=int((target != prediction).sum()),
                          baseline_hud=int((target[:, 96:] != baseline[:, 96:]).sum()),
                          learned_hud=int((target[:, 96:] != prediction[:, 96:]).sum()))
    report = {"rom_sha256": digest, "fit_split": "train", "validation": dict(totals),
              "renderer_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "records": records}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"episodes": len(records), "tiles": len(keys), **totals}))


if __name__ == "__main__":
    main()
