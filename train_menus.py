"""Learn title and game-over button responses from native ROM interventions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from pyboy import PyBoy

from collect import counters
from model import ACTIONS
from run import ROOT, step


def native(game):
    return np.asarray(game.screen.image.convert("L"), dtype=np.uint8) // 85


def intervene(rom, boot, terminal, button):
    game = PyBoy(str(rom), window="null", sound_emulated=False,
                 color_palette=(0xFFFFFF, 0xAAAAAA, 0x555555, 0))
    try:
        game.set_emulation_speed(0)
        game.tick(boot, True)
        title = native(game)
        if terminal:
            step(game, "a")
            for _ in range(12):
                step(game, "noop")
                if counters(game)["level"] == 1:
                    break
            else:
                raise ValueError("unable to start the reference game")
            missing = 0
            for _ in range(900):
                step(game, "down")
                missing = missing + 1 if counters(game)["level"] is None else 0
                if missing == 8:
                    break
            else:
                raise ValueError("reference game did not end")
            for _ in range(24):
                step(game, "noop")
        before = native(game)
        step(game, button)
        after = [native(game)]
        play = counters(game)["level"] is not None
        for _ in range(7):
            step(game, "noop")
            after.append(native(game))
            play |= counters(game)["level"] is not None
        if play:
            target = 1
        elif np.array_equal(after[-1][:118], title[:118]):
            target = 2 if terminal else 0
        else:
            target = 0
        return title, before, np.stack(after), target
    finally:
        game.stop(save=False)


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
    votes = np.zeros((2, len(ACTIONS), 3), dtype=np.int64)
    panels = Counter()
    tests = []
    records = []
    images = []
    for split, boots in (("train", (2201, 2257, 2323, 2377)),
                         ("validation", (4507, 4579, 4621))):
        for boot in boots:
            for phase in (0, 1):
                for action, button in enumerate(ACTIONS):
                    title, before, after, target = intervene(rom, boot, bool(phase), button)
                    records.append({"split": split, "boot": boot, "phase": phase, "button": button, "target": target})
                    images.append(np.concatenate((before[None], after)))
                    if split == "train":
                        panels[title.tobytes()] += 1
                        votes[phase, action, target] += 1
                    else:
                        tests.append((phase, action, target))
    transitions = votes.argmax(2).astype(np.uint8)
    title = np.frombuffer(panels.most_common(1)[0][0], dtype=np.uint8).reshape(144, 160)
    if not (transitions[0] == 1).any() or not (transitions[1] != 0).any():
        raise ValueError("missing title entry or terminal exit in reference captures")
    weights = args.output / "menus.npz"
    np.savez_compressed(weights, title=title, transitions=transitions)
    np.savez_compressed(args.output / "interventions.npz", frames=np.stack(images))
    report = {"rom_sha256": digest, "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "train_examples": int(votes.sum()), "validation_examples": len(tests),
              "validation_errors": sum(int(transitions[p, a] != y) for p, a, y in tests),
              "votes": votes.tolist(), "records": records}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"validation_errors": report["validation_errors"],
                      "actions": ACTIONS, "transitions": transitions.tolist()}))


if __name__ == "__main__":
    main()
