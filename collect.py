"""Collect independent, screen-only Adjustris trajectories before windowing."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from model import ACTIONS
from run import ROOT, observe, step

# CC0 Adjustris v1.1 gfx/numbers.asm; used only to annotate visible counters.
NUMBER_BYTES = bytes.fromhex(
    "00007f3e5d637f417f497f415d637f3e00003e1c6a365e625e627e222a363e1c"
    "00007f3e5d637f395f627f4d5d637f3e00007e3c5b667d3b3f223f395d637f3e"
    "00007f365d6b7f495f617f391d0b0f0600007f3e5d637f4e7f217f595d637f3e"
    "000078305c687f4e7d437f495d637f3e00007f3e5d637f417f391f090d0b0f06"
    "00007f3e5d637f497f227f495d637f3e00007f3e5d637f495f617f391d0b0f06")


def digit_templates():
    data = np.frombuffer(NUMBER_BYTES, dtype=np.uint8).reshape(10, 8, 2)
    return np.stack([3 - (((data[:, :, 0] >> (7 - x)) & 1) +
                           2 * ((data[:, :, 1] >> (7 - x)) & 1)) for x in range(8)], axis=2)


DIGITS = digit_templates()
COUNTER_FIELDS = (("score", 72, 112, 5), ("lines", 96, 112, 5), ("level", 120, 136, 2))


def read_number(native, y, x, length):
    value, started = 0, False
    for i in range(length):
        glyph = native[y:y + 8, x + i * 8:x + (i + 1) * 8]
        if glyph.shape != (8, 8):
            return None
        if np.all(glyph == 3) and not started:
            continue
        matches = np.flatnonzero(np.all(DIGITS == glyph, axis=(1, 2)))
        if len(matches) != 1:
            return None
        value, started = value * 10 + int(matches[0]), True
    return value if started else None


def counters(game):
    native = np.asarray(game.screen.image.convert("L"), dtype=np.uint8) // 85
    return {name: read_number(native, y, x, length) for name, y, x, length in COUNTER_FIELDS}


def occupancy(frame):
    if frame.shape != (72, 80):
        raise ValueError("expected indexed 72x80 screen")
    return np.any(frame[:, 8:48].reshape(18, 4, 10, 4).transpose(0, 2, 1, 3) < 3, axis=(2, 3))


def components(board):
    remaining = set(map(tuple, np.argwhere(board)))
    result = []
    while remaining:
        stack = [remaining.pop()]
        found = set(stack)
        while stack:
            y, x = stack.pop()
            for cell in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if cell in remaining:
                    remaining.remove(cell)
                    found.add(cell)
                    stack.append(cell)
        result.append(np.array(sorted(found), dtype=np.int64))
    return result


def normalize(shape):
    shape = np.asarray(shape, dtype=np.int64)
    return np.array(sorted(map(tuple, shape - shape.min(axis=0))), dtype=np.int64)


def best_placement(board, shape, rng):
    choices = []
    current = normalize(shape)
    seen = set()
    for rotation in range(4):
        signature = current.tobytes()
        if signature not in seen:
            seen.add(signature)
            width = int(current[:, 1].max()) + 1
            for x in range(11 - width):
                cells = current + [0, x]
                if board[cells[:, 0], cells[:, 1]].any():
                    continue
                while cells[:, 0].max() < 17 and not board[cells[:, 0] + 1, cells[:, 1]].any():
                    cells = cells + [1, 0]
                placed = board.copy()
                placed[cells[:, 0], cells[:, 1]] = True
                full = placed.all(axis=1)
                lines = int(full.sum())
                placed = np.concatenate((np.zeros((lines, 10), dtype=bool), placed[~full]), axis=0)
                heights = np.where(placed.any(axis=0), 18 - placed.argmax(axis=0), 0)
                holes = int((np.maximum.accumulate(placed, axis=0) & ~placed).sum())
                score = 8 * lines - 0.6 * heights.sum() - 2.0 * holes - 0.35 * np.abs(np.diff(heights)).sum()
                choices.append((float(score), rotation, x))
        current = normalize(np.stack((current[:, 1], -current[:, 0]), axis=1))
    if not choices:
        return 0, 3
    choices.sort(reverse=True)
    # A few near-best placements diversify boards without replacing competent play.
    pick = int(rng.integers(min(3, len(choices)))) if rng.random() < 0.12 else 0
    return choices[pick][1:]


class ScreenController:
    def __init__(self, seed, profile):
        self.rng = np.random.default_rng(seed)
        self.profile = profile
        self.locked = np.zeros((18, 10), dtype=bool)
        self.target = None
        self.rotations = 0
        self.release = False
        self.spawn_release = False
        self.spawns = 0
        self.last_top = 0
        self.hold = "noop"
        self.hold_left = 0

    def choose(self, frame):
        if self.profile == "sustained":
            if self.hold_left == 0:
                self.hold = str(self.rng.choice(("noop", "left", "right", "down")))
                self.hold_left = int(self.rng.integers(16, 49))
            self.hold_left -= 1
            return self.hold
        board = occupancy(frame)
        active = np.argwhere(board & ~self.locked)
        new_spawn = False
        if len(active) != 4 or (active[:, 0].min() <= 2 and self.last_top > 4):
            candidates = [c for c in components(board) if len(c) == 4 and c[:, 0].min() <= 2]
            if candidates:
                active = min(candidates, key=lambda c: int(c[:, 0].min()))
                self.locked = board.copy()
                self.locked[active[:, 0], active[:, 1]] = False
                new_spawn = True
            else:
                self.target = None
                return "down" if self.profile == "topout" else "noop"
        if self.target is None or new_spawn:
            self.spawns += 1
            self.rotations, self.target = best_placement(self.locked, active, self.rng)
            self.spawn_release = True
        self.last_top = int(active[:, 0].min())
        if self.profile == "topout":
            return "down"
        if self.profile == "exploration":
            if self.hold_left <= 0:
                self.hold = str(self.rng.choice(["noop", "left", "right", "down", "a", "b", "up"],
                                                p=[0.2, 0.16, 0.16, 0.32, 0.07, 0.07, 0.02]))
                self.hold_left = int(self.rng.integers(1, 10))
            self.hold_left -= 1
            return self.hold
        if self.release or self.spawn_release:
            self.release = self.spawn_release = False
            return "noop"
        if self.rotations:
            self.rotations -= 1
            self.release = True
            return "a"
        column = int(active[:, 1].min())
        return "left" if column > self.target else "right" if column < self.target else "down"


def split_assignments(episodes, seed):
    if episodes < 10:
        raise ValueError("at least ten independent episodes required")
    validation = max(1, episodes // 10)
    labels = ["train"] * (episodes - 2 * validation) + ["validation"] * validation + ["test"] * validation
    np.random.default_rng(seed).shuffle(labels)
    return labels


def collect_episode(context, seed, steps, profile):
    from pyboy import PyBoy
    rng = np.random.default_rng(seed)
    game = PyBoy(str(ROOT / context["rom"]["file"]), window="null", sound_emulated=False,
                 color_palette=(0xFFFFFF, 0xAAAAAA, 0x555555, 0x000000))
    controller = ScreenController(seed, "heuristic" if profile == "sustained" else profile)
    try:
        game.set_emulation_speed(0)
        boot_frames = int(rng.integers(240, 1000))
        game.tick(boot_frames, True)
        step(game, "a")
        for _ in range(10):
            frame = step(game, "noop")
            if counters(game)["level"] == 1:
                break
        else:
            raise RuntimeError("could not verify gameplay from displayed level")
        # Exclude shared boot/menu frames; burn-in later reconstructs hidden timing.
        warmup_steps = int(rng.integers(24, 36 if profile == "topout" else 90))
        for _ in range(warmup_steps):
            frame = step(game, controller.choose(frame))
        if counters(game)["level"] is None:
            raise RuntimeError("warmup did not leave a gameplay screen")
        controller.profile = profile
        frames = np.empty((steps + 1, 72, 80), dtype=np.uint8)
        actions = np.empty(steps, dtype=np.uint8)
        frames[0] = frame
        current = counters(game)
        events = {"line_clear_steps": [], "lines_cleared": 0, "changed_transitions": 0,
                  "max_occupied_cells": int(occupancy(frame).sum()), "pause_steps": [],
                  "terminal_observed": False}
        invalid = 0
        paused = False
        pause_at = int(rng.integers(120, max(121, steps // 2))) if profile == "pause" else -100
        used = steps
        for t in range(steps):
            if t in (pause_at, pause_at + 30):
                action, paused = "start", not paused
                events["pause_steps"].append(t)
            elif paused:
                action = "noop"
            else:
                action = controller.choose(frame)
            frame = step(game, action)
            frames[t + 1], actions[t] = frame, ACTIONS.index(action)
            events["changed_transitions"] += int(np.any(frames[t] != frame))
            events["max_occupied_cells"] = max(events["max_occupied_cells"], int(occupancy(frame).sum()))
            observed = counters(game)
            if observed["lines"] is not None and current["lines"] is not None and observed["lines"] > current["lines"]:
                events["lines_cleared"] += observed["lines"] - current["lines"]
                events["line_clear_steps"].append(t)
            if observed["lines"] is not None:
                current = observed
            invalid = invalid + 1 if observed["level"] is None and not paused else 0
            if invalid >= 8:
                events["terminal_observed"] = True
                used = t + 1
                break
        events["pieces_observed"] = controller.spawns
        metadata = {"context": context, "source": "capture", "seed": seed, "profile": profile,
                    "boot_frames": boot_frames, "warmup_steps": warmup_steps,
                    "terminal_annotated": bool(events["terminal_observed"]),
                    "events": events, "final_displayed_counters": current,
                    "counter_annotation": "exact native-screen glyph matching; never model inputs"}
        return frames[:used + 1], actions[:used], metadata
    finally:
        game.stop(save=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--profile", action="append",
                        choices=("heuristic", "exploration", "topout", "pause", "sustained"),
                        help="cycle through the supplied capture profiles")
    args = parser.parse_args()
    if args.episodes > 128 or not 256 <= args.steps <= 6000:
        parser.error("require 10..128 episodes and 256..6000 steps")
    if args.episodes * (args.steps + 1) * 72 * 80 > 576 * 1024 ** 2:
        parser.error("raw dataset exceeds 576 MiB budget")
    assignments = split_assignments(args.episodes, args.seed)
    context = json.loads((ROOT / "context.json").read_text())
    rom = ROOT / context["rom"]["file"]
    if hashlib.sha256(rom.read_bytes()).hexdigest() != context["rom"]["sha256"]:
        raise ValueError("ROM hash mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"context": context, "status": "collecting", "episodes": [], "coverage": {},
                "seed": args.seed, "split_unit": "independent run before windowing",
                "limits": {"raw_bytes": 576 * 1024 ** 2, "compressed_bytes": 128 * 1024 ** 2}}
    total_bytes, signatures, action_counts = 0, set(), Counter()
    schedule = args.profile or ("heuristic",) * 7 + ("exploration", "topout", "pause")
    for i, split in enumerate(assignments):
        profile = schedule[i % len(schedule)]
        for attempt in range(16):
            frames, actions, metadata = collect_episode(context, args.seed + i * 1009 + attempt * 1000003, args.steps, profile)
            signature = hashlib.sha256(frames[:min(32, len(frames))].tobytes() + actions[:31].tobytes()).hexdigest()
            if signature not in signatures:
                break
        else:
            raise RuntimeError("duplicate observed prefixes after bounded independent-run attempts")
        signatures.add(signature)
        action_counts.update(ACTIONS[int(a)] for a in actions)
        metadata.update({"split": split, "run_id": i, "prefix_sha256": signature})
        directory = args.output / split
        directory.mkdir(exist_ok=True)
        file = directory / f"episode-{i:03d}.npz"
        with file.open("xb") as output:
            np.savez_compressed(output, frames=frames, actions=actions, metadata=json.dumps(metadata))
        total_bytes += file.stat().st_size
        if total_bytes > manifest["limits"]["compressed_bytes"]:
            raise RuntimeError("compressed dataset budget exceeded")
        manifest["episodes"].append({"file": str(file.relative_to(args.output)), "split": split,
                                     "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                                     "steps": len(actions), "events": metadata["events"], "profile": profile,
                                     "prefix_sha256": signature})
        manifest["coverage"] = {"lines_cleared": sum(e["events"]["lines_cleared"] for e in manifest["episodes"]),
                                "terminal_episodes": sum(e["events"]["terminal_observed"] for e in manifest["episodes"]),
                                "actions": dict(action_counts),
                                "compressed_bytes": total_bytes}
        temporary = args.output / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(args.output / "manifest.json")
    manifest["status"] = "complete"
    manifest["coverage"]["missing_line_clear_coverage"] = manifest["coverage"]["lines_cleared"] == 0
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(args.output), "episodes": len(manifest["episodes"]), "coverage": manifest["coverage"]}))


if __name__ == "__main__":
    main()
