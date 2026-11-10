"""Record paired rotation and movement branches from independent Adjustris games."""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from pyboy import PyBoy

from audit_controls_reference import replay_prefix, verify_frame
from collect import collect_episode
from collect_interventions import choose_contact_anchors, contact_stratum, fork_frames
from model import ACTIONS
from object_annotations import annotate_sequence, classify_board
from run import ROOT, step


HISTORY = 16
BRANCHES = (
    ("noop",) * 4, ("a", "noop", "noop", "noop"), ("b", "noop", "noop", "noop"),
    ("a", "b", "a", "noop"), ("b", "a", "b", "noop"),
    ("left", "noop", "noop", "noop"), ("right", "noop", "noop", "noop"),
    ("left",) * 4, ("right",) * 4, ("down",) * 4,
)
COMMANDS = np.array([[ACTIONS.index(key) for key in branch] for branch in BRANCHES], dtype=np.uint8)
SPLITS = {"train": range(81530, 81548), "validation": range(90530, 90536), "test": range(99530, 99536)}
PROFILES = ("heuristic", "exploration", "sustained")


def collect(context, seed, profile, factory):
    frames, actions, metadata = collect_episode(context, seed, 480, profile)
    labels, _ = annotate_sequence(frames)
    strata = {}
    for t in range(HISTORY, len(actions) - COMMANDS.shape[1] + 1):
        if labels["evidence"][t] not in (1, 2):
            continue
        board = classify_board(frames[t])
        cells = labels["cells"][t]
        x = cells[:, 1]
        wall = "left" if x.min() <= 1 else "right" if x.max() >= 8 else "center"
        contact = contact_stratum(board, cells, actions[:t])
        contact = contact if contact and ("/gap0/" in contact or "/gap1/" in contact) else "air/down0"
        strata[t] = f"{contact}/{wall}/piece{int(labels['texture'][t])}"
    anchors = choose_contact_anchors(strata, (), count=16)
    if not anchors:
        raise ValueError(f"no visible piece anchors in seed {seed}")

    branches = []
    game = factory()
    try:
        replay_prefix(game, frames, actions, metadata, 0)
        wanted = set(anchors)
        for t in range(anchors[-1] + 1):
            if t in wanted:
                buffer = BytesIO()
                game.save_state(buffer)
                snapshot = buffer.getvalue()
                previous = ACTIONS[actions[t - 1]]
                trace = [ACTIONS[a] for a in actions[t:t + COMMANDS.shape[1]]]
                witness = fork_frames(factory, snapshot, previous, frames[t], trace)
                for i, actual in enumerate(witness):
                    verify_frame(t + i, frames[t + i], actual)
                branches.append(np.stack([fork_frames(factory, snapshot, previous, frames[t], keys)
                                          for keys in BRANCHES]))
            verify_frame(t + 1, frames[t + 1], step(game, ACTIONS[actions[t]]))
    finally:
        game.stop(save=False)
    indices = np.asarray(anchors)
    return {"frames": frames, "actions": actions, "anchors": indices, "branches": np.stack(branches),
            "strata": np.array([strata[t] for t in anchors]), "cells": labels["cells"][indices],
            "texture": labels["texture"][indices]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    context = json.loads((ROOT / "context.json").read_text())
    rom = ROOT / context["rom"]["file"]
    if hashlib.sha256(rom.read_bytes()).hexdigest() != context["rom"]["sha256"]:
        raise ValueError("reference ROM checksum mismatch")
    args.output.mkdir()

    def factory():
        return PyBoy(str(rom), window="null", sound_emulated=False,
                     color_palette=(0xFFFFFF, 0xAAAAAA, 0x555555, 0))

    episodes = []
    for split, seeds in SPLITS.items():
        counts = Counter()
        for i, seed in enumerate(seeds):
            profile = PROFILES[i % len(PROFILES)]
            capture = collect(context, seed, profile, factory)
            path = args.output / f"{split}-{seed}.npz"
            np.savez_compressed(path, **capture)
            episodes.append({"split": split, "seed": seed, "profile": profile, "file": path.name,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "anchors": len(capture["anchors"])})
            counts.update(capture["strata"].tolist())
        print(json.dumps({"split": split, "anchors": sum(counts.values()), "strata": len(counts)}), flush=True)
    (args.output / "manifest.json").write_text(json.dumps({
        "context": context, "commands": BRANCHES, "history": HISTORY, "episodes": episodes
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
