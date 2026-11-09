"""Collect paired A, B and idle ROM outcomes near the floor and pile."""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from pyboy import PyBoy

from audit_controls_reference import replay_prefix, verify_frame
from collect_interventions import choose_contact_anchors, contact_stratum, fork_frames
from model import ACTIONS
from object_annotations import annotate_sequence, classify_board
from run import ROOT, read_episode, step


COMMANDS = tuple((button, "noop", "noop", "noop") for button in ("a", "b", "noop"))
BRANCH_LENGTH = len(COMMANDS[0])


def gather(path, rom, expected_hash):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"capture checksum mismatch: {path}")
    frames, actions, metadata = read_episode(path)
    if metadata["source"] != "capture" or metadata["context"]["rom"]["sha256"] != rom[1]:
        raise ValueError(f"invalid capture lineage: {path}")
    labels, _ = annotate_sequence(frames)
    strata = {}
    for t in np.flatnonzero(np.isin(labels["evidence"], (1, 2))):
        if t < 16 or t + BRANCH_LENGTH >= len(frames):
            continue
        stratum = contact_stratum(classify_board(frames[t]), labels["cells"][t], actions[:t])
        if stratum and ("/gap0/" in stratum or "/gap1/" in stratum):
            strata[t] = stratum
    anchors = choose_contact_anchors(strata, (), count=12)
    if not anchors:
        return []

    def factory():
        return PyBoy(str(rom[0]), window="null", sound_emulated=False,
                     color_palette=(0xFFFFFF, 0xAAAAAA, 0x555555, 0))

    records = []
    game = factory()
    try:
        replay_prefix(game, frames, actions, metadata, 0)
        wanted = set(anchors)
        for t, action in enumerate(actions[:anchors[-1] + 1]):
            if t in wanted:
                with BytesIO() as buffer:
                    game.save_state(buffer)
                    snapshot = buffer.getvalue()
                previous = ACTIONS[int(actions[t - 1])]
                witness = fork_frames(factory, snapshot, previous, frames[t],
                                      [ACTIONS[int(a)] for a in actions[t:t + BRANCH_LENGTH]])
                for i, frame in enumerate(witness):
                    verify_frame(t + i, frames[t + i], frame)
                outputs = np.stack([fork_frames(factory, snapshot, previous, frames[t], controls)
                                    for controls in COMMANDS])
                records.append((t, strata[t], outputs, labels["cells"][t], labels["texture"][t]))
            verify_frame(t + 1, frames[t + 1], step(game, ACTIONS[int(action)]))
    finally:
        game.stop(save=False)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = json.loads((ROOT / "context.json").read_text())
    rom = ROOT / context["rom"]["file"], context["rom"]["sha256"]
    if hashlib.sha256(rom[0].read_bytes()).hexdigest() != rom[1]:
        raise ValueError("reference ROM checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    entries, seen = [], set()
    counts = {"train": Counter(), "validation": Counter()}
    for corpus in args.corpus:
        manifest = json.loads((corpus / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"incomplete corpus: {corpus}")
        for source in manifest["episodes"]:
            prefix = source["prefix_sha256"]
            if prefix in seen:
                raise ValueError("duplicate capture across corpora or splits")
            seen.add(prefix)
            split = source["split"]
            if split not in counts:
                continue
            path = (corpus / source["file"]).resolve()
            if not path.is_relative_to(corpus.resolve()):
                raise ValueError("capture path outside corpus")
            records = gather(path, rom, source["sha256"])
            if not records:
                continue
            indices, strata, outcomes, cells, textures = zip(*records, strict=True)
            output = args.output / f"{split}-{len(entries):03d}.npz"
            np.savez_compressed(output, anchors=indices, strata=strata, frames=outcomes,
                                cells=cells, textures=textures, commands=COMMANDS)
            entries.append({"file": output.name, "split": split, "anchors": len(records),
                            "source_sha256": source["sha256"],
                            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
            counts[split].update(strata)
    if not all(counts.values()):
        raise ValueError("rotation interventions require both training and validation anchors")
    report = {"rom_sha256": rom[1], "commands": COMMANDS, "counts": counts, "records": entries}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
