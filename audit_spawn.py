"""Prove visible deposit/spawn pairs and compare the previous preview offline."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_controls import supported
from model import ACTIONS
from object_annotations import PATTERNS, classify_board, shape_key
from run import ROOT, read_episode


def preview_texture(frame):
    patches = frame[:24, 52:76].reshape(6, 4, 6, 4).transpose(0, 2, 1, 3)
    matches = (patches[:, :, None] == PATTERNS[None, None]).all(axis=(3, 4))
    counts = matches.sum(axis=(0, 1))
    if counts.sum() != 4 or counts.max() != 4:
        return None
    return int(counts.argmax())


def spawn_evidence(before, after, cells):
    old = classify_board(before)
    new = classify_board(after)
    if (old < -1).any() or (new < -1).any():
        return None
    background = old.copy()
    background[cells[:, 0], cells[:, 1]] = -1
    if not supported(background, cells, -1):
        return None
    changed = old != new
    inserted = np.argwhere(changed)
    if len(inserted) != 4 or inserted[:, 0].max() > 3 or np.any(old[changed] != -1):
        return None
    styles = new[changed]
    if styles[0] < 0 or not np.all(styles == styles[0]):
        return None
    return {"texture": int(styles[0]), "cells": inserted.tolist(), "preview": preview_texture(before)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from audit_conditioning import reference

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = ROOT / "data"
    annotation_dir = data / "object-annotations-v2"
    manifest_path, annotation_path = data / "manifest.json", annotation_dir / "object-audit.json"
    manifest, annotation = (json.loads(p.read_bytes()) for p in (manifest_path, annotation_path))
    manifest_sha = sha(manifest_path)
    if annotation["test_used"] or annotation["manifest_sha256"] != manifest_sha:
        raise ValueError("annotation lineage mismatch")
    records = {e["file"]: e for e in annotation["episodes"]}
    dictionary = {shape_key(item["texture"], item["cells"]) for item in annotation["dictionary"]}
    counts = {split: Counter() for split in ("train", "validation")}
    examples = []
    replay = None
    for entry in manifest["episodes"]:
        split, name = entry["split"], entry["file"]
        if split not in counts:
            continue
        source = (data / name).resolve()
        label_path = (annotation_dir / records[name]["labels_file"]).resolve()
        if (not source.is_relative_to(data.resolve()) or not label_path.is_relative_to(annotation_dir.resolve())
                or sha(source) != entry["sha256"] or records[name]["split"] != split):
            raise ValueError("unsafe capture or label path")
        frames, actions, metadata = read_episode(source)
        if split == "validation" and Path(name).name == "episode-063.npz":
            commands = np.full(16, ACTIONS.index("down"), dtype=np.uint8)
            replay = reference(frames, actions, metadata, 813, commands, False)
        with np.load(label_path, allow_pickle=False) as labels:
            if json.loads(str(labels["metadata"]))["source_sha256"] != entry["sha256"]:
                raise ValueError("label source mismatch")
            known = (labels["shape_id"][:-1] >= 0) & np.isin(labels["evidence"][:-1], (1, 2))
            cells = labels["cells"].copy()
        changed_at_top = (frames[:-1, :16, 8:48] != frames[1:, :16, 8:48]).any(axis=(1, 2))
        for t in np.flatnonzero(known & changed_at_top):
            proof = spawn_evidence(frames[t], frames[t + 1], cells[t])
            if proof is None:
                continue
            if shape_key(proof["texture"], proof["cells"]) not in dictionary:
                counts[split]["unrecognized_insertions"] += 1
                continue
            row = counts[split]
            row["proven_spawns"] += 1
            row["readable_previews"] += proof["preview"] is not None
            row["matching_previews"] += proof["preview"] == proof["texture"]
            row[f"texture/{proof['texture']}"] += 1
            if len(examples) < 14 or (proof["preview"] is not None and proof["preview"] != proof["texture"]):
                examples.append({"split": split, "file": name, "step": int(t), "button": int(actions[t]), **proof})
    if replay is None:
        raise ValueError("missing validation replay for the recurring collision")
    images, prefix = replay
    with args.output.with_suffix(".npz").open("xb") as handle:
        np.savez_compressed(handle, frames=images, anchor=813)
    report = {"test_used": False, "manifest_sha256": manifest_sha, "annotations_sha256": sha(annotation_path),
              "counts": counts, "examples": examples, "reference_replay": prefix}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
