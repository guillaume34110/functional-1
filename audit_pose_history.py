"""Measure exact visible-history ambiguity for proven next-piece poses."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model import ACTIONS
from run import ROOT, read_episode
from token_model import encode_frames


LENGTHS = (1, 4, 16, 32, 64)
CONDITIONS = (*LENGTHS, "explicit_memory", "board_cadence")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temporal_keys(codes, ids, actions):
    """Use only tokens through t and the command at t."""
    frames = codes[:-1]
    changed = frames[1:] != frames[:-1]
    times = np.arange(len(actions), dtype=np.int32)
    tile_change = np.concatenate((np.ones_like(frames[:1], dtype=bool), changed))
    last_tile_change = np.maximum.accumulate(np.where(tile_change, times[:, None, None], 0), axis=0)
    ages = (times[:, None, None] - last_tile_change).clip(0, 255).astype(np.uint8)
    button_change = np.r_[True, actions[1:] != actions[:-1]]
    held = (times - np.maximum.accumulate(np.where(button_change, times, 0)) + 1).clip(0, 255)
    previous = np.r_[ids[0], ids[:-2]]
    previous_action = np.r_[actions[0], actions[:-1]]
    explicit = [age.tobytes() + prev.tobytes() + bytes((int(duration), int(button)))
                for age, prev, duration, button in zip(ages, previous, held, previous_action)]

    # The age of the latest board change is already implied by tile ages.
    # The two completed intervals expose additional cadence across changes.
    board_changes = np.flatnonzero(tile_change[:, :, 2:12].any(axis=(1, 2)))
    intervals = np.diff(board_changes).clip(0, 255).astype(np.int16)
    history = np.r_[-1, -1, intervals]
    last = np.searchsorted(board_changes, times, side="right") - 1
    cadence = np.stack((history[last + 1], history[last]), axis=1).astype(np.int16)
    return explicit, cadence


def load_rows(checkpoint):
    data = ROOT / "data"
    annotations = data / "object-annotations-v2"
    manifest_path, audit_path = data / "manifest.json", annotations / "object-audit.json"
    manifest, audit = (json.loads(p.read_bytes()) for p in (manifest_path, audit_path))
    manifest_sha = digest(manifest_path)
    if audit["test_used"] or audit["manifest_sha256"] != manifest_sha:
        raise ValueError("annotation provenance mismatch")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if weights["config"]["manifest_sha256"] != manifest_sha:
        raise ValueError("checkpoint provenance mismatch")
    codebook = weights["model"]["codebook"].numpy()
    records = {r["file"]: r for r in audit["episodes"]}
    # Intern exact, reversibly encoded screens. Dictionary equality compares bytes, including hash collisions.
    screens = {}
    episodes = {"train": [], "validation": []}
    for entry in manifest["episodes"]:
        split, name = entry["split"], entry["file"]
        if split not in episodes:
            continue
        source = (data / name).resolve()
        label_path = (annotations / records[name]["labels_file"]).resolve()
        if (not source.is_relative_to(data.resolve()) or not label_path.is_relative_to(annotations.resolve())
                or digest(source) != entry["sha256"] or records[name]["split"] != split):
            raise ValueError("capture or annotation lineage mismatch")
        frames, actions, _ = read_episode(source)
        codes, unknown = encode_frames(frames, codebook)
        if unknown:
            raise ValueError("codec is not reversible on the audited episodes")
        tile_dtype = np.min_scalar_type(len(codebook) - 1)
        ids = np.array([screens.setdefault(frame.tobytes(), len(screens)) for frame in codes.astype(tile_dtype)],
                       dtype=np.uint32)
        with np.load(label_path, allow_pickle=False) as labels:
            if json.loads(str(labels["metadata"]))["source_sha256"] != entry["sha256"]:
                raise ValueError("annotation source mismatch")
            origins = labels["cells"].min(1)
            shape = labels["shape_id"]
            known = (shape >= 0) & np.isin(labels["evidence"], (1, 2))
        pose = np.where(known, shape * 180 + origins[:, 0] * 10 + origins[:, 1], -1)
        actions = actions.astype(np.uint8)
        explicit, cadence = temporal_keys(codes, ids, actions)
        episodes[split].append((name, ids, actions, pose, explicit, cadence))
    return episodes, manifest_sha, digest(audit_path)


def summarize(groups):
    counts = [sum(g.values()) for g, _ in groups.values()]
    ambiguous = [(g, witnesses) for g, witnesses in groups.values() if len(g) > 1]
    errors = sum(sum(g.values()) - max(g.values()) for g, _ in ambiguous)
    return {"known_rows": sum(counts), "distinct_keys": len(groups), "repeated_rows": sum(n for n in counts if n > 1),
            "multi_episode_keys": sum(len({file for file, _, _ in w}) > 1 for _, w in groups.values()),
            "ambiguous_keys": len(ambiguous), "conflicting_rows": errors,
            "empirical_minimum_error": errors / sum(counts) if sum(counts) else None,
            "examples": [w for _, w in ambiguous[:5]]}


def group(episodes, condition, down_only):
    groups = defaultdict(lambda: (Counter(), []))
    unknown = 0
    for name, ids, actions, pose, explicit, cadence in episodes:
        # A common cohort across lengths keeps early frames from changing the denominator.
        for t in range(max(LENGTHS) - 1, len(actions)):
            if down_only and actions[t] != ACTIONS.index("down"):
                continue
            outcome = int(pose[t + 1])
            if outcome < 0:
                unknown += 1
                continue
            if isinstance(condition, int):
                start = t - condition + 1
                key = ids[start:t + 1].tobytes() + actions[start:t + 1].tobytes()
            else:
                key = ids[t].tobytes() + actions[t].tobytes() + explicit[t]
                if condition == "board_cadence":
                    key += cadence[t].tobytes()
            counts, witnesses = groups[key]
            counts[outcome] += 1
            witnesses.append((name, t, outcome))
    return groups, unknown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes, manifest_sha, annotations_sha = load_rows(args.checkpoint)
    report = {"manifest_sha256": manifest_sha, "annotations_sha256": annotations_sha,
              "checkpoint_sha256": digest(args.checkpoint), "test_used": False, "results": {}}
    for condition in CONDITIONS:
        result = report["results"][str(condition)] = {}
        for scope in ("all", "down"):
            train, unknown_train = group(episodes["train"], condition, scope == "down")
            validation, unknown_validation = group(episodes["validation"], condition, scope == "down")
            seen = correct = 0
            for key in validation.keys() & train.keys():
                outcomes = validation[key][0]
                seen += sum(outcomes.values())
                majority = train[key][0]
                predicted = min(majority, key=lambda pose: (-majority[pose], pose))
                correct += outcomes[predicted]
            result[scope] = {"train": {**summarize(train), "unknown_rows": unknown_train},
                             "validation": {**summarize(validation), "unknown_rows": unknown_validation},
                             "validation_train_key_rows": seen, "validation_train_majority_correct": correct}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
