"""Calibrate two V2 readouts using proved spawns and paired ROM rotations."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_spawn import preview_texture, spawn_evidence
from object_annotations import PATTERNS, candidates, classify_board, shape_key, without
from run import read_episode
from scene_model import scene_from_weights
from token_model import decode_tokens


def partitions(corpora):
    records = {"train": [], "validation": []}
    seen = set()
    for root in corpora:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"incomplete corpus: {root}")
        for row in manifest["episodes"]:
            prefix = row["prefix_sha256"]
            if prefix in seen:
                raise ValueError("duplicate capture prefix across corpora or splits")
            seen.add(prefix)
            if row["split"] in records:
                path = (root / row["file"]).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("capture path outside corpus")
                records[row["split"]].append((path, row["sha256"]))
    return records


def spawn_counts(records):
    counts = Counter()
    for path, expected in records:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"invalid capture digest: {path}")
        frames, _, metadata = read_episode(path)
        if metadata["source"] != "capture":
            raise ValueError(f"expected a ROM capture: {path}")
        boards = [classify_board(frame) for frame in frames]
        for i, (before, after) in enumerate(zip(boards, boards[1:])):
            changed = before != after
            if changed.sum() != 4 or ((before == -1) & (after >= 0) & changed).sum() != 4:
                continue
            previous, following = (preview_texture(frame) for frame in frames[i:i + 2])
            if previous is None or following is None:
                continue
            for _, cells in candidates(before):
                proof = spawn_evidence(frames[i], frames[i + 1], np.asarray(cells))
                if proof and proof["texture"] == previous:
                    counts[previous, following] += 1
                    break
    return counts


def rotation_examples(root, model):
    kinds = {}
    for i, (shape, tile) in enumerate(zip(model.object_shapes, model.object_tiles, strict=True)):
        colors = np.flatnonzero((PATTERNS == model.codebook[tile].numpy()).all((1, 2)))
        if len(colors) == 1:
            kinds[shape_key(int(colors[0]), sorted(shape.tolist()))] = i
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest["records"]:
        if entry["split"] not in ("train", "validation"):
            continue
        path = (root / entry["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("invalid rotation capture path or digest")
        with np.load(path, allow_pickle=False) as data:
            for branches, cells, texture in zip(data["frames"], data["cells"], data["textures"], strict=True):
                before = classify_board(branches[0, 0])
                pile = without(before, cells)
                adjacent = np.pad(pile >= 0, 1)
                adjacent[-1] = True
                contact = any(adjacent[y:y + 3, x:x + 3].any() for y, x in cells)
                origin = cells.min(0)
                source = kinds.get(shape_key(int(texture), cells.tolist()))
                if source is None:
                    continue
                for button in range(2):
                    after = classify_board(branches[button, 1])
                    idle = classify_board(branches[2, 1])
                    if np.array_equal(after, idle) or (after < -1).any():
                        continue
                    matches = [np.asarray(pose) for style, pose in candidates(after)
                               if style == texture and np.array_equal(without(after, pose), pile)]
                    if len(matches) != 1:
                        continue
                    pose = matches[0]
                    target = kinds.get(shape_key(int(texture), pose.tolist()))
                    dy, dx = map(int, pose.min(0) - origin)
                    if target is not None and target != source and max(abs(dy), abs(dx)) <= 3:
                        yield entry["split"], contact, (source, button), (target, dy, dx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True, action="append")
    parser.add_argument("--rotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    model = scene_from_weights(checkpoint["model"]).eval()
    records = partitions(args.corpus)
    counts = {split: spawn_counts(entries) for split, entries in records.items()}
    motifs = decode_tokens(torch.nn.functional.pad(model.preview_patterns, (13, 1, 0, 12)), model.codebook)
    textures = [preview_texture(frame[0].numpy()) for frame in motifs]
    if set(textures) != set(range(7)) or len(textures) != 7:
        raise ValueError("expected one frozen preview motif for each texture")
    frequency = torch.tensor([sum(n for (_, after), n in counts["train"].items() if after == color)
                              for color in textures], dtype=torch.float32)
    if not (frequency > 0).all():
        raise ValueError("missing training observations for a preview motif")
    prior = frequency / frequency.sum()
    votes = {contact: defaultdict(Counter) for contact in (False, True)}
    witnesses = {contact: [] for contact in (False, True)}
    for split, contact, key, target in rotation_examples(args.rotations, model):
        if split == "train":
            votes[contact][key][target] += 1
        else:
            witnesses[contact].append((key, target))
    readouts = {contact: {key: next(iter(outcomes)) for key, outcomes in by_key.items()
                          if len(outcomes) == 1}
                for contact, by_key in votes.items()}
    learned, validation = readouts[True], witnesses[True]
    if not learned or not validation:
        raise ValueError("incomplete witnessed contact rotation coverage")
    weights = {**checkpoint["model"], "preview_prior": prior}
    for contact, name in ((True, "contact_rotations"), (False, "free_rotations")):
        rotations = torch.zeros((len(model.object_shapes), 2, 3), dtype=torch.int16)
        rotations[..., 0] = -1
        for key, target in readouts[contact].items():
            rotations[key] = torch.tensor(target, dtype=torch.int16)
        weights[name] = rotations
    scene_from_weights(weights)
    report = {
        "preview_train_counts": frequency.int().tolist(),
        "preview_texture_order": textures,
        "preview_validation": sum(counts["validation"].values()),
        "contact_train_examples": sum(sum(outcomes.values()) for outcomes in votes[True].values()),
        "contact_learned_contexts": len(learned),
        "contact_validation": len(validation),
        "contact_validation_covered": sum(key in learned for key, _ in validation),
        "contact_validation_exact": sum(learned.get(key) == target for key, target in validation),
        "free_train_examples": sum(sum(outcomes.values()) for outcomes in votes[False].values()),
        "free_validation": len(witnesses[False]),
        "free_validation_covered": sum(key in readouts[False] for key, _ in witnesses[False]),
        "free_validation_exact": sum(readouts[False].get(key) == target for key, target in witnesses[False]),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "rotation_manifest_sha256": hashlib.sha256((args.rotations / "manifest.json").read_bytes()).hexdigest(),
        "test_used": False,
    }
    args.output.mkdir(parents=True)
    torch.save({**checkpoint, "model": weights, "v2_readout_fit": report}, args.output / "model.pt")
    (args.output / "fit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
