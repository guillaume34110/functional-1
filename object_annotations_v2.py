"""Offline target-only propagation of V1 visual proofs; never runtime inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from object_annotations import classify_board, shape_key, without
from run import read_episode

VERSION = "visual-proof-propagation-v2"


def connected(cells):
    remaining = set(map(tuple, cells))
    stack = [remaining.pop()]
    while stack:
        y, x = stack.pop()
        for point in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if point in remaining:
                remaining.remove(point); stack.append(point)
    return not remaining


def extend_labels(frames, original, dictionary):
    """Strict identical-board backfill, then residual-four forward proof.

    Existing incompatible proofs and invalid/ambiguous frames are barriers.
    No new dictionary entries; an unseen residual silhouette is rejected.
    """
    labels = {key: value.copy() for key, value in original.items()}
    labels["propagation"] = np.zeros(len(frames), dtype=np.uint8)
    grids = np.stack([classify_board(frame) for frame in frames])
    valid = np.all(grids >= -1, axis=(1, 2))
    stats = {"backfilled": 0, "continued": 0, "conflicts": 0, "unseen_rejected": 0}

    def assign(index, cells, texture, track, source):
        if not valid[index] or labels["evidence"][index] in (3, 4):
            return False
        cells = np.asarray(sorted(map(tuple, cells)), dtype=np.int16)
        if not np.all(grids[index][cells[:, 0], cells[:, 1]] == texture):
            return False
        if labels["texture"][index] >= 0:
            same = labels["texture"][index] == texture and np.array_equal(labels["cells"][index], cells)
            stats["conflicts"] += int(not same)
            return same
        key = shape_key(texture, cells)
        if key not in dictionary:
            stats["unseen_rejected"] += 1
            return False
        labels["cells"][index], labels["texture"][index] = cells, texture
        labels["track"][index], labels["evidence"][index] = track, 2
        labels["shape_id"][index] = dictionary[key]
        labels["propagation"][index] = source
        stats["backfilled" if source == 1 else "continued"] += 1
        return True

    def backfill():
        for index in range(len(frames) - 1, 0, -1):
            if (labels["shape_id"][index] >= 0 and labels["evidence"][index] in (1, 2)
                    and valid[index] and valid[index - 1]
                    and np.array_equal(grids[index], grids[index - 1])):
                assign(index - 1, labels["cells"][index], int(labels["texture"][index]),
                       int(labels["track"][index]), 1)

    backfill()
    for index in range(1, len(frames)):
        if (labels["shape_id"][index - 1] < 0 or labels["evidence"][index - 1] not in (1, 2)
                or not valid[index - 1] or not valid[index]):
            continue
        texture = int(labels["texture"][index - 1])
        old = labels["cells"][index - 1]
        background = without(grids[index - 1], tuple(map(tuple, old)))
        cells = np.argwhere(grids[index] != background)
        if (len(cells) != 4 or not np.all(grids[index][cells[:, 0], cells[:, 1]] == texture)
                or not connected(cells) or not np.array_equal(without(grids[index], tuple(map(tuple, cells))), background)):
            continue
        assign(index, cells, texture, int(labels["track"][index - 1]), 2)
    backfill()
    stats.update({"frames": len(frames), "labeled_frames": int((labels["shape_id"] >= 0).sum()),
                  "groups": len(np.unique(labels["track"][labels["track"] >= 0])),
                  "invalid_pattern_frames": int((~valid).sum())})
    return labels, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = (args.data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    provenance = (args.v1 / "object-audit.json").read_bytes()
    v1 = json.loads(provenance)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest.get("status") != "complete" or v1["test_used"] or v1["manifest_sha256"] != digest:
        raise ValueError("invalid V1/dataset lineage")
    dictionary = {shape_key(item["texture"], item["cells"]): item["id"] for item in v1["dictionary"]}
    records = {item["file"]: item for item in v1["episodes"]}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"algorithm_version": VERSION, "test_used": False, "dictionary_from": "train-only-frozen-v1",
              "sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("object_annotations_v2.py", "object_annotations.py")},
              "dictionary": v1["dictionary"], "manifest_sha256": digest,
              "v1_report_sha256": hashlib.sha256(provenance).hexdigest(), "episodes": [],
              "future_pair_is_target_only": True, "propagation_codes": {"0": "V1", "1": "identical-board-backfill", "2": "residual-four-continuation"}}
    for entry in manifest["episodes"]:
        if entry["split"] not in ("train", "validation"):
            continue
        source = (args.data / entry["file"]).resolve()
        if not source.is_relative_to(args.data.resolve()) or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("episode source mismatch")
        record = records[entry["file"]]
        label_path = (args.v1 / record["labels_file"]).resolve()
        if record["split"] != entry["split"] or not label_path.is_relative_to(args.v1.resolve()):
            raise ValueError("V1 split/path mismatch")
        frames, _, metadata = read_episode(source)
        if metadata.get("source") != "capture":
            raise ValueError("reference capture required")
        with np.load(label_path, allow_pickle=False) as archive:
            if json.loads(str(archive["metadata"]))["source_sha256"] != entry["sha256"]:
                raise ValueError("V1 source mismatch")
            original = {key: archive[key].copy() for key in ("cells", "texture", "track", "evidence", "shape_id")}
        labels, stats = extend_labels(frames, original, dictionary)
        target = args.output / record["labels_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **labels, metadata=json.dumps({"source_sha256": entry["sha256"],
                            "v1_labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
                            "algorithm_version": VERSION, "future_pair_is_target_only": True}))
        stats.update({"file": entry["file"], "split": entry["split"], "labels_file": record["labels_file"]})
        report["episodes"].append(stats)
    report["coverage"] = {split: {key: sum(e[key] for e in report["episodes"] if e["split"] == split)
                                  for key in ("frames", "labeled_frames", "backfilled", "continued", "unseen_rejected")}
                          for split in ("train", "validation")}
    (args.output / "object-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(args.output), "coverage": report["coverage"]}))


if __name__ == "__main__":
    main()
