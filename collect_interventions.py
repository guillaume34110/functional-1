"""Immutable screen/action intervention addon; emulator snapshots stay ephemeral."""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from audit_controls import BUTTONS
from audit_controls_reference import ReplayMismatch, replay_prefix, verify_frame
from model import ACTIONS
from intervention_training import verify_extension
from object_annotations import annotate_sequence, candidates, classify_board, shape_key, without
from object_annotations_v2 import extend_labels
from run import ROOT, observe, read_episode, step


def choose_anchors(labels, action_count, count=8, exclude=()):
    eligible = np.flatnonzero((labels["shape_id"] >= 0) & np.isin(labels["evidence"], (1, 2)))
    eligible = eligible[(eligible >= 32) & (eligible + 8 <= action_count)]
    eligible = eligible[~np.isin(eligible, list(exclude))]
    if not len(eligible):
        return []
    indices = np.rint(np.linspace(0, len(eligible) - 1, min(count, len(eligible)))).astype(int)
    return eligible[indices].tolist()


def contact_stratum(board, cells, past_actions):
    background = without(board, cells)
    best = None
    for y, x in cells:
        for gap in range(4):
            row = y + 1 + gap
            if row >= 18 or background[row, x] >= 0:
                candidate = (gap, "floor" if row >= 18 else "pile")
                if best is None or candidate < best:
                    best = candidate
                break
    if best is None:
        return None
    held = 0
    for action in reversed(past_actions[-8:]):
        if action != ACTIONS.index("down"):
            break
        held += 1
    return f"{best[1]}/gap{best[0]}/down{min(held, 2)}"


def contact_candidates(frames, actions, labels):
    """Use only a unique movement in the observed prefix to identify the object."""
    previous, old_objects, active = None, [], None
    strata = {}
    for t, frame in enumerate(frames):
        board = classify_board(frame)
        objects = candidates(board) if np.all(board >= -1) else []
        if previous is None or not np.array_equal(previous, board):
            matches = []
            if previous is not None and np.count_nonzero(previous != board) <= 8:
                for old_style, old_cells in old_objects:
                    old_background = without(previous, old_cells)
                    for style, cells in objects:
                        if style == old_style and cells != old_cells and np.array_equal(old_background, without(board, cells)):
                            matches.append((style, cells))
            active = matches[0] if len(matches) == 1 else None
        previous, old_objects = board, objects
        if active is None or t < 32 or t + 8 > len(actions):
            continue
        style, cells = active
        # Existing frozen labels must agree with the independent causal observation.
        if (labels["shape_id"][t] < 0 or labels["evidence"][t] not in (1, 2)
                or labels["texture"][t] != style or set(map(tuple, labels["cells"][t])) != set(cells)):
            continue
        stratum = contact_stratum(board, cells, actions[:t])
        if stratum is not None:
            strata[t] = stratum
    return strata


def choose_contact_anchors(strata, exclude, count=8):
    groups = {}
    for anchor, stratum in sorted(strata.items()):
        if anchor not in exclude:
            groups.setdefault(stratum, []).append(anchor)
    order = sorted(groups, key=lambda key: (-int(key[-1]), not key.startswith("pile/"), key))
    quotas = Counter()
    while sum(quotas.values()) < count:
        available = [key for key in order if quotas[key] < len(groups[key])]
        if not available:
            break
        for key in available[:count - sum(quotas.values())]:
            quotas[key] += 1
    chosen = []
    for key, quota in quotas.items():
        group = groups[key]
        # Bin midpoints distribute samples even when a stratum gets just one slot.
        chosen.extend(group[(2 * i + 1) * len(group) // (2 * quota)] for i in range(quota))
    return sorted(chosen)


def inherit_base(directory, output, report, parents):
    payload = (directory / "manifest.json").read_bytes()
    base = json.loads(payload)
    if (base["version"] not in (1, 2) or report.get("version", base["version"] + 1) != base["version"] + 1
            or base["status"] != "complete" or base["test_used"]
            or base["witness_frames"] != len(base["episodes"]) * 9):
        raise ValueError("extension requires a complete immutable V1 or V2")
    if base["version"] == 2:
        verify_extension(directory)
    for key in ("parent_manifest_sha256", "annotations_sha256", "dictionary", "buttons", "steps"):
        if base[key] != report[key]:
            raise ValueError(f"base addon lineage mismatch: {key}")
    report.update({"base_directory": directory.name, "base_manifest_sha256": hashlib.sha256(payload).hexdigest(),
                   "inherited_anchors": len(base["episodes"]), "new_anchors": 0,
                   "inherited_witness_frames": len(base["episodes"]) * 9, "new_witness_frames": 0,
                   "inherited_coverage": base.get("coverage", {})})
    if base["version"] == 2:
        report["inherited_validation_action_histogram"] = base.get("parent_action_histograms", {}).get("validation", {})
    for record in base["episodes"]:
        parent = parents.get(record["parent_file"])
        if parent is None or record["split"] != parent["split"] or record["parent_sha256"] != parent["sha256"] or not record["witness_verified"]:
            raise ValueError("base record crosses parent split or lacks witness")
        source, target = (directory / record["file"]).resolve(), (output / record["file"]).resolve()
        if not source.is_relative_to(directory.resolve()) or not target.is_relative_to(output.resolve()):
            raise ValueError("unsafe inherited path")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("base NPZ hash mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("inherited NPZ copy mismatch")
        report["episodes"].append(record)
        report["compressed_bytes"] += len(data)
    report["raw_image_bytes"] = len(base["episodes"]) * 6 * 9 * 72 * 80
    report["witness_frames"] = report["inherited_witness_frames"]


def fork_frames(factory, snapshot, previous_action, initial, commands, step_fn=step, observe_fn=observe):
    game = factory()
    try:
        game.set_emulation_speed(0)
        game.load_state(BytesIO(snapshot))
        if previous_action != "noop":
            game.button_release(previous_action)
        verify_frame(0, initial, observe_fn(game))
        frames = [initial.copy()]
        for command in commands:
            frames.append(step_fn(game, command))
        return np.stack(frames)
    finally:
        game.stop(save=False)


def witnessed_forks(factory, snapshot, previous, initial, actual_actions, expected, fork_fn=fork_frames):
    witness = fork_fn(factory, snapshot, previous, initial, [ACTIONS[int(a)] for a in actual_actions])
    for index, (reference, actual) in enumerate(zip(expected, witness, strict=True)):
        verify_frame(index, reference, actual)
    commands = [[button if button not in ("a", "b") or t == 0 else "noop" for t in range(8)] for button in BUTTONS]
    frames = np.stack([fork_fn(factory, snapshot, previous, initial, branch) for branch in commands])
    actions = np.array([[ACTIONS.index(action) for action in branch] for branch in commands], dtype=np.uint8)
    return frames, actions


def branch_labels(frames, seed, dictionary):
    labels, _ = annotate_sequence(frames)
    ids = np.full(len(frames), -1, dtype=np.int32)
    unknown = 0
    for index in np.flatnonzero(labels["texture"] >= 0):
        ids[index] = dictionary.get(shape_key(labels["texture"][index], labels["cells"][index]), -1)
        if ids[index] < 0:
            unknown += 1
            labels["cells"][index] = -1; labels["texture"][index] = -1
            labels["track"][index] = -1; labels["evidence"][index] = 0
    labels["shape_id"] = ids
    if labels["texture"][0] >= 0 and (labels["texture"][0] != seed["texture"] or not np.array_equal(labels["cells"][0], seed["cells"])):
        raise ValueError("branch visual proof contradicts captured seed")
    for key in ("cells", "texture", "shape_id", "track", "evidence"):
        labels[key][0] = seed[key]
    labels, stats = extend_labels(frames, labels, dictionary)
    return labels, unknown + stats["unseen_rejected"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--annotations", type=Path, default=Path("data/object-annotations-v2"))
    parser.add_argument("--output", type=Path, default=Path("data/action-interventions-v1"))
    parser.add_argument("--extend-from", type=Path, help="extend immutable V1 or add targeted TRAIN anchors to V2")
    args = parser.parse_args()
    manifest_bytes = (args.data / "manifest.json").read_bytes()
    annotation_bytes = (args.annotations / "object-audit.json").read_bytes()
    parent, annotations = json.loads(manifest_bytes), json.loads(annotation_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if parent["status"] != "complete" or annotations["test_used"] or annotations["manifest_sha256"] != digest:
        raise ValueError("parent/annotation lineage mismatch")
    entries = [entry for entry in parent["episodes"] if entry["split"] in ("train", "validation")]
    if len(entries) > 58:
        raise ValueError("addon limit is58 parent episodes")
    dictionary = {shape_key(item["texture"], item["cells"]): item["id"] for item in annotations["dictionary"]}
    records = {record["file"]: record for record in annotations["episodes"]}
    version = json.loads((args.extend_from / "manifest.json").read_bytes())["version"] + 1 if args.extend_from else 1
    if version not in (1, 2, 3):
        raise ValueError("supported addon versions are V1, V2 and V3")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"version": version, "status": "collecting", "test_used": False, "algorithm": f"witnessed-screen-interventions-v{version}",
              "parent_manifest_sha256": digest, "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
              "dictionary": annotations["dictionary"], "dictionary_from": "frozen-parent-train-only",
              "buttons": list(BUTTONS), "steps": 8, "anchors_per_parent_max": {1: 8, 2: 32, 3: 40}[version], "episodes": [],
              "anchor_selection": "up-to8 equal quantiles of all proven eligible frames; no model/outcome filtering",
              "witness_frames": 0, "verified_parent_frames": 0, "raw_image_bytes": 0, "compressed_bytes": 0,
              "parent_action_histograms": {}, "coverage": {"train": {}, "validation": {}},
              "sourcehashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                          ("collect_interventions.py", "collect.py", "run.py", "audit_controls_reference.py", "object_annotations.py", "object_annotations_v2.py", "intervention_training.py")},
              "snapshot_policy": "BytesIO only; never persisted or supplied to neural model; queued previous release reconstructed and witnessed",
              "labels_target_only": True}
    report["raw_image_cap_bytes"] = (768 if args.extend_from else 192) * 1024 ** 2
    if version == 2:
        report["anchor_selection"] = "preserve V1 anchors/bytes, add up-to24 quantiles of remaining proven frames; no neural-outcome selection"
    if version == 3:
        report["anchor_selection"] = "preserve V2 records/bytes; add at most8 TRAIN anchors per parent, stratified by observed floor/pile gap0..3 and recent down tenure0/1/2+; causal object proof and no branch-outcome selection"
        report["train_contact_strata"] = {"eligible": {}, "inherited": {}, "added": {}, "after": {}}
    histograms = {split: Counter() for split in ("train", "validation")}
    coverage = {split: Counter() for split in ("train", "validation")}
    try:
        if args.extend_from:
            inherit_base(args.extend_from, args.output, report, {entry["file"]: entry for entry in entries})
            for split in coverage:
                coverage[split].update(report["inherited_coverage"].get(split, {}))
            if version == 3:
                histograms["validation"].update(report["inherited_validation_action_histogram"])
        for entry in entries:
            if version == 3 and entry["split"] != "train":
                continue
            source = (args.data / entry["file"]).resolve()
            record = records[entry["file"]]
            label_path = (args.annotations / record["labels_file"]).resolve()
            if (not source.is_relative_to(args.data.resolve()) or not label_path.is_relative_to(args.annotations.resolve())
                    or record["split"] != entry["split"] or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]):
                raise ValueError("parent source/split mismatch")
            frames, parent_actions, metadata = read_episode(source)
            if metadata.get("source") != "capture":
                raise ValueError("parent must be a real capture")
            with np.load(label_path, allow_pickle=False) as archive:
                if json.loads(str(archive["metadata"]))["source_sha256"] != entry["sha256"]:
                    raise ValueError("annotation source mismatch")
                labels = {key: archive[key].copy() for key in ("cells", "texture", "shape_id", "track", "evidence")}
            inherited = {record["anchor"] for record in report["episodes"] if record["parent_file"] == entry["file"]} if args.extend_from else set()
            if version == 3:
                strata = contact_candidates(frames, parent_actions, labels)
                anchors = set(choose_contact_anchors(strata, inherited))
                stats = report["train_contact_strata"]
                additions = {"eligible": Counter(strata.values()),
                             "inherited": Counter(strata.get(t, "outside_target_strata") for t in inherited),
                             "added": Counter(strata[t] for t in anchors),
                             "after": Counter(strata.get(t, "outside_target_strata") for t in inherited | anchors)}
                for name, values in additions.items():
                    combined = Counter(stats[name])
                    combined.update(values)
                    stats[name] = dict(combined)
            else:
                anchors = set(choose_anchors(labels, len(parent_actions), 24 if args.extend_from else 8, inherited))
            histograms[entry["split"]].update(ACTIONS[int(action)] for action in parent_actions)
            if not args.extend_from:
                coverage[entry["split"]]["parent_episodes"] += 1
                coverage[entry["split"]]["parents_without_anchors"] += int(not anchors)
            if version == 3 and not anchors:
                continue
            context = metadata["context"]
            rom = ROOT / context["rom"]["file"]
            if hashlib.sha256(rom.read_bytes()).hexdigest() != context["rom"]["sha256"]:
                raise ValueError("ROM digest mismatch")
            from pyboy import PyBoy
            def factory():
                return PyBoy(str(rom), window="null", sound_emulated=False,
                             color_palette=(0xFFFFFF, 0xAAAAAA, 0x555555, 0x000000))
            game = factory()
            try:
                replay_prefix(game, frames, parent_actions, metadata, 0)
                report["verified_parent_frames"] += 1
                for index in range(len(parent_actions) + 1):
                    if index in anchors:
                        with BytesIO() as buffer:
                            game.save_state(buffer)
                            snapshot = buffer.getvalue()
                        previous = ACTIONS[int(parent_actions[index - 1])]
                        images, actions = witnessed_forks(factory, snapshot, previous, frames[index],
                            parent_actions[index:index + 8], frames[index:index + 9])
                        del snapshot
                        seed = {key: labels[key][index] for key in labels}
                        annotated, unknown = [], 0
                        for branch in images:
                            target, missing = branch_labels(branch, seed, dictionary)
                            annotated.append(target); unknown += missing
                        arrays = {key: np.stack([target[key] for target in annotated]) for key in annotated[0]}
                        raw_bytes = images.nbytes
                        if report["raw_image_bytes"] + raw_bytes > report["raw_image_cap_bytes"]:
                            raise ValueError("raw addon budget exceeded")
                        folder = args.output / entry["split"]
                        folder.mkdir(exist_ok=True)
                        target = folder / f"anchor-{len(report['episodes']):04d}.npz"
                        info = {"source": "capture-intervention", "parent_file": entry["file"], "parent_sha256": entry["sha256"],
                                "split": entry["split"], "anchor": index, "previous_action": previous,
                                "context": context, "rom_sha256": context["rom"]["sha256"], "witness_verified": True,
                                "labels_target_only": True, "snapshot_persisted": False}
                        if version == 3:
                            info["contact_stratum"] = strata[index]
                        np.savez_compressed(target, frames=images, actions=actions, **arrays, metadata=json.dumps(info))
                        size = target.stat().st_size
                        report["compressed_bytes"] += size; report["raw_image_bytes"] += raw_bytes
                        if report["compressed_bytes"] > 128 * 1024 ** 2:
                            raise ValueError("compressed addon budget exceeded")
                        report["witness_frames"] += 9
                        if args.extend_from:
                            report["new_anchors"] += 1
                            report["new_witness_frames"] += 9
                        report["episodes"].append({**info, "file": str(target.relative_to(args.output)),
                            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "unknown_shape_observations": unknown})
                        stats = coverage[entry["split"]]
                        stats["anchors"] += 1; stats["branches"] += 6; stats["unknown_shape_observations"] += unknown
                        stats[f"seed_shape_{int(seed['shape_id'])}"] += 1
                        for name, command in BUTTONS.items():
                            stats[f"action_{name}_steps"] += int((actions == command).sum())
                            position = list(BUTTONS).index(name)
                            ids = arrays["shape_id"][position]
                            known = (ids[:-1] >= 0) & (ids[1:] >= 0)
                            stats[f"{name}_known_shape_changes"] += int((known & (ids[:-1] != ids[1:])).sum())
                    if index < len(parent_actions):
                        actual = step(game, ACTIONS[int(parent_actions[index])])
                        verify_frame(index + 1, frames[index + 1], actual)
                        report["verified_parent_frames"] += 1
            finally:
                game.stop(save=False)
        report["status"] = "complete"
        if args.extend_from:
            # Validate the finished superset including every inherited record and NPZ byte hash.
            (args.output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False))
            verify_extension(args.output, args.extend_from)
    except Exception as error:
        report["status"] = "failed-no-training-use"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        if isinstance(error, ReplayMismatch):
            report["error"]["evidence"] = error.evidence
        report["failed_parent"] = entry["file"] if "entry" in locals() else None
        raise
    finally:
        report["parent_action_histograms"] = {key: dict(value) for key, value in histograms.items()}
        report["coverage"] = {key: dict(value) for key, value in coverage.items()}
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output), "status": report["status"], "anchors": len(report["episodes"]),
                      "compressed_bytes": report["compressed_bytes"]}))


if __name__ == "__main__":
    main()
