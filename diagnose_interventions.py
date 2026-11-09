"""No-fit, train/validation intervention diagnosis with identical observed contexts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from intervention_training import BUTTONS, rotation_pair, temporal_weights, tensors
from run import ROOT
from token_model import load_tile_model
from train_scene import load_action_interventions, load_scene_data


def add_metric(counts, key, correct, known):
    """Unknown targets contribute coverage only; never a success or a failure."""
    counts[key + "/total"] += int(bool(known))
    counts[key + "/correct"] += int(bool(known) and bool(correct))
    counts[key + "/unknown"] += int(not bool(known))


def aggregate_step(counts, prefix, instant, t, requested, predicted, target, evidence, visible, mode):
    """A late requested instant must not accidentally count earlier predictions."""
    if t != requested:
        return
    known = int(target) >= 0 and int(evidence) in (1, 2)
    key = f"{prefix}/{instant}"
    add_metric(counts, key + "/shape", int(predicted) // 180 == int(target) // 180, known)
    add_metric(counts, key + "/joint", int(predicted) == int(target), known)
    add_metric(counts, key + "/visible", bool(visible), known)
    add_metric(counts, key + "/joint_visible", int(predicted) == int(target) and bool(visible), known)
    counts[key + f"/predicted_background_mode_{int(mode)}"] += 1


def finish(counts):
    result = dict(counts)
    for key, denominator in counts.items():
        if key.endswith("/total"):
            stem = key.removesuffix("/total")
            result[stem + "/rate"] = counts[stem + "/correct"] / denominator if denominator else None
    return result


@torch.inference_mode()
def audit_bundle(model, bundle, context, split, counts, histogram):
    frames, actions, labels, _ = tensors(model, bundle, context)
    first = temporal_weights(frames).argmax(1)
    different = (frames[:, 1:, :, 2:12] != frames[0:1, 1:, :, 2:12]).flatten(2).any(2)
    for arm, button in enumerate(BUTTONS):
        index = int(first[arm])
        histogram[f"{split}/{button}/first_target_difference/" + (str(index) if bool(different[arm].any()) else "none")] += 1
        evidence = int(labels["evidence"][arm, index + 1])
        known = int(labels["shape_id"][arm, index + 1]) >= 0 and evidence in (1, 2)
        histogram[f"{split}/{button}/first_effect_label_" + ("known" if known else "unknown")] += 1
        histogram[f"{split}/{button}/first_effect_evidence_{evidence}"] += 1
        histogram[f"{split}/{button}/first_effect_partial_or_ambiguous"] += int(evidence >= 3)
        histogram[f"{split}/{button}/first_effect_unknown_shape"] += int(labels["shape_id"][arm, index + 1] < 0)
        histogram[f"{split}/{button}/first_effect_fallback_zero"] += int(not bool(different[arm].any()))
    for protocol in ("teacher", "autonomous"):
        _, _, _, state = tensors(model, bundle, context)
        current = frames[:, 0]
        for t in range(8):
            current = frames[:, t] if protocol == "teacher" else current
            details = model.scene_details(current, actions[:, t], state)
            state = details["state"]
            selected = details["next"]["joint"].flatten(1).argmax(1)
            visible = details["next"]["visible"][:, 0] > 0
            modes = details["mode"].argmax(1)
            origin = labels["cells"][:, t + 1].amin(1)
            shape = labels["shape_id"][:, t + 1]
            wanted = torch.where(shape >= 0, shape * 180 + origin[:, 0] * 10 + origin[:, 1], -1)
            evidence = labels["evidence"][:, t + 1]
            for arm, button in enumerate(BUTTONS):
                prefix = f"{split}/{protocol}/{button}"
                instants = (("first_step", 0), ("first_effect", int(first[arm])), ("last", 7))
                for instant, requested in instants:
                    aggregate_step(counts, prefix, instant, t, requested, selected[arm], wanted[arm],
                                   evidence[arm], visible[arm], modes[arm])
                    if arm not in (4, 5) or t != requested:
                        continue
                    category = rotation_pair(model, frames, labels, t + 1, arm)
                    key = f"{prefix}/{instant}/rotation_{category or 'unqualified'}"
                    if category is None:
                        counts[key + "/count"] += 1
                        continue
                    pair_correct = bool((selected[arm] == wanted[arm]) & (selected[0] == wanted[0]) & visible[arm] & visible[0])
                    pair_shape = bool((selected[arm] // 180 == wanted[arm] // 180) & (selected[0] // 180 == wanted[0] // 180))
                    add_metric(counts, key + "/pair_joint_visible", pair_correct, True)
                    add_metric(counts, key + "/pair_shape", pair_shape, True)
            if protocol == "autonomous":
                current = model.decode_scene(current, details, hard=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interventions", type=Path)
    parser.add_argument("--migrate-interventions-from", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    started = time.monotonic()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    raw, episodes, annotations, _, _, digest, annotation_digest = load_scene_data(checkpoint)
    del raw
    model = load_tile_model(args.checkpoint).eval()
    tables, addon_digest = load_action_interventions(checkpoint, model, episodes, annotations, digest, annotation_digest,
                                                    args.interventions, args.migrate_interventions_from)
    counts, histogram, coverage = Counter(), Counter(), Counter()
    with torch.inference_mode():
        for split in ("train", "validation"):
            for episode, (tokens, actions) in enumerate(episodes[split]):
                anchors = sorted(anchor for local_episode, anchor in tables[split] if local_episode == episode)
                if not anchors:
                    continue
                state = None
                for t in range(anchors[-1] + 1):
                    if (episode, t) in tables[split]:
                        audit_bundle(model, tables[split][(episode, t)], state, split, counts, histogram)
                        coverage[f"{split}/anchors"] += 1
                    if t < anchors[-1]:
                        state = model.advance(torch.from_numpy(tokens[t].copy())[None].long(), torch.tensor([int(actions[t])]), state)
                        coverage[f"{split}/observed_context_steps"] += 1
    expected = {split: len(tables[split]) for split in tables}
    if any(coverage[f"{split}/anchors"] != expected[split] for split in expected):
        raise RuntimeError("incomplete diagnostic coverage")
    report = {"status": "diagnostic-no-fit", "test_used": False, "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "manifest_sha256": digest, "annotations_sha256": annotation_digest, "interventions_sha256": addon_digest,
              "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                          ("diagnose_interventions.py", "intervention_training.py", "token_model.py", "scene_model.py")},
              "scope": "all train+validation anchors; identical observed-prefix context; 8 teacher and 8 autonomous B6 forwards",
              "first_effect_definition": "first target board difference from simultaneous noop target, fallback0 when none; histogram distinguishes fallback",
              "shape_and_joint_rates": "shape ignores position; joint_visible requires exact shape+pose and visibility; unknown labels masked",
              "coverage": dict(coverage), "first_target_difference_histogram": dict(histogram),
              "metrics": finish(counts), "seconds": time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "coverage": dict(coverage), "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
