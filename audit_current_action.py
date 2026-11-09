"""Audit how the current-object estimate changes under simultaneous button forks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from intervention_training import BUTTONS, tensors
from scene_model import scene_from_weights
from train_scene import composition_targets, label_mask, load_action_interventions, load_scene_data


def overlap_metrics(model, frames, labels, details, counters):
    valid = torch.ones(len(frames), dtype=torch.bool, device=frames.device)
    old_known, old_mask = label_mask(labels, 0, valid)
    new_known, new_mask = label_mask(labels, 1, valid)
    mode, _ = composition_targets(model, frames[:, 0], frames[:, 1], old_mask, new_mask, old_known, new_known)
    copy = old_known & new_known & (mode == 0)
    board = ((frames[:, 0] != model.blank_id) & ~old_mask)[:, :, 2:12].flatten(1)
    masks = model.shape_masks.flatten(0, 1).flatten(1)
    clashes = (board.to(masks.dtype) @ masks.T) > 0
    valid_poses = model.valid_poses.flatten()
    if "overlap_logits" in details:
        scores = details["overlap_logits"].flatten(1)[old_known][:, valid_poses]
        target = clashes[old_known][:, valid_poses]
        counters["overlap/occupied"] += int(target.sum())
        counters["overlap/free"] += int((~target).sum())
        counters["overlap/true_positive"] += int(((scores > 0) & target).sum())
        counters["overlap/true_negative"] += int(((scores <= 0) & ~target).sum())
    joint = details["next"]["joint"].flatten(1)[copy]
    target = clashes[copy]
    counters["copy/count"] += int(copy.sum())
    counters["copy/overlap_argmax"] += int(target.gather(1, joint.argmax(1, keepdim=True)).sum())
    counters["copy/overlap_mass_sum"] += float((joint.exp() * target).sum())


def audit_anchor(model, bundle, state, counters, examples, location, score_overlap=False):
    frames, commands, labels, states = tensors(model, bundle, state)
    if not torch.equal(frames[:, 0], frames[:1, 0].expand_as(frames[:, 0])):
        raise ValueError("fork starts from different observed images")
    details = model.scene_details(frames[:, 0], commands[:, 0], states)
    if score_overlap:
        overlap_metrics(model, frames, labels, details, counters)
    current = details["current"]
    joint = current["joint"].flatten(1)
    visible = current["visible"][:, 0] > 0
    prediction = joint.argmax(1).masked_fill(~visible, -1)
    y, x = labels["cells"][0, 0].amin(0)
    expected = int(labels["shape_id"][0, 0] * 180 + y * 10 + x)
    correct = prediction == expected
    different = prediction != prediction[0]
    distribution_tv = (joint.exp() - joint[0].exp()).abs().sum(1) / 2
    counters["anchors"] += 1
    counters["hard_command_dependent"] += int(different.any())
    counters["visibility_command_dependent"] += int((visible != visible[0]).any())
    counters["mixed_correctness"] += int(correct.any() and not correct.all())
    counters["all_correct"] += int(correct.all())
    counters["tv_max_ge_0.1"] += int(distribution_tv.max() >= .1)
    for i, button in enumerate(BUTTONS):
        counters[f"correct/{button}"] += int(correct[i])
        counters[f"differs_from_noop/{button}"] += int(different[i])
        counters[f"wrong_given_correct_noop/{button}"] += int(correct[0] and not correct[i])
        counters[f"tv_sum/{button}"] += float(distribution_tv[i])
    if different.any() and len(examples) < 12:
        control = model.scene_details(frames[:, 0], commands[:1, 0].expand_as(commands[:, 0]), states)["current"]
        torch.testing.assert_close(control["joint"], control["joint"][:1].expand_as(control["joint"]), rtol=0, atol=0)
        examples.append({"location": location, "expected": expected, "predictions": prediction.tolist(),
                         "tv_from_noop": distribution_tv.tolist()})


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--score-overlap", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _, episodes, annotations, _, _, digest, annotation_digest = load_scene_data(checkpoint)
    model = scene_from_weights(checkpoint["model"]).eval()
    tables, addon_digest = load_action_interventions(checkpoint, model, episodes, annotations, digest, annotation_digest)
    report = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "manifest_sha256": digest, "interventions_sha256": addon_digest,
              "test_used": False, "buttons": BUTTONS, "splits": {}}
    for split, table in tables.items():
        if args.validation_only and split != "validation":
            continue
        counts, examples = Counter(), []
        for episode, (tokens, commands) in enumerate(episodes[split]):
            anchors = sorted(t for i, t in table if i == episode)
            if not anchors:
                continue
            state = None
            for t in range(anchors[-1] + 1):
                if (episode, t) in table:
                    audit_anchor(model, table[episode, t], state, counts, examples, [episode, t], args.score_overlap)
                if t < anchors[-1]:
                    frame = torch.from_numpy(tokens[t:t + 1].copy()).long()
                    state = model.advance(frame, torch.tensor([int(commands[t])]), state)
        if counts["anchors"] != len(table):
            raise ValueError("incomplete fork coverage")
        report["splits"][split] = {"counts": counts, "examples": examples}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
