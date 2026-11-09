"""Bounded validation-only scene decoder audit; no training or model changes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from token_model import load_tile_model
from train_scene import composition_targets, label_mask, load_scene_data, visual_absence


def count_metric(stats, name, correct, mask):
    """Aggregate numerators/denominators, including empty supervised subsets."""
    mask = torch.broadcast_to(mask, correct.shape)
    stats[name + "/numerator"] += int((correct & mask).sum())
    stats[name + "/denominator"] += int(mask.sum())


def confidence(stats, name, probability, target, mask):
    for lo, hi in ((0., .1), (.1, .5), (.5, .9), (.9, 1.01)):
        selected = mask & (probability >= lo) & (probability < hi)
        key = f"{name}/bin-{lo:g}-{hi:g}"
        stats[key + "/count"] += int(selected.sum())
        stats[key + "/probability_sum"] += float(probability[selected].sum())
        stats[key + "/positive_sum"] += int(target[selected].sum())


def assess(model, current, target, truth, details, labels, t, stats):
    soft = model.decode_scene(current, details, hard=False).argmax(1)
    hard = model.decode_scene(current, details, hard=True)
    all_cells = torch.ones_like(current, dtype=torch.bool)
    board = model.board_mask[None].expand_as(current)
    current_matches = (current == truth).flatten(1).all(1)
    valid = torch.ones(len(current), dtype=torch.bool)
    positives, masks, known = {}, {}, {}
    for name, index, observation, causal in (("current", 0, current, current_matches),
                                             ("next", 1, target, valid)):
        positive, mask = label_mask(labels, index, causal)
        negative = visual_absence(observation, labels["evidence"][:, index], causal, model.object_tiles)
        positives[name], masks[name], known[name] = positive, mask, positive | negative
        obj = details[name]
        selected = obj["joint"].flatten(1).argmax(1)
        shape, pose = selected // 180, selected % 180
        origin = labels["cells"][:, index].amin(1)
        wanted_pose = origin[:, 0] * 10 + origin[:, 1]
        wanted_shape = labels["shape_id"][:, index]
        count_metric(stats, name + "/shape_accuracy", shape == wanted_shape, positive)
        count_metric(stats, name + "/pose_accuracy", pose == wanted_pose, positive)
        count_metric(stats, name + "/joint_accuracy", (shape == wanted_shape) & (pose == wanted_pose), positive)
        visibility = obj["visible"][:, 0].sigmoid()
        count_metric(stats, name + "/visibility_recall", visibility > .5, positive)
        count_metric(stats, name + "/visibility_false_positive", visibility > .5, negative)
        confidence(stats, name + "/visibility", visibility, positive, positive | negative)
        confidence(stats, name + "/joint_confidence", obj["joint"].flatten(1).exp().max(1).values,
                   (shape == wanted_shape) & (pose == wanted_pose), positive)
    changed = current != target
    for region, mask in (("all", all_cells), ("board", board), ("hud", ~board)):
        count_metric(stats, region + "/soft_hard_disagreement", soft != hard, mask)
        for name, prediction in (("soft", soft), ("hard", hard)):
            count_metric(stats, region + "/" + name + "/token_error", prediction != target, mask)
            pixel_error = model.codebook[prediction] != model.codebook[target]
            count_metric(stats, region + "/" + name + "/pixel_error", pixel_error, mask[..., None, None])
            count_metric(stats, region + "/" + name + "/addition_accuracy", prediction == target,
                         mask & changed & (target != model.blank_id))
            count_metric(stats, region + "/" + name + "/erasure_accuracy", prediction == target,
                         mask & changed & (target == model.blank_id))
            count_metric(stats, region + "/" + name + "/false_positive", prediction != current, mask & ~changed)
    gate = (details["gate"][:, 0] - model.gate_logit_offset).sigmoid()
    confidence(stats, "gate", gate, changed, all_cells)
    mode = details["mode"].argmax(1)
    before = current.masked_fill(masks["current"], model.blank_id)
    after = target.masked_fill(masks["next"], model.blank_id)
    wanted, mode_known = composition_targets(model, current, target, masks["current"], masks["next"],
                                              known["current"], known["next"])
    count_metric(stats, "mode/accuracy", mode == wanted, mode_known)
    confidence(stats, "mode/confidence", details["mode"].softmax(1).max(1).values, mode == wanted, mode_known)
    for actual in range(3):
        stats[f"mode/selected-{actual}"] += int((mode == actual).sum())
        for expected in range(3):
            stats[f"mode/confusion-target-{expected}-selected-{actual}"] += int((mode_known & (wanted == expected) & (mode == actual)).sum())
    # Separate learned canvas rewrite from object placement, using only known visual labels.
    predicted_mask, _ = model._hard_object(details["current"])
    predicted_background = current.masked_fill(predicted_mask, model.blank_id)
    rewritten = model.combine_logits(predicted_background, details["write"],
                                     details["gate"] - model.gate_logit_offset).argmax(1)
    selected_background = torch.where(mode[:, None, None] == 0, predicted_background,
                                     torch.where(mode[:, None, None] == 1, current, rewritten))
    for name, prediction in (("rewrite", rewritten), ("selected", selected_background)):
        count_metric(stats, "background/" + name + "/error", prediction != after,
                     board & (known["current"] & known["next"])[:, None, None])
    stats["steps"] += len(current)
    stats["causal_current_labels_steps"] += int(current_matches.sum())
    stats[f"offset-{t % 8}/steps"] += len(current)
    count_metric(stats, f"offset-{t % 8}/hard_error", hard != target, all_cells)
    return hard


def finish(stats):
    result = dict(stats)
    for key, value in stats.items():
        if key.endswith("/denominator"):
            stem = key.removesuffix("/denominator")
            result[stem + "/rate"] = stats[stem + "/numerator"] / value if value else None
    return result


@torch.inference_mode()
def audit(model, episodes, annotations, steps):
    result = {}
    model.eval()
    for protocol in ("teacher", "rollout8"):
        stats = defaultdict(float)
        for (frames, actions), annotation in zip(episodes, annotations):
            state, generated = None, None
            for t in range(min(steps, len(actions))):
                truth = torch.from_numpy(frames[t].copy())[None].long()
                # First32 form an identical observed prefix. Every8 afterwards reanchors image;
                # state is carried, matching the trainer (no privileged temporal reset).
                current = truth if protocol == "teacher" or t < 32 or (t - 32) % 8 == 0 else generated
                details = model.scene_details(current, torch.tensor([int(actions[t])]), state)
                state = details["state"]
                if t < 32:
                    continue
                target = torch.from_numpy(frames[t + 1].copy())[None].long()
                labels = {key: torch.from_numpy(np.asarray(annotation[key][t:t + 2]).copy())[None].long()
                          for key in ("cells", "shape_id", "evidence")}
                generated = assess(model, current, target, truth, details, labels, t, stats)
        result[protocol] = finish(stats)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 33 <= args.steps <= 800 or not 1 <= args.threads <= 2:
        parser.error("steps must be33..800 and threads1..2")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint[0], map_location="cpu", weights_only=True)
    raw, episodes, labels, shapes, tiles, digest, annotation_digest = load_scene_data(checkpoint)
    reference_codebook = checkpoint["model"]["codebook"].clone()
    del raw, checkpoint
    episodes["train"].clear(); labels["train"].clear()
    reports = []
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["config"]["manifest_sha256"] != digest:
            raise ValueError("checkpoint dataset differs")
        model = load_tile_model(path)
        if not torch.equal(model.codebook, reference_codebook):
            raise ValueError("token vocabulary differs")
        if not torch.equal(model.object_shapes, shapes) or not torch.equal(model.object_tiles, tiles):
            raise ValueError("object vocabulary differs")
        reports.append({"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "step": checkpoint.get("step"), "metrics": audit(model, episodes["validation"], labels["validation"], args.steps)})
        del model, checkpoint
    report = {"test_used": False, "manifest_sha256": digest, "annotations_sha256": annotation_digest,
              "prefix": 32, "steps_per_episode": args.steps, "episodes": len(episodes["validation"]),
              "protocol": "full observed prefix then teacher or hard-generated8-step segments; carry state; exclude first32",
              "limitations": ["current/mode labels excluded whenever generated screen differs from reference",
                              "background errors require both visual objects known; no internal locked-state labels",
                              "gate calibration uses actual input-target changes, which include rollout correction"],
              "checkpoints": reports}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "checkpoints": len(reports), "test_used": False}))


if __name__ == "__main__":
    main()
