"""Train full-screen dynamics with learned object poses and generic graphic composition."""
import argparse
from collections import Counter
from functools import partial
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from evaluate import comparison_image
from object_annotations import PATTERNS
from model import ACTIONS
from intervention_training import (BUTTONS, branch_objective, branch_validation, finish_validation,
                                   load_interventions, verify_extension, windowanchors)
from run import ROOT
from scene_model import (ObjectSceneDynamics, ExtendedSceneDynamics, RelativeSceneDynamics, FootprintSceneDynamics,
                         PoseRerankSceneDynamics, pose_rerank_scene_weights,
                         action_invariant_current_scene_weights,
                         LearnedOverlapSceneDynamics, learned_overlap_scene_weights,
                         SpawnPreviewSceneDynamics, spawn_preview_scene_weights,
                         expand_scene_weights, expand_scene_state, footprint_scene_weights, relative_scene_weights,
                         scene_from_weights, upgrade_scene_weights)
from token_model import encode_frames, load_tile_model
from train import load_data
from train_stateful import StreamWindows
from train_tokens import measure


def append_scene_optimizer(old, model):
    result = copy.deepcopy(old)
    if len(result["param_groups"]) != 1 or len(result["param_groups"][0]["params"]) != model.base_parameter_count:
        raise ValueError("scene upgrade requires the existing single core optimizer group")
    group = copy.deepcopy(result["param_groups"][0])
    first = max(group["params"]) + 1
    extra = len(list(model.parameters())) - model.base_parameter_count
    group["params"] = list(range(first, first + extra)); group["lr"] = 3e-3
    result["param_groups"].append(group)
    return result


def load_scene_data(checkpoint):
    raw, digest, manifest = load_data(ROOT / "data")
    directory = ROOT / "data/object-annotations-v2"
    annotation_bytes = (directory / "object-audit.json").read_bytes()
    report = json.loads(annotation_bytes)
    if report["test_used"] or report["manifest_sha256"] != digest or checkpoint["config"]["manifest_sha256"] != digest:
        raise ValueError("annotation/checkpoint/dataset lineage mismatch")
    dictionary = sorted(report["dictionary"], key=lambda x: x["id"])
    if [x["id"] for x in dictionary] != list(range(len(dictionary))):
        raise ValueError("object dictionary ids must be contiguous")
    book = checkpoint["model"]["codebook"].numpy()
    style_ids = []
    for item in dictionary:
        matches = np.flatnonzero(np.all(book == PATTERNS[item["texture"]], axis=(1, 2)))
        if len(matches) != 1:
            raise ValueError("object appearance not present in the training codebook")
        style_ids.append(int(matches[0]))
    shapes = torch.tensor([x["cells"] for x in dictionary], dtype=torch.int64)
    tiles = torch.tensor(style_ids, dtype=torch.int64)
    records = {x["file"]: x for x in report["episodes"]}
    episodes, labels = {}, {}
    raw["test"].clear()
    for split in ("train", "validation"):
        entries = [e for e in manifest["episodes"] if e["split"] == split]
        episodes[split], labels[split] = [], []
        for entry, (frames, actions) in zip(entries, raw[split]):
            record = records[entry["file"]]
            path = (directory / record["labels_file"]).resolve()
            if record["split"] != split or not path.is_relative_to(directory.resolve()):
                raise ValueError("annotation split or path mismatch")
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"]))
                if metadata["source_sha256"] != entry["sha256"]:
                    raise ValueError("annotation source hash mismatch")
                annotation = {k: data[k].copy() for k in ("cells", "shape_id", "evidence", "track")}
            if (annotation["cells"].shape != (len(frames), 4, 2)
                    or annotation["shape_id"].shape != (len(frames),)
                    or np.any(annotation["shape_id"] >= len(shapes))):
                raise ValueError("invalid object labels")
            codes, missing = encode_frames(frames, book)
            if missing:
                raise ValueError("unseen visual tile patterns")
            episodes[split].append((codes.astype(np.min_scalar_type(len(book) - 1)), actions))
            labels[split].append(annotation)
        if split == "train":
            raw[split].clear()
    return raw, episodes, labels, shapes, tiles, digest, hashlib.sha256(annotation_bytes).hexdigest()


def window_labels(cursors, valid, annotations):
    results = {k: [] for k in ("cells", "shape_id", "evidence", "track")}
    length = valid.shape[1]
    for slot, (episode, end) in enumerate(cursors):
        used = int(valid[slot].sum()); start = end - used
        for name in results:
            values = annotations[episode][name][start:end + 1]
            fill = np.zeros((length - used, *values.shape[1:]), dtype=values.dtype)
            if name != "evidence":
                fill.fill(-1)
            results[name].append(np.concatenate((values, fill)))
    return {k: torch.from_numpy(np.stack(v)).long() for k, v in results.items()}


def label_mask(labels, t, valid):
    positive = (labels["shape_id"][:, t] >= 0) & (labels["evidence"][:, t] > 0) & (labels["evidence"][:, t] < 3) & valid
    mask = torch.zeros(len(valid), 18, 20, dtype=torch.bool, device=valid.device)
    rows = labels["cells"][:, t, :, 0].clamp(0, 17)
    cols = (labels["cells"][:, t, :, 1] + 2).clamp(0, 19)
    mask[torch.arange(len(valid), device=valid.device)[:, None], rows, cols] = positive[:, None]
    return positive, mask


def visual_absence(tokens, evidence, valid, object_tiles):
    represented = (tokens[:, :, 2:12, None] == object_tiles[None, None, None, :]).any(3).flatten(1).any(1)
    return (evidence == 3) & valid & ~represented


def composition_targets(model, current, target, current_mask, next_mask, current_known, next_known):
    before = current.masked_fill(current_mask, model.blank_id)[:, :, 2:12]
    after = target.masked_fill(next_mask, model.blank_id)[:, :, 2:12]
    same = (before == after).flatten(1).all(1)
    composed = current_mask.flatten(1).any(1) & (current[:, :, 2:12] == after).flatten(1).all(1)
    mode = torch.where(same, 0, torch.where(composed, 1, 2))

    def overlay_possible(background):
        changed = background != target[:, :, 2:12]
        colors = target[:, :, 2:12, None] == model.object_tiles[None, None, None, :]
        one_color = (~changed[..., None] | colors).flatten(1, 2).all(1).any(1)
        return (changed.flatten(1).sum(1) <= 4) & one_color

    # A single graphic object cannot explain these changes, even when its pose is unknown.
    rewrite_required = current_known & ~overlay_possible(before) & ~overlay_possible(current[:, :, 2:12])
    mode = torch.where(rewrite_required, 2, mode)
    return mode, (current_known & next_known) | rewrite_required


def composition_coverage(model, episodes, annotations):
    counts = [0, 0, 0]
    transitions = 0
    for (codes, actions), annotation in zip(episodes, annotations):
        for start in range(0, len(actions), 128):
            end = min(start + 128, len(actions))
            frames = torch.from_numpy(codes[start:end + 1].copy()).long()
            labels = {key: torch.from_numpy(np.stack((annotation[key][start:end], annotation[key][start + 1:end + 1]), 1)).long()
                      for key in ("cells", "shape_id", "evidence")}
            valid = torch.ones(end - start, dtype=torch.bool)
            current_positive, current_mask = label_mask(labels, 0, valid)
            next_positive, next_mask = label_mask(labels, 1, valid)
            current_known = current_positive | visual_absence(frames[:-1], labels["evidence"][:, 0], valid, model.object_tiles)
            next_known = next_positive | visual_absence(frames[1:], labels["evidence"][:, 1], valid, model.object_tiles)
            mode, known = composition_targets(model, frames[:-1], frames[1:], current_mask, next_mask, current_known, next_known)
            for kind in range(3):
                counts[kind] += int((known & (mode == kind)).sum())
            transitions += len(valid)
    return {"copy": counts[0], "compose": counts[1], "rewrite": counts[2], "masked": transitions - sum(counts)}


def mode_class_weights(coverage):
    counts = [coverage[name] for name in ("copy", "compose", "rewrite")]
    if min(counts) <= 0:
        raise ValueError("all three mode classes must occur in the training corpus")
    total = sum(counts)
    return [total / (3 * count) for count in counts]


def training_mode_balance(model, episodes, annotations, interventions):
    parent = composition_coverage(model, episodes["train"], annotations["train"])
    branches = Counter()
    for bundle in interventions["train"].values():
        branch_episodes = [(bundle["tokens"][arm], bundle["actions"][arm]) for arm in range(6)]
        branch_annotations = [{key: values[arm] for key, values in bundle["labels"].items()} for arm in range(6)]
        branches.update(composition_coverage(model, branch_episodes, branch_annotations))
    coverage = {"parent": parent, "interventions": dict(branches)}
    return coverage, {name: mode_class_weights(counts) for name, counts in coverage.items()}


def weighted_mode_loss(logits, targets, known, mode_weights=None):
    if not torch.any(known):
        return logits[known].sum() * 0
    weights = None if mode_weights is None else torch.as_tensor(mode_weights, device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits[known], targets[known], weight=weights, reduction="none").mean()


def relative_targets(model, current, target, labels, t, current_positive, next_positive, current_mask, next_mask):
    both = current_positive & next_positive
    old_kind, next_kind = labels["shape_id"][:, t].clamp_min(0), labels["shape_id"][:, t + 1].clamp_min(0)
    same_shape = old_kind == next_kind
    same_style = model.object_tiles[old_kind] == model.object_tiles[next_kind]
    origins = labels["cells"][:, t + 1].amin(1) - labels["cells"][:, t].amin(1)
    local = (origins.abs() <= 3).all(1)
    before = current.masked_fill(current_mask, model.blank_id)[:, :, 2:12]
    after = target.masked_fill(next_mask, model.blank_id)[:, :, 2:12]
    same_background = (before == after).flatten(1).all(1)
    positive = both & same_shape & same_style & local & same_background
    negative = both & (~same_shape | ~same_style | ~local)
    delta = (origins[:, 0] + 3) * 7 + origins[:, 1] + 3
    return positive, negative, delta


def relative_aux_loss(model, current, target, details, labels, t, current_positive, next_positive, current_mask, next_mask):
    transport = details.get("transport")
    if transport is None: return details["write"].sum() * 0
    positive, negative, delta = relative_targets(model, current, target, labels, t, current_positive, next_positive, current_mask, next_mask)
    loss = transport["delta_logits"][positive].sum() * 0 + transport["gate_logits"][positive | negative].sum() * 0
    if positive.any(): loss = loss + .1 * F.cross_entropy(transport["delta_logits"][positive], delta[positive])
    known = positive | negative
    if known.any(): loss = loss + .1 * F.binary_cross_entropy_with_logits(transport["gate_logits"][known, 0], positive[known].float())
    return loss


def relative_supervision_coverage(model, episodes, annotations, interventions):
    result = {}
    for corpus in ("parent", "interventions"):
        records = zip(episodes["train"], annotations["train"], strict=True) if corpus == "parent" else (
            ((bundle["tokens"][arm], bundle["actions"][arm]), {key: value[arm] for key, value in bundle["labels"].items()})
            for bundle in interventions["train"].values() for arm in range(6))
        counts = Counter()
        for (codes, commands), annotation in records:
            for start in range(0, len(commands), 128):
                end = min(start + 128, len(commands))
                current, target = torch.from_numpy(codes[start:end].copy()).long(), torch.from_numpy(codes[start + 1:end + 1].copy()).long()
                labels = {key: torch.from_numpy(np.stack((annotation[key][start:end], annotation[key][start + 1:end + 1]), 1)).long() for key in ("cells", "shape_id", "evidence")}
                valid = torch.ones(len(current), dtype=torch.bool)
                cp, cm = label_mask(labels, 0, valid); np_, nm = label_mask(labels, 1, valid)
                positive, negative, _ = relative_targets(model, current, target, labels, 0, cp, np_, cm, nm)
                counts.update(total=len(current), gate_positive=int(positive.sum()), gate_negative=int(negative.sum()),
                              gate_masked=int((~(positive | negative)).sum()), delta_known=int(positive.sum()))
        result[corpus] = dict(counts)
    return result


def copy_overlap_loss(model, current, next_joint, current_mask, next_mask, proven_copy):
    occupied = (current != model.blank_id) & ~current_mask
    board = occupied[:, :, 2:12]
    eligible = proven_copy & ~(occupied & next_mask).flatten(1).any(1) & board.flatten(1).any(1)
    joint = next_joint[eligible].exp().flatten(1)
    masks = model.shape_masks.flatten(0, 1).flatten(1)
    overlap = (board[eligible].flatten(1).to(masks.dtype) @ masks.T) > 0
    return (joint * overlap).sum() / eligible.sum().clamp_min(1)


def descent_focal_loss(next_joint, cells, shape_ids, eligible, known):
    origin = cells[eligible].amin(1)
    target = shape_ids[eligible] * 180 + origin[:, 0] * 10 + origin[:, 1]
    log_probability = next_joint[eligible].flatten(1).gather(1, target[:, None])[:, 0]
    return (-log_probability * (-log_probability.expm1()).square()).sum() / known.sum().clamp_min(1)


def descent_ranking_loss(model, next_joint, cells, shape_ids, proven_descent):
    origins = cells.amin(2)
    same_shape = shape_ids[:, 0] == shape_ids[:, 1]
    same_column = origins[:, 0, 1] == origins[:, 1, 1]
    eligible = proven_descent & same_shape & same_column
    joint = next_joint[eligible].flatten(1)
    wanted_shape = shape_ids[eligible, 1]
    y, x = origins[eligible, 1].unbind(1)
    wanted = wanted_shape * 180 + y * 10 + x
    shapes = torch.arange(model.valid_poses.shape[0], device=joint.device)
    columns = torch.arange(180, device=joint.device) % 10
    alternatives = (shapes[None, :, None] != wanted_shape[:, None, None]) | (columns[None, None, :] != x[:, None, None])
    alternatives = (alternatives & model.valid_poses).flatten(1)
    comparable = alternatives.any(1) & model.valid_poses.flatten()[wanted]
    strongest = joint[comparable].masked_fill(~alternatives[comparable], -torch.inf).amax(1)
    truth = joint[comparable].gather(1, wanted[comparable, None])[:, 0]
    return F.relu(.5 + strongest - truth).sum() / comparable.sum().clamp_min(1)


def learned_overlap_targets(model, current, current_mask):
    occupied = (current != model.blank_id) & ~current_mask
    board = occupied[:, :, 2:12].flatten(1)
    masks = model.shape_masks.flatten(0, 1).flatten(1)
    return (board.to(masks.dtype) @ masks.T > 0).reshape(len(board), *model.valid_poses.shape)


def learned_overlap_loss(model, logits, current, current_mask, current_positive):
    if not current_positive.any():
        return logits[current_positive].sum()
    logits = logits[current_positive].flatten(1)[:, model.valid_poses.flatten()]
    targets = learned_overlap_targets(model, current[current_positive], current_mask[current_positive])
    targets = targets.flatten(1)[:, model.valid_poses.flatten()]
    error = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
    positive_count = targets.sum(1)
    negative_count = (~targets).sum(1)
    positive_error = (error * targets).sum(1) / positive_count.clamp_min(1)
    negative_error = (error * ~targets).sum(1) / negative_count.clamp_min(1)
    classes = (positive_count > 0).long() + (negative_count > 0).long()
    return ((positive_error + negative_error) / classes.clamp_min(1)).mean()


def overlap_ranking_loss(model, current, next_joint, current_mask, cells, shape_ids, proven_copy):
    joint = next_joint[proven_copy].flatten(1)
    clashes = learned_overlap_targets(model, current[proven_copy], current_mask[proven_copy]).flatten(1)
    clashes = clashes & model.valid_poses.flatten()
    y, x = cells[proven_copy].amin(1).unbind(1)
    wanted = (shape_ids[proven_copy] * 180 + y * 10 + x)[:, None]
    comparable = (clashes.any(1) & ~clashes.gather(1, wanted)[:, 0]
                  & model.valid_poses.flatten()[wanted[:, 0]])
    truth = joint.gather(1, wanted)[comparable, 0]
    strongest = joint[comparable].masked_fill(~clashes[comparable], -torch.inf).amax(1)
    return F.relu(.5 + strongest - truth).sum() / comparable.sum().clamp_min(1)


def spawn_preview_targets(model, current, target, current_mask, current_positive, current_shapes):
    before, after, old = current[:, :, 2:12], target[:, :, 2:12], current_mask[:, :, 2:12]
    fixed = (before != model.blank_id) & ~old
    supported = old[:, -1].any(1) | (old[:, :-1] & fixed[:, 1:]).flatten(1).any(1)
    old_texture = model.object_tiles[current_shapes.clamp_min(0)]
    old_exact = (old.sum((1, 2)) == 4) & (~old | (before == old_texture[:, None, None])).flatten(1).all(1)
    added = (before == model.blank_id) & (after != model.blank_id)
    eligible = (current_positive & old_exact & supported & (added.sum((1, 2)) == 4)
                & ~added[:, 4:].flatten(1).any(1) & ((before == after) | added).flatten(1).all(1))
    rows = eligible.nonzero(as_tuple=True)[0]
    added = added[rows]
    if rows.numel() == 0:
        return rows, rows
    textures = after[rows].masked_fill(~added, -1).flatten(1).amax(1)
    same_texture = (~added | (after[rows] == textures[:, None, None])).flatten(1).all(1)
    masks = model.shape_masks.flatten(2).flatten(0, 1)
    matching = (added.flatten(1).to(torch.float32) @ masks.to(torch.float32).T == 4).reshape(-1, *model.valid_poses.shape)
    matching = matching & model.valid_poses & (textures[:, None, None] == model.object_tiles[None, :, None])
    matching = matching.flatten(1)
    unique = same_texture & (matching.sum(1) == 1)
    return rows[unique], matching[unique].long().argmax(1)


def scene_loss(model, current, target, details, labels, t, valid, current_matches, mode_weights=None,
               copy_overlap_weight=0., descent_focal_weight=0., actions=None, overlap_ranking_weight=0.,
               descent_ranking_weight=0.):
    current_positive, current_mask = label_mask(labels, t, valid & current_matches)
    next_positive, next_mask = label_mask(labels, t + 1, valid)
    log_probs = model.decode_scene(current, details, hard=False)
    weights = (1 + 31 * ((current != target) | current_mask | next_mask)).float() * valid[:, None, None]
    pixel = (F.nll_loss(log_probs, target, reduction="none") * weights).sum() / weights.sum().clamp_min(1)
    auxiliary = pixel * 0
    negatives = {}
    for name, index, positive, causal, observation in (("current", t, current_positive, valid & current_matches, current),
                                                       ("next", t + 1, next_positive, valid, target)):
        origin = labels["cells"][:, index].amin(1).clamp_min(0)
        wanted = labels["shape_id"][:, index].clamp_min(0) * 180 + origin[:, 0] * 10 + origin[:, 1]
        # Mask before NLL: an excluded geometric -inf target must never produce inf * 0.
        if torch.any(positive):
            auxiliary = auxiliary + 0.1 * F.nll_loss(details[name]["joint"].flatten(1)[positive], wanted[positive])
        negatives[name] = visual_absence(observation, labels["evidence"][:, index], causal, model.object_tiles)
        known = positive | negatives[name]
        visibility = F.binary_cross_entropy_with_logits(details[name]["visible"][:, 0], positive.float(), reduction="none")
        auxiliary = auxiliary + 0.1 * (visibility * known).sum() / known.sum().clamp_min(1)
    current_known = current_positive | negatives["current"]
    next_known = next_positive | negatives["next"]
    mode, known = composition_targets(model, current, target, current_mask, next_mask, current_known, next_known)
    auxiliary = auxiliary + 0.1 * weighted_mode_loss(details["mode"], mode, known, mode_weights)
    if "overlap_logits" in details:
        auxiliary = auxiliary + 0.1 * learned_overlap_loss(
            model, details["overlap_logits"], current, current_mask, current_positive)
    if "preview_joint_offset" in details:
        rows, wanted = spawn_preview_targets(
            model, current, target, current_mask, current_positive, labels["shape_id"][:, t])
        if rows.numel():
            visible = details["next"]["visible"][rows]
            auxiliary = auxiliary + 0.1 * (
                F.nll_loss(details["next"]["joint"][rows].flatten(1), wanted)
                + F.cross_entropy(details["mode"][rows], torch.ones_like(wanted))
                + F.binary_cross_entropy_with_logits(visible, torch.ones_like(visible)))
    if copy_overlap_weight:
        auxiliary = auxiliary + copy_overlap_weight * copy_overlap_loss(
            model, current, details["next"]["joint"], current_mask, next_mask,
            current_positive & next_positive & known & (mode == 0))
    if overlap_ranking_weight:
        auxiliary = auxiliary + overlap_ranking_weight * overlap_ranking_loss(
            model, current, details["next"]["joint"], current_mask, labels["cells"][:, t + 1],
            labels["shape_id"][:, t + 1], current_positive & next_positive & known & (mode == 0))
    if descent_focal_weight:
        eligible = current_positive & next_positive & (mode == 0) & (actions == ACTIONS.index("down"))
        auxiliary = auxiliary + descent_focal_weight * descent_focal_loss(
            details["next"]["joint"], labels["cells"][:, t + 1], labels["shape_id"][:, t + 1], eligible, next_positive)
    if descent_ranking_weight and actions is not None:
        eligible = current_positive & next_positive & known & (mode == 0) & (actions == ACTIONS.index("down"))
        auxiliary = auxiliary + descent_ranking_weight * descent_ranking_loss(
            model, details["next"]["joint"], labels["cells"][:, t:t + 2], labels["shape_id"][:, t:t + 2], eligible)
    auxiliary = auxiliary + relative_aux_loss(model, current, target, details, labels, t, current_positive, next_positive, current_mask, next_mask)
    return pixel + auxiliary


def critical_event_counts(model, current, target, details, labels):
    valid = torch.ones(len(current), dtype=torch.bool)
    current_positive, current_mask = label_mask(labels, 0, valid)
    next_positive, next_mask = label_mask(labels, 1, valid)
    current_known = current_positive | visual_absence(current, labels["evidence"][:, 0], valid, model.object_tiles)
    next_known = next_positive | visual_absence(target, labels["evidence"][:, 1], valid, model.object_tiles)
    wanted_mode, known = composition_targets(model, current, target, current_mask, next_mask, current_known, next_known)
    predicted_mode = details["mode"].argmax(1)
    confusion = [[int((known & (wanted_mode == i) & (predicted_mode == j)).sum()) for j in range(3)] for i in range(3)]
    origin = labels["cells"][:, 1].amin(1)
    wanted_joint = labels["shape_id"][:, 1] * 180 + origin[:, 0] * 10 + origin[:, 1]
    predicted_joint = details["next"]["joint"].flatten(1).argmax(1)
    changed = (labels["shape_id"][:, 0] != labels["shape_id"][:, 1]) | (current_mask != next_mask).flatten(1).any(1)
    counts = {"mode_confusion": confusion}
    for name, subset in (("motion", changed), ("still", ~changed)):
        mask = current_positive & next_positive & subset
        counts[name] = int(mask.sum())
        counts[name + "_correct"] = int((mask & (predicted_joint == wanted_joint)).sum())
    return counts


def critical_event_rates(counts):
    matrix = counts["mode_confusion"]
    return {"mode_recall": {name: matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else None
                            for i, name in enumerate(("copy", "compose", "rewrite"))},
            "next_joint_accuracy": {name: counts[name + "_correct"] / counts[name] if counts[name] else None
                                    for name in ("motion", "still")}}


def rendered_teacher(model, episodes, annotations, interventions=None):
    counts = {k: 0 for k in ("add", "add_correct", "erase", "erase_correct", "tp", "fp", "unchanged", "changed")}
    critical = {"mode_confusion": [[0] * 3 for _ in range(3)], "motion": 0, "motion_correct": 0, "still": 0, "still_correct": 0}
    intervention_counts = Counter()
    transport_counts = Counter()
    blank = (model.codebook.flatten(1) == 3).all(1)
    model.eval()
    with torch.inference_mode():
        for episode, ((tokens, actions), annotation) in enumerate(zip(episodes, annotations, strict=True)):
            state = None
            for t in range(len(actions)):
                current = torch.from_numpy(tokens[t].copy())[None].long()
                action = torch.tensor([int(actions[t])])
                if interventions is not None and (episode, t) in interventions:
                    intervention_counts.update(branch_validation(model, interventions[(episode, t)], state))
                details = model.scene_details(current, action, state)
                prediction = model.decode_scene(current, details, hard=True)
                state = details["state"]
                if t < 32:
                    continue
                if "transport" in details:
                    info = details["transport"]
                    transport_counts.update(samples=len(current), alpha_sum=float(info["alpha"].sum()),
                                            retained_mass_sum=float(info["retained_mass"].sum()),
                                            identity_delta_sum=float(info["identity_delta"].sum()),
                                            numerical_fallbacks=int(info["numerical_fallback"].sum()))
                target = torch.from_numpy(tokens[t + 1].copy())[None].long()
                labels = {key: torch.from_numpy(annotation[key][t:t + 2].copy())[None].long()
                          for key in ("cells", "shape_id", "evidence")}
                events = critical_event_counts(model, current, target, details, labels)
                for i in range(3):
                    for j in range(3):
                        critical["mode_confusion"][i][j] += events["mode_confusion"][i][j]
                for key in ("motion", "motion_correct", "still", "still_correct"):
                    critical[key] += events[key]
                changed, predicted_change = current != target, current != prediction
                correct = prediction == target
                add, erase = changed & ~blank[target], changed & blank[target]
                for key, mask in (("add", add), ("add_correct", add & correct), ("erase", erase), ("erase_correct", erase & correct),
                                  ("tp", changed & predicted_change), ("fp", ~changed & predicted_change),
                                  ("unchanged", ~changed), ("changed", changed)):
                    counts[key] += int(mask.sum())
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    return {"metric": "rendered-scene-v5-action-interventions" if interventions is not None else "rendered-scene-v4-critical-events",
            **({"transport_descriptive": dict(transport_counts)} if transport_counts else {}),
            **({"interventions": finish_validation(intervention_counts)} if interventions is not None else {}),
            "counts": counts, "critical_counts": critical,
            **critical_event_rates(critical),
            "addition_accuracy": ratio(counts["add_correct"], counts["add"]),
            "erasure_accuracy": ratio(counts["erase_correct"], counts["erase"]),
            "event_precision": ratio(counts["tp"], counts["tp"] + counts["fp"]),
            "event_recall": ratio(counts["tp"], counts["changed"]),
            "event_fpr": ratio(counts["fp"], counts["unchanged"])}


def score(teacher, rollout):
    critical = teacher.get("critical_counts", {})
    matrix = critical.get("mode_confusion", [])
    if (len(matrix) != 3 or any(len(row) != 3 or sum(row) <= 0 for row in matrix)
            or any(critical.get(name, 0) <= 0 for name in ("motion", "still"))):
        return math.inf
    modes = [teacher.get("mode_recall", {}).get(name) for name in ("copy", "compose", "rewrite")]
    joints = [teacher.get("next_joint_accuracy", {}).get(name) for name in ("motion", "still")]
    values = [teacher.get("addition_accuracy"), teacher.get("erasure_accuracy"), teacher.get("event_fpr"), rollout.get("tile_error"), *modes, *joints]
    if any(value is None or not math.isfinite(value) for value in values):
        return math.inf
    value = (2 - values[0] - values[1] + 10 * values[2] + values[3]
             + 1 - sum(modes) / 3 + 1 - sum(joints) / 2)
    if "interventions" in teacher:
        metrics = teacher["interventions"]
        counts, rates = metrics.get("counts", {}), metrics.get("rates", {})
        denominators = [f"{button}/joint_total" for button in BUTTONS]
        denominators += [f"{button}/{kind}_total" for button in ("a", "b") for kind in ("effect", "noeffect")]
        denominators += [f"{button}/board_pixels" for button in BUTTONS]
        required = intervention_requirements(teacher)
        pixel_rates = [rates.get(f"{button}/board_pixel_error") for button in BUTTONS]
        if (any(counts.get(key, 0) <= 0 for key in denominators)
                or any(rate is None or not math.isfinite(rate) for rate in required + pixel_rates)):
            return math.inf
        value += 1 - sum(required[:6]) / 6 + 1 - sum(required[6:]) / 4 + sum(pixel_rates) / 6
    return value


def intervention_requirements(teacher):
    rates = teacher.get("interventions", {}).get("rates", {})
    return ([rates.get(f"{button}/joint_accuracy") for button in BUTTONS]
            + [rates.get(f"{button}/rotation_{kind}_rendered_accuracy") for button in ("a", "b") for kind in ("effect", "noeffect")])


def rollout_horizon(previous, validation_score=None, previous_score=None, teacher=None):
    if previous not in (1, 2, 4, 8):
        raise ValueError("rollout horizon must be 1, 2, 4 or 8")
    if validation_score is None and previous_score is None:
        return previous
    if validation_score is None or previous_score is None:
        raise ValueError("both validation scores are required")
    teacher = teacher or {}
    required = [teacher.get("addition_accuracy"), teacher.get("erasure_accuracy")]
    required += [teacher.get("mode_recall", {}).get(name) for name in ("copy", "compose", "rewrite")]
    required += [teacher.get("next_joint_accuracy", {}).get(name) for name in ("motion", "still")]
    if "interventions" in teacher:
        required += intervention_requirements(teacher)
    ready = all(value is not None and math.isfinite(value) and value >= .95 for value in required)
    return min(8, previous * 2) if ready and math.isfinite(validation_score) and validation_score < previous_score else previous


def training_input(truth_current, generated, t, horizon):
    current = truth_current if generated is None or t % horizon == 0 else generated
    return current, (current == truth_current).flatten(1).all(1)


def recovery_mask(generated, observed, predicted_joint, labels, t, predecessor_matches, valid):
    """Keep one-step errors on the board when the active pose is still correct."""
    proven, _ = label_mask(labels, t, valid & predecessor_matches)
    origin = labels["cells"][:, t].amin(1)
    wanted = labels["shape_id"][:, t] * 180 + origin[:, 0] * 10 + origin[:, 1]
    difference = (generated[:, :, 2:12] != observed[:, :, 2:12]).flatten(1).sum(1)
    return proven & (predicted_joint == wanted) & (difference > 0) & (difference <= 8)


def selection_baseline(resume, selection, current_model, migration=None):
    """Copy only compatible selection weights; never its optimizer, streams or RNG."""
    for checkpoint in (resume, selection):
        protocol = checkpoint["config"].get("protocol")
        if protocol not in ("object-scene-v4-critical-events", "object-scene-v5-action-interventions"):
            raise ValueError("independent selection requires compatible v4/v5 scene checkpoints")
        if protocol == "object-scene-v5-action-interventions" and not checkpoint["config"].get("interventions_sha256"):
            raise ValueError("v5 selection checkpoint lacks intervention provenance")
    hashes = [checkpoint["config"].get("interventions_sha256") for checkpoint in (resume, selection)]
    allowed = {migration["from_sha256"], migration["to_sha256"]} if migration else set()
    if all(hashes) and hashes[0] != hashes[1] and not set(hashes) <= allowed:
        raise ValueError("selection intervention lineage mismatch")
    for key in ("manifest_sha256", "annotations_sha256"):
        if not resume["config"].get(key) or resume["config"][key] != selection["config"].get(key):
            raise ValueError(f"selection checkpoint lineage mismatch: {key}")
    reference = current_model.state_dict()
    for key in ("codebook", "object_shapes", "object_tiles"):
        value = selection["model"].get(key)
        if value is None or value.dtype != reference[key].dtype or not torch.equal(value, reference[key]):
            raise ValueError(f"selection checkpoint vocabulary mismatch: {key}")
    # Keep the selected checkpoint's real family, including a legacy baseline during expansion.
    baseline = scene_from_weights(selection["model"])
    return baseline


def intervention_directory(checkpoint, directory=None):
    value = directory if directory is not None else checkpoint["config"].get("interventions_path", "data/action-interventions-v1")
    path = Path(value)
    # Existing V1 checkpoints recorded an absolute path on their original worker.
    if directory is None and path.is_absolute() and not path.is_relative_to((ROOT / "data").resolve()) and path.name == "action-interventions-v1":
        path = Path("data/action-interventions-v1")
    path = (ROOT / path).resolve()
    if not path.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("intervention directory must remain inside ROOT/data")
    return path


def load_action_interventions(checkpoint, model, episodes, annotations, digest, annotation_digest,
                              directory=None, migrate_from=None):
    directory = intervention_directory(checkpoint, directory)
    base = intervention_directory(checkpoint, migrate_from) if migrate_from is not None else None
    proof = verify_extension(directory, base) if base is not None else None
    parent = json.loads((ROOT / "data/manifest.json").read_bytes())
    tables, addon_digest = load_interventions(directory, model, parent, episodes, annotations, digest, annotation_digest, base)
    recorded = checkpoint["config"].get("interventions_sha256")
    authorized = proof is not None and recorded == proof["from_sha256"] and addon_digest == proof["to_sha256"]
    if ((recorded is not None and recorded != addon_digest and not authorized)
            or (checkpoint["config"].get("protocol") == "object-scene-v5-action-interventions" and not recorded)):
        raise ValueError("resume intervention lineage mismatch")
    return tables, addon_digest


def resume_exposure(checkpoint, table, manifest_sha, migration=None):
    step = checkpoint["step"]
    if type(step) is not int or step < 0:
        raise ValueError("invalid training exposure checkpoint step")
    saved = checkpoint.get("training_exposure")
    reason = "legacy-checkpoint-no-history"
    if saved is not None:
        integers = ("version", "tracking_since_step", "last_completed_step", "bundles")
        if (not isinstance(saved, dict) or any(type(saved.get(key)) is not int for key in integers)
                or saved["version"] != 1 or saved["last_completed_step"] != step
                or not 1 <= saved["tracking_since_step"] <= step + 1 or saved["bundles"] < 0
                or not isinstance(saved.get("manifest_sha256"), str) or not saved["manifest_sha256"]
                or not isinstance(saved.get("keys"), list)):
            raise ValueError("invalid training exposure schema or step sequence")
        keys = saved["keys"]
        if (any(not isinstance(key, list) or len(key) != 2 or any(type(value) is not int for value in key) for key in keys)
                or keys != [list(key) for key in sorted({tuple(key) for key in keys})]
                or any(tuple(key) not in table for key in keys) or saved["bundles"] < len(keys)
                or bool(keys) != bool(saved["bundles"]) or (saved["tracking_since_step"] > step and saved["bundles"] != 0)):
            raise ValueError("invalid training exposure keys/count")
        if saved["manifest_sha256"] == manifest_sha:
            return copy.deepcopy(saved)
        if not migration or migration["from_sha256"] != saved["manifest_sha256"] or migration["to_sha256"] != manifest_sha:
            raise ValueError("training exposure corpus changed without verified migration")
        reason = "verified-corpus-migration-reset"
    return {"version": 1, "manifest_sha256": manifest_sha, "tracking_since_step": step + 1,
            "last_completed_step": step, "bundles": 0, "keys": [], "initialization": reason}


def exposure_key(cursors, valid, slot, t):
    if not 0 <= t < valid.shape[1] or not bool(valid[slot, t]):
        raise ValueError("cannot count a padded intervention")
    episode, end = cursors[slot]
    return (int(episode), int(end - int(valid[slot].sum()) + t))


def optimizer_step_with_exposure(optimizer, exposure, step, applied_keys, table):
    if (step != exposure["last_completed_step"] + 1
            or any(not isinstance(key, tuple) or len(key) != 2 or any(type(value) is not int for value in key)
                   or key not in table for key in applied_keys)):
        raise ValueError("training exposure update sequence/key mismatch")
    updated = {**exposure, "last_completed_step": step, "bundles": exposure["bundles"] + len(applied_keys),
               "keys": [list(key) for key in sorted({tuple(key) for key in exposure["keys"]} | set(applied_keys))]}
    optimizer.step()  # Failed optimizer steps never publish the prepared exposure update.
    return updated


def exposure_summary(exposure, table):
    return {"manifest_sha256": exposure["manifest_sha256"], "tracking_since_step": exposure["tracking_since_step"],
            "last_completed_step": exposure["last_completed_step"], "initialization": exposure.get("initialization"),
            "unique_anchors": len(exposure["keys"]), "total_train_anchors": len(table), "applied_bundles": exposure["bundles"]}


def scale_optimizer_lr(optimizer, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("lr-scale must be positive and finite")
    before = [group["lr"] for group in optimizer.param_groups]
    after = [value * scale for value in before]
    if any(not math.isfinite(value) or value <= 0 for value in after):
        raise ValueError("scaled learning rates must be positive and finite")
    for group, value in zip(optimizer.param_groups, after, strict=True): group["lr"] = value
    return {"before": before, "after": after, "scale": scale}


def branch_passes(mode):
    if mode == "teacher": return ((False, 1.),)
    if mode == "mixed": return ((False, .5), (True, .5))
    raise ValueError("branch-inputs must be teacher or mixed")


def validate_branch_supervision(inputs, supervision):
    if supervision not in ("first-effect", "uniform-next") or (supervision == "uniform-next" and inputs != "teacher"):
        raise ValueError("uniform-next requires --branch-inputs teacher")


def next_label_coverage(table):
    result = {"valid_next_per_arm": [0] * 6, "total_next_per_arm": [len(table) * 8] * 6,
              "valid_count_histogram_per_arm": [[0] * 9 for _ in range(6)]}
    for bundle in table.values():
        labels = bundle["labels"]
        known = (labels["shape_id"][:, 1:] >= 0) & np.isin(labels["evidence"][:, 1:], (1, 2))
        for arm in range(6):
            count = int(known[arm].sum())
            result["valid_next_per_arm"][arm] += count
            result["valid_count_histogram_per_arm"][arm][count] += 1
    return result


def append_memory_optimizer(old, model):
    result = copy.deepcopy(old)
    parameters = list(model.parameters())
    preview = isinstance(model, SpawnPreviewSceneDynamics)
    group_count = 3 if preview else 2
    if not isinstance(model, (ExtendedSceneDynamics, RelativeSceneDynamics, FootprintSceneDynamics,
                              PoseRerankSceneDynamics, LearnedOverlapSceneDynamics)):
        raise ValueError("scene family has no optimizer migration")
    old_count = model.preview_base_parameter_count if preview else model.legacy_parameter_count
    if (len(result["param_groups"]) != group_count
            or sum(len(group["params"]) for group in result["param_groups"]) != old_count):
        raise ValueError(f"scene migration requires {group_count} source optimizer groups")
    first = max(identifier for group in result["param_groups"] for identifier in group["params"]) + 1
    group = copy.deepcopy(result["param_groups"][0])
    group.update(params=list(range(first, first + len(parameters) - old_count)), lr=3e-3)
    result["param_groups"].append(group)
    return result


def scene_optimizer(model):
    parameters = list(model.parameters()); core = model.base_parameter_count
    legacy = getattr(model, "legacy_parameter_count", len(parameters))
    groups = [{"params": parameters[:core], "lr": 3e-4}, {"params": parameters[core:legacy], "lr": 3e-3}]
    overlap_end = getattr(model, "preview_base_parameter_count", len(parameters))
    if legacy < overlap_end: groups.append({"params": parameters[legacy:overlap_end], "lr": 3e-3})
    if overlap_end < len(parameters): groups.append({"params": parameters[overlap_end:], "lr": 3e-3})
    return torch.optim.AdamW(groups, weight_decay=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/observed-last-checkpoint.pt")
    parser.add_argument("--selection-checkpoint", type=Path)
    parser.add_argument("--interventions", type=Path)
    parser.add_argument("--migrate-interventions-from", type=Path)
    parser.add_argument("--lr-scale", type=float, default=1.)
    parser.add_argument("--branch-inputs", choices=("teacher", "mixed"), default="teacher")
    parser.add_argument("--branch-supervision", choices=("first-effect", "uniform-next"), default="uniform-next")
    parser.add_argument("--expand-memory", action="store_true")
    parser.add_argument("--relative-motion", action="store_true")
    parser.add_argument("--local-pose", action="store_true")
    parser.add_argument("--pose-rerank", action="store_true")
    parser.add_argument("--action-invariant-current", action="store_true")
    parser.add_argument("--learned-overlap", action="store_true")
    parser.add_argument("--spawn-preview", action="store_true")
    parser.add_argument("--copy-overlap-weight", type=float, default=0.)
    parser.add_argument("--overlap-ranking-weight", type=float, default=0.)
    parser.add_argument("--descent-focal-weight", type=float, default=0.)
    parser.add_argument("--descent-ranking-weight", type=float, default=0.)
    parser.add_argument("--local-recovery-weight", type=float, default=0.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=3000)
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if min(args.seconds, args.updates, args.threads) < 1 or args.output.exists():
        parser.error("positive bounds and new output required")
    if not math.isfinite(args.lr_scale) or args.lr_scale <= 0:
        parser.error("lr-scale must be positive and finite")
    try: validate_branch_supervision(args.branch_inputs, args.branch_supervision)
    except ValueError as error: parser.error(str(error))
    if sum((args.relative_motion, args.expand_memory, args.local_pose, args.pose_rerank,
            args.action_invariant_current, args.learned_overlap, args.spawn_preview)) > 1:
        parser.error("choose at most one scene architecture migration")
    if not math.isfinite(args.copy_overlap_weight) or args.copy_overlap_weight < 0:
        parser.error("copy-overlap-weight must be finite and nonnegative")
    if not math.isfinite(args.overlap_ranking_weight) or args.overlap_ranking_weight < 0:
        parser.error("overlap-ranking-weight must be finite and nonnegative")
    if not math.isfinite(args.descent_focal_weight) or args.descent_focal_weight < 0:
        parser.error("descent-focal-weight must be finite and nonnegative")
    if not math.isfinite(args.descent_ranking_weight) or args.descent_ranking_weight < 0:
        parser.error("descent-ranking-weight must be finite and nonnegative")
    if not math.isfinite(args.local_recovery_weight) or args.local_recovery_weight < 0:
        parser.error("local-recovery-weight must be finite and nonnegative")
    torch.set_num_threads(args.threads); torch.manual_seed(1042029)
    rng = random.Random(1042029)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    raw, episodes, annotations, shapes, tiles, digest, annotation_digest = load_scene_data(checkpoint)
    upgrading = "_object_scene" not in checkpoint["model"]
    weights = upgrade_scene_weights(checkpoint["model"], shapes, tiles) if upgrading else checkpoint["model"]
    if not torch.equal(weights["object_shapes"], shapes) or not torch.equal(weights["object_tiles"], tiles):
        raise ValueError("changed object vocabulary on resume")
    adding_parameters = (args.expand_memory or args.relative_motion or args.local_pose or args.pose_rerank
                         or args.learned_overlap or args.spawn_preview)
    migrating_family = adding_parameters or args.action_invariant_current
    if migrating_family and upgrading:
        raise ValueError("scene migration requires a full legacy scene checkpoint")
    if args.spawn_preview:
        model = spawn_preview_scene_weights(weights)
    elif args.learned_overlap:
        model = learned_overlap_scene_weights(weights)
    elif args.action_invariant_current:
        model = action_invariant_current_scene_weights(weights)
    elif args.pose_rerank:
        model = pose_rerank_scene_weights(weights)
    elif args.local_pose:
        model = footprint_scene_weights(weights)
    elif args.relative_motion:
        model = relative_scene_weights(weights)
    elif args.expand_memory:
        model = expand_scene_weights(weights)
    else:
        model = scene_from_weights(weights)
    addon_directory = intervention_directory(checkpoint, args.interventions)
    interventions, addon_digest = load_action_interventions(checkpoint, model, episodes, annotations, digest, annotation_digest,
                                                           addon_directory, args.migrate_interventions_from)
    migration = verify_extension(addon_directory, intervention_directory(checkpoint, args.migrate_interventions_from)) if args.migrate_interventions_from else None
    exposure = resume_exposure(checkpoint, interventions["train"], addon_digest, migration)
    coverage = {split: composition_coverage(model, episodes[split], annotations[split]) for split in ("train", "validation")}
    if any(min(counts[kind] for kind in ("copy", "compose", "rewrite")) < 1 for counts in coverage.values()):
        raise ValueError(f"missing composition supervision: {coverage}")
    mode_train_coverage, mode_weights = training_mode_balance(model, episodes, annotations, interventions)
    relative_coverage = relative_supervision_coverage(model, episodes, annotations, interventions) if isinstance(model, RelativeSceneDynamics) else None
    parent_loss_fn = partial(scene_loss, mode_weights=mode_weights["parent"], copy_overlap_weight=args.copy_overlap_weight,
                             descent_focal_weight=args.descent_focal_weight, overlap_ranking_weight=args.overlap_ranking_weight,
                             descent_ranking_weight=args.descent_ranking_weight)
    branch_loss_fn = partial(scene_loss, mode_weights=mode_weights["interventions"], copy_overlap_weight=args.copy_overlap_weight,
                             overlap_ranking_weight=args.overlap_ranking_weight)
    optimizer = scene_optimizer(model)
    optimizer_weights = append_scene_optimizer(checkpoint["optimizer"], model) if upgrading else checkpoint["optimizer"]
    if adding_parameters: optimizer_weights = append_memory_optimizer(optimizer_weights, model)
    optimizer.load_state_dict(optimizer_weights)
    learning_rates = scale_optimizer_lr(optimizer, args.lr_scale)
    batch = checkpoint["config"]["batch"]
    streams = StreamWindows(episodes["train"], batch, rng)
    streams.cursors = checkpoint["stream_cursors"]
    state = expand_scene_state(checkpoint["stream_state"]) if args.expand_memory else checkpoint["stream_state"]
    if tuple(state.shape) != (batch, model.state_channels, 18, 20) or args.updates <= checkpoint["step"]:
        raise ValueError("invalid preserved context or update target")
    rng.setstate(checkpoint["python_rng"]); torch.set_rng_state(checkpoint["torch_rng"])
    selection_path = args.selection_checkpoint or args.checkpoint
    selected_exposure = (exposure_summary(exposure, interventions["train"])
                         if (checkpoint.get("training_exposure") or {}).get("manifest_sha256") == addon_digest else None)
    if args.selection_checkpoint is not None:
        selection = torch.load(selection_path, map_location="cpu", weights_only=True)
        allowed = {addon_digest, migration["from_sha256"]} if migration else {addon_digest}
        if selection["config"].get("interventions_sha256", addon_digest) not in allowed:
            raise ValueError("selection checkpoint does not match loaded interventions")
        baseline_model = selection_baseline(checkpoint, selection, model, migration)
        selection_ledger = selection.get("training_exposure")
        validated_ledger = resume_exposure(selection, interventions["train"], addon_digest, migration) if selection_ledger is not None else None
        selected_exposure = (exposure_summary(validated_ledger, interventions["train"])
                             if selection_ledger is not None and selection_ledger["manifest_sha256"] == addon_digest else None)
        del selection
    else:
        baseline_model = scene_from_weights(weights) if migrating_family else copy.deepcopy(model)
    baseline_teacher = rendered_teacher(baseline_model, episodes["validation"], annotations["validation"], interventions["validation"])
    baseline_rollout, _ = measure(baseline_model, episodes["validation"], raw["validation"])
    best, best_path = score(baseline_teacher, baseline_rollout), selection_path
    if not math.isfinite(best):
        raise ValueError("critical validation classes are missing or nonfinite")
    initial_relative = None
    if args.relative_motion:
        initial_teacher = rendered_teacher(model, episodes["validation"], annotations["validation"], interventions["validation"])
        initial_rollout, _ = measure(model, episodes["validation"], raw["validation"])
        initial_relative = {"teacher": initial_teacher, "validation": initial_rollout,
                            "score": score(initial_teacher, initial_rollout), "prediction_parity_expected": False}
    horizon = rollout_horizon(checkpoint.get("horizon", 1))
    args.output.mkdir(parents=True)
    config = {"protocol": "object-scene-v5-action-interventions", "manifest_sha256": digest, "annotations_sha256": annotation_digest,
              "model_family": type(model).__name__, "state_channels": model.state_channels,
              "spawn_preview_auxiliary": {
                  "joint_weight": .1, "compose_weight": .1, "visibility_weight": .1,
                  "input_crop_yx": [[0, 6], [13, 19]],
                  "new_parameters": sum(parameter.numel() for parameter in model.spawn_preview_head.parameters()),
                  "targets": "proved supported current object retained, with a unique four-tile addition in the top four rows"
              } if isinstance(model, SpawnPreviewSceneDynamics) else None,
              "spawn_preview_migration": {
                  "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  "zero_init": True, "legacy_optimizer_groups_preserved": 3
              } if args.spawn_preview else None,
              "learned_overlap_auxiliary": {"weight": .1, "balanced_classes_per_example": True,
                                              "targets": "observed board minus proved current object"} if isinstance(model, LearnedOverlapSceneDynamics) else None,
              "learned_overlap_migration": {"source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                             "zero_init": True, "legacy_moments_preserved": True} if args.learned_overlap else None,
              "action_invariant_current_migration": {"source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                                      "marker_only": True, "condition_id": 0,
                                                      "legacy_moments_preserved": True} if args.action_invariant_current else None,
              "local_pose_migration": {"source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                       "zero_init": True, "legacy_moments_preserved": True} if args.local_pose else None,
              "pose_rerank_migration": {"source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                         "shortlist": 3, "group_mass_preserved": True, "zero_init": True,
                                         "legacy_moments_preserved": True} if args.pose_rerank else None,
              "copy_overlap_weight": args.copy_overlap_weight,
              "overlap_ranking_weight": args.overlap_ranking_weight,
              "descent_focal_weight": args.descent_focal_weight,
              "descent_ranking_weight": args.descent_ranking_weight,
              "local_recovery_weight": args.local_recovery_weight,
              "memory_migration": {"from_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "state_from": 20, "state_to": 36, "legacy_moments_preserved": True} if args.expand_memory else None,
              "relative_motion_migration": {"from_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                            "legacy_moments_preserved": True, "state_channels": 20,
                                            "initial_alpha": float(torch.sigmoid(torch.tensor(-4.))), "initial_prediction_parity": False,
                                            "generic_joint_transport": "49learned shifts; no collision/gravity/button-direction rules; numerical-zero-mass-only absolute fallback"} if args.relative_motion else None,
              "relative_motion_initial_evaluation": initial_relative,
              "relative_auxiliary": {"delta_ce_weight": .1, "gate_bce_weight": .1,
                                     "positive": "both-proven,same-shape/style,unchanged-background,local3",
                                     "negative": "both-proven,shape/style-change-or-outside-local3; changed-background-alone-masked",
                                     "gate_semantics": "inductive-preference-not-calibrated-posterior", "train_coverage": relative_coverage} if relative_coverage else None,
              "interventions_path": str(addon_directory.relative_to(ROOT.resolve())), "interventions_sha256": addon_digest,
              "interventions_migration": migration,
              "lr_before": learning_rates["before"], "lr_after": learning_rates["after"], "lr_scale": args.lr_scale,
              "branch_inputs": args.branch_inputs,
              "branch_supervision": args.branch_supervision, "next_label_train_coverage": next_label_coverage(interventions["train"]),
              "intervention_loss_weight": 1, "intervention_training": ("teacher8; immediate-backward; mean-per-anchor; detached-cloned-context" if args.branch_inputs == "teacher" else "0.5teacher8+0.5autonomous8; mean-per-anchor; immediate-backward-each-half; same-detached-cloned-context; RNN8-linked; hard-feedback-detached; current-labels-masked-on-input-mismatch"),
              "intervention_temporal_weighting": "per-arm: half at first target-board divergence from noop (or step1 if none), half across seven other steps; target-only",
              "intervention_joint_auxiliary": ("weight1; all-proven-next8 NLL; mean per valid-arm and known-step; uniform-scene1/8; targets-only; no contact qualification" if args.branch_supervision == "uniform-next" else "weight1; joint shape-origin NLL on all6 qualified arms once at first effect/fallback0; mean qualified arms; noop not duplicated; target-only"),
              "intervention_validation": "autonomous8; metrics-all-steps1..8; labels-never-neural-inputs",
              "mode_training_coverage": mode_train_coverage, "mode_class_weights": mode_weights,
              "mode_weight_definition": "N/(3*n_class), train-only, separate-parent/intervention-corpora; weighted-CE/known-count",
              "batch": batch, "unroll": 32, "parameters": sum(p.numel() for p in model.parameters()),
              "initial_rollout_horizon": horizon, "rollout_schedule": "double-on-best-improvement-and-all-critical-events-at-least-0.95", "mode_supervision": coverage,
              "selection": "v4-score-plus-macro-six-button-joint-error-plus-macro-four-rendered-rotation-errors-plus-macro-board-pixel-error; missing-class-invalid",
              "teacher_scope": "all-validation-transitions-after-observed-prefix32; baseline-remeasured-identically",
              "baseline": {"teacher": baseline_teacher, "validation": baseline_rollout, "score": best,
                           "checkpoint_path": str(selection_path.resolve()), "checkpoint_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest()},
              "curriculum_criteria": {"minimum": .95, "metrics": ["addition_accuracy", "erasure_accuracy", "mode_recall.copy", "mode_recall.compose", "mode_recall.rewrite", "next_joint_accuracy.motion", "next_joint_accuracy.still"],
                                      "interventions": "all-six-joint-accuracies-and-A/B-effect-and-noeffect-rendered-accuracies"},
              "rollout_anchor_phase": "optimizer-step-mod-horizon",
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "torch": str(torch.__version__), "sources": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ("train_scene.py", "scene_model.py", "token_model.py", "intervention_training.py")}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    started, losses = time.monotonic(), []
    recovery_rows = 0
    model.train()
    with (args.output / "metrics.jsonl").open("x", buffering=1) as log:
        for step in range(checkpoint["step"] + 1, args.updates + 1):
            frames, actions, valid, reset, prefix, prefix_actions = streams.next()
            labels = window_labels(streams.cursors, valid, annotations["train"])
            anchors = windowanchors(streams.cursors, valid, interventions["train"])
            anchor_count = sum(len(bundles) for bundles in anchors.values())
            state = state.detach()
            chosen = reset.nonzero(as_tuple=True)[0]
            if len(chosen):
                warm = None
                with torch.no_grad():
                    for t in range(32):
                        _, warm = model(prefix[chosen, t], prefix_actions[chosen, t], warm)
                state[chosen] = warm
            optimizer.zero_grad(set_to_none=True)
            total, generated = torch.zeros(()), None
            branch_value = 0.
            applied_keys = []
            for t in range(32):
                for slot, bundle in anchors.get(t, []):
                    for autonomous, weight in branch_passes(args.branch_inputs):
                        objective = weight * branch_objective(model, bundle, state[slot:slot + 1], branch_loss_fn,
                                                         autonomous=autonomous, supervision=args.branch_supervision) / anchor_count
                        if not torch.isfinite(objective):
                            raise RuntimeError("nonfinite intervention objective")
                        branch_value += float(objective.detach())
                        objective.backward()
                        del objective
                    applied_keys.append(exposure_key(streams.cursors, valid, slot, t))
                current, current_matches = training_input(frames[:, t], generated, t + step, horizon)
                if args.local_recovery_weight and generated is not None:
                    recover = recovery_mask(generated, frames[:, t], generated_joint, labels, t,
                                            predecessor_matches, valid[:, t])
                    if recover.any():
                        corrupted = generated[recover]
                        repaired = model.scene_details(corrupted, actions[recover, t], state[recover])
                        recovery_labels = {key: values[recover] for key, values in labels.items()}
                        recovery_count = int(recover.sum())
                        total = total + args.local_recovery_weight * recovery_count / len(recover) * scene_loss(
                            model, corrupted, frames[recover, t + 1], repaired, recovery_labels, t,
                            valid[recover, t], torch.zeros_like(valid[recover, t]), mode_weights["parent"])
                        recovery_rows += recovery_count
                details = model.scene_details(current, actions[:, t], state)
                total = total + parent_loss_fn(model, current, frames[:, t + 1], details, labels, t, valid[:, t],
                                               current_matches, actions=actions[:, t])
                state = details["state"]
                generated = model.decode_scene(current, details, hard=True).detach()
                if args.local_recovery_weight:
                    next_object = details["next"]
                    generated_joint = next_object["joint"].detach().flatten(1).argmax(1)
                    generated_joint = generated_joint.masked_fill(next_object["visible"][:, 0].detach() <= 0, -1)
                    predecessor_matches = current_matches & valid[:, t]
            loss = total / 32
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite scene loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            exposure = optimizer_step_with_exposure(optimizer, exposure, step, applied_keys, interventions["train"])
            losses.append(float(loss.detach()) + branch_value)
            expired = time.monotonic() - started >= args.seconds
            if step % 1000 == 0 or step == args.updates or expired:
                teacher = rendered_teacher(model, episodes["validation"], annotations["validation"], interventions["validation"])
                rollout, _ = measure(model, episodes["validation"], raw["validation"])
                value = score(teacher, rollout)
                trained_horizon = horizon
                horizon = rollout_horizon(horizon, value, best, teacher)
                record = {"step": step, "seconds": time.monotonic() - started, "loss": float(np.mean(losses)),
                          "local_recovery_rows": recovery_rows,
                          "training_exposure": exposure_summary(exposure, interventions["train"]),
                          "trained_horizon": trained_horizon, "horizon": horizon, "teacher": teacher, "validation": rollout}
                log.write(json.dumps(record) + "\n"); print(json.dumps(record), flush=True); losses = []
                if value < best or expired or step == args.updates:
                    path = args.output / f"checkpoint-{step:06d}.pt"
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": config,
                                "training_exposure": copy.deepcopy(exposure),
                                "teacher": teacher, "validation": rollout, "horizon": horizon, "stream_cursors": streams.cursors,
                                "stream_state": state.detach(), "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state()}, path)
                    if value < best:
                        best, best_path = value, path
                        selected_exposure = exposure_summary(exposure, interventions["train"])
                model.train()
                if expired:
                    break
    selected = baseline_model if best_path == selection_path else load_tile_model(best_path)
    torch.save(selected.state_dict(), args.output / "model.pt")
    teacher = rendered_teacher(selected, episodes["validation"], annotations["validation"], interventions["validation"])
    rollout, examples = measure(selected, episodes["validation"], raw["validation"])
    for i, (predicted, reference) in enumerate(examples):
        comparison_image(predicted, reference, args.output / f"comparison-{i:02d}.png")
    result = {"status": "trained-gameplay-unverified", "best_checkpoint": best_path.name,
              "last_exposure": exposure_summary(exposure, interventions["train"]), "selected_exposure": selected_exposure,
              "best_checkpoint_path": str(best_path.resolve()),
              "best_from_selection_baseline": best_path.resolve() == selection_path.resolve(),
              "best_from_resume": best_path.resolve() == args.checkpoint.resolve(),
              "export_model_family": type(selected).__name__, "teacher": teacher, "validation": rollout, "test_used": False,
              "seconds": time.monotonic() - started, "config": config}
    (args.output / "result.json").write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
