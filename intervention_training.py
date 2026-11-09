"""Verified observable interventions: training targets and held-out causal metrics."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model import ACTIONS
from object_annotations import PATTERNS
from token_model import encode_frames

BUTTONS = ("noop", "left", "right", "down", "a", "b")
LABELS = ("cells", "shape_id", "evidence", "texture", "track")


def verify_extension(directory, base_directory=None):
    directory = Path(directory)
    target_bytes = (directory / "manifest.json").read_bytes()
    target = json.loads(target_bytes)
    name = target.get("base_directory", "")
    if not name or name in (".", "..") or Path(name).name != name:
        raise ValueError("extension base must be a sibling directory name")
    if base_directory is None:
        base_directory = (directory.parent / name).resolve()
        if base_directory.parent != directory.parent.resolve() or base_directory == directory.resolve():
            raise ValueError("resolved extension base must be a distinct sibling directory")
    else:
        base_directory = Path(base_directory)
    base_bytes = (base_directory / "manifest.json").read_bytes()
    base = json.loads(base_bytes)
    base_sha = hashlib.sha256(base_bytes).hexdigest()
    if ((base["version"], target["version"]) not in ((1, 2), (2, 3))
            or base["status"] != "complete" or target["status"] != "complete"
            or base["test_used"] or target["test_used"] or target.get("base_manifest_sha256") != base_sha):
        raise ValueError("extension version/completion/base lineage mismatch")
    if target["version"] == 3:
        verify_extension(base_directory)
    for key in ("parent_manifest_sha256", "annotations_sha256", "dictionary", "buttons", "steps"):
        if target[key] != base[key]:
            raise ValueError(f"extension changes immutable corpus contract: {key}")
    records = {record["file"]: record for record in target["episodes"]}
    if len(records) != len(target["episodes"]):
        raise ValueError("duplicate extension file")
    for record in base["episodes"]:
        if records.get(record["file"]) != record:
            raise ValueError("extension changed or removed an inherited record")
        for root in (base_directory, directory):
            path = (root / record["file"]).resolve()
            if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("inherited NPZ bytes differ from base")
    inherited, total = len(base["episodes"]), len(target["episodes"])
    parent_counts = {}
    anchors = set()
    for record in target["episodes"]:
        parent_counts[record["parent_file"]] = parent_counts.get(record["parent_file"], 0) + 1
        key = (record["parent_file"], record["anchor"])
        if key in anchors:
            raise ValueError("duplicate extension anchor")
        anchors.add(key)
    cap = 32 if target["version"] == 2 else 40
    if total <= inherited or any(count > cap for count in parent_counts.values()):
        raise ValueError(f"extension must be a strict superset with at most{cap} anchors per parent")
    if target["version"] == 3:
        base_files = {record["file"] for record in base["episodes"]}
        parent_identity = {record["parent_file"]: (record["split"], record["parent_sha256"]) for record in base["episodes"]}
        additions = Counter()
        for record in target["episodes"]:
            if record["file"] in base_files:
                continue
            parent = record["parent_file"]
            if (record["split"] != "train" or not record["witness_verified"]
                    or parent_identity.get(parent, ("train", record["parent_sha256"])) != ("train", record["parent_sha256"])):
                raise ValueError("V3 may only add witnessed TRAIN records with consistent parent provenance")
            additions[parent] += 1
            path = (directory / record["file"]).resolve()
            if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("new V3 NPZ hash mismatch")
        if any(count > 8 for count in additions.values()):
            raise ValueError("V3 may add at most eight anchors per parent")
    if (target.get("inherited_anchors") != inherited or target.get("new_anchors") != total - inherited
            or target["witness_frames"] != total * 9 or target.get("inherited_witness_frames") != inherited * 9
            or target.get("new_witness_frames") != (total - inherited) * 9 or target["anchors_per_parent_max"] != cap):
        raise ValueError("extension witness/count proof mismatch")
    return {"from_sha256": base_sha, "to_sha256": hashlib.sha256(target_bytes).hexdigest(),
            "inherited_anchors": inherited, "new_anchors": total - inherited,
            "inherited_npz_sha256_preserved": True, "parent_annotations_dictionary_unchanged": True}


def load_interventions(directory, model, parent_manifest, parent_episodes, parent_annotations,
                       parent_sha, annotation_sha, base_directory=None, *, splits=("train", "validation")):
    directory = Path(directory)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    records = manifest["episodes"]
    if manifest["version"] in (2, 3):
        verify_extension(directory, base_directory)
    if (manifest["status"] != "complete" or manifest["test_used"] or manifest["version"] not in (1, 2, 3)
            or manifest["parent_manifest_sha256"] != parent_sha or manifest["annotations_sha256"] != annotation_sha
            or manifest["buttons"] != list(BUTTONS) or manifest["steps"] != 8
            or manifest["witness_frames"] != len(records) * 9 or not manifest["labels_target_only"]):
        raise ValueError("intervention provenance, completion or witness mismatch")
    dictionary = sorted(manifest["dictionary"], key=lambda item: item["id"])
    if ([item["id"] for item in dictionary] != list(range(len(dictionary)))
            or not torch.equal(torch.tensor([item["cells"] for item in dictionary]), model.object_shapes.cpu())):
        raise ValueError("intervention frozen shape dictionary mismatch")
    book = model.codebook.detach().cpu().numpy()
    if len(book) > 256:
        raise ValueError("intervention uint8 tokens require vocabulary<=256")
    for item, tile in zip(dictionary, model.object_tiles.cpu(), strict=True):
        if not np.array_equal(book[int(tile)], PATTERNS[item["texture"]]):
            raise ValueError("intervention frozen appearance dictionary mismatch")
    if not splits or set(splits) - {"train", "validation"}:
        raise ValueError("invalid intervention splits")
    parents, tables = {}, {split: {} for split in splits}
    for split in tables:
        entries = [entry for entry in parent_manifest["episodes"] if entry["split"] == split]
        if len(entries) != len(parent_episodes[split]) or len(entries) != len(parent_annotations[split]):
            raise ValueError("cached parent split alignment mismatch")
        parents.update({entry["file"]: (split, index, entry) for index, entry in enumerate(entries)})
    allowed = {"frames", "actions", "metadata", "propagation", *LABELS}
    for record in records:
        if record["split"] not in ("train", "validation"):
            raise ValueError("intervention record uses a reserved split")
        if record["split"] not in tables:
            continue
        if record["parent_file"] not in parents:
            raise ValueError("intervention parent is missing or reserved test")
        split, episode, parent = parents[record["parent_file"]]
        anchor = record["anchor"]
        if (record["split"] != split or record["parent_sha256"] != parent["sha256"]
                or record["source"] != "capture-intervention" or not record["witness_verified"]
                or not record["labels_target_only"] or record["snapshot_persisted"]):
            raise ValueError("intervention parent split or capture proof mismatch")
        path = (directory / record["file"]).resolve()
        if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("intervention file hash/path mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) - allowed or not {"frames", "actions", "metadata", *LABELS} <= set(archive.files):
                raise ValueError("intervention contains forbidden/internal-state or missing fields")
            frames, actions = archive["frames"], archive["actions"]
            metadata = json.loads(str(archive["metadata"]))
            labels = {key: archive[key].copy() for key in LABELS}
        for key in ("source", "parent_file", "parent_sha256", "split", "anchor", "context", "rom_sha256",
                    "witness_verified", "labels_target_only", "snapshot_persisted"):
            if metadata.get(key) != record.get(key):
                raise ValueError(f"intervention metadata mismatch: {key}")
        if metadata["context"] != parent_manifest["context"] or metadata["rom_sha256"] != parent_manifest["context"]["rom"]["sha256"]:
            raise ValueError("intervention ROM/context mismatch")
        if (frames.dtype != np.uint8 or frames.shape != (6, 9, 72, 80) or np.any(frames > 3)
                or actions.dtype != np.uint8 or actions.shape != (6, 8)):
            raise ValueError("invalid intervention image/action format")
        expected_actions = np.array([[ACTIONS.index(button) if button not in ("a", "b") or t == 0 else 0
                                      for t in range(8)] for button in BUTTONS], dtype=np.uint8)
        if not np.array_equal(actions, expected_actions) or not np.all(frames[:, 0] == frames[0, 0]):
            raise ValueError("intervention action schedule or shared initial image mismatch")
        for key, values in labels.items():
            if values.shape != ((6, 9, 4, 2) if key == "cells" else (6, 9)) or values.dtype.kind not in "iu":
                raise ValueError(f"invalid intervention labels: {key}")
        ids, evidence = labels["shape_id"], labels["evidence"]
        positive = (ids >= 0) & np.isin(evidence, (1, 2))
        if (np.any(ids < -1) or np.any(ids >= len(dictionary)) or np.any(evidence > 4)
                or np.any(labels["cells"][positive] < 0) or np.any(labels["cells"][positive, :, 0] >= 18)
                or np.any(labels["cells"][positive, :, 1] >= 10)):
            raise ValueError("intervention labels outside visible vocabulary/board")
        for cells, shape_id, texture in zip(labels["cells"][positive], ids[positive], labels["texture"][positive], strict=True):
            item = dictionary[int(shape_id)]
            normalized = cells - cells.min(0)
            if (int(texture) != item["texture"] or len(np.unique(cells, axis=0)) != 4
                    or not np.array_equal(normalized, np.asarray(item["cells"]))):
                raise ValueError("intervention object label contradicts frozen shape dictionary")
        tokens, missing = encode_frames(frames.reshape(-1, 72, 80), book)
        if missing:
            raise ValueError(f"intervention has{missing} unseen visual tiles; frozen vocabulary cannot encode {record['file']}")
        tokens = tokens.reshape(6, 9, 18, 20).astype(np.uint8)
        parent_tokens, parent_actions = parent_episodes[split][episode]
        if anchor < 32 or anchor + 8 > len(parent_actions) or not np.all(tokens[:, 0] == parent_tokens[anchor]):
            raise ValueError("intervention anchor image does not match cached parent")
        if record.get("previous_action") != ACTIONS[int(parent_actions[anchor - 1])]:
            raise ValueError("intervention previous button differs from parent")
        annotation = parent_annotations[split][episode]
        if (not np.all(ids[:, 0] == annotation["shape_id"][anchor])
                or not np.all(labels["cells"][:, 0] == annotation["cells"][anchor])
                or not np.all(positive[:, 0])):
            raise ValueError("intervention initial object differs from proven parent")
        key = (episode, anchor)
        if key in tables[split]:
            raise ValueError("duplicate intervention anchor")
        tables[split][key] = {"tokens": tokens, "actions": actions.copy(), "labels": labels}
    return tables, hashlib.sha256(manifest_bytes).hexdigest()


def windowanchors(cursors, valid, table):
    result = {}
    for slot, (episode, end) in enumerate(cursors):
        used = int(valid[slot].sum())
        start = end - used
        for t in range(used):
            if not bool(valid[slot, t]):
                raise ValueError("stream validity must be a contiguous prefix")
            bundle = table.get((episode, start + t))
            if bundle is not None:
                result.setdefault(t, []).append((slot, bundle))
    return result


def tensors(model, bundle, context):
    device = model.codebook.device
    frames = torch.as_tensor(bundle["tokens"], dtype=torch.long, device=device)
    actions = torch.as_tensor(bundle["actions"], dtype=torch.long, device=device)
    labels = {key: torch.as_tensor(value, dtype=torch.long, device=device) for key, value in bundle["labels"].items()}
    if context is not None and context.shape[0] != 1:
        raise ValueError("intervention context must describe one parent stream")
    state = None if context is None else context.detach().to(device).expand(6, *context.shape[1:]).clone()
    return frames, actions, labels, state


@torch.no_grad()
def temporal_weights(tokens):
    """Half the target-only weight at first board divergence from the noop arm."""
    tokens = torch.as_tensor(tokens)
    if tuple(tokens.shape) != (6, 9, 18, 20):
        raise ValueError("temporal weights require tokens[6,9,18,20]")
    different = (tokens[:, 1:, :, 2:12] != tokens[0:1, 1:, :, 2:12]).flatten(2).any(2)
    first = different.long().argmax(1)  # All-false (including noop) deliberately selects0.
    weights = torch.full((6, 8), .5 / 7, dtype=torch.float32, device=tokens.device)
    return weights.scatter(1, first[:, None], .5)


def _slice_arm(value, arm):
    if isinstance(value, torch.Tensor):
        return value[arm:arm + 1] if value.ndim and value.shape[0] == 6 else value
    if isinstance(value, dict):
        return {key: _slice_arm(item, arm) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_slice_arm(item, arm) for item in value)
    return value


def first_effect_joint_loss(model, frames, labels, details, t, first):
    """Sum one qualified shape-and-origin NLL per arm at its first effect only."""
    joint = details.get("next", {}).get("joint")
    zero = joint.exp().sum() * 0 if joint is not None else frames.new_zeros((), dtype=torch.float32)
    total, count = zero, 0
    if not {"cells", "shape_id", "evidence"} <= labels.keys():
        return total, count
    for arm in range(6):
        if t != int(first[arm]) or rotation_pair(model, frames, labels, t + 1, arm) is None:
            continue
        if joint is None:
            raise ValueError("qualified intervention requires next joint probabilities")
        origin = labels["cells"][arm, t + 1].amin(0)
        target = labels["shape_id"][arm, t + 1] * 180 + origin[0] * 10 + origin[1]
        total = total - joint[arm].flatten()[target]
        count += 1
    return total, count


def uniform_next_weights(labels):
    positive = (labels["shape_id"][:, 1:] >= 0) & ((labels["evidence"][:, 1:] == 1) | (labels["evidence"][:, 1:] == 2))
    per_arm = positive.sum(1)
    arms = (per_arm > 0).sum().clamp_min(1)
    return positive.float() / per_arm.clamp_min(1)[:, None] / arms


def uniform_next_joint_loss(joint, labels, t, weights):
    selected = weights[:, t] > 0
    if not torch.any(selected): return joint.exp().sum() * 0
    origin = labels["cells"][selected, t + 1].amin(1)
    target = labels["shape_id"][selected, t + 1] * 180 + origin[:, 0] * 10 + origin[:, 1]
    return (-joint[selected].flatten(1).gather(1, target[:, None])[:, 0] * weights[selected, t]).sum()


def branch_objective(model, bundle, context, loss_fn, autonomous=False, supervision="first-effect"):
    if supervision not in ("first-effect", "uniform-next") or (supervision == "uniform-next" and autonomous):
        raise ValueError("uniform-next supervision requires real teacher inputs")
    frames, actions, labels, state = tensors(model, bundle, context)
    weights = temporal_weights(frames) if supervision == "first-effect" else torch.full((6, 8), 1 / 8, device=frames.device)
    joint_weights = uniform_next_weights(labels) if supervision == "uniform-next" else None
    first = weights.argmax(1)
    valid = torch.ones(1, dtype=torch.bool, device=frames.device)
    loss = frames.new_zeros((), dtype=torch.float32)
    joint_loss, joint_pairs = loss, 0
    current = frames[:, 0]
    for t in range(8):
        if not autonomous:
            current = frames[:, t]
        current_matches = (current == frames[:, t]).flatten(1).all(1)
        details = model.scene_details(current, actions[:, t], state)
        for arm in range(6):
            loss = loss + weights[arm, t] * loss_fn(model, current[arm:arm + 1], frames[arm:arm + 1, t + 1],
                                                   _slice_arm(details, arm), _slice_arm(labels, arm), t, valid, current_matches[arm:arm + 1])
        if supervision == "uniform-next":
            auxiliary, pairs = uniform_next_joint_loss(details["next"]["joint"], labels, t, joint_weights), 0
        else:
            auxiliary, pairs = first_effect_joint_loss(model, frames, labels, details, t, first)
        joint_loss = joint_loss + auxiliary
        joint_pairs += pairs
        state = details["state"]
        if autonomous:
            current = model.decode_scene(current, details, hard=True).detach()
    return loss / 6 + joint_loss / max(1, joint_pairs)


@torch.inference_mode()
def branch_validation(model, bundle, context):
    frames, actions, labels, state = tensors(model, bundle, context)
    current = frames[:, 0]
    counts = {}
    for t in range(8):
        details = model.scene_details(current, actions[:, t], state)
        prediction = model.decode_scene(current, details, hard=True)
        state = details["state"]
        next_object = details["next"]
        selected = next_object["joint"].flatten(1).argmax(1)
        origin = labels["cells"][:, t + 1].amin(1)
        wanted = labels["shape_id"][:, t + 1] * 180 + origin[:, 0] * 10 + origin[:, 1]
        positive = (labels["shape_id"][:, t + 1] >= 0) & ((labels["evidence"][:, t + 1] == 1) | (labels["evidence"][:, t + 1] == 2))
        visible = next_object["visible"][:, 0] > 0
        joint_correct = (selected == wanted) & visible
        rendered_correct = (prediction[:, :, 2:12] == frames[:, t + 1, :, 2:12]).flatten(1).all(1)
        for arm, button in enumerate(BUTTONS):
            for suffix, value in (("joint_total", int(positive[arm])), ("joint_correct", int(positive[arm] & joint_correct[arm])),
                                  ("visible_correct", int(positive[arm] & visible[arm]))):
                key = f"{button}/{suffix}"
                counts[key] = counts.get(key, 0) + value
            wrong = model.codebook[prediction[arm, :, 2:12]] != model.codebook[frames[arm, t + 1, :, 2:12]]
            counts[f"{button}/board_pixel_errors"] = counts.get(f"{button}/board_pixel_errors", 0) + int(wrong.sum())
            counts[f"{button}/board_pixels"] = counts.get(f"{button}/board_pixels", 0) + wrong.numel()
        for arm, button in ((4, "a"), (5, "b")):
            category = rotation_pair(model, frames, labels, t + 1, arm)
            counts.setdefault(f"{button}/effect_total", 0); counts.setdefault(f"{button}/effect_correct", 0)
            counts.setdefault(f"{button}/noeffect_total", 0); counts.setdefault(f"{button}/noeffect_correct", 0)
            counts.setdefault(f"{button}/effect_rendered_correct", 0); counts.setdefault(f"{button}/noeffect_rendered_correct", 0)
            counts.setdefault(f"{button}/unqualified", 0)
            if category is None:
                counts[f"{button}/unqualified"] += 1
            else:
                counts[f"{button}/{category}_total"] += 1
                counts[f"{button}/{category}_correct"] += int(joint_correct[0] & joint_correct[arm])
                counts[f"{button}/{category}_rendered_correct"] += int(joint_correct[0] & joint_correct[arm] & rendered_correct[0] & rendered_correct[arm])
        current = prediction
    return counts


def rotation_pair(model, frames, labels, t, arm):
    """Qualify counterfactual rotation using targets only, with unchanged visible canvas."""
    arms = (0, arm)
    for a in arms:
        if any(int(labels["shape_id"][a, index]) < 0 or int(labels["evidence"][a, index]) not in (1, 2) for index in (0, t)):
            return None
    backgrounds, shapes, colors = [], [], []
    for a in arms:
        for index in (0, t):
            shape_id = labels["shape_id"][a, index]
            colors.append(int(model.object_tiles[shape_id]))
            board = frames[a, index, :, 2:12].clone()
            cells = labels["cells"][a, index]
            if not torch.all(board[cells[:, 0], cells[:, 1]] == model.object_tiles[shape_id]):
                return None
            board[cells[:, 0], cells[:, 1]] = model.blank_id
            backgrounds.append(board)
        cells = labels["cells"][a, t]
        shapes.append(sorted(map(tuple, (cells - cells.amin(0)).cpu().tolist())))
    if len(set(colors)) != 1 or not all(torch.equal(backgrounds[0], board) for board in backgrounds[1:]):
        return None
    return "effect" if shapes[0] != shapes[1] else "noeffect"


def finish_validation(counts):
    rates = {}
    for button in BUTTONS:
        for name, numerator, denominator in (("joint_accuracy", "joint_correct", "joint_total"),
                                              ("visibility_recall", "visible_correct", "joint_total"),
                                              ("board_pixel_error", "board_pixel_errors", "board_pixels"),
                                              ("rotation_effect_accuracy", "effect_correct", "effect_total"),
                                              ("rotation_noeffect_accuracy", "noeffect_correct", "noeffect_total"),
                                              ("rotation_effect_rendered_accuracy", "effect_rendered_correct", "effect_total"),
                                              ("rotation_noeffect_rendered_accuracy", "noeffect_rendered_correct", "noeffect_total")):
            total = counts.get(f"{button}/{denominator}", 0)
            rates[f"{button}/{name}"] = counts.get(f"{button}/{numerator}", 0) / total if total else None
    return {"counts": dict(counts), "rates": rates}
