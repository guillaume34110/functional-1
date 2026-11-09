"""Compare observed and generated conditioning against the ROM offline."""
import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F
from pyboy import PyBoy

from audit_controls import supported
from audit_controls_reference import replay_prefix, verify_frame
from audit_long import recorded_path, summarize
from audit_spawn import preview_texture, spawn_evidence
from model import ACTIONS
from object_annotations import shape_key
from object_annotations_v2 import extend_labels
from run import ROOT, read_episode, step
from scene_model import LearnedOverlapSceneDynamics, ObjectSceneDynamics
from token_model import decode_tokens, encode_frames, load_tile_model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(frames, actions, metadata, anchor, commands, recorded):
    source = metadata["context"]["rom"]
    rom = ROOT / source["file"]
    if digest(rom) != source["sha256"]:
        raise ValueError("capture ROM hash mismatch")
    game = PyBoy(str(rom), window="null", sound_emulated=False,
                 color_palette=(0xFFFFFF, 0xAAAAAA, 0x555555, 0x000000))
    try:
        proof = replay_prefix(game, frames, actions, metadata, anchor)
        images = [frames[anchor]]
        for i, command in enumerate(commands, 1):
            image = step(game, ACTIONS[int(command)])
            if recorded:
                verify_frame(anchor + i, frames[anchor + i], image)
            images.append(image)
        return np.stack(images), proof
    finally:
        game.stop(save=False)


def pose_ranks(log_joint, shape, pose):
    expected = log_joint[shape, pose]
    shape_scores = log_joint.logsumexp(1)
    pose_scores = log_joint.logsumexp(0)
    row, col = divmod(pose, 10)
    same_shape = log_joint[shape].reshape(18, 10)
    best_shape, best_pose = divmod(int(log_joint.flatten().argmax()), 180)
    cross = torch.stack((log_joint[best_shape, pose], log_joint[shape, best_pose]))
    interaction = (float(expected + log_joint[best_shape, best_pose] - cross.sum())
                   if torch.isfinite(cross).all() else None)
    return {
        "joint": 1 + int((log_joint > expected).sum()),
        "shape_marginal": 1 + int((shape_scores > shape_scores[shape]).sum()),
        "position_marginal": 1 + int((pose_scores > pose_scores[pose]).sum()),
        "shape_at_position": 1 + int((log_joint[:, pose] > expected).sum()),
        "position_given_shape": 1 + int((log_joint[shape] > expected).sum()),
        "row_given_shape_and_column": 1 + int((same_shape[:, col] > expected).sum()),
        "column_given_shape_and_row": 1 + int((same_shape[row] > expected).sum()),
        "log_gap_to_best": float(log_joint.max() - expected),
        "cross_interaction": interaction,
    }


def overlap_attribution(model, tokens, details, reference_current, reference_next):
    raw_overlap = details["overlap_logits"][0]
    copy_probability = details.get("event_base_mode", details["mode"]).softmax(1)[0, 0]
    correction = model.overlap_scale * copy_probability * (
        F.softplus(raw_overlap.new_zeros(())) - F.softplus(raw_overlap))
    adjusted = details.get("event_base_joint", details["next"]["joint"])[0]
    raw, _ = model.next_head(details["state"][:, :16, :, 2:12], model.valid_poses)
    base = raw["joint"][0]
    preview = details.get("preview_joint_offset")
    if preview is not None:
        preview = preview[0].expand_as(base)
    motion = details.get("motion_joint_offset")
    consistency = details.get("consistency_joint_offset")
    rebuilt = (base + correction + (0 if preview is None else preview)
               + (0 if motion is None else motion[0])
               + (0 if consistency is None else consistency[0])).flatten().log_softmax(0).reshape_as(adjusted)
    torch.testing.assert_close(adjusted, rebuilt, rtol=1e-5, atol=1e-5)
    before, after = [divmod(int(scores.flatten().argmax()), 180) for scores in (base, adjusted)]
    expected_shape, y, x = reference_next
    expected = (expected_shape, y * 10 + x)
    old_shape, old_y, old_x = reference_current
    background = (tokens[0, :, 2:12] != model.blank_id) & ~model.shape_masks[old_shape, old_y * 10 + old_x].bool()
    candidates = {}
    for key, (shape, pose) in (("base_choice", before), ("adjusted_choice", after), ("reference", expected)):
        candidates[key] = {
            "shape_y_x": [shape, pose // 10, pose % 10],
            "base_log_probability": float(base[shape, pose]),
            "adjusted_log_probability": float(adjusted[shape, pose]),
            "overlap_logit": float(raw_overlap[shape, pose]),
            "correction": float(correction[shape, pose]),
            "preview_offset": None if preview is None else float(preview[shape, pose]),
            "motion_offset": None if motion is None else float(motion[0, shape, pose]),
            "consistency_offset": None if consistency is None else float(consistency[0, shape, pose]),
            "observed_input_overlap": bool((background & model.shape_masks[shape, pose].bool()).any()),
        }
    return {"copy_probability": float(copy_probability), "candidates": candidates,
            "reference_ranks_before_correction": pose_ranks(base, *expected)}


def inspect_prediction(model, tokens, target, command, state, labels, t, observed=False):
    details = model.scene_details(tokens, command, state)
    image = model.decode_scene(tokens, details, hard=True)
    result = {"board_errors": int((image[:, :, 2:12] != target[:, :, 2:12]).sum()),
              "mode": int(details["mode"].argmax(1))}
    if "contact_logit" in details:
        result["contact_probability"] = float(details["contact_logit"].sigmoid().item())
        result["contact_offset"] = float(details["contact_offset"].item())
        if details["contact_visible"].item():
            kind, position = divmod(int(details["current"]["joint"].flatten().argmax()), 180)
            local = torch.tensor([position, position + 10] if position < 170 else [position])
            mask = model.shape_masks[kind, position].bool()
            background = (tokens[0, :, 2:12] != model.blank_id) & ~mask
            result["contact_observation"] = {
                "supported": bool(supported(background.numpy(), mask.nonzero().numpy(), False)),
                "piece_tile_age": int(details["state"][0, 17, :, 2:12][mask].min()),
                "held_duration": int(details["state"][0, 18, 0, 0]),
            }
            free = ~(model.shape_masks[kind, local].bool() & background).flatten(1).any(1)
            local = local[free & model.valid_poses[kind, local]]
            if len(local):
                joint = details.get("event_base_joint", details["next"]["joint"])[0]
                best_local = joint[kind, local].max()
                result["local_stay_or_drop"] = {"base_rank": 1 + int((joint > best_local).sum()),
                                                "base_log_gap": float(joint.max() - best_local)}
                if "event_scores" in details:
                    copy_scores = details["event_scores"][0, 0]
                    local_best = copy_scores[kind * 180 + local].max()
                    result["local_stay_or_drop"]["copy_rank"] = 1 + int((copy_scores > local_best).sum())
                    result["local_stay_or_drop"]["copy_log_gap"] = float(copy_scores.max() - local_best)
                if len(local) == 2 and "event_scores" in details:
                    copy = details["event_scores"][0, 0, kind * 180 + local]
                    result["local_stay_or_drop"]["copy_stay_minus_drop"] = float(copy[0] - copy[1])
    if observed and "mode_base_logits" in details:
        result["mode_probabilities"] = details["mode"].softmax(1)[0].tolist()
        result["mode_base_probabilities"] = details["mode_base_logits"].softmax(1)[0].tolist()
        if "spatial_mode_base_logits" in details:
            result["spatial_mode_base_probabilities"] = details["spatial_mode_base_logits"].softmax(1)[0].tolist()
        if "rewrite_mode_base_logits" in details:
            result["rewrite_mode_base_probabilities"] = details["rewrite_mode_base_logits"].softmax(1)[0].tolist()
    for name, index in (("current", t), ("next", t + 1)):
        obj = details[name]
        joint = obj["joint"].flatten()
        selected = int(joint.argmax())
        shape, pose = divmod(selected, 180)
        visible = bool(obj["visible"].item() > 0)
        result[name] = {"selected": [shape, pose // 10, pose % 10] if visible else None}
        if labels["shape_id"][index] >= 0 and labels["evidence"][index] in (1, 2):
            y, x = map(int, labels["cells"][index].min(0))
            expected = int(labels["shape_id"][index])
            result[name]["reference"] = [expected, y, x]
            result[name]["reference_probability"] = float(joint[expected * 180 + y * 10 + x].exp())
            result[name]["reference_ranks"] = pose_ranks(obj["joint"][0], expected, y * 10 + x)
    if observed and "overlap_logits" in details and all("reference" in result[name] for name in ("current", "next")):
        result["overlap"] = overlap_attribution(model, tokens, details, result["current"]["reference"],
                                                result["next"]["reference"])
    return result, image, model.rendered_state(details, image)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--anchor", type=int, help="restrict a diagnostic to one proven validation anchor")
    parser.add_argument("--branch", choices=("recorded", "down"), help="restrict a diagnostic to one command sequence")
    parser.add_argument("--hud-attribution", action="store_true")
    parser.add_argument("--disable-overlap", action="store_true", help="offline ablation on a learned-overlap checkpoint")
    args = parser.parse_args()
    if not 1 <= args.steps <= 256:
        parser.error("steps must be between 1 and 256")
    torch.set_num_threads(2)
    manifest_path = ROOT / "data/manifest.json"
    annotation_path = ROOT / "data/object-annotations-v2/object-audit.json"
    seed_path = ROOT / "docs/long-audit-seeds-v15.json"
    manifest, annotations, seeds = [json.loads(p.read_bytes()) for p in (manifest_path, annotation_path, seed_path)]
    if (seeds["test_used"] or annotations["test_used"] or seeds["manifest_sha256"] != digest(manifest_path)
            or annotations["manifest_sha256"] != seeds["manifest_sha256"]
            or seeds["annotations_sha256"] != digest(annotation_path)):
        raise ValueError("validation seed provenance mismatch")
    model = load_tile_model(args.checkpoint).eval()
    if args.disable_overlap:
        if type(model) is not LearnedOverlapSceneDynamics:
            parser.error("disable-overlap requires the base learned-overlap family")
        model.scene_details = MethodType(ObjectSceneDynamics.scene_details, model)
    book = model.codebook.numpy()
    entries = {e["file"]: e for e in manifest["episodes"] if e["split"] == "validation"}
    records = {e["file"]: e for e in annotations["episodes"] if e["split"] == "validation"}
    dictionary = {shape_key(item["texture"], item["cells"]): item["id"] for item in annotations["dictionary"]}
    report = {"checkpoint_sha256": digest(args.checkpoint), "seed_sha256": digest(seed_path),
              "test_used": False, "steps": args.steps, "overlap_disabled": args.disable_overlap, "paths": []}
    for seed in seeds["seeds"]:
        file, anchor = seed["file"], seed["frame"]
        if args.anchor is not None and anchor != args.anchor:
            continue
        source = ROOT / "data" / file
        if not source.resolve().is_relative_to((ROOT / "data").resolve()) or digest(source) != entries[file]["sha256"]:
            raise ValueError("capture path or hash mismatch")
        frames, actions, metadata = read_episode(source)
        if anchor < 32 or anchor + args.steps > len(actions) or ACTIONS[actions[anchor - 1]] != "noop":
            raise ValueError("invalid released-button audit anchor")
        label_dir = annotation_path.parent
        label_path = (label_dir / records[file]["labels_file"]).resolve()
        if not label_path.is_relative_to(label_dir.resolve()):
            raise ValueError("annotation path outside dataset")
        with np.load(label_path, allow_pickle=False) as archive:
            if json.loads(str(archive["metadata"]))["source_sha256"] != entries[file]["sha256"]:
                raise ValueError("annotation lineage mismatch")
            anchor_labels = {k: archive[k][anchor].copy() for k in ("cells", "shape_id", "texture", "track", "evidence")}
        if anchor_labels["shape_id"] < 0 or not np.array_equal(anchor_labels["cells"], seed["initial_cells"]):
            raise ValueError("object annotation disagrees with audit anchor")
        prefix, unknown = encode_frames(frames[:anchor + 1], book)
        if unknown:
            raise ValueError("unknown tiles in audit prefix")
        prefix = torch.from_numpy(prefix).long()
        initial = None
        for t in range(anchor):
            initial = model.advance_observed(prefix[t:t+1], torch.tensor([int(actions[t])]), initial)
        for name in ("recorded", "down"):
            if args.branch is not None and name != args.branch:
                continue
            commands = (actions[anchor:anchor + args.steps] if name == "recorded"
                        else np.full(args.steps, ACTIONS.index("down")))
            images, proof = reference(frames, actions, metadata, anchor, commands, name == "recorded")
            codes, unknown = encode_frames(images, book)
            codes = torch.from_numpy(codes).long()
            original = {k: np.full((len(images), *v.shape), 0 if k == "evidence" else -1, dtype=v.dtype)
                        for k, v in anchor_labels.items()}
            for k, v in anchor_labels.items():
                original[k][0] = v
            labels, _ = extend_labels(images, original, dictionary)
            auto = codes[:1]
            teacher_state = auto_state = initial
            events, generated_images = recorded_path(model, auto, initial, anchor_labels["cells"], commands, "noop")
            rows = []
            for t, command in enumerate(commands):
                action = torch.tensor([int(command)])
                observed, target = codes[t:t+1], codes[t+1:t+2]
                results = {}
                spawn = None
                if labels["shape_id"][t] >= 0 and labels["evidence"][t] in (1, 2):
                    spawn = spawn_evidence(images[t], images[t + 1], labels["cells"][t])
                    if spawn is not None and shape_key(spawn["texture"], spawn["cells"]) not in dictionary:
                        spawn = None
                for key, tokens, state in (("autonomous", auto, auto_state),
                                           ("observed", observed, teacher_state),
                                           ("replace_pixels", observed, auto_state),
                                           ("replace_state", auto, teacher_state)):
                    result, prediction, next_state = inspect_prediction(
                        model, tokens, target, action, state, labels, t, observed=key == "observed")
                    if spawn is not None and key in ("autonomous", "observed"):
                        input_matches = torch.equal(tokens[:, :, 2:12], observed[:, :, 2:12])
                        pixels = decode_tokens(prediction, model.codebook)[0, 0].numpy()
                        result["spawn"] = (spawn_evidence(images[t], pixels, labels["cells"][t])
                                           if input_matches else None)
                    results[key] = result
                    if key == "autonomous":
                        next_auto, next_auto_state = prediction, next_state
                    elif key == "observed":
                        next_teacher_state = next_state
                if args.hud_attribution:
                    preview = torch.zeros_like(auto, dtype=torch.bool)
                    preview[:, :6, 13:19] = True
                    board = model.board_mask[None]
                    other_hud = ~(preview | board)
                    results["hud"] = {"input_preview_errors": int(((auto != observed) & preview).sum()),
                                      "input_other_errors": int(((auto != observed) & other_hud).sum())}
                    for region, mask in (("preview", preview), ("other", other_hud), ("all", ~board)):
                        patched = torch.where(mask, observed, auto)
                        for state_name, state in (("auto", auto_state), ("observed", teacher_state)):
                            detail = model.scene_details(patched, action, state)
                            prediction = model.decode_scene(patched, detail, hard=True)
                            error = int((prediction[:, :, 2:12] != target[:, :, 2:12]).sum())
                            results["hud"][f"{region}/{state_name}_state"] = error
                rows.append({"step": t, "action": ACTIONS[int(command)], "input_board_errors": int(
                    (auto[:, :, 2:12] != observed[:, :, 2:12]).sum()), "reference_spawn": spawn,
                    "reference_board_changed": bool((images[t, :, 8:48] != images[t + 1, :, 8:48]).any()),
                    "autonomous_board_changed": bool((auto[:, :, 2:12] != next_auto[:, :, 2:12]).any()),
                    "reference_preview_texture": preview_texture(images[t + 1]),
                    "autonomous_preview_texture": preview_texture(generated_images[t + 1]),
                    "reference_preview_changed": bool((images[t, :24, 52:76] != images[t + 1, :24, 52:76]).any()),
                    "autonomous_preview_changed": bool((generated_images[t][:24, 52:76] !=
                                                          generated_images[t + 1][:24, 52:76]).any()),
                    "preview_pixels_exact": bool(np.array_equal(images[t + 1, :24, 52:76],
                                                                generated_images[t + 1][:24, 52:76])),
                    **results})
                auto, auto_state, teacher_state = next_auto, next_auto_state, next_teacher_state
            report["paths"].append({"file": file, "anchor": anchor, "name": name, "proof": proof,
                                    "unknown_reference_tiles": unknown, "first_failure": summarize(events)["first_failure"],
                                    "rows": rows})
    if not report["paths"]:
        parser.error("no matching proven validation path")
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
