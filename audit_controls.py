"""Offline validation oracle for neural button response; never imported by runtime."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from object_annotations import PATTERNS, classify_board
from run import read_episode
from token_model import decode_tokens, encode_frames, load_tile_model

BUTTONS = {"noop": 0, "left": 1, "right": 2, "down": 3, "a": 4, "b": 5}


def normalized(cells):
    cells = np.asarray(cells)
    return sorted(map(tuple, (cells - cells.min(0)).tolist()))


def erase(board, cells, blank):
    result = board.copy()
    for y, x in cells:
        result[y, x] = blank
    return result


def supported(background, cells, blank):
    occupied = set(map(tuple, cells))
    return any(y == 17 or ((y + 1, x) not in occupied and background[y + 1, x] != blank)
               for y, x in cells)


def transition_oracle(before, after, old, new, blank, invalid=False):
    """Classify only visible geometry. Missing/ambiguous objects are never passes."""
    if invalid or old is None or new is None:
        return {"status": "unknown", "reason": "invalid-pattern-or-missing-complete-object"}
    background = erase(before, old, blank)
    after_background = erase(after, new, blank)
    collision = any(background[y, x] != blank for y, x in new)
    if np.array_equal(after_background, before):
        if supported(background, old, blank):
            return {"status": "deposit", "collision": collision, "reason": "old-canvas-retained-and-old-object-supported"}
        return {"status": "violation", "collision": collision, "reason": "unsupported-deposit"}
    if collision:
        return {"status": "violation", "collision": True, "reason": "proposed-overlay-intersects-visible-background"}
    if not np.array_equal(after_background, background):
        if supported(background, old, blank):
            return {"status": "unknown", "reason": "supported-boundary-or-clear-outside-fall-coverage"}
        return {"status": "violation", "collision": False, "reason": "unexplained-background-mutation"}
    if np.max(np.abs(np.asarray(new).mean(0) - np.asarray(old).mean(0))) > 4:
        return {"status": "unknown", "reason": "discontinuous-object-possible-spawn"}
    return {"status": "motion", "collision": False}


def compare_controls(branches, initial):
    def usable(name):
        branch = branches[name]
        return branch["continuous"] and branch["endpoint"] is not None
    result = {}
    if all(usable(name) for name in ("left", "noop", "right")):
        centers = [np.asarray(branches[name]["endpoint"]).mean(0)[1] for name in ("left", "noop", "right")]
        result["horizontal"] = {"status": "pass" if centers[0] < centers[1] < centers[2] else "fail", "centers": centers}
    else:
        result["horizontal"] = {"status": "inconclusive"}
    if usable("down") and usable("noop"):
        rows = [np.asarray(branches[name]["endpoint"]).mean(0)[0] for name in ("down", "noop")]
        result["down"] = {"status": "pass" if rows[0] > rows[1] else "fail", "rows": rows}
    else:
        result["down"] = {"status": "inconclusive"}
    for name in ("a", "b"):
        if usable(name) and usable("noop"):
            orientation = normalized(branches[name]["endpoint"])
            changed = orientation != normalized(initial) and orientation != normalized(branches["noop"]["endpoint"])
            result[name] = {"status": "pass" if changed else "fail", "normalized_endpoint": orientation}
        else:
            result[name] = {"status": "inconclusive"}
    return result


def select_seed(frames, annotation, actions):
    for index in range(32, len(frames)):
        if int(actions[index - 1]) != BUTTONS["noop"]:
            continue
        if annotation["shape_id"][index] < 0 or annotation["evidence"][index] not in (1, 2):
            continue
        cells = annotation["cells"][index]
        low, high = cells.min(0), cells.max(0)
        if low[0] < 1 or high[0] > 7 or low[1] < 3 or high[1] > 6:
            continue
        if high[0] - low[0] == 1 and high[1] - low[1] == 1:  # Rotation-invariant square excluded.
            continue
        board = classify_board(frames[index])
        if np.any(board < -1) or not np.all(board[cells[:, 0], cells[:, 1]] == annotation["texture"][index]):
            continue
        background = erase(board, cells, -1)
        if np.all(background[low[0]:high[0] + 6, low[1] - 2:high[1] + 3] == -1):
            return index
    return None


def direction_oracle(old, new, button, rotation_changes, previous_button=None):
    """Evaluate only continuous, same-style visual motion; never alter generation."""
    rotated = normalized(old) != normalized(new)
    dx = int(np.asarray(new)[:, 1].min() - np.asarray(old)[:, 1].min())
    if button in ("a", "b"):
        rotation_changes += int(rotated)
        return ("multiple-orientation-changes-after-one-press" if rotation_changes > 1 else None), rotation_changes
    if rotated:
        return "uncommanded-orientation-change", rotation_changes
    if button in ("noop", "down") and dx != 0:
        if (dx == -1 and previous_button == "left") or (dx == 1 and previous_button == "right"):
            return "possible-delayed-horizontal", rotation_changes
        return "uncommanded-horizontal-motion", rotation_changes
    if (button == "left" and dx > 0) or (button == "right" and dx < 0):
        return "horizontal-motion-opposes-command", rotation_changes
    return None, rotation_changes


def observed_object(model, details, rendered):
    mask, color = model._hard_object(details["next"])
    cells = torch.nonzero(mask[0, :, 2:12]).cpu().numpy()
    if len(cells) != 4 or not torch.all(rendered[0][mask[0]] == color[0]):
        return None, None
    return cells, int(color[0])


@torch.inference_mode()
def branch(model, tokens, state, cells, button, steps, previous_button=None):
    current, old = tokens.clone(), cells.copy()
    state = None if state is None else state.clone()
    images = [decode_tokens(current, model.codebook)[0, 0].numpy()]
    events, continuous, original_color = [], True, int(current[0, old[0, 0], old[0, 1] + 2])
    rotation_changes = 0
    for step in range(steps):
        action = BUTTONS[button] if button not in ("a", "b") or step == 0 else 0
        details = model.scene_details(current, torch.tensor([action]), state)
        prediction = model.decode_scene(current, details, hard=True)
        state = model.rendered_state(details, prediction)
        new, color = observed_object(model, details, prediction)
        pixels = decode_tokens(prediction, model.codebook)[0, 0].numpy()
        before, after = current[0, :, 2:12].numpy(), prediction[0, :, 2:12].numpy()
        invalid = np.any(classify_board(pixels) < -1)
        event = transition_oracle(before, after, old, new, model.blank_id, invalid)
        if new is not None and color != original_color:
            event = {"status": "unknown", "reason": "object-style-changed-possible-spawn", "previous_oracle": event}
        if event["status"] == "motion" and color == original_color:
            reason, rotation_changes = direction_oracle(old, new, button, rotation_changes, previous_button)
            if reason is not None:
                event = {"status": "unknown" if reason == "possible-delayed-horizontal" else "violation", "reason": reason}
        event.update({"step": step, "button": action, "mode": int(details["mode"].argmax(1)),
                      "cells": None if new is None else new.tolist(), "style": color, "previous_button": previous_button})
        events.append(event)
        continuous &= event["status"] == "motion"
        images.append(pixels)
        current, old = prediction, new
        previous_button = next((name for name, value in BUTTONS.items() if value == action), None)
    endpoint = None if old is None else old.tolist()
    displacement = None if old is None else (np.asarray(old).mean(0) - np.asarray(cells).mean(0)).tolist()
    return {"events": events, "continuous": bool(continuous), "endpoint": endpoint,
            "displacement_yx": displacement, "orientation_changes": rotation_changes}, images


def screenshot(branch_images, path):
    sheet = Image.new("RGB", (len(branch_images["noop"]) * 160, len(BUTTONS) * 166), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (button, frames) in enumerate(branch_images.items()):
        draw.text((3, row * 166), button, fill="black")
        for column, frame in enumerate(frames):
            image = Image.fromarray((frame * 85).astype(np.uint8)).convert("RGB")
            sheet.paste(image.resize((160, 144), Image.Resampling.NEAREST), (column * 160, row * 166 + 20))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--annotations", type=Path, default=Path("data/object-annotations-v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.steps <= 8:
        parser.error("steps must be1..8")
    torch.set_num_threads(2)
    model = load_tile_model(args.checkpoint).eval()
    manifest_bytes = (args.data / "manifest.json").read_bytes()
    annotation_bytes = (args.annotations / "object-audit.json").read_bytes()
    manifest, annotations = json.loads(manifest_bytes), json.loads(annotation_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if annotations["test_used"] or annotations["manifest_sha256"] != digest:
        raise ValueError("annotation lineage differs")
    dictionary = sorted(annotations["dictionary"], key=lambda item: item["id"])
    if not torch.equal(torch.tensor([item["cells"] for item in dictionary]), model.object_shapes):
        raise ValueError("annotation shapes differ from model")
    for item, tile in zip(dictionary, model.object_tiles, strict=True):
        if not np.array_equal(model.codebook[tile].numpy(), PATTERNS[item["texture"]]):
            raise ValueError("annotation styles differ from model")
    records = {item["file"]: item for item in annotations["episodes"]}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "manifest_sha256": digest, "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
              "test_used": False, "seeds": [], "selection_failures": [], "steps": args.steps,
              "seed_criterion": "first geometric candidate per validation episode after32 with actions[index-1]==noop; no neural-outcome selection",
              "limitations": ["offline visible-geometry oracle; never runtime rules or model inputs",
                              "counterfactual branches have no ROM reference; controls alone do not prove playability",
                              "deposits, spawns, missing objects, invalid graphics or violations make endpoint controls inconclusive"]}
    for entry in manifest["episodes"]:
        if entry["split"] != "validation":
            continue
        source = (args.data / entry["file"]).resolve()
        record = records[entry["file"]]
        label_path = (args.annotations / record["labels_file"]).resolve()
        if (not source.is_relative_to(args.data.resolve()) or not label_path.is_relative_to(args.annotations.resolve())
                or record["split"] != "validation" or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]):
            raise ValueError("validation source mismatch")
        frames, actions, _ = read_episode(source)
        with np.load(label_path, allow_pickle=False) as archive:
            if json.loads(str(archive["metadata"]))["source_sha256"] != entry["sha256"]:
                raise ValueError("annotation source mismatch")
            labels = {key: archive[key].copy() for key in ("cells", "shape_id", "evidence", "texture")}
        index = select_seed(frames, labels, actions)
        if index is None:
            report["selection_failures"].append({"file": entry["file"], "reason": "no-released-proven-centered-non-square-object-with-clearance"})
            continue
        codes, unknown = encode_frames(frames[:index + 1], model.codebook.numpy())
        if unknown:
            raise ValueError("unknown observed visual tiles")
        state = None
        with torch.inference_mode():
            for t in range(index):
                state = model.advance_observed(torch.from_numpy(codes[t].copy())[None].long(), torch.tensor([int(actions[t])]), state)
            current = torch.from_numpy(codes[index].copy())[None].long()
            branches, images = {}, {}
            for button in BUTTONS:
                prior = next((name for name, value in BUTTONS.items() if value == int(actions[index - 1])), None)
                branches[button], images[button] = branch(model, current, state, labels["cells"][index], button, args.steps, prior)
        seed = {"file": entry["file"], "frame": index, "initial_cells": labels["cells"][index].tolist(),
                "previous_observed_actions": [int(action) for action in actions[index - 8:index]],
                "branches": branches, "controls": compare_controls(branches, labels["cells"][index])}
        report["seeds"].append(seed)
        screenshot(images, args.output / f"seed-{len(report['seeds']):02d}.png")
    (args.output / "controls-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output / "controls-audit.json"), "seeds": len(report["seeds"]),
                      "selection_failures": len(report["selection_failures"]), "test_used": False}))


if __name__ == "__main__":
    main()
