"""Bounded neural-only long rollout observations; no runtime rules or ROM calls."""
import argparse
from collections import Counter
import hashlib
from itertools import groupby
import json
from pathlib import Path

import numpy as np
import torch

from audit_controls import (BUTTONS, direction_oracle, normalized, observed_object,
                            transition_oracle)
from audit_spawn import preview_texture
from object_annotations import classify_board
from run import read_episode
from token_model import decode_tokens, encode_frames, load_tile_model


def summarize(events):
    counts, geometric = Counter(), Counter()
    first_unknown = first_violation = None
    valid_prefix = 0
    chain_known = True
    for index, event in enumerate(events):
        status = event["status"]
        counts[status] += 1
        geometric[event.get("previous_oracle", event)["status"]] += 1
        if status == "unknown" and first_unknown is None:
            first_unknown = {"step": index, "event": event}
        if status == "violation" and first_violation is None:
            first_violation = {"step": index, "event": event, "before_first_unknown": first_unknown is None}
        if chain_known and status in ("motion", "deposit"):
            valid_prefix += 1
        else:
            chain_known = False
    first_failure = next(({"step": i, "event": event} for i, event in enumerate(events)
                          if event["status"] not in ("motion", "deposit")), None)
    return {"counts": dict(counts), "geometric_counts_including_uncertain_boundaries": dict(geometric),
            "first_unknown": first_unknown, "first_violation": first_violation, "first_failure": first_failure,
            "known_geometry_prefix_steps": valid_prefix,
            "chain_status": "violation" if first_violation is not None and first_violation["before_first_unknown"] else "inconclusive" if first_unknown is not None else "no-observed-violation-gameplay-unverified"}


def possible_delayed_rotation(old, new, command, previous):
    return (int(command) == BUTTONS["noop"] and previous in (BUTTONS["a"], BUTTONS["b"])
            and normalized(old) != normalized(new))


def classify_identity(event, color, previous_color):
    """A collision-free visible deposit is the only proven identity boundary."""
    if event["status"] == "deposit":
        if event.get("collision", False):
            return {**event, "status": "violation", "reason": "deposit-overlay-collision"}, previous_color
        return {**event, "identity_reset": True}, color
    if color is not None and color != previous_color:
        return {"status": "unknown", "reason": "object-style-changed-possible-spawn", "previous_oracle": event}, previous_color
    return event, previous_color


@torch.inference_mode()
def recorded_path(model, tokens, state, cells, commands, previous_button=None, *, observer=None):
    current, old = tokens.clone(), cells.copy()
    state = None if state is None else state.clone()
    original_color = int(current[0, old[0, 0], old[0, 1] + 2])
    images = [decode_tokens(current, model.codebook)[0, 0].numpy()]
    events = []
    for index, command in enumerate(commands):
        details = model.scene_details(current, torch.tensor([int(command)]), state)
        prediction = model.decode_scene(current, details, hard=True)
        state = model.rendered_state(details, prediction)
        new, color = observed_object(model, details, prediction)
        pixels = decode_tokens(prediction, model.codebook)[0, 0].numpy()
        event = transition_oracle(current[0, :, 2:12].numpy(), prediction[0, :, 2:12].numpy(), old, new,
                                  model.blank_id, np.any(classify_board(pixels) < -1))
        event, original_color = classify_identity(event, color, original_color)
        if event["status"] == "motion":
            name = next((name for name, action in BUTTONS.items() if action == int(command)), None)
            previous = int(commands[index - 1]) if index else None
            if possible_delayed_rotation(old, new, command, previous):
                event = {"status": "unknown", "reason": "possible-delayed-rotation"}
            elif name is None:
                event = {"status": "unknown", "reason": "button-outside-direction-oracle-coverage"}
            else:
                reason, _ = direction_oracle(old, new, name, 0, previous_button)
                if reason:
                    event = {"status": "unknown" if reason == "possible-delayed-horizontal" else "violation", "reason": reason}
        event.update({"step": index, "button": int(command), "mode": int(details["mode"].argmax(1)),
                      "cells": None if new is None else new.tolist(), "style": color, "previous_button": previous_button})
        if "terminal" in details:
            event["terminal_phase"] = True
        events.append(event); images.append(pixels)
        if observer is not None and not observer(current, old, details, event):
            break
        current, old = prediction, new
        previous_button = next((name for name, action in BUTTONS.items() if action == int(command)), None)
    return events, images


def screen_progress(images):
    changes = [not np.array_equal(before[:, 8:48], after[:, 8:48])
               for before, after in zip(images, images[1:])]
    longest = max((sum(1 for _ in span) for changed, span in groupby(changes) if not changed), default=0)
    preview_changes = sum(not np.array_equal(before[:24, 52:76], after[:24, 52:76])
                          for before, after in zip(images, images[1:]))
    return {"board_changes": sum(changes), "longest_static_board": longest, "preview_changes": preview_changes,
            "readable_previews": sum(preview_texture(image) is not None for image in images[1:])}


def save_samples(images, events, path):
    indices = {min(index, len(images) - 1) for index in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)}
    for status in ("unknown", "violation", "deposit"):
        first = next((i + 1 for i, event in enumerate(events) if event["status"] == status), None)
        if first is not None:
            indices.update((first - 1, first))
    indices = sorted(indices)
    np.savez_compressed(path, indices=np.array(indices, dtype=np.int16), frames=np.stack([images[i] for i in indices]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit", "--seed-report", dest="audit", type=Path, required=True, help="preserved controls-audit.json or compact proven seed manifest")
    parser.add_argument("--reference", type=Path, help="already verified reference-controls-audit.json; unnecessary with compact proven seed manifest")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--annotations", type=Path, default=Path("data/object-annotations-v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest_bytes = (args.data / "manifest.json").read_bytes()
    annotation_bytes = (args.annotations / "object-audit.json").read_bytes()
    audit_bytes = args.audit.read_bytes()
    manifest, annotations, audit = map(json.loads, (manifest_bytes, annotation_bytes, audit_bytes))
    if args.reference is not None:
        reference_bytes = args.reference.read_bytes()
        reference = json.loads(reference_bytes)
        reference_digest = hashlib.sha256(reference_bytes).hexdigest()
    else:
        reference = {"test_used": audit["test_used"], "manifest_sha256": audit["manifest_sha256"],
                     "status": audit["reference_status"], "seeds": audit["seeds"]}
        reference_digest = audit["verified_reference_sha256"]
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    anno_digest = hashlib.sha256(annotation_bytes).hexdigest()
    if (any(report["test_used"] for report in (annotations, audit, reference))
            or any(report["manifest_sha256"] != digest for report in (annotations, audit, reference))
            or audit["annotations_sha256"] != anno_digest
            or reference["status"] != "completed-reference-audit-gameplay-unverified"):
        raise ValueError("seed/reference/dataset lineage mismatch")
    verified = {(seed["file"], seed["frame"]): seed for seed in reference["seeds"]}
    if len(audit["seeds"]) != 5 or {(seed["file"], seed["frame"]) for seed in audit["seeds"]} != set(verified):
        raise ValueError("audit must use the exact five already verified seeds")
    parents = {entry["file"]: entry for entry in manifest["episodes"] if entry["split"] == "validation"}
    records = {entry["file"]: entry for entry in annotations["episodes"] if entry["split"] == "validation"}
    model = load_tile_model(args.checkpoint).eval()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "diagnostic-gameplay-unverified", "test_used": False,
              "manifest_sha256": digest, "annotations_sha256": anno_digest,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "controls_audit_sha256": audit.get("controls_audit_sha256", hashlib.sha256(audit_bytes).hexdigest()),
              "seed_manifest_sha256": hashlib.sha256(audit_bytes).hexdigest(),
              "verified_reference_sha256": reference_digest, "seeds": [],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": "256 neural steps after observed prefix; rendered geometry only; unknown is absorbing for chain qualification; no ROM/model fallback; controls or deposits alone never prove playability"}
    for seed in audit["seeds"]:
        entry = parents[seed["file"]]
        index = seed["frame"]
        proof = verified[(seed["file"], index)]
        verified_frames = proof.get("reference_verified_frames") or {name: value["verified_frames"] for name, value in proof["replays"].items()}
        if set(verified_frames) != set(BUTTONS) or any(value != index + 1 for value in verified_frames.values()):
            raise ValueError("reference does not verify full observed prefix")
        source = (args.data / entry["file"]).resolve()
        label_path = (args.annotations / records[seed["file"]]["labels_file"]).resolve()
        if (not source.is_relative_to(args.data.resolve()) or not label_path.is_relative_to(args.annotations.resolve())
                or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]):
            raise ValueError("capture path/hash mismatch")
        frames, actions, _ = read_episode(source)
        if index < 32 or index + 256 > len(actions):
            raise ValueError("seed lacks full observed prefix or recorded256 actions")
        with np.load(label_path, allow_pickle=False) as archive:
            if json.loads(str(archive["metadata"]))["source_sha256"] != entry["sha256"]:
                raise ValueError("seed annotation source mismatch")
            cells = archive["cells"][index].copy()
            if archive["shape_id"][index] < 0 or archive["evidence"][index] not in (1, 2) or not np.array_equal(cells, seed["initial_cells"]):
                raise ValueError("seed has no matching whole-object proof")
        codes, unknown = encode_frames(frames[:index + 1], model.codebook.numpy())
        if unknown:
            raise ValueError("unknown observed tiles")
        state = None
        with torch.inference_mode():
            for t in range(index):
                state = model.advance_observed(torch.from_numpy(codes[t].copy())[None].long(), torch.tensor([int(actions[t])]), state)
        current = torch.from_numpy(codes[index].copy())[None].long()
        result = {"file": seed["file"], "frame": index, "paths": {}}
        for name in ("recorded", "down"):
            if name == "down":
                commands = np.full(256, BUTTONS["down"], dtype=np.uint8)
            else:
                commands = actions[index:index + 256].copy()
            prior = next((name for name, action in BUTTONS.items() if action == int(actions[index - 1])), None)
            events, images = recorded_path(model, current, state, cells, commands, prior)
            sample = f"seed-{len(report['seeds']):02d}-{name}.npz"
            save_samples(images, events, args.output / sample)
            result["paths"][name] = {**summarize(events), "screen_progress": screen_progress(images),
                                     "events": events, "actions": commands.tolist(), "sampled_pixels": sample}
        report["seeds"].append(result)
    (args.output / "long-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output / "long-audit.json"), "seeds": len(report["seeds"])}))


if __name__ == "__main__":
    main()
