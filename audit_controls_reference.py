"""Counterfactual ROM oracle for released-button neural audits; validation only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_controls import (BUTTONS, branch, compare_controls, direction_oracle,
                            transition_oracle)
from collect import ScreenController, counters
from evaluate import comparison_image
from model import ACTIONS
from object_annotations import classify_board, shape_key
from object_annotations_v2 import extend_labels
from run import ROOT, read_episode, step
from token_model import encode_frames, load_tile_model


class ReplayMismatch(RuntimeError):
    def __init__(self, index, expected, actual):
        self.evidence = {"frame": index, "different_pixels": int(np.count_nonzero(expected != actual)) if expected.shape == actual.shape else None,
                         "expected_shape": list(expected.shape), "actual_shape": list(actual.shape),
                         "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                         "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest()}
        super().__init__(f"reference replay diverged: {self.evidence}")


def verify_frame(index, expected, actual):
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        raise ReplayMismatch(index, expected, actual)


def replay_prefix(game, frames, actions, metadata, index, step_fn=step,
                  counter_fn=counters, controller_factory=ScreenController):
    rng = np.random.default_rng(metadata["seed"])
    boot = int(rng.integers(240, 1000))
    if boot != metadata["boot_frames"]:
        raise ValueError("boot metadata does not match collection seed")
    controller = controller_factory(metadata["seed"], metadata["profile"])
    game.set_emulation_speed(0); game.tick(boot, True)
    step_fn(game, "a")
    for _ in range(10):
        frame = step_fn(game, "noop")
        if counter_fn(game)["level"] == 1:
            break
    else:
        raise ValueError("reference boot has no verified displayed level1")
    warmup = int(rng.integers(24, 36 if metadata["profile"] == "topout" else 90))
    if warmup != metadata["warmup_steps"]:
        raise ValueError("warmup metadata does not match collection seed")
    for _ in range(warmup):
        frame = step_fn(game, controller.choose(frame))
    verify_frame(0, frames[0], frame)
    for t in range(index):
        frame = step_fn(game, ACTIONS[int(actions[t])])
        verify_frame(t + 1, frames[t + 1], frame)
    return {"verified_frames": index + 1, "boot_frames": boot, "warmup_steps": warmup}


def reference_branch(factory, frames, actions, metadata, index, button, steps):
    game = factory()
    try:
        replay = replay_prefix(game, frames, actions, metadata, index)
        images = [frames[index].copy()]
        for t in range(steps):
            action = button if button not in ("a", "b") or t == 0 else "noop"
            images.append(step(game, action))
        return np.stack(images), replay
    finally:
        game.stop(save=False)


def reference_metrics(images, seed, dictionary, button):
    count = len(images)
    initial = {"cells": np.full((count, 4, 2), -1, dtype=np.int16), "texture": np.full(count, -1, dtype=np.int8),
               "shape_id": np.full(count, -1, dtype=np.int32), "track": np.full(count, -1, dtype=np.int32),
               "evidence": np.zeros(count, dtype=np.uint8)}
    for key in initial:
        initial[key][0] = seed[key]
    labels, propagation = extend_labels(images, initial, dictionary)
    events, continuous, rotations = [], True, 0
    original_texture = int(seed["texture"])
    for t in range(count - 1):
        old = labels["cells"][t] if labels["shape_id"][t] >= 0 else None
        new = labels["cells"][t + 1] if labels["shape_id"][t + 1] >= 0 else None
        before, after = classify_board(images[t]), classify_board(images[t + 1])
        event = transition_oracle(before, after, old, new, -1, np.any(before < -1) or np.any(after < -1))
        if new is not None and int(labels["texture"][t + 1]) != original_texture:
            event = {"status": "unknown", "reason": "reference-spawn-or-style-change"}
        if event["status"] == "motion":
            reason, rotations = direction_oracle(old, new, button, rotations)
            if reason:
                event = {"status": "violation", "reason": reason}
        event["step"] = t
        events.append(event); continuous &= event["status"] == "motion"
    endpoint = labels["cells"][-1].tolist() if labels["shape_id"][-1] >= 0 else None
    return {"events": events, "continuous": bool(continuous), "endpoint": endpoint,
            "orientation_changes": rotations, "annotation_propagation": propagation}


def conditional_controls(reference, neural):
    """A neural command can fail this comparison only when the ROM validates it."""
    result = {}
    for command, oracle in reference.items():
        predicted = neural[command]
        if oracle["status"] != "pass":
            status = "inconclusive"
        elif predicted["status"] == "inconclusive":
            status = "inconclusive"
        else:
            status = predicted["status"]
        result[command] = {"status": status, "reference": oracle, "neural": predicted}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True, help="preserved released-button audit JSON")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--annotations", type=Path, default=Path("data/object-annotations-v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit = json.loads(args.audit.read_bytes())
    manifest_bytes = (args.data / "manifest.json").read_bytes()
    annotation_bytes = (args.annotations / "object-audit.json").read_bytes()
    manifest, annotations = json.loads(manifest_bytes), json.loads(annotation_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    weight_digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if (audit["test_used"] or annotations["test_used"] or audit["steps"] != 8 or audit["manifest_sha256"] != digest
            or annotations["manifest_sha256"] != digest or audit["checkpoint_sha256"] != weight_digest
            or audit["annotations_sha256"] != hashlib.sha256(annotation_bytes).hexdigest()):
        raise ValueError("audit/weights/annotations lineage mismatch")
    model = load_tile_model(args.checkpoint).eval()
    entries = {e["file"]: e for e in manifest["episodes"] if e["split"] == "validation"}
    records = {e["file"]: e for e in annotations["episodes"] if e["split"] == "validation"}
    dictionary = {shape_key(item["texture"], item["cells"]): item["id"] for item in annotations["dictionary"]}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"test_used": False, "checkpoint_sha256": weight_digest, "manifest_sha256": digest,
              "released_audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(), "seeds": [],
              "status": "running", "limitations": "offline ROM reference only; no ROM pixels fed to neural branches after seed; command response alone does not prove playability"}
    try:
        for captured_seed in audit["seeds"]:
            entry = entries[captured_seed["file"]]
            source = (args.data / entry["file"]).resolve()
            record = records[entry["file"]]
            labels_path = (args.annotations / record["labels_file"]).resolve()
            if (not source.is_relative_to(args.data.resolve()) or not labels_path.is_relative_to(args.annotations.resolve())
                    or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]):
                raise ValueError("unsafe path or capture digest mismatch")
            frames, actions, metadata = read_episode(source)
            index = captured_seed["frame"]
            if index < 32 or int(actions[index - 1]) != 0:
                raise ValueError("seed is not a released-button validation seed")
            with np.load(labels_path, allow_pickle=False) as archive:
                if json.loads(str(archive["metadata"]))["source_sha256"] != entry["sha256"]:
                    raise ValueError("label source mismatch")
                seed = {key: archive[key][index].copy() for key in ("cells", "texture", "shape_id", "track", "evidence")}
            if seed["shape_id"] < 0 or seed["evidence"] not in (1, 2):
                raise ValueError("seed lacks complete visual proof")
            context = metadata["context"]
            rom = ROOT / context["rom"]["file"]
            if hashlib.sha256(rom.read_bytes()).hexdigest() != context["rom"]["sha256"]:
                raise ValueError("capture ROM hash mismatch")
            from pyboy import PyBoy
            def factory():
                return PyBoy(str(rom), window="null", sound_emulated=False,
                             color_palette=(0xFFFFFF, 0xAAAAAA, 0x555555, 0x000000))
            codes, unknown = encode_frames(frames[:index + 1], model.codebook.numpy())
            if unknown:
                raise ValueError("unknown capture visual tiles")
            state = None
            with torch.inference_mode():
                for t in range(index):
                    state = model.advance_observed(torch.from_numpy(codes[t].copy())[None].long(), torch.tensor([int(actions[t])]), state)
            current = torch.from_numpy(codes[index].copy())[None].long()
            neural, reference, replays = {}, {}, {}
            for button in BUTTONS:
                rom_images, replays[button] = reference_branch(factory, frames, actions, metadata, index, button, 8)
                reference[button] = reference_metrics(rom_images, seed, dictionary, button)
                neural[button], images = branch(model, current, state, seed["cells"], button, 8)
                comparison_image(np.stack(images), rom_images, args.output / f"seed-{len(report['seeds']):02d}-{button}.png")
            ref_controls = compare_controls(reference, seed["cells"])
            neural_controls = compare_controls(neural, seed["cells"])
            report["seeds"].append({"file": entry["file"], "frame": index, "replays": replays,
                                    "previous_observed_actions": actions[index - 8:index].tolist(),
                                    "reference": reference, "neural": neural,
                                    "controls": conditional_controls(ref_controls, neural_controls)})
        report["status"] = "completed-reference-audit-gameplay-unverified"
    except ReplayMismatch as error:
        report["status"] = "blocked-reference-replay-mismatch"
        report["error"] = error.evidence
        report["failed_seed"] = {"file": captured_seed["file"], "frame": index, "button": button}
        raise
    except Exception as error:
        report["status"] = "blocked-reference-audit-error"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        (args.output / "reference-controls-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output), "status": report["status"], "seeds": len(report["seeds"])}))


if __name__ == "__main__":
    main()
