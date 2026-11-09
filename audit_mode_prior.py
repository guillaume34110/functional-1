"""Fixed no-fit mode-prior ablation; heuristic equal-component reference, not calibration."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_long import recorded_path, save_samples, summarize
from intervention_training import BUTTONS
from run import ROOT
from token_model import load_tile_model
from train_scene import load_action_interventions, load_scene_data, rendered_teacher
from train_tokens import measure


def prior_offset(mode_weights, strength):
    if strength not in (0., .5, 1.):
        raise ValueError("only the three predefined strengths0,.5,1 are permitted")
    parent = torch.tensor(mode_weights["parent"], dtype=torch.float64)
    branches = torch.tensor(mode_weights["interventions"], dtype=torch.float64)
    if parent.shape != (3,) or branches.shape != (3,) or not torch.isfinite(parent).all() or not torch.isfinite(branches).all() or torch.any(parent <= 0) or torch.any(branches <= 0):
        raise ValueError("mode class weights must be three positive finite values per corpus")
    reference = (parent + branches) / 2
    # Common logit translation does not affect probabilities or argmax; keep copy at0.
    return strength * (reference.log() - reference[0].log())


class ModePrior:
    """Read-only model view; change only the returned mode logits, never recurrent state."""
    def __init__(self, base, offset):
        self.base = base
        self.offset = offset.detach().clone()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def train(self, mode=True):
        self.base.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def scene_details(self, tokens, actions, state=None):
        original = self.base.scene_details(tokens, actions, state)
        result = dict(original)
        result["mode"] = original["mode"] - self.offset.to(original["mode"])
        return result

    def forward(self, tokens, actions, state=None):
        details = self.scene_details(tokens, actions, state)
        return self.base.decode_scene(tokens, details, hard=False), details["state"]

    def __call__(self, tokens, actions, state=None):
        return self.forward(tokens, actions, state)

    def predict_tokens(self, tokens, actions, state=None):
        details = self.scene_details(tokens, actions, state)
        return self.base.decode_scene(tokens, details, hard=True), details["state"]

    def predict_frame(self, frame, action, state=None):
        # Bind the inherited codec bridge to this view, so its self.predict_tokens dispatches here.
        return type(self.base).predict_frame(self, frame, action, state)


def mode_trace(events):
    counts = Counter(event["mode"] for event in events)
    longest = run = 0
    for event in events:
        run = run + 1 if event["mode"] == 2 else 0
        longest = max(longest, run)
    return {"mode_counts": {name: counts[index] for index, name in enumerate(("copy", "compose", "rewrite"))},
            "longest_rewrite_run": longest}


def checkpoint_comparison(teacher, rollout, checkpoint):
    reference = checkpoint.get("teacher", {})
    result = {}
    for key in ("addition_accuracy", "erasure_accuracy", "event_fpr"):
        expected, actual = reference.get(key), teacher.get(key)
        result[key] = {"checkpoint": expected, "lambda0": actual,
                       "delta": actual - expected if expected is not None and actual is not None else None}
    expected = checkpoint.get("validation", {}).get("tile_error")
    result["tile_error"] = {"checkpoint": expected, "lambda0": rollout["tile_error"],
                            "delta": rollout["tile_error"] - expected if expected is not None else None}
    result["pixel_change_counts_exact"] = teacher["counts"] == reference.get("counts")
    result["critical_counts_exact"] = teacher["critical_counts"] == reference.get("critical_counts")
    result["intervention_counts_exact"] = teacher["interventions"]["counts"] == reference.get("interventions", {}).get("counts")
    return result


@torch.inference_mode()
def long_paths(model, seed_report, episodes, annotations, entries, output):
    parents = {entry["file"]: (episodes[index], annotations[index]) for index, entry in enumerate(entries)}
    results = []
    for seed in seed_report["seeds"]:
        (codes, actions), labels = parents[seed["file"]]
        index = seed["frame"]
        cells = labels["cells"][index]
        if (index < 32 or index + 256 > len(actions) or int(actions[index - 1]) != 0
                or labels["shape_id"][index] < 0 or labels["evidence"][index] not in (1, 2)
                or not np.array_equal(cells, seed["initial_cells"])):
            raise ValueError("long-audit seed lacks its exact released whole-object prefix")
        state = None
        for t in range(index):
            state = model.advance(torch.from_numpy(codes[t].copy())[None].long(), torch.tensor([int(actions[t])]), state)
        current = torch.from_numpy(codes[index].copy())[None].long()
        item = {"file": seed["file"], "frame": index, "paths": {}}
        for name in ("recorded", "down"):
            commands = actions[index:index + 256] if name == "recorded" else np.full(256, 3, dtype=np.uint8)
            events, images = recorded_path(model, current, state, cells, commands)
            filename = f"seed-{len(results):02d}-{name}.npz"
            save_samples(images, events, output / filename)
            item["paths"][name] = {**summarize(events), **mode_trace(events), "events": events,
                                    "actions": commands.tolist(), "sampled_pixels": filename}
        results.append(item)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/states-checkpoint-v17.pt")
    parser.add_argument("--seed-report", type=Path, default=ROOT / "docs/long-audit-seeds-v15.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    started = time.monotonic()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    raw, episodes, annotations, _, _, digest, annotation_digest = load_scene_data(checkpoint)
    model = load_tile_model(args.checkpoint).eval()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    tables, addon_digest = load_action_interventions(checkpoint, model, episodes, annotations, digest, annotation_digest)
    seed_bytes = args.seed_report.read_bytes()
    seeds = json.loads(seed_bytes)
    if (seeds["test_used"] or seeds["manifest_sha256"] != digest or seeds["annotations_sha256"] != annotation_digest
            or seeds["reference_status"] != "completed-reference-audit-gameplay-unverified" or len(seeds["seeds"]) != 5):
        raise ValueError("mode ablation seed provenance mismatch")
    for seed in seeds["seeds"]:
        proof = seed["reference_verified_frames"]
        if set(proof) != set(BUTTONS) or any(count != seed["frame"] + 1 for count in proof.values()):
            raise ValueError("mode ablation seed prefix is not completely verified")
    manifest = json.loads((ROOT / "data/manifest.json").read_bytes())
    entries = [entry for entry in manifest["episodes"] if entry["split"] == "validation"]
    expected = 192
    if len(tables["validation"]) != expected:
        raise ValueError("fixed ablation requires the same192 V2 validation anchors")
    args.output.mkdir(parents=True)
    report = {"status": "diagnostic-gameplay-unverified", "test_used": False, "fit_performed": False,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "checkpoint_config": checkpoint["config"],
              "manifest_sha256": digest, "annotations_sha256": annotation_digest, "interventions_sha256": addon_digest,
              "seed_report_sha256": hashlib.sha256(seed_bytes).hexdigest(),
              "controls_source_sha256": seeds["controls_audit_sha256"], "reference_source_sha256": seeds["verified_reference_sha256"],
              "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                          ("audit_mode_prior.py", "audit_long.py", "audit_controls.py", "train_scene.py", "intervention_training.py", "scene_model.py", "token_model.py")},
              "reference": "arithmetic mean of configured train parent/intervention class weights; equal-component heuristic, NOT true calibrated posterior or player prior",
              "limitations": "coupled pixel loss, masks, temporal weighting and unequal feature distributions invalidate a pure weighted-CE correction theorem; no runtime choice or gameplay success inferred",
              "strengths_predeclared": [0., .5, 1.], "validation_anchors": expected, "variants": []}
    for strength in (0., .5, 1.):
        offset = prior_offset(checkpoint["config"]["mode_class_weights"], strength)
        view = ModePrior(model, offset).eval()
        teacher = rendered_teacher(view, episodes["validation"], annotations["validation"], tables["validation"])
        rollout, _ = measure(view, episodes["validation"], raw["validation"])
        view.eval()
        output = args.output / f"lambda-{strength:g}"
        output.mkdir()
        traces = long_paths(view, seeds, episodes["validation"], annotations["validation"], entries, output)
        teacher_matrix = teacher["critical_counts"]["mode_confusion"]
        selected_modes = {name: sum(row[index] for row in teacher_matrix) for index, name in enumerate(("copy", "compose", "rewrite"))}
        variant = {"strength": strength, "offset_relative_to_copy": offset.tolist(), "teacher": teacher,
                   "validation": rollout, "teacher_known_mode_counts": selected_modes, "long_traces": traces}
        if strength == 0:
            variant["checkpoint_reproduction"] = checkpoint_comparison(teacher, rollout, checkpoint)
        report["variants"].append(variant)
        if any(not torch.equal(initial[name], value) for name, value in model.state_dict().items()):
            raise RuntimeError("mode ablation modified model weights or buffers")
        (args.output / "mode-prior-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    report["seconds"] = time.monotonic() - started
    (args.output / "mode-prior-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output / "mode-prior-audit.json"), "variants": 3, "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
