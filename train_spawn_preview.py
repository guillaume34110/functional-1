"""Fit only the preview head on proved spawns from parent training captures.

The export contains model weights and audit provenance. It has no resume optimizer
or stream state because it is an independent fit of one frozen-model head.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from run import ROOT, read_episode
from scene_model import LearnedOverlapSceneDynamics, spawn_preview_scene_weights
from token_model import encode_frames
from train_scene import label_mask, spawn_preview_targets


EPOCHS, BATCH, LR, SEED = 30, 64, 0.003, 39


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_captures(config, book):
    return observed_captures(config, book, "train")


def observed_captures(config, book, split):
    if split not in ("train", "validation"):
        raise ValueError("reserved captures are excluded")
    directory = ROOT / "data"
    annotation_dir = directory / "object-annotations-v2"
    manifest_path = directory / "manifest.json"
    annotation_path = annotation_dir / "object-audit.json"
    manifest, annotation = [json.loads(path.read_bytes()) for path in (manifest_path, annotation_path)]
    if (manifest["status"] != "complete" or annotation["test_used"]
            or annotation["manifest_sha256"] != config["manifest_sha256"]
            or sha(manifest_path) != config["manifest_sha256"]
            or sha(annotation_path) != config["annotations_sha256"]):
        raise ValueError("checkpoint, manifest, or annotation lineage mismatch")
    records = {entry["file"]: entry for entry in annotation["episodes"] if entry["split"] == split}
    for entry in manifest["episodes"]:
        if entry["split"] != split:
            continue
        record = records[entry["file"]]
        source = (directory / entry["file"]).resolve()
        labels_path = (annotation_dir / record["labels_file"]).resolve()
        if (not source.is_relative_to(directory.resolve()) or not labels_path.is_relative_to(annotation_dir.resolve())
                or sha(source) != entry["sha256"]):
            raise ValueError("capture digest or path mismatch")
        frames, actions, metadata = read_episode(source)
        if metadata["source"] != "capture":
            raise ValueError("expected a real training capture")
        with np.load(labels_path, allow_pickle=False) as labels:
            if json.loads(str(labels["metadata"]))["source_sha256"] != entry["sha256"]:
                raise ValueError("label source digest mismatch")
            arrays = {key: torch.from_numpy(labels[key]).long() for key in ("cells", "shape_id", "evidence")}
        if (arrays["cells"].shape != (len(frames), 4, 2)
                or any(arrays[key].shape != (len(frames),) for key in ("shape_id", "evidence"))):
            raise ValueError("invalid training annotation dimensions")
        tokens, unknown = encode_frames(frames, book)
        if unknown:
            raise ValueError("frozen codec cannot encode training capture")
        yield entry["file"], torch.from_numpy(tokens).long(), torch.from_numpy(actions).long(), arrays


@torch.no_grad()
def cache_examples(model, captures):
    features, joints, modes, wanted, counts = [], [], [], [], {}
    witness = None
    for name, tokens, actions, labels in captures:
        if torch.any(labels["shape_id"] >= len(model.object_shapes)):
            raise ValueError("training shape outside checkpoint dictionary")
        current_labels = {key: value[:-1, None] for key, value in labels.items()}
        positive, mask = label_mask(current_labels, 0, torch.ones(len(actions), dtype=torch.bool))
        rows, targets = spawn_preview_targets(model, tokens[:-1], tokens[1:], mask, positive, labels["shape_id"][:-1])
        events = dict(zip(rows.tolist(), targets.unbind()))
        counts[name] = len(events)
        if not events:
            continue
        state = None
        for t in range(max(events) + 1):
            frame, button = tokens[t:t + 1], actions[t:t + 1]
            if t not in events:
                state = model.advance(frame, button, state)
                continue
            if witness is None:
                witness = (frame, button, state)
            details = LearnedOverlapSceneDynamics.scene_details(model, frame, button, state)
            state = details["state"]
            preview = model.embedding(frame[:, :6, 13:19]).permute(0, 3, 1, 2)
            features.append(F.adaptive_avg_pool2d(preview, (2, 2)).flatten(1))
            joints.append(details["next"]["joint"])
            modes.append(details["mode"].softmax(1)[:, 1, None, None])
            wanted.append(events[t])
    if not wanted:
        raise ValueError("no proved parent train spawn transitions")
    return torch.cat(features), torch.cat(joints), torch.cat(modes), torch.stack(wanted), counts, witness


def adjusted_joint(head, features, joints, compose):
    adjusted = joints + compose * head(features)[:, :, None]
    normalizer = adjusted.flatten(1).logsumexp(1) - joints.flatten(1).logsumexp(1)
    return (adjusted - normalizer[:, None, None]).flatten(1)


@torch.no_grad()
def metrics(head, features, joints, compose, targets):
    joint = adjusted_joint(head, features, joints, compose)
    choices = joint.argmax(1)
    return {"nll": float(F.nll_loss(joint, targets)), "joint_correct": int((choices == targets).sum()),
            "shape_correct": int((choices // 180 == targets // 180).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory for model.pt and result.json")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = spawn_preview_scene_weights(checkpoint["model"]).eval().requires_grad_(False)
    head = model.spawn_preview_head.requires_grad_(True)
    features, joints, compose, targets, counts, witness = cache_examples(
        model, training_captures(config, model.codebook.numpy()))
    with torch.no_grad():
        torch.testing.assert_close(adjusted_joint(head, features, joints, compose), joints.flatten(1), rtol=0, atol=0)
        if not torch.isfinite(joints.flatten(1).gather(1, targets[:, None])).all():
            raise ValueError("proved spawn target outside valid model poses")
    before = metrics(head, features, joints, compose, targets)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        for index in torch.randperm(len(targets), generator=generator).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            joint = adjusted_joint(head, features[index], joints[index], compose[index])
            loss = F.nll_loss(joint, targets[index])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite preview fit")
            loss.backward()
            optimizer.step()
    after = metrics(head, features, joints, compose, targets)
    with torch.no_grad():
        runtime_joint = model.scene_details(*witness)["next"]["joint"].flatten(1)
        cached_joint = adjusted_joint(head, features[:1], joints[:1], compose[:1])
        torch.testing.assert_close(runtime_joint, cached_joint, rtol=0, atol=0)
    weights = model.state_dict()
    for name, tensor in checkpoint["model"].items():
        # Compare raw storage bytes, including floating point sign bits.
        torch.testing.assert_close(weights[name].reshape(-1).view(torch.uint8),
                                   tensor.reshape(-1).view(torch.uint8), rtol=0, atol=0)
    report = {"source_checkpoint_sha256": sha(args.checkpoint), "source_step": checkpoint["step"],
              "script_sha256": sha(Path(__file__)), "manifest_sha256": config["manifest_sha256"],
              "annotations_sha256": config["annotations_sha256"], "train_examples": len(targets),
              "train_examples_by_capture": counts, "validation_used": False, "test_used": False,
              "epochs": EPOCHS, "batch": BATCH, "lr": LR, "seed": SEED,
              "parameters_fitted": sum(p.numel() for p in head.parameters()),
              "before": before, "after": after, "optimizer_exported": False}
    args.output.mkdir(parents=True)
    torch.save({"model": weights, "step": checkpoint["step"], "config": {
        **config, "model_family": type(model).__name__, "spawn_preview_head_only_fit": report
    }}, args.output / "model.pt")
    report["checkpoint_sha256"] = sha(args.output / "model.pt")
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
