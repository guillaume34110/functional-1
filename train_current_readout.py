"""Fit a current-pose appearance residual on parent train captures only."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scene_model import current_readout_scene_weights, scene_from_weights
from train_spawn_preview import sha, training_captures


SEED, EPOCHS, BATCH, LR = 50, 6, 64, 0.003


@torch.no_grad()
def cache_current_labels(model, captures):
    boards, shapes, poses, targets, counts = [], [], [], [], {}
    witness = None
    for name, tokens, actions, labels in captures:
        shape_id = labels["shape_id"][:-1]
        eligible = (shape_id >= 0) & ((labels["evidence"][:-1] == 1) | (labels["evidence"][:-1] == 2))
        times = eligible.nonzero(as_tuple=True)[0]
        counts[name] = len(times)
        if not len(times):
            continue
        cells = labels["cells"][times]
        origin = cells.amin(1)
        kind = shape_id[times]
        if torch.any((cells < 0) | (cells >= cells.new_tensor([18, 10]))) or torch.any(kind >= len(model.object_shapes)):
            raise ValueError("proved current label outside checkpoint geometry")
        pose = origin[:, 0] * 10 + origin[:, 1]
        if not model.valid_poses[kind, pose].all():
            raise ValueError("proved current label has an invalid pose")
        frames = tokens[times, :, 2:12]
        targets.append(kind * 180 + pose)
        boards.append(frames.byte())
        state, selected_hidden = None, []
        selected = iter(times.tolist())
        next_time = next(selected)
        for t in range(int(times[-1]) + 1):
            frame, action = tokens[t:t + 1], actions[t:t + 1]
            if t == next_time and witness is None:
                witness = (frame, action, state)
            state = model.advance(frame, action, state)
            if t == next_time:
                selected_hidden.append(state[:, :16, :, 2:12].clone())
                next_time = next(selected, -1)
        for hidden in torch.cat(selected_hidden).split(BATCH):
            shape_logits, pose_logits, _ = model.current_head.components(hidden)
            shapes.append(shape_logits)
            poses.append(pose_logits)
    if not targets:
        raise ValueError("no proved current-object training labels")
    return tuple(torch.cat(parts) for parts in (boards, shapes, poses, targets)), counts, witness


def baseline_joint(model, shape, pose):
    logits = (shape[:, :, None] + pose[:, None, :]).masked_fill(~model.valid_poses, -torch.inf)
    return logits.flatten(1).log_softmax(1).reshape_as(logits)


def prediction(model, lookup, data, index):
    board, shape, pose, _ = data
    joint = baseline_joint(model, shape[index], pose[index])
    return model.correct_current_joint(joint, board[index].long(), lookup).flatten(1)


@torch.no_grad()
def metrics(model, lookup, data):
    labels = data[-1]
    nll, correct, shape_correct = 0., 0, 0
    for index in torch.arange(len(labels)).split(BATCH):
        joint = prediction(model, lookup, data, index)
        wanted = labels[index]
        choice = joint.argmax(1)
        nll += float(F.nll_loss(joint, wanted, reduction="sum"))
        correct += int((choice == wanted).sum())
        shape_correct += int((choice // 180 == wanted // 180).sum())
    return {"examples": len(labels), "joint_correct": correct, "joint_accuracy": correct / len(labels),
            "shape_correct": shape_correct, "shape_accuracy": shape_correct / len(labels), "nll": nll / len(labels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = current_readout_scene_weights(checkpoint["model"]).eval().requires_grad_(False)
    if len(model.codebook) > 256:
        raise ValueError("compact cache requires at most 256 frozen tokens")
    head = model.current_readout_head.requires_grad_(True)
    data, counts, witness = cache_current_labels(model, training_captures(config, model.codebook.numpy()))
    lookup = model.current_readout_lookup()
    labels = data[-1]
    before = metrics(model, lookup, data)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(SEED)
    updates = 0
    for _ in range(EPOCHS):
        for index in torch.randperm(len(labels), generator=generator).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = F.nll_loss(prediction(model, lookup, data, index), labels[index])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite current readout fit")
            loss.backward()
            optimizer.step()
            updates += 1
    after = metrics(model, lookup, data)
    with torch.no_grad():
        joint = model.scene_details(*witness)["current"]["joint"].flatten(1)
        cached = prediction(model, lookup, data, slice(0, 1))
        # Cached pose components are computed in larger batches than live inference.
        torch.testing.assert_close(joint, cached, rtol=1e-6, atol=1e-5)
    weights = model.state_dict()
    for name, original in checkpoint["model"].items():
        torch.testing.assert_close(weights[name].reshape(-1).view(torch.uint8),
                                   original.reshape(-1).view(torch.uint8), rtol=0, atol=0)
    report = {"source_checkpoint_sha256": sha(args.checkpoint), "source_step": checkpoint.get("step"),
              "script_sha256": sha(Path(__file__)), "manifest_sha256": config["manifest_sha256"],
              "annotations_sha256": config["annotations_sha256"], "validation_used": False, "test_used": False,
              "train_examples_by_capture": counts, "train_examples": len(labels), "label_uses": len(labels) * EPOCHS,
              "cache_bytes": sum(tensor.numel() * tensor.element_size() for tensor in data),
              "updates": updates, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "seed": SEED,
              "parameters_fitted": sum(p.numel() for p in head.parameters()), "before": before, "after": after,
              "optimizer_exported": False}
    args.output.mkdir(parents=True)
    output = args.output / "model.pt"
    torch.save({"model": weights, **({"step": checkpoint["step"]} if "step" in checkpoint else {}), "config": {
        **config, "model_family": type(model).__name__, "current_readout_head_only_fit": report
    }}, output)
    restored = scene_from_weights(torch.load(output, map_location="cpu", weights_only=True)).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored.scene_details(*witness)["current"]["joint"].flatten(1), joint,
                                   rtol=0, atol=0)
    report["checkpoint_sha256"] = sha(output)
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
