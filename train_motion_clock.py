"""Fit a movement clock on causally observed, proven training transitions."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scene_model import (RowMotionClockSceneDynamics, RowTransportSceneDynamics,
                         SpawnPreviewSceneDynamics, motion_clock_scene_weights, row_motion_clock_scene_weights)
from train_scene import label_mask
from train_spawn_preview import sha, training_captures


@torch.no_grad()
def training_examples(model, captures):
    features, base_odds, copy_weights, moved, map_odds = [], [], [], [], []
    witness = None
    row_clock = isinstance(model, RowMotionClockSceneDynamics)
    base_scene = RowTransportSceneDynamics if row_clock else SpawnPreviewSceneDynamics
    for _, frames, actions, labels in captures:
        valid = torch.ones(len(actions), dtype=torch.bool)
        current, old_mask = label_mask({k: v[:-1, None] for k, v in labels.items()}, 0, valid)
        future, next_mask = label_mask({k: v[1:, None] for k, v in labels.items()}, 0, valid)
        before = frames[:-1].masked_fill(old_mask, model.blank_id)[:, :, 2:12]
        after = frames[1:].masked_fill(next_mask, model.blank_id)[:, :, 2:12]
        eligible = current & future & (before == after).flatten(1).all(1)
        origins = labels["cells"].amin(1)
        poses = labels["shape_id"] * 180 + origins[:, 0] * 10 + origins[:, 1]
        if row_clock:
            eligible &= (actions == 3) & (poses[:-1] // 180 == poses[1:] // 180)
            eligible &= (origins[:-1, 1] == origins[1:, 1]) & (origins[:-1, 0] <= origins[1:, 0])
        state = None
        for t, command in enumerate(actions):
            frame, action = frames[t:t + 1], command[None]
            if not eligible[t]:
                state = model.advance(frame, action, state)
                continue
            prior = state
            details = base_scene.scene_details(model, frame, action, prior)
            state = details["state"]
            current = details["current"]
            if current["visible"].item() <= 0 or int(current["joint"].flatten().argmax()) != int(poses[t]):
                continue
            joint = details["next"]["joint"].flatten()
            if not torch.isfinite(joint[poses[t + 1]]):
                raise ValueError("invalid proven next pose")
            if row_clock:
                candidates = model.motion_candidates(current["joint"], action).flatten()
                if not torch.isfinite(joint[candidates]).any():
                    continue
                stay = joint[~candidates]
                moving = joint[candidates]
            else:
                stay = joint[poses[t]:poses[t] + 1]
                moving = joint.clone()
                moving[poses[t]] = -torch.inf
            features.append(model.motion_clock_features(state, action)[0])
            base_odds.append(moving.logsumexp(0) - stay.logsumexp(0))
            map_odds.append(moving.max() - stay.max())
            copy_weights.append(details["mode"].softmax(1)[0, 0])
            moved.append(poses[t + 1] != poses[t])
            if witness is None:
                witness = (frame, action, prior)
    if not moved:
        raise ValueError("no proven motion examples")
    return tuple(torch.stack(x) for x in (features, base_odds, copy_weights, moved, map_odds)), witness


def correction(head, features, copy):
    return head(features)[:, 0] * copy


@torch.no_grad()
def metrics(head, cache):
    features, base, copy, moved, top = cache
    offset = correction(head, features, copy)
    predicted = top + offset > 0
    return {"nll": float(F.binary_cross_entropy_with_logits(base + offset, moved.float())),
            "move_class_correct": int((predicted & moved).sum()),
            "still_class_correct": int((~predicted & ~moved).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--row-transport", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(40)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    migrate = row_motion_clock_scene_weights if args.row_transport else motion_clock_scene_weights
    model = migrate(checkpoint["model"]).eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("motion_clock_head."))
    head = model.motion_clock_head
    cache, witness = training_examples(model, training_captures(checkpoint["config"], model.codebook.numpy()))
    features, base, copy, moved, _ = cache
    if not (moved.any() and (~moved).any()) or not torch.isfinite(base).all():
        raise ValueError("unbalanced or invalid training cache")
    before = metrics(head, cache)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003)
    for _ in range(20):
        for batch in torch.randperm(len(moved)).split(128):
            scores = base[batch] + correction(head, features[batch], copy[batch])
            loss = F.binary_cross_entropy_with_logits(scores, moved[batch].float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
    with torch.no_grad():
        base_scene = RowTransportSceneDynamics if args.row_transport else SpawnPreviewSceneDynamics
        raw = base_scene.scene_details(model, *witness)
        runtime = model.scene_details(*witness)
        torch.testing.assert_close(runtime["next"]["joint"].flatten(1),
                                   (raw["next"]["joint"] + runtime["motion_joint_offset"]).flatten(1).log_softmax(1)
                                   + raw["next"]["joint"].flatten(1).logsumexp(1, keepdim=True))
    state = model.state_dict()
    for key, tensor in checkpoint["model"].items():
        torch.testing.assert_close(tensor, state[key], rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "epochs": 20, "batch": 128, "lr": .003,
              "examples": len(moved), "moved": int(moved.sum()), "before": before, "after": metrics(head, cache)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "motion_clock_fit": result}, destination)
    if args.row_transport:
        from scene_model import scene_from_weights
        reloaded = scene_from_weights(torch.load(destination, map_location="cpu", weights_only=True)).eval()
        with torch.no_grad():
            torch.testing.assert_close(reloaded.scene_details(*witness)["next"]["joint"], runtime["next"]["joint"])
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
