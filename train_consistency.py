"""Fit shape/column score corrections on observed training transitions."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from model import ACTIONS
from scene_model import FactorizedPreviewRenderSceneDynamics, consistency_scene_weights
from train_scene import label_mask
from train_spawn_preview import sha, training_captures


@torch.no_grad()
def cache_examples(model, captures):
    features, masses, maxima, copy, target, buttons = [], [], [], [], [], []
    witness = None
    groups = torch.arange(4)[:, None]
    for _, frames, actions, labels in captures:
        valid = torch.ones(len(actions), dtype=torch.bool)
        old, old_mask = label_mask({k: v[:-1, None] for k, v in labels.items()}, 0, valid)
        new, new_mask = label_mask({k: v[1:, None] for k, v in labels.items()}, 0, valid)
        before = frames[:-1].masked_fill(old_mask, model.blank_id)[:, :, 2:12]
        after = frames[1:].masked_fill(new_mask, model.blank_id)[:, :, 2:12]
        eligible = old & new & (before == after).flatten(1).all(1)
        origins = labels["cells"].amin(1)
        poses = labels["shape_id"] * 180 + origins[:, 0] * 10 + origins[:, 1]
        state = None
        for t, command in enumerate(actions):
            frame, action = frames[t:t + 1], command[None]
            if not eligible[t]:
                state = model.advance(frame, action, state)
                continue
            prior = state
            details = FactorizedPreviewRenderSceneDynamics.scene_details(model, frame, action, prior)
            state = details["state"]
            current = details["current"]
            if current["visible"].item() <= 0 or int(current["joint"].flatten().argmax()) != int(poses[t]):
                continue
            memberships = model.consistency_group(current["joint"]).flatten()
            joint = details["next"]["joint"].flatten()
            if not torch.isfinite(joint[poses[t + 1]]):
                raise ValueError("invalid proven next pose")
            masked = joint[None].masked_fill(memberships[None] != groups, -torch.inf)
            features.append(model.consistency_features(state, action)[0])
            masses.append(masked.logsumexp(1))
            maxima.append(masked.amax(1))
            copy.append(details["mode"].softmax(1)[0, 0])
            target.append(memberships[poses[t + 1]])
            buttons.append(command)
            if witness is None:
                witness = (frame, action, prior)
    if not target:
        raise ValueError("no proven training transitions")
    return tuple(torch.stack(x) for x in (features, masses, maxima, copy, target, buttons)), witness


def offsets(head, features, copy):
    shape, column = (head(features) * copy[:, None]).unbind(1)
    return torch.stack((torch.zeros_like(shape), shape, column, shape + column), 1)


@torch.no_grad()
def metrics(head, cache):
    features, masses, maxima, copy, target, buttons = cache
    offset = offsets(head, features, copy)
    prediction = (maxima + offset).argmax(1)
    return {"nll": float(F.cross_entropy(masses + offset, target)),
            "by_action": {name: {"count": int((buttons == i).sum()),
                                 "correct_group": int(((buttons == i) & (prediction == target)).sum())}
                          for i, name in enumerate(ACTIONS)},
            "group_counts": torch.bincount(target, minlength=4).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(44)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = consistency_scene_weights(checkpoint["model"]).eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("consistency_head."))
    head = model.consistency_head
    cache, witness = cache_examples(model, training_captures(checkpoint["config"], model.codebook.numpy()))
    features, masses, _, copy, target, _ = cache
    before = metrics(head, cache)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003)
    for _ in range(20):
        for batch in torch.randperm(len(target)).split(128):
            logits = masses[batch] + offsets(head, features[batch], copy[batch])
            loss = F.cross_entropy(logits, target[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
    with torch.no_grad():
        base = FactorizedPreviewRenderSceneDynamics.scene_details(model, *witness)
        runtime = model.scene_details(*witness)
        groups = model.consistency_group(base["current"]["joint"]).flatten(1)
        expected = offsets(head, model.consistency_features(base["state"], witness[1]),
                           base["mode"].softmax(1)[:, 0]).gather(1, groups)
        torch.testing.assert_close(runtime["next"]["joint"].flatten(1),
                                   (base["next"]["joint"].flatten(1) + expected).log_softmax(1))
    state = model.state_dict()
    for key, tensor in checkpoint["model"].items():
        torch.testing.assert_close(tensor, state[key], rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "epochs": 20, "batch": 128, "lr": .003,
              "examples": len(target), "before": before, "after": metrics(head, cache)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "consistency_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
