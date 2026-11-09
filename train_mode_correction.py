"""Fit a residual composition mode classifier from proven training screens."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scene_model import (RowTransportSceneDynamics, mode_correction_scene_weights, rewrite_mode_scene_weights,
                         scene_from_weights, spatial_mode_scene_weights)
from train_scene import composition_targets, label_mask, spawn_preview_targets, visual_absence
from train_spawn_preview import sha, training_captures


def proven_composition(model, frames, labels):
    before, after = frames[:-1], frames[1:]
    valid = torch.ones(len(before), dtype=torch.bool)
    old_labels = {key: value[:-1, None] for key, value in labels.items()}
    new_labels = {key: value[1:, None] for key, value in labels.items()}
    old, old_mask = label_mask(old_labels, 0, valid)
    new, new_mask = label_mask(new_labels, 0, valid)
    old_known = old | visual_absence(before, labels["evidence"][:-1], valid, model.object_tiles)
    new_known = new | visual_absence(after, labels["evidence"][1:], valid, model.object_tiles)
    modes, known = composition_targets(model, before, after, old_mask, new_mask, old_known, new_known)
    spawns, _ = spawn_preview_targets(model, before, after, old_mask, old, labels["shape_id"][:-1])
    modes[spawns] = 1
    known[spawns] = True
    return modes, known, old, old_mask, new


@torch.no_grad()
def cache_examples(model, captures, feature_key, base_key, autonomous=False):
    samples = []
    witness = None
    for episode, (_, frames, actions, labels) in enumerate(captures):
        modes, known, _, _, _ = proven_composition(model, frames, labels)
        state = generated = generated_state = None
        for t, command in enumerate(actions):
            frame, action = frames[t:t + 1], command[None]
            prior = state
            observed = known[t] and (not autonomous or modes[t] != 0 or t % 4 == 0)
            start_fork = autonomous and (t + episode) % 24 == 0
            if observed or start_fork:
                details = model.scene_details(frame, action, prior)
                state = details["state"]
                if observed:
                    samples.append((details[feature_key][0], details[base_key][0], modes[t], False))
                    if witness is None:
                        witness = (frame, action, prior)
            else:
                state = model.advance(frame, action, prior)
            if start_fork:
                generated = model.decode_scene(frame, details, hard=True)
                generated_state = state
            elif generated is not None:
                if not torch.equal(generated[:, :, 2:12], frame[:, :, 2:12]):
                    generated = None
                    continue
                rollout = model.scene_details(generated, action, generated_state)
                if known[t]:
                    samples.append((rollout[feature_key][0], rollout[base_key][0], modes[t], True))
                generated = model.decode_scene(generated, rollout, hard=True)
                generated_state = rollout["state"]
    if not samples:
        raise ValueError("no proven training modes")
    features, logits, targets, generated = zip(*samples)
    return (torch.stack(features), torch.stack(logits), torch.stack(targets)), torch.tensor(generated), witness


@torch.no_grad()
def metrics(head, cache, weights):
    features, logits, target = cache
    adjusted = logits + head(features)
    prediction = adjusted.argmax(1)
    matrix = torch.bincount(target * 3 + prediction, minlength=9).reshape(3, 3)
    return {"nll": float(F.cross_entropy(adjusted, target)),
            "weighted_nll": float(F.cross_entropy(adjusted, target, weight=weights)),
            "confusion_true_by_predicted": matrix.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--spatial", action="store_true", help="fit a row-based residual on the V45 checkpoint")
    variant.add_argument("--rewrite-proposal", action="store_true", help="fit a residual from V48 rewrite proposals")
    variant.add_argument("--autonomous", action="store_true", help="refit the existing mode head on verified rollout boards")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(49 if args.rewrite_proposal else 46 if args.spatial else 45)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.autonomous:
        model = scene_from_weights(checkpoint).eval()
        if type(model) is not RowTransportSceneDynamics:
            parser.error("autonomous mode training requires a row transport checkpoint")
        head_name, feature_key, base_key = "mode_correction_head", "mode_correction_features", "mode_base_logits"
    elif args.rewrite_proposal:
        model = rewrite_mode_scene_weights(checkpoint["model"]).eval()
        head_name, feature_key, base_key = "rewrite_mode_head", "rewrite_mode_features", "rewrite_mode_base_logits"
    elif args.spatial:
        model = spatial_mode_scene_weights(checkpoint["model"]).eval()
        head_name, feature_key, base_key = "spatial_mode_head", "spatial_mode_features", "spatial_mode_base_logits"
    else:
        model = mode_correction_scene_weights(checkpoint["model"]).eval()
        head_name, feature_key, base_key = "mode_correction_head", "mode_correction_features", "mode_base_logits"
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(head_name + "."))
    head = getattr(model, head_name)
    cache, generated, witness = cache_examples(model, training_captures(checkpoint["config"], model.codebook.numpy()),
                                               feature_key, base_key, args.autonomous)
    features, logits, target = cache
    counts = torch.bincount(target, minlength=3)
    if not (counts > 0).all():
        raise ValueError("each composition mode needs proven targets")
    weights = (counts.max() / counts).sqrt()
    if args.autonomous and not torch.bincount(target[generated], minlength=3).all():
        raise ValueError("rollouts must include all three proven modes")
    before = metrics(head, cache, weights)
    if args.autonomous:
        generated_cache = tuple(values[generated] for values in cache)
        generated_before = metrics(head, generated_cache, weights)
    lr = .001 if args.autonomous else .003
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    for _ in range(20):
        for batch in torch.randperm(len(target)).split(128):
            adjusted = logits[batch] + head(features[batch])
            loss = F.cross_entropy(adjusted, target[batch], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
    with torch.no_grad():
        details = model.scene_details(*witness)
        torch.testing.assert_close(details["mode"], details[base_key] + head(details[feature_key]))
    state = model.state_dict()
    for key, tensor in checkpoint["model"].items():
        if not (args.autonomous and key.startswith(head_name + ".")):
            torch.testing.assert_close(tensor, state[key], rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "epochs": 20, "batch": 128, "lr": lr,
              "examples": len(target), "mode_counts": counts.tolist(), "weights": weights.tolist(), "head": head_name,
              "before": before, "after": metrics(head, cache, weights)}
    if args.autonomous:
        result["autonomous"] = {"examples": int(generated.sum()), "before": generated_before,
                                "after": metrics(head, generated_cache, weights)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "mode_correction_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
