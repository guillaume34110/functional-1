"""Learn how much to trust a pose carried from the last rendered frame."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scene_model import scene_from_weights, tracked_canvas_scene_weights
from train_scene import label_mask
from train_spawn_preview import sha, training_captures


def rendered_pose(details):
    future = details["next"]
    return torch.where(future["visible"][:, 0] > 0, future["joint"].flatten(1).argmax(1), -1)


@torch.no_grad()
def collect(model, base, captures):
    samples = []

    def record(details, tokens, carry, truth, reference, background, generated):
        features, enabled = model.carry_features(tokens, details["current"], carry)
        if not enabled.item():
            return
        if generated and not torch.equal(tokens[:, :, 2:12], reference[:, :, 2:12]):
            if carry.item() // 180 != truth.item() // 180:
                return
            mask = base.shape_masks.flatten(0, 1)[carry.item()].bool()
            if not torch.equal(tokens[0, :, 2:12].masked_fill(mask, base.blank_id), background):
                return
            truth = carry[0]
        joint = details["current"]["joint"].flatten()
        samples.append((features[0], carry[0] == truth, joint.argmax() == truth,
                        joint[carry[0]], joint.max(), generated))

    for episode, (_, frames, actions, labels) in enumerate(captures):
        known, masks = label_mask({key: value[:, None] for key, value in labels.items()}, 0,
                                  torch.ones(len(frames), dtype=torch.bool))
        origins = labels["cells"].amin(1)
        truth = labels["shape_id"] * 180 + origins[:, 0] * 10 + origins[:, 1]
        backgrounds = frames.masked_fill(masks, base.blank_id)[:, :, 2:12]
        selected = known[:-1] & ((torch.arange(len(actions)) + episode) % 4 == 0)
        predecessors = F.pad(selected[1:], (0, 1))
        state = carry = generated = generated_state = generated_carry = None
        for t, command in enumerate(actions):
            frame, action = frames[t:t + 1], command[None]
            phase = (t + episode) % 40
            if selected[t] or predecessors[t] or phase == 0:
                details = base.scene_details(frame, action, state)
                state = details["state"]
                if selected[t] and carry is not None:
                    record(details, frame, carry, truth[t], frame, backgrounds[t], False)
                carry = rendered_pose(details)
            else:
                state = base.advance(frame, action, state)
                carry = None
            if phase == 0:
                generated, generated_state, generated_carry = base.decode_scene(frame, details, hard=True), state, carry
            elif phase < 16 and generated is not None:
                rollout = base.scene_details(generated, action, generated_state)
                if known[t]:
                    record(rollout, generated, generated_carry, truth[t], frame, backgrounds[t], True)
                generated = base.decode_scene(generated, rollout, hard=True)
                generated_state, generated_carry = rollout["state"], rendered_pose(rollout)
    if not samples:
        raise ValueError("no visually corroborated training carries")
    *columns, generated = zip(*samples)
    return (*map(torch.stack, columns), torch.tensor(generated))


@torch.no_grad()
def metrics(head, cache):
    features, valid, base_correct, carry, best, generated = cache
    trust = head(features)[:, 0]
    picked = torch.logaddexp(carry + F.logsigmoid(-trust), F.logsigmoid(trust)) >= best + F.logsigmoid(-trust)
    correct = torch.where(picked, valid, base_correct)
    return {name: {"examples": int(mask.sum()), "incorrect_carries": int((mask & ~valid).sum()),
                   "base_correct": int((mask & base_correct).sum()), "tracked_correct": int((mask & correct).sum())}
            for name, mask in (("observed", ~generated), ("autonomous", generated))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(59)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    model = tracked_canvas_scene_weights(checkpoint["model"]).eval()
    cache = collect(model, base, training_captures(checkpoint["config"], base.codebook.numpy()))
    features, valid, _, _, _, generated = cache
    if not (valid.any() and (~valid).any() and generated.any() and (~generated).any()):
        raise ValueError("training needs both correct and incorrect carry examples")
    head = model.pose_carry
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head(features)[:, 0], valid.float())
        loss = loss + .0001 * head.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    state = model.state_dict()
    for key, value in checkpoint["model"].items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "metrics": metrics(head, cache)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "pose_carry_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
