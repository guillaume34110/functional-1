"""Calibrate carried poses when the observed board differs from the last render."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scene_model import feedback_carry_scene_weights, scene_from_weights
from train_scene import label_mask
from train_spawn_preview import sha, training_captures


@torch.inference_mode()
def collect(base, model, captures):
    samples = []
    identical_boards = 0
    for episode, (_, frames, actions, labels) in enumerate(captures):
        known, _ = label_mask({key: value[:, None] for key, value in labels.items()}, 0,
                              torch.ones(len(frames), dtype=torch.bool))
        origin = labels["cells"].amin(1)
        truth = labels["shape_id"] * 180 + origin[:, 0] * 10 + origin[:, 1]
        state = None
        for t, command in enumerate(actions):
            frame, action = frames[t:t + 1], command[None]
            if (t // 96 + episode) % 4:
                state = base.advance(frame, action, state)
                continue
            prior = state
            details = base.scene_details(frame, action, prior)
            prediction = base.decode_scene(frame, details, hard=True)
            state = model.rendered_state(details, prediction)
            if not known[t] or prior is None or prior[0, -1, 0, 1] != -2:
                continue
            previous = prior[:, -1, 0, 0].long()
            if previous.item() < 0:
                continue
            hidden = details["state"][:, :16, :, 2:12]
            current, _ = base._current_details(frame, base._current_hidden(frame, action, prior, hidden))
            features, enabled = base.carry_features(frame, current, previous)
            if not enabled.item():
                continue
            feedback = model.agreement_features(frame, prior, features)[0]
            if not feedback[0]:
                identical_boards += 1
                continue
            joint = current["joint"].flatten()
            samples.append((feedback, base.pose_carry(features).reshape(()), previous[0] == truth[t],
                            joint[truth[t]], joint[previous[0]], joint.max(), joint.argmax() == truth[t]))
    if not samples:
        raise ValueError("no visually confirmed carries on changed boards")
    return tuple(torch.stack(column) for column in zip(*samples)), identical_boards


@torch.no_grad()
def counts(gate, valid, carry, best, raw_correct):
    updated = torch.logaddexp(carry + F.logsigmoid(-gate), F.logsigmoid(gate))
    selected = updated >= best + F.logsigmoid(-gate)
    return {"exact_pose": int(torch.where(selected, valid, raw_correct).sum()),
            "wrong_carried_pose": int((selected & ~valid).sum()),
            "right_carried_pose": int((selected & valid).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    model = feedback_carry_scene_weights(checkpoint["model"]).eval()
    cache, identical = collect(base, model, training_captures(checkpoint["config"], base.codebook.numpy()))
    features, original, valid, truth, carry, best, raw_correct = (column.clone() for column in cache)
    if not valid.any() or valid.all() or not torch.isfinite(truth).all():
        raise ValueError("need valid labels and both correct and incorrect changed-board carries")
    weights = torch.bincount(valid.long()).float().rsqrt()[valid.long()]
    weights = weights / weights.sum()
    head = model.carry_agreement
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        gate = original + head(features)[:, 0]
        held = F.logsigmoid(gate).masked_fill(~valid, -torch.inf)
        log_likelihood = torch.logaddexp(truth + F.logsigmoid(-gate), held)
        loss = -(weights * log_likelihood).sum()
        loss += .2 * F.binary_cross_entropy_with_logits(gate, valid.float(), weight=weights, reduction="sum")
        loss += .0001 * head.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    state = model.state_dict()
    for key, value in checkpoint["model"].items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "changed_boards": len(features),
              "identical_boards": identical, "correct_carries": int(valid.sum()),
              "before": counts(original, valid, carry, best, raw_correct),
              "after": counts(original + head(features)[:, 0], valid, carry, best, raw_correct),
              "coefficients": head.weight.detach()[0].tolist()}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "feedback_carry_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
