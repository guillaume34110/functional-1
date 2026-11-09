"""Refit the overlap head on short train rollouts with a verified board."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scene_model import scene_from_weights
from train_scene import label_mask, learned_overlap_loss, learned_overlap_targets, overlap_ranking_loss
from train_spawn_preview import sha, training_captures


@torch.no_grad()
def collect(model, captures):
    samples = []
    for episode, (_, frames, actions, labels) in enumerate(captures):
        known, masks = label_mask({key: value[:, None] for key, value in labels.items()}, 0,
                                  torch.ones(len(frames), dtype=torch.bool))
        board = frames.masked_fill(masks, model.blank_id)[:, :, 2:12]
        copy = known[:-1] & known[1:] & (board[:-1] == board[1:]).flatten(1).all(1)
        copy &= (board[:-1] != model.blank_id).flatten(1).any(1)
        state = generated = generated_state = None
        for t, command in enumerate(actions):
            frame, action = frames[t:t + 1], command[None]
            phase = (t + episode) % 16
            if phase == 0:
                generated, generated_state = frame, state
            if phase < 5 and generated is not None:
                if (generated_state is not None and copy[t]
                        and torch.equal(generated[:, :, 2:12], frame[:, :, 2:12])):
                    samples.append((generated[0], action[0], generated_state[0], masks[t],
                                    labels["cells"][t + 1], labels["shape_id"][t + 1], phase > 0))
                generated, generated_state = model.predict_tokens(generated, action, generated_state)
            state = model.advance(frame, action, state)
    if not samples:
        raise ValueError("no proven overlap examples")
    *columns, autonomous = zip(*samples)
    return (*map(torch.stack, columns), torch.tensor(autonomous))


def batch_loss(model, cache, indices):
    frames, actions, states, masks, cells, kinds, _ = (values[indices] for values in cache)
    details = model.scene_details(frames, actions, states)
    joint = details["next"]["joint"]
    valid = torch.ones(len(frames), dtype=torch.bool)
    rank = overlap_ranking_loss(model, frames, joint, masks, cells, kinds, valid)
    classifier = learned_overlap_loss(model, details["overlap_logits"], frames, masks, valid)
    origins = cells.amin(1)
    target = kinds * 180 + origins[:, 0] * 10 + origins[:, 1]
    flat = joint.flatten(1)
    pose = F.cross_entropy(flat, target)
    overlaps = learned_overlap_targets(model, frames, masks).flatten(1)
    overlaps = overlaps.gather(1, flat.argmax(1, keepdim=True)).sum()
    return rank + 0.1 * classifier + pose, overlaps, (flat.argmax(1) == target).sum()


@torch.no_grad()
def measure(model, cache):
    result = {}
    for name, rows in (("observed", ~cache[-1]), ("autonomous", cache[-1])):
        indices = rows.nonzero().flatten()
        scores = [batch_loss(model, cache, batch) for batch in indices.split(64)]
        result[name] = {"examples": len(indices), "overlaps": sum(int(score[1]) for score in scores),
                        "exact_poses": sum(int(score[2]) for score in scores)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(53)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = scene_from_weights(checkpoint).eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("overlap_head."))
    cache = collect(model, training_captures(checkpoint["config"], model.codebook.numpy()))
    if not (cache[-1].any() and (~cache[-1]).any()):
        raise ValueError("need proven observed and autonomous contexts")
    before = measure(model, cache)
    optimizer = torch.optim.AdamW(model.overlap_head.parameters(), lr=0.0003)
    for _ in range(4):
        for indices in torch.randperm(len(cache[-1])).split(32):
            loss, _, _ = batch_loss(model, cache, indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.overlap_head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "before": before, "after": measure(model, cache)}
    state = model.state_dict()
    for key, value in checkpoint["model"].items():
        if not key.startswith("overlap_head."):
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "overlap_recovery_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
