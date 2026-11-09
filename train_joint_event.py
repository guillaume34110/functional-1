"""Fit composition and pose jointly from proven train targets and continuations."""
import argparse
import json
from pathlib import Path

import torch

from audit_controls import BUTTONS, supported
from scene_model import joint_event_scene_weights, scene_from_weights
from train_mode_correction import proven_composition
from train_spawn_preview import sha, training_captures


def continuation(base, tokens, details, current, kind, background, moves):
    if current // 180 != kind:
        return None
    mask, _ = base._hard_object(details["current"])
    board = tokens.masked_fill(mask, base.blank_id)[0, :, 2:12]
    if not torch.equal(board, background):
        return None
    cells = mask[0, :, 2:12].nonzero().numpy()
    if supported(board.numpy(), cells, base.blank_id):
        return None
    candidates = current + moves * 10
    candidates = candidates[candidates // 180 == kind]
    candidates = candidates[base.valid_poses.flatten()[candidates]]
    collision = (base.shape_masks.flatten(0, 1)[candidates].bool() & (board != base.blank_id)).flatten(1).any(1)
    return candidates[~collision]


@torch.no_grad()
def collect(base, captures):
    samples = []

    def remember(details, current, mode, poses, protocol):
        allowed = torch.full((18,), -1, dtype=torch.long)
        allowed[:len(poses)] = poses
        samples.append((details["mode"][0], details["next"]["joint"][0].flatten(), current,
                        details["contact_logit"][0].sigmoid(), mode, allowed, protocol))

    down = BUTTONS["down"]
    for episode, (_, frames, actions, labels) in enumerate(captures):
        modes, known, old, old_mask, new = proven_composition(base, frames, labels)
        background = frames[:-1].masked_fill(old_mask, base.blank_id)[:, :, 2:12]
        kind = labels["shape_id"]
        y, x = labels["cells"].amin(1).unbind(1)
        pose = kind * 180 + y * 10 + x
        held = (actions == down) & torch.cat((torch.tensor([False]), actions[:-1] == down))
        dy = y.diff()
        continuous = old & new & known & (modes == 0) & held & (kind[1:] == kind[:-1]) & (x[1:] == x[:-1]) & (dy >= 0)
        moves = dy[continuous].unique()
        state = generated = generated_state = None
        for t, command in enumerate(actions):
            frame, action = frames[t:t + 1], command[None]
            phase = (t + episode) % 40
            observed = known[t] and (modes[t] != 0 or t % 8 == 0)
            target = pose[t + 1:t + 2] if new[t] else pose[:0]
            if observed or phase == 0:
                details = base.scene_details(frame, action, state)
                state = details["state"]
                if observed and details["contact_visible"][0]:
                    current = details["current"]["joint"].flatten().argmax()
                    remember(details, current, modes[t], target, 0)
            else:
                state = base.advance(frame, action, state)
            if phase == 0:
                generated, generated_state = base.decode_scene(frame, details, hard=True), state
            elif phase < 24 and generated is not None:
                rollout = base.scene_details(generated, action, generated_state)
                if rollout["contact_visible"][0]:
                    current = rollout["current"]["joint"].flatten().argmax()
                    if known[t] and torch.equal(generated[:, :, 2:12], frame[:, :, 2:12]):
                        remember(rollout, current, modes[t], target, 1)
                    elif continuous[t]:
                        allowed = continuation(base, generated, rollout, current, kind[t], background[t], moves)
                        if allowed is not None and len(allowed):
                            remember(rollout, current, modes[t], allowed, 2)
                generated, generated_state = base.decode_scene(generated, rollout, hard=True), rollout["state"]
    if not samples:
        raise ValueError("no proven event targets")
    *columns, protocols = zip(*samples)
    return (*map(torch.stack, columns), torch.tensor(protocols))


def evaluate_batch(model, cache, indices):
    mode, future, current, contact, target, allowed, _ = (values[indices] for values in cache)
    scores = model.event_scores(mode, future, current, contact, torch.ones_like(target, dtype=torch.bool))
    rows = torch.arange(len(indices))
    given_mode = scores[rows, target]
    log_mass = given_mode.logsumexp(1)
    qualified = allowed[:, 0] >= 0
    choices = allowed[qualified]
    selected = given_mode[qualified].gather(1, choices.clamp_min(0)).masked_fill(choices < 0, -torch.inf)
    log_mass = log_mass.index_copy(0, qualified.nonzero().flatten(), selected.logsumexp(1))
    flat = scores.flatten(1)
    loss = flat.logsumexp(1) - log_mass
    best = flat.argmax(1)
    mode_correct = best // future.size(1) == target
    pose_correct = (~qualified) | (allowed == (best % future.size(1))[:, None]).any(1)
    return loss, mode_correct, mode_correct & pose_correct


@torch.no_grad()
def metrics(model, cache):
    result = {}
    for code, name in enumerate(("observed", "autonomous", "continuation")):
        indices = (cache[-1] == code).nonzero().flatten()
        loss, mode, joint = [torch.cat(column) for column in zip(
            *(evaluate_batch(model, cache, batch) for batch in indices.split(64)))]
        result[name] = {"examples": len(indices), "nll": float(loss.mean()),
                        "correct_mode": int(mode.sum()), "correct_event": int(joint.sum())}
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
    torch.manual_seed(56)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    model = joint_event_scene_weights(checkpoint["model"]).eval()
    cache = collect(base, training_captures(checkpoint["config"], base.codebook.numpy()))
    counts = torch.bincount(cache[4], minlength=3)
    if not counts.all() or not torch.bincount(cache[-1], minlength=3).all():
        raise ValueError("all modes and supervision sources need proven examples")
    parameters = [parameter for name, parameter in model.named_parameters() if name.startswith("event_")]
    optimizer = torch.optim.AdamW(parameters, lr=.025)
    weights = (counts.max() / counts).sqrt()
    before = metrics(model, cache)
    for _ in range(6):
        for indices in torch.randperm(len(cache[-1])).split(64):
            loss, _, _ = evaluate_batch(model, cache, indices)
            sample_weights = weights[cache[4][indices]]
            loss = (loss * sample_weights).sum() / sample_weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
    state = model.state_dict()
    for key, tensor in checkpoint["model"].items():
        torch.testing.assert_close(state[key], tensor, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "mode_counts": counts.tolist(),
              "before": before, "after": metrics(model, cache)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "joint_event_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
