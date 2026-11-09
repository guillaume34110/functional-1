"""Fit an independent local copy proposal with aligned visible train targets."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from model import ACTIONS
from scene_model import (ContinuationSceneDynamics, oriented_proposal_scene_weights,
                         relative_proposal_scene_weights, scene_from_weights)
from train_mode_correction import proven_composition
from train_spawn_preview import sha, training_captures


def proposal_example(model, details, action, current, mode, target, generated):
    features = model.proposal_features(details, action, current[None])[0]
    return (features, details["event_base_mode"][0], details["event_base_joint"][0].flatten(),
            details["state"][0, 16:20].clone(), current, details["contact_logit"][0],
            details["previous_action"][0], action[0], mode, target, generated)


@torch.no_grad()
def collect(model, base, captures, *, motion_only=False):
    samples = []
    rotations = (ACTIONS.index("a"), ACTIONS.index("b"))

    def record(details, action, current, mode, target, generated):
        active = details["contact_visible"][0] & (details["next"]["visible"][0, 0] > 0)
        if not active or int(details["current"]["joint"].flatten().argmax()) != int(current):
            return
        if motion_only:
            if int(action[0]) in rotations or int(details["previous_action"][0]) in rotations:
                return
            scores = model.event_scores(details["event_base_mode"], details["event_base_joint"].flatten(1),
                                        current[None], details["contact_logit"].sigmoid(), details["contact_visible"])
            if int(model.motion_agreement(scores, current[None])[0]) == 0:
                return
        samples.append(proposal_example(model, details, action, current, mode, target, generated))

    for episode, (_, frames, actions, labels) in enumerate(captures):
        modes, known, old, _, new = proven_composition(base, frames, labels)
        kind = labels["shape_id"]
        y, x = labels["cells"].amin(1).unbind(1)
        poses = kind * 180 + y * 10 + x
        qualified = known & old
        uncommon = (modes != 0) | (kind[:-1] != kind[1:])
        selected = qualified & (uncommon | ((torch.arange(len(actions)) + episode) % 8 == 0))
        predecessors = F.pad(selected[1:], (0, 1))
        targets = torch.where(new, poses[1:], -1)
        state = generated = generated_state = None
        for t, command in enumerate(actions):
            frame, action = frames[t:t + 1], command[None]
            phase = (t + episode) % 32
            if selected[t] or predecessors[t] or phase == 0:
                details = base.scene_details(frame, action, state)
                state = details["state"]
                if selected[t]:
                    record(details, action, poses[t], modes[t], targets[t], False)
            else:
                state = base.advance(frame, action, state)
            if phase == 0:
                generated, generated_state = base.decode_scene(frame, details, hard=True), state
            elif phase < 24 and generated is not None:
                if not torch.equal(generated[:, :, 2:12], frame[:, :, 2:12]):
                    generated = None
                    continue
                rollout = base.scene_details(generated, action, generated_state)
                if qualified[t] and (uncommon[t] or t % 3 == 0):
                    record(rollout, action, poses[t], modes[t], targets[t], True)
                generated = base.decode_scene(generated, rollout, hard=True)
                generated_state = rollout["state"]
    if not samples:
        raise ValueError("no aligned proof for local proposals")
    *columns, generated = zip(*samples)
    return (*map(torch.stack, columns), torch.tensor(generated))


def batch(model, cache, indices, enabled, force_local=False, supervise_local=False, shape_only=False, copy_mask=None):
    features, mode, future, metadata, current, contact, previous, action, want_mode, want_pose, _ = (
        values[indices] for values in cache)
    visible = torch.ones_like(current, dtype=torch.bool)
    scores = model.event_scores(mode, future, current, contact.sigmoid(), visible)
    if supervise_local:
        translation = model.relative_proposal(features)[:, :-1]
        local = model.local_proposal(scores[:, 0], translation, features, current)
        local_target = local.gather(1, want_pose.clamp_min(0)[:, None])[:, 0]
        eligible = (want_mode == 0) & (want_pose >= 0) & torch.isfinite(local_target)
    if enabled:
        gate = 40 if force_local else model.proposal_gate_delta(scores, current, action, previous)
        scores = model.blend_copy(scores, features, current, visible, gate)
    current_joint = torch.full_like(future, -torch.inf).scatter(1, current[:, None], 0).reshape(
        len(indices), len(model.object_shapes), 180)
    details = {"state": torch.cat((metadata.new_zeros(len(indices), 16, 18, 20), metadata), dim=1),
               "contact_visible": visible, "contact_logit": contact, "previous_action": previous,
               "current": {"joint": current_joint, "visible": visible[:, None]},
               "next": {"visible": visible[:, None]}}
    scores = ContinuationSceneDynamics._refine_events(model, scores, details, action, current)
    if copy_mask is not None:
        eligible = copy_mask[indices]
        copy = scores[:, 0]
        loss = copy.logsumexp(1) - copy.masked_fill(~eligible, -torch.inf).logsumexp(1)
        chosen_copy = scores.amax(2).argmax(1) == 0
        safe = eligible.gather(1, copy.argmax(1, keepdim=True))[:, 0]
        return loss, chosen_copy, chosen_copy & safe, torch.zeros_like(chosen_copy)
    event = scores[torch.arange(len(indices)), want_mode]
    if shape_only:
        exact = event.reshape(len(indices), -1, 180).logsumexp(2)
        exact = exact.gather(1, want_pose.clamp_min(0)[:, None] // 180)[:, 0]
    else:
        exact = event.gather(1, want_pose.clamp_min(0)[:, None])[:, 0]
    log_target = torch.where(want_pose >= 0, exact, event.logsumexp(1))
    all_events = scores.flatten(1)
    loss = all_events.logsumexp(1) - log_target
    if supervise_local:
        loss = loss - .5 * local_target.masked_fill(~eligible, 0)
    best = all_events.argmax(1)
    correct_mode = best // future.size(1) == want_mode
    picked_pose = best % future.size(1)
    matched = picked_pose // 180 == want_pose // 180 if shape_only else picked_pose == want_pose
    correct = correct_mode & ((want_pose < 0) | matched)
    rotation = (want_mode == 0) & (want_pose >= 0) & (want_pose // 180 != current // 180)
    return loss, correct_mode, correct, rotation


@torch.no_grad()
def metrics(model, cache, enabled):
    result = {}
    for name, rows in (("observed", ~cache[-1]), ("autonomous", cache[-1])):
        batches = (batch(model, cache, part, enabled) for part in rows.nonzero().flatten().split(64))
        loss, modes, correct, rotation = (torch.cat(column) for column in zip(*batches))
        result[name] = {"examples": int(rows.sum()), "nll": float(loss.mean()),
                        "correct_mode": int(modes.sum()), "correct_event": int(correct.sum()),
                        "rotations": int(rotation.sum()), "rotations_correct": int((rotation & correct).sum())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orientations", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(61)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    migrate = oriented_proposal_scene_weights if args.orientations else relative_proposal_scene_weights
    model = migrate(checkpoint["model"]).eval()
    trainable = (("oriented_proposal.", "proposal_shape_embedding.", "proposal_style_scale", "proposal_gate_calibration")
                 if args.orientations else ("relative_proposal.",))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(trainable))
    cache = collect(model, base, training_captures(checkpoint["config"], base.codebook.numpy()))
    want_mode, want_pose = cache[-3:-1]
    counts = torch.bincount(want_mode, minlength=3)
    if not counts.all() or not cache[-1].any() or cache[-1].all():
        raise ValueError("train needs each mode and both conditioning protocols")
    weights = (counts.max() / counts).pow(.25)[want_mode]
    rotation = (want_mode == 0) & (want_pose >= 0) & (want_pose // 180 != cache[4] // 180)
    weights = weights * (1 + rotation.float())
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.002)
    before = metrics(base if args.orientations else model, cache, args.orientations)
    for _ in range(5):
        for indices in torch.randperm(len(want_mode)).split(64):
            losses, _, _, _ = batch(model, cache, indices, True, supervise_local=args.orientations)
            loss = (losses * weights[indices]).sum() / weights[indices].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
    state = model.state_dict()
    for key, tensor in checkpoint["model"].items():
        if not key.startswith(trainable):
            torch.testing.assert_close(state[key], tensor, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "counts": counts.tolist(), "orientation_proposals": args.orientations,
              "before": before, "after": metrics(model, cache, True)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "relative_proposal_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
