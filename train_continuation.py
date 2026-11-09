"""Fit button-conditioned orientation and contact-conditioned event scores."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from audit_controls import supported
from model import ACTIONS
from scene_model import continuation_scene_weights, scene_from_weights
from train_mode_correction import proven_composition
from train_spawn_preview import sha, training_captures


@torch.no_grad()
def collect(base, captures):
    samples = []
    pose_kinds = base.event_poses // 180

    def record(details, tokens, frame, action, kind, background, mode, next_kind, generated):
        if not (details["contact_visible"][0] and details["next"]["visible"][0, 0] > 0):
            return
        if details["current"]["joint"].flatten().argmax() // 180 != kind:
            return
        mask, _ = base._hard_object(details["current"])
        mask = mask[0, :, 2:12]
        if not torch.equal(tokens[0, :, 2:12].masked_fill(mask, base.blank_id), background):
            return
        floating = not supported(background.numpy(), mask.nonzero().numpy(), base.blank_id)
        if generated and not torch.equal(tokens[:, :, 2:12], frame[:, :, 2:12]):
            if not (mode == 0 and kind == next_kind and floating):
                return
        scores = details["event_scores"][0]
        same = pose_kinds == kind
        groups = (scores[0, same], scores[0, ~same], scores[1], scores[2])
        target = int(next_kind != kind) if mode == 0 else int(mode) + 1
        samples.append((torch.stack([v.logsumexp(0) for v in groups]), torch.stack([v.max() for v in groups]),
                        action[0], details["previous_action"][0], details["contact_logit"][0], target,
                        floating, generated))

    rotation = (ACTIONS.index("a"), ACTIONS.index("b"))
    for episode, (_, frames, actions, labels) in enumerate(captures):
        modes, known, old, masks, new = proven_composition(base, frames, labels)
        kind = labels["shape_id"]
        background = frames[:-1].masked_fill(masks, base.blank_id)[:, :, 2:12]
        qualified = known & old & (new | (modes == 2))
        selected = qualified & ((modes != 0) | (kind[:-1] != kind[1:]) | torch.isin(actions, torch.tensor(rotation))
                                | ((torch.arange(len(actions)) + episode) % 8 == 0))
        predecessors = F.pad(selected[1:], (0, 1))
        state = generated = generated_state = None
        for t, command in enumerate(actions):
            frame, action = frames[t:t + 1], command[None]
            phase = (t + episode) % 48
            if selected[t] or predecessors[t] or phase == 0:
                details = base.scene_details(frame, action, state)
                state = details["state"]
                if selected[t]:
                    record(details, frame, frame, action, kind[t], background[t], modes[t], kind[t + 1], False)
            else:
                state = base.advance(frame, action, state)
            if phase == 0:
                generated, generated_state = base.decode_scene(frame, details, hard=True), state
            elif phase < 24 and generated is not None:
                rollout = base.scene_details(generated, action, generated_state)
                if qualified[t]:
                    record(rollout, generated, frame, action, kind[t], background[t], modes[t], kind[t + 1], True)
                generated = base.decode_scene(generated, rollout, hard=True)
                generated_state = rollout["state"]
    if not samples:
        raise ValueError("no proven next-event samples")
    *values, targets, floating, generated = zip(*samples)
    return (*map(torch.stack, values), torch.tensor(targets), torch.tensor(floating), torch.tensor(generated))


@torch.no_grad()
def metrics(best, targets, floating, generated):
    picked = best.argmax(1)
    return {name: {"examples": int(rows.sum()), "correct": int((rows & (picked == targets)).sum()),
                   "unsupported_deposits": int((rows & floating & (picked == 2)).sum()),
                   "rotations_correct": int((rows & (picked == 1) & (targets == 1)).sum()),
                   "copy_shape_changes_in_error": int((rows & (picked == 1) & (targets == 0)).sum())}
            for name, rows in (("observed", ~generated), ("autonomous", generated))}


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
    model = continuation_scene_weights(checkpoint["model"]).eval()
    mass, best, actions, previous, contact, targets, floating, generated = collect(
        base, training_captures(checkpoint["config"], base.codebook.numpy()))
    counts = torch.bincount(targets, minlength=4)
    if not counts.all() or not generated.any():
        raise ValueError("need proven examples of each event and generated contexts")
    weights = (counts.max() / counts).pow(.25)
    safe = best[:, [0, 1, 3]].amax(1)
    floating_rows = floating.nonzero().flatten()
    if not len(floating_rows):
        raise ValueError("no proven floating-piece candidates")
    hardest = (best[:, 2] - safe)[floating_rows]
    difficult = floating_rows[hardest.topk(max(1, len(hardest) // 20)).indices]
    rotate = torch.tensor([ACTIONS.index("a"), ACTIONS.index("b")])
    stable = (targets == 0) & ~torch.isin(actions, rotate) & ~torch.isin(previous, rotate)
    stable_rows = stable.nonzero().flatten()
    if not len(stable_rows):
        raise ValueError("no proven orientation-preserving non-rotation commands")
    hardest = (best[:, 1] - best[:, 0])[stable_rows]
    hard_stable = stable_rows[hardest.topk(max(1, len(hardest) // 20)).indices]
    parameters = [model.orientation_transition.weight, model.compose_contact_scale]
    optimizer = torch.optim.LBFGS(parameters, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        offsets = model.continuation_offsets(actions, previous, contact)
        loss = F.cross_entropy(mass + offsets, targets, weight=weights)
        adjusted = best[difficult] + offsets[difficult]
        risk = F.softplus(1 + adjusted[:, 2] - adjusted[:, [0, 1, 3]].amax(1)).mean()
        rotation_risk = F.softplus(1 + best[hard_stable, 1] + offsets[hard_stable, 1] - best[hard_stable, 0]).mean()
        loss = loss + .2 * (risk + rotation_risk) + .001 * sum(parameter.square().sum() for parameter in parameters)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        adjusted = best + model.continuation_offsets(actions, previous, contact)
    state = model.state_dict()
    for key, tensor in checkpoint["model"].items():
        torch.testing.assert_close(state[key], tensor, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "counts": counts.tolist(), "difficult_floating": len(difficult),
              "contact_scale": float(F.softplus(model.compose_contact_scale.detach())),
              "before": metrics(best, targets, floating, generated),
              "after": metrics(adjusted, targets, floating, generated)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "continuation_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
