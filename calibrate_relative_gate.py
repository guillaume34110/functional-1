"""Calibrate the relative proposal gate on train-only expert disagreements."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from audit_long import recorded_path
from model import ACTIONS
from scene_model import OrientedProposalSceneDynamics, RelativeProposalSceneDynamics, scene_from_weights
from train_relative_proposal import batch, collect, metrics, proposal_example
from train_spawn_preview import sha, training_captures


@torch.no_grad()
def correct(model, cache, indices, enabled, force_local=False, shape_only=False):
    return torch.cat([batch(model, cache, part, enabled, force_local, shape_only=shape_only)[2]
                      for part in indices.split(64)])


@torch.no_grad()
def motion_rollouts(model, captures):
    examples, masks, disagreements = [], [], []
    reasons = Counter()
    rotation = (ACTIONS.index("a"), ACTIONS.index("b"))
    styles = torch.cat((model.object_tiles, torch.tensor([model.blank_id])))
    poses = model.event_poses
    span = model.proposal_offsets.abs().amax()

    def observe(tokens, old, details, event):
        if event["status"] in ("motion", "deposit"):
            return True
        reason = event["reason"]
        motion_error = reason in ("uncommanded-orientation-change", "discontinuous-object-possible-spawn")
        motion_error |= reason == "object-style-changed-possible-spawn" and event.get("previous_oracle", {}).get("status") == "motion"
        if not motion_error or event["mode"] != 0 or not bool(details["contact_visible"][0]):
            return False
        previous = int(details["previous_action"][0])
        action = int(event["button"])
        if action in rotation or previous in rotation or not torch.isin(tokens[0, :, 2:12], styles).all():
            return False
        current = details["current"]["joint"].flatten().argmax()
        footprint = model.shape_masks.flatten(0, 1)[current].bool()
        if old is None or len(old) != 4 or not footprint[old[:, 0], old[:, 1]].all():
            return False
        scores = model.event_scores(details["event_base_mode"], details["event_base_joint"].flatten(1),
                                    current[None], details["contact_logit"].sigmoid(), details["contact_visible"])
        code = int(model.motion_agreement(scores, current[None])[0])
        if not code or details["next"]["visible"][0, 0] <= 0:
            return False
        row = (poses % 180 // 10 - current % 180 // 10).abs()
        col = (poses % 10 - current % 10).abs()
        safe = ((poses // 180 == current // 180) & (row <= span) & (col <= span)
                & model.valid_poses.flatten() & (model.canvas_overlap(details)[0, 0] == 0))
        assert safe[current], "certified current pose must be clear in the copy canvas"
        examples.append(proposal_example(model, details, torch.tensor([action]), current, torch.tensor(0),
                                         torch.tensor(-1), True))
        masks.append(safe)
        disagreements.append(code)
        reasons[reason] += 1
        return False

    for _, frames, actions, labels in captures:
        known = (labels["shape_id"] >= 0) & torch.isin(labels["evidence"], torch.tensor([1, 2]))
        anchors = (known[48:-256].nonzero().flatten() + 48).tolist()
        if len(anchors) < 3:
            continue
        for branch, quantile in enumerate((1, 2, 3)):
            index = anchors[quantile * len(anchors) // 4]
            state = None
            for t in range(index - 48, index):
                state = model.advance_observed(frames[t:t+1], actions[t:t+1], state)
            commands = actions[index:index+256] if branch == 0 else [ACTIONS.index("down")] * 256
            recorded_path(model, frames[index:index+1], state, labels["cells"][index].numpy(), commands,
                          ACTIONS[int(actions[index - 1])], observer=observe)
    if not examples:
        raise ValueError("no certified copy disagreements in generated train trajectories")
    *columns, generated = zip(*examples)
    cache = (*map(torch.stack, columns), torch.tensor(generated))
    return cache, torch.stack(masks), torch.tensor(disagreements), dict(reasons)


@torch.no_grad()
def fit_motion_gate(model, cache, rollouts=None):
    _, mode, future, _, current, contact, previous, action, _, _, _ = cache
    disagreements = []
    for part in torch.arange(len(action)).split(64):
        scores = model.event_scores(mode[part], future[part], current[part], contact[part].sigmoid(),
                                    torch.ones_like(current[part], dtype=torch.bool))
        disagreements.append(model.motion_agreement(scores, current[part]))
    disagreements = torch.cat(disagreements)
    rotation = torch.tensor([ACTIONS.index("a"), ACTIONS.index("b")])
    continuous = ~torch.isin(action, rotation) & ~torch.isin(previous, rotation)
    commands = previous * len(ACTIONS) + action
    weight = model.motion_agreement_gate.weight
    original = weight.clone()
    edited = torch.zeros_like(weight, dtype=torch.bool)
    report = {}
    for code, name in ((1, "orientation"), (2, "outside_local_support")):
        indices = (continuous & (disagreements == code)).nonzero().flatten()
        if not len(indices):
            report[name] = {"examples": 0, "bias": 0.}
            continue
        rows = commands[indices].unique()
        shape_only = code == 1
        absolute = correct(model, cache, indices, False, shape_only=shape_only)
        local = correct(model, cache, indices, True, force_local=True, shape_only=shape_only)
        relative_only = local & ~absolute
        absolute_only = absolute & ~local
        before = correct(model, cache, indices, True, shape_only=shape_only)
        probes = torch.empty(0, dtype=torch.long)
        if rollouts is not None:
            observed, masks, categories, _ = rollouts
            probes = (categories == code).nonzero().flatten()
            probe_commands = observed[6][probes] * len(ACTIONS) + observed[7][probes]
            rows = torch.cat((rows, probe_commands)).unique()
        if len(probes):
            probe_before = int(batch(model, observed, probes, True, copy_mask=masks)[2].sum())
        bias = 0.
        if relative_only.any() or len(probes):
            importance = torch.ones(len(indices))
            for priority in (relative_only, absolute_only):
                if priority.any():
                    importance[priority] += len(indices) / (2 * int(priority.sum()))
            choices = []
            for candidate in (0., 2., 4., 6., 8., 10., 12., 16., 20., 24.):
                weight[rows, code] = original[rows, code] + candidate
                losses = torch.cat([batch(model, cache, part, True, shape_only=shape_only)[0]
                                    for part in indices.split(64)])
                objective = (losses * importance).sum() / importance.sum() + .0001 * candidate ** 2
                if len(probes):
                    weak = [batch(model, observed, part, True, copy_mask=masks)[0] for part in probes.split(64)]
                    objective += .7 * torch.cat(weak).mean()
                choices.append((float(objective), candidate))
            _, bias = min(choices)
            weight[rows, code] = original[rows, code] + bias
            edited[rows, code] = True
        after = correct(model, cache, indices, True, shape_only=shape_only)
        report[name] = {"examples": len(indices), "relative_only": int(relative_only.sum()),
                        "absolute_only": int(absolute_only.sum()), "bias": bias,
                        "correct_before_after": [int(before.sum()), int(after.sum())],
                        "command_rows": rows.tolist()}
        if len(probes):
            report[name]["generated_disagreements"] = len(probes)
            report[name]["generated_safe_before_after"] = [
                probe_before, int(batch(model, observed, probes, True, copy_mask=masks)[2].sum())]
    torch.testing.assert_close(weight[~edited], original[~edited], rtol=0, atol=0)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion", action="store_true")
    parser.add_argument("--motion-rollouts", action="store_true")
    args = parser.parse_args()
    if args.motion_rollouts and not args.motion:
        parser.error("--motion-rollouts requires --motion")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(62)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = scene_from_weights(checkpoint).eval()
    family = OrientedProposalSceneDynamics if args.motion else RelativeProposalSceneDynamics
    if type(model) is not family:
        raise ValueError("gate calibration needs a relative proposal checkpoint")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not args.motion and name == "proposal_gate_calibration")
    cache = collect(model, model, training_captures(checkpoint["config"], model.codebook.numpy()), motion_only=args.motion)
    if not cache[-1].any() or cache[-1].all():
        raise ValueError("training requires observed and generated inputs")
    before = metrics(model, cache, True)
    if args.motion:
        changed = "motion_agreement_gate.weight"
        rollouts = (motion_rollouts(model, training_captures(checkpoint["config"], model.codebook.numpy()))
                    if args.motion_rollouts else None)
        calibration = {"motion_calibration": fit_motion_gate(model, cache, rollouts)}
        if rollouts is not None:
            calibration["rollout_failure_reasons"] = rollouts[3]
    else:
        changed = "proposal_gate_calibration"
        indices = torch.arange(len(cache[-1]))
        absolute = correct(model, cache, indices, False)
        local = correct(model, cache, indices, True, force_local=True)
        local_only = indices[local & ~absolute]
        absolute_only = indices[absolute & ~local]
        if not (len(local_only) and len(absolute_only)):
            raise ValueError("train must prove hard disagreements in both directions")
        before_hard = correct(model, cache, torch.cat((local_only, absolute_only)), True)
        parameter = model.proposal_gate_calibration
        optimizer = torch.optim.Adam((parameter,), lr=.025)
        for _ in range(512):
            chosen = torch.cat((indices[torch.randint(len(indices), (32,))],
                                local_only[torch.randint(len(local_only), (16,))],
                                absolute_only[torch.randint(len(absolute_only), (16,))]))
            loss = batch(model, cache, chosen, True)[0].mean() + .002 * parameter.square().sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((parameter,), 1., error_if_nonfinite=True)
            optimizer.step()
        after_hard = correct(model, cache, torch.cat((local_only, absolute_only)), True)
        calibration = {"calibration": parameter.detach().tolist(), "local_only": len(local_only),
                       "absolute_only": len(absolute_only),
                       "local_only_correct_before_after": [int(v[:len(local_only)].sum()) for v in (before_hard, after_hard)],
                       "absolute_only_correct_before_after": [int(v[len(local_only):].sum()) for v in (before_hard, after_hard)]}
    state = model.state_dict()
    for key, value in checkpoint["model"].items():
        if key != changed:
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), **calibration,
              "before": before, "after": metrics(model, cache, True)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "relative_gate_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
