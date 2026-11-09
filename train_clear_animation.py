"""Learn the sequence from a witnessed lock through the first cleared screen."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from audit_clear_events import exact_foreground_pose
from model import ACTIONS
from scene_model import (ClearAnimationSceneDynamics, ClearDurationSceneDynamics,
                         clear_animation_scene_weights, clear_duration_scene_weights, scene_from_weights)
from train_rewrite_evidence import load_events
from train_row_transport import verified_background
from train_spawn_preview import sha, training_captures


def flash_tile(captures, events):
    changed = Counter()
    for name, frames, _, _ in captures:
        board = frames[:, :, 2:12]
        for event in events.get(name, ()):
            initial = board[event["last_consecutive_lock_frame"]]
            later = board[event["flash_frames"]]
            changed.update(later[later != initial].tolist())
    if not changed:
        raise ValueError("no witnessed flash tiles")
    tile, count = changed.most_common(1)[0]
    if count < .98 * changed.total():
        raise ValueError("witnesses need more than one flash tile")
    return tile, dict(changed)


@torch.no_grad()
def collect(model, captures, events):
    onset_features, onset_labels, sequence = [], [], []
    count = Counter()
    shapes, tiles = model.object_shapes.numpy(), model.object_tiles.numpy()
    for name, frames, actions, labels in captures:
        boards = frames[:, :, 2:12]
        label_arrays = {k: v.numpy() for k, v in labels.items()}
        previous = torch.cat((boards[:1], boards[:-2]))
        previous_actions = F.pad(actions[:-1], (1, 0), value=len(ACTIONS))
        starts = torch.zeros(len(actions), dtype=torch.bool)
        ordinary_tiles = torch.isin(boards[:-1], model.object_tiles)
        qualified = ~ordinary_tiles.all(2).any(1)
        for event in events.get(name, ()):
            start = event["first_lock_frame"]
            last = event["last_consecutive_lock_frame"]
            t, background = verified_background(event, boards.numpy(), label_arrays, model)
            if not 0 <= start <= last <= t or not torch.equal(boards[start:last+1], boards[last:last+1].expand(last-start+1, -1, -1)):
                raise ValueError("clear proof does not preserve its source board")
            if t - start >= model.phase_limit:
                raise ValueError("witnessed animation exceeds the elapsed phase vocabulary")
            qualified[start:event["confirmation_frame"]] = False
            starts[start] = True
            target = boards[start+1:t+2].clone()
            target[-1] = torch.from_numpy(background)
            age = torch.arange(len(target))
            end = age == age[-1]
            pose = torch.full_like(age, -1)
            if event["outcome_proof"][0] != "raw":
                exact = exact_foreground_pose(boards[t+1].numpy(), background, shapes, tiles, model.blank_id)
                if exact is None or exact < 0:
                    raise ValueError("clear endpoint has no unique full foreground")
                pose[-1] = exact
            group = torch.full_like(age, event["row_count"])
            sequence.append((frames[start:start+1].expand(len(age), -1, -1), age, actions[start:t+1],
                             previous_actions[start:t+1], target, end, pose, group))
            count[event["row_count"]] += 1
        use = qualified | starts
        onset_features.append(model.onset_features(boards[:-1][use], previous[use]))
        onset_labels.append(starts[use])
    if sum(count.values()) != sum(map(len, events.values())) or not sequence:
        raise ValueError("clear animation evidence coverage mismatch")
    onset = (torch.cat(onset_features), torch.cat(onset_labels))
    source, age, action, prior, target, end, pose, group = map(torch.cat, zip(*sequence))
    valid = (model.animation_row_indices >= 0) & (model.animation_row_indices < 18)
    matches = (model.animation_row_choices(source) == target[:, :, None]).all(3) & F.pad(valid, (0, 2), value=True)
    if not matches[end].any(2).all() or not (pose >= 0).any():
        raise ValueError("unrepresentable endpoint background or absent spawn evidence")
    changed = (target != source[:, :, 2:12]).any(2)
    return onset, (source, age, action, prior, matches, end, pose, changed, group), dict(count)


@torch.no_grad()
def measure(model, onset, sequence):
    features, wanted = onset
    selected = model.animation_onset(features)[:, 0] > 0
    source, age, action, prior, matches, end, pose, _, _ = sequence
    phase, density = model.animation_features(source, age, action)
    output_phase = model.animation_output_phase(phase, end)
    scores = model.animation_row_scores(output_phase, density)
    correct = matches.gather(2, scores.argmax(2, keepdim=True))[:, :, 0]
    finish, show = model.animation_end_logits(phase, age)
    visible = pose >= 0
    joint = model.animation_spawn(source[visible], output_phase[visible], prior[visible])
    return {"onset_correct": int((selected & wanted).sum()), "onset_total": int(wanted.sum()),
            "false_onsets": int((selected & ~wanted).sum()), "normal_samples": int((~wanted).sum()),
            "rows_exact": int(correct.sum()), "rows_proved": int(matches.any(2).sum()),
            "first_frame_exact": int(correct[age == 0].all(1).sum()),
            "endpoint_background_exact": int(correct[end].all(1).sum()),
            "endpoint_frames": int(end.sum()), "finish_correct": int(((finish > 0) & end).sum()),
            "premature_finish": int(((finish > 0) & ~end).sum()),
            "spawn_exact": int((joint.flatten(1).argmax(1) == pose[visible]).sum()),
            "spawn_total": int(visible.sum()), "spawn_visible_correct": int(((show[end] > 0) == visible[end]).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(72)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    events = load_events(args.events, checkpoint, "train")
    captures = list(training_captures(checkpoint["config"], base.codebook.numpy()))
    tile, histogram = flash_tile(captures, events)
    timed = type(base) is ClearAnimationSceneDynamics
    migration = clear_duration_scene_weights if timed else clear_animation_scene_weights
    model = migration(checkpoint["model"], tile).eval()
    onset, sequence, count = collect(model, captures, events)
    before = measure(base if timed else model, onset, sequence)
    del captures
    if timed:
        with torch.no_grad():
            age, end = sequence[1], sequence[5]
            mean = age[end].float().mean() / (model.phase_limit - 1)
            model.animation_end[-1].weight[0].zero_()
            model.animation_end[-1].bias[0] = mean.logit()
            model.animation_time.weight[-1] = model.animation_time(age[end]).mean(0)
    parameters = []
    trainable_names = set()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("animation_"))
        if parameter.requires_grad:
            parameters.append(parameter)
            trainable_names.add(name)
    features, wanted = onset
    gate_optimizer = torch.optim.LBFGS(model.animation_onset.parameters(), max_iter=120, line_search_fn="strong_wolfe")

    def onset_loss():
        gate_optimizer.zero_grad(set_to_none=True)
        logits = model.animation_onset(features)[:, 0]
        penalty = F.softplus(-logits[wanted]).mean() + 2 * F.softplus(logits[~wanted]).mean()
        penalty += .0001 * sum(parameter.square().sum() for parameter in model.animation_onset.parameters())
        penalty.backward()
        return penalty

    gate_optimizer.step(onset_loss)
    optimizer = torch.optim.AdamW(parameters, lr=.004)
    severity = torch.ones(5)
    if timed:
        for rows, number in count.items():
            severity[rows] = (max(count.values()) / number) ** .5
    for _ in range(160):
        for indices in torch.randperm(len(sequence[1])).split(128):
            source, age, action, prior, matches, end, pose, changed, group = (column[indices] for column in sequence)
            phase, density = model.animation_features(source, age, action)
            output_phase = model.animation_output_phase(phase, end)
            scores = model.animation_row_scores(output_phase, density)
            proved = matches.any(2)
            qualified = scores[proved]
            mass = qualified.logsumexp(1) - qualified.masked_fill(~matches[proved], -torch.inf).logsumexp(1)
            weight = 1 + 3 * changed[proved].float() + end[:, None].expand_as(proved)[proved].float()
            importance = severity[group]
            weight = weight * importance[:, None].expand_as(proved)[proved]
            loss = (mass * weight).sum() / weight.sum()
            finish, show = model.animation_end_logits(phase, age)
            finish_loss = F.binary_cross_entropy_with_logits(finish, end.float(), pos_weight=finish.new_tensor(1.5),
                                                            reduction="none")
            loss += .5 * (finish_loss * importance).sum() / importance.sum()
            if end.any():
                loss += .1 * F.binary_cross_entropy_with_logits(show[end], (pose[end] >= 0).float())
                if timed:
                    loss += .05 * F.smooth_l1_loss(model.clear_duration(phase[end]), age[end].float())
            visible = pose >= 0
            if visible.any():
                joint = model.animation_spawn(source[visible], output_phase[visible], prior[visible]).flatten(1)
                loss += .3 * F.nll_loss(joint, pose[visible])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
    state = model.state_dict()
    for key, tensor in checkpoint["model"].items():
        if not timed or key not in trainable_names:
            torch.testing.assert_close(state[key], tensor, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "events_sha256": sha(args.events),
              "flash_tile": tile, "flash_tile_histogram": histogram, "witnessed_by_rows": count,
              "source_conditioned_duration": timed,
              "trainable_parameters": sum(p.numel() for p in parameters), "before": before,
              "after": measure(model, onset, sequence)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "clear_animation_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
