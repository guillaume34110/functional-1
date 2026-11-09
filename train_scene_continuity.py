"""Fit two residual scores for copy identity and ordinary-board rewrites."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from audit_controls import BUTTONS, direction_oracle, observed_object, transition_oracle
from audit_long import recorded_path
from model import ACTIONS
from scene_model import CounterSceneDynamics, scene_from_weights
from train_event_replay import BURN, Window, event_targets, replay_windows
from train_spawn_preview import sha, training_captures


SPAN = 8


def spaced(indices, count):
    if len(indices) <= count:
        return indices
    return indices[torch.arange(1, count + 1) * len(indices) // (count + 1)]


def windows(base, captures):
    result = []
    for _, frames, actions, labels in captures:
        if len(actions) < BURN + SPAN:
            continue
        target = event_targets(base, frames, labels)
        old, mode, new, _ = target.unbind(1)
        known = (old >= 0) & (new >= 0) & (mode >= 0)
        known[:BURN] = False
        rotations = known & (mode == 0) & ((actions == BUTTONS["a"]) | (actions == BUTTONS["b"]))
        ground = known & (old % 180 // 10 >= 8)
        selected = torch.cat(tuple(spaced(group.nonzero().flatten(), n) for group, n in
                                   ((rotations, 9), (ground, 9), (known & (mode == 2), 6))))
        starts = {max(BURN, min(int(t) // SPAN * SPAN, len(actions) - SPAN)) for t in selected}
        for start in sorted(starts):
            end = start + SPAN
            result.append(Window(frames[start-BURN:start], actions[start-BURN:start], frames[start:end+1],
                                 actions[start:end], target[start:end]))
    return result


def margins(details, changed):
    scores = details["event_scores"]
    copy = scores[:, 0]
    identity = copy.masked_fill(~changed, -torch.inf).amax(1) - copy.masked_fill(changed, -torch.inf).amax(1)
    rewrite = scores[:, 2].amax(1) - scores[:, :2].amax((1, 2))
    return identity, rewrite


def rendered_alternative(base, tokens, old, details, mode, action, previous, allowed=None):
    score = details["event_scores"][:, mode]
    if allowed is not None:
        score = score.masked_fill(~allowed, -torch.inf)
    route = torch.full_like(details["mode"], -torch.inf)
    route[:, mode] = 0
    joint = score.log_softmax(1).reshape_as(details["next"]["joint"])
    trial = {**details, "mode": route, "next": {**details["next"], "joint": joint}}
    prediction = base.decode_scene(tokens, trial, hard=True)
    new, _ = observed_object(base, trial, prediction)
    before, after = (x[0, :, 2:12] for x in (tokens, prediction))
    ordinary = torch.isin(after, base.object_tiles) | (after == base.blank_id)
    if new is None or not ordinary.all():
        return False
    geometry = transition_oracle(before.numpy(), after.numpy(), old, new, base.blank_id)
    if mode == 1:
        return geometry["status"] == "deposit" and not geometry["collision"]
    return geometry["status"] == "motion" and direction_oracle(old, new, action, 0, previous)[0] is None


@torch.inference_mode()
def generated(base, captures):
    rows = [[], []]
    paths = 0
    horizon = 96
    for _, frames, actions, labels in captures:
        if len(actions) < BURN + horizon:
            continue
        evidence = labels["evidence"][:-1]
        visible = (labels["shape_id"][:-1] >= 0) & ((evidence == 1) | (evidence == 2))
        visible[:BURN] = False
        visible[len(actions)-horizon+1:] = False
        ground = visible & (labels["cells"][:-1, :, 0].amin(1) >= 6)
        rotated = visible & ((actions == BUTTONS["a"]) | (actions == BUTTONS["b"]))
        for start in torch.cat((spaced(ground.nonzero().flatten(), 2), spaced(rotated.nonzero().flatten(), 1))).unique().tolist():
            state = None
            for t in range(start-BURN, start):
                state = base.advance_observed(frames[t:t+1], actions[t:t+1], state)

            def watch(tokens, old, details, event):
                if event["status"] in ("motion", "deposit"):
                    return True
                known, changed, rewrite = base.continuity_signals(tokens, details)
                button = ACTIONS[event["button"]]
                if not known[0] or old is None or button not in BUTTONS:
                    return False
                mask, color = base._hard_object(details["current"])
                current = mask[0, :, 2:12].nonzero().numpy()
                if set(map(tuple, current)) != set(map(tuple, old)):
                    return False
                raw = event.get("previous_oracle", event)
                if event["mode"] == 0 and raw["status"] == "motion" and event["style"] != int(color[0]):
                    if rendered_alternative(base, tokens, old, details, 0, button, event["previous_button"], ~changed):
                        rows[0].append(margins(details, changed)[0][0])
                elif event["mode"] == 2 and rewrite[0]:
                    out = base.decode_scene(tokens, details, hard=True)[0, :, 2:12]
                    ordinary = torch.isin(out, base.object_tiles) | (out == base.blank_id)
                    mode = int(details["mode"][0, :2].argmax())
                    if not ordinary.all() and rendered_alternative(base, tokens, old, details, mode, button,
                                                                  event["previous_button"], ~changed if mode == 0 else None):
                        rows[1].append(margins(details, changed)[1][0])
                return False

            recorded_path(base, frames[start:start+1], state, labels["cells"][start].numpy(),
                          actions[start:start+horizon], ACTIONS[int(actions[start-1])], observer=watch)
            paths += 1
    return rows, paths


@torch.no_grad()
def collect(base, captures):
    cohorts = windows(base, captures)
    rows = [[], []]
    for frame, _, details, targets, _, aligned in replay_windows(base, cohorts):
        old, mode, new, _ = targets.unbind(1)
        known, changed, rewrite = base.continuity_signals(frame, details)
        current = details["current"]["joint"].flatten(1).argmax(1)
        known &= (mode >= 0) & (current == old) & (new >= 0)
        style_margin, rewrite_margin = margins(details, changed)
        visible_motion = known & (mode == 0) & style_margin.isfinite()
        wanted_style = changed.gather(1, new.clamp_min(0)[:, None])[:, 0]
        for column, (score, target, keep) in enumerate(((style_margin, wanted_style, visible_motion),
                                                        (rewrite_margin, mode == 2, rewrite & known))):
            if keep.any():
                rows[column].append((score[keep], target[keep], torch.full_like(target[keep], int(aligned), dtype=torch.long)))
    replay, paths = generated(base, captures)
    result = []
    for column, samples in enumerate(rows):
        if not samples:
            raise ValueError("missing proved continuity samples")
        if replay[column]:
            score = torch.stack(replay[column])
            samples.append((score, torch.zeros_like(score, dtype=torch.bool), torch.full_like(score, 2, dtype=torch.long)))
        result.append(tuple(torch.cat(items) for items in zip(*samples)))
    return result, len(cohorts), paths


def report_counts(data, shift):
    scores, targets, source = data
    return [{"samples": int((source == i).sum()),
             "errors_before": int((((scores > 0) != targets) & (source == i)).sum()),
             "errors_after": int(((((scores + shift) > 0) != targets) & (source == i)).sum()),
             "positive_targets": int((targets & (source == i)).sum())} for i in range(3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    if not isinstance(base, CounterSceneDynamics) or not base.counter.pulsed or base.scene_continuity is not None:
        raise ValueError("continuity fitting requires a pulsed-counter checkpoint")
    state = {**checkpoint["model"], "scene_continuity": torch.zeros(2)}
    model = scene_from_weights(state).eval()
    data, windows_count, paths = collect(base, list(training_captures(checkpoint["config"], base.codebook.numpy())))
    gate = model.scene_continuity
    optimizer = torch.optim.LBFGS([gate], max_iter=60, line_search_fn="strong_wolfe")
    weights = []
    fitted = []
    for scores, targets, source in data:
        if not torch.isfinite(scores).all():
            raise ValueError("nonfinite continuity margins")
        fitted.append(bool(((scores > 0) & ~targets).any()))
        group = source * 2 + targets.long()
        weights.append(torch.bincount(group)[group].float().reciprocal())
    if not any(fitted):
        raise ValueError("no proved errors for either continuity adjustment")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = .001 * gate.square().sum()
        for k, (scores, targets, _) in enumerate(data):
            if fitted[k]:
                loss = loss + F.binary_cross_entropy_with_logits(scores + gate[k], targets.float(),
                                                                  weight=weights[k], reduction="sum")
        loss.backward()
        return loss

    optimizer.step(closure)
    state = model.state_dict()
    for name, tensor in checkpoint["model"].items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    result = {"source_sha256": sha(args.checkpoint), "fit_split": "train", "validation_used": False, "test_used": False,
              "windows": windows_count, "generated_paths": paths, "coefficients": gate.detach().tolist(),
              "fitted": fitted,
              "style": report_counts(data[0], gate[0]), "rewrite": report_counts(data[1], gate[1])}
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "scene_continuity_fit": result}, path)
    result["checkpoint_sha256"] = sha(path)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
