"""Fit one compose residual from proved train transitions and generated motion."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit_controls import BUTTONS, direction_oracle, observed_object, transition_oracle
from audit_long import recorded_path
from model import ACTIONS
from scene_model import compose_contact_scene_weights, scene_from_weights
from train_event_replay import BURN, HORIZON, Window, event_targets, replay_windows
from train_spawn_preview import sha, training_captures


def known_play(details):
    play = details["contact_visible"].clone()
    for name in ("animation", "terminal"):
        if name in details:
            play[details[name]["active"]] = False
    return play


def observed_windows(base, captures):
    windows = []
    for _, frames, actions, labels in captures:
        if len(actions) < BURN + HORIZON:
            continue
        targets = event_targets(base, frames, labels)
        old, mode, new, _ = targets.unbind(1)
        valid = (old >= 0) & (new >= 0) & ((mode == 0) | (mode == 1))
        mode.masked_fill_(~valid, -1)
        movement = (mode[BURN:] == 0).nonzero().flatten() + BURN
        if len(movement) == 0:
            continue
        locks = (mode[BURN:] == 1).nonzero().flatten() + BURN
        sampled = movement[torch.arange(1, 9) * len(movement) // 9]
        starts = {max(BURN, min(int(t) // HORIZON * HORIZON, len(actions) - HORIZON))
                  for t in torch.cat((locks, sampled))}
        for start in sorted(starts):
            end = start + HORIZON
            windows.append(Window(frames[start-BURN:start], actions[start-BURN:start], frames[start:end+1],
                                  actions[start:end], targets[start:end]))
    return windows


def proved_copy(base, tokens, old, details, event):
    if not bool(known_play(details)[0]) or details["contact_neighbors"][0, 7] != 0:
        return False
    mask, color = base._hard_object(details["current"])
    current = mask[0, :, 2:12].nonzero().numpy()
    if set(map(tuple, current)) != set(map(tuple, old)):
        return False
    button = ACTIONS[event["button"]]
    if button not in BUTTONS:
        return False
    mode = torch.full_like(details["mode"], -torch.inf)
    mode[:, 0] = 0
    joint = details["event_scores"][:, 0].log_softmax(1).reshape_as(details["next"]["joint"])
    copy = {**details, "mode": mode, "next": {**details["next"], "joint": joint}}
    rendered = base.decode_scene(tokens, copy, hard=True)
    new, style = observed_object(base, copy, rendered)
    if new is None or style != int(color[0]):
        return False
    board, predicted = (x[0, :, 2:12] for x in (tokens, rendered))
    if not torch.isin(predicted, torch.cat((base.object_tiles, predicted.new_tensor([base.blank_id])))).all():
        return False
    geometry = transition_oracle(board.numpy(), predicted.numpy(), old, new, base.blank_id)
    return geometry["status"] == "motion" and direction_oracle(old, new, button, 0, event["previous_button"])[0] is None


@torch.inference_mode()
def generated_negatives(base, model, captures):
    samples = []
    paths = 0
    horizon = 96
    for _, frames, actions, labels in captures:
        if len(actions) < BURN + horizon:
            continue
        evidence = labels["evidence"][:-1]
        visible = (labels["shape_id"][:-1] >= 0) & ((evidence == 1) | (evidence == 2))
        visible &= (labels["cells"][:-1, :, 0].amin(1) >= 5)
        anchors = visible[BURN:len(actions)-horizon+1].nonzero().flatten() + BURN
        if not len(anchors):
            continue
        starts = {int(anchors[q * len(anchors) // 5]) for q in (1, 2, 3, 4)}
        for start in sorted(starts):
            state = None
            for t in range(start - BURN, start):
                state = base.advance_observed(frames[t:t+1], actions[t:t+1], state)

            def observe(tokens, old, details, event):
                geometry = event.get("previous_oracle", event)
                if (event["mode"] == 1 and geometry.get("reason") == "unsupported-deposit"
                        and proved_copy(base, tokens, old, details, event)):
                    samples.append((model.compose_contact_signal(details), details["mode"], torch.tensor([2])))
                return event["status"] in ("motion", "deposit")

            recorded_path(base, frames[start:start+1], state, labels["cells"][start].numpy(),
                          actions[start:start+horizon], ACTIONS[int(actions[start-1])], observer=observe)
            paths += 1
    return samples, paths


@torch.no_grad()
def collect(base, model, captures):
    windows = observed_windows(base, captures)
    samples = []
    for _, _, details, targets, _, _ in replay_windows(base, windows):
        old, mode, _, _ = targets.unbind(1)
        current = details["current"]["joint"].flatten(1).argmax(1)
        use = (mode >= 0) & (old == current) & known_play(details)
        if use.any():
            samples.append((model.compose_contact_signal(details)[use], details["mode"][use], mode[use]))
    generated, paths = generated_negatives(base, model, captures)
    if not samples:
        raise ValueError("no observed or aligned train transitions")
    samples.extend(generated)
    return tuple(torch.cat(column) for column in zip(*samples)), len(windows), paths


def metrics(signal, scores, group):
    proposed = scores.argmax(1)
    return {name: {"samples": int((group == k).sum()),
                   "unsupported": int(((group == k) & (signal[:, 0] > 0)).sum()),
                   "correct": int(((group == k) & (proposed == (k == 1))).sum()),
                   "compose": int(((group == k) & (proposed == 1)).sum())}
            for k, name in enumerate(("proved_motion", "proved_deposit", "generated_motion"))}


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
    model = compose_contact_scene_weights(checkpoint["model"]).eval()
    captures = list(training_captures(checkpoint["config"], base.codebook.numpy()))
    (signal, raw, group), windows, paths = collect(base, model, captures)
    counts = torch.bincount(group, minlength=3)
    if not counts[:2].all() or not ((group != 1) & (signal[:, 0] > 0) & (raw.argmax(1) == 1)).any():
        raise ValueError("training lacks proved deposits or unsupported compose errors")
    weights = 1. / counts[group]
    target = (group == 1).long()
    gate = model.compose_contact_gate
    optimizer = torch.optim.LBFGS(gate.parameters(), max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        scores = raw + F.pad(gate(signal), (1, 1))
        loss = (F.cross_entropy(scores, target, reduction="none") * weights).sum()
        loss += .0003 * gate.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    state = model.state_dict()
    for key, value in checkpoint["model"].items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "windows": windows, "generated_paths": paths,
              "coefficient": float(gate.weight.detach()[0, 0]), "before": metrics(signal, raw, group),
              "after": metrics(signal, raw + F.pad(gate(signal), (1, 1)), group)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "compose_contact_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
