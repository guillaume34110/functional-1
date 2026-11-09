"""Calibrate clear onset from proved locks and full rows on still-moving pieces."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scene_model import ClearAnimationSceneDynamics, clear_onset_scene_weights, scene_from_weights
from train_event_replay import BURN, Window, replay_windows
from train_mode_correction import proven_composition
from train_rewrite_evidence import load_events
from train_row_transport import verified_background
from train_spawn_preview import sha, training_captures


SPAN = 8


def label_windows(model, captures, events):
    windows = defaultdict(list)
    locks = provisional = 0
    for name, frames, actions, labels in captures:
        board = frames[:, :, 2:12]
        full = torch.isin(board[:-1], model.object_tiles).all(2).any(1)
        mode, known, old, _, new = proven_composition(model, frames, labels)
        style = model.object_tiles[labels["shape_id"].clamp_min(0)]
        continuity = known & old & new & (mode == 0) & (style[:-1] == style[1:])
        changed = (labels["cells"][1:] != labels["cells"][:-1]).flatten(1).any(1)
        moving = continuity & changed
        for _ in range(4):
            moving[:-1] |= continuity[:-1] & ~changed[:-1] & moving[1:]
        target = torch.where(full & moving, 1, -1)
        previous = torch.cat((board[:1], board[:-2]))
        score = model.animation_onset(model.onset_features(board[:-1], previous))[:, 0]
        ordinary = (~full).nonzero().flatten()
        if len(ordinary):
            best = ordinary[score[ordinary].topk(min(6, len(ordinary))).indices]
            target[best] = 2
        label_arrays = {key: value.numpy() for key, value in labels.items()}
        for event in events.get(name, ()):
            verified_background(event, board.numpy(), label_arrays, model)
            start = event["first_lock_frame"]
            if not 0 <= start < len(actions) or not full[start]:
                raise ValueError("proved lock lacks a full row")
            target[start:event["confirmation_frame"]] = -1
            target[start] = 0
            locks += 1
        provisional += int((target == 1).sum())
        starts = {max(0, min(int(t) // SPAN * SPAN, len(actions) - SPAN))
                  for t in (target >= 0).nonzero().flatten()}
        for start in sorted(starts):
            end = start + SPAN
            begin = max(0, start - BURN)
            windows[start-begin].append(Window(frames[begin:start], actions[begin:start], frames[start:end+1],
                                              actions[start:end], target[start:end]))
    if locks != sum(map(len, events.values())) or not provisional:
        raise ValueError("incomplete lock or provisional-row coverage")
    return windows, locks, provisional


@torch.no_grad()
def collect(base, model, captures, events):
    windows, locks, provisional = label_windows(base, captures, events)
    samples = []

    def disabled(tokens, _):
        return tokens.new_full((len(tokens),), -torch.inf, dtype=torch.float32)

    with patch.object(base, "animation_start_logit", disabled), patch.object(base, "terminal_start_logit", disabled):
        for cohort in windows.values():
            for frame, _, details, labels, _, generated in replay_windows(base, cohort):
                use = labels >= 0
                if use.any():
                    samples.append((model.animation_contact_features(frame, details)[use],
                                    ClearAnimationSceneDynamics.animation_start_logit(base, frame, details)[use],
                                    labels[use], torch.full_like(labels[use], generated, dtype=torch.bool)))
    if not samples:
        raise ValueError("no fitted clear-onset examples")
    return tuple(torch.cat(col) for col in zip(*samples)), sum(map(len, windows.values())), locks, provisional


@torch.no_grad()
def measure(logits, group, generated):
    names = ("proved_lock", "moving_provisional_row", "ordinary")
    return {name: {"samples": int(mask.sum()), "onsets": int((mask & (logits > 0)).sum()),
                   "generated_samples": int((mask & generated).sum())}
            for i, name in enumerate(names) for mask in (group == i,)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    model = clear_onset_scene_weights(checkpoint["model"]).eval()
    events = load_events(args.events, checkpoint, "train")
    captures = training_captures(checkpoint["config"], base.codebook.numpy())
    cache, windows, locks, provisional = collect(base, model, captures, events)
    features, raw, group, generated = cache
    counts = torch.bincount(group, minlength=3)
    if not counts.all() or int(((group == 0) & ~generated).sum()) < locks:
        raise ValueError("missing clear onset supervision")
    importance = torch.tensor([1.4, 1., .4])[group] / counts[group]
    head = model.animation_contact_gate
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = raw + head(features)[:, 0]
        loss = F.binary_cross_entropy_with_logits(logits, (group == 0).float(), weight=importance, reduction="sum")
        loss += .0002 * head.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    state = model.state_dict()
    for key, value in checkpoint["model"].items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "events_sha256": sha(args.events), "windows": windows,
              "proved_train_locks": locks, "proved_provisional_frames": provisional,
              "coefficients": head.weight.detach()[0].tolist(), "before": measure(raw, group, generated),
              "after": measure(raw + head(features)[:, 0], group, generated)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "clear_onset_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
