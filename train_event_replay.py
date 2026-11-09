"""Fit joint event scores on verified train sequences and aligned generated screens."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import NamedTuple

import torch
from torch.nn import functional as F

from intervention_training import BUTTONS, load_interventions
from scene_model import replay_event_scene_weights, scene_from_weights
from train_mode_correction import proven_composition
from train_rewrite_evidence import load_events
from train_row_transport import verified_background
from train_spawn_preview import sha, training_captures


BURN, HORIZON = 32, 24
EVENTS = ("wait", "translation", "rotation", "deposit", "rewrite", "proved_clear")


class Window(NamedTuple):
    prefix: torch.Tensor
    prefix_actions: torch.Tensor
    frames: torch.Tensor
    actions: torch.Tensor
    targets: torch.Tensor


def event_targets(model, frames, labels, clears=()):
    mode, known, old, _, new = proven_composition(model, frames, labels)
    y, x = labels["cells"].amin(1).unbind(1)
    kind = labels["shape_id"]
    pose = kind * 180 + y * 10 + x
    group = torch.where(kind[1:] != kind[:-1], 2, (pose[1:] != pose[:-1]).long())
    group = torch.where(mode > 0, mode + 2, group)
    mode = mode.masked_fill(~known, -1)
    if clears:
        mode[list(clears)] = 2
        group[list(clears)] = 5
    return torch.stack((pose[:-1].masked_fill(~old, -1), mode, pose[1:].masked_fill(~new, -1), group), dim=1)


def windows(model, parents, events, bundles):
    by_episode = defaultdict(list)
    for episode, anchor in sorted(bundles):
        by_episode[episode].append(anchor)
    recorded, intervention = [], []
    witnessed = 0
    for episode, (name, frames, actions, labels) in enumerate(parents):
        boards = frames[:, :, 2:12].numpy()
        label_values = {key: value.numpy() for key, value in labels.items()}
        clears = [verified_background(event, boards, label_values, model)[0] for event in events.get(name, ())]
        targets = event_targets(model, frames, labels, clears)
        candidates = (targets[BURN:len(actions) - HORIZON, 1] >= 0).nonzero().flatten() + BURN
        anchors = [int(candidates[len(candidates) * q // 5]) for q in range(1, 5)] if len(candidates) else []
        starts = {max(BURN, t - 8) for t in anchors}
        starts.update(max(BURN, min(t - HORIZON + 1, len(actions) - HORIZON)) for t in clears)
        witnessed += sum(any(start <= t < start + HORIZON for start in starts) for t in clears)
        for start in sorted(starts):
            end = start + HORIZON
            recorded.append(Window(frames[start-BURN:start], actions[start-BURN:start],
                                   frames[start:end+1], actions[start:end], targets[start:end]))
        branch_anchors = by_episode[episode]
        if not branch_anchors:
            continue
        for start in sorted({branch_anchors[len(branch_anchors) * q // 4] for q in (1, 3)}):
            bundle = bundles[episode, start]
            for name in ("noop", "a", "b", "down"):
                arm = BUTTONS.index(name)
                images = torch.from_numpy(bundle["tokens"][arm]).long()
                commands = torch.from_numpy(bundle["actions"][arm]).long()
                branch_labels = {key: torch.from_numpy(value[arm]).long() for key, value in bundle["labels"].items()}
                intervention.append(Window(frames[start-BURN:start], actions[start-BURN:start], images,
                                           commands, event_targets(model, images, branch_labels)))
    if witnessed != sum(map(len, events.values())) or not recorded or not intervention:
        raise ValueError("incomplete train window or clear evidence coverage")
    return recorded, intervention, witnessed


def compress_scores(model, scores, current, mode, target):
    # Copy poses in one bin receive the same correction, so grouping preserves their probability mass.
    bins = model.replay_pose_bins(current)
    copy = scores[:, 0]
    peaks = copy.new_full((len(copy), 98), -torch.inf).scatter_reduce(1, bins, copy, reduce="amax")
    weights = (copy - peaks.gather(1, bins)).exp().nan_to_num(nan=0)
    totals = torch.zeros_like(peaks).scatter_add(1, bins, weights)
    masses = torch.cat((peaks + totals.log(), scores[:, 1:].logsumexp(2)), dim=1)
    peaks = torch.cat((peaks, scores[:, 1:].amax(2)), dim=1)
    rows = torch.arange(len(scores))
    safe_target = target.clamp_min(0)
    gold_bin = torch.where(mode == 0, bins[rows, safe_target], 97 + mode)
    gold_score = scores[rows, mode, safe_target]
    if not torch.isfinite(gold_score[target >= 0]).all():
        raise ValueError("proved target pose outside the event vocabulary")
    return masses, peaks, gold_bin, gold_score


@torch.no_grad()
def replay_windows(base, windows):
    for start in range(0, len(windows), 16):
        prefix, previous, frames, actions, targets = map(torch.stack, zip(*windows[start:start+16]))
        teacher = None
        for t in range(prefix.size(1)):
            teacher = base.advance_observed(prefix[:, t], previous[:, t], teacher)
        live = torch.arange(len(frames))
        for t in range(actions.size(1)):
            frame, action = frames[:, t], actions[:, t]
            details = base.scene_details(frame, action, teacher)
            yield frame, action, details, targets[:, t], frames[:, t+1], False
            prediction = base.decode_scene(frame, details, hard=True)
            teacher = base.rendered_state(details, prediction)
            if t == 0:
                generated, carried = prediction, teacher
            elif len(live):
                same = (generated[:, :, 2:12] == frame[live, :, 2:12]).flatten(1).all(1)
                live = live[same]
                if not len(live):
                    continue
                generated, carried = generated[same], carried[same]
                rollout = base.scene_details(generated, action[live], carried)
                yield generated, action[live], rollout, targets[live, t], frames[live, t+1], True
                generated = base.decode_scene(generated, rollout, hard=True)
                carried = base.rendered_state(rollout, generated)


@torch.no_grad()
def collect(base, model, cohorts):
    samples = []
    for cohort, windows in enumerate(cohorts):
        for _, action, details, targets, _, generated in replay_windows(base, windows):
            current = details["current"]["joint"].flatten(1).argmax(1)
            old, mode, pose, group = targets.unbind(1)
            known = (mode >= 0) & ((current == old) | ((old < 0) & (mode > 0)) | (group == 5))
            if not known.any():
                continue
            features, tracked = model.replay_features(details, action, current)
            fields = compress_scores(model, details["event_scores"][known], current[known], mode[known], pose[known])
            samples.append((features[known], tracked[known], *fields, mode[known], pose[known], group[known],
                            torch.full_like(group[known], 2 * cohort + generated)))
    if not samples:
        raise ValueError("no qualified train events")
    return tuple(torch.cat(columns) for columns in zip(*samples))


def batch(model, cache, ids):
    features, tracked, mass, peak, gold_bin, gold, mode, pose, _, _ = (field[ids] for field in cache)
    offsets = model.replay_offsets(features, tracked)
    mass, peak = mass + offsets, peak + offsets
    mode_mass = torch.cat((mass[:, :-2].logsumexp(1, keepdim=True), mass[:, -2:]), dim=1)
    mode_peak = torch.cat((peak[:, :-2].amax(1, keepdim=True), peak[:, -2:]), dim=1)
    rows = torch.arange(len(ids))
    gold = gold + offsets[rows, gold_bin]
    mass_loss = mode_mass.logsumexp(1) - torch.where(pose >= 0, gold, mode_mass[rows, mode])
    mode_loss = F.cross_entropy(mode_peak, mode, reduction="none")
    mode_ok = mode_peak.argmax(1) == mode
    pose_ok = (peak.argmax(1) == gold_bin) & (peak[rows, gold_bin] == gold)
    return mass_loss + .25 * mode_loss, mode_ok, mode_ok & ((pose < 0) | pose_ok), offsets


@torch.no_grad()
def metrics(model, cache):
    groups, source = cache[-2:]
    loss, mode, joint = map(torch.cat, zip(*(batch(model, cache, ids)[:3]
                                            for ids in torch.arange(len(groups)).split(128))))
    masks = {name: groups == i for i, name in enumerate(EVENTS)}
    masks["autonomous"] = source.remainder(2) == 1
    return {name: {"examples": int(mask.sum()), "loss": float(loss[mask].mean()),
                   "mode_correct": int(mode[mask].sum()), "joint_correct": int(joint[mask].sum())}
            for name, mask in masks.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(70)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    model = replay_event_scene_weights(checkpoint["model"]).eval()
    config = checkpoint["config"]
    parents = list(training_captures(config, base.codebook.numpy()))
    manifest_file = Path("data/manifest.json")
    manifest = json.loads(manifest_file.read_bytes())
    if [name for name, *_ in parents] != [row["file"] for row in manifest["episodes"] if row["split"] == "train"]:
        raise ValueError("training parent order differs")
    episodes = {"train": [(frames.numpy(), actions.numpy()) for _, frames, actions, _ in parents]}
    annotations = {"train": [{k: v.numpy() for k, v in labels.items()} for _, _, _, labels in parents]}
    interventions, digest = load_interventions(Path("data/action-interventions-v3"), base, manifest,
                                               episodes, annotations, sha(manifest_file),
                                               config["annotations_sha256"], splits=("train",))
    events = load_events(args.events, checkpoint, "train")
    recorded, branches, clears = windows(base, parents, events, interventions["train"])
    cache = collect(base, model, (recorded, branches))
    group, source = cache[-2:]
    counts = torch.bincount(group, minlength=len(EVENTS))
    if not counts.all() or not torch.bincount(source, minlength=4).all():
        raise ValueError("missing required event or rollout supervision")
    before = metrics(model, cache)
    weights = (counts.max() / counts).pow(.35)
    head = model.replay_calibration
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003)
    for _ in range(8):
        for ids in torch.randperm(len(group)).split(96):
            loss, _, _, offsets = batch(model, cache, ids)
            importance = weights[group[ids]]
            objective = (loss * importance).sum() / importance.sum() + .01 * offsets.square().mean()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
    state = model.state_dict()
    for key, value in checkpoint["model"].items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "interventions_sha256": digest,
              "events_sha256": sha(args.events), "witnessed_train_clears": clears,
              "windows": {"recorded": len(recorded), "intervention": len(branches), "burn_in": BURN,
                          "recorded_steps": HORIZON, "intervention_steps": 8},
              "trainable_parameters": sum(p.numel() for p in head.parameters()),
              "before": before, "after": metrics(model, cache)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": config, "event_replay_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
