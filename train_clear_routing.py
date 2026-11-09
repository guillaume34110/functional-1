"""Learn rendering choices from exact visible backgrounds in train rollouts."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scene_model import clear_routing_scene_weights, scene_from_weights
from train_event_replay import BURN, HORIZON, Window, replay_windows
from train_mode_correction import proven_composition
from train_rewrite_evidence import load_events
from train_row_transport import verified_background
from train_scene import label_mask, visual_absence
from train_spawn_preview import sha, training_captures


ROUTES = ("copy", "deposit", "rewrite", "transport")


def route_windows(model, captures, events):
    windows, witnessed = [], 0
    for name, frames, actions, labels in captures:
        boards = frames[:, :, 2:12].numpy()
        labels_np = {key: value.numpy() for key, value in labels.items()}
        clears = dict(verified_background(event, boards, labels_np, model) for event in events.get(name, ()))
        mode, known, _, _, _ = proven_composition(model, frames, labels)
        valid = torch.ones(len(actions), dtype=torch.bool)
        new, mask = label_mask({key: value[1:, None] for key, value in labels.items()}, 0, valid)
        visible = new | visual_absence(frames[1:], labels["evidence"][1:], valid, model.object_tiles)
        background = frames[1:].masked_fill(mask, model.blank_id)[:, :, 2:12].clone()
        group = torch.where(known & visible, (mode > 0).long(), -1)
        for t, target in clears.items():
            background[t] = torch.from_numpy(target)
            group[t] = 2
        targets = torch.cat((group[:, None], background.flatten(1)), dim=1)
        anchors = (group[BURN:len(actions)-HORIZON] >= 0).nonzero().flatten() + BURN
        ordinary = {int(anchors[len(anchors) * k // 4]) for k in (1, 2, 3)} if len(anchors) else set()
        starts = ordinary | {max(BURN, min(t - 18, len(actions) - HORIZON)) for t in clears}
        witnessed += sum(any(start <= t < start + HORIZON for start in starts) for t in clears)
        for start in sorted(starts):
            end = start + HORIZON
            windows.append(Window(frames[start-BURN:start], actions[start-BURN:start],
                                  frames[start:end+1], actions[start:end], targets[start:end]))
    if witnessed != sum(map(len, events.values())) or not witnessed:
        raise ValueError("incomplete train clear evidence coverage")
    return windows, witnessed


def route_matches(model, frame, details, target, reference):
    row_gate = details["row_transport_gate"]
    raw_details = {**details, "render_transport_gate": torch.full_like(row_gate, -1)}
    copy, raw = model._hard_rewrite_candidates(frame, raw_details, details["gate"] - model.gate_logit_offset)
    backgrounds = torch.stack((copy[:, :, 2:12], frame[:, :, 2:12], raw[:, :, 2:12],
                               model.transported_rows(frame, details)), dim=1)
    exact = (backgrounds == target[:, None]).flatten(2).all(2)
    poses = details["event_scores"].argmax(2)[:, (0, 1, 2, 2)]
    kinds = poses // 180
    mask = model.shape_masks[kinds, poses % 180].bool()
    mask = mask & (details["next"]["visible"][:, :, None, None] > 0)
    rendered = torch.where(mask, model.object_tiles[kinds, None, None], backgrounds)
    exact_screen = (rendered == reference[:, None, :, 2:12]).flatten(2).all(2) & exact
    return exact, torch.where(exact_screen.any(1, keepdim=True), exact_screen, exact)


@torch.no_grad()
def collect(base, model, windows):
    samples = []
    for frame, _, details, targets, reference, generated in replay_windows(base, windows):
        group = targets[:, 0]
        selected = group >= 0
        if not selected.any():
            continue
        # Route evidence is independent of the mode used to render this step.
        exact, preferred = route_matches(base, frame, details, targets[:, 1:].reshape(-1, 18, 10), reference)
        features = model.clear_routing_features(details)
        samples.append(tuple(value[selected] for value in (features, details["mode"], details["row_transport_gate"],
                                                             exact, preferred, group,
                                                             torch.full_like(group, generated, dtype=torch.bool))))
    if not samples:
        raise ValueError("no qualified train render choices")
    return tuple(torch.cat(column) for column in zip(*samples))


def route_log_prob(mode, gate):
    mode = mode.log_softmax(1)
    rewrites = mode[:, 2:] + F.logsigmoid(torch.stack((-gate, gate), dim=1))
    return torch.cat((mode[:, :2], rewrites), dim=1)


@torch.no_grad()
def metrics(head, cache):
    features, mode, gate, exact, preferred, group, generated = cache
    correction = head(features)
    selected = (mode + correction[:, :3]).argmax(1)
    selected = selected + ((selected == 2) & (gate + correction[:, 3] > 0))
    rows = torch.arange(len(selected))
    masks = {"copy_continuity": group == 0, "deposit_or_rewrite": group == 1,
             "proved_clear": group == 2, "generated": generated}
    return {name: {"examples": int(mask.sum()), "exact": int((mask & exact[rows, selected]).sum()),
                   "preferred": int((mask & preferred[rows, selected]).sum()),
                   "renderable": int((mask & exact.any(1)).sum()),
                   "matching_routes": exact[mask].sum(0).tolist(),
                   "selected_routes": torch.bincount(selected[mask], minlength=len(ROUTES)).tolist()}
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    model = clear_routing_scene_weights(checkpoint["model"]).eval()
    events = load_events(args.events, checkpoint, "train")
    windows, witnessed = route_windows(base, training_captures(checkpoint["config"], base.codebook.numpy()), events)
    cache = collect(base, model, windows)
    features, mode, gate, _, preferred, group, generated = cache
    qualified = preferred.any(1)
    counts = torch.bincount(group[qualified], minlength=3)
    if not counts.all() or not (generated & qualified).any():
        raise ValueError("missing renderable train events or aligned generated samples")
    weights = counts.float().rsqrt()[group[qualified]]
    weights = weights / weights.sum()
    head = model.clear_routing
    before = metrics(head, cache)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        correction = head(features)
        change = correction[qualified]
        log_prob = route_log_prob(mode[qualified] + change[:, :3], gate[qualified] + change[:, 3])
        nll = -log_prob.masked_fill(~preferred[qualified], -torch.inf).logsumexp(1)
        loss = (weights * nll).sum() + .02 * correction.square().mean() + .0005 * head.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    state = model.state_dict()
    for key, value in checkpoint["model"].items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "events_sha256": sha(args.events),
              "witnessed_train_clears": witnessed, "windows": len(windows), "routes": list(ROUTES),
              "trainable_parameters": sum(p.numel() for p in head.parameters()),
              "before": before, "after": metrics(head, cache)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "clear_routing_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
