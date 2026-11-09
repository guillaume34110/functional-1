"""Calibrate terminal scores on witnessed contact and still-playable train replays."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from audit_long import recorded_path
from model import ACTIONS
from scene_model import scene_from_weights, terminal_contact_scene_weights
from train_event_replay import BURN, HORIZON, Window, replay_windows
from train_spawn_preview import sha, training_captures
from train_terminal import observed_ages, terminal_witness


@torch.no_grad()
def observed_windows(model, captures, witnesses):
    windows = []
    for name, frames, actions, _ in captures:
        board = frames[:-1, :, 2:12]
        raw = model.terminal_onset(model.terminal_features(board, observed_ages(board)))[:, 0]
        labels = torch.zeros(len(actions), dtype=torch.long)
        if name in witnesses:
            event = terminal_witness(model, frames)
            if event is None or tuple(witnesses[name]) != event[:4]:
                raise ValueError("terminal proof differs from the fitted source")
            start, wipe, *_ = event
            labels[start:wipe] = 1
            labels[wipe:] = -1
        selected = ((raw > -4) | (labels == 1)) & (labels >= 0)
        starts = {max(BURN, min(int(index) // HORIZON * HORIZON, len(actions) - HORIZON))
                  for index in selected.nonzero().flatten()}
        labels = torch.where(selected, labels, -1)
        for start in sorted(starts):
            end = start + HORIZON
            windows.append(Window(frames[start-BURN:start], actions[start-BURN:start], frames[start:end+1],
                                  actions[start:end], labels[start:end]))
    return windows


@torch.no_grad()
def replay_negatives(base, model, driver, captures):
    negatives = []
    paths = 0
    for _, frames, actions, labels in captures:
        board = frames[:-1, :, 2:12]
        near = (labels["cells"][:-1, :, 0].amin(1) <= 4) & (labels["shape_id"][:-1] >= 0)
        proved = near & ((labels["evidence"][:-1] == 1) | (labels["evidence"][:-1] == 2))
        proved &= (board != base.blank_id).flatten(1).sum(1) >= 16
        eligible = proved[BURN:len(actions)-128].nonzero().flatten() + BURN
        if not len(eligible):
            continue
        anchors = {int(eligible[len(eligible) * part // 3]) for part in (1, 2)}
        for start in sorted(anchors):
            state = None
            for t in range(start - BURN, start):
                state = driver.advance_observed(frames[t:t+1], actions[t:t+1], state)
            risks = []

            def observe(tokens, old, details, event):
                if event["status"] not in ("motion", "deposit"):
                    return False
                if event["status"] == "motion" and old is not None and "animation" not in details:
                    raw = base.terminal_start_logit(tokens, details)
                    if raw.item() > -3:
                        risks.append((event["step"], old.tolist(), model.terminal_contact_features(tokens, details)[0],
                                      raw[0]))
                return True

            cells = labels["cells"][start].numpy()
            events, _ = recorded_path(driver, frames[start:start+1], state, cells, actions[start:start+160],
                                      ACTIONS[int(actions[start-1])], observer=observe)
            for index, old, features, raw in risks:
                for event in events[index:index+16]:
                    if event["status"] != "motion" or event["style"] != events[index]["style"]:
                        break
                    if event["cells"] != old:
                        negatives.append((features, raw))
                        break
            paths += 1
    if not negatives:
        raise ValueError("no generated high-score terminal proposals with a proved continuation")
    features, raw = map(torch.stack, zip(*negatives))
    return features, raw, paths


@torch.no_grad()
def collect(base, model, driver, captures, witnesses):
    windows = observed_windows(base, captures, witnesses)
    samples = []
    for frame, _, details, labels, _, _ in replay_windows(driver, windows):
        use = labels >= 0
        animation = details.get("animation")
        if animation is not None:
            use[animation["active"]] = False
        if use.any():
            group = torch.where(labels[use] == 1, 0, 1)
            samples.append((model.terminal_contact_features(frame, details)[use],
                            base.terminal_start_logit(frame, details)[use], group))
    if not samples:
        raise ValueError("no qualified observed terminal contexts")
    features, raw, group = map(torch.cat, zip(*samples))
    replay, replay_scores, paths = replay_negatives(base, model, driver, captures)
    return (torch.cat((features, replay)), torch.cat((raw, replay_scores)),
            torch.cat((group, torch.full((len(replay),), 2, dtype=torch.long)))), len(windows), paths


@torch.no_grad()
def measure(scores, group):
    return {name: {"samples": int((group == k).sum()), "terminal_proposals": int(((scores > 0) & (group == k)).sum())}
            for k, name in enumerate(("proved_terminal", "observed_or_aligned_play", "generated_continuation"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    driver_checkpoint = torch.load(args.driver, map_location="cpu", weights_only=True)
    if sha(args.driver) != checkpoint["terminal_fit"]["source_sha256"] or driver_checkpoint["config"] != checkpoint["config"]:
        raise ValueError("driver differs from the frozen nonterminal source")
    base = scene_from_weights(checkpoint).eval()
    driver = scene_from_weights(driver_checkpoint).eval()
    model = terminal_contact_scene_weights(checkpoint["model"]).eval()
    captures = list(training_captures(checkpoint["config"], base.codebook.numpy()))
    cache, windows, paths = collect(base, model, driver, captures, checkpoint["terminal_fit"]["witnesses"])
    features, raw, group = cache
    counts = torch.bincount(group, minlength=3)
    if not counts.all():
        raise ValueError("terminal contact training lacks a supervision group")
    weight = torch.tensor([1., .7, 1.3])[group] / counts[group]
    head = model.terminal_contact_gate
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = raw + head(features)[:, 0]
        loss = F.binary_cross_entropy_with_logits(logits, (group == 0).float(), weight=weight, reduction="sum")
        loss += .0003 * head.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    state = model.state_dict()
    for key, value in checkpoint["model"].items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "driver_sha256": sha(args.driver), "observed_windows": windows,
              "generated_paths": paths, "coefficients": head.weight.detach()[0].tolist(),
              "before": measure(raw, group), "after": measure(raw + head(features)[:, 0], group)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "terminal_contact_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
