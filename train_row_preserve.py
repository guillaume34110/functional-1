"""Learn to choose between the network's proposed and preserved background rows."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scene_model import row_preserve_scene_weights
from train_row_transport import verified_background
from train_scene import label_mask
from train_spawn_preview import sha, training_captures


@torch.no_grad()
def collect(model, captures, events):
    features, copy_errors, write_errors, clear = [], [], [], []
    counts = Counter()
    for name, frames, actions, labels in captures:
        boards = frames[:, :, 2:12].numpy()
        labels_np = {key: value.numpy() for key, value in labels.items()}
        ends = dict(verified_background(event, boards, labels_np, model) for event in events.get(name, ()))
        if len(ends) != len(events.get(name, ())):
            raise ValueError("duplicate clear target")
        counts["clear"] += len(ends)
        valid = torch.ones(len(actions), dtype=torch.bool)
        old, old_mask = label_mask({k: v[:-1, None] for k, v in labels.items()}, 0, valid)
        new, next_mask = label_mask({k: v[1:, None] for k, v in labels.items()}, 0, valid)
        before = frames[:-1].masked_fill(old_mask, model.blank_id)[:, :, 2:12]
        after = frames[1:].masked_fill(next_mask, model.blank_id)[:, :, 2:12]
        known_copy = old & new & (before == after).flatten(1).all(1)
        state = None
        for t, command in enumerate(actions):
            frame, action = frames[t:t + 1], command[None]
            is_clear = t in ends
            if not is_clear and not (t % 4 == 0 and known_copy[t]):
                state = model.advance(frame, action, state)
                continue
            target = torch.from_numpy(ends[t]) if is_clear else after[t]
            details = model.scene_details(frame, action, state)
            state = details["state"]
            copy, write = model.row_preserve_candidates(frame, details)
            copy_error = (copy[0, :, 2:12] != target).sum(1)
            write_error = (write[0, :, 2:12] != target).sum(1)
            features.append(details["row_preserve_features"][0])
            copy_errors.append(copy_error)
            write_errors.append(write_error)
            clear.append(is_clear)
            counts["copy"] += not is_clear
    if counts["clear"] != sum(map(len, events.values())):
        raise ValueError("missing verified clear examples")
    return torch.stack(features), torch.stack(copy_errors), torch.stack(write_errors), torch.tensor(clear), counts


@torch.no_grad()
def metrics(head, features, copy, write, clear):
    errors = torch.where(head(features) > 0, copy, write)
    return {name: {"rows": int(mask.sum()) * 18, "tile_errors": int(errors[mask].sum()),
                   "exact_backgrounds": int((errors[mask].sum(1) == 0).sum())}
            for name, mask in (("copy", ~clear), ("clear", clear))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(51)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    evidence = json.loads(args.events.read_text())
    if (evidence["split"] != "train" or evidence["test_used"] or evidence["validation_used"]
            or any(evidence[k] != checkpoint["config"][k] for k in ("manifest_sha256", "annotations_sha256"))):
        raise ValueError("row gate fit requires verified training evidence")
    events = {}
    for event in evidence["events"]:
        events.setdefault(event["file"], []).append(event)
    model = row_preserve_scene_weights(checkpoint["model"]).eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("row_preserve_head."))
    head = model.row_preserve_head
    features, copy, write, clear, counts = collect(
        model, training_captures(checkpoint["config"], model.codebook.numpy()), events)
    before = metrics(head, features, copy, write, clear)
    usable = copy != write
    targets = copy < write
    labels = targets[usable].float()
    if not 0 < labels.sum() < len(labels):
        raise ValueError("both row choices need training examples")
    copy_weight = (len(labels) / labels.sum()).sqrt()
    write_weight = (len(labels) / (len(labels) - labels.sum())).sqrt()
    frames_with_signal = usable.any(1).nonzero().flatten()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003)
    for _ in range(20):
        for batch in frames_with_signal[torch.randperm(len(frames_with_signal))].split(128):
            mask = usable[batch]
            logits = head(features[batch])[mask]
            target = targets[batch][mask].float()
            weights = torch.where(target > 0, copy_weight, write_weight)
            loss = F.binary_cross_entropy_with_logits(logits, target, weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
    state = model.state_dict()
    for name, previous in checkpoint["model"].items():
        torch.testing.assert_close(state[name], previous, rtol=0, atol=0)
    report = {"fit_split": "train", "validation_used": False, "test_used": False,
              "checkpoint_source_sha256": sha(args.checkpoint), "evidence_sha256": sha(args.events),
              "coverage": counts, "comparable_rows": len(labels), "prefer_copy": int(labels.sum()),
              "before": before, "after": metrics(head, features, copy, write, clear)}
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "row_preserve_fit": report}, output)
    report["checkpoint_sha256"] = sha(output)
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
