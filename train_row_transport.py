"""Fit a row pointer on visually verified training clear events."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit_clear_events import board_hash, exact_foreground_pose, foreground_background
from audit_row_transport import compacted_background
from scene_model import RowTransportSceneDynamics, row_transport_scene_weights, scene_from_weights
from train_scene import label_mask
from train_spawn_preview import observed_captures, sha, training_captures


def matching_sources(source, target, blank):
    candidates = np.concatenate((source, np.full_like(source[:1], blank)))
    return (target[:, None] == candidates[None]).all(2)


def verified_background(event, boards, labels, model):
    lock = event["last_consecutive_lock_frame"]
    outcome = event["first_outcome_frame"]
    confirmed = event["confirmation_frame"]
    if not 0 <= lock < outcome < confirmed == outcome + 1 < len(boards):
        raise ValueError("clear event indices outside capture")
    if board_hash(boards[lock]) != event["source_board_sha256"]:
        raise ValueError("clear lock digest mismatch")
    for i, t in enumerate((outcome, confirmed)):
        if board_hash(boards[t]) != event["outcome_board_sha256"][i]:
            raise ValueError("clear outcome digest mismatch")
        proof = event["outcome_proof"][i]
        if proof == "masked_proved_foreground":
            background = foreground_background(boards[t], labels, model.object_shapes.numpy(),
                                               model.object_tiles.numpy(), model.blank_id, t)
        elif proof == "raw":
            background = boards[t]
        elif proof in ("masked_exact_foreground", "masked_clipped_foreground"):
            background = compacted_background(boards[lock], model.blank_id)
            if background is None:
                raise ValueError("visible overlay has no proved compacted source")
            pose = exact_foreground_pose(boards[t], background, model.object_shapes.numpy(),
                                        model.object_tiles.numpy(), model.blank_id)
            supported = pose == -2 if proof == "masked_clipped_foreground" else pose is not None and pose >= 0
            if not supported:
                raise ValueError("clear foreground contradicts visible overlay proof")
        else:
            raise ValueError(f"unsupported outcome proof: {proof}")
        if background is None or board_hash(background) != event["proved_background_sha256"]:
            raise ValueError("clear background digest mismatch")
    return outcome - 1, background


def verified_outcome(event, boards, labels, model):
    step, background = verified_background(event, boards, labels, model)
    return step, matching_sources(boards[step], background, model.blank_id)


@torch.no_grad()
def cache_examples(model, captures, events):
    features, matches, positive, base_rewrite = [], [], [], []
    counts = Counter()
    for name, frames, actions, labels in captures:
        if name not in events:
            continue
        boards = frames[:, :, 2:12].numpy()
        labels_np = {key: value.numpy() for key, value in labels.items()}
        positive_at = {}
        flash_at = set()
        for event in events[name]:
            t, possible = verified_outcome(event, boards, labels_np, model)
            counts["proved_events"] += 1
            if possible.any(1).all():
                if t in positive_at:
                    raise ValueError("duplicate clear transition")
                positive_at[t] = possible
            else:
                counts["unrepresentable_events"] += 1
            flash_at.update(index - 1 for index in event["flash_frames"] if index - 1 < t)
        valid = torch.ones(len(actions), dtype=torch.bool)
        old, old_mask = label_mask({k: v[:-1, None] for k, v in labels.items()}, 0, valid)
        new, next_mask = label_mask({k: v[1:, None] for k, v in labels.items()}, 0, valid)
        before = frames[:-1].masked_fill(old_mask, model.blank_id)[:, :, 2:12]
        after = frames[1:].masked_fill(next_mask, model.blank_id)[:, :, 2:12]
        copy = old & new & (before == after).flatten(1).all(1)
        state = None
        for t, command in enumerate(actions):
            frame, action = frames[t:t + 1], command[None]
            is_positive = t in positive_at
            if not (is_positive or t in flash_at or (t % 4 == 0 and copy[t])):
                state = model.advance(frame, action, state)
                continue
            details = model.scene_details(frame, action, state)
            state = details["state"]
            features.append(model.row_transport_features(frame, details)[0])
            positive.append(is_positive)
            matches.append(torch.from_numpy(positive_at[t]) if is_positive else torch.zeros(18, 19, dtype=torch.bool))
            base_rewrite.append(int(details["mode"].argmax(1)) == 2)
            counts["positive" if is_positive else "flash_negative" if t in flash_at else "copy_negative"] += 1
    if counts["positive"] == 0 or counts["proved_events"] != sum(map(len, events.values())):
        raise ValueError(f"missing representable or verified transport data: {dict(counts)}")
    return torch.stack(features), torch.stack(matches), torch.tensor(positive), torch.tensor(base_rewrite), counts


@torch.no_grad()
def metrics(head, features, matches, positive, rewrite):
    logits, gate = head(features)
    chosen = logits[positive].argmax(2, keepdim=True)
    correct = matches[positive].gather(2, chosen)[:, :, 0]
    return {"correct_rows": int(correct.sum()), "target_rows": correct.numel(),
            "correct_boards": int(correct.all(1).sum()), "positive_boards": int(positive.sum()),
            "positive_gate": int(((gate > 0) & positive).sum()),
            "false_gate": int(((gate > 0) & ~positive).sum()),
            "positive_routed": int(((gate > 0) & positive & rewrite).sum()),
            "routed_correct_boards": int((correct.all(1) & (gate[positive] > 0) & rewrite[positive]).sum()),
            "false_routed": int(((gate > 0) & ~positive & rewrite).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--evaluate", action="store_true", help="measure a fitted model on validation clear events")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(47)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    evidence = json.loads(args.events.read_text())
    split = "validation" if args.evaluate else "train"
    if (evidence["split"] != split or evidence["test_used"] or evidence["validation_used"] != args.evaluate
            or any(evidence[key] != checkpoint["config"][key]
                   for key in ("manifest_sha256", "annotations_sha256"))):
        raise ValueError("clear-event dataset provenance mismatch")
    events = defaultdict(list)
    for event in evidence["events"]:
        events[event["file"]].append(event)
    if args.evaluate:
        model = scene_from_weights(checkpoint["model"]).eval()
        if not isinstance(model, RowTransportSceneDynamics):
            parser.error("evaluation requires a fitted row transport checkpoint")
        features, matches, positive, rewrite, counts = cache_examples(
            model, observed_captures(checkpoint["config"], model.codebook.numpy(), split), events)
        report = {"fit_used": False, "split": split, "test_used": False,
                  "checkpoint_sha256": sha(args.checkpoint), "evidence_sha256": sha(args.events),
                  "coverage": counts, "metrics": metrics(model.row_transport_head, features, matches, positive, rewrite)}
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        return
    model = row_transport_scene_weights(checkpoint["model"]).eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("row_transport_head."))
    head = model.row_transport_head
    features, matches, positive, rewrite, counts = cache_examples(
        model, training_captures(checkpoint["config"], model.codebook.numpy()), events)
    before = metrics(head, features, matches, positive, rewrite)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003)
    true = positive.nonzero().flatten()
    false = (~positive).nonzero().flatten()
    if not len(false):
        raise ValueError("no proven negative transitions")
    for _ in range(40):
        for pos in true[torch.randperm(len(true))].split(32):
            neg = false[torch.randint(len(false), (len(pos),))]
            logits, gate = head(features[torch.cat((pos, neg))])
            selected = logits[:len(pos)]
            exact = selected.masked_fill(~matches[pos], -torch.inf)
            pointer_loss = (selected.logsumexp(2) - exact.logsumexp(2)).mean()
            gate_loss = F.binary_cross_entropy_with_logits(
                gate, (torch.arange(len(gate), device=gate.device) < len(pos)).to(gate.dtype))
            optimizer.zero_grad(set_to_none=True)
            (pointer_loss + gate_loss).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
    state = model.state_dict()
    for key, tensor in checkpoint["model"].items():
        torch.testing.assert_close(tensor, state[key], rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False, "epochs": 40, "lr": .003,
              "source_sha256": sha(args.checkpoint), "evidence_sha256": sha(args.events),
              "coverage": counts, "before": before, "after": metrics(head, features, matches, positive, rewrite)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "row_transport_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
