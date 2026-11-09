"""Fit local fall timing from proven held-down ROM transitions."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from audit_controls import BUTTONS, supported
from scene_model import timed_event_scene_weights
from train_mode_correction import proven_composition
from train_spawn_preview import sha, training_captures


def unchanged_ages(sequence):
    ticks = torch.arange(len(sequence)).reshape((-1,) + (1,) * (sequence.ndim - 1))
    changes = torch.zeros_like(sequence, dtype=torch.bool)
    changes[1:] = sequence[1:] != sequence[:-1]
    last = torch.where(changes, ticks, 0).cummax(0).values
    return (ticks - last).clamp(max=255)


def collect(model, captures):
    samples = []
    down = BUTTONS["down"]
    for _, frames, actions, labels in captures:
        modes, known, old, masks, new = proven_composition(model, frames, labels)
        y, x = labels["cells"].amin(1).unbind(1)
        dy = y.diff()
        held = (unchanged_ages(actions) + 1).clamp(max=255)
        same_shape = labels["shape_id"][1:] == labels["shape_id"][:-1]
        candidate = (known & old & new & (actions == down) & (held > 1) & (modes == 0)
                     & same_shape & (x[1:] == x[:-1]) & ((dy == 0) | (dy == 1)))
        indices = candidate.nonzero().flatten()
        if not len(indices):
            continue
        backgrounds = frames[indices].masked_fill(masks[indices], model.blank_id)[:, :, 2:12].numpy()
        cells = labels["cells"][indices].numpy()
        floating = torch.tensor([not supported(board, piece, model.blank_id) for board, piece in zip(backgrounds, cells)])
        indices = indices[floating]
        ages = unchanged_ages(frames)[indices].masked_fill(~masks[indices], 255).amin((1, 2))
        samples.append((y[indices], ages, held[indices], dy[indices].bool()))
    if not samples:
        raise ValueError("no proven floating piece transitions")
    return tuple(torch.cat(column) for column in zip(*samples))


@torch.no_grad()
def metrics(model, row, age, held, move):
    logits = model.drop_log_odds(row, age, held)
    report = {}
    for name, region in (("spawn_row", row == 0), ("second_row", row == 1), ("lower_rows", row > 1)):
        report[name] = {"examples": int(region.sum()), "move": int((region & move).sum()),
                        "correct_wait": int((region & ~move & (logits <= 0)).sum()),
                        "correct_move": int((region & move & (logits > 0)).sum()),
                        "maximum_observed_wait_age": int(age[region & ~move].max()) if (region & ~move).any() else None}
    ticks = torch.arange(1, 256)
    for y in (0, 1):
        for duration in (16, 128):
            ready = ticks[model.drop_log_odds(torch.full_like(ticks, y), ticks, torch.full_like(ticks, duration)) > 0]
            report[f"first_delayed_drop_age_row_{y}_held_{duration}"] = int(ready[0]) if len(ready) else None
    report["nll"] = float(F.binary_cross_entropy_with_logits(logits, move.float()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(57)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = timed_event_scene_weights(checkpoint["model"]).eval()
    row, age, held, move = collect(model, training_captures(checkpoint["config"], model.codebook.numpy()))
    groups = row.clamp_max(2) * 2 + move.long()
    counts = torch.bincount(groups, minlength=6)
    if not counts.all():
        raise ValueError("training needs proven waits and moves in each row group")
    weights = len(move) / (6 * counts[groups])
    before = metrics(model, row, age, held, move)
    optimizer = torch.optim.LBFGS(model.drop_clock.parameters(), max_iter=120, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = model.drop_log_odds(row, age, held)
        loss = F.binary_cross_entropy_with_logits(logits, move.float(), weight=weights)
        loss = loss + 1e-5 * sum(parameter.square().sum() for parameter in model.drop_clock.parameters())
        loss.backward()
        return loss

    optimizer.step(closure)
    state = model.state_dict()
    for key, tensor in checkpoint["model"].items():
        torch.testing.assert_close(state[key], tensor, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "before": before, "after": metrics(model, row, age, held, move)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "event_timing_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
