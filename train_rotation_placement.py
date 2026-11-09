"""Fit the horizontal placement of the learned A proposal on visible TRAIN turns."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scene_model import CounterSceneDynamics, scene_from_weights
from train_command_transitions import training_interventions
from train_rotation_proposal import rotation_examples
from train_spawn_preview import sha


@torch.no_grad()
def column_votes(model, bundles):
    counts = torch.zeros((len(model.object_shapes), 7), dtype=torch.long)
    for bundle in bundles.values():
        records, first = rotation_examples(model, bundle)
        if first[0] != 0:
            continue
        frames, _, _, _, poses, _ = records[1]
        old, new = poses[:2]
        kind = old // 180
        if new // 180 != model.a_rotation_target[kind]:
            continue
        mask = model.shape_masks.flatten(0, 1)[old:old+1].bool()
        neighbors = model.contact_features(frames[:1], F.pad(mask, (2, 8)))
        if not model.rotation_placement_visible(old[None], neighbors)[0]:
            continue
        dx = new % 10 - old % 10
        if -3 <= dx <= 3:
            counts[kind, dx+3] += 1
    if not counts.any():
        raise ValueError("no witnessed open A placement")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interventions", type=Path, default=Path("data/action-interventions-v3"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    if not isinstance(base, CounterSceneDynamics) or base._a_proposal_gate is None or base.a_column_offset is not None:
        raise ValueError("expected a fitted local A proposal checkpoint")
    _, bundles, digest = training_interventions(base, checkpoint, args.interventions)
    if digest != checkpoint["command_transition_fit"]["interventions_sha256"]:
        raise ValueError("placement and orientation training inputs differ")
    votes = column_votes(base, bundles)
    support, target = votes.max(1)
    seen = votes.sum(1)
    odds = ((support + .5) / (seen - support + .5)).log().clamp_min(0)
    offset = (target - 3).masked_fill(seen == 0, 0)
    state = {**checkpoint["model"], "a_column_offset": offset, "a_column_odds": odds}
    model = scene_from_weights(state).eval()
    state = model.state_dict()
    for name, value in checkpoint["model"].items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)
    report = {"source_sha256": sha(args.checkpoint), "interventions_sha256": digest, "fit_split": "train",
              "validation_used": False, "test_used": False, "witnesses": int(votes.sum()),
              "offset": offset.tolist(), "odds": odds.tolist(), "votes": votes.tolist()}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({**checkpoint, "model": state, "rotation_placement_fit": report}, destination)
    report["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
