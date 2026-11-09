"""Fit command priors from verified visual rotations and completed translations."""
import argparse
import json
from pathlib import Path

import torch

from intervention_training import load_interventions
from model import ACTIONS
from scene_model import CounterSceneDynamics, scene_from_weights
from train_event_replay import event_targets
from train_rotation_proposal import rotation_examples
from train_spawn_preview import sha, training_captures


def training_interventions(model, checkpoint, directory):
    parents = list(training_captures(checkpoint["config"], model.codebook.numpy()))
    manifest_path = Path("data/manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    names = [entry["file"] for entry in manifest["episodes"] if entry["split"] == "train"]
    if names != [name for name, _, _, _ in parents]:
        raise ValueError("training parent order differs from the manifest")
    episodes = {"train": [(frame.numpy(), action.numpy()) for _, frame, action, _ in parents]}
    annotations = {"train": [{key: value.numpy() for key, value in label.items()} for _, _, _, label in parents]}
    tables, digest = load_interventions(directory, model, manifest, episodes, annotations,
                                        sha(manifest_path), checkpoint["config"]["annotations_sha256"], splits=("train",))
    return parents, tables["train"], digest


def rotation_votes(model, bundles):
    count = len(model.object_shapes)
    votes = torch.zeros((count, count), dtype=torch.long)
    for bundle in bundles.values():
        records, first = rotation_examples(model, bundle)
        effect = first[0]
        if effect is None:
            continue
        pose = records[1][4]
        old, new = pose[effect:effect+2] // 180
        if old != new and model.object_tiles[old] == model.object_tiles[new]:
            votes[old, new] += 1
    support, target = votes.max(1)
    seen = votes.sum(1)
    odds = ((support + .5) / (seen - support + .5)).log().clamp_min(0)
    target = target.masked_fill(seen == 0, -1)
    if not (odds > 0).any():
        raise ValueError("no witnessed preference for an A orientation")
    return target, odds, votes


def switch_counts(model, parents):
    counts = torch.zeros((2, 2), dtype=torch.long)
    for _, frames, actions, labels in parents:
        old, mode, new, _ = event_targets(model, frames, labels).unbind(1)
        known = (old >= 0) & (new >= 0) & (mode == 0) & (old // 180 == new // 180)
        pair = known[:-1] & known[1:] & (new[:-1] == old[1:]) & (actions[1:] == ACTIONS.index("down"))
        dx = new % 10 - old % 10
        dy = new % 180 // 10 - old % 180 // 10
        pair &= (dy[:-1] >= 0) & (dy[:-1] <= 1)
        for row, (button, direction) in enumerate((("left", -1), ("right", 1))):
            visible = pair & (actions[:-1] == ACTIONS.index(button)) & (dx[:-1] == direction)
            counts[row] += torch.bincount((dx[1:][visible] != 0).long(), minlength=2)
    if not counts.sum(1).all():
        raise ValueError("both directions need witnessed completed translations")
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
    if (not isinstance(base, CounterSceneDynamics) or not base.counter.pulsed
            or base.scene_continuity is None or base.scene_continuity.shape != (2,)):
        raise ValueError("expected a fitted continuity checkpoint")
    parents, bundles, digest = training_interventions(base, checkpoint, args.interventions)
    target, odds, votes = rotation_votes(base, bundles)
    counts = switch_counts(base, parents)
    straight, lateral = counts.sum(0)
    scale = ((lateral + .5) / (straight + .5)).log()
    state = {**checkpoint["model"], "scene_continuity": torch.cat((base.scene_continuity.detach(), scale[None])),
             "a_rotation_target": target, "a_rotation_odds": odds, "_a_proposal_gate": torch.tensor(True)}
    candidate = scene_from_weights(state).eval()
    final = candidate.state_dict()
    for name, tensor in checkpoint["model"].items():
        torch.testing.assert_close(tensor, final[name][:2] if name == "scene_continuity" else final[name],
                                   rtol=0, atol=0)
    report = {"source_sha256": sha(args.checkpoint), "interventions_sha256": digest, "fit_split": "train",
              "validation_used": False, "test_used": False, "rotation_readout": "local-proposal-gate",
              "a": {"witnesses": int(votes.sum()), "target": target.tolist(), "log_odds": odds.tolist(),
                    "votes": votes.tolist()},
              "lateral_to_down": {"direction_counts": counts.tolist(), "coefficient": float(scale)}}
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "model.pt"
    torch.save({"model": final, "config": checkpoint["config"], "command_transition_fit": report}, path)
    report["checkpoint_sha256"] = sha(path)
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
