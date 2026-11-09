"""Fit the V34 overlap multiplier on proven first-step training targets."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from intervention_training import tensors
from scene_model import LearnedOverlapSceneDynamics, scene_from_weights
from train_scene import label_mask, load_action_interventions, load_scene_data


def grid_nll(base, offset, wanted, scales):
    scores = base[:, None, :] + scales[None, :, None] * offset[:, None, :]
    index = wanted[:, None, None].expand(-1, len(scales), 1)
    return scores.logsumexp(2) - scores.gather(2, index)[:, :, 0]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = scene_from_weights(checkpoint["model"]).eval()
    if not isinstance(model, LearnedOverlapSceneDynamics):
        parser.error("expected a learned-overlap checkpoint")
    _, episodes, annotations, _, _, manifest_sha, annotation_sha = load_scene_data(checkpoint)
    tables, intervention_sha = load_action_interventions(
        checkpoint, model, episodes, annotations, manifest_sha, annotation_sha)
    table = tables["train"]
    scales = torch.linspace(0., 4., 17)
    nll_sum = torch.zeros_like(scales, dtype=torch.float64)
    rows = covered = 0
    for episode, (tokens, commands) in enumerate(episodes["train"]):
        anchors = sorted(t for i, t in table if i == episode)
        if not anchors:
            continue
        state = None
        for t in range(anchors[-1] + 1):
            if (episode, t) in table:
                frames, actions, labels, states = tensors(model, table[episode, t], state)
                valid = torch.ones(len(actions), dtype=torch.bool)
                old_known, _ = label_mask(labels, 0, valid)
                new_known, _ = label_mask(labels, 1, valid)
                eligible = old_known & new_known
                if eligible.any():
                    details = model.scene_details(frames[:, 0], actions[:, 0], states)
                    raw, _ = model.next_head(details["state"][:, :16, :, 2:12], model.valid_poses)
                    base = raw["joint"][eligible].flatten(1)
                    logits = details["overlap_logits"][eligible].flatten(1)
                    copy = details["mode"][eligible].softmax(1)[:, :1]
                    offset = copy * (F.softplus(logits.new_zeros(())) - F.softplus(logits))
                    y, x = labels["cells"][eligible, 1].amin(1).unbind(1)
                    wanted = labels["shape_id"][eligible, 1] * 180 + y * 10 + x
                    loss = grid_nll(base, offset, wanted, scales)
                    if not torch.isfinite(loss).all():
                        raise ValueError("nonfinite calibration loss")
                    nll_sum += loss.sum(0, dtype=torch.float64)
                    rows += len(loss)
                covered += 1
            if t < anchors[-1]:
                frame = torch.from_numpy(tokens[t:t + 1].copy()).long()
                state = model.advance(frame, torch.tensor([int(commands[t])]), state)
    if rows == 0 or covered != len(table):
        raise ValueError("incomplete calibration coverage")
    losses = nll_sum / rows
    index = min(range(len(scales)), key=lambda i: (float(losses[i]), abs(float(scales[i]) - 1)))
    model.overlap_scale.fill_(scales[index])
    report = {"fit_split": "train", "validation_used_for_fit": False, "test_used": False,
              "base_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "manifest_sha256": manifest_sha, "annotations_sha256": annotation_sha,
              "interventions_sha256": intervention_sha, "anchors": covered, "proven_targets": rows,
              "scale": float(scales[index]), "grid": list(zip(scales.tolist(), losses.tolist()))}
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint["model"] = model.state_dict()
    checkpoint["overlap_calibration"] = report
    destination = args.output / "checkpoint.pt"
    torch.save(checkpoint, destination)
    report["checkpoint_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
