"""Calibrate the row transport gate on verified training events."""
import argparse
import json
from pathlib import Path

import torch

from calibrate_preview_gate import best_gate_threshold, gate_metrics
from scene_model import RowTransportSceneDynamics, scene_from_weights
from train_row_transport import cache_examples, metrics
from train_spawn_preview import sha, training_captures


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    evidence = json.loads(args.events.read_text())
    if (evidence["split"] != "train" or evidence["test_used"] or evidence["validation_used"]
            or sha(args.events) != checkpoint["row_transport_fit"]["evidence_sha256"]
            or any(evidence[key] != checkpoint["config"][key]
                   for key in ("manifest_sha256", "annotations_sha256"))):
        raise ValueError("calibration requires the original verified training events")
    events = {}
    for event in evidence["events"]:
        events.setdefault(event["file"], []).append(event)
    model = scene_from_weights(checkpoint["model"]).eval()
    if not isinstance(model, RowTransportSceneDynamics):
        parser.error("expected a row transport checkpoint")
    head = model.row_transport_head
    features, matches, positive, rewrite, counts = cache_examples(
        model, training_captures(checkpoint["config"], model.codebook.numpy()), events)
    rows, gate = head(features)
    correct = positive & matches.gather(2, rows.argmax(2, keepdim=True)).flatten(1).all(1)
    # A routed but wrong board is a false positive, including on true clear events.
    threshold = best_gate_threshold(gate.masked_fill(~rewrite, -1.), correct)
    before = metrics(head, features, matches, positive, rewrite)
    ideal = rewrite & (gate > threshold)
    head.gate.bias.sub_(threshold)
    calibrated_rows, calibrated_gate = head(features)
    torch.testing.assert_close(calibrated_rows, rows, rtol=0, atol=0)
    state = model.state_dict()
    for name, old in checkpoint["model"].items():
        if name != "row_transport_head.gate.bias":
            torch.testing.assert_close(state[name], old, rtol=0, atol=0)
    triggered = rewrite & (calibrated_gate > 0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "evidence_sha256": sha(args.events),
              "threshold": threshold, "coverage": counts,
              "gate_before": gate_metrics(rewrite & (gate > 0), correct),
              "gate_after": gate_metrics(triggered, correct),
              "rounding_decision_differences": int((triggered != ideal).sum()),
              "before": before, "after": metrics(head, features, matches, positive, rewrite)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({**checkpoint, "model": state, "row_gate_calibration": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
