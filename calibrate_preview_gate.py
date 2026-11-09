"""Calibrate only the V42 preview gate bias using causal parent train examples."""
import argparse
from itertools import groupby
import json
from pathlib import Path

import torch

from scene_model import FactorizedPreviewRenderSceneDynamics, scene_from_weights
from train_preview_render import factorized_metrics, training_examples
from train_spawn_preview import sha, training_captures


def best_gate_threshold(scores, changes):
    """Optimize F1 exactly over strict score > threshold decisions, with threshold >= 0."""
    if scores.ndim != 1 or scores.shape != changes.shape or not torch.isfinite(scores).all():
        raise ValueError("calibration requires aligned finite scalar scores")
    actual = int(changes.sum())
    if not 0 < actual < len(changes):
        raise ValueError("calibration requires both copy and change targets")
    positive = scores > 0
    ranking = sorted(zip(scores[positive].tolist(), changes[positive].tolist()))
    predicted, hits = len(ranking), int((positive & changes).sum())
    threshold, best_hits, best_predicted = 0., hits, predicted
    for score, ties in groupby(ranking, key=lambda pair: pair[0]):
        for _, changed in ties:
            predicted -= 1
            hits -= int(changed)
        # Compare integer fractions. Strict improvement retains the lowest tied threshold.
        if hits * (actual + best_predicted) > best_hits * (actual + predicted):
            threshold, best_hits, best_predicted = score, hits, predicted
    return threshold


def gate_metrics(predicted, changes):
    actual = int(changes.sum())
    triggered = int(predicted.sum())
    hits = int((predicted & changes).sum())
    return {"true_positives": hits, "false_positives": triggered - hits, "false_negatives": actual - hits,
            "true_negatives": len(changes) - (triggered + actual - hits),
            "actual_changes": actual, "predicted_changes": triggered,
            "precision": hits / triggered if triggered else None, "recall": hits / actual,
            "f1": 2 * hits / (actual + triggered)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory for model.pt and result.json")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = scene_from_weights(checkpoint).eval()
    if type(model) is not FactorizedPreviewRenderSceneDynamics or len(model.codebook) > 256:
        raise ValueError("calibration requires a factorized V42 preview checkpoint")
    config = checkpoint["config"]
    captures = ((name, tokens.byte(), actions) for name, tokens, actions, _ in
                training_captures(config, model.codebook.numpy()))
    features, labels, previous, counts, _ = training_examples(model, captures)
    head = model.preview_render_head
    before = head(features)
    changes = labels != 0
    threshold = best_gate_threshold(before[:, 0], changes)
    ideal_decisions = before[:, 0] > threshold
    head[-1].bias[0].sub_(threshold)
    after = head(features)
    torch.testing.assert_close(after[:, 1:], before[:, 1:], rtol=0, atol=0)
    expected = before[:, 0] - threshold
    torch.testing.assert_close(after[:, 0], expected, rtol=1e-6, atol=1e-5)
    weights = model.state_dict()
    for name, original in checkpoint["model"].items():
        exported = weights[name]
        if name == "preview_render_head.2.bias":
            original, exported = original[1:], exported[1:]
        torch.testing.assert_close(exported.reshape(-1).view(torch.uint8),
                                   original.reshape(-1).view(torch.uint8), rtol=0, atol=0)
    actual_decisions = after[:, 0] > 0
    report = {"source_checkpoint_sha256": sha(args.checkpoint), "source_step": checkpoint["step"],
              "manifest_sha256": config["manifest_sha256"], "annotations_sha256": config["annotations_sha256"],
              "script_sha256": sha(Path(__file__)), "validation_used": False, "test_used": False,
              "examples": len(labels), "examples_by_capture": counts, "threshold": threshold,
              "tie_break": "lowest threshold among equal F1 values", "comparison": "score > threshold",
              "before": gate_metrics(before[:, 0] > 0, changes),
              "ideal_threshold": gate_metrics(ideal_decisions, changes),
              "after": gate_metrics(actual_decisions, changes),
              "bias_rounding_decision_differences": int((ideal_decisions != actual_decisions).sum()),
              "conditional_exact_motif_rate": factorized_metrics(head, features, labels, previous)["conditional_exact_motif_rate"],
              "max_gate_logit_error": float((after[:, 0] - expected).abs().max()),
              "changed_parameter": "preview_render_head.2.bias[0]", "optimizer_exported": False}
    args.output.mkdir(parents=True)
    output = args.output / "model.pt"
    torch.save({"model": weights, "step": checkpoint["step"], "config": {
        **config, "preview_gate_calibration": report
    }}, output)
    restored = scene_from_weights(torch.load(output, map_location="cpu", weights_only=True)).eval()
    torch.testing.assert_close(restored.preview_render_head(features), after, rtol=0, atol=0)
    report["checkpoint_sha256"] = sha(output)
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
