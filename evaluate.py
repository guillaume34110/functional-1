"""Held-out free-running comparison: no ROM or truth feedback after warm-up."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from model import NeuralTetris
from train import load_data, rollout_metrics


def comparison_image(predictions, reference, output):
    times = sorted(set(min(t, len(predictions) - 1) for t in (0, 1, 16, 64, 256)))
    sheet = Image.new("RGB", (len(times) * 160, 316), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    for column, step in enumerate(times):
        draw.text((column * 160 + 4, 0), f"t={step}: reference / neural", fill="black")
        for row, frames in enumerate((reference, predictions)):
            image = Image.fromarray((frames[step] * 85).astype(np.uint8)).convert("RGB")
            sheet.paste(image.resize((160, 144), Image.Resampling.NEAREST), (column * 160, 20 + row * 148))
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    splits, digest, manifest = load_data(args.data)
    model = NeuralTetris().eval()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    metrics, examples = rollout_metrics(model, splits["test"], maximum=len(splits["test"]))
    args.output.mkdir(parents=True)
    for index, (predicted, reference) in enumerate(examples):
        comparison_image(predicted, reference, args.output / f"comparison-{index:02d}.png")
        np.savez_compressed(args.output / f"rollout-{index:02d}.npz", predicted=predicted, reference=reference)
    frame = torch.from_numpy(splits["test"][0][0][0].copy())[None, None]
    latencies = []
    state = None
    model.eval()
    with torch.inference_mode():
        for i in range(120):
            started = time.perf_counter()
            logits, state = model(frame, torch.tensor([i % 6]), state)
            frame = logits.argmax(1, keepdim=True).to(torch.uint8)
            if i >= 20:
                latencies.append((time.perf_counter() - started) * 1000)
    # These are necessary numerical gates, never proof of coherent game rules.
    gates = {"pixel_error_256_below_0.005": metrics["256"]["pixel_error"] < 0.005,
             "changed_error_64_below_0.05": metrics["64"]["changed_pixel_error"] < 0.05,
             "beats_frozen_screen": metrics["256"]["pixel_error"] < metrics["256"]["frozen_screen_error"],
             "latency_p95_below_66ms": float(np.percentile(latencies, 95)) < 66}
    result = {"status": "numerical-gates-passed-rules-unverified" if all(gates.values()) else "not-playable-validated",
              "model_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(), "manifest_sha256": digest,
              "test_episodes": len(splits["test"]), "horizons": metrics, "gates": gates,
              "latency_ms": {"median": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95))},
              "dataset_coverage": manifest.get("coverage"), "gameplay_rules_verified": False}
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2))
    # Preserve a real prefix for the interactive model; no future screens are used.
    original, actions = splits["test"][0]
    np.savez_compressed(args.output / "play-seed.npz", frames=original[:33], actions=actions[:32],
                        metadata=json.dumps({"context": manifest["context"], "source": "capture", "prefix_only": True}))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
