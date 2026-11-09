"""Inspect one-step visual transitions without using the reserved test."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from train import load_data
from token_model import encode_frames, load_tile_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/tiles-v2.pt"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_tile_model(args.weights)
    raw, _, _ = load_data("data")
    results = []
    with torch.inference_mode():
        for frames, actions in raw["validation"]:
            tokens, missing = encode_frames(frames[:801], model.codebook.numpy())
            state = None
            counts = {k: 0 for k in ("changed", "changed_correct", "board_changed", "board_changed_correct",
                                    "additions", "additions_correct", "removals", "removals_correct", "action_sensitive", "steps")}
            for t in range(len(tokens) - 1):
                current = torch.from_numpy(tokens[t].copy())[None]
                action = torch.tensor([int(actions[t])])
                if t >= 32:
                    ablated, _ = model(current, torch.zeros_like(action), state)
                logits, state = model(current, action, state)
                if t < 32:
                    continue
                prediction = logits.argmax(1)[0].numpy()
                truth = tokens[t + 1]
                changed = truth != tokens[t]
                correct = prediction == truth
                counts["changed"] += int(changed.sum()); counts["changed_correct"] += int((correct & changed).sum())
                counts["board_changed"] += int(changed[:, 2:12].sum())
                counts["board_changed_correct"] += int((correct & changed)[:, 2:12].sum())
                before = model.codebook[tokens[t]].numpy()[:, 2:12].min(axis=(2, 3)) < 3
                after = model.codebook[truth].numpy()[:, 2:12].min(axis=(2, 3)) < 3
                additions, removals = ~before & after, before & ~after
                counts["additions"] += int(additions.sum()); counts["additions_correct"] += int((correct[:, 2:12] & additions).sum())
                counts["removals"] += int(removals.sum()); counts["removals_correct"] += int((correct[:, 2:12] & removals).sum())
                counts["action_sensitive"] += int(np.any(prediction != ablated.argmax(1)[0].numpy()))
                counts["steps"] += 1
            counts["unknown"] = missing
            results.append(counts)
    totals = {k: sum(x[k] for x in results) for k in results[0]}
    result = {"copy_strength": float(model.copy_strength.detach()) if hasattr(model, "copy_strength") else None, "teacher_forced": totals,
              "teacher_changed_accuracy": totals["changed_correct"] / max(1, totals["changed"]),
              "board_changed_accuracy": totals["board_changed_correct"] / max(1, totals["board_changed"]),
              "addition_accuracy": totals["additions_correct"] / max(1, totals["additions"]),
              "removal_accuracy": totals["removals_correct"] / max(1, totals["removals"])}
    Path("diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
