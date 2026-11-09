"""Audit learned event detection versus mixture MAP decoding on validation only."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate import comparison_image
from token_model import EventTileDynamics, decode_tokens, encode_frames, load_tile_model
from train import load_data


def select_tokens(tokens, write, change, mode):
    if mode == "factor":
        return torch.where(change[:, 0] > 0, write.argmax(1), tokens)
    if mode == "map":
        return EventTileDynamics.combine_logits(tokens, write, change).argmax(1)
    raise ValueError("unknown decoding mode")


def fresh_counts():
    return {name: 0 for name in (
        "tiles", "changed", "unchanged", "gate_tp", "gate_fp", "gate_fn", "gate_tn",
        "additions", "erasures", "write_additions_correct", "write_erasures_correct",
        "map_wrong", "factor_wrong", "map_changed_wrong", "factor_changed_wrong",
        "map_additions_correct", "factor_additions_correct", "map_erasures_correct", "factor_erasures_correct",
        "positive_gate_but_map_copies_addition")}


def update_counts(counts, current, target, write, change, blank):
    changed = target != current
    additions, erasures = changed & ~blank[target], changed & blank[target]
    gate = change[:, 0] > 0
    writing = write.argmax(1)
    predictions = {mode: select_tokens(current, write, change, mode) for mode in ("map", "factor")}
    masks = {"changed": changed, "unchanged": ~changed, "gate_tp": gate & changed,
             "gate_fp": gate & ~changed, "gate_fn": ~gate & changed, "gate_tn": ~gate & ~changed,
             "additions": additions, "erasures": erasures,
             "write_additions_correct": (writing == target) & additions,
             "write_erasures_correct": (writing == target) & erasures,
             "positive_gate_but_map_copies_addition": gate & additions & (predictions["map"] == current)}
    counts["tiles"] += target.numel()
    for mode, predicted in predictions.items():
        wrong = predicted != target
        masks.update({f"{mode}_wrong": wrong, f"{mode}_changed_wrong": wrong & changed,
                      f"{mode}_additions_correct": ~wrong & additions,
                      f"{mode}_erasures_correct": ~wrong & erasures})
    for key, mask in masks.items():
        counts[key] += int(mask.sum())


def summarize(counts):
    def ratio(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else None

    return {"counts": counts,
            "gate": {"recall": ratio("gate_tp", "changed"), "false_positive_rate": ratio("gate_fp", "unchanged"),
                     "precision": counts["gate_tp"] / max(1, counts["gate_tp"] + counts["gate_fp"])},
            "write_given_event": {"additions_accuracy": ratio("write_additions_correct", "additions"),
                                  "erasures_accuracy": ratio("write_erasures_correct", "erasures")},
            **{mode: {"tile_error": ratio(f"{mode}_wrong", "tiles"),
                      "changed_tile_error": ratio(f"{mode}_changed_wrong", "changed"),
                      "additions_accuracy": ratio(f"{mode}_additions_correct", "additions"),
                      "erasures_accuracy": ratio(f"{mode}_erasures_correct", "erasures")}
               for mode in ("map", "factor")}}


def teacher_audit(model, episodes, max_steps=800):
    total, after_32 = fresh_counts(), fresh_counts()
    effective = []
    blank = (model.codebook.flatten(1) == 3).all(1)
    with torch.inference_mode():
        for frames, actions in episodes:
            state = None
            length = min(max_steps, len(actions))
            effective.append(length)
            for t in range(length):
                current = torch.from_numpy(frames[t].copy())[None].long()
                target = torch.from_numpy(frames[t + 1].copy())[None].long()
                write, change, state = model.forward_details(current, torch.tensor([int(actions[t])]), state)
                update_counts(total, current, target, write, change, blank)
                if t >= 32:
                    update_counts(after_32, current, target, write, change, blank)
    return {"effective_steps": effective, "state": "continuous true prefix, reset only at capture start",
            "all_steps": summarize(total), "after_first_32_steps": summarize(after_32)}


def rollout_audit(model, encoded, raw, output, horizon=256):
    episodes = []
    with torch.inference_mode():
        for index, ((tokens, actions), (pixels, _)) in enumerate(zip(encoded, raw)):
            start = min(32, len(actions) - 1)
            state = None
            for t in range(start):
                _, _, state = model.forward_details(torch.from_numpy(tokens[t].copy())[None].long(),
                                                   torch.tensor([int(actions[t])]), state)
            length = min(horizon, len(actions) - start)
            record = {"episode": index, "prefix_steps": start, "effective_steps": length}
            for mode in ("map", "factor"):
                current = torch.from_numpy(tokens[start].copy())[None].long()
                local_state = state.clone() if state is not None else None
                predictions = [pixels[start].copy()]
                wrong = changed_wrong = changed_count = frozen_wrong = active_frames = 0
                for offset in range(length):
                    t = start + offset
                    write, change, local_state = model.forward_details(current, torch.tensor([int(actions[t])]), local_state)
                    next_tokens = select_tokens(current, write, change, mode)
                    active_frames += int(torch.any(next_tokens != current))
                    current = next_tokens
                    prediction = current[0].numpy()
                    error = prediction != tokens[t + 1]
                    changed = tokens[t + 1] != tokens[t]
                    wrong += int(error.sum())
                    changed_wrong += int((error & changed).sum())
                    changed_count += int(changed.sum())
                    frozen_wrong += int((tokens[start] != tokens[t + 1]).sum())
                    predictions.append(decode_tokens(current, model.codebook)[0, 0].numpy().copy())
                total_tiles = length * 18 * 20
                record[mode] = {"tile_error": wrong / max(1, total_tiles),
                                "changed_tile_error": changed_wrong / max(1, changed_count),
                                "changed_tiles": changed_count, "tiles": total_tiles,
                                "wrong": wrong, "changed_wrong": changed_wrong,
                                "frozen_wrong": frozen_wrong, "changed_prediction_frames": active_frames}
                comparison_image(np.stack(predictions), pixels[start:start + length + 1],
                                 output / f"{mode}-comparison-{index:02d}.png")
            episodes.append(record)
    aggregate = {}
    for mode in ("map", "factor"):
        tiles = sum(episode[mode]["tiles"] for episode in episodes)
        changed = sum(episode[mode]["changed_tiles"] for episode in episodes)
        aggregate[mode] = {"tile_error": sum(e[mode]["wrong"] for e in episodes) / max(1, tiles),
                           "changed_tile_error": sum(e[mode]["changed_wrong"] for e in episodes) / max(1, changed),
                           "frozen_screen_tile_error": sum(e[mode]["frozen_wrong"] for e in episodes) / max(1, tiles)}
    return {"horizon": horizon, "episodes": episodes, "aggregate": aggregate,
            "full_horizon_coverage": bool(episodes) and min(e["effective_steps"] for e in episodes) >= horizon}


def self_check():
    current = torch.zeros((1, 18, 20), dtype=torch.int64)
    write = torch.tensor([0.05, 0.45, 0.50]).log()[None, :, None, None].expand(1, 3, 18, 20)
    change = torch.full((1, 1, 18, 20), math.log(0.6 / 0.4))
    assert torch.all(select_tokens(current, write, change, "map") == 0)
    assert torch.all(select_tokens(current, write, change, "factor") == 2)
    frames = np.zeros((3, 18, 20), dtype=np.int64)
    frames[1, 2, 3] = 1
    actions = np.array([1, 4], dtype=np.uint8)

    class ExactEvents:
        codebook = torch.stack((torch.full((4, 4), 3, dtype=torch.uint8), torch.zeros((4, 4), dtype=torch.uint8)))

        def forward_details(self, tokens, action, state):
            t = 0 if state is None else state
            assert torch.equal(tokens[0], torch.from_numpy(frames[t]))
            assert int(action[0]) == int(actions[t])
            target = torch.from_numpy(frames[t + 1])[None]
            writes = torch.full((1, 2, 18, 20), -20.0).scatter_(1, target[:, None], 20.0)
            changes = torch.where(target[:, None] != tokens[:, None], 20.0, -20.0)
            return writes, changes, t + 1

    result = teacher_audit(ExactEvents(), [(frames, actions)], max_steps=2)["all_steps"]
    assert result["counts"]["additions"] == result["counts"]["erasures"] == 1
    assert result["map"]["tile_error"] == result["factor"]["tile_error"] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rollout", action="store_true")
    args = parser.parse_args()
    if min(args.max_steps, args.episodes, args.threads) < 1:
        parser.error("positive diagnostic bounds required")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    self_check()
    model = load_tile_model(args.weights)
    if not isinstance(model, EventTileDynamics):
        raise ValueError("audit requires an event-head checkpoint")
    splits, digest, _ = load_data(args.data)
    raw = splits["validation"][:args.episodes]
    encoded, unknown = [], 0
    for frames, actions in raw:
        tokens, missing = encode_frames(frames, model.codebook.numpy())
        encoded.append((tokens, actions))
        unknown += missing
    args.output.mkdir(parents=True)
    result = {"checkpoint_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(), "manifest_sha256": digest,
              "split": "validation", "test_used": False, "unknown_tiles": unknown,
              "threshold": 0.5, "teacher_forcing": teacher_audit(model, encoded, args.max_steps),
              "self_check": "passed", "rules_verified": False}
    if args.rollout:
        result["rollout"] = rollout_audit(model, encoded, raw, args.output)
    (args.output / "decode-audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(args.output / "decode-audit.json"), "self_check": result["self_check"],
                      "teacher": result["teacher_forcing"]["after_first_32_steps"],
                      "rollout": result.get("rollout", {}).get("aggregate")}), flush=True)


if __name__ == "__main__":
    main()
