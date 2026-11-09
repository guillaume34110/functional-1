"""Validation-only audit of observable last-rendered overlay context; no training."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import comparison_image
from run import read_episode
from token_model import decode_tokens, encode_frames, load_tile_model


class OverlayContext:
    """Cache only rendered observations and the learned overlay that produced them."""
    def __init__(self, model, layer=False):
        self.model = model
        self.layer = layer
        self.last_output = self.last_next = None
        self.last_background = None

    def decode(self, tokens, details):
        matched = torch.zeros(len(tokens), dtype=torch.bool, device=tokens.device)
        adjusted = dict(details)
        render_input = tokens
        restored_nonblank = torch.zeros_like(tokens, dtype=torch.bool)
        inferred, _ = self.model._hard_object(details["current"])
        reused = inferred
        if self.last_output is not None and self.last_output.shape == tokens.shape:
            matched = (tokens == self.last_output).flatten(1).all(1)
            last_mask, _ = self.model._hard_object(self.last_next)
            reused = torch.where(matched[:, None, None], last_mask, inferred)
            adjusted["current"] = {
                key: torch.where(matched.reshape(-1, *([1] * (value.ndim - 1))), self.last_next[key], value)
                for key, value in details["current"].items()}
            if self.layer:
                restore = matched[:, None, None] & last_mask & (details["mode"].argmax(1) != 1)[:, None, None]
                restored_nonblank = restore & (self.last_background != self.model.blank_id)
                render_input = torch.where(restore, self.last_background, tokens)
                adjusted["current"] = dict(details["current"])
                adjusted["current"]["visible"] = torch.where(matched[:, None],
                    torch.full_like(details["current"]["visible"], -torch.inf), details["current"]["visible"])
        original = self.model.decode_scene(tokens, details, hard=True)
        predicted = self.model.decode_scene(render_input, adjusted, hard=True)
        no_overlay = dict(adjusted)
        no_overlay["next"] = dict(details["next"])
        no_overlay["next"]["visible"] = torch.full_like(details["next"]["visible"], -torch.inf)
        self.last_background = self.model.decode_scene(render_input, no_overlay, hard=True).detach().clone()
        self.last_output = predicted.detach().clone()
        self.last_next = {key: value.detach().clone() for key, value in details["next"].items()}
        return predicted, {"matched": matched, "original": original,
                           "inferred_mask": inferred, "used_mask": reused,
                           "restored_nonblank": restored_nonblank}

    def step(self, tokens, actions, state=None):
        details = self.model.scene_details(tokens, actions, state)
        predicted, evidence = self.decode(tokens, details)
        return predicted, details["state"], evidence, details


def add_counts(stats, model, current, predicted, target, evidence, details):
    matched = evidence["matched"]
    inferred, used = evidence["inferred_mask"], evidence["used_mask"]
    board = model.board_mask[None].expand_as(current)
    occupied = current != model.blank_id
    original = evidence["original"]
    stats["steps"] += len(current)
    stats["cache_matches"] += int(matched.sum())
    stats["restored_nonblank_cells"] += int(evidence.get("restored_nonblank", torch.zeros_like(current)).sum())
    stats["mask_disagreement_steps"] += int(((inferred != used).flatten(1).any(1) & matched).sum())
    stats["mask_disagreement_cells"] += int(((inferred != used) & matched[:, None, None]).sum())
    stats["inferred_erase_outside_previous_overlay"] += int((inferred & ~used & occupied & matched[:, None, None]).sum())
    stats["original_output_erases_outside_previous_overlay"] += int(((original == model.blank_id) & occupied & ~used & board & matched[:, None, None]).sum())
    stats["cached_output_erases_outside_previous_overlay"] += int(((predicted == model.blank_id) & occupied & ~used & board & matched[:, None, None]).sum())
    stats["external_mismatch_steps"] += int((~matched).sum())
    stats["external_parity_failures"] += int(((predicted != original).flatten(1).any(1) & ~matched).sum())
    stats["output_changed_by_cache_steps"] += int((predicted != original).flatten(1).any(1).sum())
    stats["token_errors"] += int((predicted != target).sum())
    stats["same_input_original_token_errors"] += int((original != target).sum())
    stats["tokens"] += target.numel()
    stats["pixel_errors"] += int((model.codebook[predicted] != model.codebook[target]).sum())
    stats["pixels"] += target.numel() * 16
    modes = details["mode"].argmax(1)
    gate = (details["gate"][:, 0] - model.gate_logit_offset).sigmoid()
    rewrite = model.combine_logits(current, details["write"], details["gate"] - model.gate_logit_offset).argmax(1)
    stats["gate_positive_board_cells"] += int(((gate > .5) & board).sum())
    stats["raw_rewrite_changes_outside_overlay"] += int(((rewrite != current) & board & ~used).sum())
    for mode in range(3):
        stats[f"mode_{mode}_steps"] += int((modes == mode).sum())


def summarize(stats):
    result = dict(stats)
    for name, numerator, denominator in (("token_error", "token_errors", "tokens"),
                                         ("same_input_original_token_error", "same_input_original_token_errors", "tokens"),
                                         ("pixel_error", "pixel_errors", "pixels"),
                                         ("mask_disagreement_rate", "mask_disagreement_steps", "cache_matches")):
        result[name] = stats[numerator] / stats[denominator] if stats[denominator] else None
    return result


def palette(model, tokens):
    return decode_tokens(tokens, model.codebook)[0, 0].cpu().numpy()


@torch.inference_mode()
def audit(model, episodes, output, steps):
    result = {}
    for layer in (False, True):
        teacher = defaultdict(int)
        for frames, actions in episodes:
            context, state = OverlayContext(model, layer=layer), None
            for t in range(min(800, len(actions))):
                current = torch.from_numpy(frames[t].copy())[None].long()
                predicted, state, evidence, details = context.step(current, torch.tensor([int(actions[t])]), state)
                if t >= 32:
                    target = torch.from_numpy(frames[t + 1].copy())[None].long()
                    add_counts(teacher, model, current, predicted, target, evidence, details)
        result["teacher_layer" if layer else "teacher_white"] = summarize(teacher)
    originals = []
    for name in ("original", "cached_white", "cached_layer"):
        totals = {h: defaultdict(int) for h in (256, 1024) if h <= steps}
        for episode_index, (frames, actions) in enumerate(episodes):
            context, state = OverlayContext(model, layer=name == "cached_layer"), None
            for t in range(32):
                observed = torch.from_numpy(frames[t].copy())[None].long()
                _, state, _, _ = context.step(observed, torch.tensor([int(actions[t])]), state)
            current = torch.from_numpy(frames[32].copy())[None].long()
            images = [palette(model, current)]
            for i in range(min(steps, len(actions) - 32)):
                t = i + 32
                details = model.scene_details(current, torch.tensor([int(actions[t])]), state)
                state = details["state"]
                if name != "original":
                    prediction, evidence = context.decode(current, details)
                else:
                    prediction = model.decode_scene(current, details, hard=True)
                    mask, _ = model._hard_object(details["current"])
                    evidence = {"matched": torch.zeros(1, dtype=torch.bool), "original": prediction,
                                "inferred_mask": mask, "used_mask": mask}
                target = torch.from_numpy(frames[t + 1].copy())[None].long()
                for horizon, stats in totals.items():
                    if i < horizon:
                        add_counts(stats, model, current, prediction, target, evidence, details)
                current = prediction
                images.append(palette(model, prediction))
            reference = np.stack([palette(model, torch.from_numpy(frame.copy())[None].long())
                                  for frame in frames[32:32 + len(images)]])
            images = np.stack(images)
            if name == "original":
                originals.append(images)
            else:
                comparison_image(images, originals[episode_index], output / f"{name}-vs-original-{episode_index:02d}.png")
            comparison_image(images, reference, output / f"{name}-episode-{episode_index:02d}.png")
            if len(images) > 768:
                comparison_image(images[768:], reference[768:], output / f"{name}-tail-episode-{episode_index:02d}.png")
        result[name] = {str(h): summarize(stats) for h, stats in totals.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(256, 1024), default=1024)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    manifest_bytes = (args.data / "manifest.json").read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if checkpoint["config"]["manifest_sha256"] != digest:
        raise ValueError("checkpoint/dataset lineage mismatch")
    model = load_tile_model(args.checkpoint).eval()
    episodes = []
    for entry in json.loads(manifest_bytes)["episodes"]:
        if entry["split"] != "validation":
            continue
        path = (args.data / entry["file"]).resolve()
        if not path.is_relative_to(args.data.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("invalid validation episode")
        frames, actions, _ = read_episode(path)
        tokens, unknown = encode_frames(frames, model.codebook.numpy())
        if unknown or len(actions) <= 32:
            raise ValueError("unknown tiles or missing observed prefix")
        episodes.append((tokens, actions))
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "manifest_sha256": digest, "test_used": False, "episodes": len(episodes), "prefix": 32,
              "effective_steps": [min(args.steps, len(actions) - 32) for _, actions in episodes],
              "full_requested_coverage": all(len(actions) >= args.steps + 32 for _, actions in episodes),
              "context": "last rendered tokens, learned future joint/visibility and pre-overlay background; exact per-sample equality gate; all three variants share unchanged weights",
              "metrics": audit(model, episodes, args.output, args.steps)}
    if any(report["metrics"][name]["external_parity_failures"] for name in ("teacher_white", "teacher_layer")):
        raise RuntimeError("overlay context changed an externally supplied different screen")
    (args.output / "overlay-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    compact = {name: {h: values["token_error"] for h, values in metrics.items()}
               for name, metrics in report["metrics"].items() if not name.startswith("teacher")}
    print(json.dumps({"output": str(args.output / "overlay-audit.json"), "rollout_token_error": compact}))


if __name__ == "__main__":
    main()
