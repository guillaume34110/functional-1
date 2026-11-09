"""Fit the categorical preview renderer on parent train captures only."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scene_model import (FactorizedPreviewRenderSceneDynamics, ObjectSceneDynamics, SpawnPreviewSceneDynamics,
                         preview_render_scene_weights, scene_from_weights)
from train_spawn_preview import sha, training_captures


SEED, EPOCHS, BATCH, LR = 41, 30, 256, 0.003


def patterns_from_train(captures, object_tiles):
    counts = Counter()
    for _, tokens, _ in captures:
        crops, occurrences = torch.unique(tokens[:, :6, 13:19].flatten(1), dim=0, return_counts=True)
        textures = (crops[..., None] == object_tiles).sum(1)
        valid = (textures.sum(1) == 4) & (textures.amax(1) == 4)
        counts.update({tuple(crop): count for crop, count in zip(crops[valid].tolist(), occurrences[valid].tolist())})
    frequent = [(pattern, count) for pattern, count in counts.items() if count >= 2]
    frequent.sort(key=lambda pair: (-pair[1], pair[0]))
    if not frequent:
        raise ValueError("no repeated four-tile preview motif in the training captures")
    if len(frequent) > 64:
        raise ValueError("more than 64 repeated preview motifs; inspect crop alignment before fitting")
    return torch.tensor([pattern for pattern, _ in frequent]).reshape(-1, 6, 6), [count for _, count in frequent]


@torch.no_grad()
def training_examples(model, captures):
    features, labels, previous, counts = [], [], [], {}
    factorized = isinstance(model, FactorizedPreviewRenderSceneDynamics)
    witness = None
    for name, tokens, actions in captures:
        crops = tokens[:, :6, 13:19]
        matches = (crops[:, None] == model.preview_patterns).flatten(2).all(2)
        ids = torch.where(matches.any(1), matches.long().argmax(1) + 1, 0)
        known = (ids[:-1] != 0) & (ids[1:] != 0)
        target = torch.where(ids[:-1] == ids[1:], 0, ids[1:])
        positions = known.nonzero(as_tuple=True)[0].tolist()
        counts[name] = {"known": len(positions), "unknown": len(actions) - len(positions),
                        "changed": int((known & (target != 0)).sum())}
        if not positions:
            continue
        remaining = iter(positions)
        next_known = next(remaining)
        state = None
        for t in range(positions[-1] + 1):
            frame, button = tokens[t:t + 1].long(), actions[t:t + 1]
            if witness is None and t == next_known:
                witness = (frame, button, state)
            if factorized and t == next_known:
                details = ObjectSceneDynamics.scene_details(model, frame, button, state)
                state, mode = details["state"], details["mode"]
            else:
                state = model.advance(frame, button, state)
                mode = None
            if t == next_known:
                features.append(model.preview_render_features(frame, state, mode))
                labels.append(target[t])
                previous.append(ids[t] - 1)
                next_known = next(remaining, -1)
    if not labels:
        raise ValueError("no train transitions with two known preview crops")
    return torch.cat(features), torch.stack(labels), torch.stack(previous), counts, witness


def replacement_logits(logits, previous):
    return logits[:, 1:].scatter(1, previous[:, None], -torch.inf)


def factorized_loss(logits, labels, previous, positive_weight):
    changes = labels != 0
    gate = F.binary_cross_entropy_with_logits(logits[:, 0], changes.to(logits.dtype), pos_weight=positive_weight)
    motif = logits[:0, 1:].sum()
    if changes.any():
        motif = F.cross_entropy(replacement_logits(logits[changes], previous[changes]), labels[changes] - 1)
    return gate + motif


@torch.no_grad()
def metrics(head, features, labels):
    logits = head(features)
    chosen = logits.argmax(1)
    changes = labels != 0
    return {"nll": float(F.cross_entropy(logits, labels)), "correct_copy": int((~changes & (chosen == 0)).sum()),
            "correct_change": int((changes & (chosen == labels)).sum()),
            "false_changes": int((~changes & (chosen != 0)).sum()),
            "predicted_changes": int((chosen != 0).sum())}


@torch.no_grad()
def factorized_metrics(head, features, labels, previous):
    logits = head(features)
    changes, predicted = labels != 0, logits[:, 0] > 0
    replacements = replacement_logits(logits, previous)
    exact = (replacements.argmax(1) + 1 == labels) & changes
    true_positives, changed, triggered = int((predicted & changes).sum()), int(changes.sum()), int(predicted.sum())
    return {"gate_true_positives": true_positives, "gate_predicted_changes": triggered, "actual_changes": changed,
            "gate_precision": true_positives / triggered if triggered else None,
            "gate_recall": true_positives / changed,
            "conditional_exact_motif_count": int(exact.sum()), "conditional_exact_motif_rate": int(exact.sum()) / changed,
            "rendered_exact_change_count": int((predicted & exact).sum()),
            "correct_copy": int((~changes & ~predicted).sum()),
            "gate_bce": float(F.binary_cross_entropy_with_logits(logits[:, 0], changes.float())),
            "motif_nll": float(F.cross_entropy(replacements[changes], labels[changes] - 1))}


@torch.no_grad()
def assert_crop_only(model, baseline, witness, feature):
    tokens, actions, state = witness
    expected = baseline.scene_details(tokens, actions, state)
    actual = model.scene_details(tokens, actions, state)
    torch.testing.assert_close(actual["state"], expected["state"], rtol=0, atol=0)
    torch.testing.assert_close(actual["next"]["joint"], expected["next"]["joint"], rtol=0, atol=0)
    torch.testing.assert_close(actual["preview_render_logits"], model.preview_render_head(feature), rtol=0, atol=0)
    outside = torch.ones_like(tokens[0], dtype=torch.bool)
    outside[:6, 13:19] = False
    factorized = isinstance(model, FactorizedPreviewRenderSceneDynamics)
    for category in (0, 1):
        logits = torch.full_like(actual["preview_render_logits"], -40.)
        crop = tokens[:, :6, 13:19] if category == 0 else model.preview_patterns[:1]
        if factorized:
            matching = (tokens[:, :6, 13:19] == model.preview_patterns).flatten(1).all(1)
            if int(matching.sum()) != 1:
                raise ValueError("expected a known preview in the cache witness")
            alternative = int((~matching).nonzero(as_tuple=True)[0][0])
            current = int(matching.nonzero(as_tuple=True)[0][0])
            logits[:, 0] = 40. if category else -40.
            logits[:, alternative + 1] = 40.
            logits[:, current + 1] = 80.
            if category:
                crop = model.preview_patterns[alternative:alternative + 1]
        else:
            logits[:, category] = 40.
        forced = {**actual, "preview_render_logits": logits}
        for hard in (True, False):
            before = baseline.decode_scene(tokens, expected, hard)
            after = model.decode_scene(tokens, forced, hard)
            torch.testing.assert_close(after[..., outside], before[..., outside], rtol=0, atol=0)
            selected = after if hard else after.argmax(1)
            torch.testing.assert_close(selected[:, :6, 13:19], crop, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--factorized", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    baseline = scene_from_weights(checkpoint).eval()
    if type(baseline) is not SpawnPreviewSceneDynamics or len(baseline.codebook) > 256:
        raise ValueError("fit requires a V39 spawn preview checkpoint with at most 256 visual tokens")
    config = checkpoint["config"]
    captures = [(name, tokens.byte(), actions) for name, tokens, actions, _ in
                training_captures(config, baseline.codebook.numpy())]
    patterns, pattern_counts = patterns_from_train(captures, torch.unique(baseline.object_tiles))
    model = preview_render_scene_weights(checkpoint["model"], patterns, args.factorized).eval().requires_grad_(False)
    head = model.preview_render_head.requires_grad_(True)
    features, labels, previous, counts, witness = training_examples(model, captures)
    del captures
    copies = int((labels == 0).sum())
    changes = len(labels) - copies
    if min(copies, changes) == 0:
        raise ValueError("preview renderer requires both copy and change targets")
    change_weight = copies / changes if args.factorized else min(16., max(1., copies / changes))
    weights = torch.tensor([1.] + [change_weight] * len(patterns))
    log_weights = weights.log()
    def measure():
        return factorized_metrics(head, features, labels, previous) if args.factorized else metrics(head, features, labels)

    before = measure()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-5)
    rng = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        for index in torch.randperm(len(labels), generator=rng).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            wanted = labels[index]
            logits = head(features[index])
            if args.factorized:
                loss = factorized_loss(logits, wanted, previous[index], weights[1])
            else:
                loss = F.cross_entropy(logits + log_weights, wanted, weight=weights)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite preview render fit")
            loss.backward()
            optimizer.step()
    after = measure()
    assert_crop_only(model, baseline, witness, features[:1])
    weights = model.state_dict()
    for name, tensor in checkpoint["model"].items():
        torch.testing.assert_close(weights[name].reshape(-1).view(torch.uint8),
                                   tensor.reshape(-1).view(torch.uint8), rtol=0, atol=0)
    report = {"source_checkpoint_sha256": sha(args.checkpoint), "source_step": checkpoint["step"],
              "manifest_sha256": config["manifest_sha256"], "annotations_sha256": config["annotations_sha256"],
              "script_sha256": sha(Path(__file__)), "test_used": False, "validation_used": False,
              "patterns": len(patterns), "pattern_train_occurrences": pattern_counts, "copies": copies,
              "changes": changes, "examples_by_capture": counts, "epochs": EPOCHS, "batch": BATCH,
              "lr": LR, "seed": SEED, "change_weight": change_weight, "train_log_weight_correction": not args.factorized,
              "factorized": args.factorized, "mode_probability_features": args.factorized,
              "parameters_fitted": sum(p.numel() for p in head.parameters()), "before": before, "after": after,
              "optimizer_exported": False}
    args.output.mkdir(parents=True)
    output = args.output / "model.pt"
    torch.save({"model": weights, "step": checkpoint["step"], "config": {
        **config, "model_family": type(model).__name__, "preview_render_head_only_fit": report
    }}, output)
    restored = scene_from_weights(torch.load(output, map_location="cpu", weights_only=True)).eval()
    torch.testing.assert_close(restored.state_dict(), weights, rtol=0, atol=0)
    assert_crop_only(restored, baseline, witness, features[:1])
    report["checkpoint_sha256"] = sha(output)
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
