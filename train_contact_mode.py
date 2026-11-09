"""Fit the visible-contact head on ROM captures and proven generated geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit_controls import BUTTONS, supported
from audit_long import recorded_path
from scene_model import contact_mode_scene_weights, scene_from_weights
from token_model import encode_frames
from train_scene import label_mask
from train_spawn_preview import sha, training_captures


def examples(model, tokens, masks, cells):
    backgrounds = tokens.masked_fill(masks, model.blank_id)[:, :, 2:12].numpy()
    targets = [supported(board, piece, model.blank_id) for board, piece in zip(backgrounds, cells)]
    return model.contact_features(tokens, masks), torch.tensor(targets, dtype=torch.bool)


@torch.no_grad()
def collect(model, base, captures):
    observed, autonomous = [], []
    for episode, (_, frames, actions, labels) in enumerate(captures):
        known, masks = label_mask({key: value[:, None] for key, value in labels.items()}, 0,
                                  torch.ones(len(frames), dtype=torch.bool))
        if not known.any():
            continue
        observed.append(examples(model, frames[known], masks[known], labels["cells"][known].numpy()))
        anchors = known[32:-32].nonzero().flatten() + 32
        if not len(anchors):
            continue
        anchor = int(anchors[len(anchors) * (episode % 5 + 1) // 6])
        state = None
        for t in range(anchor - 32, anchor):
            state = base.advance(frames[t:t + 1], actions[t:t + 1], state)
        previous = next((button for button, value in BUTTONS.items() if value == int(actions[anchor - 1])), None)
        commands = actions[anchor:anchor + 32].numpy() if episode % 2 else np.full(32, BUTTONS["down"])
        events, images = recorded_path(base, frames[anchor:anchor + 1], state,
                                       labels["cells"][anchor].numpy(), commands, previous)
        cells = []
        for event in events:
            if event["status"] not in ("motion", "deposit") or event.get("collision"):
                break
            cells.append(event["cells"])
        if not cells:
            continue
        tokens, unknown = encode_frames(np.stack(images[1:len(cells) + 1]), model.codebook.numpy())
        if unknown:
            raise ValueError("generated geometry contains an unknown tile")
        tokens = torch.from_numpy(tokens).long()
        cells = np.asarray(cells)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask[torch.arange(len(cells))[:, None], cells[:, :, 0], cells[:, :, 1] + 2] = True
        autonomous.append(examples(model, tokens, mask, cells))
    if not autonomous:
        raise ValueError("no certified autonomous contact examples")
    return {name: tuple(torch.cat(column) for column in zip(*samples))
            for name, samples in (("observed", observed), ("autonomous", autonomous))}


@torch.no_grad()
def metrics(head, examples):
    result = {}
    for name, (features, targets) in examples.items():
        logits = head(features)[:, 0]
        result[name] = {"unsupported": int((~targets).sum()), "supported": int(targets.sum()),
                        "false_support": int(((logits > 0) & ~targets).sum()),
                        "missed_support": int(((logits <= 0) & targets).sum()),
                        "nll": float(F.binary_cross_entropy_with_logits(logits, targets.float()))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(55)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    model = contact_mode_scene_weights(checkpoint["model"]).eval()
    cache = collect(model, base, training_captures(checkpoint["config"], model.codebook.numpy()))
    features, targets = (torch.cat(column) for column in zip(*cache.values()))
    if any(not y.any() or y.all() for _, y in cache.values()):
        raise ValueError("both contact classes must be proven in each source")
    head = model.contact_head
    before = metrics(head, cache)
    count = torch.bincount(targets.long())
    weights = len(targets) / (2 * count[targets.long()])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.08)
    for _ in range(256):
        loss = F.binary_cross_entropy_with_logits(head(features)[:, 0], targets.float(), weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    state = model.state_dict()
    for key, tensor in checkpoint["model"].items():
        torch.testing.assert_close(state[key], tensor, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "before": before, "after": metrics(head, cache)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "contact_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
