"""Fit the V3 pose readout on witnessed rotation and movement branches."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from checks.v3_motion import audit, episodes, warm
from collect_v3_motion import COMMANDS
from model import ACTIONS
from object_annotations import PATTERNS, classify_board, shape_key, without
from scene_model import scene_from_weights


def pose(frame, background, texture, kinds):
    board = classify_board(frame)
    blank = background == -1
    if not np.array_equal(board[~blank], background[~blank]) or (board < -1).any():
        return None
    cells = np.argwhere(blank & (board >= 0))
    if len(cells) != 4 or not (board[tuple(cells.T)] == texture).all():
        return None
    kind = kinds.get(shape_key(texture, cells))
    if kind is None:
        return None
    y, x = map(int, cells.min(0))
    return kind, y, x


@torch.inference_mode()
def samples(model, corpus, split):
    kinds = {}
    for i, (shape, token) in enumerate(zip(model.object_shapes, model.object_tiles, strict=True)):
        colors = np.flatnonzero((PATTERNS == model.codebook[token].numpy()).all((1, 2)))
        if len(colors) == 1:
            kinds[shape_key(int(colors[0]), sorted(shape.tolist()))] = i
    capture = []

    def inspect(tokens, action, details):
        active, current, _, features = model.motion_context(tokens, action, details, collect_samples=True)
        proposed = details["next"]["joint"].flatten(1).argmax(1)
        capture.append((active, current, features, proposed[active]))

    model.motion_head = nn.Identity()
    model.apply_motion_readout = inspect
    result = []
    try:
        for entry, data in episodes(corpus, split):
            state = warm(model, data)
            initial = data["frames"][data["anchors"]]
            count = len(initial)
            backgrounds = [without(classify_board(frame), cells)
                           for frame, cells in zip(initial, data["cells"], strict=True)]
            supervised = np.isin(COMMANDS, [ACTIONS.index(button) for button in ("a", "b", "left", "right", "down")])
            branches = np.flatnonzero(supervised.any(1))
            carry = state.repeat(len(branches), 1, 1, 1)
            for step in range(COMMANDS.shape[1]):
                if not len(branches):
                    break
                frames = data["branches"][:, branches, step].transpose(1, 0, 2, 3).reshape(-1, 72, 80)
                inputs = torch.from_numpy(frames[:, None])
                actions = torch.from_numpy(COMMANDS[branches, step].repeat(count)).long()
                selected = np.flatnonzero(supervised[branches, step].repeat(count))
                for indexes in np.array_split(selected, max(1, (len(selected) + 31) // 32)):
                    if not len(indexes):
                        continue
                    model.predict_frame(inputs[indexes], actions[indexes], carry[indexes])
                    active, current, features, proposed = capture.pop()
                    for row, local in enumerate(active.tolist()):
                        index = indexes[local]
                        branch, anchor = branches[index // count], index % count
                        stack, texture = backgrounds[anchor], int(data["texture"][anchor])
                        before = pose(frames[index], stack, texture, kinds)
                        if before is None:
                            continue
                        kind, y, x = before
                        if current[row] != kind * 180 + y * 10 + x:
                            continue
                        after = pose(data["branches"][anchor, branch, step + 1], stack, texture, kinds)
                        if after is None:
                            continue
                        delta = (after[0], after[1] - y, after[2] - x)
                        if max(abs(delta[1]), abs(delta[2])) > 3:
                            continue
                        candidate = int(proposed[row])
                        baseline = (candidate // 180, candidate % 180 // 10 - y, candidate % 10 - x)
                        result.append((features[row], int(actions[index]), delta, baseline))
                remaining = supervised[branches, step + 1:].any(1)
                branches = branches[remaining]
                indexes = np.flatnonzero(remaining.repeat(count))
                if len(indexes):
                    carry = torch.cat([model.advance_observed_frame(inputs[batch], actions[batch], carry[batch])
                                       for batch in np.array_split(indexes, (len(indexes) + 31) // 32)])
            print(json.dumps({"split": split, "seed": entry["seed"], "samples_so_far": len(result)}), flush=True)
    finally:
        model.motion_head = None
        del model.apply_motion_readout
    return result


def fit(base, train, validation):
    vocabulary = sorted({sample[2] for sample in train})
    lookup = {label: index for index, label in enumerate(vocabulary)}
    features = torch.stack([sample[0] for sample in train])
    targets = torch.tensor([lookup[sample[2]] for sample in train])
    width = 48
    net = nn.Sequential(nn.Linear(features.shape[1], width), nn.ReLU(), nn.Linear(width, len(vocabulary)))
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.006)
    for _ in range(65):
        for batch in torch.randperm(len(features)).split(128):
            loss = F.cross_entropy(net(features[batch]), targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    confidence = torch.full((len(ACTIONS),), 1.1)
    net.eval()
    with torch.inference_mode():
        chances, classes = net(torch.stack([sample[0] for sample in validation])).softmax(1).max(1)
    decisions = {}
    for group in (("a", "b"), ("left", "right"), ("down",)):
        indexes = [i for i, sample in enumerate(validation) if ACTIONS[sample[1]] in group]
        baseline = sum(validation[i][2] == validation[i][3] for i in indexes)
        scores = {}
        for cut in (0.8, 0.9, 0.95, 0.99):
            scores[cut] = sum(validation[i][2] == (vocabulary[classes[i]] if chances[i] >= cut
                                                 else validation[i][3]) for i in indexes)
        cut, best = max(scores.items(), key=lambda pair: (pair[1], pair[0]))
        if best > baseline:
            for button in group:
                confidence[ACTIONS.index(button)] = cut
        decisions["/".join(group)] = {"samples": len(indexes), "baseline": baseline, "candidate": max(best, baseline),
                                       "threshold": float(confidence[ACTIONS.index(group[0])])}
    weights = {"motion_offsets": torch.tensor(vocabulary), "motion_confidence": confidence,
               **{f"motion_head.{name}": weight for name, weight in net.state_dict().items()}}
    return scene_from_weights({**base, **weights}).eval(), weights, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(193)
    args.output.mkdir()
    base = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    original = scene_from_weights(base).eval()
    train = samples(original, args.corpus, "train")
    validation = samples(original, args.corpus, "validation")
    candidate, patch, decisions = fit(base, train, validation)
    torch.save(patch, args.output / "motion.pt")
    results = audit(candidate, args.corpus, "validation")
    report = {
        "dataset_sha256": hashlib.sha256((args.corpus / "manifest.json").read_bytes()).hexdigest(),
        "base_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "train_samples": len(train),
        "train_actions": dict(Counter(ACTIONS[row[1]] for row in train)),
        "validation_samples": len(validation),
        "validation_gates": decisions,
        "validation_rollouts": results,
    }
    (args.output / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gates": decisions, "rollouts": {k: v for k, v in results.items() if "/" not in k}}), flush=True)


if __name__ == "__main__":
    main()
