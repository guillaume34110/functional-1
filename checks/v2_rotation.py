"""Compare V2 with paired A/B/idle ROM branches from held-out contact positions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model import ACTIONS
from object_annotations import classify_board
from run import read_episode
from scene_model import scene_from_weights


def checked_path(root, entry):
    path = (root / entry["file"]).resolve()
    if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"invalid capture path or digest: {path}")
    return path


@torch.inference_mode()
def audit(model, corpora, rotation_root):
    sources = {}
    for root in corpora:
        manifest = json.loads((root / "manifest.json").read_text())
        for entry in manifest["episodes"]:
            if entry["split"] == "validation":
                sources[entry["sha256"]] = checked_path(root, entry)
    report = Counter()
    manifest = json.loads((rotation_root / "manifest.json").read_text())
    for entry in manifest["records"]:
        if entry["split"] != "validation":
            continue
        source, actions, _ = read_episode(sources[entry["source_sha256"]])
        path = checked_path(rotation_root, entry)
        with np.load(path, allow_pickle=False) as capture:
            for anchor, branches in zip(capture["anchors"], capture["frames"], strict=True):
                anchor = int(anchor)
                state = None
                for t in range(max(0, anchor - 16), anchor):
                    frame = torch.from_numpy(source[t].copy())[None, None]
                    state = model.advance_observed_frame(frame, torch.tensor([int(actions[t])]), state)
                if not np.array_equal(source[anchor], branches[0, 0]):
                    raise ValueError("rotation anchor differs from its original capture")
                start = torch.from_numpy(source[anchor].copy())[None, None]
                reference = [classify_board(branch[1]) for branch in branches]
                generated = []
                for button in ("a", "b", "noop"):
                    next_frame, _ = model.predict_frame(start, torch.tensor([ACTIONS.index(button)]), state)
                    generated.append(classify_board(next_frame[0, 0].numpy()))
                report["anchors"] += 1
                for i in range(2):
                    if (reference[i] < -1).any() or (reference[2] < -1).any():
                        continue
                    rotation = not np.array_equal(reference[i], reference[2])
                    label = "button_changes_board" if rotation else "button_matches_idle"
                    report[label] += 1
                    report[label + "_exact"] += np.array_equal(reference[i], generated[i])
    if not report["anchors"]:
        raise ValueError("missing held-out rotation witnesses")
    return dict(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--rotations", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = scene_from_weights(checkpoint["model"]).eval()
    print(json.dumps(audit(model, args.corpus, args.rotations)), flush=True)


if __name__ == "__main__":
    main()
