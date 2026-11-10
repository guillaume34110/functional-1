"""Compare causal rollouts with the V3 movement intervention corpus."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_v3_motion import BRANCHES, COMMANDS, HISTORY
from scene_model import scene_from_weights


GROUPS = ("idle", "rotation_tap", "rotation_tap", "rotation_burst", "rotation_burst",
          "lateral_tap", "lateral_tap", "lateral_hold", "lateral_hold", "drop_hold")


def episodes(root, split):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["commands"] != [list(branch) for branch in BRANCHES] or manifest["history"] != HISTORY:
        raise ValueError("unexpected motion dataset contract")
    for entry in manifest["episodes"]:
        if entry["split"] != split:
            continue
        path = root / entry["file"]
        if not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"invalid capture: {path}")
        with np.load(path, allow_pickle=False) as data:
            yield entry, dict(data)


def warm(model, data):
    state = None
    anchors = data["anchors"]
    for delta in range(-HISTORY, 0):
        indexes = anchors + delta
        frame = torch.from_numpy(data["frames"][indexes, None])
        actions = torch.from_numpy(data["actions"][indexes]).long()
        state = model.advance_observed_frame(frame, actions, state)
    return state


@torch.inference_mode()
def audit(model, root, split):
    report = {}
    for _, data in episodes(root, split):
        state = warm(model, data)
        initial = torch.from_numpy(data["frames"][data["anchors"], None])
        for branch, commands in enumerate(COMMANDS):
            frame, carry = initial, state
            group = GROUPS[branch]
            for step, action in enumerate(commands, 1):
                frame, carry = model.predict_frame(frame, torch.full((len(initial),), int(action)), carry)
                if step not in (1, len(commands)):
                    continue
                reference = data["branches"][:, branch, step, :, 8:48]
                expected_idle = data["branches"][:, 0, step, :, 8:48]
                predicted = frame[:, 0, :, 8:48].numpy()
                exact = (predicted == reference).all((1, 2))
                effects = (reference != expected_idle).any((1, 2))
                for index, stratum in enumerate(data["strata"]):
                    wall = stratum.split("/")[-2]
                    for key in (f"{group}@{step}", f"{group}@{step}/{wall}"):
                        values = report.setdefault(key, Counter())
                        values.update(total=1, exact=int(exact[index]),
                                      changed=int(effects[index]), changed_exact=int(effects[index] and exact[index]))
    if not report:
        raise ValueError(f"no motion captures for {split}")
    return {key: dict(counts) for key, counts in sorted(report.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    results = audit(scene_from_weights(weights["model"]).eval(), args.corpus, args.split)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: value for key, value in results.items() if "/" not in key}), flush=True)


if __name__ == "__main__":
    main()
