"""Compare down branches on V3 with the actual training exposure ledger."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_controls import supported
from intervention_training import BUTTONS
from run import ROOT
from scene_model import scene_from_weights
from train_scene import load_action_interventions, load_scene_data, resume_exposure


def measure(model, bundle, context, counts):
    arm = BUTTONS.index("down")
    frames = torch.from_numpy(bundle["tokens"][arm].copy()).long()
    actions = torch.from_numpy(bundle["actions"][arm].copy()).long()
    labels = {key: value[arm] for key, value in bundle["labels"].items()}
    known = (labels["shape_id"] >= 0) & np.isin(labels["evidence"], (1, 2))
    states = {"observed": context, "autonomous": context}
    auto = frames[:1]
    for t, action in enumerate(actions):
        observed, target = frames[t:t + 1], frames[t + 1:t + 2]
        category = "unknown_contact"
        if known[t]:
            old = labels["cells"][t]
            background = observed[0, :, 2:12].numpy().copy()
            background[old[:, 0], old[:, 1]] = model.blank_id
            category = "supported" if supported(background, old, model.blank_id) else "clear"
        for protocol, current in (("observed", observed), ("autonomous", auto)):
            details = model.scene_details(current, action[None], states[protocol])
            states[protocol] = details["state"]
            prediction = model.decode_scene(current, details, hard=True)
            if protocol == "autonomous":
                auto = prediction
            row = counts[f"{protocol}/{category}"]
            row["steps"] += 1
            board_error = int((prediction[:, :, 2:12] != target[:, :, 2:12]).sum())
            row["exact_boards"] += board_error == 0
            row["board_tile_errors"] += board_error
            if known[t + 1]:
                next_obj = details["next"]
                y, x = map(int, labels["cells"][t + 1].min(0))
                wanted = int(labels["shape_id"][t + 1]) * 180 + y * 10 + x
                row["known_poses"] += 1
                row["exact_poses"] += int(next_obj["joint"].flatten().argmax()) == wanted and next_obj["visible"].item() > 0


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--interventions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.candidate, map_location="cpu", weights_only=True)
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=True)
    _, episodes, annotations, _, _, digest, annotation_digest = load_scene_data(checkpoint)
    if baseline["config"]["manifest_sha256"] != digest:
        raise ValueError("baseline capture lineage mismatch")
    models = {name: scene_from_weights(weights["model"]).eval()
              for name, weights in (("baseline", baseline), ("candidate", checkpoint))}
    model = models["candidate"]
    for buffer in ("codebook", "object_shapes", "object_tiles"):
        if not torch.equal(getattr(model, buffer), getattr(models["baseline"], buffer)):
            raise ValueError("baseline and candidate do not share the visible vocabulary")
    tables, addon_digest = load_action_interventions(
        checkpoint, model, episodes, annotations, digest, annotation_digest, args.interventions)
    ledger = resume_exposure(checkpoint, tables["train"], addon_digest)
    exposed = set(map(tuple, ledger["keys"]))
    manifest = json.loads((ROOT / "data/manifest.json").read_bytes())
    addon = json.loads((args.interventions / "manifest.json").read_bytes())
    names = {split: [row["file"] for row in manifest["episodes"] if row["split"] == split] for split in tables}
    records = {(row["parent_file"], row["anchor"]): row for row in addon["episodes"]}
    coverage, counts = Counter(), {}
    for split, table in tables.items():
        for i, (tokens, actions) in enumerate(episodes[split]):
            anchors = sorted(t for episode, t in table if episode == i)
            if not anchors:
                continue
            states = dict.fromkeys(models)
            for t in range(anchors[-1] + 1):
                key = (i, t)
                if key in table:
                    row = records[(names[split][i], t)]
                    group = "validation"
                    if split == "train":
                        origin = "new" if "contact_stratum" in row else "inherited"
                        group = f"{origin}/{'exposed' if key in exposed else 'unseen'}"
                    coverage[group] += 1
                    for name, network in models.items():
                        measure(network, table[key], states[name],
                                counts.setdefault(f"{name}/{group}", defaultdict(Counter)))
                if t < anchors[-1]:
                    frame = torch.from_numpy(tokens[t:t + 1].copy()).long()
                    command = torch.tensor([int(actions[t])])
                    for name, network in models.items():
                        states[name] = network.advance(frame, command, states[name])
    if (sum(coverage.values()) != sum(map(len, tables.values()))
            or coverage["new/exposed"] + coverage["inherited/exposed"] != len(exposed)
            or coverage["new/exposed"] + coverage["new/unseen"] != addon["new_anchors"]):
        raise ValueError("incomplete anchor or exposure coverage")
    report = {"test_used": False, "manifest_sha256": digest, "interventions_sha256": addon_digest,
              "checkpoint_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for name, path in (("baseline", args.baseline), ("candidate", args.candidate))},
              "candidate_step": checkpoint["step"], "anchors": dict(coverage), "counts": counts}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
