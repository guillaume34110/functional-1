"""Fit a small pause readout from frame-verified TRAIN branches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model import ACTIONS
from scene_model import CounterSceneDynamics, scene_from_weights
from token_model import encode_frames
from train_spawn_preview import sha


def fit(base, archive, samples, panel):
    counts = torch.zeros((3, len(ACTIONS), 2), dtype=torch.long)
    resume = torch.zeros(len(ACTIONS), dtype=torch.long)
    for sample in samples:
        i = sample["id"]
        frames = archive[f"frames_{i}"]
        actions = [ACTIONS.index(a) for a in sample["commands"]]
        if frames.shape != (len(actions) + 1, 72, 80) or frames.dtype != np.uint8 or (frames > 3).any():
            raise ValueError("invalid captured pause branch")
        paused = (frames == panel).all(axis=(1, 2))
        if int(paused[1:].sum()) != sample["paused_frames"]:
            raise ValueError("pause observations disagree with the witnessed panel")
        if sample["context"] == "terminal":
            counts[2, actions[0], int(paused[1])] += 1
            continue
        for t, a in enumerate(actions):
            counts[int(paused[t]), a, int(paused[t + 1])] += 1
        if "control_commands" in sample:
            reference = archive[f"control_frames_{i}"]
            glyphs = archive[f"control_digits_{i}"]
            suffix = len(reference) - 1
            if (suffix > 0 and np.array_equal(frames[-suffix:], reference[1:])
                    and np.array_equal(archive[f"digits_{i}"][-suffix:], glyphs[1:])):
                resume[ACTIONS.index(sample["control_commands"][0])] += 1
    if (not counts[0, :, 1].any() or not counts[1, :, 0].any() or not resume.any()
            or (counts > 0).all(2).any() or int((resume > 0).sum()) != 1):
        raise ValueError("insufficient or contradictory pause witnesses")
    tokens, unknown = encode_frames(panel[None], base.codebook.numpy())
    if unknown:
        raise ValueError("pause panel is outside the frozen scene vocabulary")
    return {"pause_panel": torch.from_numpy(tokens[0]).to(torch.uint8),
            "pause_transition": ((counts[:, :, 1] + 1) / (counts[:, :, 0] + 1)).log(),
            "pause_resume_action": resume.argmax()}, counts, resume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--branches", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    if not isinstance(base, CounterSceneDynamics) or base.pause_transition is not None:
        raise ValueError("expected a frozen scene and counter checkpoint")
    manifest_path = Path("data/manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    train = {r["file"]: r["sha256"] for r in manifest["episodes"] if r["split"] == "train"}
    lineage = json.loads(args.branches.with_suffix(".json").read_bytes())
    digest = sha(args.branches)
    if (lineage["test_used"] or lineage["archive_sha256"] != digest
            or lineage["source_manifest_sha256"] != sha(manifest_path)
            or lineage["source_manifest_sha256"] != checkpoint["config"]["manifest_sha256"]
            or lineage["rom_sha256"] != manifest["context"]["rom"]["sha256"]):
        raise ValueError("pause capture lineage does not match the checkpoint")
    samples = [r for r in lineage["samples"] if r["split"] == "train"]
    if (not samples or len({r["id"] for r in samples}) != len(samples)
            or any(train.get(r["file"]) != r["sha256"]
                   or r["proof"]["verified_frames"] != r["anchor"] + 1 for r in samples)):
        raise ValueError("pause fit requires independent verified TRAIN branches")
    with np.load(args.branches, allow_pickle=False) as archive:
        sample = next(r for r in samples if r["context"] == "recorded" and r["paused_frames"] > 0)
        panel = archive[f'frames_{sample["id"]}'][1]
        if hashlib.sha256(panel.tobytes()).hexdigest() != lineage["panel_sha256"]:
            raise ValueError("pause panel differs from its capture")
        learned, counts, resume = fit(base, archive, samples, panel)
    model = scene_from_weights({**checkpoint["model"], **learned}).eval()
    state = model.state_dict()
    for name, value in checkpoint["model"].items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)
    report = {"source_sha256": sha(args.checkpoint), "branches_sha256": digest, "fit_split": "train",
              "validation_used": False, "test_used": False, "train_branches": len(samples),
              "counts": counts.tolist(), "resume_votes": resume.tolist()}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({**checkpoint, "model": state, "pause_fit": report}, destination)
    report["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
