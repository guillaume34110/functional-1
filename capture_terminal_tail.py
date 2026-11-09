"""Extend verified ROM terminal captures with released-button observations."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from audit_topout import replay_reference
from run import ROOT, read_episode
from scene_model import scene_from_weights
from token_model import encode_frames
from train_spawn_preview import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = scene_from_weights(checkpoint).eval()
    manifest_path = ROOT / "data/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if sha(manifest_path) != checkpoint["config"]["manifest_sha256"]:
        raise ValueError("terminal corpus manifest mismatch")
    captures, entries, rejected = [], [], []
    for entry in manifest["episodes"]:
        if entry["split"] != args.split or not entry["events"]["terminal_observed"]:
            continue
        path = ROOT / "data" / entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("terminal capture digest mismatch")
        frames, _, metadata = read_episode(path)
        if not metadata["terminal_annotated"] or not np.all(frames[-8:-6, :, 8:48] == 3):
            rejected.append(entry["file"])
            continue
        _, _, _, tail = replay_reference(entry, tail_steps=48)
        captures.append(np.concatenate((frames[-8:], np.stack(tail))))
        entries.append({"file": entry["file"], "sha256": entry["sha256"], "first_panel": len(frames) - 8})
    if not captures:
        raise ValueError("no observed terminal tails")
    frames = np.stack(captures)
    book = model.codebook.numpy()
    flat = frames.reshape(-1, 72, 80)
    codes, missing = encode_frames(flat, book)
    rebuilt = book[codes].transpose(0, 1, 3, 2, 4).reshape(-1, 72, 80)
    exact = (flat == rebuilt).all((1, 2)).reshape(frames.shape[:2])
    agreement = [Counter(frame.tobytes() for frame in phase).most_common(1)[0][1]
                 for phase in frames.transpose(1, 0, 2, 3)]
    report = {"split": args.split, "test_used": False, "fit_used": False,
              "manifest_sha256": sha(manifest_path), "rom_sha256": manifest["context"]["rom"]["sha256"],
              "entries": entries, "excluded_terminal_flags": rejected, "tail_noop_steps": 48,
              "unknown_tiles": int(missing), "exact_codebook_by_step": exact.sum(0).tolist(),
              "most_common_image_by_step": agreement}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "tails.npz", frames=frames, metadata=json.dumps(report, allow_nan=False))
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
