"""Reserve a new independent test after the first test was opened for diagnosis."""
import hashlib
import json
from pathlib import Path
import numpy as np
from collect import collect_episode
from run import ROOT


def main():
    root = ROOT / "data"
    original = json.loads((root / "adjustris-v1/manifest.json").read_text())
    output = root / "heldout-v2"
    output.mkdir(exist_ok=False)
    manifest = dict(original)
    manifest["episodes"] = [dict(e, file="adjustris-v1/" + e["file"]) for e in original["episodes"] if e["split"] != "test"]
    seen = {e["prefix_sha256"] for e in original["episodes"]}
    for i in range(8):
        profile = "exploration" if i == 5 else "topout" if i == 6 else "pause" if i == 7 else "heuristic"
        frames, actions, metadata = collect_episode(original["context"], 9032026 + 1013 * i, 1400, profile)
        prefix = hashlib.sha256(frames[:32].tobytes() + actions[:31].tobytes()).hexdigest()
        if prefix in seen:
            raise RuntimeError("new test overlaps a previously observed prefix")
        seen.add(prefix)
        metadata.update({"split": "test", "run_id": f"holdout-v2-{i}", "prefix_sha256": prefix})
        path = output / f"episode-{i:02d}.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, frames=frames, actions=actions, metadata=json.dumps(metadata))
        manifest["episodes"].append({"file": str(path.relative_to(root)), "split": "test",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "steps": len(actions),
            "events": metadata["events"], "profile": profile, "prefix_sha256": prefix})
    manifest["coverage"] = {"lines_cleared": sum(e["events"]["lines_cleared"] for e in manifest["episodes"]),
        "terminal_episodes": sum(e["events"]["terminal_observed"] for e in manifest["episodes"]),
        "first_test_opened_for_diagnosis": True, "final_test_seed": 9032026}
    manifest["status"] = "complete"
    with (root / "manifest.json").open("x") as f:
        json.dump(manifest, f, indent=2)
    print("New test reserved before token training; not used to fit codebook or select weights.")


if __name__ == "__main__":
    main()
