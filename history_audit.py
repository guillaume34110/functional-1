"""Measure contradictory next-screen labels for identical visible histories."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(episodes, lengths=(1, 4, 16, 32, 64, 128)):
    results = {}
    for length in lengths:
        groups = defaultdict(Counter)
        changed_counts = Counter()
        for hashes, actions in episodes:
            for t in range(length - 1, len(actions)):
                condition = hashlib.blake2b(b"".join(hashes[t-length+1:t+1]) + actions[t-length+1:t+1].tobytes(), digest_size=16).digest()
                groups[condition][hashes[t + 1]] += 1
                if hashes[t + 1] != hashes[t]:
                    changed_counts[condition] += 1
        total = sum(sum(g.values()) for g in groups.values())
        ambiguous = {k for k, g in groups.items() if len(g) > 1}
        results[str(length)] = {"transitions": total, "distinct_histories": len(groups),
            "ambiguous_histories": len(ambiguous),
            "empirical_minimum_exact_screen_error": sum(sum(g.values()) - max(g.values()) for g in groups.values()) / max(1, total),
            "ambiguous_changed_fraction": sum(changed_counts[k] for k in ambiguous) / max(1, sum(changed_counts.values()))}
    return results


def main():
    check = audit([([b"p", b"x", b"y"], np.zeros(2, dtype=np.uint8)),
                   ([b"q", b"x", b"z"], np.zeros(2, dtype=np.uint8))], (1, 2))
    assert check["1"]["ambiguous_histories"] == 1 and check["2"]["ambiguous_histories"] == 0
    root = Path("data")
    manifest = json.loads((root / "manifest.json").read_text())
    episodes = []
    for entry in manifest["episodes"]:
        if entry["split"] == "test":
            continue
        with np.load(root / entry["file"], allow_pickle=False) as data:
            hashes = [hashlib.blake2b(f.tobytes(), digest_size=16).digest() for f in data["frames"]]
            episodes.append((hashes, data["actions"].copy()))
    result = audit(episodes)
    Path("history-audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
