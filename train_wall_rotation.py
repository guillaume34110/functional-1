"""Fit visible wall rotations on disjoint native ROM episodes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model import ACTIONS
from object_annotations import PATTERNS, candidates, classify_board, shape_key
from scene_model import scene_from_weights


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation_rows(frames, actions, kinds):
    boards = [classify_board(frame) for frame in frames]
    poses = [candidates(board) if (board >= 0).sum() == 4 and (board >= -1).all() else [] for board in boards]
    for t, action in enumerate(actions):
        name = ACTIONS[action]
        if name not in ("a", "b") or len(poses[t]) != 1 or len(poses[t + 1]) != 1:
            continue
        color, before = poses[t][0]
        after_color, after = poses[t + 1][0]
        if color != after_color:
            continue
        cells = np.asarray(before)
        y, x = map(int, cells.min(0))
        if x > 1 and cells[:, 1].max() < 8:
            continue
        v, w = map(int, np.min(after, axis=0))
        source = kinds.get(shape_key(color, before))
        target = kinds.get(shape_key(color, after))
        if source is not None and target is not None and max(abs(v - y), abs(w - x)) <= 3:
            yield t, (source, int(name == "b"), x, min(y, 2)), (target, v - y, w - x)


def collect_phase_votes(votes, rows, actions, first_kind):
    by_step = {t: (key, target) for t, key, target in rows}
    turns = [t for t in range(1, len(actions)) if ACTIONS[actions[t]] in ("a", "b")]
    if not turns or turns[0] not in by_step or by_step[turns[0]][0][0] != first_kind:
        return
    angle, flipped = 0, False
    for step in turns:
        if step not in by_step:
            break
        (source, button, _, _), target = by_step[step]
        reverse = bool(button) ^ flipped
        if source != target[0]:
            votes[angle, int(reverse)][target] += 1
            angle = (angle + (-1 if reverse else 1)) % 4
        flipped = not flipped


@torch.inference_mode()
def witnessed_replay(base, model, clips, table):
    results = Counter()
    for frames, actions, rows in clips:
        rotation = {t: key for t, key, _ in rows}
        start = next(t for t, f in enumerate(frames) if len(candidates(classify_board(f))) == 1)
        memory = None
        for t in range(start, max(rotation) + 1):
            frame = torch.from_numpy(frames[t:t+1, None].copy())
            action = torch.tensor([int(actions[t])])
            if t in rotation:
                expected = frames[t + 1, :, 8:48]
                v1, _ = base.predict_frame(frame, action, memory)
                v2, _ = model.predict_frame(frame, action, memory)
                covered = table[rotation[t]][0] >= 0
                results.update(total=1, covered=int(covered),
                               original_exact=int(np.array_equal(v1[0, 0, :, 8:48].numpy(), expected)),
                               candidate_exact=int(np.array_equal(v2[0, 0, :, 8:48].numpy(), expected)))
            memory = model.advance_observed_frame(frame, action, memory)
    return dict(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fit-zigzag-phase", action="store_true",
                        help="add pivot phases to an existing wall checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = checkpoint.get("model", checkpoint)
    base = scene_from_weights(weights).eval()
    if args.fit_zigzag_phase:
        if base.wall_poses is None or base.wall_first_poses is None:
            raise ValueError("pivot fitting requires the existing wall readouts")
        first_kind = base.object_shapes.tolist().index([[0, 0], [0, 1], [1, 1], [1, 2]])
    phase_votes = defaultdict(Counter)
    kinds = {}
    for i, (shape, token) in enumerate(zip(base.object_shapes, base.object_tiles, strict=True)):
        colors = np.flatnonzero((PATTERNS == base.codebook[token].numpy()).all((1, 2)))
        if len(colors) == 1:
            kinds[shape_key(int(colors[0]), sorted(shape.tolist()))] = i
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    if manifest["rom_sha256"] != json.loads(Path("context.json").read_text())["rom"]["sha256"]:
        raise ValueError("wall corpus refers to a different ROM")
    votes, first_votes, validation = defaultdict(Counter), defaultdict(Counter), []
    for row in manifest["records"]:
        if row["scenario"] not in ("left_wall", "right_wall"):
            continue
        file = args.corpus / row["file"]
        if digest(file) != row["sha256"]:
            raise ValueError("wall capture digest mismatch")
        with np.load(file, allow_pickle=False) as capture:
            frames, actions = capture["frames"][:, 1::2, 1::2], capture["actions"]
        rows = list(rotation_rows(frames, actions, kinds))
        if row["split"] == "train":
            if args.fit_zigzag_phase:
                collect_phase_votes(phase_votes, rows, actions, first_kind)
            first_turn = next((t for t in range(1, len(actions)) if ACTIONS[actions[t]] in ("a", "b")), None)
            for t, key, target in rows:
                votes[key][target] += 1
                if t == first_turn:
                    first_votes[key][target] += 1
        elif row["split"] == "validation" and rows:
            validation.append((frames, actions, rows))
    empty = torch.zeros((len(base.object_shapes), 2, 10, 3, 3), dtype=torch.int16)
    empty[..., 0] = -1
    table = weights.get("wall_poses", empty).clone()
    learned = {key: next(iter(values)) for key, values in votes.items()
               if len(values) == 1 and table[key][0] < 0}
    if not validation or (not args.fit_zigzag_phase and not learned):
        raise ValueError("wall corpus has no proved training and validation cases")
    for key, target in learned.items():
        table[key] = torch.tensor(target, dtype=table.dtype)
    first = weights.get("wall_first_poses", empty).clone()
    learned_first = {key: next(iter(values)) for key, values in first_votes.items()
                     if len(values) == 1 and first[key][0] < 0}
    for key, target in learned_first.items():
        first[key] = torch.tensor(target, dtype=table.dtype)
    state = {**weights, "wall_poses": table, "wall_first_poses": first}
    if args.fit_zigzag_phase:
        if any(len(phase_votes[angle, direction]) != 1 for angle in range(4) for direction in range(2)):
            raise ValueError("pivot readout requires one consistent result for each phase and direction")
        table = base.wall_poses
        phases = [[next(iter(phase_votes[angle, direction])) for direction in range(2)] for angle in range(4)]
        state = {**weights, "wall_phase_poses": torch.tensor(phases, dtype=torch.int16)}
    candidate = scene_from_weights(state).eval()
    report = {"source_sha256": digest(args.checkpoint), "train_rotations": sum(sum(v.values()) for v in votes.values()),
              "learned_contexts": len(learned), "learned_first_contexts": len(learned_first),
              "validation": witnessed_replay(base, candidate, validation, table)}
    if args.fit_zigzag_phase:
        report["phase_witnesses"] = sum(sum(v.values()) for v in phase_votes.values())
    args.output.mkdir()
    report_name = "wall_phase_fit" if args.fit_zigzag_phase else "wall_fit"
    torch.save({**checkpoint, "model": state, report_name: report}, args.output / "model.pt")
    report["checkpoint_sha256"] = digest(args.output / "model.pt")
    (args.output / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
