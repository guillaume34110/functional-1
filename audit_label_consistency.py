"""Offline pixel/label compositor audit, never an oracle of internal game state."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from audit_controls import supported
from intervention_training import load_interventions
from object_annotations import candidates, classify_board, without
from run import ROOT, read_episode
from token_model import encode_frames, load_tile_model
from train_scene import composition_targets, label_mask, visual_absence


def delta_evidence(current, target, old, new, old_style, next_style, blank, known):
    """Observable translation coverage, with deposits and identity uncertainty separated."""
    if not known:
        return {"category": "unknown"}
    old, new = np.asarray(old), np.asarray(new)
    before, after = current.copy(), target.copy()
    before[old[:, 0], old[:, 1]] = blank
    after[new[:, 0], new[:, 1]] = blank
    if np.array_equal(after, current): return {"category": "deposit"}
    if old_style != next_style: return {"category": "new_style"}
    if not np.array_equal(before, after): return {"category": "background_rewrite"}
    delta = new.min(0) - old.min(0)
    centre_half = new.sum(0) - old.sum(0)  # Integer quarter-cell units: four cells/object.
    maximum = int(np.abs(delta).max())
    def clearance(cells):
        low, high = cells.min(0), cells.max(0)
        return int(min(low[0], low[1], 17 - high[0], 9 - high[1]))
    # Generic49 translations of the observed cells, without collision or game rules.
    valid_offsets = sum(bool(np.all((old + [dy, dx] >= 0) & (old + [dy, dx] < [18, 10])))
                        for dy in range(-3, 4) for dx in range(-3, 4))
    return {"category": "continuous_visual", "delta_origin_yx": delta.tolist(),
            "delta_centre_quarter_cells_yx": centre_half.tolist(),
            "shape_equal": bool(np.array_equal(old - old.min(0), new - new.min(0))),
            "max_delta": maximum, "local3": maximum <= 3, "immobile": bool(np.array_equal(old, new)),
            "old_border_clearance": clearance(old), "next_border_clearance": clearance(new),
            "valid_translation_candidates49": valid_offsets, "all_translation_candidates_invalid": valid_offsets == 0,
            "old_supported": supported(before, old, blank), "next_supported": supported(before, new, blank)}


def add_delta_counts(counts, result, action):
    prefix = f"action{int(action)}"
    counts[prefix + "/total_transitions"] += 1
    category = result["category"]
    counts[prefix + "/category/" + category] += 1
    if category != "continuous_visual": return
    counts[prefix + "/continuous"] += 1
    maximum = result["max_delta"]
    counts[prefix + "/max_delta/" + (str(maximum) if maximum <= 3 else ">3")] += 1
    counts[prefix + "/local3"] += int(result["local3"])
    counts[prefix + "/grand_delta"] += int(not result["local3"])
    counts[prefix + "/immobile"] += int(result["immobile"])
    counts[prefix + "/nonzero_origin"] += int(maximum > 0)
    counts[prefix + "/shape_change"] += int(not result["shape_equal"])
    counts[prefix + "/old_at_border"] += int(result["old_border_clearance"] == 0)
    counts[prefix + "/next_at_border"] += int(result["next_border_clearance"] == 0)
    counts[prefix + "/all_candidates_invalid"] += int(result["all_translation_candidates_invalid"])
    for name in ("old_supported", "next_supported"):
        counts[prefix + "/" + name] += int(result[name])
    if result["local3"]:
        dy, dx = result["delta_origin_yx"]
        counts[prefix + f"/hist49/{dy},{dx}"] += 1


def delta_rates(counts):
    result = {}
    for action in range(8):
        prefix = f"action{action}"
        total, continuous = counts.get(prefix + "/total_transitions", 0), counts.get(prefix + "/continuous", 0)
        local = counts.get(prefix + "/local3", 0)
        result[prefix] = {"total": total, "continuous": continuous, "local3": local,
                          "local3_given_continuous": local / continuous if continuous else None,
                          "local3_of_all_transitions": local / total if total else None,
                          "border_given_continuous": counts.get(prefix + "/old_at_border", 0) / continuous if continuous else None,
                          "hist49": [[counts.get(prefix + f"/hist49/{dy},{dx}", 0) for dx in range(-3, 4)] for dy in range(-3, 4)]}
    return result


def input_history(parent_actions, t, anchor=None, branch_actions=None):
    if t < 0 or (anchor is not None and anchor < 1): raise ValueError("negative timing or missing real anchor predecessor")
    if anchor is None:
        if t > len(parent_actions): raise ValueError("parent timing out of bounds")
        history = parent_actions[max(0, t - 8):t]
    else:
        if anchor > len(parent_actions) or branch_actions is None or t > len(branch_actions): raise ValueError("branch timing out of bounds")
        history = np.concatenate((parent_actions[max(0, anchor - 8):anchor], branch_actions[:t]))[-8:]
    return [int(action) for action in history]


def latency_evidence(delta, action, history):
    if action not in (0, 3) or delta["category"] != "continuous_visual" or delta["delta_origin_yx"][1] == 0:
        return None
    history = history[-8:]
    dx = int(delta["delta_origin_yx"][1])
    previous = next(((lag, value) for lag, value in enumerate(reversed(history), 1) if value in (1, 2)), None)
    lag, horizontal = previous if previous else (None, None)
    matching = (dx < 0 and horizontal == 1) or (dx > 0 and horizontal == 2)
    return {"history8": history, "previous_action": history[-1] if history else None,
            "last_horizontal_action": horizontal, "last_horizontal_lag": lag,
            "signed_dx": dx, "amplitude": abs(dx), "same_sense": matching if previous else None,
            "shape_equal": delta["shape_equal"], "old_supported": delta["old_supported"], "next_supported": delta["next_supported"]}


def paired_latency(noop, down):
    if noop is None or down is None: return "unprovable"
    return "equal_signed_dx" if noop["signed_dx"] == down["signed_dx"] else "mismatched_signed_dx"


def analyze(model, current, target, labels):
    current, target = current[None].long(), target[None].long()
    valid = torch.ones(1, dtype=torch.bool)
    labels = {key: torch.as_tensor(value)[None].long() for key, value in labels.items()}
    cp, cm = label_mask(labels, 0, valid); np_, nm = label_mask(labels, 1, valid)
    ck = cp | visual_absence(current, labels["evidence"][:, 0], valid, model.object_tiles)
    nk = np_ | visual_absence(target, labels["evidence"][:, 1], valid, model.object_tiles)
    mode, known = composition_targets(model, current, target, cm, nm, ck, nk)
    before = current.masked_fill(cm, model.blank_id)
    after = target.masked_fill(nm, model.blank_id)
    copy_image = before.clone(); copy_image[nm] = target[nm]
    compose_image = current.clone(); compose_image[nm] = target[nm]
    board = (slice(None), slice(None), slice(2, 12))
    copy_ok = bool(torch.equal(copy_image[board], target[board])) if bool(nk) else None
    compose_ok = bool(torch.equal(compose_image[board], target[board])) if bool(nk) else None
    oldcells = labels["cells"][0, 0].numpy()
    newcells = labels["cells"][0, 1].numpy()
    b0 = before[0, :, 2:12].numpy()
    old_supported = supported(b0, oldcells, model.blank_id) if bool(cp) else None
    next_contact = supported(b0, newcells, model.blank_id) if bool(np_) else None
    collision = bool((nm & (before != model.blank_id))[board].any()) if bool(np_) and bool(ck) else None
    texture_errors = 0
    for index, positive, tokens in ((0, cp, current), (1, np_, target)):
        if bool(positive):
            cells = labels["cells"][0, index]
            kind = labels["shape_id"][0, index]
            texture_errors += int(not bool(torch.all(tokens[0, cells[:, 0], cells[:, 1] + 2] == model.object_tiles[kind])))
    changed = current != target
    ambiguous_static = False
    # A static labelled group plus a different independently moving component is only a warning.
    if bool(cp & np_) and np.array_equal(oldcells, newcells) and bool(changed[board].any()):
        def texture_grid(tokens):
            pixels = model.codebook[tokens[0]].reshape(18, 20, 4, 4).permute(0, 2, 1, 3).reshape(72, 80).numpy()
            return classify_board(pixels)
        grid0, grid1 = texture_grid(current), texture_grid(target)
        for texture0, cells0 in candidates(grid0):
            if set(cells0) == set(map(tuple, oldcells)): continue
            for texture1, cells1 in candidates(grid1):
                if texture0 == texture1 and cells0 != cells1 and np.array_equal(without(grid0, cells0), without(grid1, cells1)):
                    ambiguous_static = True
    track0, track1 = int(labels["track"][0, 0]), int(labels["track"][0, 1])
    result = {"current_positive": bool(cp), "next_positive": bool(np_), "current_known": bool(ck), "next_known": bool(nk),
              "mode_known": bool(known), "mode": int(mode) if bool(known) else None,
              "forced_rewrite_next_unknown": bool(known & ~nk), "copy_feasible": copy_ok, "compose_feasible": compose_ok,
              "background_equal": bool(torch.equal(before[board], after[board])) if bool(ck & nk) else None,
              "old_supported": old_supported, "next_contact": next_contact, "next_collision": collision,
              "label_texture_errors": texture_errors, "track_changed": track0 != track1 if bool(cp & np_) else None,
              "same_track_background_changed": bool(cp & np_) and track0 == track1 and not torch.equal(before[board], after[board]),
              "static_label_other_moving_group_ambiguous": ambiguous_static,
              "current_cells": oldcells.tolist() if bool(cp) else None, "next_cells": newcells.tolist() if bool(np_) else None,
              "track": [track0, track1], "evidence": labels["evidence"][0].tolist(),
              "propagation": labels.get("propagation", torch.zeros(1, 2, dtype=torch.long))[0].tolist()}
    result["delta"] = delta_evidence(current[0, :, 2:12].numpy(), target[0, :, 2:12].numpy(), oldcells, newcells,
                                      int(model.object_tiles[labels["shape_id"][0, 0]]) if bool(cp) else -1,
                                      int(model.object_tiles[labels["shape_id"][0, 1]]) if bool(np_) else -1,
                                      model.blank_id, bool(cp & np_))
    categories = []
    if texture_errors: categories.append("label_texture_mismatch")
    if bool(known) and int(mode) == 0 and copy_ok is False: categories.append("copy_target_not_renderable")
    if bool(known) and int(mode) == 1 and compose_ok is False: categories.append("compose_target_not_renderable")
    if bool(known) and int(mode) == 2 and (copy_ok is True or compose_ok is True): categories.append("rewrite_when_simple_render_possible")
    if collision: categories.append("next_overlay_collision")
    if bool(known) and int(mode) == 1 and old_supported is False: categories.append("compose_without_visible_support")
    if result["same_track_background_changed"]: categories.append("same_track_background_changed")
    if ambiguous_static: categories.append("static_heap_identity_ambiguous")
    if result["forced_rewrite_next_unknown"]: categories.append("forced_rewrite_unknown_next")
    return result, categories


def add_counts(counts, result):
    counts["transitions"] += 1
    for key in ("current_positive", "next_positive", "current_known", "next_known", "mode_known", "forced_rewrite_next_unknown",
                "copy_feasible", "compose_feasible", "background_equal", "old_supported", "next_contact", "next_collision",
                "track_changed", "same_track_background_changed", "static_label_other_moving_group_ambiguous"):
        if result[key] is not None: counts[key + "/known"] += 1; counts[key + "/true"] += int(result[key])
    counts["label_texture_errors"] += result["label_texture_errors"]
    counts["mode/" + (str(result["mode"]) if result["mode_known"] else "unknown")] += 1
    for name in ("old_supported", "next_contact"):
        value = result[name]
        group = f"{name}/{value}"
        counts[group + "/total"] += 1; counts[group + "/mode_known"] += int(result["mode_known"])
    for propagation in result["propagation"]: counts[f"propagation/{propagation}"] += 1


def rates(counts):
    result = {}
    for key, count in counts.items():
        if key.endswith("/known"):
            stem = key.removesuffix("/known")
            result[stem + "/rate"] = counts.get(stem + "/true", 0) / count if count else None
        elif key.endswith("/total"):
            stem = key.removesuffix("/total")
            result[stem + "/known_mode_rate"] = counts.get(stem + "/mode_known", 0) / count if count else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/states-best-v18.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/object-annotations-v2")
    parser.add_argument("--interventions", type=Path, default=ROOT / "data/action-interventions-v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(2)
    manifest_bytes = (args.data / "manifest.json").read_bytes(); annotation_bytes = (args.annotations / "object-audit.json").read_bytes()
    manifest, annotation_report = json.loads(manifest_bytes), json.loads(annotation_bytes)
    digest, annotation_digest = hashlib.sha256(manifest_bytes).hexdigest(), hashlib.sha256(annotation_bytes).hexdigest()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (manifest["status"] != "complete" or annotation_report["test_used"] or annotation_report["manifest_sha256"] != digest
            or checkpoint["config"]["manifest_sha256"] != digest or checkpoint["config"]["annotations_sha256"] != annotation_digest):
        raise ValueError("audit provenance mismatch")
    model = load_tile_model(args.checkpoint)
    def forbidden(*args, **kwargs): raise RuntimeError("model inference forbidden in label audit")
    model.forward = model.advance = model.scene_details = forbidden
    records = {entry["file"]: entry for entry in annotation_report["episodes"]}
    episodes, annotations, sources = {"train": [], "validation": []}, {"train": [], "validation": []}, {"train": [], "validation": []}
    for entry in manifest["episodes"]:
        if entry["split"] not in episodes: continue  # Never open a reserved-test NPZ.
        source = (args.data / entry["file"]).resolve(); record = records[entry["file"]]
        label_path = (args.annotations / record["labels_file"]).resolve()
        if (not source.is_relative_to(args.data.resolve()) or not label_path.is_relative_to(args.annotations.resolve())
                or record["split"] != entry["split"] or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]):
            raise ValueError("parent capture/annotation path or split mismatch")
        frames, actions, metadata = read_episode(source)
        if metadata["source"] != "capture": raise ValueError("real capture required")
        tokens, unknown = encode_frames(frames, model.codebook.numpy())
        if unknown: raise ValueError("frozen codec unknown patterns")
        with np.load(label_path, allow_pickle=False) as archive:
            if json.loads(str(archive["metadata"]))["source_sha256"] != entry["sha256"]: raise ValueError("label source mismatch")
            labels = {key: archive[key].copy() for key in ("cells", "shape_id", "evidence", "track", "propagation")}
        split = entry["split"]
        episodes[split].append((tokens.astype(np.uint8), actions)); annotations[split].append(labels); sources[split].append(entry)
    tables, addon_digest = load_interventions(args.interventions, model, manifest, episodes, annotations, digest, annotation_digest)
    args.output.mkdir(parents=True)
    counts, delta_counts, latency_counts, examples, example_counts, latency_rows = {}, {}, {}, [], Counter(), []
    def inspect(split, corpus, codes, actions, labels, provenance, parent_actions, anchor=None):
        counter = counts.setdefault(split + "/" + corpus, Counter())
        deltas = delta_counts.setdefault(split + "/" + corpus, Counter())
        for t in range(len(actions)):
            pair = {key: values[t:t + 2] for key, values in labels.items()}
            result, categories = analyze(model, torch.from_numpy(codes[t].copy()), torch.from_numpy(codes[t + 1].copy()), pair)
            add_counts(counter, result)
            add_delta_counts(deltas, result["delta"], actions[t])
            history = input_history(parent_actions, t, anchor, actions if anchor is not None else None)
            latency = latency_evidence(result["delta"], int(actions[t]), history)
            if latency is not None:
                rows = latency_counts.setdefault(split + "/" + corpus, Counter())
                key = f"button{int(actions[t])}/lag{latency['last_horizontal_lag'] or 'none'}/sense{latency['same_sense']}/amp{latency['amplitude']}/shape_equal{latency['shape_equal']}/old_support{latency['old_supported']}/next_support{latency['next_supported']}"
                rows[key] += 1
                latency_rows.append({"split": split, "corpus": corpus, **provenance, "step": t, "current_action": int(actions[t]), **latency})
                result["input_latency"] = latency
                categories = ["horizontal_under_noop_down"] + categories
            for category in categories:
                counter["category/" + category] += 1
                if len(examples) >= 24 or example_counts[category] >= 3: continue
                filename = f"example-{len(examples):02d}.png"
                image = Image.new("RGB", (160, 160), "white"); draw = ImageDraw.Draw(image)
                draw.text((2, 1), "before             after", fill="black")
                for column, code in enumerate((codes[t], codes[t + 1])):
                    pixels = model.codebook[torch.from_numpy(code.astype(np.int64))].reshape(18, 20, 4, 4).permute(0, 2, 1, 3).reshape(72, 80).numpy()
                    board = Image.fromarray((pixels[:, 8:48] * 85).astype(np.uint8)).resize((80, 144), Image.Resampling.NEAREST)
                    image.paste(board.convert("RGB"), (column * 80, 16))
                image.save(args.output / filename)
                examples.append({"category": category, "split": split, "corpus": corpus, **provenance,
                                 "step": t, "action": int(actions[t]), "analysis": result, "png": filename})
                example_counts[category] += 1
    for split in episodes:
        for index, ((tokens, actions), labels) in enumerate(zip(episodes[split], annotations[split], strict=True)):
            inspect(split, "parents", tokens, actions, labels, {"file": sources[split][index]["file"], "sha256": sources[split][index]["sha256"]}, actions)
        for (episode, anchor), bundle in tables[split].items():
            for arm in range(6):
                inspect(split, "interventions", bundle["tokens"][arm], bundle["actions"][arm],
                        {key: value[arm] for key, value in bundle["labels"].items()},
                        {"parent_file": sources[split][episode]["file"], "parent_sha256": sources[split][episode]["sha256"], "anchor": anchor, "arm": arm,
                         "parent_action_before_anchor": int(episodes[split][episode][1][anchor - 1])}, episodes[split][episode][1], anchor)
    matched_train = [row for row in latency_rows if row["split"] == "train" and row["same_sense"] is True]
    window = max((row["last_horizontal_lag"] for row in matched_train), default=None)
    amplitudes = max((row["amplitude"] for row in matched_train), default=None)
    val_outside = [row for row in latency_rows if row["split"] == "validation" and
                   (row["same_sense"] is not True or window is None or row["last_horizontal_lag"] is None
                    or row["last_horizontal_lag"] > window or row["amplitude"] > amplitudes)]
    pairs = Counter()
    by_anchor = {}
    for row in latency_rows:
        if row["corpus"] == "interventions" and row["arm"] in (0, 3):
            by_anchor.setdefault((row["split"], row["parent_file"], row["anchor"], row["step"]), {})[row["arm"]] = row
    for split in tables:
        for (episode, anchor) in tables[split]:
            for t in range(8):
                row = by_anchor.get((split, sources[split][episode]["file"], anchor, t), {})
                pairs[split + "/" + paired_latency(row.get(0), row.get(3))] += 1
    report = {"version": 3, "delta_audit_version": 1, "latency_audit_version": 1, "status": "hypothesis-audit-not-groundtruth-internal-state", "test_used": False, "model_forward_used": False,
              "manifest_sha256": digest, "annotations_sha256": annotation_digest, "interventions_sha256": addon_digest,
              "codec_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "counts": {key: dict(value) for key, value in counts.items()}, "examples": examples,
              "rates": {key: rates(value) for key, value in counts.items()},
              "delta_counts": {key: dict(value) for key, value in delta_counts.items()},
              "delta_rates": {key: delta_rates(value) for key, value in delta_counts.items()},
              "input_latency": {"counts": {key: dict(value) for key, value in latency_counts.items()},
                                "cases": latency_rows, "paired_noop_down": dict(pairs),
                                "train_derived_max_matching_lag": window, "train_derived_max_matching_amplitude": amplitudes,
                                "train_unexplained_count": sum(row["split"] == "train" and row["same_sense"] is not True for row in latency_rows),
                                "validation_outside_train_support_count": len(val_outside), "validation_outside_train_support": val_outside,
                                "interpretation": "qualitative train-derived lag bound checked on validation, not authorization to change an oracle; same anchor/time paired even if actual branch histories differ"},
              "delta_definition": "both objects positively proved; same texture and exact background equality; deposit/style/rewrite/unknown separate; origin-bbox deltas; centres integer quarter-cell units; track not used as identity truth; generic translations use bounds only",
              "limits": "support/collision and static heap warnings are visible-geometry evidence, never internal active/locked truth; forced rewrite uses existing permissive <=4 single-texture bound; no exhaustive overlay search"}
    (args.output / "label-consistency.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output), "transitions": sum(value["transitions"] for value in counts.values()), "examples": len(examples)}))


if __name__ == "__main__": main()
