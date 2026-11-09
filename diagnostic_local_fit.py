"""One real train-bundle memorization diagnostic; never exports fitted weights."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit_controls import screenshot
from intervention_training import (BUTTONS, branch_objective, first_effect_joint_loss, rotation_pair,
                                   temporal_weights, tensors, verify_extension)
from run import ROOT, read_episode
from token_model import decode_tokens, encode_frames, load_tile_model
from train_scene import composition_targets, label_mask, scene_loss, scene_optimizer, visual_absence


def select_bundle(bundles, model):
    for key, bundle in sorted(bundles.items()):
        frames, _, labels, _ = tensors(model, bundle, None)
        positive = (labels["shape_id"] >= 0) & ((labels["evidence"] == 1) | (labels["evidence"] == 2))
        if not bool(positive.all()): continue
        first = temporal_weights(frames).argmax(1)
        if any(rotation_pair(model, frames, labels, int(first[arm]) + 1, arm) != "effect" for arm in (4, 5)): continue
        if not bool((labels["cells"][:, 1:] != labels["cells"][:, :-1]).any()): continue
        return key, bundle
    raise ValueError("no train bundle meets all criteria:6x9 proven objects, qualified A/B effects and visible motion; no relaxation")


def qualified_bundle(directory, record, book):
    path = (directory / record["file"]).resolve()
    if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("train intervention digest/path mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) - {"frames", "actions", "metadata", "cells", "shape_id", "evidence", "texture", "track", "propagation"}:
            raise ValueError("unexpected/internal-state addon fields")
        metadata = json.loads(str(archive["metadata"]))
        for key in ("parent_file", "parent_sha256", "anchor", "split", "witness_verified", "snapshot_persisted", "source"):
            if metadata[key] != record[key]: raise ValueError("addon metadata differs")
        if metadata["split"] != "train" or not metadata["witness_verified"] or metadata["snapshot_persisted"] or metadata["source"] != "capture-intervention":
            raise ValueError("not a witnessed train capture")
        frames, actions = archive["frames"], archive["actions"].copy()
        labels = {key: archive[key].copy() for key in ("cells", "shape_id", "evidence", "texture", "track")}
        if frames.shape != (6, 9, 72, 80) or frames.dtype != np.uint8 or actions.shape != (6, 8): raise ValueError("invalid addon shape")
        tokens, unknown = encode_frames(frames.reshape(-1, 72, 80), book)
        if unknown: raise ValueError("unknown frozen visual tiles")
    return {"tokens": tokens.reshape(6, 9, 18, 20).astype(np.uint8), "actions": actions, "labels": labels}


def fixed_context(model, tokens, actions, anchor):
    context = None
    with torch.no_grad():
        for t in range(anchor): context = model.advance(torch.from_numpy(tokens[t].copy())[None].long(), torch.tensor([int(actions[t])]), context)
    return context.detach().clone()


def first_failing(candidates, evaluate, limit=64):
    """Explicit train-only hard-example scan; no validation or target modification."""
    examined = []
    for candidate in candidates:
        if len(examined) >= limit: break
        context, teacher, autonomous = evaluate(candidate)
        errors = {"teacher": sum(not row["board_exact"] for row in teacher),
                  "autonomous": sum(not row["board_exact"] for row in autonomous)}
        examined.append({"parent_file": candidate["record"]["parent_file"], "anchor": candidate["record"]["anchor"],
                         "qualified": True, "baseline_board_errors": errors})
        if errors["teacher"] or errors["autonomous"]:
            return candidate, context, examined
    raise ValueError(f"no failing qualified train bundle among{len(examined)} candidates (cap{limit}); no fit run; examined={examined}")


def blocks(model, bundle, context, loss_fn):
    frames, actions, labels, state = tensors(model, bundle, context)
    weights = temporal_weights(frames); first = weights.argmax(1)
    scene, pixel, auxiliary, pairs = frames.new_zeros((), dtype=torch.float32), frames.new_zeros((), dtype=torch.float32), frames.new_zeros((), dtype=torch.float32), 0
    valid = torch.ones(1, dtype=torch.bool)
    for t in range(8):
        details = model.scene_details(frames[:, t], actions[:, t], state)
        for arm in range(6):
            sliced = {key: {k: v[arm:arm + 1] for k, v in value.items()} if isinstance(value, dict) else value[arm:arm + 1] for key, value in details.items()}
            lab = {key: value[arm:arm + 1] for key, value in labels.items()}
            current, target = frames[arm:arm + 1, t], frames[arm:arm + 1, t + 1]
            scene = scene + weights[arm, t] * loss_fn(model, current, target, sliced, lab, t, valid, valid) / 6
            _, cm = label_mask(lab, t, valid); _, nm = label_mask(lab, t + 1, valid)
            w = (1 + 31 * ((current != target) | cm | nm)).float()
            pixel = pixel + weights[arm, t] * (F.nll_loss(model.decode_scene(current, sliced), target, reduction="none") * w).sum() / w.sum() / 6
        extra, count = first_effect_joint_loss(model, frames, labels, details, t, first)
        auxiliary = auxiliary + extra; pairs += count; state = details["state"]
    return {"scene": scene, "pixel": pixel, "scene_auxiliary": scene - pixel, "first_effect": auxiliary / max(1, pairs)}


def gradient_norms(model, loss):
    named = list(model.named_parameters())
    gradients = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
    values = {}
    for prefix in ("gates", "candidate", "current_head", "next_head", "mode_head", "readout", "change_head"):
        values[prefix] = sum(float(g.detach().square().sum()) for (name, _), g in zip(named, gradients) if g is not None and name.startswith(prefix)) ** .5
    return values


@torch.no_grad()
def metrics(model, bundle, context, autonomous):
    frames, actions, labels, state = tensors(model, bundle, context)
    current = frames[:, 0]; report, images = [], {button: [decode_tokens(current, model.codebook)[arm, 0].numpy()] for arm, button in enumerate(BUTTONS)}
    for t in range(8):
        if not autonomous: current = frames[:, t]
        details = model.scene_details(current, actions[:, t], state)
        prediction = model.decode_scene(current, details, hard=True); state = details["state"]
        valid = torch.ones(6, dtype=torch.bool)
        matches = (current == frames[:, t]).flatten(1).all(1)
        cp, cm = label_mask(labels, t, valid & matches); np_, nm = label_mask(labels, t + 1, valid)
        ck = cp | visual_absence(current, labels["evidence"][:, t], valid & matches, model.object_tiles)
        nk = np_ | visual_absence(frames[:, t + 1], labels["evidence"][:, t + 1], valid, model.object_tiles)
        modes, mode_known = composition_targets(model, current, frames[:, t + 1], cm, nm, ck, nk)
        for arm, button in enumerate(BUTTONS):
            row = {"button": button, "step": t, "input_matches_truth": bool(torch.equal(current[arm], frames[arm, t])),
                   "board_exact": bool(torch.equal(prediction[arm, :, 2:12], frames[arm, t + 1, :, 2:12])),
                   "board_token_error": float((prediction[arm, :, 2:12] != frames[arm, t + 1, :, 2:12]).float().mean()),
                   "mode": int(details["mode"][arm].argmax())}
            row["mode_target"] = int(modes[arm]) if bool(mode_known[arm]) else None
            row["mode_correct"] = row["mode"] == int(modes[arm]) if bool(mode_known[arm]) else None
            for head, index in (("current", t), ("next", t + 1)):
                known = int(labels["shape_id"][arm, index]) >= 0 and int(labels["evidence"][arm, index]) in (1, 2)
                if head == "current": known &= row["input_matches_truth"]
                origin = labels["cells"][arm, index].amin(0)
                wanted = int(labels["shape_id"][arm, index]) * 180 + int(origin[0]) * 10 + int(origin[1])
                selected = int(details[head]["joint"][arm].flatten().argmax())
                row[head] = {"known": known, "joint": selected == wanted if known else None,
                             "shape": selected // 180 == wanted // 180 if known else None,
                             "visible": bool(details[head]["visible"][arm, 0] > 0) if known else None}
            report.append(row)
            images[button].append(decode_tokens(prediction, model.codebook)[arm, 0].numpy())
        current = prediction
    return report, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/states-best-v18.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-failing-train", action="store_true")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(2)
    checkpoint_bytes = args.checkpoint.read_bytes(); checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = load_tile_model(args.checkpoint)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    directory = ROOT / checkpoint["config"]["interventions_path"]
    addon_bytes = (directory / "manifest.json").read_bytes(); addon = json.loads(addon_bytes)
    proof = verify_extension(directory)
    if addon["test_used"] or addon["status"] != "complete" or proof["to_sha256"] != checkpoint["config"]["interventions_sha256"]: raise ValueError("addon lineage mismatch")
    dictionary = sorted(addon["dictionary"], key=lambda item: item["id"])
    if not torch.equal(torch.tensor([item["cells"] for item in dictionary]), model.object_shapes): raise ValueError("frozen shape dictionary mismatch")
    parent_bytes = (ROOT / "data/manifest.json").read_bytes(); parent = json.loads(parent_bytes)
    anno_bytes = (ROOT / "data/object-annotations-v2/object-audit.json").read_bytes()
    if hashlib.sha256(parent_bytes).hexdigest() != checkpoint["config"]["manifest_sha256"] or hashlib.sha256(anno_bytes).hexdigest() != checkpoint["config"]["annotations_sha256"]: raise ValueError("parent/annotation mismatch")
    counts = checkpoint["config"]["mode_training_coverage"]["interventions"]
    if [counts[k] for k in ("copy", "compose", "rewrite")] != [67973, 281, 525]: raise ValueError("unexpected V2 mode coverage")
    train_entries = [entry for entry in parent["episodes"] if entry["split"] == "train"]
    parents = {entry["file"]: (index, entry) for index, entry in enumerate(train_entries)}
    def candidates():
        for record in sorted((r for r in addon["episodes"] if r["split"] == "train"), key=lambda r: (parents[r["parent_file"]][0], r["anchor"])):
            bundle = qualified_bundle(directory, record, model.codebook.numpy())
            try: select_bundle({(parents[record["parent_file"]][0], record["anchor"]): bundle}, model)
            except ValueError: continue
            yield {"record": record, "bundle": bundle}
    model.eval()
    def evaluate_candidate(candidate):
        record, bundle = candidate["record"], candidate["bundle"]
        entry = parents[record["parent_file"]][1]
        path = (ROOT / "data" / entry["file"]).resolve()
        if not path.is_relative_to((ROOT / "data").resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"] or record["parent_sha256"] != entry["sha256"]:
            raise ValueError("candidate parent hash/path mismatch")
        pixels, commands, metadata = read_episode(path)
        tokens, unknown = encode_frames(pixels[:record["anchor"] + 1], model.codebook.numpy())
        if unknown or metadata["source"] != "capture" or not np.all(bundle["tokens"][:, 0] == tokens[-1]): raise ValueError("candidate causal seed mismatch")
        context = fixed_context(model, tokens, commands, record["anchor"])
        with torch.no_grad():
            teacher, _ = metrics(model, bundle, context, False)
            autonomous, _ = metrics(model, bundle, context, True)
        return context, teacher, autonomous
    examined = []
    chosen_context = None
    if args.first_failing_train:
        candidate, chosen_context, examined = first_failing(candidates(), evaluate_candidate)
    else:
        candidate = next(candidates(), None)
        if candidate is None: raise ValueError("no qualifying train bundle; no criteria relaxed")
    selected, bundle = candidate["record"], candidate["bundle"]
    entry = parents[selected["parent_file"]][1]; source = (ROOT / "data" / entry["file"]).resolve()
    if not source.is_relative_to((ROOT / "data").resolve()) or hashlib.sha256(source.read_bytes()).hexdigest() != selected["parent_sha256"] or entry["sha256"] != selected["parent_sha256"]: raise ValueError("train parent capture mismatch")
    pixels, commands, metadata = read_episode(source)
    tokens, unknown = encode_frames(pixels[:selected["anchor"] + 1], model.codebook.numpy())
    if unknown or metadata["source"] != "capture" or not np.all(bundle["tokens"][:, 0] == tokens[-1]): raise ValueError("causal seed mismatch")
    annotation_record = next(r for r in json.loads(anno_bytes)["episodes"] if r["file"] == selected["parent_file"])
    labels_path = (ROOT / "data/object-annotations-v2" / annotation_record["labels_file"]).resolve()
    if annotation_record["split"] != "train" or not labels_path.is_relative_to((ROOT / "data/object-annotations-v2").resolve()): raise ValueError("parent label path/split mismatch")
    with np.load(labels_path, allow_pickle=False) as archive:
        if json.loads(str(archive["metadata"]))["source_sha256"] != entry["sha256"]: raise ValueError("parent label provenance mismatch")
        anchor = selected["anchor"]
        if not np.all(bundle["labels"]["cells"][:, 0] == archive["cells"][anchor]) or not np.all(bundle["labels"]["shape_id"][:, 0] == archive["shape_id"][anchor]): raise ValueError("seed proof differs from parent labels")
    context = chosen_context if chosen_context is not None else fixed_context(model, tokens, commands, selected["anchor"])
    context_copy = context.clone()
    loss_fn = partial(scene_loss, mode_weights=checkpoint["config"]["mode_class_weights"]["interventions"])
    optimizer = scene_optimizer(model); optimizer.load_state_dict(checkpoint["optimizer"])
    if [g["lr"] for g in optimizer.param_groups] != [3e-4, 3e-3]: raise ValueError("expected original two-group best18 learning rates")
    args.output.mkdir(parents=True)
    report = {"status": "local-memorization-diagnostic-not-final-training", "validation_used": False, "test_used": False,
              "runtime_adopted": False, "fitted_weights_exported": False, "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
              "checkpoint_config_sha256": hashlib.sha256(json.dumps(checkpoint["config"], sort_keys=True).encode()).hexdigest(),
              "interventions_sha256": proof["to_sha256"], "selected_record": selected,
              "mode_weights": checkpoint["config"]["mode_class_weights"]["interventions"], "lr": [g["lr"] for g in optimizer.param_groups],
              "selection": "explicit-first-failing-train-model18-eval-no-grad" if args.first_failing_train else "first-qualified-train-no-prediction-selection",
              "candidate_cap": 64, "examined_candidates": examined,
              "scope": "one fixed detached base18 causal state; teacher8 branch objective only; no parent32 loss; no prefix retraining",
              "parent_capture_read": str(source), "parent_labels_read": str(labels_path),
              "parent_labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
              "limitations": "memorization success is not global capacity or playability; fixed initial hidden may become off-distribution after updates;500 failures do not alone prove a bug",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "measurements": []}
    for update in range(501):
        if update in (0, 100, 500):
            model.eval(); components = blocks(model, bundle, context, loss_fn)
            total = components["scene"] + components["first_effect"]
            actual = branch_objective(model, bundle, context, loss_fn)
            torch.testing.assert_close(total, actual)
            gradients = gradient_norms(model, total)
            teacher, _ = metrics(model, bundle, context, False)
            autonomous, images = metrics(model, bundle, context, True)
            screenshot(images, args.output / f"update-{update:03d}.png")
            np.savez_compressed(args.output / f"samples-{update:03d}.npz", frames=np.stack([np.stack(images[b]) for b in BUTTONS]))
            report["measurements"].append({"update": update, "components": {k: float(v.detach()) for k, v in components.items()},
                                            "objective_exact_match": True, "grad_norms": gradients, "teacher": teacher, "autonomous": autonomous})
            del components, total, actual
        if update == 500: break
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = branch_objective(model, bundle, context, loss_fn)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite local objective")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True); optimizer.step()
    if not torch.equal(context, context_copy): raise RuntimeError("fixed context mutated")
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]: raise RuntimeError("base file modified")
    report["initial_weights_changed_in_memory"] = any(not torch.equal(original[k], v) for k, v in model.state_dict().items())
    report["fixed_context_unchanged"] = True
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output), "status": report["status"]}))


if __name__ == "__main__": main()
