"""Fit A/B orientation proposals on witnessed ROM interventions."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from intervention_training import BUTTONS, load_interventions
from model import ACTIONS
from scene_model import OrientedProposalSceneDynamics, scene_from_weights
from train_mode_correction import proven_composition
from train_relative_proposal import batch
from train_spawn_preview import sha, training_captures


def rotation_examples(model, bundle):
    records = []
    for name in ("noop", "a", "b"):
        arm = BUTTONS.index(name)
        frames = torch.from_numpy(bundle["tokens"][arm]).long()
        actions = torch.from_numpy(bundle["actions"][arm]).long()
        labels = {key: torch.from_numpy(value[arm]).long() for key, value in bundle["labels"].items()}
        modes, known, old, _, new = proven_composition(model, frames, labels)
        kinds = labels["shape_id"]
        y, x = labels["cells"].amin(1).unbind(1)
        pose = kinds * 180 + y * 10 + x
        qualified = known & old & new
        effect = qualified & (modes == 0) & (kinds[1:] != kinds[0])
        records.append((frames, actions, modes, qualified, pose, effect))
    noop = records[0][4][1:] // 180
    first = []
    for record in records[1:]:
        difference = record[5] & records[0][3] & (record[4][1:] // 180 != noop)
        hits = difference.nonzero().flatten()
        first.append(int(hits[0]) if len(hits) else None)
    return records, first


@torch.no_grad()
def collect(model, parents, bundles):
    samples, arms, effects = [], [], []

    def record(details, action, pose, mode, target, generated, arm, effect):
        if not (details["contact_visible"][0] and details["next"]["visible"][0, 0] > 0):
            return
        if int(details["current"]["joint"].flatten().argmax()) != int(pose):
            return
        features = model.proposal_features(details, action, pose[None])[0]
        samples.append((features, details["event_base_mode"][0], details["event_base_joint"][0].flatten(),
                        details["state"][0, 16:20].clone(), pose, details["contact_logit"][0],
                        details["previous_action"][0], action[0], mode, target, generated))
        arms.append(arm)
        effects.append(effect)

    grouped = defaultdict(list)
    for episode, anchor in sorted(bundles):
        grouped[episode].append(anchor)
    used = 0
    for episode, (_, frames, actions, _) in enumerate(parents):
        anchors = set(grouped[episode][episode % 2::2])
        if not anchors:
            continue
        state = None
        for t in range(max(anchors) + 1):
            if t in anchors:
                branches, first = rotation_examples(model, bundles[episode, t])
                for arm, (images, commands, modes, known, pose, _) in enumerate(branches):
                    selected = {0, 1}
                    relevant = first if arm == 0 else (first[arm - 1],)
                    selected.update(step for step in relevant if step is not None)
                    teacher = auto_state = state
                    auto = images[:1]
                    for step in range(max(selected) + 1):
                        frame, action = images[step:step + 1], commands[step:step + 1]
                        details = model.scene_details(frame, action, teacher)
                        teacher = details["state"]
                        effect = arm != 0 and first[arm - 1] == step
                        if step in selected and known[step]:
                            record(details, action, pose[step], modes[step], pose[step + 1], False, arm, effect)
                        if auto is not None:
                            if not torch.equal(auto[:, :, 2:12], frame[:, :, 2:12]):
                                auto = None
                                continue
                            rollout = details if step == 0 else model.scene_details(auto, action, auto_state)
                            if step and step in selected and known[step]:
                                record(rollout, action, pose[step], modes[step], pose[step + 1], True, arm, effect)
                            auto = model.decode_scene(auto, rollout, hard=True)
                            auto_state = rollout["state"]
                used += 1
            frame, action = frames[t:t + 1], actions[t:t + 1]
            if t + 1 in anchors or t + 2 in anchors:
                state = model.advance_observed(frame, action, state)
            else:
                state = model.advance(frame, action, state)
    if not samples:
        raise ValueError("no verified rotation inputs")
    *columns, generated = zip(*samples)
    return (*map(torch.stack, columns), torch.tensor(generated)), torch.tensor(arms), torch.tensor(effects), used


@torch.no_grad()
def metrics(model, cache, arms, effects):
    results = []
    for part in torch.arange(len(arms)).split(64):
        results.append(batch(model, cache, part, True)[2])
    exact = torch.cat(results)
    return {name: {"examples": int(rows.sum()), "exact": int((rows & exact).sum()),
                   "first_effects": int((rows & effects).sum()), "first_effect_exact": int((rows & effects & exact).sum())}
            for arm, name in enumerate(("noop", "a", "b")) if (rows := arms == arm).any()}


@torch.no_grad()
def fit_b_gate(model, cache, effects):
    previous, action = cache[6:8]
    b, noop = ACTIONS.index("b"), ACTIONS.index("noop")
    actions = len(ACTIONS)
    weight = model.oriented_command_gate.weight
    frozen = weight.clone()
    contexts = (("press", action == b), ("release", (previous == b) & (action == noop)))
    result = {}
    edited = torch.zeros(len(weight), dtype=torch.bool)
    for name, included in contexts:
        indices = included.nonzero().flatten()
        if not len(indices):
            raise ValueError(f"missing proven B {name} examples")
        rows = (previous[indices] * actions + action[indices]).unique()
        edited[rows] = True
        original = weight[rows].clone()
        parts = indices.split(64)
        absolute = torch.cat([batch(model, cache, part, False, shape_only=True)[2] for part in parts])
        local = torch.cat([batch(model, cache, part, True, force_local=True, shape_only=True)[2] for part in parts])
        relative_only = local & ~absolute
        absolute_only = absolute & ~local
        effective = effects[indices]
        importance = torch.ones(len(indices))
        for priority in (relative_only, absolute_only, effective):
            if priority.any():
                importance[priority] += len(indices) / (3 * int(priority.sum()))
        scores = []
        for bias in (-4., 0., 2., 4., 6., 8., 10., 12., 16., 20., 24.):
            weight[rows] = original + bias
            losses = []
            for part in parts:
                orientation = batch(model, cache, part, True, shape_only=True)[0]
                pose = batch(model, cache, part, True)[0]
                losses.append(.7 * orientation + .3 * pose)
            objective = (torch.cat(losses) * importance).sum() / importance.sum() + .00005 * bias ** 2
            scores.append((float(objective), abs(bias), bias))
        _, _, bias = min(scores)
        weight[rows] = original + bias
        result[name] = {"bias": bias, "examples": len(indices), "first_effects": int(effective.sum()),
                        "relative_only": int(relative_only.sum()), "absolute_only": int(absolute_only.sum()),
                        "command_rows": rows.tolist()}
    torch.testing.assert_close(weight[~edited], frozen[~edited], rtol=0, atol=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interventions", type=Path, default=Path("data/action-interventions-v3"))
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(64)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = scene_from_weights(checkpoint).eval()
    if type(model) is not OrientedProposalSceneDynamics:
        raise ValueError("rotation fitting requires the oriented proposal checkpoint")
    parents = list(training_captures(checkpoint["config"], model.codebook.numpy()))
    manifest_path = Path("data/manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    names = [entry["file"] for entry in manifest["episodes"] if entry["split"] == "train"]
    if names != [name for name, _, _, _ in parents]:
        raise ValueError("training parent order differs")
    episodes = {"train": [(frames.numpy(), actions.numpy()) for _, frames, actions, _ in parents]}
    annotations = {"train": [{key: value.numpy() for key, value in labels.items()} for _, _, _, labels in parents]}
    tables, digest = load_interventions(args.interventions, model, manifest, episodes, annotations,
                                        sha(manifest_path), checkpoint["config"]["annotations_sha256"], splits=("train",))
    cache, arms, effects, used = collect(model, parents, tables["train"])
    counts = torch.bincount(arms * 2 + effects.long(), minlength=6)
    if not counts[[0, 2, 3, 4, 5]].all():
        raise ValueError("training needs A/B effects and no-effect controls")
    before = metrics(model, cache, arms, effects)
    if args.gate_only:
        trainable = ("oriented_command_gate.",)
        calibration = fit_b_gate(model, cache, effects)
    else:
        weights = counts[arms * 2 + effects.long()].float().rsqrt()
        trainable = ("oriented_proposal.", "proposal_shape_embedding.", "oriented_command_gate.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable))
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=.003)
        for _ in range(6):
            for indices in torch.randperm(len(arms)).split(64):
                loss = batch(model, cache, indices, True, supervise_local=True)[0]
                loss = (loss * weights[indices]).sum() / weights[indices].sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
        calibration = None
    state = model.state_dict()
    for key, value in checkpoint["model"].items():
        if not key.startswith(trainable):
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "interventions_sha256": digest, "bundles": used,
              "gate_calibration": calibration, "before": before, "after": metrics(model, cache, arms, effects)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "rotation_proposal_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
