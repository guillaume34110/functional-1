"""One frozen-core conditional-pose probe; diagnostic only, no runtime adoption."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from audit_mode_prior import ModePrior, checkpoint_comparison, long_paths
from intervention_training import rotation_pair, temporal_weights, tensors
from probe_identity_prior import verify_seed_prefixes
from run import ROOT
from token_model import load_tile_model
from train_scene import load_action_interventions, load_scene_data, rendered_teacher
from train_tokens import measure


class ActionResidual(nn.Module):
    def __init__(self, kinds):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(8, kinds, 16))
        self.bias = nn.Parameter(torch.zeros(8, kinds))

    def forward(self, hidden, actions):
        return torch.einsum("bkc,bcxy->bkxy", self.weight[actions], hidden) + self.bias[actions, :, None, None]


def residual_head(kinds, action_conditioned=False):
    if action_conditioned:
        return ActionResidual(kinds)
    layer = nn.Conv2d(16, kinds, 1)
    nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
    return layer


def future_logits(base, details):
    hidden = details["state"][:, :16, :, 2:12]
    pose = base.next_head.pose(hidden).flatten(1)
    pool = torch.einsum("bcp,bp->bc", hidden.flatten(2), pose.softmax(1))
    return hidden.detach(), pose.detach(), base.next_head.shape(pool).detach()


def conditional_joint(hidden, pose, shape, residual, valid, actions=None):
    if isinstance(residual, ActionResidual):
        if actions is None or actions.shape != (len(hidden),) or actions.dtype != torch.int64 or torch.any((actions < 0) | (actions >= 8)):
            raise ValueError("action residual requires actual button IDs[B] in0..7")
        correction = residual(hidden.detach(), actions)
    else:
        correction = residual(hidden.detach())
    logits = shape.detach()[:, :, None] + pose.detach()[:, None] + correction.flatten(2)
    logits = logits.masked_fill(~valid.detach()[None], -torch.inf)
    return logits.flatten(1).log_softmax(1).reshape_as(logits)


class ConditionalPoseView(ModePrior):
    def __init__(self, base, residual):
        self.base, self.residual = base, residual

    def scene_details(self, tokens, actions, state=None):
        original = self.base.scene_details(tokens, actions, state)
        hidden, pose, shape = future_logits(self.base, original)
        result = dict(original); result["next"] = dict(original["next"])
        result["next"]["joint"] = conditional_joint(hidden, pose, shape, self.residual, self.base.valid_poses, actions)
        return result


@torch.inference_mode()
def cache_bundle(base, zero, bundle, context):
    frames, actions, labels, state = tensors(base, bundle, context)
    first = temporal_weights(frames).argmax(1)
    qualified = [arm for arm in range(6) if rotation_pair(base, frames, labels, int(first[arm]) + 1, arm) is not None]
    rows = []
    for t in range(8):
        details = base.scene_details(frames[:, t], actions[:, t], state)
        hidden, pose, shape = future_logits(base, details)
        joint = conditional_joint(hidden, pose, shape, zero, base.valid_poses, actions[:, t])
        if not torch.equal(joint, details["next"]["joint"]):
            raise RuntimeError("zero conditional-pose reconstruction differs from base")
        for arm in range(6):
            if t != int(first[arm]): continue
            origin = labels["cells"][arm, t + 1].amin(0)
            kind = int(labels["shape_id"][arm, t + 1])
            known = kind >= 0 and int(labels["evidence"][arm, t + 1]) in (1, 2)
            target = kind * 180 + int(origin[0]) * 10 + int(origin[1]) if known else 0
            cells = labels["cells"][arm, t + 1]
            border = bool(((cells[:, 0] == 0) | (cells[:, 0] == 17) | (cells[:, 1] == 0) | (cells[:, 1] == 9)).any()) if known else False
            rows.append((arm, {"hidden": hidden[arm].cpu().numpy().copy(), "pose": pose[arm].cpu().numpy().copy(),
                         "shape": shape[arm].cpu().numpy().copy(), "target": np.int64(target),
                         "weight": np.float32(1 / len(qualified) if arm in qualified else 0),
                         "known": np.bool_(known), "qualified": np.bool_(arm in qualified),
                         "border": np.bool_(border), "button": np.int8(arm), "effect_step": np.int8(t)}))
            rows[-1][1]["actual_action"] = np.int8(actions[arm, t])
        state = details["state"]
    rows.sort(key=lambda item: item[0])
    return {key: np.stack([item[key] for _, item in rows]) for key in rows[0][1]}


@torch.inference_mode()
def build_caches(base, episodes, tables, zero):
    result, coverage = {}, {}
    for split in ("train", "validation"):
        pieces = []
        for episode, (tokens, actions) in enumerate(episodes[split]):
            anchors = sorted(anchor for index, anchor in tables[split] if index == episode)
            if not anchors: continue
            state = None
            for t in range(anchors[-1] + 1):
                if (episode, t) in tables[split]: pieces.append(cache_bundle(base, zero, tables[split][episode, t], state))
                if t < anchors[-1]: state = base.advance(torch.from_numpy(tokens[t].copy())[None].long(), torch.tensor([int(actions[t])]), state)
        if len(pieces) != len(tables[split]): raise RuntimeError("incomplete causal first-effect cache")
        result[split] = {key: np.concatenate([item[key] for item in pieces]) for key in pieces[0]}
        coverage[split] = {"anchors": len(pieces), "rows": len(pieces) * 6,
                           "qualified": int(result[split]["qualified"].sum()),
                           "bytes": sum(value.nbytes for value in result[split].values())}
    return result, coverage


def batch_joint(residual, cache, valid, indices):
    return conditional_joint(torch.from_numpy(cache["hidden"][indices]), torch.from_numpy(cache["pose"][indices]),
                             torch.from_numpy(cache["shape"][indices]), residual, valid,
                             torch.from_numpy(cache["actual_action"][indices]).long())


def cache_loss(residual, cache, valid, indices):
    joint = batch_joint(residual, cache, valid, indices).flatten(1)
    weights = torch.from_numpy(cache["weight"][indices])
    selected = weights > 0
    if not selected.any(): return joint.exp().sum() * 0
    targets = torch.from_numpy(cache["target"][indices]).long()
    losses = -joint[selected].gather(1, targets[selected, None])[:, 0]
    return (losses * weights[selected] * 6).sum() / len(indices)


@torch.inference_mode()
def cache_metrics(residual, cache, valid):
    counts = Counter()
    for start in range(0, len(cache["target"]), 128):
        indices = np.arange(start, min(start + 128, len(cache["target"])))
        joint = batch_joint(residual, cache, valid, indices)
        selected = joint.flatten(1).argmax(1).numpy()
        target = cache["target"][indices]
        for i, index in enumerate(indices):
            for subset in ("all-known", "qualified"):
                if not cache["known"][index] or (subset == "qualified" and not cache["qualified"][index]): continue
                for region in ("all", "border" if cache["border"][index] else "centre"):
                    prefix = f"{subset}/{region}/button{int(cache['button'][index])}"
                    same_shape = selected[i] // 180 == target[i] // 180
                    counts[prefix + "/total"] += 1
                    counts[prefix + "/shape_correct"] += int(same_shape)
                    counts[prefix + "/joint_correct"] += int(selected[i] == target[i])
                    if same_shape:
                        counts[prefix + "/pose_given_correct_shape_total"] += 1
                        counts[prefix + "/pose_given_correct_shape_correct"] += int(selected[i] == target[i])
                        dy = (selected[i] % 180) // 10 - (target[i] % 180) // 10
                        dx = selected[i] % 10 - target[i] % 10
                        counts[prefix + "/delta_y_sum"] += int(dy); counts[prefix + "/delta_x_sum"] += int(dx)
                        counts[prefix + "/absolute_delta_y_sum"] += int(abs(dy)); counts[prefix + "/absolute_delta_x_sum"] += int(abs(dx))
    report = dict(counts)
    for key, total in counts.items():
        if key.endswith("/total"):
            stem = key[:-6]
            for name in ("shape", "joint"):
                report[stem + f"/{name}_accuracy"] = counts[stem + f"/{name}_correct"] / total if total else None
        elif key.endswith("/pose_given_correct_shape_total"):
            stem = key.removesuffix("/pose_given_correct_shape_total")
            report[stem + "/pose_given_correct_shape_accuracy"] = counts[stem + "/pose_given_correct_shape_correct"] / total if total else None
            for axis in ("y", "x"):
                report[stem + f"/mean_delta_{axis}"] = counts[stem + f"/delta_{axis}_sum"] / total if total else None
                report[stem + f"/mean_absolute_delta_{axis}"] = counts[stem + f"/absolute_delta_{axis}_sum"] / total if total else None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/states-best-v18.pt")
    parser.add_argument("--seed-report", type=Path, default=ROOT / "docs/long-audit-seeds-v15.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-conditioned", action="store_true")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(2); torch.manual_seed(20261106)
    started = time.monotonic()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    raw, episodes, annotations, _, _, digest, annotation_digest = load_scene_data(checkpoint)
    base = load_tile_model(args.checkpoint).eval()
    for parameter in base.parameters(): parameter.requires_grad_(False)
    original = {key: value.clone() for key, value in base.state_dict().items()}
    tables, addon_digest = load_action_interventions(checkpoint, base, episodes, annotations, digest, annotation_digest)
    if (len(tables["train"]), len(tables["validation"])) != (1664, 192): raise ValueError("expected exact V2 1664/192 anchors")
    seed_bytes = args.seed_report.read_bytes(); seeds = json.loads(seed_bytes)
    if seeds["test_used"] or seeds["manifest_sha256"] != digest or seeds["annotations_sha256"] != annotation_digest or seeds["reference_status"] != "completed-reference-audit-gameplay-unverified":
        raise ValueError("seed lineage mismatch")
    verify_seed_prefixes(seeds["seeds"])
    args.output.mkdir(parents=True)
    residual = residual_head(len(base.object_shapes), args.action_conditioned)
    caches, coverage = build_caches(base, episodes, tables, residual)
    if sum(value["bytes"] for value in coverage.values()) > 192 * 1024 ** 2: raise RuntimeError("first-effect cache exceeds192MiB")
    for split, cache in caches.items(): np.savez_compressed(args.output / f"cache-{split}.npz", **cache)
    before = {split: cache_metrics(residual, cache, base.valid_poses) for split, cache in caches.items()}
    zero = residual_head(len(base.object_shapes), args.action_conditioned)
    baseline = ConditionalPoseView(base, zero).eval()
    baseline_teacher = rendered_teacher(baseline, episodes["validation"], annotations["validation"], tables["validation"])
    baseline_rollout, _ = measure(baseline, episodes["validation"], raw["validation"])
    optimizer = torch.optim.AdamW(residual.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(20261106); losses = []
    for update in range(2000):
        indices = torch.randint(len(caches["train"]["target"]), (128,), generator=generator).numpy()
        optimizer.zero_grad(set_to_none=True)
        loss = cache_loss(residual, caches["train"], base.valid_poses, indices)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite conditional-pose objective")
        loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    after = {split: cache_metrics(residual, cache, base.valid_poses) for split, cache in caches.items()}
    view = ConditionalPoseView(base, residual).eval()
    teacher = rendered_teacher(view, episodes["validation"], annotations["validation"], tables["validation"])
    rollout, _ = measure(view, episodes["validation"], raw["validation"])
    entries = [entry for entry in json.loads((ROOT / "data/manifest.json").read_bytes())["episodes"] if entry["split"] == "validation"]
    traces = []
    for label, candidate in (("zero", baseline), ("fitted", view)):
        folder = args.output / label; folder.mkdir()
        traces.append({"variant": label, "long_traces": long_paths(candidate.eval(), seeds, episodes["validation"], annotations["validation"], entries, folder)})
    if any(not torch.equal(value, original[key]) for key, value in base.state_dict().items()) or any(parameter.grad is not None for parameter in base.parameters()):
        raise RuntimeError("frozen base changed")
    config = {"protocol": "action-content-frozen-probe" if args.action_conditioned else "conditional-pose-frozen-probe",
              "action_conditioned": args.action_conditioned, "parameters": sum(p.numel() for p in residual.parameters()),
              "action_input": "actual_action from bundle.actions[arm,effect_step]; branch button is metadata only; delayedA/B atnoop usesnoop filter",
              "stopping_rule": "if cache improves without physical rollout gain, reject and stop next-head probes",
              "seed": 20261106, "updates": 2000, "batch": 128, "lr": 1e-3, "optimizer": "AdamW", "weight_decay": .01,
              "checkpoint_path": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "manifest_sha256": digest, "annotations_sha256": annotation_digest, "interventions_sha256": addon_digest,
              "seed_report_sha256": hashlib.sha256(seed_bytes).hexdigest(), "cache": coverage,
              "objective": "first-effect joint NLL; qualified arms once; mean qualified per anchor; uniform6 rows weighted6; labels target-only",
              "cache_zero_exact_all_joints": True, "validation_selection": False, "torch": str(torch.__version__),
              "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                  ("probe_conditional_pose.py", "scene_model.py", "token_model.py", "intervention_training.py", "train_scene.py", "audit_long.py")}}
    torch.save({"base": base.state_dict(), "conditional_pose": residual.state_dict(), "config": config}, args.output / "probe.pt")
    report = {"status": "diagnostic-gameplay-unverified", "test_used": False, "runtime_adopted": False, "config": config,
              "limits": "conditional-pose hypothesis only; no mathematical necessity or calibrated-posterior claim; fixed-core probe without full-render training",
              "cache_before": before, "cache_after": after, "baseline_teacher": baseline_teacher, "baseline_validation": baseline_rollout,
              "zero_checkpoint_comparison": checkpoint_comparison(baseline_teacher, baseline_rollout, checkpoint),
              "teacher": teacher, "validation": rollout, "long_variants": traces, "base_weights_exact": True,
              "loss_first100": sum(losses[:100]) / 100, "loss_last100": sum(losses[-100:]) / 100,
              "seconds": time.monotonic() - started}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output), "status": report["status"], "seconds": report["seconds"]}))


if __name__ == "__main__": main()
