"""One frozen-core75-parameter identity probe; no runtime model adoption."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from audit_mode_prior import ModePrior, checkpoint_comparison, long_paths
from intervention_training import BUTTONS, rotation_pair, temporal_weights, tensors
from run import ROOT
from token_model import load_tile_model
from train_scene import load_action_interventions, load_scene_data, rendered_teacher
from train_tokens import measure


class IdentityGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(24, 3)
        nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)

    def forward(self, features):
        return self.linear(features).tanh()


def current_priors(joint):
    """Finite detached log marginals: shapeK, origin x10, origin y18."""
    grid = joint.detach().reshape(len(joint), joint.shape[1], 18, 10)
    shape = grid.flatten(2).logsumexp(2)
    x = grid.logsumexp(1).logsumexp(1)
    y = grid.logsumexp(1).logsumexp(2)
    result = torch.cat((shape, x, y), 1).clamp_min(math.log(torch.finfo(joint.dtype).tiny))
    if not torch.isfinite(result).all():
        raise RuntimeError("nonfinite predicted current marginals")
    return result.detach()


def reconstruct_joint(pose, shape, features, priors, gate, valid):
    coefficients = gate(features.detach())
    kinds = shape.shape[1]
    prior = priors.detach()
    shape_prior = prior[:, :kinds, None]
    x_prior = prior[:, kinds:kinds + 10].repeat(1, 18)[:, None]
    y_prior = prior[:, kinds + 10:].repeat_interleave(10, 1)[:, None]
    logits = shape.detach()[:, :, None] + pose.detach()[:, None]
    logits = (logits + coefficients[:, 0, None, None] * shape_prior
              + coefficients[:, 1, None, None] * x_prior
              + coefficients[:, 2, None, None] * y_prior)
    logits = logits.masked_fill(~valid.detach()[None], -torch.inf)
    return logits.flatten(1).log_softmax(1).reshape_as(logits)


def head_inputs(model, details, actions):
    hidden = details["state"][:, :16, :, 2:12]
    # Exactly _ObjectHead.forward: pose softmax pool precedes the geometric joint mask.
    current_pose = model.current_head.pose(hidden).flatten(1)
    current_pool = torch.einsum("bcp,bp->bc", hidden.flatten(2), current_pose.softmax(1))
    next_pose = model.next_head.pose(hidden).flatten(1)
    next_pool = torch.einsum("bcp,bp->bc", hidden.flatten(2), next_pose.softmax(1))
    next_shape = model.next_head.shape(next_pool)
    features = torch.cat((current_pool, model.action_embedding(actions)), 1)
    return next_pose.detach(), next_shape.detach(), features.detach(), current_priors(details["current"]["joint"])


class IdentityView(ModePrior):
    def __init__(self, base, gate):
        self.base, self.gate = base, gate

    def scene_details(self, tokens, actions, state=None):
        original = self.base.scene_details(tokens, actions, state)
        pose, shape, features, priors = head_inputs(self.base, original, actions)
        result = dict(original)
        result["next"] = dict(original["next"])
        result["next"]["joint"] = reconstruct_joint(pose, shape, features, priors, self.gate, self.base.valid_poses)
        return result


def target_weights(model, frames, labels):
    positive = (labels["shape_id"] >= 0) & ((labels["evidence"] == 1) | (labels["evidence"] == 2))
    temporal = temporal_weights(frames)
    first = temporal.argmax(1)
    qualified = [arm for arm in range(6) if rotation_pair(model, frames, labels, int(first[arm]) + 1, arm) is not None]
    weights = .1 * temporal / 6 * positive[:, 1:]
    for arm in qualified:
        weights[arm, first[arm]] += 1 / len(qualified)
    origin = labels["cells"][:, 1:].amin(2)
    wanted = labels["shape_id"][:, 1:] * 180 + origin[:, :, 0] * 10 + origin[:, :, 1]
    wanted = torch.where(positive[:, 1:], wanted, torch.zeros_like(wanted))
    return wanted, weights, len(qualified)


@torch.inference_mode()
def cache_bundle(model, zero, bundle, context):
    frames, actions, labels, state = tensors(model, bundle, context)
    wanted, weights, qualified = target_weights(model, frames, labels)
    arrays = {key: [] for key in ("pose", "shape", "features", "priors", "target", "weight")}
    for t in range(8):
        details = model.scene_details(frames[:, t], actions[:, t], state)
        pose, shape, features, priors = head_inputs(model, details, actions[:, t])
        joint = reconstruct_joint(pose, shape, features, priors, zero, model.valid_poses)
        if not torch.equal(joint, details["next"]["joint"]):
            raise RuntimeError("zero-gate cache reconstruction differs from the base next joint")
        for key, value in (("pose", pose), ("shape", shape), ("features", features), ("priors", priors),
                           ("target", wanted[:, t]), ("weight", weights[:, t])):
            arrays[key].append(value.cpu().numpy().copy())
        state = details["state"]
    return {key: np.concatenate(values) for key, values in arrays.items()}, qualified


@torch.inference_mode()
def build_caches(model, episodes, tables, zero):
    caches, coverage = {}, {}
    for split in ("train", "validation"):
        pieces = {key: [] for key in ("pose", "shape", "features", "priors", "target", "weight")}
        count = qualified = 0
        for episode, (tokens, actions) in enumerate(episodes[split]):
            anchors = sorted(anchor for index, anchor in tables[split] if index == episode)
            if not anchors:
                continue
            state = None
            for t in range(anchors[-1] + 1):
                if (episode, t) in tables[split]:
                    data, known = cache_bundle(model, zero, tables[split][(episode, t)], state)
                    for key, value in data.items(): pieces[key].append(value)
                    qualified += known; count += 1
                if t < anchors[-1]:
                    state = model.advance(torch.from_numpy(tokens[t].copy())[None].long(), torch.tensor([int(actions[t])]), state)
        if count != len(tables[split]):
            raise RuntimeError("incomplete causal cache")
        caches[split] = {key: np.concatenate(values) for key, values in pieces.items()}
        coverage[split] = {"anchors": count, "rows": count * 48, "qualified_first_effect_arms": qualified,
                           "bytes": sum(value.nbytes for value in caches[split].values())}
    return caches, coverage


def cache_loss(gate, cache, valid, indices):
    data = {key: torch.as_tensor(value[indices]) for key, value in cache.items()}
    joint = reconstruct_joint(data["pose"], data["shape"], data["features"], data["priors"], gate, valid)
    losses = -joint.flatten(1).gather(1, data["target"].long()[:, None])[:, 0]
    return (losses * data["weight"] * 48).mean()


def verify_seed_prefixes(seeds):
    if len(seeds) != 5 or len({(seed["file"], seed["frame"]) for seed in seeds}) != 5:
        raise ValueError("probe requires five distinct verified seed pairs")
    for seed in seeds:
        proof = seed["reference_verified_frames"]
        if set(proof) != set(BUTTONS) or any(value != seed["frame"] + 1 for value in proof.values()):
            raise ValueError("unverified long-rollout prefix")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/states-best-v18.pt")
    parser.add_argument("--seed-report", type=Path, default=ROOT / "docs/long-audit-seeds-v15.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(2); torch.manual_seed(20261106)
    started = time.monotonic()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    raw, episodes, annotations, _, _, digest, annotation_digest = load_scene_data(checkpoint)
    model = load_tile_model(args.checkpoint).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    tables, addon_digest = load_action_interventions(checkpoint, model, episodes, annotations, digest, annotation_digest)
    if (len(tables["train"]), len(tables["validation"])) != (1664, 192):
        raise ValueError("probe requires exact V2 train1664/validation192")
    seed_bytes = args.seed_report.read_bytes(); seeds = json.loads(seed_bytes)
    if (seeds["test_used"] or seeds["manifest_sha256"] != digest or seeds["annotations_sha256"] != annotation_digest
            or seeds["reference_status"] != "completed-reference-audit-gameplay-unverified" or len(seeds["seeds"]) != 5):
        raise ValueError("probe seed lineage mismatch")
    verify_seed_prefixes(seeds["seeds"])
    args.output.mkdir(parents=True)
    gate = IdentityGate()
    caches, coverage = build_caches(model, episodes, tables, gate)
    if sum(item["bytes"] for item in coverage.values()) > 128 * 1024 ** 2:
        raise RuntimeError("compact float32 cache exceeds128MiB")
    for split, cache in caches.items(): np.savez_compressed(args.output / f"cache-{split}.npz", **cache)
    config = {"protocol": "frozen-identity-probe75", "seed": 20261106, "updates": 2000, "batch": 128,
              "optimizer": "AdamW", "lr": 1e-3, "weight_decay": .01, "parameters": 75,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "base_checkpoint_path": str(args.checkpoint), "torch": str(torch.__version__),
              "manifest_sha256": digest, "annotations_sha256": annotation_digest, "interventions_sha256": addon_digest,
              "seed_report_sha256": hashlib.sha256(seed_bytes).hexdigest(),
              "controls_source_sha256": seeds["controls_audit_sha256"], "reference_source_sha256": seeds["verified_reference_sha256"],
              "cache": coverage, "dtype": "float32", "all_cached_joints_zero_gate_exact": True,
              "objective": "next joint CE .1*temporal/6 plus1/nqualified at each qualified first effect; uniform rows weighted48",
              "inputs": "detached predicted current marginals, predicted current pose-pool16 and action embedding8 only; labels targets only",
              "validation_selection": False, "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                  ("probe_identity_prior.py", "scene_model.py", "token_model.py", "intervention_training.py", "train_scene.py", "audit_long.py")}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    baseline_teacher = rendered_teacher(IdentityView(model, gate).eval(), episodes["validation"], annotations["validation"], tables["validation"])
    baseline_rollout, _ = measure(IdentityView(model, gate).eval(), episodes["validation"], raw["validation"])
    optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(20261106)
    losses = []
    for update in range(2000):
        indices = torch.randint(len(caches["train"]["target"]), (128,), generator=generator).numpy()
        optimizer.zero_grad(set_to_none=True)
        loss = cache_loss(gate, caches["train"], model.valid_poses, indices)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite frozen-probe objective")
        loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    view = IdentityView(model, gate).eval()
    teacher = rendered_teacher(view, episodes["validation"], annotations["validation"], tables["validation"])
    rollout, _ = measure(view, episodes["validation"], raw["validation"])
    entries = [entry for entry in json.loads((ROOT / "data/manifest.json").read_bytes())["episodes"] if entry["split"] == "validation"]
    traces = []
    for label, candidate in (("zero", IdentityView(model, IdentityGate())), ("fitted", view)):
        folder = args.output / label; folder.mkdir()
        traces.append({"variant": label, "long_traces": long_paths(candidate.eval(), seeds, episodes["validation"], annotations["validation"], entries, folder)})
    if any(not torch.equal(value, original[key]) for key, value in model.state_dict().items()) or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("frozen core was modified")
    torch.save({"base": model.state_dict(), "identity_prior": gate.state_dict(), "config": config}, args.output / "probe.pt")
    report = {"status": "diagnostic-gameplay-unverified", "test_used": False, "runtime_adopted": False, "config": config,
              "baseline_teacher": baseline_teacher, "baseline_validation": baseline_rollout,
              "zero_checkpoint_comparison": checkpoint_comparison(baseline_teacher, baseline_rollout, checkpoint),
              "teacher": teacher, "validation": rollout, "long_variants": traces,
              "loss_first100": sum(losses[:100]) / 100, "loss_last100": sum(losses[-100:]) / 100,
              "base_weights_exact": True, "base_gradients_absent": True, "seconds": time.monotonic() - started}
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output), "status": report["status"], "seconds": report["seconds"]}))


if __name__ == "__main__": main()
