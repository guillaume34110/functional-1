"""Calibrate collision scores on proven observed and generated canvases."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scene_model import canvas_overlap_scene_weights, scene_from_weights
from train_mode_correction import proven_composition
from train_spawn_preview import sha, training_captures


@torch.no_grad()
def collect(model, base, captures):
    samples, protocols = [], []

    def record(details, tokens, background, kind, generated):
        if not (details["contact_visible"][0] and details["next"]["visible"][0, 0] > 0):
            return
        current = int(details["current"]["joint"].flatten().argmax()) // 180
        if current != kind:
            return
        mask, _ = base._hard_object(details["current"])
        if not torch.equal(tokens.masked_fill(mask, base.blank_id)[0, :, 2:12], background):
            return
        scores = details["event_scores"][0]
        overlap = model.canvas_overlap(details)[0]
        grouped = scores.new_full((2, 5), -torch.inf)
        grouped.scatter_reduce_(1, overlap, scores[:2], reduce="amax")
        samples.append(torch.cat((grouped.flatten(), scores[2].max()[None])))
        protocols.append(generated)

    for episode, (_, frames, actions, labels) in enumerate(captures):
        modes, known, old, masks, _ = proven_composition(base, frames, labels)
        background = frames[:-1].masked_fill(masks, base.blank_id)[:, :, 2:12]
        state = generated = generated_state = None
        for t, command in enumerate(actions):
            frame, action = frames[t:t + 1], command[None]
            phase = (t + episode) % 40
            observed = old[t] and (t % 4 == 0 or (known[t] and modes[t] != 0))
            if observed or phase == 0:
                details = base.scene_details(frame, action, state)
                state = details["state"]
                if observed:
                    record(details, frame, background[t], labels["shape_id"][t], False)
            else:
                state = base.advance(frame, action, state)
            if phase == 0:
                generated, generated_state = base.decode_scene(frame, details, hard=True), state
            elif phase < 16 and generated is not None:
                rollout = base.scene_details(generated, action, generated_state)
                if old[t]:
                    record(rollout, generated, background[t], labels["shape_id"][t], True)
                generated, generated_state = base.decode_scene(generated, rollout, hard=True), rollout["state"]
    if not samples or not any(protocols):
        raise ValueError("need observed and generated canvases with proven foreground")
    return torch.stack(samples), torch.tensor(protocols)


@torch.no_grad()
def metrics(grouped, generated, scale):
    hits = torch.arange(5)
    scores = grouped[:, :10].reshape(-1, 2, 5) - scale[None, :, None] * hits
    best = torch.cat((scores.flatten(1), grouped[:, 10:]), dim=1).argmax(1)
    return {name: {"examples": int(mask.sum()),
                   "copy_collisions": int((mask & (best > 0) & (best < 5)).sum()),
                   "compose_collisions": int((mask & (best > 5) & (best < 10)).sum()),
                   "rewrite": int((mask & (best == 10)).sum())}
            for name, mask in (("observed", ~generated), ("autonomous", generated))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    model = canvas_overlap_scene_weights(checkpoint["model"]).eval()
    grouped, generated = collect(model, base, training_captures(checkpoint["config"], base.codebook.numpy()))
    geometry = grouped[:, :10].reshape(-1, 2, 5)
    free = geometry[:, :, 0]
    hit = geometry[:, :, 1:]
    hardest = hit.amax(2)
    qualified = torch.isfinite(free) & torch.isfinite(hardest) & (hardest > free - 2)
    if not qualified.any(0).all():
        raise ValueError("both canvases require difficult overlap candidates")
    raw_scale = model.canvas_overlap_log_scale
    optimizer = torch.optim.Adam((raw_scale,), lr=.05)
    counts = torch.arange(1, 5)
    for _ in range(400):
        scale = F.softplus(raw_scale)
        corrected = (hit - scale[None, :, None] * counts).amax(2)
        margin = (1 + corrected.masked_fill(~qualified, 0) - free.masked_fill(~qualified, 0))
        loss = (F.softplus(margin) * qualified).sum(0).div(qualified.sum(0)).mean() + .001 * scale.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((raw_scale,), 1., error_if_nonfinite=True)
        optimizer.step()
    state = model.state_dict()
    for key, value in checkpoint["model"].items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    scale = F.softplus(raw_scale.detach())
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "learned_scale": scale.tolist(),
              "qualified": qualified.sum(0).tolist(), "before": metrics(grouped, generated, torch.zeros(2)),
              "after": metrics(grouped, generated, scale)}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "canvas_overlap_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
