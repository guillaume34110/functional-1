"""Train screen dynamics on independent ROM runs, selecting by free-running validation."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import NeuralTetris
from run import ROOT, read_episode


def load_data(directory):
    directory = Path(directory)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "complete":
        raise ValueError("dataset collection is incomplete")
    context = json.loads((ROOT / "context.json").read_text())
    if manifest["context"] != context:
        raise ValueError("dataset/model context mismatch")
    splits = {name: [] for name in ("train", "validation", "test")}
    hashes = {}
    prefixes = {}
    for entry in manifest["episodes"]:
        path = (directory / entry["file"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("dataset path escapes its root")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"episode hash mismatch: {path}")
        frames, actions, metadata = read_episode(path)
        split = entry["split"]
        if metadata.get("source") != "capture" or split not in splits:
            raise ValueError("training requires reference capture and known split")
        trajectory = hashlib.sha256(frames.tobytes() + actions.tobytes()).hexdigest()
        if trajectory in hashes and hashes[trajectory] != split:
            raise ValueError("identical trajectory crosses dataset splits")
        hashes[trajectory] = split
        prefix = hashlib.sha256(frames[:32].tobytes() + actions[:31].tobytes()).hexdigest()
        if prefix in prefixes and prefixes[prefix] != split:
            raise ValueError("identical observed prefix crosses dataset splits")
        prefixes[prefix] = split
        if len(actions) < 80:
            raise ValueError("episodes must contain at least 80 transitions")
        splits[split].append((frames, actions))
    if any(not episodes for episodes in splits.values()):
        raise ValueError("train/validation/test must all contain independent episodes")
    return splits, hashlib.sha256(manifest_bytes).hexdigest(), manifest


def transition_loss(logits, target, current):
    errors = F.cross_entropy(logits, target.long(), reduction="none")
    weights = 1.0 + 31.0 * (target != current).float()
    return (errors * weights).sum() / weights.sum()


def sample_batch(episodes, rng, batch_size, length):
    frames, actions = [], []
    for _ in range(batch_size):
        episode_frames, episode_actions = episodes[rng.randrange(len(episodes))]
        start = rng.randrange(len(episode_actions) - length + 1)
        frames.append(episode_frames[start:start + length + 1])
        actions.append(episode_actions[start:start + length])
    return torch.from_numpy(np.stack(frames)), torch.from_numpy(np.stack(actions)).long()


def rollout_metrics(model, episodes, horizon=256, burn_in=32, maximum=4):
    """Validation only; no reference screen is fed back after the initial prefix."""
    model.eval()
    stats = {str(k): {"wrong": 0, "pixels": 0, "changed_wrong": 0, "changed": 0,
                      "copy_wrong": 0, "copy_pixels": 0} for k in (1, 16, 64, 256)}
    examples = []
    with torch.inference_mode():
        for reference, actions in episodes[:maximum]:
            state = None
            for t in range(min(burn_in, len(actions) - 1)):
                _, state = model(torch.from_numpy(reference[t].copy())[None, None], torch.tensor([int(actions[t])]), state)
            start = min(burn_in, len(actions) - 1)
            frame = torch.from_numpy(reference[start].copy())[None, None]
            predictions = [reference[start].copy()]
            for offset in range(min(horizon, len(actions) - start)):
                t = start + offset
                logits, state = model(frame, torch.tensor([int(actions[t])]), state)
                frame = logits.argmax(1, keepdim=True).to(torch.uint8)
                prediction = frame[0, 0].numpy()
                predictions.append(prediction.copy())
                wrong = prediction != reference[t + 1]
                changed = reference[t + 1] != reference[t]
                for cutoff in (1, 16, 64, 256):
                    if offset < cutoff:
                        s = stats[str(cutoff)]
                        s["wrong"] += int(wrong.sum()); s["pixels"] += wrong.size
                        s["changed_wrong"] += int((wrong & changed).sum()); s["changed"] += int(changed.sum())
                        s["copy_wrong"] += int((reference[start] != reference[t + 1]).sum())
                        s["copy_pixels"] += wrong.size
            examples.append((np.stack(predictions), reference[start:start + len(predictions)].copy()))
    metrics = {}
    for horizon_name, values in stats.items():
        metrics[horizon_name] = {"pixel_error": values["wrong"] / max(1, values["pixels"]),
                                 "changed_pixel_error": values["changed_wrong"] / max(1, values["changed"]),
                                 "changed_pixels": values["changed"],
                                 "frozen_screen_error": values["copy_wrong"] / max(1, values["copy_pixels"])}
    model.train()
    return metrics, examples


def write_checkpoint(output, step, model, optimizer, config, digest, rng, metrics):
    path = output / f"checkpoint-{step:06d}.pt"
    if path.exists():
        raise FileExistsError(path)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
               "config": config, "manifest_sha256": digest, "python_rng": rng.getstate(),
               "torch_rng": torch.get_rng_state(), "validation": metrics,
               "context": json.loads((ROOT / "context.json").read_text())}
    torch.save(payload, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--seconds", type=int, default=2700)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--unroll", type=int, default=16)
    parser.add_argument("--burn-in", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1042026)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.updates, args.seconds, args.batch, args.unroll, args.threads) < 1 or args.burn_in < 0:
        parser.error("invalid training bounds")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    episodes, digest, manifest = load_data(args.data)
    if args.burn_in + args.unroll > min(len(a) for _, a in episodes["train"]):
        raise ValueError("windows exceed an episode")
    args.output.mkdir(parents=True)
    model = NeuralTetris()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    first_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["manifest_sha256"] != digest or checkpoint["context"] != manifest["context"]:
            raise ValueError("resume dataset or model context mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_step = checkpoint["step"]
        rng.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["torch_version"] = str(torch.__version__)
    config["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("model.py", "train.py", "run.py", "context.json")}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    started = time.monotonic()
    best_score = float("inf")
    best_path = None
    if args.resume:
        metrics = checkpoint["validation"]
        best_score = metrics["256"]["pixel_error"] + metrics["256"]["changed_pixel_error"]
        best_path = args.resume
    running_loss = []
    final_step = first_step
    model.train()
    with (args.output / "metrics.jsonl").open("x", buffering=1) as log:
        for step_number in range(first_step + 1, args.updates + 1):
            frames, actions = sample_batch(episodes["train"], rng, args.batch, args.burn_in + args.unroll)
            state = None
            with torch.no_grad():
                for t in range(args.burn_in):
                    _, state = model(frames[:, t, None], actions[:, t], state)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros(())
            generated = None
            progress = max(step_number / args.updates, (time.monotonic() - started) / args.seconds)
            sampling = min(0.5, progress)
            for t in range(args.burn_in, args.burn_in + args.unroll):
                current = generated if generated is not None and rng.random() < sampling else frames[:, t, None]
                logits, state = model(current, actions[:, t], state)
                loss = loss + transition_loss(logits, frames[:, t + 1], frames[:, t])
                generated = logits.detach().argmax(1, keepdim=True).to(torch.uint8)
            loss = loss / args.unroll
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            final_step = step_number
            running_loss.append(float(loss.detach()))
            expired = time.monotonic() - started >= args.seconds
            if step_number % 250 == 0 or step_number == args.updates or expired:
                metrics, _ = rollout_metrics(model, episodes["validation"], maximum=2)
                score = metrics["256"]["pixel_error"] + metrics["256"]["changed_pixel_error"]
                record = {"step": step_number, "seconds": round(time.monotonic() - started, 3),
                          "train_loss": float(np.mean(running_loss)), "sampling": sampling, "validation": metrics}
                log.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                running_loss = []
                if score < best_score or expired or step_number == args.updates:
                    path = write_checkpoint(args.output, step_number, model, optimizer, config, digest, rng, metrics)
                    if score < best_score:
                        best_score, best_path = score, path
                if expired:
                    break
    if best_path is None:
        raise RuntimeError("no validated checkpoint produced")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    torch.save(best["model"], args.output / "model.pt")
    result = {"status": "trained-not-yet-proven-playable", "step": final_step, "best_checkpoint": best_path.name,
              "manifest_sha256": digest, "validation": best["validation"], "seconds": time.monotonic() - started}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
