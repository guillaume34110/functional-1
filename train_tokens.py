"""Fit recurrent categorical tile transitions, preserving exact visual tile patterns."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from evaluate import comparison_image
from run import ROOT
from train import load_data, sample_batch
from token_model import TileDynamics, build_codebook, encode_frames, decode_tokens


def event_loss(write_logits, change_logits, target, current, codebook):
    changed = target != current
    bce = F.binary_cross_entropy_with_logits(change_logits[:, 0], changed.float(), reduction="none")
    ce = F.cross_entropy(write_logits, target, reduction="none")
    blank = (codebook.reshape(len(codebook), -1) == 3).all(1)
    create, erase = changed & ~blank[target], changed & blank[target]

    def mean_mask(values, mask):
        return values[mask].mean() if torch.any(mask) else values.sum() * 0

    # Each event category contributes even when sparse; no reward for only erasing pieces.
    detection = (mean_mask(bce, changed) + mean_mask(bce, ~changed)) * 0.5
    writing = (mean_mask(ce, create) + mean_mask(ce, erase)) * 0.5
    return detection + writing


def measure(model, episodes, raw_episodes, maximum=6):
    model.eval()
    errors, changed_errors, count, changed_count = 0, 0, 0, 0
    effective = []
    examples = []
    with torch.inference_mode():
        for (tokens, actions), (pixels, _) in zip(episodes[:maximum], raw_episodes[:maximum]):
            state = None
            start = min(32, len(actions) - 1)
            for t in range(start):
                _, state = model(torch.from_numpy(tokens[t].copy())[None].long(), torch.tensor([int(actions[t])]), state)
            current = torch.from_numpy(tokens[start].copy())[None].long()
            predicted = [pixels[start].copy()]
            length = min(256, len(actions) - start)
            effective.append(length)
            for offset in range(length):
                t = start + offset
                if hasattr(model, "predict_tokens"):
                    current, state = model.predict_tokens(current, torch.tensor([int(actions[t])]), state)
                else:
                    logits, state = model(current, torch.tensor([int(actions[t])]), state)
                    current = logits.argmax(1)
                truth = torch.from_numpy(tokens[t + 1].copy())[None]
                changed = tokens[t + 1] != tokens[t]
                wrong = (current != truth).numpy()[0]
                errors += int(wrong.sum()); count += wrong.size
                changed_errors += int((wrong & changed).sum()); changed_count += int(changed.sum())
                predicted.append(decode_tokens(current, model.codebook)[0, 0].numpy().copy())
            examples.append((np.stack(predicted), pixels[start:start + length + 1]))
    model.train()
    return {"tile_error": errors / max(1, count), "changed_tile_error": changed_errors / max(1, changed_count),
            "changed_tiles": changed_count, "effective_steps": effective,
            "full_256_coverage": bool(effective) and min(effective) >= 256}, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=2400)
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--events", action="store_true")
    parser.add_argument("--warm-start", type=Path)
    args = parser.parse_args()
    if min(args.seconds, args.updates, args.batch, args.threads) < 1:
        parser.error("invalid bounds")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(1042027)
    rng = random.Random(1042027)
    raw, digest, manifest = load_data(args.data)
    codebook = build_codebook(frames for frames, _ in raw["train"])
    codebook = np.asarray(codebook, dtype=np.uint8)
    if args.events:
        from token_model import EventTileDynamics
        model = EventTileDynamics(torch.from_numpy(codebook))
    else:
        model = TileDynamics(torch.from_numpy(codebook))
    if args.warm_start:
        if not args.events or args.resume:
            parser.error("warm-start is for the event-head architecture, without resume")
        state_dict = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        if not torch.equal(state_dict["codebook"], model.codebook):
            raise ValueError("warm-start codebook mismatch")
        shared = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        mismatch = model.load_state_dict(shared, strict=False)
        if set(mismatch.missing_keys) != {"change_head.weight", "change_head.bias"}:
            raise ValueError("unexpected warm-start architecture mismatch")
    encoded, unknown = {}, {}
    for split in ("train", "validation"):
        encoded[split] = []
        unknown[split] = 0
        for frames, actions in raw[split]:
            tokens, missing = encode_frames(frames, codebook)
            encoded[split].append((tokens, actions))
            unknown[split] += int(missing)
    if unknown["train"]:
        raise RuntimeError("train codebook failed to encode its own visual tiles")
    args.output.mkdir(parents=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    first_step, previous_sampling = 0, 0.0
    previous = None
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=True)
        if previous["config"]["manifest_sha256"] != digest or not torch.equal(previous["model"]["codebook"], model.codebook):
            raise ValueError("resume data/codebook mismatch")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        rng.setstate(previous["python_rng"])
        torch.set_rng_state(previous["torch_rng"])
        first_step, previous_sampling = previous["step"], previous["sampling"]
        if args.updates <= first_step:
            raise ValueError("resume needs a greater total update target")
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
               for name in ("token_model.py", "train_tokens.py", "collect.py", "context.json")}
    config = {"representation": "visual-tiles-4x4-events" if args.events else "visual-tiles-4x4", "codebook_from": "train-only", "vocabulary": len(codebook),
              "parameters": sum(p.numel() for p in model.parameters()), "unknown_tiles": unknown,
              "manifest_sha256": digest, "source_sha256": sources, "seconds": args.seconds,
              "updates": args.updates, "batch": args.batch, "burn_in": 16, "unroll": 32, "seed": 1042027}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    started = time.monotonic()
    best_score, best_path = float("inf"), None
    if previous:
        best_score = previous["validation"]["tile_error"] + previous["validation"]["changed_tile_error"]
        best_path = args.resume
    losses = []
    with (args.output / "metrics.jsonl").open("x", buffering=1) as log:
        for step in range(first_step + 1, args.updates + 1):
            frames, actions = sample_batch(encoded["train"], rng, args.batch, 48)
            state = None
            with torch.no_grad():
                for t in range(16):
                    _, state = model(frames[:, t].long(), actions[:, t], state)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros(())
            generated = None
            progress = max(step / args.updates, (time.monotonic() - started) / args.seconds)
            sampling = max(previous_sampling, min(1.0, 2.0 * progress))
            for t in range(16, 48):
                current = generated if generated is not None and rng.random() < sampling else frames[:, t].long()
                target = frames[:, t + 1].long()
                if args.events:
                    write, change, state = model.forward_details(current, actions[:, t], state)
                    loss = loss + event_loss(write, change, target, current, model.codebook)
                    logits = model.combine_logits(current, write, change)
                else:
                    logits, state = model(current, actions[:, t], state)
                    weights = 1.0 + 31.0 * (target != current).float()
                    loss = loss + (F.cross_entropy(logits, target, reduction="none") * weights).sum() / weights.sum()
                generated = logits.detach().argmax(1)
            loss = loss / 32
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite tile training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            expired = time.monotonic() - started >= args.seconds
            if step % 200 == 0 or step == args.updates or expired:
                metrics, _ = measure(model, encoded["validation"], raw["validation"])
                score = metrics["tile_error"] + metrics["changed_tile_error"]
                record = {"step": step, "seconds": time.monotonic() - started, "loss": float(np.mean(losses)),
                          "sampling": sampling, "validation": metrics}
                log.write(json.dumps(record) + "\n"); print(json.dumps(record), flush=True)
                losses = []
                if score < best_score or expired or step == args.updates:
                    path = args.output / f"checkpoint-{step:06d}.pt"
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                                "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                                "sampling": sampling,
                                "config": config, "validation": metrics, "dataset_context": manifest["context"]}, path)
                    if score < best_score:
                        best_score, best_path = score, path
                if expired:
                    break
    if best_path is None:
        raise RuntimeError("no validated model")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    torch.save(model.state_dict(), args.output / "model.pt")
    # Do not open a fresh test when validation already disproves usable dynamics.
    validation = checkpoint["validation"]
    ready = validation["full_256_coverage"] and validation["tile_error"] < 0.005 and validation["changed_tile_error"] < 0.05
    encoded_test, unknown_test = [], 0
    if ready:
        for frames, actions in raw["test"]:
            tokens, missing = encode_frames(frames, codebook)
            encoded_test.append((tokens, actions)); unknown_test += int(missing)
        test, examples = measure(model, encoded_test, raw["test"], maximum=len(raw["test"]))
        seed_split = "test"
    else:
        test = None
        _, examples = measure(model, encoded["validation"], raw["validation"])
        seed_split = "validation"
    for index, (predicted, reference) in enumerate(examples):
        comparison_image(predicted, reference, args.output / f"comparison-{index:02d}.png")
        np.savez_compressed(args.output / f"rollout-{index:02d}.npz", predicted=predicted, reference=reference)
    pixels, actions = raw[seed_split][0]
    np.savez_compressed(args.output / "play-seed.npz", frames=pixels[:33], actions=actions[:32],
                        metadata=json.dumps({"context": manifest["context"], "source": "capture", "prefix_only": True}))
    result = {"status": "trained-gameplay-unverified", "best_checkpoint": best_path.name, "config": config,
              "validation": validation, "test": test, "test_reserved_unopened": not ready, "unknown_test_tiles": unknown_test if ready else None,
              "seconds": time.monotonic() - started, "model_sha256": hashlib.sha256((args.output / "model.pt").read_bytes()).hexdigest()}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
