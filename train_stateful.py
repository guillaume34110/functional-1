"""Continuous-episode visual dynamics with calibrated rare-event supervision."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from evaluate import comparison_image
from run import ROOT
from token_model import CalibratedEventDynamics, ObservedHistoryDynamics, encode_frames, expand_history_kernel, upgrade_history_weights, load_tile_model
from train import load_data
from train_tokens import measure


class StreamWindows:
    def __init__(self, episodes, batch, rng, length=32):
        self.episodes, self.batch, self.rng, self.length = episodes, batch, rng, length
        self.cursors = [None] * batch

    def next(self):
        frames, actions, masks, resets, prefixes, prefix_actions = [], [], [], [], [], []
        for i, cursor in enumerate(self.cursors):
            reset = cursor is None or cursor[1] >= len(self.episodes[cursor[0]][1])
            if reset:
                cursor = (self.rng.randrange(len(self.episodes)), 32)
            index, start = cursor
            observations, buttons = self.episodes[index]
            used = min(self.length, len(buttons) - start)
            sequence = observations[start:start + used + 1]
            frames.append(np.concatenate((sequence, np.repeat(sequence[-1:], self.length - used, axis=0))))
            actions.append(np.pad(buttons[start:start + used], (0, self.length - used)))
            masks.append(np.arange(self.length) < used)
            resets.append(reset)
            prefixes.append(observations[:32]); prefix_actions.append(buttons[:32])
            self.cursors[i] = (index, start + used)
        return (torch.from_numpy(np.stack(frames)).long(), torch.from_numpy(np.stack(actions)).long(),
                torch.from_numpy(np.stack(masks)), torch.tensor(resets),
                torch.from_numpy(np.stack(prefixes)).long(), torch.from_numpy(np.stack(prefix_actions)).long())


def calibrated_loss(write, change, target, current, book, valid):
    changed = target != current
    mask = valid[:, None, None].expand_as(target)
    bce = F.binary_cross_entropy_with_logits(change[:, 0], changed.float(),
                                             pos_weight=torch.tensor(16.0, device=change.device), reduction="none")
    detection = (bce * mask).sum() / mask.sum().clamp_min(1)
    blank = (book.flatten(1) == 3).all(1)
    ce = F.cross_entropy(write, target, reduction="none")
    create, erase = changed & ~blank[target] & mask, changed & blank[target] & mask
    writing = 0.5 * ((ce * create).sum() / create.sum().clamp_min(1) + (ce * erase).sum() / erase.sum().clamp_min(1))
    return detection + writing


def teacher_metrics(model, episodes, maximum=6):
    counts = dict(additions=0, additions_correct=0, erasures=0, erasures_correct=0,
                  true_changes=0, gate_tp=0, gate_fp=0, unchanged=0)
    blank = (model.codebook.flatten(1) == 3).all(1)
    model.eval()
    with torch.inference_mode():
        for tokens, actions in episodes[:maximum]:
            state = None
            for t in range(min(800, len(actions))):
                current = torch.from_numpy(tokens[t].copy())[None].long()
                write, raw_gate, state = model.forward_details(current, torch.tensor([int(actions[t])]), state)
                if t < 32:
                    continue
                gate = raw_gate - model.gate_logit_offset
                logits = model.combine_logits(current, write, gate)
                predicted = logits.argmax(1)
                target = torch.from_numpy(tokens[t + 1].copy())[None].long()
                changed, correct = target != current, predicted == target
                add, erase = changed & ~blank[target], changed & blank[target]
                counts["additions"] += int(add.sum()); counts["additions_correct"] += int((add & correct).sum())
                counts["erasures"] += int(erase.sum()); counts["erasures_correct"] += int((erase & correct).sum())
                counts["true_changes"] += int(changed.sum()); counts["gate_tp"] += int(((gate[:, 0] > 0) & changed).sum())
                counts["gate_fp"] += int(((gate[:, 0] > 0) & ~changed).sum()); counts["unchanged"] += int((~changed).sum())
    return {"counts": counts, "addition_accuracy": counts["additions_correct"] / max(1, counts["additions"]),
            "erasure_accuracy": counts["erasures_correct"] / max(1, counts["erasures"]),
            "gate_precision": counts["gate_tp"] / max(1, counts["gate_tp"] + counts["gate_fp"]),
            "gate_recall": counts["gate_tp"] / max(1, counts["true_changes"]),
            "gate_fpr": counts["gate_fp"] / max(1, counts["unchanged"])}


def selection_score(teacher, metrics):
    return (1 - teacher["addition_accuracy"]) + (1 - teacher["erasure_accuracy"]) + 10 * teacher["gate_fpr"] + metrics["tile_error"]


def checkpoint_baseline(checkpoint):
    if checkpoint["config"].get("protocol") == "stateful-contiguous-calibrated":
        return selection_score(checkpoint["teacher"], checkpoint["validation"])
    return float("inf")


def expand_history_optimizer(state_dict, model):
    result = copy.deepcopy(state_dict)
    if len(result["param_groups"]) != 1:
        raise ValueError("history upgrade expects the existing single optimizer group")
    group = result["param_groups"][0]
    named = list(model.named_parameters())
    old_count = len(group["params"])
    if [n for n, _ in named[old_count:]] != ["tile_age_embedding.weight", "held_duration_embedding.weight"]:
        raise ValueError("unexpected parameter order during history upgrade")
    for identifier, (name, _) in zip(group["params"], named[:old_count]):
        if name in ("gates.weight", "candidate.weight"):
            for key in ("exp_avg", "exp_avg_sq"):
                if key in result["state"].get(identifier, {}):
                    result["state"][identifier][key] = expand_history_kernel(result["state"][identifier][key])
    first_new = max(group["params"]) + 1
    group["params"].extend((first_new, first_new + 1))
    return result


def observed_stream_state(old, cursors, episodes):
    if tuple(old.shape[1:]) != (16, 18, 20) or len(old) != len(cursors):
        raise ValueError("expected existing sixteen-channel stream state")
    result = old.new_zeros((len(old), 20, 18, 20))
    result[:, :16] = old
    for slot, (episode, cursor) in enumerate(cursors):
        frames, actions = episodes[episode]
        previous = frames[cursor - 1]
        age = np.zeros((18, 20), dtype=np.uint16)
        same = np.ones((18, 20), dtype=bool)
        for t in range(cursor - 2, max(-1, cursor - 257), -1):
            same &= frames[t] == previous
            age += same
        buttons = actions[max(0, cursor - 255):cursor]
        changed = np.flatnonzero(buttons != buttons[-1])
        held = len(buttons) - int(changed[-1]) - 1 if len(changed) else len(buttons)
        result[slot, 16] = torch.from_numpy(previous.copy()).to(old.dtype)
        result[slot, 17] = torch.from_numpy(age).to(old.dtype)
        result[slot, 18].fill_(held)
        result[slot, 19].fill_(int(actions[cursor - 1]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/events-last-checkpoint.pt")
    parser.add_argument("--seconds", type=int, default=2400)
    parser.add_argument("--updates", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--observed-history", action="store_true")
    args = parser.parse_args()
    if min(args.seconds, args.updates, args.batch, args.threads) < 1 or args.output.exists():
        parser.error("require positive bounds and a new output directory")
    torch.set_num_threads(args.threads)
    torch.manual_seed(1042028)
    rng = random.Random(1042028)
    raw, digest, manifest = load_data(ROOT / "data")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["config"]["manifest_sha256"] != digest:
        raise ValueError("checkpoint/dataset mismatch")
    history_upgrade = args.observed_history and "_observed_history" not in checkpoint["model"]
    weights = upgrade_history_weights(checkpoint["model"]) if history_upgrade else checkpoint["model"]
    model_type = ObservedHistoryDynamics if "_observed_history" in weights else CalibratedEventDynamics
    model = model_type(weights["codebook"])
    missing = model.load_state_dict(weights, strict=False)
    if missing.unexpected_keys or set(missing.missing_keys) - {"gate_logit_offset"}:
        raise ValueError("checkpoint architecture mismatch")
    episodes = {}
    cache_dtype = np.min_scalar_type(len(model.codebook) - 1)
    # The reserved test is not used here; release decoded pixels before building caches.
    raw["test"].clear()
    for split in ("train", "validation"):
        episodes[split] = []
        for frames, actions in raw[split]:
            codes, unknown = encode_frames(frames, model.codebook.numpy())
            if unknown:
                raise ValueError("unexpected unseen visual patterns in train/validation")
            episodes[split].append((codes.astype(cache_dtype, copy=False), actions))
        if split == "train":
            raw[split].clear()
    model.to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    optimizer_state = expand_history_optimizer(checkpoint["optimizer"], model) if history_upgrade else checkpoint["optimizer"]
    optimizer.load_state_dict(optimizer_state)
    for group in optimizer.param_groups:
        group["lr"] = 3e-4
    args.output.mkdir(parents=True)
    config = {"protocol": "stateful-contiguous-calibrated", "manifest_sha256": digest,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "device": args.device, "torch": str(torch.__version__), "batch": args.batch, "prefix": 32, "unroll": 32, "seconds": args.seconds,
              "gate_pos_weight": 16, "lr": 3e-4, "parameters": sum(p.numel() for p in model.parameters()),
              "cache_dtype": np.dtype(cache_dtype).name,
              "validation_every": 1000,
              "model_family": type(model).__name__, "state_channels": model.state_channels,
              "sources": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in ("train_stateful.py", "token_model.py")}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "process.json").write_text(json.dumps({"pid": os.getpid(), "parent_pid": os.getppid(), "started": time.time()}))
    streams = StreamWindows(episodes["train"], args.batch, rng)
    state = None
    first_step = checkpoint["step"]
    if args.updates <= first_step:
        raise ValueError("update target must exceed checkpoint step")
    generated_horizon = 1
    if checkpoint["config"].get("protocol") == "stateful-contiguous-calibrated":
        preserved = (observed_stream_state(checkpoint["stream_state"], checkpoint["stream_cursors"], episodes["train"])
                     if history_upgrade else checkpoint["stream_state"])
        if (checkpoint["config"]["batch"] != args.batch or len(checkpoint["stream_cursors"]) != args.batch
                or tuple(preserved.shape) != (args.batch, model.state_channels, 18, 20)):
            raise ValueError("resume batch must match the preserved stream state")
        if not torch.isfinite(preserved).all():
            raise ValueError("nonfinite preserved stream state")
        rng.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        streams.cursors = checkpoint["stream_cursors"]
        state = preserved.to(args.device)
        generated_horizon = checkpoint["horizon"]
    best = checkpoint_baseline(checkpoint)
    best_path = args.checkpoint if best < float("inf") else None
    losses = []
    started = time.monotonic()
    with (args.output / "metrics.jsonl").open("x", buffering=1) as log:
        for step in range(first_step + 1, args.updates + 1):
            frames, actions, valid, reset, prefix, prefix_actions = streams.next()
            frames, actions, valid = frames.to(args.device), actions.to(args.device), valid.to(args.device)
            if state is None:
                state = torch.zeros(args.batch, model.state_channels, 18, 20, device=args.device)
            else:
                state = state.detach()
            chosen = reset.nonzero(as_tuple=True)[0]
            if len(chosen):
                warm = None
                with torch.no_grad():
                    warm_frames, warm_actions = prefix[chosen].to(args.device), prefix_actions[chosen].to(args.device)
                    for t in range(32):
                        _, warm = model(warm_frames[:, t], warm_actions[:, t], warm)
                state[chosen.to(args.device)] = warm
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=args.device)
            generated = None
            for t in range(32):
                current = frames[:, t] if generated is None or t % generated_horizon == 0 else generated
                write, gate, state = model.forward_details(current, actions[:, t], state)
                total = total + calibrated_loss(write, gate, frames[:, t + 1], current, model.codebook, valid[:, t])
                combined = model.combine_logits(current, write, gate - model.gate_logit_offset)
                generated = combined.detach().argmax(1)
            loss = total / 32
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            expired = time.monotonic() - started >= args.seconds
            if step % 1000 == 0 or expired or step == args.updates:
                validation_model = type(model)(model.codebook.cpu())
                validation_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
                teacher = teacher_metrics(validation_model, episodes["validation"])
                metrics, _ = measure(validation_model, episodes["validation"], raw["validation"])
                if teacher["addition_accuracy"] > 0.95 and teacher["erasure_accuracy"] > 0.95 and teacher["gate_precision"] > 0.95:
                    generated_horizon = min(8, generated_horizon * 2)
                score = selection_score(teacher, metrics)
                record = {"step": step, "seconds": time.monotonic() - started, "loss": float(np.mean(losses)),
                          "horizon": generated_horizon, "teacher": teacher, "validation": metrics}
                log.write(json.dumps(record) + "\n"); print(json.dumps(record), flush=True); losses = []
                if score < best or expired or step == args.updates:
                    path = args.output / f"checkpoint-{step:06d}.pt"
                    torch.save({"model": validation_model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                                "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(), "config": config,
                                "validation": metrics, "teacher": teacher, "horizon": generated_horizon,
                                "stream_cursors": streams.cursors, "stream_state": state.detach().cpu(),
                                "dataset_context": manifest["context"]}, path)
                    if score < best:
                        best, best_path = score, path
                if expired:
                    break
    if best_path is None:
        raise RuntimeError("no validated checkpoint")
    selected = torch.load(best_path, map_location="cpu", weights_only=True)
    torch.save(selected["model"], args.output / "model.pt")
    evaluation_model = load_tile_model(best_path)
    result = {"status": "trained-gameplay-unverified", "best_checkpoint": best_path.name,
              "best_from_resume": best_path == args.checkpoint,
              "export_model_family": type(evaluation_model).__name__,
              "teacher": selected["teacher"], "validation": selected["validation"], "test_used": False,
              "seconds": time.monotonic() - started, "config": config}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    _, examples = measure(evaluation_model, episodes["validation"], raw["validation"])
    for i, (predicted, reference) in enumerate(examples):
        comparison_image(predicted, reference, args.output / f"comparison-{i:02d}.png")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
