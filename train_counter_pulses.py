"""Fit clear magnitudes and event timing on frame-verified native counters."""
import argparse
import json
from pathlib import Path

import torch

from collect_counters import read_digit_captures
from counter_model import row_strength
from scene_model import CounterSceneDynamics, scene_from_weights
from train_counters import MANIFEST, values
from train_rewrite_evidence import load_events
from train_row_transport import verified_background
from train_spawn_preview import observed_captures, sha


def observations(model, checkpoint, digits_path, events_path, split):
    if sha(MANIFEST) != checkpoint["config"]["manifest_sha256"]:
        raise ValueError("counter manifest differs from the checkpoint")
    digits = read_digit_captures(digits_path, MANIFEST, split)
    events = load_events(events_path, checkpoint, split)
    vocabulary = model.counter.addends @ (10 ** torch.arange(4, -1, -1))
    delays = torch.zeros((len(vocabulary), 2), dtype=torch.long)
    clear_examples = []
    unmatched = 0
    for name, tokens, _, labels in observed_captures(checkpoint["config"], model.codebook.numpy(), split):
        numeric = values(torch.from_numpy(digits.pop(name)).long())
        known = (numeric >= 0).all(1)
        delta = numeric.diff(dim=0)[:, :2]
        valid = known[:-1] & known[1:]
        if len(numeric) != len(tokens):
            raise ValueError("native counter observations are not aligned")
        board = tokens[:, :, 2:12]
        occupied = torch.isin(board, model.object_tiles)
        retained = ((board[:-1] == board[1:]) | ~occupied[:-1]).flatten(1).all(1)
        growth = occupied.sum((1, 2)).diff()
        composed = (growth == 4) & retained
        deposits = (valid & (delta == vocabulary[1]).all(1)).nonzero().flatten().tolist()
        for t in deposits:
            candidates = [delay for delay in (0, 1) if t >= delay and composed[t-delay]]
            if len(candidates) == 1:
                delays[1, candidates[0]] += 1
            else:
                unmatched += 1
        label_arrays = {key: value.numpy() for key, value in labels.items()}
        for event in events.pop(name, []):
            first, _ = verified_background(event, board.numpy(), label_arrays, model)
            witnesses = []
            for delay in (0, 1):
                index = first + delay
                if valid[index] and delta[index, 1] == event["row_count"]:
                    matches = (vocabulary == delta[index]).all(1).nonzero().flatten().tolist()
                    witnesses.extend((code, delay) for code in matches if code >= 2)
            if len(witnesses) != 1:
                raise ValueError("clear proof lacks a unique native counter transition")
            code, delay = witnesses[0]
            strength = row_strength(board[event["first_lock_frame"]:event["first_lock_frame"]+1],
                                    model.object_tiles)[0, 0]
            clear_examples.append((strength, code))
            delays[code, delay] += 1
    if digits or events or not clear_examples or not delays[1:].sum(1).all():
        raise ValueError("incomplete counter pulse evidence")
    strength, code = zip(*clear_examples)
    return torch.stack(strength), torch.tensor(code), delays, unmatched, vocabulary


def measure(model, observations):
    strength, code, delays, unmatched, vocabulary = observations
    magnitude = model.counter.events(strength[:, None]).argmax(1) + 2
    chosen = (model.counter.delay.weight[:, 0] > 0).long()
    correct_delays = delays[torch.arange(len(delays)), chosen]
    return {"events": vocabulary.tolist(), "clear_magnitudes_exact": int((magnitude == code).sum()),
            "clears": len(code), "deposit_delta_without_composition": unmatched,
            "delay_witnesses": delays.tolist(), "delay_chosen": chosen.tolist(),
            "delays_exact": int(correct_delays.sum()), "delays_total": int(delays.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--digits", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    if not isinstance(base, CounterSceneDynamics) or base.counter.pulsed:
        raise ValueError("counter pulse training requires a fitted digit checkpoint")
    source = sha(args.checkpoint)
    split = "validation" if args.candidate else "train"
    data = observations(base, checkpoint, args.digits, args.events, split)
    provenance = {"source_sha256": source, "digits_sha256": sha(args.digits),
                  "events_sha256": sha(args.events), "test_used": False}
    if args.candidate:
        checkpoint = torch.load(args.candidate, map_location="cpu", weights_only=True)
        if checkpoint["counter_pulse_fit"]["source_sha256"] != source:
            raise ValueError("counter pulse candidate has a different source")
        model = scene_from_weights(checkpoint).eval()
        report = {**provenance, "split": split, "fit_used": False, "candidate_sha256": sha(args.candidate),
                  **measure(model, data)}
        with args.output.open("x") as out:
            json.dump(report, out, indent=2)
        return

    model = CounterSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns,
                                 int(base.animation_flash_id), int(base.terminal_fill_id), base.terminal_panels.long(),
                                 base.counter.glyphs, base.counter.legacy_ids, base.counter.addends, pulsed=True)
    frozen = {name: tensor for name, tensor in checkpoint["model"].items() if not name.startswith("counter.events.")}
    model.load_state_dict({**model.state_dict(), **frozen})
    strength, code, delays, _, _ = data
    means = torch.stack([strength[code == k].mean() for k in range(2, len(delays))])
    if not torch.isfinite(means).all():
        raise ValueError("missing clear magnitude prototype")
    with torch.no_grad():
        # Nearest learned class centroid written as affine logits.
        model.counter.events.weight.copy_(2 * means[:, None])
        model.counter.events.bias.copy_(-means.square())
        probability = (delays[:, 1].float() + .5) / (delays.sum(1) + 1)
        model.counter.delay.weight.copy_(probability.logit()[:, None])
    state = model.state_dict()
    for name, value in frozen.items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)
    report = {**provenance, "fit_split": split, "validation_used": False, "clear_centroids": means.tolist(),
              "parameters": sum(p.numel() for p in model.counter.parameters()), **measure(model, data)}
    if report["clear_magnitudes_exact"] != report["clears"]:
        raise ValueError("learned clear magnitudes overlap on proved locks")
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "counter_pulse_fit": report}, destination)
    report["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
