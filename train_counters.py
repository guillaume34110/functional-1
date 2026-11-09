"""Train the event and decimal readouts from verified visible counter changes."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from collect_counters import GLYPHS, read_digit_captures
from counter_model import FEATURES
from scene_model import ComposeContactSceneDynamics, CounterSceneDynamics, scene_from_weights
from train_event_replay import BURN, Window
from train_spawn_preview import observed_captures, sha


MANIFEST = Path("data/manifest.json")
SPAN = 8


def values(digits):
    fields = []
    for field in digits.split((5, 5, 2), dim=1):
        numeric = (field >= 0) & (field < 10)
        known = (field >= 0).all(1) & numeric[:, -1]
        known &= ~(numeric.to(torch.int8).cummax(1).values.bool() & ~numeric).any(1)
        decoded = field.masked_fill(~numeric, 0) @ (10 ** torch.arange(field.size(1) - 1, -1, -1))
        fields.append(decoded.masked_fill(~known, -1))
    return torch.stack(fields, dim=1)


def recorded(checkpoint, model, sidecar, split):
    if sha(MANIFEST) != checkpoint["config"]["manifest_sha256"]:
        raise ValueError("counter manifest differs from the checkpoint")
    digits = read_digit_captures(sidecar, MANIFEST, split)
    result = []
    for name, frames, actions, _ in observed_captures(checkpoint["config"], model.codebook.numpy(), split):
        readout = torch.from_numpy(digits.pop(name)).long()
        if len(readout) != len(frames):
            raise ValueError("counter labels are not aligned with captured frames")
        numeric = values(readout)
        valid = (numeric >= 0).all(1)
        delta = numeric[1:] - numeric[:-1]
        result.append((frames, actions, readout, delta, valid[1:] & valid[:-1]))
    if digits or not result:
        raise ValueError("incomplete counter dataset")
    return result


def digits_of(values):
    return values[..., None] // (10 ** torch.arange(4, -1, -1)) % 10


def with_counter(base, events):
    legacy = torch.from_numpy(GLYPHS[:, 1::2, 1::2].copy())
    matches = (legacy[:, None] == base.codebook).flatten(2).all(2)
    if not (matches.sum(1) == 1).all():
        raise ValueError("legacy HUD glyph is outside the frozen vocabulary")
    rendered = legacy.clone()
    rendered[8:10] = torch.from_numpy(GLYPHS[8:10, ::2, ::2].copy())
    if (rendered[8:10, None] == base.codebook).flatten(2).all(2).any():
        raise ValueError("new counter glyph collides with the legacy codec")
    model = CounterSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns,
                                 int(base.animation_flash_id), int(base.terminal_fill_id), base.terminal_panels.long(),
                                 rendered, matches.long().argmax(1), digits_of(events))
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model.eval()


def windows(records, event_values, validation):
    result = []
    for frames, actions, _, deltas, valid in records:
        if len(actions) < BURN + SPAN + 1:
            continue
        matches = (deltas[:, None, :2] == event_values).all(2)
        if not matches[valid].any(1).all():
            raise ValueError("native transition absent from the train event vocabulary")
        targets = matches.long().argmax(1).masked_fill(~valid, -1)
        early = BURN + 1
        ordinary = (targets[early:] == 0).nonzero().flatten() + early
        deposits = (targets[early:] == 1).nonzero().flatten() + early
        clears = (targets[early:] > 1).nonzero().flatten() + early
        if len(ordinary):
            ordinary = ordinary[len(ordinary) * torch.arange(1, 5) // 5]
        if len(deposits) > 12 and not validation:
            deposits = deposits[len(deposits) * torch.arange(1, 13) // 13]
        starts = {max(BURN, min(int(t) // SPAN * SPAN - 1, len(actions) - SPAN - 1))
                  for t in torch.cat((ordinary, deposits, clears))}
        for start in sorted(starts):
            end = start + SPAN + 1
            result.append(Window(frames[start-BURN:start], actions[start-BURN:start],
                                 frames[start:end+1], actions[start:end], targets[start:end]))
    return result


@torch.no_grad()
def cache(model, windows):
    examples = []
    for index in range(0, len(windows), 16):
        prefix, commands, frames, actions, labels = map(torch.stack, zip(*windows[index:index+16]))
        state = None
        for t in range(prefix.size(1)):
            state = model.advance_observed(prefix[:, t], commands[:, t], state)
        previous = torch.zeros((len(frames), FEATURES))
        for t in range(actions.size(1)):
            frame, action = frames[:, t], actions[:, t]
            details = model.scene_details(frame, action, state)
            prediction = model.decode_scene(frame, details, hard=True)
            state = model.rendered_state(details, prediction)
            current = model.counter_features(frame, action, details, prediction)
            use = labels[:, t] >= 0
            if t and use.any():
                examples.append((torch.cat((previous, current), dim=1)[use], labels[use, t]))
            previous = current
    if not examples:
        raise ValueError("no readable counter training windows")
    return tuple(torch.cat(column) for column in zip(*examples))


def fit_digits(head):
    source, addend, carry = torch.cartesian_prod(torch.arange(11), torch.arange(10), torch.arange(2)).T
    total = source.masked_fill(source == 10, 0) + addend + carry
    result = (total % 10).masked_fill((source == 10) & (total == 0), 10) + (total // 10) * 11
    optimizer = torch.optim.Adam(head.digit.parameters(), lr=.035)
    for iteration in range(1, 1501):
        logits = head.digit_logits(source, addend, carry)
        if iteration % 25 == 0 and torch.equal(logits.argmax(1), result):
            return len(source)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(logits, result).backward()
        optimizer.step()
    raise ValueError("digit cell did not learn all decimal transitions")


def fit_level(head, records):
    carry, up = [], []
    for _, _, digits, delta, valid in records:
        carry.append((digits[:-1, 9][valid] + delta[valid, 1] >= 10).float())
        up.append(delta[valid, 2].float())
    carry, up = torch.cat(carry), torch.cat(up)
    if not up.any() or not ((up == 0) | (up == 1)).all():
        raise ValueError("unqualified level transitions")
    weights = torch.where(up > 0, 1 / up.sum(), 1 / (1 - up).sum())
    optimizer = torch.optim.LBFGS(head.level.parameters(), max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = head.level(carry[:, None])[:, 0]
        loss = F.binary_cross_entropy_with_logits(logits, up, weight=weights, reduction="sum")
        loss += .001 * head.level.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {"up": int(up.sum()), "exact": int(((head.level(carry[:, None])[:, 0] > 0) == up.bool()).sum()),
            "examples": len(up)}


@torch.no_grad()
def measure(head, x, y):
    predicted = head.events(x).argmax(1)
    return [{"samples": int((y == index).sum()), "exact": int(((y == index) & (predicted == index)).sum()),
             "predicted": int((predicted == index).sum())} for index in range(len(head.addends))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--digits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(78)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint).eval()
    if type(base) is not ComposeContactSceneDynamics:
        raise ValueError("counter training requires the compose contact model")
    if args.candidate:
        candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
        if candidate["counter_fit"]["source_sha256"] != sha(args.checkpoint):
            raise ValueError("counter candidate uses a different frozen source")
        model = scene_from_weights(candidate).eval()
        events = model.counter.addends @ (10 ** torch.arange(4, -1, -1))
        data = recorded(checkpoint, base, args.digits, "validation")
        cohorts = windows(data, events, validation=True)
        x, y = cache(model, cohorts)
        with args.output.open("x") as out:
            json.dump({"fit_used": False, "test_used": False, "split": "validation", "windows": len(cohorts),
                       "candidate_sha256": sha(args.candidate), "sidecar_sha256": sha(args.digits),
                       "events": events.tolist(), "teacher_events": measure(model.counter, x, y)}, out, indent=2)
        return

    data = recorded(checkpoint, base, args.digits, "train")
    events = torch.cat([delta[valid, :2] for _, _, _, delta, valid in data]).unique(dim=0, sorted=True)
    if (events < 0).any() or (events > 99999).any():
        raise ValueError("counter increments outside the observed monotone vocabulary")
    model = with_counter(base, events)
    cohorts = windows(data, events, validation=False)
    x, y = cache(model, cohorts)
    counts = torch.bincount(y, minlength=len(events))
    if not counts.all():
        raise ValueError("training windows omit an observed counter event")
    head = model.counter
    weights = counts.float().reciprocal()
    weights[0] *= 6
    optimizer = torch.optim.AdamW(head.events.parameters(), lr=.015, weight_decay=.001)
    for _ in range(220):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(head.events(x), y, weight=weights).backward()
        optimizer.step()
    digit_cases = fit_digits(head)
    level = fit_level(head, data)
    state = model.state_dict()
    for name, value in checkpoint["model"].items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)
    report = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "sidecar_sha256": sha(args.digits), "windows": len(cohorts),
              "parameters": sum(p.numel() for p in head.parameters()), "digit_cases": digit_cases, "level": level,
              "events": events.tolist(), "train_events": measure(head, x, y)}
    args.output.mkdir(parents=True, exist_ok=False)
    dest = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "counter_fit": report}, dest)
    report["checkpoint_sha256"] = sha(dest)
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
