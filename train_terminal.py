"""Fit terminal readiness and the visible wipe from witnessed ROM episodes."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scene_model import scene_from_weights, terminal_scene_weights
from token_model import encode_frames
from train_spawn_preview import observed_captures, sha


def terminal_witness(model, tokens):
    panel = len(tokens) - 8
    if panel < 12 or not torch.equal(tokens[panel], tokens[panel+1]):
        return None
    if (tokens[panel, :, 2:12] != model.blank_id).any():
        return None
    board = tokens[:, :, 2:12]
    tile, frequency = torch.unique(board[panel-1], return_counts=True)
    fill = tile[frequency.argmax()]
    if frequency.max() < 144 or fill == model.blank_id or (fill == model.object_tiles).any():
        return None
    wiped = (board == fill).all(2)
    first = panel
    while first > 0 and wiped[first - 1].any():
        first -= 1
    if first == panel or first == 0 or (wiped[first:panel-1] & ~wiped[first+1:panel]).any():
        return None
    start = first - 1
    while start > 0 and torch.equal(tokens[start - 1], tokens[first - 1]):
        start -= 1
    if (board[start:panel] != torch.where(wiped[start:panel, :, None], fill, board[start])).any():
        return None
    return start, first, panel, int(fill), wiped


def observed_ages(board):
    previous = torch.cat((board[:1], board[:-1]))
    index = torch.arange(len(board))[:, None, None]
    changed = torch.where(board != previous, index, 0).cummax(0).values
    return (index - changed).clamp_max(255)


def read_tails(path, model, manifest, manifest_sha256, split):
    with np.load(path, allow_pickle=False) as archive:
        frames = archive["frames"]
        metadata = json.loads(str(archive["metadata"]))
    if (metadata["split"] != split or metadata["test_used"] or metadata["fit_used"]
            or metadata["manifest_sha256"] != manifest_sha256
            or metadata["rom_sha256"] != manifest["context"]["rom"]["sha256"]
            or frames.dtype != np.uint8 or frames.shape[1:] != (8 + metadata["tail_noop_steps"], 72, 80)
            or len(frames) != len(metadata["entries"]) or np.any(frames > 3)):
        raise ValueError("incompatible terminal tails")
    expected = {row["file"]: row["sha256"] for row in manifest["episodes"] if row["split"] == split}
    names = [entry["file"] for entry in metadata["entries"]]
    if len(set(names)) != len(names) or any(expected.get(entry["file"]) != entry["sha256"]
                                          for entry in metadata["entries"]):
        raise ValueError("terminal tail source digest mismatch")
    codes, missing = encode_frames(frames.reshape(-1, 72, 80), model.codebook.numpy())
    if missing:
        raise ValueError("terminal tails contain tiles outside the frozen vocabulary")
    return dict(zip(names, torch.from_numpy(codes.reshape(*frames.shape[:2], 18, 20)).long(), strict=True))


def aligned_tail(frames, actions, tail, panel):
    if not torch.equal(tail[:8], frames[panel:]):
        raise ValueError("terminal tail differs from the observed capture")
    buttons = F.pad(actions[panel-1:], (0, len(tail) - 8))
    return torch.cat((frames[panel-1:panel], tail[:-1])), buttons


@torch.no_grad()
def collect(model, captures, witnesses):
    features, wanted, starts, sequences = [], [], [], []
    for name, frames, _, _ in captures:
        count = len(frames) - 1
        positive = torch.zeros(count, dtype=torch.bool)
        first = torch.zeros(count, dtype=torch.bool)
        if name in witnesses:
            start, wipe, panel, _, masks = witnesses[name]
            count = wipe
            first[start] = True
            positive[start:wipe] = True
            age = torch.arange(panel - start - 1)
            sequences.append((frames[start:start+1].expand(len(age), -1, -1), age,
                              masks[start+1:panel], torch.full_like(age, panel - start - 1)))
        board = frames[:count, :, 2:12]
        features.append(model.terminal_features(board, observed_ages(board)))
        wanted.append(positive[:count])
        starts.append(first[:count])
    return (torch.cat(features), torch.cat(wanted), torch.cat(starts)), tuple(torch.cat(col) for col in zip(*sequences))


@torch.no_grad()
def measure(model, onset, phases):
    features, positive, starts = onset
    selected = model.terminal_onset(features)[:, 0] > 0
    source, age, expected, length = phases
    rows, duration = model.terminal_phase(source, age)
    correct = ((rows > 0) == expected).all(1)
    return {"first_onsets": int((selected & starts).sum()), "witnesses": int(starts.sum()),
            "onset_contexts": int((selected & positive).sum()), "proved_onset_contexts": int(positive.sum()),
            "false_onsets": int((selected & ~positive).sum()), "negative_contexts": int((~positive).sum()),
            "wipe_frames_exact": int(correct.sum()), "wipe_frames": len(correct),
            "duration_mae_steps": float((duration[age == 0] - length[age == 0]).abs().mean())}


def fit(checkpoint, path, model, captures, terminal_files, tails):
    witnesses = {}
    rejected = []
    for name, frames, _, _ in captures:
        if name not in terminal_files:
            continue
        witness = terminal_witness(model, frames)
        if witness is None:
            rejected.append(name)
        else:
            witnesses[name] = witness
    if not witnesses or set(witnesses) != set(tails):
        raise ValueError("proved train terminal screens differ from the observed tails")
    fill_ids = Counter(record[3] for record in witnesses.values())
    if len(fill_ids) != 1:
        raise ValueError("incompatible witnessed wipe tiles")
    panels = torch.unique(torch.cat(list(tails.values())).flatten(1), dim=0).reshape(-1, 18, 20)
    candidate = terminal_scene_weights(checkpoint["model"], next(iter(fill_ids)), panels).eval()
    menus = []
    for name, frames, actions, _ in captures:
        if name not in witnesses:
            continue
        tail = tails[name]
        previous, buttons = aligned_tail(frames, actions, tail, witnesses[name][2])
        exact = (tail[:, None] == panels[None]).flatten(2).all(2)
        menus.append((previous, buttons, torch.arange(len(tail)), exact.long().argmax(1)))
    menu_frames, menu_buttons, menu_age, menu_target = (torch.cat(column) for column in zip(*menus))
    onset, phases = collect(candidate, captures, witnesses)
    before = measure(candidate, onset, phases)
    for name, parameter in candidate.named_parameters():
        parameter.requires_grad_(name.startswith("terminal_"))
    features, positive, first = onset
    negative_weight = 1 + 40 * features[~positive, :4].amax(1) * features[~positive, 13]
    negative_weight = negative_weight / negative_weight.sum()
    optimizer = torch.optim.LBFGS(candidate.terminal_onset.parameters(), max_iter=180, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = candidate.terminal_onset(features)[:, 0]
        loss = .5 * (F.softplus(-logits[first]).mean() + F.softplus(-logits[positive]).mean())
        loss += 3 * (F.softplus(logits[~positive]) * negative_weight).sum()
        loss += .0003 * sum(p.square().sum() for p in candidate.terminal_onset.parameters())
        loss.backward()
        return loss

    optimizer.step(closure)
    source, age, expected, duration = phases
    onset_rows = age == 0
    with torch.no_grad():
        candidate.terminal_duration.weight.zero_()
        candidate.terminal_duration.bias.fill_((duration[onset_rows].float().mean() / 40).logit().item())
    phase_parameters = [*candidate.terminal_duration.parameters(), *candidate.terminal_wipe.parameters()]
    optimizer = torch.optim.AdamW(phase_parameters, lr=.015)
    for _ in range(400):
        rows, predicted = candidate.terminal_phase(source, age)
        loss = F.binary_cross_entropy_with_logits(rows, expected.float())
        loss += .03 * F.smooth_l1_loss(predicted[onset_rows], duration[onset_rows].float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(phase_parameters, 1., error_if_nonfinite=True)
        optimizer.step()
    optimizer = torch.optim.AdamW(candidate.terminal_menu.parameters(), lr=.015)
    weight = torch.where(menu_age < 8, 6., 1.)
    for _ in range(360):
        logits = candidate.terminal_menu_scores(menu_frames, menu_buttons, menu_age)
        loss = (F.cross_entropy(logits, menu_target, reduction="none") * weight).sum() / weight.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        selected = candidate.terminal_menu_scores(menu_frames, menu_buttons, menu_age).argmax(1)
    state = candidate.state_dict()
    for key, value in checkpoint["model"].items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    result = {"fit_split": "train", "validation_used": False, "test_used": False, "source_sha256": sha(path),
              "witnesses": {name: record[:4] for name, record in witnesses.items()}, "excluded_terminal_flags": rejected,
              "trainable_parameters": sum(p.numel() for p in candidate.parameters() if p.requires_grad),
              "terminal_motifs": len(panels), "terminal_frames_exact": int((selected == menu_target).sum()),
              "terminal_frames": len(selected),
              "before": before, "after": measure(candidate, onset, phases)}
    return state, result


@torch.inference_mode()
def audit(model, frames, actions, witness, tail):
    start, wipe, panel, _, _ = witness
    aligned_tail(frames, actions, tail, panel)
    frames = torch.cat((frames[:panel], tail))
    actions = F.pad(actions, (0, len(tail) - 8))
    state = None
    early = None
    for t in range(start):
        state = model.advance_observed(frames[t:t+1], actions[t:t+1], state)
        if state[0, -1, 0, 0] < 0 and early is None:
            early = t + 1
    current = frames[start:start+1]
    first_panel = None
    first_menu = None
    first_change = None
    exact = 0
    for t in range(start, len(actions)):
        current, state = model.predict_tokens(current, actions[t:t+1], state)
        exact += int(torch.equal(current[0], frames[t+1]))
        if first_change is None and not torch.equal(current[:, :, 2:12], frames[start:start+1, :, 2:12]):
            first_change = t + 1
        if first_panel is None and torch.equal(current[0], frames[panel]):
            first_panel = t + 1
        if first_menu is None and torch.equal(current[0], frames[panel+2]):
            first_menu = t + 1
    return {"exact_frames": exact, "frames": len(actions) - start, "early_latch_frame": early,
            "first_board_change": first_change, "first_exact_panel": first_panel,
            "first_exact_menu": first_menu, "final_menu_exact": torch.equal(current[0], frames[-1]),
            "reference_start": start, "reference_wipe": wipe, "reference_panel": panel}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--tails", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(74)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = scene_from_weights(checkpoint).eval()
    split = "validation" if args.candidate else "train"
    manifest = json.loads(Path("data/manifest.json").read_bytes())
    tails = read_tails(args.tails, model, manifest, checkpoint["config"]["manifest_sha256"], split)
    terminal_files = {row["file"] for row in manifest["episodes"] if row["split"] == split
                      and row["events"]["terminal_observed"]}
    captures = observed_captures(checkpoint["config"], model.codebook.numpy(), split)
    if args.candidate:
        weights = torch.load(args.candidate, map_location="cpu", weights_only=True)
        fit_result = (weights.get("compose_contact_fit") or weights.get("clear_onset_fit")
                      or weights.get("terminal_contact_fit") or weights["terminal_fit"])
        if fit_result["source_sha256"] != sha(args.checkpoint):
            raise ValueError("candidate terminal source mismatch")
        candidate = scene_from_weights(weights).eval()
        results = {}
        for name, frames, actions, _ in captures:
            if name in terminal_files:
                witness = terminal_witness(model, frames)
                if witness is None:
                    raise ValueError(f"unverified validation terminal flag: {name}")
                results[name] = {"before": audit(model, frames, actions, witness, tails[name]),
                                 "after": audit(candidate, frames, actions, witness, tails[name])}
        if set(results) != terminal_files or not results:
            raise ValueError("incomplete terminal validation")
        report = {"split": split, "test_used": False, "fit_used": False, "source_sha256": sha(args.checkpoint),
                  "candidate_sha256": sha(args.candidate), "tails_sha256": sha(args.tails), "results": results}
        with args.output.open("x") as out:
            json.dump(report, out, indent=2)
        return
    state, result = fit(checkpoint, args.checkpoint, model, list(captures), terminal_files, tails)
    result["tails_sha256"] = sha(args.tails)
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "terminal_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
