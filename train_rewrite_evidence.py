"""Calibrate rewrite scores on witnessed line endings and false train rewrites."""
import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from audit_clear_events import MAX_DELAY
from audit_controls import supported
from audit_long import recorded_path
from model import ACTIONS
from scene_model import (ClearAnimationSceneDynamics, ClearRoutingSceneDynamics, FeedbackCarrySceneDynamics,
                         rewrite_evidence_scene_weights, scene_from_weights)
from train_mode_correction import proven_composition
from train_row_transport import verified_background
from train_spawn_preview import observed_captures, sha


def load_events(path, checkpoint, split):
    evidence = json.loads(path.read_bytes())
    if (evidence["split"] != split or evidence["test_used"] or evidence["fit_used"]
            or evidence["validation_used"] != (split == "validation")
            or evidence["manifest_sha256"] != checkpoint["config"]["manifest_sha256"]
            or evidence["annotations_sha256"] != checkpoint["config"]["annotations_sha256"]):
        raise ValueError("clear evidence lineage or split mismatch")
    grouped = defaultdict(list)
    for event in evidence["events"]:
        grouped[event["file"]].append(event)
    return grouped


def false_rewrite(model, board, old, details, history):
    if old is None or len(old) != 4 or not details["contact_visible"].item() or len(history) != MAX_DELAY:
        return False
    current = int(details["current"]["joint"].argmax())
    shapes = model.shape_masks.flatten(0, 1).bool()
    active = shapes[current]
    if not active[old[:, 0], old[:, 1]].all():
        return False
    recent = torch.stack(tuple(history))
    styles = torch.cat((model.object_tiles, torch.tensor([model.blank_id])))
    if not torch.isin(recent, styles).all() or (recent != model.blank_id).all(2).any():
        return False
    background = board.masked_fill(active, model.blank_id)
    occupied = background != model.blank_id
    if occupied[:4].any() or supported(background.numpy(), old, model.blank_id):
        return False
    style = model.object_tiles[model.event_poses // 180] == model.object_tiles[current // 180]
    eligible = style & model.valid_poses.flatten() & ~(shapes & occupied).flatten(1).any(1)
    can_fill = ((occupied.sum(1) + shapes.sum(2)) == 10).any(1)
    return not bool((eligible & can_fill).any())


@torch.no_grad()
def collect(base, model, captures, events):
    features, margins, groups = [], [], []
    verified_count = 0

    def record(details, group):
        features.append(model.rewrite_features(details)[0])
        score = details["mode"][0]
        margins.append(score[2] - score[:2].max())
        groups.append(group)

    for name, frames, actions, labels in captures:
        boards = frames[:, :, 2:12].numpy()
        labels_np = {key: values.numpy() for key, values in labels.items()}
        verified = {verified_background(event, boards, labels_np, base)[0] for event in events.get(name, ())}
        verified_count += len(verified)
        modes, known, old, _, _ = proven_composition(base, frames, labels)
        selected = known & ((modes == 2) | (torch.arange(len(actions)) % 8 == 0))
        selected[list(verified)] = True
        state = None
        for t, command in enumerate(actions):
            frame, action = frames[t:t+1], command[None]
            if selected[t]:
                details = base.scene_details(frame, action, state)
                state = details["state"]
                record(details, 2 if t in verified else 1 if modes[t] == 2 else 0)
            else:
                warm = t + 1 < len(actions) and selected[t + 1]
                state = base.advance_observed(frame, action, state) if warm else base.advance(frame, action, state)
        anchors = (old[48:-255].nonzero().flatten() + 48).tolist()
        if len(anchors) < 3:
            continue
        for branch, quantile in enumerate((1, 2, 3)):
            index = anchors[quantile * len(anchors) // 4]
            state = None
            for t in range(index - 48, index):
                state = base.advance_observed(frames[t:t+1], actions[t:t+1], state)
            history = deque(frames[index-MAX_DELAY:index, :, 2:12], maxlen=MAX_DELAY)

            def observe(tokens, old_cells, details, event):
                board = tokens[0, :, 2:12]
                history.append(board)
                if event["status"] in ("motion", "deposit"):
                    return True
                if (event["mode"] == 2 and ACTIONS[event["button"]] not in ("start", "up")
                        and false_rewrite(base, board, old_cells, details, history)):
                    record(details, 3)
                return False

            commands = actions[index:index+256] if branch == 0 else [ACTIONS.index("down")] * 256
            recorded_path(base, frames[index:index+1], state, labels_np["cells"][index], commands,
                          ACTIONS[int(actions[index - 1])], observer=observe)
    if verified_count != sum(map(len, events.values())) or not {0, 1, 2, 3} <= set(groups):
        raise ValueError(f"incomplete positive or negative proof: {verified_count} clears, {sorted(set(groups))} groups")
    return torch.stack(features), torch.stack(margins), torch.tensor(groups)


def counts_for(margin, groups):
    return {name: {"examples": int((groups == code).sum()),
                   "rewrite_selected": int(((groups == code) & (margin > 0)).sum())}
            for code, name in enumerate(("proved_continuity", "proved_rewrite", "clear_end", "generated_false_rewrite"))}


@torch.inference_mode()
def clear_rollout(model, current, state, actions, target, reference):
    background_matches, screen_matches = [], []
    lock = current[0, :, 2:12]
    full = (lock != model.blank_id).all(1)
    trace = []
    for command, expected in zip(actions, reference, strict=True):
        details = model.scene_details(current, command[None], state)
        predicted = model.decode_scene(current, details, hard=True)
        background = model._hard_background(current, details, details["gate"] - model.gate_logit_offset)
        background_matches.append(torch.equal(background[0, :, 2:12], target))
        screen_matches.append(torch.equal(predicted[0, :, 2:12], expected[:, 2:12]))
        changed = (predicted[0, :, 2:12] != lock).any(1)
        pointer = model.transported_rows(current, details)[0]
        animation = details.get("animation")
        age = max(0, int(state[0, -1, 0, 0])) if animation is not None and state is not None else 0
        source = state[0, -1, :, 2:12].long() if age else current[0, :, 2:12]
        trace.append({"mode": int(details["mode"].argmax(1)), "button": int(command),
                      "uses_transport": bool(details.get("render_transport_gate", details["row_transport_gate"])[0] > 0),
                      "pointer_exact_rows": int((pointer == target).all(1).sum()),
                      "full_rows": int((predicted[0, :, 2:12] != model.blank_id).all(1).sum()),
                      "changed_from_lock": int(changed.sum()),
                      "flash_like": bool(changed.any() and not (changed & ~full).any()),
                      "animation_active": animation is not None,
                      "animation_ending": bool(animation is not None and animation["ending"][0]),
                      "animation_age": age if animation is not None else None,
                      "animation_source_rows": int(torch.isin(source, model.object_tiles).all(1).sum())
                      if animation is not None else None,
                      "visible_next": bool(details["next"]["visible"][0, 0] > 0),
                      "reference_exact": screen_matches[-1], "background_exact": background_matches[-1]})
        state = model.rendered_state(details, predicted)
        current = predicted
    return {"any_exact_background": any(background_matches),
            "first_board_exact": screen_matches[0],
            "confirmed_background": all(background_matches[-2:]),
            "confirmed_screen": all(screen_matches[-2:])}, trace


@torch.inference_mode()
def evaluate(base, candidate, captures, events):
    summary = {"events": 0, "changed_outcomes": []}
    for tag in ("before", "after"):
        summary[tag] = {"rewrite": 0, "exact_background": 0, "renderable_background": 0,
                        "exact_by_mode": [0, 0, 0], "exact_row_pointer": 0}
    separate_history = isinstance(candidate, FeedbackCarrySceneDynamics)
    base_feedback = isinstance(base, FeedbackCarrySceneDynamics)
    replay_clears = isinstance(candidate, ClearRoutingSceneDynamics)
    first_lock = isinstance(candidate, ClearAnimationSceneDynamics)
    summary["history"] = "separate_observed_prefixes" if separate_history else "shared_baseline_prefix"
    if replay_clears:
        summary["autonomous"] = {tag: {"events": 0, "any_exact_background": 0, "first_board_exact": 0, "confirmed_background": 0,
                                       "confirmed_screen": 0} for tag in ("before", "after")}
        summary["autonomous_traces"] = []
        summary["autonomous_start"] = "first_lock_frame" if first_lock else "last_consecutive_lock_frame"
    for name, frames, actions, labels in captures:
        if name not in events:
            continue
        boards = frames[:, :, 2:12].numpy()
        label_values = {key: values.numpy() for key, values in labels.items()}
        targets = dict(verified_background(event, boards, label_values, base) for event in events[name])
        starts = {event[summary["autonomous_start"]]: event for event in events[name]} if replay_clears else {}
        state = candidate_state = None
        for t in range(max(targets) + 1):
            frame, action = frames[t:t+1], actions[t:t+1]
            if t in starts:
                event = starts[t]
                end = event["confirmation_frame"]
                target = torch.from_numpy(targets[event["first_outcome_frame"] - 1])
                traces = {"file": name, "lock_frame": t, "expected_clear_frame": event["first_outcome_frame"],
                          "expected_flash_frames": event["flash_frames"]}
                for tag, model, initial in (("before", base, state), ("after", candidate, candidate_state)):
                    metrics, traces[tag] = clear_rollout(model, frame, initial, actions[t:end], target, frames[t+1:end+1])
                    counters = summary["autonomous"][tag]
                    counters["events"] += 1
                    for key, value in metrics.items():
                        counters[key] += int(value)
                summary["autonomous_traces"].append(traces)
            details = base.scene_details(frame, action, state)
            if t in targets:
                summary["events"] += 1
                prior = candidate_state if separate_history else state
                selected = {}
                target = torch.from_numpy(targets[t])
                for tag, model, proposal in (("before", base, details), ("after", candidate, candidate.scene_details(frame, action, prior))):
                    copy, rewrite = model._hard_rewrite_candidates(frame, proposal, proposal["gate"] - model.gate_logit_offset)
                    rewrite = model._select_rewritten_rows(proposal, copy, rewrite, hard=True)
                    mode = int(proposal["mode"].argmax(1))
                    matches = [torch.equal(board[0, :, 2:12], target) for board in (copy, frame, rewrite)]
                    animation = proposal.get("animation")
                    if animation is not None:
                        rendered = model._hard_background(frame, proposal, proposal["gate"] - model.gate_logit_offset)
                        matches[2] = torch.equal(rendered[0, :, 2:12], target)
                    pointer = model.transported_rows(frame, proposal)[0]
                    counters = summary[tag]
                    counters["rewrite"] += int(mode == 2)
                    counters["exact_background"] += int(matches[mode])
                    counters["renderable_background"] += int(any(matches))
                    counters["exact_by_mode"] = [count + int(match) for count, match in zip(counters["exact_by_mode"], matches)]
                    counters["exact_row_pointer"] += int(torch.equal(pointer, target))
                    selected[tag] = {"mode": mode, "exact": matches[mode],
                                     "matching_modes": [i for i, match in enumerate(matches) if match]}
                if selected["before"]["exact"] != selected["after"]["exact"]:
                    summary["changed_outcomes"].append({"file": name, "frame": t, **selected})
            if base_feedback:
                state = base.rendered_state(details, base.decode_scene(frame, details, hard=True))
            else:
                state = details["state"]
            if separate_history:
                candidate_state = candidate.advance_observed(frame, action, candidate_state)
    if summary["events"] != sum(map(len, events.values())):
        raise ValueError("validation clear coverage mismatch")
    if replay_clears and any(result["events"] != summary["events"] for result in summary["autonomous"].values()):
        raise ValueError("validation autonomous clear coverage mismatch")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", type=Path, help="evaluate an already fitted checkpoint on held-out clears")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(68)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = scene_from_weights(checkpoint["model"]).eval()
    split = "validation" if args.candidate else "train"
    events = load_events(args.events, checkpoint, split)
    captures = observed_captures(checkpoint["config"], base.codebook.numpy(), split)
    if args.candidate:
        fitted = torch.load(args.candidate, map_location="cpu", weights_only=True)
        fit = (fitted.get("compose_contact_fit") or fitted.get("clear_onset_fit") or fitted.get("terminal_contact_fit")
               or fitted.get("terminal_fit") or fitted.get("clear_animation_fit")
               or fitted.get("clear_routing_fit") or fitted.get("event_replay_fit")
               or fitted.get("feedback_carry_fit") or fitted["rewrite_evidence_fit"])
        if fit["source_sha256"] != sha(args.checkpoint):
            raise ValueError("candidate uses a different baseline")
        candidate = scene_from_weights(fitted["model"]).eval()
        report = {"validation_used": True, "fit_used": False, "test_used": False,
                  "source_sha256": sha(args.checkpoint), "candidate_sha256": sha(args.candidate),
                  **evaluate(base, candidate, captures, events)}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    model = rewrite_evidence_scene_weights(checkpoint["model"]).eval()
    x, margin, groups = collect(base, model, captures, events)
    y = (groups == 1) | (groups == 2)
    preserve = y & (margin > 0)
    weights = 1 / torch.bincount(groups)[groups].float()
    head = model.rewrite_evidence
    optimizer = torch.optim.AdamW(head.parameters(), lr=.025)
    for _ in range(240):
        logit = head(x)[:, 0]
        revised = margin + F.logsigmoid(logit)
        loss = F.binary_cross_entropy_with_logits(revised, y.float(), weight=weights, reduction="sum")
        loss += .1 * F.binary_cross_entropy_with_logits(logit, y.float(), weight=weights, reduction="sum")
        if preserve.any():
            loss += F.relu(.05 - revised[preserve]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    state = model.state_dict()
    for key, value in checkpoint["model"].items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    with torch.no_grad():
        after = counts_for(margin + F.logsigmoid(head(x)[:, 0]), groups)
    result = {"fit_split": "train", "validation_used": False, "test_used": False,
              "source_sha256": sha(args.checkpoint), "events_sha256": sha(args.events),
              "before": counts_for(margin, groups), "after": after}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "model.pt"
    torch.save({"model": state, "config": checkpoint["config"], "rewrite_evidence_fit": result}, destination)
    result["checkpoint_sha256"] = sha(destination)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
